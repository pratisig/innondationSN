import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping
import json
import ee
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(page_title="FloodWatch WA - Dashboard Impact", layout="wide")
ox.settings.timeout = 180
ox.settings.use_cache = True

# ============================================================================
# GEE INIT
# ============================================================================
@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    creds = ee.ServiceAccountCredentials(
        key["client_email"], key_data=json.dumps(key)
    )
    ee.Initialize(creds)
    return True

gee_available = init_gee()

# ============================================================================
# FLOOD MASK – VERSION AMÉLIORÉE (ANTI-SURESTIMATION)
# ============================================================================
def get_flood_mask(aoi, start_flood, end_flood, threshold=1.3):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    # Référence saisonnière cohérente (année précédente)
    ref_start = f"{int(start_flood[:4])-1}{start_flood[4:]}"
    ref_end = f"{int(end_flood[:4])-1}{end_flood[4:]}"

    ref = s1.filterDate(ref_start, ref_end).median()
    flood = s1.filterDate(start_flood, end_flood).median()

    # Speckle filtering
    ref = ref.focal_median(30, 'circle', 'meters')
    flood = flood.focal_median(30, 'circle', 'meters')

    # Ratio log (méthode robuste)
    ratio = flood.subtract(ref)

    flood_mask = ratio.lt(-threshold)

    # Masque pente (SRTM)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    flood_mask = flood_mask.updateMask(slope.lt(5))

    # Nettoyage morphologique
    flood_mask = flood_mask.focal_min(20).focal_max(20)

    return flood_mask.selfMask()

# ============================================================================
# POPULATION
# ============================================================================
def population_stats(aoi, flood_mask):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").mosaic()
    total = pop.reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo() or 0

    exposed = pop.updateMask(flood_mask).reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo() or 0

    return int(total), int(exposed)

# ============================================================================
# CLIMAT – NASA POWER
# ============================================================================
def climate_timeseries(aoi, start, end):
    c = aoi.centroid().getInfo()["coordinates"]
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"latitude={c[1]}&longitude={c[0]}"
        f"&start={start.replace('-','')}&end={end.replace('-','')}"
        "&parameters=PRECTOTCORR,T2M"
        "&community=AG&format=JSON"
    )
    data = requests.get(url).json()["properties"]["parameter"]
    df = pd.DataFrame({
        "date": pd.to_datetime(list(data["T2M"].keys())),
        "temp": list(data["T2M"].values()),
        "rain": list(data["PRECTOTCORR"].values())
    })
    return df

# ============================================================================
# OSM
# ============================================================================
def get_osm_data(gdf):
    poly = gdf.unary_union
    graph = ox.graph_from_polygon(poly, network_type="all")
    routes = ox.graph_to_gdfs(graph, nodes=False, edges=True).clip(gdf)

    tags = {"building": True}
    buildings = ox.features_from_polygon(poly, tags=tags)
    buildings = buildings[buildings.geometry.type.isin(["Polygon","MultiPolygon"])]

    return buildings.reset_index(), routes.reset_index()

# ============================================================================
# IMPACT INFRA
# ============================================================================
def impacted_infra(flood_mask, gdf):
    feats = [
        ee.Feature(ee.Geometry(mapping(g)), {"id": i})
        for i, g in enumerate(gdf.geometry)
    ]
    fc = ee.FeatureCollection(feats)
    stats = flood_mask.reduceRegions(fc, ee.Reducer.mean(), 10)
    impacted_ids = [
        f["properties"]["id"]
        for f in stats.filter(ee.Filter.gt("mean",0)).getInfo()["features"]
    ]
    gdf["impacted"] = gdf.index.isin(impacted_ids)
    return gdf

# ============================================================================
# UI
# ============================================================================
st.title("🌊 FloodWatch WA – Dashboard Impact")

uploaded = st.sidebar.file_uploader("Importer une zone (GeoJSON/KML)", ["geojson","kml"])
start = st.sidebar.date_input("Début", datetime(2024,8,1))
end = st.sidebar.date_input("Fin", datetime(2024,9,30))

if uploaded:
    zone = gpd.read_file(uploaded).to_crs(4326)
    aoi = ee.Geometry(mapping(zone.unary_union))

    if st.sidebar.button("ANALYSER", type="primary"):
        flood = get_flood_mask(aoi, str(start), str(end))

        buildings, routes = get_osm_data(zone)
        buildings = impacted_infra(flood, buildings)
        routes = impacted_infra(flood, routes)

        pop_total, pop_exp = population_stats(aoi, flood)

        # KPIs
        c1,c2,c3 = st.columns(3)
        c1.metric("Population totale", f"{pop_total:,}")
        c2.metric("Population exposée", f"{pop_exp:,}")
        c3.metric("Bâtiments impactés", buildings.impacted.sum())

        # Carte
        m = folium.Map(location=[zone.centroid.y.mean(), zone.centroid.x.mean()], zoom_start=12)
        folium.GeoJson(zone, style_function=lambda x:{"color":"orange","fillOpacity":0}).add_to(m)

        folium.GeoJson(
            routes,
            style_function=lambda x: {
                "color": "red" if x["properties"]["impacted"] else "#777",
                "weight": 2
            }
        ).add_to(m)

        folium.GeoJson(
            buildings[buildings.impacted],
            style_function=lambda x: {"color":"darkred","fillColor":"red","fillOpacity":0.8}
        ).add_to(m)

        st_folium(m, height=600)

        # EXPORT
        st.subheader("📤 Export")
        st.download_button(
            "Bâtiments impactés (GeoJSON)",
            buildings[buildings.impacted].to_json(),
            "buildings_impacted.geojson"
        )
        st.download_button(
            "Routes impactées (GeoJSON)",
            routes[routes.impacted].to_json(),
            "routes_impacted.geojson"
        )
