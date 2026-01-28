import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping
import json, ee, requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="FloodWatch WA - Dashboard Impact",
    layout="wide",
    page_icon="🌊"
)

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

gee_ok = init_gee()

# ============================================================================
# GADM
# ============================================================================
@st.cache_data
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
    return gpd.read_file(url, layer=f"ADM_ADM_{level}").to_crs(4326)

# ============================================================================
# FLOOD MASK (ANTI-SURESTIMATION)
# ============================================================================
def get_flood_mask(aoi, start, end, thr=1.3):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    ref_start = f"{int(start[:4])-1}{start[4:]}"
    ref_end   = f"{int(end[:4])-1}{end[4:]}"

    ref = s1.filterDate(ref_start, ref_end).median()
    flood = s1.filterDate(start, end).median()

    ref = ref.focal_median(30, "circle", "meters")
    flood = flood.focal_median(30, "circle", "meters")

    ratio = flood.subtract(ref)
    mask = ratio.lt(-thr)

    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    mask = mask.updateMask(slope.lt(5))

    return mask.focal_min(20).focal_max(20).selfMask()

# ============================================================================
# POPULATION
# ============================================================================
def population_stats(aoi, mask):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").mosaic()
    total = pop.reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo() or 0

    exposed = pop.updateMask(mask).reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo() or 0

    return int(total), int(exposed)

# ============================================================================
# OSM
# ============================================================================
def get_osm(gdf):
    poly = gdf.unary_union

    graph = ox.graph_from_polygon(poly, network_type="all")
    roads = ox.graph_to_gdfs(graph, nodes=False, edges=True).clip(gdf)

    buildings = ox.features_from_polygon(poly, tags={"building": True})
    buildings = buildings[
        buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].clip(gdf)

    return buildings.reset_index(), roads.reset_index()

# ============================================================================
# IMPACT INFRA
# ============================================================================
def impact_infra(mask, gdf):
    feats = [
        ee.Feature(ee.Geometry(mapping(g)), {"i": i})
        for i, g in enumerate(gdf.geometry)
    ]
    fc = ee.FeatureCollection(feats)
    stats = mask.reduceRegions(fc, ee.Reducer.mean(), 10)
    ids = [
        f["properties"]["i"]
        for f in stats.filter(ee.Filter.gt("mean", 0))
        .getInfo()["features"]
    ]
    gdf["impacted"] = gdf.index.isin(ids)
    return gdf

# ============================================================================
# SIDEBAR – ZONE SELECTION
# ============================================================================
st.sidebar.header("🗺️ Zone d’étude")

mode = st.sidebar.radio(
    "Méthode de sélection",
    ["Administrative", "Dessiner", "Importer"]
)

zone = None
zone_name = "Zone d’étude"

if mode == "Administrative":
    countries = {"Sénégal":"SEN","Mali":"MLI","Niger":"NER","Burkina Faso":"BFA"}
    c = st.sidebar.selectbox("Pays", countries.keys())
    lvl = st.sidebar.slider("Niveau Admin", 0, 5, 2)

    gadm = load_gadm(countries[c], lvl)
    col = [c for c in gadm.columns if c.startswith("NAME_")][-1]
    name = st.sidebar.selectbox("Subdivision", sorted(gadm[col].unique()))
    zone = gadm[gadm[col] == name]
    zone_name = name

elif mode == "Dessiner":
    st.sidebar.info("Dessinez un polygone")
    mdraw = folium.Map(location=[14.5,-14.5], zoom_start=6)
    Draw().add_to(mdraw)
    out = st_folium(mdraw, height=300)
    if out and out["last_active_drawing"]:
        geom = shape(out["last_active_drawing"]["geometry"])
        zone = gpd.GeoDataFrame(geometry=[geom], crs=4326)
        zone_name = "Zone dessinée"

elif mode == "Importer":
    up = st.sidebar.file_uploader("GeoJSON / KML", ["geojson","kml"])
    if up:
        zone = gpd.read_file(up).to_crs(4326)
        zone_name = "Zone importée"

# ============================================================================
# DATES
# ============================================================================
st.sidebar.header("📅 Période")
start = st.sidebar.date_input("Début", datetime(2024,8,1))
end   = st.sidebar.date_input("Fin", datetime(2024,9,30))

# ============================================================================
# MAIN
# ============================================================================
st.title(f"🌊 FloodWatch – {zone_name}")

if zone is None:
    m = folium.Map(location=[14.5,-14.5], zoom_start=6)
    st_folium(m, height=500)
    st.info("Sélectionne une zone pour commencer")
    st.stop()

if st.button("🚀 ANALYSER", type="primary", use_container_width=True):
    aoi = ee.Geometry(mapping(zone.unary_union))
    flood = get_flood_mask(aoi, str(start), str(end))

    buildings, roads = get_osm(zone)
    buildings = impact_infra(flood, buildings)
    roads = impact_infra(flood, roads)

    pop_tot, pop_exp = population_stats(aoi, flood)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Population totale", f"{pop_tot:,}")
    c2.metric("Population exposée", f"{pop_exp:,}")
    c3.metric("Bâtiments impactés", buildings.impacted.sum())
    c4.metric("Routes impactées (segments)", roads.impacted.sum())

    # MAP
    center = zone.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=11)

    folium.GeoJson(zone, style_function=lambda x:{"color":"orange","fillOpacity":0}).add_to(m)

    folium.GeoJson(
        roads,
        style_function=lambda x:{
            "color":"red" if x["properties"]["impacted"] else "#666",
            "weight":2
        }
    ).add_to(m)

    folium.GeoJson(
        buildings[buildings.impacted],
        style_function=lambda x:{
            "color":"darkred","fillColor":"red","fillOpacity":0.8
        }
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
        roads[roads.impacted].to_json(),
        "roads_impacted.geojson"
    )
