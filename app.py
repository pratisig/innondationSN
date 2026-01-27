# ===============================================================
# FloodWatch WA – VERSION STABLE CORRIGÉE
# ===============================================================

import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping
import ee
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# ===============================================================
# CONFIG STREAMLIT
# ===============================================================
st.set_page_config(
    page_title="FloodWatch WA Pro",
    page_icon="🌊",
    layout="wide"
)

# ===============================================================
# INITIALISATION GEE (SERVICE ACCOUNT)
# ===============================================================
@st.cache_resource
def init_gee():
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key["client_email"],
            key_data=json.dumps(key)
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Erreur GEE : {e}")
        return False

GEE_OK = init_gee()

# ===============================================================
# ÉTAT GLOBAL
# ===============================================================
if "results" not in st.session_state:
    st.session_state.results = None

if "zone" not in st.session_state:
    st.session_state.zone = None

# ===============================================================
# DONNÉES ADMINISTRATIVES (GADM)
# ===============================================================
@st.cache_data
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
    gdf = gpd.read_file(url, layer=f"ADM_ADM_{level}")
    return gdf.to_crs(4326)

# ===============================================================
# FLOOD DETECTION – MÉTHODE ROBUSTE SAR
# ===============================================================
def detect_flood(aoi, ref_start, ref_end, flood_start, flood_end):
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(aoi) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .select("VV")

    ref = s1.filterDate(ref_start, ref_end).median()
    flood = s1.filterDate(flood_start, flood_end).percentile([10])

    ref_db = ee.Image(10).multiply(ref.log10())
    flood_db = ee.Image(10).multiply(flood.log10())

    diff = ref_db.subtract(flood_db)
    water = diff.gt(1.25)

    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    water = water.updateMask(slope.lt(5))

    return water.selfMask().clip(aoi)

# ===============================================================
# POPULATION (WORLDPOP – CORRECT)
# ===============================================================
def population_stats(aoi, flood_mask):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filterDate("2020-01-01", "2021-01-01") \
        .mosaic() \
        .select("population") \
        .clip(aoi)

    total = pop.reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population")

    exposed = pop.updateMask(flood_mask).reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population")

    return (
        int(total.getInfo() or 0),
        int(exposed.getInfo() or 0)
    )

# ===============================================================
# CLIMAT – CHIRPS
# ===============================================================
def climate_stats(aoi, start, end):
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(aoi) \
        .filterDate(start, end)

    def f(img):
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "rain": img.reduceRegion(
                ee.Reducer.mean(), aoi, 5000
            ).get("precipitation")
        })

    data = rain.map(f).getInfo()
    df = pd.DataFrame([d["properties"] for d in data])
    df["date"] = pd.to_datetime(df["date"])
    return df

# ===============================================================
# OSM – BÂTIMENTS & ROUTES + IMPACT
# ===============================================================
def osm_assets(zone, flood_mask):
    poly = zone.unary_union
    buildings = ox.features_from_polygon(poly, tags={"building": True})
    roads = ox.features_from_polygon(poly, tags={"highway": True})

    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]

    # Impact via centroid sampling
    def impacted(gdf):
        centroids = gdf.geometry.centroid
        points = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point(c.x, c.y))
            for c in centroids
        ])
        sampled = flood_mask.sampleRegions(points, scale=10)
        flags = sampled.aggregate_array("flood").getInfo()
        return gdf.assign(impacted=[f == 1 for f in flags])

    return impacted(buildings), impacted(roads)

# ===============================================================
# SIDEBAR
# ===============================================================
with st.sidebar:
    st.header("Zone d’étude")

    country = st.selectbox("Pays", ["SEN", "MLI", "NER", "BFA"])
    level = st.slider("Niveau admin", 0, 3, 2)

    gadm = load_gadm(country, level)
    name_col = f"NAME_{level}" if level > 0 else "COUNTRY"

    units = st.multiselect("Unités", gadm[name_col].unique())
    if units:
        st.session_state.zone = gadm[gadm[name_col].isin(units)]

    st.subheader("Dates")
    ref = st.date_input("Période sèche", [datetime(2023,1,1), datetime(2023,4,30)])
    flood = st.date_input("Période inondation", [datetime(2024,8,1), datetime(2024,10,30)])

# ===============================================================
# ANALYSE AUTOMATIQUE
# ===============================================================
if st.session_state.zone is not None and GEE_OK:
    zone = st.session_state.zone
    aoi = ee.Geometry(mapping(zone.unary_union))

    flood_mask = detect_flood(
        aoi,
        str(ref[0]), str(ref[1]),
        str(flood[0]), str(flood[1])
    )

    pop_total, pop_exp = population_stats(aoi, flood_mask)
    climate = climate_stats(aoi, str(flood[0]), str(flood[1]))
    bld, rds = osm_assets(zone, flood_mask)

    st.session_state.results = {
        "pop_total": pop_total,
        "pop_exp": pop_exp,
        "bld": bld,
        "rds": rds,
        "climate": climate,
        "flood": flood_mask.getMapId({"palette": ["0000ff"]})
    }

# ===============================================================
# AFFICHAGE
# ===============================================================
st.title("🌊 FloodWatch WA – Résultats")

res = st.session_state.results

col1, col2, col3, col4 = st.columns(4)
col1.metric("Population totale", f"{res['pop_total']:,}" if res else "0")
col2.metric("Population exposée", f"{res['pop_exp']:,}" if res else "0")
col3.metric("Bâtiments impactés", int(res["bld"].impacted.sum()) if res else 0)
col4.metric("Routes impactées", int(res["rds"].impacted.sum()) if res else 0)

# ===============================================================
# CARTE
# ===============================================================
m = folium.Map(location=[14.5, -14.5], zoom_start=6)

if res:
    folium.TileLayer(
        tiles=res["flood"]["tile_fetcher"].url_format,
        attr="GEE",
        name="Inondation"
    ).add_to(m)

    folium.GeoJson(
        res["bld"][res["bld"].impacted],
        style_function=lambda x: {"color": "red", "fillColor": "red"},
        name="Bâtiments impactés"
    ).add_to(m)

    folium.GeoJson(
        res["rds"][res["rds"].impacted],
        style_function=lambda x: {"color": "red"},
        name="Routes impactées"
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, height=600, use_container_width=True)

# ===============================================================
# CLIMAT
# ===============================================================
if res:
    st.subheader("🌧️ Pluviométrie")
    fig = px.bar(res["climate"], x="date", y="rain")
    st.plotly_chart(fig, use_container_width=True)
