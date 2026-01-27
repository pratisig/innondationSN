# ===============================================================
# FLOODWATCH WA – VERSION STABLE & CORRIGÉE
# ===============================================================

import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import ee
import json
import osmnx as ox
import pandas as pd
import plotly.express as px
from shapely.geometry import mapping
from datetime import datetime

# ===============================================================
# CONFIG
# ===============================================================
st.set_page_config(
    page_title="FloodWatch WA Pro",
    page_icon="🌊",
    layout="wide"
)

# ===============================================================
# GEE INIT (SERVICE ACCOUNT OK)
# ===============================================================
@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    creds = ee.ServiceAccountCredentials(
        key["client_email"],
        key_data=json.dumps(key)
    )
    ee.Initialize(creds)
    return True

gee_ok = init_gee()

# ===============================================================
# SESSION STATE
# ===============================================================
for k in ["results", "aoi_gdf"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ===============================================================
# LOAD GADM
# ===============================================================
@st.cache_data
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
    return gpd.read_file(url, layer=f"ADM_{level}").to_crs(4326)

# ===============================================================
# FLOOD DETECTION (ROBUST)
# ===============================================================
def detect_flood(aoi, d1, d2, d3, d4):
    col = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(aoi) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .select("VV")

    ref = col.filterDate(d1, d2).median()
    crisis = col.filterDate(d3, d4).median()

    flood = ref.subtract(crisis).gt(1.2)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    flood = flood.updateMask(slope.lt(5))

    return flood.clip(aoi).selfMask()

# ===============================================================
# POPULATION (WORLDPOP 100m CORRECT)
# ===============================================================
def population_stats(aoi, flood):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filterDate("2020-01-01", "2021-01-01") \
        .mosaic() \
        .clip(aoi)

    total = pop.reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo()

    exposed = pop.updateMask(flood).reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo()

    return int(total or 0), int(exposed or 0)

# ===============================================================
# OSM DATA
# ===============================================================
def load_osm(aoi_gdf):
    poly = aoi_gdf.unary_union
    buildings = ox.features_from_polygon(poly, tags={"building": True})
    roads = ox.graph_to_gdfs(
        ox.graph_from_polygon(poly, network_type="drive"),
        nodes=False, edges=True
    )
    return buildings.to_crs(4326), roads.to_crs(4326)

# ===============================================================
# SIDEBAR
# ===============================================================
with st.sidebar:
    st.header("Zone d'étude")

    country = st.selectbox("Pays", {"Sénégal":"SEN","Mali":"MLI"})
    level = st.slider("Niveau admin", 0, 3, 2)

    gadm = load_gadm(country, level)
    name_col = gadm.columns[gadm.columns.str.contains("NAME")][0]
    choice = st.selectbox("Zone", gadm[name_col].unique())

    aoi = gadm[gadm[name_col] == choice]
    st.session_state.aoi_gdf = aoi

    d_ref = st.date_input("Période sèche", [datetime(2023,1,1), datetime(2023,4,30)])
    d_evt = st.date_input("Période inondation", [datetime(2024,8,1), datetime(2024,10,30)])

    run = st.button("🚀 Lancer analyse", type="primary")

# ===============================================================
# ANALYSE
# ===============================================================
if run:
    aoi_ee = ee.Geometry(mapping(st.session_state.aoi_gdf.unary_union))
    flood = detect_flood(aoi_ee, *map(str, d_ref), *map(str, d_evt))

    total_pop, exp_pop = population_stats(aoi_ee, flood)

    buildings, roads = load_osm(st.session_state.aoi_gdf)

    st.session_state.results = {
        "flood": flood.getMapId({"palette":["#0000FF"]}),
        "total_pop": total_pop,
        "exp_pop": exp_pop,
        "buildings": buildings,
        "roads": roads
    }

# ===============================================================
# DISPLAY RESULTS (STABLE)
# ===============================================================
if st.session_state.results:
    r = st.session_state.results

    c1,c2,c3 = st.columns(3)
    c1.metric("Population totale", f"{r['total_pop']:,}")
    c2.metric("Population impactée", f"{r['exp_pop']:,}")
    c3.metric("Bâtiments", len(r["buildings"]))

    center = st.session_state.aoi_gdf.centroid.iloc[0]
    m = folium.Map([center.y, center.x], zoom_start=11)

    folium.GeoJson(st.session_state.aoi_gdf).add_to(m)

    folium.TileLayer(
        tiles=r["flood"]["tile_fetcher"].url_format,
        attr="GEE Flood",
        name="Inondation"
    ).add_to(m)

    folium.GeoJson(
        r["buildings"],
        style_function=lambda x: {"color":"red","weight":1},
        name="Bâtiments"
    ).add_to(m)

    folium.GeoJson(
        r["roads"],
        style_function=lambda x: {"color":"red","weight":2},
        name="Routes"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, height=650, width="100%")

else:
    st.info("Sélectionne une zone et lance l’analyse.")
