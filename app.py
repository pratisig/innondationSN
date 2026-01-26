# app.py
# ============================================================
# FLOOD ANALYSIS & MAPPING APP — SENEGAL / WEST AFRICA
# Uses real open data via Google Earth Engine (GEE)
# ============================================================
# REQUIREMENTS:
# - Google Earth Engine account enabled
# - Service Account JSON uploaded as Streamlit secret or local file
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import tempfile
import os
import json
import pandas as pd
import plotly.express as px

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Flood Impact Analysis – West Africa",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Flood Impact Analysis & Mapping")
st.caption("Satellite-based flood detection using Sentinel-1 SAR and open datasets")

# ------------------------------------------------------------
# AUTHENTICATE GOOGLE EARTH ENGINE
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key["client_email"],
                key_data=st.secrets["GEE_SERVICE_ACCOUNT"]
            )
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        st.error("Google Earth Engine authentication failed.")
        st.exception(e)
        return False

gee_ok = init_gee()
if not gee_ok:
    st.stop()

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Zone d’étude")
uploaded_file = st.sidebar.file_uploader(
    "Charger une zone (GeoJSON / SHP / KML)",
    type=["geojson", "shp", "kml"]
)

if not uploaded_file:
    st.info("Veuillez charger une zone géographique pour commencer.")
    st.stop()

# Save uploaded file temporarily
with tempfile.TemporaryDirectory() as tmpdir:
    file_path = os.path.join(tmpdir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    gdf = gpd.read_file(file_path)

# Convert to EE Geometry
geom = ee.Geometry.Polygon(gdf.geometry.iloc[0].__geo_interface__["coordinates"])

# ------------------------------------------------------------
# DATE SELECTION
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période d’analyse")
start_date = st.sidebar.date_input("Date de début")
end_date = st.sidebar.date_input("Date de fin")

if start_date >= end_date:
    st.error("La date de fin doit être postérieure à la date de début.")
    st.stop()

# ------------------------------------------------------------
# SENTINEL-1 FLOOD DETECTION
# ------------------------------------------------------------
@st.cache_data
def detect_floods(aoi, start, end):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    before = s1.filterDate(str(start), str(pd.to_datetime(start) + pd.Timedelta(days=7))).median()
    after = s1.filterDate(str(pd.to_datetime(end) - pd.Timedelta(days=7)), str(end)).median()

    diff = after.subtract(before)

    flood = diff.lt(-3)  # SAR adaptive threshold
    flood = flood.updateMask(flood)

    return flood

with st.spinner("Détection des zones inondées (Sentinel-1 SAR)…"):
    flood_img = detect_floods(geom, start_date, end_date)

# ------------------------------------------------------------
# ANCILLARY DATA
# ------------------------------------------------------------
dem = ee.Image("USGS/SRTMGL1_003")
slope = ee.Terrain.slope(dem)
flood_img = flood_img.updateMask(slope.lt(5))

water_perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
flood_img = flood_img.updateMask(water_perm.lt(10))

# ------------------------------------------------------------
# INDICATORS
# ------------------------------------------------------------
pixel_area = ee.Image.pixelArea()

flood_area = flood_img.multiply(pixel_area).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geom,
    scale=30,
    maxPixels=1e13
)

flood_area_km2 = ee.Number(flood_area.get("VV")).divide(1e6)

# Rainfall (CHIRPS)
rain = (
    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterBounds(geom)
    .filterDate(str(start_date), str(end_date))
    .sum()
)

rain_mm = rain.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=geom,
    scale=5000,
    maxPixels=1e13
).get("precipitation")

# Population
pop = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01", "2020-12-31").mean()
pop_exposed = pop.updateMask(flood_img).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geom,
    scale=100,
    maxPixels=1e13
).get("population")

# ------------------------------------------------------------
# DISPLAY METRICS
# ------------------------------------------------------------
st.subheader("📊 Indicateurs clés")

col1, col2, col3 = st.columns(3)
col1.metric("Surface inondée (km²)", flood_area_km2.getInfo())
col2.metric("Pluie cumulée (mm)", rain_mm.getInfo())
col3.metric("Population exposée", pop_exposed.getInfo())

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------
st.subheader("🗺️ Carte interactive")

m = folium.Map(location=[14.5, -14.5], zoom_start=7)
folium.GeoJson(gdf, name="Zone d’étude").add_to(m)

flood_vis = {"min": 0, "max": 1, "palette": ["blue"]}
flood_layer = folium.raster_layers.TileLayer(
    tiles=flood_img.getMapId(flood_vis)["tile_fetcher"].url_format,
    attr="Flood extent",
    name="Zones inondées",
    overlay=True,
    control=True
)
flood_layer.add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=1100, height=600)

# ------------------------------------------------------------
# TIME SERIES (OPTIONAL)
# ------------------------------------------------------------
st.subheader("📈 Évolution temporelle (pluie)")
dates = pd.date_range(start_date, end_date, freq="D")
rain_series = []

for d in dates:
    daily = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(geom)
        .filterDate(str(d.date()), str(d.date() + pd.Timedelta(days=1)))
        .mean()
    )
    val = daily.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=5000,
        maxPixels=1e13
    ).get("precipitation")
    rain_series.append(val.getInfo())

df = pd.DataFrame({"Date": dates, "Rain_mm": rain_series})
fig = px.line(df, x="Date", y="Rain_mm", title="Pluviométrie journalière")
st.plotly_chart(fig, use_container_width=True)
