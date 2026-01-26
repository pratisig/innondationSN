# ============================================================
# FLOOD IMPACT ANALYSIS APP — WEST AFRICA
# ============================================================

import streamlit as st
import ee
import json
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import tempfile, os, zipfile
from shapely.ops import unary_union
import plotly.express as px
import osmnx as ox

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Flood Impact Analysis",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Flood Impact Analysis & Emergency Planning")
st.caption("Sentinel-1 / Sentinel-2 / CHIRPS / WorldPop / OpenStreetMap")

# ============================================================
# EARTH ENGINE AUTH — SERVICE ACCOUNT ONLY
# ============================================================
@st.cache_resource
def init_ee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    creds = ee.ServiceAccountCredentials(
        key["client_email"],
        key_data=json.dumps(key)
    )
    ee.Initialize(creds)

init_ee()

# ============================================================
# FILE UPLOAD
# ============================================================
st.sidebar.header("1️⃣ Zone d’étude")

uploaded = st.sidebar.file_uploader(
    "GeoJSON / SHP (zip)",
    type=["geojson", "zip"]
)

if not uploaded:
    st.stop()

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    if uploaded.name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        shp = [f for f in os.listdir(tmp) if f.endswith(".shp")][0]
        gdf = gpd.read_file(os.path.join(tmp, shp))
    else:
        gdf = gpd.read_file(path)

# CRS
gdf = gdf.to_crs(4326)

# Nom des zones
if "name" not in gdf.columns:
    gdf["name"] = [f"Zone {i+1}" for i in range(len(gdf))]

# ============================================================
# SURFACE CORRECTE (UTM)
# ============================================================
gdf_utm = gdf.to_crs(32628)  # Sénégal
gdf["area_km2"] = gdf_utm.area / 1e6

# ============================================================
# GEE GEOMETRY
# ============================================================
aoi = unary_union(gdf.geometry)
ee_geom = ee.Geometry(aoi.__geo_interface__)

# ============================================================
# DATE
# ============================================================
st.sidebar.header("2️⃣ Période")
start = st.sidebar.date_input("Début", pd.to_datetime("2024-08-01"))
end = st.sidebar.date_input("Fin", pd.to_datetime("2024-09-30"))

# ============================================================
# FLOOD DETECTION — SENTINEL-1
# ============================================================
@st.cache_data
def flood_map(aoi_json, s, e):
    geom = ee.Geometry(aoi_json)

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(str(s), str(e))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    before = s1.filterDate(str(s), str(pd.to_datetime(s)+pd.Timedelta(days=15))).median()
    after = s1.filterDate(str(pd.to_datetime(e)-pd.Timedelta(days=15)), str(e)).median()

    flood = after.subtract(before).lt(-3)

    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

    return flood.updateMask(slope.lt(5)).updateMask(perm.lt(10)).selfMask()

flood = flood_map(ee_geom.getInfo(), start, end)

# ============================================================
# INDICATEURS GLOBAUX
# ============================================================
pixel_area = ee.Image.pixelArea()

flood_area = flood.multiply(pixel_area).reduceRegion(
    ee.Reducer.sum(), ee_geom, 100, maxPixels=1e13
).get("VV")

flood_km2 = ee.Number(flood_area).divide(1e6).getInfo()

rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
    .filterBounds(ee_geom).filterDate(str(start), str(end)).sum()

rain_mm = rain.reduceRegion(
    ee.Reducer.mean(), ee_geom, 5000, maxPixels=1e13
).get("precipitation").getInfo()

pop = ee.ImageCollection("WorldPop/GP/100m/pop").mean()

pop_exp = pop.updateMask(flood).reduceRegion(
    ee.Reducer.sum(), ee_geom, 100, maxPixels=1e13
).get("population").getInfo()

# ============================================================
# DISPLAY METRICS
# ============================================================
st.subheader("📊 Indicateurs clés")

c1, c2, c3 = st.columns(3)
c1.metric("Surface inondée", f"{flood_km2:.2f} km²")
c2.metric("Population exposée", f"{int(pop_exp):,}")
c3.metric("Pluie cumulée", f"{rain_mm:.1f} mm")

# ============================================================
# OSM — INFRASTRUCTURES EXPOSÉES
# ============================================================
st.subheader("🏗️ Infrastructures exposées")

tags = {
    "amenity": True,
    "highway": True,
    "building": True
}

osm = ox.geometries_from_polygon(aoi, tags)
osm = osm.to_crs(32628)

# Approximation exposition
osm["exposed"] = osm.intersects(unary_union(gdf_utm.geometry))

infra_stats = osm["exposed"].value_counts()

st.write({
    "Total infrastructures": len(osm),
    "Infrastructures exposées": int(infra_stats.get(True, 0))
})

# ============================================================
# MAP
# ============================================================
st.subheader("🗺️ Carte interactive")

bounds = gdf.total_bounds
m = folium.Map(
    location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2],
    zoom_start=9,
    tiles="CartoDB positron"
)

# Flood layer
flood_id = flood.getMapId({"palette":["0000FF"]})
folium.TileLayer(
    tiles=flood_id["tile_fetcher"].url_format,
    name="Zones inondées",
    overlay=True,
    opacity=0.6
).add_to(m)

# Zones + popup
for _, r in gdf.iterrows():
    html = f"""
    <b>{r['name']}</b><br>
    Surface totale : {r['area_km2']:.2f} km²<br>
    """
    folium.GeoJson(
        r.geometry,
        popup=html,
        style_function=lambda x: {"color":"red","weight":2,"fillOpacity":0}
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, height=600)

# ============================================================
# TABLE RÉCAP
# ============================================================
st.subheader("📋 Tableau récapitulatif")

df = gdf[["name","area_km2"]].copy()
df["Flood_km2_est"] = flood_km2 * (df["area_km2"] / df["area_km2"].sum())
df["% Inondée"] = (df["Flood_km2_est"] / df["area_km2"]) * 100

st.dataframe(df, use_container_width=True)
