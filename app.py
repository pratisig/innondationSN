# ============================================================
# FLOOD IMPACT ANALYSIS & RESPONSE PLANNING APP
# West Africa – Sentinel-1 + OSM + Population
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import tempfile, os, json, zipfile
import pandas as pd
import plotly.express as px
from shapely.geometry import mapping

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config("Flood Impact Analysis", layout="wide", page_icon="🌊")
st.title("🌊 Flood Impact Analysis & Infrastructure Exposure")
st.caption("Sentinel-1 SAR · Population · OSM · Decision Support")

# ------------------------------------------------------------
# INIT GEE
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(
        key["client_email"], key_data=json.dumps(key)
    )
    ee.Initialize(credentials)
init_gee()

# ------------------------------------------------------------
# SHAPELY → EE
# ------------------------------------------------------------
def shapely_to_ee(geom):
    return ee.Geometry(mapping(geom))

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Zone d’étude")
uploaded = st.sidebar.file_uploader("GeoJSON / SHP (ZIP)", ["geojson", "zip"])

if not uploaded:
    st.stop()

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, uploaded.name)
    open(path, "wb").write(uploaded.getbuffer())

    if uploaded.name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        shp = [f for f in os.listdir(tmp) if f.endswith(".shp")][0]
        gdf = gpd.read_file(os.path.join(tmp, shp))
    else:
        gdf = gpd.read_file(path)

gdf = gdf.to_crs(4326)
gdf_metric = gdf.to_crs(3857)

# Nom des zones
name_field = gdf.columns[0] if gdf.columns[0] != "geometry" else None

# AOI global
aoi = shapely_to_ee(gdf.unary_union)

# ------------------------------------------------------------
# DATES
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période")
start = st.sidebar.date_input("Début", pd.to_datetime("2024-08-01"))
end = st.sidebar.date_input("Fin", pd.to_datetime("2024-09-30"))

# ------------------------------------------------------------
# FLOOD DETECTION
# ------------------------------------------------------------
@st.cache_data
def flood_map(aoi, start, end):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(str(start), str(end))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .select("VV"))
    before = s1.filterDate(str(start), str(start + pd.Timedelta(days=15))).median()
    after = s1.filterDate(str(end - pd.Timedelta(days=15)), str(end)).median()
    flood = after.subtract(before).lt(-3)

    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

    return flood.updateMask(slope.lt(5)).updateMask(water.lt(10)).selfMask()

flood = flood_map(aoi, start, end)

# ------------------------------------------------------------
# POPULATION & OSM
# ------------------------------------------------------------
pop = ee.ImageCollection("WorldPop/GP/100m/pop").mean()

buildings = ee.FeatureCollection("projects/sat-io/open-datasets/OSM/OSM_buildings")
roads = ee.FeatureCollection("projects/sat-io/open-datasets/OSM/OSM_roads")

# ------------------------------------------------------------
# METRICS PER ZONE
# ------------------------------------------------------------
results = []

for i, row in gdf.iterrows():
    poly = row.geometry
    poly_m = gdf_metric.iloc[i].geometry
    ee_poly = shapely_to_ee(poly)

    area_km2 = poly_m.area / 1e6

    flood_km2 = ee.Number(
        flood.multiply(ee.Image.pixelArea())
        .reduceRegion(ee.Reducer.sum(), ee_poly, 100)
        .get("VV")
    ).divide(1e6).getInfo() or 0

    pop_total = ee.Number(
        pop.reduceRegion(ee.Reducer.sum(), ee_poly, 100).get("population")
    ).getInfo() or 0

    pop_exp = ee.Number(
        pop.updateMask(flood).reduceRegion(ee.Reducer.sum(), ee_poly, 100).get("population")
    ).getInfo() or 0

    bldg_exp = buildings.filterBounds(ee_poly).filterBounds(flood.geometry()).size().getInfo()
    road_exp_km = roads.filterBounds(ee_poly).filterBounds(flood.geometry()).geometry().length().divide(1000).getInfo()

    results.append({
        "Zone": row[name_field] if name_field else f"Zone {i+1}",
        "Surface_km2": area_km2,
        "Flood_km2": flood_km2,
        "% Flood": flood_km2 / area_km2 * 100 if area_km2 > 0 else 0,
        "Pop totale": int(pop_total),
        "Pop exposée": int(pop_exp),
        "% Pop exposée": pop_exp / pop_total * 100 if pop_total > 0 else 0,
        "Bâtiments exposés": int(bldg_exp),
        "Routes exposées (km)": round(road_exp_km, 2)
    })

df = pd.DataFrame(results)

# ------------------------------------------------------------
# INDICATORS
# ------------------------------------------------------------
st.subheader("📊 Indicateurs clés")
c1,c2,c3 = st.columns(3)
c1.metric("Surface inondée totale", f"{df['Flood_km2'].sum():.2f} km²")
c2.metric("Population exposée", f"{df['Pop exposée'].sum():,}")
c3.metric("Bâtiments exposés", f"{df['Bâtiments exposés'].sum():,}")

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------
st.subheader("🗺️ Carte – Inondations & Infrastructures")

m = folium.Map(location=[14.5, -14.5], zoom_start=8, tiles="CartoDB positron")

folium.TileLayer(
    flood.getMapId({"palette":["0000ff"]})["tile_fetcher"].url_format,
    name="Zones inondées", overlay=True, opacity=0.5
).add_to(m)

for i,row in gdf.iterrows():
    d = df.iloc[i]
    popup = f"""
    <b>{d.Zone}</b><br>
    Surface: {d.Surface_km2:.2f} km²<br>
    Inondée: {d.Flood_km2:.2f} km² ({d['% Flood']:.1f}%)<br>
    Pop exposée: {d['Pop exposée']:,} ({d['% Pop exposée']:.1f}%)<br>
    Bâtiments exposés: {d['Bâtiments exposés']}<br>
    Routes exposées: {d['Routes exposées (km)']} km
    """
    folium.GeoJson(row.geometry, popup=popup).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, height=650, width="100%")

# ------------------------------------------------------------
# TABLE
# ------------------------------------------------------------
st.subheader("📋 Tableau récapitulatif")
st.dataframe(df.style.format({
    "Surface_km2":"{:.2f}",
    "Flood_km2":"{:.2f}",
    "% Flood":"{:.1f}%",
    "% Pop exposée":"{:.1f}%"
}))
