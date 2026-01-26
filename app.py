# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import geopandas as gpd
import osmnx as ox
import ee
import json
from shapely.ops import unary_union

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Flood Impact Analyzer",
    layout="wide",
    page_icon="🌊"
)

# ============================================================
# EARTH ENGINE INIT (CORRIGÉ)
# ============================================================
try:
    service_account = st.secrets["ee"]["client_email"]
    key_dict = json.loads(st.secrets["ee"]["private_key"])
    credentials = ee.ServiceAccountCredentials(service_account, key_dict)
    ee.Initialize(credentials)
except Exception as e:
    st.error("❌ Impossible d'initialiser Google Earth Engine")
    st.exception(e)
    st.stop()

# ============================================================
# UTILS
# ============================================================
def ee_polygon_from_gdf(gdf):
    geom = gdf.geometry.unary_union.__geo_interface__
    return ee.Geometry(geom)

# ============================================================
# ADMIN DATA LOADING (ADM1 / ADM2 / ADM3)
# ============================================================
@st.cache_data
def load_admin_layer(path):
    return gpd.read_file(path).to_crs(4326)

adm_level = st.selectbox("Niveau administratif", ["ADM1", "ADM2", "ADM3"])

if adm_level == "ADM1":
    gdf_admin = load_admin_layer("data/adm1.geojson")
elif adm_level == "ADM2":
    gdf_admin = load_admin_layer("data/adm2.geojson")
else:
    gdf_admin = load_admin_layer("data/adm3.geojson")

admin_name = st.selectbox("Zone", gdf_admin["NAME"].unique())
zone_gdf = gdf_admin[gdf_admin["NAME"] == admin_name]

merged_poly = unary_union(zone_gdf.geometry)

# ============================================================
# FLOOD ANALYSIS (EARTH ENGINE)
# ============================================================
def analyze_flood_extent(ee_geom):
    flood = (
        ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
        .select("water")
        .filterDate("2023-01-01", "2023-12-31")
        .mean()
        .gt(2)
        .selfMask()
    )

    area = flood.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ee_geom,
        scale=30,
        maxPixels=1e13
    )

    return flood, area.get("water")

ee_zone = ee_polygon_from_gdf(zone_gdf)
flood_img, flood_area = analyze_flood_extent(ee_zone)

flood_km2 = ee.Number(flood_area).divide(1e6).getInfo()

# ============================================================
# POPULATION IMPACT
# ============================================================
def analyze_population_exposed(flood, ee_geom):
    pop = ee.Image("WorldPop/GP/100m/pop").select("population")
    exposed = pop.updateMask(flood)

    stats = exposed.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ee_geom,
        scale=100,
        maxPixels=1e13
    )

    return stats.get("population").getInfo()

pop_exposed = analyze_population_exposed(flood_img, ee_zone)

# ============================================================
# OSM ANALYSIS (OSMNX – 100% PYTHON)
# ============================================================
def analyze_infrastructure_impact_osmnx(admin_polygon):
    tags = {
        "building": True,
        "highway": True,
        "amenity": ["hospital", "school"]
    }

    try:
        osm = ox.geometries_from_polygon(admin_polygon, tags)
    except Exception:
        return dict(buildings="N/A", roads_km="N/A", health="N/A", education="N/A")

    buildings = osm[osm["building"].notna()]
    roads = osm[osm["highway"].notna()]
    health = osm[osm["amenity"] == "hospital"]
    education = osm[osm["amenity"] == "school"]

    roads_km = roads.length.sum() / 1000 if not roads.empty else 0

    return {
        "buildings": len(buildings),
        "roads_km": round(roads_km, 2),
        "health": len(health),
        "education": len(education)
    }

osm_data = analyze_infrastructure_impact_osmnx(merged_poly)

# ============================================================
# DASHBOARD
# ============================================================
st.subheader("📊 Impacts estimés")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("🌊 Surface inondée (km²)", round(flood_km2, 2))
c2.metric("👥 Population exposée", int(pop_exposed) if pop_exposed else "N/A")
c3.metric("🏠 Bâtiments", osm_data["buildings"])
c4.metric("🏥 Santé", osm_data["health"])
c5.metric("🎓 Éducation", osm_data["education"])

st.metric("🛣️ Routes (km)", osm_data["roads_km"])
