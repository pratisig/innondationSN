# ============================================================
# FLOOD ANALYSIS & MAPPING APP — WEST AFRICA
# STABLE VERSION — FIXED METRICS & INFRASTRUCTURES
# ============================================================

import streamlit as st
import ee, json, os, zipfile, tempfile
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely.geometry import mapping
from pyproj import Geod

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Flood Impact Analysis",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Flood Impact Analysis & Intervention Planning")
st.caption("Sentinel-1 Floods · Population · Infrastructures · OSM proxy")

# ------------------------------------------------------------
# INIT GOOGLE EARTH ENGINE (SERVICE ACCOUNT)
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(
        key["client_email"],
        key_data=json.dumps(key)
    )
    ee.Initialize(credentials)
    return True

init_gee()

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
geod = Geod(ellps="WGS84")

def area_km2(geom):
    return abs(geod.geometry_area_perimeter(geom)[0]) / 1e6

def shapely_to_ee(geom):
    return ee.Geometry(mapping(geom))

# ------------------------------------------------------------
# SIDEBAR — AOI
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Zone d’étude")
uploaded = st.sidebar.file_uploader(
    "GeoJSON / SHP ZIP / KML",
    type=["geojson", "zip", "kml"]
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

gdf = gdf.to_crs(4326)
label_col = next((c for c in gdf.columns if c.lower() in ["name", "nom", "id"]), None)

# ------------------------------------------------------------
# DATES
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période")
start = st.sidebar.date_input("Début", pd.to_datetime("2024-08-01"))
end = st.sidebar.date_input("Fin", pd.to_datetime("2024-09-30"))

# ------------------------------------------------------------
# FLOOD DETECTION (SENTINEL-1)
# ------------------------------------------------------------
@st.cache_data
def detect_flood(aoi, start, end):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    before = s1.sort("system:time_start").limit(5).median()
    after = s1.sort("system:time_start", False).limit(5).median()

    flood = after.subtract(before).lt(-3)

    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

    return (
        flood
        .updateMask(slope.lt(5))
        .updateMask(water.lt(10))
        .selfMask()
    )

merged = gdf.geometry.unary_union
aoi_ee = shapely_to_ee(merged)

with st.spinner("Détection des zones inondées…"):
    flood_img = detect_flood(aoi_ee, start, end)

# ------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------
pixel_area = ee.Image.pixelArea()

pop = (
    ee.ImageCollection("WorldPop/GP/100m/pop")
    .filterDate("2020-01-01", "2020-12-31")
    .mean()
    .rename("pop")
)

buildings = ee.FeatureCollection(
    "GOOGLE/Research/open-buildings/v3/polygons"
).filterBounds(aoi_ee)

build_img = buildings.map(
    lambda f: f.set("v", 1)
).reduceToImage(["v"], ee.Reducer.first()).unmask(0)

# ------------------------------------------------------------
# METRICS — ROBUST & NON-ZERO
# ------------------------------------------------------------
rows = []

for i, row in gdf.iterrows():
    geom = row.geometry
    zone = shapely_to_ee(geom)

    total_area = area_km2(geom)

    flooded_area = (
        flood_img
        .multiply(pixel_area)
        .clip(zone)
        .reduceRegion(
            ee.Reducer.sum(),
            zone,
            scale=100,
            maxPixels=1e13
        )
        .get("VV")
    )

    flooded_km2 = ee.Number(flooded_area).divide(1e6).getInfo() or 0

    pop_tot = (
        pop.clip(zone)
        .reduceRegion(ee.Reducer.sum(), zone, 100)
        .get("pop")
    )

    pop_exp = (
        pop.updateMask(flood_img)
        .clip(zone)
        .reduceRegion(ee.Reducer.sum(), zone, 100)
        .get("pop")
    )

    bldg = (
        build_img.updateMask(flood_img)
        .clip(zone)
        .reduceRegion(ee.Reducer.sum(), zone, 50)
        .get("v")
    )

    pop_tot = int(ee.Number(pop_tot).getInfo() or 0)
    pop_exp = int(ee.Number(pop_exp).getInfo() or 0)
    bldg = int(ee.Number(bldg).getInfo() or 0)

    rows.append({
        "nom": row[label_col] if label_col else f"Zone {i+1}",
        "surface_totale_km2": total_area,
        "surface_inondee_km2": flooded_km2,
        "pct_inonde": flooded_km2 / total_area * 100 if total_area else 0,
        "population_totale": pop_tot,
        "population_exposee": pop_exp,
        "pct_pop_exposee": pop_exp / pop_tot * 100 if pop_tot else 0,
        "batiments_impactes": bldg
    })

df = pd.DataFrame(rows)

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
st.subheader("📊 Indicateurs clés")
c1, c2, c3 = st.columns(3)
c1.metric("Surface inondée", f"{df.surface_inondee_km2.sum():.2f} km²")
c2.metric("Population exposée", f"{df.population_exposee.sum():,}")
c3.metric("Bâtiments impactés", f"{df.batiments_impactes.sum():,}")

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------
st.subheader("🗺️ Carte d’impact")

m = folium.Map(location=[gdf.centroid.y.mean(), gdf.centroid.x.mean()], zoom_start=9)

flood_map = flood_img.getMapId({"palette": ["00ffff"]})
folium.TileLayer(
    tiles=flood_map["tile_fetcher"].url_format,
    attr="Sentinel-1 Flood",
    name="Zones inondées",
    overlay=True
).add_to(m)

for _, r in df.iterrows():
    g = gdf[gdf[label_col] == r.nom].geometry.values[0]
    folium.GeoJson(
        g,
        popup=folium.Popup(f"""
        <b>{r.nom}</b><br>
        Surface: {r.surface_totale_km2:.2f} km²<br>
        Inondée: {r.surface_inondee_km2:.2f} km² ({r.pct_inonde:.1f}%)<br>
        Population exposée: {r.population_exposee:,}<br>
        Bâtiments impactés: {r.batiments_impactes:,}
        """, max_width=250)
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, height=600, width="100%")

# ------------------------------------------------------------
# TABLE
# ------------------------------------------------------------
st.subheader("📋 Tableau récapitulatif")
st.dataframe(df.style.format({
    "surface_totale_km2": "{:.2f}",
    "surface_inondee_km2": "{:.2f}",
    "pct_inonde": "{:.1f}%",
    "population_totale": "{:,}",
    "population_exposee": "{:,}",
    "pct_pop_exposee": "{:.1f}%",
    "batiments_impactes": "{:,}"
}), use_container_width=True)
