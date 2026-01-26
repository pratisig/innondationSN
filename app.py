# app.py
# ============================================================
# FLOOD ANALYSIS & MAPPING APP — SENEGAL / WEST AFRICA
# Uses real open data via Google Earth Engine (GEE)
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
from shapely.geometry import Polygon, MultiPolygon
import zipfile

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
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(
        key["client_email"],
        key_data=json.dumps(key)
    )
    ee.Initialize(credentials)
    return True

init_gee()

# ------------------------------------------------------------
# CONVERSION SHAPELY -> EARTH ENGINE (FIXED)
# ------------------------------------------------------------
def shapely_to_ee(poly):
    """
    Convertit une géométrie Shapely (Polygon ou MultiPolygon) en 
    géométrie Google Earth Engine en utilisant le standard GeoJSON.
    """
    # L'interface __geo_interface__ fournit le dictionnaire GeoJSON
    geojson = poly.__geo_interface__
    
    if geojson['type'] == 'Polygon':
        return ee.Geometry.Polygon(geojson['coordinates'])
    elif geojson['type'] == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(geojson['coordinates'])
    else:
        # Gérer les autres types (Point, LineString) si nécessaire
        coords = geojson['coordinates']
        return ee.Geometry(geojson)

# ------------------------------------------------------------
# LOGIQUE DE CHARGEMENT ET TRAITEMENT
# ------------------------------------------------------------

init_gee()

st.title("🌊 Analyse des Inondations")

uploaded_file = st.sidebar.file_uploader(
    "Charger une zone (GeoJSON / SHP ZIP / KML)",
    type=["geojson", "kml", "zip"]
)

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Chargement selon l'extension
            if uploaded_file.name.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
                gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
            else:
                gdf = gpd.read_file(file_path)

            # Vérification de la géométrie
            if gdf.empty:
                st.error("Le fichier est vide.")
                st.stop()

            # Union des géométries pour obtenir la zone d'étude globale
            # Cela gère les fichiers avec plusieurs lignes (ex: plusieurs communes)
            merged_poly = gdf.geometry.unary_union
            geom = shapely_to_ee(merged_poly)
            
            st.success("✅ Géométrie convertie avec succès pour Google Earth Engine.")

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

# Population
pop = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01", "2020-12-31").mean()

# ------------------------------------------------------------
# INDICATORS (GLOBAL)
# ------------------------------------------------------------
pixel_area = ee.Image.pixelArea()

flood_area = flood_img.multiply(pixel_area).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geom,
    scale=30,
    maxPixels=1e13
)

flood_area_km2 = ee.Number(flood_area.get("VV")).divide(1e6)

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
# MAP INTERACTIVE AVEC POPUPS
# ------------------------------------------------------------
st.subheader("🗺️ Carte interactive avec popups")

# Centrer la carte automatiquement
bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

# Fonction pour calculer metrics par polygone
def compute_metrics(poly_geom):
    ee_poly = shapely_to_ee(poly_geom)
    
    # Surface inondée
    flood_area_poly = flood_img.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ee_poly,
        scale=30,
        maxPixels=1e13
    )
    flood_km2 = ee.Number(flood_area_poly.get("VV")).divide(1e6).getInfo() or 0
    
    # Population exposée
    pop_exposed_poly = pop.updateMask(flood_img).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ee_poly,
        scale=100,
        maxPixels=1e13
    ).getInfo().get("population", 0)
    
    # Surface totale polygone (km²)
    total_area = poly_geom.area / 1e6  # m² → km²
    return total_area, flood_km2, int(pop_exposed_poly)

# Ajouter chaque polygone avec popup
for idx, row in gdf.iterrows():
    poly = row.geometry
    total_area, flood_area_poly, pop_exposed_poly = compute_metrics(poly)
    
    popup_html = f"""
    <b>Zone {idx + 1}</b><br>
    Surface totale: {total_area:.2f} km²<br>
    Surface inondée: {flood_area_poly:.2f} km²<br>
    Population exposée: {pop_exposed_poly}
    """
    
    folium.GeoJson(
        poly,
        style_function=lambda feature: {
            "fillColor": "#ff7800",
            "color": "#ff7800",
            "weight": 2,
            "fillOpacity": 0.2,
        },
        tooltip=folium.Tooltip(f"Zone {idx + 1}"),
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

# Zones inondées overlay
flood_vis = {"min": 0, "max": 1, "palette": ["blue"]}
flood_layer = folium.raster_layers.TileLayer(
    tiles=flood_img.getMapId(flood_vis)["tile_fetcher"].url_format,
    attr="Flood extent",
    name="Zones inondées",
    overlay=True,
    control=True,
    opacity=0.6,
)
flood_layer.add_to(m)

# Légende
legend_html = """
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 60px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; padding: 5px;">
     <b>Légende</b><br>
     <i style="background: #ff7800; width: 15px; height: 15px; float: left; margin-right: 5px;"></i> Zone d'étude<br>
     <i style="background: blue; width: 15px; height: 15px; float: left; margin-right: 5px;"></i> Zones inondées
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
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
