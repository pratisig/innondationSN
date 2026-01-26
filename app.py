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
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("Secret 'GEE_SERVICE_ACCOUNT' manquant dans les paramètres Streamlit.")
        st.stop()
    
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
    Convertit une géométrie Shapely en Earth Engine via l'interface GeoJSON.
    C'est la méthode la plus robuste pour éviter les erreurs de profondeur de listes.
    """
    geojson = poly.__geo_interface__
    if geojson['type'] == 'Polygon':
        return ee.Geometry.Polygon(geojson['coordinates'])
    elif geojson['type'] == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(geojson['coordinates'])
    else:
        return ee.Geometry(geojson)

# ------------------------------------------------------------
# LOGIQUE DE CHARGEMENT DES FICHIERS
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Zone d’étude")
uploaded_file = st.sidebar.file_uploader(
    "Charger une zone (GeoJSON / SHP ZIP / KML)",
    type=["geojson", "kml", "zip"]
)

if not uploaded_file:
    st.info("Veuillez charger une zone géographique (ex: limites administratives) pour commencer.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    file_path = os.path.join(tmpdir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                st.error("Aucun fichier .shp trouvé dans le zip.")
                st.stop()
            gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
        else:
            gdf = gpd.read_file(file_path)
            
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")

    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        st.stop()

# Création de la géométrie globale pour GEE
# Simplification légère (0.001 deg ~ 100m) pour éviter les erreurs de payload trop lourd
merged_poly = gdf.geometry.unary_union.simplify(0.001, preserve_topology=True)
geom = shapely_to_ee(merged_poly)
st.success(f"✅ Zone d’étude chargée : {len(gdf)} entité(s) détectée(s).")

# ------------------------------------------------------------
# DATE SELECTION
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période d’analyse")
start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2024-08-01"))
end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2024-09-30"))

if start_date >= end_date:
    st.error("La date de fin doit être postérieure à la date de début.")
    st.stop()

# ------------------------------------------------------------
# SENTINEL-1 FLOOD DETECTION
# ------------------------------------------------------------
@st.cache_data
def detect_floods(aoi_serialized, start, end):
    # On reconstruit la géométrie car ee.Geometry n'est pas sérialisable par st.cache
    aoi = ee.Geometry(aoi_serialized)
    
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    # Réduction temporelle (médiane) pour stabiliser le signal
    # "Before" : début de période, "After" : fin de période
    before = s1.filterDate(str(start), str(pd.to_datetime(start) + pd.Timedelta(days=15))).median()
    after = s1.filterDate(str(pd.to_datetime(end) - pd.Timedelta(days=15)), str(end)).median()

    # Détection de changement (Seuil adaptatif SAR)
    diff = after.subtract(before)
    flood = diff.lt(-3)  # Seuil classique de -3dB pour l'eau
    
    # Nettoyage : retirer les pentes fortes (SRTM) et l'eau permanente (JRC)
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    water_perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    
    flood_clean = flood.updateMask(slope.lt(5)) # Moins de 5 degrés de pente
    flood_clean = flood_clean.updateMask(water_perm.lt(10)) # Moins de 10% d'occurrence d'eau
    
    return flood_clean.selfMask()

with st.spinner("Analyse satellite Sentinel-1 en cours..."):
    # On passe la géométrie en format dict pour le cache
    flood_img = detect_floods(geom.getInfo(), start_date, end_date)

# ------------------------------------------------------------
# INDICATEURS GLOBAUX (OPTIMISÉS)
# ------------------------------------------------------------
pixel_area = ee.Image.pixelArea()

# On regroupe les réductions pour éviter plusieurs .getInfo()
# Ajout d'un Try/Except pour gérer les timeouts ou erreurs de calcul
try:
    stats = ee.Dictionary({
        'flood_km2': flood_img.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e13
        ).get("VV"),
        'pop_exposed': ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01", "2020-12-31").mean()
            .updateMask(flood_img)
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e13)
            .get("population"),
        'rain_total': ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(geom).filterDate(str(start_date), str(end_date)).sum()
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=5000, maxPixels=1e13)
            .get("precipitation")
    }).getInfo()
except Exception as e:
    st.warning(f"⚠️ Impossible de calculer les statistiques globales (zone trop grande ou complexe). Erreur : {e}")
    stats = {}

flood_area_val = (stats.get('flood_km2') or 0) / 1e6
pop_val = int(stats.get('pop_exposed') or 0)
rain_val = stats.get('rain_total') or 0

st.subheader("📊 Indicateurs clés de la zone")
c1, c2, c3 = st.columns(3)
c1.metric("Surface inondée", f"{flood_area_val:.2f} km²")
c2.metric("Population impactée (est.)", f"{pop_val:,}")
c3.metric("Précipitations cumulées", f"{rain_val:.1f} mm")

# ------------------------------------------------------------
# CARTE INTERACTIVE
# ------------------------------------------------------------
st.subheader("🗺️ Cartographie de l'étendue des eaux")

bounds = gdf.total_bounds
m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], zoom_start=10, tiles="CartoDB positron")

# Overlay Inondations
flood_mapid = flood_img.getMapId({'min': 0, 'max': 1, 'palette': ['#0000FF']})
folium.TileLayer(
    tiles=flood_mapid['tile_fetcher'].url_format,
    attr='Google Earth Engine - Sentinel-1',
    name='Zones Inondées',
    overlay=True,
    control=True,
    opacity=0.7
).add_to(m)

# Ajout du GeoJSON original pour les popups
folium.GeoJson(
    gdf,
    name="Limites administratives",
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 2},
    tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:3]), labels=True)
).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=600)

# ------------------------------------------------------------
# SÉRIE TEMPORELLE (OPTIMISÉE)
# ------------------------------------------------------------
st.subheader("📈 Dynamique des précipitations (CHIRPS)")

@st.cache_data
def get_rain_series(aoi_serialized, start, end):
    try:
        aoi = ee.Geometry(aoi_serialized)
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(str(start), str(end))
        
        def extract_val(img):
            val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=5000).get('precipitation')
            return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'rain': val})

        fc = chirps.map(extract_val).getInfo()
        return pd.DataFrame([f['properties'] for f in fc['features']])
    except Exception:
        return pd.DataFrame()

with st.spinner("Génération de la courbe de pluie..."):
    df_rain = get_rain_series(geom.getInfo(), start_date, end_date)
    if not df_rain.empty:
        df_rain['date'] = pd.to_datetime(df_rain['date'])
        fig = px.bar(df_rain, x='date', y='rain', title="Précipitations quotidiennes (mm)", color_discrete_sequence=['#00aaff'])
        st.plotly_chart(fig, use_container_width=True)
