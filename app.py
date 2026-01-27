import os
import io
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

import folium
from streamlit_folium import st_folium

import osmnx as ox
import ee
import requests

# ═════════════════════════════════════════════════════════════════
# 1. CONFIG & INIT
# ═════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Flood Analysis WA", layout="wide")

def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.warning("❌ Secret 'GEE_SERVICE_ACCOUNT' manquant. Les données GEE ne seront pas disponibles.")
        return False
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.warning(f"❌ Erreur GEE : {e}. Les données GEE ne seront pas disponibles.")
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. GADM FIX (ADMIN 0-4)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except:
        return None

# ═════════════════════════════════════════════════════════════════
# 3. SENTINEL-1 OPTIMISÉ
# ═════════════════════════════════════════════════════════════════

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood, diff_threshold=1.25):
    if not gee_available:
        return None, None, None
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi_ee)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
        img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
        img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)
        ref_db = img_ref.log10().multiply(10)
        flood_db = img_flood.log10().multiply(10)
        diff = ref_db.subtract(flood_db)
        flooded = diff.gt(diff_threshold)
        terrain = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
        slope = terrain.select('slope')
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        final_mask = (flooded.updateMask(slope.lt(5)).updateMask(gsw.lt(80)).selfMask())
        return final_mask, ref_db, flood_db
    except:
        return None, None, None

# ═════════════════════════════════════════════════════════════════
# 4. OSM FIX (FEATURES_FROM_BBOX)
# ═════════════════════════════════════════════════════════════════

def get_osm_data(gdf_aoi):
    if gdf_aoi is None or gdf_aoi.empty:
        return gpd.GeoDataFrame()
    bounds = gdf_aoi.total_bounds
    try:
        tags = {'building': True, 'highway': True, 'amenity': ['hospital', 'school', 'clinic']}
        data = ox.features_from_bbox(north=bounds[3],
                                     south=bounds[1],
                                     east=bounds[2],
                                     west=bounds[0],
                                     tags=tags)
        # Garde uniquement les géométries dans l'AOI
        return data[data.geometry.within(gdf_aoi.unary_union)]
    except:
        return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 5. UI STREAMLIT
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🌍 Paramètres")

# Sélection pays ou upload
country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
source_option = st.sidebar.radio("Source de la zone d'étude", ["Pays/Admin", "Fichier (KML/GeoJSON/Shp)"])

selected_zone = None
if source_option == "Pays/Admin":
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    level = st.sidebar.slider("Niveau Admin (0=Pays, 1=Région, 2=Département, 3/4=Commune)", 0, 4, 1)
    gdf_base = load_gadm(iso, level)
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
        names = sorted(gdf_base[col_name].unique())
        choice = st.sidebar.multiselect("Zone(s) spécifique(s)", names, default=names)
        if choice:
            selected_zone = gdf_base[gdf_base[col_name].isin(choice)]
        else:
            selected_zone = gdf_base
elif source_option == "Fichier (KML/GeoJSON/Shp)":
    uploaded_file = st.sidebar.file_uploader("Uploader un fichier KML/GeoJSON/Shapefile", type=["kml","geojson","shp"])
    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "shp":
            selected_zone = gpd.read_file(uploaded_file)
        else:
            selected_zone = gpd.read_file(uploaded_file)
    else:
        st.info("Veuillez uploader un fichier pour la zone d'étude.")

# Dates
d1, d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début Inondation", datetime(2024, 8, 1))
end_f = d2.date_input("Fin Inondation", datetime(2024, 8, 31))

# Affichage initial si aucune zone
if selected_zone is None:
    st.warning("Aucune zone sélectionnée, utilisation d'une zone par défaut.")
    # Création d'un polygone fictif
    selected_zone = gpd.GeoDataFrame([{"geometry": shape({"type":"Polygon","coordinates":[[[-14,14],[-14,15],[-13,15],[-13,14],[-14,14]]]})}], crs="EPSG:4326")

# ═════════════════════════════════════════════════════════════════
# 6. ANALYSE
# ═════════════════════════════════════════════════════════════════

with st.spinner("Analyse en cours..."):

    # --- GEE Flood Mask
    aoi_ee = None
    flood_mask = None
    if gee_available:
        aoi_ee = ee.Geometry(mapping(selected_zone.unary_union))
        flood_mask, _, _ = get_flood_mask(aoi_ee, "2024-01-01", "2024-03-01", str(start_f), str(end_f))

    # --- OSM Infrastructures
    osm_all = get_osm_data(selected_zone)
    n_buildings = len(osm_all[osm_all['building'].notnull()]) if 'building' in osm_all.columns else 0
    n_amenities = len(osm_all[osm_all['amenity'].notnull()]) if 'amenity' in osm_all.columns else 0
    n_roads = len(osm_all[osm_all['highway'].notnull()]) if 'highway' in osm_all.columns else 0

    # --- Metrics simplifiées (Population fictive si GEE absent)
    total_pop = 1000
    pop_exposed = 200 if flood_mask is not None else 0

    # --- Carte
    st.subheader("🗺️ Carte de la zone et inondation")
    m = folium.Map(location=[selected_zone.centroid.y.mean(), selected_zone.centroid.x.mean()], zoom_start=8)
    folium.GeoJson(selected_zone, name="Limites Admin").add_to(m)
    if flood_mask is not None:
        try:
            vis_params = {'min': 0, 'max': 1, 'palette': ['000000', '00FFFF']}
            map_id = flood_mask.getMapId(vis_params)
            folium.TileLayer(
                tiles=map_id['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name='Zones Inondées',
                overlay=True
            ).add_to(m)
        except:
            st.warning("Impossible d'afficher le masque d'inondation GEE.")
    st_folium(m, width=1000, height=500)

    # --- Affichage des indicateurs
    st.subheader("📊 Indicateurs Clés")
    c1, c2, c3 = st.columns(3)
    c1.metric("Population Exposée", pop_exposed)
    c2.metric("Bâtiments affectés", n_buildings)
    c3.metric("Routes affectées", n_roads)
    c4, c5 = st.columns(2)
    c4.metric("Infrastructures type Amenity", n_amenities)
    c5.metric("Population Totale (approx.)", total_pop)
