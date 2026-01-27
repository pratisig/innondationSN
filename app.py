import os
import io
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping

import folium
from streamlit_folium import st_folium

import osmnx as ox
import ee

# ═════════════════════════════════════════════════════════════════
# 1. CONFIG & INIT
# ═════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Flood Analysis WA", layout="wide")

# Configuration globale OSMnx
ox.settings.timeout = 60
ox.settings.use_cache = True

# Initialisation des variables d'état pour éviter la page blanche
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'flood_mask' not in st.session_state:
    st.session_state.flood_mask = None
if 'osm_data' not in st.session_state:
    st.session_state.osm_data = gpd.GeoDataFrame()
if 'stats' not in st.session_state:
    st.session_state.stats = {"pop_exposed": 0, "total_pop": 0, "n_buildings": 0, "n_roads": 0, "n_amenities": 0}

# ────────── INITIALISATION GEE ──────────
@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = st.secrets["GEE_SERVICE_ACCOUNT"]
            if isinstance(key_dict, str):
                key_dict = json.loads(key_dict)
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            return True
        ee.Initialize()
        return True
    except Exception:
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS UTILITAIRES
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Erreur GADM: {e}")
        return None

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood, threshold=5):
    if not gee_available:
        return None
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi_ee)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .select('VV'))

        img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
        img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)
        
        img_ref = img_ref.focal_median(50, 'circle', 'meters')
        img_flood = img_flood.focal_median(50, 'circle', 'meters')

        diff = img_ref.subtract(img_flood)
        return diff.gt(threshold).selfMask()
    except Exception:
        return None

def get_population_exposure(aoi_ee, flood_mask):
    if not gee_available or flood_mask is None:
        return 0, 0
    try:
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01', '2021-01-01') \
                        .mosaic().clip(aoi_ee)
        
        stats_total = pop_dataset.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
        total_pop = stats_total.get('population').getInfo()
        
        pop_exposed_img = pop_dataset.updateMask(flood_mask)
        stats_exposed = pop_exposed_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
        exposed_pop = stats_exposed.get('population').getInfo()
        
        return int(total_pop or 0), int(exposed_pop or 0)
    except Exception:
        return 0, 0

@st.cache_data(show_spinner=False)
def get_osm_data(_gdf_aoi):
    if _gdf_aoi is None or _gdf_aoi.empty:
        return gpd.GeoDataFrame()
    
    area_sq_km = _gdf_aoi.to_crs(epsg=3857).area.sum() / 1e6
    if area_sq_km > 800:
        return gpd.GeoDataFrame()

    try:
        poly = _gdf_aoi.unary_union
        if poly.geom_type != 'Polygon':
            poly = poly.convex_hull

        tags = {'building': True, 'highway': True, 'amenity': ['hospital','school','clinic', 'pharmacy']}
        data = ox.geometries_from_polygon(poly, tags=tags)
        
        if not data.empty:
            return data.clip(_gdf_aoi)
        return data
    except Exception:
        return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. UI SIDEBAR
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🌍 Paramètres")

country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
source_option = st.sidebar.radio("Source de la zone", ["Pays/Admin", "Fichier"])

selected_zone = None

if source_option == "Pays/Admin":
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    level = st.sidebar.slider("Niveau Administratif", 0, 3, 2)
    gdf_base = load_gadm(iso, level)
    
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
        names = sorted(gdf_base[col_name].astype(str).unique())
        choice = st.sidebar.multiselect("Zone(s)", names)
        selected_zone = gdf_base[gdf_base[col_name].isin(choice)] if choice else gdf_base.iloc[[0]] 

elif source_option == "Fichier":
    uploaded_file = st.sidebar.file_uploader("Importer KML/GeoJSON", type=["kml","geojson","shp"])
    if uploaded_file:
        selected_zone = gpd.read_file(uploaded_file).to_crs(epsg=4326)

st.sidebar.markdown("---")
d1, d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début", datetime(2024, 8, 1))
end_f = d2.date_input("Fin", datetime(2024, 9, 30))
flood_threshold = st.sidebar.slider("Sensibilité (dB)", 3.0, 10.0, 5.0, 0.5)

if selected_zone is None or selected_zone.empty:
    selected_zone = gpd.GeoDataFrame([{"geometry": shape({"type":"Polygon","coordinates":[[[-17.5,14.6],[-17.5,14.8],[-17.3,14.8],[-17.3,14.6],[-17.5,14.6]]]})}], crs="EPSG:4326")

# ═════════════════════════════════════════════════════════════════
# 4. LOGIQUE D'ANALYSE
# ═════════════════════════════════════════════════════════════════

st.title("🌊 Analyse d'Impact Inondation")

if st.button("🚀 LANCER L'ANALYSE", type="primary"):
    st.session_state.analysis_done = True
    
    with st.spinner("Analyse en cours..."):
        # GEE
        poly_geom = selected_zone.unary_union
        aoi_ee = ee.Geometry(mapping(poly_geom)) if gee_available else None
        
        st.session_state.flood_mask = get_flood_mask(aoi_ee, "2023-01-01", "2023-05-01", str(start_f), str(end_f), flood_threshold)
        
        t_pop, e_pop = get_population_exposure(aoi_ee, st.session_state.flood_mask)
        
        # OSM
        st.session_state.osm_data = get_osm_data(selected_zone)
        
        # Stats
        st.session_state.stats = {
            "pop_exposed": e_pop,
            "total_pop": t_pop,
            "n_buildings": st.session_state.osm_data.get('building', pd.Series()).count(),
            "n_roads": st.session_state.osm_data.get('highway', pd.Series()).count(),
            "n_amenities": st.session_state.osm_data.get('amenity', pd.Series()).count()
        }

# ═════════════════════════════════════════════════════════════════
# 5. AFFICHAGE DES RÉSULTATS (PERSISTANTS)
# ═════════════════════════════════════════════════════════════════

if st.session_state.analysis_done:
    col_map, col_stats = st.columns([2, 1])
    
    with col_stats:
        st.subheader("📊 Résultats")
        st.metric("Population Exposée", f"{st.session_state.stats['pop_exposed']:,} pers.")
        st.metric("Population Totale", f"{st.session_state.stats['total_pop']:,} pers.")
        st.divider()
        st.write(f"🏠 Bâtiments : **{st.session_state.stats['n_buildings']}**")
        st.write(f"🛣️ Routes : **{st.session_state.stats['n_roads']}**")
        st.write(f"🏥 Services : **{st.session_state.stats['n_amenities']}**")

    with col_map:
        center = selected_zone.centroid.iloc[0]
        m = folium.Map(location=[center.y, center.x], zoom_start=11)
        
        # Zone
        folium.GeoJson(selected_zone, name="Zone", style_function=lambda x: {'fillColor': '#00000000', 'color': 'black'}).add_to(m)
        
        # Inondation
        if st.session_state.flood_mask:
            try:
                map_id = st.session_state.flood_mask.getMapId({'palette': ['#00FFFF']})
                folium.TileLayer(
                    tiles=map_id['tile_fetcher'].url_format,
                    attr='GEE',
                    name='Inondation',
                    overlay=True,
                    opacity=0.6
                ).add_to(m)
            except: pass

        # Marqueurs Infra
        if not st.session_state.osm_data.empty and 'amenity' in st.session_state.osm_data.columns:
            infras = st.session_state.osm_data[st.session_state.osm_data['amenity'].notna()]
            for _, row in infras.iterrows():
                c = row.geometry.centroid
                folium.CircleMarker([c.y, c.x], radius=3, color='red', fill=True, popup=row['amenity']).add_to(m)

        st_folium(m, width="100%", height=500, key="flood_map")
else:
    st.info("Sélectionnez une zone et lancez l'analyse.")
