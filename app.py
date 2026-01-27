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

# Configuration globale OSMnx pour éviter les blocages
ox.settings.timeout = 60  # Augmente le timeout à 60 secondes
ox.settings.use_cache = True

# ────────── INITIALISATION GEE ──────────
@st.cache_resource
def init_gee():
    try:
        # Cas 1 : Secrets Streamlit (Production)
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = st.secrets["GEE_SERVICE_ACCOUNT"]
            if isinstance(key_dict, str):
                key_dict = json.loads(key_dict)
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            return True
        
        # Cas 2 : Environnement local (Fallback)
        ee.Initialize()
        return True

    except Exception as e:
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS UTILITAIRES
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    """Charge GADM depuis le GPKG distant avec mise en cache pour éviter les lenteurs"""
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Erreur de connexion aux données GADM: {e}")
        return None

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood, threshold=5):
    """Détecte les inondations via Sentinel-1 (VV)"""
    if not gee_available:
        return None
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi_ee)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .select('VV'))

        # Calcul des médianes pour stabiliser le signal
        img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
        img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)
        
        # Lissage spatial
        img_ref = img_ref.focal_median(50, 'circle', 'meters')
        img_flood = img_flood.focal_median(50, 'circle', 'meters')

        # Calcul de la différence (Change Detection)
        diff = img_ref.subtract(img_flood)
        flooded = diff.gt(threshold).selfMask()
        return flooded
    except Exception as e:
        st.error(f"Erreur GEE (Inondation): {e}")
        return None

def get_population_exposure(aoi_ee, flood_mask):
    """Calcule la population exposée via WorldPop"""
    if not gee_available or flood_mask is None:
        return 0, 0
    try:
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01', '2021-01-01') \
                        .mosaic() \
                        .clip(aoi_ee)
        
        # Population Totale
        stats_total = pop_dataset.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e9
        )
        total_pop = stats_total.get('population').getInfo()
        
        # Population Exposée
        pop_exposed_img = pop_dataset.updateMask(flood_mask)
        stats_exposed = pop_exposed_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e9
        )
        exposed_pop = stats_exposed.get('population').getInfo()
        
        return int(total_pop or 0), int(exposed_pop or 0)
    except Exception as e:
        return 0, 0

@st.cache_data(show_spinner=False)
def get_osm_data(_gdf_aoi):
    """Récupère les données OSM pour la zone sélectionnée avec des protections contre le gel"""
    if _gdf_aoi is None or _gdf_aoi.empty:
        return gpd.GeoDataFrame()
    
    # Calcul de la surface pour limiter les abus
    area_sq_km = _gdf_aoi.to_crs(epsg=3857).area.sum() / 1e6
    if area_sq_km > 800: # Seuil réduit pour plus de sécurité
        st.warning(f"⚠️ Zone trop large ({area_sq_km:.0f} km²) pour l'acquisition OSM temps réel. Veuillez restreindre la zone.")
        return gpd.GeoDataFrame()

    try:
        # On simplifie la géométrie pour accélérer la requête Overpass
        poly = _gdf_aoi.unary_union
        if poly.geom_type != 'Polygon':
            poly = poly.convex_hull # On utilise l'enveloppe convexe si multi-polygone complexe

        tags = {
            'building': True, 
            'highway': True, 
            'amenity': ['hospital','school','clinic', 'doctors', 'pharmacy']
        }
        
        # Utilisation de geometries_from_polygon au lieu de features_from_bbox pour être plus précis
        data = ox.geometries_from_polygon(poly, tags=tags)
        
        if not data.empty:
            return data.clip(_gdf_aoi)
        return data
    except Exception as e:
        # En cas d'erreur ou timeout, on renvoie un dataframe vide pour ne pas bloquer l'UI
        return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. UI STREAMLIT
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🌍 Paramètres")

country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
source_option = st.sidebar.radio("Source de la zone", ["Pays/Admin", "Fichier (KML/GeoJSON/Shp)"])

selected_zone = None

if source_option == "Pays/Admin":
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    level = st.sidebar.slider("Niveau Administratif", 0, 3, 2)
    
    with st.spinner("Chargement des limites..."):
        gdf_base = load_gadm(iso, level)
    
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
        if col_name not in gdf_base.columns:
            col_name = gdf_base.select_dtypes(include='object').columns[0]
            
        names = sorted(gdf_base[col_name].astype(str).unique())
        choice = st.sidebar.multiselect("Zone(s) spécifique(s)", names)
        
        if choice:
            selected_zone = gdf_base[gdf_base[col_name].isin(choice)]
        else:
            selected_zone = gdf_base.iloc[[0]] 

elif source_option == "Fichier (KML/GeoJSON/Shp)":
    uploaded_file = st.sidebar.file_uploader("Importer un fichier", type=["kml","geojson","shp"])
    if uploaded_file:
        try:
            selected_zone = gpd.read_file(uploaded_file).to_crs(epsg=4326)
        except Exception as e:
            st.error(f"Erreur fichier: {e}")

# Dates
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Période d'Inondation")
d1, d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début", datetime(2024, 8, 1))
end_f = d2.date_input("Fin", datetime(2024, 9, 30))
flood_threshold = st.sidebar.slider("Sensibilité (dB)", 3.0, 10.0, 5.0, 0.5)

# Zone de secours si rien n'est sélectionné
if selected_zone is None or selected_zone.empty:
    selected_zone = gpd.GeoDataFrame([{"geometry": shape({"type":"Polygon","coordinates":[[[-17.5,14.6],[-17.5,14.8],[-17.3,14.8],[-17.3,14.6],[-17.5,14.6]]]})}], crs="EPSG:4326")

# ═════════════════════════════════════════════════════════════════
# 4. EXÉCUTION DE L'ANALYSE
# ═════════════════════════════════════════════════════════════════

st.title("🌊 Tableau de Bord - Analyse d'Impact Inondation")

if st.button("🚀 LANCER L'ANALYSE", type="primary"):
    
    col_map, col_stats = st.columns([2, 1])
    
    # 1. Préparation GEE
    with st.spinner("Traitement des données satellites..."):
        try:
            poly_geom = selected_zone.unary_union
            aoi_ee = ee.Geometry(mapping(poly_geom)) if gee_available else None
            
            # Calcul Inondation
            flood_mask = None
            if gee_available and aoi_ee:
                flood_mask = get_flood_mask(aoi_ee, "2023-01-01", "2023-05-01", str(start_f), str(end_f), flood_threshold)
            
            # Calcul Population
            total_pop, pop_exposed = 0, 0
            if gee_available and flood_mask:
                total_pop, pop_exposed = get_population_exposure(aoi_ee, flood_mask)
        except Exception as e:
            st.error(f"Erreur GEE : {e}")
            flood_mask, total_pop, pop_exposed = None, 0, 0

    # 2. Préparation OSM
    with st.spinner("Acquisition des infrastructures (OSM)..."):
        osm_all = get_osm_data(selected_zone)
        n_buildings, n_roads, n_amenities = 0, 0, 0
        if not osm_all.empty:
            if 'building' in osm_all.columns: n_buildings = osm_all['building'].count()
            if 'highway' in osm_all.columns: n_roads = osm_all['highway'].count()
            if 'amenity' in osm_all.columns: n_amenities = osm_all['amenity'].count()

    # --- AFFICHAGE DES RÉSULTATS ---
    with col_stats:
        st.subheader("📊 Résultats de l'Impact")
        st.metric("Population Exposée", f"{pop_exposed:,.0f} pers.")
        st.metric("Population Totale", f"{total_pop:,.0f} pers.")
        st.divider()
        st.markdown("##### Éléments OSM dans la zone")
        st.write(f"🏠 Bâtiments : **{n_buildings}**")
        st.write(f"🛣️ Segments routiers : **{n_roads}**")
        st.write(f"🏥 Services critiques : **{n_amenities}**")
        
        if not gee_available:
            st.warning("Mode dégradé : Earth Engine non disponible.")

    with col_map:
        st.subheader("🗺️ Cartographie")
        center = selected_zone.centroid.iloc[0]
        m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="OpenStreetMap")
        
        # Style de la zone
        folium.GeoJson(selected_zone, name="Zone d'étude", style_function=lambda x: {'fillColor': '#ffffff00', 'color': 'black'}).add_to(m)
        
        # Couche Inondation
        if flood_mask:
            try:
                map_id = flood_mask.getMapId({'palette': ['#00FFFF']})
                folium.TileLayer(
                    tiles=map_id['tile_fetcher'].url_format,
                    attr='Google Earth Engine',
                    name='Zones Inondées',
                    overlay=True,
                    opacity=0.7
                ).add_to(m)
            except:
                st.info("La couche d'inondation ne contient peut-être pas de pixels détectés.")

        # Infrastructures
        if not osm_all.empty and 'amenity' in osm_all.columns:
            # On prend le centroïde pour l'affichage des marqueurs
            points = osm_all[osm_all['amenity'].notna()].centroid
            for idx, pt in points.items():
                folium.CircleMarker([pt.y, pt.x], radius=3, color='red', fill=True, popup="Infra").add_to(m)

        st_folium(m, width=None, height=550)
else:
    st.info("Veuillez sélectionner une zone et cliquer sur 'Lancer l'Analyse'.")
