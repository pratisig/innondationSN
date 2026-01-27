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

# ────────── INITIALISATION GEE ──────────
def init_gee():
    try:
        # Cas 1 : Secrets Streamlit
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = st.secrets["GEE_SERVICE_ACCOUNT"]
            if isinstance(key_dict, str):
                key_dict = json.loads(key_dict)
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            return True
        
        # Cas 2 : Environnement local déjà authentifié (fallback)
        ee.Initialize()
        return True

    except Exception as e:
        # En mode démo publique sans secrets, on retourne False
        # st.warning(f"GEE non actif (Mode dégradé OSM uniquement) : {e}")
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS UTILITAIRES
# ═════════════════════════════════════════════════════════════════

def load_gadm(iso, level):
    """Charge GADM depuis le GPKG distant"""
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        # Lecture optimisée : on ne lit que les géométries nécessaires si possible, 
        # mais read_file charge tout par défaut. C'est acceptable pour un niveau admin.
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Erreur chargement GADM: {e}")
        return None

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood, threshold=5):
    """
    Détecte les inondations par différence (Change Detection) sur Sentinel-1.
    Ref (sec) - Flood (mouillé) > Seuil (dB)
    """
    if not gee_available:
        return None
    try:
        # Collection Sentinel-1
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi_ee)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .select('VV')) # On travaille sur la polarisation VV sensible à l'eau

        # Images médianes (pour réduire le speckle)
        # Période de référence (Sèche)
        img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
        # Période inondation (Humide) - On prend le MIN pour capturer le pire cas (eau = sombre)
        img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)
        
        # Lissage pour enlever le bruit "poivre et sel"
        smooth_radius = 50
        img_ref = img_ref.focal_median(smooth_radius, 'circle', 'meters')
        img_flood = img_flood.focal_median(smooth_radius, 'circle', 'meters')

        # Calcul de la différence
        # Si pixel sec (-10dB) devient inondé (-20dB) -> Diff = 10dB
        diff = img_ref.subtract(img_flood)
        
        # Masque : on garde ce qui a changé de plus de X dB
        flooded = diff.gt(threshold).selfMask()
        return flooded
    except Exception as e:
        st.warning(f"Erreur calcul masque inondation: {e}")
        return None

def get_population_exposure(aoi_ee, flood_mask):
    """Calcule la population sous le masque d'inondation (WorldPop)"""
    if not gee_available or flood_mask is None:
        return 0, 0
    try:
        # WorldPop Global Project Population Data (approx 2020)
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01', '2021-01-01') \
                        .mosaic() \
                        .clip(aoi_ee)
        
        # 1. Population Totale Zone
        stats_total = pop_dataset.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e9
        )
        total_pop = stats_total.get('population').getInfo()
        
        # 2. Population Exposée (Masquée par l'eau)
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
        st.warning(f"Erreur calcul population: {e}")
        return 0, 0

def get_osm_data(gdf_aoi):
    """Télécharge les données OSM pour la zone"""
    if gdf_aoi is None or gdf_aoi.empty:
        return gpd.GeoDataFrame()
    
    # Vérification de la taille pour éviter crash OSMnx
    area_sq_km = gdf_aoi.to_crs(epsg=3857).area.sum() / 1e6
    if area_sq_km > 500: # Limite arbitraire pour la démo
        st.warning(f"⚠️ Zone trop vaste ({area_sq_km:.0f} km²) pour l'analyse infra détaillée. Sélectionnez une commune ou un département plus petit.")
        return gpd.GeoDataFrame()

    bounds = gdf_aoi.total_bounds
    try:
        # Tags demandés
        tags = {
            'building': True, 
            'highway': True, 
            'amenity': ['hospital','school','clinic', 'doctors', 'pharmacy']
        }
        
        # Features from bbox est plus robuste que from_polygon pour les géométries complexes
        data = ox.features_from_bbox(
            bbox=(bounds[3], bounds[1], bounds[2], bounds[0]), # North, South, East, West
            tags=tags
        )
        
        # Clip strict à la zone
        if not data.empty:
            # On projette pour le clip propre
            return data.clip(gdf_aoi)
        return data

    except Exception as e:
        st.warning(f"Données OSM non disponibles ou erreur: {e}")
        return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. UI STREAMLIT
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🌍 Paramètres")

# Sélection Zone
country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
source_option = st.sidebar.radio("Source de la zone", ["Pays/Admin", "Fichier (KML/GeoJSON/Shp)"])

selected_zone = None

if source_option == "Pays/Admin":
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    # Default level 2 (Departement) pour éviter de charger tout le pays
    level = st.sidebar.slider("Niveau Admin (0=Pays, 1=Région, 2=Département, 3=Commune)", 0, 3, 2)
    
    with st.spinner("Chargement des limites administratives..."):
        gdf_base = load_gadm(iso, level)
    
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
        # Gestion si la colonne n'existe pas dans le GPKG (parfois VARNAME ou autre)
        if col_name not in gdf_base.columns:
            # Fallback sur la première colonne de texte
            col_name = gdf_base.select_dtypes(include='object').columns[0]
            
        names = sorted(gdf_base[col_name].astype(str).unique())
        choice = st.sidebar.multiselect("Zone(s) spécifique(s)", names)
        
        if choice:
            selected_zone = gdf_base[gdf_base[col_name].isin(choice)]
        else:
            selected_zone = gdf_base.head(1) # Par défaut on prend le 1er pour éviter une carte vide
            st.sidebar.info("💡 Sélectionnez une zone spécifique pour affiner l'analyse.")

elif source_option == "Fichier (KML/GeoJSON/Shp)":
    uploaded_file = st.sidebar.file_uploader("Uploader KML/GeoJSON/Shapefile", type=["kml","geojson","shp"])
    if uploaded_file:
        try:
            selected_zone = gpd.read_file(uploaded_file).to_crs(epsg=4326)
        except Exception as e:
            st.error(f"Erreur lecture fichier: {e}")
    else:
        st.info("Uploader un fichier pour la zone d'étude.")

# Dates & Paramètres Analyse
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Période & Sensibilité")
d1, d2 = st.sidebar.columns(2)
# Période "Avant" (Référence sèche)
start_ref = "2023-01-01"
end_ref = "2023-05-01"

# Période "Pendant" (Inondation)
start_f = d1.date_input("Début Inondation", datetime(2024, 8, 1))
end_f = d2.date_input("Fin Inondation", datetime(2024, 9, 30))
flood_threshold = st.sidebar.slider("Seuil Détection (dB)", 3.0, 10.0, 5.0, 0.5, help="Différence d'intensité requise. Plus bas = plus sensible (bruit), Plus haut = ne détecte que les grosses eaux.")

# Zone par défaut si vide (Dakar approx)
if selected_zone is None or selected_zone.empty:
    selected_zone = gpd.GeoDataFrame([{"geometry": shape({"type":"Polygon","coordinates":[[[-17.5,14.6],[-17.5,14.8],[-17.3,14.8],[-17.3,14.6],[-17.5,14.6]]]})}], crs="EPSG:4326")

# ═════════════════════════════════════════════════════════════════
# 4. ANALYSE & RENDU
# ═════════════════════════════════════════════════════════════════

st.title(f"Analyse d'Impact : {country if source_option == 'Pays/Admin' else 'Zone Importée'}")

if st.button("🚀 LANCER L'ANALYSE", type="primary"):
    
    col_map, col_stats = st.columns([2, 1])
    
    with st.spinner("🔄 Analyse satellite et infrastructure en cours..."):
        
        # 1. Conversion Géométrie pour GEE
        try:
            # Unary union pour gérer les multi-sélections
            poly_geom = selected_zone.unary_union
            # Mapping Shapely -> GeoJSON -> EE
            aoi_ee = ee.Geometry(mapping(poly_geom)) if gee_available else None
        except Exception as e:
            st.error(f"Erreur géométrie: {e}")
            aoi_ee = None

        # 2. Calcul Masque Inondation
        flood_mask = None
        if gee_available and aoi_ee:
            flood_mask = get_flood_mask(aoi_ee, start_ref, end_ref, str(start_f), str(end_f), flood_threshold)
        
        # 3. Calcul Population (WorldPop)
        total_pop, pop_exposed = 0, 0
        if gee_available and flood_mask:
            total_pop, pop_exposed = get_population_exposure(aoi_ee, flood_mask)
        
        # 4. Chargement OSM
        osm_all = get_osm_data(selected_zone)
        
        # Calcul Indicateurs OSM (Sécurisé)
        n_buildings = 0
        n_roads = 0
        n_amenities = 0
        
        if not osm_all.empty:
            if 'building' in osm_all.columns:
                n_buildings = osm_all['building'].notna().sum()
            if 'highway' in osm_all.columns:
                n_roads = osm_all['highway'].notna().sum()
            if 'amenity' in osm_all.columns:
                n_amenities = osm_all['amenity'].notna().sum()

    # --- AFFICHAGE ---

    with col_stats:
        st.subheader("📊 Bilan Rapide")
        
        st.metric("Population Estimée Exposée", f"{pop_exposed:,.0f}", help="Basé sur WorldPop 2020 croisé avec zone inondée")
        st.metric("Population Totale Zone", f"{total_pop:,.0f}")
        
        st.divider()
        st.markdown("#### Infrastructures (OSM)")
        c1, c2 = st.columns(2)
        c1.metric("Bâtiments", f"{n_buildings:,}")
        c2.metric("Routes (segments)", f"{n_roads:,}")
        st.metric("Services (Santé/École)", f"{n_amenities:,}")
        
        if not gee_available:
            st.warning("⚠️ Moteur Google Earth Engine non connecté. Les données satellite (Inondation/Population) ne sont pas disponibles.")

    with col_map:
        st.subheader("🗺️ Carte de situation")
        
        # Centrage carte
        centroid = selected_zone.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="CartoDB positron")
        
        # Couche Zone d'étude
        folium.GeoJson(
            selected_zone, 
            name="Zone d'étude",
            style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 2}
        ).add_to(m)
        
        # Couche OSM (Points critiques)
        if not osm_all.empty and 'amenity' in osm_all.columns:
            critique = osm_all[osm_all['amenity'].notna()]
            # On convertit en points si ce sont des polygones pour l'affichage simple
            critique_points = critique.copy()
            critique_points['geometry'] = critique_points.centroid
            
            folium.GeoJson(
                critique_points,
                name="Infrastructures Critiques",
                marker=folium.CircleMarker(radius=4, color='red', fill=True, fill_color='red'),
                tooltip=folium.GeoJsonTooltip(fields=['amenity', 'name'] if 'name' in osm_all.columns else ['amenity'])
            ).add_to(m)
        
        # Couche Inondation GEE
        if flood_mask:
            try:
                vis_params = {'palette': ['00FFFF']} # Cyan pour l'eau
                map_id = flood_mask.getMapId(vis_params)
                
                folium.TileLayer(
                    tiles=map_id['tile_fetcher'].url_format,
                    attr='Google Earth Engine & Copernicus',
                    name='Zones Inondées détectées',
                    overlay=True,
                    control=True,
                    opacity=0.7
                ).add_to(m)
            except Exception as e:
                st.warning(f"Erreur affichage couche inondation: {e}")

        st_folium(m, width=None, height=600)

else:
    st.info("👈 Configurez la zone et les dates à gauche, puis cliquez sur 'LANCER L'ANALYSE'.")
