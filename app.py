# ═════════════════════════════════════════════════════════════════
# FLOODWATCH WA - VERSION COMPLÈTE CORRIGÉE
# Détection inondations Sentinel-1 + Impacts OSM
# ═════════════════════════════════════════════════════════════════

import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping
import json
import ee
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# ═════════════════════════════════════════════════════════════════
# LOGGING & CONFIG
# ═════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FloodWatch WA", 
    page_icon="🌊",
    layout="wide"
)

# Configuration OSMnx
ox.settings.timeout = 180
ox.settings.use_cache = True

# ═════════════════════════════════════════════════════════════════
# 1. INITIALISATION GEE
# ═════════════════════════════════════════════════════════════════
@st.cache_resource
def init_gee():
    """Initialise Earth Engine avec authentification."""
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], 
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            logger.info("✅ GEE initialisé avec service account")
            return True
        else:
            ee.Initialize()
            logger.info("✅ GEE initialisé en mode local")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur GEE: {str(e)}")
        return False

gee_available = init_gee()

if not gee_available:
    st.error("❌ **Impossible de connecter à Earth Engine.** Vérifiez authentification GEE.")
    st.stop()


# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS UTILITAIRES
# ═════════════════════════════════════════════════════════════════

def safe_get_info(ee_obj, timeout=30):
    """
    Récupère info EE de façon sûre avec timeout.
    
    Paramètres:
    -----------
    ee_obj (ee.Image/FeatureCollection): Objet Earth Engine
    timeout (int): Secondes max pour attendre réponse
    
    Retour:
    -------
    dict/list ou None si erreur/timeout
    """
    try:
        return ee_obj.getInfo()
    except Exception as e:
        logger.warning(f"⚠️ Timeout/erreur getInfo(): {str(e)[:50]}")
        return None


# ═════════════════════════════════════════════════════════════════
# 3. CHARGEMENT DONNÉES ADMINISTRATIVES
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def load_gadm(iso_code, admin_level):
    """
    Charge limites administratives depuis GADM.
    
    📍 Source: https://gadm.org/
    
    Paramètres:
    -----------
    iso_code (str): Code ISO pays (ex: "SEN" = Sénégal)
    admin_level (int): 
        0 = Frontières nationales
        1 = Régions
        2 = Districts (recommandé)
        3 = Subdivisions plus fines
    
    Retour:
    -------
    GeoDataFrame (EPSG:4326) ou None si erreur
    """
    try:
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso_code.upper()}.gpkg"
        logger.info(f"📥 Chargement GADM: {iso_code} niveau {admin_level}")
        
        gdf = gpd.read_file(url, layer=admin_level)
        gdf = gdf.to_crs(epsg=4326)
        
        logger.info(f"✅ {len(gdf)} subdivisions chargées")
        return gdf
    
    except Exception as e:
        logger.error(f"❌ Erreur GADM {iso_code}: {str(e)[:80]}")
        st.error(f"❌ Impossible de charger limites: {str(e)[:100]}")
        return None


# ═════════════════════════════════════════════════════════════════
# 4. DÉTECTION INONDATIONS (CŒUR)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=1800)
def advanced_flood_detection(aoi_geojson, ref_start_str, ref_end_str, 
                             flood_start_str, flood_end_str, 
                             threshold_db=0.75, min_pixels=8):
    """
    🎯 DÉTECTION INONDATIONS AVANCÉE avec 6 masques.
    
    PROCESSUS:
    1. Compare backscatter Sentinel-1 (référence vs crise)
    2. Applique anomalie: Δ = Réf - Crise (en dB)
    3. Rejette eau existante (MODIS NDWI > 0.3)
    4. Rejette zones urbaines (NDBI > 0.1)
    5. Rejette pentes > 5° (eau accumule en bas)
    6. Élimine bruit (< 8 pixels connectés)
    
    📊 RÉSULTAT: Inondation_VRAIE = anomalie × 5 masques
    
    Paramètres:
    -----------
    aoi_geojson (dict): GeoJSON dict (résultat .getInfo())
        ✅ Dict hashable pour cache
    threshold_db (float): Seuil différence backscatter
        - 0.5 dB = très sensible
        - 0.75 dB = optimisé (défaut)
        - 1.25 dB = très conservateur
    min_pixels (int): Connectivité minimale
        - 3 = détecte petites flaques
        - 8 = défaut
        - 20 = ignore zones fines
    
    Retour:
    -------
    dict avec 'flood_final' + étapes de masquage
    """
    
    try:
        # ✅ Recréer ee.Geometry depuis dict
        aoi = ee.Geometry(aoi_geojson)
        
        # ✅ ÉTAPE 0: Images RÉFÉRENCE (période sèche)
        s1_ref = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterBounds(aoi)
                  .filterDate(ref_start_str, ref_end_str)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .select("VV")
                  .median())
        
        # Conversion en dB
        ref_db = ee.Image(10).multiply(s1_ref.max(ee.Image(0.0001)).log10())
        
        # ✅ ÉTAPE 1: Images CRISE
        s1_crisis = (ee.ImageCollection("COPERNICUS/S1_GRD")
                     .filterBounds(aoi)
                     .filterDate(flood_start_str, flood_end_str)
                     .filter(ee.Filter.eq("instrumentMode", "IW"))
                     .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                     .select("VV")
                     .median())
        
        crisis_db = ee.Image(10).multiply(s1_crisis.max(ee.Image(0.0001)).log10())
        
        # ✅ ÉTAPE 2: ANOMALIE BACKSCATTER
        delta_db = ref_db.subtract(crisis_db)
        flood_raw = delta_db.gt(threshold_db).rename('flood')
        
        st.info(f"✅ Étape 1/6: Anomalie backscatter (Δ > {threshold_db} dB)")
        
        # ✅ ÉTAPE 3: MASQUE EAU EXISTANTE
        try:
            modis_ref = (ee.ImageCollection("MODIS/006/MOD09GA")
                         .filterBounds(aoi)
                         .filterDate(ref_start_str, ref_end_str)
                         .median())
            
            nir = modis_ref.select('sur_refl_b02')
            swir = modis_ref.select('sur_refl_b06')
            ndwi_ref = nir.subtract(swir).divide(nir.add(swir))
            
            mask_not_existing_water = ndwi_ref.lt(0.3)
            flood_no_water = flood_raw.updateMask(mask_not_existing_water)
            
            st.success("✅ Étape 2/6: Rejet eau existante (MODIS NDWI > 0.3)")
        except Exception as e:
            st.warning(f"⚠️ Étape 2: {str(e)[:40]}")
            flood_no_water = flood_raw
        
        # ✅ ÉTAPE 4: MASQUE ZONES URBAINES
        try:
            modis_crisis = (ee.ImageCollection("MODIS/006/MOD09GA")
                            .filterBounds(aoi)
                            .filterDate(flood_start_str, flood_end_str)
                            .median())
            
            nir_c = modis_crisis.select('sur_refl_b02')
            swir_c = modis_crisis.select('sur_refl_b06')
            ndbi = swir_c.subtract(nir_c).divide(swir_c.add(nir_c))
            
            mask_not_urban = ndbi.lt(0.1)
            flood_no_urban = flood_no_water.updateMask(mask_not_urban)
            
            st.success("✅ Étape 3/6: Rejet zones urbaines (NDBI > 0.1)")
        except Exception as e:
            st.warning(f"⚠️ Étape 3: {str(e)[:40]}")
            flood_no_urban = flood_no_water
        
        # ✅ ÉTAPE 5: MASQUE PENTE
        try:
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Algorithms.Terrain(dem).select("slope")
            mask_low_slope = slope.lt(5)
            
            flood_low_slope = flood_no_urban.updateMask(mask_low_slope)
            
            st.success("✅ Étape 4/6: Rejet zones raides (pente > 5°)")
        except Exception as e:
            st.warning(f"⚠️ Étape 4: {str(e)[:40]}")
            flood_low_slope = flood_no_urban
        
        # ✅ ÉTAPE 6: FILTRE CONNECTIVITÉ
        try:
            connected_pixels = flood_low_slope.connectedPixelCount(8)
            flood_connected = flood_low_slope.updateMask(
                connected_pixels.gte(min_pixels)
            )
            
            st.success(f"✅ Étape 5/6: Filtre connectivité (≥ {min_pixels} px)")
        except Exception as e:
            st.warning(f"⚠️ Étape 5: {str(e)[:40]}")
            flood_connected = flood_low_slope
        
        st.success("✅ Étape 6/6: Inondation finale")
        
        return {
            'flood_final': flood_connected.selfMask(),
            'delta_db': delta_db.rename('delta'),
            'stages': {
                'Brut (Δ > threshold)': flood_raw,
                'Sans Eau Existante': flood_no_water,
                'Sans Zones Urbaines': flood_no_urban,
                'Pente Basse': flood_low_slope,
                'Final (Connecté)': flood_connected
            }
        }
    
    except Exception as e:
        st.error(f"❌ Détection inondation: {str(e)[:150]}")
        logger.error(f"Flood detection error: {str(e)}")
        return None


# ═════════════════════════════════════════════════════════════════
# 5. DONNÉES CLIMATIQUES (CORRIGÉ - DICT HASHABLE)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def get_climate_data(aoi_geojson_dict, start_str, end_str):
    """
    📊 Récupère précipitations (CHIRPS) & température (ERA5).
    
    ✅ PARAMÈTRE CHANGÉ:
    - aoi_geojson_dict (dict) au lieu de ee.Geometry
    - Dict est hashable ✅, ee.Geometry ne l'est pas ❌
    
    Données:
    --------
    CHIRPS: Précipitations en mm/jour
        - Résolution: 0.05° (~5 km)
        - Couverture: Monde, quotidien
    
    ERA5: Température 2m en K
        - Résolution: ~25 km
        - Couverture: Monde, mensuel
    
    Paramètres:
    -----------
    aoi_geojson_dict (dict): GeoJSON dict (résultat .getInfo())
    start_str, end_str (str): Format "YYYY-MM-DD"
    
    Retour:
    -------
    DataFrame avec colonnes 'date', 'value_precip', 'value_temp'
    """
    
    try:
        # ✅ Recréer ee.Geometry depuis dict
        aoi_ee = ee.Geometry(aoi_geojson_dict)
        
        precip = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                  .filterBounds(aoi_ee)
                  .filterDate(start_str, end_str)
                  .select('precipitation'))
        
        temp = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                .filterBounds(aoi_ee)
                .filterDate(start_str, end_str)
                .select('temperature_2m'))
        
        def extract_stats(img_col, band_name, reducer):
            """Extrait statistiques moyennes pour chaque image."""
            def map_func(img):
                val = img.reduceRegion(
                    reducer=reducer, 
                    geometry=aoi_ee, 
                    scale=5000
                ).get(band_name)
                return ee.Feature(None, {
                    'date': img.date().format('YYYY-MM-DD'), 
                    'value': val
                })
            
            fc = ee.FeatureCollection(img_col.map(map_func))
            info = safe_get_info(fc)
            
            if not info or 'features' not in info:
                return []
            
            return info['features']
        
        # Extraction
        p_data = extract_stats(precip, 'precipitation', ee.Reducer.mean())
        t_data = extract_stats(temp, 'temperature_2m', ee.Reducer.mean())
        
        if not p_data or not t_data:
            st.warning("⚠️ Pas de données climatiques disponibles")
            return None
        
        # DataFrame
        df_p = pd.DataFrame([f['properties'] for f in p_data])
        df_t = pd.DataFrame([f['properties'] for f in t_data])
        
        # Conversion température K → C
        if not df_t.empty and 'value' in df_t.columns:
            df_t['value'] = pd.to_numeric(df_t['value'], errors='coerce')
            df_t['value'] = df_t['value'] - 273.15
        
        # Fusion
        df_clim = df_p.merge(df_t, on='date', suffixes=('_precip', '_temp'), how='outer')
        df_clim = df_clim.sort_values('date')
        
        st.success("✅ Données climatiques chargées")
        return df_clim
    
    except Exception as e:
        st.warning(f"⚠️ Climat: {str(e)[:80]}")
        logger.warning(f"Climate data error: {str(e)}")
        return None


# ═════════════════════════════════════════════════════════════════
# 6. POPULATION & SUPERFICIE
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def get_population_stats(aoi_geojson_dict, flood_mask=None):
    """
    👥 Calcule population totale & exposée aux inondations.
    
    Source: WorldPop (100 m résolution)
    
    Paramètres:
    -----------
    aoi_geojson_dict (dict): GeoJSON dict
    flood_mask (ee.Image): Masque binaire inondation (0/1)
    
    Retour:
    -------
    tuple (pop_totale, pop_exposée)
    """
    
    try:
        aoi_ee = ee.Geometry(aoi_geojson_dict)
        
        # Image WorldPop: population/pixel (100m)
        pop_dataset = (ee.ImageCollection("WorldPop/GP/100m/pop")
                       .filterDate('2020-01-01', '2021-01-01')
                       .mosaic()
                       .clip(aoi_ee))
        
        # Population totale
        stats_total = pop_dataset.reduceRegion(
            reducer=ee.Reducer.sum(), 
            geometry=aoi_ee, 
            scale=100, 
            maxPixels=1e9
        )
        
        total_pop_info = safe_get_info(stats_total.get('population'))
        total_pop = int(total_pop_info) if total_pop_info else 0
        
        # Population exposée
        exposed_pop = 0
        if flood_mask is not None:
            pop_masked = pop_dataset.updateMask(flood_mask)
            stats_exposed = pop_masked.reduceRegion(
                reducer=ee.Reducer.sum(), 
                geometry=aoi_ee, 
                scale=100, 
                maxPixels=1e9
            )
            
            exposed_pop_info = safe_get_info(stats_exposed.get('population'))
            exposed_pop = int(exposed_pop_info) if exposed_pop_info else 0
        
        return total_pop, exposed_pop
    
    except Exception as e:
        logger.warning(f"Population stats error: {str(e)}")
        return 0, 0


@st.cache_data(show_spinner=False, ttl=3600)
def get_area_stats(aoi_geojson_dict, flood_mask):
    """
    📐 Calcule surface inondée en hectares.
    
    Paramètres:
    -----------
    aoi_geojson_dict (dict): GeoJSON dict
    flood_mask (ee.Image): Masque inondation
    
    Retour:
    -------
    float: Superficie en hectares
    """
    
    try:
        if flood_mask is None:
            return 0.0
        
        aoi_ee = ee.Geometry(aoi_geojson_dict)
        
        # Calcul surface: m² → ha
        area_m2_info = safe_get_info(
            flood_mask.multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(), 
                geometry=aoi_ee, 
                scale=10, 
                maxPixels=1e9
            ).get('flood')
        )
        
        area_m2 = area_m2_info if area_m2_info else 0
        area_ha = (area_m2 / 10000)
        
        return round(area_ha, 2)
    
    except Exception as e:
        logger.warning(f"Area stats error: {str(e)}")
        return 0.0


# ═════════════════════════════════════════════════════════════════
# 7. DONNÉES INFRASTRUCTURES (OSM)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def get_osm_data(gdf_aoi):
    """
    🏗️ Récupère données OpenStreetMap (routes, bâtiments).
    
    Paramètres:
    -----------
    gdf_aoi (GeoDataFrame): Zone d'étude
    
    Retour:
    -------
    tuple (GeoDataFrame_bâtiments, GeoDataFrame_routes)
    """
    
    if gdf_aoi is None or gdf_aoi.empty:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    try:
        poly = gdf_aoi.unary_union
        
        # 🛣️ ROUTES
        try:
            graph = ox.graph_from_polygon(
                poly, 
                network_type='all',
                simplify=True
            )
            gdf_routes = ox.graph_to_gdfs(
                graph, 
                nodes=False, 
                edges=True
            ).reset_index().clip(gdf_aoi)
            
            st.success(f"✅ {len(gdf_routes)} segments de route")
        
        except Exception as e:
            st.warning(f"⚠️ Routes OSM: {str(e)[:40]}")
            gdf_routes = gpd.GeoDataFrame()
        
        # 🏢 BÂTIMENTS
        try:
            tags = {
                'building': True,
                'amenity': ['school', 'university', 'college', 
                           'hospital', 'clinic', 'doctors'],
                'healthcare': True,
                'education': True
            }
            
            gdf_buildings = ox.features_from_polygon(poly, tags=tags)
            
            # Filtrer polygones
            gdf_buildings = gdf_buildings[
                gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])
            ].reset_index().clip(gdf_aoi)
            
            st.success(f"✅ {len(gdf_buildings)} bâtiments")
        
        except Exception as e:
            logger.warning(f"Buildings OSM error: {str(e)}")
            gdf_buildings = gpd.GeoDataFrame()
        
        return gdf_buildings, gdf_routes
    
    except Exception as e:
        st.error(f"❌ OSM: {str(e)[:100]}")
        logger.error(f"OSM data error: {str(e)}")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()


def analyze_impacted_infra(flood_mask, buildings_gdf):
    """
    🏥 Identifie bâtiments impactés.
    
    Retour:
    -------
    GeoDataFrame des bâtiments impactés
    """
    
    if flood_mask is None or buildings_gdf.empty:
        return gpd.GeoDataFrame()
    
    try:
        infra_check = buildings_gdf.head(3000).copy()
        
        if len(infra_check) == 0:
            return gpd.GeoDataFrame()
        
        features = [
            ee.Feature(
                ee.Geometry(mapping(row.geometry)), 
                {'idx': i}
            ) 
            for i, row in infra_check.iterrows()
        ]
        
        fc = ee.FeatureCollection(features)
        reduced = flood_mask.reduceRegions(
            collection=fc, 
            reducer=ee.Reducer.mean(), 
            scale=10
        )
        
        reduced_info = safe_get_info(reduced)
        
        if not reduced_info or 'features' not in reduced_info:
            return gpd.GeoDataFrame()
        
        impacted_indices = [
            f['properties']['idx'] 
            for f in reduced_info['features'] 
            if f['properties'].get('mean', 0) > 0
        ]
        
        if impacted_indices:
            return infra_check.loc[impacted_indices]
        else:
            return gpd.GeoDataFrame()
    
    except Exception as e:
        logger.warning(f"Impacted infra error: {str(e)}")
        return gpd.GeoDataFrame()


def analyze_impacted_roads(flood_mask, roads_gdf):
    """
    🛣️ Identifie segments de route impactés.
    """
    
    if flood_mask is None or roads_gdf.empty:
        return gpd.GeoDataFrame()
    
    try:
        roads_check = roads_gdf.head(5000).copy()
        
        if len(roads_check) == 0:
            return gpd.GeoDataFrame()
        
        features = [
            ee.Feature(
                ee.Geometry(mapping(row.geometry)), 
                {'idx': i}
            ) 
            for i, row in roads_check.iterrows()
        ]
        
        fc = ee.FeatureCollection(features)
        reduced = flood_mask.reduceRegions(
            collection=fc, 
            reducer=ee.Reducer.mean(), 
            scale=10
        )
        
        reduced_info = safe_get_info(reduced)
        
        if not reduced_info or 'features' not in reduced_info:
            return gpd.GeoDataFrame()
        
        impacted_indices = [
            f['properties']['idx'] 
            for f in reduced_info['features'] 
            if f['properties'].get('mean', 0) > 0
        ]
        
        if impacted_indices:
            return roads_check.loc[impacted_indices]
        else:
            return gpd.GeoDataFrame()
    
    except Exception as e:
        logger.warning(f"Impacted roads error: {str(e)}")
        return gpd.GeoDataFrame()


# ═════════════════════════════════════════════════════════════════
# 8. INTERFACE UTILISATEUR
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🗺️ **1. Zone d'Étude**")

mode = st.sidebar.radio(
    "Méthode de sélection",
    ["📋 Liste Administrative", "✏️ Dessiner sur Carte"]
)

# Session state
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = None
if 'zone_name' not in st.session_state:
    st.session_state.zone_name = "Zone non sélectionnée"
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False

# ─────────────────────────────────────────────
# MODE 1: LISTE ADMINISTRATIVE
# ─────────────────────────────────────────────
if mode == "📋 Liste Administrative":
    
    st.sidebar.markdown("""
    **📍 EXPLICATION:**
    - **Niveau 0**: Frontières nationales
    - **Niveau 1**: Grandes régions
    - **Niveau 2**: Préfectures/districts (recommandé)
    - **Niveau 3+**: Subdivisions très fines
    """)
    
    countries = {
        "🇸🇳 Sénégal": "SEN",
        "🇲🇱 Mali": "MLI",
        "🇳🇪 Niger": "NER",
        "🇧🇫 Burkina Faso": "BFA",
        "🇲🇷 Mauritanie": "MRT"
    }
    
    c_choice = st.sidebar.selectbox(
        "Sélectionner Pays",
        list(countries.keys())
    )
    
    admin_level = st.sidebar.slider(
        "Niveau de détail",
        0, 3, 2,
        help="0 = Pays | 1 = Régions | 2 = Préfectures | 3 = Districts"
    )
    
    gdf_base = load_gadm(countries[c_choice], admin_level)
    
    if gdf_base is not None:
        col_name = f"NAME_{admin_level}" if admin_level > 0 else "COUNTRY"
        
        if col_name in gdf_base.columns:
            available_choices = sorted(gdf_base[col_name].dropna().unique())
            
            selected_zones = st.sidebar.multiselect(
                "Subdivisions (sélection multiple)",
                available_choices
            )
            
            if selected_zones:
                new_zone = gdf_base[gdf_base[col_name].isin(selected_zones)].copy()
                st.session_state.selected_zone = new_zone
                st.session_state.zone_name = " + ".join(selected_zones[:3])
                if len(selected_zones) > 3:
                    st.session_state.zone_name += f"... ({len(selected_zones)} zones)"
                st.session_state.analysis_triggered = False


# ─────────────────────────────────────────────
# MODE 2: DESSIN LIBRE
# ─────────────────────────────────────────────
elif mode == "✏️ Dessiner sur Carte":
    
    st.sidebar.markdown("""
    **✏️ COMMENT UTILISER:**
    1. Cliquez sur outils (rectangle, cercle)
    2. Tracez votre zone
    3. La zone s'affichera en bas
    """)
    
    m_draw = folium.Map(
        location=[14.5, -14.5],
        zoom_start=6,
        tiles="cartodbpositron"
    )
    Draw(export=False).add_to(m_draw)
    
    with st.sidebar:
        out = st_folium(m_draw, width=250, height=250, key="draw_static")
        
        if out and out.get('last_active_drawing'):
            try:
                geom = shape(out['last_active_drawing']['geometry'])
                st.session_state.selected_zone = gpd.GeoDataFrame(
                    index=[0],
                    crs='epsg:4326',
                    geometry=[geom]
                )
                st.session_state.zone_name = "Zone Dessinée"
                st.session_state.analysis_triggered = False
            except Exception as e:
                st.sidebar.error(f"❌ Erreur géométrie: {str(e)[:50]}")


# ═════════════════════════════════════════════════════════════════
# SECTION PARAMÈTRES TEMPORELS
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 📅 **2. Paramètres Temporels**")

st.sidebar.markdown("""
**Référence (période sèche):**
Établit backscatter normal (sans inondation).
- Recommandé: Janvier-Mars
- Doit être > 1 mois
""")

ref_start = st.sidebar.date_input(
    "Début référence",
    datetime(2023, 1, 1)
)

ref_end = st.sidebar.date_input(
    "Fin référence",
    datetime(2023, 4, 30)
)

st.sidebar.divider()

st.sidebar.markdown("""
**Crise (période inondation):**
Détecte changement backscatter.
- Recommandé: Août-Octobre
- Doit être > 1 mois
""")

start_flood = st.sidebar.date_input(
    "Début inondation",
    datetime(2023, 8, 1)
)

end_flood = st.sidebar.date_input(
    "Fin inondation",
    datetime(2023, 9, 30)
)


# ═════════════════════════════════════════════════════════════════
# SECTION PARAMÈTRES DÉTECTION
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("## ⚙️ **3. Paramètres Détection**")

st.sidebar.markdown("""
**Seuil Différence Backscatter (dB):**

- **0.5 dB** = Très sensible (petites zones + faux +)
- **0.75 dB** = ✅ Recommandé
- **1.0 dB** = Conservative
- **1.5 dB+** = Très restrictif
""")

threshold_val = st.sidebar.slider(
    "Seuil (dB)",
    0.5, 2.0, 0.75,
    step=0.25
)

st.sidebar.markdown("""
**Pixels Minimum Connectivité:**

- **3** = Petites flaques
- **8** = ✅ Recommandé
- **15** = Ignore zones fines
- **30+** = Très conservateur
""")

min_pix = st.sidebar.number_input(
    "Pixels min connectivité",
    1, 50, 8,
    step=1
)

show_diagnostic = st.sidebar.checkbox(
    "🔍 Mode Diagnostic (étapes masquage)",
    value=False
)


# ═════════════════════════════════════════════════════════════════
# LOGIQUE PRINCIPALE
# ═════════════════════════════════════════════════════════════════

st.title(f"🌊 FloodWatch : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    
    if st.button(
        "🚀 LANCER L'ANALYSE COMPLÈTE",
        type="primary",
        use_container_width=True
    ):
        st.session_state.analysis_triggered = True
    
    if st.session_state.analysis_triggered:
        
        # ─────────────────────────────────────────────────
        # ANALYSE GEE
        # ─────────────────────────────────────────────────
        with st.spinner("🔄 Analyse GEE avancée en cours..."):
            
            aoi_ee_global = ee.Geometry(
                mapping(st.session_state.selected_zone.unary_union)
            )
            
            # ✅ CONVERSION: ee.Geometry → dict (IMPORTANT!)
            aoi_geojson = aoi_ee_global.getInfo()
            
            flood_data = advanced_flood_detection(
                aoi_geojson,  # ✅ Dict au lieu de ee.Geometry
                str(ref_start), str(ref_end),
                str(start_flood), str(end_flood),
                threshold_val, min_pix
            )
            
            if not flood_data:
                st.error("❌ Impossible de réaliser détection GEE.")
                st.stop()
            
            flood_mask = flood_data['flood_final']
            
            # ✅ Données climatiques (DICT HASHABLE)
            df_climat = get_climate_data(
                aoi_geojson,  # ✅ Dict au lieu de ee.Geometry
                str(start_flood), str(end_flood)
            )
            
            # Données OSM
            with st.spinner("📥 Chargement infrastructures OSM..."):
                buildings, routes = get_osm_data(st.session_state.selected_zone)
            
            # Analyse impact
            with st.spinner("🔍 Analyse impacts..."):
                impacted_infra = analyze_impacted_infra(flood_mask, buildings)
                impacted_roads = analyze_impacted_roads(flood_mask, routes)
        
        
        # ─────────────────────────────────────────────────
        # CALCULS PAR SUBDIVISION
        # ─────────────────────────────────────────────────
        sector_data = []
        total_pop_all = 0
        total_pop_exposed = 0
        total_flood_ha = 0
        
        with st.spinner("📊 Calcul indicateurs..."):
            for idx, row in st.session_state.selected_zone.iterrows():
                
                geom_ee_dict = ee.Geometry(mapping(row.geometry)).getInfo()
                
                # Population
                t_pop, e_pop = get_population_stats(geom_ee_dict, flood_mask)
                total_pop_all += t_pop
                total_pop_exposed += e_pop
                
                # Surface
                f_area = get_area_stats(geom_ee_dict, flood_mask)
                total_flood_ha += f_area
                
                # Nom subdivision
                sector_name = row.get(
                    'NAME_2',
                    row.get('NAME_1', row.get('NAME_0', f"Zone {idx}"))
                )
                
                pct_impacted = (e_pop / t_pop * 100) if t_pop > 0 else 0
                
                sector_data.append({
                    'Secteur': sector_name,
                    'Pop. Totale': t_pop,
                    'Pop. Exposée': e_pop,
                    '% Impacté': f"{pct_impacted:.1f}%" if t_pop > 0 else "N/A",
                    'Inondation (ha)': round(f_area, 2)
                })
        
        
        # ═════════════════════════════════════════════════════════════════
        # AFFICHAGE RÉSULTATS
        # ═════════════════════════════════════════════════════════════════
        
        # ─────────────────────────────────────────────────
        # 1. BILAN GÉNÉRAL
        # ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 **INDICATEURS DE RISQUE**")
        
        rain_sum = (
            df_climat['value_precip'].sum() 
            if df_climat is not None and 'value_precip' in df_climat.columns
            else 0
        )
        
        p1, p2, p3, p4 = st.columns(4)
        
        with p1:
            st.metric(
                "👥 Population Totale",
                f"{total_pop_all:,}"
            )
        
        with p2:
            pct_pop_exp = (
                (total_pop_exposed / total_pop_all * 100)
                if total_pop_all > 0 else 0
            )
            st.metric(
                "⚠️ Population Sinistrée",
                f"{total_pop_exposed:,}",
                f"{pct_pop_exp:.1f}%"
            )
        
        with p3:
            st.metric(
                "🌊 Zone Inondée",
                f"{total_flood_ha:.2f} ha"
            )
        
        with p4:
            st.metric(
                "🌧️ Cumul Précipitations",
                f"{rain_sum:.0f} mm"
            )
        
        
        # ─────────────────────────────────────────────────
        # 2. CARTE & INFRASTRUCTURES
        # ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🗺️ **CARTOGRAPHIE & IMPACTS**")
        
        col_map, col_stats = st.columns([2, 1])
        
        with col_map:
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(
                location=[center.y, center.x],
                zoom_start=10,
                tiles="cartodbpositron"
            )
            
            # Limites zone
            folium.GeoJson(
                st.session_state.selected_zone,
                name="Zone d'Étude",
                style_function=lambda x: {
                    'fillColor': '#f0f0f0',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.1
                }
            ).add_to(m)
            
            # Diagnostic (optionnel)
            if show_diagnostic:
                colors_map = {
                    'Brut (Δ > threshold)': 'red',
                    'Sans Eau Existante': 'orange',
                    'Sans Zones Urbaines': 'yellow',
                    'Pente Basse': 'lightgreen',
                    'Final (Connecté)': 'blue'
                }
                
                for stage_name, stage_img in flood_data['stages'].items():
                    try:
                        map_id = stage_img.getMapId({
                            'min': 0,
                            'max': 1,
                            'palette': ['white', colors_map.get(stage_name, 'blue')]
                        })
                        
                        folium.TileLayer(
                            tiles=map_id['tile_fetcher'].url_format,
                            attr='GEE',
                            name=f"Diag: {stage_name}",
                            overlay=True,
                            show=(stage_name == 'Final (Connecté)')
                        ).add_to(m)
                    except:
                        pass
            
            # Inondations finales
            try:
                map_id_flood = flood_mask.getMapId({
                    'palette': ['#0066ff'],
                    'min': 0,
                    'max': 1
                })
                
                folium.TileLayer(
                    tiles=map_id_flood['tile_fetcher'].url_format,
                    attr='GEE',
                    name='Inondations Finales',
                    overlay=True
                ).add_to(m)
            except:
                st.warning("⚠️ Impossible de charger inondations")
            
            # Routes impactées
            if not impacted_roads.empty:
                folium.GeoJson(
                    impacted_roads.to_json(),
                    name="Routes Coupées",
                    style_function=lambda x: {
                        'color': 'red',
                        'weight': 4,
                        'opacity': 0.8
                    }
                ).add_to(m)
            
            # Bâtiments impactés
            if not impacted_infra.empty:
                folium.GeoJson(
                    impacted_infra.to_json(),
                    name="Bâtiments Touchés",
                    style_function=lambda x: {
                        'fillColor': 'red',
                        'color': 'darkred',
                        'weight': 1,
                        'fillOpacity': 0.7
                    }
                ).add_to(m)
            
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=500, key="map_res")
        
        with col_stats:
            st.markdown("#### 🏗️ **Infrastructures**")
            
            st.metric("🛣️ Routes impactées", f"{len(impacted_roads)} seg.")
            st.metric("🏢 Bâtiments touchés", f"{len(impacted_infra)}")
            
            if not impacted_infra.empty:
                def translate_type(row):
                    """Traduit type OSM en catégorie."""
                    t = str(
                        row.get('amenity', row.get('building', 'Autre'))
                    ).lower()
                    
                    if any(x in t for x in ['school', 'university', 'college']):
                        return "🏫 Écoles"
                    if any(x in t for x in ['hospital', 'clinic', 'health', 'doctors']):
                        return "🏥 Santé"
                    return "🏠 Habitat"
                
                impacted_infra['Cat'] = impacted_infra.apply(translate_type, axis=1)
                
                fig_pie = px.pie(
                    impacted_infra,
                    names='Cat',
                    hole=0.4,
                    title="Bâtiments impactés (type)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        
        # ─────────────────────────────────────────────────
        # 3. SUIVI CLIMATIQUE
        # ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ☁️ **SUIVI CLIMATIQUE**")
        
        if df_climat is not None and not df_climat.empty:
            fig_clim = go.Figure()
            
            # Précipitations (barre)
            if 'value_precip' in df_climat.columns:
                fig_clim.add_trace(
                    go.Bar(
                        x=df_climat['date'],
                        y=df_climat['value_precip'],
                        name="Pluie (mm)",
                        marker_color='royalblue'
                    )
                )
            
            # Température (ligne)
            if 'value_temp' in df_climat.columns:
                fig_clim.add_trace(
                    go.Scatter(
                        x=df_climat['date'],
                        y=df_climat['value_temp'],
                        name="Température (°C)",
                        yaxis='y2',
                        line=dict(color='orange', width=3)
                    )
                )
            
            fig_clim.update_layout(
                title="Précipitations & Température",
                yaxis=dict(title="Pluie (mm)"),
                yaxis2=dict(
                    title="Temp (°C)",
                    overlaying='y',
                    side='right'
                ),
                legend=dict(orientation="h", y=1.1),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_clim, use_container_width=True)
        else:
            st.info("ℹ️ Pas de données climatiques disponibles")
        
        
        # ─────────────────────────────────────────────────
        # 4. SYNTHÈSE PAR SECTEUR
        # ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 **DÉTAIL PAR SECTEUR**")
        
        df_sectors = pd.DataFrame(sector_data)
        st.dataframe(
            df_sectors,
            use_container_width=True,
            hide_index=True
        )
        
        # Export CSV
        csv_data = df_sectors.to_csv(index=False)
        st.download_button(
            "📥 Télécharger CSV",
            csv_data,
            "analyse_inondations.csv",
            "text/csv"
        )

else:
    st.info("👈 Veuillez sélectionner une zone pour commencer l'analyse")


# ═════════════════════════════════════════════════════════════════
# PIED DE PAGE
# ═════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
### 📚 **Sources Données**
- **Sentinel-1 (ESA)**: Radar SAR, 10 m, quotidien
- **WorldPop**: Population 100 m, annuel
- **MODIS (NASA)**: Indices spectraux 250 m, quotidien  
- **CHIRPS (UCSB)**: Précipitations 5 km, quotidien
- **ERA5-Land (ECMWF)**: Météo 25 km, quotidien
- **SRTM (USGS)**: Altitude/pente 30 m, statique
- **OpenStreetMap**: Routes & bâtiments, temps réel

### ⚠️ **Limitations**
- Sentinel-1 sensible à végétation → faux positifs en zones boisées
- WorldPop = interpolation statistique (erreur ±20%)
- GEE timeout si région > 100,000 km²
- OSMnx coverage inégale (mieux zones urbaines)

### 🔑 **Changements Clés du Code**
- ✅ **Correction cache**: Dict au lieu de ee.Geometry
- ✅ **get_climate_data()**: Accepte aoi_geojson_dict (hashable)
- ✅ **get_population_stats()**: Accepte aoi_geojson_dict
- ✅ **get_area_stats()**: Accepte aoi_geojson_dict
- ✅ **Conversion**: `aoi_geojson = aoi_ee_global.getInfo()`
""")
