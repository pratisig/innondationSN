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

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & INITIALISATION
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FloodWatch WA", 
    page_icon="🌊",
    layout="wide"
)

# Initialisation GEE
@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(key_dict["client_email"], key_data=json.dumps(key_dict))
            ee.Initialize(credentials)
            return True
        ee.Initialize()
        return True
    except Exception:
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS DE TRAITEMENT (GEE, CLIMAT & OSM)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return None

def advanced_flood_detection(aoi, ref_start, ref_end, flood_start, flood_end, threshold_db=0.75, min_pixels=8):
    if not gee_available: return None
    try:
        # S1 Référence et Crise
        def get_s1(start, end):
            return (ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filterBounds(aoi)
                    .filterDate(start, end)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                    .select("VV")
                    .median())

        s1_ref = get_s1(ref_start, ref_end)
        s1_crisis = get_s1(flood_start, flood_end)
        
        ref_db = ee.Image(10).multiply(s1_ref.max(0.0001).log10())
        crisis_db = ee.Image(10).multiply(s1_crisis.max(0.0001).log10())
        
        delta_db = ref_db.subtract(crisis_db)
        flood_raw = delta_db.gt(threshold_db).rename('flood')
        
        # Masque Eau Existante (NDWI)
        modis_ref = ee.ImageCollection("MODIS/006/MOD09GA").filterBounds(aoi).filterDate(ref_start, ref_end).median()
        ndwi_ref = modis_ref.normalizedDifference(['sur_refl_b02', 'sur_refl_b06'])
        mask_not_existing_water = ndwi_ref.lt(0.3).unmask(1)
        
        flood_no_water = flood_raw.updateMask(mask_not_existing_water)
        
        # Masque Urbain (NDBI)
        modis_crisis = ee.ImageCollection("MODIS/006/MOD09GA").filterBounds(aoi).filterDate(flood_start, flood_end).median()
        ndbi = modis_crisis.normalizedDifference(['sur_refl_b06', 'sur_refl_b02'])
        mask_not_urban = ndbi.lt(0.1).unmask(1)
        
        flood_no_urban = flood_no_water.updateMask(mask_not_urban)
        
        # Masque Pente (SRTM)
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Algorithms.Terrain(dem).select("slope")
        mask_low_slope = slope.lt(5).unmask(1)
        
        flood_low_slope = flood_no_urban.updateMask(mask_low_slope)
        
        # Connectivité
        connected_pixels = flood_low_slope.connectedPixelCount(8)
        flood_connected = flood_low_slope.updateMask(connected_pixels.gte(min_pixels))
        
        return {
            'flood_final': flood_connected.selfMask(),
            'stages': {
                'Brut': flood_raw.selfMask(),
                'Sans_Eau': flood_no_water.selfMask(),
                'Sans_Urbain': flood_no_urban.selfMask(),
                'Pente_Basse': flood_low_slope.selfMask(),
                'Final': flood_connected.selfMask()
            }
        }
    except Exception as e:
        st.error(f"Erreur GEE : {str(e)}")
        return None

def get_population_stats(aoi_ee, flood_mask):
    if not gee_available: return 0, 0
    try:
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01').mosaic().clip(aoi_ee)
        total_pop = pop_dataset.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100).get('population').getInfo() or 0
        exposed_pop = 0
        if flood_mask:
            exposed_pop = pop_dataset.updateMask(flood_mask).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100).get('population').getInfo() or 0
        return int(total_pop), int(exposed_pop)
    except: return 0, 0

def get_area_stats(aoi_ee, flood_mask):
    if not gee_available or not flood_mask: return 0.0
    try:
        area = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=10).get('flood').getInfo()
        return (area or 0) / 10000
    except: return 0.0

def get_osm_data(_gdf_aoi):
    if _gdf_aoi is None or _gdf_aoi.empty: return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    try:
        poly = _gdf_aoi.unary_union
        graph = ox.graph_from_polygon(poly, network_type='all', simplify=True)
        routes = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index().clip(_gdf_aoi)
        tags = {'building': True, 'amenity': ['school', 'hospital']}
        try: buildings = ox.features_from_polygon(poly, tags=tags)
        except: buildings = ox.geometries_from_polygon(poly, tags=tags)
        buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])].reset_index().clip(_gdf_aoi)
        return buildings, routes
    except: return gpd.GeoDataFrame(), gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE & LOGIQUE
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🗺️ 1. Zone d'Étude")
countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
c_choice = st.sidebar.selectbox("Pays", list(countries.keys()))
level = st.sidebar.slider("Niveau Administratif", 0, 3, 2)
gdf_base = load_gadm(countries[c_choice], level)

selected_zone = None
zone_name = "Zone non définie"

if gdf_base is not None:
    col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
    choices = st.sidebar.multiselect("Sélectionner commune(s)/département(s)", sorted(gdf_base[col_name].unique()))
    if choices:
        selected_zone = gdf_base[gdf_base[col_name].isin(choices)].copy()
        zone_name = ", ".join(choices)

st.sidebar.markdown("## 📅 2. Paramètres")
ref_start = st.sidebar.date_input("Réf. Début", datetime(2023, 1, 1))
ref_end = st.sidebar.date_input("Réf. Fin", datetime(2023, 4, 30))
flood_start = st.sidebar.date_input("Crise Début", datetime(2024, 8, 1))
flood_end = st.sidebar.date_input("Crise Fin", datetime(2024, 9, 30))
threshold_db = st.sidebar.slider("Seuil (dB)", 0.5, 3.0, 0.75)
min_pix = st.sidebar.slider("Filtre Bruit (px)", 1, 20, 8)
show_diagnostic = st.sidebar.checkbox("Afficher masques intermédiaires", False)

st.title(f"🌊 FloodWatch : {zone_name}")

if selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE", type="primary", use_container_width=True):
        with st.spinner("Calculs spatiaux en cours..."):
            aoi_ee = ee.Geometry(mapping(selected_zone.unary_union))
            res = advanced_flood_detection(aoi_ee, str(ref_start), str(ref_end), str(flood_start), str(flood_end), threshold_db, min_pix)
            
            if res:
                flood_mask = res['flood_final']
                
                # Stats Globales
                t_pop, e_pop = get_population_stats(aoi_ee, flood_mask)
                f_ha = get_area_stats(aoi_ee, flood_mask)
                buildings, routes = get_osm_data(selected_zone)
                
                # Affichage Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Pop. Exposée", f"{e_pop:,}")
                m2.metric("Surface Inondée", f"{f_ha:.1f} ha")
                m3.metric("Bâtiments à risque", f"{len(buildings) if not buildings.empty else 0}")
                
                # Carte
                st.markdown("### 🗺️ Visualisation")
                center = selected_zone.centroid.iloc[0]
                m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron")
                folium.GeoJson(selected_zone, name="AOI", style_function=lambda x: {'fillColor': 'none', 'color': 'black'}).add_to(m)
                
                if show_diagnostic:
                    for stage_key, stage_img in res['stages'].items():
                        try:
                            # Test si l'image a des données pour éviter l'erreur GEE Maps API
                            mid = stage_img.getMapId({'palette': ['orange'] if 'Sans' in stage_key else ['blue']})
                            folium.TileLayer(tiles=mid['tile_fetcher'].url_format, attr='GEE', name=f"Etape: {stage_key}", overlay=True, show=False).add_to(m)
                        except: pass
                
                try:
                    mid_final = flood_mask.getMapId({'palette': ['#00bfff']})
                    folium.TileLayer(tiles=mid_final['tile_fetcher'].url_format, attr='GEE', name='Inondation Finale', overlay=True).add_to(m)
                except: st.warning("Aucune zone inondée détectée avec ces paramètres.")
                
                folium.LayerControl().add_to(m)
                st_folium(m, width="100%", height=600)
                
                # Tableau par subdivision
                st.markdown("### 📋 Détail par secteur")
                details = []
                for _, row in selected_zone.iterrows():
                    g_ee = ee.Geometry(mapping(row.geometry))
                    tp, ep = get_population_stats(g_ee, flood_mask)
                    details.append({'Secteur': row[col_name], 'Pop. Totale': tp, 'Pop. Impactée': ep, 'Surface (ha)': get_area_stats(g_ee, flood_mask)})
                st.table(pd.DataFrame(details))
else:
    st.info("Veuillez sélectionner au moins une commune dans la barre latérale pour commencer.")
