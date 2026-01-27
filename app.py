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
import tempfile

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

# Initialisation du state pour persistance des données
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = None
if 'zone_name' not in st.session_state:
    st.session_state.zone_name = "Zone non définie"

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS DE TRAITEMENT
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return None

def advanced_flood_detection(aoi, ref_start, ref_end, flood_start, flood_end, threshold_db=1.25, min_pixels=5):
    if not gee_available: return None
    try:
        # S1 Collection
        def get_collection(start, end):
            return (ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filterBounds(aoi)
                    .filterDate(start, end)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                    .select("VV"))

        # Référence (Médiane pour la stabilité)
        s1_ref = get_collection(ref_start, ref_end).median()
        # Crise (Percentile 10 au lieu de Min pour éviter les artefacts de bruit)
        s1_crisis = get_collection(flood_start, flood_end).reduce(ee.Reducer.percentile([10])).rename('VV')
        
        # Conversion Log
        ref_db = ee.Image(10).multiply(s1_ref.log10())
        crisis_db = ee.Image(10).multiply(s1_crisis.log10())
        
        # Détection (Inondation = baisse de la rétrodiffusion radar)
        flood_raw = ref_db.subtract(crisis_db).gt(threshold_db).rename('flood')
        
        # Masquage Eau Permanente (JRC Global Surface Water)
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        mask_not_permanent = jrc.lt(80).unmask(1)
        
        # Masquage Pente
        slope = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003")).select('slope')
        mask_flat = slope.lt(5).unmask(1)
        
        # Application des masques
        flood_final = flood_raw.updateMask(mask_not_permanent).updateMask(mask_flat)
        
        # Nettoyage morphologique corrigé (Utilisation de Kernel au lieu de l'argument shape)
        kernel = ee.Kernel.circle(radius=1)
        flood_final = flood_final.focal_mode(kernel=kernel, iterations=1)
        
        connected = flood_final.connectedPixelCount(15) 
        flood_final = flood_final.updateMask(connected.gte(min_pixels))
        
        return {
            'flood_mask': flood_final.selfMask(),
            'ref_db': ref_db,
            'crisis_db': crisis_db
        }
    except Exception as e:
        st.error(f"Erreur GEE : {str(e)}")
        return None

def get_stats(aoi_ee, flood_mask):
    if not gee_available or not flood_mask: return 0, 0.0
    try:
        # Population
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01').mosaic().clip(aoi_ee)
        exposed_pop = pop_dataset.updateMask(flood_mask).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9
        ).get('population').getInfo() or 0
        
        # Surface
        area = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=20, maxPixels=1e9
        ).get('flood').getInfo() or 0
        
        return int(exposed_pop), float(area / 10000)
    except: return 0, 0.0

def get_osm_data(_gdf):
    try:
        poly = _gdf.unary_union
        tags = {'building': True}
        try: b = ox.features_from_polygon(poly, tags=tags)
        except: b = ox.geometries_from_polygon(poly, tags=tags)
        if b.empty: return pd.DataFrame(), None
        return b.reset_index().clip(_gdf), None
    except: return pd.DataFrame(), None

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🗺️ 1. Zone d'Étude")
mode = st.sidebar.selectbox("Mode", ["GADM", "Dessin", "Upload"])

if mode == "GADM":
    countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
    c_choice = st.sidebar.selectbox("Pays", list(countries.keys()))
    level = st.sidebar.slider("Niveau Administratif", 0, 4, 2)
    gdf_base = load_gadm(countries[c_choice], level)
    if gdf_base is not None:
        col = f"NAME_{level}" if level > 0 else "COUNTRY"
        choices = st.sidebar.multiselect("Unités", sorted(gdf_base[col].unique()))
        if choices:
            st.session_state.selected_zone = gdf_base[gdf_base[col].isin(choices)].copy()
            st.session_state.zone_name = ", ".join(choices)

elif mode == "Dessin":
    m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6)
    Draw(export=False).add_to(m_draw)
    res_draw = st_folium(m_draw, width=300, height=300, key="draw_sidebar")
    if res_draw and res_draw.get('last_active_drawing'):
        geom = shape(res_draw['last_active_drawing']['geometry'])
        st.session_state.selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
        st.session_state.zone_name = "Zone dessinée"

elif mode == "Upload":
    up = st.sidebar.file_uploader("Fichier", type=['geojson', 'kml'])
    if up:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(up.read())
            st.session_state.selected_zone = gpd.read_file(f.name).to_crs(epsg=4326)
            st.session_state.zone_name = up.name

st.sidebar.markdown("## 📅 2. Paramètres")
# Utilisation de listes pour s'assurer que date_input renvoie toujours 2 valeurs
d_dry = st.sidebar.date_input("Période Sèche", [datetime(2025, 1, 1), datetime(2025, 4, 30)])
d_wet = st.sidebar.date_input("Période Humide", [datetime(2025, 6, 1), datetime(2025, 9, 30)])
sens = st.sidebar.slider("Sensibilité (dB)", 0.5, 5.0, 1.5)
noise = st.sidebar.slider("Filtre Bruit", 1, 15, 5)

st.title(f"🌊 FloodWatch : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    # Bouton d'analyse
    if st.button("🚀 ANALYSER LA ZONE", type="primary", use_container_width=True):
        if len(d_dry) < 2 or len(d_wet) < 2:
            st.error("Veuillez sélectionner une plage de dates (début et fin) pour chaque période.")
        else:
            with st.spinner("Traitement des données satellites..."):
                aoi_ee = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
                res = advanced_flood_detection(
                    aoi_ee, 
                    str(d_dry[0]), str(d_dry[1]), 
                    str(d_wet[0]), str(d_wet[1]), 
                    sens, noise
                )
                
                if res:
                    exposed_pop, flood_ha = get_stats(aoi_ee, res['flood_mask'])
                    buildings, _ = get_osm_data(st.session_state.selected_zone)
                    
                    # Sauvegarde dans le state pour éviter la disparition au refresh
                    st.session_state.results = {
                        'pop': exposed_pop,
                        'area': flood_ha,
                        'buildings': len(buildings) if not buildings.empty else 0,
                        'mask_id': res['flood_mask'].getMapId({'palette': ['#00BFFF']}) if res['flood_mask'] else None
                    }

    # Affichage permanent des résultats s'ils existent dans le state
    if st.session_state.results:
        res_data = st.session_state.results
        c1, c2, c3 = st.columns(3)
        c1.metric("Population Exposée", f"{res_data['pop']:,}")
        c2.metric("Surface Inondée", f"{res_data['area']:.2f} ha")
        c3.metric("Bâtiments Impactés", f"{res_data['buildings']}")
        
        # Carte
        center = st.session_state.selected_zone.centroid.iloc[0]
        m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
        folium.GeoJson(st.session_state.selected_zone, name="Zone", style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2}).add_to(m)
        
        if res_data['mask_id']:
            folium.TileLayer(
                tiles=res_data['mask_id']['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name='Inondation (Satellite)',
                overlay=True,
                control=True
            ).add_to(m)
        else:
            st.warning("Aucune inondation détectée avec ces paramètres.")
            
        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=600, key="map_final_static")

else:
    st.info("Sélectionnez une zone pour commencer.")
