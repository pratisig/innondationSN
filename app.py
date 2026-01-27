import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping
import json
import ee
from datetime import datetime, timedelta

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & INITIALISATION
# ═════════════════════════════════════════════════════════════════
st.set_page_config(page_title="FloodWatch WA - Surveillance Inondations", layout="wide")

# Paramètres OSMnx
ox.settings.timeout = 180
ox.settings.use_cache = True

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
# 2. FONCTIONS DE TRAITEMENT (GEE & OSM)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return None

def get_flood_mask(aoi_ee, start_flood, end_flood, threshold=4.0):
    """Génère un masque d'inondation Sentinel-1 via GEE"""
    if not gee_available: return None
    try:
        start_ref, end_ref = "2023-01-01", "2023-04-30"
        
        def get_s1_collection(start, end):
            return (ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filterBounds(aoi_ee)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                    .filterDate(start, end)
                    .select('VV'))

        img_ref = get_s1_collection(start_ref, end_ref).median().clip(aoi_ee)
        img_flood = get_s1_collection(start_flood, end_flood).min().clip(aoi_ee)
        
        img_ref = img_ref.focal_median(50, 'circle', 'meters')
        img_flood = img_flood.focal_median(50, 'circle', 'meters')

        diff = img_ref.subtract(img_flood)
        return diff.gt(threshold).rename('flood').selfMask()
    except: return None

def get_osm_data(_gdf_aoi):
    """Récupère bâtiments et routes via OSMnx"""
    if _gdf_aoi is None or _gdf_aoi.empty: return None, None
    try:
        poly = _gdf_aoi.unary_union
        graph = ox.graph_from_polygon(poly, network_type='all', simplify=True)
        gdf_routes = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index().clip(_gdf_aoi)
        
        tags = {'building': True, 'amenity': True, 'healthcare': True, 'education': True}
        try:
            gdf_buildings = ox.features_from_polygon(poly, tags=tags)
        except:
            gdf_buildings = ox.geometries_from_polygon(poly, tags=tags)
            
        gdf_buildings = gdf_buildings[gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        gdf_buildings = gdf_buildings.reset_index().clip(_gdf_aoi)
        
        return gdf_buildings, gdf_routes
    except: return gpd.GeoDataFrame(), gpd.GeoDataFrame()

def analyze_impacts(flood_mask, buildings_gdf):
    """Analyse d'impact précise en injectant OSM dans GEE"""
    if flood_mask is None or buildings_gdf.empty: return gpd.GeoDataFrame()
    try:
        infra_check = buildings_gdf.head(1500).copy()
        features = []
        for i, row in infra_check.iterrows():
            features.append(ee.Feature(ee.Geometry(mapping(row.geometry)), {'idx': i}))
        
        fc = ee.FeatureCollection(features)
        reduced = flood_mask.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        impacted_indices = [f['properties']['idx'] for f in reduced.filter(ee.Filter.gt('mean', 0)).getInfo()['features']]
        
        return infra_check.loc[impacted_indices]
    except: return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE UTILISATEUR (SIDEBAR)
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🗺️ 1. Zone d'Étude")
mode = st.sidebar.radio("Méthode de sélection :", ["Liste Administrative", "Dessiner sur Carte", "Importer Fichier"])

if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = None
if 'zone_name' not in st.session_state:
    st.session_state.zone_name = "Zone personnalisée"
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False

if mode == "Liste Administrative":
    countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
    c_choice = st.sidebar.selectbox("Pays", list(countries.keys()))
    level = st.sidebar.slider("Niveau Admin", 0, 5, 2)
    gdf_base = load_gadm(countries[c_choice], level)
    if gdf_base is not None:
        col = f"NAME_{level}" if level > 0 else "COUNTRY"
        if col in gdf_base.columns:
            choice = st.sidebar.selectbox("Sélectionner la subdivision", sorted(gdf_base[col].dropna().unique()))
            # Réinitialiser le trigger si la zone change
            new_zone = gdf_base[gdf_base[col] == choice].copy()
            if st.session_state.zone_name != choice:
                st.session_state.selected_zone = new_zone
                st.session_state.zone_name = choice
                st.session_state.analysis_triggered = False
        else:
            st.sidebar.error(f"Le niveau {level} n'est pas disponible.")

elif mode == "Dessiner sur Carte":
    st.sidebar.info("Dessinez un polygone sur la carte de prévisualisation.")
    m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    Draw(export=False, draw_options={'polyline':False, 'circle':False, 'marker':False, 'circlemarker':False}).add_to(m_draw)
    with st.sidebar:
        out = st_folium(m_draw, width=250, height=250, key="draw_sidebar_static")
        if out and out.get('last_active_drawing'):
            geom = shape(out['last_active_drawing']['geometry'])
            st.session_state.selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
            st.session_state.zone_name = "Zone Dessinée"

elif mode == "Importer Fichier":
    up = st.sidebar.file_uploader("Fichier Géo (GeoJSON, KML)", type=['geojson', 'kml'])
    if up: 
        try:
            st.session_state.selected_zone = gpd.read_file(up).to_crs("epsg:4326")
            st.session_state.zone_name = "Zone Importée"
            st.session_state.analysis_triggered = False
        except:
            st.sidebar.error("Erreur de lecture.")

st.sidebar.header("📅 2. Période d'Analyse")
col_d1, col_d2 = st.sidebar.columns(2)
start_f = col_d1.date_input("Début", datetime(2024, 8, 1))
end_f = col_d2.date_input("Fin", datetime(2024, 9, 30))
threshold_val = st.sidebar.slider("Sensibilité (dB)", 2.0, 8.0, 4.0)

# ═════════════════════════════════════════════════════════════════
# 4. LOGIQUE PRINCIPALE
# ═════════════════════════════════════════════════════════════════

st.title(f"🌊 FloodWatch : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE D'IMPACT", type="primary", use_container_width=True):
        st.session_state.analysis_triggered = True

    if st.session_state.analysis_triggered:
        with st.spinner(f"Analyse de {st.session_state.zone_name} en cours..."):
            # Extraction et Analyse
            aoi_ee = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
            buildings, routes = get_osm_data(st.session_state.selected_zone)
            flood_mask = get_flood_mask(aoi_ee, str(start_f), str(end_f), threshold_val)
            impacted_infra = analyze_impacts(flood_mask, buildings)
            
            # --- RÉSULTATS ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Bâtiments total", len(buildings))
            m2.metric("⚠️ Impacts détectés", len(impacted_infra))
            m3.metric("Routes (segments)", len(routes))

            # --- CARTE FINALE ---
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="cartodbpositron")
            
            # Couches
            if flood_mask:
                try:
                    map_id = flood_mask.getMapId({'palette': ['#00bfff']})
                    folium.TileLayer(
                        tiles=map_id['tile_fetcher'].url_format,
                        attr='GEE', name='Inondation', overlay=True, opacity=0.7
                    ).add_to(m)
                except: pass

            folium.GeoJson(st.session_state.selected_zone, name="Zone d'étude", 
                           style_function=lambda x: {'fillColor': 'none', 'color': 'orange', 'weight': 2}).add_to(m)

            if not routes.empty:
                folium.GeoJson(routes, name="Réseau routier", style_function=lambda x: {'color':'#555','weight':1}).add_to(m)
            
            if not impacted_infra.empty:
                impacted_infra['type'] = impacted_infra.get('amenity', impacted_infra.get('building', 'Inconnu')).fillna('Bâtiment')
                folium.GeoJson(
                    impacted_infra,
                    name="Impacts",
                    style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.8},
                    tooltip=folium.GeoJsonTooltip(fields=['type', 'name'], aliases=['Type:', 'Nom:'])
                ).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600, key="result_map")
            
            if not impacted_infra.empty:
                st.subheader("📋 Infrastructures impactées")
                st.dataframe(impacted_infra[['name', 'type']].dropna(subset=['name']), use_container_width=True)
    else:
        # Prévisualisation de la zone sélectionnée
        center = st.session_state.selected_zone.centroid.iloc[0]
        m_pre = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="cartodbpositron")
        folium.GeoJson(st.session_state.selected_zone, style_function=lambda x: {'color': 'orange'}).add_to(m_pre)
        st_folium(m_pre, width="100%", height=500, key="pre_view")
else:
    st.info("💡 Sélectionnez une zone dans la barre latérale pour débloquer l'analyse.")
    m_default = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    st_folium(m_default, width="100%", height=500, key="default_map")
