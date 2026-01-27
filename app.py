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
        # Période de référence (saison sèche 2023 par défaut pour comparaison)
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
        
        # Lissage pour réduire le speckle
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
        # Routes
        graph = ox.graph_from_polygon(poly, network_type='all', simplify=True)
        gdf_routes = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index().clip(_gdf_aoi)
        
        # Bâtiments & Infrastructures
        tags = {'building': True, 'amenity': True, 'healthcare': True, 'education': True}
        gdf_buildings = ox.features_from_polygon(poly, tags=tags)
        gdf_buildings = gdf_buildings[gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        gdf_buildings = gdf_buildings.reset_index().clip(_gdf_aoi)
        
        return gdf_buildings, gdf_routes
    except Exception: return gpd.GeoDataFrame(), gpd.GeoDataFrame()

def analyze_impacts(flood_mask, buildings_gdf):
    """Analyse d'impact précise en injectant OSM dans GEE"""
    if flood_mask is None or buildings_gdf.empty: return gpd.GeoDataFrame()
    try:
        # On limite pour éviter les timeouts (batch de 1000)
        infra_check = buildings_gdf.head(1500).copy()
        features = []
        for i, row in infra_check.iterrows():
            features.append(ee.Feature(ee.Geometry(mapping(row.geometry)), {'idx': i}))
        
        fc = ee.FeatureCollection(features)
        # Calcul de la moyenne de pixels d'inondation par bâtiment
        reduced = flood_mask.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        impacted_indices = [f['properties']['idx'] for f in reduced.filter(ee.Filter.gt('mean', 0)).getInfo()['features']]
        
        return infra_check.loc[impacted_indices]
    except: return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE UTILISATEUR
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🗺️ 1. Zone d'Étude")
mode = st.sidebar.radio("Méthode de sélection :", ["Liste Administrative", "Dessiner sur Carte", "Importer Fichier"])

selected_zone = None
zone_name = "Zone personnalisée"

if mode == "Liste Administrative":
    countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
    c_choice = st.sidebar.selectbox("Pays", list(countries.keys()))
    level = st.sidebar.slider("Niveau Admin", 0, 3, 2)
    gdf_base = load_gadm(countries[c_choice], level)
    if gdf_base is not None:
        col = f"NAME_{level}" if level > 0 else "COUNTRY"
        choice = st.sidebar.selectbox("Région/Cercle", sorted(gdf_base[col].unique()))
        selected_zone = gdf_base[gdf_base[col] == choice].copy()
        zone_name = choice

elif mode == "Dessiner sur Carte":
    st.sidebar.info("Dessinez un polygone sur la carte de droite.")
    
elif mode == "Importer Fichier":
    up = st.sidebar.file_uploader("Fichier Géo", type=['geojson', 'kml'])
    if up: 
        selected_zone = gpd.read_file(up).to_crs(epsg=4326)
        zone_name = "Zone Importée"

st.sidebar.header("📅 2. Période d'Analyse")
col_d1, col_d2 = st.sidebar.columns(2)
start_f = col_d1.date_input("Début", datetime(2024, 8, 1))
end_f = col_d2.date_input("Fin", datetime(2024, 9, 30))
threshold_val = st.sidebar.slider("Sensibilité détection (dB)", 2.0, 8.0, 4.0)

# ═════════════════════════════════════════════════════════════════
# 4. LOGIQUE PRINCIPALE
# ═════════════════════════════════════════════════════════════════

st.title(f"🌊 FloodWatch : {zone_name}")

if mode == "Dessiner sur Carte" and selected_zone is None:
    m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    Draw(export=False, draw_options={'polyline':False, 'circle':False, 'marker':False, 'circlemarker':False}).add_to(m_draw)
    out = st_folium(m_draw, width="100%", height=400, key="draw")
    if out and out.get('last_active_drawing'):
        geom = shape(out['last_active_drawing']['geometry'])
        selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
        st.rerun()

if selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE D'IMPACT", type="primary"):
        with st.spinner("Analyse spatiale en cours (Sentinel-1 + OSM + GEE)..."):
            # A. Préparation GEE
            aoi_ee = ee.Geometry(mapping(selected_zone.unary_union))
            
            # B. Extraction OSM
            buildings, routes = get_osm_data(selected_zone)
            
            # C. Masque Inondation
            flood_mask = get_flood_mask(aoi_ee, str(start_f), str(end_f), threshold_val)
            
            # D. Croisement des données
            impacted_infra = analyze_impacts(flood_mask, buildings)
            
            # --- RÉSULTATS ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Bâtiments total", len(buildings))
            m2.metric("⚠️ Infrastructures touchées", len(impacted_infra))
            m3.metric("Réseau routier (seg.)", len(routes))

            # --- CARTE FINALE ---
            center = selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="cartodbpositron")
            
            # Couche Inondation (GEE)
            if flood_mask:
                try:
                    map_id = flood_mask.getMapId({'palette': ['#00bfff']})
                    folium.TileLayer(
                        tiles=map_id['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name='Zones Inondées',
                        overlay=True, opacity=0.7
                    ).add_to(m)
                except: pass

            # Couche Routes
            folium.GeoJson(routes, name="Réseau routier", style_function=lambda x: {'color':'#555','weight':1}).add_to(m)
            
            # Couche Infrastructures Impactées
            if not impacted_infra.empty:
                impacted_infra['type'] = impacted_infra.get('amenity', impacted_infra.get('building', 'Inconnu')).fillna('Bâtiment')
                folium.GeoJson(
                    impacted_infra,
                    name="Impacts Détectés",
                    style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.8},
                    tooltip=folium.GeoJsonTooltip(fields=['type', 'name'], aliases=['Type:', 'Nom:'])
                ).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=700, key="final_map")
            
            if not impacted_infra.empty:
                st.subheader("📋 Liste des infrastructures impactées")
                st.dataframe(impacted_infra[['name', 'type']].dropna(subset=['name']), use_container_width=True)

else:
    st.info("Sélectionnez une zone dans la barre latérale pour commencer l'analyse.")
