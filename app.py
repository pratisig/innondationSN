import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import json

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & STYLE
# ═════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Visualiseur OSM Multi-Sources", layout="wide")

# Paramètres OSMnx pour la performance
ox.settings.timeout = 180
ox.settings.use_cache = True

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# 2. CHARGEMENT DES DONNÉES
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Erreur lors du chargement des limites GADM : {e}")
        return None

def get_osm_data(_gdf_aoi):
    """Récupère les bâtiments et les routes depuis OSM via OSMnx"""
    if _gdf_aoi is None or _gdf_aoi.empty:
        return None, None
    
    try:
        # Création du polygone de recherche (convex hull pour éviter les géométries trop complexes)
        poly = _gdf_aoi.unary_union
        
        # 1. Récupération des routes
        graph = ox.graph_from_polygon(poly, network_type='all', simplify=True)
        gdf_routes = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        
        # 2. Récupération des bâtiments et équipements
        tags = {
            'building': True, 
            'amenity': True,
            'healthcare': True,
            'education': True
        }
        
        try:
            gdf_buildings = ox.features_from_polygon(poly, tags=tags)
        except AttributeError:
            gdf_buildings = ox.geometries_from_polygon(poly, tags=tags)
        
        # Nettoyage et simplification
        if not gdf_buildings.empty:
            gdf_buildings = gdf_buildings[gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            gdf_buildings = gdf_buildings.reset_index()
            gdf_buildings = gdf_buildings.clip(_gdf_aoi)
            
        if not gdf_routes.empty:
            gdf_routes = gdf_routes.reset_index()
            gdf_routes = gdf_routes.clip(_gdf_aoi)
            
        return gdf_buildings, gdf_routes
    except Exception as e:
        st.warning(f"Note : Les données OSM n'ont pas pu être récupérées ({e})")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE UTILISATEUR (SIDEBAR)
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🗺️ Mode de Sélection")
mode = st.sidebar.radio("Choisir la méthode :", ["Liste Administrative", "Dessiner sur Carte", "Importer Fichier"])

selected_zone = None
zone_name = "Zone personnalisée"

if mode == "Liste Administrative":
    country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    level = st.sidebar.slider("Niveau Administratif (GADM)", 0, 3, 2)
    
    gdf_base = load_gadm(iso, level)
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
        names = sorted(gdf_base[col_name].astype(str).unique())
        choice = st.sidebar.selectbox("Zone", names)
        selected_zone = gdf_base[gdf_base[col_name] == choice].copy()
        zone_name = choice

elif mode == "Dessiner sur Carte":
    st.sidebar.info("Utilisez les outils de dessin sur la carte à droite pour définir votre zone.")
    # La zone sera récupérée via le retour de st_folium

elif mode == "Importer Fichier":
    uploaded_file = st.sidebar.file_uploader("Fichier (GeoJSON, KML)", type=['geojson', 'kml'])
    if uploaded_file is not None:
        try:
            selected_zone = gpd.read_file(uploaded_file).to_crs(epsg=4326)
            zone_name = "Import Fichier"
        except Exception as e:
            st.sidebar.error(f"Erreur : {e}")

# ═════════════════════════════════════════════════════════════════
# 4. AFFICHAGE PRINCIPAL
# ═════════════════════════════════════════════════════════════════

st.title(f"Visualiseur d'Infrastructures : {zone_name}")

# Gestion spécifique du mode dessin
if mode == "Dessiner sur Carte":
    m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    Draw(export=False, draw_options={'polyline':False, 'circle':False, 'marker':False, 'circlemarker':False}).add_to(m_draw)
    output = st_folium(m_draw, width="100%", height=400, key="draw_map")
    
    if output and output.get('last_active_drawing'):
        geom = shape(output['last_active_drawing']['geometry'])
        selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])

# Traitement des données si une zone est définie
if selected_zone is not None:
    with st.spinner("Extraction des données OpenStreetMap..."):
        buildings, routes = get_osm_data(selected_zone)
        
    # --- Statistiques ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Bâtiments", len(buildings) if buildings is not None else 0)
    c2.metric("Routes (segments)", len(routes) if routes is not None else 0)
    c3.metric("Source", mode)

    # --- Carte de visualisation ---
    center = selected_zone.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=14, tiles="cartodbpositron")

    # Zone
    folium.GeoJson(selected_zone, style_function=lambda x: {'fillColor': '#ff7800', 'color': '#ff7800', 'weight': 2, 'fillOpacity': 0.1}, name="Zone d'étude").add_to(m)

    # Routes
    if routes is not None and not routes.empty:
        folium.GeoJson(routes.__geo_interface__, style_function=lambda x: {'color': '#333333', 'weight': 2}, name="Routes").add_to(m)

    # Bâtiments
    if buildings is not None and not buildings.empty:
        buildings['display_type'] = buildings.get('amenity', buildings.get('building', 'Bâtiment')).fillna('Bâtiment')
        if 'name' not in buildings.columns: buildings['name'] = "Inconnu"
        
        folium.GeoJson(
            buildings.__geo_interface__,
            style_function=lambda x: {'fillColor': '#2ecc71', 'color': '#27ae60', 'weight': 1, 'fillOpacity': 0.7},
            tooltip=folium.GeoJsonTooltip(fields=['display_type', 'name'], aliases=['Type:', 'Nom:']),
            name="Infrastructures"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=600, key="main_map")
    
    if st.checkbox("Afficher les données brutes"):
        st.dataframe(buildings[['name', 'display_type']].dropna(how='all'), use_container_width=True)
else:
    if mode != "Dessiner sur Carte":
        st.info("Veuillez sélectionner ou importer une zone pour commencer.")
