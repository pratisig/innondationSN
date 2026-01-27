import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import mapping

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & STYLE
# ═════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Visualiseur OSM", layout="wide")

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
# 2. CHARGEMENT DES DONNÉES ADMINISTRATIVES
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

@st.cache_data(show_spinner=False)
def get_osm_data(_gdf_aoi):
    """Récupère les bâtiments et les routes depuis OSM via OSMnx"""
    if _gdf_aoi is None or _gdf_aoi.empty:
        return None, None
    
    try:
        # Création du polygone de recherche
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
        gdf_buildings = ox.geometries_from_polygon(poly, tags=tags)
        
        # Nettoyage des données
        if not gdf_buildings.empty:
            gdf_buildings = gdf_buildings[gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            gdf_buildings = gdf_buildings.clip(_gdf_aoi).reset_index(drop=True)
            
        if not gdf_routes.empty:
            gdf_routes = gdf_routes.clip(_gdf_aoi).reset_index(drop=True)
            
        return gdf_buildings, gdf_routes
    except Exception as e:
        st.warning(f"Note : Certaines données OSM n'ont pas pu être récupérées pour cette zone ({e})")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE UTILISATEUR (SIDEBAR)
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🗺️ Configuration")

country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
country = st.sidebar.selectbox("Sélectionner un Pays", list(country_dict.keys()))
iso = country_dict[country]

level = st.sidebar.slider("Niveau Administratif (GADM)", 0, 3, 2)
gdf_base = load_gadm(iso, level)

selected_zone = None
if gdf_base is not None:
    col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
    names = sorted(gdf_base[col_name].astype(str).unique())
    choice = st.sidebar.selectbox("Choisir la Zone", names)
    selected_zone = gdf_base[gdf_base[col_name] == choice].copy()

st.sidebar.markdown("---")
st.sidebar.info("""
**Données affichées :**
- 🏠 Bâtiments (OSM)
- 🛣️ Réseau Routier (OSM)
- 🟧 Limites Admin (GADM)
""")

# ═════════════════════════════════════════════════════════════════
# 4. AFFICHAGE PRINCIPAL
# ═════════════════════════════════════════════════════════════════

st.title(f"Visualiseur d'Infrastructures : {choice if selected_zone is not None else ''}")

if selected_zone is not None:
    with st.spinner("Extraction des données OpenStreetMap en cours..."):
        buildings, routes = get_osm_data(selected_zone)
        
    # --- Statistiques ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Bâtiments détectés", len(buildings) if buildings is not None else 0)
    c2.metric("Segments de route", len(routes) if routes is not None else 0)
    c3.metric("Zone Admin", choice)

    # --- Carte ---
    center = selected_zone.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="cartodbpositron")

    # 1. Limite Administrative
    folium.GeoJson(
        selected_zone,
        name="Limites Administratives",
        style_function=lambda x: {
            'fillColor': '#ff7800', 
            'color': '#ff7800', 
            'weight': 3, 
            'fillOpacity': 0.1
        }
    ).add_to(m)

    # 2. Réseau Routier
    if routes is not None and not routes.empty:
        folium.GeoJson(
            routes,
            name="Routes & Chemins",
            style_function=lambda x: {
                'color': '#555555', 
                'weight': 1.5, 
                'opacity': 0.8
            }
        ).add_to(m)

    # 3. Bâtiments
    if buildings is not None and not buildings.empty:
        # Création d'une colonne type simplifiée pour le style
        buildings['display_type'] = buildings['amenity'].fillna(buildings['building']).fillna('Bâtiment')
        
        folium.GeoJson(
            buildings,
            name="Bâtiments & Infrastructures",
            style_function=lambda x: {
                'fillColor': '#2ecc71', 
                'color': '#27ae60', 
                'weight': 1, 
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['display_type', 'name'], 
                aliases=['Type:', 'Nom:'],
                localize=True
            )
        ).add_to(m)

    # Contrôle des couches
    folium.LayerControl().add_to(m)
    
    # Affichage de la carte
    st_folium(m, width="100%", height=700, key="osm_viewer_map")
    
    # --- Table des données (optionnel) ---
    if st.checkbox("Afficher la liste des infrastructures"):
        if not buildings.empty:
            st.dataframe(buildings[['name', 'display_type']].dropna(subset=['name']), use_container_width=True)
        else:
            st.write("Aucune donnée textuelle disponible pour ces bâtiments.")

else:
    st.warning("Veuillez sélectionner une zone administrative dans la barre latérale.")
