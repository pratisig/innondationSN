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
from datetime import datetime, timedelta

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & INITIALISATION
# ═════════════════════════════════════════════════════════════════
st.set_page_config(page_title="FloodWatch WA - Dashboard Impact", layout="wide")

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
        # Période de référence fixe pour la comparaison
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
        
        # Filtre de réduction du bruit (Speckle)
        img_ref = img_ref.focal_median(50, 'circle', 'meters')
        img_flood = img_flood.focal_median(50, 'circle', 'meters')

        diff = img_ref.subtract(img_flood)
        return diff.gt(threshold).rename('flood').selfMask()
    except: return None

def get_population_stats(aoi_ee, flood_mask):
    """Calcule la population via WorldPop mosaïquée"""
    if not gee_available: return 0, 0
    try:
        # Utilisation de la méthode mosaïque suggérée pour assurer la couverture
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01', '2021-01-01') \
                        .mosaic().clip(aoi_ee)
        
        # Population totale
        stats_total = pop_dataset.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e9
        )
        total_pop = stats_total.get('population').getInfo() or 0
        
        exposed_pop = 0
        if flood_mask:
            # Population exposée via masque
            stats_exposed = pop_dataset.updateMask(flood_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi_ee,
                scale=100,
                maxPixels=1e9
            )
            exposed_pop = stats_exposed.get('population').getInfo() or 0
            
        return int(total_pop), int(exposed_pop)
    except Exception as e:
        st.error(f"Erreur Population: {e}")
        return 0, 0

def get_area_stats(aoi_ee, flood_mask):
    """Calcule la superficie inondée en hectares"""
    if not gee_available or not flood_mask: return 0.0
    try:
        area_m2 = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=10,
            maxPixels=1e9
        ).get('flood').getInfo()
        return (area_m2 or 0) / 10000
    except: return 0.0

def get_osm_data(_gdf_aoi):
    """Récupère bâtiments et routes via OSMnx"""
    if _gdf_aoi is None or _gdf_aoi.empty: return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    try:
        poly = _gdf_aoi.unary_union
        # Récupération du réseau routier
        graph = ox.graph_from_polygon(poly, network_type='all', simplify=True)
        gdf_routes = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index().clip(_gdf_aoi)
        
        # Récupération des bâtiments
        tags = {'building': True, 'amenity': True, 'healthcare': True, 'education': True}
        try:
            gdf_buildings = ox.features_from_polygon(poly, tags=tags)
        except:
            gdf_buildings = ox.geometries_from_polygon(poly, tags=tags)
            
        gdf_buildings = gdf_buildings[gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        gdf_buildings = gdf_buildings.reset_index().clip(_gdf_aoi)
        
        return gdf_buildings, gdf_routes
    except: return gpd.GeoDataFrame(), gpd.GeoDataFrame()

def analyze_impacted_infra(flood_mask, buildings_gdf):
    """Vérifie quels bâtiments OSM touchent le masque d'inondation"""
    if flood_mask is None or buildings_gdf.empty: return gpd.GeoDataFrame()
    try:
        # Limite pour performance
        infra_check = buildings_gdf.head(2000).copy()
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
    level = st.sidebar.slider("Niveau Admin (0=Pays, 2=Département)", 0, 5, 2)
    gdf_base = load_gadm(countries[c_choice], level)
    if gdf_base is not None:
        col = f"NAME_{level}" if level > 0 else "COUNTRY"
        if col in gdf_base.columns:
            choice = st.sidebar.selectbox("Sélectionner la subdivision", sorted(gdf_base[col].dropna().unique()))
            new_zone = gdf_base[gdf_base[col] == choice].copy()
            if st.session_state.zone_name != choice:
                st.session_state.selected_zone = new_zone
                st.session_state.zone_name = choice
                st.session_state.analysis_triggered = False
        else:
            st.sidebar.error(f"Le niveau {level} n'est pas disponible.")

elif mode == "Dessiner sur Carte":
    st.sidebar.info("Dessinez un polygone sur la carte ci-dessous.")
    m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    Draw(export=False, draw_options={'polyline':False, 'circle':False, 'marker':False, 'circlemarker':False}).add_to(m_draw)
    with st.sidebar:
        out = st_folium(m_draw, width=250, height=250, key="draw_sidebar_static")
        if out and out.get('last_active_drawing'):
            geom = shape(out['last_active_drawing']['geometry'])
            st.session_state.selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
            st.session_state.zone_name = "Zone Dessinée"
            st.session_state.analysis_triggered = False

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

st.title(f"🌊 FloodWatch Dashboard : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 ANALYSER L'IMPACT MULTI-SOURCE", type="primary", use_container_width=True):
        st.session_state.analysis_triggered = True

    if st.session_state.analysis_triggered:
        with st.spinner("Analyse spatiale en cours (WorldPop + OSM + Sentinel-1)..."):
            # A. Préparation GEE & Masque
            aoi_ee_global = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
            flood_mask = get_flood_mask(aoi_ee_global, str(start_f), str(end_f), threshold_val)
            
            # B. Données OSM
            buildings, routes = get_osm_data(st.session_state.selected_zone)
            impacted_infra = analyze_impacted_infra(flood_mask, buildings)
            
            # C. Analyse par polygone (Boucle suggérée)
            temp_list = []
            for idx, row in st.session_state.selected_zone.iterrows():
                geom_ee = ee.Geometry(mapping(row.geometry))
                t_pop, e_pop = get_population_stats(geom_ee, flood_mask)
                f_area = get_area_stats(geom_ee, flood_mask)
                
                # Détails infra pour ce polygone
                if not impacted_infra.empty:
                    infra_in_poly = impacted_infra[impacted_infra.intersects(row.geometry)]
                    counts = infra_in_poly.get('amenity', infra_in_poly.get('building', 'Inconnu')).fillna('Résidentiel').value_counts().to_dict()
                    n_total_infra = len(infra_in_poly)
                else:
                    counts = {}
                    n_total_infra = 0
                
                temp_list.append({
                    'name': row.get('NAME_2', row.get('NAME_1', st.session_state.zone_name)),
                    'pop_total': t_pop,
                    'pop_exposed': e_pop,
                    'flood_ha': round(f_area, 2),
                    'n_infra': n_total_infra,
                    'infra_details': counts
                })
            
            # Agrégation globale pour les KPIs
            total_pop_exposed = sum(d['pop_exposed'] for d in temp_list)
            total_pop_all = sum(d['pop_total'] for d in temp_list)
            total_flood_ha = sum(d['flood_ha'] for d in temp_list)
            
            # --- SECTION 1: INDICATEURS CLÉS ---
            st.subheader("📊 Indicateurs de Risque")
            k1, k2, k3, k4 = st.columns(4)
            
            with k1:
                st.metric("Population Totale", f"{total_pop_all:,}")
            with k2:
                perc_pop = (total_pop_exposed / total_pop_all * 100) if total_pop_all > 0 else 0
                st.metric("Population Exposée", f"{total_pop_exposed:,}", f"{perc_pop:.1f}%", delta_color="inverse")
            with k3:
                st.metric("Surface Inondée", f"{total_flood_ha:,.2f} ha")
            with k4:
                impact_rate = (len(impacted_infra) / len(buildings) * 100) if len(buildings) > 0 else 0
                st.metric("Infrastructures Impactées", len(impacted_infra), f"{impact_rate:.1f}%", delta_color="inverse")

            # --- SECTION 2: GRAPHIQUES ---
            c1, c2 = st.columns([1, 1])
            
            with c1:
                if not impacted_infra.empty:
                    impacted_infra['type_clean'] = impacted_infra.get('amenity', impacted_infra.get('building', 'Inconnu')).fillna('Bâtiment')
                    type_counts = impacted_infra['type_clean'].value_counts().reset_index()
                    type_counts.columns = ['Type', 'Nombre']
                    
                    fig = px.pie(type_counts, values='Nombre', names='Type', hole=0.5,
                                 title="Répartition des Bâtiments Impactés",
                                 color_discrete_sequence=px.colors.sequential.Reds_r)
                    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucun impact infrastructurel détecté.")

            with c2:
                # Gauge de dangerosité basée sur le pourcentage de population
                fig_gauge = px.bar(x=[perc_pop], y=["Alerte"], orientation='h', range_x=[0, 100],
                                  title="Proportion de Population Affectée (%)",
                                  color=[perc_pop], color_continuous_scale="Reds")
                fig_gauge.update_layout(height=180, margin=dict(t=40, b=0, l=0, r=0), xaxis_title="% Exposé")
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.write(f"🛣️ **Réseau routier :** {len(routes)} segments analysés.")

            # --- SECTION 3: CARTOGRAPHIE ---
            st.subheader("🗺️ Visualisation Spatiale")
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            
            if flood_mask:
                try:
                    map_id = flood_mask.getMapId({'palette': ['#00bfff']})
                    folium.TileLayer(
                        tiles=map_id['tile_fetcher'].url_format,
                        attr='GEE', name='Masque Inondation', overlay=True, opacity=0.7
                    ).add_to(m)
                except: pass

            folium.GeoJson(st.session_state.selected_zone, name="Emprise d'étude", 
                           style_function=lambda x: {'fillColor': 'none', 'color': 'orange', 'weight': 2}).add_to(m)

            if not routes.empty:
                folium.GeoJson(routes, name="Réseau routier", style_function=lambda x: {'color':'#555','weight':1}).add_to(m)
            
            if not impacted_infra.empty:
                folium.GeoJson(
                    impacted_infra,
                    name="Impacts Infrastructurels",
                    style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.8},
                    tooltip=folium.GeoJsonTooltip(fields=['type_clean', 'name'], aliases=['Type:', 'Nom:'])
                ).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600, key="result_map")
            
            # --- SECTION 4: TABLEAU RÉCAPITULATIF ---
            st.subheader("📋 Bilan par Secteur")
            res_df = pd.DataFrame(temp_list)
            st.dataframe(res_df.drop(columns=['infra_details']), use_container_width=True)
    else:
        st.info("Veuillez cliquer sur 'Analyser' pour charger les données GEE et OSM.")
else:
    st.info("💡 Sélectionnez une zone dans la barre latérale pour commencer.")
    m_default = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    st_folium(m_default, width="100%", height=500, key="default_map")
