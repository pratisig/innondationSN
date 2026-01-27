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
st.set_page_config(page_title="FloodWatch WA - Tableau de Bord", layout="wide")

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
    """Génère un masque d'inondation Sentinel-1 en excluant l'eau permanente"""
    if not gee_available: return None
    try:
        # 1. Données Sentinel-1 (Période d'inondation)
        s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterBounds(aoi_ee)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .filterDate(start_flood, end_flood)
                  .select('VV'))
        
        img_flood = s1_col.min().clip(aoi_ee).focal_median(50, 'circle', 'meters')

        # 2. Référence historique (Saison sèche ou médiane annuelle)
        # On utilise une médiane sur une période stable pour comparer
        img_ref = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterBounds(aoi_ee)
                   .filterDate("2023-01-01", "2023-04-30")
                   .select('VV').median().clip(aoi_ee).focal_median(50, 'circle', 'meters'))

        # 3. Calcul de la différence
        diff = img_ref.subtract(img_flood)
        flood_raw = diff.gt(threshold)

        # 4. Exclusion des eaux permanentes (JRC Global Surface Water)
        # On exclut les zones où l'eau est présente plus de 80% du temps
        permanent_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        water_mask = permanent_water.gt(80).clip(aoi_ee)
        
        # Le masque final : Zones inondées ET qui ne sont pas de l'eau permanente
        final_flood = flood_raw.where(water_mask, 0).selfMask()
        
        return final_flood.rename('flood')
    except: return None

def get_population_stats(aoi_ee, flood_mask):
    """Calcule la population via WorldPop (Découpage strict par AOI)"""
    if not gee_available: return 0, 0
    try:
        # Mosaïque 2020 pour la couverture. Le .clip(aoi_ee) assure le découpage spatial.
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01', '2021-01-01') \
                        .mosaic().clip(aoi_ee)
        
        # Population totale (le paramètre geometry assure que seul le polygone est compté)
        stats_total = pop_dataset.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e9
        )
        total_pop = stats_total.get('population').getInfo() or 0
        
        exposed_pop = 0
        if flood_mask:
            # Population dans les zones inondées uniquement
            stats_exposed = pop_dataset.updateMask(flood_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi_ee,
                scale=100,
                maxPixels=1e9
            )
            exposed_pop = stats_exposed.get('population').getInfo() or 0
            
        return int(total_pop), int(exposed_pop)
    except:
        return 0, 0

def get_area_stats(aoi_ee, flood_mask):
    """Superficie inondée en hectares"""
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
    """Bâtiments et routes via OSM"""
    if _gdf_aoi is None or _gdf_aoi.empty: return gpd.GeoDataFrame(), gpd.GeoDataFrame()
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

def analyze_impacted_infra(flood_mask, buildings_gdf):
    """Identification des bâtiments touchés"""
    if flood_mask is None or buildings_gdf.empty: return gpd.GeoDataFrame()
    try:
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
# 3. INTERFACE UTILISATEUR
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🗺️ 1. Zone d'Étude")
mode = st.sidebar.radio("Méthode de sélection :", ["Liste Administrative", "Dessiner sur Carte", "Importer Fichier"])

if 'selected_zone' not in st.session_state: st.session_state.selected_zone = None
if 'zone_name' not in st.session_state: st.session_state.zone_name = "Zone personnalisée"
if 'analysis_triggered' not in st.session_state: st.session_state.analysis_triggered = False

if mode == "Liste Administrative":
    countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
    c_choice = st.sidebar.selectbox("Pays", list(countries.keys()))
    level = st.sidebar.slider("Niveau Admin", 0, 5, 2)
    gdf_base = load_gadm(countries[c_choice], level)
    if gdf_base is not None:
        col = f"NAME_{level}" if level > 0 else "COUNTRY"
        choice = st.sidebar.selectbox("Subdivision", sorted(gdf_base[col].dropna().unique()))
        new_zone = gdf_base[gdf_base[col] == choice].copy()
        if st.session_state.zone_name != choice:
            st.session_state.selected_zone = new_zone
            st.session_state.zone_name = choice
            st.session_state.analysis_triggered = False

elif mode == "Dessiner sur Carte":
    st.sidebar.info("Utilisez les outils de dessin à droite.")
    m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    Draw(export=False).add_to(m_draw)
    with st.sidebar:
        out = st_folium(m_draw, width=250, height=250, key="draw_static")
        if out and out.get('last_active_drawing'):
            geom = shape(out['last_active_drawing']['geometry'])
            st.session_state.selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
            st.session_state.zone_name = "Zone Dessinée"
            st.session_state.analysis_triggered = False

st.sidebar.header("📅 2. Paramètres")
col_d1, col_d2 = st.sidebar.columns(2)
start_f = col_d1.date_input("Début inondation", datetime(2024, 8, 1))
end_f = col_d2.date_input("Fin inondation", datetime(2024, 9, 30))
threshold_val = st.sidebar.slider("Sensibilité détection (dB)", 2.0, 8.0, 4.0)

# ═════════════════════════════════════════════════════════════════
# 4. LOGIQUE PRINCIPALE
# ═════════════════════════════════════════════════════════════════

st.title(f"🌊 FloodWatch : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE D'IMPACT", type="primary", use_container_width=True):
        st.session_state.analysis_triggered = True

    if st.session_state.analysis_triggered:
        with st.spinner("Calculs en cours... (Exclusion eaux permanentes activée)"):
            # A. GEE
            aoi_ee_global = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
            flood_mask = get_flood_mask(aoi_ee_global, str(start_f), str(end_f), threshold_val)
            
            # B. OSM & Impacts
            buildings, routes = get_osm_data(st.session_state.selected_zone)
            impacted_infra = analyze_impacted_infra(flood_mask, buildings)
            
            # C. Stats par subdivision
            temp_list = []
            for idx, row in st.session_state.selected_zone.iterrows():
                geom_ee = ee.Geometry(mapping(row.geometry))
                t_pop, e_pop = get_population_stats(geom_ee, flood_mask)
                f_area = get_area_stats(geom_ee, flood_mask)
                
                n_infra = 0
                if not impacted_infra.empty:
                    n_infra = len(impacted_infra[impacted_infra.intersects(row.geometry)])
                
                temp_list.append({
                    'Secteur': row.get('NAME_2', row.get('NAME_1', st.session_state.zone_name)),
                    'Pop. Totale': t_pop,
                    'Pop. Exposée': e_pop,
                    'Inondation (ha)': round(f_area, 2),
                    'Bâtiments Touchés': n_infra
                })
            
            # KPIs Globaux
            total_pop_exposed = sum(d['Pop. Exposée'] for d in temp_list)
            total_pop_all = sum(d['Pop. Totale'] for d in temp_list)
            total_flood_ha = sum(d['Inondation (ha)'] for d in temp_list)
            
            # --- SECTION 1: MÉTRIQUES ---
            st.subheader("📊 Bilan Global de l'Évènement")
            k1, k2, k3, k4 = st.columns(4)
            with k1: st.metric("Population Totale", f"{total_pop_all:,}")
            with k2: 
                perc_pop = (total_pop_exposed / total_pop_all * 100) if total_pop_all > 0 else 0
                st.metric("Population Exposée", f"{total_pop_exposed:,}")
            with k3: st.metric("Surface Inondée", f"{total_flood_ha:,.1f} ha")
            with k4: st.metric("Bâtiments Impactés", len(impacted_infra))

            # --- SECTION 2: GRAPHIQUES ---
            c1, c2 = st.columns([1, 1])
            with c1:
                if not impacted_infra.empty:
                    impacted_infra['Type'] = impacted_infra.get('amenity', impacted_infra.get('building', 'Autre')).fillna('Résidentiel')
                    fig = px.pie(impacted_infra, names='Type', hole=0.5, title="Typologie des Infrastructures Touchées")
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                # Barre de proportion sans flèches delta
                df_gauge = pd.DataFrame({"État": ["Exposé", "Sain"], 
                                         "Valeur": [perc_pop, 100-perc_pop]})
                fig_bar = px.bar(df_gauge, x="Valeur", y="État", orientation='h', 
                                 color="État", color_discrete_map={"Exposé": "#d62728", "Sain": "#2ca02c"},
                                 title="Proportion de la Population en Zone Inondable (%)")
                st.plotly_chart(fig_bar, use_container_width=True)

            # --- SECTION 3: CARTE ---
            st.subheader("🗺️ Carte des Risques (Bleu = Eau Temporaire)")
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            
            if flood_mask:
                map_id = flood_mask.getMapId({'palette': ['#00bfff']})
                folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='GEE', name='Inondations').add_to(m)

            folium.GeoJson(st.session_state.selected_zone, name="Zone", 
                           style_function=lambda x: {'fillColor': 'none', 'color': 'orange'}).add_to(m)

            if not impacted_infra.empty:
                folium.GeoJson(impacted_infra, name="Impacts",
                               style_function=lambda x: {'fillColor': 'red', 'color': 'red'}).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600, key="map_res")
            
            st.subheader("📋 Détails par Subdivision")
            st.table(pd.DataFrame(temp_list))
