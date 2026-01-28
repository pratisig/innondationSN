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
import requests
import numpy as np

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & INITIALISATION
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FloodWatch WA Pro", 
    page_icon="🌊",
    layout="wide"
)

# Initialisation GEE optimisée avec gestion du Project ID
@st.cache_resource
def init_gee_singleton():
    """Initialise Google Earth Engine avec gestion robuste des secrets et du projet."""
    try:
        # Recherche des secrets dans Streamlit Cloud
        secret_key = "gee_service_account"
        if secret_key in st.secrets:
            try:
                credentials_info = st.secrets[secret_key]
                
                # Streamlit peut fournir le secret comme dict ou comme chaîne JSON
                if isinstance(credentials_info, str):
                    credentials_info = json.loads(credentials_info)
                
                # Extraction du Project ID (Crucial pour les nouvelles API Cloud)
                project_id = credentials_info.get('project_id')
                
                # Conversion des secrets en credentials GEE
                credentials = ee.ServiceAccountCredentials(
                    credentials_info['client_email'],
                    key_data=json.dumps(credentials_info)
                )
                
                # Initialisation avec le projet spécifié
                if project_id:
                    ee.Initialize(credentials, project=project_id)
                else:
                    ee.Initialize(credentials)
                return True
            except Exception as e:
                st.error(f"Erreur d'authentification GEE : {e}")
                return False
        else:
            # Tentative d'initialisation par défaut (environnement local)
            ee.Initialize()
            return True
    except Exception as e:
        st.warning(f"GEE non disponible : {e}")
        return False

gee_available = init_gee_singleton()

if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = None
if 'zone_name' not in st.session_state:
    st.session_state.zone_name = "Zone non définie"

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS DE TRAITEMENT (OPTIMISÉES)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return None

def detect_flood(aoi, d1, d2, d3, d4, threshold=-1.25):
    if not gee_available: return None
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .select("VV"))

        ref = s1.filterDate(d1, d2).median()
        flood = s1.filterDate(d3, d4).median()

        diff = flood.subtract(ref)
        # On nomme explicitement la bande pour le calcul de stats plus tard
        flood_mask = diff.lt(threshold).rename('flood_mask')

        slope = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003")).select('slope')
        flood_mask = flood_mask.updateMask(slope.lt(5))

        return flood_mask.selfMask().clip(aoi)
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def get_pop_stats_cached(aoi_json):
    if not gee_available: return 0
    try:
        aoi_ee = ee.Geometry(aoi_json)
        pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").median().clip(aoi_ee)
        
        # Utilisation d'une extraction de dictionnaire plus sûre
        stats = pop_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e13
        ).getInfo()
        
        total_pop = list(stats.values())[0] if stats else 0
        return int(total_pop) if total_pop else 0
    except Exception as e:
        return 0

@st.cache_data(show_spinner=False)
def get_climate_data(centroid_coords, start, end):
    try:
        s_date = start.replace("-", "")
        e_date = end.replace("-", "")
        url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
               f"latitude={centroid_coords[1]}&longitude={centroid_coords[0]}&start={s_date}&end={e_date}"
               f"&parameters=PRECTOTCORR,T2M&community=AG&format=JSON")
        
        resp = requests.get(url, timeout=10).json()
        params = resp["properties"]["parameter"]
        df = pd.DataFrame({
            'date': pd.to_datetime(list(params["PRECTOTCORR"].keys()), format='%Y%m%d'),
            'precip': list(params["PRECTOTCORR"].values()),
            'temp': list(params["T2M"].values())
        })
        return df.to_json()
    except:
        return None

@st.cache_data(show_spinner=False)
def get_osm_assets_cached(aoi_json):
    try:
        geom = shape(aoi_json)
        # Bâtiments
        try: 
            b = ox.features_from_polygon(geom, tags={'building': True})
            b = b[b.geometry.type.isin(['Polygon', 'MultiPolygon'])].reset_index()
            b_json = b.to_json()
        except: b_json = None
        
        # Routes
        try:
            graph = ox.graph_from_polygon(geom, network_type='all')
            r = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
            r_json = r.to_json()
        except: r_json = None
        
        return b_json, r_json
    except:
        return None, None

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE
# ═════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🌍 Paramètres de la zone")
    mode = st.selectbox("Sélection", ["GADM", "Dessin", "Upload"], key="select_mode")

    if mode == "GADM":
        countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
        c_choice = st.selectbox("Pays", list(countries.keys()))
        level = st.slider("Niveau Administratif", 0, 4, 2)
        gdf_base = load_gadm(countries[c_choice], level)
        if gdf_base is not None:
            col = f"NAME_{level}" if level > 0 else "COUNTRY"
            choices = st.multiselect("Unités", sorted(gdf_base[col].unique()))
            if choices:
                st.session_state.selected_zone = gdf_base[gdf_base[col].isin(choices)].copy()
                st.session_state.zone_name = ", ".join(choices)

    elif mode == "Dessin":
        st.write("Utilisez les outils à gauche de la carte pour dessiner.")
        m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6)
        Draw(export=False, draw_options={'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False}).add_to(m_draw)
        res_draw = st_folium(m_draw, width=250, height=250, key="sidebar_draw")
        if res_draw and res_draw.get('last_active_drawing'):
            geom = shape(res_draw['last_active_drawing']['geometry'])
            st.session_state.selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
            st.session_state.zone_name = "Zone dessinée"

    elif mode == "Upload":
        file = st.file_uploader("Fichier Géo", type=['geojson', 'kml'], key="uploader")
        if file:
            st.session_state.selected_zone = gpd.read_file(file).to_crs(4326)
            st.session_state.zone_name = "Zone importée"

    st.subheader("📅 Périodes d'analyse")
    d_ref = st.date_input("Référence (Sec)", [datetime(2023, 1, 1), datetime(2023, 3, 30)], key="date_ref")
    d_flood = st.date_input("Analyse (Pluies)", [datetime(2024, 8, 1), datetime(2024, 10, 30)], key="date_flood")

st.title(f"🌊 FloodWatch WA : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE D'IMPACT", type="primary", use_container_width=True):
        if not gee_available:
            st.error("L'analyse satellite (GEE) est désactivée. Vérifiez vos secrets 'gee_service_account' et assurez-vous que 'project_id' y figure.")
        else:
            with st.spinner("Analyse en cours..."):
                geom_union = st.session_state.selected_zone.unary_union
                aoi_json = mapping(geom_union)
                aoi_ee = ee.Geometry(aoi_json)
                
                # 1. Détection Inondation
                flood_img = detect_flood(aoi_ee, str(d_ref[0]), str(d_ref[1]), str(d_flood[0]), str(d_flood[1]))
                
                # 2. Population Totale
                t_pop = get_pop_stats_cached(aoi_json)
                
                # 3. Population Impactée
                e_pop = 0
                if flood_img:
                    try:
                        pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").median().clip(aoi_ee)
                        e_pop_stats = pop_img.updateMask(flood_img).reduceRegion(
                            reducer=ee.Reducer.sum(), 
                            geometry=aoi_ee, 
                            scale=100, 
                            maxPixels=1e13
                        ).getInfo()
                        e_pop_val = list(e_pop_stats.values())[0] if e_pop_stats else 0
                        e_pop = int(e_pop_val) if e_pop_val else 0
                    except:
                        e_pop = 0
                
                # 4. Infrastructures OSM
                b_json, r_json = get_osm_assets_cached(aoi_json)
                
                # 5. Climat
                centroid = geom_union.centroid
                df_clim_json = get_climate_data([centroid.x, centroid.y], str(d_flood[0]), str(d_flood[1]))
                
                # 6. Surface & MapId
                area_ha = 0
                mask_id = None
                if flood_img:
                    mask_id = flood_img.getMapId({'palette': ['#00BFFF']})
                    # On réduit la région sur la bande 'flood_mask'
                    stats = flood_img.multiply(ee.Image.pixelArea()).reduceRegion(
                        reducer=ee.Reducer.sum(), 
                        geometry=aoi_ee, 
                        scale=10, 
                        maxPixels=1e9
                    ).getInfo()
                    # On récupère la valeur de la première bande disponible
                    area_px = list(stats.values())[0] if stats else 0
                    area_ha = (area_px / 10000) if area_px else 0

                st.session_state.results = {
                    't_pop': t_pop, 'e_pop': e_pop, 'p_pop': (e_pop/t_pop*100) if t_pop > 0 else 0,
                    'area': area_ha, 'mask_id': mask_id, 'df_clim_json': df_clim_json,
                    'b_geo': b_json, 'r_geo': r_json
                }

    if st.session_state.results:
        res = st.session_state.results
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Population Impactée", f"{res['e_pop']:,}", f"{res['p_pop']:.1f}%")
        m2.metric("Surface Inondée", f"{res['area']:.1f} ha")
        b_count = len(json.loads(res['b_geo'])['features']) if res['b_geo'] else 0
        r_count = len(json.loads(res['r_geo'])['features']) if res['r_geo'] else 0
        m3.metric("Bâtiments", b_count)
        m4.metric("Routes (Segments)", r_count)

        tab1, tab2 = st.tabs(["🗺️ Cartographie", "📊 Climat"])
        with tab1:
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            if res['mask_id']:
                folium.TileLayer(tiles=res['mask_id']['tile_fetcher'].url_format, attr='GEE', name='Inondation', overlay=True).add_to(m)
            if res['b_geo']:
                folium.GeoJson(res['b_geo'], name="Bâtiments", style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 1}).add_to(m)
            if res['r_geo']:
                folium.GeoJson(res['r_geo'], name="Routes", style_function=lambda x: {'color': 'orange', 'weight': 2}).add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600, key="main_map")
        with tab2:
            if res['df_clim_json']:
                df_c = pd.read_json(res['df_clim_json'])
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_c['date'], y=df_c['precip'], name="Pluie (mm)"))
                fig.add_trace(go.Scatter(x=df_c['date'], y=df_c['temp'], name="Temp (°C)", yaxis="y2"))
                fig.update_layout(title="Climat (NASA POWER)", yaxis2=dict(overlaying='y', side='right'))
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sélectionnez une zone pour débuter.")
