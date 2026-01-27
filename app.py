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

# Initialisation GEE sécurisée pour Streamlit Cloud
def init_gee():
    try:
        # Vérification si déjà initialisé
        ee.Initialize()
        return True
    except Exception:
        try:
            # Tentative avec les secrets Streamlit
            if "gee_service_account" in st.secrets:
                key_dict = json.loads(st.secrets["gee_service_account"])
                credentials = ee.ServiceAccountCredentials(
                    key_dict["client_email"], 
                    key_data=json.dumps(key_dict)
                )
                ee.Initialize(credentials)
                return True
            else:
                st.error("Secret 'gee_service_account' manquant dans la configuration.")
                return False
        except Exception as e:
            st.error(f"Échec de l'initialisation GEE : {e}")
            return False

gee_available = init_gee()

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

        # Différence logarithmique (méthode robuste)
        diff = flood.subtract(ref)
        flood_mask = diff.lt(threshold)

        # Filtre pente (SRTM) pour éliminer les ombres radar
        slope = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003")).select('slope')
        flood_mask = flood_mask.updateMask(slope.lt(5))

        return flood_mask.selfMask().clip(aoi)
    except Exception as e:
        st.error(f"Erreur de détection radar : {e}")
        return None

def get_pop_stats(aoi_ee, flood_mask):
    if not gee_available: return 0, 0, 0
    try:
        # WorldPop 100m
        pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").median().clip(aoi_ee)
        
        # Population Totale
        total_pop = pop_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e13
        ).get('population').getInfo()
        total_pop = int(total_pop) if total_pop else 0
        
        # Population Impactée
        exposed_pop = 0
        if flood_mask:
            exposed_pop = pop_img.updateMask(flood_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi_ee,
                scale=100,
                maxPixels=1e13
            ).get('population').getInfo()
            exposed_pop = int(exposed_pop) if exposed_pop else 0
            
        perc = (exposed_pop / total_pop * 100) if total_pop > 0 else 0
        return total_pop, exposed_pop, perc
    except:
        return 0, 0, 0

def get_climate_data(aoi_ee, start, end):
    try:
        # Utilisation de NASA POWER API via centroid (plus rapide que GEE pour du ponctuel)
        c = aoi_ee.centroid().getInfo()["coordinates"]
        s_date = start.replace("-", "")
        e_date = end.replace("-", "")
        
        url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
               f"latitude={c[1]}&longitude={c[0]}&start={s_date}&end={e_date}"
               f"&parameters=PRECTOTCORR,T2M&community=AG&format=JSON")
        
        resp = requests.get(url).json()
        params = resp["properties"]["parameter"]
        
        dates = list(params["PRECTOTCORR"].keys())
        precip = list(params["PRECTOTCORR"].values())
        temp = list(params["T2M"].values())
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates, format='%Y%m%d'),
            'precip': precip,
            'temp': temp
        })
        return df
    except:
        return None

def get_osm_assets(_gdf):
    try:
        poly = _gdf.unary_union
        # Extraction OSM simplifiée via OSMnx
        try: 
            b = ox.features_from_polygon(poly, tags={'building': True})
            b = b[b.geometry.type.isin(['Polygon', 'MultiPolygon'])].reset_index().clip(_gdf)
        except: b = gpd.GeoDataFrame()
        
        try:
            graph = ox.graph_from_polygon(poly, network_type='all')
            r = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index().clip(_gdf)
        except: r = gpd.GeoDataFrame()
        
        return b, r
    except: return gpd.GeoDataFrame(), gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE
# ═════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🌍 Paramètres de la zone")
    mode = st.selectbox("Sélection", ["GADM", "Dessin", "Upload"])

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
        m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6)
        Draw(export=False).add_to(m_draw)
        res_draw = st_folium(m_draw, width=250, height=250, key="sidebar_draw")
        if res_draw and res_draw.get('last_active_drawing'):
            geom = shape(res_draw['last_active_drawing']['geometry'])
            st.session_state.selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
            st.session_state.zone_name = "Zone dessinée"

    elif mode == "Upload":
        file = st.file_uploader("Fichier Géo", type=['geojson', 'kml'])
        if file:
            st.session_state.selected_zone = gpd.read_file(file).to_crs(4326)
            st.session_state.zone_name = "Zone importée"

    st.subheader("📅 Périodes d'analyse")
    d_ref = st.date_input("Référence (Sec)", [datetime(2023, 1, 1), datetime(2023, 3, 30)])
    d_flood = st.date_input("Analyse (Pluies)", [datetime(2024, 8, 1), datetime(2024, 10, 30)])

st.title(f"🌊 FloodWatch WA : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE D'IMPACT", type="primary", use_container_width=True):
        if not gee_available:
            st.error("Google Earth Engine n'est pas disponible. Vérifiez vos secrets.")
        else:
            with st.spinner("Calcul des impacts spatiaux (Radar + Population)..."):
                aoi_ee = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
                
                # 1. Détection Inondation
                flood_img = detect_flood(
                    aoi_ee, 
                    str(d_ref[0]), str(d_ref[1]), 
                    str(d_flood[0]), str(d_flood[1])
                )
                
                # 2. Statistiques Population
                t_pop, e_pop, p_pop = get_pop_stats(aoi_ee, flood_img)
                
                # 3. Infrastructures OSM
                bld, rts = get_osm_assets(st.session_state.selected_zone)
                
                # 4. Données Climatiques
                df_clim = get_climate_data(aoi_ee, str(d_flood[0]), str(d_flood[1]))
                
                # 5. Préparation Carte
                mask_id = flood_img.getMapId({'palette': ['#00BFFF']}) if flood_img else None
                
                # 6. Surface Inondée (Ha)
                area_ha = 0
                if flood_img:
                    area_px = flood_img.multiply(ee.Image.pixelArea()).reduceRegion(
                        reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=10, maxPixels=1e9
                    ).get('flood').getInfo()
                    area_ha = (area_px / 10000) if area_px else 0

                st.session_state.results = {
                    't_pop': t_pop, 'e_pop': e_pop, 'p_pop': p_pop,
                    'area': area_ha, 'bld_count': len(bld), 'rts_count': len(rts),
                    'mask_id': mask_id, 'df_clim': df_clim,
                    'b_geo': bld.to_json() if not bld.empty else None,
                    'r_geo': rts.to_json() if not rts.empty else None
                }

    if st.session_state.results:
        res = st.session_state.results
        
        # Dashboard
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Population Impactée", f"{res['e_pop']:,}", f"{res['p_pop']:.1f}%")
        m2.metric("Surface Inondée", f"{res['area']:.1f} ha")
        m3.metric("Bâtiments", res['bld_count'])
        m4.metric("Routes (Segments)", res['rts_count'])

        tab1, tab2 = st.tabs(["🗺️ Cartographie d'Urgence", "📊 Analyse Climatique"])
        
        with tab1:
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            
            # Masque Inondation
            if res['mask_id']:
                folium.TileLayer(
                    tiles=res['mask_id']['tile_fetcher'].url_format, 
                    attr='GEE Sentinel-1', name='Zone Inondée (Radar)', overlay=True
                ).add_to(m)
            
            # Infrastructures
            if res['b_geo']:
                folium.GeoJson(res['b_geo'], name="Bâtiments (OSM)", style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 1}).add_to(m)
            if res['r_geo']:
                folium.GeoJson(res['r_geo'], name="Routes (OSM)", style_function=lambda x: {'color': 'orange', 'weight': 2}).add_to(m)
                
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600)

        with tab2:
            if res['df_clim'] is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=res['df_clim']['date'], y=res['df_clim']['precip'], name="Pluie (mm)"))
                fig.add_trace(go.Scatter(x=res['df_clim']['date'], y=res['df_clim']['temp'], name="Temp (°C)", yaxis="y2"))
                fig.update_layout(title="Précipitations et Température (NASA POWER)", 
                                  yaxis2=dict(overlaying='y', side='right'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Données climatiques indisponibles pour cette zone.")
else:
    st.info("Veuillez définir une zone d'étude dans la barre latérale.")
