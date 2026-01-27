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
    page_title="FloodWatch WA Pro", 
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

# Initialisation du state
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
        def get_collection(start, end):
            # Filtrage strict par les limites du polygone (AOI)
            return (ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filterBounds(aoi)
                    .filterDate(start, end)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                    .select("VV"))

        s1_ref = get_collection(ref_start, ref_end).median().clip(aoi)
        s1_crisis = get_collection(flood_start, flood_end).reduce(ee.Reducer.percentile([10])).rename('VV').clip(aoi)
        
        ref_db = ee.Image(10).multiply(s1_ref.log10())
        crisis_db = ee.Image(10).multiply(s1_crisis.log10())
        
        # Détection de changement (baisse de signal = eau potentielle)
        flood_raw = ref_db.subtract(crisis_db).gt(threshold_db).rename('flood')
        
        # Masquage des eaux permanentes
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        mask_not_permanent = jrc.lt(80).unmask(1)
        
        # Masquage pente (l'eau ne stagne pas sur les fortes pentes)
        slope = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003")).select('slope')
        mask_flat = slope.lt(5).unmask(1)
        
        flood_final = flood_raw.updateMask(mask_not_permanent).updateMask(mask_flat).clip(aoi)
        
        # Filtrage morphologique
        kernel = ee.Kernel.circle(radius=1)
        flood_final = flood_final.focal_mode(kernel=kernel, iterations=1)
        
        # Filtrage des petits objets (bruit)
        connected = flood_final.connectedPixelCount(25) 
        flood_final = flood_final.updateMask(connected.gte(min_pixels))
        
        return flood_final.selfMask()
    except Exception as e:
        st.error(f"Erreur GEE Radar : {str(e)}")
        return None

def get_pop_data(aoi_ee, flood_mask):
    if not gee_available: return 0, 0, 0
    try:
        # WorldPop 100m - Mosaic et Clip par AOI d'abord
        pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2021-01-01').mosaic().clip(aoi_ee)
        
        # Population totale dans le polygone
        total_pop_val = pop_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9
        ).get('population').getInfo()
        total_pop = int(total_pop_val) if total_pop_val is not None else 0
        
        exposed_pop = 0
        if flood_mask:
            # Intersection de la population avec le masque d'inondation
            exposed_pop_val = pop_img.updateMask(flood_mask).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9
            ).get('population').getInfo()
            exposed_pop = int(exposed_pop_val) if exposed_pop_val is not None else 0
            
        perc = (exposed_pop / total_pop * 100) if total_pop > 0 else 0
        return total_pop, exposed_pop, perc
    except: return 0, 0, 0

def get_climate_data(aoi_ee, start, end):
    if not gee_available: return None
    try:
        # CHIRPS Daily
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi_ee).filterDate(start, end).select('precipitation')
        
        def get_daily_sum(img):
            d = img.date().format('YYYY-MM-DD')
            val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi_ee, scale=5000).get('precipitation')
            return ee.Feature(None, {'date': d, 'precip': val})
        
        stats = chirps.map(get_daily_sum).getInfo()
        df = pd.DataFrame([f['properties'] for f in stats])
        if df.empty: return None
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except: return None

def get_osm_assets(_gdf):
    try:
        poly = _gdf.unary_union
        # Bâtiments avec tags types
        try: 
            b = ox.features_from_polygon(poly, tags={'building': True})
            b = b[b.geometry.type.isin(['Polygon', 'MultiPolygon'])].reset_index().clip(_gdf)
        except: b = gpd.GeoDataFrame()
        
        # Routes
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
    st.header("🗺️ Configuration")
    mode = st.selectbox("Sélection Zone", ["GADM", "Dessin", "Upload"])

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

    st.subheader("📅 Périodes")
    d_dry = st.date_input("Référence (Sec)", [datetime(2023, 1, 1), datetime(2023, 4, 30)])
    d_wet = st.date_input("Analyse (Humide)", [datetime(2024, 8, 1), datetime(2024, 10, 30)])
    
    st.subheader("⚙️ Radar")
    sens = st.slider("Sensibilité (dB)", 0.5, 5.0, 1.25)
    noise = st.slider("Filtre Bruit (px)", 1, 30, 10)
    
    with st.expander("ℹ️ Aide aux paramètres"):
        st.markdown("""
        **Sensibilité (dB)** : Seuil de détection. Une valeur de 1.25 est standard. Si la zone est très sèche, baissez vers 0.8.
        **Filtre Bruit** : Taille du noyau pour éliminer les pixels isolés. Augmentez si vous voyez trop de "points" bleus épars.
        """)

st.title(f"🌊 FloodWatch Pro : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE COMPLÈTE", type="primary", use_container_width=True):
        with st.spinner("Calculs spatiaux multi-sources en cours..."):
            aoi_ee = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
            
            # 1. Radar
            flood_mask = advanced_flood_detection(aoi_ee, str(d_dry[0]), str(d_dry[1]), str(d_wet[0]), str(d_wet[1]), sens, noise)
            
            # 2. Population 100m
            t_pop, e_pop, p_pop = get_pop_data(aoi_ee, flood_mask)
            
            # 3. Infrastructures OSM
            bld, rts = get_osm_assets(st.session_state.selected_zone)
            
            # 4. Climat
            df_rain = get_climate_data(aoi_ee, str(d_wet[0]), str(d_wet[1]))
            
            # 5. Surface
            area_ha = 0
            mask_id = None
            if flood_mask:
                area_val = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=10, maxPixels=1e9
                ).get('flood').getInfo()
                area_ha = (float(area_val) / 10000) if area_val is not None else 0
                mask_id = flood_mask.getMapId({'palette': ['#00BFFF']})

            st.session_state.results = {
                't_pop': int(t_pop), 
                'e_pop': int(e_pop), 
                'p_pop': float(p_pop),
                'area': float(area_ha),
                'b_count': len(bld), 
                'r_count': len(rts),
                'mask_id': mask_id,
                'df_rain': df_rain,
                'b_geo': bld.to_json() if not bld.empty else None,
                'r_geo': rts.to_json() if not rts.empty else None,
                'b_data': bld[['building', 'geometry']].drop(columns='geometry').value_counts('building').to_dict() if not bld.empty else {}
            }

    if st.session_state.results:
        res = st.session_state.results
        
        # Dashboard
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Population Exposée", f"{res['e_pop']:,}", f"{res['p_pop']:.1f}% de {res['t_pop']:,} habitants")
        m2.metric("Surface Inondée", f"{res['area']:.2f} ha")
        m3.metric("Bâtiments à risque", f"{res['b_count']}")
        m4.metric("Segments routiers", f"{res['r_count']}")

        tab1, tab2, tab3 = st.tabs(["🗺️ Carte Interactive", "📊 Statistiques & Climat", "📥 Exports"])
        
        with tab1:
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            
            # Zone d'étude
            folium.GeoJson(st.session_state.selected_zone, name="AOI", style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2}).add_to(m)
            
            # Inondation GEE
            if res['mask_id']:
                folium.TileLayer(tiles=res['mask_id']['tile_fetcher'].url_format, attr='GEE', name='Inondation Satellite', overlay=True).add_to(m)
            
            # Infrastructures
            if res['b_geo']:
                folium.GeoJson(res['b_geo'], name="Bâtiments", style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 1}).add_to(m)
            if res['r_geo']:
                folium.GeoJson(res['r_geo'], name="Routes", style_function=lambda x: {'color': 'orange', 'weight': 2}).add_to(m)
                
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600, key="map_final")

        with tab2:
            col_a, col_b = st.columns(2)
            
            with col_a:
                if res['df_rain'] is not None:
                    fig = px.bar(res['df_rain'], x='date', y='precip', title="Cumul Pluviométrique Journalier (CHIRPS)")
                    fig.update_layout(yaxis_title="Précipitations (mm/jour)", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Données pluviométriques non disponibles.")

            with col_b:
                if res['b_data']:
                    fig_b = px.pie(names=list(res['b_data'].keys()), values=list(res['b_data'].values()), title="Répartition des Bâtiments par Type")
                    st.plotly_chart(fig_b, use_container_width=True)
                else:
                    st.info("Aucune donnée typologique sur les bâtiments.")

        with tab3:
            st.subheader("Rapport d'analyse")
            export_df = pd.DataFrame([{
                'Date_Analyse': datetime.now().strftime("%Y-%m-%d"),
                'Zone': st.session_state.zone_name,
                'Pop_Totale': res['t_pop'],
                'Pop_Exposee': res['e_pop'],
                'Ratio_Exposition_%': res['p_pop'],
                'Surface_Inondee_Ha': res['area'],
                'Nb_Batiments': res['b_count'],
                'Nb_Routes': res['r_count']
            }])
            st.dataframe(export_df, use_container_width=True)
            st.download_button("📥 Télécharger Rapport CSV", export_df.to_csv(index=False), "flood_report.csv", "text/csv")
            
else:
    st.info("Sélectionnez une zone d'étude et lancez l'analyse.")
