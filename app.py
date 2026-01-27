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

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & INITIALISATION
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FloodWatch WA Pro", 
    page_icon="🌊",
    layout="wide"
)

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

def advanced_flood_detection(aoi, ref_start, ref_end, flood_start, flood_end, threshold_db=1.25, min_pixels=10):
    if not gee_available: return None
    try:
        def get_collection(start, end):
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
        
        # Détection de l'eau (différence de backscatter)
        flood_raw = ref_db.subtract(crisis_db).gt(threshold_db).rename('flood')
        
        # Filtres topographiques et hydrologiques
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        mask_not_permanent = jrc.lt(80).unmask(1)
        slope = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003")).select('slope')
        mask_flat = slope.lt(5).unmask(1)
        
        flood_final = flood_raw.updateMask(mask_not_permanent).updateMask(mask_flat).clip(aoi)
        
        # Nettoyage du bruit
        kernel = ee.Kernel.circle(radius=1)
        flood_final = flood_final.focal_mode(kernel=kernel, iterations=1)
        connected = flood_final.connectedPixelCount(50) 
        flood_final = flood_final.updateMask(connected.gte(min_pixels))
        
        return flood_final.selfMask()
    except Exception as e:
        st.error(f"Erreur Radar : {str(e)}")
        return None

def get_pop_data(aoi_ee, flood_mask):
    """
    Méthode robuste d'agrégation de population.
    Découpe WorldPop par le polygone, puis multiplie par le masque d'inondation.
    """
    if not gee_available: return 0, 0, 0
    try:
        # Charger WorldPop 100m et assurer la résolution
        pop_col = ee.ImageCollection("WorldPop/GP/100m/pop")
        pop_img = pop_col.filterBounds(aoi_ee).filterDate('2020-01-01', '2021-01-01').mosaic().clip(aoi_ee)
        
        # 1. Population totale dans le polygone
        stats_total = pop_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=100,
            maxPixels=1e9
        ).getInfo()
        total_pop = int(stats_total.get('population', 0)) if stats_total else 0
        
        exposed_pop = 0
        if flood_mask:
            # S'assurer que le masque est binaire (1 pour inondé)
            binary_mask = flood_mask.gt(0).unmask(0)
            
            # Intersection spatiale : on masque l'image de pop par les zones inondées
            pop_exposed_img = pop_img.updateMask(binary_mask)
            
            stats_exposed = pop_exposed_img.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi_ee,
                scale=100,
                maxPixels=1e9
            ).getInfo()
            exposed_pop = int(stats_exposed.get('population', 0)) if stats_exposed else 0
            
        perc = (exposed_pop / total_pop * 100) if total_pop > 0 else 0
        return total_pop, exposed_pop, perc
    except Exception as e:
        st.warning(f"Erreur Population : {str(e)}")
        return 0, 0, 0

def get_climate_data(aoi_ee, start, end):
    if not gee_available: return None
    try:
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
        # Bâtiments
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
    st.header("🗺️ Zone d'Étude")
    mode = st.selectbox("Méthode", ["GADM", "Dessin"])

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

    st.subheader("📅 Dates")
    d_dry = st.date_input("Référence (Sec)", [datetime(2023, 1, 1), datetime(2023, 4, 30)])
    d_wet = st.date_input("Crise (Inondation)", [datetime(2024, 8, 1), datetime(2024, 10, 30)])
    
    st.subheader("⚙️ Paramètres Radar")
    sens = st.slider("Sensibilité (dB)", 0.5, 5.0, 1.25)
    noise = st.slider("Seuil Bruit (px)", 1, 50, 20)

st.title(f"🌊 FloodWatch : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 ANALYSER LA ZONE", type="primary", use_container_width=True):
        with st.spinner("Intersection des couches satellite et population..."):
            aoi_ee = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
            
            # Inondation
            flood_mask = advanced_flood_detection(aoi_ee, str(d_dry[0]), str(d_dry[1]), str(d_wet[0]), str(d_wet[1]), sens, noise)
            
            # Population (Corrigé)
            t_pop, e_pop, p_pop = get_pop_data(aoi_ee, flood_mask)
            
            # Infrastructures
            bld, rts = get_osm_assets(st.session_state.selected_zone)
            
            # Pluviométrie
            df_rain = get_climate_data(aoi_ee, str(d_wet[0]), str(d_wet[1]))
            
            # Surface
            area_ha = 0
            mask_id = None
            if flood_mask:
                area_val = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=10, maxPixels=1e9
                ).get('flood').getInfo()
                area_ha = (float(area_val) / 10000) if area_val is not None else 0
                mask_id = flood_mask.getMapId({'palette': ['#00BFFF']})

            st.session_state.results = {
                't_pop': int(t_pop), 'e_pop': int(e_pop), 'p_pop': float(p_pop),
                'area': float(area_ha), 'b_count': len(bld), 'r_count': len(rts),
                'mask_id': mask_id, 'df_rain': df_rain,
                'b_geo': bld.to_json() if not bld.empty else None,
                'r_geo': rts.to_json() if not rts.empty else None,
                'b_data': bld['building'].value_counts().to_dict() if not bld.empty and 'building' in bld.columns else {}
            }

    if st.session_state.results:
        res = st.session_state.results
        
        # Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Population Impactée", f"{res['e_pop']:,}", f"{res['p_pop']:.1f}% du total")
        col2.metric("Surface Inondée", f"{res['area']:.1f} ha")
        col3.metric("Bâtiments", f"{res['b_count']}")
        col4.metric("Routes (Sgmts)", f"{res['r_count']}")

        tab1, tab2, tab3 = st.tabs(["🗺️ Carte", "📊 Graphiques", "📥 Rapports"])
        
        with tab1:
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            folium.GeoJson(st.session_state.selected_zone, name="Zone", style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2}).add_to(m)
            
            if res['mask_id']:
                folium.TileLayer(tiles=res['mask_id']['tile_fetcher'].url_format, attr='GEE', name='Zones Bleues (Eau)', overlay=True).add_to(m)
            
            if res['b_geo']:
                folium.GeoJson(res['b_geo'], name="Bâtiments", style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 1}).add_to(m)
            if res['r_geo']:
                folium.GeoJson(res['r_geo'], name="Routes", style_function=lambda x: {'color': 'orange', 'weight': 2}).add_to(m)
                
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600, key="map_view")

        with tab2:
            c_rain, c_bld = st.columns(2)
            with c_rain:
                if res['df_rain'] is not None:
                    fig_r = px.bar(res['df_rain'], x='date', y='precip', title="Pluies Journalières (mm)")
                    st.plotly_chart(fig_r, use_container_width=True)
                else: st.info("Pas de données de pluie.")
            with c_bld:
                if res['b_data']:
                    fig_p = px.pie(names=list(res['b_data'].keys()), values=list(res['b_data'].values()), title="Types de Bâtiments")
                    st.plotly_chart(fig_p, use_container_width=True)

        with tab3:
            st.write("### Synthèse des résultats")
            summary = pd.DataFrame([{
                'Paramètre': 'Population Totale', 'Valeur': res['t_pop']},
                {'Paramètre': 'Population Impactée', 'Valeur': res['e_pop']},
                {'Paramètre': 'Surface inondée (ha)', 'Valeur': round(res['area'], 2)},
                {'Paramètre': 'Bâtiments impactés', 'Valeur': res['b_count']},
                {'Paramètre': 'Routes impactées', 'Valeur': res['r_count']
            }])
            st.table(summary)
            st.download_button("📥 Exporter CSV", summary.to_csv(index=False), "rapport_inondation.csv")

else:
    st.info("Veuillez choisir une zone dans le menu de gauche pour commencer.")
