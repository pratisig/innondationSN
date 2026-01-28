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

# Paramètres OSMnx
ox.settings.timeout = 180
ox.settings.use_cache = True

def ensure_gee():
    """Vérifie et initialise GEE avec gestion du projet Cloud."""
    try:
        ee.api.get_project_id()
        return True
    except:
        secret_key = "gee_service_account"
        if secret_key in st.secrets:
            try:
                creds = st.secrets[secret_key]
                if isinstance(creds, str): creds = json.loads(creds)
                project_id = creds.get('project_id')
                ee_creds = ee.ServiceAccountCredentials(creds['client_email'], key_data=json.dumps(creds))
                if project_id:
                    ee.Initialize(ee_creds, project=project_id)
                else:
                    ee.Initialize(ee_creds)
                return True
            except Exception as e:
                st.error(f"Erreur d'initialisation GEE : {e}")
                return False
        return False

gee_available = ensure_gee()

if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = None
if 'zone_name' not in st.session_state:
    st.session_state.zone_name = "Zone non définie"

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS DE TRAITEMENT AVANCÉES
# ═════════════════════════════════════════════════════════════════

def detect_flood_refined(aoi, d1, d2, d3, d4, threshold=-1.25):
    """
    Détection affinée pour éviter la surestimation (méthode de différence de log).
    Intègre un filtre de pente et un masque d'eau permanente.
    """
    if not ensure_gee(): return None
    try:
        # Collection Sentinel-1
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .select("VV"))

        ref = s1.filterDate(d1, d2).median()
        flood = s1.filterDate(d3, d4).median()
        
        # Différence logarithmique
        diff = flood.subtract(ref)
        
        # Masquage initial (seuil radar)
        flood_mask = diff.lt(threshold)
        
        # 1. Filtre de pente (SRTM) : Supprime les ombres radar en zone montagneuse (> 5 degrés)
        slope = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003")).select('slope')
        flood_mask = flood_mask.updateMask(slope.lt(5))
        
        # 2. Masque d'eau permanente (JRC) : Supprime l'eau déjà existante
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        flood_mask = flood_mask.updateMask(jrc.lt(10))
        
        # 3. Nettoyage Morphologique (Suppression du bruit)
        kernel = ee.Kernel.circle(radius=1)
        flood_mask = flood_mask.focal_min(kernel).focal_max(kernel).rename('flood')
        
        return flood_mask.selfMask().clip(aoi)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_pop_stats_cached(aoi_json, flood_mask_ee=None):
    if not ensure_gee(): return 0, 0
    try:
        aoi_ee = ee.Geometry(aoi_json)
        pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").median().clip(aoi_ee)
        
        total_stats = pop_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e13).getInfo()
        total_pop = int(list(total_stats.values())[0]) if total_stats else 0
        
        exposed_pop = 0
        if flood_mask_ee:
            exp_stats = pop_img.updateMask(flood_mask_ee).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e13).getInfo()
            exposed_pop = int(list(exp_stats.values())[0]) if exp_stats else 0
            
        return total_pop, exposed_pop
    except: return 0, 0

@st.cache_data(show_spinner=False)
def get_osm_impact(aoi_json, flood_mask_ee):
    """Analyse d'impact sur les infrastructures OSM."""
    try:
        geom = shape(aoi_json)
        # Téléchargement OSM
        b_raw = ox.features_from_polygon(geom, tags={'building': True})
        r_raw = ox.features_from_polygon(geom, tags={'highway': True})
        
        b_gdf = b_raw[b_raw.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        r_gdf = r_raw[r_raw.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
        
        # Conversion du masque flood_mask en vecteur pour intersection spatiale
        # On utilise une échelle réduite pour la performance
        flood_vec = flood_mask_ee.reduceToVectors(geometry=ee.Geometry(aoi_json), scale=40, maxPixels=1e9)
        flood_geom = shape(flood_vec.geometry().getInfo())
        
        # Intersection spatiale
        b_impact = b_gdf[b_gdf.intersects(flood_geom)].copy()
        r_impact = r_gdf[r_gdf.intersects(flood_geom)].copy()
        
        return b_gdf, b_impact, r_gdf, r_impact
    except:
        return None, None, None, None

@st.cache_data(show_spinner=False)
def get_climate_data(centroid_coords, start, end):
    """Récupère les données NASA POWER pour le graphique ombrothermique."""
    try:
        s_date = start.replace("-", "")
        e_date = end.replace("-", "")
        url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
               f"latitude={centroid_coords[1]}&longitude={centroid_coords[0]}&start={s_date}&end={e_date}"
               f"&parameters=PRECTOTCORR,T2M&community=AG&format=JSON")
        resp = requests.get(url, timeout=15).json()
        params = resp["properties"]["parameter"]
        df = pd.DataFrame({
            'date': pd.to_datetime(list(params["PRECTOTCORR"].keys()), format='%Y%m%d'),
            'Pluie': list(params["PRECTOTCORR"].values()),
            'Température': list(params["T2M"].values())
        }).sort_values('date')
        return df.to_json()
    except: return None

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE & LOGIQUE PRINCIPALE
# ═════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🌍 Paramètres")
    mode = st.selectbox("Sélection Zone", ["GADM", "Dessin", "Upload"])
    
    if mode == "GADM":
        countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
        c_choice = st.selectbox("Pays", list(countries.keys()))
        level = st.slider("Niveau Administratif", 0, 3, 2)
        url_gadm = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{countries[c_choice]}.gpkg"
        try:
            gdf_base = gpd.read_file(url_gadm, layer=level)
            col = f"NAME_{level}" if level > 0 else "COUNTRY"
            choices = st.multiselect("Unités", sorted(gdf_base[col].unique()))
            if choices:
                st.session_state.selected_zone = gdf_base[gdf_base[col].isin(choices)].copy()
                st.session_state.zone_name = ", ".join(choices)
        except: st.error("Erreur de chargement GADM.")

    st.subheader("📅 Périodes radar")
    d_ref = st.date_input("Référence (Sec)", [datetime(2023, 1, 1), datetime(2023, 3, 30)])
    d_flood = st.date_input("Analyse (Inondation)", [datetime(2024, 8, 1), datetime(2024, 10, 30)])

st.title(f"🌊 FloodWatch WA Pro : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE D'IMPACT", type="primary", use_container_width=True):
        with st.spinner("Traitement satellite et spatial en cours..."):
            geom_union = st.session_state.selected_zone.unary_union
            aoi_json = mapping(geom_union)
            aoi_ee = ee.Geometry(aoi_json)
            
            # 1. Détection inondation affinée (Correction surestimation)
            flood_mask = detect_flood_refined(aoi_ee, str(d_ref[0]), str(d_ref[1]), str(d_flood[0]), str(d_flood[1]))
            
            # 2. Stats Population
            t_pop, e_pop = get_pop_stats_cached(aoi_json, flood_mask)
            
            # 3. Analyse Infrastructures (OSM)
            b_all, b_hit, r_all, r_hit = get_osm_impact(aoi_json, flood_mask)
            
            # 4. Données Climatiques
            centroid = geom_union.centroid
            df_clim = get_climate_data([centroid.x, centroid.y], str(d_flood[0]), str(d_flood[1]))
            
            # 5. Calcul Surface
            area_ha = 0
            mask_url = None
            if flood_mask:
                map_info = flood_mask.getMapId({'palette': ['#00BFFF']})
                mask_url = map_info['tile_fetcher'].url_format
                stats = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=20, maxPixels=1e9
                ).getInfo()
                area_ha = (list(stats.values())[0] / 10000) if stats else 0

            st.session_state.results = {
                'area': area_ha,
                't_pop': t_pop,
                'e_pop': e_pop,
                'mask_url': mask_url,
                'df_clim': df_clim,
                'b_all': b_all.to_json() if b_all is not None else None,
                'b_hit': b_hit.to_json() if b_hit is not None else None,
                'r_all': r_all.to_json() if r_all is not None else None,
                'r_hit': r_hit.to_json() if r_hit is not None else None
            }

    if st.session_state.results:
        res = st.session_state.results
        
        def safe_count(geojson_str):
            if not geojson_str: return 0
            try: return len(json.loads(geojson_str).get('features', []))
            except: return 0

        bh = safe_count(res.get('b_hit'))
        rh = safe_count(res.get('r_hit'))

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Surface Inondée", f"{res['area']:.1f} ha")
        c2.metric("Population Exposée", f"{res['e_pop']:,}", f"{ (res['e_pop']/res['t_pop']*100 if res['t_pop']>0 else 0):.1f}%")
        c3.metric("Bâtiments Touchés", bh, delta_color="inverse")
        c4.metric("Segments Routes", rh, delta_color="inverse")

        tab1, tab2, tab3 = st.tabs(["🗺️ Cartographie d'Impact", "📊 Analyse Climatique", "📥 Exportations"])
        
        with tab1:
            center = st.session_state.selected_zone.centroid.iloc[0]
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
            
            # Masque Inondation
            if res['mask_url']:
                folium.TileLayer(tiles=res['mask_url'], attr='GEE', name='Masque Inondation', overlay=True, opacity=0.6).add_to(m)
            
            # Couches Infrastructures
            if res['b_hit']:
                folium.GeoJson(res['b_hit'], name="Bâtiments IMPACTÉS (Rouge)", 
                               style_function=lambda x: {'color': '#e74c3c', 'fillColor': '#e74c3c', 'weight': 1, 'fillOpacity': 0.8}).add_to(m)
            if res['r_hit']:
                folium.GeoJson(res['r_hit'], name="Routes IMPACTÉES (Rouge)", 
                               style_function=lambda x: {'color': '#c0392b', 'weight': 4, 'opacity': 0.9}).add_to(m)
            
            if res['b_all']:
                folium.GeoJson(res['b_all'], name="Bâtiments (Total)", show=False,
                               style_function=lambda x: {'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.1}).add_to(m)
            
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600, key="main_map")

        with tab2:
            if res['df_clim']:
                df = pd.read_json(res['df_clim'])
                
                # Indicateurs météo moyens
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Cumul Précipitations", f"{df['Pluie'].sum():.1f} mm")
                mc2.metric("Température Moyenne", f"{df['Température'].mean():.1f} °C")
                mc3.metric("Jour le plus pluvieux", df.loc[df['Pluie'].idxmax(), 'date'].strftime('%d/%m/%Y'))

                # Graphique Ombrothermique
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df['date'], y=df['Pluie'], name="Précipitations (mm)", marker_color='#3498db', yaxis='y1'))
                fig.add_trace(go.Scatter(x=df['date'], y=df['Température'], name="Température (°C)", line=dict(color='#e67e22', width=3), yaxis='y2'))
                
                fig.update_layout(
                    title="Graphique Ombrothermique - Période d'Analyse",
                    xaxis=dict(title="Chronologie"),
                    yaxis=dict(title="Précipitations (mm)", titlefont=dict(color="#3498db"), tickfont=dict(color="#3498db")),
                    yaxis2=dict(title="Température (°C)", titlefont=dict(color="#e67e22"), tickfont=dict(color="#e67e22"), overlaying='y', side='right'),
                    legend=dict(x=0.01, y=0.99),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données climatiques non disponibles pour cette zone.")

        with tab3:
            st.subheader("Exporter les résultats")
            
            # Rapport CSV
            report_df = pd.DataFrame({
                "Indicateur": ["Surface Inondée (ha)", "Population Totale", "Population Exposée", "Bâtiments Impactés", "Routes Impactées"],
                "Valeur": [res['area'], res['t_pop'], res['e_pop'], bh, rh]
            })
            st.table(report_df)
            
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Rapport (CSV)", csv, "rapport_impact.csv", "text/csv", use_container_width=True)
            
            with ec2:
                if res['b_hit']:
                    st.download_button("📥 Bâtiments (GeoJSON)", res['b_hit'], "batiments_impactes.geojson", "application/json", use_container_width=True)
            
            with ec3:
                if res['r_hit']:
                    st.download_button("📥 Routes (GeoJSON)", res['r_hit'], "routes_impactees.geojson", "application/json", use_container_width=True)
else:
    st.info("💡 Sélectionnez une zone dans la barre latérale et cliquez sur Analyser pour générer le rapport d'impact.")
