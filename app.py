import os
import io
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping

import folium
from streamlit_folium import st_folium

import osmnx as ox
import ee

# ═════════════════════════════════════════════════════════════════
# 1. CONFIG & INIT
# ═════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Flood Analysis WA", layout="wide")

# Configuration globale OSMnx
ox.settings.timeout = 60
ox.settings.use_cache = True

# Initialisation des variables d'état
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'flood_mask' not in st.session_state:
    st.session_state.flood_mask = None
if 'impacted_infra' not in st.session_state:
    st.session_state.impacted_infra = gpd.GeoDataFrame()
if 'results_gdf' not in st.session_state:
    st.session_state.results_gdf = gpd.GeoDataFrame()
if 'precip' not in st.session_state:
    st.session_state.precip = 0.0
if 'stats' not in st.session_state:
    st.session_state.stats = {
        "pop_exposed": 0,
        "total_pop": 0,
        "total_flood_ha": 0,
        "total_infra": 0
    }

# ────────── INITIALISATION GEE ──────────
@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = st.secrets["GEE_SERVICE_ACCOUNT"]
            if isinstance(key_dict, str):
                key_dict = json.loads(key_dict)
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            return True
        ee.Initialize()
        return True
    except Exception:
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS UTILITAIRES
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Erreur GADM: {e}")
        return None

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood, threshold=5):
    if not gee_available: return None
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi_ee)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .select('VV'))

        img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
        img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)
        
        img_ref = img_ref.focal_median(50, 'circle', 'meters')
        img_flood = img_flood.focal_median(50, 'circle', 'meters')

        diff = img_ref.subtract(img_flood)
        return diff.gt(threshold).selfMask()
    except Exception: return None

def get_precip_cumul(aoi_ee, start_date, end_date):
    if not gee_available: return 0
    try:
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                   .filterBounds(aoi_ee) \
                   .filterDate(start_date, end_date) \
                   .select('precipitation') \
                   .sum()
        stats = chirps.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi_ee, scale=5000)
        return float(stats.get('precipitation').getInfo() or 0)
    except: return 0

def get_area_stats(aoi_ee, flood_mask):
    if not gee_available or flood_mask is None: return 0
    try:
        area_img = flood_mask.multiply(ee.Image.pixelArea())
        stats = area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=30, maxPixels=1e9)
        area_m2 = stats.get('VV').getInfo() or 0
        return area_m2 / 10000  # Hectares
    except: return 0

def get_population_stats(aoi_ee, flood_mask):
    if not gee_available: return 0, 0
    try:
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01', '2021-01-01') \
                        .mosaic().clip(aoi_ee)
        
        stats_total = pop_dataset.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
        total_pop = stats_total.get('population').getInfo() or 0
        
        exposed_pop = 0
        if flood_mask:
            pop_exposed_img = pop_dataset.updateMask(flood_mask)
            stats_exposed = pop_exposed_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
            exposed_pop = stats_exposed.get('population').getInfo() or 0
        
        return int(total_pop), int(exposed_pop)
    except: return 0, 0

@st.cache_data(show_spinner=False)
def get_osm_buildings(_gdf_aoi):
    if _gdf_aoi is None or _gdf_aoi.empty: return gpd.GeoDataFrame()
    try:
        poly = _gdf_aoi.unary_union.convex_hull
        tags = {
            'building': True, 
            'amenity': ['hospital', 'school', 'clinic', 'pharmacy', 'marketplace', 'place_of_worship']
        }
        data = ox.geometries_from_polygon(poly, tags=tags)
        if data.empty: return gpd.GeoDataFrame()
        
        # Filtrer et nettoyer
        data = data[data.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        return data.clip(_gdf_aoi)
    except Exception: return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. UI SIDEBAR
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🌍 Paramètres")

country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
source_option = st.sidebar.radio("Source de la zone", ["Pays/Admin", "Fichier"])

selected_zone = None

if source_option == "Pays/Admin":
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    level = st.sidebar.slider("Niveau Administratif", 0, 3, 2)
    gdf_base = load_gadm(iso, level)
    
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
        names = sorted(gdf_base[col_name].astype(str).unique())
        choice = st.sidebar.multiselect("Zone(s)", names)
        selected_zone = gdf_base[gdf_base[col_name].isin(choice)].copy() if choice else gdf_base.iloc[[0]].copy() 

elif source_option == "Fichier":
    uploaded_file = st.sidebar.file_uploader("Importer KML/GeoJSON", type=["kml","geojson","shp"])
    if uploaded_file:
        selected_zone = gpd.read_file(uploaded_file).to_crs(epsg=4326)

st.sidebar.markdown("---")
d1, d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début Inondation", datetime(2024, 8, 1))
end_f = d2.date_input("Fin Inondation", datetime(2024, 9, 30))
flood_threshold = st.sidebar.slider("Seuil Détection (dB)", 3.0, 10.0, 5.0, 0.5)

# ═════════════════════════════════════════════════════════════════
# 4. LOGIQUE D'ANALYSE
# ═════════════════════════════════════════════════════════════════

st.title("🌊 Analyse des Infrastructures Impactées")

if st.button("🚀 LANCER L'ANALYSE", type="primary"):
    st.session_state.analysis_done = True
    
    with st.spinner("Analyse spatiale et croisement des données..."):
        # 1. Masque Inondation GEE
        full_aoi_ee = ee.Geometry(mapping(selected_zone.unary_union))
        st.session_state.flood_mask = get_flood_mask(full_aoi_ee, "2023-01-01", "2023-05-01", str(start_f), str(end_f), flood_threshold)
        
        # 2. Précipitations
        st.session_state.precip = get_precip_cumul(full_aoi_ee, str(start_f), str(end_f))
        
        # 3. Récupération OSM
        buildings_gdf = get_osm_buildings(selected_zone)
        
        # 4. Conversion du masque GEE en vecteur pour intersection spatiale locale
        # Note: On vectorise uniquement pour l'intersection avec les bâtiments OSM
        impacted_infra_list = []
        if st.session_state.flood_mask and not buildings_gdf.empty:
            flood_vectors = st.session_state.flood_mask.reduceToVectors(
                geometry=full_aoi_ee, scale=30, maxPixels=1e9
            ).getInfo()
            
            if flood_vectors and 'features' in flood_vectors:
                flood_polys = [shape(f['geometry']) for f in flood_vectors['features']]
                flood_gdf = gpd.GeoDataFrame(geometry=flood_polys, crs="EPSG:4326")
                
                # Intersection spatiale : bâtiments qui touchent les zones inondées
                st.session_state.impacted_infra = gpd.sjoin(buildings_gdf, flood_gdf, how="inner", predicate="intersects")
        
        # 5. Analyse par polygone administratif
        temp_list = []
        for idx, row in selected_zone.iterrows():
            geom_ee = ee.Geometry(mapping(row.geometry))
            t_pop, e_pop = get_population_stats(geom_ee, st.session_state.flood_mask)
            f_area = get_area_stats(geom_ee, st.session_state.flood_mask)
            
            # Filtrer les infrastructures impactées dans ce polygone spécifique
            if not st.session_state.impacted_infra.empty:
                infra_in_poly = st.session_state.impacted_infra.clip(row.geometry)
                counts = infra_in_poly['amenity'].fillna('Bâtiment/Résidentiel').value_counts().to_dict()
                n_total_infra = len(infra_in_poly)
            else:
                counts = {}
                n_total_infra = 0
            
            temp_list.append({
                'name': row.get('NAME_2', row.get('NAME_1', 'Zone')),
                'pop_total': t_pop,
                'pop_exposed': e_pop,
                'flood_ha': round(f_area, 2),
                'n_infra': n_total_infra,
                'infra_details': counts,
                'geometry': row.geometry
            })
        
        st.session_state.results_gdf = gpd.GeoDataFrame(temp_list, crs="EPSG:4326")
        
        # Statistiques Globales
        st.session_state.stats = {
            "pop_exposed": sum(d['pop_exposed'] for d in temp_list),
            "total_pop": sum(d['pop_total'] for d in temp_list),
            "total_flood_ha": sum(d['flood_ha'] for d in temp_list),
            "total_infra": sum(d['n_infra'] for d in temp_list)
        }

# ═════════════════════════════════════════════════════════════════
# 5. AFFICHAGE DES RÉSULTATS
# ═════════════════════════════════════════════════════════════════

if st.session_state.analysis_done:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pop. Exposée", f"{st.session_state.stats.get('pop_exposed', 0):,}")
    m2.metric("Superficie Inondée", f"{st.session_state.stats.get('total_flood_ha', 0):,} ha")
    m3.metric("Pluviométrie (moy)", f"{st.session_state.precip:.1f} mm")
    m4.metric("Infrastructures Touchées", f"{st.session_state.stats.get('total_infra', 0):,}")

    col_map, col_list = st.columns([3, 1])
    
    with col_list:
        st.markdown("### 🏘️ Impact par type")
        if not st.session_state.impacted_infra.empty:
            summary = st.session_state.impacted_infra['amenity'].fillna('Résidentiel').value_counts()
            st.dataframe(summary, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📍 Détails par zone")
        for _, r in st.session_state.results_gdf.iterrows():
            with st.expander(f"**{r['name']}**"):
                st.write(f"🌊 Inondé : {r['flood_ha']:,} ha")
                st.write(f"👥 Pop. Exposée : {r['pop_exposed']:,}")
                st.write(f"🏠 Infras : {r['n_infra']}")
                if r['infra_details']:
                    for k, v in r['infra_details'].items():
                        st.caption(f"- {k}: {v}")

    with col_map:
        center = selected_zone.centroid.iloc[0]
        m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron")
        
        # 1. Limites administratives (Orange)
        folium.GeoJson(
            selected_zone,
            name="Limites Administratives",
            style_function=lambda x: {'fillColor': 'none', 'color': 'orange', 'weight': 3}
        ).add_to(m)

        # 2. Masque Inondation GEE (Bleu)
        if st.session_state.flood_mask:
            try:
                map_id = st.session_state.flood_mask.getMapId({'palette': ['#00d4ff']})
                folium.TileLayer(
                    tiles=map_id['tile_fetcher'].url_format,
                    attr='Google Earth Engine',
                    name='Zones Inondées',
                    overlay=True,
                    opacity=0.6
                ).add_to(m)
            except: pass

        # 3. Polygones invisibles pour les Popups de zone
        for _, row in st.session_state.results_gdf.iterrows():
            infra_str = "<br>".join([f"- {k}: {v}" for k, v in row['infra_details'].items()])
            popup_content = f"""
            <div style='width:200px'>
                <b>{row['name']}</b><br>
                Pop. Exposée: <b style='color:red'>{row['pop_exposed']:,}</b><br>
                Surface: {row['flood_ha']} ha<br>
                <b>Infras Impactées:</b><br>{infra_str if infra_str else 'Aucune'}
            </div>
            """
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {'fillColor': 'none', 'color': 'none'},
                tooltip=row['name']
            ).add_child(folium.Popup(popup_content)).add_to(m)

        # 4. Bâtiments Impactés (Rouge)
        if not st.session_state.impacted_infra.empty:
            folium.GeoJson(
                st.session_state.impacted_infra,
                name="Bâtiments Touchés",
                style_function=lambda x: {
                    'fillColor': 'red', 
                    'color': 'darkred', 
                    'weight': 1, 
                    'fillOpacity': 0.8
                },
                tooltip=folium.GeoJsonTooltip(fields=['amenity', 'name'], aliases=['Type:', 'Nom:'])
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=650, key="flood_map")
else:
    st.info("Sélectionnez vos zones à gauche puis cliquez sur 'Lancer l'Analyse'.")
