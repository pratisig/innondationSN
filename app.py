import os
import io
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

import folium
from streamlit_folium import st_folium

import osmnx as ox
import ee

# ═════════════════════════════════════════════════════════════════
# 1. CONFIG & INIT
# ═════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Flood Analysis WA", layout="wide")

ox.settings.timeout = 120
ox.settings.use_cache = True

# Initialisation variables état
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
        diff = img_ref.subtract(img_flood)
        return diff.gt(threshold).rename('flood').selfMask()
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
        area_m2 = stats.get('flood').getInfo() or 0
        return area_m2 / 10000
    except: return 0

def get_population_stats(aoi_ee, flood_mask):
    if not gee_available: return 0,0
    try:
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01','2021-01-01') \
                        .mosaic().clip(aoi_ee)
        stats_total = pop_dataset.reduceRegion(ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
        total_pop = stats_total.get('population').getInfo() or 0
        exposed_pop = 0
        if flood_mask:
            stats_exposed = pop_dataset.updateMask(flood_mask).reduceRegion(ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
            exposed_pop = stats_exposed.get('population').getInfo() or 0
        return int(total_pop), int(exposed_pop)
    except: return 0,0

def get_osm_buildings_stable(selected_zone):
    if selected_zone is None or selected_zone.empty:
        return gpd.GeoDataFrame()
    try:
        poly = selected_zone.unary_union.convex_hull.simplify(0.001)
        building_tags = {"building": True}
        amenity_tags = {"amenity": ["hospital","school","clinic","marketplace","place_of_worship"]}
        try:
            df_building = ox.geometries_from_polygon(poly, tags=building_tags)
        except: df_building = gpd.GeoDataFrame()
        try:
            df_amenity = ox.geometries_from_polygon(poly, tags=amenity_tags)
        except: df_amenity = gpd.GeoDataFrame()
        df = pd.concat([df_building, df_amenity], ignore_index=True)
        if df.empty:
            bounds = selected_zone.total_bounds
            north, south, east, west = bounds[3], bounds[1], bounds[2], bounds[0]
            df = ox.features_from_bbox(north, south, east, west, tags={**building_tags, **amenity_tags})
        df = df[df.geometry.type.isin(["Polygon","MultiPolygon"])]
        df = df.clip(selected_zone).reset_index(drop=True)
        return df
    except: return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. UI SIDEBAR
# ═════════════════════════════════════════════════════════════════
st.sidebar.header("🌍 Paramètres")

country_dict = {"Sénégal":"SEN","Mali":"MLI","Niger":"NER","Burkina Faso":"BFA"}
source_option = st.sidebar.radio("Source de la zone", ["Pays/Admin","Fichier"])

selected_zone = None

if source_option=="Pays/Admin":
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    level = st.sidebar.slider("Niveau Administratif", 0, 3, 2)
    gdf_base = load_gadm(iso, level)
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level>0 else "COUNTRY"
        names = sorted(gdf_base[col_name].astype(str).unique())
        choice = st.sidebar.multiselect("Zone(s)", names)
        selected_zone = gdf_base[gdf_base[col_name].isin(choice)].copy() if choice else gdf_base.iloc[[0]].copy()
elif source_option=="Fichier":
    uploaded_file = st.sidebar.file_uploader("Importer KML/GeoJSON", type=["kml","geojson","shp"])
    if uploaded_file:
        selected_zone = gpd.read_file(uploaded_file).to_crs(epsg=4326)

st.sidebar.markdown("---")
d1,d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début Inondation", datetime(2024,8,1))
end_f = d2.date_input("Fin Inondation", datetime(2024,9,30))
flood_threshold = st.sidebar.slider("Seuil Détection (dB)", 2.0,10.0,4.0,0.5)

# ═════════════════════════════════════════════════════════════════
# 4. LOGIQUE ANALYSE
# ═════════════════════════════════════════════════════════════════
st.title("🌊 Analyse des Infrastructures Impactées")

if st.button("🚀 LANCER L'ANALYSE", type="primary"):
    st.session_state.analysis_done=True
    with st.spinner("Récupération des données et analyse..."):
        full_aoi_ee = ee.Geometry(mapping(selected_zone.unary_union))
        st.session_state.flood_mask = get_flood_mask(full_aoi_ee,"2023-01-01","2023-05-01",str(start_f),str(end_f),flood_threshold)
        st.session_state.precip = get_precip_cumul(full_aoi_ee,str(start_f),str(end_f))
        buildings_gdf = get_osm_buildings_stable(selected_zone)
        
        # Intersection bâtiments impactés
        if st.session_state.flood_mask and not buildings_gdf.empty:
            infra_to_check = buildings_gdf.head(2000).copy()
            features=[]
            for i,row in infra_to_check.iterrows():
                geom = mapping(row.geometry)
                props={'osm_index':i,'type':str(row.get('amenity', row.get('building','Inconnu'))),'name':str(row.get('name','Sans nom'))}
                features.append(ee.Feature(ee.Geometry(geom),props))
            fc_infra = ee.FeatureCollection(features)
            impact_results = st.session_state.flood_mask.reduceRegions(fc_infra, ee.Reducer.mean(), 10).filter(ee.Filter.gt('mean',0)).getInfo()
            impacted_indices = [f['properties']['osm_index'] for f in impact_results['features']]
            st.session_state.impacted_infra = infra_to_check.loc[impacted_indices].copy()
        else:
            st.session_state.impacted_infra = gpd.GeoDataFrame()
        
        # Analyse par polygone
        temp_list=[]
        for idx,row in selected_zone.iterrows():
            geom_ee = ee.Geometry(mapping(row.geometry))
            t_pop,e_pop = get_population_stats(geom_ee,st.session_state.flood_mask)
            f_area = get_area_stats(geom_ee,st.session_state.flood_mask)
            if not st.session_state.impacted_infra.empty:
                infra_in_poly = st.session_state.impacted_infra[st.session_state.impacted_infra.intersects(row.geometry)]
                counts = infra_in_poly['amenity'].fillna('Bâtiment/Résidentiel').value_counts().to_dict()
                n_total_infra=len(infra_in_poly)
            else: counts={}; n_total_infra=0
            temp_list.append({
                'name':row.get('NAME_2',row.get('NAME_1','Zone')),
                'pop_total':t_pop,'pop_exposed':e_pop,
                'flood_ha':round(f_area,2),'n_infra':n_total_infra,
                'infra_details':counts,'geometry':row.geometry
            })
        st.session_state.results_gdf = gpd.GeoDataFrame(temp_list,crs="EPSG:4326")
        st.session_state.stats = {
            "pop_exposed": sum(d['pop_exposed'] for d in temp_list),
            "total_pop": sum(d['pop_total'] for d in temp_list),
            "total_flood_ha": sum(d['flood_ha'] for d in temp_list),
            "total_infra": len(st.session_state.impacted_infra)
        }

# ═════════════════════════════════════════════════════════════════
# 5. AFFICHAGE RÉSULTATS
# ═════════════════════════════════════════════════════════════════
if st.session_state.analysis_done:
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Pop. Exposée",f"{st.session_state.stats.get('pop_exposed',0):,}")
    m2.metric("Superficie Inondée",f"{st.session_state.stats.get('total_flood_ha',0):,} ha")
    m3.metric("Pluviométrie (moy)",f"{st.session_state.precip:.1f} mm")
    m4.metric("Infrastructures Touchées",f"{st.session_state.stats.get('total_infra',0):,}")
    
    col_map,col_list=st.columns([3,1])
    with col_list:
        st.markdown("### 🏘️ Typologie des Impacts")
        if not st.session_state.impacted_infra.empty:
            summary=st.session_state.impacted_infra['amenity'].fillna('Bâtiment/Logement').value_counts()
            st.dataframe(summary,use_container_width=True)
        else: st.warning("Aucune infrastructure touchée détectée.")
        st.markdown("---")
        st.markdown("### 📍 Bilan par Secteur")
        for _,r in st.session_state.results_gdf.iterrows():
            with st.expander(f"**{r['name']}**"):
                st.write(f"🌊 Inondé : {r['flood_ha']:,} ha")
                st.write(f"🏠 Infras touchées : {r['n_infra']}")
                if r['infra_details']:
                    for k,v in r['infra_details'].items():
                        st.caption(f"- {k}: {v}")

    with col_map:
        center = selected_zone.centroid.iloc[0]
        m=folium.Map(location=[center.y,center.x],zoom_start=11,tiles="cartodbpositron")
        folium.GeoJson(selected_zone,name="Limite administrative",
                       style_function=lambda x:{'fillColor':'none','color':'#ff7800','weight':4,'opacity':0.7}).add_to(m)
        if st.session_state.flood_mask:
            try:
                map_id=st.session_state.flood_mask.getMapId({'palette':['#0077be']})
                folium.TileLayer(tiles=map_id['tile_fetcher'].url_format,
                                 attr='Google Earth Engine',name='Masque Inondation',
                                 overlay=True,opacity=0.6).add_to(m)
            except: pass
        if not st.session_state.impacted_infra.empty:
            folium.GeoJson(st.session_state.impacted_infra,name="Infrastructures Impactées",
                           style_function=lambda x:{'fillColor':'#e31a1c','color':'#800026','weight':1.5,'fillOpacity':0.9},
                           tooltip=folium.GeoJsonTooltip(fields=['amenity','name'],aliases=['Type:','Nom:'])).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m,width="100%",height=700,key="map_final")
else:
    st.info("Sélectionnez une zone et lancez l'analyse pour visualiser les dégâts.")
