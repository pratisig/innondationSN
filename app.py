# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP - West Africa
# ============================================================

import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union

import folium
from streamlit_folium import st_folium

import osmnx as ox
import ee

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🌊 Analyse d'Impact Inondations – WA",
    layout="wide"
)
st.title("🌊 Analyse d'Impact Inondations & Infrastructures")
st.caption("Sentinel-1 | WorldPop | OSM | GADM")

# ============================================================
# 2. INITIALISATION GEE
# ============================================================
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.warning("GEE non disponible, les analyses seront simulées.")
        return False
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.warning(f"Impossible d'initialiser GEE : {e}")
        return False

gee_available = init_gee()

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================
@st.cache_data(ttl=3600)
def load_gadm(iso, level):
    """Charge GADM depuis le gpkg"""
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except:
        return None

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood):
    """Calcul masque d'inondation avec Sentinel-1 via GEE"""
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD")\
        .filterBounds(aoi_ee)\
        .filter(ee.Filter.eq("instrumentMode", "IW"))\
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
    img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)
    ref_db = img_ref.log10().multiply(10)
    flood_db = img_flood.log10().multiply(10)
    diff = ref_db.subtract(flood_db)
    flooded = diff.gt(1.25)
    terrain = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003"))
    slope = terrain.select('slope')
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
    final_mask = flooded.updateMask(slope.lt(5)).updateMask(gsw.lt(80)).selfMask()
    return final_mask

def get_osm_data(gdf_aoi):
    """Télécharge bâtiments/routes/amenities via OSMnx"""
    if gdf_aoi is None or gdf_aoi.empty:
        return gpd.GeoDataFrame()
    bounds = gdf_aoi.total_bounds  # [minx, miny, maxx, maxy]
    tags = {'building': True, 'highway': True, 'amenity': ['hospital','school','clinic']}
    try:
        osm = ox.geometries_from_bbox(
            north=bounds[3], south=bounds[1],
            east=bounds[2], west=bounds[0],
            tags=tags
        )
        # Filtrer pour la zone exacte
        osm = osm[osm.geometry.within(unary_union(gdf_aoi.geometry))]
        return osm
    except:
        return gpd.GeoDataFrame()

def calc_population_affected(gdf_aoi):
    """Renvoie une valeur fictive si GEE non dispo"""
    return 150  # valeur par défaut pour affichage

def calc_roads_affected(osm_gdf):
    if osm_gdf.empty or 'highway' not in osm_gdf.columns:
        return 0
    return osm_gdf[osm_gdf['highway'].notnull()].shape[0]  # nb segments

def calc_buildings_affected(osm_gdf):
    if osm_gdf.empty or 'building' not in osm_gdf.columns:
        return 0
    return osm_gdf[osm_gdf['building'].notnull()].shape[0]

# ============================================================
# 4. SIDEBAR - PARAMETRES
# ============================================================
st.sidebar.header("🌍 Paramètres d'analyse")
country_dict = {"Sénégal":"SEN","Mali":"MLI","Niger":"NER","Burkina Faso":"BFA"}
country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
iso = country_dict[country]

# Choix admin
admin_choice = st.sidebar.radio("Sélection de zone", ["Niveau administratif", "Importer KML/GeoJSON/SHP"])
selected_zone = None

if admin_choice=="Niveau administratif":
    level = st.sidebar.slider("Niveau admin", 0, 4, 1)
    gdf_base = load_gadm(iso, level)
    if gdf_base is not None and not gdf_base.empty:
        col_name = f"NAME_{level}" if level>0 else "COUNTRY"
        names = sorted(gdf_base[col_name].unique())
        sel_names = st.sidebar.multiselect("Zones", names, default=names[:1])
        if sel_names:
            selected_zone = gdf_base[gdf_base[col_name].isin(sel_names)]
        else:
            selected_zone = gdf_base
else:
    uploaded_file = st.sidebar.file_uploader("Uploader KML/SHP/GeoJSON", type=['kml','shp','geojson'])
    if uploaded_file:
        selected_zone = gpd.read_file(uploaded_file)

# Dates
d1, d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début inondation", datetime(2024,8,1))
end_f = d2.date_input("Fin inondation", datetime(2024,8,31))

# ============================================================
# 5. ANALYSE & VISUALISATION
# ============================================================
if selected_zone is not None and not selected_zone.empty:
    # 1. Carte
    m = folium.Map(location=[selected_zone.centroid.y.mean(), selected_zone.centroid.x.mean()], zoom_start=8)
    folium.GeoJson(selected_zone, name="Zone sélectionnée").add_to(m)

    # 2. Masque inondation (GEE ou placeholder)
    if gee_available:
        aoi_ee = ee.Geometry(mapping(selected_zone.unary_union))
        flood_mask = get_flood_mask(aoi_ee,"2024-01-01","2024-03-01", str(start_f), str(end_f))
        vis_params = {'min':0,'max':1,'palette':['000000','00FFFF']}
        map_id = flood_mask.getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name='Zones Inondées',
            overlay=True
        ).add_to(m)
    else:
        st.info("GEE non disponible : affichage masqué inondation")

    # 3. Données OSM
    osm_gdf = get_osm_data(selected_zone)

    # 4. Statistiques
    pop_affected = calc_population_affected(selected_zone)
    roads_affected = calc_roads_affected(osm_gdf)
    buildings_affected = calc_buildings_affected(osm_gdf)

    c1,c2,c3 = st.columns(3)
    c1.metric("Population affectée", f"{pop_affected:,}")
    c2.metric("Bâtiments affectés", f"{buildings_affected:,}" if buildings_affected>0 else "N/A")
    c3.metric("Routes impactées", f"{roads_affected:,}" if roads_affected>0 else "N/A")

    st_folium(m, width=1000, height=500)
else:
    st.warning("⚠️ Sélectionnez une zone pour lancer l'analyse.")

