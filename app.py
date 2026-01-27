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

# ────────── INITIALISATION GEE ──────────
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" not in st.secrets:
            st.warning("❌ Secret 'GEE_SERVICE_ACCOUNT' manquant. GEE désactivé.")
            return False

        # Lecture JSON de la clé service account
        key_dict = st.secrets["GEE_SERVICE_ACCOUNT"]
        if isinstance(key_dict, str):
            key_dict = json.loads(key_dict)

        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"], key_data=json.dumps(key_dict)
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.warning(f"❌ Erreur initialisation GEE : {e}")
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS UTILITAIRES
# ═════════════════════════════════════════════════════════════════

def load_gadm(iso, level):
    """Charge GADM depuis le GPKG distant"""
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Erreur chargement GADM: {e}")
        return None

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood):
    if not gee_available:
        return None
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi_ee)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
        img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
        img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)
        diff = img_ref.subtract(img_flood)
        flooded = diff.gt(0)
        return flooded
    except Exception as e:
        st.warning(f"Impossible de générer le masque GEE: {e}")
        return None

def get_osm_data(gdf_aoi):
    if gdf_aoi is None or gdf_aoi.empty:
        return gpd.GeoDataFrame()
    bounds = gdf_aoi.total_bounds
    try:
        tags = {'building': True, 'highway': True, 'amenity': ['hospital','school','clinic']}
        data = ox.features_from_bbox(north=bounds[3],
                                     south=bounds[1],
                                     east=bounds[2],
                                     west=bounds[0],
                                     tags=tags)
        return data[data.geometry.within(gdf_aoi.unary_union)]
    except Exception as e:
        st.warning(f"Erreur chargement OSM: {e}")
        return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. UI STREAMLIT
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🌍 Paramètres")

country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
source_option = st.sidebar.radio("Source de la zone", ["Pays/Admin", "Fichier (KML/GeoJSON/Shp)"])

selected_zone = None
if source_option == "Pays/Admin":
    country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
    iso = country_dict[country]
    level = st.sidebar.slider("Niveau Admin (0=Pays, 1=Région, 2=Département, 3/4=Commune)", 0, 4, 1)
    gdf_base = load_gadm(iso, level)
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
        names = sorted(gdf_base[col_name].unique())
        choice = st.sidebar.multiselect("Zone(s) spécifique(s)", names, default=names)
        if choice:
            selected_zone = gdf_base[gdf_base[col_name].isin(choice)]
        else:
            selected_zone = gdf_base
elif source_option == "Fichier (KML/GeoJSON/Shp)":
    uploaded_file = st.sidebar.file_uploader("Uploader KML/GeoJSON/Shapefile", type=["kml","geojson","shp"])
    if uploaded_file:
        selected_zone = gpd.read_file(uploaded_file)
    else:
        st.info("Uploader un fichier pour la zone d'étude.")

# Dates
d1, d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début Inondation", datetime(2024,8,1))
end_f = d2.date_input("Fin Inondation", datetime(2024,8,31))

# Zone par défaut si vide
if selected_zone is None:
    selected_zone = gpd.GeoDataFrame([{"geometry": shape({"type":"Polygon","coordinates":[[[-14,14],[-14,15],[-13,15],[-13,14],[-14,14]]]})}], crs="EPSG:4326")

# ═════════════════════════════════════════════════════════════════
# 4. ANALYSE
# ═════════════════════════════════════════════════════════════════

with st.spinner("Analyse en cours..."):

    aoi_ee = ee.Geometry(mapping(selected_zone.unary_union)) if gee_available else None
    flood_mask = get_flood_mask(aoi_ee, "2024-01-01", "2024-03-01", str(start_f), str(end_f)) if gee_available else None

    osm_all = get_osm_data(selected_zone)
    n_buildings = len(osm_all[osm_all['building'].notnull()]) if 'building' in osm_all.columns else 0
    n_amenities = len(osm_all[osm_all['amenity'].notnull()]) if 'amenity' in osm_all.columns else 0
    n_roads = len(osm_all[osm_all['highway'].notnull()]) if 'highway' in osm_all.columns else 0

    total_pop = 1000
    pop_exposed = 200 if flood_mask is not None else 0

# Carte
st.subheader("🗺️ Carte")
m = folium.Map(location=[selected_zone.centroid.y.mean(), selected_zone.centroid.x.mean()], zoom_start=8)
folium.GeoJson(selected_zone, name="Zone").add_to(m)
if flood_mask is not None:
    try:
        vis_params = {'min':0,'max':1,'palette':['000000','00FFFF']}
        map_id = flood_mask.getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name='Zones Inondées',
            overlay=True
        ).add_to(m)
    except:
        st.warning("Impossible d'afficher le masque d'inondation GEE.")
st_folium(m, width=1000,height=500)

# Indicateurs
st.subheader("📊 Indicateurs")
c1,c2,c3 = st.columns(3)
c1.metric("Population exposée", pop_exposed)
c2.metric("Bâtiments affectés", n_buildings)
c3.metric("Routes affectées", n_roads)
c4,c5 = st.columns(2)
c4.metric("Infrastructures (Amenity)", n_amenities)
c5.metric("Population totale approx.", total_pop)
