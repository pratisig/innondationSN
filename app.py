import os
import io
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

import folium
from streamlit_folium import st_folium

import osmnx as ox
import ee
import requests

# ═════════════════════════════════════════════════════════════════
# 1. CONFIG & INIT
# ═════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Flood Analysis WA", layout="wide")

def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("❌ Secret 'GEE_SERVICE_ACCOUNT' manquant.")
        return False
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"❌ Erreur GEE : {e}")
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. GADM FIX (ADMIN 0-4)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        # Utilisation de fiona pour lister les couches si besoin, mais on assume le nom standard
        layer_name = f"ADM_ADM_{level}" # GADM GPKG naming convention
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except:
        return None

# ═════════════════════════════════════════════════════════════════
# 3. SENTINEL-1 OPTIMISÉ
# ═════════════════════════════════════════════════════════════════

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood, diff_threshold=1.25):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi_ee)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))

    # Référence : Médiane pour stabiliser le signal sec
    img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
    
    # Crise : Minimum pour capturer la baisse de signal (eau = spéculaire = noir)
    img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)

    # Conversion dB simplifiée
    ref_db = img_ref.log10().multiply(10)
    flood_db = img_flood.log10().multiply(10)

    # Différence
    diff = ref_db.subtract(flood_db)
    flooded = diff.gt(diff_threshold)

    # Masques (Pente et Eau Permanente)
    terrain = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003"))
    slope = terrain.select('slope')
    
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
    
    final_mask = (flooded
                  .updateMask(slope.lt(5))
                  .updateMask(gsw.lt(80))
                  .selfMask())
    
    return final_mask, ref_db, flood_db

# ═════════════════════════════════════════════════════════════════
# 4. OSM FIX (FEATURES_FROM_BBOX)
# ═════════════════════════════════════════════════════════════════

def get_osm_data(gdf_aoi):
    bounds = gdf_aoi.total_bounds # [minx, miny, maxx, maxy]
    # OSMnx attend [north, south, east, west]
    try:
        # Nouveau OSMnx (v1.3+) utilise features_from_bbox
        tags = {
            'building': True, 
            'highway': True, 
            'amenity': ['hospital', 'school', 'clinic']
        }
        # On télécharge les données dans la bounding box
        data = ox.features_from_bbox(bounds[3], bounds[1], bounds[2], bounds[0], tags=tags)
        return data[data.geometry.within(gdf_aoi.unary_union)]
    except:
        return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 5. UI STREAMLIT
# ═════════════════════════════════════════════════════════════════

st.sidebar.header("🌍 Paramètres")
country_dict = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
iso = country_dict[country]

# Cascade Admin
level = st.sidebar.slider("Niveau de précision (0=Pays, 2=Dépt, 3/4=Communes)", 0, 4, 1)
gdf_base = load_gadm(iso, level)

selected_zone = None
if gdf_base is not None:
    col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
    if col_name in gdf_base.columns:
        names = sorted(gdf_base[col_name].unique())
        choice = st.sidebar.multiselect("Zone(s) spécifique(s)", names)
        if choice:
            selected_zone = gdf_base[gdf_base[col_name].isin(choice)]
        else:
            selected_zone = gdf_base

# Dates
d1, d2 = st.sidebar.columns(2)
start_f = d1.date_input("Début Inondation", datetime(2024, 8, 1))
end_f = d2.date_input("Fin Inondation", datetime(2024, 8, 31))

if st.sidebar.button("Lancer l'analyse") and selected_zone is not None:
    with st.spinner("Analyse GEE + OSM en cours..."):
        # 1. GEE
        aoi_ee = ee.Geometry(mapping(selected_zone.unary_union))
        flood_mask, ref_img, flood_img = get_flood_mask(
            aoi_ee, "2024-01-01", "2024-03-01", str(start_f), str(end_f)
        )
        
        # 2. OSM Impact
        osm_all = get_osm_data(selected_zone)
        
        # 3. Visualisation
        st.subheader(f"Résultats pour {country}")
        
        m = folium.Map(location=[selected_zone.centroid.y.mean(), selected_zone.centroid.x.mean()], zoom_start=8)
        
        # Overlay Flood
        vis_params = {'min': 0, 'max': 1, 'palette': ['000000', '00FFFF']}
        map_id = flood_mask.getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name='Zones Inondées',
            overlay=True
        ).add_to(m)
        
        folium.GeoJson(selected_zone, name="Limites Admin").add_to(m)
        st_folium(m, width=1000)
        
        # Stats
        if not osm_all.empty:
            st.write(f"Infrastructures détectées dans la zone : {len(osm_all)}")
            if 'building' in osm_all.columns:
                st.metric("Bâtiments OSM", len(osm_all[osm_all['building'].notnull()]))
            if 'amenity' in osm_all.columns:
                st.metric("Écoles/Santé", len(osm_all[osm_all['amenity'].notnull()]))
