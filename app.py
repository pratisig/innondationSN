# ============================================================
# FLOOD ANALYSIS – WEST AFRICA
# Sentinel-1 | GADM | OSMnx | Streamlit | Earth Engine
# ============================================================

import json
from datetime import datetime

import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
from shapely.ops import unary_union

import folium
from streamlit_folium import st_folium

import osmnx as ox
import ee

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Flood Analysis – West Africa",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Flood Impact Analysis – West Africa")
st.caption("Sentinel-1 Radar | GADM | OpenStreetMap | Google Earth Engine")

# ============================================================
# EARTH ENGINE INIT (SERVICE ACCOUNT)
# ============================================================

def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("❌ Secret GEE_SERVICE_ACCOUNT manquant.")
        st.stop()
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key["client_email"],
            key_data=json.dumps(key)
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"❌ Erreur Earth Engine : {e}")
        st.stop()

init_gee()

# ============================================================
# GADM (ADMIN LEVELS 0–4)
# ============================================================

@st.cache_data(ttl=3600)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    gdf = gpd.read_file(url, layer=f"ADM_ADM_{level}")
    return gdf.to_crs(epsg=4326)

# ============================================================
# SENTINEL-1 FLOOD MASK (ROBUST)
# ============================================================

def get_flood_mask(aoi_ee, ref_start, ref_end, flood_start, flood_end, threshold=1.25):

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi_ee)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    ref = s1.filterDate(ref_start, ref_end).median().clip(aoi_ee)
    flood = s1.filterDate(flood_start, flood_end).min().clip(aoi_ee)

    ref_db = ref.log10().multiply(10).rename("ref_db")
    flood_db = flood.log10().multiply(10).rename("flood_db")

    diff = ref_db.subtract(flood_db).rename("diff_db")
    flooded = diff.gt(threshold)

    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

    flood_mask = (
        flooded
        .updateMask(slope.lt(5))
        .updateMask(gsw.lt(80))
        .rename("flood")
        .toByte()
        .selfMask()
    )

    return flood_mask

# ============================================================
# OSM DATA (OSMNX – SAFE)
# ============================================================

def get_osm_data(aoi_gdf):
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    try:
        tags = {
            "building": True,
            "highway": True,
            "amenity": ["hospital", "clinic", "school", "university"]
        }
        gdf = ox.features_from_bbox(
            north=maxy,
            south=miny,
            east=maxx,
            west=minx,
            tags=tags
        )
        return gdf[gdf.geometry.within(aoi_gdf.unary_union)]
    except Exception:
        return gpd.GeoDataFrame()

# ============================================================
# SIDEBAR – USER INPUT
# ============================================================

st.sidebar.header("🌍 Zone d'étude")

countries = {
    "Sénégal": "SEN",
    "Mali": "MLI",
    "Niger": "NER",
    "Burkina Faso": "BFA"
}

country = st.sidebar.selectbox("Pays", list(countries.keys()))
iso = countries[country]

level = st.sidebar.slider("Niveau administratif", 0, 4, 1)
gadm = load_gadm(iso, level)

name_col = f"NAME_{level}" if level > 0 else "COUNTRY"
zones = sorted(gadm[name_col].unique())
selected = st.sidebar.multiselect("Zones", zones)

if selected:
    aoi_gdf = gadm[gadm[name_col].isin(selected)]
else:
    aoi_gdf = gadm

st.sidebar.header("📅 Période")
start_flood = st.sidebar.date_input("Début inondation", datetime(2024, 8, 1))
end_flood = st.sidebar.date_input("Fin inondation", datetime(2024, 8, 31))

run = st.sidebar.button("🚀 Lancer l’analyse")

# ============================================================
# RUN ANALYSIS
# ============================================================

if run:

    with st.spinner("Analyse Sentinel-1 & OSM…"):

        merged = unary_union(aoi_gdf.geometry)
        aoi_ee = ee.Geometry(mapping(merged))

        flood_mask = get_flood_mask(
            aoi_ee,
            "2024-01-01", "2024-03-01",
            str(start_flood), str(end_flood)
        )

        osm = get_osm_data(aoi_gdf)

    # ========================================================
    # MAP
    # ========================================================

    st.subheader("🗺️ Zones inondées")

    m = folium.Map(
        location=[merged.centroid.y, merged.centroid.x],
        zoom_start=8,
        tiles="CartoDB positron"
    )

    flood_vis = {
        "bands": ["flood"],
        "min": 0,
        "max": 1,
        "palette": ["00FFFF"]
    }

    map_id = flood_mask.getMapId(flood_vis)

    folium.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name="Zones inondées",
        overlay=True
    ).add_to(m)

    folium.GeoJson(aoi_gdf, name="Limites admin").add_to(m)
    folium.LayerControl().add_to(m)

    st_folium(m, height=550, width="100%")

    # ========================================================
    # INDICATORS
    # ========================================================

    st.subheader("📊 Indicateurs clés")

    col1, col2, col3 = st.columns(3)

    col1.metric("Zones analysées", len(aoi_gdf))

    if not osm.empty:
        col2.metric("Bâtiments OSM", osm["building"].notna().sum())
        col3.metric("Infrastructures sensibles",
                    osm["amenity"].notna().sum())
    else:
        col2.metric("Bâtiments OSM", "N/A")
        col3.metric("Infrastructures sensibles", "N/A")

    st.success("✅ Analyse terminée sans erreur.")
