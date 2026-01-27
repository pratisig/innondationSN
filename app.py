# ============================================================
# FLOOD IMPACT ANALYSIS — WEST AFRICA
# Admin selection (GADM) OR Upload
# Sentinel-1 + WorldPop + OSMnx
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import ee
import osmnx as ox
import json
import tempfile
import zipfile
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Geod
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Flood Impact Analysis",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Flood Impact Analysis – Decision Support Tool")
st.caption("Sentinel-1 | WorldPop | OpenStreetMap")

# ============================================================
# GEE INIT
# ============================================================
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("❌ Missing GEE_SERVICE_ACCOUNT in secrets.")
        st.stop()
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(
        key["client_email"],
        key_data=json.dumps(key)
    )
    ee.Initialize(credentials)

init_gee()

# ============================================================
# UTILS
# ============================================================
geod = Geod(ellps="WGS84")

def area_km2(geom):
    return abs(geod.geometry_area_perimeter(geom)[0]) / 1e6

def to_ee(geom):
    return ee.Geometry(geom.__geo_interface__)

def clean_gdf(gdf):
    return gdf[gdf.geometry.notnull()].copy()

# ============================================================
# SIDEBAR — ZONE SELECTION
# ============================================================
st.sidebar.header("1️⃣ Zone d’étude")

mode = st.sidebar.radio(
    "Méthode de sélection",
    ["Sélection administrative (GADM)", "Uploader un fichier"]
)

aoi_gdf = None

# ------------------------------------------------------------
# OPTION 1 — GADM
# ------------------------------------------------------------
if mode == "Sélection administrative (GADM)":
    country_map = {
        "Sénégal": "SEN",
        "Mali": "MLI",
        "Niger": "NER",
        "Burkina Faso": "BFA"
    }
    country = st.sidebar.selectbox("Pays", list(country_map.keys()))
    iso = country_map[country]

    level = st.sidebar.slider("Niveau administratif", 0, 3, 1)

    @st.cache_data(ttl=86400)
    def load_gadm(iso, level):
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
        return gpd.read_file(url, layer=level).to_crs(4326)

    gadm = load_gadm(iso, level)
    name_col = f"NAME_{level}" if level > 0 else "COUNTRY"
    choices = st.sidebar.multiselect("Zone(s)", sorted(gadm[name_col].unique()))

    if choices:
        aoi_gdf = gadm[gadm[name_col].isin(choices)]
    else:
        aoi_gdf = gadm

# ------------------------------------------------------------
# OPTION 2 — UPLOAD
# ------------------------------------------------------------
else:
    file = st.sidebar.file_uploader(
        "Charger un fichier (GeoJSON / SHP ZIP / KML)",
        type=["geojson", "zip", "kml"]
    )

    if file:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/{file.name}"
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".zip"):
                with zipfile.ZipFile(path) as z:
                    z.extractall(tmp)
                shp = [f for f in z.namelist() if f.endswith(".shp")][0]
                aoi_gdf = gpd.read_file(f"{tmp}/{shp}")
            else:
                aoi_gdf = gpd.read_file(path)

            if aoi_gdf.crs is None:
                aoi_gdf.set_crs(4326, inplace=True)
            else:
                aoi_gdf = aoi_gdf.to_crs(4326)

# ============================================================
# DEFAULT VIEW (NO PAGE BLANK)
# ============================================================
if aoi_gdf is None:
    m = folium.Map(location=[14, -14], zoom_start=5, tiles="CartoDB positron")
    st_folium(m, height=550)
    st.info("⬅️ Sélectionnez une zone et une période pour lancer l’analyse.")
    st.stop()

# ============================================================
# AOI PREP
# ============================================================
aoi_gdf = clean_gdf(aoi_gdf)
aoi_geom = unary_union(aoi_gdf.geometry)
aoi_ee = to_ee(aoi_geom)
total_area = area_km2(aoi_geom)

# ============================================================
# DATE SELECTION
# ============================================================
st.sidebar.header("2️⃣ Période")
start = st.sidebar.date_input("Début", datetime(2024, 8, 1))
end = st.sidebar.date_input("Fin", datetime(2024, 8, 31))

# ============================================================
# SENTINEL-1 FLOOD
# ============================================================
def flood_mask(aoi, start, end):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .select("VV"))

    before = s1.sort("system:time_start").limit(5).median()
    after = s1.sort("system:time_start", False).limit(5).median()

    flood = after.subtract(before).lt(-3)

    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

    return flood.updateMask(slope.lt(5)).updateMask(perm.lt(80)).selfMask()

flood = flood_mask(aoi_ee, str(start), str(end))

# ============================================================
# POPULATION
# ============================================================
pixel = ee.Image.pixelArea()
pop = (ee.ImageCollection("WorldPop/GP/100m/pop")
       .filterDate("2020-01-01", "2020-12-31")
       .mean())

stats = ee.Image.cat([
    flood.multiply(pixel).rename("flood_area"),
    pop.rename("pop_total"),
    pop.updateMask(flood).rename("pop_flood")
]).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=aoi_ee,
    scale=100,
    maxPixels=1e13
).getInfo()

flood_km2 = (stats.get("flood_area") or 0) / 1e6
pop_total = int(stats.get("pop_total") or 0)
pop_flood = int(stats.get("pop_flood") or 0)

# ============================================================
# OSMNX — INFRASTRUCTURES
# ============================================================
tags = {
    "building": True,
    "highway": True,
    "amenity": ["hospital", "clinic", "school"]
}

bbox = aoi_gdf.total_bounds  # minx, miny, maxx, maxy
osm = ox.features_from_bbox(
    bbox[3], bbox[1], bbox[2], bbox[0],
    tags=tags
)

osm = osm[osm.geometry.within(aoi_geom)]

buildings = osm[osm.geometry.type.isin(["Polygon", "MultiPolygon"]) & osm.get("building").notnull()]
roads = osm[osm.geometry.type.isin(["LineString", "MultiLineString"]) & osm.get("highway").notnull()]
infra = osm[osm.get("amenity").notnull()]

roads_km = roads.to_crs(3857).length.sum() / 1000

# ============================================================
# DASHBOARD
# ============================================================
st.subheader("📊 Indicateurs clés")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Surface inondée", f"{flood_km2:.2f} km²")
c2.metric("Population exposée", f"{pop_flood:,}")
c3.metric("Bâtiments", f"{len(buildings):,}")
c4.metric("Routes", f"{roads_km:.1f} km")

# ============================================================
# MAP
# ============================================================
st.subheader("🗺️ Carte")

center = aoi_gdf.geometry.centroid
m = folium.Map(
    location=[center.y.mean(), center.x.mean()],
    zoom_start=8,
    tiles="CartoDB positron"
)

flood_id = flood.getMapId({"palette": ["00FFFF"]})
folium.TileLayer(
    tiles=flood_id["tile_fetcher"].url_format,
    name="Zones inondées",
    overlay=True
).add_to(m)

folium.GeoJson(aoi_gdf, name="Zone").add_to(m)
folium.LayerControl().add_to(m)

st_folium(m, height=600)
