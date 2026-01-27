# ============================================================================
# FLOODWATCH WA – APPLICATION PROFESSIONNELLE INONDATIONS
# Auteur : Version stable corrigée
# ============================================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import ee
import json
import requests
from shapely.geometry import shape, mapping
from streamlit_folium import st_folium

# ============================================================================
# CONFIG STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="FloodWatch WA – Inondations",
    layout="wide",
    page_icon="🌊"
)

# ============================================================================
# INITIALISATION GEE (STREAMLIT CLOUD SAFE)
# ============================================================================
def init_gee():
    if not ee.data._initialized:
        key = json.loads(st.secrets["gee_service_account"])
        credentials = ee.ServiceAccountCredentials(
            key["client_email"], key_data=key
        )
        ee.Initialize(credentials)

init_gee()

# ============================================================================
# SIDEBAR – ZONE D'ÉTUDE
# ============================================================================
st.sidebar.title("🌍 Zone d’étude")

uploaded = st.sidebar.file_uploader(
    "Charger une zone (GeoJSON / SHP / KML)",
    type=["geojson", "shp", "kml"]
)

draw_zone = st.sidebar.checkbox("✏️ Dessiner la zone manuellement")

# ============================================================================
# LECTURE ZONE
# ============================================================================
def load_zone(file):
    gdf = gpd.read_file(file)
    return gdf.to_crs(4326)

aoi_gdf = None

if uploaded:
    aoi_gdf = load_zone(uploaded)

# ============================================================================
# CARTE DE DESSIN
# ============================================================================
if draw_zone and not aoi_gdf:
    st.info("Dessine la zone puis valide")
    m = folium.Map(location=[14.5, -14.5], zoom_start=7)
    folium.plugins.Draw(export=True).add_to(m)
    output = st_folium(m, height=500)

    if output and output.get("last_active_drawing"):
        geom = shape(output["last_active_drawing"]["geometry"])
        aoi_gdf = gpd.GeoDataFrame(geometry=[geom], crs=4326)

# ============================================================================
# STOP SI PAS DE ZONE
# ============================================================================
if aoi_gdf is None:
    st.stop()

aoi = ee.Geometry(mapping(aoi_gdf.geometry.iloc[0]))

# ============================================================================
# SENTINEL-1 – DÉTECTION INONDATION (ROBUSTE)
# ============================================================================
def detect_flood(aoi, d1, d2, d3, d4):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    ref = s1.filterDate(d1, d2).median()
    flood = s1.filterDate(d3, d4).median()

    diff = flood.subtract(ref)
    flood_mask = diff.lt(-1.25)

    return flood_mask.selfMask()

# ============================================================================
# POPULATION – WORLDPOP (RÉEL)
# ============================================================================
def population_total(aoi):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").median()
    stats = pop.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100,
        maxPixels=1e13
    )
    return int(stats.get("population").getInfo() or 0)

# ============================================================================
# BÂTIMENTS & ROUTES – OSM VIA OVERPASS
# ============================================================================
def osm_query(aoi, key):
    geom = aoi_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds

    query = f"""
    [out:json];
    (
      way["{key}"]({miny},{minx},{maxy},{maxx});
    );
    out geom;
    """

    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query
    )
    data = r.json()["elements"]

    geoms = []
    for el in data:
        if "geometry" in el:
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            geoms.append(shape({"type": "LineString", "coordinates": coords}))

    return gpd.GeoDataFrame(geometry=geoms, crs=4326)

buildings_gdf = osm_query(aoi, "building")
roads_gdf = osm_query(aoi, "highway")

# ============================================================================
# IMPACT INONDATION SUR OSM
# ============================================================================
flood_img = detect_flood(aoi, "2024-07-01", "2024-07-15", "2024-08-01", "2024-08-15")

def impacted(gdf):
    impacted = []
    for geom in gdf.geometry:
        pts = ee.Feature(ee.Geometry(mapping(geom)))
        val = flood_img.reduceRegion(
            ee.Reducer.anyNonZero(),
            pts.geometry(),
            scale=10,
            maxPixels=1e8
        ).values().get(0).getInfo()
        impacted.append(bool(val))
    gdf["impacted"] = impacted
    return gdf

buildings_gdf = impacted(buildings_gdf)
roads_gdf = impacted(roads_gdf)

# ============================================================================
# CLIMAT – NASA POWER (CORRIGÉ)
# ============================================================================
def climate(aoi, start, end):
    c = aoi.centroid().getInfo()["coordinates"]
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"latitude={c[1]}&longitude={c[0]}"
        f"&start={start.replace('-', '')}"
        f"&end={end.replace('-', '')}"
        "&parameters=PRECTOTCORR,T2M"
        "&community=AG&format=JSON"
    )
    data = requests.get(url).json()["properties"]["parameter"]
    rain = sum(data["PRECTOTCORR"].values())
    temp = np.mean(list(data["T2M"].values()))
    return rain, temp

rain, temp = climate(aoi, "2024-08-01", "2024-08-15")

# ============================================================================
# INDICATEURS
# ============================================================================
st.title("🌊 FloodWatch WA – Analyse d’impact")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Population totale", f"{population_total(aoi):,}")
c2.metric("Bâtiments", len(buildings_gdf))
c3.metric("Bâtiments impactés", buildings_gdf.impacted.sum())
c4.metric("Routes impactées", roads_gdf.impacted.sum())

st.metric("🌧️ Pluie cumulée (mm)", f"{rain:.1f}")
st.metric("🌡️ Température moyenne (°C)", f"{temp:.1f}")

# ============================================================================
# CARTE
# ============================================================================
m = folium.Map(location=[14.5, -14.5], zoom_start=8)

folium.GeoJson(
    buildings_gdf[buildings_gdf.impacted],
    style_function=lambda _: {"color": "red"}
).add_to(m)

folium.GeoJson(
    roads_gdf[roads_gdf.impacted],
    style_function=lambda _: {"color": "red"}
).add_to(m)

st_folium(m, height=600)
