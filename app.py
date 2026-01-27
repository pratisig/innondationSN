# ==========================================================
# FLOODWATCH WA – VERSION FINALE STABLE
# ==========================================================

import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping
import ee
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="FloodWatch WA Pro",
    page_icon="🌊",
    layout="wide"
)

# ==========================================================
# INIT GEE (SERVICE ACCOUNT)
# ==========================================================
@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(
        key["client_email"],
        key_data=json.dumps(key)
    )
    ee.Initialize(credentials)
    return True

GEE_OK = init_gee()

# ==========================================================
# SESSION STATE
# ==========================================================
if "zone" not in st.session_state:
    st.session_state.zone = None

if "results" not in st.session_state:
    st.session_state.results = {
        "pop_total": 0,
        "pop_exp": 0,
        "bld_total": 0,
        "bld_imp": 0,
        "rds_total": 0,
        "rds_imp": 0,
        "climate": pd.DataFrame(),
        "flood_tiles": None,
        "bld_gdf": gpd.GeoDataFrame(),
        "rds_gdf": gpd.GeoDataFrame()
    }

# ==========================================================
# GADM LOADER (ADMIN 0–4)
# ==========================================================
@st.cache_data
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
    return gpd.read_file(url, layer=f"ADM_ADM_{level}").to_crs(4326)

# ==========================================================
# FLOOD DETECTION (SAR – CORRIGÉ)
# ==========================================================
def detect_flood(aoi, d0, d1, d2, d3):
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(aoi) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .select("VV")

    ref = s1.filterDate(d0, d1).median()
    flood = s1.filterDate(d2, d3).reduce(ee.Reducer.percentile([10]))

    diff = ee.Image(10).multiply(ref.log10()) \
        .subtract(ee.Image(10).multiply(flood.log10()))

    water = diff.gt(1.25)

    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    water = water.updateMask(slope.lt(5))

    return water.selfMask().clip(aoi)

# ==========================================================
# POPULATION WORLDPOP
# ==========================================================
def population_stats(aoi, flood_mask):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filterDate("2020-01-01", "2021-01-01") \
        .mosaic() \
        .select("population") \
        .clip(aoi)

    total = pop.reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo() or 0

    exposed = pop.updateMask(flood_mask).reduceRegion(
        ee.Reducer.sum(), aoi, 100, maxPixels=1e9
    ).get("population").getInfo() or 0

    return int(total), int(exposed)

# ==========================================================
# CLIMAT CHIRPS
# ==========================================================
def climate_stats(aoi, start, end):
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(aoi) \
        .filterDate(start, end)

    def f(img):
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "rain": img.reduceRegion(
                ee.Reducer.mean(), aoi, 5000
            ).get("precipitation")
        })

    data = rain.map(f).getInfo()
    df = pd.DataFrame([d["properties"] for d in data])
    df["date"] = pd.to_datetime(df["date"])
    return df

# ==========================================================
# OSM – IMPACT INFRASTRUCTURES
# ==========================================================
def osm_assets(zone, flood_mask):
    poly = zone.unary_union

    bld = ox.features_from_polygon(poly, tags={"building": True})
    rds = ox.features_from_polygon(poly, tags={"highway": True})

    bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])]
    rds = rds[rds.geometry.type.isin(["LineString", "MultiLineString"])]

    def impact(gdf):
        pts = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point(g.centroid.x, g.centroid.y))
            for g in gdf.geometry
        ])
        vals = flood_mask.sampleRegions(pts, scale=10) \
            .aggregate_array("flood").getInfo()
        gdf["impacted"] = [v == 1 for v in vals]
        return gdf

    return impact(bld), impact(rds)

# ==========================================================
# SIDEBAR – ZONE D’ÉTUDE
# ==========================================================
with st.sidebar:
    st.header("🗺️ Zone d’étude")

    mode = st.radio("Méthode", ["GADM", "Upload", "Dessin"])

    if mode == "GADM":
        country = st.selectbox("Pays", ["SEN", "MLI", "NER", "BFA"])
        level = st.slider("Niveau admin", 0, 4, 2)
        gadm = load_gadm(country, level)

        col = f"NAME_{level}" if level > 0 else "COUNTRY"
        units = st.multiselect("Unités", gadm[col].unique())
        if units:
            st.session_state.zone = gadm[gadm[col].isin(units)]

    elif mode == "Upload":
        f = st.file_uploader("GeoJSON / SHP / KML", type=["geojson", "json", "shp", "kml"])
        if f:
            st.session_state.zone = gpd.read_file(f).to_crs(4326)

    elif mode == "Dessin":
        m0 = folium.Map(location=[14.5, -14.5], zoom_start=6)
        Draw().add_to(m0)
        res = st_folium(m0, height=300)
        if res and res.get("last_active_drawing"):
            geom = shape(res["last_active_drawing"]["geometry"])
            st.session_state.zone = gpd.GeoDataFrame(geometry=[geom], crs=4326)

    st.subheader("📅 Dates")
    ref = st.date_input("Période sèche", [datetime(2023,1,1), datetime(2023,4,30)])
    flood = st.date_input("Période inondation", [datetime(2024,8,1), datetime(2024,10,30)])

# ==========================================================
# ANALYSE
# ==========================================================
if st.session_state.zone is not None and GEE_OK:
    zone = st.session_state.zone
    aoi = ee.Geometry(mapping(zone.unary_union))

    flood_mask = detect_flood(
        aoi, str(ref[0]), str(ref[1]), str(flood[0]), str(flood[1])
    )

    pop_total, pop_exp = population_stats(aoi, flood_mask)
    climate = climate_stats(aoi, str(flood[0]), str(flood[1]))
    bld, rds = osm_assets(zone, flood_mask)

    st.session_state.results = {
        "pop_total": pop_total,
        "pop_exp": pop_exp,
        "bld_total": len(bld),
        "bld_imp": int(bld.impacted.sum()),
        "rds_total": len(rds),
        "rds_imp": int(rds.impacted.sum()),
        "climate": climate,
        "flood_tiles": flood_mask.getMapId({"palette": ["0000ff"]}),
        "bld_gdf": bld,
        "rds_gdf": rds
    }

# ==========================================================
# DASHBOARD
# ==========================================================
res = st.session_state.results
st.title("🌊 FloodWatch WA – Résultats")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Population totale", f"{res['pop_total']:,}")
c2.metric("Population impactée", f"{res['pop_exp']:,}")
c3.metric("Bâtiments impactés", res["bld_imp"])
c4.metric("Routes impactées", res["rds_imp"])

# ==========================================================
# MAP
# ==========================================================
m = folium.Map(location=[14.5, -14.5], zoom_start=6)

if res["flood_tiles"]:
    folium.TileLayer(
        tiles=res["flood_tiles"]["tile_fetcher"].url_format,
        attr="GEE",
        name="Inondation"
    ).add_to(m)

if not res["bld_gdf"].empty:
    folium.GeoJson(
        res["bld_gdf"][res["bld_gdf"].impacted],
        style_function=lambda x: {"color": "red", "fillColor": "red"},
        name="Bâtiments impactés"
    ).add_to(m)

if not res["rds_gdf"].empty:
    folium.GeoJson(
        res["rds_gdf"][res["rds_gdf"].impacted],
        style_function=lambda x: {"color": "red"},
        name="Routes impactées"
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, height=600, use_container_width=True)

# ==========================================================
# CLIMAT
# ==========================================================
if not res["climate"].empty:
    st.subheader("🌧️ Pluviométrie")
    st.plotly_chart(
        px.bar(res["climate"], x="date", y="rain"),
        use_container_width=True
    )
