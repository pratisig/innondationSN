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
from datetime import datetime

# ===============================================================
# 1. CONFIGURATION
# ===============================================================
st.set_page_config(
    page_title="FloodWatch WA Pro",
    page_icon="🌊",
    layout="wide"
)

ox.settings.use_cache = True
ox.settings.timeout = 180

# ===============================================================
# 2. INITIALISATION GEE
# ===============================================================
@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key["client_email"],
                key_data=json.dumps(key)
            )
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"GEE init error: {e}")
        return False

gee_available = init_gee()

# ===============================================================
# 3. DONNÉES ADMIN
# ===============================================================
@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
    return gpd.read_file(url, layer=level).to_crs(4326)

# ===============================================================
# 4. DÉTECTION INONDATION SAR (AMÉLIORÉE)
# ===============================================================
def advanced_flood_detection(aoi, ref_start, ref_end, flood_start, flood_end,
                             threshold_db=1.25, min_pixels=20):

    def s1(start, end):
        return (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(aoi)
                .filterDate(start, end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .select("VV"))

    ref = s1(ref_start, ref_end).median().clip(aoi)
    crisis = s1(flood_start, flood_end).reduce(
        ee.Reducer.percentile([10])
    ).rename("VV").clip(aoi)

    ref_db = ref.log10().multiply(10)
    crisis_db = crisis.log10().multiply(10)

    flood = ref_db.subtract(crisis_db).gt(threshold_db)

    slope = ee.Algorithms.Terrain(
        ee.Image("USGS/SRTMGL1_003")
    ).select("slope")

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

    flood = (
        flood
        .updateMask(slope.lt(5))
        .updateMask(jrc.lt(80))
    )

    flood = flood.focal_mode(1)
    connected = flood.connectedPixelCount(100)
    flood = flood.updateMask(connected.gte(min_pixels))

    return flood.rename("flood").selfMask()

# ===============================================================
# 5. POPULATION & SURFACE – POLYGONE PAR POLYGONE
# ===============================================================
def population_and_area_by_polygon(poly_geom, flood_mask):
    pop_img = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterDate("2020-01-01", "2021-01-01")
        .mosaic()
        .clip(poly_geom)
    )

    total_pop = pop_img.reduceRegion(
        ee.Reducer.sum(),
        poly_geom,
        scale=100,
        maxPixels=1e9
    ).get("population")

    exposed_pop = 0
    flood_area = 0

    if flood_mask:
        flood_local = flood_mask.clip(poly_geom)

        exposed_pop = pop_img.updateMask(flood_local).reduceRegion(
            ee.Reducer.sum(),
            poly_geom,
            scale=100,
            maxPixels=1e9
        ).get("population")

        flood_area = flood_local.multiply(
            ee.Image.pixelArea()
        ).reduceRegion(
            ee.Reducer.sum(),
            poly_geom,
            scale=10,
            maxPixels=1e9
        ).get("flood")

    return (
        int(total_pop.getInfo() or 0),
        int(exposed_pop.getInfo() or 0),
        float((flood_area.getInfo() or 0) / 10000)
    )

# ===============================================================
# 6. OSM
# ===============================================================
def get_osm_assets(gdf):
    poly = gdf.unary_union

    buildings = ox.features_from_polygon(
        poly, tags={"building": True}
    )
    buildings = buildings[
        buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].reset_index().clip(gdf)

    graph = ox.graph_from_polygon(poly, network_type="all")
    roads = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index().clip(gdf)

    return buildings, roads

# ===============================================================
# 7. INTERFACE
# ===============================================================
st.sidebar.header("🗺️ Zone")
countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
country = st.sidebar.selectbox("Pays", list(countries))
level = st.sidebar.slider("Niveau Admin", 0, 4, 2)

gdf_admin = load_gadm(countries[country], level)
name_col = f"NAME_{level}" if level > 0 else "COUNTRY"
choice = st.sidebar.multiselect("Zone(s)", sorted(gdf_admin[name_col].unique()))
zone = gdf_admin[gdf_admin[name_col].isin(choice)] if choice else None

st.sidebar.header("📅 Dates")
ref = st.sidebar.date_input("Référence sèche", [datetime(2023,1,1), datetime(2023,4,30)])
flood = st.sidebar.date_input("Crise", [datetime(2024,8,1), datetime(2024,10,31)])

sens = st.sidebar.slider("Sensibilité SAR (dB)", 0.5, 5.0, 1.25)

# ===============================================================
# 8. ANALYSE
# ===============================================================
st.title("🌊 FloodWatch WA – Analyse SAR précise")

if zone is not None and st.button("🚀 Lancer l'analyse", use_container_width=True):

    with st.spinner("Analyse GEE en cours..."):
        aoi = ee.Geometry(mapping(zone.unary_union))
        flood_mask = advanced_flood_detection(
            aoi,
            str(ref[0]), str(ref[1]),
            str(flood[0]), str(flood[1]),
            sens
        )

        total_pop = exposed_pop = total_area = 0

        for _, row in zone.iterrows():
            geom = ee.Geometry(mapping(row.geometry))
            t, e, a = population_and_area_by_polygon(geom, flood_mask)
            total_pop += t
            exposed_pop += e
            total_area += a

        buildings, roads = get_osm_assets(zone)

    # ===========================================================
    # KPI
    # ===========================================================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population totale", f"{total_pop:,}")
    c2.metric("Population exposée", f"{exposed_pop:,}", f"{exposed_pop/total_pop*100:.1f}%")
    c3.metric("Surface inondée", f"{total_area:,.1f} ha")
    c4.metric("Bâtiments", len(buildings))

    # ===========================================================
    # CARTE
    # ===========================================================
    center = zone.centroid.iloc[0]
    m = folium.Map([center.y, center.x], zoom_start=11, tiles="cartodbpositron")

    try:
        mid = flood_mask.getMapId({"palette": ["#00bfff"]})
        folium.TileLayer(
            tiles=mid["tile_fetcher"].url_format,
            name="Inondation",
            overlay=True
        ).add_to(m)
    except:
        pass

    folium.GeoJson(zone, name="Zone").add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, height=600, use_container_width=True)

else:
    st.info("Sélectionne une zone et lance l'analyse.")
