# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP
# West Africa – Sentinel / CHIRPS / WorldPop / OSM / FAO GAUL
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import pandas as pd
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from pyproj import Geod
import datetime
from fpdf import FPDF
import base64
import osmnx as ox

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Analyse d'Impact Inondations – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Analyse d'Impact Inondations & Planification d'Urgence")
st.caption("Sentinel-1 | CHIRPS | WorldPop | OSM | FAO GAUL (Admin 1-3)")

# ------------------------------------------------------------
# INIT GEE
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("Secret 'GEE_SERVICE_ACCOUNT' manquant dans Streamlit.")
        st.stop()
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Erreur d'initialisation GEE : {e}")
        return False

init_gee()

# ------------------------------------------------------------
# UTILS & EXPORTS
# ------------------------------------------------------------
def create_pdf_report(df, country, p1, p2, stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, f"Rapport d'Impact Inondation - {country}", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(190, 10, f"Periode: {p1} au {p2}", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "1. Resume des Indicateurs Clefs", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(190, 8, f"- Surface Inondee Totale: {stats['area']:.2f} km2", ln=True)
    pdf.cell(190, 8, f"- Population Exposee: {stats['pop']:,}", ln=True)
    pdf.cell(190, 8, f"- Batiments Touches: {stats['buildings']:,}", ln=True)
    pdf.cell(190, 8, f"- Routes Affectees: {stats['roads']:.1f} km", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "2. Detail par Zone Administrative", ln=True)
    pdf.set_font("Arial", "B", 7)
    cols = ["Zone", "Surf.(km2)", "Pop.Exp", "Bat.Touch", "Routes(km)"]
    for col in cols: pdf.cell(38, 8, col, border=1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 7)
    for _, row in df.iterrows():
        pdf.cell(38, 8, str(row['Zone'])[:22], border=1)
        pdf.cell(38, 8, f"{row['Inondé (km2)']:.2f}", border=1)
        pdf.cell(38, 8, f"{row['Pop. Exposée']:,}", border=1)
        pdf.cell(38, 8, f"{row['Bâtiments Affectés']:,}", border=1)
        pdf.cell(38, 8, f"{row['Routes Affectées (km)']:.1f}", border=1, ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

# ------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------
GAUL = ee.FeatureCollection("FAO/GAUL/2015/level2")
GAUL_A1 = ee.FeatureCollection("FAO/GAUL/2015/level1")

# ------------------------------------------------------------
# SIDEBAR - CASCADE ADMINISTRATIVE
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection Administrative")
country_name = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

C0, C1, C2 = 'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME'

def safe_get_info(ee_obj):
    try: return ee_obj.getInfo()
    except Exception as e:
        return None

a1_fc = GAUL_A1.filter(ee.Filter.eq(C0, country_name))
a1_list = safe_get_info(a1_fc.aggregate_array(C1).distinct().sort())
sel_a1 = st.sidebar.multiselect("Régions (Admin 1)", a1_list if a1_list else [])

final_aoi_fc = None
label_col = C1

if sel_a1:
    a2_fc = GAUL.filter(ee.Filter.eq(C0, country_name)).filter(ee.Filter.inList(C1, sel_a1))
    a2_list = safe_get_info(a2_fc.aggregate_array(C2).distinct().sort())
    sel_a2 = st.sidebar.multiselect("Zones (Admin 2)", a2_list if a2_list else [])
    
    if sel_a2:
        final_aoi_fc = a2_fc.filter(ee.Filter.inList(C2, sel_a2))
        label_col = C2
    else:
        final_aoi_fc = a1_fc.filter(ee.Filter.inList(C1, sel_a1))
        label_col = C1
else:
    st.info("Veuillez sélectionner au moins une région.")
    st.stop()

with st.spinner("Chargement de la zone d'étude..."):
    aoi_info = safe_get_info(final_aoi_fc)
    if not aoi_info or not aoi_info['features']:
        st.error("Aucune géométrie trouvée.")
        st.stop()
    gdf = gpd.GeoDataFrame.from_features(aoi_info, crs="EPSG:4326")
    merged_poly = unary_union(gdf.geometry)
    geom_ee = ee.Geometry(mapping(merged_poly))

# ------------------------------------------------------------
# TEMPORAL CONFIG
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Analyse Temporelle")
start_date = st.sidebar.date_input("Début", pd.to_datetime("2024-07-01"))
end_date = st.sidebar.date_input("Fin", pd.to_datetime("2024-10-31"))
analysis_mode = st.sidebar.radio("Mode", ["Synthèse Globale", "Série Temporelle Animée"])
interval = 15 if st.sidebar.checkbox("Quinzaines", value=True) else 30

# ------------------------------------------------------------
# CORE ENGINES
# ------------------------------------------------------------
@st.cache_data
def get_flood_and_rain(aoi_json, start_str, end_str):
    aoi = ee.Geometry(aoi_json)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start_str, end_str)\
           .filter(ee.Filter.eq("instrumentMode","IW")).select("VV")
    
    count = safe_get_info(s1.size())
    if count is None or count < 1: return None, None
    
    ref = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate("2024-01-01", "2024-03-31").median()
    flood = s1.median().subtract(ref).lt(-3).select(0)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(start_str, end_str).sum().rename('precip')
    
    return flood.updateMask(slope.lt(5)).selfMask(), rain

# ------------------------------------------------------------
# ANALYSE D'IMPACT INFRASTRUCTURE (OSMnx)
# ------------------------------------------------------------
@st.cache_data
def analyze_infrastructure_impact_osmnx(polygon_geojson):
    """
    Télécharge les infrastructures OSM pour un polygone donné.
    Renvoie deux GeoDataFrames: bâtiments et routes.
    """
    polygon_shapely = shape(polygon_geojson)
    try:
        buildings = ox.geometries_from_polygon(polygon_shapely, tags={"building": True})
        roads = ox.geometries_from_polygon(polygon_shapely, tags={"highway": True})
        return buildings, roads
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger OSM: {e}")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

with st.spinner("Analyse des infrastructures OSM..."):
    buildings_gdf, roads_gdf = analyze_infrastructure_impact_osmnx(mapping(merged_poly))

# Santé et Education
health_tags = ['hospital', 'clinic', 'doctors', 'pharmacy']
edu_tags = ['school', 'university', 'kindergarten', 'college']

health_gdf = buildings_gdf[buildings_gdf['amenity'].isin(health_tags)] if not buildings_gdf.empty and 'amenity' in buildings_gdf.columns else gpd.GeoDataFrame()
edu_gdf = buildings_gdf[buildings_gdf['amenity'].isin(edu_tags)] if not buildings_gdf.empty and 'amenity' in buildings_gdf.columns else gpd.GeoDataFrame()

# ------------------------------------------------------------
# ANALYSE & DASHBOARD
# ------------------------------------------------------------
flood_all, rain_all = get_flood_and_rain(mapping(merged_poly), str(start_date), str(end_date))
pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(geom_ee).mean().select(0)

features_list = []
for idx, row in gdf.iterrows():
    f_geom = row.geometry
    loc_stats = safe_get_info(ee.Image.cat([
        flood_all.multiply(ee.Image.pixelArea()).rename('f_area'),
        pop_img.updateMask(flood_all.select(0)).rename('p_exp')
    ]).reduceRegion(ee.Reducer.sum(), ee.Geometry(mapping(f_geom)), 250))

    f_km2 = (loc_stats.get('f_area', 0) if loc_stats else 0) / 1e6
    p_exp = (loc_stats.get('p_exp', 0) if loc_stats else 0)

    b_count = len(buildings_gdf[buildings_gdf.intersects(f_geom)]) if not buildings_gdf.empty else 0
    h_count = len(health_gdf[health_gdf.intersects(f_geom)]) if not health_gdf.empty else 0
    e_count = len(edu_gdf[edu_gdf.intersects(f_geom)]) if not edu_gdf.empty else 0
    r_count = len(roads_gdf[roads_gdf.intersects(f_geom)]) if not roads_gdf.empty else 0

    features_list.append({
        "Zone": row[label_col],
        "Inondé (km2)": round(f_km2, 2),
        "% Inondé": round((f_km2 / get_true_area_km2(f_geom) * 100), 1) if f_km2 > 0 else 0,
        "Pop. Exposée": int(p_exp),
        "Bâtiments": int(b_count),
        "Santé": int(h_count),
        "Éducation": int(e_count),
        "Segments Route": int(r_count),
        "orig_id": idx
    })

df_res = pd.DataFrame(features_list)

# ------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------
m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=9, tiles="CartoDB dark_matter")
if flood_all:
    mid = flood_all.select(0).getMapId({'palette':['#00D4FF']})
    folium.TileLayer(tiles=mid['tile_fetcher'].url_format, attr='GEE', name="Zones Inondées", overlay=True).add_to(m)

for _, r in df_res.iterrows():
    geom = gdf.iloc[int(r['orig_id'])].geometry
    pop_html = f"<b>{r['Zone']}</b><br>Pop Exp: {r['Pop. Exposée']:,}<br>Inondé: {r['Inondé (km2)']} km²"
    folium.GeoJson(geom, style_function=lambda x: {'fillColor': 'red', 'color': 'white', 'weight': 1, 'fillOpacity': 0.1},
                   popup=folium.Popup(pop_html, max_width=200)).add_to(m)

st_folium(m, width="100%", height=500)

# Dashboard metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pop. Exposée", f"{df_res['Pop. Exposée'].sum():,}")
c2.metric("Bâtiments", f"{df_res['Bâtiments'].sum():,}" if not buildings_gdf.empty else "N/A")
c3.metric("🏥 Santé", f"{df_res['Santé'].sum():,}" if not health_gdf.empty else "N/A")
c4.metric("🎓 Éducation", f"{df_res['Éducation'].sum():,}" if not edu_gdf.empty else "N/A")
c5.metric("🛣️ Routes", f"{df_res['Segments Route'].sum():,}" if not roads_gdf.empty else "N/A")

# Export PDF
st.sidebar.header("3️⃣ Export")
pdf_b = create_pdf_report(df_res.rename(columns={'Bâtiments': 'Bâtiments Affectés', 'Segments Route': 'Routes Affectées'}), country_name, start_date, end_date, {
    'area': df_res['Inondé (km2)'].sum(), 
    'pop': df_res['Pop. Exposée'].sum(),
    'buildings': df_res['Bâtiments'].sum(), 
    'roads': df_res['Segments Route'].sum(),
    'rain': safe_get_info(rain_all.reduceRegion(ee.Reducer.mean(), geom_ee, 2000)).get('precip',0) if rain_all else 0
})
st.sidebar.download_button("📄 Télécharger Rapport Décisionnel", pdf_b, "rapport_impact.pdf")

# Tableau de données
st.dataframe(df_res.drop(columns=['orig_id']), use_container_width=True)
