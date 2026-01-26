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
OSM_BUILDINGS = ee.FeatureCollection("projects/google/osm/buildings")
OSM_ROADS = ee.FeatureCollection("projects/google/osm/roads")

# ------------------------------------------------------------
# SIDEBAR - CASCADE ADMINISTRATIVE
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection Administrative")
country_name = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

C0, C1, C2 = 'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME'

def safe_get_info(ee_obj):
    try: return ee_obj.getInfo()
    except Exception as e:
        st.error(f"Erreur GEE : {e}")
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
# ANALYSE D'IMPACT INFRASTRUCTURE (OSM)
# ------------------------------------------------------------
def analyze_infrastructure_impact(flood_img, aoi_ee):
    # Buffer de l'inondation pour l'intersection vectorielle
    flood_vec = flood_img.reduceToVectors(geometry=aoi_ee, scale=100, eightConnected=True)
    
    # 1. Bâtiments
    buildings = OSM_BUILDINGS.filterBounds(aoi_ee)
    affected_buildings = buildings.filterBounds(flood_vec)
    
    # 2. Routes
    roads = OSM_ROADS.filterBounds(aoi_ee)
    affected_roads = roads.filterBounds(flood_vec)
    
    return affected_buildings, affected_roads

# ------------------------------------------------------------
# VISUALISATION & IMPACT
# ------------------------------------------------------------
if analysis_mode == "Série Temporelle Animée":
    st.subheader("🎞️ Évolution Temporelle")
    dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval}D')
    ts_rows = []
    images = []

    with st.spinner("Calcul de la série temporelle..."):
        for i in range(len(dates)-1):
            d1, d2 = str(dates[i].date()), str(dates[i+1].date())
            f, r = get_flood_and_rain(geom_ee.getInfo(), d1, d2)
            if f:
                area_res = safe_get_info(f.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), geom_ee, 300))
                area_val = area_res.get('VV', 0) if area_res else 0
                ts_rows.append({"Date": d1, "Surface (km2)": (area_val/1e6)})
                images.append(f.visualize(palette=['#00D4FF']))

        if images:
            col_a, col_b = st.columns([2, 1])
            gif_url = ee.ImageCollection(images).getVideoThumbURL({'dimensions': 600, 'region': geom_ee, 'framesPerSecond': 2})
            col_a.image(gif_url, use_container_width=True)
            col_b.line_chart(pd.DataFrame(ts_rows).set_index("Date"))

st.subheader("🗺️ Analyse d'Impact Spatiale & Infrastructures")

with st.spinner("Analyse approfondie (Population & OSM)..."):
    flood_all, rain_all = get_flood_and_rain(geom_ee.getInfo(), str(start_date), str(end_date))
    pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(geom_ee).mean().select(0)

    if flood_all:
        aff_buildings, aff_roads = analyze_infrastructure_impact(flood_all, geom_ee)
        rain_stats = safe_get_info(rain_all.reduceRegion(ee.Reducer.mean(), geom_ee, 2000))
        total_rain = rain_stats.get('precip', 0) if rain_stats else 0
        
        features_list = []
        for idx, row in gdf.iterrows():
            f_geom = ee.Geometry(mapping(row.geometry))
            
            # Stats Population & Surface
            loc_stats = safe_get_info(ee.Image.cat([
                flood_all.multiply(ee.Image.pixelArea()).rename('f_area'),
                pop_img.updateMask(flood_all.select(0)).rename('p_exp')
            ]).reduceRegion(ee.Reducer.sum(), f_geom, 250))
            
            # Stats OSM (Nombre de bâtiments et longueur de routes)
            b_count = safe_get_info(aff_buildings.filterBounds(f_geom).size())
            # Simplification : calcul longueur approximative des routes affectées
            r_count = safe_get_info(aff_roads.filterBounds(f_geom).aggregate_sum('length'))
            r_km = (float(r_count) / 1000.0) if r_count else 0
            
            f_km2 = (loc_stats.get('f_area', 0) if loc_stats else 0) / 1e6
            p_exp = (loc_stats.get('p_exp', 0) if loc_stats else 0)
            
            features_list.append({
                "Zone": row[label_col],
                "Inondé (km2)": round(f_km2, 2),
                "% Inondé": round((f_km2 / get_true_area_km2(row.geometry) * 100), 1) if f_km2 > 0 else 0,
                "Pop. Exposée": int(p_exp),
                "Bâtiments Affectés": int(b_count) if b_count else 0,
                "Routes Affectées (km)": round(r_km, 1),
                "orig_id": idx
            })
            
        df_res = pd.DataFrame(features_list)

        m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=9, tiles="CartoDB dark_matter")
        mid = flood_all.select(0).getMapId({'palette':['#00D4FF']})
        folium.TileLayer(tiles=mid['tile_fetcher'].url_format, attr='GEE', name="Zones Inondées", overlay=True).add_to(m)

        for _, r in df_res.iterrows():
            geom = gdf.iloc[int(r['orig_id'])].geometry
            pop_html = f"<b>{r['Zone']}</b><br>Bâtiments affectés: {r['Bâtiments Affectés']}<br>Routes: {r['Routes Affectées (km)']} km"
            folium.GeoJson(geom, style_function=lambda x: {'fillColor': 'red', 'color': 'white', 'weight': 1, 'fillOpacity': 0.1},
                           popup=folium.Popup(pop_html, max_width=200)).add_to(m)

        st_folium(m, width="100%", height=500)

        st.write("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Surface Inondée", f"{df_res['Inondé (km2)'].sum():.2f} km²")
        c2.metric("Pop. Exposée", f"{df_res['Pop. Exposée'].sum():,}")
        c3.metric("Bâtiments Touchés", f"{df_res['Bâtiments Affectés'].sum():,}")
        c4.metric("Routes Coupées", f"{df_res['Routes Affectées (km)'].sum():.1f} km")

        st.sidebar.header("3️⃣ Export")
        pdf_b = create_pdf_report(df_res, country_name, start_date, end_date, {
            'area': df_res['Inondé (km2)'].sum(), 'pop': df_res['Pop. Exposée'].sum(),
            'buildings': df_res['Bâtiments Affectés'].sum(), 'roads': df_res['Routes Affectées (km)'].sum(), 'rain': total_rain
        })
        st.sidebar.download_button("📄 Télécharger Rapport Décisionnel", pdf_b, "rapport_decision_urgence.pdf")

if 'df_res' in locals():
    st.dataframe(df_res.drop(columns=['orig_id']), use_container_width=True)
