# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP
# West Africa – Sentinel / CHIRPS / WorldPop / OSM / GAUL
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
st.caption("Sentinel-1 | CHIRPS | WorldPop | OpenStreetMap | FAO GAUL")

# ------------------------------------------------------------
# INIT GEE
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("Secret 'GEE_SERVICE_ACCOUNT' manquant dans Streamlit.")
        st.stop()
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
    ee.Initialize(credentials)
    return True

init_gee()

# ------------------------------------------------------------
# UTILS & PDF EXPORT
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
    pdf.cell(190, 8, f"- Batiments Impactes (Est.): {stats['bldgs']:,}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "2. Detail par Zone Administrative", ln=True)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 8, "Zone", border=1)
    pdf.cell(40, 8, "Surf. Inon. (km2)", border=1)
    pdf.cell(40, 8, "% Inonde", border=1)
    pdf.cell(50, 8, "Pop. Exposee", border=1, ln=True)
    
    pdf.set_font("Arial", "", 10)
    for _, row in df.iterrows():
        pdf.cell(60, 8, str(row['Zone'])[:25], border=1)
        pdf.cell(40, 8, f"{row['Inondé (km2)']:.2f}", border=1)
        pdf.cell(40, 8, f"{row['% Inondé']:.1f}%", border=1)
        pdf.cell(50, 8, f"{row['Pop. Exposée']:,}", border=1, ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

GAUL_COLLECTIONS = {
    "Admin 1": ee.FeatureCollection("FAO/GAUL/2015/level1"),
    "Admin 2": ee.FeatureCollection("FAO/GAUL/2015/level2")
}

# ------------------------------------------------------------
# SIDEBAR - SELECTION ADMINISTRATIVE
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Zone d'Étude")

country = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

a1_fc = GAUL_COLLECTIONS["Admin 1"].filter(ee.Filter.eq('ADM0_NAME', country))
a1_list = a1_fc.aggregate_array('ADM1_NAME').sort().getInfo()
sel_a1 = st.sidebar.multiselect(f"Régions (Admin 1)", a1_list)

final_aoi_fc = None
label_col = 'ADM1_NAME'

if sel_a1:
    a2_fc = GAUL_COLLECTIONS["Admin 2"].filter(ee.Filter.inList('ADM1_NAME', sel_a1))
    a2_list = a2_fc.aggregate_array('ADM2_NAME').sort().getInfo()
    sel_a2 = st.sidebar.multiselect("Départements (Admin 2)", a2_list)
    
    if sel_a2:
        final_aoi_fc = a2_fc.filter(ee.Filter.inList('ADM2_NAME', sel_a2))
        label_col = 'ADM2_NAME'
    else:
        final_aoi_fc = a2_fc
        label_col = 'ADM2_NAME'
else:
    st.info("Sélectionnez une région pour commencer.")
    st.stop()

with st.spinner("Chargement de la zone..."):
    gdf = gpd.GeoDataFrame.from_features(final_aoi_fc.getInfo(), crs="EPSG:4326")
    gdf = gdf.reset_index(drop=True)
    merged_poly = unary_union(gdf.geometry)
    geom_ee = ee.Geometry(mapping(merged_poly))

# ------------------------------------------------------------
# TEMPORAL CONFIG
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Paramètres Temporels")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2024-06-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2024-10-31"))

analysis_mode = st.sidebar.radio("Mode d'Analyse", ["Synthèse Globale", "Série Temporelle (Animée)"])
interval_days = 15 if st.sidebar.checkbox("Utiliser Quinzaines (vs Mois)", value=True) else 30

# ------------------------------------------------------------
# ANALYSIS ENGINE
# ------------------------------------------------------------
@st.cache_data
def get_flood_and_rain(aoi_json, start_str, end_str):
    aoi = ee.Geometry(aoi_json)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start_str, end_str)\
           .filter(ee.Filter.eq("instrumentMode","IW")).select("VV")
    
    if s1.size().getInfo() < 1: return None, None

    mean_img = s1.median()
    ref = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate("2024-01-01", "2024-03-31")\
            .filter(ee.Filter.eq("instrumentMode","IW")).select("VV").median()
    
    flood = mean_img.subtract(ref).lt(-3)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    flood_final = flood.updateMask(slope.lt(5)).selfMask()

    # Précipitations CHIRPS
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(start_str, end_str).sum().rename('precip')
    
    return flood_final, rain

# ------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------
df_final = pd.DataFrame()

if analysis_mode == "Série Temporelle (Animée)":
    st.subheader("🎞️ Évolution Temporelle : Inondation & Pluie")
    dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval_days}D')
    time_series_data = []
    
    col_anim, col_chart = st.columns([1, 1])
    
    with st.spinner("Génération de la série temporelle..."):
        images_list = []
        for i in range(len(dates)-1):
            d1, d2 = str(dates[i].date()), str(dates[i+1].date())
            f_img, r_img = get_flood_and_rain(geom_ee.getInfo(), d1, d2)
            
            if f_img:
                area = f_img.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=geom_ee, scale=100, maxPixels=1e9
                ).get('VV').getInfo()
                
                precip = r_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=geom_ee, scale=5000
                ).get('precip').getInfo()
                
                area_km2 = (area / 1e6) if area else 0
                time_series_data.append({"Date": d1, "Inondation (km2)": area_km2, "Pluie Cumulée (mm)": precip})
                vis_img = f_img.visualize(palette=['#00D4FF']).clip(geom_ee)
                images_list.append(vis_img.set('system:time_start', dates[i].value))

        if images_list:
            video_args = {'dimensions': 600, 'region': geom_ee, 'framesPerSecond': 2, 'crs': 'EPSG:3857'}
            gif_url = ee.ImageCollection(images_list).getVideoThumbURL(video_args)
            col_anim.image(gif_url, caption="Dynamique des eaux (Sentinel-1)", use_container_width=True)
            ts_df = pd.DataFrame(time_series_data)
            col_chart.line_chart(ts_df.set_index("Date"))
        else:
            st.warning("Données insuffisantes.")

# ------------------------------------------------------------
# FINAL CALCULATION & EXPORTS
# ------------------------------------------------------------
st.subheader("🗺️ Cartographie & Rapport de Synthèse")

with st.spinner("Calcul des indicateurs finaux..."):
    flood_final, rain_final = get_flood_and_rain(geom_ee.getInfo(), str(start_date), str(end_date))
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(geom_ee).mean().rename('pop')
    infra_fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(geom_ee)

    if flood_final:
        features = []
        for idx, row in gdf.iterrows():
            f = ee.Feature(ee.Geometry(mapping(row.geometry)), {
                'orig_index': int(idx), 'nom': str(row[label_col]), 'area_km2': get_true_area_km2(row.geometry)
            })
            features.append(f)
        fc = ee.FeatureCollection(features)
        
        pix_area = ee.Image.pixelArea()
        stats = ee.Image.cat([
            flood_final.multiply(pix_area).rename('f_area'),
            pop.updateMask(flood_final).rename('p_exp')
        ]).reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100).getInfo()

        impacted_rows = []
        for f in stats['features']:
            p = f['properties']
            f_km2 = (p.get('f_area', 0)) / 1e6
            total = p.get('area_km2', 1)
            impacted_rows.append({
                "orig_index": p['orig_index'], "Zone": p['nom'],
                "Inondé (km2)": f_km2, "% Inondé": (f_km2/total*100),
                "Pop. Exposée": int(p.get('p_exp', 0))
            })
        df_final = pd.DataFrame(impacted_rows)

        # Dashboard
        c1, c2, c3 = st.columns(3)
        total_inond = df_final['Inondé (km2)'].sum()
        total_pop = df_final['Pop. Exposée'].sum()
        c1.metric("Surface Inondée", f"{total_inond:.2f} km²")
        c2.metric("Population Exposée", f"{total_pop:,}")
        
        # Actions d'exportation
        st.sidebar.header("3️⃣ Exportation")
        
        # PDF
        report_stats = {'area': total_inond, 'pop': total_pop, 'bldgs': 0} # bldgs calculated on demand
        pdf_bytes = create_pdf_report(df_final, country, start_date, end_date, report_stats)
        st.sidebar.download_button("📄 Télécharger Rapport PDF", pdf_bytes, f"rapport_{country}.pdf", "application/pdf")
        
        # GeoJSON (GIS)
        geojson_data = gdf.merge(df_final, left_index=True, right_on="orig_index").to_json()
        st.sidebar.download_button("🌍 Exporter GIS (GeoJSON)", geojson_data, f"flood_data_{country}.geojson", "application/json")

        # Map
        m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=9, tiles="CartoDB dark_matter")
        mid = flood_final.getMapId({'palette':['#00D4FF']})
        folium.TileLayer(tiles=mid['tile_fetcher'].url_format, attr='GEE', name="Eau", overlay=True).add_to(m)
        
        for _, r in df_final.iterrows():
            geom = gdf.iloc[int(r['orig_index'])].geometry
            color = "red" if r['% Inondé'] > 5 else "orange"
            folium.GeoJson(geom, style_function=lambda x, c=color: {'fillColor': c, 'color': 'white', 'weight': 1, 'fillOpacity': 0.3}).add_to(m)
        
        st_folium(m, width="100%", height=500)

st.dataframe(df_final.drop(columns=['orig_index']), use_container_width=True)
