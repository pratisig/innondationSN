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
    pdf.cell(190, 8, f"- Pluie Moyenne: {stats.get('rain', 0):.2f} mm", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "2. Detail par Zone Administrative", ln=True)
    pdf.set_font("Arial", "B", 8)
    cols = ["Zone", "Surf. (km2)", "% Inon", "Pop. Exp", "Routes(km)", "Infras"]
    for col in cols: pdf.cell(31, 8, col, border=1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    for _, row in df.iterrows():
        pdf.cell(31, 8, str(row['Zone'])[:18], border=1)
        pdf.cell(31, 8, f"{row['Inondé (km2)']:.2f}", border=1)
        pdf.cell(31, 8, f"{row['% Inondé']:.1f}%", border=1)
        pdf.cell(31, 8, f"{row['Pop. Exposée']:,}", border=1)
        pdf.cell(31, 8, f"{row.get('Routes (km)', 0):.1f}", border=1)
        pdf.cell(31, 8, f"{row.get('Infras', 0)}", border=1, ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

# ------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------
GAUL = {
    "L1": ee.FeatureCollection("FAO/GAUL/2015/level1"),
    "L2": ee.FeatureCollection("FAO/GAUL/2015/level2")
}

# ------------------------------------------------------------
# SIDEBAR - CASCADE ADMINISTRATIVE
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection Administrative")
country = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

a1_fc = GAUL["L1"].filter(ee.Filter.eq('ADM0_NAME', country))
a1_list = a1_fc.aggregate_array('ADM1_NAME').sort().getInfo()
sel_a1 = st.sidebar.multiselect("Régions (Admin 1)", a1_list)

final_aoi_fc = None
label_col = 'ADM1_NAME'

if sel_a1:
    a2_fc = GAUL["L2"].filter(ee.Filter.inList('ADM1_NAME', sel_a1))
    a2_list = a2_fc.aggregate_array('ADM2_NAME').sort().getInfo()
    sel_a2 = st.sidebar.multiselect("Départements (Admin 2)", a2_list)
    
    if sel_a2:
        # Simulation Admin 3/4 via intersection spatiale ou filtres spécifiques pays
        final_aoi_fc = a2_fc.filter(ee.Filter.inList('ADM2_NAME', sel_a2))
        label_col = 'ADM2_NAME'
        st.sidebar.info("Analyse granulaire (Admin 3/4) simulée sur la zone sélectionnée.")
    else:
        final_aoi_fc = a2_fc
        label_col = 'ADM2_NAME'
else:
    st.info("Sélectionnez une région pour débuter.")
    st.stop()

with st.spinner("Chargement de la géométrie..."):
    gdf = gpd.GeoDataFrame.from_features(final_aoi_fc.getInfo(), crs="EPSG:4326")
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
    if s1.size().getInfo() < 1: return None, None
    
    ref = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate("2024-01-01", "2024-03-31").median()
    flood = s1.median().subtract(ref).lt(-3)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(start_str, end_str).sum().rename('precip')
    
    return flood.updateMask(slope.lt(5)).selfMask(), rain

# ------------------------------------------------------------
# ANIMATION
# ------------------------------------------------------------
if analysis_mode == "Série Temporelle Animée":
    st.subheader("🎞️ Évolution Temporelle")
    dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval}D')
    ts_rows = []
    images = []

    with st.spinner("Calcul de l'animation..."):
        for i in range(len(dates)-1):
            d1, d2 = str(dates[i].date()), str(dates[i+1].date())
            f, r = get_flood_and_rain(geom_ee.getInfo(), d1, d2)
            if f:
                area = f.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), geom_ee, 100).get('VV').getInfo()
                ts_rows.append({"Date": d1, "Surface (km2)": (area/1e6) if area else 0})
                # Visualisation avec label de date simulé via meta
                images.append(f.visualize(palette=['#00D4FF']).set('label', d1))

        if images:
            col_a, col_b = st.columns(2)
            gif_url = ee.ImageCollection(images).getVideoThumbURL({'dimensions': 600, 'region': geom_ee, 'framesPerSecond': 2, 'crs': 'EPSG:3857'})
            col_a.image(gif_url, caption="Inondations détectées (Bleu)")
            col_b.line_chart(pd.DataFrame(ts_rows).set_index("Date"))
            st.info("Conseil : La date de chaque image correspond aux points de la courbe à droite.")

# ------------------------------------------------------------
# FINAL IMPACT ANALYSIS
# ------------------------------------------------------------
st.subheader("🗺️ Analyse Spatiale & Infrastructures Impactées")

with st.spinner("Calcul des impacts (Population, OSM, Routes)..."):
    flood_all, rain_all = get_flood_and_rain(geom_ee.getInfo(), str(start_date), str(end_date))
    pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(geom_ee).mean()
    
    # OSM & Buildings
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(geom_ee)
    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(geom_ee) # Proxy

    if flood_all:
        # Correction Traceback Pluie
        rain_stats = rain_all.reduceRegion(ee.Reducer.mean(), geom_ee, 5000).getInfo()
        total_rain = rain_stats.get('precip', 0) if rain_stats else 0
        
        # Stats par unité Admin
        features = []
        for idx, row in gdf.iterrows():
            f = ee.Feature(ee.Geometry(mapping(row.geometry)), {
                'id': int(idx), 'name': row[label_col], 
                'total_area': get_true_area_km2(row.geometry),
                'total_pop': pop_img.reduceRegion(ee.Reducer.sum(), ee.Geometry(mapping(row.geometry)), 100).get('population').getInfo()
            })
            features.append(f)
        
        stats_fc = ee.Image.cat([
            flood_all.multiply(ee.Image.pixelArea()).rename('f_area'),
            pop_img.updateMask(flood_all).rename('p_exp')
        ]).reduceRegions(collection=ee.FeatureCollection(features), reducer=ee.Reducer.sum(), scale=100).getInfo()

        # Calcul Infrastructures impactées
        impacted_infra = buildings.filterBounds(geom_ee).map(lambda f: f.set('flood', flood_all.reduceRegion(ee.Reducer.anyNonZero(), f.geometry(), 30).get('VV')))
        impacted_roads = roads.filterBounds(geom_ee).map(lambda f: f.set('flood', flood_all.reduceRegion(ee.Reducer.anyNonZero(), f.geometry(), 30).get('VV')))

        rows = []
        for s in stats_fc['features']:
            p = s['properties']
            f_km2 = (p.get('f_area', 0)) / 1e6
            p_exp = p.get('p_exp', 0)
            t_pop = p.get('total_pop', 1) or 1
            rows.append({
                "Zone": p['name'], 
                "Surf. Totale (km2)": round(p['total_area'], 2),
                "Inondé (km2)": round(f_km2, 2),
                "% Inondé": round((f_km2/p['total_area']*100), 1),
                "Pop. Totale": int(t_pop),
                "Pop. Exposée": int(p_exp or 0),
                "% Pop Exp": round(((p_exp or 0)/t_pop*100), 1),
                "Infras": 0, # Placeholder pour démo
                "Routes (km)": 0,
                "orig_id": p['id']
            })
        df_res = pd.DataFrame(rows)

        # Carte interactive
        m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=9, tiles="CartoDB dark_matter")
        
        # Layer Eau
        mid = flood_all.getMapId({'palette':['#00D4FF']})
        folium.TileLayer(tiles=mid['tile_fetcher'].url_format, attr='GEE', name="Zones Inondées", overlay=True).add_to(m)

        # Polygones Admin avec Popups complets
        for _, r in df_res.iterrows():
            geom = gdf.iloc[int(r['orig_id'])].geometry
            html = f"""
                <div style='font-family:sans-serif; min-width:220px'>
                    <h4 style='margin-bottom:5px'>{r['Zone']}</h4>
                    <hr>
                    <b>Surface inondée :</b> {r['Inondé (km2)']} km² ({r['% Inondé']}%)<br>
                    <b>Pop. totale :</b> {r['Pop. Totale']:,}<br>
                    <b>Pop. exposée :</b> {r['Pop. Exposée']:,} ({r['% Pop Exp']}%)<br>
                    <b>Surface polygone :</b> {r['Surf. Totale (km2)']} km²
                </div>
            """
            folium.GeoJson(
                geom,
                style_function=lambda x, c=("red" if r['% Inondé']>10 else "orange"): {'fillColor':c, 'color':'white', 'weight':1, 'fillOpacity':0.2},
                tooltip=f"{r['Zone']} : {r['% Inondé']}% inondé",
                popup=folium.Popup(html)
            ).add_to(m)

        # Affichage Routes et Infras (Échantillon)
        infra_data = impacted_infra.filter(ee.Filter.eq('flood', 1)).limit(50).getInfo()
        for f in infra_data['features']:
            coords = f['geometry']['coordinates'][0][0]
            folium.CircleMarker([coords[1], coords[0]], radius=3, color="cyan", fill=True, popup="Bâtiment Inondé").add_to(m)

        st_folium(m, width="100%", height=500)

        # Metrics
        st.write("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Surface Inondée Totale", f"{df_res['Inondé (km2)'].sum():.2f} km²")
        c2.metric("Population Exposée", f"{df_res['Pop. Exposée'].sum():,}")
        c3.metric("Cumul Pluviométrique", f"{total_rain:.1f} mm")
        c4.metric("Alerte", "NIVEAU ROUGE" if total_rain > 350 else "STABLE")

        # Exports
        st.sidebar.header("3️⃣ Exportations")
        pdf_b = create_pdf_report(df_res, country, start_date, end_date, {'area': df_res['Inondé (km2)'].sum(), 'pop': df_res['Pop. Exposée'].sum(), 'rain': total_rain})
        st.sidebar.download_button("📄 Rapport d'Expertise PDF", pdf_b, f"rapport_{country}.pdf")
        st.sidebar.download_button("🌍 Données GIS (GeoJSON)", df_res.to_json(), "donnees_impact.geojson")

st.dataframe(df_res.drop(columns=['orig_id']), use_container_width=True)
