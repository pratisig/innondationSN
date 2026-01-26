# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP
# West Africa – Sentinel / CHIRPS / WorldPop / OSM / WFP
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
st.caption("Sentinel-1 | CHIRPS | WorldPop | OSM | WFP/VAM (Admin 1-4)")

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
    cols = ["Zone", "Surf. (km2)", "% Inon", "Pop. Exp", "Infras", "Pluie(mm)"]
    for col in cols: pdf.cell(31, 8, col, border=1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    for _, row in df.iterrows():
        pdf.cell(31, 8, str(row['Zone'])[:18], border=1)
        pdf.cell(31, 8, f"{row['Inondé (km2)']:.2f}", border=1)
        pdf.cell(31, 8, f"{row['% Inondé']:.1f}%", border=1)
        pdf.cell(31, 8, f"{row['Pop. Exposée']:,}", border=1)
        pdf.cell(31, 8, f"{row.get('Infras', 0)}", border=1)
        pdf.cell(31, 8, f"{row.get('Pluie(mm)', 0):.1f}", border=1, ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

# ------------------------------------------------------------
# DATASETS - WFP VAM (Admin 0, 1, 2, 3)
# ------------------------------------------------------------
# On utilise WFP qui est plus à jour et descend souvent plus bas que GAUL
WFP = ee.FeatureCollection("projects/wfp-vam-stats/admin_boundaries/global_admin_boundaries")

# ------------------------------------------------------------
# SIDEBAR - CASCADE ADMINISTRATIVE (Admin 0 -> 4)
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection Administrative")
country_name = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

# Admin 1
a1_fc = WFP.filter(ee.Filter.eq('adm0_name', country_name))
a1_list = a1_fc.aggregate_array('adm1_name').distinct().sort().getInfo()
sel_a1 = st.sidebar.multiselect("Régions (Admin 1)", a1_list)

final_aoi_fc = None
label_col = 'adm1_name'

if sel_a1:
    # Admin 2
    a2_fc = WFP.filter(ee.Filter.inList('adm1_name', sel_a1))
    a2_list = a2_fc.aggregate_array('adm2_name').distinct().sort().getInfo()
    sel_a2 = st.sidebar.multiselect("Départements (Admin 2)", a2_list)
    
    if sel_a2:
        # Admin 3 (Communes/Arrondissements)
        a3_fc = WFP.filter(ee.Filter.inList('adm2_name', sel_a2))
        a3_list = a3_fc.aggregate_array('adm3_name').distinct().sort().getInfo()
        
        # Nettoyage si Admin 3 est vide (selon pays)
        a3_list = [x for x in a3_list if x]
        
        if a3_list:
            sel_a3 = st.sidebar.multiselect("Communes (Admin 3)", a3_list)
            if sel_a3:
                final_aoi_fc = a3_fc.filter(ee.Filter.inList('adm3_name', sel_a3))
                label_col = 'adm3_name'
            else:
                final_aoi_fc = a3_fc
                label_col = 'adm2_name'
        else:
            final_aoi_fc = a2_fc
            label_col = 'adm2_name'
    else:
        final_aoi_fc = a1_fc.filter(ee.Filter.inList('adm1_name', sel_a1))
        label_col = 'adm1_name'
else:
    st.info("Veuillez sélectionner au moins une région.")
    st.stop()

with st.spinner("Chargement de la zone d'étude..."):
    # On limite le nombre de features pour éviter les erreurs getInfo()
    aoi_info = final_aoi_fc.getInfo()
    if not aoi_info['features']:
        st.error("Aucune géométrie trouvée pour cette sélection.")
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
    
    if s1.size().getInfo() < 1: return None, None
    
    # Référence saison sèche
    ref = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate("2024-01-01", "2024-03-31").median()
    flood = s1.median().subtract(ref).lt(-3)
    
    # Masquage pente (zones non inondables)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    
    # Pluie (CHIRPS)
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(start_str, end_str).sum().rename('precip')
    
    return flood.updateMask(slope.lt(5)).selfMask(), rain

# ------------------------------------------------------------
# ANIMATION
# ------------------------------------------------------------
if analysis_mode == "Série Temporelle Animée":
    st.subheader("🎞️ Évolution Temporelle des Inondations")
    dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval}D')
    ts_rows = []
    images = []

    with st.spinner("Calcul de la série temporelle..."):
        for i in range(len(dates)-1):
            d1, d2 = str(dates[i].date()), str(dates[i+1].date())
            f, r = get_flood_and_rain(geom_ee.getInfo(), d1, d2)
            if f:
                area_val = f.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), geom_ee, 250).get('VV').getInfo()
                ts_rows.append({"Date": d1, "Surface (km2)": (area_val/1e6) if area_val else 0})
                
                # Ajout de label visuel (Approximation par le nom du layer)
                images.append(f.visualize(palette=['#00D4FF']))

        if images:
            col_a, col_b = st.columns([2, 1])
            gif_url = ee.ImageCollection(images).getVideoThumbURL({
                'dimensions': 600, 'region': geom_ee, 'framesPerSecond': 2, 'crs': 'EPSG:3857'
            })
            col_a.image(gif_url, use_container_width=True, caption="Animation de l'eau détectée (Bleu)")
            col_b.write("**Dates analysées :**")
            col_b.line_chart(pd.DataFrame(ts_rows).set_index("Date"))
            st.info(f"Visualisation de {len(images)} périodes d'observation.")

# ------------------------------------------------------------
# IMPACT ANALYSIS
# ------------------------------------------------------------
st.subheader("🗺️ Carte d'Impact & Statistiques")

with st.spinner("Analyse des impacts par polygone..."):
    flood_all, rain_all = get_flood_and_rain(geom_ee.getInfo(), str(start_date), str(end_date))
    pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(geom_ee).mean()
    
    # Infrastructures (Google Buildings as proxy for high resolution)
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(geom_ee)
    # Routes (GRIP v4)
    roads = ee.FeatureCollection("projects/sat-io/open-datasets/grip4/Africa").filterBounds(geom_ee)

    if flood_all:
        # Correction Traceback: calcul de pluie sécurisé
        rain_stats = rain_all.reduceRegion(ee.Reducer.mean(), geom_ee, 2000).getInfo()
        total_rain = rain_stats.get('precip', 0) if rain_stats else 0
        
        # Stats par unité Admin (Correction: On réduit le scale pour éviter les Timeouts)
        features_list = []
        for idx, row in gdf.iterrows():
            f_geom = ee.Geometry(mapping(row.geometry))
            # Calcul localisé
            loc_stats = ee.Image.cat([
                flood_all.multiply(ee.Image.pixelArea()).rename('f_area'),
                pop_img.updateMask(flood_all).rename('p_exp'),
                rain_all.rename('rain')
            ]).reduceRegion(ee.Reducer.sum(), f_geom, 250).getInfo()
            
            # Calcul Pop Totale
            t_pop = pop_img.reduceRegion(ee.Reducer.sum(), f_geom, 250).get('population').getInfo() or 0
            
            f_km2 = (loc_stats.get('f_area', 0) or 0) / 1e6
            p_exp = (loc_stats.get('p_exp', 0) or 0)
            
            features_list.append({
                "Zone": row[label_col],
                "Surf. Poly (km2)": round(get_true_area_km2(row.geometry), 2),
                "Inondé (km2)": round(f_km2, 2),
                "% Inondé": round((f_km2 / get_true_area_km2(row.geometry) * 100), 1) if f_km2 > 0 else 0,
                "Pop. Totale": int(t_pop),
                "Pop. Exposée": int(p_exp),
                "Pluie(mm)": round(total_rain, 1), # On utilise la moyenne pour le moment
                "orig_id": idx
            })
            
        df_res = pd.DataFrame(features_list)

        # Affichage Carte
        m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=9, tiles="CartoDB dark_matter")
        
        # Layer Eau
        mid = flood_all.getMapId({'palette':['#00D4FF']})
        folium.TileLayer(tiles=mid['tile_fetcher'].url_format, attr='GEE', name="Zones Inondées", overlay=True).add_to(m)

        # Polygones avec Popups Enrichis
        for _, r in df_res.iterrows():
            geom = gdf.iloc[int(r['orig_id'])].geometry
            pop_html = f"""
                <div style='font-family:sans-serif; min-width:200px'>
                    <h4 style='color:#00D4FF'>{r['Zone']}</h4>
                    <hr>
                    <b>Superficie Polygone:</b> {r['Surf. Poly (km2)']} km²<br>
                    <b>Superficie Inondée:</b> {r['Inondé (km2)']} km² ({r['% Inondé']}%)<br>
                    <b>Population Totale:</b> {r['Pop. Totale']:,}<br>
                    <b>Population Exposée:</b> <span style='color:red'>{r['Pop. Exposée']:,}</span><br>
                    <b>Impact Pop:</b> { (r['Pop. Exposée']/r['Pop. Totale']*100 if r['Pop. Totale']>0 else 0):.1f}%
                </div>
            """
            folium.GeoJson(
                geom,
                style_function=lambda x, c=("red" if r['% Inondé']>5 else "orange"): {
                    'fillColor': c, 'color': 'white', 'weight': 1, 'fillOpacity': 0.2
                },
                tooltip=f"{r['Zone']}",
                popup=folium.Popup(pop_html)
            ).add_to(m)

        # Affichage des infrastructures et routes impactées
        impacted_b = buildings.filterBounds(geom_ee).limit(100).getInfo()
        for b in impacted_b['features']:
            c = b['geometry']['coordinates'][0][0]
            folium.CircleMarker([c[1], c[0]], radius=2, color="#FFD700", popup="Bâtiment / Infrastructure").add_to(m)

        st_folium(m, width="100%", height=500)

        # Tableau et Indicateurs
        st.write("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Surface Inondée Totale", f"{df_res['Inondé (km2)'].sum():.2f} km²")
        c2.metric("Population Exposée", f"{df_res['Pop. Exposée'].sum():,}")
        c3.metric("Pluviométrie Cumulée", f"{total_rain:.1f} mm")

        # Export PDF
        st.sidebar.header("3️⃣ Export")
        pdf_b = create_pdf_report(df_res, country_name, start_date, end_date, {
            'area': df_res['Inondé (km2)'].sum(), 
            'pop': df_res['Pop. Exposée'].sum(),
            'rain': total_rain
        })
        st.sidebar.download_button("📄 Télécharger Rapport PDF", pdf_b, "rapport_impact.pdf")

st.dataframe(df_res.drop(columns=['orig_id']), use_container_width=True)
