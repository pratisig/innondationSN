# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP - ENHANCED
# West Africa – Sentinel-1 / CHIRPS / WorldPop / OSM / FAO GAUL
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import pandas as pd
import osmnx as ox
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from pyproj import Geod
import datetime
from fpdf import FPDF
import base64
import requests
import tempfile


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Analyse d'Impact Inondations – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Analyse d'Impact Inondations & Planification d'Urgence")
st.caption("Sentinel-1 | CHIRPS | WorldPop | OSMnx | FAO GAUL (Admin 1-4)")


# ============================================================
# INIT GEE
# ============================================================
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


# ============================================================
# UTILS & EXPORTS
# ============================================================
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
    pdf.cell(190, 8, f"- Batiments Touches: {stats['buildings']}", ln=True)
    pdf.cell(190, 8, f"- Routes Affectees: {stats['roads']} km", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "2. Detail par Zone Administrative", ln=True)
    pdf.set_font("Arial", "B", 7)
    cols = ["Zone", "Surf.(km2)", "Pop.Exp", "Bat.Touch", "Routes(km)"]
    for col in cols: 
        pdf.cell(38, 8, col, border=1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 7)
    for _, row in df.iterrows():
        pdf.cell(38, 8, str(row['Zone'])[:22], border=1)
        pdf.cell(38, 8, f"{row['Inondé (km2)']:.2f}", border=1)
        pdf.cell(38, 8, f"{row['Pop. Exposée']:,}", border=1)
        pdf.cell(38, 8, f"{row['Bâtiments']}", border=1)
        pdf.cell(38, 8, f"{row['Segments Route']}", border=1, ln=True)
        
    return pdf.output(dest='S').encode('latin-1')


def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6


def ee_polygon_from_gdf(gdf_obj):
    geom = gdf_obj.geometry.unary_union.__geo_interface__
    return ee.Geometry(geom)


# ============================================================
# DATASETS
# ============================================================
GAUL_A0 = ee.FeatureCollection("FAO/GAUL/2015/level0")
GAUL_A1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
GAUL_A2 = ee.FeatureCollection("FAO/GAUL/2015/level2")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def safe_get_info(ee_obj):
    try:
        return ee_obj.getInfo()
    except:
        return None


def get_admin_level_name_col(level):
    """Retourne le nom de colonne pour un niveau administratif."""
    levels = {0: 'ADM0_NAME', 1: 'ADM1_NAME', 2: 'ADM2_NAME', 3: 'ADM3_NAME', 4: 'ADM4_NAME'}
    return levels.get(level, 'ADM1_NAME')


# ============================================================
# SIDEBAR - CASCADE ADMINISTRATIVE HIÉRARCHIQUE (ADM 0-4)
# ============================================================
st.sidebar.header("1️⃣ Sélection Administrative")
country_name = st.sidebar.selectbox(
    "Pays", 
    ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea", "Guinea-Bissau", "Burkina Faso", "Niger", "Nigeria"]
)

# ─────────────────────────────────────────────────────────
# ADMIN 1 (Régions)
# ─────────────────────────────────────────────────────────
a1_fc = GAUL_A1.filter(ee.Filter.eq('ADM0_NAME', country_name))
a1_list = safe_get_info(a1_fc.aggregate_array('ADM1_NAME').distinct().sort())
sel_a1 = st.sidebar.multiselect("Régions (Admin 1)", a1_list if a1_list else [])

if not sel_a1:
    st.info("Veuillez sélectionner au moins une région (Admin 1).")
    st.stop()

# ─────────────────────────────────────────────────────────
# ADMIN 2 (Zones)
# ─────────────────────────────────────────────────────────
a2_fc = GAUL_A2.filter(ee.Filter.eq('ADM0_NAME', country_name)).filter(ee.Filter.inList('ADM1_NAME', sel_a1))
a2_list = safe_get_info(a2_fc.aggregate_array('ADM2_NAME').distinct().sort())
sel_a2 = st.sidebar.multiselect("Zones (Admin 2)", a2_list if a2_list else [])

# Déterminer le niveau d'agrégation actuel
current_level = 1
current_col = 'ADM1_NAME'
filter_fc = a1_fc.filter(ee.Filter.inList('ADM1_NAME', sel_a1))

if sel_a2:
    filter_fc = a2_fc.filter(ee.Filter.inList('ADM2_NAME', sel_a2))
    current_level = 2
    current_col = 'ADM2_NAME'

# ─────────────────────────────────────────────────────────
# ADMIN 3 (Districts) - Optionnel
# ─────────────────────────────────────────────────────────
try:
    GAUL_A3 = ee.FeatureCollection("FAO/GAUL/2015/level3")
    a3_fc = GAUL_A3.filter(ee.Filter.eq('ADM0_NAME', country_name))
    
    if sel_a2:
        a3_fc = a3_fc.filter(ee.Filter.inList('ADM2_NAME', sel_a2))
    elif sel_a1:
        a3_fc = a3_fc.filter(ee.Filter.inList('ADM1_NAME', sel_a1))
    
    a3_list = safe_get_info(a3_fc.aggregate_array('ADM3_NAME').distinct().sort())
    
    if a3_list and len(a3_list) > 0:
        sel_a3 = st.sidebar.multiselect("Districts (Admin 3)", a3_list if a3_list else [])
        
        if sel_a3:
            filter_fc = a3_fc.filter(ee.Filter.inList('ADM3_NAME', sel_a3))
            current_level = 3
            current_col = 'ADM3_NAME'
except:
    sel_a3 = []

# ─────────────────────────────────────────────────────────
# ADMIN 4 (Sous-districts) - Ultra optionnel
# ─────────────────────────────────────────────────────────
try:
    GAUL_A4 = ee.FeatureCollection("FAO/GAUL/2015/level4")
    a4_fc = GAUL_A4.filter(ee.Filter.eq('ADM0_NAME', country_name))
    
    if sel_a3:
        a4_fc = a4_fc.filter(ee.Filter.inList('ADM3_NAME', sel_a3))
    elif sel_a2:
        a4_fc = a4_fc.filter(ee.Filter.inList('ADM2_NAME', sel_a2))
    elif sel_a1:
        a4_fc = a4_fc.filter(ee.Filter.inList('ADM1_NAME', sel_a1))
    
    a4_list = safe_get_info(a4_fc.aggregate_array('ADM4_NAME').distinct().sort())
    
    if a4_list and len(a4_list) > 0:
        sel_a4 = st.sidebar.multiselect("Sous-districts (Admin 4)", a4_list if a4_list else [])
        
        if sel_a4:
            filter_fc = a4_fc.filter(ee.Filter.inList('ADM4_NAME', sel_a4))
            current_level = 4
            current_col = 'ADM4_NAME'
except:
    sel_a4 = []


# ─────────────────────────────────────────────────────────
# Charger la géométrie finale
# ─────────────────────────────────────────────────────────
with st.spinner("Chargement de la zone d'étude..."):
    aoi_info = safe_get_info(filter_fc)
    if not aoi_info or not aoi_info['features']:
        st.error("Aucune géométrie trouvée.")
        st.stop()
    gdf = gpd.GeoDataFrame.from_features(aoi_info, crs="EPSG:4326")
    merged_poly = unary_union(gdf.geometry)
    geom_ee = ee_polygon_from_gdf(gdf)


# ============================================================
# TEMPORAL CONFIG
# ============================================================
st.sidebar.header("2️⃣ Analyse Temporelle")

# Date référence (avant inondation)
st.sidebar.subheader("📅 Période de Référence (Sèche)")
ref_start = st.sidebar.date_input("Début référence", pd.to_datetime("2024-01-01"))
ref_end = st.sidebar.date_input("Fin référence", pd.to_datetime("2024-03-31"))

# Date crise (inondation)
st.sidebar.subheader("🌊 Période Crise (Inondation)")
start_date = st.sidebar.date_input("Début crise", pd.to_datetime("2024-07-01"))
end_date = st.sidebar.date_input("Fin crise", pd.to_datetime("2024-10-31"))

analysis_mode = st.sidebar.radio("Mode", ["Synthèse Globale", "Série Temporelle"])
interval = 15 if st.sidebar.checkbox("Quinzaines", value=True) else 30


# ============================================================
# CORE ENGINES - FLOOD DETECTION
# ============================================================
@st.cache_data
def get_flood_detection(aoi_json, ref_start_str, ref_end_str, flood_start_str, flood_end_str):
    """
    Détection inondation Sentinel-1 VV avec référence.
    Compare backscatter période sèche vs crise.
    """
    aoi = ee.Geometry(aoi_json)
    
    # Collection Sentinel-1
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.eq("resolution_meters", 10))
          .select("VV"))
    
    # Image de référence (sèche)
    ref_img = s1.filterDate(ref_start_str, ref_end_str).median()
    
    # Image crise
    crisis_img = s1.filterDate(flood_start_str, flood_end_str).median()
    
    # Conversion en dB
    def to_db(img):
        return ee.Image(10).multiply(img.max(ee.Image(-30)).log10())
    
    ref_db = to_db(ref_img)
    crisis_db = to_db(crisis_img)
    
    # Différence de backscatter (eau = réduction du signal)
    diff = ref_db.subtract(crisis_db)
    
    # Seuil inondation
    flooded_raw = diff.gt(1.25)
    
    # Masque pente
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Algorithms.Terrain(dem).select("slope")
    mask_slope = slope.lt(5)
    
    # Masque eau permanente
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    permanent_water = gsw.select("occurrence").gte(90)
    mask_perm = permanent_water.Not()
    
    # Application des masques
    flooded = (flooded_raw
               .updateMask(mask_slope)
               .updateMask(mask_perm)
               .selfMask())
    
    # Filtre connectivité
    flooded = flooded.updateMask(flooded.connectedPixelCount(8).gte(5))
    
    # Précipitations CHIRPS
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(aoi)
            .filterDate(flood_start_str, flood_end_str)
            .sum()
            .rename('precip'))
    
    return flooded, rain, s1.size()


@st.cache_data
def get_rainfall_data(aoi_json, start_str, end_str):
    """Récupère données de précipitations CHIRPS."""
    aoi = ee.Geometry(aoi_json)
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(aoi)
            .filterDate(start_str, end_str)
            .sum()
            .rename('precip'))
    return rain


# ============================================================
# INFRASTRUCTURE IMPACT (OSMNX)
# ============================================================
def analyze_infrastructure_impact_osmnx(admin_polygon):
    """Analyse impacts infrastructures via OSMnx."""
    tags = {
        "building": True,
        "highway": True,
        "amenity": ["hospital", "school", "clinic", "university", "health_centre"]
    }
    try:
        osm = ox.geometries_from_polygon(admin_polygon, tags)
        if osm.empty:
            return dict(buildings=0, roads_km=0, health=0, education=0)
        
        buildings = osm[osm.get("building", pd.Series()).notna()].shape[0] if "building" in osm.columns else 0
        
        # Routes
        roads = osm[osm.get("highway", pd.Series()).notna()] if "highway" in osm.columns else None
        roads_km = 0
        if roads is not None and not roads.empty:
            road_lines = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
            roads_km = round(road_lines.geometry.length.sum() / 1000, 2) if not road_lines.empty else 0
        
        # Santé & Éducation
        health_list = ["hospital", "clinic", "health_centre"]
        edu_list = ["school", "college", "university"]
        
        health = 0
        education = 0
        if "amenity" in osm.columns:
            health = osm[osm["amenity"].isin(health_list)].shape[0]
            education = osm[osm["amenity"].isin(edu_list)].shape[0]
        
        return {
            "buildings": buildings,
            "roads_km": roads_km,
            "health": health,
            "education": education
        }
    except Exception as e:
        st.warning(f"⚠️ Erreur OSMnx : {str(e)[:50]}")
        return dict(buildings=0, roads_km=0, health=0, education=0)


# ============================================================
# MAIN ANALYSIS & VISUALIZATION
# ============================================================
st.subheader("🗺️ Analyse d'Impact Spatiale")

with st.spinner("Analyse GEE & OSMnx en cours..."):
    # Obtenir mask d'inondation
    flood_all, rain_all, s1_count = get_flood_detection(
        geom_ee.getInfo(),
        str(ref_start),
        str(ref_end),
        str(start_date),
        str(end_date)
    )
    
    if s1_count < 1:
        st.error("❌ Pas de données Sentinel-1 pour cette période/région.")
        st.stop()
    
    # Population WorldPop
    pop_img = (ee.ImageCollection("WorldPop/GP/100m/pop")
               .filterBounds(geom_ee)
               .filterDate("2020-01-01", "2020-12-31")
               .mean()
               .select(0))
    
    # Calculs par zone
    rain_stats = safe_get_info(rain_all.reduceRegion(ee.Reducer.mean(), geom_ee, 2000))
    total_rain = rain_stats.get('precip', 0) if rain_stats else 0
    
    features_list = []
    
    for idx, row in gdf.iterrows():
        f_geom = ee.Geometry(mapping(row.geometry))
        
        # Stats GEE
        try:
            loc_stats = safe_get_info(ee.Image.cat([
                flood_all.multiply(ee.Image.pixelArea()).rename('f_area'),
                pop_img.updateMask(flood_all.select(0)).rename('p_exp')
            ]).reduceRegion(ee.Reducer.sum(), f_geom, 250))
            
            f_km2 = (loc_stats.get('f_area', 0) if loc_stats else 0) / 1e6
            p_exp = int(loc_stats.get('p_exp', 0) if loc_stats else 0)
        except:
            f_km2 = 0
            p_exp = 0
        
        # Stats OSMnx
        osm_data = analyze_infrastructure_impact_osmnx(row.geometry)
        
        # Calcul pourcentage
        zone_area = get_true_area_km2(row.geometry)
        pct_flooded = (f_km2 / zone_area * 100) if zone_area > 0 else 0
        
        features_list.append({
            "Zone": row[current_col],
            "Inondé (km2)": round(f_km2, 2),
            "% Inondé": round(pct_flooded, 1),
            "Pop. Exposée": p_exp,
            "Bâtiments": osm_data["buildings"],
            "Santé": osm_data["health"],
            "Éducation": osm_data["education"],
            "Segments Route": osm_data["roads_km"],
            "orig_id": idx
        })
    
    df_res = pd.DataFrame(features_list)
    
    # Fonction safe_sum
    def safe_sum(col):
        return df_res[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).sum()


# ─────────────────────────────────────────────────────────
# CARTE INTERACTIVE
# ─────────────────────────────────────────────────────────
try:
    m = folium.Map(
        location=[merged_poly.centroid.y, merged_poly.centroid.x],
        zoom_start=9,
        tiles="CartoDB positron"
    )
    
    # Overlay flood map
    try:
        # Corriger l'image inondation pour visualisation
        flooded_binary = flood_all.select(0).unmask(0)
        flooded_vis = flooded_binary.visualize(min=0, max=1, palette=["white", "blue"])
        map_id = flooded_vis.getMapId({"min": 0, "max": 1, "palette": ["white", "#0066FF"]})
        
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='GEE Flood',
            name='Zones Inondées',
            overlay=True,
            opacity=0.6
        ).add_to(m)
    except Exception as e:
        st.warning(f"⚠️ Overlay flood : {str(e)[:50]}")
    
    # GeoJSON zones admin
    for _, row in df_res.iterrows():
        geom = gdf.iloc[int(row['orig_id'])].geometry
        pop_text = f"<b>{row['Zone']}</b><br>"
        pop_text += f"Inondé: {row['Inondé (km2)']} km² ({row['% Inondé']}%)<br>"
        pop_text += f"Pop Exp: {row['Pop. Exposée']:,}<br>"
        pop_text += f"Bâtiments: {row['Bâtiments']}"
        
        folium.GeoJson(
            geom,
            style_function=lambda x: {
                'fillColor': 'orange',
                'color': 'red',
                'weight': 2,
                'fillOpacity': 0.2
            },
            popup=folium.Popup(pop_text, max_width=250)
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=600)

except Exception as e:
    st.error(f"Erreur cartographie : {e}")


# ─────────────────────────────────────────────────────────
# DASHBOARD MÉTRIQUES
# ─────────────────────────────────────────────────────────
st.write("---")
st.markdown("### 📊 Tableau de Bord Synthétique")

c1, c2, c3, c4, c5 = st.columns(5)

total_flooded = df_res["Inondé (km2)"].sum()
total_pop_exp = df_res["Pop. Exposée"].sum()
total_buildings = int(safe_sum('Bâtiments'))
total_health = int(safe_sum('Santé'))
total_edu = int(safe_sum('Éducation'))
total_roads = round(safe_sum('Segments Route'), 1)

c1.metric("🌊 Surface Inondée", f"{total_flooded:.2f} km²")
c2.metric("👥 Pop. Exposée", f"{total_pop_exp:,}")
c3.metric("🏠 Bâtiments", total_buildings)
c4.metric("🏥 Santé / 🎓 Édu", f"{total_health} / {total_edu}")
c5.metric("🛣️ Routes", f"{total_roads} km")

st.write("---")
st.markdown("### 📋 Détails par Zone")
st.dataframe(df_res.drop(columns=['orig_id']), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────
# EXPORT PDF
# ─────────────────────────────────────────────────────────
st.sidebar.header("3️⃣ Export")
pdf_b = create_pdf_report(df_res, country_name, start_date, end_date, {
    'area': total_flooded,
    'pop': total_pop_exp,
    'buildings': total_buildings,
    'roads': total_roads,
    'rain': total_rain
})

st.sidebar.download_button(
    "📄 Télécharger Rapport PDF",
    pdf_b,
    "rapport_impact_inondation.pdf",
    "application/pdf"
)

# Méta-infos
st.sidebar.write("---")
st.sidebar.markdown(f"""
**📍 Zone d'étude**: {country_name}  
**📊 Niveau**: Admin {current_level}  
**📅 Période crise**: {start_date} → {end_date}  
**📊 Images S1**: {s1_count}  
""")
