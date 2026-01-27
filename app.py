# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP - FIXED S1
# West Africa – Sentinel-1 (Assouplissements) / CHIRPS / WorldPop / OSM
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import pandas as pd
import osmnx as ox
from shapely.geometry import mapping, shape, Point, box
from shapely.ops import unary_union
from pyproj import Geod
import datetime
from fpdf import FPDF
import base64
import requests
import tempfile
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Analyse d'Impact Inondations – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Analyse d'Impact Inondations & Planification d'Urgence")
st.caption("Sentinel-1 (VV+VH) | CHIRPS | WorldPop | OSMnx")


# ============================================================
# INIT GEE
# ============================================================
@st.cache_resource
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("❌ Secret 'GEE_SERVICE_ACCOUNT' manquant dans Streamlit.")
        st.stop()
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation GEE : {e}")
        return False


init_gee()


# ============================================================
# PREDEFINED REGIONS (Sénégal exemple)
# ============================================================
REGIONS_DATA = {
    "Senegal": {
        "Saint-Louis": {"lat": 16.0193, "lon": -16.4901, "buffer_km": 50},
        "Louga": {"lat": 15.6167, "lon": -14.9167, "buffer_km": 50},
        "Thiès": {"lat": 14.7911, "lon": -16.3656, "buffer_km": 40},
        "Dakar": {"lat": 14.6928, "lon": -17.0469, "buffer_km": 30},
        "Kaolack": {"lat": 13.9717, "lon": -15.9371, "buffer_km": 45},
        "Tambacounda": {"lat": 13.7727, "lon": -13.7691, "buffer_km": 60},
        "Matam": {"lat": 15.6508, "lon": -13.3526, "buffer_km": 50},
        "Kolda": {"lat": 13.1608, "lon": -14.9428, "buffer_km": 45},
        "Sédhiou": {"lat": 13.6638, "lon": -15.1483, "buffer_km": 40},
        "Ziguinchor": {"lat": 13.3673, "lon": -15.5694, "buffer_km": 40},
    },
    "Mali": {
        "Kayes": {"lat": 13.9476, "lon": -11.4406, "buffer_km": 60},
        "Koulikoro": {"lat": 12.6520, "lon": -8.0029, "buffer_km": 60},
        "Bamako": {"lat": 12.6500, "lon": -8.0029, "buffer_km": 40},
        "Ségou": {"lat": 13.4549, "lon": -6.2655, "buffer_km": 60},
        "Mopti": {"lat": 14.2743, "lon": -4.1843, "buffer_km": 70},
        "Tombouctou": {"lat": 16.7769, "lon": -3.0064, "buffer_km": 80},
    },
    "Mauritania": {
        "Nouakchott": {"lat": 18.0735, "lon": -15.9582, "buffer_km": 40},
        "Rosso": {"lat": 16.5167, "lon": -14.7833, "buffer_km": 50},
        "Kaédi": {"lat": 16.9631, "lon": -13.9506, "buffer_km": 50},
    },
    "Gambia": {
        "Banjul": {"lat": 13.4549, "lon": -16.5790, "buffer_km": 30},
        "Serekunda": {"lat": 13.4516, "lon": -16.7146, "buffer_km": 35},
    },
    "Guinea": {
        "Conakry": {"lat": 9.5412, "lon": -13.6578, "buffer_km": 40},
        "Kindia": {"lat": 9.4667, "lon": -10.0000, "buffer_km": 45},
    },
}


# ============================================================
# UTILS & EXPORTS
# ============================================================
def safe_get_info(ee_obj):
    """Évalue un objet EE en toute sécurité."""
    try:
        return ee_obj.getInfo()
    except Exception as e:
        return None


def create_pdf_report(df, country, p1, p2, stats):
    """Génère rapport PDF complet."""
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
    pdf.cell(190, 8, f"- Precipitations Moyennes: {stats['rain']:.1f} mm", ln=True)
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
        pdf.cell(38, 8, str(row.get('Zone', 'N/A'))[:22], border=1)
        pdf.cell(38, 8, f"{row.get('Inondé (km2)', 0):.2f}", border=1)
        pdf.cell(38, 8, f"{row.get('Pop. Exposée', 0):,}", border=1)
        pdf.cell(38, 8, f"{row.get('Bâtiments', 0)}", border=1)
        pdf.cell(38, 8, f"{row.get('Segments Route', 0)}", border=1, ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "3. Sources & Méthodologie", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(190, 5, 
        "Données: Sentinel-1 (ESA), CHIRPS (UCSB), WorldPop (Univ. Southampton), "
        "OSMnx (OpenStreetMap). "
        "Détection: Comparaison backscatter VV (dB), seuil différence > 1.25 dB, "
        "masques pente < 5°, excl. eau permanente, filtre connectivité."
    )
    
    return pdf.output(dest='S').encode('latin-1')


def get_true_area_km2(geom_shapely):
    """Calcule surface vraie en km² avec géodésie."""
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6


def ee_polygon_from_gdf(gdf_obj):
    """Convertit GeoDataFrame en géométrie EE."""
    geom = gdf_obj.geometry.unary_union.__geo_interface__
    return ee.Geometry(geom)


def safe_sum(df, col):
    """Somme sûre pour colonne avec possibles non-numériques."""
    return df[col].apply(lambda x: x if isinstance(x, (int, float)) else 0).sum()


def create_circular_region(lat, lon, buffer_km):
    """Crée une région circulaire autour d'un point."""
    point = Point(lon, lat)
    buffer_degrees = buffer_km / 111.0
    circle = point.buffer(buffer_degrees)
    return circle


# ============================================================
# SIDEBAR - SELECTION REGIONS
# ============================================================
st.sidebar.header("1️⃣ Sélection Administrative")

country_name = st.sidebar.selectbox(
    "Pays", 
    list(REGIONS_DATA.keys())
)

if country_name not in REGIONS_DATA:
    st.error("❌ Pays non disponible.")
    st.stop()

available_regions = list(REGIONS_DATA[country_name].keys())
sel_regions = st.sidebar.multiselect(
    "Régions", 
    available_regions,
    default=[available_regions[0]] if available_regions else []
)

if not sel_regions:
    st.info("ℹ️ Veuillez sélectionner au moins une région.")
    st.stop()

# Créer GeoDataFrame à partir des régions sélectionnées
geometries = []
region_names = []

for region_name in sel_regions:
    region_info = REGIONS_DATA[country_name][region_name]
    circle = create_circular_region(
        region_info["lat"],
        region_info["lon"],
        region_info["buffer_km"]
    )
    geometries.append(circle)
    region_names.append(region_name)

if not geometries:
    st.error("❌ Aucune géométrie créée.")
    st.stop()

gdf = gpd.GeoDataFrame(
    {"region": region_names},
    geometry=geometries,
    crs="EPSG:4326"
)

merged_poly = unary_union(gdf.geometry)
geom_ee = ee_polygon_from_gdf(gdf)

st.sidebar.success(f"✅ {len(sel_regions)} région(s) sélectionnée(s)")


# ============================================================
# TEMPORAL CONFIG
# ============================================================
st.sidebar.header("2️⃣ Analyse Temporelle")

st.sidebar.subheader("📅 Période de Référence (Sèche)")
ref_start = st.sidebar.date_input("Début référence", pd.to_datetime("2023-01-01"))
ref_end = st.sidebar.date_input("Fin référence", pd.to_datetime("2023-03-31"))

st.sidebar.subheader("🌊 Période Crise (Inondation)")
start_date = st.sidebar.date_input("Début crise", pd.to_datetime("2023-08-01"))
end_date = st.sidebar.date_input("Fin crise", pd.to_datetime("2023-10-31"))

st.sidebar.info("💡 Utilise 2023 pour avoir plus de données Sentinel-1 disponibles.")


# ============================================================
# CORE ENGINES - FLOOD DETECTION (ASSOUPLISSEMENTS)
# ============================================================
@st.cache_data
def get_flood_detection(aoi_json, ref_start_str, ref_end_str, flood_start_str, flood_end_str):
    """
    Détection inondation Sentinel-1 VV avec référence.
    ASSOUPLISSEMENTS : accepte VV+VH, toutes les passes orbitales.
    """
    aoi = ee.Geometry(aoi_json)
    
    st.write(f"🔍 **Recherche Sentinel-1** pour {flood_start_str} → {flood_end_str}")
    
    # ✅ ASSOUPLISSEMENT 1 : Accepter VV ET VH
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(flood_start_str, flood_end_str)
          .filter(ee.Filter.eq("instrumentMode", "IW")))
    
    # Évaluer le count - SANS filtrer par pass ou polarisation d'abord
    s1_count_all = safe_get_info(s1.size())
    st.write(f"   → Images S1 trouvées (tous types) : {s1_count_all}")
    
    if s1_count_all is None or s1_count_all < 1:
        st.error(f"❌ Aucune donnée Sentinel-1 pour cette période/région.")
        return None, None, 0
    
    # ✅ ASSOUPLISSEMENT 2 : Sélectionner VV si disponible, sinon VH
    s1_vv = s1.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    s1_vv_count = safe_get_info(s1_vv.size())
    
    if s1_vv_count and s1_vv_count > 0:
        st.write(f"   → Images VV trouvées : {s1_vv_count}")
        s1_selected = s1_vv.select("VV")
    else:
        st.write("   ⚠️ Pas de VV, utilisation de VH...")
        s1_selected = s1.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select("VH")
        s1_count_all = safe_get_info(s1_selected.size())
        if not s1_count_all or s1_count_all < 1:
            st.error("❌ Aucune polarisation (VV/VH) disponible.")
            return None, None, 0
    
    # Image de référence (sèche)
    ref_s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi)
              .filterDate(ref_start_str, ref_end_str)
              .filter(ee.Filter.eq("instrumentMode", "IW")))
    
    # Sélectionner même polarisation pour référence
    if s1_vv_count and s1_vv_count > 0:
        ref_img = ref_s1.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).select("VV").median()
    else:
        ref_img = ref_s1.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select("VH").median()
    
    crisis_img = s1_selected.median()
    
    # Vérifier que les images ont des données
    ref_has_data = safe_get_info(ref_img.hasData())
    crisis_has_data = safe_get_info(crisis_img.hasData())
    
    if not ref_has_data or not crisis_has_data:
        st.error(f"❌ Données vides : ref={ref_has_data}, crisis={crisis_has_data}")
        return None, None, s1_count_all
    
    # Conversion en dB
    def to_db(img):
        clamped = img.max(ee.Image(-30))
        return ee.Image(10).multiply(clamped.log10())
    
    ref_db = to_db(ref_img)
    crisis_db = to_db(crisis_img)
    
    # Différence de backscatter
    diff = ref_db.subtract(crisis_db)
    
    # Seuil inondation (eau = backscatter diminué)
    flooded_raw = diff.gt(1.25)
    
    # Masque pente
    try:
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Algorithms.Terrain(dem).select("slope")
        mask_slope = slope.lt(5)
    except:
        mask_slope = ee.Image(1)
    
    # Masque eau permanente
    try:
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        permanent_water = gsw.select("occurrence").gte(90)
        mask_perm = permanent_water.Not()
    except:
        mask_perm = ee.Image(1)
    
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
    
    return flooded, rain, s1_count_all


# ============================================================
# INFRASTRUCTURE IMPACT (OSMNX)
# ============================================================
def analyze_infrastructure_impact_osmnx(admin_polygon):
    """Analyse impacts infrastructures via OSMnx."""
    try:
        # Utiliser features_from_polygon
        buildings_gdf = ox.features_from_polygon(admin_polygon, {"building": True})
        buildings_count = len(buildings_gdf) if not buildings_gdf.empty else 0
        
        # Routes
        roads_gdf = ox.features_from_polygon(admin_polygon, {"highway": True})
        roads_km = 0
        if not roads_gdf.empty:
            road_lines = roads_gdf[roads_gdf.geometry.type.isin(["LineString", "MultiLineString"])]
            roads_km = round(road_lines.geometry.length.sum() / 1000, 2) if not road_lines.empty else 0
        
        # Santé & Éducation
        health_gdf = ox.features_from_polygon(
            admin_polygon,
            {"amenity": ["hospital", "clinic", "health_centre"]}
        )
        health_count = len(health_gdf) if not health_gdf.empty else 0
        
        edu_gdf = ox.features_from_polygon(
            admin_polygon,
            {"amenity": ["school", "college", "university"]}
        )
        edu_count = len(edu_gdf) if not edu_gdf.empty else 0
        
        return {
            "buildings": buildings_count,
            "roads_km": roads_km,
            "health": health_count,
            "education": edu_count
        }
    
    except Exception as e:
        st.warning(f"⚠️ OSMnx ({str(e)[:40]})")
        return dict(buildings=0, roads_km=0, health=0, education=0)


# ============================================================
# MAIN ANALYSIS & VISUALIZATION
# ============================================================
st.subheader("🗺️ Analyse d'Impact Spatiale")

with st.spinner("Analyse GEE & OSMnx en cours..."):
    
    # ✅ Afficher progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("1/3 - Détection inondations (Sentinel-1)...")
    
    # Obtenir mask d'inondation
    flood_all, rain_all, s1_count = get_flood_detection(
        geom_ee.getInfo(),
        str(ref_start),
        str(ref_end),
        str(start_date),
        str(end_date)
    )
    
    progress_bar.progress(33)
    
    if flood_all is None or s1_count < 1:
        st.error(f"❌ Impossible de continuer (S1 count={s1_count}).")
        st.info("💡 **Suggestions:**\n- Élargir la plage temporelle\n- Changer de région\n- Utiliser 2023 au lieu de 2024")
        st.stop()
    
    status_text.text("2/3 - Population & analyse spatiale...")
    
    # Population WorldPop
    pop_img = (ee.ImageCollection("WorldPop/GP/100m/pop")
               .filterBounds(geom_ee)
               .mean()
               .select(0))
    
    # Calculs par zone
    try:
        rain_stats = safe_get_info(rain_all.reduceRegion(ee.Reducer.mean(), geom_ee, 2000))
        total_rain = rain_stats.get('precip', 0) if rain_stats else 0
    except:
        total_rain = 0
    
    features_list = []
    
    for idx, row in gdf.iterrows():
        f_geom = ee.Geometry(mapping(row.geometry))
        
        # Stats GEE
        try:
            flood_area_img = flood_all.multiply(ee.Image.pixelArea()).rename('f_area')
            pop_masked = pop_img.updateMask(flood_all.select(0)).rename('p_exp')
            
            loc_stats = safe_get_info(ee.Image.cat([
                flood_area_img,
                pop_masked
            ]).reduceRegion(ee.Reducer.sum(), f_geom, 250))
            
            f_km2 = (loc_stats.get('f_area', 0) if loc_stats else 0) / 1e6
            p_exp = int(loc_stats.get('p_exp', 0) if loc_stats else 0)
        except Exception as e:
            f_km2 = 0
            p_exp = 0
        
        # Stats OSMnx
        osm_data = analyze_infrastructure_impact_osmnx(row.geometry)
        
        # Calcul pourcentage
        zone_area = get_true_area_km2(row.geometry)
        pct_flooded = (f_km2 / zone_area * 100) if zone_area > 0 else 0
        
        features_list.append({
            "Zone": row['region'],
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
    progress_bar.progress(66)
    status_text.text("3/3 - Cartographie...")


# ─────────────────────────────────────────────────────────
# CARTE INTERACTIVE
# ─────────────────────────────────────────────────────────
try:
    m = folium.Map(
        location=[merged_poly.centroid.y, merged_poly.centroid.x],
        zoom_start=8,
        tiles="CartoDB positron"
    )
    
    # Overlay flood map
    try:
        flood_has_data = safe_get_info(flood_all.hasData())
        
        if flood_has_data:
            flooded_binary = flood_all.select(0).unmask(0)
            flooded_vis = flooded_binary.visualize(min=0, max=1, palette=["white", "#0066FF"])
            map_id = flooded_vis.getMapId()
            
            folium.TileLayer(
                tiles=map_id['tile_fetcher'].url_format,
                attr='GEE Flood',
                name='Zones Inondées (Sentinel-1)',
                overlay=True,
                opacity=0.6
            ).add_to(m)
        else:
            st.warning("⚠️ Pas de pixels inondés détectés dans cette région.")
    except Exception as e:
        st.warning(f"⚠️ Overlay : {str(e)[:40]}")
    
    # GeoJSON zones admin
    for _, row in df_res.iterrows():
        geom = gdf.iloc[int(row['orig_id'])].geometry
        pop_text = f"<b>{row['Zone']}</b><br>"
        pop_text += f"Inondé: {row['Inondé (km2)']} km² ({row['% Inondé']}%)<br>"
        pop_text += f"Pop Exp: {row['Pop. Exposée']:,}<br>"
        pop_text += f"Bâtiments: {row['Bâtiments']}<br>"
        pop_text += f"Santé: {row['Santé']} | Éduc: {row['Éducation']}"
        
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
    st.error(f"❌ Carte : {e}")

progress_bar.progress(100)
status_text.text("✅ Analyse terminée!")


# ─────────────────────────────────────────────────────────
# DASHBOARD MÉTRIQUES
# ─────────────────────────────────────────────────────────
st.write("---")
st.markdown("### 📊 Tableau de Bord Synthétique")

total_flooded = df_res["Inondé (km2)"].sum()
total_pop_exp = df_res["Pop. Exposée"].sum()
total_buildings = int(safe_sum(df_res, 'Bâtiments'))
total_health = int(safe_sum(df_res, 'Santé'))
total_edu = int(safe_sum(df_res, 'Éducation'))
total_roads = round(safe_sum(df_res, 'Segments Route'), 1)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("🌊 Surface Inondée", f"{total_flooded:.2f} km²")
c2.metric("👥 Pop. Exposée", f"{total_pop_exp:,}")
c3.metric("🏠 Bâtiments", total_buildings)
c4.metric("🏥 Santé / 🎓 Édu", f"{total_health} / {total_edu}")
c5.metric("🛣️ Routes", f"{total_roads} km")

st.write("---")
st.markdown("### 📋 Détails par Région")
st.dataframe(df_res.drop(columns=['orig_id']), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────
# EXPORT PDF
# ─────────────────────────────────────────────────────────
st.sidebar.header("3️⃣ Export")

if len(df_res) > 0:
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
### 📍 Métadonnées
- **Pays**: {country_name}  
- **Régions**: {len(sel_regions)}  
- **Période crise**: {start_date} → {end_date}  
- **Images S1**: {s1_count}  
- **Précipitations moyennes**: {total_rain:.1f} mm  

### 🔧 Assouplissements appliqués
- ✅ Accepte polarisations VV et VH
- ✅ Toutes les passes orbitales (ASCENDING/DESCENDING)
- ✅ Mode IW seulement (10m résolution)
""")
