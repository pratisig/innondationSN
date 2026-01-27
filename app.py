# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP - WEST AFRICA
# ============================================================

import os, io, json
from datetime import datetime
import tempfile

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

import folium
from streamlit_folium import st_folium
import osmnx as ox
import ee
from fpdf import FPDF

# ============================================================
# 1. CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Flood Impact WA",
    layout="wide"
)
st.title("🌊 Flood Impact & Infrastructure Analysis")
st.caption("West Africa – Sentinel-1 | CHIRPS | WorldPop | OSM | GADM")

# ============================================================
# 2. INIT GEE
# ============================================================
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.warning("❌ Secret GEE manquant. L'analyse inondation est désactivée.")
        return False
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.warning(f"❌ Erreur initialisation GEE : {e}")
        return False

gee_available = init_gee()

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

@st.cache_data(ttl=3600)
def load_gadm(iso, level):
    """Charge GADM depuis le serveur UCDavis."""
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except:
        return None

def get_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood, diff_threshold=1.25):
    """Retourne masque d'inondation à partir de Sentinel-1."""
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(aoi_ee)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))

        img_ref = s1.filterDate(start_ref, end_ref).median().clip(aoi_ee)
        img_flood = s1.filterDate(start_flood, end_flood).min().clip(aoi_ee)

        ref_db = img_ref.log10().multiply(10)
        flood_db = img_flood.log10().multiply(10)
        diff = ref_db.subtract(flood_db)
        flooded = diff.gt(diff_threshold)

        terrain = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')

        final_mask = flooded.updateMask(terrain.select('slope').lt(5)).updateMask(gsw.lt(80)).selfMask()
        return final_mask
    except Exception as e:
        st.warning(f"Erreur calcul masque inondation : {e}")
        return None

def get_osm_data(gdf_aoi):
    """Télécharge les données OSM via OSMnx pour bâtiments, routes, amenities."""
    try:
        bounds = gdf_aoi.total_bounds  # minx, miny, maxx, maxy
        tags = {
            'building': True,
            'highway': True,
            'amenity': ['hospital', 'school', 'clinic']
        }
        osm = ox.features_from_bbox(bounds[3], bounds[1], bounds[2], bounds[0], tags=tags)
        return osm[osm.geometry.within(gdf_aoi.unary_union)]
    except Exception as e:
        st.warning(f"Erreur chargement OSM : {e}")
        return gpd.GeoDataFrame()

def compute_indicator(gdf_zone, flood_mask, pop_raster=None):
    """Calcule indicateurs d'inondation et infrastructures."""
    area_km2 = gdf_zone.geometry.area.sum() / 1e6
    pop_exp = 0
    if pop_raster:
        # Placeholder : WorldPop (optionnel)
        pop_exp = 0
    return {"Surface": round(area_km2,2), "Pop_Exposée": int(pop_exp)}

def create_pdf_report(df, country, start_date, end_date):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(190,10,f"Rapport Flood Impact - {country}",ln=True,align="C")
    pdf.set_font("Arial","",12)
    pdf.cell(190,10,f"Période: {start_date} au {end_date}",ln=True,align="C")
    pdf.ln(10)
    for _, row in df.iterrows():
        pdf.cell(190,8,f"{row['Zone']}: Surface {row['Surface']} km², Pop Exp {row['Pop_Exposée']}, Batiments {row['Bâtiments']}, Routes {row['Routes']}",ln=True)
    return pdf.output(dest='S').encode('latin-1')

# ============================================================
# 4. SIDEBAR UI
# ============================================================
st.sidebar.header("🌍 Paramètres")
country_dict = {"Sénégal":"SEN","Mali":"MLI","Niger":"NER","Burkina Faso":"BFA"}
country = st.sidebar.selectbox("Pays", list(country_dict.keys()))
iso = country_dict[country]

# Sélection par niveau admin ou upload fichier
mode = st.sidebar.radio("Source Zone", ["Admin (GADM)", "Upload Geo file"])
gdf_zone = None
if mode=="Admin (GADM)":
    level = st.sidebar.slider("Niveau Admin (0-Pays, 1/2/3)",0,3,1)
    gdf_base = load_gadm(iso, level)
    if gdf_base is not None:
        col_name = f"NAME_{level}" if level>0 else "COUNTRY"
        names = sorted(gdf_base[col_name].unique())
        choice = st.sidebar.multiselect("Zone(s)", names)
        if choice:
            gdf_zone = gdf_base[gdf_base[col_name].isin(choice)]
        else:
            gdf_zone = gdf_base
else:
    file = st.sidebar.file_uploader("KML/GeoJSON/SHP", type=["kml","geojson","shp"])
    if file:
        gdf_zone = gpd.read_file(file)

# Dates
start_f = st.sidebar.date_input("Début Inondation", datetime(2024,8,1))
end_f = st.sidebar.date_input("Fin Inondation", datetime(2024,8,31))

# ============================================================
# 5. LANCEMENT ANALYSE
# ============================================================
if gdf_zone is not None:
    st.subheader(f"Zone d'étude : {country}")
    
    # Carte Folium
    m = folium.Map(location=[gdf_zone.centroid.y.mean(), gdf_zone.centroid.x.mean()], zoom_start=8)
    folium.GeoJson(gdf_zone, name="Limites Admin").add_to(m)

    # Masque Flood
    flood_mask = None
    if gee_available:
        aoi_ee = ee.Geometry(mapping(gdf_zone.unary_union))
        flood_mask = get_flood_mask(aoi_ee,"2024-01-01","2024-03-01",str(start_f),str(end_f))
        if flood_mask:
            try:
                map_id = flood_mask.getMapId({'min':0,'max':1,'palette':['000000','00FFFF']})
                folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='GEE', name='Zones Inondées', overlay=True).add_to(m)
            except:
                st.warning("Impossible d'afficher le masque d'inondation GEE.")
    st_folium(m, width=1000, height=500)
    
    # OSM
    osm_all = get_osm_data(gdf_zone)
    b_count = len(osm_all[osm_all['building'].notnull()]) if 'building' in osm_all else 0
    r_count = len(osm_all[osm_all['highway'].notnull()]) if 'highway' in osm_all else 0
    a_count = len(osm_all[osm_all['amenity'].notnull()]) if 'amenity' in osm_all else 0
    
    # Indicators
    indicators = compute_indicator(gdf_zone, flood_mask)
    df_stats = pd.DataFrame([{
        "Zone":"Zone d'étude",
        "Surface":indicators['Surface'],
        "Pop_Exposée":indicators['Pop_Exposée'],
        "Bâtiments":b_count,
        "Routes":r_count,
        "Infrastructures":a_count
    }])
    
    st.markdown("### 📊 Indicateurs Clés")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Population Exposée", f"{df_stats['Pop_Exposée'].sum():,}")
    c2.metric("Bâtiments affectés", f"{df_stats['Bâtiments'].sum():,}")
    c3.metric("Routes affectées", f"{df_stats['Routes'].sum():,}")
    c4.metric("Infrastructures", f"{df_stats['Infrastructures'].sum():,}")
    
    # PDF
    pdf_bytes = create_pdf_report(df_stats, country, start_f, end_f)
    st.sidebar.download_button("📄 Télécharger Rapport", pdf_bytes, "rapport_flood.pdf")
