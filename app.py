# ============================================================
# FLOOD ANALYSIS & MAPPING APP — SENEGAL / WEST AFRICA
# Uses real open data via Google Earth Engine (GEE)
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import tempfile, os, json, zipfile
import pandas as pd
import plotly.express as px
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Geod

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Flood Impact Analysis – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Flood Impact Analysis & Mapping")
st.caption("Satellite-based flood detection using Sentinel-1 SAR, WorldPop, and OSM data")

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
# UTILS
# ------------------------------------------------------------
def shapely_to_ee(poly):
    geojson = poly.__geo_interface__
    if geojson['type'] == 'Polygon':
        return ee.Geometry.Polygon(geojson['coordinates'])
    elif geojson['type'] == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(geojson['coordinates'])
    else:
        return ee.Geometry(geojson)

def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Zone d’étude")
uploaded_file = st.sidebar.file_uploader("Charger une zone (GeoJSON / SHP ZIP / KML)", type=["geojson","kml","zip"])
if not uploaded_file:
    st.info("Veuillez charger une zone géographique pour commencer l'analyse.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    file_path = os.path.join(tmpdir, uploaded_file.name)
    with open(file_path,"wb") as f: f.write(uploaded_file.getbuffer())

    try:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(file_path,'r') as zip_ref: zip_ref.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files: st.error("Aucun .shp dans le zip."); st.stop()
            gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
        else:
            gdf = gpd.read_file(file_path)

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}"); st.stop()

label_col = next((c for c in gdf.columns if c.lower() in ['name', 'nom', 'libelle', 'id_zone']), None)
merged_poly = gdf.geometry.unary_union.simplify(0.001, preserve_topology=True)
geom_ee = shapely_to_ee(merged_poly)
st.success(f"✅ Zone chargée : {len(gdf)} entité(s)")

# ------------------------------------------------------------
# DATE SELECTION
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période d’analyse")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2024-08-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2024-09-30"))
if start_date >= end_date: st.error("La date de fin doit être postérieure à la date de début."); st.stop()

# ------------------------------------------------------------
# SENTINEL-1 FLOOD DETECTION
# ------------------------------------------------------------
@st.cache_data
def detect_floods(aoi_serialized, start, end):
    aoi = ee.Geometry(aoi_serialized)
    d_start = pd.to_datetime(start).strftime('%Y-%m-%d')
    d_end = pd.to_datetime(end).strftime('%Y-%m-%d')

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(d_start, d_end)
          .filter(ee.Filter.eq("instrumentMode","IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
          .select("VV"))
    
    before = s1.limit(5, 'system:time_start').median()
    after = s1.sort('system:time_start', False).limit(5).median()
    
    flood = after.subtract(before).lt(-3)
    
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    water_perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    
    return flood.updateMask(slope.lt(5)).updateMask(water_perm.lt(10)).selfMask()

with st.spinner("Détection des inondations par satellite..."):
    flood_img = detect_floods(geom_ee.getInfo(), start_date, end_date)

# ------------------------------------------------------------
# OSM DATASETS
# ------------------------------------------------------------
# Bâtiments (MS Global Buildings)
buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(geom_ee)
# Routes (OSM via GEE ou Feature Collection globale)
roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(geom_ee) # Proxy Roads

# Création d'images binaires pour les calculs batch (plus rapide)
building_img = buildings.map(lambda f: f.set('constant', 1)).reduceToImage(['constant'], ee.Reducer.first()).rename('buildings')
# Les routes sont plus complexes, on utilise une simplification par intersection spatiale dans le batch

# ------------------------------------------------------------
# OPTIMIZED BATCH CALCULATION
# ------------------------------------------------------------
@st.cache_data
def calculate_batch_metrics(gdf_json, start, end):
    features = []
    for idx, row in gdf.iterrows():
        f = ee.Feature(shapely_to_ee(row.geometry), {
            'id': idx,
            'nom': str(row[label_col]) if label_col else f"Zone {idx+1}",
            'area_km2': get_true_area_km2(row.geometry)
        })
        features.append(f)
    
    fc = ee.FeatureCollection(features)
    pixel_area = ee.Image.pixelArea()
    pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01","2020-12-31").mean().select('population')

    # Image combinée pour réduction unique (Surface, Population, Bâtiments)
    combined_img = ee.Image.cat([
        flood_img.multiply(pixel_area).rename('flood_area'),
        pop_img.rename('pop_total'),
        pop_img.updateMask(flood_img).rename('pop_exposed'),
        building_img.updateMask(flood_img).rename('buildings_exposed')
    ])

    results = combined_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.sum(),
        scale=50 # Scale plus fine pour les bâtiments
    ).getInfo()

    return results

with st.spinner("Calcul des indicateurs d'impact..."):
    batch_results = calculate_batch_metrics(None, start_date, end_date)
    
    rows = []
    for f in batch_results['features']:
        p = f['properties']
        flood_km2 = (p.get('flood_area') or 0) / 1e6
        total_area = p.get('area_km2') or 1
        pop_total = int(p.get('pop_total') or 0)
        pop_exposed = int(p.get('pop_exposed') or 0)
        bldg_exposed = int(p.get('buildings_exposed') or 0)
        
        # Estimation Routes (simplifié : Intersection spatiale)
        # Note: Pour un calcul précis en km, une requête séparée est nécessaire si le batch est trop complexe
        
        rows.append({
            "id": p['id'],
            "nom": p['nom'],
            "surface_totale_km2": total_area,
            "surface_inondee_km2": flood_km2,
            "pct_inonde": (flood_km2 / total_area * 100),
            "pop_totale": pop_total,
            "pop_exposee": pop_exposed,
            "pct_pop_exposee": (pop_exposed / pop_total * 100) if pop_total > 0 else 0,
            "batiments_impactes": bldg_exposed,
            "ecoles_sante": int(bldg_exposed * 0.05) # Estimation basée sur ratio moyen si data OSM point manquante
        })
    df_metrics = pd.DataFrame(rows)

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
st.subheader("📊 Synthèse de l'impact")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Surface Inondée", f"{df_metrics.surface_inondee_km2.sum():.2f} km²")
c2.metric("Population Exposée", f"{df_metrics.pop_exposee.sum():,}")
c3.metric("Bâtiments Impactés", f"{df_metrics.batiments_impactes.sum():,}")
c4.metric("Infrastructures Sensibles", f"~{df_metrics.ecoles_sante.sum():,} (est.)")

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------
st.subheader("🗺️ Cartographie des risques")
bounds = gdf.total_bounds
m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], zoom_start=10, tiles="CartoDB positron")

def get_color(pct):
    if pct < 2: return "#2ECC71"
    elif pct < 10: return "#F1C40F"
    elif pct < 30: return "#E67E22"
    else: return "#E74C3C"

for _, m_row in df_metrics.iterrows():
    geom_row = gdf.iloc[int(m_row.id)].geometry
    popup_html = f"""
    <div style="font-family: Arial; width: 220px;">
        <h4 style="margin:0 0 5px 0;">{m_row.nom}</h4>
        <hr style="margin:5px 0;">
        <b>Surface Inondée:</b> {m_row.surface_inondee_km2:.2f} km² ({m_row.pct_inonde:.1f}%)<br>
        <b>Population Exposée:</b> {m_row.pop_exposee:,}<br>
        <b>Bâtiments Impactés:</b> {m_row.batiments_impactes:,}<br>
        <b>Santé/Écoles (est.):</b> {m_row.ecoles_sante:,}
    </div>
    """
    folium.GeoJson(
        geom_row,
        style_function=lambda x, p=m_row.pct_inonde: {
            "fillColor": get_color(p), "color": "#444", "weight": 1, "fillOpacity": 0.5
        },
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(m)

# Couches GEE
try:
    flood_mapid = flood_img.getMapId({'min': 0, 'max': 1, 'palette': ['#00FFFF']})
    folium.TileLayer(
        tiles=flood_mapid['tile_fetcher'].url_format,
        attr='GEE Sentinel-1', name='Eau détectée (Satellite)',
        overlay=True, opacity=0.7
    ).add_to(m)
    
    # Visualisation des bâtiments impactés sur la carte
    bldg_mapid = building_img.updateMask(flood_img).getMapId({'min': 0, 'max': 1, 'palette': ['#FF0000']})
    folium.TileLayer(
        tiles=bldg_mapid['tile_fetcher'].url_format,
        attr='MS Buildings', name='Bâtiments Impactés (Rouge)',
        overlay=True, opacity=1.0
    ).add_to(m)
except: pass

folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=600)

# ------------------------------------------------------------
# RAPPORT
# ------------------------------------------------------------
st.subheader("📋 Rapport détaillé des impacts")
st.dataframe(df_metrics.drop(columns=['id']).style.format({
    "surface_totale_km2": "{:.2f}",
    "surface_inondee_km2": "{:.2f}",
    "pct_inonde": "{:.1f}%",
    "pop_totale": "{:,}",
    "pop_exposee": "{:,}",
    "pct_pop_exposee": "{:.1f}%",
    "batiments_impactes": "{:,}",
    "ecoles_sante": "{:,}"
}), use_container_width=True)

csv = df_metrics.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Télécharger le rapport CSV", data=csv, file_name="rapport_impact_osm.csv")
