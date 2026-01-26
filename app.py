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
    """Calcule la surface réelle en km2 via pyproj Geod pour éviter les erreurs de projection."""
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

# Nettoyage des noms de colonnes pour trouver un label
label_col = next((c for c in gdf.columns if c.lower() in ['name', 'nom', 'libelle', 'id_zone']), None)

# Géométrie globale simplifiée
merged_poly = gdf.geometry.unary_union.simplify(0.001, preserve_topology=True)
geom = shapely_to_ee(merged_poly)
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
    t_start, t_end = pd.to_datetime(start), pd.to_datetime(end)
    d_start = t_start.strftime('%Y-%m-%d')
    d_start_p15 = (t_start+pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    d_end_m15 = (t_end-pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    d_end = t_end.strftime('%Y-%m-%d')

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(d_start, d_end)
          .filter(ee.Filter.eq("instrumentMode","IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
          .select("VV"))
    
    before = s1.filterDate(d_start, d_start_p15).median()
    after = s1.filterDate(d_end_m15, d_end).median()
    
    diff = after.subtract(before)
    flood = diff.lt(-3)
    
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    water_perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    
    flood_clean = flood.updateMask(slope.lt(5)).updateMask(water_perm.lt(10))
    return flood_clean.selfMask()

with st.spinner("Analyse satellite Sentinel-1..."):
    flood_img = detect_floods(geom.getInfo(), start_date, end_date)

# ------------------------------------------------------------
# INDICATEURS PAR ZONE (ENRICHIS)
# ------------------------------------------------------------
pixel_area = ee.Image.pixelArea()
pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01","2020-12-31").mean().select('population')

# Dataset OSM (Infrastructures) via MS Global ML buildings ou OSM High-res
# On utilise ici un proxy OSM via un FeatureCollection si dispo, ou extraction de points
infra_osm = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(geom)

zone_metrics = []
with st.spinner("Calcul des impacts par zone..."):
    for idx, row in gdf.iterrows():
        poly = row.geometry
        ee_poly = shapely_to_ee(poly)
        name_zone = row[label_col] if label_col else f"Zone {idx+1}"
        
        # Surface totale réelle
        total_area_km2 = get_true_area_km2(poly)
        
        try:
            # Stats GEE
            stats = ee.Dictionary({
                'flood_m2': flood_img.multiply(pixel_area).reduceRegion(ee.Reducer.sum(), ee_poly, 100).get("VV"),
                'pop_total': pop_img.reduceRegion(ee.Reducer.sum(), ee_poly, 100).get("population"),
                'pop_exposed': pop_img.updateMask(flood_img).reduceRegion(ee.Reducer.sum(), ee_poly, 100).get("population"),
                'infra_exposed': infra_osm.filterBounds(ee_poly).filter(ee.Filter.intersects(".geo", ee.Feature(ee_poly).geometry())).size()
            }).getInfo()
            
            flood_km2 = (stats.get('flood_m2') or 0) / 1e6
            pop_total = int(stats.get('pop_total') or 0)
            pop_exposed = int(stats.get('pop_exposed') or 0)
            infra_count = int(stats.get('infra_exposed') or 0)
            
        except Exception as e:
            flood_km2, pop_total, pop_exposed, infra_count = 0, 0, 0, 0
            
        pct_flood = (flood_km2 / total_area_km2 * 100) if total_area_km2 > 0 else 0
        pct_pop = (pop_exposed / pop_total * 100) if pop_total > 0 else 0
        
        zone_metrics.append({
            "id": idx,
            "nom": name_zone,
            "surface_totale_km2": total_area_km2,
            "surface_inondee_km2": flood_km2,
            "pct_inonde": pct_flood,
            "pop_totale": pop_total,
            "pop_exposee": pop_exposed,
            "pct_pop_exposee": pct_pop,
            "infra_impactees": infra_count
        })

df_metrics = pd.DataFrame(zone_metrics)

# ------------------------------------------------------------
# INDICATEURS GLOBAUX
# ------------------------------------------------------------
st.subheader("📊 Synthèse de l'impact")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Surface Inondée", f"{df_metrics.surface_inondee_km2.sum():.2f} km²")
c2.metric("Population Exposée", f"{df_metrics.pop_exposee.sum():,}")
c3.metric("Bâtiments impactés", f"{df_metrics.infra_impactees.sum():,}")
c4.metric("Zone la plus touchée", f"{df_metrics.loc[df_metrics.pct_inonde.idxmax(), 'nom']}" if not df_metrics.empty else "-")

# ------------------------------------------------------------
# CARTE INTERACTIVE
# ------------------------------------------------------------
st.subheader("🗺️ Cartographie des risques")
bounds = gdf.total_bounds
m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], zoom_start=10, tiles="CartoDB positron")

def get_color(pct):
    if pct < 5: return "#2ECC71"
    elif pct < 15: return "#F1C40F"
    elif pct < 40: return "#E67E22"
    else: return "#E74C3C"

for idx, row in gdf.iterrows():
    m_row = df_metrics[df_metrics.id == idx].iloc[0]
    
    popup_html = f"""
    <div style="font-family: Arial; width: 200px;">
        <h4 style="margin-bottom:5px;">{m_row.nom}</h4>
        <hr style="margin:5px 0;">
        <b>Surface Totale:</b> {m_row.surface_totale_km2:.2f} km²<br>
        <b>Inondée:</b> {m_row.surface_inondee_km2:.2f} km² ({m_row.pct_inonde:.1f}%)<br>
        <br>
        <b>Pop. Totale:</b> {m_row.pop_totale:,}<br>
        <b>Pop. Exposée:</b> {m_row.pop_exposee:,} ({m_row.pct_pop_exposee:.1f}%)<br>
        <br>
        <b>Bâtiments impactés:</b> {m_row.infra_impactees:,}
    </div>
    """
    
    folium.GeoJson(
        row.geometry,
        style_function=lambda x, p=m_row.pct_inonde: {
            "fillColor": get_color(p),
            "color": "#333", "weight": 1.5, "fillOpacity": 0.5
        },
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{m_row.nom}: {m_row.pct_inonde:.1f}% inondé"
    ).add_to(m)

# Overlay Sentinel-1 (Bleu vif)
try:
    flood_mapid = flood_img.getMapId({'min': 0, 'max': 1, 'palette': ['#00FFFF']})
    folium.TileLayer(
        tiles=flood_mapid['tile_fetcher'].url_format,
        attr='GEE Sentinel-1', name='Détection Satellite (Eau)',
        overlay=True, opacity=0.8
    ).add_to(m)
except: pass

folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=600)

# ------------------------------------------------------------
# TABLEAU ET EXPORT
# ------------------------------------------------------------
st.subheader("📋 Rapport détaillé par zone")
st.dataframe(df_metrics.drop(columns=['id']).style.format({
    "surface_totale_km2": "{:.2f}",
    "surface_inondee_km2": "{:.2f}",
    "pct_inonde": "{:.1f}%",
    "pop_totale": "{:,}",
    "pop_exposee": "{:,}",
    "pct_pop_exposee": "{:.1f}%",
    "infra_impactees": "{:,}"
}), use_container_width=True)

csv = df_metrics.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Télécharger le rapport complet (CSV)", data=csv, file_name="rapport_inondation.csv", mime="text/csv")
