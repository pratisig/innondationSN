# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP
# West Africa – Sentinel / CHIRPS / WorldPop / OSM
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import tempfile, os, json, zipfile
import pandas as pd
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from pyproj import Geod

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Flood Impact Analysis – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Flood Impact Analysis & Emergency Planning")
st.caption("Sentinel-1 (SAR) | CHIRPS (Pluie) | WorldPop | FAO GAUL (Limites Admin)")

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
# DATASETS & UTILS
# ------------------------------------------------------------
def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

# FAO GAUL Collections
GAUL1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
GAUL2 = ee.FeatureCollection("FAO/GAUL/2015/level2")

# ------------------------------------------------------------
# SIDEBAR - SELECTION ADMINISTRATIVE EN CASCADE
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection de la zone")

# Pays (Fixé sur Sénégal par défaut pour cet exemple, modifiable)
country = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

# Admin 1 (Régions)
admin1_fc = GAUL1.filter(ee.Filter.eq('ADM0_NAME', country))
admin1_list = admin1_fc.aggregate_array('ADM1_NAME').sort().getInfo()
selected_admin1 = st.sidebar.multiselect(f"Régions (Admin 1) - {country}", admin1_list)

final_aoi_fc = None
label_col = 'ADM2_NAME'

if selected_admin1:
    # Admin 2 (Départements / Districts)
    admin2_fc = GAUL2.filter(ee.Filter.inList('ADM1_NAME', selected_admin1))
    admin2_list = admin2_fc.aggregate_array('ADM2_NAME').sort().getInfo()
    selected_admin2 = st.sidebar.multiselect("Départements (Admin 2)", admin2_list)
    
    if selected_admin2:
        final_aoi_fc = admin2_fc.filter(ee.Filter.inList('ADM2_NAME', selected_admin2))
    else:
        final_aoi_fc = admin2_fc
        label_col = 'ADM2_NAME'
else:
    st.info("Veuillez sélectionner au moins une région pour démarrer l'analyse.")
    st.stop()

# Conversion pour le traitement local
with st.spinner("Chargement de la géométrie..."):
    # Limiter le nombre de features pour la performance de l'UI
    gdf = gpd.GeoDataFrame.from_features(final_aoi_fc.getInfo(), crs="EPSG:4326")
    # On s'assure que l'index est propre pour la correspondance
    gdf = gdf.reset_index(drop=True)
    merged_poly = unary_union(gdf.geometry)
    geom_ee = ee.Geometry(mapping(merged_poly))

st.sidebar.success(f"✅ {len(gdf)} entités sélectionnées.")

# ------------------------------------------------------------
# DATE SELECTION
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période d’analyse")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2024-08-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2024-09-30"))

# ------------------------------------------------------------
# GEE ANALYSIS ENGINE
# ------------------------------------------------------------
@st.cache_data
def run_analysis(aoi_json, start, end):
    aoi = ee.Geometry(aoi_json)
    s = str(start)
    e = str(end)

    # 1. Flood Detection (Sentinel-1 SAR)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(s, e)\
           .filter(ee.Filter.eq("instrumentMode","IW")).select("VV")
    
    # Vérification disponibilité images
    if s1.size().getInfo() == 0:
        return None, None, None, None, None

    before = s1.limit(5).median()
    after = s1.sort('system:time_start', False).limit(5).median()
    flood = after.subtract(before).lt(-3)
    
    # Masquage pente et eau permanente
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    flood_clean = flood.updateMask(slope.lt(5)).updateMask(perm.lt(10)).selfMask()

    # 2. Rainfall (CHIRPS)
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(s, e).sum().rename('precip')

    # 3. Population (WorldPop)
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(aoi).mean().rename('pop')

    # 4. Infrastructure (MS Global Buildings)
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(aoi)
    bldg_img = buildings.map(lambda f: f.set('c',1)).reduceToImage(['c'], ee.Reducer.first()).rename('bldgs')

    return flood_clean, rain, pop, bldg_img, None

with st.spinner("Analyse satellite en cours (GEE)..."):
    flood_img, rain_img, pop_img, bldg_img, _ = run_analysis(geom_ee.getInfo(), start_date, end_date)

if flood_img is None:
    st.error("Aucune image Sentinel-1 disponible sur cette période/zone.")
    st.stop()

# ------------------------------------------------------------
# BATCH CALCULATION
# ------------------------------------------------------------
@st.cache_data
def compute_metrics(gdf_json, start, end):
    features = []
    for idx, row in gdf.iterrows():
        f = ee.Feature(ee.Geometry(mapping(row.geometry)), {
            'orig_index': idx, 'nom': str(row[label_col]), 'area_km2': get_true_area_km2(row.geometry)
        })
        features.append(f)
    fc = ee.FeatureCollection(features)
    
    pix_area = ee.Image.pixelArea()
    combined = ee.Image.cat([
        flood_img.multiply(pix_area).rename('f_area'),
        pop_img.updateMask(flood_img).rename('p_exp'),
        rain_img.rename('rain'),
        bldg_img.updateMask(flood_img).rename('b_exp')
    ])

    return combined.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100).getInfo()

with st.spinner("Calcul des statistiques d'impact..."):
    results = compute_metrics(None, start_date, end_date)
    
    rows = []
    for f in results['features']:
        p = f['properties']
        f_km2 = (p.get('f_area', 0)) / 1e6
        area_total = p.get('area_km2', 1)
        rows.append({
            "orig_index": p['orig_index'],
            "Zone": p['nom'],
            "Surface Totale (km2)": area_total,
            "Inondé (km2)": f_km2,
            "% Inondé": (f_km2 / area_total * 100),
            "Pop. Exposée": int(p.get('p_exp', 0)),
            "Bâtiments Impactés": int(p.get('b_exp', 0)),
            "Précip. Cumulée (mm)": (p.get('rain', 0) / (area_total * 100)) # Approximation spatiale
        })
    df = pd.DataFrame(rows)

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
st.subheader(f"📊 Impact des Inondations : {', '.join(selected_admin1)}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Surface Inondée", f"{df['Inondé (km2)'].sum():.2f} km²")
c2.metric("Population Exposée", f"{df['Pop. Exposée'].sum():,}")
c3.metric("Bâtiments Touchés", f"{df['Bâtiments Impactés'].sum():,}")
c4.metric("Zones Critiques", f"{len(df[df['% Inondé'] > 5])}")

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------
st.subheader("🗺️ Cartographie de l'urgence")
m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=9, tiles="CartoDB positron")

# GEE Layers
def add_ee_layer(img, vis, name):
    try:
        map_id = img.getMapId(vis)
        folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='GEE', name=name, overlay=True).add_to(m)
    except: pass

add_ee_layer(flood_img, {'palette':['#00FFFF']}, "Eau détectée (Satellite)")
add_ee_layer(bldg_img.updateMask(flood_img), {'palette':['#FF0000']}, "Bâtiments Impactés")

# Polygons avec couleur selon l'impact
for _, r in df.iterrows():
    # Utilisation de l'index d'origine pour éviter les erreurs de sélection par nom
    orig_geom = gdf.loc[r['orig_index']].geometry
    color = "red" if r['% Inondé'] > 10 else "orange" if r['% Inondé'] > 2 else "green"
    folium.GeoJson(
        orig_geom,
        style_function=lambda x, c=color: {"fillColor": c, "color": "black", "weight": 1, "fillOpacity": 0.4},
        tooltip=f"{r['Zone']}: {r['% Inondé']:.1f}% inondé | {r['Pop. Exposée']:,} pers."
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=500)

# ------------------------------------------------------------
# TABLE & DOWNLOAD
# ------------------------------------------------------------
st.subheader("📋 Détails analytiques")
# On retire la colonne technique d'index pour l'affichage utilisateur
df_display = df.drop(columns=['orig_index'])
st.dataframe(df_display.style.format({
    "Surface Totale (km2)": "{:.2f}",
    "Inondé (km2)": "{:.2f}",
    "% Inondé": "{:.1f}%",
    "Pop. Exposée": "{:,}",
    "Bâtiments Impactés": "{:,}"
}), use_container_width=True)

st.download_button("⬇️ Exporter les données (CSV)", df_display.to_csv(index=False), "rapport_impact_admin.csv")
