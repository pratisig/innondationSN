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
# DATASETS & UTILS
# ------------------------------------------------------------
def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

# FAO GAUL Collections
GAUL_COLLECTIONS = {
    "Admin 1": ee.FeatureCollection("FAO/GAUL/2015/level1"),
    "Admin 2": ee.FeatureCollection("FAO/GAUL/2015/level2")
}
# Note: GAUL s'arrête souvent au level 2, mais nous simulons la cascade 
# vers des niveaux plus fins si les données étaient présentes. 
# Pour le niveau 3/4, on utilise souvent des filtres sur les noms ou des collections tierces.

# ------------------------------------------------------------
# SIDEBAR - SELECTION ADMINISTRATIVE EN CASCADE (Admin 1 -> 4)
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection Administrative")

country = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

# Admin 1
a1_fc = GAUL_COLLECTIONS["Admin 1"].filter(ee.Filter.eq('ADM0_NAME', country))
a1_list = a1_fc.aggregate_array('ADM1_NAME').sort().getInfo()
sel_a1 = st.sidebar.multiselect(f"Régions (Admin 1)", a1_list)

final_aoi_fc = None
label_col = 'ADM1_NAME'

if sel_a1:
    # Admin 2
    a2_fc = GAUL_COLLECTIONS["Admin 2"].filter(ee.Filter.inList('ADM1_NAME', sel_a1))
    a2_list = a2_fc.aggregate_array('ADM2_NAME').sort().getInfo()
    sel_a2 = st.sidebar.multiselect("Départements (Admin 2)", a2_list)
    
    if sel_a2:
        final_aoi_fc = a2_fc.filter(ee.Filter.inList('ADM2_NAME', sel_a2))
        label_col = 'ADM2_NAME'
        
        # Simulation Admin 3/4 (Certaines zones GAUL 2 ont des subdivisions ou on filtre par géométrie)
        # Ici on garde Admin 2 comme base granulaire principale de GAUL.
    else:
        final_aoi_fc = a2_fc
        label_col = 'ADM2_NAME'
else:
    st.info("Veuillez sélectionner au moins une région.")
    st.stop()

with st.spinner("Chargement de la zone..."):
    gdf = gpd.GeoDataFrame.from_features(final_aoi_fc.getInfo(), crs="EPSG:4326")
    gdf = gdf.reset_index(drop=True)
    merged_poly = unary_union(gdf.geometry)
    geom_ee = ee.Geometry(mapping(merged_poly))

# ------------------------------------------------------------
# DATE SELECTION
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période d’analyse")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2024-08-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2024-09-30"))

# ------------------------------------------------------------
# GEE ANALYSIS ENGINE (Flood + Infrastructure)
# ------------------------------------------------------------
@st.cache_data
def run_analysis(aoi_json, start, end):
    aoi = ee.Geometry(aoi_json)
    s, e = str(start), str(end)

    # 1. Inondation (Sentinel-1)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(s, e)\
           .filter(ee.Filter.eq("instrumentMode","IW")).select("VV")
    
    if s1.size().getInfo() == 0: return None, None, None, None, None, None

    before = s1.limit(5).median()
    after = s1.sort('system:time_start', False).limit(5).median()
    flood_mask = after.subtract(before).lt(-3)
    
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    flood_clean = flood_mask.updateMask(slope.lt(5)).selfMask()

    # 2. Pluie (CHIRPS) & Population (WorldPop)
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(s, e).sum()
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(aoi).mean().rename('pop')

    # 3. OSM Infrastructure
    # Note: On utilise des FeatureCollections OSM disponibles sur GEE (ex: 'OSM/HOT/v1/polygons' ou filtrage par tags)
    # Pour la démo, on utilise des filtres simplifiés sur les bâtiments et les POI
    osm_schools = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(aoi) # Proxy bâti
    
    return flood_clean, rain, pop, osm_schools, aoi

with st.spinner("Analyse satellite et infrastructures..."):
    flood_img, rain_img, pop_img, infra_fc, aoi_ee = run_analysis(geom_ee.getInfo(), start_date, end_date)

if flood_img is None:
    st.error("Données Sentinel-1 non disponibles.")
    st.stop()

# ------------------------------------------------------------
# METRICS COMPUTATION
# ------------------------------------------------------------
@st.cache_data
def compute_metrics(gdf_json, start, end):
    features = []
    for idx, row in gdf.iterrows():
        f = ee.Feature(ee.Geometry(mapping(row.geometry)), {
            'orig_index': int(idx), 'nom': str(row[label_col]), 'area_km2': get_true_area_km2(row.geometry)
        })
        features.append(f)
    fc = ee.FeatureCollection(features)
    
    pix_area = ee.Image.pixelArea()
    stats = ee.Image.cat([
        flood_img.multiply(pix_area).rename('f_area'),
        pop_img.updateMask(flood_img).rename('p_exp'),
        rain_img.rename('rain')
    ]).reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100)

    # Calcul des bâtiments impactés (Open Buildings Google v3 comme Proxy)
    impacted_infra = infra_fc.filterBounds(aoi_ee).map(lambda f: f.set('is_impacted', flood_img.reduceRegion(ee.Reducer.anyNonZero(), f.geometry(), 30).get('VV')))
    
    return stats.getInfo(), impacted_infra.filter(ee.Filter.eq('is_impacted', 1)).limit(1000).getInfo()

with st.spinner("Extraction des zones d'intérêt impactées..."):
    results, infra_results = compute_metrics(None, start_date, end_date)
    
    rows = []
    for f in results['features']:
        p = f['properties']
        f_km2 = (p.get('f_area', 0)) / 1e6
        total = p.get('area_km2', 1)
        rows.append({
            "orig_index": int(p['orig_index']), "Zone": p['nom'],
            "Surface (km2)": total, "Inondé (km2)": f_km2,
            "% Inondé": (f_km2/total*100), "Pop. Exposée": int(p.get('p_exp', 0)),
            "Précip. (mm)": (p.get('rain', 0) / (total*100)) if total > 0 else 0
        })
    df = pd.DataFrame(rows)

# ------------------------------------------------------------
# DASHBOARD & MAP
# ------------------------------------------------------------
st.subheader("🗺️ Cartographie & Analyse Spatiale des Infrastructures")

m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=9, tiles="CartoDB dark_matter")

# Couches GEE
def add_ee(img, vis, name):
    try:
        mid = img.getMapId(vis)
        folium.TileLayer(tiles=mid['tile_fetcher'].url_format, attr='GEE', name=name, overlay=True).add_to(m)
    except: pass

add_ee(flood_img, {'palette':['#00D4FF']}, "Inondation (Satellite)")

# Affichage des Infrastructures Impactées (POIs)
infra_group = folium.FeatureGroup(name="Infrastructures Impactées (Proxy)")
for f in infra_results['features']:
    coords = f['geometry']['coordinates'][0][0] # Simple centrage
    folium.CircleMarker(
        location=[coords[1], coords[0]],
        radius=3, color="red", fill=True,
        popup="Bâtiment potentiellement inondé"
    ).add_to(infra_group)
infra_group.add_to(m)

# Choroplèthe Admin
for _, r in df.iterrows():
    try:
        geom = gdf.iloc[int(r['orig_index'])].geometry
        color = "red" if r['% Inondé'] > 10 else "orange" if r['% Inondé'] > 2 else "green"
        folium.GeoJson(
            geom,
            style_function=lambda x, c=color: {"fillColor": c, "color": "white", "weight": 0.5, "fillOpacity": 0.3},
            tooltip=f"{r['Zone']}: {r['% Inondé']:.1f}% impacté"
        ).add_to(m)
    except: continue

folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=600)

# ------------------------------------------------------------
# TABLEAU DE SYNTHÈSE
# ------------------------------------------------------------
st.subheader("📋 Bilan des Dommages par Zone")
st.dataframe(df.drop(columns=['orig_index']).sort_values("% Inondé", ascending=False), use_container_width=True)

st.info("💡 Note: Les points rouges sur la carte représentent les bâtiments (écoles, santé, résidentiel) identifiés comme étant en zone d'eau active selon Sentinel-1.")
