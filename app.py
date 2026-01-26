# ============================================================
# APP INONDATIONS – VERSION STABLE OSMNX
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import folium
from shapely.geometry import shape
from streamlit_folium import st_folium

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="🌊 Flood Impact Analyzer",
    layout="wide"
)

st.title("🌊 Analyse d'Impact des Inondations")

# ============================================================
# UPLOAD ZONE
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Charger une zone (GeoJSON / SHP)",
    type=["geojson", "shp"]
)

if uploaded_file is None:
    st.stop()

# ============================================================
# LECTURE ZONE
# ============================================================

if uploaded_file.name.endswith(".geojson"):
    zone = gpd.read_file(uploaded_file)
else:
    zone = gpd.read_file(uploaded_file)

zone = zone.to_crs(epsg=4326)
aoi_polygon = zone.geometry.unary_union

# ============================================================
# SIMULATION ZONE INONDÉE (à remplacer par raster réel)
# ============================================================

flood_zone = zone.copy()
flood_zone["flood"] = 1

# ============================================================
# SURFACES
# ============================================================

zone_m = zone.to_crs(epsg=3857)
flood_m = flood_zone.to_crs(epsg=3857)

total_area_km2 = zone_m.area.sum() / 1e6
flood_area_km2 = flood_m.area.sum() / 1e6
pct_flood = (flood_area_km2 / total_area_km2) * 100

# ============================================================
# POPULATION (exemple fixe – remplaçable WorldPop)
# ============================================================

population_exposed = int(flood_area_km2 * 120)  # densité simulée

# ============================================================
# CHARGEMENT OSM VIA OSMNX
# ============================================================

with st.spinner("📡 Chargement des infrastructures OSM..."):

    osm = ox.features_from_polygon(
        aoi_polygon,
        tags={
            "building": True,
            "highway": True,
            "amenity": ["hospital", "school"]
        }
    )

buildings = osm[osm["building"].notna()]
roads = osm[osm["highway"].notna()]
health = osm[osm["amenity"] == "hospital"]
schools = osm[osm["amenity"] == "school"]

# ============================================================
# INTERSECTION AVEC INONDATION
# ============================================================

buildings_f = gpd.overlay(buildings, flood_zone, how="intersection")
roads_f = gpd.overlay(roads, flood_zone, how="intersection")
health_f = gpd.overlay(health, flood_zone, how="intersection")
schools_f = gpd.overlay(schools, flood_zone, how="intersection")

# ============================================================
# INDICATEURS
# ============================================================

roads_f_m = roads_f.to_crs(epsg=3857)
roads_f_m["km"] = roads_f_m.length / 1000

# ============================================================
# DASHBOARD – INDICATEURS AVANT TABLE
# ============================================================

st.subheader("📊 Indicateurs clés")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Surface totale (km²)", f"{total_area_km2:.2f}")
col2.metric("Surface inondée (km²)", f"{flood_area_km2:.2f}")
col3.metric("% Inondé", f"{pct_flood:.1f}%")
col4.metric("Population exposée", population_exposed)
col5.metric("Routes affectées (km)", f"{roads_f_m['km'].sum():.2f}")

col6, col7, col8 = st.columns(3)
col6.metric("🏠 Bâtiments", len(buildings_f))
col7.metric("🏥 Santé", len(health_f))
col8.metric("🎓 Éducation", len(schools_f))

# ============================================================
# FILTRES
# ============================================================

st.subheader("🎛️ Filtres carte")

show_buildings = st.checkbox("Bâtiments", True)
show_roads = st.checkbox("Routes", True)
show_health = st.checkbox("Santé", True)
show_schools = st.checkbox("Éducation", True)

# ============================================================
# CARTE
# ============================================================

st.subheader("🗺️ Carte des impacts")

m = folium.Map(location=[zone.geometry.centroid.y.mean(),
                          zone.geometry.centroid.x.mean()],
               zoom_start=10)

folium.GeoJson(zone, name="Zone").add_to(m)
folium.GeoJson(flood_zone, name="Zone inondée",
               style_function=lambda x: {"fillColor": "blue", "fillOpacity": 0.4}).add_to(m)

if show_buildings:
    folium.GeoJson(buildings_f, name="Bâtiments impactés").add_to(m)

if show_roads:
    folium.GeoJson(roads_f, name="Routes impactées").add_to(m)

if show_health:
    folium.GeoJson(health_f, name="Santé").add_to(m)

if show_schools:
    folium.GeoJson(schools_f, name="Éducation").add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, height=600)

# ============================================================
# TABLEAU RÉCAPITULATIF (EN DERNIER)
# ============================================================

st.subheader("📋 Tableau récapitulatif")

table = pd.DataFrame({
    "Indicateur": [
        "Surface totale (km²)",
        "Surface inondée (km²)",
        "% Inondé",
        "Population exposée",
        "Bâtiments impactés",
        "Routes impactées (km)",
        "Hôpitaux",
        "Écoles"
    ],
    "Valeur": [
        round(total_area_km2, 2),
        round(flood_area_km2, 2),
        round(pct_flood, 1),
        population_exposed,
        len(buildings_f),
        round(roads_f_m["km"].sum(), 2),
        len(health_f),
        len(schools_f)
    ]
})

st.dataframe(table, use_container_width=True)
