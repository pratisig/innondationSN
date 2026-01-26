# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP (OSMnx version)
# West Africa – FAO GAUL + OSM
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import mapping, Polygon
from shapely.ops import unary_union
from pyproj import Geod
from streamlit_folium import st_folium
from fpdf import FPDF
import osmnx as ox
import random

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Analyse d'Impact Inondations – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Analyse d'Impact Inondations & Planification d'Urgence")
st.caption("OSM | FAO GAUL (Admin 1-3)")

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def get_true_area_km2(geom_shapely):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom_shapely)[0])
    return area / 1e6

def create_pdf_report(df, country, stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, f"Rapport d'Impact Inondation - {country}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "1. Resume des Indicateurs Clefs", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(190, 8, f"- Surface Inondee Totale: {stats['area']:.2f} km2", ln=True)
    pdf.cell(190, 8, f"- Population Exposee: {stats['pop']:,}", ln=True)
    pdf.cell(190, 8, f"- Batiments Touches: {stats['buildings']:,}", ln=True)
    pdf.cell(190, 8, f"- Routes Affectees: {stats['roads']:.1f} km", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "2. Detail par Zone Administrative", ln=True)
    pdf.set_font("Arial", "B", 7)
    cols = ["Zone", "Surf.(km2)", "Pop.Exp", "Bat.Touch", "Routes(km)"]
    for col in cols: pdf.cell(38, 8, col, border=1)
    pdf.ln()

    pdf.set_font("Arial", "", 7)
    for _, row in df.iterrows():
        pdf.cell(38, 8, str(row['Zone'])[:22], border=1)
        pdf.cell(38, 8, f"{row['Inondé (km2)']:.2f}", border=1)
        pdf.cell(38, 8, f"{row['Pop. Exposée']:,}", border=1)
        pdf.cell(38, 8, f"{row['Bâtiments Affectés']:,}", border=1)
        pdf.cell(38, 8, f"{row['Routes Affectées (km)']:.1f}", border=1, ln=True)

    return pdf.output(dest='S').encode('latin-1')

# ------------------------------------------------------------
# SIDEBAR - CASCADE ADMINISTRATIVE
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection Administrative")
country_name = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])
adm_level1 = st.sidebar.text_input("Région (Admin 1, ex: Dakar)", "Dakar")
adm_level2 = st.sidebar.text_input("Zone (Admin 2, ex: Pikine)", "Pikine")

# ------------------------------------------------------------
# LOAD GAUL POLYGONS
# ------------------------------------------------------------
# Pour démo, on simule un polygon
poly = Polygon([[-17.5, 14.7], [-17.4, 14.7], [-17.4, 14.8], [-17.5, 14.8]])
gdf = gpd.GeoDataFrame([{"Zone": adm_level2, "geometry": poly}], crs="EPSG:4326")
merged_poly = unary_union(gdf.geometry)

# ------------------------------------------------------------
# SIMULATED FLOOD EXTENT
# ------------------------------------------------------------
# Pour demo, on fait un buffer aléatoire pour simuler l'inondation
flood_poly = merged_poly.buffer(0.005)  # ~0.5 km
flood_km2 = get_true_area_km2(flood_poly)
pop_exposed = random.randint(50, 200)

# ------------------------------------------------------------
# OSMnx INFRASTRUCTURE
# ------------------------------------------------------------
with st.spinner("Chargement des données OSM..."):
    # Bâtiments
    buildings = ox.geometries_from_polygon(merged_poly, tags={"building": True})
    buildings_count = len(buildings)

    # Routes
    roads = ox.graph_from_polygon(merged_poly, network_type='drive')
    roads_length = sum(ox.utils_graph.get_route_edge_attributes(roads, list(roads.edges), "length")) / 1000  # km

    # Santé
    health = ox.geometries_from_polygon(merged_poly, tags={"amenity": ["hospital","clinic","doctors","pharmacy"]})
    health_count = len(health)

    # Education
    edu = ox.geometries_from_polygon(merged_poly, tags={"amenity": ["school","university","college","kindergarten"]})
    edu_count = len(edu)

# ------------------------------------------------------------
# DASHBOARD DATAFRAME
# ------------------------------------------------------------
df_res = pd.DataFrame([{
    "Zone": adm_level2,
    "Inondé (km2)": round(flood_km2, 2),
    "% Inondé": round(flood_km2 / get_true_area_km2(poly) * 100, 1),
    "Pop. Exposée": pop_exposed,
    "Bâtiments Affectés": buildings_count,
    "Santé": health_count,
    "Éducation": edu_count,
    "Routes Affectées (km)": round(roads_length,1)
}])

# ------------------------------------------------------------
# FOLIUM MAP
# ------------------------------------------------------------
m = folium.Map(location=[merged_poly.centroid.y, merged_poly.centroid.x], zoom_start=13, tiles="CartoDB dark_matter")
folium.GeoJson(flood_poly, style_function=lambda x: {'fillColor':'#00D4FF','color':'white','weight':1,'fillOpacity':0.4},
               popup=f"Inondé: {flood_km2:.2f} km²").add_to(m)

st.subheader("🗺️ Carte d'Inondation et Infrastructures")
st_folium(m, width=1000, height=500)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
st.write("---")
st.markdown("### 📊 Tableau de Bord des Dommages")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Pop. Exposée", f"{pop_exposed}")
c2.metric("Bâtiments", f"{buildings_count}")
c3.metric("🏥 Santé", f"{health_count}")
c4.metric("🎓 Éducation", f"{edu_count}")
c5.metric("🛣️ Routes", f"{roads_length:.1f} km")

# ------------------------------------------------------------
# PDF EXPORT
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Export")
pdf_b = create_pdf_report(df_res, country_name, {
    "area": flood_km2, "pop": pop_exposed,
    "buildings": buildings_count, "roads": roads_length
})
st.sidebar.download_button("📄 Télécharger Rapport Décisionnel", pdf_b, "rapport_impact.pdf")

# ------------------------------------------------------------
# TABLEAU DE DONNÉES
# ------------------------------------------------------------
st.dataframe(df_res, use_container_width=True)
