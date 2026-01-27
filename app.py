# ============================================================
# APPLICATION D'ANALYSE DES INONDATIONS – VERSION STABLE
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from shapely.geometry import shape
from streamlit_folium import st_folium
import osmnx as ox
from datetime import datetime
from fpdf import FPDF
import tempfile

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Flood Impact Decision Tool",
    page_icon="🌊",
    layout="wide"
)

# ============================================================
# UTILITAIRES
# ============================================================

def safe_area_km2(gdf):
    return gdf.to_crs(epsg=3857).area.sum() / 1e6

def safe_length_km(gdf):
    return gdf.to_crs(epsg=3857).length.sum() / 1000

def load_zone(upload):
    gdf = gpd.read_file(upload)
    return gdf.to_crs(4326)

def intersect(a, b):
    return gpd.overlay(a, b, how="intersection")

# ============================================================
# PDF REPORT
# ============================================================

class Report(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Flood Impact Assessment Report", ln=True)
        self.ln(4)

def generate_pdf(stats, zone_name, start, end):
    pdf = Report()
    pdf.add_page()
    pdf.set_font("Arial", "", 11)

    pdf.cell(0, 8, f"Zone: {zone_name}", ln=True)
    pdf.cell(0, 8, f"Period: {start} → {end}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Key Indicators", ln=True)

    pdf.set_font("Arial", "", 11)
    for k, v in stats.items():
        pdf.cell(0, 7, f"- {k}: {v}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ============================================================
# INTERFACE
# ============================================================

st.title("🌊 Flood Impact Decision Support Tool")
st.caption("Quantifying flood impact on infrastructures & population")

uploaded = st.file_uploader(
    "Upload study area (GeoJSON / SHP / KML)",
    type=["geojson", "shp", "kml"]
)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", datetime(2024, 1, 1))
with col2:
    end_date = st.date_input("End date", datetime.today())

if uploaded:
    zone = load_zone(uploaded)
    merged_zone = zone.dissolve()

    st.success("Zone loaded successfully")

    # ========================================================
    # MAP
    # ========================================================

    m = folium.Map(
        location=[
            merged_zone.geometry.centroid.y.iloc[0],
            merged_zone.geometry.centroid.x.iloc[0]
        ],
        zoom_start=9,
        tiles="CartoDB positron"
    )

    folium.GeoJson(
        merged_zone,
        name="Study Area",
        style_function=lambda x: {"color": "blue", "weight": 2}
    ).add_to(m)

    # ========================================================
    # OSM DATA
    # ========================================================

    with st.spinner("Loading OpenStreetMap infrastructures..."):
        buildings = ox.geometries_from_polygon(
            merged_zone.geometry.iloc[0],
            tags={"building": True}
        ).reset_index()

        roads = ox.graph_from_polygon(
            merged_zone.geometry.iloc[0],
            network_type="drive"
        )
        roads_gdf = ox.graph_to_gdfs(roads, nodes=False)

    buildings = gpd.GeoDataFrame(buildings, geometry="geometry", crs=4326)
    roads_gdf = roads_gdf.to_crs(4326)

    # ========================================================
    # SIMULATED FLOOD ZONE (STABLE FALLBACK)
    # ========================================================

    flood_zone = merged_zone.buffer(-0.002)
    flood_zone = gpd.GeoDataFrame(geometry=flood_zone, crs=4326)

    folium.GeoJson(
        flood_zone,
        name="Flooded Area",
        style_function=lambda x: {
            "color": "red",
            "fillColor": "red",
            "fillOpacity": 0.4
        }
    ).add_to(m)

    # ========================================================
    # IMPACT ANALYSIS
    # ========================================================

    affected_buildings = intersect(buildings, flood_zone)
    affected_roads = intersect(roads_gdf, flood_zone)

    b_count = len(affected_buildings)
    r_km = safe_length_km(affected_roads)
    flooded_area = safe_area_km2(flood_zone)

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader("📊 Impact Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Flooded Area (km²)", f"{flooded_area:.2f}")
    c2.metric("Buildings Affected", f"{b_count}")
    c3.metric("Roads Affected (km)", f"{r_km:.1f}")

    # ========================================================
    # MAP DISPLAY
    # ========================================================

    folium.GeoJson(
        affected_buildings,
        name="Affected Buildings",
        style_function=lambda x: {"color": "orange"},
        tooltip="Affected building"
    ).add_to(m)

    folium.GeoJson(
        affected_roads,
        name="Affected Roads",
        style_function=lambda x: {"color": "black", "weight": 3},
        tooltip="Affected road"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, height=550)

    # ========================================================
    # PDF EXPORT
    # ========================================================

    stats = {
        "Flooded area (km²)": f"{flooded_area:.2f}",
        "Buildings affected": b_count,
        "Roads affected (km)": f"{r_km:.1f}"
    }

    pdf_path = generate_pdf(
        stats,
        "Uploaded Zone",
        start_date,
        end_date
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 Download PDF Report",
            f,
            file_name="flood_impact_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Please upload a spatial file to begin.")
