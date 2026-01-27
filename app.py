# ============================================================
# FLOOD IMPACT DECISION TOOL – VERSION OSMNX FIXED
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from shapely.ops import unary_union
from streamlit_folium import st_folium
import osmnx as ox
from datetime import datetime
from fpdf import FPDF
import tempfile

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Flood Impact Decision Tool",
    page_icon="🌊",
    layout="wide"
)

# ============================================================
# UTILS
# ============================================================

def km2(gdf):
    return gdf.to_crs(3857).area.sum() / 1e6

def km_len(gdf):
    return gdf.to_crs(3857).length.sum() / 1000

def intersect(a, b):
    return gpd.overlay(a, b, how="intersection")

# ============================================================
# PDF REPORT
# ============================================================

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Flood Impact Assessment Report", ln=True)
        self.ln(5)

def make_pdf(stats, start, end):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 11)

    pdf.cell(0, 8, f"Period: {start} → {end}", ln=True)
    pdf.ln(5)

    for k, v in stats.items():
        pdf.cell(0, 8, f"- {k}: {v}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ============================================================
# UI
# ============================================================

st.title("🌊 Flood Impact Decision Support Tool")
st.caption("Decision-oriented quantification of flood impacts")

uploaded = st.file_uploader(
    "Upload AOI (GeoJSON / SHP / KML)",
    type=["geojson", "shp", "kml"]
)

start_date = st.date_input("Start date", datetime(2024, 1, 1))
end_date = st.date_input("End date", datetime.today())

if uploaded:
    aoi = gpd.read_file(uploaded).to_crs(4326)
    aoi = gpd.GeoDataFrame(geometry=[aoi.unary_union], crs=4326)

    st.success("AOI loaded")

    # ========================================================
    # MAP INIT
    # ========================================================

    center = aoi.geometry.centroid.iloc[0]
    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=9,
        tiles="CartoDB positron"
    )

    folium.GeoJson(
        aoi,
        name="Study Area",
        style_function=lambda x: {"color": "blue", "weight": 2}
    ).add_to(m)

    # ========================================================
    # OSM DATA (FIXED)
    # ========================================================

    with st.spinner("Downloading OpenStreetMap infrastructures…"):
        tags = {
            "building": True,
            "highway": True
        }

        osm_features = ox.features_from_polygon(
            aoi.geometry.iloc[0],
            tags=tags
        )

    buildings = osm_features[osm_features["building"].notna()].copy()
    roads = osm_features[osm_features["highway"].notna()].copy()

    buildings = gpd.GeoDataFrame(buildings, geometry="geometry", crs=4326)
    roads = gpd.GeoDataFrame(roads, geometry="geometry", crs=4326)

    # ========================================================
    # FLOOD ZONE (SAFE FALLBACK)
    # ========================================================

    flood = aoi.buffer(-0.002)
    flood = gpd.GeoDataFrame(geometry=flood, crs=4326)

    folium.GeoJson(
        flood,
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

    affected_buildings = intersect(buildings, flood)
    affected_roads = intersect(roads, flood)

    b_count = len(affected_buildings)
    r_km = km_len(affected_roads)
    flooded_km2 = km2(flood)

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader("📊 Impact Indicators")

    c1, c2, c3 = st.columns(3)
    c1.metric("Flooded area (km²)", f"{flooded_km2:.2f}")
    c2.metric("Buildings affected", b_count)
    c3.metric("Roads affected (km)", f"{r_km:.1f}")

    # ========================================================
    # MAP LAYERS
    # ========================================================

    folium.GeoJson(
        affected_buildings,
        name="Affected Buildings",
        style_function=lambda x: {"color": "orange"},
        tooltip="Building impacted"
    ).add_to(m)

    folium.GeoJson(
        affected_roads,
        name="Affected Roads",
        style_function=lambda x: {"color": "black", "weight": 3},
        tooltip="Road impacted"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, height=550)

    # ========================================================
    # PDF EXPORT
    # ========================================================

    stats = {
        "Flooded area (km²)": f"{flooded_km2:.2f}",
        "Buildings affected": b_count,
        "Roads affected (km)": f"{r_km:.1f}"
    }

    pdf = make_pdf(stats, start_date, end_date)

    with open(pdf, "rb") as f:
        st.download_button(
            "📄 Download PDF report",
            f,
            file_name="flood_impact_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Upload a spatial file to start the analysis.")
