# ============================================================
# FLOOD IMPACT DECISION TOOL – STABLE ADMIN VERSION
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from shapely.ops import unary_union
from streamlit_folium import st_folium
import osmnx as ox
from datetime import datetime
import tempfile
from fpdf import FPDF

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Flood Impact Decision Tool",
    page_icon="🌊",
    layout="wide"
)

# ============================================================
# GADM LOADER
# ============================================================

@st.cache_data(ttl=86400)
def load_gadm(country, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{country}.gpkg"
    gdf = gpd.read_file(url, layer=f"ADM_ADM_{level}")
    return gdf.to_crs(4326)

# ============================================================
# GEOMETRY UTILS
# ============================================================

def km2(gdf):
    return gdf.to_crs(3857).area.sum() / 1e6

def km_len(gdf):
    return gdf.to_crs(3857).length.sum() / 1000

def only_polygons(gdf):
    return gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

def only_lines(gdf):
    return gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]

# ============================================================
# PDF
# ============================================================

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Flood Impact Assessment Report", ln=True)
        self.ln(5)

def generate_pdf(stats):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 11)

    for k, v in stats.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ============================================================
# SIDEBAR – ADMIN SELECTION
# ============================================================

st.sidebar.header("🌍 Area selection")

countries = {
    "Senegal": "SEN",
    "Mali": "MLI",
    "Niger": "NER",
    "Burkina Faso": "BFA"
}

country_name = st.sidebar.selectbox("Country", list(countries.keys()))
country = countries[country_name]

level = st.sidebar.selectbox(
    "Administrative level",
    {0: "Country", 1: "Region", 2: "Department", 3: "Commune"}
)

gadm = load_gadm(country, level)

col_name = f"NAME_{level}" if level > 0 else "COUNTRY"
zones = sorted(gadm[col_name].unique())

selected = st.sidebar.multiselect("Select zone(s)", zones)

if selected:
    aoi = gadm[gadm[col_name].isin(selected)]
else:
    aoi = gadm

aoi = gpd.GeoDataFrame(geometry=[aoi.unary_union], crs=4326)

# ============================================================
# RUN ANALYSIS
# ============================================================

if st.sidebar.button("🚀 Run flood impact analysis"):

    with st.spinner("Downloading OSM infrastructures…"):

        tags = {
            "building": True,
            "highway": True
        }

        osm = ox.features_from_polygon(
            aoi.geometry.iloc[0],
            tags=tags
        )

    buildings = only_polygons(osm[osm["building"].notna()])
    roads = only_lines(osm[osm["highway"].notna()])

    buildings = gpd.GeoDataFrame(buildings, geometry="geometry", crs=4326)
    roads = gpd.GeoDataFrame(roads, geometry="geometry", crs=4326)

    # --------------------------------------------------------
    # FLOOD ZONE (placeholder – ready for Sentinel-1)
    # --------------------------------------------------------

    flood = aoi.buffer(-0.01)
    flood = gpd.GeoDataFrame(geometry=flood, crs=4326)

    # --------------------------------------------------------
    # IMPACT ANALYSIS (SAFE)
    # --------------------------------------------------------

    affected_buildings = gpd.overlay(buildings, flood, how="intersection")
    affected_roads = gpd.overlay(roads, flood, how="intersection")

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    flooded_area = km2(flood)
    building_count = len(affected_buildings)
    road_km = km_len(affected_roads)

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader("📊 Decision Indicators")

    c1, c2, c3 = st.columns(3)
    c1.metric("Flooded area (km²)", f"{flooded_area:.2f}")
    c2.metric("Buildings affected", building_count)
    c3.metric("Roads affected (km)", f"{road_km:.1f}")

    # ========================================================
    # MAP
    # ========================================================

    center = aoi.geometry.centroid.iloc[0]
    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=8,
        tiles="CartoDB positron"
    )

    folium.GeoJson(aoi, name="Admin boundary").add_to(m)
    folium.GeoJson(flood, name="Flooded area",
        style_function=lambda x: {
            "color": "red",
            "fillOpacity": 0.4
        }).add_to(m)

    folium.GeoJson(affected_buildings, name="Affected buildings",
        style_function=lambda x: {"color": "orange"}).add_to(m)

    folium.GeoJson(affected_roads, name="Affected roads",
        style_function=lambda x: {"color": "black", "weight": 3}).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, height=550)

    # ========================================================
    # PDF
    # ========================================================

    pdf = generate_pdf({
        "Country": country_name,
        "Flooded area (km²)": f"{flooded_area:.2f}",
        "Buildings affected": building_count,
        "Roads affected (km)": f"{road_km:.1f}"
    })

    with open(pdf, "rb") as f:
        st.download_button(
            "📄 Download PDF report",
            f,
            file_name="flood_impact_report.pdf",
            mime="application/pdf"
        )
