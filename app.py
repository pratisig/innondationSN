# ============================================================
# FLOOD IMPACT ASSESSMENT APP – VERSION AVANCÉE
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import ee
import geemap.foliumap as geemap
import osmnx as ox
from shapely.geometry import shape
from shapely.ops import unary_union
from pyproj import CRS
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# INITIALISATION
# ============================================================

st.set_page_config(
    page_title="Flood Impact Assessment Pro",
    layout="wide",
    page_icon="🌊"
)

ee.Initialize()

# ============================================================
# SIDEBAR – PARAMÈTRES
# ============================================================

st.sidebar.title("⚙️ Paramètres d’analyse")

start_date = st.sidebar.date_input("📅 Date début", datetime(2024, 8, 1))
end_date = st.sidebar.date_input("📅 Date fin", datetime(2024, 9, 30))

scenario = st.sidebar.selectbox(
    "📊 Scénario",
    ["Crue actuelle", "Crue décennale", "Crue extrême"]
)

uploaded_file = st.sidebar.file_uploader(
    "📂 Zone d’étude (GeoJSON / SHP / KML)",
    type=["geojson", "shp", "kml"]
)

# ============================================================
# LECTURE ZONE
# ============================================================

if uploaded_file:
    gdf = gpd.read_file(uploaded_file)
    gdf = gdf.to_crs(epsg=3857)  # PROJECTION MÉTRIQUE
    gdf["zone_id"] = gdf.index + 1
    gdf["zone_name"] = gdf.get("name", gdf["zone_id"].apply(lambda x: f"Zone {x}"))

    total_geom = unary_union(gdf.geometry)
    total_area_km2 = total_geom.area / 1e6

    st.success(f"✅ Zone chargée – Surface totale : {total_area_km2:.2f} km²")

# ============================================================
# FONCTIONS GEE
# ============================================================

def get_flood_image(start, end):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(str(start), str(end))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select("VV")
        .mean()
        .lt(-15)
    )

def zonal_sum(img, geom):
    return img.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=30,
        maxPixels=1e13
    ).getInfo()

def get_population(geom):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").mean()
    return pop.reduceRegion(
        ee.Reducer.sum(), geom, 100, maxPixels=1e13
    ).getInfo()["population"]

def get_rainfall(geom, start, end):
    rain = ee.ImageCollection("NASA/POWER/Daily").select("PRECTOT").filterDate(
        str(start), str(end)
    )
    return rain.sum().reduceRegion(
        ee.Reducer.mean(), geom, 5000
    ).getInfo()["PRECTOT"]

def get_ndvi_loss(geom):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(geom)
    before = s2.filterDate("2024-06-01", "2024-07-15").median()
    after = s2.filterDate("2024-08-15", "2024-09-30").median()

    ndvi_before = before.normalizedDifference(["B8", "B4"])
    ndvi_after = after.normalizedDifference(["B8", "B4"])

    loss = ndvi_before.subtract(ndvi_after)
    return loss.reduceRegion(
        ee.Reducer.mean(), geom, 20, maxPixels=1e13
    ).getInfo()

# ============================================================
# ANALYSE
# ============================================================

if uploaded_file:

    flood_img = get_flood_image(start_date, end_date)

    results = []

    for _, row in gdf.iterrows():
        geom_ee = ee.Geometry(row.geometry.__geo_interface__)

        flood_area = zonal_sum(flood_img, geom_ee)
        flood_km2 = flood_area["VV"] / 1e6 if flood_area["VV"] else 0

        total_km2 = row.geometry.area / 1e6
        flood_pct = (flood_km2 / total_km2) * 100 if total_km2 > 0 else 0

        pop_total = get_population(geom_ee)
        pop_exposed = pop_total * (flood_pct / 100)

        rain = get_rainfall(geom_ee, start_date, end_date)
        ndvi_loss = get_ndvi_loss(geom_ee)

        # OSM INFRASTRUCTURES
        poly_wgs = gpd.GeoSeries([row.geometry], crs=3857).to_crs(4326).iloc[0]
        tags = {"amenity": True, "highway": True, "building": True}
        osm = ox.geometries_from_polygon(poly_wgs, tags)
        infra_count = len(osm)

        priority_index = (
            flood_pct * 0.4 +
            (pop_exposed / max(pop_total, 1)) * 30 +
            infra_count * 0.05
        )

        results.append({
            "Zone": row.zone_name,
            "Surface totale (km²)": total_km2,
            "Surface inondée (km²)": flood_km2,
            "% inondée": flood_pct,
            "Population totale": int(pop_total),
            "Population exposée": int(pop_exposed),
            "Pluie cumulée (mm)": rain,
            "Infrastructures exposées": infra_count,
            "NDVI perte": ndvi_loss,
            "Indice priorité": priority_index
        })

    df = pd.DataFrame(results)

# ============================================================
# INDICATEURS AVANT TABLE
# ============================================================

st.subheader("📊 Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🌊 Surface inondée totale (km²)", df["Surface inondée (km²)"].sum())
col2.metric("👥 Population exposée", df["Population exposée"].sum())
col3.metric("🛣️ Infrastructures exposées", df["Infrastructures exposées"].sum())
col4.metric("🧠 Zone prioritaire", df.sort_values("Indice priorité", ascending=False).iloc[0]["Zone"])

# ============================================================
# CARTE
# ============================================================

st.subheader("🗺️ Carte interactive")

m = geemap.Map()
m.add_basemap("SATELLITE")
m.addLayer(flood_img.updateMask(flood_img), {"palette": ["blue"]}, "Inondation")

for _, row in gdf.iterrows():
    popup = f"""
    <b>{row.zone_name}</b><br>
    Surface: {df.loc[df.Zone == row.zone_name, "Surface totale (km²)"].values[0]:.2f} km²<br>
    Inondée: {df.loc[df.Zone == row.zone_name, "Surface inondée (km²)"].values[0]:.2f} km²<br>
    Population: {df.loc[df.Zone == row.zone_name, "Population totale"].values[0]}<br>
    Population exposée: {df.loc[df.Zone == row.zone_name, "Population exposée"].values[0]}
    """
    folium.GeoJson(row.geometry, popup=popup).add_to(m)

m.to_streamlit(height=600)

# ============================================================
# TABLE RÉCAPITULATIVE + FILTRES
# ============================================================

st.subheader("📋 Tableau récapitulatif")

min_pct = st.slider("Filtrer % inondée", 0, 100, 0)
df_filtered = df[df["% inondée"] >= min_pct]

st.dataframe(df_filtered)

st.download_button(
    "⬇️ Télécharger CSV",
    df_filtered.to_csv(index=False),
    "flood_analysis.csv",
    "text/csv"
)

# ============================================================
# RAPPORT PDF
# ============================================================

if st.button("📄 Générer rapport PDF"):
    pdf_path = "/tmp/rapport_inondation.pdf"
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    story = [Paragraph("Rapport d’analyse des inondations", styles["Title"])]

    for _, r in df.iterrows():
        story.append(Spacer(1, 12))
        story.append(Paragraph(str(r.to_dict()), styles["Normal"]))

    doc.build(story)
    st.download_button("📥 Télécharger PDF", open(pdf_path, "rb"), "rapport.pdf")
