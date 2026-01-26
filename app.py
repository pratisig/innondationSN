# ============================================================
# FLOOD IMPACT ASSESSMENT PRO – STABLE STREAMLIT
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import ee
import osmnx as ox

from shapely.ops import unary_union
from datetime import datetime
from pyproj import CRS

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Flood Impact Assessment Pro",
    layout="wide",
    page_icon="🌊"
)

ee.Initialize()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Paramètres")

start_date = st.sidebar.date_input("📅 Début", datetime(2024, 8, 1))
end_date = st.sidebar.date_input("📅 Fin", datetime(2024, 9, 30))

uploaded_file = st.sidebar.file_uploader(
    "📂 Zone d’étude (GeoJSON / SHP / KML)",
    type=["geojson", "shp", "kml"]
)

# ============================================================
# ZONE
# ============================================================

if not uploaded_file:
    st.warning("Veuillez charger une zone.")
    st.stop()

gdf = gpd.read_file(uploaded_file)
gdf = gdf.to_crs(epsg=3857)

if "name" not in gdf.columns:
    gdf["name"] = [f"Zone {i+1}" for i in range(len(gdf))]

# ============================================================
# EARTH ENGINE FUNCTIONS
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

def zonal_area(img, geom):
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

def get_rain(geom, start, end):
    rain = ee.ImageCollection("NASA/POWER/Daily").select("PRECTOT").filterDate(
        str(start), str(end)
    )
    return rain.sum().reduceRegion(
        ee.Reducer.mean(), geom, 5000
    ).getInfo()["PRECTOT"]

# ============================================================
# ANALYSE
# ============================================================

flood_img = get_flood_image(start_date, end_date)

records = []

for _, row in gdf.iterrows():
    geom_ee = ee.Geometry(row.geometry.__geo_interface__)

    total_km2 = row.geometry.area / 1e6

    flood = zonal_area(flood_img, geom_ee)
    flood_km2 = flood["VV"] / 1e6 if flood and "VV" in flood else 0

    flood_pct = (flood_km2 / total_km2) * 100 if total_km2 > 0 else 0

    pop_total = get_population(geom_ee)
    pop_exposed = int(pop_total * flood_pct / 100)

    rain = get_rain(geom_ee, start_date, end_date)

    # OSM
    poly_wgs = gpd.GeoSeries([row.geometry], crs=3857).to_crs(4326).iloc[0]
    osm = ox.geometries_from_polygon(
        poly_wgs,
        tags={"amenity": True, "highway": True, "building": True}
    )

    infra_count = len(osm)

    priority = (
        flood_pct * 0.5 +
        (pop_exposed / max(pop_total, 1)) * 30 +
        infra_count * 0.05
    )

    records.append({
        "Zone": row["name"],
        "Surface totale (km²)": round(total_km2, 2),
        "Surface inondée (km²)": round(flood_km2, 2),
        "% inondée": round(flood_pct, 1),
        "Population totale": int(pop_total),
        "Population exposée": pop_exposed,
        "Pluie cumulée (mm)": round(rain, 1),
        "Infrastructures exposées": infra_count,
        "Indice priorité": round(priority, 1)
    })

df = pd.DataFrame(records)

# ============================================================
# INDICATEURS
# ============================================================

st.subheader("📊 Indicateurs clés")

c1, c2, c3, c4 = st.columns(4)

c1.metric("🌊 Surface inondée (km²)", df["Surface inondée (km²)"].sum())
c2.metric("👥 Population exposée", df["Population exposée"].sum())
c3.metric("🏗️ Infrastructures exposées", df["Infrastructures exposées"].sum())
c4.metric("🧠 Zone prioritaire", df.sort_values("Indice priorité", ascending=False).iloc[0]["Zone"])

# ============================================================
# CARTE
# ============================================================

st.subheader("🗺️ Carte interactive")

center = gdf.to_crs(4326).unary_union.centroid
m = folium.Map(location=[center.y, center.x], zoom_start=8, tiles="CartoDB positron")

# Flood layer
flood_map = flood_img.getMapId({"palette": ["0000FF"]})
folium.TileLayer(
    tiles=flood_map["tile_fetcher"].url_format,
    attr="Sentinel-1 Flood",
    name="Inondation",
    overlay=True
).add_to(m)

# Polygons + popup
for _, row in gdf.iterrows():
    info = df[df.Zone == row["name"]].iloc[0]

    popup = f"""
    <b>{row['name']}</b><br>
    Surface totale: {info['Surface totale (km²)']} km²<br>
    Surface inondée: {info['Surface inondée (km²)']} km²<br>
    % inondée: {info['% inondée']} %<br>
    Population: {info['Population totale']}<br>
    Population exposée: {info['Population exposée']}<br>
    Infrastructures: {info['Infrastructures exposées']}
    """

    folium.GeoJson(
        row.geometry.to_crs(4326),
        popup=popup,
        style_function=lambda x: {"fillOpacity": 0.1, "color": "red"}
    ).add_to(m)

folium.LayerControl().add_to(m)
st.components.v1.html(m._repr_html_(), height=600)

# ============================================================
# TABLE + FILTRES
# ============================================================

st.subheader("📋 Tableau récapitulatif")

min_pct = st.slider("Filtrer % inondée", 0, 100, 0)
st.dataframe(df[df["% inondée"] >= min_pct])

st.download_button(
    "⬇️ Télécharger CSV",
    df.to_csv(index=False),
    "flood_results.csv"
)

# ============================================================
# PDF
# ============================================================

if st.button("📄 Générer rapport PDF"):
    path = "/tmp/rapport_flood.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    story = [Paragraph("Rapport d’analyse des inondations", styles["Title"])]

    for _, r in df.iterrows():
        story.append(Spacer(1, 10))
        story.append(Paragraph(str(r.to_dict()), styles["Normal"]))

    doc.build(story)
    st.download_button("📥 Télécharger PDF", open(path, "rb"), "rapport.pdf")
