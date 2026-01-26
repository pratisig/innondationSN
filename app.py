# ======================================================
# Flood Infra Tracker Sénégal – V1 (VERSION STREAMLIT CLOUD SAFE)
# ❌ geopandas supprimé
# ✅ OSMnx + Shapely + Folium uniquement
# ======================================================

import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from datetime import datetime
import requests

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Flood Infra Tracker Sénégal",
    layout="wide",
    page_icon="🌊"
)

st.title("🌊 Flood Infra Tracker – Sénégal")
st.caption("Suivi institutionnel des infrastructures exposées aux inondations")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Paramètres d'analyse")

place = st.sidebar.text_input(
    "Zone d'intérêt",
    value="Senegal",
    help="Pays, région ou département (ex: Dakar, Kaolack)"
)

start_date = st.sidebar.date_input("Date début", datetime(2024, 8, 1))
end_date = st.sidebar.date_input("Date fin", datetime(2024, 8, 10))

rain_threshold = st.sidebar.slider(
    "Seuil pluie cumulée (mm)",
    min_value=20,
    max_value=150,
    value=80
)

run = st.sidebar.button("Lancer l'analyse")

# ======================================================
# FONCTIONS
# ======================================================

def get_osm_objects(place):
    tags = {
        "highway": True,
        "bridge": True,
        "amenity": ["school", "hospital", "clinic"]
    }
    gdf = ox.geometries_from_place(place, tags)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def classify(row):
    if row.get("bridge"):
        return "Pont"
    if row.get("amenity") == "school":
        return "École"
    if row.get("amenity") in ["hospital", "clinic"]:
        return "Centre de santé"
    if row.get("highway"):
        return "Route"
    return "Autre"


def get_nasa_rain(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    values = data["properties"]["parameter"]["PRECTOTCORR"]
    return sum(values.values())

# ======================================================
# MAIN
# ======================================================

if run:
    with st.spinner("Chargement OSM..."):
        gdf = get_osm_objects(place)

    gdf["type"] = gdf.apply(classify, axis=1)

    st.success(f"{len(gdf)} objets chargés depuis OSM")

    centroid = gdf.geometry.unary_union.centroid

    rain = get_nasa_rain(
        centroid.y,
        centroid.x,
        start_date,
        end_date
    )

    st.metric("Pluie cumulée (mm)", round(rain, 1))

    if rain >= rain_threshold:
        exposure = "ÉLEVÉE"
        color = "red"
    elif rain >= rain_threshold * 0.6:
        exposure = "MODÉRÉE"
        color = "orange"
    else:
        exposure = "FAIBLE"
        color = "green"

    st.markdown(f"### Exposition estimée : **{exposure}**")

    # ==================================================
    # MAP
    # ==================================================
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Point":
            folium.CircleMarker(
                location=[geom.y, geom.x],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=f"Type : {row['type']}<br>Exposition : {exposure}"
            ).add_to(m)

    st_folium(m, height=600, width=1200)

    # ==================================================
    # INDICATEURS
    # ==================================================
    st.subheader("Indicateurs clés")

    col1, col2, col3 = st.columns(3)
    col1.metric("Routes", (gdf.type == "Route").sum())
    col2.metric("Ponts", (gdf.type == "Pont").sum())
    col3.metric("Infrastructures critiques", gdf.type.isin(["École", "Centre de santé"]).sum())

    # ==================================================
    # SIGNALEMENT
    # ==================================================
    st.subheader("Signalement terrain")

    with st.form("signalement"):
        infra = st.selectbox("Type", ["Route", "Pont", "École", "Centre de santé"])
        statut = st.selectbox("Statut", ["Fonctionnel", "Impacté", "Coupé"])
        commentaire = st.text_area("Commentaire")
        submit = st.form_submit_button("Enregistrer")

        if submit:
            st.success("Signalement enregistré (V1 local)")

st.caption("Prototype institutionnel – OSM + NASA POWER")
