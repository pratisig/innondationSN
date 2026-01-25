#======================================================

#Flood Infra Tracker Sénégal – V1

#Application institutionnelle Streamlit

#Données : OpenStreetMap + NASA POWER

#Auteur : Prototype GIS institutionnel

#======================================================

import streamlit as st 
import geopandas as gpd 
import pandas as pd 
import osmnx as ox 
import folium 
from streamlit_folium import st_folium 
from datetime import datetime 
import requests

#======================================================

#CONFIG STREAMLIT

#======================================================

st.set_page_config( page_title="Flood Infra Tracker Sénégal", layout="wide", page_icon="🌊" )

st.title("🌊 Flood Infra Tracker – Sénégal") st.caption("Suivi institutionnel des infrastructures exposées aux inondations")

#======================================================

#SIDEBAR – PARAMÈTRES

#======================================================

st.sidebar.header("Paramètres d'analyse")

region = st.sidebar.text_input( "Zone d'intérêt", value="Senegal", help="Pays, région ou département (ex: Dakar, Kaolack)" )

start_date = st.sidebar.date_input("Date début", datetime(2024, 8, 1)) end_date = st.sidebar.date_input("Date fin", datetime(2024, 8, 10))

rain_threshold = st.sidebar.slider( "Seuil pluie cumulée (mm)", min_value=20, max_value=150, value=80 )

load_data = st.sidebar.button("Lancer l'analyse")

#======================================================

#FONCTIONS DONNÉES

#======================================================

def get_osm_infrastructure(place): tags = { "highway": True, "bridge": True, "amenity": ["school", "hospital", "clinic"] } gdf = ox.geometries_from_place(place, tags) gdf = gdf.reset_index() gdf = gdf[gdf.geometry.notnull()] gdf = gdf.to_crs(epsg=4326)

def classify(row):
    if row.get("highway"):
        return "Route"
    if row.get("bridge"):
        return "Pont"
    if row.get("amenity") in ["school"]:
        return "École"
    if row.get("amenity") in ["hospital", "clinic"]:
        return "Centre de santé"
    return "Autre"

gdf["type"] = gdf.apply(classify, axis=1)
return gdf[["type", "geometry"]]

def get_nasa_power_rain(lat, lon, start, end): 
    url = "https://power.larc.nasa.gov/api/temporal/daily/point" 
    params = { "parameters": "PRECTOTCORR", "community": "AG", "longitude": lon, "latitude": lat, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON" } 
    r = requests.get(url, params=params) 
    data = r.json() 
    values = data["properties"]["parameter"]["PRECTOTCORR"] 
    return sum(values.values())

#======================================================

#TRAITEMENT PRINCIPAL

#======================================================

if load_data: with st.spinner("Chargement des infrastructures OSM..."): gdf_infra = get_osm_infrastructure(region)

st.success(f"{len(gdf_infra)} infrastructures chargées depuis OSM")

# Calcul pluie moyenne sur la zone (centre)
centroid = gdf_infra.unary_union.centroid
rain_cum = get_nasa_power_rain(
    centroid.y,
    centroid.x,
    start_date,
    end_date
)

st.metric(
    "Pluie cumulée sur la période (mm)",
    round(rain_cum, 1)
)

# Exposition simplifiée
if rain_cum >= rain_threshold:
    exposure = "ÉLEVÉE"
    color = "red"
elif rain_cum >= rain_threshold * 0.6:
    exposure = "MODÉRÉE"
    color = "orange"
else:
    exposure = "FAIBLE"
    color = "green"

st.markdown(f"### Niveau d'exposition estimé : **{exposure}**")

# ==================================================
# CARTE
# ==================================================
st.subheader("Carte des infrastructures exposées")

m = folium.Map(
    location=[centroid.y, centroid.x],
    zoom_start=7,
    tiles="OpenStreetMap"
)

for _, row in gdf_infra.iterrows():
    if row.geometry.geom_type == "Point":
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
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
col1.metric("Routes", (gdf_infra.type == "Route").sum())
col2.metric("Ponts", (gdf_infra.type == "Pont").sum())
col3.metric("Infrastructures critiques", gdf_infra.type.isin(["École", "Centre de santé"]).sum())

# ==================================================
# SIGNALMENT TERRAIN
# ==================================================
st.subheader("Signalement terrain")

with st.form("signalement"):
    infra_type = st.selectbox("Type d'infrastructure", ["Route", "Pont", "École", "Centre de santé"])
    statut = st.selectbox("Statut", ["Fonctionnel", "Impacté", "Coupé"])
    commentaire = st.text_area("Commentaire")
    submitted = st.form_submit_button("Enregistrer")

    if submitted:
        st.success("Signalement enregistré (local – V1)")

#======================================================

#FOOTER

#======================================================

st.caption("Prototype institutionnel – Données ouvertes OSM & NASA POWER")
