# ======================================================
# Flood Infra Tracker Sénégal – V1 (ULTRA STREAMLIT CLOUD SAFE)
# ❌ geopandas supprimé
# ❌ osmnx supprimé
# ✅ Overpass API + Folium uniquement
# ======================================================

import streamlit as st
import folium
from streamlit_folium import st_folium
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
st.caption("Suivi institutionnel des infrastructures exposées aux inondations (OSM + pluie)")

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
# FONCTIONS DONNÉES
# ======================================================

def get_osm_points(place):
    """
    Télécharge routes (simplifiées), ponts, écoles, centres de santé
    via Overpass API (sans dépendances lourdes)
    """
    query = f"""
    [out:json][timeout:60];
    area[name="{place}"]->.searchArea;
    (
      node["amenity"="school"](area.searchArea);
      node["amenity"="hospital"](area.searchArea);
      node["amenity"="clinic"](area.searchArea);
      node["bridge"](area.searchArea);
    );
    out center;
    """

    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query
    )
    data = r.json()
    return data["elements"]


def classify_osm(el):
    tags = el.get("tags", {})
    if tags.get("amenity") == "school":
        return "École"
    if tags.get("amenity") in ["hospital", "clinic"]:
        return "Centre de santé"
    if "bridge" in tags:
        return "Pont"
    return "Infrastructure"


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
    values = r.json()["properties"]["parameter"]["PRECTOTCORR"]
    return sum(values.values())

# ======================================================
# MAIN
# ======================================================

if run:
    with st.spinner("Téléchargement des infrastructures depuis OSM..."):
        elements = get_osm_points(place)

    if not elements:
        st.error("Aucune donnée OSM trouvée pour cette zone")
        st.stop()

    # Centroid approximatif
    lats = [el.get("lat") for el in elements if "lat" in el]
    lons = [el.get("lon") for el in elements if "lon" in el]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    rain = get_nasa_rain(center_lat, center_lon, start_date, end_date)

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
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    stats = {"École": 0, "Centre de santé": 0, "Pont": 0}

    for el in elements:
        if "lat" in el and "lon" in el:
            infra_type = classify_osm(el)
            if infra_type in stats:
                stats[infra_type] += 1

            folium.CircleMarker(
                location=[el["lat"], el["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=f"Type : {infra_type}<br>Exposition : {exposure}"
            ).add_to(m)

    st_folium(m, height=600, width=1200)

    # ==================================================
    # INDICATEURS
    # ==================================================
    st.subheader("Indicateurs clés")

    col1, col2, col3 = st.columns(3)
    col1.metric("Écoles", stats["École"])
    col2.metric("Centres de santé", stats["Centre de santé"])
    col3.metric("Ponts", stats["Pont"])

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

st.caption("Prototype institutionnel – OSM (Overpass) + NASA POWER")
