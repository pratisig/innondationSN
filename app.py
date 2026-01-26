# ============================================================
# FLOOD ANALYSIS & EMERGENCY PLANNING APP
# West Africa – Sentinel / CHIRPS / WorldPop / OSMnx / FAO GAUL
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
import pandas as pd
from shapely.geometry import mapping
from shapely.ops import unary_union
from pyproj import Geod
import datetime
from fpdf import FPDF
import base64
import osmnx as ox

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Analyse d'Impact Inondations – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Analyse d'Impact Inondations & Planification d'Urgence")
st.caption("Sentinel-1 | CHIRPS | WorldPop | OSMnx | FAO GAUL (Admin 1–3)")

# ------------------------------------------------------------
# INIT GEE
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(
        key["client_email"],
        key_data=json.dumps(key)
    )
    ee.Initialize(credentials)
    return True

init_gee()

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def get_true_area_km2(geom):
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geom)[0])
    return area / 1e6

def safe_get_info(obj):
    try:
        return obj.getInfo()
    except Exception:
        return None

# ------------------------------------------------------------
# PDF REPORT
# ------------------------------------------------------------
def create_pdf_report(df, country, d1, d2, stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 15)
    pdf.cell(190, 10, f"Rapport Inondation – {country}", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(190, 8, f"Période : {d1} au {d2}", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "Résumé global", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(190, 7, f"Surface inondée : {stats['area']:.2f} km²", ln=True)
    pdf.cell(190, 7, f"Population exposée : {stats['pop']:,}", ln=True)
    pdf.cell(190, 7, f"Bâtiments impactés : {stats['buildings']:,}", ln=True)
    pdf.cell(190, 7, f"Routes affectées : {stats['roads']:.1f} segments", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 10)
    headers = ["Zone", "Inondé km²", "Pop exposée", "Bât.", "Santé", "Éduc.", "Routes"]
    for h in headers:
        pdf.cell(27, 7, h, border=1)
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    for _, r in df.iterrows():
        pdf.cell(27, 7, str(r["Zone"])[:15], border=1)
        pdf.cell(27, 7, f"{r['Inondé (km2)']:.2f}", border=1)
        pdf.cell(27, 7, f"{r['Pop. Exposée']:,}", border=1)
        pdf.cell(27, 7, str(r["Bâtiments"]), border=1)
        pdf.cell(27, 7, str(r["Santé"]), border=1)
        pdf.cell(27, 7, str(r["Éducation"]), border=1)
        pdf.cell(27, 7, str(r["Segments Route"]), border=1)
        pdf.ln()

    return pdf.output(dest="S").encode("latin-1")

# ------------------------------------------------------------
# DATASETS GAUL
# ------------------------------------------------------------
GAUL_L1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
GAUL_L2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
C0, C1, C2 = "ADM0_NAME", "ADM1_NAME", "ADM2_NAME"

# ------------------------------------------------------------
# SIDEBAR – ADMIN SELECTION
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Sélection Administrative")
country = st.sidebar.selectbox("Pays", ["Senegal", "Mali", "Mauritania", "Gambia", "Guinea"])

a1_fc = GAUL_L1.filter(ee.Filter.eq(C0, country))
a1_list = safe_get_info(a1_fc.aggregate_array(C1).distinct().sort())
sel_a1 = st.sidebar.multiselect("Admin 1", a1_list or [])

if not sel_a1:
    st.stop()

a2_fc = GAUL_L2.filter(ee.Filter.eq(C0, country)).filter(ee.Filter.inList(C1, sel_a1))
a2_list = safe_get_info(a2_fc.aggregate_array(C2).distinct().sort())
sel_a2 = st.sidebar.multiselect("Admin 2", a2_list or [])

if sel_a2:
    final_fc = a2_fc.filter(ee.Filter.inList(C2, sel_a2))
    label_col = C2
else:
    final_fc = a1_fc.filter(ee.Filter.inList(C1, sel_a1))
    label_col = C1

gdf = gpd.GeoDataFrame.from_features(safe_get_info(final_fc), crs="EPSG:4326")
merged_poly = unary_union(gdf.geometry)
geom_ee = ee.Geometry(mapping(merged_poly))

# ------------------------------------------------------------
# TEMPORAL CONFIG
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période")
start_date = st.sidebar.date_input("Début", datetime.date(2024, 7, 1))
end_date = st.sidebar.date_input("Fin", datetime.date(2024, 10, 31))

# ------------------------------------------------------------
# FLOOD & RAIN ENGINE
# ------------------------------------------------------------
@st.cache_data
def get_flood_and_rain(aoi_json, d1, d2):
    aoi = ee.Geometry(aoi_json)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(aoi).filterDate(d1, d2) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .select("VV")

    ref = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(aoi).filterDate("2024-01-01", "2024-03-31") \
        .median()

    flood = s1.median().subtract(ref).lt(-3)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))

    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(aoi).filterDate(d1, d2).sum().rename("precip")

    return flood.updateMask(slope.lt(5)).selfMask(), rain

# ------------------------------------------------------------
# OSMNX INFRASTRUCTURE IMPACT
# ------------------------------------------------------------
def analyze_infrastructure_impact_osmnx(flood_img, merged_poly):
    flood_vec = flood_img.reduceToVectors(
        geometry=ee.Geometry(mapping(merged_poly)),
        scale=100,
        maxPixels=1e9
    ).getInfo()

    if not flood_vec or not flood_vec["features"]:
        return None

    flood_gdf = gpd.GeoDataFrame.from_features(flood_vec, crs="EPSG:4326")
    flood_union = unary_union(flood_gdf.geometry)

    tags_build = {"building": True, "amenity": True}
    tags_roads = {"highway": True}

    buildings = ox.geometries_from_polygon(merged_poly, tags_build)
    roads = ox.geometries_from_polygon(merged_poly, tags_roads)

    buildings = buildings.to_crs("EPSG:4326")
    roads = roads.to_crs("EPSG:4326")

    buildings_aff = buildings[buildings.intersects(flood_union)]
    roads_aff = roads[roads.intersects(flood_union)]

    health = buildings_aff[buildings_aff["amenity"].isin(
        ["hospital", "clinic", "doctors", "pharmacy"]
    )]
    edu = buildings_aff[buildings_aff["amenity"].isin(
        ["school", "college", "university", "kindergarten"]
    )]

    return buildings_aff, health, edu, roads_aff

# ------------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------------
flood, rain = get_flood_and_rain(
    geom_ee.getInfo(),
    str(start_date),
    str(end_date)
)

pop_img = ee.ImageCollection("WorldPop/GP/100m/pop") \
    .filterBounds(geom_ee).mean().select(0)

osm_data = analyze_infrastructure_impact_osmnx(flood, merged_poly)

rows = []

for idx, row in gdf.iterrows():
    geom = ee.Geometry(mapping(row.geometry))

    stats = safe_get_info(
        ee.Image.cat([
            flood.multiply(ee.Image.pixelArea()).rename("f_area"),
            pop_img.updateMask(flood).rename("p_exp")
        ]).reduceRegion(ee.Reducer.sum(), geom, 250)
    )

    f_km2 = (stats.get("f_area", 0) or 0) / 1e6
    p_exp = int(stats.get("p_exp", 0) or 0)

    if osm_data:
        b, h, e, r = osm_data
        geom_shp = row.geometry
        b_count = b[b.intersects(geom_shp)].shape[0]
        h_count = h[h.intersects(geom_shp)].shape[0]
        e_count = e[e.intersects(geom_shp)].shape[0]
        r_count = r[r.intersects(geom_shp)].shape[0]
    else:
        b_count = h_count = e_count = r_count = 0

    rows.append({
        "Zone": row[label_col],
        "Inondé (km2)": round(f_km2, 2),
        "Pop. Exposée": p_exp,
        "Bâtiments": b_count,
        "Santé": h_count,
        "Éducation": e_count,
        "Segments Route": r_count,
        "orig_id": idx
    })

df_res = pd.DataFrame(rows)

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------
m = folium.Map(
    location=[merged_poly.centroid.y, merged_poly.centroid.x],
    zoom_start=8,
    tiles="CartoDB positron"
)

mid = flood.getMapId({"palette": ["#00D4FF"]})
folium.TileLayer(
    tiles=mid["tile_fetcher"].url_format,
    name="Zones inondées",
    overlay=True
).add_to(m)

for _, r in df_res.iterrows():
    geom = gdf.iloc[int(r["orig_id"])].geometry
    html = f"""
    <b>{r['Zone']}</b><br>
    Inondé : {r['Inondé (km2)']} km²<br>
    Pop exposée : {r['Pop. Exposée']:,}<br>
    Bâtiments : {r['Bâtiments']}<br>
    Santé : {r['Santé']}<br>
    Éducation : {r['Éducation']}<br>
    Routes : {r['Segments Route']}
    """
    folium.GeoJson(
        geom,
        popup=folium.Popup(html, max_width=250),
        style_function=lambda x: {
            "fillColor": "#E74C3C",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.15
        }
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=550)

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
st.markdown("### 📊 Indicateurs Clés")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pop exposée", f"{df_res['Pop. Exposée'].sum():,}")
c2.metric("Bâtiments", f"{df_res['Bâtiments'].sum():,}")
c3.metric("🏥 Santé", f"{df_res['Santé'].sum():,}")
c4.metric("🎓 Éducation", f"{df_res['Éducation'].sum():,}")
c5.metric("🛣️ Routes", f"{df_res['Segments Route'].sum():,}")

# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------
pdf = create_pdf_report(
    df_res,
    country,
    start_date,
    end_date,
    {
        "area": df_res["Inondé (km2)"].sum(),
        "pop": df_res["Pop. Exposée"].sum(),
        "buildings": df_res["Bâtiments"].sum(),
        "roads": df_res["Segments Route"].sum()
    }
)

st.sidebar.download_button(
    "📄 Télécharger rapport PDF",
    pdf,
    file_name="rapport_inondation.pdf"
)

st.dataframe(df_res.drop(columns=["orig_id"]), use_container_width=True)
