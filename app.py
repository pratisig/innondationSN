import os
import io
import json
import math
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

import folium
from streamlit_folium import st_folium

import osmnx as ox
from pyproj import CRS, Geod

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib import colors

import ee

# =========================
# 1. CONFIG STREAMLIT
# =========================

st.set_page_config(
    page_title="Analyse Inondations Afrique de l'Ouest",
    layout="wide"
)

st.title("🌊 Plateforme d'analyse des inondations – Afrique de l'Ouest")
st.markdown("**Détection d'inondations • Impact humanitaire • Aide à la décision**")

# =========================
# 2. DONNÉES PAYS (Tes 8 pays)
# =========================

PAYS_CONFIG = {
    "Senegal": {"iso3": "SEN", "code": 221},
    "Mali": {"iso3": "MLI", "code": 223},
    "Niger": {"iso3": "NER", "code": 562},
    "Gambia": {"iso3": "GMB", "code": 270},
    "Mauritania": {"iso3": "MRT", "code": 478},
    "Burkina Faso": {"iso3": "BFA", "code": 854},
    "Nigeria": {"iso3": "NGA", "code": 566},
    "Guinea": {"iso3": "GIN", "code": 324},
    "Guinea-Bissau": {"iso3": "GNB", "code": 624},
}

PAYS_LISTE = list(PAYS_CONFIG.keys())

# =========================
# 3. AUTHENTIFICATION GEE
# =========================

@st.cache_resource
def init_gee():
    """Initialiser Google Earth Engine avec credentials."""
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("❌ Secret 'GEE_SERVICE_ACCOUNT' manquant.")
        return False
    
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key["client_email"],
            key_data=json.dumps(key)
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"❌ Erreur GEE : {e}")
        return False

gee_available = init_gee()

# =========================
# 4. GESTION LIMITES ADMINISTRATIVES (GADM 4.1 + OSM fallback)
# =========================

@st.cache_data(ttl=3600)
def load_gadm_admin_boundaries(country_name: str, admin_level: int = None):
    """
    Charge les limites GADM pour un pays donné.
    admin_level: None (tous), ou 1, 2, 3, 4
    
    GADM disponible: https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/
    Colonnes: GID_0, NAME_0, GID_1, NAME_1, GID_2, NAME_2, GID_3, NAME_3, GID_4, NAME_4
    """
    if country_name not in PAYS_CONFIG:
        st.error(f"❌ Pays '{country_name}' non supporté.")
        return None
    
    iso3 = PAYS_CONFIG[country_name]["iso3"]
    
    # URL GADM v4.1 GeoPackage
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso3}.gpkg"
    
    try:
        # Charger le GeoPackage (contient toutes les couches admin)
        # Structure: adm0, adm1, adm2, adm3, adm4
        layers = gpd.read_file(url, layer=0, engine="pyogrio")  # ou driver='GPKG'
        
        if layers.empty:
            st.warning(f"⚠️ Pas de données GADM pour {country_name}.")
            return None
        
        layers = layers.to_crs(epsg=4326)
        
        return layers
    
    except Exception as e:
        st.warning(f"⚠️ Erreur GADM {country_name} : {e}. Fallback OSM.")
        return load_osm_admin_boundaries(country_name, admin_level)


def load_osm_admin_boundaries(country_name: str, admin_level: int = None):
    """
    Fallback : télécharger limites admin depuis OSM (Overpass).
    admin_level en OSM: 4=pays, 6=région, 8=district, 10=commune
    """
    try:
        # Query Overpass pour récupérer les boundary admin
        tags = {
            "boundary": "administrative",
            "name": country_name
        }
        if admin_level:
            tags["admin_level"] = str(admin_level)
        
        gdf = ox.features_from_place(country_name, tags)
        
        if gdf.empty:
            return None
        
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    
    except Exception as e:
        st.warning(f"⚠️ Erreur OSM pour {country_name} : {e}")
        return None


@st.cache_data(ttl=3600)
def get_admin_list(country_name: str, admin_level: int):
    """
    Récupère la liste des noms pour un niveau admin donné.
    admin_level: 1, 2, 3, ou 4
    """
    gdf = load_gadm_admin_boundaries(country_name)
    
    if gdf is None or gdf.empty:
        return []
    
    # Les colonnes GADM sont: NAME_1, NAME_2, NAME_3, NAME_4
    col_name = f"NAME_{admin_level}"
    
    if col_name not in gdf.columns:
        return []
    
    names = sorted(gdf[col_name].dropna().unique().tolist())
    return [n for n in names if n and isinstance(n, str)]


def filter_gadm_by_path(country_name: str, path_dict: dict):
    """
    Filtre GADM par chemin hiérarchique.
    path_dict : {"ADM1": "région", "ADM2": "département", ...}
    """
    gdf = load_gadm_admin_boundaries(country_name)
    
    if gdf is None or gdf.empty:
        return None
    
    for level, name in path_dict.items():
        col = f"NAME_{level[-1]}"  # ex: "ADM1" -> "NAME_1"
        if col in gdf.columns and name:
            gdf = gdf[gdf[col] == name]
    
    if gdf.empty:
        return None
    
    return gdf.to_crs(epsg=4326)


def dissolve_and_simplify(gdf, tolerance=0.0005):
    """Dissoudre et simplifier géométries."""
    gdf = gdf.to_crs(epsg=4326)
    geom = unary_union(gdf.geometry)
    if isinstance(geom, (MultiPolygon, Polygon)):
        geom_simpl = geom.simplify(tolerance, preserve_topology=True)
    else:
        geom_simpl = geom
    out = gpd.GeoDataFrame(geometry=[geom_simpl], crs="EPSG:4326")
    return out


def compute_geodetic_area_km2(geom):
    """Aire réelle WGS84."""
    geod = Geod(ellps="WGS84")
    if hasattr(geom, 'exterior'):
        lon, lat = geom.exterior.coords.xy
    else:
        lon, lat = geom.coords.xy
    area, _ = geod.polygon_area_perimeter(lon, lat)
    return abs(area) / 1e6


def guess_utm_epsg_from_geom(gdf: gpd.GeoDataFrame):
    """Guess UTM zone."""
    centroid = gdf.to_crs(epsg=4326).geometry.unary_union.centroid
    lon = centroid.x
    lat = centroid.y
    zone = int((lon + 180) / 6) + 1
    south = lat < 0
    epsg_code = 32700 + zone if south else 32600 + zone
    return epsg_code

# =========================
# 5. FONCTIONS GEE
# =========================

def get_s1_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood,
                      orbit_pass="DESCENDING",
                      difference_threshold=1.25,
                      slope_threshold=5,
                      permanent_water_prob=90):
    """Détection inondation Sentinel-1 VV."""
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi_ee)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.eq("resolution_meters", 10)))

    s1_ref = (s1.filterDate(start_ref, end_ref)
                .select("VV")
                .median()
                .clip(aoi_ee))

    s1_flood = (s1.filterDate(start_flood, end_flood)
                  .select("VV")
                  .median()
                  .clip(aoi_ee))

    def to_db(img):
        return ee.Image(10).multiply(img.log10())

    s1_ref_db = to_db(s1_ref)
    s1_flood_db = to_db(s1_flood)
    diff = s1_ref_db.subtract(s1_flood_db)

    flooded_raw = diff.gt(difference_threshold)

    try:
        dem = ee.Image("WWF/HydroSHEDS/03VFDEM")
    except Exception:
        dem = ee.Image("USGS/SRTMGL1_003")
    
    slope = ee.Algorithms.Terrain(dem).select("slope")
    mask_slope = slope.lt(slope_threshold)

    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occ = gsw.select("occurrence")
    permanent_water = occ.gte(permanent_water_prob)
    mask_perm = permanent_water.Not()

    flooded = (flooded_raw
               .updateMask(mask_slope)
               .updateMask(mask_perm)
               .selfMask())

    flooded = flooded.updateMask(flooded.connectedPixelCount(8).gte(5))

    return {"flooded": flooded, "s1_ref": s1_ref_db, "s1_flood": s1_flood_db}


def get_worldpop_population(aoi_ee, year=2020):
    """WorldPop 100m."""
    wp = (ee.ImageCollection("WorldPop/GP/100m/pop")
          .filter(ee.Filter.eq("year", year))
          .mosaic()
          .clip(aoi_ee))
    return wp.select("population")


def aggregate_indicators(aoi_ee, flooded_img, worldpop_img, scale=30):
    """Agrège les indicateurs clés."""
    area_img = ee.Image.pixelArea().divide(1e6)

    total_area_dict = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_ee,
        scale=scale,
        maxPixels=1e12
    )

    flooded_area_dict = area_img.updateMask(flooded_img).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_ee,
        scale=scale,
        maxPixels=1e12
    )

    total_pop_dict = worldpop_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_ee,
        scale=100,
        maxPixels=1e12
    )

    exposed_pop_dict = worldpop_img.updateMask(flooded_img).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_ee,
        scale=100,
        maxPixels=1e12
    )

    total_area = total_area_dict.get("area").getInfo() if total_area_dict.get("area") else 0
    flooded_area = flooded_area_dict.get("area").getInfo() if flooded_area_dict.get("area") else 0
    total_pop = total_pop_dict.get("population").getInfo() if total_pop_dict.get("population") else 0
    exposed_pop = exposed_pop_dict.get("population").getInfo() if exposed_pop_dict.get("population") else 0

    return {
        "surface_totale_km2": float(total_area),
        "surface_inondee_km2": float(flooded_area),
        "pop_totale": float(total_pop),
        "pop_exposee": float(exposed_pop)
    }


def export_flood_mask_to_geotiff(flooded_img, aoi_ee, scale=10):
    """Export flood mask GeoTIFF."""
    params = {
        "scale": scale,
        "crs": "EPSG:4326",
        "region": aoi_ee,
        "fileFormat": "GeoTIFF"
    }
    url = flooded_img.toByte().getDownloadURL(params)
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name

# =========================
# 6. FONCTIONS OSM
# =========================

@st.cache_data(ttl=3600)
def download_osm_layer(aoi_gdf: gpd.GeoDataFrame, tags: dict):
    """Télécharge OSM via OSMnx."""
    aoi_bounds = aoi_gdf.to_crs(epsg=4326).total_bounds
    north, south, east, west = aoi_bounds[3], aoi_bounds[1], aoi_bounds[2], aoi_bounds[0]

    try:
        gdf = ox.geometries_from_bbox(north, south, east, west, tags)
        if gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        st.warning(f"⚠️ Erreur OSM : {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def compute_osm_impacts(aoi_gdf: gpd.GeoDataFrame, flood_mask_tif: str):
    """Calcule impacts OSM."""
    import rasterio
    from rasterio.features import geometry_mask

    with rasterio.open(flood_mask_tif) as src:
        flood_data = src.read(1)
        flood_transform = src.transform
        flood_crs = src.crs

    flooded_shapes = []
    for geom, val in rasterio.features.shapes(flood_data, transform=flood_transform):
        if val > 0:
            flooded_shapes.append(shape(geom))

    if not flooded_shapes:
        flood_poly = None
    else:
        flood_poly = unary_union(flooded_shapes)

    if flood_poly is None or flood_poly.is_empty:
        return {
            "batiments_affectes": 0,
            "sante_affectees": 0,
            "education_affectees": 0,
            "routes_affectees_km": 0.0
        }

    flood_gdf = gpd.GeoDataFrame(geometry=[flood_poly], crs=flood_crs)
    flood_gdf = flood_gdf.to_crs(epsg=4326)

    bldg = download_osm_layer(aoi_gdf, {"building": True})
    road = download_osm_layer(aoi_gdf, {"highway": True})
    health = download_osm_layer(aoi_gdf, {"amenity": ["hospital", "clinic", "healthcare"]})
    edu = download_osm_layer(aoi_gdf, {"amenity": ["school", "college", "university"]})

    utm_epsg = guess_utm_epsg_from_geom(aoi_gdf)
    flood_utm = flood_gdf.to_crs(epsg=utm_epsg)

    results = {
        "batiments_affectes": 0,
        "sante_affectees": 0,
        "education_affectees": 0,
        "routes_affectees_km": 0.0
    }

    if not bldg.empty:
        bldg = bldg.to_crs(epsg=utm_epsg)
        inter_bldg = gpd.overlay(bldg, flood_utm, how="intersection")
        results["batiments_affectes"] = len(inter_bldg)

    if not health.empty:
        health = health.to_crs(epsg=utm_epsg)
        inter_health = gpd.overlay(health, flood_utm, how="intersection")
        results["sante_affectees"] = len(inter_health)

    if not edu.empty:
        edu = edu.to_crs(epsg=utm_epsg)
        inter_edu = gpd.overlay(edu, flood_utm, how="intersection")
        results["education_affectees"] = len(inter_edu)

    if not road.empty:
        road = road.to_crs(epsg=utm_epsg)
        road_lines = road[road.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        inter_road = gpd.overlay(road_lines, flood_utm, how="intersection")
        inter_road["length_m"] = inter_road.geometry.length
        results["routes_affectees_km"] = inter_road["length_m"].sum() / 1000.0

    return results

# =========================
# 7. GÉNÉRATION PDF
# =========================

def generate_pdf_report(aoi_name: str,
                        indicators: dict,
                        period_ref: str,
                        period_flood: str,
                        data_sources: str,
                        warning_text: str,
                        map_png: bytes = None):
    """Génère PDF."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm, "🌊 Rapport d'analyse des inondations")

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 3 * cm, f"Zone d'étude : {aoi_name}")
    c.drawString(2 * cm, height - 3.5 * cm, f"Référence : {period_ref}")
    c.drawString(2 * cm, height - 4 * cm, f"Événement : {period_flood}")
    c.drawString(2 * cm, height - 4.5 * cm, f"Date : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    y = height - 6 * cm

    if map_png:
        try:
            img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img_tmp.write(map_png)
            img_tmp.flush()
            img_tmp.close()
            c.drawImage(img_tmp.name, 2 * cm, y - 8 * cm, width=12 * cm, height=8 * cm)
            y = y - 9 * cm
        except Exception:
            pass

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "📊 Indicateurs")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    for key, val in indicators.items():
        c.drawString(2 * cm, y, f"• {key} : {val}")
        y -= 0.5 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "📚 Sources")
    y -= 0.7 * cm
    c.setFont("Helvetica", 9)
    for line in data_sources.split("\n"):
        c.drawString(2 * cm, y, line)
        y -= 0.4 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "⚠️ Avertissement")
    y -= 0.7 * cm
    c.setFont("Helvetica", 9)
    for line in warning_text.split("\n"):
        c.drawString(2 * cm, y, line)
        y -= 0.4 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# =========================
# 8. SIDEBAR - CASCADE ADMINISTRATIVE (GADM 4 NIVEAUX)
# =========================

st.sidebar.header("⚙️ Paramètres d'analyse")

mode_zone = st.sidebar.radio(
    "Mode de sélection",
    ["📍 Administrative (GADM)", "📁 Upload personnalisé"]
)

gdf_aoi = None
aoi_name = "Zone personnalisée"

if mode_zone == "📍 Administrative (GADM)":
    
    # Pays
    country_name = st.sidebar.selectbox("🌍 Pays", options=PAYS_LISTE)
    
    # Admin 1
    a1_list = get_admin_list(country_name, admin_level=1)
    if not a1_list:
        st.sidebar.warning(f"⚠️ Pas de données Admin 1 pour {country_name}.")
        st.stop()
    
    sel_a1 = st.sidebar.selectbox("🏘️ Admin 1 (Région)", options=["(Tous)"] + a1_list)
    
    # Admin 2
    if sel_a1 == "(Tous)":
        gdf_a1 = filter_gadm_by_path(country_name, {})
    else:
        gdf_a1 = filter_gadm_by_path(country_name, {"ADM1": sel_a1})
    
    if gdf_a1 is None or gdf_a1.empty:
        st.sidebar.error(f"❌ Pas de données Admin 2.")
        st.stop()
    
    a2_list = sorted(gdf_a1["NAME_2"].dropna().unique().tolist())
    a2_list = [a for a in a2_list if a and isinstance(a, str)]
    
    sel_a2 = st.sidebar.selectbox("📍 Admin 2 (Département)", options=["(Tous)"] + a2_list)
    
    # Admin 3
    if sel_a2 == "(Tous)":
        gdf_a2 = gdf_a1
    else:
        gdf_a2 = gdf_a1[gdf_a1["NAME_2"] == sel_a2]
    
    if gdf_a2 is None or gdf_a2.empty:
        a3_list = []
    else:
        a3_list = sorted(gdf_a2["NAME_3"].dropna().unique().tolist())
        a3_list = [a for a in a3_list if a and isinstance(a, str)]
    
    sel_a3 = st.sidebar.selectbox("🏘️ Admin 3 (Commune)", 
                                   options=["(Tous)"] + a3_list if a3_list else ["(Aucun)"])
    
    # Admin 4
    if sel_a3 == "(Tous)" or sel_a3 == "(Aucun)":
        gdf_a3 = gdf_a2 if gdf_a2 is not None else gdf_a1
    else:
        gdf_a3 = gdf_a2[gdf_a2["NAME_3"] == sel_a3] if gdf_a2 is not None else gpd.GeoDataFrame()
    
    if gdf_a3 is not None and not gdf_a3.empty:
        a4_list = sorted(gdf_a3["NAME_4"].dropna().unique().tolist())
        a4_list = [a for a in a4_list if a and isinstance(a, str)]
    else:
        a4_list = []
    
    sel_a4 = st.sidebar.selectbox("🗺️ Admin 4 (Subdivision)", 
                                   options=["(Tous)"] + a4_list if a4_list else ["(Aucun)"])
    
    # Construire la sélection finale
    if sel_a4 != "(Tous)" and sel_a4 != "(Aucun)" and sel_a4:
        final_gdf = gdf_a3[gdf_a3["NAME_4"] == sel_a4]
        aoi_name = f"{country_name} › {sel_a1} › {sel_a2} › {sel_a3} › {sel_a4}"
    elif sel_a3 != "(Tous)" and sel_a3 != "(Aucun)" and sel_a3:
        final_gdf = gdf_a3
        aoi_name = f"{country_name} › {sel_a1} › {sel_a2} › {sel_a3}"
    elif sel_a2 != "(Tous)" and sel_a2:
        final_gdf = gdf_a2
        aoi_name = f"{country_name} › {sel_a1} › {sel_a2}"
    else:
        final_gdf = gdf_a1
        aoi_name = f"{country_name} › {sel_a1}"
    
    if final_gdf is not None and not final_gdf.empty:
        gdf_aoi = dissolve_and_simplify(final_gdf)

else:
    # Mode upload
    file = st.sidebar.file_uploader(
        "📤 Uploader GeoJSON, SHP (ZIP) ou KML",
        type=["geojson", "json", "zip", "kml"]
    )
    
    if file is not None:
        suffix = os.path.splitext(file.name)[1].lower()
        try:
            if suffix in [".geojson", ".json"]:
                gdf_aoi = gpd.read_file(file)
            elif suffix == ".kml":
                gdf_aoi = gpd.read_file(file, driver="KML")
            elif suffix == ".zip":
                tmp_dir = tempfile.mkdtemp()
                tmp_zip = os.path.join(tmp_dir, "upload.zip")
                with open(tmp_zip, "wb") as f:
                    f.write(file.getvalue())
                import zipfile
                with zipfile.ZipFile(tmp_zip, "r") as z:
                    z.extractall(tmp_dir)
                shp_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".shp")]
                if not shp_files:
                    st.error("❌ Aucun .shp dans le ZIP.")
                    st.stop()
                gdf_aoi = gpd.read_file(shp_files[0])
            
            if gdf_aoi is not None and not gdf_aoi.empty:
                gdf_aoi = dissolve_and_simplify(gdf_aoi)
                aoi_name = file.name
        except Exception as e:
            st.sidebar.error("❌ Erreur de lecture.")
            st.sidebar.exception(e)
            st.stop()

# Paramètres dates
st.sidebar.subheader("📅 Périodes Sentinel-1")
col_date1, col_date2 = st.sidebar.columns(2)
ref_start = col_date1.date_input("Réf. début", value=datetime(2023, 1, 1))
ref_end = col_date2.date_input("Réf. fin", value=datetime(2023, 2, 1))

col_date3, col_date4 = st.sidebar.columns(2)
flood_start = col_date3.date_input("Crise début", value=datetime(2023, 8, 1))
flood_end = col_date4.date_input("Crise fin", value=datetime(2023, 8, 15))

wp_year = st.sidebar.number_input("Année WorldPop", min_value=2015, max_value=2030, value=2020, step=1)

st.sidebar.subheader("🔧 SAR avancé")
diff_threshold = st.sidebar.slider("Seuil VV (dB)", 0.5, 3.0, 1.25, 0.05)
slope_thresh = st.sidebar.slider("Pente max (°)", 1, 10, 5, 1)
perm_water_prob = st.sidebar.slider("Eau permanente (%)", 50, 100, 90, 5)

run_button = st.sidebar.button("▶️ Lancer l'analyse", key="run_btn")

# =========================
# 9. TRAITEMENT PRINCIPAL
# =========================

if run_button:
    if not gee_available:
        st.error("❌ GEE indisponible.")
        st.stop()
    
    if gdf_aoi is None or gdf_aoi.empty:
        st.error("❌ Sélectionnez une zone d'étude.")
        st.stop()
    
    with st.spinner("⏳ Traitement (Sentinel‑1, WorldPop, OSM)..."):
        try:
            aoi_geom = gdf_aoi.to_crs(epsg=4326).geometry.unary_union
            aoi_ee = ee.Geometry(mapping(aoi_geom))
            
            s1_dict = get_s1_flood_mask(
                aoi_ee=aoi_ee,
                start_ref=str(ref_start),
                end_ref=str(ref_end),
                start_flood=str(flood_start),
                end_flood=str(flood_end),
                difference_threshold=diff_threshold,
                slope_threshold=slope_thresh,
                permanent_water_prob=perm_water_prob
            )
            flooded_img = s1_dict["flooded"]
            st.success("✅ Sentinel-1 traité")
            
            wp_img = get_worldpop_population(aoi_ee, year=wp_year)
            st.success("✅ WorldPop chargé")
            
            ind = aggregate_indicators(aoi_ee, flooded_img, wp_img, scale=30)
            surf_tot = ind["surface_totale_km2"]
            surf_inond = ind["surface_inondee_km2"]
            pop_tot = ind["pop_totale"]
            pop_exp = ind["pop_exposee"]
            pct_inond = (surf_inond / surf_tot * 100) if surf_tot > 0 else 0
            pct_pop_exp = (pop_exp / pop_tot * 100) if pop_tot > 0 else 0
            
            st.success("✅ Indicateurs GEE calculés")
            
            flood_tif = export_flood_mask_to_geotiff(flooded_img, aoi_ee, scale=10)
            st.success("✅ Raster exporté")
            
            osm_impacts = compute_osm_impacts(gdf_aoi, flood_tif)
            st.success("✅ Analyses OSM complétées")
            
            # =========================
            # KPIs
            # =========================
            st.subheader("📊 Indicateurs clés")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Surface totale (km²)", f"{surf_tot:,.1f}")
            col2.metric("Surface inondée (km²)", f"{surf_inond:,.1f}")
            col3.metric("% surface inondée", f"{pct_inond:,.1f} %")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Population totale", f"{int(pop_tot):,}")
            col5.metric("Population exposée", f"{int(pop_exp):,}")
            col6.metric("% pop. exposée", f"{pct_pop_exp:,.1f} %")
            
            col7, col8, col9, col10 = st.columns(4)
            col7.metric("Bâtiments affectés", f"{osm_impacts['batiments_affectes']:,}")
            col8.metric("Santé affectées", f"{osm_impacts['sante_affectees']:,}")
            col9.metric("Éducation affectées", f"{osm_impacts['education_affectees']:,}")
            col10.metric("Routes affectées (km)", f"{osm_impacts['routes_affectees_km']:.1f}")
            
            # =========================
            # TABLEAU
            # =========================
            st.subheader("📋 Tableau des indicateurs")
            
            df_ind = pd.DataFrame([{
                "zone": aoi_name,
                "surface_totale_km2": surf_tot,
                "surface_inondee_km2": surf_inond,
                "pct_surface_inondee": pct_inond,
                "population_totale": pop_tot,
                "population_exposee": pop_exp,
                "pct_population_exposee": pct_pop_exp,
                "batiments_affectes": osm_impacts["batiments_affectes"],
                "sante_affectees": osm_impacts["sante_affectees"],
                "education_affectees": osm_impacts["education_affectees"],
                "routes_affectees_km": osm_impacts["routes_affectees_km"]
            }])
            
            st.dataframe(df_ind, use_container_width=True)
            
            # CSV
            csv_bytes = df_ind.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ CSV indicateurs",
                data=csv_bytes,
                file_name="indicateurs_inondation.csv",
                mime="text/csv"
            )
            
            # =========================
            # CARTE
            # =========================
            st.subheader("🗺️ Carte interactive")
            
            flooded_vis = flooded_img.visualize(min=0, max=1, palette=["000000", "0000FF"])
            url_png = flooded_vis.getThumbURL({
                "region": aoi_ee,
                "dimensions": 1024,
                "format": "png"
            })
            
            aoi_bounds = gdf_aoi.to_crs(epsg=4326).total_bounds
            center_lat = (aoi_bounds[1] + aoi_bounds[3]) / 2
            center_lon = (aoi_bounds[0] + aoi_bounds[2]) / 2
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="cartodbpositron")
            
            folium.GeoJson(
                data=json.loads(gdf_aoi.to_json()),
                name="Zone d'étude",
                style_function=lambda x: {
                    "fillColor": "#00000000",
                    "color": "#FF8800",
                    "weight": 2
                },
                tooltip=aoi_name
            ).add_to(m)
            
            folium.raster_layers.ImageOverlay(
                name="Zone inondée (S1)",
                image=url_png,
                bounds=[[aoi_bounds[1], aoi_bounds[0]], [aoi_bounds[3], aoi_bounds[2]]],
                opacity=0.6,
                interactive=True,
                cross_origin=False
            ).add_to(m)
            
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; 
                        background-color: white; padding: 10px; border:2px solid grey;">
                <b>Légende</b><br>
                <i style="background: #0000FF; width: 10px; height: 10px; float: left; 
                           margin-right: 5px; opacity:0.7;"></i>Zone inondée (SAR)<br>
                <i style="border: 2px solid #FF8800; width: 10px; height: 10px; 
                           float: left; margin-right: 5px;"></i>Zone d'étude
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
            folium.LayerControl().add_to(m)
            
            st_folium(m, width=900, height=600)
            
            import requests
            r_png = requests.get(url_png)
            map_png_bytes = r_png.content if r_png.status_code == 200 else None
            
            # =========================
            # GRAPHIQUES
            # =========================
            st.subheader("📈 Visualisations")
            
            import plotly.express as px
            
            df_surf = pd.DataFrame({
                "type": ["Inondée", "Non inondée"],
                "valeur": [surf_inond, max(surf_tot - surf_inond, 0)]
            })
            fig_surf = px.bar(df_surf, x="type", y="valeur", 
                            title="Répartition des surfaces (km²)",
                            labels={"valeur": "Surface (km²)"})
            st.plotly_chart(fig_surf, use_container_width=True)
            
            df_pop = pd.DataFrame({
                "type": ["Exposée", "Non exposée"],
                "valeur": [pop_exp, max(pop_tot - pop_exp, 0)]
            })
            fig_pop = px.pie(df_pop, values="valeur", names="type",
                           title="Population exposée",
                           hole=0.5)
            st.plotly_chart(fig_pop, use_container_width=True)
            
            # =========================
            # PDF
            # =========================
            st.subheader("📄 Rapport PDF")
            
            data_sources = (
                "🛰️ Sentinel-1 GRD (Copernicus, ESA)\n"
                "🏔️ HydroSHEDS / SRTM\n"
                "💧 JRC Global Surface Water\n"
                "👥 WorldPop 100m (CC BY 4.0)\n"
                "🏢 OpenStreetMap (Overpass / OSMnx)\n"
                "📋 GADM 4.1 (limites administratives)"
            )
            
            warning_text = (
                "Ce rapport fournit une estimation rapide des zones inondées à partir de Sentinel-1 "
                "et d'autres sources ouvertes. Les résultats peuvent être affectés par la couverture "
                "nuageuse, la qualité du DEM, la configuration SAR et les erreurs de classification.\n\n"
                "Les estimations de population et infrastructures sont basées sur des bases globales "
                "pouvant être incomplètes ou obsolètes. Ce produit ne remplace pas des évaluations "
                "de terrain mais fournit un appui décisionnel pour la priorisation humanitaire."
            )
            
            pdf_buffer = generate_pdf_report(
                aoi_name=aoi_name,
                indicators={
                    "Surface totale (km²)": f"{surf_tot:,.1f}",
                    "Surface inondée (km²)": f"{surf_inond:,.1f}",
                    "% surface inondée": f"{pct_inond:,.1f}",
                    "Population totale": f"{int(pop_tot):,}",
                    "Population exposée": f"{int(pop_exp):,}",
                    "% population exposée": f"{pct_pop_exp:,.1f}",
                    "Bâtiments affectés": f"{osm_impacts['batiments_affectes']:,}",
                    "Infrastructures santé": f"{osm_impacts['sante_affectees']:,}",
                    "Infrastructures éducation": f"{osm_impacts['education_affectees']:,}",
                    "Routes affectées (km)": f"{osm_impacts['routes_affectees_km']:.1f}"
                },
                period_ref=f"{ref_start} → {ref_end}",
                period_flood=f"{flood_start} → {flood_end}",
                data_sources=data_sources,
                warning_text=warning_text,
                map_png=map_png_bytes
            )
            
            st.download_button(
                label="⬇️ Rapport PDF",
                data=pdf_buffer,
                file_name="rapport_inondations.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error("❌ Une erreur est survenue.")
            st.exception(e)
else:
    st.info("ℹ️ Sélectionnez une zone et cliquez « Lancer l'analyse ».")
