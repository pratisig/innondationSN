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
    "Senegal": {"iso3": "SEN"},
    "Mali": {"iso3": "MLI"},
    "Niger": {"iso3": "NER"},
    "Gambia": {"iso3": "GMB"},
    "Mauritania": {"iso3": "MRT"},
    "Burkina Faso": {"iso3": "BFA"},
    "Nigeria": {"iso3": "NGA"},
    "Guinea": {"iso3": "GIN"},
    "Guinea-Bissau": {"iso3": "GNB"},
}

PAYS_LISTE = list(PAYS_CONFIG.keys())

# =========================
# 3. AUTHENTIFICATION GEE
# =========================

@st.cache_resource
def init_gee():
    """Initialiser Google Earth Engine."""
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
# 4. GESTION LIMITES ADMINISTRATIVES (GADM 4.1)
# =========================

@st.cache_data(ttl=3600)
def load_gadm_layer(country_iso3: str, layer: int = 0):
    """
    Charge une couche GADM.
    layer: 0=ADM0, 1=ADM1, 2=ADM2, 3=ADM3, 4=ADM4
    
    Structure GADM GeoPackage:
    - Layer 0: level 0 (pays)
    - Layer 1: level 1 (admin 1)
    - Layer 2: level 2 (admin 2)
    - Layer 3: level 3 (admin 3)
    - Layer 4: level 4 (admin 4)
    """
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{country_iso3.upper()}.gpkg"
    
    try:
        # Charger la couche spécifique
        gdf = gpd.read_file(url, layer=layer, engine="pyogrio")
        
        if gdf.empty:
            return None
        
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    
    except Exception as e:
        st.warning(f"⚠️ Erreur GADM {country_iso3} (layer {layer}) : {str(e)[:50]}")
        return None


def get_admin_names(country_iso3: str, admin_level: int):
    """
    Récupère la liste des noms pour un niveau admin.
    admin_level: 1, 2, 3, ou 4
    """
    gdf = load_gadm_layer(country_iso3, layer=admin_level)
    
    if gdf is None or gdf.empty:
        return []
    
    # Colonnes GADM pour noms: NAME_1, NAME_2, NAME_3, NAME_4
    col_name = f"NAME_{admin_level}"
    
    if col_name not in gdf.columns:
        return []
    
    names = sorted(gdf[col_name].dropna().unique().tolist())
    return [n for n in names if n and isinstance(n, str)]


def filter_gadm_by_names(country_iso3: str, admin_level: int, selected_names: list):
    """
    Filtre GADM par niveau et noms sélectionnés.
    """
    gdf = load_gadm_layer(country_iso3, layer=admin_level)
    
    if gdf is None or gdf.empty:
        return None
    
    col_name = f"NAME_{admin_level}"
    
    if col_name not in gdf.columns:
        return None
    
    filtered = gdf[gdf[col_name].isin(selected_names)]
    
    if filtered.empty:
        return None
    
    return filtered.to_crs(epsg=4326)


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
# 5. FONCTIONS GEE (CORRIGÉES)
# =========================

def get_s1_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood,
                      orbit_pass="DESCENDING",
                      difference_threshold=1.25,
                      slope_threshold=5,
                      permanent_water_prob=90):
    """Détection inondation Sentinel-1 VV (CORRIGÉE)."""
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
    """
    Agrège les indicateurs clés.
    CORRECTION : vérifier que les images ont des bandes avant multiply.
    """
    # Vérifier que les images ont des bandes
    if flooded_img is None or worldpop_img is None:
        st.error("❌ Erreur : images GEE vides ou inexistantes.")
        return None
    
    # Créer image de surface pixel
    area_img = ee.Image.pixelArea().divide(1e6)  # km²
    
    # Surface totale
    total_area_dict = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_ee,
        scale=scale,
        maxPixels=1e12
    )
    
    # Surface inondée
    # CORRECTION : updateMask(flooded_img) avant multiply
    flooded_area_dict = (ee.Image.pixelArea()
                         .divide(1e6)
                         .updateMask(flooded_img)
                         .reduceRegion(
                            reducer=ee.Reducer.sum(),
                            geometry=aoi_ee,
                            scale=scale,
                            maxPixels=1e12
                         ))
    
    # Population totale
    total_pop_dict = worldpop_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_ee,
        scale=100,
        maxPixels=1e12
    )
    
    # Population exposée
    exposed_pop_dict = (worldpop_img
                        .updateMask(flooded_img)
                        .reduceRegion(
                            reducer=ee.Reducer.sum(),
                            geometry=aoi_ee,
                            scale=100,
                            maxPixels=1e12
                        ))
    
    # Extraire valeurs
    total_area = total_area_dict.get("area").getInfo() or 0
    flooded_area = flooded_area_dict.get("area").getInfo() or 0
    total_pop = total_pop_dict.get("population").getInfo() or 0
    exposed_pop = exposed_pop_dict.get("population").getInfo() or 0
    
    return {
        "surface_totale_km2": float(total_area) if total_area else 0,
        "surface_inondee_km2": float(flooded_area) if flooded_area else 0,
        "pop_totale": float(total_pop) if total_pop else 0,
        "pop_exposee": float(exposed_pop) if exposed_pop else 0
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
        st.warning(f"⚠️ Erreur OSM : {str(e)[:50]}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def compute_osm_impacts(aoi_gdf: gpd.GeoDataFrame, flood_mask_tif: str):
    """Calcule impacts OSM."""
    import rasterio
    from rasterio.features import shapes as rasterio_shapes

    try:
        with rasterio.open(flood_mask_tif) as src:
            flood_data = src.read(1)
            flood_transform = src.transform
            flood_crs = src.crs

        flooded_shapes = []
        for geom, val in rasterio_shapes(flood_data, transform=flood_transform):
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
    
    except Exception as e:
        st.warning(f"⚠️ Erreur OSM impacts : {str(e)[:50]}")
        return {
            "batiments_affectes": 0,
            "sante_affectees": 0,
            "education_affectees": 0,
            "routes_affectees_km": 0.0
        }

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
# 8. SIDEBAR - CASCADE ADMINISTRATIVE SIMPLIFIÉE
# =========================

st.sidebar.header("⚙️ Analyse des inondations")

mode_zone = st.sidebar.radio(
    "Comment sélectionner la zone ?",
    ["📍 Sélection administrative", "📁 Fichier personnalisé"]
)

gdf_aoi = None
aoi_name = "Zone personnalisée"

if mode_zone == "📍 Sélection administrative":
    
    # Pays
    country_name = st.sidebar.selectbox("🌍 Choisir un pays", options=PAYS_LISTE)
    country_iso3 = PAYS_CONFIG[country_name]["iso3"]
    
    # Admin 1
    st.sidebar.write("**Étape 1 : Choisir une région**")
    a1_list = get_admin_names(country_iso3, admin_level=1)
    
    if not a1_list:
        st.sidebar.error(f"❌ Pas de données pour {country_name}.")
        st.stop()
    
    sel_a1_list = st.sidebar.multiselect(
        "Région(s)",
        options=a1_list,
        default=[a1_list[0]] if a1_list else []
    )
    
    if not sel_a1_list:
        st.sidebar.info("ℹ️ Sélectionnez au moins une région.")
        st.stop()
    
    # Admin 2
    st.sidebar.write("**Étape 2 : Affiner (optionnel)**")
    
    # Charger Admin 2 pour les Admin 1 sélectionnées
    gdf_a1 = filter_gadm_by_names(country_iso3, admin_level=1, selected_names=sel_a1_list)
    
    if gdf_a1 is not None and not gdf_a1.empty and "NAME_2" in gdf_a1.columns:
        a2_list = sorted(gdf_a1["NAME_2"].dropna().unique().tolist())
        a2_list = [a for a in a2_list if a and isinstance(a, str)]
        
        if a2_list:
            sel_a2_list = st.sidebar.multiselect(
                "Département(s) [optionnel]",
                options=a2_list,
                default=[]
            )
        else:
            sel_a2_list = []
    else:
        sel_a2_list = []
    
    # Construire la sélection finale
    if sel_a2_list:
        final_gdf = filter_gadm_by_names(country_iso3, admin_level=2, selected_names=sel_a2_list)
        aoi_name = f"{country_name} › {', '.join(sel_a1_list[:2])}{'...' if len(sel_a1_list) > 2 else ''} › {', '.join(sel_a2_list[:2])}"
    else:
        final_gdf = gdf_a1
        aoi_name = f"{country_name} › {', '.join(sel_a1_list[:2])}{'...' if len(sel_a1_list) > 2 else ''}"
    
    if final_gdf is not None and not final_gdf.empty:
        gdf_aoi = dissolve_and_simplify(final_gdf)
    else:
        st.sidebar.error("❌ Aucune géométrie trouvée.")
        st.stop()

else:
    # Mode upload
    file = st.sidebar.file_uploader(
        "📤 Uploader un fichier (GeoJSON, SHP ZIP ou KML)",
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
                    st.sidebar.error("❌ Aucun .shp dans le ZIP.")
                    st.stop()
                gdf_aoi = gpd.read_file(shp_files[0])
            
            if gdf_aoi is not None and not gdf_aoi.empty:
                gdf_aoi = dissolve_and_simplify(gdf_aoi)
                aoi_name = file.name
        except Exception as e:
            st.sidebar.error("❌ Erreur de lecture.")
            st.sidebar.exception(e)
            st.stop()

# =========================
# 9. PARAMÈTRES ANALYSE (SIMPLIFIÉS + EXPANDER POUR AVANCÉS)
# =========================

st.sidebar.subheader("📅 Analyse temporelle")
col_date1, col_date2 = st.sidebar.columns(2)
ref_start = col_date1.date_input("Référence début", value=datetime(2023, 1, 1))
ref_end = col_date2.date_input("Référence fin", value=datetime(2023, 2, 1))

col_date3, col_date4 = st.sidebar.columns(2)
flood_start = col_date3.date_input("Crise début", value=datetime(2023, 8, 1))
flood_end = col_date4.date_input("Crise fin", value=datetime(2023, 8, 15))

wp_year = st.sidebar.number_input("Année données population", min_value=2015, max_value=2030, value=2020, step=1)

# ✅ PARAMÈTRES AVANCÉS EN EXPANDER (POUR NON-TECHNIQUE)
with st.sidebar.expander("⚙️ Paramètres avancés (experts uniquement)", expanded=False):
    st.write("Ces paramètres affectent la précision de la détection radar. Valeurs par défaut recommandées.")
    diff_threshold = st.slider(
        "Seuil de sensibilité (dB)",
        min_value=0.5, max_value=3.0, value=1.25, step=0.05,
        help="Plus bas = plus de pixels détectés comme inondés. Défaut : 1.25"
    )
    slope_thresh = st.slider(
        "Pente maximale (°)",
        min_value=1, max_value=10, value=5, step=1,
        help="Exclut les zones en pente. Défaut : 5°"
    )
    perm_water_prob = st.slider(
        "Probabilité eau permanente (%)",
        min_value=50, max_value=100, value=90, step=5,
        help="Exclut l'eau qui est toujours présente. Défaut : 90%"
    )

run_button = st.sidebar.button("▶️ LANCER L'ANALYSE", key="run_btn")

# =========================
# 10. TRAITEMENT PRINCIPAL
# =========================

if run_button:
    if not gee_available:
        st.error("❌ GEE indisponible.")
        st.stop()
    
    if gdf_aoi is None or gdf_aoi.empty:
        st.error("❌ Sélectionnez une zone d'étude.")
        st.stop()
    
    progress_placeholder = st.empty()
    
    with st.spinner("⏳ Traitement en cours..."):
        try:
            aoi_geom = gdf_aoi.to_crs(epsg=4326).geometry.unary_union
            aoi_ee = ee.Geometry(mapping(aoi_geom))
            
            progress_placeholder.info("📡 Récupération données Sentinel-1...")
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
            
            progress_placeholder.info("👥 Récupération données population...")
            wp_img = get_worldpop_population(aoi_ee, year=wp_year)
            
            progress_placeholder.info("📊 Calcul des indicateurs...")
            ind = aggregate_indicators(aoi_ee, flooded_img, wp_img, scale=30)
            
            if ind is None:
                st.error("❌ Erreur lors du calcul des indicateurs.")
                st.stop()
            
            surf_tot = ind["surface_totale_km2"]
            surf_inond = ind["surface_inondee_km2"]
            pop_tot = ind["pop_totale"]
            pop_exp = ind["pop_exposee"]
            pct_inond = (surf_inond / surf_tot * 100) if surf_tot > 0 else 0
            pct_pop_exp = (pop_exp / pop_tot * 100) if pop_tot > 0 else 0
            
            progress_placeholder.info("🗺️ Export du raster d'inondation...")
            flood_tif = export_flood_mask_to_geotiff(flooded_img, aoi_ee, scale=10)
            
            progress_placeholder.info("🏢 Analyse des infrastructures OSM...")
            osm_impacts = compute_osm_impacts(gdf_aoi, flood_tif)
            
            progress_placeholder.success("✅ Analyse complétée !")
            
            # =========================
            # KPIs
            # =========================
            st.subheader("📊 Résultats")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Surface totale (km²)", f"{surf_tot:,.0f}")
            col2.metric("Surface inondée (km²)", f"{surf_inond:,.0f}")
            col3.metric("% inondé", f"{pct_inond:,.1f}%")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Population (hab.)", f"{int(pop_tot):,}")
            col5.metric("Population exposée", f"{int(pop_exp):,}")
            col6.metric("% exposé", f"{pct_pop_exp:,.1f}%")
            
            col7, col8, col9, col10 = st.columns(4)
            col7.metric("Bâtiments affectés", f"{osm_impacts['batiments_affectes']:,}")
            col8.metric("Structures santé", f"{osm_impacts['sante_affectees']:,}")
            col9.metric("Écoles affectées", f"{osm_impacts['education_affectees']:,}")
            col10.metric("Routes affectées", f"{osm_impacts['routes_affectees_km']:.0f} km")
            
            # =========================
            # TABLEAU EXPORT
            # =========================
            st.subheader("📋 Données détaillées")
            
            df_ind = pd.DataFrame([{
                "Zone": aoi_name,
                "Surface totale (km²)": f"{surf_tot:,.1f}",
                "Surface inondée (km²)": f"{surf_inond:,.1f}",
                "% surface inondée": f"{pct_inond:,.1f}",
                "Population totale": f"{int(pop_tot):,}",
                "Population exposée": f"{int(pop_exp):,}",
                "% population exposée": f"{pct_pop_exp:,.1f}",
                "Bâtiments affectés": osm_impacts["batiments_affectes"],
                "Santé affectées": osm_impacts["sante_affectees"],
                "Éducation affectées": osm_impacts["education_affectees"],
                "Routes affectées (km)": f"{osm_impacts['routes_affectees_km']:.1f}"
            }])
            
            st.dataframe(df_ind, use_container_width=True)
            
            csv_bytes = df_ind.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Télécharger en CSV",
                data=csv_bytes,
                file_name="inondations_resultats.csv",
                mime="text/csv"
            )
            
            # =========================
            # CARTE
            # =========================
            st.subheader("🗺️ Cartographie")
            
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
                }
            ).add_to(m)
            
            folium.raster_layers.ImageOverlay(
                name="Zones inondées détectées",
                image=url_png,
                bounds=[[aoi_bounds[1], aoi_bounds[0]], [aoi_bounds[3], aoi_bounds[2]]],
                opacity=0.6
            ).add_to(m)
            
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
                "Catégorie": ["Inondée", "Non inondée"],
                "Superficie (km²)": [surf_inond, max(surf_tot - surf_inond, 0)]
            })
            fig_surf = px.bar(df_surf, x="Catégorie", y="Superficie (km²)",
                            title="Répartition des surfaces")
            st.plotly_chart(fig_surf, use_container_width=True)
            
            df_pop = pd.DataFrame({
                "Statut": ["Exposée", "Non exposée"],
                "Population (hab.)": [pop_exp, max(pop_tot - pop_exp, 0)]
            })
            fig_pop = px.pie(df_pop, values="Population (hab.)", names="Statut",
                           title="Population exposée aux inondations")
            st.plotly_chart(fig_pop, use_container_width=True)
            
            # =========================
            # PDF
            # =========================
            st.subheader("📄 Rapport")
            
            data_sources = (
                "🛰️ Sentinel-1 GRD (Copernicus ESA)\n"
                "🏔️ Modèle de terrain HydroSHEDS/SRTM\n"
                "💧 JRC Global Surface Water\n"
                "👥 WorldPop 100 m (CC BY 4.0)\n"
                "🏢 Infrastructures OpenStreetMap\n"
                "📋 Limites GADM 4.1"
            )
            
            warning_text = (
                "Ce rapport fournit une évaluation rapide des zones inondées basée sur les données "
                "radar Sentinel-1 et d'autres sources ouvertes. Les résultats peuvent être affectés par "
                "la couverture nuageuse, la qualité du modèle de terrain et les caractéristiques du "
                "capteur radar.\n\n"
                "Les estimations de population et d'infrastructures proviennent de bases de données "
                "globales pouvant être incomplètes ou obsolètes localement. Ce produit fournit un appui "
                "décisionnel pour la priorisation humanitaire et ne remplace pas les évaluations de terrain."
            )
            
            pdf_buffer = generate_pdf_report(
                aoi_name=aoi_name,
                indicators={
                    "Surface totale (km²)": f"{surf_tot:,.1f}",
                    "Surface inondée (km²)": f"{surf_inond:,.1f}",
                    "% inondé": f"{pct_inond:,.1f}",
                    "Population totale": f"{int(pop_tot):,}",
                    "Population exposée": f"{int(pop_exp):,}",
                    "% exposé": f"{pct_pop_exp:,.1f}",
                    "Bâtiments affectés": osm_impacts["batiments_affectes"],
                    "Structures santé": osm_impacts["sante_affectees"],
                    "Écoles affectées": osm_impacts["education_affectees"],
                    "Routes affectées (km)": f"{osm_impacts['routes_affectees_km']:.1f}"
                },
                period_ref=f"{ref_start} → {ref_end}",
                period_flood=f"{flood_start} → {flood_end}",
                data_sources=data_sources,
                warning_text=warning_text,
                map_png=map_png_bytes
            )
            
            st.download_button(
                label="📥 Télécharger le rapport PDF",
                data=pdf_buffer,
                file_name="rapport_inondations.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error("❌ Une erreur est survenue.")
            st.exception(e)
else:
    st.info("👈 Utilisez le panneau de gauche pour sélectionner une zone, puis cliquez sur « LANCER L'ANALYSE »")
