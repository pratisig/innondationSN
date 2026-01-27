import os
import io
import json
import math
import tempfile
from datetime import datetime
import ee
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

import folium
from streamlit_folium import st_folium

import osmnx as ox
from pyproj import CRS

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# =========================
# 1. CONFIG STREAMLIT
# =========================

st.set_page_config(
    page_title="Analyse Inondations Afrique de l’Ouest",
    layout="wide"
)

st.title("Plateforme d’analyse des inondations – Afrique de l’Ouest")

# ------------------------------------------------------------
# INIT GEE
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("Secret 'GEE_SERVICE_ACCOUNT' manquant dans Streamlit.")
        st.stop()
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Erreur d'initialisation GEE : {e}")
        return False

init_gee()

# =========================
# 3. FONCTIONS UTILITAIRES
# =========================

@st.cache_data
def load_admin_boundaries():
    """
    Charger une base de limites administratives (ex: GADM / GAUL / HDX).
    Ici : placeholder. Tu peux connecter ton propre fichier GeoPackage ou GeoJSON.
    Retourne un GeoDataFrame en EPSG:4326 avec colonnes :
    ['ADM0_NAME','ADM1_NAME','ADM2_NAME','ADM3_NAME','geometry'].
    """
    # Exemple : charger un fichier local (à adapter)
    # gdf = gpd.read_file("data/admin_westafrica.gpkg")
    # gdf = gdf.to_crs(epsg=4326)
    # return gdf

    # Placeholder minimal vide (l'app restera fonctionnelle avec upload custom)
    gdf = gpd.GeoDataFrame(
        columns=["ADM0_NAME", "ADM1_NAME", "ADM2_NAME", "ADM3_NAME", "geometry"],
        geometry="geometry",
        crs="EPSG:4326"
    )
    return gdf


def dissolve_and_simplify(gdf, tolerance=0.0005):
    """
    Dissoudre les géométries en une seule puis simplifier légèrement.
    """
    gdf = gdf.to_crs(epsg=4326)
    geom = gdf.unary_union
    if isinstance(geom, (MultiPolygon, Polygon)):
        geom_simpl = geom.simplify(tolerance, preserve_topology=True)
    else:
        geom_simpl = geom
    out = gpd.GeoDataFrame(geometry=[geom_simpl], crs="EPSG:4326")
    return out


def compute_geodetic_area_km2(geom: Polygon):
    """
    Aire réelle (km²) sur ellispoïde WGS84.
    Utilise pyproj.Geod.
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lon, lat = geom.exterior.coords.xy
    area, _ = geod.polygon_area_perimeter(lon, lat)
    return abs(area) / 1e6  # m² → km²


def guess_utm_epsg_from_geom(gdf: gpd.GeoDataFrame):
    """
    Devine le code EPSG UTM le plus approprié à partir du centroïde.
    """
    centroid = gdf.to_crs(epsg=4326).geometry.unary_union.centroid
    lon = centroid.x
    lat = centroid.y
    zone = int((lon + 180) / 6) + 1
    south = lat < 0
    epsg_code = 32700 + zone if south else 32600 + zone
    return epsg_code


# =========================
# 4. FONCTIONS GEE
# =========================

def get_s1_flood_mask(aoi_ee, start_ref, end_ref, start_flood, end_flood,
                      orbit_pass="DESCENDING",
                      difference_threshold=1.25,
                      slope_threshold=5,
                      permanent_water_prob=90):
    """
    Détection d'inondation Sentinel-1 VV (méthode différence, inspirée de tutos SAR).
    aoi_ee : ee.Geometry
    dates : chaînes YYYY-MM-DD
    Retourne : dict {'flooded': ee.Image, 's1_ref': ee.Image, 's1_flood': ee.Image}
    """
    # Sentinel-1 collection
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi_ee)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.eq("resolution_meters", 10)))

    # Image de référence "sèche"
    s1_ref = (s1.filterDate(start_ref, end_ref)
                .select("VV")
                .median()
                .clip(aoi_ee))

    # Image "crise"
    s1_flood = (s1.filterDate(start_flood, end_flood)
                  .select("VV")
                  .median()
                  .clip(aoi_ee))

    # Convertir en dB
    def to_db(img):
        return ee.Image(10).multiply(img.log10())

    s1_ref_db = to_db(s1_ref)
    s1_flood_db = to_db(s1_flood)

    # Différence (inondation: forte baisse de rétrodiffusion)
    diff = s1_ref_db.subtract(s1_flood_db)

    # Seuil sur la différence (ajustable)
    flooded_raw = diff.gt(difference_threshold)

    # Masque pente
    try:
        dem = ee.Image("WWF/HydroSHEDS/03VFDEM")
    except Exception:
        dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Algorithms.Terrain(dem).select("slope")
    mask_slope = slope.lt(slope_threshold)

    # Masque eau permanente JRC GSW
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occ = gsw.select("occurrence")
    permanent_water = occ.gte(permanent_water_prob)
    mask_perm = permanent_water.Not()

    flooded = (flooded_raw
               .updateMask(mask_slope)
               .updateMask(mask_perm)
               .selfMask())

    # Nettoyage petits pixels
    flooded = flooded.updateMask(flooded.connectedPixelCount(8).gte(5))

    return {
        "flooded": flooded,
        "s1_ref": s1_ref_db,
        "s1_flood": s1_flood_db
    }


def get_worldpop_population(aoi_ee, year=2020):
    """
    Récupère l'image WorldPop total population pour l'année donnée.
    Utilise le catalogue communautaire WorldPop 2015-2030 100m.
    """
    # Collection WorldPop (ex: 'WorldPop/GP/100m/pop' ou community catalog)
    # Ici on utilise le dataset global WorldPop publié sous CC BY 4.0.
    wp = (ee.ImageCollection("WorldPop/GP/100m/pop")
          .filter(ee.Filter.eq("year", year))
          .mosaic()
          .clip(aoi_ee))
    return wp.select("population")


def aggregate_indicators(aoi_ee, flooded_img, worldpop_img, scale=30):
    """
    Calcule :
    - surface totale
    - surface inondée
    - population totale
    - population exposée
    via reduceRegion (single AOI).
    """
    # Surface totale (km²) à partir d'une image constante
    area_img = ee.Image.pixelArea().divide(1e6)  # km²
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

    # Population totale / exposée
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
        "surface_totale_km2": total_area,
        "surface_inondee_km2": flooded_area,
        "pop_totale": total_pop,
        "pop_exposee": exposed_pop
    }


def export_flood_mask_to_geotiff(flooded_img, aoi_ee, scale=10):
    """
    Export du masque d'inondation en GeoTIFF dans un fichier temporaire.
    Utilise ee.Image.getDownloadURL puis écriture locale.
    """
    params = {
        "scale": scale,
        "crs": "EPSG:4326",
        "region": aoi_ee,
        "fileFormat": "GeoTIFF"
    }
    url = flooded_img.toByte().getDownloadURL(params)
    import requests
    r = requests.get(url)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name


# =========================
# 5. FONCTIONS OSM / OSMNX
# =========================

@st.cache_data
def download_osm_layer(aoi_gdf: gpd.GeoDataFrame,
                       tags: dict):
    """
    Télécharge des entités OSM (bâtiments, routes, POI) pour l'emprise de aoi_gdf.
    tags : dict OSMnx (ex: {'building': True} ou {'highway': True}).
    """
    aoi_bounds = aoi_gdf.to_crs(epsg=4326).total_bounds
    north, south, east, west = aoi_bounds[3], aoi_bounds[1], aoi_bounds[2], aoi_bounds[0]

    # OSMnx geocode_to_gdf / features_from_bbox
    try:
        gdf = ox.geometries_from_bbox(north, south, east, west, tags)
        if gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def compute_osm_impacts(aoi_gdf: gpd.GeoDataFrame,
                        flood_mask_tif: str):
    """
    Calcule les impacts OSM :
    - bâtiments affectés
    - infrastructures santé / éducation affectées
    - km de routes affectées
    en intersectant les géométries OSM avec le raster d'inondation.
    """
    import rasterio
    from rasterio.features import geometry_mask

    # Charger le raster de masque inondation
    with rasterio.open(flood_mask_tif) as src:
        flood_data = src.read(1)
        flood_transform = src.transform
        flood_crs = src.crs

    # Créer un GeoDataFrame de la zone inondée comme polygone (vectorisation simple)
    # On considère toutes les valeurs > 0 comme inondées.
    flooded_shapes = []
    for geom, val in rasterio.features.shapes(flood_data, transform=flood_transform):
        if val > 0:
            flooded_shapes.append(shape(geom))

    if not flooded_shapes:
        flood_poly = None
    else:
        flood_poly = unary_union(flooded_shapes)

    if flood_poly is None or flood_poly.is_empty:
        # Pas d'inondation détectée
        return {
            "batiments_affectes": 0,
            "sante_affectees": 0,
            "education_affectees": 0,
            "routes_affectees_km": 0.0
        }

    flood_gdf = gpd.GeoDataFrame(geometry=[flood_poly], crs=flood_crs)
    flood_gdf = flood_gdf.to_crs(epsg=4326)

    # Télécharger OSM par type
    bldg = download_osm_layer(aoi_gdf, {"building": True})
    road = download_osm_layer(aoi_gdf, {"highway": True})
    health = download_osm_layer(aoi_gdf, {"amenity": ["hospital", "clinic", "healthcare"]})
    edu = download_osm_layer(aoi_gdf, {"amenity": ["school", "college", "university"]})

    # Reprojeter en UTM pour calcul des longueurs / distances
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
        # On ne garde que les lignes
        road_lines = road[road.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        inter_road = gpd.overlay(road_lines, flood_utm, how="intersection")
        inter_road["length_m"] = inter_road.geometry.length
        results["routes_affectees_km"] = inter_road["length_m"].sum() / 1000.0

    return results


# =========================
# 6. GENERATION PDF
# =========================

def generate_pdf_report(aoi_name: str,
                        indicators: dict,
                        period_ref: str,
                        period_flood: str,
                        data_sources: str,
                        warning_text: str,
                        map_png: bytes = None):
    """
    Génère un rapport PDF en mémoire.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Titre
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm, "Rapport d'analyse des inondations")

    # Métadonnées
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 3 * cm, f"Zone d'étude : {aoi_name}")
    c.drawString(2 * cm, height - 3.5 * cm, f"Période de référence : {period_ref}")
    c.drawString(2 * cm, height - 4 * cm, f"Période d'événement : {period_flood}")
    c.drawString(2 * cm, height - 4.5 * cm, f"Date de génération : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    y = height - 6 * cm

    # Carte (si disponible)
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

    # Indicateurs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Indicateurs clés")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    for key, val in indicators.items():
        c.drawString(2 * cm, y, f"- {key} : {val}")
        y -= 0.5 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    # Sources
    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Sources de données")
    y -= 0.7 * cm
    c.setFont("Helvetica", 9)
    for line in data_sources.split("\n"):
        c.drawString(2 * cm, y, line)
        y -= 0.4 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    # Avertissement
    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Avertissement méthodologique")
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
# 7. UI – PANNEAU LATÉRAL
# =========================

st.sidebar.header("Paramètres d’analyse")

admin_gdf = load_admin_boundaries()

mode_zone = st.sidebar.radio(
    "Mode de sélection de la zone",
    ["Zone administrative", "Zone personnalisée (upload)"]
)

uploaded_geom = None
aoi_name = "Zone personnalisée"

if mode_zone == "Zone personnalisée (upload)":
    file = st.sidebar.file_uploader(
        "Uploader GeoJSON, SHP (ZIP) ou KML",
        type=["geojson", "json", "zip", "kml"]
    )
    if file is not None:
        suffix = os.path.splitext(file.name)[1].lower()
        try:
            if suffix in [".geojson", ".json"]:
                gdf_u = gpd.read_file(file)
            elif suffix == ".kml":
                gdf_u = gpd.read_file(file, driver="KML")
            elif suffix == ".zip":
                tmp_dir = tempfile.mkdtemp()
                tmp_zip = os.path.join(tmp_dir, "upload.zip")
                with open(tmp_zip, "wb") as f:
                    f.write(file.getvalue())
                import zipfile
                with zipfile.ZipFile(tmp_zip, "r") as z:
                    z.extractall(tmp_dir)
                # chercher un shp
                shp_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".shp")]
                if not shp_files:
                    st.error("Aucun fichier .shp dans le ZIP.")
                    gdf_u = None
                else:
                    gdf_u = gpd.read_file(shp_files[0])
            else:
                gdf_u = None

            if gdf_u is not None and not gdf_u.empty:
                uploaded_geom = dissolve_and_simplify(gdf_u)
                aoi_name = file.name
        except Exception as e:
            st.sidebar.error("Erreur lors de la lecture du fichier géospatial.")
            st.sidebar.exception(e)
else:
    if admin_gdf.empty:
        st.sidebar.warning("Aucune base administrative n’est chargée. Utilisez le mode upload.")
    else:
        countries = sorted(admin_gdf["ADM0_NAME"].dropna().unique().tolist())
        country = st.sidebar.selectbox("Pays", options=countries)
        subset0 = admin_gdf[admin_gdf["ADM0_NAME"] == country]

        admins1 = sorted(subset0["ADM1_NAME"].dropna().unique().tolist())
        admin1 = st.sidebar.selectbox("Région (Admin1)", options=["(Tous)"] + admins1)

        subset1 = subset0 if admin1 == "(Tous)" else subset0[subset0["ADM1_NAME"] == admin1]

        admins2 = sorted(subset1["ADM2_NAME"].dropna().unique().tolist())
        admin2 = st.sidebar.selectbox("Département (Admin2)", options=["(Tous)"] + admins2)

        subset2 = subset1 if admin2 == "(Tous)" else subset1[subset1["ADM2_NAME"] == admin2]

        admins3 = sorted(subset2["ADM3_NAME"].dropna().unique().tolist())
        admin3 = st.sidebar.selectbox("Commune (Admin3)", options=["(Tous)"] + admins3)

        subset3 = subset2 if admin3 == "(Tous)" else subset2[subset2["ADM3_NAME"] == admin3]

        if not subset3.empty:
            uploaded_geom = dissolve_and_simplify(subset3)
            aoi_name = f"{country} / {admin1} / {admin2} / {admin3}"

# Dates
st.sidebar.subheader("Périodes Sentinel-1")
col_date1, col_date2 = st.sidebar.columns(2)
ref_start = col_date1.date_input("Réf. début", value=datetime(2023, 1, 1))
ref_end = col_date2.date_input("Réf. fin", value=datetime(2023, 2, 1))

col_date3, col_date4 = st.sidebar.columns(2)
flood_start = col_date3.date_input("Crise début", value=datetime(2023, 8, 1))
flood_end = col_date4.date_input("Crise fin", value=datetime(2023, 8, 15))

wp_year = st.sidebar.number_input("Année WorldPop", min_value=2015, max_value=2030, value=2020, step=1)

st.sidebar.subheader("Paramètres SAR avancés")
diff_threshold = st.sidebar.slider("Seuil différence VV (dB)", 0.5, 3.0, 1.25, 0.05)
slope_thresh = st.sidebar.slider("Pente max (°)", 1, 10, 5, 1)
perm_water_prob = st.sidebar.slider("Probabilité eau permanente (%)", 50, 100, 90, 5)

run_button = st.sidebar.button("Lancer l’analyse")

# =========================
# 8. TRAITEMENT PRINCIPAL
# =========================

if run_button:
    if not gee_available:
        st.error("Google Earth Engine n'est pas disponible. Analyse impossible.")
    elif uploaded_geom is None:
        st.error("Veuillez sélectionner ou uploader une zone d'étude.")
    else:
        with st.spinner("Traitement en cours (Sentinel‑1, WorldPop, OSM)..."):
            try:
                # Zone AOI
                aoi_geom = uploaded_geom.to_crs(epsg=4326).geometry.unary_union
                aoi_ee = ee.Geometry(mapping(aoi_geom))

                # Flood mask Sentinel-1
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

                # WorldPop
                wp_img = get_worldpop_population(aoi_ee, year=wp_year)

                # Indicateurs de base
                ind = aggregate_indicators(aoi_ee, flooded_img, wp_img, scale=30)
                surf_tot = ind["surface_totale_km2"]
                surf_inond = ind["surface_inondee_km2"]
                pop_tot = ind["pop_totale"]
                pop_exp = ind["pop_exposee"]
                pct_inond = (surf_inond / surf_tot * 100) if surf_tot > 0 else 0
                pct_pop_exp = (pop_exp / pop_tot * 100) if pop_tot > 0 else 0

                # Export flood mask en GeoTIFF
                flood_tif = export_flood_mask_to_geotiff(flooded_img, aoi_ee, scale=10)

                # Impacts OSM
                osm_impacts = compute_osm_impacts(uploaded_geom, flood_tif)

                # =========================
                # KPIs
                # =========================
                st.subheader("Indicateurs clés")

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
                # TABLEAU INDICATEURS
                # =========================
                st.subheader("Tableau des indicateurs (zone courante)")

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

                st.dataframe(df_ind)

                # =========================
                # EXPORT CSV
                # =========================
                csv_bytes = df_ind.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Télécharger les indicateurs (CSV)",
                    data=csv_bytes,
                    file_name="indicateurs_inondation.csv",
                    mime="text/csv"
                )

                # =========================
                # CARTE INTERACTIVE
                # =========================
                st.subheader("Carte interactive")

                # Créer une image RGB simple pour visualiser le flood mask
                flooded_vis = flooded_img.visualize(
                    min=0, max=1, palette=["000000", "0000FF"]
                )

                url_png = flooded_vis.getThumbURL({
                    "region": aoi_ee,
                    "dimensions": 1024,
                    "format": "png"
                })

                # Base map
                aoi_bounds = uploaded_geom.to_crs(epsg=4326).total_bounds
                center_lat = (aoi_bounds[1] + aoi_bounds[3]) / 2
                center_lon = (aoi_bounds[0] + aoi_bounds[2]) / 2

                m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="cartodbpositron")

                # AOI
                folium.GeoJson(
                    data=json.loads(uploaded_geom.to_json()),
                    name="Zone d'étude",
                    style_function=lambda x: {
                        "fillColor": "#00000000",
                        "color": "#FF8800",
                        "weight": 2
                    },
                    tooltip=aoi_name
                ).add_to(m)

                # Flood overlay (PNG tile via ImageOverlay)
                folium.raster_layers.ImageOverlay(
                    name="Zone inondée (S1)",
                    image=url_png,
                    bounds=[[aoi_bounds[1], aoi_bounds[0]], [aoi_bounds[3], aoi_bounds[2]]],
                    opacity=0.6,
                    interactive=True,
                    cross_origin=False
                ).add_to(m)

                # Légende simple
                legend_html = """
                <div style="
                    position: fixed;
                    bottom: 50px;
                    left: 50px;
                    z-index:9999;
                    background-color: white;
                    padding: 10px;
                    border:2px solid grey;
                    ">
                    <b>Légende</b><br>
                    <i style="background: #0000FF; width: 10px; height: 10px; float: left; margin-right: 5px; opacity:0.7;"></i>
                    Zone inondée (SAR)<br>
                    <i style="border: 2px solid #FF8800; width: 10px; height: 10px; float: left; margin-right: 5px;"></i>
                    Zone d'étude
                </div>
                """
                m.get_root().html.add_child(folium.Element(legend_html))

                folium.LayerControl().add_to(m)

                map_obj = st_folium(m, width=900, height=600)

                # Sauvegarde de la carte en PNG pour rapport PDF
                # (simplifié : on utilise le thumbnail du flood mask)
                import requests
                r_png = requests.get(url_png)
                map_png_bytes = r_png.content if r_png.status_code == 200 else None

                # =========================
                # GRAPHES SIMPLES
                # =========================
                st.subheader("Visualisations (barres / donuts)")

                import plotly.express as px

                # Barres surface
                df_surf = pd.DataFrame({
                    "type": ["Surface inondée", "Surface non inondée"],
                    "valeur": [surf_inond, max(surf_tot - surf_inond, 0)]
                })
                fig_surf = px.bar(df_surf, x="type", y="valeur", title="Répartition des surfaces (km²)")
                st.plotly_chart(fig_surf, use_container_width=True)

                # Donut population
                df_pop = pd.DataFrame({
                    "type": ["Exposée", "Non exposée"],
                    "valeur": [pop_exp, max(pop_tot - pop_exp, 0)]
                })
                fig_pop = px.pie(df_pop, values="valeur", names="type",
                                 title="Population exposée aux inondations",
                                 hole=0.5)
                st.plotly_chart(fig_pop, use_container_width=True)

                # =========================
                # PDF
                # =========================
                st.subheader("Génération rapport PDF")

                data_sources = (
                    "Données satellitaires : Sentinel-1 GRD (Copernicus, ESA).\n"
                    "Modèle de terrain : HydroSHEDS / SRTM.\n"
                    "Eau permanente : JRC Global Surface Water.\n"
                    "Population : WorldPop 100 m (CC BY 4.0).\n"
                    "Infrastructures : OpenStreetMap via Overpass API / OSMnx.\n"
                )

                warning_text = (
                    "Ce rapport fournit une estimation rapide des zones inondées à partir de données "
                    "radar Sentinel-1 et d'autres sources ouvertes. Les résultats peuvent être affectés "
                    "par la couverture nuageuse résiduelle, la qualité du modèle de terrain, la "
                    "configuration du capteur radar, ainsi que des erreurs de classification.\n\n"
                    "Les estimations de population exposée et d'infrastructures affectées sont basées "
                    "sur des bases de données globales pouvant être incomplètes ou obsolètes localement. "
                    "Ce produit ne remplace pas des évaluations de terrain ou des données officielles, "
                    "mais fournit un appui décisionnel pour la priorisation et le ciblage humanitaire."
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
                        "Infrastructures santé affectées": f"{osm_impacts['sante_affectees']:,}",
                        "Infrastructures éducation affectées": f"{osm_impacts['education_affectees']:,}",
                        "Routes affectées (km)": f"{osm_impacts['routes_affectees_km']:.1f}"
                    },
                    period_ref=f"{ref_start} → {ref_end}",
                    period_flood=f"{flood_start} → {flood_end}",
                    data_sources=data_sources,
                    warning_text=warning_text,
                    map_png=map_png_bytes
                )

                st.download_button(
                    label="Télécharger le rapport PDF",
                    data=pdf_buffer,
                    file_name="rapport_inondations.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error("Une erreur est survenue pendant le traitement.")
                st.exception(e)
else:
    st.info("Sélectionnez une zone d'étude, définissez les périodes, puis cliquez sur « Lancer l’analyse ».")
