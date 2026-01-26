# ============================================================
# FLOOD ANALYSIS & MAPPING APP — SENEGAL / WEST AFRICA
# Uses real open data via Google Earth Engine (GEE)
# ============================================================

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import tempfile, os, json, zipfile
import pandas as pd
import plotly.express as px
from shapely.geometry import Polygon, MultiPolygon

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Flood Impact Analysis – West Africa",
    layout="wide",
    page_icon="🌊"
)
st.title("🌊 Flood Impact Analysis & Mapping")
st.caption("Satellite-based flood detection using Sentinel-1 SAR and open datasets")

# ------------------------------------------------------------
# INIT GEE
# ------------------------------------------------------------
@st.cache_resource
def init_gee():
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        st.error("Secret 'GEE_SERVICE_ACCOUNT' manquant dans Streamlit.")
        st.stop()
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
    ee.Initialize(credentials)
    return True

init_gee()

# ------------------------------------------------------------
# SHAPELY -> EE GEOMETRY
# ------------------------------------------------------------
def shapely_to_ee(poly):
    geojson = poly.__geo_interface__
    if geojson['type'] == 'Polygon':
        return ee.Geometry.Polygon(geojson['coordinates'])
    elif geojson['type'] == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(geojson['coordinates'])
    else:
        return ee.Geometry(geojson)

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Zone d’étude")
uploaded_file = st.sidebar.file_uploader("Charger une zone (GeoJSON / SHP ZIP / KML)", type=["geojson","kml","zip"])
if not uploaded_file:
    st.info("Veuillez charger une zone géographique.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    file_path = os.path.join(tmpdir, uploaded_file.name)
    with open(file_path,"wb") as f: f.write(uploaded_file.getbuffer())

    try:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(file_path,'r') as zip_ref: zip_ref.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files: st.error("Aucun .shp dans le zip."); st.stop()
            gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
        else:
            gdf = gpd.read_file(file_path)

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}"); st.stop()

# Géométrie globale simplifiée pour GEE
merged_poly = gdf.geometry.unary_union.simplify(0.001, preserve_topology=True)
geom = shapely_to_ee(merged_poly)
st.success(f"✅ Zone chargée : {len(gdf)} entité(s)")

# ------------------------------------------------------------
# DATE SELECTION
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Période d’analyse")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2024-08-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("2024-09-30"))
if start_date >= end_date: st.error("La date de fin doit être postérieure à la date de début."); st.stop()

# ------------------------------------------------------------
# SENTINEL-1 FLOOD DETECTION
# ------------------------------------------------------------
@st.cache_data
def detect_floods(aoi_serialized,start,end):
    aoi = ee.Geometry(aoi_serialized)
    t_start, t_end = pd.to_datetime(start), pd.to_datetime(end)
    d_start = t_start.strftime('%Y-%m-%d')
    d_start_p15 = (t_start+pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    d_end_m15 = (t_end-pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    d_end = t_end.strftime('%Y-%m-%d')

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(d_start,d_end)
          .filter(ee.Filter.eq("instrumentMode","IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
          .select("VV"))
    before = s1.filterDate(d_start,d_start_p15).median()
    after = s1.filterDate(d_end_m15,d_end).median()
    diff = after.subtract(before)
    flood = diff.lt(-3)
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    water_perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    flood = flood.updateMask(slope.lt(5)).updateMask(water_perm.lt(10))
    return flood.selfMask()

with st.spinner("Analyse satellite Sentinel-1..."):
    flood_img = detect_floods(geom.getInfo(), start_date, end_date)

# ------------------------------------------------------------
# CALCUL DES INDICATEURS GLOBAUX
# ------------------------------------------------------------
pixel_area = ee.Image.pixelArea()
pop_img = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01","2020-12-31").mean()

total_flood_area = ee.Number(flood_img.multiply(pixel_area).reduceRegion(ee.Reducer.sum(),geom,100,maxPixels=1e13).get("VV")).divide(1e6).getInfo()
total_pop_exposed = int(ee.Number(pop_img.updateMask(flood_img).reduceRegion(ee.Reducer.sum(),geom,100,maxPixels=1e13).get("population")).getInfo())
total_rain = float(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(geom).filterDate(str(start_date),str(end_date)).sum().reduceRegion(ee.Reducer.mean(),geom,5000,maxPixels=1e13).get("precipitation").getInfo())

st.subheader("📊 Indicateurs globaux")
c1,c2,c3 = st.columns(3)
c1.metric("Surface inondée", f"{total_flood_area:.2f} km²")
c2.metric("Population exposée", f"{total_pop_exposed:,}")
c3.metric("Pluie cumulée", f"{total_rain:.1f} mm")

# ------------------------------------------------------------
# CALCUL DES INDICATEURS PAR ZONE
# ------------------------------------------------------------
zone_metrics = []
for idx,row in gdf.iterrows():
    poly = row.geometry
    ee_poly = shapely_to_ee(poly)
    try:
        flood_km2 = ee.Number(flood_img.multiply(pixel_area).reduceRegion(ee.Reducer.sum(),ee_poly,100,1e13).get("VV")).divide(1e6).getInfo()
        pop_exposed = int(ee.Number(pop_img.updateMask(flood_img).reduceRegion(ee.Reducer.sum(),ee_poly,100,1e13).get("population")).getInfo())
        rain_total = float(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(ee_poly).filterDate(str(start_date),str(end_date)).sum().reduceRegion(ee.Reducer.mean(),ee_poly,5000,1e13).get("precipitation").getInfo())
    except: flood_km2=0; pop_exposed=0; rain_total=0
    pct_flood = flood_km2/(poly.area/1e6) if poly.area>0 else 0
    zone_metrics.append({"zone":idx+1,"surface_km2":poly.area/1e6,"flood_km2":flood_km2,
                         "pct_flood":pct_flood*100,"pop_exposed":pop_exposed,"rain_mm":rain_total})

df_metrics = pd.DataFrame(zone_metrics)

# ------------------------------------------------------------
# CARTE INTERACTIVE AVEC DEGRADÉ
# ------------------------------------------------------------
st.subheader("🗺️ Carte interactive des zones inondées")
bounds = gdf.total_bounds
m = folium.Map(location=[(bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2],zoom_start=10,tiles="CartoDB positron")

def get_color(pct):
    if pct<10: return "#2ECC71"
    elif pct<30: return "#F1C40F"
    elif pct<60: return "#E67E22"
    else: return "#E74C3C"

for idx,row in gdf.iterrows():
    poly = row.geometry
    metrics = df_metrics.iloc[idx]
    popup_html = f"""
    <b>Zone {metrics.zone}</b><br>
    Surface totale: {metrics.surface_km2:.2f} km²<br>
    Surface inondée: {metrics.flood_km2:.2f} km²<br>
    % inondée: {metrics.pct_flood:.1f}%<br>
    Population exposée: {metrics.pop_exposed:,}<br>
    Pluie cumulée: {metrics.rain_mm:.1f} mm
    """
    folium.GeoJson(
        poly,
        style_function=lambda feature,pct=metrics.pct_flood: {"fillColor":get_color(pct),
                                                              "color":"#555555","weight":2,"fillOpacity":0.6},
        tooltip=folium.Tooltip(f"Zone {metrics.zone} – {metrics.pct_flood:.1f}% inondée"),
        popup=folium.Popup(popup_html,max_width=300)
    ).add_to(m)

# Overlay Sentinel-1
try:
    flood_mapid = flood_img.getMapId({'min':0,'max':1,'palette':['#0000FF']})
    folium.TileLayer(tiles=flood_mapid['tile_fetcher'].url_format,
                     attr='GEE - Sentinel-1',name='Zones Inondées',overlay=True,control=True,opacity=0.4).add_to(m)
except: pass

# Légende
legend_html="""
<div style="position: fixed; bottom:50px; left:50px; width:220px; height:160px; background:white; z-index:9999; font-size:14px; border:2px solid grey; padding:5px;">
<b>Légende % inondation</b><br>
<i style="background:#2ECC71;width:15px;height:15px;float:left;margin-right:5px;"></i> <10%<br>
<i style="background:#F1C40F;width:15px;height:15px;float:left;margin-right:5px;"></i> 10-30%<br>
<i style="background:#E67E22;width:15px;height:15px;float:left;margin-right:5px;"></i> 30-60%<br>
<i style="background:#E74C3C;width:15px;height:15px;float:left;margin-right:5px;"></i> >60%<br>
<i style="background:blue;width:15px;height:15px;float:left;margin-right:5px;"></i> Sentinel-1
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl().add_to(m)
st_folium(m,width="100%",height=600)

# ------------------------------------------------------------
# SERIE TEMPORELLE
# ------------------------------------------------------------
st.subheader("📈 Dynamique des précipitations quotidiennes (CHIRPS)")

@st.cache_data
def get_rain_series(aoi_serialized,start,end):
    try:
        aoi = ee.Geometry(aoi_serialized)
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate(str(start),str(end))
        def extract_val(img):
            val = img.reduceRegion(ee.Reducer.mean(),geometry=aoi,scale=5000).get('precipitation')
            return ee.Feature(None,{'date':img.date().format('YYYY-MM-dd'),'rain':val})
        fc = chirps.map(extract_val).getInfo()
        return pd.DataFrame([f['properties'] for f in fc['features']])
    except: return pd.DataFrame()

with st.spinner("Calcul des précipitations journalières..."):
    df_rain = get_rain_series(geom.getInfo(),start_date,end_date)
    if not df_rain.empty:
        df_rain['date']=pd.to_datetime(df_rain['date'])
        fig = px.bar(df_rain,x='date',y='rain',title="Précipitations quotidiennes (mm)",color_discrete_sequence=['#00aaff'])
        st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------------------
# TABLEAU RECAPITULATIF
# ------------------------------------------------------------
st.subheader("📋 Tableau récapitulatif par zone")

# Filtres interactifs
min_pct = st.slider("Filtrer par % inondée minimum",0,100,0)
max_pct = st.slider("Filtrer par % inondée maximum",0,100,100)
df_filtered = df_metrics[(df_metrics.pct_flood>=min_pct)&(df_metrics.pct_flood<=max_pct)]

st.dataframe(df_filtered.style.format({"surface_km2":"{:.2f}","flood_km2":"{:.2f}",
                                       "pct_flood":"{:.1f}%","rain_mm":"{:.1f}","pop_exposed":"{:,}"}))

# Export CSV
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Télécharger le tableau CSV", data=csv, file_name="zone_metrics_filtered.csv", mime="text/csv")
