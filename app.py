import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import shape, mapping
import json
import ee
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & INITIALISATION
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FloodWatch WA", 
    page_icon="🌊",
    layout="wide"
)

# Paramètres OSMnx
ox.settings.timeout = 180
ox.settings.use_cache = True

# Initialisation GEE
@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(key_dict["client_email"], key_data=json.dumps(key_dict))
            ee.Initialize(credentials)
            return True
        ee.Initialize()
        return True
    except Exception:
        return False

gee_available = init_gee()

# ═════════════════════════════════════════════════════════════════
# 2. FONCTIONS DE TRAITEMENT (GEE, CLIMAT & OSM)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gadm(iso, level):
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso.upper()}.gpkg"
    try:
        gdf = gpd.read_file(url, layer=level)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return None

def get_flood_mask(aoi_ee, start_flood, end_flood, threshold=4.0):
    if not gee_available: return None
    try:
        s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterBounds(aoi_ee)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .filterDate(start_flood, end_flood)
                  .select('VV'))
        
        img_flood = s1_col.min().clip(aoi_ee).focal_median(50, 'circle', 'meters')
        img_ref = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterBounds(aoi_ee)
                   .filterDate("2023-01-01", "2023-04-30")
                   .select('VV').median().clip(aoi_ee).focal_median(50, 'circle', 'meters'))

        diff = img_ref.subtract(img_flood)
        flood_raw = diff.gt(threshold)

        permanent_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        water_mask = permanent_water.gt(80).clip(aoi_ee)
        
        final_flood = flood_raw.where(water_mask, 0).selfMask()
        return final_flood.rename('flood')
    except: return None

def get_climate_data(aoi_ee, start_date, end_date):
    """Récupère le cumul de pluie (CHIRPS) et la température (ERA5)"""
    if not gee_available: return None
    try:
        precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterBounds(aoi_ee) \
            .filterDate(start_date, end_date) \
            .select('precipitation')
        
        temp = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterBounds(aoi_ee) \
            .filterDate(start_date, end_date) \
            .select('temperature_2m')

        def extract_stats(img_col, band_name, reducer):
            def wrap(img):
                val = img.reduceRegion(reducer=reducer, geometry=aoi_ee, scale=5000).get(band_name)
                return ee.Feature(None, {'date': img.date().format('YYYY-MM-DD'), 'value': val})
            return ee.FeatureCollection(img_col.map(wrap)).getInfo()['features']

        p_data = extract_stats(precip, 'precipitation', ee.Reducer.mean())
        t_data = extract_stats(temp, 'temperature_2m', ee.Reducer.mean())

        df_p = pd.DataFrame([f['properties'] for f in p_data])
        df_t = pd.DataFrame([f['properties'] for f in t_data])
        df_t['value'] = df_t['value'] - 273.15 
        
        df_clim = df_p.merge(df_t, on='date', suffixes=('_precip', '_temp'))
        return df_clim
    except: return None

def get_population_stats(aoi_ee, flood_mask):
    if not gee_available: return 0, 0
    try:
        pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                        .filterDate('2020-01-01', '2021-01-01') \
                        .mosaic().clip(aoi_ee)
        
        stats_total = pop_dataset.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
        total_pop = stats_total.get('population').getInfo() or 0
        
        exposed_pop = 0
        if flood_mask:
            stats_exposed = pop_dataset.updateMask(flood_mask).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=100, maxPixels=1e9)
            exposed_pop = stats_exposed.get('population').getInfo() or 0
            
        return int(total_pop), int(exposed_pop)
    except: return 0, 0

def get_area_stats(aoi_ee, flood_mask):
    if not gee_available or not flood_mask: return 0.0
    try:
        area_m2 = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi_ee, scale=10, maxPixels=1e9).get('flood').getInfo()
        return (area_m2 or 0) / 10000
    except: return 0.0

def get_osm_data(_gdf_aoi):
    if _gdf_aoi is None or _gdf_aoi.empty: return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    try:
        poly = _gdf_aoi.unary_union
        graph = ox.graph_from_polygon(poly, network_type='all', simplify=True)
        gdf_routes = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index().clip(_gdf_aoi)
        
        tags = {'building': True, 'amenity': ['school', 'university', 'college', 'hospital', 'clinic', 'doctors'], 'healthcare': True, 'education': True}
        try:
            gdf_buildings = ox.features_from_polygon(poly, tags=tags)
        except:
            gdf_buildings = ox.geometries_from_polygon(poly, tags=tags)
            
        gdf_buildings = gdf_buildings[gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        gdf_buildings = gdf_buildings.reset_index().clip(_gdf_aoi)
        return gdf_buildings, gdf_routes
    except: return gpd.GeoDataFrame(), gpd.GeoDataFrame()

def analyze_impacted_infra(flood_mask, buildings_gdf):
    if flood_mask is None or buildings_gdf.empty: return gpd.GeoDataFrame()
    try:
        infra_check = buildings_gdf.head(3000).copy()
        features = [ee.Feature(ee.Geometry(mapping(row.geometry)), {'idx': i}) for i, row in infra_check.iterrows()]
        fc = ee.FeatureCollection(features)
        reduced = flood_mask.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        impacted_indices = [f['properties']['idx'] for f in reduced.filter(ee.Filter.gt('mean', 0)).getInfo()['features']]
        return infra_check.loc[impacted_indices]
    except: return gpd.GeoDataFrame()

def analyze_impacted_roads(flood_mask, roads_gdf):
    if flood_mask is None or roads_gdf.empty: return gpd.GeoDataFrame()
    try:
        roads_check = roads_gdf.head(5000).copy()
        features = [ee.Feature(ee.Geometry(mapping(row.geometry)), {'idx': i}) for i, row in roads_check.iterrows()]
        fc = ee.FeatureCollection(features)
        reduced = flood_mask.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        impacted_indices = [f['properties']['idx'] for f in reduced.filter(ee.Filter.gt('mean', 0)).getInfo()['features']]
        return roads_check.loc[impacted_indices]
    except: return gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════
# 3. INTERFACE UTILISATEUR
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🗺️ 1. Zone d'Étude")
mode = st.sidebar.radio("Méthode :", ["Liste Administrative", "Dessiner sur Carte"])

if 'selected_zone' not in st.session_state: st.session_state.selected_zone = None
if 'zone_name' not in st.session_state: st.session_state.zone_name = "Zone personnalisée"
if 'analysis_triggered' not in st.session_state: st.session_state.analysis_triggered = False

if mode == "Liste Administrative":
    countries = {"Sénégal": "SEN", "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA"}
    c_choice = st.sidebar.selectbox("Pays", list(countries.keys()))
    level = st.sidebar.slider("Niveau Admin", 0, 5, 2)
    gdf_base = load_gadm(countries[c_choice], level)
    if gdf_base is not None:
        col = f"NAME_{level}" if level > 0 else "COUNTRY"
        choice = st.sidebar.selectbox("Subdivision", sorted(gdf_base[col].dropna().unique()))
        new_zone = gdf_base[gdf_base[col] == choice].copy()
        if st.session_state.zone_name != choice:
            st.session_state.selected_zone = new_zone
            st.session_state.zone_name = choice
            st.session_state.analysis_triggered = False

elif mode == "Dessiner sur Carte":
    m_draw = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles="cartodbpositron")
    Draw(export=False).add_to(m_draw)
    with st.sidebar:
        out = st_folium(m_draw, width=250, height=250, key="draw_static")
        if out and out.get('last_active_drawing'):
            geom = shape(out['last_active_drawing']['geometry'])
            st.session_state.selected_zone = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
            st.session_state.zone_name = "Zone Dessinée"
            st.session_state.analysis_triggered = False

st.sidebar.markdown("## 📅 2. Paramètres")
start_f = st.sidebar.date_input("Début période", datetime(2024, 8, 1))
end_f = st.sidebar.date_input("Fin période", datetime(2024, 9, 30))
threshold_val = st.sidebar.slider("Seuil radar (dB)", 2.0, 8.0, 4.0)

# ═════════════════════════════════════════════════════════════════
# 4. LOGIQUE PRINCIPALE
# ═════════════════════════════════════════════════════════════════

st.title(f"🌊 FloodWatch : {st.session_state.zone_name}")

if st.session_state.selected_zone is not None:
    if st.button("🚀 LANCER L'ANALYSE COMPLÈTE", type="primary", use_container_width=True):
        st.session_state.analysis_triggered = True

    if st.session_state.analysis_triggered:
        with st.spinner("Analyse multicritère en cours (Climat, Population, Infrastructures)..."):
            # A. GEE
            aoi_ee_global = ee.Geometry(mapping(st.session_state.selected_zone.unary_union))
            flood_mask = get_flood_mask(aoi_ee_global, str(start_f), str(end_f), threshold_val)
            df_climat = get_climate_data(aoi_ee_global, str(start_f), str(end_f))
            
            # B. OSM & Impacts
            buildings, routes = get_osm_data(st.session_state.selected_zone)
            impacted_infra = analyze_impacted_infra(flood_mask, buildings)
            impacted_roads = analyze_impacted_roads(flood_mask, routes)
            
            # C. Population et Superficie par subdivision
            sector_data = []
            total_pop_all = 0
            total_pop_exposed = 0
            total_flood_ha = 0
            
            for idx, row in st.session_state.selected_zone.iterrows():
                geom_ee = ee.Geometry(mapping(row.geometry))
                t_pop, e_pop = get_population_stats(geom_ee, flood_mask)
                f_area = get_area_stats(geom_ee, flood_mask)
                
                total_pop_all += t_pop
                total_pop_exposed += e_pop
                total_flood_ha += f_area
                
                sector_data.append({
                    'Secteur': row.get('NAME_2', row.get('NAME_1', st.session_state.zone_name)),
                    'Pop. Totale': t_pop,
                    'Pop. Exposée': e_pop,
                    '% Impacté': f"{(e_pop/t_pop*100):.1f}%" if t_pop > 0 else "0%",
                    'Inondation (ha)': round(f_area, 2)
                })
            
            # --- SECTION 1: BILAN IMPACT POPULATION & INDICATEURS ---
            st.markdown("### 📊 Indicateurs de Risque et d'Impact")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.markdown(f"**🏠 Population Totale**\n## {total_pop_all:,}")
            with p2:
                color = "red" if total_pop_exposed > 0 else "gray"
                perc = (total_pop_exposed / total_pop_all * 100) if total_pop_all > 0 else 0
                st.markdown(f"**⚠️ Population Sinistrée**\n<h2 style='color:{color}'>{total_pop_exposed:,} <span style='font-size: 16px; font-weight: normal; color: #555;'>( {perc:.1f}% )</span></h2>", unsafe_allow_html=True)
            with p3:
                st.markdown(f"**🌊 Zone Inondée**\n## {total_flood_ha:.2f} ha")
            with p4:
                # Calcul rapide du cumul de pluie si disponible
                rain_sum = df_climat['value_precip'].sum() if df_climat is not None else 0
                st.markdown(f"**🌧️ Cumul Pluie**\n## {rain_sum:.1f} mm")

            # --- SECTION 2: CARTE & INFRA ---
            col_map, col_stats = st.columns([2, 1])
            with col_map:
                st.markdown("#### 🗺️ Cartographie des Dégâts")
                center = st.session_state.selected_zone.centroid.iloc[0]
                m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
                
                # Ajout des polygones de la zone sélectionnée
                folium.GeoJson(
                    st.session_state.selected_zone,
                    name="Zone d'Étude",
                    style_function=lambda x: {
                        'fillColor': '#f0f0f0',
                        'color': 'black',
                        'weight': 2,
                        'fillOpacity': 0.1
                    }
                ).add_to(m)

                if flood_mask:
                    map_id = flood_mask.getMapId({'palette': ['#00bfff']})
                    folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='GEE', name='Inondations').add_to(m)
                
                if not impacted_roads.empty:
                    folium.GeoJson(impacted_roads, name="Routes Coupées", style_function=lambda x: {'color': 'red', 'weight': 4}).add_to(m)
                
                if not impacted_infra.empty:
                    folium.GeoJson(impacted_infra, name="Bâtiments Touchés", style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 1, 'fillOpacity': 0.7}).add_to(m)
                
                folium.LayerControl().add_to(m)
                st_folium(m, width="100%", height=500, key="map_res")

            with col_stats:
                st.markdown("#### 🏗️ Infrastructures")
                st.metric("Routes impactées", f"{len(impacted_roads)} seg.")
                st.metric("Bâtiments touchés", f"{len(impacted_infra)}")
                
                if not impacted_infra.empty:
                    def translate_type(row):
                        t = str(row.get('amenity', row.get('building', 'Autre'))).lower()
                        if any(x in t for x in ['school', 'university', 'college']): return "🏫 Écoles"
                        if any(x in t for x in ['hospital', 'clinic', 'health']): return "🏥 Santé"
                        return "🏠 Habitat"
                    impacted_infra['Cat'] = impacted_infra.apply(translate_type, axis=1)
                    st.plotly_chart(px.pie(impacted_infra, names='Cat', hole=0.4, title="Répartition des bât. impactés"), use_container_width=True)

            # --- SECTION 3: CLIMAT (DÉPLACÉ SOUS LA CARTE) ---
            st.markdown("### ☁️ Suivi Climatique & Précipitations")
            if df_climat is not None:
                fig_clim = go.Figure()
                fig_clim.add_trace(go.Bar(x=df_climat['date'], y=df_climat['value_precip'], name="Pluie (mm)", marker_color='royalblue'))
                fig_clim.add_trace(go.Scatter(x=df_climat['date'], y=df_climat['value_temp'], name="Température (°C)", yaxis='y2', line=dict(color='orange', width=3)))
                
                fig_clim.update_layout(
                    title="Diagramme Ombrothermique (Précipitations et Température)",
                    yaxis=dict(title="Précipitations (mm)"),
                    yaxis2=dict(title="Température (°C)", overlaying='y', side='right'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_clim, use_container_width=True)
            
            # --- SECTION 4: TABLEAU DE SYNTHÈSE ---
            st.markdown("### 📋 Synthèse détaillée par Secteur")
            st.table(pd.DataFrame(sector_data))

            # --- SECTION 5: EXPORTATION ---
            st.divider()
            st.markdown("### 📥 Exportation des Données")
            ex1, ex2, ex3 = st.columns(3)
            
            # Export CSV des impacts consolidés
            df_impact = pd.DataFrame(sector_data)
            
            with ex1:
                st.download_button("📊 Export Rapport (CSV)", df_impact.to_csv(index=False), "rapport_impact_secteurs.csv", "text/csv")
            with ex2:
                if not impacted_infra.empty:
                    st.download_button("🏘️ Bâtiments Impactés (GeoJSON)", impacted_infra.to_json(), "batiments_impact.geojson", "application/json")
            with ex3:
                if df_climat is not None:
                    st.download_button("🌡️ Données Climatiques (CSV)", df_climat.to_csv(index=False), "donnees_climat.csv", "text/csv")
