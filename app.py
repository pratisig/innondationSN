# ═════════════════════════════════════════════════════════════════
# CORRECTION: get_osm_data() - SANS @st.cache_data
# ═════════════════════════════════════════════════════════════════

def get_osm_data(gdf_aoi):
    """
    🏗️ Récupère données OpenStreetMap (routes, bâtiments).
    
    ✅ CHANGEMENT: Enlever @st.cache_data
    - GeoDataFrame pas hashable → cache échoue
    - OSMnx a son propre cache (ox.settings.use_cache = True)
    - Assez rapide sans cache Streamlit
    
    Paramètres:
    -----------
    gdf_aoi (GeoDataFrame): Zone d'étude
    
    Retour:
    -------
    tuple (GeoDataFrame_bâtiments, GeoDataFrame_routes)
    """
    
    if gdf_aoi is None or gdf_aoi.empty:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    try:
        poly = gdf_aoi.unary_union
        
        # 🛣️ ROUTES
        try:
            st.info("📥 Chargement routes OSM...")
            graph = ox.graph_from_polygon(
                poly, 
                network_type='all',
                simplify=True
            )
            gdf_routes = ox.graph_to_gdfs(
                graph, 
                nodes=False, 
                edges=True
            ).reset_index().clip(gdf_aoi)
            
            st.success(f"✅ {len(gdf_routes)} segments de route")
        
        except Exception as e:
            st.warning(f"⚠️ Routes OSM: {str(e)[:60]}")
            logger.warning(f"Routes error: {str(e)}")
            gdf_routes = gpd.GeoDataFrame()
        
        # 🏢 BÂTIMENTS
        try:
            st.info("📥 Chargement bâtiments OSM...")
            tags = {
                'building': True,
                'amenity': ['school', 'university', 'college', 
                           'hospital', 'clinic', 'doctors'],
                'healthcare': True,
                'education': True
            }
            
            gdf_buildings = ox.features_from_polygon(poly, tags=tags)
            
            # Filtrer polygones
            gdf_buildings = gdf_buildings[
                gdf_buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])
            ].reset_index().clip(gdf_aoi)
            
            st.success(f"✅ {len(gdf_buildings)} bâtiments")
        
        except Exception as e:
            st.warning(f"⚠️ Bâtiments OSM: {str(e)[:60]}")
            logger.warning(f"Buildings error: {str(e)}")
            gdf_buildings = gpd.GeoDataFrame()
        
        return gdf_buildings, gdf_routes
    
    except Exception as e:
        st.error(f"❌ OSM: {str(e)[:100]}")
        logger.error(f"OSM data error: {str(e)}")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
