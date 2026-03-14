import logging
import numpy as np
from stl import mesh as stl_mesh
from shapely.geometry import Polygon
from shapely.ops import unary_union, triangulate
from skimage import measure

logger = logging.getLogger(__name__)

def get_polygons(binary_mask, dpmm):
    """
    Extracts geometric contours from a binary image mask and converts them into 
    Shapely Polygons, resolving nested topologies (holes).

    Args:
        binary_mask (np.ndarray): Boolean image where True is copper.
        dpmm (int): Dots Per Millimeter resolution for scaling.

    Returns:
        shapely.geometry.base.BaseGeometry: A unified Polygon or MultiPolygon object.
    """
    logger.info("Starting contour extraction from binary mask...")
    contours = measure.find_contours(binary_mask, level=0.5)
    logger.info(f"Found {len(contours)} raw contour loops.")
    
    # Convert raw pixel contours to physical Shapely Polygons
    scale = 1.0 / dpmm
    polys = []
    
    for i, c in enumerate(contours):
        # Filter out tiny noise (less than 3 points cannot form a polygon)
        if len(c) < 3:
            continue
            
        # skimage uses (row, col) -> (y, x)
        # We map to (x, y) = (col, row) and apply the scaling factor
        # Note: c[:, 1] is Column (X), c[:, 0] is Row (Y)
        poly_pts = np.column_stack((c[:, 1] * scale, c[:, 0] * scale))
        
        try:
            p = Polygon(poly_pts)
            if p.is_valid:
                polys.append(p)
            else:
                # Attempt to fix self-intersecting polygons by buffering with 0
                fixed_p = p.buffer(0)
                if fixed_p.is_valid and not fixed_p.is_empty:
                    polys.append(fixed_p)
                else:
                    logger.debug(f"Skipping invalid polygon index {i}")
        except Exception as e:
            logger.warning(f"Failed to create polygon for contour {i}: {e}")

    logger.debug(f"Converted {len(polys)} valid polygons. Resolving topology...")

    if not polys:
        return Polygon()
    
    # --- TOPOLOGY RESOLUTION ---
    # We use Symmetric Difference (XOR) to combine the polygons.
    # This works because of the "Even-Odd" rule for isolines:
    # - Outer contours add material.
    # - Inner contours (holes) overlap the outer ones, so XOR removes the material (creating a hole).
    # - Islands inside holes overlap again, so XOR adds the material back.
    # Optimization: Use a starting empty polygon
    merged_geometry = Polygon()
    
    # Note: For very large numbers of contours, a simple loop can be slow.
    # However, it is the most robust way to ensure holes are cut correctly without
    # explicit hierarchy data.
    for i, p in enumerate(polys):
        try:
            merged_geometry = merged_geometry.symmetric_difference(p)
        except Exception as e:
            logger.error(f"Topology error at index {i}: {e}")
            # Fallback: try buffering if difference fails due to precision
            p = p.buffer(0)
            merged_geometry = merged_geometry.symmetric_difference(p)

    geom_type = merged_geometry.geom_type
    count = len(merged_geometry.geoms) if hasattr(merged_geometry, 'geoms') else 1
    logger.info(f"Topology resolved: {geom_type} with ~{count} component(s).")
    
    return merged_geometry

def extrude_multipolygon(multipoly, thickness):
    """
    Extrudes 2D Shapely geometry into a 3D STL mesh.

    Args:
        multipoly (shapely.geometry.BaseGeometry): The 2D geometry (Polygon or MultiPolygon).
        thickness (float): The extrusion height (Z-axis) in mm.

    Returns:
        stl.mesh.Mesh: The 3D mesh object.
    """
    logger.info(f"Extruding geometry with thickness {thickness}mm...")
    all_facets = []
    
    # Normalize input to a list of Polygons
    geoms = multipoly.geoms if hasattr(multipoly, 'geoms') else [multipoly]

    logger.debug(f"Processing {len(geoms)} independent polygon islands...")

    for idx, poly in enumerate(geoms):
        if poly.is_empty: continue
        
        # --- 1. Generate Vertical Walls (Side Surfaces) ---
        
        # A. Exterior boundary walls
        coords = np.array(poly.exterior.coords)
        for i in range(len(coords)-1):
            p1, p2 = coords[i], coords[i+1]
            # Triangle 1: BL -> BR -> TL
            all_facets.append([[p1[0], p1[1], 0], [p2[0], p2[1], 0], [p1[0], p1[1], thickness]])
            # Triangle 2: BR -> TR -> TL
            all_facets.append([[p2[0], p2[1], 0], [p2[0], p2[1], thickness], [p1[0], p1[1], thickness]])
            
        # B. Interior hole walls
        # Reversing order isn't strictly necessary for visualizers but helps with standard STL normals
        for interior in poly.interiors:
            coords = np.array(interior.coords)
            for i in range(len(coords)-1):
                p1, p2 = coords[i], coords[i+1]
                # Wall faces for holes (pointing inward to the void)
                all_facets.append([[p1[0], p1[1], 0], [p1[0], p1[1], thickness], [p2[0], p2[1], 0]])
                all_facets.append([[p2[0], p2[1], 0], [p1[0], p1[1], thickness], [p2[0], p2[1], thickness]])

        # --- 2. Generate Horizontal Caps (Top and Bottom Surfaces) ---
        
        # Triangulate the polygon surface (handling holes)
        try:
            triangles = triangulate(poly)
            valid_triangles = 0
            
            for tri in triangles:
                # 'triangulate' creates a convex hull triangulation; we must filter 
                # strictly what is inside our specific polygon shape.
                if poly.contains(tri.centroid):
                    v = list(tri.exterior.coords)
                    
                    # Bottom Cap (Z=0) - Normal Down
                    all_facets.append([[v[0][0], v[0][1], 0], [v[2][0], v[2][1], 0], [v[1][0], v[1][1], 0]])
                    
                    # Top Cap (Z=thickness) - Normal Up
                    all_facets.append([[v[0][0], v[0][1], thickness], [v[1][0], v[1][1], thickness], [v[2][0], v[2][1], thickness]])
                    valid_triangles += 1
            
            if idx % 50 == 0:
                logger.debug(f"island {idx}: generated caps with {valid_triangles} triangles.")
                
        except Exception as e:
            logger.error(f"Triangulation failed for polygon {idx}: {e}")

    logger.info(f"Geometry generation complete. Total facets: {len(all_facets)}")

    # Convert to numpy-stl Mesh
    if not all_facets:
        logger.warning("No facets were generated! The output STL will be empty.")
        return stl_mesh.Mesh(np.zeros(0, dtype=stl_mesh.Mesh.dtype))

    data = np.zeros(len(all_facets), dtype=stl_mesh.Mesh.dtype)
    for i, f in enumerate(all_facets):
        data['vectors'][i] = np.array(f)
        
    return stl_mesh.Mesh(data)