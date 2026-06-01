import logging
import numpy as np
from stl import mesh as stl_mesh
from shapely.geometry import Polygon
from skimage import measure
import trimesh
from shapely.geometry import Polygon, MultiPolygon

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
    Extrudes a Shapely Polygon or MultiPolygon into a 3D mesh,
    and returns a native numpy-stl Mesh object.
    """
    # 1. Safely handle individual Polygons vs MultiPolygons
    if isinstance(multipoly, Polygon):
        polygons = [multipoly]
    elif isinstance(multipoly, MultiPolygon):
        polygons = list(multipoly.geoms)
    elif hasattr(multipoly, 'geoms'): 
        polygons = list(multipoly.geoms)
    else:
        polygons = list(multipoly)

    meshes = []
    
    # 2. Extrude each component polygon separately
    for poly in polygons:
        if poly.is_empty:
            continue
        
        mesh = trimesh.creation.extrude_polygon(polygon=poly, height=thickness)
        meshes.append(mesh)

    if not meshes:
        logger.warning("No valid geometry found to extrude. Returning empty STL.")
        return stl_mesh.Mesh(np.zeros(0, dtype=stl_mesh.Mesh.dtype))

    # 3. Concatenate all individual meshes into one trimesh object
    combined_trimesh = trimesh.util.concatenate(meshes)

    # 4. BRIDGE TO NUMPY-STL: Convert trimesh data to numpy-stl layout
    # trimesh.triangles gives an (N, 3, 3) array, which matches numpy-stl's 'vectors'
    data = np.zeros(len(combined_trimesh.triangles), dtype=stl_mesh.Mesh.dtype)
    data['vectors'] = combined_trimesh.triangles
    
    # Return native stl_mesh object so main.py doesn't crash on `.vectors`
    return stl_mesh.Mesh(data)
