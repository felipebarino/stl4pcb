import logging
import numpy as np
from stl import mesh as stl_mesh

logger = logging.getLogger(__name__)

def create_pixel_mesh(mask, thickness, dpmm):
    """
    Creates a voxel-style 3D mesh by treating every white pixel as a solid rectangular block.
    
    This method is extremely robust against malformed geometry but inefficient in terms of 
    triangle count.
    
    Args:
        mask (np.ndarray): Boolean 2D array where True represents copper.
        thickness (float): The extrusion height in mm.
        dpmm (int): Dots Per Millimeter, used to scale pixel indices to physical coordinates.

    Returns:
        stl.mesh.Mesh: The resulting 3D mesh.
    """
    logger.info(f"Starting Voxel/Pixel mesh generation (DPMM={dpmm})...")
    
    facets = []
    scale = 1.0 / dpmm
    
    total_pixels = np.count_nonzero(mask)
    logger.debug(f"Processing {total_pixels} active pixels.")

    # --- 1. Vertical Walls via Gradient Detection ---
    
    # Horizontal gradients (Left/Right walls)
    # diff returns != 0 exactly at the edges of copper features
    diff_h = np.diff(mask.astype(int), axis=1)
    # np.argwhere returns coordinates (row, col)
    edges_h = np.argwhere(diff_h != 0)
    
    for r, c in edges_h:
        x = (c + 1) * scale 
        y1, y2 = r * scale, (r + 1) * scale
        
        # Two triangles for the vertical face
        facets.append([[x, y1, 0], [x, y2, 0], [x, y1, thickness]])
        facets.append([[x, y2, 0], [x, y2, thickness], [x, y1, thickness]])
        
    logger.debug(f"Generated {len(edges_h) * 2} facets for vertical (Y-aligned) walls.")

    # Vertical gradients (Top/Bottom walls)
    diff_v = np.diff(mask.astype(int), axis=0)
    edges_v = np.argwhere(diff_v != 0)
    
    for r, c in edges_v:
        y = (r + 1) * scale
        x1, x2 = c * scale, (c + 1) * scale
        
        # Two triangles for the horizontal face
        facets.append([[x1, y, 0], [x2, y, 0], [x1, y, thickness]])
        facets.append([[x2, y, 0], [x2, y, thickness], [x1, y, thickness]])

    logger.debug(f"Generated {len(edges_v) * 2} facets for horizontal (X-aligned) walls.")

    # --- 2. Top and Bottom Caps ---
    # Create a floor and ceiling for every active pixel
    active_pixels = np.argwhere(mask)
    
    for r, c in active_pixels:
        x1, x2 = c * scale, (c + 1) * scale
        y1, y2 = r * scale, (r + 1) * scale
        
        # Bottom floor (Z=0)
        facets.append([[x1, y1, 0], [x2, y1, 0], [x1, y2, 0]])
        facets.append([[x2, y1, 0], [x2, y2, 0], [x1, y2, 0]])
        
        # Top ceiling (Z=thickness)
        facets.append([[x1, y1, thickness], [x1, y2, thickness], [x2, y1, thickness]])
        facets.append([[x2, y1, thickness], [x2, y2, thickness], [x1, y2, thickness]])
        
    logger.debug(f"Generated {len(active_pixels) * 4} facets for caps.")
    logger.info(f"Voxel mesh complete. Total facets: {len(facets)}")

    data = np.zeros(len(facets), dtype=stl_mesh.Mesh.dtype)
    for i, f in enumerate(facets):
        data['vectors'][i] = np.array(f)
    return stl_mesh.Mesh(data)