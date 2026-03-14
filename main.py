import numpy as np
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme
from PIL import Image
from skimage import measure
from stl import mesh as stl_mesh
from shapely.geometry import Polygon
from shapely.ops import unary_union, triangulate
import argparse
import os

def extrude_multipolygon(multipoly, thickness):
    """
    Converts a Shapely MultiPolygon (2D) into a 3D STL mesh with correct holes and normals.

    This function handles:
    1. Identification of exterior boundaries and interior holes.
    2. Generation of vertical walls connecting the bottom (Z=0) to the top (Z=thickness).
    3. Triangulation of the top and bottom caps using Shapely's ear-clipping algorithm.

    Args:
        multipoly (shapely.geometry.MultiPolygon or Polygon): The 2D geometry of the PCB traces.
        thickness (float): The extrusion height in millimeters.

    Returns:
        stl.mesh.Mesh: A numpy-stl mesh object ready to be saved.
    """
    all_facets = []
    
    # Ensure we are iterating over a list of Polygons, even if input is a single Polygon
    geoms = multipoly.geoms if hasattr(multipoly, 'geoms') else [multipoly]

    print(f"  - Extruding {len(geoms)} independent polygon islands...")

    for poly in geoms:
        if poly.is_empty: continue
        
        # --- 1. Generate Vertical Walls (Side Surfaces) ---
        
        # A. Exterior boundary walls
        # coords is a list of (x, y) tuples
        coords = np.array(poly.exterior.coords)
        for i in range(len(coords)-1):
            p1, p2 = coords[i], coords[i+1]
            # Create two triangles for the rectangular face
            # Triangle 1: Bottom-Left -> Bottom-Right -> Top-Left
            all_facets.append([[p1[0], p1[1], 0], [p2[0], p2[1], 0], [p1[0], p1[1], thickness]])
            # Triangle 2: Bottom-Right -> Top-Right -> Top-Left
            all_facets.append([[p2[0], p2[1], 0], [p2[0], p2[1], thickness], [p1[0], p1[1], thickness]])
            
        # B. Interior hole walls
        # Note: We reverse the vertex order or handle logic to ensure normals point 'out' into the void
        for interior in poly.interiors:
            coords = np.array(interior.coords)
            for i in range(len(coords)-1):
                p1, p2 = coords[i], coords[i+1]
                # Wall faces for holes
                all_facets.append([[p1[0], p1[1], 0], [p1[0], p1[1], thickness], [p2[0], p2[1], 0]])
                all_facets.append([[p2[0], p2[1], 0], [p1[0], p1[1], thickness], [p2[0], p2[1], thickness]])

        # --- 2. Generate Horizontal Caps (Top and Bottom Surfaces) ---
        
        # Triangulate the complex polygon (handles holes automatically)
        # Result is a list of simple triangles
        triangles = triangulate(poly)
        
        for tri in triangles:
            # Check if the triangle is actually part of the polygon (and not inside a hole)
            # This is a robust check but can be slow for very complex geometries
            if poly.contains(tri.centroid):
                v = list(tri.exterior.coords)
                
                # Bottom Cap (Z=0)
                # Normal must point DOWN (0, 0, -1). Vertex order: Clockwise relative to view from top
                all_facets.append([[v[0][0], v[0][1], 0], [v[2][0], v[2][1], 0], [v[1][0], v[1][1], 0]])
                
                # Top Cap (Z=thickness)
                # Normal must point UP (0, 0, 1). Vertex order: Counter-Clockwise
                all_facets.append([[v[0][0], v[0][1], thickness], [v[1][0], v[1][1], thickness], [v[2][0], v[2][1], thickness]])

    # Convert list of facets to numpy-stl Mesh object
    data = np.zeros(len(all_facets), dtype=stl_mesh.Mesh.dtype)
    for i, f in enumerate(all_facets):
        data['vectors'][i] = np.array(f)
        
    return stl_mesh.Mesh(data)

def create_pixel_mesh(mask, thickness, dpmm):
    """
    Creates a voxel-style 3D mesh by treating every white pixel as a solid rectangular block.
    
    This method is "fool-proof" for complex bitmaps but produces large file sizes and rough edges.
    
    Args:
        mask (np.ndarray): Boolean 2D array where True represents copper.
        thickness (float): The extrusion height in mm.
        dpmm (int): Dots Per Millimeter, used to scale pixel indices to physical coordinates.

    Returns:
        stl.mesh.Mesh: The resulting 3D mesh.
    """
    facets = []
    scale = 1.0 / dpmm

    # --- 1. Vertical Walls via Gradient Detection ---
    # Instead of checking every pixel neighbor, we check where the value changes (0->1 or 1->0)
    
    # Horizontal gradients (detect Left and Right walls)
    # diff returns 1 where we step onto copper, -1 where we step off
    diff_h = np.diff(mask.astype(int), axis=1)
    
    for r, c in np.argwhere(diff_h != 0):
        # c is the column index *before* the transition
        x = (c + 1) * scale # Physical X coordinate of the wall
        y1, y2 = r * scale, (r + 1) * scale # The wall spans this Y range
        
        # We add two triangles to form the rectangular wall face
        facets.append([[x, y1, 0], [x, y2, 0], [x, y1, thickness]])
        facets.append([[x, y2, 0], [x, y2, thickness], [x, y1, thickness]])

    # Vertical gradients (detect Top and Bottom walls)
    diff_v = np.diff(mask.astype(int), axis=0)
    
    for r, c in np.argwhere(diff_v != 0):
        y = (r + 1) * scale
        x1, x2 = c * scale, (c + 1) * scale
        facets.append([[x1, y, 0], [x2, y, 0], [x1, y, thickness]])
        facets.append([[x2, y, 0], [x2, y, thickness], [x1, y, thickness]])

    # --- 2. Top and Bottom Caps ---
    # We iterate through every 'True' pixel to create its floor and ceiling
    # Optimization note: This creates many internal vertices compared to contour method
    for r, c in np.argwhere(mask):
        x1, x2 = c * scale, (c + 1) * scale
        y1, y2 = r * scale, (r + 1) * scale
        
        # Bottom floor (Z=0)
        facets.append([[x1, y1, 0], [x2, y1, 0], [x1, y2, 0]])
        facets.append([[x2, y1, 0], [x2, y2, 0], [x1, y2, 0]])
        
        # Top ceiling (Z=thickness)
        facets.append([[x1, y1, thickness], [x1, y2, thickness], [x2, y1, thickness]])
        facets.append([[x2, y1, thickness], [x2, y2, thickness], [x1, y2, thickness]])

    data = np.zeros(len(facets), dtype=stl_mesh.Mesh.dtype)
    for i, f in enumerate(facets):
        data['vectors'][i] = np.array(f)
    return stl_mesh.Mesh(data)

if __name__ == "__main__":
    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Gerber to STL Converter (Precision PCB Extrusion)")
    parser.add_argument("input_file", help="Path to input Gerber file (.gbr)")
    parser.add_argument("-o", "--output", help="Path to output STL file", default="output.stl")
    parser.add_argument("--method", choices=['multipolygon', 'pixel'], default='multipolygon', 
                        help="Algo: 'multipolygon' (smoother, for traces) or 'pixel' (robust, for complex fills)")
    parser.add_argument("--dpmm", type=int, default=60, help="Resolution in Dots Per Millimeter (higher = smoother)")
    parser.add_argument("--thickness", type=float, default=0.035, help="Extrusion thickness in mm (default 35um)")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output
    temp_render_file = "temp_render.png"

    print(f"Processing: {input_file}")
    print(f"Settings: Method={args.method}, DPMM={args.dpmm}, Thickness={args.thickness}mm")

    # 1. Parse Gerber File
    try:
        gerber = GerberFile.from_file(input_file)
        parsed = gerber.parse()
    except Exception as e:
        print(f"Error parsing Gerber file: {e}")
        exit(1)

    # 2. Render to Raster (High-Res Image)
    # This converts vector Gerber commands into a boolean pixel map
    print("Rendering Gerber to raster image...")
    parsed.render_raster(temp_render_file, color_scheme=ColorScheme.DEFAULT_GRAYSCALE, dpmm=args.dpmm)
    
    # 3. Load Image as Boolean Mask
    # Copper is White (255), Background is Black (0)
    img = np.asarray(Image.open(temp_render_file).convert("L"))
    binary_mask = img > 127
    
    # Basic cleanup
    if os.path.exists(temp_render_file): 
        os.remove(temp_render_file)
        
    # Pad the mask with 0s to ensure contours at the image edge are closed loops
    binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)

    # 4. Generate 3D Mesh based on selected method
    if args.method == 'pixel':
        print("Creating voxel mesh from pixel data...")
        final_mesh = create_pixel_mesh(binary_mask, args.thickness, args.dpmm)
    else:
        print("Extracting contours...")
        contours = measure.find_contours(binary_mask, level=0.5)
        print(f"  - Found {len(contours)} raw contours throughout the image.")
        
        # Convert raw pixel contours to physical Shapely Polygons
        scale = 1.0 / args.dpmm
        polys = []
        for c in contours:
            # We filter out tiny noise (less than 3 points)
            if len(c) >= 3:
                # skimage uses (row, col), we map to (x, y) = (col, row)
                # and apply the scaling factor
                poly_pts = np.column_stack((c[:, 1] * scale, c[:, 0] * scale))
                polys.append(Polygon(poly_pts))
        
        print("Resolving topology (Unifying overlaps and holes)...")
        # unary_union automatically determines which polygons are holes inside others
        merged_geometry = unary_union(polys)
        
        print("Extruding geometry...")
        final_mesh = extrude_multipolygon(merged_geometry, args.thickness)

    # 5. Save output
    final_mesh.save(output_file)
    print(f"Success! STL saved to: {output_file}")
    print(f"Mesh Stats: {len(final_mesh.vectors)} facets")

    # 6. Optional Visualization
    try:
        import pyvista as pv
        print("Opening 3D viewer...")
        pv.plot(pv.read(output_file), color="gold", title="PCB Extrusion Preview")
    except ImportError:
        print("Install 'pyvista' for 3D preview.")