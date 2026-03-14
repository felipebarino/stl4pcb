import numpy as np
import matplotlib.pyplot as plt
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme
from PIL import Image
from skimage import measure
from stl import mesh as stl_mesh
from shapely.geometry import Polygon
from shapely.ops import triangulate
import argparse
import os
import pyvista as pv

def extrude_contours(contours, thickness, dpmm):
    all_facets = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
            
        # 1. Scale to mm and map (row, col) -> (x, y)
        # scipy/skimage returns (y, x), we want (x, y)
        points_2d = np.column_stack((contour[:, 1] / dpmm, contour[:, 0] / dpmm))
        
        # 2. Create Side Walls
        for i in range(len(points_2d) - 1):
            p1, p2 = points_2d[i], points_2d[i+1]
            # Wall Triangle 1
            all_facets.append([[p1[0], p1[1], 0], [p2[0], p2[1], 0], [p1[0], p1[1], thickness]])
            # Wall Triangle 2
            all_facets.append([[p2[0], p2[1], 0], [p2[0], p2[1], thickness], [p1[0], p1[1], thickness]])

        # 3. Create Caps using Shapely (Ear Clipping / Triangulation)
        poly = Polygon(points_2d)
        if not poly.is_valid:
            poly = poly.buffer(0) # Fix self-intersections if any
            
        # Triangulate the polygon surface
        # Note: get_parts/interiors handles holes if we had them grouped
        from shapely.ops import triangulate
        triangles = triangulate(poly)
        
        for tri in triangles:
            # Only keep triangles that are actually inside our polygon
            if poly.contains(tri.centroid):
                coords = list(tri.exterior.coords)
                v1, v2, v3 = coords[0], coords[1], coords[2]
                
                # Bottom Cap (Z=0)
                all_facets.append([[v1[0], v1[1], 0], [v3[0], v3[1], 0], [v2[0], v2[1], 0]])
                # Top Cap (Z=thickness)
                all_facets.append([[v1[0], v1[1], thickness], [v2[0], v2[1], thickness], [v3[0], v3[1], thickness]])

    # 4. Create the final STL mesh
    data = np.zeros(len(all_facets), dtype=stl_mesh.Mesh.dtype)
    for i, f in enumerate(all_facets):
        data['vectors'][i] = np.array(f)
        
    return stl_mesh.Mesh(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Gerber file and extract contours")
    parser.add_argument("input_file", help="Path to the input Gerber file")
    parser.add_argument("-o", "--output", help="Path to output STL file (optional)", default="output.stl")
    parser.add_argument("--dpmm", type=int, default=50, help="Dots per millimeter for rasterization")
    parser.add_argument("--thickness", type=float, default=1.0, help="Thickness of the extruded contours")
    args = parser.parse_args()

    input_file = args.input_file
    dpmm = args.dpmm
    thickness = args.thickness
    output_file = args.output

    # 1. Load and parse the Gerber file
    gerber = GerberFile.from_file(input_file)
    parsed = gerber.parse()
    
    # 2. Render to a temporary PNG (Grayscale: Copper=255, Background=0)
    temp_render = "temp_render.png"
    parsed.render_raster(temp_render, color_scheme=ColorScheme.DEFAULT_GRAYSCALE, dpmm=dpmm)
    
    # 3. Load image and convert to binary mask
    img_pil = Image.open(temp_render).convert("L")
    img_raw = np.asarray(img_pil)
    
    # Thresholding: 1 for copper, 0 for background
    binary_mask_raw = img_raw > 127
    
    # 4. Add Padding (extension)
    # Adding a 2mm border (expressed in pixels)
    pad_px = int(2 * dpmm) 
    binary_mask = np.pad(binary_mask_raw, pad_width=pad_px, mode='constant', constant_values=0)
    
    # 5. Extract contours using Marching Squares
    # result is a list of (N, 2) arrays (row, col) coordinates
    contours = measure.find_contours(binary_mask, level=0.5)
    
    # Clean up temp file
    if os.path.exists(temp_render):
        os.remove(temp_render)

    # 6. Plotting
    plt.figure(figsize=(12, 12))
    # Display the padded mask so contours align perfectly
    plt.imshow(binary_mask, cmap="gray", origin="upper")
    
    for contour in contours:
        # contour is (y, x), so we plot (x, y)
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color="red")

    plt.title(f"Extracted {len(contours)} Contours with Padding ({pad_px}px)")
    plt.axis("equal")
    plt.show()

    print(f"Padding added: {pad_px} pixels.")
    print(f"Found {len(contours)} separate contour loops.")

    # Generate Mesh
    out_mesh = extrude_contours(contours, args.thickness, args.dpmm)
    out_mesh.save(output_file)
    print(f"STL saved to: {output_file}")

    # 3D Plotting Fix
    try:
        # Load the STL file we just saved
        mesh_pv = pv.read(args.output)
        
        # Create a plotter window
        plotter = pv.Plotter(title="3D Extrusion Preview")
        plotter.add_mesh(mesh_pv, color="gold", show_edges=True, smooth_shading=True)
        plotter.add_axes()
        plotter.show_grid()
        
        print("Opening 3D viewer...")
        plotter.show()
    except Exception as e:
        print(f"Could not open PyVista viewer: {e}")