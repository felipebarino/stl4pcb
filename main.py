import numpy as np
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme
from PIL import Image
import argparse
import os
import logging
from methods.multipolygon import get_polygons, extrude_multipolygon
from methods.pixel import create_pixel_mesh

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("main")

def main():
    """
    Main execution flow:
    1. Parse arguments.
    2. Convert Gerber to a binary raster mask.
    3. Generate 3D geometry using the selected algorithm.
    4. Save STL and preview.
    """
    args = arg_parse()

    logger.info(f"Processing input: {args.input_file}")
    logger.info(f"Configuration: Method={args.method}, DPMM={args.dpmm}, Thickness={args.thickness}mm")

    # 1. Get binary mask from Gerber
    # This step rasterizes the vector Gerber data into a high-res pixel grid.
    binary_mask = get_raster_mask(args.input_file, args.dpmm)

    # 2. Generate 3D Mesh based on selected method
    final_mesh = None
    try:
        if args.method == 'pixel':
            logger.info("Starting Voxel/Pixel mesh generation...")
            final_mesh = create_pixel_mesh(binary_mask, args.thickness, args.dpmm)
        else:
            logger.info("Starting MultiPolygon extrusion...")
            # Step A: Extract vector contours from the raster mask
            polygons = get_polygons(binary_mask, args.dpmm)
            # Step B: Extrude the purified 2D geometry into 3D
            final_mesh = extrude_multipolygon(polygons, args.thickness)
    except Exception as e:
        logger.critical(f"Mesh generation failed unrecoverably: {e}", exc_info=True)
        return # Exit gracefully without crashing the interpreter

    # 3. Save output
    if final_mesh is not None and len(final_mesh.vectors) > 0:
        final_mesh.save(args.output)
        logger.info(f"STL successfully saved to: {args.output}")
        logger.info(f"Statistics: {len(final_mesh.vectors)} total facets generated.")
    else:
        logger.warning("Mesh result is empty! No file was saved.")
        return

    # 4. Optional Visualization
    if not args.no_preview:
        try:
            display_preview(args.output)
        except ImportError:
            logger.warning("PyVista library not found. Skipping 3D preview.")
        except Exception as e:
            logger.warning(f"Failed to initialize 3D viewer: {e}")

def arg_parse():
    """
    Parses command line arguments and configures global logging levels.
    """
    parser = argparse.ArgumentParser(description="Gerber to STL Converter (Precision PCB Extrusion)")
    parser.add_argument("input_file", help="Path to input Gerber file (.gbr)")
    parser.add_argument("-o", "--output", help="Path to output STL file", default="output.stl")
    parser.add_argument("--method", choices=['multipolygon', 'pixel'], default='multipolygon', 
                        help="Algorithm: 'multipolygon' (smoother, best for traces) or 'pixel' (robust, best for complex fills)")
    parser.add_argument("--dpmm", type=int, default=60, help="Resolution in Dots Per Millimeter (higher = smoother but slower)")
    parser.add_argument("--thickness", type=float, default=0.035, help="Extrusion thickness in mm (default 35um for 1oz copper)")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging")
    parser.add_argument("--no-preview", action="store_true", help="Disable the 3D preview window after processing")

    args = parser.parse_args()

    if args.verbose:
        # Set root logger to debug to capture logs from imported modules too
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    return args

def get_raster_mask(file_path, dpmm):
    """
    Parses a Gerber file and renders it into a boolean numpy array.
    
    Args:
        file_path (str): Path to the .gbr file.
        dpmm (int): Dots Per Millimeter (resolution).
        
    Returns:
        np.ndarray: A boolean mask where True=Copper, False=Background.
    """
    temp_render_file = "temp_render.png"
    
    # 1. Parse Gerber File
    try:
        gerber = GerberFile.from_file(file_path)
        parsed = gerber.parse()
    except FileNotFoundError:
        logger.critical(f"Input file not found: {file_path}")
        exit(1)
    except Exception as e:
        logger.critical(f"Error parsing Gerber format: {e}")
        exit(1)

    # 2. Render to Raster
    try:
        logger.debug(f"Rendering raster at {dpmm} DPMM...")
        parsed.render_raster(temp_render_file, color_scheme=ColorScheme.DEFAULT_GRAYSCALE, dpmm=dpmm)
        
        # Load Image
        # Convert to Grayscale ('L') -> 0..255
        img_pil = Image.open(temp_render_file).convert("L")
        img_arr = np.asarray(img_pil)
        
        # Create Boolean Mask (Thresholding)
        # Copper is typically White (255) in standard Gerber renders
        binary_mask = img_arr > 127
        
        # Cleanup temp file directly
        img_pil.close()
        if os.path.exists(temp_render_file): 
            os.remove(temp_render_file)
            
        # Pad the mask with 0s (Background)
        # This ensures that traces touching the edge of the Gerber bounding box
        # are closed loops, preventing 'open manifold' errors during extrusion.
        binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        
        logger.info(f"Raster generated. Dimensions: {binary_mask.shape} pixels.")
        return binary_mask
        
    except Exception as e:
        logger.critical(f"Failed during rasterization process: {e}")
        if os.path.exists(temp_render_file): os.remove(temp_render_file)
        exit(1)

def display_preview(file_path):
    """
    Opens a PyVista 3D interactive window to view the generated STL.
    """
    import pyvista as pv
    if os.environ.get("SSH_CONNECTION") or os.environ.get("REMOTE_CONTAINERS"):
        logger.warning("Headless environment detected; skipping GUI window.")
        return
        
    logger.info("Opening 3D viewer (close window to exit)...")
    mesh = pv.read(file_path)
    
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="gold", show_edges=False, metallic=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.title = f"STL Preview: {os.path.basename(file_path)}"
    plotter.show()

if __name__ == "__main__":
    main()