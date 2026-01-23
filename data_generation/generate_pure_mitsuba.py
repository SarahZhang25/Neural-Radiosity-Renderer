
"""
Generate dataset using pure Mitsuba 3 rendering.
Ignores radiosity patch generation completely.
"""

import os
import sys
import numpy as np
import trimesh
from tqdm import tqdm
import multiprocessing

# Add parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from pure_mitsuba_utils import render_pure_mitsuba_scene, Camera
from utils import save_image

# Constants
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
RENDER_DIR = os.path.join(OUTPUT_ROOT, "renders")
MESH_SOURCE_DIR = os.path.join(OUTPUT_ROOT, "raw_meshes", "simple_objects")
os.makedirs(RENDER_DIR, exist_ok=True)

CLASS_IDS = {
    "cube": "cube",
    "sphere": "sphere",
    "torus": "torus",
    "cylinder": "cylinder"
}

PREDEFINED_COLORS = [
    np.array([0.2, 0.4, 0.6]), # blue
    np.array([0.7, 0.1, 0.1]), # red
    np.array([0.4, 0.2, 0.5]), # purple
    np.array([0.9, 0.75, 0.1]), # gold
]

# Scene Settings
BOX_SIZE = 2.0
SPP = 2048 # High quality sample count
LIGHT_INTENSITY = 40.0
LIGHT_SIZE_RATIO = 0.5  # Fraction of ceiling covered by the emitter (larger = softer shadows, more light)

# Camera
CAMERA = Camera(
    position=np.array([0.0, BOX_SIZE * 0.5, BOX_SIZE * 1.5]),
    look_at=np.array([0.0, BOX_SIZE * 0.5, 0.0]),
    fov=50.0,
    width=256,
    height=256
)

def create_cornell_box_meshes(size=2.0, light_size_ratio: float = LIGHT_SIZE_RATIO) -> dict:
    """Create Trimesh objects for Cornell Box walls."""
    half = size / 2.0
    meshes = {}
    
    def quad(v0, v1, v2, v3):
        return trimesh.Trimesh(vertices=[v0, v1, v2, v3], faces=[[0, 1, 2], [0, 2, 3]])

    # Floor (y=0) - White
    meshes['floor'] = quad(
        [-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]
    )
    # Ceiling (y=size) - White
    meshes['ceiling'] = quad(
        [-half, size, -half], [half, size, -half], [half, size, half], [-half, size, half]
    )
    # Back (z=-half) - White
    meshes['back'] = quad(
        [-half, 0, -half], [half, 0, -half], [half, size, -half], [-half, size, -half]
    )
    # Left (x=-half) - Red
    meshes['left'] = quad(
        [-half, 0, half], [-half, 0, -half], [-half, size, -half], [-half, size, half]
    )
    # Right (x=half) - Green
    meshes['right'] = quad(
        [half, 0, -half], [half, 0, half], [half, size, half], [half, size, -half]
    )
    
    # Light (ceiling area emitter) - oriented to face DOWN (-Y)
    light_size = size * light_size_ratio
    l_half = light_size / 2.0
    light_y = size - 0.005 # epsilon
    # Winding chosen so the normal points downward (-Y) for a one-sided area emitter
    meshes['light'] = quad(
        [-l_half, light_y, -l_half], [l_half, light_y, -l_half],
        [l_half, light_y,  l_half], [-l_half, light_y,  l_half]
    )
    
    return meshes

def get_walls_point_cloud(box_meshes, n_points=2048):
    """Sample points from wall meshes for dataset."""
    # Concatenate visual walls
    combined = trimesh.util.concatenate([
        box_meshes['floor'], box_meshes['ceiling'], 
        box_meshes['back'], box_meshes['left'], box_meshes['right']
    ])
    return combined.sample(n_points)


def process_case(case_idx, shape_name, color, rotation, light_intensity: float, light_size_ratio: float, exposure: float, scale_min: float, scale_max: float, pos_variation: float, random_seed: int):
    # Set random seed for reproducibility
    np.random.seed(random_seed + case_idx)
    
    print(f"Generating Case {case_idx}: {shape_name}, rot={rotation:.1f}...")
    
    # 1. Load Object Mesh
    mesh_path = os.path.join(MESH_SOURCE_DIR, f"{shape_name}.obj")
    if not os.path.exists(mesh_path):
        print(f"Skipping {shape_name}: {mesh_path} not found")
        return

    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Fix normals immediately after loading
    mesh.fix_normals()
    if not mesh.is_winding_consistent:
        mesh.invert()
        
    # 2. Transform Object (Scale, Rotate, Position)
    # Random scale between scale_min and scale_max of box size
    scale_ratio = np.random.uniform(scale_min, scale_max)
    target_size = BOX_SIZE * scale_ratio
    bounds = mesh.bounds
    current_size = np.max(bounds[1] - bounds[0])
    if current_size > 0:
        scale_factor = target_size / current_size
        mesh.apply_scale(scale_factor)
    
    # Rotate Y
    if rotation != 0:
        rot_mat = trimesh.transformations.rotation_matrix(np.radians(rotation), [0, 1, 0])
        mesh.apply_transform(rot_mat)
        
    # Position: Center horizontally, Bottom on floor
    bounds = mesh.bounds
    object_height = bounds[1][1] - bounds[0][1]
    floor_y = 0 + (object_height / 2) # mesh center
    # Wait, apply_translation moves centroid.
    # We want bottom of bounding box at y=0.
    # Current centroid y: mesh.centroid[1]
    # Current bottom y: bounds[0][1]
    # We want bounds[0][1] = 0.
    # diff = 0 - bounds[0][1].
    translation = np.zeros(3)
    translation[1] = -bounds[0][1] # Move bottom to 0
    
    # Center XZ with random variation:
    translation[0] = - (bounds[0][0] + bounds[1][0]) / 2.0 + np.random.uniform(-pos_variation, pos_variation) * BOX_SIZE
    translation[2] = - (bounds[0][2] + bounds[1][2]) / 2.0 + np.random.uniform(-pos_variation, pos_variation) * BOX_SIZE
    
    # Optional: Add small Y variation (keep object above floor)
    translation[1] += np.random.uniform(0, pos_variation * 0.5) * BOX_SIZE
    
    mesh.apply_translation(translation)
    
    # 3. Create Scene Meshes
    box_meshes = create_cornell_box_meshes(BOX_SIZE, light_size_ratio=light_size_ratio)
    
    # 4. Render
    image = render_pure_mitsuba_scene(
        box_meshes, mesh, color, CAMERA, spp=SPP, light_intensity=light_intensity
    )
    
    # 5. Save Data for Model
    fn_prefix = f"case_{case_idx:03d}_{shape_name}_r{int(rotation)}_data"
    npz_path = os.path.join(RENDER_DIR, f"{fn_prefix}.npz")
    png_path = os.path.join(RENDER_DIR, f"{fn_prefix[:-5]}_render.png") # strip _data
    
    save_image(image, png_path, tone_mapping="reinhard", exposure=exposure)
    
    # Point Clouds
    obj_pc = mesh.sample(2048)
    wall_pc = get_walls_point_cloud(box_meshes, 2048)
    
    np.savez(
        npz_path,
        object_vertices=obj_pc,
        wall_vertices=wall_pc,
        color=color,
        rotation=rotation,
        scale_ratio=scale_ratio,
        position_offset=translation,
        # Dummy fields for compatibility
        object_radiosity=np.zeros_like(obj_pc),
        wall_radiosity=np.zeros_like(wall_pc),
        camera_pos=CAMERA.position,
        camera_lookat=CAMERA.look_at
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Allow user to specify which shapes to generate
    parser.add_argument('--shapes', nargs='+', default=list(CLASS_IDS.keys()), 
                      help=f"Specific shapes to generate. Available: {list(CLASS_IDS.keys())}")
    
    # Control number of rotations
    parser.add_argument('--num_rotations', type=int, default=5, 
                      help="Number of rotation angles to generate (0 to 360 degrees)")
    
    # Control render quality
    parser.add_argument('--spp', type=int, default=SPP, 
                      help=f"Samples per pixel (default: {SPP})")
    parser.add_argument('--light_intensity', type=float, default=LIGHT_INTENSITY, help='Ceiling light radiance')
    parser.add_argument('--light_size_ratio', type=float, default=LIGHT_SIZE_RATIO, help='Ceiling light size as fraction of box size (0-1)')
    parser.add_argument('--exposure', type=float, default=1.2, help='Tone mapping exposure multiplier')
    
    # Control object variation
    parser.add_argument('--scale_min', type=float, default=0.3, help='Minimum object scale as fraction of box size (default: 0.3)')
    parser.add_argument('--scale_max', type=float, default=0.5, help='Maximum object scale as fraction of box size (default: 0.5)')
    parser.add_argument('--pos_variation', type=float, default=0.15, help='Position variation as fraction of box size (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()

    # Update global settings based on args
    SPP = args.spp
    LIGHT_INTENSITY = args.light_intensity
    LIGHT_SIZE_RATIO = args.light_size_ratio
    exposure = args.exposure
    rotations = np.linspace(0, 360, args.num_rotations, endpoint=False)
    
    print(f"Settings:")
    print(f"  Shapes: {args.shapes}")
    print(f"  Rotations: {len(rotations)} steps")
    print(f"  Samples per pixel: {SPP}")
    print(f"  Light size ratio: {LIGHT_SIZE_RATIO}")
    print(f"  Light intensity: {LIGHT_INTENSITY}")
    print(f"  Exposure: {exposure}")
    print(f"  Scale range: {args.scale_min} - {args.scale_max}")
    print(f"  Position variation: {args.pos_variation}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Total images to generate: {len(args.shapes) * len(rotations) * len(PREDEFINED_COLORS)}")

    case_count = 1
    
    # Generate variations
    for shape in args.shapes:
        if shape not in CLASS_IDS:
            print(f"Warning: Shape '{shape}' not found in defined classes. Skipping.")
            continue
            
        for rot in rotations:
             # Iterate through all predefined colors for every rotation
            for col in PREDEFINED_COLORS:
                process_case(case_count, shape, col, rot, LIGHT_INTENSITY, LIGHT_SIZE_RATIO, exposure, args.scale_min, args.scale_max, args.pos_variation, args.random_seed)
                case_count += 1

