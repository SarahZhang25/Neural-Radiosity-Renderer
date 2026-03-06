
"""
Generate dataset using pure Mitsuba 3 rendering.
Ignores radiosity patch generation completely.
"""

import os
import sys
import numpy as np
import trimesh
from tqdm import tqdm
import json
import multiprocessing
import random 
import time
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Add parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from pure_mitsuba_utils import render_pure_mitsuba_scene, Camera
from utils import save_image

# Constants
SHAPENET_ROOT = "/home/sazhang/ShapeNetCorev2"
# OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
# SURFACE_MESHES_DIR = os.path.join(SCRIPT_DIR, "output/raw_meshes")  # Normalized meshes (unit cube, zero-centered)
# TET_MESHES_DIR = os.path.join(SCRIPT_DIR, "output/tet_meshes")


OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
# RENDER_DIR = os.path.join(OUTPUT_ROOT, "renders")
RENDER_DIR = os.path.join(OUTPUT_ROOT, "datasets/data_generation/output/datasets/shapenet_take3_table_vessel_chair_guitar_motorbike")
MESH_SOURCE_DIR = os.path.join(OUTPUT_ROOT, "raw_meshes", "simple_objects")
os.makedirs(RENDER_DIR, exist_ok=True)

# CLASS_IDS = {
#     # "cube": "cube",
#     # "sphere": "sphere",
#     # "torus": "torus",
#     # "cylinder": "cylinder"
#     "shapenet_lamp": "shapenet_lamp"
# }
# ShapeNet class IDs
CLASS_IDS = {
    # "02691156": "airplane",
    # "02958343": "car",
    "04379243": "table",
    "04530566": "vessel",
    "03001627": "chair",
    # "03636649": "lamp", # Excluded due to small size and low comparative complexity
    "03467517": "guitar",
    "03790512": "motorbike"
}

PREDEFINED_COLORS = [
    np.array([0.2, 0.4, 0.6]), # blue
    # np.array([0.7, 0.1, 0.1]), # red
    np.array([0.9, 0.75, 0.1]), # gold
    # np.array([0.9, 0.8, 0.2]),  # yellow
    np.array([0.6, 0.3, 0.8]),  # purple
    np.array([0.8, 0.8, 0.8]),  # gray
    np.array([0.9, 0.9, 0.9]),  # white
    np.array([0.9, 0.5, 0.1]),  # orange
]

# Rotations around up-axis (in degrees)
NUM_ROTATIONS = 3

# Number of colors per shape
NUM_COLORS = 3

# Number of shapes per class
SHAPES_PER_CLASS = 15 #120
MAX_TRIANGLES = 100000  # Skip and resample meshes with more triangles

# Scene Settings
BOX_SIZE = 2.0
SPP = 2048 # High quality sample count
LIGHT_INTENSITY = 10.0
LIGHT_SIZE_RATIO = 0.5  # Fraction of ceiling covered by the emitter (larger = softer shadows, more light)

# Camera
CAMERA = Camera(
    position=np.array([0.0, BOX_SIZE * 0.5, BOX_SIZE * 1.5]),
    look_at=np.array([0.0, BOX_SIZE * 0.5, 0.0]),
    fov=50.0,
    width=256,
    height=256
)

def select_random_shapes(num_per_class: int = 4) -> List[Tuple[str, str, str]]:
    """Select random shapes from each class.

    Returns:
        List of (class_id, shape_id, obj_path) tuples
    """
    selected = []

    for class_id, class_name in CLASS_IDS.items():
        class_dir = os.path.join(SHAPENET_ROOT, class_id)
        shape_ids = [d for d in os.listdir(class_dir)
                     if os.path.isdir(os.path.join(class_dir, d)) and not d.startswith('.')]
        print(f"Selecting shapes for class {class_name} ({class_id}): total available = {len(shape_ids)}")

        # Randomly select shapes
        random.seed(42)  # For reproducibility
        random.shuffle(shape_ids)

        # Select shapes, skipping those with too many triangles
        chosen_count = 0
        for shape_id in shape_ids:
            if chosen_count >= num_per_class:
                print("  Reached desired number of shapes for this class.")
                break

            obj_path = os.path.join(class_dir, shape_id, "models", "model_normalized.obj")
            if not os.path.exists(obj_path):
                print(f"  Skipping {obj_path}: OBJ file not found")
                continue

            # Check triangle count
            try:
                mesh = trimesh.load(obj_path, force='mesh')
                num_triangles = len(mesh.faces)
                if num_triangles > MAX_TRIANGLES:
                    print(f"  Skipping {class_name}/{shape_id}: {num_triangles} triangles > {MAX_TRIANGLES}")
                    continue
            except Exception as e:
                print(f"  Skipping {class_name}/{shape_id}: failed to load mesh - {e}")
                continue

            selected.append((class_id, shape_id, obj_path))
            print(f"  Selected: {class_name}/{shape_id} ({num_triangles} triangles)")
            chosen_count += 1

    return selected


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
    points, face_indices = trimesh.sample.sample_surface(combined, n_points)
    normals = combined.face_normals[face_indices]
    return points, normals


# def process_case(case_idx, shape_name, color, rotation, light_intensity: float, light_size_ratio: float, exposure: float, scale_min: float, scale_max: float, pos_variation: float, random_seed: int):

def render_single_case(args):
    """Worker function for parallel rendering. Takes a tuple of case parameters."""
    case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, renders_dir = args

    try:
        # Create scene
        # Set random seed for reproducibility
        random_seed = 42
        np.random.seed(random_seed + case_idx)
        
        # print(f"Generating Case {case_idx}: {class_id}, rot={rotation:.1f}...")
        
        # 1. Load Object Mesh
        # mesh_path = os.path.join(MESH_SOURCE_DIR, f"{shape_name}.obj")
        if not os.path.exists(mesh_path):
            print(f"Skipping: {mesh_path} not found")
            return

        mesh = trimesh.load(mesh_path, force='mesh')
            
        # 2. Transform Object (Scale, Rotate, Position)
        # Random scale between scale_min and scale_max of box size
        scale_ratio = .75 # np.random.uniform(scale_min, scale_max)
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
        translation = np.zeros(3)
        translation[1] = -bounds[0][1] # Move bottom to 0
        
        # Center XZ with random variation:
        # translation[0] = - (bounds[0][0] + bounds[1][0]) / 2.0 + np.random.uniform(-pos_variation, pos_variation) * BOX_SIZE
        # translation[2] = - (bounds[0][2] + bounds[1][2]) / 2.0 + np.random.uniform(-pos_variation, pos_variation) * BOX_SIZE
        
        # # Optional: Add small Y variation (keep object above floor)
        # translation[1] += np.random.uniform(0, pos_variation * 0.5) * BOX_SIZE
        
        mesh.apply_translation(translation)
        
        # 3. Create Scene Meshes
        box_meshes = create_cornell_box_meshes(BOX_SIZE)#, light_size_ratio=light_size_ratio)
        
        # 4. Render
        image_tensor = render_pure_mitsuba_scene(
            box_meshes, mesh, color, CAMERA, spp=SPP#, light_intensity=light_intensity
        )
        # Convert the Mitsuba tensor directly into a standard NumPy float32 array
        hdr_image_np = np.array(image_tensor, dtype=np.float32)
        
        # 5. Save Data for Model
        fn_prefix = f"case_{case_idx:03d}_{class_id}_{shape_id}_c{color_idx}_r{rot_idx}"
        npz_path = os.path.join(RENDER_DIR, f"{fn_prefix}.npz")
        png_path = os.path.join(RENDER_DIR, f"{fn_prefix[:-5]}_render.png") # strip _data
        
        save_image(image_tensor, png_path, tone_mapping="reinhard", exposure=1)
        
        # Point Clouds
        obj_pc, obj_face_idx = trimesh.sample.sample_surface(mesh, 2048)
        obj_normals = mesh.face_normals[obj_face_idx]
        
        wall_pc, wall_normals = get_walls_point_cloud(box_meshes, 2048)
        
        np.savez(
            npz_path,
            object_vertices=obj_pc,
            object_normals=obj_normals,
            wall_vertices=wall_pc,
            wall_normals=wall_normals,
            color=color,
            rotation=rotation,
            scale_ratio=1.,#scale_ratio,
            position_offset=None,#translation,
            # Dummy fields for compatibility
            object_radiosity=np.zeros_like(obj_pc),
            wall_radiosity=np.zeros_like(wall_pc),
            camera_pos=CAMERA.position,
            camera_lookat=CAMERA.look_at
        )

        # Params
        params = {
            "class_id": class_id,
            "class_name": shape_id,
            "shape_id": shape_id,
            "color": color.tolist(),
            "color_idx": color_idx,
            "rotation": rotation,
            "rotation_idx": rot_idx
        }

        # Render and save
        output_prefix = os.path.join(
            renders_dir,
            f"case_{case_idx:03d}_{class_id}_{shape_id}_c{color_idx}_r{rot_idx}"
        )

        # data = render_and_save(scene, output_prefix, params, object_color=color)
        # params["num_object_patches"] = data["num_object_patches"]
        # params["num_wall_patches"] = data["num_wall_patches"]

        return (case_idx, params, None)
    except Exception as e:
        import traceback
        return (case_idx, None, str(e) + "\n" + traceback.format_exc())



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
    obj_pc, obj_face_idx = trimesh.sample.sample_surface(mesh, 2048)
    obj_normals = mesh.face_normals[obj_face_idx]
    
    wall_pc, wall_normals = get_walls_point_cloud(box_meshes, 2048)
    
    np.savez(
        npz_path,
        object_vertices=obj_pc,
        object_normals=obj_normals,
        wall_vertices=wall_pc,
        wall_normals=wall_normals,
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
    parser.add_argument('--num_rotations', type=int, default=NUM_ROTATIONS, 
                      help="Number of rotation angles to generate (0 to 360 degrees)")
    
    # Control render quality
    parser.add_argument('--spp', type=int, default=SPP, 
                      help=f"Samples per pixel (default: {SPP})")
    parser.add_argument('--light_intensity', type=float, default=LIGHT_INTENSITY, help='Ceiling light radiance')
    parser.add_argument('--light_size_ratio', type=float, default=LIGHT_SIZE_RATIO, help='Ceiling light size as fraction of box size (0-1)')
    parser.add_argument('--exposure', type=float, default=1.2, help='Tone mapping exposure multiplier')
    # TODO: actually use this in rendering, and control object variation
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
    
    start_time = time.time()
    print("=" * 60)
    print("Dataset Generation for Neural Rendering")
    print("=" * 60)

    # Create output directories
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # os.makedirs(SURFACE_MESHES_DIR, exist_ok=True)
    # os.makedirs(TET_MESHES_DIR, exist_ok=True)

    # Step 1: Find all processed meshes in SURFACE_MESHES_DIR
    print("\n[1/4] Finding processed meshes...")
    # processed_shapes = []
    # for filename in sorted(os.listdir(SURFACE_MESHES_DIR)):
    #     if filename.endswith('.obj'):
    #         # Use filename (without .obj) as shape name
    #         shape_name = filename.replace('.obj', '')
    #         mesh_path = os.path.join(SURFACE_MESHES_DIR, filename)
    #         print(f"  Found: {shape_name}")
    #         processed_shapes.append((shape_name, shape_name, mesh_path))
            
    processed_shapes = select_random_shapes(num_per_class=SHAPES_PER_CLASS)

    # Step 2: Summary
    print(f"\n[2/4] Loaded {len(processed_shapes)} meshes")

    # Step 3: Generate colors and rotations
    print("\n[3/4] Generating render configurations...")

    # Generate random rotations
    random.seed(123)
    rotations = [random.uniform(0, 360) for _ in range(NUM_ROTATIONS)]
    print(f"      Rotations: {[f'{r:.1f}°' for r in rotations]}")

    # Assign NUM_COLORS colors per shape
    shape_colors = {}
    for i, (class_id, shape_id, _) in enumerate(processed_shapes):
        key = f"{class_id}_{shape_id}"
        # Pick NUM_COLORS different colors
        color_indices = random.sample(range(len(PREDEFINED_COLORS)), NUM_COLORS)
        shape_colors[key] = [PREDEFINED_COLORS[idx] for idx in color_indices]


    # Step 4: Generate all renders
    print("\n[4/4] Generating renders...")
    total_cases = len(processed_shapes) * NUM_COLORS * NUM_ROTATIONS
    print(f"Settings:")
    print(f"  Shapes: {args.shapes}")
    print(f"  Rotations: {len(rotations)} steps")
    print(f"  Samples per pixel: {SPP}")
    print(f"  Light size ratio: {LIGHT_SIZE_RATIO}")
    print(f"  Light intensity: {LIGHT_INTENSITY}")
    print(f"  Exposure: {exposure}")
    # print(f"  Scale range: {args.scale_min} - {args.scale_max}")
    # print(f"  Position variation: {args.pos_variation}")
    print(f"  Random seed: {args.random_seed}")
    # print(f"  Total images to generate: {len(args.shapes) * len(rotations) * len(PREDEFINED_COLORS)}")

    print(f"  Total cases: {total_cases}")
    
    # Build list of all render cases with args for worker function
    render_cases = []
    case_idx = 0
    for class_id, shape_id, mesh_path in processed_shapes:
        key = f"{class_id}_{shape_id}"
        colors = shape_colors[key]
        for color_idx, color in enumerate(colors):
            for rot_idx, rotation in enumerate(rotations):
                case_idx += 1
                # Pack all args for worker: (case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, renders_dir)
                render_cases.append((case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, RENDER_DIR))

    all_metadata = []

    # Parallel rendering
    n_workers = min(multiprocessing.cpu_count(), len(render_cases))
    n_workers = 4
    print(f"      Using {n_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(render_single_case, case): case[0] for case in render_cases}

        # Progress bar with as_completed
        with tqdm(total=len(render_cases), desc="Rendering", unit="case") as pbar:
            for future in as_completed(futures):
                case_idx = futures[future]
                try:
                    idx, params, error = future.result()
                    if error:
                        tqdm.write(f"Error in case {idx}: {error}")
                    elif params:
                        all_metadata.append(params)
                except Exception as e:
                    tqdm.write(f"Exception in case {case_idx}: {e}")
                pbar.update(1)

    # Save master metadata
    metadata_path = os.path.join(OUTPUT_ROOT, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"  Total renders: {len(all_metadata)}")
    print(f"  Output directory: {OUTPUT_ROOT}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("=" * 60)


    # # Generate variations
    # for shape in args.shapes:
    #     if shape not in CLASS_IDS:
    #         print(f"Warning: Shape '{shape}' not found in defined classes. Skipping.")
    #         continue
            
    #     for rot in rotations:
    #          # Iterate through all predefined colors for every rotation
    #         for col in PREDEFINED_COLORS:
    #             process_case(case_count, shape, col, rot, LIGHT_INTENSITY, LIGHT_SIZE_RATIO, exposure, args.scale_min, args.scale_max, args.pos_variation, args.random_seed)
    #             case_count += 1

