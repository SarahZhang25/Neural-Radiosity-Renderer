"""
Dataset generation script for radiosity rendering.
Generates 120 cases: 12 shapes × 2 colors × 5 rotations
"""

import os
import sys
import json
import random
import subprocess
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Add parent directory to path (radiosity module)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# RADIOSITY_DIR = os.path.dirname(SCRIPT_DIR)
RADIOSITY_DIR = SCRIPT_DIR
sys.path.insert(0, RADIOSITY_DIR)

from scene import create_cornell_box_with_car, Scene, CORNELL_WHITE
from solver import solve_progressive
from renderer import Camera
from patches import Patch
from utils import save_image

# Import Mitsuba renderer
from renderer_mitsuba import render_mitsuba, Camera as MitsubaCamera
MITSUBA_AVAILABLE = True

# Configuration
SHAPENET_ROOT = "/home/sazhang/.cache/kagglehub/datasets/hajareddagni/shapenetcorev2/versions/1/ShapeNetCore.v2/ShapeNetCore.v2"
# TETWILD_PATH = os.path.join(RADIOSITY_DIR, "../TetWild/build/TetWild")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
SURFACE_MESHES_DIR = os.path.join(SCRIPT_DIR, "output/raw_meshes")  # Normalized meshes (unit cube, zero-centered)
TET_MESHES_DIR = os.path.join(SCRIPT_DIR, "output/tet_meshes")

# ShapeNet class IDs
CLASS_IDS = {
    # "02691156": "airplane",
    "02958343": "car",
    # "04379243": "table",
    # "04530566": "vessel"
}

# Number of shapes per class
SHAPES_PER_CLASS = 1 #120

# Realistic car paint colors (avoiding red/green to not blend with Cornell box walls)
PREDEFINED_COLORS = [
    # np.array([0.7, 0.7, 0.72]),    # Metallic Silver
    # np.array([0.08, 0.08, 0.08]),  # Glossy Black
    # np.array([0.1, 0.2, 0.55]),    # Deep Blue Metallic
    # np.array([0.9, 0.9, 0.88]),    # Pearl White
    # np.array([0.25, 0.25, 0.28]),  # Gunmetal Gray
    # np.array([0.75, 0.55, 0.3]),   # Champagne Gold
    # np.array([0.85, 0.5, 0.1]),    # Orange Metallic
    # np.array([0.4, 0.2, 0.5]),     # Purple Metallic
    # np.array([0.2, 0.4, 0.6]),     # Sky Blue
    # np.array([0.9, 0.75, 0.1]),    # Yellow Gold
    np.array([0.7, 0.1, 0.1]),     # Bright Red
]

# Rotations around up-axis (in degrees)
NUM_ROTATIONS = 1

# Number of colors per shape
NUM_COLORS = 1


MAX_TRIANGLES = 100000  # Skip and resample meshes with more triangles


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

            obj_path = os.path.join(class_dir, shape_id, "models", "model_normalized.ply")#obj")
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


def copy_mesh(input_obj: str, output_obj: str) -> bool:
    """Copy mesh from source to destination.

    Note: User will run TetWild processing separately. This just copies the
    original mesh for now. Replace with processed mesh later.

    Args:
        input_obj: Path to input OBJ file
        output_obj: Path to save the mesh

    Returns:
        True if successful
    """
    try:
        shutil.copy(input_obj, output_obj)
        return True
    except Exception as e:
        print(f"    Copy error: {e}")
        return False


def render_single_case(args):
    """Worker function for parallel rendering. Takes a tuple of case parameters."""
    case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, renders_dir = args

    try:
        # Create scene
        scene = create_scene_with_object(
            mesh_path=mesh_path,
            object_color=color,
            rotation_deg=rotation,
            light_intensity=50.0,
            car_scale=1.0
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

        data = render_and_save(scene, output_prefix, params, object_color=color)
        params["num_object_patches"] = data["num_object_patches"]
        params["num_wall_patches"] = data["num_wall_patches"]

        return (case_idx, params, None)
    except Exception as e:
        import traceback
        return (case_idx, None, str(e) + "\n" + traceback.format_exc())


def create_scene_with_object(
    mesh_path: str,
    object_color: np.ndarray,
    rotation_deg: float,
    box_size: float = 2.0,
    light_intensity: float = 12.0,
    subdivisions: int = 8,  # Subdivision for walls/floor
    car_scale: float = 1.0
) -> Scene:
    """Create Cornell box scene with rotated object - uses same setup as main.py."""

    # Use the same function as main.py for consistency
    scene = create_cornell_box_with_car(
        mesh_path=mesh_path,
        box_size=box_size,
        car_scale=car_scale,
        car_height_offset=0.0,  # Car on floor (same as main.py)
        light_intensity=light_intensity,
        simplify_ratio=1.0,  # No simplification (same as main.py)
        car_color=object_color,
        subdivisions=subdivisions,
        rotate_car=rotation_deg,  # Apply rotation
    )

    return scene


def render_and_save(
    scene: Scene,
    output_prefix: str,
    params: Dict,
    exposure: float = 1.0,
    object_color: np.ndarray = None
) -> Dict:
    """Render scene and save radiosity data."""

    # Build mesh
    scene.build_combined_mesh()
    mesh = scene.get_trimesh()

    # Solve radiosity with shadows
    solve_progressive(scene.patches, mesh, max_iterations=20)

    # Setup camera - match main.py settings
    box_size = 2.0

    camera = MitsubaCamera(
        position=np.array([0.0, box_size * 0.5, box_size * 1.5]),
        look_at=np.array([0.0, box_size * 0.5, 0.0]),
        fov=50.0,
        width=1024,
        height=1024
    )
    # Render with Mitsuba - match main.py settings
    image = render_mitsuba(
        scene, camera,
        object_color=object_color if object_color is not None else np.array([0.7, 0.1, 0.1]),
        spp=256
    )

    # Save image using same function as main.py
    save_image(image, f"{output_prefix}_render.png", tone_mapping="reinhard", exposure=exposure)

    # Extract radiosity data
    # Separate object patches from wall patches based on material color
    object_radiosity = []
    wall_radiosity = []
    object_vertices = []
    wall_vertices = []

    for patch in scene.patches:
        is_wall = (
            np.allclose(patch.reflectance, CORNELL_WHITE, atol=0.1) or
            np.allclose(patch.reflectance, np.array([0.65, 0.05, 0.05]), atol=0.1) or  # Red
            np.allclose(patch.reflectance, np.array([0.12, 0.45, 0.15]), atol=0.1) or  # Green
            np.sum(patch.emission) > 0  # Light
        )

        if is_wall:
            wall_radiosity.append(patch.radiosity.tolist())
            wall_vertices.append(patch.center.tolist())
        else:
            object_radiosity.append(patch.radiosity.tolist())
            object_vertices.append(patch.center.tolist())

    # Save data
    data = {
        "params": params,
        "object_radiosity": object_radiosity,
        "object_vertices": object_vertices,
        "wall_radiosity": wall_radiosity,
        "wall_vertices": wall_vertices,
        "num_object_patches": len(object_radiosity),
        "num_wall_patches": len(wall_radiosity)
    }

    with open(f"{output_prefix}_data.json", "w") as f:
        json.dump(data, f, indent=2)

    # Also save as numpy for easier loading
    np.savez(
        f"{output_prefix}_data.npz",
        object_radiosity=np.array(object_radiosity),
        object_vertices=np.array(object_vertices),
        wall_radiosity=np.array(wall_radiosity),
        wall_vertices=np.array(wall_vertices),
        color=params["color"],
        rotation=params["rotation"]
    )

    return data


def main():
    print("=" * 60)
    print("Dataset Generation for Radiosity Rendering")
    print("=" * 60)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SURFACE_MESHES_DIR, exist_ok=True)
    os.makedirs(TET_MESHES_DIR, exist_ok=True)

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
    print(f"      Total cases: {total_cases}")

    renders_dir = os.path.join(OUTPUT_DIR, "renders")
    os.makedirs(renders_dir, exist_ok=True)

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
                render_cases.append((case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, renders_dir))

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
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"  Total renders: {len(all_metadata)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
