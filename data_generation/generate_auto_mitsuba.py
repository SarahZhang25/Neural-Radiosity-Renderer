"""
Up to date mitsuba data generation.
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
from typing import List, Tuple, Dict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import mitsuba as mi

from utils import save_image

multiprocessing.set_start_method('spawn', force=True)

# Add parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Set to 'cuda_ad_rgb' for GPU acceleration or 'llvm_ad_rgb' for CPU
mi.set_variant('cuda_ad_rgb')

# Add parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Constants
SHAPENET_ROOT = "/home/sazhang/ShapeNetCorev2"
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output_auto")
RENDER_DIR = os.path.join(OUTPUT_ROOT, "datasets/attempt6_table_chair_540")
os.makedirs(RENDER_DIR, exist_ok=True)


# ShapeNet class IDs
CLASS_IDS = {
    # "02691156": "airplane",
    # "02958343": "car",
    "04379243": "table",
    # "04530566": "vessel",
    "03001627": "chair",
    # "03636649": "lamp", # Excluded due to small size and low comparative complexity
    # "03467517": "guitar",
    # "03790512": "motorbike"
}
PREDEFINED_COLORS = [
    np.array([0.2, 0.4, 0.6]), # blue
    # # np.array([0.7, 0.1, 0.1]), # red
    # np.array([0.9, 0.75, 0.1]), # gold
    np.array([0.9, 0.8, 0.2]),  # yellow
    np.array([0.6, 0.3, 0.8]),  # purple
    np.array([0.8, 0.8, 0.8]),  # gray
    # np.array([0.9, 0.9, 0.9]),  # white
    np.array([0.9, 0.5, 0.1]),  # orange 
]

# Rotations around up-axis (in degrees)
NUM_ROTATIONS = 3

# Number of colors per shape
NUM_COLORS = 3#5

# Number of shapes per class
SHAPES_PER_CLASS = 5#20 #120
MAX_TRIANGLES = 10000#0  # Skip and resample meshes with more triangles

# Scene Settings
BOX_SIZE = 1.0
SPP = 2048 # High quality sample count
# LIGHT_INTENSITY = 10.0
# LIGHT_SIZE_RATIO = 0.5  # Fraction of ceiling covered by the emitter (larger = softer shadows, more light)


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
                
                # Check if Mitsuba can parse the valid vertex normals
                mi.load_dict({
                    'type': 'obj',
                    'filename': obj_path
                })
            except Exception as e:
                print(f"  Skipping {class_name}/{shape_id}: failed to load mesh or bad normals - {e}")
                continue

            selected.append((class_id, shape_id, obj_path))
            print(f"  Selected: {class_name}/{shape_id} ({num_triangles} triangles)")
            chosen_count += 1

    return selected



def get_transform_matrix(transform_dict, mesh):
    """
    Computes a 4x4 transformation matrix from the JSON config.
    Handles the 'normalize' flag by centering and scaling to a unit box first.
    """
    T = np.eye(4)
    
    # 1. Normalize (Center and scale max dimension to 1.0)
    if transform_dict.get("normalize", False):
        vertices = mesh.vertices
        bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        max_dim = np.max(bbox_max - bbox_min)
        
        if max_dim > 0:
            scale_norm = 1.0 / max_dim
            T_norm = trimesh.transformations.translation_matrix(-center)
            S_norm = trimesh.transformations.scale_matrix(scale_norm)
            # Apply translation, then scale
            T = trimesh.transformations.concatenate_matrices(S_norm, T_norm)
            
    # 2. Apply explicit scale
    scale_vec = transform_dict.get("scale", [1.0, 1.0, 1.0])
    S = np.diag([scale_vec[0], scale_vec[1], scale_vec[2], 1.0])
    T = trimesh.transformations.concatenate_matrices(S, T)
    
    # 3. Apply rotation (assuming XYZ Euler angles in degrees)
    rot = transform_dict.get("rotation", [0.0, 0.0, 0.0])
    rx = trimesh.transformations.rotation_matrix(np.radians(rot[0]), [1, 0, 0])
    ry = trimesh.transformations.rotation_matrix(np.radians(rot[1]), [0, 1, 0])
    rz = trimesh.transformations.rotation_matrix(np.radians(rot[2]), [0, 0, 1])
    R = trimesh.transformations.concatenate_matrices(rx, ry, rz)
    T = trimesh.transformations.concatenate_matrices(R, T)
    
    # 4. Apply explicit translation
    trans = transform_dict.get("translation", [0.0, 0.0, 0.0])
    T_trans = trimesh.transformations.translation_matrix(trans)
    T = trimesh.transformations.concatenate_matrices(T_trans, T)
    
    return T

def get_material_vector(material_dict):
    """Converts a material dict into a flat 10D feature array."""
    diffuse = material_dict.get("diffuse", [0.8, 0.8, 0.8])
    specular = material_dict.get("specular", [0.0, 0.0, 0.0])
    emission = material_dict.get("emissive", [0.0, 0.0, 0.0])
    roughness = material_dict.get("roughness", 0.5)
    # return np.concatenate([diffuse, emission, [roughness]]).astype(np.float32)

    return np.concatenate([diffuse, specular, emission, [roughness]]).astype(np.float32)


def construct_full_config(template_path, additional_objs_config=None, cam_config=None):
    """Loads the base JSON and merges in any additional objects or camera config."""
    with open(template_path, 'r') as f:
        scene_config = json.load(f)

    if additional_objs_config:
        scene_config['objects'].update(additional_objs_config)

    if cam_config:
        scene_config['cameras'][0].update(cam_config)
    
    return scene_config

def process_scene(
        scene_config, 
        output_dir, 
        spp=256, 
        points_per_object=2048, 
        resolution=(256, 256),
        scene_name="scene"
    ):
    """From scene config, samples geometry, renders, and saves the .npz."""
        
    base_dir = "json_templates" # Assumes obj files are relative to json
    
    # Initialize data structures
    entity_vertices = []
    entity_normals = []
    entity_materials = []
    
    # Initialize Mitsuba Dictionary  
    cam_config =   scene_config['cameras'][0]
    mi_scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 6},
        'sensor': {
            'type': 'perspective',
            'fov': cam_config['fov'],
            'to_world': mi.ScalarTransform4f.look_at(
                target=cam_config['look_at'],
                origin=cam_config['position'],
                up=cam_config['up']
            ),
            'sampler': {'type': 'independent', 'sample_count': spp},
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb'
            }
        }
    }

    # Process all objects
    for obj_name, obj_data in scene_config['objects'].items():
        # Load mesh with trimesh for geometric sampling
        mesh_path = os.path.join(base_dir, obj_data['mesh_path'])
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Calculate exactly one transform matrix to use for both Trimesh and Mitsuba
        T = get_transform_matrix(obj_data['transform'], mesh)
        
        # 1. Trimesh Processing (Apply transform and sample)
        mesh.apply_transform(T)
        points, face_indices = trimesh.sample.sample_surface(mesh, points_per_object)
        normals = mesh.face_normals[face_indices]
        
        mat_vec = get_material_vector(obj_data['material'])
        
        entity_vertices.append(points)
        entity_normals.append(normals)
        entity_materials.append(mat_vec)
        
        # 2. Mitsuba Processing (Build shape and attach transform)
        mat = obj_data['material']
        emissive = mat.get('emissive', [0.0, 0.0, 0.0])
        
        mi_shape = {
            'type': 'ply' if mesh_path.endswith('.ply') else 'obj',
            'filename': mesh_path,
            'to_world': mi.ScalarTransform4f(T.tolist()),
            # Face normals parameter handles the smooth_shading flag organically
            'face_normals': not mat.get('smooth_shading', True)
        }
        
        if sum(emissive) > 0:
            mi_shape['emitter'] = {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': emissive}
            }
            
        # Use a twosided principled BSDF to prevent backfaces from capturing light
        # and to properly calculate both diffuse and specular material properties
        specular_list = mat.get('specular', [0.5, 0.5, 0.5])
        spec_val = float(np.mean(specular_list))

        if "shapenet" in obj_name: # shapenet objects need twosided 
            mi_shape['bsdf'] = {
                'type': 'twosided', # fix backface culling/black faces issue
                'material': {
                    'type': 'principled',
                    'base_color': {'type': 'rgb', 'value': mat.get('diffuse', [1, 1, 1])},
                    'specular': spec_val,
                    'roughness': mat.get('roughness', 0.5)
                }
            }
        else:
            mi_shape['bsdf'] = {
                'type': 'principled',
                'base_color': {'type': 'rgb', 'value': mat.get('diffuse', [1, 1, 1])},
                'specular': spec_val,
                'roughness': mat.get('roughness', 0.5)
                # 'type': 'diffuse',
                # 'reflectance': {
                #     'type': 'rgb',
                #     'value': mat.get('diffuse', [1, 1, 1]),
                # }
            }
            
        mi_scene_dict[obj_name] = mi_shape

    # Stack geometry into tensors
    scene_data = {
        "vertices": np.stack(entity_vertices, axis=0).astype(np.float32),
        "normals": np.stack(entity_normals, axis=0).astype(np.float32),
        "materials": np.stack(entity_materials, axis=0).astype(np.float32),
        "num_objects": len(scene_config['objects'])
    }
    
    # Render with Mitsuba
    # print(f"Loading Mitsuba scene: {scene_config.get('scene_name', 'Unnamed')}")
    mi_scene = mi.load_dict(mi_scene_dict)
    
    # print("Rendering...")
    image_tensor = mi.render(mi_scene)
    hdr_image_np = np.array(image_tensor, dtype=np.float32)
    
    # Save Outputs
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, f"{scene_name}.npz")
    # exr_path = os.path.join(output_dir, f"{scene_name}_render.exr")
    png_path = os.path.join(output_dir, f"{scene_name}_render.png")

    # Save the visual EXR for debugging and tone-mapped PNG for easy viewing 
    # mi.Bitmap(image_tensor).write(exr_path) # don't need to save exr directly since it gets saved in npz already
    save_image(image_tensor, png_path, tone_mapping="reinhard", exposure=1)

    # Save the pipeline data
    np.savez(
        npz_path,
        hdr_target_image=hdr_image_np,
        camera_pos=cam_config['position'],          
        camera_lookat=cam_config['look_at'],
        entity_vertices=scene_data["vertices"],      # [N, 2048, 3]
        entity_normals=scene_data["normals"],        # [N, 2048, 3]
        entity_materials=scene_data["materials"],    # [N, 10]
        num_objects=scene_data["num_objects"]        # Scalar
    )
    
    # print(f"Successfully saved {scene_data['num_objects']} objects to {npz_path}")



def render_single_case(args):
    """Worker function for parallel rendering. Takes a tuple of case parameters."""
    case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, randomize_camera, random_seed, renders_dir = args

    try:
        # Create scene
        np.random.seed(random_seed + case_idx)
        
        # print(f"Generating Case {case_idx}: {class_id}, rot={rotation:.1f}...")

        target = np.array([0.0, 0.0, 0.0])
        r = BOX_SIZE * 2.0

        if randomize_camera:
            # # Randomize camera position with fixed distance and moderate angles around [0.0, -r, 0.0]
            # theta = np.random.uniform(-np.radians(20), np.radians(20)) # horizontal angle
            # phi = np.random.uniform(-np.radians(15), np.radians(15))   # vertical angle

            # Random viewpoint from 9 evenly spaced positions around [0.0, -r, 0.0]
            thetas_deg = [-20, -10, 0, 10, 20]
            phis_deg = [-15, 0, 15]
            angle_pairs = [(t, p) for t in thetas_deg for p in phis_deg]
            angle_pairs.remove((0, 0)) # remove head-on view from random selection since we add it in separately if include_head_on_view is True
            chosen_pair = angle_pairs[np.random.randint(len(angle_pairs))]
            
            theta = np.radians(chosen_pair[0])
            phi = np.radians(chosen_pair[1])

            cam_x = target[0] + r * np.sin(theta) * np.cos(phi)
            cam_y = target[1] - r * np.cos(theta) * np.cos(phi)
            cam_z = target[2] + r * np.sin(phi)

            cam_config = {
                "position": [cam_x, cam_y, cam_z],
                "look_at": target.tolist(),
                "up": [0.0, 0.0, 1.0],
                "fov": 37.5
            }

        else:
            cam_config = {
                "position": [0.0, -r, 0.0],
                "look_at": target.tolist(),
                "up": [0.0, 0.0, 1.0],
                "fov": 37.5
            }
        
        # Build object config
        if not os.path.exists(mesh_path):
            print(f"Skipping: {mesh_path} not found")
            return

        
        scale_factor = 0.75 # np.random.uniform(0.25, 0.75)
        #TODO: add variation
        dx = 0 #np.random.uniform(-BOX_SIZE * 0.25, BOX_SIZE * 0.25)
        dy = 0 #np.random.uniform(-BOX_SIZE * 0.25, BOX_SIZE * 0.25)
        dz = 0 #np.random.uniform(-BOX_SIZE * 0.25, BOX_SIZE * 0.25)

        rx = 0 #np.random.uniform(-90, 90) # degrees
        ry = rotation # 0 #np.random.uniform(-90, 90) # degrees
        rz = 0 #np.random.uniform(-90, 90) # degrees

        # Position: Center horizontally, Bottom on floor
        
        obj_config = {"shapenet_object": {
            "mesh_path": mesh_path,
            "transform": {
                "translation": [
                    0.0 + dx,
                    0.0 + dy,
                    -0.25 + dz #-0.125
                ],
                "rotation": [
                    90.0 + rx,
                    180.0 + ry,
                    0.0 + rz
                ],
                "scale": [scale_factor] * 3,
                "normalize": True
            },
            "material": {
                "diffuse": color.tolist(),
                "specular": [0.9, 0.9, 0.9],
                "random_diffuse_max": 0.5,
                "roughness": 1.0,
                "emissive": [0.0, 0.0, 0.0],
                "smooth_shading": True,
            }
        }}

        scene_config = construct_full_config(
            template_path="json_templates/cbox_template.json", # NOTE: this is hardcoded...
            # template_path="json_templates/light_only_template.json", # NOTE: this is hardcoded...
            additional_objs_config=obj_config,
            cam_config=cam_config
        )

        fn_prefix = f"case_{case_idx:03d}_{class_id}_{shape_id}_c{color_idx}_r{rot_idx}"
        process_scene(scene_config, renders_dir, spp=SPP, points_per_object=2048, resolution=(256, 256), scene_name=fn_prefix)

        return (case_idx, None)
    except Exception as e:
        import traceback
        return (case_idx, str(e) + "\n" + traceback.format_exc())




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Allow user to specify which shapes to generate
    parser.add_argument('--classes', nargs='+', default=list(CLASS_IDS.keys()), 
                      help=f"Specific shape classes to generate. Available: {list(CLASS_IDS.keys())}")
    
    # Control number of rotations
    parser.add_argument('--num_rotations', type=int, default=NUM_ROTATIONS, 
                      help="Number of rotation angles to generate (0 to 360 degrees)")
    
    # Control render quality
    parser.add_argument('--spp', type=int, default=SPP, 
                      help=f"Samples per pixel (default: {SPP})")
    # parser.add_argument('--light_intensity', type=float, default=LIGHT_INTENSITY, help='Ceiling light radiance')
    # parser.add_argument('--light_size_ratio', type=float, default=LIGHT_SIZE_RATIO, help='Ceiling light size as fraction of box size (0-1)')
    parser.add_argument('--exposure', type=float, default=1.2, help='Tone mapping exposure multiplier')
    # TODO: actually use this in rendering, and control object variation
    # Control object variation
    parser.add_argument('--scale_min', type=float, default=0.3, help='Minimum object scale as fraction of box size (default: 0.3)')
    parser.add_argument('--scale_max', type=float, default=0.5, help='Maximum object scale as fraction of box size (default: 0.5)')
    parser.add_argument('--pos_variation', type=float, default=0.15, help='Position variation of object as fraction of box size (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--shapes_per_class', type=int, default=SHAPES_PER_CLASS, help='Number of shapes to select per class from ShapeNet (default: 15)')
    parser.add_argument('--randomize_camera', type=bool, default=False, help='Whether to randomize camera position for each render (default: False)')
    parser.add_argument('--viewpoints_per_case', type=int, default=1, help='Number of different camera viewpoints to render per object configuration (default: 1)')
    parser.add_argument('--include_head_on_view', type=bool, default=True, help='Whether to include a head-on view (camera directly facing object) for each case (default: True)')
    
    args = parser.parse_args()

    # Update global settings based on args
    SPP = args.spp
    # LIGHT_INTENSITY = args.light_intensity
    # LIGHT_SIZE_RATIO = args.light_size_ratio

    exposure = args.exposure
    shapes_per_class = args.shapes_per_class

    num_rotations=args.num_rotations
    random_seed = args.random_seed
    randomize_camera = args.randomize_camera
    viewpoints_per_case = args.viewpoints_per_case
    include_head_on_view = args.include_head_on_view
    if not randomize_camera:
        viewpoints_per_case = 1 # Override to 1 if not randomizing camera

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
            
    processed_shapes = select_random_shapes(num_per_class=shapes_per_class)

    # Step 2: Summary
    print(f"\n[2/4] Loaded {len(processed_shapes)} meshes")

    # Step 3: Generate colors and rotations
    print("\n[3/4] Generating render configurations...")

    ## Generate random rotations
    # Generate uniform rotations
    random.seed(123)
    # rotations = [random.uniform(0, 360) for _ in range(num_rotations)]
    rotations = list(range(0, 360, 360 // num_rotations)) if num_rotations > 0 else [0]
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
    total_cases = len(processed_shapes) * NUM_COLORS * num_rotations
    print(f"Settings:")
    print(f"  Classes: {args.classes}")
    print(f"  Rotations: {len(rotations)} steps")
    print(f"  Samples per pixel: {SPP}")
    # print(f"  Light size ratio: {LIGHT_SIZE_RATIO}")
    # print(f"  Light intensity: {LIGHT_INTENSITY}")
    # print(f"  Exposure: {exposure}")
    # print(f"  Scale range: {args.scale_min} - {args.scale_max}")
    # print(f"  Position variation: {args.pos_variation}")
    print(f"  Random seed: {args.random_seed}")
    # print(f"  Total images to generate: {len(args.classes) * len(rotations) * len(PREDEFINED_COLORS)}") # TODO: needs to be updated with new settings

    print(f"  Total cases: {total_cases}")
    
    # Build list of all render cases with args for worker function
    render_cases = []
    case_idx = 0
    for class_id, shape_id, mesh_path in processed_shapes:
        key = f"{class_id}_{shape_id}"
        colors = shape_colors[key]
        for color_idx, color in enumerate(colors):
            for rot_idx, rotation in enumerate(rotations):
                for vp in range(viewpoints_per_case):
                    if vp == 0 and include_head_on_view:
                        # First viewpoint is head-on (no random camera)
                        randomize_camera = False
                    else:
                        randomize_camera = args.randomize_camera
                    case_idx += 1
                    # Pack all args for worker: (case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, randomize_camera, random_seed, RENDER_DIR)
                    render_cases.append((case_idx, class_id, shape_id, mesh_path, color, color_idx, rotation, rot_idx, randomize_camera, random_seed, RENDER_DIR))

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
                    idx, error = future.result()
                    if error:
                        tqdm.write(f"Error in case {idx}: {error}")
                    # elif params:
                    #     all_metadata.append(params)
                except Exception as e:
                    tqdm.write(f"Exception in case {case_idx}: {e}")
                pbar.update(1)

    # Save master metadata
    metadata_path = os.path.join(RENDER_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"  Total renders: {len(all_metadata)}")
    print(f"  Output directory: {RENDER_DIR}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("=" * 60)
