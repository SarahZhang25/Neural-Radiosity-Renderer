"""
Reads in a directory of JSON configurations, parsing all mesh geometry (using the transforms) 
along with materials, cameras, and rendering the target HDR and PNGs, exporting the output .npz 
files identically to generate_auto_mitsuba.py.

TODO: add support for multiple workers using concurrent.futures to speed up processing.
though tbh it's pretty fast -- 500 scenes in < 1 min?

Ex usage:
cd /home/sazhang/Neural-Radiosity-Renderer/data_generation
python to_npz_from_json_scenes.py \
    --input_dir /path/to/renderformer/json/scenes \
    --output_dir /path/to/output/npz \
    --points 2048

python to_npz_from_json_scenes.py \
    --input_dir /home/sazhang/Neural-Radiosity-Renderer/renderformer/data_generation/dataset_single_obj \
    --output_dir /home/sazhang/Neural-Radiosity-Renderer/renderformer/tmp/dataset_single_obj \
    --points 2048
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import trimesh
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from generate_auto_mitsuba import get_transform_matrix, get_material_vector

def process_renderformer_scene(json_path, output_dir, points_per_object=2048):
    with open(json_path, 'r') as f:
        scene_config = json.load(f)
    
    scene_name = os.path.splitext(os.path.basename(json_path))[0]
    base_dir = os.path.dirname(json_path)

    entity_vertices = []
    entity_normals = []
    entity_materials = []

    cam_config = scene_config['cameras'][0]

    # Process all objects
    for obj_name, obj_data in scene_config['objects'].items():
        # Load mesh with trimesh for geometric sampling
        mesh_path = os.path.normpath(os.path.join(base_dir, obj_data['mesh_path']))
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Calculate exactly one transform matrix
        T = get_transform_matrix(obj_data['transform'], mesh)
        
        # Trimesh Processing (Apply transform and sample)
        mesh.apply_transform(T)
        points, face_indices = trimesh.sample.sample_surface(mesh, points_per_object)
        normals = mesh.face_normals[face_indices]
        
        mat_vec = get_material_vector(obj_data['material'])
        
        entity_vertices.append(points)
        entity_normals.append(normals)
        entity_materials.append(mat_vec)

    # Stack geometry into tensors
    scene_data = {
        "vertices": np.stack(entity_vertices, axis=0).astype(np.float32),
        "normals": np.stack(entity_normals, axis=0).astype(np.float32),
        "materials": np.stack(entity_materials, axis=0).astype(np.float32),
        "num_objects": len(scene_config['objects'])
    }
    
    # Save Outputs
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, f"{scene_name}.npz")

    np.savez(
        npz_path,
        camera_pos=cam_config['position'],          
        camera_lookat=cam_config['look_at'],
        entity_vertices=scene_data["vertices"],      # [N, 2048, 3]
        entity_normals=scene_data["normals"],        # [N, 2048, 3]
        entity_materials=scene_data["materials"],    # [N, 10]
        num_objects=scene_data["num_objects"]        # Scalar
    )

def main():
    parser = argparse.ArgumentParser(description="Extract GlobalIlluminationModel training data from Renderformer scenes (geometry only)")
    parser.add_argument("--input_dir", type=str, help="Directory containing JSON scene configurations")
    parser.add_argument("--output_dir", type=str, help="Directory to save generated .npz files")
    parser.add_argument("--points", type=int, default=2048, help="Points to sample per object")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} scene files. Starting processing...")
    
    for json_file in tqdm(json_files):
        try:
            process_renderformer_scene(
                json_path=json_file,
                output_dir=args.output_dir,
                points_per_object=args.points
            )
        except Exception as e:
            import traceback
            print(f"Error processing {json_file}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
