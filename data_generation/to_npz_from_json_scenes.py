"""
Reads in a directory of JSON configurations, parsing all mesh geometry (using the transforms) 
along with materials, cameras, and rendering the target HDR and PNGs, exporting the output .h5 
chunk files natively supporting packing of hundreds of scenes into single scalable files.

Ex usage:
cd /home/sazhang/Neural-Radiosity-Renderer/data_generation
python to_npz_from_json_scenes.py \
    --input_dir /path/to/renderformer/json/scenes \
    --exr_dir /path/to/exrs_dir \
    --output_dir /path/to/output/h5 \
    --points 2048 \
    --chunk_size 5000
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import trimesh
import h5py
import concurrent.futures
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from generate_auto_mitsuba import get_transform_matrix, get_material_vector

def load_exr(path):
    try:
        import cv2
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] >= 3:
                # OpenCV loads in BGR, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.float32)
    except Exception:
        pass

    try:
        import imageio
        if hasattr(imageio, 'v3'):
            return imageio.v3.imread(path).astype(np.float32)
        else:
            return imageio.imread(path).astype(np.float32)
    except Exception:
        pass

    try:
        import OpenEXR
        import Imath
        file = OpenEXR.InputFile(path)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r = np.frombuffer(file.channel('R', pt), dtype=np.float32).reshape(size[1], size[0])
        g = np.frombuffer(file.channel('G', pt), dtype=np.float32).reshape(size[1], size[0])
        b = np.frombuffer(file.channel('B', pt), dtype=np.float32).reshape(size[1], size[0])
        return np.stack([r, g, b], axis=-1)
    except Exception:
        pass

    raise RuntimeError(f"Failed to load EXR file: {path}. Ensure cv2, imageio, or OpenEXR is installed.")

def compute_c2w(pos, target, up=[0.0, 0.0, 1.0]) -> np.ndarray:
    """Build a camera-to-world 4x4 matrix. Camera looks down -Z."""
    pos = np.array(pos, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = pos - target
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = pos
    return c2w

def process_renderformer_scene(json_path, exr_dir, points_per_object=2048):
    """
    Convert one JSON scene file to dataset inputs. H5 formation deferred to 
    chunking script.
    
    Args:
        json_path: Path to the JSON scene configuration file.
        exr_dir: Directory containing EXR files.
        points_per_object: Number of points to sample per object.
    
    Returns:
        scene_name: Name of the scene.
        img_hdr: Ground truth HDR image. [H, W, 3]
        all_points: Array of surface positions. [N_obj, N_pts, 3]
        all_normals: Array of surface normals. [N_obj, N_pts, 3]
        all_materials: Array of material vectors. [N_obj, N_pts, 11]
        c2w: Camera-to-world matrix. [4, 4]
        cam_fov: Camera field of view. [2,]
    """
    with open(json_path, 'r') as f:
        scene_config = json.load(f)
    
    scene_name = os.path.splitext(os.path.basename(json_path))[0]
    base_dir = os.path.dirname(json_path)

    # 1. Load EXR target image from the EXR directory
    exr_path = os.path.join(exr_dir, f"{scene_name}_0.exr") # TODO: eliminate _0 part
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"Missing ground truth EXR file: {exr_path}")
        
    img_hdr = load_exr(exr_path)

    # 2. Extract Geometry
    entity_points = []
    entity_normals = []
    entity_materials = []

    cam_config = scene_config['cameras'][0]

    for obj_name, obj_data in scene_config['objects'].items():
        raw_mesh_path = obj_data['mesh_path']
        if raw_mesh_path.startswith('~'):
            mesh_path = os.path.expanduser(raw_mesh_path)
        else:
            mesh_path = os.path.normpath(os.path.join(base_dir, raw_mesh_path))
            
        if not os.path.exists(mesh_path):
            # Fallback if the JSON path incorrectly assumes .objaverse is in the repo dir
            if ".objaverse" in raw_mesh_path:
                fallback_path = os.path.expanduser(f"~/.objaverse/{raw_mesh_path.split('.objaverse/')[-1]}")
                if os.path.exists(fallback_path):
                    mesh_path = fallback_path
            
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            
        mesh = trimesh.load(mesh_path, force='mesh')
        
        T = get_transform_matrix(obj_data['transform'], mesh)
        
        mesh.apply_transform(T)
        points, face_indices = trimesh.sample.sample_surface(mesh, points_per_object)
        normals = mesh.face_normals[face_indices]
        
        # Transform points back to normalized local space for texturing
        inv_T = np.linalg.inv(T)
        points_homogeneous = np.hstack((points, np.ones((points_per_object, 1))))
        local_points = (inv_T @ points_homogeneous.T).T[:, :3]
        # Pass local_points to evaluate per-point textures
        mat_vec = get_material_vector(obj_data['material'], local_points=local_points)
        
        entity_points.append(points)
        entity_normals.append(normals)
        entity_materials.append(mat_vec)

    all_points = np.stack(entity_points, axis=0).astype(np.float32)
    all_normals = np.stack(entity_normals, axis=0).astype(np.float32)
    all_materials = np.stack(entity_materials, axis=0).astype(np.float32)
    
    c2w = compute_c2w(cam_config['position'], cam_config['look_at'], cam_config.get('up', [0.0, 0.0, 1.0]))
    cam_fov = np.array(cam_config['fov'], dtype=np.float32)
    
    return scene_name, img_hdr, all_points, all_normals, all_materials, c2w, cam_fov

def main():
    parser = argparse.ArgumentParser(description="Extract GlobalIlluminationModel training data from Renderformer scenes (geometry + embedded EXRs to chunked H5)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSON scene configurations")
    parser.add_argument("--exr_dir", type=str, required=True, help="Directory containing EXR files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated .h5 chunk files")
    parser.add_argument("--points", type=int, default=2048, help="Points to sample per object")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of scenes to pack into a single .h5 file")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} scene files. Starting H5 packing...")
    
    existing_chunks = glob.glob(os.path.join(args.output_dir, "dataset_chunk_*.h5"))
    next_chunk_id = 0
    if existing_chunks:
        chunk_ids = [int(os.path.basename(c).split('_')[-1].split('.')[0]) for c in existing_chunks]
        next_chunk_id = max(chunk_ids) + 1

    chunk_size = args.chunk_size
    for i in range(0, len(json_files), chunk_size):
        chunk_files = json_files[i:i+chunk_size]
        chunk_filename = f"dataset_chunk_{next_chunk_id:04d}.h5"
        chunk_path = os.path.join(args.output_dir, chunk_filename)
        
        print(f"Processing {len(chunk_files)} scenes for {chunk_filename}...")
        
        # Use ProcessPoolExecutor to parallelize CPU-heavy parsing and matrix math
        # We write to the H5 file sequentially in the main thread because H5py writing is not thread-safe
        with h5py.File(chunk_path, 'w') as h5f:
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Submit all parsing jobs
                futures = {
                    executor.submit(process_renderformer_scene, json_file, args.exr_dir, args.points): json_file 
                    for json_file in chunk_files
                }
                
                # Write to HDF5 as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Writing {chunk_filename}"):
                    json_file = futures[future]
                    try:
                        res = future.result()
                        scene_name, img_hdr, vertices, normals, materials, c2w, cam_fov = res
                        
                        grp = h5f.create_group(scene_name)
                        grp.create_dataset("hdr_target_image", data=img_hdr, compression="lzf")
                        grp.create_dataset("entity_vertices", data=vertices, compression="lzf")
                        grp.create_dataset("entity_normals", data=normals, compression="lzf")
                        grp.create_dataset("entity_materials", data=materials, compression="lzf")
                        grp.create_dataset("c2w", data=c2w, compression="lzf")
                        grp.create_dataset("camera_fov", data=cam_fov)
                    except Exception as e:
                        import traceback
                        print(f"\nError processing {json_file}: {e}")
                        traceback.print_exc()
        
        next_chunk_id += 1
        print(f"Finished writing {chunk_filename}")

if __name__ == "__main__":
    main()
