"""
Unified dataset generation script for Neural-Radiosity-Renderer (nmr) and Renderformer (rf).
Reads in a directory of JSON configurations, parsing all mesh geometry (using the transforms) 
along with materials, cameras, and rendering the target HDR and PNGs, exporting the output .h5 
chunk files natively supporting packing of hundreds of scenes into single scalable files.

It will extract the data required for the specified model formats and save them 
in separate chunked .h5 files, e.g., dataset_chunk_0000_nmr.h5 and dataset_chunk_0000_rf.h5.

Example usage:
cd /home/sazhang/Neural-Radiosity-Renderer
python data_generation/write_training_datasets_from_jsons.py \
    --input_dir renderformer/datasets/json_scenes/dataset_single_obj \
    --exr_dir renderformer/datasets/processed_datasets/dataset_single_obj \
    --output_dir renderformer/tmp/dataset_single_obj \
    --points 2048 \
    --chunk_size 5000 \
    --formats nmr rf
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
        print("failed cv2 import")
        pass

    try:
        import imageio
        if hasattr(imageio, 'v3'):
            return imageio.v3.imread(path, plugin="EXR-FI").astype(np.float32)
        else:
            return imageio.imread(path, plugin="EXR-FI").astype(np.float32)
    except Exception:
        print("failed imageio import")
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
        print("failed OpenEXR import")
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

def process_json_scene(json_path, exr_dir, points_per_object=2048, formats=['nmr', 'rf']):
    """
    Convert one JSON scene file to dataset inputs for the requested formats.
    H5 formation deferred to chunking script.
    
    Args:
        json_path: Path to the JSON scene configuration file.
        exr_dir: Directory containing EXR files.
        points_per_object: Number of points to sample per object (nmr only).
        formats: List of formats to generate ('nmr', 'rf').
    
    Returns:
        results: Dictionary containing 'scene_name' and sub-dictionaries for 'nmr' and 'rf'.
        For nmr:
        hdr_target_image: Ground truth HDR image. [H, W, 3]
        all_points: Array of surface positions. [N_obj, N_pts, 3]
        all_normals: Array of surface normals. [N_obj, N_pts, 3]
        all_materials: Array of material vectors. [N_obj, N_pts, 11]
        c2w: Camera-to-world matrix. [4, 4]
        camera_fov: Camera field of view. [2,]
        For rf:
        hdr_target_image: Ground truth HDR image. [H, W, 3] # TODO make names consistent
        triangles: Mesh triangles. [N_tri, 3, 3] in world coordinates (if pre-transformed).
        vn: Vertex normals (matched to triangles). [N_tri, 3, 3].
        texture: Unique material textures for each triangle. [N_tri, 32, 32, 3]
        c2w: Camera-to-world matrix. [4, 4]
        fov: Camera field of view. [2,]
    """
    with open(json_path, 'r') as f:
        scene_config = json.load(f)
    
    scene_name = os.path.splitext(os.path.basename(json_path))[0]
    base_dir = os.path.dirname(json_path)

    # 1. Load EXR target images for ALL views (cameras) in the scene
    img_hdrs = []
    cam_configs = scene_config.get('cameras', [])
    for i in range(len(cam_configs)):
        exr_path = os.path.join(exr_dir, f"{scene_name}_{i}.exr")
        if not os.path.exists(exr_path):
            raise FileNotFoundError(f"Missing ground truth EXR file: {exr_path}")
        img_hdrs.append(load_exr(exr_path))
        
    # Stack all views into a single tensor [num_views, H, W, 3]
    img_hdrs_stacked = np.stack(img_hdrs, axis=0) if len(img_hdrs) > 0 else np.array([])

    # Extract camera parameters for all views
    all_c2w = []
    all_fov = []
    for cam_config in cam_configs:
        c2w = compute_c2w(cam_config['position'], cam_config['look_at'], cam_config.get('up', [0.0, 0.0, 1.0]))
        all_c2w.append(c2w)
        all_fov.append(np.array(cam_config['fov'], dtype=np.float32))
        
    all_c2w = np.stack(all_c2w, axis=0) if len(all_c2w) > 0 else np.array([])
    all_fov = np.array(all_fov) if len(all_fov) > 0 else np.array([])

    results = {'scene_name': scene_name}

    # 2. Extract Geometry based on requested formats
    if 'nmr' in formats:
        # NMR Format arrays
        entity_points = []
        entity_normals = []
        entity_materials = []
        
    if 'rf' in formats:
        # Renderformer Format arrays
        all_triangles = []
        all_vn = []
        all_texture = []
        
        # Precompute the mask for 32x32 texture mapping
        size = 32
        mask = np.zeros((size, size), dtype=bool)
        x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        mask[x + y <= size] = 1

    for obj_name, obj_data in scene_config['objects'].items():
        raw_mesh_path = obj_data['mesh_path']
        if os.path.isabs(raw_mesh_path):
            mesh_path = raw_mesh_path
        elif raw_mesh_path.startswith('~'):
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
            
        # process=False ensures we keep exact vertex indices matching faces
        kwargs = {'process': False}
        if raw_mesh_path.endswith('.glb') or ('shapenet' in raw_mesh_path.lower()):
            kwargs['force'] = 'mesh'
            
        mesh = trimesh.load(mesh_path, **kwargs)
        
        # Apply transformation matrix to put object into world space
        T = get_transform_matrix(obj_data['transform'], mesh)
        mesh.apply_transform(T)
        
        # -----------------------------------------------------
        # NMR Geometry Extraction (Point-based)
        # -----------------------------------------------------
        if 'nmr' in formats:
            # Sample points uniformly from the surface
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

        # -----------------------------------------------------
        # Renderformer Geometry Extraction (Triangle-based)
        # -----------------------------------------------------
        if 'rf' in formats:
            # Extract data
            triangles = mesh.triangles
            vn = mesh.vertex_normals[mesh.faces]
            material_config = obj_data.get('material', {})
            
            if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) > 0:
                diffuse = mesh.visual.face_colors[..., :3] / 255.
            else:
                diffuse = np.ones((triangles.shape[0], 3))
                
            spec = material_config.get('specular', [0.0, 0.0, 0.0])
            specular = np.array(spec)[None].repeat(triangles.shape[0], axis=0)
            
            rough = material_config.get('roughness', 0.5)
            roughness = np.array([rough])[None].repeat(triangles.shape[0], axis=0)
            
            normal_map = np.zeros_like(diffuse)
            normal_map[..., 0] = 0.5
            normal_map[..., 1] = 0.5
            normal_map[..., 2] = 1.
            
            emis = material_config.get('emissive', [0.0, 0.0, 0.0])
            irradiance = np.array(emis)[None, :].repeat(triangles.shape[0], axis=0)
            
            # Concatenate textures and pad to 32x32 per triangle face
            texture = np.concatenate([diffuse, specular, roughness, normal_map, irradiance], axis=1)
            texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
            texture[:, :, ~mask] = 0.0

            all_triangles.append(triangles)
            all_vn.append(vn)
            all_texture.append(texture)

    # 3. Store Results back in dictionary
    if 'nmr' in formats:
        results['nmr'] = {
            'entity_vertices': np.stack(entity_points, axis=0).astype(np.float32) if entity_points else np.array([]),
            'entity_normals': np.stack(entity_normals, axis=0).astype(np.float32) if entity_normals else np.array([]),
            'entity_materials': np.stack(entity_materials, axis=0).astype(np.float32) if entity_materials else np.array([]),
            'hdr_target_image': img_hdrs_stacked,
            'c2w': all_c2w,
            'camera_fov': all_fov
        }
        
    if 'rf' in formats:
        results['rf'] = {
            'triangles': np.concatenate(all_triangles, axis=0).astype(np.float32) if all_triangles else np.array([]),
            'vn': np.concatenate(all_vn, axis=0).astype(np.float32) if all_vn else np.array([]),
            'texture': np.concatenate(all_texture, axis=0).astype(np.float16) if all_texture else np.array([]),
            'c2w': all_c2w,
            'fov': all_fov,
            'hdr_target_image': img_hdrs_stacked
        }
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract training data from Renderformer scenes to chunked H5 (NMR and RF formats supported)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSON scene configurations")
    parser.add_argument("--exr_dir", type=str, required=True, help="Directory containing EXR files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated .h5 chunk files")
    parser.add_argument("--points", type=int, default=2048, help="Points to sample per object (nmr format)")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of scenes to pack into a single .h5 file")
    parser.add_argument("--formats", nargs='+', choices=['nmr', 'rf'], default=['nmr', 'rf'], help="Formats to generate")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} scene files. Generating formats: {args.formats}")
    
    existing_chunks = glob.glob(os.path.join(args.output_dir, "*_dataset_chunk_*.h5"))
    next_chunk_id = 0
    if existing_chunks:
        # Find the highest existing chunk ID to avoid overwriting
        chunk_ids = [int(os.path.basename(c).split('_')[3].split('.')[0]) for c in existing_chunks if len(os.path.basename(c).split('_')) >= 4]
        if chunk_ids:
            next_chunk_id = max(chunk_ids) + 1

    chunk_size = args.chunk_size
    for i in range(0, len(json_files), chunk_size):
        chunk_files = json_files[i:i+chunk_size]
        base_chunk_name = f"dataset_chunk_{next_chunk_id:04d}"
        
        print(f"Processing {len(chunk_files)} scenes for {base_chunk_name}...")
        
        # Prepare file paths based on requested formats
        h5_paths = {}
        for fmt in args.formats:
            h5_paths[fmt] = os.path.join(args.output_dir, f"{fmt}_{base_chunk_name}.h5")
        
        # We use a ProcessPoolExecutor to heavily parallelize CPU tasks (mesh reading, array math)
        # We write to the H5 files sequentially in the main thread to avoid HDF5 concurrency corruption
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(process_json_scene, json_file, args.exr_dir, args.points, args.formats): json_file 
                for json_file in chunk_files
            }
            
            # Open the h5 files for writing
            h5_files = {fmt: h5py.File(h5_paths[fmt], 'w') for fmt in args.formats}
            
            try:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Writing {base_chunk_name}"):
                    json_file = futures[future]
                    try:
                        res = future.result()
                        scene_name = res['scene_name']
                        
                        # Save NMR data
                        if 'nmr' in args.formats and 'nmr' in res:
                            nmr_grp = h5_files['nmr'].create_group(scene_name)
                            # Compression LZF is fast and prevents massive file bloat
                            nmr_grp.create_dataset("hdr_target_image", data=res['nmr']['hdr_target_image'], compression="lzf")
                            nmr_grp.create_dataset("entity_vertices", data=res['nmr']['entity_vertices'], compression="lzf")
                            nmr_grp.create_dataset("entity_normals", data=res['nmr']['entity_normals'], compression="lzf")
                            nmr_grp.create_dataset("entity_materials", data=res['nmr']['entity_materials'], compression="lzf")
                            nmr_grp.create_dataset("c2w", data=res['nmr']['c2w'], compression="lzf")
                            # Camera FOV is a scalar/array so we don't compress it
                            nmr_grp.create_dataset("camera_fov", data=res['nmr']['camera_fov'])
                            
                        # Save RF data
                        if 'rf' in args.formats and 'rf' in res:
                            rf_grp = h5_files['rf'].create_group(scene_name)
                            rf_grp.create_dataset("hdr_target_image", data=res['rf']['hdr_target_image'], compression="lzf")
                            rf_grp.create_dataset("triangles", data=res['rf']['triangles'], compression="lzf")
                            rf_grp.create_dataset("vn", data=res['rf']['vn'], compression="lzf")
                            rf_grp.create_dataset("texture", data=res['rf']['texture'], compression="lzf")
                            rf_grp.create_dataset("c2w", data=res['rf']['c2w'], compression="lzf")
                            rf_grp.create_dataset("fov", data=res['rf']['fov'])

                    except Exception as e:
                        import traceback
                        print(f"\nError processing {json_file}: {e}")
                        traceback.print_exc()
            finally:
                # Ensure all files are properly closed
                for f in h5_files.values():
                    f.close()
        
        next_chunk_id += 1
        print(f"Finished {base_chunk_name} (formats: {args.formats})")

if __name__ == "__main__":
    main()
