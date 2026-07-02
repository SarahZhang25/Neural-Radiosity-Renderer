"""
Reads in a directory of JSON configurations, parsing all mesh geometry (using the transforms) 
along with materials, cameras, and rendering the target HDR and PNGs, exporting the output .h5 
chunk files natively supporting packing of hundreds of scenes into single scalable files.

Ex usage:
cd /home/sazhang/Neural-Radiosity-Renderer/data_generation
python to_npz_from_json_scenes.py \
    --input_dir /path/to/renderformer/json/scenes \
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
from tqdm import tqdm

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

def process_renderformer_scene(json_path, h5f, output_dir, points_per_object=2048):
    with open(json_path, 'r') as f:
        scene_config = json.load(f)
    
    scene_name = os.path.splitext(os.path.basename(json_path))[0]
    base_dir = os.path.dirname(json_path)

    # 1. Load EXR target image from the output directory
    exr_path = os.path.join(output_dir, f"{scene_name}_0.exr")
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"Missing ground truth EXR file: {exr_path}")
        
    img_hdr = load_exr(exr_path)

    # 2. Extract Geometry
    entity_vertices = []
    entity_normals = []
    entity_materials = []

    cam_config = scene_config['cameras'][0]

    for obj_name, obj_data in scene_config['objects'].items():
        mesh_path = os.path.normpath(os.path.join(base_dir, obj_data['mesh_path']))
        mesh = trimesh.load(mesh_path, force='mesh')
        
        T = get_transform_matrix(obj_data['transform'], mesh)
        
        mesh.apply_transform(T)
        points, face_indices = trimesh.sample.sample_surface(mesh, points_per_object)
        normals = mesh.face_normals[face_indices]
        
        inv_T = np.linalg.inv(T)
        points_homogeneous = np.hstack((points, np.ones((points_per_object, 1))))
        local_points = (inv_T @ points_homogeneous.T).T[:, :3]
        
        mat_vec = get_material_vector(obj_data['material'], local_points=local_points)
        
        entity_vertices.append(points)
        entity_normals.append(normals)
        entity_materials.append(mat_vec)

    scene_data = {
        "vertices": np.stack(entity_vertices, axis=0).astype(np.float32),
        "normals": np.stack(entity_normals, axis=0).astype(np.float32),
        "materials": np.stack(entity_materials, axis=0).astype(np.float32),
    }
    
    # 3. Write to H5 Group (using LZF compression for blazing fast reads)
    grp = h5f.create_group(scene_name)
    grp.create_dataset("hdr_target_image", data=img_hdr, compression="lzf")
    grp.create_dataset("entity_vertices", data=scene_data["vertices"], compression="lzf")
    grp.create_dataset("entity_normals", data=scene_data["normals"], compression="lzf")
    grp.create_dataset("entity_materials", data=scene_data["materials"], compression="lzf")
    
    grp.create_dataset("camera_pos", data=np.array(cam_config['position'], dtype=np.float32))
    grp.create_dataset("camera_lookat", data=np.array(cam_config['look_at'], dtype=np.float32))
    grp.create_dataset("camera_fov", data=np.array(cam_config['fov'], dtype=np.float32))

def main():
    parser = argparse.ArgumentParser(description="Extract GlobalIlluminationModel training data from Renderformer scenes (geometry + embedded EXRs to chunked H5)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSON scene configurations")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated .h5 chunk files (and where EXRs are located)")
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
        
        print(f"Writing {len(chunk_files)} scenes to {chunk_filename}...")
        
        with h5py.File(chunk_path, 'w') as h5f:
            for json_file in tqdm(chunk_files):
                try:
                    process_renderformer_scene(
                        json_path=json_file,
                        h5f=h5f,
                        output_dir=args.output_dir,
                        points_per_object=args.points
                    )
                except Exception as e:
                    import traceback
                    print(f"Error processing {json_file}: {e}")
                    traceback.print_exc()
        
        next_chunk_id += 1
        print(f"Finished writing {chunk_filename}")

if __name__ == "__main__":
    main()
