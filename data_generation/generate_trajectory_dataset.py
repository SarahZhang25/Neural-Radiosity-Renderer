import numpy as np
from generate_auto_mitsuba import construct_full_config, process_scene
from utils import save_image


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
# import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import mitsuba as mi


multiprocessing.set_start_method('spawn', force=True)

# Add parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Set to 'cuda_ad_rgb' for GPU acceleration or 'llvm_ad_rgb' for CPU
mi.set_variant('cuda_ad_rgb')

# Constants
SHAPENET_ROOT = "/home/sazhang/ShapeNetCorev2"
# OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output_auto")
RENDER_DIR = os.path.join(SCRIPT_DIR, "trajectories/attempt6_table_chair_540/v1_chair_20f/")

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


def slerp_vectors(v0, v1, t):
    """
    Spherical linear interpolation between two 3D vectors.
    t is the interpolation parameter [0, 1].
    """
    v0 = np.array(v0)
    v1 = np.array(v1)
    r = np.linalg.norm(v0) # Radius
    
    # Normalize
    v0_norm = v0 / r
    v1_norm = v1 / r
    
    # Dot product gives the cosine of the angle between vectors
    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta_0 = np.arccos(dot)
    
    # If vectors are exactly the same, or angle is 0, fallback to linear
    if np.sin(theta_0) < 1e-6:
        return (1 - t) * v0 + t * v1
        
    # Calculate interpolated angles
    theta_t = theta_0 * t
    s0 = np.sin(theta_0 - theta_t) / np.sin(theta_0)
    s1 = np.sin(theta_t) / np.sin(theta_0)
    
    # Return scaled interpolated vector
    return (s0 * v0_norm + s1 * v1_norm) * r

def render_trajectory(mesh_path, output_dir, color, rotation=0.0, num_frames=20):
    """
    Renders a 20-frame camera trajectory for a single object.
    """
    BOX_SIZE = 1.0 # Assuming this is your global BOX_SIZE
    target = np.array([0.0, 0.0, 0.0])
    r = BOX_SIZE * 2.0
    
    # Define Start View (e.g., extreme bottom-left from your training set)
    theta_start = np.radians(-20)
    phi_start = np.radians(-15)
    pos_start = [
        target[0] + r * np.sin(theta_start) * np.cos(phi_start),
        target[1] - r * np.cos(theta_start) * np.cos(phi_start),
        target[2] + r * np.sin(phi_start)
    ]
    
    # Define End View (e.g., extreme top-right from your training set)
    theta_end = np.radians(20)
    phi_end = np.radians(15)
    pos_end = [
        target[0] + r * np.sin(theta_end) * np.cos(phi_end),
        target[1] - r * np.cos(theta_end) * np.cos(phi_end),
        target[2] + r * np.sin(phi_end)
    ]

    print(f"Generating {num_frames}-frame trajectory...")
    
    for frame_idx, t in enumerate(np.linspace(0, 1.0, num_frames)):
        
        # Slerp the camera position
        current_pos = slerp_vectors(pos_start, pos_end, t)
        
        cam_config = {
            "position": current_pos.tolist(),
            "look_at": target.tolist(),
            "up": [0.0, 0.0, 1.0],
            "fov": 37.5
        }
        
        # Use exact same object setup as your standard rendering
        obj_config = {"shapenet_object": {
            "mesh_path": mesh_path,
            "transform": {
                "translation": [0.0, 0.0, -0.25], 
                "rotation": [90.0, 180.0 + rotation, 0.0],
                "scale": [0.75] * 3,
                "normalize": True
            },
            "material": {
                "diffuse": color,
                "specular": [0.9, 0.9, 0.9],
                "roughness": 1.0,
                "emissive": [0.0, 0.0, 0.0],
                "smooth_shading": True,
            }
        }}

        scene_config = construct_full_config(
            template_path="json_templates/cbox_template.json",
            additional_objs_config=obj_config,
            cam_config=cam_config
        )

        fn_prefix = f"trajectory_frame_{frame_idx:02d}"
        
        # Call your existing process_scene function
        process_scene(
            scene_config=scene_config, 
            output_dir=output_dir, 
            spp=SPP, 
            points_per_object=2048, 
            resolution=(256, 256), # Bumped resolution for the video
            scene_name=fn_prefix
        )
        print(f"Rendered frame {frame_idx + 1}/{num_frames}")

if __name__ == "__main__":
    # Example usage: render a trajectory for a single mesh
    # a chair from the ShapeNet dataset used in 
    example_mesh = f"{SHAPENET_ROOT}/03001627/248e014f31771b31d3ddfaaa242f81a1/models/model_normalized.obj"
    example_color = PREDEFINED_COLORS[0].tolist() # Blue
    render_trajectory(
        mesh_path=example_mesh, 
        output_dir=RENDER_DIR, 
        color=example_color, 
        rotation=0.0, 
        num_frames=20
    )

