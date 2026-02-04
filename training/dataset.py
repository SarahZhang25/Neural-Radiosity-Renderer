import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob
from typing import Tuple, List, Optional
from utils.ray_generator import RayGenerator

class RadiosityDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        image_res: int = 128,
        num_points_per_object: int = 2048,
        split: str = 'train'
    ):
        self.data_dir = data_dir
        self.image_res = image_res
        self.num_points = num_points_per_object
        
        # Find all completed cases (must have .npz and .png)
        # Pattern: case_*.npz
        self.files = glob.glob(os.path.join(data_dir, "case_*_data.npz"))
        self.files.sort()
        
        # Simple split (first 90% train, last 10% val)
        split_idx = int(len(self.files) * 0.9)
        if split == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]
            
        print(f"[{split}] Found {len(self.files)} samples in {data_dir}")

        self.ray_generator = RayGenerator()
        
        # Fixed Camera Parameters (from generate_cube_only.py)
        box_size = 2.0
        self.cam_pos = np.array([0.0, box_size * 0.5, box_size * 1.5])
        self.cam_lookat = np.array([0.0, box_size * 0.5, 0.0])
        self.cam_up = np.array([0.0, 1.0, 0.0])
        self.fov_deg = 50.0
        
        # Precompute c2w for efficiency (it's constant for this dataset)
        self.c2w = self._compute_c2w(self.cam_pos, self.cam_lookat, self.cam_up)
        # FOV in radians
        self.fov_rad = torch.tensor(np.deg2rad(self.fov_deg)).float().view(1) 
        
        # Ray map is constant for same camera/resolution
        self.precomputed_ray_data = None


    def _compute_c2w(self, pos, target, up):
        # Camera Coordinate System:
        # Camera looks down -Z
        z_axis = pos - target
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 4x4 Matrix
        c2w = np.eye(4)
        c2w[:3, 0] = x_axis
        c2w[:3, 1] = y_axis
        c2w[:3, 2] = z_axis
        c2w[:3, 3] = pos
        return torch.from_numpy(c2w).float()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load NPZ
        data = np.load(file_path)
        
        # 1. Load Image
        img_path = file_path.replace("_data.npz", "_render.png")
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_res, self.image_res), Image.Resampling.BILINEAR)
        image_tensor = transforms.ToTensor()(image) # (3, H, W)
        
        # 2. Process Point Clouds
        # Object 0: The central object
        obj_verts = data['object_vertices'] # (N, 3)
        obj_normals = data['object_normals'] if 'object_normals' in data else np.zeros_like(obj_verts)
        obj_color = data['color'] # (3,)
        
        # Object 1: The box walls
        wall_verts = data['wall_vertices'] # (M, 3)
        wall_normals = data['wall_normals'] if 'wall_normals' in data else np.zeros_like(wall_verts)
        wall_color = np.array([0.9, 0.9, 0.9]) # Placeholder white walls
        
        # Resample/Pad Point Clouds
        obj_verts, obj_normals = self._sample_points(obj_verts, self.num_points, normals=obj_normals)
        wall_verts, wall_normals = self._sample_points(wall_verts, self.num_points, normals=wall_normals)
        
        # Prepare Tensors
        # Shape: (N_obj, N_v, 3) -> (2, N_v, 3)
        positions = torch.stack([
            torch.from_numpy(obj_verts).float(),
            torch.from_numpy(wall_verts).float()
        ])

        normals = torch.stack([
            torch.from_numpy(obj_normals).float(),
            torch.from_numpy(wall_normals).float()
        ])
        
        # Properties: (N_obj, 3) -> (2, 3)
        properties = torch.stack([
            torch.from_numpy(obj_color).float(),
            torch.from_numpy(wall_color).float()
        ])
        
        # Class IDs: (N_obj,) -> 0 for Object, 1 for Wall (arbitrary mapping)
        class_ids = torch.tensor([0, 1]).long() 
        
        # 3. Generate Rays
        # If camera was variable, we'd do this per item. Since fixed, we could cache.
        if self.precomputed_ray_data is None:
             rays_o, rays_d = self.ray_generator(self.c2w, self.fov_rad, self.image_res)
             self.precomputed_ray_data = (rays_o, rays_d)
        else:
             rays_o, rays_d = self.precomputed_ray_data
             
        # rays_d is (H, W, 3), we need it channel last?
        # Model expects rays_o (3,) and ray_map (H, W, 3)
        # RayGenerator returns rays_d as (H, W, 3)
        
        return {
            'rays_o': rays_o,                  # (3,)
            'rays_d': rays_d,                  # (H, W, 3) - used as ray_map
            'obj_positions': positions,        # (2, N_p, 3)
            'obj_normals': normals,            # (2, N_p, 3)
            'obj_properties': properties,      # (2, 3)
            'obj_class_ids': class_ids,        # (2,)
            'target_image': image_tensor       # (3, H, W)
        }

    def _sample_points(self, points, num_points, normals=None):
        """Randomly sample points to fix size."""
        N = points.shape[0]
        if N >= num_points:
            indices = np.random.choice(N, num_points, replace=False)
        else:
            indices = np.random.choice(N, num_points, replace=True)
        
        if normals is not None:
            return points[indices], normals[indices]
        return points[indices]
