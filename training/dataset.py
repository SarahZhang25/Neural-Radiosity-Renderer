import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
from training.ray_generator import RayGenerator
import mitsuba as mi
mi.set_variant('scalar_rgb')

class SceneDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        image_res: int = 128,
        num_points_per_object: int = 2048,
        split: str = 'train',
    ):
        self.data_dir = data_dir
        self.image_res = image_res
        self.num_points = num_points_per_object
        
        # Find all completed cases (must have .npz)
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.files.sort()

        # Filter out any corrupted images (NaN or Inf values in target HDR image)
        corrupted_paths = []
        for file_path in self.files:
            npz_data = np.load(file_path)
            img_hdr = npz_data["hdr_target_image"]
            if np.isnan(img_hdr).any() or np.isinf(img_hdr).any():
                corrupted_paths.append(file_path)

        self.files = [item for item in self.files if item not in corrupted_paths]
        print(f"Removed {len(corrupted_paths)} corrupted images")

        # Shuffle with a fixed seed
        if split == "all":
            print(f"[{split}] Using all {len(self.files)} samples in {data_dir}")
        else:
            assert split in ['train', 'val'], "split must be 'train', 'val', or 'all'"
            rng = np.random.RandomState(42)
            rng.shuffle(self.files)
            
            split_idx = int(len(self.files) * 0.9)
            if split == 'train':
                self.files = self.files[:split_idx]
            else:
                self.files = self.files[split_idx:]
                
        print(f"[{split}] Found {len(self.files)} samples in {data_dir}")

        self.ray_generator = RayGenerator()

        self.cam_up = [0.0, 0.0, 1.0] # TODO: unhardcode....
        self.fov_deg = 37.5 # TODO: unhardcode...
        
        self.fov_rad = torch.tensor(np.deg2rad(self.fov_deg)).float().view(1)
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

    def _sample_points(self, points, num_points, normals=None):
        N = points.shape[0]
        if N >= num_points:
            indices = np.random.choice(N, num_points, replace=(N < num_points))
        if normals is not None:
            return points[indices], normals[indices]
        return points[indices]

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        
        # 1. Load Image
        image_np = data['hdr_target_image']
        image_np = image_np[..., :3]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(self.image_res, self.image_res), mode='bilinear', align_corners=False).squeeze(0)
        
        # 2. Geometry & Materials
        entity_vertices = data['entity_vertices']
        entity_normals = data['entity_normals']
        entity_materials = data['entity_materials']
        
        N_obj, V, C = entity_vertices.shape
        if V != self.num_points:
            new_verts = []
            new_norms = []
            for i in range(N_obj):
                v, n = self._sample_points(entity_vertices[i], self.num_points, entity_normals[i])
                new_verts.append(v)
                new_norms.append(n)
            entity_vertices = np.stack(new_verts)
            entity_normals = np.stack(new_norms)
            
        positions = torch.from_numpy(entity_vertices).float()
        normals = torch.from_numpy(entity_normals).float()
        properties = torch.from_numpy(entity_materials).float()
        
        cam_pos = data['camera_pos']
        cam_lookat = data['camera_lookat']

        c2w = self._compute_c2w(cam_pos, cam_lookat, self.cam_up)
        w2c = torch.inverse(c2w)
        
        # if self.precomputed_ray_data is None:
        c2w_eye = torch.eye(4)
        rays_o, rays_d = self.ray_generator(c2w_eye, self.fov_rad, self.image_res)
        #     self.precomputed_ray_data = (rays_o, rays_d)
        # else:
            # rays_o, rays_d = self.precomputed_ray_data
        
        return {
            'rays_o': rays_o,               # (3,) camera origin in world space
            'rays_d': rays_d,               # (H, W, 3) - used as ray_map
            'obj_positions': positions,     # (N_obj, N_p, 3)
            'obj_normals': normals,         # (N_obj, N_p, 3)
            'obj_properties': properties,   # (N_obj, C)
            'w2c': w2c,                     # (4, 4)
            'target_image': image_tensor    # (3, H, W)
        }
