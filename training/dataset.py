import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import h5py
def scene_collate_fn(batch):
    max_objs = max(item['obj_positions'].shape[0] for item in batch)
    
    batched_data = {
        'c2w': [],
        'fov_deg': [],
        'obj_positions': [],
        'obj_normals': [],
        'obj_properties': [],
        'obj_mask': [],
        'target_image': []
    }
    
    for item in batch:
        num_objs = item['obj_positions'].shape[0]
        pad_size = max_objs - num_objs
        
        if pad_size > 0:
            obj_positions = torch.cat([item['obj_positions'], item['obj_positions'].new_zeros(pad_size, *item['obj_positions'].shape[1:])], dim=0)
            obj_normals = torch.cat([item['obj_normals'], item['obj_normals'].new_zeros(pad_size, *item['obj_normals'].shape[1:])], dim=0)
            obj_properties = torch.cat([item['obj_properties'], item['obj_properties'].new_zeros(pad_size, *item['obj_properties'].shape[1:])], dim=0)
            obj_mask = torch.cat([item['obj_mask'], item['obj_mask'].new_zeros(pad_size, *item['obj_mask'].shape[1:])], dim=0)
        else:
            obj_positions = item['obj_positions']
            obj_normals = item['obj_normals']
            obj_properties = item['obj_properties']
            obj_mask = item['obj_mask']
            
        batched_data['obj_positions'].append(obj_positions)
        batched_data['obj_normals'].append(obj_normals)
        batched_data['obj_properties'].append(obj_properties)
        batched_data['obj_mask'].append(obj_mask)
        batched_data['c2w'].append(item['c2w'])
        batched_data['fov_deg'].append(item['fov_deg'])
        batched_data['target_image'].append(item['target_image'])
        
    return {k: torch.stack(v, dim=0) for k, v in batched_data.items()}

# NOTE: is this a duplicate function now? It also exists in data_generation/to_npz_from_json_scenes.py
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

class NPZSceneDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        image_res: int = 128,
        num_points_per_object: int = 2048,
        split: str = 'train',
        max_dataset_size: int = None,
        shuffle: bool = True,
        shuffle_seed: int = 42
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
            exr_path = os.path.splitext(file_path)[0] + '_0.exr' # TODO: resolve these paths.....
            if os.path.exists(exr_path):
                try:
                    img_hdr = load_exr(exr_path)
                except Exception as e:
                    print(f"Error loading {exr_path}: {e}")
                    corrupted_paths.append(file_path)
            else:
                npz_data = np.load(file_path)
                if "hdr_target_image" not in npz_data:
                    print(f"Warning: Missing 'hdr_target_image' in {file_path}")
                    continue
                img_hdr = npz_data["hdr_target_image"]
            
            if np.isnan(img_hdr).any() or np.isinf(img_hdr).any():
                corrupted_paths.append(file_path)

        self.files = [item for item in self.files if item not in corrupted_paths]
        print(f"Removed {len(corrupted_paths)} corrupted images")

        # Shuffle with a fixed seed
        if shuffle:
            rng = np.random.RandomState(shuffle_seed)
            rng.shuffle(self.files)
        
        if max_dataset_size is not None and len(self.files) > max_dataset_size:
            self.files = self.files[:max_dataset_size]
            
        if split == "all":
            print(f"[{split}] Using all {len(self.files)} samples in {data_dir}")
        else:
            assert split in ['train', 'val'], "split must be 'train', 'val', or 'all'"
            split_idx = int(len(self.files) * 0.9)
            if split == 'train':
                self.files = self.files[:split_idx]
            else:
                self.files = self.files[split_idx:]

        print(f"[{split}] Found {len(self.files)} samples in {data_dir}")

        self.cam_up = np.array([0.0, 0.0, 1.0])  # NOTE: may need to unhardcode...

    def _compute_c2w(self, pos, target, up):
        """Build a camera-to-world 4x4 matrix. Camera looks down -Z."""
        z_axis = pos - target
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

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
        if 'hdr_target_image' in data:
            image_np = data['hdr_target_image']
        else:
            exr_path = os.path.splitext(file_path)[0] + '_0.exr' # TODO: resolve this path
            image_np = load_exr(exr_path)
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
        mask = torch.ones(positions.shape[0], dtype=torch.bool)
        
        cam_fov = data.get('camera_fov', 37.5)  # Default FOV if not present
        cam_pos = data['camera_pos']
        cam_lookat = data['camera_lookat']
        c2w = self._compute_c2w(cam_pos, cam_lookat, self.cam_up)

        return {
            'c2w': c2w,                             # (4, 4) camera-to-world
            'fov_deg': torch.tensor(cam_fov).float(),  # scalar
            'obj_positions': positions,             # (N_obj, N_p, 3)
            'obj_normals': normals,                 # (N_obj, N_p, 3)
            'obj_properties': properties,           # (N_obj, C)
            'obj_mask': mask,                       # (N_obj)
            'target_image': image_tensor            # (3, H, W)
        }

class H5SceneDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        image_res: int = 128,
        num_points_per_object: int = 2048,
        split: str = 'train',
        max_dataset_size: int = None,
        shuffle: bool = True,
        shuffle_seed: int = 42
    ):
        self.data_dir = data_dir
        self.image_res = image_res
        self.num_points = num_points_per_object
        
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        self.scene_index = []
        
        # Build index mapping global_idx -> (chunk_file, scene_name)
        for chunk_file in self.chunk_files:
            with h5py.File(chunk_file, 'r') as f:
                for scene_name in f.keys():
                    self.scene_index.append((chunk_file, scene_name))
        
        # Handle shuffling and dataset limits BEFORE splitting
        if shuffle:
            rng = np.random.RandomState(shuffle_seed)
            # Convert to list for shuffling
            idx_list = list(range(len(self.scene_index)))
            rng.shuffle(idx_list)
            self.scene_index = [self.scene_index[i] for i in idx_list]
            
        if max_dataset_size is not None and len(self.scene_index) > max_dataset_size:
            self.scene_index = self.scene_index[:max_dataset_size]
            
        # Handle splitting
        if split == "all":
            print(f"[{split}] Using all {len(self.scene_index)} samples in {data_dir}")
        else:
            assert split in ['train', 'val'], "split must be 'train', 'val', or 'all'"
            split_idx = int(len(self.scene_index) * 0.9)
            if split == 'train':
                self.scene_index = self.scene_index[:split_idx]
            else:
                self.scene_index = self.scene_index[split_idx:]
                
        print(f"[{split}] Found {len(self.scene_index)} samples across {len(self.chunk_files)} chunks in {data_dir}")

        # Lazily store opened H5 handles per worker to avoid multiprocess fork issues
        self._h5_handles = {}

    def _get_h5_file(self, chunk_path):
        if chunk_path not in self._h5_handles:
            # swmr=True enables Single Writer Multiple Reader, safe for multiprocess dataloading
            self._h5_handles[chunk_path] = h5py.File(chunk_path, 'r', swmr=True)
        return self._h5_handles[chunk_path]

    def _sample_points(self, points, num_points, normals=None):
        N = points.shape[0]
        if N >= num_points:
            indices = np.random.choice(N, num_points, replace=(N < num_points))
        else:
            indices = np.random.choice(N, num_points, replace=True)
        if normals is not None:
            return points[indices], normals[indices]
        return points[indices]

    def __len__(self):
        return len(self.scene_index)

    def __getitem__(self, idx):
        chunk_file, scene_name = self.scene_index[idx]
        f = self._get_h5_file(chunk_file)
        grp = f[scene_name]
        
        # 1. Load Image
        image_np = grp['hdr_target_image'][:]
        image_np = image_np[..., :3]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        # We can dynamically scale down based on self.image_res
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), 
            size=(self.image_res, self.image_res), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # 2. Geometry & Materials
        entity_vertices = grp['entity_vertices'][:]
        entity_normals = grp['entity_normals'][:]
        entity_materials = grp['entity_materials'][:]
        
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
        mask = torch.ones(positions.shape[0], dtype=torch.bool)
        
        # 3. Camera config
        cam_fov = grp['camera_fov'][()]
        c2w = torch.from_numpy(grp['c2w'][:]).float()

        return {
            'c2w': c2w,                             # (4, 4) camera-to-world
            'fov_deg': torch.tensor(cam_fov).float(),  # scalar
            'obj_positions': positions,             # (N_obj, N_p, 3)
            'obj_normals': normals,                 # (N_obj, N_p, 3)
            'obj_properties': properties,           # (N_obj, C)
            'obj_mask': mask,                       # (N_obj)
            'target_image': image_tensor            # (3, H, W)
        }

    def __del__(self):
        # Close all H5 handles on destruction
        for f in self._h5_handles.values():
            try:
                f.close()
            except Exception:
                pass
