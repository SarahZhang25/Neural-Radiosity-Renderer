"""
Mitsuba 3 renderer for radiosity visualization.
Uses pre-computed radiosity as base illumination with proper metallic materials.
"""

import numpy as np
import mitsuba as mi

# Use LLVM for speed without CUDA driver issues
mi.set_variant('llvm_ad_rgb')

from scene import Scene
from dataclasses import dataclass


@dataclass
class Camera:
    """Camera for Mitsuba rendering."""
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray = None
    fov: float = 45.0
    width: int = 1024
    height: int = 1024

    def __post_init__(self):
        if self.up is None:
            self.up = np.array([0.0, 1.0, 0.0])


def render_mitsuba(
    scene: Scene,
    camera: Camera,
    object_color: np.ndarray,
    roughness: float = 0.3,
    spp: int = 64,
    smooth_object: bool = True,  # Smooth shading for object
) -> np.ndarray:
    """
    Render scene using Mitsuba 3 with pre-computed radiosity.
    Fully vectorized - no Python loops.
    """
    patch_verts = np.array([p.vertices for p in scene.patches], dtype=object)
    patch_radiosity = np.array([p.radiosity for p in scene.patches])
    patch_reflectance = np.array([p.reflectance for p in scene.patches])
    patch_emission = np.array([p.emission for p in scene.patches])

    # Identify object vs wall patches (vectorized)
    is_emissive = np.sum(patch_emission, axis=1) > 0
    is_object = np.all(np.abs(patch_reflectance - object_color) < 0.001, axis=1) & ~is_emissive

    # Build wall mesh arrays
    wall_mask = ~is_object
    wall_indices = np.where(wall_mask)[0]

    # Build object mesh arrays
    obj_indices = np.where(is_object)[0]

    def build_mesh_arrays_smooth(indices, group_by_material=False):
        """Build smooth-shaded mesh (shared verts, averaged colors).
        If group_by_material=True, only merge vertices within same material group.
        """
        if len(indices) == 0:
            return None, None, None

        if group_by_material:
            # Group patches by reflectance (material)
            mat_groups = {}
            for idx in indices:
                mat_key = tuple(np.round(patch_reflectance[idx], 2))
                if mat_key not in mat_groups:
                    mat_groups[mat_key] = []
                mat_groups[mat_key].append(idx)

            # Process each material group separately then combine
            all_verts = []
            all_faces = []
            all_colors = []
            vertex_offset = 0

            for mat_key, group_indices in mat_groups.items():
                v, f, c = build_mesh_arrays_smooth(np.array(group_indices), group_by_material=False)
                if v is not None:
                    all_verts.append(v)
                    all_faces.append(f + vertex_offset)
                    all_colors.append(c)
                    vertex_offset += len(v)

            if not all_verts:
                return None, None, None

            return (np.concatenate(all_verts).astype(np.float32),
                    np.concatenate(all_faces).astype(np.uint32),
                    np.concatenate(all_colors).astype(np.float32))

        # Collect all triangles
        all_tri_verts = []
        all_tri_colors = []
        for idx in indices:
            verts = patch_verts[idx]
            rad = patch_radiosity[idx]
            n_v = len(verts)
            if n_v == 3:
                all_tri_verts.append(verts)
                all_tri_colors.append(rad)
            elif n_v == 4:
                all_tri_verts.append(verts[[0, 1, 2]])
                all_tri_colors.append(rad)
                all_tri_verts.append(verts[[0, 2, 3]])
                all_tri_colors.append(rad)

        all_tri_verts = np.array(all_tri_verts)
        all_tri_colors = np.array(all_tri_colors)

        # Compute triangle areas for area-weighted averaging
        v0 = all_tri_verts[:, 0, :]
        v1 = all_tri_verts[:, 1, :]
        v2 = all_tri_verts[:, 2, :]
        tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # (n_tri,)

        # Weld vertices
        flat_verts = all_tri_verts.reshape(-1, 3)
        rounded = np.round(flat_verts, decimals=5)
        unique_verts, inverse = np.unique(rounded, axis=0, return_inverse=True)

        # Area-weighted average color per unique vertex
        n_unique = len(unique_verts)
        color_sum = np.zeros((n_unique, 3), dtype=np.float64)
        weight_sum = np.zeros(n_unique, dtype=np.float64)
        face_colors = np.repeat(all_tri_colors, 3, axis=0)  # (n_tri*3, 3)
        weights = np.repeat(tri_area, 3)  # (n_tri*3,)
        np.add.at(color_sum, inverse, face_colors * weights[:, None])
        np.add.at(weight_sum, inverse, weights)
        avg_colors = (color_sum / np.maximum(weight_sum[:, None], 1e-12)).astype(np.float32)

        faces = inverse.reshape(-1, 3).astype(np.uint32)
        return unique_verts.astype(np.float32), faces, avg_colors

    # Smooth walls within each material group (not across wall boundaries)
    wall_verts, wall_faces, wall_colors = build_mesh_arrays_smooth(wall_indices, group_by_material=True)
    # Smooth object fully
    obj_verts, obj_faces, obj_colors = build_mesh_arrays_smooth(obj_indices, group_by_material=False)

    # Convert radiosity (exitance) -> radiance for Lambertian emitter: L = B / π
    if wall_colors is not None:
        wall_emission = np.maximum(wall_colors, 0.0) / np.pi
    if obj_colors is not None:
        obj_emission = np.maximum(obj_colors, 0.0) / np.pi
        
    # Save meshes as binary PLY (fast)
    if wall_verts is not None:
        _save_ply_binary('/tmp/mitsuba_walls.ply', wall_verts, wall_faces, wall_emission)
    if obj_verts is not None:
        _save_ply_binary('/tmp/mitsuba_object.ply', obj_verts, obj_faces, obj_emission)

    # Build Mitsuba scene
    # No extra light needed - the radiosity-emitting surfaces provide both
    # diffuse illumination AND specular reflections on the metallic car
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'direct', 'hide_emitters': False},
        'sensor': {
            'type': 'perspective',
            'fov': camera.fov,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=camera.position.tolist(),
                target=camera.look_at.tolist(),
                up=camera.up.tolist()
            ),
            'film': {
                'type': 'hdrfilm',
                'width': camera.width,
                'height': camera.height,
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {'type': 'independent', 'sample_count': spp},
        },
    }

    if wall_verts is not None:
        scene_dict['walls'] = {
            'type': 'ply',
            'filename': '/tmp/mitsuba_walls.ply',
            'face_normals': False,  # Smooth shading
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0, 0, 0]}},
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'mesh_attribute', 'name': 'vertex_color'},
            },
        }

    if obj_verts is not None:
        # Metallic car with radiosity emission + specular reflections
        scene_dict['object'] = {
            'type': 'ply',
            'filename': '/tmp/mitsuba_object.ply',
            'face_normals': not smooth_object,  # Use vertex normals for smooth shading
            'bsdf': {
                'type': 'roughconductor',
                'material': 'none',
                'specular_reflectance': {'type': 'rgb', 'value': object_color.tolist()},
                'alpha': roughness,
            },
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'mesh_attribute', 'name': 'vertex_color'},
            },
        }

    # Render
    mi_scene = mi.load_dict(scene_dict)
    image = mi.render(mi_scene)
    image_np = np.array(image)

    # Return linear HDR - tone mapping done once in main.py/save_image
    return image_np


def _save_ply_binary(filename: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    """Save binary PLY with vertex colors (fully vectorized)."""
    n_verts = len(vertices)
    n_faces = len(faces)
    # Keep HDR values, don't clamp to [0,1]
    colors = np.maximum(colors, 0).astype(np.float32)

    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_verts}
property float x
property float y
property float z
property float red
property float green
property float blue
element face {n_faces}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('ascii'))

        # Vertices with colors (interleaved) - single write
        vertex_data = np.column_stack([vertices, colors]).astype(np.float32)
        f.write(vertex_data.tobytes())

        # Faces - build as structured array for single write
        # Each face: 1 byte (count=3) + 3 int32s (indices)
        face_data = np.zeros(n_faces, dtype=[('count', 'u1'), ('v0', 'i4'), ('v1', 'i4'), ('v2', 'i4')])
        face_data['count'] = 3
        face_data['v0'] = faces[:, 0]
        face_data['v1'] = faces[:, 1]
        face_data['v2'] = faces[:, 2]
        f.write(face_data.tobytes())
