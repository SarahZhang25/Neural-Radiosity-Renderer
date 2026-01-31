
"""
Pure Mitsuba data generation utils.
"""

import numpy as np
import mitsuba as mi
import trimesh
import os
from dataclasses import dataclass

# Set variant once
try:
#     mi.set_variant('llvm_ad_rgb')
    mi.set_variant("cuda_ad_rgb")
except Exception:
    pass # Might be already set

@dataclass
class Camera:
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray = None
    fov: float = 45.0
    width: int = 512
    height: int = 512

    def __post_init__(self):
        if self.up is None:
            self.up = np.array([0.0, 1.0, 0.0])

def _save_mesh_to_ply(mesh: trimesh.Trimesh, filename: str):
    """Save a Trimesh object to a binary PLY file."""
    mesh.export(filename, file_type='ply')

def render_pure_mitsuba_scene(
    box_meshes: dict,
    object_mesh: trimesh.Trimesh,
    object_color: np.ndarray,
    camera: Camera,
    spp: int = 256,
    light_intensity: float = 15.0
) -> np.ndarray:
    """
    Render a scene with a Cornell box and a central object using Mitsuba.
    
    Args:
        box_meshes: Dictionary of trimesh.Trimesh for 'floor', 'ceiling', 'back', 'left', 'right', 'light'
        object_mesh: Trimesh object for the central object
        object_color: RGB color of the object
        camera: Camera configuration
        spp: Samples per pixel
        light_intensity: Intensity of ceiling light area
    """
    
    # Create temporary directory for meshes
    temp_dir = "/tmp/mitsuba_pure_gen"
    os.makedirs(temp_dir, exist_ok=True)
    
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path', # Use path tracer for global illumination
            'max_depth': 6,
        },
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

    # Helper to add a mesh to the scene dict
    def add_mesh_to_dict(name, mesh, bsdf, emitter=None):
        filename = os.path.join(temp_dir, f"{name}.ply")
        _save_mesh_to_ply(mesh, filename)
        
        obj_dict = {
            'type': 'ply',
            'filename': filename,
            'bsdf': bsdf
        }
        if emitter:
            obj_dict['emitter'] = emitter
            
        scene_dict[name] = obj_dict

    # Materials
    # Cornell Box colors (approximate)
    white_bsdf = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.76, 0.75, 0.50]}}
    red_bsdf = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.63, 0.06, 0.04]}}
    green_bsdf = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.09]}}
    
    # Add Walls
    if 'floor' in box_meshes: add_mesh_to_dict('floor', box_meshes['floor'], white_bsdf)
    if 'ceiling' in box_meshes: add_mesh_to_dict('ceiling', box_meshes['ceiling'], white_bsdf)
    if 'back' in box_meshes: add_mesh_to_dict('back', box_meshes['back'], white_bsdf)
    if 'left' in box_meshes: add_mesh_to_dict('left', box_meshes['left'], red_bsdf)
    if 'right' in box_meshes: add_mesh_to_dict('right', box_meshes['right'], green_bsdf)
    
    # Add Light
    if 'light' in box_meshes:
        add_mesh_to_dict('light', box_meshes['light'], 
                        {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0, 0, 0]}}, # Black surface
                        {'type': 'area', 'radiance': {'type': 'rgb', 'value': [light_intensity]*3}}
        )

    # Add Object
    # Using a principled BSDF or roughconductor/plastic to look nice
    # object_bsdf = {
    #     'type': 'roughplastic',
    #     'diffuse_reflectance': {'type': 'rgb', 'value': object_color.tolist()},
    #     'nonlinear': True
    # }
    object_bsdf = {
        'type': 'diffuse', 
        'reflectance': {'type': 'rgb', 'value': object_color.tolist()}
    }
    
    if object_mesh is not None:
        add_mesh_to_dict('object', object_mesh, object_bsdf)

    # Load and Render
    mi_scene = mi.load_dict(scene_dict)
    image = mi.render(mi_scene)
    
    # Denoise (Optional, requires OIDN)
    # image = mi.TensorXf(image) # Already tensor
    # denoiser = mi.OptixDenoiser(image.shape[:2])
    # image = denoiser(image)
    
    return np.array(image)
