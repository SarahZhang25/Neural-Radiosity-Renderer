"""
Camera and rendering for radiosity visualization.
Uses pyrender for fast GPU-accelerated rendering.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import trimesh

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False

from patches import Patch
from data_generation_old.scene import Scene


@dataclass
class Camera:
    """Perspective camera for rendering."""
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray = None
    fov: float = 45.0
    width: int = 64
    height: int = 64

    def __post_init__(self):
        if self.up is None:
            self.up = np.array([0.0, 1.0, 0.0])

        self.forward = self.look_at - self.position
        self.forward = self.forward / np.linalg.norm(self.forward)

        self.right = np.cross(self.forward, self.up)
        self.right = self.right / np.linalg.norm(self.right)

        self.up_vec = np.cross(self.right, self.forward)

        self.aspect = self.width / self.height
        self.tan_half_fov = np.tan(np.radians(self.fov / 2))

    def get_rays_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all rays for the entire image (one per pixel)."""
        py, px = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        px_flat = px.flatten()
        py_flat = py.flatten()

        # Center of each pixel
        ndc_x = (2.0 * (px_flat + 0.5) / self.width - 1.0) * self.aspect * self.tan_half_fov
        ndc_y = (1.0 - 2.0 * (py_flat + 0.5) / self.height) * self.tan_half_fov

        directions = (
            self.forward[np.newaxis, :] +
            ndc_x[:, np.newaxis] * self.right[np.newaxis, :] +
            ndc_y[:, np.newaxis] * self.up_vec[np.newaxis, :]
        )
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        origins = np.tile(self.position, (len(ndc_x), 1))

        return origins, directions

    def get_ray_directions_vectorized(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Get ray directions for specific pixel coordinates (vectorized)."""
        ndc_x = (2.0 * (px + 0.5) / self.width - 1.0) * self.aspect * self.tan_half_fov
        ndc_y = (1.0 - 2.0 * (py + 0.5) / self.height) * self.tan_half_fov

        directions = (
            self.forward[np.newaxis, :] +
            ndc_x[:, np.newaxis] * self.right[np.newaxis, :] +
            ndc_y[:, np.newaxis] * self.up_vec[np.newaxis, :]
        )
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        return directions


def render(
    scene: Scene,
    camera: Camera,
    save_contributions: bool = False,
    background_color: np.ndarray = None
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Render the scene showing computed radiosity values directly.
    No additional shading - radiosity already includes all lighting.
    """
    if background_color is None:
        background_color = np.array([0.0, 0.0, 0.0])

    scene.build_combined_mesh()
    mesh = scene.get_trimesh()

    origins, directions = camera.get_rays_batch()

    # Cast rays
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    # Initialize image
    image = np.zeros((camera.height, camera.width, 3))
    image[:] = background_color

    contributions = {} if save_contributions else None

    if len(locations) == 0:
        return image, contributions

    # Get patch indices for all hits
    patch_indices = np.array([scene.face_to_patch.get(tri_idx, -1) for tri_idx in index_tri])

    # Get radiosity for all patches
    all_radiosity = np.array([p.radiosity for p in scene.patches])

    # Valid hits
    valid = patch_indices >= 0
    valid_patch_idx = patch_indices[valid]
    valid_ray_idx = index_ray[valid]

    if len(valid_patch_idx) == 0:
        return image, contributions

    # Get radiosity values directly - no additional shading
    colors = all_radiosity[valid_patch_idx]

    # Write to image
    y_coords = valid_ray_idx // camera.width
    x_coords = valid_ray_idx % camera.width
    image[y_coords, x_coords] = colors

    # Save contributions
    if save_contributions:
        for _, (x, y, pidx) in enumerate(zip(x_coords, y_coords, valid_patch_idx)):
            contributions[(x, y)] = pidx

    return image, contributions


def render_with_interpolation(
    scene: Scene,
    camera: Camera,
    save_contributions: bool = False,
    background_color: np.ndarray = None,
    n_shadow_samples: int = 4,
    verbose: bool = False
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Render with radiosity interpolation and soft shadows.

    Args:
        n_shadow_samples: Number of shadow samples for soft shadows (e.g., 16 = 4x4 grid)
    """
    if background_color is None:
        background_color = np.array([0.0, 0.0, 0.0])

    scene.build_combined_mesh()
    mesh = scene.get_trimesh()

    # Pre-compute vertex radiosity
    vertex_radiosity = compute_vertex_radiosity(scene)

    # Find light sources
    light_patches = [p for p in scene.patches if np.sum(p.emission) > 0]

    # Get rays (one per pixel)
    if verbose:
        print("      Generating rays...")
    origins, directions = camera.get_rays_batch()
    n_pixels = camera.width * camera.height

    if verbose:
        print(f"      Casting {n_pixels} rays...")
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    # Initialize image
    image = np.zeros((camera.height, camera.width, 3))
    image[:] = background_color

    contributions = {} if save_contributions else None

    if len(locations) == 0:
        return image, contributions

    hit_points = locations
    n_hits = len(hit_points)
    if verbose:
        print(f"      {n_hits} hits, computing shading...")

    # Get patch indices
    patch_indices = np.array([scene.face_to_patch.get(tri_idx, -1) for tri_idx in index_tri])

    # Barycentric interpolation for smooth shading
    faces = scene.faces[index_tri]
    v0 = scene.vertices[faces[:, 0]]
    v1 = scene.vertices[faces[:, 1]]
    v2 = scene.vertices[faces[:, 2]]

    bary = barycentric_coords_batch(hit_points, v0, v1, v2)

    r0 = vertex_radiosity[faces[:, 0]]
    r1 = vertex_radiosity[faces[:, 1]]
    r2 = vertex_radiosity[faces[:, 2]]
    base_colors = bary[:, 0:1] * r0 + bary[:, 1:2] * r1 + bary[:, 2:3] * r2

    # Fully vectorized soft shadow computation
    if light_patches and n_shadow_samples > 0:
        if verbose:
            print(f"      Computing shadows (vectorized, {n_shadow_samples} samples)...")

        all_normals = np.array([p.normal for p in scene.patches])
        hit_normals = np.zeros((n_hits, 3))
        valid_patches = patch_indices >= 0
        hit_normals[valid_patches] = all_normals[patch_indices[valid_patches]]

        shadow_origins_base = hit_points + hit_normals * 0.01

        # Generate stratified light samples (fixed grid)
        light = light_patches[0]
        light_verts = light.vertices
        sqrt_samples = int(np.sqrt(n_shadow_samples))
        actual_samples = sqrt_samples * sqrt_samples

        # Create grid of sample points on light
        su, sv = np.meshgrid(
            (np.arange(sqrt_samples) + 0.5) / sqrt_samples,
            (np.arange(sqrt_samples) + 0.5) / sqrt_samples
        )
        su = su.flatten()
        sv = sv.flatten()

        if len(light_verts) == 4:
            light_samples = ((1-su)*(1-sv))[:, None]*light_verts[0] + \
                           (su*(1-sv))[:, None]*light_verts[1] + \
                           (su*sv)[:, None]*light_verts[2] + \
                           ((1-su)*sv)[:, None]*light_verts[3]
        else:
            # For triangles, fold coordinates
            mask = su + sv > 1
            su[mask], sv[mask] = 1 - su[mask], 1 - sv[mask]
            light_samples = ((1-su-sv))[:, None]*light_verts[0] + \
                           su[:, None]*light_verts[1] + \
                           sv[:, None]*light_verts[2]

        # Create ALL shadow rays at once: n_hits * actual_samples rays
        # Each hit point connects to each light sample
        shadow_origins = np.repeat(shadow_origins_base, actual_samples, axis=0)  # (n_hits * samples, 3)
        light_targets = np.tile(light_samples, (n_hits, 1))  # (n_hits * samples, 3)

        shadow_dirs = light_targets - shadow_origins
        shadow_dists = np.linalg.norm(shadow_dirs, axis=1)
        shadow_dirs = shadow_dirs / (shadow_dists[:, np.newaxis] + 1e-8)

        # Cast ALL shadow rays at once
        try:
            locs, idx_ray, _ = mesh.ray.intersects_location(
                ray_origins=shadow_origins,
                ray_directions=shadow_dirs,
                multiple_hits=False
            )

            # Initialize all as visible
            all_visible = np.ones(n_hits * actual_samples, dtype=np.float32)

            if len(locs) > 0:
                hit_dists = np.linalg.norm(locs - shadow_origins[idx_ray], axis=1)
                blocked = hit_dists < shadow_dists[idx_ray] - 0.02
                all_visible[idx_ray[blocked]] = 0.0

            # Reshape to (n_hits, actual_samples) and average
            all_visible = all_visible.reshape(n_hits, actual_samples)
            shadow_visibility = np.mean(all_visible, axis=1)

        except Exception:
            shadow_visibility = np.ones(n_hits, dtype=np.float32)

        # Apply shadows
        ambient_factor = 0.3
        shadow_factor = ambient_factor + (1 - ambient_factor) * shadow_visibility[:, np.newaxis]
        colors = base_colors * shadow_factor
    else:
        colors = base_colors

    # Write to image
    y_coords = index_ray // camera.width
    x_coords = index_ray % camera.width
    image[y_coords, x_coords] = colors

    # Save contributions
    if save_contributions:
        for i, pidx in enumerate(patch_indices):
            y = index_ray[i] // camera.width
            x = index_ray[i] % camera.width
            contributions[(x, y)] = pidx

    return image, contributions


def barycentric_coords_batch(
    points: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> np.ndarray:
    """Compute barycentric coordinates for multiple points (vectorized)."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    v0_to_point = points - v0

    d00 = np.sum(edge1 * edge1, axis=1)
    d01 = np.sum(edge1 * edge2, axis=1)
    d11 = np.sum(edge2 * edge2, axis=1)
    d20 = np.sum(v0_to_point * edge1, axis=1)
    d21 = np.sum(v0_to_point * edge2, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.stack([u, v, w], axis=1)


def compute_vertex_radiosity(scene: Scene) -> np.ndarray:
    """Compute per-vertex radiosity by averaging adjacent patch values."""
    n_vertices = len(scene.vertices)
    vertex_radiosity = np.zeros((n_vertices, 3))
    vertex_counts = np.zeros(n_vertices)

    for face_idx, face in enumerate(scene.faces):
        patch_idx = scene.face_to_patch.get(face_idx, -1)
        if patch_idx >= 0 and patch_idx < len(scene.patches):
            radiosity = scene.patches[patch_idx].radiosity
            for v_idx in face:
                vertex_radiosity[v_idx] += radiosity
                vertex_counts[v_idx] += 1

    nonzero = vertex_counts > 0
    vertex_radiosity[nonzero] /= vertex_counts[nonzero, np.newaxis]

    return vertex_radiosity


def compute_vertex_normals(scene: Scene) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    n_vertices = len(scene.vertices)
    vertex_normals = np.zeros((n_vertices, 3))
    vertex_counts = np.zeros(n_vertices)

    for face_idx, face in enumerate(scene.faces):
        patch_idx = scene.face_to_patch.get(face_idx, -1)
        if patch_idx >= 0 and patch_idx < len(scene.patches):
            normal = scene.patches[patch_idx].normal
            for v_idx in face:
                vertex_normals[v_idx] += normal
                vertex_counts[v_idx] += 1

    nonzero = vertex_counts > 0
    vertex_normals[nonzero] /= vertex_counts[nonzero, np.newaxis]

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vertex_normals = vertex_normals / norms

    return vertex_normals


def compute_per_pixel_shadows(
    scene: Scene,
    camera: Camera,
    depth_buffer: np.ndarray,
    n_shadow_samples: int = 4
) -> np.ndarray:
    """
    Compute per-pixel shadow factor by casting rays to light source.
    Returns shadow map (0=shadow, 1=lit) at image resolution.
    """
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    height, width = camera.height, camera.width
    shadow_map = np.ones((height, width), dtype=np.float32)

    # Get scene mesh for ray casting
    mesh = scene.get_trimesh()
    ray_intersector = RayMeshIntersector(mesh)

    # Find light patches
    light_patches = [p for p in scene.patches if np.sum(p.emission) > 0]
    if not light_patches:
        return shadow_map

    # Get light center and sample points
    light = light_patches[0]
    light_center = light.center
    light_verts = light.vertices

    # Generate light sample points
    if len(light_verts) >= 4:
        u = np.random.rand(n_shadow_samples)
        v = np.random.rand(n_shadow_samples)
        light_samples = (
            np.outer((1-u)*(1-v), light_verts[0]) +
            np.outer(u*(1-v), light_verts[1]) +
            np.outer(u*v, light_verts[2]) +
            np.outer((1-u)*v, light_verts[3])
        )
    else:
        light_samples = np.array([light_center])
        n_shadow_samples = 1

    # For each pixel, compute 3D position and cast shadow ray
    # Only process pixels with valid depth (not background)
    valid_mask = depth_buffer < 100.0  # Assuming far plane is 100

    # Generate rays for all pixels
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixel_dirs = camera.get_ray_directions_vectorized(
        x_coords.flatten().astype(np.float32),
        y_coords.flatten().astype(np.float32)
    )

    # Compute 3D positions from depth
    depths = depth_buffer.flatten()
    positions = camera.position + pixel_dirs * depths[:, np.newaxis]

    # Process in batches for memory efficiency
    batch_size = 50000
    valid_indices = np.where(valid_mask.flatten())[0]

    for batch_start in range(0, len(valid_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]

        batch_positions = positions[batch_indices]
        n_batch = len(batch_indices)

        # Cast rays to each light sample
        visibility = np.zeros(n_batch, dtype=np.float32)

        for light_point in light_samples:
            ray_dirs = light_point - batch_positions
            ray_dists = np.linalg.norm(ray_dirs, axis=1)
            ray_dirs = ray_dirs / (ray_dists[:, np.newaxis] + 1e-8)

            # Offset origin slightly to avoid self-intersection
            ray_origins = batch_positions + ray_dirs * 0.01

            # Cast rays
            locations, index_ray, _ = ray_intersector.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_dirs,
                multiple_hits=False
            )

            # Check which rays are NOT blocked
            sample_vis = np.ones(n_batch, dtype=np.float32)
            if len(locations) > 0:
                hit_dists = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
                blocked = hit_dists < ray_dists[index_ray] - 0.02
                sample_vis[index_ray[blocked]] = 0.0

            visibility += sample_vis

        visibility /= n_shadow_samples

        # Store in shadow map
        shadow_map.flat[batch_indices] = visibility

    return shadow_map


def render_pyrender(
    scene: Scene,
    camera: Camera,
    exposure: float = 1.0,
    background_color: np.ndarray = None,
    per_pixel_shadows: bool = True
) -> np.ndarray:
    """
    Fast GPU-accelerated rendering using pyrender.
    Uses smooth vertex colors + optional per-pixel shadow pass.
    """
    if not PYRENDER_AVAILABLE:
        raise ImportError("pyrender not available. Install with: pip install pyrender")

    if background_color is None:
        background_color = np.array([0.0, 0.0, 0.0])

    # Group patches by material (reflectance) to avoid color bleeding at boundaries
    from collections import defaultdict
    material_groups = defaultdict(list)
    for patch_idx, patch in enumerate(scene.patches):
        # Use reflectance as material key (round to avoid floating point issues)
        mat_key = tuple(np.round(patch.reflectance, 3))
        material_groups[mat_key].append(patch_idx)

    # Create pyrender scene
    pr_scene = pyrender.Scene(
        bg_color=np.append(background_color, 1.0),
        ambient_light=[1.0, 1.0, 1.0]
    )

    # Build separate mesh for each material group
    for mat_key, patch_indices in material_groups.items():
        all_vertices = []
        all_faces = []
        all_radiosity = []
        vertex_offset = 0

        for patch_idx in patch_indices:
            patch = scene.patches[patch_idx]
            n_verts = len(patch.vertices)
            all_vertices.extend(patch.vertices)
            all_radiosity.extend([patch.radiosity] * n_verts)

            # Create faces for this patch
            if n_verts == 3:
                all_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            elif n_verts == 4:
                all_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                all_faces.append([vertex_offset, vertex_offset + 2, vertex_offset + 3])
            else:
                for i in range(1, n_verts - 1):
                    all_faces.append([vertex_offset, vertex_offset + i, vertex_offset + i + 1])

            vertex_offset += n_verts

        if not all_vertices:
            continue

        vertices = np.array(all_vertices)
        faces = np.array(all_faces)
        radiosity = np.array(all_radiosity)

        # Apply ACES tone mapping
        v_colors = radiosity * exposure
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        v_colors = (v_colors * (a * v_colors + b)) / (v_colors * (c * v_colors + d) + e)
        v_colors = np.clip(v_colors, 0, 1)
        v_colors = np.power(v_colors, 1.0/2.2)

        # Convert to uint8 RGBA
        vertex_colors = np.zeros((len(v_colors), 4), dtype=np.uint8)
        vertex_colors[:, :3] = (v_colors * 255).astype(np.uint8)
        vertex_colors[:, 3] = 255

        # Create trimesh for this material
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        # smooth=False for sharp flat shading on object surfaces
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        pr_scene.add(pyrender_mesh)

    # Build combined mesh for shadow ray tracing (still needed for per-pixel shadows)
    scene.build_combined_mesh()

    # Create camera
    pr_camera = pyrender.PerspectiveCamera(
        yfov=np.radians(camera.fov),
        aspectRatio=camera.width / camera.height,
        znear=0.01,
        zfar=100.0
    )

    # Camera pose - pyrender uses OpenGL convention (camera looks down -Z)
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = camera.right
    camera_pose[:3, 1] = camera.up_vec
    camera_pose[:3, 2] = -camera.forward
    camera_pose[:3, 3] = camera.position

    pr_scene.add(pr_camera, pose=camera_pose)

    # Render color and depth
    renderer = pyrender.OffscreenRenderer(camera.width, camera.height)
    color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.FLAT)
    renderer.delete()

    # Convert to float
    image = color.astype(np.float32) / 255.0

    # Apply per-pixel shadow pass for sharp shadows
    if per_pixel_shadows:
        print("      Computing per-pixel shadows...")
        shadow_map = compute_per_pixel_shadows(scene, camera, depth, n_shadow_samples=4)

        # Apply shadow map to simulate direct light occlusion
        # In a Cornell box with multiple bounces, indirect light dominates
        # Shadow only affects direct component: final = indirect + direct * shadow
        # Approximation: final = base_color * (indirect_ratio + direct_ratio * shadow)
        # With indirect_ratio=0.7, direct_ratio=0.3:
        #   lit areas: 0.7 + 0.3*1.0 = 1.0
        #   shadow areas: 0.7 + 0.3*0.0 = 0.7
        indirect_ratio = 0.7
        direct_ratio = 0.3
        shadow_factor = indirect_ratio + direct_ratio * shadow_map[:, :, np.newaxis]
        image = image * shadow_factor

    return image


def render_glossy(
    scene: Scene,
    camera: Camera,
    exposure: float = 1.0,
    background_color: np.ndarray = None,
    per_pixel_shadows: bool = True,
    # Object material - REQUIRED to identify which geometry is the object
    object_color: np.ndarray = None,  # The reflectance color of the object
    # Material parameters for glossy/metallic look
    metallic: float = 0.9,        # How metallic (0=dielectric, 1=metal)
    roughness: float = 0.15,      # Surface roughness (0=mirror, 1=diffuse)
    clearcoat: float = 0.8,       # Clearcoat intensity for car paint
    env_intensity: float = 0.3,   # Environment reflection intensity
) -> np.ndarray:
    """
    Render with glossy/metallic materials for realistic car paint appearance.
    Combines radiosity diffuse with specular highlights and Fresnel reflections.

    Args:
        object_color: The exact reflectance color of the object to apply glossy effects to.
                     This is matched exactly (atol=0.001) to identify object patches.
    """
    if not PYRENDER_AVAILABLE:
        raise ImportError("pyrender not available")

    if background_color is None:
        background_color = np.array([0.0, 0.0, 0.0])

    if object_color is None:
        raise ValueError("object_color must be provided to identify which geometry gets glossy material")

    # Simple and reliable: match exact object color
    def is_object_material(reflectance):
        return np.allclose(reflectance, object_color, atol=0.001)

    # Build geometry with normals for specular computation
    all_vertices = []
    all_faces = []
    all_normals = []  # Per-vertex normals
    all_radiosity = []
    all_is_object = []  # Boolean mask for object vs wall
    all_reflectance = []
    vertex_offset = 0

    for patch_idx, patch in enumerate(scene.patches):
        n_verts = len(patch.vertices)
        all_vertices.extend(patch.vertices)
        all_normals.extend([patch.normal] * n_verts)
        all_radiosity.extend([patch.radiosity] * n_verts)
        all_reflectance.extend([patch.reflectance] * n_verts)

        is_obj = is_object_material(patch.reflectance) and np.sum(patch.emission) == 0
        all_is_object.extend([is_obj] * n_verts)

        if n_verts == 3:
            all_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
        elif n_verts == 4:
            all_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            all_faces.append([vertex_offset, vertex_offset + 2, vertex_offset + 3])
        else:
            for i in range(1, n_verts - 1):
                all_faces.append([vertex_offset, vertex_offset + i, vertex_offset + i + 1])

        vertex_offset += n_verts

    vertices = np.array(all_vertices)
    faces = np.array(all_faces)
    normals = np.array(all_normals)
    radiosity = np.array(all_radiosity)
    is_object = np.array(all_is_object)
    reflectance = np.array(all_reflectance)

    # Create mesh with vertex attributes
    # We'll render multiple passes to get all the data we need

    # Pass 1: Render radiosity colors
    pr_scene = pyrender.Scene(
        bg_color=np.append(background_color, 1.0),
        ambient_light=[1.0, 1.0, 1.0]
    )

    # Apply tone mapping to radiosity
    v_colors = radiosity * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    v_colors = (v_colors * (a * v_colors + b)) / (v_colors * (c * v_colors + d) + e)
    v_colors = np.clip(v_colors, 0, 1)
    v_colors = np.power(v_colors, 1.0/2.2)

    vertex_colors = np.zeros((len(v_colors), 4), dtype=np.uint8)
    vertex_colors[:, :3] = (v_colors * 255).astype(np.uint8)
    vertex_colors[:, 3] = 255

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    pr_scene.add(pyrender_mesh)

    # Camera setup
    pr_camera = pyrender.PerspectiveCamera(
        yfov=np.radians(camera.fov),
        aspectRatio=camera.width / camera.height,
        znear=0.01, zfar=100.0
    )
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = camera.right
    camera_pose[:3, 1] = camera.up_vec
    camera_pose[:3, 2] = -camera.forward
    camera_pose[:3, 3] = camera.position
    pr_scene.add(pr_camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(camera.width, camera.height)
    color_img, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.FLAT)
    renderer.delete()

    # Pass 2: Render normals (encode as RGB)
    pr_scene2 = pyrender.Scene(
        bg_color=[0.5, 0.5, 0.5, 1.0],
        ambient_light=[1.0, 1.0, 1.0]
    )

    # Encode normals as colors: normal * 0.5 + 0.5
    normal_colors = np.zeros((len(normals), 4), dtype=np.uint8)
    normal_colors[:, :3] = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
    normal_colors[:, 3] = 255

    mesh2 = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=normal_colors)
    pyrender_mesh2 = pyrender.Mesh.from_trimesh(mesh2, smooth=False)
    pr_scene2.add(pyrender_mesh2)
    pr_scene2.add(pr_camera, pose=camera_pose)

    renderer2 = pyrender.OffscreenRenderer(camera.width, camera.height)
    normal_img, _ = renderer2.render(pr_scene2, flags=pyrender.RenderFlags.FLAT)
    renderer2.delete()

    # Pass 3: Render object mask
    pr_scene3 = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 1.0],
        ambient_light=[1.0, 1.0, 1.0]
    )

    mask_colors = np.zeros((len(is_object), 4), dtype=np.uint8)
    mask_colors[:, 0] = (is_object * 255).astype(np.uint8)
    mask_colors[:, 3] = 255

    mesh3 = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=mask_colors)
    pyrender_mesh3 = pyrender.Mesh.from_trimesh(mesh3, smooth=False)
    pr_scene3.add(pyrender_mesh3)
    pr_scene3.add(pr_camera, pose=camera_pose)

    renderer3 = pyrender.OffscreenRenderer(camera.width, camera.height)
    mask_img, _ = renderer3.render(pr_scene3, flags=pyrender.RenderFlags.FLAT)
    renderer3.delete()

    # Pass 4: Render base reflectance for object
    pr_scene4 = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 1.0],
        ambient_light=[1.0, 1.0, 1.0]
    )

    refl_colors = np.zeros((len(reflectance), 4), dtype=np.uint8)
    refl_colors[:, :3] = (reflectance * 255).astype(np.uint8)
    refl_colors[:, 3] = 255

    mesh4 = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=refl_colors)
    pyrender_mesh4 = pyrender.Mesh.from_trimesh(mesh4, smooth=False)
    pr_scene4.add(pyrender_mesh4)
    pr_scene4.add(pr_camera, pose=camera_pose)

    renderer4 = pyrender.OffscreenRenderer(camera.width, camera.height)
    refl_img, _ = renderer4.render(pr_scene4, flags=pyrender.RenderFlags.FLAT)
    renderer4.delete()

    # Convert to float
    base_color = color_img.astype(np.float32) / 255.0
    pixel_normals = (normal_img.astype(np.float32) / 255.0 - 0.5) * 2.0
    object_mask = mask_img[:, :, 0].astype(np.float32) / 255.0
    base_reflectance = refl_img.astype(np.float32) / 255.0

    # Normalize pixel normals
    norm_lengths = np.linalg.norm(pixel_normals, axis=2, keepdims=True)
    norm_lengths = np.maximum(norm_lengths, 1e-8)
    pixel_normals = pixel_normals / norm_lengths

    # Compute view direction per pixel
    h, w = camera.height, camera.width
    py, px = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Convert pixel coords to NDC
    ndc_x = (2.0 * px / w - 1.0) * camera.aspect * camera.tan_half_fov
    ndc_y = (1.0 - 2.0 * py / h) * camera.tan_half_fov

    # Ray directions in world space
    ray_dirs = (ndc_x[:, :, np.newaxis] * camera.right +
                ndc_y[:, :, np.newaxis] * camera.up_vec +
                camera.forward)
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)

    view_dirs = -ray_dirs  # View direction points toward camera

    # Light position (center of ceiling light in Cornell box)
    box_size = 2.0
    light_pos = np.array([0.0, box_size - 0.01, 0.0])
    light_color = np.array([1.0, 0.98, 0.95])  # Warm white

    # Reconstruct world positions from depth
    # Approximate: use ray direction * depth
    valid_depth = depth > 0
    world_pos = np.zeros((h, w, 3))
    world_pos[valid_depth] = camera.position + ray_dirs[valid_depth] * depth[valid_depth, np.newaxis]

    # Light direction per pixel
    light_dirs = light_pos - world_pos
    light_dist = np.linalg.norm(light_dirs, axis=2, keepdims=True)
    light_dist = np.maximum(light_dist, 1e-8)
    light_dirs = light_dirs / light_dist

    # Half vector for Blinn-Phong
    half_vectors = light_dirs + view_dirs
    half_vectors = half_vectors / np.linalg.norm(half_vectors, axis=2, keepdims=True)

    # Specular computation (GGX-like)
    NdotH = np.sum(pixel_normals * half_vectors, axis=2)
    NdotH = np.clip(NdotH, 0, 1)
    NdotV = np.sum(pixel_normals * view_dirs, axis=2)
    NdotV = np.clip(NdotV, 0, 1)
    NdotL = np.sum(pixel_normals * light_dirs, axis=2)
    NdotL = np.clip(NdotL, 0, 1)

    # GGX distribution
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0
    D = alpha2 / (np.pi * denom * denom + 1e-8)

    # Fresnel (Schlick approximation)
    F0 = 0.04 + metallic * (base_reflectance - 0.04)  # Metallic uses base color as F0
    F = F0 + (1.0 - F0) * np.power(1.0 - NdotV[:, :, np.newaxis], 5)

    # Geometry term (simplified)
    k = (roughness + 1.0) ** 2 / 8.0
    G1_V = NdotV / (NdotV * (1 - k) + k + 1e-8)
    G1_L = NdotL / (NdotL * (1 - k) + k + 1e-8)
    G = G1_V * G1_L

    # Specular BRDF
    specular = D[:, :, np.newaxis] * F * G[:, :, np.newaxis] / (4.0 * NdotV[:, :, np.newaxis] * NdotL[:, :, np.newaxis] + 1e-8)
    specular = specular * NdotL[:, :, np.newaxis] * light_color
    specular = np.clip(specular, 0, 5)  # Clamp specular highlights

    # Environment reflection (simple box reflection)
    # Reflect view direction around normal
    reflect_dirs = 2.0 * NdotV[:, :, np.newaxis] * pixel_normals - view_dirs

    # Sample environment color based on reflection direction
    # Approximate with Cornell box colors
    env_color = np.zeros((h, w, 3))
    # Right wall (green) if reflecting toward +X
    env_color += np.maximum(reflect_dirs[:, :, 0:1], 0) * np.array([0.12, 0.45, 0.15])
    # Left wall (red) if reflecting toward -X
    env_color += np.maximum(-reflect_dirs[:, :, 0:1], 0) * np.array([0.65, 0.05, 0.05])
    # Ceiling (white/light) if reflecting toward +Y
    env_color += np.maximum(reflect_dirs[:, :, 1:2], 0) * np.array([0.9, 0.88, 0.85])
    # Floor (white) if reflecting toward -Y
    env_color += np.maximum(-reflect_dirs[:, :, 1:2], 0) * np.array([0.73, 0.73, 0.73])
    # Back wall if reflecting toward -Z
    env_color += np.maximum(-reflect_dirs[:, :, 2:3], 0) * np.array([0.73, 0.73, 0.73])

    env_reflection = env_color * F * env_intensity

    # Clearcoat layer (additional sharp reflection)
    clearcoat_roughness = 0.05
    clearcoat_alpha = clearcoat_roughness * clearcoat_roughness
    clearcoat_alpha2 = clearcoat_alpha * clearcoat_alpha
    clearcoat_denom = NdotH * NdotH * (clearcoat_alpha2 - 1.0) + 1.0
    clearcoat_D = clearcoat_alpha2 / (np.pi * clearcoat_denom * clearcoat_denom + 1e-8)
    clearcoat_F = 0.04 + 0.96 * np.power(1.0 - NdotV, 5)
    # Combine scalar terms first, then expand to 3 channels
    clearcoat_scalar = clearcoat_D * clearcoat_F * NdotL * clearcoat
    clearcoat_spec = clearcoat_scalar[:, :, np.newaxis] * light_color

    # Combine: diffuse + specular + environment + clearcoat
    # Only apply glossy effects to object pixels
    final_image = base_color.copy()

    # For object pixels, blend in specular and reflections
    obj_mask_3ch = object_mask[:, :, np.newaxis]

    # Tone map the specular contribution
    spec_contribution = specular * 0.5 + env_reflection + clearcoat_spec * 0.3
    spec_contribution = spec_contribution * exposure
    spec_contribution = (spec_contribution * (a * spec_contribution + b)) / (spec_contribution * (c * spec_contribution + d) + e)
    spec_contribution = np.clip(spec_contribution, 0, 1)
    spec_contribution = np.power(spec_contribution, 1.0/2.2)

    # Add specular to object areas
    final_image = final_image + obj_mask_3ch * spec_contribution
    final_image = np.clip(final_image, 0, 1)

    # Apply per-pixel shadows if requested
    if per_pixel_shadows:
        scene.build_combined_mesh()
        print("      Computing per-pixel shadows...")
        shadow_map = compute_per_pixel_shadows(scene, camera, depth, n_shadow_samples=4)
        indirect_ratio = 0.7
        direct_ratio = 0.3
        shadow_factor = indirect_ratio + direct_ratio * shadow_map[:, :, np.newaxis]
        final_image = final_image * shadow_factor

    return final_image
