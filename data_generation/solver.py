"""
Fast radiosity solver using vectorized operations.
Uses Embree for accelerated visibility computation.
"""

import numpy as np
from typing import List, Optional, Callable
import trimesh

# Try to use embree for fast ray tracing
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    EMBREE_AVAILABLE = True
except ImportError:
    EMBREE_AVAILABLE = False

from patches import Patch


def compute_form_factors_fast(patches: List[Patch], chunk_size: int = 1000) -> np.ndarray:
    """
    Compute form factor matrix using vectorized operations with chunking for memory efficiency.
    Assumes no occlusion (fast approximation).

    Args:
        patches: List of patches
        chunk_size: Process this many rows at a time to limit memory usage

    Returns:
        (n, n) form factor matrix (float32 for memory efficiency)
    """
    n = len(patches)

    # Extract patch data into arrays
    centers = np.array([p.center for p in patches], dtype=np.float32)  # (n, 3)
    normals = np.array([p.normal for p in patches], dtype=np.float32)  # (n, 3)
    areas = np.array([p.area for p in patches], dtype=np.float32)      # (n,)

    # Pre-allocate result matrix
    form_factors = np.zeros((n, n), dtype=np.float32)

    # Process in chunks to limit memory usage
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_centers = centers[start:end]  # (chunk, 3)
        chunk_normals = normals[start:end]  # (chunk, 3)

        # Compute differences from chunk patches to ALL patches: (chunk, n, 3)
        diff = centers[np.newaxis, :, :] - chunk_centers[:, np.newaxis, :]

        # Distances: (chunk, n)
        distances = np.linalg.norm(diff, axis=2)
        distances = np.maximum(distances, 1e-6)

        # Normalized directions: (chunk, n, 3)
        directions = diff / distances[:, :, np.newaxis]

        # Cosine at source patches (chunk): dot(chunk_normal, direction)
        cos_theta_i = np.einsum('ik,ijk->ij', chunk_normals, directions)

        # Cosine at target patches (all n): dot(target_normal, -direction)
        cos_theta_j = np.einsum('jk,ijk->ij', normals, -directions)

        # Form factor
        chunk_ff = (cos_theta_i * cos_theta_j * areas[np.newaxis, :]) / (np.pi * distances**2)
        chunk_ff = np.maximum(chunk_ff, 0)
        chunk_ff = np.minimum(chunk_ff, 1.0)

        form_factors[start:end, :] = chunk_ff

        # Free memory
        del diff, directions, cos_theta_i, cos_theta_j, chunk_ff

    # Zero diagonal
    np.fill_diagonal(form_factors, 0)

    # Normalize rows for energy conservation
    row_sums = form_factors.sum(axis=1, keepdims=True)
    # row_sums = np.maximum(row_sums, 1.0)
    form_factors = form_factors / row_sums

    return form_factors


def solve_progressive(
    patches: List[Patch],
    scene_mesh: trimesh.Trimesh = None,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    progress_callback: Optional[Callable[[int, float], None]] = None
) -> int:
    """
    Radiosity solver with visibility-aware form factors for proper shadows.
    Uses Embree for fast ray tracing when available.
    """
    return solve_with_visibility_embree(patches, scene_mesh, max_iterations, tolerance, progress_callback)


def compute_direct_light_shadows_fast(
    patches: List[Patch],
    ray_intersector,
    centers: np.ndarray,
    normals: np.ndarray,
    n_samples: int = 16
) -> np.ndarray:
    """
    Fully vectorized shadow computation - NO Python loops.
    Returns shadow factor per patch (0 = shadowed, 1 = lit).
    """
    n = len(patches)

    # Find light patches (vectorized)
    emissions = np.array([p.emission for p in patches], dtype=np.float32)
    light_mask = emissions.sum(axis=1) > 0
    light_indices = np.where(light_mask)[0]
    n_lights = len(light_indices)

    if n_lights == 0:
        return np.ones(n, dtype=np.float32)

    # Generate stratified grid ONCE
    m = int(np.sqrt(n_samples))
    n_samples = m * m
    grid = (np.arange(m, dtype=np.float32) + 0.5) / m
    uu, vv = np.meshgrid(grid, grid, indexing='xy')
    u, v = uu.ravel(), vv.ravel()  # (n_samples,)

    # Bilinear weights: (n_samples,)
    w00, w10, w11, w01 = (1-u)*(1-v), u*(1-v), u*v, (1-u)*v

    # Get light vertices: (n_lights, 4, 3) - assume quads
    light_verts = np.array([patches[li].vertices[:4] for li in light_indices], dtype=np.float32)

    # Generate ALL light samples via broadcasting: (n_lights, n_samples, 3)
    # w[i] is (n_samples,), light_verts[:, j, :] is (n_lights, 3)
    light_samples = (w00[None, :, None] * light_verts[:, None, 0, :] +
                     w10[None, :, None] * light_verts[:, None, 1, :] +
                     w11[None, :, None] * light_verts[:, None, 2, :] +
                     w01[None, :, None] * light_verts[:, None, 3, :])

    light_centers = centers[light_indices]  # (n_lights, 3)

    # Non-light patches
    non_light_mask = ~light_mask
    non_light_idx = np.where(non_light_mask)[0]
    n_non_light = len(non_light_idx)

    # Direction to each light: (n_non_light, n_lights, 3)
    to_lights = light_centers[None, :, :] - centers[non_light_idx, None, :]

    # Facing check via dot product: (n_non_light, n_lights)
    facing = np.einsum('ij,ikj->ik', normals[non_light_idx], to_lights) > 0

    # Get all facing (patch, light) pairs
    local_patch_idx, local_light_idx = np.where(facing)
    n_pairs = len(local_patch_idx)

    if n_pairs == 0:
        return np.zeros(n, dtype=np.float32)

    global_patch_idx = non_light_idx[local_patch_idx]  # Map to global indices

    # Build ALL rays: n_pairs * n_samples rays total
    n_total_rays = n_pairs * n_samples

    # Origins: repeat each patch center n_samples times
    ray_origins = np.repeat(centers[global_patch_idx], n_samples, axis=0)
    ray_normals_rep = np.repeat(normals[global_patch_idx], n_samples, axis=0)
    ray_origins = ray_origins + ray_normals_rep * 0.01

    # Targets: for each pair, get the n_samples points from corresponding light
    # light_samples[local_light_idx] -> (n_pairs, n_samples, 3) -> reshape to (n_total_rays, 3)
    ray_targets = light_samples[local_light_idx].reshape(-1, 3)

    # Directions and distances
    ray_dirs = ray_targets - ray_origins
    ray_dists = np.linalg.norm(ray_dirs, axis=1)
    ray_dirs = ray_dirs / (ray_dists[:, None] + 1e-8)

    print(f"  Casting {n_total_rays} shadow rays (vectorized)...")

    # SINGLE batched ray cast
    locations, index_ray, _ = ray_intersector.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        multiple_hits=False
    )

    # Visibility computation (vectorized)
    visibility = np.ones(n_total_rays, dtype=np.float32)
    if len(locations) > 0:
        hit_dists = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
        blocked = hit_dists < (ray_dists[index_ray] - 0.02)
        visibility[index_ray[blocked]] = 0.0

    # Mean over samples per pair: (n_pairs,)
    vis_per_pair = visibility.reshape(n_pairs, n_samples).mean(axis=1)

    # Accumulate per patch (max over all lights hitting that patch)
    shadow_factor = np.zeros(n, dtype=np.float32)
    np.maximum.at(shadow_factor, global_patch_idx, vis_per_pair)

    return shadow_factor


def solve_with_visibility_embree(
    patches: List[Patch],
    scene_mesh: trimesh.Trimesh,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    progress_callback: Optional[Callable] = None
) -> int:
    """
    Solve radiosity with visibility-aware form factors for proper shadows.
    Uses Embree for fast ray tracing to compute occlusion.
    """
    n = len(patches)
    print(f"Computing radiosity with shadows for {n} patches...")

    # Extract patch data
    centers = np.array([p.center for p in patches], dtype=np.float32)
    normals = np.array([p.normal for p in patches], dtype=np.float32)
    emission = np.array([p.emission for p in patches], dtype=np.float32)
    reflectance = np.array([p.reflectance for p in patches], dtype=np.float32)

    # Compute geometric form factors
    print("  Computing geometric form factors...")
    F = compute_form_factors_fast(patches)

    # Create ray intersector
    ray_intersector = RayMeshIntersector(scene_mesh)

    # Compute visibility from each patch to light sources
    light_indices = [i for i, p in enumerate(patches) if np.sum(p.emission) > 0]

    if light_indices:
        print(f"  Computing shadows from {len(light_indices)} light(s)...")

        # Compute shadow visibility for direct light
        # n_samples=16 for soft shadows with stratified sampling
        shadow_factor = compute_direct_light_shadows_fast(
            patches, ray_intersector, centers, normals, n_samples=16
        )

        # Apply shadow factor to form factors TO the light
        # F[:, light_idx] is form factors from all patches to light
        for light_idx in light_indices:
            F[:, light_idx] *= shadow_factor
            # F[light_idx, :] *= shadow_factor

    # Iterative radiosity solution
    print("  Solving radiosity equations...")
    radiosity = emission.copy()

    for iteration in range(max_iterations):
        incoming = F @ radiosity
        # incoming = F.T @ radiosity  # (n, 3)
        
        new_radiosity = emission + reflectance * incoming
        max_change = np.max(np.abs(new_radiosity - radiosity))
        radiosity = new_radiosity

        if progress_callback:
            progress_callback(iteration, max_change)

        if max_change < tolerance:
            break

    # Copy results back to patches
    for i, patch in enumerate(patches):
        patch.radiosity = radiosity[i]

    print(f"  Converged in {iteration + 1} iterations")
    return iteration + 1
