"""
Utility functions for point cloud processing.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def fps_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) to select representative points.

    Args:
        points: Point cloud (N, 3)
        n_samples: Number of points to sample

    Returns:
        Indices of sampled points (n_samples,)
    """
    device = points.device
    N = points.shape[0]

    if n_samples >= N:
        return torch.arange(N, device=device)

    # Initialize with random first point
    indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10

    # Select first point randomly
    farthest = torch.randint(0, N, (1,), device=device)
    indices[0] = farthest

    for i in range(1, n_samples):
        # Compute distance from new point to all points
        curr_points = points[farthest]
        dist = torch.sum((points - curr_points) ** 2, dim=-1)

        # Update minimum distances
        distances = torch.minimum(distances, dist)

        # Select point with maximum minimum distance
        farthest = torch.argmax(distances)
        indices[i] = farthest

    return indices


def compute_knn_distance(
    positions: torch.Tensor,
    k: int = 1
) -> torch.Tensor:
    """
    Compute distance to k-nearest neighbors from other objects.

    Args:
        positions: Positions of all objects (B, N_obj, N_vertices, 3)
        k: Number of nearest neighbors

    Returns:
        Distance vectors to nearest points (B, N_obj, N_vertices, 3)
    """
    B, N_obj, N_v, _ = positions.shape
    device = positions.device

    dist_vecs = torch.zeros_like(positions)

    for b in range(B):
        for obj_i in range(N_obj):
            points_i = positions[b, obj_i]  # (N_v, 3)

            # Collect points from all other objects
            other_points = []
            for obj_j in range(N_obj):
                if obj_j != obj_i:
                    other_points.append(positions[b, obj_j])

            if len(other_points) > 0:
                other_points = torch.cat(other_points, dim=0)  # (N_other, 3)

                # Compute pairwise distances
                dists = torch.cdist(points_i, other_points)  # (N_v, N_other)

                # Find nearest neighbor
                min_dists, min_indices = torch.min(dists, dim=1)  # (N_v,)

                # Compute distance vectors
                nearest_points = other_points[min_indices]  # (N_v, 3)
                dist_vecs[b, obj_i] = nearest_points - points_i

    return dist_vecs


def scatter_to_full(
    anchor_values: torch.Tensor,
    anchor_indices: torch.Tensor,
    n_vertices: int
) -> torch.Tensor:
    """
    Scatter anchor values to full point cloud.

    Args:
        anchor_values: Values at anchor points (B, N_obj, N_anchors, D)
        anchor_indices: Anchor indices (B, N_obj, N_anchors)
        n_vertices: Total number of vertices

    Returns:
        Full values with anchor values scattered (B, N_obj, n_vertices, D)
    """
    B, N_obj, N_anchors, D = anchor_values.shape
    device = anchor_values.device

    # Initialize full array
    full_values = torch.zeros(B, N_obj, n_vertices, D, device=device)

    # Scatter anchor values
    for b in range(B):
        for obj in range(N_obj):
            indices = anchor_indices[b, obj]  # (N_anchors,)
            full_values[b, obj, indices] = anchor_values[b, obj]

    return full_values


def kabsch_align(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kabsch algorithm for rigid alignment.

    Args:
        source: Source points (N, 3)
        target: Target points (N, 3)
        weights: Optional per-point weights (N,)

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    """
    # Center points
    if weights is not None:
        weights = weights / weights.sum()
        source_center = (source * weights.unsqueeze(-1)).sum(dim=0)
        target_center = (target * weights.unsqueeze(-1)).sum(dim=0)
    else:
        source_center = source.mean(dim=0)
        target_center = target.mean(dim=0)

    source_centered = source - source_center
    target_centered = target - target_center

    # Compute covariance matrix
    if weights is not None:
        H = torch.mm(
            (source_centered * weights.unsqueeze(-1)).T,
            target_centered
        )
    else:
        H = torch.mm(source_centered.T, target_centered)

    # SVD
    U, _, Vt = torch.linalg.svd(H)
    R = torch.mm(Vt.T, U.T)

    # Handle reflection
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = torch.mm(Vt.T, U.T)

    # Translation
    t = target_center - torch.mm(R, source_center.unsqueeze(-1)).squeeze()

    return R, t