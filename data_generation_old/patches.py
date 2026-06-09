"""
Patch representation for radiosity computation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Patch:
    """
    Represents a surface patch in the radiosity system.

    Attributes:
        vertices: Array of shape (N, 3) defining the patch polygon vertices
        center: 3D centroid of the patch
        normal: Unit normal vector of the patch surface
        area: Surface area of the patch
        emission: RGB emission values (non-zero for light sources)
        reflectance: RGB diffuse reflectance (0-1 range per channel)
        radiosity: Computed RGB radiosity values
        unshot: Unshot energy for progressive radiosity
        patch_id: Unique identifier for the patch
    """
    vertices: np.ndarray
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    area: float = 0.0
    emission: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reflectance: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    radiosity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    unshot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    patch_id: int = -1

    def __post_init__(self):
        """Compute derived properties after initialization."""
        if self.vertices is not None and len(self.vertices) >= 3:
            self._compute_properties()

    def _compute_properties(self):
        """Compute center, normal, and area from vertices."""
        # Center is the centroid
        self.center = np.mean(self.vertices, axis=0)

        # For triangles and quads, compute normal using cross product
        if len(self.vertices) >= 3:
            v0, v1, v2 = self.vertices[:3]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                self.normal = normal / norm
            else:
                self.normal = np.array([0, 1, 0])

        # Compute area
        self.area = self._compute_area()

    def _compute_area(self) -> float:
        """Compute the area of the polygon."""
        n = len(self.vertices)
        if n < 3:
            return 0.0

        if n == 3:
            # Triangle area
            v0, v1, v2 = self.vertices
            return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        elif n == 4:
            # Quad area (split into two triangles)
            v0, v1, v2, v3 = self.vertices
            area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            area2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))
            return area1 + area2
        else:
            # General polygon - fan triangulation
            total_area = 0.0
            v0 = self.vertices[0]
            for i in range(1, n - 1):
                v1 = self.vertices[i]
                v2 = self.vertices[i + 1]
                total_area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            return total_area

    def initialize_radiosity(self):
        """Initialize radiosity and unshot energy from emission."""
        self.radiosity = self.emission.copy()
        self.unshot = self.emission.copy()

    def unshot_power(self) -> float:
        """Compute the total unshot power (energy * area)."""
        return np.sum(self.unshot) * self.area


def create_quad_patch(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    reflectance: np.ndarray = None,
    emission: np.ndarray = None,
    patch_id: int = -1
) -> Patch:
    """
    Create a quad patch from 4 vertices.

    Args:
        v0, v1, v2, v3: Corner vertices in counter-clockwise order
        reflectance: RGB diffuse reflectance
        emission: RGB emission values
        patch_id: Unique patch identifier

    Returns:
        Patch object
    """
    vertices = np.array([v0, v1, v2, v3])

    patch = Patch(
        vertices=vertices,
        reflectance=reflectance if reflectance is not None else np.array([0.5, 0.5, 0.5]),
        emission=emission if emission is not None else np.zeros(3),
        patch_id=patch_id
    )

    return patch


def create_triangle_patch(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    reflectance: np.ndarray = None,
    emission: np.ndarray = None,
    patch_id: int = -1
) -> Patch:
    """
    Create a triangle patch from 3 vertices.

    Args:
        v0, v1, v2: Corner vertices in counter-clockwise order
        reflectance: RGB diffuse reflectance
        emission: RGB emission values
        patch_id: Unique patch identifier

    Returns:
        Patch object
    """
    vertices = np.array([v0, v1, v2])

    patch = Patch(
        vertices=vertices,
        reflectance=reflectance if reflectance is not None else np.array([0.5, 0.5, 0.5]),
        emission=emission if emission is not None else np.zeros(3),
        patch_id=patch_id
    )

    return patch


def create_subdivided_quad(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    subdivisions: int = 10,
    reflectance: np.ndarray = None,
    start_id: int = 0
) -> list:
    """
    Create a subdivided quad as a grid of triangle patches.

    Args:
        v0, v1, v2, v3: Corner vertices (v0=bottom-left, v1=bottom-right, v2=top-right, v3=top-left)
        subdivisions: Number of divisions along each edge
        reflectance: RGB diffuse reflectance
        start_id: Starting patch ID

    Returns:
        List of triangle patches forming the subdivided quad
    """
    patches = []
    default_reflectance = reflectance if reflectance is not None else np.array([0.5, 0.5, 0.5])

    # Create grid of vertices
    n = subdivisions + 1
    vertices = np.zeros((n, n, 3))

    for i in range(n):
        for j in range(n):
            u = i / subdivisions
            v = j / subdivisions
            # Bilinear interpolation
            p0 = (1 - u) * v0 + u * v1
            p1 = (1 - u) * v3 + u * v2
            vertices[i, j] = (1 - v) * p0 + v * p1

    # Create triangles
    patch_id = start_id
    for i in range(subdivisions):
        for j in range(subdivisions):
            # Two triangles per grid cell
            p00 = vertices[i, j]
            p10 = vertices[i + 1, j]
            p11 = vertices[i + 1, j + 1]
            p01 = vertices[i, j + 1]

            # Triangle 1
            patch1 = create_triangle_patch(
                p00, p10, p11,
                reflectance=default_reflectance.copy(),
                patch_id=patch_id
            )
            patches.append(patch1)
            patch_id += 1

            # Triangle 2
            patch2 = create_triangle_patch(
                p00, p11, p01,
                reflectance=default_reflectance.copy(),
                patch_id=patch_id
            )
            patches.append(patch2)
            patch_id += 1

    return patches


def patches_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    reflectance: np.ndarray = None,
    start_id: int = 0
) -> list:
    """
    Create patches from a triangle mesh.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
        reflectance: RGB diffuse reflectance for all patches
        start_id: Starting patch ID

    Returns:
        List of Patch objects
    """
    patches = []
    default_reflectance = reflectance if reflectance is not None else np.array([0.5, 0.5, 0.5])

    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        patch = create_triangle_patch(
            v0, v1, v2,
            reflectance=default_reflectance.copy(),
            patch_id=start_id + i
        )
        patches.append(patch)

    return patches
