"""
Scene setup including Cornell box creation and mesh loading.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
from patches import Patch, create_quad_patch, create_subdivided_quad, patches_from_mesh


# Cornell box colors (warm tone to match reference)
CORNELL_RED = np.array([0.63, 0.06, 0.04])      # Left wall - deep red
CORNELL_GREEN = np.array([0.14, 0.45, 0.09])    # Right wall - forest green
CORNELL_WHITE = np.array([0.76, 0.75, 0.50])    # Warmer cream/tan floor


class Scene:
    """
    Container for all scene geometry and patches.
    """

    def __init__(self):
        self.patches: List[Patch] = []
        self.vertices: np.ndarray = None  # Combined vertex array
        self.faces: np.ndarray = None     # Combined face array
        self.face_to_patch: dict = {}     # Map face index to patch index
        self._next_patch_id = 0

    def add_patch(self, patch: Patch) -> int:
        """Add a single patch to the scene."""
        if patch.patch_id < 0:
            patch.patch_id = self._next_patch_id
        self._next_patch_id = max(self._next_patch_id, patch.patch_id + 1)
        self.patches.append(patch)
        return patch.patch_id

    def add_patches(self, patches: List[Patch]):
        """Add multiple patches to the scene."""
        for patch in patches:
            self.add_patch(patch)

    def build_combined_mesh(self):
        """Build combined vertex and face arrays for ray tracing."""
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for patch_idx, patch in enumerate(self.patches):
            n_verts = len(patch.vertices)
            all_vertices.append(patch.vertices)

            # Create faces for this patch
            if n_verts == 3:
                # Triangle
                faces = [[vertex_offset, vertex_offset + 1, vertex_offset + 2]]
            elif n_verts == 4:
                # Quad - split into two triangles
                faces = [
                    [vertex_offset, vertex_offset + 1, vertex_offset + 2],
                    [vertex_offset, vertex_offset + 2, vertex_offset + 3]
                ]
            else:
                # Fan triangulation for polygons
                faces = []
                for i in range(1, n_verts - 1):
                    faces.append([vertex_offset, vertex_offset + i, vertex_offset + i + 1])

            for face in faces:
                face_idx = len(all_faces)
                all_faces.append(face)
                self.face_to_patch[face_idx] = patch_idx

            vertex_offset += n_verts

        self.vertices = np.vstack(all_vertices) if all_vertices else np.zeros((0, 3))
        self.faces = np.array(all_faces, dtype=np.int32) if all_faces else np.zeros((0, 3), dtype=np.int32)

        # Merge duplicate vertices for proper interpolation
        if len(self.vertices) > 0:
            self._merge_duplicate_vertices()

    def _merge_duplicate_vertices(self, tolerance: float = 1e-6):
        """Merge vertices that are at the same position AND same material."""
        from scipy.spatial import cKDTree

        # Build mapping from vertex index to patch material
        vertex_to_material = {}
        vertex_offset = 0
        for patch in self.patches:
            n_verts = len(patch.vertices)
            material_key = tuple(patch.reflectance.round(3))  # Round for comparison
            for i in range(n_verts):
                vertex_to_material[vertex_offset + i] = material_key
            vertex_offset += n_verts

        # Build KD-tree for fast nearest neighbor lookup
        tree = cKDTree(self.vertices)

        # Find groups of duplicate vertices
        pairs = tree.query_pairs(r=tolerance)

        # Build union-find structure to group duplicates
        n_verts = len(self.vertices)
        parent = list(range(n_verts))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Only merge if same material
        for i, j in pairs:
            mat_i = vertex_to_material.get(i)
            mat_j = vertex_to_material.get(j)
            if mat_i == mat_j:  # Same material - safe to merge
                union(i, j)

        # Create mapping from old to new vertex indices
        unique_roots = {}
        old_to_new = np.zeros(n_verts, dtype=np.int32)
        new_vertices = []

        for i in range(n_verts):
            root = find(i)
            if root not in unique_roots:
                unique_roots[root] = len(new_vertices)
                new_vertices.append(self.vertices[root])
            old_to_new[i] = unique_roots[root]

        # Update faces with new vertex indices
        self.vertices = np.array(new_vertices)
        self.faces = old_to_new[self.faces]

    def get_trimesh(self) -> trimesh.Trimesh:
        """Get the scene as a trimesh object for ray tracing."""
        if self.vertices is None or self.faces is None:
            self.build_combined_mesh()
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def initialize_radiosity(self):
        """Initialize radiosity values for all patches."""
        for patch in self.patches:
            patch.initialize_radiosity()


def create_cornell_box(
    size: float = 2.0,
    light_size_ratio: float = 0.3,
    light_intensity: float = 15.0,
    subdivisions: int = 8
) -> Scene:
    """
    Create a standard Cornell box scene with subdivided surfaces.

    Args:
        size: Size of the box (cube with this side length)
        light_size_ratio: Size of light relative to ceiling (0-1)
        light_intensity: Emission intensity of the ceiling light
        subdivisions: Number of subdivisions for floor/walls (for shadow detail)

    Returns:
        Scene object with Cornell box patches
    """
    scene = Scene()
    half = size / 2.0
    eps = 0.001  # Tiny overlap for z-fighting, walls render in front of floor/ceiling edges

    # Floor (y = 0)
    floor_patches = create_subdivided_quad(
        v0=np.array([-half, 0, half]),
        v1=np.array([half, 0, half]),
        v2=np.array([half, 0, -half]),
        v3=np.array([-half, 0, -half]),
        subdivisions=subdivisions,
        reflectance=CORNELL_WHITE.copy(),
        start_id=scene._next_patch_id
    )
    scene.add_patches(floor_patches)

    # Ceiling (y = size)
    ceiling_patches = create_subdivided_quad(
        v0=np.array([-half, size, -half]),
        v1=np.array([half, size, -half]),
        v2=np.array([half, size, half]),
        v3=np.array([-half, size, half]),
        subdivisions=subdivisions,
        reflectance=CORNELL_WHITE.copy(),
        start_id=scene._next_patch_id
    )
    scene.add_patches(ceiling_patches)

    # Back wall (z = -half) - extends slightly past floor/ceiling
    back_wall_patches = create_subdivided_quad(
        v0=np.array([-half, -eps, -half]),
        v1=np.array([half, -eps, -half]),
        v2=np.array([half, size + eps, -half]),
        v3=np.array([-half, size + eps, -half]),
        subdivisions=subdivisions,
        reflectance=CORNELL_WHITE.copy(),
        start_id=scene._next_patch_id
    )
    scene.add_patches(back_wall_patches)

    # Left wall (x = -half) - RED - extends slightly past floor/ceiling
    left_wall_patches = create_subdivided_quad(
        v0=np.array([-half, -eps, half]),
        v1=np.array([-half, -eps, -half]),
        v2=np.array([-half, size + eps, -half]),
        v3=np.array([-half, size + eps, half]),
        subdivisions=subdivisions,
        reflectance=CORNELL_RED.copy(),
        start_id=scene._next_patch_id
    )
    scene.add_patches(left_wall_patches)

    # Right wall (x = half) - GREEN - extends slightly past floor/ceiling
    right_wall_patches = create_subdivided_quad(
        v0=np.array([half, -eps, -half]),
        v1=np.array([half, -eps, half]),
        v2=np.array([half, size + eps, half]),
        v3=np.array([half, size + eps, -half]),
        subdivisions=subdivisions,
        reflectance=CORNELL_GREEN.copy(),
        start_id=scene._next_patch_id
    )
    scene.add_patches(right_wall_patches)

    # Ceiling light (small quad slightly below ceiling) - normal should point DOWN [0, -1, 0]
    light_half = half * light_size_ratio
    light_y = size - 0.01  # Slightly below ceiling to avoid z-fighting

    # Warm white light (slightly yellow)
    ceiling_light = create_quad_patch(
        v0=np.array([-light_half, light_y, -light_half]),
        v1=np.array([light_half, light_y, -light_half]),
        v2=np.array([light_half, light_y, light_half]),
        v3=np.array([-light_half, light_y, light_half]),
        reflectance=np.zeros(3),
        emission=np.array([light_intensity, light_intensity * 0.95, light_intensity * 0.85])
    )
    scene.add_patch(ceiling_light)

    return scene


def load_mesh(
    mesh_path: str,
    target_center: np.ndarray = None,
    target_scale: float = None,
    reflectance: np.ndarray = None,
    simplify_ratio: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an OBJ mesh and optionally transform it.

    Args:
        mesh_path: Path to OBJ file
        target_center: Target center position (if None, keep original)
        target_scale: Target scale to fit mesh (if None, keep original)
        reflectance: Diffuse reflectance for the mesh
        simplify_ratio: If set, simplify mesh to this fraction of original faces (0-1)

    Returns:
        Tuple of (vertices, faces, face_normals)
    """
    mesh = trimesh.load(mesh_path, force='mesh')

    # Simplify mesh using quadric decimation
    if simplify_ratio is not None and simplify_ratio < 1.0:
        n_faces = len(mesh.faces)
        target_faces = max(100, int(n_faces * simplify_ratio))
        if target_faces < n_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

    if target_scale is not None:
        # Scale mesh to fit within target size
        bounds = mesh.bounds
        current_size = np.max(bounds[1] - bounds[0])
        scale_factor = target_scale / current_size
        mesh.apply_scale(scale_factor)

    if target_center is not None:
        # Center mesh at target position
        current_center = mesh.centroid
        translation = target_center - current_center
        mesh.apply_translation(translation)

    return mesh.vertices.copy(), mesh.faces.copy(), mesh.face_normals.copy()


def add_mesh_to_scene(
    scene: Scene,
    mesh_path: str,
    center: np.ndarray = None,
    scale: float = None,
    reflectance: np.ndarray = None,
    simplify_ratio: float = None
):
    """
    Load a mesh and add its triangles as patches to the scene.

    Args:
        scene: Scene to add patches to
        mesh_path: Path to OBJ file
        center: Target center position
        scale: Target scale
        reflectance: Diffuse reflectance for mesh patches
        simplify_ratio: If set, simplify mesh to this fraction of faces
    """
    vertices, faces, _ = load_mesh(mesh_path, center, scale, reflectance, simplify_ratio)

    default_reflectance = reflectance if reflectance is not None else np.array([0.7, 0.7, 0.7])

    mesh_patches = patches_from_mesh(
        vertices, faces,
        reflectance=default_reflectance,
        start_id=scene._next_patch_id
    )

    scene.add_patches(mesh_patches)


def create_cornell_box_with_car(
    mesh_path: str,
    box_size: float = 2.0,
    car_scale: float = 1.2,
    car_height_offset: float = 0.0,
    light_intensity: float = 8.0,
    simplify_ratio: float = 0.07,
    car_color: np.ndarray = None,
    subdivisions: int = 16,
    rotate_car: float = 0.0,  # Rotation in degrees around Y axis (0 = no rotation)
    light_size_ratio: float = 0.3,  # Light size relative to ceiling (smaller = harder shadows)
) -> Scene:
    """
    Create a Cornell box with a car mesh inside.

    Args:
        mesh_path: Path to car OBJ file
        box_size: Size of Cornell box
        car_scale: Scale factor for car (relative to box size)
        car_height_offset: Vertical offset for car placement
        light_intensity: Ceiling light intensity
        simplify_ratio: Mesh simplification ratio
        car_color: RGB color for the car (default blue)
        subdivisions: Wall/floor patch subdivisions (higher = smoother shadows)

    Returns:
        Scene with Cornell box and car
    """
    scene = create_cornell_box(
        size=box_size,
        light_intensity=light_intensity,
        subdivisions=subdivisions,
        light_size_ratio=light_size_ratio
    )

    if car_color is None:
        car_color = np.array([0.4, 0.55, 0.75])  # Sky blue car

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')

    # Fix mesh normals and winding order
    mesh.fix_normals()

    # Simplify only if explicitly requested with ratio < 1.0
    if simplify_ratio is not None and simplify_ratio < 1.0:
        n_faces = len(mesh.faces)
        target_faces = max(100, int(n_faces * simplify_ratio))
        if target_faces < n_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            mesh.fix_normals()
    # Note: with simplify_ratio >= 1.0, original mesh is used unchanged

    # Rotate car around Y axis if specified
    if rotate_car != 0.0:
        rotation_y = trimesh.transformations.rotation_matrix(np.radians(rotate_car), [0, 1, 0])
        mesh.apply_transform(rotation_y)

    # Scale to fit
    bounds = mesh.bounds
    current_size = np.max(bounds[1] - bounds[0])
    target_size = box_size * car_scale
    scale_factor = target_size / current_size
    mesh.apply_scale(scale_factor)

    # Position car so it sits on the floor
    # First center horizontally, then place bottom at floor level
    bounds = mesh.bounds
    car_height = bounds[1][1] - bounds[0][1]
    floor_y = car_height / 2 + car_height_offset + 0.01  # Small offset to avoid z-fighting
    car_center = np.array([0.0, floor_y, 0.0])
    current_center = mesh.centroid
    mesh.apply_translation(car_center - current_center)

    # Create patches from mesh
    from patches import patches_from_mesh
    mesh_patches = patches_from_mesh(
        mesh.vertices, mesh.faces,
        reflectance=car_color,
        start_id=scene._next_patch_id
    )
    scene.add_patches(mesh_patches)

    return scene
