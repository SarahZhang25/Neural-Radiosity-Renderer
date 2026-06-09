import trimesh
import numpy as np
import os

# Create output directory for meshes
output_dir = "/home/sazhang/Neural-Radiosity-Renderer/data_generation/output/raw_meshes/simple_objects"
os.makedirs(output_dir, exist_ok=True)

def save_mesh_flat_shaded(mesh, filename):
    """Save mesh with face normals (for flat surfaces like cubes/cylinders)."""
    # Fix topology
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    
    # Check and fix normal direction
    face_centers = mesh.triangles_center
    centroid = mesh.centroid
    outward_dirs = face_centers - centroid
    outward_dirs = outward_dirs / np.linalg.norm(outward_dirs, axis=1, keepdims=True)
    dots = np.sum(outward_dirs * mesh.face_normals, axis=1)
    
    if np.sum(dots < 0) > np.sum(dots > 0):
        print(f"  Inverting normals for {os.path.basename(filename)}")
        mesh.invert()
    
    # DO NOT include vertex normals - let face normals be computed from geometry
    mesh.export(filename, include_normals=False)
    print(f"  Saved: {filename} (face normals only)")
    print(f"    Watertight: {mesh.is_watertight}, Winding: {mesh.is_winding_consistent}")

def save_mesh_smooth_shaded(mesh, filename):
    """Save mesh with vertex normals (for smooth surfaces like spheres)."""
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    
    # Check and fix normal direction
    face_centers = mesh.triangles_center
    centroid = mesh.centroid
    outward_dirs = face_centers - centroid
    outward_dirs = outward_dirs / np.linalg.norm(outward_dirs, axis=1, keepdims=True)
    dots = np.sum(outward_dirs * mesh.face_normals, axis=1)
    
    if np.sum(dots < 0) > np.sum(dots > 0):
        print(f"  Inverting normals for {os.path.basename(filename)}")
        mesh.invert()
    
    # Include vertex normals for smooth shading
    mesh.export(filename, include_normals=True)
    print(f"  Saved: {filename} (smooth vertex normals)")
    print(f"    Watertight: {mesh.is_watertight}, Winding: {mesh.is_winding_consistent}")

# ============= CREATE CUBE (FLAT SHADING) =============
print("Creating cube...")
cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
cube.apply_translation(-cube.centroid)
save_mesh_flat_shaded(cube, os.path.join(output_dir, "cube.obj"))

# ============= CREATE SPHERE (SMOOTH SHADING) =============
print("\nCreating sphere...")
sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
sphere.apply_translation(-sphere.centroid)
save_mesh_smooth_shaded(sphere, os.path.join(output_dir, "sphere.obj"))

# ============= CREATE TORUS (SMOOTH SHADING) =============
print("\nCreating torus...")
torus = trimesh.creation.torus(major_radius=0.5, minor_radius=0.2)
torus.apply_translation(-torus.centroid)
save_mesh_smooth_shaded(torus, os.path.join(output_dir, "torus.obj"))

# ============= CREATE CYLINDER (FLAT SHADING) =============
print("\nCreating cylinder...")
cylinder = trimesh.creation.cylinder(radius=0.4, height=1.0, sections=32)
cylinder.apply_translation(-cylinder.centroid)
save_mesh_flat_shaded(cylinder, os.path.join(output_dir, "cylinder.obj"))

print(f"\n✓ All meshes saved to {output_dir}")