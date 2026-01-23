import os
import glob
import json
import numpy as np
import trimesh
import mitsuba as mi
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------

# Set Mitsuba variant - consistent with test_scenes.ipynb
try:
    mi.set_variant('cuda_ad_rgb')
except Exception:
    try:
        mi.set_variant('llvm_ad_rgb')
    except Exception:
        mi.set_variant('scalar_rgb')

print(f"Using Mitsuba variant: {mi.variant()}")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESHES_DIR = os.path.join(SCRIPT_DIR, "output/raw_meshes/simple_objects")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output/renders")

# Scene Constants
BOX_SIZE = 2.0
HALF_SIZE = BOX_SIZE / 2.0
LIGHT_INTENSITY = 50.0
LIGHT_POS = [0.0, 1.8, 0.0]  # Near ceiling

# Cornell Box Colors from scene.py
CORNELL_WHITE = "0.76, 0.75, 0.50"
CORNELL_RED = "0.63, 0.06, 0.04"
CORNELL_GREEN = "0.14, 0.45, 0.09"

# Dataset Generation Params
NUM_ROTATIONS = 2 #5  # 5 rotations (0, 72, 144, 216, 288)
IMAGE_RES = 512    # Resolution
SPP = 256          # Samples per pixel for rendering

# Object Colors (Diffuse Reflectance)
OBJECT_COLORS = [
    # ([0.7, 0.1, 0.1], "red"),
    # ([0.1, 0.7, 0.1], "green"),
    ([0.1, 0.1, 0.7], "blue"),
    # ([0.8, 0.8, 0.8], "white")
]

# ------------------------------------------------------------------------------------
# XML Template
# ------------------------------------------------------------------------------------

SCENE_TEMPLATE = """
<scene version='3.0.0'>
    <default name="spp" value="{spp}"/>
    <default name="res" value="{res}"/>

    <integrator type='path'>
        <integer name="max_depth" value="5"/>
    </integrator>

    <sensor type="perspective" id="sensor">
        <float name="fov" value="39.3077"/> <!-- Matches approx 50mm -->
        <transform name="to_world">
            <lookat target="0, 1.0, 0" 
                    origin="0, 1.0, 3.4" 
                    up="0, 1, 0"/>
        </transform>
        <sampler type="independent">
            <integer name="sample_count" value="$spp"/>
        </sampler>
        <film type="hdrfilm">
            <rfilter type="gaussian"/>
            <integer name="width"  value="$res"/>
            <integer name="height" value="$res"/>
        </film>
    </sensor>

    <!-- Room Geometry (Cornell Box) -->
    
    <!-- Floor (White) -->
    <shape type="rectangle">
        <transform name="to_world">
            <rotate x="1" angle="-90"/>
            <scale value="{half_size}"/>
            <translate y="0"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{white}"/>
        </bsdf>
    </shape>

    <!-- Ceiling (White) -->
    <shape type="rectangle">
        <transform name="to_world">
            <rotate x="1" angle="90"/>
            <scale value="{half_size}"/>
            <translate y="{size}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{white}"/>
        </bsdf>
    </shape>

    <!-- Back Wall (White) -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale value="{half_size}"/>
            <translate y="{half_size}" z="-{half_size}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{white}"/>
        </bsdf>
    </shape>

    <!-- Left Wall (Red) -->
    <shape type="rectangle">
        <transform name="to_world">
            <rotate y="1" angle="90"/>
            <scale value="{half_size}"/>
            <translate x="-{half_size}" y="{half_size}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{red}"/>
        </bsdf>
    </shape>

    <!-- Right Wall (Green) -->
    <shape type="rectangle">
        <transform name="to_world">
            <rotate y="1" angle="-90"/>
            <scale value="{half_size}"/>
            <translate x="{half_size}" y="{half_size}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{green}"/>
        </bsdf>
    </shape>

    <!-- Point Light Source -->
    <emitter type="point">
        <point name="position" x="{light_x}" y="{light_y}" z="{light_z}"/>
        <rgb name="intensity" value="{light_intensity}"/>
    </emitter>

    <!-- Main Object -->
    <shape type="obj">
        <string name="filename" value="{mesh_path}"/>
        <transform name="to_world">
            <!-- First scale, then rotate around Y, then translate to position -->
            <scale value="{obj_scale}"/>
            <rotate y="1" angle="{obj_rotation}"/>
            <translate x="{obj_x}" y="{obj_y}" z="{obj_z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{obj_color}"/>
        </bsdf>
    </shape>

</scene>
"""

# ------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------

def get_camera_to_world_matrix(origin, target, up):
    """Compute camera-to-world matrix from lookat parameters."""
    f = (np.array(target) - np.array(origin))
    f = f / np.linalg.norm(f)
    
    u = np.array(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    
    u_new = np.cross(s, f)
    
    # Rotation matrix (columns are R, U, -F) - Mitsuba/OpenGL convention
    # Camera looks down -Z in its local frame
    R = np.eye(4)
    R[0, :3] = s
    R[1, :3] = u_new
    R[2, :3] = -f  # Forward is -Z
    R[:3, 3] = origin
    
    return R

def sample_mesh_surface(mesh, count=10000):
    """Sample points from the surface of the mesh."""
    points, face_indices = trimesh.sample.sample_surface(mesh, count)
    
    # Interpolate normals
    triangles = mesh.faces[face_indices]
    bary = trimesh.triangles.points_to_barycentric(
        mesh.vertices[triangles], points
    )
    
    vertex_normals = mesh.vertex_normals[triangles]
    # Weighted sum of normals
    normals = trimesh.unitize(
        (vertex_normals * bary.reshape((-1, 3, 1))).sum(axis=1)
    )
    
    return points, normals

def main():
    if not os.path.exists(MESHES_DIR):
        print(f"Error: Meshes directory not found at {MESHES_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find .obj files
    mesh_files = glob.glob(os.path.join(MESHES_DIR, "*.obj"))
    mesh_files.sort()
    
    print(f"Found {len(mesh_files)} meshes: {[os.path.basename(m) for m in mesh_files]}")

    case_idx = 0

    # Camera settings (fixed for this dataset)
    cam_origin = np.array([0.0, 1.0, 3.4])
    cam_target = np.array([0.0, 1.0, 0.0])
    cam_up = np.array([0.0, 1.0, 0.0])
    c2w_matrix = get_camera_to_world_matrix(cam_origin, cam_target, cam_up)

    for mesh_path in mesh_files:
        shape_name = os.path.splitext(os.path.basename(mesh_path))[0]
        
        # Load mesh once to check bounds and center it
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
        except Exception as e:
            print(f"Failed to load {mesh_path}: {e}")
            continue

        # Normalize mesh size to fit nicel in the box (e.g., max dim = 0.8)
        # Meshes in 'simple_objects' are assumed to be somewhat normalized, but let's be sure.
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        max_extent = np.max(extents)
        target_scale_size = 0.8
        scale_factor = target_scale_size / max_extent if max_extent > 0 else 1.0
        
        # We want to place the object on the floor (y=0)
        # The object local origin might be center or bottom. 
        # We will assume we want to translate the bottom of the scaled mesh to y=0.
        # But in the XML, we apply Scale -> Rotate -> Translate.
        # It's easier to pre-transform the mesh in memory for sampling, 
        # and calculate the correct translation for the XML.
        
        # For simplicity in XML:
        # 1. Scale
        # 2. Rotate
        # 3. Translate so min_y ends up at 0
        
        # The center of the scaled mesh (without rotation) relative to its local origin:
        center_offset = (bounds[0] + bounds[1]) / 2.0 * scale_factor
        
        # Calculate vertical shift to put on floor
        # The lowest point after scaling is bounds[0][1] * scale_factor
        # We want this to be at y = 0
        # So Translation Y = - (bounds[0][1] * scale_factor)
        trans_y = - (bounds[0][1] * scale_factor)
        
        # Center X and Z
        trans_x = - (center_offset[0]) 
        trans_z = - (center_offset[2])

        # Color Loop
        for c_idx, (color_rgb, color_name) in enumerate(OBJECT_COLORS):
            
            # Rotation Loop
            for r_idx in range(NUM_ROTATIONS):
                case_idx += 1
                rot_angle = r_idx * (360.0 / NUM_ROTATIONS)
                
                print(f"Processing Case {case_idx}: {shape_name} | {color_name} | {rot_angle} deg")
                
                # 1. Construct XML
                color_str = f"{color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}"
                
                # Note: The XML transform order is applied from bottom up usually in Mitsuba docs (pre-multiplication)
                # But intuitively: translate * rotate * scale * v
                # Mitsuba XML <transform>: operations are applied in order written? 
                # Documentation says: "The transformations are applied in the order in which they are specified"
                # So if we write Scale then Rotate then Translate: T * R * S * v. This is what we want.
                
                scene_xml = SCENE_TEMPLATE.format(
                    spp=SPP,
                    res=IMAGE_RES,
                    half_size=HALF_SIZE,
                    size=BOX_SIZE,
                    white=CORNELL_WHITE,
                    red=CORNELL_RED,
                    green=CORNELL_GREEN,
                    light_x=LIGHT_POS[0],
                    light_y=LIGHT_POS[1],
                    light_z=LIGHT_POS[2],
                    light_intensity=LIGHT_INTENSITY,
                    mesh_path=mesh_path,
                    obj_scale=scale_factor,
                    obj_rotation=rot_angle,
                    obj_x=trans_x, # This centers it on X/Z but we need to account for rotation? 
                                   # If we rotate around Y, the bounding box changes.
                                   # Better approach: The mesh is usually centered at 0,0,0 in local file.
                                   # We assume simple_objects are centered.
                                   # If not, the rotation will orbit the origin.
                                   # Let's assume we want to rotate around the object's centroid.
                    obj_y=trans_y,
                    obj_z=trans_z,
                    obj_color=color_str
                )
                
                # 2. Render
                scene = mi.load_string(scene_xml)
                image = mi.render(scene)
                
                # Save Image
                filename_base = f"case_{case_idx:03d}_{shape_name}_c{c_idx}_r{r_idx}"
                output_path_png = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
                mi.util.write_bitmap(output_path_png, image)
                
                # 3. Generate Point Cloud Data
                # We need to reproduce the transformation applied in XML to the Trimesh object
                # T_world = T_trans * T_rot * T_scale
                
                # Scale
                T_scale = np.eye(4)
                T_scale[0,0] = T_scale[1,1] = T_scale[2,2] = scale_factor
                
                # Rotate (around Y)
                r = Rotation.from_euler('y', rot_angle, degrees=True)
                T_rot = np.eye(4)
                T_rot[:3, :3] = r.as_matrix()
                
                # Translate
                T_trans = np.eye(4)
                T_trans[:3, 3] = [trans_x, trans_y, trans_z] # This translation was calculated for non-rotated mesh?
                # Actually, if the mesh is not centered at 0,0 in XZ, rotating it around 0,0 (World Y) 
                # will move the object. 
                # The XML applies Scale, then Rotate (around local 0), then Translate.
                # So the object rotates around its local origin.
                # If the local origin is not the centroid, it wobbles.
                # For `simple_objects`, usually origin is centroid. Let's assume that.
                
                # Combined Matrix
                model_matrix = T_trans @ T_rot @ T_scale
                
                # Apply to a copy of mesh
                mesh_transformed = mesh.copy()
                mesh_transformed.apply_transform(model_matrix)
                
                # Sample
                points, normals = sample_mesh_surface(mesh_transformed, count=20000)
                
                # Colors
                points_colors = np.tile(color_rgb, (points.shape[0], 1))
                
                # 4. Save Data (.npz)
                output_path_npz = os.path.join(OUTPUT_DIR, f"{filename_base}_data.npz")
                
                # Convert Render to numpy (H, W, 3) - Linear RGB
                image_np = np.array(image)[:, :, :3] # Drop alpha if exists
                
                np.savez(
                    output_path_npz,
                    points=points.astype(np.float32),
                    colors=points_colors.astype(np.float32),
                    normals=normals.astype(np.float32),
                    camera_pos=cam_origin.astype(np.float32),
                    view_matrix=c2w_matrix.astype(np.float32), # Save C2W
                    image=image_np.astype(np.float32),
                    shape_id=shape_name,
                    rotation=rot_angle
                )
    
    print("Dataset generation completed.")

if __name__ == "__main__":
    main()