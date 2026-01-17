"""
TransferVertexColors V1 - Transfer Vertex Colors Between Meshes

Transfers vertex colors from a high-poly source mesh to a low-poly target mesh
using nearest-point BVH lookup. Perfect for recovering colors lost during remeshing.

USE CASES:
    - Mesh lost vertex colors during remeshing/decimation
    - Retopologized mesh needs colors from original
    - Baked vertex colors to high-poly, need them on low-poly version

ALGORITHM:
    For each face on target mesh:
    1. Find closest point on source mesh (BVH nearest-point query)
    2. Get color from source face at that point
    3. Apply same color to all loops of target face (flat shading)

USAGE:
    from TransferVertexColors_v1 import transfer_colors, transfer_colors_by_name

    # Method 1: By object reference
    transfer_colors(source_obj, target_obj)

    # Method 2: By object name
    transfer_colors_by_name("HighPoly_Mesh", "LowPoly_Mesh")

    # Method 3: Configure and run
    SOURCE_MESH = "HighPoly_Mesh"
    TARGET_MESH = "LowPoly_Mesh"
    # then run script

SCENE SETUP:
    Source mesh must have vertex colors in one of these layers:
        - "Col" (standard)
        - "BakedColors"
        - "TransferredColors"
        - Or any color attribute (will use first found)

    Target mesh can be any mesh - vertex colors will be created.

OUTPUT:
    - Creates "Col" color attribute on target mesh
    - Sets as active render color for FBX export
    - Creates optional debug material to preview colors
"""

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source mesh (with vertex colors) - "SELECTED" uses second selected object
SOURCE_MESH = "SELECTED"

# Target mesh (receives colors) - "ACTIVE" uses active object
TARGET_MESH = "ACTIVE"

# Color attribute name to create on target
OUTPUT_COLOR_NAME = "Col"

# Source color attribute names to look for (in priority order)
SOURCE_COLOR_NAMES = ["Col", "BakedColors", "TransferredColors"]

# Transfer mode:
#   "FACE" = Each target face gets one color from nearest source face (flat shading)
#   "VERTEX" = Each target vertex gets color from nearest source point (smooth blending)
TRANSFER_MODE = "FACE"

# Create debug material showing vertex colors
CREATE_DEBUG_MATERIAL = True

# Show progress updates
SHOW_PROGRESS = True
PROGRESS_INTERVAL = 5000

# ============================================================================
# LOGGING
# ============================================================================

LOG_TEXT_NAME = "TransferVertexColors_V1_Log.txt"


def log_to_text(s: str):
    """Write log to Blender text block."""
    txt = bpy.data.texts.get(LOG_TEXT_NAME)
    if not txt:
        txt = bpy.data.texts.new(LOG_TEXT_NAME)
    txt.clear()
    txt.write(s)


def log_print(msg, report=None):
    """Print to console and optionally add to report."""
    print(msg)
    if report is not None:
        report.append(msg)


# ============================================================================
# HELPERS
# ============================================================================

def ensure_object_mode():
    """Ensure we're in object mode."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def get_source_color_attribute(mesh):
    """Find vertex color attribute on source mesh."""
    if not hasattr(mesh, 'color_attributes') or not mesh.color_attributes:
        return None

    # Try priority names first
    for name in SOURCE_COLOR_NAMES:
        attr = mesh.color_attributes.get(name)
        if attr:
            return attr

    # Fall back to first available
    if len(mesh.color_attributes) > 0:
        return mesh.color_attributes[0]

    return None


def create_target_color_attribute(mesh, name="Col"):
    """Create or reset color attribute on target mesh."""
    if hasattr(mesh, 'color_attributes'):
        # Remove existing if present
        existing = mesh.color_attributes.get(name)
        if existing:
            mesh.color_attributes.remove(existing)

        # Create new CORNER-domain attribute (per face-corner, like UVs)
        attr = mesh.color_attributes.new(
            name=name,
            type='BYTE_COLOR',  # 8-bit per channel, good for export
            domain='CORNER'
        )
        return attr
    return None


def set_active_color_attribute(mesh, name):
    """Set color attribute as active for rendering/export."""
    if hasattr(mesh, 'color_attributes'):
        attr = mesh.color_attributes.get(name)
        if attr:
            mesh.color_attributes.active_color = attr
            idx = list(mesh.color_attributes).index(attr)
            if hasattr(mesh.color_attributes, 'render_color_index'):
                mesh.color_attributes.render_color_index = idx


# ============================================================================
# CORE TRANSFER FUNCTION
# ============================================================================

def transfer_vertex_colors(source_obj, target_obj, report=None):
    """
    Transfer vertex colors from source mesh to target mesh.

    Args:
        source_obj: Blender object with vertex colors (high-poly)
        target_obj: Blender object to receive colors (low-poly)
        report: Optional list to collect log messages

    Returns:
        True on success, False on failure
    """
    if report is None:
        report = []

    start_time = time.time()

    ensure_object_mode()

    # Validate inputs
    if not source_obj or source_obj.type != 'MESH':
        log_print("[ERROR] Source object is not a mesh!", report)
        return False

    if not target_obj or target_obj.type != 'MESH':
        log_print("[ERROR] Target object is not a mesh!", report)
        return False

    log_print(f"\n{'='*60}", report)
    log_print("TransferVertexColors V1", report)
    log_print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report)
    log_print(f"{'='*60}\n", report)

    source_mesh = source_obj.data
    target_mesh = target_obj.data

    log_print(f"[Source] {source_obj.name}", report)
    log_print(f"         Faces: {len(source_mesh.polygons)}", report)
    log_print(f"         Vertices: {len(source_mesh.vertices)}", report)

    log_print(f"[Target] {target_obj.name}", report)
    log_print(f"         Faces: {len(target_mesh.polygons)}", report)
    log_print(f"         Vertices: {len(target_mesh.vertices)}", report)

    # Find source color attribute
    source_color_attr = get_source_color_attribute(source_mesh)
    if not source_color_attr:
        log_print("[ERROR] No color attribute found on source mesh!", report)
        log_print("        Expected one of: " + ", ".join(SOURCE_COLOR_NAMES), report)
        return False

    log_print(f"\n[Source Color] Using: {source_color_attr.name}", report)
    log_print(f"               Domain: {source_color_attr.domain}", report)
    log_print(f"               Type: {source_color_attr.data_type}", report)

    # Build face color lookup from source
    log_print("\n[Step 1] Building source face color lookup...", report)

    source_face_colors = []
    unique_colors = set()

    for poly in source_mesh.polygons:
        # Get color from first loop of face (they should all be the same for flat shading)
        first_loop_idx = poly.loop_start
        if first_loop_idx < len(source_color_attr.data):
            col = source_color_attr.data[first_loop_idx].color
            color = (col[0], col[1], col[2], col[3])
            source_face_colors.append(color)
            unique_colors.add((round(col[0], 2), round(col[1], 2), round(col[2], 2)))
        else:
            source_face_colors.append((1.0, 0.0, 1.0, 1.0))  # Magenta for missing

    log_print(f"         Cached {len(source_face_colors)} face colors", report)
    log_print(f"         Found {len(unique_colors)} unique colors", report)

    if unique_colors:
        sample = list(unique_colors)[:5]
        log_print(f"         Sample colors: {sample}", report)

    # Build BVH tree from source mesh
    log_print("\n[Step 2] Building BVH tree for source mesh...", report)

    # Get vertices in world space for accurate nearest-point queries
    source_matrix = source_obj.matrix_world
    vertices = [(source_matrix @ v.co) for v in source_mesh.vertices]
    polygons = [tuple(p.vertices) for p in source_mesh.polygons]

    bvh = BVHTree.FromPolygons(vertices, polygons)
    log_print(f"         BVH tree built with {len(polygons)} faces", report)

    # Create target color attribute
    log_print(f"\n[Step 3] Creating color attribute on target...", report)
    target_color_attr = create_target_color_attribute(target_mesh, OUTPUT_COLOR_NAME)
    if not target_color_attr:
        log_print("[ERROR] Failed to create color attribute on target!", report)
        return False
    log_print(f"         Created: {OUTPUT_COLOR_NAME}", report)

    # Transfer colors
    log_print(f"\n[Step 4] Transferring colors ({TRANSFER_MODE} mode)...", report)

    target_matrix = target_obj.matrix_world
    total_faces = len(target_mesh.polygons)
    colors_found = 0
    colors_missing = 0
    color_distribution = {}

    for i, poly in enumerate(target_mesh.polygons):
        # Get face center in world space
        face_center_local = poly.center
        face_center_world = target_matrix @ face_center_local

        # Find closest point on source mesh
        location, normal, face_idx, distance = bvh.find_nearest(face_center_world)

        color = None
        if face_idx is not None and face_idx < len(source_face_colors):
            color = source_face_colors[face_idx]
            colors_found += 1

            # Track distribution
            color_key = (round(color[0], 2), round(color[1], 2), round(color[2], 2))
            color_distribution[color_key] = color_distribution.get(color_key, 0) + 1
        else:
            color = (1.0, 0.0, 1.0, 1.0)  # Magenta for missing
            colors_missing += 1

        # Apply same color to all loops of target face (flat shading)
        for loop_idx in poly.loop_indices:
            target_color_attr.data[loop_idx].color = color

        # Progress update
        if SHOW_PROGRESS and i > 0 and i % PROGRESS_INTERVAL == 0:
            pct = (i / total_faces) * 100
            print(f"         Progress: {i}/{total_faces} ({pct:.1f}%)")

    # Finalize
    target_mesh.update()
    set_active_color_attribute(target_mesh, OUTPUT_COLOR_NAME)

    elapsed = time.time() - start_time

    log_print(f"\n[Results]", report)
    log_print(f"         Colors found: {colors_found}", report)
    log_print(f"         Colors missing: {colors_missing}", report)
    log_print(f"         Unique colors transferred: {len(color_distribution)}", report)
    log_print(f"         Time: {elapsed:.2f}s", report)

    if color_distribution:
        sorted_colors = sorted(color_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        log_print(f"         Top colors by frequency: {sorted_colors}", report)

    # Create debug material if requested
    if CREATE_DEBUG_MATERIAL:
        log_print("\n[Step 5] Creating debug material...", report)
        create_vertex_color_material(target_obj, report)

    log_print(f"\n{'='*60}", report)
    log_print("Transfer complete!", report)
    log_print(f"{'='*60}\n", report)

    # Write log
    log_to_text("\n".join(report))

    return True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def transfer_colors(source_obj, target_obj):
    """
    Transfer vertex colors from source to target mesh.

    Args:
        source_obj: Blender mesh object with vertex colors
        target_obj: Blender mesh object to receive colors

    Returns:
        True on success, False on failure
    """
    report = []
    result = transfer_vertex_colors(source_obj, target_obj, report)
    return result


def transfer_colors_by_name(source_name: str, target_name: str):
    """
    Transfer vertex colors using object names.

    Args:
        source_name: Name of source mesh object
        target_name: Name of target mesh object

    Returns:
        True on success, False on failure
    """
    source_obj = bpy.data.objects.get(source_name)
    target_obj = bpy.data.objects.get(target_name)

    if not source_obj:
        print(f"[ERROR] Source object '{source_name}' not found!")
        return False

    if not target_obj:
        print(f"[ERROR] Target object '{target_name}' not found!")
        return False

    return transfer_colors(source_obj, target_obj)


def transfer_colors_selected():
    """
    Transfer colors from second selected object to active object.

    Select both meshes, with target as active (last selected).
    """
    selected = bpy.context.selected_objects
    active = bpy.context.active_object

    if len(selected) < 2:
        print("[ERROR] Select exactly 2 meshes: source and target (target should be active)")
        return False

    # Find source (selected but not active)
    source_obj = None
    for obj in selected:
        if obj != active and obj.type == 'MESH':
            source_obj = obj
            break

    if not source_obj:
        print("[ERROR] Could not find source mesh in selection")
        return False

    if not active or active.type != 'MESH':
        print("[ERROR] Active object is not a mesh")
        return False

    print(f"[Transfer] Source: {source_obj.name} -> Target: {active.name}")
    return transfer_colors(source_obj, active)


# ============================================================================
# DEBUG MATERIAL
# ============================================================================

def create_vertex_color_material(obj, report=None):
    """
    Create a material that displays vertex colors.
    """
    mat_name = "M_TransferredColors"
    layer_name = OUTPUT_COLOR_NAME

    # Remove existing
    mat = bpy.data.materials.get(mat_name)
    if mat:
        bpy.data.materials.remove(mat)

    # Create new material
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Emission shader (shows colors without lighting interference)
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (200, 0)
    emission.inputs['Strength'].default_value = 1.0
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # Vertex color node
    vert_color = None

    # Try Color Attribute node (Blender 4.0+)
    try:
        vert_color = nodes.new('ShaderNodeColorAttribute')
        vert_color.layer_name = layer_name
    except:
        pass

    # Fall back to Vertex Color node (older Blender)
    if not vert_color:
        try:
            vert_color = nodes.new('ShaderNodeVertexColor')
            vert_color.layer_name = layer_name
        except:
            pass

    # Fall back to Attribute node
    if not vert_color:
        vert_color = nodes.new('ShaderNodeAttribute')
        vert_color.attribute_name = layer_name

    if vert_color:
        vert_color.location = (-100, 0)

        # Find color output
        color_out = None
        for out in vert_color.outputs:
            if out.type == 'RGBA' or out.name == 'Color':
                color_out = out
                break
        if not color_out and len(vert_color.outputs) > 0:
            color_out = vert_color.outputs[0]

        if color_out:
            links.new(color_out, emission.inputs['Color'])

    # Assign to object
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    if report:
        report.append(f"         Created material: {mat_name}")
        report.append(f"         View in Material Preview mode (Z > Material Preview)")


# ============================================================================
# MAIN
# ============================================================================

def run():
    """
    Run based on configuration at top of file.
    """
    report = []

    # Determine source and target
    source_obj = None
    target_obj = None

    if TARGET_MESH == "ACTIVE":
        target_obj = bpy.context.active_object
    else:
        target_obj = bpy.data.objects.get(TARGET_MESH)

    if SOURCE_MESH == "SELECTED":
        # Use second selected object
        selected = bpy.context.selected_objects
        for obj in selected:
            if obj != target_obj and obj.type == 'MESH':
                source_obj = obj
                break
    else:
        source_obj = bpy.data.objects.get(SOURCE_MESH)

    if not source_obj:
        print(f"[ERROR] Source mesh not found: {SOURCE_MESH}")
        print("        Select both meshes with target as active, or set SOURCE_MESH name.")
        return

    if not target_obj:
        print(f"[ERROR] Target mesh not found: {TARGET_MESH}")
        return

    transfer_vertex_colors(source_obj, target_obj, report)


if __name__ == "__main__":
    run()
