"""
Decimate V1 - Stylized Low-Poly Reduction

Reduces high-poly meshes to stylized low-poly with:
- Hole filling
- Internal geometry removal (jacketing/skin cleanup)
- Optional remesh (voxel or quadriflow)
- Flat planar regions preserved
- Hard edges at angle boundaries
- Clean topology for vertex color painting

Pipeline:
1. [Optional] Remesh - clean topology, fill holes, remove internal geo
2. [Optional] Fill holes - close remaining open edges
3. [Optional] Remove internal geometry - ray-based hidden face detection
4. Planar decimate - merge coplanar faces (within angle threshold)
5. Collapse decimate - reduce to target poly count
6. Mark sharp edges - for hard edge rendering and clean UV islands
7. Cleanup - merge verts, triangulate n-gons

Best for: Stylized characters, props with flat-shaded aesthetic
"""

import bpy
import bmesh
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from datetime import datetime
import time

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            CONFIGURATION                                   ║
# ║                     Edit these settings before running                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ---- Remesh Options ----
# Remesh mode: "NONE", "SHARP", "VOXEL", "VOXEL_HIGH", or "QUAD"
#   NONE       = Skip remesh, keep original topology
#   SHARP      = Octree-based remesh with sharp edge preservation (BEST FOR THIN GEOMETRY!)
#                Creates clean watertight mesh while preserving thin walls like lips, ears, fingers
#   VOXEL      = Voxel remesh at moderate resolution
#                WARNING: Destroys thin geometry! Only use for solid/thick meshes
#   VOXEL_HIGH = High-resolution voxel remesh (1M+ faces) then decimate down
#                Best for: Clean normals, proper cavity handling, watertight mesh
#                Trade-off: Loses some sharp edges (planar decimate helps recover them)
#   QUAD       = Quadriflow remesh (creates smooth quad flow, NOT flat-faceted)
#
# RECOMMENDED:
#   - For THIN geometry (mouths, lips, ears, fingers): Use SHARP
#   - For SOLID geometry with cavities (full characters): Use VOXEL_HIGH
#   - For simple solid props: Use VOXEL
#   - For already-clean meshes: Use NONE
REMESH_MODE = "NONE"

# ---- SHARP Remesh Settings ----
# Octree depth for SHARP remesh (higher = more detail, more polygons)
# 6 = ~50K faces, 7 = ~200K faces, 8 = ~800K faces, 9 = ~2M faces
# Use high value (8-9) to preserve detail, then decimate down
SHARP_OCTREE_DEPTH = 8

# Sharpness threshold for SHARP remesh (0.0-1.0)
# Higher = more sharp edges preserved
SHARP_THRESHOLD = 1.0

# ---- VOXEL Remesh Settings ----
# Voxel size for VOXEL remesh (smaller = more detail/higher poly, larger = less detail)
# "AUTO" = Calculate based on mesh size to preserve detail
# Or specify a number: 0.0005 = very fine, 0.002 = fine, 0.01 = medium, 0.05 = coarse
VOXEL_SIZE = "AUTO"

# Target poly count AFTER voxel remesh (before decimation)
# This controls how detailed the voxel remesh should be
VOXEL_TARGET_POLYS = 100000

# ---- VOXEL_HIGH Settings ---- (DEPRECATED - use SHARP instead for characters)
# NOTE: VOXEL_HIGH creates double-walled meshes with small voxel sizes,
#       resulting in 50% inverted normals. Not recommended for characters.
#       Use SHARP remesh for characters with thin features (lips, ears, etc.)
VOXEL_HIGH_TARGET = 1000000
VOXEL_HIGH_VOXEL_SIZE = None  # Manual override (None = auto)

# Target face count for QUAD remesh (Quadriflow)
QUAD_TARGET_FACES = 10000

# ---- Hole Filling ----
# Fill holes in mesh (open boundaries)
FILL_HOLES = True

# Maximum edges for a hole to be filled (0 = fill all holes)
FILL_HOLES_MAX_SIDES = 100

# ---- Internal Geometry Removal ----
# Remove faces that can't be seen from outside (jacketing/skin)
REMOVE_INTERNAL = True

# Internal removal method:
#   "SIMPLE"  = Use Blender's select_interior_faces (fast but less accurate)
#   "RAYCAST" = Cast rays from multiple directions to detect hidden faces (more accurate)
# RAYCAST is better for complex geometry but slower
INTERNAL_REMOVAL_METHOD = "RAYCAST"

# Ray samples per face for RAYCAST internal detection (more = accurate but slower)
# 6 = cardinal directions, 14 = +corners, 26 = +edges (very thorough)
INTERNAL_RAY_SAMPLES = 14

# Offset for ray origin (to avoid self-intersection)
RAY_OFFSET = 0.001

# ---- Decimation ----
# Target face count (approximate) - set to 0 to skip collapse decimate
TARGET_FACES = 5000

# Angle threshold for planar merging (degrees)
# Faces within this angle are considered coplanar and will be merged
PLANAR_ANGLE = 7.0

# ---- Edge Marking ----
# Angle threshold for marking sharp edges (degrees)
# Edges with face angle greater than this become sharp/hard edges
SHARP_ANGLE = 14.0

# ---- Normal Fixing ----
# Fix face normals to point outward
#   "AUTO" = Only fix if inverted percentage exceeds threshold (see below)
#   True   = Always fix normals (Step 8)
#   False  = Never fix normals (trust source geometry)
FIX_NORMALS = "AUTO"

# Threshold for AUTO mode - fix normals if inverted percentage exceeds this value
# Set to 0 to always fix in AUTO mode, 5 to only fix if >5% inverted, etc.
FIX_NORMALS_THRESHOLD = 0  # Percent (0-100). Default 0 = always fix any inversions

# Normal fix method:
#   "BLENDER"   = Use Blender's normals_make_consistent (topology-based, best for clean meshes)
#   "DIRECTION" = Flip faces pointing toward mesh center (simple but fails on cavities like mouths)
#   "BOTH"      = Try Blender first, then direction-based if still inverted (current behavior)
# NOTE: For meshes with interior cavities (mouths, nostrils), "BLENDER" may work better
#       as it preserves cavity orientation. "DIRECTION" assumes all faces should point outward.
FIX_NORMALS_METHOD = "BOTH"

# ---- Cleanup ----
MERGE_DISTANCE = 0.0001  # Merge vertices closer than this (0 to skip)
REMOVE_DOUBLES = True    # Remove duplicate vertices

# ---- Pre-Cleanup (before decimate) ----
# Clean mesh BEFORE decimation to avoid creating non-manifold geometry
PRE_CLEANUP = True       # Fix mesh issues before decimating
FIX_NON_MANIFOLD = True  # Attempt to fix non-manifold geometry
AGGRESSIVE_MANIFOLD_FIX = True  # More aggressive non-manifold fixing (may lose some geometry)

# ---- Target Selection ----
# "ACTIVE" = use active/selected mesh
# "collection_name" = process all meshes in named collection
TARGET_MESH = "ACTIVE"

# ---- Preservation ----
PRESERVE_BOUNDARIES = True  # Preserve mesh boundaries during decimate
PRESERVE_SEAMS = True       # Preserve UV seams during decimation
APPLY_MODIFIERS = True      # Apply modifiers (False = keep editable)

# ---- Color Edge Detection ----
# Detect color edges from texture and mark as sharp/preserve
DETECT_COLOR_EDGES = True  # Enable color-based edge detection

# When to detect color edges:
#   "BEFORE" = Before decimate (slower on high-poly, but preserves edges during decimate)
#   "AFTER"  = After decimate (much faster, marks edges for rendering only)
DETECT_COLOR_EDGES_WHEN = "BEFORE"

# Color difference threshold (0-1) for marking edge as color boundary
# Lower = more edges detected, Higher = only major color changes
COLOR_EDGE_THRESHOLD = 0.15

# Source for color detection:
#   "TEXTURE" = Sample from UV-mapped texture
#   "VERTEX_COLOR" = Use existing vertex colors
#   "MATERIAL" = Use material assignments (different material = boundary)
COLOR_SOURCE = "TEXTURE"

# Mark detected color edges as sharp (for rendering)
MARK_COLOR_EDGES_SHARP = True

# Mark detected color edges as seams (for UV unwrapping)
MARK_COLOR_EDGES_SEAM = True

# ---- Color Preservation (for Remesh) ----
# Bake texture to vertex colors BEFORE remesh, then transfer back after
# This preserves color information through topology-destroying operations
PRESERVE_COLORS_THROUGH_REMESH = True

# ---- Vertex Color Baking ----
# Bake texture to vertex colors (works with or without remesh)
# Useful when UVs will be destroyed by decimation
BAKE_VERTEX_COLORS = True  # Bake texture colors to vertices before decimate

# ---- Progress & Performance ----
# Show progress in Blender's header bar
SHOW_PROGRESS = True

# Print step timings to console
SHOW_TIMINGS = True

# Force viewport update every N faces processed (0 = no updates, may freeze UI)
# Lower = more responsive but slower, Higher = faster but less feedback
PROGRESS_UPDATE_INTERVAL = 10000

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         END CONFIGURATION                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

LOG_TEXT_NAME = "Decimate_V1_Log.txt"


# ---------------- Logging ----------------
def log_to_text(s: str):
    txt = bpy.data.texts.get(LOG_TEXT_NAME)
    if not txt:
        txt = bpy.data.texts.new(LOG_TEXT_NAME)
    txt.clear()
    txt.write(s)


# ---------------- Progress Indicators ----------------
class ProgressTracker:
    """Track and display progress for long operations."""

    def __init__(self, total_steps, description="Processing"):
        self.total = total_steps
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
        self.wm = bpy.context.window_manager

        if SHOW_PROGRESS and self.total > 0:
            self.wm.progress_begin(0, self.total)
            print(f"\n[Progress] {description}: 0/{total_steps}")

    def update(self, step=None, message=None):
        """Update progress. Call frequently during long operations."""
        if step is not None:
            self.current = step
        else:
            self.current += 1

        if SHOW_PROGRESS and self.total > 0:
            self.wm.progress_update(self.current)

            # Print periodic updates
            if PROGRESS_UPDATE_INTERVAL > 0:
                if self.current - self.last_update >= PROGRESS_UPDATE_INTERVAL:
                    elapsed = time.time() - self.start_time
                    pct = (self.current / self.total) * 100
                    eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
                    msg = message if message else self.description
                    print(f"[Progress] {msg}: {self.current}/{self.total} ({pct:.1f}%) - ETA: {eta:.1f}s")
                    self.last_update = self.current

                    # Force UI update (helps prevent "Not Responding")
                    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def finish(self, message=None):
        """Complete progress tracking."""
        if SHOW_PROGRESS:
            self.wm.progress_end()

        elapsed = time.time() - self.start_time
        msg = message if message else self.description
        if SHOW_TIMINGS:
            print(f"[Complete] {msg}: {elapsed:.2f}s")

        return elapsed


def step_timer(name):
    """Simple decorator/context manager for timing steps."""
    class Timer:
        def __init__(self, name):
            self.name = name
            self.start = None

        def __enter__(self):
            self.start = time.time()
            if SHOW_PROGRESS:
                print(f"[Step] {self.name}...")
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self.start
            if SHOW_TIMINGS:
                print(f"[Step] {self.name}: {elapsed:.2f}s")

    return Timer(name)


# ---------------- Color Preservation ----------------
def bake_texture_to_vertex_colors(obj, report):
    """
    Bake texture colors to vertex colors.
    This preserves color information before remesh destroys UVs.

    Uses mesh API directly instead of bmesh to ensure colors are persisted.
    """
    ensure_object_mode()

    with step_timer("Baking texture to vertex colors"):
        bpy.context.view_layer.objects.active = obj
        mesh = obj.data

        # Create color attribute if needed
        color_attr = None
        if hasattr(mesh, 'color_attributes'):
            color_attr = mesh.color_attributes.get("BakedColors")
            if not color_attr:
                color_attr = mesh.color_attributes.new(
                    name="BakedColors",
                    type='FLOAT_COLOR',
                    domain='CORNER'
                )
                report.append("[Bake Colors] Created color attribute: BakedColors")
            else:
                report.append("[Bake Colors] Using existing color attribute: BakedColors")

        if not color_attr:
            report.append("[Bake Colors] ERROR: Could not create color attribute")
            return False

        # Get UV layer
        if not mesh.uv_layers.active:
            report.append("[Bake Colors] WARNING: No UV layer, cannot bake texture.")
            return False

        uv_layer = mesh.uv_layers.active

        # Find texture image
        image = None
        if obj.active_material and obj.active_material.node_tree:
            for node in obj.active_material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    image = node.image
                    break

        if not image:
            for img in bpy.data.images:
                if img.type == 'IMAGE' and img.size[0] > 0:
                    image = img
                    break

        if not image:
            report.append("[Bake Colors] WARNING: No texture image found.")
            return False

        report.append(f"[Bake Colors] Using image: {image.name} ({image.size[0]}x{image.size[1]})")

        # Get pixel data
        width, height = image.size
        pixels = list(image.pixels[:])  # Copy to list for faster access

        # Debug: check some pixel samples from the texture
        sample_pixels = []
        for test_y in [0, height//4, height//2, 3*height//4]:
            for test_x in [0, width//4, width//2, 3*width//4]:
                idx = (test_y * width + test_x) * 4
                if idx + 3 < len(pixels):
                    sample_pixels.append((round(pixels[idx], 2), round(pixels[idx+1], 2), round(pixels[idx+2], 2)))
        report.append(f"[Bake Colors] Texture sample pixels: {sample_pixels[:8]}")

        # Get UV data - mesh.uv_layers.active.data has one entry per loop
        uv_data = uv_layer.data

        total_loops = len(mesh.loops)
        progress = ProgressTracker(total_loops, "Baking vertex colors")

        unique_baked = set()
        sample_uvs = []

        # Write colors directly to mesh color attribute
        for loop_idx in range(total_loops):
            # Get UV for this loop
            uv = uv_data[loop_idx].uv

            # Sample image at UV
            px = int(uv.x % 1.0 * width) % width
            py = int(uv.y % 1.0 * height) % height
            idx = (py * width + px) * 4

            if idx + 3 < len(pixels):
                color = (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])
                color_attr.data[loop_idx].color = color
                unique_baked.add((round(color[0], 2), round(color[1], 2), round(color[2], 2)))
            else:
                color_attr.data[loop_idx].color = (1.0, 0.0, 1.0, 1.0)  # Magenta for missing

            # Collect some UV samples for debug
            if loop_idx < 10:
                sample_uvs.append((round(uv.x, 3), round(uv.y, 3)))

            if loop_idx % 50000 == 0:
                progress.update(loop_idx)

        progress.finish()

        report.append(f"[Bake Colors] Sample UVs: {sample_uvs}")
        report.append(f"[Bake Colors] Unique colors baked: {len(unique_baked)}")
        if unique_baked:
            report.append(f"[Bake Colors] Sample baked colors: {list(unique_baked)[:10]}")

        # Verify colors were saved
        verify_colors = set()
        for i in range(min(1000, len(color_attr.data))):
            c = color_attr.data[i].color
            verify_colors.add((round(c[0], 2), round(c[1], 2), round(c[2], 2)))
        report.append(f"[Bake Colors] Verified {len(verify_colors)} unique colors in mesh data")

        # Mark mesh as updated
        mesh.update()

        report.append(f"[Bake Colors] Baked {total_loops} vertex colors.")
        return True


def create_color_reference_copy(obj, report):
    """
    Create a hidden copy of the mesh to use as color reference after remesh.
    """
    ensure_object_mode()

    with step_timer("Creating color reference copy"):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate()

        ref_obj = bpy.context.active_object
        ref_obj.name = obj.name + "_ColorRef"
        ref_obj.hide_set(True)
        ref_obj.hide_render = True

        report.append(f"[Color Ref] Created reference copy: {ref_obj.name}")
        return ref_obj


def transfer_vertex_colors_from_reference(target_obj, source_obj, report):
    """
    Transfer vertex colors from reference mesh to target mesh after remesh.

    Uses direct nearest-point sampling to avoid color blending/averaging.
    For each face in target mesh, finds closest point on source mesh and
    copies the exact color without interpolation.
    """
    ensure_object_mode()

    with step_timer("Transferring vertex colors from reference"):
        # Unhide source temporarily
        source_obj.hide_set(False)
        source_obj.hide_viewport = False

        bpy.context.view_layer.objects.active = source_obj
        source_mesh = source_obj.data

        # Find the color attribute on source
        source_color_attr = None
        if hasattr(source_mesh, 'color_attributes') and source_mesh.color_attributes:
            for attr in source_mesh.color_attributes:
                if attr.name in ("BakedColors", "TransferredColors", "Color"):
                    source_color_attr = attr
                    break
            if not source_color_attr:
                source_color_attr = source_mesh.color_attributes[0]

        if not source_color_attr:
            report.append("[Color Transfer] WARNING: No color attribute found on source mesh")
            source_obj.hide_set(True)
            return False

        report.append(f"[Color Transfer] Using source color layer: {source_color_attr.name}")
        report.append(f"[Color Transfer] Source has {len(source_mesh.polygons)} faces, {len(source_color_attr.data)} color entries")

        # Build color lookup using mesh polygon data
        # We'll create a list where index = polygon index, value = color
        source_face_colors = []
        unique_colors = set()

        for poly in source_mesh.polygons:
            first_loop_idx = poly.loop_start
            if first_loop_idx < len(source_color_attr.data):
                col = source_color_attr.data[first_loop_idx].color
                color = (col[0], col[1], col[2], col[3])
                source_face_colors.append(color)
                # Track unique colors (rounded for comparison)
                unique_colors.add((round(col[0], 2), round(col[1], 2), round(col[2], 2)))
            else:
                source_face_colors.append((1.0, 0.0, 1.0, 1.0))

        report.append(f"[Color Transfer] Cached {len(source_face_colors)} face colors")
        report.append(f"[Color Transfer] Found {len(unique_colors)} unique colors in source")

        # Debug: show some sample colors
        if unique_colors:
            sample = list(unique_colors)[:10]
            report.append(f"[Color Transfer] Sample unique colors: {sample}")

        # Build BVH tree directly from mesh vertices and polygons
        # This ensures face indices match our color lookup
        # Use .copy() to avoid reference issues
        vertices = [v.co.copy() for v in source_mesh.vertices]
        polygons = [tuple(p.vertices) for p in source_mesh.polygons]

        bvh = BVHTree.FromPolygons(vertices, polygons)

        # Prepare target mesh
        bpy.context.view_layer.objects.active = target_obj
        bpy.ops.object.mode_set(mode='EDIT')

        target_bm = bmesh.from_edit_mesh(target_obj.data)
        target_bm.faces.ensure_lookup_table()

        # Create or get color layer on target
        target_color_layer = target_bm.loops.layers.color.get("TransferredColors")
        if not target_color_layer:
            target_color_layer = target_bm.loops.layers.color.new("TransferredColors")

        # For each face in target, find nearest face on source and copy color
        total_faces = len(target_bm.faces)
        progress = ProgressTracker(total_faces, "Transferring colors")

        colors_found = 0
        colors_missing = 0
        color_distribution = {}

        for i, face in enumerate(target_bm.faces):
            # Find closest point on source mesh to face center
            face_center = face.calc_center_median()
            location, normal, face_idx, distance = bvh.find_nearest(face_center)

            color = None
            if face_idx is not None and face_idx < len(source_face_colors):
                color = source_face_colors[face_idx]
                colors_found += 1

                # Track distribution
                color_key = (round(color[0], 2), round(color[1], 2), round(color[2], 2))
                color_distribution[color_key] = color_distribution.get(color_key, 0) + 1
            else:
                # Fallback: magenta for missing
                color = (1.0, 0.0, 1.0, 1.0)
                colors_missing += 1

            # Apply same color to all loops of target face (flat/faceted look)
            for loop in face.loops:
                loop[target_color_layer] = color

            if i % 10000 == 0:
                progress.update(i)

        progress.finish()

        bmesh.update_edit_mesh(target_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        source_obj.hide_set(True)

        report.append(f"[Color Transfer] Found colors for {colors_found} faces, missing {colors_missing}")
        report.append(f"[Color Transfer] Color distribution in output: {len(color_distribution)} unique colors")

        # Show top colors by frequency
        if color_distribution:
            sorted_colors = sorted(color_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
            report.append(f"[Color Transfer] Top colors: {sorted_colors}")

        return True


def cleanup_reference_copy(ref_obj, report):
    """Remove the temporary reference copy."""
    if ref_obj:
        bpy.data.objects.remove(ref_obj, do_unlink=True)
        report.append("[Cleanup] Removed color reference copy.")


def finalize_color_attribute(obj, report):
    """
    Finalize color attribute for export:
    1. Rename to "Color" (standard)
    2. Set as active render color
    3. Remove other color attributes
    """
    mesh = obj.data

    if not hasattr(mesh, 'color_attributes') or not mesh.color_attributes:
        report.append("[Color Finalize] WARNING: No color attributes found")
        return False

    # Find the color attribute we want to keep (prefer TransferredColors > BakedColors > first)
    source_attr = None
    for name in ("TransferredColors", "BakedColors", "Color"):
        for attr in mesh.color_attributes:
            if attr.name == name:
                source_attr = attr
                break
        if source_attr:
            break

    if not source_attr and len(mesh.color_attributes) > 0:
        source_attr = mesh.color_attributes[0]

    if not source_attr:
        report.append("[Color Finalize] WARNING: No valid color attribute found")
        return False

    # If already named "Color", just ensure it's active
    if source_attr.name == "Color":
        mesh.color_attributes.active_color = source_attr
        if hasattr(mesh.color_attributes, 'render_color_index'):
            mesh.color_attributes.render_color_index = list(mesh.color_attributes).index(source_attr)
        report.append("[Color Finalize] 'Color' already exists and set as active")
        return True

    # Create new "Color" attribute and copy data
    old_name = source_attr.name
    col_attr = mesh.color_attributes.new(
        name="Color",
        type=source_attr.data_type,
        domain=source_attr.domain
    )

    # Copy color data
    for i, src_color in enumerate(source_attr.data):
        col_attr.data[i].color = src_color.color

    # Set as active
    mesh.color_attributes.active_color = col_attr
    if hasattr(mesh.color_attributes, 'render_color_index'):
        mesh.color_attributes.render_color_index = list(mesh.color_attributes).index(col_attr)

    # Remove old attributes (keep only "Color")
    attrs_to_remove = [attr.name for attr in mesh.color_attributes if attr.name != "Color"]
    for attr_name in attrs_to_remove:
        attr = mesh.color_attributes.get(attr_name)
        if attr:
            mesh.color_attributes.remove(attr)

    report.append(f"[Color Finalize] Renamed '{old_name}' -> 'Color'")
    report.append(f"[Color Finalize] Set 'Color' as active render color")
    return True


def create_vertex_color_material(obj, report):
    """
    Create a material that displays vertex colors.
    Also finalizes color attribute to "Color" standard name.
    """
    mat_name = "M_VertexColors"

    # First, finalize the color attribute to "Color"
    finalize_color_attribute(obj, report)

    # Now use "Color" as the layer name
    layer_name = "Color"

    # Verify it exists
    if not hasattr(obj.data, 'color_attributes') or "Color" not in obj.data.color_attributes:
        # Fallback: find any color attribute
        if hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0:
            layer_name = obj.data.color_attributes[0].name
            report.append(f"[Material] Fallback to color layer: {layer_name}")
        else:
            report.append("[Material] WARNING: No color attribute found on mesh!")
            return

    # Always recreate material to ensure correct settings
    mat = bpy.data.materials.get(mat_name)
    if mat:
        bpy.data.materials.remove(mat)

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    # Clear default nodes
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    vertex_color = nodes.new('ShaderNodeVertexColor')
    vertex_color.location = (-300, 0)
    vertex_color.layer_name = layer_name

    # Connect nodes
    links = mat.node_tree.links
    links.new(vertex_color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    report.append(f"[Material] Created material with color layer: {layer_name}")

    # Assign to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    report.append(f"[Material] Assigned to {obj.name}")


# ---------------- Helpers ----------------
def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def get_face_count(obj):
    """Get face count from mesh."""
    if obj.type != 'MESH':
        return 0
    return len(obj.data.polygons)


def get_vertex_count(obj):
    """Get vertex count from mesh."""
    if obj.type != 'MESH':
        return 0
    return len(obj.data.vertices)


def get_target_meshes(report):
    """Get list of meshes to process based on TARGET_MESH setting."""
    meshes = []

    if TARGET_MESH == "ACTIVE":
        obj = bpy.context.active_object
        if obj and obj.type == 'MESH':
            meshes.append(obj)
        else:
            raise RuntimeError("No active mesh selected. Select a mesh and try again.")
    else:
        # Look for collection
        col = None
        for c in bpy.data.collections:
            if c.name.lower() == TARGET_MESH.lower():
                col = c
                break

        if col:
            meshes = [o for o in col.all_objects if o.type == 'MESH']
            if not meshes:
                raise RuntimeError(f"No meshes found in collection '{TARGET_MESH}'.")
        else:
            raise RuntimeError(f"Collection '{TARGET_MESH}' not found.")

    report.append(f"[Setup] Found {len(meshes)} mesh(es) to process.")
    return meshes


# ---------------- Color Edge Detection ----------------
def get_face_color_from_texture(obj, face, uv_layer, image, pixels_cache=None, width=None, height=None):
    """Sample texture color at face center UV."""
    if not uv_layer or not image:
        return None

    # Get average UV of face
    uv_sum_x = 0.0
    uv_sum_y = 0.0
    loop_count = len(face.loops)
    for loop in face.loops:
        uv = loop[uv_layer].uv
        uv_sum_x += uv.x
        uv_sum_y += uv.y
    uv_avg_x = uv_sum_x / loop_count
    uv_avg_y = uv_sum_y / loop_count

    # Use cached values if provided
    if width is None:
        width, height = image.size
    if pixels_cache is None:
        pixels_cache = image.pixels

    # Sample image at UV
    px = int(uv_avg_x % 1.0 * width) % width
    py = int(uv_avg_y % 1.0 * height) % height

    # Get pixel (RGBA)
    idx = (py * width + px) * 4
    if idx + 3 < len(pixels_cache):
        return Vector((pixels_cache[idx], pixels_cache[idx + 1], pixels_cache[idx + 2]))
    return None


def get_face_color_from_vertex_colors(obj, face, color_layer):
    """Get average vertex color for face."""
    if not color_layer:
        return None

    color_sum = Vector((0, 0, 0))
    count = 0
    for loop in face.loops:
        col = loop[color_layer]
        color_sum += Vector((col[0], col[1], col[2]))
        count += 1

    if count > 0:
        return color_sum / count
    return None


def color_difference(c1, c2):
    """Calculate color difference (0-1 range)."""
    if c1 is None or c2 is None:
        return 0
    return (c1 - c2).length / 1.732  # Normalize by max RGB distance (sqrt(3))


def detect_color_edges(obj, threshold, source, report):
    """
    Detect edges where color changes significantly.
    Mark them as sharp and/or seams for preservation during decimation.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Get color source
    uv_layer = None
    image = None
    color_layer = None

    if source == "TEXTURE":
        # Find active UV layer
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            report.append("[Color Edges] WARNING: No UV layer found, skipping color detection.")
            bpy.ops.object.mode_set(mode='OBJECT')
            return 0

        # Find texture from material
        if obj.active_material and obj.active_material.node_tree:
            for node in obj.active_material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    image = node.image
                    break

        if not image:
            # Try to find any image in the blend file
            for img in bpy.data.images:
                if img.type == 'IMAGE' and img.size[0] > 0:
                    image = img
                    report.append(f"[Color Edges] Using image: {img.name}")
                    break

        if not image:
            report.append("[Color Edges] WARNING: No texture image found, skipping color detection.")
            bpy.ops.object.mode_set(mode='OBJECT')
            return 0

        # Cache image data for faster access (HUGE performance boost)
        report.append(f"[Color Edges] Caching {image.size[0]}x{image.size[1]} texture pixels...")
        width, height = image.size
        pixels_cache = list(image.pixels[:])  # Copy to list for faster indexing
        report.append(f"[Color Edges] Cached {len(pixels_cache)} pixel values.")

    elif source == "VERTEX_COLOR":
        # Try to find any color layer (Blender 3.2+ uses different names)
        color_layer = bm.loops.layers.color.active
        if not color_layer:
            # Try to find any color layer by iterating
            for layer in bm.loops.layers.color.values():
                color_layer = layer
                report.append(f"[Color Edges] Using color layer: {layer.name}")
                break

        if not color_layer:
            report.append("[Color Edges] WARNING: No vertex color layer found, skipping.")
            report.append("[Color Edges] TIP: Try COLOR_SOURCE = 'TEXTURE' instead.")
            bpy.ops.object.mode_set(mode='OBJECT')
            return 0

    elif source == "MATERIAL":
        # Material-based - different materials = boundary
        pass

    # Calculate color for each face
    face_colors = {}
    total_faces = len(bm.faces)
    progress = ProgressTracker(total_faces, "Sampling face colors")

    # Get cached values for texture mode (defined above)
    if source == "TEXTURE":
        # pixels_cache, width, height defined above when source == "TEXTURE"
        pass
    else:
        pixels_cache = None
        width = height = None

    for i, face in enumerate(bm.faces):
        if source == "TEXTURE":
            face_colors[face.index] = get_face_color_from_texture(
                obj, face, uv_layer, image, pixels_cache, width, height
            )
        elif source == "VERTEX_COLOR":
            face_colors[face.index] = get_face_color_from_vertex_colors(obj, face, color_layer)
        elif source == "MATERIAL":
            # Use material index as "color"
            face_colors[face.index] = Vector((face.material_index, 0, 0))

        if i % 10000 == 0:
            progress.update(i)

    progress.finish("Face color sampling")

    # Find edges where adjacent faces have different colors
    color_edges = []
    total_edges = len(bm.edges)
    progress2 = ProgressTracker(total_edges, "Comparing edge colors")

    for i, edge in enumerate(bm.edges):
        if len(edge.link_faces) != 2:
            continue

        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        c1 = face_colors.get(f1.index)
        c2 = face_colors.get(f2.index)

        if source == "MATERIAL":
            # Different material = boundary
            if c1 is not None and c2 is not None and c1.x != c2.x:
                color_edges.append(edge)
        else:
            # Color difference threshold
            diff = color_difference(c1, c2)
            if diff > threshold:
                color_edges.append(edge)

        if i % 20000 == 0:
            progress2.update(i)

    progress2.finish("Edge color comparison")

    # Mark edges
    marked_count = 0
    for edge in color_edges:
        if MARK_COLOR_EDGES_SHARP:
            edge.smooth = False
        if MARK_COLOR_EDGES_SEAM:
            edge.seam = True
        marked_count += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    report.append(f"[Color Edges] Detected {marked_count} color boundary edges (threshold: {threshold})")
    return marked_count


def detect_material_boundaries(obj, report):
    """
    Simple material boundary detection - mark edges between different materials.
    Works even without textures.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    marked_count = 0

    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue

        f1, f2 = edge.link_faces[0], edge.link_faces[1]

        # Different material index = boundary
        if f1.material_index != f2.material_index:
            if MARK_COLOR_EDGES_SHARP:
                edge.smooth = False
            if MARK_COLOR_EDGES_SEAM:
                edge.seam = True
            marked_count += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    report.append(f"[Material Boundaries] Marked {marked_count} material boundary edges")
    return marked_count


# ---------------- Remesh Operations ----------------
def apply_sharp_remesh(obj, octree_depth, threshold, report):
    """
    Apply SHARP (octree-based) remesh - preserves sharp edges and thin geometry.

    This is the best option for thin geometry like lips, ears, fingers because
    it uses an octree subdivision that respects sharp features rather than
    a uniform voxel grid that can collapse thin walls.

    Higher octree_depth = more polygons but better detail preservation.
    The mesh will be high-poly after this, then decimated down.
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Add remesh modifier with SHARP mode
    mod = obj.modifiers.new(name="Sharp_Remesh", type='REMESH')
    mod.mode = 'SHARP'
    mod.octree_depth = octree_depth
    mod.sharpness = threshold
    mod.use_remove_disconnected = True  # Remove floating geometry
    mod.use_smooth_shade = False  # Flat shading for stylized look

    if APPLY_MODIFIERS:
        bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)
    report.append(f"[Sharp Remesh] {obj.name}: {initial_faces} -> {final_faces} faces (octree depth: {octree_depth})")

    # This is expected to INCREASE poly count significantly - that's intentional
    if final_faces > initial_faces:
        report.append(f"[Sharp Remesh] High-poly clean mesh created - will be reduced by decimation")

    return final_faces


def calculate_auto_voxel_size(obj, target_polys):
    """
    Calculate voxel size to achieve approximately target_polys faces.

    Voxel remesh creates roughly (volume / voxel_size^3) * surface_factor faces.
    We estimate based on bounding box and target poly count.
    """
    # Get bounding box dimensions
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_co = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_co = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))

    dimensions = max_co - min_co

    # Estimate surface area (rough approximation using bounding box)
    surface_area = 2 * (dimensions.x * dimensions.y +
                        dimensions.y * dimensions.z +
                        dimensions.x * dimensions.z)

    # Each voxel face contributes roughly voxel_size^2 to surface
    # target_polys ≈ surface_area / voxel_size^2
    # voxel_size ≈ sqrt(surface_area / target_polys)

    if target_polys > 0 and surface_area > 0:
        voxel_size = math.sqrt(surface_area / target_polys)
        # Clamp to reasonable range
        voxel_size = max(0.001, min(0.1, voxel_size))
        return voxel_size

    # Fallback: 1% of smallest dimension
    min_dim = min(dimensions.x, dimensions.y, dimensions.z)
    return max(0.001, min_dim * 0.01)


def apply_voxel_remesh(obj, voxel_size, report):
    """
    Apply voxel remesh - creates watertight mesh, fills holes, removes internal geo.
    Use small voxel size for high detail, then decimate afterwards.
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Calculate auto voxel size if needed
    if voxel_size == "AUTO" or voxel_size is None:
        actual_voxel_size = calculate_auto_voxel_size(obj, VOXEL_TARGET_POLYS)
        report.append(f"[Voxel Remesh] Auto voxel size: {actual_voxel_size:.4f} (targeting ~{VOXEL_TARGET_POLYS} polys)")
    else:
        actual_voxel_size = float(voxel_size)

    # Add remesh modifier
    mod = obj.modifiers.new(name="Voxel_Remesh", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = actual_voxel_size
    mod.use_smooth_shade = False  # Flat shading for stylized look
    mod.adaptivity = 0.0  # Uniform voxels

    if APPLY_MODIFIERS:
        bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)
    report.append(f"[Voxel Remesh] {obj.name}: {initial_faces} -> {final_faces} faces (voxel size: {actual_voxel_size:.4f})")

    # Warn if massive face reduction (might have lost thin geometry)
    if initial_faces > 0:
        reduction_ratio = final_faces / initial_faces
        if reduction_ratio < 0.3:
            report.append(f"[Voxel Remesh] WARNING: Large reduction ({reduction_ratio:.1%}) - thin geometry may be lost!")
            report.append(f"[Voxel Remesh] TIP: For thin geometry (lips, ears), use REMESH_MODE='NONE' instead")

    return final_faces


def apply_voxel_high_remesh(obj, target_faces, report):
    """
    Apply high-resolution voxel remesh then decimate down.

    This mode:
    1. Voxel remesh at high resolution (VOXEL_HIGH_TARGET faces)
    2. Results in clean watertight mesh with proper normals
    3. Later pipeline steps decimate down to final target

    Benefits:
    - Cleaner normals than SHARP remesh
    - Better cavity handling (mouths, nostrils)
    - Watertight mesh guaranteed

    Trade-offs:
    - DESTROYS thin geometry (lips, ears, fingers) - use SHARP instead!
    - Loses some sharp edges (planar decimate helps recover them)
    - Higher processing time

    WARNING: Voxel remesh is volumetric - geometry thinner than ~2x voxel size
    will be destroyed. For characters with thin features, use SHARP remesh.
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Use manual override if specified, otherwise auto-calculate with iteration
    if VOXEL_HIGH_VOXEL_SIZE is not None:
        voxel_size = VOXEL_HIGH_VOXEL_SIZE
        report.append(f"[VOXEL_HIGH] Using manual voxel size: {voxel_size:.6f}")
    else:
        # Iterative approach: start with estimate, refine based on actual results
        # Initial estimate from bounding box
        initial_estimate = calculate_auto_voxel_size(obj, target_faces)
        report.append(f"[VOXEL_HIGH] Initial voxel estimate: {initial_estimate:.6f}")

        # Test with non-destructive modifier to get actual face count
        mod = obj.modifiers.new(name="Voxel_Test", type='REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = initial_estimate
        mod.use_smooth_shade = False

        # Get face count from modifier preview (without applying)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        test_faces = len(obj_eval.data.polygons)

        # Remove test modifier
        obj.modifiers.remove(mod)

        report.append(f"[VOXEL_HIGH] Test result: {test_faces:,} faces at voxel {initial_estimate:.6f}")

        # Adjust voxel size based on ratio
        # faces ∝ 1/voxel_size², so voxel_size ∝ sqrt(faces/target)
        if test_faces > 0 and target_faces > 0:
            ratio = math.sqrt(test_faces / target_faces)
            voxel_size = initial_estimate * ratio

            # Clamp to reasonable range
            voxel_size = max(0.0001, min(0.1, voxel_size))
            report.append(f"[VOXEL_HIGH] Adjusted voxel size: {voxel_size:.6f} (ratio: {ratio:.2f})")
        else:
            voxel_size = initial_estimate

    # Warn about thin geometry destruction
    report.append(f"[VOXEL_HIGH] WARNING: Geometry thinner than ~{voxel_size*2:.4f} units will be destroyed!")
    report.append(f"[VOXEL_HIGH] For thin features (lips, ears), use REMESH_MODE='SHARP' instead")

    # Add remesh modifier
    mod = obj.modifiers.new(name="Voxel_High_Remesh", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = voxel_size
    mod.use_smooth_shade = False  # Flat shading for stylized look
    mod.adaptivity = 0.0  # Uniform voxels

    if APPLY_MODIFIERS:
        bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)
    report.append(f"[VOXEL_HIGH] {obj.name}: {initial_faces:,} -> {final_faces:,} faces")

    # Report accuracy
    accuracy = (final_faces / target_faces) * 100 if target_faces > 0 else 0
    report.append(f"[VOXEL_HIGH] Target accuracy: {accuracy:.1f}% of {target_faces:,} target")

    if final_faces < target_faces * 0.5:
        report.append(f"[VOXEL_HIGH] WARNING: Got fewer faces than expected.")
        report.append(f"[VOXEL_HIGH] Try: VOXEL_HIGH_VOXEL_SIZE = {voxel_size * 0.5:.6f}")

    return final_faces


def apply_quadriflow_remesh(obj, target_faces, report):
    """
    Apply Quadriflow remesh - creates clean quad topology.
    NOTE: Quadriflow creates smooth quad flow, NOT flat-faceted style.
    For stylized low-poly, use REMESH_MODE = "NONE" with planar decimate.
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Quadriflow API varies by Blender version - try different parameter sets
    try:
        # Blender 4.0+
        bpy.ops.object.quadriflow_remesh(
            target_faces=target_faces,
            use_mesh_symmetry=False,
            use_preserve_sharp=True,
            use_preserve_boundary=PRESERVE_BOUNDARIES,
        )
    except TypeError:
        # Older Blender versions
        try:
            bpy.ops.object.quadriflow_remesh(
                target_faces=target_faces,
                use_mesh_symmetry=False,
                use_preserve_sharp=True,
                use_preserve_boundary=PRESERVE_BOUNDARIES,
                preserve_paint_mask=False,
                smooth_normals=False
            )
        except TypeError:
            # Minimal parameters
            bpy.ops.object.quadriflow_remesh(target_faces=target_faces)

    final_faces = get_face_count(obj)
    report.append(f"[Quadriflow] {obj.name}: {initial_faces} -> {final_faces} faces (target: {target_faces})")
    report.append(f"[Quadriflow] WARNING: Quadriflow creates smooth quads, not flat-faceted style!")
    return final_faces


# ---------------- Hole Filling ----------------
def fill_holes(obj, max_sides, report):
    """
    Fill holes (open boundaries) in the mesh.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Find boundary edges (edges with only one face)
    boundary_edges = [e for e in bm.edges if e.is_boundary]

    if not boundary_edges:
        report.append(f"[Fill Holes] {obj.name}: No holes found.")
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    # Find boundary loops (holes)
    holes_filled = 0
    processed_edges = set()

    for start_edge in boundary_edges:
        if start_edge in processed_edges:
            continue

        # Trace the boundary loop
        loop_edges = []
        current_edge = start_edge
        current_vert = start_edge.verts[0]

        while True:
            loop_edges.append(current_edge)
            processed_edges.add(current_edge)

            # Find next boundary edge
            other_vert = current_edge.other_vert(current_vert)
            next_edge = None

            for e in other_vert.link_edges:
                if e.is_boundary and e not in processed_edges:
                    next_edge = e
                    break

            if next_edge is None or next_edge == start_edge:
                break

            current_edge = next_edge
            current_vert = other_vert

        # Check if we should fill this hole
        if max_sides == 0 or len(loop_edges) <= max_sides:
            # Get vertices of the hole
            verts = []
            for edge in loop_edges:
                for v in edge.verts:
                    if v not in verts:
                        verts.append(v)

            if len(verts) >= 3:
                try:
                    bmesh.ops.contextual_create(bm, geom=verts)
                    holes_filled += 1
                except:
                    pass  # Some holes can't be filled cleanly

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    report.append(f"[Fill Holes] {obj.name}: Filled {holes_filled} holes")
    return holes_filled


def fill_holes_simple(obj, report):
    """
    Simple hole filling using Blender's built-in operator.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all boundary edges
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True,
                                      use_multi_face=False, use_non_contiguous=False, use_verts=False)

    # Fill holes
    try:
        bpy.ops.mesh.fill_holes(sides=FILL_HOLES_MAX_SIDES if FILL_HOLES_MAX_SIDES > 0 else 1000)
        report.append(f"[Fill Holes] {obj.name}: Holes filled")
    except:
        report.append(f"[Fill Holes] {obj.name}: No holes to fill or fill failed")

    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')


# ---------------- Internal Geometry Removal ----------------
def remove_internal_geometry(obj, ray_samples, report):
    """
    Remove internal/hidden faces using ray casting.
    Faces that can't be seen from outside are considered internal.
    """
    ensure_object_mode()

    # Create BVH tree for ray casting
    bpy.context.view_layer.objects.active = obj
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.data

    bm = bmesh.new()
    bm.from_mesh(mesh_eval)
    bm.faces.ensure_lookup_table()

    bvh = BVHTree.FromBMesh(bm)

    # Calculate mesh bounds for ray directions
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    for v in bm.verts:
        for i in range(3):
            min_co[i] = min(min_co[i], v.co[i])
            max_co[i] = max(max_co[i], v.co[i])

    # Expand bounds slightly
    padding = (max_co - min_co).length * 0.1
    min_co -= Vector((padding, padding, padding))
    max_co += Vector((padding, padding, padding))

    # Ray directions (from 6 cardinal + 8 corners = 14 directions)
    ray_origins = [
        Vector((min_co.x, (min_co.y + max_co.y) / 2, (min_co.z + max_co.z) / 2)),  # -X
        Vector((max_co.x, (min_co.y + max_co.y) / 2, (min_co.z + max_co.z) / 2)),  # +X
        Vector(((min_co.x + max_co.x) / 2, min_co.y, (min_co.z + max_co.z) / 2)),  # -Y
        Vector(((min_co.x + max_co.x) / 2, max_co.y, (min_co.z + max_co.z) / 2)),  # +Y
        Vector(((min_co.x + max_co.x) / 2, (min_co.y + max_co.y) / 2, min_co.z)),  # -Z
        Vector(((min_co.x + max_co.x) / 2, (min_co.y + max_co.y) / 2, max_co.z)),  # +Z
    ]

    # Track which faces are visible from outside
    visible_faces = set()

    for face in bm.faces:
        face_center = face.calc_center_median()
        face_normal = face.normal

        # Check if face is visible from any direction
        for origin in ray_origins:
            direction = (face_center - origin).normalized()

            # Cast ray from outside toward face center
            hit, hit_normal, hit_index, hit_dist = bvh.ray_cast(origin, direction)

            if hit is not None and hit_index == face.index:
                # This face is the first hit - it's visible
                visible_faces.add(face.index)
                break

            # Also check from face center outward (catches some edge cases)
            offset_origin = face_center + face_normal * RAY_OFFSET
            outward_dir = face_normal

            hit, hit_normal, hit_index, hit_dist = bvh.ray_cast(offset_origin, outward_dir)

            if hit is None:
                # Ray escaped - face is on the outside
                visible_faces.add(face.index)
                break

    bm.free()

    # Now remove internal faces from the actual mesh
    internal_count = 0

    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    faces_to_delete = []
    for face in bm.faces:
        if face.index not in visible_faces:
            faces_to_delete.append(face)
            internal_count += 1

    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    report.append(f"[Internal Removal] {obj.name}: Removed {internal_count} internal faces")
    return internal_count


def remove_internal_simple(obj, report):
    """
    Simpler internal geometry removal - select interior faces.
    Uses Blender's built-in selection tools.
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')

    # Select interior faces (faces that are completely surrounded)
    bpy.ops.mesh.select_interior_faces()

    # Delete selected
    bpy.ops.mesh.delete(type='FACE')

    bpy.ops.object.mode_set(mode='OBJECT')

    final_faces = get_face_count(obj)
    removed = initial_faces - final_faces

    report.append(f"[Internal Removal] {obj.name}: Removed {removed} interior faces")
    return removed


# ---------------- Decimate Operations ----------------
def apply_planar_decimate(obj, angle_deg, report):
    """
    Apply planar decimation to merge coplanar faces.
    This is the key step for stylized flat-shaded look.
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)
    angle_rad = math.radians(angle_deg)

    # Add planar decimate modifier
    mod = obj.modifiers.new(name="Planar_Decimate", type='DECIMATE')
    mod.decimate_type = 'DISSOLVE'  # Planar mode
    mod.angle_limit = angle_rad
    mod.use_dissolve_boundaries = not PRESERVE_BOUNDARIES

    # Delimit options - what edges to preserve
    # NORMAL = preserve edges based on normal angle
    # SEAM = preserve UV seams (important for color edge detection)
    # SHARP = preserve sharp edges
    # MATERIAL = preserve material boundaries
    delimit_set = {'NORMAL'}
    if PRESERVE_SEAMS or DETECT_COLOR_EDGES:
        delimit_set.add('SEAM')
    if DETECT_COLOR_EDGES and MARK_COLOR_EDGES_SHARP:
        delimit_set.add('SHARP')
    delimit_set.add('MATERIAL')  # Always preserve material boundaries

    mod.delimit = delimit_set
    report.append(f"[Planar] Delimit: {delimit_set}")

    if APPLY_MODIFIERS:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)
    reduction = ((initial_faces - final_faces) / initial_faces * 100) if initial_faces > 0 else 0

    report.append(f"[Planar] {obj.name}: {initial_faces} -> {final_faces} faces ({reduction:.1f}% reduction)")
    return final_faces


def apply_collapse_decimate(obj, target_faces, report):
    """
    Apply collapse decimation to reach target face count.
    """
    ensure_object_mode()

    current_faces = get_face_count(obj)
    if current_faces <= target_faces:
        report.append(f"[Collapse] {obj.name}: Already at {current_faces} faces (target: {target_faces}), skipping.")
        return current_faces

    # Calculate ratio
    ratio = target_faces / current_faces

    # Add collapse decimate modifier
    mod = obj.modifiers.new(name="Collapse_Decimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    mod.use_collapse_triangulate = False  # Keep quads where possible

    if APPLY_MODIFIERS:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)
    report.append(f"[Collapse] {obj.name}: {current_faces} -> {final_faces} faces (target was {target_faces})")
    return final_faces


def mark_sharp_edges(obj, angle_deg, report):
    """
    Mark edges as sharp based on face angle.
    Sharp edges = hard edges for shading and UV boundaries.
    """
    ensure_object_mode()

    angle_rad = math.radians(angle_deg)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    sharp_count = 0

    for edge in bm.edges:
        # Skip boundary edges (only one face)
        if len(edge.link_faces) != 2:
            continue

        # Calculate angle between adjacent faces
        face1, face2 = edge.link_faces[0], edge.link_faces[1]

        # Skip degenerate faces (zero-length normal)
        n1 = face1.normal
        n2 = face2.normal
        if n1.length < 0.0001 or n2.length < 0.0001:
            continue

        try:
            angle = n1.angle(n2)
        except ValueError:
            # Zero-length vector, skip
            continue

        # Mark as sharp if angle exceeds threshold
        if angle > angle_rad:
            edge.smooth = False
            sharp_count += 1
        else:
            edge.smooth = True

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Enable auto smooth to use sharp edges (Blender version compatibility)
    # Blender 4.1+ removed use_auto_smooth - sharp edges work automatically
    if hasattr(obj.data, 'use_auto_smooth'):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.pi  # Use sharp marks, not auto angle
    # else: Blender 4.1+ - sharp edges are respected automatically

    report.append(f"[Sharp] {obj.name}: Marked {sharp_count} sharp edges (threshold: {angle_deg}°)")
    return sharp_count


def cleanup_mesh(obj, report):
    """
    Clean up mesh - merge vertices, remove doubles, remove loose geometry.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    initial_verts = len(bm.verts)
    initial_faces = len(bm.faces)

    # Step 1: Remove duplicate vertices
    if MERGE_DISTANCE > 0 or REMOVE_DOUBLES:
        merge_dist = MERGE_DISTANCE if MERGE_DISTANCE > 0 else 0.0001
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)

    doubles_removed = initial_verts - len(bm.verts)

    # Step 2: Remove loose vertices (not connected to any face)
    # This is critical - collapse decimate can leave orphaned vertices
    loose_verts = [v for v in bm.verts if len(v.link_faces) == 0]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    # Step 3: Remove loose edges (not connected to any face)
    loose_edges = [e for e in bm.edges if len(e.link_faces) == 0]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')

    final_verts = len(bm.verts)
    final_faces = len(bm.faces)
    loose_removed = (initial_verts - doubles_removed) - final_verts

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    if doubles_removed > 0:
        report.append(f"[Cleanup] {obj.name}: Removed {doubles_removed} duplicate vertices")
    if loose_removed > 0:
        report.append(f"[Cleanup] {obj.name}: Removed {loose_removed} loose vertices/edges")
    if initial_faces != final_faces:
        report.append(f"[Cleanup] {obj.name}: {initial_faces} -> {final_faces} faces")


def fix_non_manifold_geometry(obj, report):
    """
    Aggressively fix non-manifold geometry.

    Non-manifold types:
    - Wire edges: edges not connected to any face
    - Boundary edges: edges connected to only 1 face (open mesh)
    - Multi-face edges: edges connected to 3+ faces (impossible geometry)
    - Non-contiguous: faces sharing only a vertex, not an edge

    Strategy:
    1. Delete wire edges (useless)
    2. Delete faces causing multi-face edges
    3. Fill small holes (boundary edges)
    4. Split non-contiguous parts
    5. Repeat until clean or max iterations
    """
    max_iterations = 5

    for iteration in range(max_iterations):
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        initial_faces = len(bm.faces)

        # Step 1: Remove wire edges (edges with no faces)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='EDGE')
        try:
            bpy.ops.mesh.select_non_manifold(extend=False, use_wire=True, use_boundary=False,
                                              use_multi_face=False, use_non_contiguous=False, use_verts=False)
            bm = bmesh.from_edit_mesh(obj.data)
            wire_edges = sum(1 for e in bm.edges if e.select)
            if wire_edges > 0:
                bpy.ops.mesh.delete(type='EDGE')
                report.append(f"[Manifold Fix] Iteration {iteration+1}: Removed {wire_edges} wire edges")
        except:
            pass

        # Step 2: Find and delete faces causing multi-face edges (3+ faces sharing an edge)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='FACE')

        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        multi_face_edges = [e for e in bm.edges if len(e.link_faces) > 2]

        if multi_face_edges:
            # Select faces connected to multi-face edges
            # Keep the 2 largest faces, delete the rest
            faces_to_delete = set()

            for edge in multi_face_edges:
                linked_faces = list(edge.link_faces)
                if len(linked_faces) > 2:
                    # Sort by area, keep 2 largest
                    linked_faces.sort(key=lambda f: f.calc_area(), reverse=True)
                    for f in linked_faces[2:]:
                        faces_to_delete.add(f)

            if faces_to_delete:
                for face in faces_to_delete:
                    if face.is_valid:
                        face.select = True

                bmesh.update_edit_mesh(obj.data)
                bpy.ops.mesh.delete(type='FACE')
                report.append(f"[Manifold Fix] Iteration {iteration+1}: Removed {len(faces_to_delete)} faces from multi-face edges")

        # Step 3: Remove interior faces (faces completely inside the mesh)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='FACE')
        try:
            bpy.ops.mesh.select_interior_faces()
            bm = bmesh.from_edit_mesh(obj.data)
            interior = sum(1 for f in bm.faces if f.select)
            if interior > 0:
                bpy.ops.mesh.delete(type='FACE')
                report.append(f"[Manifold Fix] Iteration {iteration+1}: Removed {interior} interior faces")
        except:
            pass

        # Step 4: Dissolve degenerate geometry
        bpy.ops.mesh.select_all(action='SELECT')
        try:
            bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
        except:
            pass

        # Step 5: Fill small holes (up to 8 sides)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='EDGE')
        try:
            bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True,
                                              use_multi_face=False, use_non_contiguous=False, use_verts=False)
            bm = bmesh.from_edit_mesh(obj.data)
            boundary_edges = sum(1 for e in bm.edges if e.select)
            if boundary_edges > 0 and boundary_edges < 500:  # Don't fill if too many (probably intentional openings)
                bpy.ops.mesh.fill_holes(sides=8)
                report.append(f"[Manifold Fix] Iteration {iteration+1}: Filled holes (had {boundary_edges} boundary edges)")
        except:
            pass

        # Step 6: Merge doubles that might have been created
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=MERGE_DISTANCE)

        # Check remaining non-manifold
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='VERT')
        try:
            bpy.ops.mesh.select_non_manifold(extend=False, use_wire=True, use_boundary=True,
                                              use_multi_face=True, use_non_contiguous=True, use_verts=True)
            bm = bmesh.from_edit_mesh(obj.data)
            remaining = sum(1 for v in bm.verts if v.select)

            if remaining == 0:
                report.append(f"[Manifold Fix] Success! Mesh is now manifold after {iteration+1} iteration(s)")
                break
            elif iteration == max_iterations - 1:
                report.append(f"[Manifold Fix] {remaining} non-manifold verts remain after {max_iterations} iterations")
                report.append(f"[Manifold Fix] TIP: Some geometry may need manual fixing or VOXEL remesh")
        except:
            pass

        bm = bmesh.from_edit_mesh(obj.data)
        final_faces = len(bm.faces)

        # If we didn't change anything, stop iterating
        if final_faces == initial_faces and iteration > 0:
            report.append(f"[Manifold Fix] No more changes possible, stopping at iteration {iteration+1}")
            break

    bpy.ops.mesh.select_all(action='DESELECT')


def pre_cleanup_mesh(obj, report):
    """
    Pre-cleanup mesh BEFORE decimation.
    Fixes common issues that cause non-manifold geometry after decimate.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')

    initial_verts = len(obj.data.vertices)
    initial_faces = len(obj.data.polygons)

    # Merge doubles first
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=MERGE_DISTANCE)

    merged = initial_verts - len(obj.data.vertices)
    if merged > 0:
        report.append(f"[Pre-Cleanup] Merged {merged} duplicate vertices")

    # Recalculate normals
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    report.append(f"[Pre-Cleanup] Normals recalculated")

    if FIX_NON_MANIFOLD:
        # Select and delete loose vertices/edges
        bpy.ops.mesh.select_all(action='DESELECT')
        try:
            bpy.ops.mesh.select_loose()
            # Use bmesh to get accurate selection count (obj.data.vertices doesn't sync in edit mode)
            bm = bmesh.from_edit_mesh(obj.data)
            loose_count = sum(1 for v in bm.verts if v.select)
            if loose_count > 0:
                bpy.ops.mesh.delete(type='VERT')
                report.append(f"[Pre-Cleanup] Removed {loose_count} loose vertices")
        except:
            pass

        # Remove degenerate faces (zero-area)
        bpy.ops.mesh.select_all(action='SELECT')
        try:
            bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
            report.append(f"[Pre-Cleanup] Removed degenerate geometry")
        except:
            pass

        # Check for non-manifold geometry
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='VERT')
        try:
            bpy.ops.mesh.select_non_manifold(extend=False, use_wire=True, use_boundary=True,
                                              use_multi_face=True, use_non_contiguous=True, use_verts=True)
            bm = bmesh.from_edit_mesh(obj.data)
            non_manifold_verts = sum(1 for v in bm.verts if v.select)

            if non_manifold_verts > 0:
                report.append(f"[Pre-Cleanup] Found {non_manifold_verts} non-manifold vertices")

                if AGGRESSIVE_MANIFOLD_FIX:
                    report.append(f"[Pre-Cleanup] Attempting aggressive manifold fix...")
                    fix_non_manifold_geometry(obj, report)
                else:
                    report.append(f"[Pre-Cleanup] TIP: Enable AGGRESSIVE_MANIFOLD_FIX or use VOXEL remesh")
        except:
            pass

    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    final_verts = len(obj.data.vertices)
    final_faces = len(obj.data.polygons)
    report.append(f"[Pre-Cleanup] {obj.name}: {initial_verts} -> {final_verts} verts, {initial_faces} -> {final_faces} faces")


def triangulate_ngons(obj, report):
    """
    Triangulate any n-gons (faces with more than 4 vertices).
    Keeps quads but splits larger polygons for game engine compatibility.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Find n-gons (faces with > 4 verts)
    ngons = [f for f in bm.faces if len(f.verts) > 4]

    if ngons:
        bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY', ngon_method='BEAUTY')
        report.append(f"[Triangulate] {obj.name}: Triangulated {len(ngons)} n-gons")

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')


def fix_normals_blender(obj, report):
    """
    Fix normals using Blender's topology-based recalculation.
    Best for clean meshes and preserves cavity orientation.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def fix_normals_after_remesh(obj, report):
    """
    Fix normals immediately after remesh.
    Uses the method specified by FIX_NORMALS_METHOD config.
    """
    ensure_object_mode()

    # Check current state
    out_before, in_before, total = check_normal_orientation(obj)
    if total == 0:
        return

    in_pct_before = (in_before / total) * 100
    report.append(f"[Post-Remesh Normals] Before: {out_before} out, {in_before} in ({in_pct_before:.1f}% inverted)")

    # If already good, skip
    if in_pct_before < 1.0:
        report.append("[Post-Remesh Normals] Already correct, skipping fix")
        return

    report.append(f"[Post-Remesh Normals] Using method: {FIX_NORMALS_METHOD}")

    if FIX_NORMALS_METHOD == "BLENDER":
        # Only use Blender's topology-based method
        fix_normals_blender(obj, report)
        out_after, in_after, total = check_normal_orientation(obj)
        in_pct_after = (in_after / total) * 100 if total > 0 else 0
        report.append(f"[Post-Remesh Normals] After Blender fix: {in_pct_after:.1f}% inverted")

    elif FIX_NORMALS_METHOD == "DIRECTION":
        # Only use direction-based method (may flip cavity faces incorrectly)
        fix_normals_by_direction(obj, report)

    else:  # "BOTH" - try Blender first, then direction if needed
        fix_normals_blender(obj, report)
        out_after, in_after, total = check_normal_orientation(obj)
        in_pct_after = (in_after / total) * 100 if total > 0 else 0
        report.append(f"[Post-Remesh Normals] After Blender: {in_pct_after:.1f}% inverted")

        if in_pct_after > 5:
            report.append("[Post-Remesh Normals] Still inverted, using direction-based fix...")
            report.append("[Post-Remesh Normals] WARNING: Direction fix may flip cavity faces (mouths, nostrils)")
            fix_normals_by_direction(obj, report)


# ---------------- Analysis ----------------
def check_normal_orientation(obj):
    """
    Check normal orientation and return statistics.
    Returns (pointing_out, pointing_in, total) without modifying the mesh.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    if len(bm.faces) == 0:
        bpy.ops.object.mode_set(mode='OBJECT')
        return (0, 0, 0)

    # Calculate mesh center from bounding box
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    for v in bm.verts:
        for i in range(3):
            min_co[i] = min(min_co[i], v.co[i])
            max_co[i] = max(max_co[i], v.co[i])

    mesh_center = (min_co + max_co) / 2

    pointing_out = 0
    pointing_in = 0

    for face in bm.faces:
        if face.normal.length < 0.0001:
            continue
        face_center = face.calc_center_median()
        to_face = face_center - mesh_center
        if face.normal.dot(to_face) >= 0:
            pointing_out += 1
        else:
            pointing_in += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    return (pointing_out, pointing_in, pointing_out + pointing_in)


def verify_normals(obj, report):
    """
    Verify normals are pointing outward after all processing.
    Reports statistics for debugging normal issues.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    if len(bm.faces) == 0:
        bpy.ops.object.mode_set(mode='OBJECT')
        return

    # Calculate mesh center
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    for v in bm.verts:
        for i in range(3):
            min_co[i] = min(min_co[i], v.co[i])
            max_co[i] = max(max_co[i], v.co[i])

    mesh_center = (min_co + max_co) / 2

    # Check each face
    pointing_out = 0
    pointing_in = 0
    degenerate = 0

    for face in bm.faces:
        face_center = face.calc_center_median()
        face_normal = face.normal

        if face_normal.length < 0.0001:
            degenerate += 1
            continue

        to_face = face_center - mesh_center
        dot = face_normal.dot(to_face)

        if dot >= 0:
            pointing_out += 1
        else:
            pointing_in += 1

    bpy.ops.object.mode_set(mode='OBJECT')

    total = pointing_out + pointing_in
    if total > 0:
        out_pct = (pointing_out / total) * 100
        in_pct = (pointing_in / total) * 100

        report.append(f"\n[Verify Normals] Final check:")
        report.append(f"    Outward: {pointing_out} ({out_pct:.1f}%)")
        report.append(f"    Inward:  {pointing_in} ({in_pct:.1f}%)")
        if degenerate > 0:
            report.append(f"    Degenerate: {degenerate}")

        if in_pct > 10:
            report.append(f"    WARNING: {in_pct:.1f}% of faces may be inverted!")
            report.append(f"    TIP: Some inward faces may be intentional (mouth, nostrils)")
        else:
            report.append(f"    OK: Normals look correct")


def fix_normals_by_direction(obj, report):
    """
    Fix normals by ensuring each face points away from mesh center.

    For non-manifold meshes with many disconnected components, graph-based
    propagation doesn't work well. Instead, we:
    1. Flip each individual face that points toward mesh center
    2. Run Blender's consistency check to fix winding order issues

    This is more aggressive but handles fragmented geometry better.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    if len(bm.faces) == 0:
        bpy.ops.object.mode_set(mode='OBJECT')
        report.append("[Step8] No faces to check")
        return

    # Calculate mesh center from bounding box
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    for v in bm.verts:
        for i in range(3):
            min_co[i] = min(min_co[i], v.co[i])
            max_co[i] = max(max_co[i], v.co[i])

    mesh_center = (min_co + max_co) / 2

    # First pass: count current state
    pointing_out = 0
    pointing_in = 0

    for face in bm.faces:
        if face.normal.length < 0.0001:
            continue
        face_center = face.calc_center_median()
        to_face = face_center - mesh_center
        if face.normal.dot(to_face) > 0:
            pointing_out += 1
        else:
            pointing_in += 1

    total = pointing_out + pointing_in
    if total > 0:
        report.append(f"[Step8] Before fix: {pointing_out} outward ({pointing_out*100/total:.1f}%), {pointing_in} inward ({pointing_in*100/total:.1f}%)")

    # Flip each face that points inward
    flipped = 0
    for face in bm.faces:
        if face.normal.length < 0.0001:
            continue

        face_center = face.calc_center_median()
        to_face = face_center - mesh_center

        # If normal points toward center (dot < 0), flip it
        if face.normal.dot(to_face) < 0:
            face.normal_flip()
            flipped += 1

    bmesh.update_edit_mesh(obj.data)
    report.append(f"[Step8] Flipped {flipped} inward-facing faces")

    # Verify after individual flips
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    pointing_out = 0
    pointing_in = 0
    for face in bm.faces:
        if face.normal.length < 0.0001:
            continue
        face_center = face.calc_center_median()
        to_face = face_center - mesh_center
        if face.normal.dot(to_face) > 0:
            pointing_out += 1
        else:
            pointing_in += 1

    total = pointing_out + pointing_in
    if total > 0:
        report.append(f"[Step8] After flip: {pointing_out} outward ({pointing_out*100/total:.1f}%), {pointing_in} inward ({pointing_in*100/total:.1f}%)")

    # DON'T run normals_make_consistent - it can undo our fixes on non-manifold meshes
    # The individual flip approach is more reliable for this geometry

    bpy.ops.object.mode_set(mode='OBJECT')
    report.append("[Step8] Normal fix complete")


def analyze_mesh(obj, report):
    """Analyze mesh topology for debugging."""
    if obj.type != 'MESH':
        return

    me = obj.data
    verts = len(me.vertices)
    edges = len(me.edges)
    faces = len(me.polygons)

    # Count by face size
    tris = sum(1 for p in me.polygons if len(p.vertices) == 3)
    quads = sum(1 for p in me.polygons if len(p.vertices) == 4)
    ngons = sum(1 for p in me.polygons if len(p.vertices) > 4)

    # Check for non-manifold
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')  # Required for select_non_manifold
    bpy.ops.mesh.select_all(action='DESELECT')

    try:
        bpy.ops.mesh.select_non_manifold()
        bm = bmesh.from_edit_mesh(obj.data)
        non_manifold = sum(1 for v in bm.verts if v.select)
    except RuntimeError:
        non_manifold = 0  # Skip if operator fails

    bpy.ops.object.mode_set(mode='OBJECT')

    report.append(f"[Analysis] {obj.name}:")
    report.append(f"           Verts: {verts}, Edges: {edges}, Faces: {faces}")
    report.append(f"           Tris: {tris}, Quads: {quads}, N-gons: {ngons}")
    if non_manifold > 0:
        report.append(f"           WARNING: {non_manifold} non-manifold vertices")


# ============================ MAIN ============================
def main():
    """Main decimation pipeline."""
    report = []
    report.append("Decimate V1 - Stylized Low-Poly Reduction")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("Settings:")
    report.append(f"  Remesh mode:  {REMESH_MODE}")
    if REMESH_MODE == "SHARP":
        report.append(f"  Sharp octree depth: {SHARP_OCTREE_DEPTH}")
        report.append(f"  Sharp threshold: {SHARP_THRESHOLD}")
    elif REMESH_MODE == "VOXEL":
        report.append(f"  Voxel size:   {VOXEL_SIZE}")
        report.append(f"  Voxel target polys: {VOXEL_TARGET_POLYS}")
    elif REMESH_MODE == "VOXEL_HIGH":
        report.append(f"  Voxel high target: {VOXEL_HIGH_TARGET:,} faces")
        if VOXEL_HIGH_VOXEL_SIZE is not None:
            report.append(f"  Voxel size override: {VOXEL_HIGH_VOXEL_SIZE}")
    elif REMESH_MODE == "QUAD":
        report.append(f"  Quad target:  {QUAD_TARGET_FACES}")
    report.append(f"  Pre-cleanup:  {PRE_CLEANUP}")
    report.append(f"  Aggressive manifold fix: {AGGRESSIVE_MANIFOLD_FIX}")
    report.append(f"  Fill holes:   {FILL_HOLES}")
    report.append(f"  Remove internal: {REMOVE_INTERNAL}")
    if REMOVE_INTERNAL:
        report.append(f"    Method:     {INTERNAL_REMOVAL_METHOD}")
        if INTERNAL_REMOVAL_METHOD == "RAYCAST":
            report.append(f"    Ray samples: {INTERNAL_RAY_SAMPLES}")
    report.append(f"  Bake vertex colors: {BAKE_VERTEX_COLORS}")
    report.append(f"  Preserve colors (remesh): {PRESERVE_COLORS_THROUGH_REMESH}")
    report.append(f"  Color edges:  {DETECT_COLOR_EDGES}")
    if DETECT_COLOR_EDGES:
        report.append(f"    When:       {DETECT_COLOR_EDGES_WHEN}")
        report.append(f"    Source:     {COLOR_SOURCE}")
        report.append(f"    Threshold:  {COLOR_EDGE_THRESHOLD}")
    report.append(f"  Target faces: {TARGET_FACES if TARGET_FACES > 0 else 'N/A (planar only)'}")
    report.append(f"  Planar angle: {PLANAR_ANGLE}°")
    report.append(f"  Sharp angle:  {SHARP_ANGLE}°")
    report.append(f"  Fix normals:  {FIX_NORMALS}")
    if FIX_NORMALS == "AUTO":
        report.append(f"    Threshold:  {FIX_NORMALS_THRESHOLD}% (fix if more inverted)")
    report.append(f"    Method:     {FIX_NORMALS_METHOD}")
    report.append(f"  Show progress: {SHOW_PROGRESS}")
    report.append(f"  Show timings: {SHOW_TIMINGS}")
    report.append("")

    total_start = time.time()

    # Get meshes to process
    meshes = get_target_meshes(report)

    for obj in meshes:
        report.append(f"\n{'='*50}")
        report.append(f"Processing: {obj.name}")
        report.append(f"{'='*50}")

        initial_faces = get_face_count(obj)
        initial_verts = get_vertex_count(obj)
        report.append(f"[Start] Initial: {initial_verts} verts, {initial_faces} faces")

        # Track reference for color preservation
        color_ref_obj = None

        # Step -1: Pre-cleanup (BEFORE anything else)
        if PRE_CLEANUP:
            report.append("\n[Pre-Cleanup] Fixing mesh issues before decimation...")
            with step_timer("Pre-cleanup"):
                pre_cleanup_mesh(obj, report)
            report.append(f"[Pre-Cleanup] Now: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")

        # Step 0a: Bake vertex colors and create reference (for both remesh and decimate)
        # Vertex colors get lost/averaged during topology changes, so we need to transfer back after
        vertex_colors_baked = False
        if BAKE_VERTEX_COLORS:
            report.append("\n[Step0a] Baking texture to vertex colors...")
            with step_timer("Baking vertex colors"):
                vertex_colors_baked = bake_texture_to_vertex_colors(obj, report)
                if vertex_colors_baked:
                    report.append("[Step0a] Vertex colors baked successfully!")
                    # Create reference copy to transfer colors back after decimation
                    color_ref_obj = create_color_reference_copy(obj, report)
                    report.append("[Step0a] Reference copy created for color preservation")
                else:
                    report.append("[Step0a] WARNING: Could not bake vertex colors.")

        # Step 0b: Color edge detection BEFORE decimate (if configured)
        # Note: Remesh destroys UVs, so we only detect before if not remeshing
        if DETECT_COLOR_EDGES and DETECT_COLOR_EDGES_WHEN == "BEFORE" and REMESH_MODE == "NONE":
            report.append("\n[Step0] Detecting color edges BEFORE decimation...")
            report.append(f"[Step0] This may take a while on {get_face_count(obj)} faces...")
            with step_timer("Color edge detection (BEFORE)"):
                detect_color_edges(obj, COLOR_EDGE_THRESHOLD, COLOR_SOURCE, report)

        # Step 1: Remesh (optional)
        if REMESH_MODE == "SHARP":
            with step_timer("Sharp remesh"):
                apply_sharp_remesh(obj, SHARP_OCTREE_DEPTH, SHARP_THRESHOLD, report)
            # Fix normals immediately after remesh (remesh doesn't guarantee consistent normals)
            with step_timer("Post-remesh normal fix"):
                fix_normals_after_remesh(obj, report)
        elif REMESH_MODE == "VOXEL":
            with step_timer("Voxel remesh"):
                apply_voxel_remesh(obj, VOXEL_SIZE, report)
            # Fix normals immediately after remesh
            with step_timer("Post-remesh normal fix"):
                fix_normals_after_remesh(obj, report)
        elif REMESH_MODE == "VOXEL_HIGH":
            with step_timer("Voxel high-res remesh"):
                apply_voxel_high_remesh(obj, VOXEL_HIGH_TARGET, report)
            # Fix normals immediately after remesh
            with step_timer("Post-remesh normal fix"):
                fix_normals_after_remesh(obj, report)
        elif REMESH_MODE == "QUAD":
            with step_timer("Quadriflow remesh"):
                try:
                    apply_quadriflow_remesh(obj, QUAD_TARGET_FACES, report)
                except Exception as e:
                    report.append(f"[Quadriflow] Failed: {e}")
                    report.append("[Quadriflow] Falling back to voxel remesh...")
                    apply_voxel_remesh(obj, VOXEL_SIZE, report)
            # Fix normals immediately after remesh
            with step_timer("Post-remesh normal fix"):
                fix_normals_after_remesh(obj, report)

        # Step 1b: Transfer colors back from reference after remesh (ONLY if remesh was used)
        if color_ref_obj and REMESH_MODE != "NONE":
            report.append("\n[Step1b] Restoring colors after remesh...")
            transfer_vertex_colors_from_reference(obj, color_ref_obj, report)

            # Create material to display vertex colors
            create_vertex_color_material(obj, report)

            # Now detect color edges from the transferred vertex colors
            if DETECT_COLOR_EDGES:
                report.append("[Step1c] Detecting color edges from recovered vertex colors...")
                # Force vertex color source since we just transferred them
                with step_timer("Color edge detection (post-remesh)"):
                    detect_color_edges(obj, COLOR_EDGE_THRESHOLD, "VERTEX_COLOR", report)

        # Step 2: Fill holes
        if FILL_HOLES:
            with step_timer("Fill holes"):
                fill_holes_simple(obj, report)

        # Step 3: Remove internal geometry
        if REMOVE_INTERNAL:
            with step_timer("Remove internal geometry"):
                if INTERNAL_REMOVAL_METHOD == "RAYCAST":
                    remove_internal_geometry(obj, INTERNAL_RAY_SAMPLES, report)
                else:  # "SIMPLE"
                    remove_internal_simple(obj, report)

        # Step 4: Planar decimate (merge coplanar faces)
        if PLANAR_ANGLE > 0:
            with step_timer("Planar decimate"):
                apply_planar_decimate(obj, PLANAR_ANGLE, report)

        # Step 5: Triangulate n-gons BEFORE collapse (planar creates n-gons, collapse needs accurate count)
        with step_timer("Triangulate n-gons"):
            triangulate_ngons(obj, report)
        report.append(f"[Triangulate] Now: {get_face_count(obj)} faces (ready for collapse)")

        # Step 6: Collapse decimate to target count
        if TARGET_FACES > 0:
            with step_timer("Collapse decimate"):
                apply_collapse_decimate(obj, TARGET_FACES, report)
        report.append(f"[Debug] After collapse: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")

        # Step 7: Clean up
        if REMOVE_DOUBLES or MERGE_DISTANCE > 0:
            with step_timer("Cleanup mesh"):
                cleanup_mesh(obj, report)
        report.append(f"[Debug] After cleanup: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")

        # Step 7b: Transfer vertex colors back from reference (for non-remesh mode)
        # If remesh was used, colors were already transferred in Step 1b
        if color_ref_obj and vertex_colors_baked and REMESH_MODE == "NONE":
            report.append("\n[Step7b] Transferring vertex colors from reference...")
            report.append(f"[Debug] Before transfer: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")
            with step_timer("Transfer vertex colors"):
                transfer_vertex_colors_from_reference(obj, color_ref_obj, report)
            report.append(f"[Debug] After transfer: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")
            # Create material to display vertex colors
            create_vertex_color_material(obj, report)

        report.append(f"[Debug] Before fix normals: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")

        # Step 8: Fix normals (optional based on FIX_NORMALS setting)
        # Determine if we should fix normals
        should_fix_normals = False
        if FIX_NORMALS == "AUTO":
            # AUTO: Check orientation and fix if inverted percentage exceeds threshold
            out_count, in_count, total = check_normal_orientation(obj)
            if total > 0:
                in_pct = (in_count / total) * 100
                if in_pct > FIX_NORMALS_THRESHOLD:
                    should_fix_normals = True
                    report.append(f"\n[Step8] AUTO: {in_pct:.1f}% faces inverted (>{FIX_NORMALS_THRESHOLD}%), fixing normals...")
                else:
                    report.append(f"\n[Step8] AUTO: Normals OK ({in_pct:.1f}% inverted, threshold {FIX_NORMALS_THRESHOLD}%), skipping fix")
        elif FIX_NORMALS is True:
            should_fix_normals = True
            report.append("\n[Step8] Fixing normals (FIX_NORMALS=True)...")
        else:  # FIX_NORMALS is False
            report.append("\n[Step8] Skipping normal fix (FIX_NORMALS=False)")

        if should_fix_normals:
            with step_timer("Fix normals"):
                report.append(f"[Step8] Using method: {FIX_NORMALS_METHOD}")
                if FIX_NORMALS_METHOD == "BLENDER":
                    fix_normals_blender(obj, report)
                elif FIX_NORMALS_METHOD == "DIRECTION":
                    fix_normals_by_direction(obj, report)
                else:  # "BOTH"
                    fix_normals_blender(obj, report)
                    out_after, in_after, total = check_normal_orientation(obj)
                    in_pct_after = (in_after / total) * 100 if total > 0 else 0
                    if in_pct_after > FIX_NORMALS_THRESHOLD:
                        report.append(f"[Step8] Blender method: still {in_pct_after:.1f}% inverted, trying direction fix...")
                        fix_normals_by_direction(obj, report)
        report.append(f"[Debug] After fix normals: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")

        # Step 9: Mark sharp edges (geometry-based)
        if SHARP_ANGLE > 0:
            with step_timer("Mark sharp edges"):
                mark_sharp_edges(obj, SHARP_ANGLE, report)

        # Step 9: Color edge detection AFTER decimate (much faster on low-poly)
        if DETECT_COLOR_EDGES and DETECT_COLOR_EDGES_WHEN == "AFTER" and REMESH_MODE == "NONE":
            report.append(f"\n[Step9] Detecting color edges AFTER decimation ({get_face_count(obj)} faces)...")
            with step_timer("Color edge detection (AFTER)"):
                detect_color_edges(obj, COLOR_EDGE_THRESHOLD, COLOR_SOURCE, report)

        # Step 10: Cleanup color reference
        if color_ref_obj:
            cleanup_reference_copy(color_ref_obj, report)

        # Final analysis
        with step_timer("Mesh analysis"):
            analyze_mesh(obj, report)

        # Verify normals are correct after all processing
        verify_normals(obj, report)

        final_faces = get_face_count(obj)
        final_verts = get_vertex_count(obj)
        total_reduction = ((initial_faces - final_faces) / initial_faces * 100) if initial_faces > 0 else 0
        report.append(f"[Done] {initial_verts} verts, {initial_faces} faces -> {final_verts} verts, {final_faces} faces")
        report.append(f"       ({total_reduction:.1f}% face reduction)")

    total_elapsed = time.time() - total_start

    report.append(f"\n{'='*50}")
    report.append("Decimate V1 complete!")
    report.append(f"Total time: {total_elapsed:.2f}s")
    report.append("")
    report.append("Next steps:")
    report.append("  1. Check mesh in viewport (sharp edges should show)")
    report.append("  2. Run Segmentation_v1.py for materials/UVs")
    report.append("  3. Run VertexColors_v1.py for painting")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log saved to: {LOG_TEXT_NAME} ---")
    print(f"--- Total time: {total_elapsed:.2f}s ---")


# Run
if __name__ == "__main__":
    main()
