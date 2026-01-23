"""
mesh_ops.colors - Vertex Color Operations

Functions for baking, transferring, and managing vertex colors.
"""

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .utils import (
    ensure_object_mode, ensure_edit_mode, get_face_count,
    log, ProgressTracker, step_timer
)

# ============================================================================
# COLOR ATTRIBUTE MANAGEMENT
# ============================================================================

def get_color_attribute(mesh, name=None, priority_names=None):
    """
    Find a color attribute on the mesh.

    Args:
        mesh: Blender mesh data
        name: Specific name to find
        priority_names: List of names to try in order

    Returns:
        Color attribute or None
    """
    if not hasattr(mesh, 'color_attributes') or not mesh.color_attributes:
        return None

    # Try specific name first
    if name:
        attr = mesh.color_attributes.get(name)
        if attr:
            return attr

    # Try priority names
    if priority_names:
        for pname in priority_names:
            attr = mesh.color_attributes.get(pname)
            if attr:
                return attr

    # Fall back to first available
    if len(mesh.color_attributes) > 0:
        return mesh.color_attributes[0]

    return None


def create_color_attribute(mesh, name="Color", domain='CORNER', data_type='BYTE_COLOR'):
    """
    Create or reset a color attribute on the mesh.

    Args:
        mesh: Blender mesh data
        name: Name for the color attribute
        domain: 'CORNER' (per face-corner) or 'POINT' (per vertex)
        data_type: 'BYTE_COLOR' (8-bit) or 'FLOAT_COLOR' (32-bit)

    Returns:
        The created color attribute
    """
    if not hasattr(mesh, 'color_attributes'):
        return None

    # Remove existing if present
    existing = mesh.color_attributes.get(name)
    if existing:
        mesh.color_attributes.remove(existing)

    # Create new
    attr = mesh.color_attributes.new(
        name=name,
        type=data_type,
        domain=domain
    )
    return attr


def set_active_color_attribute(mesh, name):
    """Set a color attribute as active for rendering and export."""
    if not hasattr(mesh, 'color_attributes'):
        return False

    attr = mesh.color_attributes.get(name)
    if not attr:
        return False

    mesh.color_attributes.active_color = attr
    idx = list(mesh.color_attributes).index(attr)
    if hasattr(mesh.color_attributes, 'render_color_index'):
        mesh.color_attributes.render_color_index = idx

    return True


def finalize_color_attribute(obj, source_name=None, target_name="Color", report=None):
    """
    Finalize color attribute for export.

    Renames source attribute to standard "Color" name and sets as active.

    Args:
        obj: Blender mesh object
        source_name: Source attribute name (None = auto-detect)
        target_name: Target name (default "Color")
        report: Optional report list

    Returns:
        True on success
    """
    mesh = obj.data

    if not hasattr(mesh, 'color_attributes') or not mesh.color_attributes:
        log("[Color Finalize] WARNING: No color attributes found", report)
        return False

    # Find source attribute
    source_attr = None
    if source_name:
        source_attr = mesh.color_attributes.get(source_name)
    else:
        # Auto-detect: prefer TransferredColors > BakedColors > first
        for name in ("TransferredColors", "BakedColors", "Color"):
            attr = mesh.color_attributes.get(name)
            if attr:
                source_attr = attr
                break

        if not source_attr and len(mesh.color_attributes) > 0:
            source_attr = mesh.color_attributes[0]

    if not source_attr:
        log("[Color Finalize] WARNING: No valid color attribute found", report)
        return False

    # If already named correctly, just set as active
    if source_attr.name == target_name:
        set_active_color_attribute(mesh, target_name)
        log(f"[Color Finalize] '{target_name}' already exists and set as active", report)
        return True

    # Create new attribute with target name and copy data
    old_name = source_attr.name
    new_attr = mesh.color_attributes.new(
        name=target_name,
        type=source_attr.data_type,
        domain=source_attr.domain
    )

    # Copy color data
    for i, src_color in enumerate(source_attr.data):
        new_attr.data[i].color = src_color.color

    # Set as active
    set_active_color_attribute(mesh, target_name)

    # Remove old attributes (keep only target)
    attrs_to_remove = [attr.name for attr in mesh.color_attributes if attr.name != target_name]
    for attr_name in attrs_to_remove:
        attr = mesh.color_attributes.get(attr_name)
        if attr:
            mesh.color_attributes.remove(attr)

    log(f"[Color Finalize] Renamed '{old_name}' -> '{target_name}'", report)
    return True


# ============================================================================
# TEXTURE TO VERTEX COLOR BAKING
# ============================================================================

def bake_texture_to_vertex_colors(obj, image=None, output_name="BakedColors", report=None):
    """
    Bake texture colors to vertex colors.

    Args:
        obj: Blender mesh object with UVs
        image: Blender image to sample (None = auto-detect from material)
        output_name: Name for output color attribute
        report: Optional report list

    Returns:
        True on success
    """
    ensure_object_mode()

    with step_timer("Baking texture to vertex colors"):
        bpy.context.view_layer.objects.active = obj
        mesh = obj.data

        # Create color attribute
        color_attr = create_color_attribute(mesh, output_name, 'CORNER', 'FLOAT_COLOR')
        if not color_attr:
            log("[Bake Colors] ERROR: Could not create color attribute", report)
            return False

        log(f"[Bake Colors] Created color attribute: {output_name}", report)

        # Get UV layer
        if not mesh.uv_layers.active:
            log("[Bake Colors] WARNING: No UV layer, cannot bake texture", report)
            return False

        uv_layer = mesh.uv_layers.active

        # Find image
        if image is None:
            # Try from material
            if obj.active_material and obj.active_material.node_tree:
                for node in obj.active_material.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        image = node.image
                        break

            # Try any image in blend file
            if image is None:
                for img in bpy.data.images:
                    if img.type == 'IMAGE' and img.size[0] > 0:
                        image = img
                        break

        if not image:
            log("[Bake Colors] WARNING: No texture image found", report)
            return False

        log(f"[Bake Colors] Using image: {image.name} ({image.size[0]}x{image.size[1]})", report)

        # Cache pixel data
        width, height = image.size
        pixels = list(image.pixels[:])

        # Get UV data
        uv_data = uv_layer.data
        total_loops = len(mesh.loops)
        progress = ProgressTracker(total_loops, "Baking vertex colors")

        # Sample and assign colors
        for loop_idx in range(total_loops):
            uv = uv_data[loop_idx].uv

            # Sample image at UV
            px = int(uv.x % 1.0 * width) % width
            py = int(uv.y % 1.0 * height) % height
            idx = (py * width + px) * 4

            if idx + 3 < len(pixels):
                color = (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])
            else:
                color = (1.0, 0.0, 1.0, 1.0)  # Magenta for missing

            color_attr.data[loop_idx].color = color

            if loop_idx % 50000 == 0:
                progress.update(loop_idx)

        progress.finish()
        mesh.update()

        log(f"[Bake Colors] Baked {total_loops} vertex colors", report)
        return True


# ============================================================================
# VERTEX COLOR TRANSFER
# ============================================================================

def transfer_vertex_colors(source_obj, target_obj, output_name="Color", report=None):
    """
    Transfer vertex colors from source mesh to target mesh using BVH lookup.

    For each face on target, finds nearest point on source and copies color.
    Uses face-based transfer for flat shading (all loops get same color).

    Args:
        source_obj: Source mesh with vertex colors
        target_obj: Target mesh to receive colors
        output_name: Name for output color attribute
        report: Optional report list

    Returns:
        True on success
    """
    ensure_object_mode()

    with step_timer("Transferring vertex colors"):
        source_mesh = source_obj.data
        target_mesh = target_obj.data

        log(f"[Color Transfer] Source: {source_obj.name} ({get_face_count(source_obj)} faces)", report)
        log(f"[Color Transfer] Target: {target_obj.name} ({get_face_count(target_obj)} faces)", report)

        # Find source color attribute
        source_color_attr = get_color_attribute(
            source_mesh,
            priority_names=["Color", "BakedColors", "TransferredColors"]
        )

        if not source_color_attr:
            log("[Color Transfer] WARNING: No color attribute found on source mesh", report)
            return False

        log(f"[Color Transfer] Using source color layer: {source_color_attr.name}", report)

        # Build face color lookup from source
        source_face_colors = []
        unique_colors = set()

        for poly in source_mesh.polygons:
            first_loop_idx = poly.loop_start
            if first_loop_idx < len(source_color_attr.data):
                col = source_color_attr.data[first_loop_idx].color
                color = (col[0], col[1], col[2], col[3])
                source_face_colors.append(color)
                unique_colors.add((round(col[0], 2), round(col[1], 2), round(col[2], 2)))
            else:
                source_face_colors.append((1.0, 0.0, 1.0, 1.0))

        log(f"[Color Transfer] Cached {len(source_face_colors)} face colors", report)
        log(f"[Color Transfer] Found {len(unique_colors)} unique colors", report)

        # Build BVH tree from source mesh (world space)
        source_matrix = source_obj.matrix_world
        vertices = [(source_matrix @ v.co) for v in source_mesh.vertices]
        polygons = [tuple(p.vertices) for p in source_mesh.polygons]

        bvh = BVHTree.FromPolygons(vertices, polygons)

        # Create target color attribute
        target_color_attr = create_color_attribute(target_mesh, output_name, 'CORNER', 'BYTE_COLOR')
        if not target_color_attr:
            log("[Color Transfer] ERROR: Failed to create color attribute on target", report)
            return False

        # Transfer colors
        target_matrix = target_obj.matrix_world
        total_faces = len(target_mesh.polygons)
        progress = ProgressTracker(total_faces, "Transferring colors")

        colors_found = 0
        colors_missing = 0

        for i, poly in enumerate(target_mesh.polygons):
            # Get face center in world space
            face_center_world = target_matrix @ poly.center

            # Find closest point on source mesh
            location, normal, face_idx, distance = bvh.find_nearest(face_center_world)

            if face_idx is not None and face_idx < len(source_face_colors):
                color = source_face_colors[face_idx]
                colors_found += 1
            else:
                color = (1.0, 0.0, 1.0, 1.0)  # Magenta for missing
                colors_missing += 1

            # Apply same color to all loops of target face
            for loop_idx in poly.loop_indices:
                target_color_attr.data[loop_idx].color = color

            if i % 5000 == 0:
                progress.update(i)

        progress.finish()
        target_mesh.update()
        set_active_color_attribute(target_mesh, output_name)

        log(f"[Color Transfer] Colors found: {colors_found}", report)
        log(f"[Color Transfer] Colors missing: {colors_missing}", report)

        return True


def create_color_reference_copy(obj, report=None):
    """
    Create a hidden copy of the mesh to use as color reference.

    Returns:
        The reference object
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

        log(f"[Color Ref] Created reference copy: {ref_obj.name}", report)
        return ref_obj


def cleanup_color_reference(ref_obj, report=None):
    """Remove the temporary color reference copy."""
    if ref_obj:
        bpy.data.objects.remove(ref_obj, do_unlink=True)
        log("[Cleanup] Removed color reference copy", report)


# ============================================================================
# COLOR EDGE DETECTION
# ============================================================================

def detect_color_edges(obj, threshold=0.15, source="TEXTURE",
                       mark_sharp=True, mark_seam=True, report=None):
    """
    Detect edges where color changes significantly and mark them.

    Args:
        obj: Blender mesh object
        threshold: Color difference threshold (0-1)
        source: "TEXTURE", "VERTEX_COLOR", or "MATERIAL"
        mark_sharp: Mark detected edges as sharp
        mark_seam: Mark detected edges as seams
        report: Optional report list

    Returns:
        Number of edges marked
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
    pixels_cache = None
    width = height = None

    if source == "TEXTURE":
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            log("[Color Edges] WARNING: No UV layer found", report)
            bpy.ops.object.mode_set(mode='OBJECT')
            return 0

        # Find texture from material
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
            log("[Color Edges] WARNING: No texture image found", report)
            bpy.ops.object.mode_set(mode='OBJECT')
            return 0

        # Cache pixels
        width, height = image.size
        pixels_cache = list(image.pixels[:])
        log(f"[Color Edges] Using image: {image.name}", report)

    elif source == "VERTEX_COLOR":
        color_layer = bm.loops.layers.color.active
        if not color_layer:
            for layer in bm.loops.layers.color.values():
                color_layer = layer
                break

        if not color_layer:
            log("[Color Edges] WARNING: No vertex color layer found", report)
            bpy.ops.object.mode_set(mode='OBJECT')
            return 0

    # Calculate face colors
    face_colors = {}
    total_faces = len(bm.faces)
    progress = ProgressTracker(total_faces, "Sampling face colors")

    for i, face in enumerate(bm.faces):
        if source == "TEXTURE":
            face_colors[face.index] = _get_face_color_from_texture(
                face, uv_layer, pixels_cache, width, height
            )
        elif source == "VERTEX_COLOR":
            face_colors[face.index] = _get_face_color_from_vertex_colors(face, color_layer)
        elif source == "MATERIAL":
            face_colors[face.index] = Vector((face.material_index, 0, 0))

        if i % 10000 == 0:
            progress.update(i)

    progress.finish("Face color sampling")

    # Find edges where adjacent faces have different colors
    color_edges = []

    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue

        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        c1 = face_colors.get(f1.index)
        c2 = face_colors.get(f2.index)

        if source == "MATERIAL":
            if c1 is not None and c2 is not None and c1.x != c2.x:
                color_edges.append(edge)
        else:
            diff = _color_difference(c1, c2)
            if diff > threshold:
                color_edges.append(edge)

    # Mark edges
    marked_count = 0
    for edge in color_edges:
        if mark_sharp:
            edge.smooth = False
        if mark_seam:
            edge.seam = True
        marked_count += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Color Edges] Detected {marked_count} color boundary edges (threshold: {threshold})", report)
    return marked_count


def _get_face_color_from_texture(face, uv_layer, pixels_cache, width, height):
    """Sample texture color at face center UV."""
    if not uv_layer or not pixels_cache:
        return None

    # Average UV of face
    uv_sum_x = uv_sum_y = 0.0
    loop_count = len(face.loops)
    for loop in face.loops:
        uv = loop[uv_layer].uv
        uv_sum_x += uv.x
        uv_sum_y += uv.y

    uv_avg_x = uv_sum_x / loop_count
    uv_avg_y = uv_sum_y / loop_count

    # Sample image
    px = int(uv_avg_x % 1.0 * width) % width
    py = int(uv_avg_y % 1.0 * height) % height
    idx = (py * width + px) * 4

    if idx + 3 < len(pixels_cache):
        return Vector((pixels_cache[idx], pixels_cache[idx + 1], pixels_cache[idx + 2]))
    return None


def _get_face_color_from_vertex_colors(face, color_layer):
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


def _color_difference(c1, c2):
    """Calculate color difference (0-1 range)."""
    if c1 is None or c2 is None:
        return 0
    return (c1 - c2).length / 1.732  # Normalize by max RGB distance


# ============================================================================
# DEBUG MATERIAL
# ============================================================================

def create_vertex_color_material(obj, layer_name="Color", mat_name="M_VertexColors", report=None):
    """
    Create a material that displays vertex colors.

    Args:
        obj: Blender mesh object
        layer_name: Name of color attribute to use
        mat_name: Name for the material
        report: Optional report list

    Returns:
        The created material
    """
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
    output.location = (300, 0)

    # Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # Vertex color node
    vert_color = nodes.new('ShaderNodeVertexColor')
    vert_color.location = (-300, 0)
    vert_color.layer_name = layer_name

    # Connect
    links.new(vert_color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Assign to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    log(f"[Material] Created '{mat_name}' with color layer '{layer_name}'", report)
    return mat
