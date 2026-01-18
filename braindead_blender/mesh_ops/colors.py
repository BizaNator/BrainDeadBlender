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


def create_color_attribute(mesh, name="Col", domain='CORNER', data_type='BYTE_COLOR'):
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


def finalize_color_attribute(obj, source_name=None, target_name="Col", report=None):
    """
    Finalize color attribute for export.

    Renames source attribute to standard "Col" name and sets as active.

    Args:
        obj: Blender mesh object
        source_name: Source attribute name (None = auto-detect)
        target_name: Target name (default "Col" for Unreal/FBX)
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
        for name in ("TransferredColors", "BakedColors", "Col"):
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

def transfer_vertex_colors(source_obj, target_obj, output_name="Col", mode="FACE", report=None):
    """
    Transfer vertex colors from source mesh to target mesh using BVH lookup.

    Args:
        source_obj: Source mesh with vertex colors
        target_obj: Target mesh to receive colors
        output_name: Name for output color attribute
        mode: Transfer mode:
            - "FACE": Each face gets ONE solid color (no blending)
            - "VERTEX": Each vertex gets a color, blended across faces
            - "CORNER": Each face-corner sampled independently
        report: Optional report list

    Returns:
        True on success
    """
    ensure_object_mode()

    with step_timer(f"Transferring vertex colors (mode={mode})"):
        source_mesh = source_obj.data
        target_mesh = target_obj.data

        log(f"[Color Transfer] Source: {source_obj.name} ({get_face_count(source_obj)} faces)", report)
        log(f"[Color Transfer] Target: {target_obj.name} ({get_face_count(target_obj)} faces)", report)
        log(f"[Color Transfer] Mode: {mode}", report)

        # Find source color attribute
        source_color_attr = get_color_attribute(
            source_mesh,
            priority_names=["Col", "BakedColors", "TransferredColors"]
        )

        if not source_color_attr:
            log("[Color Transfer] WARNING: No color attribute found on source mesh", report)
            return False

        log(f"[Color Transfer] Using source color layer: {source_color_attr.name}", report)

        # Build BVH tree from source mesh (world space)
        source_matrix = source_obj.matrix_world
        vertices = [(source_matrix @ v.co) for v in source_mesh.vertices]
        polygons = [tuple(p.vertices) for p in source_mesh.polygons]
        bvh = BVHTree.FromPolygons(vertices, polygons)

        # Build source color lookup based on mode
        if mode == "FACE":
            # Cache one DOMINANT color per face from source (handles source variation)
            source_face_colors = []
            for poly in source_mesh.polygons:
                loop_indices = list(poly.loop_indices)
                if not loop_indices:
                    source_face_colors.append((1.0, 0.0, 1.0, 1.0))
                    continue

                # Collect all loop colors for this face
                face_loop_colors = []
                for loop_idx in loop_indices:
                    if loop_idx < len(source_color_attr.data):
                        col = source_color_attr.data[loop_idx].color
                        face_loop_colors.append((col[0], col[1], col[2], col[3]))

                if not face_loop_colors:
                    source_face_colors.append((1.0, 0.0, 1.0, 1.0))
                    continue

                # Find dominant color (most common, rounded for comparison)
                color_counts = {}
                for col in face_loop_colors:
                    key = (round(col[0], 2), round(col[1], 2), round(col[2], 2))
                    if key not in color_counts:
                        color_counts[key] = {'count': 0, 'color': col}
                    color_counts[key]['count'] += 1

                dominant = max(color_counts.values(), key=lambda x: x['count'])
                source_face_colors.append(dominant['color'])
        else:
            # For VERTEX/CORNER modes, we need per-loop colors
            source_loop_colors = []
            for loop_idx in range(len(source_mesh.loops)):
                if loop_idx < len(source_color_attr.data):
                    col = source_color_attr.data[loop_idx].color
                    source_loop_colors.append((col[0], col[1], col[2], col[3]))
                else:
                    source_loop_colors.append((1.0, 0.0, 1.0, 1.0))

        # Create target color attribute
        target_color_attr = create_color_attribute(target_mesh, output_name, 'CORNER', 'BYTE_COLOR')
        if not target_color_attr:
            log("[Color Transfer] ERROR: Failed to create color attribute on target", report)
            return False

        # Transfer colors based on mode
        target_matrix = target_obj.matrix_world
        total_faces = len(target_mesh.polygons)
        progress = ProgressTracker(total_faces, "Transferring colors")

        colors_found = 0
        colors_missing = 0

        if mode == "FACE":
            # Each face gets ONE solid color from nearest source face
            for i, poly in enumerate(target_mesh.polygons):
                face_center_world = target_matrix @ poly.center
                location, normal, face_idx, distance = bvh.find_nearest(face_center_world)

                if face_idx is not None and face_idx < len(source_face_colors):
                    color = source_face_colors[face_idx]
                    colors_found += 1
                else:
                    color = (1.0, 0.0, 1.0, 1.0)
                    colors_missing += 1

                # Apply same color to ALL loops of this face
                for loop_idx in poly.loop_indices:
                    target_color_attr.data[loop_idx].color = color

                if i % 5000 == 0:
                    progress.update(i)

        elif mode == "VERTEX":
            # Each vertex gets a color, shared across all faces using it
            vertex_colors = {}

            for i, poly in enumerate(target_mesh.polygons):
                for loop_idx in poly.loop_indices:
                    vert_idx = target_mesh.loops[loop_idx].vertex_index

                    if vert_idx not in vertex_colors:
                        # Sample color for this vertex
                        vert_world = target_matrix @ target_mesh.vertices[vert_idx].co
                        location, normal, face_idx, distance = bvh.find_nearest(vert_world)

                        if face_idx is not None:
                            # Get color from nearest point on source
                            src_poly = source_mesh.polygons[face_idx]
                            src_loop_idx = src_poly.loop_start
                            color = source_loop_colors[src_loop_idx]
                            colors_found += 1
                        else:
                            color = (1.0, 0.0, 1.0, 1.0)
                            colors_missing += 1

                        vertex_colors[vert_idx] = color

                    target_color_attr.data[loop_idx].color = vertex_colors[vert_idx]

                if i % 5000 == 0:
                    progress.update(i)

        elif mode == "CORNER":
            # Each face-corner sampled independently (can have variation within face)
            for i, poly in enumerate(target_mesh.polygons):
                for loop_idx in poly.loop_indices:
                    vert_idx = target_mesh.loops[loop_idx].vertex_index
                    vert_world = target_matrix @ target_mesh.vertices[vert_idx].co

                    location, normal, face_idx, distance = bvh.find_nearest(vert_world)

                    if face_idx is not None:
                        # Find closest loop on source face
                        src_poly = source_mesh.polygons[face_idx]
                        best_loop = src_poly.loop_start
                        best_dist = float('inf')

                        for src_loop_idx in src_poly.loop_indices:
                            src_vert_idx = source_mesh.loops[src_loop_idx].vertex_index
                            src_vert_world = source_matrix @ source_mesh.vertices[src_vert_idx].co
                            dist = (vert_world - src_vert_world).length
                            if dist < best_dist:
                                best_dist = dist
                                best_loop = src_loop_idx

                        color = source_loop_colors[best_loop]
                        colors_found += 1
                    else:
                        color = (1.0, 0.0, 1.0, 1.0)
                        colors_missing += 1

                    target_color_attr.data[loop_idx].color = color

                if i % 5000 == 0:
                    progress.update(i)

        progress.finish()
        target_mesh.update()
        set_active_color_attribute(target_mesh, output_name)

        log(f"[Color Transfer] Colors found: {colors_found}", report)
        log(f"[Color Transfer] Colors missing: {colors_missing}", report)

        return True


def apply_flat_shading(obj, report=None):
    """
    Apply flat shading to mesh - required for solid face colors to display correctly.

    Args:
        obj: Blender mesh object
        report: Optional report list
    """
    ensure_object_mode()

    mesh = obj.data

    # Explicitly set EVERY face to flat shading (use_smooth = False)
    flat_count = 0
    for poly in mesh.polygons:
        if poly.use_smooth:
            poly.use_smooth = False
            flat_count += 1

    # Disable auto-smooth if present (Blender < 4.1)
    if hasattr(mesh, 'use_auto_smooth'):
        mesh.use_auto_smooth = False

    # Clear any custom split normals that might interfere
    if mesh.has_custom_normals:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.mesh.customdata_custom_splitnormals_clear()

    mesh.update()

    log(f"[Flat Shading] Set {flat_count} faces to flat (total: {len(mesh.polygons)})", report)


# ============================================================================
# COLOR SOLIDIFY / SMOOTH OPERATIONS
# ============================================================================

def solidify_face_colors(obj, color_attr_name=None, method="DOMINANT", report=None):
    """
    Convert vertex colors to solid face colors (no blending within faces).

    Args:
        obj: Blender mesh object
        color_attr_name: Color attribute to modify (None = active)
        method: How to determine face color:
            - "DOMINANT": Most common color among face vertices
            - "AVERAGE": Average of all vertex colors
            - "FIRST": Use first vertex color (fastest)
        report: Optional report list

    Returns:
        Number of faces processed
    """
    ensure_object_mode()

    mesh = obj.data
    color_attr = get_color_attribute(mesh, color_attr_name)

    if not color_attr:
        log("[Solidify] ERROR: No color attribute found", report)
        return 0

    log(f"[Solidify] Processing '{color_attr.name}' with method={method}", report)

    total_faces = len(mesh.polygons)
    progress = ProgressTracker(total_faces, "Solidifying colors")

    for i, poly in enumerate(mesh.polygons):
        loop_indices = list(poly.loop_indices)

        if not loop_indices:
            continue

        if method == "FIRST":
            # Use first vertex color
            color = color_attr.data[loop_indices[0]].color[:]

        elif method == "AVERAGE":
            # Average all vertex colors
            r = g = b = a = 0.0
            count = len(loop_indices)
            for loop_idx in loop_indices:
                col = color_attr.data[loop_idx].color
                r += col[0]
                g += col[1]
                b += col[2]
                a += col[3]
            color = (r / count, g / count, b / count, a / count)

        elif method == "DOMINANT":
            # Find most common color (rounded for comparison)
            color_counts = {}
            for loop_idx in loop_indices:
                col = color_attr.data[loop_idx].color
                # Round for comparison
                key = (round(col[0], 2), round(col[1], 2), round(col[2], 2))
                if key not in color_counts:
                    color_counts[key] = {'count': 0, 'color': (col[0], col[1], col[2], col[3])}
                color_counts[key]['count'] += 1

            # Get most common
            dominant = max(color_counts.values(), key=lambda x: x['count'])
            color = dominant['color']

        else:
            color = color_attr.data[loop_indices[0]].color[:]

        # Apply same color to all loops of face
        for loop_idx in loop_indices:
            color_attr.data[loop_idx].color = color

        if i % 5000 == 0:
            progress.update(i)

    progress.finish()
    mesh.update()

    log(f"[Solidify] Processed {total_faces} faces", report)
    return total_faces


def smooth_vertex_colors(obj, color_attr_name=None, iterations=1, report=None):
    """
    Smooth/blend vertex colors across the mesh.

    For each vertex, averages colors from all faces using that vertex.

    Args:
        obj: Blender mesh object
        color_attr_name: Color attribute to modify (None = active)
        iterations: Number of smoothing passes
        report: Optional report list

    Returns:
        Number of vertices processed
    """
    ensure_object_mode()

    mesh = obj.data
    color_attr = get_color_attribute(mesh, color_attr_name)

    if not color_attr:
        log("[Smooth] ERROR: No color attribute found", report)
        return 0

    log(f"[Smooth] Processing '{color_attr.name}' with {iterations} iterations", report)

    for iteration in range(iterations):
        # Build vertex -> loops mapping
        vert_to_loops = {}
        for loop_idx, loop in enumerate(mesh.loops):
            vert_idx = loop.vertex_index
            if vert_idx not in vert_to_loops:
                vert_to_loops[vert_idx] = []
            vert_to_loops[vert_idx].append(loop_idx)

        # Calculate averaged color for each vertex
        vertex_colors = {}
        for vert_idx, loop_indices in vert_to_loops.items():
            r = g = b = a = 0.0
            count = len(loop_indices)
            for loop_idx in loop_indices:
                col = color_attr.data[loop_idx].color
                r += col[0]
                g += col[1]
                b += col[2]
                a += col[3]
            vertex_colors[vert_idx] = (r / count, g / count, b / count, a / count)

        # Apply averaged colors back to all loops
        for vert_idx, color in vertex_colors.items():
            for loop_idx in vert_to_loops[vert_idx]:
                color_attr.data[loop_idx].color = color

        log(f"[Smooth] Iteration {iteration + 1}/{iterations} complete", report)

    mesh.update()

    log(f"[Smooth] Processed {len(vertex_colors)} vertices", report)
    return len(vertex_colors)


# ============================================================================
# EDIT MODE OPERATIONS (Selected faces only)
# ============================================================================

def solidify_selected_faces(obj, color_attr_name=None, method="DOMINANT", report=None):
    """
    Solidify colors only on selected faces (edit mode).

    Args:
        obj: Blender mesh object (must be in edit mode)
        color_attr_name: Color attribute to modify (None = active)
        method: DOMINANT, AVERAGE, or FIRST
        report: Optional report list

    Returns:
        Number of faces processed
    """
    if obj.mode != 'EDIT':
        log("[Solidify Selected] ERROR: Must be in edit mode", report)
        return 0

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Get color layer
    color_layer = bm.loops.layers.color.active
    if color_attr_name:
        color_layer = bm.loops.layers.color.get(color_attr_name)
    if not color_layer:
        # Try to get any color layer
        if bm.loops.layers.color:
            color_layer = bm.loops.layers.color[0]

    if not color_layer:
        log("[Solidify Selected] ERROR: No color layer found", report)
        return 0

    # Process only selected faces
    processed = 0
    for face in bm.faces:
        if not face.select:
            continue

        loops = list(face.loops)
        if not loops:
            continue

        if method == "FIRST":
            color = loops[0][color_layer][:]

        elif method == "AVERAGE":
            r = g = b = a = 0.0
            count = len(loops)
            for loop in loops:
                col = loop[color_layer]
                r += col[0]
                g += col[1]
                b += col[2]
                a += col[3]
            color = (r / count, g / count, b / count, a / count)

        elif method == "DOMINANT":
            color_counts = {}
            for loop in loops:
                col = loop[color_layer]
                key = (round(col[0], 2), round(col[1], 2), round(col[2], 2))
                if key not in color_counts:
                    color_counts[key] = {'count': 0, 'color': (col[0], col[1], col[2], col[3])}
                color_counts[key]['count'] += 1
            dominant = max(color_counts.values(), key=lambda x: x['count'])
            color = dominant['color']
        else:
            color = loops[0][color_layer][:]

        # Apply to all loops of this face
        for loop in loops:
            loop[color_layer] = color

        processed += 1

    bmesh.update_edit_mesh(obj.data)

    log(f"[Solidify Selected] Processed {processed} selected faces", report)
    return processed


def smooth_selected_faces(obj, color_attr_name=None, iterations=1, report=None):
    """
    Smooth colors only on selected faces (edit mode).

    Args:
        obj: Blender mesh object (must be in edit mode)
        color_attr_name: Color attribute to modify (None = active)
        iterations: Number of smoothing passes
        report: Optional report list

    Returns:
        Number of vertices processed
    """
    if obj.mode != 'EDIT':
        log("[Smooth Selected] ERROR: Must be in edit mode", report)
        return 0

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Get color layer
    color_layer = bm.loops.layers.color.active
    if color_attr_name:
        color_layer = bm.loops.layers.color.get(color_attr_name)
    if not color_layer and bm.loops.layers.color:
        color_layer = bm.loops.layers.color[0]

    if not color_layer:
        log("[Smooth Selected] ERROR: No color layer found", report)
        return 0

    # Get selected face indices
    selected_faces = [f for f in bm.faces if f.select]
    if not selected_faces:
        log("[Smooth Selected] No faces selected", report)
        return 0

    # Get vertices that belong to selected faces
    selected_verts = set()
    for face in selected_faces:
        for vert in face.verts:
            selected_verts.add(vert.index)

    for iteration in range(iterations):
        # Build vertex -> loops mapping (only for selected vertices)
        vert_to_loops = {}
        for face in selected_faces:
            for loop in face.loops:
                vert_idx = loop.vert.index
                if vert_idx not in vert_to_loops:
                    vert_to_loops[vert_idx] = []
                vert_to_loops[vert_idx].append(loop)

        # Calculate averaged color for each vertex
        vertex_colors = {}
        for vert_idx, loops in vert_to_loops.items():
            r = g = b = a = 0.0
            count = len(loops)
            for loop in loops:
                col = loop[color_layer]
                r += col[0]
                g += col[1]
                b += col[2]
                a += col[3]
            vertex_colors[vert_idx] = (r / count, g / count, b / count, a / count)

        # Apply averaged colors back
        for vert_idx, color in vertex_colors.items():
            for loop in vert_to_loops[vert_idx]:
                loop[color_layer] = color

    bmesh.update_edit_mesh(obj.data)

    log(f"[Smooth Selected] Processed {len(vertex_colors)} vertices on {len(selected_faces)} faces", report)
    return len(vertex_colors)


def apply_flat_shading_selected(obj, report=None):
    """
    Apply flat shading only to selected faces (edit mode).

    Args:
        obj: Blender mesh object (must be in edit mode)
        report: Optional report list

    Returns:
        Number of faces changed
    """
    if obj.mode != 'EDIT':
        log("[Flat Selected] ERROR: Must be in edit mode", report)
        return 0

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    changed = 0
    for face in bm.faces:
        if face.select and face.smooth:
            face.smooth = False
            changed += 1

    bmesh.update_edit_mesh(obj.data)

    log(f"[Flat Selected] Set {changed} selected faces to flat shading", report)
    return changed


def apply_smooth_shading_selected(obj, report=None):
    """
    Apply smooth shading only to selected faces (edit mode).

    Args:
        obj: Blender mesh object (must be in edit mode)
        report: Optional report list

    Returns:
        Number of faces changed
    """
    if obj.mode != 'EDIT':
        log("[Smooth Selected] ERROR: Must be in edit mode", report)
        return 0

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    changed = 0
    for face in bm.faces:
        if face.select and not face.smooth:
            face.smooth = True
            changed += 1

    bmesh.update_edit_mesh(obj.data)

    log(f"[Smooth Selected] Set {changed} selected faces to smooth shading", report)
    return changed


def paint_selected_faces(obj, color, color_attr_name=None, report=None):
    """
    Paint selected faces with a solid color (edit mode).

    Args:
        obj: Blender mesh object (must be in edit mode)
        color: RGBA color tuple (r, g, b, a) with values 0-1
        color_attr_name: Color attribute to modify (None = active)
        report: Optional report list

    Returns:
        Number of faces painted
    """
    if obj.mode != 'EDIT':
        log("[Paint] ERROR: Must be in edit mode", report)
        return 0

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Get color layer
    color_layer = bm.loops.layers.color.active
    if color_attr_name:
        color_layer = bm.loops.layers.color.get(color_attr_name)
    if not color_layer:
        if bm.loops.layers.color:
            color_layer = bm.loops.layers.color[0]

    if not color_layer:
        log("[Paint] ERROR: No color layer found", report)
        return 0

    # Ensure color is a 4-tuple
    if len(color) == 3:
        color = (color[0], color[1], color[2], 1.0)

    # Paint selected faces
    painted = 0
    for face in bm.faces:
        if face.select:
            for loop in face.loops:
                loop[color_layer] = color
            painted += 1

    bmesh.update_edit_mesh(obj.data)

    log(f"[Paint] Painted {painted} faces with color {color[:3]}", report)
    return painted


def sample_face_color(obj, report=None):
    """
    Sample color from the active/selected face (edit mode).

    Args:
        obj: Blender mesh object (must be in edit mode)
        report: Optional report list

    Returns:
        RGBA color tuple or None if no face selected
    """
    if obj.mode != 'EDIT':
        log("[Sample] ERROR: Must be in edit mode", report)
        return None

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Get color layer
    color_layer = bm.loops.layers.color.active
    if not color_layer and bm.loops.layers.color:
        color_layer = bm.loops.layers.color[0]

    if not color_layer:
        log("[Sample] ERROR: No color layer found", report)
        return None

    # Find active face or first selected face
    active_face = bm.faces.active
    if not active_face or not active_face.select:
        # Find first selected face
        for face in bm.faces:
            if face.select:
                active_face = face
                break

    if not active_face:
        log("[Sample] ERROR: No face selected", report)
        return None

    # Get color from first loop of the face
    color = active_face.loops[0][color_layer][:]
    log(f"[Sample] Sampled color: ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}, {color[3]:.3f})", report)

    return tuple(color)


def convert_color_domain(obj, target_domain="CORNER", color_attr_name=None, report=None):
    """
    Convert color attribute between domains (CORNER vs POINT).

    Args:
        obj: Blender mesh object
        target_domain: "CORNER" (per face-corner) or "POINT" (per vertex)
        color_attr_name: Color attribute to convert (None = active)
        report: Optional report list

    Returns:
        True on success
    """
    ensure_object_mode()

    mesh = obj.data

    # Find source color attribute
    source_attr = get_color_attribute(mesh, color_attr_name)
    if not source_attr:
        log("[Convert Domain] ERROR: No color attribute found", report)
        return False

    source_name = source_attr.name
    source_domain = source_attr.domain
    source_type = source_attr.data_type

    if source_domain == target_domain:
        log(f"[Convert Domain] Already in {target_domain} domain", report)
        return True

    log(f"[Convert Domain] Converting '{source_name}' from {source_domain} to {target_domain}", report)

    # Cache colors based on source domain
    if source_domain == 'CORNER':
        # CORNER -> POINT: Average colors per vertex
        vert_colors = {}
        vert_counts = {}

        for loop_idx, loop in enumerate(mesh.loops):
            vert_idx = loop.vertex_index
            col = source_attr.data[loop_idx].color

            if vert_idx not in vert_colors:
                vert_colors[vert_idx] = [0.0, 0.0, 0.0, 0.0]
                vert_counts[vert_idx] = 0

            vert_colors[vert_idx][0] += col[0]
            vert_colors[vert_idx][1] += col[1]
            vert_colors[vert_idx][2] += col[2]
            vert_colors[vert_idx][3] += col[3]
            vert_counts[vert_idx] += 1

        # Average
        for vert_idx in vert_colors:
            count = vert_counts[vert_idx]
            vert_colors[vert_idx] = tuple(c / count for c in vert_colors[vert_idx])

    else:
        # POINT -> CORNER: Copy vertex color to all corners
        vert_colors = {}
        for vert_idx in range(len(mesh.vertices)):
            if vert_idx < len(source_attr.data):
                col = source_attr.data[vert_idx].color
                vert_colors[vert_idx] = (col[0], col[1], col[2], col[3])
            else:
                vert_colors[vert_idx] = (1.0, 1.0, 1.0, 1.0)

    # Remove old attribute
    mesh.color_attributes.remove(source_attr)

    # Create new attribute with target domain
    new_attr = mesh.color_attributes.new(
        name=source_name,
        type=source_type,
        domain=target_domain
    )

    # Apply colors
    if target_domain == 'POINT':
        for vert_idx, color in vert_colors.items():
            if vert_idx < len(new_attr.data):
                new_attr.data[vert_idx].color = color
    else:
        # CORNER domain
        for loop_idx, loop in enumerate(mesh.loops):
            vert_idx = loop.vertex_index
            if vert_idx in vert_colors:
                new_attr.data[loop_idx].color = vert_colors[vert_idx]

    set_active_color_attribute(mesh, source_name)
    mesh.update()

    log(f"[Convert Domain] Converted to {target_domain} domain", report)
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

def create_vertex_color_material(obj, layer_name="Col", mat_name="M_VertexColors", report=None):
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
