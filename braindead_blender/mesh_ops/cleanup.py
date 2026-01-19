"""
mesh_ops.cleanup - Mesh Cleanup and Repair Operations

Functions for filling holes, removing internal geometry, fixing manifold issues.
"""

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .utils import (
    ensure_object_mode, get_face_count, get_vertex_count,
    log, ProgressTracker, step_timer
)

# ============================================================================
# HOLE FILLING
# ============================================================================

def fill_holes(obj, max_sides=100, report=None):
    """
    Fill holes (open boundaries) in the mesh.

    Args:
        obj: Blender mesh object
        max_sides: Maximum edges for a hole to be filled (0 = fill all)
        report: Optional report list

    Returns:
        Number of holes filled
    """
    ensure_object_mode()

    with step_timer("Filling holes"):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        # Find boundary edges
        boundary_edges = [e for e in bm.edges if e.is_boundary]

        if not boundary_edges:
            log(f"[Fill Holes] {obj.name}: No holes found", report)
            bpy.ops.object.mode_set(mode='OBJECT')
            return 0

        # Find and fill boundary loops
        holes_filled = 0
        processed_edges = set()

        for start_edge in boundary_edges:
            if start_edge in processed_edges:
                continue

            # Trace boundary loop
            loop_edges = []
            current_edge = start_edge
            current_vert = start_edge.verts[0]

            while True:
                loop_edges.append(current_edge)
                processed_edges.add(current_edge)

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

            # Fill if within size limit
            if max_sides == 0 or len(loop_edges) <= max_sides:
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
                        pass

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        log(f"[Fill Holes] {obj.name}: Filled {holes_filled} holes", report)
        return holes_filled


def fill_holes_simple(obj, max_sides=1000, report=None):
    """
    Simple hole filling using Blender's built-in operator.

    Args:
        obj: Blender mesh object
        max_sides: Maximum hole size to fill
        report: Optional report list
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Select boundary edges
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True,
                                      use_multi_face=False, use_non_contiguous=False, use_verts=False)

    try:
        bpy.ops.mesh.fill_holes(sides=max_sides)
        log(f"[Fill Holes] {obj.name}: Holes filled", report)
    except:
        log(f"[Fill Holes] {obj.name}: No holes to fill or fill failed", report)

    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')


# ============================================================================
# INTERNAL GEOMETRY REMOVAL
# ============================================================================

def remove_internal_geometry(obj, method="RAYCAST", ray_samples=14, report=None):
    """
    Remove internal/hidden faces from mesh.

    Args:
        obj: Blender mesh object
        method: "RAYCAST" (accurate) or "SIMPLE" (fast)
        ray_samples: Number of ray directions (6, 14, or 26)
        report: Optional report list

    Returns:
        Number of faces removed
    """
    if method.upper() == "SIMPLE":
        return _remove_internal_simple(obj, report)
    else:
        return _remove_internal_raycast(obj, ray_samples, report)


def _remove_internal_raycast(obj, ray_samples, report):
    """Remove internal faces using ray casting."""
    ensure_object_mode()

    with step_timer("Removing internal geometry (raycast)"):
        bpy.context.view_layer.objects.active = obj
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_eval = obj_eval.data

        bm = bmesh.new()
        bm.from_mesh(mesh_eval)
        bm.faces.ensure_lookup_table()

        bvh = BVHTree.FromBMesh(bm)

        # Calculate mesh bounds
        min_co = Vector((float('inf'), float('inf'), float('inf')))
        max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

        for v in bm.verts:
            for i in range(3):
                min_co[i] = min(min_co[i], v.co[i])
                max_co[i] = max(max_co[i], v.co[i])

        padding = (max_co - min_co).length * 0.1
        min_co -= Vector((padding, padding, padding))
        max_co += Vector((padding, padding, padding))

        # Ray directions (6 cardinal directions)
        ray_directions = [
            Vector((1, 0, 0)), Vector((-1, 0, 0)),
            Vector((0, 1, 0)), Vector((0, -1, 0)),
            Vector((0, 0, 1)), Vector((0, 0, -1)),
        ]

        # Add corners if more samples requested
        if ray_samples >= 14:
            corners = [
                Vector((1, 1, 1)), Vector((1, 1, -1)),
                Vector((1, -1, 1)), Vector((1, -1, -1)),
                Vector((-1, 1, 1)), Vector((-1, 1, -1)),
                Vector((-1, -1, 1)), Vector((-1, -1, -1)),
            ]
            ray_directions.extend([c.normalized() for c in corners])

        initial_faces = len(bm.faces)
        visible_faces = set()
        ray_offset = 0.001

        progress = ProgressTracker(len(bm.faces), "Checking face visibility")

        for i, face in enumerate(bm.faces):
            face_center = face.calc_center_median()
            face_normal = face.normal

            # Cast rays from outside toward face
            is_visible = False

            for ray_dir in ray_directions:
                # Check if face could be visible from this direction
                if face_normal.dot(ray_dir) > 0.1:
                    continue

                # Cast ray from outside toward face center
                ray_origin = face_center - ray_dir * (max_co - min_co).length
                hit_loc, hit_normal, hit_idx, hit_dist = bvh.ray_cast(ray_origin, ray_dir)

                if hit_idx is not None and hit_idx == face.index:
                    is_visible = True
                    break

            if is_visible:
                visible_faces.add(face.index)

            if i % 5000 == 0:
                progress.update(i)

        progress.finish()

        # Delete internal faces
        faces_to_delete = [f for f in bm.faces if f.index not in visible_faces]
        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')

        # Apply to mesh
        bm.to_mesh(obj.data)
        bm.free()

        final_faces = get_face_count(obj)
        removed = initial_faces - final_faces

        log(f"[Internal Geo] {obj.name}: Removed {removed} internal faces ({initial_faces} -> {final_faces})", report)
        return removed


def _remove_internal_simple(obj, report):
    """Remove internal faces using Blender's built-in selection."""
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')

    try:
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type='FACE')
    except:
        pass

    bpy.ops.object.mode_set(mode='OBJECT')

    final_faces = get_face_count(obj)
    removed = initial_faces - final_faces

    log(f"[Internal Geo] {obj.name}: Removed {removed} internal faces (simple method)", report)
    return removed


# ============================================================================
# NON-MANIFOLD FIXES
# ============================================================================

def fix_non_manifold(obj, aggressive=False, report=None):
    """
    Fix non-manifold geometry.

    Args:
        obj: Blender mesh object
        aggressive: Use more aggressive fixing (may lose geometry)
        report: Optional report list

    Returns:
        True if any fixes were applied
    """
    ensure_object_mode()

    with step_timer("Fixing non-manifold geometry"):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        initial_verts = len(bm.verts)
        initial_faces = len(bm.faces)

        fixes_applied = False

        # Remove loose vertices
        loose_verts = [v for v in bm.verts if not v.link_faces]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
            log(f"[Manifold] Removed {len(loose_verts)} loose vertices", report)
            fixes_applied = True

        # Remove loose edges
        loose_edges = [e for e in bm.edges if not e.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')
            log(f"[Manifold] Removed {len(loose_edges)} loose edges", report)
            fixes_applied = True

        # Remove duplicate vertices
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)

        if aggressive:
            # Find and handle non-manifold edges
            non_manifold_edges = [e for e in bm.edges if not e.is_manifold]
            if non_manifold_edges:
                # Try to dissolve problematic edges
                try:
                    bmesh.ops.dissolve_edges(bm, edges=non_manifold_edges)
                    log(f"[Manifold] Dissolved {len(non_manifold_edges)} non-manifold edges", report)
                    fixes_applied = True
                except:
                    pass

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        final_verts = get_vertex_count(obj)
        final_faces = get_face_count(obj)

        if initial_verts != final_verts or initial_faces != final_faces:
            log(f"[Manifold] {obj.name}: {initial_verts} -> {final_verts} verts, {initial_faces} -> {final_faces} faces", report)
            fixes_applied = True

        return fixes_applied


# ============================================================================
# GENERAL CLEANUP
# ============================================================================

def cleanup_mesh(obj, merge_distance=0.0001, remove_doubles=True, report=None):
    """
    General mesh cleanup operations.

    Args:
        obj: Blender mesh object
        merge_distance: Distance for merging vertices
        remove_doubles: Whether to remove duplicate vertices
        report: Optional report list
    """
    ensure_object_mode()

    with step_timer("Mesh cleanup"):
        initial_verts = get_vertex_count(obj)

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        if remove_doubles and merge_distance > 0:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        final_verts = get_vertex_count(obj)
        merged = initial_verts - final_verts

        if merged > 0:
            log(f"[Cleanup] {obj.name}: Merged {merged} vertices", report)


def pre_cleanup_mesh(obj, fix_manifold=True, aggressive=False, report=None):
    """
    Pre-decimation cleanup to avoid creating bad geometry.

    Args:
        obj: Blender mesh object
        fix_manifold: Fix non-manifold geometry
        aggressive: Use aggressive manifold fixing
        report: Optional report list
    """
    ensure_object_mode()

    log(f"[Pre-Cleanup] Processing {obj.name}...", report)

    # Remove doubles first
    cleanup_mesh(obj, merge_distance=0.0001, report=report)

    # Fix manifold issues
    if fix_manifold:
        fix_non_manifold(obj, aggressive=aggressive, report=report)


def triangulate_ngons(obj, report=None):
    """
    Triangulate n-gons (faces with more than 4 vertices).

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Number of faces triangulated
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Find n-gons
    ngons = [f for f in bm.faces if len(f.verts) > 4]

    if ngons:
        bmesh.ops.triangulate(bm, faces=ngons)
        log(f"[Triangulate] {obj.name}: Triangulated {len(ngons)} n-gons", report)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    return len(ngons)


# ============================================================================
# LOOSE GEOMETRY REMOVAL
# ============================================================================

def remove_loose_geometry(obj, verts=True, edges=True, faces=True, report=None):
    """
    Remove loose/disconnected geometry.

    Args:
        obj: Blender mesh object
        verts: Remove loose vertices (not connected to any edge)
        edges: Remove loose edges (not connected to any face)
        faces: Remove loose faces (not connected to other faces - single isolated faces)
        report: Optional report list

    Returns:
        Dict with counts of removed elements
    """
    ensure_object_mode()

    # Force mesh data update before we start
    obj.data.update()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Get fresh bmesh
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    removed = {'verts': 0, 'edges': 0, 'faces': 0}

    # Remove isolated single faces FIRST (before edges/verts become orphaned)
    if faces:
        isolated_faces = []
        for face in bm.faces:
            # Check if any edge is shared with another face
            shared = False
            for edge in face.edges:
                if len(edge.link_faces) > 1:
                    shared = True
                    break
            if not shared:
                isolated_faces.append(face)

        if isolated_faces:
            bmesh.ops.delete(bm, geom=isolated_faces, context='FACES')
            removed['faces'] = len(isolated_faces)
            log(f"[Loose] Removed {len(isolated_faces)} isolated faces", report)
            # Refresh lookup tables after deletion
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()

    # Remove loose edges (no faces) - wire edges
    if edges:
        loose_edges = [e for e in bm.edges if not e.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')
            removed['edges'] = len(loose_edges)
            log(f"[Loose] Removed {len(loose_edges)} loose/wire edges", report)
            # Refresh lookup table after deletion
            bm.verts.ensure_lookup_table()

    # Remove loose vertices (no edges) LAST
    if verts:
        loose_verts = [v for v in bm.verts if not v.link_edges]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
            removed['verts'] = len(loose_verts)
            log(f"[Loose] Removed {len(loose_verts)} loose vertices", report)

    # Force update and sync
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Force another mesh update to ensure sync
    obj.data.update()

    total = removed['verts'] + removed['edges'] + removed['faces']
    log(f"[Loose] Total removed: {total} elements", report)

    return removed


def select_loose_geometry(obj, report=None):
    """
    Select loose geometry for inspection (edit mode).

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Dict with counts of selected elements
    """
    ensure_object_mode()

    # Force mesh data update before we start
    obj.data.update()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Get fresh bmesh
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Deselect all
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False

    selected = {'verts': 0, 'edges': 0, 'faces': 0}

    # Select loose vertices
    for v in bm.verts:
        if not v.link_edges:
            v.select = True
            selected['verts'] += 1

    # Select loose/wire edges
    for e in bm.edges:
        if not e.link_faces:
            e.select = True
            selected['edges'] += 1

    # Select isolated faces
    for face in bm.faces:
        shared = False
        for edge in face.edges:
            if len(edge.link_faces) > 1:
                shared = True
                break
        if not shared:
            face.select = True
            selected['faces'] += 1

    # Force sync
    bmesh.update_edit_mesh(obj.data)
    bm.free()  # Release bmesh to ensure clean state

    total = selected['verts'] + selected['edges'] + selected['faces']
    log(f"[Select Loose] Selected: {selected['verts']} verts, {selected['edges']} edges, {selected['faces']} faces", report)

    return selected


# ============================================================================
# MESH ISLAND OPERATIONS
# ============================================================================

def get_mesh_islands(bm):
    """
    Get list of disconnected mesh islands.

    Args:
        bm: BMesh object

    Returns:
        List of sets, each containing face indices for one island
    """
    bm.faces.ensure_lookup_table()

    visited = set()
    islands = []

    for face in bm.faces:
        if face.index in visited:
            continue

        # BFS to find all connected faces
        island = set()
        queue = [face]

        while queue:
            current = queue.pop(0)
            if current.index in visited:
                continue

            visited.add(current.index)
            island.add(current.index)

            # Add neighboring faces (share an edge)
            for edge in current.edges:
                for linked_face in edge.link_faces:
                    if linked_face.index not in visited:
                        queue.append(linked_face)

        islands.append(island)

    return islands


def remove_small_islands(obj, min_faces=10, keep_largest=True, report=None):
    """
    Remove small disconnected mesh islands.

    Args:
        obj: Blender mesh object
        min_faces: Minimum face count to keep an island
        keep_largest: Always keep the largest island regardless of size
        report: Optional report list

    Returns:
        Number of islands removed
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    islands = get_mesh_islands(bm)
    log(f"[Islands] Found {len(islands)} mesh islands", report)

    if not islands:
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    # Sort by size (largest first)
    islands.sort(key=len, reverse=True)

    # Log island sizes
    for i, island in enumerate(islands[:5]):
        log(f"[Islands]   Island {i+1}: {len(island)} faces", report)
    if len(islands) > 5:
        log(f"[Islands]   ... and {len(islands) - 5} more islands", report)

    # Determine which islands to remove
    faces_to_remove = []
    islands_removed = 0

    for i, island in enumerate(islands):
        # Always keep largest if requested
        if keep_largest and i == 0:
            continue

        # Remove if below threshold
        if len(island) < min_faces:
            for face_idx in island:
                faces_to_remove.append(bm.faces[face_idx])
            islands_removed += 1

    if faces_to_remove:
        bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
        log(f"[Islands] Removed {islands_removed} small islands ({len(faces_to_remove)} faces)", report)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    return islands_removed


def select_small_islands(obj, min_faces=10, invert=False, report=None):
    """
    Select mesh islands based on face count (enters edit mode).

    Args:
        obj: Blender mesh object
        min_faces: Face threshold for selection
        invert: If False, select islands with FEWER faces than threshold
                If True, select islands with MORE faces than threshold
        report: Optional report list

    Returns:
        Number of islands selected
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Deselect all
    for f in bm.faces:
        f.select = False

    islands = get_mesh_islands(bm)
    islands.sort(key=len, reverse=True)

    # Log island info for debugging
    log(f"[Select Islands] Found {len(islands)} total island(s)", report)
    for i, island in enumerate(islands[:5]):
        log(f"[Select Islands]   Island {i+1}: {len(island)} faces", report)
    if len(islands) > 5:
        log(f"[Select Islands]   ... and {len(islands) - 5} more", report)

    if len(islands) <= 1:
        log(f"[Select Islands] Only 1 island exists (main mesh) - no separate islands to select", report)
        bmesh.update_edit_mesh(obj.data)
        return 0

    islands_selected = 0
    faces_selected = 0
    comparison = ">" if invert else "<"

    for i, island in enumerate(islands):
        island_size = len(island)

        # Check threshold based on invert mode
        should_select = False
        if invert:
            # Select islands with MORE faces than threshold
            should_select = island_size > min_faces
        else:
            # Select islands with FEWER faces than threshold (skip largest by default)
            if i == 0:
                continue  # Always skip largest when looking for small islands
            should_select = island_size < min_faces

        if should_select:
            for face_idx in island:
                bm.faces[face_idx].select = True
                faces_selected += 1
            islands_selected += 1
            log(f"[Select Islands]   -> Selected island {i+1} ({island_size} faces, {comparison}{min_faces})", report)

    bmesh.update_edit_mesh(obj.data)

    if islands_selected == 0:
        log(f"[Select Islands] No islands match criteria ({comparison}{min_faces} faces)", report)

    log(f"[Select Islands] Result: {islands_selected} islands, {faces_selected} faces selected", report)
    return islands_selected


# ============================================================================
# N-GON OPERATIONS
# ============================================================================

def dissolve_large_ngons(obj, max_verts=6, method="TRIANGULATE", report=None):
    """
    Handle faces with too many vertices.

    Args:
        obj: Blender mesh object
        max_verts: Faces with more than this many verts will be processed
        method: How to handle large n-gons:
            - "TRIANGULATE": Split into triangles (recommended)
            - "DISSOLVE": Try limited dissolve to simplify (may create larger n-gons)
            - "DELETE": Remove the faces entirely (creates holes)
        report: Optional report list

    Returns:
        Number of faces affected
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    initial_faces = len(bm.faces)

    # Deselect all first
    for f in bm.faces:
        f.select = False

    # Find and select large n-gons
    target_count = 0
    for f in bm.faces:
        if len(f.verts) > max_verts:
            f.select = True
            target_count += 1

    if target_count == 0:
        log(f"[N-gons] No faces with more than {max_verts} vertices found", report)
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    log(f"[N-gons] Found {target_count} faces with >{max_verts} verts", report)

    # Update mesh to apply selection
    bmesh.update_edit_mesh(obj.data)

    if method == "TRIANGULATE":
        # Use Blender operator for triangulation
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bm = bmesh.from_edit_mesh(obj.data)
        final_faces = len(bm.faces)
        log(f"[N-gons] Triangulated: {initial_faces} -> {final_faces} faces", report)

    elif method == "DELETE":
        bpy.ops.mesh.delete(type='FACE')
        bm = bmesh.from_edit_mesh(obj.data)
        final_faces = len(bm.faces)
        deleted = initial_faces - final_faces
        log(f"[N-gons] Deleted {deleted} faces: {initial_faces} -> {final_faces}", report)

    elif method == "DISSOLVE":
        # Limited dissolve tries to simplify by removing edges where possible
        # This may actually create LARGER n-gons, not smaller ones!
        # It's more useful for simplifying geometry, not fixing n-gons
        log(f"[N-gons] Warning: Dissolve may create larger n-gons, not remove them", report)
        log(f"[N-gons] Consider using TRIANGULATE to split, or DELETE to remove", report)

        # Try limited dissolve anyway
        try:
            bpy.ops.mesh.dissolve_limited(angle_limit=0.0872665)  # ~5 degrees
            bm = bmesh.from_edit_mesh(obj.data)
            final_faces = len(bm.faces)
            log(f"[N-gons] Limited dissolve: {initial_faces} -> {final_faces} faces", report)
        except Exception as e:
            log(f"[N-gons] Dissolve failed: {e}", report)

    bpy.ops.object.mode_set(mode='OBJECT')

    # Verify results
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    remaining = sum(1 for f in bm.faces if len(f.verts) > max_verts)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[N-gons] Remaining faces with >{max_verts} verts: {remaining}", report)
    return target_count


def select_large_ngons(obj, max_verts=6, invert=False, report=None):
    """
    Select faces based on vertex count for inspection.

    Args:
        obj: Blender mesh object
        max_verts: Vertex threshold for selection
        invert: If False, select faces with MORE vertices than threshold
                If True, select faces with FEWER vertices than threshold
        report: Optional report list

    Returns:
        Number of faces selected
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Deselect all
    for f in bm.faces:
        f.select = False

    # Select faces based on vert count
    count = 0
    for f in bm.faces:
        if invert:
            # Select faces with FEWER verts than threshold
            if len(f.verts) < max_verts:
                f.select = True
                count += 1
        else:
            # Select faces with MORE verts than threshold
            if len(f.verts) > max_verts:
                f.select = True
                count += 1

    bmesh.update_edit_mesh(obj.data)

    comparison = "<" if invert else ">"
    log(f"[Select N-gons] Selected {count} faces with {comparison}{max_verts} verts", report)
    return count


def get_ngon_statistics(obj, report=None):
    """
    Get statistics about face vertex counts.

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Dict with statistics
    """
    ensure_object_mode()

    mesh = obj.data

    stats = {}
    for poly in mesh.polygons:
        vert_count = len(poly.vertices)
        if vert_count not in stats:
            stats[vert_count] = 0
        stats[vert_count] += 1

    log(f"[N-gon Stats] Face vertex distribution:", report)
    for vert_count in sorted(stats.keys()):
        label = "tris" if vert_count == 3 else "quads" if vert_count == 4 else f"{vert_count}-gons"
        log(f"[N-gon Stats]   {vert_count} verts ({label}): {stats[vert_count]} faces", report)

    return stats


# ============================================================================
# KEEP LARGEST ISLAND (AGGRESSIVE CLEANUP)
# ============================================================================

def keep_largest_island(obj, report=None):
    """
    Keep only the largest connected mesh island, removing all other geometry.
    This is an aggressive cleanup that removes all loose verts, edges, and small islands.

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Dict with removal statistics
    """
    ensure_object_mode()

    stats = {
        'loose_verts': 0,
        'loose_edges': 0,
        'islands_removed': 0,
        'faces_removed': 0
    }

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Step 1: Remove loose vertices (not connected to any edge)
    loose_verts = [v for v in bm.verts if not v.link_edges]
    if loose_verts:
        stats['loose_verts'] = len(loose_verts)
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
        log(f"[Keep Largest] Removed {stats['loose_verts']} loose vertices", report)

    # Step 2: Remove wire edges (not connected to any face)
    bm.edges.ensure_lookup_table()
    wire_edges = [e for e in bm.edges if not e.link_faces]
    if wire_edges:
        stats['loose_edges'] = len(wire_edges)
        bmesh.ops.delete(bm, geom=wire_edges, context='EDGES')
        log(f"[Keep Largest] Removed {stats['loose_edges']} wire edges", report)

    # Step 3: Find mesh islands and keep only the largest
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    if len(bm.faces) == 0:
        log(f"[Keep Largest] No faces remaining after loose geometry removal", report)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        return stats

    islands = get_mesh_islands(bm)
    log(f"[Keep Largest] Found {len(islands)} mesh islands", report)

    if len(islands) <= 1:
        log(f"[Keep Largest] Only one island, nothing to remove", report)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        return stats

    # Sort by size (largest first)
    islands.sort(key=len, reverse=True)

    # Log island sizes
    for i, island in enumerate(islands[:5]):
        log(f"[Keep Largest]   Island {i+1}: {len(island)} faces", report)
    if len(islands) > 5:
        log(f"[Keep Largest]   ... and {len(islands) - 5} more islands", report)

    # Remove all islands except the largest
    faces_to_remove = []
    for i, island in enumerate(islands[1:], start=2):  # Skip first (largest)
        for face_idx in island:
            faces_to_remove.append(bm.faces[face_idx])
        stats['islands_removed'] += 1

    if faces_to_remove:
        stats['faces_removed'] = len(faces_to_remove)
        bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
        log(f"[Keep Largest] Removed {stats['islands_removed']} islands ({stats['faces_removed']} faces)", report)

    # Final cleanup: remove any orphaned verts/edges created by face removal
    orphan_verts = [v for v in bm.verts if not v.link_faces]
    if orphan_verts:
        bmesh.ops.delete(bm, geom=orphan_verts, context='VERTS')
        log(f"[Keep Largest] Cleaned up {len(orphan_verts)} orphaned vertices", report)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    total_removed = stats['loose_verts'] + stats['loose_edges'] + stats['faces_removed']
    log(f"[Keep Largest] Total cleanup: {total_removed} elements removed", report)

    return stats


def merge_by_distance(obj, distance=0.0001, report=None):
    """
    Merge vertices that are very close together (remove doubles).

    Args:
        obj: Blender mesh object
        distance: Merge distance threshold
        report: Optional report list

    Returns:
        Number of vertices removed
    """
    ensure_object_mode()

    initial_verts = get_vertex_count(obj)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # remove_doubles returns dict with 'verts' key containing merged verts
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    final_verts = get_vertex_count(obj)
    actual_removed = initial_verts - final_verts

    log(f"[Merge] Merged {actual_removed} vertices (distance: {distance})", report)
    return actual_removed


def select_interior_faces(obj, report=None):
    """
    Select faces that appear to be interior (facing inward or occluded).
    Uses face normal direction relative to mesh center.

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Number of faces selected
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Calculate mesh center
    center = Vector((0, 0, 0))
    for v in bm.verts:
        center += v.co
    center /= len(bm.verts)

    # Deselect all
    for f in bm.faces:
        f.select = False

    # Select faces whose normals point toward center (interior faces)
    selected = 0
    for face in bm.faces:
        face_center = face.calc_center_median()
        to_mesh_center = center - face_center

        # If normal points toward mesh center, it's likely interior
        if face.normal.dot(to_mesh_center) > 0.1:
            face.select = True
            selected += 1

    bmesh.update_edit_mesh(obj.data)

    log(f"[Interior] Selected {selected} interior-facing faces", report)
    return selected


def delete_interior_faces(obj, report=None):
    """
    Delete faces that appear to be interior (facing inward).

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Number of faces deleted
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Calculate mesh center
    center = Vector((0, 0, 0))
    for v in bm.verts:
        center += v.co
    center /= len(bm.verts)

    # Find interior faces
    interior_faces = []
    for face in bm.faces:
        face_center = face.calc_center_median()
        to_mesh_center = center - face_center

        if face.normal.dot(to_mesh_center) > 0.1:
            interior_faces.append(face)

    deleted = len(interior_faces)

    if interior_faces:
        bmesh.ops.delete(bm, geom=interior_faces, context='FACES')
        log(f"[Interior] Deleted {deleted} interior-facing faces", report)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    return deleted


def flip_selected_faces(obj, report=None):
    """
    Flip normals of selected faces.

    Args:
        obj: Blender mesh object (must be in edit mode with faces selected)
        report: Optional report list

    Returns:
        Number of faces flipped
    """
    # Must be in edit mode
    if obj.mode != 'EDIT':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    selected_faces = [f for f in bm.faces if f.select]
    flipped = len(selected_faces)

    if selected_faces:
        bmesh.ops.reverse_faces(bm, faces=selected_faces)
        log(f"[Flip] Flipped {flipped} selected faces", report)

    bmesh.update_edit_mesh(obj.data)

    return flipped


def flip_interior_faces(obj, report=None):
    """
    Flip normals of faces that appear to be interior (facing inward).

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Number of faces flipped
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Calculate mesh center
    center = Vector((0, 0, 0))
    for v in bm.verts:
        center += v.co
    center /= len(bm.verts)

    # Find interior faces
    interior_faces = []
    for face in bm.faces:
        face_center = face.calc_center_median()
        to_mesh_center = center - face_center

        if face.normal.dot(to_mesh_center) > 0.1:
            interior_faces.append(face)

    flipped = len(interior_faces)

    if interior_faces:
        bmesh.ops.reverse_faces(bm, faces=interior_faces)
        log(f"[Interior] Flipped {flipped} interior-facing faces", report)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    return flipped


def select_embedded_faces(obj, max_verts=4, invert=False, report=None):
    """
    Select faces that are "embedded" in regions of different-sized faces.

    Args:
        obj: Blender mesh object
        max_verts: Vertex threshold for selection
        invert: If False, select faces with FEWER verts surrounded by larger faces
                If True, select faces with MORE verts surrounded by smaller faces
        report: Optional report list

    Returns:
        Number of faces selected
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Deselect all
    for f in bm.faces:
        f.select = False

    selected = 0
    comparison = ">" if invert else "<"

    for face in bm.faces:
        face_verts = len(face.verts)

        # Check vertex count threshold
        if invert:
            # Looking for large faces (MORE verts than threshold)
            if face_verts <= max_verts:
                continue
        else:
            # Looking for small faces (FEWER verts than threshold)
            if face_verts > max_verts:
                continue

        # Check neighbors
        neighbor_count = 0
        different_size_neighbors = 0

        for edge in face.edges:
            for neighbor in edge.link_faces:
                if neighbor != face:
                    neighbor_count += 1
                    neighbor_verts = len(neighbor.verts)
                    if invert:
                        # For large faces, look for smaller neighbors
                        if neighbor_verts < face_verts:
                            different_size_neighbors += 1
                    else:
                        # For small faces, look for larger neighbors
                        if neighbor_verts > face_verts:
                            different_size_neighbors += 1

        # Select if most neighbors are different size (embedded)
        if neighbor_count > 0 and different_size_neighbors >= neighbor_count * 0.5:
            face.select = True
            selected += 1

    bmesh.update_edit_mesh(obj.data)

    if invert:
        log(f"[Embedded] Selected {selected} large faces ({comparison}{max_verts} verts) embedded in smaller regions", report)
    else:
        log(f"[Embedded] Selected {selected} small faces ({comparison}{max_verts} verts) embedded in larger regions", report)
    return selected


def dissolve_embedded_faces(obj, max_verts=4, invert=False, clear_sharp=True, repeat=False, report=None):
    """
    Dissolve faces that are "embedded" in regions of different-sized faces.

    Uses Blender's built-in dissolve operator for more reliable results.

    Args:
        obj: Blender mesh object
        max_verts: Vertex threshold for selection
        invert: If False, dissolve faces with FEWER verts surrounded by larger faces
                If True, dissolve faces with MORE verts surrounded by smaller faces
        clear_sharp: Clear sharp edges on affected faces before dissolving
        repeat: If True, keep dissolving until no more progress is made
        report: Optional report list

    Returns:
        Total number of faces dissolved
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')

    bm = bmesh.from_edit_mesh(obj.data)
    initial_faces = len(bm.faces)
    total_dissolved = 0
    iteration = 0
    max_iterations = 50 if repeat else 1  # Safety limit

    comparison = ">" if invert else "<"

    while iteration < max_iterations:
        iteration += 1
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        before_count = len(bm.faces)

        # Deselect all first
        for f in bm.faces:
            f.select = False

        # Find and select embedded faces
        faces_to_dissolve = []
        for face in bm.faces:
            face_verts = len(face.verts)

            # Check vertex count threshold
            if invert:
                if face_verts <= max_verts:
                    continue
            else:
                if face_verts > max_verts:
                    continue

            # Check neighbors
            neighbor_count = 0
            different_size_neighbors = 0

            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor != face:
                        neighbor_count += 1
                        neighbor_verts = len(neighbor.verts)
                        if invert:
                            if neighbor_verts < face_verts:
                                different_size_neighbors += 1
                        else:
                            if neighbor_verts > face_verts:
                                different_size_neighbors += 1

            if neighbor_count > 0 and different_size_neighbors >= neighbor_count * 0.5:
                faces_to_dissolve.append(face)
                face.select = True

        if not faces_to_dissolve:
            if iteration == 1:
                log(f"[Embedded] No embedded faces found ({comparison}{max_verts} verts)", report)
            break

        target_count = len(faces_to_dissolve)
        if repeat:
            log(f"[Embedded] Pass {iteration}: Found {target_count} embedded faces", report)
        else:
            log(f"[Embedded] Found {target_count} embedded faces to dissolve", report)

        # Clear sharp edges if requested
        if clear_sharp:
            edges_cleared = 0
            edges_to_clear = set()
            for face in faces_to_dissolve:
                for edge in face.edges:
                    if not edge.smooth:
                        edges_to_clear.add(edge)

            for edge in edges_to_clear:
                edge.smooth = True
                edges_cleared += 1

            if edges_cleared > 0:
                log(f"[Embedded] Cleared {edges_cleared} sharp edges", report)

        bmesh.update_edit_mesh(obj.data)

        # Dissolve
        try:
            bpy.ops.mesh.dissolve_faces()
        except Exception as e:
            log(f"[Embedded] Dissolve failed: {e}", report)
            try:
                bpy.ops.mesh.dissolve_limited(angle_limit=0.087)
            except:
                pass

        # Check progress
        bm = bmesh.from_edit_mesh(obj.data)
        after_count = len(bm.faces)
        dissolved_this_pass = before_count - after_count
        total_dissolved += dissolved_this_pass

        if repeat:
            log(f"[Embedded] Pass {iteration}: {before_count} -> {after_count} ({dissolved_this_pass} removed)", report)

        # Stop if no progress
        if dissolved_this_pass == 0:
            if repeat and iteration > 1:
                log(f"[Embedded] Stopping - no more faces can be dissolved", report)
            break

        if not repeat:
            break

    bpy.ops.object.mode_set(mode='OBJECT')

    final_faces = initial_faces - total_dissolved
    if repeat and iteration > 1:
        log(f"[Embedded] Total: {initial_faces} -> {final_faces} faces ({total_dissolved} removed in {iteration} passes)", report)
    else:
        log(f"[Embedded] Result: {initial_faces} -> {final_faces} faces ({total_dissolved} removed)", report)

    return total_dissolved


# ============================================================================
# EDGE MARKING OPERATIONS (Sharp, Crease, Bevel Weight)
# ============================================================================

def get_crease_layer(bm, create=True):
    """
    Get the edge crease layer from BMesh.
    Blender 4.0+ stores crease as a float layer named 'crease_edge'.

    Args:
        bm: BMesh object
        create: If True, create the layer if it doesn't exist

    Returns:
        The crease layer or None
    """
    # Blender 4.0+ uses float layer with name 'crease_edge'
    crease_layer = bm.edges.layers.float.get('crease_edge')
    if crease_layer is None and create:
        crease_layer = bm.edges.layers.float.new('crease_edge')
    return crease_layer


def get_bevel_layer(bm, create=True):
    """
    Get the edge bevel weight layer from BMesh.
    Blender 4.0+ stores bevel as a float layer named 'bevel_weight_edge'.

    Args:
        bm: BMesh object
        create: If True, create the layer if it doesn't exist

    Returns:
        The bevel layer or None
    """
    bevel_layer = bm.edges.layers.float.get('bevel_weight_edge')
    if bevel_layer is None and create:
        bevel_layer = bm.edges.layers.float.new('bevel_weight_edge')
    return bevel_layer


def clear_edge_marks(obj, clear_sharp=True, clear_crease=True, clear_bevel=False, report=None):
    """
    Clear edge marks from the mesh.

    Args:
        obj: Blender mesh object
        clear_sharp: Clear sharp edge marks
        clear_crease: Clear edge crease values
        clear_bevel: Clear bevel weight values
        report: Optional report list

    Returns:
        Dict with counts of cleared marks
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Get crease layer (Blender 4.0+ API)
    crease_layer = get_crease_layer(bm, create=False) if clear_crease else None
    bevel_layer = get_bevel_layer(bm, create=False) if clear_bevel else None

    cleared = {'sharp': 0, 'crease': 0, 'bevel': 0}

    for edge in bm.edges:
        if clear_sharp and not edge.smooth:
            edge.smooth = True
            cleared['sharp'] += 1

        if clear_crease and crease_layer and edge[crease_layer] > 0:
            edge[crease_layer] = 0.0
            cleared['crease'] += 1

        if clear_bevel and bevel_layer and edge[bevel_layer] > 0:
            edge[bevel_layer] = 0.0
            cleared['bevel'] += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    parts = []
    if cleared['sharp'] > 0:
        parts.append(f"{cleared['sharp']} sharp")
    if cleared['crease'] > 0:
        parts.append(f"{cleared['crease']} crease")
    if cleared['bevel'] > 0:
        parts.append(f"{cleared['bevel']} bevel")

    if parts:
        log(f"[Edge Marks] Cleared: {', '.join(parts)}", report)
    else:
        log(f"[Edge Marks] No marks to clear", report)

    return cleared


def mark_edges_from_colors(obj, threshold=0.15, mode='ADD', mark_type='CREASE',
                           crease_value=1.0, report=None):
    """
    Mark edges where vertex colors differ significantly.

    Args:
        obj: Blender mesh object
        threshold: Color difference threshold (0-1)
        mode: 'ADD' to add to existing, 'REPLACE' to clear first
        mark_type: 'SHARP', 'CREASE', or 'BOTH'
        crease_value: Crease strength (0-1)
        report: Optional report list

    Returns:
        Number of edges marked
    """
    ensure_object_mode()

    mesh = obj.data
    if not mesh.color_attributes:
        log(f"[Edge Marks] No vertex colors found on {obj.name}", report)
        return 0

    # Get color layer
    color_layer = mesh.color_attributes.active_color
    if not color_layer:
        color_layer = mesh.color_attributes[0]

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Get/create crease layer (Blender 4.0+ API)
    crease_layer = get_crease_layer(bm, create=True) if mark_type in ('CREASE', 'BOTH') else None

    # Clear existing if replacing
    if mode == 'REPLACE':
        cleared = 0
        for edge in bm.edges:
            if mark_type in ('SHARP', 'BOTH') and not edge.smooth:
                edge.smooth = True
                cleared += 1
            if mark_type in ('CREASE', 'BOTH') and crease_layer and edge[crease_layer] > 0:
                edge[crease_layer] = 0.0
        if cleared > 0:
            log(f"[Edge Marks] Cleared existing marks", report)

    # Get color layer in bmesh
    color_lay = bm.loops.layers.color.active
    if not color_lay:
        log(f"[Edge Marks] Could not access color layer in edit mode", report)
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    marked = 0
    for edge in bm.edges:
        # Get faces on each side of edge
        if len(edge.link_faces) != 2:
            continue

        face1, face2 = edge.link_faces

        # Get colors from each face at the shared edge
        def get_face_edge_colors(face, edge):
            colors = []
            for loop in face.loops:
                if loop.vert in edge.verts:
                    col = loop[color_lay]
                    colors.append((col[0], col[1], col[2]))
            return colors

        colors1 = get_face_edge_colors(face1, edge)
        colors2 = get_face_edge_colors(face2, edge)

        if not colors1 or not colors2:
            continue

        # Compare colors - use average of each face's edge colors
        avg1 = (sum(c[0] for c in colors1) / len(colors1),
                sum(c[1] for c in colors1) / len(colors1),
                sum(c[2] for c in colors1) / len(colors1))
        avg2 = (sum(c[0] for c in colors2) / len(colors2),
                sum(c[1] for c in colors2) / len(colors2),
                sum(c[2] for c in colors2) / len(colors2))

        # Calculate color difference
        diff = ((avg1[0] - avg2[0])**2 +
                (avg1[1] - avg2[1])**2 +
                (avg1[2] - avg2[2])**2) ** 0.5

        if diff > threshold:
            if mark_type in ('SHARP', 'BOTH'):
                edge.smooth = False
            if mark_type in ('CREASE', 'BOTH') and crease_layer:
                edge[crease_layer] = crease_value
            marked += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    type_str = mark_type.lower()
    log(f"[Edge Marks] Marked {marked} edges as {type_str} (color threshold: {threshold})", report)
    return marked


def mark_edges_from_angle(obj, angle_threshold=30.0, mode='ADD', mark_type='CREASE',
                          crease_value=1.0, report=None):
    """
    Mark edges based on angle between faces.

    Args:
        obj: Blender mesh object
        angle_threshold: Angle in degrees above which edges are marked
        mode: 'ADD' to add to existing, 'REPLACE' to clear first
        mark_type: 'SHARP', 'CREASE', or 'BOTH'
        crease_value: Crease strength (0-1)
        report: Optional report list

    Returns:
        Number of edges marked
    """
    import math

    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Get/create crease layer (Blender 4.0+ API)
    crease_layer = get_crease_layer(bm, create=True) if mark_type in ('CREASE', 'BOTH') else None

    # Clear existing if replacing
    if mode == 'REPLACE':
        for edge in bm.edges:
            if mark_type in ('SHARP', 'BOTH'):
                edge.smooth = True
            if mark_type in ('CREASE', 'BOTH') and crease_layer:
                edge[crease_layer] = 0.0

    angle_rad = math.radians(angle_threshold)
    marked = 0

    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue

        face1, face2 = edge.link_faces
        angle = face1.normal.angle(face2.normal)

        if angle > angle_rad:
            if mark_type in ('SHARP', 'BOTH'):
                edge.smooth = False
            if mark_type in ('CREASE', 'BOTH') and crease_layer:
                edge[crease_layer] = crease_value
            marked += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    type_str = mark_type.lower()
    log(f"[Edge Marks] Marked {marked} edges as {type_str} (angle > {angle_threshold}°)", report)
    return marked


def convert_sharp_to_crease(obj, crease_value=1.0, clear_sharp=False, report=None):
    """
    Convert sharp edges to edge crease.

    Args:
        obj: Blender mesh object
        crease_value: Crease strength to apply (0-1)
        clear_sharp: Also clear the sharp marking after converting
        report: Optional report list

    Returns:
        Number of edges converted
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    # Get/create crease layer (Blender 4.0+ API)
    crease_layer = get_crease_layer(bm, create=True)

    converted = 0
    for edge in bm.edges:
        if not edge.smooth:  # Edge is sharp
            if crease_layer:
                edge[crease_layer] = crease_value
            converted += 1
            if clear_sharp:
                edge.smooth = True

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    if clear_sharp:
        log(f"[Edge Marks] Converted {converted} sharp edges to crease (cleared sharp)", report)
    else:
        log(f"[Edge Marks] Converted {converted} sharp edges to crease", report)
    return converted


def convert_crease_to_sharp(obj, threshold=0.5, clear_crease=False, report=None):
    """
    Convert edge crease to sharp edges.

    Args:
        obj: Blender mesh object
        threshold: Minimum crease value to convert (0-1)
        clear_crease: Also clear the crease after converting
        report: Optional report list

    Returns:
        Number of edges converted
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    # Get crease layer (Blender 4.0+ API) - don't create if doesn't exist
    crease_layer = get_crease_layer(bm, create=False)

    if not crease_layer:
        log(f"[Edge Marks] No crease data found on {obj.name}", report)
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    converted = 0
    for edge in bm.edges:
        if edge[crease_layer] >= threshold:
            edge.smooth = False  # Mark as sharp
            converted += 1
            if clear_crease:
                edge[crease_layer] = 0.0

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Edge Marks] Converted {converted} creased edges to sharp (threshold: {threshold})", report)
    return converted


# Keep old function names for backwards compatibility
def clear_all_sharp_edges(obj, report=None):
    """Clear all sharp edges (backwards compatible wrapper)."""
    result = clear_edge_marks(obj, clear_sharp=True, clear_crease=False, report=report)
    return result.get('sharp', 0)


def mark_sharp_from_colors(obj, threshold=0.15, mode='ADD', report=None):
    """Mark sharp edges from colors (backwards compatible wrapper)."""
    return mark_edges_from_colors(obj, threshold, mode, mark_type='SHARP', report=report)


def mark_sharp_from_angle(obj, angle_threshold=30.0, mode='ADD', report=None):
    """Mark sharp edges from angle (backwards compatible wrapper)."""
    return mark_edges_from_angle(obj, angle_threshold, mode, mark_type='SHARP', report=report)


def smart_cleanup(obj, remove_loose=True, keep_largest=True, min_island_faces=10,
                  merge_verts=False, merge_distance=0.0001,
                  do_fill_holes=False, fill_max_sides=8,
                  do_fix_normals=False,
                  dissolve_embedded=False, embedded_max_verts=4,
                  triangulate_ngons=False, ngon_threshold=6,
                  report=None):
    """
    Smart cleanup that combines multiple operations in the right order.

    Workflow (in order):
    1. Merge close vertices (remove doubles)
    2. Remove loose vertices and wire edges
    3. Remove small islands (or keep only largest)
    4. Fill small holes
    5. Dissolve embedded faces
    6. Triangulate large n-gons
    7. Fix normals
    8. Final orphan cleanup

    Args:
        obj: Blender mesh object
        remove_loose: Remove loose verts and wire edges
        keep_largest: If True, keep only largest island. If False, use min_island_faces threshold
        min_island_faces: Minimum faces to keep an island (only used if keep_largest=False)
        merge_verts: Merge vertices within distance (remove doubles)
        merge_distance: Distance for vertex merging
        do_fill_holes: Fill small holes in mesh
        fill_max_sides: Maximum edges for a hole to be filled
        do_fix_normals: Recalculate normals to face outward
        dissolve_embedded: Dissolve small embedded faces
        embedded_max_verts: Max vertices for embedded face detection
        triangulate_ngons: Triangulate large n-gons
        ngon_threshold: Faces with more vertices than this are triangulated
        report: Optional report list

    Returns:
        Dict with cleanup statistics
    """
    ensure_object_mode()

    stats = {
        'initial_verts': len(obj.data.vertices),
        'initial_faces': get_face_count(obj),
        'merged_verts': 0,
        'loose_verts': 0,
        'loose_edges': 0,
        'islands_removed': 0,
        'island_faces_removed': 0,
        'holes_filled': 0,
        'embedded_dissolved': 0,
        'ngons_triangulated': 0,
        'normals_fixed': 0,
        'final_verts': 0,
        'final_faces': 0
    }

    log(f"[Smart Cleanup] Starting cleanup of {obj.name}", report)
    log(f"[Smart Cleanup] Initial: {stats['initial_verts']} verts, {stats['initial_faces']} faces", report)

    bpy.context.view_layer.objects.active = obj

    # Step 1: Merge vertices (remove doubles) - do this first to clean up overlapping geometry
    if merge_verts and merge_distance > 0:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        verts_before = len(obj.data.vertices)
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        bpy.ops.object.mode_set(mode='OBJECT')
        stats['merged_verts'] = verts_before - len(obj.data.vertices)
        if stats['merged_verts'] > 0:
            log(f"[Smart Cleanup] Merged {stats['merged_verts']} vertices (distance={merge_distance})", report)

    # Step 2: Remove loose geometry
    if remove_loose:
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        # Loose vertices (not connected to edges)
        loose_verts = [v for v in bm.verts if not v.link_edges]
        if loose_verts:
            stats['loose_verts'] = len(loose_verts)
            bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
            log(f"[Smart Cleanup] Removed {stats['loose_verts']} loose vertices", report)

        # Wire edges (not connected to faces)
        bm.edges.ensure_lookup_table()
        wire_edges = [e for e in bm.edges if not e.link_faces]
        if wire_edges:
            stats['loose_edges'] = len(wire_edges)
            bmesh.ops.delete(bm, geom=wire_edges, context='EDGES')
            log(f"[Smart Cleanup] Removed {stats['loose_edges']} wire edges", report)

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Step 3: Handle islands
    if keep_largest or min_island_faces > 1:
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        if len(bm.faces) > 0:
            islands = get_mesh_islands(bm)

            if len(islands) > 1:
                islands.sort(key=len, reverse=True)
                log(f"[Smart Cleanup] Found {len(islands)} islands (largest: {len(islands[0])} faces)", report)

                faces_to_remove = []

                if keep_largest:
                    # Remove all except largest
                    for island in islands[1:]:
                        for face_idx in island:
                            if face_idx < len(bm.faces):
                                faces_to_remove.append(bm.faces[face_idx])
                        stats['islands_removed'] += 1
                else:
                    # Remove islands below threshold (keep largest always)
                    for i, island in enumerate(islands):
                        if i == 0:  # Always keep largest
                            continue
                        if len(island) < min_island_faces:
                            for face_idx in island:
                                if face_idx < len(bm.faces):
                                    faces_to_remove.append(bm.faces[face_idx])
                            stats['islands_removed'] += 1

                if faces_to_remove:
                    stats['island_faces_removed'] = len(faces_to_remove)
                    bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
                    log(f"[Smart Cleanup] Removed {stats['islands_removed']} islands ({stats['island_faces_removed']} faces)", report)
            else:
                log(f"[Smart Cleanup] Only 1 island exists - no islands to remove", report)

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Step 4: Fill holes
    if do_fill_holes:
        # fill_holes returns an integer (number of holes filled)
        stats['holes_filled'] = fill_holes(obj, max_sides=fill_max_sides, report=report)
        if stats['holes_filled'] > 0:
            log(f"[Smart Cleanup] Filled {stats['holes_filled']} holes (max_sides={fill_max_sides})", report)

    # Step 5: Dissolve embedded faces (repeat until stable)
    if dissolve_embedded:
        total_dissolved = 0
        max_iterations = 50
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            faces_before = get_face_count(obj)

            # Select embedded faces
            select_embedded_faces(obj, max_verts=embedded_max_verts, invert=False, report=None)

            # Count selected
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            selected = [f for f in bm.faces if f.select]
            count = len(selected)
            bpy.ops.object.mode_set(mode='OBJECT')

            if count == 0:
                break

            # Dissolve
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.dissolve_faces()
            bpy.ops.object.mode_set(mode='OBJECT')

            faces_after = get_face_count(obj)
            dissolved = faces_before - faces_after

            if dissolved <= 0:
                break

            total_dissolved += dissolved

        stats['embedded_dissolved'] = total_dissolved
        if total_dissolved > 0:
            log(f"[Smart Cleanup] Dissolved {total_dissolved} embedded faces ({iteration} passes)", report)

    # Step 6: Triangulate large n-gons
    if triangulate_ngons:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        # Select n-gons larger than threshold
        ngons = [f for f in bm.faces if len(f.verts) > ngon_threshold]
        for f in ngons:
            f.select = True
        stats['ngons_triangulated'] = len(ngons)

        bmesh.update_edit_mesh(obj.data)

        if ngons:
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
            log(f"[Smart Cleanup] Triangulated {stats['ngons_triangulated']} n-gons (>{ngon_threshold} verts)", report)

        bpy.ops.object.mode_set(mode='OBJECT')

    # Step 7: Fix normals
    if do_fix_normals:
        from . import normals as normals_module
        # fix_normals returns True if normals were fixed, False otherwise
        result = normals_module.fix_normals(obj, method='BOTH', report=report)
        stats['normals_fixed'] = 1 if result else 0
        if result:
            log(f"[Smart Cleanup] Fixed normals", report)

    # Step 8: Final orphan cleanup
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    orphan_verts = [v for v in bm.verts if not v.link_faces]
    if orphan_verts:
        bmesh.ops.delete(bm, geom=orphan_verts, context='VERTS')
        log(f"[Smart Cleanup] Cleaned up {len(orphan_verts)} orphaned vertices", report)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Final stats
    stats['final_verts'] = len(obj.data.vertices)
    stats['final_faces'] = get_face_count(obj)

    log(f"[Smart Cleanup] Complete: {stats['initial_verts']} -> {stats['final_verts']} verts, "
        f"{stats['initial_faces']} -> {stats['final_faces']} faces", report)

    return stats
