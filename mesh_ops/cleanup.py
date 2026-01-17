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
