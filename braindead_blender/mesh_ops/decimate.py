"""
mesh_ops.decimate - Decimation Operations

Functions for reducing mesh polygon count while preserving quality.
"""

import bpy
import bmesh
import math

from .utils import (
    ensure_object_mode, get_face_count,
    log, step_timer
)

# ============================================================================
# PLANAR DECIMATION
# ============================================================================

def apply_planar_decimate(obj, angle=7.0, use_dissolve_boundaries=False, report=None):
    """
    Apply planar decimation - merge coplanar faces.

    This preserves flat surfaces while reducing polygon count.
    Ideal for stylized low-poly where you want large flat faces.

    Args:
        obj: Blender mesh object
        angle: Angle threshold in degrees (faces within this angle are merged)
        use_dissolve_boundaries: Whether to dissolve boundary edges
        report: Optional report list

    Returns:
        Final face count
    """
    ensure_object_mode()

    with step_timer(f"Planar decimate (angle={angle}°)"):
        initial_faces = get_face_count(obj)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Add decimate modifier
        mod = obj.modifiers.new(name="Planar_Decimate", type='DECIMATE')
        mod.decimate_type = 'DISSOLVE'
        mod.angle_limit = math.radians(angle)
        mod.use_dissolve_boundaries = use_dissolve_boundaries

        bpy.ops.object.modifier_apply(modifier=mod.name)

        final_faces = get_face_count(obj)
        reduction = ((initial_faces - final_faces) / initial_faces * 100) if initial_faces > 0 else 0

        log(f"[Planar Decimate] {obj.name}: {initial_faces:,} -> {final_faces:,} faces ({reduction:.1f}% reduction)", report)

        return final_faces


def apply_collapse_decimate(obj, target_faces=None, ratio=None,
                            preserve_boundaries=True, report=None):
    """
    Apply collapse decimation - reduce to target face count.

    Args:
        obj: Blender mesh object
        target_faces: Target face count (if None, use ratio)
        ratio: Decimation ratio 0-1 (if target_faces is None)
        preserve_boundaries: Preserve mesh boundary edges
        report: Optional report list

    Returns:
        Final face count
    """
    ensure_object_mode()

    with step_timer("Collapse decimate"):
        initial_faces = get_face_count(obj)

        # Calculate ratio from target faces
        if target_faces is not None:
            if initial_faces > 0:
                ratio = target_faces / initial_faces
                ratio = min(1.0, max(0.01, ratio))
            else:
                ratio = 1.0
        elif ratio is None:
            ratio = 0.5

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Add decimate modifier
        mod = obj.modifiers.new(name="Collapse_Decimate", type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        mod.use_collapse_triangulate = True

        if preserve_boundaries:
            # Blender 3.x and 4.x have different attribute names
            if hasattr(mod, 'use_symmetry'):
                mod.use_symmetry = False

        bpy.ops.object.modifier_apply(modifier=mod.name)

        final_faces = get_face_count(obj)

        log(f"[Collapse Decimate] {obj.name}: {initial_faces:,} -> {final_faces:,} faces (ratio: {ratio:.3f})", report)

        if target_faces and abs(final_faces - target_faces) > target_faces * 0.2:
            log(f"[Collapse Decimate] Note: Result differs from target by {abs(final_faces - target_faces):,} faces", report)

        return final_faces


# ============================================================================
# EDGE MARKING
# ============================================================================

def mark_sharp_edges(obj, angle=30.0, report=None):
    """
    Mark edges as sharp based on angle between faces.

    Args:
        obj: Blender mesh object
        angle: Angle threshold in degrees
        report: Optional report list

    Returns:
        Number of edges marked
    """
    ensure_object_mode()

    with step_timer(f"Marking sharp edges (angle={angle}°)"):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()

        angle_rad = math.radians(angle)
        marked_count = 0

        for edge in bm.edges:
            if len(edge.link_faces) != 2:
                continue

            f1, f2 = edge.link_faces[0], edge.link_faces[1]
            face_angle = f1.normal.angle(f2.normal)

            if face_angle > angle_rad:
                edge.smooth = False
                marked_count += 1
            else:
                edge.smooth = True

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        log(f"[Sharp Edges] {obj.name}: Marked {marked_count} sharp edges (angle > {angle}°)", report)

        return marked_count


def clear_sharp_edges(obj, report=None):
    """
    Clear all sharp edge marks.

    Args:
        obj: Blender mesh object
        report: Optional report list
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    for edge in bm.edges:
        edge.smooth = True

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Sharp Edges] {obj.name}: Cleared all sharp edges", report)


def mark_seams_from_sharp(obj, report=None):
    """
    Mark UV seams on all sharp edges.

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        Number of seams marked
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    seam_count = 0
    for edge in bm.edges:
        if not edge.smooth:
            edge.seam = True
            seam_count += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Seams] {obj.name}: Marked {seam_count} seams from sharp edges", report)

    return seam_count


# ============================================================================
# COMBINED OPERATIONS
# ============================================================================

def decimate_stylized(obj, target_faces=5000, planar_angle=7.0, sharp_angle=30.0,
                      preserve_boundaries=True, report=None):
    """
    Full stylized decimation pipeline.

    1. Planar decimate to merge coplanar faces
    2. Triangulate n-gons
    3. Collapse decimate to target face count
    4. Mark sharp edges

    Args:
        obj: Blender mesh object
        target_faces: Target face count
        planar_angle: Angle for planar decimation
        sharp_angle: Angle for sharp edge marking
        preserve_boundaries: Preserve mesh boundaries
        report: Optional report list

    Returns:
        Final face count
    """
    from .cleanup import triangulate_ngons

    log(f"\n[Stylized Decimate] Processing {obj.name}...", report)
    log(f"[Stylized Decimate] Target: {target_faces:,} faces", report)

    initial_faces = get_face_count(obj)
    log(f"[Stylized Decimate] Initial: {initial_faces:,} faces", report)

    # Step 1: Planar decimate
    if planar_angle > 0:
        apply_planar_decimate(obj, angle=planar_angle, report=report)

    # Step 2: Triangulate n-gons (required for accurate collapse)
    triangulate_ngons(obj, report=report)

    # Step 3: Collapse to target
    if target_faces > 0:
        apply_collapse_decimate(obj, target_faces=target_faces,
                                preserve_boundaries=preserve_boundaries, report=report)

    # Step 4: Mark sharp edges
    if sharp_angle > 0:
        mark_sharp_edges(obj, angle=sharp_angle, report=report)

    final_faces = get_face_count(obj)
    total_reduction = ((initial_faces - final_faces) / initial_faces * 100) if initial_faces > 0 else 0

    log(f"[Stylized Decimate] Complete: {initial_faces:,} -> {final_faces:,} faces ({total_reduction:.1f}% total reduction)", report)

    return final_faces


def decimate_to_ratio(obj, ratio=0.5, method="COLLAPSE", report=None):
    """
    Simple decimation by ratio.

    Args:
        obj: Blender mesh object
        ratio: Target ratio (0.5 = half the faces)
        method: "COLLAPSE" or "UNSUBDIV"
        report: Optional report list

    Returns:
        Final face count
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')

    if method.upper() == "UNSUBDIV":
        mod.decimate_type = 'UNSUBDIV'
        mod.iterations = max(1, int(-math.log2(ratio))) if ratio > 0 else 1
    else:
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio

    bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)

    log(f"[Decimate] {obj.name}: {initial_faces:,} -> {final_faces:,} faces (ratio: {ratio:.2f})", report)

    return final_faces
