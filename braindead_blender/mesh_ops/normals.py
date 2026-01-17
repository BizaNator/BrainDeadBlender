"""
mesh_ops.normals - Normal Operations

Functions for fixing and verifying mesh normals.
"""

import bpy
import bmesh
from mathutils import Vector

from .utils import (
    ensure_object_mode, get_face_count,
    log, ProgressTracker, step_timer
)

# ============================================================================
# NORMAL CHECKING
# ============================================================================

def check_normal_orientation(obj):
    """
    Check what percentage of normals point inward vs outward.

    Args:
        obj: Blender mesh object

    Returns:
        dict with 'inward', 'outward', 'inward_pct' keys
    """
    ensure_object_mode()

    mesh = obj.data

    # Calculate mesh center
    center = Vector((0, 0, 0))
    for v in mesh.vertices:
        center += v.co
    center /= len(mesh.vertices)

    inward = 0
    outward = 0

    for poly in mesh.polygons:
        # Vector from center to face center
        to_face = poly.center - center

        # Check if normal points same direction as to_face
        if poly.normal.dot(to_face) > 0:
            outward += 1
        else:
            inward += 1

    total = inward + outward
    inward_pct = (inward / total * 100) if total > 0 else 0

    return {
        'inward': inward,
        'outward': outward,
        'total': total,
        'inward_pct': inward_pct,
        'outward_pct': 100 - inward_pct
    }


def verify_normals(obj, report=None):
    """
    Verify and report normal orientation.

    Args:
        obj: Blender mesh object
        report: Optional report list

    Returns:
        True if normals look correct (mostly outward)
    """
    stats = check_normal_orientation(obj)

    log(f"[Normals] {obj.name}: {stats['outward']} outward, {stats['inward']} inward ({stats['inward_pct']:.1f}% inverted)", report)

    if stats['inward_pct'] > 10:
        log(f"[Normals] WARNING: High percentage of inverted normals!", report)
        return False

    return True


# ============================================================================
# NORMAL FIXING
# ============================================================================

def fix_normals(obj, method="BLENDER", threshold=0, report=None):
    """
    Fix face normals to point outward.

    Args:
        obj: Blender mesh object
        method: "BLENDER" (topology-based), "DIRECTION" (center-based), or "BOTH"
        threshold: Only fix if inverted percentage exceeds this value
        report: Optional report list

    Returns:
        True if normals were fixed
    """
    ensure_object_mode()

    # Check current state
    stats = check_normal_orientation(obj)

    if stats['inward_pct'] <= threshold:
        log(f"[Normals] {obj.name}: {stats['inward_pct']:.1f}% inverted (below threshold {threshold}%), skipping", report)
        return False

    log(f"[Normals] {obj.name}: {stats['inward_pct']:.1f}% inverted, fixing...", report)

    method = method.upper()

    if method == "BLENDER":
        return _fix_normals_blender(obj, report)

    elif method == "DIRECTION":
        return _fix_normals_by_direction(obj, report)

    elif method == "BOTH":
        # Try Blender first
        _fix_normals_blender(obj, report)

        # Check if still bad
        stats = check_normal_orientation(obj)
        if stats['inward_pct'] > 10:
            log(f"[Normals] Still {stats['inward_pct']:.1f}% inverted, trying direction method...", report)
            _fix_normals_by_direction(obj, report)

        return True

    else:
        log(f"[Normals] Unknown method: {method}", report)
        return False


def _fix_normals_blender(obj, report):
    """Fix normals using Blender's topology-based method."""
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)

    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Normals] {obj.name}: Applied Blender normals_make_consistent", report)
    return True


def _fix_normals_by_direction(obj, report):
    """Fix normals by flipping faces that point toward mesh center."""
    ensure_object_mode()

    with step_timer("Fixing normals by direction"):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        # Calculate mesh center
        center = Vector((0, 0, 0))
        for v in bm.verts:
            center += v.co
        center /= len(bm.verts)

        flipped = 0

        for face in bm.faces:
            to_face = face.calc_center_median() - center

            # If normal points toward center, flip it
            if face.normal.dot(to_face) < 0:
                face.normal_flip()
                flipped += 1

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        log(f"[Normals] {obj.name}: Flipped {flipped} faces by direction", report)
        return flipped > 0


def fix_normals_after_remesh(obj, method="BOTH", report=None):
    """
    Fix normals after remesh operation.

    Remeshing can sometimes create inverted normals, especially voxel remesh
    which can create double-walled geometry with ~50% inverted.

    Args:
        obj: Blender mesh object
        method: Fix method to use
        report: Optional report list

    Returns:
        True if normals were fixed
    """
    stats = check_normal_orientation(obj)

    log(f"[Post-Remesh Normals] {obj.name}: {stats['inward_pct']:.1f}% inverted", report)

    if stats['inward_pct'] > 5:
        return fix_normals(obj, method=method, threshold=0, report=report)

    return False


# ============================================================================
# SMOOTH/FLAT SHADING
# ============================================================================

def set_flat_shading(obj, report=None):
    """
    Set mesh to flat shading.

    Args:
        obj: Blender mesh object
        report: Optional report list
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_flat()

    log(f"[Shading] {obj.name}: Set to flat shading", report)


def set_smooth_shading(obj, report=None):
    """
    Set mesh to smooth shading.

    Args:
        obj: Blender mesh object
        report: Optional report list
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    log(f"[Shading] {obj.name}: Set to smooth shading", report)


def set_auto_smooth(obj, angle=30.0, report=None):
    """
    Enable auto-smooth with specified angle.

    Args:
        obj: Blender mesh object
        angle: Angle threshold in degrees
        report: Optional report list
    """
    import math

    ensure_object_mode()

    mesh = obj.data

    # Blender 4.1+ uses different API
    if hasattr(mesh, 'use_auto_smooth'):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.radians(angle)
    else:
        # Blender 4.1+ - auto smooth is per-object
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))

    log(f"[Shading] {obj.name}: Auto-smooth enabled at {angle}°", report)
