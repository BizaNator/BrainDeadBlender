"""
mesh_ops.remesh - Remesh Operations

Functions for remeshing meshes using various methods.
"""

import bpy
import math
from mathutils import Vector

from .utils import (
    ensure_object_mode, get_face_count,
    log, step_timer
)

# ============================================================================
# SHARP REMESH (Octree-based)
# ============================================================================

def apply_sharp_remesh(obj, octree_depth=8, sharpness=1.0, apply_modifier=True, report=None):
    """
    Apply SHARP (octree-based) remesh - preserves sharp edges and thin geometry.

    Best for thin geometry like lips, ears, fingers because it uses octree
    subdivision that respects sharp features rather than a uniform voxel grid.

    Args:
        obj: Blender mesh object
        octree_depth: Octree subdivision depth (6=~50K, 7=~200K, 8=~800K faces)
        sharpness: Edge sharpness threshold (0-1, higher = more sharp edges)
        apply_modifier: Whether to apply the modifier
        report: Optional report list

    Returns:
        Final face count
    """
    ensure_object_mode()

    with step_timer(f"Sharp remesh (depth={octree_depth})"):
        initial_faces = get_face_count(obj)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Add remesh modifier
        mod = obj.modifiers.new(name="Sharp_Remesh", type='REMESH')
        mod.mode = 'SHARP'
        mod.octree_depth = octree_depth
        mod.sharpness = sharpness
        mod.use_remove_disconnected = True
        mod.use_smooth_shade = False  # Flat shading for stylized look

        if apply_modifier:
            bpy.ops.object.modifier_apply(modifier=mod.name)

        final_faces = get_face_count(obj)
        log(f"[Sharp Remesh] {obj.name}: {initial_faces:,} -> {final_faces:,} faces", report)

        if final_faces > initial_faces:
            log(f"[Sharp Remesh] High-poly clean mesh created - will be reduced by decimation", report)

        return final_faces


# ============================================================================
# VOXEL REMESH
# ============================================================================

def calculate_auto_voxel_size(obj, target_polys):
    """
    Calculate voxel size to achieve approximately target_polys faces.

    Args:
        obj: Blender mesh object
        target_polys: Target polygon count

    Returns:
        Calculated voxel size
    """
    # Get bounding box dimensions
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_co = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_co = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))

    dimensions = max_co - min_co

    # Estimate surface area
    surface_area = 2 * (dimensions.x * dimensions.y +
                        dimensions.y * dimensions.z +
                        dimensions.x * dimensions.z)

    # voxel_size ≈ sqrt(surface_area / target_polys)
    if target_polys > 0 and surface_area > 0:
        voxel_size = math.sqrt(surface_area / target_polys)
        voxel_size = max(0.001, min(0.1, voxel_size))
        return voxel_size

    # Fallback
    min_dim = min(dimensions.x, dimensions.y, dimensions.z)
    return max(0.001, min_dim * 0.01)


def apply_voxel_remesh(obj, voxel_size=None, target_polys=100000, apply_modifier=True, report=None):
    """
    Apply voxel remesh - creates watertight mesh, fills holes.

    WARNING: Destroys thin geometry! Use apply_sharp_remesh for characters.

    Args:
        obj: Blender mesh object
        voxel_size: Voxel size (None = auto-calculate from target_polys)
        target_polys: Target polygon count (used if voxel_size is None)
        apply_modifier: Whether to apply the modifier
        report: Optional report list

    Returns:
        Final face count
    """
    ensure_object_mode()

    with step_timer("Voxel remesh"):
        initial_faces = get_face_count(obj)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Calculate voxel size if needed
        if voxel_size is None:
            actual_voxel_size = calculate_auto_voxel_size(obj, target_polys)
            log(f"[Voxel Remesh] Auto voxel size: {actual_voxel_size:.4f} (targeting ~{target_polys:,} polys)", report)
        else:
            actual_voxel_size = float(voxel_size)

        # Add remesh modifier
        mod = obj.modifiers.new(name="Voxel_Remesh", type='REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = actual_voxel_size
        mod.use_smooth_shade = False
        mod.adaptivity = 0.0

        if apply_modifier:
            bpy.ops.object.modifier_apply(modifier=mod.name)

        final_faces = get_face_count(obj)
        log(f"[Voxel Remesh] {obj.name}: {initial_faces:,} -> {final_faces:,} faces (voxel: {actual_voxel_size:.4f})", report)

        # Warn about thin geometry loss
        if initial_faces > 0:
            reduction_ratio = final_faces / initial_faces
            if reduction_ratio < 0.3:
                log(f"[Voxel Remesh] WARNING: Large reduction ({reduction_ratio:.1%}) - thin geometry may be lost!", report)

        return final_faces


def apply_voxel_high_remesh(obj, target_faces=1000000, voxel_size_override=None,
                            apply_modifier=True, report=None):
    """
    Apply high-resolution voxel remesh then decimate down.

    Creates clean watertight mesh with proper normals, but destroys thin geometry.

    Args:
        obj: Blender mesh object
        target_faces: Target face count for high-res mesh
        voxel_size_override: Manual voxel size (None = auto-calculate)
        apply_modifier: Whether to apply the modifier
        report: Optional report list

    Returns:
        Final face count
    """
    ensure_object_mode()

    with step_timer("Voxel high remesh"):
        initial_faces = get_face_count(obj)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        if voxel_size_override is not None:
            voxel_size = voxel_size_override
            log(f"[VOXEL_HIGH] Using manual voxel size: {voxel_size:.6f}", report)
        else:
            # Iterative approach: test and refine
            initial_estimate = calculate_auto_voxel_size(obj, target_faces)

            # Test with non-destructive modifier
            mod = obj.modifiers.new(name="Voxel_Test", type='REMESH')
            mod.mode = 'VOXEL'
            mod.voxel_size = initial_estimate
            mod.use_smooth_shade = False

            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(depsgraph)
            test_faces = len(obj_eval.data.polygons)

            obj.modifiers.remove(mod)

            log(f"[VOXEL_HIGH] Test: {test_faces:,} faces at voxel {initial_estimate:.6f}", report)

            # Adjust based on ratio
            if test_faces > 0 and target_faces > 0:
                ratio = math.sqrt(test_faces / target_faces)
                voxel_size = initial_estimate * ratio
                voxel_size = max(0.0001, min(0.1, voxel_size))
            else:
                voxel_size = initial_estimate

        log(f"[VOXEL_HIGH] WARNING: Geometry thinner than ~{voxel_size*2:.4f} units will be destroyed!", report)

        # Apply final remesh
        mod = obj.modifiers.new(name="Voxel_High_Remesh", type='REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = voxel_size
        mod.use_smooth_shade = False
        mod.adaptivity = 0.0

        if apply_modifier:
            bpy.ops.object.modifier_apply(modifier=mod.name)

        final_faces = get_face_count(obj)
        log(f"[VOXEL_HIGH] {obj.name}: {initial_faces:,} -> {final_faces:,} faces", report)

        accuracy = (final_faces / target_faces) * 100 if target_faces > 0 else 0
        log(f"[VOXEL_HIGH] Target accuracy: {accuracy:.1f}% of {target_faces:,} target", report)

        return final_faces


# ============================================================================
# QUADRIFLOW REMESH
# ============================================================================

def apply_quadriflow_remesh(obj, target_faces=10000, preserve_boundaries=True,
                            preserve_sharp=True, report=None):
    """
    Apply Quadriflow remesh - creates clean quad topology.

    NOTE: Creates smooth quad flow, NOT flat-faceted style.

    Args:
        obj: Blender mesh object
        target_faces: Target face count
        preserve_boundaries: Preserve mesh boundaries
        preserve_sharp: Preserve sharp edges
        report: Optional report list

    Returns:
        Final face count
    """
    ensure_object_mode()

    with step_timer("Quadriflow remesh"):
        initial_faces = get_face_count(obj)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Try different parameter sets for version compatibility
        try:
            bpy.ops.object.quadriflow_remesh(
                target_faces=target_faces,
                use_mesh_symmetry=False,
                use_preserve_sharp=preserve_sharp,
                use_preserve_boundary=preserve_boundaries,
            )
        except TypeError:
            try:
                bpy.ops.object.quadriflow_remesh(
                    target_faces=target_faces,
                    use_mesh_symmetry=False,
                    use_preserve_sharp=preserve_sharp,
                    use_preserve_boundary=preserve_boundaries,
                    preserve_paint_mask=False,
                    smooth_normals=False
                )
            except TypeError:
                bpy.ops.object.quadriflow_remesh(target_faces=target_faces)

        final_faces = get_face_count(obj)
        log(f"[Quadriflow] {obj.name}: {initial_faces:,} -> {final_faces:,} faces (target: {target_faces:,})", report)
        log(f"[Quadriflow] WARNING: Creates smooth quads, not flat-faceted style!", report)

        return final_faces


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def remesh(obj, mode="SHARP", target_faces=None, voxel_size=None,
           octree_depth=8, apply_modifier=True, report=None):
    """
    Remesh using specified mode.

    Args:
        obj: Blender mesh object
        mode: "SHARP", "VOXEL", "VOXEL_HIGH", "QUAD", or "NONE"
        target_faces: Target face count (for VOXEL_HIGH, QUAD)
        voxel_size: Voxel size (for VOXEL)
        octree_depth: Octree depth (for SHARP)
        apply_modifier: Whether to apply modifier
        report: Optional report list

    Returns:
        Final face count, or None if mode is "NONE"
    """
    mode = mode.upper()

    if mode == "NONE":
        log(f"[Remesh] Skipping remesh (mode=NONE)", report)
        return get_face_count(obj)

    elif mode == "SHARP":
        return apply_sharp_remesh(obj, octree_depth=octree_depth,
                                   apply_modifier=apply_modifier, report=report)

    elif mode == "VOXEL":
        return apply_voxel_remesh(obj, voxel_size=voxel_size,
                                   apply_modifier=apply_modifier, report=report)

    elif mode == "VOXEL_HIGH":
        return apply_voxel_high_remesh(obj, target_faces=target_faces or 1000000,
                                        apply_modifier=apply_modifier, report=report)

    elif mode == "QUAD":
        return apply_quadriflow_remesh(obj, target_faces=target_faces or 10000,
                                        report=report)

    else:
        log(f"[Remesh] Unknown mode: {mode}", report)
        return get_face_count(obj)
