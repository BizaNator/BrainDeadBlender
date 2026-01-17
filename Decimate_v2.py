"""
Decimate V2 - Stylized Low-Poly Reduction (Modular Version)

Orchestrates mesh_ops modules to reduce high-poly meshes to stylized low-poly.

This version uses the modular mesh_ops package for all operations,
making individual functions available for ComfyUI nodes and addon panels.

PIPELINE:
    1. [Optional] Bake texture to vertex colors (preserves color through remesh)
    2. [Optional] Remesh (sharp/voxel/quad)
    3. [Optional] Fill holes
    4. [Optional] Remove internal geometry
    5. [Optional] Fix normals
    6. Planar decimate (merge coplanar faces)
    7. Triangulate n-gons
    8. Collapse decimate (reduce to target)
    9. Mark sharp edges
    10. Transfer colors from reference (if remeshed)
    11. Finalize color attribute

USAGE:
    # Method 1: Full pipeline with config
    from Decimate_v2 import run_pipeline
    run_pipeline(obj, target_faces=5000, remesh_mode="SHARP")

    # Method 2: Individual operations via mesh_ops
    from mesh_ops import remesh, cleanup, decimate, colors

    remesh.apply_sharp_remesh(obj, octree_depth=8)
    cleanup.fill_holes(obj)
    decimate.apply_planar_decimate(obj, angle=7.0)
    decimate.apply_collapse_decimate(obj, target_faces=5000)

    # Method 3: Configure and run
    CONFIG['TARGET_FACES'] = 3000
    CONFIG['REMESH_MODE'] = "NONE"
    main()
"""

import bpy
from datetime import datetime

# Import modular operations
from mesh_ops import utils, colors, remesh, cleanup, normals, decimate

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Target mesh: "ACTIVE" or collection name
    "TARGET_MESH": "ACTIVE",

    # Remesh mode: "NONE", "SHARP", "VOXEL", "VOXEL_HIGH", "QUAD"
    "REMESH_MODE": "NONE",
    "SHARP_OCTREE_DEPTH": 8,
    "SHARP_THRESHOLD": 1.0,
    "VOXEL_SIZE": None,  # None = auto-calculate
    "VOXEL_TARGET_POLYS": 100000,
    "VOXEL_HIGH_TARGET": 1000000,

    # Hole filling
    "FILL_HOLES": True,
    "FILL_HOLES_MAX_SIDES": 100,

    # Internal geometry removal
    "REMOVE_INTERNAL": True,
    "INTERNAL_REMOVAL_METHOD": "RAYCAST",  # "RAYCAST" or "SIMPLE"
    "INTERNAL_RAY_SAMPLES": 14,

    # Normal fixing
    "FIX_NORMALS": "AUTO",  # "AUTO", True, or False
    "FIX_NORMALS_THRESHOLD": 0,
    "FIX_NORMALS_METHOD": "BOTH",  # "BLENDER", "DIRECTION", or "BOTH"

    # Decimation
    "TARGET_FACES": 5000,
    "PLANAR_ANGLE": 7.0,
    "SHARP_ANGLE": 14.0,
    "PRESERVE_BOUNDARIES": True,

    # Color preservation
    "PRESERVE_COLORS_THROUGH_REMESH": True,
    "BAKE_VERTEX_COLORS": True,
    "DETECT_COLOR_EDGES": True,
    "DETECT_COLOR_EDGES_WHEN": "BEFORE",  # "BEFORE" or "AFTER"
    "COLOR_EDGE_THRESHOLD": 0.15,
    "COLOR_SOURCE": "TEXTURE",  # "TEXTURE", "VERTEX_COLOR", or "MATERIAL"

    # Pre-cleanup
    "PRE_CLEANUP": True,
    "FIX_NON_MANIFOLD": True,
    "AGGRESSIVE_MANIFOLD_FIX": True,

    # Output
    "CREATE_DEBUG_MATERIAL": True,
}

LOG_TEXT_NAME = "Decimate_V2_Log.txt"


# ============================================================================
# PIPELINE
# ============================================================================

def run_pipeline(obj, report=None, **kwargs):
    """
    Run the full decimation pipeline on a mesh.

    Args:
        obj: Blender mesh object
        report: Optional report list for logging
        **kwargs: Override CONFIG values

    Returns:
        True on success
    """
    if report is None:
        report = []

    # Merge kwargs into config
    config = CONFIG.copy()
    config.update(kwargs)

    report.append("=" * 60)
    report.append("Decimate V2 - Modular Pipeline")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)

    utils.log(f"\n[Pipeline] Processing: {obj.name}", report)
    utils.print_mesh_stats(obj, report)

    # Track reference mesh for color transfer
    color_ref_obj = None
    needs_color_transfer = False

    try:
        # ================================================================
        # STEP 1: Pre-cleanup
        # ================================================================
        if config["PRE_CLEANUP"]:
            utils.log("\n[Step 1] Pre-cleanup...", report)
            cleanup.pre_cleanup_mesh(
                obj,
                fix_manifold=config["FIX_NON_MANIFOLD"],
                aggressive=config["AGGRESSIVE_MANIFOLD_FIX"],
                report=report
            )

        # ================================================================
        # STEP 2: Bake colors before remesh
        # ================================================================
        if config["BAKE_VERTEX_COLORS"] and config["REMESH_MODE"] != "NONE":
            utils.log("\n[Step 2] Baking texture to vertex colors...", report)
            colors.bake_texture_to_vertex_colors(obj, output_name="BakedColors", report=report)

            if config["PRESERVE_COLORS_THROUGH_REMESH"]:
                utils.log("[Step 2] Creating color reference copy...", report)
                color_ref_obj = colors.create_color_reference_copy(obj, report)
                needs_color_transfer = True

        # ================================================================
        # STEP 3: Color edge detection (BEFORE decimation)
        # ================================================================
        if config["DETECT_COLOR_EDGES"] and config["DETECT_COLOR_EDGES_WHEN"] == "BEFORE":
            utils.log("\n[Step 3] Detecting color edges...", report)
            colors.detect_color_edges(
                obj,
                threshold=config["COLOR_EDGE_THRESHOLD"],
                source=config["COLOR_SOURCE"],
                report=report
            )

        # ================================================================
        # STEP 4: Remesh
        # ================================================================
        if config["REMESH_MODE"] != "NONE":
            utils.log(f"\n[Step 4] Remesh ({config['REMESH_MODE']})...", report)

            if config["REMESH_MODE"] == "SHARP":
                remesh.apply_sharp_remesh(
                    obj,
                    octree_depth=config["SHARP_OCTREE_DEPTH"],
                    sharpness=config["SHARP_THRESHOLD"],
                    report=report
                )
            elif config["REMESH_MODE"] == "VOXEL":
                remesh.apply_voxel_remesh(
                    obj,
                    voxel_size=config["VOXEL_SIZE"],
                    target_polys=config["VOXEL_TARGET_POLYS"],
                    report=report
                )
            elif config["REMESH_MODE"] == "VOXEL_HIGH":
                remesh.apply_voxel_high_remesh(
                    obj,
                    target_faces=config["VOXEL_HIGH_TARGET"],
                    report=report
                )
            elif config["REMESH_MODE"] == "QUAD":
                remesh.apply_quadriflow_remesh(
                    obj,
                    target_faces=config["TARGET_FACES"],
                    report=report
                )

            # Fix normals after remesh
            normals.fix_normals_after_remesh(obj, method=config["FIX_NORMALS_METHOD"], report=report)

        # ================================================================
        # STEP 5: Fill holes
        # ================================================================
        if config["FILL_HOLES"]:
            utils.log("\n[Step 5] Filling holes...", report)
            cleanup.fill_holes(obj, max_sides=config["FILL_HOLES_MAX_SIDES"], report=report)

        # ================================================================
        # STEP 6: Remove internal geometry
        # ================================================================
        if config["REMOVE_INTERNAL"]:
            utils.log("\n[Step 6] Removing internal geometry...", report)
            cleanup.remove_internal_geometry(
                obj,
                method=config["INTERNAL_REMOVAL_METHOD"],
                ray_samples=config["INTERNAL_RAY_SAMPLES"],
                report=report
            )

        # ================================================================
        # STEP 7: Fix normals
        # ================================================================
        if config["FIX_NORMALS"]:
            utils.log("\n[Step 7] Fixing normals...", report)
            threshold = config["FIX_NORMALS_THRESHOLD"] if config["FIX_NORMALS"] == "AUTO" else 0
            normals.fix_normals(
                obj,
                method=config["FIX_NORMALS_METHOD"],
                threshold=threshold,
                report=report
            )

        # ================================================================
        # STEP 8: Planar decimate
        # ================================================================
        if config["PLANAR_ANGLE"] > 0:
            utils.log(f"\n[Step 8] Planar decimate (angle={config['PLANAR_ANGLE']}°)...", report)
            decimate.apply_planar_decimate(obj, angle=config["PLANAR_ANGLE"], report=report)

        # ================================================================
        # STEP 9: Triangulate n-gons
        # ================================================================
        utils.log("\n[Step 9] Triangulating n-gons...", report)
        cleanup.triangulate_ngons(obj, report=report)

        # ================================================================
        # STEP 10: Collapse decimate
        # ================================================================
        if config["TARGET_FACES"] > 0:
            utils.log(f"\n[Step 10] Collapse decimate (target={config['TARGET_FACES']})...", report)
            decimate.apply_collapse_decimate(
                obj,
                target_faces=config["TARGET_FACES"],
                preserve_boundaries=config["PRESERVE_BOUNDARIES"],
                report=report
            )

        # ================================================================
        # STEP 11: Mark sharp edges
        # ================================================================
        if config["SHARP_ANGLE"] > 0:
            utils.log(f"\n[Step 11] Marking sharp edges (angle={config['SHARP_ANGLE']}°)...", report)
            decimate.mark_sharp_edges(obj, angle=config["SHARP_ANGLE"], report=report)

        # ================================================================
        # STEP 12: Color edge detection (AFTER decimation)
        # ================================================================
        if config["DETECT_COLOR_EDGES"] and config["DETECT_COLOR_EDGES_WHEN"] == "AFTER":
            utils.log("\n[Step 12] Detecting color edges (post-decimate)...", report)
            colors.detect_color_edges(
                obj,
                threshold=config["COLOR_EDGE_THRESHOLD"],
                source=config["COLOR_SOURCE"],
                report=report
            )

        # ================================================================
        # STEP 13: Transfer colors from reference
        # ================================================================
        if needs_color_transfer and color_ref_obj:
            utils.log("\n[Step 13] Transferring colors from reference...", report)
            colors.transfer_vertex_colors(color_ref_obj, obj, output_name="TransferredColors", report=report)

        # ================================================================
        # STEP 14: Finalize color attribute
        # ================================================================
        utils.log("\n[Step 14] Finalizing color attribute...", report)
        colors.finalize_color_attribute(obj, target_name="Col", report=report)

        # ================================================================
        # STEP 15: Create debug material
        # ================================================================
        if config["CREATE_DEBUG_MATERIAL"]:
            utils.log("\n[Step 15] Creating debug material...", report)
            colors.create_vertex_color_material(obj, layer_name="Col", report=report)

        # ================================================================
        # DONE
        # ================================================================
        report.append("\n" + "=" * 60)
        report.append("Pipeline complete!")
        utils.print_mesh_stats(obj, report)
        report.append("=" * 60)

        # Write log
        utils.log_to_text("\n".join(report), LOG_TEXT_NAME)

        return True

    except Exception as e:
        utils.log(f"\n[ERROR] Pipeline failed: {e}", report)
        import traceback
        utils.log(traceback.format_exc(), report)
        utils.log_to_text("\n".join(report), LOG_TEXT_NAME)
        return False

    finally:
        # Cleanup reference object
        if color_ref_obj:
            colors.cleanup_color_reference(color_ref_obj, report)


# ============================================================================
# ENTRY POINTS
# ============================================================================

def get_target_meshes():
    """Get meshes to process based on CONFIG."""
    target = CONFIG["TARGET_MESH"]

    if target == "ACTIVE":
        obj = bpy.context.active_object
        if obj and obj.type == 'MESH':
            return [obj]
        raise RuntimeError("No active mesh selected")

    # Collection name
    meshes = utils.find_meshes_in_collection(target)
    if meshes:
        return meshes

    raise RuntimeError(f"Collection '{target}' not found or has no meshes")


def main():
    """
    Main entry point. Processes meshes based on CONFIG.
    """
    report = []

    try:
        meshes = get_target_meshes()

        for obj in meshes:
            run_pipeline(obj, report)

    except Exception as e:
        utils.log(f"[ERROR] {e}", report)
        utils.log_to_text("\n".join(report), LOG_TEXT_NAME)


if __name__ == "__main__":
    main()
