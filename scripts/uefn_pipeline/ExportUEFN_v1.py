"""
ExportUEFN V1 - Export FBX for UEFN/Unreal

Exports the Export collection with UEFN-compatible settings:
- No leaf bones (avoids _end bones in Unreal)
- Z up, Y forward (Unreal coordinate system)
- No animation data
- Vertex colors as linear
- Smoothing groups enabled
"""

import bpy
import os
from datetime import datetime

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            CONFIGURATION                                   ║
# ║                     Edit these settings before running                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Collection to export
EXPORT_COLLECTION = "Export"

# Output filename (without .fbx extension)
# Leave empty to use the blend file name
OUTPUT_NAME = ""

# Output directory
# Leave empty to use the same folder as the blend file
OUTPUT_DIR = ""

# Scale factor (1.0 = no change, Blender uses meters, Unreal uses cm)
EXPORT_SCALE = 1.0

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         END CONFIGURATION                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

LOG_TEXT_NAME = "ExportUEFN_V1_Log.txt"


# ---------------- Logging ----------------
def log_to_text(s: str):
    txt = bpy.data.texts.get(LOG_TEXT_NAME)
    if not txt:
        txt = bpy.data.texts.new(LOG_TEXT_NAME)
    txt.clear()
    txt.write(s)


# ---------------- Helpers ----------------
def find_collection_ci(name: str):
    want = name.strip().lower()
    for col in bpy.data.collections:
        if col.name.strip().lower() == want:
            return col
    return None


def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def force_armature_name_root(arm, report):
    """Force armature to be named exactly 'root' - no numbers allowed."""
    if arm.name == "root" and arm.data.name == "root":
        return  # Already correct

    report.append(f"[Validate] Armature is '{arm.name}', must be exactly 'root'.")

    # First, rename any OTHER objects using "root" name
    for obj in list(bpy.data.objects):
        if obj != arm and (obj.name == "root" or obj.name.startswith("root.")):
            old_name = obj.name
            obj.name = "_old_" + old_name
            report.append(f"[Validate] Renamed conflicting object '{old_name}' to '{obj.name}'")

    # Rename any OTHER armature data using "root" name
    for arm_data in list(bpy.data.armatures):
        if arm_data != arm.data and (arm_data.name == "root" or arm_data.name.startswith("root.")):
            old_name = arm_data.name
            arm_data.name = "_old_" + old_name
            report.append(f"[Validate] Renamed conflicting armature data '{old_name}' to '{arm_data.name}'")

    # Now we can safely rename our armature
    arm.name = "root"
    arm.data.name = "root"

    # Verify
    if arm.name != "root":
        raise RuntimeError(f"CRITICAL: Failed to rename armature to 'root'. Got '{arm.name}'.")

    report.append(f"[Validate] Renamed armature to 'root'.")


def validate_export_collection(col, report):
    """Validate the export collection has proper structure."""
    objects = list(col.all_objects)

    if not objects:
        raise RuntimeError(f"Collection '{col.name}' is empty.")

    # Find armature
    armatures = [o for o in objects if o.type == "ARMATURE"]
    meshes = [o for o in objects if o.type == "MESH"]

    if not armatures:
        raise RuntimeError(f"No armature found in '{col.name}' collection.")

    if len(armatures) > 1:
        report.append(f"[Validate] WARNING: Multiple armatures found, using first one.")

    arm = armatures[0]

    # CRITICAL: Force armature name to exactly "root"
    force_armature_name_root(arm, report)

    # Check for root bone or armature at root level
    root_bones = [b for b in arm.data.bones if b.parent is None]
    report.append(f"[Validate] Armature: {arm.name}")
    report.append(f"[Validate] Root bones: {[b.name for b in root_bones]}")
    report.append(f"[Validate] Meshes: {[m.name for m in meshes]}")

    # Check mesh is parented to armature
    for mesh in meshes:
        if mesh.parent != arm:
            report.append(f"[Validate] WARNING: Mesh '{mesh.name}' not parented to armature.")

    return arm, meshes


def select_collection_objects(col):
    """Select all objects in collection."""
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')

    for obj in col.all_objects:
        obj.select_set(True)

    # Set active object to armature if present
    for obj in col.all_objects:
        if obj.type == "ARMATURE":
            bpy.context.view_layer.objects.active = obj
            break


def get_output_path(report):
    """Determine the output file path."""
    # Get base name
    if OUTPUT_NAME:
        base_name = OUTPUT_NAME
    elif bpy.data.filepath:
        base_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    else:
        base_name = "UEFN_Export"

    # Get directory
    if OUTPUT_DIR:
        out_dir = OUTPUT_DIR
    elif bpy.data.filepath:
        out_dir = os.path.dirname(bpy.data.filepath)
    else:
        out_dir = os.path.expanduser("~")
        report.append(f"[Export] WARNING: Blend file not saved, using home directory.")

    # Create full path
    filepath = os.path.join(out_dir, f"{base_name}.fbx")

    return filepath


# ============================ MAIN ============================
def main():
    """Export the Export collection as FBX for UEFN."""
    report = []
    report.append("ExportUEFN V1 - FBX Export for UEFN")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Find export collection
    export_col = find_collection_ci(EXPORT_COLLECTION)
    if not export_col:
        raise RuntimeError(f"Collection '{EXPORT_COLLECTION}' not found.")

    report.append(f"[Setup] Export collection: {export_col.name}")

    # Validate structure
    arm, meshes = validate_export_collection(export_col, report)
    report.append("[Validate] Structure OK.\n")

    # Get output path
    filepath = get_output_path(report)
    report.append(f"[Export] Output: {filepath}\n")

    # Select objects for export
    select_collection_objects(export_col)
    report.append(f"[Export] Selected {len(list(export_col.all_objects))} objects.")

    # Export FBX
    report.append("[Export] Exporting FBX with UEFN settings...")
    report.append("")
    report.append("  FBX Settings:")
    report.append("  - Selected objects only")
    report.append("  - Apply scalings: FBX All")
    report.append("  - Forward: -Y")
    report.append("  - Up: Z")
    report.append("  - Apply unit scale: ON")
    report.append("  - Mesh smoothing: Face")
    report.append("  - Vertex colors: Linear")
    report.append("  - Add leaf bones: OFF")
    report.append("  - Bake animation: OFF")
    report.append("")

    bpy.ops.export_scene.fbx(
        filepath=filepath,

        # Selection
        use_selection=True,
        use_visible=False,
        use_active_collection=False,

        # Scale
        global_scale=EXPORT_SCALE,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',

        # Orientation (Unreal: Z up, -Y forward)
        axis_forward='-Y',
        axis_up='Z',

        # Objects
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,
        use_mesh_modifiers_render=True,

        # Mesh settings
        mesh_smooth_type='FACE',  # Smoothing groups
        use_subsurf=False,
        use_mesh_edges=False,
        use_tspace=False,

        # Vertex colors - Linear (sRGB disabled)
        colors_type='LINEAR',

        # Armature settings
        use_armature_deform_only=False,
        add_leaf_bones=False,  # CRITICAL: No _end bones
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        armature_nodetype='NULL',

        # Animation - disabled
        bake_anim=False,
        bake_anim_use_all_bones=False,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=False,

        # Other
        batch_mode='OFF',
        use_batch_own_dir=False,
        use_metadata=True,
    )

    report.append(f"[Export] SUCCESS! Exported to:")
    report.append(f"         {filepath}")
    report.append("")
    report.append("=" * 50)
    report.append("ExportUEFN V1 complete!")
    report.append("")
    report.append("Import into Unreal:")
    report.append("  1. Right-click in Content Browser > Import")
    report.append("  2. Select the FBX file")
    report.append("  3. In import dialog, set Skeleton to your UEFN skeleton")
    report.append("  4. Bind pose warning is expected and OK")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log saved to: {LOG_TEXT_NAME} ---")

    return filepath


# Run
if __name__ == "__main__":
    main()
