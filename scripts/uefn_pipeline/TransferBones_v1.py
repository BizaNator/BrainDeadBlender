"""
TransferBones V1 - Rebind mesh to UEFN skeleton

Use Case:
- You have a mesh with UEFN vertex groups but lost the armature
- Both mesh and source skeleton are in the same pose/position

The script will:
1. Duplicate the UEFN armature
2. Optionally transfer weights from source mesh
3. Bind your mesh to the duplicated armature
4. Create Export collection with proper hierarchy
"""

import bpy
from mathutils import Matrix
from datetime import datetime

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            CONFIGURATION                                   ║
# ║                     Edit these settings before running                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Transfer weights from source mesh?
#   False = use existing vertex groups on target mesh (default)
#   True  = clear and re-transfer weights from SKM_UEFN_Mannequin
TRANSFER_WEIGHTS = False

# Collection names
SOURCE_COLLECTION = "Source"   # Must contain UEFN armature + mannequin mesh
TARGET_COLLECTION = "Target"   # Must contain your mesh

# Log file name (created in Blender's Text Editor)
LOG_TEXT_NAME = "TransferBones_V1_Log.txt"

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         END CONFIGURATION                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ---------------- Logging ----------------
def log_to_text(s: str):
    txt = bpy.data.texts.get(LOG_TEXT_NAME)
    if not txt:
        txt = bpy.data.texts.new(LOG_TEXT_NAME)
    txt.clear()
    txt.write(s)


# ---------------- Collection / Object helpers ----------------
def find_collection_ci(name: str):
    want = name.strip().lower()
    for col in bpy.data.collections:
        if col.name.strip().lower() == want:
            return col
    return None


def objects_in_collection(col):
    return list(col.all_objects)


def find_single_armature(col):
    arms = [o for o in objects_in_collection(col) if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"Collection '{col.name}' must contain exactly 1 armature; found {len(arms)}.")
    return arms[0]


def mesh_objects(col):
    return [o for o in objects_in_collection(col) if o.type == "MESH"]


def get_armature_modifier(mesh_obj):
    for m in mesh_obj.modifiers:
        if m.type == "ARMATURE" and getattr(m, "object", None):
            return m
    return None


def pick_mesh(col, arm_obj=None):
    """Pick the main mesh from collection."""
    meshes = mesh_objects(col)
    if not meshes:
        raise RuntimeError(f"No mesh objects found in collection '{col.name}'.")

    # If armature provided, prefer mesh driven by it
    if arm_obj:
        driven = []
        for m in meshes:
            am = get_armature_modifier(m)
            if am and am.object == arm_obj:
                driven.append(m)
        if driven:
            driven.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
            return driven[0]

    # Fallback to largest mesh
    meshes.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes[0]


# ---------------- Mode / depsgraph helpers ----------------
def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def depsgraph_update():
    bpy.context.view_layer.update()


# ---------------- Transform helpers ----------------
def apply_all_transforms(obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    depsgraph_update()


def clear_parent_keep_transform(obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')


# ---------------- Armature helpers ----------------
def set_armature_rest(arm_obj):
    ensure_object_mode()
    arm_obj.data.pose_position = 'REST'
    depsgraph_update()


def clear_name_for_root(report):
    """Remove any existing objects/data named 'root' so we can use that name."""
    # Rename any existing objects named "root" or "root.XXX"
    for obj in list(bpy.data.objects):
        if obj.name == "root" or obj.name.startswith("root."):
            old_name = obj.name
            obj.name = "_old_root_" + old_name
            report.append(f"[Cleanup] Renamed object '{old_name}' to '{obj.name}'")

    # Rename any existing armature data named "root"
    for arm_data in list(bpy.data.armatures):
        if arm_data.name == "root" or arm_data.name.startswith("root."):
            old_name = arm_data.name
            arm_data.name = "_old_root_" + old_name
            report.append(f"[Cleanup] Renamed armature data '{old_name}' to '{arm_data.name}'")


def duplicate_armature(arm_obj, report):
    """Duplicate armature for the target mesh."""
    ensure_object_mode()

    # Clear any existing "root" names first
    clear_name_for_root(report)

    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.duplicate()

    new_arm = bpy.context.active_object
    new_arm.name = "root"  # Must match UEFN skeleton naming exactly
    new_arm.data.name = "root"

    # Verify the name stuck
    if new_arm.name != "root":
        raise RuntimeError(f"Failed to name armature 'root', got '{new_arm.name}' instead.")

    report.append(f"[Duplicate] Created armature: {new_arm.name}")
    return new_arm


# ---------------- Weight Transfer ----------------
def create_vertex_groups_from_armature(mesh_obj, arm_obj, report):
    """Create empty vertex groups on mesh for each bone in the armature."""
    existing = {vg.name for vg in mesh_obj.vertex_groups}
    created = 0

    for bone in arm_obj.data.bones:
        if bone.name not in existing:
            mesh_obj.vertex_groups.new(name=bone.name)
            created += 1

    report.append(f"[VGroups] Created {created} new vertex groups from armature bones.")
    report.append(f"[VGroups] Total vertex groups: {len(mesh_obj.vertex_groups)}")
    return created


def transfer_weights_via_modifier(source_mesh, target_mesh, report):
    """Transfer vertex weights from source to target using Data Transfer modifier."""
    ensure_object_mode()
    depsgraph_update()

    bpy.ops.object.select_all(action='DESELECT')
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh

    mod = target_mesh.modifiers.new(name="DT_Weights", type='DATA_TRANSFER')
    mod.object = source_mesh

    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}

    # Set vertex mapping
    available_map = [e.identifier for e in mod.bl_rna.properties['vert_mapping'].enum_items]
    preferred = ["POLYINTERP_NEAREST", "NEAREST"]
    mod.vert_mapping = next((v for v in preferred if v in available_map), available_map[0])

    mod.mix_mode = 'REPLACE'
    mod.mix_factor = 1.0

    report.append(f"[Weights] vert_mapping={mod.vert_mapping}")

    bpy.ops.object.modifier_apply(modifier=mod.name)
    report.append("[Weights] Transfer complete.")


# ---------------- Bind ----------------
def bind_mesh_to_armature(mesh_obj, arm_obj, report):
    """
    Bind mesh to armature with proper bind pose.

    Transforms mesh vertices to armature local space so the mesh
    has identity local transform when parented.
    """
    ensure_object_mode()

    arm_obj.data.pose_position = 'REST'
    depsgraph_update()

    # Remove existing armature modifiers
    for m in list(mesh_obj.modifiers):
        if m.type == "ARMATURE":
            mesh_obj.modifiers.remove(m)

    # Clear any existing parent
    if mesh_obj.parent:
        clear_parent_keep_transform(mesh_obj)
        depsgraph_update()

    # Apply transforms to mesh first
    apply_all_transforms(mesh_obj)

    # Decompose armature's world matrix - only use location + rotation, NOT scale
    loc, rot, scale = arm_obj.matrix_world.decompose()
    loc_rot_matrix = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
    arm_world_inv = loc_rot_matrix.inverted()

    report.append(f"[Bind] Armature world: loc={loc}, scale={scale}")

    # Transform all mesh vertices from world space to armature local space
    me = mesh_obj.data
    for v in me.vertices:
        v.co = arm_world_inv @ v.co

    me.update()
    report.append("[Bind] Transformed mesh vertices to armature local space.")

    # Parent mesh to armature with ARMATURE type
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE', keep_transform=False)
    depsgraph_update()
    report.append("[Bind] Parented mesh to armature with ARMATURE type.")

    # Configure the armature modifier
    arm_mod = None
    for m in mesh_obj.modifiers:
        if m.type == "ARMATURE":
            arm_mod = m
            break

    if arm_mod:
        arm_mod.use_vertex_groups = True
        arm_mod.use_bone_envelopes = False
        report.append("[Bind] Configured Armature modifier.")
    else:
        report.append("[Bind] WARNING: No Armature modifier found after parenting!")

    depsgraph_update()


# ============================ MAIN ============================
def main():
    """Main function to transfer bones to target mesh."""
    do_transfer = TRANSFER_WEIGHTS

    report = []
    report.append("TransferBones V1 - Rebind Mesh to UEFN Skeleton")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Transfer Weights: {do_transfer}\n")

    # Find collections
    src_col = find_collection_ci(SOURCE_COLLECTION)
    tgt_col = find_collection_ci(TARGET_COLLECTION)
    if not src_col:
        raise RuntimeError(f"Missing '{SOURCE_COLLECTION}' collection.")
    if not tgt_col:
        raise RuntimeError(f"Missing '{TARGET_COLLECTION}' collection.")

    # Find source armature and mesh
    src_arm = find_single_armature(src_col)
    src_mesh = pick_mesh(src_col, src_arm)

    report.append(f"Source Armature: {src_arm.name}")
    report.append(f"Source Mesh:     {src_mesh.name}")

    # Find target mesh (no armature expected)
    tgt_mesh = pick_mesh(tgt_col)
    report.append(f"Target Mesh:     {tgt_mesh.name}")

    # Check existing vertex groups
    existing_vgroups = len(tgt_mesh.vertex_groups)
    report.append(f"Target Vertex Groups: {existing_vgroups}\n")

    # ---- Step 1: Prep ----
    report.append("[Step1] Setting armatures to REST pose...")
    set_armature_rest(src_arm)
    report.append("[Step1] Done.\n")

    # ---- Step 2: Duplicate armature ----
    report.append("[Step2] Duplicating Source armature...")
    new_arm = duplicate_armature(src_arm, report)
    report.append("[Step2] Done.\n")

    # ---- Step 3: Weights ----
    if do_transfer:
        report.append("[Step3] Clearing existing vertex groups...")
        tgt_mesh.vertex_groups.clear()

        report.append("[Step3] Creating vertex groups from armature...")
        create_vertex_groups_from_armature(tgt_mesh, new_arm, report)

        report.append("[Step3] Transferring weights from Source mesh...")
        transfer_weights_via_modifier(src_mesh, tgt_mesh, report)
    else:
        report.append("[Step3] Using existing vertex groups (TRANSFER_WEIGHTS=False)")

        # Ensure all bone vertex groups exist
        create_vertex_groups_from_armature(tgt_mesh, new_arm, report)

    report.append("[Step3] Done.\n")

    # ---- Step 4: Bind ----
    report.append("[Step4] Binding mesh to armature...")
    bind_mesh_to_armature(tgt_mesh, new_arm, report)
    report.append("[Step4] Done.\n")

    # ---- Step 5: Export collection ----
    report.append("[Step5] Creating Export collection...")

    export_col = bpy.data.collections.get("Export")
    if not export_col:
        export_col = bpy.data.collections.new("Export")
        bpy.context.scene.collection.children.link(export_col)

    # Clear armature's parent so it's at root level
    if new_arm.parent:
        bpy.ops.object.select_all(action='DESELECT')
        new_arm.select_set(True)
        bpy.context.view_layer.objects.active = new_arm
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        depsgraph_update()

    # Link to Export collection
    if new_arm.name not in [o.name for o in export_col.objects]:
        export_col.objects.link(new_arm)
    if tgt_mesh.name not in [o.name for o in export_col.objects]:
        export_col.objects.link(tgt_mesh)

    report.append(f"[Step5] Created hierarchy: {new_arm.name} > {tgt_mesh.name}")

    # Optionally hide source collection
    # src_col.hide_viewport = True
    # tgt_col.hide_viewport = True

    report.append("[Step5] Done.\n")

    report.append("=" * 50)
    report.append("TransferBones V1 complete!")
    report.append("")
    report.append("Export Hierarchy:")
    report.append(f"  root [ARMATURE]")
    report.append(f"    └── {tgt_mesh.name} [MESH]")
    report.append("")
    report.append("Next: Export FBX from 'Export' collection")
    report.append("IMPORTANT: Disable 'Add Leaf Bones' in FBX export!")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log saved to: {LOG_TEXT_NAME} ---")


# Run with default settings
if __name__ == "__main__":
    main()
