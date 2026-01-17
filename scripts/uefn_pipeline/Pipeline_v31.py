"""
Pipeline V31 - Skip pose bone rotation, align at object level only

KEY INSIGHT: UEFN and H3D skeletons have fundamentally different bone conventions:
- UEFN: Bones point +Y (forward in bone local space)
- H3D/Mixamo: Bones point anatomically (spine=+Z, legs=-Z, arms=+X/-X)

Both are already in T-pose/A-pose at REST. Rotating pose bones to match directions
causes the mesh to deform because the bones have ~90° different orientations.

SOLUTION:
1. Scale to match
2. Translate/yaw armature objects to align
3. DON'T rotate pose bones - the mesh is already in the right pose
4. Bake geometry as-is
5. Transfer weights from Source mannequin
6. Rebind to Source skeleton
"""

import bpy
import math
from datetime import datetime
from mathutils import Vector, Matrix

LOG_TEXT_NAME = "Pipeline_V31_Log.txt"

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

def pick_mesh_driven_by_armature(col, arm_obj):
    meshes = mesh_objects(col)
    if not meshes:
        raise RuntimeError(f"No mesh objects found in collection '{col.name}'.")

    driven = []
    for m in meshes:
        am = get_armature_modifier(m)
        if am and am.object == arm_obj:
            driven.append(m)

    if driven:
        driven.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
        return driven[0], "armature_modifier_match"

    meshes.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes[0], "fallback_largest_mesh"

# ---------------- Mode / depsgraph helpers ----------------
def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def depsgraph_update():
    bpy.context.view_layer.update()

# ---------------- Transform helpers ----------------
def clear_parent_keep_transform(obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

def apply_scale_to_objects(objs):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    depsgraph_update()

def apply_all_transforms(obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    depsgraph_update()

# ---------------- Measurement ----------------
def robust_z_height_world(mesh_obj, max_samples=200000, p_low=0.01, p_high=0.99):
    arm_mod = get_armature_modifier(mesh_obj)
    prev_vis = None
    if arm_mod:
        prev_vis = arm_mod.show_viewport
        arm_mod.show_viewport = False
        depsgraph_update()

    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = mesh_obj.evaluated_get(dg)
    me = obj_eval.to_mesh()
    try:
        n = len(me.vertices)
        if n == 0:
            raise RuntimeError(f"Mesh '{mesh_obj.name}' has 0 vertices.")
        step = max(1, n // max_samples) if n > max_samples else 1

        zs = []
        mw = obj_eval.matrix_world
        for i in range(0, n, step):
            zs.append((mw @ me.vertices[i].co).z)

        zs.sort()
        lo_i = int(p_low * (len(zs) - 1))
        hi_i = int(p_high * (len(zs) - 1))
        return zs[hi_i] - zs[lo_i]
    finally:
        obj_eval.to_mesh_clear()
        if arm_mod and prev_vis is not None:
            arm_mod.show_viewport = prev_vis
            depsgraph_update()

# ---------------- Armature helpers ----------------
def set_armature_rest_and_clear_pose(arm_obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    arm_obj.data.pose_position = 'REST'
    depsgraph_update()

    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()

def bone_head_world(arm_obj, bone_name: str):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    return arm_obj.matrix_world @ b.head_local

# ---------------- Object-level alignment ----------------
def facing_forward_from_lr(arm_obj, left_bone, right_bone):
    L = bone_head_world(arm_obj, left_bone)
    R = bone_head_world(arm_obj, right_bone)
    if L is None or R is None:
        return None

    right = (R - L)
    right.z = 0.0
    if right.length < 1e-6:
        return None
    right.normalize()

    up = Vector((0, 0, 1))
    fwd = up.cross(right)
    if fwd.length < 1e-6:
        return None
    return fwd.normalized()

def yaw_angle_about_z(from_vec: Vector, to_vec: Vector):
    a = Vector((from_vec.x, from_vec.y, 0.0))
    b = Vector((to_vec.x, to_vec.y, 0.0))
    if a.length < 1e-8 or b.length < 1e-8:
        return 0.0
    a.normalize()
    b.normalize()
    cross_z = a.cross(b).z
    dot = max(-1.0, min(1.0, a.dot(b)))
    return math.atan2(cross_z, dot)

def align_armature_object(src_arm, tgt_arm, src_pelvis, tgt_hips, report):
    """Align target armature OBJECT to source via translate + yaw."""
    # Translate
    src_p = bone_head_world(src_arm, src_pelvis)
    tgt_p = bone_head_world(tgt_arm, tgt_hips)
    if src_p is None or tgt_p is None:
        raise RuntimeError(f"Missing pelvis/hips bone.")

    offset = src_p - tgt_p
    tgt_arm.location += offset
    depsgraph_update()
    report.append(f"[Align] Translate: ({offset.x:.4f}, {offset.y:.4f}, {offset.z:.4f})")

    # Yaw
    src_fwd = facing_forward_from_lr(src_arm, "clavicle_l", "clavicle_r")
    tgt_fwd = facing_forward_from_lr(tgt_arm, "LeftShoulder", "RightShoulder")

    if src_fwd is None or tgt_fwd is None:
        src_fwd = facing_forward_from_lr(src_arm, "thigh_l", "thigh_r")
        tgt_fwd = facing_forward_from_lr(tgt_arm, "LeftUpLeg", "RightUpLeg")

    if src_fwd and tgt_fwd:
        yaw = yaw_angle_about_z(tgt_fwd, src_fwd)
        tgt_arm.rotation_mode = 'XYZ'
        tgt_arm.rotation_euler.z += yaw
        depsgraph_update()
        report.append(f"[Align] Yaw: {math.degrees(yaw):.2f}°")
    else:
        report.append("[Align] Yaw skipped (missing landmarks)")

# ---------------- Bake / Clean ----------------
def apply_armature_modifier(mesh_obj, arm_obj):
    """Bake current pose into mesh geometry."""
    ensure_object_mode()
    depsgraph_update()

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    mod = None
    for m in mesh_obj.modifiers:
        if m.type == 'ARMATURE' and m.object == arm_obj:
            mod = m
            break
    if not mod:
        raise RuntimeError(f"No Armature modifier on '{mesh_obj.name}'.")

    bpy.ops.object.modifier_apply(modifier=mod.name)

def clear_vertex_groups(mesh_obj):
    if mesh_obj.type == "MESH":
        mesh_obj.vertex_groups.clear()

def remove_armature_modifiers(mesh_obj):
    if mesh_obj.type != "MESH":
        return
    for m in list(mesh_obj.modifiers):
        if m.type == "ARMATURE":
            mesh_obj.modifiers.remove(m)

# ---------------- Weight Transfer ----------------
def create_vertex_groups_from_armature(mesh_obj, arm_obj, report):
    """
    Create empty vertex groups on mesh for each bone in the armature.
    This is REQUIRED before Data Transfer can copy weights.
    """
    existing = {vg.name for vg in mesh_obj.vertex_groups}
    created = 0

    for bone in arm_obj.data.bones:
        if bone.name not in existing:
            mesh_obj.vertex_groups.new(name=bone.name)
            created += 1

    report.append(f"[VGroups] Created {created} vertex groups from armature bones.")
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

    For correct bind pose, the mesh must be parented to the armature with
    the mesh's local transform at identity (0,0,0 location, no rotation, scale 1).
    This ensures the bind pose matrices are calculated correctly for FBX export.
    """
    ensure_object_mode()

    arm_obj.data.pose_position = 'REST'
    depsgraph_update()

    # Remove existing armature modifiers
    for m in list(mesh_obj.modifiers):
        if m.type == "ARMATURE":
            mesh_obj.modifiers.remove(m)

    # CRITICAL FOR BIND POSE:
    # The mesh vertices are currently in world space (after apply_all_transforms).
    # We need to transform them into the armature's local space so that
    # when parented, the mesh has identity local transform.
    #
    # IMPORTANT: Only apply location and rotation, NOT scale!
    # The armature might have scene-level scale that we don't want to invert.

    # Decompose armature's world matrix
    loc, rot, scale = arm_obj.matrix_world.decompose()

    # Create transform matrix with only location and rotation (no scale)
    loc_rot_matrix = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
    arm_world_inv = loc_rot_matrix.inverted()

    report.append(f"[Bind] Armature world: loc={loc}, scale={scale}")

    # Transform all mesh vertices from world space to armature local space
    me = mesh_obj.data
    for v in me.vertices:
        # Vertex is in world space (mesh has identity transform)
        # Transform to armature local space (location + rotation only)
        v.co = arm_world_inv @ v.co

    me.update()
    report.append("[Bind] Transformed mesh vertices to armature local space (preserving scale).")

    # Now parent mesh to armature using ARMATURE type (not OBJECT)
    # This explicitly establishes the armature deform binding for proper bind pose
    # Note: type='ARMATURE' automatically adds an Armature modifier
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    # Use 'ARMATURE' type - keeps existing vertex groups, establishes proper bind
    bpy.ops.object.parent_set(type='ARMATURE', keep_transform=False)
    depsgraph_update()
    report.append("[Bind] Parented mesh to armature with ARMATURE type (proper bind pose).")

    # Configure the armature modifier (created by parent_set)
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
    report = []
    report.append("Pipeline V31 - Object-Level Alignment Only")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    src_col = find_collection_ci("Source")
    tgt_col = find_collection_ci("Target")
    if not src_col or not tgt_col:
        raise RuntimeError("Missing 'Source' and/or 'Target' collections.")

    src_arm = find_single_armature(src_col)
    tgt_arm = find_single_armature(tgt_col)

    src_mesh, src_pick = pick_mesh_driven_by_armature(src_col, src_arm)
    tgt_mesh, tgt_pick = pick_mesh_driven_by_armature(tgt_col, tgt_arm)

    report.append(f"Source Armature: {src_arm.name}")
    report.append(f"Source Mesh:     {src_mesh.name} [{src_pick}]")
    report.append(f"Target Armature: {tgt_arm.name}")
    report.append(f"Target Mesh:     {tgt_mesh.name} [{tgt_pick}]\n")

    # ---- Step 0: Prep ----
    report.append("[Step0] Target -> REST + clear pose...")
    set_armature_rest_and_clear_pose(tgt_arm)
    set_armature_rest_and_clear_pose(src_arm)

    if tgt_mesh.parent == tgt_arm:
        report.append("[Step0] Clearing Target mesh parent...")
        clear_parent_keep_transform(tgt_mesh)

    # Scale
    report.append("[Step0] Measuring heights...")
    src_h = robust_z_height_world(src_mesh)
    tgt_h = robust_z_height_world(tgt_mesh)
    if tgt_h < 1e-6:
        raise RuntimeError("Target height near zero.")

    scale_factor = src_h / tgt_h
    report.append(f"[Step0] Source: {src_h:.4f}, Target: {tgt_h:.4f}, Scale: {scale_factor:.4f}")

    tgt_arm.scale *= scale_factor
    tgt_mesh.scale *= scale_factor
    depsgraph_update()

    apply_scale_to_objects([tgt_mesh, tgt_arm])
    report.append("[Step0] Scale applied.\n")

    # ---- Step 1: Object-level alignment ----
    report.append("[Step1] Aligning Target armature object...")
    align_armature_object(src_arm, tgt_arm, "pelvis", "Hips", report)
    report.append("[Step1] Done.\n")

    # ---- Step 2: Bake geometry ----
    report.append("[Step2] Baking Target mesh (applying armature modifier)...")
    apply_armature_modifier(tgt_mesh, tgt_arm)
    report.append("[Step2] Baked.\n")

    # ---- Step 3: Clean ----
    report.append("[Step3] Cleaning Target mesh...")
    remove_armature_modifiers(tgt_mesh)
    clear_vertex_groups(tgt_mesh)
    apply_all_transforms(tgt_mesh)
    report.append("[Step3] Done.\n")

    # ---- Step 4: Create vertex groups + Transfer weights ----
    report.append("[Step4] Creating vertex groups on Target mesh...")
    create_vertex_groups_from_armature(tgt_mesh, src_arm, report)

    report.append("[Step4] Transferring weights from Source mesh...")
    transfer_weights_via_modifier(src_mesh, tgt_mesh, report)
    report.append("[Step4] Done.\n")

    # ---- Step 5: Bind to Source armature ----
    report.append("[Step5] Binding Target mesh to Source armature...")
    bind_mesh_to_armature(tgt_mesh, src_arm, report)
    report.append("[Step5] Done.\n")

    # ---- Step 6: Create Export collection (no container - armature at root) ----
    report.append("[Step6] Creating Export collection...")

    # Create or get Export collection
    export_col = bpy.data.collections.get("Export")
    if not export_col:
        export_col = bpy.data.collections.new("Export")
        bpy.context.scene.collection.children.link(export_col)
    report.append("[Step6] Export collection ready.")

    # Clear armature's parent so it's at root level (no container empty)
    if src_arm.parent:
        bpy.ops.object.select_all(action='DESELECT')
        src_arm.select_set(True)
        bpy.context.view_layer.objects.active = src_arm
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        depsgraph_update()
        report.append(f"[Step6] Cleared armature parent (now at root level).")

    # Remove source mesh from armature's children (unparent it)
    # Keep only our target mesh as child of the armature
    for child in list(src_arm.children):
        if child.type == 'MESH' and child != tgt_mesh:
            bpy.ops.object.select_all(action='DESELECT')
            child.select_set(True)
            bpy.context.view_layer.objects.active = child
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            report.append(f"[Step6] Removed source mesh '{child.name}' from armature children.")
    depsgraph_update()

    # Note: mesh is already parented to armature in Step 5 (bind_mesh_to_armature)
    # with proper bind pose (identity local transform)

    # Link armature and mesh to Export collection
    if src_arm.name not in [o.name for o in export_col.objects]:
        export_col.objects.link(src_arm)
    if tgt_mesh.name not in [o.name for o in export_col.objects]:
        export_col.objects.link(tgt_mesh)

    report.append(f"[Step6] Created hierarchy: {src_arm.name} > {tgt_mesh.name}")

    # Hide Source and Target collections
    src_col.hide_viewport = True
    tgt_col.hide_viewport = True
    report.append("[Step6] Hidden Source/Target collections.")

    report.append("[Step6] Done.\n")

    report.append("Pipeline V31 complete.")
    report.append("Export: Select 'Export' collection and export FBX.")
    report.append("IMPORTANT: Disable 'Add Leaf Bones' in FBX export to avoid _end bones!")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log: {LOG_TEXT_NAME} ---")

main()
