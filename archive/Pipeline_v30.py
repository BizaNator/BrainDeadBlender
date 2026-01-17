import bpy
import math
from datetime import datetime
from mathutils import Vector, Matrix, Quaternion

LOG_TEXT_NAME = "Pipeline_V30_Log.txt"

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
    return meshes[0], "fallback_largest_mesh (WARNING: no armature-modifier match)"

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

# ---------------- Measurement (robust height) ----------------
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
        step = 1
        if n > max_samples:
            step = max(1, n // max_samples)

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

# ---------------- Pose prep / cleanup ----------------
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

def set_armature_pose_and_clear_pose(arm_obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    arm_obj.data.pose_position = 'POSE'
    depsgraph_update()

    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()

# ---------------- Bone direction queries (ARMATURE LOCAL space) ----------------
def bone_dir_local(arm_obj, bone_name: str):
    """
    Get bone direction (head->tail) in ARMATURE LOCAL space.
    This is independent of object transform and pose.
    """
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    v = b.tail_local - b.head_local
    if v.length < 1e-8:
        return None
    return v.normalized()

def bone_head_world(arm_obj, bone_name: str):
    """Get bone head in world space (rest pose)."""
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    return arm_obj.matrix_world @ b.head_local

# ---------------- Object-level alignment ----------------
def facing_forward_from_lr(arm_obj, left_bone, right_bone):
    """
    Compute forward direction on XY plane from left/right bone positions.
    Forward = up x right (where up = Z, right = R - L)
    """
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
    """Signed yaw angle (radians) from from_vec to to_vec around Z axis."""
    a = Vector((from_vec.x, from_vec.y, 0.0))
    b = Vector((to_vec.x, to_vec.y, 0.0))
    if a.length < 1e-8 or b.length < 1e-8:
        return 0.0
    a.normalize()
    b.normalize()
    cross_z = a.cross(b).z
    dot = max(-1.0, min(1.0, a.dot(b)))
    return math.atan2(cross_z, dot)

def align_target_translate_and_yaw(src_arm, tgt_arm,
                                   src_pelvis="pelvis", tgt_hips="Hips",
                                   report=None):
    """
    Align target armature OBJECT to source:
    1. Translate so hips match pelvis position
    2. Yaw-rotate around Z so facing direction matches
    """
    # Translate
    src_p = bone_head_world(src_arm, src_pelvis)
    tgt_p = bone_head_world(tgt_arm, tgt_hips)
    if src_p is None or tgt_p is None:
        raise RuntimeError(f"Missing pelvis/hips bone for translation alignment.")

    offset = src_p - tgt_p
    tgt_arm.location += offset
    depsgraph_update()
    if report is not None:
        report.append(f"[Align] Translate by: ({offset.x:.4f}, {offset.y:.4f}, {offset.z:.4f})")

    # Yaw from shoulders
    src_fwd = facing_forward_from_lr(src_arm, "clavicle_l", "clavicle_r")
    tgt_fwd = facing_forward_from_lr(tgt_arm, "LeftShoulder", "RightShoulder")

    # Fallback to thighs
    if src_fwd is None or tgt_fwd is None:
        src_fwd = facing_forward_from_lr(src_arm, "thigh_l", "thigh_r")
        tgt_fwd = facing_forward_from_lr(tgt_arm, "LeftUpLeg", "RightUpLeg")

    if src_fwd is None or tgt_fwd is None:
        if report is not None:
            report.append("[Align] WARNING: yaw align skipped (missing landmarks).")
        return False

    yaw = yaw_angle_about_z(tgt_fwd, src_fwd)
    tgt_arm.rotation_mode = 'XYZ'
    tgt_arm.rotation_euler.z += yaw
    depsgraph_update()

    if report is not None:
        report.append(f"[Align] Yaw around Z: {math.degrees(yaw):.2f}°")
    return True


# ============================================================================
# CORE FIX: Direction-only alignment using rotation_difference
#
# The key insight: We want to rotate the TARGET bone so its direction matches
# the SOURCE bone's direction. Both directions are in ARMATURE LOCAL space
# (because after yaw alignment, both armatures face the same way).
#
# We compute: rotation_difference(target_dir, source_dir)
# This gives us the DELTA rotation needed.
#
# Then we apply this as the pose bone's rotation (which is a delta from rest).
# ============================================================================

def align_bone_direction_only(src_arm, tgt_arm, src_bone, tgt_bone, report):
    """
    Rotate target pose bone so its direction matches source bone direction.

    IMPORTANT: This SETS the rotation, not accumulates it.
    Works because both armatures are already yaw-aligned at object level.
    """
    src_dir = bone_dir_local(src_arm, src_bone)
    tgt_dir = bone_dir_local(tgt_arm, tgt_bone)

    if src_dir is None:
        report.append(f"  [MISS] Source bone not found: {src_bone}")
        return False
    if tgt_dir is None:
        report.append(f"  [MISS] Target bone not found: {tgt_bone}")
        return False

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if pb is None:
        report.append(f"  [MISS] Pose bone not found: {tgt_bone}")
        return False

    # Rotation to align tgt_dir to src_dir (in armature local space)
    q = tgt_dir.rotation_difference(src_dir)

    # SET the rotation (don't accumulate)
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = q

    angle_deg = math.degrees(2 * math.acos(min(1.0, abs(q.w))))
    report.append(f"  [OK] {src_bone} -> {tgt_bone} (rot: {angle_deg:.1f}°)")
    return True


# ============================================================================
# Build hierarchy order for processing bones parent-first
# ============================================================================

def get_bone_depth(arm_obj, bone_name):
    """Get depth in hierarchy (0 = root)."""
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return 999
    depth = 0
    while b.parent:
        depth += 1
        b = b.parent
    return depth

def sort_by_hierarchy(arm_obj, bone_names):
    """Sort bone names so parents come before children."""
    return sorted(bone_names, key=lambda n: get_bone_depth(arm_obj, n))


# ============================ MAIN ============================
def main():
    report = []
    report.append("UEFN -> H3D Pipeline V30 (Direction-Only Alignment)")
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

    # ---- Step 0: Prep Target ----
    report.append("[Step0] Target -> REST + clear pose...")
    set_armature_rest_and_clear_pose(tgt_arm)

    if tgt_mesh.parent == tgt_arm:
        report.append("[Step0] Clearing Target mesh parent (keep transform)...")
        clear_parent_keep_transform(tgt_mesh)

    report.append("[Step0] Measuring heights...")
    src_h = robust_z_height_world(src_mesh)
    tgt_h = robust_z_height_world(tgt_mesh)
    if tgt_h < 1e-6:
        raise RuntimeError("Target height near zero.")

    scale_factor = src_h / tgt_h
    report.append(f"[Step0] Source height: {src_h:.4f}")
    report.append(f"[Step0] Target height: {tgt_h:.4f}")
    report.append(f"[Step0] Scale factor: {scale_factor:.4f}")

    tgt_arm.scale *= scale_factor
    tgt_mesh.scale *= scale_factor
    depsgraph_update()

    report.append("[Step0] Applying scale...")
    apply_scale_to_objects([tgt_mesh, tgt_arm])
    report.append("[Step0] Done.\n")

    # ---- Object-level alignment ----
    report.append("[Align] Translate + yaw Target armature...")
    align_target_translate_and_yaw(src_arm, tgt_arm, report=report)
    report.append("[Align] Done.\n")

    # ---- Step 1: Pose alignment ----
    report.append("[PoseAlign] Switch to POSE mode + clear...")
    set_armature_pose_and_clear_pose(tgt_arm)

    # Bone mapping: UEFN (source) -> H3D/Mixamo (target)
    BONE_MAP = {
        "pelvis": "Hips",
        "spine_01": "Spine",
        "spine_02": "Spine1",
        "spine_03": "Spine2",
        "neck_01": "Neck",
        "head": "Head",

        "clavicle_l": "LeftShoulder",
        "upperarm_l": "LeftArm",
        "lowerarm_l": "LeftForeArm",
        "hand_l": "LeftHand",

        "clavicle_r": "RightShoulder",
        "upperarm_r": "RightArm",
        "lowerarm_r": "RightForeArm",
        "hand_r": "RightHand",

        "thigh_l": "LeftUpLeg",
        "calf_l": "LeftLeg",
        "foot_l": "LeftFoot",
        "ball_l": "LeftToeBase",

        "thigh_r": "RightUpLeg",
        "calf_r": "RightLeg",
        "foot_r": "RightFoot",
        "ball_r": "RightToeBase",
    }

    # Sort target bones by hierarchy depth
    tgt_bones_sorted = sort_by_hierarchy(tgt_arm, list(BONE_MAP.values()))
    tgt_to_src = {v: k for k, v in BONE_MAP.items()}

    report.append("[PoseAlign] Aligning bones (direction-only, hierarchy order)...")

    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode='POSE')

    ok_count = 0
    miss_count = 0

    for tgt_bone in tgt_bones_sorted:
        src_bone = tgt_to_src.get(tgt_bone)
        if not src_bone:
            continue

        if align_bone_direction_only(src_arm, tgt_arm, src_bone, tgt_bone, report):
            ok_count += 1
        else:
            miss_count += 1

        # Update depsgraph after each bone so children see parent's new position
        depsgraph_update()

    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()

    report.append(f"[PoseAlign] Done. OK={ok_count}, MISS={miss_count}\n")
    report.append("Pipeline V30 complete (Step0 + Step1 only).")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log saved to: {LOG_TEXT_NAME} ---")

main()
