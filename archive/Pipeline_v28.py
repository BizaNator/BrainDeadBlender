import bpy
import math
from mathutils import Vector, Matrix, Quaternion
from datetime import datetime

LOG_TEXT_NAME = "Pipeline_V28_Log.txt"

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

def pick_mesh_driven_by_armature_STRICT(col, arm_obj):
    """
    STRICT: requires at least one mesh whose Armature modifier points to arm_obj.
    Avoids 'largest mesh' mistakes when imports contain junk meshes.
    """
    meshes = mesh_objects(col)
    if not meshes:
        raise RuntimeError(f"No mesh objects found in collection '{col.name}'.")

    driven = []
    for m in meshes:
        am = get_armature_modifier(m)
        if am and am.object == arm_obj:
            driven.append(m)

    if not driven:
        names = [m.name for m in meshes]
        raise RuntimeError(
            f"No mesh in '{col.name}' is driven by armature '{arm_obj.name}'.\n"
            f"Meshes present: {names}\n"
            f"Fix: ensure your intended mesh has an Armature modifier targeting that armature."
        )

    driven.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return driven[0], "armature_modifier_match"

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

# ---------------- Measurement (robust height) ----------------
def robust_z_height_world(mesh_obj, max_samples=200000, p_low=0.01, p_high=0.99):
    """
    Measures robust height in WORLD space, with this mesh's Armature modifier disabled (if present),
    so pose/skeleton doesn’t affect measurement.
    """
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

# ---------------- Bone helpers ----------------
def bone_world_head(arm_obj, bone_name):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    return arm_obj.matrix_world @ b.head_local

def facing_forward_world_from_lr(arm_obj, left_bone, right_bone):
    """
    Build a stable forward direction using L/R landmarks.
    right = R - L (projected to XY)
    forward = Zup x right
    """
    L = bone_world_head(arm_obj, left_bone)
    R = bone_world_head(arm_obj, right_bone)
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
    fwd.normalize()
    return fwd

def yaw_angle_about_z(from_vec: Vector, to_vec: Vector):
    """
    Signed yaw angle (radians) to rotate from 'from_vec' to 'to_vec' about world Z.
    Uses atan2 on cross/dot, stable for 3D vectors projected to XY.
    """
    a = Vector((from_vec.x, from_vec.y, 0.0))
    b = Vector((to_vec.x, to_vec.y, 0.0))
    if a.length < 1e-8 or b.length < 1e-8:
        return 0.0
    a.normalize(); b.normalize()
    cross_z = a.cross(b).z
    dot = max(-1.0, min(1.0, a.dot(b)))
    return math.atan2(cross_z, dot)

def align_target_translate_and_yaw(
    src_arm, tgt_arm,
    src_pelvis="pelvis", tgt_hips="Hips",
    report=None,
    **_ignored_kwargs  # <-- IMPORTANT: prevents "unexpected keyword" crashes
):
    # --- translate by pelvis/hips ---
    src_p = bone_world_head(src_arm, src_pelvis)
    tgt_p = bone_world_head(tgt_arm, tgt_hips)
    if src_p is None or tgt_p is None:
        raise RuntimeError(f"Missing pelvis/hips head for translate: {src_pelvis} / {tgt_hips}")

    offset = src_p - tgt_p
    tgt_arm.location += offset
    depsgraph_update()
    if report is not None:
        report.append(f"[Align] Translate by pelvis offset: ({offset.x:.4f},{offset.y:.4f},{offset.z:.4f})")

    # --- yaw from shoulders (preferred), fallback to thighs ---
    src_fwd = facing_forward_world_from_lr(src_arm, "clavicle_l", "clavicle_r")
    tgt_fwd = facing_forward_world_from_lr(tgt_arm, "LeftShoulder", "RightShoulder")

    if src_fwd is None or tgt_fwd is None:
        src_fwd = facing_forward_world_from_lr(src_arm, "thigh_l", "thigh_r")
        tgt_fwd = facing_forward_world_from_lr(tgt_arm, "LeftUpLeg", "RightUpLeg")

    if src_fwd is None or tgt_fwd is None:
        if report is not None:
            report.append("[Align] WARNING: yaw align skipped (missing shoulder/thigh landmarks).")
        return False

    ang = yaw_angle_about_z(tgt_fwd, src_fwd)
    tgt_arm.rotation_mode = 'XYZ'
    tgt_arm.rotation_euler.z += ang
    depsgraph_update()
    if report is not None:
        report.append(f"[Align] Yaw-only around Z (shoulders/thighs): {ang:.4f} rad")
    return True

# ---------------- Pose alignment (FRAME + explicit UP) ----------------
def make_frame_blender_bone(forward: Vector, up_hint: Vector) -> Matrix:
    """
    Build orthonormal basis matching Blender bone convention:
      +Y = forward (head->tail)
      +Z = up
      +X = right
    Returns a 3x3 rotation matrix.
    """
    y = forward.normalized()

    u = up_hint
    if u.length < 1e-8:
        u = Vector((0, 0, 1))
    u = u.normalized()

    # Make Z orthogonal to Y
    z = (u - y * y.dot(u))
    if z.length < 1e-8:
        z = Vector((0, 0, 1))
        if abs(y.dot(z)) > 0.95:
            z = Vector((0, 1, 0))
        z = (z - y * y.dot(z))
    z.normalize()

    x = y.cross(z).normalized()
    z = x.cross(y).normalized()

    return Matrix((x, y, z)).transposed()

def align_pose_bone_frame_explicit_up(src_arm, tgt_arm,
                                     src_bone, tgt_bone,
                                     src_up_bone, tgt_up_bone,
                                     report):
    """
    Align target pose bone orientation so that:
      - forward = head->tail matches source forward
      - up hint = (up_bone_head - bone_head) matches source up hint
    Applies via pose bone matrix_basis (local delta).
    """
    sb = src_arm.data.bones.get(src_bone)
    tb = tgt_arm.data.bones.get(tgt_bone)
    su = src_arm.data.bones.get(src_up_bone)
    tu = tgt_arm.data.bones.get(tgt_up_bone)

    if not sb or not tb:
        report.append(f"  [MISS] bone missing: {src_bone}->{tgt_bone}")
        return False
    if not su or not tu:
        report.append(f"  [MISS] up ref missing: {src_up_bone}->{tgt_up_bone}")
        return False

    src_f = (sb.tail_local - sb.head_local)
    tgt_f = (tb.tail_local - tb.head_local)
    if src_f.length < 1e-8 or tgt_f.length < 1e-8:
        report.append(f"  [MISS] forward too small: {src_bone}->{tgt_bone}")
        return False

    src_up = (su.head_local - sb.head_local)
    tgt_up = (tu.head_local - tb.head_local)
    if src_up.length < 1e-8 or tgt_up.length < 1e-8:
        report.append(f"  [MISS] up vector too small: {src_up_bone}->{tgt_up_bone}")
        return False

    src_R = make_frame_blender_bone(src_f, src_up)
    tgt_R = make_frame_blender_bone(tgt_f, tgt_up)
    delta_R = src_R @ tgt_R.inverted()

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if not pb:
        report.append(f"  [MISS] pose bone missing: {tgt_bone}")
        return False

    pb.matrix_basis = (delta_R.to_4x4() @ pb.matrix_basis)
    report.append(f"  [OK] FRAME {src_bone}->{tgt_bone} (up {src_up_bone}->{tgt_up_bone})")
    return True


# ============================ ARM Twists ==========================================
def _bone_head_tail_local(arm_obj, bone_name):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None, None
    return b.head_local.copy(), b.tail_local.copy()

def _plane_normal_from_chain(arm_obj, upper, lower, hand):
    # Use heads to define shoulder->elbow and elbow->wrist vectors in ARMATURE LOCAL space
    u_h, _ = _bone_head_tail_local(arm_obj, upper)
    l_h, _ = _bone_head_tail_local(arm_obj, lower)
    h_h, _ = _bone_head_tail_local(arm_obj, hand)
    if None in (u_h, l_h, h_h):
        return None

    v1 = (l_h - u_h)   # shoulder -> elbow
    v2 = (h_h - l_h)   # elbow -> wrist
    if v1.length < 1e-6 or v2.length < 1e-6:
        return None

    n = v1.cross(v2)
    if n.length < 1e-6:
        return None
    return n.normalized()

def _hand_forward_axis(arm_obj, hand, hand_end=None, fallback_child=True):
    # Prefer hand->hand_end if available; otherwise use bone tail-head (bone direction)
    h_h, h_t = _bone_head_tail_local(arm_obj, hand)
    if h_h is None:
        return None

    if hand_end:
        e_h, _ = _bone_head_tail_local(arm_obj, hand_end)
        if e_h is not None:
            f = (e_h - h_h)
            if f.length > 1e-6:
                return f.normalized()

    f = (h_t - h_h)
    if f.length > 1e-6:
        return f.normalized()

    return None

def twist_hand_to_match_arm_plane(src_arm, tgt_arm,
                                  src_upper, src_lower, src_hand,
                                  tgt_upper, tgt_lower, tgt_hand,
                                  tgt_hand_end=None,
                                  report=None):
    """
    Applies ONLY a twist about the target hand forward axis so the arm-bend plane matches source.
    Call AFTER you already aligned upper/lower/hand direction.
    """
    n_src = _plane_normal_from_chain(src_arm, src_upper, src_lower, src_hand)
    n_tgt = _plane_normal_from_chain(tgt_arm, tgt_upper, tgt_lower, tgt_hand)
    if n_src is None or n_tgt is None:
        if report is not None:
            report.append(f"  [MISS] twist plane normal missing for {src_hand}->{tgt_hand}")
        return False

    f_tgt = _hand_forward_axis(tgt_arm, tgt_hand, hand_end=tgt_hand_end)
    if f_tgt is None:
        if report is not None:
            report.append(f"  [MISS] target hand forward axis missing: {tgt_hand}")
        return False

    # Project normals onto plane perpendicular to forward axis (so we measure only roll)
    n_src_p = (n_src - f_tgt * n_src.dot(f_tgt))
    n_tgt_p = (n_tgt - f_tgt * n_tgt.dot(f_tgt))
    if n_src_p.length < 1e-6 or n_tgt_p.length < 1e-6:
        if report is not None:
            report.append(f"  [MISS] projected normals too small (straight arm?) {src_hand}->{tgt_hand}")
        return False

    n_src_p.normalize()
    n_tgt_p.normalize()

    # Signed angle around forward axis to rotate target plane normal into source plane normal
    angle = math.atan2(f_tgt.dot(n_tgt_p.cross(n_src_p)), n_tgt_p.dot(n_src_p))


    pb = tgt_arm.pose.bones.get(tgt_hand)
    if pb is None:
        if report is not None:
            report.append(f"  [MISS] pose bone missing: {tgt_hand}")
        return False

    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = Quaternion(f_tgt, angle) @ pb.rotation_quaternion

    if report is not None:
        report.append(f"  [OK] TWIST {src_hand}->{tgt_hand} angle={angle:.4f} rad")
    return True

# ============================ MAIN (Step0 + Step1 only) ============================
def main():
    report = []
    report.append("UEFN → H3D Pipeline TEST (Step0 + Step1 only)")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    src_col = find_collection_ci("Source")
    tgt_col = find_collection_ci("Target")
    if not src_col or not tgt_col:
        raise RuntimeError("Missing 'Source' and/or 'Target' collections (case-insensitive).")

    src_arm = find_single_armature(src_col)
    tgt_arm = find_single_armature(tgt_col)

    src_mesh, src_pick = pick_mesh_driven_by_armature_STRICT(src_col, src_arm)
    tgt_mesh, tgt_pick = pick_mesh_driven_by_armature_STRICT(tgt_col, tgt_arm)

    report.append(f"Source Armature: {src_arm.name}")
    report.append(f"Source Mesh:     {src_mesh.name} [{src_pick}]")
    report.append(f"Target Armature: {tgt_arm.name}")
    report.append(f"Target Mesh:     {tgt_mesh.name} [{tgt_pick}]\n")

    # ---- Step 0: Prep Target deterministically ----
    report.append("[Step0] Target -> REST + clear pose...")
    set_armature_rest_and_clear_pose(tgt_arm)

    if tgt_mesh.parent == tgt_arm:
        report.append("[Step0] Clearing Target mesh parent (keep transform) to avoid double scale...")
        clear_parent_keep_transform(tgt_mesh)

    report.append("[Step0] Measuring robust heights (armature modifiers disabled)...")
    src_h = robust_z_height_world(src_mesh)
    tgt_h = robust_z_height_world(tgt_mesh)
    if tgt_h < 1e-6:
        raise RuntimeError("Target robust height near zero; cannot scale.")

    scale_factor = src_h / tgt_h
    report.append(f"[Step0] Source robust height: {src_h:.6f}")
    report.append(f"[Step0] Target robust height: {tgt_h:.6f}")
    report.append(f"[Step0] Scaling Target mesh+armature by: {scale_factor:.6f}")

    tgt_arm.scale *= scale_factor
    tgt_mesh.scale *= scale_factor
    depsgraph_update()

    report.append("[Step0] Applying scale to Target mesh+armature (reset to 1.0)...")
    apply_scale_to_objects([tgt_mesh, tgt_arm])
    report.append("[Step0] Done.\n")

    # ---- Align: translate + yaw at OBJECT level only ----
    report.append("[Align] Translate + yaw-align Target armature OBJECT (world Z only)...")
    align_target_translate_and_yaw(
        src_arm, tgt_arm,
        src_pelvis="pelvis", tgt_hips="Hips",
        report=report
    )
    report.append("[Align] Done.\n")

    # ---- Step 1: Pose align (rotation-only) ----
    report.append("[PoseAlign] Switching Target to POSE position + clearing pose...")
    set_armature_pose_and_clear_pose(tgt_arm)

    report.append("[PoseAlign] Aligning Target pose bones (frame+explicit up)...")
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode='POSE')

    ok_count = 0
    miss_count = 0

    TASKS = [
        # Left hand (explicit up)
        ("hand_l", "LeftHand", "index_metacarpal_l", "LeftHand_end"),

        # Right hand (later)
        # ("hand_r", "RightHand", "index_metacarpal_r", "RightHand_end"),
    ]

    for (s_b, t_b, s_up, t_up) in TASKS:
        ok = align_pose_bone_frame_explicit_up(
            src_arm, tgt_arm,
            src_bone=s_b, tgt_bone=t_b,
            src_up_bone=s_up, tgt_up_bone=t_up,
            report=report
        )
        ok_count += 1 if ok else 0
        miss_count += 0 if ok else 1

    # Twist AFTER direction alignment (Blender 5-safe signed angle)
    twist_hand_to_match_arm_plane(
        src_arm, tgt_arm,
        src_upper="upperarm_l", src_lower="lowerarm_l", src_hand="hand_l",
        tgt_upper="LeftArm",    tgt_lower="LeftForeArm", tgt_hand="LeftHand",
        tgt_hand_end="LeftHand_end",
        report=report
    )

    # Uncomment when ready
    # twist_hand_to_match_arm_plane(
    #     src_arm, tgt_arm,
    #     src_upper="upperarm_r", src_lower="lowerarm_r", src_hand="hand_r",
    #     tgt_upper="RightArm",   tgt_lower="RightForeArm", tgt_hand="RightHand",
    #     tgt_hand_end="RightHand_end",
    #     report=report
    # )

    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()
    report.append(f"[PoseAlign] Done. OK={ok_count} MISS={miss_count}\n")

main()
