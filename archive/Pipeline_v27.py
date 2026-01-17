import bpy
import math
from mathutils import Vector, Matrix
from datetime import datetime

LOG_TEXT_NAME = "Pipeline_V27_Log.txt"

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
    This avoids 'largest mesh' picking mistakes when imports contain junk meshes.
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

def bone_world_dir(arm_obj, bone_name):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    h = arm_obj.matrix_world @ b.head_local
    t = arm_obj.matrix_world @ b.tail_local
    v = (t - h)
    if v.length < 1e-8:
        return None
    return v.normalized()

def facing_forward_world_from_lr(arm_obj, left_bone, right_bone):
    """
    Build a stable forward direction using L/R landmarks.
    right = R - L
    forward = Zup x right   (right-handed)
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
    fwd = up.cross(right)   # <- this defines “forward”
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
    report=None
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
        # fallback
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

def _rest_forward_and_up_hint(arm_obj, bone_name, up_bone_name=None):
    """
    Returns:
      forward: Vector (armature-local) = head->tail
      up_hint: Vector (armature-local) = (up_ref_head - bone_head) if available else world-up-like
    """
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None, None

    forward = (b.tail_local - b.head_local)
    if forward.length < 1e-8:
        return None, None

    # Choose an up reference bone
    up_ref = None
    if up_bone_name:
        up_ref = arm_obj.data.bones.get(up_bone_name)
    else:
        # auto-pick: first child if any
        if b.children:
            up_ref = b.children[0]

    if up_ref:
        up_hint = (up_ref.head_local - b.head_local)
        if up_hint.length < 1e-8:
            up_ref = None

    if not up_ref:
        # fallback up: world Z expressed in armature-local
        up_hint = arm_obj.matrix_world.to_3x3().inverted() @ Vector((0, 0, 1))
        if up_hint.length < 1e-8:
            up_hint = Vector((0, 0, 1))

    return forward, up_hint


def _make_bone_frame_Y_forward(forward: Vector, up_hint: Vector) -> Matrix:
    """
    Build orthonormal basis matching Blender bone convention:
      Y = forward (head->tail)
      Z = up
      X = right
    Returns 3x3 rotation matrix (columns are X,Y,Z).
    """
    y = forward.normalized()

    u = up_hint
    if u.length < 1e-8:
        u = Vector((0, 0, 1))
    u = u.normalized()

    # Make Z orthogonal to Y
    z = (u - y * y.dot(u))
    if z.length < 1e-8:
        # pick a fallback not parallel to y
        z = Vector((0, 0, 1))
        if abs(y.dot(z)) > 0.95:
            z = Vector((0, 1, 0))
        z = (z - y * y.dot(z))
    z.normalize()

    x = y.cross(z).normalized()
    z = x.cross(y).normalized()

    return Matrix((x, y, z)).transposed()

# ---------------- Frame-based pose rotation (rotation-only) ----------------
def arm_space_dir(arm_obj, bone_name):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    v = (b.tail_local - b.head_local)
    if v.length < 1e-8:
        return None
    return v.normalized()

def arm_space_head(arm_obj, bone_name):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    return b.head_local.copy()

def auto_child_up_bone_name(arm_obj, bone_name):
    """
    Pick a stable "up reference": first child in bone hierarchy (armature data bones).
    Returns None if no children.
    """
    b = arm_obj.data.bones.get(bone_name)
    if not b or not b.children:
        return None
    return b.children[0].name

from mathutils import Vector, Matrix

def make_frame_blender_bone(forward: Vector, up_hint: Vector) -> Matrix:
    """
    Build an orthonormal basis that matches Blender bone convention:
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
        # fallback up if parallel
        z = Vector((0, 0, 1))
        if abs(y.dot(z)) > 0.95:
            z = Vector((0, 1, 0))
        z = (z - y * y.dot(z))
    z.normalize()

    x = y.cross(z).normalized()
    z = x.cross(y).normalized()

    # Columns are basis vectors in local space: X, Y, Z
    return Matrix((x, y, z)).transposed()

def align_bone_with_up(src_arm, tgt_arm, src_bone, tgt_bone, src_up_bone, tgt_up_bone, report):
    """
    Compute a delta rotation that makes the TARGET bone's forward(+Y) and up(+Z)
    match the SOURCE bone's forward and up, using head->tail as forward and
    (up_bone_head - bone_head) as up hint.
    Applies the delta in pose space via matrix_basis.
    """
    sb = src_arm.data.bones.get(src_bone)
    tb = tgt_arm.data.bones.get(tgt_bone)
    if not sb or not tb:
        report.append(f"  [MISS] bone missing: {src_bone}->{tgt_bone}")
        return False

    # rest-space forward directions (armature local)
    src_f = (sb.tail_local - sb.head_local)
    tgt_f = (tb.tail_local - tb.head_local)
    if src_f.length < 1e-8 or tgt_f.length < 1e-8:
        report.append(f"  [MISS] forward too small: {src_bone}->{tgt_bone}")
        return False

    # up hints from head to 'up bone' head (armature local)
    su = src_arm.data.bones.get(src_up_bone)
    tu = tgt_arm.data.bones.get(tgt_up_bone)
    if not su or not tu:
        report.append(f"  [MISS] up ref missing: {src_up_bone}->{tgt_up_bone}")
        return False

    src_up = (su.head_local - sb.head_local)
    tgt_up = (tu.head_local - tb.head_local)
    if src_up.length < 1e-8 or tgt_up.length < 1e-8:
        report.append(f"  [MISS] up vector too small: {src_up_bone}->{tgt_up_bone}")
        return False

    src_R = make_frame_blender_bone(src_f, src_up)
    tgt_R = make_frame_blender_bone(tgt_f, tgt_up)

    # rotation that maps target->source in armature local
    delta_R = src_R @ tgt_R.inverted()

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if not pb:
        report.append(f"  [MISS] pose bone missing: {tgt_bone}")
        return False

    # Apply as local pose delta (matrix_basis is safest & avoids drift)
    pb.matrix_basis = (delta_R.to_4x4() @ pb.matrix_basis)

    report.append(f"  [OK] FRAME {src_bone}->{tgt_bone} (up {src_up_bone}->{tgt_up_bone})")
    return True


def align_pose_bone_by_frame(src_arm, tgt_arm, src_bone, tgt_bone,
                             src_up_bone=None, tgt_up_bone=None,
                             report=None):
    """
    Rotation-only alignment:
      - builds a rest-space frame for src and tgt using Y-forward + up hint
      - delta = src_frame * inv(tgt_frame)
      - applies delta to target pose bone's matrix_basis (local pose delta)
    """
    if report is None:
        report = []

    src_f, src_u = _rest_forward_and_up_hint(src_arm, src_bone, src_up_bone)
    tgt_f, tgt_u = _rest_forward_and_up_hint(tgt_arm, tgt_bone, tgt_up_bone)

    if src_f is None or tgt_f is None:
        report.append(f"  [MISS] forward missing: {src_bone}->{tgt_bone}")
        return False

    src_R = _make_bone_frame_Y_forward(src_f, src_u)
    tgt_R = _make_bone_frame_Y_forward(tgt_f, tgt_u)

    delta_R = src_R @ tgt_R.inverted()

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if not pb:
        report.append(f"  [MISS] pose bone missing: {tgt_bone}")
        return False

    # Apply delta in local pose space (stable)
    pb.matrix_basis = (delta_R.to_4x4() @ pb.matrix_basis)

    up_note = f"(auto)" if (src_up_bone is None or tgt_up_bone is None) else f"({src_up_bone}->{tgt_up_bone})"
    report.append(f"  [OK] FRAME {src_bone}->{tgt_bone} up{up_note}")
    return True
# ============================ MAIN ============================
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

    # Avoid double scale: target mesh must NOT be parented to target armature while scaling
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

    # Translate + yaw-align at OBJECT LEVEL only
    report.append("[Align] Translate + yaw-align Target armature OBJECT (world Z only)...")
    ok_yaw = align_target_translate_and_yaw(
        src_arm, tgt_arm,
        src_pelvis="pelvis", tgt_hips="Hips",
        report=report
    )

    report.append("[Align] Done.\n")

    # ---- Step 1: Pose align (rotation-only) ----
    report.append("[PoseAlign] Switching Target to POSE position + clearing pose...")
    set_armature_pose_and_clear_pose(tgt_arm)

    report.append("[PoseAlign] Aligning Target pose bones (frame+up)...")
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode='POSE')

    ok_count = 0
    miss_count = 0

    # IMPORTANT: pelvis/hips handled by OBJECT alignment above -> do NOT pose-align pelvis here.
    # Start with a tiny set while debugging, then expand.
    BONE_MAP = {
#         "spine_01": "Spine",
        "hand_l": "LeftHand",
    }

    for s_bone, t_bone in BONE_MAP.items():
        ok = align_pose_bone_by_frame(
            src_arm, tgt_arm,
            src_bone=s_bone, tgt_bone=t_bone,
            # Leave up bones None to AUTO-pick child bones (safer across rigs)
            src_up_bone=None, tgt_up_bone=None,
            report=report
        )
        ok_count += 1 if ok else 0
        miss_count += 0 if ok else 1

    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()
    report.append(f"[PoseAlign] Done. OK={ok_count} MISS={miss_count}\n")

    report.append("✅ Finished Step0 + Step1 only (no bake / no weights / no binding).")

    log_to_text("\n".join(report))
    print(f"✅ Done. See Text Editor: {LOG_TEXT_NAME}")

main()
