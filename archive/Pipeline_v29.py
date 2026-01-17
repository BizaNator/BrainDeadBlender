import bpy
import math
from mathutils import Vector, Matrix, Quaternion
from datetime import datetime

LOG_TEXT_NAME = "Pipeline_V29_Log.txt"

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
    **_ignored_kwargs
):
    src_p = bone_world_head(src_arm, src_pelvis)
    tgt_p = bone_world_head(tgt_arm, tgt_hips)
    if src_p is None or tgt_p is None:
        raise RuntimeError(f"Missing pelvis/hips head for translate: {src_pelvis} / {tgt_hips}")

    offset = src_p - tgt_p
    tgt_arm.location += offset
    depsgraph_update()
    if report is not None:
        report.append(f"[Align] Translate by pelvis offset: ({offset.x:.4f},{offset.y:.4f},{offset.z:.4f})")

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


# ============================================================================
# CORE FIX: World-space bone direction alignment with proper hierarchy handling
# ============================================================================

def get_bone_world_matrix(arm_obj, bone_name):
    """
    Get the CURRENT world-space matrix of a bone, accounting for pose.
    Works in both REST and POSE mode.
    """
    pb = arm_obj.pose.bones.get(bone_name)
    if pb is None:
        return None
    # pose bone matrix is in armature space, multiply by armature world
    return arm_obj.matrix_world @ pb.matrix

def get_bone_rest_world_direction(arm_obj, bone_name):
    """
    Get the REST pose direction (head->tail) in WORLD space.
    This is the direction the bone points when no pose is applied.
    """
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    # In rest pose, bone direction in armature local space
    local_dir = (b.tail_local - b.head_local)
    if local_dir.length < 1e-8:
        return None
    # Transform direction to world space (rotation only, no translation)
    world_dir = arm_obj.matrix_world.to_3x3() @ local_dir
    return world_dir.normalized()

def get_bone_posed_world_direction(arm_obj, bone_name):
    """
    Get the CURRENT (posed) bone direction in WORLD space.
    This accounts for any pose bone rotations.
    """
    pb = arm_obj.pose.bones.get(bone_name)
    if pb is None:
        return None
    # Bone's Y axis is head->tail direction in bone local space
    # pb.matrix is in armature space, so we need world transform
    bone_matrix_world = arm_obj.matrix_world @ pb.matrix
    # Y axis of the bone matrix is the direction
    world_dir = bone_matrix_world.to_3x3() @ Vector((0, 1, 0))
    return world_dir.normalized()

def get_bone_posed_world_head(arm_obj, bone_name):
    """Get the world position of bone head in current pose."""
    pb = arm_obj.pose.bones.get(bone_name)
    if pb is None:
        return None
    return arm_obj.matrix_world @ pb.head

def get_bone_posed_world_tail(arm_obj, bone_name):
    """Get the world position of bone tail in current pose."""
    pb = arm_obj.pose.bones.get(bone_name)
    if pb is None:
        return None
    return arm_obj.matrix_world @ pb.tail


def align_bone_direction_world(src_arm, tgt_arm, src_bone, tgt_bone, report):
    """
    Rotate the TARGET pose bone so its world-space direction matches SOURCE.

    KEY INSIGHT: We must work in world space and compute what local rotation
    achieves the desired world-space result.

    This function must be called HIERARCHICALLY (parents before children)
    because child bone world positions depend on parent poses.
    """
    # Get source bone's REST direction in world space
    # (Source is our reference pose, we want target to match it)
    src_dir = get_bone_rest_world_direction(src_arm, src_bone)
    if src_dir is None:
        report.append(f"  [MISS] Source bone not found: {src_bone}")
        return False

    # Get target bone's CURRENT world direction (after any parent rotations)
    tgt_dir = get_bone_posed_world_direction(tgt_arm, tgt_bone)
    if tgt_dir is None:
        report.append(f"  [MISS] Target bone not found: {tgt_bone}")
        return False

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if pb is None:
        report.append(f"  [MISS] Target pose bone not found: {tgt_bone}")
        return False

    # Calculate the rotation needed to align tgt_dir to src_dir
    # This is the world-space rotation
    rot_world = tgt_dir.rotation_difference(src_dir)

    # Convert world rotation to local rotation for the pose bone
    # The pose bone's local space is defined by its parent (or armature if root)
    if pb.parent:
        # Get parent's world rotation
        parent_world_mat = tgt_arm.matrix_world @ pb.parent.matrix
        parent_world_rot = parent_world_mat.to_quaternion()
        # Local rotation = inverse(parent_world) @ world_rotation @ parent_world
        # But we want to add this rotation to the bone, so:
        # new_world_dir = rot_world @ old_world_dir
        # The bone's matrix in armature space needs adjustment

        # Get current bone rotation in armature space
        bone_arm_rot = pb.matrix.to_quaternion()
        # Armature rotation to world
        arm_rot = tgt_arm.matrix_world.to_quaternion()

        # Current world rotation of bone
        current_world_rot = arm_rot @ bone_arm_rot
        # Desired world rotation
        desired_world_rot = rot_world @ current_world_rot

        # Convert back to armature space
        desired_arm_rot = arm_rot.inverted() @ desired_world_rot

        # The pose bone's matrix_basis is the LOCAL delta from rest pose
        # We need to find what matrix_basis achieves desired_arm_rot

        # Rest pose bone matrix in armature space
        rest_mat = pb.bone.matrix_local
        rest_rot = rest_mat.to_quaternion()

        # Parent's posed matrix in armature space
        parent_arm_mat = pb.parent.matrix
        parent_arm_rot = parent_arm_mat.to_quaternion()

        # desired_arm_rot = parent_arm_rot @ local_delta @ rest_local_rot
        # where rest_local_rot is the rest rotation relative to parent
        rest_local = pb.parent.bone.matrix_local.inverted() @ rest_mat
        rest_local_rot = rest_local.to_quaternion()

        # Solve for local_delta:
        # local_delta = parent_arm_rot.inv @ desired_arm_rot @ rest_local_rot.inv
        local_delta = parent_arm_rot.inverted() @ desired_arm_rot @ rest_local_rot.inverted()

        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = local_delta

    else:
        # Root bone - simpler case
        # desired = armature_rot @ bone_basis @ bone_rest
        arm_rot = tgt_arm.matrix_world.to_quaternion()
        rest_rot = pb.bone.matrix_local.to_quaternion()

        # Current world dir comes from: arm_rot @ (basis @ rest_rot)
        # We want: rot_world @ current = arm_rot @ (new_basis @ rest_rot)
        # So: new_basis = arm_rot.inv @ rot_world @ arm_rot @ old_basis

        old_basis = pb.rotation_quaternion if pb.rotation_mode == 'QUATERNION' else pb.matrix_basis.to_quaternion()
        current_world_rot = arm_rot @ old_basis @ rest_rot
        desired_world_rot = rot_world @ current_world_rot

        new_basis = arm_rot.inverted() @ desired_world_rot @ rest_rot.inverted()

        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = new_basis

    depsgraph_update()

    # Verify
    new_tgt_dir = get_bone_posed_world_direction(tgt_arm, tgt_bone)
    angle_error = math.degrees(src_dir.angle(new_tgt_dir)) if new_tgt_dir else 999
    report.append(f"  [OK] {src_bone} -> {tgt_bone} (error: {angle_error:.2f}°)")

    return True


def align_bone_direction_simple(src_arm, tgt_arm, src_bone, tgt_bone, report):
    """
    SIMPLER approach: Use rotation_difference directly on rest-pose directions.
    Apply the rotation as a pose delta.

    This works well when bones have similar hierarchies and rest poses aren't too different.
    """
    # Source direction (rest pose, armature local)
    sb = src_arm.data.bones.get(src_bone)
    tb = tgt_arm.data.bones.get(tgt_bone)

    if not sb or not tb:
        report.append(f"  [MISS] Bone not found: {src_bone} or {tgt_bone}")
        return False

    src_dir = (sb.tail_local - sb.head_local).normalized()
    tgt_dir = (tb.tail_local - tb.head_local).normalized()

    if src_dir.length < 1e-8 or tgt_dir.length < 1e-8:
        report.append(f"  [MISS] Zero-length bone: {src_bone} or {tgt_bone}")
        return False

    # Rotation to align target to source (in armature local space)
    # Assumes armatures are already yaw-aligned
    rot = tgt_dir.rotation_difference(src_dir)

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if not pb:
        report.append(f"  [MISS] Pose bone not found: {tgt_bone}")
        return False

    # Apply rotation in bone's local space
    # Need to convert armature-space rotation to bone-local rotation

    # Get the bone's rest matrix in armature space
    bone_rest = pb.bone.matrix_local
    bone_rest_inv = bone_rest.inverted()

    # Convert rotation to bone local space
    rot_local = (bone_rest_inv.to_quaternion() @ rot @ bone_rest.to_quaternion())

    pb.rotation_mode = 'QUATERNION'
    # Accumulate with existing rotation
    pb.rotation_quaternion = rot_local @ pb.rotation_quaternion

    depsgraph_update()
    report.append(f"  [OK] {src_bone} -> {tgt_bone}")
    return True


def align_bone_chain_iterative(src_arm, tgt_arm, bone_pairs, report, max_iterations=5):
    """
    Iteratively align a chain of bones.

    bone_pairs: list of (src_bone, tgt_bone) tuples, ordered from root to tip.

    After each bone rotation, we update and re-check because child positions
    depend on parent rotations.
    """
    report.append(f"[ChainAlign] Aligning {len(bone_pairs)} bones iteratively...")

    for iteration in range(max_iterations):
        total_error = 0.0

        for src_bone, tgt_bone in bone_pairs:
            src_dir = get_bone_rest_world_direction(src_arm, src_bone)
            tgt_dir = get_bone_posed_world_direction(tgt_arm, tgt_bone)

            if src_dir is None or tgt_dir is None:
                continue

            angle_error = src_dir.angle(tgt_dir)
            total_error += angle_error

            if angle_error > 0.001:  # ~0.06 degrees threshold
                align_bone_direction_simple(src_arm, tgt_arm, src_bone, tgt_bone, report)

        avg_error = math.degrees(total_error / len(bone_pairs))
        report.append(f"  Iteration {iteration+1}: avg error = {avg_error:.3f}°")

        if avg_error < 0.1:  # Good enough
            break

    report.append(f"[ChainAlign] Done.")


# ============================================================================
# NEW: Hierarchical alignment that respects bone parent chain
# ============================================================================

def build_bone_hierarchy_order(arm_obj, bone_names):
    """
    Given a list of bone names, return them sorted in hierarchy order
    (parents before children).
    """
    def get_depth(bone_name):
        b = arm_obj.data.bones.get(bone_name)
        depth = 0
        while b and b.parent:
            depth += 1
            b = b.parent
        return depth

    return sorted(bone_names, key=get_depth)


def align_all_mapped_bones(src_arm, tgt_arm, bone_map, report):
    """
    Align all bones in bone_map, processing in hierarchy order.

    bone_map: dict of src_bone -> tgt_bone
    """
    # Get target bone names and sort by hierarchy depth
    tgt_bones = list(bone_map.values())
    tgt_bones_sorted = build_bone_hierarchy_order(tgt_arm, tgt_bones)

    # Reverse map for lookup
    tgt_to_src = {v: k for k, v in bone_map.items()}

    ok_count = 0
    miss_count = 0

    for tgt_bone in tgt_bones_sorted:
        src_bone = tgt_to_src.get(tgt_bone)
        if not src_bone:
            continue

        ok = align_bone_direction_simple(src_arm, tgt_arm, src_bone, tgt_bone, report)
        if ok:
            ok_count += 1
        else:
            miss_count += 1

    return ok_count, miss_count


# ============================ MAIN (Step0 + Step1 only) ============================
def main():
    report = []
    report.append("UEFN -> H3D Pipeline V29 (Hierarchical Bone Alignment)")
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

    # ---- Step 1: Pose align (hierarchical) ----
    report.append("[PoseAlign] Switching Target to POSE position + clearing pose...")
    set_armature_pose_and_clear_pose(tgt_arm)

    # Full bone map - UEFN (Source) -> H3D/Mixamo (Target)
    BONE_MAP = {
        # Spine chain (process first, root to tip)
        "pelvis": "Hips",
        "spine_01": "Spine",
        "spine_02": "Spine1",
        "spine_03": "Spine2",
        "neck_01": "Neck",
        "head": "Head",

        # Left arm chain
        "clavicle_l": "LeftShoulder",
        "upperarm_l": "LeftArm",
        "lowerarm_l": "LeftForeArm",
        "hand_l": "LeftHand",

        # Right arm chain
        "clavicle_r": "RightShoulder",
        "upperarm_r": "RightArm",
        "lowerarm_r": "RightForeArm",
        "hand_r": "RightHand",

        # Left leg chain
        "thigh_l": "LeftUpLeg",
        "calf_l": "LeftLeg",
        "foot_l": "LeftFoot",
        "ball_l": "LeftToeBase",

        # Right leg chain
        "thigh_r": "RightUpLeg",
        "calf_r": "RightLeg",
        "foot_r": "RightFoot",
        "ball_r": "RightToeBase",
    }

    report.append("[PoseAlign] Aligning Target pose bones hierarchically...")

    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode='POSE')

    ok_count, miss_count = align_all_mapped_bones(src_arm, tgt_arm, BONE_MAP, report)

    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()
    report.append(f"[PoseAlign] Done. OK={ok_count} MISS={miss_count}\n")

    # ---- Final verification ----
    report.append("[Verify] Checking final bone direction errors...")
    bpy.ops.object.mode_set(mode='POSE')

    for src_bone, tgt_bone in BONE_MAP.items():
        src_dir = get_bone_rest_world_direction(src_arm, src_bone)
        tgt_dir = get_bone_posed_world_direction(tgt_arm, tgt_bone)
        if src_dir and tgt_dir:
            error_deg = math.degrees(src_dir.angle(tgt_dir))
            status = "OK" if error_deg < 5 else "WARN"
            report.append(f"  [{status}] {src_bone}->{tgt_bone}: {error_deg:.2f}°")

    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()

    # ---- Output ----
    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log saved to Text Editor: {LOG_TEXT_NAME} ---")

# Run
main()
