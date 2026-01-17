import bpy
from mathutils import Vector
from datetime import datetime

LOG_TEXT_NAME = "Pipeline_Log.txt"

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

def apply_object_transforms(mesh_obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

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

# ---------------- Pose align (direction-based) ----------------
def bone_dir_arm_space(arm_obj, bone_name: str):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    v = (b.tail_local - b.head_local)
    if v.length < 1e-8:
        return None
    return v.normalized()

def rotate_pose_bone_to_match_direction(src_arm, tgt_arm, src_bone, tgt_bone, report_lines):
    src_dir = bone_dir_arm_space(src_arm, src_bone)
    tgt_dir = bone_dir_arm_space(tgt_arm, tgt_bone)
    if src_dir is None:
        report_lines.append(f"  [MISS] Source dir missing: {src_bone}")
        return False
    if tgt_dir is None:
        report_lines.append(f"  [MISS] Target dir missing: {tgt_bone}")
        return False

    q = tgt_dir.rotation_difference(src_dir)

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if pb is None:
        report_lines.append(f"  [MISS] Target pose bone missing: {tgt_bone}")
        return False

    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = q @ pb.rotation_quaternion
    report_lines.append(f"  [OK] DIR {src_bone} -> {tgt_bone}")
    return True

# ---------------- Bake / clean ----------------
def apply_armature_modifier(mesh_obj, arm_obj):
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
        raise RuntimeError(f"No Armature modifier on '{mesh_obj.name}' referencing '{arm_obj.name}'.")

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

# ---------------- Weight transfer (FIXED for your Blender 5.0) ----------------
def _set_enum_if_possible(obj, prop_name, preferred_values, report_lines, label):
    """
    Set obj.prop_name to the first value in preferred_values that exists in the enum.
    Logs what it chose.
    """
    if not hasattr(obj, prop_name):
        report_lines.append(f"[Weights] {label}: property '{prop_name}' not found (skipped).")
        return None

    prop = obj.bl_rna.properties.get(prop_name)
    if not prop or not hasattr(prop, "enum_items"):
        report_lines.append(f"[Weights] {label}: '{prop_name}' has no enum_items (skipped).")
        return None

    available = [e.identifier for e in prop.enum_items]
    chosen = next((v for v in preferred_values if v in available), None)
    if chosen is None:
        report_lines.append(f"[Weights] {label}: no preferred value found for '{prop_name}'. Available={available}")
        return None

    setattr(obj, prop_name, chosen)
    report_lines.append(f"[Weights] {label}: set {prop_name}={chosen}")
    return chosen

def transfer_weights_via_modifier(source_mesh, target_mesh, report_lines):
    """
    Blender-5.x safe weight transfer using Data Transfer modifier.
    Auto-detects the enum style for vgroup selection/matching.
    """
    ensure_object_mode()
    depsgraph_update()

    bpy.ops.object.select_all(action='DESELECT')
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh

    mod = target_mesh.modifiers.new(name="DT_Weights", type='DATA_TRANSFER')
    mod.object = source_mesh

    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}

    # --- Handle BOTH possible enum styles across Blender 5.x builds ---
    # Some builds want NAME/INDEX (match mode). Others want ALL / ACTIVE / single group.
    _set_enum_if_possible(mod, "layers_vgroup_select_src", ["NAME", "ALL", "ACTIVE", "INDEX"], report_lines, "VGroup SRC select")
    _set_enum_if_possible(mod, "layers_vgroup_select_dst", ["NAME", "ALL", "ACTIVE", "INDEX"], report_lines, "VGroup DST select")

    # Vertex mapping selection (this one is fairly stable)
    preferred_map = ["POLYINTERP_NEAREST", "POLYINTERP_VNORPROJ", "POLY_NEAREST", "NEAREST"]
    available_map = [e.identifier for e in mod.bl_rna.properties['vert_mapping'].enum_items]
    mod.vert_mapping = next((v for v in preferred_map if v in available_map), available_map[0])

    mod.mix_mode = 'REPLACE'
    mod.mix_factor = 1.0

    report_lines.append(f"[Weights] DataTransfer vert_mapping={mod.vert_mapping}")
    report_lines.append(f"[Weights] Source={source_mesh.name} -> Target={target_mesh.name}")

    bpy.ops.object.modifier_apply(modifier=mod.name)
    report_lines.append("[OK] Weight transfer applied (Data Transfer modifier).")
    
# ---------------- Remap Vertex Groups to match UEFN --------
def remap_vertex_groups_by_bone_map(target_mesh, bone_map, report_lines):
    """
    Make Target mesh's vertex groups compatible with Source armature by:
    - copying weights from existing groups (H3D names) into new groups (UEFN names)
    - leaving original groups in place (optional cleanup later)
    """
    # Build existing group name set
    vg = target_mesh.vertex_groups
    existing = {g.name for g in vg}

    # Ensure destination groups exist
    for src_bone, tgt_bone in bone_map.items():
        # src_bone = UEFN name, tgt_bone = H3D name (your current map is UEFN->H3D)
        uefn = src_bone
        h3d  = tgt_bone
        if h3d not in existing:
            report_lines.append(f"[VGroups] MISS source group on Target: {h3d}")
            continue
        if uefn not in existing:
            vg.new(name=uefn)
            existing.add(uefn)
            report_lines.append(f"[VGroups] Created group: {uefn}")

        # Copy weights vertex-by-vertex (slow but correct; OK for one mesh at a time)
        src_g = vg[h3d]
        dst_g = vg[uefn]

        # iterate vertices; copy if weight exists
        for v in target_mesh.data.vertices:
            w = 0.0
            for e in v.groups:
                if vg[e.group].name == h3d:
                    w = e.weight
                    break
            if w > 0.0:
                dst_g.add([v.index], w, 'REPLACE')

    report_lines.append("[VGroups] Remap complete (H3D->UEFN groups added).")


# ---------------- Bind ----------------
def bind_mesh_to_armature(mesh_obj, arm_obj, report_lines):
    """
    Bind mesh to armature WITHOUT bpy.ops parent_set (no surprises),
    but correctly keeps the mesh's world transform by setting parent inverse properly.
    """
    ensure_object_mode()

    # Ensure Source armature is in REST (critical for predictable deformation)
    arm_obj.data.pose_position = 'REST'
    depsgraph_update()

    # Remove any existing Armature modifiers (deterministic)
    for m in list(mesh_obj.modifiers):
        if m.type == "ARMATURE":
            mesh_obj.modifiers.remove(m)

    # Add Armature modifier
    am = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    am.object = arm_obj

    # Parent KEEP WORLD TRANSFORM (the important part)
    child_world = mesh_obj.matrix_world.copy()
    mesh_obj.parent = arm_obj
    mesh_obj.matrix_parent_inverse = arm_obj.matrix_world.inverted() @ child_world

    depsgraph_update()
    report_lines.append("[Bind] Added Armature modifier + parented to Source armature (keep world transform).")

def bind_mesh_to_armature_modifier_only(mesh_obj, arm_obj, report_lines):
    """
    Bind via Armature modifier only. No parenting at all.
    This eliminates parent inverse / double-transform issues completely.
    """
    ensure_object_mode()

    arm_obj.data.pose_position = 'REST'
    depsgraph_update()

    # Remove existing armature mods
    for m in list(mesh_obj.modifiers):
        if m.type == "ARMATURE":
            mesh_obj.modifiers.remove(m)

    am = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    am.object = arm_obj
    am.use_vertex_groups = True
    am.use_bone_envelopes = False

    # IMPORTANT: don't parent
    mesh_obj.parent = None

    depsgraph_update()
    report_lines.append("[Bind] Added Armature modifier ONLY (no parenting). Envelopes OFF, VGroups ON.")


# Debug Function 
def report_weight_stats(mesh_obj, groups_to_check, report_lines, sample_limit=50):
    vg = mesh_obj.vertex_groups
    existing = {g.name for g in vg}
    report_lines.append("[WeightsDebug] Vertex group influence counts (approx):")

    # quick per-group count: scan vertices and count if weight>0
    for name in groups_to_check:
        if name not in existing:
            report_lines.append(f"  - {name}: MISSING")
            continue

        count = 0
        total_w = 0.0
        for v in mesh_obj.data.vertices:
            w = 0.0
            for e in v.groups:
                if vg[e.group].name == name:
                    w = e.weight
                    break
            if w > 0.0001:
                count += 1
                total_w += w

        report_lines.append(f"  - {name}: verts={count} sum_w={total_w:.2f}")

    # also detect if almost everything is weighted to one group
    # check the biggest group among those
    best = None
    best_count = -1
    for name in groups_to_check:
        if name not in existing:
            continue
        count = 0
        for v in mesh_obj.data.vertices:
            for e in v.groups:
                if vg[e.group].name == name and e.weight > 0.0001:
                    count += 1
                    break
        if count > best_count:
            best_count = count
            best = name
    if best:
        report_lines.append(f"[WeightsDebug] Largest checked group: {best} (verts={best_count})")


# ============================ MAIN ============================
def main():
    report = []
    report.append("UEFN → H3D Ingest Pipeline (V2.1)")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    src_col = find_collection_ci("Source")
    tgt_col = find_collection_ci("Target")
    if not src_col or not tgt_col:
        raise RuntimeError("Missing 'Source' and/or 'Target' collections (case-insensitive).")

    src_arm = find_single_armature(src_col)
    tgt_arm = find_single_armature(tgt_col)

    src_mesh, src_pick = pick_mesh_driven_by_armature(src_col, src_arm)
    tgt_mesh, tgt_pick = pick_mesh_driven_by_armature(tgt_col, tgt_arm)

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

    # Robust scale match (pose-free)
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

    # ---- Step 1: Pose align Target toward Source ----
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

    report.append("[PoseAlign] Switching Target to POSE position + clearing pose...")
    set_armature_pose_and_clear_pose(tgt_arm)

    report.append("[PoseAlign] Aligning Target bones by direction...")
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    tgt_arm.select_set(True)
    bpy.context.view_layer.objects.active = tgt_arm
    bpy.ops.object.mode_set(mode='POSE')

    ok_count = 0
    miss_count = 0
    for s_bone, t_bone in BONE_MAP.items():
        ok = rotate_pose_bone_to_match_direction(src_arm, tgt_arm, s_bone, t_bone, report)
        ok_count += 1 if ok else 0
        miss_count += 0 if ok else 1

    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()
    report.append(f"[PoseAlign] Done. OK={ok_count} MISS={miss_count}\n")

    # ---- Step 2: Bake posed geometry ----
    report.append("[Bake] Applying Target Armature modifier to bake posed geometry...")
    apply_armature_modifier(tgt_mesh, tgt_arm)
    depsgraph_update()
    report.append("[Bake] Baked.\n")

    # IMPORTANT: after bake, restore a clean REST state (so you don't see scrambled pose)
    report.append("[PostBake] Setting Target armature to REST + clearing pose (visual sanity)...")
    set_armature_rest_and_clear_pose(tgt_arm)

    # ---- Step 3: Clean baked mesh ----
    report.append("[Clean] Removing armature mods + vertex groups, applying transforms...")
    remove_armature_modifiers(tgt_mesh)
    # clear_vertex_groups(tgt_mesh)
    apply_object_transforms(tgt_mesh)
    depsgraph_update()
    report.append("[Clean] Done.\n")

    # ---- Step 4: Transfer weights while unskinned ----
    # report.append("[Weights] Transferring weights from Source mannequin mesh to baked mesh...")
    # transfer_weights_via_modifier(src_mesh, tgt_mesh, report)
    # depsgraph_update()
    # report.append("[Weights] Done.\n")
    

    # ---- Step 4: Remap Vertex Groups to Match UEFN (instead of projecting mannequin weights) ----
    report.append("[VGroups] Remapping Target vertex groups (H3D -> UEFN)...")
    remap_vertex_groups_by_bone_map(tgt_mesh, BONE_MAP, report)
    depsgraph_update()
    report.append("[VGroups] Remap done.\n")

    # ---- DEBUG - Weights
    report_weight_stats(
    tgt_mesh,
    ["pelvis","spine_01","clavicle_l","upperarm_l","thigh_l","foot_l","head"],
    report
    )



    # ---- Step 5: Bind to Source armature ----
    report.append("[Bind] Binding baked mesh to Source armature...")
    bind_mesh_to_armature_modifier_only(tgt_mesh, src_arm, report)
    depsgraph_update()
    report.append("[Bind] Done.\n")

    # ---- Step 6: Cleanup target armature (optional) ----
    DO_DELETE_TARGET_ARMATURE = False # Set to TRUE in Production
    if DO_DELETE_TARGET_ARMATURE:
        report.append("[Cleanup] Deleting Target armature (scaffolding)...")
        ensure_object_mode()
        bpy.ops.object.select_all(action='DESELECT')
        tgt_arm.select_set(True)
        bpy.context.view_layer.objects.active = tgt_arm
        bpy.ops.object.delete()
        depsgraph_update()
        report.append("[Cleanup] Target armature deleted.\n")

    report.append("✅ Pipeline complete.")
    report.append("Target mesh is scaled, baked to A-pose, weighted, and bound to Source armature.")

    log_to_text("\n".join(report))
    print(f"✅ Pipeline finished. See Text Editor: {LOG_TEXT_NAME}")

# Run
main()
