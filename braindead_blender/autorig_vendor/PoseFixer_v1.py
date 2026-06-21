import bpy
import mathutils
from datetime import datetime


LOG_TEXT_NAME = "Pipeline_Log.txt"

def log_to_text(s: str):
    txt = bpy.data.texts.get(LOG_TEXT_NAME)
    if not txt:
        txt = bpy.data.texts.new(LOG_TEXT_NAME)
    txt.clear()
    txt.write(s)

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

def has_armature_modifier(obj):
    return any(m.type == "ARMATURE" for m in obj.modifiers)

def armature_modifier_targets(obj):
    out = []
    for m in obj.modifiers:
        if m.type == "ARMATURE" and getattr(m, "object", None):
            out.append(m.object)
    return out

def pick_source_mesh(col, src_arm):
    meshes = mesh_objects(col)
    skinned = [m for m in meshes if has_armature_modifier(m)]
    direct = [m for m in skinned if src_arm in armature_modifier_targets(m)]
    if direct:
        return direct[0], "armature_modifier_matches_source_armature"
    if skinned:
        return skinned[0], "fallback_any_skinned_mesh"
    raise RuntimeError(f"No skinned mesh found in Source collection '{col.name}'.")

def pick_target_mesh(col, tgt_arm):
    meshes = mesh_objects(col)
    if not meshes:
        raise RuntimeError(f"No mesh objects found in Target collection '{col.name}'.")
    skinned = [m for m in meshes if has_armature_modifier(m)]
    direct = [m for m in skinned if tgt_arm in armature_modifier_targets(m)]
    if direct:
        return direct[0], "armature_modifier_matches_target_armature"
    meshes_sorted = sorted(meshes, key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes_sorted[0], "fallback_largest_mesh"

def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def depsgraph_update():
    bpy.context.view_layer.update()

def bone_local_rest_quat(arm_obj, bone_name: str):
    """
    Return bone local-space rest rotation quaternion relative to parent (from armature data).
    """
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    if b.parent:
        parent = b.parent
        m_local = parent.matrix_local.inverted() @ b.matrix_local
    else:
        m_local = b.matrix_local
    return m_local.to_quaternion()

def bone_head_world(arm_obj, bone_name: str):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    # bone.head_local is in armature local space
    head_local = b.head_local.copy()
    return arm_obj.matrix_world @ head_local

def set_pose_delta_from_source_rest(src_arm, tgt_arm, src_bone, tgt_bone, report_lines):
    """
    Compute delta quaternion between source local rest and target local rest,
    and apply that delta as target pose rotation (QUAT).
    """
    q_src = bone_local_rest_quat(src_arm, src_bone)
    q_tgt = bone_local_rest_quat(tgt_arm, tgt_bone)
    if q_src is None:
        report_lines.append(f"  [MISS] Source bone not found: {src_bone}")
        return False
    if q_tgt is None:
        report_lines.append(f"  [MISS] Target bone not found: {tgt_bone}")
        return False

    delta = q_src @ q_tgt.inverted()

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if pb is None:
        report_lines.append(f"  [MISS] Target pose bone missing: {tgt_bone}")
        return False

    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = delta
    report_lines.append(f"  [OK] {src_bone} -> {tgt_bone}")
    return True

def apply_armature_modifier(mesh_obj, arm_obj):
    """
    Apply the armature modifier that references arm_obj (if present).
    """
    ensure_object_mode()
    depsgraph_update()

    # Make mesh active
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    # Find armature modifier targeting arm_obj
    mod = None
    for m in mesh_obj.modifiers:
        if m.type == 'ARMATURE' and m.object == arm_obj:
            mod = m
            break
    if not mod:
        raise RuntimeError(f"No Armature modifier on '{mesh_obj.name}' referencing '{arm_obj.name}'.")

    bpy.ops.object.modifier_apply(modifier=mod.name)

def parent_to_armature_empty_groups(mesh_obj, arm_obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE_NAME', keep_transform=True)

    # Now ensure empty groups exist (Blender may create them; we enforce via operator)
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)  # harmless; ensures armature relation
    

def transfer_weights_via_modifier(source_mesh, target_mesh, report_lines):
    """
    Blender 5.x-safe weight transfer using a Data Transfer modifier.
    Transfers ALL vertex group weights from source_mesh -> target_mesh.
    """
    ensure_object_mode()
    depsgraph_update()

    # Make target active
    bpy.ops.object.select_all(action='DESELECT')
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh

    # Add Data Transfer modifier
    mod = target_mesh.modifiers.new(name="DT_Weights", type='DATA_TRANSFER')
    mod.object = source_mesh

    # IMPORTANT: transfer vertex group weights (all groups)
    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}

    # This is the Blender 5.x enum you hit earlier (ACTIVE/ALL/<groupnames>)
    # Set BOTH to ALL so it transfers every group, not just active.
    if hasattr(mod, "layers_vgroup_select_src"):
        mod.layers_vgroup_select_src = 'ALL'
    if hasattr(mod, "layers_vgroup_select_dst"):
        mod.layers_vgroup_select_dst = 'ALL'

    # Mapping: pick best available
    preferred = ["POLYINTERP_NEAREST", "POLYINTERP_VNORPROJ", "POLY_NEAREST", "NEAREST"]
    available = [e.identifier for e in mod.bl_rna.properties['vert_mapping'].enum_items]
    mod.vert_mapping = next((v for v in preferred if v in available), available[0])

    mod.mix_mode = 'REPLACE'
    mod.mix_factor = 1.0

    report_lines.append(f"[Weights] DataTransfer vert_mapping={mod.vert_mapping}")
    report_lines.append("[Weights] VGroup select: SRC=ALL DST=ALL")

    bpy.ops.object.modifier_apply(modifier=mod.name)
    report_lines.append("[OK] Weight transfer applied (ALL vertex groups).")


    
def clear_vertex_groups(mesh_obj):
    if mesh_obj.type != "MESH":
        return
    mesh_obj.vertex_groups.clear()

def remove_armature_modifiers(mesh_obj):
    if mesh_obj.type != "MESH":
        return
    for m in list(mesh_obj.modifiers):
        if m.type == "ARMATURE":
            mesh_obj.modifiers.remove(m)

def apply_object_transforms(mesh_obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)



def bone_dir_arm_space(arm_obj, bone_name: str):
    """Return bone direction (tail-head) in armature local space from rest pose."""
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    v = (b.tail_local - b.head_local)
    if v.length < 1e-8:
        return None
    return v.normalized()

def set_pose_to_match_bone_direction(src_arm, tgt_arm, src_bone, tgt_bone, report_lines):
    """
    Rotate tgt_bone in pose mode so its rest-direction aligns to src_bone rest-direction.
    This ignores roll differences and is much more stable for Mixamo/H3D rigs.
    """
    src_dir = bone_dir_arm_space(src_arm, src_bone)
    tgt_dir = bone_dir_arm_space(tgt_arm, tgt_bone)
    if src_dir is None:
        report_lines.append(f"  [MISS] Source dir missing: {src_bone}")
        return False
    if tgt_dir is None:
        report_lines.append(f"  [MISS] Target dir missing: {tgt_bone}")
        return False

    # Rotation that takes target direction to source direction
    q = tgt_dir.rotation_difference(src_dir)

    pb = tgt_arm.pose.bones.get(tgt_bone)
    if pb is None:
        report_lines.append(f"  [MISS] Target pose bone missing: {tgt_bone}")
        return False

    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = q
    report_lines.append(f"  [OK] DIR {src_bone} -> {tgt_bone}")
    return True




# ---------------- MAIN ----------------
src_col = find_collection_ci("Source")
tgt_col = find_collection_ci("Target")
if not src_col or not tgt_col:
    raise RuntimeError("Missing 'Source' and/or 'Target' collections (case-insensitive).")

src_arm = find_single_armature(src_col)
tgt_arm = find_single_armature(tgt_col)

src_mesh, src_mesh_reason = pick_source_mesh(src_col, src_arm)
tgt_mesh, tgt_mesh_reason = pick_target_mesh(tgt_col, tgt_arm)

report = []
report.append("UEFN → H3D Ingest Pipeline")
report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append(f"Source Armature: {src_arm.name}")
report.append(f"Source Mesh:     {src_mesh.name} [{src_mesh_reason}]")
report.append(f"Target Armature: {tgt_arm.name}")
report.append(f"Target Mesh:     {tgt_mesh.name} [{tgt_mesh_reason}]\n")

# ---- Step 1: Scale target using pelvis->foot distance ----
src_pelvis = "pelvis"
src_foot_l = "foot_l"
tgt_hips   = "Hips"
tgt_foot_l = "LeftFoot"

src_p = bone_head_world(src_arm, src_pelvis)
src_f = bone_head_world(src_arm, src_foot_l)
tgt_p = bone_head_world(tgt_arm, tgt_hips)
tgt_f = bone_head_world(tgt_arm, tgt_foot_l)

if not (src_p and src_f and tgt_p and tgt_f):
    raise RuntimeError("Missing one of the required bones for scaling (pelvis/foot_l vs Hips/LeftFoot).")

src_len = (src_p - src_f).length
tgt_len = (tgt_p - tgt_f).length
if tgt_len < 1e-6:
    raise RuntimeError("Target pelvis->foot length is near zero; cannot scale.")

scale_factor = src_len / tgt_len

report.append(f"[Scale] Source pelvis→foot_l: {src_len:.4f}")
report.append(f"[Scale] Target Hips→LeftFoot: {tgt_len:.4f}")
report.append(f"[Scale] Applying uniform scale factor to Target: {scale_factor:.6f}\n")

# Apply scale to target armature object and target mesh object (keeps skin relationship)
tgt_arm.scale *= scale_factor
tgt_mesh.scale *= scale_factor
depsgraph_update()

# ---- Step 2: Translate target so hips aligns to pelvis ----
src_p2 = bone_head_world(src_arm, src_pelvis)
tgt_p2 = bone_head_world(tgt_arm, tgt_hips)
offset = src_p2 - tgt_p2
tgt_arm.location += offset
tgt_mesh.location += offset
depsgraph_update()

report.append(f"[Align] Translated Target by offset: ({offset.x:.4f}, {offset.y:.4f}, {offset.z:.4f})\n")

# ---- Step 3: Pose-align (delta from source rest to target rest) ----
# Deterministic mapping for your current bone lists (Fortnite → Mixamo/H3D)
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

report.append("[PoseAlign] Applying pose deltas for mapped bones:")
ensure_object_mode()
bpy.ops.object.select_all(action='DESELECT')
tgt_arm.select_set(True)
bpy.context.view_layer.objects.active = tgt_arm

# Put target armature in Pose Mode to set pose rotations
bpy.ops.object.mode_set(mode='POSE')

ok_count = 0
miss_count = 0
for s_bone, t_bone in BONE_MAP.items():
    lines_before = len(report)
    ok = set_pose_to_match_bone_direction(src_arm, tgt_arm, s_bone, t_bone, report)
    if ok:
        ok_count += 1
    else:
        miss_count += 1

bpy.ops.object.mode_set(mode='OBJECT')
depsgraph_update()
report.append(f"\n[PoseAlign] Done. OK={ok_count} MISS={miss_count}\n")

# ---- Step 4: Bake target mesh (apply armature mod) ----
report.append("[Bake] Applying Target Armature modifier to bake posed geometry...")
apply_armature_modifier(tgt_mesh, tgt_arm)
depsgraph_update()
report.append("[Bake] Baked.\n")

# ---- Step 4.5: Clean baked mesh (CRITICAL) ----
report.append("[Clean] Removing old armature mods + vertex groups, applying transforms...")
remove_armature_modifiers(tgt_mesh)
clear_vertex_groups(tgt_mesh)
apply_object_transforms(tgt_mesh)
depsgraph_update()
report.append("[Clean] Done.\n")

# ---- Step 5: Transfer weights while UN-SKINNED ----
report.append("[Weights] Transferring weights from Source mannequin mesh to baked mesh (unskinned)...")
transfer_weights_via_modifier(src_mesh, tgt_mesh, report)
depsgraph_update()
report.append("[Weights] Done.\n")

remove_armature_modifiers(tgt_mesh)  # nukes any leftover Armature mods (incl. H3D)
depsgraph_update()


# ---- Step 6: Now bind to Source armature ----
report.append("[Bind] Parenting baked mesh to Source armature (ARMATURE, keep transform)...")
ensure_object_mode()
bpy.ops.object.select_all(action='DESELECT')
tgt_mesh.select_set(True)
src_arm.select_set(True)
bpy.context.view_layer.objects.active = src_arm
bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)
depsgraph_update()
report.append("[Bind] Done.\n")


# ---- Step 7: NOW delete target armature (scaffolding) ----
report.append("[Cleanup] Deleting Target armature (scaffolding)...")
bpy.ops.object.select_all(action='DESELECT')
tgt_arm.select_set(True)
bpy.context.view_layer.objects.active = tgt_arm
bpy.ops.object.delete()
depsgraph_update()
report.append("[Cleanup] Target armature deleted.\n")



report.append("✅ Pipeline complete. Your Target mesh is now:")
report.append("- A-pose baked geometry")
report.append("- Bound to UEFN Source armature")
report.append("- Weighted from mannequin")

log_to_text("\n".join(report))
print(f"✅ Pipeline finished. See Text Editor: {LOG_TEXT_NAME}")
 