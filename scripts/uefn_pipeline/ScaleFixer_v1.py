import bpy
from datetime import datetime

LOG_TEXT_NAME = "ScaleOnly_Log.txt"

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

def pick_mesh_with_armature_mod(col, arm_obj):
    for m in mesh_objects(col):
        for mod in m.modifiers:
            if mod.type == "ARMATURE" and mod.object == arm_obj:
                return m
    # fallback largest mesh
    meshes = mesh_objects(col)
    meshes.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes[0] if meshes else None

def bone_head_world(arm_obj, bone_name: str):
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    return arm_obj.matrix_world @ b.head_local.copy()

def depsgraph_update():
    bpy.context.view_layer.update()

src_col = find_collection_ci("Source")
tgt_col = find_collection_ci("Target")
if not src_col or not tgt_col:
    raise RuntimeError("Missing 'Source' and/or 'Target' collections.")

src_arm = find_single_armature(src_col)
tgt_arm = find_single_armature(tgt_col)

src_mesh = pick_mesh_with_armature_mod(src_col, src_arm)
tgt_mesh = pick_mesh_with_armature_mod(tgt_col, tgt_arm)

# Bone names
src_pelvis = "pelvis"
src_foot_l = "foot_l"
tgt_hips   = "Hips"
tgt_foot_l = "LeftFoot"

src_p = bone_head_world(src_arm, src_pelvis)
src_f = bone_head_world(src_arm, src_foot_l)
tgt_p = bone_head_world(tgt_arm, tgt_hips)
tgt_f = bone_head_world(tgt_arm, tgt_foot_l)

if not (src_p and src_f and tgt_p and tgt_f):
    raise RuntimeError("Missing required bones for scaling (pelvis/foot_l vs Hips/LeftFoot).")

src_len = (src_p - src_f).length
tgt_len = (tgt_p - tgt_f).length
scale_factor = src_len / tgt_len

# Apply scale uniformly to target armature + mesh
tgt_arm.scale *= scale_factor
if tgt_mesh:
    tgt_mesh.scale *= scale_factor
depsgraph_update()

# Align hips to pelvis (translation)
tgt_p2 = bone_head_world(tgt_arm, tgt_hips)
src_p2 = bone_head_world(src_arm, src_pelvis)
offset = src_p2 - tgt_p2

tgt_arm.location += offset
if tgt_mesh:
    tgt_mesh.location += offset
depsgraph_update()

report = []
report.append("Scale/Align Only")
report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append(f"Source Armature: {src_arm.name}")
report.append(f"Target Armature: {tgt_arm.name}")
report.append(f"Source Mesh: {src_mesh.name if src_mesh else '(none found)'}")
report.append(f"Target Mesh: {tgt_mesh.name if tgt_mesh else '(none found)'}\n")
report.append(f"Source pelvis→foot_l: {src_len:.6f}")
report.append(f"Target Hips→LeftFoot: {tgt_len:.6f}")
report.append(f"Applied scale factor: {scale_factor:.6f}")
report.append(f"Applied offset: ({offset.x:.6f}, {offset.y:.6f}, {offset.z:.6f})\n")
report.append("✅ Done. (No posing, no baking, no weighting.)")

log_to_text("\n".join(report))
print(f"✅ Wrote log to Text Editor: {LOG_TEXT_NAME}")
