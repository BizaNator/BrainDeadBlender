import bpy
from datetime import datetime

LOG_TEXT_NAME = "Step0_TargetPrep_Log.txt"

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

def pick_largest_mesh(col):
    meshes = [o for o in objects_in_collection(col) if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects in collection '{col.name}'.")
    meshes.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes[0]

def pick_mesh_driven_by_armature(col, arm_obj):
    """
    Prefer meshes in `col` that have an Armature modifier pointing at `arm_obj`.
    If multiple, pick the one with the most verts among those.
    If none, fall back to largest mesh in the collection (and you should log a warning).
    """
    meshes = [o for o in col.all_objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects found in collection '{col.name}'.")

    driven = []
    for m in meshes:
        for mod in m.modifiers:
            if mod.type == "ARMATURE" and getattr(mod, "object", None) == arm_obj:
                driven.append(m)
                break

    if driven:
        driven.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
        return driven[0], "armature_modifier_match"

    # fallback
    meshes.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes[0], "fallback_largest_mesh"


def find_single_armature(col):
    arms = [o for o in objects_in_collection(col) if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"Collection '{col.name}' must contain exactly 1 armature; found {len(arms)}.")
    return arms[0]

def get_armature_modifier(mesh_obj):
    for m in mesh_obj.modifiers:
        if m.type == "ARMATURE" and getattr(m, "object", None):
            return m
    return None

def depsgraph_update():
    bpy.context.view_layer.update()

def robust_z_height_world(obj, disable_armature_mod=True, max_samples=200000, p_low=0.01, p_high=0.99):
    arm_mod = get_armature_modifier(obj)
    prev_vis = None
    if disable_armature_mod and arm_mod:
        prev_vis = arm_mod.show_viewport
        arm_mod.show_viewport = False
        depsgraph_update()

    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    me = obj_eval.to_mesh()
    try:
        n = len(me.vertices)
        if n == 0:
            raise RuntimeError(f"Mesh '{obj.name}' has 0 verts.")
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
        return (zs[hi_i] - zs[lo_i]), len(zs), step
    finally:
        obj_eval.to_mesh_clear()
        if disable_armature_mod and arm_mod and prev_vis is not None:
            arm_mod.show_viewport = prev_vis
            depsgraph_update()

def clear_parent_keep_transform(obj):
    if obj.parent is None:
        return False
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    return True

def apply_scale_to(obj_list):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for o in obj_list:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obj_list[0]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# ---- MAIN ----

report = []

src_col = find_collection_ci("Source")
tgt_col = find_collection_ci("Target")
if not src_col or not tgt_col:
    raise RuntimeError("Missing Source/Target collections.")

src_arm = find_single_armature(src_col)
tgt_arm = find_single_armature(tgt_col)

src_mesh, src_pick_mode = pick_mesh_driven_by_armature(src_col, src_arm)
tgt_mesh, tgt_pick_mode = pick_mesh_driven_by_armature(tgt_col, tgt_arm)

# Log these so you know when it fell back
report.append(f"Source mesh pick: {src_mesh.name} ({src_pick_mode})")
report.append(f"Target mesh pick: {tgt_mesh.name} ({tgt_pick_mode})")
if "fallback" in src_pick_mode or "fallback" in tgt_pick_mode:
    report.append("⚠️ WARNING: Fell back to largest mesh because no armature-modifier match was found.")

tgt_arm_mod = get_armature_modifier(tgt_mesh)
if not tgt_arm_mod or tgt_arm_mod.object != tgt_arm:
    raise RuntimeError(f"Target mesh '{tgt_mesh.name}' must have an Armature modifier pointing to '{tgt_arm.name}'.")

# 1) Force Target armature to REST for consistent measurement
prev_pose_pos = tgt_arm.data.pose_position
tgt_arm.data.pose_position = 'REST'
depsgraph_update()

# Optional: clear pose transforms so pose is clean (safe)
# --- Ensure armature is active, then clear pose transforms ---
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
tgt_arm.select_set(True)
bpy.context.view_layer.objects.active = tgt_arm

bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.select_all(action='SELECT')
bpy.ops.pose.transforms_clear()
bpy.ops.object.mode_set(mode='OBJECT')
depsgraph_update()


# 2) Unparent mesh from armature to avoid double scale (keep world transform)
was_parented = (tgt_mesh.parent == tgt_arm)
did_unparent = False
if was_parented:
    did_unparent = clear_parent_keep_transform(tgt_mesh)
depsgraph_update()

# 3) Measure robust heights with armature mods disabled (pose-free)
src_h, _, _ = robust_z_height_world(src_mesh, disable_armature_mod=True)
tgt_h, _, _ = robust_z_height_world(tgt_mesh, disable_armature_mod=True)
scale_factor = src_h / tgt_h

# 4) Scale BOTH target mesh and target armature together
tgt_mesh.scale *= scale_factor
tgt_arm.scale  *= scale_factor
depsgraph_update()

# 5) Apply scale to both so they return to (1,1,1)
apply_scale_to([tgt_mesh, tgt_arm])
depsgraph_update()

# Restore pose position (optional; keep REST if you prefer)
# tgt_arm.data.pose_position = prev_pose_pos
tgt_arm.data.pose_position = 'REST'
depsgraph_update()


report.append("Step 0 — Target Prep (REST → Unparent → Scale Match → Apply Scale)")
report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append(f"Source Mesh: {src_mesh.name}  height≈{src_h:.6f}")
report.append(f"Target Mesh: {tgt_mesh.name}  height≈{tgt_h:.6f}")
report.append(f"Target Armature: {tgt_arm.name}")
report.append(f"Target Armature modifier OK: {tgt_arm_mod.name}\n")
report.append(f"Unparented from armature: {did_unparent}")
report.append(f"Applied scale factor: {scale_factor:.6f}\n")
report.append(f"Target Mesh scale now: {tuple(tgt_mesh.scale)}")
report.append(f"Target Armature scale now: {tuple(tgt_arm.scale)}\n")
report.append("✅ Done. Target is clean for pose alignment / bake / retopo.")

log_to_text("\n".join(report))
print(f"✅ Wrote log to Text Editor: {LOG_TEXT_NAME}")
