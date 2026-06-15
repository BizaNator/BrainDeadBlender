import bpy
from datetime import datetime

LOG_TEXT_NAME = "TargetScaleNormalize_Log.txt"

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
    """
    Robust height from evaluated mesh verts in world space.
    Optionally disables armature modifier visibility for measurement.
    """
    arm_mod = get_armature_modifier(obj)
    prev = None
    if disable_armature_mod and arm_mod:
        prev = arm_mod.show_viewport
        arm_mod.show_viewport = False
        depsgraph_update()

    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    me = obj_eval.to_mesh()
    try:
        n = len(me.vertices)
        if n == 0:
            raise RuntimeError(f"Mesh '{obj.name}' has 0 verts")
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
        z_lo = zs[lo_i]
        z_hi = zs[hi_i]
        return z_hi - z_lo, z_lo, z_hi, len(zs), step
    finally:
        obj_eval.to_mesh_clear()
        if disable_armature_mod and arm_mod and prev is not None:
            arm_mod.show_viewport = prev
            depsgraph_update()

def apply_scale_selected():
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

def set_scale_on_both(mesh_obj, arm_obj, factor):
    mesh_obj.scale *= factor
    arm_obj.scale *= factor

# ---- MAIN ----
src_col = find_collection_ci("Source")
tgt_col = find_collection_ci("Target")
if not src_col or not tgt_col:
    raise RuntimeError("Missing Source/Target collections.")

src_mesh = pick_largest_mesh(src_col)

tgt_mesh = pick_largest_mesh(tgt_col)
tgt_arm = find_single_armature(tgt_col)

# IMPORTANT: ensure the target mesh is actually skinned to this target armature
tgt_arm_mod = get_armature_modifier(tgt_mesh)
if not tgt_arm_mod or tgt_arm_mod.object != tgt_arm:
    raise RuntimeError(
        f"Target mesh '{tgt_mesh.name}' is not using Target armature '{tgt_arm.name}' in its Armature modifier."
    )

depsgraph_update()

# Measure heights in a deformation-free way (armature modifier disabled for both meshes)
src_h, _, _, _, _ = robust_z_height_world(src_mesh, disable_armature_mod=True)
tgt_h, _, _, _, _ = robust_z_height_world(tgt_mesh, disable_armature_mod=True)

scale_factor = src_h / tgt_h

# Apply factor to BOTH target mesh and target armature (keeps their spaces consistent)
set_scale_on_both(tgt_mesh, tgt_arm, scale_factor)
depsgraph_update()

# Now APPLY SCALE to both so Scale returns to (1,1,1) but world size stays
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
tgt_mesh.select_set(True)
tgt_arm.select_set(True)
bpy.context.view_layer.objects.active = tgt_mesh  # active can be either
apply_scale_selected()
depsgraph_update()

report = []
report.append("Target Normalize: scale mesh+armature together, then apply scale (keeps bone alignment consistent)")
report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append(f"Source mesh: {src_mesh.name}  robust height: {src_h:.6f}")
report.append(f"Target mesh: {tgt_mesh.name}  robust height: {tgt_h:.6f}")
report.append(f"Target armature: {tgt_arm.name}")
report.append(f"Applied scale factor (Target only): {scale_factor:.6f}\n")
report.append(f"Resulting Target mesh scale: {tuple(tgt_mesh.scale)} (should be ~1,1,1)")
report.append(f"Resulting Target armature scale: {tuple(tgt_arm.scale)} (should be ~1,1,1)")
report.append("✅ Done. Target is now clean for bone alignment + retopo.")

log_to_text("\n".join(report))
print(f"✅ Wrote log to Text Editor: {LOG_TEXT_NAME}")
