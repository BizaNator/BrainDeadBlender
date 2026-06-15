import bpy
from datetime import datetime

LOG_TEXT_NAME = "ScaleTargetTo192cm_Log.txt"

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

def mesh_objects(col):
    return [o for o in objects_in_collection(col) if o.type == "MESH"]

def pick_largest_mesh(col):
    meshes = mesh_objects(col)
    if not meshes:
        raise RuntimeError(f"No mesh objects found in collection '{col.name}'.")
    meshes.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes[0]

def depsgraph_update():
    bpy.context.view_layer.update()

def robust_z_stats_world(obj, p_low=0.01, p_high=0.99, max_samples=200000):
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    me = obj_eval.to_mesh()
    try:
        verts = me.vertices
        n = len(verts)
        if n == 0:
            raise RuntimeError(f"Mesh '{obj.name}' has 0 vertices.")
        step = 1
        if n > max_samples:
            step = max(1, n // max_samples)

        zs = []
        mw = obj_eval.matrix_world
        for i in range(0, n, step):
            zs.append((mw @ verts[i].co).z)

        zs.sort()
        lo_i = int(p_low * (len(zs) - 1))
        hi_i = int(p_high * (len(zs) - 1))
        return zs[lo_i], zs[hi_i], len(zs), step
    finally:
        obj_eval.to_mesh_clear()

def get_armature_modifier(mesh_obj):
    for m in mesh_obj.modifiers:
        if m.type == "ARMATURE" and getattr(m, "object", None):
            return m
    return None

# ---- MAIN ----
tgt_col = find_collection_ci("Target")
if not tgt_col:
    raise RuntimeError("Missing 'Target' collection.")

tgt_mesh = pick_largest_mesh(tgt_col)

# Disable armature modifier for measurement (posed toes etc.)
tgt_arm_mod = get_armature_modifier(tgt_mesh)
prev_mod = None
if tgt_arm_mod:
    prev_mod = tgt_arm_mod.show_viewport
    tgt_arm_mod.show_viewport = False

depsgraph_update()

z_lo, z_hi, samples, step = robust_z_stats_world(tgt_mesh, 0.01, 0.99)
height = z_hi - z_lo

# Restore mod
if tgt_arm_mod and prev_mod is not None:
    tgt_arm_mod.show_viewport = prev_mod
depsgraph_update()

TARGET_HEIGHT_M = 1.65  # 192 cm in meters
scale_factor = TARGET_HEIGHT_M / height

# Scale the target mesh object
tgt_mesh.scale *= scale_factor

# Also scale the deforming armature object if one exists (keeps them consistent)
tgt_arm_name = "(none)"
if tgt_arm_mod and tgt_arm_mod.object:
    tgt_arm_mod.object.scale *= scale_factor
    tgt_arm_name = tgt_arm_mod.object.name

depsgraph_update()

report = []
report.append("Scale Target To Fortnite 192cm (robust 1–99% height, modifier disabled for measurement)")
report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append(f"Target Mesh: {tgt_mesh.name} (samples={samples}, step={step})")
report.append(f"Target Armature (if any): {tgt_arm_name}\n")
report.append(f"Measured height: {height:.6f} m")
report.append(f"Desired height:  {TARGET_HEIGHT_M:.6f} m")
report.append(f"Applied scale factor: {scale_factor:.6f}\n")
report.append("✅ Done.")

log_to_text("\n".join(report))
print(f"✅ Wrote log to Text Editor: {LOG_TEXT_NAME}")
