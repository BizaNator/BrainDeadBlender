"""One-shot: rebuild the torso with the construction-guide-driven values.

Paste into Blender Python console:

    exec(open(r"B:\Brains\Tools\BrainDeadBlender\scripts\face_base_pipeline\run_iter25e_torso.py").read())
"""

import sys
import bpy

SCRIPT_DIR = r"B:\Brains\Tools\BrainDeadBlender\scripts\face_base_pipeline"
ITER_25D = r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\VABase\Female\iter\iter_25d_continuous_torso.blend"
ITER_25E = r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\VABase\Female\iter\iter_25e_torso_docvalues.blend"

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

for m in ("limb_anatomy_geonode",):
    if m in sys.modules:
        del sys.modules[m]

# Load iter_25d if not already loaded
if bpy.data.filepath != ITER_25D:
    print(f"Opening {ITER_25D}")
    bpy.ops.wm.open_mainfile(filepath=ITER_25D)

import limb_anatomy_geonode as lag
import importlib
importlib.reload(lag)

# Force REBUILD the torso group with new socket list + clear existing torso obj
if "Torso" in bpy.data.objects:
    obj = bpy.data.objects["Torso"]
    # remove obj + its empties
    for e in [bpy.data.objects.get("Torso_P0"), bpy.data.objects.get("Torso_P1")]:
        if e: bpy.data.objects.remove(e, do_unlink=True)
    bpy.data.objects.remove(obj, do_unlink=True)

# Force-rebuild the GN group to pick up new TORSO_MUSCLES sockets
lag.ensure_torso_anatomy_group(force_rebuild=True)

# Create new torso at canonical position (waist-to-clavicle)
obj, p0, p1 = lag.create_anatomy_object(
    "torso_f", "Torso",
    p0=(0.0, 0.020, 0.95),
    p1=(0.0, 0.020, 1.30),
    start_radius=0.073,
    end_radius=0.114,
    limb_side=1.0,
    archetype="average",   # F average
)
bpy.context.view_layer.update()

# Verify
depsgraph = bpy.context.evaluated_depsgraph_get()
eval_obj = obj.evaluated_get(depsgraph)
me = eval_obj.to_mesh()
ys = [v.co.y for v in me.vertices]
print(f"\nNew torso:")
print(f"  Verts: {len(me.vertices)}")
print(f"  Y span: {min(ys)*1000:.0f}mm to {max(ys)*1000:.0f}mm (front-back, +Y=forward)")
print(f"  Y range: {(max(ys)-min(ys))*1000:.0f}mm")
eval_obj.to_mesh_clear()

# Show active muscles
ng = bpy.data.node_groups['BD_LimbAnatomy_Torso']
sid = {s.name: s.identifier for s in ng.interface.items_tree
       if s.item_type=='SOCKET' and s.in_out=='INPUT'}
mod = obj.modifiers[0]
print(f"\nActive muscles (Amount × Archetype > 0):")
for muscle in ('Sternum','Trapezius','Pec','Breast','Lat','Clavicle','Serratus'):
    # Check L/R or single
    if muscle in ('Sternum','Trapezius'):
        amt = mod[sid[f"{muscle} Amount"]]
        arch = mod[sid[f"{muscle} Archetype"]]
        print(f"  {muscle:12s}: amt={amt*1000:5.1f}mm × arch={arch:.2f} = {amt*arch*1000:5.1f}mm")
    else:
        amt = mod[sid[f"{muscle}_L Amount"]]
        arch = mod[sid[f"{muscle}_L Archetype"]]
        print(f"  {muscle:12s}: amt={amt*1000:5.1f}mm × arch={arch:.2f} = {amt*arch*1000:5.1f}mm")

# Frame the torso
for o in bpy.data.objects: o.select_set(False)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for r in area.regions:
            if r.type == 'WINDOW':
                with bpy.context.temp_override(area=area, region=r):
                    bpy.ops.view3d.view_axis(type='FRONT')
                    bpy.ops.view3d.view_selected()
                    bpy.ops.view3d.view_orbit(angle=0.4, type='ORBITRIGHT')
                    bpy.ops.view3d.view_orbit(angle=0.15, type='ORBITUP')
        break

bpy.ops.wm.save_as_mainfile(filepath=ITER_25E)
print(f"\nSaved: {ITER_25E}")
