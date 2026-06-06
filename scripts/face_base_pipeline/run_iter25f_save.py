"""Save iter_25f (elliptical torso + grooves). Run in Blender Text Editor."""
import bpy
out = r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\VABase\Female\iter\iter_25f_torso_elliptical.blend"
bpy.ops.wm.save_as_mainfile(filepath=out)
print(f"Saved: {out}")

# Frame 3/4 view of torso for screenshot
torso = bpy.data.objects['Torso']
for o in bpy.data.objects: o.select_set(False)
torso.select_set(True)
bpy.context.view_layer.objects.active = torso

# Hide ref + show
ref = bpy.data.objects.get('REF_Chest')
if ref: ref.hide_viewport = False  # keep visible for comparison

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for r in area.regions:
            if r.type == 'WINDOW':
                with bpy.context.temp_override(area=area, region=r):
                    bpy.ops.view3d.view_axis(type='FRONT')
                    bpy.ops.view3d.view_selected()
                    bpy.ops.view3d.view_orbit(angle=0.3, type='ORBITRIGHT')
                    bpy.ops.view3d.view_orbit(angle=0.1, type='ORBITUP')
        break
bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
print("Framed 3/4 view — please take a screenshot of the viewport")
