"""One-shot: load iter_22b, build all 8 joints, save as iter_24.

Paste this into Blender's Text Editor and run (Alt+P / Run Script button).
Or in the Python console:

    exec(open(r"B:\Brains\Tools\BrainDeadBlender\scripts\face_base_pipeline\run_iter24_joints.py").read())
"""

import sys
import bpy

SCRIPT_DIR = r"B:\Brains\Tools\BrainDeadBlender\scripts\face_base_pipeline"
ITER_22B   = r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\VABase\Female\iter\iter_22b_full_anatomy.blend"
ITER_24    = r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\VABase\Female\iter\iter_24_joints.blend"

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Force re-import in case the modules were already loaded
for m in ("limb_primitives_lib", "limb_joints_assembly"):
    if m in sys.modules:
        del sys.modules[m]

# Load iter_22b if not already loaded
current = bpy.data.filepath
if current != ITER_22B:
    print(f"Opening {ITER_22B}")
    bpy.ops.wm.open_mainfile(filepath=ITER_22B)

# Import joint assembly (these expect the modules to be importable)
import limb_primitives_lib  # noqa: F401  (loaded for use by limb_joints_assembly)
import limb_joints_assembly as lja

# Reload just in case
import importlib
importlib.reload(limb_primitives_lib)
importlib.reload(lja)

# Build all 8 joints
results = lja.assemble_all_joints_for_iter22b()

print("\n=== Joints built ===")
for joint_name, parts in results.items():
    primitives = [k for k in parts if not k.endswith("_anchor") and k != "root"]
    print(f"  {joint_name}: {primitives}")

# Save as iter_24
print(f"\nSaving to {ITER_24}")
bpy.ops.wm.save_as_mainfile(filepath=ITER_24)
print("Done.")
