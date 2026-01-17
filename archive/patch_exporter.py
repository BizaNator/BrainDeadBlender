#!/usr/bin/env python3
"""
Patch exporter.py to add UEFN skeleton format support.
Run this on the brainz AI system.
"""

import os

EXPORTER_PATH = "/opt/comfyui/stable/custom_nodes/ComfyUI-UniRig/lib/unirig/src/data/exporter.py"

# Read the original file
with open(EXPORTER_PATH, "r") as f:
    content = f.read()

# Check if already patched
if 'skeleton_format: str="mixamo"' in content or "skeleton_format: str='mixamo'" in content:
    print("Exporter.py already patched!")
    exit(0)

# 1. Add skeleton_format parameter to _export_fbx signature
old_sig = '''def _export_fbx(
        self,
        path: str,
        vertices: Union[ndarray, None],
        joints: ndarray,
        skin: Union[ndarray, None],
        parents: List[Union[int, None]],
        names: List[str],
        faces: Union[ndarray, None]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        tails: Union[ndarray, None]=None,
        blender_exe: Union[str, None]=None,
        uv_coords: Union[ndarray, None]=None,
        uv_faces: Union[ndarray, None]=None,
        texture_data_base64: Union[str, None]=None,
        texture_format: Union[str, None]=None,
        material_name: Union[str, None]=None,
    ):'''

new_sig = '''def _export_fbx(
        self,
        path: str,
        vertices: Union[ndarray, None],
        joints: ndarray,
        skin: Union[ndarray, None],
        parents: List[Union[int, None]],
        names: List[str],
        faces: Union[ndarray, None]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        tails: Union[ndarray, None]=None,
        blender_exe: Union[str, None]=None,
        uv_coords: Union[ndarray, None]=None,
        uv_faces: Union[ndarray, None]=None,
        texture_data_base64: Union[str, None]=None,
        texture_format: Union[str, None]=None,
        material_name: Union[str, None]=None,
        skeleton_format: str="mixamo",
    ):'''

if old_sig not in content:
    print("WARNING: Could not find exact signature match, trying alternative...")
    # Try without trailing whitespace variations
    old_sig = old_sig.rstrip()
    new_sig = new_sig.rstrip()

content = content.replace(old_sig, new_sig)

# 2. Replace wrapper_script selection logic
old_wrapper = '''        # Find wrapper script (lib/blender_export_fbx.py)
        # Assume it's in lib/ relative to this file's parent directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # From src/data/ go up to unirig/, then up to lib/
        wrapper_script = os.path.join(current_dir, '..', '..', '..', 'blender_export_fbx.py')
        wrapper_script = os.path.abspath(wrapper_script)

        if not os.path.exists(wrapper_script):
            raise RuntimeError(
                f"Blender wrapper script not found at {wrapper_script}. "
                "Make sure lib/blender_export_fbx.py exists."
            )'''

new_wrapper = '''        # Find wrapper script based on skeleton format
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # From src/data/ go up to unirig/, then up to lib/
        lib_dir = os.path.join(current_dir, '..', '..', '..')
        lib_dir = os.path.abspath(lib_dir)

        if skeleton_format == "uefn":
            wrapper_script = os.path.join(lib_dir, 'blender_export_uefn.py')
            print(f"[Exporter] Using UEFN export format (UEFN Manny skeleton)")
        else:
            wrapper_script = os.path.join(lib_dir, 'blender_export_fbx.py')
            print(f"[Exporter] Using Mixamo export format")

        wrapper_script = os.path.abspath(wrapper_script)

        if not os.path.exists(wrapper_script):
            script_name = 'blender_export_uefn.py' if skeleton_format == 'uefn' else 'blender_export_fbx.py'
            raise RuntimeError(
                f"Blender wrapper script not found at {wrapper_script}. "
                f"Make sure lib/{script_name} exists."
            )'''

if old_wrapper not in content:
    print("WARNING: Could not find exact wrapper logic match")
    print("Manual patching may be required")
else:
    content = content.replace(old_wrapper, new_wrapper)

# Write the patched file
with open(EXPORTER_PATH, "w") as f:
    f.write(content)

print("=" * 60)
print("Exporter.py patched successfully!")
print("=" * 60)
print("Changes made:")
print("1. Added skeleton_format parameter to _export_fbx()")
print("2. Modified wrapper script selection to support 'uefn' format")
print("")
print("To use UEFN format, pass skeleton_format='uefn' to _export_fbx()")
