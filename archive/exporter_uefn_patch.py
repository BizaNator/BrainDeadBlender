"""
Patch for exporter.py to add UEFN skeleton format support.

This adds a `skeleton_format` parameter to _export_fbx that can be:
- "mixamo" (default): Uses blender_export_fbx.py for Mixamo skeleton output
- "uefn": Uses blender_export_uefn.py for UEFN Manny skeleton output

Apply this patch by adding the skeleton_format parameter to _export_fbx signature
and modifying the wrapper_script selection logic.
"""

# Add this parameter to _export_fbx signature:
# skeleton_format: str = "mixamo",

# Then replace the wrapper_script selection logic with:

WRAPPER_SCRIPT_SELECTION = '''
        # Find wrapper script based on skeleton format
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # From src/data/ go up to unirig/, then up to lib/
        lib_dir = os.path.join(current_dir, '..', '..', '..')
        lib_dir = os.path.abspath(lib_dir)

        if skeleton_format == "uefn":
            wrapper_script = os.path.join(lib_dir, 'blender_export_uefn.py')
            print(f"[Exporter] Using UEFN export format")
        else:
            wrapper_script = os.path.join(lib_dir, 'blender_export_fbx.py')
            print(f"[Exporter] Using Mixamo export format")

        wrapper_script = os.path.abspath(wrapper_script)

        if not os.path.exists(wrapper_script):
            raise RuntimeError(
                f"Blender wrapper script not found at {wrapper_script}. "
                f"Make sure lib/blender_export_{skeleton_format if skeleton_format == 'uefn' else 'fbx'}.py exists."
            )
'''

print("Patch instructions:")
print("1. Add 'skeleton_format: str = \"mixamo\",' to _export_fbx parameters")
print("2. Replace wrapper_script selection with the code above")
print("3. Pass skeleton_format from ComfyUI nodes")
