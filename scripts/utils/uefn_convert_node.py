"""
UniRig UEFN Conversion Node

Converts Mixamo-rigged FBX to UEFN Manny skeleton.
Drop this file into: ComfyUI-UniRig/nodes/
Then add to nodes/__init__.py

Usage:
    UniRigAutoRig -> UniRigConvertToUEFN -> (UEFN-ready FBX)
"""

import os
import sys
import subprocess
import time
import folder_paths

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .base import LIB_DIR, BLENDER_EXE, setup_subprocess_env
except ImportError:
    # Fallback for standalone testing
    LIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BLENDER_EXE = os.environ.get('BLENDER_EXE', 'blender')
    def setup_subprocess_env():
        return os.environ.copy()


class UniRigConvertToUEFN:
    """
    Convert Mixamo-rigged FBX to UEFN Manny skeleton.

    Takes the FBX output from UniRigAutoRig (Mixamo format) and converts
    bone names to UEFN naming convention for use in Fortnite/UEFN.

    Bone mapping:
    - mixamorig:Hips -> pelvis
    - mixamorig:Spine -> spine_01
    - mixamorig:LeftArm -> upperarm_l
    - etc.

    Also adds a 'root' bone if not present.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_input_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to Mixamo-rigged FBX (output from UniRigAutoRig)"
                }),
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output filename (without extension). If empty, appends '_uefn' to input name."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("uefn_fbx_path",)
    FUNCTION = "convert_to_uefn"
    CATEGORY = "UniRig"

    def convert_to_uefn(self, fbx_input_path, output_name=""):
        """Convert Mixamo FBX to UEFN format."""
        print(f"[UniRigConvertToUEFN] Starting Mixamo -> UEFN conversion...")
        print(f"[UniRigConvertToUEFN] Input: {fbx_input_path}")

        # Validate input exists
        if not fbx_input_path or not os.path.exists(fbx_input_path):
            raise RuntimeError(f"Input FBX not found: {fbx_input_path}")

        # Determine output path
        input_dir = os.path.dirname(fbx_input_path)
        input_basename = os.path.basename(fbx_input_path)
        input_name = os.path.splitext(input_basename)[0]

        if output_name and output_name.strip():
            out_name = output_name.strip()
            if out_name.lower().endswith('.fbx'):
                out_name = out_name[:-4]
            output_filename = f"{out_name}.fbx"
        else:
            # Replace 'mixamo' suffix with 'uefn' if present, otherwise append '_uefn'
            if '_mixamo' in input_name:
                output_filename = input_name.replace('_mixamo', '_uefn') + '.fbx'
            else:
                output_filename = f"{input_name}_uefn.fbx"

        # Put in ComfyUI output directory
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, output_filename)

        print(f"[UniRigConvertToUEFN] Output: {output_path}")

        # Find the conversion script
        converter_script = os.path.join(LIB_DIR, 'blender_convert_mixamo_to_uefn.py')
        if not os.path.exists(converter_script):
            raise RuntimeError(
                f"Conversion script not found at {converter_script}. "
                f"Make sure lib/blender_convert_mixamo_to_uefn.py exists."
            )

        # Find Blender executable
        blender_exe = BLENDER_EXE
        if not blender_exe or not os.path.exists(blender_exe):
            blender_exe = os.environ.get('BLENDER_EXE')
        if not blender_exe or not os.path.exists(blender_exe):
            raise RuntimeError(
                "Blender executable not found. "
                "Set BLENDER_EXE environment variable or check UniRig installation."
            )

        # Build command
        cmd = [
            blender_exe,
            '--background',
            '--python', converter_script,
            '--',
            fbx_input_path,
            output_path,
        ]

        print(f"[UniRigConvertToUEFN] Running Blender conversion...")
        start_time = time.time()

        try:
            env = setup_subprocess_env()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=120
            )

            if result.stdout:
                # Print key lines from output
                for line in result.stdout.split('\n'):
                    if '[Mixamo->UEFN]' in line:
                        print(line)

            if result.returncode != 0:
                print(f"[UniRigConvertToUEFN] Blender stderr:\n{result.stderr}")
                raise RuntimeError(
                    f"Conversion failed with exit code {result.returncode}"
                )

            conversion_time = time.time() - start_time
            print(f"[UniRigConvertToUEFN] Conversion completed in {conversion_time:.2f}s")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Conversion timed out (>120s)")
        except Exception as e:
            raise RuntimeError(f"Conversion failed: {str(e)}")

        # Verify output exists
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output FBX not created: {output_path}")

        file_size = os.path.getsize(output_path)
        print(f"[UniRigConvertToUEFN] Output file size: {file_size} bytes")
        print(f"[UniRigConvertToUEFN] UEFN-ready FBX: {output_path}")

        return (output_path,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "UniRigConvertToUEFN": UniRigConvertToUEFN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigConvertToUEFN": "Convert to UEFN Skeleton",
}
