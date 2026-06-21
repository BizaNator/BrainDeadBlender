"""
BrainDeadBlender — Local AutoRig Orchestrator

Self-contained, no ComfyUI required. Wraps:
  1. autorig_bootstrap.bootstrap()       — venv + deps + MIA source clone
  2. autorig_bootstrap.ensure_mia_weights() — HF model weight download
  3. subprocess(venv_python, autorig_runner.py)   — inference (PyTorch)
  4. subprocess(blender, mia_export.py)          — Blender FBX assembly
  5. BD_MixamoToUEFN rename (in-process Python)   — UEFN bone names

Called by the BD AutoRig sidebar panel when backend = "Local".
"""

import bpy
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from . import autorig_bootstrap as _bootstrap


# Paths to the vendored Blender script + Mixamo template
_THIS_DIR = Path(__file__).resolve().parent
VENDOR_DIR = _THIS_DIR / "autorig_vendor"
MIA_EXPORT_SCRIPT = VENDOR_DIR / "mia_export.py"
MIXAMO_TEMPLATE = VENDOR_DIR / "mixamo.fbx"

# Mirror of the Mixamo→UEFN bone map from
# ComfyUI-BrainDead/nodes/autorig/bone_remap.py — kept in sync there.
MIXAMO_TO_UEFN: dict[str, str] = {
    "Hips":         "pelvis",
    "Spine":        "spine_01",
    "Spine1":       "spine_02",
    "Spine2":       "spine_03",
    "Spine3":       "spine_04",
    "Neck":         "neck_01",
    "Neck1":        "neck_02",
    "Head":         "head",
    "LeftShoulder":  "clavicle_l",
    "LeftArm":       "upperarm_l",
    "LeftForeArm":   "lowerarm_l",
    "LeftHand":      "hand_l",
    "RightShoulder": "clavicle_r",
    "RightArm":      "upperarm_r",
    "RightForeArm":  "lowerarm_r",
    "RightHand":     "hand_r",
    "LeftHandThumb1":  "thumb_01_l",
    "LeftHandThumb2":  "thumb_02_l",
    "LeftHandThumb3":  "thumb_03_l",
    "LeftHandIndex1":  "index_01_l",
    "LeftHandIndex2":  "index_02_l",
    "LeftHandIndex3":  "index_03_l",
    "LeftHandMiddle1": "middle_01_l",
    "LeftHandMiddle2": "middle_02_l",
    "LeftHandMiddle3": "middle_03_l",
    "LeftHandRing1":   "ring_01_l",
    "LeftHandRing2":   "ring_02_l",
    "LeftHandRing3":   "ring_03_l",
    "LeftHandPinky1":  "pinky_01_l",
    "LeftHandPinky2":  "pinky_02_l",
    "LeftHandPinky3":  "pinky_03_l",
    "RightHandThumb1":  "thumb_01_r",
    "RightHandThumb2":  "thumb_02_r",
    "RightHandThumb3":  "thumb_03_r",
    "RightHandIndex1":  "index_01_r",
    "RightHandIndex2":  "index_02_r",
    "RightHandIndex3":  "index_03_r",
    "RightHandMiddle1": "middle_01_r",
    "RightHandMiddle2": "middle_02_r",
    "RightHandMiddle3": "middle_03_r",
    "RightHandRing1":   "ring_01_r",
    "RightHandRing2":   "ring_02_r",
    "RightHandRing3":   "ring_03_r",
    "RightHandPinky1":  "pinky_01_r",
    "RightHandPinky2":  "pinky_02_r",
    "RightHandPinky3":  "pinky_03_r",
    "LeftUpLeg":   "thigh_l",
    "LeftLeg":     "calf_l",
    "LeftFoot":    "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg":  "thigh_r",
    "RightLeg":    "calf_r",
    "RightFoot":   "foot_r",
    "RightToeBase":"ball_r",
}


def _strip_mixamo_prefix(name: str) -> str:
    for pfx in ("mixamorig:", "Armature|", "Armature:", "Avatar:"):
        if name.startswith(pfx):
            return name[len(pfx):]
    return name


def remap_imported_to_uefn(arm_obj: bpy.types.Object) -> tuple[int, int, list[str]]:
    """Walk arm_obj's bones + every weighted mesh's vertex groups, applying
    the Mixamo→UEFN rename. Returns (renamed_bones, renamed_vgroups, unmapped).
    No external Blender process — runs in the active session."""
    if arm_obj.type != "ARMATURE":
        raise ValueError(f"remap target must be ARMATURE, got {arm_obj.type}")

    # Bones (edit mode required to rename)
    bpy.context.view_layer.objects.active = arm_obj
    try:
        bpy.ops.object.mode_set(mode="EDIT")
    except RuntimeError:
        pass

    renamed_bones = 0
    unmapped: list[str] = []
    for b in list(arm_obj.data.edit_bones):
        stripped = _strip_mixamo_prefix(b.name)
        if stripped != b.name:
            b.name = stripped
        if b.name in MIXAMO_TO_UEFN:
            tgt = MIXAMO_TO_UEFN[b.name]
            if tgt != b.name:
                b.name = tgt
                renamed_bones += 1
        elif b.name not in MIXAMO_TO_UEFN.values():
            unmapped.append(b.name)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    # Vertex groups on every mesh
    renamed_vgroups = 0
    for m in bpy.data.objects:
        if m.type != "MESH":
            continue
        for vg in list(m.vertex_groups):
            stripped = _strip_mixamo_prefix(vg.name)
            if stripped != vg.name:
                vg.name = stripped
            if vg.name in MIXAMO_TO_UEFN:
                tgt = MIXAMO_TO_UEFN[vg.name]
                if tgt != vg.name:
                    vg.name = tgt
                    renamed_vgroups += 1
    return renamed_bones, renamed_vgroups, unmapped


def _blender_binary() -> str:
    """Path to the currently-running Blender executable."""
    return bpy.app.binary_path


def run_local_autorig(
    glb_path: Path,
    output_fbx: Path,
    *,
    no_fingers: bool = True,
    use_normal: bool = False,
    reset_to_rest: bool = True,
    progress_cb=None,
) -> tuple[bool, str]:
    """End-to-end local autorig:
      1. Ensure venv + MIA source + weights
      2. Run inference via venv subprocess
      3. Run Blender FBX assembly via subprocess
    Returns (ok, message).
    """
    if progress_cb is None:
        progress_cb = lambda msg: print(f"[BD_AutoRig:local] {msg}", flush=True)

    # 1) Bootstrap (idempotent)
    if not _bootstrap.is_bootstrapped():
        progress_cb("First run — bootstrapping venv (this can take "
                     "5–15 min depending on network)…")
        ok = _bootstrap.bootstrap(progress_cb=progress_cb)
        if not ok:
            return False, "Bootstrap failed — see system console"

    # 1.5) Ensure weights
    ok = _bootstrap.ensure_mia_weights(progress_cb=progress_cb)
    if not ok:
        return False, "MIA weight download failed"

    venv_py = _bootstrap.venv_python()

    # 2) Inference subprocess
    work_dir = Path(tempfile.mkdtemp(prefix="bd_autorig_"))
    progress_cb(f"Running MIA inference (output → {work_dir})…")
    runner_script = _THIS_DIR / "autorig_runner.py"
    cmd = [
        str(venv_py), str(runner_script),
        "--input", str(glb_path),
        "--output_dir", str(work_dir),
    ]
    if no_fingers:    cmd.append("--no_fingers")
    if use_normal:    cmd.append("--use_normal")
    if reset_to_rest: cmd.append("--reset_to_rest")

    env = os.environ.copy()
    env["BD_AUTORIG_CACHE"] = str(_bootstrap.CACHE_ROOT)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=600)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            progress_cb(f"  inf: {line}")
    if proc.returncode != 0:
        progress_cb(f"  inf STDERR:\n{proc.stderr[-2000:]}")
        return False, f"Inference failed (rc={proc.returncode})"

    # 3) Blender FBX assembly subprocess
    progress_cb("Assembling rigged FBX via Blender…")
    json_path = work_dir / "data.json"
    cmd = [
        _blender_binary(),
        "--background",
        "--python", str(MIA_EXPORT_SCRIPT),
        "--",
        "--input_path", str(json_path),
        "--output_path", str(output_fbx),
        "--template_path", str(MIXAMO_TEMPLATE),
    ]
    if no_fingers:    cmd.append("--remove_fingers")
    if reset_to_rest: cmd.append("--reset_to_rest")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.stdout:
        for line in proc.stdout.splitlines()[-30:]:
            progress_cb(f"  fbx: {line}")
    if proc.returncode != 0:
        progress_cb(f"  fbx STDERR:\n{proc.stderr[-2000:]}")
        return False, f"FBX assembly failed (rc={proc.returncode})"
    if not output_fbx.exists():
        return False, f"FBX not written: {output_fbx}"

    progress_cb(f"OK — rigged FBX at {output_fbx}")
    return True, str(output_fbx)
