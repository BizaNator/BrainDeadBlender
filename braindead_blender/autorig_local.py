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


_BONE_NAME_CANDIDATES = {
    # Vertex groups are usually already UEFN-named at align-time (mia_export
    # renames them during FBX assembly), but the bones may still have the
    # Mixamo prefix. List all forms; first match wins.
    "head":   ("head", "Head", "mixamorig:Head"),
    "pelvis": ("pelvis", "Hips", "mixamorig:Hips"),
    # Left/right lateral. Prefer clavicle since hand bones are merged out
    # when no_fingers=True. Fall back to upperarm/lowerarm/foot.
    "lat_l": ("clavicle_l", "upperarm_l", "lowerarm_l", "foot_l",
                "LeftShoulder", "LeftArm", "mixamorig:LeftShoulder",
                "mixamorig:LeftArm", "mixamorig:LeftFoot"),
    "lat_r": ("clavicle_r", "upperarm_r", "lowerarm_r", "foot_r",
                "RightShoulder", "RightArm", "mixamorig:RightShoulder",
                "mixamorig:RightArm", "mixamorig:RightFoot"),
    "foot_l": ("foot_l", "LeftFoot", "mixamorig:LeftFoot"),
}


def _find_named(container, candidates):
    """Find the first matching name in `container` (vertex_groups or bones).
    `container` must support .get(name)."""
    for name in candidates:
        x = container.get(name)
        if x is not None:
            return x
    return None


def _vgroup_centroid(mesh: bpy.types.Object, vg_name: str):
    """Mean position of vertices weighted to the given vertex group, in
    mesh-data space. Returns None if no weighted verts found."""
    from mathutils import Vector
    vg = mesh.vertex_groups.get(vg_name)
    if vg is None:
        return None
    total = Vector((0, 0, 0))
    total_w = 0.0
    idx = vg.index
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == idx:
                total += v.co * g.weight
                total_w += g.weight
                break
    if total_w < 1e-9:
        return None
    return total / total_w


def _build_align_rotation(mesh: bpy.types.Object,
                              bone_targets: dict):
    """Compute a rotation matrix that aligns the mesh's anatomy to the
    armature's anatomy.

    Strategy:
      1. up_mesh    = (head_centroid - pelvis_centroid).normalized
      2. up_target  = (head_bone_world_pos - pelvis_bone_world_pos).normalized
      3. lr_mesh    = (hand_l_centroid - hand_r_centroid).normalized
      4. lr_target  = (hand_l_bone - hand_r_bone).normalized
      5. Build an orthonormal target frame from (lr_target, up_target)
         and a mesh frame from (lr_mesh, up_mesh), then R = target * mesh⁻¹.

    Args:
        mesh: with vertex groups still in Mixamo naming
        bone_targets: dict[name → world Vector] for "head", "pelvis",
            "hand_l", "hand_r" (precomputed from the bones)

    Returns:
        (R: mathutils.Matrix, debug: dict) or (None, debug) on failure
    """
    from mathutils import Vector, Matrix

    # Mesh centroids (mesh-local space)
    c_head    = _vgroup_centroid(mesh, _vg_name_for(mesh, "head"))
    c_pelvis  = _vgroup_centroid(mesh, _vg_name_for(mesh, "pelvis"))
    c_lat_l   = _vgroup_centroid(mesh, _vg_name_for(mesh, "lat_l"))
    c_lat_r   = _vgroup_centroid(mesh, _vg_name_for(mesh, "lat_r"))

    debug = {
        "mesh_centroids": {
            "head": list(c_head) if c_head else None,
            "pelvis": list(c_pelvis) if c_pelvis else None,
            "lat_l": list(c_lat_l) if c_lat_l else None,
            "lat_r": list(c_lat_r) if c_lat_r else None,
        },
        "bone_targets": {k: list(v) for k, v in bone_targets.items()},
        "vg_names_used": {
            "head":   _vg_name_for(mesh, "head"),
            "pelvis": _vg_name_for(mesh, "pelvis"),
            "lat_l":  _vg_name_for(mesh, "lat_l"),
            "lat_r":  _vg_name_for(mesh, "lat_r"),
        },
    }

    if c_head is None or c_pelvis is None or c_lat_l is None or c_lat_r is None:
        debug["error"] = "missing centroids (head/pelvis/lat_l/lat_r)"
        return None, debug
    if "head" not in bone_targets or "pelvis" not in bone_targets:
        debug["error"] = "missing head/pelvis bone target"
        return None, debug
    if "lat_l" not in bone_targets or "lat_r" not in bone_targets:
        debug["error"] = "missing lat_l/lat_r bone target"
        return None, debug

    bt_head, bt_pelvis = bone_targets["head"], bone_targets["pelvis"]
    bt_lat_l, bt_lat_r = bone_targets["lat_l"], bone_targets["lat_r"]

    up_mesh = (c_head - c_pelvis).normalized()
    up_targ = (bt_head - bt_pelvis).normalized()
    lr_mesh = (c_lat_l - c_lat_r).normalized()
    lr_targ = (bt_lat_l - bt_lat_r).normalized()

    # Orthogonalize the lateral axis against up
    lr_mesh = (lr_mesh - lr_mesh.dot(up_mesh) * up_mesh).normalized()
    lr_targ = (lr_targ - lr_targ.dot(up_targ) * up_targ).normalized()
    fw_mesh = up_mesh.cross(lr_mesh).normalized()
    fw_targ = up_targ.cross(lr_targ).normalized()

    # 3×3 frames as column vectors
    M_mesh = Matrix((lr_mesh, fw_mesh, up_mesh)).transposed()
    M_targ = Matrix((lr_targ, fw_targ, up_targ)).transposed()
    R = M_targ @ M_mesh.inverted()
    debug["up_mesh"], debug["up_targ"] = list(up_mesh), list(up_targ)
    debug["lr_mesh"], debug["lr_targ"] = list(lr_mesh), list(lr_targ)
    return R, debug


def _vg_name_for(mesh: bpy.types.Object, role: str) -> str:
    """Return the actual vertex group name on `mesh` for the given canonical
    role (head/pelvis/hand_l/...). Picks the first candidate that exists."""
    cands = _BONE_NAME_CANDIDATES.get(role, (role,))
    for c in cands:
        if c in mesh.vertex_groups:
            return c
    return role  # fallback (likely missing)


def align_imported_to_uefn(arm: bpy.types.Object,
                              mesh: bpy.types.Object,
                              source_mesh: Optional[bpy.types.Object] = None
                              ) -> dict:
    """Land the FBX import in canonical UEFN coords without hardcoding
    MIA's particular orientation.

    The FBX from autorig_vendor/mia_export.py needs:
      1. Identity armature object transform (FBX scale 0.01 + 90°X applied)
      2. Mesh data scaled to UEFN-skeleton height (~2m)
      3. Mesh data rotated so its anatomy lines up with the armature
         (head-over-pelvis matches the armature, arms-out matches)
      4. Mesh translated so feet sit on Z=0

    Steps 3 is the brittle one — we use weighted-vertex-group centroids
    on the mesh (head/pelvis/hand_l/hand_r) and corresponding bone world
    positions to *derive* the rotation, so it auto-corrects regardless of
    what orientation MIA emitted.
    """
    import bpy as _bpy
    from mathutils import Vector, Matrix
    from math import radians

    # 1) Apply armature+mesh transforms so armature is identity & in meters
    _bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True); mesh.select_set(True)
    _bpy.context.view_layer.objects.active = arm
    _bpy.ops.object.transform_apply(location=False, rotation=True, scale=True,
                                       properties=True, isolate_users=False)

    # 2) Always scale to UEFN-skeleton height (the armature's Z extent post-
    #    apply is exactly that). Fall back to source_mesh height if armature
    #    is degenerate.
    arm_pts = [arm.matrix_world @ Vector(c) for c in arm.bound_box]
    arm_h = max(p.z for p in arm_pts) - min(p.z for p in arm_pts)
    target_h = arm_h if arm_h > 0.5 else 1.8

    mesh_pts = [Vector(c) for c in mesh.bound_box]
    mesh_diag = (Vector(mesh.bound_box[6]) - Vector(mesh.bound_box[0])).length
    # We don't know which mesh axis is "height" yet — use diagonal as a
    # conservative size, refine after rotation.
    pre_scale = (target_h * 1.8) / max(mesh_diag, 1e-9)

    # 3) Pre-scale + parent_clear so subsequent matrix ops are cheap
    _bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    _bpy.context.view_layer.objects.active = mesh
    _bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    _bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    mesh.scale = (pre_scale, pre_scale, pre_scale)
    _bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 4) Gather bone world-position targets
    def _bone_world(bn):
        b = arm.data.bones.get(bn)
        return arm.matrix_world @ b.head_local if b else None
    bone_targets = {}
    for role in ("head", "pelvis", "lat_l", "lat_r", "foot_l"):
        for cand in _BONE_NAME_CANDIDATES[role]:
            v = _bone_world(cand)
            if v is not None:
                bone_targets[role] = v
                break

    debug = {"pre_scale": pre_scale, "target_h": target_h}

    # 5) Compute alignment rotation from anatomical centroids
    R, align_dbg = _build_align_rotation(mesh, bone_targets)
    debug["align"] = align_dbg

    if R is not None:
        # Apply the rotation matrix to mesh data
        rot_4 = R.to_4x4()
        mesh.matrix_world = rot_4 @ mesh.matrix_world
        _bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        debug["rotated_via"] = "anatomical_alignment"
    else:
        # Fallback: try Rx(-90)·Rz(180) (the empirically-correct sequence
        # for MIA's current output)
        mesh.rotation_euler = (radians(-90), 0, radians(180))
        _bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        debug["rotated_via"] = "fallback_rx90_rz180"

    # 6) Refine scale: now mesh is oriented correctly, scale so its Z extent
    #    matches the armature's exactly.
    mesh_z_min = min(v.co.z for v in mesh.data.vertices)
    mesh_z_max = max(v.co.z for v in mesh.data.vertices)
    mesh_h = mesh_z_max - mesh_z_min
    if mesh_h > 1e-6:
        refine_scale = target_h / mesh_h
        if abs(refine_scale - 1.0) > 0.01:
            mesh.scale = (refine_scale, refine_scale, refine_scale)
            _bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            debug["refine_scale"] = refine_scale

    # 7) Translate so feet on Z=0
    mz_min = min(v.co.z for v in mesh.data.vertices)
    if abs(mz_min) > 1e-4:
        mesh.location = (0, 0, -mz_min)
        _bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # 8) Re-parent to armature
    arm.select_set(True)
    _bpy.context.view_layer.objects.active = arm
    _bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    debug["mesh_world_bbox_after"] = [
        list(mesh.matrix_world @ Vector(mesh.bound_box[0])),
        list(mesh.matrix_world @ Vector(mesh.bound_box[6])),
    ]
    debug["arm_world_bbox_after"] = [
        list(arm.matrix_world @ Vector(arm.bound_box[0])),
        list(arm.matrix_world @ Vector(arm.bound_box[6])),
    ]
    return debug


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
    env["BD_AUTORIG_CACHE"] = str(_bootstrap.actual_cache_root())
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=600, errors="replace")
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

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            errors="replace")
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
