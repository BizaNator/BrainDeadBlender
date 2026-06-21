"""
BrainDeadBlender — AutoRig Inference Runner

Runs inside the bootstrapped venv (NOT inside Blender). Mirrors the proven
ComfyUI-BrainDead lib/autorig/mia_inference.py pattern: loads the 5 MIA
PCAE models with the correct constructor kwargs, runs prepare→preprocess→
infer→bw_post_process, and dumps result data as JSON + raw binary for the
in-Blender FBX assembler (mia_export.py) to consume.

Invocation:
    <venv>/python autorig_runner.py \\
        --input <mesh.glb> --output_dir <tmp_dir> \\
        [--no_fingers] [--use_normal] [--reset_to_rest]

Outputs in <output_dir>:
    data.json     metadata + paths to bw.bin, joints.bin, mesh.glb
    bw.bin        skinning weights (float32, shape = [V, num_bones])
    joints.bin    bone head positions (float32, shape = [num_bones, 3])
    joints_tail.bin   bone tail positions (optional)
    pose.bin      pose deltas to reset_to_rest (optional)
    mesh.glb      possibly-resampled mesh (trimesh)
"""

import argparse
import json
import os
import sys
from pathlib import Path


# Cache root passed in via env by autorig_local.py
_CACHE_ROOT = Path(os.environ.get("BD_AUTORIG_CACHE",
                                      Path.home() / ".cache" /
                                      "braindead_blender" / "autorig"))
MIA_MODELS_DIR = _CACHE_ROOT / "models" / "mia"

# Vendored MIA package (pruned of bpy imports). Lives next to this script
# at autorig_vendor/mia/. We import it as a top-level package "mia".
_VENDOR_DIR = Path(__file__).resolve().parent / "autorig_vendor"
if (_VENDOR_DIR / "mia").exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))


def _mock_bpy():
    """Vendored MIA still references bpy at module load (in training-time
    helpers). Mock so imports don't fail. Inference itself never calls bpy."""
    from unittest.mock import MagicMock
    if "bpy" not in sys.modules:
        sys.modules["bpy"] = MagicMock()


def _resolve_models_dir() -> Path:
    """The dir holding the 5 .pth files."""
    p = MIA_MODELS_DIR / "output" / "best" / "new"
    if p.exists() and any(p.glob("*.pth")):
        return p
    raise RuntimeError(
        f"MIA model weights not found under {p}. Bootstrap the autorig "
        "(BD AutoRig panel → Install Local Autorig).")


def load_mia_models(cache_to_gpu: bool = True):
    """Load the 5 MIA PCAE models. Mirrors ComfyUI-BrainDead's
    lib/autorig/mia_inference.py::load_mia_models exactly."""
    _mock_bpy()
    import torch
    from mia.model import PCAE  # type: ignore
    from mia.dataset_mixamo import JOINTS_NUM, KINEMATIC_TREE  # type: ignore

    device = torch.device("cuda" if (cache_to_gpu and torch.cuda.is_available())
                            else "cpu")
    print(f"[runner] device={device}", flush=True)

    models_dir = _resolve_models_dir()
    N = 32768
    hands_resample_ratio = 0.5
    geo_resample_ratio = 0.0
    hierarchical_ratio = hands_resample_ratio + geo_resample_ratio

    print(f"[runner] loading joints_coarse…", flush=True)
    model_coarse = PCAE(
        N=N, input_normal=False, deterministic=True,
        output_dim=JOINTS_NUM,
        predict_bw=False, predict_joints=True, predict_joints_tail=True,
    )
    model_coarse.load(str(models_dir / "joints_coarse.pth")).to(device).eval()

    print(f"[runner] loading bw…", flush=True)
    model_bw = PCAE(
        N=N, input_normal=False, input_attention=False,
        deterministic=True, hierarchical_ratio=hierarchical_ratio,
    )
    model_bw.load(str(models_dir / "bw.pth")).to(device).eval()

    print(f"[runner] loading bw_normal…", flush=True)
    model_bw_normal = PCAE(
        N=N, input_normal=True, input_attention=True,
        deterministic=True, hierarchical_ratio=hierarchical_ratio,
    )
    model_bw_normal.load(str(models_dir / "bw_normal.pth")).to(device).eval()

    print(f"[runner] loading joints…", flush=True)
    model_joints = PCAE(
        N=N, input_normal=False, deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM, kinematic_tree=KINEMATIC_TREE,
        predict_bw=False, predict_joints=True, predict_joints_tail=True,
        joints_attn_causal=True,
    )
    model_joints.load(str(models_dir / "joints.pth")).to(device).eval()

    print(f"[runner] loading pose…", flush=True)
    model_pose = PCAE(
        N=N, input_normal=False, deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM, kinematic_tree=KINEMATIC_TREE,
        predict_bw=False, predict_pose_trans=True,
        pose_mode="ortho6d", pose_input_joints=True, pose_attn_causal=True,
    )
    model_pose.load(str(models_dir / "pose.pth")).to(device).eval()

    return {
        "device": device,
        "N": N,
        "hands_resample_ratio": hands_resample_ratio,
        "geo_resample_ratio": geo_resample_ratio,
        "model_coarse":    model_coarse,
        "model_bw":        model_bw,
        "model_bw_normal": model_bw_normal,
        "model_joints":    model_joints,
        "model_pose":      model_pose,
    }


def run_inference(mesh_path: str, output_dir: Path, *,
                    no_fingers: bool, use_normal: bool,
                    reset_to_rest: bool):
    """End-to-end inference: load mesh → MIA pipeline → write artifacts."""
    _mock_bpy()
    import numpy as np
    import torch
    import trimesh

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[runner] loading mesh: {mesh_path}", flush=True)
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Could not load mesh as Trimesh: {mesh_path}")

    models = load_mia_models()

    from mia.pipeline import prepare_input, preprocess, infer, bw_post_process  # type: ignore
    from mia.dataset_mixamo import BONES_IDX_DICT  # type: ignore

    print(f"[runner] preparing input…", flush=True)
    data = prepare_input(
        mesh,
        N=models["N"],
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        get_normals=use_normal,
    )

    print(f"[runner] preprocessing…", flush=True)
    data = preprocess(
        data,
        model_coarse=models["model_coarse"],
        device=models["device"],
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        N=models["N"],
    )

    print(f"[runner] inference…", flush=True)
    data = infer(
        data,
        model_bw=models["model_bw"],
        model_bw_normal=models["model_bw_normal"],
        model_joints=models["model_joints"],
        model_pose=models["model_pose"],
        device=models["device"],
        use_normal=use_normal,
    )

    print(f"[runner] post-processing…", flush=True)
    joints = data.joints
    head_idx = BONES_IDX_DICT["mixamorig:Head"]
    head_y = joints[..., head_idx, 4]  # tail y (cols 3:6 are tail)
    above_head_mask = data.verts[..., 1] >= head_y
    bw = bw_post_process(
        data.bw,
        bones_idx_dict=BONES_IDX_DICT,
        above_head_mask=above_head_mask,
        no_fingers=no_fingers,
    )

    # ── Write outputs in mia_export.py-compatible format ────────────────────
    joints_np = data.joints.squeeze(0).cpu().numpy() if hasattr(
        data.joints, "cpu") else np.asarray(data.joints).squeeze(0)
    bw_np = bw.squeeze(0).cpu().numpy() if hasattr(
        bw, "cpu") else np.asarray(bw).squeeze(0)

    pose_np = None
    if reset_to_rest and getattr(data, "pose", None) is not None:
        pose_np = (data.pose.squeeze(0).cpu().numpy()
                   if hasattr(data.pose, "cpu")
                   else np.asarray(data.pose).squeeze(0))

    joints_head = joints_np[..., :3]
    joints_tail = (joints_np[..., 3:] if joints_np.shape[-1] >= 6 else None)

    # Resampled mesh
    out_mesh_path = output_dir / "mesh.glb"
    data.mesh.export(str(out_mesh_path))

    bw_path = output_dir / "bw.bin"
    joints_path = output_dir / "joints.bin"
    bw_np.astype(np.float32).tofile(str(bw_path))
    joints_head.astype(np.float32).tofile(str(joints_path))

    json_data = {
        "mesh_path":   str(out_mesh_path),
        "bw_path":     str(bw_path),
        "bw_shape":    list(bw_np.shape),
        "joints_path": str(joints_path),
        "joints_shape": list(joints_head.shape),
        "bones_idx_dict": dict(BONES_IDX_DICT),
        "pose_ignore_list": [],
        "no_fingers": no_fingers,
        "reset_to_rest": reset_to_rest,
    }
    if joints_tail is not None:
        joints_tail_path = output_dir / "joints_tail.bin"
        joints_tail.astype(np.float32).tofile(str(joints_tail_path))
        json_data["joints_tail_path"] = str(joints_tail_path)
        json_data["joints_tail_shape"] = list(joints_tail.shape)
    if pose_np is not None:
        pose_path = output_dir / "pose.bin"
        pose_np.astype(np.float32).tofile(str(pose_path))
        json_data["pose_path"] = str(pose_path)
        json_data["pose_shape"] = list(pose_np.shape)

    json_path = output_dir / "data.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"[runner] wrote {json_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--no_fingers", action="store_true")
    ap.add_argument("--use_normal", action="store_true")
    ap.add_argument("--reset_to_rest", action="store_true")
    args = ap.parse_args()
    run_inference(
        args.input, Path(args.output_dir),
        no_fingers=args.no_fingers,
        use_normal=args.use_normal,
        reset_to_rest=args.reset_to_rest,
    )


if __name__ == "__main__":
    main()
