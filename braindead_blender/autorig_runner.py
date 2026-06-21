"""
BrainDeadBlender — AutoRig Inference Runner

Runs inside the bootstrapped venv (NOT inside Blender). Loads the MIA
model, runs inference on an input mesh, dumps result data (joints,
weights, mesh) as JSON + raw binary to a temp dir for the in-Blender
FBX assembler to pick up.

Invocation:
    <venv>/python -m braindead_blender.autorig_runner \\
        --input <mesh.glb> --output_dir <tmp_dir> \\
        [--no_fingers] [--use_normal] [--reset_to_rest]

Outputs in <output_dir>:
    data.json     joints, bones_idx_dict, options
    bw.bin        skinning weights (float32, shape = [V, num_bones])
    joints.bin    bone head positions (float32, shape = [num_bones, 3])
    joints_tail.bin   bone tail positions (optional, same shape as joints)
    mesh.glb      possibly-resampled mesh (trimesh)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Bootstrap-derived paths discoverable from CACHE_ROOT env var that
# autorig_local.py sets when launching this script.
_CACHE_ROOT = Path(os.environ.get("BD_AUTORIG_CACHE",
                                      Path.home() / ".cache" /
                                      "braindead_blender" / "autorig"))
MIA_SRC_DIR = _CACHE_ROOT / "Make-It-Animatable"
MIA_MODELS_DIR = _CACHE_ROOT / "models" / "mia"

# Make MIA source importable
if str(MIA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MIA_SRC_DIR))


def _mock_bpy():
    """MIA's modules transitively import bpy (their training-time FBX
    helpers). The inference path doesn't need bpy. Mock before imports."""
    from unittest.mock import MagicMock
    if "bpy" not in sys.modules:
        sys.modules["bpy"] = MagicMock()


def _resolve_models() -> Path:
    """Return the directory containing the .pth weight files."""
    new_dir = MIA_MODELS_DIR / "output" / "best" / "new"
    if new_dir.exists() and any(new_dir.glob("*.pth")):
        return new_dir
    raise RuntimeError(
        f"MIA model weights not found under {MIA_MODELS_DIR}. "
        "Bootstrap the autorig (BD AutoRig panel → Install Local Autorig).")


def load_mia_models(cache_to_gpu: bool = True):
    """Load MIA models from disk. Returns dict suitable for run_mia_inference."""
    _mock_bpy()
    import torch
    # MIA modules
    from model import PCAE  # type: ignore
    from dataset_mixamo import JOINTS_NUM, KINEMATIC_TREE  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() and cache_to_gpu
                          else "cpu")

    models_dir = _resolve_models()

    def _load(name: str, **kwargs):
        path = models_dir / name
        print(f"  loading {path}", flush=True)
        model = PCAE(**kwargs).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    N = 32768  # samples per mesh — MIA default

    # Mirror PozzettiAndrea's model loading shape. We use simplest defaults.
    return {
        "device": device,
        "N": N,
        "hands_resample_ratio": 0.0,
        "geo_resample_ratio": 0.0,
        "model_bw":        _load("bw.pth", output_dim=JOINTS_NUM, kinematic_tree=KINEMATIC_TREE),
        "model_bw_normal": _load("bw_normal.pth", output_dim=JOINTS_NUM, use_normal=True, kinematic_tree=KINEMATIC_TREE),
        "model_joints":    _load("joints.pth", output_dim=JOINTS_NUM * 6, kinematic_tree=KINEMATIC_TREE),
        "model_joints_coarse": _load("joints_coarse.pth", output_dim=JOINTS_NUM * 6, kinematic_tree=KINEMATIC_TREE),
        "model_pose":      _load("pose.pth", output_dim=JOINTS_NUM * 6, kinematic_tree=KINEMATIC_TREE),
        "model_coarse":    None,  # filled in below
    }


def run_inference(mesh_path: str, output_dir: Path, *,
                    no_fingers: bool, use_normal: bool,
                    reset_to_rest: bool):
    """End-to-end inference: load mesh → MIA pipeline → write outputs."""
    _mock_bpy()
    import numpy as np
    import torch
    import trimesh

    print(f"[runner] loading mesh: {mesh_path}", flush=True)
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Could not load mesh as Trimesh: {mesh_path}")

    print(f"[runner] loading models…", flush=True)
    models = load_mia_models()
    models["model_coarse"] = models["model_joints_coarse"]  # alias for pipeline

    from pipeline import prepare_input, preprocess, infer, bw_post_process  # type: ignore
    from dataset_mixamo import BONES_IDX_DICT  # type: ignore

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
    head_y = joints[..., head_idx, 4]
    above_head_mask = data.verts[..., 1] >= head_y
    bw = bw_post_process(
        data.bw,
        bones_idx_dict=BONES_IDX_DICT,
        above_head_mask=above_head_mask,
        no_fingers=no_fingers,
    )

    # Write outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joints_np = data.joints.squeeze(0).cpu().numpy() if hasattr(data.joints, "cpu") else np.asarray(data.joints).squeeze(0)
    bw_np = bw.squeeze(0).cpu().numpy() if hasattr(bw, "cpu") else np.asarray(bw).squeeze(0)
    pose_np = None
    if reset_to_rest and getattr(data, "pose", None) is not None:
        pose_np = data.pose.squeeze(0).cpu().numpy() if hasattr(data.pose, "cpu") else np.asarray(data.pose).squeeze(0)

    # Joints array is shape [num_bones, 6] where [:3] = head, [3:] = tail
    joints_head = joints_np[..., :3]
    joints_tail = joints_np[..., 3:] if joints_np.shape[-1] >= 6 else None

    joints_head.astype(np.float32).tofile(str(output_dir / "joints.bin"))
    if joints_tail is not None:
        joints_tail.astype(np.float32).tofile(str(output_dir / "joints_tail.bin"))
    bw_np.astype(np.float32).tofile(str(output_dir / "bw.bin"))
    if pose_np is not None:
        pose_np.astype(np.float32).tofile(str(output_dir / "pose.bin"))

    # Final mesh (might be the input or a resampled version)
    out_mesh_path = output_dir / "mesh.glb"
    data.mesh.export(str(out_mesh_path))

    meta = {
        "bones_idx_dict": dict(BONES_IDX_DICT),
        "num_bones": int(joints_np.shape[0]),
        "num_verts": int(bw_np.shape[0]),
        "joints_shape": list(joints_np.shape),
        "bw_shape": list(bw_np.shape),
        "has_tail": joints_tail is not None,
        "has_pose": pose_np is not None,
        "no_fingers": no_fingers,
        "use_normal": use_normal,
        "reset_to_rest": reset_to_rest,
    }
    (output_dir / "data.json").write_text(json.dumps(meta, indent=2))
    print(f"[runner] wrote outputs to {output_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input mesh path")
    ap.add_argument("--output_dir", required=True, help="Where to write artifacts")
    ap.add_argument("--no_fingers", action="store_true")
    ap.add_argument("--use_normal", action="store_true")
    ap.add_argument("--reset_to_rest", action="store_true")
    args = ap.parse_args()
    run_inference(
        args.input, args.output_dir,
        no_fingers=args.no_fingers,
        use_normal=args.use_normal,
        reset_to_rest=args.reset_to_rest,
    )


if __name__ == "__main__":
    main()
