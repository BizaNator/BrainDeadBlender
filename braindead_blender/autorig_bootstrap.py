"""
BrainDeadBlender — AutoRig Local Bootstrap

Creates a managed Python venv at ~/.cache/braindead_blender/autorig/venv/
on first use. Installs:
  - PyTorch (CUDA if available, else CPU)
  - numpy, trimesh, huggingface_hub
  - Make-It-Animatable source (cloned shallow from upstream)
  - MIA pretrained weights (auto-downloaded from HuggingFace at runtime)

After bootstrap, BDB calls subprocess(<venv>/python autorig_runner.py …)
to run inference without polluting Blender's bundled Python. The
subprocess writes joints/weights/mesh as JSON+raw-binary to a temp dir;
BDB then builds the armature + skinned mesh in-process using bpy.

No external Blender add-on dependency, no ComfyUI dependency.
"""

import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional


# ── Cache layout ────────────────────────────────────────────────────────────

def _cache_root() -> Path:
    """Return the BDB autorig cache root, OS-appropriate."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA",
                                       Path.home() / "AppData" / "Local"))
        return base / "BrainDeadBlender" / "autorig"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "BrainDeadBlender" / "autorig"
    return Path.home() / ".cache" / "braindead_blender" / "autorig"


CACHE_ROOT = _cache_root()
VENV_DIR = CACHE_ROOT / "venv"
MIA_SRC_DIR = CACHE_ROOT / "Make-It-Animatable"
MIA_MODELS_DIR = CACHE_ROOT / "models" / "mia"
SENTINEL_FILE = CACHE_ROOT / "_bootstrapped.txt"


def venv_python() -> Path:
    """Return the path to the venv's Python interpreter."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def is_bootstrapped() -> bool:
    """True if the venv exists and the sentinel marks first-run install
    complete."""
    return SENTINEL_FILE.exists() and venv_python().exists()


# ── Bootstrap implementation ────────────────────────────────────────────────

def _run(cmd, *, cwd: Optional[Path] = None, env=None,
          progress_cb=None) -> int:
    """Run a subprocess streaming stdout/stderr through progress_cb (which
    receives one line at a time)."""
    if progress_cb is None:
        progress_cb = lambda line: print(line, flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(cwd) if cwd else None,
        env=env,
    )
    for line in proc.stdout:
        progress_cb(line.rstrip())
    proc.wait()
    return proc.returncode


def detect_cuda_availability() -> Optional[str]:
    """Return the CUDA major.minor string available locally, or None."""
    # Try nvidia-smi first
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        if out.strip():
            return "auto"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def bootstrap(progress_cb=None) -> bool:
    """Create the venv + install all deps + clone MIA source. Idempotent —
    skips steps that are already complete. Returns True on success.

    progress_cb: callable receiving status lines, for the Blender UI to
    surface to the user."""
    if progress_cb is None:
        progress_cb = lambda msg: print(f"[BD_AutoRig:bootstrap] {msg}",
                                          flush=True)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    MIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if is_bootstrapped():
        progress_cb("Already bootstrapped at "
                     f"{CACHE_ROOT} — skipping.")
        return True

    # ── 1) Create venv ──────────────────────────────────────────────────────
    if not venv_python().exists():
        progress_cb(f"Creating venv at {VENV_DIR}…")
        # Use the system python3 (not Blender's bundled one) since Blender's
        # bundled Python is missing pip/venv on some platforms.
        sys_python = _find_system_python()
        if sys_python is None:
            progress_cb("ERROR: no system python3 found. Install Python 3.10+ "
                          "and ensure 'python' is on PATH.")
            return False
        rc = _run(
            [sys_python, "-m", "venv", str(VENV_DIR)],
            progress_cb=progress_cb,
        )
        if rc != 0:
            progress_cb(f"ERROR: venv creation failed (rc={rc}).")
            return False

    # ── 2) Upgrade pip + base wheels ────────────────────────────────────────
    pip_cmd = [str(venv_python()), "-m", "pip", "install", "--upgrade",
               "pip", "wheel", "setuptools"]
    rc = _run(pip_cmd, progress_cb=progress_cb)
    if rc != 0:
        progress_cb(f"WARN: pip upgrade rc={rc} — continuing")

    # ── 3) PyTorch (CUDA or CPU) ────────────────────────────────────────────
    cuda = detect_cuda_availability()
    if cuda:
        progress_cb("CUDA detected — installing torch with CUDA wheels")
        torch_cmd = [
            str(venv_python()), "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu124",
        ]
    else:
        progress_cb("No CUDA — installing CPU torch")
        torch_cmd = [
            str(venv_python()), "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ]
    rc = _run(torch_cmd, progress_cb=progress_cb)
    if rc != 0:
        progress_cb(f"ERROR: torch install rc={rc}")
        return False

    # ── 4) Other deps ───────────────────────────────────────────────────────
    deps = [
        "numpy<2.3",       # MIA was authored against numpy 1.x; 2.x mostly ok but pin loosely
        "trimesh>=4.0",
        "huggingface_hub>=0.20",
        "scipy",
        "scikit-learn",
        "einops",
    ]
    rc = _run(
        [str(venv_python()), "-m", "pip", "install", *deps],
        progress_cb=progress_cb,
    )
    if rc != 0:
        progress_cb(f"ERROR: deps install rc={rc}")
        return False

    # ── 5) Clone MIA upstream (shallow) ─────────────────────────────────────
    if not MIA_SRC_DIR.exists():
        progress_cb(f"Cloning Make-It-Animatable to {MIA_SRC_DIR}…")
        rc = _run(
            ["git", "clone", "--depth", "1",
             "https://github.com/jasongzy/Make-It-Animatable.git",
             str(MIA_SRC_DIR)],
            progress_cb=progress_cb,
        )
        if rc != 0:
            progress_cb(f"ERROR: clone rc={rc} — is git installed + on PATH?")
            return False
    else:
        progress_cb(f"MIA source already at {MIA_SRC_DIR}")

    # ── 6) Sentinel ─────────────────────────────────────────────────────────
    SENTINEL_FILE.write_text(
        f"bootstrapped\n"
        f"cuda={cuda}\n"
        f"mia_src={MIA_SRC_DIR}\n"
        f"venv={VENV_DIR}\n"
        f"models={MIA_MODELS_DIR}\n"
    )
    progress_cb(f"Bootstrap complete. Sentinel at {SENTINEL_FILE}")
    return True


def _find_system_python() -> Optional[str]:
    """Find a system Python 3 outside of Blender's bundle."""
    import shutil
    for cand in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        p = shutil.which(cand)
        if p and "blender" not in p.lower():
            return p
    return None


# ── Model weight download (idempotent) ──────────────────────────────────────

MIA_HF_REPO = "jasongzy/Make-It-Animatable"
MIA_MODEL_FILES = [
    "output/best/new/bw.pth",
    "output/best/new/bw_normal.pth",
    "output/best/new/joints.pth",
    "output/best/new/joints_coarse.pth",
    "output/best/new/pose.pth",
]


def ensure_mia_weights(progress_cb=None) -> bool:
    """Download MIA pretrained weights if missing. Uses the venv's
    huggingface_hub (must already be installed)."""
    if progress_cb is None:
        progress_cb = lambda msg: print(f"[BD_AutoRig:weights] {msg}", flush=True)

    MIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        f for f in MIA_MODEL_FILES
        if not (MIA_MODELS_DIR / f).exists()
    ]
    if not missing:
        return True

    progress_cb(f"Downloading {len(missing)} weight file(s) from {MIA_HF_REPO}…")
    # Run via the venv so huggingface_hub is available
    cmd = [
        str(venv_python()), "-c",
        f"""
import sys
from huggingface_hub import hf_hub_download
files = {missing!r}
for f in files:
    print(f'  - {{f}}', flush=True)
    hf_hub_download(
        repo_id={MIA_HF_REPO!r},
        filename=f,
        local_dir={str(MIA_MODELS_DIR)!r},
    )
print('weights ready', flush=True)
""",
    ]
    rc = _run(cmd, progress_cb=progress_cb)
    if rc != 0:
        progress_cb(f"ERROR: HF download rc={rc}")
        return False
    return True


# ── CLI for manual testing ──────────────────────────────────────────────────

if __name__ == "__main__":
    ok = bootstrap()
    if ok:
        ensure_mia_weights()
        print(f"venv python: {venv_python()}")
        print(f"MIA src:     {MIA_SRC_DIR}")
        print(f"MIA models:  {MIA_MODELS_DIR}")
    else:
        sys.exit(1)
