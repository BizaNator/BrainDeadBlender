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


def _python_is_real(path: Path) -> bool:
    """Truly run the candidate python; return True only if it responds."""
    if not path.exists():
        return False
    try:
        r = subprocess.run([str(path), "--version"], capture_output=True,
                              timeout=8)
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def venv_python() -> Path:
    """Return the path to the venv's Python interpreter."""
    if platform.system() == "Windows":
        suffix = Path("Scripts") / "python.exe"
    else:
        suffix = Path("bin") / "python"
    return VENV_DIR / suffix


def is_bootstrapped() -> bool:
    """True if the venv exists and the sentinel marks first-run install
    complete."""
    return SENTINEL_FILE.exists() and _python_is_real(venv_python())


def actual_cache_root() -> Path:
    """The cache directory holding the venv + MIA src + models. With uv-
    managed venvs this is just CACHE_ROOT — no sandbox redirect to chase."""
    return CACHE_ROOT


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


def _find_uv() -> Optional[str]:
    """Locate the uv executable. Preferred over system python because uv
    manages its own standalone CPython (python-build-standalone), avoiding
    the Microsoft Store Python sandbox-redirect on Windows and any
    PATH-shim weirdness."""
    import shutil
    p = shutil.which("uv")
    if p:
        return p
    # Common Windows install locations
    if platform.system() == "Windows":
        for cand in (
            Path.home() / ".local" / "bin" / "uv.exe",
            Path(os.environ.get("LOCALAPPDATA",
                                  Path.home() / "AppData" / "Local"))
                / "Programs" / "uv" / "uv.exe",
        ):
            if cand.exists():
                return str(cand)
    return None


def bootstrap(progress_cb=None) -> bool:
    """Create the venv + install all deps + clone MIA source. Idempotent —
    skips steps that are already complete. Returns True on success.

    Prefers uv (https://docs.astral.sh/uv/) for venv + installs, falling
    back to system python3 if uv isn't available."""
    if progress_cb is None:
        progress_cb = lambda msg: print(f"[BD_AutoRig:bootstrap] {msg}",
                                          flush=True)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    MIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if is_bootstrapped():
        progress_cb(f"Already bootstrapped at {CACHE_ROOT} — skipping.")
        return True

    uv = _find_uv()
    if uv is None:
        progress_cb("ERROR: uv not found. Install uv from "
                     "https://docs.astral.sh/uv/getting-started/installation/")
        return False
    progress_cb(f"Using uv at {uv}")

    # ── 1) Create venv via uv (downloads standalone Python 3.12 if needed)
    if not _python_is_real(venv_python()):
        progress_cb(f"Creating venv at {VENV_DIR} via uv "
                     "(may download standalone Python first)…")
        rc = _run(
            [uv, "venv", str(VENV_DIR), "--python", "3.12"],
            progress_cb=progress_cb,
        )
        if rc != 0:
            progress_cb(f"ERROR: uv venv creation failed (rc={rc}).")
            return False

    # Helper to invoke `uv pip install` against the managed venv. uv reads
    # VIRTUAL_ENV to know which interpreter to target.
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV_DIR)

    def uv_pip_install(*args, extra_env=None) -> int:
        e = dict(env)
        if extra_env:
            e.update(extra_env)
        return _run([uv, "pip", "install", *args], progress_cb=progress_cb,
                      env=e)

    # ── 2) PyTorch (CUDA or CPU) ────────────────────────────────────────────
    cuda = detect_cuda_availability()
    if cuda:
        progress_cb("CUDA detected — installing torch with CUDA wheels")
        rc = uv_pip_install(
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu124",
        )
    else:
        progress_cb("No CUDA — installing CPU torch")
        rc = uv_pip_install(
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        )
    if rc != 0:
        progress_cb(f"ERROR: torch install rc={rc}")
        return False

    # ── 3) Other deps ───────────────────────────────────────────────────────
    deps = [
        "numpy<2.3",
        "trimesh>=4.0",
        "huggingface_hub>=0.20",
        "scipy",
        "scikit-learn",
        "einops",
        # MIA model imports
        "timm",
        "PyMCubes",
        "plyfile",
        "potpourri3d",
        "shapely",  # trimesh.slice_plane dep
    ]
    rc = uv_pip_install(*deps)
    if rc != 0:
        progress_cb(f"ERROR: deps install rc={rc}")
        return False

    # ── 3.5) torch_cluster (PyG dep — wheels matched to torch+CUDA) ─────────
    progress_cb("Installing torch_cluster…")
    rc = _run(
        [str(venv_python()), "-c",
         "import torch; v=torch.__version__.split('+')[0]; "
         "cu=(torch.version.cuda or 'cpu').replace('.', ''); "
         "tag=f'cu{cu}' if cu!='cpu' else 'cpu'; "
         "import subprocess,sys; "
         "subprocess.check_call(["
         f"'{uv.replace(chr(92), chr(92)*2)}','pip','install',"
         "'torch_cluster','-f',"
         "f'https://data.pyg.org/whl/torch-{v}+{tag}.html'])"],
        progress_cb=progress_cb,
        env=env,
    )
    if rc != 0:
        progress_cb(f"WARN: torch_cluster install rc={rc} — "
                     "MIA inference will fail without it")

    # ── 4) Sentinel ─────────────────────────────────────────────────────────
    # The vendored pruned MIA package ships next to this file (in
    # autorig_vendor/mia/), so we don't need an upstream clone for
    # inference. Just record what we installed.
    SENTINEL_FILE.write_text(
        f"bootstrapped\n"
        f"backend=uv\n"
        f"uv={uv}\n"
        f"cuda={cuda}\n"
        f"venv={VENV_DIR}\n"
        f"models={MIA_MODELS_DIR}\n"
    )
    progress_cb(f"Bootstrap complete. Sentinel at {SENTINEL_FILE}")
    return True




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
