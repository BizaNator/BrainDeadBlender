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


def _candidate_venv_dirs() -> list[Path]:
    """List of paths where the venv might actually live.

    Microsoft Store Python redirects %LOCALAPPDATA% writes into a
    per-package sandbox under ``Local\\Packages\\PythonSoftwareFoundation\
    .Python.<ver>\\LocalCache\\Local\\...``. So when we ask ``python -m
    venv C:\\Users\\X\\AppData\\Local\\BrainDeadBlender\\autorig\\venv``
    the venv ends up at
    ``...\\LocalCache\\Local\\BrainDeadBlender\\autorig\\venv`` and
    the requested path stays empty. We probe both.
    """
    paths = [VENV_DIR]
    if platform.system() == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA",
                                       Path.home() / "AppData" / "Local"))
        try:
            rel = VENV_DIR.relative_to(local)
        except ValueError:
            return paths
        pkg_root = local / "Packages"
        if pkg_root.exists():
            for pkg_dir in pkg_root.iterdir():
                if pkg_dir.name.startswith("PythonSoftwareFoundation.Python."):
                    redirected = (pkg_dir / "LocalCache" / "Local" / rel)
                    if redirected not in paths:
                        paths.append(redirected)
    return paths


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
    """Return the path to the venv's Python interpreter, following any
    Microsoft Store Python redirect that may have moved the venv into a
    sandboxed AppData\\Packages location. Probes by actually running each
    candidate's --version since path.exists() is unreliable under
    virtualization."""
    if platform.system() == "Windows":
        suffix = Path("Scripts") / "python.exe"
    else:
        suffix = Path("bin") / "python"
    candidates = [c / suffix for c in _candidate_venv_dirs()]
    # On Windows, prefer the Store-sandbox path when both report existing
    # (the original location is virtualized; only the sandbox path is the
    # real one). We do this by sorting the sandbox candidates first.
    if platform.system() == "Windows":
        candidates.sort(key=lambda p: 0 if "LocalCache\\Local" in str(p)
                                            or "LocalCache/Local" in str(p)
                                            else 1)
    for p in candidates:
        if _python_is_real(p):
            return p
    return VENV_DIR / suffix


def is_bootstrapped() -> bool:
    """True if the venv exists and the sentinel marks first-run install
    complete."""
    if not venv_python().exists():
        return False
    # Sentinel may have been written under the redirected sandbox path
    for cand in _candidate_venv_dirs():
        sentinel = cand.parent / "_bootstrapped.txt"
        if sentinel.exists():
            return True
    return SENTINEL_FILE.exists()


def actual_cache_root() -> Path:
    """The cache directory that actually holds the venv + MIA src + models,
    honoring Store-redirect."""
    py = venv_python()
    # If we picked a real python, its venv parent is the truth
    if _python_is_real(py):
        return py.parent.parent.parent
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
                     f"{actual_cache_root()} — skipping.")
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
        # MIA model imports
        "timm",
        "PyMCubes",
        "plyfile",
        "potpourri3d",
        "shapely",  # trimesh.slice_plane dep
    ]
    rc = _run(
        [str(venv_python()), "-m", "pip", "install", *deps],
        progress_cb=progress_cb,
    )
    if rc != 0:
        progress_cb(f"ERROR: deps install rc={rc}")
        return False

    # ── 4.5) torch_cluster (PyG dep MIA's model.py imports) ─────────────────
    # Wheels live at data.pyg.org/whl/torch-<ver>+<cu>.html. We derive the
    # tag from the installed torch.
    progress_cb("Installing torch_cluster…")
    rc = _run(
        [str(venv_python()), "-c",
         "import torch; v=torch.__version__.split('+')[0]; "
         "cu=(torch.version.cuda or 'cpu').replace('.', ''); "
         "tag=f'cu{cu}' if cu!='cpu' else 'cpu'; "
         "import subprocess,sys; "
         "subprocess.check_call([sys.executable,'-m','pip','install',"
         "'torch_cluster','-f',"
         "f'https://data.pyg.org/whl/torch-{v}+{tag}.html'])"],
        progress_cb=progress_cb,
    )
    if rc != 0:
        progress_cb(f"WARN: torch_cluster install rc={rc} — "
                     "MIA inference will fail without it; please install "
                     "manually from data.pyg.org/whl/")

    # ── 5) Clone MIA upstream (shallow) ─────────────────────────────────────
    #
    # Place the clone next to the venv that actually exists (Store-redirect
    # may have moved the cache into a sandboxed path).
    real_cache = actual_cache_root()
    real_mia_src = real_cache / "Make-It-Animatable"
    real_mia_models = real_cache / "models" / "mia"
    real_mia_models.mkdir(parents=True, exist_ok=True)

    if not real_mia_src.exists():
        progress_cb(f"Cloning Make-It-Animatable to {real_mia_src}…")
        rc = _run(
            ["git", "clone", "--depth", "1",
             "https://github.com/jasongzy/Make-It-Animatable.git",
             str(real_mia_src)],
            progress_cb=progress_cb,
        )
        if rc != 0:
            progress_cb(f"ERROR: clone rc={rc} — is git installed + on PATH?")
            return False
    else:
        progress_cb(f"MIA source already at {real_mia_src}")

    # ── 6) Sentinel ─────────────────────────────────────────────────────────
    sentinel_path = real_cache / "_bootstrapped.txt"
    sentinel_path.write_text(
        f"bootstrapped\n"
        f"cuda={cuda}\n"
        f"mia_src={real_mia_src}\n"
        f"venv={venv_python().parent.parent}\n"
        f"models={real_mia_models}\n"
    )
    progress_cb(f"Bootstrap complete. Sentinel at {sentinel_path}")
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

    real_models = actual_cache_root() / "models" / "mia"
    real_models.mkdir(parents=True, exist_ok=True)
    missing = [
        f for f in MIA_MODEL_FILES
        if not (real_models / f).exists()
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
        local_dir={str(real_models)!r},
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
