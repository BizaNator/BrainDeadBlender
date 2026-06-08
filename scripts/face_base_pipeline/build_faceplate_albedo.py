"""
Build the channel-packed face plate overlay texture.

Channel layout:
    R = front greyscale   (albedo_light or mannequin_light, zone 0)
    G = side greyscale    (albedo_light or mannequin_light, zone 1)
    B = front whitelines  (anti-aliased highlight/line mask, front angle only)
    A = 0                 (reserved — roughness variation or ILM shadow later)

Neither zone needs the source alpha packed: non-face pixels are [0,0,0,0] in the
source, so they load as grey=0 (black). The head mesh geometry defines the face
boundary in the Unreal material — no per-pixel alpha blend needed.

Material blend (Unreal M_FacePlate_Head):
    albedo_grey = Lerp(R, G, FacePlate_Zone.B)        -- zone 0 = front, zone 1 = side
    base_color  = SkinTint × albedo_grey               -- tinted skin detail
    final       = Lerp(base_color, White, B × 0.4)    -- whiteline highlight overlay

Usage:
    # Standard: pack albedo_light front+side + whitelines
    python build_faceplate_albedo.py ernest_chavez v3SR

    # For MediaPipe-calibrated UV: pack the photorealistic headshots instead
    python build_faceplate_albedo.py ernest_chavez v3SR --front-path "B:/..." --side-path "B:/..."

    python build_faceplate_albedo.py ernest_chavez v3SR --size 4096
    python build_faceplate_albedo.py ernest_chavez v3SR --load-blender

Output:
    B:/Brains/Characters/<char>/images/faceplate/<char>_<ver>_faceplate_<type>.png
"""

import argparse, os
from PIL import Image
import numpy as np

NAS_BASE           = r"B:\Brains\Characters"
ATLAS_SIZE_DEFAULT = 2048


def load_grey(path: str, size: int) -> np.ndarray:
    """
    Load image, resize to size×size.
    Returns BT.709 luminance float32 (size, size) in 0-1.
    Non-face pixels (transparent background) map to grey=0 naturally.
    """
    img = Image.open(path).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    px  = np.array(img, dtype=np.float32) / 255.0
    return 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]


def build_faceplate_albedo(
    char:           str,
    ver:            str,
    img_type:       str  = "albedo_light",
    size:           int  = ATLAS_SIZE_DEFAULT,
    nas_base:       str  = NAS_BASE,
    out_dir:        str  = None,
    front_path:        str  = None,   # explicit override; skips NAS path construction
    side_path:         str  = None,   # explicit override
    whitelines_path:   str  = None,   # explicit override for the whitelines source
    no_whitelines:     bool = False,  # force B channel = 0 (whitelines reserved)
    whitelines_mode:   str  = "separate",   # "separate" (recommended): write WL to its own BC4-friendly file,
                                            # atlas B channel = 0. "packed" (legacy): WL into atlas B channel.
    wl_size:           int  = 1024,          # whitelines texture resolution when whitelines_mode="separate"
    mp_json_path:      str  = None,          # optional: MediaPipe JSON to sanity-check before building
) -> dict:
    """
    Build the channel-packed faceplate atlas. Returns dict with output paths.

    Atlas channel layout (front-only whitelines per studio rule — side WL not generated):
        R = front greyscale
        G = side greyscale
        B = front whitelines  (only when whitelines_mode="packed")
            0                  (when whitelines_mode="separate" — WL is in its own texture)
        A = 1.0  (unused pad — note UE FBX V-flip flips this on import)

    front_path / side_path: if provided, use these directly instead of the NAS
    convention. Required when using the photorealistic headshots as UV sources
    for MediaPipe-calibrated UV (those images live outside the 2d_front / 2d dirs).

    Whitelines are FRONT-only by studio rule (B:/Brains/Skills/Rules/NamingConventions.md
    / project_faceplate_uv_projection memory): the WL pass is hand-painted anime-style
    edge emulation on the front face. A side-WL generation would not match anatomically
    and looks wrong. So this script only consumes a front whiteline pass; if you set
    whitelines_mode="separate", it writes a single-channel front-only texture.

    Sanity check (optional, when mp_json_path is provided):
      Detects the known ComfyUI MP-export anomaly where the lips bbox accidentally
      copies left_iris coords (lip cy ends up at eye altitude). Raises if detected;
      pass strict_sanity=False to downgrade to a warning.
    """
    if front_path is None:
        front_path = os.path.join(nas_base, char, "images", "2d_front",
                                  f"{char}_head_{ver}_{img_type}_1.png")
    if side_path is None:
        side_path  = os.path.join(nas_base, char, "images", "2d",
                                  f"{char}_head_{ver}_{img_type}_1.png")
    if whitelines_path is not None:
        wl_path = whitelines_path
    else:
        wl_path = os.path.join(nas_base, char, "images", "2d_front",
                               f"{char}_head_{ver}_whitelines_1.png")

    for p in (front_path, side_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing source image: {p}")

    has_whitelines = (not no_whitelines) and os.path.exists(wl_path)
    if no_whitelines:
        print(f"Whitelines : disabled (B=0)")
    elif not has_whitelines:
        print(f"Note: no whitelines at {wl_path}")

    # Sanity check MediaPipe data for the lips=eyes coord anomaly
    if mp_json_path and os.path.exists(mp_json_path):
        _sanity_check_mp_landmarks(mp_json_path)

    print(f"Front           : {front_path}")
    print(f"Side            : {side_path}")
    if not no_whitelines:
        print(f"Whitelines      : {wl_path if has_whitelines else 'N/A'}")
    print(f"Atlas size      : {size}×{size}")
    print(f"Whitelines mode : {whitelines_mode}")

    front_grey = load_grey(front_path, size)
    side_grey  = load_grey(side_path,  size)
    wl_grey    = (load_grey(wl_path, size) if has_whitelines
                  else np.zeros((size, size), dtype=np.float32))

    packed = np.zeros((size, size, 4), dtype=np.uint8)
    packed[..., 0] = (front_grey * 255).clip(0, 255).astype(np.uint8)  # R = front grey
    packed[..., 1] = (side_grey  * 255).clip(0, 255).astype(np.uint8)  # G = side grey
    if whitelines_mode == "packed":
        packed[..., 2] = (wl_grey * 255).clip(0, 255).astype(np.uint8) # B = WL packed (legacy)
    else:
        packed[..., 2] = 0                                             # B = 0 (WL is separate)
    packed[..., 3] = 255                                               # A = 1.0 pad

    # Skin-tone patch at UV (0.99, 0.99) = top-right of image array.
    sky = int(0.35 * size); skx = int(0.50 * size)
    swatch_r = int(front_grey[sky-8:sky+8, skx-8:skx+8].mean() * 255)
    swatch_g = int(side_grey [int(0.45*size)-8:int(0.45*size)+8,
                              int(0.50*size)-8:int(0.50*size)+8].mean() * 255)
    patch_size = max(8, int(size * 0.05))
    packed[0:patch_size, size - patch_size:size, 0] = swatch_r
    packed[0:patch_size, size - patch_size:size, 1] = swatch_g
    packed[0:patch_size, size - patch_size:size, 2] = 0
    packed[0:patch_size, size - patch_size:size, 3] = 255
    print(f"Skin corner (UV ~0.99,0.99): swatch R={swatch_r} G={swatch_g}, "
          f"patch {patch_size}px")

    if out_dir is None:
        out_dir = os.path.join(nas_base, char, "images", "faceplate")
    os.makedirs(out_dir, exist_ok=True)

    # New: studio naming convention — T_<char>_face_atlas.png
    atlas_path = os.path.join(out_dir, f"T_{char}_face_atlas.png")
    Image.fromarray(packed, mode="RGBA").save(atlas_path)
    print(f"Saved atlas     : {atlas_path}")

    result = {"atlas": atlas_path}

    # If "separate" mode, also write a standalone WL texture at wl_size
    if whitelines_mode == "separate" and has_whitelines:
        wl_grey_lores = load_grey(wl_path, wl_size)
        # Single-channel L8 PNG (BC4 import target in UE)
        wl_packed = (wl_grey_lores * 255).clip(0, 255).astype(np.uint8)
        wl_out = os.path.join(out_dir, f"T_{char}_face_wl.png")
        Image.fromarray(wl_packed, mode="L").save(wl_out)
        print(f"Saved WL (sep)  : {wl_out}  [{wl_size}x{wl_size}, single channel]")
        result["whitelines"] = wl_out

    return result


def _sanity_check_mp_landmarks(json_path: str, strict: bool = True):
    """
    Check a MediaPipe Face Mesh JSON export for known data anomalies.

    Currently checks: does the lips bbox have the same coords as left_iris?
    (A known ComfyUI workflow bug where the lips landmark accidentally inherits
    eye coordinates. Lip cy should be substantially below eye cy.)

    Raises RuntimeError if the anomaly is detected and strict=True.
    """
    import json
    with open(json_path) as f:
        data = json.load(f)

    # Look for landmark sub-objects with the names we care about
    lips_bbox     = None
    left_iris_bbox = None
    for k, v in data.items():
        if not isinstance(v, dict): continue
        if k.lower() == "lips":      lips_bbox = v
        if k.lower() == "left_iris": left_iris_bbox = v

    if lips_bbox is None or left_iris_bbox is None:
        return   # nothing to check

    # If the lips have a cy and left_iris has a cy, lip cy should be larger
    # (lower on the face = larger y in image coords, since y=0 is top).
    lip_cy = lips_bbox.get("cy") or lips_bbox.get("center_y")
    eye_cy = left_iris_bbox.get("cy") or left_iris_bbox.get("center_y")

    if lip_cy is None or eye_cy is None:
        return

    if abs(lip_cy - eye_cy) < 0.05:
        msg = (f"MediaPipe anomaly: lips cy ({lip_cy:.4f}) is too close to "
               f"left_iris cy ({eye_cy:.4f}). Expected lips to be well below eyes "
               f"(diff >= 0.05). This is the known ComfyUI workflow bug where lips "
               f"bbox coords accidentally clone left_iris. Re-run the MediaPipe "
               f"export or hand-edit the JSON.")
        if strict:
            raise RuntimeError(msg)
        else:
            print(f"[warn] {msg}")


def load_into_blender(path: str, name: str = None):
    """Load the packed atlas into the active Blender session and set it in the UV editor."""
    import bpy
    img_name = name or os.path.basename(path).replace(".png", "")
    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])
    img = bpy.data.images.load(path)
    img.name = img_name
    img.colorspace_settings.name = "Non-Color"
    for area in bpy.context.screen.areas:
        if area.type == "IMAGE_EDITOR":
            area.spaces.active.image = img
            for region in area.regions:
                if region.type == "WINDOW":
                    with bpy.context.temp_override(area=area, region=region):
                        bpy.ops.image.view_all(fit_view=True)
                    break
            break
    return img_name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("char",  help="Character slug, e.g. ernest_chavez")
    ap.add_argument("ver",   help="Version, e.g. v3SR")
    ap.add_argument("--type",         default="albedo_light", dest="img_type",
                    help="Image type suffix (default: albedo_light)")
    ap.add_argument("--size",         type=int, default=ATLAS_SIZE_DEFAULT,
                    help=f"Output resolution (default: {ATLAS_SIZE_DEFAULT})")
    ap.add_argument("--out-dir",      default=None)
    ap.add_argument("--front-path",   default=None,
                    help="Explicit front image path (overrides NAS convention)")
    ap.add_argument("--side-path",    default=None,
                    help="Explicit side image path (overrides NAS convention)")
    ap.add_argument("--whitelines-path", default=None,
                    help="Explicit whitelines image path (overrides NAS convention)")
    ap.add_argument("--no-whitelines", action="store_true",
                    help="Disable whitelines entirely (B=0, no separate WL texture written)")
    ap.add_argument("--whitelines-mode", default="separate",
                    choices=("separate", "packed"),
                    help="separate (default): write WL to its own BC4-friendly texture, "
                         "atlas B channel = 0. packed: pack WL into atlas B channel (legacy).")
    ap.add_argument("--wl-size",      type=int, default=1024,
                    help="Resolution for the separate whitelines texture (default: 1024).")
    ap.add_argument("--mp-json",      default=None,
                    help="Optional MediaPipe JSON to sanity-check for the lips=eyes anomaly.")
    ap.add_argument("--load-blender", action="store_true",
                    help="Load result into active Blender session after building")
    args = ap.parse_args()

    result = build_faceplate_albedo(
        char             = args.char,
        ver              = args.ver,
        img_type         = args.img_type,
        size             = args.size,
        out_dir          = args.out_dir,
        front_path       = args.front_path,
        side_path        = args.side_path,
        whitelines_path  = args.whitelines_path,
        no_whitelines    = args.no_whitelines,
        whitelines_mode  = args.whitelines_mode,
        wl_size          = args.wl_size,
        mp_json_path     = args.mp_json,
    )

    if args.load_blender:
        loaded = load_into_blender(result["atlas"])
        print(f"Loaded into Blender as: {loaded}")
