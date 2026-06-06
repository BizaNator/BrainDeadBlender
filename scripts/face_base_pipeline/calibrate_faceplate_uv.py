"""
Calibrate FacePlate UV using MediaPipe Face Mesh landmark correspondences.

Replaces the pure-geometry camera projection with anatomy-aware UV coordinates.

Algorithm (per camera angle):
  1. Run MediaPipe FaceMesh on the source photo → up to 468 landmarks at (x,y) in
     normalized [0,1] image space.  These positions ARE the target UV coordinates.
  2. For each landmark: ray-cast through the 3D mesh using the orthographic camera
     model (same FILL / bb_ctr / scale as camera_projection_uv.py) to find which
     3D point on the mesh that landmark sees.
  3. Build control points: { 3D hit position → (u, v) in source photo }.
  4. Thin-plate-spline RBF interpolation over all mesh vertex positions → UV.
  5. Vertices with no valid hit (back of head, no landmark coverage) → back zone.

Zone assignment:
  Normal-based (same as camera_projection_uv.py) — determines whether a face reads
  R channel (front) or G channel (side).  The RBF calibrates WHERE in that channel
  the face maps, not which channel it uses.

IMPORTANT — image requirements:
  MediaPipe requires photorealistic face images for detection.  Stylized renders
  (albedo, mannequin_light) will NOT detect.  Use the original AI-generated
  character headshots.  The UV coordinates produced are positions in THOSE image
  spaces, so pack those same images into the R / G atlas channels via
  build_faceplate_albedo.py (pass the same paths as --front / --side).

Usage (run inside Blender via MCP or Text Editor):
    import calibrate_faceplate_uv, importlib
    importlib.reload(calibrate_faceplate_uv)
    result = calibrate_faceplate_uv.calibrate(
        "ernest_step1",
        front_img = r"B:\Brains\Characters\ernest_chavez\images\ernest_chavez_head_front_00001_.png",
        side_img  = r"B:\Brains\Characters\ernest_chavez\images\2d_3q\ernest_chavez_head_v1SR_sr_light_3qleft_colorfix.png",
    )
    print(result)

Standalone (run from Windows to test MediaPipe on a single image):
    python calibrate_faceplate_uv.py --image path/to/photo.png
"""

import math, os, sys
import numpy as np

# ── camera / projection constants (must match camera_projection_uv.py) ────────
FILL           = 0.80
SIDE_ANGLE_DEG = 55
BACK_TOL       = 0.10
FRONT_BIAS     = 0.12

_a  = math.radians(SIDE_ANGLE_DEG)
_sa = math.sin(_a)
_ca = math.cos(_a)


# ── MediaPipe helpers ─────────────────────────────────────────────────────────

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".models", "face_landmarker.task"
)


def _run_mediapipe(img_path: str, model_path: str = _MODEL_PATH) -> list:
    """
    Run MediaPipe FaceLandmarker (task API, 0.10+) on img_path.
    Returns list of (x, y) tuples in [0,1] image-normalised coords,
    or [] if no face detected.
    x=0 left, x=1 right, y=0 top, y=1 bottom.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.FaceLandmarkerOptions(
        base_options                    = base_opts,
        running_mode                    = vision.RunningMode.IMAGE,
        num_faces                       = 1,
        min_face_detection_confidence   = 0.3,
        min_face_presence_confidence    = 0.3,
        output_face_blendshapes         = False,
        output_facial_transformation_matrixes = False,
    )

    mp_img = mp.Image.create_from_file(img_path)

    with vision.FaceLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return []

    return [(lm.x, lm.y) for lm in result.face_landmarks[0]]


# ── ray-cast helpers ──────────────────────────────────────────────────────────

def _front_ray_from_photo_uv(u, v, scale, bb_ctr):
    """
    Convert a (u,v) position in the front source photo to a world-space ray.
    Front camera: looks along +Y; right axis = X; up axis = Z.
    u=0 left, u=1 right; v=0 bottom (Blender UV), v=1 top.
    The photo has v=0 at top in image space → we flip when calling:
      pass v = 1 - landmark.y
    """
    from mathutils import Vector
    wx = (u - 0.5) * scale + bb_ctr.x
    wz = (v - 0.5) * scale + bb_ctr.z
    origin = Vector((wx, bb_ctr.y - 10.0, wz))
    direction = Vector((0.0, 1.0, 0.0))
    return origin, direction


def _side_ray_from_photo_uv(u, v, scale, bb_ctr):
    """Default side ray using module-level SIDE_ANGLE_DEG."""
    return _make_side_ray_fn(SIDE_ANGLE_DEG)(u, v, scale, bb_ctr)


def _make_side_ray_fn(angle_deg: float):
    """
    Return a side-view ray function for a specific camera angle.
    angle_deg: degrees from front (0=front-facing, 90=pure side profile).
    The side image is assumed to have the nose on the RIGHT side of the frame.
    """
    import math as _math
    sa = _math.sin(_math.radians(angle_deg))
    ca = _math.cos(_math.radians(angle_deg))

    def _ray_fn(u, v, scale, bb_ctr):
        from mathutils import Vector
        # Convention: face at -Y (toward camera in Blender FRONT view).
        # Camera at subject's LEFT-FRONT (-X, -Y), looking toward subject's
        # right-back (+X, +Y).  Image right axis = (+ca, -sa, 0) so the visible
        # ear (-X) maps to image LEFT and the nose (-Y) maps to image RIGHT.
        right    = Vector((ca, -sa, 0.0))
        up       = Vector((0.0, 0.0, 1.0))
        cam_look = Vector((sa, ca, 0.0))    # toward head from left-front

        plane_pt = (
            Vector(bb_ctr)
            + right * ((u - 0.5) * scale)
            + up    * ((v - 0.5) * scale)
        )
        origin    = plane_pt - cam_look * 10.0
        direction = cam_look
        return origin, direction

    return _ray_fn


def _detect_image_head_bbox(img_path: str, bg_threshold: float = 0.05) -> tuple:
    """
    Detect the head silhouette bbox in a source image (non-background pixels).
    Returns (x_min, x_max, y_min, y_max) in normalized [0,1] image space.
    y=0 is top (image convention).
    """
    from PIL import Image
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    H, W = img.shape[:2]
    # Luminance — non-background = head
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    mask = lum > bg_threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError(f"No non-background pixels in {img_path}")
    return (xs.min() / W, xs.max() / W,
            ys.min() / H, ys.max() / H)


def _front_face_mesh_bbox(me, mw, mw3, normal_threshold: float = 0.5,
                          trim_neck: bool = True):
    """
    Bounding box of front-facing mesh polys in world XZ (Y is depth).

    Used to derive the projection scale from data rather than a hard-coded FILL.

    trim_neck: if True, automatically detect and exclude the neck region by
        looking at the X-width of forward-facing geometry as a function of Z.
        The chin is the lowest Z where the head is still ≥60% of its maximum
        width.  Below that, the geometry is the neck and is excluded from the
        face bbox.

    Returns (x_min, x_max, z_min, z_max).
    """
    xs, zs = [], []
    z_to_xs: dict = {}
    # Blender convention: face position at -Y, outward face normals point -Y.
    # Front-facing (visible from FRONT view camera) means n.y < -threshold.
    for poly in me.polygons:
        n = (mw3 @ poly.normal).normalized()
        if n.y < -normal_threshold:
            for vi in poly.vertices:
                p = mw @ me.vertices[vi].co
                xs.append(p.x); zs.append(p.z)
                z_bin = round(p.z, 1)
                z_to_xs.setdefault(z_bin, []).append(p.x)
    if not xs:
        raise RuntimeError("No front-facing polys found (normal_threshold too high?)")

    z_min, z_max = min(zs), max(zs)
    x_min, x_max = min(xs), max(xs)

    if trim_neck and z_to_xs:
        # Width per Z bin (10cm slices)
        widths = {z: max(xs_bin) - min(xs_bin) for z, xs_bin in z_to_xs.items()
                  if len(xs_bin) >= 5}
        if widths:
            max_w = max(widths.values())
            # Find the lowest Z still ≥40% of max width — that's the chin/jaw line.
            # The neck (below this) is narrower and folds away.  Do NOT trim the
            # scalp on top — the image silhouette includes it.
            chin_thresh = 0.4 * max_w
            chin_z_candidates = [z for z, w in widths.items() if w >= chin_thresh]
            if chin_z_candidates:
                chin_z = min(chin_z_candidates) - 0.05   # half a slice for clearance
                if chin_z > z_min:
                    print(f"  Neck trim: z_min {z_min:.4f} → {chin_z:.4f} (max width={max_w:.3f})")
                    z_min = chin_z

    return x_min, x_max, z_min, z_max


def _landmark_image_bbox(lms_photo):
    """
    Bounding box of MediaPipe landmarks in image space.
    Returns (x_min, x_max, y_min, y_max) where y=0 is top (MediaPipe convention).
    """
    xs = [lm[0] for lm in lms_photo]
    ys = [lm[1] for lm in lms_photo]
    return min(xs), max(xs), min(ys), max(ys)


def _front_landmarks_to_control_points_bbox(lms_photo, me, mw, mw3, obj,
                                            image_bbox=None):
    """
    BBOX-correspondence front projection.

    Maps an image bbox to the front-facing mesh XZ bbox via similarity transform.
    For each landmark:
      1. Compute its position within the IMAGE bbox (normalized).
      2. Apply that same normalized position to the front-face mesh XZ bbox.
      3. Ray-cast forward (-Y to +Y) from that mesh XZ to find the surface.
      4. Record (hit_world_pos → (lm_x, 1-lm_y)) as a control point.

    The atlas UV (lm_x, 1 - lm_y) is used directly — no virtual camera FOV.
    The image IS the atlas.

    image_bbox: (x_min, x_max, y_min, y_max) — image region that the mesh face
        bbox maps to.  If None, uses the landmark bbox.  Pass the head silhouette
        bbox (from _detect_image_head_bbox) for correct scale when the image
        contains more than just the MediaPipe-detected face region (scalp, ears).

    Returns (ctrl_xyz, ctrl_uv) as numpy arrays.
    """
    from mathutils import Vector
    mw_inv = mw.inverted()

    # Image bbox: silhouette if provided, else landmark bbox
    if image_bbox is None:
        ix_min, ix_max, iy_min, iy_max = _landmark_image_bbox(lms_photo)
        bbox_kind = "landmark bbox"
    else:
        ix_min, ix_max, iy_min, iy_max = image_bbox
        bbox_kind = "head silhouette"
    img_w, img_h = ix_max - ix_min, iy_max - iy_min

    # Mesh front-face bbox in world XZ
    mx_min, mx_max, mz_min, mz_max = _front_face_mesh_bbox(me, mw, mw3)
    mesh_w, mesh_h = mx_max - mx_min, mz_max - mz_min

    print(f"  Image {bbox_kind:>15}: x=[{ix_min:.4f},{ix_max:.4f}] y=[{iy_min:.4f},{iy_max:.4f}]  w/h={img_w/img_h:.3f}")
    print(f"  Mesh  face bbox     : x=[{mx_min:.4f},{mx_max:.4f}] z=[{mz_min:.4f},{mz_max:.4f}]  w/h={mesh_w/mesh_h:.3f}")

    # Y range for ray origin (start well behind the mesh)
    y_origins = [(mw @ v.co).y for v in me.vertices]
    y_min_world = min(y_origins)

    ctrl_xyz = []
    ctrl_uv  = []
    for lm_x, lm_y in lms_photo:
        # Normalized position WITHIN the landmark bbox (0..1)
        nx = (lm_x - ix_min) / img_w if img_w > 0 else 0.5
        ny = (lm_y - iy_min) / img_h if img_h > 0 else 0.5

        # Same normalized position on the mesh front-face bbox
        mesh_x = mx_min + nx * mesh_w
        # image y=0 at top, mesh z=high at top — flip
        mesh_z = mz_max - ny * mesh_h

        # Ray-cast forward (-Y → +Y) at this XZ
        origin_w = Vector((mesh_x, y_min_world - 1.0, mesh_z))
        dir_w    = Vector((0.0, 1.0, 0.0))
        origin_l = mw_inv @ origin_w
        dir_l    = (mw_inv.to_3x3() @ dir_w).normalized()
        hit, loc_l, _n, _fi = obj.ray_cast(origin_l, dir_l)
        if not hit:
            continue
        loc_w = mw @ loc_l

        # UV is the landmark image position directly
        u = lm_x
        v = 1.0 - lm_y
        ctrl_xyz.append((loc_w.x, loc_w.y, loc_w.z))
        ctrl_uv.append((u, v))

    return (np.array(ctrl_xyz, dtype=np.float32),
            np.array(ctrl_uv,  dtype=np.float32))


def _side_face_mesh_projection_bbox(me, mw, mw3, side_angle_deg: float,
                                     trim_neck: bool = True):
    """
    Silhouette bbox of the ENTIRE mesh, projected onto the side camera plane.

    This is the mesh's silhouette extent from the side view — which is what
    matches the image silhouette bbox.  Only the neck is trimmed (same auto-
    detect as the front).

    Returns (u_min, u_max, v_min, v_max) where u is the side image's
    horizontal axis and v is world Z.
    """
    import math as _math
    sa = _math.sin(_math.radians(side_angle_deg))
    ca = _math.cos(_math.radians(side_angle_deg))

    us, vs = [], []
    z_to_us: dict = {}
    # Right axis = (ca, -sa, 0) — must match _make_side_ray_fn.  In the new
    # convention (face at -Y), nose (-Y position) projects to high u_proj
    # (image RIGHT where nose is), and the visible left ear (-X) projects to
    # low u_proj (image LEFT where ear is).
    for v in me.vertices:
        p = mw @ v.co
        u_proj = p.x * ca - p.y * sa
        us.append(u_proj)
        vs.append(p.z)
        z_bin = round(p.z, 1)
        z_to_us.setdefault(z_bin, []).append(u_proj)

    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    if trim_neck and z_to_us:
        # Width per Z slice in the side projection (neck is the narrow bottom)
        widths = {z: max(us_bin) - min(us_bin) for z, us_bin in z_to_us.items()
                  if len(us_bin) >= 5}
        if widths:
            max_w = max(widths.values())
            chin_thresh = 0.4 * max_w
            chin_z_candidates = [z for z, w in widths.items() if w >= chin_thresh]
            if chin_z_candidates:
                chin_z = min(chin_z_candidates) - 0.05
                if chin_z > v_min:
                    print(f"  Side neck trim: v_min {v_min:.4f} → {chin_z:.4f}")
                    v_min = chin_z

    return u_min, u_max, v_min, v_max


def _side_landmarks_to_control_points_bbox(lms_photo, me, mw, mw3, obj,
                                            side_angle_deg: float,
                                            image_bbox: tuple):
    """
    Side bbox-correspondence projection.

    Maps the side image silhouette bbox onto the mesh's side-projection bbox.
    For each landmark in the side image:
      1. Normalize its position within the image silhouette bbox.
      2. Apply that normalized position to the mesh side-projection bbox to get
         a target (u_proj, v_proj) on the camera plane.
      3. Convert to a world-space ray from the camera direction, and ray-cast
         onto the mesh.
      4. Record (hit_world_pos → (lm_x, 1-lm_y)) as a control point.
    """
    import math as _math
    from mathutils import Vector

    sa = _math.sin(_math.radians(side_angle_deg))
    ca = _math.cos(_math.radians(side_angle_deg))
    # Match _make_side_ray_fn convention (face at -Y).
    right    = Vector((ca, -sa, 0.0))
    up       = Vector((0.0, 0.0, 1.0))
    cam_look = Vector((sa, ca, 0.0))

    mw_inv = mw.inverted()

    ix_min, ix_max, iy_min, iy_max = image_bbox
    img_w, img_h = ix_max - ix_min, iy_max - iy_min

    mu_min, mu_max, mv_min, mv_max = _side_face_mesh_projection_bbox(
        me, mw, mw3, side_angle_deg)
    mesh_u_w, mesh_v_h = mu_max - mu_min, mv_max - mv_min

    print(f"  Image silhouette : x=[{ix_min:.4f},{ix_max:.4f}] y=[{iy_min:.4f},{iy_max:.4f}]  w/h={img_w/img_h:.3f}")
    print(f"  Mesh side proj   : u=[{mu_min:.4f},{mu_max:.4f}] v=[{mv_min:.4f},{mv_max:.4f}]  w/h={mesh_u_w/mesh_v_h:.3f}")

    ctrl_xyz = []
    ctrl_uv  = []
    for lm_x, lm_y in lms_photo:
        nx = (lm_x - ix_min) / img_w if img_w > 0 else 0.5
        ny = (lm_y - iy_min) / img_h if img_h > 0 else 0.5

        u_proj = mu_min + nx * mesh_u_w
        v_proj = mv_max - ny * mesh_v_h

        plane_pt = right * u_proj + up * v_proj
        origin_w = plane_pt - cam_look * 10.0
        dir_w    = cam_look

        origin_l = mw_inv @ origin_w
        dir_l    = (mw_inv.to_3x3() @ dir_w).normalized()
        hit, loc_l, _n, _fi = obj.ray_cast(origin_l, dir_l)
        if not hit:
            continue
        loc_w = mw @ loc_l

        ctrl_xyz.append((loc_w.x, loc_w.y, loc_w.z))
        ctrl_uv.append((lm_x, 1.0 - lm_y))

    return (np.array(ctrl_xyz, dtype=np.float32),
            np.array(ctrl_uv,  dtype=np.float32))


def _face_aligned_center(lms_photo, bb_ctr, scale, me, mw, mw3):
    """
    Return a corrected bb_ctr whose Z is aligned to the face, not the full-mesh
    bounding box.

    The MediaPipe landmark centroid in image space (v_face_center) tells us where
    the face sits within the photo frame.  The camera model projects v=0.5 to
    bb_ctr.z, but when the face is off-center in the image (v_face_center != 0.5),
    that assumption is wrong — every ray lands shifted, and lower-face landmarks
    fall outside the mesh entirely.

    This function recomputes bb_ctr.z so that v_face_center projects to the Z
    centroid of front-facing mesh polygons, giving correct landmark placement for
    the full face including the chin.
    """
    from mathutils import Vector

    cy_img       = sum(lm[1] for lm in lms_photo) / len(lms_photo)
    v_face_center = 1.0 - cy_img  # MediaPipe y=0=top → UV v=1=top

    # Z centroid of front-facing polys
    front_z_vals = []
    for poly in me.polygons:
        n_world = (mw3 @ poly.normal).normalized()
        if _zone_for_poly(n_world) == "front":
            poly_ctr_z = sum((mw @ me.vertices[vi].co).z for vi in poly.vertices) / len(poly.vertices)
            front_z_vals.append(poly_ctr_z)

    if not front_z_vals:
        return bb_ctr

    face_center_z = sum(front_z_vals) / len(front_z_vals)
    ctr = Vector(bb_ctr)
    ctr.z = face_center_z - (v_face_center - 0.5) * scale
    print(f"  Face centroid: img_v={v_face_center:.4f}  mesh_z={face_center_z:.4f}  "
          f"bb_ctr_z: {bb_ctr.z:.4f} → {ctr.z:.4f}")
    return ctr


def _landmarks_to_control_points(lms_photo, scale, bb_ctr, obj, ray_fn):
    """
    For each MediaPipe landmark, ray-cast through obj and collect
    (hit_world_pos, uv) control points.

    lms_photo : list of (lm_x, lm_y) in [0,1] image-normalised coords
                (y=0 top, y=1 bottom — MediaPipe convention)
    ray_fn    : _front_ray_from_photo_uv or _side_ray_from_photo_uv
    Returns (ctrl_xyz, ctrl_uv) as float32 arrays.
    """
    from mathutils import Vector

    mw     = obj.matrix_world
    mw_inv = mw.inverted()

    ctrl_xyz = []
    ctrl_uv  = []

    for lm_x, lm_y in lms_photo:
        u = lm_x
        v = 1.0 - lm_y   # flip: MediaPipe y=0=top → UV v=1=top

        origin_w, dir_w = ray_fn(u, v, scale, bb_ctr)

        # Transform to object local space for ray_cast
        origin_l = mw_inv @ origin_w
        dir_l    = (mw_inv.to_3x3() @ dir_w).normalized()

        hit, loc_l, _normal, _fi = obj.ray_cast(origin_l, dir_l)
        if not hit:
            continue

        loc_w = mw @ loc_l
        ctrl_xyz.append((loc_w.x, loc_w.y, loc_w.z))
        ctrl_uv.append((u, v))

    return np.array(ctrl_xyz, dtype=np.float32), np.array(ctrl_uv, dtype=np.float32)


# ── RBF interpolation ─────────────────────────────────────────────────────────

def _rbf_uv(ctrl_xyz, ctrl_uv, query_xyz, clamp_bbox=None):
    """
    Thin-plate-spline RBF: ctrl_xyz (N,3) → ctrl_uv (N,2).

    clamp_bbox: (u_min, u_max, v_min, v_max) for the output UV clamp.  If None,
        clamp to the observed control-point UV range (face features only).
        Pass the image silhouette bbox to allow mesh verts outside the face
        landmark coverage to still reach the image's full skin/head area.
    """
    from scipy.interpolate import RBFInterpolator

    rbf = RBFInterpolator(ctrl_xyz, ctrl_uv, kernel='thin_plate_spline', smoothing=0.0)
    uvs = rbf(query_xyz)

    if clamp_bbox is None:
        u_min, u_max = float(ctrl_uv[:, 0].min()), float(ctrl_uv[:, 0].max())
        v_min, v_max = float(ctrl_uv[:, 1].min()), float(ctrl_uv[:, 1].max())
    else:
        u_min, u_max, v_min, v_max = clamp_bbox
    uvs[:, 0] = np.clip(uvs[:, 0], u_min, u_max)
    uvs[:, 1] = np.clip(uvs[:, 1], v_min, v_max)
    return uvs.astype(np.float32)


# ── zone assignment (unchanged from camera_projection_uv.py) ─────────────────

def _zone_for_poly(n_world):
    """Normal-based zone. Returns 'front', 'qleft', 'qright', 'back'."""
    from mathutils import Vector
    F2C_FRONT  = Vector(( 0.0, +1.0, 0.0))
    F2C_QLEFT  = Vector((-_sa, +_ca, 0.0))
    F2C_QRIGHT = Vector(( _sa, +_ca, 0.0))
    sf = n_world.dot(F2C_FRONT)
    sl = n_world.dot(F2C_QLEFT)
    sr = n_world.dot(F2C_QRIGHT)
    best = max(sf, sl, sr)
    if best < BACK_TOL:
        return "back"
    if sf + FRONT_BIAS >= sl and sf + FRONT_BIAS >= sr:
        return "front"
    return "qleft" if sl >= sr else "qright"


# ── main calibration entry point ──────────────────────────────────────────────

def calibrate(
    obj_name:      str,
    front_img:     str,
    side_img:      str,
    uv_name:       str = "FacePlate",
    has_right_img: bool = False,
) -> dict:
    """
    Build or replace the FacePlate UV layer on obj_name using MediaPipe calibration.

    front_img : path to front SR/albedo photo (full resolution OK)
    side_img  : path to 3/4-left SR/albedo photo
    has_right_img : if True, a mirrored right-side image is assumed (reserved)

    Returns a stats dict.
    """
    import bpy
    from mathutils import Vector

    obj = bpy.data.objects[obj_name]
    me  = obj.data
    mw  = obj.matrix_world
    mw3 = mw.to_3x3()

    # ray_cast requires Object Mode — ensure it
    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Bounding box + scale (same framing as camera_projection_uv.py)
    world_verts = [mw @ v.co for v in me.vertices]
    bb_min = Vector([min(v[i] for v in world_verts) for i in range(3)])
    bb_max = Vector([max(v[i] for v in world_verts) for i in range(3)])
    bb_ctr = (bb_min + bb_max) * 0.5
    head_h = bb_max.z - bb_min.z
    scale  = head_h / FILL

    # ── Step 1: MediaPipe landmarks on source photos ──────────────────────────
    print("Running MediaPipe on front image …")
    lms_front = _run_mediapipe(front_img)
    print(f"  Front: {len(lms_front)} landmarks")

    print("Running MediaPipe on side image …")
    lms_side  = _run_mediapipe(side_img)
    print(f"  Side : {len(lms_side)} landmarks")

    if not lms_front:
        raise RuntimeError(f"MediaPipe found no face in front image: {front_img}")
    if not lms_side:
        raise RuntimeError(f"MediaPipe found no face in side image:  {side_img}")

    # ── Step 2: Align camera center, then ray-cast ────────────────────────────
    print("Aligning camera center to face …")
    bb_ctr_front = _face_aligned_center(lms_front, bb_ctr, scale, me, mw, mw3)
    bb_ctr_side  = _face_aligned_center(lms_side,  bb_ctr, scale, me, mw, mw3)

    print("Ray-casting front landmarks onto mesh …")
    ctrl_xyz_front, ctrl_uv_front = _landmarks_to_control_points(
        lms_front, scale, bb_ctr_front, obj, _front_ray_from_photo_uv)
    print(f"  Front hits: {len(ctrl_xyz_front)} / {len(lms_front)}")

    print("Ray-casting side landmarks onto mesh …")
    ctrl_xyz_side, ctrl_uv_side = _landmarks_to_control_points(
        lms_side, scale, bb_ctr_side, obj, _make_side_ray_fn(SIDE_ANGLE_DEG))
    print(f"  Side  hits: {len(ctrl_xyz_side)} / {len(lms_side)}")

    if len(ctrl_xyz_front) < 10:
        raise RuntimeError("Too few front landmark hits — check FILL / framing")
    if len(ctrl_xyz_side) < 10:
        raise RuntimeError("Too few side landmark hits — check FILL / framing")

    # ── Step 3: RBF → UV for every mesh vertex ────────────────────────────────
    all_xyz = np.array([[*(mw @ v.co),] for v in me.vertices], dtype=np.float32)

    print("Building front RBF …")
    uv_front = _rbf_uv(ctrl_xyz_front, ctrl_uv_front, all_xyz)  # (N, 2)

    print("Building side RBF …")
    uv_side  = _rbf_uv(ctrl_xyz_side,  ctrl_uv_side,  all_xyz)  # (N, 2)

    # ── Step 4: Write UV layer + zone vertex color ────────────────────────────
    if uv_name in me.uv_layers:
        me.uv_layers.remove(me.uv_layers[uv_name])
    uv_layer = me.uv_layers.new(name=uv_name)
    me.uv_layers.active = uv_layer
    uv_data  = uv_layer.data

    zone_vcol_name = "FacePlate_Zone"
    if zone_vcol_name not in me.color_attributes:
        me.color_attributes.new(name=zone_vcol_name, type='FLOAT_COLOR', domain='CORNER')
    vc_data = me.color_attributes[zone_vcol_name].data

    verts  = me.vertices
    loops  = me.loops
    counts = {"front": 0, "qleft": 0, "qright": 0, "back": 0}

    for poly in me.polygons:
        n_world = (mw3 @ poly.normal).normalized()
        zone    = _zone_for_poly(n_world)
        counts[zone] += 1

        # zone_b=0 → reads R (front grey), zone_b=1 → reads G (side grey)
        # qright mirrors the side image horizontally (u_right = 1 - u_left)
        zone_b = 1.0 if zone in ("qleft", "qright") else 0.0

        for li in poly.loop_indices:
            vi = loops[li].vertex_index

            if zone == "back":
                uv_data[li].uv = (0.5, 0.5)
            elif zone == "front":
                uv_data[li].uv = (float(uv_front[vi, 0]), float(uv_front[vi, 1]))
            elif zone == "qleft":
                uv_data[li].uv = (float(uv_side[vi, 0]),  float(uv_side[vi, 1]))
            else:  # qright — same UV as qleft; both sides sample the face region
                uv_data[li].uv = (float(uv_side[vi, 0]),  float(uv_side[vi, 1]))

            vc_data[li].color = (0.0, 0.0, zone_b, 1.0)

    me.update()

    total = sum(counts.values())
    return {
        "scale":             round(scale, 4),
        "head_h":            round(head_h, 4),
        "front_ctrl_pts":    len(ctrl_xyz_front),
        "side_ctrl_pts":     len(ctrl_xyz_side),
        "zone_counts":       counts,
        "front_pct":         round(counts["front"] / total * 100, 1),
        "back_pct":          round(counts["back"]  / total * 100, 1),
        "uv_layer":          uv_name,
    }


# ── JSON-based calibration (reads ComfyUI MediaPipeFaceExport output) ────────

def _load_landmarks_json(json_path: str) -> tuple:
    """
    Load landmark list from a MediaPipeFaceExport JSON file.
    Returns (landmarks, angle) where landmarks is a list of (x, y) tuples
    in [0,1] image-normalised coords (origin top-left, x right, y down),
    and angle is "front" | "side_left" | "side_right".
    Returns ([], angle) if landmark_count == 0.
    """
    import json
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Landmark JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    angle = data.get("angle", "front")
    count = data.get("landmark_count", 0)
    if count == 0:
        return [], angle
    return [tuple(lm) for lm in data["landmarks"]], angle


def calibrate_front_only(
    obj_name:    str,
    front_json:  str,
    front_img:   str   = None,
    uv_name:     str   = "FacePlate",
    skin_uv:     tuple = (0.5, 0.5),
) -> dict:
    """
    Front-only calibration using bbox-correspondence projection.

    No virtual-camera FOV. The atlas IS the source image, so atlas UV (u,v) =
    (lm_x, 1 - lm_y) for every MediaPipe landmark.  Mesh ↔ image correspondence
    is established by mapping the landmark image bbox onto the front-face mesh
    bbox via similarity transform.

    Zones:
      - "front" (n.y > 0.4)  → RBF-interpolated UV from front landmarks (atlas R)
      - any other            → skin_uv (neutral skin sample, R channel; covered
                                by hair or base material anyway)

    zone vertex color B channel: always 0 (R / front sample) for the moment.

    Args:
        skin_uv: (u, v) sample point used for back / side polys.  Pick a clean
                 mid-cheek pixel in the front image so the back of head reads as
                 a flat skin patch.
    """
    import bpy
    from mathutils import Vector

    obj = bpy.data.objects[obj_name]
    me  = obj.data
    mw  = obj.matrix_world
    mw3 = mw.to_3x3()

    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    print(f"Loading front landmarks from {front_json} …")
    lms_front, _ = _load_landmarks_json(front_json)
    if not lms_front:
        raise RuntimeError(f"No landmarks in front JSON: {front_json}")
    print(f"  Front: {len(lms_front)} landmarks")

    image_bbox = None
    if front_img:
        print(f"Detecting head silhouette in {front_img} …")
        image_bbox = _detect_image_head_bbox(front_img)
        print(f"  Silhouette bbox: x=[{image_bbox[0]:.4f},{image_bbox[1]:.4f}] "
              f"y=[{image_bbox[2]:.4f},{image_bbox[3]:.4f}]")

    print("Building front control points (bbox correspondence) …")
    ctrl_xyz, ctrl_uv = _front_landmarks_to_control_points_bbox(
        lms_front, me, mw, mw3, obj, image_bbox=image_bbox)
    print(f"  Front hits: {len(ctrl_xyz)} / {len(lms_front)}")

    if len(ctrl_xyz) < 20:
        raise RuntimeError("Too few front landmark hits — check mesh orientation / normals")

    all_xyz = np.array([[*(mw @ v.co),] for v in me.vertices], dtype=np.float32)

    print("Building front RBF …")
    uv_front = _rbf_uv(ctrl_xyz, ctrl_uv, all_xyz)

    # Write UV layer + zone vertex color
    if uv_name in me.uv_layers:
        me.uv_layers.remove(me.uv_layers[uv_name])
    uv_layer = me.uv_layers.new(name=uv_name)
    me.uv_layers.active = uv_layer
    uv_data = uv_layer.data

    zone_vcol_name = "FacePlate_Zone"
    if zone_vcol_name not in me.color_attributes:
        me.color_attributes.new(name=zone_vcol_name, type='FLOAT_COLOR', domain='CORNER')
    vc_data = me.color_attributes[zone_vcol_name].data

    loops  = me.loops
    counts = {"front": 0, "skin_fill": 0}

    for poly in me.polygons:
        n_world = (mw3 @ poly.normal).normalized()
        is_front = n_world.y > 0.4
        if is_front:
            counts["front"] += 1
        else:
            counts["skin_fill"] += 1

        for li in poly.loop_indices:
            vi = loops[li].vertex_index
            if is_front:
                uv_data[li].uv = (float(uv_front[vi, 0]), float(uv_front[vi, 1]))
            else:
                uv_data[li].uv = skin_uv
            # B channel = 0 means "sample R" (front) in the material lerp
            vc_data[li].color = (0.0, 0.0, 0.0, 1.0)

    me.update()

    total = sum(counts.values())
    return {
        "front_ctrl_pts":  len(ctrl_xyz),
        "front_polys":     counts["front"],
        "skin_fill_polys": counts["skin_fill"],
        "front_pct":       round(counts["front"] / total * 100, 1),
        "uv_layer":        uv_name,
        "skin_uv":         skin_uv,
    }


# ─── Canonical pipeline dimensions ────────────────────────────────────────────
# All character heads must be scaled to these dimensions before calibration so
# that:
#   - eye/lash/teeth/jaw library parts drop in at native scale 1.0
#   - world-unit thresholds (ear_y_offset) work consistently per character
#   - skeleton bind weights and ARKit shape key transfer use consistent geometry
CANONICAL_HEAD_WIDTH = 0.252     # Fortnite_Head_LOD0 world bbox X
CANONICAL_EYE_WIDTH  = 0.045     # Lib_Eye world bbox X at scale 1.0
EYE_LIBRARY_BLEND    = (r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters"
                        r"\Head\CanonHead_WIP4.blend")

# MediaPipe eye landmark groups (full eye ring including upper/lower lid points)
MP_LEFT_EYE_RING  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                     157, 158, 159, 160, 161, 246]
MP_RIGHT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398,
                     384, 385, 386, 387, 388, 466]
MP_LEFT_EYE_OUTER  = 33
MP_LEFT_EYE_INNER  = 133
MP_RIGHT_EYE_OUTER = 263
MP_RIGHT_EYE_INNER = 362


def recalculate_normals_outside(obj_name: str) -> dict:
    """
    Force all mesh face normals to point OUTWARD (away from mesh center).

    Trellis / Tripo / InstaLOD meshes often arrive partially inside-out:
    some polys have outward normals, others (often the dense face region)
    point inward.  Unreal renders with backface culling by default, so the
    inward-facing polys' vertex colors and shading appear on the inside of
    the head — invisible from outside.

    CALL FIRST in the pipeline (before `normalize_head_to_canonical`).
    Required so all downstream normal-based zone checks (front/top) catch
    the correct polys.
    """
    import bpy
    obj = bpy.data.objects[obj_name]

    if obj.name not in bpy.context.view_layer.objects:
        bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    override = {"active_object": obj, "object": obj,
                "selected_objects": [obj],
                "selected_editable_objects": [obj]}
    window = bpy.context.window or (bpy.context.window_manager.windows[0]
             if bpy.context.window_manager.windows else None)
    if window:
        screen = window.screen
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        override.update(window=window, screen=screen,
                                        area=area, region=region)
                        break
                break

    with bpy.context.temp_override(**override):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)  # = Recalculate Outside
        bpy.ops.object.mode_set(mode='OBJECT')

    # Second pass: position-based outward check.  For each poly compute the
    # vector from the mesh center to the poly center.  If the poly's normal
    # has a NEGATIVE dot with that vector, the normal points inward — flip
    # the face.  This catches the non-manifold stragglers that
    # Recalculate Outside can't resolve from topology alone.
    from mathutils import Vector
    me = obj.data
    mw = obj.matrix_world
    mw3 = mw.to_3x3()

    world_verts = [mw @ v.co for v in me.vertices]
    bb_min = Vector([min(v[i] for v in world_verts) for i in range(3)])
    bb_max = Vector([max(v[i] for v in world_verts) for i in range(3)])
    mesh_ctr = (bb_min + bb_max) * 0.5

    flipped_idxs = []
    for poly in me.polygons:
        ctr_w = Vector((0,0,0))
        for vi in poly.vertices:
            ctr_w += mw @ me.vertices[vi].co
        ctr_w /= len(poly.vertices)
        outward = (ctr_w - mesh_ctr)
        if outward.length < 1e-6:
            continue
        outward = outward.normalized()
        n_world = (mw3 @ poly.normal).normalized()
        # Negative dot = normal points toward center (inward) = needs flip
        if n_world.dot(outward) < -0.1:
            flipped_idxs.append(poly.index)

    if flipped_idxs:
        with bpy.context.temp_override(**override):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')
            for idx in flipped_idxs:
                me.polygons[idx].select = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Recalculated normals outside on {obj_name} "
          f"(+ flipped {len(flipped_idxs)} non-manifold stragglers)")
    return {"recalculated": True, "obj": obj_name,
            "second_pass_flipped": len(flipped_idxs)}


def normalize_head_to_canonical(obj_name: str,
                                target_width: float = CANONICAL_HEAD_WIDTH
                                ) -> dict:
    """
    Scale a head mesh to canonical Fortnite head width (X dimension).

    CALL FIRST, before `calibrate_full`.  Calibration thresholds — especially
    `ear_y_offset` which is in world units — are tuned for the canonical scale.
    Without normalizing, an oversized head will have the ear-line cut land in
    the wrong place and pipeline library parts (eyes, lashes, teeth) won't fit.

    Applies the scale (so `obj.scale` returns to 1.0 but the mesh data carries
    the new size).  Does NOT touch rotation or location.
    """
    import bpy
    obj = bpy.data.objects[obj_name]
    cur_w = obj.dimensions.x
    if cur_w <= 0:
        raise RuntimeError(f"{obj_name} has zero X dimension")
    scale = target_width / cur_w

    # Ensure operator context
    if obj.name not in bpy.context.view_layer.objects:
        bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    obj.scale = (obj.scale[0] * scale,
                 obj.scale[1] * scale,
                 obj.scale[2] * scale)

    override = {"active_object": obj, "object": obj,
                "selected_objects": [obj],
                "selected_editable_objects": [obj]}
    window = bpy.context.window or (bpy.context.window_manager.windows[0]
             if bpy.context.window_manager.windows else None)
    if window:
        screen = window.screen
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        override["window"] = window
                        override["screen"] = screen
                        override["area"] = area
                        override["region"] = region
                        break
                break

    with bpy.context.temp_override(**override):
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    print(f"  Head normalized: scale ×{scale:.4f}, new W={obj.dimensions.x:.4f}")
    return {
        "scale_applied":   round(scale, 4),
        "new_dimensions":  [round(v, 4) for v in obj.dimensions],
        "target_width":    target_width,
    }


def fit_canonical_eyes(obj_name:        str,
                       front_json:      str,
                       lib_blend:       str  = EYE_LIBRARY_BLEND,
                       eye_target_w:    float = CANONICAL_EYE_WIDTH,
                       inset:           float = None,
                       parent_to_head:  bool  = True) -> dict:
    """
    Append `Lib_Eye_L` and `Lib_Eye_R` from the pipeline library blend, scale
    to canonical eye width, and position at MediaPipe landmark-derived eye
    centroids.  Optionally parent to the head mesh.

    Run AFTER `normalize_head_to_canonical`, `calibrate_full`,
    `cleanup_interior_faces`, and `decimate_back_zone` (in that order) — the
    eye-centroid lookup uses the front-zone UVs which are set by calibration.

    GOTCHAS handled automatically here:
      1. Lib_Eye_L/R origin is offset ~1.45 below their geometry → BOUNDS-center
         origin first so `location` matches the eye centroid.
      2. Lib_Eye sizing is for canonical scale; if the head isn't normalized
         yet, the eyes will look comically small or large.
    """
    import bpy, json
    from mathutils import Vector

    head = bpy.data.objects[obj_name]

    # ── Append library eyes (rename per-character) ────────────────────────────
    new_l_name = f"{obj_name}_Eye_L"
    new_r_name = f"{obj_name}_Eye_R"
    for n in (new_l_name, new_r_name):
        if n in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)

    with bpy.data.libraries.load(lib_blend, link=False) as (src, dst):
        wanted = [n for n in src.objects if n in ("Lib_Eye_L", "Lib_Eye_R")]
        dst.objects = wanted
    scene = bpy.context.scene
    eye_l = eye_r = None
    for obj in dst.objects:
        if obj is None:
            continue
        scene.collection.objects.link(obj)
        if obj.name.startswith("Lib_Eye_L"):
            obj.name = new_l_name
            eye_l = obj
        elif obj.name.startswith("Lib_Eye_R"):
            obj.name = new_r_name
            eye_r = obj
    if eye_l is None or eye_r is None:
        raise RuntimeError(f"Failed to append Lib_Eye_L / Lib_Eye_R from {lib_blend}")

    # ── Build operator context override (one-shot) ────────────────────────────
    def _make_override(obj):
        override = {"active_object": obj, "object": obj,
                    "selected_objects": [obj],
                    "selected_editable_objects": [obj]}
        window = bpy.context.window or (bpy.context.window_manager.windows[0]
                 if bpy.context.window_manager.windows else None)
        if window:
            screen = window.screen
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            override.update(window=window, screen=screen,
                                            area=area, region=region)
                            break
                    break
        return override

    # ── Center origin on geometry (fixes the 1.45-unit baked-in offset) ──────
    for eye in (eye_l, eye_r):
        bpy.context.view_layer.objects.active = eye
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        eye.select_set(True)
        with bpy.context.temp_override(**_make_override(eye)):
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # ── Scale eyes to canonical width (in case lib version drifted) ──────────
    for eye in (eye_l, eye_r):
        cur_w = eye.dimensions.x
        s = eye_target_w / cur_w if cur_w > 0 else 1.0
        eye.scale = (eye.scale[0] * s, eye.scale[1] * s, eye.scale[2] * s)
        bpy.context.view_layer.objects.active = eye
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        eye.select_set(True)
        with bpy.context.temp_override(**_make_override(eye)):
            bpy.ops.object.transform_apply(location=False, rotation=False,
                                           scale=True)

    # ── Compute eye centroids from MediaPipe landmarks via front UV ──────────
    with open(front_json) as f:
        lms = json.load(f)["landmarks"]

    me = head.data
    mw = head.matrix_world
    uv_layer = me.uv_layers["FacePlate"]
    zone_attr = me.color_attributes["FacePlate_Zone"]

    def eye_centroid(indices):
        cx_uv = sum(lms[i][0] for i in indices) / len(indices)
        cy_uv = sum(lms[i][1] for i in indices) / len(indices)
        tu, tv = cx_uv, 1.0 - cy_uv
        best_d = 1e9; best_pos = None
        for poly in me.polygons:
            li0 = poly.loop_indices[0]
            if zone_attr.data[li0].color[2] > 0.5:
                continue
            u_sum = v_sum = 0.0
            for li in poly.loop_indices:
                u, v = uv_layer.data[li].uv
                u_sum += u; v_sum += v
            u_ctr = u_sum / len(poly.loop_indices)
            v_ctr = v_sum / len(poly.loop_indices)
            d = (u_ctr - tu)**2 + (v_ctr - tv)**2
            if d < best_d:
                best_d = d
                ctr = Vector((0,0,0))
                for vi in poly.vertices:
                    ctr += mw @ me.vertices[vi].co
                ctr /= len(poly.vertices)
                best_pos = ctr
        return best_pos

    l_pos = eye_centroid(MP_LEFT_EYE_RING)
    r_pos = eye_centroid(MP_RIGHT_EYE_RING)
    if l_pos is None or r_pos is None:
        raise RuntimeError("Eye centroid lookup failed — does the mesh have a FacePlate UV layer?")

    # Default inset: half the eye width so the eye CENTER sits behind the head
    # surface, putting the front hemisphere of the sphere AT the eye opening.
    # The eyelid socket walls then hide the equator of the sphere from view.
    if inset is None:
        inset = eye_target_w * 0.55   # slightly more than radius for safe clearance
    eye_l.location = (l_pos.x, l_pos.y + inset, l_pos.z)
    eye_r.location = (r_pos.x, r_pos.y + inset, r_pos.z)

    # ── Parent eyes to head ──────────────────────────────────────────────────
    if parent_to_head:
        bpy.context.view_layer.objects.active = head
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        eye_l.select_set(True)
        eye_r.select_set(True)
        head.select_set(True)
        with bpy.context.temp_override(**_make_override(head)):
            bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

    return {
        "eye_L_pos":  [round(v,4) for v in eye_l.location],
        "eye_R_pos":  [round(v,4) for v in eye_r.location],
        "eye_L_size": [round(v,4) for v in eye_l.dimensions],
        "eye_R_size": [round(v,4) for v in eye_r.dimensions],
    }


# MediaPipe landmark groups for feature masks (baked into FacePlate_Zone R/G).
# Unreal reads ONE vertex color attribute, so we pack everything into FacePlate_Zone:
#   R = lips mask     (1.0 where the poly's front UV lies inside the lips hull)
#   G = eyebrow mask  (1.0 where the poly's front UV lies inside the brow hull)
#   B = side zone     (0 = sample R/front, 1 = sample G/side image)
#   A = 1.0
MP_LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
MP_RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
MP_OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


def calibrate_full(
    obj_name:       str,
    front_json:     str,
    front_img:      str,
    side_json:      str,
    side_img:       str,
    side_angle_deg: float = 45.0,
    uv_name:        str   = "FacePlate",
    # Both top and back zones default to the corner skin patch — every
    # non-projected poly samples the same skin swatch the atlas builder paints
    # in the top-right corner (5% × 5% block sampled from the front-image
    # forehead area).  This makes the "hard body" decimated regions read as a
    # single clean color rather than picking up dark mouth/nose pixels.
    top_skin_uv:    tuple = (0.99, 0.99),
    back_skin_uv:   tuple = (0.99, 0.99),
    n_top_threshold:     float = 0.45,      # normal Z to qualify as "top"
    z_top_frac:          float = 0.70,      # fraction of head height: top zone if z above this frac
    front_seam_offset:   float = 0.015,     # front/side seam: eye_y + this  (positive = slightly behind eye)
    back_seam_offset:    float = 0.04,      # side/back seam:  ear_y + this  (positive = behind the ear)
    bisect_seams:        bool  = True,      # actually cut the mesh at the seam Y planes
    blend_mode:          str   = "soft",    # "soft" = normal-based front/side classification, no front
                                            #          bisect, jagged-but-smooth boundary. Best look.
                                            # "hard" = position-based zones, mesh bisected at front_seam_y,
                                            #          one straight visible seam line.
    n_front_threshold:   float = 0.30,      # soft-mode: |n.y| above this = front zone (else side)
    soft_blend_inner:    float = 0.40,      # soft-mode: |x|/half_w below this = pure front (vc_b=0)
    soft_blend_outer:    float = 0.85,      # soft-mode: |x|/half_w above this = pure side  (vc_b=1)
    dual_uv:             bool  = False,     # True = build TWO UV maps (FacePlate=front, FacePlate_Side=side),
                                            #        each covering the whole mesh.  Material samples both
                                            #        and blends with vc_b — true di-planar projection.
                                            #        Forces blend_mode='soft' (front bisect makes no sense).
    side_uv_name:        str   = "FacePlate_Side",
) -> dict:
    """
    Four-zone calibration: front (R, projection), side (G, projection),
    top of skull (skin fill, R), back of head (skin fill, G).

    The side zone is the WRAP region only (temples, ears, cheeks, jaw) —
    not the crown or occiput.  Hair / base material covers top and back in
    Unreal, so they only need a clean skin sample.

    Zones (priority order):
      back : poly center y < ear_y + ear_y_offset  (behind the ears)
      top  : n.z > n_top_threshold AND z > z_top_frac of head height
      front: n.y > n_front_threshold
      side : everything else (cheek/temple/ear/jaw wrap)

    The "behind the ears" cut is the natural projection boundary — anything in
    front of the ear line is reachable by the front or side image, anything
    behind it isn't visible from either camera and gets the fill skin patch.

    Natural UV extents observed on ernest_chavez (for reference / sanity):
      front zone : u=[0.143, 0.856] v=[0.041, 0.941]  — fills front silhouette
      side  zone : u=[0.170, 0.842] v=[0.004, 0.925]  — fills side silhouette
                   left polys mirror to the same image area as right polys
      top   zone : single sample at top_skin_uv
      back  zone : single sample at back_skin_uv
    """
    import bpy
    from mathutils import Vector

    # Dual UV implies soft mode (no point cutting the front seam if both maps
    # cover the whole mesh — the blend is in the shader, not the geometry).
    if dual_uv and blend_mode != "soft":
        print(f"  [dual_uv] forcing blend_mode='soft' (was '{blend_mode}')")
        blend_mode = "soft"

    obj = bpy.data.objects[obj_name]
    me  = obj.data
    mw  = obj.matrix_world
    mw3 = mw.to_3x3()

    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # ─── FRONT ────────────────────────────────────────────────────────────────
    print(f"Loading front landmarks from {front_json} …")
    lms_front, _ = _load_landmarks_json(front_json)
    if not lms_front:
        raise RuntimeError(f"No landmarks in front JSON: {front_json}")
    print(f"  Front: {len(lms_front)} landmarks")

    print(f"Detecting front head silhouette …")
    front_image_bbox = _detect_image_head_bbox(front_img)
    print(f"  Front silhouette: x=[{front_image_bbox[0]:.4f},{front_image_bbox[1]:.4f}] "
          f"y=[{front_image_bbox[2]:.4f},{front_image_bbox[3]:.4f}]")

    print("Building front control points …")
    ctrl_xyz_f, ctrl_uv_f = _front_landmarks_to_control_points_bbox(
        lms_front, me, mw, mw3, obj, image_bbox=front_image_bbox)
    print(f"  Front hits: {len(ctrl_xyz_f)} / {len(lms_front)}")

    # ─── SIDE ─────────────────────────────────────────────────────────────────
    print(f"Loading side landmarks from {side_json} …")
    lms_side, side_angle_label = _load_landmarks_json(side_json)
    if not lms_side:
        raise RuntimeError(f"No landmarks in side JSON: {side_json}")
    print(f"  Side : {len(lms_side)} landmarks  angle={side_angle_label}")

    print(f"Detecting side head silhouette …")
    side_image_bbox = _detect_image_head_bbox(side_img)
    print(f"  Side silhouette: x=[{side_image_bbox[0]:.4f},{side_image_bbox[1]:.4f}] "
          f"y=[{side_image_bbox[2]:.4f},{side_image_bbox[3]:.4f}]")

    print(f"Building side control points (angle={side_angle_deg}°) …")
    ctrl_xyz_s, ctrl_uv_s = _side_landmarks_to_control_points_bbox(
        lms_side, me, mw, mw3, obj, side_angle_deg, side_image_bbox)
    print(f"  Side hits: {len(ctrl_xyz_s)} / {len(lms_side)}")

    if len(ctrl_xyz_f) < 20:
        raise RuntimeError("Too few front landmark hits")
    if len(ctrl_xyz_s) < 20:
        raise RuntimeError("Too few side landmark hits")

    all_xyz = np.array([[*(mw @ v.co),] for v in me.vertices], dtype=np.float32)

    # Find the mesh's symmetry plane in X — the center of front-facing polys.
    # Used to mirror left-side polys onto the right-side RBF.
    mesh_x_center = 0.5 * (
        max((mw @ v.co).x for v in me.vertices)
      + min((mw @ v.co).x for v in me.vertices)
    )

    # Compute adaptive z_top_threshold from the mesh head extent (chin to crown).
    # The front_face_mesh_bbox already auto-trimmed the neck.
    front_mx_min, front_mx_max, front_mz_min, front_mz_max = _front_face_mesh_bbox(me, mw, mw3)
    z_top_threshold = front_mz_min + (front_mz_max - front_mz_min) * z_top_frac
    print(f"  Zone z_top_threshold = {z_top_threshold:.4f} "
          f"(head z=[{front_mz_min:.4f}, {front_mz_max:.4f}], top_frac={z_top_frac})")

    # Detect the EYE and EAR Y positions on the mesh, used to anchor the two
    # texture-island seams:
    #   - EYE Y: from the MediaPipe eye-ring landmark centroid, raycast onto
    #            the mesh — that's the most-forward "face landmark line".
    #   - EAR Y: leftmost/rightmost X verts are the ear tips.
    # Front seam sits slightly BEHIND the eye, back seam slightly behind the ear.
    rightmost_y_vert = max(me.vertices, key=lambda v: (mw @ v.co).x)
    leftmost_y_vert  = min(me.vertices, key=lambda v: (mw @ v.co).x)
    ear_y_avg = 0.5 * ((mw @ rightmost_y_vert.co).y + (mw @ leftmost_y_vert.co).y)

    # Compute eye Y from MediaPipe LEFT_EYE_RING centroid
    cy_uv = sum(lms_front[i][1] for i in MP_LEFT_EYE_RING) / len(MP_LEFT_EYE_RING)
    cx_uv = sum(lms_front[i][0] for i in MP_LEFT_EYE_RING) / len(MP_LEFT_EYE_RING)
    # Map landmark UV → mesh XZ using the front bbox correspondence
    ix_min_f, ix_max_f, iy_min_f, iy_max_f = front_image_bbox
    mx_min_f, mx_max_f, mz_min_f, mz_max_f = _front_face_mesh_bbox(me, mw, mw3)
    nx_eye = (cx_uv - ix_min_f) / (ix_max_f - ix_min_f)
    ny_eye = (cy_uv - iy_min_f) / (iy_max_f - iy_min_f)
    mesh_x_eye = mx_min_f + nx_eye * (mx_max_f - mx_min_f)
    mesh_z_eye = mz_max_f - ny_eye * (mz_max_f - mz_min_f)
    # Raycast forward (+Y from far -Y) to find world Y of mesh at the eye XZ
    y_min_w = min((mw @ v.co).y for v in me.vertices)
    origin_w = Vector((mesh_x_eye, y_min_w - 1.0, mesh_z_eye))
    dir_w    = Vector((0.0, 1.0, 0.0))
    mw_inv = mw.inverted()
    origin_l = mw_inv @ origin_w
    dir_l    = (mw_inv.to_3x3() @ dir_w).normalized()
    eye_hit, eye_loc_l, _n, _fi = obj.ray_cast(origin_l, dir_l)
    if eye_hit:
        eye_y_avg = (mw @ eye_loc_l).y
    else:
        # Fallback: eye is roughly 60% from ear toward the front
        front_min_y = min((mw @ v.co).y for v in me.vertices)
        eye_y_avg = front_min_y * 0.6 + ear_y_avg * 0.4

    front_seam_y = eye_y_avg + front_seam_offset
    back_seam_y  = ear_y_avg + back_seam_offset
    # ear_y_line kept as alias for downstream code that still references it
    ear_y_line = back_seam_y
    print(f"  Eye y = {eye_y_avg:.4f}  Ear y = {ear_y_avg:.4f}")
    print(f"  Front seam y = {front_seam_y:.4f}  Back seam y = {back_seam_y:.4f}")

    # Optionally bisect the mesh along the two seam Y planes — gives perfectly
    # straight zone boundaries instead of jagged poly edges.  Done BEFORE the
    # RBF query so the new vertices get proper front/side UVs.
    if bisect_seams:
        import bmesh as _bmesh
        # In "soft" mode, only bisect the BACK seam (skin-pin boundary wants a
        # clean hard edge); leave the front face uncut so the normal-based
        # front/side classification can produce a soft jagged transition.
        # In "hard" mode, bisect both — gives explicit straight seam lines.
        seams_to_cut = ([back_seam_y]
                        if blend_mode == "soft"
                        else [front_seam_y, back_seam_y])
        bm = _bmesh.new()
        bm.from_mesh(me)
        for seam_y in seams_to_cut:
            seam_y_local = (mw_inv @ Vector((0, seam_y, 0))).y
            geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
            _bmesh.ops.bisect_plane(
                bm,
                geom=geom,
                plane_co=Vector((0.0, seam_y_local, 0.0)),
                plane_no=Vector((0.0, 1.0, 0.0)),
                clear_inner=False, clear_outer=False,
            )
        bm.to_mesh(me)
        bm.free()
        me.update()
        print(f"  Bisected mesh at {len(seams_to_cut)} seam plane(s) "
              f"({'back only — soft blend mode' if blend_mode == 'soft' else 'front + back — hard seam mode'})")
        # Rebuild all_xyz to include the new vertices created by the bisect.
        all_xyz = np.array([[*(mw @ v.co),] for v in me.vertices], dtype=np.float32)

    # Build UV clamp bboxes from the image silhouettes (in atlas UV space:
    # u = image x, v = 1 - image y).  This lets mesh verts outside the landmark
    # coverage still reach the image's full skin/silhouette area instead of
    # being squashed into the tight landmark bbox.
    f_clamp = (front_image_bbox[0], front_image_bbox[1],
               1.0 - front_image_bbox[3], 1.0 - front_image_bbox[2])
    s_clamp = (side_image_bbox[0],  side_image_bbox[1],
               1.0 - side_image_bbox[3],  1.0 - side_image_bbox[2])

    print("Building front RBF …")
    uv_front = _rbf_uv(ctrl_xyz_f, ctrl_uv_f, all_xyz, clamp_bbox=f_clamp)
    print("Building side RBF …")
    uv_side  = _rbf_uv(ctrl_xyz_s, ctrl_uv_s, all_xyz, clamp_bbox=s_clamp)

    # For left-side mesh verts, query the side RBF with their X-mirrored
    # position; the mirrored UV is used directly (left = right's image content).
    mirrored_xyz = all_xyz.copy()
    mirrored_xyz[:, 0] = 2 * mesh_x_center - mirrored_xyz[:, 0]
    uv_side_mirror = _rbf_uv(ctrl_xyz_s, ctrl_uv_s, mirrored_xyz, clamp_bbox=s_clamp)

    # ─── Write UV + zone vertex color ─────────────────────────────────────────
    if uv_name in me.uv_layers:
        me.uv_layers.remove(me.uv_layers[uv_name])
    uv_layer = me.uv_layers.new(name=uv_name)
    me.uv_layers.active = uv_layer
    uv_data = uv_layer.data

    # In dual_uv mode, create a second UV map for the side projection.  Both
    # maps cover the whole mesh; the material samples both and blends with vc_b.
    uv_side_data = None
    if dual_uv:
        if side_uv_name in me.uv_layers:
            me.uv_layers.remove(me.uv_layers[side_uv_name])
        uv_side_layer = me.uv_layers.new(name=side_uv_name)
        uv_side_data  = uv_side_layer.data
        print(f"  [dual_uv] Created second UV map: {side_uv_name}")

    zone_vcol_name = "FacePlate_Zone"
    if zone_vcol_name not in me.color_attributes:
        me.color_attributes.new(name=zone_vcol_name, type='FLOAT_COLOR', domain='CORNER')
    vc_data = me.color_attributes[zone_vcol_name].data

    loops  = me.loops
    counts = {"front": 0, "side": 0, "top": 0, "back": 0}

    # Pre-compute head half-width and center X for the soft vc_b gradient.
    head_half_w = max(1e-6, 0.5 * (front_mx_max - front_mx_min))
    head_ctr_x  = 0.5 * (front_mx_max + front_mx_min)

    def _soft_vc_b(world_x: float) -> float:
        """Smoothstep blend factor: 0 near centerline → 1 at ear/temple."""
        t = abs(world_x - head_ctr_x) / head_half_w
        t = max(0.0, min(1.0, (t - soft_blend_inner)
                              / max(1e-6, soft_blend_outer - soft_blend_inner)))
        return t * t * (3.0 - 2.0 * t)

    for poly in me.polygons:
        n = (mw3 @ poly.normal).normalized()
        ctr_x = sum((mw @ me.vertices[vi].co).x for vi in poly.vertices) / len(poly.vertices)
        ctr_y = sum((mw @ me.vertices[vi].co).y for vi in poly.vertices) / len(poly.vertices)
        ctr_z = sum((mw @ me.vertices[vi].co).z for vi in poly.vertices) / len(poly.vertices)
        poly_x = ctr_x

        # ─── Zone classification ──────────────────────────────────────────────
        # Top + back zones are always position-based (skin-pin: clean hard
        # boundary where the back skin patch starts).
        # Front vs side depends on mode:
        #   "soft"  → normal-based (n.y forward = front, otherwise side).  No
        #             front bisect, jagged but natural boundary, soft vc_b
        #             gradient smooths the visual transition.
        #   "hard"  → position-based at front_seam_y, mesh bisected.  One clean
        #             straight visible line.
        if n.z > n_top_threshold and ctr_z > z_top_threshold:
            zone = "top"
        elif ctr_y > back_seam_y:
            zone = "back"
        elif blend_mode == "soft":
            zone = "front" if n.y < -n_front_threshold else "side"
        else:
            zone = "side" if ctr_y > front_seam_y else "front"

        counts[zone] += 1
        # Pipeline convention: side image has the head facing image right,
        # showing the subject's visible ear at image LEFT.  MediaPipe lands
        # reliable control points on subject's LEFT side; the "right_tragion"
        # is an unreliable estimate.  So RIGHT-side mesh polys must MIRROR
        # to query the RBF at their left-side anatomical equivalent (where
        # the reliable control points live).  LEFT-side polys use direct RBF.
        is_right_side = poly_x > mesh_x_center

        for li in poly.loop_indices:
            vi = loops[li].vertex_index

            if dual_uv:
                # Both UV maps cover every poly. Top/back zones still pin to
                # the skin patch in BOTH maps so they sample skin grey in R+G.
                if zone == "top":
                    uv_data[li].uv      = top_skin_uv
                    uv_side_data[li].uv = top_skin_uv
                elif zone == "back":
                    uv_data[li].uv      = back_skin_uv
                    uv_side_data[li].uv = back_skin_uv
                else:
                    # FacePlate = front projection (correct anatomy from front image)
                    uv_data[li].uv = (float(uv_front[vi, 0]), float(uv_front[vi, 1]))
                    # FacePlate_Side = side projection (correct anatomy from side image)
                    if is_right_side:
                        uv_side_data[li].uv = (float(uv_side_mirror[vi, 0]),
                                                float(uv_side_mirror[vi, 1]))
                    else:
                        uv_side_data[li].uv = (float(uv_side[vi, 0]),
                                                float(uv_side[vi, 1]))
            else:
                # Single-UV path (existing behavior)
                if zone == "front":
                    uv_data[li].uv = (float(uv_front[vi, 0]), float(uv_front[vi, 1]))
                elif zone == "side":
                    if is_right_side:
                        uv_data[li].uv = (float(uv_side_mirror[vi, 0]),
                                           float(uv_side_mirror[vi, 1]))
                    else:
                        uv_data[li].uv = (float(uv_side[vi, 0]), float(uv_side[vi, 1]))
                elif zone == "top":
                    uv_data[li].uv = top_skin_uv
                else:  # back
                    uv_data[li].uv = back_skin_uv

            # ─── vc_b assignment ──────────────────────────────────────────────
            # top/back: skin-pinned UV is the same in R and G channels, so vc_b
            # value doesn't change the visual.  Set 0 for consistency.
            # front/side in HARD mode: binary 0 / 1 to match the position seam.
            # front/side in SOFT mode: per-loop smoothstep based on the loop's
            # own vertex world X — this gives a smooth gradient INSIDE polys
            # along the cheek instead of a hard poly-by-poly flip.
            if zone in ("top", "back"):
                vc_b_val = 0.0
            elif blend_mode == "soft":
                v_world_x = (mw @ me.vertices[vi].co).x
                vc_b_val = _soft_vc_b(v_world_x)
            else:
                vc_b_val = 1.0 if zone == "side" else 0.0

            vc_data[li].color = (0.0, 0.0, vc_b_val, 1.0)

    me.update()

    # ─── Bake brow / lips masks into R / G of FacePlate_Zone ─────────────────
    # Unreal reads one vertex-color attribute; pack the feature masks into the
    # spare R and G channels here so the engine material can drive eyebrow tint
    # and lip color without needing extra UV-space lookups.
    try:
        from scipy.spatial import ConvexHull
        def _uv_hull(indices):
            pts = np.array([(lms_front[i][0], 1.0 - lms_front[i][1]) for i in indices])
            return pts[ConvexHull(pts).vertices]
        def _point_in_polygon(pt, polygon):
            x, y = pt
            n_pts = len(polygon); inside = False; j = n_pts - 1
            for i in range(n_pts):
                xi, yi = polygon[i]; xj, yj = polygon[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
                j = i
            return inside

        left_brow_hull  = _uv_hull(MP_LEFT_BROW)
        right_brow_hull = _uv_hull(MP_RIGHT_BROW)
        lips_hull       = _uv_hull(MP_OUTER_LIPS)
        feature_counts = {"brow": 0, "lips": 0}
        for poly in me.polygons:
            li0 = poly.loop_indices[0]
            existing_b = vc_data[li0].color[2]
            u_sum = v_sum = 0.0
            for li in poly.loop_indices:
                u, v = uv_layer.data[li].uv
                u_sum += u; v_sum += v
            u_ctr = u_sum / len(poly.loop_indices)
            v_ctr = v_sum / len(poly.loop_indices)
            is_corner = abs(u_ctr - 0.99) < 0.01 and abs(v_ctr - 0.99) < 0.01
            # Only front-projected polys (B=0, not corner) can carry brow/lip masks
            if existing_b >= 0.5 or is_corner:
                continue
            pt = (u_ctr, v_ctr)
            r = g = 0.0
            if _point_in_polygon(pt, lips_hull):
                r = 1.0; feature_counts["lips"] += 1
            elif _point_in_polygon(pt, left_brow_hull) or _point_in_polygon(pt, right_brow_hull):
                g = 1.0; feature_counts["brow"] += 1
            else:
                continue
            for li in poly.loop_indices:
                vc_data[li].color = (r, g, existing_b, 1.0)
        print(f"  Feature masks: {feature_counts}")
    except ImportError:
        print("  [warn] scipy not available — skipping feature mask bake")

    # Compute natural UV extents per zone — useful for diagnosing alignment.
    uv_extents = {"front": [[],[]], "side": [[],[]]}
    for poly in me.polygons:
        n = (mw3 @ poly.normal).normalized()
        ctr_y = sum((mw @ me.vertices[vi].co).y for vi in poly.vertices) / len(poly.vertices)
        ctr_z = sum((mw @ me.vertices[vi].co).z for vi in poly.vertices) / len(poly.vertices)
        if n.z > n_top_threshold and ctr_z > z_top_threshold:
            continue   # top zone
        elif ctr_y > back_seam_y:
            continue   # back zone
        elif ctr_y > front_seam_y:
            zname = "side"
        else:
            zname = "front"
        for li in poly.loop_indices:
            u, v = uv_layer.data[li].uv
            uv_extents[zname][0].append(u)
            uv_extents[zname][1].append(v)

    uv_bounds = {}
    for zname, (us, vs) in uv_extents.items():
        if us:
            uv_bounds[zname] = {
                "u": [round(min(us),4), round(max(us),4)],
                "v": [round(min(vs),4), round(max(vs),4)],
            }

    total = sum(counts.values())
    return {
        "front_ctrl_pts":  len(ctrl_xyz_f),
        "side_ctrl_pts":   len(ctrl_xyz_s),
        "zone_counts":     counts,
        "zone_pcts": {k: round(v / total * 100, 1) for k, v in counts.items()},
        "uv_bounds":       uv_bounds,
        "uv_layer":        uv_name,
        "side_angle_deg":  side_angle_deg,
        "z_top_threshold": round(z_top_threshold, 4),
    }


def cleanup_interior_faces(obj_name: str, n_dirs: int = 128) -> dict:
    """
    Delete interior faces (anything not visible from outside the bounding sphere).

    For each face center, cast rays from N_DIRS Fibonacci-distributed outside
    points TOWARD the center.  A face is exterior if it is the first hit from
    at least one direction.  Anything never reached is interior.

    Returns counts.  Operates in-place on the mesh.
    """
    import bpy, math as _math
    from mathutils import Vector

    obj = bpy.data.objects[obj_name]
    me  = obj.data
    mw  = obj.matrix_world
    mw_inv = mw.inverted()

    # Ensure the object is in the active view layer and selectable
    if obj.name not in bpy.context.view_layer.objects:
        bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    world_verts = [mw @ v.co for v in me.vertices]
    bb_min = Vector([min(v[i] for v in world_verts) for i in range(3)])
    bb_max = Vector([max(v[i] for v in world_verts) for i in range(3)])
    radius = max((bb_max - bb_min).length, 1.0) * 3.0

    # Fibonacci-sphere directions
    sample_dirs = []
    golden = _math.pi * (3 - _math.sqrt(5))
    for i in range(n_dirs):
        y = 1 - (i / float(n_dirs - 1)) * 2
        r = _math.sqrt(1 - y * y)
        theta = golden * i
        sample_dirs.append(Vector((_math.cos(theta) * r, y, _math.sin(theta) * r)))

    # Poly centers in world space
    poly_centers = []
    for poly in me.polygons:
        ctr = Vector((0, 0, 0))
        for vi in poly.vertices:
            ctr = ctr + (mw @ me.vertices[vi].co)
        poly_centers.append(ctr / len(poly.vertices))

    exterior = set()
    for dir_vec in sample_dirs:
        for idx, ctr in enumerate(poly_centers):
            if idx in exterior:
                continue
            origin_w = ctr + dir_vec * radius
            ray_dir_w = -dir_vec
            origin_l = mw_inv @ origin_w
            dir_l    = (mw_inv.to_3x3() @ ray_dir_w).normalized()
            hit, _loc, _n, face_idx = obj.ray_cast(origin_l, dir_l)
            if hit and face_idx == idx:
                exterior.add(idx)

    # Select interior polys and delete
    for poly in me.polygons:
        poly.select = (poly.index not in exterior)

    polys_before = len(me.polygons)

    # Find a 3D viewport to use as context for mode_set / delete
    # Find a 3D viewport for the operator context (bpy.context.screen may be
    # None after a fresh open_mainfile call)
    override = {"active_object": obj, "object": obj,
                "selected_objects": [obj],
                "selected_editable_objects": [obj]}
    window = bpy.context.window
    if window is None and bpy.context.window_manager.windows:
        window = bpy.context.window_manager.windows[0]
    if window:
        screen = window.screen
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        override["window"] = window
                        override["screen"] = screen
                        override["area"] = area
                        override["region"] = region
                        break
                break

    with bpy.context.temp_override(**override):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.delete(type='FACE')
        bpy.ops.object.mode_set(mode='OBJECT')

    return {
        "polys_before":  polys_before,
        "polys_after":   len(me.polygons),
        "removed":       polys_before - len(me.polygons),
        "sample_dirs":   n_dirs,
    }


def beautify_faces(obj_name:        str,
                    angle_limit_deg: float = 180.0,
                    exclude_skin_pin: bool = True,
                    skin_uv:         tuple = (0.99, 0.99),
                    uv_name:         str   = "FacePlate",
                    ) -> dict:
    """
    Run Blender's `beautify_fill` on the mesh to flip triangle diagonals toward
    equilateral. Makes the topology more uniform — better visual consistency
    on the bary line bake and on lit shading.

    Run AFTER `decimate_back_zone` and BEFORE `bake_edge_mask` so the bake sees
    the final uniform triangulation.

    Args:
        angle_limit_deg: max angle (deg) between adjacent face normals for a
            diagonal to be a candidate for flipping. 180 = unlimited.
        exclude_skin_pin: if True, do NOT beautify polys whose UV is at the
            corner skin patch (those polys were dissolved into planar groups
            and beautifying them would re-fragment those plates).
        skin_uv: the corner-skin UV used by the back/top/inside zones.
        uv_name: name of the UV map to read for the skin-pin check.

    Returns counts.
    """
    import bpy, math
    obj = bpy.data.objects[obj_name]
    me  = obj.data

    if obj.name not in bpy.context.view_layer.objects:
        bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Select all polys EXCEPT skin-pin if requested
    n_selected = 0
    if exclude_skin_pin and uv_name in me.uv_layers:
        uv_layer = me.uv_layers[uv_name]
        cu, cv = skin_uv
        for poly in me.polygons:
            li0 = poly.loop_indices[0]
            u, v = uv_layer.data[li0].uv
            is_corner = abs(u - cu) < 0.001 and abs(v - cv) < 0.001
            poly.select = not is_corner
            if not is_corner:
                n_selected += 1
    else:
        for poly in me.polygons:
            poly.select = True
            n_selected += 1

    polys_before = len(me.polygons)

    override = {"active_object": obj, "object": obj,
                "selected_objects": [obj],
                "selected_editable_objects": [obj]}
    window = bpy.context.window or (bpy.context.window_manager.windows[0]
             if bpy.context.window_manager.windows else None)
    if window:
        screen = window.screen
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        override.update(window=window, screen=screen,
                                        area=area, region=region)
                        break
                break

    with bpy.context.temp_override(**override):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        # beautify_fill operates only on triangles — flips diagonals of pairs
        # of triangles sharing an edge toward more equilateral shape.
        bpy.ops.mesh.beautify_fill(angle_limit=math.radians(angle_limit_deg))
        bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Beautified {n_selected} / {polys_before} faces "
          f"(skin-pin excluded={exclude_skin_pin}, angle≤{angle_limit_deg}°)")
    return {
        "selected_for_beautify": n_selected,
        "polys_before":          polys_before,
        "polys_after":           len(me.polygons),
        "angle_limit_deg":       angle_limit_deg,
        "skin_pin_excluded":     exclude_skin_pin,
    }


def bake_edge_mask(obj_name:           str,
                    sharp_angle_deg:    float = 12.0,
                    uv_name_a:          str   = "EdgeMask_A",
                    uv_name_b:          str   = "EdgeMask_B",
                    ensure_triangulated: bool = True,
                    include_uv_seams:   bool  = False,
                    include_boundary:   bool  = True,
                    min_face_area:      float = 0.0,
                    max_face_area:      float = 0.0,
                    exclude_zones:      tuple = (),
                    zone_vcol_name:     str   = "FacePlate_Zone",
                    zone_b_threshold:   float = 0.5,
                    manual_sharp_flags: str   = "augment",
                    suppress_with_seam: bool  = True,
                    ) -> dict:
    """
    Bake per-corner edge-distance values into two UV channels (UV2 + UV3) so
    Unreal can draw clean anti-aliased lines ONLY along the boundaries between
    big planar regions — not on every interior triangulation edge.

    Run AFTER `decimate_back_zone` (which creates the planar regions) and
    BEFORE `fit_canonical_eyes` (eyes are separate objects, unaffected).

    Algorithm:
      1. Triangulate the mesh deterministically (so the bake matches what Unreal
         sees post-import — no FBX-side re-triangulation surprises).
      2. For each edge, mark as "sharp" if the angle between its two adjacent
         face normals exceeds `sharp_angle_deg`. Boundary edges (1 face) are
         optionally treated as sharp.
      3. For each triangle, write per-corner UV values that encode a "masked
         barycentric": the three components are distances from the three edges
         of the triangle, but non-sharp edges are pushed to 1.0 so they never
         become the minimum.

    Encoding per triangle (V0, V1, V2) and edges e12 (opp V0), e02 (opp V1),
    e01 (opp V2):
        At V0:
          UV_A.x = 1.0                                # weight of V0 / dist from e12
          UV_A.y = 0.0 if e02 sharp else 1.0          # weight of V1 / dist from e02
          UV_B.x = 0.0 if e01 sharp else 1.0          # weight of V2 / dist from e01
        At V1:
          UV_A.x = 0.0 if e12 sharp else 1.0
          UV_A.y = 1.0
          UV_B.x = 0.0 if e01 sharp else 1.0
        At V2:
          UV_A.x = 0.0 if e12 sharp else 1.0
          UV_A.y = 0.0 if e02 sharp else 1.0
          UV_B.x = 1.0

    In the Unreal material:
        a, b   = TexCoord[2].xy
        c      = TexCoord[3].x
        edge_d = min(a, min(b, c))
        aa     = abs(DDX(edge_d)) + abs(DDY(edge_d))     # screen-space width
        line_t = 1 - smoothstep(0, aa * thickness, edge_d)
        final  = Lerp(base_color, LineColor, line_t)

    Args:
        sharp_angle_deg: angle threshold (degrees) between face normals for an
            edge to be considered "sharp" (drawn). 10–15° is typical after
            Limited Dissolve at 15°.
        uv_name_a: name for the first edge-mask UV channel (becomes UV2 in UE).
        uv_name_b: name for the second edge-mask UV channel (becomes UV3).
        ensure_triangulated: triangulate in Blender first so the bake is
            deterministic. Set False only if the mesh is already triangulated.
        include_uv_seams: also mark UV seam edges as sharp (catches projection
            boundaries that aren't crease boundaries).
        include_boundary: treat edges with only one adjacent face as sharp.
        min_face_area: drop sharp marks where EITHER adjacent face area is
            below this threshold (world units²). Kills high-frequency feature
            edges (nose tip, lip pucker) while keeping big planar boundaries.
            0 disables the filter.
        max_face_area: drop sharp marks where BOTH adjacent face areas are
            above this threshold. Useful if you only want lines on small
            facets. 0 disables.
        exclude_zones: tuple of zone names {"front","side","top","back"} —
            edges where BOTH adjacent face centers are in an excluded zone
            won't be drawn. Reads zone from the FacePlate_Zone vertex color
            B channel: B<threshold → "front", B>=threshold → "side". Use
            exclude_zones=("front",) to keep all the facial detail edges
            from drawing and only show lines on cheek/temple/back/jaw.
        zone_vcol_name: vertex color attribute carrying zone info.
        zone_b_threshold: vc_b value above which a poly is considered "side"
            (vs "front"). Matches the soft-blend midpoint.

    Returns counts and the names of the new UV channels.
    """
    import bpy, bmesh, math

    obj = bpy.data.objects[obj_name]
    me  = obj.data

    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    sharp_rad = math.radians(sharp_angle_deg)

    # ─── Step 1: triangulate deterministically ────────────────────────────────
    if ensure_triangulated:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces[:],
                              quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.to_mesh(me)
        bm.free()
        me.update()
        print(f"  Triangulated for deterministic edge bake.")

    # ─── Step 2: identify sharp edges ─────────────────────────────────────────
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Pre-fetch per-poly zone (for exclude_zones filter) once per face.
    # Zones derived from mesh geometry the same way calibrate_full does:
    #   - top  : n.z > 0.45 AND ctr_z > top fraction of head height
    #   - back : ctr_y > ear_y + 0.04 (behind the ears)
    #   - side : vc_b >= zone_b_threshold (side-leaning per dual-UV gradient)
    #   - front: everything else (the visible face / centerline polys)
    # The bake reads these directly from geometry; no extra attribute needed.
    poly_zone = {}    # face_index → "front" | "side" | "top" | "back"
    if exclude_zones:
        from mathutils import Vector as _V
        mw  = obj.matrix_world
        mw3 = mw.to_3x3()
        world_verts = [mw @ v.co for v in me.vertices]
        # Ear y line: average of leftmost + rightmost X verts (matches
        # calibrate_full's heuristic).
        rightmost = max(me.vertices, key=lambda v: (mw @ v.co).x)
        leftmost  = min(me.vertices, key=lambda v: (mw @ v.co).x)
        ear_y     = 0.5 * ((mw @ rightmost.co).y + (mw @ leftmost.co).y)
        back_y    = ear_y + 0.04
        # Top z threshold: 70% of head height from chin to crown.
        z_min = min(v.z for v in world_verts)
        z_max = max(v.z for v in world_verts)
        z_top = z_min + (z_max - z_min) * 0.70
        n_top_threshold = 0.45

        vc = (me.color_attributes[zone_vcol_name].data
              if zone_vcol_name in me.color_attributes else None)

        for poly in me.polygons:
            n = (mw3 @ poly.normal).normalized()
            ctr_y = sum((mw @ me.vertices[vi].co).y for vi in poly.vertices) / len(poly.vertices)
            ctr_z = sum((mw @ me.vertices[vi].co).z for vi in poly.vertices) / len(poly.vertices)
            if n.z > n_top_threshold and ctr_z > z_top:
                zone = "top"
            elif ctr_y > back_y:
                zone = "back"
            elif vc is not None and vc[poly.loop_indices[0]].color[2] >= zone_b_threshold:
                zone = "side"
            else:
                zone = "front"
            poly_zone[poly.index] = zone
        print(f"  Zone counts for filter: "
              f"front={sum(1 for z in poly_zone.values() if z=='front')} "
              f"side={sum(1 for z in poly_zone.values() if z=='side')} "
              f"top={sum(1 for z in poly_zone.values() if z=='top')} "
              f"back={sum(1 for z in poly_zone.values() if z=='back')}")

    sharp_edges = set()   # frozenset({v_idx_a, v_idx_b})
    n_boundary    = 0
    n_crease      = 0
    n_seam        = 0
    n_filt_area   = 0
    n_filt_zone   = 0
    n_manual_add  = 0
    n_manual_supp = 0
    n_manual_kept = 0
    for edge in bm.edges:
        # Always grab faces up-front (may be None on boundary)
        f1 = edge.link_faces[0] if len(edge.link_faces) >= 1 else None
        f2 = edge.link_faces[1] if len(edge.link_faces) >= 2 else None

        # ── auto-detection (angle, boundary, UV seam) ────────────────────────
        is_sharp = False
        if f2 is None:
            if include_boundary and f1 is not None:
                is_sharp = True
                n_boundary += 1
        else:
            try:
                angle = f1.normal.angle(f2.normal, 0.0)
            except ValueError:
                angle = 0.0
            if angle > sharp_rad:
                is_sharp = True
                n_crease += 1
        if include_uv_seams and edge.seam:
            is_sharp = True
            n_seam += 1

        # ── manual sharp / seam tracking ─────────────────────────────────────
        # `edge.smooth = False` ↔ Mark Sharp. `edge.seam = True` ↔ Mark Seam.
        is_manual_sharp = (not edge.smooth)
        is_manual_seam  = bool(edge.seam)

        # ── apply manual sharp BEFORE filters (unconditional force-draw) ─────
        # "only":   ignore auto-detection; only manually-marked edges draw.
        # "augment" (default): manual sharp ADDS to auto-detection.
        # "ignore": pure auto-detection.
        if manual_sharp_flags == "only":
            is_sharp = is_manual_sharp
            if is_manual_sharp:
                n_manual_add += 1
        elif manual_sharp_flags == "augment" and is_manual_sharp and not is_sharp:
            is_sharp = True
            n_manual_add += 1

        # ── filters (area + zone) — SKIPPED for manually-marked sharp edges ──
        # If the artist explicitly marked it sharp, respect that. Filters only
        # apply to auto-detected edges.
        manual_override = (manual_sharp_flags in ("only", "augment")
                           and is_manual_sharp)

        if is_sharp and not manual_override and (min_face_area > 0 or max_face_area > 0):
            a1 = f1.calc_area() if f1 else 0.0
            a2 = f2.calc_area() if f2 else a1
            if min_face_area > 0 and (a1 < min_face_area or a2 < min_face_area):
                is_sharp = False
                n_filt_area += 1
            if is_sharp and max_face_area > 0 \
                       and a1 > max_face_area and a2 > max_face_area:
                is_sharp = False
                n_filt_area += 1

        if is_sharp and not manual_override and exclude_zones and poly_zone:
            z1 = poly_zone.get(f1.index) if f1 else None
            z2 = poly_zone.get(f2.index) if f2 else z1
            if z1 in exclude_zones and (z2 is None or z2 in exclude_zones):
                is_sharp = False
                n_filt_zone += 1

        if manual_override and is_sharp:
            n_manual_kept += 1

        # ── manual SEAM = "do not draw" override (FINAL pass, beats everything)
        # If you really need to remove a specific line, this kills it last.
        if suppress_with_seam and is_manual_seam and is_sharp:
            is_sharp = False
            n_manual_supp += 1

        if is_sharp:
            sharp_edges.add(frozenset(v.index for v in edge.verts))
    bm.free()

    # ─── Step 3: create / replace the two edge-mask UV maps ──────────────────
    # IMPORTANT: keep the existing FacePlate and FacePlate_Side UV maps as
    # UV0 / UV1; append the new ones AFTER so they become UV2 / UV3 in FBX.
    for nm in (uv_name_a, uv_name_b):
        if nm in me.uv_layers:
            me.uv_layers.remove(me.uv_layers[nm])
    uv_a = me.uv_layers.new(name=uv_name_a)
    uv_b = me.uv_layers.new(name=uv_name_b)
    uv_a_data = uv_a.data
    uv_b_data = uv_b.data

    # Note: keep the active UV map as the primary FacePlate (used in the viewport
    # preview). The bake only writes data; rendering with the existing material
    # is unaffected.

    # ─── Step 4: per-corner bake ──────────────────────────────────────────────
    n_tri        = 0
    edge_sharp   = 0
    edge_smooth  = 0
    for poly in me.polygons:
        if len(poly.loop_indices) != 3:
            # Should not happen after triangulate; skip defensively
            continue
        n_tri += 1
        l0, l1, l2 = poly.loop_indices
        v0 = me.loops[l0].vertex_index
        v1 = me.loops[l1].vertex_index
        v2 = me.loops[l2].vertex_index

        e_v1v2_sharp = frozenset([v1, v2]) in sharp_edges  # opposite V0
        e_v0v2_sharp = frozenset([v0, v2]) in sharp_edges  # opposite V1
        e_v0v1_sharp = frozenset([v0, v1]) in sharp_edges  # opposite V2

        edge_sharp  += int(e_v1v2_sharp) + int(e_v0v2_sharp) + int(e_v0v1_sharp)
        edge_smooth += (3 - int(e_v1v2_sharp)
                          - int(e_v0v2_sharp)
                          - int(e_v0v1_sharp))

        # Corner V0: bary = (1, 0, 0) standard. Push y/z to 1 if their edges
        # are smooth (no draw line on that side).
        uv_a_data[l0].uv = (1.0,
                            0.0 if e_v0v2_sharp else 1.0)
        uv_b_data[l0].uv = (0.0 if e_v0v1_sharp else 1.0, 0.0)

        # Corner V1: bary = (0, 1, 0).
        uv_a_data[l1].uv = (0.0 if e_v1v2_sharp else 1.0,
                            1.0)
        uv_b_data[l1].uv = (0.0 if e_v0v1_sharp else 1.0, 0.0)

        # Corner V2: bary = (0, 0, 1).
        uv_a_data[l2].uv = (0.0 if e_v1v2_sharp else 1.0,
                            0.0 if e_v0v2_sharp else 1.0)
        uv_b_data[l2].uv = (1.0, 0.0)

    me.update()

    print(f"  Edge mask bake: {n_tri} triangles, {len(sharp_edges)} unique sharp edges "
          f"({n_crease} crease, {n_boundary} boundary, {n_seam} uv-seam)")
    if n_filt_area or n_filt_zone:
        print(f"  Filtered out: {n_filt_area} by face-area, {n_filt_zone} by zone")
    if n_manual_add or n_manual_supp or n_manual_kept:
        print(f"  Manual overrides: +{n_manual_add} added by sharp flag "
              f"(of which {n_manual_kept} survived filters), "
              f"-{n_manual_supp} suppressed by seam flag")
    print(f"  Per-triangle edge instances: {edge_sharp} sharp / "
          f"{edge_smooth} smooth")
    print(f"  UV map order: {[l.name for l in me.uv_layers]}")

    return {
        "uv_maps_added":           [uv_name_a, uv_name_b],
        "uv_map_order":            [l.name for l in me.uv_layers],
        "triangles":               n_tri,
        "sharp_edges_unique":      len(sharp_edges),
        "sharp_from_crease":       n_crease,
        "sharp_from_boundary":     n_boundary,
        "sharp_from_uv_seam":      n_seam,
        "filtered_by_face_area":   n_filt_area,
        "filtered_by_zone":        n_filt_zone,
        "edge_instances_sharp":    edge_sharp,
        "edge_instances_smooth":   edge_smooth,
        "sharp_angle_deg":         sharp_angle_deg,
        "min_face_area":           min_face_area,
        "max_face_area":           max_face_area,
        "exclude_zones":           list(exclude_zones),
    }


def decimate_back_zone(obj_name:   str,
                       uv_name:    str  = "FacePlate",
                       angle_deg:  float = 15.0,
                       corner_uv:  tuple = (0.99, 0.99)) -> dict:
    """
    Reproducible hard-body pass on the BACK zone.

    Identifies polys whose UVs are at the corner-skin-UV (the back/top/fill
    zone collapsed there by `calibrate_full`) and applies Limited Dissolve
    at `angle_deg` to merge near-coplanar faces into larger planes.

    Run AFTER `calibrate_full` (UVs must already be set) and AFTER
    `cleanup_interior_faces` (so we're only working on the exterior shell).
    """
    import bpy, math as _math

    obj = bpy.data.objects[obj_name]
    me  = obj.data

    if obj.name not in bpy.context.view_layer.objects:
        bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    uv_layer = me.uv_layers[uv_name]
    cu, cv = corner_uv

    # Select polys whose first loop UV is at the corner
    selected = 0
    for poly in me.polygons:
        li0 = poly.loop_indices[0]
        u, v = uv_layer.data[li0].uv
        if abs(u - cu) < 0.001 and abs(v - cv) < 0.001:
            poly.select = True
            selected += 1
        else:
            poly.select = False

    polys_before = len(me.polygons)

    # Find a 3D viewport for the operator context (bpy.context.screen may be
    # None after a fresh open_mainfile call)
    override = {"active_object": obj, "object": obj,
                "selected_objects": [obj],
                "selected_editable_objects": [obj]}
    window = bpy.context.window
    if window is None and bpy.context.window_manager.windows:
        window = bpy.context.window_manager.windows[0]
    if window:
        screen = window.screen
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        override["window"] = window
                        override["screen"] = screen
                        override["area"] = area
                        override["region"] = region
                        break
                break

    with bpy.context.temp_override(**override):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.dissolve_limited(
            angle_limit=_math.radians(angle_deg),
            use_dissolve_boundaries=False,
            delimit={'NORMAL'},
        )
        bpy.ops.mesh.faces_shade_flat()
        bpy.ops.object.mode_set(mode='OBJECT')

    return {
        "selected_for_dissolve": selected,
        "polys_before":          polys_before,
        "polys_after":           len(me.polygons),
        "removed":               polys_before - len(me.polygons),
        "angle_deg":             angle_deg,
    }


def calibrate_from_json(
    obj_name:        str,
    front_json:      str,
    side_json:       str,
    uv_name:         str   = "FacePlate",
    has_right_img:   bool  = False,
    side_angle_deg:  float = SIDE_ANGLE_DEG,
) -> dict:
    """
    Build or replace the FacePlate UV layer using pre-computed MediaPipe JSON files
    written by the ComfyUI MediaPipeFaceExport node.

    front_json : path to <char>_mp_front.json
    side_json  : path to <char>_mp_side_left.json
    has_right_img : reserved — always False for crowd chars (qright mirrors qleft)

    Returns the same stats dict as calibrate().

    Usage (Blender MCP / Text Editor):
        import calibrate_faceplate_uv, importlib
        importlib.reload(calibrate_faceplate_uv)
        result = calibrate_faceplate_uv.calibrate_from_json(
            "ernest_step1",
            front_json = r"B:\\Brains\\Characters\\ernest_chavez\\images\\mp\\ernest_chavez_head_v3SR_mp_front.json",
            side_json  = r"B:\\Brains\\Characters\\ernest_chavez\\images\\mp\\ernest_chavez_head_v3SR_mp_side_left.json",
        )
        print(result)
    """
    import bpy
    from mathutils import Vector

    obj = bpy.data.objects[obj_name]
    me  = obj.data
    mw  = obj.matrix_world
    mw3 = mw.to_3x3()

    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    world_verts = [mw @ v.co for v in me.vertices]
    bb_min = Vector([min(v[i] for v in world_verts) for i in range(3)])
    bb_max = Vector([max(v[i] for v in world_verts) for i in range(3)])
    bb_ctr = (bb_min + bb_max) * 0.5
    head_h = bb_max.z - bb_min.z
    scale  = head_h / FILL

    print(f"Loading front landmarks from {front_json} …")
    lms_front, _front_angle = _load_landmarks_json(front_json)
    print(f"  Front: {len(lms_front)} landmarks")

    print(f"Loading side landmarks from {side_json} …")
    lms_side, side_angle = _load_landmarks_json(side_json)
    print(f"  Side : {len(lms_side)} landmarks  angle={side_angle}")

    if not lms_front:
        raise RuntimeError(f"No landmarks in front JSON (landmark_count=0): {front_json}")
    if not lms_side:
        raise RuntimeError(f"No landmarks in side JSON (landmark_count=0): {side_json}")

    # Align the camera center to the actual face position in the image.
    # The companion image may not be centered at v=0.5; using the raw bounding-box
    # center projects lower-face landmarks below the mesh and produces a shifted RBF.
    print("Aligning camera center to face …")
    bb_ctr_front = _face_aligned_center(lms_front, bb_ctr, scale, me, mw, mw3)

    # side_right images have nose on the right side of the image (u≈0.7+).
    # The side camera 'right' direction points toward the nose, so no x-flip needed.
    # (A flip was previously applied here but produced a mirrored/reversed projection.)
    if side_angle == "side_right":
        print(f"  side_right: nose expected on right side of image — no flip applied")
    bb_ctr_side = _face_aligned_center(lms_side, bb_ctr, scale, me, mw, mw3)

    print("Ray-casting front landmarks onto mesh …")
    ctrl_xyz_front, ctrl_uv_front = _landmarks_to_control_points(
        lms_front, scale, bb_ctr_front, obj, _front_ray_from_photo_uv)
    print(f"  Front hits: {len(ctrl_xyz_front)} / {len(lms_front)}")

    print("Ray-casting side landmarks onto mesh …")
    ctrl_xyz_side, ctrl_uv_side = _landmarks_to_control_points(
        lms_side, scale, bb_ctr_side, obj, _side_ray_from_photo_uv)
    print(f"  Side  hits: {len(ctrl_xyz_side)} / {len(lms_side)}")

    if len(ctrl_xyz_front) < 10:
        raise RuntimeError("Too few front landmark hits — check FILL / framing")
    if len(ctrl_xyz_side) < 10:
        raise RuntimeError("Too few side landmark hits — check FILL / framing")

    all_xyz = np.array([[*(mw @ v.co),] for v in me.vertices], dtype=np.float32)

    print("Building front RBF …")
    uv_front = _rbf_uv(ctrl_xyz_front, ctrl_uv_front, all_xyz)

    print("Building side RBF …")
    uv_side  = _rbf_uv(ctrl_xyz_side,  ctrl_uv_side,  all_xyz)

    if uv_name in me.uv_layers:
        me.uv_layers.remove(me.uv_layers[uv_name])
    uv_layer = me.uv_layers.new(name=uv_name)
    me.uv_layers.active = uv_layer
    uv_data  = uv_layer.data

    zone_vcol_name = "FacePlate_Zone"
    if zone_vcol_name not in me.color_attributes:
        me.color_attributes.new(name=zone_vcol_name, type='FLOAT_COLOR', domain='CORNER')
    vc_data = me.color_attributes[zone_vcol_name].data

    loops  = me.loops
    counts = {"front": 0, "qleft": 0, "qright": 0, "back": 0}

    for poly in me.polygons:
        n_world = (mw3 @ poly.normal).normalized()
        zone    = _zone_for_poly(n_world)
        counts[zone] += 1

        zone_b = 1.0 if zone in ("qleft", "qright") else 0.0

        for li in poly.loop_indices:
            vi = loops[li].vertex_index

            if zone == "back":
                uv_data[li].uv = (0.5, 0.5)
            elif zone == "front":
                uv_data[li].uv = (float(uv_front[vi, 0]), float(uv_front[vi, 1]))
            elif zone == "qleft":
                uv_data[li].uv = (float(uv_side[vi, 0]),  float(uv_side[vi, 1]))
            else:  # qright — same UV as qleft; both sides sample the face region
                uv_data[li].uv = (float(uv_side[vi, 0]),  float(uv_side[vi, 1]))

            vc_data[li].color = (0.0, 0.0, zone_b, 1.0)

    me.update()

    total = sum(counts.values())
    return {
        "scale":          round(scale, 4),
        "head_h":         round(head_h, 4),
        "front_ctrl_pts": len(ctrl_xyz_front),
        "side_ctrl_pts":  len(ctrl_xyz_side),
        "zone_counts":    counts,
        "front_pct":      round(counts["front"]  / total * 100, 1),
        "back_pct":       round(counts["back"]   / total * 100, 1),
        "uv_layer":       uv_name,
    }


# ── standalone test (outside Blender) ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Test MediaPipe landmark detection on an image")
    ap.add_argument("--image", required=True, help="Path to face photo")
    args = ap.parse_args()

    lms = _run_mediapipe(args.image)
    if lms:
        print(f"Detected {len(lms)} landmarks")
        print(f"  Sample: {lms[:5]}")
    else:
        print("No face detected")
