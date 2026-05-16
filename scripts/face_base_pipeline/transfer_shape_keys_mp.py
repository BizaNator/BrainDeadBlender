"""
transfer_shape_keys_mp.py

MediaPipe-driven, anatomy-aware shape key transfer.

The naive `transfer_shape_keys.py` uses bbox-aligned BVH closest-point
matching, which is anatomically blind: a target vert in the lip area
might inherit a morph delta from a donor vert in the cheek area, because
"closest-point in world space" doesn't equal "same anatomical region".

This script does it properly:
    1. Detect 478 MediaPipe Face Mesh landmarks on DONOR and TARGET via
       front renders (reusing align_landmarks.py helpers).
    2. Back-project each landmark to a 3D point on its respective mesh.
    3. For each landmark, find the nearest MESH VERT on each side, giving
       N landmark-pair correspondences (donor_vert_i, target_vert_i)
       where i indexes the same anatomical landmark on both heads.
    4. For each donor shape key:
         - Compute the WORLD delta at each donor landmark vert.
         - Build a thin-plate-spline / RBF interpolation that maps
           donor-vert delta values onto target landmark verts (1:1 by
           landmark index) and smooths to all other target verts.
         - Set the shape key on target with these interpolated deltas.

Result: a morph that activates the right anatomical region on the
target (lip morphs deform lip verts, brow morphs deform brow verts,
etc.), regardless of mesh topology differences.

Designed to drop into the BrainDeadBlender add-on as a richer
replacement / alternative to transfer_shape_keys.py.
"""

import os
import sys
import site
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

_USER_SITE = site.getusersitepackages()
if _USER_SITE not in sys.path:
    sys.path.insert(0, _USER_SITE)


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "target": "LowPolyHead_Rigged",
    "donor":  "ARKit_Head",

    # MediaPipe model + render tempdir (reuse align_landmarks settings)
    "model_path":  r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\.models\face_landmarker.task",
    "render_dir":  r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\.models",
    "render_size": 1024,
    "cam_y_offset": 1.0,

    # Which donor keys to transfer. Empty = ALL keys except Basis.
    "key_filter": [],
    "key_blacklist_substr": [],

    # RBF smoothing for delta interpolation. 0 = exact (landmark verts get
    # exact donor delta); higher = smoother (deltas average across landmarks).
    "rbf_smoothing": 0.001,
    "rbf_kernel":    "thin_plate_spline",

    # Sub-mm noise floor (world meters); per-vert deltas below this get dropped.
    "delta_epsilon": 1e-5,

    # Overwrite existing key with same name on target (else skip).
    "overwrite": True,

    # If a donor landmark fails to back-project (lands off-mesh), skip it
    # rather than fail. Same for target.
    "skip_missing_landmarks": True,

    # When the target is a sub-mesh (CustomLips, Eyelid_*, Eyebrow_*, etc.)
    # rendering only the target gives MediaPipe an unrecognizable shape and
    # detection fails. With render_with_face_companions=True, the listed
    # companion objects stay visible during the render so MediaPipe sees a
    # full face; back-projection still casts only against the target mesh.
    "render_with_face_companions": True,
    "face_companion_objects": [
        "LowPolyHead_Rigged",
        "Eye_L", "Eye_R", "CustomLips", "Tongue",
        "Eyelid_L_Upper", "Eyelid_L_Lower",
        "Eyelid_R_Upper", "Eyelid_R_Lower",
        "Eyebrow_L", "Eyebrow_R",
        "Nose",
    ],

    # Minimum landmark pairs required to attempt RBF. If a sub-mesh only
    # picks up e.g. 8 lip landmarks, that's enough for a lip-region morph.
    "min_landmark_pairs": 8,
}


# ------------------------------- HELPERS ------------------------------------
def _import_align_landmarks():
    """Load align_landmarks.py once and return its namespace."""
    path = os.path.join(os.path.dirname(os.path.abspath(_align_path())),
                        "align_landmarks.py")
    ns = {}
    exec(compile(open(path).read(), "align_landmarks.py", 'exec'), ns)
    return ns


def _align_path():
    # Sibling to this file
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters"
    return os.path.join(here, "align_landmarks.py")


def _basis_world_coords(obj):
    me = obj.data
    mw = obj.matrix_world
    sk = me.shape_keys
    if sk and sk.key_blocks:
        kb = sk.key_blocks[0]
        if kb.name.lower() == "basis":
            return [mw @ kb.data[i].co for i in range(len(me.vertices))]
    return [mw @ v.co for v in me.vertices]


def _render_face_with_companions(target_obj, companion_names, cam, image_path, render_size):
    """Render `target_obj` plus any companion meshes that exist in the scene
    (head + accessories). Hides everything else. Restores hide state after."""
    scene = bpy.context.scene
    saved = {
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "filepath": scene.render.filepath,
        "engine": scene.render.engine,
        "cam": scene.camera,
    }
    keep_visible = {target_obj.name}
    for n in companion_names:
        if bpy.data.objects.get(n):
            keep_visible.add(n)

    def _safe_hide_get(o):
        try: return o.hide_get()
        except: return None
    def _safe_hide_set(o, val):
        try:
            o.hide_set(val)
            return True
        except:
            return False

    hidden = []
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        state = _safe_hide_get(o)
        if state is None:
            continue
        if o.name in keep_visible:
            if state:
                _safe_hide_set(o, False)
                hidden.append((o, True))
            continue
        if not state:
            if _safe_hide_set(o, True):
                hidden.append((o, False))

    scene.camera = cam
    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.filepath = image_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.ops.render.render(write_still=True)

    for o, was_hidden in hidden:
        _safe_hide_set(o, was_hidden)
    scene.render.resolution_x = saved["res_x"]
    scene.render.resolution_y = saved["res_y"]
    scene.render.filepath = saved["filepath"]
    scene.render.engine = saved["engine"]
    scene.camera = saved["cam"]
    return image_path


def _detect_face_landmarks_3d(obj, ns_align, cfg, frame_obj=None):
    """Render scene with full face visible, run MediaPipe, back-project each
    landmark onto `obj` (the target sub-mesh). Returns list of N world-position
    Vectors (or None for misses). `frame_obj` is used to set the camera bbox
    (defaults to obj itself; pass the full head when target is a tiny sub-mesh)."""
    framing = frame_obj if frame_obj is not None else obj
    cam, ortho = ns_align["_setup_ortho_camera"](framing, cfg["cam_y_offset"], cfg["render_size"])
    render_path = os.path.join(cfg["render_dir"], f"_mp_{obj.name}.png")
    os.makedirs(cfg["render_dir"], exist_ok=True)

    if cfg.get("render_with_face_companions", True):
        _render_face_with_companions(
            obj, cfg.get("face_companion_objects", []),
            cam, render_path, cfg["render_size"])
    else:
        ns_align["_render_only_object"](obj, cam, render_path, cfg["render_size"])

    lms = ns_align["_detect_landmarks"](render_path, cfg["model_path"])
    if lms is None:
        raise RuntimeError(f"MediaPipe failed to detect a face on render of '{obj.name}'")
    print(f"  detected {len(lms)} landmarks (rendered scene around '{obj.name}')")

    # Back-project each ray against ONLY the target obj (so we only get
    # landmarks that anatomically land on this sub-mesh's surface).
    points = []
    misses = 0
    for uv in lms:
        p = ns_align["_backproject_ortho"](obj, cam, uv)
        if p is None:
            misses += 1
            points.append(None)
        else:
            points.append(p)
    print(f"  back-projected {len(points)-misses}/{len(points)} onto '{obj.name}' ({misses} off-mesh)")
    return points


def _nearest_vert_index(obj_world_coords, target_world_pos):
    """Linear nearest-vert lookup. For 478 lookups on ~10k-vert meshes
    this is fine (~5M comparisons)."""
    best_i = -1
    best_d = float('inf')
    for i, c in enumerate(obj_world_coords):
        d = (c - target_world_pos).length_squared
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _rbf_interp_3d(control_pts, control_vals, query_pts, kernel, smoothing):
    """RBF interpolation: control_pts (N x 3) -> control_vals (N x 3),
    interpolated to query_pts (M x 3) -> M x 3 deltas."""
    from scipy.interpolate import RBFInterpolator
    interp = RBFInterpolator(np.asarray(control_pts),
                              np.asarray(control_vals),
                              kernel=kernel, smoothing=smoothing)
    return interp(np.asarray(query_pts))


# --------------------------------- ENTRY ------------------------------------
def transfer_shape_keys_mp(cfg):
    target = bpy.data.objects.get(cfg["target"])
    donor  = bpy.data.objects.get(cfg["donor"])
    if target is None: raise RuntimeError(f"target '{cfg['target']}' not found")
    if donor  is None: raise RuntimeError(f"donor '{cfg['donor']}' not found")
    if donor.data.shape_keys is None:
        print(f"  skip: donor '{donor.name}' has no shape keys")
        return 0

    print(f"=== transfer_shape_keys_mp: {donor.name} -> {target.name} ===")
    ns_align = _import_align_landmarks()

    # 1+2: MediaPipe + back-project on each
    # For sub-mesh targets, frame the camera on the FULL HEAD (or an
    # explicit companion if set) so the rendered image looks like a face
    # to MediaPipe. Back-projection still only hits the sub-mesh.
    print(f"\n  [donor]")
    donor_3d = _detect_face_landmarks_3d(donor, ns_align, cfg)
    print(f"\n  [target]")
    # Pick the largest-bbox face_companion object actually in scene as the
    # framing reference (so the camera frames the full head, not e.g. tiny
    # CustomLips).
    frame_obj = None
    best_vol = 0
    for n in cfg.get("face_companion_objects", []):
        o = bpy.data.objects.get(n)
        if o and o.type == 'MESH' and o.data.vertices:
            ws = [o.matrix_world @ v.co for v in o.data.vertices]
            xs=[p.x for p in ws]; ys=[p.y for p in ws]; zs=[p.z for p in ws]
            vol = (max(xs)-min(xs)) * (max(ys)-min(ys)) * (max(zs)-min(zs))
            if vol > best_vol:
                best_vol = vol; frame_obj = o
    if frame_obj:
        print(f"  framing camera on '{frame_obj.name}' (largest companion bbox)")
    target_3d = _detect_face_landmarks_3d(target, ns_align, cfg, frame_obj=frame_obj)

    # 3: pair landmarks - both lists are 478 long, index i = same landmark
    donor_world = _basis_world_coords(donor)
    target_world = _basis_world_coords(target)
    print(f"\n  finding nearest verts to each landmark...")
    pairs = []  # list of (donor_vert_idx, target_vert_idx, donor_world_pos, target_world_pos)
    for i, (d_pt, t_pt) in enumerate(zip(donor_3d, target_3d)):
        if d_pt is None or t_pt is None:
            continue
        d_vi = _nearest_vert_index(donor_world, d_pt)
        t_vi = _nearest_vert_index(target_world, t_pt)
        pairs.append((d_vi, t_vi, donor_world[d_vi], target_world[t_vi]))
    print(f"  {len(pairs)} usable landmark pairs")

    min_pairs = cfg.get("min_landmark_pairs", 10)
    if len(pairs) < min_pairs:
        raise RuntimeError(f"too few landmarks for RBF interpolation: {len(pairs)} < {min_pairs}")

    # 4: for each donor shape key, compute landmark deltas + RBF-interpolate to all target verts
    donor_sk = donor.data.shape_keys
    donor_basis_kb = donor_sk.key_blocks[0]
    donor_mw = donor.matrix_world

    name_filter = set(cfg.get("key_filter") or [])
    black = [s.lower() for s in cfg.get("key_blacklist_substr", [])]
    candidates = [kb for kb in donor_sk.key_blocks
                  if kb.name != donor_basis_kb.name
                  and (not name_filter or kb.name in name_filter)
                  and not any(s in kb.name.lower() for s in black)]
    print(f"  {len(candidates)} donor keys to transfer")

    # Ensure target Basis
    if target.data.shape_keys is None:
        target.shape_key_add(name="Basis", from_mix=False)
    target_sk = target.data.shape_keys
    target_basis_kb = target_sk.key_blocks[0]
    target_mw_inv = target.matrix_world.inverted()
    overwrite = cfg.get("overwrite", True)
    eps = cfg.get("delta_epsilon", 1e-5)

    # Control points (donor landmark world positions) and target verts (where we want deltas)
    control_pts = [p[3] for p in pairs]  # target-world landmark positions
    # We'll query at ALL target verts in world space
    query_pts = list(target_world)
    target_basis_local = [target_basis_kb.data[i].co.copy() for i in range(len(target_world))]

    n_created = 0
    for kb in candidates:
        if not overwrite and target_sk.key_blocks.get(kb.name):
            print(f"    skip '{kb.name}': exists")
            continue
        # Donor world delta at each landmark donor vert
        control_vals = []
        for (d_vi, t_vi, d_pos, t_pos) in pairs:
            local_delta = kb.data[d_vi].co - donor_basis_kb.data[d_vi].co
            world_delta = donor_mw.to_3x3() @ local_delta
            control_vals.append([world_delta.x, world_delta.y, world_delta.z])
        # RBF interpolate landmark deltas to all target verts
        try:
            target_deltas_world = _rbf_interp_3d(
                control_pts, control_vals, query_pts,
                cfg["rbf_kernel"], cfg["rbf_smoothing"])
        except Exception as e:
            print(f"    skip '{kb.name}': RBF failed -- {e}")
            continue
        # Replace existing
        existing = target_sk.key_blocks.get(kb.name)
        if existing:
            target.shape_key_remove(existing)
        new_kb = target.shape_key_add(name=kb.name, from_mix=False)
        for vi in range(len(target_world)):
            wd = Vector(target_deltas_world[vi])
            if wd.length < eps:
                continue
            local_delta = target_mw_inv.to_3x3() @ wd
            new_kb.data[vi].co = target_basis_local[vi] + local_delta
        n_created += 1

    print(f"\n[done] created {n_created} shape keys via MediaPipe-based RBF transfer")
    return n_created


if __name__ == "__main__":
    transfer_shape_keys_mp(CONFIG)
