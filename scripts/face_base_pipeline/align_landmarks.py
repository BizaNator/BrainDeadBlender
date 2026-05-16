"""
align_landmarks.py

Use MediaPipe Face Mesh to detect 478 facial landmarks on a source and target
head (rendered from the front), back-project them to 3D, and produce a
TEMPORARY deformed copy of the target head whose anatomy lines up with the
source's. headswap_transfer.py can then BVH-bind against the aligned copy --
giving correctly placed weights even when the source and target have wildly
different proportions (big-forehead lowpoly vs. realistic Penny).

Pipeline
--------
    1. Render target + source heads from a front orthographic camera.
    2. Run MediaPipe Face Landmarker on each render -> 478 normalized 2D
       landmarks per face.
    3. Back-project each landmark via the ortho camera -> a 3D world hit on
       the mesh (closest surface point along the camera ray).
    4. Pair corresponding landmarks (same MP index on both faces) -> control
       point pairs (target_world_3d, source_world_3d).
    5. Build a thin-plate-spline / radial-basis-function deformation that
       maps target landmarks to source landmarks.
    6. Apply the deformation to a COPY of the target mesh (verts only --
       no weights/UVs change). Save as `output_name`.
    7. headswap_transfer.py runs with `dst_head = output_name`, so its
       BVH binding compares the ALIGNED target against the source -- giving
       correctly placed weights -- then transfers them onto the original
       target (which still lives at its real positions).

The aligned mesh is a working file, not a deliverable. The original target
mesh is untouched. After headswap finishes, we re-bind the transferred
weights onto the original target.

Dependencies
------------
  * mediapipe (pip install mediapipe). The face_landmarker.task model file
    must be at MODEL_PATH (downloaded from MP's public storage).
  * scipy.interpolate.RBFInterpolator (pip install scipy).
"""

import os
import sys
import site
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# Ensure --user site-packages (where mediapipe + scipy live for Blender 5.1)
_USER_SITE = site.getusersitepackages()
if _USER_SITE not in sys.path:
    sys.path.insert(0, _USER_SITE)


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "target":      "LowPolyHead_Rigged",
    "source":      "Fortnite_Head_LOD0",
    "output_name": "LowPolyHead_Aligned",   # deformed copy that becomes the headswap dst

    # MediaPipe Face Landmarker model file. Download from:
    # https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    "model_path":  r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\.models\face_landmarker.task",

    # Where to drop the temporary renders (one per mesh).
    "render_dir":  r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\.models",
    "render_size": 1024,

    # Mediapipe landmark indices to use as control points. 478 are available,
    # but a small anatomical set is enough for a coarse RBF and reduces noise.
    # Indices reference MP Face Mesh topology:
    #   1 nose tip, 4 nose root, 10 forehead, 152 chin
    #   33 / 263 eye outer corners (L / R)
    #   133 / 362 eye inner corners (L / R)
    #   61 / 291 mouth corners (L / R)
    #   13 / 14 upper / lower lip mid
    #   468 / 473 iris centers (L / R)
    "landmark_indices": [1, 4, 10, 152,
                         33, 263, 133, 362,
                         61, 291, 13, 14,
                         468, 473],

    # Camera Y-offset behind the mesh (along the front-facing axis). Just
    # has to clear the mesh; ortho camera doesn't care about distance.
    "cam_y_offset": 1.0,

    # RBF smoothness. Higher = softer deformation (averaging across landmarks);
    # lower = stricter (each landmark hits exactly). 0 = exact interpolation.
    "rbf_smoothing": 0.0,
    "rbf_kernel":    "thin_plate_spline",
}


# ------------------------------- UTILITIES ----------------------------------
def _obj(name):
    o = bpy.data.objects.get(name)
    if o is None:
        raise RuntimeError(f"object '{name}' not found")
    return o


def _setup_ortho_camera(target_obj, cam_y_offset, render_size):
    """Create or reuse 'lm_cam' as an orthographic camera framed on target."""
    ws = [target_obj.matrix_world @ v.co for v in target_obj.data.vertices]
    # If a body-attached mesh, filter to head region by Z >1.4
    head_zs = [w.z for w in ws if w.z > 1.4]
    if not head_zs:
        head_zs = [w.z for w in ws]
    head_xs = [w.x for w in ws if w.z > 1.4] or [w.x for w in ws]
    head_ys = [w.y for w in ws if w.z > 1.4] or [w.y for w in ws]
    cx = (min(head_xs) + max(head_xs)) / 2
    cy_min = min(head_ys)
    cz = (min(head_zs) + max(head_zs)) / 2
    size = max(max(head_xs) - min(head_xs),
               max(head_zs) - min(head_zs)) * 1.15

    cam_data = bpy.data.cameras.get("lm_cam") or bpy.data.cameras.new("lm_cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = size
    cam_obj = bpy.data.objects.get("lm_cam") or bpy.data.objects.new("lm_cam", cam_data)
    if cam_obj.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = (cx, cy_min - cam_y_offset, cz)
    cam_obj.rotation_euler = (1.5707963, 0, 0)  # 90deg X -> look +Y
    return cam_obj, size


def _render_only_object(target_obj, cam_obj, image_path, render_size):
    """Render a single object via Workbench. Other meshes get hidden first."""
    scene = bpy.context.scene
    saved = {
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "filepath": scene.render.filepath,
        "engine": scene.render.engine,
        "cam": scene.camera,
    }
    # Hide everything visible that isn't target_obj (or non-mesh siblings).
    # Some objects may not be in the active view layer (e.g. excluded
    # collections) -- skip those silently.
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
            continue  # not in this view layer
        if o == target_obj:
            if state:
                _safe_hide_set(o, False)
                hidden.append((o, True))
            continue
        if not state:
            if _safe_hide_set(o, True):
                hidden.append((o, False))

    scene.camera = cam_obj
    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.filepath = image_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.ops.render.render(write_still=True)

    # Restore
    for o, was_hidden in hidden:
        try: o.hide_set(was_hidden)
        except: pass
    scene.render.resolution_x = saved["res_x"]
    scene.render.resolution_y = saved["res_y"]
    scene.render.filepath = saved["filepath"]
    scene.render.engine = saved["engine"]
    scene.camera = saved["cam"]
    return image_path


def _detect_landmarks(image_path, model_path):
    """Return list of (x_norm, y_norm) tuples for one detected face, or None."""
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (FaceLandmarker,
                                               FaceLandmarkerOptions,
                                               RunningMode)
    from mediapipe.tasks.python import BaseOptions
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    img = mp.Image.create_from_file(image_path)
    with FaceLandmarker.create_from_options(options) as det:
        result = det.detect(img)
    if not result.face_landmarks:
        return None
    return [(lm.x, lm.y) for lm in result.face_landmarks[0]]


def _backproject_ortho(mesh_obj, cam_obj, lm_uv, image_aspect=1.0):
    """Raycast normalized image landmark (u, v) through the ortho camera onto
    mesh_obj; return world hit position or None if miss."""
    u, v = lm_uv
    ortho = cam_obj.data.ortho_scale
    local_offset = Vector(((u - 0.5) * ortho,
                           (0.5 - v) * ortho / image_aspect,
                           0.0))
    rot = cam_obj.matrix_world.to_3x3()
    origin = cam_obj.matrix_world.translation + rot @ local_offset
    direction = (rot @ Vector((0, 0, -1))).normalized()

    mw = mesh_obj.matrix_world
    mw_inv = mw.inverted()
    local_origin = mw_inv @ origin
    local_direction = (mw_inv.to_3x3() @ direction).normalized()
    bvh = BVHTree.FromObject(mesh_obj, bpy.context.evaluated_depsgraph_get())
    hit = bvh.ray_cast(local_origin, local_direction)
    if hit[0] is None:
        return None
    return mw @ hit[0]


def _build_rbf(src_pts, dst_pts, kernel, smoothing):
    """Return a function f(points_array) -> deformed_points_array that maps
    each src landmark to its dst landmark, smoothly extending in between."""
    from scipy.interpolate import RBFInterpolator
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    rbf = RBFInterpolator(src, dst, kernel=kernel, smoothing=smoothing)

    def deform(points):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 1:
            return rbf(arr.reshape(1, -1))[0]
        return rbf(arr)
    return deform


def _detect_and_backproject(mesh_obj, render_dir, render_size, cam_y_offset,
                            model_path, indices, suffix):
    """Run the full render -> mediapipe -> backproject chain for one mesh.
    Returns dict[mp_index] -> Vector(world_3d), and the camera object used."""
    cam, size = _setup_ortho_camera(mesh_obj, cam_y_offset, render_size)
    image_path = os.path.join(render_dir, f"align_{suffix}.png")
    _render_only_object(mesh_obj, cam, image_path, render_size)
    lms = _detect_landmarks(image_path, model_path)
    if lms is None:
        raise RuntimeError(f"MediaPipe found no face on render of {mesh_obj.name}")
    out = {}
    misses = []
    for idx in indices:
        if idx >= len(lms):
            continue
        pos = _backproject_ortho(mesh_obj, cam, lms[idx])
        if pos is None:
            misses.append(idx)
        else:
            out[idx] = pos
    if misses:
        print(f"  WARN: {len(misses)} landmarks missed mesh {mesh_obj.name}: {misses}")
    return out, cam


# --------------------------------- ENTRY ------------------------------------
def align_landmarks(cfg):
    target = _obj(cfg["target"])
    source = _obj(cfg["source"])

    os.makedirs(cfg["render_dir"], exist_ok=True)

    print(f"=== align_landmarks ===")
    print(f"  target: {target.name}")
    print(f"  source: {source.name}")

    src_lms, _ = _detect_and_backproject(
        source, cfg["render_dir"], cfg["render_size"], cfg["cam_y_offset"],
        cfg["model_path"], cfg["landmark_indices"], suffix="src")
    dst_lms, _ = _detect_and_backproject(
        target, cfg["render_dir"], cfg["render_size"], cfg["cam_y_offset"],
        cfg["model_path"], cfg["landmark_indices"], suffix="dst")

    # Pair: only indices that hit BOTH meshes
    common = sorted(set(src_lms.keys()) & set(dst_lms.keys()))
    if len(common) < 4:
        raise RuntimeError(f"only {len(common)} common landmarks; need >= 4 for RBF")
    print(f"\nUsing {len(common)} paired landmarks:")
    for idx in common:
        s = src_lms[idx]; d = dst_lms[idx]
        print(f"  [{idx:3d}] src ({s.x:6.3f},{s.y:6.3f},{s.z:6.3f}) "
              f"<- dst ({d.x:6.3f},{d.y:6.3f},{d.z:6.3f})  "
              f"delta={(s - d).length*100:5.2f}cm")

    src_pts = [tuple(src_lms[i]) for i in common]
    dst_pts = [tuple(dst_lms[i]) for i in common]
    deform = _build_rbf(dst_pts, src_pts, cfg["rbf_kernel"], cfg["rbf_smoothing"])

    # Build the deformed copy
    out_name = cfg["output_name"]
    existing = bpy.data.objects.get(out_name)
    if existing:
        em = existing.data
        bpy.data.objects.remove(existing, do_unlink=True)
        if isinstance(em, bpy.types.Mesh) and em.users == 0:
            bpy.data.meshes.remove(em)

    new_me = target.data.copy()
    new_obj = target.copy()
    new_obj.data = new_me
    new_obj.name = out_name
    new_me.name = out_name + "_mesh"
    for coll in target.users_collection:
        coll.objects.link(new_obj)
    if not new_obj.users_collection:
        bpy.context.scene.collection.objects.link(new_obj)
    # Drop modifiers so binding isn't double-rigged
    for m in list(new_obj.modifiers):
        new_obj.modifiers.remove(m)
    new_obj.parent = None
    new_obj.matrix_parent_inverse.identity()

    # Deform the verts in WORLD space, then bake back to local.
    # FIRST record the original mesh-local positions so we can restore them
    # after headswap binding (so the final rigged geometry keeps the lowpoly's
    # native shape, not Penny's shape).
    mw = new_obj.matrix_world
    mw_inv = mw.inverted()
    orig_local = [v.co.copy() for v in new_me.vertices]
    new_me["__align_landmarks_orig__"] = [(c.x, c.y, c.z) for c in orig_local]

    world_verts = np.array([list(mw @ v.co) for v in new_me.vertices])
    deformed_world = deform(world_verts)
    for i, v in enumerate(new_me.vertices):
        v.co = mw_inv @ Vector(deformed_world[i].tolist())
    if new_me.shape_keys:
        # Save original shape-key positions per key, then deform.
        orig_sk = {}
        for kb in new_me.shape_keys.key_blocks:
            orig_sk[kb.name] = [kb.data[i].co.copy() for i in range(len(kb.data))]
            kb_w = np.array([list(mw @ kb.data[i].co) for i in range(len(kb.data))])
            kb_d = deform(kb_w)
            for i in range(len(kb.data)):
                kb.data[i].co = mw_inv @ Vector(kb_d[i].tolist())
        new_me["__align_landmarks_orig_sk__"] = {
            name: [(c.x, c.y, c.z) for c in coords]
            for name, coords in orig_sk.items()
        }
    new_me.update()

    print(f"\n[done] '{out_name}' created with {len(new_me.vertices)} verts deformed via RBF "
          f"({len(common)} control points). Use as headswap dst_head.")
    print(f"        Original positions saved in custom property '__align_landmarks_orig__'.")
    print(f"        Run restore_geometry() after headswap to swap shape back to the lowpoly.")
    return new_obj


def restore_geometry(rigged_obj_name, source_aligned_obj_name=None):
    """After headswap has run on a landmark-aligned mesh and produced a rigged
    output, swap the rigged output's vert positions back to the original
    pre-deformation positions, so the final geometry has the lowpoly's native
    shape (not Penny's shape).

    Vert indices in the rigged output must match the aligned mesh's indices.
    That requires running headswap_transfer with all mesh-altering steps
    disabled (weld_distance=None, neck_cut_local_z=None, cleanup_mesh=False,
    align=False) -- the binding/weights are still done normally.

    Parameters
    ----------
    rigged_obj_name : str
        Object name of the rigged output (e.g. "LowPolyHead_Rigged").
    source_aligned_obj_name : str, optional
        Object name of the aligned mesh whose `__align_landmarks_orig__`
        property holds the original positions. Defaults to the rigged object
        itself (since headswap's duplicate_head preserves custom properties).
    """
    rigged = bpy.data.objects.get(rigged_obj_name)
    if rigged is None or rigged.type != 'MESH':
        raise RuntimeError(f"rigged object '{rigged_obj_name}' not found")

    src_name = source_aligned_obj_name or rigged_obj_name
    holder = bpy.data.objects.get(src_name)
    if holder is None:
        raise RuntimeError(f"source object '{src_name}' not found")

    orig = holder.data.get("__align_landmarks_orig__")
    if orig is None:
        raise RuntimeError(f"'{src_name}' has no '__align_landmarks_orig__' "
                           "custom property -- did align_landmarks run on it?")

    me = rigged.data
    if len(orig) != len(me.vertices):
        raise RuntimeError(
            f"vert-count mismatch: aligned has {len(orig)}, rigged has "
            f"{len(me.vertices)}. Headswap must run with cleanup disabled.")

    # Apply original positions (orig was stored in aligned-mesh local space;
    # headswap may have re-localized verts but the mesh-local space is the
    # same since align didn't add/remove verts).
    for i, v in enumerate(me.vertices):
        v.co = Vector(tuple(orig[i]))

    orig_sk = holder.data.get("__align_landmarks_orig_sk__")
    if orig_sk and me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            if kb.name in orig_sk:
                for i, co in enumerate(orig_sk[kb.name]):
                    if i < len(kb.data):
                        kb.data[i].co = Vector(tuple(co))

    me.update()
    print(f"[restore_geometry] '{rigged_obj_name}': restored {len(orig)} verts "
          f"to original pre-align positions")
    return rigged


if __name__ == "__main__":
    align_landmarks(CONFIG)
