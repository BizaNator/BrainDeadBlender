"""
rig_test_animation.py

Build a keyframed animation that exercises every part of the rig --
face poses (jaw / eyes / smile / brows / shape keys) AND body poses
(spine bend, arm raise, hip rotation, walk pose) -- so you can scrub
the timeline (Space to play) and visually check that the rig still
works after every pipeline change.

Each pose lives for a few frames; transitions are short. Rest pose is
held at the start and end so the loop reads cleanly.

The action is named `RigTest` so subsequent runs replace it cleanly.
The scene frame range is set to fit the animation.

When `playblast=True` (config flag), after the action is built the
script also renders an MP4 of the viewport playing back the action --
no need to sit and watch it live. Output defaults to
`./RigTest_Playblast.mp4` next to the .blend (or absolute path via
`playblast_path`).
"""

import math
import os
import bpy


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "armature": "Fortnite_Armature",
    "action_name": "RigTest",

    # Shape-key targets to exercise (per mesh object). Each entry sets the key
    # value to 1.0 at a frame; the entry order matches the keyframe sequence
    # below. Missing keys are skipped silently.
    "shape_key_targets": {
        "LowPolyHead_Parts": ["HairFit_01", "HairFit_02", "HairFit_03"],
    },

    # Face pose magnitudes (radians / metres).
    "jaw_open_x":  math.radians(22),
    "eye_up_x":    math.radians(-18),
    "eye_down_x":  math.radians(18),
    "eye_in_z":    math.radians(20),     # looking inward (cross-eyed)
    "eye_out_z":   math.radians(-20),    # looking outward
    "brow_up_z":   math.radians(0),      # brow bones often need translate, not rotate -- skip
    "smile_lift":  0.005,                # 5mm translate up for lip corners
    "frown_drop": -0.005,                # 5mm translate down

    # Eye-look driver: Fortnite's L_eye_lid_* bones are children of L_eye, so
    # rotating L_eye drags the eyelid mesh with the eyeball. The skeleton
    # hierarchy is immutable for UEFN compatibility, so we drive eye look via
    # ARKit blendshapes (eyeLookUp/Down/In/Out_L/R + Blink/Squint/Wide) on the
    # merged head instead -- which is also how Unreal LiveLink drives faces.
    # Set False to fall back to bone rotation (debugging only -- looks wrong).
    "use_arkit_for_eyes": True,
    "arkit_eye_target": "LowPolyHead_Rigged",

    # Body poses. Whole-character QA — only land if the armature carries
    # UEFN body bones (spine_01..05, neck_01, head, clavicle_l/r, upperarm_l/r,
    # lowerarm_l/r, thigh_l/r, calf_l/r). Missing bones are skipped silently,
    # so a face-only armature just sees these as no-ops.
    "include_body_poses": True,
    "neck_turn_z":     math.radians(25),
    "head_tilt_y":     math.radians(-15),
    "spine_bend_x":    math.radians(15),   # forward bend, distributed across spine
    "shoulder_lift_y": math.radians(-30),  # clavicle up/out
    "arm_raise_y":     math.radians(-70),  # upperarm rotate forward
    "arm_bend_z":      math.radians(80),   # lowerarm bend at elbow
    "hip_rotate_z":    math.radians(20),
    "leg_step_y":      math.radians(-25),  # thigh forward
    "knee_bend_z":     math.radians(40),

    # Timing (frames). Each pose holds for hold_frames; transitions take
    # transition_frames between poses.
    "fps": 24,
    "transition_frames": 6,
    "hold_frames": 12,

    # --- Playblast / viewport recording --------------------------------------
    # When True, after the action is built, render the timeline to MP4 via
    # `bpy.ops.render.opengl(animation=True)`. This is a viewport playblast --
    # what you see in the viewport is what you get, with whatever shading/light
    # is active. Faster than F12 render; ideal for rig QA.
    "playblast": False,
    # Output path. Relative paths resolve next to the current .blend file.
    # Override per character: e.g. ".\\Renders\\<Name>_RigTest.mp4".
    "playblast_path": "//RigTest_Playblast.mp4",
    "playblast_resolution": (1280, 720),
    "playblast_quality": "HIGH",            # FFmpeg constant rate factor preset
}


# ------------------------------- UTILITIES ----------------------------------
def _ensure_action(arm, name):
    """Get or create an action named `name`, assigned to arm's animation data."""
    if arm.animation_data is None:
        arm.animation_data_create()
    existing = bpy.data.actions.get(name)
    if existing:
        bpy.data.actions.remove(existing, do_unlink=True)
    action = bpy.data.actions.new(name)
    arm.animation_data.action = action
    return action


def _reset_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)


def _key_shape(obj, key_name, value, frame):
    if obj.data.shape_keys is None:
        return False
    kb = obj.data.shape_keys.key_blocks.get(key_name)
    if kb is None:
        return False
    kb.value = value
    kb.keyframe_insert(data_path='value', frame=frame)
    return True


def _hold_all_at(arm, frame):
    """Insert a keyframe at `frame` for every pose bone -- locks the current
    state so subsequent edits don't bleed back through the timeline."""
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
        pb.keyframe_insert(data_path='rotation_euler', frame=frame)
        pb.keyframe_insert(data_path='location', frame=frame)


def _has_any_bones(arm, names):
    return any(arm.pose.bones.get(n) is not None for n in names)


def _build_body_poses(arm, cfg):
    """Return a list of (label, bone_pose_dict, shape_poses) entries for the
    UEFN body bones. Empty list if the armature has no body bones."""
    body_probe = ["spine_01", "spine_03", "upperarm_l", "thigh_l", "neck_01"]
    if not cfg.get("include_body_poses", True) or not _has_any_bones(arm, body_probe):
        return []

    poses = []

    # Spread spine bend across 5 segments so no one joint pops.
    spine_seg = cfg["spine_bend_x"] / 5.0
    spine_bend = {f"spine_0{i}": {"euler": (spine_seg, 0, 0)} for i in range(1, 6)}
    poses.append(("body_spine_forward", spine_bend, {}))

    # Backward bend (slight, half the magnitude).
    spine_back = {f"spine_0{i}": {"euler": (-spine_seg * 0.5, 0, 0)} for i in range(1, 6)}
    poses.append(("body_spine_back", spine_back, {}))

    # Neck turn + head tilt.
    poses.append(("body_neck_turn", {
        "neck_01": {"euler": (0, 0, cfg["neck_turn_z"])},
        "neck_02": {"euler": (0, 0, cfg["neck_turn_z"] * 0.4)},
        "head":    {"euler": (0, cfg["head_tilt_y"], 0)},
    }, {}))

    # Arms up (T-pose -> overhead).
    poses.append(("body_arms_up", {
        "clavicle_l": {"euler": (cfg["shoulder_lift_y"], 0, 0)},
        "clavicle_r": {"euler": (cfg["shoulder_lift_y"], 0, 0)},
        "upperarm_l": {"euler": (0, cfg["arm_raise_y"], 0)},
        "upperarm_r": {"euler": (0, -cfg["arm_raise_y"], 0)},
    }, {}))

    # Arms forward + elbows bent (offering hands).
    poses.append(("body_arms_offer", {
        "upperarm_l": {"euler": (0, cfg["arm_raise_y"] * 0.5, 0)},
        "upperarm_r": {"euler": (0, -cfg["arm_raise_y"] * 0.5, 0)},
        "lowerarm_l": {"euler": (0, 0, cfg["arm_bend_z"])},
        "lowerarm_r": {"euler": (0, 0, -cfg["arm_bend_z"])},
    }, {}))

    # Hip rotate (twist torso).
    poses.append(("body_hip_twist", {
        "pelvis": {"euler": (0, 0, cfg["hip_rotate_z"])},
    }, {}))

    # Walk pose: left leg forward + right arm forward (contralateral step).
    poses.append(("body_walk_step", {
        "thigh_l":    {"euler": (0, cfg["leg_step_y"], 0)},
        "calf_l":     {"euler": (0, 0, cfg["knee_bend_z"])},
        "thigh_r":    {"euler": (0, -cfg["leg_step_y"] * 0.4, 0)},
        "upperarm_l": {"euler": (0, -cfg["arm_raise_y"] * 0.3, 0)},
        "upperarm_r": {"euler": (0, cfg["arm_raise_y"] * 0.4, 0)},
    }, {}))

    print(f"  body bones detected -- adding {len(poses)} body poses")
    return poses


# --------------------------- PLAYBLAST RECORDING ----------------------------
def _record_playblast(scene, cfg):
    """Render the current scene's timeline as a PNG sequence via viewport
    OpenGL render, then stitch it into an MP4 using external `ffmpeg`.

    Blender 5.x ships without FFmpeg integration, so we render frames to
    a temp subdir and call `ffmpeg` ourselves. PNGs are deleted on success
    unless `keep_frames=True`.
    """
    import subprocess, shutil, glob

    out = cfg.get("playblast_path", "//RigTest_Playblast.mp4")
    res = cfg.get("playblast_resolution", (1280, 720))
    keep_frames = cfg.get("playblast_keep_frames", False)

    out_abs = bpy.path.abspath(out)
    out_dir = os.path.dirname(out_abs)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(out_abs))[0]
    frames_dir = os.path.join(out_dir, f"_{base}_frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    r = scene.render
    prev = {
        "filepath": r.filepath,
        "format":   r.image_settings.file_format,
        "res_x":    r.resolution_x,
        "res_y":    r.resolution_y,
        "res_pct":  r.resolution_percentage,
    }
    # Trailing slash + token-free name -> Blender appends 4-digit frame numbers.
    r.filepath = os.path.join(frames_dir, "frame_")
    r.resolution_x = res[0]
    r.resolution_y = res[1]
    r.resolution_percentage = 100
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_mode = 'RGB'

    area = next((a for win in bpy.context.window_manager.windows
                 for a in win.screen.areas if a.type == 'VIEW_3D'), None)
    if area is None:
        print(f"  WARN: no VIEW_3D area open -- skipping playblast")
        return None
    region = next((rg for rg in area.regions if rg.type == 'WINDOW'), None)

    print(f"  playblast: rendering {scene.frame_end} frames -> {frames_dir}")
    try:
        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.render.opengl(animation=True, view_context=True)
    finally:
        r.filepath = prev["filepath"]
        r.image_settings.file_format = prev["format"]
        r.resolution_x = prev["res_x"]
        r.resolution_y = prev["res_y"]
        r.resolution_percentage = prev["res_pct"]

    pngs = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not pngs:
        print(f"  WARN: no frames rendered -- ffmpeg skipped")
        return None
    print(f"  rendered {len(pngs)} frames; stitching with ffmpeg...")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"  WARN: ffmpeg not on PATH -- PNG sequence left at {frames_dir}")
        return frames_dir

    # Match Blender's actual naming (4 digits, starting at scene.frame_start).
    first_frame = scene.frame_start
    pattern = os.path.join(frames_dir, "frame_%04d.png")
    cmd = [ffmpeg, "-y",
           "-framerate", str(scene.render.fps),
           "-start_number", str(first_frame),
           "-i", pattern,
           "-c:v", "libx264",
           "-pix_fmt", "yuv420p",
           "-crf", "18",
           "-preset", "medium",
           # Pad odd dimensions for yuv420p compatibility.
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           out_abs]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg FAILED: {e.stderr[-400:]}")
        return frames_dir

    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"  [playblast done] {out_abs}")
    return out_abs


# ----------------------------- ORCHESTRATOR ---------------------------------
def rig_test_animation(cfg):
    arm = bpy.data.objects.get(cfg["armature"])
    if arm is None or arm.type != 'ARMATURE':
        raise RuntimeError(f"armature '{cfg['armature']}' not found")

    print(f"=== rig_test_animation -> {arm.name} ===")
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

    action = _ensure_action(arm, cfg["action_name"])
    _reset_pose(arm)

    hold = cfg["hold_frames"]
    trans = cfg["transition_frames"]
    step = hold + trans  # frames between consecutive pose centers

    # Face poses (always).
    poses = [
        ("rest", {}, {}),
        ("jaw_open", {
            "C_jaw": {"euler": (cfg["jaw_open_x"], 0, 0)},
        }, {}),
    ]

    # Eye look: in Fortnite the L_eye / R_eye bones rotate the EYEBALLS (separate
    # Eye_L / Eye_R objects, rigid-rotated via Armature modifier). The lid bones
    # are children of the eye bones, so eyelid mesh on the head naturally follows
    # the eye -- that's the in-engine behaviour. ARKit blendshapes layer on top
    # for correctives. Test rig drives BOTH: rotate eye bones for visible
    # eyeball motion + activate the corresponding eye-look ARKit shape so the
    # lid region gets its full ARKit deformation. Disable via cfg["rotate_eye_bones"].
    arkit_target = cfg.get("arkit_eye_target", "LowPolyHead_Rigged")
    rot_eyes = cfg.get("rotate_eye_bones", True)
    use_arkit = cfg.get("use_arkit_for_eyes", True)

    def eye_pose(label, euler_l, euler_r, arkit_keys):
        bones = {}
        if rot_eyes:
            bones["L_eye"] = {"euler": euler_l}
            bones["R_eye"] = {"euler": euler_r}
        shapes = {arkit_target: arkit_keys} if use_arkit and arkit_keys else {}
        return (label, bones, shapes)

    eyes_up_x = cfg["eye_up_x"]; eyes_down_x = cfg["eye_down_x"]
    eyes_in_z = cfg["eye_in_z"]; eyes_out_z = cfg["eye_out_z"]
    poses.extend([
        eye_pose("eyes_up",    (eyes_up_x, 0, 0),    (eyes_up_x, 0, 0),
                 {"eyeLookUpLeft": 1.0, "eyeLookUpRight": 1.0}),
        eye_pose("eyes_down",  (eyes_down_x, 0, 0),  (eyes_down_x, 0, 0),
                 {"eyeLookDownLeft": 1.0, "eyeLookDownRight": 1.0}),
        eye_pose("eyes_left",  (0, 0, eyes_in_z),    (0, 0, eyes_in_z),
                 {"eyeLookInLeft": 1.0, "eyeLookOutRight": 1.0}),
        eye_pose("eyes_right", (0, 0, eyes_out_z),   (0, 0, eyes_out_z),
                 {"eyeLookOutLeft": 1.0, "eyeLookInRight": 1.0}),
    ])
    if use_arkit:
        poses.extend([
            ("eyes_blink",  {}, {arkit_target: {"eyeBlinkLeft":  1.0, "eyeBlinkRight":  1.0}}),
            ("eyes_squint", {}, {arkit_target: {"eyeSquintLeft": 1.0, "eyeSquintRight": 1.0}}),
            ("eyes_wide",   {}, {arkit_target: {"eyeWideLeft":   1.0, "eyeWideRight":   1.0}}),
        ])

    # Brow up/down: rotate the brow bones. Eyebrow mesh sections are weighted
    # to L_brow_mid / R_brow_mid 100%, so the merged head's brow region moves
    # rigidly with the brow bones.
    brow_up_x = cfg.get("brow_up_x", -__import__("math").radians(15))
    brow_down_x = cfg.get("brow_down_x", __import__("math").radians(10))
    poses.extend([
        ("brows_up",   {"L_brow_mid": {"euler": (brow_up_x,   0, 0)},
                        "R_brow_mid": {"euler": (brow_up_x,   0, 0)}}, {}),
        ("brows_down", {"L_brow_mid": {"euler": (brow_down_x, 0, 0)},
                        "R_brow_mid": {"euler": (brow_down_x, 0, 0)}}, {}),
    ])

    poses.extend([
        ("smile", {
            "L_lip_corner": {"loc": (0, 0, cfg["smile_lift"])},
            "R_lip_corner": {"loc": (0, 0, cfg["smile_lift"])},
        }, {}),
        ("frown", {
            "L_lip_corner": {"loc": (0, 0, cfg["frown_drop"])},
            "R_lip_corner": {"loc": (0, 0, cfg["frown_drop"])},
        }, {}),
    ])

    # Body poses (only if the armature has UEFN body bones).
    poses.extend(_build_body_poses(arm, cfg))

    # Shape-key targets (one pose per key at value 1.0).
    sk_targets = cfg["shape_key_targets"]
    for mesh_name, keys in sk_targets.items():
        for key_name in keys:
            poses.append((f"shape_{key_name}", {}, {mesh_name: {key_name: 1.0}}))

    poses.append(("rest_end", {}, {}))

    print(f"  {len(poses)} poses, hold={hold}f transition={trans}f")

    frame = 1
    sk_state = {}  # (mesh_name, key_name) -> last set value

    for i, (label, bone_poses, shape_poses) in enumerate(poses):
        _reset_pose(arm)

        for bone_name, pose in bone_poses.items():
            pb = arm.pose.bones.get(bone_name)
            if pb is None:
                continue
            pb.rotation_mode = 'XYZ'
            if 'euler' in pose:
                pb.rotation_euler = pose['euler']
            if 'loc' in pose:
                pb.location = pose['loc']

        # Zero previously-active shape keys, then activate this pose's.
        for (mn, kn), prev in list(sk_state.items()):
            if mn not in shape_poses or kn not in shape_poses[mn]:
                if prev != 0:
                    obj = bpy.data.objects.get(mn)
                    if obj:
                        _key_shape(obj, kn, 0.0, frame)
                        sk_state[(mn, kn)] = 0
        for mn, keys in shape_poses.items():
            obj = bpy.data.objects.get(mn)
            if obj is None:
                continue
            for kn, v in keys.items():
                if _key_shape(obj, kn, v, frame):
                    sk_state[(mn, kn)] = v

        _hold_all_at(arm, frame)
        _hold_all_at(arm, frame + hold)

        print(f"  [{frame:4d}-{frame+hold:4d}] {label}")
        frame += step

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame
    scene.render.fps = cfg["fps"]
    scene.frame_current = 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"\n[done] Action '{cfg['action_name']}' built. Timeline: 1 - {frame}, {cfg['fps']} fps.")

    if cfg.get("playblast", False):
        _record_playblast(scene, cfg)
    else:
        print(f"  Press SPACE in viewport to play, or set cfg['playblast']=True to record MP4.")

    return action


if __name__ == "__main__":
    rig_test_animation(CONFIG)
