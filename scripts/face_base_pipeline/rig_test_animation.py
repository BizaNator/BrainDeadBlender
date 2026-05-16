"""
rig_test_animation.py

Build a keyframed animation that exercises every part of the face rig --
jaw open/close, eye look up/down/left/right, brow up, smile, frown, blendshape
sliders -- so you can scrub the timeline (Space to play) and visually check
that the rig still works after every pipeline change.

Each pose lives for a few frames; transitions are short. Rest pose is held
at the start and end so the loop reads cleanly.

The action is named `RigTest` so subsequent runs replace it cleanly. The
scene frame range is set to fit the animation.

Designed to drop into the BrainDeadBlender add-on -- run as a sanity check
after `headswap_transfer`, `cleanup_face_weights`, `retarget_armature`, etc.
"""

import math
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

    # Pose magnitudes (radians).
    "jaw_open_x":  math.radians(22),
    "eye_up_x":    math.radians(-18),
    "eye_down_x":  math.radians(18),
    "eye_in_z":    math.radians(20),     # looking inward (cross-eyed)
    "eye_out_z":   math.radians(-20),    # looking outward
    "brow_up_z":   math.radians(0),      # brow bones often need translate, not rotate -- skip
    "smile_lift":  0.005,                # 5mm translate up for lip corners
    "frown_drop": -0.005,                # 5mm translate down

    # Timing (frames). Each pose holds for hold_frames; transitions take
    # transition_frames between poses.
    "fps": 24,
    "transition_frames": 6,
    "hold_frames": 12,
}


# ------------------------------- UTILITIES ----------------------------------
def _ensure_action(arm, name):
    """Get or create an action named `name`, assigned to arm's animation data."""
    if arm.animation_data is None:
        arm.animation_data_create()
    # Replace existing action with same name
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


def _key_pose(arm, frame, bone_pose_dict):
    """Insert keyframes at `frame` for the listed pose bones. `bone_pose_dict`
    maps bone_name -> dict with optional 'euler' (3-tuple radians) and 'loc'
    (3-tuple meters). Bones not listed get their current value keyed
    automatically so the timeline holds them still during transitions."""
    for bone_name, pose in bone_pose_dict.items():
        pb = arm.pose.bones.get(bone_name)
        if pb is None:
            continue
        pb.rotation_mode = 'XYZ'
        if 'euler' in pose:
            pb.rotation_euler = pose['euler']
            pb.keyframe_insert(data_path='rotation_euler', frame=frame)
        if 'loc' in pose:
            pb.location = pose['loc']
            pb.keyframe_insert(data_path='location', frame=frame)


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

    # The poses, in order. Each entry: (label, bone_pose_dict, shape_keys_dict).
    # `shape_keys_dict` is mesh_name -> {key_name: value}.
    poses = [
        ("rest", {}, {}),
        ("jaw_open", {
            "C_jaw": {"euler": (cfg["jaw_open_x"], 0, 0)},
        }, {}),
        ("eyes_up", {
            "L_eye": {"euler": (cfg["eye_up_x"], 0, 0)},
            "R_eye": {"euler": (cfg["eye_up_x"], 0, 0)},
        }, {}),
        ("eyes_down", {
            "L_eye": {"euler": (cfg["eye_down_x"], 0, 0)},
            "R_eye": {"euler": (cfg["eye_down_x"], 0, 0)},
        }, {}),
        ("eyes_left", {
            "L_eye": {"euler": (0, 0, cfg["eye_in_z"])},
            "R_eye": {"euler": (0, 0, cfg["eye_in_z"])},
        }, {}),
        ("eyes_right", {
            "L_eye": {"euler": (0, 0, cfg["eye_out_z"])},
            "R_eye": {"euler": (0, 0, cfg["eye_out_z"])},
        }, {}),
        ("smile", {
            "L_lip_corner": {"loc": (0, 0, cfg["smile_lift"])},
            "R_lip_corner": {"loc": (0, 0, cfg["smile_lift"])},
        }, {}),
        ("frown", {
            "L_lip_corner": {"loc": (0, 0, cfg["frown_drop"])},
            "R_lip_corner": {"loc": (0, 0, cfg["frown_drop"])},
        }, {}),
    ]

    # Add one pose per shape-key target so each gets a turn at 1.0
    sk_targets = cfg["shape_key_targets"]
    for mesh_name, keys in sk_targets.items():
        for key_name in keys:
            poses.append((f"shape_{key_name}", {}, {mesh_name: {key_name: 1.0}}))

    poses.append(("rest_end", {}, {}))

    print(f"  {len(poses)} poses, hold={hold}f transition={trans}f")

    frame = 1
    sk_state = {}  # (mesh_name, key_name) -> last set value -- used to zero out at transitions

    for i, (label, bone_poses, shape_poses) in enumerate(poses):
        # Hold "rest" state at the END of the previous pose's transition
        # (in other words: key the new pose at `frame`, hold until frame+hold,
        # then transition over `trans` frames into the next pose).
        # First reset all-but-this-pose to rest, so non-mentioned bones return.
        _reset_pose(arm)

        # Apply this pose's bone deltas
        for bone_name, pose in bone_poses.items():
            pb = arm.pose.bones.get(bone_name)
            if pb is None:
                continue
            pb.rotation_mode = 'XYZ'
            if 'euler' in pose:
                pb.rotation_euler = pose['euler']
            if 'loc' in pose:
                pb.location = pose['loc']

        # Zero any previously-active shape keys, then activate this pose's
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

        # Key every bone at frame (locks pose) and at frame+hold (still locked)
        _hold_all_at(arm, frame)
        _hold_all_at(arm, frame + hold)

        print(f"  [{frame:4d}-{frame+hold:4d}] {label}")
        frame += step

    # Set scene frame range + fps
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame
    scene.render.fps = cfg["fps"]
    scene.frame_current = 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"\n[done] Action '{cfg['action_name']}' built. Timeline: 1 - {frame}, {cfg['fps']} fps.")
    print(f"  Press SPACE in viewport to play, scrub timeline to step through poses.")
    return action


if __name__ == "__main__":
    rig_test_animation(CONFIG)
