"""
verify_rig_poses.py

Render one image per RigTest pose so the full sweep can be eyeballed (or
AI-reviewed) before FBX export. Reads pose ranges from the print log of
rig_test_animation -- duplicates the labels here so the verifier doesn't
depend on action introspection.

Output: <render_dir>/pose_NN_<label>.png  (NN zero-padded so they sort)
Plus a contact-sheet text index `_INDEX.txt` listing pose -> frame -> file.

Camera framing: head-only by default (face poses don't need the body).
The script positions a perspective camera in front of the head at rest
and uses the same camera across all poses so the visual diff between
poses is clean.

Run AFTER rig_test_animation has built the RigTest action. The verifier
just scrubs the timeline and renders -- no scene mutation beyond camera
setup + render path.
"""

import bpy
import os
import math
from mathutils import Vector, Euler


# All poses created by rig_test_animation, in order, with their LABEL.
# Each is (label, hold_start_frame). Hold runs hold_start .. hold_start+hold-1.
# Default hold=12, transition=6. First pose 'rest' starts at frame 1; each
# subsequent pose starts at prev_start + hold + transition = prev_start + 18.
POSE_LABELS = [
    "rest", "jaw_open", "eyes_up", "eyes_down", "eyes_left", "eyes_right",
    "eyes_blink", "eyes_squint", "eyes_wide", "brows_up", "brows_down",
    "smile", "frown", "cheek_puff", "jaw_forward", "jaw_left", "jaw_right",
    "mouth_close", "mouth_pucker", "mouth_funnel", "mouth_left", "mouth_right",
    "mouth_dimple", "mouth_press", "mouth_stretch", "mouth_roll_upper",
    "mouth_roll_lower", "mouth_shrug_upper", "mouth_shrug_lower",
    "mouth_lower_down", "mouth_upper_up", "brow_inner_up", "brow_outer_up",
    "brow_down_arkit", "nose_sneer", "tongue_out",
    "body_spine_forward", "body_spine_back", "body_neck_turn",
    "body_arms_up", "body_arms_offer", "body_hip_twist", "body_walk_step",
    "shape_HairFit_01", "shape_HairFit_02", "shape_HairFit_03",
    "rest_end",
]


CONFIG = {
    "armature":   "Fortnite_Armature",
    "head_obj":   "LowPolyHead_Rigged",
    "action":     "RigTest",
    "hold":       12,
    "transition": 6,

    # Frame within hold-range to render. Default = hold//2 (middle of hold).
    "frame_in_hold": 6,

    # Render output dir. _INDEX.txt is written alongside the images.
    "render_dir": r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters\Head\renders\pose_verify",
    "resolution": (800, 800),

    # Hide these objects during render (donor + clutter meshes).
    "hide_for_render": [
        "ARKit_Head", "Mutable_BaseBody", "Fortnite_Head_LOD0",
        "F_Med_Soldier_01_LOD0", "M_Med_Soldier_04_Preview_LOD0",
        "F_Med_Commando_01_Preview_LOD0", "Maya_BigGirl-154",
        "geometry_0", "Neck", "Nose", "CustomLips_PRE_WELD_BACKUP",
        "Fortnite_Teeth_Upper", "SK_L_MALE_Base_Skeleton_LOD0",
        "lips.001_RemoveInterior_Remesh to HardBody",
        "SKM_UEFN_Mannequin",  # body meshes hidden -- face focus
        "SK_Rhea_Vault_Identity1",
    ],

    # Camera: 'perspective' framed on head, or 'orthographic' (face only).
    "camera_type": "PERSP",
    "camera_lens": 50,
    "camera_distance_factor": 2.0,  # multiplier of head height (1.2 was too zoomed)

    # If True, append camera label (front/side) to filename and render both.
    "render_side_view": False,

    # Lighting setup. None = use whatever is in scene. "three_point" creates
    # a soft three-point rig + neutral world background so flat-shaded
    # placeholder materials render with visible form (key/fill/rim).
    # Idempotent: existing _VerifyLight_* objects are reused/replaced.
    "lighting": "three_point",
    # Energies tuned so avg pixel ~120/255 and 99%ile <240.
    # With AgX view transform + light placeholder materials (warm tan
    # at base_color=0.8+), even modest lights blow out fast.
    "key_energy":   25.0,
    "fill_energy":  10.0,
    "rim_energy":   15.0,
    "world_color": (0.12, 0.12, 0.14, 1.0),
    "view_exposure": -1.5,  # exposure compensation; ~35% brightness vs default

    # Render engine override. None = keep current.
    # 'BLENDER_EEVEE_NEXT' for fast preview, 'CYCLES' for final.
    "render_engine": "BLENDER_EEVEE",
}


def _hide_clutter(names):
    hidden = 0
    for n in names:
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        try:
            o.hide_render = True
            o.hide_set(True)
            hidden += 1
        except Exception:
            pass
    return hidden


def _setup_three_point_lighting(cfg, head):
    """Create or update three-point lighting around the head + a neutral
    world background. Idempotent: deletes prior _VerifyLight_* lamps.
    """
    # Remove old verify lamps
    for o in list(bpy.data.objects):
        if o.name.startswith("_VerifyLight_"):
            bpy.data.objects.remove(o, do_unlink=True)

    # Head center for aim
    cs = [head.matrix_world @ v.co for v in head.data.vertices]
    cx = sum(c.x for c in cs)/len(cs)
    cy = sum(c.y for c in cs)/len(cs)
    cz = sum(c.z for c in cs)/len(cs)
    height = max(c.z for c in cs) - min(c.z for c in cs)
    dist = max(height * 1.2, 0.3)

    def _add(name, loc_offset, energy, color=(1, 1, 1)):
        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_data.energy = energy
        light_data.size = 0.35
        light_data.color = color
        obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.scene.collection.objects.link(obj)
        from mathutils import Vector
        target = Vector((cx, cy, cz))
        pos = target + Vector(loc_offset) * dist
        obj.location = pos
        # Aim at head center
        direction = target - pos
        rot = direction.to_track_quat('-Z', 'Y').to_euler()
        obj.rotation_euler = rot
        return obj

    # Key: front-left-above
    _add("_VerifyLight_Key",  (-0.7, -1.0,  0.5), cfg["key_energy"],
         color=(1.0, 0.96, 0.92))
    # Fill: front-right, dimmer, cooler
    _add("_VerifyLight_Fill", ( 0.9, -0.7,  0.0), cfg["fill_energy"],
         color=(0.85, 0.9, 1.0))
    # Rim: back-above for edge separation
    _add("_VerifyLight_Rim",  ( 0.0,  1.0,  0.7), cfg["rim_energy"],
         color=(1.0, 1.0, 1.0))

    # Neutral world background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = cfg["world_color"]
        bg.inputs[1].default_value = 0.2  # was 0.6 -- too bright


def _setup_camera(cfg, head):
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or cam.type != 'CAMERA':
        # Find any camera in scene
        for o in bpy.data.objects:
            if o.type == 'CAMERA':
                cam = o; break
        if cam is None:
            cam_data = bpy.data.cameras.new("VerifyCam")
            cam = bpy.data.objects.new("VerifyCam", cam_data)
            bpy.context.scene.collection.objects.link(cam)
        scene.camera = cam
    cam.data.type = cfg["camera_type"]
    cam.data.lens = cfg["camera_lens"]
    # Frame on head
    cs = [head.matrix_world @ v.co for v in head.data.vertices]
    xs = [c.x for c in cs]; ys = [c.y for c in cs]; zs = [c.z for c in cs]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    height = max(zs) - min(zs)
    cam.location = Vector((cx, cy - height * cfg["camera_distance_factor"], cz))
    cam.rotation_euler = Euler((math.radians(90), 0, 0))
    return cam


def verify_rig_poses(cfg):
    arm = bpy.data.objects.get(cfg["armature"])
    head = bpy.data.objects.get(cfg["head_obj"])
    if arm is None or head is None:
        raise RuntimeError("armature or head not found")

    if arm.animation_data is None or arm.animation_data.action is None \
       or arm.animation_data.action.name != cfg["action"]:
        # Try to bind RigTest if it exists
        act = bpy.data.actions.get(cfg["action"])
        if act is None:
            raise RuntimeError(f"action '{cfg['action']}' not found -- run rig_test_animation first")
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = act
    arm.data.pose_position = 'POSE'

    n_hidden = _hide_clutter(cfg["hide_for_render"])
    cam = _setup_camera(cfg, head)
    if cfg.get("lighting") == "three_point":
        _setup_three_point_lighting(cfg, head)

    # Render engine
    if cfg.get("render_engine"):
        try:
            bpy.context.scene.render.engine = cfg["render_engine"]
        except Exception as e:
            print(f"  WARN: could not set engine {cfg['render_engine']}: {e}")
    if "view_exposure" in cfg:
        bpy.context.scene.view_settings.exposure = cfg["view_exposure"]

    out_dir = cfg["render_dir"]
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = cfg["resolution"]
    scene.render.image_settings.file_format = 'PNG'

    hold = cfg["hold"]
    trans = cfg["transition"]
    frame_in_hold = cfg["frame_in_hold"]

    index_lines = [f"# Pose verification render", f"# action: {cfg['action']}",
                   f"# hold={hold} transition={trans} render_at_offset={frame_in_hold}",
                   f"# head: {head.name}", f"# armature: {arm.name}",
                   f"# hidden clutter: {n_hidden}", ""]

    print(f"=== verify_rig_poses ===")
    print(f"  out: {out_dir}")
    print(f"  rendering {len(POSE_LABELS)} poses")

    pose_start = 1
    for i, label in enumerate(POSE_LABELS):
        render_frame = pose_start + frame_in_hold
        fname = f"pose_{i:02d}_{label}.png"
        scene.frame_set(render_frame)
        scene.render.filepath = os.path.join(out_dir, fname)
        bpy.ops.render.render(write_still=True)
        line = f"  pose {i:02d}  frame {render_frame:4d}  {label:25s}  -> {fname}"
        index_lines.append(line)
        print(line)
        pose_start += hold + trans

    index_path = os.path.join(out_dir, "_INDEX.txt")
    with open(index_path, "w") as f:
        f.write("\n".join(index_lines))
    print(f"[done] index: {index_path}")
    return len(POSE_LABELS)


if __name__ == "__main__":
    verify_rig_poses(CONFIG)
