"""
pose_editor_panel.py

A Blender side-panel for previewing, editing, and re-baking individual
RigTest poses. Pragmatic add-on that turns the 47-pose rig test into an
interactive per-character tuning workflow.

What it does
------------
- Lists every pose label from RigTest (read from the action's keyframe
  comments or fallback to a baked-in list).
- "Jump" button scrubs the timeline to the chosen pose's hold-middle
  frame so you can see the pose at rest.
- "Bake current pose" re-keys all pose-bone and shape-key values at the
  hold-start and hold-end frames of the chosen pose -- so adjustments
  you make in Pose Mode (or shape-key sliders) get cemented for that
  pose only, without disturbing other poses.
- "Re-render verifier" runs verify_rig_poses to refresh the contact sheet.
- "Save per-character" dumps the current per-pose values (bone matrices +
  shape key values at each hold frame) to a JSON next to the .blend so
  the same tweaks can be re-applied to the next head.

Install: copy pose_editor_panel.py next to the other face_base_pipeline
scripts. The BrainDeadBlender add-on will auto-discover it via the
Pipeline panel's loader.

This file is also runnable as a standalone script (`exec(open(...).read())`
in Blender's Python console) to register/unregister at runtime.
"""

import bpy
import os
import json
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import StringProperty, IntProperty, EnumProperty, PointerProperty


# Hold/transition pattern from rig_test_animation
HOLD = 12
TRANSITION = 6


def _armature_enum_items(self, context):
    """List every armature in the scene, sorted so the most-likely RigTest
    target lands at the top (which becomes the enum default). Priority:
      1) armature whose Animation Data action is named 'RigTest'
      2) armatures with canonical names: 'root', 'Fortnite_Armature', 'Armature'
      3) everything else alphabetical
    Re-runs every draw, so the list stays fresh as donors get added/hidden."""
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if not arms:
        return [("", "<no armatures>", "Scene has no armature object")]

    def _score(o):
        if o.animation_data and o.animation_data.action and \
           o.animation_data.action.name == "RigTest":
            return 0
        if o.name in ("root", "Fortnite_Armature", "Armature"):
            return 1
        return 2
    arms.sort(key=lambda o: (_score(o), o.name))
    return [(a.name, a.name, f"Armature '{a.name}'") for a in arms]


def _resolve_armature(name):
    """Pose Editor used a free-form StringProperty originally; existing
    .blends may have stale names ('Fortnite_Armature') saved. If the named
    armature isn't found, fall back to the first armature in the scene
    so we don't error out the bake."""
    o = bpy.data.objects.get(name) if name else None
    if o and o.type == 'ARMATURE':
        return o
    for a in bpy.data.objects:
        if a.type == 'ARMATURE':
            return a
    return None

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


def _pose_frames(index):
    """Return (hold_start, hold_end) for pose at given index."""
    start = 1 + index * (HOLD + TRANSITION)
    return start, start + HOLD


def _pose_enum_items(self, context):
    items = []
    for i, lbl in enumerate(POSE_LABELS):
        hs, he = _pose_frames(i)
        items.append((str(i), f"{i:02d}  {lbl}  [{hs}-{he}]",
                      f"Pose '{lbl}' holds at frames {hs}-{he}"))
    return items


class POSE_EDITOR_PG_settings(PropertyGroup):
    pose_index: EnumProperty(
        name="Pose",
        description="RigTest pose to navigate / edit",
        items=_pose_enum_items,
    )
    armature_name: EnumProperty(
        name="Armature",
        description="Armature to drive (auto-lists scene armatures)",
        items=_armature_enum_items,
    )
    head_name: StringProperty(
        name="Head",
        default="LowPolyHead_Rigged",
    )


class POSE_EDITOR_OT_jump(Operator):
    bl_idname = "pose_editor.jump"
    bl_label = "Jump to Pose"
    bl_description = "Scrub timeline to the middle of this pose's hold range"

    def execute(self, context):
        s = context.scene.pose_editor
        idx = int(s.pose_index)
        hs, he = _pose_frames(idx)
        context.scene.frame_set(hs + HOLD // 2)
        # Switch armature to POSE mode so user can immediately edit bones.
        # Wrap mode/select logic in try/except so an excluded armature
        # collection doesn't fail the whole jump -- timeline still moves.
        arm = _resolve_armature(s.armature_name)
        if arm:
            try:
                arm.data.pose_position = 'POSE'
            except Exception:
                pass
            # Only try select/mode swap if armature is actually in view layer
            if arm.name in context.view_layer.objects:
                try:
                    for o in context.view_layer.objects:
                        o.select_set(False)
                    arm.select_set(True)
                    context.view_layer.objects.active = arm
                    bpy.ops.object.mode_set(mode='POSE')
                except Exception as e:
                    self.report({'WARNING'}, f"Frame set but couldn't switch to Pose Mode: {e}")
            else:
                self.report({'WARNING'},
                            f"Armature '{arm.name}' not in view layer "
                            f"(its collection is excluded). Frame set, but "
                            f"unexclude the armature's collection to edit bones.")
        self.report({'INFO'}, f"Jumped to pose '{POSE_LABELS[idx]}' frame {hs+HOLD//2}")
        return {'FINISHED'}


class POSE_EDITOR_OT_bake_current(Operator):
    bl_idname = "pose_editor.bake_current"
    bl_label = "Bake Current to This Pose"
    bl_description = ("Re-key all pose-bone rotations/locations and shape "
                      "key values at the current frame across the chosen "
                      "pose's hold range. Cements your live edits for "
                      "this pose only.")

    def execute(self, context):
        s = context.scene.pose_editor
        idx = int(s.pose_index)
        hs, he = _pose_frames(idx)
        arm = _resolve_armature(s.armature_name)
        if arm is None:
            self.report({'ERROR'},
                        f"No armature found (picked '{s.armature_name}', "
                        f"no fallback armature in scene)")
            return {'CANCELLED'}

        # Bake bone transforms at hs and he
        baked_bones = 0
        for pb in arm.pose.bones:
            pb.rotation_mode = 'XYZ'
            for f in (hs, he):
                pb.keyframe_insert(data_path='rotation_euler', frame=f)
                pb.keyframe_insert(data_path='location', frame=f)
            baked_bones += 1

        # Bake shape key values at hs and he on all armature-driven meshes
        baked_sks = 0
        for o in bpy.data.objects:
            if o.type != 'MESH':
                continue
            if not any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers):
                continue
            sk = o.data.shape_keys
            if sk is None:
                continue
            for kb in sk.key_blocks:
                if kb == sk.reference_key:
                    continue
                for f in (hs, he):
                    kb.keyframe_insert(data_path='value', frame=f)
                baked_sks += 1

        self.report({'INFO'},
                    f"Baked {baked_bones} bone kfs x2 frames + {baked_sks} "
                    f"shape-key kfs x2 frames for pose '{POSE_LABELS[idx]}' "
                    f"({hs}-{he})")
        return {'FINISHED'}


class POSE_EDITOR_OT_render_verify(Operator):
    bl_idname = "pose_editor.render_verify"
    bl_label = "Re-render All Poses"
    bl_description = "Run verify_rig_poses to refresh the per-pose PNGs + contact sheet"

    def execute(self, context):
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            script = os.path.join(here, "verify_rig_poses.py")
            ns = {"__file__": script, "__name__": "__inline__"}
            exec(compile(open(script).read(), script, 'exec'), ns)
            n = ns["verify_rig_poses"](ns["CONFIG"])
            self.report({'INFO'}, f"Re-rendered {n} pose images")
        except Exception as e:
            self.report({'ERROR'}, f"verify failed: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class POSE_EDITOR_OT_save_per_character(Operator):
    bl_idname = "pose_editor.save_per_character"
    bl_label = "Save Per-Character Pose JSON"
    bl_description = ("Dump current per-pose values (bone matrices + shape "
                      "key values at hold-mid frame) to a JSON next to the "
                      ".blend so tweaks port to the next head.")

    def execute(self, context):
        s = context.scene.pose_editor
        arm = _resolve_armature(s.armature_name)
        if arm is None:
            self.report({'ERROR'},
                        f"No armature found (picked '{s.armature_name}', "
                        f"no fallback armature in scene)")
            return {'CANCELLED'}

        out = {"poses": {}}
        scene = context.scene
        original_frame = scene.frame_current
        for i, label in enumerate(POSE_LABELS):
            hs, _ = _pose_frames(i)
            scene.frame_set(hs + HOLD // 2)
            bones = {}
            for pb in arm.pose.bones:
                bones[pb.name] = {
                    "loc":   list(pb.location),
                    "euler": list(pb.rotation_euler),
                }
            shapes = {}
            for o in bpy.data.objects:
                if o.type != 'MESH':
                    continue
                if not any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers):
                    continue
                sk = o.data.shape_keys
                if sk is None:
                    continue
                nz = {kb.name: kb.value for kb in sk.key_blocks
                      if kb != sk.reference_key and abs(kb.value) > 1e-5}
                if nz:
                    shapes[o.name] = nz
            out["poses"][label] = {"frame": hs + HOLD // 2,
                                    "bones": bones,
                                    "shape_keys": shapes}
        scene.frame_set(original_frame)

        path = bpy.path.abspath("//pose_overrides.json")
        if not bpy.data.filepath:
            path = os.path.expanduser("~/pose_overrides.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        self.report({'INFO'}, f"Wrote {path} ({len(out['poses'])} poses)")
        return {'FINISHED'}


class POSE_EDITOR_PT_panel(Panel):
    bl_label = "Pose Editor"
    bl_idname = "POSE_EDITOR_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "BrainDead"

    def draw(self, context):
        layout = self.layout
        s = context.scene.pose_editor

        layout.prop(s, "armature_name", text="Armature")
        layout.prop(s, "head_name", text="Head")
        layout.separator()
        layout.prop(s, "pose_index", text="")

        row = layout.row(align=True)
        row.operator("pose_editor.jump", icon='RESTRICT_VIEW_OFF')
        row.operator("pose_editor.bake_current", icon='KEY_HLT', text="Bake")

        layout.separator()
        layout.operator("pose_editor.render_verify", icon='RENDER_STILL')
        layout.operator("pose_editor.save_per_character", icon='FILE_TICK')

        idx = int(s.pose_index)
        hs, he = _pose_frames(idx)
        col = layout.column(align=True)
        col.label(text=f"Hold range: frames {hs}-{he}")
        col.label(text=f"Hold middle: frame {hs + HOLD // 2}")


CLASSES = (
    POSE_EDITOR_PG_settings,
    POSE_EDITOR_OT_jump,
    POSE_EDITOR_OT_bake_current,
    POSE_EDITOR_OT_render_verify,
    POSE_EDITOR_OT_save_per_character,
    POSE_EDITOR_PT_panel,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.pose_editor = PointerProperty(type=POSE_EDITOR_PG_settings)


def unregister():
    for c in reversed(CLASSES):
        try: bpy.utils.unregister_class(c)
        except Exception: pass
    try: del bpy.types.Scene.pose_editor
    except Exception: pass


if __name__ == "__main__":
    try: unregister()
    except Exception: pass
    register()
    print("[pose_editor_panel] registered. Find under N-panel -> BrainDead -> Pose Editor")
