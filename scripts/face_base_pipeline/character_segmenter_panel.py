"""character_segmenter_panel.py — BDTools 'Character Segmenter' sidebar panel."""
import bpy
import os
from importlib import util as _util

def _seg_module():
    """Load character_segmenter.py sitting next to this file."""
    path = os.path.join(os.path.dirname(__file__), "character_segmenter.py")
    spec = _util.spec_from_file_location("character_segmenter", path)
    mod = _util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ALL_PARTS = ["Head","Neck","Torso","Arm_L","Arm_R","Hand_L","Hand_R",
              "Hips","Leg_L","Leg_R","Foot_L","Foot_R"]


class BD_OT_segment_character(bpy.types.Operator):
    bl_idname = "braindead.segment_character"
    bl_label = "Segment Character"
    bl_description = "Cut the active body mesh into standardized kit parts"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a body mesh first")
            return {'CANCELLED'}
        seg = _seg_module()
        s = context.scene
        wanted = {p for p in _ALL_PARTS
                  if getattr(s, f"bd_seg_{p.lower()}", True)}
        parts = seg.segment_character(obj, parts_wanted=wanted)
        col = bpy.data.collections.get("Character_Kit_Parts")
        if not col:
            col = bpy.data.collections.new("Character_Kit_Parts")
            context.scene.collection.children.link(col)
        for o in parts.values():
            for c in list(o.users_collection):
                c.objects.unlink(o)
            col.objects.link(o)
        obj.hide_set(True)
        self.report({'INFO'}, f"Segmented into {len(parts)} parts")
        return {'FINISHED'}


class BD_PT_character_segmenter(bpy.types.Panel):
    bl_label = "Character Segmenter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'      # match procedural_body_panel.py's bl_category

    def draw(self, context):
        layout = self.layout
        layout.label(text="Parts to extract:")
        grid = layout.grid_flow(columns=2, even_columns=True)
        for p in _ALL_PARTS:
            grid.prop(context.scene, f"bd_seg_{p.lower()}", text=p)
        layout.separator()
        layout.operator("braindead.segment_character", icon='MOD_BOOLEAN')


_CLASSES = (BD_OT_segment_character, BD_PT_character_segmenter)


def register():
    for p in _ALL_PARTS:
        setattr(bpy.types.Scene, f"bd_seg_{p.lower()}",
                bpy.props.BoolProperty(name=p, default=True))
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
    for p in _ALL_PARTS:
        delattr(bpy.types.Scene, f"bd_seg_{p.lower()}")
