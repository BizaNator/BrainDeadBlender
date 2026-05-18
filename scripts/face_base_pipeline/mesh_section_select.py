"""
mesh_section_select.py

N-panel utility for working with merged face sub-meshes. After
merge_face_meshes joins Tongue / CustomLips / Eyelid_* / Eyebrow_* /
Ear_* into LowPolyHead_Rigged, the only thing that distinguishes their
origin is the `_section` face attribute (STRING domain=FACE) plus the
`section_<name>` vertex groups.

This panel reads `_section` from the active mesh and lets you:
  - Select all faces matching a chosen section (e.g. "Tongue")
  - Hide / Isolate (hide everything else) for sculpting / shape-key
    editing without other geometry getting in the way

Use case: editing the `tongueOut` shape key on the merged head. Pick
the Tongue section, click "Isolate", switch active shape key to
`tongueOut`, translate the verts, and the shape key records the
offset. Click "Reveal All" when done.

Standalone-runnable too: `exec(open(...).read())` in Blender's Python
console registers the panel under N-panel -> BrainDead -> Mesh Sections.
"""

import bpy
import bmesh
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import EnumProperty, PointerProperty, BoolProperty, StringProperty


SECTION_ATTR = "_section"


def _shape_key_enum_items(self, context):
    obj = context.active_object
    if obj is None or obj.type != 'MESH' or obj.data.shape_keys is None:
        return [("", "<no shape keys>", "Active mesh has no shape keys")]
    sk = obj.data.shape_keys
    items = []
    for i, kb in enumerate(sk.key_blocks):
        if kb == sk.reference_key:  # skip Basis
            continue
        items.append((kb.name, kb.name, f"Shape key '{kb.name}'"))
    if not items:
        return [("", "<only Basis>", "Only Basis shape key present")]
    return items


def _decode_section(raw):
    """STRING attributes store .value as bytes; some Blender builds give str.
    Normalize to str either way."""
    if isinstance(raw, (bytes, bytearray)):
        try: return raw.decode("utf-8")
        except Exception: return ""
    return str(raw or "")


def _section_names(obj):
    if obj is None or obj.type != 'MESH':
        return []
    attr = obj.data.attributes.get(SECTION_ATTR)
    if attr is None or attr.domain != 'FACE':
        return []
    names = set()
    for d in attr.data:
        s = _decode_section(d.value)
        if s:
            names.add(s)
    return sorted(names)


def _section_enum_items(self, context):
    obj = context.active_object
    names = _section_names(obj)
    if not names:
        return [("", "<no _section attr>", "Active mesh has no _section face attribute")]
    return [(n, n, f"Faces tagged _section = '{n}'") for n in names]


class BD_SectionSettings(PropertyGroup):
    section: EnumProperty(
        name="Section",
        description="Which _section value to act on",
        items=_section_enum_items,
    )
    extend: BoolProperty(
        name="Extend",
        description="Add to current selection instead of replacing it",
        default=False,
    )
    sculpt_shape_key: EnumProperty(
        name="Shape Key",
        description="Shape key whose offsets you want to sculpt",
        items=_shape_key_enum_items,
    )
    sculpt_show_armature: BoolProperty(
        name="Show Armature",
        description=("Display armature deform in Edit Mode too. Useful when "
                     "you need to see the verts in their posed positions "
                     "(but edits still apply to the BASIS mesh)."),
        default=True,
    )


def _section_faces(obj, sect):
    """Return list of face indices whose _section attribute matches sect."""
    attr = obj.data.attributes.get(SECTION_ATTR)
    if attr is None:
        return []
    return [i for i, d in enumerate(attr.data)
            if _decode_section(d.value) == sect]


def _ensure_edit_mode(obj, context):
    if obj.mode != 'EDIT':
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')


def _set_face_select_mode():
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)


class BD_OT_section_select(Operator):
    bl_idname = "braindead.section_select"
    bl_label = "Select"
    bl_description = ("Select all faces tagged with the chosen _section "
                      "value. Enters Edit Mode + face-select mode if needed.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        s = context.scene.bd_section
        sect = s.section
        if not sect:
            self.report({'ERROR'}, "No section chosen")
            return {'CANCELLED'}

        _ensure_edit_mode(obj, context)
        _set_face_select_mode()
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        sec_layer = bm.faces.layers.string.get(SECTION_ATTR)
        if sec_layer is None:
            self.report({'ERROR'}, f"No {SECTION_ATTR} face attribute")
            return {'CANCELLED'}

        if not s.extend:
            for f in bm.faces:
                f.select = False
        n = 0
        for f in bm.faces:
            if f[sec_layer].decode('utf-8', errors='ignore') == sect:
                f.select = True
                n += 1
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, f"Selected {n} '{sect}' faces")
        return {'FINISHED'}


class BD_OT_section_isolate(Operator):
    bl_idname = "braindead.section_isolate"
    bl_label = "Isolate"
    bl_description = ("Hide every face that does NOT belong to the chosen "
                      "section. Useful for shape-key sculpting on one "
                      "sub-mesh without other geometry interfering.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        sect = context.scene.bd_section.section
        if not sect:
            self.report({'ERROR'}, "No section chosen")
            return {'CANCELLED'}

        _ensure_edit_mode(obj, context)
        _set_face_select_mode()
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        sec_layer = bm.faces.layers.string.get(SECTION_ATTR)
        if sec_layer is None:
            self.report({'ERROR'}, f"No {SECTION_ATTR} face attribute")
            return {'CANCELLED'}

        kept = hidden = 0
        for f in bm.faces:
            match = f[sec_layer].decode('utf-8', errors='ignore') == sect
            f.hide = not match
            f.select = match
            if match: kept += 1
            else:     hidden += 1
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'},
                    f"Isolated '{sect}': {kept} faces visible, {hidden} hidden")
        return {'FINISHED'}


class BD_OT_section_hide(Operator):
    bl_idname = "braindead.section_hide"
    bl_label = "Hide"
    bl_description = "Hide all faces of the chosen section"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        sect = context.scene.bd_section.section
        _ensure_edit_mode(obj, context)
        bm = bmesh.from_edit_mesh(obj.data)
        sec_layer = bm.faces.layers.string.get(SECTION_ATTR)
        if sec_layer is None:
            self.report({'ERROR'}, f"No {SECTION_ATTR} face attribute")
            return {'CANCELLED'}
        n = 0
        for f in bm.faces:
            if f[sec_layer].decode('utf-8', errors='ignore') == sect:
                f.hide = True; n += 1
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, f"Hid {n} '{sect}' faces")
        return {'FINISHED'}


class BD_OT_sculpt_prep(Operator):
    bl_idname = "braindead.sculpt_prep"
    bl_label = "Sculpt Prep"
    bl_description = (
        "Set up the active mesh for shape-key sculpting in Edit Mode:\n"
        "  - Sets the chosen shape key as active + value to 1.0\n"
        "  - Pins it ('Show active shape key in Edit Mode') so its offsets "
        "are visible in the edit cage\n"
        "  - Enables Armature modifier 'Display in Edit Mode' + 'On Cage' "
        "so you see verts in their POSED positions (edits still go to BASIS)\n"
        "  - Isolates the chosen section (hides everything else)\n"
        "  - Drops you into Edit Mode + face-select\n\n"
        "Click 'Sculpt Restore' when done to revert these display flags."
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        s = context.scene.bd_section
        # Section is optional -- if not picked (or active mesh has no _section
        # attribute), do shape-key prep only and skip the isolate step.
        if not s.sculpt_shape_key:
            self.report({'ERROR'}, "Pick a shape key first")
            return {'CANCELLED'}
        sk = obj.data.shape_keys
        if sk is None:
            self.report({'ERROR'}, "Active mesh has no shape keys")
            return {'CANCELLED'}
        kb = sk.key_blocks.get(s.sculpt_shape_key)
        if kb is None:
            self.report({'ERROR'}, f"Shape key '{s.sculpt_shape_key}' missing")
            return {'CANCELLED'}

        # Make sure we're in Object Mode while we flip flags + activate the
        # key, then enter Edit Mode at the end.
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # 1. activate + value 1.0
        obj.active_shape_key_index = list(sk.key_blocks).index(kb)
        kb.value = 1.0
        # 2. pin the active shape key so it's visible in edit mode
        obj.show_only_shape_key = True
        # 3. armature mod display flags
        for m in obj.modifiers:
            if m.type == 'ARMATURE':
                m.show_in_editmode = bool(s.sculpt_show_armature)
                m.show_on_cage    = bool(s.sculpt_show_armature)

        # 4. isolate section if picked + valid; otherwise just enter Edit Mode
        sect_label = s.section or "<none>"
        if s.section and obj.data.attributes.get(SECTION_ATTR) is not None:
            bpy.ops.braindead.section_isolate()
        else:
            if obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            _set_face_select_mode()

        self.report({'INFO'},
                    f"Sculpt prep: active='{kb.name}'=1.0, pin=on, "
                    f"armature_in_edit={s.sculpt_show_armature}, "
                    f"section={sect_label}")
        return {'FINISHED'}


class BD_OT_sculpt_restore(Operator):
    bl_idname = "braindead.sculpt_restore"
    bl_label = "Sculpt Restore"
    bl_description = (
        "Revert the display flags Sculpt Prep set:\n"
        "  - Unpin the active shape key\n"
        "  - Turn off Armature modifier 'Display in Edit Mode' / 'On Cage'\n"
        "  - Reveal hidden faces\n"
        "  - Return to Object Mode"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        if obj.mode == 'EDIT':
            bpy.ops.mesh.reveal()
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.show_only_shape_key = False
        for m in obj.modifiers:
            if m.type == 'ARMATURE':
                m.show_in_editmode = False
                m.show_on_cage    = False
        self.report({'INFO'}, "Restored display flags + revealed faces")
        return {'FINISHED'}


class BD_OT_section_reveal(Operator):
    bl_idname = "braindead.section_reveal"
    bl_label = "Reveal All"
    bl_description = ("Unhide every face/vert/edge on the active mesh, "
                      "regardless of current mode. Works around the gotcha "
                      "where Isolate's hidden faces persist into the next "
                      "Edit Mode session (Blender stores hide state on the "
                      "mesh data, not on the editor view).")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        # Unhide directly on the mesh data so this works from Object Mode too.
        n = 0
        for v in obj.data.vertices:
            if v.hide: v.hide = False; n += 1
        for e in obj.data.edges:
            if e.hide: e.hide = False
        for p in obj.data.polygons:
            if p.hide: p.hide = False
        # If we happen to be in Edit Mode, also flush via the operator so the
        # edit cage display refreshes immediately.
        if obj.mode == 'EDIT':
            bpy.ops.mesh.reveal()
        self.report({'INFO'}, f"Revealed {n} hidden verts (and all hidden edges/faces)")
        return {'FINISHED'}


class BD_PT_sections(Panel):
    bl_label = "Mesh Sections"
    bl_idname = "BD_PT_sections"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "BrainDead"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            layout.label(text="Select a mesh object")
            return

        s = context.scene.bd_section
        names = _section_names(obj)
        layout.label(text=f"Active: {obj.name}")
        if not names:
            layout.label(text=f"No '{SECTION_ATTR}' face attribute", icon='ERROR')
            layout.label(text="(this mesh has not been through merge_face_meshes)")
            return

        layout.prop(s, "section", text="")
        layout.prop(s, "extend")

        row = layout.row(align=True)
        row.operator("braindead.section_select",  icon='RESTRICT_SELECT_OFF')
        row.operator("braindead.section_isolate", icon='HIDE_OFF')

        row = layout.row(align=True)
        row.operator("braindead.section_hide",   icon='HIDE_ON')
        row.operator("braindead.section_reveal", icon='RESTRICT_VIEW_OFF')

        # Shape-key sculpt prep -- the workflow that solves "edit mode
        # snaps to rest position and I can't see what I'm doing".
        layout.separator()
        box = layout.box()
        box.label(text="Sculpt Shape Key on Section", icon='SHAPEKEY_DATA')
        if obj.data.shape_keys is None:
            box.label(text="(mesh has no shape keys)", icon='INFO')
        else:
            box.prop(s, "sculpt_shape_key", text="Key")
            box.prop(s, "sculpt_show_armature")
            row = box.row(align=True)
            row.operator("braindead.sculpt_prep",    icon='EDITMODE_HLT')
            row.operator("braindead.sculpt_restore", icon='LOOP_BACK')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Prep: pins shape key + shows rig in edit mode.")
            col.label(text="Edits go to the active shape key's BASIS deltas.")

        layout.separator()
        col = layout.column(align=True)
        col.scale_y = 0.85
        col.label(text=f"{len(names)} section(s) in mesh:")
        for n in names:
            col.label(text=f"  - {n}")


CLASSES = (
    BD_SectionSettings,
    BD_OT_section_select,
    BD_OT_section_isolate,
    BD_OT_section_hide,
    BD_OT_section_reveal,
    BD_OT_sculpt_prep,
    BD_OT_sculpt_restore,
    BD_PT_sections,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.bd_section = PointerProperty(type=BD_SectionSettings)


def unregister():
    for c in reversed(CLASSES):
        try: bpy.utils.unregister_class(c)
        except Exception: pass
    try: del bpy.types.Scene.bd_section
    except Exception: pass


if __name__ == "__main__":
    try: unregister()
    except Exception: pass
    register()
    print("[mesh_section_select] registered. N-panel -> BrainDead -> Mesh Sections")
