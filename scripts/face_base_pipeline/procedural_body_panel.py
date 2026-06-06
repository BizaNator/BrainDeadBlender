"""
procedural_body_panel.py

A Blender side-panel ("Procedural Body Builder") that wraps the procedural
torso Geometry-Nodes system (BD_LimbAnatomy_Torso) with a friendly control
surface so an artist can drive it without hunting through raw GN sockets.

What it does
------------
- Body Type   : dropdown of muscle archetypes (athletic / curvy / heavyset…)
                with an F/M gender toggle + Apply -> apply_torso_archetype.
- Body Shape  : dropdown of silhouette presets (v_taper / hourglass / round…)
                + Apply -> apply_torso_shape.
- Sliders     : the high-level master sockets drawn live off the active
                object's GN modifier, so they stay drag-able.
- Reset       : rewrites raw sockets to TORSO_DEFAULTS -> reset_torso_defaults.

Operates on the active object only when it carries a Geometry Nodes modifier
whose node_group is named 'BD_LimbAnatomy_Torso'. When it doesn't, the panel
still draws and shows a hint label.

Install: copy procedural_body_panel.py next to the other face_base_pipeline
scripts. The BrainDeadBlender add-on auto-discovers it via the _SIDECAR_PANELS
loader in the add-on __init__.py (this file is listed there).

This file is also runnable as a standalone script (`exec(open(...).read())`
in Blender's Python console) to register/unregister at runtime.
"""

import bpy
import os
from importlib import util
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import EnumProperty, PointerProperty, BoolProperty, FloatProperty


# Node group name that identifies the torso anatomy object.
TORSO_NODE_GROUP = "BD_LimbAnatomy_Torso"
# Anatomy node groups this panel drives (torso + pelvis both have master sliders).
ANATOMY_NODE_GROUPS = ("BD_LimbAnatomy_Torso", "BD_LimbAnatomy_Pelvis", "BD_Pelvis")

# Raw per-muscle socket suffixes — hidden from the friendly slider list (they
# go under the Advanced expander). Everything else float-typed is a "master".
_RAW_SUFFIXES = (" Axial", " Amount", " Sigma", " Conc", " Flat", " Archetype")
# Structural / build-time sockets — also kept out of the sculpt sliders.
_STRUCTURAL = {"Start Empty", "End Empty", "Start Radius", "End Radius",
               "N Sides", "N Axial", "Limb Side"}


def _is_master_socket(name):
    """True for high-level sculpt sliders (not raw per-muscle, not structural).
    Dynamic so any socket added to the node group auto-appears in the panel."""
    if name in _STRUCTURAL:
        return False
    return not any(name.endswith(suf) for suf in _RAW_SUFFIXES)


# ---------------------------------------------------------------------------
# Meta-layer module loading
# ---------------------------------------------------------------------------
# limb_anatomy_geonode.py lives next to this file. Load it fresh by path the
# same way the add-on's ARKit operator loads donor_registry, so script edits
# don't need a Blender restart and we don't depend on the add-on package name.

_META = None


def _meta():
    """Lazy-load and cache the limb_anatomy_geonode module."""
    global _META
    if _META is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "limb_anatomy_geonode.py")
        spec = util.spec_from_file_location("limb_anatomy_geonode", path)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _META = mod
    return _META


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torso_modifier(obj):
    """Return the Geometry Nodes modifier on `obj` whose node group is the
    torso anatomy group, or None if `obj` isn't a torso anatomy object."""
    if obj is None or obj.type != 'MESH':
        return None
    for mod in obj.modifiers:
        if (mod.type == 'NODES' and mod.node_group is not None
                and mod.node_group.name == TORSO_NODE_GROUP):
            return mod
    return None


def _anatomy_modifier(obj):
    """Return the GN modifier for any anatomy group this panel drives (torso
    OR pelvis), or None. The sculpt sliders work for both."""
    if obj is None or obj.type != 'MESH':
        return None
    for mod in obj.modifiers:
        if (mod.type == 'NODES' and mod.node_group is not None
                and mod.node_group.name in ANATOMY_NODE_GROUPS):
            return mod
    return None


def _socket_identifiers(node_group):
    """Map {socket .name: .identifier} for INPUT sockets of a node group --
    same resolution the meta-layer code uses."""
    return {s.name: s.identifier
            for s in node_group.interface.items_tree
            if s.item_type == 'SOCKET' and s.in_out == 'INPUT'}


def _archetype_enum_items(self, context):
    """Archetype keys for the current gender toggle. Rebuilt each draw so the
    list tracks the gender switch."""
    meta = _meta()
    gender = getattr(context.scene.bd_procbody, "gender", "f")
    table = meta.TORSO_ARCHETYPES_F if gender == "f" else meta.TORSO_ARCHETYPES_M
    return [(k, k.replace("_", " ").title(), f"Body type '{k}'")
            for k in table]


def _shape_enum_items(self, context):
    """Body-shape silhouette preset keys."""
    meta = _meta()
    return [(k, k.replace("_", " ").title(), f"Body shape '{k}'")
            for k in meta.TORSO_SHAPE_PRESETS]


# ---------------------------------------------------------------------------
# Body-wide trait sliders (character-creator layer)
# ---------------------------------------------------------------------------

def _trait_prop_name(trait):
    """Scene-property name for a trait, e.g. 'Body Fat' -> 'trait_body_fat'."""
    return "trait_" + trait.lower().replace(" ", "_")


def _traits_update(self, context):
    """A trait slider moved — recompute every trait-driven socket on all parts."""
    meta = _meta()
    vals = {t: getattr(self, _trait_prop_name(t))
            for (t, _lo, _hi) in meta.BODY_TRAIT_SLIDERS}
    meta.apply_body_traits(vals, context.scene)


# ---------------------------------------------------------------------------
# Property group
# ---------------------------------------------------------------------------

class BD_PG_procbody(PropertyGroup):
    gender: EnumProperty(
        name="Gender",
        description="Which archetype table to pick body types from",
        items=[
            ('f', "Female", "Female archetype table"),
            ('m', "Male", "Male archetype table"),
        ],
        default='f',
    )
    archetype: EnumProperty(
        name="Body Type",
        description="Muscle archetype to apply (athletic / curvy / heavyset…)",
        items=_archetype_enum_items,
    )
    shape: EnumProperty(
        name="Body Shape",
        description="Silhouette preset to apply (V-taper / hourglass / round…)",
        items=_shape_enum_items,
    )
    show_advanced: BoolProperty(
        name="Advanced",
        description="Show the raw per-muscle sockets (Axial/Amount/Sigma/…)",
        default=False,
    )


# Inject one slider property per body trait onto the property group — the
# trait list lives entirely in limb_anatomy_geonode.BODY_TRAIT_SLIDERS, so
# adding a trait there auto-adds its slider here.
for _t, _lo, _hi in _meta().BODY_TRAIT_SLIDERS:
    BD_PG_procbody.__annotations__[_trait_prop_name(_t)] = FloatProperty(
        name=_t,
        description=f"{_t} — drives every tagged socket across the whole body",
        default=0.0, min=_lo, max=_hi,
        subtype='FACTOR' if _lo >= 0.0 else 'NONE',
        update=_traits_update,
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class BD_OT_procbody_apply_archetype(Operator):
    """Apply the selected body-type archetype to the active torso object"""
    bl_idname = "braindead.procbody_apply_archetype"
    bl_label = "Apply Body Type"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _torso_modifier(context.active_object) is not None

    def execute(self, context):
        obj = context.active_object
        if _torso_modifier(obj) is None:
            self.report({'ERROR'}, "Active object is not a torso anatomy object")
            return {'CANCELLED'}
        s = context.scene.bd_procbody
        try:
            _meta().apply_torso_archetype(obj, s.archetype, s.gender)
        except Exception as e:
            self.report({'ERROR'}, f"Apply body type failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Applied body type '{s.archetype}' ({s.gender})")
        return {'FINISHED'}


class BD_OT_procbody_apply_shape(Operator):
    """Apply the selected body-shape silhouette preset to the active torso"""
    bl_idname = "braindead.procbody_apply_shape"
    bl_label = "Apply Body Shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _torso_modifier(context.active_object) is not None

    def execute(self, context):
        obj = context.active_object
        if _torso_modifier(obj) is None:
            self.report({'ERROR'}, "Active object is not a torso anatomy object")
            return {'CANCELLED'}
        s = context.scene.bd_procbody
        try:
            _meta().apply_torso_shape(obj, s.shape)
        except Exception as e:
            self.report({'ERROR'}, f"Apply body shape failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Applied body shape '{s.shape}'")
        return {'FINISHED'}


class BD_OT_procbody_reset(Operator):
    """Rewrite every raw torso socket back to TORSO_DEFAULTS. Use this when a
    raw value was hand-cranked -- the master sliders multiply and can't undo it"""
    bl_idname = "braindead.procbody_reset"
    bl_label = "Reset to Defaults"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _torso_modifier(context.active_object) is not None

    def execute(self, context):
        obj = context.active_object
        if _torso_modifier(obj) is None:
            self.report({'ERROR'}, "Active object is not a torso anatomy object")
            return {'CANCELLED'}
        try:
            _meta().reset_torso_defaults(obj)
        except Exception as e:
            self.report({'ERROR'}, f"Reset failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, "Torso sockets reset to defaults")
        return {'FINISHED'}


class BD_OT_procbody_reset_traits(Operator):
    """Zero every Character-Creator trait slider (back to the neutral body)"""
    bl_idname = "braindead.procbody_reset_traits"
    bl_label = "Reset Traits"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.bd_procbody
        for (t, _lo, _hi) in _meta().BODY_TRAIT_SLIDERS:
            setattr(s, _trait_prop_name(t), 0.0)
        self.report({'INFO'}, "Character Creator traits reset")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class BD_PT_procbody(Panel):
    """Procedural Body Builder - friendly controls for the BD_LimbAnatomy_Torso
    Geometry-Nodes system."""
    bl_label = "Procedural Body Builder"
    bl_idname = "BD_PT_procbody"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "BrainDead"

    def draw(self, context):
        layout = self.layout
        s = context.scene.bd_procbody

        # --- Character Creator — body-wide trait sliders (whole scene) -------
        cc = layout.box()
        cc.label(text="Character Creator:", icon='OUTLINER_OB_ARMATURE')
        cccol = cc.column(align=True)
        for (t, _lo, _hi) in _meta().BODY_TRAIT_SLIDERS:
            cccol.prop(s, _trait_prop_name(t), text=t, slider=True)
        cc.operator("braindead.procbody_reset_traits", icon='LOOP_BACK')

        obj = context.active_object
        mod = _anatomy_modifier(obj)

        if mod is None:
            box = layout.box()
            box.label(text="No anatomy object selected.", icon='INFO')
            box.label(text="Select a torso or pelvis mesh with a")
            box.label(text="BD_LimbAnatomy Geometry Nodes modifier.")
            return

        is_torso = mod.node_group.name == TORSO_NODE_GROUP
        layout.label(text=f"Active: {obj.name}", icon='OUTLINER_OB_MESH')

        # --- Body Type / Shape (torso-only apply functions) ------------------
        if is_torso:
            box = layout.box()
            box.label(text="Body Type:", icon='ARMATURE_DATA')
            box.row().prop(s, "gender", expand=True)
            row = box.row(align=True)
            row.prop(s, "archetype", text="")
            row.operator("braindead.procbody_apply_archetype", text="Apply")

            box = layout.box()
            box.label(text="Body Shape:", icon='MESH_CUBE')
            row = box.row(align=True)
            row.prop(s, "shape", text="")
            row.operator("braindead.procbody_apply_shape", text="Apply")

        # --- Master sliders — drawn dynamically from the node group ----------
        box = layout.box()
        box.label(text="Sculpt Controls:", icon='MODIFIER')
        sid = _socket_identifiers(mod.node_group)
        drawn = 0
        for name, ident in sid.items():
            if not _is_master_socket(name):
                continue
            box.prop(mod, f'["{ident}"]', text=name)
            drawn += 1
        if drawn == 0:
            box.label(text="(no master sockets on this node group)", icon='ERROR')

        # --- Advanced: raw per-muscle sockets --------------------------------
        adv = layout.box()
        adv.prop(s, "show_advanced", text="Advanced (raw muscle sockets)",
                 icon='TRIA_DOWN' if s.show_advanced else 'TRIA_RIGHT', emboss=False)
        if s.show_advanced:
            col = adv.column(align=True)
            for name, ident in sid.items():
                if _is_master_socket(name) or name in _STRUCTURAL:
                    continue
                col.prop(mod, f'["{ident}"]', text=name)

        # --- Reset (torso-only) ----------------------------------------------
        if is_torso:
            layout.separator()
            layout.operator("braindead.procbody_reset", icon='LOOP_BACK')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CLASSES = (
    BD_PG_procbody,
    BD_OT_procbody_apply_archetype,
    BD_OT_procbody_apply_shape,
    BD_OT_procbody_reset,
    BD_OT_procbody_reset_traits,
    BD_PT_procbody,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.bd_procbody = PointerProperty(type=BD_PG_procbody)


def unregister():
    try:
        del bpy.types.Scene.bd_procbody
    except Exception:
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
    print("[procedural_body_panel] registered. Find under "
          "N-panel -> BrainDead -> Procedural Body Builder")
