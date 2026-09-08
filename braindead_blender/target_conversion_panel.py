"""Explicit target selection and conversion for already-skinned characters."""
import json
from pathlib import Path

import bpy
from bpy.props import EnumProperty, PointerProperty, StringProperty

from . import target_conversion, native_workflow


def armature_only(self, obj):
    return obj.type == 'ARMATURE'


def profile_changed(self, context):
    self.reference_fbx, self.reference_contract = native_workflow.reference_paths(self.profile)
    self.reference = None


class BD_TargetSettings(bpy.types.PropertyGroup):
    profile: EnumProperty(name='Target skeleton', items=[
        (name, values[0], values[1] or 'Original downloaded Epic Fab mannequin')
        for name, values in target_conversion.PROFILES.items()], default='NATIVE_DEVICE', update=profile_changed)
    source: PointerProperty(name='Source rig', type=bpy.types.Object, poll=armature_only)
    reference: PointerProperty(name='Loaded reference', type=bpy.types.Object, poll=armature_only)
    reference_fbx: StringProperty(name='Reference FBX', subtype='FILE_PATH',
                                  default=native_workflow.reference_paths()[0])
    reference_contract: StringProperty(name='Reference provenance JSON', subtype='FILE_PATH',
                                       description='Native reference: asset_path, fbx_sha256, armature_object_name',
                                       default=native_workflow.reference_paths()[1])


class BD_OT_LoadTargetReference(bpy.types.Operator):
    bl_idname = 'bd.load_target_reference'
    bl_label = 'Load Target Reference'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.bd_target_conversion
        path = Path(bpy.path.abspath(settings.reference_fbx))
        if context.mode != 'OBJECT' or not path.is_file():
            self.report({'ERROR'}, 'Use Object Mode and choose a reference FBX')
            return {'CANCELLED'}
        if settings.profile != 'FAB_UEFN' and not settings.reference_contract:
            self.report({'ERROR'}, 'Native targets require the exported reference provenance JSON')
            return {'CANCELLED'}
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False,
                                     automatic_bone_orientation=False, use_image_search=False)
            context.view_layer.update()
            created = set(bpy.data.objects) - before
            rigs = [o for o in created if o.type == 'ARMATURE']
            if len(rigs) != 1:
                raise ValueError('Expected exactly one named reference armature')
            contract = bpy.path.abspath(settings.reference_contract) if settings.reference_contract else None
            target_conversion.attach_provenance(rigs[0], path, contract, settings.profile)
            settings.reference = rigs[0]
            self.report({'INFO'}, 'Reference loaded. Convert creates a separate rig and mesh copy.')
        except Exception as exc:
            for obj in set(bpy.data.objects) - before:
                bpy.data.objects.remove(obj, do_unlink=True)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class BD_OT_ConvertTargetCopy(bpy.types.Operator):
    bl_idname = 'bd.convert_target_copy'
    bl_label = 'Convert Copy to Target'
    bl_description = 'Repose geometry and every morph to the selected reference; preserve source objects'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.bd_target_conversion
        if not settings.source or not settings.reference:
            self.report({'ERROR'}, 'Choose a source rig and load the target reference')
            return {'CANCELLED'}
        meshes = [o for o in context.scene.objects if o.type == 'MESH'
                  and any(m.type == 'ARMATURE' and m.object == settings.source for m in o.modifiers)]
        try:
            contract = json.loads(settings.reference.get('bdb_reference_contract', '{}'))
            result = target_conversion.rebind_copy(settings.source, settings.reference, meshes,
                                                   profile=settings.profile, bone_map=contract.get('bone_map'))
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        text = bpy.data.texts.new('BDB_Target_Conversion.json')
        text.write(json.dumps(result['report'], indent=2))
        bpy.ops.object.select_all(action='DESELECT')
        result['rig'].select_set(True)
        context.view_layer.objects.active = result['rig']
        self.report({'INFO'}, 'Converted copy created. Validate playback on the selected target in UEFN.')
        return {'FINISHED'}


class BD_PT_TargetConversion(bpy.types.Panel):
    bl_label = 'Fortnite Skeleton Target'
    bl_idname = 'BD_PT_TargetConversion'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_target_conversion
        layout.prop(settings, 'profile')
        layout.prop(settings, 'source')
        layout.prop(settings, 'reference_fbx')
        if settings.profile != 'FAB_UEFN':
            layout.prop(settings, 'reference_contract')
        layout.operator('bd.load_target_reference', icon='IMPORT')
        layout.prop(settings, 'reference')
        layout.operator('bd.convert_target_copy', icon='ARMATURE_DATA')
        layout.label(text='Creates a copy with all morphs.', icon='DUPLICATE')
        layout.label(text='Engine playback is a separate check.')


CLASSES = (BD_TargetSettings, BD_OT_LoadTargetReference, BD_OT_ConvertTargetCopy, BD_PT_TargetConversion)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bd_target_conversion = PointerProperty(type=BD_TargetSettings)


def unregister():
    del bpy.types.Scene.bd_target_conversion
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
