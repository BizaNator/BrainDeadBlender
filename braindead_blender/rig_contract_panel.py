"""Read-only audit panel. Does not rename, rebind, move, or export an asset."""
import json
import bpy
from bpy.props import PointerProperty, BoolProperty
from . import rig_contract
from . import target_conversion_panel


def armature_only(self, obj):
    return obj.type=='ARMATURE'


class BD_RigContractSettings(bpy.types.PropertyGroup):
    reference: PointerProperty(name='Reference rig',type=bpy.types.Object,poll=armature_only)
    target: PointerProperty(name='Target rig',type=bpy.types.Object,poll=armature_only)
    require_fingers: BoolProperty(name='Require all 30 finger weights',default=True,
                                  description='Disable for a part that does not contain both hands')


class BD_OT_AuditRigContract(bpy.types.Operator):
    bl_idname='bd.audit_rig_contract'
    bl_label='Audit Skeleton and Skin'
    bl_description='Compare reference transforms and inspect real skin weights; write a JSON Text report'

    def execute(self,context):
        settings=context.scene.bd_rig_contract
        ref,target=settings.reference,settings.target
        if ref is None or target is None or ref==target:
            self.report({'ERROR'},'Choose two different reference and target armatures')
            return {'CANCELLED'}
        if context.mode!='OBJECT':
            self.report({'ERROR'},'Switch to Object Mode before auditing')
            return {'CANCELLED'}
        meshes=[o for o in context.scene.objects if o.type=='MESH'
                and any(m.type=='ARMATURE' and m.object==target for m in o.modifiers)]
        report=rig_contract.compare(rig_contract.snapshot(ref),rig_contract.snapshot(target,meshes),
                                     required_weighted=rig_contract.fingers() if settings.require_fingers else ())
        report['reference_object']=ref.name;report['target_object']=target.name
        text=bpy.data.texts.get('BDB_Rig_Contract.json') or bpy.data.texts.new('BDB_Rig_Contract.json')
        text.clear();text.write(json.dumps(report,indent=2,allow_nan=False))
        if report['structural_pass']:
            self.report({'INFO'},'Structural audit passed. Engine playback remains a separate check. See BDB_Rig_Contract.json')
        else:
            self.report({'WARNING'},f"{len(report['errors'])} findings: see BDB_Rig_Contract.json in the Text Editor")
        return {'FINISHED'}


class BD_PT_RigContract(bpy.types.Panel):
    bl_label='Skeleton Contract'
    bl_idname='BD_PT_RigContract'
    bl_space_type='VIEW_3D'
    bl_region_type='UI'
    bl_category='BrainDead'
    bl_options={'DEFAULT_CLOSED'}

    def draw(self,context):
        layout=self.layout;s=context.scene.bd_rig_contract
        layout.prop(s,'reference');layout.prop(s,'target');layout.prop(s,'require_fingers')
        layout.operator('bd.audit_rig_contract',icon='VIEWZOOM')
        layout.label(text='Report: BDB_Rig_Contract.json',icon='TEXT')


CLASSES=(BD_RigContractSettings,BD_OT_AuditRigContract,BD_PT_RigContract)


def register():
    for cls in CLASSES: bpy.utils.register_class(cls)
    bpy.types.Scene.bd_rig_contract=PointerProperty(type=BD_RigContractSettings)
    target_conversion_panel.register()


def unregister():
    target_conversion_panel.unregister()
    del bpy.types.Scene.bd_rig_contract
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)
