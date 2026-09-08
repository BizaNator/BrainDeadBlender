"""Exercise the existing AutoRig native default, failure path and real export."""
import bpy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from braindead_blender import autorig, rig_contract, target_conversion, target_conversion_panel

bpy.ops.wm.open_mainfile(filepath='/mnt/tank/Studio/Brains/Characters/_base_models/codex_male_v02/COB_Male_Base_v02.blend')
autorig.register()
target_conversion_panel.register()
panel = bpy.context.scene.bd_target_conversion
assert panel.profile == 'NATIVE_DEVICE'
panel.profile = 'FAB_UEFN'
assert panel.reference_fbx.endswith('skm_uefn_mannequin.FBX') and not panel.reference_contract
panel.profile = 'NATIVE_DEVICE'
assert panel.reference_fbx.endswith('CP_Device_Mannequin_named.fbx') and panel.reference_contract
target_conversion_panel.unregister()
settings = bpy.context.scene.bd_autorig
assert settings.target_pose == 'NATIVE_DEVICE', settings.target_pose
source = bpy.data.objects['root']
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
def original_state():
    return json.dumps({'rig': target_conversion.reference_fingerprint(source),
                      'meshes': {o.name: {'co': [list(v.co) for v in o.data.vertices],
                         'weights': [[(g.group,g.weight) for g in v.groups] for v in o.data.vertices],
                         'keys': {k.name: [list(v.co) for v in k.data] for k in o.data.shape_keys.key_blocks}}
                         for o in meshes}},sort_keys=True)
before = original_state()
collection = bpy.data.collections.new('Export')
bpy.context.scene.collection.children.link(collection)
for obj in [source] + meshes:
    collection.objects.link(obj)
temp = Path(tempfile.mkdtemp(prefix='bdb_native_autorig_'))
correct_path = settings.native_reference_fbx
settings.native_reference_fbx = str(temp / 'absent.fbx')
try:
    status = bpy.ops.braindead.export_uefn_fbx(filepath=str(temp / 'must_not_exist.fbx'))
except RuntimeError:
    status = {'CANCELLED'}
assert status == {'CANCELLED'} and not (temp / 'must_not_exist.fbx').exists()
assert bpy.data.collections['Export'] == collection and original_state() == before
settings.native_reference_fbx = correct_path
status = bpy.ops.braindead.export_uefn_fbx(filepath=str(temp / 'native.fbx'))
assert status == {'FINISHED'}
export = bpy.data.collections['Export']
rig = next(o for o in export.all_objects if o.type == 'ARMATURE')
assert rig != source and rig['bdb_target_profile'] == 'NATIVE_DEVICE'
assert original_state() == before
assert len(rig.data.bones) == 87
reference = rig_contract.snapshot(rig)
autorig.unregister()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(temp / 'native.fbx'), use_anim=False)
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
assert arm.name == 'root', arm.name
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
result = rig_contract.compare(reference, rig_contract.snapshot(arm, objects),
    required_weighted=rig_contract.fingers(), required_morphs=['Belly', 'BodyMass', 'Muscular'])
assert result['structural_pass'], result['errors']
assert len(objects) == 1
assert 'Nails' in [m.name for m in objects[0].data.materials]
print('NATIVE_AUTORIG_PASS default missing_reference_no_export frozen_originals root fingers morphs nails', temp, flush=True)
