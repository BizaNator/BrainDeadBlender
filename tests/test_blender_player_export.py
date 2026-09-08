"""Exercise the full Player target and its non-unit unweighted leaf export."""
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
settings.target_pose = 'NATIVE_PLAYER'
assert settings.native_reference_fbx.endswith('Fortnite_Player_authoring_v03.fbx')
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
assert rig != source and rig['bdb_target_profile'] == 'NATIVE_PLAYER'
assert original_state() == before
assert len(rig.data.bones) == 279
reference = rig_contract.snapshot(rig)
autorig.unregister()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(temp / 'native.fbx'), use_anim=False)
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
assert arm.name == 'root', arm.name
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
result = rig_contract.compare(reference, rig_contract.snapshot(arm, objects),
    required_weighted=rig_contract.fingers(), required_morphs=['Belly', 'BodyMass', 'Muscular'])
(temp/'player_export_audit.json').write_text(json.dumps({'result':result,'reference':reference,'incoming':rig_contract.snapshot(arm,objects)},indent=2))
assert result['structural_pass'], result['errors']
assert len(objects) == 1
assert 'Nails' in [m.name for m in objects[0].data.materials]
from io_scene_fbx import parse_fbx
_tree,_version=parse_fbx.parse(str(temp / 'native.fbx'))
_objs=next(e for e in _tree.elems if e.id==b'Objects')
_leaf=next(e for e in _objs.elems if e.id==b'Model' and e.props[1].split(b'\0\1')[0]==b'pelvisRigidBodyShape1Transform')
_props=next(e for e in _leaf.elems if e.id==b'Properties70')
_scale=next(e.props[-3:] for e in _props.elems if e.id==b'P' and e.props[0]==b'Lcl Scaling')
assert all(abs(v-1.1239911317825317)<1e-10 for v in _scale),_scale
# A deforming finger is not a supported scale leaf. Rejection must leave bytes intact.
from braindead_blender.fbx_reference_scale import restore_leaf_scales
_bytes=(temp/'native.fbx').read_bytes()
try:
    restore_leaf_scales(temp/'native.fbx', {'index_03_l':[1.1]*3})
    raise AssertionError('Weighted leaf accepted')
except ValueError as exc:
    assert 'weighted' in str(exc),str(exc)
assert (temp/'native.fbx').read_bytes()==_bytes
print('NATIVE_PLAYER_AUTORIG_PASS default missing_reference_no_export frozen_originals root fingers morphs nails', temp, flush=True)
