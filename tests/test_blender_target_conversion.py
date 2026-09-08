"""Blender integration: full reference matrices, all morphs, frozen originals."""
import bpy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('target_conversion',ROOT/'braindead_blender/target_conversion.py')
convert=importlib.util.module_from_spec(spec);spec.loader.exec_module(convert)
BASE=Path('/mnt/tank/Studio/Brains/Characters/_base_models/codex_male_v02/COB_Male_Base_v02.blend')
bpy.ops.wm.open_mainfile(filepath=str(BASE))
source=bpy.data.objects['root'];meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']

def state():
    data={'rig':{b.name:[list(r) for r in source.matrix_world@b.matrix_local] for b in source.data.bones},'meshes':{}}
    for o in meshes:
        data['meshes'][o.name]={'vertices':[list(v.co) for v in o.data.vertices],
          'keys':{k.name:[list(v.co) for v in k.data] for k in o.data.shape_keys.key_blocks},
          'weights':[[[g.group,g.weight] for g in v.groups] for v in o.data.vertices]}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

before=state()
reference=source.copy();reference.data=source.data.copy();bpy.context.collection.objects.link(reference)
reference.name='Synthetic_Reference';bpy.context.view_layer.update()
identity=convert.rebind_copy(source,reference,meshes,profile='FAB_UEFN')
assert identity['report']['max_basis_delta_m']<1e-6
assert state()==before
assert len(identity['meshes'])==len(meshes)
for old,new in zip(meshes,identity['meshes']):
    assert old.data!=new.data and old.data.shape_keys!=new.data.shape_keys
    assert len(old.data.vertices)==len(new.data.vertices)
    assert [tuple(p.vertices) for p in old.data.polygons]==[tuple(p.vertices) for p in new.data.polygons]
    for key in old.data.shape_keys.key_blocks:
        assert max((a.co-b.co).length for a,b in zip(key.data,new.data.shape_keys.key_blocks[key.name].data))<1e-6

# A roll-only change leaves joint heads/tails fixed. A head-direction-only
# converter incorrectly returns identity; full rest matrices must move the skin.
bpy.ops.object.select_all(action='DESELECT');reference.select_set(True);bpy.context.view_layer.objects.active=reference
bpy.ops.object.mode_set(mode='EDIT');reference.data.edit_bones['upperarm_l'].roll+=math.radians(35)
bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
rotated=convert.rebind_copy(source,reference,meshes,profile='FAB_UEFN')
assert rotated['report']['max_basis_delta_m']>.005
for old,new in zip(meshes,rotated['meshes']):
    for key in old.data.shape_keys.key_blocks:
        for vertex,datum in zip(old.data.vertices,key.data):
            world=old.matrix_world@datum.co
            expected=world*0
            for g in vertex.groups:
                name=old.vertex_groups[g.group].name
                a=source.matrix_world@source.data.bones[name].matrix_local
                b=reference.matrix_world@reference.data.bones[name].matrix_local
                expected+=(b@a.inverted()@world)*g.weight
            actual=new.matrix_world@new.data.shape_keys.key_blocks[key.name].data[vertex.index].co
            assert (expected-actual).length<2e-6,(old.name,key.name,vertex.index)
    if new.data.shape_keys.animation_data:
        for driver in new.data.shape_keys.animation_data.drivers:
            for var in driver.driver.variables:
                for target in var.targets:
                    if target.id:assert target.id!=source
assert state()==before

objects_before=set(bpy.data.objects)
try:convert.rebind_copy(source,reference,meshes,profile='NATIVE_DEVICE')
except ValueError as exc:assert 'provenance' in str(exc).lower()
else:raise AssertionError('Native profile accepted unprovenanced reference')
assert set(bpy.data.objects)==objects_before

bpy.ops.object.select_all(action='DESELECT');reference.select_set(True);bpy.context.view_layer.objects.active=reference
bpy.ops.object.mode_set(mode='EDIT');reference.data.edit_bones.remove(reference.data.edit_bones['thumb_03_l']);bpy.ops.object.mode_set(mode='OBJECT')
try:convert.rebind_copy(source,reference,meshes,profile='FAB_UEFN')
except ValueError as exc:assert 'thumb_03_l' in str(exc)
else:raise AssertionError('Missing weighted reference bone accepted')
assert set(bpy.data.objects)==objects_before and state()==before
print('TARGET_CONVERSION_PASS identity all_morphs roll_only_delta drivers missing_bone native_provenance originals_unchanged')
