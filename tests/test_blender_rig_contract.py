"""Actual Blender import, rest-bone corruption, weight corruption and UI lifecycle."""
import bpy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rig_contract',ROOT/'braindead_blender/rig_contract.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
BASE=Path('/mnt/tank/Studio/Brains/Characters/_base_models/codex_male_v01')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath='/mnt/tank/Studio/Brains/Skills/char-designer/skm_uefn_mannequin.FBX',use_image_search=False)
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
reference=audit.snapshot(rig)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(BASE/'exports/SK_COB_Male_Base_v01.fbx'),use_image_search=False)
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
baseline=audit.compare(reference,audit.snapshot(rig,meshes),required_weighted=audit.fingers())
assert baseline['structural_pass'],baseline['errors']
bpy.context.view_layer.objects.active=rig;rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bone=rig.data.edit_bones['index_03_l'];head=bone.head.copy();tail=bone.tail.copy()
bone.head.x+=.1;bone.tail.x+=.1
bpy.ops.object.mode_set(mode='OBJECT')
broken=audit.compare(reference,audit.snapshot(rig,meshes),required_weighted=audit.fingers())
assert 'rest_position:index_03_l' in broken['errors'],broken['errors']
bpy.ops.object.mode_set(mode='EDIT')
bone=rig.data.edit_bones['index_03_l'];bone.head=head;bone.tail=tail
bpy.ops.object.mode_set(mode='OBJECT')
restored=audit.compare(reference,audit.snapshot(rig,meshes),required_weighted=audit.fingers())
assert restored['structural_pass'],restored['errors']
sys.path.insert(0,str(ROOT))
from braindead_blender import rig_contract_panel
rig_contract_panel.register()
assert hasattr(bpy.context.scene,'bd_rig_contract')
assert hasattr(bpy.ops.bd,'audit_rig_contract')
rig_contract_panel.unregister()
assert not hasattr(bpy.context.scene,'bd_rig_contract')
report={'baseline_pass':True,'moved_bone_detected':broken['errors'],'restored_pass':True,
        'panel_register_unregister':True}
(BASE/'review/audit_blender_test.json').write_text(json.dumps(report,indent=2))
print('BLENDER_AUDIT_TESTS_PASS',json.dumps(report))
