"""Build a named reference FBX from complete UE local/world reference captures.

Calibrate the coordinate conversion against an existing FBX and its UE capture.
Carry that FBX's bone-axis conventions through full matrices, including roll.
The output must then pass an independent Unreal import against the target JSON.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(value):
    x, y, z, w = value['quat']
    return Matrix.LocRotScale(Vector(value['loc']), Quaternion((w, x, y, z)), Vector(value['scale']))


def read_capture(path):
    data = json.loads(Path(path).read_text())
    bones = {b['name']: b for b in data['bones']}
    if len(bones) != data['bone_count'] or len(bones) != len(data['bones']):
        raise ValueError('Duplicate or missing captured names')
    if data['parents_unresolved'] or data['parents_authoritative'] != len(bones):
        raise ValueError('Every captured parent must be authoritative')
    errors = []
    for bone in bones.values():
        if bone['parent_source'] != 'authoritative':
            raise ValueError('Unresolved parent: ' + bone['name'])
        if any(abs(x - 1) > 1e-6 for x in bone['local']['scale'] + bone['world']['scale']):
            raise ValueError('Non-unit captured bone scales are not supported')
        composed = transform(bone['local'])
        if bone['parent']:
            composed = transform(bones[bone['parent']]['world']) @ composed
        world = transform(bone['world'])
        errors.append(max(abs(composed[i][j] - world[i][j]) for i in range(4) for j in range(4)))
    if max(errors) > .002:
        raise ValueError('Captured local/world transforms do not compose consistently')
    return data, bones, max(errors)


def calibrate(rig, bones):
    """Choose the signed axis permutation by measured points, not a hardcoded pose."""
    if set(bones) != set(rig.data.bones.keys()) | {rig.name}:
        raise ValueError('Calibration FBX and source capture bone names differ')
    for bone in rig.data.bones:
        expected = bone.parent.name if bone.parent else rig.name
        if bones[bone.name]['parent'] != expected:
            raise ValueError('Calibration parent mismatch: ' + bone.name)
    fits = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = Matrix.Identity(4)
            for i in range(3):
                for j in range(3):
                    matrix[i][j] = .01 * signs[i] if permutation[i] == j else 0
            matrix.translation = rig.matrix_world.translation - matrix @ Vector(bones[rig.name]['world']['loc'])
            errors = [(rig.matrix_world @ b.head_local - matrix @ Vector(bones[b.name]['world']['loc'])).length
                      for b in rig.data.bones]
            fits.append((sum(e * e for e in errors), max(errors), matrix))
    fits.sort(key=lambda row: row[0])
    if fits[0][1] > 2e-5:
        raise ValueError('Source capture does not match the calibration FBX: %.8f m' % fits[0][1])
    return fits[0][2], fits[0][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-fbx', type=Path, required=True)
    parser.add_argument('--source-capture', type=Path, required=True)
    parser.add_argument('--target-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        parser.error('Use a new output directory')
    source_data, source_bones, source_error = read_capture(args.source_capture)
    target_data, target_bones, target_error = read_capture(args.target_capture)
    if set(source_bones) != set(target_bones):
        raise ValueError('This reconstruction requires matching complete bone sets')
    if any(a['parent'] != target_bones[n]['parent'] for n, a in source_bones.items()):
        raise ValueError('Captured source/target hierarchies differ')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.
    bpy.ops.import_scene.fbx(filepath=str(args.source_fbx), use_anim=False, automatic_bone_orientation=False)
    bpy.context.view_layer.update()
    rigs = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if len(rigs) != 1:
        raise ValueError('Expected exactly one calibration armature')
    source = rigs[0]
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH'
              and any(m.type == 'ARMATURE' and m.object == source for m in o.modifiers)]
    matrix, calibration_error = calibrate(source, source_bones)
    # Purposefully use the wrong pose as the calibration. It must be rejected.
    negative_control = None
    if source_data != target_data:
        try:
            calibrate(source, target_bones)
        except ValueError as exc:
            negative_control = str(exc)
        if not negative_control:
            raise ValueError('Wrong-capture negative control did not fail')
    inverse = matrix.inverted()
    targets = {}
    for name, bone in source_bones.items():
        old = source.matrix_world if name == source.name else source.matrix_world @ source.data.bones[name].matrix_local
        delta = transform(target_bones[name]['world']) @ transform(bone['world']).inverted()
        targets[name] = matrix @ delta @ inverse @ old
    reference = source.copy()
    reference.data = source.data.copy()
    bpy.context.scene.collection.objects.link(reference)
    reference.matrix_world = targets[source.name]
    bpy.ops.object.select_all(action='DESELECT')
    reference.select_set(True)
    bpy.context.view_layer.objects.active = reference
    bpy.ops.object.mode_set(mode='EDIT')
    rig_inverse = reference.matrix_world.inverted()
    lengths = {b.name: b.length for b in reference.data.edit_bones}
    for bone in reference.data.edit_bones:
        bone.use_connect = False
    for bone in reference.data.edit_bones:
        bone.matrix = rig_inverse @ targets[bone.name]
        bone.length = lengths[bone.name]
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
    script = Path(__file__).resolve().parents[2] / 'braindead_blender/target_conversion.py'
    spec = importlib.util.spec_from_file_location('target_conversion', script)
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)
    # Internal authoring step. Provenance is emitted after FBX creation, then
    # independent Unreal verification establishes its native-reference accuracy.
    result = convert.rebind_copy(source, reference, meshes, profile='FAB_UEFN')
    rig = result['rig']
    objects = result['meshes']
    root_name = source.name
    keep = set(objects + [rig])
    for obj in list(bpy.data.objects):
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)
    rig.name = root_name
    rig['bdb_target_profile'] = 'NATIVE_DEVICE'
    args.output.mkdir(parents=True, exist_ok=False)
    fbx = args.output / 'CP_Device_Mannequin_named.fbx'
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'CP_Device_Mannequin_named.blend'))
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects + [rig]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.fbx(filepath=str(fbx), use_selection=True, object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=False, use_armature_deform_only=False, add_leaf_bones=False,
        primary_bone_axis='Y', secondary_bone_axis='X', armature_nodetype='NULL', bake_anim=False,
        apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL', use_space_transform=True,
        bake_space_transform=False, axis_forward='-Y', axis_up='Z', global_scale=1.,
        mesh_smooth_type='FACE', path_mode='AUTO', use_custom_props=False, colors_type='LINEAR')
    provenance = {'asset_path': target_data['engine_asset_path'], 'fbx_sha256': sha(fbx),
                  'armature_object_name': rig.name, 'bone_count_in_engine': len(target_bones),
                  'method': 'Complete editor capture reconstructed through measured FBX bone frames',
                  'not_a_direct_engine_mesh_export': True, 'source_fbx': str(args.source_fbx),
                  'source_fbx_sha256': sha(args.source_fbx), 'source_capture': str(args.source_capture),
                  'source_capture_sha256': sha(args.source_capture), 'target_capture': str(args.target_capture),
                  'target_capture_sha256': sha(args.target_capture),
                  'UE_cm_to_Blender_m': [list(row) for row in matrix],
                  'calibration_max_position_error_m': calibration_error,
                  'source_local_world_max_component_error': source_error,
                  'target_local_world_max_component_error': target_error,
                  'wrong_capture_rejected': negative_control,
                  'engine_reference_verification': 'PENDING', 'engine_playback_verified': False}
    (args.output / 'CP_Device_Mannequin_named.provenance.json').write_text(json.dumps(provenance, indent=2))
    print('NATIVE_REFERENCE_BUILT', json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
