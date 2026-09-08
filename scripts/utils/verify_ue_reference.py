"""Run in the isolated UE commandlet; compare fresh FBX imports to editor capture.

Set BDB_REFERENCE_FBX, BDB_REFERENCE_CAPTURE, BDB_REFERENCE_REPORT and optional
BDB_CANDIDATE_FBX. Uses only the commandlet's private /Game/BDBReference_* assets.
Do not run unmodified in the owner's shared UEFN editor.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import unreal


def import_mesh(path, destination, skeleton=None):
    options = unreal.FbxImportUI()
    values = {'automated_import_should_detect_type': False,
              'mesh_type_to_import': unreal.FBXImportType.FBXIT_SKELETAL_MESH,
              'import_as_skeletal': True, 'import_mesh': True, 'import_animations': False,
              'import_materials': False, 'import_textures': False, 'create_physics_asset': False}
    for name, value in values.items():
        options.set_editor_property(name, value)
    if skeleton:
        options.set_editor_property('skeleton', skeleton)
    data = options.get_editor_property('skeletal_mesh_import_data')
    for name, value in {'import_morph_targets': True, 'update_skeleton_reference_pose': False,
                        'use_t0_as_ref_pose': False, 'import_uniform_scale': 1.,
                        'convert_scene': True, 'convert_scene_unit': True, 'force_front_x_axis': False,
                        'preserve_smoothing_groups': True, 'import_meshes_in_bone_hierarchy': True}.items():
        data.set_editor_property(name, value)
    data.set_editor_property('normal_import_method', unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS)
    data.set_editor_property('vertex_color_import_option', unreal.VertexColorImportOption.REPLACE)
    task = unreal.AssetImportTask()
    task.filename = str(path)
    task.destination_path = destination
    task.automated = True
    task.replace_existing = True
    task.replace_existing_settings = True
    task.save = True
    task.options = options
    task.factory = unreal.FbxFactory()
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    meshes = [o for o in task.get_objects() if isinstance(o, unreal.SkeletalMesh)]
    if len(meshes) != 1:
        raise ValueError('Expected one nonempty imported skeletal mesh')
    if unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem).get_num_verts(meshes[0], 0) <= 0:
        raise ValueError('Empty imported mesh')
    return meshes[0]


def skeleton_data(mesh):
    skeleton = mesh.get_editor_property('skeleton')
    pose = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
    names = unreal.AnimPoseExtensions.get_bone_names(pose)
    component = unreal.SkeletalMeshComponent()
    component.set_skinned_asset_and_update(mesh)
    bones = []
    for name in names:
        tr = unreal.AnimPoseExtensions.get_ref_bone_pose(pose, name, unreal.AnimPoseSpaces.LOCAL)
        parent = str(component.get_parent_bone(name))
        bones.append({'name': str(name), 'parent': None if parent == 'None' else parent,
                      'local': {'loc': [tr.translation.x, tr.translation.y, tr.translation.z],
                                'quat': [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w],
                                'scale': [tr.scale3d.x, tr.scale3d.y, tr.scale3d.z]}})
    return {'engine_asset_path': skeleton.get_path_name(), 'bones': bones}


def compare(expected, actual):
    left = {b['name']: b for b in expected['bones']}
    right = {b['name']: b for b in actual['bones']}
    errors = []
    if set(left) != set(right):
        errors.append('Bone names differ')
    if list(left) != list(right):
        errors.append('Bone order differs')
    per_bone = {}
    for name in set(left) & set(right):
        a, b = left[name], right[name]
        if a['parent'] != b['parent']:
            errors.append('Parent mismatch: ' + name)
        qa, qb = a['local']['quat'], b['local']['quat']
        dot = abs(sum(x*y for x, y in zip(qa, qb))) / math.sqrt(sum(x*x for x in qa) * sum(x*x for x in qb))
        per_bone[name] = {'translation_cm': math.dist(a['local']['loc'], b['local']['loc']),
                          'rotation_deg': math.degrees(2*math.acos(min(1., dot))),
                          'scale': max(abs(x-y) for x, y in zip(a['local']['scale'], b['local']['scale']))}
    maxima = {key: max([p[key] for p in per_bone.values()], default=float('inf'))
              for key in ('translation_cm', 'rotation_deg', 'scale')}
    # Match the Blender contract's 2e-5 scale tolerance. The already-approved
    # v01 independently imported control has a 1.70469e-5 FBX scale drift at
    # ik_foot_r; the native reference measures 1.69277e-5 at that same bone.
    tolerances = {'translation_cm': .002, 'rotation_deg': .02, 'scale': 2e-5}
    for key, limit in tolerances.items():
        if maxima[key] > limit:
            errors.append(key + ' exceeds tolerance')
    return {'pass': not errors, 'errors': errors, 'maxima': maxima,
            'tolerances': tolerances, 'per_bone': per_bone}


def main():
    fbx = Path(os.environ['BDB_REFERENCE_FBX'])
    capture = Path(os.environ['BDB_REFERENCE_CAPTURE'])
    output = Path(os.environ['BDB_REFERENCE_REPORT'])
    expected = json.loads(capture.read_text())
    destination = '/Game/BDBReference_' + uuid.uuid4().hex[:12]
    report = {'pass': False, 'engine': unreal.SystemLibrary.get_engine_version(),
              'reference_fbx': str(fbx), 'reference_fbx_sha256': hashlib.sha256(fbx.read_bytes()).hexdigest(),
              'capture': str(capture), 'capture_sha256': hashlib.sha256(capture.read_bytes()).hexdigest(),
              'actual_native_engine_object_available_in_this_project': False,
              'comparison_target': expected['engine_asset_path'], 'runtime_playback_verified': False}
    try:
        reference = import_mesh(fbx, destination + '/Reference')
        actual = skeleton_data(reference)
        report['fresh_reference_import'] = actual
        report['reference_comparison'] = compare(expected, actual)
        broken = copy.deepcopy(actual)
        next(b for b in broken['bones'] if b['name'] == 'upperarm_l')['local']['loc'][0] += .1
        report['deliberate_1mm_error_rejected'] = not compare(expected, broken)['pass']
        assert report['deliberate_1mm_error_rejected']
        assert report['reference_comparison']['pass'], report['reference_comparison']['errors']
        candidate = os.environ.get('BDB_CANDIDATE_FBX')
        if candidate:
            independent = import_mesh(candidate, destination + '/CandidateIndependent')
            own = skeleton_data(independent)
            report['candidate_fbx_sha256'] = hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
            report['independent_candidate_import'] = own
            report['candidate_comparison'] = compare(expected, own)
            assigned = import_mesh(candidate, destination + '/CandidateAssigned', reference.get_editor_property('skeleton'))
            report['same_skeleton_object'] = assigned.get_editor_property('skeleton') == reference.get_editor_property('skeleton')
            report['reference_unchanged_by_assignment'] = actual == skeleton_data(reference)
            report['morphs'] = [str(n) for n in assigned.get_all_morph_target_names()]
            report['materials'] = [str(m.material_slot_name) for m in assigned.get_editor_property('materials')]
            report['bounds_cm'] = [getattr(assigned.get_imported_bounds().box_extent, axis)*2 for axis in ('x', 'y', 'z')]
            assert report['candidate_comparison']['pass'], report['candidate_comparison']['errors']
            assert report['same_skeleton_object'] and report['reference_unchanged_by_assignment']
            assert {'BodyMass', 'Muscular', 'Lean', 'Belly', 'ChestWidth', 'HipWidth', 'ShoulderMass', 'HeadWidth'}.issubset(report['morphs'])
            assert 'Nails' in report['materials']
            assert 165 < report['bounds_cm'][2] < 180
        report['pass'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))
        unreal.log('BDB_REFERENCE_RESULT ' + json.dumps({k: v for k, v in report.items() if k not in ('fresh_reference_import', 'independent_candidate_import', 'reference_comparison', 'candidate_comparison')}))


if __name__ == '__main__':
    main()
