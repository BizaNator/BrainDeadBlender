"""Blender CPU CLI: convert a copy, save .blend, export and audit full/part FBXs.

Usage: blender -b --python-exit-code 1 --python this.py -- --help
The native reference FBX must have provenance tied to the measured UE reference.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import bpy


def module(name):
    path = Path(__file__).resolve().parents[2] / 'braindead_blender' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    convert, audit = module('target_conversion'), module('rig_contract')
    scale_export = module('fbx_reference_scale')
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source-blend', type=Path)
    inputs.add_argument('--source-fbx', type=Path)
    parser.add_argument('--source-rig', default='root')
    parser.add_argument('--reference-fbx', required=True, type=Path)
    parser.add_argument('--reference-contract', type=Path)
    parser.add_argument('--profile', default='NATIVE_DEVICE', choices=list(convert.PROFILES))
    parser.add_argument('--asset-name', default='COB_Male_Base')
    parser.add_argument('--require-fingers', action='store_true',
                        help='Full characters: reject missing weights on any of the 30 phalanges')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; use a fresh directory')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', args.asset_name):
        parser.error('Asset name must use letters, digits and underscores')
    source_path = args.source_blend or args.source_fbx
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if args.source_blend:
        bpy.ops.wm.open_mainfile(filepath=str(source_path))
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.unit_settings.scale_length = 1.
        bpy.ops.import_scene.fbx(filepath=str(source_path), use_anim=False,
                                 automatic_bone_orientation=False, use_image_search=False)
    source = bpy.data.objects[args.source_rig]
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH'
              and any(m.type == 'ARMATURE' and m.object == source for m in o.modifiers)]
    source_mesh_contracts = {m['name']: m for m in audit.snapshot(source, meshes)['meshes']}
    if args.require_fingers:
        weighted = {n for m in source_mesh_contracts.values() for n, w in m['weight_totals'].items() if w > 1e-6}
        missing = sorted(set(audit.fingers()) - weighted)
        if missing:
            raise ValueError('Source needs actual finger weights before native completion: ' + ', '.join(missing))
    previous = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(args.reference_fbx), use_anim=False,
                             automatic_bone_orientation=False, use_image_search=False)
    bpy.context.view_layer.update()
    references = [o for o in set(bpy.data.objects) - previous if o.type == 'ARMATURE']
    if len(references) != 1:
        raise ValueError('Expected exactly one named reference armature')
    reference = references[0]
    contract = convert.attach_provenance(reference, args.reference_fbx, args.reference_contract, args.profile)
    reference_snapshot = audit.snapshot(reference)
    result = convert.rebind_copy(source, reference, meshes, profile=args.profile, bone_map=contract.get('bone_map'))
    rig, converted = result['rig'], result['meshes']
    visible = [o for o in converted if not o.hide_render]
    if not visible:
        raise ValueError('No default-visible mesh for combined export')
    # This dedicated background scene becomes the new authoring file. Input files
    # remain untouched; removing its original in-memory objects frees exact names.
    keep = set(converted + [rig])
    for obj in list(bpy.data.objects):
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)
    rig.name = contract['armature_object_name']
    for obj in converted:
        obj.name = obj['bdb_source_object']
    output.mkdir(parents=True, exist_ok=False)
    exports = output / 'exports'
    exports.mkdir()
    base_name = args.asset_name + '_' + args.profile
    expected_exports = {}
    bpy.ops.wm.save_as_mainfile(filepath=str(output / (base_name + '.blend')))

    def export(path, selected):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            obj.hide_set(False)
            obj.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.export_scene.fbx(filepath=str(path), use_selection=True, object_types={'ARMATURE', 'MESH'},
            use_mesh_modifiers=False, use_armature_deform_only=False, add_leaf_bones=False,
            primary_bone_axis='Y', secondary_bone_axis='X', armature_nodetype='NULL', bake_anim=False,
            apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL', use_space_transform=True,
            bake_space_transform=False, axis_forward='-Y', axis_up='Z', global_scale=1.,
            mesh_smooth_type='FACE', path_mode='COPY', embed_textures=False, use_custom_props=False, colors_type='LINEAR')
        scale_export.restore_reference_bind(path, contract['fbx_path'], contract['fbx_sha256'])

    bpy.ops.object.select_all(action='DESELECT')
    copies = []
    for obj in visible:
        clone = obj.copy()
        clone.data = obj.data.copy()
        bpy.context.collection.objects.link(clone)
        clone.hide_set(False)
        clone.select_set(True)
        copies.append(clone)
    bpy.context.view_layer.objects.active = copies[0]
    bpy.ops.object.join()
    combined = bpy.context.object
    combined.name = base_name
    full = exports / ('SK_' + base_name + '.fbx')
    expected_exports[full.name] = [source_mesh_contracts[o['bdb_source_object']] for o in visible]
    export(full, [rig, combined])
    bpy.data.objects.remove(combined, do_unlink=True)
    for obj in converted:
        path = exports / ('SK_' + base_name + '_' + obj.name + '.fbx')
        expected_exports[path.name] = [source_mesh_contracts[obj['bdb_source_object']]]
        export(path, [rig, obj])
    reports = {}
    for path in sorted(exports.glob('*.fbx')):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(path), use_image_search=False)
        arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
        objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        expected = expected_exports[path.name]
        names = {n for m in expected for n, weight in m['weight_totals'].items() if weight > 1e-6}
        names = {contract.get('bone_map', {}).get(n, n) for n in names}
        morphs = {n for m in expected for n in m['morphs']}
        report = audit.compare(reference_snapshot, audit.snapshot(arm, objects),
                               required_weighted=names, required_morphs=morphs)
        report['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        reports[path.name] = report
        if not report['structural_pass']:
            (output / 'export_audits.json').write_text(json.dumps(reports, indent=2))
            raise ValueError(str(report['errors']))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    result['report'].update({'source_asset': str(source_path), 'source_asset_sha256': source_hash,
                            'source_format': 'blend' if args.source_blend else 'fbx',
                            'source_blend': str(args.source_blend) if args.source_blend else None,
                            'source_blend_sha256': source_hash if args.source_blend else None,
                            'primary_fbx': full.name, 'full_fbx_sha256': reports[full.name]['sha256'],
                            'fbx_count': len(reports), 'export_root_name': contract['armature_object_name']})
    (output / 'conversion.json').write_text(json.dumps(result['report'], indent=2))
    (output / 'export_audits.json').write_text(json.dumps(reports, indent=2))
    print('TARGET_EXPORT_PASS', args.profile, len(reports), output, flush=True)


if __name__ == '__main__':
    main()
