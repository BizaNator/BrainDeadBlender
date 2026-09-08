"""Create a separately bound copy using the target's full reference matrices.

This transforms Basis and every shape key, carries exact target hierarchy/rest
data, and preserves original objects. Native profiles require a hashed reference
with its measured in-engine asset provenance. Playback remains a separate gate.
"""
import hashlib
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

PROFILES = {
    'NATIVE_DEVICE': ('Native Character Device', '/Game/Creative/Devices/Mannequin/Meshes/CP_Device_Mannequin_Skeleton'),
    'NATIVE_PLAYER': ('Native Player (280 physical bones)', '/Game/Characters/Player/Male/Male_Avg_Base/Fortnite_M_Avg_Player_Skeleton'),
    'FAB_UEFN': ('Fab UEFN mannequin (legacy)', None),
}


def reference_fingerprint(reference):
    rows = [(b.name, b.parent.name if b.parent else None,
             [list(r) for r in reference.matrix_world @ b.matrix_local])
            for b in reference.data.bones]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def attach_provenance(reference, fbx_path, contract_path, profile):
    """Validate an exported reference's source declaration and hash before use."""
    if profile not in PROFILES:
        raise ValueError('Unknown target profile: ' + profile)
    fbx_path = Path(fbx_path).resolve()
    sha = hashlib.sha256(fbx_path.read_bytes()).hexdigest()
    contract = json.loads(Path(contract_path).read_text()) if contract_path else {}
    if profile != 'FAB_UEFN':
        if contract.get('asset_path') != PROFILES[profile][1]:
            raise ValueError('Native reference provenance must identify ' + PROFILES[profile][1])
        if contract.get('fbx_sha256') != sha:
            raise ValueError('Native reference provenance hash does not match its FBX')
        if not contract.get('armature_object_name'):
            raise ValueError('Native provenance needs the original armature_object_name for export')
    contract.update({'profile': profile, 'fbx_path': str(fbx_path), 'fbx_sha256': sha,
                     'loaded_rest_fingerprint': reference_fingerprint(reference)})
    if profile == 'FAB_UEFN':
        contract.setdefault('armature_object_name', 'root')
    reference['bdb_reference_contract'] = json.dumps(contract, sort_keys=True)
    return contract


def _preflight(source, reference, meshes, profile, bone_map):
    if profile not in PROFILES:
        raise ValueError('Unknown target profile: ' + profile)
    if bpy.context.mode != 'OBJECT':
        raise ValueError('Switch to Object Mode before conversion')
    if source.type != 'ARMATURE' or reference.type != 'ARMATURE' or source == reference:
        raise ValueError('Choose different source and reference armatures')
    if not meshes:
        raise ValueError('No meshes are bound to the source')
    contract = json.loads(reference.get('bdb_reference_contract', '{}'))
    if profile != 'FAB_UEFN':
        if contract.get('profile') != profile or contract.get('asset_path') != PROFILES[profile][1]:
            raise ValueError('Native reference provenance is missing or identifies a different profile')
        if contract.get('loaded_rest_fingerprint') != reference_fingerprint(reference):
            raise ValueError('Native reference changed after its provenance was attached; reload it')
        fbx = Path(contract.get('fbx_path', ''))
        if not fbx.is_file() or hashlib.sha256(fbx.read_bytes()).hexdigest() != contract.get('fbx_sha256'):
            raise ValueError('Native reference provenance file hash is stale')
    if reference.constraints or any(b.constraints for b in reference.pose.bones):
        raise ValueError('Reference must be a static imported rig without constraints')
    source_world = {b.name: source.matrix_world @ b.matrix_local for b in source.data.bones}
    target_world = {b.name: reference.matrix_world @ b.matrix_local for b in reference.data.bones}
    if not source_world or not target_world:
        raise ValueError('Empty reference or source rig')
    for matrix in [source.matrix_world, reference.matrix_world, *source_world.values(), *target_world.values()]:
        if not all(math.isfinite(x) for row in matrix for x in row) or abs(matrix.determinant()) < 1e-12:
            raise ValueError('Non-finite or singular reference transform')
    prepared = []
    for obj in meshes:
        if obj.type != 'MESH' or not obj.data.vertices or not obj.data.polygons:
            raise ValueError('Expected nonempty mesh: ' + obj.name)
        if abs(obj.matrix_world.determinant()) < 1e-12:
            raise ValueError('Singular mesh transform: ' + obj.name)
        arms = [m for m in obj.modifiers if m.type == 'ARMATURE']
        if len(arms) != 1 or arms[0].object != source or arms[0].use_deform_preserve_volume:
            raise ValueError('Expected one linear Armature modifier bound to the source: ' + obj.name)
        if obj.parent and obj.parent != source:
            raise ValueError('Detach other object parents with transforms preserved before conversion: ' + obj.name)
        if obj.parent_type == 'BONE':
            raise ValueError('Bone-parented mesh is unsupported: ' + obj.name)
        coordinates = [v.co for v in obj.data.vertices]
        if obj.data.shape_keys:
            coordinates.extend(v.co for k in obj.data.shape_keys.key_blocks for v in k.data)
        if not all(math.isfinite(x) for co in coordinates for x in co):
            raise ValueError('Non-finite mesh or morph coordinates: ' + obj.name)
        names = {g.index: g.name for g in obj.vertex_groups}
        mapped = {i: bone_map.get(name, name) for i, name in names.items()}
        if len(set(mapped.values())) != len(mapped):
            raise ValueError('Bone mapping must be one-to-one: ' + obj.name)
        deltas = {}
        weights = []
        for vertex in obj.data.vertices:
            used = []
            for group in vertex.groups:
                if not math.isfinite(group.weight) or group.weight < 0:
                    raise ValueError('Invalid weight on ' + obj.name)
                if group.weight <= 1e-8:
                    continue
                name = names[group.group]
                target_name = mapped[group.group]
                if name not in source_world or target_name not in target_world:
                    raise ValueError('Missing weighted reference bone: ' + name + ' -> ' + target_name)
                if not reference.data.bones[target_name].use_deform:
                    raise ValueError('Weighted reference bone has deformation disabled: ' + target_name)
                if target_name not in deltas:
                    deltas[target_name] = target_world[target_name] @ source_world[name].inverted()
                used.append((target_name, group.weight))
            if not used or abs(sum(w for _, w in used) - 1) > 1e-4:
                raise ValueError('Unweighted or unnormalized vertex on ' + obj.name)
            weights.append(used)
        prepared.append((obj, mapped, deltas, weights))
    return contract, prepared


def rebind_copy(source, reference, meshes, *, profile, bone_map=None):
    """Repose a skinned surface and all morph coordinates onto an actual target.

    Linear skinning of each coordinate uses sum(weight * target_rest *
    inverse(source_rest) * coordinate). Bone roll is included. This is a reference
    conversion, not an animation retargeter or a cure for arbitrary bad weights.
    """
    meshes = list(meshes)
    contract, prepared = _preflight(source, reference, meshes, profile, bone_map or {})
    collection = bpy.data.collections.new('BDB_' + profile)
    bpy.context.scene.collection.children.link(collection)
    created = []
    try:
        rig = reference.copy()
        rig.data = reference.data.copy()
        rig.data.pose_position = 'POSE'
        rig.animation_data_clear()
        world = reference.matrix_world.copy()
        rig.parent = None
        rig.matrix_world = world
        rig.name = 'BDB_' + profile + '_Rig'
        collection.objects.link(rig)
        created.append(rig)
        for bone in rig.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        for name in source.keys():
            if isinstance(source[name], (float, int, bool)):
                rig[name] = source[name]
                rig.id_properties_ui(name).update_from(source.id_properties_ui(name))
        rig['bdb_target_profile'] = profile
        rig['bdb_export_armature_name'] = contract.get('armature_object_name', 'root')
        converted, changes = [], []
        for original, mapped, deltas, weights in prepared:
            obj = original.copy()
            obj.data = original.data.copy()
            obj.name = original.name + '__' + profile
            world = original.matrix_world.copy()
            obj.parent = rig if original.parent == source else None
            obj.matrix_world = world
            collection.objects.link(obj)
            created.append(obj)
            inverse = world.inverted()

            def transform(coordinate, index):
                point = world @ coordinate
                result = Vector((0, 0, 0))
                for bone, weight in weights[index]:
                    result += (deltas[bone] @ point) * weight
                return inverse @ result

            basis_positions = [transform(v.co, v.index) for v in original.data.vertices]
            if original.data.shape_keys:
                if obj.data.shape_keys == original.data.shape_keys:
                    raise ValueError('Shape keys were not independently copied')
                for key in original.data.shape_keys.key_blocks:
                    output_key = obj.data.shape_keys.key_blocks[key.name]
                    for index, datum in enumerate(key.data):
                        output_key.data[index].co = transform(datum.co, index)
                animation = obj.data.shape_keys.animation_data
                if animation:
                    for driver in animation.drivers:
                        for variable in driver.driver.variables:
                            for target in variable.targets:
                                if target.id == source:
                                    target.id = rig
                                elif target.id == original:
                                    target.id = obj
            for vertex, coordinate in zip(obj.data.vertices, basis_positions):
                vertex.co = coordinate
            for group in obj.vertex_groups:
                group.name = '__BDB_TMP_' + str(group.index)
            for group in obj.vertex_groups:
                group.name = mapped[group.index]
            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE':
                    modifier.object = rig
            obj.data.update()
            obj['bdb_source_object'] = original.name
            obj['bdb_target_profile'] = profile
            changes.append(max(((world @ a.co) - (world @ b.co)).length
                               for a, b in zip(original.data.vertices, obj.data.vertices)))
            converted.append(obj)
        bpy.context.view_layer.update()
        report = {'profile': profile, 'reference_contract': contract,
                  'source_rest_fingerprint': reference_fingerprint(source),
                  'target_rest_fingerprint': reference_fingerprint(reference),
                  'output_rest_fingerprint': reference_fingerprint(rig),
                  'bone_count': len(rig.data.bones), 'mesh_count': len(converted),
                  'max_basis_delta_m': max(changes) * bpy.context.scene.unit_settings.scale_length,
                  'all_shape_keys_transformed': True, 'originals_preserved': True,
                  'engine_playback_verified': False}
        if report['target_rest_fingerprint'] != report['output_rest_fingerprint']:
            raise ValueError('Output rig does not preserve the reference rest transforms')
        return {'rig': rig, 'meshes': converted, 'collection': collection, 'report': report}
    except Exception:
        for obj in reversed(created):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)
        raise
