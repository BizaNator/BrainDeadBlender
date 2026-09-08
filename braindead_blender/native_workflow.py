"""Connect the measured native conversion to the existing AutoRig/export buttons."""
import json
from pathlib import Path

import bpy

from . import rig_contract, target_conversion, fbx_reference_scale
from .reference_paths import reference_paths as resolve_reference_paths


def reference_paths(profile='NATIVE_DEVICE'):
    # Invalid/missing shared data must not prevent the entire add-on loading.
    # Empty fields force an explicit valid reference before conversion/export.
    try:
        fbx, contract = resolve_reference_paths(profile)
    except (OSError, ValueError):
        return '', ''
    return str(fbx), str(contract) if contract else ''


def prepare_export_collection(context, fbx, provenance, profile='NATIVE_DEVICE'):
    """Replace the active Export collection with a native copy; retain originals."""
    export = bpy.data.collections.get('Export')
    rigs = [o for o in export.all_objects if o.type == 'ARMATURE'] if export else []
    if len(rigs) != 1:
        raise ValueError('Export must contain exactly one source armature')
    source = rigs[0]
    meshes = [o for o in export.all_objects if o.type == 'MESH'
              and any(m.type == 'ARMATURE' and m.object == source for m in o.modifiers)]
    if not Path(fbx).is_file() or not Path(provenance).is_file():
        raise ValueError('Choose the measured native reference FBX and provenance JSON')
    before = set(bpy.data.objects)
    imported = set()
    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx), use_anim=False, automatic_bone_orientation=False, use_image_search=False)
        imported = set(bpy.data.objects) - before
        references = [o for o in imported if o.type == 'ARMATURE']
        if len(references) != 1:
            raise ValueError('Expected one named native reference rig')
        reference = references[0]
        contract = target_conversion.attach_provenance(reference, fbx, provenance, profile)
        if source.get('bdb_target_profile') == profile:
            result = rig_contract.compare(rig_contract.snapshot(reference), rig_contract.snapshot(source, meshes))
            if not result['structural_pass']:
                raise ValueError('Native rig changed from its reference: ' + str(result['errors']))
            return source
        result = target_conversion.rebind_copy(source, reference, meshes, profile=profile, bone_map=contract.get('bone_map'))
    finally:
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
    export.name = 'Export_BeforeNative'
    export.hide_viewport = True
    export.hide_render = True
    result['collection'].name = 'Export'
    text = bpy.data.texts.new('BDB_Native_Conversion.json')
    text.write(json.dumps(result['report'], indent=2))
    return result['rig']


def export_native_copy(context, rig, meshes, path):
    """Export a neutral combined mesh and exact root name without altering sources."""
    meshes = [o for o in meshes if not o.hide_render]
    if not meshes or rig.get('bdb_target_profile') not in ('NATIVE_DEVICE', 'NATIVE_PLAYER'):
        raise ValueError('Prepare a native Export collection first')
    contract = json.loads(rig.get('bdb_reference_contract', '{}'))
    if target_conversion.reference_fingerprint(rig) != contract.get('loaded_rest_fingerprint'):
        raise ValueError('Native rest matrices changed after conversion; reload the reference')
    reference = rig_contract.snapshot(rig, meshes)
    if any(m['invalid_weights'] or m['unweighted_vertices'] for m in reference['meshes']):
        raise ValueError('Invalid or missing skin weights')
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    created = []
    occupied = bpy.data.objects.get(contract['armature_object_name'])
    occupied_name = occupied.name if occupied else None
    try:
        bpy.ops.object.select_all(action='DESELECT')
        arm = rig.copy()
        arm.data = rig.data.copy()
        arm.animation_data_clear()
        context.scene.collection.objects.link(arm)
        created.append(arm)
        if occupied:
            occupied.name = occupied_name + '_ExportSource'
        arm.name = contract['armature_object_name']
        for bone in arm.pose.bones:
            bone.matrix_basis.identity()
        copies = []
        for original in meshes:
            obj = original.copy()
            obj.data = original.data.copy()
            world = obj.matrix_world.copy()
            obj.parent = arm if original.parent == rig else None
            obj.matrix_world = world
            if obj.data.shape_keys:
                obj.data.shape_keys.animation_data_clear()
                for key in obj.data.shape_keys.key_blocks:
                    key.value = 0.
            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE':
                    modifier.object = arm
            context.scene.collection.objects.link(obj)
            created.append(obj)
            obj.hide_viewport = False
            obj.hide_set(False)
            obj.select_set(True)
            copies.append(obj)
        context.view_layer.objects.active = copies[0]
        bpy.ops.object.join()
        combined = context.object
        combined.name = Path(path).stem
        arm.select_set(True)
        context.view_layer.objects.active = arm
        bpy.ops.export_scene.fbx(filepath=str(path), use_selection=True, object_types={'ARMATURE', 'MESH'},
            use_mesh_modifiers=False, use_armature_deform_only=False, add_leaf_bones=False,
            primary_bone_axis='Y', secondary_bone_axis='X', armature_nodetype='NULL', bake_anim=False,
            apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL', use_space_transform=True,
            bake_space_transform=False, axis_forward='-Y', axis_up='Z', global_scale=1.,
            mesh_smooth_type='FACE', path_mode='COPY', embed_textures=False, use_custom_props=False, colors_type='LINEAR')
        fbx_reference_scale.restore_reference_bind(path, contract['fbx_path'], contract['fbx_sha256'])
    finally:
        for obj in reversed(created):
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except ReferenceError:
                pass  # Joined copies have already been removed by Blender.
        if occupied:
            occupied.name = occupied_name
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active
