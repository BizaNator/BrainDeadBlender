"""Bake same-hierarchy Fab FBX animations onto the native Device contract.

Run in a dedicated Unreal editor with -NullRHI -ExecutePythonScript, not a
commandlet: IKRetargetBatchOperation requires Slate. BDB_ANIMATION_JOB points
to a JSON job. See docs/native-device-animation-bake.md for the job schema.
No source animation, character, shared skeleton, or live editor is modified.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import traceback
import unreal


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_file(item):
    path = Path(item['path']).resolve()
    if sha(path) != item['sha256']:
        raise ValueError('SHA256 mismatch: ' + str(path))
    return path


def angle(a, b):
    a = [a.x,a.y,a.z,a.w] if isinstance(a,unreal.Quat) else a
    b = [b.x,b.y,b.z,b.w] if isinstance(b,unreal.Quat) else b
    dot = abs(sum(x*y for x,y in zip(a,b))) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return math.degrees(2*math.acos(min(1.0,dot)))


def set_properties(obj, values):
    for key,value in values.items(): obj.set_editor_property(key,value)


def save(asset):
    assert unreal.EditorAssetLibrary.save_loaded_asset(asset,only_if_is_dirty=False), asset.get_path_name()


def import_fbx(path, folder, name, skeleton=None):
    assert not unreal.EditorAssetLibrary.does_asset_exist(folder+'/'+name)
    animation = skeleton is not None
    ui = unreal.FbxImportUI()
    set_properties(ui,{'automated_import_should_detect_type':False,
        'mesh_type_to_import':unreal.FBXImportType.FBXIT_ANIMATION if animation else unreal.FBXImportType.FBXIT_SKELETAL_MESH,
        'import_mesh':not animation,'import_animations':animation,'import_materials':False,'import_textures':False})
    if animation:
        ui.set_editor_property('skeleton',skeleton)
        data = ui.get_editor_property('anim_sequence_import_data')
        set_properties(data,{'animation_length':unreal.FBXAnimationLengthImportType.FBXALIT_EXPORTED_TIME,
                             'use_default_sample_rate':False})
    else:
        set_properties(ui,{'import_as_skeletal':True,'create_physics_asset':False})
        data = ui.get_editor_property('skeletal_mesh_import_data')
        set_properties(data,{'update_skeleton_reference_pose':False,'use_t0_as_ref_pose':False,
                             'import_uniform_scale':1.0,'import_morph_targets':False})
    set_properties(data,{'convert_scene':True,'convert_scene_unit':True,'force_front_x_axis':False})
    task = unreal.AssetImportTask()
    set_properties(task,{'filename':str(path),'destination_path':folder,'destination_name':name,
        'automated':True,'replace_existing':False,'save':True,'options':ui,'factory':unreal.FbxFactory()})
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    kind = unreal.AnimSequence if animation else unreal.SkeletalMesh
    objects = [o for o in task.get_objects() if isinstance(o,kind)]
    assert len(objects)==1, task.imported_object_paths
    obj = objects[0]
    if not animation:
        assert unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem).get_num_verts(obj,0)>0
        save(obj.get_editor_property('skeleton'))
    save(obj)
    return obj


def check_reference(mesh, capture):
    skeleton = mesh.get_editor_property('skeleton')
    pose = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
    names = [str(n) for n in unreal.AnimPoseExtensions.get_bone_names(pose)]
    assert names == [b['name'] for b in capture['bones']], 'Reference bone names/order differ'
    component = unreal.SkeletalMeshComponent()
    component.set_skinned_asset_and_update(mesh)
    actual = []
    maxima = {'position_cm':0.0,'rotation_deg':0.0,'scale':0.0}
    for bone in capture['bones']:
        name = bone['name']
        parent = str(component.get_parent_bone(name))
        assert (None if parent=='None' else parent)==bone['parent'], 'Parent differs: '+name
        tr = unreal.AnimPoseExtensions.get_ref_bone_pose(pose,name,unreal.AnimPoseSpaces.LOCAL)
        local = bone['local']
        error = {'position_cm':math.dist([tr.translation.x,tr.translation.y,tr.translation.z],local['loc']),
                 'rotation_deg':angle(tr.rotation,local['quat']),
                 'scale':max(abs(v-e) for v,e in zip([tr.scale3d.x,tr.scale3d.y,tr.scale3d.z],local['scale']))}
        for key,value in error.items(): maxima[key]=max(maxima[key],value)
        actual.append({'name':name,'parent':bone['parent'],'local':{'quat':[tr.rotation.x,tr.rotation.y,tr.rotation.z,tr.rotation.w]}})
    assert maxima['position_cm']<.002 and maxima['rotation_deg']<.02 and maxima['scale']<2e-5, maxima
    return actual,maxima


def retargeter(folder, meshes, references):
    assets = unreal.AssetToolsHelpers.get_asset_tools()
    rigs = {}
    for key,mesh in meshes.items():
        rig = assets.create_asset('IK_'+key,folder,unreal.IKRigDefinition,unreal.IKRigDefinitionFactory())
        ctl = unreal.IKRigController.get_controller(rig)
        assert ctl.set_skeletal_mesh(mesh) and ctl.set_retarget_root('pelvis')
        for bone in references['source']:
            name = bone['name']
            assert str(ctl.add_retarget_chain(name,name,name,'None'))==name
        save(rig)
        rigs[key]=rig
    asset = assets.create_asset('RTG_FabToNativeDevice',folder,unreal.IKRetargeter,unreal.IKRetargetFactory())
    ctl = unreal.IKRetargeterController.get_controller(asset)
    ctl.set_ik_rig(unreal.RetargetSourceOrTarget.SOURCE,rigs['source'])
    ctl.set_ik_rig(unreal.RetargetSourceOrTarget.TARGET,rigs['target'])
    while ctl.get_num_retarget_ops(): assert ctl.remove_retarget_op(0)
    idx = ctl.add_retarget_op('/Script/IKRig.IKRetargetFKChainsOp')
    assert idx>=0
    ctl.assign_ik_rig_to_all_ops(unreal.RetargetSourceOrTarget.SOURCE,rigs['source'])
    ctl.assign_ik_rig_to_all_ops(unreal.RetargetSourceOrTarget.TARGET,rigs['target'])
    ctl.auto_map_chains(unreal.AutoMapChainType.EXACT,True)
    fk = ctl.get_op_controller(idx)
    settings = fk.get_settings()
    settings.chains_to_retarget = [unreal.RetargetFKChainSettings(
        target_chain_name=b['name'],enable_fk=True,rotation_mode=unreal.FKChainRotationMode.ONE_TO_ONE,rotation_alpha=1.0,
        translation_mode=unreal.FKChainTranslationMode.ABSOLUTE if b['name'] in ('root','pelvis') or b['name'].startswith('ik_') else unreal.FKChainTranslationMode.NONE,
        translation_alpha=1.0) for b in references['source']]
    fk.set_settings(settings)
    ctl.create_retarget_pose('MeasuredSourcePose',unreal.RetargetSourceOrTarget.TARGET)
    assert ctl.set_current_retarget_pose('MeasuredSourcePose',unreal.RetargetSourceOrTarget.TARGET)
    for source,target in zip(references['source'],references['target']):
        assert (source['name'],source['parent'])==(target['name'],target['parent'])
        s,t = unreal.Quat(*source['local']['quat']),unreal.Quat(*target['local']['quat'])
        ctl.set_rotation_offset_for_retarget_pose_bone(source['name'],unreal.Quat(-t.x,-t.y,-t.z,t.w)*s,unreal.RetargetSourceOrTarget.TARGET)
    save(asset)
    return asset


def evaluation_options(mesh):
    options = unreal.AnimPoseEvaluationOptions()
    options.evaluation_type = unreal.AnimDataEvalType.RAW
    options.optional_skeletal_mesh = mesh
    options.should_retarget = False
    options.extract_root_motion = False
    options.incorporate_root_motion_into_pose = True
    return options


def preserve_helpers(source, target, mesh):
    model = source.get_editor_property('data_model_interface')
    names = [str(n) for n in model.get_bone_track_names() if str(n).startswith('ik_')]
    options = evaluation_options(mesh)
    keys = {n:([],[],[]) for n in names}
    for frame in range(model.get_number_of_frames()+1):
        pose = source.get_anim_pose_at_frame(frame,options)
        for name in names:
            tr = unreal.AnimPoseExtensions.get_bone_pose(pose,name,unreal.AnimPoseSpaces.LOCAL)
            keys[name][0].append(tr.translation);keys[name][1].append(tr.rotation);keys[name][2].append(tr.scale3d)
    controller = target.get_editor_property('controller')
    controller.open_bracket('Preserve authored IK helpers',False)
    try:
        for name,(positions,rotations,scales) in keys.items():
            assert controller.set_bone_track_keys(name,positions,rotations,scales,False)
    finally:
        controller.close_bracket(False)
    return names


def align_fbx_end_tick(filename, frames, rate):
    """Round the generated take's stop upward to its intended frame boundary.

    FBX stores integer time ticks. UE's importer floors duration to a frame;
    a few ticks below the boundary can discard a whole frame (observed at 39 Hz).
    Only the new export's two local stop fields change, by at most one microsecond.
    Animation keys and all other bytes remain unchanged.
    """
    data = bytearray(filename.read_bytes())
    assert data[:23] == b'Kaydara FBX Binary  \x00\x1a\x00'
    version = struct.unpack_from('<I',data,23)[0]
    header,size = ('<QQQB',25) if version>=7500 else ('<IIIB',13)
    stops,starts = [],[]
    scalar = {'Y':'h','C':'?','I':'i','F':'f','D':'d','L':'q'}

    def visit(offset, path=()):
        end,count,prop_bytes,name_len = struct.unpack_from(header,data,offset)
        if not end:return offset+size
        assert offset<end<=len(data)
        name = bytes(data[offset+size:offset+size+name_len])
        p = offset+size+name_len
        prop_end = p+prop_bytes
        values = []
        for _ in range(count):
            tag = chr(data[p]);p+=1
            value_offset = p
            if tag in 'SR':
                length = struct.unpack_from('<I',data,p)[0];p+=4
                value = bytes(data[p:p+length]);p+=length
            elif tag in scalar:
                fmt = '<'+scalar[tag]
                value = struct.unpack_from(fmt,data,p)[0];p+=struct.calcsize(fmt)
            elif tag in 'fdilcb':
                _,_,length = struct.unpack_from('<III',data,p);p+=12+length;value=None
            else:raise ValueError('Unsupported FBX property '+tag)
            values.append((tag,value,value_offset))
        assert p==prop_end
        if path==(b'Objects',b'AnimationStack',b'Properties70') and name==b'P':
            if values[0][1] in (b'LocalStart',b'LocalStop'):
                assert values[-1][0]=='L'
                (starts if values[0][1]==b'LocalStart' else stops).append(values[-1])
        if path==(b'Takes',b'Take') and name==b'LocalTime':
            assert len(values)==2 and all(row[0]=='L' for row in values)
            starts.append(values[0]);stops.append(values[1])
        while p<end-size:p=visit(p,path+(name,))
        return end

    offset=27
    while struct.unpack_from(header,data,offset)[0]:offset=visit(offset)
    assert len(stops)==2 and all(row[1]==0 for row in starts), 'One zero-based take required'
    # FBX 7.x files use legacy ticks. SDK 2020.2 converts them to a coarser
    # 141120000 Hz clock when loading. Align upward on BOTH clocks, otherwise
    # rounding only the file's legacy tick can still floor to the previous frame.
    ticks_per_second=46186158000  # FBXSDK_TC_LEGACY_SECOND
    sdk_ticks_per_second=141120000  # FBXSDK_TC_SECOND in the tested SDK
    numerator=frames*rate.denominator*sdk_ticks_per_second
    sdk_stop=(numerator+rate.numerator-1)//rate.numerator
    desired=(sdk_stop*ticks_per_second+sdk_ticks_per_second-1)//sdk_ticks_per_second
    for _,previous,offset in stops:
        assert abs(desired-previous)<=ticks_per_second/1_000_000, 'Unexpected FBX take duration'
        struct.pack_into('<q',data,offset,desired)
    filename.write_bytes(data)
    return {'local_stops_before':[row[1] for row in stops],'local_stop_after':desired,
            'max_adjustment_seconds':max(abs(desired-row[1]) for row in stops)/ticks_per_second}


def export_animation(clip, mesh, filename):
    assert not filename.exists()
    clip.set_preview_skeletal_mesh(mesh)
    save(clip)
    options = unreal.FbxExportOption()
    set_properties(options,{'ascii':False,'export_preview_mesh':False,'force_front_x_axis':False,
                            'map_skeletal_motion_to_root':False,'export_local_time':True,
                            'bake_material_inputs':unreal.FbxMaterialBakeMode.DISABLED})
    task = unreal.AssetExportTask()
    set_properties(task,{'object':clip,'filename':str(filename),'automated':True,'prompt':False,
                         'replace_identical':False,'options':options,'exporter':unreal.AnimSequenceExporterFBX()})
    assert unreal.Exporter.run_asset_export_task(task),task.errors
    assert filename.stat().st_size>10000
    model=clip.get_editor_property('data_model_interface')
    return align_fbx_end_tick(filename,model.get_number_of_frames(),model.get_frame_rate())


def check_motion(source, baked, fresh, meshes, names):
    models = [c.get_editor_property('data_model_interface') for c in (source,baked,fresh)]
    frames = models[0].get_number_of_frames()
    rate = models[0].get_frame_rate()
    for model in models:
        assert [str(n) for n in model.get_bone_track_names()]==names
        assert model.get_number_of_frames()==frames
        r=model.get_frame_rate();assert (r.numerator,r.denominator)==(rate.numerator,rate.denominator)
    assert abs(source.get_play_length()-fresh.get_play_length())<1e-5
    opts = [evaluation_options(meshes[k]) for k in ('source','target','target')]
    maxima = {kind:{n:{'position_cm':0.,'rotation_deg':0.,'scale':0.} for n in names} for kind in ('readback','source_motion')}
    for frame in range(frames+1):
        poses = [c.get_anim_pose_at_frame(frame,o) for c,o in zip((source,baked,fresh),opts)]
        for name in names:
            transforms = [unreal.AnimPoseExtensions.get_bone_pose(p,name,unreal.AnimPoseSpaces.WORLD) for p in poses]
            for kind,(a,b) in zip(('readback','source_motion'),((transforms[1],transforms[2]),(transforms[0],transforms[1]))):
                error={'position_cm':(a.translation-b.translation).length(),'rotation_deg':angle(a.rotation,b.rotation),
                       'scale':max(abs(getattr(a.scale3d,k)-getattr(b.scale3d,k)) for k in ('x','y','z'))}
                for key,value in error.items():maxima[kind][name][key]=max(maxima[kind][name][key],value)
    body = {'root','pelvis','clavicle_l','clavicle_r','upperarm_l','upperarm_r','lowerarm_l','lowerarm_r','hand_l','hand_r','foot_l','foot_r'}
    limits = {'position_cm':.02,'rotation_deg':.02,'scale':.0001}
    for name in names:
        assert all(maxima['readback'][name][k]<limit for k,limit in limits.items()),('FBX readback',name,maxima['readback'][name])
        if name in body or name.startswith('ik_'):
            assert all(maxima['source_motion'][name][k]<limit for k,limit in limits.items()),('Source motion changed',name,maxima['source_motion'][name])
    return {'frames':frames,'samples_per_track':frames+1,'frame_rate':[rate.numerator,rate.denominator],
            'bone_tracks':len(names),'bone_frame_comparisons':len(names)*(frames+1),'per_bone_maxima':maxima}


def main():
    job = json.loads(Path(os.environ['BDB_ANIMATION_JOB']).read_text())
    expected_project = Path(job['project_file']).resolve()
    assert Path(unreal.Paths.get_project_file_path()).resolve()==expected_project, 'Wrong project'
    assert expected_project.stem.startswith(('BD_Native_Animation_Review','BDB_AnimationBake_')), 'Dedicated bake project required'
    engine = unreal.SystemLibrary.get_engine_version()
    assert engine.startswith(job['engine_version_prefix']), (engine,job['engine_version_prefix'])
    folder = job['asset_root']
    assert re.fullmatch(r'/Game/BDB_AnimBake_[A-Za-z0-9_]+',folder), 'Use a dedicated asset root'
    assert not unreal.EditorAssetLibrary.does_directory_exist(folder), 'Asset root already exists'
    output = Path(job['output_dir']).resolve()
    assert not output.exists(), 'Output already exists; choose a new version'
    assert job['clips'], 'No clips'
    paths = {key:check_file(job[key]) for key in ('source_reference','target_reference','source_capture','target_capture')}
    inputs = [(item,check_file(item)) for item in job['clips']]
    assert len({item['output_name'].casefold() for item,_ in inputs})==len(inputs)
    assert all(re.fullmatch(r'[A-Za-z0-9_]+',item['output_name']) for item,_ in inputs)
    captures = {key:json.loads(paths[key+'_capture'].read_text()) for key in ('source','target')}
    assert captures['target']['engine_asset_path'].split('.')[0]=='/Game/Creative/Devices/Mannequin/Meshes/CP_Device_Mannequin_Skeleton'
    output.mkdir(parents=True)
    (output/'exports').mkdir()
    report = {'complete':False,'engine':engine,'job':job,'clips':[],'native_runtime_verified':False}
    try:
        meshes,references={},{}
        report['references']={}
        for key in ('source','target'):
            meshes[key]=import_fbx(paths[key+'_reference'],folder+'/References/'+key,'Reference_'+key)
            references[key],errors=check_reference(meshes[key],captures[key])
            report['references'][key]={'mesh':meshes[key].get_path_name(),'max_error':errors}
        names=[b['name'] for b in references['source']]
        assert len(names)==88 and names==[b['name'] for b in references['target']]
        rtg=retargeter(folder+'/Rig',meshes,references)
        report['retargeter']=rtg.get_path_name()
        for index,(item,path) in enumerate(inputs):
            source=import_fbx(path,folder+'/Source','Source_'+str(index),meshes['source'].get_editor_property('skeleton'))
            assert source.get_editor_property('additive_anim_type')==unreal.AdditiveAnimationType.AAT_NONE, 'Additive clips need a separate workflow'
            model=source.get_editor_property('data_model_interface')
            assert [str(n) for n in model.get_bone_track_names()]==names, 'Full named skeleton animation required'
            assert 0<model.get_number_of_frames()<=job.get('max_frames_per_clip',10000)
            produced=unreal.IKRetargetBatchOperation.duplicate_and_retarget(
                [unreal.EditorAssetLibrary.find_asset_data(source.get_path_name())],meshes['source'],meshes['target'],rtg,
                suffix='__'+folder.rsplit('/',1)[1]+'_'+str(index),include_referenced_assets=False,overwrite_existing_files=False)
            assert len(produced)==1
            clip=produced[0].get_asset()
            helpers=preserve_helpers(source,clip,meshes['source'])
            target_path=folder+'/Baked/'+item['output_name']
            assert unreal.EditorAssetLibrary.rename_asset(clip.get_path_name(),target_path)
            clip=unreal.load_asset(target_path)
            assert clip.get_editor_property('skeleton')==meshes['target'].get_editor_property('skeleton')
            filename=output/'exports'/(item['output_name']+'.fbx')
            time_alignment=export_animation(clip,meshes['target'],filename)
            fresh=import_fbx(filename,folder+'/Readback',item['output_name'],meshes['target'].get_editor_property('skeleton'))
            validation=check_motion(source,clip,fresh,meshes,names)
            row={'name':item['output_name'],'source_sha256':item['sha256'],'fbx_sha256':sha(filename),
                 'fbx_bytes':filename.stat().st_size,'asset':clip.get_path_name(),'preserved_ik_helpers':helpers,
                 'fbx_time_alignment':time_alignment,'validation':validation,'native_runtime_verified':False}
            report['clips'].append(row)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('BDB_NATIVE_ANIMATION_READBACK_PASS',item['output_name'],validation['bone_frame_comparisons'])
        for item,path in inputs:assert sha(path)==item['sha256'],'Source bytes changed'
        report['complete']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BDB_NATIVE_ANIMATION_BATCH_PASS',len(report['clips']))


if __name__=='__main__':main()
