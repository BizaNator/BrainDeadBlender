"""Read a named Skeleton reference in Unreal/UEFN; never modify or save assets.

In the editor's Python environment:
    import runpy
    tools = runpy.run_path('/path/to/capture_reference.py')
    tools['capture']('/Game/path/MySkeleton', '/output/reference.json')
"""
import json
from pathlib import Path
import re


def capture(asset_path, output_path):
    import unreal
    output = Path(output_path)
    if output.exists() or output.with_suffix('.animpose.txt').exists():
        raise FileExistsError('Choose a new capture path: '+str(output))
    skeleton = unreal.load_asset(asset_path)
    if not isinstance(skeleton, unreal.Skeleton):
        raise ValueError('Expected a loaded Skeleton: '+asset_path)
    pose = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
    names = list(map(str, unreal.AnimPoseExtensions.get_bone_names(pose)))
    if not names or len(names) != len(set(n.casefold() for n in names)):
        raise ValueError('Reference is empty or contains ambiguous names')
    raw_pose = pose.export_text()
    match = re.search(r'ParentBoneIndices=\(([-\d,\s]+)\)', raw_pose)
    if not match:
        raise ValueError('This engine does not expose the measured ParentBoneIndices text layout')
    parents = [int(n) for n in match.group(1).split(',')]
    if len(parents) != len(names):
        raise ValueError('Parent/name counts differ')
    for i, parent in enumerate(parents):
        if parent < -1 or parent >= i:
            raise ValueError('Unexpected reference parent ordering at '+names[i])

    def transform(value):
        return {'loc':[value.translation.x,value.translation.y,value.translation.z],
                'quat':[value.rotation.x,value.rotation.y,value.rotation.z,value.rotation.w],
                'scale':[value.scale3d.x,value.scale3d.y,value.scale3d.z]}

    physical_count = None
    physical_error = None
    try:
        physical_count = len(skeleton.get_editor_property('bone_tree'))
    except Exception as error:
        physical_error = str(error)
    rows = []
    for i,name in enumerate(names):
        rows.append({'name':name,'index':i,'parent_index':parents[i],
            'parent':names[parents[i]] if parents[i]>=0 else None,
            'local':transform(pose.get_ref_bone_pose(name,unreal.AnimPoseSpaces.LOCAL)),
            'component':transform(pose.get_ref_bone_pose(name,unreal.AnimPoseSpaces.WORLD)),
            'virtual_name_prefix':name.startswith('VB ')})
    report = {'engine_asset_path':skeleton.get_path_name(),
        'engine':unreal.SystemLibrary.get_engine_version(),
        'bone_count':len(rows),'physical_bone_count':physical_count,
        'physical_count_error':physical_error,'bones':rows,
        'parent_source':'AnimPose.export_text ParentBoneIndices',
        'transform_source':'get_ref_bone_pose; translation cm; WORLD is reference component space',
        'virtual_definitions':{'captured':False,'note':'VB name prefix is recorded separately, not a definition'},
        'sockets':{'captured':False}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.animpose.txt').write_text(raw_pose,encoding='utf-8')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
