"""CPU-only Blender CLI: --reference original.fbx --target candidate.fbx --output audit.json.

Run in a disposable background process; import clears that process's scene.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import bpy

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('rig_contract',ROOT/'braindead_blender/rig_contract.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def load(path):
    if not bpy.app.background: raise RuntimeError('Run in a dedicated background Blender process')
    path=Path(path)
    if path.suffix.lower()=='.blend': bpy.ops.wm.open_mainfile(filepath=str(path))
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(path),use_image_search=False)
    rigs=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
    if len(rigs)!=1: raise ValueError(f'Expected one armature, found {len(rigs)}')
    data=audit.snapshot(rigs[0],[o for o in bpy.context.scene.objects if o.type=='MESH'])
    data['file']=str(path);data['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',required=True);parser.add_argument('--target',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--require-fingers',action='store_true')
    parser.add_argument('--require-morph',action='append',default=[])
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    reference=load(args.reference);target=load(args.target)
    report=audit.compare(reference,target,required_weighted=audit.fingers() if args.require_fingers else (),
                         required_morphs=args.require_morph)
    report['source_files']={key:{k:value[k] for k in ('file','sha256')}
                            for key,value in [('reference',reference),('target',target)]}
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('structural_pass','errors','max_position_error_m','max_axis_error_deg')}))
    if not report['structural_pass']: raise RuntimeError('Rig contract failed; see report')
