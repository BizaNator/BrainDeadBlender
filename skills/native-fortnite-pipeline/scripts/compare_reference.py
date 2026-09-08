"""Compare named reference captures. Pure Python; no Blender/Unreal dependency."""
import argparse
import json
import math
from pathlib import Path


def quaternion_degrees(a,b):
    norms = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    if norms < 1e-12:
        raise ValueError('Zero quaternion is not a rotation')
    return math.degrees(2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b)))/norms)))


def compare(first,second,position_cm=.002,rotation_deg=.02,scale=2e-5,exact_names=False):
    def rows(capture):
        result = {r['name']:r for r in capture['bones']}
        if not result or len(result)!=len(capture['bones']):
            raise ValueError('Empty or duplicate reference names')
        for row in result.values():
            for space in ('local','component'):
                tr=row[space]
                for field,count in [('loc',3),('quat',4),('scale',3)]:
                    if len(tr[field])!=count or not all(math.isfinite(v) for v in tr[field]):
                        raise ValueError('Invalid transform: '+row['name'])
        return result
    a,b=rows(first),rows(second)
    shared=sorted(a.keys()&b.keys())
    if not shared:
        raise ValueError('No shared named bones')
    maxima={s:{'position_cm':0.,'rotation_deg':0.,'scale':0.} for s in ('local','component')}
    mismatches=[]
    for name in shared:
        errors={}
        if a[name]['parent']!=b[name]['parent']:
            errors['parent']=[a[name]['parent'],b[name]['parent']]
        for space in maxima:
            av,bv=a[name][space],b[name][space]
            values={'position_cm':math.dist(av['loc'],bv['loc']),
                    'rotation_deg':quaternion_degrees(av['quat'],bv['quat']),
                    'scale':max(abs(x-y) for x,y in zip(av['scale'],bv['scale']))}
            for field,value in values.items():maxima[space][field]=max(maxima[space][field],value)
            if values['position_cm']>position_cm or values['rotation_deg']>rotation_deg or values['scale']>scale:
                errors[space]=values
        if errors:mismatches.append({'name':name,**errors})
    only_first=sorted(a.keys()-b.keys());only_second=sorted(b.keys()-a.keys())
    return {'shared_bones':len(shared),'only_first':only_first,'only_second':only_second,
        'maxima':maxima,'mismatches':mismatches,
        'reference_matches':not mismatches and (not exact_names or not(only_first or only_second)),
        'scope':'named parents and local/component reference transforms; not runtime compatibility',
        'tolerances':{'position_cm':position_cm,'rotation_deg':rotation_deg,'scale':scale},
        'exact_names_required':exact_names}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('first',type=Path);p.add_argument('second',type=Path)
    p.add_argument('--exact-names',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--position-cm',type=float,default=.002)
    p.add_argument('--rotation-deg',type=float,default=.02)
    p.add_argument('--scale',type=float,default=2e-5)
    args=p.parse_args()
    result=compare(json.loads(args.first.read_text()),json.loads(args.second.read_text()),position_cm=args.position_cm,rotation_deg=args.rotation_deg,scale=args.scale,exact_names=args.exact_names)
    text=json.dumps(result,indent=2)
    if args.output:
        if args.output.exists():p.error('Refuse to overwrite comparison output')
        args.output.write_text(text,encoding='utf-8')
    else:print(text)
    raise SystemExit(0 if result['reference_matches'] else 1)
