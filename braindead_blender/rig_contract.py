"""Read-only skeleton and skin contract. Pure comparison also runs outside Blender.

Structural parity is necessary evidence, never a claim of engine playback.
Matrices use Blender world coordinates in meters and retain bone orientation/scale.
"""
import math


def _finite(values):
    return all(math.isfinite(v) for v in values)


def _matrix_error(a, b):
    if not _finite([v for m in (a,b) for row in m for v in row]):
        return None
    position=math.sqrt(sum((a[i][3]-b[i][3])**2 for i in range(3)))
    scales=[]; angles=[]
    for col in range(3):
        sa=math.sqrt(sum(a[i][col]**2 for i in range(3)))
        sb=math.sqrt(sum(b[i][col]**2 for i in range(3)))
        if min(sa,sb)<1e-12:
            return None
        dot=sum(a[i][col]*b[i][col] for i in range(3))/(sa*sb)
        angles.append(math.degrees(math.acos(max(-1.,min(1.,dot)))))
        scales.append(abs(sb/sa-1))
    return position,max(angles),max(scales)


def compare(reference, target, *, required_weighted=(), required_morphs=(),
            position_tolerance_m=2e-5, angle_tolerance_deg=.02,
            scale_tolerance=2e-5, max_influences=8):
    """Compare measured snapshots, with explicit FBX round-trip tolerances."""
    errors=[]; bone_errors={}
    ref=reference.get('bones',{}); actual=target.get('bones',{})
    if not ref: errors.append('empty_reference')
    if not actual: errors.append('empty_target_skeleton')
    for name in sorted(ref.keys()-actual.keys()): errors.append('missing_bone:'+name)
    for name in sorted(actual.keys()-ref.keys()): errors.append('extra_bone:'+name)
    for name in sorted(ref.keys() & actual.keys()):
        a,b=ref[name],actual[name]
        if a['parent']!=b['parent']: errors.append('parent:'+name)
        delta=_matrix_error(a['matrix_world'],b['matrix_world'])
        if delta is None:
            errors.append('invalid_matrix:'+name)
            continue
        pos,angle,scale=delta
        bone_errors[name]={'position_m':pos,'axis_angle_deg':angle,'relative_scale':scale}
        if pos>position_tolerance_m: errors.append('rest_position:'+name)
        if angle>angle_tolerance_deg: errors.append('rest_orientation:'+name)
        if scale>scale_tolerance: errors.append('rest_scale:'+name)
    totals={}; morphs=set()
    meshes=target.get('meshes',[])
    if not meshes: errors.append('no_meshes')
    for mesh in meshes:
        name=mesh['name']
        if not mesh['vertices'] or not mesh['polygons']: errors.append('empty_geometry:'+name)
        if not mesh['bound']: errors.append('unbound:'+name)
        for field in ('invalid_coordinates','invalid_weights','unweighted_vertices'):
            if mesh[field]: errors.append(field+':'+name)
        if mesh['max_weight_sum_error']>1e-4: errors.append('weight_normalization:'+name)
        if mesh['max_influences']>max_influences: errors.append('too_many_influences:'+name)
        if mesh['unknown_groups']: errors.append('unknown_weight_groups:'+name)
        if not mesh['uv_layers']: errors.append('missing_uv:'+name)
        if not mesh['materials'] or any(m is None for m in mesh['materials']):
            errors.append('missing_material:'+name)
        if mesh['invalid_morphs']: errors.append('invalid_morphs:'+name)
        for bone,weight in mesh['weight_totals'].items():
            totals[bone]=totals.get(bone,0)+weight
        morphs.update(mesh['morphs'])
    for name in required_weighted:
        if totals.get(name,0)<=1e-6: errors.append('missing_weight:'+name)
    for name in required_morphs:
        if name not in morphs: errors.append('missing_morph:'+name)
    return {'schema':'braindead-rig-contract-1','structural_pass':not errors,
            'engine_animation_tested':False,'errors':errors,
            'tolerances':{'position_m':position_tolerance_m,'axis_angle_deg':angle_tolerance_deg,
                          'relative_scale':scale_tolerance,'max_influences':max_influences},
            'reference_bones':len(ref),'target_bones':len(actual),'bone_errors':bone_errors,
            'max_position_error_m':max((e['position_m'] for e in bone_errors.values()),default=0),
            'max_axis_error_deg':max((e['axis_angle_deg'] for e in bone_errors.values()),default=0),
            'weight_totals':totals,'morphs':sorted(morphs),'meshes':meshes}


def snapshot(armature, meshes=()):
    """Read raw data without modifying geometry, poses, selection or modifiers."""
    if armature.type!='ARMATURE': raise ValueError('Reference/target must be an armature')
    import bpy
    bpy.context.view_layer.update()
    unit=bpy.context.scene.unit_settings.scale_length
    bones={}
    for bone in armature.data.bones:
        m=armature.matrix_world @ bone.matrix_local
        rows=[list(row) for row in m]
        # Position and scale both carry scene units; compare physical transforms.
        for i in range(3):
            for j in range(4): rows[i][j]*=unit
        bones[bone.name]={'parent':bone.parent.name if bone.parent else None,'matrix_world':rows}
    reports=[]
    for obj in meshes:
        data=obj.data
        totals={};unknown=set();unweighted=0;invalid_weights=0;max_error=0.;influences=0
        for v in data.vertices:
            usable=[]
            for item in v.groups:
                name=obj.vertex_groups[item.group].name
                if not math.isfinite(item.weight) or item.weight<0:
                    invalid_weights+=1
                    continue
                if item.weight<=1e-8: continue
                if name not in bones:
                    unknown.add(name)
                    continue
                usable.append(item.weight)
                totals[name]=totals.get(name,0)+item.weight
            total=sum(usable)
            unweighted+=total<=1e-8
            influences=max(influences,len(usable))
            max_error=max(max_error,abs(total-1))
        morphs={};invalid_morphs=[]
        if data.shape_keys:
            basis=data.shape_keys.reference_key
            for key in data.shape_keys.key_blocks:
                if key==basis: continue
                if len(key.data)!=len(data.vertices) or any(not _finite(p.co) for p in key.data):
                    invalid_morphs.append(key.name)
                    continue
                delta=max(((obj.matrix_world.to_3x3() @ (a.co-b.co)).length*unit
                           for a,b in zip(key.data,basis.data)),default=0)
                if delta<1e-8: invalid_morphs.append(key.name)
                morphs[key.name]={'max_delta_m':delta}
        data.calc_loop_triangles()
        reports.append({'name':obj.name,'vertices':len(data.vertices),'polygons':len(data.polygons),
                        'triangles':len(data.loop_triangles),
                        'invalid_coordinates':sum(not _finite(v.co) for v in data.vertices),
                        'invalid_weights':invalid_weights,'unweighted_vertices':unweighted,
                        'max_weight_sum_error':max_error,'max_influences':influences,
                        'unknown_groups':sorted(unknown),'weight_totals':totals,
                        'bound':any(m.type=='ARMATURE' and m.object==armature and m.show_viewport
                                    for m in obj.modifiers),
                        'uv_layers':[u.name for u in data.uv_layers],
                        'materials':[m.name if m else None for m in data.materials],
                        'morphs':morphs,'invalid_morphs':invalid_morphs})
    return {'bones':bones,'meshes':reports}


def fingers():
    return [f'{digit}_{i:02}_{side}' for side in ('l','r')
            for digit in ('thumb','index','middle','ring','pinky') for i in (1,2,3)]
