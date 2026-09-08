"""Preserve captured non-unit rest scales on unweighted FBX leaf bones.

Blender edit bones do not store a separate reference scale. This deliberately
small postprocessor supports uniform, unweighted leaves only. Weighted bones
or ancestors require a full bind-space conversion and are rejected.
"""
import hashlib
import math
from pathlib import Path
import os
import tempfile
import copy


def restore_reference_bind(filename, reference_filename, expected_sha256):
    """Retain a verified reference's FBX bone fields across Blender re-exports.

    Called only after the DCC reference fingerprint gate. Names, parents and
    FBX coordinate conventions must agree. Meshes and skin weight arrays are
    untouched; bone local transforms and world bind matrices come from the
    measured reference instead of repeated float/Euler reconstruction.
    """
    from io_scene_fbx import parse_fbx
    ref_path = Path(reference_filename)
    if hashlib.sha256(ref_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Reference FBX hash changed')
    tree, version = parse_fbx.parse(str(filename))
    ref, ref_version = parse_fbx.parse(str(ref_path))
    if version != ref_version:
        raise ValueError('Reference and export FBX versions differ')

    def content(root):
        objects = next(e for e in root.elems if e.id == b'Objects')
        connections = next(e for e in root.elems if e.id == b'Connections')
        models = {e.props[0]: e for e in objects.elems
                  if e.id == b'Model' and e.props[-1] in (b'Null', b'LimbNode')}
        names = {key: e.props[1].split(b'\0\1')[0].decode() for key, e in models.items()}
        if len(set(names.values())) != len(names):
            raise ValueError('Duplicate skeletal model names')
        parents = {name: None for name in names.values()}
        for link in connections.elems:
            if link.id == b'C' and link.props[0] == b'OO':
                _, child, parent = link.props
                if child in names and parent in names:
                    parents[names[child]] = names[parent]
        poses = {}
        for item in objects.elems:
            if item.id != b'Pose':
                continue
            for node in item.elems:
                if node.id != b'PoseNode':
                    continue
                key = next(e.props[0] for e in node.elems if e.id == b'Node')
                if key in names:
                    poses[names[key]] = next(e for e in node.elems if e.id == b'Matrix')
        globals_node = next(e for e in root.elems if e.id == b'GlobalSettings')
        properties = next(e for e in globals_node.elems if e.id == b'Properties70')
        axes = {e.props[0]: e.props[-1] for e in properties.elems
                if e.id == b'P' and e.props[0] in (b'UpAxis', b'UpAxisSign', b'FrontAxis',
                    b'FrontAxisSign', b'CoordAxis', b'CoordAxisSign', b'UnitScaleFactor')}
        return objects, connections, models, names, parents, poses, axes

    objects, connections, models, names, parents, poses, axes = content(tree)
    _, _, ref_models, ref_names, ref_parents, ref_poses, ref_axes = content(ref)
    if parents != ref_parents or axes != ref_axes or set(poses) != set(ref_poses):
        raise ValueError('Export hierarchy, bind coverage or FBX coordinate conventions differ from reference')
    by_name = {ref_names[key]: model for key, model in ref_models.items()}
    for key, model in models.items():
        name = names[key]
        target_props = next(e for e in model.elems if e.id == b'Properties70')
        source_props = next(e for e in by_name[name].elems if e.id == b'Properties70')
        target_props.elems[:] = copy.deepcopy(source_props.elems)
        poses[name].props[:] = copy.deepcopy(ref_poses[name].props)
    clusters = {e.props[0]: e for e in objects.elems
                if e.id == b'Deformer' and e.props[-1] == b'Cluster'}
    for link in connections.elems:
        if link.id != b'C' or link.props[0] != b'OO':
            continue
        _, child, parent = link.props
        if child in names and parent in clusters:
            matrix = next(e for e in clusters[parent].elems if e.id == b'TransformLink')
            matrix.props[:] = copy.deepcopy(ref_poses[names[child]].props)
    _write_verified(filename, tree, version)
    return {'restored_bone_models': len(models), 'reference_sha256': expected_sha256,
            'scope': 'Verified reference FBX local transforms and bone bind matrices; geometry and weights retained'}


def restore_leaf_scales(filename, overrides):
    if not overrides:
        return {'patched_leaves': [], 'scope': 'No reference scale overrides'}
    from io_scene_fbx import parse_fbx, encode_bin
    path = Path(filename)
    before = path.read_bytes()
    tree, version = parse_fbx.parse(str(path))
    objects = next(e for e in tree.elems if e.id == b'Objects')
    connections = next(e for e in tree.elems if e.id == b'Connections')
    models = {e.props[0]: e for e in objects.elems if e.id == b'Model'}
    clusters = {e.props[0]: e for e in objects.elems
                if e.id == b'Deformer' and e.props[-1] == b'Cluster'}
    links = [e.props for e in connections.elems if e.id == b'C' and e.props[0] == b'OO']
    patches = []
    for name, scale in overrides.items():
        if len(scale) != 3 or not all(math.isfinite(x) and x > 0 for x in scale):
            raise ValueError('Invalid reference scale: ' + name)
        if max(scale) - min(scale) > 1e-8:
            raise ValueError('Only uniform reference leaf scales are supported: ' + name)
        matches = [(key, e) for key, e in models.items()
                   if e.props[1].split(b'\0\1')[0].decode() == name]
        if len(matches) != 1:
            raise ValueError('Expected one named scale-override bone: ' + name)
        key, model = matches[0]
        if any(child in models and parent == key for _, child, parent in links):
            raise ValueError('Scale override has child models: ' + name)
        linked = [clusters[parent] for _, child, parent in links
                  if child == key and parent in clusters]
        for cluster in linked:
            weights = next((e.props[0] for e in cluster.elems if e.id == b'Weights'), [])
            if any(abs(w) > 1e-12 for w in weights):
                raise ValueError('Scale override is weighted: ' + name)
        props = next(e for e in model.elems if e.id == b'Properties70')
        current = [e for e in props.elems if e.id == b'P' and e.props[0] == b'Lcl Scaling']
        old = list(current[0].props[-3:]) if current else [1., 1., 1.]
        if max(old) - min(old) > 1e-8 or min(old) <= 0:
            raise ValueError('Exported leaf has unsupported scaling: ' + name)
        if current:
            current[0].props[-3:] = list(scale)
        else:
            props.elems.append(type(model)(b'P',
                [b'Lcl Scaling', b'Lcl Scaling', b'', b'A', *scale], bytearray(b'SSSSDDD'), []))
        matrices = []
        for item in objects.elems:
            if item.id == b'Pose':
                for node in item.elems:
                    if node.id == b'PoseNode' and any(e.id == b'Node' and e.props[0] == key for e in node.elems):
                        matrices.append(next(e for e in node.elems if e.id == b'Matrix'))
        matrices.extend(next(e for e in c.elems if e.id == b'TransformLink') for c in linked)
        if not matrices:
            raise ValueError('No bind matrix found for scale override: ' + name)
        ratio = scale[0] / old[0]
        for matrix in matrices:
            values = matrix.props[0][:]
            for index in (0, 1, 2, 4, 5, 6, 8, 9, 10):
                values[index] *= ratio
            matrix.props[0] = values
        patches.append({'bone': name, 'scale': list(scale), 'bind_matrices': len(matrices)})

    _write_verified(path, tree, version)
    return {'patched_leaves': patches, 'source_sha256': hashlib.sha256(before).hexdigest(),
            'fbx_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'FBX local and bind scale on uniform unweighted leaves; verify incoming engine reference'}


def _write_verified(filename, tree, version):
    from io_scene_fbx import parse_fbx, encode_bin
    path = Path(filename)
    methods = {'B': 'bool', 'C': 'char', 'Z': 'int8', 'Y': 'int16',
               'I': 'int32', 'L': 'int64', 'F': 'float32', 'D': 'float64',
               'R': 'bytes', 'S': 'string', 'f': 'float32_array', 'd': 'float64_array',
               'i': 'int32_array', 'l': 'int64_array', 'b': 'bool_array', 'c': 'byte_array'}

    def encode(elem):
        result = encode_bin.FBXElem(elem.id)
        for tag, value in zip(elem.props_type, elem.props):
            getattr(result, 'add_' + methods[chr(tag)])(value)
        result.elems = [encode(e) for e in elem.elems]
        return result

    def check(a, b):
        if a.id != b.id or a.props_type != b.props_type or a.props != b.props or len(a.elems) != len(b.elems):
            raise ValueError('FBX encoder changed element semantics: ' + str(a.id))
        for x, y in zip(a.elems, b.elems):
            check(x, y)

    fd, temporary = tempfile.mkstemp(prefix=path.stem + '_scale_', suffix='.fbx', dir=path.parent)
    os.close(fd)
    try:
        encode_bin.write(temporary, encode(tree), version)
        reread, actual_version = parse_fbx.parse(temporary)
        if actual_version != version:
            raise ValueError('FBX version changed')
        check(tree, reread)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
