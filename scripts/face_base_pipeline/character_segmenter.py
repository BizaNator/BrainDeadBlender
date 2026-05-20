"""character_segmenter.py — BDTools multi-part character kit cutter.

Cuts a finished body mesh into 12 closed, swappable kit parts along STANDARD
planes (fractions of mesh height / X-extent), so every character segments
identically and parts interchange across the kit library.

Parts: Head, Neck, Torso, Arm_L, Arm_R, Hand_L, Hand_R, Hips, Leg_L, Leg_R,
Foot_L, Foot_R.
"""
import bpy, bmesh, math

# Horizontal seam planes — fraction of mesh height, measured from zmin.
SEG = dict(
    head   = 0.860,   # Head  / Neck
    neck   = 0.800,   # Neck  / Torso
    waist  = 0.610,   # Torso / Hips
    apex   = 0.475,   # crotch apex — origin of the leg-V planes
    ankle  = 0.060,   # Leg   / Foot
)
LEG_SPLAY_DEG = 45.0          # leg-socket V planes, degrees from vertical
# Arm/hand seam planes — fraction of the mesh's max |X| (the fingertip in T-pose).
ARM_ROOT_XFRAC  = 0.34        # Torso / Arm
WRIST_XFRAC     = 0.80        # Arm   / Hand


def _classify(c, zmin, H, maxx, apex_z, nL, nR):
    """c = face centroid Vector. Returns one of the 12 part-id strings."""
    z = c.z
    head_z  = zmin + SEG['head']  * H
    neck_z  = zmin + SEG['neck']  * H
    waist_z = zmin + SEG['waist'] * H
    ankle_z = zmin + SEG['ankle'] * H
    arm_x   = ARM_ROOT_XFRAC * maxx
    wrist_x = WRIST_XFRAC    * maxx
    side = 'L' if c.x >= 0.0 else 'R'

    if z > head_z:
        return 'Head'
    if z > neck_z:
        return 'Neck'
    if z > waist_z:
        # torso band — arms/hands branch off in X (T-pose)
        if abs(c.x) > wrist_x:
            return f'Hand_{side}'
        if abs(c.x) > arm_x:
            return f'Arm_{side}'
        return 'Torso'
    # lower body — leg-V planes decide hips vs legs
    dL = c.x * nL[0] + (z - apex_z) * nL[2]
    dR = c.x * nR[0] + (z - apex_z) * nR[2]
    if dL >= -1e-6 and dR >= -1e-6:
        return 'Hips'
    if z < ankle_z:
        return f'Foot_{side}'
    return f'Leg_{side}'


def segment_character(body_obj, parts_wanted=None, prefix="Seg"):
    """Cut body_obj into closed kit parts. parts_wanted = optional set of
    part-id strings to keep (default all 12). Returns {part_id: object}."""
    deps = bpy.context.evaluated_depsgraph_get()
    tmp = bpy.data.meshes.new_from_object(body_obj.evaluated_get(deps))
    tmp.transform(body_obj.matrix_world)

    xs = [v.co.x for v in tmp.vertices]
    zs = [v.co.z for v in tmp.vertices]
    zmin, zmax = min(zs), max(zs)
    H = zmax - zmin
    maxx = max(abs(min(xs)), abs(max(xs)))
    apex_z = zmin + SEG['apex'] * H
    th = math.radians(LEG_SPLAY_DEG)
    nL = (-math.cos(th), 0.0, math.sin(th))
    nR = ( math.cos(th), 0.0, math.sin(th))

    bm = bmesh.new()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0003)

    # bisect along every seam plane (cut only — keep both sides)
    planes = [
        ((0,0,zmin+SEG['head'] *H), (0,0,1)),
        ((0,0,zmin+SEG['neck'] *H), (0,0,1)),
        ((0,0,zmin+SEG['waist']*H), (0,0,1)),
        ((0,0,zmin+SEG['ankle']*H), (0,0,1)),
        ((0,0,apex_z), nL),
        ((0,0,apex_z), nR),
        (( ARM_ROOT_XFRAC*maxx,0,0), (1,0,0)),
        ((-ARM_ROOT_XFRAC*maxx,0,0), (1,0,0)),
        (( WRIST_XFRAC*maxx,0,0),    (1,0,0)),
        ((-WRIST_XFRAC*maxx,0,0),    (1,0,0)),
    ]
    for co, no in planes:
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:]+bm.edges[:]+bm.faces[:],
                               dist=1e-5, plane_co=co, plane_no=no,
                               clear_inner=False, clear_outer=False)
    bm.faces.ensure_lookup_table()

    fcls = {f: _classify(f.calc_center_median(), zmin, H, maxx, apex_z, nL, nR)
            for f in bm.faces}
    fcls = _island_cleanup(bm, fcls)          # Task 3

    keep = parts_wanted or set(fcls.values())
    parts = {}
    for pid in sorted(set(fcls.values())):
        if pid not in keep:
            continue
        faces = [f for f in bm.faces if fcls[f] == pid]
        if not faces:
            continue
        parts[pid] = _extract_part(faces, f"{prefix}_{pid}")
    bm.free()
    return parts


def _extract_part(faces, name):
    nb = bmesh.new()
    vmap = {}
    for f in faces:
        nv = []
        for v in f.verts:
            if v not in vmap:
                vmap[v] = nb.verts.new(v.co)
            nv.append(vmap[v])
        try:
            nb.faces.new(nv)
        except ValueError:
            pass
    nb.edges.ensure_lookup_table()
    bnd = [e for e in nb.edges if len(e.link_faces) == 1]
    if bnd:
        try:
            bmesh.ops.holes_fill(nb, edges=bnd, sides=0)
        except Exception:
            pass
    nb.normal_update()
    bmesh.ops.recalc_face_normals(nb, faces=nb.faces[:])
    me = bpy.data.meshes.new(name)
    nb.to_mesh(me)
    nb.free()
    for p in me.polygons:
        p.use_smooth = False          # hard-body flat shading
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj


def _island_cleanup(bm, fcls):
    """Re-tag small disconnected islands to their bordering part."""
    from collections import Counter, deque
    # group faces by class, flood-fill connected islands within each class
    seen = set()
    for start in bm.faces:
        if start in seen:
            continue
        cls = fcls[start]
        island, q = [], deque([start])
        seen.add(start)
        while q:
            f = q.popleft()
            island.append(f)
            for e in f.edges:
                for nf in e.link_faces:
                    if nf not in seen and fcls[nf] == cls:
                        seen.add(nf)
                        q.append(nf)
        # is this island the dominant one for its class? if tiny, re-home it
        same_class_total = sum(1 for ff in bm.faces if fcls[ff] == cls)
        if len(island) < 0.30 * same_class_total and len(island) < 400:
            border = Counter()
            iset = set(island)
            for f in island:
                for e in f.edges:
                    for nf in e.link_faces:
                        if nf not in iset:
                            border[fcls[nf]] += 1
            if border:
                newcls = border.most_common(1)[0][0]
                for f in island:
                    fcls[f] = newcls
    return fcls
