"""
headswap_transfer.py

Transfer a rigged character's head rig data (skin weights + shape keys) onto a
standalone head mesh, then bind that head to the same armature. The standalone
head keeps its OWN geometry, UVs, and material -- it just gains the source
body's skeleton weighting and morph targets.

This is the inverse of reshaping the body's head: instead of bending the body
mesh to look like the low-poly head, we make the low-poly head riggable using
the body's rig, so the low-poly "look" and source material are fully preserved.

Pipeline
--------
    1. Duplicate the standalone head (keeps its material).
    2. (optional) Weld coincident verts -- collapses triangle-soup imports
       into a proper connected mesh.
    3. (optional) Clean planar bisect at the neck -- keeps a flat edge loop
       instead of the jagged fringe a raw vertex-delete leaves.
    4. (optional) Cleanup -- drop stray islands + interior bits, delete loose
       geometry, recalc normals.
    5. (optional) Uniform-scale + center it over the source head's bbox.
    6. Apply the transform so the working mesh lives in world space.
    7. Bind every destination vertex to the closest source-head triangle
       (barycentric coords) -- ONE binding reused for weights + shape keys.
    8. Interpolate skin weights through that binding -> new vertex groups.
    9. Interpolate each shape key's per-vertex delta through that binding
       -> new shape keys.
   10. Transfer the source head's UVs (Blender Data Transfer, head-only proxy)
       -> new UV layer; the standalone low-poly head usually has no usable
       unwrap of its own.
   11. Parent to the armature + add an Armature modifier.
   12. Audit the result mesh for integrity issues.

Known limitations (v1 -- iterate as needed)
-------------------------------------------
  * Binding is nearest-triangle. Where the low-poly head and the source head
    differ a lot in proportion (e.g. ears that don't line up), a destination
    vertex can bind to the "wrong" region and inherit the wrong weights.
    Tighten alignment, or add per-region masks, if that shows up.
  * Shape keys are transferred as interpolated deltas, so a morph that relied
    on source-head topology detail will look softer on a coarser low-poly head.

Designed to drop into the BrainDeadBlender add-on: every step is a pure
function, `headswap_transfer(cfg)` is the orchestrator, and the `CONFIG` dict
plus the `__main__` guard let you run it straight from Blender's text editor.
"""

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    # Object names in the current .blend.
    # The src is a Fortnite/UEFN-compatible donor head: its face-bone weights
    # are what drives the lowpoly head in-engine (Fortnite animates faces with
    # BONES, not blendshapes -- ARKit/LiveLink/MetaHuman Animator all drive
    # those face bones via UE's animation system). Penny is the sample.
    "src_body":    "Fortnite_Head_LOD0",   # weights + UVs donor
    "dst_head":    "geometry_0.002",       # standalone head to rig
    "armature":    "Fortnite_Armature",    # Fortnite head rig (56 bones)
    "output_name": "LowPolyHead_Rigged",

    # How to find the head region on src_body: the material slot whose name
    # contains this substring marks the head faces ("Head" matches Penny's
    # F_LRG_Constructor_Head_01 and BaseBody's Mat_MaleHeadYoung).
    "head_material_hint": "Head",

    # Restrict binding + UV transfer to the largest connected component of
    # head-material faces -- excludes interior eyeball / teeth / tongue
    # submeshes that share the head material on Fortnite & MetaHuman heads.
    "src_outer_shell_only": True,

    # Clean up the imported head before transfer (all evaluated in LOCAL space,
    # before alignment):
    #   weld_distance      -- merge coincident verts so a triangle-soup import
    #                         becomes a proper connected mesh; None to skip.
    #   neck_cut_local_z   -- clean planar bisect at this local Z, dropping
    #                         everything below it; None to skip. For a bust-style
    #                         head, put this at the neck pinch.
    #   neck_cut_fill_hole -- cap the neck opening with an n-gon afterwards.
    "weld_distance": 0.0001,
    "neck_cut_local_z": -0.15,
    "neck_cut_fill_hole": False,

    # Mesh cleanup (run after the neck cut, before transfer):
    #   cleanup_keep_largest_island -- drop every connected component except the
    #       biggest; clears stray fragments and the low-poly head's own interior
    #       eye geometry (separate eyeballs get added later as their own object).
    #   cleanup_recalc_normals      -- make face winding consistent.
    "cleanup_mesh": True,
    "cleanup_keep_largest_island": True,
    "cleanup_recalc_normals": True,

    # Auto-fit the head over the source head's bbox before transfer.
    "align": True,
    "align_scale_mode": "avg",   # "avg" | "z" | "max" | "min" -- uniform scale basis

    # What to transfer / do
    "transfer_weights": True,
    "transfer_shape_keys": True,
    "transfer_uvs": True,
    "parent_to_armature": True,

    # Only transfer vertex groups that actually influence the head region.
    # False = copy every group from the body (most will be near-zero on a head).
    "weights_head_groups_only": True,
    "weight_epsilon": 1e-5,      # don't store weights smaller than this

    # UVs: which source layer to sample (None = the body's active UV layer),
    # how Data Transfer maps dst corners onto the source surface, and whether
    # to drop the low-poly head's own (placeholder) UV layers.
    "head_uv_layer": None,
    "uv_loop_mapping": "POLYINTERP_LNORPROJ",   # or "NEAREST_POLYNOR", "NEAREST_POLY"
    "replace_dst_uv_layers": True,

    # When True, every step that ADDS / REMOVES / RE-INDEXES verts is skipped
    # (welding, neck cut, island cleanup, bbox align, transform apply). Set
    # this when dst_head is already a landmark-aligned proxy mesh whose
    # vertex indices must be preserved -- so align_landmarks.restore_geometry()
    # can swap the output's verts back to the lowpoly's native shape after
    # binding. Binding, weight/shape-key/UV transfer, parenting, and
    # relocalize_to_src still run normally.
    "preserve_geometry": False,
}


# ----------------------------------- helpers --------------------------------

def _obj(name):
    o = bpy.data.objects.get(name)
    if o is None:
        raise RuntimeError("Object not found: %r" % name)
    return o


def _basis_world_coords(obj):
    """World-space rest positions of obj's verts (Basis shape key if present)."""
    me = obj.data
    mw = obj.matrix_world
    if me.shape_keys:
        basis = me.shape_keys.key_blocks[0].data
        return [mw @ basis[i].co for i in range(len(me.vertices))]
    return [mw @ v.co for v in me.vertices]


def barycentric(p, a, b, c):
    """Barycentric weights of point p within triangle (a, b, c). Clamped + normalized."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    # clamp tiny negatives from float error on edge/corner hits, then renormalize
    u, v, w = max(u, 0.0), max(v, 0.0), max(w, 0.0)
    s = u + v + w
    if s <= 0.0:
        return (1.0, 0.0, 0.0)
    return (u / s, v / s, w / s)


# ------------------------------- pipeline steps -----------------------------

def duplicate_head(dst_head, output_name):
    """Fresh working copy of the standalone head; keeps its material + UVs."""
    existing = bpy.data.objects.get(output_name)
    if existing:
        me = existing.data
        bpy.data.objects.remove(existing, do_unlink=True)
        if isinstance(me, bpy.types.Mesh) and me.users == 0:
            bpy.data.meshes.remove(me)

    src = _obj(dst_head)
    new_me = src.data.copy()
    new_obj = src.copy()
    new_obj.data = new_me
    new_obj.name = output_name
    new_me.name = output_name + "_mesh"

    # link first so it's in a view layer, then detach + place
    for coll in src.users_collection:
        coll.objects.link(new_obj)
    if not new_obj.users_collection:
        bpy.context.scene.collection.objects.link(new_obj)

    world = src.matrix_world.copy()
    new_obj.parent = None
    new_obj.matrix_world = world
    return new_obj


def weld_mesh(obj, distance):
    """Merge coincident vertices.

    Triangle-soup imports (every face its own disconnected tri) bisect and edge
    badly. Welding collapses them into a proper connected mesh; vertex
    positions, face shapes, and material assignment are unchanged, so the
    low-poly look is identical. Returns the number of verts removed.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
    after = len(bm.verts)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return before - after


def bisect_neck(obj, local_z, fill_hole=False):
    """Clean planar cut at local Z: keep everything above, drop everything below.

    Unlike a raw vertex-delete (which leaves a jagged fringe of slivers on a
    coarse mesh), bisect_plane splits the straddling faces exactly on the plane,
    so the result has a flat boundary at z == local_z. Optionally caps the neck
    hole with an n-gon. Returns the number of verts removed.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = len(bm.verts)
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    res = bmesh.ops.bisect_plane(
        bm, geom=geom,
        plane_co=(0.0, 0.0, local_z),
        plane_no=(0.0, 0.0, 1.0),
        clear_inner=True,    # remove the -Z (neck / shoulders) side
        clear_outer=False,
    )
    if fill_hole:
        cut_edges = [e for e in res.get("geom_cut", [])
                     if isinstance(e, bmesh.types.BMEdge)]
        if cut_edges:
            bmesh.ops.holes_fill(bm, edges=cut_edges)
    after = len(bm.verts)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return before - after


def _largest_face_component(faces):
    """Return the largest connected component (by face count) of the given faces.

    Used to isolate the OUTER head shell from a donor mesh whose head material
    also covers interior submeshes (eyeballs, teeth, tongue) -- otherwise the
    binding + UV transfer happily sample from those interior bits and the
    result lands eye/teeth weights and UVs onto outer-head verts.
    """
    if not faces:
        return []
    vert_to_faces = {}
    for i, f in enumerate(faces):
        for v in f.verts:
            vert_to_faces.setdefault(v.index, []).append(i)
    visited = [False] * len(faces)
    components = []
    for s in range(len(faces)):
        if visited[s]:
            continue
        stack = [s]
        comp = []
        while stack:
            x = stack.pop()
            if visited[x]:
                continue
            visited[x] = True
            comp.append(x)
            for v in faces[x].verts:
                for nf in vert_to_faces[v.index]:
                    if not visited[nf]:
                        stack.append(nf)
        components.append(comp)
    components.sort(key=len, reverse=True)
    return [faces[i] for i in components[0]]


def _vertex_islands(bm):
    """List of connected vertex-index components in a bmesh, largest first."""
    bm.verts.ensure_lookup_table()
    visited = [False] * len(bm.verts)
    islands = []
    for s in range(len(bm.verts)):
        if visited[s]:
            continue
        stack = [s]
        comp = []
        while stack:
            x = stack.pop()
            if visited[x]:
                continue
            visited[x] = True
            comp.append(x)
            for e in bm.verts[x].link_edges:
                o = e.other_vert(bm.verts[x])
                if not visited[o.index]:
                    stack.append(o.index)
        islands.append(comp)
    islands.sort(key=len, reverse=True)
    return islands


def cleanup_mesh(obj, keep_largest_island=True, recalc_normals=True,
                 delete_loose=True):
    """Tidy a head mesh after weld / cut / before transfer.

    - delete_loose:         drop verts/edges not used by any face
    - keep_largest_island:  drop every connected component except the biggest --
                            clears stray fragments and the low-poly head's own
                            interior eye geometry (real eyeballs are added later
                            as a separate object)
    - recalc_normals:       make face winding consistent (fixes shading)
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    info = {}

    doomed = set()
    if delete_loose:
        loose = [v for v in bm.verts if not v.link_faces]
        info["loose_verts_removed"] = len(loose)
        doomed.update(v.index for v in loose)

    if keep_largest_island:
        islands = _vertex_islands(bm)
        info["islands_found"] = len(islands)
        if len(islands) > 1:
            for isl in islands[1:]:
                doomed.update(isl)
            info["islands_removed"] = len(islands) - 1
            info["island_verts_removed"] = sum(len(i) for i in islands[1:])
        else:
            info["islands_removed"] = 0

    if doomed:
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in doomed], context='VERTS')

    if recalc_normals:
        bm.faces.ensure_lookup_table()
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        info["normals_recalculated"] = True

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return info


def audit_mesh(obj):
    """Read-only mesh-integrity report: islands, loose geo, non-manifold, etc."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    island_sizes = [len(i) for i in _vertex_islands(bm)]
    # Count only near back-to-back adjacent faces (dot < -0.95) as a real issue --
    # merely "sharp" angles (down to ~120 deg) are normal on a low-poly mesh and
    # would otherwise drown the signal.
    back_to_back = sum(1 for e in bm.edges if len(e.link_faces) == 2
                       and e.link_faces[0].normal.dot(e.link_faces[1].normal) < -0.95)
    report = {
        "verts": len(obj.data.vertices),
        "faces": len(obj.data.polygons),
        "islands": len(island_sizes),
        "largest_island": island_sizes[0] if island_sizes else 0,
        "loose_verts": sum(1 for v in bm.verts if not v.link_faces),
        "wire_edges": sum(1 for e in bm.edges if not e.link_faces),
        "boundary_edges": sum(1 for e in bm.edges if len(e.link_faces) == 1),
        "nonmanifold_edges": sum(1 for e in bm.edges if len(e.link_faces) > 2),
        "degenerate_faces": sum(1 for f in bm.faces if f.calc_area() < 1e-9),
        "ngons": sum(1 for f in bm.faces if len(f.verts) > 4),
        "back_to_back_faces": back_to_back,
    }
    bm.free()
    return report


def head_bbox_world(src_body, head_material_hint):
    """World-space bbox of the source body's head region (by material slot)."""
    me = src_body.data
    head_idx = next((i for i, m in enumerate(me.materials)
                     if m and head_material_hint in m.name), None)
    if head_idx is None:
        raise RuntimeError("No material on %r contains %r"
                           % (src_body.name, head_material_hint))
    coords = _basis_world_coords(src_body)
    head_verts = set()
    for p in me.polygons:
        if p.material_index == head_idx:
            head_verts.update(p.vertices)
    pts = [coords[i] for i in head_verts]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    bmin = Vector((min(xs), min(ys), min(zs)))
    bmax = Vector((max(xs), max(ys), max(zs)))
    return bmin, bmax, head_idx, head_verts


def align_to_bbox(obj, bmin, bmax, scale_mode="avg"):
    """Uniform-scale + center obj so its bbox sits over the target bbox.

    Uniform scale only -- a non-uniform fit would distort the head and lose the
    look we're trying to keep. The binding step tolerates an imperfect overlap.
    """
    def world_bbox():
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        lo = Vector((min(xs), min(ys), min(zs)))
        hi = Vector((max(xs), max(ys), max(zs)))
        return lo, hi

    src_min, src_max = world_bbox()
    src_size = src_max - src_min
    dst_size = bmax - bmin
    dst_center = (bmin + bmax) * 0.5

    ratios = [dst_size[i] / src_size[i] if src_size[i] > 1e-9 else 1.0
              for i in range(3)]
    if scale_mode == "z":
        scale = ratios[2]
    elif scale_mode == "max":
        scale = max(ratios)
    elif scale_mode == "min":
        scale = min(ratios)
    else:  # "avg"
        scale = sum(ratios) / 3.0

    obj.scale = (obj.scale.x * scale, obj.scale.y * scale, obj.scale.z * scale)
    bpy.context.view_layer.update()

    new_min, new_max = world_bbox()
    new_center = (new_min + new_max) * 0.5
    obj.location = obj.location + (dst_center - new_center)
    bpy.context.view_layer.update()
    return scale


def apply_transform(obj):
    """Bake obj's transform into its mesh so it lives in world space."""
    bpy.context.view_layer.objects.active = obj
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def relocalize_to_src(obj, src_body):
    """Put obj's mesh-data into the same local frame as the source body's.

    After ``apply_transform`` the destination mesh-data lives in world space,
    while the source body's mesh-data lives in its object-local frame
    (with the object positioned in the world via its location). The armature
    modifier interprets mesh-local positions against bone rest matrices the
    same way for both meshes -- so if the source mesh-data is centered near
    origin and the destination mesh-data is offset by 1.5m, bone-driven
    deformations on the destination get amplified by that offset, producing
    a catastrophic explosion on any rotation.

    This shifts obj's mesh-data by -src_body.matrix_world.translation and
    sets obj.location to +src_body.matrix_world.translation, leaving the
    world-space position unchanged but bringing the mesh-data into the
    SAME object-local frame the source body uses.
    """
    offset = src_body.matrix_world.translation.copy()
    if offset.length < 1e-6:
        return offset  # already in src-local frame

    me = obj.data
    for v in me.vertices:
        v.co = v.co - offset
    if me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            for d in kb.data:
                d.co = d.co - offset
    obj.location = offset
    obj.matrix_parent_inverse.identity()
    me.update()
    return offset


def build_src_head_bvh(src_body, head_idx, outer_shell_only=True):
    """BVH of the source head's triangulated rest mesh (world space).

    Returns (bvh, tri_table) where tri_table[i] is the (a, b, c) source mesh
    vertex indices of BVH triangle i. Drives the per-vertex binding only --
    UVs are handled separately by transfer_uvs.

    With outer_shell_only=True (the default), restricts the BVH to the largest
    connected component of head-material faces. This keeps the binding from
    leaking onto interior submeshes (eyeballs, teeth, tongue) that share the
    same material slot as the head shell.
    """
    me = src_body.data
    mw = src_body.matrix_world
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    # rest positions in WORLD space (Basis if the mesh has shape keys)
    if me.shape_keys:
        basis = me.shape_keys.key_blocks[0].data
        for v in bm.verts:
            v.co = mw @ basis[v.index].co
    else:
        for v in bm.verts:
            v.co = mw @ v.co

    head_faces = [f for f in bm.faces if f.material_index == head_idx]
    bmesh.ops.triangulate(bm, faces=head_faces)
    bm.faces.ensure_lookup_table()
    head_faces = [f for f in bm.faces if f.material_index == head_idx]
    if outer_shell_only:
        head_faces = _largest_face_component(head_faces)

    tri_table = [tuple(l.vert.index for l in f.loops) for f in head_faces]
    verts = [v.co.copy() for v in bm.verts]   # bmesh vert index == mesh vert index
    bm.free()

    bvh = BVHTree.FromPolygons(verts, tri_table, all_triangles=True)
    return bvh, tri_table


def _detect_uv_layer(src_body, requested=None):
    """Resolve which source UV layer to transfer (requested name, else active)."""
    uvs = src_body.data.uv_layers
    if not uvs:
        return None
    if requested and requested in uvs:
        return requested
    return uvs.active.name if uvs.active else uvs[0].name


def _make_head_only_proxy(src_body, head_idx, uv_name=None, outer_shell_only=True):
    """Temp object containing ONLY the source head's outer shell, at rest pose.

    Used as the Data Transfer source so the low-poly head samples head UVs
    only -- never body UVs near the neck, and (with outer_shell_only=True)
    never the interior eyeball / teeth / tongue submeshes that share the
    head material slot on a Fortnite/MetaHuman-style head. Caller is
    responsible for deleting it.
    """
    me = src_body.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    if me.shape_keys:
        basis = me.shape_keys.key_blocks[0].data
        for v in bm.verts:
            v.co = basis[v.index].co

    head_faces = [f for f in bm.faces if f.material_index == head_idx]
    if outer_shell_only:
        keep = set(f.index for f in _largest_face_component(head_faces))
        doomed = [f for f in bm.faces if f.index not in keep]
    else:
        doomed = [f for f in bm.faces if f.material_index != head_idx]
    bmesh.ops.delete(bm, geom=doomed, context='FACES')

    proxy_me = bpy.data.meshes.new("_headswap_uvsrc")
    bm.to_mesh(proxy_me)
    bm.free()
    proxy = bpy.data.objects.new("_headswap_uvsrc", proxy_me)
    proxy.matrix_world = src_body.matrix_world.copy()
    bpy.context.scene.collection.objects.link(proxy)
    if uv_name and uv_name in proxy_me.uv_layers:
        proxy_me.uv_layers.active = proxy_me.uv_layers[uv_name]
    return proxy


def bind_vertices(dst_obj, src_body, bvh, tri_table):
    """For each dst vert: (a, b, c, (u, v, w)) of the closest src-head triangle.

    This per-vertex binding drives skin weights and shape keys -- both are
    per-vertex quantities with no UV-island discontinuities, so nearest-triangle
    barycentric interpolation is safe. (UVs use a separate per-FACE binding; see
    transfer_uvs.)

    Entries are None for verts the BVH could not resolve (should not happen
    with a closed-ish head, but handled defensively).
    """
    src_coords = _basis_world_coords(src_body)   # world space, indexed by vert idx
    dst_mw = dst_obj.matrix_world
    binding = []
    misses = 0
    for v in dst_obj.data.vertices:
        p = dst_mw @ v.co
        loc, _normal, tri_idx, _dist = bvh.find_nearest(p)
        if loc is None:
            binding.append(None)
            misses += 1
            continue
        a, b, c = tri_table[tri_idx]
        bary = barycentric(loc, src_coords[a], src_coords[b], src_coords[c])
        binding.append((a, b, c, bary))
    return binding, misses


def transfer_weights(dst_obj, src_body, binding, head_verts,
                     head_groups_only=True, eps=1e-5):
    """Interpolate skin weights through the binding into new vertex groups."""
    src_vgs = src_body.vertex_groups

    if head_groups_only:
        relevant = set()
        for vi in head_verts:
            for g in src_body.data.vertices[vi].groups:
                if g.weight > eps:
                    relevant.add(g.group)
        group_indices = sorted(relevant)
    else:
        group_indices = [g.index for g in src_vgs]

    idx_to_name = {g.index: g.name for g in src_vgs}

    # fast per-vert weight lookup for the source: src_w[vert_idx][group_idx] = weight
    src_w = [dict() for _ in range(len(src_body.data.vertices))]
    for v in src_body.data.vertices:
        for g in v.groups:
            src_w[v.index][g.group] = g.weight

    # (re)create destination groups
    made = []
    for gi in group_indices:
        name = idx_to_name[gi]
        if name in dst_obj.vertex_groups:
            dst_obj.vertex_groups.remove(dst_obj.vertex_groups[name])
        made.append((gi, dst_obj.vertex_groups.new(name=name)))

    for vi, bind in enumerate(binding):
        if bind is None:
            continue
        a, b, c, (u, v, w) = bind
        for gi, dst_vg in made:
            wt = (u * src_w[a].get(gi, 0.0)
                  + v * src_w[b].get(gi, 0.0)
                  + w * src_w[c].get(gi, 0.0))
            if wt > eps:
                dst_vg.add([vi], wt, 'REPLACE')
    return len(made)


def transfer_shape_keys(dst_obj, src_body, binding):
    """Interpolate each src shape key's delta through the binding into new keys.

    Deltas are computed in source-local space, lifted to world space, then
    pushed into destination-local space -- so a "nose moves 5 mm" morph keeps
    its world-space magnitude regardless of object transforms.
    """
    src_sk = src_body.data.shape_keys
    if not src_sk or len(src_sk.key_blocks) < 2:
        return 0

    src_mw3 = src_body.matrix_world.to_3x3()
    dst_mw3_inv = dst_obj.matrix_world.to_3x3().inverted()

    if not dst_obj.data.shape_keys:
        dst_obj.shape_key_add(name="Basis", from_mix=False)
    dst_basis = dst_obj.data.shape_keys.key_blocks[0]

    src_basis = src_sk.key_blocks[0]
    count = 0
    for kb in src_sk.key_blocks:
        if kb == src_basis:
            continue
        dst_kb = dst_obj.data.shape_keys.key_blocks.get(kb.name)
        if dst_kb is None:
            dst_kb = dst_obj.shape_key_add(name=kb.name, from_mix=False)
        for vi, bind in enumerate(binding):
            if bind is None:
                continue
            a, b, c, (u, v, w) = bind
            da = src_mw3 @ (kb.data[a].co - src_basis.data[a].co)
            db = src_mw3 @ (kb.data[b].co - src_basis.data[b].co)
            dc = src_mw3 @ (kb.data[c].co - src_basis.data[c].co)
            delta_world = u * da + v * db + w * dc
            dst_kb.data[vi].co = dst_basis.data[vi].co + dst_mw3_inv @ delta_world
        count += 1
    return count


def transfer_uvs(dst_obj, src_body, head_idx, uv_name,
                 loop_mapping='POLYINTERP_LNORPROJ', replace_existing=True,
                 outer_shell_only=True):
    """Transfer the source head's UV layout onto dst via Blender's Data Transfer.

    Data Transfer's loop mapping is Blender's production-tested surface-to-
    surface UV transfer -- it resolves each destination corner against the
    source surface and interpolates within a single source polygon, so it does
    not streak the way a hand-rolled per-vertex barycentric pass does.
    POLYINTERP_LNORPROJ projects each dst corner along its normal onto the
    source; NEAREST_POLYNOR is a cheaper nearest-corner fallback.

    The source is a head-only proxy (body faces stripped) so the low-poly head
    can never inherit body UVs near the neck. The low-poly head's own UV layers
    are placeholders, so by default they're dropped first.
    """
    if uv_name is None:
        return None
    me = dst_obj.data
    if replace_existing:
        while me.uv_layers:
            me.uv_layers.remove(me.uv_layers[0])

    proxy = _make_head_only_proxy(src_body, head_idx, uv_name,
                                  outer_shell_only=outer_shell_only)
    try:
        bpy.ops.object.select_all(action='DESELECT')
        dst_obj.select_set(True)
        proxy.select_set(True)
        bpy.context.view_layer.objects.active = proxy   # data flows active -> selected
        bpy.ops.object.data_transfer(
            data_type='UV',
            use_create=True,
            loop_mapping=loop_mapping,
            layers_select_src='ACTIVE',
            layers_select_dst='NAME',
            mix_mode='REPLACE',
        )
    finally:
        pm = proxy.data
        bpy.data.objects.remove(proxy, do_unlink=True)
        if pm.users == 0:
            bpy.data.meshes.remove(pm)

    if not me.uv_layers:
        return None
    uvl = me.uv_layers[0]
    uvl.name = uv_name
    me.uv_layers.active = uvl
    uvl.active_render = True
    return uvl.name


def parent_to_armature(dst_obj, armature_name):
    """Parent dst to the armature and add an Armature modifier bound to it."""
    arm = _obj(armature_name)
    dst_obj.parent = arm
    dst_obj.matrix_parent_inverse = arm.matrix_world.inverted()
    mod = next((m for m in dst_obj.modifiers if m.type == 'ARMATURE'), None)
    if mod is None:
        mod = dst_obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    mod.use_vertex_groups = True
    return mod


# ------------------------------- orchestrator -------------------------------

def headswap_transfer(cfg):
    """Run the full transfer. Returns a dict of diagnostics."""
    src_body = _obj(cfg["src_body"])
    out = duplicate_head(cfg["dst_head"], cfg["output_name"])

    preserve = cfg.get("preserve_geometry", False)

    report = {
        "output": out.name,
        "src_verts": len(src_body.data.vertices),
        "dst_verts_before": len(out.data.vertices),
        "preserve_geometry": preserve,
    }

    if not preserve and cfg.get("weld_distance"):
        report["welded_verts"] = weld_mesh(out, cfg["weld_distance"])

    if not preserve and cfg.get("neck_cut_local_z") is not None:
        report["neck_cut_removed_verts"] = bisect_neck(
            out, cfg["neck_cut_local_z"],
            fill_hole=cfg.get("neck_cut_fill_hole", False))

    if not preserve and cfg.get("cleanup_mesh", True):
        report["cleanup"] = cleanup_mesh(
            out,
            keep_largest_island=cfg.get("cleanup_keep_largest_island", True),
            recalc_normals=cfg.get("cleanup_recalc_normals", True))
    report["dst_verts"] = len(out.data.vertices)

    bmin, bmax, head_idx, head_verts = head_bbox_world(
        src_body, cfg["head_material_hint"])
    report["head_region_verts"] = len(head_verts)

    if not preserve and cfg.get("align", True):
        report["align_scale"] = align_to_bbox(
            out, bmin, bmax, cfg.get("align_scale_mode", "avg"))

    if not preserve:
        apply_transform(out)

    bvh, tri_table = build_src_head_bvh(
        src_body, head_idx,
        outer_shell_only=cfg.get("src_outer_shell_only", True))
    report["src_head_tris"] = len(tri_table)

    binding, misses = bind_vertices(out, src_body, bvh, tri_table)
    report["bind_misses"] = misses

    if cfg.get("transfer_weights", True):
        report["groups_transferred"] = transfer_weights(
            out, src_body, binding, head_verts,
            head_groups_only=cfg.get("weights_head_groups_only", True),
            eps=cfg.get("weight_epsilon", 1e-5))

    if cfg.get("transfer_shape_keys", True):
        report["shape_keys_transferred"] = transfer_shape_keys(
            out, src_body, binding)

    if cfg.get("transfer_uvs", True):
        src_uv_name = _detect_uv_layer(src_body, cfg.get("head_uv_layer"))
        report["src_uv_layer"] = src_uv_name
        report["uv_layer_transferred"] = transfer_uvs(
            out, src_body, head_idx, src_uv_name,
            loop_mapping=cfg.get("uv_loop_mapping", "POLYINTERP_LNORPROJ"),
            replace_existing=cfg.get("replace_dst_uv_layers", True),
            outer_shell_only=cfg.get("src_outer_shell_only", True))

    if cfg.get("parent_to_armature", True):
        parent_to_armature(out, cfg["armature"])
        report["parented_to"] = cfg["armature"]
        # Bring dst mesh-data into the same local frame as src_body. Without
        # this, apply_transform leaves mesh-data in world space while bones
        # interpret mesh-local positions against their rest matrices -- a
        # mismatch that amplifies any pose rotation into an explosion.
        offset = relocalize_to_src(out, src_body)
        report["relocalized_by"] = tuple(round(v, 4) for v in offset)

    out.data.update()
    report["audit"] = audit_mesh(out)
    return report


if __name__ == "__main__":
    import pprint
    pprint.pprint(headswap_transfer(CONFIG))
