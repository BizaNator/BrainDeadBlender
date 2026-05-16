"""
fit_custom_lips.py

Take a raw lip mesh (e.g. `lips.001_RemoveInterior_Remesh to HardBody`)
and slot it onto the rigged lowpoly head as a permanent separate object
in the face base, alongside Eye_L / Eye_R / Tongue.

What it does
------------
    1. Duplicate the raw lip object so the original stays untouched.
    2. Scale + translate so the lip mesh fits at the head's mouth
       region. Scale comes from matching the lip mesh's bbox width to
       the C_lip_upper_mid <-> ... bone span (or a configured target
       width). Translation places the bbox center at the configured
       mouth anchor.
    3. Bind weights from the source head (Penny LOD0) -> the fitted
       lip mesh using BVH closest-point + barycentric, the same way
       headswap_transfer does. Lip / jaw weights flow through; weights
       on other bones get filtered out unless `keep_all_groups` is set.
    4. Parent the fitted lip to the armature with an Armature modifier
       so it deforms with C_jaw / lip_corner / lip_mid rotations.

Result is `CustomLips` (configurable), positioned + rigged. Use it
alongside the split parts (Eye_L / Eye_R / Tongue) to replace Penny's
own teeth_upper / teeth_lower / inner-lip geometry.
"""

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "src_lip":     "lips.001_RemoveInterior_Remesh to HardBody",
    "src_body":    "Fortnite_Head_LOD0",   # weight donor
    "armature":    "Fortnite_Armature",
    "output_name": "CustomLips",

    # Where on the head should the lip's bbox center land (world coords).
    # Default uses C_lip_upper_mid bone position; override to nudge.
    "anchor_bone": "C_lip_upper_mid",
    "anchor_offset_world": (0.0, 0.0, 0.0),

    # Target width (world meters) -- the lip mesh is scaled so its X-bbox
    # matches this. If None, uses the L_lip_corner <-> R_lip_corner distance.
    "target_width_m": None,

    # Vertex-group filter. By default, only lip / jaw / cheek weights
    # transfer (everything else on Penny would just smear the lips).
    "keep_groups_substr": ["lip", "jaw", "cheek", "tongue"],

    # Weights below this get pruned.
    "weight_epsilon": 1e-5,

    # Replace the existing CustomLips if it's already in the scene.
    "replace_existing": True,
}


# ------------------------------- HELPERS ------------------------------------
def _obj(name):
    o = bpy.data.objects.get(name)
    if o is None:
        raise RuntimeError(f"object '{name}' not found")
    return o


def _bbox_world(obj):
    ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [w.x for w in ws]; ys = [w.y for w in ws]; zs = [w.z for w in ws]
    return (Vector((min(xs), min(ys), min(zs))),
            Vector((max(xs), max(ys), max(zs))))


def _duplicate(obj, new_name):
    new_me = obj.data.copy()
    new_obj = obj.copy()
    new_obj.data = new_me
    new_obj.name = new_name
    new_me.name = new_name + "_mesh"
    for c in obj.users_collection:
        c.objects.link(new_obj)
    if not new_obj.users_collection:
        bpy.context.scene.collection.objects.link(new_obj)
    # Clear parent / modifiers / hide -- fresh start
    new_obj.parent = None
    for m in list(new_obj.modifiers):
        new_obj.modifiers.remove(m)
    new_obj.hide_set(False)
    return new_obj


def _barycentric(p, a, b, c):
    v0 = b - a; v1 = c - a; v2 = p - a
    d00 = v0.dot(v0); d01 = v0.dot(v1); d11 = v1.dot(v1)
    d20 = v2.dot(v0); d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    u, v, w = max(u, 0.0), max(v, 0.0), max(w, 0.0)
    s = u + v + w
    if s <= 0.0:
        return (1.0, 0.0, 0.0)
    return (u / s, v / s, w / s)


def _build_src_bvh(src_body):
    me = src_body.data
    mw = src_body.matrix_world
    coords = [mw @ v.co for v in me.vertices]
    tris = []
    tri_table = []
    for p in me.polygons:
        vs = list(p.vertices)
        for i in range(1, len(vs) - 1):
            tris.append([coords[vs[0]], coords[vs[i]], coords[vs[i + 1]]])
            tri_table.append((vs[0], vs[i], vs[i + 1]))
    bvh = BVHTree.FromPolygons(
        [c for tri in tris for c in tri],
        [(i * 3, i * 3 + 1, i * 3 + 2) for i in range(len(tris))],
        all_triangles=True)
    return bvh, tri_table, coords


def _bind_and_transfer(dst, src_body, keep_substr, eps):
    bvh, tri_table, src_coords = _build_src_bvh(src_body)
    mw = dst.matrix_world

    # Decide which src vgroups to copy
    src_vgs = src_body.vertex_groups
    keep_idx = set()
    for vg in src_vgs:
        if any(s.lower() in vg.name.lower() for s in keep_substr):
            keep_idx.add(vg.index)
    idx_to_name = {vg.index: vg.name for vg in src_vgs}

    # Per-src-vert weight lookup
    src_w = [dict() for _ in range(len(src_body.data.vertices))]
    for v in src_body.data.vertices:
        for g in v.groups:
            if g.group in keep_idx:
                src_w[v.index][g.group] = g.weight

    # Create dst vgroups
    made = []
    for gi in sorted(keep_idx):
        name = idx_to_name[gi]
        if name in dst.vertex_groups:
            dst.vertex_groups.remove(dst.vertex_groups[name])
        made.append((gi, dst.vertex_groups.new(name=name)))

    # Bind each dst vert to nearest src tri, interpolate weights
    misses = 0
    for vi, v in enumerate(dst.data.vertices):
        p = mw @ v.co
        hit = bvh.find_nearest(p)
        if hit[0] is None:
            misses += 1
            continue
        tri_i = hit[2]
        a, b, c = tri_table[tri_i]
        u, vc, w = _barycentric(hit[0], src_coords[a], src_coords[b], src_coords[c])
        for gi, dst_vg in made:
            wt = u * src_w[a].get(gi, 0) + vc * src_w[b].get(gi, 0) + w * src_w[c].get(gi, 0)
            if wt > eps:
                dst_vg.add([vi], wt, 'REPLACE')
    return len(made), misses


# --------------------------------- ENTRY ------------------------------------
def fit_custom_lips(cfg):
    src_lip = _obj(cfg["src_lip"])
    src_body = _obj(cfg["src_body"])
    arm = _obj(cfg["armature"])

    print(f"=== fit_custom_lips -> {cfg['output_name']} ===")

    # Resolve anchor target position
    anchor_bone = arm.data.bones.get(cfg["anchor_bone"])
    if anchor_bone is None:
        raise RuntimeError(f"bone '{cfg['anchor_bone']}' not found")
    anchor_world = (arm.matrix_world @ anchor_bone.head_local) + Vector(cfg["anchor_offset_world"])
    print(f"  anchor target world: ({anchor_world.x:.3f},{anchor_world.y:.3f},{anchor_world.z:.3f})")

    # Resolve target width
    target_w = cfg["target_width_m"]
    if target_w is None:
        L = arm.data.bones.get("L_lip_corner")
        R = arm.data.bones.get("R_lip_corner")
        if L and R:
            target_w = abs((arm.matrix_world @ L.head_local).x
                           - (arm.matrix_world @ R.head_local).x) * 1.05
        else:
            target_w = 0.05  # 5cm fallback
    print(f"  target width: {target_w*100:.2f} cm")

    # Replace existing
    out_name = cfg["output_name"]
    if cfg["replace_existing"]:
        existing = bpy.data.objects.get(out_name)
        if existing:
            em = existing.data
            bpy.data.objects.remove(existing, do_unlink=True)
            if isinstance(em, bpy.types.Mesh) and em.users == 0:
                bpy.data.meshes.remove(em)

    # Duplicate the source lip
    dst = _duplicate(src_lip, out_name)

    # Current bbox -> compute uniform scale + translate to anchor
    bmin, bmax = _bbox_world(dst)
    cur_w = bmax.x - bmin.x
    scale = target_w / cur_w if cur_w > 1e-6 else 1.0

    # Apply scale at world origin (since lip data is around origin) then translate
    # Easier: just set object scale and location
    dst.scale = (scale, scale, scale)
    # Recompute center after scale -- bbox center scales with mesh
    cur_center = (bmin + bmax) * 0.5 * scale
    dst.location = anchor_world - cur_center
    print(f"  scaled by {scale:.4f}, located at ({dst.location.x:.3f},{dst.location.y:.3f},{dst.location.z:.3f})")

    # Apply transform so mesh-data lives at the fitted positions (so the
    # subsequent BVH bind uses the same world coords the user sees).
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.select_all(action='DESELECT')
    dst.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Transfer weights from Penny via BVH
    n_groups, misses = _bind_and_transfer(
        dst, src_body, cfg["keep_groups_substr"], cfg["weight_epsilon"])
    print(f"  transferred {n_groups} vgroups, {misses} verts missed")

    # Parent + Armature modifier
    mod = dst.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    mod.use_vertex_groups = True
    dst.parent = arm
    dst.matrix_parent_inverse = arm.matrix_world.inverted()

    # Bring the dst mesh-data into src's local frame (same trick
    # headswap_transfer.relocalize_to_src uses) so bone rotations work
    src_offset = src_body.matrix_world.translation.copy()
    if src_offset.length > 1e-6:
        me = dst.data
        for v in me.vertices:
            v.co = v.co - src_offset
        dst.location = src_offset
        dst.matrix_parent_inverse.identity()
        me.update()

    bbox = _bbox_world(dst)
    bsize = bbox[1] - bbox[0]
    print(f"  final world bbox size: {bsize.x:.3f} x {bsize.y:.3f} x {bsize.z:.3f}")
    print(f"  '{out_name}' ready: {len(dst.data.vertices)}v, {len(dst.vertex_groups)} vgroups")
    return dst


if __name__ == "__main__":
    fit_custom_lips(CONFIG)
