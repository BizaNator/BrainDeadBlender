"""
weld_lip_to_head.py

Position the CustomLips mesh so its silhouette plane sits at the head's
mouth-area FRONT PROFILE (the chin / upper-lip line) instead of being
left floating in space or sucked into the recessed mouth cavity. The
subsequent merge_face_meshes pass then welds the lips into a coherent
seam.

Strategy: WHOLE-OBJECT TRANSLATION (not per-vert snap)
------------------------------------------------------
Two problems the previous per-vert snap caused:
  - On a head whose mouth area is RECESSED (Tripo characters often
    are), "snap silhouette verts to nearest head surface" pulls them
    INWARD into the cavity, making the lips look recessed behind the
    chin.
  - Per-vert snap distorts the lip's natural shape because each vert
    moves independently.

This script preserves the lip's natural geometry and just moves the
whole object on +/- Y (depth axis) so its silhouette plane lands at
the head's front profile at the lip's Z range.

Algorithm:
  1. Detect lip silhouette (front-face vs back-face boundary edges)
     and compute its centroid -- this is the natural "lip line" plane.
  2. Find the head's FRONT PROFILE at the lip's Z range:
     for each head vert in the lip's Z band, take its Y value;
     the most-negative Y (most forward) defines the chin/upper-lip
     outline. Take the average of the front N verts as the target Y.
  3. Translate CustomLips so its silhouette centroid Y matches target.

Per-vert silhouette snap is kept as an opt-in mode for heads where
the front profile is already flat enough that snapping doesn't recess.

Run AFTER extract_mouth_parts and BEFORE merge_face_meshes.
"""

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "lip_object":  "CustomLips",
    "head_object": "LowPolyHead_Rigged",

    # "translate" (default): preserve lip shape, shift whole object so
    #   silhouette centroid sits at head front profile at lip's Z range.
    #   Correct for recessed-mouth heads.
    # "snap_verts": per-vert silhouette -> nearest head surface. Only
    #   safe when head's mouth area is flat (no recessed cavity).
    "mode": "translate",

    # Silhouette classification: face is FRONT if world normal Y < this.
    "silhouette_face_normal_y_threshold": 0.0,

    # Translate mode: how to compute the head's front profile target Y.
    #   "front_avg_in_lip_z": average Y of head verts in the lip's Z
    #     range, taking only the front N most-negative-Y verts (excludes
    #     the recessed cavity verts).
    "front_profile_method": "front_avg_in_lip_z",

    # Number of front-most head verts (in the lip Z band) to average to
    # get the target depth. Robust to noisy single front verts.
    "front_profile_avg_count": 8,

    # Extra millimetres of standoff in front of the head profile -- lip
    # silhouette will sit this far IN FRONT of the chin/upper-lip outline.
    # Default 0 = silhouette on profile (lips just begin emerging from
    # the face skin). Set positive (-Y) to extend forward.
    "standoff_mm": 0.0,

    # Per-vert snap (snap_verts mode) settings -- kept for opt-in.
    "max_snap_distance_m": 0.030,
}


# ------------------------------- HELPERS ------------------------------------
def _walk_edge_loops(edges):
    vert_edges = {}
    for e in edges:
        for v in e.verts:
            vert_edges.setdefault(v.index, []).append(e)
    loops = []
    visited = set()
    for start_e in edges:
        if start_e.index in visited:
            continue
        loop = []
        cur_e = start_e
        cur_v = cur_e.verts[0]
        loop.append(cur_v.index)
        while True:
            visited.add(cur_e.index)
            nxt_v = cur_e.other_vert(cur_v)
            loop.append(nxt_v.index)
            next_es = [e for e in vert_edges.get(nxt_v.index, [])
                       if e.index not in visited]
            if not next_es:
                break
            cur_v = nxt_v
            cur_e = next_es[0]
        if len(loop) > 2 and loop[0] == loop[-1]:
            loop.pop()
        loops.append(loop)
    return loops


def _find_silhouette_verts(obj, y_threshold):
    me = obj.data
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    nm = obj.matrix_world.to_3x3()
    face_back = {f.index: ((nm @ f.normal).normalized().y > y_threshold)
                 for f in bm.faces}
    sil_edges = []
    for e in bm.edges:
        if len(e.link_faces) < 2:
            continue
        classes = set(face_back[f.index] for f in e.link_faces)
        if len(classes) > 1:
            sil_edges.append(e)
    loops = _walk_edge_loops(sil_edges)
    vert_set = set()
    for L in loops:
        vert_set.update(L)
    bpy.ops.object.mode_set(mode='OBJECT')
    return vert_set, loops


def _world_centroid(obj, vert_indices):
    cs = [obj.matrix_world @ obj.data.vertices[vi].co for vi in vert_indices]
    n = len(cs)
    return Vector((sum(c.x for c in cs)/n,
                   sum(c.y for c in cs)/n,
                   sum(c.z for c in cs)/n))


def _head_front_profile_y(head_obj, z_min, z_max, top_n):
    """Return average Y of the `top_n` head verts in the [z_min, z_max] band
    that have the most-negative Y (most forward). Excludes recessed cavity
    verts because they have less-negative Y than the chin/upper-lip outline."""
    cs = [(head_obj.matrix_world @ v.co) for v in head_obj.data.vertices]
    in_band = [c for c in cs if z_min <= c.z <= z_max]
    if not in_band:
        return None
    in_band.sort(key=lambda c: c.y)  # most-negative Y first
    front = in_band[:max(1, min(top_n, len(in_band)))]
    return sum(c.y for c in front) / len(front)


def _translate_object(obj, delta):
    """Translate `obj` in world space by `delta`. Uses parent matrix
    inverse to set obj.location correctly under any parent."""
    if obj.parent is not None:
        pw_inv = obj.parent.matrix_world.inverted()
        delta_local = pw_inv.to_3x3() @ delta
        obj.location = (obj.location[0] + delta_local.x,
                        obj.location[1] + delta_local.y,
                        obj.location[2] + delta_local.z)
    else:
        obj.location = (obj.location[0] + delta.x,
                        obj.location[1] + delta.y,
                        obj.location[2] + delta.z)


# --------------------------------- ENTRY ------------------------------------
def weld_lip_to_head(cfg):
    lip = bpy.data.objects.get(cfg["lip_object"])
    head = bpy.data.objects.get(cfg["head_object"])
    if lip is None or head is None:
        raise RuntimeError(f"missing object: lip={lip}, head={head}")
    if lip.type != 'MESH' or head.type != 'MESH':
        raise RuntimeError(f"lip/head must be MESH; got {lip.type}/{head.type}")

    mode = cfg.get("mode", "translate")
    print(f"=== weld_lip_to_head ({mode}): '{lip.name}' -> '{head.name}' ===")

    # 1. Detect silhouette (used by both modes)
    sil_verts, sil_loops = _find_silhouette_verts(
        lip, cfg["silhouette_face_normal_y_threshold"])
    if not sil_verts:
        print(f"  no silhouette detected -- abort")
        return 0
    print(f"  silhouette: {len(sil_verts)} verts in {len(sil_loops)} loops "
          f"({sorted([len(L) for L in sil_loops], reverse=True)})")

    sil_centroid = _world_centroid(lip, sil_verts)
    sil_zs = [(lip.matrix_world @ lip.data.vertices[vi].co).z for vi in sil_verts]
    z_min, z_max = min(sil_zs), max(sil_zs)
    print(f"  silhouette centroid Y={sil_centroid.y:.4f}  "
          f"Z band [{z_min:.4f},{z_max:.4f}]")

    if mode == "translate":
        target_y = _head_front_profile_y(
            head, z_min, z_max, cfg.get("front_profile_avg_count", 8))
        if target_y is None:
            print(f"  no head verts in lip Z band -- abort")
            return 0
        standoff = cfg.get("standoff_mm", 0.0) / 1000.0
        # standoff_mm pulls lips forward (more negative Y in this coord system)
        target_y -= standoff
        delta_y = target_y - sil_centroid.y
        print(f"  head front profile Y={target_y:.4f} (avg of "
              f"{cfg.get('front_profile_avg_count',8)} front-most verts "
              f"in lip Z band, standoff={standoff*1000:.1f}mm)")
        print(f"  translating lip on Y by {delta_y*1000:.2f}mm")
        _translate_object(lip, Vector((0.0, delta_y, 0.0)))
        return len(sil_verts)

    elif mode == "snap_verts":
        from mathutils.bvhtree import BVHTree
        bm = bmesh.new(); bm.from_mesh(head.data); bm.transform(head.matrix_world)
        bm.faces.ensure_lookup_table(); bvh = BVHTree.FromBMesh(bm); bm.free()
        bpy.context.view_layer.objects.active = lip
        bpy.ops.object.mode_set(mode='EDIT')
        bml = bmesh.from_edit_mesh(lip.data)
        bml.verts.ensure_lookup_table()
        inv = lip.matrix_world.inverted()
        max_d = cfg["max_snap_distance_m"]
        snapped = 0; skipped = 0
        for vi in sil_verts:
            world = lip.matrix_world @ bml.verts[vi].co
            hit = bvh.find_nearest(world)
            if hit is None or hit[0] is None:
                skipped += 1; continue
            dist = (hit[0] - world).length
            if dist > max_d:
                skipped += 1; continue
            bml.verts[vi].co = inv @ hit[0]
            snapped += 1
        bmesh.update_edit_mesh(lip.data, loop_triangles=True, destructive=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  snapped {snapped} verts (skipped {skipped})")
        return snapped

    else:
        raise ValueError(f"unknown mode '{mode}'")


if __name__ == "__main__":
    weld_lip_to_head(CONFIG)
