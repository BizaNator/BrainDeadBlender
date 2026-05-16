"""
cleanup_lowpoly_head.py

Cleanup pass for a lowpoly head mesh: strips interior geometry (closed eyeball
shells, mouth-bag, ear cavities, etc.) and tidies up topology, leaving an
outer-shell-only mesh with the expected open boundaries (eye sockets, mouth,
nostrils, ear holes, neck cut).

Separate from headswap_transfer.py so you can:
  * Run it on a raw import BEFORE the headswap, producing a clean source mesh
    that the transfer doesn't have to fight with.
  * Run it on the rigged output AFTER the headswap, when the binding is fine
    but the destination mesh still carries junk geometry the donor didn't.
  * Skip it entirely on a head that's already clean.

The script never touches vertex groups, shape keys, UVs, materials, or
modifiers -- only mesh topology -- so it's safe to apply at any pipeline
stage. Vertex groups whose verts get deleted shrink naturally; the data
structures stay intact.

Pipeline
--------
    1. (optional) Merge by distance -- collapses near-duplicate verts.
    2. (optional) Delete non-largest connected face-components -- removes
       fully-disconnected interior shells (e.g. a closed eyeball mesh that
       isn't welded to the outer skin).
    3. (optional) Remove back-to-back face pairs -- for two faces sharing an
       edge with opposing normals (one points outward, one inward), delete
       the inward-pointing one. Catches interior shells that ARE welded to
       the outer skin through shared edges (e.g. an inside-the-socket eyelid
       cap that shares its rim with the outer eyelid).
    4. (optional) Delete loose verts/edges (no linked faces).
    5. (optional) Recalculate normals outside.
    6. Audit report -- non-manifold edges, boundary edges, face/vert/edge
       counts, connected-component breakdown.

The "inward-facing" detection uses the mesh centroid as a reference point and
deletes whichever face of a back-to-back pair has its normal pointing more
toward that centroid. Works for any roughly-convex blob (a head); won't be
reliable for a U-shape or hollow ring.

Designed to drop into the BrainDeadBlender add-on alongside headswap_transfer.
Every step is a pure function; `cleanup_lowpoly_head(cfg)` is the orchestrator.
"""

import bpy
import bmesh
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    # Object name (in current .blend) to clean. MUST be a MESH.
    "target": "LowPolyHead_Rigged",

    # --- step flags ---
    "merge_by_distance": True,
    "merge_distance": 0.0002,           # ~0.2mm at 1m head scale

    "drop_stray_components": True,      # delete every face-component except the largest

    "remove_back_to_back_faces": True,
    "b2b_normal_dot_max": -0.5,         # opposing-normal threshold (dot < this = b2b pair)
    "b2b_keep_outward": True,           # of the pair, keep the face whose normal points away from mesh centroid

    "delete_loose": True,               # loose verts + loose edges (no linked faces)
    "recalc_normals_outside": True,

    # --- audit thresholds (informational) ---
    "audit_b2b_dot": -0.5,
}


# ------------------------------- UTILITIES ----------------------------------
def _ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def _select_only(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _face_components(bm):
    """Return list[set[face_index]] -- connected components by shared edges."""
    bm.faces.ensure_lookup_table()
    visited = set()
    comps = []
    for f in bm.faces:
        if f.index in visited:
            continue
        stack = [f]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur.index in comp:
                continue
            comp.add(cur.index)
            for e in cur.edges:
                for fn in e.link_faces:
                    if fn.index not in comp:
                        stack.append(fn)
        visited |= comp
        comps.append(comp)
    return comps


def _mesh_centroid(bm):
    if not bm.verts:
        return Vector((0, 0, 0))
    s = Vector((0, 0, 0))
    for v in bm.verts:
        s += v.co
    return s / len(bm.verts)


# --------------------------------- STEPS ------------------------------------
def merge_by_distance(obj, distance):
    """Collapse near-duplicate vertices in-place using bmesh remove_doubles."""
    _select_only(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=distance)
    bpy.ops.object.mode_set(mode='OBJECT')


def drop_stray_components(obj):
    """Delete all but the largest connected face-component (+orphan verts)."""
    _ensure_object_mode()
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    comps = _face_components(bm)
    if len(comps) <= 1:
        bm.free()
        return 0, 0

    comps.sort(key=len, reverse=True)
    del_faces = []
    for comp in comps[1:]:
        for fi in comp:
            del_faces.append(bm.faces[fi])
    bmesh.ops.delete(bm, geom=del_faces, context='FACES')

    bm.verts.ensure_lookup_table()
    orphans = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=orphans, context='VERTS')

    bm.to_mesh(me)
    bm.free()
    return len(del_faces), len(orphans)


def remove_back_to_back_faces(obj, dot_max=-0.5, keep_outward=True):
    """
    Find edges shared by two faces with opposing normals (dot < dot_max).
    Delete one face of each pair -- the one whose normal points MORE toward
    the mesh centroid (interior shell), keeping the outward one.

    If keep_outward=False, just deletes one face per pair without picking.
    """
    _ensure_object_mode()
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    centroid = _mesh_centroid(bm)
    inward_face_idxs = set()
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        f1, f2 = e.link_faces
        if f1.normal.dot(f2.normal) >= dot_max:
            continue
        if not keep_outward:
            inward_face_idxs.add(f1.index)
            continue
        # Outward direction from centroid to each face's centroid
        c1 = sum((v.co for v in f1.verts), Vector((0, 0, 0))) / len(f1.verts)
        c2 = sum((v.co for v in f2.verts), Vector((0, 0, 0))) / len(f2.verts)
        out1 = (c1 - centroid).normalized() if (c1 - centroid).length > 1e-9 else Vector((0, 1, 0))
        out2 = (c2 - centroid).normalized() if (c2 - centroid).length > 1e-9 else Vector((0, 1, 0))
        d1 = f1.normal.dot(out1)
        d2 = f2.normal.dot(out2)
        inward_face_idxs.add((f1 if d1 < d2 else f2).index)

    if not inward_face_idxs:
        bm.free()
        return 0, 0

    del_faces = [bm.faces[fi] for fi in inward_face_idxs]
    bmesh.ops.delete(bm, geom=del_faces, context='FACES')

    bm.verts.ensure_lookup_table()
    orphans = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=orphans, context='VERTS')

    bm.to_mesh(me)
    bm.free()
    return len(del_faces), len(orphans)


def delete_loose(obj):
    """Remove loose verts (no linked edges) and loose edges (no linked faces)."""
    _select_only(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode='OBJECT')


def recalc_normals_outside(obj):
    """Make all face normals consistent and point outward from the mesh hull."""
    _select_only(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def audit_mesh(obj, b2b_dot=-0.5):
    """Print mesh-integrity stats. Returns dict for programmatic use."""
    _ensure_object_mode()
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    edges_3plus = sum(1 for e in bm.edges if len(e.link_faces) >= 3)
    loose_verts = sum(1 for v in bm.verts if not v.link_edges)
    loose_edges = sum(1 for e in bm.edges if not e.link_faces)
    b2b = sum(
        1 for e in bm.edges
        if len(e.link_faces) == 2
        and e.link_faces[0].normal.dot(e.link_faces[1].normal) < b2b_dot
    )
    comps = _face_components(bm)
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    bm.free()

    stats = {
        "verts": len(me.vertices),
        "edges": len(me.edges),
        "faces": len(me.polygons),
        "non_manifold_edges": non_manifold,
        "boundary_edges": boundary,
        "edges_3plus_faces": edges_3plus,
        "loose_verts": loose_verts,
        "loose_edges": loose_edges,
        "back_to_back_pairs": b2b,
        "components": len(comps),
        "component_sizes_top10": comp_sizes[:10],
    }
    print(f"[audit] {obj.name}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


# ----------------------------- ORCHESTRATOR ---------------------------------
def cleanup_lowpoly_head(cfg):
    obj = bpy.data.objects.get(cfg["target"])
    if obj is None or obj.type != 'MESH':
        raise RuntimeError(f"target '{cfg['target']}' not found or not a MESH")

    print(f"=== cleanup_lowpoly_head -> {obj.name} ===")
    print("[before]")
    audit_mesh(obj, b2b_dot=cfg["audit_b2b_dot"])

    if cfg["merge_by_distance"]:
        merge_by_distance(obj, cfg["merge_distance"])
        print(f"[step] merge_by_distance({cfg['merge_distance']}) done")

    if cfg["drop_stray_components"]:
        df, dv = drop_stray_components(obj)
        print(f"[step] drop_stray_components: removed {df} faces, {dv} orphan verts")

    if cfg["remove_back_to_back_faces"]:
        df, dv = remove_back_to_back_faces(
            obj,
            dot_max=cfg["b2b_normal_dot_max"],
            keep_outward=cfg["b2b_keep_outward"],
        )
        print(f"[step] remove_back_to_back_faces: removed {df} faces, {dv} orphan verts")
        # b2b removal often spawns new strays -- run another component pass
        if cfg["drop_stray_components"]:
            df, dv = drop_stray_components(obj)
            print(f"[step] drop_stray_components (post-b2b): removed {df} faces, {dv} orphan verts")

    if cfg["delete_loose"]:
        delete_loose(obj)
        print("[step] delete_loose done")

    if cfg["recalc_normals_outside"]:
        recalc_normals_outside(obj)
        print("[step] recalc_normals_outside done")

    print("[after]")
    return audit_mesh(obj, b2b_dot=cfg["audit_b2b_dot"])


# --------------------------------- ENTRY ------------------------------------
if __name__ == "__main__":
    cleanup_lowpoly_head(CONFIG)
