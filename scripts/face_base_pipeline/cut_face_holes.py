"""
cut_face_holes.py

Open eye sockets and (optionally) the mouth on a rigged lowpoly head so
separate submesh eyes / teeth / lips can sit inside without fighting the
head's own surface.

Without this step the head still has its baked eyelid skin + lip skin --
which (a) Z-fights with the submesh eyes/lips and (b) deforms badly when
the eye and lip bones rotate (lid weights smear across forehead/cheek).
After this script, the head has CLEAN open holes at each eye and a clean
opening at the mouth; the submesh parts (eyes, teeth, tongue, custom
lips) carry the visible deformation.

Strategy
--------
For each hole:
  1. Compute the hole center in mesh-local space from the relevant bone's
     rest position (more reliable than weight centroid; the bone has
     already been retargeted to anatomy by retarget_armature).
  2. Compute a radius from the bbox of "interior" verts -- those weighted
     heavily to the corresponding bone group on the HEAD (e.g. eye lid
     weights for the eye hole).
  3. Delete every head face whose centroid falls inside that sphere AND
     whose own weight to the interior group is high (so we don't take
     out skin that just happens to be near, only the actual interior
     fill).

Designed to run AFTER:
    align_landmarks -> headswap_transfer (preserve_geometry) -> restore_geometry
    -> cleanup_face_weights -> retarget_armature.

Drop into the BrainDeadBlender add-on after the post-headswap cleanup
pass.
"""

import bpy
import bmesh
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "head":     "LowPolyHead_Rigged",
    "armature": "Fortnite_Armature",

    # Each "hole" cuts faces around a bone's rest position whose verts are
    # weighted to one of `interior_groups`. The radius is computed from the
    # bbox of weighted verts, multiplied by `radius_scale`.
    "holes": [
        {
            "name": "L_eye",
            "anchor_bone": "L_eye",
            "interior_groups": ["L_eye", "L_eye_lid_upper_mid", "L_eye_lid_lower_mid"],
            "radius_scale": 0.55,    # tighter -- avoid taking cheek/forehead skin
            "min_weight_face": 0.5,  # only faces dominated by lid weight
        },
        {
            "name": "R_eye",
            "anchor_bone": "R_eye",
            "interior_groups": ["R_eye", "R_eye_lid_upper_mid", "R_eye_lid_lower_mid"],
            "radius_scale": 0.55,
            "min_weight_face": 0.5,
        },
        # Mouth is OPTIONAL -- user may want to keep their lip ring and
        # only remove the interior fill. Toggle via `enabled`.
        {
            "name": "mouth",
            "enabled": False,
            "anchor_bone": "C_lip_upper_mid",
            "interior_groups": ["C_lip_upper_mid", "C_lip_lower_mid",
                                "L_lip_corner", "R_lip_corner",
                                "L_lip_upper_outer", "L_lip_lower_outer",
                                "R_lip_upper_outer", "R_lip_lower_outer",
                                "teeth_upper", "teeth_lower", "tongue"],
            "radius_scale": 0.7,
            "min_weight_face": 0.3,
        },
    ],

    # If True, after cutting holes, also recompute boundary normals so the
    # new edge loops light correctly.
    "recalc_normals": True,

    # If True, after cutting, also REMOVE every vert from the listed vgroups
    # on the head (so e.g. eye rotation does not drag remaining head verts
    # via leftover lid weights). The separate submesh parts (eyes / teeth /
    # tongue) carry the deformation instead.
    "zero_remaining_weights": True,
    "zero_groups": ["L_eye", "R_eye",
                    "L_eye_lid_upper_mid", "L_eye_lid_lower_mid",
                    "R_eye_lid_upper_mid", "R_eye_lid_lower_mid",
                    "L_brow_outer", "L_brow_mid",
                    "R_brow_outer", "R_brow_mid",
                    "C_brow_mid"],
}


# ------------------------------- UTILITIES ----------------------------------
def _bone_world_head(arm, bone_name):
    b = arm.data.bones.get(bone_name)
    if b is None:
        return None
    return arm.matrix_world @ b.head_local


def _vg_weighted_world_bbox(obj, vg_names, min_weight=0.1):
    """Bbox of verts weighted to any of vg_names above min_weight."""
    indices = {}
    for n in vg_names:
        vg = obj.vertex_groups.get(n)
        if vg:
            indices[n] = vg.index
    if not indices:
        return None
    mw = obj.matrix_world
    pts = []
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group in indices.values() and g.weight > min_weight:
                pts.append(mw @ v.co)
                break
    if not pts:
        return None
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    return (Vector((min(xs), min(ys), min(zs))),
            Vector((max(xs), max(ys), max(zs))),
            len(pts))


def _face_weight_sum(face, vg_indices, vert_groups):
    """Average over face verts of their summed weights to vg_indices."""
    total = 0.0
    for v in face.verts:
        w = 0.0
        for g in vert_groups[v.index]:
            if g[0] in vg_indices:
                w += g[1]
        total += w
    return total / max(1, len(face.verts))


# --------------------------------- STEPS ------------------------------------
def cut_face_holes(cfg):
    head = bpy.data.objects.get(cfg["head"])
    if head is None or head.type != 'MESH':
        raise RuntimeError(f"head '{cfg['head']}' not found")
    arm = bpy.data.objects.get(cfg["armature"])
    if arm is None or arm.type != 'ARMATURE':
        raise RuntimeError(f"armature '{cfg['armature']}' not found")

    print(f"=== cut_face_holes -> {head.name} ===")
    print(f"  starting: {len(head.data.vertices)}v, {len(head.data.polygons)}f")

    # Resolve hole anchor + radius per hole. Anchor is the world centroid of
    # interior-group weighted verts on the HEAD (not the bone position --
    # bones live in armature-local space which may be offset from the mesh's
    # world position). Radius is the bbox half-diagonal of those verts,
    # scaled.
    plans = []
    for hole in cfg["holes"]:
        if not hole.get("enabled", True):
            continue
        bbox = _vg_weighted_world_bbox(head, hole["interior_groups"])
        if bbox is None:
            print(f"  skip '{hole['name']}': no interior-group weights on head")
            continue
        bmin, bmax, n = bbox
        anchor = (bmin + bmax) * 0.5
        radius = (bmax - bmin).length * 0.5 * hole["radius_scale"]
        plans.append({
            "name": hole["name"],
            "anchor": anchor,
            "radius": radius,
            "interior_groups": hole["interior_groups"],
            "min_weight_face": hole["min_weight_face"],
            "n_weighted": n,
        })
        print(f"  '{hole['name']}': anchor=({anchor.x:.3f},{anchor.y:.3f},{anchor.z:.3f}) "
              f"r={radius*100:.1f}cm  weighted_verts={n}")

    # Build a flat lookup: vert_idx -> list[(vg_idx, weight)]
    vert_groups = [[(g.group, g.weight) for g in v.groups] for v in head.data.vertices]

    # bmesh once, evaluate all plans
    me = head.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    mw = head.matrix_world

    faces_to_delete = set()
    per_plan = {}
    for plan in plans:
        vg_indices = set()
        for n in plan["interior_groups"]:
            vg = head.vertex_groups.get(n)
            if vg:
                vg_indices.add(vg.index)
        if not vg_indices:
            continue
        plan_faces = []
        for f in bm.faces:
            if f.index in faces_to_delete:
                continue
            cw = sum((mw @ v.co for v in f.verts), Vector((0, 0, 0))) / len(f.verts)
            if (cw - plan["anchor"]).length > plan["radius"]:
                continue
            w = _face_weight_sum(f, vg_indices, vert_groups)
            if w >= plan["min_weight_face"]:
                plan_faces.append(f)
        for f in plan_faces:
            faces_to_delete.add(f.index)
        per_plan[plan["name"]] = len(plan_faces)
        print(f"    '{plan['name']}': flagged {len(plan_faces)} faces")

    if faces_to_delete:
        bmesh.ops.delete(bm,
                         geom=[bm.faces[i] for i in faces_to_delete],
                         context='FACES')
        bm.verts.ensure_lookup_table()
        orphans = [v for v in bm.verts if not v.link_faces]
        if orphans:
            bmesh.ops.delete(bm, geom=orphans, context='VERTS')
            print(f"  removed {len(orphans)} orphan verts")
    bm.to_mesh(me)
    bm.free()

    if cfg["recalc_normals"]:
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.select_all(action='DESELECT')
        head.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

    weights_removed = 0
    if cfg.get("zero_remaining_weights", False):
        for gn in cfg.get("zero_groups", []):
            vg = head.vertex_groups.get(gn)
            if vg is None:
                continue
            verts = set()
            for v in head.data.vertices:
                for g in v.groups:
                    if g.group == vg.index:
                        verts.add(v.index)
                        break
            if verts:
                vg.remove(list(verts))
                weights_removed += len(verts)
        print(f"  zeroed {weights_removed} vert weights across "
              f"{len(cfg['zero_groups'])} eye/brow groups (head decoupled "
              f"from those bones)")

    print(f"\n[done] {head.name}: {len(me.vertices)}v {len(me.polygons)}f after cutting "
          f"{sum(per_plan.values())} faces across {len(per_plan)} hole(s)")
    return per_plan


if __name__ == "__main__":
    cut_face_holes(CONFIG)
