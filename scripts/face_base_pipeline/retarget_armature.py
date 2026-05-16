"""
retarget_armature.py

Move armature bone rest positions onto the destination mesh's actual
anatomy, so rotations pivot around where the verts are -- not where the
source character's bones happened to sit.

Background
----------
headswap_transfer.py copies a donor (Penny) head's bone WEIGHTS onto a new
lowpoly head, but the bone REST POSITIONS stay at Penny's positions. When
the lowpoly head has different proportions (larger forehead, smaller
eye-to-mouth distance), Penny's eye bones can end up in the new head's
forehead -- the weights still drive the right verts, but the rotation
pivot is in the wrong place and the deformation looks off.

This script walks every targeted bone, computes the world-space centroid
of verts weighted to that bone (across one or more destination meshes),
and moves the bone HEAD to that centroid. Tail direction is preserved by
translating both head and tail by the same delta.

Each bone is repositioned INDEPENDENTLY of its parent -- bones in Blender
can be disconnected (bone.use_connect = False), so a child bone's head
doesn't have to coincide with its parent's tail. Vertex weights reference
bones by NAME, not position, so existing weights keep working unchanged.
The rest pose of the mesh is preserved (because at rest, pose_matrix ==
rest_matrix, so the deformation evaluates to identity regardless of where
the bone sits).

Designed to drop into the BrainDeadBlender add-on alongside the other
post-headswap cleanup scripts.
"""

import bpy
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "armature":  "Fortnite_Armature",

    # Destination meshes to scan for weighted verts. Each bone gathers verts
    # from ALL of these. Usually you want the rigged head + any extracted
    # part meshes (eyes/teeth/tongue) that share the same armature.
    "dst_meshes": ["LowPolyHead_Rigged", "LowPolyHead_Parts"],

    # Bone name patterns that get retargeted. Anything not matching is left
    # alone (e.g. root/spine/neck, FX/socket bones).
    "include_patterns": [
        "C_jaw",
        "L_eye", "R_eye",       # also matches L_eye_lid_*, R_eye_lid_*
        "_brow",
        "_lip_", "C_lip_",
        "_cheek",
        "_nose",
        "teeth_upper", "teeth_lower",
        "tongue",
    ],

    # Don't retarget if fewer than this many verts are weighted to the bone
    # (above min_weight). Avoids snapping a bone onto a single stray vert.
    "min_weighted_verts": 3,
    "min_weight": 0.1,

    # Report-only mode -- prints planned moves without applying them.
    "dry_run": False,
}


# ------------------------------- UTILITIES ----------------------------------
def _matches_any(name, patterns):
    return any(p in name for p in patterns)


def _weighted_centroid(mesh_objs, vg_name, min_weight):
    """World-space centroid of verts weighted to vg_name across all mesh_objs."""
    pts = []
    for obj in mesh_objs:
        vg = obj.vertex_groups.get(vg_name)
        if vg is None:
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index and g.weight > min_weight:
                    pts.append(mw @ v.co)
                    break
    if not pts:
        return None, 0
    return sum(pts, Vector((0, 0, 0))) / len(pts), len(pts)


# --------------------------------- ENTRY ------------------------------------
def retarget_armature(cfg):
    arm = bpy.data.objects.get(cfg["armature"])
    if arm is None or arm.type != 'ARMATURE':
        raise RuntimeError(f"armature '{cfg['armature']}' not found")

    dst_meshes = [bpy.data.objects.get(n) for n in cfg["dst_meshes"]]
    dst_meshes = [m for m in dst_meshes if m and m.type == 'MESH']
    if not dst_meshes:
        raise RuntimeError("no destination meshes found")

    patterns = cfg["include_patterns"]
    min_n = cfg["min_weighted_verts"]
    min_w = cfg["min_weight"]
    dry = cfg["dry_run"]

    print(f"=== retarget_armature -> {arm.name} ===")
    print(f"  meshes: {[m.name for m in dst_meshes]}")
    print(f"  patterns: {patterns}")
    print(f"  min_weighted_verts={min_n}  min_weight={min_w}  dry_run={dry}")

    # Plan moves in object mode (read), apply in edit mode (write)
    arm_mw_inv = arm.matrix_world.inverted()
    moves = []  # (bone_name, current_local_head, new_local_head, delta_local, n_verts)

    for bone in arm.data.bones:
        if not _matches_any(bone.name, patterns):
            continue
        centroid_world, n = _weighted_centroid(dst_meshes, bone.name, min_w)
        if centroid_world is None or n < min_n:
            continue
        new_local_head = arm_mw_inv @ centroid_world
        delta = new_local_head - bone.head_local
        moves.append((bone.name, bone.head_local.copy(), new_local_head, delta, n))

    moves.sort(key=lambda m: -m[3].length)
    print(f"\n[plan {len(moves)} moves -- sorted by displacement]")
    for name, old, new, delta, n in moves:
        print(f"  {name}: ({old.x:6.3f},{old.y:6.3f},{old.z:6.3f}) "
              f"-> ({new.x:6.3f},{new.y:6.3f},{new.z:6.3f}) "
              f"delta={delta.length*100:5.2f}cm  verts={n}")

    if dry:
        return {"planned": [m[0] for m in moves], "applied": []}

    # Apply: enter edit mode, move bones (preserving tail direction)
    prev_active = bpy.context.view_layer.objects.active
    prev_mode = bpy.context.mode
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')

    move_map = {name: delta for name, _, _, delta, _ in moves}
    applied = []
    for eb in arm.data.edit_bones:
        d = move_map.get(eb.name)
        if d is None:
            continue
        eb.head = eb.head + d
        eb.tail = eb.tail + d
        applied.append(eb.name)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = prev_active

    print(f"\n[applied {len(applied)} bone moves]")
    return {"planned": [m[0] for m in moves], "applied": applied}


if __name__ == "__main__":
    retarget_armature(CONFIG)
