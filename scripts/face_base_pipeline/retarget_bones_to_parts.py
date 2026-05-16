"""
retarget_bones_to_parts.py

Move face bone HEADS to match the world bbox centers of their rigged
parts, so each bone's rotation pivots at the visible part center.

Why
---
After the user manually positions the split parts (Eye_L, Eye_R,
Teeth_Upper, Teeth_Lower, Tongue, CustomLips) onto the lowpoly head,
the corresponding bones (L_eye, R_eye, teeth_upper, ..., C_jaw, etc.)
are still at the donor's original positions. Bone rotation then pivots
around a remote point and translates the part as a whole instead of
rotating it in place.

What it does
------------
    For each (part_name, primary_bone, ...) entry in CONFIG['parts']:
      1. Compute the part's world bbox center.
      2. Move the bone's head to that point in armature-local space.
      3. Preserve the bone's length + orientation by translating the
         tail by the same delta.
      4. Optionally re-orient the bone's tail along a configured world
         direction (e.g. eye-bones should point along +Y / forward so
         their local rotation axes are predictable).

Run in OBJECT mode -- the script switches to EDIT mode internally to
edit bone head/tail, then back to OBJECT.

Designed to drop into the BrainDeadBlender add-on after the user has
hand-positioned the face parts.
"""

import bpy
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "armature": "Fortnite_Armature",

    # Each entry snaps `primary_bone` to the bbox center of `part`.
    # `tail_dir_world` (optional 3-tuple): re-orient the bone so its tail
    # points in this world direction (length kept).
    # `extra_bones`: additional bones to translate by the same delta
    # (useful for groups like upper-mid + upper-outer that move together).
    "parts": [
        {
            "part": "Eye_L",
            "primary_bone": "L_eye",
            "tail_dir_world": (0, -1, 0),   # eyes look forward (-Y)
        },
        {
            "part": "Eye_R",
            "primary_bone": "R_eye",
            "tail_dir_world": (0, -1, 0),
        },
        {
            "part": "Teeth_Upper",
            "primary_bone": "teeth_upper",
        },
        {
            "part": "Teeth_Lower",
            "primary_bone": "teeth_lower",
        },
        {
            "part": "Tongue",
            "primary_bone": "tongue",
        },
    ],

    # Lip-control bones aren't snapped to a part bbox center -- they live
    # on the boundary of CustomLips. Optional: derive their positions from
    # the lip mesh's bbox (upper/lower mid + corners).
    "snap_lip_bones": True,
    "lips_part": "CustomLips",
    "lip_bone_map": {
        # bone_name : (x_frac, y_frac, z_frac)
        # within the lip bbox. x: 0=left, 1=right (then mirrored).
        # y: 0=back, 1=front (we use 1.0 = front-most)
        # z: 0=bottom, 1=top
        "C_lip_upper_mid":   ( 0.5, 1.0, 0.85),
        "C_lip_lower_mid":   ( 0.5, 1.0, 0.15),
        "L_lip_corner":      ( 1.0, 1.0, 0.5),
        "R_lip_corner":      ( 0.0, 1.0, 0.5),
        "L_lip_upper_outer": ( 0.75, 1.0, 0.75),
        "R_lip_upper_outer": ( 0.25, 1.0, 0.75),
        "L_lip_lower_outer": ( 0.75, 1.0, 0.25),
        "R_lip_lower_outer": ( 0.25, 1.0, 0.25),
    },
}


# ------------------------------- HELPERS ------------------------------------
def _bbox_world(obj):
    ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not ws:
        return None
    xs = [w.x for w in ws]; ys = [w.y for w in ws]; zs = [w.z for w in ws]
    return (Vector((min(xs), min(ys), min(zs))),
            Vector((max(xs), max(ys), max(zs))))


def _world_to_arm_local(arm, p_world):
    return arm.matrix_world.inverted() @ p_world


def _snap_bone(eb, target_local, tail_dir_world=None, arm_world_inv=None):
    """Move edit_bone's head to `target_local`. Preserve length; either
    keep current direction or re-orient along `tail_dir_world`."""
    length = (eb.tail - eb.head).length
    if length < 1e-6:
        length = 0.01
    eb.head = target_local
    if tail_dir_world is not None:
        # Transform world direction into armature-local
        dir_local = (arm_world_inv.to_3x3() @ Vector(tail_dir_world)).normalized()
        eb.tail = target_local + dir_local * length
    else:
        # Keep current direction
        old_dir = (eb.tail - eb.head)
        if old_dir.length < 1e-6:
            old_dir = Vector((0, 0.01, 0))
        eb.tail = target_local + old_dir.normalized() * length


# --------------------------------- ENTRY ------------------------------------
def retarget_bones_to_parts(cfg):
    arm = bpy.data.objects.get(cfg["armature"])
    if arm is None or arm.type != 'ARMATURE':
        raise RuntimeError(f"armature '{cfg['armature']}' not found")

    print(f"=== retarget_bones_to_parts -> {arm.name} ===")

    # Gather targets in WORLD space before entering edit mode
    arm_world_inv = arm.matrix_world.inverted()
    targets = []  # list of (bone_name, target_local, tail_dir_world)

    skip_names = set(cfg.get("skip_parts", []))
    skip_hidden = cfg.get("skip_if_hidden", True)
    for entry in cfg["parts"]:
        if entry["part"] in skip_names:
            print(f"  skip '{entry['part']}': in skip_parts")
            continue
        part = bpy.data.objects.get(entry["part"])
        if part is None:
            print(f"  skip '{entry['part']}': not in scene")
            continue
        if skip_hidden and part.hide_get():
            print(f"  skip '{entry['part']}': hidden in viewport")
            continue
        bbox = _bbox_world(part)
        if bbox is None:
            print(f"  skip '{entry['part']}': empty mesh")
            continue
        bmin, bmax = bbox
        center_world = (bmin + bmax) * 0.5
        center_local = arm_world_inv @ center_world
        targets.append((entry["primary_bone"], center_local,
                        entry.get("tail_dir_world")))
        print(f"  '{entry['part']}' -> bone '{entry['primary_bone']}'  "
              f"world center=({center_world.x:.3f},{center_world.y:.3f},{center_world.z:.3f})")

    # Optional: lip-control bones derived from CustomLips bbox
    if cfg.get("snap_lip_bones", False):
        lips = bpy.data.objects.get(cfg.get("lips_part", "CustomLips"))
        if lips is not None:
            bbox = _bbox_world(lips)
            if bbox is not None:
                bmin, bmax = bbox
                size = bmax - bmin
                for bone_name, (fx, fy, fz) in cfg["lip_bone_map"].items():
                    p = Vector((bmin.x + fx * size.x,
                                bmin.y + fy * size.y,
                                bmin.z + fz * size.z))
                    p_local = arm_world_inv @ p
                    targets.append((bone_name, p_local, None))
                print(f"  CustomLips bbox: x={size.x*100:.1f}cm y={size.y*100:.1f}cm z={size.z*100:.1f}cm "
                      f"-> {len(cfg['lip_bone_map'])} lip bones")

    # Apply in edit mode
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    moved = 0
    for bone_name, target_local, tail_dir in targets:
        eb = arm.data.edit_bones.get(bone_name)
        if eb is None:
            print(f"    skip bone '{bone_name}': not in armature")
            continue
        old_head = eb.head.copy()
        _snap_bone(eb, target_local, tail_dir_world=tail_dir, arm_world_inv=arm_world_inv)
        delta = (eb.head - old_head).length * 1000
        print(f"    '{bone_name}': moved {delta:.1f}mm")
        moved += 1
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"\n[done] snapped {moved} bones to part centers")
    return moved


if __name__ == "__main__":
    retarget_bones_to_parts(CONFIG)
