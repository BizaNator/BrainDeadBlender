"""
fit_face_parts.py

Position + clean-weight each independent face submesh (Eye_L, Eye_R,
Teeth_Upper, Teeth_Lower, Tongue) so each rides exactly one bone with no
weight bleed from the donor.

Why
---
After `split_face_parts`, each submesh inherits the donor's full weight
map -- which means Eye_L has weights on jaw, lip, cheek bones too, and
when any of those rotate the eyeball drifts. Likewise Teeth_Upper may
have stray jaw weight that drags it down when the mouth opens.

What it does (per part)
-----------------------
    1. Clear ALL existing vertex groups.
    2. Re-add only the part's "primary" bone group with weight=1.0
       (full rigid binding). Optionally add a "secondary" bone (e.g.
       eyelid follow) at lower weight.
    3. Compute the part's current world bbox center, snap to the
       primary bone's world head position (+ optional offset).
    4. Optionally scale to a target dimension (e.g. eye diameter).
    5. Re-parent to the armature with matrix_parent_inverse so the
       transform sticks.

Run after `split_face_parts` (and after `fit_custom_lips` for lips, if
those are managed separately). Per-part overrides live in CONFIG['parts'].
"""

import bpy
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "armature": "Fortnite_Armature",

    # Per-part fitting rules.
    #   primary_bone: bone the part rigidly follows (weight 1.0)
    #   anchor_bone:  bone whose world head becomes the part's bbox center
    #                 (defaults to primary_bone)
    #   anchor_offset_world: (x,y,z) world-space nudge from the anchor
    #   target_dim_m: optional target X-bbox size (e.g. eyeball ~10mm)
    "parts": [
        {
            "name": "Eye_L",
            "primary_bone": "L_eye",
            "target_dim_m": None,
            "anchor_offset_world": (0, 0, 0),
        },
        {
            "name": "Eye_R",
            "primary_bone": "R_eye",
            "target_dim_m": None,
            "anchor_offset_world": (0, 0, 0),
        },
        {
            "name": "Teeth_Upper",
            "primary_bone": "teeth_upper",
            "target_dim_m": None,
            "anchor_offset_world": (0, 0, 0),
        },
        {
            "name": "Teeth_Lower",
            "primary_bone": "teeth_lower",
            "target_dim_m": None,
            "anchor_offset_world": (0, 0, 0),
        },
        {
            "name": "Tongue",
            "primary_bone": "tongue",
            "target_dim_m": None,
            "anchor_offset_world": (0, 0, 0),
        },
    ],

    # If True, skip the positional snap (just clean weights).
    "weights_only": False,

    # If True, also clear shape keys (some donor parts have inherited
    # shape keys that aren't relevant on the standalone part).
    "clear_shape_keys": False,
}


# ------------------------------- HELPERS ------------------------------------
def _obj(name):
    o = bpy.data.objects.get(name)
    if o is None:
        print(f"  skip: object '{name}' not found")
    return o


def _bbox_world(obj):
    ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not ws:
        return None
    xs = [w.x for w in ws]; ys = [w.y for w in ws]; zs = [w.z for w in ws]
    return (Vector((min(xs), min(ys), min(zs))),
            Vector((max(xs), max(ys), max(zs))))


def _apply_object_transform(obj):
    """Bake current loc/rot/scale into the mesh data so subsequent reads see
    the world positions and the object transform resets to identity."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _fit_one(part_cfg, arm, weights_only, clear_shape_keys):
    obj = _obj(part_cfg["name"])
    if obj is None:
        return None

    primary = part_cfg["primary_bone"]
    bone = arm.data.bones.get(primary)
    if bone is None:
        print(f"  skip '{obj.name}': primary bone '{primary}' not found")
        return None

    # ---- 1. clean weights ----
    while obj.vertex_groups:
        obj.vertex_groups.remove(obj.vertex_groups[0])
    vg = obj.vertex_groups.new(name=primary)
    vg.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')

    # ---- 2. shape keys (optional) ----
    if clear_shape_keys and obj.data.shape_keys:
        obj.shape_key_clear()

    # ---- 3. positional snap ----
    if not weights_only:
        anchor_world = arm.matrix_world @ bone.head_local
        anchor_world = anchor_world + Vector(part_cfg.get("anchor_offset_world", (0, 0, 0)))

        bbox = _bbox_world(obj)
        if bbox is not None:
            bmin, bmax = bbox
            cur_center = (bmin + bmax) * 0.5

            # Scale (optional)
            target = part_cfg.get("target_dim_m")
            if target is not None:
                cur_x = bmax.x - bmin.x
                if cur_x > 1e-6:
                    s = target / cur_x
                    obj.scale = (obj.scale.x * s, obj.scale.y * s, obj.scale.z * s)
                    cur_center = anchor_world  # after scale we'll just place new center

            obj.location = obj.location + (anchor_world - cur_center)

            # Bake so weights/positions match
            _apply_object_transform(obj)

    # ---- 4. parent to armature ----
    mods = [m for m in obj.modifiers if m.type == 'ARMATURE']
    if not mods:
        mod = obj.modifiers.new("Armature", 'ARMATURE')
    else:
        mod = mods[0]
    mod.object = arm
    mod.use_vertex_groups = True
    obj.parent = arm
    obj.matrix_parent_inverse = arm.matrix_world.inverted()

    bbox_after = _bbox_world(obj)
    center_after = (bbox_after[0] + bbox_after[1]) * 0.5 if bbox_after else Vector((0, 0, 0))
    size_after = bbox_after[1] - bbox_after[0] if bbox_after else Vector((0, 0, 0))
    print(f"  '{obj.name}' -> bone '{primary}'  "
          f"center=({center_after.x:.3f},{center_after.y:.3f},{center_after.z:.3f}) "
          f"size=({size_after.x*100:.1f},{size_after.y*100:.1f},{size_after.z*100:.1f})cm")
    return obj


# --------------------------------- ENTRY ------------------------------------
def fit_face_parts(cfg):
    arm = _obj(cfg["armature"])
    if arm is None:
        raise RuntimeError(f"armature '{cfg['armature']}' not found")

    print(f"=== fit_face_parts (armature: {arm.name}) ===")
    skip_names = set(cfg.get("skip_parts", []))
    skip_missing = cfg.get("skip_if_missing", True)
    skip_hidden = cfg.get("skip_if_hidden", True)
    fitted = []
    for part_cfg in cfg["parts"]:
        if part_cfg["name"] in skip_names:
            print(f"  skip '{part_cfg['name']}': in skip_parts")
            continue
        obj = bpy.data.objects.get(part_cfg["name"])
        if obj is None:
            if skip_missing:
                print(f"  skip '{part_cfg['name']}': missing from scene")
                continue
        elif skip_hidden and obj.hide_get():
            print(f"  skip '{part_cfg['name']}': hidden in viewport")
            continue
        out = _fit_one(part_cfg, arm,
                       weights_only=cfg["weights_only"],
                       clear_shape_keys=cfg["clear_shape_keys"])
        if out:
            fitted.append(out)

    print(f"\n[done] fitted {len(fitted)} parts")
    return fitted


if __name__ == "__main__":
    fit_face_parts(CONFIG)
