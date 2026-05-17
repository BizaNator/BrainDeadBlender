"""
fit_accessories.py

Weight non-deforming accessory meshes (eyelids, brows, ears, nose, neck)
to single appropriate bones so they ride the rig without per-vert binding
through BVH transfer.

Why
---
After the head + eyes + lips are bound via `headswap_transfer`, parts
like the eyelids, brows, ears, and nose are still independent meshes
with no weights. Without rigging them, they stay floating at their
authoring positions while the head + bones move.

For low-deformation accessories the simplest correct binding is full
weight (1.0) to one bone:
    - Eyelids ride their corresponding lid bone (for blinks)
    - Brows ride the brow bone (for raise / lower)
    - Ears / nose / neck ride `head` (translate + rotate with the head)

Each accessory becomes parented to the armature with an Armature
modifier; existing weights / parent are cleared so this is idempotent.

Drop into the BrainDeadBlender add-on as the final fitting step after
fit_face_parts + rebind_lip_weights + retarget_bones_to_parts.
"""

import bpy
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "armature": "Fortnite_Armature",

    # Accessory name -> primary bone. If `bone` is missing on the armature
    # the entry falls back to `fallback_bone` (default: "head").
    "accessories": [
        {"name": "Eyelid_L_Upper", "bone": "L_eye_lid_upper_mid"},
        {"name": "Eyelid_L_Lower", "bone": "L_eye_lid_lower_mid"},
        {"name": "Eyelid_R_Upper", "bone": "R_eye_lid_upper_mid"},
        {"name": "Eyelid_R_Lower", "bone": "R_eye_lid_lower_mid"},
        {"name": "Eyebrow_L",      "bone": "L_brow_mid"},
        {"name": "Eyebrow_R",      "bone": "R_brow_mid"},
        {"name": "Ear_L",          "bone": "head"},
        {"name": "Ear_R",          "bone": "head"},
        {"name": "Nose",           "bone": "head"},
        {"name": "Neck",           "bone": "neck_02", "fallback_bone": "head"},
        # Teeth are reusable library parts dropped into each character. They
        # come in from the donor at the donor's mouth position; `fit_to_section`
        # auto-translates them to the merged head's lip cavity centre so they
        # land in the right place per-character with no manual fiddling.
        # Upper anchored to `head` (rigid skull), Lower rides `C_jaw` (drops
        # with jaw open). Kept SEPARATE from the merged head so ARKit lip
        # morphs (mouthSmile, mouthFunnel, etc) don't warp teeth geometry.
        {"name": "Teeth_Upper",    "bone": "head",  "fit_to_section": "lips"},
        {"name": "Teeth_Lower",    "bone": "C_jaw", "fit_to_section": "lips"},
        # Skull would go here too if it's a separate accessory, but the
        # pipeline currently joins Skull into LowPolyHead_Rigged in
        # face_base_apply.
    ],

    # When an accessory has `fit_to_section`, the object's center is translated
    # to that section's center on `section_target` (the merged head). Section
    # tags are written by merge_face_meshes; this only works post-merge.
    "section_target": "LowPolyHead_Rigged",
    "section_attr":   "_section",

    "fallback_bone": "head",

    # Skip accessories whose objects are missing or hidden.
    "skip_if_missing": True,
    "skip_if_hidden":  True,

    # If True, also clear any inherited shape keys (they're rarely useful
    # on these accessories and confuse downstream morph layering).
    "clear_shape_keys": False,
}


# ------------------------------- HELPERS ------------------------------------
def _bone_exists(arm, name):
    return arm.data.bones.get(name) is not None


def _section_center_world(target_obj, section_attr, section_name):
    """Return world-space center of all faces whose `section_attr` value
    equals `section_name`. Used to drop reusable library parts (teeth) into
    the right per-character mouth position post-merge. Returns None if the
    section has no faces."""
    me = target_obj.data
    a = me.attributes.get(section_attr)
    if a is None:
        return None
    vert_idxs = set()
    for fi, p in enumerate(me.polygons):
        if a.data[fi].value.decode('utf-8') == section_name:
            for vi in p.vertices:
                vert_idxs.add(vi)
    if not vert_idxs:
        return None
    mw = target_obj.matrix_world
    coords = [mw @ me.vertices[vi].co for vi in vert_idxs]
    n = len(coords)
    return Vector((sum(c.x for c in coords)/n,
                   sum(c.y for c in coords)/n,
                   sum(c.z for c in coords)/n))


def _translate_to_world_point(obj, target_world):
    """Translate `obj` so its current geometric center lands at `target_world`
    in world space. Preserves object's rotation + scale + parent.
    """
    om = obj.matrix_world
    coords = [om @ v.co for v in obj.data.vertices]
    n = len(coords)
    cur = Vector((sum(c.x for c in coords)/n,
                  sum(c.y for c in coords)/n,
                  sum(c.z for c in coords)/n))
    delta_world = target_world - cur
    # Translation in world space -> add to obj.location regardless of parent,
    # because location is parent-space but the parent's basis only rotates +
    # scales the delta. For most accessories the parent (armature) is at
    # identity, so this works directly; for non-identity parents we transform
    # the delta into parent local space.
    if obj.parent is not None:
        parent_world_inv = obj.parent.matrix_world.inverted()
        # Strip translation component for delta-only transform
        rs_only = parent_world_inv.to_3x3()
        delta_local = rs_only @ delta_world
        obj.location = (obj.location[0] + delta_local.x,
                        obj.location[1] + delta_local.y,
                        obj.location[2] + delta_local.z)
    else:
        obj.location = (obj.location[0] + delta_world.x,
                        obj.location[1] + delta_world.y,
                        obj.location[2] + delta_world.z)


def _fit_one(obj, bone_name, fallback_name, arm, clear_shape_keys,
             section_target=None, section_attr=None, fit_to_section=None):
    if not _bone_exists(arm, bone_name):
        if _bone_exists(arm, fallback_name):
            print(f"  '{obj.name}': bone '{bone_name}' missing, falling back to '{fallback_name}'")
            bone_name = fallback_name
        else:
            print(f"  '{obj.name}': SKIP -- neither '{bone_name}' nor fallback '{fallback_name}' exist on armature")
            return None

    # Optional auto-position: drop the accessory into the target section's
    # center. Used by teeth (Lib_Teeth_* copies arrive at the donor's mouth
    # position, not the current character's). Runs BEFORE parent reset so
    # location math is in the existing parent's space.
    if fit_to_section and section_target:
        tgt = bpy.data.objects.get(section_target)
        if tgt is not None and section_attr:
            center = _section_center_world(tgt, section_attr, fit_to_section)
            if center is not None:
                _translate_to_world_point(obj, center)
                print(f"    fit_to_section: translated to '{fit_to_section}' "
                      f"center on '{section_target}'")
            else:
                print(f"    fit_to_section: section '{fit_to_section}' has no "
                      f"faces on '{section_target}' -- skip translate")
        else:
            print(f"    fit_to_section: target '{section_target}' missing -- "
                  f"skip translate")

    # Clear all existing weights, then assign 100% to the chosen bone
    while obj.vertex_groups:
        obj.vertex_groups.remove(obj.vertex_groups[0])
    vg = obj.vertex_groups.new(name=bone_name)
    vg.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')

    if clear_shape_keys and obj.data.shape_keys:
        obj.shape_key_clear()

    # Replace any existing armature modifier with a fresh one bound to arm
    for m in [m for m in obj.modifiers if m.type == 'ARMATURE']:
        obj.modifiers.remove(m)
    mod = obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    mod.use_vertex_groups = True

    # Re-parent to armature WITHOUT moving the object. matrix_parent_inverse
    # must be derived from the current world transform so the visible world
    # position is preserved -- naively setting it to arm.matrix_world.inverted()
    # snaps the object back to its local (pre-transform) coordinates if the
    # old parent had its own scale/rotate/translate (e.g. BaseHeadTripo).
    old_world = obj.matrix_world.copy()
    obj.parent = arm
    # World transform = arm.matrix_world @ matrix_parent_inverse @ matrix_basis
    #                 = old_world
    # Solve for matrix_parent_inverse:
    obj.matrix_parent_inverse = arm.matrix_world.inverted() @ old_world @ obj.matrix_basis.inverted()

    print(f"  '{obj.name}' -> '{bone_name}' ({len(obj.data.vertices)}v)")
    return obj


# --------------------------------- ENTRY ------------------------------------
def fit_accessories(cfg):
    arm = bpy.data.objects.get(cfg["armature"])
    if arm is None or arm.type != 'ARMATURE':
        raise RuntimeError(f"armature '{cfg['armature']}' not found")

    print(f"=== fit_accessories (armature: {arm.name}) ===")
    fallback = cfg.get("fallback_bone", "head")
    skip_missing = cfg.get("skip_if_missing", True)
    skip_hidden = cfg.get("skip_if_hidden", True)

    fitted = []
    for entry in cfg["accessories"]:
        name = entry["name"]
        obj = bpy.data.objects.get(name)
        if obj is None:
            if skip_missing:
                print(f"  skip '{name}': missing from scene")
                continue
        elif skip_hidden and obj.hide_get():
            print(f"  skip '{name}': hidden in viewport")
            continue
        if obj is None or obj.type != 'MESH':
            print(f"  skip '{name}': not a mesh")
            continue
        fb = entry.get("fallback_bone", fallback)
        out = _fit_one(obj, entry["bone"], fb, arm,
                       clear_shape_keys=cfg.get("clear_shape_keys", False),
                       section_target=cfg.get("section_target"),
                       section_attr=cfg.get("section_attr"),
                       fit_to_section=entry.get("fit_to_section"))
        if out:
            fitted.append(out)

    print(f"\n[done] fit {len(fitted)} accessories")
    return fitted


if __name__ == "__main__":
    fit_accessories(CONFIG)
