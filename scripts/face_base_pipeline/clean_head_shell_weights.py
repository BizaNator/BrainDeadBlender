"""
clean_head_shell_weights.py

After headswap_transfer binds Penny's weights to the lowpoly head, the
resulting head shell ends up with weights on bones that shouldn't drive
it -- spine, clavicle, lip, jaw -- because BVH closest-point grabs
Penny's neck/lip verts for the head's lower areas.

This step strips those vgroups from the head shell and re-normalizes so
the remaining weights (head, neck, cheek, brow, nose) sum to 1.0 on
every vert.

Why
---
- Spine / clavicle weights cause the head to twist when the head bone
  rotates (parts of the head stay anchored to spine).
- Lip / jaw weights on the head shell mismatch after
  retarget_bones_to_parts moves those bones to fit CustomLips -- the
  old positions baked into the bind no longer match.

Both classes of bone shouldn't be on the head shell anyway -- the head
shell only needs `head`, `neck_01/02`, and the soft-expression bones
(cheeks, brow, nose).

Runs after retarget_bones_to_parts, before rig_test_animation.
"""

import bpy


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "head": "LowPolyHead_Rigged",

    # Bones with these prefixes are stripped from the head shell entirely.
    "strip_prefixes": (
        "spine_", "clavicle_", "pelvis", "thigh_", "calf_", "foot_",
        "upperarm_", "lowerarm_", "hand_", "FX_", "dyn_hair", "earpiece",
        "hat", "Charm_", "Attach", "attach", "root",
    ),

    # Exact bone names to strip (face bones that are owned by submeshes
    # like CustomLips / Tongue / Teeth, not the head shell).
    "strip_names": (
        "C_jaw", "tongue", "teeth_upper", "teeth_lower",
        "C_lip_upper_mid", "C_lip_lower_mid",
        "L_lip_corner", "R_lip_corner",
        "L_lip_upper_outer", "R_lip_upper_outer",
        "L_lip_lower_outer", "R_lip_lower_outer",
    ),

    # Bone the head verts get reassigned to if all weights got stripped.
    "fallback_bone": "head",
}


def clean_head_shell_weights(cfg):
    head = bpy.data.objects.get(cfg["head"])
    if head is None:
        print(f"  skip: '{cfg['head']}' not in scene")
        return None

    print(f"=== clean_head_shell_weights -> {head.name} ===")

    prefixes = tuple(cfg["strip_prefixes"])
    names = set(cfg["strip_names"])

    # Find vgroups to strip
    to_strip = [vg for vg in head.vertex_groups
                if vg.name.startswith(prefixes) or vg.name in names]
    print(f"  stripping {len(to_strip)} vgroups: {[vg.name for vg in to_strip]}")
    for vg in to_strip:
        head.vertex_groups.remove(vg)

    # Detect verts that ended up with near-zero total weight and push them to fallback_bone
    totals = [0.0] * len(head.data.vertices)
    for v in head.data.vertices:
        for g in v.groups:
            totals[v.index] += g.weight
    orphans = [i for i, w in enumerate(totals) if w < 0.01]
    if orphans:
        fb_name = cfg["fallback_bone"]
        fb_vg = head.vertex_groups.get(fb_name)
        if fb_vg is None:
            fb_vg = head.vertex_groups.new(name=fb_name)
        fb_vg.add(orphans, 1.0, 'REPLACE')
        print(f"  reassigned {len(orphans)} orphan verts -> '{fb_name}' weight 1.0")

    # Renormalize all groups so per-vert weight sum = 1.0
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    bpy.ops.object.vertex_group_normalize_all(group_select_mode='ALL', lock_active=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Report final top weights
    totals = {}
    for v in head.data.vertices:
        for g in v.groups:
            n = head.vertex_groups[g.group].name
            totals[n] = totals.get(n, 0.0) + g.weight
    print(f"\n  remaining {len(head.vertex_groups)} vgroups, top weights:")
    for n, w in sorted(totals.items(), key=lambda x: -x[1])[:8]:
        print(f"    {n:30s}: {w:.2f}")
    return head


if __name__ == "__main__":
    clean_head_shell_weights(CONFIG)
