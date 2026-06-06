"""
polish_face_weights.py

CONFIG-driven post-merge weight polish. Smooths specific vgroups across
their neighbourhood so neck/head/jaw/lip transitions blend cleanly
instead of producing the sharp boundaries that cause visible chin and
neck warping during body bone rotations (frame 685-697 body_neck_turn,
681-702 spine_back, etc.).

Why CONFIG and not hardcoded
----------------------------
Each character may need different smoothing -- a head with sharper neck
geometry needs more iterations than a soft Tripo blob. Hardcoding 8
iterations in the merge script forces every head to use the same value
and makes tuning a code edit instead of a 1-line config change.

What it does
------------
For each entry in `cfg["smooth_groups"]`, runs the equivalent of
Blender's vertex_group_smooth operator (factor + iterations) on the
named vgroup, then runs vertex_group_normalize_all so weights still
sum to 1.0. Optional `lip_jaw_rebind` re-runs the section-aware lip
upper/lower split via IDW.

Run AFTER merge_face_meshes and BEFORE rig test rendering.
"""

import bpy
from mathutils import Vector


CONFIG = {
    "target": "LowPolyHead_Rigged",

    # Per-vgroup smoothing pass. Tuned for current Tripo head; widen
    # iterations for sharper heads, reduce for softer ones.
    "smooth_groups": [
        # Smoothing C_jaw bleeds jaw influence into chin/lower-cheek so
        # they ride the jaw partially instead of staying rigid on head.
        {"name": "C_jaw",   "factor": 0.5, "iterations": 10, "expand": 0.3},
        # Smoothing head + neck_01 eliminates the 10x deformation spread
        # between chin (head/C_jaw) and upper neck (neck_01) at body
        # poses (body_neck_turn especially).
        {"name": "head",    "factor": 0.8, "iterations": 15, "expand": 0.5},
        {"name": "neck_01", "factor": 0.7, "iterations": 15, "expand": 0.0},
        {"name": "neck_02", "factor": 0.6, "iterations": 8,  "expand": 0.0},
    ],

    # If True, run an IDW lip upper/lower jaw-split rebind on the lips
    # section (verts with _section == "lips"). Use this if lips section
    # weights got destroyed and you need them rebuilt cleanly.
    "lip_jaw_rebind": False,
    "lip_jaw_floor": 0.85,   # lower-lip C_jaw weight
    "lip_jaw_upper": 0.0,    # upper-lip C_jaw weight (0 = anchored)
    "lip_split_band": 0.004, # m, smoothstep band around midZ

    # After smoothing + normalize, any verts whose total weight to
    # NON-section_* vgroups is below this get rebound to fallback bone.
    "orphan_fallback_bone": "head",
    "orphan_threshold": 0.5,
}


def _smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a))) if b != a else 0.0
    return t * t * (3 - 2 * t)


def _lip_jaw_rebind(obj, cfg):
    """Section-aware IDW rebind for lips section (upper anchored to
    head, lower 85% to C_jaw via smoothstep on Z midline)."""
    me = obj.data
    a = me.attributes.get("_section")
    if a is None:
        print("  lip_jaw_rebind: no _section attribute -- skip")
        return 0
    lip_verts = set()
    for fi, p in enumerate(me.polygons):
        if a.data[fi].value.decode('utf-8') == 'lips':
            lip_verts.update(p.vertices)
    if not lip_verts:
        print("  lip_jaw_rebind: no lip-section verts found")
        return 0

    arm = None
    for m in obj.modifiers:
        if m.type == 'ARMATURE' and m.object:
            arm = m.object; break
    if arm is None:
        print("  lip_jaw_rebind: no armature -- skip")
        return 0

    lip_bones = ["C_lip_upper_mid", "L_lip_upper_outer", "R_lip_upper_outer",
                 "C_lip_lower_mid", "L_lip_lower_outer", "R_lip_lower_outer",
                 "L_lip_corner",    "R_lip_corner"]
    bone_pts = {}
    for bn in lip_bones:
        b = arm.data.bones.get(bn)
        if b: bone_pts[bn] = arm.matrix_world @ b.head_local
    if not bone_pts:
        print("  lip_jaw_rebind: no lip bones in armature -- skip")
        return 0

    hw = obj.matrix_world
    ws = {vi: hw @ me.vertices[vi].co for vi in lip_verts}
    zs = [p.z for p in ws.values()]
    z_mid = (min(zs) + max(zs)) * 0.5
    band = cfg["lip_split_band"]
    z_lo = z_mid - band * 0.5
    z_hi = z_mid + band * 0.5

    # Ensure all lip vgroups exist on target
    for bn in list(bone_pts) + ["C_jaw"]:
        if bn not in obj.vertex_groups:
            obj.vertex_groups.new(name=bn)
    vg = {bn: obj.vertex_groups[bn] for bn in list(bone_pts) + ["C_jaw"]}

    # Clear lip-related weights on lip verts
    related = set(list(bone_pts) + ["C_jaw"])
    for vi in lip_verts:
        for bn in related:
            vg[bn].remove([vi])

    K = 3
    POWER = 2.0
    EPS = 1e-4
    jaw_floor = cfg["lip_jaw_floor"]
    jaw_upper = cfg["lip_jaw_upper"]
    for vi in lip_verts:
        p = ws[vi]
        dists = sorted(((bn, (p - pt).length) for bn, pt in bone_pts.items()),
                       key=lambda x: x[1])[:K]
        weights = {bn: 1.0 / max(d, 1e-6) ** POWER for bn, d in dists}
        wsum = sum(weights.values())
        for bn in weights: weights[bn] /= wsum
        t = 1.0 - _smoothstep(z_lo, z_hi, p.z)
        jaw_w = jaw_floor * t + jaw_upper * (1.0 - t)
        scale = 1.0 - jaw_w
        for bn in weights: weights[bn] *= scale
        if jaw_w > EPS:
            vg["C_jaw"].add([vi], jaw_w, 'REPLACE')
        for bn, w in weights.items():
            if w > EPS:
                vg[bn].add([vi], w, 'REPLACE')
    print(f"  lip_jaw_rebind: rebound {len(lip_verts)} lip verts (jaw_floor={jaw_floor})")
    return len(lip_verts)


def polish_face_weights(cfg):
    obj = bpy.data.objects.get(cfg["target"])
    if obj is None or obj.type != 'MESH':
        raise RuntimeError(f"target '{cfg['target']}' not a mesh")

    print(f"=== polish_face_weights -> {obj.name} ===")

    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    smoothed = 0
    for entry in cfg["smooth_groups"]:
        name = entry["name"]
        vg = obj.vertex_groups.get(name)
        if vg is None:
            print(f"  skip '{name}': not on target")
            continue
        obj.vertex_groups.active_index = vg.index
        bpy.ops.object.vertex_group_smooth(
            group_select_mode='ACTIVE',
            factor=entry.get("factor", 0.5),
            repeat=entry.get("iterations", 5),
            expand=entry.get("expand", 0.0))
        print(f"  smoothed '{name}': factor={entry.get('factor',0.5)} "
              f"iter={entry.get('iterations',5)} expand={entry.get('expand',0)}")
        smoothed += 1

    bpy.ops.object.vertex_group_normalize_all(group_select_mode='ALL', lock_active=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    if cfg.get("lip_jaw_rebind"):
        _lip_jaw_rebind(obj, cfg)

    # Fix orphans (any vert with no real-bone weight after smoothing)
    arm_bones = set()
    for m in obj.modifiers:
        if m.type == 'ARMATURE' and m.object:
            arm_bones = {b.name for b in m.object.data.bones}
            break
    fb_name = cfg.get("orphan_fallback_bone", "head")
    fb_vg = obj.vertex_groups.get(fb_name)
    if fb_vg is None:
        fb_vg = obj.vertex_groups.new(name=fb_name)
    threshold = cfg.get("orphan_threshold", 0.5)
    orphans = []
    for vi, v in enumerate(obj.data.vertices):
        real_w = sum(g.weight for g in v.groups
                     if obj.vertex_groups[g.group].name in arm_bones)
        if real_w < threshold:
            orphans.append(vi)
    if orphans:
        for vi in orphans:
            fb_vg.add([vi], 1.0, 'REPLACE')
        print(f"  orphans fixed: {len(orphans)} verts re-bound to '{fb_name}'")

    print(f"[done] {smoothed} groups smoothed, orphans fixed")
    return smoothed


if __name__ == "__main__":
    polish_face_weights(CONFIG)
