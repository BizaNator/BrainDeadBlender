"""
rebind_lip_weights.py

Replace CustomLips' weights with a clean upper/lower split, so the upper
lip stays put when C_jaw rotates open and the lower lip rides the jaw.

Why
---
`fit_custom_lips` binds via BVH closest-point to Penny's body mesh, but in
the lip region almost every nearby Penny vert is dominated by C_jaw -- so
the entire CustomLips ends up at ~100% C_jaw and the lips move as one
rigid block with the jaw.

What it does
------------
    1. Clear all weights on CustomLips (we are rebuilding from scratch).
    2. For each lip vert, compute distance to the lip-control bone heads
       (upper-mid / upper-outer L/R / lower-mid / lower-outer L/R /
       corner L/R) in world space.
    3. Inverse-distance-weight the K nearest bones; normalize so the
       lip-control influence sums to (1 - jaw_floor).
    4. Add a C_jaw contribution that is `jaw_floor` for verts in the
       lower half and 0 (or small) for verts in the upper half. This
       makes the lower lip ride C_jaw while the upper lip stays anchored
       to the head.

The split between upper / lower is a soft Z transition (smoothstep over
`split_band`), not a hard cut, so the lip corners blend cleanly.

Drop into the BrainDeadBlender add-on; runs after `fit_custom_lips`.
"""

import bpy
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "lips":     "CustomLips",
    "armature": "Fortnite_Armature",

    # Lip-control bones (head positions used as influence points).
    # Each entry: bone_name -> "upper" / "lower" / "corner" tag.
    "lip_bones": {
        "C_lip_upper_mid":   "upper",
        "L_lip_upper_outer": "upper",
        "R_lip_upper_outer": "upper",
        "C_lip_lower_mid":   "lower",
        "L_lip_lower_outer": "lower",
        "R_lip_lower_outer": "lower",
        "L_lip_corner":      "corner",
        "R_lip_corner":      "corner",
    },

    # K nearest bones to weight per vert.
    "k_nearest": 3,

    # Inverse-distance power; higher = more localized weights.
    "idw_power": 2.0,

    # Jaw blending. `jaw_floor` is the C_jaw weight added to lower-lip
    # verts (so they ride the jaw when it opens). Upper-lip verts get 0.
    # split_band controls how soft the upper/lower transition is, in
    # world meters around the lips' mid-Z.
    "jaw_floor":  0.85,    # lower lip is 85% jaw, 15% lip controls
    "jaw_upper":  0.0,     # upper lip 0% jaw -- stays anchored to head
    "split_band": 0.004,   # 4mm smoothstep band around midZ

    # Min weight kept (prune anything below).
    "epsilon": 1e-4,
}


# ------------------------------- HELPERS ------------------------------------
def _smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a))) if b != a else 0.0
    return t * t * (3 - 2 * t)


def rebind_lip_weights(cfg):
    lips = bpy.data.objects.get(cfg["lips"])
    arm = bpy.data.objects.get(cfg["armature"])
    if lips is None or arm is None:
        raise RuntimeError("lips or armature not found")

    print(f"=== rebind_lip_weights -> {lips.name} ===")

    # Resolve bone head positions in WORLD space
    bone_pts = {}
    bone_tag = {}
    for bname, tag in cfg["lip_bones"].items():
        b = arm.data.bones.get(bname)
        if b is None:
            print(f"  skip bone '{bname}': not in armature")
            continue
        bone_pts[bname] = arm.matrix_world @ b.head_local
        bone_tag[bname] = tag
    if not bone_pts:
        raise RuntimeError("no lip bones resolved")

    print(f"  resolved {len(bone_pts)} lip bones")

    # Compute lip Z range for upper/lower split
    mw = lips.matrix_world
    ws = [mw @ v.co for v in lips.data.vertices]
    zs = [p.z for p in ws]
    z_mid = (min(zs) + max(zs)) * 0.5
    z_lo = z_mid - cfg["split_band"] * 0.5
    z_hi = z_mid + cfg["split_band"] * 0.5
    print(f"  lip Z range: [{min(zs):.4f}, {max(zs):.4f}]  midZ={z_mid:.4f}  band=[{z_lo:.4f},{z_hi:.4f}]")

    # Clear all existing vgroups + recreate fresh ones for lip bones + C_jaw
    while lips.vertex_groups:
        lips.vertex_groups.remove(lips.vertex_groups[0])
    vg = {n: lips.vertex_groups.new(name=n) for n in bone_pts}
    vg_jaw = lips.vertex_groups.new(name="C_jaw")

    k = cfg["k_nearest"]
    power = cfg["idw_power"]
    eps = cfg["epsilon"]

    upper_count = lower_count = 0
    for vi, p in enumerate(ws):
        # IDW over K nearest lip bones
        dists = sorted(
            ((bn, (p - pt).length) for bn, pt in bone_pts.items()),
            key=lambda x: x[1])[:k]
        weights = {}
        wsum = 0.0
        for bn, d in dists:
            w = 1.0 / max(d, 1e-6) ** power
            weights[bn] = w
            wsum += w
        # Normalize
        for bn in weights:
            weights[bn] /= wsum

        # Jaw weight: smoothstep based on Z (lower = more jaw)
        # t=1 means fully lower, t=0 means fully upper
        t = 1.0 - _smoothstep(z_lo, z_hi, p.z)
        jaw_w = cfg["jaw_floor"] * t + cfg["jaw_upper"] * (1.0 - t)
        # Lip control weights get scaled by (1 - jaw_w)
        scale = 1.0 - jaw_w
        for bn in weights:
            weights[bn] *= scale

        if jaw_w > eps:
            vg_jaw.add([vi], jaw_w, 'REPLACE')
        for bn, w in weights.items():
            if w > eps:
                vg[bn].add([vi], w, 'REPLACE')

        if t > 0.5:
            lower_count += 1
        else:
            upper_count += 1

    print(f"  upper verts: {upper_count}, lower verts: {lower_count}")
    # Verify
    totals = {}
    for v in lips.data.vertices:
        for g in v.groups:
            n = lips.vertex_groups[g.group].name
            totals[n] = totals.get(n, 0.0) + g.weight
    print("\n  Final total weight per vgroup:")
    for n, w in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"    {n:30s}: {w:.2f}")

    return lips


if __name__ == "__main__":
    rebind_lip_weights(CONFIG)
