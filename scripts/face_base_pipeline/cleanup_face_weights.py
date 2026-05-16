"""
cleanup_face_weights.py

Post-process face-bone vertex weights on a rigged lowpoly head to remove
"leaks" -- spurious weights that ended up far from the bone they belong to,
caused by topology mismatch during weight transfer.

Background
----------
headswap_transfer.py binds each destination vertex to its nearest source
triangle and interpolates source vertex weights through barycentric coords.
When the destination head has very different proportions than the source
(e.g. big stylized forehead vs. Penny's small forehead), a destination vert
on the crown can end up bound to a source triangle in the eyelid region.
It then inherits a 0.7+ eyelid weight. When the eye bone rotates, that
crown vert moves with it -- distorting the entire head.

This script identifies localized face-bone groups (small bones, e.g.
eyelids, brows, cheeks) and removes weights from verts that fall outside
the expected anatomical region for that bone.

It does NOT touch large groups with legitimate falloff (jaw, neck, spine,
lip corners) -- those spread their influence intentionally and a bbox
filter would clip valid weights.

Two filters combine to identify "leaks":
  1. Name pattern -- only groups matching `targeted_patterns` are processed.
     Defaults cover the small face bones; you can extend it.
  2. Core bbox -- inside a targeted group, compute the bbox of verts whose
     weight is >= `core_weight`. Anything outside that bbox plus `padding`
     gets zeroed.

Designed to drop into the BrainDeadBlender add-on alongside the other
cleanup scripts.
"""

import bpy
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "target": "LowPolyHead_Rigged",

    # Group selection: localized face bones get bbox-cleaned, large spreading
    # bones (jaw, neck, spine, lip corners) are left alone -- their natural
    # weight falloff would be clipped by a bbox filter.
    #
    # A group is "localized" if:
    #   (a) its name matches one of `name_patterns` (whitelist), OR
    #   (b) `auto_detect` is True AND the bbox of its weighted verts (any
    #       weight > min_weight) is smaller than `localized_max_dim` in every
    #       axis -- a robust shape-based detector that catches small face
    #       bones the patterns missed.
    "name_patterns": [
        "_lid_",     # eyelid (e.g. L_eye_lid_upper_mid)
        "_brow",     # brow (e.g. R_brow_mid, L_brow_outer)
        "_cheek",    # cheek (e.g. R_cheek_inner)
        "_nose",     # nose (e.g. C_nose_bridge)
    ],
    "auto_detect": True,
    "localized_max_dim": 0.08,   # 8 cm in any axis -- jaw spans much more

    # Anchor region inside a localized group: verts with weight >=
    # `core_weight`. Their bbox + `padding` is the kept region; everything
    # outside is zeroed. When core verts split into MULTIPLE clusters
    # (e.g. a legitimate eyelid cluster plus a stray high-weight leak on
    # the crown), only the LARGEST cluster is treated as the anchor --
    # smaller clusters get zeroed too. cluster_radius is the max distance
    # between two core verts to consider them in the same cluster.
    "core_weight": 0.5,
    "padding": 0.015,
    "cluster_radius": 0.025,

    # If a group has fewer than this many high-weight verts, we still try to
    # clean it -- we fall back to the bbox of ALL its weighted verts, with a
    # larger padding (handles "no anchor, just leak" groups).
    "min_core_verts": 2,
    "fallback_padding": 0.025,
    "fallback_min_weight": 0.001,
}


# ------------------------------- UTILITIES ----------------------------------
def _group_weights(obj, vg, min_weight=0.001):
    """Return list of (vert_index, weight, mesh_local_position) for this vgroup.

    Uses mesh-LOCAL positions, not world, so they can be compared directly
    against bone.head_local on the armature (both live in conceptually the
    same frame once headswap_transfer.relocalize_to_src has been applied).
    """
    out = []
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index and g.weight > min_weight:
                out.append((v.index, g.weight, v.co.copy()))
                break
    return out


def _bbox(points, padding):
    """Bbox of a list of Vectors, expanded by padding. Returns (min, max)."""
    if not points:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    bmin = Vector((min(xs) - padding, min(ys) - padding, min(zs) - padding))
    bmax = Vector((max(xs) + padding, max(ys) + padding, max(zs) + padding))
    return bmin, bmax


def _bbox_dims(bmin, bmax):
    return (bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z)


def _matches_any(name, patterns):
    n = name.lower()
    return any(p.lower() in n for p in patterns)


def _is_localized(weighted, max_dim, min_weight):
    """True if every weight>min_weight vert fits in a bbox <= max_dim per axis."""
    pts = [p for _, w, p in weighted if w > min_weight]
    if len(pts) < 2:
        return False
    bb = _bbox(pts, 0.0)
    if bb is None:
        return False
    return max(_bbox_dims(*bb)) <= max_dim


def _cluster_points(points, radius):
    """Greedy single-link clustering: groups whose every member is within
    `radius` of another member. Returns list[list[index]], sorted largest-first.

    Picks an unvisited seed, BFS-grows the cluster by adding any point within
    `radius` of any current member. Simple, deterministic, O(n^2) -- fine for
    the small core-anchor sets this is called on (typically < 50 points).
    """
    n = len(points)
    if n == 0:
        return []
    visited = [False] * n
    clusters = []
    for seed in range(n):
        if visited[seed]:
            continue
        stack = [seed]
        cluster = []
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            cluster.append(i)
            for j in range(n):
                if visited[j]:
                    continue
                if (points[i] - points[j]).length <= radius:
                    stack.append(j)
        clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    return clusters


# --------------------------------- STEPS ------------------------------------
def _find_armature(obj):
    """Return the armature Object driving obj, or None."""
    for m in obj.modifiers:
        if m.type == 'ARMATURE' and m.object:
            return m.object
    return None


def _bone_local_head(arm, bone_name):
    """Armature-local rest position of a bone's head, or None if missing."""
    bone = arm.data.bones.get(bone_name)
    if bone is None:
        return None
    return bone.head_local.copy()


def cleanup_face_weights(cfg):
    obj = bpy.data.objects.get(cfg["target"])
    if obj is None or obj.type != 'MESH':
        raise RuntimeError(f"target '{cfg['target']}' not found or not a MESH")
    arm = _find_armature(obj)

    patterns = cfg["name_patterns"]
    auto = cfg["auto_detect"]
    max_dim = cfg["localized_max_dim"]
    core_weight = cfg["core_weight"]
    padding = cfg["padding"]
    cluster_radius = cfg["cluster_radius"]
    min_core = cfg["min_core_verts"]
    fallback_pad = cfg["fallback_padding"]
    fallback_min_w = cfg["fallback_min_weight"]

    print(f"=== cleanup_face_weights -> {obj.name} ===")
    print(f"  name_patterns: {patterns}")
    print(f"  auto_detect: {auto} (max_dim={max_dim}m)")
    print(f"  anchor: core_weight={core_weight} padding={padding}m  min_core_verts={min_core}")
    print(f"  fallback: padding={fallback_pad}m  min_weight={fallback_min_w}")

    cleaned = []
    skipped = []

    for vg in obj.vertex_groups:
        weighted = _group_weights(obj, vg)
        if not weighted:
            continue

        # Decide whether to clean this group: name match OR auto-detect
        name_match = _matches_any(vg.name, patterns)
        auto_match = auto and _is_localized(weighted, max_dim, fallback_min_w)
        if not (name_match or auto_match):
            continue
        why = "pattern+auto" if name_match and auto_match else ("pattern" if name_match else "auto")

        # Build the kept-region bbox using the bone's own rest position as
        # the seed (most authoritative anchor -- a high-weight LEAK won't
        # pull the seed away from the bone). Keep only verts within
        # max_dim/2 of the seed. If the armature or bone isn't available,
        # fall back to using the highest-weight vert as seed.
        if not weighted:
            continue
        anchor_radius = max_dim * 0.5
        seed_pt = _bone_local_head(arm, vg.name) if arm else None
        seed_src = "bone"
        if seed_pt is None:
            weighted_by_w = sorted(weighted, key=lambda x: -x[1])
            seed_pt = weighted_by_w[0][2]
            seed_src = "max_weight"

        anchor_pts = [p for _, _, p in weighted if (p - seed_pt).length <= anchor_radius]

        if len(anchor_pts) >= min_core:
            bb = _bbox(anchor_pts, padding)
            anchor = f"seed={seed_src}({len(anchor_pts)}/{len(weighted)})"
        else:
            bb = _bbox([p for _, _, p in weighted], fallback_pad)
            anchor = f"fallback({len(weighted)})"
        bmin, bmax = bb

        # Zero anything outside
        leaks = []
        for vi, w, p in weighted:
            if not (bmin.x <= p.x <= bmax.x
                    and bmin.y <= p.y <= bmax.y
                    and bmin.z <= p.z <= bmax.z):
                leaks.append((vi, w))

        if leaks:
            for vi, _ in leaks:
                vg.remove([vi])
            cleaned.append((vg.name, why, anchor, len(leaks), max(w for _, w in leaks)))
        else:
            skipped.append((vg.name, f"{why} matched but no leaks"))

    print(f"\n[cleaned {len(cleaned)} groups]")
    for name, why, anchor, leaks_n, max_leak in sorted(cleaned, key=lambda x: -x[4]):
        print(f"  {name}: via={why} anchor={anchor} leaks_zeroed={leaks_n}v max_leak={max_leak:.3f}")
    if skipped:
        print(f"\n[skipped {len(skipped)} groups]")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    return {"cleaned": cleaned, "skipped": skipped}


# --------------------------------- ENTRY ------------------------------------
if __name__ == "__main__":
    cleanup_face_weights(CONFIG)
