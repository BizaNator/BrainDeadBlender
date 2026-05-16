"""
transfer_shape_keys.py

Layer a donor head's shape keys onto a target (rigged) head as new
delta-based morphs, so the target gets the same animatable expressions
(ARKit blendshapes, custom Mutable morphs, etc.) without changing its
base geometry or weights.

How it works
------------
For each target vertex `t`:
  1. Find the closest point on the donor's BASIS mesh via BVH.
  2. Identify the donor triangle and compute barycentric coords (u,v,w).
  3. For each donor shape key:
       donor_delta = key_block.data[idx].co - basis.data[idx].co
       target_delta = u*donor_delta[a] + v*donor_delta[b] + w*donor_delta[c]
       target_key.data[t].co = target.basis.data[t].co + target_delta

Because the delta is what's interpolated (not absolute position), the
shape key morphs the target's own basis -- so the morph deforms the
target shape, not snaps it to donor shape.

Designed to drop into the BrainDeadBlender add-on after the rigged head
exists. Idempotent: re-running with a new donor APPENDS its shape keys
to the target without disturbing existing ones (unless a name collision,
in which case the existing key gets overwritten).
"""

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "target": "LowPolyHead_Rigged",
    "donor":  "ARKit_Head",   # default: ARKit donor (MechanicGirl)

    # Which donor keys to transfer. Empty list = ALL keys except Basis.
    # Otherwise, only the listed names (matched exactly).
    "key_filter": [],

    # Substring blacklist (e.g. ["jaw"] skips jaw* shape keys).
    "key_blacklist_substr": [],

    # Skip transfer if a target vert is further than this from donor surface
    # (in world meters). 0 = no max distance. For Tripo's stylized head, the
    # closest Penny-anatomy point can be a few cm away.
    "max_bind_distance": 0.05,

    # Drop a per-vert delta this small (sub-millimeter noise from
    # interpolation). In world meters.
    "delta_epsilon": 1e-5,

    # Overwrite existing key with same name on target (else skip).
    "overwrite": True,

    # If True, temporarily align (scale + translate) the donor BVH to the
    # target's bbox before binding -- handles donors at a different world
    # location / scale from the target. Rotation is NOT applied (assumes
    # both heads face the same axis). Donor is unchanged in the scene.
    "donor_align_to_target_bbox": True,
}


# ------------------------------- HELPERS ------------------------------------
def _barycentric(p, a, b, c):
    v0 = b - a; v1 = c - a; v2 = p - a
    d00 = v0.dot(v0); d01 = v0.dot(v1); d11 = v1.dot(v1)
    d20 = v2.dot(v0); d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    u, v, w = max(u, 0.0), max(v, 0.0), max(w, 0.0)
    s = u + v + w
    if s <= 0.0:
        return (1.0, 0.0, 0.0)
    return (u / s, v / s, w / s)


def _basis_world_coords(obj):
    """World-space basis positions for an object, using its Basis shape key if present."""
    me = obj.data
    mw = obj.matrix_world
    sk = me.shape_keys
    if sk and sk.key_blocks:
        basis_kb = sk.key_blocks[0]
        if basis_kb.name.lower() == "basis":
            return [mw @ basis_kb.data[i].co for i in range(len(me.vertices))]
    return [mw @ v.co for v in me.vertices]


def _bbox(coords):
    xs=[c.x for c in coords]; ys=[c.y for c in coords]; zs=[c.z for c in coords]
    return (Vector((min(xs),min(ys),min(zs))), Vector((max(xs),max(ys),max(zs))))


def _align_bbox(coords, donor_min, donor_max, target_min, target_max):
    """Scale + translate `coords` so donor bbox maps onto target bbox.
    Uses uniform scale = mean of per-axis scale ratios (preserves donor shape)."""
    donor_size = donor_max - donor_min
    target_size = target_max - target_min
    sx = target_size.x / donor_size.x if donor_size.x > 1e-6 else 1.0
    sy = target_size.y / donor_size.y if donor_size.y > 1e-6 else 1.0
    sz = target_size.z / donor_size.z if donor_size.z > 1e-6 else 1.0
    s = (sx + sy + sz) / 3.0
    donor_center = (donor_min + donor_max) * 0.5
    target_center = (target_min + target_max) * 0.5
    out = []
    for c in coords:
        rel = (c - donor_center) * s
        out.append(target_center + rel)
    return out, s


def _build_donor_bvh(donor, align_to_target_obj=None):
    """Build BVH of donor BASIS positions in world space.
    If `align_to_target_obj` is given, scale+translate donor coords to fit
    that target's bbox before building the BVH.
    Returns: bvh, tri_table[(vidx0, vidx1, vidx2)], basis_world_coords[vidx]."""
    basis_world = _basis_world_coords(donor)

    if align_to_target_obj is not None:
        target_world = _basis_world_coords(align_to_target_obj)
        d_min, d_max = _bbox(basis_world)
        t_min, t_max = _bbox(target_world)
        basis_world, scale = _align_bbox(basis_world, d_min, d_max, t_min, t_max)
        print(f"  aligned donor bbox -> target bbox  (uniform scale {scale:.4f})")
        print(f"    donor world: center=({(d_min.x+d_max.x)/2:.3f},{(d_min.y+d_max.y)/2:.3f},{(d_min.z+d_max.z)/2:.3f})  "
              f"size=({(d_max.x-d_min.x)*100:.1f},{(d_max.y-d_min.y)*100:.1f},{(d_max.z-d_min.z)*100:.1f})cm")
        print(f"    target world: center=({(t_min.x+t_max.x)/2:.3f},{(t_min.y+t_max.y)/2:.3f},{(t_min.z+t_max.z)/2:.3f})  "
              f"size=({(t_max.x-t_min.x)*100:.1f},{(t_max.y-t_min.y)*100:.1f},{(t_max.z-t_min.z)*100:.1f})cm")

    tris = []
    tri_table = []
    for p in donor.data.polygons:
        vs = list(p.vertices)
        for i in range(1, len(vs) - 1):
            tris.append([basis_world[vs[0]], basis_world[vs[i]], basis_world[vs[i + 1]]])
            tri_table.append((vs[0], vs[i], vs[i + 1]))
    bvh = BVHTree.FromPolygons(
        [c for tri in tris for c in tri],
        [(i * 3, i * 3 + 1, i * 3 + 2) for i in range(len(tris))],
        all_triangles=True)
    return bvh, tri_table, basis_world


def _ensure_target_basis(target):
    """Make sure target has a Basis shape key; return the shape_keys datablock."""
    if target.data.shape_keys is None:
        target.shape_key_add(name="Basis", from_mix=False)
    sk = target.data.shape_keys
    return sk


# --------------------------------- ENTRY ------------------------------------
def transfer_shape_keys(cfg):
    target = bpy.data.objects.get(cfg["target"])
    donor  = bpy.data.objects.get(cfg["donor"])
    if target is None or target.type != 'MESH':
        raise RuntimeError(f"target '{cfg['target']}' not found / not mesh")
    if donor is None or donor.type != 'MESH':
        raise RuntimeError(f"donor '{cfg['donor']}' not found / not mesh")
    if donor.data.shape_keys is None or len(donor.data.shape_keys.key_blocks) < 2:
        print(f"  skip: donor '{donor.name}' has no shape keys (besides Basis)")
        return 0

    print(f"=== transfer_shape_keys: {donor.name} -> {target.name} ===")

    # Filter donor keys
    donor_sk = donor.data.shape_keys
    name_filter = set(cfg.get("key_filter") or [])
    black_substr = [s.lower() for s in cfg.get("key_blacklist_substr", [])]
    candidates = []
    for kb in donor_sk.key_blocks:
        if kb.name == donor_sk.key_blocks[0].name:  # Basis
            continue
        if name_filter and kb.name not in name_filter:
            continue
        if any(s in kb.name.lower() for s in black_substr):
            continue
        candidates.append(kb)
    print(f"  donor keys to transfer: {len(candidates)} of {len(donor_sk.key_blocks)-1}")
    if not candidates:
        return 0

    # Build BVH on donor basis (optionally aligned to target bbox)
    align_target = target if cfg.get("donor_align_to_target_bbox", True) else None
    bvh, tri_table, basis_world = _build_donor_bvh(donor, align_to_target_obj=align_target)

    # If we aligned donor to target, we also need to scale deltas by the same factor.
    # Compute the alignment scale here for delta scaling.
    align_scale = 1.0
    if align_target is not None:
        donor_world_orig = _basis_world_coords(donor)
        d_min, d_max = _bbox(donor_world_orig)
        target_world = _basis_world_coords(align_target)
        t_min, t_max = _bbox(target_world)
        ds = d_max - d_min; ts = t_max - t_min
        align_scale = ((ts.x/ds.x if ds.x>1e-6 else 1) +
                       (ts.y/ds.y if ds.y>1e-6 else 1) +
                       (ts.z/ds.z if ds.z>1e-6 else 1)) / 3.0

    # Pre-compute donor basis (world) for each vert
    donor_mw = donor.matrix_world

    # Resolve donor basis key block
    donor_basis_kb = donor_sk.key_blocks[0]

    # Pre-compute donor per-key per-vert WORLD delta (basis -> key), scaled
    # by align factor so the morphs read as the same anatomical amplitude
    # on the (potentially smaller/bigger) target head.
    key_world_deltas = {}
    for kb in candidates:
        deltas = []
        for vi in range(len(donor.data.vertices)):
            local_delta = kb.data[vi].co - donor_basis_kb.data[vi].co
            world_delta = (donor_mw.to_3x3() @ local_delta) * align_scale
            deltas.append(world_delta)
        key_world_deltas[kb.name] = deltas

    # Ensure target has Basis
    target_sk = _ensure_target_basis(target)
    target_mw = target.matrix_world
    target_mw_inv = target_mw.inverted()
    target_basis_kb = target_sk.key_blocks[0]
    overwrite = cfg.get("overwrite", True)
    max_dist = cfg.get("max_bind_distance", 0.0)
    eps = cfg.get("delta_epsilon", 1e-5)

    # Pre-compute target basis world + nearest donor binding per target vert
    target_basis_world = [target_mw @ target_basis_kb.data[vi].co
                          for vi in range(len(target.data.vertices))]
    bindings = []  # per target vert: (a, b, c, u, v, w) or None if miss
    misses = 0
    for vi, p in enumerate(target_basis_world):
        hit = bvh.find_nearest(p)
        if hit[0] is None:
            bindings.append(None); misses += 1; continue
        if max_dist > 0 and (p - hit[0]).length > max_dist:
            bindings.append(None); misses += 1; continue
        tri_i = hit[2]
        a, b, c = tri_table[tri_i]
        u, vc, w = _barycentric(hit[0], basis_world[a], basis_world[b], basis_world[c])
        bindings.append((a, b, c, u, vc, w))
    print(f"  bound {len(bindings) - misses}/{len(bindings)} target verts  "
          f"(misses: {misses})")

    # Now create one shape key per donor candidate
    n_created = n_overwritten = 0
    for kb in candidates:
        existing = target_sk.key_blocks.get(kb.name)
        if existing:
            if not overwrite:
                print(f"    skip '{kb.name}': already exists")
                continue
            # Reset existing to basis (so we can overwrite cleanly)
            target.shape_key_remove(existing)
            n_overwritten += 1
        new_kb = target.shape_key_add(name=kb.name, from_mix=False)
        deltas = key_world_deltas[kb.name]
        for vi, binding in enumerate(bindings):
            if binding is None:
                continue
            a, b, c, u, vc, w = binding
            world_delta = u * deltas[a] + vc * deltas[b] + w * deltas[c]
            if world_delta.length < eps:
                continue
            # Convert world delta back to target-local
            local_delta = target_mw_inv.to_3x3() @ world_delta
            new_kb.data[vi].co = target_basis_kb.data[vi].co + local_delta
        n_created += 1

    print(f"  created {n_created} keys ({n_overwritten} overwritten)")
    return n_created


if __name__ == "__main__":
    transfer_shape_keys(CONFIG)
