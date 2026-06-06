"""Region-aware retopo reducer for VertexArcade-style base meshes.

Goal: take a clean anatomical base mesh and produce a "Hard Body, soft Poly"
silhouette suitable for UEFN — preserve overall curves and joint loops while
collapsing cross-section verts down to faceted polygonal cylinders
(hexagonal/octagonal arms and legs, blocky skull).

Strategy:
    1. Global Un-Subdivide (Decimate UNSUBDIV iter=1) — ~50% reduction, keeps
       quad topology.
    2. Per-region longitudinal-loop dissolve. For a "cylindrical" region like
       upper-arm, compute its primary axis via PCA, project each vert onto a
       cross-section angle around that axis, bin angles, then dissolve verts
       in odd bins. This halves verts-per-ring without disturbing the rings
       themselves (so deform loops at the joints stay intact).
    3. Skull cap reduction — verts above a Z threshold get the same
       longitudinal-dissolve treatment using Z as the axis.

The longitudinal-dissolve uses ``bmesh.ops.dissolve_verts``, which collapses
each dissolved vert into its two ring neighbors, halving the cross-section
quad count cleanly.

Usage::

    import bpy
    from braindead_blender.scripts.face_base_pipeline import reduce_va_basemesh
    reduce_va_basemesh.run(
        obj=bpy.data.objects['VA_F_WIP'],
        regions=[
            {"name": "arms", "vgroups": ["DEF-upper_arm.L","DEF-upper_arm.L.001",
                                          "DEF-upper_arm.R","DEF-upper_arm.R.001",
                                          "DEF-forearm.L","DEF-forearm.L.001",
                                          "DEF-forearm.R","DEF-forearm.R.001"],
             "target_verts_per_ring": 8},
            {"name": "legs", "vgroups": ["DEF-thigh.L","DEF-thigh.L.001",
                                          "DEF-thigh.R","DEF-thigh.R.001",
                                          "DEF-shin.L","DEF-shin.L.001",
                                          "DEF-shin.R","DEF-shin.R.001"],
             "target_verts_per_ring": 8},
            {"name": "skull", "z_above": 1.55, "target_verts_per_ring": 8},
        ],
    )
"""

from __future__ import annotations
import math
import bpy
import bmesh
from mathutils import Vector

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _region_vert_indices(obj, vgroup_names=None, z_above=None, weight_thresh=0.05):
    """Return a set of vertex indices belonging to the region."""
    indices = set()
    mw = obj.matrix_world
    if vgroup_names:
        vg_idxs = {obj.vertex_groups[n].index for n in vgroup_names if n in obj.vertex_groups}
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group in vg_idxs and g.weight > weight_thresh:
                    indices.add(v.index)
                    break
    if z_above is not None:
        for v in obj.data.vertices:
            if (mw @ v.co).z >= z_above:
                indices.add(v.index)
    return indices


def _principal_axis(points):
    """Crude PCA primary axis. Returns (centroid, axis_dir_normalized)."""
    if not points:
        return Vector((0, 0, 0)), Vector((0, 0, 1))
    centroid = sum(points, Vector()) / len(points)
    # Variance along each cardinal axis as a tie-breaker for picking dominant
    var_x = sum((p.x - centroid.x) ** 2 for p in points)
    var_y = sum((p.y - centroid.y) ** 2 for p in points)
    var_z = sum((p.z - centroid.z) ** 2 for p in points)
    # The "primary axis" for a long thin region is the axis with largest variance
    primary = max((var_x, 'X'), (var_y, 'Y'), (var_z, 'Z'))[1]
    axis = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}[primary]
    return centroid, axis


def _classify_angular_bin(point, centroid, axis, n_bins):
    """Project point onto plane perpendicular to axis at centroid level, then
    angle around axis -> bin in [0, n_bins).
    """
    rel = point - centroid
    along = rel.dot(axis)
    perp = rel - axis * along
    # Build a stable reference frame perpendicular to axis
    # Pick a non-parallel vector for cross product
    ref = Vector((1, 0, 0)) if abs(axis.x) < 0.9 else Vector((0, 1, 0))
    e1 = (ref - axis * ref.dot(axis)).normalized()
    e2 = axis.cross(e1).normalized()
    u = perp.dot(e1)
    v = perp.dot(e2)
    angle = math.atan2(v, u)  # -pi..pi
    bin_idx = int(((angle + math.pi) / (2 * math.pi)) * n_bins) % n_bins
    return bin_idx


def _dissolve_alternate_angular(obj, region_indices, n_target_per_ring, keep_neighbors_of=None):
    """Dissolve verts in odd angular bins. Operates on bmesh; expects obj in
    OBJECT mode at entry. Restores OBJECT mode on exit.

    n_target_per_ring: number of verts to keep per cross-section ring.
                     dissolves alternate ring verts down to this count.
    """
    if not region_indices:
        return 0

    mw = obj.matrix_world
    region_pts_world = [mw @ obj.data.vertices[i].co for i in region_indices]
    centroid, axis = _principal_axis(region_pts_world)

    # Decide bin count: we want ~half of the current verts-per-ring to survive
    n_bins = n_target_per_ring * 2  # odd bins dissolved -> n_target survive

    # Pre-compute which world-bin each region vert lives in
    drop_indices = set()
    for vi in region_indices:
        p = mw @ obj.data.vertices[vi].co
        b = _classify_angular_bin(p, centroid, axis, n_bins)
        if b % 2 == 1:
            drop_indices.add(vi)

    if keep_neighbors_of:
        drop_indices -= set(keep_neighbors_of)

    if not drop_indices:
        return 0

    # bmesh dissolve
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    drop_bm_verts = [bm.verts[i] for i in drop_indices if i < len(bm.verts)]
    try:
        bmesh.ops.dissolve_verts(bm, verts=drop_bm_verts, use_face_split=False,
                                 use_boundary_tear=False)
    finally:
        bmesh.update_edit_mesh(obj.data, destructive=True)
        bpy.ops.object.mode_set(mode='OBJECT')

    return len(drop_indices)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def strip_shape_keys(obj):
    """Remove all shape keys (incl. Basis) so destructive edits are allowed."""
    if obj.data.shape_keys is None:
        return
    # Need active object + all keys removed
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shape_key_remove(all=True)


def strip_modifiers(obj, types_to_strip=('SUBSURF', 'ARMATURE')):
    for m in list(obj.modifiers):
        if m.type in types_to_strip:
            obj.modifiers.remove(m)


def global_unsub(obj, iterations=1):
    """Apply Decimate UNSUBDIV `iterations` times. Halves vert count per pass."""
    bpy.context.view_layer.objects.active = obj
    d = obj.modifiers.new(f'BD_UnSub', 'DECIMATE')
    d.decimate_type = 'UNSUBDIV'
    d.iterations = iterations
    bpy.ops.object.modifier_apply(modifier=d.name)


def reduce_region(obj, region_spec):
    """Reduce cross-section verts for one region spec.

    region_spec: dict with keys:
        name: str (for logging)
        vgroups: list[str] OR z_above: float (one required)
        target_verts_per_ring: int (default 8)
    """
    name = region_spec.get('name', 'region')
    target = int(region_spec.get('target_verts_per_ring', 8))
    indices = _region_vert_indices(obj,
                                   vgroup_names=region_spec.get('vgroups'),
                                   z_above=region_spec.get('z_above'))
    if not indices:
        print(f"[reduce_region] {name}: no verts found")
        return 0
    dropped = _dissolve_alternate_angular(obj, indices, target)
    print(f"[reduce_region] {name}: dropped {dropped} verts (of {len(indices)} region verts)")
    return dropped


def build_polygonal_cylinder_primitive(name, p0, p1, n_sides=8, n_axial=4,
                                        radius_profile=None, bulge_at=None,
                                        bulge_amount=0.0):
    """Create a clean polygonal cylinder mesh aligned along axis p0->p1.

    Use as a shrinkwrap target. The primitive is parented to nothing, lives
    in world space, and can be edited by hand before shrinkwrapping.

    Args:
        name: object name for the new mesh
        p0, p1: world-space axis endpoints (Vector or 3-tuple)
        n_sides: 4, 6, 8, etc — sides of cross-section
        n_axial: number of axial rings (incl endpoints), >=2
        radius_profile: list of n_axial radii to vary thickness along axis.
                        None = constant radius=0.05.
        bulge_at: axial parameter in [0,1] for muscle bulge ring (e.g. 0.5)
        bulge_amount: extra radius added at bulge ring

    Returns: bpy.types.Object
    """
    import math
    import bmesh
    from mathutils import Vector
    p0 = Vector(p0); p1 = Vector(p1)
    axis = p1 - p0
    L = axis.length
    if L < 1e-6:
        raise ValueError("axis length zero")
    axis_dir = axis.normalized()
    ref = Vector((1, 0, 0)) if abs(axis_dir.x) < 0.9 else Vector((0, 1, 0))
    e1 = (ref - axis_dir * ref.dot(axis_dir)).normalized()
    e2 = axis_dir.cross(e1).normalized()

    if radius_profile is None:
        radius_profile = [0.05] * n_axial
    if len(radius_profile) != n_axial:
        # interpolate
        radius_profile = list(radius_profile) + [radius_profile[-1]] * (n_axial - len(radius_profile))

    # Build verts: n_axial rings × n_sides verts
    bm = bmesh.new()
    rings = []
    for ai in range(n_axial):
        t_norm = ai / (n_axial - 1)
        t = t_norm * L
        axis_pt = p0 + axis_dir * t
        r = radius_profile[ai]
        if bulge_at is not None and bulge_amount != 0:
            # Apply bulge gauss
            d = abs(t_norm - bulge_at)
            sigma = 0.15
            r += bulge_amount * math.exp(-(d*d) / (2*sigma*sigma))
        ring = []
        for si in range(n_sides):
            ang = (si / n_sides) * 2 * math.pi
            v = axis_pt + e1 * (r * math.cos(ang)) + e2 * (r * math.sin(ang))
            ring.append(bm.verts.new(v))
        rings.append(ring)

    # Build quad faces
    for ai in range(n_axial - 1):
        r0 = rings[ai]
        r1 = rings[ai + 1]
        for si in range(n_sides):
            ni = (si + 1) % n_sides
            bm.faces.new([r0[si], r0[ni], r1[ni], r1[si]])

    me = bpy.data.meshes.new(name + '_mesh')
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def manual_shrinkwrap_region(va_obj, target_obj, region_indices):
    """Pure-Python shrinkwrap: for each region vert, snap to nearest surface
    point on target. No bpy.ops, no modifier — works in any context.
    """
    from mathutils.bvhtree import BVHTree
    import bmesh
    tmesh = bmesh.new()
    tmesh.from_mesh(target_obj.data)
    tmesh.transform(target_obj.matrix_world)
    bvh = BVHTree.FromBMesh(tmesh)
    tmesh.free()

    va_mw = va_obj.matrix_world
    va_mw_inv = va_mw.inverted()
    n = 0
    for vi in region_indices:
        v = va_obj.data.vertices[vi]
        p_world = va_mw @ v.co
        hit, normal, idx, dist = bvh.find_nearest(p_world)
        if hit is None:
            continue
        v.co = va_mw_inv @ hit
        n += 1
    va_obj.data.update()
    return n


def shrinkwrap_region_to_target(va_obj, target_obj, region_indices,
                                  wrap_method='NEAREST_SURFACEPOINT',
                                  wrap_mode='ON_SURFACE', offset=0.0,
                                  vg_name='_BD_SW'):
    """Apply a one-shot shrinkwrap modifier targeting only specific verts.

    Creates a temporary vertex group, runs the modifier, applies, cleans up.
    Returns count of region verts.
    """
    # Remove stale vgroup if exists
    if vg_name in va_obj.vertex_groups:
        va_obj.vertex_groups.remove(va_obj.vertex_groups[vg_name])
    vg = va_obj.vertex_groups.new(name=vg_name)
    vg.add(list(region_indices), 1.0, 'REPLACE')

    bpy.context.view_layer.objects.active = va_obj
    m = va_obj.modifiers.new('BD_SW', 'SHRINKWRAP')
    m.target = target_obj
    m.wrap_method = wrap_method
    m.wrap_mode = wrap_mode
    m.vertex_group = vg_name
    m.offset = offset
    bpy.ops.object.modifier_apply(modifier=m.name)

    if vg_name in va_obj.vertex_groups:
        va_obj.vertex_groups.remove(va_obj.vertex_groups[vg_name])
    return len(region_indices)


def verts_in_capsule(obj, p0, p1, radius, joint_buffer=0.0):
    """Return set of vert indices inside a capsule (cylinder + 2 hemispheres).

    Args:
        p0, p1: world-space line segment endpoints (Vector or 3-tuple).
        radius: cylinder/hemisphere radius in world units.
        joint_buffer: if > 0, EXCLUDE verts within this distance of p0 or p1
                      (so joint deform loops stay intact).
    """
    from mathutils import Vector
    p0 = Vector(p0); p1 = Vector(p1)
    axis = p1 - p0
    L2 = axis.length_squared
    if L2 < 1e-10:
        return set()
    mw = obj.matrix_world
    result = set()
    for v in obj.data.vertices:
        p = mw @ v.co
        # Projection parameter t along axis
        t = (p - p0).dot(axis) / L2
        if t < 0 or t > 1:
            continue
        closest = p0 + axis * t
        if (p - closest).length > radius:
            continue
        # Apply joint buffer: exclude near endpoints
        if joint_buffer > 0:
            if (p - p0).length < joint_buffer or (p - p1).length < joint_buffer:
                continue
        result.add(v.index)
    return result


def verts_in_box(obj, x_range=None, y_range=None, z_range=None):
    """Return set of vert indices inside an axis-aligned world-space box.
    Any range = None means unbounded on that axis.
    """
    mw = obj.matrix_world
    result = set()
    for v in obj.data.vertices:
        p = mw @ v.co
        if x_range and not (x_range[0] <= p.x <= x_range[1]): continue
        if y_range and not (y_range[0] <= p.y <= y_range[1]): continue
        if z_range and not (z_range[0] <= p.z <= z_range[1]): continue
        result.add(v.index)
    return result


def flatten_indices(obj, region_indices, angle_threshold_deg=20.0, min_group_size=3,
                    max_group_size=None):
    """Coplanar-group flatten restricted to a set of vert indices.

    Faces are eligible only if ALL their verts are in region_indices.
    Verts in the region get snapped to their largest containing group's plane.
    """
    import math
    import bmesh
    from mathutils import Vector

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    n_faces = len(bm.faces)
    cos_thresh = math.cos(math.radians(angle_threshold_deg))

    # Eligible faces: all verts in region_indices
    eligible = [False] * n_faces
    for i, f in enumerate(bm.faces):
        if all(v.index in region_indices for v in f.verts):
            eligible[i] = True

    # Union-find over eligible faces
    parent = list(range(n_faces))
    size = [1] * n_faces
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if max_group_size and size[ra] + size[rb] > max_group_size:
            return  # would exceed cap
        # union by size
        if size[ra] < size[rb]: ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for f in bm.faces:
        if not eligible[f.index]: continue
        nA = f.normal
        for e in f.edges:
            for f2 in e.link_faces:
                if f2 is f: continue
                if not eligible[f2.index]: continue
                nB = f2.normal
                if nA.dot(nB) >= cos_thresh:
                    union(f.index, f2.index)

    # Collect groups
    groups = {}
    for i in range(n_faces):
        if not eligible[i]: continue
        groups.setdefault(find(i), []).append(i)
    groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}

    # Compute target position per vert (snap to largest containing group)
    best_for_vert = {}
    for face_idxs in groups.values():
        total_area = 0.0
        wn = Vector((0, 0, 0))
        all_verts = set()
        for fi in face_idxs:
            f = bm.faces[fi]
            a = f.calc_area()
            total_area += a
            wn += f.normal * a
            for v in f.verts:
                all_verts.add(v.index)
        if total_area < 1e-10 or not all_verts: continue
        normal = (wn / total_area).normalized()
        centroid = sum((bm.verts[vi].co for vi in all_verts), Vector()) / len(all_verts)
        for vi in all_verts:
            d = (bm.verts[vi].co - centroid).dot(normal)
            target = bm.verts[vi].co - normal * d
            cur = best_for_vert.get(vi)
            if cur is None or len(face_idxs) > cur[0]:
                best_for_vert[vi] = (len(face_idxs), target)

    for vi, (_, target) in best_for_vert.items():
        bm.verts[vi].co = target

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    stats = {
        'groups': len(groups),
        'faces_flattened': sum(len(v) for v in groups.values()),
        'verts_moved': len(best_for_vert),
    }
    return stats


def force_polygonal_cross_section(obj, region_indices, axis_p0, axis_p1,
                                   n_sides=8, n_axial_bins=6,
                                   preserve_radius_per_octant=True,
                                   joint_falloff=0.0,
                                   joint_buffer=0.0):
    """Force a limb region into a clean polygonal cylinder.

    For each vert in region:
      1. Project onto limb axis → axial parameter t in [0,1].
      2. Compute perpendicular vector → angle around axis.
      3. Bin angle to nearest octant (one of n_sides directions).
      4. Snap angle to bin center exactly.
      5. Optionally normalize radius — median radius of all verts in
         (axial_bin, angular_bin) so the bicep bulge profile stays but
         within each axial slice the radius is consistent per octant.

    Preserves axial position so muscle/joint shape stays.

    Args:
        obj: source mesh
        region_indices: vert indices to process
        axis_p0, axis_p1: world-space axis endpoints
        n_sides: polygon sides (6=hex, 8=oct)
        n_axial_bins: number of axial slices for radius normalization
        preserve_radius_per_octant: if True, each octant gets its own
                                     radius (preserves bicep bulge shape).
                                     If False, all octants in a slice
                                     share the same radius (perfect circle).
    """
    import math
    from mathutils import Vector
    p0 = Vector(axis_p0); p1 = Vector(axis_p1)
    axis = p1 - p0
    L2 = axis.length_squared
    if L2 < 1e-10:
        return 0
    axis_dir = axis.normalized()
    L = axis.length

    # Build stable perpendicular reference frame
    ref = Vector((1, 0, 0)) if abs(axis_dir.x) < 0.9 else Vector((0, 1, 0))
    e1 = (ref - axis_dir * ref.dot(axis_dir)).normalized()
    e2 = axis_dir.cross(e1).normalized()

    mw = obj.matrix_world
    mw_inv = mw.inverted()

    # First pass: compute (t, angle, radius) for each region vert
    info = {}  # vi -> (t, angle, radius)
    for vi in region_indices:
        p = mw @ obj.data.vertices[vi].co
        t = (p - p0).dot(axis_dir)
        if t < -0.05 or t > L + 0.05:
            continue
        axis_pt = p0 + axis_dir * t
        radial = p - axis_pt
        r = radial.length
        if r < 1e-5:
            continue
        u = radial.dot(e1)
        v = radial.dot(e2)
        angle = math.atan2(v, u)  # -pi..pi
        info[vi] = (t, angle, r)

    if not info:
        return 0

    # Second pass: bin by (axial_bin, angular_bin), compute median radius
    bin_size = L / max(n_axial_bins, 1)
    bin_radii = {}  # (axial_bin, angular_bin) -> list of radii
    for vi, (t, ang, r) in info.items():
        a_bin = max(0, min(n_axial_bins - 1, int(t / bin_size))) if L > 0 else 0
        # Angular bin: snap to nearest of n_sides directions
        # bins centered at angles 0, 2pi/N, 4pi/N, ...
        ang_normalized = (ang + math.pi) / (2 * math.pi) * n_sides  # 0..n_sides
        ang_bin = int(round(ang_normalized)) % n_sides
        bin_radii.setdefault((a_bin, ang_bin), []).append(r)

    bin_median = {}
    for k, rs in bin_radii.items():
        rs_sorted = sorted(rs)
        bin_median[k] = rs_sorted[len(rs_sorted) // 2]

    # Third pass: snap each vert to its bin's clean position with optional
    # joint falloff for smooth transition.
    n_moved = 0
    for vi, (t, ang, r) in info.items():
        a_bin = max(0, min(n_axial_bins - 1, int(t / bin_size))) if L > 0 else 0
        ang_normalized = (ang + math.pi) / (2 * math.pi) * n_sides
        ang_bin = int(round(ang_normalized)) % n_sides
        new_angle = (ang_bin / n_sides) * 2 * math.pi - math.pi
        if preserve_radius_per_octant:
            new_r = bin_median.get((a_bin, ang_bin), r)
        else:
            ax_radii = [bin_median[(a_bin, ab)] for ab in range(n_sides)
                        if (a_bin, ab) in bin_median]
            new_r = sum(ax_radii) / len(ax_radii) if ax_radii else r
        new_radial = e1 * (new_r * math.cos(new_angle)) + e2 * (new_r * math.sin(new_angle))
        axis_pt = p0 + axis_dir * t
        new_world = axis_pt + new_radial

        # Joint falloff: blend toward original near joints
        if joint_falloff > 0:
            # Distance from each endpoint
            d_start = max(0.0, t)
            d_end = max(0.0, L - t)
            d_joint = min(d_start, d_end)
            if d_joint < joint_falloff:
                # ratio: 0 at joint (use original), 1 at falloff distance (use new)
                ratio = d_joint / joint_falloff
                # Smoothstep for nicer transition
                ratio = ratio * ratio * (3 - 2 * ratio)
                original_world = mw @ obj.data.vertices[vi].co
                new_world = original_world.lerp(new_world, ratio)

        obj.data.vertices[vi].co = mw_inv @ new_world
        n_moved += 1

    obj.data.update()
    return n_moved


def laplacian_smooth_region(obj, region_indices, iterations=20, factor=0.5):
    """Heavy laplacian smoothing on a vert region.

    For each iteration: each vert moves toward the average of its neighbors,
    weighted by factor. Iterations multiply the effect.
    """
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Convert to set for O(1) lookup
    region_set = set(region_indices)

    for it in range(iterations):
        # Compute new positions
        new_positions = {}
        for vi in region_set:
            v = bm.verts[vi]
            if not v.link_edges:
                continue
            neighbors = [e.other_vert(v) for e in v.link_edges]
            if not neighbors:
                continue
            avg = sum((n.co for n in neighbors), v.co * 0) / len(neighbors)
            new_positions[vi] = v.co.lerp(avg, factor)
        # Apply
        for vi, p in new_positions.items():
            bm.verts[vi].co = p

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(region_set)


def project_radial_to_target(va_obj, target_obj, region_indices,
                              axis_p0, axis_p1, max_distance=0.5):
    """For each vert in region: preserve its axial position along the limb,
    radially project to nearest point on target's surface.

    This is anatomically-aware shrinkwrap — verts only move perpendicular
    to the limb axis, never along it. Prevents arm verts from being snapped
    onto torso geometry across the body.

    Args:
        va_obj: source mesh (verts will be modified).
        target_obj: target mesh (its surface defines the silhouette).
        region_indices: set of vert indices in va_obj to project.
        axis_p0, axis_p1: limb axis endpoints in world space.
        max_distance: if no surface hit within this distance, leave vert alone.
    """
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    p0 = Vector(axis_p0); p1 = Vector(axis_p1)
    axis = p1 - p0
    L2 = axis.length_squared
    if L2 < 1e-10:
        return 0
    axis_dir = axis.normalized()

    # Build BVH of target in target's world space
    import bmesh
    tmesh = bmesh.new()
    tmesh.from_mesh(target_obj.data)
    tmesh.transform(target_obj.matrix_world)
    bvh = BVHTree.FromBMesh(tmesh)
    tmesh.free()

    va_mw = va_obj.matrix_world
    va_mw_inv = va_mw.inverted()
    n_moved = 0

    for vi in region_indices:
        v = va_obj.data.vertices[vi]
        p_world = va_mw @ v.co

        # Axis parameter
        t = (p_world - p0).dot(axis_dir)
        axis_point = p0 + axis_dir * t

        # Radial direction (from axis to vert)
        radial = p_world - axis_point
        r_len = radial.length
        if r_len < 1e-6:
            continue
        radial_dir = radial / r_len

        # Cast ray FROM axis_point along radial_dir, hit target
        hit, normal, idx, dist = bvh.ray_cast(axis_point, radial_dir, max_distance)
        if hit is None:
            continue

        # New world position
        new_world = hit
        # Convert back to local
        v.co = va_mw_inv @ new_world
        n_moved += 1

    va_obj.data.update()
    return n_moved


def flatten_coplanar_groups(obj, angle_threshold_deg=7.0, min_group_size=2,
                            exclude_vgroups=None, only_vgroups=None,
                            weight_thresh=0.05):
    """Group adjacent faces whose normals differ by less than threshold, then
    snap all verts of each group to the group's mean plane.

    This produces the "Hard Body, soft Poly" faceted silhouette without
    reducing vert/face counts (so shape keys + UVs still work).

    Args:
        angle_threshold_deg: faces are grouped if their normal angle < this.
        min_group_size: only flatten groups with >= this many faces.
        exclude_vgroups: list of vgroup names whose verts are protected from
                         flattening (e.g., face, hands, breasts, butt).
        only_vgroups: list of vgroup names — if set, restricts flattening to
                      faces whose ALL verts are in these vgroups.
        weight_thresh: vgroup weight cutoff for region membership.

    Returns: dict of stats (groups_found, faces_flattened, verts_moved).
    """
    import math
    import bmesh
    from mathutils import Vector

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    # Build protected vert index set. A vert is protected if its DOMINANT
    # bone (the bone with the largest weight on it) is in the exclude list.
    # This avoids over-protection from Rigify's bleed-weighting where every
    # vert gets tiny influences from many bones.
    protected = set()
    if exclude_vgroups:
        ex_idxs = {obj.vertex_groups[n].index for n in exclude_vgroups
                   if n in obj.vertex_groups}
        for v in obj.data.vertices:
            if not v.groups: continue
            # Find the vgroup with the highest weight
            best_g = max(v.groups, key=lambda g: g.weight)
            if best_g.weight > weight_thresh and best_g.group in ex_idxs:
                protected.add(v.index)

    # Build "only" vert set if provided
    only_verts = None
    if only_vgroups:
        only_idxs = {obj.vertex_groups[n].index for n in only_vgroups
                     if n in obj.vertex_groups}
        only_verts = set()
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group in only_idxs and g.weight > weight_thresh:
                    only_verts.add(v.index)
                    break

    # Use bmesh in object mode
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    n_faces = len(bm.faces)
    cos_thresh = math.cos(math.radians(angle_threshold_deg))

    # Filter: which faces are eligible?
    eligible = [True] * n_faces
    if only_verts is not None:
        for i, f in enumerate(bm.faces):
            if not all(v.index in only_verts for v in f.verts):
                eligible[i] = False

    # Union-Find over faces
    parent = list(range(n_faces))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for f in bm.faces:
        if not eligible[f.index]:
            continue
        nA = f.normal
        for e in f.edges:
            for f2 in e.link_faces:
                if f2 is f: continue
                if not eligible[f2.index]: continue
                nB = f2.normal
                if nA.dot(nB) >= cos_thresh:
                    union(f.index, f2.index)

    # Collect groups
    groups = {}
    for i in range(n_faces):
        if not eligible[i]: continue
        r = find(i)
        groups.setdefault(r, []).append(i)

    groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}

    # For each group: mean normal + centroid -> plane. Snap verts.
    # A vert can be touched by multiple groups; snap to the LARGEST group.
    # Track vert->(largest_group_size, target_pos).
    best_for_vert = {}
    n_faces_flat = 0
    for root, face_idxs in groups.items():
        n_faces_flat += len(face_idxs)
        # Compute mean normal (area-weighted) + centroid
        total_area = 0.0
        weighted_normal = Vector((0, 0, 0))
        centroid_acc = Vector((0, 0, 0))
        face_set = set(face_idxs)
        for fi in face_idxs:
            f = bm.faces[fi]
            a = f.calc_area()
            total_area += a
            weighted_normal += f.normal * a
            for v in f.verts:
                centroid_acc += v.co * a
        if total_area < 1e-10: continue
        mean_normal = (weighted_normal / total_area).normalized()
        # Centroid via face-vert average
        all_verts = set()
        for fi in face_idxs:
            for v in bm.faces[fi].verts:
                all_verts.add(v.index)
        if not all_verts: continue
        centroid = sum((bm.verts[vi].co for vi in all_verts), Vector()) / len(all_verts)

        for vi in all_verts:
            if vi in protected: continue
            v = bm.verts[vi]
            # Vert participates in this group. We always snap it to the LARGEST
            # group's plane it belongs to — the boundary becomes a crease
            # between adjacent flattened planes (that's the faceted look).
            # Project onto plane
            d = (v.co - centroid).dot(mean_normal)
            target = v.co - mean_normal * d
            cur = best_for_vert.get(vi)
            if cur is None or len(face_idxs) > cur[0]:
                best_for_vert[vi] = (len(face_idxs), target)

    # Apply snaps
    n_verts_moved = 0
    for vi, (_, target) in best_for_vert.items():
        bm.verts[vi].co = target
        n_verts_moved += 1

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    stats = {
        'groups_found': len(groups),
        'faces_in_groups': n_faces_flat,
        'verts_moved': n_verts_moved,
        'face_pct_flattened': n_faces_flat / max(n_faces, 1),
    }
    print(f"[flatten_coplanar_groups] {stats}")
    return stats


def run(obj, regions, do_global_unsub=True):
    """Full reduction pipeline on obj. Returns dict of stats."""
    stats = {'start_verts': len(obj.data.vertices)}

    strip_modifiers(obj)
    strip_shape_keys(obj)
    obj.parent = None

    if do_global_unsub:
        global_unsub(obj, iterations=1)
        stats['after_unsub_verts'] = len(obj.data.vertices)

    for r in regions:
        reduce_region(obj, r)

    stats['end_verts'] = len(obj.data.vertices)
    stats['end_faces'] = len(obj.data.polygons)
    quad = tri = ngon = 0
    for p in obj.data.polygons:
        n = len(p.vertices)
        if n == 4: quad += 1
        elif n == 3: tri += 1
        else: ngon += 1
    stats['quad_pct'] = quad / max(quad + tri + ngon, 1)
    stats['tris'] = tri
    stats['ngons'] = ngon
    print(f"[reduce_va_basemesh] {stats}")
    return stats
