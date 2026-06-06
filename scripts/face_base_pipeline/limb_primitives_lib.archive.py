"""Low-poly primitive library for procedural joint + landmark construction.

Per `Procedural Joint Bone & Landmark Primitive Guide.md` — joints are
composite stacks of small primitives (Wedge, FlatSphere, BoxRounded, etc).
This module builds those primitives as discrete low-poly mesh objects.

Each primitive is a static mesh built via bmesh, parented to an anchor empty
for placement. Standard Blender transform tools (move/rotate/scale the
empty) handle posing. To change structural params (width ratios, apex
offset), call the create_* function again.

Force-flat shading on all primitives — matches the "Hard Body, soft Poly"
aesthetic.

Usage:
    from face_base_pipeline.limb_primitives_lib import (
        create_wedge, create_flatsphere, create_box_rounded,
        create_offset_wedge_pair, create_plane_ridge, create_tapered_capsule,
    )

    # Patella (knee cap) — flattened sphere, vertical-oriented triangular bias
    patella = create_flatsphere("KneeR_Patella",
                                location=(-0.092, -0.030, 0.55),  # front of knee
                                radius_x=0.025,    # width
                                radius_y=0.012,    # depth (into knee)
                                radius_z=0.030,    # height (vertical)
                                taper_z=-0.3)      # slight triangular taper down

    # Olecranon (elbow point) — rear-pointing wedge
    olecranon = create_wedge("ElbowR_Olecranon",
                             location=(-0.265, 0.045, 1.085),    # back of elbow
                             rotation_euler=(0, 0, 0),
                             width=0.045, depth=0.030, height=0.040,
                             apex_offset_y=0.012)  # apex projects rearward

    # Malleolus pair (ankle bones) — outer LOWER than inner per doc
    malleolus = create_offset_wedge_pair("AnkleR_Malleolus",
                                         location=(-0.092, 0.018, 0.105),
                                         inner_size=0.012,
                                         outer_size=0.014,
                                         pair_separation=0.040,
                                         outer_z_offset=-0.008)  # outer sits LOWER
"""

from __future__ import annotations
import bpy
import bmesh
from mathutils import Vector


# ============================================================
# Primitive Constructors (static mesh builders, no GN required)
# ============================================================

def _finalize_mesh(mesh, flat_shade=True):
    """Validate, recalc normals, force flat shading."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    if flat_shade:
        for f in mesh.polygons:
            f.use_smooth = False
    mesh.update()


def _create_object_with_empty(name, mesh, location=(0, 0, 0), rotation_euler=(0, 0, 0),
                                parent_to_empty=True):
    """Create a mesh object at the given location. Optionally parent it to an
    anchor empty so the user can move/rotate/scale the primitive via the empty.

    Returns: (mesh_obj, anchor_empty_or_None)
    """
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    if parent_to_empty:
        anchor = bpy.data.objects.new(f"{name}_anchor", None)
        anchor.empty_display_type = 'PLAIN_AXES'
        anchor.empty_display_size = 0.015
        anchor.location = location
        anchor.rotation_euler = rotation_euler
        bpy.context.collection.objects.link(anchor)

        # Parent + keep transform at origin (so mesh follows empty exactly)
        obj.parent = anchor
        obj.matrix_parent_inverse.identity()
        return obj, anchor
    else:
        obj.location = location
        obj.rotation_euler = rotation_euler
        return obj, None


# ----------------------------------------------------------------------
# WEDGE — asymmetric triangular prism
# ----------------------------------------------------------------------
# Used for: olecranon, femur condyles, ASIS, malleolus, acromion
#
# Geometry: 6 verts forming a wedge with apex on one face.
#
#         apex (apex_offset_y forward of base center)
#        / | \
#       /  |  \
#      /   |   \
#     +----+----+   ← top edge of base rectangle
#    /|    |    |\
#   / |    |    | \
#  +--+----+----+--+  ← bottom edge of base rectangle
#
# width = X size, depth = Y size, height = Z size, apex_offset_y shifts apex.

def build_wedge_mesh(width, depth, height, apex_offset_y=0.0):
    """Return (verts, faces) for a wedge primitive.

    Width along X, depth along Y, height along Z. apex_offset_y projects the
    top apex forward/back along Y (positive Y = "out", e.g. olecranon points back).
    """
    w, d, h = width * 0.5, depth * 0.5, height * 0.5
    # Base rectangle (bottom): 4 verts
    # Top apex: 2 verts forming a ridge line (apex_offset_y from center)
    verts = [
        (-w, -d, -h),  # 0 base front-left
        ( w, -d, -h),  # 1 base front-right
        ( w,  d, -h),  # 2 base back-right
        (-w,  d, -h),  # 3 base back-left
        (-w*0.5, apex_offset_y,  h),  # 4 apex left
        ( w*0.5, apex_offset_y,  h),  # 5 apex right
    ]
    faces = [
        (0, 1, 2, 3),    # bottom
        (0, 1, 5, 4),    # front slope (down-front)
        (2, 3, 4, 5),    # back slope (down-back)
        (0, 3, 4),       # left cap (tri)
        (1, 5, 2),       # right cap (tri)
        (4, 5),          # top ridge (edge only, no face)
    ]
    # Remove edge-only entry from faces
    faces = [f for f in faces if len(f) >= 3]
    # Top ridge is implicit (shared edge between front + back slopes)
    return verts, faces


def create_wedge(name, location=(0,0,0), rotation_euler=(0,0,0),
                  width=0.04, depth=0.04, height=0.03, apex_offset_y=0.0,
                  parent_to_empty=True):
    """Create a wedge primitive. Default = a small 4x4x3cm bone landmark."""
    verts, faces = build_wedge_mesh(width, depth, height, apex_offset_y)
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    _finalize_mesh(mesh)
    return _create_object_with_empty(name, mesh, location, rotation_euler, parent_to_empty)


# ----------------------------------------------------------------------
# FLAT SPHERE — flattened ellipsoid (patella, joint pads)
# ----------------------------------------------------------------------
# Used for: patella (kneecap), generally any "bump" landmark.
# Geometry: low-poly UV sphere scaled non-uniformly.
# taper_z: vertical asymmetry — negative tapers downward (triangular patella).

def build_flatsphere_mesh(radius_x, radius_y, radius_z, segments=12, rings=6, taper_z=0.0):
    """Build a low-poly flattened ellipsoid.

    radius_x: horizontal width
    radius_y: depth (front-back, e.g. how far patella protrudes)
    radius_z: vertical height
    taper_z: -1..1, asymmetry along Z (negative = wider top, tapered bottom = patella)
    """
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=1.0)
    # Scale into ellipsoid + apply Z taper
    for v in bm.verts:
        v.co.x *= radius_x
        v.co.y *= radius_y
        # Taper: lower verts (Z<0) get scaled by (1+taper_z), upper by (1-taper_z)
        z = v.co.z
        taper_factor = 1.0 + taper_z * (-z)  # at z=-1, factor=1+taper_z; at z=+1, factor=1-taper_z
        v.co.x *= taper_factor
        v.co.y *= taper_factor
        v.co.z *= radius_z
    mesh = bpy.data.meshes.new("flatsphere_mesh")
    bm.to_mesh(mesh)
    bm.free()
    _finalize_mesh(mesh)
    return mesh


def create_flatsphere(name, location=(0,0,0), rotation_euler=(0,0,0),
                       radius_x=0.025, radius_y=0.012, radius_z=0.030,
                       segments=12, rings=6, taper_z=0.0,
                       parent_to_empty=True):
    """Create a flattened sphere primitive.

    For a patella: radius_x=0.025 (width), radius_y=0.012 (front projection),
    radius_z=0.030 (vertical), taper_z=-0.3 (triangular taper downward).
    """
    mesh = build_flatsphere_mesh(radius_x, radius_y, radius_z, segments, rings, taper_z)
    mesh.name = f"{name}_mesh"
    return _create_object_with_empty(name, mesh, location, rotation_euler, parent_to_empty)


# ----------------------------------------------------------------------
# BOX ROUNDED — rectangular block with bevel
# ----------------------------------------------------------------------
# Used for: pelvis, ribcage, carpal block, heel block, tibial plateau.
# Low-poly cube with optional bevel on the edges (low bevel count for flat-faceted look).

def build_box_rounded_mesh(size_x, size_y, size_z, bevel=0.005, bevel_segments=1):
    """Build a rounded box."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= size_x * 0.5
        v.co.y *= size_y * 0.5
        v.co.z *= size_z * 0.5
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:],
                        offset=bevel, segments=bevel_segments,
                        profile=0.5, affect='EDGES')
    mesh = bpy.data.meshes.new("boxrounded_mesh")
    bm.to_mesh(mesh)
    bm.free()
    _finalize_mesh(mesh)
    return mesh


def create_box_rounded(name, location=(0,0,0), rotation_euler=(0,0,0),
                       size_x=0.05, size_y=0.05, size_z=0.05,
                       bevel=0.005, bevel_segments=1, parent_to_empty=True):
    """Create a rounded box primitive."""
    mesh = build_box_rounded_mesh(size_x, size_y, size_z, bevel, bevel_segments)
    mesh.name = f"{name}_mesh"
    return _create_object_with_empty(name, mesh, location, rotation_euler, parent_to_empty)


# ----------------------------------------------------------------------
# OFFSET WEDGE PAIR — malleolus (ankle bones, outer LOWER than inner)
# ----------------------------------------------------------------------
# Two wedges side by side, the outer one offset vertically downward.

def create_offset_wedge_pair(name, location=(0,0,0), rotation_euler=(0,0,0),
                              inner_size=0.012, outer_size=0.014,
                              pair_separation=0.040, outer_z_offset=-0.008,
                              outer_direction_x=1.0, parent_to_empty=True):
    """Create paired wedges (inner + outer) — for ankle malleolus.

    Inner sits at +0, outer sits at +pair_separation along X (× outer_direction_x).
    outer_z_offset is negative per doc: "outer ankle bone sits LOWER than inner".

    For Right ankle: outer_direction_x = -1 (outer = -X side).
    For Left ankle:  outer_direction_x = +1 (outer = +X side).
    """
    # Build inner + outer wedges as ONE mesh
    bm = bmesh.new()

    # Inner wedge — apex toward body center (inverse of outer_direction)
    s_in = inner_size * 0.5
    inner_apex_x = -outer_direction_x * s_in * 0.5  # apex toward midline
    inner_verts = [
        bm.verts.new((-s_in, -s_in, -s_in)),
        bm.verts.new(( s_in, -s_in, -s_in)),
        bm.verts.new(( s_in,  s_in, -s_in)),
        bm.verts.new((-s_in,  s_in, -s_in)),
        bm.verts.new((inner_apex_x - s_in*0.25, 0,  s_in)),
        bm.verts.new((inner_apex_x + s_in*0.25, 0,  s_in)),
    ]
    bm.faces.new([inner_verts[0], inner_verts[1], inner_verts[2], inner_verts[3]])
    bm.faces.new([inner_verts[0], inner_verts[1], inner_verts[5], inner_verts[4]])
    bm.faces.new([inner_verts[2], inner_verts[3], inner_verts[4], inner_verts[5]])
    bm.faces.new([inner_verts[0], inner_verts[3], inner_verts[4]])
    bm.faces.new([inner_verts[1], inner_verts[5], inner_verts[2]])

    # Outer wedge — apex pointing outward (along outer_direction_x), offset DOWN
    s_out = outer_size * 0.5
    cx = pair_separation * outer_direction_x
    cz = outer_z_offset  # negative = lower
    outer_apex_x = outer_direction_x * s_out * 0.5  # apex outward
    outer_verts = [
        bm.verts.new((cx - s_out, -s_out, cz - s_out)),
        bm.verts.new((cx + s_out, -s_out, cz - s_out)),
        bm.verts.new((cx + s_out,  s_out, cz - s_out)),
        bm.verts.new((cx - s_out,  s_out, cz - s_out)),
        bm.verts.new((cx + outer_apex_x - s_out*0.25, 0, cz + s_out)),
        bm.verts.new((cx + outer_apex_x + s_out*0.25, 0, cz + s_out)),
    ]
    bm.faces.new([outer_verts[0], outer_verts[1], outer_verts[2], outer_verts[3]])
    bm.faces.new([outer_verts[0], outer_verts[1], outer_verts[5], outer_verts[4]])
    bm.faces.new([outer_verts[2], outer_verts[3], outer_verts[4], outer_verts[5]])
    bm.faces.new([outer_verts[0], outer_verts[3], outer_verts[4]])
    bm.faces.new([outer_verts[1], outer_verts[5], outer_verts[2]])

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    _finalize_mesh(mesh)
    return _create_object_with_empty(name, mesh, location, rotation_euler, parent_to_empty)


# ----------------------------------------------------------------------
# PLANE RIDGE — thin elongated bump (clavicle, IT band, tendon ridge)
# ----------------------------------------------------------------------
# A thin elongated rounded bar.

def build_plane_ridge_mesh(length, width, height, segments=4):
    """Build a thin ridge primitive.

    length: along X (primary axis)
    width:  along Y (cross-section width)
    height: along Z (how much it protrudes)
    segments: longitudinal segment count for smooth bend
    """
    bm = bmesh.new()
    # Build as a series of cross-sections along X, each a small bump rectangle
    for i in range(segments + 1):
        t = i / segments
        x = -length * 0.5 + t * length
        # height falls off near the ends (sin curve)
        import math
        z_factor = math.sin(t * math.pi)  # 0 at ends, 1 at middle
        h_t = height * z_factor
        # Each cross-section: 4 verts (rectangle)
        v0 = bm.verts.new((x, -width*0.5, 0))
        v1 = bm.verts.new((x,  width*0.5, 0))
        v2 = bm.verts.new((x,  width*0.5, h_t))
        v3 = bm.verts.new((x, -width*0.5, h_t))
    bm.verts.ensure_lookup_table()
    # Bridge faces between adjacent cross-sections
    for i in range(segments):
        a, b, c, d = bm.verts[i*4 : i*4 + 4]
        e, f, g, h = bm.verts[(i+1)*4 : (i+1)*4 + 4]
        bm.faces.new([a, e, f, b])     # bottom
        bm.faces.new([d, c, g, h])     # top
        bm.faces.new([a, d, h, e])     # front
        bm.faces.new([b, f, g, c])     # back
    # End caps
    bm.faces.new([bm.verts[0], bm.verts[1], bm.verts[2], bm.verts[3]])
    n = segments * 4
    bm.faces.new([bm.verts[n+3], bm.verts[n+2], bm.verts[n+1], bm.verts[n+0]])

    mesh = bpy.data.meshes.new("planeridge_mesh")
    bm.to_mesh(mesh)
    bm.free()
    _finalize_mesh(mesh)
    return mesh


def create_plane_ridge(name, location=(0,0,0), rotation_euler=(0,0,0),
                        length=0.10, width=0.012, height=0.006, segments=4,
                        parent_to_empty=True):
    """Create a thin ridge primitive — e.g., clavicle, IT band, tendon ridge."""
    mesh = build_plane_ridge_mesh(length, width, height, segments)
    mesh.name = f"{name}_mesh"
    return _create_object_with_empty(name, mesh, location, rotation_euler, parent_to_empty)


# ----------------------------------------------------------------------
# TAPERED CAPSULE — Achilles taper, vertebrae
# ----------------------------------------------------------------------
# A short capsule with different start/end radii.

def build_tapered_capsule_mesh(length, radius_start, radius_end, sides=8):
    """Build a capsule with asymmetric radii (Achilles tendon style)."""
    bm = bmesh.new()
    import math
    # Build 2 rings of N verts + 2 endpoint verts (caps)
    half_l = length * 0.5
    bot_ring = []
    top_ring = []
    for i in range(sides):
        ang = 2 * math.pi * i / sides
        bot_ring.append(bm.verts.new((radius_start * math.cos(ang),
                                       radius_start * math.sin(ang),
                                       -half_l)))
        top_ring.append(bm.verts.new((radius_end * math.cos(ang),
                                       radius_end * math.sin(ang),
                                       half_l)))
    bm.verts.ensure_lookup_table()
    # Side faces
    for i in range(sides):
        ni = (i + 1) % sides
        bm.faces.new([bot_ring[i], bot_ring[ni], top_ring[ni], top_ring[i]])
    # Endcaps (flat n-gon, low-poly)
    bm.faces.new(list(reversed(bot_ring)))
    bm.faces.new(top_ring)

    mesh = bpy.data.meshes.new("taperedcapsule_mesh")
    bm.to_mesh(mesh)
    bm.free()
    _finalize_mesh(mesh)
    return mesh


def create_tapered_capsule(name, location=(0,0,0), rotation_euler=(0,0,0),
                            length=0.10, radius_start=0.015, radius_end=0.008,
                            sides=8, parent_to_empty=True):
    """Create a tapered capsule — for Achilles, Erector spinae, etc."""
    mesh = build_tapered_capsule_mesh(length, radius_start, radius_end, sides)
    mesh.name = f"{name}_mesh"
    return _create_object_with_empty(name, mesh, location, rotation_euler, parent_to_empty)


# ----------------------------------------------------------------------
# Convenience: a registry for create_primitive(type, ...) dispatch
# ----------------------------------------------------------------------

PRIMITIVE_CREATORS = {
    "wedge":              create_wedge,
    "flatsphere":         create_flatsphere,
    "box_rounded":        create_box_rounded,
    "offset_wedge_pair":  create_offset_wedge_pair,
    "plane_ridge":        create_plane_ridge,
    "tapered_capsule":    create_tapered_capsule,
}


def create_primitive(primitive_type, name, **kwargs):
    """Dispatch to a primitive constructor by type string.

    primitive_type: one of "wedge", "flatsphere", "box_rounded",
                    "offset_wedge_pair", "plane_ridge", "tapered_capsule"
    name:           object name
    **kwargs:       passed to the constructor (location, rotation_euler, size params)

    Returns: (mesh_obj, anchor_empty)
    """
    if primitive_type not in PRIMITIVE_CREATORS:
        raise ValueError(f"Unknown primitive_type {primitive_type!r}. "
                         f"Available: {list(PRIMITIVE_CREATORS)}")
    return PRIMITIVE_CREATORS[primitive_type](name, **kwargs)
