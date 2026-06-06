"""Geometry Nodes builder for BD_Pelvis — an inverted-triangle pelvis with a
semi-split-sphere butt.

Self-contained module. `ensure_pelvis_group()` builds a `GeometryNodeTree`
named `BD_Pelvis` whose base shape is:

  * an UPSIDE-DOWN TRIANGLE — wide at the hips (top), narrowing to the crotch
    (bottom). Built by a height-driven taper of a UV-sphere primitive.
  * a SEMI-SPLIT SPHERE on the posterior — one rounded butt mass with a centre
    cleft groove carving it into two cheeks.

Why this and not the old `BD_Hips`: prior attempts *displaced* a butt onto a
wrong primitive. Here the shape IS the primitive — the taper makes the inverted
triangle, the cleft makes the split sphere. No displacement-fighting.

Convention (matches the rest of the pipeline):
    Z up, X = left-right, -Y = anterior (front), +Y = posterior (back).

Faceting: the part is shaded SMOOTH (not flat). The hard-body faceting is a
later custom-split-normal pass; the GN part must stay smooth so it deforms.

Reference pattern: `ensure_hips_group()` in `limb_anatomy_geonode.py`
(UV-sphere base + Position -> SeparateXYZ -> Math chains -> one SetPosition).
The `_new_node` / `_link` helpers below are copied verbatim from that module so
this file is self-contained.
"""

from __future__ import annotations
import bpy


# ============================================================
# Internal node-graph builders (verbatim from limb_anatomy_geonode.py)
# ============================================================

def _new_node(tree, type_name, location=(0, 0), label=None):
    n = tree.nodes.new(type=type_name)
    n.location = location
    if label: n.label = label
    return n


def _link(tree, out_socket, in_socket):
    tree.links.new(out_socket, in_socket)


# ============================================================
# BD_Pelvis
# ============================================================

def ensure_pelvis_group(force_rebuild=False):
    """Build (or return) the BD_Pelvis GeometryNodeTree.

    Base primitive is a UV sphere. The graph deforms it into an inverted
    triangle (taper) with a semi-split-sphere butt (posterior bulge + centre
    cleft). One `Set Position` node receives the fully-composed position.

    Returns the BD_Pelvis GeometryNodeTree.
    """
    name = "BD_Pelvis"
    if name in bpy.data.node_groups:
        if not force_rebuild:
            return bpy.data.node_groups[name]
        bpy.data.node_groups.remove(bpy.data.node_groups[name])

    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.is_modifier = True
    iface = ng.interface
    iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    def fin(nm, dv, lo, hi):
        s = iface.new_socket(name=nm, in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = dv; s.min_value = lo; s.max_value = hi

    def iin(nm, dv, lo, hi):
        s = iface.new_socket(name=nm, in_out='INPUT', socket_type='NodeSocketInt')
        s.default_value = dv; s.min_value = lo; s.max_value = hi

    # ── float sockets (name, default, min, max) ──────────────────────────────
    fin("Hip Width",     1.30,  0.40, 2.50)   # X scale at the hips (top)
    fin("Depth",         1.00,  0.40, 2.00)   # overall front-back scale
    fin("Height",        0.70,  0.30, 1.40)   # Z scale
    fin("Crotch Narrow", 0.38,  0.10, 1.00)   # X/Y scale at the bottom -> inverted V
    fin("Front Taper",   0.55,  0.10, 1.00)   # how sharply the anterior narrows
    fin("Butt Size",     1.35,  0.50, 3.00)   # posterior bulge (the sphere mass)
    fin("Butt Height",  -0.20, -0.80, 0.40)   # Z centre of the butt mass (0 = mid)
    fin("Cleft Depth",   0.55,  0.00, 1.50)   # centre groove depth (splits sphere)
    fin("Cleft Width",   0.22,  0.05, 0.60)   # how wide the cleft groove is
    fin("Radius",        0.150, 0.02, 0.50)
    iin("Segments",      24, 6, 64)
    iin("Rings",         18, 4, 48)

    gi = _new_node(ng, 'NodeGroupInput',  (-2200, 0))
    go = _new_node(ng, 'NodeGroupOutput', (1900, 0))

    # ── base UV sphere ───────────────────────────────────────────────────────
    sph = _new_node(ng, 'GeometryNodeMeshUVSphere', (-1950, 260), "sphere")
    _link(ng, gi.outputs["Segments"], sph.inputs["Segments"])
    _link(ng, gi.outputs["Rings"],    sph.inputs["Rings"])
    _link(ng, gi.outputs["Radius"],   sph.inputs["Radius"])

    # ── position -> separate -> normalized coords  n = pos / Radius (~ -1..1) ─
    pos = _new_node(ng, 'GeometryNodeInputPosition', (-1950, -160), "pos")
    sep = _new_node(ng, 'ShaderNodeSeparateXYZ', (-1770, -160), "sep")
    _link(ng, pos.outputs["Position"], sep.inputs["Vector"])

    def divr(val_socket, y, lbl):
        d = _new_node(ng, 'ShaderNodeMath', (-1590, y), lbl)
        d.operation = 'DIVIDE'
        _link(ng, val_socket, d.inputs[0])
        _link(ng, gi.outputs["Radius"], d.inputs[1])
        return d.outputs["Value"]
    nx = divr(sep.outputs["X"], -60,  "nx")
    ny = divr(sep.outputs["Y"], -160, "ny")
    nz = divr(sep.outputs["Z"], -260, "nz")

    # ── inverted-triangle taper ──────────────────────────────────────────────
    # tnorm = (nz + 1) / 2  -> 0 at the bottom (crotch), 1 at the top (hips)
    tnorm = _new_node(ng, 'ShaderNodeMath', (-1410, -260), "tnorm")
    tnorm.operation = 'MULTIPLY_ADD'
    _link(ng, nz, tnorm.inputs[0])
    tnorm.inputs[1].default_value = 0.5
    tnorm.inputs[2].default_value = 0.5
    # taper = lerp(Crotch Narrow, 1.0, tnorm)  via MapRange From 0..1 To CN..1
    taper = _new_node(ng, 'ShaderNodeMapRange', (-1230, -260), "taper")
    taper.interpolation_type = 'LINEAR'
    taper.clamp = True
    _link(ng, tnorm.outputs["Value"], taper.inputs["Value"])
    taper.inputs["From Min"].default_value = 0.0
    taper.inputs["From Max"].default_value = 1.0
    _link(ng, gi.outputs["Crotch Narrow"], taper.inputs["To Min"])
    taper.inputs["To Max"].default_value = 1.0

    # x' = pos.x * Hip Width * taper
    xw = _new_node(ng, 'ShaderNodeMath', (-1040, 60), "x*hipw")
    xw.operation = 'MULTIPLY'
    _link(ng, sep.outputs["X"], xw.inputs[0])
    _link(ng, gi.outputs["Hip Width"], xw.inputs[1])
    xt = _new_node(ng, 'ShaderNodeMath', (-880, 60), "x*taper")
    xt.operation = 'MULTIPLY'
    _link(ng, xw.outputs["Value"], xt.inputs[0])
    _link(ng, taper.outputs["Result"], xt.inputs[1])

    # y' = pos.y * Depth * taper
    yd = _new_node(ng, 'ShaderNodeMath', (-1040, -120), "y*depth")
    yd.operation = 'MULTIPLY'
    _link(ng, sep.outputs["Y"], yd.inputs[0])
    _link(ng, gi.outputs["Depth"], yd.inputs[1])
    yt = _new_node(ng, 'ShaderNodeMath', (-880, -120), "y*taper")
    yt.operation = 'MULTIPLY'
    _link(ng, yd.outputs["Value"], yt.inputs[0])
    _link(ng, taper.outputs["Result"], yt.inputs[1])

    # z' = pos.z * Height
    zh = _new_node(ng, 'ShaderNodeMath', (-1040, -300), "z*height")
    zh.operation = 'MULTIPLY'
    _link(ng, sep.outputs["Z"], zh.inputs[0])
    _link(ng, gi.outputs["Height"], zh.inputs[1])

    # ── anterior triangle ridge ──────────────────────────────────────────────
    # ant = max(0, -ny)  -> 0..1, how anterior (front) the vert is
    negny = _new_node(ng, 'ShaderNodeMath', (-700, 240), "-ny")
    negny.operation = 'MULTIPLY'
    _link(ng, ny, negny.inputs[0])
    negny.inputs[1].default_value = -1.0
    ant = _new_node(ng, 'ShaderNodeMath', (-540, 240), "ant max0")
    ant.operation = 'MAXIMUM'
    _link(ng, negny.outputs["Value"], ant.inputs[0])
    ant.inputs[1].default_value = 0.0
    # x' = x' * (1 - Front Taper * ant)   -> front pinches toward a vertical ridge
    ftant = _new_node(ng, 'ShaderNodeMath', (-380, 240), "ftaper*ant")
    ftant.operation = 'MULTIPLY'
    _link(ng, gi.outputs["Front Taper"], ftant.inputs[0])
    _link(ng, ant.outputs["Value"], ftant.inputs[1])
    frontfac = _new_node(ng, 'ShaderNodeMath', (-220, 240), "1-ft*ant")
    frontfac.operation = 'SUBTRACT'
    frontfac.inputs[0].default_value = 1.0
    _link(ng, ftant.outputs["Value"], frontfac.inputs[1])
    xf = _new_node(ng, 'ShaderNodeMath', (-60, 60), "x*frontfac")
    xf.operation = 'MULTIPLY'
    _link(ng, xt.outputs["Value"], xf.inputs[0])
    _link(ng, frontfac.outputs["Value"], xf.inputs[1])

    # ── posterior butt — gaussian Z band centred at Butt Height ──────────────
    # zmask = exp(-((nz - Butt Height) / 0.6)^2)
    zoff = _new_node(ng, 'ShaderNodeMath', (-700, -480), "nz-bh")
    zoff.operation = 'SUBTRACT'
    _link(ng, nz, zoff.inputs[0])
    _link(ng, gi.outputs["Butt Height"], zoff.inputs[1])
    zdiv = _new_node(ng, 'ShaderNodeMath', (-540, -480), "/0.6")
    zdiv.operation = 'DIVIDE'
    _link(ng, zoff.outputs["Value"], zdiv.inputs[0])
    zdiv.inputs[1].default_value = 0.6
    zsq = _new_node(ng, 'ShaderNodeMath', (-380, -480), "zsq")
    zsq.operation = 'POWER'
    _link(ng, zdiv.outputs["Value"], zsq.inputs[0])
    zsq.inputs[1].default_value = 2.0
    zneg = _new_node(ng, 'ShaderNodeMath', (-220, -480), "-zsq")
    zneg.operation = 'MULTIPLY'
    _link(ng, zsq.outputs["Value"], zneg.inputs[0])
    zneg.inputs[1].default_value = -1.0
    zmask = _new_node(ng, 'ShaderNodeMath', (-60, -480), "zmask exp")
    zmask.operation = 'EXPONENT'
    _link(ng, zneg.outputs["Value"], zmask.inputs[0])

    # post = max(0, ny)  -> 0..1, how posterior (back)
    post = _new_node(ng, 'ShaderNodeMath', (-700, -640), "post max0")
    post.operation = 'MAXIMUM'
    _link(ng, ny, post.inputs[0])
    post.inputs[1].default_value = 0.0

    # bulge = Butt Size * 0.07 * post * zmask
    bsz = _new_node(ng, 'ShaderNodeMath', (100, -560), "bsize*0.07")
    bsz.operation = 'MULTIPLY'
    _link(ng, gi.outputs["Butt Size"], bsz.inputs[0])
    bsz.inputs[1].default_value = 0.07
    bpz = _new_node(ng, 'ShaderNodeMath', (260, -560), "*post")
    bpz.operation = 'MULTIPLY'
    _link(ng, bsz.outputs["Value"], bpz.inputs[0])
    _link(ng, post.outputs["Value"], bpz.inputs[1])
    bulge = _new_node(ng, 'ShaderNodeMath', (420, -560), "bulge")
    bulge.operation = 'MULTIPLY'
    _link(ng, bpz.outputs["Value"], bulge.inputs[0])
    _link(ng, zmask.outputs["Value"], bulge.inputs[1])

    # y' = y' + bulge  (push posterior verts further back)
    ybulge = _new_node(ng, 'ShaderNodeMath', (580, -300), "y+bulge")
    ybulge.operation = 'ADD'
    _link(ng, yt.outputs["Value"], ybulge.inputs[0])
    _link(ng, bulge.outputs["Value"], ybulge.inputs[1])

    # ── cleft — carve a vertical groove down the posterior centreline ────────
    # near_centre = exp(-(nx / Cleft Width)^2)
    cnx = _new_node(ng, 'ShaderNodeMath', (-700, -820), "nx/cw")
    cnx.operation = 'DIVIDE'
    _link(ng, nx, cnx.inputs[0])
    _link(ng, gi.outputs["Cleft Width"], cnx.inputs[1])
    cnxsq = _new_node(ng, 'ShaderNodeMath', (-540, -820), "cnx sq")
    cnxsq.operation = 'POWER'
    _link(ng, cnx.outputs["Value"], cnxsq.inputs[0])
    cnxsq.inputs[1].default_value = 2.0
    cnxneg = _new_node(ng, 'ShaderNodeMath', (-380, -820), "-cnx sq")
    cnxneg.operation = 'MULTIPLY'
    _link(ng, cnxsq.outputs["Value"], cnxneg.inputs[0])
    cnxneg.inputs[1].default_value = -1.0
    near_centre = _new_node(ng, 'ShaderNodeMath', (-220, -820), "near_centre")
    near_centre.operation = 'EXPONENT'
    _link(ng, cnxneg.outputs["Value"], near_centre.inputs[0])

    # cleft_mask = near_centre * max(0, ny) * zmask  (posterior + butt band only)
    cm1 = _new_node(ng, 'ShaderNodeMath', (100, -820), "ncentre*post")
    cm1.operation = 'MULTIPLY'
    _link(ng, near_centre.outputs["Value"], cm1.inputs[0])
    _link(ng, post.outputs["Value"], cm1.inputs[1])
    cleft_mask = _new_node(ng, 'ShaderNodeMath', (260, -820), "cleft_mask")
    cleft_mask.operation = 'MULTIPLY'
    _link(ng, cm1.outputs["Value"], cleft_mask.inputs[0])
    _link(ng, zmask.outputs["Value"], cleft_mask.inputs[1])

    # y' = y' - Cleft Depth * 0.07 * cleft_mask  (pull the centreline in)
    cd = _new_node(ng, 'ShaderNodeMath', (420, -820), "cdepth*0.07")
    cd.operation = 'MULTIPLY'
    _link(ng, gi.outputs["Cleft Depth"], cd.inputs[0])
    cd.inputs[1].default_value = 0.07
    cdm = _new_node(ng, 'ShaderNodeMath', (580, -820), "cd*mask")
    cdm.operation = 'MULTIPLY'
    _link(ng, cd.outputs["Value"], cdm.inputs[0])
    _link(ng, cleft_mask.outputs["Value"], cdm.inputs[1])
    ycleft = _new_node(ng, 'ShaderNodeMath', (740, -300), "y-cleft")
    ycleft.operation = 'SUBTRACT'
    _link(ng, ybulge.outputs["Value"], ycleft.inputs[0])
    _link(ng, cdm.outputs["Value"], ycleft.inputs[1])

    # ── combine x', y', z'  -> one Set Position ──────────────────────────────
    comb = _new_node(ng, 'ShaderNodeCombineXYZ', (940, -120), "final pos")
    _link(ng, xf.outputs["Value"],     comb.inputs["X"])
    _link(ng, ycleft.outputs["Value"], comb.inputs["Y"])
    _link(ng, zh.outputs["Value"],     comb.inputs["Z"])

    setp = _new_node(ng, 'GeometryNodeSetPosition', (1180, 160), "set pos")
    _link(ng, sph.outputs["Mesh"], setp.inputs["Geometry"])
    _link(ng, comb.outputs["Vector"], setp.inputs["Position"])

    # ── shade SMOOTH (faceting is a later custom-normal pass; stay smooth) ────
    shade = _new_node(ng, 'GeometryNodeSetShadeSmooth', (1480, 160), "smooth")
    shade.domain = 'FACE'
    shade.inputs["Shade Smooth"].default_value = True
    _link(ng, setp.outputs["Geometry"], shade.inputs["Geometry"])
    _link(ng, shade.outputs["Geometry"], go.inputs["Geometry"])
    return ng


if __name__ == "__main__":
    ensure_pelvis_group(force_rebuild=True)
    print("[pelvis_geonode] BD_Pelvis node group built.")
