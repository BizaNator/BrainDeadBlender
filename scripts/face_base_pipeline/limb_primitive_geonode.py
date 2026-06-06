"""Geometry Nodes-based limb primitive builder.

Constructs a Blender Geometry Nodes group `BD_LimbPrimitive` that generates a
clean polygonal cylinder limb (hex/octagonal) with tunable parameters:

    - p0_obj, p1_obj: Object inputs (Empties define the limb axis endpoints)
    - n_sides:       int 3..16 (cross-section sides; 6=hex, 8=oct)
    - n_axial:       int 2..16 (rings along axis)
    - start_radius:  float (radius at p0)
    - end_radius:    float (radius at p1)
    - bulge_at:      float 0..1 (axial position of muscle bulge)
    - bulge_amount:  float (extra radius at bulge — 0 = no bulge)
    - bulge_width:   float 0..1 (gaussian sigma of bulge falloff)

Each parameter is a real GN input — drives the result live. Move the empties
to reposition the limb; drag sliders to tune.

Usage:
    from face_base_pipeline.limb_primitive_geonode import (
        ensure_limb_primitive_node_group, create_limb_primitive_object
    )
    ng = ensure_limb_primitive_node_group()
    obj = create_limb_primitive_object("UpperArmL_GN", p0=(0.114,0.047,1.321),
                                       p1=(0.253,0.057,1.107),
                                       n_sides=8, start_radius=0.051,
                                       end_radius=0.029,
                                       bulge_at=0.35, bulge_amount=0.008)
"""

from __future__ import annotations
import bpy
from mathutils import Vector

NODE_GROUP_NAME = "BD_LimbPrimitive"


def _new_node(tree, type_name, location=(0, 0), label=None):
    n = tree.nodes.new(type=type_name)
    n.location = location
    if label:
        n.label = label
    return n


def _link(tree, out_socket, in_socket):
    tree.links.new(out_socket, in_socket)


def ensure_limb_primitive_node_group(force_rebuild=False):
    """Create (or return existing) `BD_LimbPrimitive` GN node group.

    If already exists, returns it untouched (so existing modifiers stay valid).
    Pass force_rebuild=True to recreate (will invalidate all referencing
    modifiers — only do this if you really want a clean slate).

    Returns: bpy.types.GeometryNodeTree
    """
    if NODE_GROUP_NAME in bpy.data.node_groups:
        if not force_rebuild:
            return bpy.data.node_groups[NODE_GROUP_NAME]
        bpy.data.node_groups.remove(bpy.data.node_groups[NODE_GROUP_NAME])

    ng = bpy.data.node_groups.new(NODE_GROUP_NAME, 'GeometryNodeTree')
    ng.is_modifier = True

    # --- Inputs (new-style interface, Blender 4.x+) ---
    iface = ng.interface

    # Geometry output (the result mesh)
    iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # Endpoints as Objects
    iface.new_socket(name="Start Empty", in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(name="End Empty",   in_out='INPUT', socket_type='NodeSocketObject')

    # Topology
    s = iface.new_socket(name="N Sides",  in_out='INPUT', socket_type='NodeSocketInt')
    s.default_value = 8; s.min_value = 3; s.max_value = 16
    s = iface.new_socket(name="N Axial",  in_out='INPUT', socket_type='NodeSocketInt')
    s.default_value = 4; s.min_value = 2; s.max_value = 16

    # Radius
    s = iface.new_socket(name="Start Radius", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.05; s.min_value = 0.001; s.max_value = 0.5
    s = iface.new_socket(name="End Radius",   in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.05; s.min_value = 0.001; s.max_value = 0.5

    # Symmetric bulge (uniform around axis)
    s = iface.new_socket(name="Bulge At",     in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.5; s.min_value = 0.0; s.max_value = 1.0
    s = iface.new_socket(name="Bulge Amount", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.0; s.min_value = -0.05; s.max_value = 0.05
    s = iface.new_socket(name="Bulge Width",  in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.15; s.min_value = 0.02; s.max_value = 0.5

    # Directional muscle peak (bicep, calf, deltoid — bulge concentrated in one direction)
    s = iface.new_socket(name="Muscle Dir",     in_out='INPUT', socket_type='NodeSocketVector')
    s.default_value = (0.0, 0.0, 0.0)  # 0 vector = peak disabled
    s = iface.new_socket(name="Muscle Axial",   in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.4; s.min_value = 0.0; s.max_value = 1.0
    s = iface.new_socket(name="Muscle Amount",  in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.0; s.min_value = -0.05; s.max_value = 0.05
    s = iface.new_socket(name="Muscle Sigma",   in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 0.15; s.min_value = 0.02; s.max_value = 0.5
    s = iface.new_socket(name="Muscle Concentration", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 2.0; s.min_value = 1.0; s.max_value = 8.0

    # --- Build the graph ---
    nodes = ng.nodes
    links = ng.links

    grp_in = _new_node(ng, 'NodeGroupInput', (-2000, 0))
    grp_out = _new_node(ng, 'NodeGroupOutput', (1800, 0))

    # ---- Pull endpoint positions from the two empties ----
    obj_info_p0 = _new_node(ng, 'GeometryNodeObjectInfo', (-1700, 600), "Start Pos")
    obj_info_p0.transform_space = 'ORIGINAL'
    _link(ng, grp_in.outputs["Start Empty"], obj_info_p0.inputs["Object"])

    obj_info_p1 = _new_node(ng, 'GeometryNodeObjectInfo', (-1700, 300), "End Pos")
    obj_info_p1.transform_space = 'ORIGINAL'
    _link(ng, grp_in.outputs["End Empty"], obj_info_p1.inputs["Object"])

    # ---- Build the axis line: Mesh Line with Count = N Axial ----
    mesh_line = _new_node(ng, 'GeometryNodeMeshLine', (-1400, 400), "Axis Line")
    mesh_line.mode = 'END_POINTS'
    _link(ng, grp_in.outputs["N Axial"], mesh_line.inputs["Count"])
    _link(ng, obj_info_p0.outputs["Location"], mesh_line.inputs["Start Location"])
    _link(ng, obj_info_p1.outputs["Location"], mesh_line.inputs["Offset"])
    # In END_POINTS mode, "Offset" socket = end position

    # ---- Convert to curve so we can sample parameter t ----
    mesh_to_curve = _new_node(ng, 'GeometryNodeMeshToCurve', (-1100, 400))
    _link(ng, mesh_line.outputs["Mesh"], mesh_to_curve.inputs["Mesh"])

    # ---- Compute spline parameter (0..1 along curve) ----
    spline_param = _new_node(ng, 'GeometryNodeSplineParameter', (-1100, 600))

    # ---- Compute radius at each point: lerp(start, end, t) + bulge ----
    # 1. lerp = start_radius + (end_radius - start_radius) * t
    sub_re_rs = _new_node(ng, 'ShaderNodeMath', (-800, 800), "end-start")
    sub_re_rs.operation = 'SUBTRACT'
    _link(ng, grp_in.outputs["End Radius"], sub_re_rs.inputs[0])
    _link(ng, grp_in.outputs["Start Radius"], sub_re_rs.inputs[1])

    mul_t = _new_node(ng, 'ShaderNodeMath', (-600, 800), "diff*t")
    mul_t.operation = 'MULTIPLY'
    _link(ng, sub_re_rs.outputs["Value"], mul_t.inputs[0])
    _link(ng, spline_param.outputs["Factor"], mul_t.inputs[1])

    lerp_r = _new_node(ng, 'ShaderNodeMath', (-400, 800), "lerp")
    lerp_r.operation = 'ADD'
    _link(ng, grp_in.outputs["Start Radius"], lerp_r.inputs[0])
    _link(ng, mul_t.outputs["Value"], lerp_r.inputs[1])

    # 2. bulge = bulge_amount * exp( -((t - bulge_at)^2) / (2*bulge_width^2) )
    diff_t = _new_node(ng, 'ShaderNodeMath', (-1100, 100), "t - bulge_at")
    diff_t.operation = 'SUBTRACT'
    _link(ng, spline_param.outputs["Factor"], diff_t.inputs[0])
    _link(ng, grp_in.outputs["Bulge At"], diff_t.inputs[1])

    sq_diff = _new_node(ng, 'ShaderNodeMath', (-900, 100), "(t-bulge_at)^2")
    sq_diff.operation = 'POWER'
    _link(ng, diff_t.outputs["Value"], sq_diff.inputs[0])
    sq_diff.inputs[1].default_value = 2.0

    bw_sq = _new_node(ng, 'ShaderNodeMath', (-900, -100), "bulge_width^2")
    bw_sq.operation = 'POWER'
    _link(ng, grp_in.outputs["Bulge Width"], bw_sq.inputs[0])
    bw_sq.inputs[1].default_value = 2.0

    two_bw_sq = _new_node(ng, 'ShaderNodeMath', (-700, -100), "2*bulge_width^2")
    two_bw_sq.operation = 'MULTIPLY'
    _link(ng, bw_sq.outputs["Value"], two_bw_sq.inputs[0])
    two_bw_sq.inputs[1].default_value = 2.0

    div_sq = _new_node(ng, 'ShaderNodeMath', (-500, 0), "div")
    div_sq.operation = 'DIVIDE'
    _link(ng, sq_diff.outputs["Value"], div_sq.inputs[0])
    _link(ng, two_bw_sq.outputs["Value"], div_sq.inputs[1])

    neg_div = _new_node(ng, 'ShaderNodeMath', (-300, 0), "neg")
    neg_div.operation = 'MULTIPLY'
    _link(ng, div_sq.outputs["Value"], neg_div.inputs[0])
    neg_div.inputs[1].default_value = -1.0

    exp_g = _new_node(ng, 'ShaderNodeMath', (-100, 0), "exp")
    exp_g.operation = 'EXPONENT'
    _link(ng, neg_div.outputs["Value"], exp_g.inputs[0])

    bulge = _new_node(ng, 'ShaderNodeMath', (100, 0), "bulge")
    bulge.operation = 'MULTIPLY'
    _link(ng, exp_g.outputs["Value"], bulge.inputs[0])
    _link(ng, grp_in.outputs["Bulge Amount"], bulge.inputs[1])

    # 3. final radius = lerp_r + bulge
    final_r = _new_node(ng, 'ShaderNodeMath', (300, 400), "final radius")
    final_r.operation = 'ADD'
    _link(ng, lerp_r.outputs["Value"], final_r.inputs[0])
    _link(ng, bulge.outputs["Value"], final_r.inputs[1])

    # ---- Create profile circle (N Sides verts) ----
    profile_circle = _new_node(ng, 'GeometryNodeMeshCircle', (-700, 0), "Profile")
    profile_circle.fill_type = 'NONE'
    _link(ng, grp_in.outputs["N Sides"], profile_circle.inputs["Vertices"])
    # Profile radius = 1.0 (will be scaled by curve radius via Curve to Mesh)
    profile_circle.inputs["Radius"].default_value = 1.0

    # Convert profile mesh to curve
    profile_to_curve = _new_node(ng, 'GeometryNodeMeshToCurve', (-500, 0))
    _link(ng, profile_circle.outputs["Mesh"], profile_to_curve.inputs["Mesh"])

    # ---- Curve to Mesh — feed radius via Scale input ----
    curve_to_mesh = _new_node(ng, 'GeometryNodeCurveToMesh', (500, 400))
    curve_to_mesh.inputs["Fill Caps"].default_value = False
    _link(ng, mesh_to_curve.outputs["Curve"], curve_to_mesh.inputs["Curve"])
    _link(ng, profile_to_curve.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
    _link(ng, final_r.outputs["Value"], curve_to_mesh.inputs["Scale"])

    # ---- DIRECTIONAL MUSCLE PEAK ----
    # For each vert: displace toward Muscle Dir by gauss(t-axial)/sigma * align^conc * amount
    # where t = vert's Y bbox factor proxy → use Spline Parameter would be ideal but verts
    # after Curve to Mesh inherit the spline parameter as an attribute. We can use Position
    # difference from axis origin projected onto the limb's curve, but simpler approach:
    # use a Capture Attribute on the SPLINE parameter just before Curve to Mesh, then
    # re-sample on the resulting mesh.

    # Capture spline factor as a mesh attribute via the curve
    capture_t = _new_node(ng, 'GeometryNodeCaptureAttribute', (300, 600), "capture_t")
    # Configure capture: domain=POINT, type=FLOAT
    try:
        capture_t.capture_items.new('FLOAT', 'CurveT')
        capture_t.domain = 'POINT'
    except Exception:
        # Older API
        capture_t.data_type = 'FLOAT'
        capture_t.domain = 'POINT'

    _link(ng, mesh_to_curve.outputs["Curve"], capture_t.inputs["Geometry"])
    # Wire spline param as the captured value
    capture_t.inputs["Value"].default_value = 0.0 if hasattr(capture_t.inputs["Value"], 'default_value') else None
    # The geometry that goes into Curve to Mesh is the captured curve
    _link(ng, spline_param.outputs["Factor"], capture_t.inputs[1])  # the value input is index 1 typically

    # Re-route: instead of mesh_to_curve→curve_to_mesh, go through capture
    # Remove existing link from mesh_to_curve→curve_to_mesh
    for link in list(ng.links):
        if link.from_node is mesh_to_curve and link.to_node is curve_to_mesh:
            ng.links.remove(link)
    _link(ng, capture_t.outputs["Geometry"], curve_to_mesh.inputs["Curve"])

    # ---- Now apply muscle peak displacement ----
    # Set Position node after Curve to Mesh
    set_pos = _new_node(ng, 'GeometryNodeSetPosition', (700, 200), "muscle peak")
    _link(ng, curve_to_mesh.outputs["Mesh"], set_pos.inputs["Geometry"])

    # axial gauss: exp(-(t - muscle_axial)^2 / (2*sigma^2))
    # 1. diff = t - muscle_axial
    t_attr_out = capture_t.outputs[1]  # captured value
    md_diff = _new_node(ng, 'ShaderNodeMath', (300, -300), "t-mAx")
    md_diff.operation = 'SUBTRACT'
    _link(ng, t_attr_out, md_diff.inputs[0])
    _link(ng, grp_in.outputs["Muscle Axial"], md_diff.inputs[1])

    md_sq = _new_node(ng, 'ShaderNodeMath', (450, -300), "diff^2")
    md_sq.operation = 'POWER'
    _link(ng, md_diff.outputs["Value"], md_sq.inputs[0])
    md_sq.inputs[1].default_value = 2.0

    md_sig_sq = _new_node(ng, 'ShaderNodeMath', (450, -450), "sig^2")
    md_sig_sq.operation = 'POWER'
    _link(ng, grp_in.outputs["Muscle Sigma"], md_sig_sq.inputs[0])
    md_sig_sq.inputs[1].default_value = 2.0

    md_2sig = _new_node(ng, 'ShaderNodeMath', (600, -450), "2*sig^2")
    md_2sig.operation = 'MULTIPLY'
    _link(ng, md_sig_sq.outputs["Value"], md_2sig.inputs[0])
    md_2sig.inputs[1].default_value = 2.0

    md_div = _new_node(ng, 'ShaderNodeMath', (750, -350), "div")
    md_div.operation = 'DIVIDE'
    _link(ng, md_sq.outputs["Value"], md_div.inputs[0])
    _link(ng, md_2sig.outputs["Value"], md_div.inputs[1])

    md_neg = _new_node(ng, 'ShaderNodeMath', (900, -350), "neg")
    md_neg.operation = 'MULTIPLY'
    _link(ng, md_div.outputs["Value"], md_neg.inputs[0])
    md_neg.inputs[1].default_value = -1.0

    md_exp = _new_node(ng, 'ShaderNodeMath', (1050, -350), "exp")
    md_exp.operation = 'EXPONENT'
    _link(ng, md_neg.outputs["Value"], md_exp.inputs[0])

    # angular alignment: dot(normalize(vert_radial), normalize(muscle_dir))
    # vert_radial = vert_pos - axis_point. Without axis_point, use a proxy:
    # the vert's normal direction. Curve to Mesh outputs verts whose displacement from
    # the spline center is parallel to (vert_pos - spline_origin) projected to plane.
    # Easier: use the vert's Position attribute and approximate axis_point as the
    # spline's midpoint by capturing the start/end via Object Info — but that's complex.
    # Simpler: use the vert's Normal which IS the outward radial direction after Curve to Mesh.

    pos_attr = _new_node(ng, 'GeometryNodeInputPosition', (300, -600), "pos")
    # Normalize muscle_dir
    norm_md = _new_node(ng, 'ShaderNodeVectorMath', (300, -750), "norm md")
    norm_md.operation = 'NORMALIZE'
    _link(ng, grp_in.outputs["Muscle Dir"], norm_md.inputs[0])

    # Use vert Normal as radial direction proxy
    normal_attr = _new_node(ng, 'GeometryNodeInputNormal', (300, -900), "normal")

    # dot(normal, muscle_dir_norm)
    md_dot = _new_node(ng, 'ShaderNodeVectorMath', (500, -800), "dot")
    md_dot.operation = 'DOT_PRODUCT'
    _link(ng, normal_attr.outputs["Normal"], md_dot.inputs[0])
    _link(ng, norm_md.outputs["Vector"], md_dot.inputs[1])

    # max(0, dot)
    md_max = _new_node(ng, 'ShaderNodeMath', (700, -800), "max0")
    md_max.operation = 'MAXIMUM'
    _link(ng, md_dot.outputs["Value"], md_max.inputs[0])
    md_max.inputs[1].default_value = 0.0

    # dot^concentration
    md_pow = _new_node(ng, 'ShaderNodeMath', (900, -800), "pow")
    md_pow.operation = 'POWER'
    _link(ng, md_max.outputs["Value"], md_pow.inputs[0])
    _link(ng, grp_in.outputs["Muscle Concentration"], md_pow.inputs[1])

    # final factor = axial_gauss * angular_pow * amount
    fac1 = _new_node(ng, 'ShaderNodeMath', (1200, -600), "ax*ang")
    fac1.operation = 'MULTIPLY'
    _link(ng, md_exp.outputs["Value"], fac1.inputs[0])
    _link(ng, md_pow.outputs["Value"], fac1.inputs[1])

    fac2 = _new_node(ng, 'ShaderNodeMath', (1400, -600), "*amount")
    fac2.operation = 'MULTIPLY'
    _link(ng, fac1.outputs["Value"], fac2.inputs[0])
    _link(ng, grp_in.outputs["Muscle Amount"], fac2.inputs[1])

    # displacement = factor * normalized_muscle_dir
    disp = _new_node(ng, 'ShaderNodeVectorMath', (1600, -600), "disp")
    disp.operation = 'SCALE'
    _link(ng, norm_md.outputs["Vector"], disp.inputs[0])
    _link(ng, fac2.outputs["Value"], disp.inputs["Scale"])

    # new_pos = pos + disp
    add_pos = _new_node(ng, 'ShaderNodeVectorMath', (700, 0), "add")
    add_pos.operation = 'ADD'
    _link(ng, pos_attr.outputs["Position"], add_pos.inputs[0])
    _link(ng, disp.outputs["Vector"], add_pos.inputs[1])

    _link(ng, add_pos.outputs["Vector"], set_pos.inputs["Position"])

    # ---- Force FLAT shading so facets read clearly ----
    set_shade = _new_node(ng, 'GeometryNodeSetShadeSmooth', (1100, 400))
    set_shade.domain = 'FACE'
    set_shade.inputs["Shade Smooth"].default_value = False
    _link(ng, set_pos.outputs["Geometry"], set_shade.inputs["Mesh"])

    # ---- Output ----
    _link(ng, set_shade.outputs["Geometry"], grp_out.inputs["Geometry"])

    return ng


def create_limb_primitive_object(name, p0, p1,
                                  n_sides=8, n_axial=4,
                                  start_radius=0.05, end_radius=0.05,
                                  bulge_at=0.5, bulge_amount=0.0, bulge_width=0.15,
                                  muscle_dir=(0.0, 0.0, 0.0), muscle_axial=0.4,
                                  muscle_amount=0.0, muscle_sigma=0.15,
                                  muscle_concentration=2.0):
    """Create an object using the BD_LimbPrimitive node group, with two
    empties at p0 and p1 as endpoints.

    Returns: (limb_obj, p0_empty, p1_empty)
    """
    ng = ensure_limb_primitive_node_group()

    # Create empties for endpoints
    p0_empty = bpy.data.objects.new(f"{name}_P0", None)
    p0_empty.empty_display_type = 'PLAIN_AXES'
    p0_empty.empty_display_size = 0.03
    p0_empty.location = Vector(p0)
    bpy.context.scene.collection.objects.link(p0_empty)

    p1_empty = bpy.data.objects.new(f"{name}_P1", None)
    p1_empty.empty_display_type = 'PLAIN_AXES'
    p1_empty.empty_display_size = 0.03
    p1_empty.location = Vector(p1)
    bpy.context.scene.collection.objects.link(p1_empty)

    # Create the limb mesh object with a GN modifier
    me = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)

    mod = obj.modifiers.new("BD_LimbPrim", 'NODES')
    mod.node_group = ng

    # Wire inputs. In Blender 4.x, inputs are accessed by socket identifier
    # (e.g., mod["Input_2"]). We need to look those up.
    # Easier: iterate the interface and set values by name.
    def set_input(name_, value):
        # Find the socket identifier matching the input name
        for item in ng.interface.items_tree:
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == name_:
                mod[item.identifier] = value
                return
        print(f"[create_limb_primitive_object] WARN: input '{name_}' not found")

    set_input("Start Empty", p0_empty)
    set_input("End Empty", p1_empty)
    set_input("N Sides", n_sides)
    set_input("N Axial", n_axial)
    set_input("Start Radius", start_radius)
    set_input("End Radius", end_radius)
    set_input("Bulge At", bulge_at)
    set_input("Bulge Amount", bulge_amount)
    set_input("Bulge Width", bulge_width)
    set_input("Muscle Dir", Vector(muscle_dir))
    set_input("Muscle Axial", muscle_axial)
    set_input("Muscle Amount", muscle_amount)
    set_input("Muscle Sigma", muscle_sigma)
    set_input("Muscle Concentration", muscle_concentration)

    # Force a depsgraph refresh
    obj.update_tag()

    return obj, p0_empty, p1_empty
