"""Joint assembly module — composite primitive stacks per joint.

Per `Procedural Joint Bone & Landmark Primitive Guide.md`. Each joint is built
from multiple low-poly primitives from `limb_primitives_lib.py`, positioned
and sized per the doc's width ratios and "outer ankle bone sits LOWER" rules.

Joint primitive stacks:

    KNEE   = Dual Femur Wedges (medial+lateral condyles)
           + Floating Patella Plate (flattened sphere, triangular taper)
           + Tibia Block Wedge (below patella)

    ELBOW  = Rear Olecranon Wedge (back of joint)
           + Flattened Hinge Plate (front cap)
           + Epicondyle Pair (side bumps)

    WRIST  = Dual Bone Wedges (Radius + Ulna heads)
           + Carpal Block (rounded box)

    ANKLE  = Offset Malleolus Pair (outer LOWER than inner per doc!)
           + Achilles Taper (rear tapered capsule)
           + Heel Block (rounded box, bottom-rear)

Width ratios per doc (relative to adjacent limb radius):
    Knee     = 0.55–0.70 of thigh
    Wrist    = 0.35–0.45 of forearm
    Ankle    = 0.35–0.45 of calf
    Heel     = 0.55

All primitives parented to a per-joint anchor empty for easy posing.

Usage:
    from face_base_pipeline.limb_joints_assembly import (
        assemble_knee, assemble_elbow, assemble_wrist, assemble_ankle
    )

    knee_objs = assemble_knee("KneeR",
                              joint_center=(-0.092, 0.023, 0.54),
                              limb_side=-1,
                              thigh_radius_at_knee=0.055)
"""

from __future__ import annotations
import sys
import os
import bpy
from mathutils import Vector

# Absolute import — limb_primitives_lib must be in sys.path. Adds the script
# dir automatically if not already there (idempotent).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from limb_primitives_lib import (
    create_wedge, create_flatsphere, create_box_rounded,
    create_offset_wedge_pair, create_plane_ridge, create_tapered_capsule,
)


# ============================================================
# JOINT WIDTH RATIOS (from Procedural Joint Bone & Landmark Primitive Guide.md)
# ============================================================

JOINT_WIDTH_RATIOS = {
    "elbow_core":   0.65,
    "knee":         0.62,  # midpoint of 0.55-0.70
    "wrist":        0.40,  # midpoint of 0.35-0.45
    "ankle":        0.40,  # midpoint of 0.35-0.45
    "heel":         0.55,
}


def _new_collection(name, parent=None):
    """Get or create a collection."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    if parent is None:
        parent = bpy.context.scene.collection
    parent.children.link(col)
    return col


def _make_joint_anchor(name, location):
    """Create the master anchor empty for a joint stack."""
    anchor = bpy.data.objects.new(f"{name}_root", None)
    anchor.empty_display_type = 'ARROWS'
    anchor.empty_display_size = 0.025
    anchor.location = location
    bpy.context.collection.objects.link(anchor)
    return anchor


def _reparent_to_root(obj, root, keep_world=False):
    """Parent obj to root. If keep_world=False, parent inverse becomes identity
    so the child sits at its specified local offset relative to root."""
    if keep_world:
        original_mw = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = original_mw
    else:
        obj.parent = root
        obj.matrix_parent_inverse.identity()


# ============================================================
# KNEE — Dual Femur Wedges + Floating Patella + Tibia Block
# ============================================================

def assemble_knee(name, joint_center, limb_side=-1, thigh_radius_at_knee=0.055,
                  collection_name=None):
    """Build the knee primitive stack.

    Anatomy: Femoral Condyles (2 side wedges) + Patella (floating flat sphere,
    triangular taper) + Tibial Plateau (block wedge below patella).
    Patella should FLOAT — not merged.

    name:           "KneeR" / "KneeL"
    joint_center:   world position of the knee center (between thigh tip + shin top)
    limb_side:      -1 for Right, +1 for Left (mirrors lateral structures)
    thigh_radius_at_knee: thigh's End Radius, used to size the joint.

    Returns: dict of created objects.
    """
    col = _new_collection(collection_name) if collection_name else None
    root = _make_joint_anchor(name, joint_center)
    if col: col.objects.link(root)

    # Knee core width
    knee_w = thigh_radius_at_knee * 2.0 * JOINT_WIDTH_RATIOS["knee"]
    knee_d = knee_w * 0.85   # depth slightly less than width
    knee_h = knee_w * 0.65   # height (vertical extent of joint)

    # 1) FEMORAL CONDYLES — dual side wedges, slightly forward
    # Medial condyle (inner side — toward midline)
    medial_x = -limb_side * knee_w * 0.30   # inner
    medial, m_anchor = create_wedge(
        f"{name}_Condyle_Medial",
        location=(joint_center[0] + medial_x,
                  joint_center[1] - knee_d * 0.10,
                  joint_center[2] + knee_h * 0.20),
        rotation_euler=(0, 0, 0),
        width=knee_w * 0.35, depth=knee_d * 0.55, height=knee_h * 0.55,
        apex_offset_y=knee_d * 0.05,
    )

    # Lateral condyle (outer side — away from midline)
    lateral_x = limb_side * knee_w * 0.30   # outer
    lateral, l_anchor = create_wedge(
        f"{name}_Condyle_Lateral",
        location=(joint_center[0] + lateral_x,
                  joint_center[1] - knee_d * 0.10,
                  joint_center[2] + knee_h * 0.20),
        rotation_euler=(0, 0, 0),
        width=knee_w * 0.40, depth=knee_d * 0.55, height=knee_h * 0.60,
        apex_offset_y=knee_d * 0.05,
    )

    # 2) PATELLA — floating flattened sphere, vertical-oriented, triangular taper
    # Sits ABOVE and FORWARD of the condyles. Should be visually independent.
    patella_offset_y = -knee_d * 0.55   # forward of joint center (- Y per scene convention)
    patella_offset_z = knee_h * 0.10    # slightly above center
    patella, p_anchor = create_flatsphere(
        f"{name}_Patella",
        location=(joint_center[0],
                  joint_center[1] + patella_offset_y,
                  joint_center[2] + patella_offset_z),
        rotation_euler=(0, 0, 0),
        radius_x=knee_w * 0.28,     # width
        radius_y=knee_d * 0.20,     # depth (project forward)
        radius_z=knee_h * 0.55,     # vertical (taller than wide)
        taper_z=-0.30,              # triangular taper downward per doc
    )

    # 3) TIBIAL PLATEAU — block wedge below patella, forming the calf-side base
    tibia, t_anchor = create_wedge(
        f"{name}_TibialPlateau",
        location=(joint_center[0],
                  joint_center[1] - knee_d * 0.05,
                  joint_center[2] - knee_h * 0.40),
        rotation_euler=(0, 0, 0),
        width=knee_w * 0.85, depth=knee_d * 0.85, height=knee_h * 0.40,
        apex_offset_y=0.0,
    )

    # Parent all primitives' anchors under the joint root
    for anchor in (m_anchor, l_anchor, p_anchor, t_anchor):
        _reparent_to_root(anchor, root, keep_world=True)

    if col:
        for o in (medial, lateral, patella, tibia, m_anchor, l_anchor, p_anchor, t_anchor):
            if o.name not in [obj.name for obj in col.objects]:
                col.objects.link(o)

    return {
        "root": root,
        "condyle_medial": medial,  "condyle_medial_anchor": m_anchor,
        "condyle_lateral": lateral, "condyle_lateral_anchor": l_anchor,
        "patella": patella,        "patella_anchor": p_anchor,
        "tibial_plateau": tibia,   "tibial_plateau_anchor": t_anchor,
    }


# ============================================================
# ELBOW — Rear Olecranon + Flattened Hinge Plate + Epicondyles
# ============================================================

def assemble_elbow(name, joint_center, limb_side=-1, upperarm_radius_at_elbow=0.030,
                    arm_axis_world=(0, 1, 0), collection_name=None):
    """Build elbow primitive stack.

    Anatomy: Olecranon Wedge (rear point) + Hinge Plate (front cap) + Epicondyle Pair.
    Elbow is asymmetric and rear-heavy per doc.

    arm_axis_world: unit vector along the arm (from shoulder to wrist) in world space.
                    Used to orient the olecranon to point opposite the arm flexion.
    """
    col = _new_collection(collection_name) if collection_name else None
    root = _make_joint_anchor(name, joint_center)
    if col: col.objects.link(root)

    elbow_w = upperarm_radius_at_elbow * 2.0 * JOINT_WIDTH_RATIOS["elbow_core"]
    elbow_d = elbow_w * 1.10   # depth > width (rear-heavy)
    elbow_h = elbow_w * 0.95

    # 1) OLECRANON — REAR wedge (back of elbow, opposite the bend direction).
    # For a T-pose arm hanging at the side, the elbow back points away from body.
    # In iter_22b: char faces +Y, R arm goes -X direction. Olecranon points away from torso (-X for R).
    olecranon_offset_x = limb_side * elbow_d * 0.30   # outward + back
    olecranon_offset_y = elbow_d * 0.40                # behind elbow center
    olecranon, ol_anchor = create_wedge(
        f"{name}_Olecranon",
        location=(joint_center[0] + olecranon_offset_x,
                  joint_center[1] + olecranon_offset_y,
                  joint_center[2]),
        rotation_euler=(0, 0, 0),
        width=elbow_w * 0.50, depth=elbow_d * 0.50, height=elbow_h * 0.70,
        apex_offset_y=elbow_d * 0.20,   # project the apex outward
    )

    # 2) HINGE PLATE — flattened front cap (between bicep tendon + radius head)
    hinge, h_anchor = create_flatsphere(
        f"{name}_HingePlate",
        location=(joint_center[0],
                  joint_center[1] - elbow_d * 0.10,   # slight forward
                  joint_center[2]),
        rotation_euler=(0, 0, 0),
        radius_x=elbow_w * 0.42, radius_y=elbow_d * 0.18, radius_z=elbow_h * 0.45,
        taper_z=0.0,
    )

    # 3) EPICONDYLE PAIR — small side bumps (medial + lateral)
    epi_medial_x = -limb_side * elbow_w * 0.45
    epi_lateral_x = limb_side * elbow_w * 0.45
    epi_medial, em_anchor = create_wedge(
        f"{name}_Epicondyle_Medial",
        location=(joint_center[0] + epi_medial_x,
                  joint_center[1],
                  joint_center[2]),
        width=elbow_w * 0.20, depth=elbow_d * 0.18, height=elbow_h * 0.30,
        apex_offset_y=0.0,
    )
    epi_lateral, el_anchor = create_wedge(
        f"{name}_Epicondyle_Lateral",
        location=(joint_center[0] + epi_lateral_x,
                  joint_center[1],
                  joint_center[2]),
        width=elbow_w * 0.22, depth=elbow_d * 0.20, height=elbow_h * 0.32,
        apex_offset_y=0.0,
    )

    for anchor in (ol_anchor, h_anchor, em_anchor, el_anchor):
        _reparent_to_root(anchor, root, keep_world=True)

    return {
        "root": root,
        "olecranon": olecranon,         "olecranon_anchor": ol_anchor,
        "hinge_plate": hinge,           "hinge_plate_anchor": h_anchor,
        "epicondyle_medial": epi_medial,    "epicondyle_medial_anchor": em_anchor,
        "epicondyle_lateral": epi_lateral,  "epicondyle_lateral_anchor": el_anchor,
    }


# ============================================================
# WRIST — Dual Bone Wedges + Carpal Block
# ============================================================

def assemble_wrist(name, joint_center, limb_side=-1, forearm_radius_at_wrist=0.022,
                    collection_name=None):
    """Build wrist primitive stack.

    Anatomy: Radius head wedge + Ulna head wedge + Carpal Block (rounded box).
    "Wrist is flat, asymmetrical, narrow — NOT cylindrical" per doc.
    """
    col = _new_collection(collection_name) if collection_name else None
    root = _make_joint_anchor(name, joint_center)
    if col: col.objects.link(root)

    wrist_w = forearm_radius_at_wrist * 2.0 * JOINT_WIDTH_RATIOS["wrist"]
    wrist_d = wrist_w * 0.70   # depth less than width (flat per doc)
    wrist_h = wrist_w * 0.85

    # 1) RADIUS HEAD — wedge on the thumb side
    # For R arm with limb_side=-1, thumb side is -X (toward body / inside) when palm faces forward
    # But this varies with hand pose. Default: thumb side = +X*-limb_side
    radius_x = -limb_side * wrist_w * 0.25
    radius_wedge, r_anchor = create_wedge(
        f"{name}_RadiusHead",
        location=(joint_center[0] + radius_x,
                  joint_center[1],
                  joint_center[2] + wrist_h * 0.10),
        width=wrist_w * 0.35, depth=wrist_d * 0.85, height=wrist_h * 0.65,
        apex_offset_y=0.0,
    )

    # 2) ULNA HEAD — wedge on the pinky side, slightly lower per anatomy
    ulna_x = limb_side * wrist_w * 0.30
    ulna_wedge, u_anchor = create_wedge(
        f"{name}_UlnaHead",
        location=(joint_center[0] + ulna_x,
                  joint_center[1],
                  joint_center[2] - wrist_h * 0.05),   # ulna head sits slightly lower than radius
        width=wrist_w * 0.30, depth=wrist_d * 0.85, height=wrist_h * 0.60,
        apex_offset_y=0.0,
    )

    # 3) CARPAL BLOCK — rounded box at the hand transition
    carpal, c_anchor = create_box_rounded(
        f"{name}_CarpalBlock",
        location=(joint_center[0],
                  joint_center[1],
                  joint_center[2] - wrist_h * 0.40),
        size_x=wrist_w * 0.95, size_y=wrist_d * 0.85, size_z=wrist_h * 0.30,
        bevel=0.003, bevel_segments=1,
    )

    for anchor in (r_anchor, u_anchor, c_anchor):
        _reparent_to_root(anchor, root, keep_world=True)

    return {
        "root": root,
        "radius_head": radius_wedge,  "radius_head_anchor": r_anchor,
        "ulna_head": ulna_wedge,      "ulna_head_anchor": u_anchor,
        "carpal_block": carpal,       "carpal_block_anchor": c_anchor,
    }


# ============================================================
# ANKLE — Offset Malleolus Pair + Achilles Taper + Heel Block
# ============================================================

def assemble_ankle(name, joint_center, limb_side=-1, calf_radius_at_ankle=0.032,
                    collection_name=None):
    """Build ankle primitive stack.

    Anatomy: Medial + Lateral Malleolus pair (outer LOWER than inner per doc!)
    + Achilles Taper (rear) + Heel Block (bottom-rear, slightly larger).
    "Outer ankle bone sits LOWER than inner — critical for realism."
    """
    col = _new_collection(collection_name) if collection_name else None
    root = _make_joint_anchor(name, joint_center)
    if col: col.objects.link(root)

    ankle_w = calf_radius_at_ankle * 2.0 * JOINT_WIDTH_RATIOS["ankle"]
    ankle_d = ankle_w * 0.85
    ankle_h = ankle_w * 0.80
    heel_w = calf_radius_at_ankle * 2.0 * JOINT_WIDTH_RATIOS["heel"]

    # 1) MALLEOLUS PAIR — outer wedge sits LOWER than inner per doc
    # For R foot (limb_side=-1): outer = -X side (away from body midline)
    # outer_direction_x = limb_side (so for R, outer is -X)
    malleolus, mal_anchor = create_offset_wedge_pair(
        f"{name}_Malleolus",
        location=joint_center,
        rotation_euler=(0, 0, 0),
        inner_size=ankle_w * 0.30,
        outer_size=ankle_w * 0.32,
        pair_separation=ankle_w * 0.50,
        outer_z_offset=-ankle_h * 0.20,   # LOWER per doc
        outer_direction_x=limb_side,
    )

    # 2) ACHILLES TAPER — rear tapered capsule (calf-side wide, ankle-side narrow)
    # Sits behind the ankle (positive Y if char faces +Y means behind = +Y... actually
    # the back of the body is -Y if char faces +Y. Let me check: in iter_22b empties,
    # ForearmR_P1 wrist is at Y=0.023, so wrist is slightly forward of joint origin.
    # For ankle: foot toes point forward (+Y), so back of ankle is -Y. Wait no, the
    # foot mesh from BaseFlatMan had X as the long axis... let me trust the convention:
    # back of ankle should be opposite the foot direction. For now, +Y per char-faces-+Y.
    # Achilles is BEHIND the ankle, so -Y in world space if char faces +Y).
    # Wait — actually re-reading my OVERNIGHT_STATUS: in iter_22b's "Coordinate Convention":
    # +Y = forward. So back of body = -Y. Back of ankle = -Y.
    achilles, a_anchor = create_tapered_capsule(
        f"{name}_AchillesTaper",
        location=(joint_center[0],
                  joint_center[1] - ankle_d * 0.55,    # behind ankle
                  joint_center[2] + ankle_h * 0.30),   # rising into calf
        rotation_euler=(0, 0, 0),    # Z-axis = vertical = capsule length axis
        length=ankle_h * 0.70,
        radius_start=ankle_w * 0.18,   # bottom = narrow (Achilles base near heel)
        radius_end=calf_radius_at_ankle * 0.65,   # top = widens into calf
        sides=8,
    )

    # 3) HEEL BLOCK — bottom-rear, slightly larger than ankle (heel pad)
    heel, h_anchor = create_box_rounded(
        f"{name}_HeelBlock",
        location=(joint_center[0],
                  joint_center[1] - heel_w * 0.30,   # behind
                  joint_center[2] - ankle_h * 0.45), # below
        size_x=heel_w * 0.85, size_y=heel_w * 0.90, size_z=ankle_h * 0.55,
        bevel=0.005, bevel_segments=1,
    )

    for anchor in (mal_anchor, a_anchor, h_anchor):
        _reparent_to_root(anchor, root, keep_world=True)

    return {
        "root": root,
        "malleolus": malleolus,    "malleolus_anchor": mal_anchor,
        "achilles": achilles,      "achilles_anchor": a_anchor,
        "heel": heel,              "heel_anchor": h_anchor,
    }


# ============================================================
# Top-level: assemble ALL 8 joints in a scene
# ============================================================

def assemble_all_joints_for_iter22b():
    """Build all 8 joints (L/R × 4 types) for the iter_22b scene.

    Reads joint centers from the existing limb endpoint empties:
        Knee  = ThighX_P1 (= ShinX_P0)
        Elbow = UpperArmX_P1 (= ForearmX_P0)
        Wrist = ForearmX_P1
        Ankle = ShinX_P1

    Returns: dict of joint name -> primitives dict
    """
    results = {}
    side_map = {"R": -1, "L": 1}

    for side, limb_side in side_map.items():
        # Knee
        knee_center = bpy.data.objects[f"Thigh{side}_P1"].location.copy()
        results[f"Knee{side}"] = assemble_knee(
            f"Knee{side}", tuple(knee_center), limb_side=limb_side,
            thigh_radius_at_knee=0.055,
        )

        # Elbow
        elbow_center = bpy.data.objects[f"UpperArm{side}_P1"].location.copy()
        results[f"Elbow{side}"] = assemble_elbow(
            f"Elbow{side}", tuple(elbow_center), limb_side=limb_side,
            upperarm_radius_at_elbow=0.030,
        )

        # Wrist
        wrist_center = bpy.data.objects[f"Forearm{side}_P1"].location.copy()
        results[f"Wrist{side}"] = assemble_wrist(
            f"Wrist{side}", tuple(wrist_center), limb_side=limb_side,
            forearm_radius_at_wrist=0.022,
        )

        # Ankle
        ankle_center = bpy.data.objects[f"Shin{side}_P1"].location.copy()
        results[f"Ankle{side}"] = assemble_ankle(
            f"Ankle{side}", tuple(ankle_center), limb_side=limb_side,
            calf_radius_at_ankle=0.032,
        )

    return results
