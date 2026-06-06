"""Torso assembly module — 4 independent builder groups.

Per `procedural_torso_shape_and_measurement_construction_guide.md` +
`Procedural Human Torso Anatomy Guide.md`. The torso is split into 4
INDEPENDENT systems each with its own archetype table:

    1. STRUCTURAL  — rib cage + clavicles + trapezius + scapula (bony layer)
    2. CHEST       — breasts (F) OR pec slabs (M) — soft tissue, separate
    3. ABDOMEN     — rectus + obliques + serratus — visibility = low body fat
    4. BACK        — lats V-taper

This split lets users independently combine archetypes (e.g. "big bust +
flat abs" curvy, "small bust + ripped abs" athlete). Each maps to its own
set of UEFN morph targets.

Per construction guide MASTER TORSO RATIOS — measurements are relative to
ShoulderWidth (the master scalar):

    Average Female: rib=0.82, waist=0.64, pelvis=1.08, chest_d=0.34, ab_d=0.28
    Curvy Female:   rib=0.80, waist=0.58, pelvis=1.18, chest_d=0.40, ab_d=0.32
    Athletic F:     rib=0.85, waist=0.62, pelvis=1.00 (estimated)
    Average Male:   rib=0.86, waist=0.72, pelvis=0.84, chest_d=0.42, ab_d=0.34
    Athletic Male:  rib=0.90, waist=0.66, pelvis=0.82, chest_d=0.48, ab_d=0.30

Notable: female PELVIS is WIDER than shoulders (1.08-1.18×), male pelvis is
NARROWER (0.82-0.84×). This is the silhouette signature.
"""

from __future__ import annotations
import sys
import os
import math
import bpy
from mathutils import Vector

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from limb_primitives_lib import (
    create_wedge, create_flatsphere, create_box_rounded,
    create_plane_ridge, create_tapered_capsule,
)


# ============================================================
# MASTER TORSO RATIOS (per construction guide)
# All measurements relative to ShoulderWidth (= 2 × shoulder_half_x).
# ============================================================

MASTER_RATIOS = {
    # gender_archetype: {rib_w, waist_w, pelvis_w, chest_d, abdomen_d}
    "f_average":     {"rib_w":0.82, "waist_w":0.64, "pelvis_w":1.08, "chest_d":0.34, "abdomen_d":0.28},
    "f_curvy":       {"rib_w":0.80, "waist_w":0.58, "pelvis_w":1.18, "chest_d":0.40, "abdomen_d":0.32},
    "f_athletic":    {"rib_w":0.85, "waist_w":0.62, "pelvis_w":1.00, "chest_d":0.36, "abdomen_d":0.26},
    "f_stylized":    {"rib_w":0.78, "waist_w":0.52, "pelvis_w":1.20, "chest_d":0.42, "abdomen_d":0.30},
    "m_average":     {"rib_w":0.86, "waist_w":0.72, "pelvis_w":0.84, "chest_d":0.42, "abdomen_d":0.34},
    "m_athletic":    {"rib_w":0.90, "waist_w":0.66, "pelvis_w":0.82, "chest_d":0.48, "abdomen_d":0.30},
    "m_heroic":      {"rib_w":0.95, "waist_w":0.62, "pelvis_w":0.80, "chest_d":0.55, "abdomen_d":0.32},
}


# ============================================================
# SUBSYSTEM ARCHETYPE TABLES (independent — mix and match)
# ============================================================

# Chest sizes (female breast volume / male pec mass)
CHEST_ARCHETYPES_F = {
    "flat":      0.40,   # very small
    "small":     0.65,
    "average":   1.00,
    "full":      1.30,
    "large":     1.65,
    "very_large":2.10,
}

CHEST_ARCHETYPES_M = {
    "narrow":    0.60,
    "average":   1.00,
    "athletic":  1.30,
    "bodybuilder":1.80,
    "heroic":    2.10,
}

# Abdomen visibility (driven by body fat — low fat = visible abs)
AB_ARCHETYPES = {
    "soft":      0.10,   # high fat — abs invisible
    "average":   0.30,
    "fit":       0.60,
    "athletic":  0.90,
    "ripped":    1.40,
    "shredded":  1.80,
}

# Back V-taper (lat width)
BACK_ARCHETYPES = {
    "narrow":    0.50,
    "average":   1.00,
    "athletic":  1.40,
    "heroic":    1.80,
    "anime":     2.20,
}

# Structural — rarely needs adjustment (bones don't grow much with fitness)
STRUCTURAL_ARCHETYPES = {
    "average":   1.00,
    "broad":     1.15,
    "narrow":    0.88,
    "heroic":    1.25,
}


# ============================================================
# Anchor framework
# ============================================================

class TorsoAnchors:
    """Holds derived measurements that drive all primitive placements.

    Construct once from limb endpoints + a master archetype, pass to each
    subsystem builder. All distances in meters.
    """

    def __init__(self, shoulder_half_x, hip_top_z, shoulder_z,
                 torso_center_y=0.020, gender_archetype="f_average"):
        self.shoulder_half_x = shoulder_half_x
        self.hip_top_z = hip_top_z
        self.shoulder_z = shoulder_z
        self.torso_center_y = torso_center_y

        # Master ratios
        r = MASTER_RATIOS.get(gender_archetype, MASTER_RATIOS["f_average"])
        self.gender_archetype = gender_archetype

        # Derived widths (half = +/- from center)
        self.shoulder_w = shoulder_half_x * 2
        self.rib_half_x    = self.shoulder_w * r["rib_w"]    * 0.5
        self.waist_half_x  = self.shoulder_w * r["waist_w"]  * 0.5
        self.pelvis_half_x = self.shoulder_w * r["pelvis_w"] * 0.5

        # Derived depths (front-back)
        self.chest_depth   = self.shoulder_w * r["chest_d"]
        self.abdomen_depth = self.shoulder_w * r["abdomen_d"]

        # Derived heights (using master torso height)
        self.torso_height = shoulder_z - hip_top_z
        self.rib_height = self.torso_height * 0.42
        self.clavicle_height_zone = self.torso_height * 0.12

        # Z positions of landmarks
        self.clavicle_z    = shoulder_z - 0.020
        self.chest_z       = shoulder_z - self.torso_height * 0.22  # upper chest center
        self.rib_bottom_z  = hip_top_z + self.torso_height * 0.50   # midway
        self.waist_z       = hip_top_z + self.torso_height * 0.28   # narrowest below ribs
        self.navel_z       = hip_top_z + self.torso_height * 0.32

        # Y positions (body depth)
        self.torso_front_y = torso_center_y + self.chest_depth * 0.5
        self.torso_back_y  = torso_center_y - self.chest_depth * 0.5

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ============================================================
# Helper
# ============================================================

def _make_root_anchor(name, location):
    root = bpy.data.objects.new(f"{name}_root", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.04
    root.location = location
    bpy.context.collection.objects.link(root)
    return root


def _reparent(child_obj, root, keep_world=True):
    """Reparent child's anchor empty (if it has one) to root."""
    if child_obj.parent and child_obj.parent.type == 'EMPTY':
        anchor = child_obj.parent
        if keep_world:
            mw = anchor.matrix_world.copy()
            anchor.parent = root
            anchor.matrix_world = mw
        else:
            anchor.parent = root


# ============================================================
# SUBSYSTEM 1 — STRUCTURAL FRAME
# Rib cage + clavicles + trapezius + scapulae (bone-layer landmarks)
# ============================================================

def assemble_torso_structural(name, anchors, structural_scale=1.0):
    """Build rib cage + clavicles + traps + scapulae.

    These are bone-layer landmarks per the joint guide — they scale LESS
    than soft tissue. The structural_scale param can broaden the frame
    (heroic builds) but defaults to 1.0 for realism.
    """
    a = anchors
    root = _make_root_anchor(name, (0, a.torso_center_y, (a.hip_top_z + a.shoulder_z)/2))
    results = {"root": root}

    # 1) RIB CAGE — egg-shape, tilted forward
    rib, _ = create_flatsphere(
        f"{name}_RibCage",
        location=(0, a.torso_center_y + 0.008, (a.rib_bottom_z + a.clavicle_z)/2),
        rotation_euler=(math.radians(-6), 0, 0),
        radius_x=a.rib_half_x * structural_scale,
        radius_y=a.chest_depth * 0.45 * structural_scale,
        radius_z=a.rib_height * 0.50,
        segments=14, rings=10,
        taper_z=0.08,  # slightly narrower at top
    )
    results["rib_cage"] = rib

    # 2) WAIST TAPER — connect rib base to hip top
    waist_z_center = (a.rib_bottom_z + a.hip_top_z) / 2
    waist_h = a.rib_bottom_z - a.hip_top_z
    waist_r_top = a.rib_half_x * 0.78
    waist_r_bot = a.pelvis_half_x * 0.88

    waist, _ = create_tapered_capsule(
        f"{name}_Waist",
        location=(0, a.torso_center_y, waist_z_center),
        length=waist_h * 1.05,
        radius_start=waist_r_bot,
        radius_end=waist_r_top,
        sides=12,
    )
    results["waist"] = waist

    # 3) CLAVICLES — pair of ridges, 8-18° upward outward angle
    clav_len = a.shoulder_w * 0.45    # per "Clavicle Width 0.78-0.92 Shoulder Width" (half each)
    for side, side_sign in (("L", 1), ("R", -1)):
        clav_mid_x = side_sign * a.shoulder_half_x * 0.50
        clav_y = a.torso_front_y * 0.85
        clav_angle_z = side_sign * math.radians(-12)   # angle outward

        clav, _ = create_plane_ridge(
            f"{name}_Clavicle_{side}",
            location=(clav_mid_x, clav_y, a.clavicle_z),
            rotation_euler=(0, 0, clav_angle_z),
            length=clav_len,
            width=0.022,
            height=0.014 * structural_scale,
            segments=5,
        )
        results[f"clavicle_{side.lower()}"] = clav

    # 4) TRAPEZIUS — upper back into neck
    trap, _ = create_plane_ridge(
        f"{name}_Trapezius",
        location=(0, a.torso_back_y + 0.025, a.clavicle_z - 0.015),
        rotation_euler=(math.radians(-25), 0, 0),
        length=a.shoulder_w * 1.10,
        width=0.050,
        height=0.022 * structural_scale,
        segments=6,
    )
    results["trapezius"] = trap

    # 5) SCAPULAE — paired floating plates on upper back
    for side, side_sign in (("L", 1), ("R", -1)):
        scap_x = side_sign * a.shoulder_half_x * 0.55
        scap, _ = create_box_rounded(
            f"{name}_Scapula_{side}",
            location=(scap_x, a.torso_back_y + 0.005, a.chest_z + 0.020),
            rotation_euler=(0, math.radians(side_sign * -8), 0),
            size_x=a.shoulder_w * 0.20,
            size_y=0.018,
            size_z=a.torso_height * 0.22,
            bevel=0.005, bevel_segments=1,
        )
        results[f"scapula_{side.lower()}"] = scap

    # Parent everything under root
    for k, obj in list(results.items()):
        if obj is root: continue
        _reparent(obj, root)

    return results


# ============================================================
# SUBSYSTEM 2 — CHEST
# Breasts (F) or pec slabs (M) — soft tissue, independent archetype
# ============================================================

def assemble_chest(name, anchors, gender="female", size_archetype="average"):
    """Build chest (breasts for F, pec slabs for M).

    size_archetype: see CHEST_ARCHETYPES_F or _M tables.
    """
    a = anchors
    root = _make_root_anchor(name, (0, a.torso_front_y * 0.5, a.chest_z))
    results = {"root": root}

    if gender == "female":
        size_scale = CHEST_ARCHETYPES_F.get(size_archetype,
                                              CHEST_ARCHETYPES_F["average"])

        # Per construction guide:
        # Breast Width = 0.18-0.34 Shoulder Width
        # Projection = 0.12-0.30 Torso Depth
        # Separation = 0.04-0.14 Chest Width
        breast_width = a.shoulder_w * (0.22 + size_scale * 0.08) * 0.5
        breast_proj  = a.chest_depth * (0.20 + size_scale * 0.10)
        breast_z_drop = max(0, size_scale - 1.0) * a.torso_height * 0.04   # gravity sag

        # Position breasts: separated horizontally, projected forward
        breast_x = a.shoulder_half_x * 0.32
        breast_y = a.torso_front_y + breast_proj * 0.4
        breast_z = a.chest_z - breast_z_drop

        for side, side_sign in (("L", 1), ("R", -1)):
            br, _ = create_flatsphere(
                f"{name}_Breast_{side}",
                location=(side_sign * breast_x, breast_y, breast_z),
                rotation_euler=(math.radians(10), 0, 0),  # downward droop
                radius_x=breast_width,
                radius_y=breast_proj * 0.55,
                radius_z=breast_width * 0.95,
                segments=14, rings=10,
                taper_z=0.20 - size_scale * 0.05,  # larger breasts droop more
            )
            results[f"breast_{side.lower()}"] = br

    else:  # male
        size_scale = CHEST_ARCHETYPES_M.get(size_archetype,
                                              CHEST_ARCHETYPES_M["average"])

        # Male: pec slabs — flatter, wider, less projection
        # Construction guide: Chest Width 0.82-0.96 SW, Projection 0.18-0.30 SW
        pec_w = a.shoulder_w * 0.42 * size_scale
        pec_d = a.shoulder_w * 0.20 * size_scale
        pec_h = a.torso_height * 0.16

        pec_x = a.shoulder_half_x * 0.45
        pec_y = a.torso_front_y - 0.005
        pec_z = a.chest_z + 0.020

        for side, side_sign in (("L", 1), ("R", -1)):
            pec, _ = create_flatsphere(
                f"{name}_Pec_{side}",
                location=(side_sign * pec_x, pec_y, pec_z),
                radius_x=pec_w * 0.50,
                radius_y=pec_d * 0.5,
                radius_z=pec_h * 0.50,
                segments=10, rings=6,
                taper_z=-0.25,  # flatter bottom (lower pec edge)
            )
            results[f"pec_{side.lower()}"] = pec

    for k, obj in list(results.items()):
        if obj is root: continue
        _reparent(obj, root)

    return results


# ============================================================
# SUBSYSTEM 3 — ABDOMEN
# Rectus abdominis + obliques + serratus
# Visibility scales with low body fat (NOT muscle size!)
# ============================================================

def assemble_abdomen(name, anchors, lean_archetype="average"):
    """Build abdominal definition.

    The doc emphasizes: visible abs = LOW BODY FAT, not "huge muscles".
    lean_archetype controls visibility, not muscle mass.
    """
    a = anchors
    lean_scale = AB_ARCHETYPES.get(lean_archetype, AB_ARCHETYPES["average"])

    if lean_scale < 0.05:
        return {"root": None}  # no abs visible

    root = _make_root_anchor(name, (0, a.torso_front_y * 0.7,
                                      (a.navel_z + a.rib_bottom_z)/2))
    results = {"root": root}

    # 1) RECTUS ABDOMINIS — central front ridge
    # Per construction guide: Abdomen Width 0.62-0.76 Rib Width, depth 0.08-0.16 Torso Depth
    # Position: at the FRONT of the body, projecting forward past the waist surface
    rectus_w = a.rib_half_x * 0.55 * 2     # narrow vertical strap, not wide
    rectus_h = (a.rib_bottom_z - a.navel_z) * 1.6
    rectus_d = a.abdomen_depth * (0.18 + lean_scale * 0.16)  # depth grows with leanness
    rectus_z = (a.navel_z + a.rib_bottom_z) / 2

    # Place at the body's front surface — protrudes forward by half the depth
    rectus, _ = create_flatsphere(
        f"{name}_RectusAbdominis",
        location=(0, a.torso_front_y + rectus_d * 0.45, rectus_z),
        radius_x=rectus_w * 0.50,    # X = side-to-side width of the ab strap
        radius_y=rectus_d,           # Y = how much it protrudes forward
        radius_z=rectus_h * 0.50,    # Z = vertical height (tall strip)
        segments=10, rings=8,
        taper_z=0.05,
    )
    results["rectus_abdominis"] = rectus

    # 2) OBLIQUES — side ridges (only at moderate+ leanness)
    # Position: at the WAIST SIDES, oriented VERTICALLY, tucked against the body
    if lean_scale >= 0.4:
        oblique_w = a.waist_half_x * 0.20 * lean_scale   # narrow width (front-back)
        oblique_d = a.abdomen_depth * 0.15 * lean_scale  # subtle protrusion
        oblique_h = (a.rib_bottom_z - a.waist_z) * 1.20  # tall vertical
        # X position: at the lateral waist surface
        oblique_x = a.waist_half_x * 0.95
        oblique_z = (a.rib_bottom_z + a.waist_z) / 2

        for side, side_sign in (("L", 1), ("R", -1)):
            obq, _ = create_flatsphere(
                f"{name}_Oblique_{side}",
                location=(side_sign * oblique_x, a.torso_center_y + a.abdomen_depth * 0.15, oblique_z),
                rotation_euler=(0, 0, 0),  # No rotation — stay vertical
                radius_x=oblique_w,   # narrow side-to-side (just a side ridge)
                radius_y=oblique_d * 1.5,   # slight forward bulge
                radius_z=oblique_h * 0.5,   # tall vertical ridge
                segments=8, rings=6,
            )
            results[f"oblique_{side.lower()}"] = obq

    # 3) SERRATUS — rib lines visible only on very lean characters
    # Position: at the side of the lower rib cage, angled DIAGONALLY DOWNWARD-FORWARD
    if lean_scale >= 0.7:
        serratus_w = a.rib_half_x * 0.40
        serratus_d = 0.005 * lean_scale
        serratus_x = a.rib_half_x * 0.85
        serratus_z = a.rib_bottom_z + 0.020

        for side, side_sign in (("L", 1), ("R", -1)):
            # Diagonal ridges sweep from upper-back to lower-front (serratus pattern)
            ser, _ = create_plane_ridge(
                f"{name}_Serratus_{side}",
                location=(side_sign * serratus_x,
                          a.torso_front_y - a.abdomen_depth * 0.25,
                          serratus_z),
                # Tilt: Z-rotation diagonal direction, Y-tilt slight wrap
                rotation_euler=(0, math.radians(side_sign * -8), math.radians(side_sign * 40)),
                length=serratus_w,
                width=0.012,
                height=serratus_d,
                segments=5,
            )
            results[f"serratus_{side.lower()}"] = ser

    for k, obj in list(results.items()):
        if obj is root or obj is None: continue
        _reparent(obj, root)

    return results


# ============================================================
# SUBSYSTEM 4 — BACK MUSCLES
# Lats (V-taper) — defines back silhouette width
# ============================================================

def assemble_back_muscles(name, anchors, lat_archetype="average"):
    """Build lats (latissimus dorsi V-taper)."""
    a = anchors
    lat_scale = BACK_ARCHETYPES.get(lat_archetype, BACK_ARCHETYPES["average"])

    root = _make_root_anchor(name, (0, a.torso_back_y + 0.020,
                                      (a.chest_z + a.waist_z)/2))
    results = {"root": root}

    # Lat width 0.18-0.34 Shoulder Width per construction guide
    lat_w = a.shoulder_w * (0.20 + lat_scale * 0.07) * 0.5
    lat_h = (a.chest_z - a.waist_z) * 0.85
    lat_d = 0.020 + lat_scale * 0.018
    lat_x = a.shoulder_half_x * 0.75
    lat_y = a.torso_back_y + lat_d * 0.4
    lat_z = (a.chest_z + a.waist_z) / 2

    for side, side_sign in (("L", 1), ("R", -1)):
        lat, _ = create_box_rounded(
            f"{name}_Lat_{side}",
            location=(side_sign * lat_x, lat_y, lat_z),
            rotation_euler=(0, math.radians(side_sign * -10), 0),
            size_x=lat_w,
            size_y=lat_d,
            size_z=lat_h,
            bevel=0.008, bevel_segments=1,
        )
        results[f"lat_{side.lower()}"] = lat

    for k, obj in list(results.items()):
        if obj is root: continue
        _reparent(obj, root)

    return results


# ============================================================
# TOP-LEVEL ORCHESTRATOR
# ============================================================

def assemble_torso_for_iter22b(gender="female",
                                 master_archetype="f_average",
                                 structural="average",
                                 chest="average",
                                 abdomen="average",
                                 back="average"):
    """Build all 4 torso subsystems for the iter_22b scene.

    Each subsystem has its own archetype string — mix and match freely.

    Examples:
        - Standard curvy: master="f_curvy",  chest="full",  abdomen="soft",  back="average"
        - Athletic F:     master="f_athletic", chest="small", abdomen="athletic", back="athletic"
        - Heroic M:       gender="male", master="m_heroic", chest="heroic",
                          abdomen="ripped", back="heroic", structural="heroic"
    """
    # Read shoulder + hip positions from limb empties
    shoulder_half_x = 0.114
    hip_top_z = 0.85
    shoulder_z = 1.321

    if "UpperArmL_P0" in bpy.data.objects and "UpperArmR_P0" in bpy.data.objects:
        ul = bpy.data.objects["UpperArmL_P0"].location
        ur = bpy.data.objects["UpperArmR_P0"].location
        shoulder_z = (ul.z + ur.z) / 2
        shoulder_half_x = abs(ul.x - ur.x) / 2

    if "ThighL_P0" in bpy.data.objects and "ThighR_P0" in bpy.data.objects:
        tl = bpy.data.objects["ThighL_P0"].location
        tr = bpy.data.objects["ThighR_P0"].location
        hip_top_z = (tl.z + tr.z) / 2

    anchors = TorsoAnchors(
        shoulder_half_x=shoulder_half_x,
        hip_top_z=hip_top_z,
        shoulder_z=shoulder_z,
        torso_center_y=0.020,
        gender_archetype=master_archetype,
    )

    print(f"Torso anchors (gender={gender}, master={master_archetype}):")
    print(f"  shoulder_w={anchors.shoulder_w*100:.1f}cm")
    print(f"  rib_w={anchors.rib_half_x*2*100:.1f}cm")
    print(f"  waist_w={anchors.waist_half_x*2*100:.1f}cm")
    print(f"  pelvis_w={anchors.pelvis_half_x*2*100:.1f}cm")
    print(f"  chest_depth={anchors.chest_depth*100:.1f}cm")
    print(f"  torso_height={anchors.torso_height*100:.1f}cm")

    s_scale = STRUCTURAL_ARCHETYPES.get(structural, 1.0)

    results = {
        "structural": assemble_torso_structural("Torso_Struct", anchors, s_scale),
        "chest":      assemble_chest("Torso_Chest", anchors, gender, chest),
        "abdomen":    assemble_abdomen("Torso_Abdomen", anchors, abdomen),
        "back":       assemble_back_muscles("Torso_Back", anchors, back),
        "anchors":    anchors,
    }
    return results


def clear_existing_torso():
    """Remove all torso primitives from the scene (for re-running)."""
    prefixes = ("Torso_", "Torso ")
    to_remove = [o for o in bpy.data.objects
                 if any(o.name.startswith(p) for p in prefixes)]
    for o in to_remove:
        bpy.data.objects.remove(o, do_unlink=True)
    print(f"Removed {len(to_remove)} existing torso objects")
