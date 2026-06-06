"""Geometry Nodes-based limb ANATOMY builder (multi-muscle, all 4 limb types).

Per the docs in `Characters/Docs/`:
    - Blender_Geometry_Nodes_Guide.md           (upper arm + forearm)
    - Procedural Human Leg Anatomy Guide.md     (thigh + calf)

Builds 4 GN node groups, one per limb segment, each with named muscle slots
matching the doc's anatomical breakdown:

    BD_LimbAnatomy_UpperArm  →  Deltoid + Bicep + Brachialis + Tricep
    BD_LimbAnatomy_Forearm   →  Brachioradialis + Extensors + Flexors
    BD_LimbAnatomy_Thigh     →  VastusLateralis + VastusMedialis + RectusFemoris + Hamstring + Glute
    BD_LimbAnatomy_Calf      →  Gastrocnemius + Soleus + TibialisAnterior

Per-muscle inputs: `<Name> Axial / Amount / Sigma / Conc` + `<Name> Archetype`.
Direction is HARDCODED per muscle (with `Limb Side` toggle for laterally-
asymmetric muscles).

The GN is the AUTHORING tool — runtime delivery is morph targets baked from
sweeping each slider 0->1 and snapshotting deltas.
"""

from __future__ import annotations
import bpy
from mathutils import Vector


def _flip_y(muscle_dict):
    """Negate the Y component of every direction tuple in a muscle dict.
    Converts +Y-forward convention (from ChatGPT-generated docs) to the
    correct -Y-forward convention used by Blender + VA basemesh.
    """
    return {name: (dx, -dy, dz, lat) for name, (dx, dy, dz, lat) in muscle_dict.items()}

# ============================================================
# MUSCLE DIRECTIONS (body world space)
# Convention: char faces -Y (FRONT = -Y), +Z up, +X = character's LEFT side.
# This matches standard 3D + the VA basemesh orientation.
# (The anatomy doc PDFs used "+Y forward" which was a transcription mistake.)
# Limb Side: Left=+1, Right=-1.
# (dir_x_factor, dir_y, dir_z, is_lateral)
#
# All Y components below are signed for -Y FORWARD:
#   anterior  (front of body) = NEGATIVE Y
#   posterior (back of body)  = POSITIVE Y
# ============================================================

UPPERARM_MUSCLES = {
    # Per Blender_Geometry_Nodes_Guide.md — mass ratio 0.60 Tri / 0.30 Bi / 0.10 Brach
    "Deltoid":   (0.5,  0.0,  1.0, True),    # outer-up shoulder cap
    "Bicep":     (0.0,  0.9,  0.4, False),   # anterior front-up
    "Brachialis":(1.0,  0.3, -0.3, True),    # outer wedge, slightly lower (pushes bicep out)
    "Tricep":    (0.0, -1.0,  0.0, False),   # straight back, dominant 60% mass
}
UPPERARM_DEFAULTS = {
    # (axial, amount, sigma, conc)
    "Deltoid":    (0.05, 0.015, 0.12, 3.0),
    "Bicep":      (0.50, 0.015, 0.18, 4.0),
    "Brachialis": (0.70, 0.008, 0.14, 4.0),
    "Tricep":     (0.55, 0.025, 0.28, 2.5),
}

FOREARM_MUSCLES = {
    # Forearm is wedge-shaped, asymmetric, mass concentrated near elbow.
    # "Brachioradialis = most visually important — diagonal outward wedge"
    # "Extensors upward/outward; Flexors inward/downward"
    "Brachiorad":(1.0,  0.4,  0.3, True),    # diagonal outer-forward-up (elbow-to-thumb bridge)
    "Extensors": (0.4,  0.0,  0.9, True),    # outer-up face (top of forearm in pose)
    "Flexors":   (-0.4, 0.0, -0.9, True),    # inner-down face (palm side)
}
FOREARM_DEFAULTS = {
    # Mass concentrated NEAR ELBOW (t≈0.2, the proximal end since t=0=elbow)
    "Brachiorad": (0.20, 0.012, 0.14, 4.0),
    "Extensors":  (0.30, 0.010, 0.18, 3.0),
    "Flexors":    (0.25, 0.010, 0.18, 3.0),
}

THIGH_MUSCLES = {
    # Per Procedural Human Leg Anatomy Guide.md — mass ratio
    # 0.45 Quads + 0.25 Glutes + 0.15 Hamstrings + 0.10 Calves + 0.05 Adductors
    # Quad = Vastus Lateralis (outer sweep) + Medialis (teardrop) + RectusFemoris (front)
    "VastusLat":  (1.0,  0.6,  0.0, True),   # outer + forward sweep (MOST important — heroic silhouette)
    "VastusMed":  (-1.0, 0.3, -0.2, True),   # inner-forward, slightly down (teardrop near knee)
    "RectusFem":  (0.0,  1.0,  0.0, False),  # front center ridge
    "Hamstring":  (-0.3,-1.0,  0.0, True),   # back, slightly inner, "flow into glutes"
    "Glute":      (0.4, -1.0,  0.3, True),   # back+outer+slightly up; "DO NOT scale down"
}
THIGH_DEFAULTS = {
    # t=0 hip, t=1 knee
    "VastusLat":  (0.55, 0.025, 0.25, 3.0),
    "VastusMed":  (0.80, 0.012, 0.14, 5.0),   # teardrop near knee = tight + low
    "RectusFem":  (0.50, 0.015, 0.22, 3.0),
    "Hamstring":  (0.55, 0.022, 0.28, 2.5),
    "Glute":      (0.05, 0.025, 0.16, 3.0),   # hip end of thigh primitive
}

CALF_MUSCLES = {
    # Per leg doc: Gastroc back+outer+upward (mass sits HIGHER), Soleus lower-back,
    # Tibialis Anterior front shin wedge (subtle — over-scaling = cartoon)
    "Gastroc":     (0.4, -1.0,  0.0, True),   # back+outer
    "Soleus":      (0.0, -1.0,  0.0, False),  # lower back mass
    "TibialisAnt": (0.3,  1.0,  0.0, True),   # front shin wedge, slightly outer
}
CALF_DEFAULTS = {
    # t=0 knee, t=1 ankle. "Most mass should sit HIGHER" = small t
    "Gastroc":     (0.25, 0.022, 0.16, 3.0),   # belly upper third, prominent
    "Soleus":      (0.55, 0.012, 0.22, 3.0),   # lower back mass
    "TibialisAnt": (0.45, 0.005, 0.18, 6.0),   # subtle, tight
}

# ============================================================
# ARCHETYPE TABLES (from anatomy docs)
# ============================================================

UPPERARM_ARCHETYPES = {
    "average":     {"Deltoid":1.0, "Bicep":1.0, "Brachialis":1.0, "Tricep":1.4},
    "athlete":     {"Deltoid":1.5, "Bicep":1.2, "Brachialis":1.2, "Tricep":1.6},
    "bodybuilder": {"Deltoid":2.5, "Bicep":1.8, "Brachialis":1.6, "Tricep":2.2},
    "anime":       {"Deltoid":3.0, "Bicep":2.0, "Brachialis":1.4, "Tricep":1.6},
}

FOREARM_ARCHETYPES = {
    # Doc doesn't break out forearm archetype; derive from upper-arm proportionally
    "average":     {"Brachiorad":1.0, "Extensors":1.0, "Flexors":1.0},
    "athlete":     {"Brachiorad":1.3, "Extensors":1.2, "Flexors":1.2},
    "bodybuilder": {"Brachiorad":1.8, "Extensors":1.6, "Flexors":1.6},
    "anime":       {"Brachiorad":1.8, "Extensors":1.5, "Flexors":1.4},
}

THIGH_ARCHETYPES = {
    # Quad multiplier from leg doc Average=1.1, Athlete=1.5, Bodybuilder=2.0, Runner=1.2
    # Splitting "Quad" mass across VastusLat, VastusMed, RectusFem
    "average":     {"VastusLat":1.1, "VastusMed":1.1, "RectusFem":1.1, "Hamstring":1.0, "Glute":1.0},
    "athlete":     {"VastusLat":1.5, "VastusMed":1.5, "RectusFem":1.5, "Hamstring":1.3, "Glute":1.3},
    "bodybuilder": {"VastusLat":2.0, "VastusMed":2.0, "RectusFem":2.0, "Hamstring":1.7, "Glute":1.5},
    "runner":      {"VastusLat":1.2, "VastusMed":1.2, "RectusFem":1.2, "Hamstring":1.4, "Glute":1.1},
    "anime":       {"VastusLat":1.8, "VastusMed":1.8, "RectusFem":1.8, "Hamstring":1.4, "Glute":1.6},
}

CALF_ARCHETYPES = {
    # Calf multiplier from leg doc Average=1.0, Athlete=1.2, Bodybuilder=1.6, Runner=1.5
    "average":     {"Gastroc":1.0, "Soleus":1.0, "TibialisAnt":1.0},
    "athlete":     {"Gastroc":1.2, "Soleus":1.2, "TibialisAnt":1.1},
    "bodybuilder": {"Gastroc":1.6, "Soleus":1.6, "TibialisAnt":1.2},
    "runner":      {"Gastroc":1.5, "Soleus":1.4, "TibialisAnt":1.2},
    "anime":       {"Gastroc":1.7, "Soleus":1.4, "TibialisAnt":1.1},
}


# ============================================================
# TORSO — continuous mesh, body trunk
# Axis: waist (P0, low Z) → clavicle (P1, high Z)
# L/R muscles use explicit name suffix + fixed dx_fac (is_lateral=False)
# Sagittal muscles centered (dx_fac=0)
# ============================================================

TORSO_MUSCLES = {
    # Trunk shell with INTEGRATED chest (Pec/Breast = same architecture as Abs).
    # All muscle peaks use normal-mode displacement → round outward bulges.
    # Convention: char faces -Y front (anterior). Y values written for that frame.

    # ---- FRONT CENTERLINE GROOVES (sagittal, negative amounts) ----
    "Sternum":      (0.0,   1.0,  0.0, False),  # upper centerline groove (chest)
    "LineaAlba":    (0.0,   1.0,  0.0, False),  # lower centerline groove (abs)
    "InframamFold": (0.0,   1.0,  0.0, False),  # HORIZONTAL line — under breast / chest-to-abs divider

    # Linea semilunaris — vertical groove down the OUTER edge of each ab column,
    # framing the ab panel left + right. Nice definition touch, kept regardless
    # of how soft/firm the abs themselves are.
    "SemilunarisL": (1.0,   0.5,  0.0, False),
    "SemilunarisR": (-1.0,  0.5,  0.0, False),

    # ---- BACK MUSCLES ----
    "Trapezius":    (0.0,  -0.5,  1.0, False),
    "Lat_L":     (1.0,  -1.0, -0.2, False),
    "Lat_R":    (-1.0,  -1.0, -0.2, False),
    # Scapulae (shoulderblades) — upper-back, paired, slightly lateral.
    "Scapula_L": (0.7,  -1.0,  0.6, False),
    "Scapula_R":(-0.7,  -1.0,  0.6, False),
    # Spinal furrow — VERTICAL groove down the back centerline (negative amount).
    # The back's answer to LineaAlba; runs between the shoulderblades and down.
    "SpineGroove":  (0.0,  -1.0,  0.0, False),
    # Lumbar curve — broad lower-back hollow (lordosis). Negative amount pulls
    # the lower back IN, so the small-of-the-back curves inward like the VA mesh.
    "LumbarCurve":  (0.0,  -1.0,  0.0, False),

    # ---- BELLY (forward bulge for pregnancy/pot-belly/concave-lean) ----
    "Belly":        (0.0,   1.0, -0.3, False),

    # ---- PEC/BREAST PAIR (chest mass) ----
    # Direction is X-DOMINANT (0.75 lateral × 0.6 forward) so peaks land on the
    # LATERAL-front face, NOT the center. This is what makes the Sternum groove
    # actually read between the breasts — center vert isn't pushed by pec.
    "Pec_L":  (0.75,  0.6, -0.1, False),
    "Pec_R": (-0.75,  0.6, -0.1, False),

    # ---- AB GRID: 2 columns × 4 rows = 8 segments ----
    # X-DOMINANT direction (same fix as Pec): each ab peaks on its own lateral-front
    # face, NOT the center — so the LineaAlba groove wins the centerline and the
    # abs read as two distinct columns instead of merging into horizontal bands.
    "Ab1L":  (0.65,  0.6,  0.0, False),
    "Ab1R": (-0.65,  0.6,  0.0, False),
    "Ab2L":  (0.65,  0.6,  0.0, False),
    "Ab2R": (-0.65,  0.6,  0.0, False),
    "Ab3L":  (0.65,  0.6,  0.0, False),
    "Ab3R": (-0.65,  0.6,  0.0, False),
    "Ab4L":  (0.65,  0.6,  0.0, False),
    "Ab4R": (-0.65,  0.6,  0.0, False),

    # ---- SHOULDER SOCKETS (arm attachment landmarks) ----
    "ShoulderSocket_L": (1.0,   0.0,  0.5, False),
    "ShoulderSocket_R": (-1.0,  0.0,  0.5, False),

    # ---- COLLAR BONES (top to neck) ----
    "Clavicle_L":(1.0,   0.3,  0.0, False),
    "Clavicle_R":(-1.0,  0.3,  0.0, False),
}
TORSO_DEFAULTS = {
    # ---- Grooves (front centerline) ----
    # Sternum + LineaAlba together span clavicle→navel as ONE continuous groove
    # Sternum: SHARP cleavage groove. Same recipe as LineaAlba (high conc, narrow angular cut)
    #   so it can stay a knife-edge even when Pecs sit close on either side.
    #   amount=-45mm = deep groove; conc=48 = sharp angular cut, won't fight Pec at lateral verts.
    #   sigma_axial=0.10 = stays within chest band (clavicle to inframam), doesn't bleed into abs.
    "Sternum":      (0.72, -0.045, 0.10, 48.0),  # axial 0.72 follows breast peak
    "LineaAlba":    (0.35, -0.012, 0.20, 48.0),  # ab area, very narrow line (drastic falloff)
    "InframamFold": (0.60, -0.012, 0.025, 3.0),  # axial 0.60 = on the 0.05 grid
    # Linea semilunaris — vertical groove (tall axial sigma) at the ab outer edge.
    # conc=40 = defined line that sits just outside the ab peak without eating it.
    "SemilunarisL": (0.35, -0.010, 0.20, 40.0),
    "SemilunarisR": (0.35, -0.010, 0.20, 40.0),
    # ---- Back ----
    # Lat axial 0.52 = mid-UPPER back (flares from under the arms), sigma 0.14 so
    # it stays clear of the waist-pinch zone (~t 0.20) — otherwise a pinched
    # hourglass still bulges out behind. Lat size is a build trait, not silhouette.
    "Trapezius":    (0.85, 0.018, 0.10, 5.0),
    "Lat_L":     (0.52, 0.030, 0.14, 3.0),
    "Lat_R":     (0.52, 0.030, 0.14, 3.0),
    "Scapula_L": (0.72, 0.012, 0.10, 4.0),  # shoulderblade — upper-back broad bump
    "Scapula_R": (0.72, 0.012, 0.10, 4.0),
    # Spinal furrow — vertical back groove. Tall axial sigma (runs the back),
    # conc 40 = tight centerline line, amount -14mm = visible indent.
    "SpineGroove":  (0.55, -0.014, 0.28, 40.0),
    # Lumbar hollow — low-back, broad (conc 3 wraps the lower back), -16mm inward.
    "LumbarCurve":  (0.16, -0.016, 0.09, 3.0),
    # ---- Belly ----
    "Belly":        (0.20,  0.020, 0.12, 2.0),
    # ---- Pec/Breast (CHEST PAIR) ----
    # Axial 0.72 = breast peak, sitting just above the inframam fold (0.60) so the
    # breast overhangs the fold. Sigma=0.13 = fuller, rounder bulge (less peaky
    # blob). Conc=12 = round breast without bleeding into the Sternum groove —
    # the front-density warp gives the centerline its own verts so conc can relax.
    "Pec_L":  (0.72, 0.022, 0.13, 12.0),
    "Pec_R":  (0.72, 0.022, 0.13, 12.0),
    # ---- Ab grid: 8 peaks (4 rows × 2 cols), one Ab Definition slider ----
    # 5-tuple: (axial, amount, sigma, conc, FLAT). The base AMOUNT carries the
    # built-in top→bottom gradient — Ab1 strongest, Ab4 faintest (a 4th row only
    # shows on the very fit). One master "Ab Definition" socket scales all 8.
    # Flat=0.5 = gentle flat-top (between soft blob and hard cube); dial per taste.
    "Ab1L":  (0.50, 0.014, 0.032, 5.0, 0.5),
    "Ab1R":  (0.50, 0.014, 0.032, 5.0, 0.5),
    "Ab2L":  (0.40, 0.011, 0.032, 5.0, 0.5),
    "Ab2R":  (0.40, 0.011, 0.032, 5.0, 0.5),
    "Ab3L":  (0.30, 0.008, 0.032, 5.0, 0.5),
    "Ab3R":  (0.30, 0.008, 0.032, 5.0, 0.5),
    "Ab4L":  (0.20, 0.005, 0.032, 5.0, 0.5),
    "Ab4R":  (0.20, 0.005, 0.032, 5.0, 0.5),
    # ---- Shoulder sockets ----
    "ShoulderSocket_L": (0.95, 0.010, 0.06, 4.0),
    "ShoulderSocket_R": (0.95, 0.010, 0.06, 4.0),
    # ---- Clavicle ----
    "Clavicle_L":(0.90, 0.007, 0.05, 8.0),
    "Clavicle_R":(0.90, 0.007, 0.05, 8.0),
}


# ============================================================
# BREAST — separate teardrop object, attaches at chest plane
# Axis: P0 (root, at chest plane) → P1 (nipple, forward+down)
# No muscle peaks needed; shape comes from radius taper.
# Depth Ratio controls oval cross-section (flatter side-to-side than top-bottom).
# ============================================================

BREAST_MUSCLES = {
    # Fullness peak — single forward bulge at middle of breast axis.
    # Direction = forward (+Y) in breast-local space; since the breast axis is
    # P0(root) → P1(nipple, forward+down), the +Y of the world maps to the
    # main projection direction.
    "Fullness": (0.0, 1.0, 0.0, False),
    # UpperPole — softer roundness at proximal half (closer to chest plane)
    "UpperPole": (0.0, 1.0, 0.3, False),
}
BREAST_DEFAULTS = {
    # (axial, amount_m, sigma, conc)
    "Fullness":  (0.55, 0.025, 0.30, 2.0),   # broad bulge, low conc = round
    "UpperPole": (0.30, 0.010, 0.25, 2.0),   # softer upper-pole roundness
}

BREAST_ARCHETYPES = {
    "athletic":  {"Fullness": 0.7, "UpperPole": 0.6},
    "average":   {"Fullness": 1.0, "UpperPole": 1.0},
    "curvy":     {"Fullness": 1.5, "UpperPole": 1.2},
    "stylized":  {"Fullness": 1.8, "UpperPole": 1.5},
}

# Per-archetype size scalars (applied to start_radius/end_radius at creation)
BREAST_SIZE_SCALARS = {
    "athletic": 0.75,
    "average":  1.00,
    "curvy":    1.40,
    "stylized": 1.65,
    "large":    1.85,
}

# ============================================================
# PEC SLAB (Male) — separate flat slab, attaches at chest plane
# Same architecture as breast but flatter + wider.
# ============================================================

PEC_MUSCLES = {}
PEC_DEFAULTS = {}
PEC_ARCHETYPES = {"average":{}, "athletic":{}, "bodybuilder":{}, "heroic":{}}
PEC_SIZE_SCALARS = {
    "average": 1.00, "athletic": 1.30, "bodybuilder": 1.80, "heroic": 2.10,
}

# MALE archetypes: Pec active, Breast = 0 (disabled via archetype scalar)
# FEMALE archetypes: Breast active, Pec = 0
# Per doc's Archetype Master Table.

def _torso_archetype(lat, trap, sternum=0.6, clavicle=1.0, shoulder=1.0,
                      belly=1.0, abs_def=0.0, linea_alba=0.0, pec=1.0,
                      inframam=0.5, chest_spacing=1.0):
    """Build a torso archetype.

    Args:
        pec: CHEST mass scalar. M average=1.0, M athletic=1.5, M bodybuilder=2.5.
              F average=2.0 (breast), F curvy=3.0+, F athletic=1.3 (small firm).
        inframam: visibility of horizontal underbreast/pec-bottom crease.
                  Scales with chest size — bigger breast = deeper fold.
        abs_def: visibility of 8 ab peaks
        linea_alba: visibility of vertical center groove between abs
        belly: forward bulge (negative = concave/sunken stomach)
        chest_spacing: lateral pec/breast separation. <1 = closer (deeper
                  cleavage), >1 = wider apart. Maps to the Chest Spacing socket.
                  Keys prefixed with "_" route to a named socket, not "{x} Archetype".
    """
    return {
        "_Chest Spacing":chest_spacing,
        # One master Ab Definition slider drives all 8 ab peaks + semilunaris
        # (semilunaris Archetype stays 1.0; the Ab Definition multiply scales it).
        # Per-ab Archetype sockets stay at 1.0 for optional fine-tuning.
        "_Ab Definition":abs_def,
        "Sternum":sternum, "LineaAlba":linea_alba, "InframamFold":inframam,
        "Trapezius":trap,
        "Belly":belly,
        # Breast/pec mass → master Chest Size slider (Pec Archetype stays 1.0).
        "_Chest Size":pec,
        "Lat_L":lat, "Lat_R":lat,
        "Scapula_L":trap, "Scapula_R":trap,  # shoulderblades track upper-back dev
        "SpineGroove":1.0,                   # spinal furrow — structural, always on
        "LumbarCurve":1.0,                   # lower-back lordosis — structural
        "ShoulderSocket_L":shoulder, "ShoulderSocket_R":shoulder,
        "Clavicle_L":clavicle, "Clavicle_R":clavicle,
    }

TORSO_ARCHETYPES_F = {
    # pec=BREAST size, sternum=CLEAVAGE depth (scales with bust + posture)
    # sternum 0=no separation; 1=visible channel; 2=deep cleavage; 3+=push-up/dramatic
    # chest_spacing: <1 pushes breasts together (cleavage), >1 spreads them apart.
    "super_lean":_torso_archetype(lat=0.6, trap=0.5, sternum=0.4, belly=-0.8, abs_def=1.6, linea_alba=1.4, pec=0.8, inframam=0.4, chest_spacing=1.15),
    "athletic":  _torso_archetype(lat=0.9, trap=0.7, sternum=0.6, clavicle=1.2, belly=-0.3, abs_def=1.3, linea_alba=1.2, pec=1.2, inframam=0.6, chest_spacing=1.10),
    "fit":       _torso_archetype(lat=0.8, trap=0.6, sternum=0.8, belly=0.2, abs_def=0.7, linea_alba=0.8, pec=1.6, inframam=0.7, chest_spacing=1.05),
    "average":   _torso_archetype(lat=0.7, trap=0.5, sternum=1.0, belly=1.0, abs_def=0.0, pec=2.0, inframam=0.8, chest_spacing=1.00),
    "curvy":     _torso_archetype(lat=0.6, trap=0.4, sternum=1.6, clavicle=0.8, belly=0.7, abs_def=0.0, pec=3.0, inframam=1.4, chest_spacing=0.90),
    "stylized":  _torso_archetype(lat=0.6, trap=0.4, sternum=2.0, clavicle=0.7, belly=0.5, abs_def=0.0, pec=3.5, inframam=1.6, chest_spacing=0.85),
    "pregnant":  _torso_archetype(lat=0.6, trap=0.5, sternum=1.2, clavicle=1.0, belly=2.5, abs_def=0.0, pec=2.5, inframam=1.2, chest_spacing=1.00),
    "heavyset":  _torso_archetype(lat=0.5, trap=0.4, sternum=1.0, clavicle=0.5, belly=1.8, abs_def=0.0, pec=2.4, inframam=1.0, chest_spacing=0.90),
    "pushup_bra":_torso_archetype(lat=0.6, trap=0.4, sternum=3.0, clavicle=0.9, belly=0.5, abs_def=0.0, pec=3.5, inframam=1.8, chest_spacing=0.70),  # dramatic deep cleavage
    "no_cleavage":_torso_archetype(lat=0.7, trap=0.5, sternum=0.0, belly=1.0, abs_def=0.0, pec=2.0, inframam=0.5, chest_spacing=1.30),  # flat between, no channel
}

TORSO_ARCHETYPES_M = {
    # pec=CHEST mass, inframam=lower pec edge crease (visible on athletic/bodybuilder)
    # chest_spacing: pec separation — bigger chest sits closer to the centerline.
    "super_lean":  _torso_archetype(lat=0.9, trap=0.9, sternum=1.1, belly=-1.0, abs_def=2.0, linea_alba=1.7, pec=0.8, inframam=0.8, chest_spacing=1.20),
    "athletic":    _torso_archetype(lat=1.4, trap=1.2, sternum=1.2, belly=-0.2, abs_def=1.4, linea_alba=1.3, pec=1.5, inframam=1.2, chest_spacing=1.10),
    "average":     _torso_archetype(lat=1.0, trap=1.0, sternum=1.0, belly=0.8, abs_def=0.3, pec=1.0, inframam=0.5, chest_spacing=1.00),
    "heroic":      _torso_archetype(lat=1.7, trap=1.5, sternum=1.3, belly=0.0, abs_def=1.7, linea_alba=1.5, shoulder=1.3, pec=2.0, inframam=1.4, chest_spacing=1.00),
    "bodybuilder": _torso_archetype(lat=2.0, trap=1.8, sternum=1.4, belly=-0.1, abs_def=2.0, linea_alba=1.7, shoulder=1.5, pec=2.5, inframam=1.5, chest_spacing=0.95),
    "pot_belly":   _torso_archetype(lat=0.7, trap=0.8, sternum=0.4, belly=1.6, abs_def=0.0, pec=1.2, inframam=0.3, chest_spacing=1.10),
}

# ============================================================
# TORSO BODY-SHAPE PRESETS — the silhouette spectrum (width sockets).
# Orthogonal to the muscle archetype: pick a body-TYPE archetype (athletic,
# curvy, heavyset…) AND a body-SHAPE preset (V, hourglass, round…) independently.
# Keys are GN socket names; "Belly Archetype" drives the belly muscle scalar.
# ============================================================

# Mid-range presets — sensible STARTING points, not extremes. The sockets still
# allow the full range (Waist 0.4-1.6) by hand; these just seed a believable look.
TORSO_SHAPE_PRESETS = {
    "v_taper":      {"Shoulder Width":1.22, "Waist":0.86, "Ribcage":1.06, "Belly Archetype":-0.3},
    "hourglass":    {"Shoulder Width":1.08, "Waist":0.80, "Ribcage":1.03, "Belly Archetype": 0.0},
    "sunken":       {"Shoulder Width":0.94, "Waist":0.78, "Ribcage":0.92, "Belly Archetype":-0.6},
    "square":       {"Shoulder Width":1.00, "Waist":1.00, "Ribcage":1.00, "Belly Archetype": 0.3},
    "pudgy":        {"Shoulder Width":1.00, "Waist":1.12, "Ribcage":1.00, "Belly Archetype": 1.0},
    "round":        {"Shoulder Width":1.04, "Waist":1.25, "Ribcage":1.04, "Belly Archetype": 1.6},
    "double_round": {"Shoulder Width":1.10, "Waist":1.45, "Ribcage":1.08, "Belly Archetype": 2.3},
}


# ============================================================
# PELVIS — continuous mesh, hip/waist region
# Axis: hip (P0, low Z) → waist (P1, high Z)
# Wider profile than torso; glute mass on back
# ============================================================

PELVIS_MUSCLES = {
    "ASIS_L":      (0.8,  1.0,  0.3, False),  # left anterior-superior iliac spine (front-out bump)
    "ASIS_R":     (-0.8,  1.0,  0.3, False),
    "IliacCrest_L":(1.0,  0.0,  0.4, False),  # left iliac crest upper-side ridge
    "IliacCrest_R":(-1.0, 0.0,  0.4, False),
    # Glutes — X-DOMINANT (like Pec/Ab) so each cheek peaks off-centre, leaving
    # the back centerline free for the GluteCleft groove.
    "Glute_L":     (0.85, -0.7,  0.0, False),  # left glute mass — back+outer
    "Glute_R":    (-0.85, -0.7,  0.0, False),
    "Mons":        (0.0,  1.0, -0.4, False),  # front center, lower (pubic mound)
    "SacrumTri":   (0.0, -1.0,  0.4, False),  # upper back center triangle
    # Gluteal cleft — VERTICAL back-centerline groove between the cheeks.
    "GluteCleft":  (0.0,  -1.0,  0.0, False),
    # PelvisFront — broad anterior recede. The pelvis is NOT symmetric front-to-
    # back (anatomy guide): the rear carries the glute mass, the front (pubic /
    # lower-belly) stays flat. Negative amount tucks the whole front face in.
    "PelvisFront": (0.0,   1.0,  0.0, False),
}
PELVIS_DEFAULTS = {
    "ASIS_L":      (0.85, 0.012, 0.06, 7.0),  # tight point landmark
    "ASIS_R":      (0.85, 0.012, 0.06, 7.0),
    "IliacCrest_L":(0.80, 0.015, 0.10, 5.0),
    "IliacCrest_R":(0.80, 0.015, 0.10, 5.0),
    "Glute_L":     (0.42, 0.030, 0.22, 5.0),  # broad mass, lifted (Glute Height shifts it)
    "Glute_R":     (0.42, 0.030, 0.22, 5.0),
    "Mons":        (0.10, 0.010, 0.12, 5.0),
    "SacrumTri":   (0.80, 0.010, 0.10, 6.0),
    # Gluteal cleft — sharp vertical groove. conc 36 = sharp angle (lower for a
    # softer gap); Flat socket → flat-bottomed vs round gap; Amount → gap depth.
    "GluteCleft":  (0.30, -0.022, 0.26, 36.0),
    # PelvisFront — broad anterior recede (conc 2.5 = whole front face), -45mm
    # tuck so the front sits flat/subtle, well short of the rear glute mass.
    "PelvisFront": (0.38, -0.045, 0.32, 2.5),
}

# Glute mass → master "_Glute Size" socket; "_Hip Width" sets the hip-flare radius.
# GluteCleft scales with glute mass — a bigger butt has a deeper cleft.
PELVIS_ARCHETYPES_F = {
    "average":  {"ASIS_L":1.0, "ASIS_R":1.0, "IliacCrest_L":1.0, "IliacCrest_R":1.0,
                  "_Glute Size":1.0, "Mons":1.0, "SacrumTri":1.0, "_Hip Width":1.0, "GluteCleft":1.0},
    "curvy":    {"ASIS_L":0.8, "ASIS_R":0.8, "IliacCrest_L":0.9, "IliacCrest_R":0.9,
                  "_Glute Size":1.6, "Mons":1.0, "SacrumTri":0.9, "_Hip Width":1.25, "GluteCleft":1.4},
    "athletic": {"ASIS_L":1.2, "ASIS_R":1.2, "IliacCrest_L":1.2, "IliacCrest_R":1.2,
                  "_Glute Size":1.3, "Mons":0.9, "SacrumTri":1.1, "_Hip Width":1.05, "GluteCleft":1.2},
    "lean":     {"ASIS_L":1.4, "ASIS_R":1.4, "IliacCrest_L":1.4, "IliacCrest_R":1.4,
                  "_Glute Size":0.8, "Mons":0.8, "SacrumTri":1.3, "_Hip Width":0.85, "GluteCleft":0.8},
}
PELVIS_ARCHETYPES_M = {
    "average":  {"ASIS_L":1.0, "ASIS_R":1.0, "IliacCrest_L":1.0, "IliacCrest_R":1.0,
                  "_Glute Size":0.8, "Mons":0.7, "SacrumTri":1.0, "_Hip Width":0.9, "GluteCleft":1.0},
    "athletic": {"ASIS_L":1.2, "ASIS_R":1.2, "IliacCrest_L":1.2, "IliacCrest_R":1.2,
                  "_Glute Size":1.2, "Mons":0.7, "SacrumTri":1.1, "_Hip Width":0.95, "GluteCleft":1.1},
    "heroic":   {"ASIS_L":1.3, "ASIS_R":1.3, "IliacCrest_L":1.3, "IliacCrest_R":1.3,
                  "_Glute Size":1.4, "Mons":0.7, "SacrumTri":1.2, "_Hip Width":1.0, "GluteCleft":1.2},
}


# ============================================================
# JOINTS — short continuous capsules with BONY landmark peaks
# (high conc, tight sigma — point-like bumps; archetype fixed at 1.0)
# ============================================================

# KNEE — axis from thigh-end (P0) to shin-top (P1)
# Lateral landmarks use is_lateral=True so the Limb Side socket flips medial/lateral
KNEE_MUSCLES = {
    "Patella":         (0.0,  1.0,  0.0, False),  # front center kneecap (sagittal)
    "MedialCondyle":   (-1.0, 0.3,  0.2, True),   # inner-front bump (medial via Limb Side flip)
    "LateralCondyle":  (1.0,  0.3,  0.2, True),   # outer-front bump
    "TibialPlateau":   (0.0,  0.7, -0.3, False),  # below patella, front
    "PoplitealRecess": (0.0, -1.0,  0.0, False),  # negative bump = back-of-knee cavity
}
KNEE_DEFAULTS = {
    "Patella":         (0.50, 0.012, 0.10, 7.0),  # tight bump
    "MedialCondyle":   (0.55, 0.008, 0.06, 8.0),
    "LateralCondyle":  (0.55, 0.008, 0.06, 8.0),
    "TibialPlateau":   (0.25, 0.008, 0.10, 6.0),
    "PoplitealRecess": (0.50,-0.010, 0.12, 4.0),  # negative amount = inward depression
}
KNEE_ARCHETYPES = {
    "default":  {"Patella":1.0, "MedialCondyle":1.0, "LateralCondyle":1.0,
                  "TibialPlateau":1.0, "PoplitealRecess":1.0},
    "lean":     {"Patella":1.4, "MedialCondyle":1.3, "LateralCondyle":1.3,
                  "TibialPlateau":1.2, "PoplitealRecess":1.2},
    "heavy":    {"Patella":0.7, "MedialCondyle":0.6, "LateralCondyle":0.6,
                  "TibialPlateau":0.7, "PoplitealRecess":0.5},
}

# ELBOW — axis from upper-arm end (P0) to forearm-start (P1)
ELBOW_MUSCLES = {
    "Olecranon":       (0.0, -1.0,  0.0, False),  # rear point (sagittal back)
    "MedialEpicondyle":(-1.0, 0.0,  0.0, True),   # inner-side bump (Limb Side flips)
    "LateralEpicondyle":(1.0, 0.0,  0.0, True),
    "Antecubital":     (0.0,  1.0,  0.0, False),  # front recess (negative)
}
ELBOW_DEFAULTS = {
    "Olecranon":       (0.50, 0.015, 0.10, 6.0),
    "MedialEpicondyle":(0.50, 0.010, 0.06, 8.0),
    "LateralEpicondyle":(0.50, 0.010, 0.06, 8.0),
    "Antecubital":     (0.50,-0.008, 0.14, 4.0),  # negative = front recess
}
ELBOW_ARCHETYPES = {
    "default":  {"Olecranon":1.0, "MedialEpicondyle":1.0, "LateralEpicondyle":1.0, "Antecubital":1.0},
    "lean":     {"Olecranon":1.5, "MedialEpicondyle":1.4, "LateralEpicondyle":1.4, "Antecubital":1.2},
    "heavy":    {"Olecranon":0.6, "MedialEpicondyle":0.5, "LateralEpicondyle":0.5, "Antecubital":0.4},
}

# WRIST — axis from forearm-end (P0) to hand-start (P1)
# Very narrow joint, tight tendon ridges
WRIST_MUSCLES = {
    "RadiusStyloid":  (-1.0, 0.4,  0.0, True),   # inner styloid (thumb-side)
    "UlnaStyloid":     (1.0, 0.4,  0.0, True),   # outer styloid (pinky-side, slightly lower)
    "FlexorRidge":     (0.0, -1.0, 0.0, False),  # palm-side tendon ridge (sagittal back)
    "ExtensorRidge":   (0.0,  1.0, 0.0, False),  # dorsal tendon ridge (sagittal front)
}
WRIST_DEFAULTS = {
    "RadiusStyloid":   (0.55, 0.008, 0.05, 8.0),
    "UlnaStyloid":     (0.45, 0.008, 0.05, 8.0),  # slightly lower (more proximal)
    "FlexorRidge":     (0.50, 0.005, 0.12, 5.0),
    "ExtensorRidge":   (0.50, 0.005, 0.12, 5.0),
}
WRIST_ARCHETYPES = {
    "default": {"RadiusStyloid":1.0, "UlnaStyloid":1.0, "FlexorRidge":1.0, "ExtensorRidge":1.0},
    "lean":    {"RadiusStyloid":1.4, "UlnaStyloid":1.4, "FlexorRidge":1.3, "ExtensorRidge":1.3},
    "heavy":   {"RadiusStyloid":0.6, "UlnaStyloid":0.6, "FlexorRidge":0.4, "ExtensorRidge":0.4},
}

# ANKLE — axis from shin-end (P0) to foot-start (P1)
# Outer (lateral) malleolus sits LOWER than inner per doc — encoded via different axial positions
ANKLE_MUSCLES = {
    "MedialMalleolus": (-1.0, 0.3,  0.0, True),   # inner ankle bone (limb-side flipped)
    "LateralMalleolus":(1.0,  0.3,  0.0, True),   # outer ankle bone
    "AchillesTaper":   (0.0, -1.0, 0.0, False),   # back tendon ridge
    "TibialisAnt":     (0.0,  1.0, 0.0, False),   # front shin tendon
}
ANKLE_DEFAULTS = {
    "MedialMalleolus": (0.55, 0.012, 0.06, 8.0),  # inner sits HIGHER
    "LateralMalleolus":(0.42, 0.012, 0.06, 8.0),  # outer sits LOWER (per doc!)
    "AchillesTaper":   (0.55, 0.012, 0.18, 4.0),  # back vertical ridge
    "TibialisAnt":     (0.55, 0.005, 0.15, 5.0),
}
ANKLE_ARCHETYPES = {
    "default": {"MedialMalleolus":1.0, "LateralMalleolus":1.0, "AchillesTaper":1.0, "TibialisAnt":1.0},
    "lean":    {"MedialMalleolus":1.4, "LateralMalleolus":1.4, "AchillesTaper":1.5, "TibialisAnt":1.3},
    "heavy":   {"MedialMalleolus":0.6, "LateralMalleolus":0.6, "AchillesTaper":0.5, "TibialisAnt":0.4},
}


# ============================================================
# CONVENTION FLIP: convert all muscle dirs to -Y forward (standard 3D + VA)
# Doc text said "+Y forward" — that was wrong. Negate Y to fix.
# Anterior (front of body) is now NEGATIVE Y.
# ============================================================
UPPERARM_MUSCLES = _flip_y(UPPERARM_MUSCLES)
FOREARM_MUSCLES  = _flip_y(FOREARM_MUSCLES)
THIGH_MUSCLES    = _flip_y(THIGH_MUSCLES)
CALF_MUSCLES     = _flip_y(CALF_MUSCLES)
TORSO_MUSCLES    = _flip_y(TORSO_MUSCLES)
BREAST_MUSCLES   = _flip_y(BREAST_MUSCLES)
PEC_MUSCLES      = _flip_y(PEC_MUSCLES)
PELVIS_MUSCLES   = _flip_y(PELVIS_MUSCLES)
KNEE_MUSCLES     = _flip_y(KNEE_MUSCLES)
ELBOW_MUSCLES    = _flip_y(ELBOW_MUSCLES)
WRIST_MUSCLES    = _flip_y(WRIST_MUSCLES)
ANKLE_MUSCLES    = _flip_y(ANKLE_MUSCLES)


# ============================================================
# Internal node-graph builders
# ============================================================

def _new_node(tree, type_name, location=(0, 0), label=None):
    n = tree.nodes.new(type=type_name)
    n.location = location
    if label: n.label = label
    return n


def _link(tree, out_socket, in_socket):
    tree.links.new(out_socket, in_socket)


def _add_muscle_inputs(iface, name, default_axial, default_amount, default_sigma,
                       default_conc, default_flat=0.0):
    s = iface.new_socket(name=f"{name} Axial",  in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_axial; s.min_value = 0.0; s.max_value = 1.0
    s = iface.new_socket(name=f"{name} Amount", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_amount; s.min_value = -0.20; s.max_value = 0.20   # ±20cm range for extreme morphs
    s = iface.new_socket(name=f"{name} Sigma",  in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_sigma; s.min_value = 0.005; s.max_value = 0.5
    s = iface.new_socket(name=f"{name} Conc",   in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_conc; s.min_value = 1.0; s.max_value = 128.0   # hairline (high) to broad channel (low)
    # Flat: 0 = round Gaussian bulge (breasts, belly). 1 = flat-topped plateau
    # with steep walls (abs — each one reads as a cube face, not a point).
    s = iface.new_socket(name=f"{name} Flat",   in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_flat; s.min_value = 0.0; s.max_value = 1.0


def _build_muscle_displacement(ng, grp_in, muscle_name, base_dir,
                                limb_side_socket, captured_t_socket,
                                normal_attr, archetype_socket, y_off,
                                dx_mult_socket=None, flat_socket=None,
                                sag_socket=None, axial_offset_socket=None):
    """Build one muscle's displacement vector chain. Returns disp VectorMath node.

    Args:
        dx_mult_socket: optional output socket to multiply the dx_fac by at
            runtime. Used for things like "Chest Spacing" controlling how
            wide-apart the Pec_L/Pec_R peaks land.
        flat_socket: optional output socket overriding the per-muscle Flat input
            (used to drive all abs from one master "Ab Flat" socket).
        sag_socket: optional output socket — tilts the push direction downward
            (breast droop). 0 = firm, ~1 = 45° sag.
    """
    dx_fac, dy, dz, is_lateral = base_dir

    if is_lateral:
        # Limb-side flipped (use Limb Side socket × dx_fac)
        side_x = _new_node(ng, 'ShaderNodeMath', (-1400, y_off), f"{muscle_name} side*dx")
        side_x.operation = 'MULTIPLY'
        _link(ng, limb_side_socket, side_x.inputs[0])
        side_x.inputs[1].default_value = dx_fac
        x_out = side_x.outputs["Value"]
    elif dx_mult_socket is not None:
        # Non-lateral but with runtime dx multiplier (e.g. Chest Spacing for Pec)
        mult_x = _new_node(ng, 'ShaderNodeMath', (-1400, y_off), f"{muscle_name} mult*dx")
        mult_x.operation = 'MULTIPLY'
        _link(ng, dx_mult_socket, mult_x.inputs[0])
        mult_x.inputs[1].default_value = dx_fac
        x_out = mult_x.outputs["Value"]
    else:
        # Fixed dx_fac (used for explicit L/R named muscles on body trunk)
        x_const = _new_node(ng, 'ShaderNodeValue', (-1400, y_off), f"{muscle_name} x={dx_fac}")
        x_const.outputs[0].default_value = dx_fac
        x_out = x_const.outputs[0]

    combine_dir = _new_node(ng, 'ShaderNodeCombineXYZ', (-1200, y_off), f"{muscle_name} dir")
    _link(ng, x_out, combine_dir.inputs["X"])
    combine_dir.inputs["Y"].default_value = dy
    combine_dir.inputs["Z"].default_value = dz

    norm_dir = _new_node(ng, 'ShaderNodeVectorMath', (-1000, y_off), f"{muscle_name} norm")
    norm_dir.operation = 'NORMALIZE'
    _link(ng, combine_dir.outputs["Vector"], norm_dir.inputs[0])

    # Effective axial = per-muscle Axial + optional offset (e.g. Glute Height
    # shifts the whole butt up/down without touching the raw Axial socket).
    axial_src = grp_in.outputs[f"{muscle_name} Axial"]
    if axial_offset_socket is not None:
        ax_off = _new_node(ng, 'ShaderNodeMath', (-950, y_off + 160), f"{muscle_name} ax+off")
        ax_off.operation = 'ADD'
        _link(ng, axial_src, ax_off.inputs[0])
        _link(ng, axial_offset_socket, ax_off.inputs[1])
        axial_src = ax_off.outputs["Value"]

    diff_t = _new_node(ng, 'ShaderNodeMath', (-800, y_off + 80), f"{muscle_name} t-ax")
    diff_t.operation = 'SUBTRACT'
    _link(ng, captured_t_socket, diff_t.inputs[0])
    _link(ng, axial_src, diff_t.inputs[1])

    sq_t = _new_node(ng, 'ShaderNodeMath', (-650, y_off + 80), f"{muscle_name} sq")
    sq_t.operation = 'POWER'
    _link(ng, diff_t.outputs["Value"], sq_t.inputs[0])
    sq_t.inputs[1].default_value = 2.0

    sig_sq = _new_node(ng, 'ShaderNodeMath', (-650, y_off - 30), f"{muscle_name} sig^2")
    sig_sq.operation = 'POWER'
    _link(ng, grp_in.outputs[f"{muscle_name} Sigma"], sig_sq.inputs[0])
    sig_sq.inputs[1].default_value = 2.0

    two_sig = _new_node(ng, 'ShaderNodeMath', (-500, y_off - 30), f"{muscle_name} 2sig^2")
    two_sig.operation = 'MULTIPLY'
    _link(ng, sig_sq.outputs["Value"], two_sig.inputs[0])
    two_sig.inputs[1].default_value = 2.0

    div_g = _new_node(ng, 'ShaderNodeMath', (-350, y_off + 30), f"{muscle_name} div")
    div_g.operation = 'DIVIDE'
    _link(ng, sq_t.outputs["Value"], div_g.inputs[0])
    _link(ng, two_sig.outputs["Value"], div_g.inputs[1])

    neg_g = _new_node(ng, 'ShaderNodeMath', (-200, y_off + 30), f"{muscle_name} neg")
    neg_g.operation = 'MULTIPLY'
    _link(ng, div_g.outputs["Value"], neg_g.inputs[0])
    neg_g.inputs[1].default_value = -1.0

    exp_g = _new_node(ng, 'ShaderNodeMath', (-50, y_off + 30), f"{muscle_name} exp")
    exp_g.operation = 'EXPONENT'
    _link(ng, neg_g.outputs["Value"], exp_g.inputs[0])

    dot_n = _new_node(ng, 'ShaderNodeVectorMath', (-800, y_off - 200), f"{muscle_name} dot")
    dot_n.operation = 'DOT_PRODUCT'
    _link(ng, normal_attr.outputs["Normal"], dot_n.inputs[0])
    _link(ng, norm_dir.outputs["Vector"], dot_n.inputs[1])

    max_d = _new_node(ng, 'ShaderNodeMath', (-650, y_off - 200), f"{muscle_name} max0")
    max_d.operation = 'MAXIMUM'
    _link(ng, dot_n.outputs["Value"], max_d.inputs[0])
    max_d.inputs[1].default_value = 0.0

    pow_d = _new_node(ng, 'ShaderNodeMath', (-500, y_off - 200), f"{muscle_name} pow")
    pow_d.operation = 'POWER'
    _link(ng, max_d.outputs["Value"], pow_d.inputs[0])
    _link(ng, grp_in.outputs[f"{muscle_name} Conc"], pow_d.inputs[1])

    g_x_p = _new_node(ng, 'ShaderNodeMath', (100, y_off - 80), f"{muscle_name} g*p")
    g_x_p.operation = 'MULTIPLY'
    _link(ng, exp_g.outputs["Value"], g_x_p.inputs[0])
    _link(ng, pow_d.outputs["Value"], g_x_p.inputs[1])

    # Plateau remap. The raw factor g_x_p is a smooth peak (→ pointy bulge).
    # MapRange with a narrow [lo,hi] window clips it into a flat-topped plateau:
    # everything above hi → 1 (flat top), below lo → 0, linear wall between.
    #   Flat=0 → lo=0, hi=1  → identity (round Gaussian bulge, unchanged).
    #   Flat=1 → lo=0.42, hi=0.58 → tight plateau (flat-topped box, e.g. abs).
    flat_src = flat_socket if flat_socket is not None else grp_in.outputs[f"{muscle_name} Flat"]
    flat_lo = _new_node(ng, 'ShaderNodeMath', (60, y_off - 250), f"{muscle_name} flat-lo")
    flat_lo.operation = 'MULTIPLY'
    _link(ng, flat_src, flat_lo.inputs[0])
    flat_lo.inputs[1].default_value = 0.42
    flat_hi = _new_node(ng, 'ShaderNodeMath', (60, y_off - 330), f"{muscle_name} flat-hi")
    flat_hi.operation = 'MULTIPLY_ADD'   # Flat × -0.42 + 1.0
    _link(ng, flat_src, flat_hi.inputs[0])
    flat_hi.inputs[1].default_value = -0.42
    flat_hi.inputs[2].default_value = 1.0
    plateau = _new_node(ng, 'ShaderNodeMapRange', (200, y_off - 250), f"{muscle_name} plateau")
    plateau.interpolation_type = 'LINEAR'
    plateau.clamp = True
    _link(ng, g_x_p.outputs["Value"], plateau.inputs["Value"])
    _link(ng, flat_lo.outputs["Value"], plateau.inputs["From Min"])
    _link(ng, flat_hi.outputs["Value"], plateau.inputs["From Max"])
    plateau.inputs["To Min"].default_value = 0.0
    plateau.inputs["To Max"].default_value = 1.0

    amt_mul = _new_node(ng, 'ShaderNodeMath', (400, y_off - 80), f"{muscle_name} *amt")
    amt_mul.operation = 'MULTIPLY'
    _link(ng, plateau.outputs["Result"], amt_mul.inputs[0])
    _link(ng, grp_in.outputs[f"{muscle_name} Amount"], amt_mul.inputs[1])

    arch_mul = _new_node(ng, 'ShaderNodeMath', (400, y_off - 80), f"{muscle_name} *arch")
    arch_mul.operation = 'MULTIPLY'
    _link(ng, amt_mul.outputs["Value"], arch_mul.inputs[0])
    _link(ng, archetype_socket, arch_mul.inputs[1])

    # Displacement = factor × vertex_normal (not muscle_dir).
    # The muscle_dir is used only as a SELECTOR (via dot-product) — once verts
    # are selected, they push OUTWARD along their own normal direction.
    # This creates round bulges instead of converging pointy cones.
    #
    # Sag: when sag_socket is wired (breast), tilt the push direction downward —
    # dir = normalize(normal + (0,0,-Sag)). Sag 0 = firm (straight out),
    # 1 ≈ 45° droop. Magnitude stays constant (normalized) so size doesn't change.
    disp_dir = normal_attr.outputs["Normal"]
    if sag_socket is not None:
        neg_sag = _new_node(ng, 'ShaderNodeMath', (250, y_off - 240), f"{muscle_name} -sag")
        neg_sag.operation = 'MULTIPLY'
        _link(ng, sag_socket, neg_sag.inputs[0])
        neg_sag.inputs[1].default_value = -1.0
        sag_vec = _new_node(ng, 'ShaderNodeCombineXYZ', (400, y_off - 240), f"{muscle_name} sagvec")
        sag_vec.inputs["X"].default_value = 0.0
        sag_vec.inputs["Y"].default_value = 0.0
        _link(ng, neg_sag.outputs["Value"], sag_vec.inputs["Z"])
        sag_add = _new_node(ng, 'ShaderNodeVectorMath', (550, y_off - 240), f"{muscle_name} sagdir")
        sag_add.operation = 'ADD'
        _link(ng, normal_attr.outputs["Normal"], sag_add.inputs[0])
        _link(ng, sag_vec.outputs["Vector"], sag_add.inputs[1])
        sag_nrm = _new_node(ng, 'ShaderNodeVectorMath', (700, y_off - 240), f"{muscle_name} sagnrm")
        sag_nrm.operation = 'NORMALIZE'
        _link(ng, sag_add.outputs["Vector"], sag_nrm.inputs[0])
        disp_dir = sag_nrm.outputs["Vector"]

    disp = _new_node(ng, 'ShaderNodeVectorMath', (850, y_off - 80), f"{muscle_name} disp")
    disp.operation = 'SCALE'
    _link(ng, disp_dir, disp.inputs[0])
    _link(ng, arch_mul.outputs["Value"], disp.inputs["Scale"])

    return disp


def _build_front_density_warp(ng, mesh_socket, density_socket, target_angle=None):
    """Redistribute a uniform profile circle's verts toward a centerline.

    target_angle: profile-local angle to cluster toward. Default -pi/2 = the
    anterior centerline (torso — sternum/breasts). Pass +pi/2 for the posterior
    centerline (pelvis — glute cleft + round cheeks).

    Why: the sternum groove and round breast/belly need dense geometry on the
    front. A uniform circle puts ONE vert on the centerline with a big flat gap
    to its lateral neighbours — the mesh physically can't dip sharply then rise
    back up. This warp clusters verts near the front so the groove can be a
    knife-edge AND the breasts can stay round.

    density = 1.0 → identity (uniform, used by limbs).
    density > 1.0 → verts cluster toward the front centerline.

    Math per profile vert (unit circle):
        theta  = atan2(y, x)
        phi    = wrap(theta - THETA_T)        # angular offset from front
        phi'   = sign(phi) * pi * (|phi|/pi) ^ density
        theta' = THETA_T + phi'
        pos'   = (cos theta', sin theta', 0)

    Returns: geometry socket of the warped mesh.
    """
    PI = 3.14159265358979
    THETA_T = (-PI / 2.0) if target_angle is None else target_angle
    bx, by = -2700, -2000

    pos = _new_node(ng, 'GeometryNodeInputPosition', (bx, by), "fd pos")
    sep = _new_node(ng, 'ShaderNodeSeparateXYZ', (bx + 200, by), "fd sep")
    _link(ng, pos.outputs["Position"], sep.inputs["Vector"])

    theta = _new_node(ng, 'ShaderNodeMath', (bx + 400, by), "fd theta")
    theta.operation = 'ARCTAN2'
    _link(ng, sep.outputs["Y"], theta.inputs[0])
    _link(ng, sep.outputs["X"], theta.inputs[1])

    phi = _new_node(ng, 'ShaderNodeMath', (bx + 600, by), "fd phi")
    phi.operation = 'SUBTRACT'
    _link(ng, theta.outputs["Value"], phi.inputs[0])
    phi.inputs[1].default_value = THETA_T

    # wrap phi into [-pi, pi] via atan2(sin, cos)
    sinp = _new_node(ng, 'ShaderNodeMath', (bx + 800, by + 160), "fd sin")
    sinp.operation = 'SINE'
    _link(ng, phi.outputs["Value"], sinp.inputs[0])
    cosp = _new_node(ng, 'ShaderNodeMath', (bx + 800, by - 160), "fd cos")
    cosp.operation = 'COSINE'
    _link(ng, phi.outputs["Value"], cosp.inputs[0])
    phiw = _new_node(ng, 'ShaderNodeMath', (bx + 1000, by), "fd phi_w")
    phiw.operation = 'ARCTAN2'
    _link(ng, sinp.outputs["Value"], phiw.inputs[0])
    _link(ng, cosp.outputs["Value"], phiw.inputs[1])

    sgn = _new_node(ng, 'ShaderNodeMath', (bx + 1200, by + 160), "fd sign")
    sgn.operation = 'SIGN'
    _link(ng, phiw.outputs["Value"], sgn.inputs[0])
    absp = _new_node(ng, 'ShaderNodeMath', (bx + 1200, by - 160), "fd abs")
    absp.operation = 'ABSOLUTE'
    _link(ng, phiw.outputs["Value"], absp.inputs[0])

    norm = _new_node(ng, 'ShaderNodeMath', (bx + 1400, by - 160), "fd norm")
    norm.operation = 'DIVIDE'
    _link(ng, absp.outputs["Value"], norm.inputs[0])
    norm.inputs[1].default_value = PI

    powd = _new_node(ng, 'ShaderNodeMath', (bx + 1600, by - 160), "fd pow")
    powd.operation = 'POWER'
    _link(ng, norm.outputs["Value"], powd.inputs[0])
    _link(ng, density_socket, powd.inputs[1])

    mpi = _new_node(ng, 'ShaderNodeMath', (bx + 1800, by - 160), "fd *pi")
    mpi.operation = 'MULTIPLY'
    _link(ng, powd.outputs["Value"], mpi.inputs[0])
    mpi.inputs[1].default_value = PI

    phip = _new_node(ng, 'ShaderNodeMath', (bx + 2000, by), "fd phi'")
    phip.operation = 'MULTIPLY'
    _link(ng, mpi.outputs["Value"], phip.inputs[0])
    _link(ng, sgn.outputs["Value"], phip.inputs[1])

    thetap = _new_node(ng, 'ShaderNodeMath', (bx + 2200, by), "fd theta'")
    thetap.operation = 'ADD'
    _link(ng, phip.outputs["Value"], thetap.inputs[0])
    thetap.inputs[1].default_value = THETA_T

    nx = _new_node(ng, 'ShaderNodeMath', (bx + 2400, by + 160), "fd nx")
    nx.operation = 'COSINE'
    _link(ng, thetap.outputs["Value"], nx.inputs[0])
    nyv = _new_node(ng, 'ShaderNodeMath', (bx + 2400, by - 160), "fd ny")
    nyv.operation = 'SINE'
    _link(ng, thetap.outputs["Value"], nyv.inputs[0])
    ncomb = _new_node(ng, 'ShaderNodeCombineXYZ', (bx + 2600, by), "fd ncomb")
    _link(ng, nx.outputs["Value"], ncomb.inputs["X"])
    _link(ng, nyv.outputs["Value"], ncomb.inputs["Y"])
    ncomb.inputs["Z"].default_value = 0.0

    setp = _new_node(ng, 'GeometryNodeSetPosition', (bx + 2850, by + 400), "fd setpos")
    _link(ng, mesh_socket, setp.inputs["Geometry"])
    _link(ng, ncomb.outputs["Vector"], setp.inputs["Position"])

    return setp.outputs["Geometry"]


def _build_dshape_profile(ng, curve_socket, depth_ratio_socket,
                          anterior_socket, posterior_socket):
    """Reshape a unit-circle profile curve into a D-shaped cross-section.

    A pelvis cross-section is never circular — the posterior (glutes/sacrum)
    is roughly 1.8x the depth of the anterior (pubis). A swept circle can
    never read as a butt no matter how much you displace it. This scales the
    profile's +Y (posterior) half by Posterior Depth and the -Y (anterior)
    half by Anterior Depth, with a smoothstep blend across the Y=0 sides so
    the two half-ellipses join C1-continuous. Depth Ratio still applies as an
    overall front-back master on top.

    Runs on the profile curve BEFORE Curve to Mesh, so the swept mesh is
    natively D-shaped and Input Normal yields correct normals — the muscle
    displacement system needs no changes.

    Returns the warped curve geometry socket.
    """
    bx, by = -900, -500
    pos = _new_node(ng, 'GeometryNodeInputPosition', (bx, by), "ds pos")
    sep = _new_node(ng, 'ShaderNodeSeparateXYZ', (bx + 180, by), "ds sep")
    _link(ng, pos.outputs["Position"], sep.inputs["Vector"])

    # t = (y + 1) / 2  → 0 at full anterior, 1 at full posterior
    t = _new_node(ng, 'ShaderNodeMath', (bx + 360, by - 140), "ds t")
    t.operation = 'MULTIPLY_ADD'
    _link(ng, sep.outputs["Y"], t.inputs[0])
    t.inputs[1].default_value = 0.5
    t.inputs[2].default_value = 0.5

    # smoothstep(t) then lerp anterior→posterior — one MapRange does both
    depth = _new_node(ng, 'ShaderNodeMapRange', (bx + 520, by - 140), "ds depth")
    depth.interpolation_type = 'SMOOTHSTEP'
    depth.clamp = True
    _link(ng, t.outputs["Value"], depth.inputs["Value"])
    depth.inputs["From Min"].default_value = 0.0
    depth.inputs["From Max"].default_value = 1.0
    _link(ng, anterior_socket, depth.inputs["To Min"])
    _link(ng, posterior_socket, depth.inputs["To Max"])

    # overall front-back master still applies
    fac = _new_node(ng, 'ShaderNodeMath', (bx + 700, by - 140), "ds fac")
    fac.operation = 'MULTIPLY'
    _link(ng, depth.outputs["Result"], fac.inputs[0])
    _link(ng, depth_ratio_socket, fac.inputs[1])

    ynew = _new_node(ng, 'ShaderNodeMath', (bx + 880, by), "ds y'")
    ynew.operation = 'MULTIPLY'
    _link(ng, sep.outputs["Y"], ynew.inputs[0])
    _link(ng, fac.outputs["Value"], ynew.inputs[1])

    comb = _new_node(ng, 'ShaderNodeCombineXYZ', (bx + 1060, by), "ds comb")
    _link(ng, sep.outputs["X"], comb.inputs["X"])
    _link(ng, ynew.outputs["Value"], comb.inputs["Y"])
    comb.inputs["Z"].default_value = 0.0

    setp = _new_node(ng, 'GeometryNodeSetPosition', (bx + 1260, by + 220), "ds setpos")
    _link(ng, curve_socket, setp.inputs["Geometry"])
    _link(ng, comb.outputs["Vector"], setp.inputs["Position"])
    return setp.outputs["Geometry"]


def _build_anatomy_group(group_name, muscle_specs, muscle_defaults,
                          start_radius=0.05, end_radius=0.03,
                          default_n_axial=14, default_depth_ratio=1.0,
                          default_n_sides=8, default_front_density=1.0,
                          default_neck_ratio=1.0, default_density_target=None,
                          default_fill_caps=False, curved=False,
                          force_rebuild=False):
    """Build a generic limb-anatomy GN group.

    muscle_specs:    dict {muscle_name: (dx_fac, dy, dz, is_lateral)}
    muscle_defaults: dict {muscle_name: (axial, amount, sigma, conc)}
    curved:          if True the sweep axis is a Quadratic Bezier through a
                     Mid Empty (the limb bends). If the Mid Empty is unset it
                     falls back to the straight P0→P1 line. Used by the thigh
                     so the leg leaves the hip socket at the splay angle and
                     curves the knee back under the body.

    Returns: bpy.types.GeometryNodeTree
    """
    if group_name in bpy.data.node_groups:
        if not force_rebuild:
            return bpy.data.node_groups[group_name]
        bpy.data.node_groups.remove(bpy.data.node_groups[group_name])

    ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
    ng.is_modifier = True
    iface = ng.interface

    iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    iface.new_socket(name="Start Empty", in_out='INPUT', socket_type='NodeSocketObject')
    iface.new_socket(name="End Empty",   in_out='INPUT', socket_type='NodeSocketObject')
    if curved:
        iface.new_socket(name="Mid Empty", in_out='INPUT', socket_type='NodeSocketObject')

    s = iface.new_socket(name="N Sides", in_out='INPUT', socket_type='NodeSocketInt')
    s.default_value = default_n_sides; s.min_value = 3; s.max_value = 48
    s = iface.new_socket(name="N Axial", in_out='INPUT', socket_type='NodeSocketInt')
    s.default_value = default_n_axial; s.min_value = 3; s.max_value = 32

    s = iface.new_socket(name="Start Radius", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = start_radius; s.min_value = 0.001; s.max_value = 0.5
    s = iface.new_socket(name="End Radius", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = end_radius; s.min_value = 0.001; s.max_value = 0.5

    s = iface.new_socket(name="Limb Side", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = -1.0; s.min_value = -1.0; s.max_value = 1.0

    # Depth Ratio: profile front-back vs side-to-side. 1.0 = circular, <1.0 = flatter front-back.
    # Defaults to 1.0 (limbs are circular). Torso uses ~0.65 (flatter chest plane).
    s = iface.new_socket(name="Depth Ratio", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_depth_ratio; s.min_value = 0.3; s.max_value = 2.0

    # Front Density: clusters profile verts toward the anterior centerline so the
    # sternum groove can be a knife-edge while breasts stay round. 1.0 = uniform
    # (limbs); torso uses ~1.8. See _build_front_density_warp.
    s = iface.new_socket(name="Front Density", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_front_density; s.min_value = 1.0; s.max_value = 4.0

    # Waist: mid-torso radius pinch/bulge (Gaussian bump centered at t≈0.20).
    # <1 = pinched waist (hourglass), 1 = straight (box), >1 = bulging (round/fat).
    s = iface.new_socket(name="Waist", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 1.0; s.min_value = 0.4; s.max_value = 1.6

    # Shoulder Width: radius bump at t≈0.92 (shoulder line). >1 = broad shoulders
    # (V-taper), <1 = narrow. Pairs with Waist to span V / hourglass / square / round.
    s = iface.new_socket(name="Shoulder Width", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 1.0; s.min_value = 0.6; s.max_value = 1.5

    # Ribcage: radius bump at t≈0.70 (chest band). The bony chest FRAME the
    # breasts/pecs sit on — slender vs broad ribcage, independent of Chest Size.
    s = iface.new_socket(name="Ribcage", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = 1.0; s.min_value = 0.6; s.max_value = 1.5

    # Neck Ratio: above t≈0.92 the radius tapers down to this fraction of the
    # shoulder radius. The taper surface IS the shoulder shelf; the t=1 opening
    # IS the neck hole. 1.0 = no taper (limbs); torso uses ~0.55.
    s = iface.new_socket(name="Neck Ratio", in_out='INPUT', socket_type='NodeSocketFloat')
    s.default_value = default_neck_ratio; s.min_value = 0.3; s.max_value = 1.0

    # Per-muscle inputs. Defaults are 4-tuples (axial, amount, sigma, conc) or
    # 5-tuples with a trailing Flat value (abs use this for plateau-topped boxes).
    for name in muscle_specs:
        vals = muscle_defaults[name]
        axial, amount, sigma, conc = vals[:4]
        flat = vals[4] if len(vals) > 4 else 0.0
        _add_muscle_inputs(iface, name, axial, amount, sigma, conc, flat)

    # Archetype scalars — range allows negative (inverts muscle direction) and very large
    for name in muscle_specs:
        s = iface.new_socket(name=f"{name} Archetype", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = -2.0; s.max_value = 8.0

    # Chest Spacing — only for torso groups that have Pec_L/Pec_R. Scales the
    # X component of Pec direction to push breast peaks closer or farther apart.
    if "Pec_L" in muscle_specs and "Pec_R" in muscle_specs:
        s = iface.new_socket(name="Chest Spacing", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = 0.2; s.max_value = 3.0
        # Chest Type: 0 = male (flat slab pec), 1 = female (round breast).
        # Drives the Pec Flat plateau — male pecs read as flat, female as round.
        s = iface.new_socket(name="Chest Type", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = 0.0; s.max_value = 1.0
        # Chest Sag: 0 = firm (projects straight out), ~1 = 45° droop. Tilts the
        # breast/pec push direction downward so the mass hangs over the fold.
        s = iface.new_socket(name="Chest Sag", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 0.0; s.min_value = 0.0; s.max_value = 1.5
        # Chest Size: master for breast/pec MASS — multiplies both Pec amounts.
        # This is the bulge out from the chest, separate from the Ribcage frame.
        s = iface.new_socket(name="Chest Size", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = 0.0; s.max_value = 10.0

    # Ab Definition — one master slider for all 8 ab peaks (+ semilunaris).
    # Multiplies each ab's per-muscle Archetype. The built-in Ab1>Ab2>Ab3>Ab4
    # gradient lives in the base Amounts, so this single dial scales the 8-pack.
    # Ab Flat — one master soft↔cube toggle: 0 = soft bulge, 1 = hard cube face.
    if "Ab1L" in muscle_specs:
        s = iface.new_socket(name="Ab Definition", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = 0.0; s.max_value = 2.5
        s = iface.new_socket(name="Ab Flat", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 0.0; s.min_value = 0.0; s.max_value = 1.0

    # Pelvis controls — only for groups with Glute_L/R.
    # Hip Width = hip-flare radius bump; Glute Size = master butt mass;
    # Glute Sag = perky(<0) ↔ saggy(>0) tilt of the glute push direction.
    if "Glute_L" in muscle_specs and "Glute_R" in muscle_specs:
        s = iface.new_socket(name="Hip Width", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = 0.5; s.max_value = 1.7
        s = iface.new_socket(name="Glute Size", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = 0.0; s.max_value = 10.0
        s = iface.new_socket(name="Glute Sag", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 0.0; s.min_value = -1.0; s.max_value = 1.5
        # Glute Spacing — gap width: scales the glute dx so cheeks sit closer
        # (<1, tighter cleft) or wider apart (>1).
        s = iface.new_socket(name="Glute Spacing", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.0; s.min_value = 0.3; s.max_value = 2.5
        # Glute Height — shifts the butt mass up (+) or down (-) along the pelvis.
        s = iface.new_socket(name="Glute Height", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 0.0; s.min_value = -0.2; s.max_value = 0.2
        # Anterior / Posterior Depth — the D-shape cross-section. The pelvis is
        # never circular: posterior (glutes/sacrum) runs ~1.8x the anterior
        # (pubis) depth. These scale the -Y / +Y halves of the profile
        # independently — the base skeletal D-shape the glute muscles sit on.
        s = iface.new_socket(name="Anterior Depth", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 0.95; s.min_value = 0.2; s.max_value = 2.5
        s = iface.new_socket(name="Posterior Depth", in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = 1.35; s.min_value = 0.2; s.max_value = 3.5

    # ---- Graph ----
    grp_in = _new_node(ng, 'NodeGroupInput', (-2400, 0))
    grp_out = _new_node(ng, 'NodeGroupOutput', (2000, 0))

    obj_info_p0 = _new_node(ng, 'GeometryNodeObjectInfo', (-2100, 600), "P0")
    obj_info_p0.transform_space = 'ORIGINAL'
    _link(ng, grp_in.outputs["Start Empty"], obj_info_p0.inputs["Object"])

    obj_info_p1 = _new_node(ng, 'GeometryNodeObjectInfo', (-2100, 300), "P1")
    obj_info_p1.transform_space = 'ORIGINAL'
    _link(ng, grp_in.outputs["End Empty"], obj_info_p1.inputs["Object"])

    # Axis curve. curved=True → Quadratic Bezier through P0 / Mid / P1 so the
    # limb bends (thigh). Else a straight P0→P1 line. axis_curve feeds both the
    # profile sweep and the captured axial-t coordinate.
    if curved:
        obj_info_mid = _new_node(ng, 'GeometryNodeObjectInfo', (-2100, 450), "Mid")
        obj_info_mid.transform_space = 'ORIGINAL'
        _link(ng, grp_in.outputs["Mid Empty"], obj_info_mid.inputs["Object"])
        qbez = _new_node(ng, 'GeometryNodeCurveQuadraticBezier', (-1800, 400), "Axis (curved)")
        _link(ng, grp_in.outputs["N Axial"], qbez.inputs["Resolution"])
        _link(ng, obj_info_p0.outputs["Location"],  qbez.inputs["Start"])
        _link(ng, obj_info_mid.outputs["Location"], qbez.inputs["Middle"])
        _link(ng, obj_info_p1.outputs["Location"],  qbez.inputs["End"])
        axis_curve = qbez.outputs["Curve"]
    else:
        mesh_line = _new_node(ng, 'GeometryNodeMeshLine', (-1800, 400), "Axis")
        mesh_line.mode = 'END_POINTS'
        _link(ng, grp_in.outputs["N Axial"], mesh_line.inputs["Count"])
        _link(ng, obj_info_p0.outputs["Location"], mesh_line.inputs["Start Location"])
        _link(ng, obj_info_p1.outputs["Location"], mesh_line.inputs["Offset"])
        mesh_to_curve = _new_node(ng, 'GeometryNodeMeshToCurve', (-1550, 400))
        _link(ng, mesh_line.outputs["Mesh"], mesh_to_curve.inputs["Mesh"])
        axis_curve = mesh_to_curve.outputs["Curve"]

    spline_param = _new_node(ng, 'GeometryNodeSplineParameter', (-1550, 600))

    sub_r = _new_node(ng, 'ShaderNodeMath', (-1250, 800), "end-start")
    sub_r.operation = 'SUBTRACT'
    _link(ng, grp_in.outputs["End Radius"], sub_r.inputs[0])
    _link(ng, grp_in.outputs["Start Radius"], sub_r.inputs[1])

    mul_r = _new_node(ng, 'ShaderNodeMath', (-1100, 800), "diff*t")
    mul_r.operation = 'MULTIPLY'
    _link(ng, sub_r.outputs["Value"], mul_r.inputs[0])
    _link(ng, spline_param.outputs["Factor"], mul_r.inputs[1])

    final_r = _new_node(ng, 'ShaderNodeMath', (-950, 800), "lerp r")
    final_r.operation = 'ADD'
    _link(ng, grp_in.outputs["Start Radius"], final_r.inputs[0])
    _link(ng, mul_r.outputs["Value"], final_r.inputs[1])

    # Waist profile — Gaussian bump at t≈0.20 modulates the radius.
    # r' = r × (1 + (Waist-1) × exp(-((t-0.20)/0.18)^2))
    w_d = _new_node(ng, 'ShaderNodeMath', (-1250, 1050), "waist t-0.2")
    w_d.operation = 'SUBTRACT'
    _link(ng, spline_param.outputs["Factor"], w_d.inputs[0])
    w_d.inputs[1].default_value = 0.20
    w_n = _new_node(ng, 'ShaderNodeMath', (-1100, 1050), "waist /w")
    w_n.operation = 'DIVIDE'
    _link(ng, w_d.outputs["Value"], w_n.inputs[0])
    w_n.inputs[1].default_value = 0.18
    w_sq = _new_node(ng, 'ShaderNodeMath', (-950, 1050), "waist sq")
    w_sq.operation = 'POWER'
    _link(ng, w_n.outputs["Value"], w_sq.inputs[0])
    w_sq.inputs[1].default_value = 2.0
    w_neg = _new_node(ng, 'ShaderNodeMath', (-800, 1050), "waist neg")
    w_neg.operation = 'MULTIPLY'
    _link(ng, w_sq.outputs["Value"], w_neg.inputs[0])
    w_neg.inputs[1].default_value = -1.0
    w_bump = _new_node(ng, 'ShaderNodeMath', (-650, 1050), "waist bump")
    w_bump.operation = 'EXPONENT'
    _link(ng, w_neg.outputs["Value"], w_bump.inputs[0])
    w_m1 = _new_node(ng, 'ShaderNodeMath', (-650, 1200), "waist-1")
    w_m1.operation = 'SUBTRACT'
    _link(ng, grp_in.outputs["Waist"], w_m1.inputs[0])
    w_m1.inputs[1].default_value = 1.0
    w_fac = _new_node(ng, 'ShaderNodeMath', (-500, 1100), "waist factor")
    w_fac.operation = 'MULTIPLY_ADD'   # (Waist-1)*bump + 1
    _link(ng, w_m1.outputs["Value"], w_fac.inputs[0])
    _link(ng, w_bump.outputs["Value"], w_fac.inputs[1])
    w_fac.inputs[2].default_value = 1.0
    r_waisted = _new_node(ng, 'ShaderNodeMath', (-350, 900), "r x waist")
    r_waisted.operation = 'MULTIPLY'
    _link(ng, final_r.outputs["Value"], r_waisted.inputs[0])
    _link(ng, w_fac.outputs["Value"], r_waisted.inputs[1])

    # Shoulder Width — Gaussian bump at t≈0.92 modulates the radius (V-taper).
    sw_d = _new_node(ng, 'ShaderNodeMath', (-1250, 1350), "shldr t-0.92")
    sw_d.operation = 'SUBTRACT'
    _link(ng, spline_param.outputs["Factor"], sw_d.inputs[0])
    sw_d.inputs[1].default_value = 0.92
    sw_n = _new_node(ng, 'ShaderNodeMath', (-1100, 1350), "shldr /w")
    sw_n.operation = 'DIVIDE'
    _link(ng, sw_d.outputs["Value"], sw_n.inputs[0])
    sw_n.inputs[1].default_value = 0.12
    sw_sq = _new_node(ng, 'ShaderNodeMath', (-950, 1350), "shldr sq")
    sw_sq.operation = 'POWER'
    _link(ng, sw_n.outputs["Value"], sw_sq.inputs[0])
    sw_sq.inputs[1].default_value = 2.0
    sw_neg = _new_node(ng, 'ShaderNodeMath', (-800, 1350), "shldr neg")
    sw_neg.operation = 'MULTIPLY'
    _link(ng, sw_sq.outputs["Value"], sw_neg.inputs[0])
    sw_neg.inputs[1].default_value = -1.0
    sw_bump = _new_node(ng, 'ShaderNodeMath', (-650, 1350), "shldr bump")
    sw_bump.operation = 'EXPONENT'
    _link(ng, sw_neg.outputs["Value"], sw_bump.inputs[0])
    sw_m1 = _new_node(ng, 'ShaderNodeMath', (-650, 1500), "shldrW-1")
    sw_m1.operation = 'SUBTRACT'
    _link(ng, grp_in.outputs["Shoulder Width"], sw_m1.inputs[0])
    sw_m1.inputs[1].default_value = 1.0
    sw_fac = _new_node(ng, 'ShaderNodeMath', (-500, 1400), "shldr factor")
    sw_fac.operation = 'MULTIPLY_ADD'   # (ShoulderWidth-1)*bump + 1
    _link(ng, sw_m1.outputs["Value"], sw_fac.inputs[0])
    _link(ng, sw_bump.outputs["Value"], sw_fac.inputs[1])
    sw_fac.inputs[2].default_value = 1.0
    r_shldr = _new_node(ng, 'ShaderNodeMath', (-200, 900), "r x shldr")
    r_shldr.operation = 'MULTIPLY'
    _link(ng, r_waisted.outputs["Value"], r_shldr.inputs[0])
    _link(ng, sw_fac.outputs["Value"], r_shldr.inputs[1])

    # Ribcage — Gaussian bump at t≈0.70 modulates the radius (chest frame).
    rb_d = _new_node(ng, 'ShaderNodeMath', (-1250, 1650), "ribcage t-0.70")
    rb_d.operation = 'SUBTRACT'
    _link(ng, spline_param.outputs["Factor"], rb_d.inputs[0])
    rb_d.inputs[1].default_value = 0.70
    rb_n = _new_node(ng, 'ShaderNodeMath', (-1100, 1650), "ribcage /w")
    rb_n.operation = 'DIVIDE'
    _link(ng, rb_d.outputs["Value"], rb_n.inputs[0])
    rb_n.inputs[1].default_value = 0.16
    rb_sq = _new_node(ng, 'ShaderNodeMath', (-950, 1650), "ribcage sq")
    rb_sq.operation = 'POWER'
    _link(ng, rb_n.outputs["Value"], rb_sq.inputs[0])
    rb_sq.inputs[1].default_value = 2.0
    rb_neg = _new_node(ng, 'ShaderNodeMath', (-800, 1650), "ribcage neg")
    rb_neg.operation = 'MULTIPLY'
    _link(ng, rb_sq.outputs["Value"], rb_neg.inputs[0])
    rb_neg.inputs[1].default_value = -1.0
    rb_bump = _new_node(ng, 'ShaderNodeMath', (-650, 1650), "ribcage bump")
    rb_bump.operation = 'EXPONENT'
    _link(ng, rb_neg.outputs["Value"], rb_bump.inputs[0])
    rb_m1 = _new_node(ng, 'ShaderNodeMath', (-650, 1800), "ribcage-1")
    rb_m1.operation = 'SUBTRACT'
    _link(ng, grp_in.outputs["Ribcage"], rb_m1.inputs[0])
    rb_m1.inputs[1].default_value = 1.0
    rb_fac = _new_node(ng, 'ShaderNodeMath', (-500, 1700), "ribcage factor")
    rb_fac.operation = 'MULTIPLY_ADD'   # (Ribcage-1)*bump + 1
    _link(ng, rb_m1.outputs["Value"], rb_fac.inputs[0])
    _link(ng, rb_bump.outputs["Value"], rb_fac.inputs[1])
    rb_fac.inputs[2].default_value = 1.0
    r_ribcaged = _new_node(ng, 'ShaderNodeMath', (-50, 900), "r x ribcage")
    r_ribcaged.operation = 'MULTIPLY'
    _link(ng, r_shldr.outputs["Value"], r_ribcaged.inputs[0])
    _link(ng, rb_fac.outputs["Value"], r_ribcaged.inputs[1])

    # Neck taper — above t≈0.92 the radius lerps down to Neck Ratio. The taper
    # surface forms the shoulder shelf; the small top opening is the neck hole.
    nt_ramp = _new_node(ng, 'ShaderNodeMapRange', (-1000, 1950), "neck ramp")
    nt_ramp.interpolation_type = 'LINEAR'
    nt_ramp.clamp = True
    _link(ng, spline_param.outputs["Factor"], nt_ramp.inputs["Value"])
    nt_ramp.inputs["From Min"].default_value = 0.92
    nt_ramp.inputs["From Max"].default_value = 1.0
    nt_ramp.inputs["To Min"].default_value = 0.0
    nt_ramp.inputs["To Max"].default_value = 1.0
    nt_inv = _new_node(ng, 'ShaderNodeMath', (-800, 1950), "1-NeckRatio")
    nt_inv.operation = 'SUBTRACT'
    nt_inv.inputs[0].default_value = 1.0
    _link(ng, grp_in.outputs["Neck Ratio"], nt_inv.inputs[1])
    nt_drop = _new_node(ng, 'ShaderNodeMath', (-650, 1950), "neck drop")
    nt_drop.operation = 'MULTIPLY'
    _link(ng, nt_inv.outputs["Value"], nt_drop.inputs[0])
    _link(ng, nt_ramp.outputs["Result"], nt_drop.inputs[1])
    nt_fac = _new_node(ng, 'ShaderNodeMath', (-500, 1950), "neck factor")
    nt_fac.operation = 'SUBTRACT'
    nt_fac.inputs[0].default_value = 1.0
    _link(ng, nt_drop.outputs["Value"], nt_fac.inputs[1])
    r_necked = _new_node(ng, 'ShaderNodeMath', (100, 900), "r x neck")
    r_necked.operation = 'MULTIPLY'
    _link(ng, r_ribcaged.outputs["Value"], r_necked.inputs[0])
    _link(ng, nt_fac.outputs["Value"], r_necked.inputs[1])

    # Hip Width — Gaussian bump at t≈0.30 modulates the radius (pelvis hip flare).
    # Only meaningful when a Hip Width socket exists (pelvis); 1.0 = no-op otherwise.
    hw_socket = None
    try:
        hw_socket = grp_in.outputs["Hip Width"]
    except (KeyError, RuntimeError):
        hw_socket = None
    if hw_socket is not None:
        hw_d = _new_node(ng, 'ShaderNodeMath', (-1250, 2250), "hip t-0.30")
        hw_d.operation = 'SUBTRACT'
        _link(ng, spline_param.outputs["Factor"], hw_d.inputs[0])
        hw_d.inputs[1].default_value = 0.30
        hw_n = _new_node(ng, 'ShaderNodeMath', (-1100, 2250), "hip /w")
        hw_n.operation = 'DIVIDE'
        _link(ng, hw_d.outputs["Value"], hw_n.inputs[0])
        hw_n.inputs[1].default_value = 0.26
        hw_sq = _new_node(ng, 'ShaderNodeMath', (-950, 2250), "hip sq")
        hw_sq.operation = 'POWER'
        _link(ng, hw_n.outputs["Value"], hw_sq.inputs[0])
        hw_sq.inputs[1].default_value = 2.0
        hw_neg = _new_node(ng, 'ShaderNodeMath', (-800, 2250), "hip neg")
        hw_neg.operation = 'MULTIPLY'
        _link(ng, hw_sq.outputs["Value"], hw_neg.inputs[0])
        hw_neg.inputs[1].default_value = -1.0
        hw_bump = _new_node(ng, 'ShaderNodeMath', (-650, 2250), "hip bump")
        hw_bump.operation = 'EXPONENT'
        _link(ng, hw_neg.outputs["Value"], hw_bump.inputs[0])
        hw_m1 = _new_node(ng, 'ShaderNodeMath', (-650, 2400), "hipW-1")
        hw_m1.operation = 'SUBTRACT'
        _link(ng, hw_socket, hw_m1.inputs[0])
        hw_m1.inputs[1].default_value = 1.0
        hw_fac = _new_node(ng, 'ShaderNodeMath', (-500, 2300), "hip factor")
        hw_fac.operation = 'MULTIPLY_ADD'   # (HipWidth-1)*bump + 1
        _link(ng, hw_m1.outputs["Value"], hw_fac.inputs[0])
        _link(ng, hw_bump.outputs["Value"], hw_fac.inputs[1])
        hw_fac.inputs[2].default_value = 1.0
        r_final = _new_node(ng, 'ShaderNodeMath', (250, 900), "r x hip")
        r_final.operation = 'MULTIPLY'
        _link(ng, r_necked.outputs["Value"], r_final.inputs[0])
        _link(ng, hw_fac.outputs["Value"], r_final.inputs[1])
    else:
        r_final = r_necked

    profile_circle = _new_node(ng, 'GeometryNodeMeshCircle', (-1300, 0), "Profile")
    profile_circle.fill_type = 'NONE'
    _link(ng, grp_in.outputs["N Sides"], profile_circle.inputs["Vertices"])
    profile_circle.inputs["Radius"].default_value = 1.0

    # Cluster profile verts toward the detail centerline (anterior for torso,
    # posterior for pelvis — see default_density_target).
    warped_profile = _build_front_density_warp(
        ng, profile_circle.outputs["Mesh"], grp_in.outputs["Front Density"],
        target_angle=default_density_target)

    profile_to_curve = _new_node(ng, 'GeometryNodeMeshToCurve', (-1100, 0))
    _link(ng, warped_profile, profile_to_curve.inputs["Mesh"])

    # Cross-section depth. Limbs/torso: uniform elliptical Y-scale by Depth
    # Ratio. Pelvis: asymmetric D-shape — posterior half ~1.8x the anterior
    # (real pelvis anatomy; a circle can never read as a butt).
    is_pelvis = ("Glute_L" in muscle_specs and "Glute_R" in muscle_specs)
    if is_pelvis:
        profile_final = _build_dshape_profile(
            ng, profile_to_curve.outputs["Curve"],
            grp_in.outputs["Depth Ratio"],
            grp_in.outputs["Anterior Depth"],
            grp_in.outputs["Posterior Depth"])
    else:
        depth_combine = _new_node(ng, 'ShaderNodeCombineXYZ', (-900, -100), "depth scale")
        depth_combine.inputs["X"].default_value = 1.0
        _link(ng, grp_in.outputs["Depth Ratio"], depth_combine.inputs["Y"])
        depth_combine.inputs["Z"].default_value = 1.0
        profile_transform = _new_node(ng, 'GeometryNodeTransform', (-900, 0), "elliptical")
        _link(ng, profile_to_curve.outputs["Curve"], profile_transform.inputs["Geometry"])
        _link(ng, depth_combine.outputs["Vector"], profile_transform.inputs["Scale"])
        profile_final = profile_transform.outputs["Geometry"]

    capture_t = _new_node(ng, 'GeometryNodeCaptureAttribute', (-1300, 400), "capture t")
    try:
        capture_t.capture_items.new('FLOAT', 'CurveT')
        capture_t.domain = 'POINT'
    except Exception:
        capture_t.data_type = 'FLOAT'
        capture_t.domain = 'POINT'
    _link(ng, axis_curve, capture_t.inputs["Geometry"])
    _link(ng, spline_param.outputs["Factor"], capture_t.inputs[1])

    curve_to_mesh = _new_node(ng, 'GeometryNodeCurveToMesh', (-700, 400))
    curve_to_mesh.inputs["Fill Caps"].default_value = default_fill_caps
    _link(ng, capture_t.outputs["Geometry"], curve_to_mesh.inputs["Curve"])
    _link(ng, profile_final, curve_to_mesh.inputs["Profile Curve"])
    _link(ng, r_final.outputs["Value"], curve_to_mesh.inputs["Scale"])

    captured_t_socket = capture_t.outputs[1]
    normal_attr = _new_node(ng, 'GeometryNodeInputNormal', (-1400, -300), "normal")

    # If this group has Pec_L/Pec_R muscles, expose a Chest Spacing socket
    # that scales their dx_fac at runtime.
    chest_spacing_socket = None
    if "Pec_L" in muscle_specs and "Pec_R" in muscle_specs:
        try:
            chest_spacing_socket = grp_in.outputs["Chest Spacing"]
        except (KeyError, RuntimeError):
            chest_spacing_socket = None

    # Ab Definition + Ab Flat master sockets.
    ab_def_socket = None
    ab_flat_socket = None
    if "Ab1L" in muscle_specs:
        try:
            ab_def_socket = grp_in.outputs["Ab Definition"]
            ab_flat_socket = grp_in.outputs["Ab Flat"]
        except (KeyError, RuntimeError):
            ab_def_socket = None
            ab_flat_socket = None

    # Chest Type → Pec Flat: pec_flat = (1 - ChestType) × 0.6.
    # ChestType 0 (male) → Flat 0.6 (flat slab); 1 (female) → Flat 0 (round).
    pec_flat_socket = None
    pec_sag_socket = None
    chest_size_socket = None
    if "Pec_L" in muscle_specs and "Pec_R" in muscle_specs:
        try:
            ctype = grp_in.outputs["Chest Type"]
            pf = _new_node(ng, 'ShaderNodeMath', (-1600, 200), "pec flat")
            pf.operation = 'MULTIPLY_ADD'   # ChestType × -0.6 + 0.6
            _link(ng, ctype, pf.inputs[0])
            pf.inputs[1].default_value = -0.6
            pf.inputs[2].default_value = 0.6
            pec_flat_socket = pf.outputs["Value"]
            pec_sag_socket = grp_in.outputs["Chest Sag"]
            chest_size_socket = grp_in.outputs["Chest Size"]
        except (KeyError, RuntimeError):
            pec_flat_socket = None
            pec_sag_socket = None
            chest_size_socket = None

    # Glute Size + Glute Sag + Glute Spacing master sockets (pelvis only).
    glute_size_socket = None
    glute_sag_socket = None
    glute_spacing_socket = None
    glute_height_socket = None
    if "Glute_L" in muscle_specs and "Glute_R" in muscle_specs:
        try:
            glute_size_socket = grp_in.outputs["Glute Size"]
            glute_sag_socket = grp_in.outputs["Glute Sag"]
            glute_spacing_socket = grp_in.outputs["Glute Spacing"]
            glute_height_socket = grp_in.outputs["Glute Height"]
        except (KeyError, RuntimeError):
            glute_size_socket = None
            glute_sag_socket = None
            glute_spacing_socket = None
            glute_height_socket = None

    # Build per-muscle displacement chains
    disp_outputs = []
    y_off = -700
    for muscle_name, base_dir in muscle_specs.items():
        archetype_socket = grp_in.outputs[f"{muscle_name} Archetype"]
        flat_override = None
        # Abs + semilunaris track the master Ab Definition slider (semilunaris is
        # an ab-definition feature — fades under fat, sharp on the fit).
        if ab_def_socket is not None and (muscle_name.startswith("Ab")
                                          or muscle_name.startswith("Semilunaris")):
            ab_mul = _new_node(ng, 'ShaderNodeMath', (-1600, y_off - 360),
                               f"{muscle_name} ×AbDef")
            ab_mul.operation = 'MULTIPLY'
            _link(ng, archetype_socket, ab_mul.inputs[0])
            _link(ng, ab_def_socket, ab_mul.inputs[1])
            archetype_socket = ab_mul.outputs["Value"]
        # Abs: soft↔cube driven by the one master Ab Flat socket.
        if ab_flat_socket is not None and muscle_name.startswith("Ab"):
            flat_override = ab_flat_socket
        # Pecs: flatness from Chest Type, sag from Chest Sag, mass from Chest Size.
        sag_override = None
        if pec_flat_socket is not None and muscle_name in ("Pec_L", "Pec_R"):
            flat_override = pec_flat_socket
            sag_override = pec_sag_socket
            if chest_size_socket is not None:
                cs_mul = _new_node(ng, 'ShaderNodeMath', (-1600, y_off - 360),
                                   f"{muscle_name} ×ChestSize")
                cs_mul.operation = 'MULTIPLY'
                _link(ng, archetype_socket, cs_mul.inputs[0])
                _link(ng, chest_size_socket, cs_mul.inputs[1])
                archetype_socket = cs_mul.outputs["Value"]
        # Glutes: mass from Glute Size, perky↔saggy from Glute Sag,
        # up/down from Glute Height.
        axial_off = None
        if glute_size_socket is not None and muscle_name in ("Glute_L", "Glute_R"):
            sag_override = glute_sag_socket
            axial_off = glute_height_socket
            gs_mul = _new_node(ng, 'ShaderNodeMath', (-1600, y_off - 360),
                               f"{muscle_name} ×GluteSize")
            gs_mul.operation = 'MULTIPLY'
            _link(ng, archetype_socket, gs_mul.inputs[0])
            _link(ng, glute_size_socket, gs_mul.inputs[1])
            archetype_socket = gs_mul.outputs["Value"]
        # Wire Chest Spacing into Pec_L/R, Glute Spacing into Glute_L/R.
        dx_mult = None
        if muscle_name in ("Pec_L", "Pec_R"):
            dx_mult = chest_spacing_socket
        elif muscle_name in ("Glute_L", "Glute_R"):
            dx_mult = glute_spacing_socket
        disp_node = _build_muscle_displacement(
            ng, grp_in, muscle_name, base_dir,
            grp_in.outputs["Limb Side"], captured_t_socket,
            normal_attr, archetype_socket, y_off,
            dx_mult_socket=dx_mult, flat_socket=flat_override,
            sag_socket=sag_override, axial_offset_socket=axial_off,
        )
        disp_outputs.append(disp_node.outputs["Vector"])
        y_off -= 400

    # Sum all displacements (chain of ADD vectors)
    if len(disp_outputs) == 0:
        # No muscles — use zero displacement (pure capsule shape)
        zero_node = _new_node(ng, 'ShaderNodeCombineXYZ', (800, -700), "zero disp")
        zero_node.inputs["X"].default_value = 0.0
        zero_node.inputs["Y"].default_value = 0.0
        zero_node.inputs["Z"].default_value = 0.0
        sum_socket = zero_node.outputs["Vector"]
    elif len(disp_outputs) == 1:
        sum_socket = disp_outputs[0]
    else:
        sum_socket = disp_outputs[0]
        for i, d in enumerate(disp_outputs[1:], 1):
            add_n = _new_node(ng, 'ShaderNodeVectorMath', (800 + i*150, -700), f"sum{i}")
            add_n.operation = 'ADD'
            _link(ng, sum_socket, add_n.inputs[0])
            _link(ng, d, add_n.inputs[1])
            sum_socket = add_n.outputs["Vector"]

    set_pos = _new_node(ng, 'GeometryNodeSetPosition', (1300, 200), "apply muscles")
    _link(ng, curve_to_mesh.outputs["Mesh"], set_pos.inputs["Geometry"])

    pos_attr = _new_node(ng, 'GeometryNodeInputPosition', (1100, -200), "pos")
    add_pos = _new_node(ng, 'ShaderNodeVectorMath', (1200, 0), "pos+disp")
    add_pos.operation = 'ADD'
    _link(ng, pos_attr.outputs["Position"], add_pos.inputs[0])
    _link(ng, sum_socket, add_pos.inputs[1])
    _link(ng, add_pos.outputs["Vector"], set_pos.inputs["Position"])

    set_shade = _new_node(ng, 'GeometryNodeSetShadeSmooth', (1500, 200), "flat")
    set_shade.domain = 'FACE'
    set_shade.inputs["Shade Smooth"].default_value = False
    _link(ng, set_pos.outputs["Geometry"], set_shade.inputs["Mesh"])

    _link(ng, set_shade.outputs["Geometry"], grp_out.inputs["Geometry"])

    return ng


# ============================================================
# Public group builders (one per limb type)
# ============================================================

def ensure_upperarm_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_LimbAnatomy_UpperArm",
        UPPERARM_MUSCLES, UPPERARM_DEFAULTS,
        start_radius=0.050, end_radius=0.030, default_n_axial=16,
        force_rebuild=force_rebuild,
    )


def ensure_forearm_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_LimbAnatomy_Forearm",
        FOREARM_MUSCLES, FOREARM_DEFAULTS,
        start_radius=0.038, end_radius=0.022, default_n_axial=14,
        force_rebuild=force_rebuild,
    )


def ensure_thigh_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_LimbAnatomy_Thigh",
        THIGH_MUSCLES, THIGH_DEFAULTS,
        start_radius=0.090, end_radius=0.055, default_n_axial=18,
        default_n_sides=12,         # 2:1 with the pelvis 6-side hex leg socket — facets align at the hip
        default_fill_caps=True,     # closed solid — segmented part, overlaps the pelvis socket
        curved=True,                # leg bends: leaves the hip socket splayed, knee curves under
        force_rebuild=force_rebuild,
    )


def ensure_calf_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_LimbAnatomy_Calf",
        CALF_MUSCLES, CALF_DEFAULTS,
        start_radius=0.058, end_radius=0.032, default_n_axial=14,
        force_rebuild=force_rebuild,
    )


def ensure_torso_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_LimbAnatomy_Torso",
        TORSO_MUSCLES, TORSO_DEFAULTS,
        start_radius=0.10,   # wide at waist (P0 = low Z)
        end_radius=0.13,     # wider at shoulder/clavicle (P1 = high Z)
        default_n_axial=21,         # 0.05 axial grid — abs land on 0.20/0.30/0.40/0.50
        default_depth_ratio=0.65,   # flatter front-back than wide for human torso
        default_n_sides=24,         # divisible by 4 → vert exactly on front/back/side centerlines
        default_front_density=1.8,  # cluster verts toward chest centerline for sharp groove
        default_neck_ratio=0.55,    # taper top to a neck opening + shoulder shelf
        force_rebuild=force_rebuild,
    )


def ensure_pelvis_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_LimbAnatomy_Pelvis",
        PELVIS_MUSCLES, PELVIS_DEFAULTS,
        start_radius=0.13,   # hip top — wide
        end_radius=0.085,    # waist — narrower
        default_n_axial=26,         # dense rings for butt/undercarriage shaping
        default_depth_ratio=1.0,    # D-shape profile carries the front-back form
        default_n_sides=24,         # enough rear verts for round cheeks + cleft
        default_front_density=1.9,  # cluster verts toward the rear centerline
        default_density_target=1.5707963,   # +pi/2 = posterior (glute cleft)
        default_fill_caps=True,     # close the undercarriage + top (solid volume)
        force_rebuild=force_rebuild,
    )


def ensure_knee_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_Joint_Knee",
        KNEE_MUSCLES, KNEE_DEFAULTS,
        start_radius=0.055, end_radius=0.045,
        default_n_axial=8,
        force_rebuild=force_rebuild,
    )


def ensure_elbow_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_Joint_Elbow",
        ELBOW_MUSCLES, ELBOW_DEFAULTS,
        start_radius=0.032, end_radius=0.028,
        default_n_axial=8,
        force_rebuild=force_rebuild,
    )


def ensure_wrist_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_Joint_Wrist",
        WRIST_MUSCLES, WRIST_DEFAULTS,
        start_radius=0.022, end_radius=0.020,
        default_n_axial=6,
        force_rebuild=force_rebuild,
    )


def ensure_ankle_anatomy_group(force_rebuild=False):
    return _build_anatomy_group(
        "BD_Joint_Ankle",
        ANKLE_MUSCLES, ANKLE_DEFAULTS,
        start_radius=0.030, end_radius=0.038,   # narrows then widens at heel
        default_n_axial=8,
        force_rebuild=force_rebuild,
    )


def ensure_breast_anatomy_group(force_rebuild=False):
    """Breast: tiny teardrop capsule. Shape from radius taper only — no peaks.
    Default radii are the AVERAGE size; archetype scales via Size socket below.
    """
    return _build_anatomy_group(
        "BD_Breast",
        BREAST_MUSCLES, BREAST_DEFAULTS,
        start_radius=0.008,    # 8mm at chest-plane root
        end_radius=0.045,      # 45mm at nipple end (average breast)
        default_n_axial=8,
        default_depth_ratio=0.85,   # slight side-flatten
        force_rebuild=force_rebuild,
    )


def ensure_pec_anatomy_group(force_rebuild=False):
    """Pec slab: wider/flatter than breast. For male chest."""
    return _build_anatomy_group(
        "BD_Pec",
        PEC_MUSCLES, PEC_DEFAULTS,
        start_radius=0.020,    # 20mm at sternum-side root
        end_radius=0.055,      # 55mm at lateral edge
        default_n_axial=6,
        default_depth_ratio=0.40,   # very flat (pec is wide+thin, not round)
        force_rebuild=force_rebuild,
    )


# ============================================================
# Object creation helper
# ============================================================

LIMB_TYPE_TO_BUILDER = {
    "upperarm": (ensure_upperarm_anatomy_group, UPPERARM_ARCHETYPES, "BD_LimbAnatomy_UpperArm"),
    "forearm":  (ensure_forearm_anatomy_group, FOREARM_ARCHETYPES,   "BD_LimbAnatomy_Forearm"),
    "thigh":    (ensure_thigh_anatomy_group, THIGH_ARCHETYPES,       "BD_LimbAnatomy_Thigh"),
    "calf":     (ensure_calf_anatomy_group, CALF_ARCHETYPES,         "BD_LimbAnatomy_Calf"),
    "torso_f":  (ensure_torso_anatomy_group, TORSO_ARCHETYPES_F,     "BD_LimbAnatomy_Torso"),
    "torso_m":  (ensure_torso_anatomy_group, TORSO_ARCHETYPES_M,     "BD_LimbAnatomy_Torso"),
    "pelvis_f": (ensure_pelvis_anatomy_group, PELVIS_ARCHETYPES_F,   "BD_LimbAnatomy_Pelvis"),
    "pelvis_m": (ensure_pelvis_anatomy_group, PELVIS_ARCHETYPES_M,   "BD_LimbAnatomy_Pelvis"),
    "knee":     (ensure_knee_anatomy_group,   KNEE_ARCHETYPES,       "BD_Joint_Knee"),
    "elbow":    (ensure_elbow_anatomy_group,  ELBOW_ARCHETYPES,      "BD_Joint_Elbow"),
    "wrist":    (ensure_wrist_anatomy_group,  WRIST_ARCHETYPES,      "BD_Joint_Wrist"),
    "ankle":    (ensure_ankle_anatomy_group,  ANKLE_ARCHETYPES,      "BD_Joint_Ankle"),
    "breast":   (ensure_breast_anatomy_group, BREAST_ARCHETYPES,     "BD_Breast"),
    "pec":      (ensure_pec_anatomy_group,    PEC_ARCHETYPES,        "BD_Pec"),
}


def create_anatomy_object(limb_type, name, p0, p1,
                          start_radius=None, end_radius=None,
                          limb_side=-1.0, archetype="average"):
    """Create a limb anatomy object + endpoint empties.

    limb_type: "upperarm" | "forearm" | "thigh" | "calf"
    archetype: "average" | "athlete" | "bodybuilder" | "anime" (and "runner"
               for thigh/calf per leg doc).

    Returns: (obj, p0_empty, p1_empty)
    """
    if limb_type not in LIMB_TYPE_TO_BUILDER:
        raise ValueError(f"Unknown limb_type {limb_type!r}; expected one of "
                         f"{list(LIMB_TYPE_TO_BUILDER)}")
    builder, arch_table, group_name = LIMB_TYPE_TO_BUILDER[limb_type]
    ng = builder()
    arch = arch_table.get(archetype, arch_table["average"])

    p0_obj = bpy.data.objects.new(f"{name}_P0", None)
    p0_obj.empty_display_type = 'SPHERE'; p0_obj.empty_display_size = 0.02
    p0_obj.location = p0
    bpy.context.collection.objects.link(p0_obj)

    p1_obj = bpy.data.objects.new(f"{name}_P1", None)
    p1_obj.empty_display_type = 'SPHERE'; p1_obj.empty_display_size = 0.02
    p1_obj.location = p1
    bpy.context.collection.objects.link(p1_obj)

    me = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)

    mod = obj.modifiers.new("BD_LimbAnatomy", 'NODES')
    mod.node_group = ng

    sid = {s.name: s.identifier for s in ng.interface.items_tree
           if s.item_type == 'SOCKET' and s.in_out == 'INPUT'}

    mod[sid["Start Empty"]] = p0_obj
    mod[sid["End Empty"]]   = p1_obj
    if start_radius is not None: mod[sid["Start Radius"]] = start_radius
    if end_radius is not None:   mod[sid["End Radius"]]   = end_radius
    mod[sid["Limb Side"]] = limb_side
    for key, scale in arch.items():
        if key.startswith("_"):
            # "_Chest Spacing" → routes to the "Chest Spacing" socket directly
            sock = key[1:]
            if sock in sid:
                mod[sid[sock]] = scale
        else:
            akey = f"{key} Archetype"
            if akey in sid:
                mod[sid[akey]] = scale

    return obj, p0_obj, p1_obj


# ============================================================
# Meta-layer — apply presets to a live torso object
# ============================================================

def _torso_mod_sid(obj):
    """Return (modifier, {socket_name: identifier}) for a torso anatomy object."""
    mod = obj.modifiers[0]
    ng = mod.node_group
    sid = {s.name: s.identifier for s in ng.interface.items_tree
           if s.item_type == 'SOCKET' and s.in_out == 'INPUT'}
    return mod, sid


def apply_torso_shape(obj, shape_name):
    """Apply a body-SHAPE preset (silhouette: V / hourglass / round / …) to a
    live torso object. Orthogonal to the muscle archetype."""
    preset = TORSO_SHAPE_PRESETS.get(shape_name)
    if preset is None:
        raise ValueError(f"Unknown shape {shape_name!r}; "
                         f"expected one of {list(TORSO_SHAPE_PRESETS)}")
    mod, sid = _torso_mod_sid(obj)
    for sock, val in preset.items():
        if sock in sid:
            mod[sid[sock]] = val
    obj.update_tag()
    return preset


def apply_torso_archetype(obj, archetype_name, gender="f"):
    """Re-apply a body-TYPE archetype (athletic / curvy / heavyset / …) to a
    live torso object — same routing as create_anatomy_object."""
    table = TORSO_ARCHETYPES_F if gender == "f" else TORSO_ARCHETYPES_M
    arch = table.get(archetype_name, table["average"])
    mod, sid = _torso_mod_sid(obj)
    for key, scale in arch.items():
        if key.startswith("_"):
            sock = key[1:]
            if sock in sid:
                mod[sid[sock]] = scale
        else:
            akey = f"{key} Archetype"
            if akey in sid:
                mod[sid[akey]] = scale
    obj.update_tag()
    return arch


def reset_torso_defaults(obj):
    """Rewrite every raw muscle socket to its TORSO_DEFAULTS value — the
    'Reset to Defaults' the add-on panel needs (undoes hand-cranked values)."""
    mod, sid = _torso_mod_sid(obj)
    for name, vals in TORSO_DEFAULTS.items():
        axial, amount, sigma, conc = vals[:4]
        flat = vals[4] if len(vals) > 4 else 0.0
        for suffix, v in (("Axial", axial), ("Amount", amount),
                          ("Sigma", sigma), ("Conc", conc), ("Flat", flat)):
            key = f"{name} {suffix}"
            if key in sid:
                mod[sid[key]] = v
    obj.update_tag()


def read_anatomy_values(obj, names=None):
    """Snapshot the current socket values off a live anatomy object.

    Dial a look in by hand, then call this to capture it — the returned dict
    can be pasted straight into a preset table. `names` optionally filters to
    a subset of socket names (e.g. the master sliders only)."""
    mod, sid = _torso_mod_sid(obj)
    out = {}
    for sock_name, ident in sid.items():
        if names is not None and sock_name not in names:
            continue
        try:
            val = mod[ident]
        except (KeyError, TypeError):
            continue
        if isinstance(val, (int, float)):
            out[sock_name] = round(float(val), 4)
    return out


# ============================================================
# COMPOSITE — fuse GN body parts into one watertight mesh (iter 27)
# ============================================================
# A swept-tube primitive cannot make the crotch Y-junction (1 pelvis -> 2
# legs), so the lower body is a *composite*: the pelvis + both thighs are
# fused and the crotch / perineum forms at the 3-way junction.
#
# Boolean solvers choke on these organic swept tubes (EXACT returns empty,
# FLOAT leaks holes). Instead we voxel-remesh the joined parts: the remesh
# computes the combined occupancy directly — any point inside ANY part is
# solid, overlaps counted once — which IS the union, and is watertight by
# construction. Every part must be a CLOSED solid first (Fill Caps on).
#
# The composite is a baked SHAPE PROXY (the iter 28 shrinkwrap target), not
# a control surface. Re-run build_lower_composite() after tuning the live
# pelvis / thigh sliders to regenerate it.

LOWER_BODY_PARTS = ["Pelvis", "Thigh_L", "Thigh_R"]


def _ensure_join_group(group_name, part_names):
    """(Re)build a GN group that joins the evaluated geometry of the named
    scene objects into one mesh (Object Info -> Join Geometry)."""
    ng = bpy.data.node_groups.get(group_name)
    if ng:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nout = ng.nodes.new('NodeGroupOutput'); nout.location = (400, 0)
    join = ng.nodes.new('GeometryNodeJoinGeometry'); join.location = (150, 0)
    for i, pname in enumerate(part_names):
        obj = bpy.data.objects.get(pname)
        if obj is None:
            continue
        oi = ng.nodes.new('GeometryNodeObjectInfo'); oi.location = (-300, i * -200)
        oi.transform_space = 'RELATIVE'
        oi.inputs["Object"].default_value = obj
        ng.links.new(oi.outputs["Geometry"], join.inputs["Geometry"])
    ng.links.new(join.outputs["Geometry"], nout.inputs["Geometry"])
    return ng


def _largest_shell_only(mesh, min_shell_verts=50):
    """Strip every connected component smaller than min_shell_verts (stray
    interior voxels the remesher leaves behind), recalc normals outward."""
    import bmesh
    bm = bmesh.new(); bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    seen = set(); shells = []
    for v in bm.verts:
        if v.index in seen:
            continue
        cc = []; stack = [v]
        while stack:
            c = stack.pop()
            if c.index in seen:
                continue
            seen.add(c.index); cc.append(c)
            for e in c.link_edges:
                o = e.other_vert(c)
                if o.index not in seen:
                    stack.append(o)
        shells.append(cc)
    shells.sort(key=len, reverse=True)
    killed = 0
    for s in shells[1:]:
        if len(s) < min_shell_verts:
            for v in s:
                bm.verts.remove(v)
            killed += 1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh); bm.free()
    return killed


def build_composite(part_names, name="Body_Composite", voxel_size=0.007,
                    min_shell_verts=50, collection=None):
    """Fuse several GN body-part objects into ONE watertight baked mesh.

    Joins each part's evaluated (post-GN) geometry, voxel-remeshes the union,
    bakes the result and strips stray interior voxel shells. Returns the
    baked composite object. Every part must be a CLOSED manifold solid
    (enable Fill Caps on its GN group) or the remesh occupancy is undefined.
    """
    old = bpy.data.objects.get(name)
    if old is not None:
        md = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if md is not None and md.users == 0:
            bpy.data.meshes.remove(md)

    seed = bpy.data.meshes.new(f"{name}_mesh")
    comp = bpy.data.objects.new(name, seed)
    (collection or bpy.context.collection).objects.link(comp)

    ng = _ensure_join_group(f"BD_Join_{name}", part_names)
    gn = comp.modifiers.new(f"BD_Join_{name}", 'NODES'); gn.node_group = ng
    rm = comp.modifiers.new("Remesh", 'REMESH')
    rm.mode = 'VOXEL'; rm.voxel_size = voxel_size
    rm.adaptivity = 0.0; rm.use_smooth_shade = True

    comp.update_tag()
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(comp.evaluated_get(deps))
    comp.modifiers.clear()
    comp.data = baked
    if seed.users == 0:
        bpy.data.meshes.remove(seed)

    _largest_shell_only(baked, min_shell_verts=min_shell_verts)
    return comp


def build_lower_composite(name="Lower_Composite", voxel_size=0.007):
    """iter 27 — fuse Pelvis + both Thighs into one watertight lower-body
    mesh. The crotch / perineum forms at the 3-way junction. Re-run after
    tuning the pelvis / thigh sliders to regenerate the proxy."""
    return build_composite(LOWER_BODY_PARTS, name=name, voxel_size=voxel_size)


# ============================================================
# PELVIS SOCKET CAP — controlled-topology leg-socket builder (iter 27e)
#
# Spec: Characters/Docs/pelvis_socket_cap_spec.md
# Replaces the GN pelvis flat bottom cap with two 6-side hex leg sockets +
# perineum saddle. Quad-dominant (only 2 crotch tris + 12 Band-B tris +
# 2 perineum tris). Thigh 12-side plugs into socket 6-side at 2:1.
# ============================================================

import math as _math


def _measure_pelvis_rim(pelvis_obj):
    """Bake GN pelvis, find flat bottom n-gon cap, return rim geometry.

    Returns dict:
      rim_co   : list[Vector] world-space, 24 verts sorted CCW by XY angle
      centroid : (cx, cy)
      rim_z    : mean Z of rim verts
      socket_z : rim_z - 0.045
    """
    pelvis_obj.update_tag()
    deps = bpy.context.evaluated_depsgraph_get()
    em = bpy.data.meshes.new_from_object(pelvis_obj.evaluated_get(deps))
    wm = pelvis_obj.matrix_world
    wv = [wm @ v.co for v in em.vertices]

    cap_face = None
    cap_z = float('inf')
    for f in em.polygons:
        if len(f.vertices) >= 10:
            cz = sum(wv[i].z for i in f.vertices) / len(f.vertices)
            if cz < cap_z:
                cap_z, cap_face = cz, f

    if cap_face is None:
        bpy.data.meshes.remove(em)
        raise RuntimeError("build_pelvis_socket_cap: no n-gon bottom cap on Pelvis")

    rim = [wv[i].copy() for i in cap_face.vertices]
    bpy.data.meshes.remove(em)

    cx = sum(v.x for v in rim) / len(rim)
    cy = sum(v.y for v in rim) / len(rim)
    rim_z = sum(v.z for v in rim) / len(rim)
    rim.sort(key=lambda v: _math.atan2(v.y - cy, v.x - cx))
    return dict(rim_co=rim, centroid=(cx, cy), rim_z=rim_z, socket_z=rim_z - 0.045)


def _measure_socket_radius(thigh_obj, socket_xy, socket_z, tol=0.020):
    """Mean radius of thigh evaluated-mesh verts within tol of socket_z."""
    thigh_obj.update_tag()
    deps = bpy.context.evaluated_depsgraph_get()
    em = bpy.data.meshes.new_from_object(thigh_obj.evaluated_get(deps))
    wm = thigh_obj.matrix_world
    sx, sy = socket_xy
    radii = []
    for v in em.vertices:
        wv = wm @ v.co
        if abs(wv.z - socket_z) <= tol:
            radii.append(_math.sqrt((wv.x - sx) ** 2 + (wv.y - sy) ** 2))
    bpy.data.meshes.remove(em)
    return (sum(radii) / len(radii)) if radii else 0.0745


def build_pelvis_socket_cap(pelvis_obj=None):
    """Replace the GN pelvis flat bottom cap with two 6-side hex leg sockets
    + perineum saddle. Controlled quad-dominant topology. Returns
    Pelvis_Socketed object. Repeatable — safe to re-run after slider changes.

    Topology:
      Band A:    Rim(24) → Mid(2×12)   pants-junction bridge (1 tri + 11 quads per side)
      Band B:    Mid(12) → Hex(6)      2:1 reduction (1 tri + 1 quad per hex edge)
      Perineum:  inner arcs bridged via PF/PB verts lifted +10 mm above rim
    """
    import bmesh
    from mathutils import Vector

    if pelvis_obj is None:
        pelvis_obj = bpy.data.objects["Pelvis"]

    # ── 1. Measure runtime inputs ──────────────────────────────────────────
    m = _measure_pelvis_rim(pelvis_obj)
    rim_co = m['rim_co']
    cx, cy = m['centroid']
    rim_z  = m['rim_z']
    sock_z = m['socket_z']

    p0_l = bpy.data.objects["Thigh_L_P0"].location
    p0_r = bpy.data.objects["Thigh_R_P0"].location
    sock_xy_l = (float(p0_l.x), float(p0_l.y))
    sock_xy_r = (float(p0_r.x), float(p0_r.y))

    thigh_l = bpy.data.objects.get("Thigh_L")
    sock_r = (_measure_socket_radius(thigh_l, sock_xy_l, sock_z)
              if thigh_l else 0.0745)

    mid_z  = rim_z - 0.020   # mid-circle Z (outer verts, toward socket)
    mid_r  = sock_r * 0.85   # mid-circle radius (~63 mm)
    arch_z = rim_z + 0.022   # perineum arch peak — inner mid verts + ctr lifted here
    peri_z = rim_z + 0.030   # pf/pb arch peaks, slightly above arch_z

    # ── 2. Bake pelvis body, remove bottom cap face ────────────────────────
    pelvis_obj.update_tag()
    deps = bpy.context.evaluated_depsgraph_get()
    body_mesh = bpy.data.meshes.new_from_object(pelvis_obj.evaluated_get(deps))
    body_mesh.transform(pelvis_obj.matrix_world)

    bm_body = bmesh.new()
    bm_body.from_mesh(body_mesh)
    bm_body.faces.ensure_lookup_table()
    cap_f = min(
        (f for f in bm_body.faces if len(f.verts) >= 10),
        key=lambda f: sum(v.co.z for v in f.verts) / len(f.verts),
        default=None,
    )
    if cap_f:
        bmesh.ops.delete(bm_body, geom=[cap_f], context='FACES_ONLY')
    bm_body.to_mesh(body_mesh)
    bm_body.free()

    # ── 3. Build cap geometry ──────────────────────────────────────────────
    bm = bmesh.new()

    # 3a. Rim verts (24, sorted CCW by XY angle from centroid)
    rim_v = [bm.verts.new(co) for co in rim_co]

    # 3b. rim_F (front-centre: min |X|, Y < cy) and rim_B (back-centre: Y >= cy)
    front_v = [v for v in rim_v if v.co.y < cy]
    back_v  = [v for v in rim_v if v.co.y >= cy]
    rim_F = min(front_v, key=lambda v: abs(v.co.x))
    rim_B = min(back_v,  key=lambda v: abs(v.co.x))
    fi = rim_v.index(rim_F)
    bi = rim_v.index(rim_B)
    N  = len(rim_v)

    def _ccw_arc(start_i, end_i):
        """Collect rim_v CCW from start_i to end_i inclusive."""
        arc = []
        i = start_i
        while True:
            arc.append(rim_v[i])
            if i == end_i:
                break
            i = (i + 1) % N
        return arc

    arc_L = _ccw_arc(fi, bi)  # 13 verts: rim_F → rim_B via X>0 (left/char-left)
    arc_R = _ccw_arc(bi, fi)  # 13 verts: rim_B → rim_F via X<0 (right)

    # 3c. Mid-circles (2×12) and hex sockets (2×6).
    # Phase 0 → vert[0] at +X from each socket centre, matching the GN Mesh Circle
    # used by the thigh (also starts at +X).  2:1: hex[k] aligns with thigh verts 2k/2k+1.

    def _make_circle(sx, sy, z, r, n, phase):
        verts = []
        for k in range(n):
            a = phase + k * (2.0 * _math.pi / n)
            verts.append(bm.verts.new(Vector((sx + r * _math.cos(a),
                                              sy + r * _math.sin(a), z))))
        return verts

    mid_L = _make_circle(*sock_xy_l, mid_z, mid_r, 12, 0.0)
    mid_R = _make_circle(*sock_xy_r, mid_z, mid_r, 12, 0.0)

    # Hex sockets: same phase → guaranteed 2:1 vert alignment with mid
    hex_L = _make_circle(*sock_xy_l, sock_z, sock_r, 6, 0.0)
    hex_R = _make_circle(*sock_xy_r, sock_z, sock_r, 6, 0.0)

    # Lift inner-facing mid verts toward arch_z (cosine falloff from inner out).
    # mid_L[6] = 180° = innermost toward centre; mid_R[0] = 0° = innermost.
    # This curves the floor of the perineum saddle upward like the hammock in
    # the reference (Belt/Butt/Pelvis parts): sockets are the lowest points,
    # arch peaks between them.
    def _arch_lift(verts, inner_idx, n):
        for i, v in enumerate(verts):
            d = min((i - inner_idx) % n, (inner_idx - i) % n)
            t = max(0.0, _math.cos(d * _math.pi / (n // 2)))  # 1 at inner, 0 at outer
            v.co.z = mid_z + (arch_z - mid_z) * t

    _arch_lift(mid_L, 6, 12)
    _arch_lift(mid_R, 0, 12)

    # Perineum arch verts: front/back centreline, above arch_z
    pf = bm.verts.new(Vector((cx, cy - 0.005, peri_z)))  # front arch peak
    pb = bm.verts.new(Vector((cx, cy + 0.005, peri_z)))  # back  arch peak

    bm.verts.ensure_lookup_table()

    # ── 4. Band A: arc(13) → mid(12) — 1 tri at arc[0] + 11 quads ─────────
    def _bridge_13_to_12(arc, mid):
        bm.faces.new([arc[0], arc[1], mid[0]])               # crotch fan tri
        for i in range(1, 12):
            bm.faces.new([arc[i], arc[i + 1], mid[i], mid[i - 1]])

    _bridge_13_to_12(arc_L, mid_L)   # tri at rim_F
    _bridge_13_to_12(arc_R, mid_R)   # tri at rim_B

    # ── 5. Band B: mid(12) → hex(6) — 2:1 reduction ───────────────────────
    # hex[k] ↔ mid[2k]/mid[2k+1]: 1 tri + 1 quad per hex edge
    def _bridge_12_to_6(mid, hex_v):
        for k in range(6):
            m0 = mid[2 * k]
            m1 = mid[2 * k + 1]
            m2 = mid[(2 * k + 2) % 12]
            h0 = hex_v[k]
            h1 = hex_v[(k + 1) % 6]
            bm.faces.new([h0, m1, m0])         # tri
            bm.faces.new([h0, h1, m2, m1])     # quad

    _bridge_12_to_6(mid_L, hex_L)
    _bridge_12_to_6(mid_R, hex_R)

    # ── 6. Perineum saddle ────────────────────────────────────────────────
    # With ph=0: mid_L[6] sits at 180° from sock_L (inner-most L vert, facing centre);
    # mid_R[0] sits at 0° from sock_R (inner-most R vert, facing centre).
    # Every intra-circle edge is already owned by Band A (above) and Band B (below),
    # so the saddle uses ONLY cross-gap edges + edges to pf/pb.
    # A centre vert bridges the 48 mm gap → 4 tris, fully manifold.
    ctr = bm.verts.new(Vector((cx, cy, arch_z)))
    bm.faces.new([mid_L[6], pf,  ctr])         # L front tri
    bm.faces.new([pf,  mid_R[0], ctr])         # R front tri
    bm.faces.new([mid_L[6], ctr, pb])          # L back  tri
    bm.faces.new([ctr,  mid_R[0], pb])         # R back  tri

    # ── 7. Write cap mesh ─────────────────────────────────────────────────
    cap_mesh = bpy.data.meshes.new("_PelvisCapTmp")
    bm.to_mesh(cap_mesh)
    bm.free()

    # ── 8. Join body + cap; weld shared rim verts ─────────────────────────
    bm2 = bmesh.new()
    bm2.from_mesh(body_mesh)
    bm2.from_mesh(cap_mesh)
    # Merge the duplicate rim verts where cap meets body (same world coords)
    bm2.verts.ensure_lookup_table()
    bmesh.ops.remove_doubles(bm2, verts=bm2.verts[:], dist=0.0002)
    # Recalc normals on the merged mesh — the body's normals seed the flood fill
    # so cap faces are oriented consistently with the body outward-pointing normals
    bm2.faces.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces[:])
    final = bpy.data.meshes.new("Pelvis_Socketed_mesh")
    bm2.to_mesh(final)
    bm2.free()
    bpy.data.meshes.remove(body_mesh)
    bpy.data.meshes.remove(cap_mesh)

    # ── 9. Create / replace Pelvis_Socketed object ─────────────────────────
    old = bpy.data.objects.get("Pelvis_Socketed")
    if old:
        old_me = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_me and old_me.users == 0:
            bpy.data.meshes.remove(old_me)

    obj = bpy.data.objects.new("Pelvis_Socketed", final)
    bpy.context.collection.objects.link(obj)
    obj.display_type = 'SOLID'

    # Unify normals outward in edit mode — select all first so it covers cap faces
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    return obj


# ============================================================
# Hips — alternative pelvis (fresh approach, prototype)
# ============================================================

def build_hips(name="Hips", **overrides):
    """Prototype alternative to the swept-tube pelvis.

    The swept tube is the wrong primitive: it has ONE axis, a flat cap, and
    the legs get bolted on afterward. A real hip region BIFURCATES — one form
    at the waist that splits into two legs, with a rounded crotch between.

    This builds the hips as a continuous lofted shell that splits into two
    INTEGRAL leg-socket tubes:

        waist ─┐
        iliac  ├─ hip body : 4 stacked D-shaped rings (24v), quad-bridged
        hip    │
        rim  ──┘
            │   pants split : rim(24) → leg_L top(24) + leg_R top(24)
            │                 + 2 natural crotch poles (front/back tris)
        leg_L tube     leg_R tube   : round 24v tubes pointing down
            └ crotch saddle ┘       : inseam-to-inseam bridge, arched up

    Why this beats the tube:
      * legs are PART of the mesh — leg sockets are integral, not a cap
      * the underside is round leg-tube wall + an arched crotch saddle;
        there is no flat n-gon cap anywhere
      * all-quad except the 2 anatomical crotch poles

    Parametric + repeatable — re-run to regenerate. Returns the Hips object.
    Leaves the existing Pelvis / BD_LimbAnatomy_Pelvis untouched.
    """
    import bmesh, math
    from mathutils import Vector
    TAU = 2.0 * math.pi

    P = dict(
        n=24,
        # ── vertical levels (Z) ──
        waist_z=0.965, iliac_z=0.908, hip_z=0.856, rim_z=0.812,
        crotch_z=0.793, legmid_z=0.728, legbot_z=0.650,
        # ── hip body D-rings: (half_width X, anterior depth Y, posterior depth Y) ──
        waist_dim=(0.098, 0.075, 0.088),
        iliac_dim=(0.150, 0.092, 0.140),
        hip_dim=(0.156, 0.104, 0.198),
        rim_dim=(0.140, 0.090, 0.152),
        # ── legs ── leg outer wall = the rim arc itself (shared verts, no shelf).
        # leg_cx/leg_r_* only shape the inseam + the tube below the rim.
        leg_cx=0.075, leg_cy=0.020,
        inseam_rx=0.062, inseam_ry=0.066,
        leg_r_mid=0.064, leg_r_bot=0.056,
        # ── crotch saddle ──
        crotch_lift=0.040,    # how high the inner crotch arches above crotch_z
        cleft=0.020,          # glute cleft depth (back-centre inward nudge)
    )
    P.update(overrides)
    n = P['n']
    half = n // 2

    # ── replace any prior Hips object ──────────────────────────────────────
    old = bpy.data.objects.get(name)
    if old:
        old_me = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_me and old_me.users == 0:
            bpy.data.meshes.remove(old_me)

    bm = bmesh.new()

    def smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def dring(z, dim):
        """D-shaped hip ring, n verts. vert 0 = front-centre (-Y), going CCW;
        vert n/2 = back-centre (+Y). Posterior half uses the deeper radius."""
        hw, ant, post = dim
        vs = []
        for k in range(n):
            a = -math.pi / 2.0 + k * TAU / n
            ca, sa = math.cos(a), math.sin(a)
            depth = ant + (post - ant) * smoothstep((sa + 1.0) * 0.5)
            vs.append(bm.verts.new(Vector((hw * ca, depth * sa, z))))
        return vs

    def leg_ring(cx, z, r, direction):
        """Plain n-vert circle for a leg tube below the rim. vert 0 = front (-Y).
        direction +1 → vert n/4 lands on +X (outer for the +X leg);
        direction -1 → vert n/4 lands on -X (outer for the -X leg)."""
        vs = []
        for k in range(n):
            a = -math.pi / 2.0 + direction * k * TAU / n
            vs.append(bm.verts.new(Vector((cx + r * math.cos(a),
                                           P['leg_cy'] + r * math.sin(a), z))))
        return vs

    def bridge(ra, rb):
        """Quad-bridge two equal closed rings (same length, same winding)."""
        m = len(ra)
        for k in range(m):
            bm.faces.new([ra[k], ra[(k + 1) % m], rb[(k + 1) % m], rb[k]])

    # ── 1. Hip body: 4 stacked D-rings ─────────────────────────────────────
    waist = dring(P['waist_z'], P['waist_dim'])
    iliac = dring(P['iliac_z'], P['iliac_dim'])
    hip   = dring(P['hip_z'],   P['hip_dim'])
    rim   = dring(P['rim_z'],   P['rim_dim'])

    # mild glute cleft — pull the back-centre verts inward (−Y)
    cl = P['cleft']
    hip[half].co.y -= cl
    rim[half].co.y -= cl * 0.7
    hip[half - 1].co.y -= cl * 0.4
    hip[half + 1].co.y -= cl * 0.4

    bridge(waist, iliac)
    bridge(iliac, hip)
    bridge(hip, rim)

    # ── 2. Pants split — legTop rings REUSE the rim arc verts (no shelf) ────
    # The outer wall of each leg IS the rim arc — same verts — so the hip
    # surface flows straight into the thigh. Only the inner crotch contour
    # (inseam) is new geometry. legTop loop = rim arc (13, shared) + inseam
    # (11, new), 24 verts total — a slanted ring (outer high, inner low).
    #   legTop_L outer arc = rim[0..half]            (front → +X → back)
    #   legTop_R outer arc = rim[0], rim[n-1..half]  (front → −X → back)
    def make_legtop(side):
        cx = side * P['leg_cx']
        if side > 0:
            outer = [rim[k] for k in range(0, half + 1)]
        else:
            outer = [rim[0]] + [rim[k] for k in range(n - 1, half - 1, -1)]
        inseam = []
        for k in range(half + 1, n):                # n/2 − 1 = 11 new verts
            a = -math.pi / 2.0 + side * k * TAU / n
            inseam.append(bm.verts.new(Vector((
                cx + P['inseam_rx'] * math.cos(a),
                P['leg_cy'] + P['inseam_ry'] * math.sin(a),
                P['crotch_z']))))
        return outer + inseam, inseam               # full 24-loop, inseam list

    legTop_L, inseam_L = make_legtop(+1)
    legTop_R, inseam_R = make_legtop(-1)

    # ── 3. Crotch — arch the inseam up, saddle-bridge L↔R, 2 poles ──────────
    # arch the inner crotch (peak mid-contour)
    m_in = len(inseam_L)                            # 11
    for j in range(m_in):
        s = math.sin(math.pi * (j + 0.5) / m_in)    # bump, 0 at ends → 1 mid
        dz = P['crotch_lift'] * s
        inseam_L[j].co.z += dz
        inseam_R[j].co.z += dz

    # saddle: bridge inseam_L ↔ inseam_R (both run back → front)
    for j in range(m_in - 1):
        bm.faces.new([inseam_L[j], inseam_L[j + 1],
                      inseam_R[j + 1], inseam_R[j]])
    # 2 crotch poles — the shared rim centre vert fans to both inseam ends
    bm.faces.new([rim[half], inseam_L[0],  inseam_R[0]])    # back pole
    bm.faces.new([rim[0],    inseam_R[-1], inseam_L[-1]])   # front pole

    # ── 4. Leg tubes — each leg-top continues down as a round tube ──────────
    legMid_L = leg_ring(+P['leg_cx'], P['legmid_z'], P['leg_r_mid'], +1)
    legBot_L = leg_ring(+P['leg_cx'], P['legbot_z'], P['leg_r_bot'], +1)
    legMid_R = leg_ring(-P['leg_cx'], P['legmid_z'], P['leg_r_mid'], -1)
    legBot_R = leg_ring(-P['leg_cx'], P['legbot_z'], P['leg_r_bot'], -1)

    bridge(legTop_L, legMid_L); bridge(legMid_L, legBot_L)
    bridge(legTop_R, legMid_R); bridge(legMid_R, legBot_R)

    # ── 5. Waist cap (n-gon). Leg bottoms stay open — sockets. ──────────────
    bm.faces.new(list(reversed(waist)))

    # ── 6. Finalise ────────────────────────────────────────────────────────
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    obj.display_type = 'SOLID'
    return obj


# ============================================================
# BD_Hips — pure-GN pelvis part (segmented body, V leg-sockets)
# ============================================================

def ensure_hips_group(force_rebuild=False):
    """Pure-GN pelvis part for the segmented body.

    Fresh GN setup — NOT a swept tube. Base primitive is a UV sphere, so the
    form is rounded everywhere: there is no flat bottom cap, the underside is
    naturally round. The sphere is GN-deformed into hip proportions:
      * Hip Width / Height                — overall scale
      * Anterior / Posterior Depth        — the D-shape curve (back deeper)
      * Glute Size / Glute Drop           — tunable buttock bulge
      * Waist Open                        — opens the top for the torso seam

    The two thighs are SEPARATE GN parts (BD_LimbAnatomy_Thigh) that plug into
    the lower-sides at hip-joint empties angled into a V — segmented body.

    Returns the BD_Hips GeometryNodeTree.
    """
    name = "BD_Hips"
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

    fin("Hip Width",      1.30, 0.40, 2.50)
    fin("Height",         0.78, 0.30, 1.60)
    fin("Anterior Depth", 0.78, 0.20, 1.60)   # front-back D-shape — front
    fin("Posterior Depth",1.34, 0.50, 2.80)   # ── back (glute side, deeper)
    fin("Glute Size",     1.40, 0.00, 5.00)   # buttock bulge master
    fin("Glute Drop",     0.24, 0.00, 1.00)   # how low the glute mass sits
    fin("Cheek Spread",   0.62, 0.10, 1.60)   # how far apart the 2 cheeks sit
    fin("Cleft Depth",    0.65, 0.00, 1.60)   # centre groove between cheeks
    fin("Waist Open",     0.42, 0.00, 0.85)   # top opening for the torso seam
    fin("Radius",         0.150, 0.02, 0.50)
    iin("Segments",       24, 6, 64)
    iin("Rings",          18, 4, 48)
    # ── leg sockets — angled stub collars on the lower-sides (the V) ──
    # The thigh (separate object) butts its top ring against the stub's open
    # end. Socket Radius MUST match the thigh Start Radius so the joint reads
    # continuous ("thickness wise they meet up").
    fin("Socket Radius", 0.078, 0.02, 0.20)
    fin("Socket Depth",  0.150, 0.03, 0.45)
    fin("Socket Splay",  15.0, 0.0, 60.0)    # degrees the socket tilts outward
    fin("Socket X",      0.140, 0.0, 0.40)   # local X of the socket centre
    fin("Socket Z",     -0.030, -0.30, 0.30) # local Z of the socket centre
    iin("Socket Sides",  12, 3, 32)

    gi = _new_node(ng, 'NodeGroupInput',  (-1900, 0))
    go = _new_node(ng, 'NodeGroupOutput', (1900, 0))

    sph = _new_node(ng, 'GeometryNodeMeshUVSphere', (-1600, 220), "sphere")
    _link(ng, gi.outputs["Segments"], sph.inputs["Segments"])
    _link(ng, gi.outputs["Rings"],    sph.inputs["Rings"])
    _link(ng, gi.outputs["Radius"],   sph.inputs["Radius"])

    pos = _new_node(ng, 'GeometryNodeInputPosition', (-1600, -180), "pos")
    sep = _new_node(ng, 'ShaderNodeSeparateXYZ', (-1420, -180), "sep")
    _link(ng, pos.outputs["Position"], sep.inputs["Vector"])

    # normalized direction  n = pos / Radius   (components in −1..1)
    def divr(val_socket, y, lbl):
        d = _new_node(ng, 'ShaderNodeMath', (-1240, y), lbl)
        d.operation = 'DIVIDE'
        _link(ng, val_socket, d.inputs[0])
        _link(ng, gi.outputs["Radius"], d.inputs[1])
        return d.outputs["Value"]
    nx = divr(sep.outputs["X"], -100, "nx")
    ny = divr(sep.outputs["Y"], -180, "ny")
    nz = divr(sep.outputs["Z"], -260, "nz")

    # D-shape: depth_fac = smoothstep-lerp(Anterior, Posterior) over (ny+1)/2
    tnode = _new_node(ng, 'ShaderNodeMath', (-1040, -180), "d t")
    tnode.operation = 'MULTIPLY_ADD'
    _link(ng, ny, tnode.inputs[0])
    tnode.inputs[1].default_value = 0.5
    tnode.inputs[2].default_value = 0.5
    dfac = _new_node(ng, 'ShaderNodeMapRange', (-880, -180), "depth")
    dfac.interpolation_type = 'SMOOTHSTEP'
    dfac.clamp = True
    _link(ng, tnode.outputs["Value"], dfac.inputs["Value"])
    dfac.inputs["From Min"].default_value = 0.0
    dfac.inputs["From Max"].default_value = 1.0
    _link(ng, gi.outputs["Anterior Depth"],  dfac.inputs["To Min"])
    _link(ng, gi.outputs["Posterior Depth"], dfac.inputs["To Max"])

    # scaled ellipsoid position
    sx = _new_node(ng, 'ShaderNodeMath', (-700, -60), "x'")
    sx.operation = 'MULTIPLY'
    _link(ng, sep.outputs["X"], sx.inputs[0])
    _link(ng, gi.outputs["Hip Width"], sx.inputs[1])
    sy = _new_node(ng, 'ShaderNodeMath', (-700, -180), "y'")
    sy.operation = 'MULTIPLY'
    _link(ng, sep.outputs["Y"], sy.inputs[0])
    _link(ng, dfac.outputs["Result"], sy.inputs[1])
    sz = _new_node(ng, 'ShaderNodeMath', (-700, -300), "z'")
    sz.operation = 'MULTIPLY'
    _link(ng, sep.outputs["Z"], sz.inputs[0])
    _link(ng, gi.outputs["Height"], sz.inputs[1])
    base = _new_node(ng, 'ShaderNodeCombineXYZ', (-520, -180), "base")
    _link(ng, sx.outputs["Value"], base.inputs["X"])
    _link(ng, sy.outputs["Value"], base.inputs["Y"])
    _link(ng, sz.outputs["Value"], base.inputs["Z"])

    # ── glute — TWO cheeks + a centre cleft ────────────────────────────────
    # Each cheek is a radial bulge whose selector is aimed posterior-down and
    # offset in X by ±Cheek Spread, so two distinct masses form. The cleft is
    # a negative groove carved down the posterior centreline between them.
    nvec = _new_node(ng, 'ShaderNodeCombineXYZ', (-1100, 80), "nvec")
    _link(ng, nx, nvec.inputs["X"])
    _link(ng, ny, nvec.inputs["Y"])
    _link(ng, nz, nvec.inputs["Z"])

    gdn = _new_node(ng, 'ShaderNodeMath', (-1100, 320), "-drop")
    gdn.operation = 'MULTIPLY'
    _link(ng, gi.outputs["Glute Drop"], gdn.inputs[0])
    gdn.inputs[1].default_value = -1.0

    def cheek_mask(side, yb):
        # cheek selector direction = normalize(side*Cheek Spread, 1, -Drop)
        csx = _new_node(ng, 'ShaderNodeMath', (-920, yb + 90), f"cheek x{side}")
        csx.operation = 'MULTIPLY'
        _link(ng, gi.outputs["Cheek Spread"], csx.inputs[0])
        csx.inputs[1].default_value = float(side)
        cdir = _new_node(ng, 'ShaderNodeCombineXYZ', (-760, yb + 90), f"cheek d{side}")
        _link(ng, csx.outputs["Value"], cdir.inputs["X"])
        cdir.inputs["Y"].default_value = 1.0
        _link(ng, gdn.outputs["Value"], cdir.inputs["Z"])
        cdn = _new_node(ng, 'ShaderNodeVectorMath', (-600, yb + 90), f"cheek n{side}")
        cdn.operation = 'NORMALIZE'
        _link(ng, cdir.outputs["Vector"], cdn.inputs[0])
        cdot = _new_node(ng, 'ShaderNodeVectorMath', (-440, yb), f"cheek dot{side}")
        cdot.operation = 'DOT_PRODUCT'
        _link(ng, nvec.outputs["Vector"], cdot.inputs[0])
        _link(ng, cdn.outputs["Vector"], cdot.inputs[1])
        cmx = _new_node(ng, 'ShaderNodeMath', (-280, yb), f"cheek max{side}")
        cmx.operation = 'MAXIMUM'
        _link(ng, cdot.outputs["Value"], cmx.inputs[0])
        cmx.inputs[1].default_value = 0.0
        cpw = _new_node(ng, 'ShaderNodeMath', (-120, yb), f"cheek pow{side}")
        cpw.operation = 'POWER'
        _link(ng, cmx.outputs["Value"], cpw.inputs[0])
        cpw.inputs[1].default_value = 3.0
        return cpw.outputs["Value"]

    mL = cheek_mask(+1, 380)
    mR = cheek_mask(-1, 200)
    cheeks = _new_node(ng, 'ShaderNodeMath', (40, 290), "cheeks sum")
    cheeks.operation = 'ADD'
    _link(ng, mL, cheeks.inputs[0])
    _link(ng, mR, cheeks.inputs[1])

    # cleft groove — gaussian on |nx| (near centreline) × posterior weight
    cfx = _new_node(ng, 'ShaderNodeMath', (-440, 30), "cleft nx/s")
    cfx.operation = 'DIVIDE'
    _link(ng, nx, cfx.inputs[0])
    cfx.inputs[1].default_value = 0.22
    cfsq = _new_node(ng, 'ShaderNodeMath', (-280, 30), "cleft sq")
    cfsq.operation = 'POWER'
    _link(ng, cfx.outputs["Value"], cfsq.inputs[0])
    cfsq.inputs[1].default_value = 2.0
    cfng = _new_node(ng, 'ShaderNodeMath', (-120, 30), "cleft neg")
    cfng.operation = 'MULTIPLY'
    _link(ng, cfsq.outputs["Value"], cfng.inputs[0])
    cfng.inputs[1].default_value = -1.0
    cfg = _new_node(ng, 'ShaderNodeMath', (40, 30), "cleft exp")
    cfg.operation = 'EXPONENT'
    _link(ng, cfng.outputs["Value"], cfg.inputs[0])
    cpo = _new_node(ng, 'ShaderNodeMath', (-120, -110), "cleft post")
    cpo.operation = 'MAXIMUM'
    _link(ng, ny, cpo.inputs[0])
    cpo.inputs[1].default_value = 0.0
    cpo2 = _new_node(ng, 'ShaderNodeMath', (40, -110), "cleft post^2")
    cpo2.operation = 'POWER'
    _link(ng, cpo.outputs["Value"], cpo2.inputs[0])
    cpo2.inputs[1].default_value = 2.0
    cleft = _new_node(ng, 'ShaderNodeMath', (200, -40), "cleft mask")
    cleft.operation = 'MULTIPLY'
    _link(ng, cfg.outputs["Value"], cleft.inputs[0])
    _link(ng, cpo2.outputs["Value"], cleft.inputs[1])

    # net radial amount = cheeks (out) − cleft (in)
    ch_amt = _new_node(ng, 'ShaderNodeMath', (200, 290), "cheek amt")
    ch_amt.operation = 'MULTIPLY'
    _link(ng, cheeks.outputs["Value"], ch_amt.inputs[0])
    _link(ng, gi.outputs["Glute Size"], ch_amt.inputs[1])
    cl_amt = _new_node(ng, 'ShaderNodeMath', (360, -40), "cleft amt")
    cl_amt.operation = 'MULTIPLY'
    _link(ng, cleft.outputs["Value"], cl_amt.inputs[0])
    _link(ng, gi.outputs["Cleft Depth"], cl_amt.inputs[1])
    net = _new_node(ng, 'ShaderNodeMath', (520, 160), "net amt")
    net.operation = 'SUBTRACT'
    _link(ng, ch_amt.outputs["Value"], net.inputs[0])
    _link(ng, cl_amt.outputs["Value"], net.inputs[1])
    netscl = _new_node(ng, 'ShaderNodeMath', (520, 20), "net*scl")
    netscl.operation = 'MULTIPLY'
    _link(ng, net.outputs["Value"], netscl.inputs[0])
    netscl.inputs[1].default_value = 0.060

    gdisp = _new_node(ng, 'ShaderNodeVectorMath', (520, -140), "g disp")
    gdisp.operation = 'SCALE'
    _link(ng, nvec.outputs["Vector"], gdisp.inputs[0])
    _link(ng, netscl.outputs["Value"], gdisp.inputs["Scale"])

    finalpos = _new_node(ng, 'ShaderNodeVectorMath', (520, -280), "final pos")
    finalpos.operation = 'ADD'
    _link(ng, base.outputs["Vector"], finalpos.inputs[0])
    _link(ng, gdisp.outputs["Vector"], finalpos.inputs[1])

    setp = _new_node(ng, 'GeometryNodeSetPosition', (680, 220), "set pos")
    _link(ng, sph.outputs["Mesh"], setp.inputs["Geometry"])
    _link(ng, finalpos.outputs["Vector"], setp.inputs["Position"])

    # waist hole — delete the top faces (normalized z above 1 − Waist Open)
    thr = _new_node(ng, 'ShaderNodeMath', (300, 420), "waist thr")
    thr.operation = 'SUBTRACT'
    thr.inputs[0].default_value = 1.0
    _link(ng, gi.outputs["Waist Open"], thr.inputs[1])
    selz = _new_node(ng, 'ShaderNodeMath', (480, 420), "nz>thr")
    selz.operation = 'GREATER_THAN'
    _link(ng, nz, selz.inputs[0])
    _link(ng, thr.outputs["Value"], selz.inputs[1])
    dele = _new_node(ng, 'GeometryNodeDeleteGeometry', (880, 220), "waist hole")
    dele.domain = 'FACE'
    _link(ng, setp.outputs["Geometry"], dele.inputs["Geometry"])
    _link(ng, selz.outputs["Value"], dele.inputs["Selection"])

    # ── leg socket stubs — two angled collar cylinders on the lower-sides ───
    # Each stub is a short open cylinder, tilted outward by Socket Splay, its
    # open outer end is the socket the thigh butts into. Joined to the body.
    def build_stub(side, yoff):
        cyl = _new_node(ng, 'GeometryNodeMeshCylinder', (300, yoff), f"stub {side}")
        cyl.fill_type = 'NONE'
        _link(ng, gi.outputs["Socket Sides"],  cyl.inputs["Vertices"])
        _link(ng, gi.outputs["Socket Radius"], cyl.inputs["Radius"])
        _link(ng, gi.outputs["Socket Depth"],  cyl.inputs["Depth"])
        # rotation: Y by (pi - side*splay) → tilt the +Z axis down-and-outward
        srad = _new_node(ng, 'ShaderNodeMath', (300, yoff - 130), f"splay rad {side}")
        srad.operation = 'MULTIPLY'
        _link(ng, gi.outputs["Socket Splay"], srad.inputs[0])
        srad.inputs[1].default_value = 0.0174532925
        ang = _new_node(ng, 'ShaderNodeMath', (460, yoff - 130), f"ang {side}")
        ang.operation = 'MULTIPLY_ADD'       # srad * (-side) + pi
        _link(ng, srad.outputs["Value"], ang.inputs[0])
        ang.inputs[1].default_value = -float(side)
        ang.inputs[2].default_value = 3.14159265
        rot = _new_node(ng, 'ShaderNodeCombineXYZ', (620, yoff - 130), f"rot {side}")
        rot.inputs["X"].default_value = 0.0
        _link(ng, ang.outputs["Value"], rot.inputs["Y"])
        rot.inputs["Z"].default_value = 0.0
        # translation: (side*Socket X, 0, Socket Z)
        sxm = _new_node(ng, 'ShaderNodeMath', (300, yoff - 250), f"socket x {side}")
        sxm.operation = 'MULTIPLY'
        _link(ng, gi.outputs["Socket X"], sxm.inputs[0])
        sxm.inputs[1].default_value = float(side)
        tr = _new_node(ng, 'ShaderNodeCombineXYZ', (620, yoff - 250), f"tr {side}")
        _link(ng, sxm.outputs["Value"], tr.inputs["X"])
        tr.inputs["Y"].default_value = 0.0
        _link(ng, gi.outputs["Socket Z"], tr.inputs["Z"])
        xf = _new_node(ng, 'GeometryNodeTransform', (820, yoff), f"stub xf {side}")
        _link(ng, cyl.outputs["Mesh"], xf.inputs["Geometry"])
        _link(ng, tr.outputs["Vector"], xf.inputs["Translation"])
        _link(ng, rot.outputs["Vector"], xf.inputs["Rotation"])
        return xf.outputs["Geometry"]

    stub_L = build_stub(+1, -240)
    stub_R = build_stub(-1, -560)

    joing = _new_node(ng, 'GeometryNodeJoinGeometry', (1180, 160), "join body+stubs")
    _link(ng, dele.outputs["Geometry"], joing.inputs["Geometry"])
    _link(ng, stub_L, joing.inputs["Geometry"])
    _link(ng, stub_R, joing.inputs["Geometry"])

    shade = _new_node(ng, 'GeometryNodeSetShadeSmooth', (1420, 160), "flat")
    shade.domain = 'FACE'
    shade.inputs["Shade Smooth"].default_value = False
    # Kit standard: BD_Hips outputs a CLOSED blob — no pre-made waist hole, no
    # collar stubs. The waist + the two 45 deg leg-socket V seams are produced
    # downstream by the standard plane cutter (cut_to_kit_planes /
    # segment_character), identical to the rico fallback path. The waist-delete
    # (dele) and the collar stubs (joing) are left in the graph but bypassed.
    _link(ng, setp.outputs["Geometry"], shade.inputs["Mesh"])
    _link(ng, shade.outputs["Geometry"], go.inputs["Geometry"])
    return ng


# ============================================================
# Character-kit cutter — the STANDARD 3-plane segmentation
# ============================================================

# The kit standard. Any body (rico fallback mesh OR a baked GN body) cut on
# these planes yields interchangeable, swappable parts.
KIT_WAIST_FRAC = 0.61    # waist plane Z, as a fraction of mesh height
KIT_APEX_FRAC  = 0.475   # crotch apex Z, as a fraction of mesh height
KIT_SPLAY_DEG  = 45.0    # leg-socket planes — degrees from vertical (the V)
KIT_GROIN_FRAC = 0.060   # groin-spacer width between the legs, frac of height


def cut_to_kit_planes(body_obj, waist_frac=KIT_WAIST_FRAC,
                      apex_frac=KIT_APEX_FRAC, splay_deg=KIT_SPLAY_DEG,
                      groin_frac=KIT_GROIN_FRAC, prefix="Kit"):
    """Cut a body / hips mesh into closed, swappable character-kit parts on the
    standard planes:

        * waist  — horizontal, at waist_frac of the mesh height
        * leg L  — splayed plane, splay_deg from vertical, offset +groin
        * leg R  — mirror of leg L, offset -groin

    The two leg planes are offset apart by `groin_frac` x height (the GROIN
    SPACER) instead of meeting at a point — so the groin / sacrum strip between
    the legs stays on the Hips part and the legs sit a small distance apart.
    groin_frac is procedurally adjustable.

    Every mesh cut on these planes produces interchangeable parts. Each part is
    bisected clean (flat planar seams) and capped closed. Works on the rico
    fallback body AND a baked procedural GN body. Returns {role: object}.
    """
    import bmesh, math

    # ── bake the source mesh in world space ────────────────────────────────
    deps = bpy.context.evaluated_depsgraph_get()
    tmp = bpy.data.meshes.new_from_object(body_obj.evaluated_get(deps))
    tmp.transform(body_obj.matrix_world)

    zs = [v.co.z for v in tmp.vertices]
    zmin, zmax = min(zs), max(zs)
    h = zmax - zmin
    waist_z = zmin + waist_frac * h
    apex_z  = zmin + apex_frac * h
    s = 0.5 * groin_frac * h                  # half the groin-spacer width
    th = math.radians(splay_deg)
    nL = (-math.cos(th), 0.0, math.sin(th))   # left  socket plane normal
    nR = ( math.cos(th), 0.0, math.sin(th))   # right socket plane normal

    bm = bmesh.new()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)

    # ── bisect along the standard planes (cut only, keep both sides) ────────
    # 45 deg leg planes through (±s, 0, apex_z) form the V socket panels;
    # vertical inseam planes at x = ±s hold a CONSTANT-width groin spacer so
    # the legs sit a fixed small distance apart (the groin/sacrum strip stays
    # on the Hips part the whole way down).
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0003)
    for co, no in (((0, 0, waist_z), (0, 0, 1)),
                   (( s, 0, apex_z), nL),
                   ((-s, 0, apex_z), nR),
                   (( s, 0, 0.0), (1, 0, 0)),
                   ((-s, 0, 0.0), (1, 0, 0))):
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                               dist=1e-5, plane_co=co, plane_no=no,
                               clear_inner=False, clear_outer=False)
    bm.faces.ensure_lookup_table()

    # ── classify every face (exact — bisect put an edge on each plane) ─────
    def faceclass(f):
        c = f.calc_center_median()
        if c.z > waist_z + 1e-6:
            return 0                                          # torso
        dL = (c.x - s) * nL[0] + (c.z - apex_z) * nL[2]
        dR = (c.x + s) * nR[0] + (c.z - apex_z) * nR[2]
        # Hips = the wedge above both V planes, OR the central groin strip
        # (|x| <= s) — that strip is the groin/sacrum spacer between the legs.
        if (dL >= -1e-6 and dR >= -1e-6) or abs(c.x) <= s + 1e-6:
            return 1                                          # hips (+ groin)
        return 2 if c.x > s else 3                            # leg L / R

    fcls = {f: faceclass(f) for f in bm.faces}

    # ── extract each part into its own mesh (pure bmesh — no operators) ────
    namemap = {0: f"{prefix}_Torso", 1: f"{prefix}_Hips",
               2: f"{prefix}_Leg_L", 3: f"{prefix}_Leg_R"}
    rolemap = {0: 'torso', 1: 'hips', 2: 'leg_L', 3: 'leg_R'}
    parts = {}
    for cval in (0, 1, 2, 3):
        cfaces = [f for f in bm.faces if fcls[f] == cval]
        if not cfaces:
            continue
        nb = bmesh.new()
        vmap = {}
        for f in cfaces:
            nv = []
            for v in f.verts:
                if v not in vmap:
                    vmap[v] = nb.verts.new(v.co)
                nv.append(vmap[v])
            try:
                nb.faces.new(nv)
            except ValueError:
                pass                                          # duplicate face
        # cap the cut openings closed
        nb.edges.ensure_lookup_table()
        bnd = [e for e in nb.edges if len(e.link_faces) == 1]
        if bnd:
            try:
                bmesh.ops.holes_fill(nb, edges=bnd, sides=0)
            except Exception:
                pass
        nb.normal_update()
        bmesh.ops.recalc_face_normals(nb, faces=nb.faces[:])
        nm = bpy.data.meshes.new(namemap[cval])
        nb.to_mesh(nm)
        nb.free()
        for p in nm.polygons:                                 # hard-body flat
            p.use_smooth = False
        o = bpy.data.objects.new(namemap[cval], nm)
        bpy.context.collection.objects.link(o)
        parts[rolemap[cval]] = o
    bm.free()
    return parts


# ============================================================
# BODY-WIDE TRAIT SYSTEM — character-creator layer
# ============================================================
# High-level trait sliders that each drive many sockets across ALL body
# parts at once (the "video-game character creator" layer). For every
# trait-driven socket:
#     final = TRAIT_BASELINE[part][socket] + Σ(trait_value × delta)
# Sockets no trait touches are left alone — the raw per-socket sliders stay
# live for fine-tuning (see memory keep-raw-sliders).
#
# Adding a trait = add one BODY_TRAITS entry + one BODY_TRAIT_SLIDERS row;
# the BDB panel picks it up automatically.

_PART_OF_GROUP = {
    "BD_LimbAnatomy_Torso": "torso",
    "BD_LimbAnatomy_Pelvis": "pelvis",
    "BD_LimbAnatomy_Thigh": "thigh",
    "BD_LimbAnatomy_Calf": "calf",
    "BD_LimbAnatomy_UpperArm": "upperarm",
    "BD_LimbAnatomy_Forearm": "forearm",
}

# Neutral character — every trait at 0. Authored clean: archetype scalars at
# 1.0 (the muscle's designed default amount), masters at sensible mid values.
# Re-snapshot a hand-tuned neutral with capture_trait_baseline().
TRAIT_BASELINE = {
    "torso": {
        "Ab Definition": 0.3, "Chest Size": 0.5, "Chest Sag": 0.10,
        "Chest Type": 0.5, "Waist": 1.0, "Shoulder Width": 1.05, "Ribcage": 1.0,
        "Belly Archetype": 1.0, "LineaAlba Archetype": 1.0,
        "SemilunarisL Archetype": 1.0, "SemilunarisR Archetype": 1.0,
        "Lat_L Archetype": 1.0, "Lat_R Archetype": 1.0,
        "Trapezius Archetype": 1.0, "Pec_L Archetype": 1.0, "Pec_R Archetype": 1.0,
    },
    "pelvis": {
        "Glute Size": 1.5, "Glute Sag": 0.15, "Hip Width": 1.10,
        "Glute_L Archetype": 1.0, "Glute_R Archetype": 1.0,
    },
    "thigh": {"VastusLat Archetype": 1.0, "VastusMed Archetype": 1.0,
              "RectusFem Archetype": 1.0, "Hamstring Archetype": 1.0,
              "Glute Archetype": 1.0},
    "calf": {"Gastroc Archetype": 1.0, "Soleus Archetype": 1.0,
             "TibialisAnt Archetype": 1.0},
    "upperarm": {"Deltoid Archetype": 1.0, "Bicep Archetype": 1.0,
                 "Brachialis Archetype": 1.0, "Tricep Archetype": 1.0},
    "forearm": {"Brachiorad Archetype": 1.0, "Extensors Archetype": 1.0,
                "Flexors Archetype": 1.0},
}

# trait -> {part: {socket: delta-per-unit-trait}}. Every socket referenced
# here MUST exist in TRAIT_BASELINE or it is silently skipped.
BODY_TRAITS = {
    "Muscular": {
        "thigh": {"VastusLat Archetype": 1.6, "VastusMed Archetype": 1.3,
                  "RectusFem Archetype": 1.4, "Hamstring Archetype": 1.5,
                  "Glute Archetype": 0.8},
        "calf": {"Gastroc Archetype": 1.7, "Soleus Archetype": 1.2,
                 "TibialisAnt Archetype": 0.7},
        "upperarm": {"Deltoid Archetype": 1.6, "Bicep Archetype": 1.9,
                     "Brachialis Archetype": 1.2, "Tricep Archetype": 1.7},
        "forearm": {"Brachiorad Archetype": 1.3, "Extensors Archetype": 1.1,
                    "Flexors Archetype": 1.1},
        "torso": {"Ab Definition": 0.7, "Lat_L Archetype": 1.3,
                  "Lat_R Archetype": 1.3, "Trapezius Archetype": 1.1,
                  "Pec_L Archetype": 0.7, "Pec_R Archetype": 0.7},
        "pelvis": {"Glute_L Archetype": 0.5, "Glute_R Archetype": 0.5},
    },
    "Abs": {"torso": {"Ab Definition": 1.7, "LineaAlba Archetype": 0.6,
                      "SemilunarisL Archetype": 0.5,
                      "SemilunarisR Archetype": 0.5}},
    "Body Fat": {"torso": {"Belly Archetype": 1.8, "Waist": 0.5,
                           "Ab Definition": -0.6},
                 "pelvis": {"Glute Size": 1.2, "Hip Width": 0.2}},
    "Breast Size": {"torso": {"Chest Size": 2.0}},
    "Butt Size": {"pelvis": {"Glute Size": 3.0}},
    "Pregnancy": {"torso": {"Belly Archetype": 3.0, "Waist": 0.3}},
    "Age": {"torso": {"Chest Sag": 0.5, "Ab Definition": -0.4},
            "pelvis": {"Glute Sag": 0.5}},
    "Sag": {"torso": {"Chest Sag": 0.7}, "pelvis": {"Glute Sag": 0.7}},
    "Curvy": {"torso": {"Waist": -0.35},
              "pelvis": {"Hip Width": 0.45, "Glute Size": 1.2}},
    # Frame: bipolar  -1 narrow .. +1 wide
    "Frame": {"torso": {"Shoulder Width": 0.25, "Ribcage": 0.20},
              "pelvis": {"Hip Width": 0.30}},
    # Gender: bipolar  -1 male .. +1 female
    "Gender": {"torso": {"Chest Type": 0.5, "Chest Size": 0.7,
                         "Shoulder Width": -0.12, "Waist": -0.15},
               "pelvis": {"Hip Width": 0.25, "Glute Size": 1.0}},
}

# (name, min, max) — drives the panel sliders + value clamps. Order = UI order.
BODY_TRAIT_SLIDERS = [
    ("Muscular", 0.0, 1.0), ("Body Fat", 0.0, 1.0), ("Abs", 0.0, 1.0),
    ("Breast Size", 0.0, 1.0), ("Butt Size", 0.0, 1.0), ("Pregnancy", 0.0, 1.0),
    ("Age", 0.0, 1.0), ("Sag", 0.0, 1.0), ("Curvy", 0.0, 1.0),
    ("Frame", -1.0, 1.0), ("Gender", -1.0, 1.0),
]


def _trait_socket_id_map(node_group):
    return {it.name: it.identifier for it in node_group.interface.items_tree
            if getattr(it, 'in_out', None) == 'INPUT' and it.item_type == 'SOCKET'}


def _anatomy_parts(scene=None):
    """All anatomy part objects in the scene → {part_type: [(obj, modifier), …]}."""
    scene = scene or bpy.context.scene
    out = {}
    for o in scene.objects:
        if o.type != 'MESH':
            continue
        for m in o.modifiers:
            if (m.type == 'NODES' and m.node_group is not None
                    and m.node_group.name in _PART_OF_GROUP):
                out.setdefault(_PART_OF_GROUP[m.node_group.name], []).append((o, m))
    return out


def apply_body_traits(trait_values, scene=None):
    """Drive every trait-controlled socket across all body parts:
        final = TRAIT_BASELINE + Σ(trait_value × delta).
    trait_values: {trait_name: float}. Sockets no trait touches keep their
    current values, so raw per-socket fine-tuning stays live."""
    for ptype, objs in _anatomy_parts(scene).items():
        base = TRAIT_BASELINE.get(ptype)
        if not base:
            continue
        targets = dict(base)
        for tname, tval in trait_values.items():
            if not tval:
                continue
            pmap = BODY_TRAITS.get(tname, {}).get(ptype)
            if not pmap:
                continue
            for sock, delta in pmap.items():
                if sock in targets:
                    targets[sock] += tval * delta
        for obj, mod in objs:
            sid = _trait_socket_id_map(mod.node_group)
            for sock, val in targets.items():
                ident = sid.get(sock)
                if ident is not None:
                    try:
                        mod[ident] = val
                    except Exception:
                        pass
            obj.update_tag()


def capture_trait_baseline(scene=None):
    """Snapshot the current value of every trait-driven socket → a dict in
    TRAIT_BASELINE format. Dial in a neutral character, call this, and paste
    the result over TRAIT_BASELINE to redefine 'all traits at 0'."""
    out = {}
    for ptype, objs in _anatomy_parts(scene).items():
        base = TRAIT_BASELINE.get(ptype)
        if not base or not objs:
            continue
        _obj, mod = objs[0]
        sid = _trait_socket_id_map(mod.node_group)
        out[ptype] = {s: round(float(mod[sid[s]]), 4)
                      for s in base if s in sid}
    return out
