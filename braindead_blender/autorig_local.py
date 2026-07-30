"""
BrainDeadBlender — Local AutoRig Orchestrator

Self-contained, no ComfyUI required. Wraps:
  1. autorig_bootstrap.bootstrap()       — venv + deps + MIA source clone
  2. autorig_bootstrap.ensure_mia_weights() — HF model weight download
  3. subprocess(venv_python, autorig_runner.py)   — inference (PyTorch)
  4. subprocess(blender, mia_export.py)          — Blender FBX assembly
  5. BD_MixamoToUEFN rename (in-process Python)   — UEFN bone names

Called by the BD AutoRig sidebar panel when backend = "Local".
"""

import bpy
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from . import autorig_bootstrap as _bootstrap


# Paths to the vendored Blender script + Mixamo template
_THIS_DIR = Path(__file__).resolve().parent
VENDOR_DIR = _THIS_DIR / "autorig_vendor"
MIA_EXPORT_SCRIPT = VENDOR_DIR / "mia_export.py"
MIXAMO_TEMPLATE = VENDOR_DIR / "mixamo.fbx"

# Mirror of the Mixamo→UEFN bone map from
# ComfyUI-BrainDead/nodes/autorig/bone_remap.py — kept in sync there.
MIXAMO_TO_UEFN: dict[str, str] = {
    "Hips":         "pelvis",
    "Spine":        "spine_01",
    "Spine1":       "spine_02",
    "Spine2":       "spine_03",
    "Spine3":       "spine_04",
    "Neck":         "neck_01",
    "Neck1":        "neck_02",
    "Head":         "head",
    "LeftShoulder":  "clavicle_l",
    "LeftArm":       "upperarm_l",
    "LeftForeArm":   "lowerarm_l",
    "LeftHand":      "hand_l",
    "RightShoulder": "clavicle_r",
    "RightArm":      "upperarm_r",
    "RightForeArm":  "lowerarm_r",
    "RightHand":     "hand_r",
    "LeftHandThumb1":  "thumb_01_l",
    "LeftHandThumb2":  "thumb_02_l",
    "LeftHandThumb3":  "thumb_03_l",
    "LeftHandIndex1":  "index_01_l",
    "LeftHandIndex2":  "index_02_l",
    "LeftHandIndex3":  "index_03_l",
    "LeftHandMiddle1": "middle_01_l",
    "LeftHandMiddle2": "middle_02_l",
    "LeftHandMiddle3": "middle_03_l",
    "LeftHandRing1":   "ring_01_l",
    "LeftHandRing2":   "ring_02_l",
    "LeftHandRing3":   "ring_03_l",
    "LeftHandPinky1":  "pinky_01_l",
    "LeftHandPinky2":  "pinky_02_l",
    "LeftHandPinky3":  "pinky_03_l",
    "RightHandThumb1":  "thumb_01_r",
    "RightHandThumb2":  "thumb_02_r",
    "RightHandThumb3":  "thumb_03_r",
    "RightHandIndex1":  "index_01_r",
    "RightHandIndex2":  "index_02_r",
    "RightHandIndex3":  "index_03_r",
    "RightHandMiddle1": "middle_01_r",
    "RightHandMiddle2": "middle_02_r",
    "RightHandMiddle3": "middle_03_r",
    "RightHandRing1":   "ring_01_r",
    "RightHandRing2":   "ring_02_r",
    "RightHandRing3":   "ring_03_r",
    "RightHandPinky1":  "pinky_01_r",
    "RightHandPinky2":  "pinky_02_r",
    "RightHandPinky3":  "pinky_03_r",
    "LeftUpLeg":   "thigh_l",
    "LeftLeg":     "calf_l",
    "LeftFoot":    "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg":  "thigh_r",
    "RightLeg":    "calf_r",
    "RightFoot":   "foot_r",
    "RightToeBase":"ball_r",
}


def _strip_mixamo_prefix(name: str) -> str:
    for pfx in ("mixamorig:", "Armature|", "Armature:", "Avatar:"):
        if name.startswith(pfx):
            return name[len(pfx):]
    return name


def remap_imported_to_uefn(arm_obj: bpy.types.Object) -> tuple[int, int, list[str]]:
    """Walk arm_obj's bones + every weighted mesh's vertex groups, applying
    the Mixamo→UEFN rename. Returns (renamed_bones, renamed_vgroups, unmapped).
    No external Blender process — runs in the active session."""
    if arm_obj.type != "ARMATURE":
        raise ValueError(f"remap target must be ARMATURE, got {arm_obj.type}")

    # Bones (edit mode required to rename)
    bpy.context.view_layer.objects.active = arm_obj
    try:
        bpy.ops.object.mode_set(mode="EDIT")
    except RuntimeError:
        pass

    renamed_bones = 0
    unmapped: list[str] = []
    for b in list(arm_obj.data.edit_bones):
        stripped = _strip_mixamo_prefix(b.name)
        if stripped != b.name:
            b.name = stripped
        if b.name in MIXAMO_TO_UEFN:
            tgt = MIXAMO_TO_UEFN[b.name]
            if tgt != b.name:
                b.name = tgt
                renamed_bones += 1
        elif b.name not in MIXAMO_TO_UEFN.values():
            unmapped.append(b.name)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass

    # Vertex groups on every mesh
    renamed_vgroups = 0
    for m in bpy.data.objects:
        if m.type != "MESH":
            continue
        for vg in list(m.vertex_groups):
            stripped = _strip_mixamo_prefix(vg.name)
            if stripped != vg.name:
                vg.name = stripped
            if vg.name in MIXAMO_TO_UEFN:
                tgt = MIXAMO_TO_UEFN[vg.name]
                if tgt != vg.name:
                    vg.name = tgt
                    renamed_vgroups += 1
    return renamed_bones, renamed_vgroups, unmapped


def _blender_binary() -> str:
    """Path to the currently-running Blender executable."""
    return bpy.app.binary_path


# Limb chains used by fit_bones_to_mesh. Listed root → tip so each segment's
# new tail can feed the next segment's head.
_LIMB_CHAINS = (
    ("upperarm_l", "lowerarm_l", "hand_l"),
    ("upperarm_r", "lowerarm_r", "hand_r"),
    ("thigh_l", "calf_l", "foot_l", "ball_l"),
    ("thigh_r", "calf_r", "foot_r", "ball_r"),
)


def fit_bones_to_mesh(arm: bpy.types.Object,
                          mesh: bpy.types.Object,
                          weight_threshold: float = 0.3,
                          min_extend_factor: float = 1.02) -> dict:
    """Extend each arm/leg chain bone's tail to cover its weighted mesh
    region. Compensates for MIA regressing toward its training-set
    typical-human proportions when the input mesh has non-typical
    proportions (e.g. Trellis bodies with extra-long arms).

    For each chain (clavicle → upperarm → lowerarm → hand), iterates
    root→tip. For each bone:
      - finds vertices weighted ≥ weight_threshold to that bone
      - projects them onto the current bone direction (tail - head)
      - extends the bone tail to the max forward projection (if the
        projection is min_extend_factor× longer than the current bone)
      - sets the next chain bone's head to this new tail

    Args:
        arm: the autorigged armature (post-rename to UEFN names)
        mesh: the skinned mesh
        weight_threshold: only consider verts with this much influence
        min_extend_factor: skip extension if the predicted length is
            already within (factor × current_length) of the target

    Returns:
        Dict mapping bone name → {"old_len": float, "new_len": float}
    """
    import bpy as _bpy
    from mathutils import Vector as _V

    # Get mesh-data → armature-local transform (mesh verts are typically in
    # mesh.matrix_world space; armature edit_bones are in arm.matrix_world)
    mesh_to_arm = arm.matrix_world.inverted() @ mesh.matrix_world

    # Index verts by vertex group for fast lookup
    vg_verts: dict[int, list] = {}  # group_idx → [(mesh-local Vector, weight)]
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.weight >= weight_threshold:
                vg_verts.setdefault(g.group, []).append((v.co.copy(), g.weight))

    # Enter edit mode on the armature
    _bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    _bpy.context.view_layer.objects.active = arm
    _bpy.ops.object.mode_set(mode="EDIT")

    report: dict[str, dict] = {}
    try:
        for chain in _LIMB_CHAINS:
            prev_tail = None  # track previous bone's new tail, used as next head
            for bone_name in chain:
                eb = arm.data.edit_bones.get(bone_name)
                if eb is None:
                    report[bone_name] = {"skipped": "bone missing"}
                    continue
                vg = mesh.vertex_groups.get(bone_name)
                if vg is None:
                    report[bone_name] = {"skipped": "vgroup missing"}
                    prev_tail = eb.tail.copy()
                    continue

                # If the previous bone in the chain was extended, snap head
                if prev_tail is not None:
                    eb.head = prev_tail.copy()

                head = eb.head.copy()
                tail = eb.tail.copy()
                bone_vec = tail - head
                bone_len = bone_vec.length

                # Gather weighted verts in armature-local space
                verts = vg_verts.get(vg.index, [])
                if not verts:
                    report[bone_name] = {"skipped": "no weighted verts"}
                    prev_tail = tail
                    continue
                weighted_pts = [(mesh_to_arm @ v_local, w) for v_local, w in verts]

                # Compute the WEIGHTED CENTROID of the influence region.
                # This is far more robust than relying on the bone's predicted
                # direction (which can be near-zero when MIA produces
                # degenerate bones like a 2cm lowerarm).
                total_w = sum(w for _, w in weighted_pts) + 1e-9
                centroid = _V((0, 0, 0))
                for p, w in weighted_pts:
                    centroid += p * w
                centroid /= total_w

                # New direction: head → centroid (the natural axis of the
                # weighted region). Fall back to existing bone vector only
                # if the centroid-direction is also degenerate (vert region
                # collapsed onto the head).
                to_centroid = centroid - head
                if to_centroid.length > 1e-4:
                    direction = to_centroid.normalized()
                elif bone_len > 1e-6:
                    direction = bone_vec / bone_len
                else:
                    report[bone_name] = {"skipped": "degenerate direction"}
                    prev_tail = tail
                    continue

                # Extend to the FARTHEST weighted vert along this direction
                # (so the tail reaches the tip of the influence region).
                max_proj = 0.0
                for p, _w in weighted_pts:
                    proj = (p - head).dot(direction)
                    if proj > max_proj:
                        max_proj = proj

                if max_proj < 1e-3:  # nothing meaningful to fit to
                    report[bone_name] = {"skipped": "centroid behind head"}
                    prev_tail = tail
                    continue

                new_tail = head + direction * max_proj
                old_len = bone_len if bone_len > 1e-6 else 0.0
                # Apply the new tail (always — the centroid-aim direction is
                # more reliable than MIA's predicted direction when bones
                # were near-degenerate).
                eb.tail = new_tail
                report[bone_name] = {
                    "old_len": round(old_len, 4),
                    "new_len": round(max_proj, 4),
                    "extend_factor": round(max_proj / max(old_len, 1e-3), 2),
                }
                prev_tail = new_tail
    finally:
        _bpy.ops.object.mode_set(mode="OBJECT")

    return report


# ── Joint retarget machinery (skeleton-fits-mesh pipeline) ───────────────────
#
# Everything here works on JOINT POSITIONS (bone heads). Tails and rolls are
# swung along with their joint deltas (or overridden with explicit canonical
# Z axes) but never drive mesh deformation — orientation-only changes must
# not deform skin.

def _children_map(arm):
    return {b.name: [c.name for c in b.children] for b in arm.data.bones}


def _fk_deltas(arm, old_heads, new_heads, children_of):
    """Per-bone (quat, old_head, new_head) built as a HIERARCHICAL FK
    chain: each bone's rotation is its parent's accumulated rotation
    composed with the minimal correction that lands its own joint-to-
    joint direction on the new layout. Independent per-bone minimal
    swings are NOT equivalent — adjacent bones then disagree in twist
    and the blend zones (elbows, wrists) candy-wrap and collapse.

    Also returns per-bone (axis, ratio) segment scales for proportion
    pre-conform: ratio = new / old joint-to-joint segment length along
    the bone's own direction (1.0 for leaf/no-child bones).
    """
    from mathutils import Quaternion as _Q

    def _depth(b):
        d, x = 0, b
        while x.parent is not None:
            x = x.parent; d += 1
        return d

    def primary_child(name):
        best = None
        for ch in children_of.get(name, ()):
            if ch in old_heads and ch in new_heads:
                L = (old_heads[ch] - old_heads[name]).length
                if L > 1e-5 and (best is None or L > best[0]):
                    best = (L, ch)
        return best[1] if best else None

    acc: dict = {}
    out = {}
    scales: dict = {}
    for b in sorted(arm.data.bones, key=_depth):
        name = b.name
        pname = b.parent.name if b.parent else None
        R_p = acc.get(pname, _Q())
        R = R_p
        nh = new_heads.get(name)
        ch = primary_child(name)
        if (nh is not None and ch is not None
                and name not in _INHERIT_PARENT_ROT):
            cur = R_p @ (old_heads[ch] - old_heads[name])
            nd = new_heads[ch] - nh
            if cur.length > 1e-6 and nd.length > 1e-6:
                R = (cur.normalized()
                       .rotation_difference(nd.normalized())) @ R_p
                # Twist bones ride the parent limb's scale (their own
                # segments fight the limb ratio and shred the blend zone)
                if "_twist" not in name:
                    scales[name] = (
                        (old_heads[ch] - old_heads[name]).normalized(),
                        nd.length / cur.length)
        elif (nh is not None and ch is not None
                and name in _INHERIT_PARENT_ROT):
            # Rotation-inherit bones (hands) still get a segment scale so
            # oversized character hands compress to canonical hand size —
            # finger-weighted verts ride this scaled delta rigidly.
            cur = R_p @ (old_heads[ch] - old_heads[name])
            nd = new_heads[ch] - nh
            if cur.length > 1e-6 and nd.length > 1e-6:
                scales[name] = (
                    (old_heads[ch] - old_heads[name]).normalized(),
                    nd.length / cur.length)
        acc[name] = R
        if nh is not None:
            out[name] = (R, old_heads[name].copy(), nh.copy())
    return out, scales


# Bones whose own joint-to-joint direction comes from unreliable autorig
# estimates (hand → finger joints): inherit the parent chain's rotation
# instead of computing a correction from noise. The wrist stays straight
# with the forearm — anatomically correct in both T- and A-pose.
_INHERIT_PARENT_ROT = {"hand_l", "hand_r"}


# Hands must move as RIGID units for mesh deformation: per-finger deltas
# amplify autorig finger-joint noise into shredded hand geometry. The
# finger BONES still retarget individually (rig correctness); only the
# mesh transform is unified onto the hand bone.
_RIGID_FOLLOW_MESH = {}
for _side in ("l", "r"):
    for _f in ("thumb", "index", "middle", "ring", "pinky"):
        for _i in (1, 2, 3):
            _RIGID_FOLLOW_MESH[f"{_f}_{_i:02d}_{_side}"] = f"hand_{_side}"


def retarget_joints(arm: bpy.types.Object,
                       new_heads: dict,
                       meshes=(),
                       roll_z: "dict | None" = None,
                       new_tails: "dict | None" = None) -> tuple[int, list]:
    """Snap the armature's rest joints to new_heads (armature-local) and
    move the given skinned meshes along via joint-swing linear-blend
    skinning (armature-modifier weight semantics).

    roll_z: optional {bone: armature-local Z axis} for the final roll.
    new_tails: optional {bone: armature-local tail position} — the bone's
    Y axis (head→tail) IS its animation frame, so when retargeting onto a
    canonical skeleton pass its tails explicitly; the fallback (swinging
    the bone's existing tail with the joint delta) preserves whatever
    frame the bone already had. Bones absent from new_heads are left
    untouched (identity for the mesh).
    """
    import bpy as _bpy
    import bmesh as _bmesh
    from mathutils import Matrix as _M

    old_heads = {b.name: b.head_local.copy() for b in arm.data.bones}
    old_tails = {b.name: b.tail_local.copy() for b in arm.data.bones}
    old_zs = {b.name: b.matrix_local.to_3x3().col[2].copy()
                for b in arm.data.bones}
    swings, seg_scales = _fk_deltas(arm, old_heads, new_heads,
                                    _children_map(arm))
    if meshes:
        print(f"[BD_AutoRig:retarget] swings={len(swings)} "
              f"seg_scales={len(seg_scales)} "
              f"ratios={sorted({round(r, 2) for _, r in seg_scales.values()})[:12]}",
              flush=True)

    _bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    _bpy.context.view_layer.objects.active = arm
    _bpy.ops.object.mode_set(mode="EDIT")
    moved_bones = 0
    try:
        for name, (q, oh, nh) in swings.items():
            eb = arm.data.edit_bones.get(name)
            if eb is None:
                continue
            eb.use_connect = False
            eb.head = nh
            t = new_tails.get(name) if new_tails is not None else None
            eb.tail = t if t is not None else nh + (q @ (old_tails[name] - oh))
            z = roll_z.get(name) if roll_z is not None else None
            if z is None:
                z = q @ old_zs[name]
            if z.length > 1e-8:
                eb.align_roll(z.normalized())
            moved_bones += 1
    finally:
        _bpy.ops.object.mode_set(mode="OBJECT")
    arm.data.pose_position = "POSE"

    deltas = {}
    for name, (q, oh, nh) in swings.items():
        if (nh - oh).length < 1e-6 and abs(q.angle) < 1e-5:
            continue
        d = (_M.Translation(nh) @ q.to_matrix().to_4x4()
             @ _M.Translation(-oh))
        # Proportion pre-conform: scale the bone's own segment along its
        # axis (mesh follows canonical bone lengths, not just positions).
        # Applied pre-rotation so it acts along the bone's new direction.
        ss = seg_scales.get(name)
        if ss is not None:
            axis, ratio = ss
            if abs(ratio - 1.0) > 1e-3:
                ax, ay, az = axis.x, axis.y, axis.z
                outer = _M(((ax*ax, ax*ay, ax*az),
                            (ay*ax, ay*ay, ay*az),
                            (az*ax, az*ay, az*az)))
                S = _M.Identity(3) + (ratio - 1.0) * outer
                d = (_M.Translation(nh) @ q.to_matrix().to_4x4()
                     @ S.to_4x4() @ _M.Translation(-oh))
        deltas[name] = d
    if meshes and seg_scales:
        _probe = next((n for n in ("upperarm_l", "thigh_l", "spine_03")
                       if n in deltas and n in seg_scales), None)
        if _probe is not None:
            _ch = _children_map(arm).get(_probe, [None])[0]
            if _ch:
                _out = deltas[_probe] @ old_heads[_ch]
                print(f"[BD_AutoRig:retarget] scale-probe {_probe}: "
                      f"child endpoint -> {_out.to_tuple()} "
                      f"(target {new_heads[_ch].to_tuple()}, "
                      f"ratio {seg_scales[_probe][1]:.3f})", flush=True)
    # Rigid hands: finger-weighted verts transform exactly like the hand
    for finger, hand in _RIGID_FOLLOW_MESH.items():
        if finger in swings:
            d = deltas.get(hand)
            if d is not None:
                deltas[finger] = d
            else:
                deltas.pop(finger, None)
    # Twist bones ride their parent limb bone's transform (their own
    # segment ratios contradict the limb's and candy-wrap the blend zone)
    for name in list(deltas):
        if "_twist" in name:
            p = arm.data.bones[name].parent
            if p is not None and p.name in deltas:
                deltas[name] = deltas[p.name]
            else:
                deltas.pop(name, None)

    stats = []
    for o in meshes:
        to_arm = arm.matrix_world.inverted_safe() @ o.matrix_world
        from_arm = to_arm.inverted_safe()
        gidx_delta = {vg.index: deltas[vg.name]
                        for vg in o.vertex_groups if vg.name in deltas}
        gidx_bone = {vg.index for vg in o.vertex_groups
                       if arm.data.bones.get(vg.name)}
        bm = _bmesh.new()
        bm.from_mesh(o.data)
        bm.verts.ensure_lookup_table()
        moved = 0
        for v in o.data.vertices:
            total = 0.0
            acc = None
            for g in v.groups:
                if g.group not in gidx_bone or g.weight <= 0.0:
                    continue
                total += g.weight
                d = gidx_delta.get(g.group)
                if d is not None:
                    contrib = (d @ (to_arm @ v.co)) * g.weight
                    acc = contrib if acc is None else acc + contrib
            if acc is None or total <= 0.0:
                continue
            base = to_arm @ v.co
            norm = max(total, 1.0)
            rest_part = base * (max(1.0 - total, 0.0) / norm) \
                if total < 1.0 else base * 0.0
            ident_w = total - sum(
                g.weight for g in v.groups
                if g.group in gidx_delta and g.weight > 0.0)
            new_arm = (acc + base * ident_w) / norm + rest_part
            bm.verts[v.index].co = from_arm @ new_arm
            moved += 1
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
        stats.append(f"{o.name}({moved})")
    return moved_bones, stats


# IK helper bones sit on their FK counterparts; keep that relationship
# when fitting the skeleton to a new anatomy.
_IK_FOLLOW = {"ik_hand_l": "hand_l", "ik_hand_r": "hand_r",
               "ik_hand_gun": "hand_r",
               "ik_foot_l": "foot_l", "ik_foot_r": "foot_r"}


def snap_leg_joints_to_mesh(new_heads: dict, mesh: bpy.types.Object,
                            donor_arm: bpy.types.Object,
                            sides=("l", "r"), z_window: float = 0.035):
    """Re-seat fitted leg joints INSIDE the character's leg mesh.

    MIA's leg estimates on wide-stance characters come out nearly I-posed:
    the whole chain stacked at x≈0.066 while the mesh legs center at
    x≈0.15-0.20 — knee/ankle joints float in the crotch gap, and every
    pose then pivots the leg around a point outside it. Joint HEIGHTS from
    MIA are good, so keep z and only re-center (x, y): each joint moves to
    the centroid of the character's leg-verts in a thin horizontal slab at
    its own height. The pelvis/hip anchor (thigh head) is left alone —
    the anatomical hip sits inside the pelvis, not the thigh surface.

    Modifies new_heads (donor-armature-local) in place; returns the set of
    adjusted bone names.
    """
    from mathutils import Vector
    w2d = donor_arm.matrix_world.inverted_safe()
    mw = mesh.matrix_world
    adjusted = set()
    for side in sides:
        # world-space leg verts for this side
        pts = []
        for vn in (f"thigh_{side}", f"calf_{side}", f"foot_{side}"):
            vg = mesh.vertex_groups.get(vn)
            if vg is None:
                continue
            idx = vg.index
            for v in mesh.data.vertices:
                for g in v.groups:
                    if g.group == idx and g.weight > 0.25:
                        pts.append(mw @ v.co)
                        break
        if not pts:
            continue
        for jn in (f"calf_{side}", f"foot_{side}", f"ball_{side}"):
            nh = new_heads.get(jn)
            if nh is None:
                continue
            jw = donor_arm.matrix_world @ nh
            slab = [p for p in pts if abs(p.z - jw.z) < z_window]
            print(f"[BD_AutoRig:legsnap] {jn}: jw_z={jw.z:.3f} "
                  f"pts={len(pts)} slab={len(slab)}")
            if len(slab) < 8:
                continue
            c = sum(slab, Vector()) / len(slab)
            moved = w2d @ Vector((c.x, c.y, jw.z))
            new_heads[jn] = moved
            adjusted.add(jn)
    return adjusted


def compute_fitted_donor_heads(donor_arm: bpy.types.Object,
                                   source_arm: bpy.types.Object) -> dict:
    """New armature-local joint positions that fit the donor skeleton onto
    the source (autorig) rig's anatomy — the donor keeps its hierarchy,
    bone set, and (swung) rolls, but its joints land where the character
    actually is.

    Rules per donor bone:
      - name-matched on the source rig → the source joint position
      - ik helper → follows its FK bone with the original offset
      - otherwise (twists, spine_04/05, neck_02, …) → interpolated
        between the nearest fitted ancestor and descendant (rotation +
        scale of that segment), or riding its ancestor's fitted segment
        when it has no fitted descendant (leaf twists)
      - no fitted ancestor at all (root) → unchanged
    """
    w2d = donor_arm.matrix_world.inverted_safe()
    s2w = source_arm.matrix_world
    src = {b.name: (w2d @ (s2w @ b.head_local))
             for b in source_arm.data.bones}
    old = {b.name: b.head_local.copy() for b in donor_arm.data.bones}

    new = {name: src[name] for name in old if name in src}
    for ik, fk in _IK_FOLLOW.items():
        if ik in old and fk in new:
            new[ik] = new[fk] + (old[ik] - old[fk])

    bones = donor_arm.data.bones

    def fitted_ancestor(b):
        p = b.parent
        while p is not None:
            if p.name in new:
                return p.name
            p = p.parent
        return None

    def fitted_descendant(b):
        queue = list(b.children)
        while queue:
            c = queue.pop(0)
            if c.name in new:
                return c.name
            queue.extend(c.children)
        return None

    def seg_map(a, d, pos):
        """Map pos through the rotation+scale taking old segment a→d to
        the new segment."""
        seg_o = old[d] - old[a]
        seg_n = new[d] - new[a]
        off = pos - old[a]
        if seg_o.length > 1e-6 and seg_n.length > 1e-6:
            q = seg_o.normalized().rotation_difference(seg_n.normalized())
            return new[a] + (q @ off) * (seg_n.length / seg_o.length)
        return new[a] + off

    for name in old:
        if name in new:
            continue
        b = bones[name]
        a = fitted_ancestor(b)
        if a is None:
            new[name] = old[name].copy()
            continue
        d = fitted_descendant(b)
        if d is None:
            # leaf helper (e.g. arm twists): ride the ancestor's own
            # fitted segment so it lands proportionally along the limb
            d = fitted_descendant(bones[a])
        if d is not None:
            new[name] = seg_map(a, d, old[name])
        else:
            new[name] = new[a] + (old[name] - old[a])
    return new


def compute_apose_heads(target_arm: bpy.types.Object,
                            ref_arm: bpy.types.Object) -> tuple[dict, dict, dict]:
    """FK-retarget the target's rest joints onto the reference pose:
    joint-to-joint DIRECTIONS become the reference's, segment LENGTHS
    stay the target's own. Root joint stays put, so the origin (feet on
    Z=0) is preserved. Returns (new_heads_local, roll_z, new_tails):
    roll_z holds the reference's canonical bone Z axes in target space,
    new_tails puts each bone's Y axis on the reference's (canonical
    animation frame) at the target's own bone length.
    """
    from mathutils import Quaternion as _Q
    w2t = target_arm.matrix_world.inverted_safe()
    r2w = ref_arm.matrix_world
    xf = w2t @ r2w
    xf3 = xf.to_3x3()
    ref = {b.name: (xf @ b.head_local) for b in ref_arm.data.bones}
    ref_z = {b.name: (xf3 @ b.matrix_local.to_3x3().col[2])
               for b in ref_arm.data.bones}
    ref_y = {}
    for b in ref_arm.data.bones:
        y = xf3 @ (b.tail_local - b.head_local)
        if y.length > 1e-8:
            ref_y[b.name] = y.normalized()
    old = {b.name: b.head_local.copy() for b in target_arm.data.bones}
    children_of = _children_map(target_arm)

    def primary_child(name, heads):
        best = None
        for ch in children_of.get(name, ()):
            if ch in heads:
                L = (heads[ch] - heads[name]).length
                if L > 1e-5 and (best is None or L > best[0]):
                    best = (L, ch)
        return best[1] if best else None

    def _depth(b):
        d, x = 0, b
        while x.parent is not None:
            x = x.parent; d += 1
        return d

    new: dict = {}
    acc: dict = {}
    for b in sorted(target_arm.data.bones, key=_depth):
        name = b.name
        pname = b.parent.name if b.parent else None
        R_p = acc.get(pname, _Q())
        if pname is None:
            # Top-level: origin helpers (attach, ik_*_root — at the origin
            # in the reference) snap to the reference so the rig's origin
            # stays canonical; anatomical top-levels (pelvis) keep their
            # own character-proportional position.
            r = ref.get(name)
            if r is not None and r.length < 0.01:
                new[name] = r.copy()
            else:
                new[name] = old[name].copy()
        else:
            new[name] = new[pname] + (R_p @ (old[name] - old[pname]))
        R = R_p
        ch = primary_child(name, old)
        if ch and name in ref and ch in ref:
            cur_dir = R_p @ (old[ch] - old[name])
            ref_dir = ref[ch] - ref[name]
            if cur_dir.length > 1e-6 and ref_dir.length > 1e-6:
                R = (cur_dir.normalized()
                       .rotation_difference(ref_dir.normalized())) @ R_p
        acc[name] = R

    # Tails: canonical Y direction at the target's own bone length
    old_tails = {b.name: b.tail_local.copy() for b in target_arm.data.bones}
    new_tails = {}
    for name, nh in new.items():
        y = ref_y.get(name)
        if y is None:
            continue
        length = (old_tails[name] - old[name]).length
        new_tails[name] = nh + y * max(length, 1e-4)
    return new, ref_z, new_tails


def compute_apose_anchored(target_arm: bpy.types.Object,
                           ref_arm: bpy.types.Object) -> tuple[dict, dict, dict]:
    """A-pose conversion that NEVER drags joints off the character's anatomy.

    T-pose → A-pose differs only in the ARM chains (clavicle→fingertips);
    legs, spine, neck and head are identical between the poses. The plain
    FK retarget (compute_apose_heads) re-marches EVERY chain from the root
    with canonical directions × character lengths, so per-segment length
    differences accumulate down the spine — shoulders land ~15cm low and
    medial, the whole skeleton sinks off the anatomy.

    Here:
      - arm chains are re-marched from their clavicle's FITTED head
        (anchored, no drift) with canonical parent→child directions and
        the character's own segment lengths — elbows/wrists swing down to
        A-pose and stay inside the arm mesh;
      - every other joint keeps its fitted (anatomical) position;
      - all bones still get canonical frames (roll_z / new_tails) for
        UEFN animation compatibility — frames don't move the mesh.

    Returns (new_heads_local, roll_z, new_tails) — same contract as
    compute_apose_heads.
    """
    from mathutils import Quaternion as _Q
    w2t = target_arm.matrix_world.inverted_safe()
    r2w = ref_arm.matrix_world
    xf = w2t @ r2w
    xf3 = xf.to_3x3()
    ref = {b.name: (xf @ b.head_local) for b in ref_arm.data.bones}
    ref_z = {b.name: (xf3 @ b.matrix_local.to_3x3().col[2])
               for b in ref_arm.data.bones}
    ref_y = {}
    for b in ref_arm.data.bones:
        y = xf3 @ (b.tail_local - b.head_local)
        if y.length > 1e-8:
            ref_y[b.name] = y.normalized()
    old = {b.name: b.head_local.copy() for b in target_arm.data.bones}
    bones = target_arm.data.bones

    # Only ARM chains march (T-pose → A-pose is an arm-only pose change).
    # Legs are identical between T- and A-pose — marching them to canonical
    # directions pulls the knees/ankles out of the mesh on wide-stance
    # characters (MIA's fitted leg joints follow the real stance; keep them,
    # canonical frames still apply via roll_z/new_tails).
    _CHAIN_ROOTS = ("clavicle_l", "clavicle_r")

    def _in_chain(b):
        x = b
        while x is not None:
            if x.name in _CHAIN_ROOTS:
                return True
            x = x.parent
        return False

    chain_bones = {b.name for b in bones if _in_chain(b)}

    def _depth(b):
        d, x = 0, b
        while x.parent is not None:
            x = x.parent; d += 1
        return d

    new: dict = {}
    for b in sorted(bones, key=_depth):
        name = b.name
        pname = b.parent.name if b.parent else None
        # March only when BOTH bone and parent are inside a chain — the
        # chain roots (clavicle, thigh) stay anchored at their fitted,
        # anatomical positions and the march starts one bone below.
        if (pname is None or name not in chain_bones
                or pname not in chain_bones):
            new[name] = old[name].copy()
            continue
        # March within the arm chain: canonical parent→child direction at
        # the character's own segment length. Directions come from JOINT
        # positions (head-to-head) — FBX stub tails are never consulted.
        rd = ref.get(name)
        rp = ref.get(pname)
        if rd is None or rp is None:
            new[name] = old[name].copy()
            continue
        ref_dir = rd - rp
        char_len = (old[name] - old[pname]).length
        if ref_dir.length < 1e-6 or char_len < 1e-6:
            new[name] = new[pname] + (old[name] - old[pname])
            continue
        new[name] = new[pname] + ref_dir.normalized() * char_len

    # IK helpers ride their FK bone with the original offset
    for ik, fk in _IK_FOLLOW.items():
        if ik in old and fk in new:
            new[ik] = new[fk] + (old[ik] - old[fk])

    # Tails: canonical Y direction at the target's own bone length
    old_tails = {b.name: b.tail_local.copy() for b in bones}
    new_tails = {}
    for name, nh in new.items():
        y = ref_y.get(name)
        if y is None:
            continue
        length = (old_tails[name] - old[name]).length
        new_tails[name] = nh + y * max(length, 1e-4)
    return new, ref_z, new_tails


# ── Pre-bind mesh conform (task #111) ────────────────────────────────────────
#
# Trellis/MIA bodies come out with non-canonical proportions (arms 30-50cm
# longer than SKM_UEFN_Mannequin). Binding those verts to canonical bones
# leaves a static offset at the extremities that every later pose rotation
# amplifies into visible warping. Fix the mesh BEFORE binding: match overall
# height, then axially scale each arm so the fingertips land at the donor's
# fingertip distance.

_FINGER_BASES = ("thumb", "index", "middle", "ring", "pinky")


def _arm_chain_vgroups(side: str) -> list[str]:
    names = [f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}"]
    names += [f"{f}_{i:02d}_{side}" for f in _FINGER_BASES for i in (1, 2, 3)]
    return names


def _donor_bone_world(donor_arm, name):
    b = donor_arm.data.bones.get(name)
    return (donor_arm.matrix_world @ b.head_local) if b else None


def _donor_bone_tail_world(donor_arm, name):
    b = donor_arm.data.bones.get(name)
    return (donor_arm.matrix_world @ b.tail_local) if b else None


def _world_z_extent(ob):
    if ob.type == "MESH":
        zs = [(ob.matrix_world @ v.co).z for v in ob.data.vertices]
    else:
        zs = [(ob.matrix_world @ b.head_local).z for b in ob.data.bones]
        zs += [(ob.matrix_world @ b.tail_local).z for b in ob.data.bones]
    return min(zs), max(zs)


def match_height_to_donor(mesh: bpy.types.Object,
                              rig_arm: bpy.types.Object,
                              ref_ob: bpy.types.Object) -> dict:
    """Uniformly scale mesh + rig so the mesh's world Z extent matches the
    reference object's, keeping feet on Z=0. Uniform — never distorts."""
    import bpy as _bpy
    report: dict = {}
    d_lo, d_hi = _world_z_extent(ref_ob)
    m_lo, m_hi = _world_z_extent(mesh)
    donor_h, mesh_h = d_hi - d_lo, m_hi - m_lo
    if mesh_h <= 1e-6 or donor_h <= 0.5:
        return report
    s = donor_h / mesh_h
    report["height_scale"] = round(s, 4)
    report["mesh_h_before"] = round(mesh_h, 4)
    report["donor_h"] = round(donor_h, 4)
    if abs(s - 1.0) > 0.005:
        # Unparent (keep transform) so each object scales independently
        was_parented = mesh.parent is rig_arm
        if was_parented:
            _bpy.ops.object.select_all(action="DESELECT")
            mesh.select_set(True)
            _bpy.context.view_layer.objects.active = mesh
            _bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        _bpy.ops.object.select_all(action="DESELECT")
        rig_arm.select_set(True); mesh.select_set(True)
        _bpy.context.view_layer.objects.active = rig_arm
        for ob in (rig_arm, mesh):
            ob.scale = tuple(c * s for c in ob.scale)
        _bpy.ops.object.transform_apply(location=False, rotation=False,
                                           scale=True)
        # Re-ground: keep feet on Z=0
        _bpy.context.view_layer.update()
        m_lo2, _ = _world_z_extent(mesh)
        if abs(m_lo2) > 1e-4:
            for ob in (rig_arm, mesh):
                ob.location.z -= m_lo2
            _bpy.ops.object.transform_apply(location=True, rotation=False,
                                               scale=False)
        if was_parented:
            _bpy.ops.object.select_all(action="DESELECT")
            mesh.select_set(True); rig_arm.select_set(True)
            _bpy.context.view_layer.objects.active = rig_arm
            _bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
        _bpy.context.view_layer.update()
    return report


def fit_donor_to_character(donor_arm: bpy.types.Object,
                               donor_mesh: "bpy.types.Object | None",
                               rig_arm: bpy.types.Object,
                               char_mesh: "bpy.types.Object | None" = None) -> dict:
    """Fit the donor skeleton (and its mannequin mesh) onto the character's
    anatomy — the inverse of mesh conform. The character mesh is never
    deformed; instead the donor's joints move to the autorig's joint
    positions, and the donor mesh follows via joint-swing LBS so
    proximity weight transfer happens between aligned surfaces.
    Idempotent (re-fitting to the same rig is a no-op)."""
    new_heads = compute_fitted_donor_heads(donor_arm, rig_arm)
    leg_snap = set()
    if char_mesh is not None:
        # Re-seat knee/ankle/toe joints inside the leg mesh (MIA leg
        # estimates run medial on wide-stance characters)
        try:
            leg_snap = snap_leg_joints_to_mesh(new_heads, char_mesh,
                                               donor_arm)
        except Exception:
            import traceback
            traceback.print_exc()
    meshes = [donor_mesh] if donor_mesh is not None else []
    moved_bones, stats = retarget_joints(donor_arm, new_heads, meshes=meshes)
    return {"fitted_bones": moved_bones, "donor_mesh_moved": stats,
            "leg_joints_snapped": sorted(leg_snap)}


def conform_mesh_to_uefn(mesh: bpy.types.Object,
                             rig_arm: bpy.types.Object,
                             donor_arm: bpy.types.Object,
                             donor_mesh: Optional[bpy.types.Object] = None,
                             ) -> dict:
    """Conform the autorigged mesh to canonical UEFN_Mannequin proportions
    before TransferBones binding.

    Two passes:
      1. Uniform scale — match the mesh's world Z extent to the donor
         mesh's (or donor armature's) Z extent, scaling about the world
         origin so feet stay on Z=0. The rig armature is scaled together
         with the mesh so its bones stay inside.
      2. Per-arm axial conform — for each side, scale arm-weighted verts
         along the shoulder→hand axis (pivot = the rig's upperarm head)
         so the farthest fingertip lands at the donor's shoulder→fingertip
         distance. Blended by per-vert arm weight so the shoulder boundary
         stays smooth. Legs are left alone: height normalization already
         anchors them and feet must stay grounded.

    Idempotent — running it twice computes scale factors ≈ 1.0 the second
    time.

    Args:
        mesh: the autorigged mesh (UEFN-named vertex groups)
        rig_arm: the armature the mesh is currently bound to (MIA rig)
        donor_arm: canonical UEFN_Mannequin armature (Source collection)
        donor_mesh: canonical mannequin mesh, for the height reference

    Returns:
        report dict with the applied factors
    """
    import bpy as _bpy
    from mathutils import Vector

    report: dict = {}

    # ── Pass 1: uniform height ───────────────────────────────────────────
    report.update(match_height_to_donor(
        mesh, rig_arm, donor_mesh if donor_mesh is not None else donor_arm))

    # ── Pass 2: per-arm axial conform ────────────────────────────────────
    mesh_world = mesh.matrix_world
    mesh_world_inv = mesh_world.inverted_safe()

    # Weights are read once per side from mesh data; writes go through
    # bmesh so they survive later edit-mode switches (Blender 5.1 reverts
    # plain object-mode vertices[].co writes on the next mode toggle).
    import bmesh as _bmesh

    for side in ("l", "r"):
        key = f"arm_{side}"
        chain = _arm_chain_vgroups(side)
        chain_idx = {mesh.vertex_groups[n].index
                       for n in chain if n in mesh.vertex_groups}
        if not chain_idx:
            report[key] = {"skipped": "no arm vgroups"}
            continue

        pivot_bone = rig_arm.data.bones.get(f"upperarm_{side}")
        if pivot_bone is None:
            report[key] = {"skipped": f"upperarm_{side} missing on rig"}
            continue
        pivot_w = rig_arm.matrix_world @ pivot_bone.head_local

        # Per-vert arm weight + world positions
        weights: dict[int, float] = {}
        for v in mesh.data.vertices:
            w = sum(g.weight for g in v.groups if g.group in chain_idx)
            if w > 1e-4:
                weights[v.index] = min(w, 1.0)
        if not weights:
            report[key] = {"skipped": "no weighted verts"}
            continue

        # Axis: shoulder → hand-region centroid (world)
        hand_idx = {mesh.vertex_groups[n].index
                      for n in ([f"hand_{side}"] +
                                 [f"{f}_{i:02d}_{side}" for f in _FINGER_BASES
                                  for i in (1, 2, 3)])
                      if n in mesh.vertex_groups}
        h_total, h_w = Vector((0, 0, 0)), 0.0
        for v in mesh.data.vertices:
            w = sum(g.weight for g in v.groups if g.group in hand_idx)
            if w > 0.25:
                h_total += (mesh_world @ v.co) * w
                h_w += w
        if h_w < 1e-6:
            report[key] = {"skipped": "no hand verts to derive axis"}
            continue
        axis = ((h_total / h_w) - pivot_w)
        if axis.length < 1e-4:
            report[key] = {"skipped": "degenerate arm axis"}
            continue
        axis.normalize()

        # Current reach: farthest solidly-arm-weighted vert from the
        # shoulder. Euclidean distance, not axis projection — pose-
        # independent, so it compares 1:1 with the donor measurement
        # below even when the two are posed differently.
        cur_len = 0.0
        for vi, w in weights.items():
            if w < 0.5:
                continue
            dist = ((mesh_world @ mesh.data.vertices[vi].co) - pivot_w).length
            cur_len = max(cur_len, dist)
        if cur_len < 1e-3:
            report[key] = {"skipped": "no forward reach"}
            continue

        # Donor reach: measured on the DONOR MESH with the identical
        # method (farthest arm-weighted vert from the shoulder joint,
        # Euclidean). Donor finger BONES are unreliable (the T-pose donor
        # build only re-posed the arm chain, leaving finger bones at
        # legacy positions) and the donor MESH DATA may be stored in a
        # different pose than the bones (A-pose data under T-pose bones)
        # — but the shoulder joint doesn't move between poses and the
        # farthest arm vert is the fingertip in any pose, so the
        # distance compares like for like.
        d_shoulder = _donor_bone_world(donor_arm, f"upperarm_{side}")
        if d_shoulder is None:
            report[key] = {"skipped": "donor upperarm missing"}
            continue

        tgt_len = None
        d_tip_pos = None
        if donor_mesh is not None:
            d_chain_idx = {donor_mesh.vertex_groups[n].index
                             for n in chain
                             if n in donor_mesh.vertex_groups}
            if d_chain_idx:
                d_world = donor_mesh.matrix_world
                reach = 0.0
                for v in donor_mesh.data.vertices:
                    w = sum(g.weight for g in v.groups
                             if g.group in d_chain_idx)
                    if w < 0.5:
                        continue
                    p = d_world @ v.co
                    dist = (p - d_shoulder).length
                    if dist > reach:
                        reach = dist
                        d_tip_pos = p
                if reach > 1e-3:
                    tgt_len = reach
        if tgt_len is None:
            # No donor mesh/weights: approximate fingertip as wrist +
            # ~50% of the shoulder→wrist distance (anthropometric hand)
            d_wrist = _donor_bone_world(donor_arm, f"hand_{side}")
            if d_wrist is None:
                report[key] = {"skipped": "donor hand bone missing"}
                continue
            tgt_len = (d_wrist - d_shoulder).length * 1.5
            report[key + "_note"] = "donor mesh unavailable — wrist*1.5 estimate"

        # Aim + reach target: our fingertip should land ON the donor's
        # fingertip position. Aiming from OUR shoulder (not just matching
        # the donor's arm direction) absorbs shoulder-height differences —
        # e.g. civilian's shoulders sit ~18cm below the mannequin's
        # because hair inflates the height-normalization reference. If the
        # arm axis stayed parallel-but-below, proximity weight transfer
        # would map our hand onto the donor thumb's underside.
        if d_tip_pos is not None:
            tgt_len = (d_tip_pos - pivot_w).length
        s = tgt_len / cur_len
        s = max(0.5, min(1.5, s))  # safety clamp
        report[key] = {"cur_len": round(cur_len, 4),
                        "tgt_len": round(tgt_len, 4),
                        "scale": round(s, 4)}

        # Rotation: swing the arm so it points at the donor fingertip
        # (fallback: parallel to the donor's shoulder→wrist axis).
        # Trellis/MIA T-poses droop ~10° below horizontal; without this
        # the hand lands below the donor's hand and the weight transfer
        # maps our fingers onto the donor's thumb side.
        from mathutils import Quaternion as _Q
        d_dir_w = None
        if d_tip_pos is not None:
            v = d_tip_pos - pivot_w
            if v.length > 1e-4:
                d_dir_w = v.normalized()
        else:
            d_wrist = _donor_bone_world(donor_arm, f"hand_{side}")
            if d_wrist is not None:
                v = d_wrist - d_shoulder
                if v.length > 1e-4:
                    d_dir_w = v.normalized()

        pivot_l = mesh_world_inv @ pivot_w
        m3 = mesh_world_inv.to_3x3()
        axis_l = (m3 @ axis).normalized()
        rot_l = None
        if d_dir_w is not None:
            d_dir_l = (m3 @ d_dir_w).normalized()
            q = axis_l.rotation_difference(d_dir_l)
            if abs(q.angle) > 0.005:
                rot_l = q
                report[key]["swing_deg"] = round(q.angle * 57.2958, 2)
            final_axis_l = d_dir_l
        else:
            final_axis_l = axis_l

        if rot_l is None and abs(s - 1.0) <= 0.01:
            report[key]["applied"] = False
            continue

        # Apply per vert, blended by arm weight: rotate about the
        # shoulder pivot (slerp toward the donor axis), then scale the
        # axial component so the fingertip lands at the donor reach.
        ident = _Q()
        bm = _bmesh.new()
        bm.from_mesh(mesh.data)
        bm.verts.ensure_lookup_table()
        moved = 0
        for vi, w in weights.items():
            bv = bm.verts[vi]
            p = bv.co - pivot_l
            if rot_l is not None:
                p = ident.slerp(rot_l, w) @ p
            d = p.dot(final_axis_l)
            if d > 0:
                p += final_axis_l * (d * (s - 1.0) * w)
            bv.co = pivot_l + p
            moved += 1
        bm.to_mesh(mesh.data)
        bm.free()
        mesh.data.update()
        report[key]["applied"] = True
        report[key]["verts_moved"] = moved

    _bpy.context.view_layer.update()
    return report


# ── Post-transfer wedge-hand weight fix (ARTS-37) ────────────────────────────

def _detect_digit_clusters(mesh, island, a, wrist_t, axis_n, mw_mesh):
    """Detect distinct digit sub-islands inside a distal hand island.

    Two routes, most-structural first:
      1. geodesic-components: connected components (mesh edges) in the
         distal zone, kept when elongated (principal-axis ratio > 1.8);
      2. anterior-kmeans (fallback for webbed fingers): 1D k-means (k=5)
         on the anterior spread coordinate of distal verts, gated by an
         interior-gaps test so a continuous mitt wedge is NOT fabricated
         into fake digits.

    Returns (digits, method, palm_boundary) with digits ordered
    thumb->pinky (anterior = world -Y for the pipeline's palms-down
    T-pose; self-correcting reversal via relative digit length), or None
    when the island is a single blob (mitt)."""
    import numpy as np
    pos = {vi: mw_mesh @ mesh.data.vertices[vi].co for vi in island}
    proj = {vi: (pos[vi] - a).dot(axis_n) for vi in island}
    pmax = max(proj.values())
    extent = max(pmax - wrist_t, 1e-6)
    palm_b = wrist_t + 0.35 * extent
    distal = [vi for vi in island if proj[vi] > wrist_t + 0.55 * extent]

    def elongated(comp):
        P = np.array([[p.x, p.y, p.z] for p in (pos[vi] for vi in comp)])
        P = P - P.mean(axis=0)
        cov = P.T @ P / len(P)
        w = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-12))
        return w[-1] ** 0.5 > 1.8 * w[-2] ** 0.5

    # route 1: connected components in the distal zone
    dset = set(distal)
    adj = {}
    for e in mesh.data.edges:
        a1, a2 = e.vertices
        if a1 in dset and a2 in dset:
            adj.setdefault(a1, set()).add(a2)
            adj.setdefault(a2, set()).add(a1)
    comps, seen = [], set()
    for vi in distal:
        if vi in seen:
            continue
        stack, comp = [vi], []
        seen.add(vi)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj.get(u, ()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    comps = [c for c in comps if len(c) >= 4]
    comps.sort(key=len, reverse=True)
    digits = [c for c in comps[:6] if elongated(c)]
    method = "geodesic-components"

    if len(digits) < 4:
        # route 2 (fallback): anterior-spread k-means with a gap guard
        ys = sorted((pos[vi].y, vi) for vi in distal)
        if len(ys) < 20:
            return None
        yvals = np.array([y for y, _ in ys])
        dy = np.diff(yvals)
        # interior gaps in the anterior coverage = separated digits; a
        # continuous mitt wedge fails this guard and stays a mitt
        if dy.size == 0 or dy.max() < 1e-6 or int((dy > 0.6 * dy.max()).sum()) < 3:
            return None
        centers = np.linspace(yvals[0], yvals[-1], 5)
        for _ in range(12):
            bins = [[] for _ in range(5)]
            for y, vi in ys:
                bins[int(np.argmin(np.abs(centers - y)))].append(vi)
            newc = np.array([np.mean([pos[vi].y for vi in b]) if b else c
                             for b, c in zip(bins, centers)])
            if np.allclose(newc, centers):
                break
            centers = newc
        digits = [b for b in bins if len(b) >= 3]
        method = "anterior-kmeans"
        if len(digits) < 4:
            return None

    # order thumb->pinky: anterior (world -Y) first on the pipeline's
    # palms-down T-pose frame; the thumb is also the shorter extreme digit,
    # so if the anterior-first cluster is LONGER than the posterior one the
    # hand was flipped (palms-up) — reverse.
    def mean_y(c):
        return sum(pos[vi].y for vi in c) / len(c)
    def tip_extent(c):
        return max(proj[vi] for vi in c) - palm_b
    digits.sort(key=mean_y)
    digits = digits[:5]
    if len(digits) >= 2 and tip_extent(digits[0]) > tip_extent(digits[-1]):
        digits.reverse()
        method += "(reversed: palms-up)"
    return digits, method, palm_b


def remap_empty_hand_weights(arm: bpy.types.Object,
                             mesh: bpy.types.Object,
                             eps: float = 0.005,
                             min_hand_ownership: float = 0.5) -> dict:
    """Rebind wrist-stub vertex islands to hand_l/r when the donor weight
    transfer left those groups empty OR effectively empty.

    Characters whose hands are wedge stubs (no finger geometry — a fused
    mitt) are failed by POLYINTERP_NEAREST in two ways:
      - run041-043 (manny donor): hand groups come back EMPTY, the whole
        wedge rides lowerarm_l/r (dead hands);
      - run055 (bigbase75 donor): hand groups catch ~5-11 stray palm
        verts while the wedge mass is scattered over arbitrary finger
        chains (index_02 swings the whole wedge; thumb drives nothing).
    Both are cured the same way: the whole distal island belongs rigidly
    on hand_<side> for a mitt.

    Trigger (per side): build the hand-region island = verts whose world
    position projects past the hand_<side> joint head along the
    lowerarm→hand axis (+eps) and whose dominant group belongs to that
    side's arm chain. Fire when hand_<side> owns less than
    `min_hand_ownership` of the island's weight mass. When firing, every
    island vert is rigidly reassigned to hand_<side> (w=1.0, all other
    group weights cleared) — anatomically correct for a wedge stub.
    Wrist-boundary strays (verts within 3cm PROXIMAL of the wrist plane
    holding partial hand_<side> weight) are folded into the remap too —
    the transfer leaves them mixed with lowerarm weights and they pinch
    at the wrist under hand rotation (run061 rico: 2 strays broke the
    rigid proof at 6.1e-02).

    Anatomy veto: if two or more finger chains have their island clusters
    sitting ON their bones (centroid within ~4cm of the bone head), the
    character has real, correctly-transferred fingers — do NOT rigidify.

    Returns {side: {"island": n, "strays": m}} for the sides that fired.
    """
    from mathutils import Vector  # noqa: F401  (parity with module style)
    mw_arm = arm.matrix_world
    mw_mesh = mesh.matrix_world
    vg_idx = {vg.index: vg.name for vg in mesh.vertex_groups}
    report = {}
    for side in ("l", "r"):
        hand_name = f"hand_{side}"
        lower_name = f"lowerarm_{side}"
        hand_vg = mesh.vertex_groups.get(hand_name)
        hand_bone = arm.data.bones.get(hand_name)
        lower_bone = arm.data.bones.get(lower_name)
        if None in (hand_vg, hand_bone, lower_bone):
            continue
        chain_prefixes = ("upperarm_", "lowerarm_", "hand_", "thumb_",
                          "index_", "middle_", "ring_", "pinky_")

        def in_arm_chain(gname):
            return (gname is not None
                    and gname.endswith("_" + side)
                    and gname.startswith(chain_prefixes))

        a = mw_arm @ lower_bone.head_local
        b = mw_arm @ hand_bone.head_local
        axis = b - a
        if axis.length < 1e-8:
            continue
        axis_n = axis.normalized()
        wrist_t = axis.length + eps  # hand joint head projection + epsilon

        # island: distal to the wrist plane, dominantly arm-chain-owned
        island, hand_mass, total_mass = [], 0.0, 0.0
        strays = []           # partial hand_<side> weights just PROXIMAL of
                              # the wrist plane — transfer leftovers that a
                              # rigid island would otherwise leave pinching
                              # at the wrist (run061 rico rigid-proof fail)
        finger_owned = {}   # group name -> [world positions]
        for v in mesh.data.vertices:
            if not v.groups:
                continue
            p = mw_mesh @ v.co
            proj = (p - a).dot(axis_n)
            if proj <= wrist_t:
                # stray window: up to 3cm proximal of the wrist plane
                if proj > wrist_t - 0.03:
                    for g in v.groups:
                        if (g.group == hand_vg.index
                                and 1e-6 < g.weight < 0.999):
                            strays.append(v.index)
                            break
                continue
            best, bw, w_hand, w_tot = None, 0.0, 0.0, 0.0
            for g in v.groups:
                nm = vg_idx.get(g.group)
                if nm is None:
                    continue
                w_tot += g.weight
                if g.group == hand_vg.index:
                    w_hand = g.weight
                if g.weight > bw:
                    best, bw = nm, g.weight
            if not in_arm_chain(best):
                continue
            island.append(v.index)
            hand_mass += w_hand
            total_mass += w_tot
            if best.startswith(("thumb_", "index_", "middle_", "ring_",
                                "pinky_")) and "metacarpal" not in best:
                finger_owned.setdefault(best, []).append(p)
        if not island:
            print(f"[BD_AutoRig:handfix] {hand_name}: no distal arm island "
                  f"found — left untouched", flush=True)
            continue
        ownership = hand_mass / max(total_mass, 1e-9)

        # Anatomy veto: real fingers = island clusters on their own bones
        placed = 0
        for gname, pts in finger_owned.items():
            bone = arm.data.bones.get(gname)
            if bone is None or len(pts) < 3:
                continue
            cen = sum(pts, Vector()) / len(pts)
            if (cen - (mw_arm @ bone.head_local)).length < 0.04:
                placed += 1
        if placed >= 2:
            print(f"[BD_AutoRig:handfix] {hand_name}: {placed} finger "
                  f"chains anatomically placed — real fingers, vetoed "
                  f"(hand ownership {ownership:.0%})", flush=True)
            continue
        if ownership >= min_hand_ownership:
            print(f"[BD_AutoRig:handfix] {hand_name}: owns "
                  f"{ownership:.0%} of the distal island — healthy, "
                  f"left untouched", flush=True)
            continue

        reason = ("empty" if hand_mass < 1e-6
                  else f"effectively-empty ({ownership:.0%} ownership)")

        # ── finger-aware branch (run072+): when the guard fires, first
        # check the island's structure. Distinct digit sub-islands = real
        # fingers → DO NOT collapse; assign per-chain instead. Single blob
        # = mitt → rigid collapse exactly as before (14 mitt characters
        # unaffected).
        digit_info = _detect_digit_clusters(mesh, island, a, wrist_t,
                                            axis_n, mw_mesh)
        if digit_info is not None:
            digits, dmethod, palm_b = digit_info
            fams = ("thumb", "index", "middle", "ring", "pinky")
            proj = {vi: (mw_mesh @ mesh.data.vertices[vi].co - a).dot(axis_n)
                    for vi in island}
            pos_yz = {vi: (mw_mesh @ mesh.data.vertices[vi].co)
                      for vi in island}
            centroids = []
            for comp in digits:
                c = sum((pos_yz[vi] for vi in comp), Vector()) / len(comp)
                centroids.append(c)
            assign = {}
            counts = {}
            for ci, comp in enumerate(digits):
                fam = fams[ci] if ci < len(fams) else f"digit{ci}"
                cprojs = [proj[vi] for vi in comp]
                cmax = max(cprojs)
                span = max(cmax - palm_b, 1e-6)
                for vi in comp:
                    t = (proj[vi] - palm_b) / span
                    seg = 1 if t < 0.45 else (2 if t < 0.8 else 3)
                    assign.setdefault(f"{fam}_{seg:02d}_{side}",
                                      []).append(vi)
            # verts between clusters (webbing) in the digit zone → nearest
            # cluster centroid; palm zone → hand_<side>
            in_digits = {vi for comp in digits for vi in comp}
            for vi in island:
                if vi in in_digits:
                    continue
                if proj[vi] > palm_b and centroids:
                    p = pos_yz[vi]
                    near = min(range(len(centroids)),
                               key=lambda k: (p - centroids[k]).length)
                    fam = (fams[near] if near < len(fams)
                           else f"digit{near}")
                    assign.setdefault(f"{fam}_01_{side}", []).append(vi)
            palm = [vi for vi in island
                    if vi not in {x for vs in assign.values() for x in vs}]
            assign.setdefault(hand_name, []).extend(palm)
            remap_d = [vi for vs in assign.values() for vi in vs]
            strays_d = [vi for vi in strays if vi not in remap_d]
            assign[hand_name].extend(strays_d)
            remap = remap_d + strays_d
            for vg in mesh.vertex_groups:
                vg.remove(remap)
            for gname, verts in assign.items():
                vg = mesh.vertex_groups.get(gname)
                if vg is not None and verts:
                    vg.add(verts, 1.0, "REPLACE")
                    counts[gname] = len(verts)
            report[side] = {"island": len(island),
                            "strays": len(strays_d),
                            "digits": counts, "method": dmethod}
            print(f"[BD_AutoRig:handfix] {hand_name}: {reason} but "
                  f"{len(digits)} digit sub-islands found ({dmethod}) — "
                  f"FINGERED, no collapse; chain assignment: "
                  + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                  flush=True)
            continue

        remap = island + [vi for vi in strays if vi not in island]
        for vg in mesh.vertex_groups:
            vg.remove(remap)
        hand_vg.add(remap, 1.0, "REPLACE")
        report[side] = {"island": len(island),
                        "strays": len(remap) - len(island)}
        print(f"[BD_AutoRig:handfix] {hand_name}: {reason} — remapped "
              f"{len(island)} distal-island verts to {hand_name} "
              f"(rigid, w=1.0)", flush=True)
        if len(remap) > len(island):
            print(f"[BD_AutoRig:handfix] {hand_name}: + "
                  f"{len(remap) - len(island)} wrist-stray verts "
                  f"(partial hand weights within 3cm proximal) folded in",
                  flush=True)
    return report


_BONE_NAME_CANDIDATES = {
    # Vertex groups are usually already UEFN-named at align-time (mia_export
    # renames them during FBX assembly), but the bones may still have the
    # Mixamo prefix. List all forms; first match wins.
    "head":   ("head", "Head", "mixamorig:Head"),
    "pelvis": ("pelvis", "Hips", "mixamorig:Hips"),
    # Left/right lateral. Prefer clavicle since hand bones are merged out
    # when no_fingers=True. Fall back to upperarm/lowerarm/foot.
    "lat_l": ("clavicle_l", "upperarm_l", "lowerarm_l", "foot_l",
                "LeftShoulder", "LeftArm", "mixamorig:LeftShoulder",
                "mixamorig:LeftArm", "mixamorig:LeftFoot"),
    "lat_r": ("clavicle_r", "upperarm_r", "lowerarm_r", "foot_r",
                "RightShoulder", "RightArm", "mixamorig:RightShoulder",
                "mixamorig:RightArm", "mixamorig:RightFoot"),
    "foot_l": ("foot_l", "LeftFoot", "mixamorig:LeftFoot"),
}


def _find_named(container, candidates):
    """Find the first matching name in `container` (vertex_groups or bones).
    `container` must support .get(name)."""
    for name in candidates:
        x = container.get(name)
        if x is not None:
            return x
    return None


def _kabsch_matrix_from_source(mesh: bpy.types.Object,
                               source_mesh: bpy.types.Object,
                               max_samples: int = 8000,
                               residual_tol: float = 0.05):
    """Similarity transform (rotation + uniform scale + translation) taking
    the MIA-imported mesh onto the user's ORIGINAL source mesh.

    Two paths:
      - vertex counts MATCH (order survives GLB→trimesh→MIA→FBX for most
        meshes): exact Kabsch over corresponding indices — anatomy-
        agnostic, no 180° ambiguity (the anatomical centroid alignment
        this replaces shipped characters facing +Y).
      - counts DIFFER (trimesh dedups verts on some meshes): PCA-
        initialized ICP over nearest neighbors (4 proper-rotation sign
        combos, best-of, 4 iterations).

    Returns a world-space 4x4 Matrix to LEFT-multiply onto
    mesh.matrix_world, or None when unusable (residual-guarded).
    """
    import numpy as np
    from mathutils import Matrix as _M

    if source_mesh is None:
        return None
    n_src = len(source_mesh.data.vertices)
    n_mia = len(mesh.data.vertices)
    if n_src == 0 or n_mia == 0:
        return None
    sw, mw = source_mesh.matrix_world, mesh.matrix_world

    def _sample(obj, mtx, n, cap):
        step = max(1, n // cap)
        return np.array([tuple(mtx @ obj.data.vertices[i].co)
                         for i in range(0, n, step)])

    A = _sample(source_mesh, sw, n_src, max_samples)   # source (target)
    B = _sample(mesh, mw, n_mia, max_samples)          # MIA (to move)
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    sa = np.sqrt((Ac ** 2).sum() / len(Ac))
    sb = np.sqrt((Bc ** 2).sum() / len(Bc))
    if sa < 1e-9 or sb < 1e-9:
        return None

    def _kabsch(P, Q):
        """R, t mapping P onto Q (uniform scale folded in by caller)."""
        H = P.T @ Q
        U, _sv, Vt = np.linalg.svd(H)
        Rm = Vt.T @ np.diag([1.0, 1.0,
                             np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        return Rm

    if n_src == n_mia:
        Rm = _kabsch(Bc, Ac)
        s = sa / sb
        t = ca - s * (Rm @ cb)
        resid = np.abs((s * (Rm @ Bc.T).T + t) - Ac).mean() / sa
    else:
        # ICP in centered, unit-normalized WORLD coordinates. Column
        # convention throughout (matches _kabsch: An ≈ R @ Bn).
        An, Bn = Ac / sa, Bc / sb
        # PCA frames (columns = principal axes) — init only
        Pa = np.linalg.eigh(An.T @ An)[1]
        Pb = np.linalg.eigh(Bn.T @ Bn)[1]
        best = None

        def _nn(P, Q, chunk=400):
            idx = np.empty(len(P), dtype=int)
            for i in range(0, len(P), chunk):
                d = ((P[i:i + chunk, None, :] - Q[None, :, :]) ** 2).sum(-1)
                idx[i:i + chunk] = d.argmin(1)
            return idx

        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            # column-convention init: B in B-PCA coords ≈ A in A-PCA coords
            # up to axis signs → R0 = Pa @ diag(signs) @ Pb.T
            R = Pa @ np.diag(signs) @ Pb.T
            for _it in range(5):
                Bw = (R @ Bn.T).T
                j = _nn(Bw, An)
                Rm = _kabsch(Bw, An[j])
                R = Rm @ R
                if np.abs(Rm - np.eye(3)).max() < 1e-6:
                    break
            Bw = (R @ Bn.T).T
            j = _nn(Bw, An)
            resid = np.abs(Bw - An[j]).mean()
            if best is None or resid < best[0]:
                best = (resid, R)
        Rm = best[1]
        # accept/reject on the ICP residual (normalized units, NN-based —
        # index residuals are impossible with different vert counts)
        s = sa / sb
        t = ca - s * (Rm @ cb)
        resid = best[0]

    if resid > residual_tol:
        print(f"[BD_AutoRig] Kabsch rejected: residual {resid:.3f} "
              f"(vertex order changed in the MIA round-trip?)")
        return None
    M = _M(((Rm[0, 0] * s, Rm[0, 1] * s, Rm[0, 2] * s, t[0]),
             (Rm[1, 0] * s, Rm[1, 1] * s, Rm[1, 2] * s, t[1]),
             (Rm[2, 0] * s, Rm[2, 1] * s, Rm[2, 2] * s, t[2]),
             (0, 0, 0, 1)))
    print(f"[BD_AutoRig] Kabsch source-align: residual {resid:.4f}, "
          f"scale {s:.3f}")
    return M


def _vgroup_centroid(mesh: bpy.types.Object, vg_name: str):
    """Mean position of vertices weighted to the given vertex group, in
    mesh-data space. Returns None if no weighted verts found."""
    from mathutils import Vector
    vg = mesh.vertex_groups.get(vg_name)
    if vg is None:
        return None
    total = Vector((0, 0, 0))
    total_w = 0.0
    idx = vg.index
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == idx:
                total += v.co * g.weight
                total_w += g.weight
                break
    if total_w < 1e-9:
        return None
    return total / total_w


def _build_align_rotation(mesh: bpy.types.Object,
                              bone_targets: dict):
    """Compute a rotation matrix that aligns the mesh's anatomy to the
    armature's anatomy.

    Strategy:
      1. up_mesh    = (head_centroid - pelvis_centroid).normalized
      2. up_target  = (head_bone_world_pos - pelvis_bone_world_pos).normalized
      3. lr_mesh    = (hand_l_centroid - hand_r_centroid).normalized
      4. lr_target  = (hand_l_bone - hand_r_bone).normalized
      5. Build an orthonormal target frame from (lr_target, up_target)
         and a mesh frame from (lr_mesh, up_mesh), then R = target * mesh⁻¹.

    Args:
        mesh: with vertex groups still in Mixamo naming
        bone_targets: dict[name → world Vector] for "head", "pelvis",
            "hand_l", "hand_r" (precomputed from the bones)

    Returns:
        (R: mathutils.Matrix, debug: dict) or (None, debug) on failure
    """
    from mathutils import Vector, Matrix

    # Mesh centroids (mesh-local space)
    c_head    = _vgroup_centroid(mesh, _vg_name_for(mesh, "head"))
    c_pelvis  = _vgroup_centroid(mesh, _vg_name_for(mesh, "pelvis"))
    c_lat_l   = _vgroup_centroid(mesh, _vg_name_for(mesh, "lat_l"))
    c_lat_r   = _vgroup_centroid(mesh, _vg_name_for(mesh, "lat_r"))

    debug = {
        "mesh_centroids": {
            "head": list(c_head) if c_head else None,
            "pelvis": list(c_pelvis) if c_pelvis else None,
            "lat_l": list(c_lat_l) if c_lat_l else None,
            "lat_r": list(c_lat_r) if c_lat_r else None,
        },
        "bone_targets": {k: list(v) for k, v in bone_targets.items()},
        "vg_names_used": {
            "head":   _vg_name_for(mesh, "head"),
            "pelvis": _vg_name_for(mesh, "pelvis"),
            "lat_l":  _vg_name_for(mesh, "lat_l"),
            "lat_r":  _vg_name_for(mesh, "lat_r"),
        },
    }

    if c_head is None or c_pelvis is None or c_lat_l is None or c_lat_r is None:
        debug["error"] = "missing centroids (head/pelvis/lat_l/lat_r)"
        return None, debug
    if "head" not in bone_targets or "pelvis" not in bone_targets:
        debug["error"] = "missing head/pelvis bone target"
        return None, debug
    if "lat_l" not in bone_targets or "lat_r" not in bone_targets:
        debug["error"] = "missing lat_l/lat_r bone target"
        return None, debug

    bt_head, bt_pelvis = bone_targets["head"], bone_targets["pelvis"]
    bt_lat_l, bt_lat_r = bone_targets["lat_l"], bone_targets["lat_r"]

    up_mesh = (c_head - c_pelvis).normalized()
    up_targ = (bt_head - bt_pelvis).normalized()
    lr_mesh = (c_lat_l - c_lat_r).normalized()
    lr_targ = (bt_lat_l - bt_lat_r).normalized()

    # Orthogonalize the lateral axis against up
    lr_mesh = (lr_mesh - lr_mesh.dot(up_mesh) * up_mesh).normalized()
    lr_targ = (lr_targ - lr_targ.dot(up_targ) * up_targ).normalized()
    fw_mesh = up_mesh.cross(lr_mesh).normalized()
    fw_targ = up_targ.cross(lr_targ).normalized()

    # 3×3 frames as column vectors
    M_mesh = Matrix((lr_mesh, fw_mesh, up_mesh)).transposed()
    M_targ = Matrix((lr_targ, fw_targ, up_targ)).transposed()
    R = M_targ @ M_mesh.inverted()
    debug["up_mesh"], debug["up_targ"] = list(up_mesh), list(up_targ)
    debug["lr_mesh"], debug["lr_targ"] = list(lr_mesh), list(lr_targ)
    return R, debug


def _vg_name_for(mesh: bpy.types.Object, role: str) -> str:
    """Return the actual vertex group name on `mesh` for the given canonical
    role (head/pelvis/hand_l/...). Picks the first candidate that exists."""
    cands = _BONE_NAME_CANDIDATES.get(role, (role,))
    for c in cands:
        if c in mesh.vertex_groups:
            return c
    return role  # fallback (likely missing)


def align_imported_to_uefn(arm: bpy.types.Object,
                              mesh: bpy.types.Object,
                              source_mesh: Optional[bpy.types.Object] = None
                              ) -> dict:
    """Land the FBX import in canonical UEFN coords without hardcoding
    MIA's particular orientation.

    The FBX from autorig_vendor/mia_export.py needs:
      1. Identity armature object transform (FBX scale 0.01 + 90°X applied)
      2. Mesh data scaled to UEFN-skeleton height (~2m)
      3. Mesh data rotated so its anatomy lines up with the armature
         (head-over-pelvis matches the armature, arms-out matches)
      4. Mesh translated so feet sit on Z=0

    Steps 3 is the brittle one — we use weighted-vertex-group centroids
    on the mesh (head/pelvis/hand_l/hand_r) and corresponding bone world
    positions to *derive* the rotation, so it auto-corrects regardless of
    what orientation MIA emitted.
    """
    import bpy as _bpy
    from mathutils import Vector, Matrix
    from math import radians

    # 1) Apply armature+mesh transforms so armature is identity & in meters
    _bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True); mesh.select_set(True)
    _bpy.context.view_layer.objects.active = arm
    _bpy.ops.object.transform_apply(location=False, rotation=True, scale=True,
                                       properties=True, isolate_users=False)

    # 1.5) Facing normalization: UEFN/Unreal expects face=-Y. The toe
    # protrudes forward of the ankle, so the foot→toe joint vector IS the
    # facing. If the rig faces +Y (common for Hunyuan3D/Trellis sources),
    # rotate armature + mesh 180° about Z together — face lands at -Y and
    # anatomical left lands at +X (exactly the UEFN convention; handedness
    # semantics preserved, no L/R rename needed).
    def _bone_head(bn):
        b = arm.data.bones.get(bn)
        return arm.matrix_world @ b.head_local if b else None
    _TOE_CANDS = ("ball_l", "LeftToeBase", "mixamorig:LeftToeBase")
    _FOOT_CANDS = ("foot_l", "LeftFoot", "mixamorig:LeftFoot")
    toe = next((h for h in (_bone_head(n) for n in _TOE_CANDS) if h), None)
    foot = next((h for h in (_bone_head(n) for n in _FOOT_CANDS) if h), None)
    facing_flip = False
    if toe is not None and foot is not None:
        fy = (toe - foot).y
        if fy > 0.02:  # toes at +Y → facing +Y → flip
            from mathutils import Matrix as _M
            rz = _M.Rotation(radians(180), 4, "Z")
            arm.matrix_world = rz @ arm.matrix_world
            mesh.matrix_world = rz @ mesh.matrix_world
            _bpy.ops.object.select_all(action="DESELECT")
            arm.select_set(True); mesh.select_set(True)
            _bpy.context.view_layer.objects.active = arm
            _bpy.ops.object.transform_apply(location=True, rotation=True,
                                               scale=True)
            facing_flip = True

    # 2) Always scale to UEFN-skeleton height (the armature's Z extent post-
    #    apply is exactly that). Fall back to source_mesh height if armature
    #    is degenerate.
    arm_pts = [arm.matrix_world @ Vector(c) for c in arm.bound_box]
    arm_h = max(p.z for p in arm_pts) - min(p.z for p in arm_pts)
    target_h = arm_h if arm_h > 0.5 else 1.8

    mesh_pts = [Vector(c) for c in mesh.bound_box]
    mesh_diag = (Vector(mesh.bound_box[6]) - Vector(mesh.bound_box[0])).length
    # We don't know which mesh axis is "height" yet — use diagonal as a
    # conservative size, refine after rotation.
    pre_scale = (target_h * 1.8) / max(mesh_diag, 1e-9)

    # 3) Pre-scale + parent_clear so subsequent matrix ops are cheap
    _bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    _bpy.context.view_layer.objects.active = mesh
    _bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    _bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    mesh.scale = (pre_scale, pre_scale, pre_scale)
    _bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 4) Gather bone world-position targets
    def _bone_world(bn):
        b = arm.data.bones.get(bn)
        return arm.matrix_world @ b.head_local if b else None
    bone_targets = {}
    for role in ("head", "pelvis", "lat_l", "lat_r", "foot_l"):
        for cand in _BONE_NAME_CANDIDATES[role]:
            v = _bone_world(cand)
            if v is not None:
                bone_targets[role] = v
                break

    debug = {"pre_scale": pre_scale, "target_h": target_h,
             "facing_flip": facing_flip}

    # 5) Compute alignment rotation — Kabsch onto the source mesh first
    # (exact, anatomy-agnostic, no 180° ambiguity); anatomical centroids
    # only as fallback when the source mesh isn't available.
    R, align_dbg = None, {}
    kabsch_mtx = None
    if source_mesh is not None:
        try:
            kabsch_mtx = _kabsch_matrix_from_source(mesh, source_mesh)
        except Exception:
            import traceback
            traceback.print_exc()
            print("[BD_AutoRig] Kabsch align failed — falling back to "
                  "anatomical alignment")
    if kabsch_mtx is not None:
        mesh.matrix_world = kabsch_mtx @ mesh.matrix_world
        _bpy.ops.object.transform_apply(location=True, rotation=True,
                                           scale=True)
        debug["rotated_via"] = "kabsch_source_mesh"
    else:
        R, align_dbg = _build_align_rotation(mesh, bone_targets)
        debug["align"] = align_dbg

    if R is not None:
        # Apply the rotation matrix to mesh data
        rot_4 = R.to_4x4()
        mesh.matrix_world = rot_4 @ mesh.matrix_world
        _bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        debug["rotated_via"] = "anatomical_alignment"

        # ── 180°-around-up disambiguation (MESH vs BONES) ────────────────
        # Head-over-pelvis + arms-out are both 180°-symmetric, so the
        # anatomical R can land on the flipped solution: the MESH then
        # faces +Y while the untouched BONES still face -Y (UEFN), and
        # every downstream canonical step twists the feet 180°. The toes
        # settle it: mesh toe verts must protrude the SAME way as the
        # rig's foot→toe joint vector. If not, spin the MESH 180° Z.
        def _vg_centroid_y(names):
            for nm in names:
                vg = mesh.vertex_groups.get(nm)
                if vg is None:
                    continue
                ys, n = 0.0, 0
                for v in mesh.data.vertices:
                    for g in v.groups:
                        if g.group == vg.index and g.weight > 0.5:
                            ys += (mesh.matrix_world @ v.co).y; n += 1
                            break
                if n:
                    return ys / n
            return None
        toe_m = _vg_centroid_y(("ball_l", "LeftToeBase",
                                   "mixamorig:LeftToeBase"))
        foot_m = _vg_centroid_y(("foot_l", "LeftFoot",
                                    "mixamorig:LeftFoot"))
        toe_b = next((h for h in (_bone_head(n) for n in
                        ("ball_l", "LeftToeBase", "mixamorig:LeftToeBase"))
                        if h), None)
        foot_b = next((h for h in (_bone_head(n) for n in
                         ("foot_l", "LeftFoot", "mixamorig:LeftFoot"))
                         if h), None)
        if (toe_m is not None and foot_m is not None
                and toe_b is not None and foot_b is not None):
            mesh_toe_dir = toe_m - foot_m
            bone_toe_dir = (toe_b - foot_b).y
            if (abs(mesh_toe_dir) > 0.02 and abs(bone_toe_dir) > 0.02
                    and (mesh_toe_dir > 0) != (bone_toe_dir > 0)):
                from mathutils import Matrix as _M
                rz = _M.Rotation(radians(180), 4, "Z")
                mesh.matrix_world = rz @ mesh.matrix_world
                _bpy.ops.object.transform_apply(location=True, rotation=True,
                                                   scale=False)
                debug["mesh_facing_flip"] = True

        # ── Forward-direction normalization ──────────────────────────────
        # Anatomical alignment with up + lateral leaves a 180°-around-up
        # ambiguity (cross product handedness). Enforce face → +Y by
        # measuring where the mesh's face protrudes; if it's at -Y,
        # rotate BOTH the armature and mesh 180° around Z together so
        # they stay consistent.
        head_vg_name = _vg_name_for(mesh, "head")
        head_vg = mesh.vertex_groups.get(head_vg_name)
        face_dir = None
        if head_vg:
            idx = head_vg.index
            ys = []
            for v in mesh.data.vertices:
                for g in v.groups:
                    if g.group == idx and g.weight > 0.5:
                        ys.append((mesh.matrix_world @ v.co).y)
                        break
            if ys:
                y_mean = sum(ys) / len(ys)
                pos_ext = max((y - y_mean for y in ys), default=0)
                neg_ext = -min((y - y_mean for y in ys), default=0)
                face_dir = +1 if pos_ext > neg_ext else -1

        # NOTE on the previous 180°Z forward-flip step (REMOVED 2026-06-22):
        #
        # We previously rotated the whole rig 180° around Z whenever the
        # mesh's face landed at -Y, to "make the character face the camera
        # in Blender's default front view." That convenience hack broke
        # downstream TransferBones binding because a 180°Z rotation also
        # mirrors +X↔-X — the mesh's hand_l vgroup ends up at -X but UEFN's
        # canonical hand_l bone is at +X (Unreal convention: face=-Y,
        # left=+X). Same-named binding then put the mesh's "left" verts
        # under the bone on the wrong side, and skinning either deforms
        # the wrong side or doesn't deform at all.
        #
        # UEFN/Unreal expects face=-Y, left=+X. MIA's output is already in
        # that orientation. Trust it.
        debug["face_dir_y"] = face_dir
        debug["forward_flip"] = "disabled_breaks_uefn_lr_binding"
    elif kabsch_mtx is None:
        # Fallback: try Rx(-90)·Rz(180) (the empirically-correct sequence
        # for MIA's current output)
        mesh.rotation_euler = (radians(-90), 0, radians(180))
        _bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        debug["rotated_via"] = "fallback_rx90_rz180"

    # 6) Refine scale: now mesh is oriented correctly, scale so its Z extent
    #    matches the armature's exactly.
    mesh_z_min = min(v.co.z for v in mesh.data.vertices)
    mesh_z_max = max(v.co.z for v in mesh.data.vertices)
    mesh_h = mesh_z_max - mesh_z_min
    if mesh_h > 1e-6:
        refine_scale = target_h / mesh_h
        if abs(refine_scale - 1.0) > 0.01:
            mesh.scale = (refine_scale, refine_scale, refine_scale)
            _bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            debug["refine_scale"] = refine_scale

    # 7) Translate so feet on Z=0
    mz_min = min(v.co.z for v in mesh.data.vertices)
    if abs(mz_min) > 1e-4:
        mesh.location = (0, 0, -mz_min)
        _bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # 8) Re-parent to armature
    arm.select_set(True)
    _bpy.context.view_layer.objects.active = arm
    _bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    debug["mesh_world_bbox_after"] = [
        list(mesh.matrix_world @ Vector(mesh.bound_box[0])),
        list(mesh.matrix_world @ Vector(mesh.bound_box[6])),
    ]
    debug["arm_world_bbox_after"] = [
        list(arm.matrix_world @ Vector(arm.bound_box[0])),
        list(arm.matrix_world @ Vector(arm.bound_box[6])),
    ]
    return debug


def run_local_autorig(
    glb_path: Path,
    output_fbx: Path,
    *,
    no_fingers: bool = True,
    use_normal: bool = False,
    reset_to_rest: bool = True,
    progress_cb=None,
) -> tuple[bool, str]:
    """End-to-end local autorig:
      1. Ensure venv + MIA source + weights
      2. Run inference via venv subprocess
      3. Run Blender FBX assembly via subprocess
    Returns (ok, message).
    """
    if progress_cb is None:
        progress_cb = lambda msg: print(f"[BD_AutoRig:local] {msg}", flush=True)

    # 1) Bootstrap (idempotent)
    if not _bootstrap.is_bootstrapped():
        progress_cb("First run — bootstrapping venv (this can take "
                     "5–15 min depending on network)…")
        ok = _bootstrap.bootstrap(progress_cb=progress_cb)
        if not ok:
            return False, "Bootstrap failed — see system console"

    # 1.5) Ensure weights
    ok = _bootstrap.ensure_mia_weights(progress_cb=progress_cb)
    if not ok:
        return False, "MIA weight download failed"

    venv_py = _bootstrap.venv_python()

    # 2) Inference subprocess
    work_dir = Path(tempfile.mkdtemp(prefix="bd_autorig_"))
    progress_cb(f"Running MIA inference (output → {work_dir})…")
    runner_script = _THIS_DIR / "autorig_runner.py"
    cmd = [
        str(venv_py), str(runner_script),
        "--input", str(glb_path),
        "--output_dir", str(work_dir),
    ]
    if no_fingers:    cmd.append("--no_fingers")
    if use_normal:    cmd.append("--use_normal")
    if reset_to_rest: cmd.append("--reset_to_rest")

    env = os.environ.copy()
    env["BD_AUTORIG_CACHE"] = str(_bootstrap.actual_cache_root())
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=600, errors="replace")
    if proc.stdout:
        for line in proc.stdout.splitlines():
            progress_cb(f"  inf: {line}")
    if proc.returncode != 0:
        progress_cb(f"  inf STDERR:\n{proc.stderr[-2000:]}")
        return False, f"Inference failed (rc={proc.returncode})"

    # 3) Blender FBX assembly subprocess
    progress_cb("Assembling rigged FBX via Blender…")
    json_path = work_dir / "data.json"
    cmd = [
        _blender_binary(),
        "--background",
        "--python", str(MIA_EXPORT_SCRIPT),
        "--",
        "--input_path", str(json_path),
        "--output_path", str(output_fbx),
        "--template_path", str(MIXAMO_TEMPLATE),
    ]
    if no_fingers:    cmd.append("--remove_fingers")
    if reset_to_rest: cmd.append("--reset_to_rest")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            errors="replace")
    if proc.stdout:
        for line in proc.stdout.splitlines()[-30:]:
            progress_cb(f"  fbx: {line}")
    if proc.returncode != 0:
        progress_cb(f"  fbx STDERR:\n{proc.stderr[-2000:]}")
        return False, f"FBX assembly failed (rc={proc.returncode})"
    if not output_fbx.exists():
        return False, f"FBX not written: {output_fbx}"

    progress_cb(f"OK — rigged FBX at {output_fbx}")
    return True, str(output_fbx)
