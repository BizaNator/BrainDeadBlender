"""
merge_head_to_body.py

Combines a face-base head (LowPolyHead_Rigged + Eye_L/R + Teeth_* +
MouthCavity + SM_EyelidPlane on a 56-bone face armature) with a UEFN body
(body mesh + 74-bone UE5 mannequin armature with limbs/IK but no face).

How the two skeletons relate
----------------------------
- Body armature: full UE5 mannequin -- pelvis / spine_01..05 / clavicle_*
  / neck_01..02 / upperarm/lowerarm/hand/fingers / thigh/calf/foot / twist
  bones / IK helpers. **Stops at neck_02. No `head` bone. No face bones.**
- Head armature: pelvis / spine_01..05 / clavicle_* / neck_01..02 PLUS
  `head` + `faceAttach` + ~40 face bones (C_jaw, L_eye, lids, brows,
  lip controls, dyn_hair, FX_Eye*, hat, earpiece, etc.). No limb bones.

Shared bones (pelvis, spine_*, clavicle_*, neck_01..02, attach) differ in
rest position by ~5-30mm in Y because the face_base pipeline retargets
the head's skeleton to its anatomy. Body mesh weights are baked to body
armature's exact positions, so we MUST NOT move body bones.

Merge strategy
--------------
1. Body armature is primary. Keep its 74 bones at their exact positions.
2. For each head-only bone (head, faceAttach, face bones, dyn_hair_*,
   FX_*, etc.), add it to the body armature at its head_armature
   position translated so the bone's parent attach point matches the
   body armature's equivalent location.
   - `head` becomes a child of body's `neck_02`
   - `faceAttach`, hat, earpiece, etc. become children of `head` as before
   - All face bones keep their parent relationships
3. Apply the alignment translation to head meshes so they sit at the
   body's neck (head mesh moves by `head_arm.neck_02 - body_arm.neck_02`,
   negated -- i.e., head meshes move OPPOSITE to that delta).
4. Re-skin every head mesh's Armature modifier to the merged body armature.
   Vertex group names already match the new bones on body armature.
5. Optionally delete the head's now-empty source armature.

After merge: single armature drives body mesh AND head meshes. Single FBX
export = full UEFN character.
"""

import bpy
from mathutils import Vector, Matrix


CONFIG = {
    "head_armature": "root",            # face_base output -- becomes empty after merge
    "body_armature": "root.001",        # imported UEFN body skeleton, becomes the merged armature

    "head_meshes": (
        "LowPolyHead_Rigged",
        "Eye_L", "Eye_R",
        "Teeth_Upper", "Teeth_Lower",
        "MouthCavity",
        "SM_EyelidPlane",
    ),
    "body_meshes": (
        "Mesh_0.001_Remesh to HardBody",
    ),

    # Bone used to align the head onto the body. We translate head meshes
    # so head_arm.<anchor>.world matches body_arm.<anchor>.world.
    # neck_02 is the chain attach point; using it lines up the neck/head
    # transition rather than only the head bone (which doesn't exist on
    # body armature).
    "alignment_bone": "neck_02",

    # When adding head-only bones to body, where should the `head` bone
    # be parented? Default = body's `neck_02` (since body doesn't have
    # a `head` bone).
    "head_parent_on_body": "neck_02",

    "delete_head_armature": False,
    "dry_run": False,
    "verbose": True,
}


def _world_bone_head(arm_obj, bone_name):
    b = arm_obj.data.bones.get(bone_name)
    if b is None:
        return None
    return arm_obj.matrix_world @ b.head_local


def merge_head_to_body(cfg):
    print(f"=== merge_head_to_body ===")
    head_arm = bpy.data.objects.get(cfg["head_armature"])
    body_arm = bpy.data.objects.get(cfg["body_armature"])
    if head_arm is None or head_arm.type != 'ARMATURE':
        raise RuntimeError(f"head armature '{cfg['head_armature']}' missing")
    if body_arm is None or body_arm.type != 'ARMATURE':
        raise RuntimeError(f"body armature '{cfg['body_armature']}' missing")

    # 1) Compute mesh alignment translation
    align = cfg["alignment_bone"]
    h_anchor = _world_bone_head(head_arm, align)
    b_anchor = _world_bone_head(body_arm, align)
    if h_anchor is None or b_anchor is None:
        raise RuntimeError(f"alignment bone '{align}' missing on one armature")
    # Head meshes need to move so their head_arm.anchor world position
    # ends up at body_arm.anchor world position.
    mesh_offset = b_anchor - h_anchor
    print(f"  alignment via '{align}':")
    print(f"    head_arm: ({h_anchor.x*1000:+.1f}, {h_anchor.y*1000:+.1f}, {h_anchor.z*1000:+.1f}) mm")
    print(f"    body_arm: ({b_anchor.x*1000:+.1f}, {b_anchor.y*1000:+.1f}, {b_anchor.z*1000:+.1f}) mm")
    print(f"    mesh offset (apply to head meshes): "
          f"({mesh_offset.x*1000:+.1f}, {mesh_offset.y*1000:+.1f}, {mesh_offset.z*1000:+.1f}) mm")

    # 2) Identify head-only bones (to be added to body)
    body_bone_names = set(b.name for b in body_arm.data.bones)
    head_bone_names = set(b.name for b in head_arm.data.bones)
    add_bones = sorted(head_bone_names - body_bone_names)
    shared_bones = sorted(head_bone_names & body_bone_names)
    print(f"  bones: head={len(head_bone_names)}  body={len(body_bone_names)}")
    print(f"    shared (will use body positions): {len(shared_bones)} -> {shared_bones}")
    print(f"    to add to body: {len(add_bones)}")

    if cfg.get("dry_run"):
        print("  [dry_run] no changes applied")
        return 0

    # 3) Enter Edit Mode on body armature to add the head-only bones.
    #    Snapshot each head bone's edit-mode transform from the source.
    bpy.context.view_layer.objects.active = head_arm
    bpy.ops.object.mode_set(mode='EDIT')
    head_edit = {}
    for n in add_bones:
        eb = head_arm.data.edit_bones[n]
        head_edit[n] = {
            "head": eb.head.copy(),
            "tail": eb.tail.copy(),
            "roll": eb.roll,
            "parent": eb.parent.name if eb.parent else None,
            "use_connect": eb.use_connect,
            "use_deform": eb.use_deform,
        }
    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply the same mesh_offset to the bone positions when copying so head
    # bones land in the same world-space slot relative to body's neck_02
    # as they were relative to head's neck_02.
    bpy.context.view_layer.objects.active = body_arm
    bpy.ops.object.mode_set(mode='EDIT')
    new_bones = {}
    parent_on_body = cfg.get("head_parent_on_body")
    for n in add_bones:
        d = head_edit[n]
        eb = body_arm.data.edit_bones.new(n)
        eb.head = d["head"] + mesh_offset
        eb.tail = d["tail"] + mesh_offset
        eb.roll = d["roll"]
        eb.use_connect = d["use_connect"]
        eb.use_deform = d["use_deform"]
        new_bones[n] = eb
    # Second pass: hook up parents now that all bones exist
    for n in add_bones:
        d = head_edit[n]
        p = d["parent"]
        eb = new_bones[n]
        if p is None:
            # head's `root` bone goes under body's pelvis (or root if exists)
            target = body_arm.data.edit_bones.get("pelvis")
            if target is not None and n != "pelvis":
                eb.parent = target
        elif p in new_bones:
            eb.parent = new_bones[p]
        elif p in body_arm.data.edit_bones:
            eb.parent = body_arm.data.edit_bones[p]
        else:
            # parent was a shared bone that lived on head_arm but not body
            # (unlikely since the diff was clean). Fall back to head_parent_on_body.
            target = body_arm.data.edit_bones.get(parent_on_body)
            eb.parent = target

    # Special-case: if `head` got added and its parent is `neck_02`, that's
    # already correct since body has neck_02. But head_arm's `head` parent
    # was already `neck_02` (verified in audit), so no override needed.

    bpy.ops.object.mode_set(mode='OBJECT')
    if cfg.get("verbose"):
        merged_bones = sorted(b.name for b in body_arm.data.bones)
        print(f"  merged armature now has {len(merged_bones)} bones")

    # 4) Translate + re-skin head meshes
    moved = 0
    for mn in cfg["head_meshes"]:
        obj = bpy.data.objects.get(mn)
        if obj is None or obj.type != 'MESH':
            print(f"    skip '{mn}': not in scene")
            continue
        # Translate world position
        obj.matrix_world.translation = obj.matrix_world.translation + mesh_offset
        # Swap Armature mod
        for m in obj.modifiers:
            if m.type == 'ARMATURE':
                old = m.object.name if m.object else None
                m.object = body_arm
                if cfg.get("verbose"):
                    print(f"    {mn}: Armature mod object {old} -> {body_arm.name}")
        # Reparent preserving world transform
        if obj.parent is not body_arm:
            world = obj.matrix_world.copy()
            obj.parent = body_arm
            obj.matrix_parent_inverse = body_arm.matrix_world.inverted() @ world @ obj.matrix_basis.inverted()
        moved += 1

    if cfg.get("delete_head_armature"):
        print(f"  deleting source head armature '{head_arm.name}'")
        arm_data = head_arm.data
        bpy.data.objects.remove(head_arm, do_unlink=True)
        if arm_data.users == 0:
            bpy.data.armatures.remove(arm_data)

    print(f"  [done] merged {moved} head meshes onto body armature with {len(add_bones)} added bones")
    return moved


if __name__ == "__main__":
    merge_head_to_body(CONFIG)
