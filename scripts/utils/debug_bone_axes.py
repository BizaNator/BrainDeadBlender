"""
Debug script to understand bone axis conventions between Source and Target armatures.
Run this to see what's going on with bone orientations.
"""
import bpy
from mathutils import Vector, Matrix

def find_collection_ci(name: str):
    want = name.strip().lower()
    for col in bpy.data.collections:
        if col.name.strip().lower() == want:
            return col
    return None

def find_single_armature(col):
    arms = [o for o in col.all_objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"Collection '{col.name}' must contain exactly 1 armature; found {len(arms)}.")
    return arms[0]

def analyze_bone(arm_obj, bone_name):
    """Analyze a bone's orientation."""
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None

    # Bone direction (head to tail) in armature local space
    direction = (b.tail_local - b.head_local).normalized()

    # Bone's local axes from its matrix
    # In Blender, bone.matrix_local transforms from bone space to armature space
    # Column 0 = X axis, Column 1 = Y axis (along bone), Column 2 = Z axis (up)
    mat = b.matrix_local
    x_axis = Vector(mat.col[0][:3]).normalized()
    y_axis = Vector(mat.col[1][:3]).normalized()  # This should be head->tail
    z_axis = Vector(mat.col[2][:3]).normalized()  # This is the "up" / roll axis

    return {
        'name': bone_name,
        'head': b.head_local.copy(),
        'tail': b.tail_local.copy(),
        'direction': direction,
        'length': (b.tail_local - b.head_local).length,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis,
    }

def print_bone_info(info, label):
    if info is None:
        print(f"  {label}: NOT FOUND")
        return
    print(f"  {label}: {info['name']}")
    print(f"    Head: ({info['head'].x:.3f}, {info['head'].y:.3f}, {info['head'].z:.3f})")
    print(f"    Tail: ({info['tail'].x:.3f}, {info['tail'].y:.3f}, {info['tail'].z:.3f})")
    print(f"    Direction: ({info['direction'].x:.3f}, {info['direction'].y:.3f}, {info['direction'].z:.3f})")
    print(f"    Length: {info['length']:.4f}")
    print(f"    Y-axis (bone forward): ({info['y_axis'].x:.3f}, {info['y_axis'].y:.3f}, {info['y_axis'].z:.3f})")
    print(f"    Z-axis (bone up): ({info['z_axis'].x:.3f}, {info['z_axis'].y:.3f}, {info['z_axis'].z:.3f})")

def main():
    src_col = find_collection_ci("Source")
    tgt_col = find_collection_ci("Target")

    src_arm = find_single_armature(src_col)
    tgt_arm = find_single_armature(tgt_col)

    print("=" * 60)
    print("BONE AXIS DEBUG")
    print("=" * 60)

    # Key bones to compare
    pairs = [
        ("pelvis", "Hips"),
        ("spine_01", "Spine"),
        ("upperarm_l", "LeftArm"),
        ("lowerarm_l", "LeftForeArm"),
        ("hand_l", "LeftHand"),
        ("thigh_l", "LeftUpLeg"),
        ("calf_l", "LeftLeg"),
        ("foot_l", "LeftFoot"),
    ]

    for src_name, tgt_name in pairs:
        print(f"\n--- {src_name} <-> {tgt_name} ---")
        src_info = analyze_bone(src_arm, src_name)
        tgt_info = analyze_bone(tgt_arm, tgt_name)
        print_bone_info(src_info, "SOURCE (UEFN)")
        print_bone_info(tgt_info, "TARGET (H3D)")

        if src_info and tgt_info:
            # How different are the directions?
            import math
            angle = math.degrees(src_info['direction'].angle(tgt_info['direction']))
            print(f"    >>> Direction difference: {angle:.1f}°")

main()
