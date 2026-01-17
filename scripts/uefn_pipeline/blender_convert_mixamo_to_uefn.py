"""
Blender script to convert Mixamo rigged FBX to UEFN Manny skeleton.

This script:
1. Loads a Mixamo-rigged FBX
2. Renames all bones from Mixamo to UEFN naming convention
3. Renames vertex groups to match
4. Re-exports as UEFN-compatible FBX

Usage: blender --background --python blender_convert_mixamo_to_uefn.py -- <input_fbx> <output_fbx>
"""

import bpy
import sys
import os

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_convert_mixamo_to_uefn.py -- <input_fbx> <output_fbx>")
    sys.exit(1)

input_fbx = argv[0]
output_fbx = argv[1]

print(f"[Mixamo->UEFN] Input: {input_fbx}")
print(f"[Mixamo->UEFN] Output: {output_fbx}")

# ============================================================================
# MIXAMO -> UEFN BONE NAME MAPPING
# ============================================================================
MIXAMO_TO_UEFN = {
    # Spine chain
    "mixamorig:Hips": "pelvis",
    "mixamorig:Spine": "spine_01",
    "mixamorig:Spine1": "spine_02",
    "mixamorig:Spine2": "spine_03",

    # Head/Neck
    "mixamorig:Neck": "neck_01",
    "mixamorig:Head": "head",

    # Left Arm
    "mixamorig:LeftShoulder": "clavicle_l",
    "mixamorig:LeftArm": "upperarm_l",
    "mixamorig:LeftForeArm": "lowerarm_l",
    "mixamorig:LeftHand": "hand_l",

    # Left Fingers
    "mixamorig:LeftHandThumb1": "thumb_01_l",
    "mixamorig:LeftHandThumb2": "thumb_02_l",
    "mixamorig:LeftHandThumb3": "thumb_03_l",
    "mixamorig:LeftHandIndex1": "index_01_l",
    "mixamorig:LeftHandIndex2": "index_02_l",
    "mixamorig:LeftHandIndex3": "index_03_l",
    "mixamorig:LeftHandMiddle1": "middle_01_l",
    "mixamorig:LeftHandMiddle2": "middle_02_l",
    "mixamorig:LeftHandMiddle3": "middle_03_l",
    "mixamorig:LeftHandRing1": "ring_01_l",
    "mixamorig:LeftHandRing2": "ring_02_l",
    "mixamorig:LeftHandRing3": "ring_03_l",
    "mixamorig:LeftHandPinky1": "pinky_01_l",
    "mixamorig:LeftHandPinky2": "pinky_02_l",
    "mixamorig:LeftHandPinky3": "pinky_03_l",

    # Right Arm
    "mixamorig:RightShoulder": "clavicle_r",
    "mixamorig:RightArm": "upperarm_r",
    "mixamorig:RightForeArm": "lowerarm_r",
    "mixamorig:RightHand": "hand_r",

    # Right Fingers
    "mixamorig:RightHandThumb1": "thumb_01_r",
    "mixamorig:RightHandThumb2": "thumb_02_r",
    "mixamorig:RightHandThumb3": "thumb_03_r",
    "mixamorig:RightHandIndex1": "index_01_r",
    "mixamorig:RightHandIndex2": "index_02_r",
    "mixamorig:RightHandIndex3": "index_03_r",
    "mixamorig:RightHandMiddle1": "middle_01_r",
    "mixamorig:RightHandMiddle2": "middle_02_r",
    "mixamorig:RightHandMiddle3": "middle_03_r",
    "mixamorig:RightHandRing1": "ring_01_r",
    "mixamorig:RightHandRing2": "ring_02_r",
    "mixamorig:RightHandRing3": "ring_03_r",
    "mixamorig:RightHandPinky1": "pinky_01_r",
    "mixamorig:RightHandPinky2": "pinky_02_r",
    "mixamorig:RightHandPinky3": "pinky_03_r",

    # Left Leg
    "mixamorig:LeftUpLeg": "thigh_l",
    "mixamorig:LeftLeg": "calf_l",
    "mixamorig:LeftFoot": "foot_l",
    "mixamorig:LeftToeBase": "ball_l",

    # Right Leg
    "mixamorig:RightUpLeg": "thigh_r",
    "mixamorig:RightLeg": "calf_r",
    "mixamorig:RightFoot": "foot_r",
    "mixamorig:RightToeBase": "ball_r",
}


def clean_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)


def find_armature():
    """Find the armature object."""
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def find_meshes():
    """Find all mesh objects."""
    return [obj for obj in bpy.data.objects if obj.type == 'MESH']


def rename_bones(armature_obj):
    """Rename armature bones from Mixamo to UEFN."""
    renamed = 0
    not_found = []

    # Need to be in edit mode to rename bones
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    for bone in armature_obj.data.edit_bones:
        old_name = bone.name
        if old_name in MIXAMO_TO_UEFN:
            bone.name = MIXAMO_TO_UEFN[old_name]
            print(f"[Mixamo->UEFN] Renamed bone: {old_name} -> {bone.name}")
            renamed += 1
        else:
            not_found.append(old_name)

    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[Mixamo->UEFN] Renamed {renamed} bones")
    if not_found:
        print(f"[Mixamo->UEFN] Bones not in mapping (kept original names): {not_found}")

    return renamed


def rename_vertex_groups(mesh_obj):
    """Rename vertex groups from Mixamo to UEFN."""
    renamed = 0

    for vg in mesh_obj.vertex_groups:
        old_name = vg.name
        if old_name in MIXAMO_TO_UEFN:
            vg.name = MIXAMO_TO_UEFN[old_name]
            renamed += 1

    print(f"[Mixamo->UEFN] Renamed {renamed} vertex groups on {mesh_obj.name}")
    return renamed


def add_root_bone(armature_obj):
    """Add a root bone if pelvis exists but root doesn't."""
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_obj.data.edit_bones

    # Check if root already exists
    if 'root' in edit_bones:
        print("[Mixamo->UEFN] Root bone already exists")
        bpy.ops.object.mode_set(mode='OBJECT')
        return False

    # Find pelvis
    pelvis = edit_bones.get('pelvis')
    if not pelvis:
        print("[Mixamo->UEFN] No pelvis bone found, skipping root creation")
        bpy.ops.object.mode_set(mode='OBJECT')
        return False

    # Create root bone
    root = edit_bones.new('root')
    root.head = (0, 0, 0)
    root.tail = (0, 0, 10)  # Point up

    # Reparent pelvis to root
    pelvis.parent = root

    bpy.ops.object.mode_set(mode='OBJECT')
    print("[Mixamo->UEFN] Added root bone and reparented pelvis")
    return True


def rename_armature(armature_obj):
    """Rename the armature object and data to 'root'."""
    armature_obj.name = "root"
    armature_obj.data.name = "root"
    print("[Mixamo->UEFN] Renamed armature to 'root'")


# ============================================================================
# MAIN
# ============================================================================
clean_scene()

# Import FBX
print(f"[Mixamo->UEFN] Importing: {input_fbx}")
try:
    bpy.ops.import_scene.fbx(filepath=input_fbx)
except Exception as e:
    print(f"[Mixamo->UEFN] Import failed: {e}")
    sys.exit(1)

# Find armature and meshes
armature = find_armature()
meshes = find_meshes()

if not armature:
    print("[Mixamo->UEFN] ERROR: No armature found in FBX!")
    sys.exit(1)

print(f"[Mixamo->UEFN] Found armature: {armature.name}")
print(f"[Mixamo->UEFN] Found {len(meshes)} mesh(es)")

# Rename bones
rename_bones(armature)

# Add root bone
add_root_bone(armature)

# Rename armature
rename_armature(armature)

# Rename vertex groups on all meshes
for mesh in meshes:
    rename_vertex_groups(mesh)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

# Export FBX
print(f"[Mixamo->UEFN] Exporting to: {output_fbx}")
try:
    # Select all relevant objects
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        use_selection=True,

        # Object settings
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,

        # Transform - UEFN uses Z-up
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Y',
        axis_up='Z',

        # Armature settings
        add_leaf_bones=False,  # CRITICAL: No _end bones
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        armature_nodetype='NULL',

        # Mesh settings
        mesh_smooth_type='FACE',
        use_tspace=True,

        # No animation
        bake_anim=False,

        # Embed textures
        path_mode='COPY',
        embed_textures=True,
    )

    print(f"[Mixamo->UEFN] Export successful!")
    print(f"[Mixamo->UEFN] Output: {output_fbx}")

except Exception as e:
    print(f"[Mixamo->UEFN] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[Mixamo->UEFN] Done!")
