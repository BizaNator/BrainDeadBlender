"""
Blender script to export UniRig/MIA rigged mesh to FBX with UEFN Manny skeleton.

This script takes skeleton data from pickle (UniRig format) and creates a rigged mesh
with UEFN bone names instead of Mixamo, ready for import into Unreal Engine / Fortnite.

Usage: blender --background --python blender_export_uefn.py -- <input_pkl> <output_fbx> [options]

Key differences from Mixamo export:
- Uses UEFN bone names (pelvis, spine_01, etc.) instead of Mixamo (mixamorig:Hips, etc.)
- Keeps Z-up coordinate system (no Y-up conversion)
- Uses centimeter scale (no 100x scale for Mixamo)
- UEFN bone orientations: all bones point +Y in local space
"""

import bpy
import sys
import os
import pickle
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from collections import defaultdict
import math

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_uefn.py -- <input_pkl> <output_fbx> [options]")
    sys.exit(1)

input_pkl = argv[0]
output_fbx = argv[1]

# Parse optional parameters
extrude_size = 0.03
add_root = True  # UEFN has a root bone
use_extrude_bone = True
use_connect_unique_child = True

for arg in argv[2:]:
    if arg.startswith("--extrude_size="):
        extrude_size = float(arg.split("=")[1])
    elif arg == "--no_root":
        add_root = False
    elif arg == "--no_extrude_bone":
        use_extrude_bone = False
    elif arg == "--no_connect_unique_child":
        use_connect_unique_child = False

print(f"[UEFN Export] Input: {input_pkl}")
print(f"[UEFN Export] Output: {output_fbx}")

# ============================================================================
# MIXAMO -> UEFN BONE NAME MAPPING
# ============================================================================
# This maps Mixamo bone names (from UniRig/MIA output) to UEFN Manny skeleton names
# Note: UEFN has additional bones (twist bones, IK bones) not present in Mixamo

MIXAMO_TO_UEFN = {
    # Spine chain
    "mixamorig:Hips": "pelvis",
    "mixamorig:Spine": "spine_01",
    "mixamorig:Spine1": "spine_02",
    "mixamorig:Spine2": "spine_03",
    # Note: UEFN has spine_04, spine_05 - we map Spine2 to spine_03

    # Head/Neck
    "mixamorig:Neck": "neck_01",
    # Note: UEFN has neck_02
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

# Reverse mapping for lookups
UEFN_TO_MIXAMO = {v: k for k, v in MIXAMO_TO_UEFN.items()}

# UEFN bone hierarchy (parent -> children)
# This defines the expected UEFN skeleton structure
UEFN_HIERARCHY = {
    "root": ["pelvis", "ik_foot_root", "ik_hand_root"],
    "pelvis": ["spine_01", "thigh_l", "thigh_r"],
    "spine_01": ["spine_02"],
    "spine_02": ["spine_03"],
    "spine_03": ["spine_04"],
    "spine_04": ["spine_05"],
    "spine_05": ["neck_01", "clavicle_l", "clavicle_r"],
    "neck_01": ["neck_02"],
    "neck_02": ["head"],
    "head": [],

    # Left arm
    "clavicle_l": ["upperarm_l"],
    "upperarm_l": ["upperarm_twist_01_l", "lowerarm_l"],
    "upperarm_twist_01_l": [],
    "lowerarm_l": ["lowerarm_twist_01_l", "hand_l"],
    "lowerarm_twist_01_l": [],
    "hand_l": ["thumb_01_l", "index_01_l", "middle_01_l", "ring_01_l", "pinky_01_l"],

    # Left fingers
    "thumb_01_l": ["thumb_02_l"],
    "thumb_02_l": ["thumb_03_l"],
    "thumb_03_l": [],
    "index_01_l": ["index_02_l"],
    "index_02_l": ["index_03_l"],
    "index_03_l": [],
    "middle_01_l": ["middle_02_l"],
    "middle_02_l": ["middle_03_l"],
    "middle_03_l": [],
    "ring_01_l": ["ring_02_l"],
    "ring_02_l": ["ring_03_l"],
    "ring_03_l": [],
    "pinky_01_l": ["pinky_02_l"],
    "pinky_02_l": ["pinky_03_l"],
    "pinky_03_l": [],

    # Right arm (mirror)
    "clavicle_r": ["upperarm_r"],
    "upperarm_r": ["upperarm_twist_01_r", "lowerarm_r"],
    "upperarm_twist_01_r": [],
    "lowerarm_r": ["lowerarm_twist_01_r", "hand_r"],
    "lowerarm_twist_01_r": [],
    "hand_r": ["thumb_01_r", "index_01_r", "middle_01_r", "ring_01_r", "pinky_01_r"],

    # Right fingers
    "thumb_01_r": ["thumb_02_r"],
    "thumb_02_r": ["thumb_03_r"],
    "thumb_03_r": [],
    "index_01_r": ["index_02_r"],
    "index_02_r": ["index_03_r"],
    "index_03_r": [],
    "middle_01_r": ["middle_02_r"],
    "middle_02_r": ["middle_03_r"],
    "middle_03_r": [],
    "ring_01_r": ["ring_02_r"],
    "ring_02_r": ["ring_03_r"],
    "ring_03_r": [],
    "pinky_01_r": ["pinky_02_r"],
    "pinky_02_r": ["pinky_03_r"],
    "pinky_03_r": [],

    # Left leg
    "thigh_l": ["thigh_twist_01_l", "calf_l"],
    "thigh_twist_01_l": [],
    "calf_l": ["calf_twist_01_l", "foot_l"],
    "calf_twist_01_l": [],
    "foot_l": ["ball_l"],
    "ball_l": [],

    # Right leg (mirror)
    "thigh_r": ["thigh_twist_01_r", "calf_r"],
    "thigh_twist_01_r": [],
    "calf_r": ["calf_twist_01_r", "foot_r"],
    "calf_twist_01_r": [],
    "foot_r": ["ball_r"],
    "ball_r": [],

    # IK bones
    "ik_foot_root": ["ik_foot_l", "ik_foot_r"],
    "ik_foot_l": [],
    "ik_foot_r": [],
    "ik_hand_root": ["ik_hand_gun", "ik_hand_l", "ik_hand_r"],
    "ik_hand_gun": [],
    "ik_hand_l": [],
    "ik_hand_r": [],
}


def map_bone_name(mixamo_name):
    """Convert Mixamo bone name to UEFN bone name."""
    if mixamo_name in MIXAMO_TO_UEFN:
        return MIXAMO_TO_UEFN[mixamo_name]
    # If not in mapping, try stripping mixamorig: prefix and lowercasing
    if mixamo_name.startswith("mixamorig:"):
        stripped = mixamo_name[10:].lower()
        # Common transformations
        stripped = stripped.replace("upleg", "thigh")
        stripped = stripped.replace("leg", "calf")
        stripped = stripped.replace("arm", "arm")
        return stripped
    return mixamo_name


def clean_bpy():
    """Clean the Blender scene."""
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)


# ============================================================================
# LOAD DATA
# ============================================================================
try:
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    # Convert to numpy arrays
    joints = np.array(data['joints'], dtype=np.float32)
    parents = data['parents']
    names = data['names']  # Mixamo names from UniRig/MIA

    # Optional data
    vertices = np.array(data['vertices'], dtype=np.float32) if data.get('vertices') is not None else None
    faces = np.array(data['faces'], dtype=np.int32) if data.get('faces') is not None else None
    skin = np.array(data['skin'], dtype=np.float32) if data.get('skin') is not None else None
    tails = np.array(data['tails'], dtype=np.float32) if data.get('tails') is not None else None

    # UV and texture data
    uv_coords = np.array(data['uv_coords'], dtype=np.float32) if data.get('uv_coords') is not None and len(data.get('uv_coords', [])) > 0 else None
    uv_faces = np.array(data['uv_faces'], dtype=np.int32) if data.get('uv_faces') is not None and len(data.get('uv_faces', [])) > 0 else None
    texture_data_base64 = data.get('texture_data_base64', "")
    texture_format = data.get('texture_format', "PNG")

    print(f"[UEFN Export] Loaded skeleton with {len(joints)} joints")
    print(f"[UEFN Export] Original bone names: {names[:5]}...")

    # Map bone names to UEFN
    uefn_names = [map_bone_name(n) for n in names]
    print(f"[UEFN Export] UEFN bone names: {uefn_names[:5]}...")

    if vertices is not None:
        print(f"[UEFN Export] Mesh: {len(vertices)} vertices, {len(faces)} faces")
    if skin is not None:
        print(f"[UEFN Export] Skin weights shape: {skin.shape}")

    # Debug bounds
    print(f"[UEFN Export] Joints bounds: {joints.min(axis=0)} to {joints.max(axis=0)}")
    if vertices is not None:
        print(f"[UEFN Export] Mesh bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")

except Exception as e:
    print(f"[UEFN Export] Failed to load pickle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# NORMALIZE TO UEFN CONVENTIONS
# ============================================================================

# Check skeleton type by name patterns
is_mixamo = any(n.startswith('mixamorig:') for n in names)
print(f"[UEFN Export] Detected Mixamo skeleton: {is_mixamo}")

if is_mixamo and vertices is not None:
    print("[UEFN Export] Normalizing Mixamo output for UEFN...")

    # 1. SCALE: UniRig outputs normalized [-1, 1] range
    # UEFN expects real-world centimeter scale
    # Target: ~170cm height for humanoid
    current_height = vertices[:, 2].max() - vertices[:, 2].min()
    target_height = 170.0  # cm (UEFN uses centimeters)

    if current_height > 0.01:
        scale_factor = target_height / current_height
        print(f"[UEFN Export] Scaling from {current_height:.3f} to {target_height:.0f}cm (factor: {scale_factor:.2f}x)")

        vertices *= scale_factor
        joints *= scale_factor
        if tails is not None:
            tails *= scale_factor

    # 2. POSITION: Move feet to ground (Z=0)
    mesh_min_z = vertices[:, 2].min()
    z_offset = -mesh_min_z
    print(f"[UEFN Export] Moving feet from Z={mesh_min_z:.3f} to Z=0 (offset: {z_offset:.3f})")

    vertices[:, 2] += z_offset
    joints[:, 2] += z_offset
    if tails is not None:
        tails[:, 2] += z_offset

    # 3. ORIENTATION: Check if model faces correct direction
    # UEFN expects face toward +Y (or -Y depending on convention)
    # For now, keep as-is since UniRig should output correctly oriented

    print(f"[UEFN Export] Final mesh bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")


# ============================================================================
# CREATE BLENDER SCENE
# ============================================================================
clean_bpy()

# Create armature
armature = bpy.data.armatures.new("root")
armature_obj = bpy.data.objects.new("root", armature)
bpy.context.collection.objects.link(armature_obj)
bpy.context.view_layer.objects.active = armature_obj

# Enter edit mode to create bones
bpy.ops.object.mode_set(mode='EDIT')
edit_bones = armature.edit_bones

# Build parent index map
parent_map = {}
for i, parent_idx in enumerate(parents):
    if parent_idx >= 0:
        parent_map[i] = parent_idx

# Generate tails if not provided
if tails is None:
    print("[UEFN Export] Generating bone tails...")
    tails = np.zeros_like(joints)
    for i in range(len(joints)):
        # Find children
        children = [j for j, p in enumerate(parents) if p == i]
        if children:
            # Average of children positions
            child_positions = joints[children]
            tails[i] = child_positions.mean(axis=0)
        else:
            # Leaf bone - extend in parent direction or default
            if i in parent_map:
                parent_pos = joints[parent_map[i]]
                direction = joints[i] - parent_pos
                if np.linalg.norm(direction) > 0.001:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.array([0, 1, 0])  # Default +Y
                tails[i] = joints[i] + direction * extrude_size * 100  # Scale for visibility
            else:
                tails[i] = joints[i] + np.array([0, 0, extrude_size * 100])

# Create bones with UEFN names
bone_objects = {}
for i, mixamo_name in enumerate(names):
    uefn_name = map_bone_name(mixamo_name)

    bone = edit_bones.new(uefn_name)
    bone.head = Vector(joints[i].tolist())
    bone.tail = Vector(tails[i].tolist())

    # Ensure minimum bone length
    if (bone.tail - bone.head).length < 0.01:
        bone.tail = bone.head + Vector((0, 0, 1))

    bone_objects[i] = bone
    print(f"[UEFN Export] Created bone: {mixamo_name} -> {uefn_name}")

# Set up parent relationships
for i, parent_idx in enumerate(parents):
    if parent_idx >= 0 and parent_idx in bone_objects:
        bone_objects[i].parent = bone_objects[parent_idx]

        # Connect if tail matches head
        if use_connect_unique_child:
            parent_bone = bone_objects[parent_idx]
            child_bone = bone_objects[i]
            if (parent_bone.tail - child_bone.head).length < 0.1:
                child_bone.use_connect = True

# Add root bone if requested (UEFN standard)
if add_root:
    root_bone = edit_bones.new("root")
    root_bone.head = Vector((0, 0, 0))
    root_bone.tail = Vector((0, 0, 10))

    # Find pelvis and reparent to root
    pelvis_bone = edit_bones.get("pelvis")
    if pelvis_bone:
        pelvis_bone.parent = root_bone

    print("[UEFN Export] Added root bone")

bpy.ops.object.mode_set(mode='OBJECT')
print(f"[UEFN Export] Created armature with {len(armature.bones)} bones")


# ============================================================================
# CREATE MESH
# ============================================================================
if vertices is not None and faces is not None:
    # Create mesh
    mesh = bpy.data.meshes.new("SKM_Character")
    mesh_obj = bpy.data.objects.new("SKM_Character", mesh)
    bpy.context.collection.objects.link(mesh_obj)

    # Set mesh data
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    # Add UV coordinates if available
    if uv_coords is not None and uv_faces is not None:
        print(f"[UEFN Export] Adding UV coordinates: {len(uv_coords)} UVs")
        uv_layer = mesh.uv_layers.new(name="UVMap")

        # Flatten UVs for loop assignment
        for face_idx, face in enumerate(mesh.polygons):
            for loop_idx, loop in enumerate(face.loop_indices):
                if face_idx < len(uv_faces):
                    uv_idx = uv_faces[face_idx][loop_idx] if loop_idx < len(uv_faces[face_idx]) else 0
                    if uv_idx < len(uv_coords):
                        uv_layer.data[loop].uv = uv_coords[uv_idx].tolist()

    # Create vertex groups with UEFN names and assign weights
    if skin is not None:
        print("[UEFN Export] Assigning skin weights...")

        # Create vertex groups for each bone (using UEFN names)
        for i, mixamo_name in enumerate(names):
            uefn_name = map_bone_name(mixamo_name)
            if uefn_name not in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups.new(name=uefn_name)

        # Assign weights
        for v_idx in range(len(vertices)):
            for bone_idx in range(skin.shape[1]):
                weight = float(skin[v_idx, bone_idx])
                if weight > 0.001 and bone_idx < len(names):
                    uefn_name = map_bone_name(names[bone_idx])
                    vg = mesh_obj.vertex_groups.get(uefn_name)
                    if vg:
                        vg.add([v_idx], weight, 'REPLACE')

        print(f"[UEFN Export] Assigned weights to {len(mesh_obj.vertex_groups)} vertex groups")

    # Parent mesh to armature
    mesh_obj.parent = armature_obj
    mesh_obj.parent_type = 'ARMATURE'

    # Add armature modifier
    arm_mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
    arm_mod.object = armature_obj
    arm_mod.use_vertex_groups = True
    arm_mod.use_bone_envelopes = False

    print(f"[UEFN Export] Created mesh: {len(vertices)} vertices, {len(faces)} faces")


# ============================================================================
# EXPORT FBX
# ============================================================================
print(f"[UEFN Export] Exporting to: {output_fbx}")

try:
    # Select objects for export
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    if 'mesh_obj' in dir():
        mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # FBX export settings for UEFN
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        use_selection=True,
        use_active_collection=False,

        # Object settings
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,

        # Transform settings - UEFN uses Z-up
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Y',
        axis_up='Z',

        # Armature settings
        add_leaf_bones=False,  # CRITICAL: Prevents _end bones in Unreal
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        armature_nodetype='NULL',
        use_armature_deform_only=False,

        # Mesh settings
        mesh_smooth_type='FACE',
        use_mesh_edges=False,
        use_tspace=True,

        # Animation (disabled for base mesh)
        bake_anim=False,

        # Path mode
        path_mode='COPY',
        embed_textures=True,
    )

    print(f"[UEFN Export] Successfully exported to: {output_fbx}")

except Exception as e:
    print(f"[UEFN Export] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[UEFN Export] Done!")
