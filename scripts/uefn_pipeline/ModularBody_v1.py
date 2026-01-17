"""
ModularBody V1 - Modular Character Body Pipeline

Operations:
1. Attach detailed hand meshes (replacing AI-generated hands)
2. Attach detailed foot meshes (replacing AI-generated feet)
3. Separate head (for Mutable modularity)

All while maintaining UEFN skeleton compatibility.

KEY INSIGHT: No separate rigging needed for hands/feet!
The body already has the UEFN skeleton with finger/toe bones.
We just merge geometry and transfer weights from SKM_UEFN_Mannequin.

SCENE SETUP:
- Body Collection: UEFN armature + body mesh with existing weights
- Hands Collection: Hand_L, Hand_R with origin at wrist center
- Feet Collection: Foot_L, Foot_R with origin at ankle center
- Source Collection: SKM_UEFN_Mannequin for weight reference
"""

import bpy
import bmesh
import fnmatch
from datetime import datetime
from mathutils import Vector, Matrix

LOG_TEXT_NAME = "ModularBody_V1_Log.txt"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Operation modes
MODE_ATTACH_HANDS = "hands"
MODE_ATTACH_FEET = "feet"
MODE_SEPARATE_HEAD = "head"
MODE_ALL = "all"

# Collection names
BODY_COLLECTION = "Body"
HANDS_COLLECTION = "Hands"
FEET_COLLECTION = "Feet"
UEFN_REFERENCE = "Source"

# Bone names for cutting
HAND_BONES_L = [
    "hand_l",
    "thumb_01_l", "thumb_02_l", "thumb_03_l",
    "index_metacarpal_l", "index_01_l", "index_02_l", "index_03_l",
    "middle_metacarpal_l", "middle_01_l", "middle_02_l", "middle_03_l",
    "ring_metacarpal_l", "ring_01_l", "ring_02_l", "ring_03_l",
    "pinky_metacarpal_l", "pinky_01_l", "pinky_02_l", "pinky_03_l",
]

HAND_BONES_R = [b.replace("_l", "_r") for b in HAND_BONES_L]

FOOT_BONES_L = ["foot_l", "ball_l"]
FOOT_BONES_R = [b.replace("_l", "_r") for b in FOOT_BONES_L]

HEAD_BONES = ["head", "neck_02"]

# Weight threshold for cutting - lower = cut more aggressively
# 0.01 = removes any vertex with even tiny hand weight (most aggressive)
# 0.1 = removes vertices with noticeable hand influence
# 0.3 = only removes vertices strongly weighted to hand
WEIGHT_THRESHOLD = 0.01  # Very low to catch all hand geometry including thumb

# Merge distance for boundary vertices - keep VERY SMALL!
# Large values will merge vertices across the body mesh itself (bad!)
MERGE_DISTANCE = 0.005  # Only merges truly overlapping vertices

# Skip automatic merge step - set False to attempt auto-merge at seam
SKIP_AUTO_MERGE = False

# Use shrinkwrap to close seam gap before joining
USE_SHRINKWRAP = True
SHRINKWRAP_OFFSET = 0.0

# Skip rotation alignment (set True if hand meshes are pre-oriented correctly)
SKIP_ROTATION = True

# Use position-based cut (at bone) vs weight-based cut
# Position cut uses known world directions (+X/-X for hands) and radius limit
USE_POSITION_CUT = True

# Radius for position-based cut (only vertices within this distance of bone are affected)
# Increase if not cutting enough, decrease if cutting into forearm
CUT_RADIUS_HAND = 0.35  # meters - needs to be large enough to capture whole hand area
CUT_RADIUS_FOOT = 0.20  # meters

# Auto-fit settings - uses old hand geometry to transform new hand
AUTO_FIT_SCALE = False      # DISABLED - let user pre-scale hand mesh
AUTO_FIT_POSITION = True    # Position new hand at old hand's wrist location
AUTO_FIT_ROTATION = False   # DISABLED - let user pre-orient hand mesh (rotation is complex)


# ============================================================================
# LOGGING
# ============================================================================

def log_to_text(s: str):
    """Write log to Blender text block."""
    txt = bpy.data.texts.get(LOG_TEXT_NAME)
    if not txt:
        txt = bpy.data.texts.new(LOG_TEXT_NAME)
    txt.clear()
    txt.write(s)


# ============================================================================
# COLLECTION / OBJECT HELPERS (from Pipeline_v31)
# ============================================================================

def find_collection_ci(name: str):
    """Find collection by name (case-insensitive)."""
    want = name.strip().lower()
    for col in bpy.data.collections:
        if col.name.strip().lower() == want:
            return col
    return None


def objects_in_collection(col):
    """Get all objects in collection."""
    return list(col.all_objects)


def find_single_armature(col):
    """Find the single armature in a collection."""
    arms = [o for o in objects_in_collection(col) if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"Collection '{col.name}' must contain exactly 1 armature; found {len(arms)}.")
    return arms[0]


def mesh_objects(col):
    """Get all mesh objects in collection."""
    return [o for o in objects_in_collection(col) if o.type == "MESH"]


def find_mesh_by_name_pattern(col, pattern: str):
    """Find mesh by name pattern (supports wildcards)."""
    meshes = mesh_objects(col)
    for m in meshes:
        if fnmatch.fnmatch(m.name.lower(), pattern.lower()):
            return m
    return None


# ============================================================================
# MODE / DEPSGRAPH HELPERS
# ============================================================================

def ensure_object_mode():
    """Ensure we're in object mode."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def ensure_edit_mode(obj):
    """Ensure we're in edit mode for specified object."""
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')


def depsgraph_update():
    """Force depsgraph update."""
    bpy.context.view_layer.update()


# ============================================================================
# TRANSFORM HELPERS
# ============================================================================

def apply_all_transforms(obj):
    """Apply all transforms to object."""
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    depsgraph_update()


# ============================================================================
# ARMATURE HELPERS
# ============================================================================

def bone_head_world(arm_obj, bone_name: str):
    """Get bone head position in world space."""
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    return arm_obj.matrix_world @ b.head_local


def bone_matrix_world(arm_obj, bone_name: str):
    """Get bone matrix in world space."""
    b = arm_obj.data.bones.get(bone_name)
    if not b:
        return None
    return arm_obj.matrix_world @ b.matrix_local


# ============================================================================
# VERTEX GROUP HELPERS
# ============================================================================

def get_vertex_weight(mesh_obj, vertex_index: int, group_name: str) -> float:
    """Get weight of vertex in specific group."""
    vg = mesh_obj.vertex_groups.get(group_name)
    if not vg:
        return 0.0
    try:
        return vg.weight(vertex_index)
    except RuntimeError:
        return 0.0


def get_combined_weight(mesh_obj, vertex_index: int, group_names: list) -> float:
    """Get combined weight of vertex across multiple groups."""
    total = 0.0
    for name in group_names:
        total += get_vertex_weight(mesh_obj, vertex_index, name)
    return min(total, 1.0)


def create_vertex_groups_from_list(mesh_obj, group_names: list, report: list):
    """Create empty vertex groups from a list of names."""
    existing = {vg.name for vg in mesh_obj.vertex_groups}
    created = 0
    for name in group_names:
        if name not in existing:
            mesh_obj.vertex_groups.new(name=name)
            created += 1
    report.append(f"[VGroups] Created {created} vertex groups.")
    return created


def create_vertex_group_for_all_verts(mesh_obj, group_name: str, report: list):
    """
    Create a vertex group containing ALL vertices of the mesh.
    Useful for tracking which vertices belong to an attached part after joining.
    """
    # Create or get the vertex group
    vg = mesh_obj.vertex_groups.get(group_name)
    if not vg:
        vg = mesh_obj.vertex_groups.new(name=group_name)

    # Add all vertices with weight 1.0
    all_indices = [v.index for v in mesh_obj.data.vertices]
    vg.add(all_indices, 1.0, 'REPLACE')

    report.append(f"[VGroups] Created '{group_name}' with {len(all_indices)} vertices.")
    return vg


def create_vertex_groups_from_armature(mesh_obj, arm_obj, report):
    """Create empty vertex groups for all bones in armature."""
    existing = {vg.name for vg in mesh_obj.vertex_groups}
    created = 0
    for bone in arm_obj.data.bones:
        if bone.name not in existing:
            mesh_obj.vertex_groups.new(name=bone.name)
            created += 1
    report.append(f"[VGroups] Created {created} vertex groups from armature bones.")
    return created


# ============================================================================
# GEOMETRY ANALYSIS (for auto-fitting new hand to old hand location)
# ============================================================================

def analyze_hand_geometry(mesh_obj, bone_names: list, threshold: float = 0.01):
    """
    Analyze the geometry of vertices weighted to hand bones.
    Returns dict with: center, bounds_min, bounds_max, size, wrist_center

    wrist_center is the center of the "base" of the hand (closest to arm)
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    # Get vertices weighted to hand bones
    hand_verts = []
    for v in mesh.vertices:
        weight = get_combined_weight(mesh_obj, v.index, bone_names)
        if weight > threshold:
            # World position
            world_pos = mesh_obj.matrix_world @ v.co
            hand_verts.append(world_pos)

    if not hand_verts:
        return None

    # Calculate bounding box
    xs = [v.x for v in hand_verts]
    ys = [v.y for v in hand_verts]
    zs = [v.z for v in hand_verts]

    bounds_min = Vector((min(xs), min(ys), min(zs)))
    bounds_max = Vector((max(xs), max(ys), max(zs)))
    size = bounds_max - bounds_min
    center = (bounds_min + bounds_max) / 2

    # Find wrist center (vertices closest to arm - highest X for left, lowest X for right)
    # We'll use the centroid of the 20% of vertices closest to the arm
    hand_verts_sorted = sorted(hand_verts, key=lambda v: v.x)

    # Determine if left or right by checking if center is positive or negative X
    is_left = center.x > 0

    # Get wrist vertices (20% closest to body)
    wrist_count = max(3, len(hand_verts) // 5)
    if is_left:
        # Left hand - wrist is at lower X values (toward body center)
        wrist_verts = hand_verts_sorted[:wrist_count]
    else:
        # Right hand - wrist is at higher X values (toward body center)
        wrist_verts = hand_verts_sorted[-wrist_count:]

    wrist_center = Vector((
        sum(v.x for v in wrist_verts) / len(wrist_verts),
        sum(v.y for v in wrist_verts) / len(wrist_verts),
        sum(v.z for v in wrist_verts) / len(wrist_verts)
    ))

    return {
        'center': center,
        'bounds_min': bounds_min,
        'bounds_max': bounds_max,
        'size': size,
        'wrist_center': wrist_center,
        'vertex_count': len(hand_verts),
        'is_left': is_left
    }


def analyze_mesh_geometry(mesh_obj):
    """
    Analyze the geometry of an entire mesh.
    Returns dict with: center, bounds_min, bounds_max, size, base_center

    base_center is the center of vertices closest to the origin (the attachment point)
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    if not mesh.vertices:
        return None

    # Get all vertices in world space
    verts = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]

    # Calculate bounding box
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]

    bounds_min = Vector((min(xs), min(ys), min(zs)))
    bounds_max = Vector((max(xs), max(ys), max(zs)))
    size = bounds_max - bounds_min
    center = (bounds_min + bounds_max) / 2

    # Find base center (vertices closest to mesh origin)
    origin = mesh_obj.matrix_world.translation
    verts_by_dist = sorted(verts, key=lambda v: (v - origin).length)

    # Get base vertices (20% closest to origin)
    base_count = max(3, len(verts) // 5)
    base_verts = verts_by_dist[:base_count]

    base_center = Vector((
        sum(v.x for v in base_verts) / len(base_verts),
        sum(v.y for v in base_verts) / len(base_verts),
        sum(v.z for v in base_verts) / len(base_verts)
    ))

    return {
        'center': center,
        'bounds_min': bounds_min,
        'bounds_max': bounds_max,
        'size': size,
        'base_center': base_center,
        'vertex_count': len(verts)
    }


def find_boundary_center_near_point(mesh_obj, target_point: Vector, max_distance: float = 0.3, report: list = None):
    """
    Find the center of boundary (non-manifold) vertices near a target point.
    This finds where the arm actually ends after cutting.

    target_point: World space point to search near (e.g., bone position)
    max_distance: Only consider boundaries within this distance
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    # Use bmesh for faster boundary detection
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    boundary_verts = []
    for v in bm.verts:
        # Check if vertex is on boundary (has any boundary edge)
        is_boundary = any(e.is_boundary for e in v.link_edges)
        if is_boundary:
            world_pos = mesh_obj.matrix_world @ v.co
            # Only include if within max_distance of target
            if (world_pos - target_point).length <= max_distance:
                boundary_verts.append(world_pos)

    bm.free()

    if not boundary_verts:
        if report:
            report.append(f"[Boundary] No boundary verts found within {max_distance}m of target")
        return None

    # Calculate center
    center = Vector((
        sum(v.x for v in boundary_verts) / len(boundary_verts),
        sum(v.y for v in boundary_verts) / len(boundary_verts),
        sum(v.z for v in boundary_verts) / len(boundary_verts)
    ))

    if report:
        report.append(f"[Boundary] Found {len(boundary_verts)} boundary verts near target, center: {center}")

    return center


def fit_new_hand_to_old(new_hand_mesh, old_hand_info, new_hand_info, arm_obj, bone_name: str, side: str, report: list, body_mesh=None):
    """
    Transform the new hand mesh to fit where the old hand was.
    - Positions so wrist aligns with arm boundary (where we cut)
    """
    target_pos = None

    # Get bone position as reference point for boundary search
    bone_pos = bone_head_world(arm_obj, bone_name)

    # Try to find arm boundary near the bone (where we just cut)
    if body_mesh and bone_pos:
        boundary_center = find_boundary_center_near_point(body_mesh, bone_pos, max_distance=0.3, report=report)
        if boundary_center:
            target_pos = boundary_center

    # Fallback to old wrist center
    if target_pos is None and old_hand_info:
        target_pos = old_hand_info['wrist_center']
        report.append(f"[Fit] Fallback to old wrist center: {target_pos}")

    # Final fallback to bone position
    if target_pos is None:
        report.append("[Fit] WARNING: No boundary found, using bone position.")
        target_pos = bone_pos

    # Position the hand
    if AUTO_FIT_POSITION and target_pos:
        new_hand_mesh.location = target_pos
        report.append(f"[Fit] Positioned hand at: {target_pos}")

    # Rotation: match the arm direction
    if AUTO_FIT_ROTATION:
        # Get forearm bone direction
        forearm_name = "lowerarm_l" if side.upper() == 'L' else "lowerarm_r"
        forearm_bone = arm_obj.data.bones.get(forearm_name)

        if forearm_bone:
            # Get forearm direction in world space
            forearm_head = arm_obj.matrix_world @ forearm_bone.head_local
            forearm_tail = arm_obj.matrix_world @ forearm_bone.tail_local
            arm_direction = (forearm_tail - forearm_head).normalized()

            # The hand mesh should point in the arm direction
            # Assuming the hand mesh is modeled with fingers pointing toward +Y or -Y
            # We need to rotate it to point along arm_direction

            # Create rotation to align hand with arm direction
            from mathutils import Quaternion
            import math

            # Assuming hand mesh fingers point toward -Y (forward in Blender)
            # We want to rotate so -Y aligns with arm_direction
            hand_forward = Vector((0, -1, 0))

            # Calculate rotation from hand_forward to arm_direction
            rotation = hand_forward.rotation_difference(arm_direction)

            new_hand_mesh.rotation_mode = 'QUATERNION'
            new_hand_mesh.rotation_quaternion = rotation
            report.append(f"[Fit] Rotated to match arm direction: {arm_direction}")
        else:
            report.append("[Fit] WARNING: Forearm bone not found, skipping rotation")

    # Apply transforms
    depsgraph_update()
    apply_all_transforms(new_hand_mesh)
    report.append("[Fit] Applied transforms.")


# ============================================================================
# CUTTING OPERATIONS
# ============================================================================

def bisect_and_delete_hand(mesh_obj, arm_obj, bone_name: str, side: str, radius: float, report: list = None):
    """
    Clean cut approach:
    1. Select vertices within radius of wrist AND past the cut plane (hand side)
    2. Delete those vertices
    3. This leaves a clean edge at the cut boundary

    Uses the forearm bone direction as the cut plane normal to handle angled arms.
    radius: Only affect vertices within this distance of the wrist bone
    """
    bone = arm_obj.data.bones.get(bone_name)
    if not bone:
        raise RuntimeError(f"Bone '{bone_name}' not found.")

    # Get the forearm bone to determine actual arm direction
    forearm_name = "lowerarm_l" if side.upper() == 'L' else "lowerarm_r"
    forearm_bone = arm_obj.data.bones.get(forearm_name)

    # Get bone head position in world space (the wrist)
    bone_head = arm_obj.matrix_world @ bone.head_local

    # Use forearm direction as plane normal (points from elbow toward wrist/hand)
    if forearm_bone:
        # Forearm: head is elbow, tail is wrist
        forearm_head_world = arm_obj.matrix_world @ forearm_bone.head_local
        forearm_tail_world = arm_obj.matrix_world @ forearm_bone.tail_local
        plane_normal = (forearm_tail_world - forearm_head_world).normalized()
        if report:
            report.append(f"[Bisect] Using forearm bone direction for cut plane.")
    else:
        # Fallback to world X if no forearm bone
        if side.upper() == 'L':
            plane_normal = Vector((1.0, 0.0, 0.0))
        else:
            plane_normal = Vector((-1.0, 0.0, 0.0))
        if report:
            report.append(f"[Bisect] WARNING: Forearm bone not found, using world X direction.")

    if report:
        report.append(f"[Bisect] Bone '{bone_name}' head (wrist): {bone_head}")
        report.append(f"[Bisect] Plane normal: {plane_normal}")
        report.append(f"[Bisect] Radius limit: {radius}")

    # Select vertices that are:
    # 1. Within radius of the wrist bone position
    # 2. On the "outer" side of the cut plane (in direction of normal = toward hand)
    ensure_object_mode()
    mesh = mesh_obj.data

    # Deselect all vertices first
    for v in mesh.vertices:
        v.select = False

    selected_count = 0
    for v in mesh.vertices:
        # Get vertex position in world space
        v_world = mesh_obj.matrix_world @ v.co

        # Vector from wrist to vertex
        to_vert = v_world - bone_head

        # Check if within radius
        if to_vert.length > radius:
            continue

        # Check if on the "hand" side of the plane (dot product > 0 means same direction as normal)
        dot = to_vert.dot(plane_normal)
        if dot > 0:  # Vertex is past the wrist, toward the hand
            v.select = True
            selected_count += 1

    if report:
        report.append(f"[Bisect] Selected {selected_count} vertices to delete (within radius, past wrist)")

    if selected_count == 0:
        if report:
            report.append(f"[Bisect] WARNING: No vertices selected! Try increasing CUT_RADIUS_HAND.")
        return

    # Delete selected vertices
    ensure_edit_mode(mesh_obj)
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()

    if report:
        report.append(f"[Bisect] Cut complete - hand portion removed.")


def cut_at_bone_position(mesh_obj, arm_obj, bone_name: str, cut_direction: Vector,
                         radius: float = 0.15, report: list = None):
    """
    Cut mesh at bone position by selecting and deleting vertices:
    - Within 'radius' of bone head
    - In the 'cut_direction' from bone head

    cut_direction: World space direction toward extremity (e.g., +X for left hand)
    radius: Only affect vertices within this distance
    """
    # Select vertices past the bone within radius
    count = select_vertices_past_bone(mesh_obj, arm_obj, bone_name, cut_direction, radius, report)

    if count == 0:
        if report:
            report.append("[Cut] WARNING: No vertices found to cut!")
        return

    # Delete selected vertices
    ensure_edit_mode(mesh_obj)

    # The vertices should already be selected from select_vertices_past_bone
    # But we need to refresh selection in edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()

    if report:
        report.append(f"[Cut] Deleted {count} vertices at '{bone_name}'.")


def select_vertices_by_weight(mesh_obj, bone_names: list, threshold: float = 0.1):
    """
    Select vertices that have combined weight above threshold for given bones.
    Returns list of selected vertex indices.
    """
    ensure_object_mode()

    mesh = mesh_obj.data
    selected_indices = []

    for i, v in enumerate(mesh.vertices):
        weight = get_combined_weight(mesh_obj, i, bone_names)
        if weight > threshold:
            selected_indices.append(i)

    return selected_indices


def select_vertices_past_bone_position(mesh_obj, arm_obj, bone_name: str, side: str, radius: float, report: list = None, offset: float = 0.0):
    """
    Select vertices that are past the bone position (toward the extremity).
    Uses forearm direction to determine which side is "past" the wrist.

    offset: Positive = move cut point toward body (cuts more), negative = toward extremity
    Returns list of vertex indices.
    """
    bone = arm_obj.data.bones.get(bone_name)
    if not bone:
        if report:
            report.append(f"[Cut] ERROR: Bone '{bone_name}' not found!")
        return []

    # Get forearm bone for direction
    forearm_name = "lowerarm_l" if side.upper() == 'L' else "lowerarm_r"
    forearm_bone = arm_obj.data.bones.get(forearm_name)

    bone_head = arm_obj.matrix_world @ bone.head_local

    # Get direction toward hand (from elbow to wrist)
    if forearm_bone:
        forearm_head = arm_obj.matrix_world @ forearm_bone.head_local
        forearm_tail = arm_obj.matrix_world @ forearm_bone.tail_local
        hand_direction = (forearm_tail - forearm_head).normalized()
    else:
        # Fallback to world X
        hand_direction = Vector((1.0, 0.0, 0.0)) if side.upper() == 'L' else Vector((-1.0, 0.0, 0.0))

    # Apply offset - move cut point toward body (opposite of hand direction)
    cut_point = bone_head - (hand_direction * offset)

    if report:
        report.append(f"[Cut] Bone head: {bone_head}")
        report.append(f"[Cut] Cut point (with offset {offset}): {cut_point}")
        report.append(f"[Cut] Direction toward hand: {hand_direction}")

    ensure_object_mode()
    mesh = mesh_obj.data
    selected_indices = []

    for v in mesh.vertices:
        v_world = mesh_obj.matrix_world @ v.co
        to_vert = v_world - cut_point

        # Check if within radius AND past the cut point (in hand direction)
        if to_vert.length <= radius:
            dot = to_vert.dot(hand_direction)
            if dot > 0:  # Past the cut point toward hand
                selected_indices.append(v.index)

    return selected_indices


def select_vertices_past_foot_position(mesh_obj, arm_obj, bone_name: str, side: str, radius: float, report: list = None, offset: float = 0.0):
    """
    Select vertices that are past the foot bone position (toward the foot/toes).
    Uses calf direction to determine which side is "past" the ankle.

    offset: Positive = move cut point toward body (cuts more), negative = toward extremity
    Returns list of vertex indices.
    """
    bone = arm_obj.data.bones.get(bone_name)
    if not bone:
        if report:
            report.append(f"[Cut] ERROR: Bone '{bone_name}' not found!")
        return []

    # Get calf bone for direction
    calf_name = "calf_l" if side.upper() == 'L' else "calf_r"
    calf_bone = arm_obj.data.bones.get(calf_name)

    bone_head = arm_obj.matrix_world @ bone.head_local

    # Get direction toward foot (from knee to ankle, then continuing down)
    if calf_bone:
        calf_head = arm_obj.matrix_world @ calf_bone.head_local
        calf_tail = arm_obj.matrix_world @ calf_bone.tail_local
        foot_direction = (calf_tail - calf_head).normalized()
    else:
        # Fallback to -Z (downward)
        foot_direction = Vector((0.0, 0.0, -1.0))

    # Apply offset - move cut point toward body (opposite of foot direction)
    cut_point = bone_head - (foot_direction * offset)

    if report:
        report.append(f"[Cut] Bone head: {bone_head}")
        report.append(f"[Cut] Cut point (with offset {offset}): {cut_point}")
        report.append(f"[Cut] Direction toward foot: {foot_direction}")

    ensure_object_mode()
    mesh = mesh_obj.data
    selected_indices = []

    for v in mesh.vertices:
        v_world = mesh_obj.matrix_world @ v.co
        to_vert = v_world - cut_point

        # Check if within radius AND past the cut point (in foot direction)
        if to_vert.length <= radius:
            dot = to_vert.dot(foot_direction)
            if dot > 0:  # Past the cut point toward foot
                selected_indices.append(v.index)

    return selected_indices


def delete_vertices_by_indices(mesh_obj, indices: list, report: list):
    """Delete vertices by their indices using bmesh."""
    ensure_object_mode()

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Select vertices by index
    mesh = mesh_obj.data
    for i in indices:
        mesh.vertices[i].select = True

    # Delete in edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')

    report.append(f"[Cut] Deleted {len(indices)} vertices.")
    depsgraph_update()


def cut_hand_from_body(body_mesh, arm_obj, side: str, report: list):
    """
    Cut hand vertices from body mesh.
    Uses weight-based selection first, falls back to position-based.
    side: 'L' or 'R'
    """
    bone_names = HAND_BONES_L if side.upper() == 'L' else HAND_BONES_R
    bone_name = "hand_l" if side.upper() == 'L' else "hand_r"

    report.append(f"[Cut] Removing hand vertices (side={side})...")

    # Try weight-based selection first
    indices = select_vertices_by_weight(body_mesh, bone_names, WEIGHT_THRESHOLD)
    report.append(f"[Cut] Weight-based found {len(indices)} vertices.")

    # If weight-based finds too few, try position-based
    MIN_EXPECTED_VERTS = 20
    if len(indices) < MIN_EXPECTED_VERTS:
        report.append(f"[Cut] Too few ({len(indices)} < {MIN_EXPECTED_VERTS}), trying position-based...")
        indices = select_vertices_past_bone_position(
            body_mesh, arm_obj, bone_name, side,
            radius=CUT_RADIUS_HAND,
            report=report
        )
        report.append(f"[Cut] Position-based found {len(indices)} vertices.")

    if indices:
        delete_vertices_by_indices(body_mesh, indices, report)
    else:
        report.append("[Cut] WARNING: No vertices found for hand!")


def bisect_and_delete_foot(mesh_obj, arm_obj, bone_name: str, side: str, radius: float, report: list = None):
    """
    Clean cut approach for feet:
    1. Select vertices within radius of ankle AND past the cut plane (foot side)
    2. Delete those vertices

    Uses the calf bone direction to handle leg angles.
    radius: Only affect vertices within this distance of the ankle bone
    """
    bone = arm_obj.data.bones.get(bone_name)
    if not bone:
        raise RuntimeError(f"Bone '{bone_name}' not found.")

    # Get the calf bone to determine actual leg direction at ankle
    calf_name = "calf_l" if side.upper() == 'L' else "calf_r"
    calf_bone = arm_obj.data.bones.get(calf_name)

    # Get bone head position in world space (the ankle)
    bone_head = arm_obj.matrix_world @ bone.head_local

    # Use calf direction as plane normal (points from knee toward ankle/foot)
    if calf_bone:
        calf_head_world = arm_obj.matrix_world @ calf_bone.head_local
        calf_tail_world = arm_obj.matrix_world @ calf_bone.tail_local
        plane_normal = (calf_tail_world - calf_head_world).normalized()
        if report:
            report.append(f"[Bisect] Using calf bone direction for cut plane.")
    else:
        # Fallback to -Z (downward) if no calf bone
        plane_normal = Vector((0.0, 0.0, -1.0))
        if report:
            report.append(f"[Bisect] WARNING: Calf bone not found, using -Z direction.")

    if report:
        report.append(f"[Bisect] Bone '{bone_name}' head (ankle): {bone_head}")
        report.append(f"[Bisect] Plane normal: {plane_normal}")
        report.append(f"[Bisect] Radius limit: {radius}")

    # Select vertices that are:
    # 1. Within radius of the ankle bone position
    # 2. On the "outer" side of the cut plane (in direction of normal = toward foot)
    ensure_object_mode()
    mesh = mesh_obj.data

    # Deselect all vertices first
    for v in mesh.vertices:
        v.select = False

    selected_count = 0
    for v in mesh.vertices:
        # Get vertex position in world space
        v_world = mesh_obj.matrix_world @ v.co

        # Vector from ankle to vertex
        to_vert = v_world - bone_head

        # Check if within radius
        if to_vert.length > radius:
            continue

        # Check if on the "foot" side of the plane (dot product > 0 means same direction as normal)
        dot = to_vert.dot(plane_normal)
        if dot > 0:  # Vertex is past the ankle, toward the foot
            v.select = True
            selected_count += 1

    if report:
        report.append(f"[Bisect] Selected {selected_count} vertices to delete (within radius, past ankle)")

    if selected_count == 0:
        if report:
            report.append(f"[Bisect] WARNING: No vertices selected! Try increasing CUT_RADIUS_FOOT.")
        return

    # Delete selected vertices
    ensure_edit_mode(mesh_obj)
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    depsgraph_update()

    if report:
        report.append(f"[Bisect] Cut complete - foot portion removed.")


def cut_foot_from_body(body_mesh, arm_obj, side: str, report: list):
    """
    Cut foot vertices from body mesh.
    Uses weight-based selection first, falls back to position-based.
    side: 'L' or 'R'
    """
    bone_names = FOOT_BONES_L if side.upper() == 'L' else FOOT_BONES_R
    bone_name = "foot_l" if side.upper() == 'L' else "foot_r"

    report.append(f"[Cut] Removing foot vertices (side={side})...")

    # Try weight-based selection first
    indices = select_vertices_by_weight(body_mesh, bone_names, WEIGHT_THRESHOLD)
    report.append(f"[Cut] Weight-based found {len(indices)} vertices.")

    # If weight-based finds too few, try position-based
    MIN_EXPECTED_VERTS = 10
    if len(indices) < MIN_EXPECTED_VERTS:
        report.append(f"[Cut] Too few ({len(indices)} < {MIN_EXPECTED_VERTS}), trying position-based...")
        indices = select_vertices_past_foot_position(
            body_mesh, arm_obj, bone_name, side,
            radius=CUT_RADIUS_FOOT,
            report=report
        )
        report.append(f"[Cut] Position-based found {len(indices)} vertices.")

    if indices:
        delete_vertices_by_indices(body_mesh, indices, report)
    else:
        report.append("[Cut] WARNING: No vertices found for foot!")


# ============================================================================
# POSITIONING OPERATIONS
# ============================================================================

def position_mesh_to_bone(mesh_obj, arm_obj, bone_name: str, report: list, apply_rotation: bool = False, offset: float = 0.0, offset_bone: str = None):
    """
    Position mesh so its origin aligns with bone head position.
    Assumes mesh origin is already set at the attachment point (e.g., wrist).

    offset: Distance to move the mesh BACK toward the body (along the parent bone direction)
    offset_bone: The bone to use for offset direction (e.g., lowerarm_l for hands)
    If apply_rotation=True, also aligns rotation to bone.
    """
    bone_pos = bone_head_world(arm_obj, bone_name)

    if bone_pos is None:
        raise RuntimeError(f"Bone '{bone_name}' not found in armature.")

    # Apply offset if specified
    if offset != 0.0 and offset_bone:
        parent_bone = arm_obj.data.bones.get(offset_bone)
        if parent_bone:
            # Get direction from parent bone (toward the extremity)
            parent_head = arm_obj.matrix_world @ parent_bone.head_local
            parent_tail = arm_obj.matrix_world @ parent_bone.tail_local
            direction = (parent_tail - parent_head).normalized()
            # Move position BACK toward body (opposite of direction)
            bone_pos = bone_pos - (direction * offset)
            report.append(f"[Position] Applied offset {offset}m back toward body")

    # Set location
    mesh_obj.location = bone_pos
    report.append(f"[Position] Moved '{mesh_obj.name}' to {bone_pos}")

    # Optionally set rotation to match bone
    if apply_rotation:
        bone_mat = bone_matrix_world(arm_obj, bone_name)
        if bone_mat:
            mesh_obj.rotation_mode = 'QUATERNION'
            mesh_obj.rotation_quaternion = bone_mat.to_quaternion()
            report.append(f"[Position] Applied bone rotation.")
    else:
        report.append(f"[Position] Rotation unchanged (SKIP_ROTATION=True).")

    depsgraph_update()


# ============================================================================
# MERGE OPERATIONS
# ============================================================================

def add_shrinkwrap_modifier(target_mesh, source_mesh, vertex_group: str = None):
    """Add shrinkwrap modifier to conform target to source surface."""
    mod = target_mesh.modifiers.new(name="Shrinkwrap_Wrist", type='SHRINKWRAP')
    mod.target = source_mesh
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    mod.wrap_mode = 'ON_SURFACE'

    if vertex_group and vertex_group in target_mesh.vertex_groups:
        mod.vertex_group = vertex_group

    return mod


def apply_modifier(mesh_obj, mod_name: str):
    """Apply modifier by name."""
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.modifier_apply(modifier=mod_name)
    depsgraph_update()


def join_meshes(target_mesh, source_mesh, report: list):
    """
    Join source mesh into target mesh.
    Target mesh becomes the combined result.
    NOTE: source_mesh will be deleted after this operation!
    """
    # Save names before join (source will be deleted)
    source_name = source_mesh.name
    target_name = target_mesh.name

    ensure_object_mode()

    bpy.ops.object.select_all(action='DESELECT')
    source_mesh.select_set(True)
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh

    bpy.ops.object.join()
    depsgraph_update()

    report.append(f"[Merge] Joined '{source_name}' into '{target_name}'.")


def select_boundary_edges(mesh_obj):
    """Select boundary (non-manifold) edges in edit mode."""
    ensure_edit_mode(mesh_obj)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True,
                                      use_multi_face=False, use_non_contiguous=False,
                                      use_verts=False)


def bridge_edge_loops(mesh_obj, report: list):
    """
    Attempt to bridge boundary edge loops.
    Must be called in edit mode with edges selected.
    """
    try:
        bpy.ops.mesh.bridge_edge_loops()
        report.append("[Merge] Bridged edge loops successfully.")
        return True
    except RuntimeError as e:
        report.append(f"[Merge] Bridge failed: {e}")
        return False


def merge_by_distance(mesh_obj, distance: float, report: list, boundary_only: bool = True):
    """
    Merge vertices by distance.
    If boundary_only=True, only merge non-manifold (boundary) vertices to preserve mesh detail.
    """
    ensure_edit_mode(mesh_obj)

    if boundary_only:
        # Only select boundary/non-manifold edges (the seam)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold(extend=False, use_wire=True, use_boundary=True,
                                          use_multi_face=False, use_non_contiguous=False,
                                          use_verts=True)
        report.append("[Merge] Selected boundary vertices only.")
    else:
        bpy.ops.mesh.select_all(action='SELECT')

    bpy.ops.mesh.remove_doubles(threshold=distance)
    bpy.ops.object.mode_set(mode='OBJECT')
    report.append(f"[Merge] Merged vertices by distance ({distance}).")


def add_data_transfer_normals(target_mesh, source_mesh, report: list):
    """Add Data Transfer modifier for normal blending at seam."""
    mod = target_mesh.modifiers.new(name="DT_Normals", type='DATA_TRANSFER')
    mod.object = source_mesh
    mod.use_loop_data = True
    mod.data_types_loops = {'CUSTOM_NORMAL'}
    mod.loop_mapping = 'NEAREST_POLYNOR'

    # Apply
    apply_modifier(target_mesh, mod.name)
    report.append("[Merge] Applied normal transfer from source mesh.")


def shrinkwrap_wrist_to_body(hand_mesh, body_mesh, report: list):
    """
    Shrinkwrap ONLY the boundary edge at the wrist to conform to body surface.
    Only affects actual boundary vertices at the wrist, not fingertips or other areas.
    """
    import bmesh

    ensure_object_mode()
    mesh = hand_mesh.data

    # Find ALL boundary vertices and their distances from origin
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    # Collect boundary vertices with their distances
    boundary_verts = []
    for v in bm.verts:
        if any(e.is_boundary for e in v.link_edges):
            dist = v.co.length  # Distance from mesh origin (which should be at wrist)
            boundary_verts.append((v.index, dist))

    bm.free()

    if not boundary_verts:
        report.append("[Shrinkwrap] No boundary vertices found - skipping.")
        return

    # Sort by distance from origin
    boundary_verts.sort(key=lambda x: x[1])

    # Find the wrist boundary (closest to origin)
    # The wrist edge loop should all be at similar distance from origin
    # Fingertip boundaries will be much further away
    min_boundary_dist = boundary_verts[0][1]

    # Only include boundary verts within 3cm of the closest boundary vertex
    # This should capture only the wrist edge loop, not fingertips
    wrist_threshold = min_boundary_dist + 0.03  # 3cm tolerance

    wrist_indices = [idx for idx, dist in boundary_verts if dist <= wrist_threshold]

    if not wrist_indices:
        report.append("[Shrinkwrap] No wrist boundary vertices found - skipping.")
        return

    report.append(f"[Shrinkwrap] Found {len(wrist_indices)} wrist boundary verts (out of {len(boundary_verts)} total boundaries)")

    # Create vertex group for wrist boundary only
    vg = hand_mesh.vertex_groups.get("Wrist_Shrinkwrap")
    if not vg:
        vg = hand_mesh.vertex_groups.new(name="Wrist_Shrinkwrap")

    vg.add(wrist_indices, 1.0, 'REPLACE')
    report.append(f"[Shrinkwrap] Created wrist vertex group with {len(wrist_indices)} vertices.")

    # Add shrinkwrap modifier targeting body
    mod = hand_mesh.modifiers.new(name="Shrinkwrap_Wrist", type='SHRINKWRAP')
    mod.target = body_mesh
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    mod.wrap_mode = 'ON_SURFACE'
    mod.vertex_group = "Wrist_Shrinkwrap"
    mod.offset = SHRINKWRAP_OFFSET

    # Apply the modifier
    apply_modifier(hand_mesh, mod.name)
    report.append("[Shrinkwrap] Applied shrinkwrap to wrist boundary only.")


def manual_bridge_boundaries(mesh_obj, bone_pos: Vector, threshold: float, report: list):
    """
    Manually bridge two boundary loops with mismatched vertex counts.
    Detects separate boundary loops and creates triangular faces to connect them.
    """
    import bmesh
    from mathutils import kdtree

    ensure_object_mode()

    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Find all boundary edges near the bone
    boundary_edges = []
    for e in bm.edges:
        if e.is_boundary:
            edge_center = mesh_obj.matrix_world @ ((e.verts[0].co + e.verts[1].co) / 2)
            dist = (edge_center - bone_pos).length
            if dist <= threshold:
                boundary_edges.append(e)

    if len(boundary_edges) < 2:
        report.append(f"[ManualBridge] Only {len(boundary_edges)} boundary edges - not enough.")
        bm.free()
        return False

    # Group edges into connected loops
    def find_connected_loop(start_edge, all_edges):
        """Find all edges connected to start_edge via shared vertices."""
        loop = {start_edge}
        verts_in_loop = {start_edge.verts[0], start_edge.verts[1]}
        remaining = set(all_edges) - loop

        changed = True
        while changed:
            changed = False
            for e in list(remaining):
                if e.verts[0] in verts_in_loop or e.verts[1] in verts_in_loop:
                    loop.add(e)
                    verts_in_loop.add(e.verts[0])
                    verts_in_loop.add(e.verts[1])
                    remaining.remove(e)
                    changed = True

        return loop, verts_in_loop

    # Find all separate boundary loops
    remaining_edges = set(boundary_edges)
    loops = []

    while remaining_edges:
        start = next(iter(remaining_edges))
        loop_edges, loop_verts = find_connected_loop(start, remaining_edges)
        loops.append(list(loop_verts))
        remaining_edges -= loop_edges

    report.append(f"[ManualBridge] Found {len(loops)} separate boundary loops.")

    if len(loops) < 2:
        report.append("[ManualBridge] Need at least 2 loops to bridge.")
        bm.free()
        return False

    # Sort loops by vertex count - we'll connect the smaller to larger
    loops.sort(key=lambda l: len(l))

    # Take the two loops with most vertices (arm and hand wrist, not fingertips)
    loop1 = loops[-1]  # Largest
    loop2 = loops[-2]  # Second largest

    report.append(f"[ManualBridge] Bridging loops with {len(loop1)} and {len(loop2)} verts.")

    # Use smaller loop as base, connect to larger
    if len(loop1) < len(loop2):
        small_loop, large_loop = loop1, loop2
    else:
        small_loop, large_loop = loop2, loop1

    # Build KD tree for larger loop
    kd = kdtree.KDTree(len(large_loop))
    for i, v in enumerate(large_loop):
        kd.insert(v.co, i)
    kd.balance()

    # For each vertex in small loop, create faces connecting to nearest in large loop
    faces_created = 0

    # Track which large loop verts have been used
    large_used = {}

    for sv in small_loop:
        # Find nearest 2 vertices in large loop
        nearest = kd.find_n(sv.co, 2)
        if len(nearest) >= 2:
            lv1 = large_loop[nearest[0][1]]
            lv2 = large_loop[nearest[1][1]]

            # Create triangle
            try:
                face = bm.faces.new([sv, lv1, lv2])
                faces_created += 1
            except ValueError:
                pass  # Face exists or is invalid

    # Also create faces from the other direction to fill gaps
    kd2 = kdtree.KDTree(len(small_loop))
    for i, v in enumerate(small_loop):
        kd2.insert(v.co, i)
    kd2.balance()

    for lv in large_loop:
        nearest = kd2.find_n(lv.co, 2)
        if len(nearest) >= 2:
            sv1 = small_loop[nearest[0][1]]
            sv2 = small_loop[nearest[1][1]]

            try:
                face = bm.faces.new([lv, sv1, sv2])
                faces_created += 1
            except ValueError:
                pass  # Face exists or is invalid

    report.append(f"[ManualBridge] Created {faces_created} bridging faces.")

    if faces_created > 0:
        # Recalculate normals
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh_obj.data)
        bm.free()
        return True

    bm.free()
    return False


def close_seam(mesh_obj, bone_pos: Vector, report: list):
    """
    Close the seam between body and attached part.
    Finds the TWO boundary edge loops closest to the bone position (arm + hand wrist)
    and bridges them, handling mismatched vertex counts with grid fill.
    """
    import bmesh

    ensure_object_mode()
    mesh = mesh_obj.data

    # Use bmesh to find and group boundary edges into loops
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Find all boundary edges and their distance to bone
    boundary_edges = []
    for e in bm.edges:
        if e.is_boundary:
            # Calculate edge center in world space
            edge_center = (mesh_obj.matrix_world @ e.verts[0].co + mesh_obj.matrix_world @ e.verts[1].co) / 2
            dist = (edge_center - bone_pos).length
            boundary_edges.append((e.index, dist, e.verts[0].index, e.verts[1].index))

    if not boundary_edges:
        report.append("[Seam] No boundary edges found - seam may already be closed.")
        bm.free()
        return

    # Sort by distance to bone
    boundary_edges.sort(key=lambda x: x[1])

    # Find the closest boundary edge distance
    min_dist = boundary_edges[0][1]

    # The seam edges should be within ~8cm of each other (arm edge + hand edge)
    # Fingertip boundaries will be 15-25cm away from wrist
    seam_threshold = min_dist + 0.08  # 8cm from closest edge

    # Select only edges within the seam threshold
    seam_edge_indices = set()
    seam_vert_indices = set()
    for edge_idx, dist, v0, v1 in boundary_edges:
        if dist <= seam_threshold:
            seam_edge_indices.add(edge_idx)
            seam_vert_indices.add(v0)
            seam_vert_indices.add(v1)

    bm.free()

    report.append(f"[Seam] Total boundary edges: {len(boundary_edges)}")
    report.append(f"[Seam] Seam edges (within {seam_threshold:.3f}m of bone): {len(seam_edge_indices)}")
    report.append(f"[Seam] Seam vertices: {len(seam_vert_indices)}")

    if len(seam_edge_indices) == 0:
        report.append("[Seam] No seam edges found near bone position.")
        return

    # First, merge any overlapping vertices at the seam only
    ensure_edit_mode(mesh_obj)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Select only the seam vertices
    for v in mesh.vertices:
        v.select = v.index in seam_vert_indices

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=0.01)  # 1cm - just for overlapping

    # Re-select seam edges for bridging
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Need to refresh bmesh after remove_doubles
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Re-find seam edges after merge (indices may have changed)
    seam_edge_count = 0
    for e in bm.edges:
        if e.is_boundary:
            edge_center = (mesh_obj.matrix_world @ e.verts[0].co + mesh_obj.matrix_world @ e.verts[1].co) / 2
            dist = (edge_center - bone_pos).length
            if dist <= seam_threshold:
                e.select = True
                seam_edge_count += 1

    bm.to_mesh(mesh_obj.data)
    bm.free()

    report.append(f"[Seam] Selected {seam_edge_count} edges for bridging after merge.")

    # Try to bridge the edge loops
    bpy.ops.object.mode_set(mode='EDIT')

    success = False

    # Method 0: Fill holes near seam only
    # First, fill holes but limit to the seam area by checking hole size
    # The seam hole should be the only significant hole near the wrist
    bpy.ops.mesh.select_all(action='DESELECT')

    # Select only boundary edges near the seam for fill
    bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.edges.ensure_lookup_table()

    seam_boundary_count = 0
    for e in bm.edges:
        if e.is_boundary:
            edge_center = (mesh_obj.matrix_world @ e.verts[0].co + mesh_obj.matrix_world @ e.verts[1].co) / 2
            dist = (edge_center - bone_pos).length
            if dist <= seam_threshold:
                e.select = True
                for v in e.verts:
                    v.select = True
                seam_boundary_count += 1

    bm.to_mesh(mesh_obj.data)
    bm.free()
    bpy.ops.object.mode_set(mode='EDIT')

    report.append(f"[Seam] Selected {seam_boundary_count} boundary edges for fill.")

    # Try different fill methods
    # Method 0a: Try to connect boundary loops by creating edge-face
    try:
        bpy.ops.mesh.edge_face_add()
        report.append("[Seam] Edge-face add successful!")
        # Triangulate the result to clean up n-gons
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        report.append("[Seam] Triangulated result.")
        success = True
    except RuntimeError as e:
        report.append(f"[Seam] Edge-face add failed: {e}")

    # Method 0b: Fill - requires single boundary, might work after edge-face
    if not success:
        try:
            bpy.ops.mesh.fill()
            report.append("[Seam] Fill successful!")
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
            success = True
        except RuntimeError as e:
            report.append(f"[Seam] Fill failed: {e}")

    # Re-select seam edges for other methods if fill_holes didn't work
    if not success:
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        bm = bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bm.edges.ensure_lookup_table()
        for e in bm.edges:
            if e.is_boundary:
                edge_center = (mesh_obj.matrix_world @ e.verts[0].co + mesh_obj.matrix_world @ e.verts[1].co) / 2
                dist = (edge_center - bone_pos).length
                if dist <= seam_threshold:
                    e.select = True
        bm.to_mesh(mesh_obj.data)
        bm.free()
        bpy.ops.object.mode_set(mode='EDIT')

    # Method 1: Bridge with cuts (helps with mismatched vertex counts)
    if not success:
        try:
            bpy.ops.mesh.bridge_edge_loops(
                type='SINGLE',
                number_cuts=2,  # Add intermediate cuts to help with density mismatch
                interpolation='LINEAR',
                smoothness=0.0
            )
            report.append("[Seam] Bridge edge loops with cuts successful!")
            success = True
        except RuntimeError as e:
            report.append(f"[Seam] Bridge with cuts failed: {e}")

    # Method 2: Grid Fill (handles non-matching edge loop counts)
    if not success:
        try:
            bpy.ops.mesh.fill_grid(span=1, offset=0, use_interp_simple=False)
            report.append("[Seam] Grid fill successful!")
            success = True
        except RuntimeError as e2:
            report.append(f"[Seam] Grid fill failed: {e2}")

    # Method 3: Simple bridge without cuts
    if not success:
        try:
            bpy.ops.mesh.bridge_edge_loops(
                type='SINGLE',
                number_cuts=0,
                interpolation='LINEAR',
                smoothness=0.0
            )
            report.append("[Seam] Simple bridge successful!")
            success = True
        except RuntimeError as e3:
            report.append(f"[Seam] Simple bridge failed: {e3}")

    # Method 4: Fill (creates n-gon)
    if not success:
        try:
            bpy.ops.mesh.fill()
            report.append("[Seam] Fill successful (may have created n-gon).")
            success = True
        except RuntimeError as e4:
            report.append(f"[Seam] Fill failed: {e4}")

    # Method 5: Select boundary and try edge_face_add
    if not success:
        try:
            bpy.ops.mesh.edge_face_add()
            report.append("[Seam] Edge face add successful!")
            success = True
        except RuntimeError as e5:
            report.append(f"[Seam] Edge face add failed: {e5}")

    # Method 6: Manual bridge using bmesh (handles mismatched vertex counts)
    if not success:
        bpy.ops.object.mode_set(mode='OBJECT')
        report.append("[Seam] Trying manual bmesh bridge...")
        success = manual_bridge_boundaries(mesh_obj, bone_pos, seam_threshold, report)

    if not success:
        report.append("[Seam] All automatic methods failed - manual bridging may be needed.")

    ensure_object_mode()
    depsgraph_update()


# ============================================================================
# WEIGHT TRANSFER
# ============================================================================

def transfer_weights_via_modifier(source_mesh, target_mesh, report: list):
    """Transfer vertex weights from source to target using Data Transfer modifier."""
    ensure_object_mode()
    depsgraph_update()

    bpy.ops.object.select_all(action='DESELECT')
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh

    mod = target_mesh.modifiers.new(name="DT_Weights", type='DATA_TRANSFER')
    mod.object = source_mesh
    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}

    # Set vertex mapping
    available_map = [e.identifier for e in mod.bl_rna.properties['vert_mapping'].enum_items]
    preferred = ["POLYINTERP_NEAREST", "NEAREST"]
    mod.vert_mapping = next((v for v in preferred if v in available_map), available_map[0])

    mod.mix_mode = 'REPLACE'
    mod.mix_factor = 1.0

    report.append(f"[Weights] vert_mapping={mod.vert_mapping}")

    bpy.ops.object.modifier_apply(modifier=mod.name)
    report.append("[Weights] Transfer complete.")


# ============================================================================
# HEAD SEPARATION
# ============================================================================

def separate_head(body_mesh, arm_obj, report: list):
    """
    Separate head from body at neck_01 bone position.
    Returns the new head mesh object.
    """
    neck_bone = arm_obj.data.bones.get("neck_01")
    if not neck_bone:
        raise RuntimeError("Bone 'neck_01' not found in armature.")

    cut_z = (arm_obj.matrix_world @ neck_bone.head_local).z
    report.append(f"[Head] Cut Z position: {cut_z:.4f}")

    # Select vertices above cut point
    ensure_object_mode()
    mesh = body_mesh.data

    # Deselect all first
    for v in mesh.vertices:
        v.select = False

    # Select vertices above neck
    selected_count = 0
    for v in mesh.vertices:
        world_co = body_mesh.matrix_world @ v.co
        if world_co.z > cut_z:
            v.select = True
            selected_count += 1

    report.append(f"[Head] Selected {selected_count} vertices above neck.")

    if selected_count == 0:
        report.append("[Head] WARNING: No vertices selected for head separation.")
        return None

    # Separate in edit mode
    ensure_edit_mode(body_mesh)
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Find the new head mesh (it will be the newly created object)
    # After separate, both objects will be selected
    head_mesh = None
    for obj in bpy.context.selected_objects:
        if obj != body_mesh and obj.type == 'MESH':
            head_mesh = obj
            break

    if head_mesh:
        head_mesh.name = "Head"
        report.append(f"[Head] Created head mesh: '{head_mesh.name}'")
    else:
        report.append("[Head] WARNING: Could not find separated head mesh.")

    depsgraph_update()
    return head_mesh


# ============================================================================
# MAIN OPERATIONS
# ============================================================================

def attach_hand(side: str, body_mesh, hand_mesh, arm_obj, ref_mesh, report: list):
    """
    Complete hand attachment operation.
    Uses geometry analysis to fit new hand to old hand location.
    side: 'L' or 'R'
    """
    bone_name = "hand_l" if side.upper() == 'L' else "hand_r"
    bone_names = HAND_BONES_L if side.upper() == 'L' else HAND_BONES_R

    report.append(f"\n=== ATTACHING HAND ({side}) ===")

    # Step 1: Analyze OLD hand geometry BEFORE cutting
    report.append("\n[Step 1] Analyzing old hand geometry...")
    old_hand_info = analyze_hand_geometry(body_mesh, bone_names, WEIGHT_THRESHOLD)
    if old_hand_info:
        report.append(f"[Analysis] Old hand: {old_hand_info['vertex_count']} verts")
        report.append(f"[Analysis] Old hand size: {old_hand_info['size']}")
        report.append(f"[Analysis] Old wrist center: {old_hand_info['wrist_center']}")
    else:
        report.append("[Analysis] WARNING: Could not analyze old hand - no weighted vertices found")

    # Step 2: Analyze NEW hand geometry
    report.append("\n[Step 2] Analyzing new hand geometry...")
    new_hand_info = analyze_mesh_geometry(hand_mesh)
    if new_hand_info:
        report.append(f"[Analysis] New hand: {new_hand_info['vertex_count']} verts")
        report.append(f"[Analysis] New hand size: {new_hand_info['size']}")
    else:
        report.append("[Analysis] WARNING: Could not analyze new hand")

    # Step 3: Cut existing hand from body
    report.append("\n[Step 3] Cutting existing hand from body...")
    cut_hand_from_body(body_mesh, arm_obj, side, report)

    # Step 4: Fit new hand to arm boundary (where we just cut)
    report.append("\n[Step 4] Fitting new hand to arm boundary...")
    fit_new_hand_to_old(hand_mesh, old_hand_info, new_hand_info, arm_obj, bone_name, side, report, body_mesh)

    # Step 5: Create vertex group to track hand vertices after joining
    hand_group_name = f"Attached_Hand_{side.upper()}"
    report.append(f"\n[Step 5] Creating vertex group '{hand_group_name}'...")
    create_vertex_group_for_all_verts(hand_mesh, hand_group_name, report)

    # Step 6: Shrinkwrap hand wrist to body surface to close gap
    if USE_SHRINKWRAP:
        report.append("\n[Step 6] Applying shrinkwrap to close gap...")
        shrinkwrap_wrist_to_body(hand_mesh, body_mesh, report)
    else:
        report.append("\n[Step 6] Skipping shrinkwrap (USE_SHRINKWRAP=False)")

    # Step 7: Join meshes
    report.append("\n[Step 7] Joining hand mesh to body...")
    join_meshes(body_mesh, hand_mesh, report)
    report.append(f"[Step 7] Hand vertices are now in group '{hand_group_name}' for easy selection.")

    # Step 8: Bridge the seam (create faces between arm and hand)
    report.append("\n[Step 8] Bridging seam...")
    wrist_pos = bone_head_world(arm_obj, bone_name)
    close_seam(body_mesh, wrist_pos, report)

    # Step 9: Create vertex groups and transfer weights
    report.append("\n[Step 9] Transferring weights...")
    create_vertex_groups_from_armature(body_mesh, arm_obj, report)
    transfer_weights_via_modifier(ref_mesh, body_mesh, report)

    report.append(f"\n=== HAND ({side}) ATTACHMENT COMPLETE ===")


def attach_foot(side: str, body_mesh, foot_mesh, arm_obj, ref_mesh, report: list):
    """
    Complete foot attachment operation.
    Uses geometry analysis to fit new foot to old foot location.
    side: 'L' or 'R'
    """
    bone_name = "foot_l" if side.upper() == 'L' else "foot_r"
    bone_names = FOOT_BONES_L if side.upper() == 'L' else FOOT_BONES_R

    report.append(f"\n=== ATTACHING FOOT ({side}) ===")

    # Step 1: Analyze OLD foot geometry BEFORE cutting
    report.append("\n[Step 1] Analyzing old foot geometry...")
    old_foot_info = analyze_hand_geometry(body_mesh, bone_names, WEIGHT_THRESHOLD)  # Works for feet too
    if old_foot_info:
        report.append(f"[Analysis] Old foot: {old_foot_info['vertex_count']} verts")
        report.append(f"[Analysis] Old foot size: {old_foot_info['size']}")
        report.append(f"[Analysis] Old ankle center: {old_foot_info['wrist_center']}")
    else:
        report.append("[Analysis] WARNING: Could not analyze old foot")

    # Step 2: Analyze NEW foot geometry
    report.append("\n[Step 2] Analyzing new foot geometry...")
    new_foot_info = analyze_mesh_geometry(foot_mesh)
    if new_foot_info:
        report.append(f"[Analysis] New foot: {new_foot_info['vertex_count']} verts")
        report.append(f"[Analysis] New foot size: {new_foot_info['size']}")

    # Step 3: Cut existing foot from body
    report.append("\n[Step 3] Cutting existing foot from body...")
    cut_foot_from_body(body_mesh, arm_obj, side, report)

    # Step 4: Fit new foot to leg boundary (where we just cut)
    report.append("\n[Step 4] Fitting new foot to leg boundary...")
    fit_new_hand_to_old(foot_mesh, old_foot_info, new_foot_info, arm_obj, bone_name, side, report, body_mesh)

    # Step 5: Create vertex group to track foot vertices after joining
    foot_group_name = f"Attached_Foot_{side.upper()}"
    report.append(f"\n[Step 5] Creating vertex group '{foot_group_name}'...")
    create_vertex_group_for_all_verts(foot_mesh, foot_group_name, report)

    # Step 6: Shrinkwrap foot ankle to body surface to close gap
    if USE_SHRINKWRAP:
        report.append("\n[Step 6] Applying shrinkwrap to close gap...")
        shrinkwrap_wrist_to_body(foot_mesh, body_mesh, report)
    else:
        report.append("\n[Step 6] Skipping shrinkwrap (USE_SHRINKWRAP=False)")

    # Step 7: Join meshes
    report.append("\n[Step 7] Joining foot mesh to body...")
    join_meshes(body_mesh, foot_mesh, report)
    report.append(f"[Step 7] Foot vertices are now in group '{foot_group_name}' for easy selection.")

    # Step 8: Bridge the seam (create faces between leg and foot)
    report.append("\n[Step 8] Bridging seam...")
    ankle_pos = bone_head_world(arm_obj, bone_name)
    close_seam(body_mesh, ankle_pos, report)

    # Step 9: Transfer weights
    report.append("\n[Step 9] Transferring weights...")
    create_vertex_groups_from_armature(body_mesh, arm_obj, report)
    transfer_weights_via_modifier(ref_mesh, body_mesh, report)

    report.append(f"\n=== FOOT ({side}) ATTACHMENT COMPLETE ===")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(mode: str = MODE_ATTACH_HANDS):
    """
    Main entry point.
    mode: 'hands', 'feet', 'head', or 'all'
    """
    report = []
    report.append("ModularBody V1")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Mode: {mode}\n")

    # Find collections
    body_col = find_collection_ci(BODY_COLLECTION)
    hands_col = find_collection_ci(HANDS_COLLECTION)
    feet_col = find_collection_ci(FEET_COLLECTION)
    ref_col = find_collection_ci(UEFN_REFERENCE)

    if not body_col:
        raise RuntimeError(f"Collection '{BODY_COLLECTION}' not found.")
    if not ref_col:
        raise RuntimeError(f"Collection '{UEFN_REFERENCE}' not found.")

    # Find armature and body mesh
    arm_obj = find_single_armature(body_col)
    body_meshes = mesh_objects(body_col)
    if not body_meshes:
        raise RuntimeError(f"No mesh found in '{BODY_COLLECTION}' collection.")
    body_mesh = body_meshes[0]  # Use first/largest mesh

    # Find reference mesh
    ref_meshes = mesh_objects(ref_col)
    if not ref_meshes:
        raise RuntimeError(f"No mesh found in '{UEFN_REFERENCE}' collection.")
    ref_mesh = ref_meshes[0]

    report.append(f"Body Armature: {arm_obj.name}")
    report.append(f"Body Mesh: {body_mesh.name}")
    report.append(f"Reference Mesh: {ref_mesh.name}")

    # Execute operations based on mode
    if mode in [MODE_ATTACH_HANDS, MODE_ALL]:
        if not hands_col:
            report.append(f"\nWARNING: Collection '{HANDS_COLLECTION}' not found - skipping hands.")
        else:
            hand_L = find_mesh_by_name_pattern(hands_col, "*hand*l*")
            hand_R = find_mesh_by_name_pattern(hands_col, "*hand*r*")

            if not hand_L:
                hand_L = find_mesh_by_name_pattern(hands_col, "*left*")
            if not hand_R:
                hand_R = find_mesh_by_name_pattern(hands_col, "*right*")

            # Fallback: use first two meshes
            hand_meshes = mesh_objects(hands_col)
            if not hand_L and len(hand_meshes) >= 1:
                hand_L = hand_meshes[0]
            if not hand_R and len(hand_meshes) >= 2:
                hand_R = hand_meshes[1]

            if hand_L:
                report.append(f"\nFound left hand: {hand_L.name}")
                attach_hand('L', body_mesh, hand_L, arm_obj, ref_mesh, report)
            else:
                report.append("\nWARNING: No left hand mesh found.")

            if hand_R:
                report.append(f"\nFound right hand: {hand_R.name}")
                attach_hand('R', body_mesh, hand_R, arm_obj, ref_mesh, report)
            else:
                report.append("\nWARNING: No right hand mesh found.")

    if mode in [MODE_ATTACH_FEET, MODE_ALL]:
        if not feet_col:
            report.append(f"\nWARNING: Collection '{FEET_COLLECTION}' not found - skipping feet.")
        else:
            foot_L = find_mesh_by_name_pattern(feet_col, "*foot*l*")
            foot_R = find_mesh_by_name_pattern(feet_col, "*foot*r*")

            if not foot_L:
                foot_L = find_mesh_by_name_pattern(feet_col, "*left*")
            if not foot_R:
                foot_R = find_mesh_by_name_pattern(feet_col, "*right*")

            foot_meshes = mesh_objects(feet_col)
            if not foot_L and len(foot_meshes) >= 1:
                foot_L = foot_meshes[0]
            if not foot_R and len(foot_meshes) >= 2:
                foot_R = foot_meshes[1]

            if foot_L:
                report.append(f"\nFound left foot: {foot_L.name}")
                attach_foot('L', body_mesh, foot_L, arm_obj, ref_mesh, report)
            else:
                report.append("\nWARNING: No left foot mesh found.")

            if foot_R:
                report.append(f"\nFound right foot: {foot_R.name}")
                attach_foot('R', body_mesh, foot_R, arm_obj, ref_mesh, report)
            else:
                report.append("\nWARNING: No right foot mesh found.")

    if mode in [MODE_SEPARATE_HEAD, MODE_ALL]:
        report.append("\n=== SEPARATING HEAD ===")
        head_mesh = separate_head(body_mesh, arm_obj, report)
        if head_mesh:
            # Transfer weights to head mesh too
            create_vertex_groups_from_armature(head_mesh, arm_obj, report)
            transfer_weights_via_modifier(ref_mesh, head_mesh, report)
        report.append("=== HEAD SEPARATION COMPLETE ===")

    report.append("\n" + "=" * 50)
    report.append("ModularBody V1 complete.")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log: {LOG_TEXT_NAME} ---")


# Run with default mode (hands only for testing)
# Change mode parameter to run different operations:
#   main(MODE_ATTACH_HANDS)  - Attach hands only
#   main(MODE_ATTACH_FEET)   - Attach feet only
#   main(MODE_SEPARATE_HEAD) - Separate head only
#   main(MODE_ALL)           - All operations

if __name__ == "__main__":
    main(MODE_ATTACH_HANDS)
