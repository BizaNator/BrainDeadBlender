"""
Segmentation V1 - Body Segment Pipeline for UEFN Characters

Creates body segments from bone vertex groups for:
- Material Slots (per-segment, = PolyGroups/Sections in Unreal)
- UV Islands (unwrap per segment with seams at boundaries)
- Vertex Color display (materials set up to show vertex colors)
- Separate Objects (optional mesh splitting)

REQUIRES: Run after Pipeline_v31 (mesh must have UEFN bone vertex groups)

WORKFLOW ORDER:
    1. Run Pipeline_v31.py - sets up skeleton/weights
    2. Run Segmentation_v1.py (MODE_ALL) - creates materials & UVs
    3. Run VertexColors_v1.py - projects colors to vertices
    4. View in Material Preview mode (Z > Material Preview)

CONFIGURATION:
    Edit RUN_MODE at top of script to change operation mode.
    No need to modify function calls at bottom.

MODES:
    MODE_ALL        - Materials + UVs (default)
    MODE_MATERIALS  - Just materials
    MODE_UV_ISLANDS - Just UVs
    MODE_UPDATE_MATS - Update materials for vertex colors (run after VertexColors)
    MODE_SEPARATE   - Split mesh into objects (destructive)
"""

import bpy
import bmesh
import fnmatch
from datetime import datetime
from mathutils import Vector

LOG_TEXT_NAME = "Segmentation_V1_Log.txt"

# ============================================================================
# OPERATION MODES
# ============================================================================

MODE_MATERIALS = "materials"      # Creates materials (= Unreal PolyGroups/Sections)
MODE_UV_ISLANDS = "uv_islands"    # Creates UVs with seams at segment boundaries
MODE_SEAMS_ONLY = "seams_only"    # Just mark seams, no UV unwrap
MODE_SEPARATE = "separate"        # Split into separate objects (destructive)
MODE_UPDATE_MATS = "update_mats"  # Update materials to show vertex colors (run after VertexColors)
MODE_ALL = "all"                  # Materials + UVs

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

# --- MODE SELECTION ---
# MODE_ALL        = Materials + UVs (default for initial setup)
# MODE_MATERIALS  = Materials only
# MODE_UV_ISLANDS = UVs only
# MODE_SEAMS_ONLY = Just mark seams, no unwrap
# MODE_UPDATE_MATS = Update materials to show vertex colors (run after VertexColors)
# MODE_SEPARATE   = Split into separate objects (destructive!)
RUN_MODE = MODE_ALL

# --- TARGET ---
# Collection to operate on (output from Pipeline_v31)
TARGET_COLLECTION = "Export"

# Segment definitions: segment_name -> list of bone names (supports wildcards)
SEGMENT_DEFINITIONS = {
    "Head": [
        "head", "neck_01", "neck_02"
    ],
    "Torso": [
        "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05"
    ],
    "UpperArm_L": [
        "clavicle_l", "upperarm_l", "upperarm_twist_01_l"
    ],
    "UpperArm_R": [
        "clavicle_r", "upperarm_r", "upperarm_twist_01_r"
    ],
    "LowerArm_L": [
        "lowerarm_l", "lowerarm_twist_01_l"
    ],
    "LowerArm_R": [
        "lowerarm_r", "lowerarm_twist_01_r"
    ],
    "Hand_L": [
        "hand_l",
        "thumb_01_l", "thumb_02_l", "thumb_03_l",
        "index_metacarpal_l", "index_01_l", "index_02_l", "index_03_l",
        "middle_metacarpal_l", "middle_01_l", "middle_02_l", "middle_03_l",
        "ring_metacarpal_l", "ring_01_l", "ring_02_l", "ring_03_l",
        "pinky_metacarpal_l", "pinky_01_l", "pinky_02_l", "pinky_03_l",
    ],
    "Hand_R": [
        "hand_r",
        "thumb_01_r", "thumb_02_r", "thumb_03_r",
        "index_metacarpal_r", "index_01_r", "index_02_r", "index_03_r",
        "middle_metacarpal_r", "middle_01_r", "middle_02_r", "middle_03_r",
        "ring_metacarpal_r", "ring_01_r", "ring_02_r", "ring_03_r",
        "pinky_metacarpal_r", "pinky_01_r", "pinky_02_r", "pinky_03_r",
    ],
    "UpperLeg_L": [
        "thigh_l", "thigh_twist_01_l"
    ],
    "UpperLeg_R": [
        "thigh_r", "thigh_twist_01_r"
    ],
    "LowerLeg_L": [
        "calf_l", "calf_twist_01_l"
    ],
    "LowerLeg_R": [
        "calf_r", "calf_twist_01_r"
    ],
    "Foot_L": [
        "foot_l", "ball_l"
    ],
    "Foot_R": [
        "foot_r", "ball_r"
    ],
}

# Segment colors for materials (RGB 0-1)
SEGMENT_COLORS = {
    "Head": (0.9, 0.8, 0.7),      # Skin tone
    "Torso": (0.2, 0.4, 0.8),     # Blue
    "UpperArm_L": (0.8, 0.3, 0.3),  # Red
    "UpperArm_R": (0.8, 0.3, 0.3),
    "LowerArm_L": (0.9, 0.5, 0.3),  # Orange
    "LowerArm_R": (0.9, 0.5, 0.3),
    "Hand_L": (0.9, 0.8, 0.7),    # Skin tone
    "Hand_R": (0.9, 0.8, 0.7),
    "UpperLeg_L": (0.3, 0.6, 0.3),  # Green
    "UpperLeg_R": (0.3, 0.6, 0.3),
    "LowerLeg_L": (0.4, 0.7, 0.4),  # Light green
    "LowerLeg_R": (0.4, 0.7, 0.4),
    "Foot_L": (0.5, 0.3, 0.2),    # Brown
    "Foot_R": (0.5, 0.3, 0.2),
}

# UV settings
UV_ISLAND_MARGIN = 0.02  # Margin between UV islands
REMOVE_OLD_UVS = True    # Remove existing UV layers (keeps only the new segmented UV)


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
# COLLECTION / OBJECT HELPERS
# ============================================================================

def find_collection_ci(name: str):
    """Find collection by name (case-insensitive)."""
    want = name.strip().lower()
    for col in bpy.data.collections:
        if col.name.strip().lower() == want:
            return col
    return None


def mesh_objects(col):
    """Get all mesh objects in collection."""
    return [o for o in col.all_objects if o.type == "MESH"]


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


def get_matching_vertex_groups(mesh_obj, patterns: list) -> list:
    """
    Get vertex group names matching any of the patterns.
    Supports wildcards via fnmatch.
    """
    matching = []
    for vg in mesh_obj.vertex_groups:
        for pattern in patterns:
            if fnmatch.fnmatch(vg.name.lower(), pattern.lower()):
                matching.append(vg.name)
                break
    return matching


def get_segment_weight(mesh_obj, vertex_index: int, bone_names: list) -> float:
    """Get combined weight of vertex for a segment (list of bones)."""
    total = 0.0
    for bone in bone_names:
        total += get_vertex_weight(mesh_obj, vertex_index, bone)
    return total


# ============================================================================
# SEGMENT ASSIGNMENT
# ============================================================================

def build_segment_bone_map(mesh_obj, report: list) -> dict:
    """
    Build mapping of segment names to actual vertex group names.
    Expands wildcard patterns to matching vertex groups.
    """
    segment_bones = {}

    for segment_name, patterns in SEGMENT_DEFINITIONS.items():
        matching = []
        for pattern in patterns:
            # Check for exact match first
            if mesh_obj.vertex_groups.get(pattern):
                matching.append(pattern)
            else:
                # Try wildcard matching
                for vg in mesh_obj.vertex_groups:
                    if fnmatch.fnmatch(vg.name.lower(), pattern.lower()):
                        if vg.name not in matching:
                            matching.append(vg.name)

        segment_bones[segment_name] = matching
        if matching:
            report.append(f"[Segments] {segment_name}: {len(matching)} bones")
        else:
            report.append(f"[Segments] WARNING: {segment_name} has no matching bones!")

    return segment_bones


def assign_faces_to_segments(mesh_obj, segment_bones: dict, report: list) -> dict:
    """
    Assign each face to a segment based on dominant bone weights.

    Returns dict: segment_name -> list of face indices
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    # Initialize assignments
    assignments = {seg: [] for seg in segment_bones.keys()}
    assignments["Unassigned"] = []

    # For each face, determine dominant segment
    for face in mesh.polygons:
        segment_weights = {}

        # Sum weights for each segment across all face vertices
        for segment_name, bones in segment_bones.items():
            total_weight = 0.0
            for vert_idx in face.vertices:
                total_weight += get_segment_weight(mesh_obj, vert_idx, bones)
            segment_weights[segment_name] = total_weight

        # Find segment with highest weight
        if segment_weights:
            max_segment = max(segment_weights, key=segment_weights.get)
            max_weight = segment_weights[max_segment]

            if max_weight > 0:
                assignments[max_segment].append(face.index)
            else:
                assignments["Unassigned"].append(face.index)
        else:
            assignments["Unassigned"].append(face.index)

    # Report statistics
    total_faces = len(mesh.polygons)
    for segment_name, faces in assignments.items():
        if faces:
            pct = len(faces) / total_faces * 100
            report.append(f"[Assignment] {segment_name}: {len(faces)} faces ({pct:.1f}%)")

    return assignments


# ============================================================================
# SEAM MARKING (for UV boundaries between segments)
# ============================================================================

def mark_segment_seams(mesh_obj, assignments: dict, report: list):
    """
    Mark UV seams at segment boundaries.
    An edge is a seam if its two adjacent faces belong to different segments.
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append("\n[Seams] Marking seams at segment boundaries...")

    # Build face -> segment lookup
    face_to_segment = {}
    for segment_name, face_indices in assignments.items():
        for face_idx in face_indices:
            face_to_segment[face_idx] = segment_name

    # Use bmesh to find and mark boundary edges
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    seam_count = 0

    for edge in bm.edges:
        # Get adjacent faces
        linked_faces = edge.link_faces

        if len(linked_faces) == 2:
            face1_idx = linked_faces[0].index
            face2_idx = linked_faces[1].index

            seg1 = face_to_segment.get(face1_idx, "Unknown")
            seg2 = face_to_segment.get(face2_idx, "Unknown")

            # If faces belong to different segments, mark as seam
            if seg1 != seg2:
                edge.seam = True
                seam_count += 1
        elif len(linked_faces) == 1:
            # Boundary edge - mark as seam
            edge.seam = True
            seam_count += 1

    bm.to_mesh(mesh)
    bm.free()

    depsgraph_update()
    report.append(f"[Seams] Marked {seam_count} edges as seams.")


# ============================================================================
# MATERIAL SLOTS
# ============================================================================

def create_segment_materials(mesh_obj, assignments: dict, report: list):
    """
    Create materials for each segment with distinct colors.
    Assigns faces to appropriate material slots.
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append("\n[Materials] Creating segment materials...")

    # Clear existing materials
    mesh_obj.data.materials.clear()

    # Create materials for each segment with faces
    segment_mat_index = {}

    for segment_name, face_indices in assignments.items():
        if not face_indices:
            continue

        # Get color for this segment
        color = SEGMENT_COLORS.get(segment_name, (0.5, 0.5, 0.5))

        # Create material
        mat_name = f"M_{segment_name}"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)

        # Set viewport color (fallback when no vertex colors)
        mat.diffuse_color = (*color, 1.0)
        mat.use_nodes = True

        # Set up node setup with Vertex Color support
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Create vertex color node - try Color Attribute (Blender 4.0+) first,
        # then Vertex Color (older versions)
        vert_color = None
        try:
            # Blender 4.0+ renamed ShaderNodeVertexColor to ShaderNodeColorAttribute
            vert_color = nodes.new('ShaderNodeColorAttribute')
            vert_color.layer_name = "Col"
        except:
            pass

        if vert_color is None:
            # Fallback to ShaderNodeVertexColor (works in all Blender versions)
            try:
                vert_color = nodes.new('ShaderNodeVertexColor')
                vert_color.layer_name = "Col"
            except:
                pass

        if vert_color is None:
            # Last resort: use Attribute node
            vert_color = nodes.new('ShaderNodeAttribute')
            vert_color.attribute_name = "Col"
            vert_color.attribute_type = 'GEOMETRY'

        vert_color.location = (-300, 0)

        # Add Principled BSDF
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)  # Fallback color
        bsdf.location = (0, 0)

        # Add output
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)

        # Connect Vertex Color -> Base Color (overrides default)
        # Output name varies by node type - try 'Color' first, then others
        color_output = None
        for output_name in ['Color', 'color', 0]:  # Try by name, then by index
            try:
                color_output = vert_color.outputs[output_name]
                break
            except:
                pass

        if color_output:
            links.new(color_output, bsdf.inputs['Base Color'])

        # Connect BSDF -> Output
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        # Add to mesh
        mesh_obj.data.materials.append(mat)
        segment_mat_index[segment_name] = len(mesh_obj.data.materials) - 1

        report.append(f"[Materials] Created '{mat_name}' (color: {color})")

    # Assign faces to materials
    for segment_name, face_indices in assignments.items():
        if segment_name not in segment_mat_index:
            continue

        mat_idx = segment_mat_index[segment_name]
        for face_idx in face_indices:
            mesh.polygons[face_idx].material_index = mat_idx

    depsgraph_update()
    report.append("[Materials] Done.")
    report.append("[Materials] Materials use Vertex Color node - run VertexColors to see colors in viewport.")


# ============================================================================
# UPDATE MATERIALS (for refreshing after vertex colors are applied)
# ============================================================================

def update_materials_for_vertex_colors(mesh_obj, report: list):
    """
    Update existing materials to properly display vertex colors.
    Call this AFTER running VertexColors script.
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append("\n[Materials] Updating materials for vertex color display...")

    for mat in mesh.materials:
        if not mat or not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find existing vertex color/attribute node
        vert_color = None
        for node in nodes:
            if node.type in ['VERTEX_COLOR', 'ATTRIBUTE']:
                vert_color = node
                break

        if not vert_color:
            # Create one
            try:
                vert_color = nodes.new('ShaderNodeColorAttribute')
                vert_color.layer_name = "Col"
            except:
                try:
                    vert_color = nodes.new('ShaderNodeVertexColor')
                    vert_color.layer_name = "Col"
                except:
                    vert_color = nodes.new('ShaderNodeAttribute')
                    vert_color.attribute_name = "Col"
                    vert_color.attribute_type = 'GEOMETRY'

            vert_color.location = (-300, 0)

        # Find BSDF
        bsdf = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
                break

        if bsdf and vert_color:
            # Connect vertex color to base color
            color_output = None
            for output_name in ['Color', 'color', 0]:
                try:
                    color_output = vert_color.outputs[output_name]
                    break
                except:
                    pass

            if color_output:
                # Clear existing base color connections
                for link in list(links):
                    if link.to_socket == bsdf.inputs['Base Color']:
                        links.remove(link)
                links.new(color_output, bsdf.inputs['Base Color'])
                report.append(f"[Materials] Updated '{mat.name}' for vertex colors")

    depsgraph_update()
    report.append("[Materials] Done updating materials.")
    report.append("[Materials] View in 'Material Preview' mode (Z key > Material Preview)")


# ============================================================================
# UV UNWRAPPING
# ============================================================================

def unwrap_by_segments(mesh_obj, assignments: dict, report: list):
    """
    UV unwrap using segment boundaries as seams.
    This keeps each segment as a cohesive UV island.

    Steps:
    1. Mark seams at segment boundaries
    2. Unwrap entire mesh (respects seams)
    3. Pack islands with margin
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append("\n[UV] Unwrapping with segment seams...")

    # Step 1: Mark seams at segment boundaries
    mark_segment_seams(mesh_obj, assignments, report)

    # Step 2: Handle existing UV layers
    uv_name = "UVMap"  # Use standard name for compatibility

    if REMOVE_OLD_UVS:
        # Remove ALL existing UV layers
        old_uv_names = [uv.name for uv in mesh.uv_layers]
        for name in old_uv_names:
            mesh.uv_layers.remove(mesh.uv_layers[name])
        report.append(f"[UV] Removed {len(old_uv_names)} old UV layer(s): {old_uv_names}")
    else:
        # Just remove our target if it exists
        if uv_name in mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers[uv_name])

    # Create fresh UV layer and set active
    uv_layer = mesh.uv_layers.new(name=uv_name)
    mesh.uv_layers.active = uv_layer
    report.append(f"[UV] Created UV layer: {uv_name}")

    # Step 2: Select all and unwrap (uses seams)
    ensure_edit_mode(mesh_obj)
    bpy.ops.mesh.select_all(action='SELECT')

    report.append("[UV] Unwrapping entire mesh (using seams)...")

    try:
        # Standard unwrap respects seams
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
        report.append("[UV] Unwrap complete.")
    except RuntimeError as e:
        report.append(f"[UV] WARNING: Unwrap failed: {e}, trying smart project...")
        try:
            bpy.ops.uv.smart_project(
                angle_limit=66.0,
                island_margin=0.0,
                area_weight=0.0,
                correct_aspect=True,
                scale_to_bounds=False
            )
        except RuntimeError as e2:
            report.append(f"[UV] WARNING: Smart project also failed: {e2}")

    # Step 3: Average island scale for consistency
    report.append("[UV] Averaging island scale...")
    try:
        bpy.ops.uv.average_islands_scale()
    except RuntimeError:
        pass  # Not critical

    # Step 4: Pack islands with margin
    report.append("[UV] Packing islands...")
    try:
        bpy.ops.uv.pack_islands(margin=UV_ISLAND_MARGIN)
        report.append("[UV] Islands packed successfully.")
    except RuntimeError as e:
        report.append(f"[UV] WARNING: Pack islands failed: {e}")

    ensure_object_mode()
    depsgraph_update()
    report.append(f"[UV] Done. Single UV layer: {uv_name}")
    report.append("[UV] Each segment is now a cohesive UV island.")


# ============================================================================
# SEPARATE OBJECTS
# ============================================================================

def separate_by_segments(mesh_obj, assignments: dict, arm_obj, report: list):
    """
    Separate mesh into distinct objects per segment.
    Uses materials to separate (more reliable than face indices).

    WARNING: This is destructive - the original mesh is split.

    NOTE: For Unreal, you usually DON'T need this!
    Materials already create "sections" in Unreal which serve the same purpose.
    Only use this if you need physically separate objects in Blender.
    """
    ensure_object_mode()
    original_name = mesh_obj.name

    report.append("\n[Separate] Splitting mesh by segments...")
    report.append("[Separate] NOTE: Materials already = Unreal sections. This creates separate Blender objects.")

    # First, ensure materials are assigned (we'll separate by material)
    if len(mesh_obj.data.materials) < 2:
        report.append("[Separate] Assigning materials first...")
        create_segment_materials(mesh_obj, assignments, report)

    # Count materials before
    num_materials = len(mesh_obj.data.materials)
    report.append(f"[Separate] Mesh has {num_materials} materials")

    if num_materials < 2:
        report.append("[Separate] WARNING: Only 1 material, nothing to separate.")
        return []

    # Enter edit mode and select all
    ensure_edit_mode(mesh_obj)
    bpy.ops.mesh.select_all(action='SELECT')

    # Separate by material - this is the reliable way
    report.append("[Separate] Separating by material...")
    try:
        bpy.ops.mesh.separate(type='MATERIAL')
    except RuntimeError as e:
        report.append(f"[Separate] WARNING: Separate failed: {e}")
        ensure_object_mode()
        return []

    ensure_object_mode()

    # Find all the new objects
    created_objects = []
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            # Get the material name to use as suffix
            if obj.data.materials and len(obj.data.materials) > 0:
                mat = obj.data.materials[0]
                if mat and mat.name.startswith("M_"):
                    segment_name = mat.name[2:]  # Remove "M_" prefix
                    obj.name = f"{original_name}_{segment_name}"
                    obj.data.name = obj.name

            # Ensure armature modifier exists
            has_armature = any(m.type == 'ARMATURE' for m in obj.modifiers)
            if not has_armature and arm_obj:
                mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                mod.object = arm_obj
                mod.use_vertex_groups = True
                mod.use_bone_envelopes = False

            created_objects.append(obj)
            report.append(f"[Separate] Created '{obj.name}'")

    depsgraph_update()
    report.append(f"[Separate] Created {len(created_objects)} separate objects.")
    report.append("[Separate] TIP: For Unreal, you usually just need materials (not separate objects).")
    return created_objects


# ============================================================================
# MAIN
# ============================================================================

def main(mode: str = MODE_ALL, collection_name: str = None):
    """
    Main entry point.

    Args:
        mode: Operation mode (MODE_ALL, MODE_FACE_MAPS, MODE_MATERIALS,
              MODE_UV_ISLANDS, MODE_SEPARATE)
        collection_name: Override target collection name
    """
    report = []
    report.append("Segmentation V1")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Mode: {mode}\n")

    # Find collection
    col_name = collection_name or TARGET_COLLECTION
    col = find_collection_ci(col_name)
    if not col:
        raise RuntimeError(f"Collection '{col_name}' not found.")

    # Find mesh
    meshes = mesh_objects(col)
    if not meshes:
        raise RuntimeError(f"No mesh objects found in '{col_name}' collection.")

    mesh_obj = meshes[0]
    report.append(f"Target Mesh: {mesh_obj.name}")
    report.append(f"Vertices: {len(mesh_obj.data.vertices)}")
    report.append(f"Faces: {len(mesh_obj.data.polygons)}")
    report.append(f"Vertex Groups: {len(mesh_obj.vertex_groups)}")

    # Find armature (for separate mode)
    arm_obj = None
    for obj in col.all_objects:
        if obj.type == 'ARMATURE':
            arm_obj = obj
            break

    if arm_obj:
        report.append(f"Armature: {arm_obj.name}")

    # Build segment to bone mapping
    report.append("\n[Step 1] Building segment definitions...")
    segment_bones = build_segment_bone_map(mesh_obj, report)

    # Assign faces to segments
    report.append("\n[Step 2] Assigning faces to segments...")
    assignments = assign_faces_to_segments(mesh_obj, segment_bones, report)

    # Execute requested operations
    if mode in [MODE_MATERIALS, MODE_ALL]:
        create_segment_materials(mesh_obj, assignments, report)
        report.append("\n[Info] Materials = PolyGroups/Sections in Unreal Engine")

    if mode == MODE_SEAMS_ONLY:
        mark_segment_seams(mesh_obj, assignments, report)

    if mode in [MODE_UV_ISLANDS, MODE_ALL]:
        unwrap_by_segments(mesh_obj, assignments, report)

    if mode == MODE_SEPARATE:
        # Separate is not included in MODE_ALL by default (destructive)
        separate_by_segments(mesh_obj, assignments, arm_obj, report)

    if mode == MODE_UPDATE_MATS:
        # Update existing materials to properly display vertex colors
        update_materials_for_vertex_colors(mesh_obj, report)

    report.append("\n" + "=" * 50)
    report.append("Segmentation V1 complete.")
    if mode != MODE_UPDATE_MATS:
        report.append("\nWORKFLOW TIP:")
        report.append("1. Run Segmentation_v1.py first (creates materials with vertex color support)")
        report.append("2. Run VertexColors_v1.py (projects/bakes colors to vertices)")
        report.append("3. Press Z > Material Preview to see vertex colors in viewport")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log: {LOG_TEXT_NAME} ---")


if __name__ == "__main__":
    main(RUN_MODE)
