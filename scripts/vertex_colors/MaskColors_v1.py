"""
MaskColors V1 - Vertex Color Mask Generation for Unreal Engine

Creates RGBA mask channels from existing vertex colors for Unreal material customization.

CHANNEL MAPPING:
    R (Red)   = Primary color mask   (designer customizable)
    G (Green) = Secondary color mask (designer customizable)
    B (Blue)  = Accent color mask    (designer customizable)
    A (Alpha) = Emissive mask        (glow intensity)
    (0,0,0,0) = Base/Unmasked        (uses base parameter)

WORKFLOW:
    1. Run VertexColors_v1.py first (PROJECTION or TEXTURE mode)
    2. Run this script to analyze colors and create masks
    3. Export FBX with vertex colors enabled

MODES:
    ANALYZE      - Show color distribution report (no changes)
    AUTO_MASK    - K-means cluster colors, assign by size rank
    MANUAL_MASK  - User defines specific colors -> channels
    MATERIAL_MASK - Use material slots for mapping

CONFIGURATION:
    Edit settings at top of script, then run.
"""

import bpy
import random
from datetime import datetime
from collections import namedtuple

LOG_TEXT_NAME = "MaskColors_V1_Log.txt"

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

# --- MODE SELECTION ---
# "ANALYZE"       = Show color distribution (no changes to mesh)
# "AUTO_MASK"     = Auto-cluster colors and assign to channels by rank
# "MANUAL_MASK"   = Use MANUAL_COLOR_MAP to assign colors to channels
# "MATERIAL_MASK" = Use material slots to assign channels
MODE = "AUTO_MASK"

# --- TARGET ---
TARGET_COLLECTION = "Export"
INPUT_VERTEX_COLOR = "Col"      # Source vertex color layer
OUTPUT_VERTEX_COLOR = "Mask"    # Output mask layer name

# --- CLUSTERING SETTINGS ---
NUM_CLUSTERS = 5                 # Number of color clusters to find
CLUSTER_METHOD = "KMEANS"        # "KMEANS" or "HISTOGRAM"
KMEANS_ITERATIONS = 20           # Max k-means iterations
QUANTIZE_LEVELS = 0              # Color quantization (0 = disabled, 16 = reduce to 16 levels)
MIN_CLUSTER_PERCENTAGE = 0.5     # Ignore clusters smaller than this %

# --- FACE-BASED ASSIGNMENT ---
# When True: All vertices of a face get the SAME mask (contiguous regions)
# When False: Each vertex assigned individually (can create fragmented masks)
FACE_BASED_MASKS = True

# How to determine face color when FACE_BASED_MASKS = True:
# "AVERAGE"  = Average color of all face vertices (good for gradients)
# "DOMINANT" = Most common color among face vertices (good for solid colors)
# "CENTER"   = Color at face center (first vertex, fast)
FACE_COLOR_METHOD = "DOMINANT"

# --- AUTO CHANNEL MAPPING (by cluster rank/vertex count) ---
# Rank 0 = largest cluster, 1 = second largest, etc.
# Set to None to skip that channel
CHANNEL_MAPPING = {
    "BASE": 0,           # Largest cluster -> no mask (0,0,0,0)
    "PRIMARY": 1,        # 2nd largest -> R channel
    "SECONDARY": 2,      # 3rd largest -> G channel
    "ACCENT": 3,         # 4th largest -> B channel
    "EMISSIVE": None,    # Manual only (no auto-assign)
}

# --- MANUAL COLOR MAP (for MANUAL_MASK mode) ---
# Map specific RGB colors to channels
# Format: (R, G, B) -> "PRIMARY" | "SECONDARY" | "ACCENT" | "EMISSIVE" | "BASE"
MANUAL_COLOR_MAP = {
    # Example mappings - edit for your asset:
    # (0.2, 0.4, 0.8): "PRIMARY",    # Blue -> Primary
    # (0.8, 0.3, 0.3): "SECONDARY",  # Red -> Secondary
    # (0.9, 0.8, 0.0): "ACCENT",     # Yellow -> Accent
    # (1.0, 1.0, 1.0): "EMISSIVE",   # White -> Emissive
}

# Color matching tolerance for manual mode (0-1, higher = more lenient)
MANUAL_COLOR_TOLERANCE = 0.15

# --- MATERIAL SLOT MAPPING (for MATERIAL_MASK mode) ---
# Map material names to channels (supports wildcards with *)
MATERIAL_CHANNEL_MAP = {
    # Example mappings - edit for your asset:
    # "M_Torso": "PRIMARY",
    # "M_Head": "BASE",
    # "M_Eyes": "EMISSIVE",
}

# --- OUTPUT CHANNEL COLORS ---
# RGBA values for each channel type
# NOTE: Alpha channel is ONLY for emissive mask!
#       All other channels should have alpha=0 to prevent unwanted glow in Unreal.
CHANNEL_COLORS = {
    "BASE":      (0.0, 0.0, 0.0, 0.0),  # Unmasked - no customization, no emissive
    "PRIMARY":   (1.0, 0.0, 0.0, 0.0),  # R = 1, no emissive
    "SECONDARY": (0.0, 1.0, 0.0, 0.0),  # G = 1, no emissive
    "ACCENT":    (0.0, 0.0, 1.0, 0.0),  # B = 1, no emissive
    "EMISSIVE":  (0.0, 0.0, 0.0, 1.0),  # A = 1 (ONLY channel with emissive)
}

# --- CLEANUP ---
# Delete source color layer after mask creation (Unreal only supports 1 vertex color layer)
DELETE_SOURCE_LAYER = True

# --- DEBUG ---
DEBUG_MODE = True                # Print detailed clustering info
CREATE_DEBUG_MATERIAL = True     # Create material to visualize masks
VERIFY_OUTPUT = True             # Check output after processing


# ============================================================================
# DATA STRUCTURES
# ============================================================================

ColorCluster = namedtuple('ColorCluster', ['centroid', 'indices', 'count', 'percentage'])


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
# HELPERS
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


def depsgraph_update():
    """Force depsgraph update."""
    bpy.context.view_layer.update()


def color_distance(c1: tuple, c2: tuple) -> float:
    """Euclidean distance in RGB space."""
    return sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])) ** 0.5


def quantize_color(color: tuple, levels: int) -> tuple:
    """Reduce color precision to reduce noise."""
    if levels <= 0:
        return color
    return tuple(round(c * levels) / levels for c in color[:3])


# ============================================================================
# COLOR EXTRACTION
# ============================================================================

def extract_vertex_colors(mesh_obj, layer_name: str, report: list) -> tuple:
    """
    Extract vertex colors from mesh.

    Returns:
        colors_by_loop: dict {loop_index: (r, g, b)}
        colors_list: list [(r, g, b), ...] ordered by loop index
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    # Find color attribute
    if hasattr(mesh, 'color_attributes'):
        color_attr = mesh.color_attributes.get(layer_name)
        if not color_attr:
            available = [a.name for a in mesh.color_attributes]
            raise RuntimeError(f"Color layer '{layer_name}' not found. Available: {available}")
    else:
        if layer_name not in mesh.vertex_colors:
            available = [vc.name for vc in mesh.vertex_colors]
            raise RuntimeError(f"Vertex color layer '{layer_name}' not found. Available: {available}")
        color_attr = mesh.vertex_colors[layer_name]

    report.append(f"[Extract] Reading from color layer: {layer_name}")
    report.append(f"[Extract] Total loops: {len(color_attr.data)}")

    colors_by_loop = {}
    colors_list = []

    for i, loop_data in enumerate(color_attr.data):
        rgb = (loop_data.color[0], loop_data.color[1], loop_data.color[2])

        # Apply quantization if enabled
        if QUANTIZE_LEVELS > 0:
            rgb = quantize_color(rgb, QUANTIZE_LEVELS)

        colors_by_loop[i] = rgb
        colors_list.append(rgb)

    return colors_by_loop, colors_list


def extract_face_colors(mesh_obj, colors_by_loop: dict, report: list) -> dict:
    """
    Extract a single representative color per face.

    Uses FACE_COLOR_METHOD to determine how to pick the face color:
    - AVERAGE: Average of all loop colors in face
    - DOMINANT: Most common color (after quantization)
    - CENTER: First loop's color (fast)

    Returns:
        dict: {face_index: (r, g, b)}
    """
    from collections import Counter
    mesh = mesh_obj.data

    face_colors = {}
    method = FACE_COLOR_METHOD

    report.append(f"[Extract] Computing face colors using method: {method}")

    for poly in mesh.polygons:
        loop_indices = list(poly.loop_indices)

        if method == "CENTER":
            # Use first loop color
            face_colors[poly.index] = colors_by_loop.get(loop_indices[0], (0, 0, 0))

        elif method == "AVERAGE":
            # Average all loop colors
            r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
            for loop_idx in loop_indices:
                c = colors_by_loop.get(loop_idx, (0, 0, 0))
                r_sum += c[0]
                g_sum += c[1]
                b_sum += c[2]
            n = len(loop_indices)
            face_colors[poly.index] = (r_sum / n, g_sum / n, b_sum / n)

        elif method == "DOMINANT":
            # Find most common color (quantized for grouping)
            quantized_colors = []
            for loop_idx in loop_indices:
                c = colors_by_loop.get(loop_idx, (0, 0, 0))
                # Quantize for comparison
                q = (round(c[0] * 8) / 8, round(c[1] * 8) / 8, round(c[2] * 8) / 8)
                quantized_colors.append(q)

            counter = Counter(quantized_colors)
            dominant = counter.most_common(1)[0][0]
            face_colors[poly.index] = dominant

    report.append(f"[Extract] Computed colors for {len(face_colors)} faces")
    return face_colors


def cluster_face_colors(face_colors: dict, k: int, max_iterations: int, report: list) -> tuple:
    """
    K-means clustering on face colors.

    Returns:
        tuple: (clusters, face_assignments)
            - clusters: list of ColorCluster with face indices instead of loop indices
            - face_assignments: dict {face_index: cluster_rank}
    """
    face_indices = list(face_colors.keys())
    colors_list = [face_colors[fi] for fi in face_indices]
    n = len(colors_list)

    if n == 0:
        return [], {}

    # Get unique colors for initial centroid selection
    unique_colors = list(set(colors_list))
    actual_k = min(k, len(unique_colors))
    if actual_k < k:
        report.append(f"[Cluster] Reduced k to {actual_k} (only {len(unique_colors)} unique face colors)")

    report.append(f"[Cluster] Running face-based k-means with k={actual_k}, {n} faces")
    print(f"[Cluster] Running face-based k-means with k={actual_k}...")

    # Initialize centroids
    random.seed(42)
    centroids = random.sample(unique_colors, actual_k)

    assignments = None

    for iteration in range(max_iterations):
        new_assignments = [[] for _ in range(actual_k)]

        for idx, color in enumerate(colors_list):
            min_dist = float('inf')
            best_cluster = 0

            for ci, centroid in enumerate(centroids):
                dist = color_distance(color, centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = ci

            new_assignments[best_cluster].append(idx)

        # Check convergence
        if assignments is not None:
            same = all(set(a) == set(b) for a, b in zip(assignments, new_assignments))
            if same:
                report.append(f"[Cluster] Converged at iteration {iteration}")
                break

        assignments = new_assignments

        # Update centroids
        for ci in range(actual_k):
            if assignments[ci]:
                centroid = tuple(
                    sum(colors_list[idx][c] for idx in assignments[ci]) / len(assignments[ci])
                    for c in range(3)
                )
                centroids[ci] = centroid

        if iteration % 5 == 0:
            print(f"[Cluster] Iteration {iteration}/{max_iterations}")

    # Build result with face indices
    clusters = []
    for ci in range(actual_k):
        # Map internal indices back to face indices
        cluster_face_indices = [face_indices[idx] for idx in assignments[ci]]
        count = len(cluster_face_indices)
        percentage = (count / n) * 100 if n > 0 else 0
        clusters.append(ColorCluster(
            centroid=centroids[ci],
            indices=cluster_face_indices,  # These are face indices now
            count=count,
            percentage=percentage
        ))

    # Sort by count descending
    clusters.sort(key=lambda c: c.count, reverse=True)

    # Build face -> rank mapping
    face_assignments = {}
    for rank, cluster in enumerate(clusters):
        for face_idx in cluster.indices:
            face_assignments[face_idx] = rank

    # Report
    report.append(f"\n[Cluster] Found {len(clusters)} face clusters:")
    for i, cluster in enumerate(clusters):
        rgb = cluster.centroid
        report.append(
            f"  Rank {i}: {cluster.count} faces ({cluster.percentage:.1f}%) "
            f"- RGB({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
        )

    return clusters, face_assignments


# ============================================================================
# COLOR ANALYSIS
# ============================================================================

def analyze_color_distribution(colors_list: list, report: list) -> dict:
    """
    Analyze color distribution and return statistics.

    Returns:
        dict: {color: count} sorted by count descending
    """
    from collections import Counter

    # Count occurrences (quantize for grouping)
    quantized = [quantize_color(c, 16) for c in colors_list]
    counter = Counter(quantized)

    total = len(colors_list)

    report.append(f"\n[Analysis] Total vertices: {total}")
    report.append(f"[Analysis] Unique colors (quantized): {len(counter)}")
    report.append("\n[Analysis] Top 10 colors by frequency:")

    for i, (color, count) in enumerate(counter.most_common(10)):
        pct = (count / total) * 100
        report.append(f"  {i+1}. RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}): {count} ({pct:.1f}%)")

    return dict(counter.most_common())


# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

def kmeans_cluster_colors(colors_list: list, k: int, max_iterations: int, report: list) -> list:
    """
    K-means clustering for vertex colors.

    Args:
        colors_list: List of (r,g,b) tuples
        k: Number of clusters
        max_iterations: Max iterations before stopping
        report: List for logging

    Returns:
        List of ColorCluster namedtuples, sorted by count descending
    """
    n = len(colors_list)
    if n == 0:
        return []

    # Get unique colors for initial centroid selection
    unique_colors = list(set(colors_list))
    if len(unique_colors) < k:
        k = len(unique_colors)
        report.append(f"[Cluster] Reduced k to {k} (only {len(unique_colors)} unique colors)")

    report.append(f"[Cluster] Running k-means with k={k}, max_iter={max_iterations}")
    print(f"[Cluster] Running k-means with k={k}...")

    # Initialize centroids randomly
    random.seed(42)  # Reproducible results
    centroids = random.sample(unique_colors, k)

    assignments = None

    for iteration in range(max_iterations):
        # Assign each color to nearest centroid
        new_assignments = [[] for _ in range(k)]

        for idx, color in enumerate(colors_list):
            min_dist = float('inf')
            best_cluster = 0

            for ci, centroid in enumerate(centroids):
                dist = color_distance(color, centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = ci

            new_assignments[best_cluster].append(idx)

        # Check convergence
        if assignments is not None:
            # Compare assignments
            same = all(
                set(a) == set(b) for a, b in zip(assignments, new_assignments)
            )
            if same:
                report.append(f"[Cluster] Converged at iteration {iteration}")
                break

        assignments = new_assignments

        # Update centroids to mean of assigned colors
        for ci in range(k):
            if assignments[ci]:
                centroid = tuple(
                    sum(colors_list[idx][c] for idx in assignments[ci]) / len(assignments[ci])
                    for c in range(3)
                )
                centroids[ci] = centroid

        # Progress
        if iteration % 5 == 0:
            print(f"[Cluster] Iteration {iteration}/{max_iterations}")

    # Build result
    clusters = []
    for ci in range(k):
        count = len(assignments[ci])
        percentage = (count / n) * 100 if n > 0 else 0
        clusters.append(ColorCluster(
            centroid=centroids[ci],
            indices=assignments[ci],
            count=count,
            percentage=percentage
        ))

    # Sort by count descending
    clusters.sort(key=lambda c: c.count, reverse=True)

    # Report
    report.append(f"\n[Cluster] Found {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        rgb = cluster.centroid
        report.append(
            f"  Rank {i}: {cluster.count} verts ({cluster.percentage:.1f}%) "
            f"- RGB({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
        )

    return clusters


# ============================================================================
# HISTOGRAM CLUSTERING (Alternative)
# ============================================================================

def histogram_cluster_colors(colors_list: list, bin_size: int, report: list) -> list:
    """
    Cluster colors using histogram binning (faster but less accurate).

    Args:
        colors_list: List of (r,g,b) tuples
        bin_size: Number of bins per channel (8 = 512 total bins)
        report: List for logging

    Returns:
        List of ColorCluster namedtuples, sorted by count descending
    """
    from collections import defaultdict

    n = len(colors_list)
    if n == 0:
        return []

    report.append(f"[Histogram] Binning with {bin_size} bins per channel...")

    bins = defaultdict(list)
    for idx, color in enumerate(colors_list):
        bin_key = tuple(int(c * bin_size) for c in color[:3])
        bins[bin_key].append(idx)

    # Convert to clusters
    clusters = []
    for bin_key, indices in bins.items():
        count = len(indices)
        percentage = (count / n) * 100

        if percentage >= MIN_CLUSTER_PERCENTAGE:
            # Centroid is center of bin
            centroid = tuple((k + 0.5) / bin_size for k in bin_key)
            clusters.append(ColorCluster(
                centroid=centroid,
                indices=indices,
                count=count,
                percentage=percentage
            ))

    # Sort by count descending
    clusters.sort(key=lambda c: c.count, reverse=True)

    report.append(f"[Histogram] Found {len(clusters)} clusters (above {MIN_CLUSTER_PERCENTAGE}%)")

    return clusters


# ============================================================================
# CHANNEL ASSIGNMENT
# ============================================================================

def assign_clusters_to_channels(clusters: list, channel_mapping: dict, report: list) -> dict:
    """
    Map ranked clusters to mask channels.

    Args:
        clusters: List of ColorCluster, already sorted by count
        channel_mapping: Dict like {"BASE": 0, "PRIMARY": 1, ...}
        report: List for logging

    Returns:
        Dict: {loop_index: channel_name, ...}
    """
    # Invert mapping: rank -> channel
    rank_to_channel = {}
    for channel, rank in channel_mapping.items():
        if rank is not None:
            rank_to_channel[rank] = channel

    loop_assignments = {}
    channel_stats = {ch: 0 for ch in CHANNEL_COLORS.keys()}

    report.append("\n[Assign] Mapping clusters to channels:")

    for rank, cluster in enumerate(clusters):
        channel = rank_to_channel.get(rank, "BASE")
        rgb = cluster.centroid

        report.append(
            f"  Rank {rank}: {cluster.count} verts ({cluster.percentage:.1f}%) "
            f"-> {channel} (RGB: {rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
        )

        for loop_idx in cluster.indices:
            loop_assignments[loop_idx] = channel
            channel_stats[channel] += 1

    report.append("\n[Assign] Channel totals:")
    for channel, count in channel_stats.items():
        if count > 0:
            report.append(f"  {channel}: {count} vertices")

    return loop_assignments


def assign_colors_manual(colors_by_loop: dict, color_map: dict, tolerance: float, report: list) -> dict:
    """
    Assign channels based on manual color definitions.

    Args:
        colors_by_loop: {loop_index: (r, g, b)}
        color_map: {(r, g, b): channel_name}
        tolerance: Max color distance for matching
        report: List for logging

    Returns:
        Dict: {loop_index: channel_name, ...}
    """
    loop_assignments = {}
    channel_stats = {ch: 0 for ch in CHANNEL_COLORS.keys()}
    unmatched = 0

    report.append(f"\n[Manual] Matching colors with tolerance {tolerance}...")
    report.append(f"[Manual] Color map has {len(color_map)} entries")

    for loop_idx, color in colors_by_loop.items():
        matched_channel = None
        min_dist = float('inf')

        for map_color, channel in color_map.items():
            dist = color_distance(color, map_color)
            if dist < min_dist and dist <= tolerance:
                min_dist = dist
                matched_channel = channel

        if matched_channel:
            loop_assignments[loop_idx] = matched_channel
            channel_stats[matched_channel] += 1
        else:
            loop_assignments[loop_idx] = "BASE"
            channel_stats["BASE"] += 1
            unmatched += 1

    report.append(f"[Manual] Matched: {len(loop_assignments) - unmatched}, Unmatched: {unmatched}")
    for channel, count in channel_stats.items():
        if count > 0:
            report.append(f"  {channel}: {count} vertices")

    return loop_assignments


def assign_by_material(mesh_obj, material_map: dict, report: list) -> dict:
    """
    Assign channels based on material slot assignments.

    Args:
        mesh_obj: Blender mesh object
        material_map: {material_name: channel_name}
        report: List for logging

    Returns:
        Dict: {loop_index: channel_name, ...}
    """
    import fnmatch

    mesh = mesh_obj.data
    loop_assignments = {}
    channel_stats = {ch: 0 for ch in CHANNEL_COLORS.keys()}

    report.append(f"\n[Material] Mapping {len(mesh.materials)} materials to channels...")

    # Build material index -> channel mapping
    mat_to_channel = {}
    for mat_idx, mat in enumerate(mesh.materials):
        if mat is None:
            continue

        matched_channel = "BASE"
        for pattern, channel in material_map.items():
            if fnmatch.fnmatch(mat.name, pattern):
                matched_channel = channel
                break

        mat_to_channel[mat_idx] = matched_channel
        report.append(f"  {mat.name} (slot {mat_idx}) -> {matched_channel}")

    # Assign loops based on face material
    for poly in mesh.polygons:
        channel = mat_to_channel.get(poly.material_index, "BASE")
        for loop_idx in poly.loop_indices:
            loop_assignments[loop_idx] = channel
            channel_stats[channel] += 1

    for channel, count in channel_stats.items():
        if count > 0:
            report.append(f"  {channel}: {count} vertices")

    return loop_assignments


# ============================================================================
# MASK CREATION
# ============================================================================

def create_mask_vertex_colors(mesh_obj, loop_assignments: dict, output_name: str, report: list):
    """
    Create output vertex color layer with mask values.

    Args:
        mesh_obj: Blender mesh object
        loop_assignments: {loop_index: channel_name, ...}
        output_name: Name for output color layer
        report: List for logging
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append(f"\n[Mask] Creating output layer: {output_name}")

    # Remove existing output layer
    if hasattr(mesh, 'color_attributes'):
        existing = mesh.color_attributes.get(output_name)
        if existing:
            mesh.color_attributes.remove(existing)

        # Create new layer
        mask_layer = mesh.color_attributes.new(
            name=output_name,
            type='BYTE_COLOR',
            domain='CORNER'
        )
        color_data = mask_layer.data
    else:
        if output_name in mesh.vertex_colors:
            mesh.vertex_colors.remove(mesh.vertex_colors[output_name])

        mask_layer = mesh.vertex_colors.new(name=output_name)
        color_data = mask_layer.data

    # Assign mask colors
    for loop_idx, channel in loop_assignments.items():
        color = CHANNEL_COLORS.get(channel, (0, 0, 0, 0))
        color_data[loop_idx].color = color

    # Force update
    mesh.update()

    # Set as render color for export
    if hasattr(mesh, 'color_attributes'):
        mesh.color_attributes.render_color_index = mesh.color_attributes.find(output_name)

    depsgraph_update()
    report.append(f"[Mask] Created mask layer with {len(loop_assignments)} assignments")


def create_face_mask_vertex_colors(mesh_obj, face_assignments: dict, channel_mapping: dict,
                                    output_name: str, report: list):
    """
    Create output vertex color layer with mask values, assigning by face.

    All loops in a face get the same mask color, ensuring contiguous regions.

    Args:
        mesh_obj: Blender mesh object
        face_assignments: {face_index: cluster_rank, ...}
        channel_mapping: {"BASE": 0, "PRIMARY": 1, ...}
        output_name: Name for output color layer
        report: List for logging
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append(f"\n[Mask] Creating face-based output layer: {output_name}")

    # Invert mapping: rank -> channel
    rank_to_channel = {}
    for channel, rank in channel_mapping.items():
        if rank is not None:
            rank_to_channel[rank] = channel

    # Remove existing output layer
    if hasattr(mesh, 'color_attributes'):
        existing = mesh.color_attributes.get(output_name)
        if existing:
            mesh.color_attributes.remove(existing)

        # Create new layer
        mask_layer = mesh.color_attributes.new(
            name=output_name,
            type='BYTE_COLOR',
            domain='CORNER'
        )
        color_data = mask_layer.data
    else:
        if output_name in mesh.vertex_colors:
            mesh.vertex_colors.remove(mesh.vertex_colors[output_name])

        mask_layer = mesh.vertex_colors.new(name=output_name)
        color_data = mask_layer.data

    # Assign mask colors by face
    channel_stats = {ch: 0 for ch in CHANNEL_COLORS.keys()}
    assigned_loops = 0

    for poly in mesh.polygons:
        # Get cluster rank for this face
        rank = face_assignments.get(poly.index, 0)
        # Map rank to channel
        channel = rank_to_channel.get(rank, "BASE")
        # Get mask color for channel
        mask_color = CHANNEL_COLORS.get(channel, (0, 0, 0, 0))

        # Apply same color to ALL loops in this face
        for loop_idx in poly.loop_indices:
            color_data[loop_idx].color = mask_color
            assigned_loops += 1
            channel_stats[channel] += 1

    # Force update
    mesh.update()

    # Set as render color for export
    if hasattr(mesh, 'color_attributes'):
        mesh.color_attributes.render_color_index = mesh.color_attributes.find(output_name)

    depsgraph_update()

    report.append(f"[Mask] Assigned {assigned_loops} loops across {len(face_assignments)} faces")
    report.append("\n[Mask] Channel distribution:")
    for channel, count in channel_stats.items():
        if count > 0:
            pct = (count / assigned_loops) * 100 if assigned_loops > 0 else 0
            report.append(f"  {channel}: {count} loops ({pct:.1f}%)")


# ============================================================================
# DEBUG MATERIAL
# ============================================================================

def create_mask_debug_material(mesh_obj, report: list):
    """
    Create material that visualizes mask channels using Separate RGB.
    R = Red, G = Green, B = Blue, A = White overlay
    """
    mat_name = "M_Mask_Debug"

    # Remove existing
    mat = bpy.data.materials.get(mat_name)
    if mat:
        bpy.data.materials.remove(mat)

    # Create new
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Emission (shows colors directly)
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (400, 0)
    emission.inputs['Strength'].default_value = 1.0
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # Vertex Color node
    try:
        vert_color = nodes.new('ShaderNodeVertexColor')
        vert_color.layer_name = OUTPUT_VERTEX_COLOR
    except:
        vert_color = nodes.new('ShaderNodeAttribute')
        vert_color.attribute_name = OUTPUT_VERTEX_COLOR
        vert_color.attribute_type = 'GEOMETRY'
    vert_color.location = (0, 0)

    # Connect directly to emission (shows RGB mask)
    links.new(vert_color.outputs['Color'], emission.inputs['Color'])

    # Clear existing materials and assign
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    report.append(f"[Material] Created debug material: {mat_name}")
    report.append("[Material] View in Material Preview mode (Z > Material Preview)")
    print(f"[Material] Created '{mat_name}' for mask visualization")


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_mask_output(mesh_obj, output_name: str, report: list):
    """Verify mask layer was created correctly."""
    mesh = mesh_obj.data

    if hasattr(mesh, 'color_attributes'):
        mask_attr = mesh.color_attributes.get(output_name)
        if not mask_attr:
            report.append(f"[Verify] ERROR: Mask layer '{output_name}' not found!")
            return False
    else:
        if output_name not in mesh.vertex_colors:
            report.append(f"[Verify] ERROR: Mask layer '{output_name}' not found!")
            return False
        mask_attr = mesh.vertex_colors[output_name]

    # Sample first few
    report.append(f"\n[Verify] Checking mask layer '{output_name}':")
    channel_counts = {ch: 0 for ch in CHANNEL_COLORS.keys()}

    for i, loop_data in enumerate(mask_attr.data):
        color = loop_data.color[:4]

        # Identify channel by color
        for ch_name, ch_color in CHANNEL_COLORS.items():
            if abs(color[0] - ch_color[0]) < 0.1 and \
               abs(color[1] - ch_color[1]) < 0.1 and \
               abs(color[2] - ch_color[2]) < 0.1:
                channel_counts[ch_name] += 1
                break

    for channel, count in channel_counts.items():
        if count > 0:
            report.append(f"  {channel}: {count} vertices")

    return True


# ============================================================================
# MAIN
# ============================================================================

def main(mode: str = None, collection_name: str = None):
    """
    Main entry point.

    Args:
        mode: Override mode from config
        collection_name: Override target collection
    """
    report = []
    report.append("MaskColors V1 - Vertex Color Mask Generation")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Mode: {mode or MODE}\n")

    run_mode = mode or MODE
    col_name = collection_name or TARGET_COLLECTION

    # Find collection
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
    report.append(f"Loops: {len(mesh_obj.data.loops)}")

    # Extract colors
    try:
        colors_by_loop, colors_list = extract_vertex_colors(mesh_obj, INPUT_VERTEX_COLOR, report)
    except RuntimeError as e:
        report.append(f"\nERROR: {e}")
        report.append("\nMake sure to run VertexColors_v1.py first to create vertex colors!")
        log_text = "\n".join(report)
        log_to_text(log_text)
        print(log_text)
        raise

    # Process based on mode
    if run_mode == "ANALYZE":
        # Just analyze and report
        analyze_color_distribution(colors_list, report)
        report.append("\n[Analyze] No changes made to mesh.")

    elif run_mode == "AUTO_MASK":
        # Cluster and auto-assign
        if FACE_BASED_MASKS:
            # Face-based clustering for contiguous masks
            report.append(f"\n[Mode] FACE_BASED_MASKS enabled (method: {FACE_COLOR_METHOD})")
            print(f"[Mode] Using face-based clustering...")

            # Extract face colors
            face_colors = extract_face_colors(mesh_obj, colors_by_loop, report)

            # Cluster faces
            clusters, face_assignments = cluster_face_colors(
                face_colors, NUM_CLUSTERS, KMEANS_ITERATIONS, report
            )

            # Create mask by face
            create_face_mask_vertex_colors(
                mesh_obj, face_assignments, CHANNEL_MAPPING,
                OUTPUT_VERTEX_COLOR, report
            )
        else:
            # Original per-loop clustering
            report.append("\n[Mode] Per-vertex clustering (FACE_BASED_MASKS = False)")
            if CLUSTER_METHOD == "KMEANS":
                clusters = kmeans_cluster_colors(colors_list, NUM_CLUSTERS, KMEANS_ITERATIONS, report)
            else:
                clusters = histogram_cluster_colors(colors_list, 8, report)

            loop_assignments = assign_clusters_to_channels(clusters, CHANNEL_MAPPING, report)
            create_mask_vertex_colors(mesh_obj, loop_assignments, OUTPUT_VERTEX_COLOR, report)

    elif run_mode == "MANUAL_MASK":
        # Use manual color map
        if not MANUAL_COLOR_MAP:
            raise RuntimeError("MANUAL_COLOR_MAP is empty! Add color mappings to config.")

        loop_assignments = assign_colors_manual(colors_by_loop, MANUAL_COLOR_MAP, MANUAL_COLOR_TOLERANCE, report)
        create_mask_vertex_colors(mesh_obj, loop_assignments, OUTPUT_VERTEX_COLOR, report)

    elif run_mode == "MATERIAL_MASK":
        # Use material slots
        if not MATERIAL_CHANNEL_MAP:
            raise RuntimeError("MATERIAL_CHANNEL_MAP is empty! Add material mappings to config.")

        loop_assignments = assign_by_material(mesh_obj, MATERIAL_CHANNEL_MAP, report)
        create_mask_vertex_colors(mesh_obj, loop_assignments, OUTPUT_VERTEX_COLOR, report)

    else:
        raise RuntimeError(f"Unknown mode: {run_mode}")

    # Post-processing
    if run_mode != "ANALYZE":
        if VERIFY_OUTPUT:
            verify_mask_output(mesh_obj, OUTPUT_VERTEX_COLOR, report)

        if CREATE_DEBUG_MATERIAL:
            create_mask_debug_material(mesh_obj, report)

        # Delete source color layer (Unreal only supports 1 vertex color layer)
        if DELETE_SOURCE_LAYER:
            mesh = mesh_obj.data
            if hasattr(mesh, 'color_attributes'):
                source_attr = mesh.color_attributes.get(INPUT_VERTEX_COLOR)
                if source_attr and INPUT_VERTEX_COLOR != OUTPUT_VERTEX_COLOR:
                    mesh.color_attributes.remove(source_attr)
                    report.append(f"\n[Cleanup] Deleted source layer '{INPUT_VERTEX_COLOR}' (Unreal compatibility)")
                    print(f"[Cleanup] Deleted '{INPUT_VERTEX_COLOR}' - only '{OUTPUT_VERTEX_COLOR}' remains")
            else:
                if INPUT_VERTEX_COLOR in mesh.vertex_colors and INPUT_VERTEX_COLOR != OUTPUT_VERTEX_COLOR:
                    mesh.vertex_colors.remove(mesh.vertex_colors[INPUT_VERTEX_COLOR])
                    report.append(f"\n[Cleanup] Deleted source layer '{INPUT_VERTEX_COLOR}' (Unreal compatibility)")

    # Finish
    report.append("\n" + "=" * 50)
    report.append("MaskColors V1 complete.")
    report.append("\nFBX Export: Enable 'Vertex Colors' in export settings.")
    report.append("Unreal: Use 'Vertex Color' node in material, separate R/G/B/A channels.")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log: {LOG_TEXT_NAME} ---")


if __name__ == "__main__":
    main()
