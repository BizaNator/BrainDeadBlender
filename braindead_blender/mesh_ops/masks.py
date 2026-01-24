"""
masks - Vertex Color Mask Generation

Creates RGBA mask channels from existing vertex colors for Unreal Engine
material customization (Primary, Secondary, Accent, Emissive channels).

Channel Mapping:
    R (Red)   = Primary color mask
    G (Green) = Secondary color mask
    B (Blue)  = Accent color mask
    A (Alpha) = Emissive mask
    (0,0,0,0) = Base/Unmasked

Usage:
    from mesh_ops import masks

    # Analyze color distribution
    info = masks.analyze_color_distribution(colors_list)

    # Auto-mask with k-means clustering
    masks.auto_mask(obj, num_clusters=4, face_based=True)

    # Manual color assignment
    masks.assign_colors_manual(colors_by_loop, color_map, tolerance=0.15)
"""

import bpy
import random
from collections import namedtuple, Counter

from .utils import ensure_object_mode

# ============================================================================
# CONSTANTS
# ============================================================================

CHANNEL_COLORS = {
    "BASE":      (0.0, 0.0, 0.0, 0.0),
    "PRIMARY":   (1.0, 0.0, 0.0, 0.0),
    "SECONDARY": (0.0, 1.0, 0.0, 0.0),
    "ACCENT":    (0.0, 0.0, 1.0, 0.0),
    "EMISSIVE":  (0.0, 0.0, 0.0, 1.0),
}

ColorCluster = namedtuple('ColorCluster', ['centroid', 'indices', 'count', 'percentage'])


# ============================================================================
# HELPERS
# ============================================================================

def color_distance(c1, c2):
    """Euclidean distance in RGB space."""
    return sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])) ** 0.5


def quantize_color(color, levels):
    """Reduce color precision to reduce noise."""
    if levels <= 0:
        return color
    return tuple(round(c * levels) / levels for c in color[:3])


# ============================================================================
# COLOR EXTRACTION
# ============================================================================

def extract_vertex_colors(obj, layer_name="Color", quantize_levels=0, report=None):
    """
    Extract vertex colors from mesh.

    Args:
        obj: Blender mesh object
        layer_name: Name of color attribute to read
        quantize_levels: Color quantization (0 = disabled)
        report: Optional list for logging

    Returns:
        tuple: (colors_by_loop, colors_list)
            - colors_by_loop: dict {loop_index: (r, g, b)}
            - colors_list: list [(r, g, b), ...] ordered by loop index
    """
    if report is None:
        report = []

    ensure_object_mode()
    mesh = obj.data

    # Find color attribute
    color_attr = mesh.color_attributes.get(layer_name)
    if not color_attr:
        available = [a.name for a in mesh.color_attributes]
        raise RuntimeError(f"Color layer '{layer_name}' not found. Available: {available}")

    report.append(f"[Extract] Reading from color layer: {layer_name}")
    report.append(f"[Extract] Total loops: {len(color_attr.data)}")

    colors_by_loop = {}
    colors_list = []

    for i, loop_data in enumerate(color_attr.data):
        rgb = (loop_data.color[0], loop_data.color[1], loop_data.color[2])

        if quantize_levels > 0:
            rgb = quantize_color(rgb, quantize_levels)

        colors_by_loop[i] = rgb
        colors_list.append(rgb)

    return colors_by_loop, colors_list


def extract_face_colors(obj, colors_by_loop, method="DOMINANT", report=None):
    """
    Extract a single representative color per face.

    Args:
        obj: Blender mesh object
        colors_by_loop: dict {loop_index: (r, g, b)}
        method: "AVERAGE", "DOMINANT", or "CENTER"
        report: Optional list for logging

    Returns:
        dict: {face_index: (r, g, b)}
    """
    if report is None:
        report = []

    mesh = obj.data
    face_colors = {}

    report.append(f"[Extract] Computing face colors using method: {method}")

    for poly in mesh.polygons:
        loop_indices = list(poly.loop_indices)

        if method == "CENTER":
            face_colors[poly.index] = colors_by_loop.get(loop_indices[0], (0, 0, 0))

        elif method == "AVERAGE":
            r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
            for loop_idx in loop_indices:
                c = colors_by_loop.get(loop_idx, (0, 0, 0))
                r_sum += c[0]
                g_sum += c[1]
                b_sum += c[2]
            n = len(loop_indices)
            face_colors[poly.index] = (r_sum / n, g_sum / n, b_sum / n)

        elif method == "DOMINANT":
            quantized_colors = []
            for loop_idx in loop_indices:
                c = colors_by_loop.get(loop_idx, (0, 0, 0))
                q = (round(c[0] * 8) / 8, round(c[1] * 8) / 8, round(c[2] * 8) / 8)
                quantized_colors.append(q)

            counter = Counter(quantized_colors)
            dominant = counter.most_common(1)[0][0]
            face_colors[poly.index] = dominant

    report.append(f"[Extract] Computed colors for {len(face_colors)} faces")
    return face_colors


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_color_distribution(colors_list, report=None):
    """
    Analyze color distribution and return statistics.

    Args:
        colors_list: List of (r, g, b) tuples
        report: Optional list for logging

    Returns:
        dict: {color: count} sorted by count descending
    """
    if report is None:
        report = []

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
# CLUSTERING
# ============================================================================

def kmeans_cluster_colors(colors_list, k=4, max_iterations=20, report=None):
    """
    K-means clustering for vertex colors.

    Args:
        colors_list: List of (r, g, b) tuples
        k: Number of clusters
        max_iterations: Max iterations before stopping
        report: Optional list for logging

    Returns:
        List of ColorCluster namedtuples, sorted by count descending
    """
    if report is None:
        report = []

    n = len(colors_list)
    if n == 0:
        return []

    unique_colors = list(set(colors_list))
    if len(unique_colors) < k:
        k = len(unique_colors)
        report.append(f"[Cluster] Reduced k to {k} (only {len(unique_colors)} unique colors)")

    report.append(f"[Cluster] Running k-means with k={k}, max_iter={max_iterations}")

    random.seed(42)
    centroids = random.sample(unique_colors, k)
    assignments = None

    for iteration in range(max_iterations):
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

        if assignments is not None:
            same = all(set(a) == set(b) for a, b in zip(assignments, new_assignments))
            if same:
                report.append(f"[Cluster] Converged at iteration {iteration}")
                break

        assignments = new_assignments

        for ci in range(k):
            if assignments[ci]:
                centroid = tuple(
                    sum(colors_list[idx][c] for idx in assignments[ci]) / len(assignments[ci])
                    for c in range(3)
                )
                centroids[ci] = centroid

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

    clusters.sort(key=lambda c: c.count, reverse=True)

    report.append(f"\n[Cluster] Found {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        rgb = cluster.centroid
        report.append(
            f"  Rank {i}: {cluster.count} verts ({cluster.percentage:.1f}%) "
            f"- RGB({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
        )

    return clusters


def cluster_face_colors(face_colors, k=4, max_iterations=20, report=None):
    """
    K-means clustering on face colors.

    Args:
        face_colors: dict {face_index: (r, g, b)}
        k: Number of clusters
        max_iterations: Max iterations
        report: Optional list for logging

    Returns:
        tuple: (clusters, face_assignments)
            - clusters: list of ColorCluster with face indices
            - face_assignments: dict {face_index: cluster_rank}
    """
    if report is None:
        report = []

    face_indices = list(face_colors.keys())
    colors_list = [face_colors[fi] for fi in face_indices]
    n = len(colors_list)

    if n == 0:
        return [], {}

    unique_colors = list(set(colors_list))
    actual_k = min(k, len(unique_colors))
    if actual_k < k:
        report.append(f"[Cluster] Reduced k to {actual_k} (only {len(unique_colors)} unique face colors)")

    report.append(f"[Cluster] Running face-based k-means with k={actual_k}, {n} faces")

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

        if assignments is not None:
            same = all(set(a) == set(b) for a, b in zip(assignments, new_assignments))
            if same:
                report.append(f"[Cluster] Converged at iteration {iteration}")
                break

        assignments = new_assignments

        for ci in range(actual_k):
            if assignments[ci]:
                centroid = tuple(
                    sum(colors_list[idx][c] for idx in assignments[ci]) / len(assignments[ci])
                    for c in range(3)
                )
                centroids[ci] = centroid

    clusters = []
    for ci in range(actual_k):
        cluster_face_indices = [face_indices[idx] for idx in assignments[ci]]
        count = len(cluster_face_indices)
        percentage = (count / n) * 100 if n > 0 else 0
        clusters.append(ColorCluster(
            centroid=centroids[ci],
            indices=cluster_face_indices,
            count=count,
            percentage=percentage
        ))

    clusters.sort(key=lambda c: c.count, reverse=True)

    face_assignments = {}
    for rank, cluster in enumerate(clusters):
        for face_idx in cluster.indices:
            face_assignments[face_idx] = rank

    report.append(f"\n[Cluster] Found {len(clusters)} face clusters:")
    for i, cluster in enumerate(clusters):
        rgb = cluster.centroid
        report.append(
            f"  Rank {i}: {cluster.count} faces ({cluster.percentage:.1f}%) "
            f"- RGB({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
        )

    return clusters, face_assignments


# ============================================================================
# CHANNEL ASSIGNMENT
# ============================================================================

def assign_clusters_to_channels(clusters, channel_mapping, report=None):
    """
    Map ranked clusters to mask channels.

    Args:
        clusters: List of ColorCluster, sorted by count
        channel_mapping: Dict like {"BASE": 0, "PRIMARY": 1, ...}
        report: Optional list for logging

    Returns:
        dict: {loop_index: channel_name}
    """
    if report is None:
        report = []

    rank_to_channel = {}
    for channel, rank in channel_mapping.items():
        if rank is not None and rank >= 0:
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


def assign_colors_manual(colors_by_loop, color_map, tolerance=0.15, report=None):
    """
    Assign channels based on manual color definitions.

    Args:
        colors_by_loop: {loop_index: (r, g, b)}
        color_map: {(r, g, b): channel_name}
        tolerance: Max color distance for matching
        report: Optional list for logging

    Returns:
        dict: {loop_index: channel_name}
    """
    if report is None:
        report = []

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


def assign_by_material(obj, material_map, report=None):
    """
    Assign channels based on material slot assignments.

    Args:
        obj: Blender mesh object
        material_map: {material_name: channel_name} or None for index-based
        report: Optional list for logging

    Returns:
        dict: {loop_index: channel_name}
    """
    if report is None:
        report = []

    import fnmatch

    mesh = obj.data
    loop_assignments = {}
    channel_stats = {ch: 0 for ch in CHANNEL_COLORS.keys()}

    report.append(f"\n[Material] Mapping {len(mesh.materials)} materials to channels...")

    # Build material index -> channel mapping
    mat_to_channel = {}
    if material_map:
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
    else:
        # Index-based: slot 0=BASE, 1=PRIMARY, 2=SECONDARY, 3=ACCENT, 4+=BASE
        channel_order = ["BASE", "PRIMARY", "SECONDARY", "ACCENT"]
        for mat_idx, mat in enumerate(mesh.materials):
            if mat is None:
                continue
            channel = channel_order[mat_idx] if mat_idx < len(channel_order) else "BASE"
            mat_to_channel[mat_idx] = channel
            mat_name = mat.name if mat else f"slot_{mat_idx}"
            report.append(f"  {mat_name} (slot {mat_idx}) -> {channel}")

    # Assign loops based on face material
    for poly in mesh.polygons:
        channel = mat_to_channel.get(poly.material_index, "BASE")
        for loop_idx in poly.loop_indices:
            loop_assignments[loop_idx] = channel
            channel_stats[channel] += 1

    for channel, count in channel_stats.items():
        if count > 0:
            report.append(f"  {channel}: {count} loops")

    return loop_assignments


# ============================================================================
# MASK CREATION
# ============================================================================

def create_mask_vertex_colors(obj, loop_assignments, output_name="Mask", report=None):
    """
    Create output vertex color layer with mask values (per-loop).

    Args:
        obj: Blender mesh object
        loop_assignments: {loop_index: channel_name}
        output_name: Name for output color layer
        report: Optional list for logging
    """
    if report is None:
        report = []

    ensure_object_mode()
    mesh = obj.data

    report.append(f"\n[Mask] Creating output layer: {output_name}")

    # Remove existing output layer
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

    # Assign mask colors
    for loop_idx, channel in loop_assignments.items():
        color = CHANNEL_COLORS.get(channel, (0, 0, 0, 0))
        color_data[loop_idx].color = color

    mesh.update()

    # Set as render color for export
    mesh.color_attributes.render_color_index = mesh.color_attributes.find(output_name)

    bpy.context.view_layer.update()
    report.append(f"[Mask] Created mask layer with {len(loop_assignments)} assignments")


def create_face_mask_vertex_colors(obj, face_assignments, channel_mapping, output_name="Mask", report=None):
    """
    Create output vertex color layer with mask values (per-face, contiguous regions).

    All loops in a face get the same mask color.

    Args:
        obj: Blender mesh object
        face_assignments: {face_index: cluster_rank}
        channel_mapping: {"BASE": 0, "PRIMARY": 1, ...}
        output_name: Name for output color layer
        report: Optional list for logging
    """
    if report is None:
        report = []

    ensure_object_mode()
    mesh = obj.data

    report.append(f"\n[Mask] Creating face-based output layer: {output_name}")

    # Invert mapping: rank -> channel
    rank_to_channel = {}
    for channel, rank in channel_mapping.items():
        if rank is not None and rank >= 0:
            rank_to_channel[rank] = channel

    # Remove existing output layer
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

    # Assign mask colors by face
    channel_stats = {ch: 0 for ch in CHANNEL_COLORS.keys()}
    assigned_loops = 0

    for poly in mesh.polygons:
        rank = face_assignments.get(poly.index, 0)
        channel = rank_to_channel.get(rank, "BASE")
        mask_color = CHANNEL_COLORS.get(channel, (0, 0, 0, 0))

        for loop_idx in poly.loop_indices:
            color_data[loop_idx].color = mask_color
            assigned_loops += 1
            channel_stats[channel] += 1

    mesh.update()

    # Set as render color for export
    mesh.color_attributes.render_color_index = mesh.color_attributes.find(output_name)

    bpy.context.view_layer.update()

    report.append(f"[Mask] Assigned {assigned_loops} loops across {len(face_assignments)} faces")
    report.append("\n[Mask] Channel distribution:")
    for channel, count in channel_stats.items():
        if count > 0:
            pct = (count / assigned_loops) * 100 if assigned_loops > 0 else 0
            report.append(f"  {channel}: {count} loops ({pct:.1f}%)")


def clear_mask(obj, output_name="Mask", report=None):
    """
    Reset all mask values to (0,0,0,0) BASE.

    Args:
        obj: Blender mesh object
        output_name: Name of mask color layer
        report: Optional list for logging
    """
    if report is None:
        report = []

    ensure_object_mode()
    mesh = obj.data

    mask_attr = mesh.color_attributes.get(output_name)
    if not mask_attr:
        report.append(f"[Clear] Mask layer '{output_name}' not found, nothing to clear")
        return

    for loop_data in mask_attr.data:
        loop_data.color = (0.0, 0.0, 0.0, 0.0)

    mesh.update()
    bpy.context.view_layer.update()
    report.append(f"[Clear] Reset all values in '{output_name}' to (0,0,0,0)")


# ============================================================================
# DEBUG MATERIAL
# ============================================================================

def create_mask_debug_material(obj, output_name="Mask", report=None):
    """
    Create material that visualizes mask channels.

    Args:
        obj: Blender mesh object
        output_name: Name of mask color layer to visualize
        report: Optional list for logging
    """
    if report is None:
        report = []

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
        vert_color.layer_name = output_name
    except Exception:
        vert_color = nodes.new('ShaderNodeAttribute')
        vert_color.attribute_name = output_name
        vert_color.attribute_type = 'GEOMETRY'
    vert_color.location = (0, 0)

    # Connect directly to emission
    links.new(vert_color.outputs['Color'], emission.inputs['Color'])

    # Assign to mesh
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    report.append(f"[Material] Created debug material: {mat_name}")


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_mask_output(obj, output_name="Mask", report=None):
    """
    Verify mask layer was created correctly.

    Args:
        obj: Blender mesh object
        output_name: Name of mask color layer
        report: Optional list for logging

    Returns:
        bool: True if valid mask layer exists
    """
    if report is None:
        report = []

    mesh = obj.data
    mask_attr = mesh.color_attributes.get(output_name)
    if not mask_attr:
        report.append(f"[Verify] ERROR: Mask layer '{output_name}' not found!")
        return False

    report.append(f"\n[Verify] Checking mask layer '{output_name}':")
    channel_counts = {ch: 0 for ch in CHANNEL_COLORS.keys()}

    for loop_data in mask_attr.data:
        color = loop_data.color[:4]

        for ch_name, ch_color in CHANNEL_COLORS.items():
            if (abs(color[0] - ch_color[0]) < 0.1 and
                    abs(color[1] - ch_color[1]) < 0.1 and
                    abs(color[2] - ch_color[2]) < 0.1):
                channel_counts[ch_name] += 1
                break

    for channel, count in channel_counts.items():
        if count > 0:
            report.append(f"  {channel}: {count} loops")

    return True


# ============================================================================
# HIGH-LEVEL FUNCTIONS
# ============================================================================

def auto_mask(obj, num_clusters=4, face_based=True, face_method="DOMINANT",
              channel_mapping=None, output_name="Mask", delete_source=False,
              create_debug_mat=True, input_layer="Color", report=None):
    """
    Automatic mask generation using k-means clustering.

    Args:
        obj: Blender mesh object
        num_clusters: Number of color clusters to find
        face_based: Use face-based clustering (contiguous regions)
        face_method: "DOMINANT", "AVERAGE", or "CENTER"
        channel_mapping: Dict {"BASE": rank, "PRIMARY": rank, ...}
                        Rank -1 means skip. None uses defaults.
        output_name: Output color layer name
        delete_source: Delete source color layer after
        create_debug_mat: Create visualization material
        input_layer: Source color layer name
        report: Optional list for logging

    Returns:
        bool: True on success
    """
    if report is None:
        report = []

    if channel_mapping is None:
        channel_mapping = {
            "BASE": 0,
            "PRIMARY": 1,
            "SECONDARY": 2,
            "ACCENT": 3,
            "EMISSIVE": -1,
        }

    # Extract colors
    colors_by_loop, colors_list = extract_vertex_colors(obj, layer_name=input_layer, report=report)

    if face_based:
        report.append(f"\n[Mode] Face-based clustering (method: {face_method})")

        face_colors = extract_face_colors(obj, colors_by_loop, method=face_method, report=report)
        clusters, face_assignments = cluster_face_colors(
            face_colors, k=num_clusters, max_iterations=20, report=report
        )
        create_face_mask_vertex_colors(
            obj, face_assignments, channel_mapping, output_name=output_name, report=report
        )
    else:
        report.append("\n[Mode] Per-vertex clustering")

        clusters = kmeans_cluster_colors(colors_list, k=num_clusters, max_iterations=20, report=report)
        loop_assignments = assign_clusters_to_channels(clusters, channel_mapping, report=report)
        create_mask_vertex_colors(obj, loop_assignments, output_name=output_name, report=report)

    # Verify
    verify_mask_output(obj, output_name=output_name, report=report)

    # Debug material
    if create_debug_mat:
        create_mask_debug_material(obj, output_name=output_name, report=report)

    # Delete source
    if delete_source and input_layer != output_name:
        mesh = obj.data
        source_attr = mesh.color_attributes.get(input_layer)
        if source_attr:
            mesh.color_attributes.remove(source_attr)
            report.append(f"\n[Cleanup] Deleted source layer '{input_layer}'")

    return True
