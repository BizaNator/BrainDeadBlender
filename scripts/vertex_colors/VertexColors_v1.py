"""
VertexColors V1 - Bake Texture to Vertex Colors

Two modes for baking colors to vertices:

Mode A: Single Texture + UVs
    Samples a texture at each vertex's UV coordinates.

Mode B: Multi-Image Projection
    Projects 4 reference images (front/back/left/right) onto mesh
    with weighted blending based on vertex normals.

Perfect for low-poly stylized characters where you want color without textures.

WORKFLOW:
    1. Load your images into Blender:
       - File > Import > Images as Planes, OR
       - Drag-drop images into Blender, OR
       - Image Editor > Open

    2. Run print_loaded_images() to see available image names

    3. Call project_images() with image names

Usage:
    from VertexColors_v1 import main, project_images, print_loaded_images

    # See what images are loaded:
    print_loaded_images()

    # Mode A: Single texture (by name or path)
    main(image_path="texture.png")       # Blender image name
    main(image_path="C:/path/tex.png")   # File path
    main()                               # Auto-detect from material

    # Mode B: Multi-image projection (use Blender image names!)
    project_images(
        front="front",      # or "front.png"
        back="back",
        left="left",
        right="right"
    )
"""

import bpy
import bmesh
import os
from datetime import datetime
from mathutils import Color

LOG_TEXT_NAME = "VertexColors_V1_Log.txt"

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

# --- MODE SELECTION ---
# "TEXTURE"    = Sample from single texture using UVs
# "PROJECTION" = Project 4 images (front/back/left/right) onto mesh
MODE = "PROJECTION"

# --- FOR TEXTURE MODE ---
# Image name in Blender, or full file path, or None to auto-detect from material
TEXTURE_IMAGE = None

# --- FOR PROJECTION MODE ---
# Image names as loaded in Blender (use print_loaded_images() to see names)
IMAGE_FRONT = "front.png"
IMAGE_BACK = "back.png"
IMAGE_LEFT = "left.png"
IMAGE_RIGHT = "right.png"

# --- GENERAL SETTINGS ---
# Target collection (output from Pipeline_v31)
TARGET_COLLECTION = "Export"

# Vertex color layer name
VERTEX_COLOR_NAME = "Col"  # Standard name Unreal recognizes

# Sampling settings
SAMPLE_METHOD = "NEAREST"  # "NEAREST" or "BILINEAR"

# Default color when UV is missing or out of bounds
DEFAULT_COLOR = (0.5, 0.5, 0.5, 1.0)  # Gray with full alpha

# --- PROJECTION SETTINGS ---
# Use alpha channel to ignore background (transparent pixels)
USE_ALPHA_MASK = True

# Minimum alpha to consider a pixel valid (0-1)
ALPHA_THRESHOLD = 0.1

# Detect and ignore background color (for images without alpha)
# Set to None to disable, or (R, G, B) to specify background color
BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # White background - set to None if images have alpha

# How close a color must be to background to be ignored (0-1, higher = more lenient)
BACKGROUND_TOLERANCE = 0.1

# Progress update frequency (print every N loops)
PROGRESS_UPDATE_FREQUENCY = 1000

# Flip V coordinate (try True if colors look wrong/upside-down)
FLIP_V = False

# --- HARD FACE COLORS ---
# When True: Each face gets ONE solid color from the best view (no blending)
# When False: Vertices blend colors from multiple views (softer, but can look noisy)
HARD_FACE_COLORS = True

# Debug: print sample values for first few vertices
DEBUG_SAMPLES = True
DEBUG_SAMPLE_COUNT = 5

# Verify vertex colors after projection (prints color values to confirm)
VERIFY_COLORS = True

# Create a simple debug material that ONLY shows vertex colors
# (replaces segment materials temporarily for debugging)
CREATE_DEBUG_MATERIAL = True


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


# ============================================================================
# IMAGE LOADING & SAMPLING
# ============================================================================

def load_image(image_ref: str, report: list):
    """
    Load an image from file path OR find by Blender image name.

    Args:
        image_ref: Either a file path OR the name of an image already loaded in Blender
        report: list for logging

    Examples:
        load_image("C:/textures/front.png", report)  # File path
        load_image("front.png", report)              # Blender image name
    """
    # First, check if it's a Blender image name (already loaded)
    img = bpy.data.images.get(image_ref)
    if img:
        report.append(f"[Image] Found in Blender: {image_ref}")
        return img

    # Check if it looks like a file path (has slashes or backslashes)
    if '/' in image_ref or '\\' in image_ref:
        # It's a file path
        if not os.path.exists(image_ref):
            raise RuntimeError(f"Image file not found: {image_ref}")

        img = bpy.data.images.load(image_ref)
        report.append(f"[Image] Loaded from file: {image_ref}")
        return img

    # Try to find by partial name match (without extension)
    base_name = os.path.splitext(image_ref)[0].lower()
    for img in bpy.data.images:
        img_base = os.path.splitext(img.name)[0].lower()
        if img_base == base_name:
            report.append(f"[Image] Found by name: {img.name}")
            return img

    raise RuntimeError(f"Image not found: '{image_ref}'\n"
                       f"Either load the image into Blender first, or provide a full file path.\n"
                       f"Loaded images: {list_loaded_images()}")


def list_loaded_images() -> list:
    """Return list of image names currently loaded in Blender."""
    return [img.name for img in bpy.data.images if not img.name.startswith('.')]


def print_loaded_images(report: list = None):
    """Print all images currently loaded in Blender. Useful for finding image names."""
    lines = []
    lines.append("\n=== Loaded Images in Blender ===")
    for img in bpy.data.images:
        if img.name.startswith('.'):
            continue
        size = f"{img.size[0]}x{img.size[1]}" if img.size[0] > 0 else "no data"
        source = img.filepath if img.filepath else "packed/generated"
        lines.append(f"  '{img.name}' ({size}) - {source}")
    lines.append("================================\n")

    output = "\n".join(lines)
    print(output)

    # Also add to report/log if provided
    if report is not None:
        report.extend(lines)


def get_material_texture(mesh_obj, report: list):
    """Try to find a texture from the mesh's materials."""
    for mat in mesh_obj.data.materials:
        if not mat or not mat.use_nodes:
            continue

        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                report.append(f"[Image] Found texture in material '{mat.name}': {node.image.name}")
                return node.image

    return None


# Global cache for pixel data (much faster than accessing img.pixels repeatedly)
_pixel_cache = {}


def cache_image_pixels(img):
    """Cache image pixels for fast access. Call before sampling."""
    if img.name not in _pixel_cache:
        print(f"[Cache] Caching pixels for '{img.name}' ({img.size[0]}x{img.size[1]})...")
        # Convert to tuple for faster access
        _pixel_cache[img.name] = {
            'pixels': tuple(img.pixels[:]),  # Copy to tuple
            'width': img.size[0],
            'height': img.size[1]
        }
        print(f"[Cache] Done caching '{img.name}'")
    return _pixel_cache[img.name]


def clear_pixel_cache():
    """Clear the pixel cache."""
    global _pixel_cache
    _pixel_cache = {}


def sample_image_cached(img_name: str, u: float, v: float, method: str = "NEAREST") -> tuple:
    """Sample from cached pixel data (much faster)."""
    import math

    if img_name not in _pixel_cache:
        return DEFAULT_COLOR

    cache = _pixel_cache[img_name]
    pixels = cache['pixels']
    width = cache['width']
    height = cache['height']

    if width == 0 or height == 0:
        return DEFAULT_COLOR

    # Handle NaN, infinity, or invalid UV values
    if math.isnan(u) or math.isnan(v) or math.isinf(u) or math.isinf(v):
        return DEFAULT_COLOR

    # Clamp UVs to 0-1 range
    u = u % 1.0
    v = v % 1.0
    if u < 0:
        u += 1.0
    if v < 0:
        v += 1.0

    # Nearest neighbor sampling
    x = int(u * (width - 1))
    y = int(v * (height - 1))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    idx = (y * width + x) * 4
    if idx + 3 < len(pixels):
        return (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])

    return DEFAULT_COLOR


def sample_image(img, u: float, v: float, method: str = "NEAREST") -> tuple:
    """
    Sample color from image at UV coordinates.

    Args:
        img: Blender image
        u, v: UV coordinates (0-1 range)
        method: "NEAREST" or "BILINEAR"

    Returns:
        (r, g, b, a) tuple with values 0-1
    """
    import math

    if not img or not img.pixels:
        return DEFAULT_COLOR

    width = img.size[0]
    height = img.size[1]

    if width == 0 or height == 0:
        return DEFAULT_COLOR

    # Handle NaN, infinity, or invalid UV values
    if math.isnan(u) or math.isnan(v) or math.isinf(u) or math.isinf(v):
        return DEFAULT_COLOR

    # Clamp UVs to 0-1 range (handle tiling/wrapping)
    u = u % 1.0
    v = v % 1.0

    # Handle negative modulo result
    if u < 0:
        u += 1.0
    if v < 0:
        v += 1.0

    if method == "NEAREST":
        # Nearest neighbor sampling
        x = int(u * (width - 1))
        y = int(v * (height - 1))

        # Clamp to valid range
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        # Image pixels are stored as flat array: [r,g,b,a, r,g,b,a, ...]
        # Row-major, bottom to top
        idx = (y * width + x) * 4

        pixels = img.pixels
        if idx + 3 < len(pixels):
            return (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])
        else:
            return DEFAULT_COLOR

    elif method == "BILINEAR":
        # Bilinear interpolation for smoother sampling
        fx = u * (width - 1)
        fy = v * (height - 1)

        x0 = int(fx)
        y0 = int(fy)
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)

        # Fractional parts
        dx = fx - x0
        dy = fy - y0

        pixels = img.pixels

        def get_pixel(x, y):
            idx = (y * width + x) * 4
            if idx + 3 < len(pixels):
                return (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])
            return DEFAULT_COLOR

        # Sample 4 corners
        c00 = get_pixel(x0, y0)
        c10 = get_pixel(x1, y0)
        c01 = get_pixel(x0, y1)
        c11 = get_pixel(x1, y1)

        # Interpolate
        def lerp(a, b, t):
            return a + (b - a) * t

        result = []
        for i in range(4):
            top = lerp(c00[i], c10[i], dx)
            bottom = lerp(c01[i], c11[i], dx)
            result.append(lerp(top, bottom, dy))

        return tuple(result)

    return DEFAULT_COLOR


# ============================================================================
# VERTEX COLOR BAKING
# ============================================================================

def bake_texture_to_vertex_colors(mesh_obj, img, report: list):
    """
    Bake texture colors to vertex colors.

    Samples the texture at each vertex's UV position and assigns
    the color to the vertex color layer.
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append(f"\n[Bake] Processing mesh: {mesh_obj.name}")
    report.append(f"[Bake] Vertices: {len(mesh.vertices)}")
    report.append(f"[Bake] Faces: {len(mesh.polygons)}")

    # Check for UV layer
    if not mesh.uv_layers:
        raise RuntimeError(f"Mesh '{mesh_obj.name}' has no UV layer!")

    uv_layer = mesh.uv_layers.active
    report.append(f"[Bake] Using UV layer: {uv_layer.name}")

    # Create or get vertex color layer
    # In Blender 3.2+, use color_attributes instead of vertex_colors
    if hasattr(mesh, 'color_attributes'):
        # Blender 3.2+
        color_attr = mesh.color_attributes.get(VERTEX_COLOR_NAME)
        if color_attr:
            mesh.color_attributes.remove(color_attr)

        # Create new - use CORNER domain for per-face-corner colors
        color_attr = mesh.color_attributes.new(
            name=VERTEX_COLOR_NAME,
            type='BYTE_COLOR',  # 8-bit per channel, good for export
            domain='CORNER'     # Per face-corner (like UVs)
        )
        report.append(f"[Bake] Created color attribute: {VERTEX_COLOR_NAME}")
    else:
        # Older Blender
        if VERTEX_COLOR_NAME in mesh.vertex_colors:
            mesh.vertex_colors.remove(mesh.vertex_colors[VERTEX_COLOR_NAME])

        color_layer = mesh.vertex_colors.new(name=VERTEX_COLOR_NAME)
        report.append(f"[Bake] Created vertex color layer: {VERTEX_COLOR_NAME}")

    # Sample texture and assign colors
    # Colors are stored per-loop (face corner), matching UV layout
    report.append(f"[Bake] Sampling texture ({img.size[0]}x{img.size[1]})...")
    print(f"[Bake] Sampling texture ({img.size[0]}x{img.size[1]})...")

    # CRITICAL: Cache image pixels before loop (prevents freeze/crash)
    cache_image_pixels(img)
    report.append(f"[Bake] Cached image pixels for fast access")

    # Get direct reference to color data
    if hasattr(mesh, 'color_attributes'):
        color_data = mesh.color_attributes[VERTEX_COLOR_NAME].data
    else:
        color_data = mesh.vertex_colors[VERTEX_COLOR_NAME].data

    import math
    sampled_count = 0
    invalid_uv_count = 0
    total_loops = len(mesh.loops)

    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            # Get UV for this loop
            uv = uv_layer.data[loop_idx].uv

            # Track invalid UVs
            if math.isnan(uv.x) or math.isnan(uv.y):
                invalid_uv_count += 1

            # Sample texture using CACHED pixels (fast!)
            color = sample_image_cached(img.name, uv.x, uv.y, SAMPLE_METHOD)

            # Assign color directly to color data reference
            color_data[loop_idx].color = color

            sampled_count += 1

            # Progress reporting
            if sampled_count % PROGRESS_UPDATE_FREQUENCY == 0:
                pct = (sampled_count / total_loops) * 100
                print(f"[Bake] Progress: {sampled_count}/{total_loops} ({pct:.1f}%)")

    report.append(f"[Bake] Sampled {sampled_count} vertex colors")
    if invalid_uv_count > 0:
        report.append(f"[Bake] WARNING: {invalid_uv_count} vertices had invalid UVs (NaN) - used default color")

    # Force mesh update to ensure changes are saved
    mesh.update()

    # Set as active for rendering/export
    if hasattr(mesh, 'color_attributes'):
        mesh.color_attributes.active_color = mesh.color_attributes[VERTEX_COLOR_NAME]
        mesh.color_attributes.render_color_index = mesh.color_attributes.find(VERTEX_COLOR_NAME)

    depsgraph_update()
    report.append("[Bake] Done.")
    print("[Bake] Complete!")


def remove_vertex_colors(mesh_obj, report: list):
    """Remove all vertex color layers from mesh."""
    ensure_object_mode()
    mesh = mesh_obj.data

    if hasattr(mesh, 'color_attributes'):
        names = [attr.name for attr in mesh.color_attributes]
        for name in names:
            mesh.color_attributes.remove(mesh.color_attributes[name])
        report.append(f"[Remove] Removed {len(names)} color attributes")
    else:
        names = [vc.name for vc in mesh.vertex_colors]
        for name in names:
            mesh.vertex_colors.remove(mesh.vertex_colors[name])
        report.append(f"[Remove] Removed {len(names)} vertex color layers")


# ============================================================================
# MULTI-IMAGE PROJECTION
# ============================================================================

# View directions (where camera is looking FROM)
VIEW_DIRECTIONS = {
    "front": (0, -1, 0),   # Camera at +Y looking toward -Y (at character's face)
    "back":  (0, 1, 0),    # Camera at -Y looking toward +Y (at character's back)
    "left":  (1, 0, 0),    # Camera at -X looking toward +X (at left side)
    "right": (-1, 0, 0),   # Camera at +X looking toward -X (at right side)
}


def get_mesh_bounds(mesh_obj):
    """Get world-space bounding box of mesh."""
    from mathutils import Vector

    ensure_object_mode()

    # Get all vertex positions in world space
    mesh = mesh_obj.data
    world_verts = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]

    if not world_verts:
        return None

    # Calculate bounds
    min_co = Vector((
        min(v.x for v in world_verts),
        min(v.y for v in world_verts),
        min(v.z for v in world_verts)
    ))
    max_co = Vector((
        max(v.x for v in world_verts),
        max(v.y for v in world_verts),
        max(v.z for v in world_verts)
    ))

    return {
        'min': min_co,
        'max': max_co,
        'size': max_co - min_co,
        'center': (min_co + max_co) / 2
    }


def calculate_view_weights(normal, view_directions: dict) -> dict:
    """
    Calculate blend weights for each view based on vertex normal.

    Returns dict of view_name -> weight (0-1, normalized to sum to 1)
    """
    from mathutils import Vector

    weights = {}
    total = 0.0

    for view_name, view_dir in view_directions.items():
        # View direction is where camera looks FROM, so we need the opposite
        # to compare with the vertex normal (which points outward)
        camera_dir = Vector(view_dir)

        # Dot product: how much does vertex face this camera?
        # Positive = facing camera, negative = facing away
        dot = normal.dot(camera_dir)

        # Only use positive (facing camera), with power for sharper falloff
        weight = max(0.0, dot) ** 2  # Square for sharper transitions

        weights[view_name] = weight
        total += weight

    # Normalize weights to sum to 1
    if total > 0.001:
        for key in weights:
            weights[key] /= total
    else:
        # Fallback: equal weights if normal is straight up/down
        for key in weights:
            weights[key] = 0.25

    return weights


def project_vertex_to_image_uv(vertex_world, view_name: str, bounds: dict) -> tuple:
    """
    Project a 3D world position to 2D image UV for a specific view.

    Returns (u, v) in 0-1 range.
    """
    # Get relative position within bounds
    rel = vertex_world - bounds['min']
    size = bounds['size']

    # Avoid division by zero
    sx = size.x if size.x > 0.001 else 1.0
    sy = size.y if size.y > 0.001 else 1.0
    sz = size.z if size.z > 0.001 else 1.0

    if view_name == "front":
        # Looking at -Y: X maps to U, Z maps to V
        u = rel.x / sx
        v = rel.z / sz
    elif view_name == "back":
        # Looking at +Y: X maps to U (flipped), Z maps to V
        u = 1.0 - (rel.x / sx)
        v = rel.z / sz
    elif view_name == "left":
        # Looking at +X: Y maps to U (flipped), Z maps to V
        u = 1.0 - (rel.y / sy)
        v = rel.z / sz
    elif view_name == "right":
        # Looking at -X: Y maps to U, Z maps to V
        u = rel.y / sy
        v = rel.z / sz
    else:
        u, v = 0.5, 0.5

    # Flip V if configured
    if FLIP_V:
        v = 1.0 - v

    return (u, v)


def create_vertex_color_layer(mesh_obj, report: list):
    """Create or reset vertex color layer, return reference."""
    mesh = mesh_obj.data

    if hasattr(mesh, 'color_attributes'):
        # Blender 3.2+
        color_attr = mesh.color_attributes.get(VERTEX_COLOR_NAME)
        if color_attr:
            mesh.color_attributes.remove(color_attr)

        color_attr = mesh.color_attributes.new(
            name=VERTEX_COLOR_NAME,
            type='BYTE_COLOR',
            domain='CORNER'
        )
        report.append(f"[Colors] Created color attribute: {VERTEX_COLOR_NAME}")
        return color_attr
    else:
        # Older Blender
        if VERTEX_COLOR_NAME in mesh.vertex_colors:
            mesh.vertex_colors.remove(mesh.vertex_colors[VERTEX_COLOR_NAME])

        color_layer = mesh.vertex_colors.new(name=VERTEX_COLOR_NAME)
        report.append(f"[Colors] Created vertex color layer: {VERTEX_COLOR_NAME}")
        return color_layer


def project_images_to_vertex_colors(mesh_obj, images: dict, report: list):
    """
    Project multiple images onto mesh using weighted blending.

    Args:
        mesh_obj: Blender mesh object
        images: dict of view_name -> Blender image (front, back, left, right)
        report: list for logging
    """
    from mathutils import Vector

    ensure_object_mode()
    mesh = mesh_obj.data

    report.append(f"\n[Project] Processing mesh: {mesh_obj.name}")
    report.append(f"[Project] Vertices: {len(mesh.vertices)}")
    report.append(f"[Project] Faces: {len(mesh.polygons)}")
    report.append(f"[Project] Loops (corners): {len(mesh.loops)}")
    print(f"[Project] Mesh: {mesh_obj.name}")
    print(f"[Project] Vertices: {len(mesh.vertices)}, Faces: {len(mesh.polygons)}, Loops: {len(mesh.loops)}")

    # Get mesh bounds for projection
    bounds = get_mesh_bounds(mesh_obj)
    if not bounds:
        raise RuntimeError("Could not calculate mesh bounds!")

    report.append(f"[Project] Bounds: min={bounds['min']}, max={bounds['max']}")
    report.append(f"[Project] Size: {bounds['size']}")

    # Log which images we have and cache their pixels
    report.append("\n[Project] Caching image pixels (this may take a moment)...")
    print("[Project] Caching image pixels...")
    clear_pixel_cache()  # Clear any old cache

    for view_name, img in images.items():
        if img:
            report.append(f"[Project] {view_name}: {img.name} ({img.size[0]}x{img.size[1]})")
            cache_image_pixels(img)  # Cache pixels for fast access
        else:
            report.append(f"[Project] {view_name}: NOT PROVIDED")

    report.append("[Project] Image caching complete.")
    print("[Project] Image caching complete.")

    # Create vertex color layer and get direct reference
    color_layer = create_vertex_color_layer(mesh_obj, report)

    # Get direct reference to color data for writing
    if hasattr(mesh, 'color_attributes'):
        color_attr = mesh.color_attributes.get(VERTEX_COLOR_NAME)
        if not color_attr:
            raise RuntimeError(f"Failed to create color attribute '{VERTEX_COLOR_NAME}'")
        color_data = color_attr.data
        report.append(f"[Project] Color attribute: {color_attr.name}, domain={color_attr.domain}, type={color_attr.data_type}")
        print(f"[Project] Color data length: {len(color_data)}, loops: {len(mesh.loops)}")
    else:
        color_data = mesh.vertex_colors[VERTEX_COLOR_NAME].data
        report.append(f"[Project] Using vertex_colors (older Blender)")

    # Pre-calculate normals (Blender 4.0+ does this automatically)
    if hasattr(mesh, 'calc_normals_split'):
        mesh.calc_normals_split()  # Blender 3.x
    # In Blender 4.0+, normals are auto-calculated when accessed

    # Process based on mode
    total_loops = len(mesh.loops)
    total_faces = len(mesh.polygons)

    if HARD_FACE_COLORS:
        report.append(f"[Project] HARD FACE MODE: Projecting {total_faces} faces (solid colors per face)...")
        print(f"[Project] HARD FACE MODE: Processing {total_faces} faces...")
    else:
        report.append(f"[Project] BLEND MODE: Projecting {total_loops} vertices (blended colors)...")
        print(f"[Project] BLEND MODE: Processing {total_loops} vertices...")

    processed = 0
    skipped_alpha = 0
    skipped_background = 0
    used_samples = 0

    # Get corner normals (Blender 4.0+ uses corner_normals, older uses loops[].normal)
    use_corner_normals = hasattr(mesh, 'corner_normals') and len(mesh.corner_normals) > 0

    for poly in mesh.polygons:

        if HARD_FACE_COLORS:
            # === HARD FACE COLORS MODE ===
            # Use face normal and center to get ONE color for entire face

            # Get face center in world space
            face_center_local = poly.center
            face_center_world = mesh_obj.matrix_world @ face_center_local

            # Get face normal in world space
            face_normal_local = Vector(poly.normal)
            face_normal_world = (mesh_obj.matrix_world.to_3x3() @ face_normal_local).normalized()

            # Find BEST view for this face (highest weight)
            weights = calculate_view_weights(face_normal_world, VIEW_DIRECTIONS)
            best_view = max(weights.items(), key=lambda x: x[1])
            best_view_name = best_view[0]
            best_weight = best_view[1]

            # Sample from best view only
            img = images.get(best_view_name)
            final_color = [0.5, 0.5, 0.5, 1.0]  # Default gray

            if img and best_weight > 0.001:
                u, v = project_vertex_to_image_uv(face_center_world, best_view_name, bounds)
                color = sample_image_cached(img.name, u, v, SAMPLE_METHOD)

                # Check alpha
                valid_sample = True
                if USE_ALPHA_MASK:
                    alpha = color[3] if len(color) > 3 else 1.0
                    if alpha < ALPHA_THRESHOLD:
                        skipped_alpha += 1
                        valid_sample = False

                # Check background
                if valid_sample and BACKGROUND_COLOR is not None:
                    bg = BACKGROUND_COLOR
                    diff = abs(color[0] - bg[0]) + abs(color[1] - bg[1]) + abs(color[2] - bg[2])
                    if diff < BACKGROUND_TOLERANCE * 3:
                        skipped_background += 1
                        valid_sample = False

                if valid_sample:
                    final_color = [color[0], color[1], color[2], 1.0]
                    used_samples += 1

            # Apply SAME color to ALL loops in this face
            rgba = (final_color[0], final_color[1], final_color[2], 1.0)
            for loop_idx in poly.loop_indices:
                color_data[loop_idx].color = rgba
                processed += 1

            # Debug
            if DEBUG_SAMPLES and poly.index < DEBUG_SAMPLE_COUNT:
                print(f"[Debug] Face {poly.index}: best_view={best_view_name}, color=RGB({rgba[0]:.3f}, {rgba[1]:.3f}, {rgba[2]:.3f})")

        else:
            # === BLEND MODE (original behavior) ===
            for loop_idx in poly.loop_indices:
                loop = mesh.loops[loop_idx]
                vert = mesh.vertices[loop.vertex_index]

                # Get world position
                vert_world = mesh_obj.matrix_world @ vert.co

                # Get normal (different access in Blender 4.0+)
                if use_corner_normals:
                    normal_local = Vector(mesh.corner_normals[loop_idx].vector)
                else:
                    try:
                        normal_local = Vector(loop.normal)
                    except:
                        normal_local = Vector(vert.normal)

                normal_world = (mesh_obj.matrix_world.to_3x3() @ normal_local).normalized()

                # Calculate blend weights based on normal
                weights = calculate_view_weights(normal_world, VIEW_DIRECTIONS)

                # Sample and blend colors from each view
                final_color = [0.0, 0.0, 0.0, 1.0]
                total_weight = 0.0

                for view_name, weight in weights.items():
                    if weight < 0.001:
                        continue

                    img = images.get(view_name)
                    if not img:
                        continue

                    u, v = project_vertex_to_image_uv(vert_world, view_name, bounds)
                    color = sample_image_cached(img.name, u, v, SAMPLE_METHOD)

                    if USE_ALPHA_MASK:
                        alpha = color[3] if len(color) > 3 else 1.0
                        if alpha < ALPHA_THRESHOLD:
                            skipped_alpha += 1
                            continue
                        effective_weight = weight * alpha
                    else:
                        effective_weight = weight

                    if BACKGROUND_COLOR is not None:
                        bg = BACKGROUND_COLOR
                        diff = abs(color[0] - bg[0]) + abs(color[1] - bg[1]) + abs(color[2] - bg[2])
                        if diff < BACKGROUND_TOLERANCE * 3:
                            skipped_background += 1
                            continue

                    final_color[0] += color[0] * effective_weight
                    final_color[1] += color[1] * effective_weight
                    final_color[2] += color[2] * effective_weight
                    total_weight += effective_weight
                    used_samples += 1

                if total_weight > 0.001:
                    final_color[0] /= total_weight
                    final_color[1] /= total_weight
                    final_color[2] /= total_weight
                else:
                    final_color = [1.0, 0.0, 1.0, 1.0]  # Magenta = no valid samples

                # Debug output
                if DEBUG_SAMPLES and processed < DEBUG_SAMPLE_COUNT:
                    print(f"[Debug] Vertex {processed}:")
                    print(f"  Position: ({vert_world.x:.3f}, {vert_world.y:.3f}, {vert_world.z:.3f})")
                    print(f"  Final color: RGB({final_color[0]:.3f}, {final_color[1]:.3f}, {final_color[2]:.3f})")

                # Assign color
                rgba = (final_color[0], final_color[1], final_color[2], final_color[3] if len(final_color) > 3 else 1.0)
                color_data[loop_idx].color = rgba

                # Debug: verify first few assignments
                if DEBUG_SAMPLES and processed < 3:
                    stored = color_data[loop_idx].color
                    print(f"  Assigned: RGBA{rgba}")
                    print(f"  Stored color [{loop_idx}]: RGBA({stored[0]:.3f}, {stored[1]:.3f}, {stored[2]:.3f}, {stored[3]:.3f})")

                processed += 1

                # Progress update
                if processed % PROGRESS_UPDATE_FREQUENCY == 0:
                    pct = (processed / total_loops) * 100
                    print(f"[Project] Progress: {processed}/{total_loops} ({pct:.1f}%)")

    # Force mesh update to ensure changes are saved
    mesh.update()

    # Set as active
    if hasattr(mesh, 'color_attributes'):
        mesh.color_attributes.active_color = mesh.color_attributes[VERTEX_COLOR_NAME]
        mesh.color_attributes.render_color_index = mesh.color_attributes.find(VERTEX_COLOR_NAME)
        report.append(f"[Project] Set active color attribute to '{VERTEX_COLOR_NAME}'")

    depsgraph_update()

    # Final verification - read back a few colors
    report.append("[Project] Verifying stored colors...")
    print("[Project] Verifying stored colors...")
    for i in range(min(5, len(color_data))):
        c = color_data[i].color
        print(f"  Final stored [{i}]: RGB({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    report.append(f"[Project] Processed {processed} vertex colors")
    report.append(f"[Project] Samples used: {used_samples}")
    report.append(f"[Project] Skipped (alpha): {skipped_alpha}")
    report.append(f"[Project] Skipped (background): {skipped_background}")

    print(f"[Project] Samples used: {used_samples}")
    print(f"[Project] Skipped (alpha): {skipped_alpha}")
    print(f"[Project] Skipped (background): {skipped_background}")

    if used_samples == 0:
        print("[WARNING] No samples were used! All pixels were skipped.")
        print("[WARNING] Try setting BACKGROUND_COLOR = None or USE_ALPHA_MASK = False")
        report.append("[WARNING] No samples used - check background detection settings!")

    report.append("[Project] Done.")


def verify_vertex_colors(mesh_obj, report: list):
    """
    Verify vertex colors exist and contain non-default data.
    Also prints sample values for debugging.
    """
    mesh = mesh_obj.data

    report.append("\n[Verify] Checking vertex colors...")
    print("\n[Verify] Checking vertex colors...")

    # Check if color attribute exists
    if hasattr(mesh, 'color_attributes'):
        color_attr = mesh.color_attributes.get(VERTEX_COLOR_NAME)
        if not color_attr:
            report.append(f"[Verify] ERROR: Color attribute '{VERTEX_COLOR_NAME}' not found!")
            print(f"[Verify] ERROR: Color attribute '{VERTEX_COLOR_NAME}' not found!")
            return False

        report.append(f"[Verify] Found color attribute: {color_attr.name}")
        report.append(f"[Verify] Domain: {color_attr.domain}, Type: {color_attr.data_type}")
        print(f"[Verify] Found: {color_attr.name}, Domain: {color_attr.domain}, Type: {color_attr.data_type}")

        # Sample first 10 colors
        non_grey = 0
        total = min(10, len(color_attr.data))
        report.append(f"[Verify] Sampling first {total} vertex colors:")
        print(f"[Verify] Sampling first {total} vertex colors:")

        for i in range(total):
            c = color_attr.data[i].color
            r, g, b = c[0], c[1], c[2]
            # Check if not grey (r≈g≈b≈0.5)
            is_grey = abs(r - 0.5) < 0.1 and abs(g - 0.5) < 0.1 and abs(b - 0.5) < 0.1
            if not is_grey:
                non_grey += 1
            report.append(f"  [{i}] RGB({r:.3f}, {g:.3f}, {b:.3f}) {'<-- colored' if not is_grey else '(grey)'}")
            print(f"  [{i}] RGB({r:.3f}, {g:.3f}, {b:.3f}) {'<-- colored' if not is_grey else '(grey)'}")

        if non_grey > 0:
            report.append(f"[Verify] SUCCESS: Found {non_grey}/{total} non-grey vertex colors!")
            print(f"[Verify] SUCCESS: Found {non_grey}/{total} non-grey vertex colors!")
            return True
        else:
            report.append("[Verify] WARNING: All sampled colors are grey - projection may have failed")
            print("[Verify] WARNING: All sampled colors are grey - projection may have failed")
            return False
    else:
        # Older Blender
        if VERTEX_COLOR_NAME not in mesh.vertex_colors:
            report.append(f"[Verify] ERROR: Vertex color layer '{VERTEX_COLOR_NAME}' not found!")
            return False
        report.append(f"[Verify] Found vertex color layer: {VERTEX_COLOR_NAME}")
        return True


def setup_vertex_color_material(mesh_obj, report: list):
    """
    Create a simple material that ONLY displays vertex colors.
    Uses Emission shader to show colors without lighting interference.
    """
    mat_name = "M_VertexColors_Debug"

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

    # Output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Use Emission shader - shows color directly without lighting
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (200, 0)
    emission.inputs['Strength'].default_value = 1.0
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # Also try Principled BSDF as backup (connected to same output via Mix)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (200, -200)

    # Try to create vertex color node
    report.append("[Material] Creating vertex color node...")
    print("[Material] Creating vertex color node...")

    vert_color = None
    node_type_used = "none"

    # Method 1: Color Attribute node (Blender 4.0+)
    try:
        vert_color = nodes.new('ShaderNodeColorAttribute')
        # Check if it was actually created as the right type
        if vert_color.bl_idname == 'ShaderNodeColorAttribute':
            vert_color.layer_name = VERTEX_COLOR_NAME
            node_type_used = "ShaderNodeColorAttribute"
            report.append(f"[Material] SUCCESS: Created ShaderNodeColorAttribute, layer_name='{VERTEX_COLOR_NAME}'")
            print(f"[Material] SUCCESS: ShaderNodeColorAttribute created")
        else:
            report.append(f"[Material] Node created but is {vert_color.bl_idname}, not ShaderNodeColorAttribute")
            nodes.remove(vert_color)
            vert_color = None
    except Exception as e:
        report.append(f"[Material] ShaderNodeColorAttribute failed: {e}")
        print(f"[Material] ShaderNodeColorAttribute failed: {e}")

    # Method 2: Vertex Color node (older Blender)
    if vert_color is None:
        try:
            vert_color = nodes.new('ShaderNodeVertexColor')
            if hasattr(vert_color, 'layer_name'):
                vert_color.layer_name = VERTEX_COLOR_NAME
                node_type_used = "ShaderNodeVertexColor"
                report.append(f"[Material] SUCCESS: Created ShaderNodeVertexColor, layer_name='{VERTEX_COLOR_NAME}'")
                print(f"[Material] SUCCESS: ShaderNodeVertexColor created")
            else:
                report.append(f"[Material] ShaderNodeVertexColor has no layer_name attr")
                nodes.remove(vert_color)
                vert_color = None
        except Exception as e:
            report.append(f"[Material] ShaderNodeVertexColor failed: {e}")
            print(f"[Material] ShaderNodeVertexColor failed: {e}")

    # Method 3: Attribute node (most compatible)
    if vert_color is None:
        try:
            vert_color = nodes.new('ShaderNodeAttribute')
            vert_color.attribute_name = VERTEX_COLOR_NAME
            if hasattr(vert_color, 'attribute_type'):
                vert_color.attribute_type = 'GEOMETRY'
            node_type_used = "ShaderNodeAttribute"
            report.append(f"[Material] SUCCESS: Created ShaderNodeAttribute, attribute_name='{VERTEX_COLOR_NAME}'")
            print(f"[Material] SUCCESS: ShaderNodeAttribute created")
        except Exception as e:
            report.append(f"[Material] ShaderNodeAttribute failed: {e}")
            print(f"[Material] ShaderNodeAttribute failed: {e}")

    if vert_color:
        vert_color.location = (-200, 0)

        # Log all outputs
        report.append(f"[Material] Node '{vert_color.bl_idname}' outputs:")
        print(f"[Material] Node outputs:")
        for i, out in enumerate(vert_color.outputs):
            report.append(f"  [{i}] '{out.name}' type={out.type}")
            print(f"  [{i}] '{out.name}' type={out.type}")

        # Find and connect color output
        color_out = None
        for out in vert_color.outputs:
            if out.type == 'RGBA' or out.name == 'Color':
                color_out = out
                break

        if color_out is None and len(vert_color.outputs) > 0:
            color_out = vert_color.outputs[0]

        if color_out:
            # Connect to emission (main display)
            links.new(color_out, emission.inputs['Color'])
            # Also connect to BSDF for comparison
            links.new(color_out, bsdf.inputs['Base Color'])
            report.append(f"[Material] Connected output '{color_out.name}' to Emission")
            print(f"[Material] Connected '{color_out.name}' to Emission shader")
        else:
            report.append("[Material] ERROR: No color output found on node!")
            print("[Material] ERROR: No color output found!")
    else:
        report.append("[Material] ERROR: Could not create any vertex color node!")
        print("[Material] ERROR: No vertex color node created!")

    # Clear existing materials and assign
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    report.append(f"[Material] Assigned '{mat_name}' using {node_type_used}")
    report.append("[Material] View in Material Preview mode (Z > Material Preview)")
    report.append("[Material] Using Emission shader - colors shown without lighting")
    print(f"[Material] Created and assigned '{mat_name}'")

    return mat


def project_images(front: str = None, back: str = None, left: str = None, right: str = None,
                   collection_name: str = None):
    """
    Project 4 reference images onto mesh with weighted blending.

    Args:
        front: Path to front view image
        back: Path to back view image
        left: Path to left view image
        right: Path to right view image
        collection_name: Override target collection name
    """
    report = []
    report.append("VertexColors V1 - Multi-Image Projection")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    # Load images
    images = {}
    image_paths = {"front": front, "back": back, "left": left, "right": right}

    for view_name, path in image_paths.items():
        if path:
            images[view_name] = load_image(path, report)
        else:
            images[view_name] = None

    # Check we have at least one image
    if not any(images.values()):
        raise RuntimeError("No images provided! Provide at least one of: front, back, left, right")

    # Project images
    project_images_to_vertex_colors(mesh_obj, images, report)

    report.append("\n" + "=" * 50)
    report.append("Multi-image projection complete.")
    report.append("\nFBX Export: Enable 'Vertex Colors' in export settings.")
    report.append("Unreal: Vertex colors available in material as 'Vertex Color' node.")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log: {LOG_TEXT_NAME} ---")


# ============================================================================
# MAIN
# ============================================================================

def main(image_path: str = None, collection_name: str = None, flip_v: bool = False):
    """
    Main entry point.

    Args:
        image_path: Path to texture image. If None, tries to find from material.
        collection_name: Override target collection name
        flip_v: Flip V coordinate (use if colors appear upside-down)
    """
    global SAMPLE_METHOD

    report = []
    report.append("VertexColors V1 - Texture to Vertex Color Baking")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    # Get image
    img = None

    if image_path:
        img = load_image(image_path, report)
    else:
        # Try to find from material
        img = get_material_texture(mesh_obj, report)

    if not img:
        raise RuntimeError("No image found! Provide image_path or ensure mesh has a textured material.")

    report.append(f"[Image] Size: {img.size[0]}x{img.size[1]}")
    report.append(f"[Image] Channels: {img.channels}")

    # Ensure pixels are loaded (for packed/external images)
    if not img.pixels:
        img.pixels  # Access to force load

    # Bake colors
    bake_texture_to_vertex_colors(mesh_obj, img, report)

    report.append("\n" + "=" * 50)
    report.append("VertexColors V1 complete.")
    report.append("\nFBX Export: Enable 'Vertex Colors' in export settings.")
    report.append("Unreal: Vertex colors available in material as 'Vertex Color' node.")

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log: {LOG_TEXT_NAME} ---")


def run():
    """
    Run the script based on configuration at top of file.
    Called automatically when script is executed.
    """
    report = []
    report.append("VertexColors V1")
    report.append(f"Mode: {MODE}")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Log loaded images
    print_loaded_images(report)

    # Find mesh for post-processing
    col = find_collection_ci(TARGET_COLLECTION)
    mesh_obj = None
    if col:
        meshes = mesh_objects(col)
        if meshes:
            mesh_obj = meshes[0]

    if MODE == "TEXTURE":
        report.append(f"\n[Mode] TEXTURE - sampling from UV coordinates")
        report.append(f"[Config] Image: {TEXTURE_IMAGE or 'auto-detect'}")
        main(image_path=TEXTURE_IMAGE)

    elif MODE == "PROJECTION":
        report.append(f"\n[Mode] PROJECTION - projecting 4 images onto mesh")
        report.append(f"[Config] Front: {IMAGE_FRONT}")
        report.append(f"[Config] Back: {IMAGE_BACK}")
        report.append(f"[Config] Left: {IMAGE_LEFT}")
        report.append(f"[Config] Right: {IMAGE_RIGHT}")
        project_images(
            front=IMAGE_FRONT,
            back=IMAGE_BACK,
            left=IMAGE_LEFT,
            right=IMAGE_RIGHT
        )

    else:
        raise RuntimeError(f"Unknown MODE: {MODE}. Use 'TEXTURE' or 'PROJECTION'")

    # Post-processing: verify and optionally create debug material
    if mesh_obj:
        if VERIFY_COLORS:
            verify_vertex_colors(mesh_obj, report)

        if CREATE_DEBUG_MATERIAL:
            report.append("\n[Debug] Creating vertex color debug material...")
            setup_vertex_color_material(mesh_obj, report)

    # Final instructions
    report.append("\n" + "=" * 50)
    report.append("TO VIEW VERTEX COLORS:")
    report.append("  1. Select mesh")
    report.append("  2. Press Z > Material Preview")
    report.append("  OR")
    report.append("  1. Select mesh")
    report.append("  2. Ctrl+Tab > Vertex Paint (to see raw colors)")
    report.append("=" * 50)

    log_text = "\n".join(report)
    log_to_text(log_text)
    print("\n".join(report[-15:]))  # Print last 15 lines


if __name__ == "__main__":
    run()
