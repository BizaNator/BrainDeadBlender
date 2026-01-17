"""
TextureProject V1 - Project Images to UV Map + Texture

Projects 4 reference images (front/back/left/right) onto a mesh,
creating a new UV map and baking to an actual texture image.

Use this BEFORE VertexColors to:
1. Preview what the projected texture looks like
2. Edit/refine the texture in an image editor
3. Then use the refined texture as input to VertexColors

WORKFLOW:
    1. Load reference images into Blender (drag-drop or File > Import)
    2. Run print_loaded_images() to see available image names
    3. Run project_to_texture() with image names
    4. Preview result in Material Preview mode
    5. (Optional) Edit the generated texture in Blender or external editor
    6. Use texture with VertexColors_v1.py for final vertex color bake

Usage:
    from TextureProject_v1 import project_to_texture, print_loaded_images

    # See loaded images:
    print_loaded_images()

    # Project to texture:
    project_to_texture(
        front="front",
        back="back",
        left="left",
        right="right"
    )
"""

import bpy
import bmesh
import os
from datetime import datetime
from mathutils import Vector

LOG_TEXT_NAME = "TextureProject_V1_Log.txt"

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- IMAGE NAMES (as loaded in Blender) ---
IMAGE_FRONT = "front.png"
IMAGE_BACK = "back.png"
IMAGE_LEFT = "left.png"
IMAGE_RIGHT = "right.png"

# --- TARGET ---
TARGET_COLLECTION = "Export"

# --- UV MAP ---
UV_MAP_NAME = "ProjectedUV"          # Name for the new UV map
REPLACE_EXISTING_UV = True            # Replace if UV map already exists

# --- OUTPUT TEXTURE ---
OUTPUT_TEXTURE_NAME = "ProjectedTexture"  # Name for output image
OUTPUT_RESOLUTION = 2048              # Output texture resolution (square)
OUTPUT_FORMAT = "PNG"                 # PNG, JPEG, TARGA, etc.

# --- PROJECTION MODE ---
# "BOX"       = Box projection (each face uses best view)
# "BEST_VIEW" = Each face projected from single best view
# "BLEND"     = Blend colors from multiple views (softer, may be blurry)
PROJECTION_MODE = "BOX"

# --- BACKGROUND HANDLING ---
# Background color to ignore (set to None to use all pixels)
BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # White
BACKGROUND_TOLERANCE = 0.1

# Use alpha channel to ignore transparent pixels
USE_ALPHA_MASK = True
ALPHA_THRESHOLD = 0.1

# --- FILL OPTIONS ---
# How to handle pixels with no valid projection
FILL_COLOR = (0.5, 0.5, 0.5, 1.0)    # Gray for unprojected areas
FILL_FROM_NEAREST = True              # Try to fill from nearest valid pixel

# --- UV MARGIN ---
# Padding between UV islands (0-1, fraction of texture size)
UV_ISLAND_MARGIN = 0.02

# --- SAMPLING ---
SAMPLE_METHOD = "BILINEAR"  # "NEAREST" or "BILINEAR"

# --- SEAM OPTIONS ---
# Where to place UV seams for unwrapping
# "AUTO"   = Use Blender's smart project
# "ANGLE"  = Mark seams at sharp angles
# "BOUNDS" = Use existing boundary edges
SEAM_MODE = "AUTO"
SEAM_ANGLE_THRESHOLD = 45  # Degrees, for ANGLE mode

# --- DEBUG ---
DEBUG_MODE = True
PROGRESS_UPDATE = 100  # Print every N faces

# --- MATERIAL ---
CREATE_PREVIEW_MATERIAL = True        # Create material to preview result


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


def ensure_edit_mode(obj):
    """Ensure we're in edit mode for the given object."""
    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.mode_set(mode='EDIT')


def depsgraph_update():
    """Force depsgraph update."""
    bpy.context.view_layer.update()


# ============================================================================
# IMAGE HANDLING (from VertexColors_v1.py)
# ============================================================================

_pixel_cache = {}


def list_loaded_images() -> list:
    """Return list of image names currently loaded in Blender."""
    return [img.name for img in bpy.data.images if not img.name.startswith('.')]


def print_loaded_images():
    """Print all images currently loaded in Blender."""
    print("\n=== Loaded Images in Blender ===")
    for img in bpy.data.images:
        if img.name.startswith('.'):
            continue
        size = f"{img.size[0]}x{img.size[1]}" if img.size[0] > 0 else "no data"
        source = img.filepath if img.filepath else "packed/generated"
        print(f"  '{img.name}' ({size}) - {source}")
    print("================================\n")


def load_image(image_ref: str, report: list):
    """Load an image from file path OR find by Blender image name."""
    # Check if already loaded
    img = bpy.data.images.get(image_ref)
    if img:
        report.append(f"[Image] Found in Blender: {image_ref}")
        return img

    # Check if it's a file path
    if '/' in image_ref or '\\' in image_ref:
        if not os.path.exists(image_ref):
            raise RuntimeError(f"Image file not found: {image_ref}")
        img = bpy.data.images.load(image_ref)
        report.append(f"[Image] Loaded from file: {image_ref}")
        return img

    # Try partial name match
    base_name = os.path.splitext(image_ref)[0].lower()
    for img in bpy.data.images:
        img_base = os.path.splitext(img.name)[0].lower()
        if img_base == base_name:
            report.append(f"[Image] Found by name: {img.name}")
            return img

    raise RuntimeError(f"Image not found: '{image_ref}'\n"
                       f"Loaded images: {list_loaded_images()}")


def cache_image_pixels(img):
    """Cache image pixels for fast access."""
    if img.name not in _pixel_cache:
        print(f"[Cache] Caching pixels for '{img.name}' ({img.size[0]}x{img.size[1]})...")
        _pixel_cache[img.name] = {
            'pixels': tuple(img.pixels[:]),
            'width': img.size[0],
            'height': img.size[1]
        }
    return _pixel_cache[img.name]


def clear_pixel_cache():
    """Clear the pixel cache."""
    global _pixel_cache
    _pixel_cache = {}


def sample_image_cached(img_name: str, u: float, v: float, method: str = "NEAREST") -> tuple:
    """Sample from cached pixel data."""
    import math

    if img_name not in _pixel_cache:
        return FILL_COLOR

    cache = _pixel_cache[img_name]
    pixels = cache['pixels']
    width = cache['width']
    height = cache['height']

    if width == 0 or height == 0:
        return FILL_COLOR

    if math.isnan(u) or math.isnan(v) or math.isinf(u) or math.isinf(v):
        return FILL_COLOR

    # Clamp to 0-1
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    if method == "NEAREST":
        x = int(u * (width - 1))
        y = int(v * (height - 1))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        idx = (y * width + x) * 4
        if idx + 3 < len(pixels):
            return (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])

    elif method == "BILINEAR":
        fx = u * (width - 1)
        fy = v * (height - 1)

        x0 = int(fx)
        y0 = int(fy)
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)

        dx = fx - x0
        dy = fy - y0

        def get_pixel(x, y):
            idx = (y * width + x) * 4
            if idx + 3 < len(pixels):
                return (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])
            return FILL_COLOR

        c00 = get_pixel(x0, y0)
        c10 = get_pixel(x1, y0)
        c01 = get_pixel(x0, y1)
        c11 = get_pixel(x1, y1)

        def lerp(a, b, t):
            return a + (b - a) * t

        result = []
        for i in range(4):
            top = lerp(c00[i], c10[i], dx)
            bottom = lerp(c01[i], c11[i], dx)
            result.append(lerp(top, bottom, dy))

        return tuple(result)

    return FILL_COLOR


# ============================================================================
# VIEW PROJECTION (adapted from VertexColors_v1.py)
# ============================================================================

VIEW_DIRECTIONS = {
    "front": (0, -1, 0),   # Camera at +Y looking toward -Y
    "back":  (0, 1, 0),    # Camera at -Y looking toward +Y
    "left":  (1, 0, 0),    # Camera at -X looking toward +X
    "right": (-1, 0, 0),   # Camera at +X looking toward -X
}


def get_mesh_bounds(mesh_obj):
    """Get world-space bounding box of mesh."""
    ensure_object_mode()
    mesh = mesh_obj.data
    world_verts = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]

    if not world_verts:
        return None

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


def get_best_view_for_normal(normal) -> str:
    """Determine which view best matches a normal direction."""
    best_view = "front"
    best_dot = -999

    for view_name, view_dir in VIEW_DIRECTIONS.items():
        camera_dir = Vector(view_dir)
        dot = normal.dot(camera_dir)
        if dot > best_dot:
            best_dot = dot
            best_view = view_name

    return best_view


def project_point_to_uv(point_world, view_name: str, bounds: dict, for_sampling: bool = False) -> tuple:
    """
    Project a 3D world position to 2D UV for a specific view.

    Args:
        point_world: World position to project
        view_name: Which view (front/back/left/right)
        bounds: Mesh bounding box
        for_sampling: If True, return 0-1 range for sampling source image
                      If False, return UV in quadrant for non-overlapping layout

    UV Layout (when for_sampling=False):
        +-------+-------+
        | front | back  |  (top half: v = 0.5-1.0)
        +-------+-------+
        | left  | right |  (bottom half: v = 0.0-0.5)
        +-------+-------+
    """
    rel = point_world - bounds['min']
    size = bounds['size']

    sx = size.x if size.x > 0.001 else 1.0
    sy = size.y if size.y > 0.001 else 1.0
    sz = size.z if size.z > 0.001 else 1.0

    # Calculate base UV (0-1 range within the view)
    if view_name == "front":
        base_u = rel.x / sx
        base_v = rel.z / sz
    elif view_name == "back":
        base_u = 1.0 - (rel.x / sx)
        base_v = rel.z / sz
    elif view_name == "left":
        base_u = 1.0 - (rel.y / sy)
        base_v = rel.z / sz
    elif view_name == "right":
        base_u = rel.y / sy
        base_v = rel.z / sz
    else:
        base_u, base_v = 0.5, 0.5

    # For sampling source images, return full 0-1 range
    if for_sampling:
        return (base_u, base_v)

    # For UV layout, place each view in its own quadrant (with small margin)
    margin = UV_ISLAND_MARGIN
    scale = 0.5 - margin  # Each quadrant is ~0.5 of UV space

    # Quadrant offsets
    quadrants = {
        "front": (margin, 0.5 + margin),           # Top-left
        "back":  (0.5 + margin, 0.5 + margin),     # Top-right
        "left":  (margin, margin),                  # Bottom-left
        "right": (0.5 + margin, margin),           # Bottom-right
    }

    offset_u, offset_v = quadrants.get(view_name, (0, 0))

    u = offset_u + base_u * scale
    v = offset_v + base_v * scale

    return (u, v)


def calculate_view_weights(normal, view_directions: dict) -> dict:
    """Calculate blend weights for each view based on normal."""
    weights = {}
    total = 0.0

    for view_name, view_dir in view_directions.items():
        camera_dir = Vector(view_dir)
        dot = normal.dot(camera_dir)
        weight = max(0.0, dot) ** 2
        weights[view_name] = weight
        total += weight

    if total > 0.001:
        for key in weights:
            weights[key] /= total
    else:
        for key in weights:
            weights[key] = 0.25

    return weights


def is_valid_sample(color) -> bool:
    """Check if a sampled color should be used (not background/transparent)."""
    # Check alpha
    if USE_ALPHA_MASK:
        alpha = color[3] if len(color) > 3 else 1.0
        if alpha < ALPHA_THRESHOLD:
            return False

    # Check background
    if BACKGROUND_COLOR is not None:
        bg = BACKGROUND_COLOR
        diff = abs(color[0] - bg[0]) + abs(color[1] - bg[1]) + abs(color[2] - bg[2])
        if diff < BACKGROUND_TOLERANCE * 3:
            return False

    return True


# ============================================================================
# UV MAP CREATION
# ============================================================================

def create_projected_uv_map(mesh_obj, bounds: dict, report: list):
    """
    Create a UV map using projection from best view per face.

    Each face is UV mapped based on which reference image it should use.
    """
    ensure_object_mode()
    mesh = mesh_obj.data

    report.append(f"\n[UV] Creating projected UV map: {UV_MAP_NAME}")

    # Create or get UV layer
    uv_layer = mesh.uv_layers.get(UV_MAP_NAME)
    if uv_layer and REPLACE_EXISTING_UV:
        mesh.uv_layers.remove(uv_layer)
        uv_layer = None

    if not uv_layer:
        uv_layer = mesh.uv_layers.new(name=UV_MAP_NAME)

    mesh.uv_layers.active = uv_layer

    # Track which view each face uses (for baking later)
    face_views = {}

    # Project each face based on its normal
    for poly in mesh.polygons:
        # Get face normal in world space
        normal_local = Vector(poly.normal)
        normal_world = (mesh_obj.matrix_world.to_3x3() @ normal_local).normalized()

        # Find best view for this face
        best_view = get_best_view_for_normal(normal_world)
        face_views[poly.index] = best_view

        # Project each vertex of the face
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            vert = mesh.vertices[loop.vertex_index]

            # Get world position
            vert_world = mesh_obj.matrix_world @ vert.co

            # Project to UV
            u, v = project_point_to_uv(vert_world, best_view, bounds)

            # Assign UV
            uv_layer.data[loop_idx].uv = (u, v)

    report.append(f"[UV] Projected {len(mesh.polygons)} faces")

    # Count faces per view
    view_counts = {}
    for view in face_views.values():
        view_counts[view] = view_counts.get(view, 0) + 1

    for view, count in view_counts.items():
        report.append(f"[UV] {view}: {count} faces")

    mesh.update()
    return face_views


def pack_uv_islands(mesh_obj, report: list):
    """Pack UV islands with margin."""
    ensure_edit_mode(mesh_obj)

    # Select all
    bpy.ops.mesh.select_all(action='SELECT')

    # Pack islands
    try:
        bpy.ops.uv.pack_islands(margin=UV_ISLAND_MARGIN)
        report.append(f"[UV] Packed islands with margin {UV_ISLAND_MARGIN}")
    except RuntimeError as e:
        report.append(f"[UV] Pack islands failed: {e}")

    ensure_object_mode()


# ============================================================================
# TEXTURE BAKING
# ============================================================================

def create_output_texture(report: list):
    """Create a new image for the output texture."""
    # Remove existing
    img = bpy.data.images.get(OUTPUT_TEXTURE_NAME)
    if img:
        bpy.data.images.remove(img)

    # Create new
    img = bpy.data.images.new(
        name=OUTPUT_TEXTURE_NAME,
        width=OUTPUT_RESOLUTION,
        height=OUTPUT_RESOLUTION,
        alpha=True,
        float_buffer=False
    )

    # Fill with fill color
    pixels = list(img.pixels)
    for i in range(0, len(pixels), 4):
        pixels[i] = FILL_COLOR[0]
        pixels[i + 1] = FILL_COLOR[1]
        pixels[i + 2] = FILL_COLOR[2]
        pixels[i + 3] = FILL_COLOR[3]
    img.pixels = pixels

    report.append(f"[Texture] Created {OUTPUT_RESOLUTION}x{OUTPUT_RESOLUTION} texture: {OUTPUT_TEXTURE_NAME}")
    return img


def bake_projection_to_texture(mesh_obj, images: dict, face_views: dict,
                                output_img, bounds: dict, report: list):
    """
    Bake projected images to output texture.

    For each pixel in output texture:
    1. Find which face it belongs to (via UV)
    2. Get the source view for that face
    3. Sample source image at projected position
    4. Write to output texture
    """
    import array

    ensure_object_mode()
    mesh = mesh_obj.data

    report.append(f"\n[Bake] Baking to {OUTPUT_RESOLUTION}x{OUTPUT_RESOLUTION} texture...")
    print(f"[Bake] Starting texture bake ({OUTPUT_RESOLUTION}x{OUTPUT_RESOLUTION})...")

    # Get UV layer
    uv_layer = mesh.uv_layers.get(UV_MAP_NAME)
    if not uv_layer:
        raise RuntimeError(f"UV layer '{UV_MAP_NAME}' not found!")

    # Build UV -> face mapping using bmesh for faster lookup
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    uv_layer_bm = bm.loops.layers.uv.get(UV_MAP_NAME)
    if not uv_layer_bm:
        bm.free()
        raise RuntimeError(f"UV layer '{UV_MAP_NAME}' not found in bmesh!")

    # Create output pixel array (faster than modifying img.pixels directly)
    width = OUTPUT_RESOLUTION
    height = OUTPUT_RESOLUTION
    output_pixels = array.array('f', FILL_COLOR * (width * height))

    # For each face, rasterize it to the texture
    total_faces = len(bm.faces)
    pixels_written = 0

    for fi, face in enumerate(bm.faces):
        if fi % PROGRESS_UPDATE == 0:
            print(f"[Bake] Processing face {fi}/{total_faces}...")

        # Get view for this face
        view_name = face_views.get(fi, "front")
        src_img = images.get(view_name)
        if not src_img:
            continue

        # Get face UVs
        uvs = []
        for loop in face.loops:
            uv = loop[uv_layer_bm].uv
            uvs.append((uv.x, uv.y))

        if len(uvs) < 3:
            continue

        # Get face world positions for source sampling
        world_positions = []
        for loop in face.loops:
            vert_world = mesh_obj.matrix_world @ loop.vert.co
            world_positions.append(vert_world)

        # Rasterize triangle(s) - simple scanline approach
        # For quads/ngons, triangulate
        for i in range(1, len(uvs) - 1):
            tri_uvs = [uvs[0], uvs[i], uvs[i + 1]]
            tri_positions = [world_positions[0], world_positions[i], world_positions[i + 1]]

            pixels_written += rasterize_triangle(
                tri_uvs, tri_positions, view_name, src_img,
                bounds, output_pixels, width, height
            )

    bm.free()

    # Write to output image
    output_img.pixels = output_pixels.tolist()
    output_img.update()

    report.append(f"[Bake] Wrote {pixels_written} pixels")
    print(f"[Bake] Complete - {pixels_written} pixels written")


def rasterize_triangle(tri_uvs, tri_positions, view_name, src_img,
                       bounds, output_pixels, width, height) -> int:
    """
    Rasterize a triangle to the output texture.

    Uses barycentric interpolation to map UV -> world position -> source sample.
    Returns number of pixels written.
    """
    # Get bounding box in UV space
    min_u = min(uv[0] for uv in tri_uvs)
    max_u = max(uv[0] for uv in tri_uvs)
    min_v = min(uv[1] for uv in tri_uvs)
    max_v = max(uv[1] for uv in tri_uvs)

    # Convert to pixel coords
    min_x = max(0, int(min_u * width) - 1)
    max_x = min(width - 1, int(max_u * width) + 1)
    min_y = max(0, int(min_v * height) - 1)
    max_y = min(height - 1, int(max_v * height) + 1)

    pixels_written = 0

    # Triangle vertices
    v0 = (tri_uvs[0][0], tri_uvs[0][1])
    v1 = (tri_uvs[1][0], tri_uvs[1][1])
    v2 = (tri_uvs[2][0], tri_uvs[2][1])

    # Precompute for barycentric
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 0.0001:
        return 0  # Degenerate triangle

    # Scan through bounding box
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            # UV coords for this pixel center
            u = (px + 0.5) / width
            v = (py + 0.5) / height

            # Barycentric coordinates
            w0 = ((v1[1] - v2[1]) * (u - v2[0]) + (v2[0] - v1[0]) * (v - v2[1])) / denom
            w1 = ((v2[1] - v0[1]) * (u - v2[0]) + (v0[0] - v2[0]) * (v - v2[1])) / denom
            w2 = 1.0 - w0 - w1

            # Check if inside triangle
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Interpolate world position
                world_pos = (
                    tri_positions[0] * w0 +
                    tri_positions[1] * w1 +
                    tri_positions[2] * w2
                )

                # Project to source UV (use for_sampling=True to get 0-1 range for source image)
                src_u, src_v = project_point_to_uv(world_pos, view_name, bounds, for_sampling=True)

                # Sample source image
                color = sample_image_cached(src_img.name, src_u, src_v, SAMPLE_METHOD)

                # Check if valid
                if is_valid_sample(color):
                    # Write to output
                    idx = (py * width + px) * 4
                    output_pixels[idx] = color[0]
                    output_pixels[idx + 1] = color[1]
                    output_pixels[idx + 2] = color[2]
                    output_pixels[idx + 3] = 1.0
                    pixels_written += 1

    return pixels_written


# ============================================================================
# MATERIAL PREVIEW
# ============================================================================

def create_preview_material(mesh_obj, output_img, report: list):
    """Create a material that displays the projected texture."""
    mat_name = "M_ProjectedTexture"

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
    output.location = (400, 0)

    # BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Texture
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.location = (-300, 0)
    tex_node.image = output_img
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

    # UV Map
    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.location = (-500, 0)
    uv_node.uv_map = UV_MAP_NAME
    links.new(uv_node.outputs['UV'], tex_node.inputs['Vector'])

    # Assign to mesh
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    report.append(f"[Material] Created preview material: {mat_name}")
    report.append("[Material] View in Material Preview mode (Z > Material Preview)")


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def project_to_texture(front: str = None, back: str = None,
                        left: str = None, right: str = None,
                        collection_name: str = None):
    """
    Project reference images onto mesh, creating UV map and texture.

    Args:
        front: Front view image (name or path)
        back: Back view image (name or path)
        left: Left view image (name or path)
        right: Right view image (name or path)
        collection_name: Override target collection
    """
    report = []
    report.append("TextureProject V1 - Image Projection to Texture")
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
    report.append(f"Vertices: {len(mesh_obj.data.vertices)}")
    report.append(f"Faces: {len(mesh_obj.data.polygons)}")

    # Load images
    images = {}
    image_paths = {
        "front": front or IMAGE_FRONT,
        "back": back or IMAGE_BACK,
        "left": left or IMAGE_LEFT,
        "right": right or IMAGE_RIGHT
    }

    report.append("\n[Images] Loading reference images...")
    clear_pixel_cache()

    for view_name, path in image_paths.items():
        if path:
            try:
                images[view_name] = load_image(path, report)
                cache_image_pixels(images[view_name])
            except RuntimeError as e:
                report.append(f"[Images] {view_name}: NOT FOUND - {e}")
                images[view_name] = None
        else:
            images[view_name] = None

    # Check we have at least one
    if not any(images.values()):
        raise RuntimeError("No images loaded! Check image names/paths.")

    # Get mesh bounds
    bounds = get_mesh_bounds(mesh_obj)
    if not bounds:
        raise RuntimeError("Could not calculate mesh bounds!")

    report.append(f"\n[Bounds] min={bounds['min']}")
    report.append(f"[Bounds] max={bounds['max']}")
    report.append(f"[Bounds] size={bounds['size']}")

    # Create UV map from projection
    face_views = create_projected_uv_map(mesh_obj, bounds, report)

    # Create output texture
    output_img = create_output_texture(report)

    # Bake projection to texture
    bake_projection_to_texture(mesh_obj, images, face_views, output_img, bounds, report)

    # Create preview material
    if CREATE_PREVIEW_MATERIAL:
        create_preview_material(mesh_obj, output_img, report)

    # Final info
    report.append("\n" + "=" * 50)
    report.append("TextureProject V1 complete.")
    report.append(f"\nOutput texture: '{OUTPUT_TEXTURE_NAME}'")
    report.append(f"UV map: '{UV_MAP_NAME}'")
    report.append("\nNEXT STEPS:")
    report.append("  1. View in Material Preview (Z > Material Preview)")
    report.append("  2. (Optional) Edit texture in Image Editor")
    report.append("  3. Use with VertexColors_v1.py for vertex color bake")
    report.append("=" * 50)

    log_text = "\n".join(report)
    log_to_text(log_text)
    print(log_text)
    print(f"\n--- Log: {LOG_TEXT_NAME} ---")


def run():
    """Run with default configuration."""
    project_to_texture(
        front=IMAGE_FRONT,
        back=IMAGE_BACK,
        left=IMAGE_LEFT,
        right=IMAGE_RIGHT
    )


if __name__ == "__main__":
    run()
