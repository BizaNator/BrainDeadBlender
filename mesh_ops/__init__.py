"""
mesh_ops - Modular Mesh Operations Package

Standalone mesh processing functions for Blender, designed for:
- Direct use in Blender Python scripts
- ComfyUI node integration
- Blender addon panels

Each module contains independent functions with consistent signatures:
    func(obj, report=None, **config) -> result

Example:
    from mesh_ops import colors, remesh, cleanup, decimate

    # Transfer colors between meshes
    colors.transfer_vertex_colors(source_obj, target_obj)

    # Remesh with sharp edge preservation
    remesh.apply_sharp_remesh(obj, octree_depth=8)

    # Fill holes and remove internal geometry
    cleanup.fill_holes(obj, max_sides=100)
    cleanup.remove_internal_geometry(obj, method="RAYCAST")

    # Decimate to target face count
    decimate.apply_planar_decimate(obj, angle=7.0)
    decimate.apply_collapse_decimate(obj, target_faces=5000)
"""

# Version
__version__ = "1.0.0"

# Import submodules for easy access
from . import utils
from . import colors
from . import remesh
from . import cleanup
from . import normals
from . import decimate

# Convenience imports for common functions
from .utils import ensure_object_mode, get_face_count, get_vertex_count
from .colors import (
    transfer_vertex_colors,
    bake_texture_to_vertex_colors,
    detect_color_edges,
    finalize_color_attribute,
)
from .remesh import (
    apply_sharp_remesh,
    apply_voxel_remesh,
    apply_quadriflow_remesh,
)
from .cleanup import (
    fill_holes,
    remove_internal_geometry,
    fix_non_manifold,
    pre_cleanup_mesh,
)
from .normals import (
    fix_normals,
    check_normal_orientation,
    verify_normals,
)
from .decimate import (
    apply_planar_decimate,
    apply_collapse_decimate,
    mark_sharp_edges,
)

__all__ = [
    # Submodules
    "utils",
    "colors",
    "remesh",
    "cleanup",
    "normals",
    "decimate",
    # Utils
    "ensure_object_mode",
    "get_face_count",
    "get_vertex_count",
    # Colors
    "transfer_vertex_colors",
    "bake_texture_to_vertex_colors",
    "detect_color_edges",
    "finalize_color_attribute",
    # Remesh
    "apply_sharp_remesh",
    "apply_voxel_remesh",
    "apply_quadriflow_remesh",
    # Cleanup
    "fill_holes",
    "remove_internal_geometry",
    "fix_non_manifold",
    "pre_cleanup_mesh",
    # Normals
    "fix_normals",
    "check_normal_orientation",
    "verify_normals",
    # Decimate
    "apply_planar_decimate",
    "apply_collapse_decimate",
    "mark_sharp_edges",
]
