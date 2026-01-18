"""
BrainDead Blender Tools - by BiloxiStudios Inc

Comprehensive mesh processing and character pipeline tools for Blender.

Features:
- UEFN Character Pipeline (Mixamo → UEFN Manny skeleton)
- Mesh decimation with color preservation
- Vertex color operations (bake, transfer, project, masks)
- Mesh cleanup and repair
- Remeshing (voxel, sharp, quadriflow)

Compatible with Blender 4.2+ extension system.
"""

bl_info = {
    "name": "BrainDead Blender Tools",
    "author": "BiloxiStudios Inc",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > BrainDead",
    "description": "Mesh processing and character pipeline tools",
    "category": "Mesh",
}

import bpy
from bpy.props import (
    IntProperty, FloatProperty, BoolProperty, EnumProperty,
    StringProperty, PointerProperty, CollectionProperty
)
from bpy.types import PropertyGroup, Panel, Operator

# Import mesh operations
from . import mesh_ops
from .mesh_ops import utils, colors, remesh, cleanup, normals, decimate


# ============================================================================
# PROPERTY GROUPS - Configuration exposed in UI
# ============================================================================

class BD_DecimateSettings(PropertyGroup):
    """Decimation settings"""
    target_faces: IntProperty(
        name="Target Faces",
        description="Target face count after decimation",
        default=5000,
        min=100,
        max=1000000
    )
    planar_angle: FloatProperty(
        name="Planar Angle",
        description="Angle threshold for merging coplanar faces (degrees)",
        default=7.0,
        min=0.0,
        max=45.0,
        subtype='ANGLE'
    )
    sharp_angle: FloatProperty(
        name="Sharp Angle",
        description="Angle threshold for marking sharp edges (degrees)",
        default=14.0,
        min=0.0,
        max=180.0,
        subtype='ANGLE'
    )
    preserve_boundaries: BoolProperty(
        name="Preserve Boundaries",
        description="Preserve mesh boundary edges during decimation",
        default=True
    )


class BD_RemeshSettings(PropertyGroup):
    """Remesh settings"""
    mode: EnumProperty(
        name="Mode",
        items=[
            ('NONE', "None", "Skip remesh"),
            ('SHARP', "Sharp", "Octree remesh - preserves thin geometry (recommended)"),
            ('VOXEL', "Voxel", "Voxel remesh - watertight but destroys thin geo"),
            ('VOXEL_HIGH', "Voxel High", "High-res voxel then decimate"),
            ('QUAD', "Quadriflow", "Clean quad topology"),
        ],
        default='NONE'
    )
    octree_depth: IntProperty(
        name="Octree Depth",
        description="Sharp remesh detail (6=~50K, 7=~200K, 8=~800K faces)",
        default=8,
        min=4,
        max=10
    )
    sharpness: FloatProperty(
        name="Sharpness",
        description="Edge sharpness threshold for sharp remesh",
        default=1.0,
        min=0.0,
        max=2.0
    )
    voxel_size: FloatProperty(
        name="Voxel Size",
        description="Voxel size (0 = auto-calculate)",
        default=0.0,
        min=0.0,
        max=0.1
    )
    target_polys: IntProperty(
        name="Target Polys",
        description="Target polygon count for voxel/quad remesh",
        default=100000,
        min=1000,
        max=2000000
    )


class BD_CleanupSettings(PropertyGroup):
    """Cleanup settings"""
    fill_holes: BoolProperty(
        name="Fill Holes",
        description="Fill holes in mesh",
        default=True
    )
    fill_holes_max_sides: IntProperty(
        name="Max Hole Sides",
        description="Maximum edges for a hole to be filled (0 = all)",
        default=100,
        min=0,
        max=1000
    )
    remove_internal: BoolProperty(
        name="Remove Internal",
        description="Remove internal/hidden geometry",
        default=True
    )
    internal_method: EnumProperty(
        name="Method",
        items=[
            ('RAYCAST', "Raycast", "Accurate ray-based detection"),
            ('SIMPLE', "Simple", "Fast built-in detection"),
        ],
        default='RAYCAST'
    )
    fix_manifold: BoolProperty(
        name="Fix Non-Manifold",
        description="Fix non-manifold geometry",
        default=True
    )
    aggressive_manifold: BoolProperty(
        name="Aggressive Fix",
        description="More aggressive manifold fixing (may lose geometry)",
        default=False
    )


class BD_NormalSettings(PropertyGroup):
    """Normal settings"""
    fix_normals: BoolProperty(
        name="Fix Normals",
        description="Fix face normals to point outward",
        default=True
    )
    method: EnumProperty(
        name="Method",
        items=[
            ('BLENDER', "Blender", "Topology-based (best for clean meshes)"),
            ('DIRECTION', "Direction", "Center-based (fails on cavities)"),
            ('BOTH', "Both", "Try Blender first, then direction"),
        ],
        default='BOTH'
    )
    threshold: IntProperty(
        name="Threshold %",
        description="Only fix if inverted percentage exceeds this",
        default=0,
        min=0,
        max=100
    )


class BD_ColorSettings(PropertyGroup):
    """Vertex color settings"""
    # Transfer settings
    output_name: StringProperty(
        name="Output Name",
        description="Name for output color attribute",
        default="Col"
    )
    transfer_mode: EnumProperty(
        name="Transfer Mode",
        description="How colors are sampled and applied",
        items=[
            ('FACE', "Face (Solid)", "Each face gets ONE solid color - clean, no blending"),
            ('VERTEX', "Vertex (Blended)", "Per-vertex colors, blended across faces"),
            ('CORNER', "Corner (Per-Loop)", "Each face-corner sampled independently"),
        ],
        default='FACE'
    )
    apply_flat_shading: BoolProperty(
        name="Apply Flat Shading",
        description="Apply flat shading after transfer (required for solid face colors to display correctly)",
        default=True
    )
    # Solidify settings
    solidify_method: EnumProperty(
        name="Solidify Method",
        description="How to determine the solid face color",
        items=[
            ('DOMINANT', "Dominant", "Most common color among face vertices"),
            ('AVERAGE', "Average", "Average of all vertex colors"),
            ('FIRST', "First", "Use first vertex color (fastest)"),
        ],
        default='DOMINANT'
    )
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Number of smoothing passes",
        default=1,
        min=1,
        max=10
    )
    # Bake settings
    bake_before_remesh: BoolProperty(
        name="Bake Before Remesh",
        description="Bake texture to vertex colors before remesh",
        default=True
    )
    preserve_through_remesh: BoolProperty(
        name="Preserve Through Remesh",
        description="Transfer colors back after remesh",
        default=True
    )
    # Edge detection settings
    detect_color_edges: BoolProperty(
        name="Detect Color Edges",
        description="Mark color boundaries as sharp edges",
        default=True
    )
    color_edge_threshold: FloatProperty(
        name="Edge Threshold",
        description="Color difference threshold for edge detection (0-1)",
        default=0.15,
        min=0.0,
        max=1.0
    )
    color_source: EnumProperty(
        name="Color Source",
        items=[
            ('TEXTURE', "Texture", "Sample from UV-mapped texture"),
            ('VERTEX_COLOR', "Vertex Color", "Use existing vertex colors"),
            ('MATERIAL', "Material", "Use material boundaries"),
        ],
        default='TEXTURE'
    )
    # Paint color
    paint_color: FloatVectorProperty(
        name="Paint Color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.0, 0.0, 1.0),
        description="Color to paint on selected faces"
    )
    # Favorite colors (up to 8)
    favorite_color_1: FloatVectorProperty(name="Favorite 1", subtype='COLOR', size=4, min=0.0, max=1.0, default=(1.0, 0.0, 0.0, 1.0))
    favorite_color_2: FloatVectorProperty(name="Favorite 2", subtype='COLOR', size=4, min=0.0, max=1.0, default=(0.0, 1.0, 0.0, 1.0))
    favorite_color_3: FloatVectorProperty(name="Favorite 3", subtype='COLOR', size=4, min=0.0, max=1.0, default=(0.0, 0.0, 1.0, 1.0))
    favorite_color_4: FloatVectorProperty(name="Favorite 4", subtype='COLOR', size=4, min=0.0, max=1.0, default=(1.0, 1.0, 0.0, 1.0))
    favorite_color_5: FloatVectorProperty(name="Favorite 5", subtype='COLOR', size=4, min=0.0, max=1.0, default=(1.0, 0.0, 1.0, 1.0))
    favorite_color_6: FloatVectorProperty(name="Favorite 6", subtype='COLOR', size=4, min=0.0, max=1.0, default=(0.0, 1.0, 1.0, 1.0))
    favorite_color_7: FloatVectorProperty(name="Favorite 7", subtype='COLOR', size=4, min=0.0, max=1.0, default=(1.0, 1.0, 1.0, 1.0))
    favorite_color_8: FloatVectorProperty(name="Favorite 8", subtype='COLOR', size=4, min=0.0, max=1.0, default=(0.0, 0.0, 0.0, 1.0))


class BD_UEFNSettings(PropertyGroup):
    """UEFN Pipeline settings"""
    # Collections
    source_collection: StringProperty(
        name="Source Collection",
        description="Collection containing UEFN mannequin reference",
        default="Source"
    )
    target_collection: StringProperty(
        name="Target Collection",
        description="Collection containing mesh to convert",
        default="Target"
    )
    export_collection: StringProperty(
        name="Export Collection",
        description="Collection for export-ready mesh",
        default="Export"
    )
    # Pipeline options
    scale_to_mannequin: BoolProperty(
        name="Scale to Mannequin",
        description="Scale target mesh to match UEFN mannequin height",
        default=True
    )
    align_armatures: BoolProperty(
        name="Align Armatures",
        description="Align target armature to UEFN armature",
        default=True
    )
    transfer_weights: BoolProperty(
        name="Transfer Weights",
        description="Transfer weights from UEFN mannequin",
        default=True
    )
    # Segmentation
    create_materials: BoolProperty(
        name="Create Segment Materials",
        description="Create colored materials for body segments",
        default=True
    )
    create_uv_islands: BoolProperty(
        name="Create UV Islands",
        description="Create UV seams at segment boundaries",
        default=True
    )


class BD_MaskSettings(PropertyGroup):
    """Mask color settings"""
    mode: EnumProperty(
        name="Mode",
        items=[
            ('ANALYZE', "Analyze", "Show color distribution (no changes)"),
            ('AUTO_MASK', "Auto Mask", "K-means cluster and assign by rank"),
            ('MANUAL_MASK', "Manual Mask", "Define specific color mappings"),
            ('MATERIAL_MASK', "Material Mask", "Use material slots for mapping"),
        ],
        default='ANALYZE'
    )
    num_clusters: IntProperty(
        name="Clusters",
        description="Number of color clusters for auto-mask",
        default=4,
        min=2,
        max=8
    )
    face_based: BoolProperty(
        name="Face Based",
        description="All vertices of a face get same mask (contiguous regions)",
        default=True
    )
    face_method: EnumProperty(
        name="Face Method",
        items=[
            ('DOMINANT', "Dominant", "Most common color among face vertices"),
            ('AVERAGE', "Average", "Average of all face vertex colors"),
            ('CENTER', "Center", "First vertex color (fastest)"),
        ],
        default='DOMINANT'
    )


class BD_TextureProjectSettings(PropertyGroup):
    """Texture projection settings"""
    front_image: StringProperty(
        name="Front Image",
        description="Front view image name in Blender"
    )
    back_image: StringProperty(
        name="Back Image",
        description="Back view image name in Blender"
    )
    left_image: StringProperty(
        name="Left Image",
        description="Left view image name in Blender"
    )
    right_image: StringProperty(
        name="Right Image",
        description="Right view image name in Blender"
    )
    output_resolution: IntProperty(
        name="Output Resolution",
        description="Output texture resolution (square)",
        default=2048,
        min=256,
        max=8192
    )
    uv_margin: FloatProperty(
        name="UV Margin",
        description="Padding between UV islands",
        default=0.02,
        min=0.0,
        max=0.1
    )


# ============================================================================
# OPERATORS - DECIMATION
# ============================================================================

class BD_OT_decimate_full(Operator):
    """Run full stylized decimation pipeline with all settings"""
    bl_idname = "braindead.decimate_full"
    bl_label = "Full Decimate Pipeline"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_decimate
        report = []

        decimate.decimate_stylized(
            obj,
            target_faces=settings.target_faces,
            planar_angle=settings.planar_angle,
            sharp_angle=settings.sharp_angle,
            preserve_boundaries=settings.preserve_boundaries,
            report=report
        )

        self.report({'INFO'}, f"Decimated to {utils.get_face_count(obj):,} faces")
        return {'FINISHED'}


class BD_OT_planar_decimate(Operator):
    """Merge coplanar faces"""
    bl_idname = "braindead.planar_decimate"
    bl_label = "Planar Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_decimate
        report = []

        decimate.apply_planar_decimate(obj, angle=settings.planar_angle, report=report)
        self.report({'INFO'}, f"Reduced to {utils.get_face_count(obj):,} faces")
        return {'FINISHED'}


class BD_OT_collapse_decimate(Operator):
    """Reduce to target face count"""
    bl_idname = "braindead.collapse_decimate"
    bl_label = "Collapse Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_decimate
        report = []

        decimate.apply_collapse_decimate(
            obj,
            target_faces=settings.target_faces,
            preserve_boundaries=settings.preserve_boundaries,
            report=report
        )
        self.report({'INFO'}, f"Reduced to {utils.get_face_count(obj):,} faces")
        return {'FINISHED'}


class BD_OT_mark_sharp_edges(Operator):
    """Mark edges as sharp based on angle"""
    bl_idname = "braindead.mark_sharp_edges"
    bl_label = "Mark Sharp Edges"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_decimate
        report = []

        marked = decimate.mark_sharp_edges(obj, angle=settings.sharp_angle, report=report)
        self.report({'INFO'}, f"Marked {marked} sharp edges")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - REMESH
# ============================================================================

class BD_OT_remesh(Operator):
    """Apply remesh with current settings"""
    bl_idname = "braindead.remesh"
    bl_label = "Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_remesh
        report = []

        if settings.mode == 'NONE':
            self.report({'INFO'}, "Remesh mode is NONE, skipping")
            return {'CANCELLED'}
        elif settings.mode == 'SHARP':
            remesh.apply_sharp_remesh(
                obj,
                octree_depth=settings.octree_depth,
                sharpness=settings.sharpness,
                report=report
            )
        elif settings.mode == 'VOXEL':
            voxel_size = settings.voxel_size if settings.voxel_size > 0 else None
            remesh.apply_voxel_remesh(
                obj,
                voxel_size=voxel_size,
                target_polys=settings.target_polys,
                report=report
            )
        elif settings.mode == 'VOXEL_HIGH':
            remesh.apply_voxel_high_remesh(
                obj,
                target_faces=settings.target_polys,
                report=report
            )
        elif settings.mode == 'QUAD':
            remesh.apply_quadriflow_remesh(
                obj,
                target_faces=settings.target_polys,
                report=report
            )

        self.report({'INFO'}, f"Remeshed to {utils.get_face_count(obj):,} faces")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - CLEANUP
# ============================================================================

class BD_OT_fill_holes(Operator):
    """Fill holes in mesh"""
    bl_idname = "braindead.fill_holes"
    bl_label = "Fill Holes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_cleanup
        report = []

        filled = cleanup.fill_holes(obj, max_sides=settings.fill_holes_max_sides, report=report)
        self.report({'INFO'}, f"Filled {filled} holes")
        return {'FINISHED'}


class BD_OT_remove_internal(Operator):
    """Remove internal/hidden faces"""
    bl_idname = "braindead.remove_internal"
    bl_label = "Remove Internal Geometry"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_cleanup
        report = []

        removed = cleanup.remove_internal_geometry(obj, method=settings.internal_method, report=report)
        self.report({'INFO'}, f"Removed {removed} internal faces")
        return {'FINISHED'}


class BD_OT_fix_manifold(Operator):
    """Fix non-manifold geometry"""
    bl_idname = "braindead.fix_manifold"
    bl_label = "Fix Non-Manifold"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_cleanup
        report = []

        cleanup.fix_non_manifold(obj, aggressive=settings.aggressive_manifold, report=report)
        self.report({'INFO'}, "Non-manifold geometry fixed")
        return {'FINISHED'}


class BD_OT_triangulate(Operator):
    """Triangulate n-gons"""
    bl_idname = "braindead.triangulate"
    bl_label = "Triangulate N-gons"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []

        count = cleanup.triangulate_ngons(obj, report=report)
        self.report({'INFO'}, f"Triangulated {count} n-gons")
        return {'FINISHED'}


class BD_OT_full_cleanup(Operator):
    """Run full cleanup pipeline"""
    bl_idname = "braindead.full_cleanup"
    bl_label = "Full Cleanup"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_cleanup
        report = []

        if settings.fill_holes:
            cleanup.fill_holes(obj, max_sides=settings.fill_holes_max_sides, report=report)

        if settings.remove_internal:
            cleanup.remove_internal_geometry(obj, method=settings.internal_method, report=report)

        if settings.fix_manifold:
            cleanup.fix_non_manifold(obj, aggressive=settings.aggressive_manifold, report=report)

        self.report({'INFO'}, f"Cleanup complete: {utils.get_face_count(obj):,} faces")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - NORMALS
# ============================================================================

class BD_OT_fix_normals(Operator):
    """Fix face normals to point outward"""
    bl_idname = "braindead.fix_normals"
    bl_label = "Fix Normals"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_normals
        report = []

        normals.fix_normals(obj, method=settings.method, threshold=settings.threshold, report=report)
        self.report({'INFO'}, "Normals fixed")
        return {'FINISHED'}


class BD_OT_verify_normals(Operator):
    """Check normal orientation"""
    bl_idname = "braindead.verify_normals"
    bl_label = "Verify Normals"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        stats = normals.check_normal_orientation(obj)
        self.report({'INFO'}, f"Normals: {stats['outward']} out, {stats['inward']} in ({stats['inward_pct']:.1f}% inverted)")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - VERTEX COLORS
# ============================================================================

class BD_OT_transfer_vertex_colors(Operator):
    """Transfer vertex colors: Click source (with colors), then Ctrl+click target (receives colors)"""
    bl_idname = "braindead.transfer_vertex_colors"
    bl_label = "Transfer Vertex Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) >= 2 and context.active_object.type == 'MESH'

    def execute(self, context):
        # Workflow: Click source first, Ctrl+click target
        # After Ctrl+click: Active = TARGET, Other = SOURCE
        target_obj = context.active_object
        settings = context.scene.bd_colors
        source_obj = None

        for obj in context.selected_objects:
            if obj != target_obj and obj.type == 'MESH':
                source_obj = obj
                break

        if not source_obj:
            self.report({'ERROR'}, "Click source mesh (with colors), then Ctrl+click target mesh")
            return {'CANCELLED'}

        # Check if source has color attributes
        source_mesh = source_obj.data
        if not hasattr(source_mesh, 'color_attributes') or len(source_mesh.color_attributes) == 0:
            self.report({'ERROR'}, f"Source '{source_obj.name}' has no vertex colors! Bake colors first.")
            return {'CANCELLED'}

        # List available colors for info
        color_names = [a.name for a in source_mesh.color_attributes]
        print(f"[Transfer] Source '{source_obj.name}' color attributes: {color_names}")
        print(f"[Transfer] Target '{target_obj.name}' will receive colors as '{settings.output_name}'")
        print(f"[Transfer] Mode: {settings.transfer_mode}")

        report = []
        success = colors.transfer_vertex_colors(
            source_obj, target_obj,
            output_name=settings.output_name,
            mode=settings.transfer_mode,
            report=report
        )

        # Print report to console
        for line in report:
            print(line)

        if success:
            # Apply flat shading if enabled (required for solid face colors)
            if settings.apply_flat_shading and settings.transfer_mode == 'FACE':
                colors.apply_flat_shading(target_obj, report=report)
                self.report({'INFO'}, f"Transferred colors from '{source_obj.name}' to '{target_obj.name}' (flat shading applied)")
            else:
                self.report({'INFO'}, f"Transferred colors from '{source_obj.name}' to '{target_obj.name}'")
            return {'FINISHED'}
        else:
            # Get the last error from report
            error_msg = "Transfer failed - check console for details"
            for line in report:
                if "WARNING" in line or "ERROR" in line:
                    error_msg = line.split("]")[-1].strip() if "]" in line else line
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


class BD_OT_bake_vertex_colors(Operator):
    """Bake texture to vertex colors"""
    bl_idname = "braindead.bake_vertex_colors"
    bl_label = "Bake Vertex Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []

        success = colors.bake_texture_to_vertex_colors(obj, report=report)
        if success:
            self.report({'INFO'}, "Vertex colors baked from texture")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Bake failed - check for UVs and texture")
            return {'CANCELLED'}


class BD_OT_detect_color_edges(Operator):
    """Detect and mark color boundaries as sharp edges"""
    bl_idname = "braindead.detect_color_edges"
    bl_label = "Detect Color Edges"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        marked = colors.detect_color_edges(
            obj,
            threshold=settings.color_edge_threshold,
            source=settings.color_source,
            report=report
        )
        self.report({'INFO'}, f"Marked {marked} color boundary edges")
        return {'FINISHED'}


class BD_OT_finalize_colors(Operator):
    """Finalize vertex colors to standard name for export"""
    bl_idname = "braindead.finalize_colors"
    bl_label = "Finalize Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        colors.finalize_color_attribute(obj, target_name=settings.output_name, report=report)
        self.report({'INFO'}, f"Color attribute finalized to '{settings.output_name}'")
        return {'FINISHED'}


class BD_OT_create_color_material(Operator):
    """Create material that displays vertex colors"""
    bl_idname = "braindead.create_color_material"
    bl_label = "Create Color Material"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        colors.create_vertex_color_material(obj, layer_name=settings.output_name, report=report)
        self.report({'INFO'}, "Vertex color material created")
        return {'FINISHED'}


class BD_OT_apply_flat_shading(Operator):
    """Apply flat shading - required for solid face colors to display correctly"""
    bl_idname = "braindead.apply_flat_shading"
    bl_label = "Apply Flat Shading"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []

        colors.apply_flat_shading(obj, report=report)

        for line in report:
            print(line)

        self.report({'INFO'}, f"Applied flat shading to {obj.name}")
        return {'FINISHED'}


class BD_OT_solidify_colors(Operator):
    """Convert vertex colors to solid face colors (no blending within faces)"""
    bl_idname = "braindead.solidify_colors"
    bl_label = "Solidify Face Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        return hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        faces = colors.solidify_face_colors(
            obj,
            color_attr_name=settings.output_name if settings.output_name else None,
            method=settings.solidify_method,
            report=report
        )

        for line in report:
            print(line)

        if faces > 0:
            self.report({'INFO'}, f"Solidified {faces:,} faces using {settings.solidify_method} method")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No color attribute found")
            return {'CANCELLED'}


class BD_OT_smooth_colors(Operator):
    """Smooth/blend vertex colors across the mesh"""
    bl_idname = "braindead.smooth_colors"
    bl_label = "Smooth Vertex Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        return hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        verts = colors.smooth_vertex_colors(
            obj,
            color_attr_name=settings.output_name if settings.output_name else None,
            iterations=settings.smooth_iterations,
            report=report
        )

        for line in report:
            print(line)

        if verts > 0:
            self.report({'INFO'}, f"Smoothed {verts:,} vertices ({settings.smooth_iterations} iterations)")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No color attribute found")
            return {'CANCELLED'}


# ============================================================================
# OPERATORS - EDIT MODE COLOR OPERATIONS
# ============================================================================

class BD_OT_solidify_selected(Operator):
    """Solidify colors on selected faces only (Edit Mode)"""
    bl_idname = "braindead.solidify_selected"
    bl_label = "Solidify Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        faces = colors.solidify_selected_faces(
            obj,
            color_attr_name=settings.output_name if settings.output_name else None,
            method=settings.solidify_method,
            report=report
        )

        for line in report:
            print(line)

        if faces > 0:
            self.report({'INFO'}, f"Solidified {faces} selected faces")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No faces selected or no color layer found")
            return {'CANCELLED'}


class BD_OT_smooth_selected(Operator):
    """Smooth colors on selected faces only (Edit Mode)"""
    bl_idname = "braindead.smooth_selected"
    bl_label = "Smooth Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        verts = colors.smooth_selected_faces(
            obj,
            color_attr_name=settings.output_name if settings.output_name else None,
            iterations=settings.smooth_iterations,
            report=report
        )

        for line in report:
            print(line)

        if verts > 0:
            self.report({'INFO'}, f"Smoothed {verts} vertices on selected faces")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No faces selected or no color layer found")
            return {'CANCELLED'}


class BD_OT_flat_shading_selected(Operator):
    """Apply flat shading to selected faces only (Edit Mode)"""
    bl_idname = "braindead.flat_shading_selected"
    bl_label = "Flat Shading Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        obj = context.active_object
        report = []

        changed = colors.apply_flat_shading_selected(obj, report=report)

        for line in report:
            print(line)

        self.report({'INFO'}, f"Set {changed} faces to flat shading")
        return {'FINISHED'}


class BD_OT_smooth_shading_selected(Operator):
    """Apply smooth shading to selected faces only (Edit Mode)"""
    bl_idname = "braindead.smooth_shading_selected"
    bl_label = "Smooth Shading Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        obj = context.active_object
        report = []

        changed = colors.apply_smooth_shading_selected(obj, report=report)

        for line in report:
            print(line)

        self.report({'INFO'}, f"Set {changed} faces to smooth shading")
        return {'FINISHED'}


class BD_OT_convert_color_domain(Operator):
    """Convert color attribute domain (Corner/Vertex)"""
    bl_idname = "braindead.convert_color_domain"
    bl_label = "Convert Color Domain"
    bl_options = {'REGISTER', 'UNDO'}

    target_domain: EnumProperty(
        name="Target Domain",
        items=[
            ('CORNER', "Corner (Per Face-Corner)", "Each face corner has its own color - allows solid faces"),
            ('POINT', "Vertex (Per Vertex)", "Each vertex has one color - always blends across faces"),
        ],
        default='CORNER'
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        return hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        success = colors.convert_color_domain(
            obj,
            target_domain=self.target_domain,
            color_attr_name=settings.output_name if settings.output_name else None,
            report=report
        )

        for line in report:
            print(line)

        if success:
            self.report({'INFO'}, f"Converted to {self.target_domain} domain")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Conversion failed")
            return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BD_OT_paint_faces(Operator):
    """Paint selected faces with the current paint color (Edit Mode)"""
    bl_idname = "braindead.paint_faces"
    bl_label = "Paint Selected Faces"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        color = tuple(settings.paint_color)
        faces = colors.paint_selected_faces(
            obj,
            color=color,
            color_attr_name=settings.output_name if settings.output_name else None,
            report=report
        )

        for line in report:
            print(line)

        if faces > 0:
            self.report({'INFO'}, f"Painted {faces} faces")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No faces selected or no color layer found")
            return {'CANCELLED'}


class BD_OT_sample_color(Operator):
    """Sample color from the selected face (Edit Mode) - Eye Dropper"""
    bl_idname = "braindead.sample_color"
    bl_label = "Sample Face Color"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.bd_colors
        report = []

        color = colors.sample_face_color(obj, report=report)

        for line in report:
            print(line)

        if color:
            settings.paint_color = color
            self.report({'INFO'}, f"Sampled color: ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No face selected or no color layer found")
            return {'CANCELLED'}


class BD_OT_use_favorite_color(Operator):
    """Use a favorite color as the paint color"""
    bl_idname = "braindead.use_favorite_color"
    bl_label = "Use Favorite"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(name="Favorite Index", default=1, min=1, max=8)

    def execute(self, context):
        settings = context.scene.bd_colors

        # Get the favorite color by index
        color = getattr(settings, f"favorite_color_{self.index}", None)
        if color:
            settings.paint_color = tuple(color)
            self.report({'INFO'}, f"Using favorite color {self.index}")
            return {'FINISHED'}
        return {'CANCELLED'}


class BD_OT_save_favorite_color(Operator):
    """Save current paint color as a favorite"""
    bl_idname = "braindead.save_favorite_color"
    bl_label = "Save to Favorite"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(name="Favorite Index", default=1, min=1, max=8)

    def execute(self, context):
        settings = context.scene.bd_colors

        # Set the favorite color by index
        setattr(settings, f"favorite_color_{self.index}", tuple(settings.paint_color))
        self.report({'INFO'}, f"Saved to favorite {self.index}")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - UEFN PIPELINE (Placeholder - needs scripts imported)
# ============================================================================

class BD_OT_uefn_convert(Operator):
    """Convert mesh to UEFN skeleton"""
    bl_idname = "braindead.uefn_convert"
    bl_label = "Convert to UEFN"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        # TODO: Import and run Pipeline_v31
        self.report({'WARNING'}, "UEFN conversion - run scripts/uefn_pipeline/Pipeline_v31.py")
        return {'CANCELLED'}


class BD_OT_uefn_modular_body(Operator):
    """Attach modular body parts"""
    bl_idname = "braindead.uefn_modular_body"
    bl_label = "Modular Body"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(
        name="Mode",
        items=[
            ('hands', "Hands", "Attach detailed hand meshes"),
            ('feet', "Feet", "Attach detailed foot meshes"),
            ('head', "Head", "Separate head from body"),
            ('all', "All", "Run all operations"),
        ],
        default='all'
    )

    def execute(self, context):
        # TODO: Import and run ModularBody_v1
        self.report({'WARNING'}, f"Modular body ({self.mode}) - run scripts/uefn_pipeline/ModularBody_v1.py")
        return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BD_OT_uefn_segmentation(Operator):
    """Create body segments"""
    bl_idname = "braindead.uefn_segmentation"
    bl_label = "Body Segmentation"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(
        name="Mode",
        items=[
            ('ALL', "Materials + UVs", "Create materials and UV islands"),
            ('MATERIALS', "Materials Only", "Create segment materials"),
            ('UV_ISLANDS', "UV Islands", "Create UV seams and unwrap"),
            ('SEAMS_ONLY', "Seams Only", "Just mark seams"),
        ],
        default='ALL'
    )

    def execute(self, context):
        # TODO: Import and run Segmentation_v1
        self.report({'WARNING'}, f"Segmentation ({self.mode}) - run scripts/uefn_pipeline/Segmentation_v1.py")
        return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BD_OT_uefn_export(Operator):
    """Export for UEFN"""
    bl_idname = "braindead.uefn_export"
    bl_label = "Export UEFN FBX"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # TODO: Import and run ExportUEFN_v1
        self.report({'WARNING'}, "UEFN export - run scripts/uefn_pipeline/ExportUEFN_v1.py")
        return {'CANCELLED'}


# ============================================================================
# OPERATORS - TEXTURE PROJECT
# ============================================================================

class BD_OT_texture_project(Operator):
    """Project images to texture"""
    bl_idname = "braindead.texture_project"
    bl_label = "Project to Texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        settings = context.scene.bd_texture_project
        # TODO: Import and run TextureProject_v1
        self.report({'WARNING'}, "Texture projection - run scripts/vertex_colors/TextureProject_v1.py")
        return {'CANCELLED'}


class BD_OT_list_images(Operator):
    """List available images in Blender"""
    bl_idname = "braindead.list_images"
    bl_label = "List Images"

    def execute(self, context):
        images = [img.name for img in bpy.data.images if img.type == 'IMAGE' and img.size[0] > 0]
        if images:
            self.report({'INFO'}, f"Images: {', '.join(images)}")
        else:
            self.report({'WARNING'}, "No images loaded")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - MASK COLORS
# ============================================================================

class BD_OT_analyze_colors(Operator):
    """Analyze vertex color distribution"""
    bl_idname = "braindead.analyze_colors"
    bl_label = "Analyze Colors"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        # TODO: Import and run MaskColors_v1 in analyze mode
        self.report({'WARNING'}, "Color analysis - run scripts/vertex_colors/MaskColors_v1.py with MODE='ANALYZE'")
        return {'CANCELLED'}


class BD_OT_auto_mask(Operator):
    """Auto-generate color masks"""
    bl_idname = "braindead.auto_mask"
    bl_label = "Auto Mask"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        settings = context.scene.bd_mask
        # TODO: Import and run MaskColors_v1 in auto_mask mode
        self.report({'WARNING'}, "Auto mask - run scripts/vertex_colors/MaskColors_v1.py with MODE='AUTO_MASK'")
        return {'CANCELLED'}


# ============================================================================
# OPERATORS - UTILS
# ============================================================================

class BD_OT_debug_bones(Operator):
    """Debug bone orientations"""
    bl_idname = "braindead.debug_bones"
    bl_label = "Debug Bone Axes"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'ARMATURE'

    def execute(self, context):
        # TODO: Import and run debug_bone_axes
        self.report({'WARNING'}, "Debug bones - run scripts/utils/debug_bone_axes.py")
        return {'CANCELLED'}


class BD_OT_check_hierarchy(Operator):
    """Check export hierarchy"""
    bl_idname = "braindead.check_hierarchy"
    bl_label = "Check Hierarchy"

    def execute(self, context):
        # Print hierarchy of Export collection
        export_col = bpy.data.collections.get("Export")
        if not export_col:
            self.report({'WARNING'}, "No 'Export' collection found")
            return {'CANCELLED'}

        def print_obj(obj, indent=0):
            prefix = "  " * indent
            print(f"{prefix}{obj.name} [{obj.type}]")
            for child in obj.children:
                print_obj(child, indent + 1)

        print("=== EXPORT HIERARCHY ===")
        for obj in export_col.objects:
            if obj.parent is None:
                print_obj(obj)

        self.report({'INFO'}, "Hierarchy printed to console")
        return {'FINISHED'}


class BD_OT_mesh_stats(Operator):
    """Show mesh statistics"""
    bl_idname = "braindead.mesh_stats"
    bl_label = "Mesh Stats"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        mesh = obj.data

        stats = (
            f"Mesh: {obj.name}\n"
            f"Vertices: {len(mesh.vertices):,}\n"
            f"Edges: {len(mesh.edges):,}\n"
            f"Faces: {len(mesh.polygons):,}\n"
            f"Materials: {len(mesh.materials)}\n"
            f"UV Layers: {len(mesh.uv_layers)}\n"
        )

        if hasattr(mesh, 'color_attributes'):
            stats += f"Color Attributes: {len(mesh.color_attributes)}"
            for attr in mesh.color_attributes:
                stats += f"\n  - {attr.name} ({attr.domain})"

        print(stats)
        self.report({'INFO'}, f"Stats printed to console ({len(mesh.polygons):,} faces)")
        return {'FINISHED'}


# ============================================================================
# PANELS
# ============================================================================

class BD_PT_main(Panel):
    """BrainDead Tools Main Panel"""
    bl_label = "BrainDead Tools"
    bl_idname = "BD_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if obj and obj.type == 'MESH':
            box = layout.box()
            box.label(text=f"{obj.name}", icon='MESH_DATA')
            row = box.row()
            row.label(text=f"V: {len(obj.data.vertices):,}")
            row.label(text=f"F: {len(obj.data.polygons):,}")
            layout.operator("braindead.mesh_stats", icon='INFO')
        elif obj and obj.type == 'ARMATURE':
            box = layout.box()
            box.label(text=f"{obj.name}", icon='ARMATURE_DATA')
            box.label(text=f"Bones: {len(obj.data.bones)}")
        else:
            layout.label(text="Select a mesh or armature", icon='INFO')


class BD_PT_decimate(Panel):
    """Decimation Panel"""
    bl_label = "Decimation"
    bl_idname = "BD_PT_decimate"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_decimate

        layout.prop(settings, "target_faces")
        layout.prop(settings, "planar_angle")
        layout.prop(settings, "sharp_angle")
        layout.prop(settings, "preserve_boundaries")

        layout.separator()
        layout.operator("braindead.decimate_full", icon='MOD_DECIM')

        row = layout.row(align=True)
        row.operator("braindead.planar_decimate", text="Planar")
        row.operator("braindead.collapse_decimate", text="Collapse")

        layout.operator("braindead.mark_sharp_edges", icon='EDGESEL')


class BD_PT_remesh(Panel):
    """Remesh Panel"""
    bl_label = "Remesh"
    bl_idname = "BD_PT_remesh"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_remesh

        layout.prop(settings, "mode")

        if settings.mode == 'SHARP':
            layout.prop(settings, "octree_depth")
            layout.prop(settings, "sharpness")
        elif settings.mode in ('VOXEL', 'VOXEL_HIGH', 'QUAD'):
            layout.prop(settings, "target_polys")
            if settings.mode == 'VOXEL':
                layout.prop(settings, "voxel_size")

        layout.separator()
        layout.operator("braindead.remesh", icon='MOD_REMESH')


class BD_PT_cleanup(Panel):
    """Cleanup Panel"""
    bl_label = "Cleanup"
    bl_idname = "BD_PT_cleanup"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_cleanup

        col = layout.column(align=True)
        col.prop(settings, "fill_holes")
        if settings.fill_holes:
            col.prop(settings, "fill_holes_max_sides")

        col = layout.column(align=True)
        col.prop(settings, "remove_internal")
        if settings.remove_internal:
            col.prop(settings, "internal_method")

        col = layout.column(align=True)
        col.prop(settings, "fix_manifold")
        if settings.fix_manifold:
            col.prop(settings, "aggressive_manifold")

        layout.separator()
        layout.operator("braindead.full_cleanup", icon='BRUSH_DATA')

        row = layout.row(align=True)
        row.operator("braindead.fill_holes", text="Holes")
        row.operator("braindead.remove_internal", text="Internal")

        row = layout.row(align=True)
        row.operator("braindead.fix_manifold", text="Manifold")
        row.operator("braindead.triangulate", text="Triangulate")


class BD_PT_normals(Panel):
    """Normals Panel"""
    bl_label = "Normals"
    bl_idname = "BD_PT_normals"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_normals

        layout.prop(settings, "method")
        layout.prop(settings, "threshold")

        layout.separator()
        row = layout.row(align=True)
        row.operator("braindead.fix_normals", icon='NORMALS_FACE')
        row.operator("braindead.verify_normals", text="", icon='ZOOM_ALL')


class BD_PT_colors(Panel):
    """Vertex Colors Panel"""
    bl_label = "Vertex Colors"
    bl_idname = "BD_PT_colors"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_colors

        layout.prop(settings, "output_name")

        layout.separator()
        layout.label(text="Transfer:", icon='BRUSH_DATA')

        # Show source/target info
        # Workflow: Click source first, Ctrl+click target
        # After Ctrl+click: Active = TARGET, Other = SOURCE
        box = layout.box()
        active = context.active_object
        other_selected = [o for o in context.selected_objects if o.type == 'MESH' and o != active]

        # Source = other selected (should have colors)
        if other_selected:
            source = other_selected[0]
            has_colors = hasattr(source.data, 'color_attributes') and len(source.data.color_attributes) > 0
            if has_colors:
                color_names = [a.name for a in source.data.color_attributes]
                box.label(text=f"Source: {source.name}", icon='BRUSH_DATA')
                box.label(text=f"  Colors: {', '.join(color_names)}")
            else:
                box.label(text=f"Source: {source.name} (NO COLORS!)", icon='ERROR')
        else:
            box.label(text="Source: (click mesh with colors first)", icon='BRUSH_DATA')

        # Target = active object (receives colors)
        if active and active.type == 'MESH':
            box.label(text=f"Target: {active.name}", icon='FORWARD')
        else:
            box.label(text="Target: (then Ctrl+click target)", icon='FORWARD')

        layout.prop(settings, "transfer_mode")
        if settings.transfer_mode == 'FACE':
            layout.prop(settings, "apply_flat_shading")
        layout.operator("braindead.transfer_vertex_colors", icon='UV_SYNC_SELECT')

        layout.separator()
        layout.label(text="Bake:")
        layout.operator("braindead.bake_vertex_colors", icon='TEXTURE')

        # Solidify / Smooth section
        layout.separator()

        # Check if in edit mode for selected-only operations
        obj = context.active_object
        in_edit_mode = obj and obj.mode == 'EDIT'

        if in_edit_mode:
            layout.label(text="Edit Mode (Selected Faces):", icon='EDITMODE_HLT')

            # Paint section
            box = layout.box()
            box.label(text="Paint:", icon='BRUSH_DATA')
            row = box.row(align=True)
            row.prop(settings, "paint_color", text="")
            row.operator("braindead.sample_color", text="", icon='EYEDROPPER')
            row.operator("braindead.paint_faces", text="Paint")

            # Favorite colors - 2 rows of 4
            row = box.row(align=True)
            for i in range(1, 5):
                col = row.column(align=True)
                col.prop(settings, f"favorite_color_{i}", text="")
                op = col.operator("braindead.use_favorite_color", text="", icon='FORWARD')
                op.index = i
            row = box.row(align=True)
            for i in range(5, 9):
                col = row.column(align=True)
                col.prop(settings, f"favorite_color_{i}", text="")
                op = col.operator("braindead.use_favorite_color", text="", icon='FORWARD')
                op.index = i

            # Save to favorite
            row = box.row(align=True)
            row.label(text="Save to:")
            for i in range(1, 9):
                op = row.operator("braindead.save_favorite_color", text=str(i))
                op.index = i

            layout.separator()

            # Solidify/Smooth section
            row = layout.row(align=True)
            row.prop(settings, "solidify_method", text="")
            row.operator("braindead.solidify_selected", text="Solidify")

            row = layout.row(align=True)
            row.prop(settings, "smooth_iterations", text="Iterations")
            row.operator("braindead.smooth_selected", text="Smooth")

            row = layout.row(align=True)
            row.operator("braindead.flat_shading_selected", text="Flat", icon='MESH_PLANE')
            row.operator("braindead.smooth_shading_selected", text="Smooth", icon='SMOOTHCURVE')

        else:
            layout.label(text="Adjust Colors (Whole Mesh):", icon='MOD_SMOOTH')

            row = layout.row(align=True)
            row.prop(settings, "solidify_method", text="")
            row.operator("braindead.solidify_colors", text="Solidify")

            row = layout.row(align=True)
            row.prop(settings, "smooth_iterations", text="Iterations")
            row.operator("braindead.smooth_colors", text="Smooth")

            layout.operator("braindead.apply_flat_shading", text="Apply Flat Shading", icon='MESH_PLANE')

        # Domain conversion (works in both modes but requires object mode)
        layout.separator()
        layout.label(text="Color Domain:")
        layout.operator("braindead.convert_color_domain", text="Convert Domain", icon='MOD_DATA_TRANSFER')

        layout.separator()
        layout.label(text="Edge Detection:")
        layout.prop(settings, "color_source")
        layout.prop(settings, "color_edge_threshold")
        layout.operator("braindead.detect_color_edges", icon='EDGESEL')

        layout.separator()
        row = layout.row(align=True)
        row.operator("braindead.finalize_colors", text="Finalize")
        row.operator("braindead.create_color_material", text="Material")


class BD_PT_uefn(Panel):
    """UEFN Pipeline Panel"""
    bl_label = "UEFN Pipeline"
    bl_idname = "BD_PT_uefn"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_uefn

        layout.label(text="Collections:")
        layout.prop(settings, "source_collection")
        layout.prop(settings, "target_collection")
        layout.prop(settings, "export_collection")

        layout.separator()
        layout.label(text="Pipeline:")
        layout.operator("braindead.uefn_convert", icon='ARMATURE_DATA')
        layout.operator("braindead.uefn_modular_body", icon='MOD_BUILD')
        layout.operator("braindead.uefn_segmentation", icon='UV_FACESEL')
        layout.operator("braindead.uefn_export", icon='EXPORT')

        layout.separator()
        layout.label(text="Utils:")
        layout.operator("braindead.check_hierarchy", icon='OUTLINER')
        layout.operator("braindead.debug_bones", icon='BONE_DATA')


class BD_PT_texture_project(Panel):
    """Texture Projection Panel"""
    bl_label = "Texture Project"
    bl_idname = "BD_PT_texture_project"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_texture_project

        layout.operator("braindead.list_images", icon='IMAGE_DATA')

        layout.separator()
        layout.label(text="View Images:")
        layout.prop(settings, "front_image")
        layout.prop(settings, "back_image")
        layout.prop(settings, "left_image")
        layout.prop(settings, "right_image")

        layout.separator()
        layout.prop(settings, "output_resolution")
        layout.prop(settings, "uv_margin")

        layout.separator()
        layout.operator("braindead.texture_project", icon='TEXTURE')


class BD_PT_masks(Panel):
    """Mask Colors Panel"""
    bl_label = "Mask Colors"
    bl_idname = "BD_PT_masks"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BrainDead'
    bl_parent_id = "BD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bd_mask

        layout.prop(settings, "mode")

        if settings.mode == 'AUTO_MASK':
            layout.prop(settings, "num_clusters")

        layout.prop(settings, "face_based")
        if settings.face_based:
            layout.prop(settings, "face_method")

        layout.separator()
        row = layout.row(align=True)
        row.operator("braindead.analyze_colors", text="Analyze")
        row.operator("braindead.auto_mask", text="Auto Mask")


# ============================================================================
# REGISTRATION
# ============================================================================

classes = [
    # Property Groups
    BD_DecimateSettings,
    BD_RemeshSettings,
    BD_CleanupSettings,
    BD_NormalSettings,
    BD_ColorSettings,
    BD_UEFNSettings,
    BD_MaskSettings,
    BD_TextureProjectSettings,
    # Operators - Decimation
    BD_OT_decimate_full,
    BD_OT_planar_decimate,
    BD_OT_collapse_decimate,
    BD_OT_mark_sharp_edges,
    # Operators - Remesh
    BD_OT_remesh,
    # Operators - Cleanup
    BD_OT_fill_holes,
    BD_OT_remove_internal,
    BD_OT_fix_manifold,
    BD_OT_triangulate,
    BD_OT_full_cleanup,
    # Operators - Normals
    BD_OT_fix_normals,
    BD_OT_verify_normals,
    # Operators - Colors
    BD_OT_transfer_vertex_colors,
    BD_OT_bake_vertex_colors,
    BD_OT_detect_color_edges,
    BD_OT_finalize_colors,
    BD_OT_create_color_material,
    BD_OT_apply_flat_shading,
    BD_OT_solidify_colors,
    BD_OT_smooth_colors,
    # Operators - Colors (Edit Mode)
    BD_OT_solidify_selected,
    BD_OT_smooth_selected,
    BD_OT_flat_shading_selected,
    BD_OT_smooth_shading_selected,
    BD_OT_convert_color_domain,
    BD_OT_paint_faces,
    BD_OT_sample_color,
    BD_OT_use_favorite_color,
    BD_OT_save_favorite_color,
    # Operators - UEFN
    BD_OT_uefn_convert,
    BD_OT_uefn_modular_body,
    BD_OT_uefn_segmentation,
    BD_OT_uefn_export,
    # Operators - Texture Project
    BD_OT_texture_project,
    BD_OT_list_images,
    # Operators - Masks
    BD_OT_analyze_colors,
    BD_OT_auto_mask,
    # Operators - Utils
    BD_OT_debug_bones,
    BD_OT_check_hierarchy,
    BD_OT_mesh_stats,
    # Panels
    BD_PT_main,
    BD_PT_decimate,
    BD_PT_remesh,
    BD_PT_cleanup,
    BD_PT_normals,
    BD_PT_colors,
    BD_PT_uefn,
    BD_PT_texture_project,
    BD_PT_masks,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register property groups
    bpy.types.Scene.bd_decimate = PointerProperty(type=BD_DecimateSettings)
    bpy.types.Scene.bd_remesh = PointerProperty(type=BD_RemeshSettings)
    bpy.types.Scene.bd_cleanup = PointerProperty(type=BD_CleanupSettings)
    bpy.types.Scene.bd_normals = PointerProperty(type=BD_NormalSettings)
    bpy.types.Scene.bd_colors = PointerProperty(type=BD_ColorSettings)
    bpy.types.Scene.bd_uefn = PointerProperty(type=BD_UEFNSettings)
    bpy.types.Scene.bd_mask = PointerProperty(type=BD_MaskSettings)
    bpy.types.Scene.bd_texture_project = PointerProperty(type=BD_TextureProjectSettings)


def unregister():
    # Unregister property groups
    del bpy.types.Scene.bd_decimate
    del bpy.types.Scene.bd_remesh
    del bpy.types.Scene.bd_cleanup
    del bpy.types.Scene.bd_normals
    del bpy.types.Scene.bd_colors
    del bpy.types.Scene.bd_uefn
    del bpy.types.Scene.bd_mask
    del bpy.types.Scene.bd_texture_project

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
