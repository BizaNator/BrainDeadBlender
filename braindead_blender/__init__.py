"""
BrainDead Blender Tools - by BiloxiStudios Inc

Comprehensive mesh processing and character pipeline tools for Blender.

Features:
- Mesh decimation with color preservation
- Vertex color transfer between meshes
- Remeshing (voxel, sharp, quadriflow)
- Mesh cleanup and repair
- Normal fixing
- UEFN/Fortnite character pipeline support

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
    IntProperty, FloatProperty, BoolProperty, EnumProperty, StringProperty
)

# Import mesh operations
from . import mesh_ops
from .mesh_ops import utils, colors, remesh, cleanup, normals, decimate


# ============================================================================
# OPERATORS - DECIMATION
# ============================================================================

class BD_OT_decimate_stylized(bpy.types.Operator):
    """Decimate mesh to stylized low-poly with color preservation"""
    bl_idname = "braindead.decimate_stylized"
    bl_label = "Stylized Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    target_faces: IntProperty(
        name="Target Faces",
        description="Target face count",
        default=5000,
        min=100,
        max=1000000
    )
    planar_angle: FloatProperty(
        name="Planar Angle",
        description="Angle for merging coplanar faces",
        default=7.0,
        min=0.0,
        max=45.0
    )
    sharp_angle: FloatProperty(
        name="Sharp Angle",
        description="Angle for marking sharp edges",
        default=14.0,
        min=0.0,
        max=180.0
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []

        decimate.decimate_stylized(
            obj,
            target_faces=self.target_faces,
            planar_angle=self.planar_angle,
            sharp_angle=self.sharp_angle,
            report=report
        )

        self.report({'INFO'}, f"Decimated to {utils.get_face_count(obj)} faces")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BD_OT_planar_decimate(bpy.types.Operator):
    """Merge coplanar faces"""
    bl_idname = "braindead.planar_decimate"
    bl_label = "Planar Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    angle: FloatProperty(
        name="Angle",
        description="Angle threshold for merging",
        default=7.0,
        min=0.0,
        max=45.0
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        decimate.apply_planar_decimate(obj, angle=self.angle, report=report)
        self.report({'INFO'}, f"Reduced to {utils.get_face_count(obj)} faces")
        return {'FINISHED'}


class BD_OT_collapse_decimate(bpy.types.Operator):
    """Reduce to target face count"""
    bl_idname = "braindead.collapse_decimate"
    bl_label = "Collapse Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    target_faces: IntProperty(
        name="Target Faces",
        description="Target face count",
        default=5000,
        min=100,
        max=1000000
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        decimate.apply_collapse_decimate(obj, target_faces=self.target_faces, report=report)
        self.report({'INFO'}, f"Reduced to {utils.get_face_count(obj)} faces")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BD_OT_mark_sharp_edges(bpy.types.Operator):
    """Mark edges as sharp based on angle"""
    bl_idname = "braindead.mark_sharp_edges"
    bl_label = "Mark Sharp Edges"
    bl_options = {'REGISTER', 'UNDO'}

    angle: FloatProperty(
        name="Angle",
        description="Angle threshold in degrees",
        default=30.0,
        min=0.0,
        max=180.0
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        marked = decimate.mark_sharp_edges(obj, angle=self.angle, report=report)
        self.report({'INFO'}, f"Marked {marked} sharp edges")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - VERTEX COLORS
# ============================================================================

class BD_OT_transfer_vertex_colors(bpy.types.Operator):
    """Transfer vertex colors from one mesh to another using BVH lookup"""
    bl_idname = "braindead.transfer_vertex_colors"
    bl_label = "Transfer Vertex Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        selected = context.selected_objects
        return len(selected) >= 2 and context.active_object.type == 'MESH'

    def execute(self, context):
        target_obj = context.active_object
        source_obj = None

        for obj in context.selected_objects:
            if obj != target_obj and obj.type == 'MESH':
                source_obj = obj
                break

        if not source_obj:
            self.report({'ERROR'}, "Select source mesh (with colors) and target mesh (active)")
            return {'CANCELLED'}

        report = []
        success = colors.transfer_vertex_colors(source_obj, target_obj, report=report)

        if success:
            self.report({'INFO'}, f"Transferred colors from {source_obj.name} to {target_obj.name}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Transfer failed - check console")
            return {'CANCELLED'}


class BD_OT_bake_vertex_colors(bpy.types.Operator):
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
            self.report({'ERROR'}, "Bake failed - check console")
            return {'CANCELLED'}


class BD_OT_detect_color_edges(bpy.types.Operator):
    """Detect and mark color boundaries as sharp edges"""
    bl_idname = "braindead.detect_color_edges"
    bl_label = "Detect Color Edges"
    bl_options = {'REGISTER', 'UNDO'}

    threshold: FloatProperty(
        name="Threshold",
        description="Color difference threshold (0-1)",
        default=0.15,
        min=0.0,
        max=1.0
    )
    source: EnumProperty(
        name="Source",
        items=[
            ('TEXTURE', "Texture", "Sample from UV-mapped texture"),
            ('VERTEX_COLOR', "Vertex Color", "Use existing vertex colors"),
            ('MATERIAL', "Material", "Use material boundaries"),
        ],
        default='TEXTURE'
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        marked = colors.detect_color_edges(
            obj,
            threshold=self.threshold,
            source=self.source,
            report=report
        )
        self.report({'INFO'}, f"Marked {marked} color boundary edges")
        return {'FINISHED'}


class BD_OT_finalize_colors(bpy.types.Operator):
    """Finalize vertex colors to 'Col' for export"""
    bl_idname = "braindead.finalize_colors"
    bl_label = "Finalize Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        colors.finalize_color_attribute(obj, target_name="Col", report=report)
        self.report({'INFO'}, "Color attribute finalized to 'Col'")
        return {'FINISHED'}


# ============================================================================
# OPERATORS - REMESH
# ============================================================================

class BD_OT_remesh_sharp(bpy.types.Operator):
    """Apply sharp (octree) remesh - preserves thin geometry"""
    bl_idname = "braindead.remesh_sharp"
    bl_label = "Sharp Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    octree_depth: IntProperty(
        name="Octree Depth",
        description="Higher = more detail (6=~50K, 7=~200K, 8=~800K faces)",
        default=8,
        min=4,
        max=10
    )
    sharpness: FloatProperty(
        name="Sharpness",
        description="Edge sharpness threshold",
        default=1.0,
        min=0.0,
        max=2.0
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        remesh.apply_sharp_remesh(
            obj,
            octree_depth=self.octree_depth,
            sharpness=self.sharpness,
            report=report
        )
        self.report({'INFO'}, f"Remeshed to {utils.get_face_count(obj)} faces")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BD_OT_remesh_voxel(bpy.types.Operator):
    """Apply voxel remesh - creates watertight mesh (destroys thin geometry!)"""
    bl_idname = "braindead.remesh_voxel"
    bl_label = "Voxel Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    target_polys: IntProperty(
        name="Target Polys",
        description="Target polygon count (voxel size auto-calculated)",
        default=100000,
        min=1000,
        max=2000000
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        remesh.apply_voxel_remesh(obj, target_polys=self.target_polys, report=report)
        self.report({'INFO'}, f"Remeshed to {utils.get_face_count(obj)} faces")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BD_OT_remesh_quadriflow(bpy.types.Operator):
    """Apply Quadriflow remesh - clean quad topology"""
    bl_idname = "braindead.remesh_quadriflow"
    bl_label = "Quadriflow Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    target_faces: IntProperty(
        name="Target Faces",
        description="Target face count",
        default=10000,
        min=100,
        max=500000
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        remesh.apply_quadriflow_remesh(obj, target_faces=self.target_faces, report=report)
        self.report({'INFO'}, f"Remeshed to {utils.get_face_count(obj)} faces")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


# ============================================================================
# OPERATORS - CLEANUP
# ============================================================================

class BD_OT_fill_holes(bpy.types.Operator):
    """Fill holes in mesh"""
    bl_idname = "braindead.fill_holes"
    bl_label = "Fill Holes"
    bl_options = {'REGISTER', 'UNDO'}

    max_sides: IntProperty(
        name="Max Sides",
        description="Maximum edges for a hole to be filled (0 = all)",
        default=100,
        min=0,
        max=1000
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        filled = cleanup.fill_holes(obj, max_sides=self.max_sides, report=report)
        self.report({'INFO'}, f"Filled {filled} holes")
        return {'FINISHED'}


class BD_OT_remove_internal(bpy.types.Operator):
    """Remove internal/hidden faces"""
    bl_idname = "braindead.remove_internal"
    bl_label = "Remove Internal Geometry"
    bl_options = {'REGISTER', 'UNDO'}

    method: EnumProperty(
        name="Method",
        items=[
            ('RAYCAST', "Raycast", "Accurate ray-based detection"),
            ('SIMPLE', "Simple", "Fast built-in detection"),
        ],
        default='RAYCAST'
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        removed = cleanup.remove_internal_geometry(obj, method=self.method, report=report)
        self.report({'INFO'}, f"Removed {removed} internal faces")
        return {'FINISHED'}


class BD_OT_fix_manifold(bpy.types.Operator):
    """Fix non-manifold geometry"""
    bl_idname = "braindead.fix_manifold"
    bl_label = "Fix Non-Manifold"
    bl_options = {'REGISTER', 'UNDO'}

    aggressive: BoolProperty(
        name="Aggressive",
        description="Use aggressive fixing (may lose geometry)",
        default=False
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        cleanup.fix_non_manifold(obj, aggressive=self.aggressive, report=report)
        self.report({'INFO'}, "Non-manifold geometry fixed")
        return {'FINISHED'}


class BD_OT_triangulate(bpy.types.Operator):
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


# ============================================================================
# OPERATORS - NORMALS
# ============================================================================

class BD_OT_fix_normals(bpy.types.Operator):
    """Fix face normals to point outward"""
    bl_idname = "braindead.fix_normals"
    bl_label = "Fix Normals"
    bl_options = {'REGISTER', 'UNDO'}

    method: EnumProperty(
        name="Method",
        items=[
            ('BLENDER', "Blender", "Topology-based (best for clean meshes)"),
            ('DIRECTION', "Direction", "Center-based (fails on cavities)"),
            ('BOTH', "Both", "Try Blender first, then direction"),
        ],
        default='BOTH'
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        normals.fix_normals(obj, method=self.method, report=report)
        self.report({'INFO'}, "Normals fixed")
        return {'FINISHED'}


class BD_OT_verify_normals(bpy.types.Operator):
    """Check normal orientation"""
    bl_idname = "braindead.verify_normals"
    bl_label = "Verify Normals"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        report = []
        result = normals.verify_normals(obj, report=report)
        stats = normals.check_normal_orientation(obj)
        self.report({'INFO'}, f"Normals: {stats['outward']} outward, {stats['inward']} inward ({stats['inward_pct']:.1f}% inverted)")
        return {'FINISHED'}


# ============================================================================
# PANELS
# ============================================================================

class BD_PT_main(bpy.types.Panel):
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
            box.label(text=f"Mesh: {obj.name}", icon='MESH_DATA')
            row = box.row()
            row.label(text=f"Verts: {len(obj.data.vertices):,}")
            row.label(text=f"Faces: {len(obj.data.polygons):,}")
        else:
            layout.label(text="Select a mesh", icon='INFO')


class BD_PT_decimate(bpy.types.Panel):
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
        layout.operator("braindead.decimate_stylized", icon='MOD_DECIM')
        layout.separator()
        layout.operator("braindead.planar_decimate", icon='MOD_DECIM')
        layout.operator("braindead.collapse_decimate", icon='MOD_DECIM')
        layout.operator("braindead.mark_sharp_edges", icon='EDGESEL')


class BD_PT_remesh(bpy.types.Panel):
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
        layout.operator("braindead.remesh_sharp", icon='MOD_REMESH')
        layout.operator("braindead.remesh_voxel", icon='MESH_GRID')
        layout.operator("braindead.remesh_quadriflow", icon='MESH_GRID')


class BD_PT_cleanup(bpy.types.Panel):
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
        layout.operator("braindead.fill_holes", icon='MESH_PLANE')
        layout.operator("braindead.remove_internal", icon='SNAP_FACE')
        layout.operator("braindead.fix_manifold", icon='MOD_SOLIDIFY')
        layout.operator("braindead.triangulate", icon='MOD_TRIANGULATE')


class BD_PT_normals(bpy.types.Panel):
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
        layout.operator("braindead.fix_normals", icon='NORMALS_FACE')
        layout.operator("braindead.verify_normals", icon='ZOOM_ALL')


class BD_PT_colors(bpy.types.Panel):
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
        layout.operator("braindead.transfer_vertex_colors", icon='BRUSH_DATA')
        layout.operator("braindead.bake_vertex_colors", icon='TEXTURE')
        layout.operator("braindead.detect_color_edges", icon='EDGESEL')
        layout.operator("braindead.finalize_colors", icon='CHECKMARK')


# ============================================================================
# REGISTRATION
# ============================================================================

classes = [
    # Operators - Decimation
    BD_OT_decimate_stylized,
    BD_OT_planar_decimate,
    BD_OT_collapse_decimate,
    BD_OT_mark_sharp_edges,
    # Operators - Colors
    BD_OT_transfer_vertex_colors,
    BD_OT_bake_vertex_colors,
    BD_OT_detect_color_edges,
    BD_OT_finalize_colors,
    # Operators - Remesh
    BD_OT_remesh_sharp,
    BD_OT_remesh_voxel,
    BD_OT_remesh_quadriflow,
    # Operators - Cleanup
    BD_OT_fill_holes,
    BD_OT_remove_internal,
    BD_OT_fix_manifold,
    BD_OT_triangulate,
    # Operators - Normals
    BD_OT_fix_normals,
    BD_OT_verify_normals,
    # Panels
    BD_PT_main,
    BD_PT_decimate,
    BD_PT_remesh,
    BD_PT_cleanup,
    BD_PT_normals,
    BD_PT_colors,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
