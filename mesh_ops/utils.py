"""
mesh_ops.utils - Common Utilities

Shared helper functions for mesh operations.
"""

import bpy
import time

# ============================================================================
# LOGGING
# ============================================================================

def create_report():
    """Create a new report list for logging."""
    return []


def log(msg, report=None):
    """Log message to console and optionally to report list."""
    print(msg)
    if report is not None:
        report.append(msg)


def log_to_text(content: str, text_name: str = "MeshOps_Log.txt"):
    """Write log content to a Blender text block."""
    txt = bpy.data.texts.get(text_name)
    if not txt:
        txt = bpy.data.texts.new(text_name)
    txt.clear()
    txt.write(content)


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track and display progress for long operations."""

    def __init__(self, total_steps, description="Processing", show_progress=True, update_interval=10000):
        self.total = total_steps
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
        self.show_progress = show_progress
        self.update_interval = update_interval
        self.wm = bpy.context.window_manager

        if self.show_progress and self.total > 0:
            self.wm.progress_begin(0, self.total)
            print(f"\n[Progress] {description}: 0/{total_steps}")

    def update(self, step=None, message=None):
        """Update progress. Call frequently during long operations."""
        if step is not None:
            self.current = step
        else:
            self.current += 1

        if self.show_progress and self.total > 0:
            self.wm.progress_update(self.current)

            if self.update_interval > 0:
                if self.current - self.last_update >= self.update_interval:
                    elapsed = time.time() - self.start_time
                    pct = (self.current / self.total) * 100
                    eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
                    msg = message if message else self.description
                    print(f"[Progress] {msg}: {self.current}/{self.total} ({pct:.1f}%) - ETA: {eta:.1f}s")
                    self.last_update = self.current

    def finish(self, message=None):
        """Complete progress tracking."""
        if self.show_progress:
            self.wm.progress_end()

        elapsed = time.time() - self.start_time
        msg = message if message else self.description
        print(f"[Complete] {msg}: {elapsed:.2f}s")

        return elapsed


class StepTimer:
    """Context manager for timing individual steps."""

    def __init__(self, name, show_timing=True):
        self.name = name
        self.start = None
        self.show_timing = show_timing

    def __enter__(self):
        self.start = time.time()
        if self.show_timing:
            print(f"[Step] {self.name}...")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        if self.show_timing:
            print(f"[Step] {self.name}: {elapsed:.2f}s")


def step_timer(name, show_timing=True):
    """Create a step timer context manager."""
    return StepTimer(name, show_timing)


# ============================================================================
# OBJECT MODE HELPERS
# ============================================================================

def ensure_object_mode():
    """Ensure Blender is in object mode."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def ensure_edit_mode(obj):
    """Ensure object is active and in edit mode."""
    ensure_object_mode()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')


def depsgraph_update():
    """Force depsgraph update."""
    bpy.context.view_layer.update()


# ============================================================================
# MESH INFO
# ============================================================================

def get_face_count(obj):
    """Get polygon count from mesh object."""
    if obj is None or obj.type != 'MESH':
        return 0
    return len(obj.data.polygons)


def get_vertex_count(obj):
    """Get vertex count from mesh object."""
    if obj is None or obj.type != 'MESH':
        return 0
    return len(obj.data.vertices)


def get_edge_count(obj):
    """Get edge count from mesh object."""
    if obj is None or obj.type != 'MESH':
        return 0
    return len(obj.data.edges)


def get_mesh_stats(obj):
    """Get comprehensive mesh statistics."""
    if obj is None or obj.type != 'MESH':
        return None

    mesh = obj.data
    return {
        "name": obj.name,
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "faces": len(mesh.polygons),
        "loops": len(mesh.loops),
        "materials": len(mesh.materials),
        "uv_layers": len(mesh.uv_layers),
        "color_attributes": len(mesh.color_attributes) if hasattr(mesh, 'color_attributes') else 0,
    }


def print_mesh_stats(obj, report=None):
    """Print mesh statistics to console and report."""
    stats = get_mesh_stats(obj)
    if stats is None:
        log(f"[Stats] {obj.name if obj else 'None'}: Not a mesh", report)
        return

    log(f"[Stats] {stats['name']}:", report)
    log(f"         Vertices: {stats['vertices']:,}", report)
    log(f"         Edges: {stats['edges']:,}", report)
    log(f"         Faces: {stats['faces']:,}", report)
    log(f"         Materials: {stats['materials']}", report)
    log(f"         UV Layers: {stats['uv_layers']}", report)
    log(f"         Color Attrs: {stats['color_attributes']}", report)


# ============================================================================
# OBJECT SELECTION
# ============================================================================

def get_active_mesh():
    """Get the active mesh object, or None if not a mesh."""
    obj = bpy.context.active_object
    if obj and obj.type == 'MESH':
        return obj
    return None


def get_selected_meshes():
    """Get all selected mesh objects."""
    return [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']


def select_only(obj):
    """Select only the specified object."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def find_mesh_in_collection(collection_name):
    """Find the first mesh in a collection by name."""
    col = None
    for c in bpy.data.collections:
        if c.name.lower() == collection_name.lower():
            col = c
            break

    if col is None:
        return None

    for obj in col.all_objects:
        if obj.type == 'MESH':
            return obj

    return None


def find_meshes_in_collection(collection_name):
    """Find all meshes in a collection by name."""
    col = None
    for c in bpy.data.collections:
        if c.name.lower() == collection_name.lower():
            col = c
            break

    if col is None:
        return []

    return [obj for obj in col.all_objects if obj.type == 'MESH']
