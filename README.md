# BrainDead Blender Tools

Comprehensive mesh processing and character pipeline tools for Blender by **BiloxiStudios Inc**.

## Features

- **Mesh Decimation** - Stylized low-poly reduction with color preservation
- **Vertex Color Transfer** - BVH-based color transfer between meshes
- **Remeshing** - Sharp (octree), Voxel, and Quadriflow methods
- **Mesh Cleanup** - Hole filling, internal geometry removal, manifold fixes
- **Normal Operations** - Fix inverted normals, verify orientation
- **UEFN Support** - Character pipeline tools for Fortnite/UEFN

## Installation

### Blender 4.2+ Extension (Recommended)

1. Download or clone this repository
2. In Blender: **Edit > Preferences > Add-ons**
3. Click **Install from Disk**
4. Navigate to `braindead_blender/` folder and select it
5. Enable "BrainDead Blender Tools"

### Development Install (Symlink)

Run as Administrator:
```cmd
mklink /D "C:\Users\YOUR_USER\AppData\Roaming\Blender Foundation\Blender\5.0\extensions\user_default\braindead_blender" "A:\Brains\Tools\BrainDeadBlender\braindead_blender"
```

## Usage

### Blender UI

After installation, find the **BrainDead** tab in the 3D Viewport Sidebar (press `N`).

Panels:
- **Decimation** - Stylized decimate, planar/collapse decimate, sharp edges
- **Remesh** - Sharp, voxel, quadriflow remeshing
- **Cleanup** - Fill holes, remove internal geometry, fix manifold
- **Normals** - Fix and verify normals
- **Vertex Colors** - Transfer, bake, detect color edges

### Python Scripts

```python
# Import mesh operations
from mesh_ops import colors, remesh, cleanup, decimate, normals

# Stylized decimation pipeline
decimate.decimate_stylized(obj, target_faces=5000, planar_angle=7.0)

# Transfer vertex colors between meshes
colors.transfer_vertex_colors(source_obj, target_obj)

# Remesh with sharp edge preservation
remesh.apply_sharp_remesh(obj, octree_depth=8)

# Cleanup operations
cleanup.fill_holes(obj, max_sides=100)
cleanup.remove_internal_geometry(obj, method="RAYCAST")
normals.fix_normals(obj, method="BOTH")
```

### Standalone Scripts

- `TransferVertexColors_v1.py` - Standalone vertex color transfer
- `Decimate_v2.py` - Full decimation pipeline orchestrator

## File Structure

```
BrainDeadBlender/
├── README.md
├── .gitignore
│
├── mesh_ops/                    # Modular mesh operations (standalone use)
│   ├── __init__.py
│   ├── utils.py                 # Logging, progress, helpers
│   ├── colors.py                # Vertex color operations
│   ├── remesh.py                # Remesh operations
│   ├── cleanup.py               # Cleanup/repair operations
│   ├── normals.py               # Normal operations
│   └── decimate.py              # Decimation operations
│
├── braindead_blender/           # Blender 4.2+ Extension
│   ├── blender_manifest.toml    # Extension manifest
│   ├── __init__.py              # Operators and panels
│   └── mesh_ops/                # Bundled mesh_ops package
│
├── TransferVertexColors_v1.py   # Standalone color transfer script
└── Decimate_v2.py               # Standalone decimation script
```

## ComfyUI Integration

These tools integrate with [ComfyUI-BrainDead](https://github.com/BizaNator/ComfyUI-BrainDead) nodes for AI-powered workflows.

## License

GPL-3.0-or-later

## Author

BiloxiStudios Inc
