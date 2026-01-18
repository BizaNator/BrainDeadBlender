# BrainDead Blender Tools

Comprehensive mesh processing and character pipeline tools for Blender by **BiloxiStudios Inc**.

## Features

- **Mesh Decimation** - Stylized low-poly reduction with color preservation
- **Vertex Color Transfer** - BVH-based color transfer between meshes with solid face support
- **Vertex Color Painting** - Paint colors on selected faces with favorites palette
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

**Windows (Run as Administrator):**
```cmd
mklink /D "C:\Users\YOUR_USER\AppData\Roaming\Blender Foundation\Blender\5.0\extensions\user_default\braindead_blender" "A:\Brains\Tools\BrainDeadBlender\braindead_blender"
```

**PowerShell (Run as Administrator):**
```powershell
New-Item -ItemType SymbolicLink -Path "C:\Users\YOUR_USER\AppData\Roaming\Blender Foundation\Blender\5.0\extensions\user_default\braindead_blender" -Target "A:\Brains\Tools\BrainDeadBlender\braindead_blender"
```

## Usage

### Blender UI

After installation, find the **BrainDead** tab in the 3D Viewport Sidebar (press `N`).

#### Panels

| Panel | Description |
|-------|-------------|
| **Decimation** | Stylized decimate, planar/collapse decimate, sharp edges |
| **Remesh** | Sharp, voxel, quadriflow remeshing |
| **Cleanup** | Fill holes, remove internal geometry, fix manifold |
| **Normals** | Fix and verify normals |
| **Vertex Colors** | Transfer, bake, paint, solidify, smooth |
| **UEFN Pipeline** | Character conversion tools |
| **Texture Project** | Project images to UV/texture |
| **Mask Colors** | Generate RGBA masks for Unreal |

---

## Vertex Colors

### Transfer Vertex Colors

Transfer colors from a high-poly mesh to a low-poly mesh using BVH nearest-point lookup.

**Workflow:**
1. Click the **source mesh** (with colors)
2. Ctrl+click the **target mesh** (receives colors)
3. Select **Transfer Mode**:
   - **Face (Solid)** - Each face gets ONE solid color (recommended)
   - **Vertex (Blended)** - Per-vertex colors, blended across faces
   - **Corner (Per-Loop)** - Each face-corner sampled independently
4. Enable **Apply Flat Shading** for solid face colors to display correctly
5. Click **Transfer Vertex Colors**

### Paint Colors (Edit Mode)

Paint solid colors on selected faces with an 8-slot favorites palette.

**Workflow:**
1. Enter **Edit Mode** (Tab)
2. Select faces (Face Select mode: press 3)
3. Pick a color or use **Eye Dropper** to sample from a face
4. Click **Paint** to apply color to selected faces
5. Save colors to favorites for quick access

**Features:**
- **Paint Color** - Current color to paint
- **Eye Dropper** - Sample color from selected face
- **Paint** - Apply color to selected faces
- **Favorites** (1-8) - Quick color palette
- **Save to** - Save current color to a favorite slot

### Adjust Colors (Object Mode)

| Operation | Description |
|-----------|-------------|
| **Solidify** | Make all face colors solid (no blending) |
| **Smooth** | Blend colors across the mesh |
| **Apply Flat Shading** | Required for solid face colors to display |
| **Convert Domain** | Switch between Corner (per face-corner) and Vertex (per vertex) |

### Adjust Colors (Edit Mode - Selected Faces)

Same operations but only affect selected faces:
- **Solidify Selected** - Solidify only selected faces
- **Smooth Selected** - Smooth only selected faces
- **Flat/Smooth Shading** - Change shading on selected faces

### Color Domains

| Domain | Description |
|--------|-------------|
| **Corner (CORNER)** | Each face-corner has its own color - allows solid face colors |
| **Vertex (POINT)** | Each vertex has one color shared by all faces - always blends |

Use **Convert Domain** to switch between them.

---

## Decimation

### Stylized Decimation Pipeline

1. **Planar Decimate** - Merge coplanar faces (angle threshold)
2. **Collapse Decimate** - Reduce to target face count
3. **Mark Sharp Edges** - Detect color boundaries as sharp edges

**Settings:**
- **Target Faces** - Final polygon count
- **Planar Angle** - Threshold for merging coplanar faces (degrees)
- **Sharp Angle** - Angle threshold for sharp edge detection

---

## Remeshing

| Mode | Description |
|------|-------------|
| **Sharp** | Octree-based, preserves thin geometry (fingers, ears) |
| **Voxel** | Creates watertight mesh, destroys thin geometry |
| **Voxel High** | High-resolution voxel then decimate |
| **Quadriflow** | Clean quad topology |

**Settings:**
- **Octree Depth** - Resolution for Sharp remesh (6-10)
- **Voxel Size** - Size for Voxel remesh
- **Target Polys** - Target for auto voxel size calculation

---

## Cleanup

| Operation | Description |
|-----------|-------------|
| **Fill Holes** | Fill open boundaries |
| **Remove Internal** | Remove hidden/internal faces |
| **Fix Manifold** | Fix non-manifold geometry |
| **Triangulate** | Convert n-gons to triangles |

---

## Normals

| Operation | Description |
|-----------|-------------|
| **Fix Normals** | Recalculate normals to point outward |
| **Verify Normals** | Check and report normal orientation |

**Methods:**
- **Blender** - Topology-based (default)
- **Direction** - Center-based
- **Both** - Try Blender first, then Direction if needed

---

## Python API

```python
from braindead_blender.mesh_ops import colors, remesh, cleanup, decimate, normals

# Transfer vertex colors with solid face mode
colors.transfer_vertex_colors(source_obj, target_obj, mode="FACE")

# Paint selected faces (edit mode)
colors.paint_selected_faces(obj, color=(1.0, 0.0, 0.0, 1.0))

# Sample color from face
color = colors.sample_face_color(obj)

# Solidify face colors
colors.solidify_face_colors(obj, method="DOMINANT")

# Apply flat shading
colors.apply_flat_shading(obj)

# Convert color domain
colors.convert_color_domain(obj, target_domain="CORNER")

# Stylized decimation
decimate.decimate_stylized(obj, target_faces=5000, planar_angle=7.0)

# Remesh with sharp edge preservation
remesh.apply_sharp_remesh(obj, octree_depth=8)

# Cleanup operations
cleanup.fill_holes(obj, max_sides=100)
cleanup.remove_internal_geometry(obj, method="RAYCAST")
normals.fix_normals(obj, method="BOTH")
```

---

## File Structure

```
BrainDeadBlender/
├── README.md
├── CLAUDE.md                   # Development notes
├── .gitignore
│
├── mesh_ops/                   # Modular mesh operations (standalone)
│   ├── __init__.py
│   ├── utils.py                # Logging, progress, helpers
│   ├── colors.py               # Vertex color operations
│   ├── remesh.py               # Remesh operations
│   ├── cleanup.py              # Cleanup/repair operations
│   ├── normals.py              # Normal operations
│   └── decimate.py             # Decimation operations
│
├── braindead_blender/          # Blender 4.2+ Extension
│   ├── blender_manifest.toml   # Extension manifest
│   ├── __init__.py             # Operators and panels
│   └── mesh_ops/               # Bundled mesh_ops package
│
├── scripts/                    # Standalone scripts
│   ├── uefn_pipeline/          # UEFN character conversion
│   ├── vertex_colors/          # Color baking scripts
│   └── utils/                  # Debug utilities
│
├── TransferVertexColors_v1.py  # Standalone color transfer
└── Decimate_v2.py              # Standalone decimation
```

---

## ComfyUI Integration

These tools integrate with [ComfyUI-BrainDead](https://github.com/BizaNator/ComfyUI-BrainDead) nodes for AI-powered character generation workflows.

---

## License

GPL-3.0-or-later

## Author

BiloxiStudios Inc
