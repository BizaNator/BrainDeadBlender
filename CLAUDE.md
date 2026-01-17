# BrainDead Blender Tools - Development Notes

**Repository**: https://github.com/BizaNator/BrainDeadBlender
**By**: BiloxiStudios Inc

## Project Summary

Comprehensive Blender tools for AI-generated character mesh processing, including:
- UEFN/Fortnite character pipeline (Mixamo → UEFN Manny skeleton)
- Mesh decimation with color preservation
- Vertex color operations (bake, transfer, masks)
- Mesh cleanup and repair

## Current State (2024-12-20)

**Working Pipeline: `Pipeline_v31.py`** - FULLY FUNCTIONAL

The pipeline successfully converts characters and imports into Unreal Engine 5.5+:

1. Scale matching - Target mesh scales to match UEFN mannequin height
2. Object alignment - Translate + yaw rotation to align armatures
3. Geometry bake - Apply armature modifier to freeze pose
4. Vertex groups - Create empty groups for all UEFN bones (required for weight transfer)
5. Weight transfer - Copy weights from UEFN mannequin via Data Transfer modifier
6. Armature binding - Transform vertices to armature local space, parent with ARMATURE type
7. Export collection - Create Export collection with correct hierarchy (armature at root)

### Export Hierarchy (CORRECT)
```
root [ARMATURE]              ← Skeleton root (no container empty!)
├── (bone) attach
├── (bone) pelvis
├── (bone) ik_foot_root
├── (bone) ik_hand_root
└── Mesh_0.001 [MESH]        ← Mesh as child of armature
```

## Resolved Issues

### Issue 1: `_end` bones in Unreal - SOLVED
- **Cause**: Blender FBX export "Add Leaf Bones" option
- **Solution**: Disable "Add Leaf Bones" in FBX export settings

### Issue 2: Export Hierarchy - SOLVED
- **Problem**: Container empty (SKM_0) was being included in skeleton hierarchy
- **Solution**: No container - armature at root level, mesh parented directly to armature
- **Key**: Mesh must be child of armature (not sibling) to match UEFN_Mannequin structure

### Issue 3: Scale Problems - SOLVED
- **Problem**: Model exported at 100x size when transforming vertices
- **Solution**: When transforming vertices to armature local space, only apply location/rotation, NOT scale
- **Code**: Decompose matrix and rebuild without scale component

## Expected Warnings (OK to ignore)

### Bind Pose Warning in Unreal
```
Warning: Imported skeletal mesh has some invalid bind poses.
Skeletal mesh skinning has been rebind using the time zero pose.
```
- **This is expected** when transferring weights to different geometry
- The source mesh (UEFN_Mannequin) has different vertex positions than target mesh
- Unreal automatically recalculates bind pose - character works correctly
- Animations play properly despite the warning

### Smoothing Groups Warning
```
Warning: No smoothing group information was found for this mesh
```
- **Solution**: In FBX export, set Smoothing to "Face" or "Edge" instead of "Normals Only"

## Key Technical Discoveries

### Bone Convention Differences
UEFN and H3D/Mixamo have fundamentally different bone orientations:

| Skeleton | Convention |
|----------|------------|
| **UEFN** | All bones point +Y in armature local space |
| **H3D/Mixamo** | Anatomical (spine=+Z, legs=-Z, arms=±X) |

**Critical Insight**: Both represent valid T-pose/A-pose, just with ~90° different internal bone directions. Rotating pose bones to match directions causes mesh deformation - so we skip bone rotation entirely and rely on weight transfer.

### Weight Transfer Requirements
Data Transfer modifier only transfers to **existing** vertex groups with matching names. Must create empty vertex groups for all UEFN bones before transfer.

### Bind Pose Requirements
For correct bind pose in FBX export:
1. Transform mesh vertices from world space to armature local space
2. Only apply location + rotation, NOT scale (prevents 100x size issue)
3. Parent mesh to armature with `type='ARMATURE'` (not `type='OBJECT'`)
4. Use `keep_transform=False` so mesh has identity local transform

## File Structure

```
BrainDeadBlender/                    # A:\Brains\Tools\BrainDeadBlender
├── README.md                        # User documentation
├── CLAUDE.md                        # This file - development notes
├── .gitignore
│
├── braindead_blender/               # Blender 4.2+ Extension
│   ├── blender_manifest.toml        # Extension manifest
│   ├── __init__.py                  # Operators and panels
│   └── mesh_ops/                    # Bundled mesh operations
│
├── mesh_ops/                        # Modular mesh operations (standalone)
│   ├── __init__.py                  # Package exports
│   ├── utils.py                     # Logging, progress, helpers
│   ├── colors.py                    # Vertex color ops
│   ├── remesh.py                    # Remesh ops
│   ├── cleanup.py                   # Cleanup/repair ops
│   ├── normals.py                   # Normal ops
│   └── decimate.py                  # Decimation ops
│
├── Decimate_v2.py                   # Modular decimation orchestrator
├── TransferVertexColors_v1.py       # Standalone color transfer
│
├── scripts/
│   ├── Decimate_v1.py               # Original monolithic decimate
│   │
│   ├── uefn_pipeline/               # UEFN Character Pipeline
│   │   ├── Pipeline_v31.py          # Skeleton conversion (WORKING)
│   │   ├── ModularBody_v1.py        # Hands/feet/head attachment
│   │   ├── Segmentation_v1.py       # Body segmentation
│   │   ├── TransferBones_v1.py      # Bone transfer
│   │   ├── ExportUEFN_v1.py         # FBX export
│   │   ├── blender_convert_mixamo_to_uefn.py
│   │   └── blender_export_uefn.py
│   │
│   ├── vertex_colors/               # Vertex Color Scripts
│   │   ├── VertexColors_v1.py       # Texture to vertex color baking
│   │   ├── TextureProject_v1.py     # Image projection to UV/texture
│   │   └── MaskColors_v1.py         # RGBA mask generation
│   │
│   └── utils/                       # Utility Scripts
│       ├── debug_bone_axes.py       # Bone orientation debug
│       ├── check_hierarchy.py       # Hierarchy verification
│       ├── UEFN_Hierarchy.py        # Hierarchy utilities
│       └── uefn_convert_node.py     # ComfyUI node helper
│
└── archive/                         # Old versions (reference only)
    ├── Pipeline_v27-30.py
    ├── Pipeline_v31_working.py
    └── FullPipeline_v20.py
```

---

## ModularBody_v1.py - Modular Character Pipeline

### Purpose
Attach detailed hand/foot meshes to AI-generated bodies and separate heads for Mutable modularity.

### Key Insight
No separate rigging needed! The UEFN skeleton already has finger/toe bones. We just:
1. Merge detailed geometry into body
2. Transfer weights from SKM_UEFN_Mannequin

### Operations
| Mode | Description |
|------|-------------|
| `hands` | Attach detailed hand meshes |
| `feet` | Attach detailed foot meshes |
| `head` | Separate head from body |
| `all` | Run all operations |

### Scene Setup
```
Body Collection:
├── [UEFN Armature]     # From Pipeline_v31 output
└── [Body Mesh]         # With existing UEFN weights

Hands Collection:
├── Hand_L              # Origin at wrist center!
└── Hand_R              # Origin at wrist center!

Feet Collection:
├── Foot_L              # Origin at ankle center!
└── Foot_R              # Origin at ankle center!

Source Collection:
└── SKM_UEFN_Mannequin  # Weight reference
```

### Usage
```python
# In Blender Text Editor, run:
from ModularBody_v1 import main, MODE_ATTACH_HANDS, MODE_ALL

main(MODE_ATTACH_HANDS)  # Hands only
main(MODE_ALL)           # All operations
```

### Workflow
1. Cut existing hand/foot by weight threshold (vertices weighted to hand/foot bones)
2. Position new mesh at bone location (hand_l, hand_r, foot_l, foot_r)
3. Join meshes
4. Merge vertices by distance (seam cleanup)
5. Transfer weights from UEFN mannequin

### Important
- **Set mesh origin at wrist/ankle** before running script
- Hand meshes should be roughly positioned/scaled to match body
- Log output saved to `ModularBody_V1_Log.txt`

---

## Segmentation_v1.py - Body Segmentation Pipeline

### Purpose
Create body segments from bone vertex groups for texturing, selection, and modularity.

### Features
| Feature | Description |
|---------|-------------|
| Materials | Colored materials per segment = **PolyGroups/Sections in Unreal** |
| UV Islands | Seams at segment boundaries, unwrap, pack islands |
| Separate Objects | Split mesh into separate objects (optional, destructive) |

### Segments (14 total)
- **Head**: head, neck_01, neck_02
- **Torso**: pelvis, spine_01-05
- **UpperArm_L/R**: clavicle, upperarm, upperarm_twist_01
- **LowerArm_L/R**: lowerarm, lowerarm_twist_01
- **Hand_L/R**: hand + all finger bones
- **UpperLeg_L/R**: thigh, thigh_twist_01
- **LowerLeg_L/R**: calf, calf_twist_01
- **Foot_L/R**: foot, ball

### Usage
```python
from Segmentation_v1 import main, MODE_ALL, MODE_MATERIALS, MODE_UV_ISLANDS, MODE_SEAMS_ONLY, MODE_SEPARATE

# After running Pipeline_v31:
main(MODE_ALL)        # Materials + UVs (recommended)
main(MODE_MATERIALS)  # Materials only (= Unreal sections/polygroups)
main(MODE_UV_ISLANDS) # UV unwrap with segment seams
main(MODE_SEAMS_ONLY) # Just mark seams, no unwrap
main(MODE_SEPARATE)   # Split into separate objects (destructive!)
```

### Requirements
- Run **after** Pipeline_v31 (mesh must have UEFN bone vertex groups)
- Mesh must be in "Export" collection (or specify custom collection)

### How It Works
1. Maps UEFN bone names to segment definitions
2. For each face, sums vertex weights per segment
3. Assigns face to segment with highest total weight
4. Creates requested outputs (face maps, materials, UVs)

---

## TextureProject_v1.py - Image Projection to UV + Texture

### Purpose
Project 4 reference images (front/back/left/right) onto a mesh, creating a new UV map and baking to an actual texture. Use this BEFORE VertexColors to preview and refine the projection.

### Workflow
```
Reference Images → TextureProject → Preview/Edit → VertexColors → MaskColors → FBX
                   (creates texture)  (optional)    (bake to verts)
```

### Usage
```python
from TextureProject_v1 import project_to_texture, print_loaded_images

# Step 1: Load images into Blender (drag-drop or File > Import)

# Step 2: See available images
print_loaded_images()

# Step 3: Project to texture
project_to_texture(
    front="front",
    back="back",
    left="left",
    right="right"
)
```

### Output
| Output | Description |
|--------|-------------|
| `ProjectedUV` | New UV map using box projection from best view per face |
| `ProjectedTexture` | Baked texture image (default 2048x2048) |
| `M_ProjectedTexture` | Preview material using the new UV + texture |

### Configuration
| Setting | Default | Description |
|---------|---------|-------------|
| `OUTPUT_RESOLUTION` | 2048 | Output texture size (square) |
| `PROJECTION_MODE` | "BOX" | Each face uses best view based on normal |
| `UV_ISLAND_MARGIN` | 0.02 | Padding between UV islands |
| `SAMPLE_METHOD` | "BILINEAR" | Texture sampling interpolation |

### How It Works
1. For each face, determine best view based on face normal
2. Create UV coordinates by projecting vertices from that view
3. Rasterize each triangle to output texture
4. Sample source images at projected positions
5. Create preview material

### When to Use
- **Before VertexColors**: Preview projection quality before committing to vertex colors
- **Texture Editing**: Output can be edited in Photoshop/GIMP before final bake
- **Higher Quality**: Texture provides more detail than vertex colors on low-poly mesh
- **UV-Based Workflow**: When you need actual UVs for other purposes

---

## VertexColors_v1.py - Texture to Vertex Color Baking

### Purpose
Bake texture colors to vertex colors for low-poly stylized characters.
Two modes: single texture sampling OR multi-image projection.

### Usage

**Mode A: Single Texture + UVs**
```python
from VertexColors_v1 import main

# Option 1: Provide image path
main(image_path="C:/textures/character.png")

# Option 2: Auto-detect from material's texture
main()
```

**Mode B: Multi-Image Projection (4 views)**
```python
from VertexColors_v1 import project_images, print_loaded_images

# Step 1: Load images into Blender (drag-drop, or File > Import)

# Step 2: See what images are available
print_loaded_images()

# Step 3: Project using Blender image names
project_images(
    front="front",      # Just the image name, not full path!
    back="back",
    left="left",
    right="right"
)
```

### How Multi-Image Projection Works
1. Calculates mesh bounding box
2. For each vertex, determines which views can "see" it based on normal direction
3. Projects vertex position to each view's image UV
4. Samples color from each visible view
5. Blends colors with weights based on normal alignment
6. Assigns blended color to vertex

### Color Assignment Modes
| Mode | Setting | Description |
|------|---------|-------------|
| **Hard Face** | `HARD_FACE_COLORS = True` | Each face gets ONE solid color from best view (no blending) |
| **Blend** | `HARD_FACE_COLORS = False` | Per-vertex blending from multiple views (softer but noisier) |

**Recommended**: Use `HARD_FACE_COLORS = True` for cleaner masks and contiguous color regions.

### Viewport Preview
- Segmentation materials are set up to use vertex colors
- After running VertexColors, viewport shows actual baked colors
- Works in both Solid and Material Preview modes

### FBX Export
- Enable "Vertex Colors" in Geometry section

### Unreal Usage
- In material, use "Vertex Color" node to access the colors
- Connect RGB output to Base Color

### Vertex Color Channels
- Full RGBA per vertex (8 bits × 4 = 32-bit)
- RGB = 16.7 million colors (NOT just 4!)
- Alpha available for masks/effects

---

## MaskColors_v1.py - Vertex Color Mask Generation

### Purpose
Create RGBA mask channels from existing vertex colors for Unreal material customization.
Allows designers to customize Primary, Secondary, and Accent colors in material instances.

### Channel Mapping (for Unreal Materials)
| Channel | Purpose | Unreal Usage |
|---------|---------|--------------|
| R (Red) | Primary color mask | Designer can customize primary color |
| G (Green) | Secondary color mask | Designer can customize secondary color |
| B (Blue) | Accent color mask | Designer can customize accent color |
| A (Alpha) | Emissive mask | Glow intensity multiplier |
| (0,0,0,0) | Base/Unmasked | Uses base parameter, no customization |

### Modes
| Mode | Description |
|------|-------------|
| `ANALYZE` | Show color distribution report (no changes) |
| `AUTO_MASK` | K-means cluster colors, assign by size rank |
| `MANUAL_MASK` | User defines specific colors → channels |
| `MATERIAL_MASK` | Use material slots for mapping |

### Usage
```python
# Configure MODE at top of script, then run:

# Mode 1: Analyze colors (no changes)
MODE = "ANALYZE"

# Mode 2: Auto-cluster and assign by rank
MODE = "AUTO_MASK"
CHANNEL_MAPPING = {
    "BASE": 0,       # Largest cluster → unmasked
    "PRIMARY": 1,    # 2nd largest → R channel
    "SECONDARY": 2,  # 3rd largest → G channel
    "ACCENT": 3,     # 4th largest → B channel
    "EMISSIVE": None # Manual only
}

# Mode 3: Manual color mapping
MODE = "MANUAL_MASK"
MANUAL_COLOR_MAP = {
    (0.2, 0.4, 0.8): "PRIMARY",   # Blue → Primary
    (0.8, 0.3, 0.3): "SECONDARY", # Red → Secondary
}

# Mode 4: Map by material slots
MODE = "MATERIAL_MASK"
MATERIAL_CHANNEL_MAP = {
    "M_Torso": "PRIMARY",
    "M_Eyes": "EMISSIVE",
}
```

### Workflow
```
Pipeline_v31 → Segmentation_v1 → VertexColors_v1 → MaskColors_v1 → FBX Export
                                 (colors)          (masks)
```

### Unreal Material Setup
```
Vertex Color (Mask layer)
├── R → Lerp(BaseColor, PrimaryColor, R)
├── G → Lerp(result, SecondaryColor, G)
├── B → Lerp(result, AccentColor, B)
└── A → Emissive multiplier
```

### Algorithm (AUTO_MASK mode)
1. Extract vertex colors from "Col" layer
2. K-means cluster into N groups
3. Rank clusters by vertex count
4. Assign: Largest=BASE, 2nd=PRIMARY, 3rd=SECONDARY, 4th=ACCENT
5. Write to "Mask" color layer

### Face-Based Masks (Contiguous Regions)
| Setting | Description |
|---------|-------------|
| `FACE_BASED_MASKS = True` | All vertices of a face get SAME mask (contiguous regions) |
| `FACE_BASED_MASKS = False` | Per-vertex assignment (can create fragmented masks) |

Face color methods when `FACE_BASED_MASKS = True`:
| Method | Description |
|--------|-------------|
| `DOMINANT` | Most common color among face vertices (recommended for solid colors) |
| `AVERAGE` | Average of all face vertex colors (good for gradients) |
| `CENTER` | First vertex color (fastest) |

**Recommended**: Use `FACE_BASED_MASKS = True` with `FACE_COLOR_METHOD = "DOMINANT"` for clean, contiguous mask regions.

### Future: SAM Integration
Placeholder hooks for Segment Anything Model:
- Semantic segmentation: buttons, shoelaces, fingernails, buckles
- Map semantic categories to mask channels

---

## TransferVertexColors_v1.py - Vertex Color Transfer

### Purpose
Transfer vertex colors from a high-poly source mesh to a low-poly target mesh using BVH nearest-point lookup. Essential for recovering colors lost during remeshing or decimation.

### Use Cases
- Mesh lost vertex colors during remeshing/decimation
- Retopologized mesh needs colors from original
- Baked vertex colors to high-poly, need them on low-poly version

### Algorithm
1. Build BVH tree from source mesh polygons
2. For each face on target mesh, find closest point on source
3. Get color from source face at that point
4. Apply same color to all loops of target face (flat shading)

### Usage
```python
from TransferVertexColors_v1 import transfer_colors, transfer_colors_by_name, transfer_colors_selected

# Method 1: By object reference
transfer_colors(source_obj, target_obj)

# Method 2: By object name
transfer_colors_by_name("HighPoly_Mesh", "LowPoly_Mesh")

# Method 3: Select both meshes (target = active), then:
transfer_colors_selected()
```

### Source Color Detection
Automatically finds color attributes in this priority order:
1. `Col` (standard Unreal/FBX name)
2. `BakedColors`
3. `TransferredColors`
4. First available color attribute

### Output
- Creates `Col` color attribute on target mesh
- Sets as active render color for FBX export
- Optionally creates debug material to preview colors

### Configuration
| Setting | Default | Description |
|---------|---------|-------------|
| `TRANSFER_MODE` | "FACE" | "FACE" = flat shading, "VERTEX" = smooth blending |
| `CREATE_DEBUG_MATERIAL` | True | Create material to preview transferred colors |
| `OUTPUT_COLOR_NAME` | "Col" | Name of color attribute to create |

---

## Decimate_v1.py - Stylized Low-Poly Reduction

### Purpose
Reduces high-poly meshes to stylized low-poly with hole filling, internal geometry removal, and color preservation.

### Pipeline Steps
1. **[Optional] Remesh** - Clean topology, fill holes, remove internal geo
2. **[Optional] Fill holes** - Close remaining open edges
3. **[Optional] Remove internal geometry** - Ray-based hidden face detection
4. **Planar decimate** - Merge coplanar faces (within angle threshold)
5. **Collapse decimate** - Reduce to target poly count
6. **Mark sharp edges** - For hard edge rendering
7. **Cleanup** - Merge verts, triangulate n-gons

### Remesh Modes
| Mode | Description | Best For |
|------|-------------|----------|
| `NONE` | Skip remesh | Already-clean meshes |
| `SHARP` | Octree remesh with sharp edge preservation | **Thin geometry** (lips, ears, fingers) |
| `VOXEL` | Moderate resolution voxel remesh | Simple solid props |
| `VOXEL_HIGH` | High-res voxel then decimate | Solid geometry with cavities |
| `QUAD` | Quadriflow (smooth quad flow) | Smooth surfaces |

### Key Configuration
```python
REMESH_MODE = "SHARP"           # Recommended for characters
TARGET_FACES = 5000             # Target poly count
PLANAR_ANGLE = 7.0              # Angle for coplanar merging
SHARP_ANGLE = 14.0              # Angle for sharp edge marking
PRESERVE_COLORS_THROUGH_REMESH = True  # Bake/transfer colors
BAKE_VERTEX_COLORS = True       # Bake texture to vertex colors
DETECT_COLOR_EDGES = True       # Mark color boundaries as sharp
```

### Color Preservation Workflow
1. Bakes texture to vertex colors BEFORE remesh (`BakedColors`)
2. Creates hidden reference copy of colored mesh
3. After remesh/decimate, transfers colors from reference via BVH lookup
4. Renames to `Col` for FBX export

### Internal Geometry Removal
| Method | Description |
|--------|-------------|
| `SIMPLE` | Blender's select_interior_faces (fast) |
| `RAYCAST` | Cast rays from multiple directions (accurate) |

### Usage
```python
# Configure settings at top of script, then:
from Decimate_v1 import main
main()

# Or process specific mesh:
TARGET_MESH = "MyCharacter"
main()
```

### Note: Refactor Needed
This script is large (~900 lines) and contains multiple independent functions that should be separated into individual modules for:
- ComfyUI node integration
- Blender addon panel
- Selective function calls

**Planned Modules:**
- `MeshRemesh.py` - Remesh operations (voxel, sharp, quad)
- `MeshCleanup.py` - Hole filling, internal removal, manifold fixes
- `MeshDecimate.py` - Planar and collapse decimation
- `MeshEdges.py` - Sharp edge detection and marking
- `ColorPreserve.py` - Color baking and transfer (partially done in TransferVertexColors_v1.py)

---

## Scene Setup Requirements

### Source Collection
- `root` - UEFN Manny armature
- `SKM_UEFN_Mannequin` - Reference mesh with proper weights

### Target Collection
- `Armature` - AI-generated rig (H3D/Mixamo style)
- `Mesh_0.001` - AI-generated character mesh

## Bone Mapping Reference

```python
BONE_MAP = {
    "pelvis": "Hips",
    "spine_01": "Spine",
    "spine_02": "Spine1",
    "spine_03": "Spine2",
    "neck_01": "Neck",
    "head": "Head",
    "clavicle_l": "LeftShoulder",
    "upperarm_l": "LeftArm",
    "lowerarm_l": "LeftForeArm",
    "hand_l": "LeftHand",
    "clavicle_r": "RightShoulder",
    "upperarm_r": "RightArm",
    "lowerarm_r": "RightForeArm",
    "hand_r": "RightHand",
    "thigh_l": "LeftUpLeg",
    "calf_l": "LeftLeg",
    "foot_l": "LeftFoot",
    "ball_l": "LeftToeBase",
    "thigh_r": "RightUpLeg",
    "calf_r": "RightLeg",
    "foot_r": "RightFoot",
    "ball_r": "RightToeBase",
}
```

## FBX Export Settings (Recommended)

- **Selected Objects**: Enabled
- **Object Types**: Armature, Mesh
- **Apply Scalings**: FBX All
- **Forward**: -Y Forward
- **Up**: Z Up
- **Apply Unit**: Enabled
- **Smoothing**: Face (or Edge) - NOT "Normals Only"
- **Add Leaf Bones**: **DISABLED** (critical!)
- **Armature**:
  - Primary Bone Axis: Y
  - Secondary Bone Axis: X

## Commands to Resume

1. Open `UEFN_Character_Pose_and_Prepare3.blend`
2. Ensure Source and Target collections are set up
3. Run `Pipeline_v31.py` from Text Editor
4. Check log in `Pipeline_V31_Log.txt`
5. Export from Export collection with settings above
6. Import to Unreal - bind pose warning is expected and OK

## Verification Commands (Blender Python Console)

### Check Export Hierarchy
```python
import bpy

def print_hierarchy(obj, indent=0):
    prefix = "  " * indent
    obj_type = obj.type
    if obj_type == 'ARMATURE':
        print(f"{prefix}{obj.name} [{obj_type}]")
        for bone in obj.data.bones:
            if bone.parent is None:
                print(f"{prefix}  (bone) {bone.name}")
    else:
        print(f"{prefix}{obj.name} [{obj_type}]")
    for child in obj.children:
        print_hierarchy(child, indent + 1)

for obj in bpy.data.collections['Export'].objects:
    if obj.parent is None:
        print("=== EXPORT HIERARCHY ===")
        print_hierarchy(obj)
```

## Failed Approaches (Don't Repeat)

1. **Pose bone rotation to match directions** - Causes mesh to ball up due to ~90° bone convention difference
2. **Manual `matrix_parent_inverse` setting** - Causes scale issues, use `parent_set` operator instead
3. **Zeroing armature transforms when parenting to container** - Breaks scale, armature is in centimeters
4. **Container empty in export hierarchy** - Gets included in Unreal skeleton tree, use armature at root instead
5. **Full matrix inverse for vertex transform** - Includes scale, causes 100x size; decompose and exclude scale
6. **`type='OBJECT'` parenting** - Doesn't establish proper bind pose; use `type='ARMATURE'` instead

## mesh_ops Package - Modular Mesh Operations

### Purpose
Modular package containing standalone mesh operation functions for:
- Direct use in Blender Python scripts
- ComfyUI node integration (via ComfyUI-BrainDead)
- Blender addon panels (braindead_blender extension)

### Modules

| Module | Functions |
|--------|-----------|
| `utils` | `ensure_object_mode`, `get_face_count`, `ProgressTracker`, `log` |
| `colors` | `transfer_vertex_colors`, `bake_texture_to_vertex_colors`, `detect_color_edges`, `finalize_color_attribute` |
| `remesh` | `apply_sharp_remesh`, `apply_voxel_remesh`, `apply_quadriflow_remesh` |
| `cleanup` | `fill_holes`, `remove_internal_geometry`, `fix_non_manifold`, `triangulate_ngons` |
| `normals` | `fix_normals`, `check_normal_orientation`, `verify_normals` |
| `decimate` | `apply_planar_decimate`, `apply_collapse_decimate`, `mark_sharp_edges`, `decimate_stylized` |

### Usage
```python
from mesh_ops import colors, remesh, cleanup, decimate

# Individual operations
remesh.apply_sharp_remesh(obj, octree_depth=8)
cleanup.fill_holes(obj, max_sides=100)
decimate.apply_planar_decimate(obj, angle=7.0)
decimate.apply_collapse_decimate(obj, target_faces=5000)
colors.transfer_vertex_colors(source_obj, target_obj)

# Full stylized pipeline
decimate.decimate_stylized(obj, target_faces=5000, planar_angle=7.0)
```

### Function Signatures
All functions follow consistent pattern:
```python
func(obj, report=None, **config) -> result
```
- `obj`: Blender mesh object
- `report`: Optional list for logging messages
- Returns: Depends on function (face count, success bool, etc.)

---

## braindead_blender - Blender Extension

### Purpose
Blender 4.2+ extension providing UI panels and operators for mesh operations.

### Installation
1. Symlink or copy `braindead_blender/` folder to Blender extensions
2. In Blender: Edit > Preferences > Add-ons
3. Enable "BrainDead Blender Tools"

**Symlink (recommended for development):**
```cmd
mklink /D "C:\Users\USERNAME\AppData\Roaming\Blender Foundation\Blender\5.0\extensions\user_default\braindead_blender" "A:\Brains\Tools\BrainDeadBlender\braindead_blender"
```

### Panels (View3D > Sidebar > BrainDead)
- **Decimation**: Stylized decimate, planar/collapse decimate, mark sharp edges
- **Remesh**: Sharp remesh, voxel remesh, quadriflow
- **Cleanup**: Fill holes, remove internal geometry, fix manifold, triangulate
- **Normals**: Fix normals, verify normals
- **Vertex Colors**: Transfer colors, bake from texture, detect color edges, finalize

### Operators
| Operator | Description |
|----------|-------------|
| `braindead.decimate_stylized` | Full stylized decimation pipeline |
| `braindead.transfer_vertex_colors` | BVH-based color transfer |
| `braindead.remesh_sharp` | Octree remesh (preserves thin geometry) |
| `braindead.remesh_voxel` | Voxel remesh (watertight) |
| `braindead.remesh_quadriflow` | Quadriflow remesh (clean quads) |
| `braindead.fill_holes` | Fill mesh holes |
| `braindead.remove_internal` | Remove hidden faces |
| `braindead.fix_normals` | Fix inverted normals |
| `braindead.mark_sharp_edges` | Mark edges by angle |
| `braindead.bake_vertex_colors` | Bake texture to vertex colors |
| `braindead.detect_color_edges` | Mark color boundaries |

### Extension Manifest
Located at `braindead_blender/blender_manifest.toml`. Required for Blender 4.2+ extension system.

---

## ComfyUI Integration

### Location
Nodes are in [ComfyUI-BrainDead](https://github.com/BizaNator/ComfyUI-BrainDead) repo:
- `nodes/blender/decimate.py`
- `nodes/blender/remesh.py`
- `nodes/blender/repair.py`
- `nodes/blender/transfer.py`

### Architecture
ComfyUI nodes embed Blender Python scripts that use mesh_ops functions.
Meshes are passed via temporary PLY/GLB files.

### V3 Node Format
Following [ComfyUI V3 migration](https://docs.comfy.org/custom-nodes/v3_migration):
```python
class BD_BlenderDecimate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderDecimate",
            inputs=[io.Mesh.Input("mesh"), io.Int.Input("target_faces")],
            outputs=[io.Mesh.Output()]
        )

    @classmethod
    def execute(cls, mesh, target_faces) -> io.NodeOutput:
        # Run Blender script with mesh_ops
        return io.NodeOutput(result_mesh)
```

---

## Future Improvements (TODO)

1. Weight painting refinement for edge cases (shoulders, hips)
2. Batch processing multiple characters
3. ~~Blender addon UI panel~~ ✅ Done (braindead_blender extension)
4. ~~Modular mesh operations~~ ✅ Done (mesh_ops package)
5. ComfyUI V3 node migration
