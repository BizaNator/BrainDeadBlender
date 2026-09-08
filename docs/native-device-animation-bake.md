# Native Character Device animation bake

`scripts/utils/native_device_animation_batch.py` converts the studio's full,
88-track Fab animation FBXs into new animation FBXs for
`/Game/Creative/Devices/Mannequin/Meshes/CP_Device_Mannequin_Skeleton`.
It imports into a dedicated stock Unreal project, builds an explicit matched-pose
FK retargeter, bakes new sequences, preserves authored IK helper tracks, exports,
and checks fresh FBX imports at every sampled frame. It does not modify characters.

This is a separate operation from Blender's **Fortnite Skeleton Target → Convert
Copy to Target** and ComfyUI's **BD Native Fortnite Skeleton**. Those convert a
character's rest geometry and morphs. They do not rebake animation sequences.

## Why a skeleton compatibility flag is insufficient

The Fab and Device references have the same 88 names and parents, but different
reference transforms. Both live skeletons were measured with all 88 translation
modes set to `ANIMATION`. That does not disable Unreal's cross-skeleton rotation
remapping.

The BreakDance pilot reproduced the collapsed arms in stock UE 5.7.4 by evaluating
the Fab-bound sequence on the reconstructed native mesh: upper/lower arm and hand
rotations differed by about 27.108 degrees while joint positions stayed nearly
unchanged. A Blender render driven by those queried poses showed the folded arms.
The native-bound bake preserved the intended articulation. This experiment is
local engine evidence; actual UEFN playback remains a separate acceptance test.

Align the **retarget poses** before baking. Leaving the two different rest poses
unaligned changed the pilot's wrist positions by about 22 cm. Adding 27 degrees
to animated Euler angles is not a substitute for the measured correction across
the complete hierarchy.

## Requirements

- Stock Unreal **5.7.4-51494982** was used for the measured pilot. Other versions
  require their own verification; do not silently relax the job's version pin.
- Enable `PythonScriptPlugin`, `EditorScriptingUtilities`, and `IKRig` in a new
  project named `BDB_AnimationBake_<name>.uproject` or
  `BD_Native_Animation_Review.uproject`.
- Use a dedicated bake project. The script refuses a project with another name
  and checks the full project path. The batch API can create temporary packages
  at `/Game` before they are moved to the dedicated output root.
- Use the complete calibrated **named** native FBX and both measured reference
  captures. A raw export with unnamed bones is not a valid reference.
- Input FBXs must contain the complete, ordered 88-bone hierarchy and known
  absolute bone transforms. Partial tracks and unrelated rigs are rejected.
  The imported sequence must be non-additive, but FBX does not preserve all
  original Unreal metadata: that property alone cannot establish whether an
  original UAsset used additive animation. Supply the absolute bake, not an
  uncharacterized additive export. Notifies, curves, root-motion settings and
  other UAsset-only metadata need a separate source-aware migration.

On the studio NAS the references are:

```
B:\Brains\Skills\char-designer\skm_uefn_mannequin.FBX
B:\Brains\Skills\blender\braindead-stylized-base\assets\native_reference\
    CP_Device_Mannequin_named.fbx
    skm_uefn_mannequin_Skeleton.refpose.json
    CP_Device_Mannequin_Skeleton.refpose.json
```

## Job and execution

Set `BDB_ANIMATION_JOB` to a JSON file. Every input, including both reference
captures, has a SHA256 pin. Use new output directories and asset roots on each run.
Paths can be local Windows paths or Linux paths as appropriate to that process.

```json
{
  "project_file": "/absolute/path/BDB_AnimationBake_Test.uproject",
  "engine_version_prefix": "5.7.4-51494982",
  "asset_root": "/Game/BDB_AnimBake_Test_v01",
  "output_dir": "/absolute/path/bake_output_v01",
  "source_reference": {"path": "/path/skm_uefn_mannequin.FBX", "sha256": "FULL_SHA256"},
  "target_reference": {"path": "/path/CP_Device_Mannequin_named.fbx", "sha256": "FULL_SHA256"},
  "source_capture": {"path": "/path/skm_uefn_mannequin_Skeleton.refpose.json", "sha256": "FULL_SHA256"},
  "target_capture": {"path": "/path/CP_Device_Mannequin_Skeleton.refpose.json", "sha256": "FULL_SHA256"},
  "clips": [
    {"path": "/path/Emote_Breakdance_M1__fab.fbx", "sha256": "FULL_SHA256",
     "output_name": "Emote_Breakdance_M1__NativeDevice_v01"}
  ]
}
```

Linux example, with a virtual display and CPU-only Unreal:

```bash
BDB_ANIMATION_JOB=/absolute/path/job.json CUDA_VISIBLE_DEVICES='' \
  xvfb-run -a /path/to/UnrealEditor /absolute/path/BDB_AnimationBake_Test.uproject \
  -NullRHI -RenderOffscreen -Unattended -NoP4 -NoSound -NoSplash \
  -ExecutePythonScript=/absolute/path/native_device_animation_batch.py
```

Do not run the IK batch API as `-run=pythonscript`: the tested commandlet has no
Slate application and asserts inside `DuplicateAndRetarget`. A regular editor
with `-NullRHI` supplies Slate without allocating GPU rendering resources.
Keep FBX **Export Preview Mesh** disabled: GPU mesh skinning is unavailable in
NullRHI and its exporter path asserts. The animation-only FBX contains the rig
and animation needed for import onto the native skeleton.

An Unreal process can return exit code zero after a Python exception. Inspect
the error log and the `report.json` contents, rehash outputs, and inspect actual
motion; exit code alone is not an acceptance gate.

## What the build checks

1. Project/version identity, new destinations, all source hashes.
2. Fresh reference mesh imports against complete captured names, order, parents,
   local rotations, positions and scales. Reference tolerance is 0.002 cm,
   0.02 degrees, and 0.00002 scale.
3. One explicit, exactly mapped FK chain per bone. The target retarget pose uses
   measured quaternion offsets to align with the source; root and pelvis motion
   are retained. No IK solve or new secondary motion is introduced.
4. Authored `ik_*` local tracks are restored after the FK bake. In the pilot,
   even matched rotation poses alone moved an IK helper by 21.845 cm because
   the reference helper translations differed. This restoration preserves the
   source's existing helper behavior; it does not claim those helpers follow hands.
5. Every bone at every frame of a fresh FBX import is compared with the baked
   sequence. Maximum round-trip tolerance: 0.02 cm, 0.02 degrees, 0.0001 scale.
6. Root, pelvis, arm/hand, foot and IK-helper motion are also compared with the
   source at those tolerances. Full per-bone source deltas are retained in the
   report; small native finger-length differences remain visible there.
7. Source bytes are rehashed on completion.

The Flex source imports at 39 fps. Its first export/import shortened 244 frames
to 243 because the exported take end rounded just below a frame boundary. The
converter aligns the two generated FBX local stop fields upward on both the
legacy file clock and SDK 2020.2 clock (46,186,158,000 and 141,120,000 ticks per
second). The adjustment is limited to one microsecond; curve keys and all other
bytes are retained. Rounding only the legacy clock, or enabling the importer's
frame-snap option, did not resolve the measured case. Full frame-count and motion
read-back checks remain required after this adjustment.

The output's `complete` means the listed files passed the local build/read-back
checks. `native_runtime_verified` remains false. It does not mean UEFN accepted
the clip, the whole library was converted, or runtime retargeting is absent.

FBX filenames and timestamps can change byte hashes across reproducible runs.
Compare source pins and measured animation payloads, then freeze each delivered
FBX's own hash. A re-export is a new artifact.

## UEFN handoff

Coordinate with the seat holding the editor. Import a **new animation asset** into
the project's content root with the actual `CP_Device_Mannequin_Skeleton`
selected. Preserve the original Fab clip for comparison; do not update the native
skeleton's reference pose. Verify the current Skeleton property after import.

**Set `force_front_x_axis=False` explicitly in `FbxAnimSequenceImportData`.** The
first EROS pilot importer omitted this field and inherited the editor's saved
`True` preference. The resulting native asset had a 90-degree root rotation and
permuted local bone axes, despite the same FBX passing the lab read-back. Importing
the identical FBX as v02 with this flag false restored the intended live poses.
Do not infer pose parity from track counts or the selected Skeleton property.
Also set scene/unit conversion, zero import rotation/translation, unit import
scale and the intended sample rate explicitly. The measured local bake uses
`convert_scene=True`, `convert_scene_unit=True`, `preserve_local_transform=False`.

UEFN reports `FortniteGame.uproject` as its host. Check the mounted project content
root and active editor world, not a substring of `Paths.get_project_file_path()`.
Do not write new assets into Fortnite's shared `/Game` engine content.

Check the same delivered clip on the native character/device, authored character,
and NPC/player proof as applicable. Record the actual saved asset bindings, build
and session result, and visible motion through the full sequence, including floor
contact and fingers where observable. Labels such as `NATIVE` or a Verse
`Playing`/`Completed` state do not prove those properties.

The Blender version pin is independent of this animation bake. Keep the approved
character recipe pinned while a later Blender upgrade is validated separately.
