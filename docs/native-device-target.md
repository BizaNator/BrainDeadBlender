# Native Character Device target

New AutoRig settings and the Fortnite Skeleton Target panel default to **Native Character Device** (`CP_Device_Mannequin_Skeleton`). Fab is an explicit legacy option. Existing saved legacy selections are retained.

The measured reference FBX and its provenance default to `B:\Brains\Characters\_uefn_reference\native_contracts\codex_device_v01\` on Windows and the corresponding NAS mount on BRAINZ. Both paths are editable. Native loading validates the target asset path, FBX hash and loaded rest fingerprint.

**Set Rest Pose** creates a new native rig and mesh copy in Export, retains the original collection as Export_BeforeNative, and transforms every shape key using existing skin weights and full rest matrices including bone roll. It does not regenerate weights. A missing or changed reference stops export.

**Export UEFN FBX** exports a neutral combined visible mesh and the complete 87-data-bone rig with the exact `root` object name. It retains morphs and paint channels, uses `FBX_SCALE_ALL`, forward -Y / up Z, no leaf bones, and no animation bake. The CLI `scripts/utils/convert_skeleton_target.py` additionally exports all parts and performs actual FBX read-back checks.

`reference_from_ue_capture.py` reconstructs a named reference using complete editor-captured transforms and calibrated source FBX bone frames. `verify_ue_reference.py` runs in an isolated Unreal commandlet project to compare freshly imported local rest data with the captured target. Do not run its private-project `/Game` import paths in the owner's shared UEFN editor.

Validation: the native AutoRig integration test first failed on the old Fab default, then passed real conversion/export with original data unchanged, exact root name, all 30 finger weights, morphs and Nails. Missing-reference export deliberately fails without creating a file. The core conversion test checks identity, a roll-only change, every morph coordinate, driver targets, missing weighted bones and rejected native provenance. The native candidate's fresh UE5.7 import matches all 88 engine names/order/parents within the measured contract tolerance.

Runtime Device/NPC shared-animation acceptance remains separate. Archived scripts retain their legacy behavior. ComfyUI-BrainDead now exposes `BD_TargetFortniteSkeleton` to invoke the same completion CLI on a rigged FBX or blend; its full-character default requires all thirty finger weights. The prior autorig node is explicitly labeled Fab and can feed this completion step. No active ComfyUI service was reloaded. The studio skill documents pinned reproduction and the pilot evidence.

## Native Player target

`NATIVE_PLAYER` selects the captured 280 physical Player bones; Device keeps its
88-entry core. Both appear in **BrainDead > Fortnite Skeleton Target**, and in
**AutoRig > Rest Pose**. Select the target, load its reference/provenance files,
then **Convert Copy to Target** or **Set Rest Pose** on the Export collection.
Use BrainDead's native FBX export to retain the reference bind fields, including
the Player's non-unit unweighted physics leaf scale. Ordinary FBX export does
not provide that guarantee. All source meshes remain available.

The reference must be supplied by the user and identified by its source asset
path and SHA256; game assets are not included in the extension. Studio defaults
read `Characters/_uefn_reference/skeleton_reference_v1.json` (override with
`BDB_SKELETON_REFERENCE`). Its `authoring_derivatives` link each profile to a
reference FBX and provenance JSON by relative path and hash. The shared
`reference_paths.py` resolver is also used by the ComfyUI wrapper. Changed bytes,
missing entries and unsupported schemas are rejected; an absent manifest uses
the existing installation paths. Invalid defaults leave Blender's path fields
empty so the rest of the add-on remains available and a valid explicit reference
can be selected. See the portable
[Native Fortnite pipeline skill](../skills/native-fortnite-pipeline/SKILL.md)
for capture, comparison and runtime test instructions.

The character conversion controls preserve morphs and weights. Animation
conversion is a separate Unreal batch script, described in
[native-device-animation-bake.md](native-device-animation-bake.md).
