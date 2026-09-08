---
name: native-fortnite-pipeline
description: Author and inspect native Fortnite skeleton contracts, convert legacy animation cohorts, and prepare same-clip Device, NPC and Player tests using Blender and Unreal or UEFN automation.
---

# Native Fortnite character and animation pipeline

Make the character reference pose and the animation library agree with the actual
native target. A matching bone list or assigning another Skeleton asset does not
convert an animation. Keep character repose and animation rebaking as distinct
operations. The owner may request either, or both.

## Establish the contract

Read the current project and requested native Skeleton from the live editor.
UEFN's host project path can be `FortniteGame.uproject`; validate the mounted
project content root and a known project asset instead of asserting that the
host filename contains the island name.

Use [scripts/capture_reference.py](scripts/capture_reference.py) inside the editor
to capture named parents, local and component reference transforms, scale, engine
version and physical/virtual counts. It only reads assets. Supply the asset and
output paths; it requires no studio drive letters or external service. Use
[scripts/compare_reference.py](scripts/compare_reference.py) outside the editor
to compare the resulting captures.

For the measured September 2026 setup, the Character Device has 88 physical
hierarchy entries; Player has 280 physical bones plus 30 virtual AnimPose entries.
Their shared body core agrees. Fab's 88-name mannequin has a different reference
pose. These are versioned measurements, not permanent engine guarantees. See
[references/measured-contract.md](references/measured-contract.md).

Author physical bones only. Preserve the exact native core's names, parents,
transforms and scale when adding the Player's physical extensions. Virtual bones
are engine-derived entries; sockets are separately authored attachment metadata.
Do not manufacture either category from a prefixed name or an aggregate count.
Require the user's own reference files; this skill does not redistribute game
assets.

## Choose the operation

- **Character mesh:** construct in the native pose or repose geometry and every
  shape key with the measured source-to-target transforms. Retain skin weights,
  UVs, materials and the intended modular boundaries. Reimport the FBX into a new
  Skeleton first to measure the incoming reference; assigning the canonical
  Skeleton during import can conceal a mismatch in the incoming mesh.
- **Animation:** bake a versioned copy against an explicit source mesh, target
  mesh and retargeter. Inspect the actual clip Skeleton, source reference and
  per-clip retarget settings. A label such as `__fab` is not that inspection.
- **Library:** inventory all sequences and dependency assets by source skeleton.
  Use one calibrated conversion per compatible family. Human, animal, prop,
  face-only and additive assets need different handling. Record exclusions and
  unresolved sources rather than forcing every asset onto a human rig.

Use [references/animation-bake.md](references/animation-bake.md) for the measured
same-hierarchy FK bake, explicit FBX import settings and metadata preservation.
The repository's `scripts/utils/native_device_animation_batch.py` is the
implementation; inspect its job schema before invoking it. An installed character
target-conversion button does not imply animation conversion support.

## Make the engine test concrete

Use a new output folder and the user's agreed editor window. Preserve existing
assets and pending version-control work. Test one converted clip, then extend the
batch once that method is supported. A resumable scheduler must journal each
completed output and prevent nested work while Unreal pumps Slate during import
or animation compression.

Read the placed device instance, not only its Verse default. For a trigger
`@editable`, `GetDeviceProperties` can return the wrapper's own address. The bound
target is the wrapper's `savedActor`; read it after assignment and save the level.
See [references/walk-on-proof.md](references/walk-on-proof.md) for the exact
distinction and the playback evidence needed.

Report separately: exported structure, incoming engine reference, assigned asset
bindings, completed cook/session, observed playback, and visual deformation.
For Device/NPC/Player interoperability, record the same actual animation asset
on each surface. A screenshot of several waving actors cannot identify their
clips or establish absence of a runtime retarget layer. A `Completed` playback
event cannot establish correct shoulders or fingers. Let the owner's direct
playback review close visual questions; avoid repeating equivalent checks.
