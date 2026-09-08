# Animation conversion

The proven Fab-to-Device path uses Unreal's IK retargeter with one-bone FK chains
and a target retarget pose matched to the measured source pose. Source and target
have the same named hierarchy. Joint orientation changes are handled as rotation
bases, not subtraction of Euler angle triples. Root and pelvis translations are
handled explicitly. The implementation restores authored IK helper tracks and
tests their motion; that restoration is specific to the measured matching
hierarchy, not a generic rule for a differently oriented source rig.

For different proportions or different chains (for example UE4's shorter spine),
use an anatomical chain mapping and an appropriate retarget pose. Do not reuse
the one-bone same-hierarchy solve merely because some bone names match.

Record source Skeleton, preview/source mesh, retarget settings, frame count,
frame-rate numerator/denominator, root-motion and additive configuration, curves,
notifies, sync markers, montage and blend-space dependencies. Keep original
sources. Save converted assets under new paths with a source-to-output manifest.
Additive assets require their base pose and reference animation to be converted
consistently. Do not silently convert an additive delta as an absolute pose.

Native asset conversion can preserve metadata that FBX export/import omits. The
tested cohort checks metadata equality and evaluates each float curve at source
frames. That verifies sampled curve values, not raw key tangents. FName identity
is case-insensitive; changed display casing is not a new curve identity.

When using FBX, explicitly set all transform-affecting options instead of relying
on the editor's remembered settings:

```
convert_scene = True
convert_scene_unit = True
force_front_x_axis = False
import_translation = (0, 0, 0)
import_rotation = (0, 0, 0)
import_uniform_scale = 1
update_skeleton_reference_pose = False
use_t0_as_ref_pose = False
```

The tested animation import also used `preserve_local_transform=False`. Verify
the chosen options against a fresh reference and a pilot. An unintended front-X
conversion previously permuted the pelvis axes and added a 90-degree root basis
change. That import defect was separate from the underlying pose conversion.

Compare rotations with normalized quaternion angular distance (absolute dot to
handle q/-q). Compare positions in declared spaces and units. Euler subtraction
can vary frame by frame even for a constant basis correction.

Unreal may pump Slate while waiting for animation compression. A post-tick batch
must use a busy guard with `try/finally`; nested ticks must not start another
conversion. Journal only after save and checks complete, and stop on an unexpected
fault. Outputs left by a crash without completed journal records remain
unverified. Resume from completed records; do not infer completion from filenames.
