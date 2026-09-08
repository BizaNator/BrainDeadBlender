# Measured native contracts

Captured in UEFN `6.0.0-57819926+++Fortnite+Release-42.10`, September 8, 2026.

| Reference | What was counted |
|---|---:|
| `/Game/Creative/Devices/Mannequin/Meshes/CP_Device_Mannequin_Skeleton` | 88 physical hierarchy entries |
| `/Game/Characters/Player/Male/Male_Avg_Base/Fortnite_M_Avg_Player_Skeleton` | 280 BoneTree entries, 310 AnimPose entries |
| Fab `skm_uefn_mannequin` | 88 hierarchy entries |
| Examined MetaHuman reference | 342 AnimPose entries; physical/virtual split not established |

Player's final 30 AnimPose entries use the `VB ` prefix. That prefix observation
and the actual 280-entry BoneTree measurement are different facts. Player adds
192 physical bones beyond the shared Device core. The remaining entries include
character-specific mechanical parts; membership alone is not a reason to weight
new clothing to an arbitrary one of them.

Across the 88 shared Device/Player names: no parent mismatches, maximum local
translation difference 0.0000038 cm, component translation difference 0.00022 cm,
and rotation difference 0.0011 degrees. Scales agree.

Across Device/Fab: parents agree, maximum local differences 53.52 cm / 54.21
degrees, component differences 28.34 cm / 28.27 degrees. Each upper arm differs
by 27.10766 degrees locally. Read each transform; applying one angle everywhere
would miss the different IK and finger corrections.

These measurements support one shared native body target. They do not alone
prove playback acceptance for every character, compression mode, missing extra
track, virtual bone, socket, or UEFN device route. Test the intended extension.
Keep raw captures with their engine version when producing new authoring rigs.

## Full Player FBX round trip

The reconstructed 280-entry Player reference was imported into a fresh UEFN
Skeleton before use. All names and parents agree; maximum local difference
was 0.002018 cm / 0.014199 degrees / 0.000016928 scale. Explicit comparison
tolerances were 0.005 cm, 0.02 degrees and 0.00005 scale. This is numerical
reference agreement, not a claim of mathematical identity or runtime approval.

`pelvisRigidBodyShape1Transform` is an unweighted leaf with uniform native rest
scale 1.1239911317825317. Blender edit bones do not retain that as a separate
rest scale. A normal Blender export therefore needs the native export path:
`fbx_reference_scale.restore_reference_bind` copies the verified reference's
named local transform fields and bone bind matrices after matching names,
parents and FBX axes/units. It preserves the new mesh geometry, weights and
morphs. This also prevents repeated float/Euler reconstruction from adding
rotation drift across exports. The source reference hash and DCC fingerprint
are checked before using it.

The lower-level scale patch supports only uniform unweighted leaves; it refuses
weighted bones or bones with child models. Its binary writer preserves every
FBX property type and verifies the parsed tree before replacing the new export.
FBX CHAR `C` and BOOL `B` are distinct in Blender's implementation; treating
`C` as a Boolean produced an SDK-rejected diagnostic file during development.
