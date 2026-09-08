# Skeleton Contract audit

BrainDead sidebar > Skeleton Contract. Import the original reference and your candidate into the same scene with their original transforms. Select the two armatures in the panel and run **Audit Skeleton and Skin**. The report appears in the Text Editor as `BDB_Rig_Contract.json`. Turn off the finger requirement when checking a part without both hands.

This audit reads bone names, parents and world rest matrices, checks physical scale and orientation, and inspects actual weights, geometry, UVs, materials and morph deltas. It never renames bones, changes a rest pose, transfers weights or exports. A structural pass does not establish animation quality or successful Unreal playback.

Position tolerance is 0.02 mm, axis tolerance 0.02 degrees, relative scale tolerance 0.00002, and maximum eight skin influences. These tolerances permit measured Blender FBX round-trip rounding; they are not a substitute for checking the target engine.

CPU CLI example (run from this repository):

```bash
CUDA_VISIBLE_DEVICES='' blender --background --factory-startup --disable-autoexec --threads 4 --python-exit-code 1 --python scripts/utils/audit_rig_contract.py -- --reference original.fbx --target candidate.fbx --output audit.json --require-fingers --require-morph BodyMass
python3 tests/test_rig_contract.py
```

`rig_contract.snapshot(armature, meshes)` and `rig_contract.compare(reference, target)` are reusable by pipeline scripts. The comparison layer has no Blender dependency. Default CLI import/export analysis clears only its disposable background scene. Keep the reference FBX untouched.

Developed and exercised on the fresh codex male base, using the owner's Epic UEFN mannequin file. Negative controls move a rest bone and remove real finger weights; passing bone counts alone cannot hide either defect.
