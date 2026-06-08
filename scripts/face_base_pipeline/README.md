# face_base_pipeline/ — bring your own scripts

This directory is intentionally empty in the public BrainDeadBlender repo. The
addon's Face Base Pipeline panel expects pipeline scripts to live here (or at a
path pointed to by the `STUDIO_PIPELINE_DIR` environment variable).

If you fork BDB and want to use the Face Base Pipeline panel, drop your own
pipeline scripts here. The studio that authored this addon keeps theirs in a
private location; the addon itself is open source but the studio-specific
automation is not.

Scripts the panel expects:
- `face_base_apply.py` — orchestrator with a `main(config_override=...)` entrypoint
- `calibrate_faceplate_uv.py` — provides `bake_edge_mask`, `beautify_faces`, `fit_canonical_eyes`, `calibrate_full`, etc.
- `build_faceplate_albedo.py` — atlas builder
- `donor_registry.py` — central object-name registry
- Plus various helpers the orchestrator imports

If your scripts have a different shape, replace the operator wrappers in
`braindead_blender/__init__.py` accordingly.
