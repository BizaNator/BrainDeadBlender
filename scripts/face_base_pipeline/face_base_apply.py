"""
face_base_apply.py

One-button per-head pipeline runner. Takes a freshly-imported lowpoly head
plus the existing "face base" (LowPolyHead_Parts + Penny armature) and
walks the full chain of scripts to produce a rigged head ready to animate.

Pipeline -- two phases. The auto half runs the head bind + creates the
split parts; then the user hand-positions Eye_L / Eye_R / Teeth_* /
Tongue / CustomLips onto the lowpoly face; then the finish half cleans
weights, rebinds lips, and snaps bones to where the parts ended up. Each
step calls into the standalone script of the same name (see those files
for details on what they do and what knobs they expose).

Auto phase (runs end-to-end on import):
    1. align_landmarks       -- MediaPipe-driven RBF deformation onto the
                                src body's anatomy (so binding maps the
                                right anatomy onto the right verts).
    2. headswap_transfer      -- bind weights / UVs / shape keys, with
       (preserve_geometry=True) vert indices stable so we can restore the
                                  lowpoly's own shape afterwards.
    3. restore_geometry       -- swap rigged verts back to the lowpoly's
                                  pre-deform positions.
    4. cleanup_lowpoly_head   -- strip interior shells, weld doubles,
                                  recalc normals.
    5. cleanup_face_weights   -- bone-seeded leak removal (so eye / lid /
                                  cheek weights stay where they belong).
    6. retarget_armature      -- snap face bones to the destination's
                                  actual anatomy.
    7. split_face_parts       -- pull Eye_L / Eye_R / Teeth_Upper /
                                  Teeth_Lower / Tongue out of the donor's
                                  combined parts mesh as independent objs.
    8. fit_custom_lips        -- duplicate + scale + bind the user's raw
                                  lip mesh as CustomLips, parented to the
                                  armature.

>>> USER manually positions Eye_L / Eye_R / Teeth_* / Tongue / CustomLips
    onto the lowpoly face. Then re-run with auto_phase=False and
    finish_phase=True (or just both).

Finish phase (runs after positioning):
    9. cut_face_holes         -- open eye + mouth holes in the head where
                                  the positioned parts now sit; drop the
                                  matching head weights so submesh parts
                                  carry the visible deformation.
   10. fit_face_parts         -- strip weight bleed on each split part so
                                  it follows ONLY its primary bone.
   11. rebind_lip_weights     -- replace CustomLips' weights with a clean
                                  upper/lower lip split (upper anchored,
                                  lower rides C_jaw).
   12. retarget_bones_to_parts -- snap eye / teeth / tongue / lip-control
                                  bones to the parts' current world
                                  positions so rotations pivot in place.

After all 12 steps, `rig_test_animation` rebuilds the test action so the
timeline can be replayed to verify each pose.

Designed to drop into the BrainDeadBlender add-on as the "Rig This Head"
entry point.
"""

import os
import sys
import site
import bpy


_USER_SITE = site.getusersitepackages()
if _USER_SITE not in sys.path:
    sys.path.insert(0, _USER_SITE)

try:
    _SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # When exec'd via compile()+exec without __file__ (e.g. from Blender's
    # text editor or `exec(open(...).read())`), fall back to the canonical
    # repo path.
    _SCRIPTS_DIR = r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters"


# Central donor registry -- ONE place to change donor object names when the
# Fortnite/ARKit/Mutable donors are swapped per-head (e.g. male vs female).
_DONOR_REG_NS = {}
exec(compile(open(os.path.join(_SCRIPTS_DIR, "donor_registry.py")).read(),
             "donor_registry.py", 'exec'), _DONOR_REG_NS)
donor = _DONOR_REG_NS["donor"]


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    # The raw lowpoly head being rigged. Should be cleaned-ish (no triangle
    # soup -- prefer welded geometry). If it isn't, run cleanup_lowpoly_head
    # on it FIRST and pass the result as `head`.
    "head":     "LowPolyHead_Rigged",

    # Face base -- pre-existing in the scene. Reused across runs.
    # Donor object names -- look up from donor_registry so swapping a donor
    # (male vs female Fortnite head, alternate ARKit donor, etc.) is one edit.
    "src_body": donor("skeleton", "head"),
    "armature": donor("skeleton", "armature"),
    "parts":    "LowPolyHead_Parts",     # optional, drives retarget for eye / teeth / tongue bones

    # Output name for the rigged result.
    "output":   "LowPolyHead_Rigged",    # overwrites in place by default

    # Phase toggles. The auto phase runs end-to-end on a fresh head; then
    # the user hand-positions parts; then the finish phase wraps up.
    "auto_phase":   True,
    "finish_phase": True,

    # Which steps to actually run. Disable for partial re-runs.
    # Auto phase
    "step_align_landmarks":     True,
    "step_headswap":            True,
    "step_restore_geometry":    True,
    "step_cleanup_topology":    True,
    "step_cleanup_weights":     True,
    "step_retarget_armature":   True,
    "step_split_face_parts":    True,
    "step_fit_custom_lips":     True,
    # Finish phase
    "step_cut_holes":               True,
    "step_fit_face_parts":          True,
    "step_rebind_lip_weights":      True,
    "step_retarget_bones_to_parts": True,
    "step_clean_head_shell":        True,
    "step_fit_accessories":         True,
    "step_layer_shape_keys":        True,
    # Always rebuild the test action at the very end so the timeline is fresh
    "step_rebuild_test_animation":  True,

    # Shape-key transfer method:
    #   "mediapipe" (default): anatomy-aware via MediaPipe Face Mesh
    #                          landmarks + RBF interpolation. Best quality.
    #   "naive": fast bbox-aligned BVH closest-point (no anatomical mapping).
    "shape_key_method": "mediapipe",

    # Shape-key donors to layer (in order). Each runs transfer.
    # If a donor isn't in the scene, that entry is skipped.
    "shape_key_donors": [
        donor("arkit"),          # 52 ARKit blendshapes
        donor("customization"),  # 13 Mutable customization morphs
    ],

    # Sub-meshes that ALSO receive shape-key transfer (CustomLips for lip
    # morphs, Eyelids for blink morphs, Eyebrows for brow morphs). Each
    # gets the same donor list. The MediaPipe pipeline renders the full
    # face and back-projects only the landmarks that anatomically land on
    # the sub-mesh, so lip morphs end up on lips, etc.
    "shape_key_subtargets": [
        "CustomLips",
        "Eyelid_L_Upper", "Eyelid_L_Lower",
        "Eyelid_R_Upper", "Eyelid_R_Lower",
        "Eyebrow_L", "Eyebrow_R",
    ],

    # Per-step overrides (each dict is merged into that script's CONFIG).
    "overrides": {
        "headswap_transfer": {
            "preserve_geometry": True,
            "weld_distance":     None,
            "neck_cut_local_z":  None,
            "cleanup_mesh":      False,
            "align":             False,
        },
    },
}


# ------------------------------- UTILITIES ----------------------------------
def _run(script_name, fn_name, cfg_override=None, ns_cache={}):
    """Execute a sibling script and call its top-level function with a config."""
    path = os.path.join(_SCRIPTS_DIR, script_name)
    if script_name not in ns_cache:
        ns = {}
        exec(compile(open(path).read(), script_name, 'exec'), ns)
        ns_cache[script_name] = ns
    ns = ns_cache[script_name]
    cfg = dict(ns["CONFIG"])
    if cfg_override:
        cfg.update(cfg_override)
    return ns[fn_name](cfg), ns


# --------------------------------- ENTRY ------------------------------------
def face_base_apply(cfg):
    print("==================================================")
    print("  face_base_apply -> {}".format(cfg["head"]))
    print("==================================================")

    aligned_name = cfg["head"] + "_aligned"

    # ---------- AUTO PHASE ----------
    if cfg.get("auto_phase", True):
        if cfg["step_align_landmarks"]:
            print("\n[1] align_landmarks")
            _run("align_landmarks.py", "align_landmarks", {
                "target":      cfg["head"],
                "source":      cfg["src_body"],
                "output_name": aligned_name,
            })

        if cfg["step_headswap"]:
            print("\n[2] headswap_transfer (preserve_geometry)")
            hs_override = dict(cfg["overrides"].get("headswap_transfer", {}))
            # If align was run, bind against the aligned mesh; otherwise bind
            # against the raw head directly. CAUTION: headswap_transfer's
            # duplicate_head step removes any existing object named
            # `output_name` BEFORE referencing `dst_head` -- so if `head` ==
            # `output`, the source mesh gets deleted before binding. Detect
            # and prevent that by snapshotting the head under a temp name
            # first.
            dst = aligned_name if cfg["step_align_landmarks"] else cfg["head"]
            if dst == cfg["output"]:
                src_obj = bpy.data.objects.get(dst)
                if src_obj is None:
                    raise RuntimeError(f"head '{dst}' not found in scene -- cannot rename for headswap")
                temp_name = dst + "_src"
                src_obj.name = temp_name
                dst = temp_name
                print(f"  renamed source head -> '{temp_name}' to avoid clash with output_name='{cfg['output']}'")
            hs_override.update({
                "src_body":    cfg["src_body"],
                "armature":    cfg["armature"],
                "dst_head":    dst,
                "output_name": cfg["output"],
            })
            _run("headswap_transfer.py", "headswap_transfer", hs_override)

        if cfg["step_restore_geometry"]:
            print("\n[3] restore_geometry")
            align_ns = {}
            exec(compile(open(os.path.join(_SCRIPTS_DIR, "align_landmarks.py")).read(),
                         "align_landmarks.py", 'exec'), align_ns)
            align_ns["restore_geometry"](cfg["output"])

        if cfg["step_cleanup_topology"]:
            print("\n[4] cleanup_lowpoly_head")
            _run("cleanup_lowpoly_head.py", "cleanup_lowpoly_head", {"target": cfg["output"]})

        if cfg["step_cleanup_weights"]:
            print("\n[5] cleanup_face_weights")
            _run("cleanup_face_weights.py", "cleanup_face_weights", {"target": cfg["output"]})

        if cfg["step_retarget_armature"]:
            print("\n[6] retarget_armature")
            dst_meshes = [cfg["output"]]
            if bpy.data.objects.get(cfg["parts"]):
                dst_meshes.append(cfg["parts"])
            _run("retarget_armature.py", "retarget_armature", {
                "armature":   cfg["armature"],
                "dst_meshes": dst_meshes,
            })

        if cfg["step_split_face_parts"]:
            print("\n[7] split_face_parts")
            _run("split_face_parts.py", "split_face_parts", {
                "source":   cfg["parts"],
                "armature": cfg["armature"],
            })

        if cfg["step_fit_custom_lips"]:
            print("\n[8] fit_custom_lips")
            _run("fit_custom_lips.py", "fit_custom_lips", {
                "src_body": cfg["src_body"],
                "armature": cfg["armature"],
            })

        print("\n--------------------------------------------------")
        print("  AUTO PHASE done.")
        print("  Now hand-position Eye_L / Eye_R / Teeth_Upper /")
        print("  Teeth_Lower / Tongue / CustomLips onto the face,")
        print("  then re-run with auto_phase=False finish_phase=True.")
        print("--------------------------------------------------")

    # ---------- FINISH PHASE ----------
    if cfg.get("finish_phase", True):
        if cfg["step_cut_holes"]:
            print("\n[9] cut_face_holes")
            _run("cut_face_holes.py", "cut_face_holes", {
                "head":     cfg["output"],
                "armature": cfg["armature"],
            })

        if cfg["step_fit_face_parts"]:
            print("\n[10] fit_face_parts (weights_only)")
            _run("fit_face_parts.py", "fit_face_parts", {
                "armature":     cfg["armature"],
                "weights_only": True,
            })

        if cfg["step_retarget_bones_to_parts"]:
            print("\n[11] retarget_bones_to_parts")
            _run("retarget_bones_to_parts.py", "retarget_bones_to_parts", {
                "armature": cfg["armature"],
            })

        if cfg.get("step_clean_head_shell", True):
            print("\n[12] clean_head_shell_weights (strip body+lip/jaw weights off head shell)")
            _run("clean_head_shell_weights.py", "clean_head_shell_weights", {
                "head": cfg["output"],
            })

        if cfg["step_rebind_lip_weights"]:
            print("\n[13] rebind_lip_weights (with bones in their final positions)")
            _run("rebind_lip_weights.py", "rebind_lip_weights", {
                "armature": cfg["armature"],
            })

        if cfg["step_fit_accessories"]:
            print("\n[14] fit_accessories (lids / brows / ears / nose / neck)")
            _run("fit_accessories.py", "fit_accessories", {
                "armature": cfg["armature"],
            })

        if cfg.get("step_layer_shape_keys", True):
            donors = cfg.get("shape_key_donors", [])
            method = cfg.get("shape_key_method", "mediapipe")
            script = "transfer_shape_keys_mp.py" if method == "mediapipe" else "transfer_shape_keys.py"
            fn     = "transfer_shape_keys_mp"    if method == "mediapipe" else "transfer_shape_keys"

            # Target list: main head + each configured sub-mesh
            targets = [cfg["output"]] + list(cfg.get("shape_key_subtargets", []))
            for ti, target_name in enumerate(targets):
                if bpy.data.objects.get(target_name) is None:
                    print(f"\n[15.{ti+1}] skip -- target '{target_name}' not in scene")
                    continue
                for di, donor_name in enumerate(donors):
                    if bpy.data.objects.get(donor_name) is None:
                        print(f"\n[15.{ti+1}.{di+1}] transfer_shape_keys[{method}]: skip -- donor '{donor_name}' not in scene")
                        continue
                    print(f"\n[15.{ti+1}.{di+1}] transfer_shape_keys[{method}]: {donor_name} -> {target_name}")
                    try:
                        _run(script, fn, {
                            "target": target_name,
                            "donor":  donor_name,
                        })
                    except Exception as e:
                        if method == "mediapipe":
                            print(f"  MediaPipe transfer failed ({e}); falling back to naive")
                            _run("transfer_shape_keys.py", "transfer_shape_keys", {
                                "target": target_name,
                                "donor":  donor_name,
                            })
                        else:
                            print(f"  naive transfer failed: {e}; skipping")

        if cfg["step_rebuild_test_animation"]:
            print("\n[16] rig_test_animation (rebuild)")
            _run("rig_test_animation.py", "rig_test_animation", {
                "armature": cfg["armature"],
            })

    print("\n==================================================")
    print("  face_base_apply done. Scrub timeline to verify rig.")
    print("==================================================")


if __name__ == "__main__":
    face_base_apply(CONFIG)
