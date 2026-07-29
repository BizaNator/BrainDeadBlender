"""
BrainDeadBlender — AutoRig panel.

Sidebar UI for "Auto-Rig This Mesh" operator. Exports the active mesh as
GLB, POSTs a workflow to a ComfyUI server (BrainZ by default — 10.15.0.20),
polls /history for completion, downloads the rigged FBX, imports it back,
and (optionally) runs PoseFixer_v1 to retarget the rest pose into UEFN
A-pose.

The dispatched workflow uses the BD_AutoRigMIA node from ComfyUI-BrainDead
which internally chains Make-It-Animatable + BD_MixamoToUEFN. Output FBX
already has UEFN-Mannequin bone names.

Backend setting points at any ComfyUI server with the BrainDead pack
installed; defaults to http://10.15.0.20:8188 (BrainZ dev).
"""

import bpy
import json
import os
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from bpy.props import (
    StringProperty, EnumProperty, BoolProperty, IntProperty,
)
from bpy.types import Operator, Panel, PropertyGroup


# ── Settings property group ──────────────────────────────────────────────────

class BD_AutoRigSettings(PropertyGroup):
    """Settings for the BD AutoRig dispatch flow."""

    backend: EnumProperty(
        name="Backend",
        description="Where to run the autorig inference",
        items=[
            ("local_managed", "Local (managed venv)",
             "Self-contained: BDB downloads PyTorch + MIA + weights into ~/.cache/braindead_blender/autorig/ on first use. No ComfyUI required."),
            ("brainz_dev",    "BrainZ Dev (10.15.0.20:8189)",
             "Dispatch to BrainZ Dev ComfyUI — Make-It-Animatable + UEFN remap"),
            ("brainz_stable", "BrainZ Stable (10.15.0.20:8188)",
             "Dispatch to BrainZ Stable ComfyUI"),
            ("local_comfy",   "Local ComfyUI (127.0.0.1:8188)",
             "Dispatch to a ComfyUI you're running on this machine"),
            ("custom",        "Custom URL",
             "Set comfy_url manually"),
        ],
        default="local_managed",
    )

    comfy_url: StringProperty(
        name="ComfyUI URL",
        description="Base URL to the ComfyUI server when backend = 'custom'",
        default="http://10.15.0.20:8189",
    )

    rigger: EnumProperty(
        name="Auto-Rigger",
        description="Which model to run inference with",
        items=[
            ("mia",    "Make-It-Animatable (fast, humanoid only)", "BD_AutoRigMIA"),
            ("unirig", "UniRig (general)", "BD_AutoRigUniRig (mixamo template)"),
        ],
        default="mia",
    )

    no_fingers: BoolProperty(
        name="Merge fingers into hand",
        description=(
            "If ON: MIA's finger weights are merged into the parent Hand "
            "bone and the 30 finger bones are stripped from the export "
            "(simpler rig: 25 bones, hand mesh follows the wrist).\n"
            "If OFF: keep all 30 finger bones (3 segments × 5 fingers × 2 "
            "hands) with MIA's own predicted weights — 65-bone rig with "
            "per-finger control (full Mixamo-style hands)"
        ),
        default=False,  # default to full fingers — MIA predicts them and
                         # they don't distort if the Hand bone survives
    )

    reset_to_rest: BoolProperty(
        name="Reset to Rest",
        description="Transform output mesh to T-pose rest position before "
                     "running PoseFixer",
        default=True,
    )

    remap_to_uefn: BoolProperty(
        name="Remap to UEFN",
        description="Rename bones from Mixamo to UEFN_Mannequin convention "
                     "in the ComfyUI workflow",
        default=True,
    )

    run_posefixer: BoolProperty(
        name="Run PoseFixer After Import",
        description="Automatically retarget the imported skeleton's rest pose "
                     "to canonical UEFN A-pose via PoseFixer_v1",
        default=False,  # leave manual until verified end-to-end on jojo
    )

    transfer_weights: BoolProperty(
        name="Transfer Weights from UEFN_Mannequin",
        description=(
            "When ON, the Transfer-to-UEFN step also transfers vertex "
            "weights from SKM_UEFN_Mannequin (source) to the target mesh "
            "via proximity. This re-binds mesh verts to the closest "
            "canonical UEFN bones, removing the static offset between "
            "mesh extremities and the (proportionally smaller) UEFN bone "
            "tips that comes from non-standard mesh proportions.\n"
            "OFF keeps MIA's autorig weights, which may leave hands/feet "
            "underweighted to the canonical bones."
        ),
        default=True,
    )

    conform_on_transfer: BoolProperty(
        name="Conform Mesh Before Binding",
        description=(
            "Automatically run 'Conform Mesh to UEFN' at the start of the "
            "Transfer-to-UEFN step: uniform-scale the mesh to the donor's "
            "height and axially scale each arm so fingertips land at the "
            "canonical reach. Removes the extremity offset that otherwise "
            "warps the mesh when the rest pose is changed"
        ),
        default=True,
    )

    donor_blend: StringProperty(
        name="Donor Blend",
        description=(
            "Library .blend containing the canonical SKM_UEFN_Mannequin "
            "donor (armature + mesh). Auto-appended into a 'Source' "
            "collection when the scene doesn't already have one"
        ),
        default=r"B:\Brains\Library\parts\skeleton_uefn_manny.blend",
        subtype="FILE_PATH",
    )

    target_pose: EnumProperty(
        name="Target Rest Pose",
        description=(
            "Which rest pose the 'Set Rest Pose' button writes onto the "
            "Export armature. T-pose is the current binding donor's pose "
            "(arms out horizontal); A-pose is the canonical Fab "
            "UEFN_Mannequin rest pose (arms angled ~60° down). For "
            "UEFN/Unreal export A-pose is canonical."
        ),
        items=[
            ("T_POSE", "T-pose", "Arms straight out horizontally (autorig native)"),
            ("A_POSE", "A-pose (Fab UEFN_Mannequin)",
             "Canonical UEFN/Unreal A-pose from skm_uefn_mannequin.FBX"),
            ("CUSTOM", "Custom reference FBX",
             "Read bone rest matrices from a user-provided FBX path"),
        ],
        default="A_POSE",
    )

    conform_bone_lengths: BoolProperty(
        name="Conform Bone Lengths",
        description=(
            "When Set Rest Pose runs, snap joints to the EXACT canonical "
            "UEFN skeleton positions (bone lengths included) — the export "
            "skeleton matches the mannequin's reference pose 1:1 and the "
            "mesh follows via joint-blend skinning. OFF keeps the "
            "character's own bone lengths (better mesh fidelity for "
            "extreme proportions; relies on UE translation retargeting)"
        ),
        default=True,
    )

    deform_mesh_on_rest_change: BoolProperty(
        name="Move Mesh With Rest Pose",
        description=(
            "When Set Rest Pose runs, bake the mesh into the new rest "
            "shape so it visually follows the bones (e.g. arms drop from "
            "T to A). Requires the pre-bind conform so proportions match "
            "the canonical bones. OFF leaves the mesh as modeled and only "
            "moves the bones"
        ),
        default=True,
    )

    pose_reference_fbx: StringProperty(
        name="Pose Reference FBX",
        description=(
            "Path to the reference skeleton FBX whose bone rest matrices "
            "will be copied (by name match) onto the Export armature when "
            "'Set Rest Pose' is clicked. Default points at Epic's Fab "
            "UEFN_Mannequin FBX in the studio's character pipeline."
        ),
        default=(r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters"
                  r"\UEFN_Mannequin\fbx\skm_uefn_mannequin.FBX"),
        subtype="FILE_PATH",
    )

    poll_timeout_sec: IntProperty(
        name="Timeout (s)",
        description="Maximum time to wait for ComfyUI workflow completion",
        default=300,
        min=30,
        max=3600,
    )

    poll_interval_sec: IntProperty(
        name="Poll Interval (s)",
        description="How often to poll ComfyUI for completion",
        default=2,
        min=1,
        max=30,
    )


def _resolve_comfy_url(settings: BD_AutoRigSettings) -> str:
    mapping = {
        "brainz_dev":    "http://10.15.0.20:8189",
        "brainz_stable": "http://10.15.0.20:8188",
        "local_comfy":   "http://127.0.0.1:8188",
    }
    if settings.backend in mapping:
        return mapping[settings.backend]
    return settings.comfy_url


# ── ComfyUI dispatch ─────────────────────────────────────────────────────────

def _build_workflow(uploaded_filename: str, settings: BD_AutoRigSettings,
                     output_stem: str) -> dict:
    """Build a ComfyUI workflow that:
      1) UniRigLoadMesh — load `uploaded_filename` from the input folder
      2) BD_AutoRigMIA or BD_AutoRigUniRig — run autorig + optional UEFN remap

    `uploaded_filename` is the relative name returned by ComfyUI's /upload
    endpoint (e.g., 'jojo_rhoads_body.glb' or '3d/jojo_rhoads_body.glb')."""
    if settings.rigger == "mia":
        rig_node = {
            "class_type": "BD_AutoRigMIA",
            "inputs": {
                "mesh": ["1", 0],
                "fbx_name": output_stem,
                "device": "auto",
                "no_fingers": settings.no_fingers,
                "use_normal": False,
                "reset_to_rest": settings.reset_to_rest,
                "remap_to_uefn": settings.remap_to_uefn,
            },
        }
    else:
        rig_node = {
            "class_type": "BD_AutoRigUniRig",
            "inputs": {
                "mesh": ["1", 0],
                "skeleton_template": "mixamo",
                "fbx_name": output_stem,
                "device": "auto",
                "remap_to_uefn": settings.remap_to_uefn,
            },
        }
    return {
        "1": {
            "class_type": "UniRigLoadMesh",
            "inputs": {
                "source_folder": "input",
                "file_path": uploaded_filename,
            },
        },
        "2": rig_node,
    }


def _upload_to_comfy(comfy_url: str, local_path: Path) -> str:
    """POST a file to ComfyUI's /upload/image endpoint (which accepts any
    file, not just images). Returns the uploaded filename (relative to the
    server's input folder) so the workflow can reference it."""
    import mimetypes
    import uuid
    boundary = f"----BDBoundary{uuid.uuid4().hex}"
    ctype, _ = mimetypes.guess_type(local_path.name)
    if not ctype:
        ctype = "application/octet-stream"
    with open(local_path, "rb") as fh:
        file_bytes = fh.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; '
        f'filename="{local_path.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode("utf-8") + file_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        f"true\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    url = comfy_url.rstrip("/") + "/upload/image"
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        info = json.loads(resp.read().decode("utf-8"))
    # ComfyUI returns {"name": ..., "subfolder": ..., "type": "input"}
    name = info.get("name") or local_path.name
    subfolder = info.get("subfolder") or ""
    if subfolder:
        return f"{subfolder}/{name}"
    return name


def _post_prompt(comfy_url: str, workflow: dict) -> str:
    """POST a workflow to /prompt; return the queued prompt_id."""
    url = comfy_url.rstrip("/") + "/prompt"
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI /prompt returned no prompt_id: {body}")
    return prompt_id


def _poll_history(comfy_url: str, prompt_id: str, *,
                    timeout: int, interval: int) -> dict:
    """Poll /history/<prompt_id> until the prompt is complete or we time out.
    Returns the history entry dict (with 'outputs' and 'status' keys)."""
    deadline = time.time() + timeout
    url = comfy_url.rstrip("/") + f"/history/{prompt_id}"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                hist = json.loads(resp.read().decode("utf-8"))
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("completed") or status.get("status_str") == "success":
                    return entry
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI workflow errored: "
                                         f"{status.get('messages')}")
        except urllib.error.URLError as e:
            print(f"[BD_AutoRig] poll error: {e}")
        time.sleep(interval)
    raise TimeoutError(
        f"ComfyUI workflow {prompt_id} did not finish within {timeout}s")


def _extract_output_path(history_entry: dict, *, stem: str = "",
                         remap_to_uefn: bool = True) -> str:
    """Pull the FBX path string from the BD_AutoRig node's output."""
    outputs = history_entry.get("outputs") or {}
    # Find any node output that contains a .fbx string
    for _node_id, node_outputs in outputs.items():
        # Look at all output values
        for val in node_outputs.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.lower().endswith(".fbx"):
                        return item
                    if isinstance(item, dict):
                        # Some nodes wrap path in {'filename': ..., 'subfolder': ..., 'type': ...}
                        if str(item.get("filename", "")).lower().endswith(".fbx"):
                            return item["filename"]
            elif isinstance(val, str) and val.lower().endswith(".fbx"):
                return val
    # V3 nodes return plain io.String outputs which never land in history
    # "outputs" (that map only carries ui payloads). The BD_AutoRig nodes
    # write deterministically to the ComfyUI output dir as
    # "{stem}_uefn.fbx" (remap on) / "{stem}_mixamo.fbx", so fall back to
    # that contract — _download_output only needs the filename.
    if stem:
        suffix = "uefn" if remap_to_uefn else "mixamo"
        return f"{stem}_{suffix}.fbx"
    raise RuntimeError(
        f"Could not find .fbx output in ComfyUI history entry: {outputs}")


def _download_output(comfy_url: str, remote_path: str, local_path: Path):
    """Download an FBX from ComfyUI's /view endpoint. remote_path may be an
    absolute server path; we just take the filename for the /view request."""
    filename = Path(remote_path).name
    url = (comfy_url.rstrip("/") +
           f"/view?filename={filename}&type=output")
    with urllib.request.urlopen(url, timeout=120) as resp:
        local_path.write_bytes(resp.read())


# ── Operators ────────────────────────────────────────────────────────────────

class BD_OT_AutoRigMesh(Operator):
    """Dispatch the active mesh to ComfyUI auto-rig and import the result."""

    bl_idname = "braindead.autorig_mesh"
    bl_label = "Auto-Rig This Mesh"
    bl_description = (
        "Export the active mesh, dispatch to ComfyUI autorigger on the "
        "configured backend, download the rigged FBX, import it into the "
        "scene, and (optionally) retarget to UEFN A-pose"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH"

    def execute(self, context):
        settings: BD_AutoRigSettings = context.scene.bd_autorig
        ob = context.active_object
        stem = ob.name

        # 1) Export GLB to a temp file
        tmp_dir = Path(tempfile.gettempdir()) / "bd_autorig"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        glb_path = tmp_dir / f"{stem}.glb"
        try:
            bpy.ops.object.select_all(action="DESELECT")
            ob.select_set(True)
            context.view_layer.objects.active = ob
            bpy.ops.export_scene.gltf(
                filepath=str(glb_path),
                export_format="GLB",
                use_selection=True,
                export_apply=True,  # apply modifiers
                # MIA only needs geometry — embedded textures bloat the
                # upload past ComfyUI's max body size (413 on multi-100MB
                # concept meshes). The user's original mesh keeps its
                # materials; this GLB is throwaway.
                export_image_format="NONE",
            )
        except Exception as e:
            self.report({"ERROR"}, f"GLB export failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {glb_path}")
        print(f"[BD_AutoRig] Exported GLB to {glb_path}")

        # ── Local-managed path: bootstrap + venv subprocess + Blender FBX
        if settings.backend == "local_managed":
            return self._execute_local(context, settings, ob, glb_path,
                                          tmp_dir, stem)
        comfy_url = _resolve_comfy_url(settings)

        # 1.5) Upload to ComfyUI's input folder so the workflow can reference
        # it by filename (BrainZ can't see the Windows-side temp dir)
        try:
            uploaded_name = _upload_to_comfy(comfy_url, glb_path)
        except Exception as e:
            self.report({"ERROR"}, f"ComfyUI upload failed: {e}")
            return {"CANCELLED"}
        print(f"[BD_AutoRig] Uploaded GLB as {uploaded_name}")

        # 2) Build + POST workflow
        workflow = _build_workflow(uploaded_name, settings, stem)
        try:
            prompt_id = _post_prompt(comfy_url, workflow)
        except Exception as e:
            self.report({"ERROR"}, f"ComfyUI /prompt failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"},
                     f"Queued ComfyUI prompt {prompt_id} on {comfy_url}")
        print(f"[BD_AutoRig] Queued prompt {prompt_id} on {comfy_url}")

        # 3) Poll for completion
        try:
            entry = _poll_history(
                comfy_url, prompt_id,
                timeout=settings.poll_timeout_sec,
                interval=settings.poll_interval_sec,
            )
        except Exception as e:
            self.report({"ERROR"}, f"Polling failed: {e}")
            return {"CANCELLED"}

        try:
            remote_fbx = _extract_output_path(
                entry, stem=stem, remap_to_uefn=settings.remap_to_uefn)
        except Exception as e:
            self.report({"ERROR"}, f"Output extraction failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Workflow output: {remote_fbx}")

        # 4) Download
        local_fbx = tmp_dir / f"{stem}_autorigged_uefn.fbx"
        try:
            _download_output(comfy_url, remote_fbx, local_fbx)
        except Exception as e:
            self.report({"ERROR"}, f"FBX download failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Downloaded {local_fbx}")

        # 5) Import + align to canonical UEFN coords + optionally retarget
        pre_arms = {o.name for o in bpy.data.objects if o.type == "ARMATURE"}
        pre_meshes = {o.name for o in bpy.data.objects if o.type == "MESH"}
        try:
            bpy.ops.import_scene.fbx(
                filepath=str(local_fbx),
                automatic_bone_orientation=True,
                use_anim=False,
            )
        except Exception as e:
            self.report({"ERROR"}, f"FBX import failed: {e}")
            return {"CANCELLED"}

        # Same normalization the local backend applies (the FBX comes in
        # with armature at scale 0.01 + 90°X and the skinned mesh ~1/100
        # scale). Without this every downstream measurement (height
        # conform, proximity weight transfer, rest-pose bake) explodes.
        new_arms = [o for o in bpy.data.objects
                       if o.type == "ARMATURE" and o.name not in pre_arms]
        new_meshes = [o for o in bpy.data.objects
                         if o.type == "MESH" and o.name not in pre_meshes]
        if new_arms and new_meshes:
            try:
                from . import autorig_local
                align_info = autorig_local.align_imported_to_uefn(
                    new_arms[0], new_meshes[0], source_mesh=ob)
                print(f"[BD_AutoRig] aligned import via "
                      f"{align_info.get('rotated_via', '?')}, "
                      f"target_h={align_info.get('target_h', 0):.3f}")
                print(f"[BD_AutoRig]   mesh_world={align_info['mesh_world_bbox_after']}")
                print(f"[BD_AutoRig]   arm_world ={align_info['arm_world_bbox_after']}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({"WARNING"}, f"Transform-align failed: {e}")
        self.report({"INFO"}, "Auto-rig complete — see imported armature + mesh")

        if settings.run_posefixer:
            try:
                _run_posefixer_on_last_import(self)
            except Exception as e:
                self.report({"WARNING"}, f"PoseFixer skipped: {e}")

        return {"FINISHED"}

    # ── Local-managed backend ────────────────────────────────────────────────

    def _execute_local(self, context, settings, source_obj, glb_path, tmp_dir,
                          stem):
        """Path taken when backend == 'local_managed'. No ComfyUI dispatch —
        runs everything in a managed venv on this machine."""
        from . import autorig_local
        from . import autorig_bootstrap

        if not autorig_bootstrap.is_bootstrapped():
            self.report(
                {"ERROR"},
                "Local autorig not installed. Click "
                "'Install Local Autorig' first (one-time, ~5-15 min).",
            )
            return {"CANCELLED"}

        local_fbx = tmp_dir / f"{stem}_autorigged_uefn.fbx"
        ok, msg = autorig_local.run_local_autorig(
            glb_path, local_fbx,
            no_fingers=settings.no_fingers,
            use_normal=False,
            reset_to_rest=settings.reset_to_rest,
            progress_cb=lambda m: print(f"[BD_AutoRig:local] {m}", flush=True),
        )
        if not ok:
            self.report({"ERROR"}, f"Local autorig failed: {msg}")
            return {"CANCELLED"}

        # Snapshot armatures pre-import so we can find what was added
        pre_arms = {o.name for o in bpy.data.objects if o.type == "ARMATURE"}
        pre_meshes = {o.name for o in bpy.data.objects if o.type == "MESH"}

        try:
            bpy.ops.import_scene.fbx(
                filepath=str(local_fbx),
                automatic_bone_orientation=True,
                use_anim=False,
            )
        except Exception as e:
            self.report({"ERROR"}, f"FBX import failed: {e}")
            return {"CANCELLED"}

        new_arms = [o for o in bpy.data.objects
                       if o.type == "ARMATURE" and o.name not in pre_arms]
        new_meshes = [o for o in bpy.data.objects
                         if o.type == "MESH" and o.name not in pre_meshes]
        arm_obj = new_arms[0] if new_arms else None
        mesh_obj = new_meshes[0] if new_meshes else None

        # Align to canonical UEFN coords (apply transforms, scale mesh to
        # match bones, flip orientation, feet on Z=0). Without this the
        # FBX comes in with armature at scale 0.01+90°X and mesh at
        # ~1/100 scale facing backwards.
        if arm_obj is not None and mesh_obj is not None:
            try:
                align_info = autorig_local.align_imported_to_uefn(
                    arm_obj, mesh_obj, source_mesh=source_obj)
                print(f"[BD_AutoRig:local] aligned via "
                       f"{align_info.get('rotated_via', '?')}, "
                       f"forward={align_info.get('forward_flip', '?')}, "
                       f"target_h={align_info.get('target_h', 0):.3f}")
                print(f"[BD_AutoRig:local]   mesh_world={align_info['mesh_world_bbox_after']}")
                print(f"[BD_AutoRig:local]   arm_world ={align_info['arm_world_bbox_after']}")
            except Exception as e:
                self.report({"WARNING"}, f"Transform-align failed: {e}")

        # Rename Mixamo-style bones in-place to UEFN canonical (if requested
        # and the FBX still has Mixamo names — Local backend doesn't run the
        # ComfyUI BD_MixamoToUEFN node)
        if settings.remap_to_uefn and arm_obj is not None:
            try:
                bones, vgs, unmapped = autorig_local.remap_imported_to_uefn(arm_obj)
                self.report({"INFO"},
                             f"Renamed {bones} bones, {vgs} vgroups "
                             f"({len(unmapped)} unmapped)")
                if unmapped:
                    print(f"[BD_AutoRig:local] unmapped bones: {unmapped}")
            except Exception as e:
                self.report({"WARNING"}, f"UEFN remap failed: {e}")

        # NOTE: fit_bones_to_mesh helper exists in autorig_local but is
        # NOT called by default. The naive centroid-aim algorithm produced
        # crooked intermediate bones (centroid of arm influence isn't on
        # the shoulder→wrist axis) and orphaned child bones (fingers
        # stayed at the old hand_l.tail position). Until that's properly
        # solved (project bones onto chain skeleton-line + propagate
        # head/tail shifts to children), the canonical-UEFN retarget
        # step downstream is the right place to conform proportions.

        self.report({"INFO"}, f"Local autorig complete — FBX at {local_fbx}")

        if settings.run_posefixer:
            try:
                _run_posefixer_on_armature(arm_obj)
                self.report({"INFO"}, "PoseFixer A-pose retarget done")
            except Exception as e:
                self.report({"WARNING"}, f"PoseFixer skipped: {e}")

        return {"FINISHED"}


# ── PoseFixer integration ────────────────────────────────────────────────────

def _find_posefixer() -> "Path | None":
    """PoseFixer_v1 ships vendored next to this file (autorig_vendor/) but
    may also live in the repo at scripts/uefn_pipeline/. Probe both."""
    here = Path(__file__).resolve().parent
    for cand in (
        here / "autorig_vendor" / "PoseFixer_v1.py",
        here.parent / "scripts" / "uefn_pipeline" / "PoseFixer_v1.py",
    ):
        if cand.exists():
            return cand
    return None


_POSEFIXER_PATH = _find_posefixer()


def _ensure_source_target_collections(target_arm: bpy.types.Object):
    """Build the Source + Target collections PoseFixer_v1 expects.

    Source = UEFN_Manny donor (linked from donors.blend if not already in scene)
    Target = collection wrapping the newly-imported armature + its meshes
    """
    src_name, tgt_name = "Source", "Target"

    # Target
    tgt_coll = bpy.data.collections.get(tgt_name) or bpy.data.collections.new(tgt_name)
    if tgt_name not in {c.name for c in bpy.context.scene.collection.children}:
        bpy.context.scene.collection.children.link(tgt_coll)
    # Move target_arm + its skinned meshes into Target
    objs = [target_arm] + [m for m in bpy.data.objects
                              if m.type == "MESH"
                              and any(mod.type == "ARMATURE"
                                       and mod.object == target_arm
                                       for mod in m.modifiers)]
    for o in objs:
        for c in list(o.users_collection):
            if c is not tgt_coll:
                c.objects.unlink(o)
        if o.name not in tgt_coll.objects:
            tgt_coll.objects.link(o)

    # Source: must already have UEFN_Manny donor. Just check.
    src_coll = bpy.data.collections.get(src_name)
    if src_coll is None or not any(o.type == "ARMATURE"
                                       for o in src_coll.all_objects):
        raise RuntimeError(
            "Source collection (UEFN_Manny donor) not found. Append/link "
            "UEFN_Manny armature into a 'Source' collection before "
            "enabling PoseFixer.")


def _run_posefixer_on_armature(target_arm: bpy.types.Object):
    """Set up Source/Target and exec PoseFixer_v1 against them."""
    if _POSEFIXER_PATH is None or not _POSEFIXER_PATH.exists():
        raise RuntimeError(
            "PoseFixer_v1.py not found (probed autorig_vendor/ and "
            "../scripts/uefn_pipeline/)")
    if target_arm is None:
        raise RuntimeError("No imported armature to retarget")
    _ensure_source_target_collections(target_arm)
    src = _POSEFIXER_PATH.read_text(encoding="utf-8")
    exec(compile(src, str(_POSEFIXER_PATH), "exec"),
          {"__name__": "__posefixer__", "__file__": str(_POSEFIXER_PATH)})


def _run_posefixer_on_last_import(operator):
    """Fallback for ComfyUI path — we don't track exact armature, find newest."""
    arms = sorted((o for o in bpy.data.objects if o.type == "ARMATURE"),
                    key=lambda o: o.name)
    if not arms:
        raise RuntimeError("No armatures in scene")
    _run_posefixer_on_armature(arms[-1])


# ── Transfer-to-UEFN operator ────────────────────────────────────────────────

_TRANSFER_BONES_PATH = (Path(__file__).resolve().parent.parent /
                            "scripts" / "uefn_pipeline" / "TransferBones_v1.py")


def _find_transferbones() -> "Path | None":
    here = Path(__file__).resolve().parent
    for cand in (
        here / "autorig_vendor" / "TransferBones_v1.py",
        here.parent / "scripts" / "uefn_pipeline" / "TransferBones_v1.py",
        _TRANSFER_BONES_PATH,
    ):
        if cand.exists():
            return cand
    return None


def _find_target_mesh(context) -> "bpy.types.Object | None":
    """The autorigged mesh to operate on: prefer the active mesh, else the
    most-recently-imported MIA_Character, else the largest mesh outside the
    Source/Export collections."""
    excluded = set()
    for col_name in ("Source", "Export"):
        col = bpy.data.collections.get(col_name)
        if col:
            excluded |= {o.name for o in col.all_objects if o.type == "MESH"}

    ob = context.active_object
    if ob is not None and ob.type == "MESH" and ob.name not in excluded:
        return ob
    candidates = [o for o in bpy.data.objects
                      if o.type == "MESH"
                      and o.name not in excluded
                      and not o.hide_viewport
                      and "world" not in o.name.lower()]
    mia = [o for o in candidates if "MIA_Character" in o.name]
    if mia:
        return mia[0]
    if candidates:
        return max(candidates, key=lambda o: len(o.data.vertices))
    return None


def _mesh_rig_armature(mesh: bpy.types.Object) -> "bpy.types.Object | None":
    """The armature the mesh is bound to (first Armature modifier)."""
    for mod in mesh.modifiers:
        if mod.type == "ARMATURE" and mod.object is not None:
            return mod.object
    return mesh.parent if (mesh.parent and mesh.parent.type == "ARMATURE") \
        else None


def _ensure_source_donor(settings) -> "tuple[bpy.types.Object, bpy.types.Object | None]":
    """Make sure a 'Source' collection with the UEFN_Mannequin donor exists,
    appending it from the library donor blend if missing. Returns
    (donor_armature, donor_mesh); raises RuntimeError if unavailable."""
    export_col = bpy.data.collections.get("Export")
    export_arms = ({o.name for o in export_col.all_objects
                       if o.type == "ARMATURE"} if export_col else set())

    def _donor_from_source():
        src = bpy.data.collections.get("Source")
        if src is None:
            return None, None
        # Skip the duplicate 'root' TransferBones leaves linked in both
        # Source and Export
        arms = [o for o in src.all_objects
                   if o.type == "ARMATURE" and o.name not in export_arms]
        meshes = [o for o in src.all_objects if o.type == "MESH"]
        donor_mesh = max(meshes, key=lambda o: len(o.data.vertices)) \
            if meshes else None
        return (arms[0] if arms else None), donor_mesh

    donor_arm, donor_mesh = _donor_from_source()
    if donor_arm is not None:
        return donor_arm, donor_mesh

    # Append from the library donor blend
    blend = Path(bpy.path.abspath(settings.donor_blend))
    if not blend.exists():
        raise RuntimeError(
            f"No Source donor in scene and donor blend not found: {blend}. "
            "Append the UEFN_Mannequin donor into a 'Source' collection or "
            "fix the Donor Blend path.")
    pre = {o.name for o in bpy.data.objects}
    with bpy.data.libraries.load(str(blend), link=False) as (src_data, dst_data):
        dst_data.objects = list(src_data.objects)
    src_col = bpy.data.collections.get("Source")
    if src_col is None:
        src_col = bpy.data.collections.new("Source")
        bpy.context.scene.collection.children.link(src_col)
    for o in bpy.data.objects:
        if o.name in pre or o is None:
            continue
        if o.type in ("ARMATURE", "MESH"):
            for c in list(o.users_collection):
                c.objects.unlink(o)
            src_col.objects.link(o)
    # Un-exclude Source in the view layer (appended collections often
    # default excluded)
    def _enable(lc):
        if lc.collection.name == "Source":
            lc.exclude = False
        for c in lc.children:
            _enable(c)
    _enable(bpy.context.view_layer.layer_collection)

    donor_arm, donor_mesh = _donor_from_source()
    if donor_arm is None:
        raise RuntimeError(
            f"Appended {blend.name} but found no armature in Source")
    print(f"[BD_AutoRig] Appended UEFN donor from {blend}")
    return donor_arm, donor_mesh


class BD_OT_ConformMeshToUEFN(Operator):
    """Pre-bind mesh conform: match the autorigged mesh's height and arm
    reach to the canonical UEFN_Mannequin donor BEFORE TransferBones
    binding, so extremity verts land where the canonical bones expect them.
    """

    bl_idname = "braindead.conform_mesh_to_uefn"
    bl_label = "Conform Mesh to UEFN"
    bl_description = (
        "Scale the autorigged mesh to canonical UEFN_Mannequin proportions "
        "(uniform height + per-arm reach). Run BEFORE 'Transfer to UEFN "
        "Skeleton' — removes the extremity offset that causes warping when "
        "the rest pose is changed later. Appends the library donor into "
        "'Source' automatically if it isn't in the scene yet."
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import autorig_local
        settings = context.scene.bd_autorig

        mesh = _find_target_mesh(context)
        if mesh is None:
            self.report({"ERROR"},
                         "No target mesh found. Run Auto-Rig first or "
                         "select the mesh to conform.")
            return {"CANCELLED"}
        rig_arm = _mesh_rig_armature(mesh)
        if rig_arm is None:
            self.report({"ERROR"},
                         f"{mesh.name} has no Armature modifier/parent — "
                         "run Auto-Rig first (the rig provides the shoulder "
                         "pivots)")
            return {"CANCELLED"}

        try:
            donor_arm, donor_mesh = _ensure_source_donor(settings)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            report = autorig_local.conform_mesh_to_uefn(
                mesh, rig_arm, donor_arm, donor_mesh)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"Conform failed: {e}")
            return {"CANCELLED"}

        print(f"[BD_AutoRig:conform] {json.dumps(report, indent=2)}")
        hs = report.get("height_scale", 1.0)
        al = report.get("arm_l", {}).get("scale", "-")
        ar = report.get("arm_r", {}).get("scale", "-")
        self.report({"INFO"},
                     f"Conformed {mesh.name}: height ×{hs}, "
                     f"arm_l ×{al}, arm_r ×{ar}")
        return {"FINISHED"}


class BD_OT_TransferToUEFN(Operator):
    """Bind the autorigged mesh to the canonical UEFN_Mannequin skeleton.

    Wraps scripts/uefn_pipeline/TransferBones_v1.py.

    Expects:
      - "Source" collection: UEFN_Mannequin donor (armature + mesh)
      - The autorigged mesh: active or only MESH in "Collection"/scene

    Produces:
      - "Target" collection (auto-populated with the autorigged mesh)
      - "Export" collection with [root armature, bound mesh] ready for FBX export
    """

    bl_idname = "braindead.transfer_to_uefn"
    bl_label = "Transfer to UEFN Skeleton"
    bl_description = (
        "Bind the autorigged mesh to the canonical SKM_UEFN_Mannequin "
        "skeleton via TransferBones_v1. Requires a 'Source' collection "
        "containing the UEFN_Manny donor."
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.bd_autorig
        tb_path = _find_transferbones()
        if tb_path is None:
            self.report({"ERROR"},
                         "TransferBones_v1.py not found (probed "
                         "autorig_vendor/ and ../scripts/uefn_pipeline/)")
            return {"CANCELLED"}

        # Make sure the UEFN donor is present (auto-append from library)
        try:
            donor_arm, donor_mesh = _ensure_source_donor(settings)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        src_col = bpy.data.collections["Source"]

        target = _find_target_mesh(context)
        if target is None:
            self.report({"ERROR"},
                         "No target mesh found. Run Auto-Rig first or "
                         "select the mesh to transfer.")
            return {"CANCELLED"}

        # Pre-bind proportion fit: uniformly scale the character to the
        # donor's height (never distorts), then fit the DONOR skeleton +
        # mannequin mesh onto the character's anatomy. The character mesh
        # itself is never deformed before binding — the old mesh-conform
        # approach squashed hands via its anisotropic arm scaling.
        # Idempotent.
        if settings.conform_on_transfer:
            rig_arm = _mesh_rig_armature(target)
            if rig_arm is not None:
                try:
                    from . import autorig_local
                    rep = autorig_local.match_height_to_donor(
                        target, rig_arm,
                        donor_mesh if donor_mesh is not None else donor_arm)
                    rep.update(autorig_local.fit_donor_to_character(
                        donor_arm, donor_mesh, rig_arm, char_mesh=target))
                    print(f"[BD_AutoRig:fit] {json.dumps(rep)}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.report({"WARNING"}, f"Pre-bind skeleton fit failed "
                                 f"(continuing unfitted): {e}")
            else:
                self.report({"WARNING"},
                             "Pre-bind skeleton fit skipped: mesh has no "
                             "rig armature (run Auto-Rig first)")

        # Move target into "Target" collection
        tgt_col = bpy.data.collections.get("Target")
        if tgt_col is None:
            tgt_col = bpy.data.collections.new("Target")
            context.scene.collection.children.link(tgt_col)
        for c in list(target.users_collection):
            if c is not tgt_col:
                c.objects.unlink(target)
        if target.name not in tgt_col.objects:
            tgt_col.objects.link(target)

        # Make sure Source + Target collections aren't excluded from view layer
        def _enable(lc):
            if lc.collection.name in ("Source", "Target"):
                lc.exclude = False
            for c in lc.children:
                _enable(c)
        _enable(context.view_layer.layer_collection)
        for o in src_col.all_objects:
            if o is not None:
                o.hide_viewport = False

        # Exec TransferBones_v1 in an isolated namespace.
        # Override its TRANSFER_WEIGHTS module-level constant from our
        # operator setting BEFORE main() runs.
        try:
            src = tb_path.read_text(encoding="utf-8")
            ns = {"__name__": "__transferbones__", "__file__": str(tb_path)}
            exec(compile(src, str(tb_path), "exec"), ns)
            ns["TRANSFER_WEIGHTS"] = bool(settings.transfer_weights)
            if "main" in ns:
                ns["main"]()
            else:
                self.report({"ERROR"},
                             "TransferBones_v1.main() not found after exec")
                return {"CANCELLED"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"TransferBones failed: {e}")
            return {"CANCELLED"}

        # Ensure the new armature is in POSE mode (TransferBones v1 patched
        # to do this itself, but belt-and-suspenders)
        export_col = bpy.data.collections.get("Export")
        if export_col:
            for o in export_col.all_objects:
                if o.type == "ARMATURE":
                    o.data.pose_position = "POSE"
                    break

        # Post-transfer wedge-hand fix: characters with stub hands receive
        # no hand skin weights from the donor transfer (empty hand_l/r
        # vgroups = dead hands on export). Rebind the distal wrist-stub
        # island to the hand bone. Guarded — no-op when the transfer did
        # bind the hand groups.
        if export_col:
            ex_arm = next((o for o in export_col.all_objects
                           if o.type == "ARMATURE"), None)
            ex_mesh = next((o for o in export_col.all_objects
                            if o.type == "MESH"), None)
            if ex_arm is not None and ex_mesh is not None:
                try:
                    from . import autorig_local
                    autorig_local.remap_empty_hand_weights(ex_arm, ex_mesh)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.report({"WARNING"},
                                f"Hand weight remap failed (continuing): {e}")

        self.report({"INFO"},
                     "UEFN bone transfer complete — see 'Export' collection")
        return {"FINISHED"}


# ── Convert-to-A-pose operator ───────────────────────────────────────────────

def _copy_rest_pose_from_reference(target_arm: bpy.types.Object,
                                        reference_arm: bpy.types.Object,
                                        deform_mesh: bool = True,
                                        conform_lengths: bool = True) -> tuple[int, list[str]]:
    """Set the target's rest pose to the reference's, moving the mesh
    along with the bones.

    Two phases:
      1. Mesh bake (if deform_mesh): pose the target bones onto the
         reference rest matrices (name match, root→leaf), which deforms
         the skinned meshes via the armature modifier, then apply the
         modifier to bake that shape into the mesh data (and re-add it).
         Safe ONLY because the pre-bind conform step already matched the
         mesh proportions to the canonical bones — on unconformed meshes
         this is what produced the warped extremities.
      2. Bone joint-snap: copy each reference bone's rest head/tail/roll
         onto the target's edit_bones by name match — exact placement,
         no accumulated pose error.

    Args:
        target_arm: armature to modify (the bound Export rig)
        reference_arm: armature to read rest data from
        deform_mesh: bake the skinned meshes into the new rest shape so
            the mesh visually follows the bones

    Returns:
        (copied_count, missing_bone_names)
    """
    from . import autorig_local

    target_names = {b.name for b in target_arm.data.bones}
    missing = [b.name for b in reference_arm.data.bones
                 if b.name not in target_names]

    if conform_lengths:
        # Exact canonical skeleton: heads, tails, and rolls verbatim from
        # the reference — the export rig matches the mannequin's reference
        # pose 1:1 (guaranteed animation compatibility, correct bone
        # frames). The mesh follows via joint-swing LBS.
        xf = target_arm.matrix_world.inverted_safe() @ reference_arm.matrix_world
        xf3 = xf.to_3x3()
        new_heads, new_tails, roll_z = {}, {}, {}
        for rb in reference_arm.data.bones:
            if rb.name not in target_names:
                continue
            new_heads[rb.name] = xf @ rb.head_local
            new_tails[rb.name] = xf @ rb.tail_local
            roll_z[rb.name] = xf3 @ rb.matrix_local.to_3x3().col[2]
    else:
        # Anchored A-pose retarget: arm chains march from the FITTED
        # clavicle with canonical directions at the character's own
        # segment lengths; every other joint keeps its anatomical
        # position; all bones get canonical frames. Joints stay inside
        # the mesh — UE translation retargeting absorbs the proportion
        # differences from the mannequin.
        new_heads, roll_z, new_tails = autorig_local.compute_apose_anchored(
            target_arm, reference_arm)

    meshes = []
    if deform_mesh:
        meshes = [o for o in bpy.data.objects
                    if o.type == "MESH"
                    and any(m.type == "ARMATURE" and m.object == target_arm
                             for m in o.modifiers)]

    copied, stats = autorig_local.retarget_joints(
        target_arm, new_heads, meshes=meshes, roll_z=roll_z,
        new_tails=new_tails)

    # Re-ground: leg-direction changes can sink the feet a couple of cm.
    # Shift every bone except root (root must stay at the origin for
    # UEFN) plus the meshes straight up so the lowest vert sits on Z=0.
    # SANITY GUARD: this is a centimeters-level nudge. A large offset
    # means an upstream scale/measurement bug — shifting the bones then
    # would destroy the canonical 1:1 skeleton match this function just
    # wrote, so refuse and report instead.
    if meshes:
        z_min = min((o.matrix_world @ v.co).z
                      for o in meshes for v in o.data.vertices)
        if abs(z_min) > 0.5:
            print(f"[BD_AutoRig:set_rest_pose] WARNING mesh feet at "
                  f"Z={z_min:.3f}m — too large to re-ground safely "
                  f"(upstream scale bug?). Bones left canonical; fix the "
                  f"mesh/import alignment instead.")
        elif abs(z_min) > 0.005:
            from mathutils import Vector as _V
            dz = _V((0, 0, -z_min))
            # Origin-helper top-level bones (attach, ik roots — at the
            # armature origin) must not leave it
            ground_heads = {b.name: b.head_local + dz
                              for b in target_arm.data.bones
                              if not (b.parent is None
                                       and b.head_local.length < 0.01)}
            autorig_local.retarget_joints(target_arm, ground_heads,
                                             meshes=meshes)
            print(f"[BD_AutoRig:set_rest_pose] Re-grounded by {-z_min:.4f}m")

    print(f"[BD_AutoRig:set_rest_pose] FK-retargeted {copied} bones "
           f"({len(missing)} ref bones absent), mesh moved: {stats}")
    return copied, missing


class BD_OT_SetRestPose(Operator):
    """Set the Export rig's rest pose by copying bone rest matrices from a
    reference skeleton (T-pose / A-pose / custom FBX), keyed on bone name.

    Mesh weights stay on the bones (vgroup name match), so the mesh
    deforms to follow the new rest positions.
    """

    bl_idname = "braindead.set_rest_pose"
    bl_label = "Set Rest Pose"
    bl_description = (
        "Copy bone rest matrices from a reference skeleton onto the Export "
        "armature. Source determined by the 'Target Rest Pose' setting: "
        "T-pose (current Source/donor armature), A-pose (the canonical Fab "
        "UEFN_Mannequin FBX), or a custom reference FBX path."
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bpy.data.collections.get("Export") is not None

    def execute(self, context):
        settings = context.scene.bd_autorig
        export_col = bpy.data.collections.get("Export")
        tgt_arm = next((o for o in export_col.all_objects if o.type == "ARMATURE"),
                          None) if export_col else None
        if tgt_arm is None:
            self.report({"ERROR"}, "No armature in Export collection")
            return {"CANCELLED"}

        mode = settings.target_pose
        ref_arm = None
        cleanup_objs = []

        if mode == "T_POSE":
            # Use existing donor in Source collection
            src_col = bpy.data.collections.get("Source")
            if src_col is None:
                self.report({"ERROR"},
                             "No Source collection — can't find T-pose donor")
                return {"CANCELLED"}
            # Cleanup: if Source still has the duplicated 'root' from
            # TransferBones (also linked to Export), exclude it
            export_arms = {o.name for o in export_col.all_objects
                              if o.type == "ARMATURE"}
            src_arms = [o for o in src_col.all_objects
                          if o.type == "ARMATURE"
                          and o.name not in export_arms]
            if not src_arms:
                self.report({"ERROR"},
                             "No standalone donor armature in Source "
                             "(only the Export duplicate present)")
                return {"CANCELLED"}
            ref_arm = src_arms[0]

        elif mode in ("A_POSE", "CUSTOM"):
            # Load reference FBX (or A-pose default), grab its armature
            from pathlib import Path
            fbx_path = settings.pose_reference_fbx if mode == "CUSTOM" else \
                       (settings.pose_reference_fbx or
                        r"C:\GameDev\P4\bizanator_Eros_Dev_2540\DCC\Characters"
                        r"\UEFN_Mannequin\fbx\skm_uefn_mannequin.FBX")
            if not Path(fbx_path).exists():
                self.report({"ERROR"},
                             f"Reference FBX not found: {fbx_path}")
                return {"CANCELLED"}

            pre = {o.name for o in bpy.data.objects}
            try:
                # automatic_bone_orientation must stay OFF: it invents
                # bone axes, and the reference's rolls become the export
                # skeleton's canonical orientations for animation
                # compatibility
                bpy.ops.import_scene.fbx(
                    filepath=str(fbx_path),
                    automatic_bone_orientation=False,
                    use_anim=False,
                )
            except Exception as e:
                self.report({"ERROR"}, f"FBX import failed: {e}")
                return {"CANCELLED"}
            new_names = [o.name for o in bpy.data.objects if o.name not in pre]
            cleanup_objs = new_names
            ref_arm = next((bpy.data.objects[n] for n in new_names
                              if bpy.data.objects[n].type == "ARMATURE"), None)
            if ref_arm is None:
                self.report({"ERROR"},
                             f"No armature found in {fbx_path}")
                for n in cleanup_objs:
                    o = bpy.data.objects.get(n)
                    if o: bpy.data.objects.remove(o, do_unlink=True)
                return {"CANCELLED"}

        try:
            copied, missing = _copy_rest_pose_from_reference(
                tgt_arm, ref_arm,
                deform_mesh=settings.deform_mesh_on_rest_change,
                conform_lengths=settings.conform_bone_lengths)
            self.report({"INFO"},
                         f"Set rest pose ({mode}): {copied} bones copied, "
                         f"{len(missing)} not in reference")
            if missing:
                print(f"[BD_AutoRig:set_rest_pose] missing in reference: "
                       f"{missing[:15]}{'...' if len(missing) > 15 else ''}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"Rest pose copy failed: {e}")
            return {"CANCELLED"}
        finally:
            # Clean up the temporary import (CUSTOM/A_POSE branches)
            for n in cleanup_objs:
                o = bpy.data.objects.get(n)
                if o:
                    try: bpy.data.objects.remove(o, do_unlink=True)
                    except Exception: pass
            # Also remove orphan armature data
            for ad in list(bpy.data.armatures):
                if ad.users == 0:
                    try: bpy.data.armatures.remove(ad)
                    except Exception: pass

        return {"FINISHED"}


# ── Export FBX operator ──────────────────────────────────────────────────────

class BD_OT_ExportUEFNFBX(Operator):
    """Export the Export collection to a UEFN-ready FBX (Add Leaf Bones OFF,
    Skinned, Y forward, Z up — Unreal conventions). Optionally applies the
    Target Rest Pose first so the FBX writes the chosen rest pose."""

    bl_idname = "braindead.export_uefn_fbx"
    bl_label = "Export UEFN FBX"
    bl_description = (
        "Export the Export collection contents (armature + skinned mesh) "
        "to a UEFN-compatible FBX. Applies the Target Rest Pose first."
    )
    bl_options = {"REGISTER"}

    filepath: StringProperty(
        name="File Path",
        description="Output FBX path",
        subtype="FILE_PATH",
        default="//exported_uefn.fbx",
    )

    apply_rest_pose: BoolProperty(
        name="Apply Target Rest Pose first",
        description="Run Set Rest Pose with the current setting before exporting",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return bpy.data.collections.get("Export") is not None

    def invoke(self, context, event):
        if self.filepath in ("", "//exported_uefn.fbx"):
            blendpath = bpy.data.filepath
            if blendpath:
                from pathlib import Path
                self.filepath = str(Path(blendpath).parent / "exported_uefn.fbx")
            else:
                import tempfile
                self.filepath = str(Path(tempfile.gettempdir()) /
                                       "exported_uefn.fbx")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if self.apply_rest_pose:
            try:
                bpy.ops.braindead.set_rest_pose()
            except Exception as e:
                self.report({"WARNING"},
                             f"Set Rest Pose failed pre-export: {e}")

        export_col = bpy.data.collections["Export"]
        # Select only Export collection contents
        bpy.ops.object.select_all(action="DESELECT")
        for o in export_col.all_objects:
            o.select_set(True)
            o.hide_viewport = False
        arm = next((o for o in export_col.all_objects if o.type == "ARMATURE"), None)
        if arm:
            context.view_layer.objects.active = arm

        try:
            bpy.ops.export_scene.fbx(
                filepath=self.filepath,
                use_selection=True,
                add_leaf_bones=False,           # UEFN convention
                bake_anim=False,
                object_types={"ARMATURE", "MESH"},
                mesh_smooth_type="FACE",
                use_armature_deform_only=True,
                primary_bone_axis="Y",          # Mixamo/UEFN style
                secondary_bone_axis="X",
                axis_forward="-Z",              # default for Unreal
                axis_up="Y",
                global_scale=1.0,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self.report({"ERROR"}, f"FBX export failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported UEFN FBX → {self.filepath}")
        return {"FINISHED"}


# ── One-click pipeline operator ──────────────────────────────────────────────

class BD_OT_RunFullUEFNPipeline(Operator):
    """One click from a selected character mesh to a UEFN-ready rig in the
    Export collection: Auto-Rig → scene tidy → Conform + Transfer to UEFN
    → Set Rest Pose. Export FBX stays a separate button (it needs a path).
    """

    bl_idname = "braindead.run_full_uefn_pipeline"
    bl_label = "One-Click UEFN Rig"
    bl_description = (
        "Run the whole pipeline on the active mesh: Auto-Rig (MIA), "
        "conform proportions to the UEFN mannequin, bind to the canonical "
        "87-bone skeleton with weight transfer, and set the target rest "
        "pose. Leaves a clean scene with the result in the 'Export' "
        "collection; the original import is hidden, donor collections "
        "excluded. Click 'Export UEFN FBX' afterwards to write the file"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH"

    def _fail(self, step, detail=""):
        self.report({"ERROR"}, f"Pipeline stopped at {step}. {detail}".strip())
        return {"CANCELLED"}

    def execute(self, context):
        src_mesh = context.active_object
        pre_objs = {o.name for o in bpy.data.objects}

        # 1) Auto-Rig the active mesh
        try:
            res = bpy.ops.braindead.autorig_mesh()
        except Exception as e:
            return self._fail("Auto-Rig", str(e))
        if "FINISHED" not in res:
            return self._fail("Auto-Rig", "see System Console")

        new_names = [o.name for o in bpy.data.objects
                        if o.name not in pre_objs]
        rigged = next((bpy.data.objects[n] for n in new_names
                          if bpy.data.objects[n].type == "MESH"), None)
        if rigged is None:
            return self._fail("Auto-Rig", "no rigged mesh produced")
        mia_rig = _mesh_rig_armature(rigged)

        # 2) Tidy the autorig import: drop GLB scene empties, hide the
        # user's original mesh (never delete a user import)
        for n in new_names:
            o = bpy.data.objects.get(n)
            if o is not None and o.type == "EMPTY":
                bpy.data.objects.remove(o, do_unlink=True)
        src_mesh.hide_render = True
        src_mesh.hide_viewport = True

        # 3) Conform + Transfer to UEFN (conform runs inside per setting)
        bpy.ops.object.select_all(action="DESELECT")
        rigged.select_set(True)
        context.view_layer.objects.active = rigged
        try:
            res = bpy.ops.braindead.transfer_to_uefn()
        except Exception as e:
            return self._fail("Transfer to UEFN", str(e))
        if "FINISHED" not in res:
            return self._fail("Transfer to UEFN", "see System Console")

        # 4) Set target rest pose on the Export rig
        try:
            res = bpy.ops.braindead.set_rest_pose()
        except Exception as e:
            return self._fail("Set Rest Pose", str(e))
        if "FINISHED" not in res:
            return self._fail("Set Rest Pose", "see System Console")

        # 5) Tidy: the intermediate MIA rig is orphaned after the rebind
        if mia_rig is not None and rigged.parent is not mia_rig:
            export_col = bpy.data.collections.get("Export")
            in_export = export_col is not None and any(
                o.name == mia_rig.name for o in export_col.all_objects)
            if not in_export:
                bpy.data.objects.remove(mia_rig, do_unlink=True)
        # Exclude donor/intermediate collections from the view layer
        def _exclude(lc):
            if lc.collection.name in ("Source", "Target"):
                lc.exclude = True
            for c in lc.children:
                _exclude(c)
        _exclude(context.view_layer.layer_collection)

        # Leave the Export armature selected + active
        export_col = bpy.data.collections.get("Export")
        exp_arm = next((o for o in export_col.all_objects
                           if o.type == "ARMATURE"), None) if export_col else None
        if exp_arm is not None:
            bpy.ops.object.select_all(action="DESELECT")
            exp_arm.select_set(True)
            context.view_layer.objects.active = exp_arm

        settings = context.scene.bd_autorig
        self.report({"INFO"},
                     f"UEFN rig complete ({settings.target_pose}) — see "
                     "'Export' collection. Use 'Export UEFN FBX' to write "
                     "the file.")
        return {"FINISHED"}


# ── Bootstrap operator ───────────────────────────────────────────────────────

class BD_OT_InstallLocalAutoRig(Operator):
    """One-time install of the local autorig venv (PyTorch + MIA + weights)."""

    bl_idname = "braindead.install_local_autorig"
    bl_label = "Install Local Autorig"
    bl_description = (
        "Download + install PyTorch, Make-It-Animatable, and model weights "
        "into a managed venv (~/.cache/braindead_blender/autorig/). "
        "Required one-time setup for the Local backend. "
        "Takes 5-15 minutes depending on network speed and CUDA availability."
    )

    def execute(self, context):
        from . import autorig_bootstrap

        if autorig_bootstrap.is_bootstrapped():
            self.report({"INFO"},
                         "Local autorig already installed at "
                         f"{autorig_bootstrap.CACHE_ROOT}")
            return {"FINISHED"}

        def progress(msg):
            print(f"[BD_AutoRig:install] {msg}", flush=True)

        self.report({"INFO"},
                     "Bootstrapping local autorig — see System Console for "
                     "progress (this can take 5-15 minutes)…")
        ok = autorig_bootstrap.bootstrap(progress_cb=progress)
        if not ok:
            self.report({"ERROR"},
                         "Bootstrap failed — see System Console for details")
            return {"CANCELLED"}

        ok = autorig_bootstrap.ensure_mia_weights(progress_cb=progress)
        if not ok:
            self.report({"ERROR"},
                         "Weight download failed — see System Console")
            return {"CANCELLED"}

        self.report({"INFO"},
                     "Local autorig install complete. Ready to run.")
        return {"FINISHED"}


# ── UI panel ─────────────────────────────────────────────────────────────────

class BD_PT_AutoRig(Panel):
    """Sidebar panel for BrainDead AutoRig."""

    bl_label = "BD AutoRig"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BrainDead"

    def draw(self, context):
        layout = self.layout
        s: BD_AutoRigSettings = context.scene.bd_autorig

        col = layout.column(align=True)
        col.label(text="Backend")
        col.prop(s, "backend", text="")
        if s.backend == "custom":
            col.prop(s, "comfy_url", text="URL")
        if s.backend == "local_managed":
            try:
                from . import autorig_bootstrap
                installed = autorig_bootstrap.is_bootstrapped()
            except Exception:
                installed = False
            row = col.row(align=True)
            if installed:
                row.label(text="Local: installed", icon="CHECKMARK")
            else:
                row.label(text="Local: NOT installed", icon="ERROR")
            col.operator(BD_OT_InstallLocalAutoRig.bl_idname,
                            icon="IMPORT")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Rigger")
        col.prop(s, "rigger", text="")
        if s.rigger == "mia":
            col.prop(s, "no_fingers")
        col.prop(s, "reset_to_rest")
        col.prop(s, "remap_to_uefn")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Post-Import")
        col.prop(s, "run_posefixer")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Network")
        row = col.row(align=True)
        row.prop(s, "poll_timeout_sec", text="Timeout")
        row.prop(s, "poll_interval_sec", text="Interval")

        layout.separator()
        big = layout.row()
        big.scale_y = 1.8
        big.operator(BD_OT_RunFullUEFNPipeline.bl_idname, icon="PLAY")

        layout.separator()
        big = layout.row()
        big.scale_y = 1.5
        big.operator(BD_OT_AutoRigMesh.bl_idname, icon="ARMATURE_DATA")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="UEFN Skeleton Transfer")
        col.prop(s, "donor_blend", text="Donor")
        col.prop(s, "conform_on_transfer")
        row = col.row(align=True)
        row.operator(BD_OT_ConformMeshToUEFN.bl_idname, icon="FULLSCREEN_EXIT")
        col.prop(s, "transfer_weights")
        row = col.row(align=True)
        row.scale_y = 1.3
        row.operator(BD_OT_TransferToUEFN.bl_idname, icon="OUTLINER_DATA_ARMATURE")

        col.separator()
        col.label(text="Rest Pose")
        col.prop(s, "target_pose", text="")
        col.prop(s, "conform_bone_lengths")
        col.prop(s, "deform_mesh_on_rest_change")
        if s.target_pose == "CUSTOM":
            col.prop(s, "pose_reference_fbx", text="FBX")
        row = col.row(align=True)
        row.scale_y = 1.3
        row.operator(BD_OT_SetRestPose.bl_idname, icon="POSE_HLT")

        col.separator()
        col.label(text="Export")
        row = col.row(align=True)
        row.scale_y = 1.3
        row.operator(BD_OT_ExportUEFNFBX.bl_idname, icon="EXPORT")


# ── Registration ─────────────────────────────────────────────────────────────

_classes = (
    BD_AutoRigSettings,
    BD_OT_AutoRigMesh,
    BD_OT_RunFullUEFNPipeline,
    BD_OT_ConformMeshToUEFN,
    BD_OT_TransferToUEFN,
    BD_OT_SetRestPose,
    BD_OT_ExportUEFNFBX,
    BD_OT_InstallLocalAutoRig,
    BD_PT_AutoRig,
)


def register():
    from bpy.utils import register_class
    for c in _classes:
        register_class(c)
    bpy.types.Scene.bd_autorig = bpy.props.PointerProperty(
        type=BD_AutoRigSettings)


def unregister():
    from bpy.utils import unregister_class
    for c in reversed(_classes):
        try:
            unregister_class(c)
        except Exception:
            pass
    try:
        del bpy.types.Scene.bd_autorig
    except Exception:
        pass
