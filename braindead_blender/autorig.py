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


def _extract_output_path(history_entry: dict) -> str:
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
            remote_fbx = _extract_output_path(entry)
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

        # 5) Import + optionally retarget
        try:
            bpy.ops.import_scene.fbx(
                filepath=str(local_fbx),
                automatic_bone_orientation=True,
                use_anim=False,
            )
        except Exception as e:
            self.report({"ERROR"}, f"FBX import failed: {e}")
            return {"CANCELLED"}
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

    @classmethod
    def poll(cls, context):
        return bpy.data.collections.get("Source") is not None

    def execute(self, context):
        tb_path = _find_transferbones()
        if tb_path is None:
            self.report({"ERROR"},
                         "TransferBones_v1.py not found (probed "
                         "autorig_vendor/ and ../scripts/uefn_pipeline/)")
            return {"CANCELLED"}

        # Find target mesh: prefer active, else the most-recently-imported
        # MIA_Character, else the largest non-Source mesh
        src_col = bpy.data.collections["Source"]
        src_meshes = {o.name for o in src_col.all_objects if o.type == "MESH"}

        target = None
        if (context.active_object is not None
                and context.active_object.type == "MESH"
                and context.active_object.name not in src_meshes):
            target = context.active_object
        else:
            candidates = [o for o in bpy.data.objects
                              if o.type == "MESH"
                              and o.name not in src_meshes
                              and not o.hide_viewport
                              and "world" not in o.name.lower()]
            mia = [o for o in candidates if "MIA_Character" in o.name]
            if mia:
                target = mia[0]
            elif candidates:
                target = max(candidates, key=lambda o: len(o.data.vertices))

        if target is None:
            self.report({"ERROR"},
                         "No target mesh found. Run Auto-Rig first or "
                         "select the mesh to transfer.")
            return {"CANCELLED"}

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
        settings = context.scene.bd_autorig
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

        self.report({"INFO"},
                     "UEFN bone transfer complete — see 'Export' collection")
        return {"FINISHED"}


# ── Convert-to-A-pose operator ───────────────────────────────────────────────

def _copy_rest_pose_from_reference(target_arm: bpy.types.Object,
                                        reference_arm: bpy.types.Object) -> tuple[int, list[str]]:
    """Snap target's rest pose to reference's rest pose by name match.

    Algorithm: direct edit-mode copy of head/tail/roll for each matching
    bone. Skinned mesh deformation is computed as:
        vert_world = bone_pose @ inv(bone_rest) @ vert_local
    At rest, bone_pose == bone_rest, so vert_world == vert_local — i.e.
    the mesh stays put even when we change the rest matrix. This means
    we can simply overwrite the rest positions without rebinding.

    Use bone.matrix to write both head/tail and roll in one assignment.

    Args:
        target_arm: armature to modify (the bound Export rig)
        reference_arm: armature to read rest data from

    Returns:
        (copied_count, missing_bone_names)
    """
    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.object.mode_set(mode="EDIT")

    copied = 0
    missing: list[str] = []
    try:
        # Walk bones from root to leaves so parents update before children
        # (Blender's edit_bones iteration isn't guaranteed in chain order,
        # so collect + sort by hierarchy depth)
        def _depth(bone):
            d = 0
            b = bone
            while b.parent is not None:
                b = b.parent
                d += 1
            return d
        sorted_refs = sorted(reference_arm.data.bones, key=_depth)

        for ref_bone in sorted_refs:
            eb = target_arm.data.edit_bones.get(ref_bone.name)
            if eb is None:
                missing.append(ref_bone.name)
                continue
            # Copy head + tail in armature space — exactly the rest pose
            eb.head = ref_bone.head_local.copy()
            eb.tail = ref_bone.tail_local.copy()
            # Roll: derive from reference's matrix_local Z column
            eb.roll = 0  # neutral; matrix overwrite next will set proper roll
            # Then set the full matrix so bone_local rotation matches
            eb.matrix = ref_bone.matrix_local.copy()
            copied += 1
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    target_arm.data.pose_position = "POSE"
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
                bpy.ops.import_scene.fbx(
                    filepath=str(fbx_path),
                    automatic_bone_orientation=True,
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
            copied, missing = _copy_rest_pose_from_reference(tgt_arm, ref_arm)
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

        # Belt-and-suspenders: make sure Export armature is in POSE mode
        export_col = bpy.data.collections.get("Export")
        if export_col:
            for o in export_col.all_objects:
                if o.type == "ARMATURE":
                    o.data.pose_position = "POSE"
                    break

        self.report({"INFO"}, "A-pose conversion complete")
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
        big.scale_y = 1.5
        big.operator(BD_OT_AutoRigMesh.bl_idname, icon="ARMATURE_DATA")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="UEFN Skeleton Transfer")
        col.prop(s, "transfer_weights")
        row = col.row(align=True)
        row.scale_y = 1.3
        row.operator(BD_OT_TransferToUEFN.bl_idname, icon="OUTLINER_DATA_ARMATURE")

        col.separator()
        col.label(text="Rest Pose")
        col.prop(s, "target_pose", text="")
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
