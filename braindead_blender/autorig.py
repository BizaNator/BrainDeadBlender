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
            ("brainz_dev",    "BrainZ Dev (10.15.0.20:8189)", "ComfyUI dev on BrainZ — Make-It-Animatable + UEFN remap"),
            ("brainz_stable", "BrainZ Stable (10.15.0.20:8188)", "ComfyUI stable on BrainZ"),
            ("local",         "Local ComfyUI (127.0.0.1:8188)", "Local ComfyUI install"),
            ("custom",        "Custom URL", "Set comfy_url manually"),
        ],
        default="brainz_dev",
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
        name="No Fingers",
        description="Merge finger weights into the hand bone (MIA only)",
        default=True,
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
        "local":         "http://127.0.0.1:8188",
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
        comfy_url = _resolve_comfy_url(settings)
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
            # PoseFixer needs Source + Target collections set up; we just
            # report a hint instead of running it blindly.
            self.report({"INFO"},
                         "Tip: set up Source = UEFN_Manny + Target = "
                         "imported armature, then run PoseFixer_v1.py to "
                         "retarget rest pose into A-pose.")

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


# ── Registration ─────────────────────────────────────────────────────────────

_classes = (
    BD_AutoRigSettings,
    BD_OT_AutoRigMesh,
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
