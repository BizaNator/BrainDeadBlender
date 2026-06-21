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

        try:
            bpy.ops.import_scene.fbx(
                filepath=str(local_fbx),
                automatic_bone_orientation=True,
                use_anim=False,
            )
        except Exception as e:
            self.report({"ERROR"}, f"FBX import failed: {e}")
            return {"CANCELLED"}

        # Rename Mixamo-style bones in-place to UEFN canonical (if requested
        # and the FBX still has Mixamo names — Local backend doesn't run the
        # ComfyUI BD_MixamoToUEFN node)
        new_arms = [o for o in bpy.data.objects
                       if o.type == "ARMATURE" and o.name not in pre_arms]
        arm_obj = new_arms[0] if new_arms else None
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

        self.report({"INFO"}, f"Local autorig complete — FBX at {local_fbx}")

        if settings.run_posefixer:
            try:
                _run_posefixer_on_armature(arm_obj)
                self.report({"INFO"}, "PoseFixer A-pose retarget done")
            except Exception as e:
                self.report({"WARNING"}, f"PoseFixer skipped: {e}")

        return {"FINISHED"}


# ── PoseFixer integration ────────────────────────────────────────────────────

_POSEFIXER_PATH = (Path(__file__).resolve().parent.parent /
                     "scripts" / "uefn_pipeline" / "PoseFixer_v1.py")


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
    if not _POSEFIXER_PATH.exists():
        raise RuntimeError(f"PoseFixer_v1.py not found at {_POSEFIXER_PATH}")
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


# ── Registration ─────────────────────────────────────────────────────────────

_classes = (
    BD_AutoRigSettings,
    BD_OT_AutoRigMesh,
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
