"""
split_face_parts.py

Split a single LowPolyHead_Parts object (with separate vgroups for
L_eye / R_eye / teeth_upper / teeth_lower / tongue, all on one mesh)
into a set of INDEPENDENT objects -- one per face component -- so each
can be translated / scaled / hidden independently while still riding
the same Penny armature.

Why
---
The eyes and teeth on the original LowPolyHead_Parts share a mesh, so
moving one moves them all. To fit Penny's eyeballs into the lowpoly
head's eye sockets while letting the teeth sit at the (different)
mouth position, they need to be separate Objects.

After splitting, the originals are kept (renamed with `_combined`
suffix) and hidden by default; the new per-component objects each
carry the relevant vgroup + shape keys + armature modifier + parent.
Each can be selected and G-key dragged into place.

Drop into the BrainDeadBlender add-on next to align_landmarks /
headswap_transfer / cut_face_holes.
"""

import bpy
import bmesh
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "source":   "LowPolyHead_Parts",
    "armature": "Fortnite_Armature",

    # One output object per entry. `name` is the new object name; each
    # entry pulls verts/faces whose dominant weight (above min_weight) is
    # in the listed vgroup(s).
    "splits": [
        {"name": "Eye_L",        "vgroups": ["L_eye"]},
        {"name": "Eye_R",        "vgroups": ["R_eye"]},
        {"name": "Teeth_Upper",  "vgroups": ["teeth_upper"]},
        {"name": "Teeth_Lower",  "vgroups": ["teeth_lower"]},
        {"name": "Tongue",       "vgroups": ["tongue"]},
    ],

    "min_weight": 0.5,

    # Hide the original combined parts after splitting (it stays in the
    # scene under `<source>_combined` so weights are recoverable).
    "hide_source": True,

    # Keep ALL vgroups on each split object (including bones that don't
    # weight the split's verts) so the armature modifier matches the
    # full skeleton. Set False to keep only the relevant vgroups.
    "keep_all_vgroups": True,
}


# ------------------------------- UTILITIES ----------------------------------
def _obj(name):
    o = bpy.data.objects.get(name)
    if o is None:
        raise RuntimeError(f"object '{name}' not found")
    return o


def _make_submesh(src, vgroup_names, min_weight, new_name, arm):
    """Build a new Object containing only the verts/faces dominantly
    weighted to one of `vgroup_names`. Returns the new Object."""
    src_me = src.data
    target_idxs = set()
    for n in vgroup_names:
        vg = src.vertex_groups.get(n)
        if vg is not None:
            target_idxs.add(vg.index)
    if not target_idxs:
        print(f"  skip '{new_name}': no source vgroup matches {vgroup_names}")
        return None

    # Pick verts where any target vgroup has weight > min_weight
    keep_verts = set()
    for v in src_me.vertices:
        for g in v.groups:
            if g.group in target_idxs and g.weight > min_weight:
                keep_verts.add(v.index)
                break
    if not keep_verts:
        print(f"  skip '{new_name}': no verts above weight {min_weight}")
        return None

    # Faces whose ALL verts are kept
    keep_faces = [p for p in src_me.polygons if all(vi in keep_verts for vi in p.vertices)]
    if not keep_faces:
        print(f"  skip '{new_name}': no faces fully inside the kept verts")
        return None

    # Build new mesh
    new_me = bpy.data.meshes.new(new_name + "_mesh")
    src_idx_to_new = {}
    new_verts = []
    for vi in keep_verts:
        src_idx_to_new[vi] = len(new_verts)
        new_verts.append(src_me.vertices[vi].co.copy())
    new_faces = [tuple(src_idx_to_new[vi] for vi in p.vertices) for p in keep_faces]
    new_me.from_pydata(new_verts, [], new_faces)
    new_me.update()

    # Materials
    for ms in src.material_slots:
        new_me.materials.append(ms.material)
    for ni, p in enumerate(keep_faces):
        new_me.polygons[ni].material_index = p.material_index

    # UVs
    if src_me.uv_layers:
        for sl in src_me.uv_layers:
            nl = new_me.uv_layers.new(name=sl.name)
            for ni, p in enumerate(keep_faces):
                np = new_me.polygons[ni]
                for li_new, li_src in zip(np.loop_indices, p.loop_indices):
                    nl.data[li_new].uv = sl.data[li_src].uv

    new_obj = bpy.data.objects.new(new_name, new_me)
    new_obj.matrix_world = src.matrix_world.copy()
    for coll in src.users_collection:
        coll.objects.link(new_obj)
    if not new_obj.users_collection:
        bpy.context.scene.collection.objects.link(new_obj)

    return new_obj, src_idx_to_new


def _copy_vgroups(src, new_obj, src_idx_to_new, keep_all=True, restrict_to=None):
    """Copy vgroup definitions + per-vert weights from src to new_obj."""
    for vg in src.vertex_groups:
        if not keep_all and restrict_to is not None and vg.name not in restrict_to:
            continue
        new_obj.vertex_groups.new(name=vg.name)
    for src_vi, new_vi in src_idx_to_new.items():
        for g in src.data.vertices[src_vi].groups:
            gname = src.vertex_groups[g.group].name
            if gname in new_obj.vertex_groups:
                new_obj.vertex_groups[gname].add([new_vi], g.weight, 'REPLACE')


def _copy_shape_keys(src, new_obj, src_idx_to_new):
    """Copy shape keys, picking only the verts that ended up in new_obj."""
    if src.data.shape_keys is None:
        return
    new_obj.shape_key_add(name="Basis", from_mix=False)
    for src_kb in src.data.shape_keys.key_blocks:
        if src_kb.name == "Basis":
            continue
        new_kb = new_obj.shape_key_add(name=src_kb.name, from_mix=False)
        for src_vi, new_vi in src_idx_to_new.items():
            new_kb.data[new_vi].co = src_kb.data[src_vi].co.copy()


# --------------------------------- ENTRY ------------------------------------
def split_face_parts(cfg):
    src = _obj(cfg["source"])
    arm = _obj(cfg["armature"])
    print(f"=== split_face_parts -> {src.name} ===")

    created = []
    for entry in cfg["splits"]:
        out = _make_submesh(src, entry["vgroups"], cfg["min_weight"],
                            entry["name"], arm)
        if out is None:
            continue
        new_obj, idx_map = out

        _copy_vgroups(src, new_obj, idx_map,
                      keep_all=cfg["keep_all_vgroups"],
                      restrict_to=set(entry["vgroups"]))
        _copy_shape_keys(src, new_obj, idx_map)

        mod = new_obj.modifiers.new("Armature", 'ARMATURE')
        mod.object = arm
        mod.use_vertex_groups = True
        new_obj.parent = arm
        new_obj.matrix_parent_inverse = arm.matrix_world.inverted()

        created.append(new_obj)
        print(f"  '{new_obj.name}': {len(new_obj.data.vertices)}v "
              f"{len(new_obj.data.polygons)}f vgroups={len(new_obj.vertex_groups)}")

    if cfg["hide_source"]:
        old_name = src.name
        src.name = old_name + "_combined"
        src.hide_set(True)
        print(f"\n  hid + renamed source -> '{src.name}'")

    print(f"\n[done] created {len(created)} submesh objects")
    return created


if __name__ == "__main__":
    split_face_parts(CONFIG)
