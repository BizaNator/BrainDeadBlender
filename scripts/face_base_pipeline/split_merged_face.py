"""
split_merged_face.py

Reverse of `merge_face_meshes.py`. Reads the `_section` face attribute
on a merged head mesh and splits the geometry back into independent
objects -- one per unique section tag -- so each part can be edited /
re-fitted / re-weighted in isolation. Run this when you need to fix
the lips or eyelids on a merged head, then re-run `merge_face_meshes`
to consolidate again.

The split objects:
  - copy verts, faces, UVs, materials, vgroups (with weights) from the
    merged source for the faces in their section.
  - parent to the same armature as the merged mesh, with an Armature
    modifier that uses vertex groups.
  - DO NOT carry shape keys (the merge dropped them; library copies in
    _PartsLibrary still have them if needed).

The merged source mesh stays untouched by default (so you can keep the
in-engine-ready version around). Set `hide_source` to clear it from the
viewport.

Section tag -> object name mapping is controlled by `section_to_object`.
Sections not in the map are split into objects named `<section>` (with
section name as-is).
"""

import bpy
import bmesh


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "source": "LowPolyHead_Rigged",
    "section_attr": "_section",

    # Which section IS the head shell itself -- this stays inside the source
    # object, never extracted as a separate object.
    "head_section": "head",

    # Optional: map section tag -> output object name. If absent, the
    # section tag is used as the object name (e.g. "ear_l" -> "ear_l").
    "section_to_object": {
        "lips": "CustomLips",
        "eyelid_l_upper": "Eyelid_L_Upper",
        "eyelid_l_lower": "Eyelid_L_Lower",
        "eyelid_r_upper": "Eyelid_R_Upper",
        "eyelid_r_lower": "Eyelid_R_Lower",
        "eyebrow_l": "Eyebrow_L",
        "eyebrow_r": "Eyebrow_R",
        "ear_l": "Ear_L",
        "ear_r": "Ear_R",
    },

    # If a name in section_to_object already exists in the scene, replace it.
    "replace_existing": True,

    # Hide the merged source after splitting (it stays in the scene).
    "hide_source": False,

    # Target collection for the split parts (created if missing). None ->
    # source object's own collection.
    "target_collection": None,
}


# ------------------------------- HELPERS ------------------------------------
def _obj(name, required=True):
    o = bpy.data.objects.get(name)
    if required and o is None:
        raise RuntimeError(f"object '{name}' not found")
    return o


def _ensure_collection(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def _remove_object(obj):
    me = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(me, bpy.types.Mesh) and me.users == 0:
        bpy.data.meshes.remove(me)


def _section_value(d):
    """Read a STRING attribute datum, normalising bytes -> str."""
    v = d.value
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    return v


def _faces_by_section(me, attr_name):
    a = me.attributes.get(attr_name)
    if a is None or a.domain != 'FACE' or a.data_type != 'STRING':
        raise RuntimeError(
            f"source mesh has no FACE/STRING attribute '{attr_name}' -- "
            f"was it built by merge_face_meshes?")
    buckets = {}
    for fi, d in enumerate(a.data):
        s = _section_value(d) or ""
        buckets.setdefault(s, []).append(fi)
    return buckets


def _extract_section(src, face_indices, new_name, target_coll):
    """Build a new Object from src restricted to face_indices. Copies
    verts, UVs, materials, vgroups + weights."""
    src_me = src.data

    keep_verts = set()
    keep_faces = [src_me.polygons[fi] for fi in face_indices]
    for p in keep_faces:
        for vi in p.vertices:
            keep_verts.add(vi)

    src_to_new = {}
    new_verts_co = []
    for vi in sorted(keep_verts):
        src_to_new[vi] = len(new_verts_co)
        new_verts_co.append(src_me.vertices[vi].co.copy())
    new_faces = [tuple(src_to_new[vi] for vi in p.vertices) for p in keep_faces]

    new_me = bpy.data.meshes.new(new_name + "_mesh")
    new_me.from_pydata(new_verts_co, [], new_faces)
    new_me.update()

    # Materials (copy ALL slots to preserve indices, then remap).
    for ms in src.material_slots:
        new_me.materials.append(ms.material)
    for ni, p in enumerate(keep_faces):
        new_me.polygons[ni].material_index = p.material_index
        new_me.polygons[ni].use_smooth = p.use_smooth

    # UVs (one new layer per source layer, same name).
    for sl in src_me.uv_layers:
        nl = new_me.uv_layers.new(name=sl.name)
        for ni, p in enumerate(keep_faces):
            np = new_me.polygons[ni]
            for li_new, li_src in zip(np.loop_indices, p.loop_indices):
                nl.data[li_new].uv = sl.data[li_src].uv

    new_obj = bpy.data.objects.new(new_name, new_me)
    new_obj.matrix_world = src.matrix_world.copy()
    target_coll.objects.link(new_obj)

    # Vertex groups: copy all defs so the armature modifier matches the
    # full skeleton; weights only for verts we kept.
    for vg in src.vertex_groups:
        new_obj.vertex_groups.new(name=vg.name)
    name_by_idx = {vg.index: vg.name for vg in src.vertex_groups}
    for src_vi, new_vi in src_to_new.items():
        for g in src_me.vertices[src_vi].groups:
            gname = name_by_idx.get(g.group)
            if gname is None:
                continue
            new_obj.vertex_groups[gname].add([new_vi], g.weight, 'REPLACE')

    return new_obj


def _wire_to_armature(new_obj, src):
    """Parent new_obj to src's armature (if any) and add Armature modifier."""
    arm = src.parent if (src.parent and src.parent.type == 'ARMATURE') else None
    if arm is None:
        for m in src.modifiers:
            if m.type == 'ARMATURE' and m.object is not None:
                arm = m.object
                break
    if arm is None:
        print(f"  '{new_obj.name}': no armature on source -- skipping rig wire")
        return
    mod = new_obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    mod.use_vertex_groups = True
    new_obj.parent = arm
    # Preserve world transform across the re-parent.
    new_obj.matrix_parent_inverse = (
        arm.matrix_world.inverted() @ new_obj.matrix_world @ new_obj.matrix_basis.inverted()
    )


# --------------------------------- ENTRY ------------------------------------
def split_merged_face(cfg):
    src = _obj(cfg["source"])
    print(f"=== split_merged_face <- {src.name} ===")

    target_coll = (_ensure_collection(cfg["target_collection"])
                   if cfg.get("target_collection") else
                   (src.users_collection[0] if src.users_collection
                    else bpy.context.scene.collection))

    buckets = _faces_by_section(src.data, cfg["section_attr"])
    head_section = cfg["head_section"]
    mapping = cfg.get("section_to_object", {})

    created = []
    for section, face_idxs in sorted(buckets.items()):
        if section == head_section:
            print(f"  skip head section: {len(face_idxs)} faces stay in source")
            continue
        if not section:
            print(f"  skip untagged faces: {len(face_idxs)}")
            continue
        new_name = mapping.get(section, section)
        existing = bpy.data.objects.get(new_name)
        if existing and cfg.get("replace_existing", True):
            _remove_object(existing)
        elif existing:
            print(f"  skip '{section}': '{new_name}' already exists")
            continue

        new_obj = _extract_section(src, face_idxs, new_name, target_coll)
        _wire_to_armature(new_obj, src)
        created.append(new_obj)
        print(f"  '{section}' -> '{new_name}': {len(new_obj.data.vertices)}v "
              f"{len(new_obj.data.polygons)}f")

    if cfg.get("hide_source", False):
        src.hide_set(True)

    print(f"\n[done] split {len(created)} parts from '{src.name}'")
    return created


if __name__ == "__main__":
    split_merged_face(CONFIG)
