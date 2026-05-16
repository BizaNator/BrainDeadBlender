"""
merge_face_meshes.py

Merge face sub-meshes (CustomLips, Eyelid_*, Eyebrow_*, Ear_*, ...) into
the head shell so the face deforms as one continuous surface (like
Penny / Fortnite heads do). Each merged face is tagged with a `_section`
face attribute so `split_merged_face.py` can round-trip the merge back
to independent objects for editing.

Run AFTER `build_parts_library.py` (so a clean copy of the originals is
preserved in `_PartsLibrary`) and AFTER all per-part fitting / weighting
is done. Eyes, teeth, tongue, hair stay SEPARATE -- they have their own
materials and animate independently.

Boundary welding: a single `bmesh.ops.remove_doubles` pass with a small
tolerance (default 0.5mm) merges coincident verts at the section seams
without collapsing interior geometry. Tune `merge_distance` if the
source meshes don't share exact boundary vert positions.

Output: a single merged mesh at `cfg["target"]` with:
  - face attribute `_section` (string) per face
  - union of all source vertex groups (weights preserved)
  - union of all source materials
  - target's shape keys preserved; source shape keys are DROPPED with
    a warning (re-transfer them after split if needed).
"""

import bpy
import bmesh
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    # The head shell receives the merge. All sources are appended INTO it.
    # This object is modified in place; library copies preserve originals.
    "target": "LowPolyHead_Rigged",

    # Section tag for the target's own faces.
    "target_section": "head",

    # Sources to merge in. Each entry: object name + section tag written
    # to its faces. Section tags MUST be unique (used by split to round-trip).
    "sources": [
        {"object": "CustomLips",       "section": "lips"},
        {"object": "Eyelid_L_Upper",   "section": "eyelid_l_upper"},
        {"object": "Eyelid_L_Lower",   "section": "eyelid_l_lower"},
        {"object": "Eyelid_R_Upper",   "section": "eyelid_r_upper"},
        {"object": "Eyelid_R_Lower",   "section": "eyelid_r_lower"},
        {"object": "Eyebrow_L",        "section": "eyebrow_l"},
        {"object": "Eyebrow_R",        "section": "eyebrow_r"},
        {"object": "Ear_L",            "section": "ear_l"},
        {"object": "Ear_R",            "section": "ear_r"},
    ],

    # Welding tolerance at section boundaries (Blender units == metres).
    # 0.0005 = 0.5mm. Set to 0 to skip welding entirely.
    "merge_distance": 0.0005,

    # Remove source objects after merging (their geometry is now in target).
    # Library copies in _PartsLibrary are unaffected.
    "remove_sources": True,

    # Name of the face attribute that tags each face with its section.
    "section_attr": "_section",
}


# ------------------------------- HELPERS ------------------------------------
def _obj(name, required=True):
    o = bpy.data.objects.get(name)
    if required and o is None:
        raise RuntimeError(f"object '{name}' not found")
    return o


def _ensure_section_attr(me, attr_name):
    """Get or create the face-domain string attribute holding section tags."""
    a = me.attributes.get(attr_name)
    if a is not None:
        if a.domain != 'FACE' or a.data_type != 'STRING':
            me.attributes.remove(a)
            a = None
    if a is None:
        a = me.attributes.new(name=attr_name, type='STRING', domain='FACE')
    return a


def _tag_all_faces(me, attr_name, tag):
    a = _ensure_section_attr(me, attr_name)
    for d in a.data:
        d.value = tag


def _append_mesh(target_obj, src_obj, section_tag, attr_name):
    """Append src_obj's geometry into target_obj's mesh. Brings:
      - verts, edges, faces (in target object space)
      - UV layers (matched by name, created if missing)
      - vertex groups (matched by name, created if missing) + weights
      - materials (deduped by reference)
      - section tag written to every appended face
    Returns (n_verts_added, n_faces_added).
    """
    tgt_me = target_obj.data
    src_me = src_obj.data

    # Material remap: append-or-find materials, remember per-source-slot index.
    mat_slot_remap = []
    tgt_mats = [ms.material for ms in target_obj.material_slots]
    for ms in src_obj.material_slots:
        m = ms.material
        if m is None:
            mat_slot_remap.append(0)
            continue
        if m in tgt_mats:
            mat_slot_remap.append(tgt_mats.index(m))
        else:
            target_obj.data.materials.append(m)
            tgt_mats.append(m)
            mat_slot_remap.append(len(tgt_mats) - 1)

    # Vertex group remap: name -> target group index. Create missing.
    vg_remap = {}  # src vgroup idx -> tgt vgroup idx
    src_vg_by_idx = {vg.index: vg.name for vg in src_obj.vertex_groups}
    tgt_vg_by_name = {vg.name: vg.index for vg in target_obj.vertex_groups}
    for src_idx, name in src_vg_by_idx.items():
        if name not in tgt_vg_by_name:
            target_obj.vertex_groups.new(name=name)
            tgt_vg_by_name[name] = target_obj.vertex_groups[name].index
        vg_remap[src_idx] = tgt_vg_by_name[name]

    # Per-vert weights collected up-front (will reapply after bmesh write).
    # Each entry: list of (tgt_vg_idx, weight) for each appended vert.
    per_vert_weights = []
    for sv in src_me.vertices:
        per_vert_weights.append([
            (vg_remap[g.group], g.weight) for g in sv.groups
            if g.group in vg_remap
        ])

    # Per-face source UV (one per loop) keyed by UV layer name.
    src_uv_layers = {sl.name: [sl.data[li].uv.copy() for li in range(len(sl.data))]
                     for sl in src_me.uv_layers}

    # Source -> target object-space transform.
    src_to_tgt = target_obj.matrix_world.inverted() @ src_obj.matrix_world

    # Open bmesh on target.
    bm = bmesh.new()
    bm.from_mesh(tgt_me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    n_verts_before = len(bm.verts)
    n_faces_before = len(bm.faces)

    # Add verts (in target local space).
    src_to_new_vert = {}
    for sv in src_me.vertices:
        co = src_to_tgt @ sv.co
        nv = bm.verts.new(co)
        src_to_new_vert[sv.index] = nv
    bm.verts.ensure_lookup_table()

    # Add faces; collect (new_face, src_polygon) pairs for UV + material + tag.
    new_face_pairs = []
    for sp in src_me.polygons:
        verts = [src_to_new_vert[vi] for vi in sp.vertices]
        try:
            nf = bm.faces.new(verts)
        except ValueError:
            # Duplicate face -- bmesh raises if a face with these verts exists.
            continue
        nf.material_index = mat_slot_remap[sp.material_index] if mat_slot_remap else 0
        nf.smooth = sp.use_smooth
        new_face_pairs.append((nf, sp))

    # Ensure UV layers exist on target for every source UV layer.
    tgt_uv_layer_names = [sl.name for sl in tgt_me.uv_layers]
    for uv_name in src_uv_layers:
        if uv_name not in tgt_uv_layer_names:
            tgt_me.uv_layers.new(name=uv_name)
    # Refresh bmesh's UV layer handles (the new layer is now visible on bm).
    bm_uv_layers = {nm: bm.loops.layers.uv.get(nm) or bm.loops.layers.uv.new(nm)
                    for nm in src_uv_layers}

    # Write UVs onto the new faces.
    for nf, sp in new_face_pairs:
        for li_idx_in_face, (nloop, src_loop_idx) in enumerate(
                zip(nf.loops, sp.loop_indices)):
            for uv_name, uv_data in src_uv_layers.items():
                nloop[bm_uv_layers[uv_name]].uv = uv_data[src_loop_idx]

    bm.faces.ensure_lookup_table()
    bm.to_mesh(tgt_me)
    bm.free()
    tgt_me.update()

    # Apply per-vert weights to the newly added verts.
    new_vert_start = n_verts_before
    for src_vi, new_offset in enumerate(range(new_vert_start, new_vert_start + len(src_me.vertices))):
        for vg_idx, w in per_vert_weights[src_vi]:
            target_obj.vertex_groups[vg_idx].add([new_offset], w, 'REPLACE')

    # Tag new faces with the section attribute.
    a = _ensure_section_attr(tgt_me, attr_name)
    new_face_count = len(tgt_me.polygons) - n_faces_before
    for fi in range(n_faces_before, n_faces_before + new_face_count):
        a.data[fi].value = tag_bytes(section_tag)

    if src_me.shape_keys is not None and len(src_me.shape_keys.key_blocks) > 1:
        print(f"  WARN: '{src_obj.name}' had shape keys -- DROPPED in merge "
              f"(library copy in _PartsLibrary still has them)")

    return len(src_me.vertices), new_face_count


def tag_bytes(s):
    """Blender STRING attributes accept bytes or str depending on version.
    Normalise to plain str -- if bytes is required, Blender will coerce.
    """
    return s


def _weld_boundaries(target_obj, merge_distance, attr_name):
    """Run a single remove_doubles pass on the whole mesh. Section tags
    propagate via the surviving face's existing attribute value, so seams
    stay correctly labelled.
    """
    if merge_distance <= 0:
        return 0
    me = target_obj.data
    n_before = len(me.vertices)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)
    bm.to_mesh(me)
    bm.free()
    me.update()
    n_after = len(me.vertices)
    print(f"  weld: {n_before} -> {n_after} verts ({n_before - n_after} merged)")
    return n_before - n_after


# --------------------------------- ENTRY ------------------------------------
def merge_face_meshes(cfg):
    tgt = _obj(cfg["target"])
    print(f"=== merge_face_meshes -> {tgt.name} ===")

    # Tag the existing head faces as 'head' BEFORE appending anything.
    _tag_all_faces(tgt.data, cfg["section_attr"], cfg["target_section"])
    print(f"  tagged {len(tgt.data.polygons)} existing faces as "
          f"'{cfg['target_section']}'")

    merged = []
    for entry in cfg["sources"]:
        src = _obj(entry["object"], required=False)
        if src is None:
            print(f"  skip '{entry['object']}': not in scene")
            continue
        nv, nf = _append_mesh(tgt, src, entry["section"], cfg["section_attr"])
        print(f"  + '{src.name}' as '{entry['section']}': +{nv}v +{nf}f")
        merged.append(src)

    _weld_boundaries(tgt, cfg["merge_distance"], cfg["section_attr"])

    if cfg.get("remove_sources", True):
        for src in merged:
            me = src.data
            bpy.data.objects.remove(src, do_unlink=True)
            if isinstance(me, bpy.types.Mesh) and me.users == 0:
                bpy.data.meshes.remove(me)
        print(f"  removed {len(merged)} source objects")

    print(f"\n[done] merged {len(merged)} sources into '{tgt.name}' "
          f"({len(tgt.data.vertices)}v {len(tgt.data.polygons)}f)")
    return tgt


if __name__ == "__main__":
    merge_face_meshes(CONFIG)
