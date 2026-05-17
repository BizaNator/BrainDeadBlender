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
    # Optional per-source flag `weld_to_target` (default True): when False,
    # this source's boundary verts are NOT welded into the target -- use for
    # parts that sit ON TOP of the head (lashes, brows, hair pieces) where a
    # crisp silhouette matters more than continuous deformation. Parts that
    # need continuous skin flow with the head (lips) keep weld_to_target=True.
    "sources": [
        {"object": "CustomLips",       "section": "lips",            "weld_to_target": True},
        {"object": "Eyelid_L_Upper",   "section": "eyelid_l_upper",  "weld_to_target": False},
        {"object": "Eyelid_L_Lower",   "section": "eyelid_l_lower",  "weld_to_target": False},
        {"object": "Eyelid_R_Upper",   "section": "eyelid_r_upper",  "weld_to_target": False},
        {"object": "Eyelid_R_Lower",   "section": "eyelid_r_lower",  "weld_to_target": False},
        {"object": "Eyebrow_L",        "section": "eyebrow_l",       "weld_to_target": False},
        {"object": "Eyebrow_R",        "section": "eyebrow_r",       "weld_to_target": False},
        {"object": "Ear_L",            "section": "ear_l",           "weld_to_target": True},
        {"object": "Ear_R",            "section": "ear_r",           "weld_to_target": True},
    ],

    # Welding tolerance at section boundaries (Blender units == metres).
    # 0.0005 = 0.5mm. Only applied to sources with weld_to_target=True;
    # non-welded sources stay as physically separate vert islands inside the
    # merged mesh (clean section silhouette, no normal blending across seam).
    "merge_distance": 0.0005,

    # After merge, mark edges sharp where adjacent faces meet at >N degrees.
    # Gives the iconic low-poly hard-faceted look uniformly across the whole
    # merged head, including parts that came in fully smooth (brows, lashes,
    # ears from Tripo). Set to None to skip. 30 is a good default for stylised
    # characters; 45 for softer creases; 60 for nearly-smooth.
    "sharpen_by_angle_deg": 30.0,

    # For each listed section, delete any "target_section" face whose center
    # falls inside that section's bbox (shrunk by `cut_underlying_shrink_m`).
    # Use for sections that integrate INTO the head shell where the head
    # underneath would create double geometry / Z-fighting (lips through
    # mouth opening). Lashes / brows / ears sit ON TOP of the head with no
    # overlap so they don't need this.
    "cut_underlying_head_for": ["lips"],
    # Proximity threshold in metres: head faces within this distance of the
    # cutting section's surface get deleted. 4mm is conservative -- raise to
    # 8mm if too much shell shows through, lower if cheeks lose faces.
    "cut_underlying_threshold_m": 0.004,

    # After the cut, weld each listed section's OUTER boundary verts onto the
    # nearest target_section vert (within snap distance). Produces continuous
    # topology at the seam instead of two adjacent islands sharing geometry by
    # proximity only -- the lip outer edge becomes the same edge as the head's
    # mouth-opening boundary. The section's INNER boundary verts (lip slit,
    # nostril interior, etc.) stay free because no nearby target verts exist.
    # Use the same list as cut_underlying_head_for in most cases.
    "boundary_weld_for": ["lips"],
    "boundary_weld_max_snap_m": 0.010,  # 10mm tolerance for matching pairs

    # Copy custom split normals (mesh.loops[].normal) from sources so any
    # artist-authored normal data survives the merge. UE imports these as the
    # baked normals when "Compute Normals" is off on FBX import.
    "copy_split_normals": True,

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
    v = tag_bytes(tag)
    for d in a.data:
        d.value = v


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

    # Per-edge attributes worth preserving: sharp / seam / subdivision crease.
    # (bevel_weight moved to a generic attribute in Blender 5.x; skip it.)
    # Keyed by sorted (vi_a, vi_b) of source verts so we can look up the
    # matching new edge after appending into the bmesh.
    src_edge_attrs = {}
    for e in src_me.edges:
        crease = getattr(e, "crease", 0)
        if not (e.use_edge_sharp or e.use_seam or crease):
            continue
        k = tuple(sorted(e.vertices))
        src_edge_attrs[k] = {
            "sharp":  e.use_edge_sharp,
            "seam":   e.use_seam,
            "crease": crease,
        }

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

    # Edge attribute pass: for each src edge that had sharp/seam/bevel/crease,
    # find the matching bmesh edge between the corresponding new verts and
    # copy the attribute over. Run before to_mesh so the values land on the
    # output mesh.
    bm.edges.ensure_lookup_table()
    sharp_count = 0
    for (vi_a, vi_b), attrs in src_edge_attrs.items():
        v_a = src_to_new_vert.get(vi_a)
        v_b = src_to_new_vert.get(vi_b)
        if v_a is None or v_b is None:
            continue
        be = bm.edges.get([v_a, v_b])
        if be is None:
            continue
        if attrs["sharp"]:
            be.smooth = False  # bmesh stores "smooth"; False == sharp
            sharp_count += 1
        if attrs["seam"]:
            be.seam = True
        if attrs["crease"]:
            cr_layer = bm.edges.layers.crease.verify()
            be[cr_layer] = attrs["crease"]
    if sharp_count:
        print(f"    preserved {sharp_count} sharp edges from {src_obj.name}")

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
    """Blender 5.x STRING attributes require bytes; older versions accepted
    str. Always encode -- a bytes input passes through, a str gets utf-8'd.
    """
    if isinstance(s, bytes):
        return s
    return s.encode('utf-8')


def _weld_boundaries(target_obj, merge_distance, weld_vert_idxs):
    """Run remove_doubles ONLY on the verts whose section opted in to welding
    (weld_to_target=True). Verts from non-welded sources stay as physically
    separate islands so their silhouette isn't blurred. Head verts always
    participate so a welding source can fuse INTO the head.
    """
    if merge_distance <= 0 or not weld_vert_idxs:
        return 0
    me = target_obj.data
    n_before = len(me.vertices)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in weld_vert_idxs if i < len(bm.verts)]
    bmesh.ops.remove_doubles(bm, verts=verts, dist=merge_distance)
    bm.to_mesh(me)
    bm.free()
    me.update()
    n_after = len(me.vertices)
    print(f"  weld: {n_before} -> {n_after} verts ({n_before - n_after} merged)")
    return n_before - n_after


def _cut_underlying_head(target_obj, target_section, cut_for_sections, threshold,
                          section_attr):
    """For each section name in cut_for_sections, build a BVH from that
    section's faces (within the merged mesh) and delete any `target_section`
    face whose center is within `threshold` metres of the cutting section's
    surface. Following the actual mesh contour means we don't over-delete
    around the section's bbox corners (where the bbox sweeps far past the
    real boundary) and don't leave a ragged-rectangular hole.

    Only for sections that integrate INTO the head shell (lips). Sections
    that sit ON TOP of the head (lashes/brows/ears) shouldn't trigger this.
    """
    if not cut_for_sections:
        return 0
    from mathutils.bvhtree import BVHTree
    me = target_obj.data
    sec = me.attributes.get(section_attr)
    if sec is None:
        return 0
    face_sec = [sec.data[fi].value.decode('utf-8') for fi in range(len(me.polygons))]

    total_cut = 0
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    sec_layer = bm.faces.layers.string.get(section_attr)

    for cut_section in cut_for_sections:
        # Collect verts/faces of the cutting section in bmesh coords.
        cut_faces = [f for f in bm.faces if f[sec_layer].decode('utf-8') == cut_section]
        if not cut_faces:
            print(f"  cut_underlying_head: skip '{cut_section}' (no faces)")
            continue
        # Build a vert-index remap + flat lists for BVHTree.FromPolygons
        idx_remap = {}
        verts = []
        polys = []
        for f in cut_faces:
            poly = []
            for v in f.verts:
                if v.index not in idx_remap:
                    idx_remap[v.index] = len(verts)
                    verts.append(v.co.copy())
                poly.append(idx_remap[v.index])
            polys.append(poly)
        bvh = BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0)

        victims = []
        for f in bm.faces:
            if f[sec_layer].decode('utf-8') != target_section:
                continue
            c = f.calc_center_median()
            loc, normal, idx, dist = bvh.find_nearest(c, threshold * 2.0)
            if loc is not None and dist <= threshold:
                victims.append(f)
        if victims:
            bmesh.ops.delete(bm, geom=victims, context='FACES')
            bm.faces.ensure_lookup_table()
            total_cut += len(victims)
            print(f"  cut_underlying_head: deleted {len(victims)} '{target_section}' "
                  f"faces within {threshold*1000:.0f}mm of '{cut_section}' surface")
    # Drop orphan verts left behind.
    orphans = [v for v in bm.verts if not v.link_faces]
    if orphans:
        bmesh.ops.delete(bm, geom=orphans, context='VERTS')
    bm.to_mesh(me); bm.free(); me.update()
    return total_cut


def _boundary_weld(target_obj, target_section, weld_sections, max_snap, section_attr):
    """For each section in weld_sections, find its boundary verts (verts on
    edges with only one face in the section) and snap each to the nearest
    target_section vert within `max_snap` metres. Then a tiny remove_doubles
    pass fuses the snapped pairs into shared verts.

    Result: the section's outer perimeter becomes the SAME edge as the
    target shell's existing boundary -- continuous topology, no double
    geometry. Inner boundary verts (mouth slit, eye iris ring) stay free
    because there's no nearby target vert (the hole was cut). Works for
    any section that integrates into the shell: lips, future nose pieces,
    eye-socket extensions, etc.
    """
    if not weld_sections:
        return 0
    from mathutils.kdtree import KDTree
    me = target_obj.data
    sec = me.attributes.get(section_attr)
    if sec is None:
        return 0

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    sec_layer = bm.faces.layers.string.get(section_attr)

    # Build a KDTree of target_section verts (where the welds will land).
    target_verts = [v for v in bm.verts
                    if any(f[sec_layer].decode('utf-8') == target_section
                           for f in v.link_faces)]
    if not target_verts:
        bm.free()
        return 0
    kd = KDTree(len(target_verts))
    for i, v in enumerate(target_verts):
        kd.insert(v.co, i)
    kd.balance()

    total_snapped = 0
    # Build a single weld map: bv (boundary vert) -> tv (target vert).
    # bmesh.ops.weld_verts replaces every reference to bv with tv in linked
    # faces/edges, then deletes bv -- proper topology merge, no non-manifold.
    target_set = set(target_verts)
    weld_map = {}
    for weld_section in weld_sections:
        section_faces = [f for f in bm.faces
                         if f[sec_layer].decode('utf-8') == weld_section]
        if not section_faces:
            continue
        section_face_set = set(section_faces)
        boundary_verts = set()
        for f in section_faces:
            for e in f.edges:
                n_in = sum(1 for ef in e.link_faces if ef in section_face_set)
                if n_in == 1:
                    boundary_verts.add(e.verts[0])
                    boundary_verts.add(e.verts[1])
        boundary_verts = [bv for bv in boundary_verts if bv not in target_set]
        if not boundary_verts:
            continue
        snapped = 0
        for bv in boundary_verts:
            co, idx, dist = kd.find(bv.co)
            if dist <= max_snap:
                tv = target_verts[idx]
                if tv is bv:
                    continue
                weld_map[bv] = tv
                snapped += 1
        total_snapped += snapped
        print(f"  boundary_weld: queued {snapped}/{len(boundary_verts)} "
              f"'{weld_section}' outer-boundary verts -> '{target_section}'")

    if weld_map:
        bmesh.ops.weld_verts(bm, targetmap=weld_map)
    bm.to_mesh(me); bm.free(); me.update()
    return total_snapped


def _sharpen_by_angle(target_obj, angle_deg):
    """Mark every edge where the angle between adjacent face normals exceeds
    angle_deg as sharp. Run AFTER weld so seams that should be one edge are
    welded first. Skips boundary edges (only-one-face) and existing sharp
    edges. Gives the low-poly hard-faceted look uniformly across the mesh.
    """
    if angle_deg is None or angle_deg <= 0:
        return 0
    import math
    me = target_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    threshold_rad = math.radians(angle_deg)
    marked = 0
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        if not e.smooth:
            continue  # already sharp
        try:
            ang = e.calc_face_angle()
        except ValueError:
            continue
        if ang > threshold_rad:
            e.smooth = False
            marked += 1
    bm.to_mesh(me)
    bm.free()
    me.update()
    print(f"  sharpen-by-angle ({angle_deg}deg): +{marked} sharp edges")
    return marked


# --------------------------------- ENTRY ------------------------------------
def merge_face_meshes(cfg):
    tgt = _obj(cfg["target"])
    print(f"=== merge_face_meshes -> {tgt.name} ===")

    # Tag the existing head faces as 'head' BEFORE appending anything.
    _tag_all_faces(tgt.data, cfg["section_attr"], cfg["target_section"])
    print(f"  tagged {len(tgt.data.polygons)} existing faces as "
          f"'{cfg['target_section']}'")

    # Verts eligible for the post-merge weld pass: head's own verts (always,
    # so welding sources can fuse INTO them) + each welding source's verts.
    weld_vert_idxs = set(range(len(tgt.data.vertices)))

    merged = []
    for entry in cfg["sources"]:
        src = _obj(entry["object"], required=False)
        if src is None:
            print(f"  skip '{entry['object']}': not in scene")
            continue
        verts_before = len(tgt.data.vertices)
        nv, nf = _append_mesh(tgt, src, entry["section"], cfg["section_attr"])
        weld_flag = entry.get("weld_to_target", True)
        if weld_flag:
            weld_vert_idxs.update(range(verts_before, verts_before + nv))
        print(f"  + '{src.name}' as '{entry['section']}': +{nv}v +{nf}f "
              f"weld={'yes' if weld_flag else 'NO (silhouette preserved)'}")
        merged.append(src)

    _weld_boundaries(tgt, cfg["merge_distance"], weld_vert_idxs)
    _cut_underlying_head(tgt, cfg["target_section"],
                          cfg.get("cut_underlying_head_for", []),
                          cfg.get("cut_underlying_threshold_m", 0.004),
                          cfg["section_attr"])
    _boundary_weld(tgt, cfg["target_section"],
                    cfg.get("boundary_weld_for", []),
                    cfg.get("boundary_weld_max_snap_m", 0.010),
                    cfg["section_attr"])
    _sharpen_by_angle(tgt, cfg.get("sharpen_by_angle_deg"))

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
