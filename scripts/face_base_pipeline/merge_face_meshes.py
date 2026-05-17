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
    # Directional check: only delete a head face if it faces the SAME hemisphere
    # as the nearest section face (dot(head_normal, section_normal) > threshold).
    # Filters out false-positives where the BACK of a 3D section (e.g. inside
    # of the lip volume) is geometrically close to a head face that should be
    # kept. dot=0.3 corresponds to ~72 degrees -- pretty permissive.
    "cut_underlying_dot_min": 0.3,

    # Layer collections to force-exclude before the merge runs. Stops donor /
    # library duplicates from sitting at the same world coords as the working
    # head + blocking the viewport / render. Names match LayerCollection.name
    # at any depth in the view layer's layer_collection tree.
    "hide_noise_collections": [
        "Skeleton (Fortnite)",
        "ARKit (MechanicGirl)",
        "Customization (Mutable)",
        "Face Parts",
        "Body Parts",
    ],

    # Seam connection method per integrating section:
    #   "weld"   -- snap section perimeter verts directly onto nearest target
    #     verts (max_snap distance). Topology-correct via bmesh.ops.weld_verts
    #     when the section's outer edge sits AT the head surface; misses
    #     when the section has 3D depth and sits OUT from the head.
    #   "bridge" -- create new bridging faces between section + target loops.
    #     Only clean when both loops are simple cycles with similar vert
    #     counts; fails on Tripo-style lips where outer+inner perimeter form
    #     one combined boundary walk.
    #   "none"   -- leave as adjacent islands; visible seam but no broken
    #     topology. Default for now -- the proper fix is knife_project
    #     (task #64) which creates 1:1 head/cutter correspondence and lets
    #     weld land cleanly.
    "boundary_method_for": {"lips": "weld"},
    "boundary_weld_max_snap_m": 0.015,

    # After merge + cut + weld, any vert that ends up with total weight == 0
    # gets implicitly bound to the armature root and pulled to world origin
    # ("dragged to the floor"). Repair by BVH closest-point + barycentric
    # weight transfer from a skeleton donor (which must be skinned to the
    # same armature). vgroups matched by name; missing groups auto-created
    # on the target. Donor's layer collection can be excluded from the view
    # layer -- only mesh data + world transform are read.
    #
    # Resolution order:
    #   1. `repair_unweighted_donor` (literal object name)  -- if set
    #   2. `repair_unweighted_donor_collection`             -- first MESH
    #      object inside this collection (any depth). Use when the BDB
    #      setup panel created the collection.
    #   3. donor_registry.donor("skeleton", "head")         -- pipeline
    #      default; one edit in donor_registry.py swaps donor per head
    #      (e.g. male vs female Fortnite head).
    # Set `repair_unweighted_donor` to "" (empty string) to disable repair.
    "repair_unweighted_donor": None,
    "repair_unweighted_donor_collection": "Skeleton (Fortnite)",

    # How to handle the TARGET's existing shape keys across the merge. The
    # bmesh write that appends new verts leaves new verts with UNINITIALISED
    # positions in every shape key; when a key fires (even at value 0.0 if
    # the runtime evaluates it), those verts get yanked to garbage coords
    # (observed: 20-METRE displacements during the rig test).
    #   "preserve" (default) -- snapshot pre-merge vert positions per key,
    #     restore them after bmesh write, set new verts to Basis (zero delta).
    #     Existing-vert morphs unaffected; new verts inert in all keys.
    #     User runs transfer_shape_keys post-merge to give new verts proper
    #     anatomical morphs.
    #   "drop"   -- strip all shape keys from target before the merge. Forces
    #     the user to re-run transfer_shape_keys post-merge; simplest and
    #     safest if you always plan to re-transfer anyway.
    "shape_keys_on_target": "preserve",

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
                          section_attr, dot_min=0.3):
    """For each section name in cut_for_sections, build a BVH from that
    section's faces (within the merged mesh) and delete any `target_section`
    face whose center is within `threshold` metres of the cutting section's
    surface AND whose normal points the same hemisphere as the nearest
    cutting face's normal (dot >= dot_min). Following the actual mesh contour
    means we don't over-delete around the section's bbox corners (where the
    bbox sweeps far past the real boundary) and don't leave a ragged-
    rectangular hole; the directional check prevents false-positives where the
    BACK of a 3D section (e.g. inside of the lip volume) is geometrically
    close to a head face that should be kept.

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
        kept_back = 0
        for f in bm.faces:
            if f[sec_layer].decode('utf-8') != target_section:
                continue
            c = f.calc_center_median()
            loc, normal, idx, dist = bvh.find_nearest(c, threshold * 2.0)
            if loc is None or dist > threshold:
                continue
            # Directional check: skip head faces that point AWAY from the
            # section face they're near (i.e. the section's back side).
            if normal is not None and f.normal.dot(normal) < dot_min:
                kept_back += 1
                continue
            victims.append(f)
        if victims:
            bmesh.ops.delete(bm, geom=victims, context='FACES')
            bm.faces.ensure_lookup_table()
            total_cut += len(victims)
        print(f"  cut_underlying_head: '{cut_section}' -> deleted {len(victims)} "
              f"'{target_section}' faces within {threshold*1000:.0f}mm "
              f"(kept {kept_back} back-facing)")
    # Drop orphan verts left behind.
    orphans = [v for v in bm.verts if not v.link_faces]
    if orphans:
        bmesh.ops.delete(bm, geom=orphans, context='VERTS')
    bm.to_mesh(me); bm.free(); me.update()
    return total_cut


def _boundary_loops_for_section(bm, sec_layer, section_name, restrict_face_set=None):
    """Return list of edge loops (each a list of bmesh edges) that form the
    closed perimeter of the named section. A boundary edge of a section is
    one with exactly ONE adjacent face in that section."""
    section_faces = ({f for f in bm.faces
                      if f[sec_layer].decode('utf-8') == section_name}
                     if restrict_face_set is None
                     else {f for f in restrict_face_set
                           if f[sec_layer].decode('utf-8') == section_name})
    if not section_faces:
        return []
    boundary_edges = set()
    for f in section_faces:
        for e in f.edges:
            n_in = sum(1 for ef in e.link_faces if ef in section_faces)
            if n_in == 1:
                boundary_edges.add(e)

    loops = []
    visited = set()
    for seed in boundary_edges:
        if seed in visited:
            continue
        loop = []
        stack = [seed]
        while stack:
            e = stack.pop()
            if e in visited:
                continue
            visited.add(e)
            loop.append(e)
            for v in e.verts:
                for ev in v.link_edges:
                    if ev in boundary_edges and ev not in visited:
                        stack.append(ev)
        loops.append(loop)
    return loops


def _bridge_seam(target_obj, target_section, sections_to_bridge, section_attr):
    """For each section in sections_to_bridge, find its OUTER boundary loop
    (the one nearest the target's matching cut boundary) and bridge it to
    the target section's nearest boundary loop with new faces. This closes
    the visible seam between an integrating section and the shell when the
    section sits OUT from the shell surface (lip mesh has depth).

    The section's INNER boundaries (lip slit, nostril interior) are left
    alone -- only the OUTER perimeter (the one closest to head boundary)
    gets bridged.

    New bridging faces are tagged with `target_section` so they shade with
    the shell material, and their `_section` value is target_section so the
    seam isn't visible as a different region.
    """
    if not sections_to_bridge:
        return 0
    from mathutils import Vector
    me = target_obj.data
    sec = me.attributes.get(section_attr)
    if sec is None:
        return 0

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    sec_layer = bm.faces.layers.string.get(section_attr)

    total_new = 0
    for sect in sections_to_bridge:
        section_loops = _boundary_loops_for_section(bm, sec_layer, sect)
        if not section_loops:
            continue
        target_loops = _boundary_loops_for_section(bm, sec_layer, target_section)
        if not target_loops:
            continue

        # For each section loop, find centroid; pair with the target loop
        # whose centroid is closest. Take the pair with smallest distance.
        def _centroid(edges):
            verts = set()
            for e in edges:
                verts.update(e.verts)
            return sum((v.co for v in verts), Vector()) / len(verts)

        section_centroids = [(loop, _centroid(loop)) for loop in section_loops]
        target_centroids = [(loop, _centroid(loop)) for loop in target_loops]

        # Find best pair: section loop closest to a target loop. We want the
        # OUTER section loop (closest to target) -- so iterate section loops,
        # for each find nearest target loop, keep best pair globally.
        best = None
        for sl, sc in section_centroids:
            for tl, tc in target_centroids:
                d = (sc - tc).length
                if best is None or d < best[0]:
                    best = (d, sl, tl)
        if best is None:
            continue
        dist, sl, tl = best
        # Skip if too far apart (probably matched wrong loops, e.g. neck opening)
        if dist > 0.05:
            print(f"  bridge_seam: skip '{sect}' (nearest target loop {dist*100:.1f}cm "
                  f"away -- probably no matching cut)")
            continue

        try:
            result = bmesh.ops.bridge_loops(bm, edges=sl + tl)
            new_faces = result.get("faces", [])
            new_edges = result.get("edges", [])
        except Exception as e:
            print(f"  bridge_seam: bridge_loops failed for '{sect}': {e}")
            continue

        # Tag the new faces with target_section so they shade with the head
        for nf in new_faces:
            nf[sec_layer] = target_section.encode('utf-8')
            nf.smooth = True  # smooth shading at seam transition

        total_new += len(new_faces)
        print(f"  bridge_seam: '{sect}' loop ({len(sl)} edges) <-> "
              f"'{target_section}' loop ({len(tl)} edges) at {dist*1000:.0f}mm: "
              f"+{len(new_faces)} bridging faces")

    bm.to_mesh(me); bm.free(); me.update()
    return total_new


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


def _resolve_donor(explicit_name, collection_name):
    """Resolve the skeleton donor mesh by priority:
      1. explicit object name (`""` empty string = disabled, returns None)
      2. first MESH inside the named collection (any depth)
      3. donor_registry.donor("skeleton", "head")
    Returns the bpy.types.Object or None.
    """
    if explicit_name == "":
        return None  # explicitly disabled
    if explicit_name:
        o = bpy.data.objects.get(explicit_name)
        if o is not None and o.type == 'MESH':
            return o
        print(f"  resolve_donor: explicit name '{explicit_name}' not a mesh -- "
              f"trying collection fallback")
    if collection_name:
        coll = bpy.data.collections.get(collection_name)
        if coll is not None:
            for o in coll.all_objects:
                if o.type == 'MESH':
                    return o
            print(f"  resolve_donor: collection '{collection_name}' has no mesh -- "
                  f"trying donor_registry")
    # Final fallback: donor_registry
    try:
        import donor_registry
        name = donor_registry.donor("skeleton", "head")
        o = bpy.data.objects.get(name)
        if o is not None and o.type == 'MESH':
            return o
        print(f"  resolve_donor: donor_registry name '{name}' not in scene")
    except Exception as e:
        print(f"  resolve_donor: donor_registry lookup failed: {e}")
    return None


def _repair_unweighted_from_donor(target_obj, donor_obj):
    """For every vert on `target_obj` whose total skinning weight is 0, find
    the closest face on `donor_obj`, barycentric-interpolate the donor face's
    vert weights, and assign the result to the target vert (normalised to
    sum=1). Bone names must match; missing vgroups are auto-created on the
    target. Returns count repaired.

    Without this, unweighted verts implicitly bind to the armature root and
    get dragged to world origin during posing. Common cause: source mesh
    cleanup left a few face-shell verts without weights, and the merge
    preserves that gap (it's not a merge regression -- the verts came in
    unweighted, but it shows up after the merge because the head is now
    being driven by the armature for the first time at scale).
    """
    if donor_obj is None:
        print(f"  repair_unweighted: no donor -- skip")
        return 0
    donor = donor_obj

    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    from mathutils.geometry import barycentric_transform

    me = target_obj.data
    # Find unweighted verts
    unweighted = [v.index for v in me.vertices
                  if sum(g.weight for g in v.groups) < 1e-6]
    if not unweighted:
        print(f"  repair_unweighted: all {len(me.vertices)} verts already weighted")
        return 0

    dm = donor.data
    dw = donor.matrix_world
    hw = target_obj.matrix_world

    verts_w = [dw @ v.co for v in dm.vertices]
    polys = [list(p.vertices) for p in dm.polygons]
    bvh = BVHTree.FromPolygons(verts_w, polys, all_triangles=False, epsilon=0.0)

    # vgroup name match; create missing groups on target
    tgt_vg_by_name = {vg.name: vg.index for vg in target_obj.vertex_groups}
    for dvg in donor.vertex_groups:
        if dvg.name not in tgt_vg_by_name:
            target_obj.vertex_groups.new(name=dvg.name)
            tgt_vg_by_name[dvg.name] = target_obj.vertex_groups[dvg.name].index
    donor_vg_to_tgt = {dvg.index: tgt_vg_by_name[dvg.name]
                       for dvg in donor.vertex_groups}

    # donor per-vert weight dict
    donor_weights = []
    for dv in dm.vertices:
        donor_weights.append({
            donor_vg_to_tgt[g.group]: g.weight
            for g in dv.groups
            if g.group in donor_vg_to_tgt and g.weight > 0
        })

    tgt_vgs = list(target_obj.vertex_groups)
    repaired = 0
    for vi in unweighted:
        co_w = hw @ me.vertices[vi].co
        loc, normal, fi, dist = bvh.find_nearest(co_w)
        if loc is None or fi is None:
            continue
        poly = polys[fi]
        if len(poly) == 3:
            a, b, c = poly
            try:
                bc = barycentric_transform(
                    loc, verts_w[a], verts_w[b], verts_w[c],
                    Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
                face_idxs = [a, b, c]
                face_ws = [bc.x, bc.y, bc.z]
            except ValueError:
                face_idxs = [a, b, c]; face_ws = [1/3.0]*3
        else:
            face_idxs = poly
            face_ws = [1.0/len(poly)] * len(poly)
        accum = {}
        for didx, w in zip(face_idxs, face_ws):
            for vg_idx, weight in donor_weights[didx].items():
                accum[vg_idx] = accum.get(vg_idx, 0.0) + w * weight
        if not accum:
            continue
        total = sum(accum.values())
        if total <= 0:
            continue
        for vg_idx, w in accum.items():
            tgt_vgs[vg_idx].add([vi], w / total, 'REPLACE')
        repaired += 1
    print(f"  repair_unweighted: {repaired}/{len(unweighted)} verts re-bound from "
          f"'{donor.name}'")
    return repaired


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
def _hide_noise_collections(noise_names):
    """Exclude donor / library / reference layer collections so the merge
    isn't visually swamped by duplicate parts sitting at the same world
    coords as the working head. Idempotent. Walks the view layer's layer
    collection tree (recursive).
    """
    if not noise_names:
        return
    targets = set(noise_names)
    hidden = []
    def walk(lc):
        if lc.name in targets and not lc.exclude:
            lc.exclude = True
            hidden.append(lc.name)
        for c in lc.children:
            walk(c)
    walk(bpy.context.view_layer.layer_collection)
    if hidden:
        print(f"  hidden noise collections: {hidden}")


def _strip_shape_keys_safely(target_obj, mode):
    """Handle target's pre-existing shape keys before bmesh writes append new
    verts. Without this, the new verts get uninitialised positions in every
    existing key -- when the rig animates a shape key, those verts get yanked
    to garbage world coords (observed: 20-METRE displacements at value=0.0).

    Modes:
      "drop"  -- remove all shape keys. User re-runs transfer_shape_keys after
                 the merge for proper anatomical morphs on new verts.
      "zero_new_verts" (default after bmesh runs) -- not callable here; done
                 post-merge by _zero_shape_keys_for_new_verts using the
                 snapshot returned by this function.

    Returns: dict with snapshot for restore (or empty if dropped).
    """
    me = target_obj.data
    sk = me.shape_keys
    if sk is None:
        return {}
    n_keys = len(sk.key_blocks)
    if mode == "drop":
        # Remove all keys; iterate by name since the list mutates.
        names = [kb.name for kb in sk.key_blocks]
        for name in names:
            kb = target_obj.data.shape_keys.key_blocks.get(name) if target_obj.data.shape_keys else None
            if kb:
                target_obj.shape_key_remove(kb)
        print(f"  shape_keys: dropped {n_keys} keys from '{target_obj.name}' "
              f"-- re-run transfer_shape_keys after merge for new-vert morphs")
        return {}
    # "preserve" -- snapshot pre-merge vert positions per key
    n_verts = len(me.vertices)
    snap = {kb.name: [kb.data[vi].co.copy() for vi in range(n_verts)]
            for kb in sk.key_blocks}
    print(f"  shape_keys: snapshot {n_keys} keys x {n_verts} verts "
          f"(new verts will be zero-delta after merge)")
    return snap


def _restore_shape_keys_with_basis_for_new(target_obj, snapshot, n_verts_before):
    """After bmesh write, restore each shape key's positions:
      - existing verts (idx < n_verts_before): use snapshotted positions
      - new verts (idx >= n_verts_before):     copy Basis position (zero delta)
    Together this means: shape keys still animate the original verts as
    before, and new verts (lips/eyelids/brows/ears added by the merge) stay
    put -- no orbital pulling.

    For new verts to actually be morphed by the keys, run transfer_shape_keys
    post-merge.
    """
    if not snapshot:
        return
    me = target_obj.data
    sk = me.shape_keys
    if sk is None:
        return
    basis = sk.reference_key
    n_now = len(me.vertices)
    restored = 0
    new_zeroed = 0
    for kb in sk.key_blocks:
        snap = snapshot.get(kb.name)
        if snap is None:
            # New key added since snapshot? Skip (treat as already valid).
            continue
        # Restore existing verts in-place; this overrides any bmesh garbage.
        for vi in range(min(n_verts_before, n_now, len(snap))):
            kb.data[vi].co = snap[vi]
            restored += 1
        # Zero new verts (basis position = zero delta).
        for vi in range(n_verts_before, n_now):
            kb.data[vi].co = basis.data[vi].co.copy()
            new_zeroed += 1
    print(f"  shape_keys: restored {restored} existing-vert positions, "
          f"zeroed {new_zeroed} new-vert positions")


def merge_face_meshes(cfg):
    _hide_noise_collections(cfg.get("hide_noise_collections", []))
    tgt = _obj(cfg["target"])
    print(f"=== merge_face_meshes -> {tgt.name} ===")

    # Snapshot or drop pre-existing shape keys before bmesh appends garbage data
    # to them. See _strip_shape_keys_safely for the gnarly story.
    sk_mode = cfg.get("shape_keys_on_target", "preserve")  # "preserve" | "drop"
    sk_snapshot = _strip_shape_keys_safely(tgt, sk_mode)
    n_verts_before_merge = len(tgt.data.vertices)

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
                          cfg["section_attr"],
                          dot_min=cfg.get("cut_underlying_dot_min", 0.3))
    # Per-section seam connection: bridge / weld / none
    method_map = cfg.get("boundary_method_for", {})
    bridge_list = [s for s, m in method_map.items() if m == "bridge"]
    weld_list   = [s for s, m in method_map.items() if m == "weld"]
    if bridge_list:
        _bridge_seam(tgt, cfg["target_section"], bridge_list, cfg["section_attr"])
    if weld_list:
        _boundary_weld(tgt, cfg["target_section"], weld_list,
                        cfg.get("boundary_weld_max_snap_m", 0.010),
                        cfg["section_attr"])
    donor = _resolve_donor(cfg.get("repair_unweighted_donor"),
                            cfg.get("repair_unweighted_donor_collection"))
    if donor is not None:
        print(f"  repair_unweighted: donor resolved to '{donor.name}'")
    _repair_unweighted_from_donor(tgt, donor)
    _restore_shape_keys_with_basis_for_new(tgt, sk_snapshot, n_verts_before_merge)
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
