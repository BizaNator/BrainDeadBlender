"""
extract_mouth_parts.py

Pre-merge prep: split a Tripo-style CustomLips mesh into separate objects
by vertex-color tag so each region gets its own material + binding without
warping each other through ARKit lip morphs.

Why
---
Tripo bakes vertex colors into the CustomLips mesh that tag each region:
    RED   (1,0,0)  -- outer lip flesh
    GREEN (0,1,0)  -- teeth
    BLACK (0,0,0)  -- back of lip volume / inside-of-cheek cavity
    BLUE  (0,0,1)  -- mouth interior / tongue cavity

If the entire CustomLips merges into the head shell as one "lips" section,
ARKit shapes like mouthSmile / jawOpen warp the teeth too (because they
share verts with the lip flesh). Same architectural reason Fortnite keeps
teeth/tongue separate from the head.

This script runs BEFORE merge_face_meshes:
  1. Extracts GREEN faces into Teeth_Upper / Teeth_Lower (split by Z median)
  2. (Optionally) extracts BLACK + BLUE faces into MouthInterior
  3. Reduces CustomLips to RED faces only (lip flesh)
  4. Preserves vertex colors on each output for in-engine masking

After this runs, merge_face_meshes merges the (now-clean) CustomLips into
the head as 'lips' section; teeth + interior stay separate.

Drop into the BrainDeadBlender add-on alongside split_face_parts.
"""

import bpy
import bmesh
from mathutils import Vector


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    # Source CustomLips object (Tripo-style with Col vertex color tags).
    "source": "CustomLips",
    # Vertex color attribute name to read tags from.
    "color_attr": "Col",

    # Per-output: which color matches it, target object name, and per-vert
    # color OVERRIDE to bake on the output (None = preserve original tag).
    # `split_by_z` means the matched faces are bisected by median Z into
    # Upper / Lower variants (used for teeth).
    #
    # Tripo bakes vertex colors only on the FRONT-facing surface of each
    # region. The full part (e.g. a tooth) has back + side faces that share
    # the same volume but aren't painted. To capture them, grow the seed set
    # via connected-flood-fill with an orientation filter:
    #
    #   `max_horizontal_z`  -- max |face.normal.z|. Teeth are VERTICAL, so
    #     their normals lie in the XY plane (|z| ~ 0). 0.5 = 60deg cone
    #     around vertical (permissive). 0.3 = stricter. None disables the
    #     filter (pure connected flood-fill bounded only by max_hops).
    #   `max_hops`          -- cap BFS depth so growth can't traverse the
    #     entire mesh. None = unbounded (rely solely on orientation filter).
    #     Use 2-4 to keep the grow local to the seed region.
    #
    # For teeth: vertical filter PLUS hop cap = back/side faces caught,
    # cavity floor/ceiling (horizontal) rejected.
    # For MouthInterior: orientation varies; use no filter + small hops to
    # avoid bleeding into teeth.
    "outputs": [
        {
            "name":      "Teeth",
            "match":     "green",          # (0, 1, 0)
            "split_by_z": True,            # produces Teeth_Upper + Teeth_Lower
            "material":  "M_Teeth",
            "tint_rgba": (1.0, 1.0, 1.0, 1.0),  # white
            "max_horizontal_z": 0.5,       # vertical-ish only
            "max_hops":          3,
            # Uniform-scale this output to match the bbox of a reference
            # object (preserves proportions). Use Fortnite teeth as the
            # canonical "proper mouth scale" reference so Tripo's thin
            # facade gets enlarged to a workable editing scale.
            "scale_to_match": {
                "Upper": "Fortnite_Teeth_Upper",
                "Lower": "Fortnite_Teeth_Lower",
            },
        },
        {
            "name":      "MouthInterior",
            "match":     ["black", "blue"],  # both interior tags
            "split_by_z": True,            # produces MouthInterior_Upper + Lower
            "material":  "M_MouthInterior",
            "tint_rgba": (0.05, 0.02, 0.02, 1.0),  # very dark red/brown
            "max_horizontal_z": None,      # any orientation
            "max_hops":          1,
            "scale_to_match": {
                "Upper": "Fortnite_Teeth_Upper",
                "Lower": "Fortnite_Teeth_Lower",
            },
        },
    ],

    # Faces left after extraction become the new CustomLips body. Tint them
    # with the lip flesh color so the merged-head's 'lips' section has a
    # consistent RGB baseline (the original RED tag is already correct;
    # this override is just to normalise stray off-color verts).
    "lips_keep_match": "red",
    "lips_material":   "M_Lips",
    "lips_tint_rgba":  (0.8, 0.25, 0.30, 1.0),  # pink/red lip flesh
    # Optional uniform scale-to-match for the reduced CustomLips, in case the
    # Tripo lip mesh is undersized relative to the head it'll merge into.
    # None = no scaling.
    "lips_scale_to_match": None,

    # When True, write per-CORNER colors on outputs so downstream masking
    # has them at loop precision (matches the source domain). False -> POINT.
    "preserve_corner_colors": True,

    # Where the new objects go.
    "target_collection": None,  # None = same as source's first collection

    # Skeleton armature + binding bones (override per-entry below as needed).
    "armature":     "Fortnite_Armature",
    "teeth_upper_bone": "head",
    "teeth_lower_bone": "C_jaw",
    "interior_bone":    "head",
}


# ------------------------------- COLOR MATCHING -----------------------------
COLOR_TAGS = {
    "red":   lambda r, g, b: r > 0.5 and g < 0.3 and b < 0.3,
    "green": lambda r, g, b: g > 0.5 and r < 0.3 and b < 0.3,
    "blue":  lambda r, g, b: b > 0.5 and r < 0.3 and g < 0.3,
    "black": lambda r, g, b: r < 0.2 and g < 0.2 and b < 0.2,
    "white": lambda r, g, b: r > 0.7 and g > 0.7 and b > 0.7,
}


def _face_dominant_tag(p, col_attr):
    """Return the COLOR_TAGS key most often matched by the face's loops."""
    counts = {}
    for li in p.loop_indices:
        c = col_attr.data[li].color
        for tag, test in COLOR_TAGS.items():
            if test(c[0], c[1], c[2]):
                counts[tag] = counts.get(tag, 0) + 1
                break
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ------------------------------- HELPERS ------------------------------------
def _grow_connected_oriented(seed_face_idxs, src_me, max_horizontal_z=None,
                              max_hops=None):
    """BFS from `seed_face_idxs` via shared edges; include neighbour faces
    only if they pass the orientation filter (e.g. for teeth: faces must be
    "vertical" -- |normal.z| <= max_horizontal_z). Optionally cap the walk
    distance.

    Use case (teeth): painted seeds catch the front-facing teeth surface.
    The teeth back + sides are unpainted but are still VERTICAL (normal lies
    in the XY plane). Walking connected vertical faces captures them while
    NOT bleeding into the lip top/bottom or cavity floor/ceiling (which are
    horizontal).

    max_horizontal_z = None disables the orientation filter (then this is
    pure connected-flood-fill, bounded only by max_hops).
    """
    if not seed_face_idxs:
        return set()

    edge_to_faces = {}
    for p in src_me.polygons:
        for ek in p.edge_keys:
            edge_to_faces.setdefault(ek, []).append(p.index)

    def passes_orientation(fi):
        if max_horizontal_z is None:
            return True
        return abs(src_me.polygons[fi].normal.z) <= max_horizontal_z

    included = set(seed_face_idxs)
    frontier = set(seed_face_idxs)
    hops = 0
    while frontier:
        if max_hops is not None and hops >= max_hops:
            break
        hops += 1
        next_frontier = set()
        for fi in frontier:
            for ek in src_me.polygons[fi].edge_keys:
                for nfi in edge_to_faces.get(ek, ()):
                    if nfi in included:
                        continue
                    if not passes_orientation(nfi):
                        continue
                    included.add(nfi)
                    next_frontier.add(nfi)
        frontier = next_frontier
    return included


def _obj(name):
    o = bpy.data.objects.get(name)
    if o is None:
        raise RuntimeError(f"object '{name}' not found")
    return o


def _world_bbox_extent(obj):
    """Return Vector(W, D, H) of obj's world-space bbox."""
    mw = obj.matrix_world
    coords = [mw @ v.co for v in obj.data.vertices]
    if not coords:
        return Vector((0, 0, 0))
    xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
    return Vector((max(xs) - min(xs),
                   max(ys) - min(ys),
                   max(zs) - min(zs)))


def _world_bbox_center(obj):
    mw = obj.matrix_world
    coords = [mw @ v.co for v in obj.data.vertices]
    if not coords:
        return Vector((0, 0, 0))
    xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
    return Vector(((max(xs)+min(xs))/2,
                   (max(ys)+min(ys))/2,
                   (max(zs)+min(zs))/2))


def _uniform_scale_to_match(obj, reference_name):
    """Uniform-scale obj so its largest bbox dimension matches the reference
    object's largest bbox dimension. Preserves proportions (no squash) and
    keeps obj's world-space center where it was. Skips silently if either
    object is missing.
    """
    ref = bpy.data.objects.get(reference_name)
    if ref is None or ref.type != 'MESH':
        print(f"    scale_to_match: reference '{reference_name}' missing -- skip")
        return
    src_ext = _world_bbox_extent(obj)
    ref_ext = _world_bbox_extent(ref)
    src_max = max(src_ext)
    ref_max = max(ref_ext)
    if src_max <= 1e-6 or ref_max <= 1e-6:
        print(f"    scale_to_match: zero-extent bbox -- skip")
        return
    ratio = ref_max / src_max
    # Apply uniform scale around current world center: temporarily move origin
    # to bbox center, scale obj.scale, restore.
    pre_center = _world_bbox_center(obj)
    # Scale data verts in-place (around the local centroid in object space)
    local_center = obj.matrix_world.inverted() @ pre_center
    me = obj.data
    for v in me.vertices:
        v.co = local_center + (v.co - local_center) * ratio
    me.update()
    print(f"    scale_to_match: {obj.name} x{ratio:.3f} "
          f"(was max={src_max*1000:.1f}mm, ref={reference_name} max={ref_max*1000:.1f}mm)")


def _ensure_material(name, rgba):
    """Return a material with this name; create + set base color if missing."""
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name=name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
    return m


def _build_submesh_object(src, face_idxs, new_name, target_collection,
                          color_attr_name, override_rgba, preserve_corner,
                          src_color_attr):
    """Create a new Object with the given subset of src's faces. Carries
    materials, UVs, vertex groups, weights, custom split normals from src.
    Writes the override colour to the new mesh's color attribute (if any)."""
    if not face_idxs:
        return None
    src_me = src.data

    keep_verts = set()
    for fi in face_idxs:
        for vi in src_me.polygons[fi].vertices:
            keep_verts.add(vi)
    if not keep_verts:
        return None

    # vert index remap
    src_to_new = {}
    new_verts = []
    for vi in sorted(keep_verts):
        src_to_new[vi] = len(new_verts)
        new_verts.append(src_me.vertices[vi].co.copy())

    new_faces = [tuple(src_to_new[vi] for vi in src_me.polygons[fi].vertices)
                 for fi in face_idxs]
    nm = bpy.data.meshes.new(new_name + "_mesh")
    nm.from_pydata(new_verts, [], new_faces)
    nm.update()

    # Materials -- copy ALL slots from src so material_index references survive,
    # then we'll override material_index per face below.
    for ms in src.material_slots:
        nm.materials.append(ms.material)
    for ni, fi in enumerate(face_idxs):
        nm.polygons[ni].material_index = src_me.polygons[fi].material_index

    # UV layers (one per source layer)
    for sl in src_me.uv_layers:
        nl = nm.uv_layers.new(name=sl.name)
        for ni, fi in enumerate(face_idxs):
            sp = src_me.polygons[fi]
            np = nm.polygons[ni]
            for li_new, li_src in zip(np.loop_indices, sp.loop_indices):
                nl.data[li_new].uv = sl.data[li_src].uv

    # Object + collection link
    no = bpy.data.objects.new(new_name, nm)
    no.matrix_world = src.matrix_world.copy()
    tcoll = target_collection
    if tcoll is None:
        tcoll = (src.users_collection[0] if src.users_collection
                 else bpy.context.scene.collection)
    tcoll.objects.link(no)

    # Vertex groups: copy definitions + per-vert weights
    for vg in src.vertex_groups:
        no.vertex_groups.new(name=vg.name)
    name_to_idx = {vg.name: vg.index for vg in no.vertex_groups}
    for src_vi, new_vi in src_to_new.items():
        for g in src_me.vertices[src_vi].groups:
            gname = src.vertex_groups[g.group].name
            no.vertex_groups[name_to_idx[gname]].add([new_vi], g.weight, 'REPLACE')

    # Color attribute: preserve source colors OR override with tint
    domain = 'CORNER' if preserve_corner else 'POINT'
    ca = nm.color_attributes.new(name=color_attr_name, type='BYTE_COLOR',
                                  domain=domain)
    if override_rgba is None and src_color_attr is not None:
        # Copy original per-loop / per-vert colors from src
        if preserve_corner and src_color_attr.domain == 'CORNER':
            for ni, fi in enumerate(face_idxs):
                sp = src_me.polygons[fi]
                np = nm.polygons[ni]
                for li_new, li_src in zip(np.loop_indices, sp.loop_indices):
                    ca.data[li_new].color = src_color_attr.data[li_src].color
        else:
            for src_vi, new_vi in src_to_new.items():
                ca.data[new_vi].color = src_color_attr.data[src_vi].color
    else:
        # Override with single tint colour
        rgba = override_rgba or (1.0, 1.0, 1.0, 1.0)
        for d in ca.data:
            d.color = rgba

    return no


# --------------------------------- ENTRY ------------------------------------
def extract_mouth_parts(cfg):
    src = _obj(cfg["source"])
    src_me = src.data
    color_attr = src_me.color_attributes.get(cfg["color_attr"])
    if color_attr is None:
        raise RuntimeError(f"source '{src.name}' has no color attribute "
                           f"'{cfg['color_attr']}' -- nothing to split on")

    print(f"=== extract_mouth_parts -> {src.name} ===")
    print(f"  reading vertex-color tags from '{color_attr.name}' "
          f"({color_attr.domain}/{color_attr.data_type})")

    # Pre-compute dominant tag per face
    face_tag = [_face_dominant_tag(p, color_attr) for p in src_me.polygons]
    from collections import Counter
    print(f"  tag distribution: {dict(Counter(face_tag).most_common())}")

    tgt_coll = cfg.get("target_collection")
    if isinstance(tgt_coll, str):
        tgt_coll = bpy.data.collections.get(tgt_coll)

    created = []
    consumed = set()

    # ---- Build per-output objects ----
    for entry in cfg["outputs"]:
        matches = entry["match"] if isinstance(entry["match"], list) else [entry["match"]]
        seed_idxs = [fi for fi, tag in enumerate(face_tag) if tag in matches]
        if not seed_idxs:
            print(f"  skip '{entry['name']}': no faces match {matches}")
            continue

        # Grow seed via connected-flood-fill with orientation filter to catch
        # back/side faces of the same volume without leaking into adjacent
        # regions (e.g. teeth caught + lip flesh excluded by vertical filter).
        n_seed = len(seed_idxs)
        grown = _grow_connected_oriented(
            seed_idxs, src_me,
            max_horizontal_z=entry.get("max_horizontal_z"),
            max_hops=entry.get("max_hops"))
        # Exclude faces already consumed by an earlier entry (avoids overlap
        # between teeth + mouth interior, etc -- first match wins).
        face_idxs = sorted(grown - consumed)
        print(f"  '{entry['name']}': seed={n_seed} -> grown={len(grown)} "
              f"-> after-dedupe={len(face_idxs)}")
        if not face_idxs:
            print(f"    all faces consumed by earlier entry -- skipping")
            continue
        consumed.update(face_idxs)

        scale_map = entry.get("scale_to_match") or {}

        if entry.get("split_by_z"):
            # Bisect by face-center Z (world)
            mw = src.matrix_world
            z_of = lambda fi: (mw @ src_me.polygons[fi].center).z
            face_idxs.sort(key=z_of)
            zs = [z_of(fi) for fi in face_idxs]
            median = zs[len(zs) // 2]
            upper = [fi for fi in face_idxs if z_of(fi) >  median]
            lower = [fi for fi in face_idxs if z_of(fi) <= median]
            mat = _ensure_material(entry["material"], entry.get("tint_rgba"))
            for side, fis in (("Upper", upper), ("Lower", lower)):
                name = f"{entry['name']}_{side}"
                obj = _build_submesh_object(
                    src, fis, name, tgt_coll, cfg["color_attr"],
                    entry.get("tint_rgba"), cfg["preserve_corner_colors"],
                    color_attr)
                if obj:
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                    for p in obj.data.polygons:
                        p.material_index = 0
                    created.append(obj)
                    print(f"  + '{name}': {len(obj.data.vertices)}v "
                          f"{len(obj.data.polygons)}f  mat={mat.name}")
                    ref_name = scale_map.get(side)
                    if ref_name:
                        _uniform_scale_to_match(obj, ref_name)
        else:
            mat = _ensure_material(entry["material"], entry.get("tint_rgba"))
            obj = _build_submesh_object(
                src, face_idxs, entry["name"], tgt_coll, cfg["color_attr"],
                entry.get("tint_rgba"), cfg["preserve_corner_colors"],
                color_attr)
            if obj:
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                for p in obj.data.polygons:
                    p.material_index = 0
                created.append(obj)
                print(f"  + '{entry['name']}': {len(obj.data.vertices)}v "
                      f"{len(obj.data.polygons)}f  mat={mat.name}")
                # For non-split outputs, look up scale_map[""] or scale_map[name]
                ref_name = (scale_map.get(entry["name"]) if isinstance(scale_map, dict)
                            else scale_map)
                if isinstance(ref_name, str):
                    _uniform_scale_to_match(obj, ref_name)

    # ---- Reduce source: keep only lip-flesh-tagged faces ----
    keep_tag = cfg["lips_keep_match"]
    keep_face_idxs = [fi for fi, tag in enumerate(face_tag) if tag == keep_tag]
    drop_face_idxs = [fi for fi in range(len(src_me.polygons))
                      if fi not in set(keep_face_idxs)]
    print(f"  reducing '{src.name}': keep {len(keep_face_idxs)} "
          f"'{keep_tag}' faces, drop {len(drop_face_idxs)} others")

    # Use Edit-mode bpy.ops to delete (preserves attributes)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = src
    src.select_set(True)
    bpy.ops.object.mode_set(mode='OBJECT')
    for p in src_me.polygons:
        p.select = False
    for fi in drop_face_idxs:
        src_me.polygons[fi].select = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='FACE')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Assign the lips material to reduced source
    lips_mat = _ensure_material(cfg["lips_material"], cfg.get("lips_tint_rgba"))
    src.data.materials.clear()
    src.data.materials.append(lips_mat)
    for p in src.data.polygons:
        p.material_index = 0

    # Overwrite source's color attribute with the lip-flesh tint so the post-
    # merge head's 'lips' section has a consistent baseline (downstream masks
    # rely on this). Keep the attribute name + domain unchanged.
    rgba = cfg.get("lips_tint_rgba", (1.0, 0.3, 0.3, 1.0))
    for d in color_attr.data:
        d.color = rgba

    print(f"  '{src.name}' reduced to: {len(src.data.vertices)}v "
          f"{len(src.data.polygons)}f  mat={lips_mat.name}")

    lips_ref = cfg.get("lips_scale_to_match")
    if lips_ref:
        _uniform_scale_to_match(src, lips_ref)

    print(f"\n[done] created {len(created)} mouth-part objects; "
          f"'{src.name}' is now lip-flesh-only")
    return created


if __name__ == "__main__":
    extract_mouth_parts(CONFIG)
