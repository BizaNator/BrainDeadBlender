"""
section_to_vgroup.py

Convert the `_section` face attribute on a merged head into named vertex
groups (`section_<name>` per section) so the section data is selectable
in any DCC and survives FBX export to Unreal.

Why
---
`_section` is a face attribute -- visible in Blender's Attributes panel
but invisible to Maya/Unreal after FBX export. Material slots are
universal but limit you to 1 material per face. Named vertex groups
travel through FBX as skin influences, so they survive the round trip
into UEFN/Unreal and can be queried at runtime (e.g. spawn particles
only on "section_lips" verts).

Output: one vertex group per unique section, named `section_<name>`,
with every vert in that section at weight=1.0. Verts shared by multiple
sections (boundary welds) get weight=1.0 in EACH section's group.

Idempotent: existing `section_*` groups are cleared before being
rebuilt.
"""

import bpy


CONFIG = {
    "target":     "LowPolyHead_Rigged",
    "section_attr": "_section",
    "prefix":     "section_",
    # If True, also write a `section_index` integer vertex attribute
    # (one per vert, value = section index). Useful for shaders that
    # want to branch on section without 8 separate channel reads.
    "write_index_attr": True,
}


def section_to_vgroup(cfg):
    obj = bpy.data.objects.get(cfg["target"])
    if obj is None or obj.type != 'MESH':
        raise RuntimeError(f"target '{cfg['target']}' not a mesh")

    me = obj.data
    attr = me.attributes.get(cfg["section_attr"])
    if attr is None:
        raise RuntimeError(f"face attribute '{cfg['section_attr']}' missing on '{obj.name}'")
    if attr.domain != 'FACE':
        raise RuntimeError(f"attribute '{cfg['section_attr']}' must be FACE domain")

    # Build: section_name -> set of vert indices
    section_verts = {}
    for fi, p in enumerate(me.polygons):
        s = attr.data[fi].value.decode('utf-8')
        section_verts.setdefault(s, set()).update(p.vertices)

    print(f"=== section_to_vgroup -> {obj.name} ===")
    print(f"  sections found ({len(section_verts)}):")

    # Clear existing section_* vgroups
    to_remove = [vg for vg in obj.vertex_groups
                 if vg.name.startswith(cfg["prefix"])]
    for vg in to_remove:
        obj.vertex_groups.remove(vg)
    if to_remove:
        print(f"  cleared {len(to_remove)} existing '{cfg['prefix']}*' groups")

    # Create vgroup per section
    for s, vis in sorted(section_verts.items()):
        vg_name = cfg["prefix"] + s
        vg = obj.vertex_groups.new(name=vg_name)
        vg.add(list(vis), 1.0, 'REPLACE')
        print(f"    {s:30s} -> {vg_name:35s}  {len(vis):4d}v  "
              f"({len([fi for fi in range(len(me.polygons)) if attr.data[fi].value.decode('utf-8') == s])} faces)")

    # Optional: integer per-vert attribute mapping vert -> section index.
    # Verts shared by multiple sections get the section index they
    # belong to MOST (most faces) -- this is a one-shot tag.
    if cfg.get("write_index_attr", True):
        from collections import Counter
        # Build vert -> Counter of section votes (one vote per face)
        vert_sec_votes = {}
        for fi, p in enumerate(me.polygons):
            s = attr.data[fi].value.decode('utf-8')
            for vi in p.vertices:
                vert_sec_votes.setdefault(vi, Counter())[s] += 1
        sec_index = {s: i for i, s in enumerate(sorted(section_verts.keys()))}
        # Write attribute
        attr_name = "section_index"
        existing = me.attributes.get(attr_name)
        if existing:
            me.attributes.remove(existing)
        idx_attr = me.attributes.new(attr_name, 'INT', 'POINT')
        for vi in range(len(me.vertices)):
            votes = vert_sec_votes.get(vi)
            if votes is None:
                idx_attr.data[vi].value = -1
            else:
                top = votes.most_common(1)[0][0]
                idx_attr.data[vi].value = sec_index[top]
        print(f"  wrote int attribute 'section_index' (POINT): "
              f"{', '.join(f'{i}={s}' for s, i in sec_index.items())}")

    print(f"[done] {len(section_verts)} section vgroups created on '{obj.name}'")
    return section_verts


if __name__ == "__main__":
    section_to_vgroup(CONFIG)
