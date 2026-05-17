"""
bake_section_masks.py

Bake the per-face `_section` attribute into per-loop RGBA vertex colors
so UE/UEFN materials can mask each section independently for player
customization (lip color, eyebrow tint, ear blush, etc.).

Two color attributes are written by default, matching the two-slot
shader scheme:

  - "MaskMap_Skin"  (for the M_Skin material):
      R = brow areas (eyebrow_l + eyebrow_r)
      G = lash areas (eyelid_*_upper + eyelid_*_lower)
      B = ear areas  (ear_l + ear_r)
      A = skin baseline (head)

  - "MaskMap_Mouth" (for the M_Mouth material):
      R = lips (whole lips region; UE shader can recolor)
      G = teeth   (placeholder; populated when teeth are merged in)
      B = tongue  (placeholder; populated when tongue is merged in)
      A = mouth_interior (placeholder)

Each material in UE/UEFN samples the SAME color attribute (Vertex Color
node), then plugs individual channels into Lerp / Mask nodes. So lip
color customisation = MaterialInstance parameter * MaskMap_Mouth.R, etc.

Per-loop (CORNER) domain gives crisp section boundaries with no
averaging at welded boundary verts.
"""

import bpy


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "target":       "LowPolyHead_Rigged",
    "section_attr": "_section",

    # Color attributes to bake. Each entry: attribute name + map of
    # channel ("R"/"G"/"B"/"A") -> list of section tags to fill in that channel.
    # Sections not mentioned in any channel of an attribute get all-zero loops
    # in that attribute (transparent / no mask).
    "attributes": [
        {
            "name": "MaskMap_Skin",
            "channels": {
                "R": ["eyebrow_l", "eyebrow_r"],
                "G": ["eyelid_l_upper", "eyelid_l_lower",
                      "eyelid_r_upper", "eyelid_r_lower"],
                "B": ["ear_l", "ear_r"],
                "A": ["head"],
            },
        },
        {
            "name": "MaskMap_Mouth",
            "channels": {
                "R": ["lips"],
                "G": ["teeth_upper", "teeth_lower"],
                "B": ["tongue"],
                "A": [],
            },
        },
    ],

    # If True, replace any existing color attribute with the same name.
    "replace_existing": True,
}


# ------------------------------- HELPERS ------------------------------------
def _ensure_color_attr(me, name, replace=True):
    """Create or replace a per-loop (CORNER) RGBA byte color attribute."""
    existing = me.color_attributes.get(name)
    if existing is not None:
        if replace:
            me.color_attributes.remove(existing)
        else:
            return existing
    attr = me.color_attributes.new(name=name, type='BYTE_COLOR', domain='CORNER')
    return attr


# --------------------------------- ENTRY ------------------------------------
def bake_section_masks(cfg):
    obj = bpy.data.objects.get(cfg["target"])
    if obj is None:
        raise RuntimeError(f"object '{cfg['target']}' not found")
    me = obj.data
    sec = me.attributes.get(cfg["section_attr"])
    if sec is None:
        raise RuntimeError(f"target has no '{cfg['section_attr']}' attribute "
                           f"-- run merge_face_meshes first")

    print(f"=== bake_section_masks -> {obj.name} ===")

    # Per-loop section assignment: each loop inherits its parent face's section.
    n_loops = len(me.loops)
    loop_section = [None] * n_loops
    for fi, p in enumerate(me.polygons):
        s = sec.data[fi].value.decode('utf-8')
        for li in p.loop_indices:
            loop_section[li] = s

    channel_idx = {"R": 0, "G": 1, "B": 2, "A": 3}

    for entry in cfg["attributes"]:
        attr = _ensure_color_attr(me, entry["name"], cfg.get("replace_existing", True))
        # Build per-section channel vector (one [r,g,b,a] per known section).
        section_color = {}
        section_summary = {}
        for ch_letter, sections in entry["channels"].items():
            ci = channel_idx[ch_letter]
            for s in sections:
                section_color.setdefault(s, [0.0, 0.0, 0.0, 0.0])
                section_color[s][ci] = 1.0
                section_summary.setdefault(s, []).append(ch_letter)

        zero_color = [0.0, 0.0, 0.0, 0.0]
        # Sweep loops and assign colors.
        section_loop_count = {}
        for li in range(n_loops):
            s = loop_section[li]
            col = section_color.get(s, zero_color)
            attr.data[li].color = col
            section_loop_count[s] = section_loop_count.get(s, 0) + 1

        print(f"  '{entry['name']}':")
        for s, chans in sorted(section_summary.items()):
            lc = section_loop_count.get(s, 0)
            print(f"      {''.join(chans):4} <- {s:18} ({lc} loops)")
        unmapped = [s for s in section_loop_count if s not in section_summary]
        if unmapped:
            print(f"      (no mask) {unmapped}")

    return [e["name"] for e in cfg["attributes"]]


if __name__ == "__main__":
    bake_section_masks(CONFIG)
