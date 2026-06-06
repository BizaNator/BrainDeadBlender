"""
apply_section_materials.py

Assign a material slot per `_section` tag on a merged head. Without this
step the merge collapses every section into whatever material slot 0
already was (usually the skin material), so brows, lashes, ears, and
lips all shade as skin and appear visually merged with the face.

By default, sections are grouped into the user's UEFN material scheme:
  - 'skin'  slot: head, eyelids, ears, eyebrows, eyelashes
    (RGBA vertex-color masks within this slot can encode brows / lashes /
     ear tint / etc. for per-character customization in-engine.)
  - 'mouth' slot: lips
    (RGB vertex-color masks within THIS slot encode lip color (R), teeth (G),
     tongue (B) since the 4-channel budget on the skin slot fills fast.)
You can override the grouping via `cfg["section_to_slot"]`.

If a target material with the slot name doesn't exist, a placeholder is
created (Principled BSDF, base color tinted by slot for previewing in
Blender). Replace the placeholder with your real shader before exporting.
"""

import bpy


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    "target":       "LowPolyHead_Rigged",
    "section_attr": "_section",

    # Per-section material-slot grouping. Add new sections here as the
    # pipeline grows (e.g. body parts will get their own slots).
    "section_to_slot": {
        "head":           "M_Skin",
        "eyelid_l_upper": "M_Skin",
        "eyelid_l_lower": "M_Skin",
        "eyelid_r_upper": "M_Skin",
        "eyelid_r_lower": "M_Skin",
        "eyebrow_l":      "M_Skin",
        "eyebrow_r":      "M_Skin",
        "ear_l":          "M_Skin",
        "ear_r":          "M_Skin",
        "lips":           "M_Mouth",
        # Tongue is now merged into the head (matches Fortnite layout).
        # Same M_Mouth slot as lips -- vertex-color channels distinguish
        # tongue (B) from lips (R) per the comment above on `lips`.
        "tongue":         "M_Mouth",
    },

    # Preview tint per slot (only used when creating a placeholder material).
    # RGB tuple in linear color space.
    "slot_preview_tint": {
        "M_Skin":  (0.85, 0.72, 0.62, 1.0),
        "M_Mouth": (0.80, 0.30, 0.30, 1.0),
        "M_Eyes":  (0.10, 0.10, 0.10, 1.0),
    },

    # If True and the slot's material doesn't exist, create a placeholder.
    # If False, error out so the user notices the missing real shader.
    "create_missing_materials": True,
}


# ------------------------------- HELPERS ------------------------------------
def _placeholder_material(name, rgba):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def _ensure_material_slot(obj, mat_name, create_if_missing, tint):
    """Return the slot index for a material named mat_name on obj. Adds the
    slot if missing. Creates a placeholder material if needed."""
    for i, ms in enumerate(obj.material_slots):
        if ms.material is not None and ms.material.name == mat_name:
            return i
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        if not create_if_missing:
            raise RuntimeError(f"material '{mat_name}' not found "
                               f"(create_missing_materials=False)")
        mat = _placeholder_material(mat_name, tint)
    obj.data.materials.append(mat)
    return len(obj.material_slots) - 1


# --------------------------------- ENTRY ------------------------------------
def apply_section_materials(cfg):
    obj = bpy.data.objects.get(cfg["target"])
    if obj is None:
        raise RuntimeError(f"object '{cfg['target']}' not found")
    me = obj.data
    attr = me.attributes.get(cfg["section_attr"])
    if attr is None:
        raise RuntimeError(f"target has no '{cfg['section_attr']}' attribute "
                           f"-- run merge_face_meshes first")

    print(f"=== apply_section_materials -> {obj.name} ===")
    mapping = cfg["section_to_slot"]
    tints = cfg.get("slot_preview_tint", {})
    create = cfg.get("create_missing_materials", True)

    # Ensure all required slots exist (and track section -> slot_index).
    section_slot_idx = {}
    for section, slot_name in mapping.items():
        tint = tints.get(slot_name, (0.5, 0.5, 0.5, 1.0))
        idx = _ensure_material_slot(obj, slot_name, create, tint)
        section_slot_idx[section] = idx
        print(f"  slot[{idx}] = '{slot_name}'  <- {section}")

    # Walk faces and reassign material_index by section.
    from collections import Counter
    unknown = Counter()
    reassigned = 0
    for fi, p in enumerate(me.polygons):
        s = attr.data[fi].value.decode('utf-8')
        target_idx = section_slot_idx.get(s)
        if target_idx is None:
            unknown[s] += 1
            continue
        if p.material_index != target_idx:
            p.material_index = target_idx
            reassigned += 1

    print(f"\n  reassigned {reassigned} faces to section-specific slots")
    if unknown:
        print(f"  WARN unknown sections (no slot mapping): {dict(unknown)}")

    return section_slot_idx


if __name__ == "__main__":
    apply_section_materials(CONFIG)
