"""
build_parts_library.py

Copy each named face sub-mesh (CustomLips, Eyelid_*, Eyebrow_*, Ear_*,
etc.) into a `_PartsLibrary` collection so the rigged source parts are
preserved before merging into the head shell. Future heads can grab a
library copy by name (e.g. `Lib_Eyebrow_L`) and skip rebuilding from
scratch.

Naming: each copied part is prefixed `Lib_<OriginalName>` to make it
obvious it's a library copy. The original objects stay where they are.

Run BEFORE `merge_face_meshes.py` so the merge has source data to copy
from. Subsequent runs replace existing library copies in place.
"""

import bpy


# ----------------------------------- CONFIG ---------------------------------
CONFIG = {
    # Which parts to library-copy. Add new entries here for future characters.
    "parts": [
        "CustomLips",
        "Tongue",
        "Eye_L", "Eye_R",
        "Eyelid_L_Upper", "Eyelid_L_Lower",
        "Eyelid_R_Upper", "Eyelid_R_Lower",
        "Eyebrow_L", "Eyebrow_R",
        "Ear_L", "Ear_R",
        # Teeth ride bones rigidly (Upper=head, Lower=C_jaw). Library copies
        # let each character re-use the same teeth mesh; fit_accessories'
        # `fit_to_section: "lips"` auto-positions them per-head. Restyle by
        # hand-sculpting / Tripo-regenning the Lib_Teeth_* objects later;
        # any character that adopts the library will pick up the new style.
        "Teeth_Upper", "Teeth_Lower",
        # MouthInterior = dark cavity backdrop extracted from CustomLips'
        # BLACK + BLUE Tripo-color faces. Rides head bone, no deformation.
        "MouthInterior",
    ],

    # Target collection (created if missing). Lives at scene root for easy
    # access across heads.
    "library_collection": "_PartsLibrary",

    # Library prefix on copied object names.
    "prefix": "Lib_",

    # Replace existing library copy if one already exists.
    "replace_existing": True,

    # If True, also strip parent + armature modifier from the library copy
    # (so it's a freestanding mesh you can drop into a new head's collection).
    "freestanding": True,
}


# ------------------------------- HELPERS ------------------------------------
def _ensure_collection(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def _remove_object(obj):
    me = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(me, bpy.types.Mesh) and me.users == 0:
        bpy.data.meshes.remove(me)


def _copy_object(src, new_name, target_coll):
    new_me = src.data.copy()
    new_obj = src.copy()
    new_obj.data = new_me
    new_obj.name = new_name
    new_me.name = new_name + "_mesh"
    target_coll.objects.link(new_obj)
    return new_obj


# --------------------------------- ENTRY ------------------------------------
def build_parts_library(cfg):
    lib = _ensure_collection(cfg["library_collection"])
    print(f"=== build_parts_library -> {lib.name} ===")

    copied = []
    for name in cfg["parts"]:
        src = bpy.data.objects.get(name)
        if src is None:
            print(f"  skip '{name}': not in scene")
            continue
        lib_name = cfg["prefix"] + name
        existing = bpy.data.objects.get(lib_name)
        if existing and cfg.get("replace_existing", True):
            _remove_object(existing)
        elif existing:
            print(f"  skip '{name}': '{lib_name}' already exists")
            continue
        new_obj = _copy_object(src, lib_name, lib)
        if cfg.get("freestanding", True):
            # Preserve world transform across the parent-strip. Without this,
            # the library copy snaps to its pre-parent local coords (matrix_basis
            # only), which for parts re-parented to the armature with non-
            # identity matrix_parent_inverse means they end up far from where
            # the artist placed them. Capture world matrix BEFORE clearing parent,
            # then restore after.
            world_mat = new_obj.matrix_world.copy()
            new_obj.parent = None
            new_obj.matrix_world = world_mat
            for m in list(new_obj.modifiers):
                if m.type == 'ARMATURE':
                    new_obj.modifiers.remove(m)
        copied.append(new_obj.name)
        print(f"  '{name}' -> '{lib_name}' ({len(new_obj.data.vertices)}v)")

    print(f"\n[done] {len(copied)} library parts in '{lib.name}'")
    return copied


if __name__ == "__main__":
    build_parts_library(CONFIG)
