"""
donor_registry.py

Central, single-source-of-truth mapping of pipeline donor ROLES to actual
Blender object names. The pipeline scripts and `face_base_apply` read
from here so swapping a donor (e.g. when processing a male head, switch
the skeleton donor from a female Fortnite head to a male one) is one
edit instead of a sweep across a dozen CONFIG dicts.

Each role maps to either a string (single object) or a dict (for the
skeleton role which has both an armature and a head mesh).

A pipeline script that wants to know "where's the ARKit donor?" calls
`donor("arkit")` and gets back the current object name.
"""


# ----------------------------------- ROLES ----------------------------------
DONORS = {
    # SKELETON / Fortnite-compatible bones + Fortnite-native blendshapes.
    # The armature defines the bone hierarchy and rest pose; the head_mesh
    # is what `headswap_transfer` BVH-binds against for weight transfer.
    # Swap this entry when processing a male head (use a male Fortnite head).
    "skeleton": {
        "armature": "Fortnite_Armature",
        "head":     "Fortnite_Head_LOD0",
    },

    # CUSTOMIZATION / Mutable mesh deformers -- nose / ear / cheek shape
    # customization morphs. Not all 14 keys are usually needed; the per-head
    # workflow picks which ones to layer.
    "customization": "Mutable_BaseBody",

    # ARKit (52 facial-capture blendshapes for LiveLink / MetaHuman Animator).
    "arkit": "ARKit_Head",
}


# ------------------------------- HELPERS ------------------------------------
def donor(role, sub=None):
    """Look up a donor by role.

        donor("arkit")                    -> "ARKit_Head"
        donor("skeleton", "armature")     -> "Fortnite_Armature"
        donor("skeleton", "head")         -> "Fortnite_Head_LOD0"
        donor("customization")            -> "Mutable_BaseBody"
    """
    val = DONORS.get(role)
    if val is None:
        raise KeyError(f"unknown donor role '{role}'  -- known: {list(DONORS.keys())}")
    if isinstance(val, dict):
        if sub is None:
            return val
        if sub not in val:
            raise KeyError(f"sub-role '{sub}' not in donor '{role}'  -- known: {list(val.keys())}")
        return val[sub]
    return val


def all_donor_object_names():
    """Flat list of every donor object name (for scripts that need to keep
    all donors visible during a render, etc.)."""
    out = []
    for v in DONORS.values():
        if isinstance(v, dict):
            out.extend(v.values())
        else:
            out.append(v)
    return out
