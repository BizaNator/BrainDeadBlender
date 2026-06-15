import bpy

def find_collection_ci(name: str):
    want = name.strip().lower()
    for col in bpy.data.collections:
        if col.name.strip().lower() == want:
            return col
    return None

def objects_in_collection(col):
    return list(col.all_objects)

def pick_largest_mesh(col):
    meshes = [o for o in objects_in_collection(col) if o.type == "MESH"]
    meshes.sort(key=lambda o: len(o.data.vertices) if o.data else 0, reverse=True)
    return meshes[0]

def find_single_armature(col):
    arms = [o for o in objects_in_collection(col) if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"Expected exactly 1 armature in '{col.name}', found {len(arms)}")
    return arms[0]

tgt_col = find_collection_ci("Target")
tgt_mesh = pick_largest_mesh(tgt_col)
tgt_arm  = find_single_armature(tgt_col)

print("Target mesh:", tgt_mesh.name)
print("Target armature:", tgt_arm.name)
print("Mesh parent:", tgt_mesh.parent.name if tgt_mesh.parent else None)

# If mesh is parented to the armature, clear parent but keep world transform
if tgt_mesh.parent == tgt_arm:
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    tgt_mesh.select_set(True)
    bpy.context.view_layer.objects.active = tgt_mesh
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    print("✅ Cleared parent (keep transform).")

print("Mesh parent now:", tgt_mesh.parent.name if tgt_mesh.parent else None)
