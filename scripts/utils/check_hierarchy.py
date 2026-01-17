import bpy

def print_hierarchy(obj, indent=0):
      prefix = "  " * indent
      obj_type = obj.type
      if obj_type == 'ARMATURE':
          print(f"{prefix}{obj.name} [{obj_type}]")
          for bone in obj.data.bones:
              if bone.parent is None:
                  print(f"{prefix}  (bone) {bone.name}")
      else:
          print(f"{prefix}{obj.name} [{obj_type}]")
      for child in obj.children:
          print_hierarchy(child, indent + 1)

# Check Export collection
for obj in bpy.data.collections['Export'].objects:
      if obj.parent is None:
          print("=== EXPORT HIERARCHY ===")
          print_hierarchy(obj)