import bpy

def print_hierarchy(obj, indent=0):
      prefix = "  " * indent
      obj_type = obj.type
      if obj_type == 'ARMATURE':
          print(f"{prefix}{obj.name} [{obj_type}]")
          # Print top-level bones
          for bone in obj.data.bones:
              if bone.parent is None:
                  print(f"{prefix}  (bone) {bone.name}")
      else:
          print(f"{prefix}{obj.name} [{obj_type}]")

      for child in obj.children:
          print_hierarchy(child, indent + 1)

# Find SKM_UEFN_Mannequin container
for obj in bpy.data.objects:
      if "UEFN_Mannequin" in obj.name and obj.parent is None and obj.type == 'EMPTY':
          print("=== SOURCE HIERARCHY ===")
          print_hierarchy(obj)
          print(f"\nContainer parent: {obj.parent}")
          break