import bpy

for o in bpy.data.objects:
    if "hand_writing" in o.name:
        mat = o.material_slots[0].material
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Emission'].default_value = bsdf.inputs['Base Color'].default_value
        bsdf.inputs['Emission Strength'].default_value = 5
        # o.visible_shadow=False
        
        
        
for o in bpy.data.objects:
    if "render" in o.name:
        mat = o.material_slots[0].material
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Emission'].default_value = bsdf.inputs['Base Color'].default_value
        bsdf.inputs['Emission Strength'].default_value = 0.5
        # o.visible_shadow=False
        
for o in bpy.data.objects:
    if "func" in o.name:
        o.visible_shadow=False
        
for c in bpy.data.curves:
    mat = c.materials[0]
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Emission'].default_value = bsdf.inputs['Base Color'].default_value
    bsdf.inputs['Emission Strength'].default_value = 1
        
for o in bpy.data.objects:
    if "func" in o.name:
        mat = o.material_slots[0].material
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Emission'].default_value = bsdf.inputs['Base Color'].default_value
        bsdf.inputs['Emission Strength'].default_value = 0.1

for o in bpy.data.objects:
    if "Display" in o.name:
        o.visible_shadow=False
        
for m in bpy.data.materials:
    if "mirror" in m.name:
        bsdf = m.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Metallic'].default_value = 1
        bsdf.inputs['Roughness'].default_value = 0
        bsdf.inputs['Specular'].default_value=0
        bsdf.inputs['Specular Tint'].default_value=0
        bsdf.inputs['Anisotropic'].default_value=0
        bsdf.inputs['Anisotropic Rotation'].default_value=0
        bsdf.inputs['Sheen'].default_value=0
        bsdf.inputs['Sheen Tint'].default_value=0
        bsdf.inputs['Clearcoat'].default_value=0
        bsdf.inputs['Clearcoat Roughness'].default_value=0
        bsdf.inputs['IOR'].default_value=0