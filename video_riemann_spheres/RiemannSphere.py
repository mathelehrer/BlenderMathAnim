import bpy
import numpy as np
from mathutils import Vector

# delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.shading_system = True

scale = 10

def Make_grid(NX,NY,dX,dY):
    x = int(NX/2)
    y = int(NY/2)
    faces = []
    v_list = list([[i*dX,j*dY,0] for i in range(-x,x+1) for j in range(-y,y+1)])
    
    for j in range(NY):
        for i in range(NX):
            lst = []
            lst.append(i*(NY+1)+j)
            lst.append(i*(NY+1)+j+1)
            lst.append((i+1)*(NY+1)+j+1)
            lst.append((i+1)*(NY+1)+j)
            faces.append(lst)

    print(v_list)
    print(faces)
    new_mesh=bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(v_list,[],faces)
    new_mesh.update()
    
    new_object=bpy.data.objects.new('new_plane',new_mesh)
#    new_collection=bpy.data.collections.new('new_collection')
#    bpy.context.scene.collection.children.link(new_collection)
    bpy.data.collections['Collection'].objects.link(new_object)
    return new_object


def get_material():
    mat = bpy.data.materials.new("complex_function_shader")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((2100,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.location = Vector((1800,300))


    node_coord = nodes.new(type="ShaderNodeTexCoord")
    node_coord.location=Vector((-1150,250))
    
    
    node_sub = nodes.new(type="ShaderNodeVectorMath")
    node_sub.location=Vector((-900,250))
    node_sub.operation='SUBTRACT'
    node_sub.inputs[1].default_value=Vector((0.5,0.5,0))
    
    node_scale = nodes.new(type="ShaderNodeVectorMath")
    node_scale.location=Vector((-650,250))
    node_scale.operation='SCALE'
    node_scale.inputs[3].default_value=2
    
    node_scale2 = nodes.new(type="ShaderNodeVectorMath")
    node_scale2.location=Vector((-400,250))
    node_scale2.operation='SCALE'
    node_scale2.inputs[3].default_value=scale
    
    node_xyz_s = nodes.new(type="ShaderNodeSeparateXYZ")
    node_xyz_s.location=Vector((-150,250))
    
    node_xyz_m = nodes.new(type="ShaderNodeCombineXYZ")
    node_xyz_m.location=Vector((50,250))
    node_xyz_m.inputs[2].default_value=0

    node_script = nodes.new(type="ShaderNodeScript")
    node_script.location=Vector((275,250))
    node_script.mode='INTERNAL'
    node_script.script = bpy.data.texts["complexFunction.osl"]
    
    node_xyz_sep = nodes.new(type="ShaderNodeSeparateXYZ")
    node_xyz_sep.location=Vector((500,250))
    
    node_atan2 = nodes.new(type="ShaderNodeMath")
    node_atan2.location=Vector((700,250))
    node_atan2.operation='ARCTAN2'
      
    node_div = nodes.new(type="ShaderNodeMath")
    node_div.location=Vector((900,250))
    node_div.operation='DIVIDE'
    node_div.inputs[1].default_value=6.282318
    
    node_add = nodes.new(type="ShaderNodeMath")
    node_add.location=Vector((1100,250))
    node_add.operation='ADD'
    node_add.inputs[1].default_value=0.5
    
    node_hue = nodes.new(type="ShaderNodeHueSaturation")
    node_hue.location=Vector((1300,250))
    node_hue.inputs[4].default_value=(0,0,1,1)
    
    node_mix = nodes.new(type="ShaderNodeMixRGB")
    node_mix.location=Vector((1500,250))
    node_mix.blend_type='MULTIPLY'
    node_mix.inputs[0].default_value=1
   
    node_r = nodes.new(type="ShaderNodeVectorMath")
    node_r.operation='LENGTH'
    node_r.location=Vector((700,0))
    
    
    node_contour = nodes.new(type="ShaderNodeScript")
    node_contour.location=Vector((1000,0))
    node_contour.mode='INTERNAL'
    node_contour.script = bpy.data.texts["contour.osl"]
    node_contour.inputs[1].default_value=0.5
   
    
    
      
    links =mat.node_tree.links
    links.new(node_coord.outputs[0],node_sub.inputs[0])
    links.new(node_sub.outputs[0],node_scale.inputs[0])
    links.new(node_scale.outputs[0],node_scale2.inputs[0])
    links.new(node_scale2.outputs[0],node_xyz_s.inputs[0])
    links.new(node_xyz_s.outputs[0],node_xyz_m.inputs[0])
    links.new(node_xyz_s.outputs[1],node_xyz_m.inputs[1])
    links.new(node_xyz_m.outputs[0],node_script.inputs[0])
    links.new(node_script.outputs[0],node_xyz_sep.inputs[0])
    links.new(node_xyz_sep.outputs[1],node_atan2.inputs[0])
    links.new(node_xyz_sep.outputs[0],node_atan2.inputs[1])
    links.new(node_atan2.outputs[0],node_div.inputs[0])
    links.new(node_div.outputs[0],node_add.inputs[0])
    links.new(node_add.outputs[0],node_hue.inputs[0])
    links.new(node_hue.outputs[0],node_mix.inputs[1])    
    links.new(node_mix.outputs[0],bsdf.inputs[0])
    
    links.new(node_script.outputs[0],node_r.inputs[0])
    links.new(node_r.outputs[1],node_contour.inputs[0])
    links.new(node_contour.outputs[0],node_mix.inputs[2])
    
    links.new(bsdf.outputs[0],output.inputs[0])
    
    return mat


#add material to the plane
#bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
#obj = bpy.context.active_object
#obj.data.materials.append(get_material())
   
NX = 400
NY = 400
dX = 0.005
dY = 0.005
plane = Make_grid(NX,NY,dX,dY)
plane.data.materials.append(get_material())

verts = plane.data.vertices

sk_basis = plane.shape_key_add(name='Basis')
sk_basis.interpolation = 'KEY_LINEAR'
plane.data.shape_keys.use_relative = True


sk = plane.shape_key_add(name='Deform')
sk.interpolation = 'KEY_LINEAR'


# position each vert
for i in range(len(verts)):
    x,y,z=sk.data[i].co[:]
    x=x*scale
    y=y*scale
    z=z*scale
    r2 = x*x + y*y
    nv = 1./(r2 + 4) * Vector((4*x, 4*y,-2*r2 ))
    sk.data[i].co= nv

#add camera
camera = bpy.ops.object.camera_add(location=(0,2.3,2.9),rotation=(55/180*3.14159,0,0))
track_to = bpy.context.object.constraints.new('TRACK_TO')
track_to.target = plane
#bpy.context.object.constraints["Track To"].target = bpy.data.objects[plane]

#add light
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 10), scale=(1, 1, 1))
bpy.context.object.data.energy = 2

#add light
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, -10), scale=(1, 1, 1))
bpy.context.object.data.energy = 2
track_to = bpy.context.object.constraints.new('TRACK_TO')
track_to.target = plane