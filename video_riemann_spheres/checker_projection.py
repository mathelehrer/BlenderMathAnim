import bpy
import numpy as np
from mathutils import Vector

# delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.shading_system = True


#set background to black
bpy.data.worlds['World'].node_tree.nodes["Background"].inputs[0].default_value=(0,0,0,1)

#basic transformation helper functions
def degToRadian(angle):
    return angle*np.pi/180

def move_obj(name,coords):
    bpy.data.objects[name].location =coords
    
def rotate_obj(name,angles):
    rotation = [degToRadian(angle) for angle in angles]
    bpy.data.objects[name].rotation_euler = rotation

def make_grid(NX,NY,dX,dY):
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

    new_mesh=bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(v_list,[],faces)
    new_mesh.update()
    
    new_object=bpy.data.objects.new('new_plane',new_mesh)
    bpy.data.collections['Collection'].objects.link(new_object)
    return new_object

def get_texture_material():
    material = bpy.data.materials.new("texture_shader")

    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((2100,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.location = Vector((1800,300))


    node_coord = nodes.new(type="ShaderNodeTexCoord")
    node_coord.location=Vector((1500,250))
    
    
    node_math = nodes.new(type="ShaderNodeVectorMath")
    node_math.operation = 'MULTIPLY'
    node_math.inputs[1].default_value=(2,1,0)
    
    
    node_checker = nodes.new(type="ShaderNodeTexChecker")
    node_checker.location=Vector((1200,250))
    node_checker.inputs[1].default_value=(1,0,0,1)
    node_checker.inputs[2].default_value=(0,1,0,1)
    node_checker.inputs[3].default_value = 10
    
    
    links =material.node_tree.links
    links.new(node_coord.outputs[0],node_math.inputs[0])
    links.new(node_math.outputs[0],node_checker.inputs[0])
    links.new(node_checker.outputs[0],bsdf.inputs[17])
    links.new(bsdf.outputs[0],output.inputs[0])
    
   
    
    return material


def scaling():
    verts = plane.data.vertices
    
    sk_plane = plane.shape_key_add(name='Plane')
    sk_plane.interpolation = 'KEY_LINEAR'
    plane.data.shape_keys.use_relative = True
    
    sk_scaled_plane = plane.shape_key_add(name='ScaledPlane')
    sk_scaled_plane.interpolation = 'KEY_LINEAR'
    
    for i in range(len(verts)):
        u,v,z = sk_scaled_plane.data[i].co[:]
        u=np.pi*u
        v=np.pi/2*v
        nv = Vector((u,v,z))
        sk_scaled_plane.data[i].co=nv
        
    return sk_scaled_plane


def projection_to_sphere():
    verts = plane.data.shape_keys.key_blocks['ScaledPlane'].data[:]
    
    sk_sphere = plane.shape_key_add(name='Sphere')
    sk_sphere.interpolation = 'KEY_LINEAR'
    
    for i,vert in enumerate(verts):
        u,v,z = vert.co[:]
        u = u
        v = v
        nv = Vector((np.cos(v)*np.cos(u),np.cos(v)*np.sin(u),np.sin(v)))
        sk_sphere.data[i].co=nv
        
    return sk_sphere

#other auxiliary functions:




scale = 1


NX = 40
NY = 40
dX = 0.05
dY = 0.05
plane = make_grid(NX,NY,dX,dY)
plane.data.materials.append(get_texture_material())

#scaling animation
# set first and last frame index
total_time = 30 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 1
end = int(total_time*fps)+1
third = end/3
twothirds = 2*end/3
bpy.context.scene.frame_end = end

#shape keying to scale the plane
shape_key1 = scaling()
setattr(shape_key1,'value',0)
shape_key1.keyframe_insert(data_path='value',frame=1)
setattr(shape_key1,'value',1)
shape_key1.keyframe_insert(data_path='value',frame=third)


#shape key for projection to sphere
shape_key2 = projection_to_sphere()
setattr(shape_key2,'value',0)
shape_key2.keyframe_insert(data_path='value',frame=twothirds)
setattr(shape_key2,'value',1)
shape_key2.keyframe_insert(data_path='value',frame=end)

setattr(shape_key1,'value',1)
shape_key1.keyframe_insert(data_path='value',frame=twothirds)
setattr(shape_key1,'value',0)
shape_key1.keyframe_insert(data_path='value',frame=end)


#create camera position
bpy.ops.object.camera_add()
camera = bpy.context.active_object.name
move_obj(camera,[0,-8,0])
rotate_obj(camera,[90,0,0])