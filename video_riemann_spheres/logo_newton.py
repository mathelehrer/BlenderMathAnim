import bpy
import numpy as np
import bmesh
import random
from mathutils import Vector



#make objects glow
bpy.context.scene.eevee.use_bloom = True
#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# add camera and target to empty
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(-2.5, -0.9, 10.7), rotation=(0.78, 0, 0), scale=(1, 1, 1))
camera = bpy.context.active_object

bpy.ops.object.empty_add(type='CIRCLE', radius=0.18, align='WORLD', location=(-2.5, 0.0, 0), scale=(1, 1, 1))
empty = bpy.context.active_object
camera.constraints.new('TRACK_TO').target = empty




bpy.ops.mesh.primitive_circle_add(radius=0.5, enter_editmode=False, align='WORLD', location=(0, 0.5, 0), scale=(1, 1, 1))
bpy.ops.mesh.primitive_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 1, 0), scale=(1, 1, 1))


def get_material(color):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=0.5  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value = color


    links =mat.node_tree.links

    links.new(bsdf.outputs[0],output.inputs[0])
    
    return mat

spheres = ["seq_z03.png","seq_z04.png","seq_z05.png","seq_z06.png","seq_z07.png","seq_z08.png","seq_z09.png","seq_z10.png","seq_z11.png","seq_z12.png","seq_z13.png"]


def load_texture(number=0):
    mat = bpy.data.materials.new(spheres[number])

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=2  #increase emission strength
    bsdf.location = Vector((300,300))
    
    textureShader = nodes.new(type='ShaderNodeTexImage')
    textureShader.location=Vector((100,0))
    image = bpy.data.images.load(filepath="/home/jmartin/Dropbox/MyBlender/ComplexSurfaces/"+spheres[number])
    textureShader.image = image
    
    links =mat.node_tree.links
    links.new(textureShader.outputs[0],bsdf.inputs[17])
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat


def extrude(object):
    me = object.data
    bm = bmesh.new()
    bm.from_mesh(me)
    faces = bm.faces[:]
    for face in faces:
        r = bmesh.ops.extrude_discrete_faces(bm, faces=[face])
        bmesh.ops.translate(bm, vec=Vector((0,0,0.1)), verts=r['faces'][0].verts)
        bm.to_mesh(me)
        me.update()


def new_sphere(number=0):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128,ring_count=128, enter_editmode=False, align='WORLD', location=(-2.5, -0.64, 4), scale=(1, 1, 1))
    obj = bpy.context.active_object
    obj.data.materials.append(load_texture(number))
    return obj
   

# set first and last frame index
total_time =30 # Animation should be 2*pi seconds long
fps =30 # Frames per second (fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = int(total_time*fps)+1


frame = 1
five_seconds= 60


dim = 5
count = 0
for i in range(-dim,dim+1):
    scale = 1/(1/4*i*i+1/2)
    r = 1/4*scale
    sphere = new_sphere(number=count)
    sphere.hide_render = True
    sphere.keyframe_insert('hide_render',frame=1)
    sphere.hide_render = False
    sphere.keyframe_insert('hide_render',frame=frame)
    bpy.context.scene.frame_set(frame)
    final_size = r
    final_location=(i/2*scale, 3/4*scale, 0)
    sphere.keyframe_insert(data_path="location")
    sphere.keyframe_insert("rotation_euler")
    sphere.keyframe_insert("scale")
    frame = frame+five_seconds
    bpy.context.scene.frame_set(frame)
    axis = int(random.random()*3)    
    sphere.rotation_euler[axis]=3.1415926
    sphere.location = final_location
    sphere.scale = (r,r,r)
    sphere.keyframe_insert(data_path="location")
    sphere.keyframe_insert("rotation_euler")
    sphere.keyframe_insert("scale")
    count=count+1
    
for i in range(-dim,dim+1):
    scale = 1/(1/4*i*i-1/4*i+96/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 9/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    obj.hide_render = True
    obj.keyframe_insert('hide_render',frame=1)
    obj.hide_render = False
    obj.keyframe_insert('hide_render',frame=730+5*i)
    obj.data.materials.append(get_material((0.00318879, 0.8, 0., 1)))
    
for i in range(-dim,dim+1):
    scale = 1/(1/4*i*i-1/4*i+240/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 15/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    obj.hide_render = True
    obj.keyframe_insert('hide_render',frame=1)
    obj.hide_render = False
    obj.keyframe_insert('hide_render',frame=730+5*i)
    obj.data.materials.append(get_material((0.00318879, 0, 0.8, 1)))

