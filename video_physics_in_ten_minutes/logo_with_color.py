import bpy
import numpy as np
import bmesh
from mathutils import Vector
from random import random



#make objects glow
bpy.context.scene.eevee.use_bloom = True
#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


camera_pos = (0,1,5.7)
# add camera and target to empty
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=camera_pos, rotation=(0.78, 0, 0), scale=(1, 1, 1))
camera = bpy.context.active_object

bpy.ops.object.empty_add(type='CIRCLE', radius=0.18, align='WORLD', location=(0, 1, 0), scale=(1, 1, 1))
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
    bsdf.inputs[18].default_value=7.5  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value = color


    links =mat.node_tree.links

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


# set first and last frame index
total_time = 10 # Animation should be 2*pi seconds long
fps =29.97  # Frames per second (fps)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(total_time*fps)+1




cylinders =[]

dim = 40
for i in range(-dim,dim+1):
    bpy.context.scene.frame_set(90+int(30*random()))
    scale = 1/(1/4*i*i+1/2)
    r = 1/4*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=(i/2*scale, 3/4*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    obj.data.materials.append(get_material((0.8, 0, 0.00318879, 1)))
    obj.keyframe_insert(data_path="location")
    obj.keyframe_insert("rotation_euler")
    cylinders.append(obj)
    
for i in range(-dim,dim+1):
    bpy.context.scene.frame_set(90+int(30*random()))
    scale = 1/(1/4*i*i-1/4*i+96/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 9/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    obj.data.materials.append(get_material((0.00318879, 0.8, 0., 1)))
    obj.keyframe_insert(data_path="location")
    obj.keyframe_insert("rotation_euler")
    cylinders.append(obj)
    
for i in range(-dim,dim+1):
    bpy.context.scene.frame_set(90+int(30*random()))
    scale = 1/(1/4*i*i-1/4*i+240/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 15/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    obj.data.materials.append(get_material((0.00318879, 0, 0.8, 1)))
    obj.keyframe_insert(data_path="location")
    obj.keyframe_insert("rotation_euler")
    cylinders.append(obj)
    


    
for cylinder in cylinders:
    bpy.context.scene.frame_set(121+int(179*random()))
    scaling = 15+15*random()
    rnd = Vector((random(),random(),random()))*20
    rel_pos_to_camera =cylinder.location-Vector(camera_pos) 
    cylinder.location = Vector(camera_pos)+rel_pos_to_camera*scaling + Vector((rel_pos_to_camera[0]*scaling,0,0))+rnd # move cylinder away from camera
    for i in range(0,3):
        cylinder.rotation_euler[i]=random()*31.41
    cylinder.keyframe_insert(data_path="location")
    cylinder.keyframe_insert("rotation_euler")
