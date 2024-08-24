import bpy

from bpy import context, data, ops
import numpy as np
import bmesh
import mathutils
import fnmatch
from pathlib import Path
from mathutils import Vector

import os
import glob
import subprocess
import tempfile
import shutil
import math

import math
from math import sin, cos, pi
import fnmatch
import bmesh
import bpy
from mathutils import Vector

context = bpy.context
scene = context.scene



#remove old labels and arrows
for ob in bpy.context.scene.objects:
    if fnmatch.fnmatch(ob.name,"XYZ Function*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Cylinder*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Plane*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"coordinate_system*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Empty*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Camera*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Sun*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"BezierCircle*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Latex**"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Sphere*"):
        ob.select_set(True)
    bpy.ops.object.delete()
    
    
def get_glass_material(color = (0.013,0.8,0,1),emission =0):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=0 #metallicity
    bsdf.inputs[7].default_value=0 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat
    
def get_coordinate_material(color = (0.013,0.8,0,1),emission =0.1):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=1 #metallicity
    bsdf.inputs[7].default_value=0.1 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat


def projection(u,v,z):
    theta = v *np.pi/2
    phi = u *np.pi
    
    x = (1+2*z)*np.cos(theta)*np.cos(phi)
    y = (1+2*z)*np.cos(theta)*np.sin(phi)
    z = (1+2*z)*np.sin(theta)
    
    return Vector((x,y,z))


def lonlat(u,v,z):
    return Vector((u*np.pi,v*np.pi/2,z))


def custom_cylinder(length,pos,r,sub_divs,principle_direction):
    bpy.ops.mesh.primitive_cylinder_add(vertices=sub_divs/2,radius=r,depth=length)
    obj = bpy.context.active_object
    me = obj.data  # Selects the cylinders's data
    bm = bmesh.new()   # Creates an empty BMesh
    bm.from_mesh(me)   # Fills it in using the plane 

    if principle_direction==0:
        #z->x
        #x->y
        #y->z
        long_dim = 0
        for vert in bm.verts:
            x,y,z=vert.co[:]
            vert.co = Vector((z,x,y))+pos
    elif principle_direction==1:
        #z->y
        #y->x
        #x->z
        long_dim = 1
        for vert in bm.verts:
            x,y,z=vert.co[:]
            vert.co = Vector((y,z,x))+pos
    else: 
        long_dim = 2
        for vert in bm.verts:
            x,y,z=vert.co[:]
            vert.co = vert.co+pos
    
    #select long edges
    sel_edges = []
    for edge in bm.edges:
        corners = edge.verts
        if corners[0].co[long_dim]!=corners[1].co[long_dim]:
            sel_edges.append(edge)
    
    bmesh.ops.subdivide_edges(bm,edges=sel_edges,cuts=sub_divs,use_grid_fill=True)
   
    bm.to_mesh(me)  # Freeing the BMesh, moving on to coloring the domain
    bm.free()
  
    return obj

grid_x=[]
mat_x = get_coordinate_material((0,0.8,0,1))

for i in range(-10,11,1):
    x=  i/10
    line = custom_cylinder(2,Vector((x,0,0)),0.005,100,1)
    line.scale[1]=1
    grid_x.append(line)
    line.data.materials.append(mat_x)

    sk_basis = line.shape_key_add(name='uv')
    
    sk = line.shape_key_add(name='lonlat')
    for i in range(len(sk.data)):
        u,v,z = sk.data[i].co[:]
        sk.data[i].co = lonlat(u,v,z)
    
    sk = line.shape_key_add(name='sphere')
    for i in range(len(sk.data)):
        u,v,z = sk.data[i].co[:]
        sk.data[i].co = projection(u,v,z)
        
grid_y=[]
mat_y = get_coordinate_material((0.8,0,0,1))

for i in range(-10,11,1):
    y=  i/10
    line = custom_cylinder(2,Vector((0,y,0)),0.01,100,0)
    line.scale[1]=1
    grid_y.append(line)
    line.data.materials.append(mat_y)

    sk_basis = line.shape_key_add(name='flat')
    sk = line.shape_key_add(name='lonlat')
    for i in range(len(sk.data)):
        u,v,z = sk.data[i].co[:]
        sk.data[i].co = lonlat(u,v,z)
    sk = line.shape_key_add(name='sphere')
    for i in range(len(sk.data)):
        u,v,z = sk.data[i].co[:]
        sk.data[i].co = projection(u,v,z)
    
   
bpy.ops.curve.primitive_bezier_circle_add(radius=10, enter_editmode=False, align='WORLD', location=(0, 0, 5), scale=(1, 1, 1))
camera_path = bpy.context.active_object
bpy.ops.object.empty_add(type='CUBE', radius=0.1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
camera_empty=context.active_object
       

#Camera0
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0,0), scale=(1, 1, 1))
camera = bpy.context.active_object

ttf = camera.constraints.new(type='FOLLOW_PATH')
ttf.offset_factor=0
ttf.use_fixed_location=True
ttf.use_curve_follow=True
ttf.influence=1
ttf.target= camera_path
ttc = camera.constraints.new(type='TRACK_TO')
ttc.target = camera_empty
    
    
    
#######################################
######################Animations
######################################################


# set first and last frame index
total_time = 30 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(total_time*fps)+1

pause = 15
offset = 15
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.5
ttf.keyframe_insert("offset_factor")
camera.data.lens=50
camera.keyframe_insert("lens") 

# grow x_lines
for line in grid_x:
    line.scale[1]=0
    line.keyframe_insert("scale")  
    
offset = offset+60
bpy.context.scene.frame_set(offset)
for line in grid_x:
    line.scale[1]=1
    line.keyframe_insert("scale")
    
# grow x_lines
offset = offset+pause
bpy.context.scene.frame_set(offset)
for line in grid_y:
    line.scale[0]=0
    line.keyframe_insert("scale")  
    
offset = offset+60
bpy.context.scene.frame_set(offset)
for line in grid_y:
    line.scale[0]=1
    line.keyframe_insert("scale")
    
    
# stretch
offset = offset+pause
bpy.context.scene.frame_set(offset)
for line in grid_y:
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 0
    shape.keyframe_insert("value")
for line in grid_x:
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 0
    shape.keyframe_insert("value")
    
offset = offset+60
bpy.context.scene.frame_set(offset)
for line in grid_y:
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 1
    shape.keyframe_insert("value")
for line in grid_x:
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 1
    shape.keyframe_insert("value")
    
    
# to sphere
offset = offset+pause
bpy.context.scene.frame_set(offset)
for line in reversed(grid_y):
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 1
    shape.keyframe_insert("value")
    shape2 = line.data.shape_keys.key_blocks['sphere']
    shape2.value = 0
    shape2.keyframe_insert("value")
    bpy.context.scene.frame_set(offset+60)
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 0
    shape.keyframe_insert("value")
    shape2 = line.data.shape_keys.key_blocks['sphere']
    shape2.value = 1
    shape2.keyframe_insert("value")
    offset = offset+15
    

# to sphere
offset = offset+pause
bpy.context.scene.frame_set(offset)
for line in grid_x:
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 1
    shape.keyframe_insert("value")
    shape2 = line.data.shape_keys.key_blocks['sphere']
    shape2.value = 0
    shape2.keyframe_insert("value")
    bpy.context.scene.frame_set(offset+60)
    shape = line.data.shape_keys.key_blocks['lonlat']
    shape.value = 0
    shape.keyframe_insert("value")
    shape2 = line.data.shape_keys.key_blocks['sphere']
    shape2.value = 1
    shape2.keyframe_insert("value")
    offset = offset+15
  
ttf.offset_factor = 0
ttf.keyframe_insert("offset_factor")
camera.data.lens=100
camera.keyframe_insert("lens")  
    
print(offset)