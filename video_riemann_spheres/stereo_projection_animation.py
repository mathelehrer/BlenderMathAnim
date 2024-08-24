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
import colorsys

context = bpy.context
scene = context.scene

bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

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
    
    
def get_sphere_material(color = (1,1,1,1),emission =0,alpha = 0.75):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs['Emission Strength'].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs['Emission'].default_value =color
    bsdf.inputs['Base Color'].default_value=color
    bsdf.inputs['Metallic'].default_value=0 #metallicity
    bsdf.inputs['Roughness'].default_value=0 #roughness
    bsdf.inputs['Alpha'].default_value= alpha#Alpha
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat
    
def get_coordinate_material(color = (0.013,0.8,0,1),emission =0.1):
    mat = bpy.data.materials.new("material")
    if len(color)==3:
        color = (color[0],color[1],color[2],1)
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs['Emission Strength'].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs['Emission'].default_value =color
    bsdf.inputs['Base Color'].default_value=color
    bsdf.inputs['Metallic'].default_value=1 #metallicity
    bsdf.inputs['Roughness'].default_value=0.1 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat

    
def get_guide_material(color = (1,1,1,1),emission =10):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs['Emission Strength'].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs['Emission'].default_value =color
    bsdf.inputs['Base Color'].default_value=color
    bsdf.inputs['Metallic'].default_value=1 #metallicity
    bsdf.inputs['Roughness'].default_value=0 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs['Base Color'],output.inputs[0])
    return mat    
    
def custom_sphere(pos,r,sub_divs):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=sub_divs, ring_count=sub_divs, radius=r, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    obj = bpy.context.active_object
    me = obj.data  # Selects the cylinders's data
    bm = bmesh.new()   # Creates an empty BMesh
    bm.from_mesh(me)   # Fills it in using the plane 

    for vert in bm.verts:
        x,y,z=vert.co[:]
        vert.co = Vector((z,x,y))+pos
    
    bm.to_mesh(me)  # Freeing the BMesh, moving on to coloring the domain
    bm.free()
  
    return obj



def projection(u,v,Z):
    Z=0
    den = (1+Z)**2+u*u+v*v
    x = 2*(1+Z)**2*u/den
    y = 2*(1+Z)**2*v/den
    z = (u*u+v*v-(1+Z)**2)/den*(1+Z)
    vec_out = Vector((x,y,z))
    
    return vec_out

spheres_x  =[]
points_x=[]
mat_emission_x = []
mat_alpha_x=[]

dim = 10
#dim = 30

for j  in range(-dim,dim+1,6):
    for i in range(-dim,dim+1,2):
        x=  i/10
        y = j/10
        points_x.append([x,y,0])
        sphere = custom_sphere(Vector((x,y,0)),0.02,8)
        angle = np.angle(x+1j*y)
        col = colorsys.hsv_to_rgb((0.5*angle/np.pi)%1,1,1)
        mat = get_coordinate_material(col,1)
        spheres_x.append(sphere)
        sphere.data.materials.append(mat)
        mat_emission_x.append(mat.node_tree.nodes["Principled BSDF"].inputs['Emission Strength'])
        mat_alpha_x.append(mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'])

        sphere.shape_key_add(name='plane')
        sk = sphere.shape_key_add(name='sphere')
        for k in range(len(sk.data)):
            X,Y,Z = sk.data[k].co[:] 
            sk.data[k].co = projection(X,Y,Z)

spheres_y  =[]
points_y = []
mat_emission_y = []
mat_alpha_y=[]


for j  in range(-dim,dim+1,6):
    for i in range(-dim,dim+1,2):
        y=  i/10
        x = j/10
        points_y.append([x,y,0])
        sphere = custom_sphere(Vector((x,y,0)),0.02,8)
        angle = np.angle(x+1j*y)
        col = colorsys.hsv_to_rgb((0.5*angle/np.pi)%1,1,1)
        mat = get_coordinate_material(col,0.3)
        spheres_y.append(sphere)
        sphere.data.materials.append(mat)
        mat_emission_y.append(mat.node_tree.nodes["Principled BSDF"].inputs['Emission Strength'])
        mat_alpha_y.append(mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'])
        
        sphere.shape_key_add(name='plane')
        sk = sphere.shape_key_add(name='sphere')
        for k in range(len(sk.data)):
            X,Y,Z = sk.data[k].co[:]
            sk.data[k].co = projection(X,Y,Z)
 
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
    
    
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
#bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
bpy.ops.object.modifier_add(type='SUBSURF')
sphere = context.active_object
transparent = get_sphere_material()
sphere.data.materials.append(transparent)

#bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
#plane = context.active_object
#plane.data.materials.append(get_sphere_material((1,1,1,1),0.01,0.4))


bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'


coords_list = [[0,0,1],[0,0,0]]
path = bpy.data.curves.new('path','CURVE')
path.dimensions = '3D'
spline = path.splines.new(type='NURBS')
spline.points.add(1)
spline.use_endpoint_u= True

for p,new_co in zip(spline.points,coords_list):
    p.co = (new_co+[1.0]) #add nurbs weight
    
track = bpy.data.objects.new('guide',path)
scene.collection.objects.link(track)

spline = bpy.data.objects['guide'].data.splines['Base Color']
    
path.bevel_mode = 'ROUND'
path.bevel_depth = 0.002
track_mat = get_guide_material()
path.materials.append(track_mat)
track_emission = track_mat.node_tree.nodes["Principled BSDF"].inputs['Emission Strength']
track.shape_key_add(name='[0,0,0]')


for i in range(len(points_x)):
    sk = track.shape_key_add(name = str(points_x[i]))
    for j in range(len(sk.data)):
        if sk.data[j].co[2]==0:
            sk.data[j].co = points_x[i]
    
for i in range(len(points_y)):
    sk = track.shape_key_add(name = str(points_y[i]))
    for j in range(len(sk.data)):
        if sk.data[j].co[2]==0:
            sk.data[j].co = points_y[i]

#######################################
######################Animations
######################################################


# set first and last frame index
total_time = 110 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = int(total_time*fps)+1
bpy.context.scene.frame_set(1)

#make all emissions zero
track_emission.default_value = 0
track_emission.keyframe_insert("default_value")

for emission in mat_emission_x:
    emission.default_value = 0
    emission.keyframe_insert("default_value")
    
for emission in mat_emission_y:
    emission.default_value = 0
    emission.keyframe_insert("default_value")

#switch on light depending on the distance

dist = 0
offset = 1
is_shining_x = []
for point in points_x:
    is_shining_x.append(False)
    
is_shining_y = []
for point in points_y:
    is_shining_y.append(False)
        
while dist<5.5:
    
    for i,point in enumerate(points_x):
        if not is_shining_x[i]:
            if np.sqrt(point['Base Color']*point['Base Color']+point[1]*point[1]+point[2]*point[2])<dist:
                is_shining_x[i]=True
                bpy.context.scene.frame_set(offset)
                emission = mat_emission_x[i]
                emission.default_value = 0
                alpha = mat_alpha_x[i]
                alpha.default_value = 0
                alpha.keyframe_insert("default_value")
                emission.keyframe_insert("default_value")
                bpy.context.scene.frame_set(offset+10)
                emission.default_value = 0.3
                alpha.default_value = 1
                alpha.keyframe_insert("default_value")
                emission.keyframe_insert("default_value")
    
    for i,point in enumerate(points_y):
        if not is_shining_y[i]:
            if np.sqrt(point['Base Color']*point['Base Color']+point[1]*point[1]+point[2]*point[2])<dist:
                bpy.context.scene.frame_set(offset)
                is_shining_y[i]=True
                emission = mat_emission_y[i]
                alpha = mat_alpha_y[i]
                alpha.default_value = 0
                alpha.keyframe_insert("default_value")
                emission.default_value = 0
                emission.keyframe_insert("default_value")
                bpy.context.scene.frame_set(offset+10)
                emission.default_value = 0.3
                alpha.default_value = 1
                alpha.keyframe_insert("default_value")
                emission.keyframe_insert("default_value")
    
    offset+=1
    bpy.context.scene.frame_set(offset)
    dist+=0.03


pause = 15

def all_track_shapes_zero():
    for block in track.data.shape_keys.key_blocks:
        block.value = 0
        block.keyframe_insert("value")


offset=300
bpy.context.scene.frame_set(offset-30)
track_emission.default_value = 0
track_emission.keyframe_insert("default_value")
bpy.context.scene.frame_set(offset)
track_emission.default_value = 1
track_emission.keyframe_insert("default_value")

bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0
ttf.keyframe_insert("offset_factor")
camera.data.lens=50
camera.data.keyframe_insert("lens") 


projection_length = 5
first = True
for i in range(len(points_x)):
    #move tractor beam
    if i>0:
        sk = track.data.shape_keys.key_blocks[str(points_x[i-1])]
        sk.value = 1
        sk.keyframe_insert("value")
    sk_next = track.data.shape_keys.key_blocks[str(points_x[i])]
    sk_next.value = 0
    sk_next.keyframe_insert("value")
    offset+=1
    bpy.context.scene.frame_set(offset)
    all_track_shapes_zero()   
    sk_next.value = 1
    sk_next.keyframe_insert("value")
    
    if first:
        first = False
        track_emission.default_value = 1
        track_emission.keyframe_insert("default_value")
    
    #take care of spheres
    if i>0:
        emission = mat_emission_x[i-1]
        emission.default_value = 1
        emission.keyframe_insert("default_value")
    ssk = spheres_x[i].data.shape_keys.key_blocks["sphere"]
    ssk.value = 0
    ssk.keyframe_insert("value")
    emission = mat_emission_x[i]
    emission.default_value = 0.3
    emission.keyframe_insert("default_value")
    
    offset+=projection_length
    bpy.context.scene.frame_set(offset)
    ssk.value = 1
    ssk.keyframe_insert("value")
    
    emission = mat_emission_x[i]
    emission.default_value = 100
    emission.keyframe_insert("default_value")
    
camera_path.location[2]=5
camera_path.keyframe_insert("location")
    
offset = offset+10
bpy.context.scene.frame_set(offset)
projection_length = 2
for i in range(len(points_y)):
    
    #move tractor beam
    if i>0:
        sk = track.data.shape_keys.key_blocks[str(points_y[i-1])]
        sk.value = 1
        sk.keyframe_insert("value")
    sk_next = track.data.shape_keys.key_blocks[str(points_y[i])]
    sk_next.value = 0
    sk_next.keyframe_insert("value")
    offset+=1
    bpy.context.scene.frame_set(offset)
    all_track_shapes_zero()
    sk_next.value = 1
    sk_next.keyframe_insert("value")
    
    
    #handle spheres
    if i>0:
        emission = mat_emission_y[i-1]
        emission.default_value = 1
        emission.keyframe_insert("default_value")
    else:
        emission = mat_emission_x[-1]
        emission.default_value = 1
        emission.keyframe_insert("default_value")
        
    ssk = spheres_y[i].data.shape_keys.key_blocks["sphere"]
    ssk.value = 0
    ssk.keyframe_insert("value")
    
    emission = mat_emission_y[i]
    emission.default_value = 0.3
    emission.keyframe_insert("default_value")
    
    offset+=projection_length
    bpy.context.scene.frame_set(offset)
    ssk.value = 1
    ssk.keyframe_insert("value")
    
    emission = mat_emission_y[i]
    emission.default_value = 100
    emission.keyframe_insert("default_value")
    

# make tractor disappear
bpy.context.scene.frame_set(offset-15)
track_emission.default_value = 1
track_emission.keyframe_insert("default_value")
bpy.context.scene.frame_set(offset)
track_emission.default_value = 0
track_emission.keyframe_insert("default_value")
emission = mat_emission_x[-1]
emission.default_value = 1
emission.keyframe_insert("default_value")

ttf.offset_factor = 0.5
ttf.keyframe_insert("offset_factor")
camera_path.location[2]=-5
camera_path.keyframe_insert("location")
#camera.data.lens=100
#camera.keyframe_insert("lens")  
    
print(offset)