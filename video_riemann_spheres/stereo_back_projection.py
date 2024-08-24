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
    bsdf.inputs[18].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=0 #metallicity
    bsdf.inputs[7].default_value=0 #roughness
    bsdf.inputs[19].default_value= alpha#Alpha
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
    bsdf.inputs[18].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=1 #metallicity
    bsdf.inputs[7].default_value=0.1 #roughness
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
    bsdf.inputs[18].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=1 #metallicity
    bsdf.inputs[7].default_value=0 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
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

def polar2cartesian(theta,phi):
    return Vector((np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),np.sin(theta)))

def cartesian2complex(coords):
    return coords.x/(1-coords.z)+1j*coords.y/(1-coords.z)


#sequences that are treated equally but sequentially
sphere_seqs = []
coords_seqs = []
emission_seqs = []
alpha_seqs = []

dim = 5
detail = 4
radius = 0.0001
reg_epsilon = 0.001

#latitudes
for t  in range(dim,-dim+1,-1):
    points = []
    spheres = []
    emissions =[]
    alphas = []
    for p in range(-detail*dim,detail*dim+1):
        theta = np.pi/2*t/dim+reg_epsilon
        phi = np.pi*p/dim/detail
        point = polar2cartesian(theta,phi)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.02, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        sphere = bpy.context.active_object
        sphere.location = point
        z = cartesian2complex(point)
        col = colorsys.hsv_to_rgb((0.5*np.angle(z)/np.pi)%1,1,1)
        mat = get_coordinate_material(col,1)
        sphere.data.materials.append(mat)
        emissions.append(mat.node_tree.nodes["Principled BSDF"].inputs[18])
        alphas.append(mat.node_tree.nodes["Principled BSDF"].inputs[19])
        spheres.append(sphere)
        points.append(point)
    sphere_seqs.append(spheres)
    coords_seqs.append(points)
    emission_seqs.append(emissions)
    alpha_seqs.append(alphas)
    
print("latitudes finished")
    
#longitudes
for p  in range(-dim,dim+1):
    points = []
    spheres = []
    emissions =[]
    alphas = []
    for t in range(-detail*dim,detail*dim+1):
        theta = np.pi/2*t/dim/detail+reg_epsilon
        phi = np.pi*p/dim
        point = polar2cartesian(theta,phi)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.02, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        sphere = bpy.context.active_object
        sphere.location = point
        z = cartesian2complex(point)
        col = colorsys.hsv_to_rgb((0.5*np.angle(z)/np.pi)%1,1,1)
        mat = get_coordinate_material(col,1)
        sphere.data.materials.append(mat)
        emissions.append(mat.node_tree.nodes["Principled BSDF"].inputs[18])
        alphas.append(mat.node_tree.nodes["Principled BSDF"].inputs[19])
        spheres.append(sphere)
        points.append(point)
    sphere_seqs.append(spheres)
    coords_seqs.append(points)
    emission_seqs.append(emissions)
    alpha_seqs.append(alphas)

print("longitudes finished")
 
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
    
bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
bpy.ops.object.modifier_add(type='SUBSURF')
sphere = context.active_object
transparent = get_sphere_material()
sphere.data.materials.append(transparent)

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'


#######################################
######################Animations
######################################################


# set first and last frame index
total_time = 80 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = int(total_time*fps)+1
bpy.context.scene.frame_set(1)

camera_path.location[2]=5
camera_path.keyframe_insert("location")
camera.data.lens = 100
camera.data.keyframe_insert("lens")

ttf.offset_factor = 0
ttf.keyframe_insert("offset_factor")
  

pause = 15
max_emission = 5

#make all emissions zero
for emissions in emission_seqs:
    for emission in emissions:
        emission.default_value = 0
        emission.keyframe_insert("default_value")
        
#make all alphas zero
for alphas in alpha_seqs:
    for alpha in alphas:
        alpha.default_value = 0
        alpha.keyframe_insert("default_value")
        
#make all scales zero
for spheres in sphere_seqs:
    for sphere in spheres:
        sphere.scale[0] = 0
        sphere.scale[1] = 0
        sphere.scale[2] = 0
        sphere.keyframe_insert("scale")
        
#turn on all spheres
offset = 5
for spheres,alphas,emissions,coords in zip(sphere_seqs,alpha_seqs,emission_seqs,coords_seqs):
    for sphere,alpha,emission,coord in zip(spheres,alphas,emissions,coords):
        bpy.context.scene.frame_set(offset)
        alpha.default_value = 0
        alpha.keyframe_insert("default_value")
        sphere.scale[0] = 0
        sphere.scale[1] = 0
        sphere.scale[2] = 0
        sphere.keyframe_insert("scale")
        bpy.context.scene.frame_set(offset+5)
        alpha.default_value = 1
        alpha.keyframe_insert("default_value")
        sphere.scale[0]=0.1+0.9*(1-np.abs(coord.z))
        sphere.scale[1]=0.1+0.9*(1-np.abs(coord.z))
        sphere.scale[2]=0.1+0.9*(1-np.abs(coord.z))
        sphere.keyframe_insert("scale")
        bpy.context.scene.frame_set(offset+15)
        emission.default_value = max_emission
        emission.keyframe_insert("default_value")
        bpy.context.scene.frame_set(offset+30)
        emission.default_value = 1
        emission.keyframe_insert("default_value")
        offset +=1    
    offset +=pause
    
print("spheres turned on")

#move all spheres to the plane
count = 0
for spheres,coords,emissions in zip(sphere_seqs,coords_seqs,emission_seqs):
    for sphere,coord,emission in zip(spheres,coords,emissions):
        bpy.context.scene.frame_set(offset)
        sphere.keyframe_insert("location")
        sphere.keyframe_insert("scale")
        emission.default_value = 1
        emission.keyframe_insert("default_value")
        bpy.context.scene.frame_set(offset+5)
        emission.default_value = max_emission
        emission.keyframe_insert("default_value")
        duration = 30-int(15*count/len(sphere_seqs))
        bpy.context.scene.frame_set(offset+duration)
        z = cartesian2complex(coord)
        pos = Vector((np.real(z),np.imag(z),0))
        sphere.location= pos
        sphere.scale[0]=1
        sphere.scale[1]=1
        sphere.scale[2]=1
        sphere.keyframe_insert("location")
        sphere.keyframe_insert("scale")
        bpy.context.scene.frame_set(offset+duration+5)
        emission.default_value = 0.3
        emission.keyframe_insert("default_value")
        offset = offset+1
    count+=1 
    offset +=15  

print("spheres moved to plane")

offset +=30
bpy.context.scene.frame_set(offset)
#set camera motion  
ttf.offset_factor = 1
ttf.keyframe_insert("offset_factor")
camera_path.location[2]=-5
camera_path.keyframe_insert("location")
camera.data.lens = 50
camera.data.keyframe_insert("lens")

    
print(offset)