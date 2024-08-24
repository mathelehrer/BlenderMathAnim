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


r_axes = 0.1
r_coords = 0.025
r_sub = 0.01
r_curve = 0.064
r_sphere = 0.3

x_color = Vector((0.9,0.9,0.9,1))
y_color = Vector((0.9,0.9,0.9,1))
z_color = Vector((0.9,0.9,1,1))


def get_coordinate_material(color = (0.013,0.8,0,1)):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=0.1  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=1 #metallicity
    bsdf.inputs[7].default_value=0.1 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat

def get_gridplane_material(color = (0.8,0.8,0.8,1)):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=0.1  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=1 #metalicity
    bsdf.inputs[7].default_value=0.1 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat

def get_hue_material(angle = 0):
    h = get_hue_from_radians(angle)
    
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=0.1  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[4].default_value=1 #metalicity
    bsdf.inputs[7].default_value=0.1 #roughness
    
    hue = nodes.new(type='ShaderNodeHueSaturation')
    hue.location=Vector((100,0))
    hue.inputs[0].default_value=h
    hue.inputs[4].default_value = (1,0,0,1)
    
    links =mat.node_tree.links
    
    links.new(bsdf.outputs[0],output.inputs[0])
    links.new(hue.outputs[0],bsdf.inputs[0]) #standard color
    links.new(hue.outputs[0],bsdf.inputs[17]) #emission
    return mat
    


#remove old labels and arrows
for ob in bpy.context.scene.objects:
    if fnmatch.fnmatch(ob.name,"XYZ Function*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Cylinder*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Plane*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Sun*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Camera*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Sphere*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Empty*"):
        ob.select_set(True)
    bpy.ops.object.delete()

dim = 40

#sun 
bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
camera_empty = bpy.context.active_object
bpy.ops.object.light_add(type='SUN', radius=10, align='WORLD', location=(50, -50, 0), scale=(1, 1, 1))
sun = bpy.context.active_object
ttc = sun.constraints.new(type='TRACK_TO')
ttc.target = camera_empty

#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
 
#Camera0
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, -75,0), scale=(1, 1, 1))
camera = bpy.context.active_object
ttc = camera.constraints.new(type='TRACK_TO')
ttc.target = camera_empty


#gridlines x  
xlines = [] 
for x in range(-int(dim/2),int(dim/2)+1,10):
    if x==0:
        r=r_axes
    else:
        r=r_coords
    bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=dim)
    cylinder = bpy.context.active_object
    cylinder.location[0]=x
    cylinder.location[1]=0
    cylinder.location[2]=0
    #cylinder.rotation_euler[0]=3.14159/2
    cylinder.data.materials.append(get_coordinate_material(y_color))
    xlines.append(cylinder)
    
#gridlines y 
ylines = []
for y in range(-int(dim/2),int(dim/2)+1,10):
    if y==0:
        r=r_axes
    else:
        r=r_coords
    bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=dim)
    cylinder = bpy.context.active_object
    cylinder.location[0]=0
    cylinder.location[2]=y
    cylinder.location[1]=0
    cylinder.rotation_euler[0]=3.14159/2
    cylinder.rotation_euler[2]=3.14159/2
    cylinder.data.materials.append(get_coordinate_material(x_color))
    ylines.append(cylinder)

#arrows
bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))",z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
bpy.ops.object.editmode_toggle()
xarrow = bpy.context.active_object
#xarrow.rotation_euler[0]=-3.14159/2
xarrow.scale[0]=r_axes/0.25
xarrow.scale[1]=r_axes/0.25
xarrow.scale[2]=10*r_axes/0.25
xarrow.location[1]=0
xarrow.location[0]=0
xarrow.location[2]=1.1*dim/2
xarrow.data.materials.append(get_coordinate_material(y_color))

bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))",z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
bpy.ops.object.editmode_toggle()
yarrow = bpy.context.active_object
yarrow.rotation_euler[1]=3.14159/2
yarrow.scale[0]=r_axes/0.25
yarrow.scale[1]=r_axes/0.25
yarrow.scale[2]=10*r_axes/0.25
yarrow.location[2]=0
yarrow.location[1]=0
yarrow.location[0]=1.1*dim/2
yarrow.data.materials.append(get_coordinate_material(x_color))


def get_hue_from_radians(angle = 0):
    if angle>2*pi:
        angle = angle-2*pi*int(angle/2*pi)
    if angle>pi:
        angle = angle - 2*pi
    angle = angle/pi/2
    return 0.5+angle

#spheres
spheres = []
arrows = []
arrow_copies = []
arrow_copies_materials = []
points=[[10,0],[-10,0]]


n = [1,2,3,4,5,7,8,9,10,11]
for i in n:
    points.append([10*np.cos(2*np.pi/12*i),10*np.sin(2*np.pi/12*i)])

for p in points:
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r_sphere/200, enter_editmode=False, align='WORLD', location=(p[0],0,p[1]), scale=(1, 1, 1))
    sphere = bpy.context.active_object
    sphere.data.materials.append(get_coordinate_material((1,1,0,1)))
    spheres.append(sphere)
    
    bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+0.75*(v>=0)*(1-v))",
    z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+0.75*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
    bpy.ops.object.editmode_toggle()
    
    arrow = bpy.context.active_object
    bpy.context.scene.cursor.location = (0.0, 0.0, -2.0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR',center ='MEDIAN')
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    
    bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+0.75*(v>=0)*(1-v))",
    z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+0.75*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
    bpy.ops.object.editmode_toggle()
    
    arrow_copy = bpy.context.active_object
    bpy.context.scene.cursor.location = (0.0, 0.0, -2.0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR',center ='MEDIAN')
    
    arrow.scale[0]=1
    arrow.scale[1]=1
    arrow.scale[2]=3.3
    arrow.location[1]=0
    arrow.location[2]=p[1]
    arrow.location[0]=p[0]
    
    arrow_copy.scale[0]=1
    arrow_copy.scale[1]=1
    arrow_copy.scale[2]=3.3
    arrow_copy.location[1]=0
    arrow_copy.location[2]=p[1]
    arrow_copy.location[0]=p[0]
    
    angle = math.atan2(p[1],p[0])
    arrow.rotation_euler[1]=3.14159/2-angle
    arrow_copy.rotation_euler[0]=3.14159/2
    
    arrow.data.materials.append(get_hue_material(angle))
    mat = get_hue_material(angle)
    
    arrow_copy.data.materials.append(mat)
    arrows.append(arrow)
    arrow_copies.append(arrow_copy)
    arrow_copies_materials.append(mat)

bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
#Animations
# set first and last frame index
total_time = 40 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(total_time*fps)+1

#grow xlines in the first second
offset = 0
bpy.context.scene.frame_set(offset+1)
xarrow.location[2]=0
xarrow.scale[2]=0
xarrow.keyframe_insert("location")
xarrow.keyframe_insert("scale")

#set all arrows to 0 length first
for arrow in arrows:
    arrow.scale[2] = 0
    arrow.scale[1] = 0
    arrow.scale[0] = 0
    arrow.keyframe_insert("scale")

for line in xlines:
    line.scale[2]=0
    line.keyframe_insert("scale")  

offset = offset+30
bpy.context.scene.frame_set(offset)
for line in xlines:
    line.scale[2]=1
    line.keyframe_insert("scale")
xarrow.location[2]=1.1*dim/2
xarrow.keyframe_insert("location")
xarrow.scale[2]=10*r_axes/0.25
xarrow.keyframe_insert("scale")    
    
#grow ylines in the second second
offset = offset+30
bpy.context.scene.frame_set(offset+1)
yarrow.location[0]=0
yarrow.keyframe_insert("location")
yarrow.scale[2]=0
yarrow.keyframe_insert("scale")
for line in ylines:
    line.scale[2]=0
    line.keyframe_insert("scale")
    
offset=offset+30
bpy.context.scene.frame_set(offset)
for line in ylines:
    line.scale[2]=1
    line.keyframe_insert("scale")
yarrow.location[0]=1.1*dim/2
yarrow.keyframe_insert("location")
yarrow.scale[2]=10*r_axes/0.25
yarrow.keyframe_insert("scale")

offset=offset+30


for i in range(0,2):
    sphere = spheres[i]
    arrow = arrows[i]
    arrow_c = arrow_copies[i]
    bpy.context.scene.frame_set(offset)
    sphere.keyframe_insert("scale")
    offset=offset+30
    bpy.context.scene.frame_set(offset)
    sphere.scale[0]=400
    sphere.scale[1]=400
    sphere.scale[2]=400
    sphere.keyframe_insert("scale")
    offset = offset+60
    bpy.context.scene.frame_set(offset)
    arrow.keyframe_insert("scale")
    offset = offset+30
    bpy.context.scene.frame_set(offset)
    arrow.scale[2]=3.3
    arrow.scale[1]=1
    arrow.scale[0]=1
    arrow.keyframe_insert("scale")
    arrow_c.scale[2]=3.3
    arrow_c.scale[1]=1
    arrow_c.scale[0]=1
    arrow_c.keyframe_insert("scale")
    
for i in range(2,len(spheres)):
    sphere = spheres[i]
    arrow = arrows[i]
    arrow_c = arrow_copies[i]
    bpy.context.scene.frame_set(offset)
    sphere.keyframe_insert("scale")
    offset=offset+30
    bpy.context.scene.frame_set(offset)
    sphere.scale[0]=400
    sphere.scale[1]=400
    sphere.scale[2]=400
    sphere.keyframe_insert("scale")
    arrow.keyframe_insert("scale")
    offset = offset+30
    bpy.context.scene.frame_set(offset)
    arrow.scale[2]=3.3
    arrow.scale[1]=1
    arrow.scale[0]=1
    arrow.keyframe_insert("scale")
    arrow_c.scale[2]=3.3
    arrow_c.scale[1]=1
    arrow_c.scale[0]=1
    arrow_c.keyframe_insert("scale")    
        
camera.rotation_euler[0]=0
camera.keyframe_insert("rotation_euler")


offset = offset+30
bpy.context.scene.frame_set(offset)
camera.rotation_euler[0]=-3.14159/10*7
camera.keyframe_insert("rotation_euler")

offset0 = offset+60
offset1 = offset0+30
offset2 = offset1+30
offset = offset2+30


for i in range(0,len(spheres)):
    sphere = spheres[i]
    arrow = arrows[i]
    arrow_c = arrow_copies[i]
    bpy.context.scene.frame_set(offset0)
    sphere.keyframe_insert("location") 
    arrow.keyframe_insert("location")
    arrow.keyframe_insert("scale")
    arrow_c.keyframe_insert("location")
    arrow_c.keyframe_insert("scale")
    
    material = arrow_copies_materials[i]
    alpha_channel = material.node_tree.nodes["Principled BSDF"].inputs[19]
    bpy.context.scene.frame_set(offset1-1)
    alpha_channel.default_value=0
    alpha_channel.keyframe_insert("default_value")
    
    bpy.context.scene.frame_set(offset1+5)
    alpha_channel.default_value=1
    alpha_channel.keyframe_insert("default_value")
    
    
    bpy.context.scene.frame_set(offset1)
    sphere.location[0] = sphere.location[0]*0.1    
    sphere.location[1] = sphere.location[1]*0.1    
    sphere.location[2] = sphere.location[2]*0.1    
    arrow.location[0] = arrow.location[0]*0.1
    arrow.location[1] = arrow.location[1]*0.1
    arrow.location[2] = arrow.location[2]*0.1
    arrow.scale[2] = arrow.scale[2]*0.1
    arrow_c.location[0] = arrow_c.location[0]*0.1
    arrow_c.location[1] = arrow_c.location[1]*0.1
    arrow_c.location[2] = arrow_c.location[2]*0.1
    arrow_c.scale[2] =arrow_c.scale[2]*0.1
    sphere.keyframe_insert("location")
    arrow.keyframe_insert("location")
    arrow.keyframe_insert("scale")
    arrow_c.keyframe_insert("location")
    arrow_c.keyframe_insert("scale")
    
    bpy.context.scene.frame_set(offset2)
    sphere.location[0] = sphere.location[0]*15   
    sphere.location[1] = sphere.location[1]*15    
    sphere.location[2] = sphere.location[2]*15    
    arrow.location[0] = arrow.location[0]*15
    arrow.location[1] = arrow.location[1]*15
    arrow.location[2] = arrow.location[2]*15
    arrow.scale[2] = arrow.scale[2]*15
    arrow_c.location[0] = arrow.location[0]
    arrow_c.location[1] = arrow.location[1]
    arrow_c.location[2] = arrow.location[2]
    arrow_c.scale[2] = arrow.scale[2]
    sphere.keyframe_insert("location")
    arrow.keyframe_insert("location")
    arrow.keyframe_insert("scale")
    arrow_c.keyframe_insert("location")
    arrow_c.keyframe_insert("scale")
    alpha_channel.default_value=1
    alpha_channel.keyframe_insert("default_value")
    
    
    bpy.context.scene.frame_set(offset2+2)
    alpha_channel.default_value=0
    alpha_channel.keyframe_insert("default_value")
    
    bpy.context.scene.frame_set(offset)
    sphere.location[0] = sphere.location[0]*10/15   
    sphere.location[1] = sphere.location[1]*10/15    
    sphere.location[2] = sphere.location[2]*10/15    
    arrow.location[0] = arrow.location[0]*10/15
    arrow.location[1] = arrow.location[1]*10/15
    arrow.location[2] = arrow.location[2]*10/15
    arrow.scale[2] = arrow.scale[2]*10/15
    sphere.keyframe_insert("location")
    arrow.keyframe_insert("location")
    arrow.keyframe_insert("scale")
   



print(offset)