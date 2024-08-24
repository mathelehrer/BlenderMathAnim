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
link_object = scene.collection.objects.link if bpy.app.version >= (2, 80) else scene.objects.link
unlink_object = scene.collection.objects.unlink if bpy.app.version >= (2, 80) else scene.objects.unlink


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
    if fnmatch.fnmatch(ob.name,"Parametric*"):
        ob.select_set(True)
    bpy.ops.object.delete()


def get_mirror_material(alpha = 1):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links =mat.node_tree.links

    output = nodes.get("Material Output")
    output.location = Vector((1000,200))
    
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs['Emission Strength'].default_value=0.1  #increase emission strength
    bsdf.location = Vector((600,300))
    bsdf.inputs['Metallic'].default_value=1 #metalicity
    bsdf.inputs['Roughness'].default_value=0.0 #roughness
    bsdf.inputs['Alpha'].default_value = alpha #set alpha value
    bsdf.inputs['Emission'].default_value=(0.1,0.1,0.1,1) # some gray emission
    links.new(bsdf.outputs[0],output.inputs[0])
    
    return mat

def get_line_material(alpha=1):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links =mat.node_tree.links

    output = nodes.get("Material Output")
    output.location = Vector((1000,200))
    
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs['Emission Strength'].default_value=0.1  #increase emission strength
    bsdf.location = Vector((600,300))
    bsdf.inputs['Metallic'].default_value=1 #metalicity
    bsdf.inputs['Roughness'].default_value=0.1 #roughness
    bsdf.inputs['Alpha'].default_value = alpha #set alpha value
    links.new(bsdf.outputs[0],output.inputs[0])
    
    hue = nodes.new(type='ShaderNodeHueSaturation')
    hue.location = Vector((400,300))
    hue.inputs[1].default_value=1
    hue.inputs[2].default_value=1
    hue.inputs[3].default_value=1
    hue.inputs[4].default_value=(0,1,1,1.)
    links.new(hue.outputs[0],bsdf.inputs['Base Color'])
    links.new(hue.outputs[0],bsdf.inputs['Emission'])
    
    # convert radians to hue 
    
    m1 = nodes.new(type='ShaderNodeMath')
    m1.operation='MODULO'
    m1.location=Vector((200,300))
    m1.inputs[1].default_value = 1
    links.new(m1.outputs[0],hue.inputs[0])
    
    m2 = nodes.new(type='ShaderNodeMath')
    m2.operation='DIVIDE'
    m2.location=Vector((0,300))
    m2.inputs[1].default_value = 2.*np.pi
    links.new(m2.outputs[0],m1.inputs[0])
    
    m3 = nodes.new(type='ShaderNodeMath')
    m3.operation='ARCTAN2'
    m3.location=Vector((-200,300))
    links.new(m3.outputs[0],m2.inputs[0])
       
    #  mix between the hue of the complex plane and
    #              the hue of the cosine function
       
    mix1 = nodes.new(type='ShaderNodeMixRGB')
    mix1.location=Vector((-500,500))
    mix1.inputs[0].default_value=0
    links.new(mix1.outputs[0],m3.inputs[1])
    
    mix2 = nodes.new(type='ShaderNodeMixRGB')
    mix2.location=Vector((-500,0)) 
    mix2.inputs[0].default_value=0
    links.new(mix2.outputs[0],m3.inputs[0])
    
    # calculate the real and imaginary part of z*z*z-1
    # real
    # x**3-3x*y**2-1
    r1= nodes.new(type='ShaderNodeMath')
    r1.operation='SUBTRACT'
    r1.location=Vector((-700,100))
    r1.inputs[1].default_value = 1
    links.new(r1.outputs[0],mix1.inputs[2])
    
    r2= nodes.new(type='ShaderNodeMath')
    r2.operation='SUBTRACT'
    r2.location=Vector((-900,100))
    r2.inputs[1].default_value = 1
    links.new(r2.outputs[0],r1.inputs[0])

    # x**3
    x1= nodes.new(type='ShaderNodeMath')
    x1.operation='POWER'
    x1.location=Vector((-1100,300))
    x1.inputs[1].default_value = 3
    links.new(x1.outputs[0],r2.inputs[0])
     
    #3*x*y**2
    r4= nodes.new(type='ShaderNodeMath')
    r4.operation='MULTIPLY'
    r4.location=Vector((-1300,100))
    links.new(r4.outputs[0],r2.inputs[1])
    
    #y**2
    y1= nodes.new(type='ShaderNodeMath')
    y1.operation='POWER'
    y1.location=Vector((-1500,100))
    y1.inputs[1].default_value = 2
    links.new(y1.outputs[0],r4.inputs[1])
        
    #x*3
    x2= nodes.new(type='ShaderNodeMath')
    x2.operation='MULTIPLY'
    x2.location=Vector((-1700,100))
    x2.inputs[1].default_value = 3
    links.new(x2.outputs[0],r4.inputs[0])
    
    
    #imag
    # 3x**2*y-y**3
    i1= nodes.new(type='ShaderNodeMath')
    i1.operation='SUBTRACT'
    i1.location=Vector((-700,-100))
    i1.inputs[1].default_value = 1
    links.new(i1.outputs[0],mix2.inputs[2])

    # y**3
    y2= nodes.new(type='ShaderNodeMath')
    y2.operation='POWER'
    y2.location=Vector((-1100,-300))
    y2.inputs[1].default_value = 3
    links.new(y2.outputs[0],i1.inputs[1])
     
    #3*y*x**2
    i3= nodes.new(type='ShaderNodeMath')
    i3.operation='MULTIPLY'
    i3.location=Vector((-1300,-100))
    links.new(i3.outputs[0],i1.inputs[0])
    
    #x**2
    x3= nodes.new(type='ShaderNodeMath')
    x3.operation='POWER'
    x3.location=Vector((-1500,-100))
    x3.inputs[1].default_value = 2
    links.new(x3.outputs[0],i3.inputs[1])
        
    #y*3
    y3= nodes.new(type='ShaderNodeMath')
    y3.operation='MULTIPLY'
    y3.location=Vector((-1700,-100))
    y3.inputs[1].default_value = 3
    links.new(y3.outputs[0],i3.inputs[0])
   
    # mix between plane coordinates and coordinates of the Riemann sphere
    
    mix_real = nodes.new(type='ShaderNodeMixRGB')
    mix_real.location=Vector((-2100,500))
    mix_real.inputs[0].default_value=0
    links.new(mix_real.outputs[0],x1.inputs[0]) 
    links.new(mix_real.outputs[0],x2.inputs[0]) 
    links.new(mix_real.outputs[0],x3.inputs[0]) 
    links.new(mix_real.outputs[0],mix1.inputs[1]) 
   
    
    mix_imag = nodes.new(type='ShaderNodeMixRGB')
    mix_imag.location=Vector((-2100,0)) 
    mix_imag.inputs[0].default_value=0
    links.new(mix_imag.outputs[0],y1.inputs[0]) 
    links.new(mix_imag.outputs[0],y2.inputs[0]) 
    links.new(mix_imag.outputs[0],y3.inputs[0]) 
    links.new(mix_imag.outputs[0],mix2.inputs[1]) 
   
    #do spherical reprojection and rescaling for radius 5
       
    real = nodes.new(type='ShaderNodeMath')
    real.operation='DIVIDE'
    real.location = Vector((-2300,500))
    links.new(real.outputs[0],mix_real.inputs[2])
   
    imag = nodes.new(type='ShaderNodeMath')
    imag.operation='DIVIDE'
    imag.location = Vector((-2300,0))
    links.new(imag.outputs[0],mix_imag.inputs[2])
    
    m_mul_real = nodes.new(type='ShaderNodeMath')
    m_mul_real.operation='MULTIPLY'
    m_mul_real.location = Vector((-2500,500))
    m_mul_real.inputs[1].default_value = 0.4     # 2 divided by r = 5
    links.new(m_mul_real.outputs[0],real.inputs[0])
   
    m_mul_imag = nodes.new(type='ShaderNodeMath')
    m_mul_imag.operation='MULTIPLY'
    m_mul_imag.inputs[1].default_value = 0.4     # 2 divided by r = 5
    m_mul_imag.location = Vector((-2500,100))
    links.new(m_mul_imag.outputs[0],imag.inputs[0])
   
    
    m_abs = nodes.new(type='ShaderNodeMath')
    m_abs.operation ='ABSOLUTE'
    m_abs.location = Vector((-2600,0))
    links.new(m_abs.outputs[0],real.inputs[1])
    links.new(m_abs.outputs[0],imag.inputs[1])
   
    m_sub = nodes.new(type='ShaderNodeMath')
    m_sub.operation = 'SUBTRACT'
    m_sub.location =Vector((-2700,200))
    m_sub.inputs[0].default_value=2
    links.new(m_sub.outputs[0],m_abs.inputs[0])
    
   
    sep = nodes.new(type='ShaderNodeSeparateXYZ')
    sep.location = Vector((-3100,300))
    links.new(sep.outputs[0],mix_real.inputs[1])
    links.new(sep.outputs[1],mix_imag.inputs[1])
    links.new(sep.outputs[0],real.inputs[0])
    links.new(sep.outputs[1],imag.inputs[0])
    links.new(sep.outputs[2],m_sub.inputs[1])
    
   
    input = nodes.new(type='ShaderNodeNewGeometry')
    input.location = Vector((-3300,300))
    links.new(input.outputs[0],sep.inputs[0])
    return mat
    

def get_coordinate_material(color = (0.013,0.8,0,1),emission =0.1):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs['Emission'].default_value=emission  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs['Emission'].default_value =color
    bsdf.inputs['Base Color'].default_value=color
    bsdf.inputs['Metallic'].default_value=1 #metallicity
    bsdf.inputs['Roughness'].default_value=0.1 #roughness
    links =mat.node_tree.links
    links.new(bsdf.outputs[0],output.inputs[0])
    return mat


def apply_stereo_proj_to(object):

    verts = object.data.vertices

    if object.data.shape_keys ==None:
        sk_basis = object.shape_key_add(name='Basis')
        sk_basis.interpolation = 'KEY_LINEAR'
        object.data.shape_keys.use_relative = True


    sk = object.shape_key_add(name='Stereo')
    sk.interpolation = 'KEY_LINEAR'

    # position each vert
    for i in range(len(verts)):
        x,y,z=sk.data[i].co[:]
        r2 = x*x + y*y
        nv = 1./(r2 + 1) * Vector((2*x, 2*y, (r2 - 1)))+Vector((0,0,1))
        sk.data[i].co = nv

        
def apply_real_function_to(object,function):        
    verts = object.data.vertices

    sk_basis = object.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    object.data.shape_keys.use_relative = True

    sk = object.shape_key_add(name='Sine')
    sk.interpolation = 'KEY_LINEAR'

    # position each vert
    for i in range(len(verts)):
        x,y,z=sk.data[i].co[:]
        delta = function(x)
        nv = Vector((x, y,z+delta))
        sk.data[i].co = nv
        
def apply_complex_function_to(object,function):        
    verts = object.data.vertices
    
    if object.data.shape_keys==None:
        sk_basis = object.shape_key_add(name='Basis')
        sk_basis.interpolation = 'KEY_LINEAR'
        object.data.shape_keys.use_relative = True


    sk = object.shape_key_add(name='Complex')
    sk.interpolation = 'KEY_LINEAR'

    # position each vert
    for i in range(len(verts)):
        x,y,z=sk.data[i].co[:]
        delta = function(x+1j*y)
        nv = Vector((x, y,z+delta))
        sk.data[i].co = nv

#sin_fnc
def sin_fnc(t,offset: float = 0.0):
    return t*t*t-1

#sin_fnc
def abs_sin_fnc(z,offset: float = 0.0):
    return np.abs(z*z*z-1)

 
def custom_cylinder(length,pos,r,sub_divs,depth,principle_direction):
    bpy.ops.mesh.primitive_cylinder_add(vertices=depth,radius=r,depth=length)
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


def custom_plane(x_min,x_max,y_min,y_max,cx,cy,res):
    bpy.ops.mesh.primitive_plane_add()  # Adds in a plane to work with
    plane = bpy.context.active_object
    data = plane.data  # Selects the plane's data
    bm = bmesh.new()   # Creates an empty BMesh
    bm.from_mesh(data)   # Fills it in using the plane 

    plane_size = [x_max-x_min, y_max-y_min, 0.1]  # Takes user inputs

    location = [cx, cy, 0]  # later there has to be an adjustment for unsymmetric ranges in x and y
    bmesh.ops.subdivide_edges(bm,edges=bm.edges,cuts=res,use_grid_fill=True)
    bmesh.ops.scale(bm,vec=plane_size,verts=bm.verts)
    bmesh.ops.translate(bm,vec=location,verts=bm.verts)

    bm.to_mesh(data)  # Freeing the BMesh, moving on to coloring the domain
    bm.free()
    
    # Setting to object mode:
    bpy.ops.object.mode_set(mode='OBJECT')
    return plane
    
#Camera 
bpy.ops.curve.primitive_bezier_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
camera_circle = bpy.context.active_object
camera_circle.rotation_euler[1]=3.14159/5
camera_circle.scale=(10,10,10)

bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
camera_empty = bpy.context.active_object

#sun
bpy.ops.object.light_add(type='SUN', radius=10, align='WORLD', location=(50, -50, 50), scale=(1, 1, 1))
sun = bpy.context.active_object
ttc = sun.constraints.new(type='TRACK_TO')
ttc.target = camera_empty

#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
 

#Camera0
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0,0), scale=(1, 1, 1))
camera = bpy.context.active_object

ttf = camera.constraints.new(type='FOLLOW_PATH')
ttf.offset_factor=0.1
ttf.use_fixed_location=True
ttf.use_curve_follow=True
ttf.influence=1
ttf.target= camera_circle
ttc = camera.constraints.new(type='TRACK_TO')
ttc.target = camera_empty


dim_x = 12
dim_y = 12

r_axes = 0.02
r_coords = 0.01

zero_line = None

#gridlines x  
xlines = [] 
x_parallels = []
line0_material = get_line_material()
linex_material = get_line_material()
liney_material = get_line_material()

for i in range(-dim_y,dim_y+1,1):
    y = i/4
    if y==0:
        r=r_axes
        depth= 20
    else:
        r=r_coords
        depth= 10
    cylinder = custom_cylinder(6,Vector((0,y,0)),r,100,depth,0)
    xlines.append(cylinder)
    if y==0:
        zero_line = cylinder
        apply_real_function_to(cylinder,sin_fnc)
        apply_complex_function_to(cylinder,abs_sin_fnc)
        cylinder.data.materials.append(line0_material)
    else:
        apply_complex_function_to(cylinder,abs_sin_fnc)
        cylinder.data.materials.append(linex_material)
        x_parallels.append(cylinder)
   
#gridlines y 
ylines = []
for x in range(-int(dim_x),int(dim_x)+1):
    if x==0:
        r=r_axes
        depth = 20
    else:
        r=r_coords
        depth = 10
    cylinder = custom_cylinder(6,Vector((x/4,0,0)),r,100,depth,1)
    cylinder = bpy.context.active_object
    apply_complex_function_to(cylinder,abs_sin_fnc)
    cylinder.data.materials.append(liney_material)
    ylines.append(cylinder)  
   
plane = custom_plane(-1.5,1.5,-1.5,1.5,0,0,200)
bpy.ops.object.modifier_add(type='SUBSURF')
plane_material = get_line_material(0.45)
plane.data.materials.append(plane_material)
apply_complex_function_to(plane,abs_sin_fnc)


plane2 = custom_plane(-3,3,-3,3,0,0,10)
plane2.location[2]=0.01
plane2.data.materials.append(get_mirror_material(0.75))

apply_stereo_proj_to(plane)
for line in xlines:
    apply_stereo_proj_to(line)
for line in ylines:
    apply_stereo_proj_to(line)
 

#######################################
######################Animations
######################################################


# set first and last frame index
total_time = 20 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(total_time*fps)+1

alpha_val=plane_material.node_tree.nodes["Principled BSDF"].inputs['Alpha']
pause = 15

# grow xlines 
offset = 1
bpy.context.scene.frame_set(offset)
alpha_val.default_value = 0
alpha_val.keyframe_insert('default_value')

camera.data.lens = 40
camera.data.keyframe_insert("lens")
ttf.offset_factor = 0.1
ttf.keyframe_insert("offset_factor")


for line in xlines:
    line.scale[0]=0
    line.keyframe_insert("scale")  
    
offset = offset+60
bpy.context.scene.frame_set(offset)
for line in xlines:
    line.scale[0]=1
    line.keyframe_insert("scale")
    
# grow ylines   
for line in ylines:
    line.scale[1]=0
    line.keyframe_insert("scale")
    
offset = offset+30
bpy.context.scene.frame_set(offset)
for line in ylines:
    line.scale[1]=1
    line.keyframe_insert("scale")
    
plane2.keyframe_insert("location")
    
    
# grow function on real axis
offset=offset+pause
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.1
ttf.keyframe_insert("offset_factor")
mix_val=line0_material.node_tree.nodes["Mix"].inputs[0]
mix_val2=line0_material.node_tree.nodes["Mix.001"].inputs[0]
mix_val.default_value = 0
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 0
mix_val2.keyframe_insert("default_value")
shape = zero_line.data.shape_keys.key_blocks["Sine"]
shape.value = 0
shape.keyframe_insert("value")

offset=offset+30
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.0
ttf.keyframe_insert("offset_factor")
mix_val.default_value = 1
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 1
mix_val2.keyframe_insert("default_value")
shape.value = 1
shape.keyframe_insert("value")
plane2.location[2]=-0.001
plane2.keyframe_insert("location")

# turn zero_line to absolute values
offset = offset+pause
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.0
ttf.keyframe_insert("offset_factor")
shape = zero_line.data.shape_keys.key_blocks["Sine"]
shape.value=1
shape.keyframe_insert("value")
shape2 = zero_line.data.shape_keys.key_blocks["Complex"]
shape2.value = 0
shape2.keyframe_insert("value")

offset = offset+30
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.1
ttf.keyframe_insert("offset_factor")
shape = zero_line.data.shape_keys.key_blocks["Sine"]
shape.value=0
shape.keyframe_insert("value")
shape2.value = 1
shape2.keyframe_insert("value")


# grow function parallel to real axis
offset=offset+pause
bpy.context.scene.frame_set(offset)
alpha_val.default_value = 0
alpha_val.keyframe_insert('default_value')
camera_empty.location[2]=0
camera_empty.keyframe_insert("location")
mix_val=linex_material.node_tree.nodes["Mix"].inputs[0]
mix_val2=linex_material.node_tree.nodes["Mix.001"].inputs[0]
mix_val.default_value = 0
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 0
mix_val2.keyframe_insert("default_value")
for parallel in x_parallels:
    p_shape = parallel.data.shape_keys.key_blocks["Complex"]
    p_shape.value = 0
    p_shape.keyframe_insert("value")
   

offset=offset+30
bpy.context.scene.frame_set(offset)
alpha_val.default_value = 0.4
alpha_val.keyframe_insert('default_value')
camera_empty.location[2]=1
camera_empty.keyframe_insert("location")
mix_val.default_value = 1
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 1
mix_val2.keyframe_insert("default_value")
for parallel in x_parallels:
    p_shape = parallel.data.shape_keys.key_blocks["Complex"]
    p_shape.value = 1
    p_shape.keyframe_insert("value")
   
   
# grow function orthogonal to real axis
offset=offset+pause
bpy.context.scene.frame_set(offset)
mix_val=liney_material.node_tree.nodes["Mix"].inputs[0]
mix_val2=liney_material.node_tree.nodes["Mix.001"].inputs[0]
mix_val.default_value = 0
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 0
mix_val2.keyframe_insert("default_value")
for line in ylines:
    p_shape = line.data.shape_keys.key_blocks["Complex"]
    p_shape.value = 0
    p_shape.keyframe_insert("value")
   

offset=offset+30
bpy.context.scene.frame_set(offset)
mix_val.default_value = 1
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 1
mix_val2.keyframe_insert("default_value")
for line in ylines:
    p_shape = line.data.shape_keys.key_blocks["Complex"]
    p_shape.value = 1
    p_shape.keyframe_insert("value")
   
#recolor plane
offset=offset+pause
bpy.context.scene.frame_set(offset)
mix_val=plane_material.node_tree.nodes["Mix"].inputs[0]
mix_val2=plane_material.node_tree.nodes["Mix.001"].inputs[0]
mix_val.default_value = 0
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 0
mix_val2.keyframe_insert("default_value")

offset=offset+30
bpy.context.scene.frame_set(offset)
mix_val.default_value = 1
mix_val.keyframe_insert("default_value")
mix_val2.default_value = 1
mix_val2.keyframe_insert("default_value")
for line in ylines:
    p_shape = line.data.shape_keys.key_blocks["Complex"]
    p_shape.value = 1
    p_shape.keyframe_insert("value")
    
#displace plane
offset=offset+pause
bpy.context.scene.frame_set(offset)
shape = plane.data.shape_keys.key_blocks["Complex"]
shape.value=0
shape.keyframe_insert("value")


offset=offset+30
bpy.context.scene.frame_set(offset)
shape.value=1
shape.keyframe_insert("value")

#convert to sphere
offset=offset+pause
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.1
ttf.keyframe_insert("offset_factor")
pshape = plane.data.shape_keys.key_blocks["Stereo"]
pshape2 = plane.data.shape_keys.key_blocks["Complex"]
mix_vals = [plane_material.node_tree.nodes["Mix.002"].inputs[0],plane_material.node_tree.nodes["Mix.003"].inputs[0],
line0_material.node_tree.nodes["Mix.002"].inputs[0],line0_material.node_tree.nodes["Mix.003"].inputs[0],
linex_material.node_tree.nodes["Mix.002"].inputs[0],linex_material.node_tree.nodes["Mix.003"].inputs[0],
liney_material.node_tree.nodes["Mix.002"].inputs[0],liney_material.node_tree.nodes["Mix.003"].inputs[0]
]
for val in mix_vals:
    val.default_value= 0
    val.keyframe_insert("default_value")
pshape.value=0
pshape2.value = 1
pshape.keyframe_insert("value")
pshape2.keyframe_insert("value")
for line in xlines:
    shape = line.data.shape_keys.key_blocks["Stereo"]
    shape2 = line.data.shape_keys.key_blocks["Complex"]
    shape.value = 0
    shape2.value = 1
    shape.keyframe_insert("value")
    shape2.keyframe_insert("value")
for line in ylines:
    shape = line.data.shape_keys.key_blocks["Stereo"]
    shape2 = line.data.shape_keys.key_blocks["Complex"]
    shape.value = 0
    shape2.value = 1
    shape.keyframe_insert("value")
    shape2.keyframe_insert("value")

offset=offset+90
bpy.context.scene.frame_set(offset)
for val in mix_vals:
    val.default_value= 1
    val.keyframe_insert("default_value")
pshape.value=1
pshape2.value=0
pshape.keyframe_insert("value")
pshape2.keyframe_insert("value")
for line in xlines:
    shape = line.data.shape_keys.key_blocks["Stereo"]
    shape2 = line.data.shape_keys.key_blocks["Complex"]
    shape.value = 1
    shape2.value = 0
    shape.keyframe_insert("value")
    shape2.keyframe_insert("value")
for line in ylines:
    shape = line.data.shape_keys.key_blocks["Stereo"]
    shape2 = line.data.shape_keys.key_blocks["Complex"]
    shape.value = 1
    shape2.value = 0
    shape.keyframe_insert("value")
    shape2.keyframe_insert("value")


#make sphere intransparent
offset = offset+pause
bpy.context.scene.frame_set(offset)
camera.data.lens = 40
camera.data.keyframe_insert("lens")
alpha_val.default_value = 0.5
alpha_val.keyframe_insert("default_value")

offset = offset+30
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.4
ttf.keyframe_insert("offset_factor")
alpha_val.default_value = 0.9
alpha_val.keyframe_insert("default_value")

offset = offset+60
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.25
ttf.keyframe_insert("offset_factor")
alpha_val.default_value = 1
alpha_val.keyframe_insert("default_value")
camera.data.lens = 70
camera.data.keyframe_insert("lens")

print(offset)

