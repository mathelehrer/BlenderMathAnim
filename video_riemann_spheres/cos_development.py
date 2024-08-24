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
    if fnmatch.fnmatch(ob.name,"camera_path*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Latex**"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Parametric*"):
        ob.select_set(True)
    bpy.ops.object.delete()



def derive_bezier_handles(a, b, c, d, tb, tc):
    """
    Derives bezier handles by using the start and end of the curve with 2 intermediate
    points to use for interpolation.
    :param a:
        The start point.
    :param b:
        The first mid-point, located at `tb` on the bezier segment, where 0 < `tb` < 1.
    :param c:
        The second mid-point, located at `tc` on the bezier segment, where 0 < `tc` < 1.
    :param d:
        The end point.
    :param tb:
        The position of the first point in the bezier segment.
    :param tc:
        The position of the second point in the bezier segment.
    :return:
        A tuple of the two intermediate handles, that is, the right handle of the start point
        and the left handle of the end point.
    """

    # Calculate matrix coefficients
    matrix_a = 3 * math.pow(1 - tb, 2) * tb
    matrix_b = 3 * (1 - tb) * math.pow(tb, 2)
    matrix_c = 3 * math.pow(1 - tc, 2) * tc
    matrix_d = 3 * (1 - tc) * math.pow(tc, 2)

    # Calculate the matrix determinant
    matrix_determinant = 1 / ((matrix_a * matrix_d) - (matrix_b * matrix_c))

    # Calculate the components of the target position vector
    final_b = b - (math.pow(1 - tb, 3) * a) - (math.pow(tb, 3) * d)
    final_c = c - (math.pow(1 - tc, 3) * a) - (math.pow(tc, 3) * d)

    # Multiply the inversed matrix with the position vector to get the handle points
    bezier_b = matrix_determinant * ((matrix_d * final_b) + (-matrix_b * final_c))
    bezier_c = matrix_determinant * ((-matrix_c * final_b) + (matrix_a * final_c))

    # Return the handle points
    return (bezier_b, bezier_c)


def create_parametric_curve(
        function,
        *args,
        min: float = 0.0,
        max: float = 1.0,
        use_cubic: bool = True,
        iterations: int = 8,
        resolution_u: int = 10,
        **kwargs
    ):
    """
    Creates a Blender bezier curve object from a parametric function.
    This "plots" the function in 3D space from `min <= t <= max`.
    :param function:
        The function to plot as a Blender curve.
        This function should take in a float value of `t` and return a 3-item tuple or list
        of the X, Y and Z coordinates at that point:
        `function(t) -> (x, y, z)`
        `t` is plotted according to `min <= t <= max`, but if `use_cubic` is enabled, this function
        needs to be able to take values less than `min` and greater than `max`.
    :param *args:
        Additional positional arguments to be passed to the plotting function.
        These are not required.
    :param use_cubic:
        Whether or not to calculate the cubic bezier handles as to create smoother splines.
        Turning this off reduces calculation time and memory usage, but produces more jagged
        splines with sharp edges.
    :param iterations:
        The 'subdivisions' of the parametric to plot.
        Setting this higher produces more accurate curves but increases calculation time and
        memory usage.
    :param resolution_u:
        The preview surface resolution in the U direction of the bezier curve.
        Setting this to a higher value produces smoother curves in rendering, and increases the
        number of vertices the curve will get if converted into a mesh (e.g. for edge looping)
    :param **kwargs:
        Additional keyword arguments to be passed to the plotting function.
        These are not required.
    :return:
        The new Blender object.
    """

    # Create the Curve to populate with points.
    curve = bpy.data.curves.new('Parametric', type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2

    # Add a new spline and give it the appropriate amount of points
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(iterations)

    if use_cubic:
        points = [
            function(((i - 3) / (3 * iterations)) * (max - min) + min, *args, **kwargs) for i in range((3 * (iterations + 2)) + 1)
        ]

        # Convert intermediate points into handles
        for i in range(iterations + 2):
            a = points[(3 * i)]
            b = points[(3 * i) + 1]
            c = points[(3 * i) + 2]
            d = points[(3 * i) + 3]

            bezier_bx, bezier_cx = derive_bezier_handles(a[0], b[0], c[0], d[0], 1 / 3, 2 / 3)
            bezier_by, bezier_cy = derive_bezier_handles(a[1], b[1], c[1], d[1], 1 / 3, 2 / 3)
            bezier_bz, bezier_cz = derive_bezier_handles(a[2], b[2], c[2], d[2], 1 / 3, 2 / 3)

            points[(3 * i) + 1] = (bezier_bx, bezier_by, bezier_bz)
            points[(3 * i) + 2] = (bezier_cx, bezier_cy, bezier_cz)

        # Set point coordinates and handles
        for i in range(iterations + 1):
            spline.bezier_points[i].co = points[3 * (i + 1)]

            spline.bezier_points[i].handle_left_type = 'FREE'
            spline.bezier_points[i].handle_left = Vector(points[(3 * (i + 1)) - 1])

            spline.bezier_points[i].handle_right_type = 'FREE'
            spline.bezier_points[i].handle_right = Vector(points[(3 * (i + 1)) + 1])

    else:
        points = [function(i / iterations, *args, **kwargs) for i in range(iterations + 1)]

        # Set point coordinates, disable handles
        for i in range(iterations + 1):
            spline.bezier_points[i].co = points[i]
            spline.bezier_points[i].handle_left_type = 'VECTOR'
            spline.bezier_points[i].handle_right_type = 'VECTOR'

    # Create the Blender object and link it to the scene
    curve_object = bpy.data.objects.new('Parametric', curve)
    link_object(curve_object)

    # Return the new object
    return curve_object


def make_edge_loops(*objects):
    """
    Turns a set of Curve objects into meshes, creates vertex groups,
    and merges them into a set of edge loops.
    :param *objects:
        Positional arguments for each object to be converted and merged.
    """

    mesh_objects = []
    vertex_groups = []

    # Convert all curves to meshes
    for obj in objects:
        # Unlink old object
        unlink_object(obj)

        # Convert curve to a mesh
        if bpy.app.version >= (2, 80):
            new_mesh = obj.to_mesh().copy()
        else:
            new_mesh = obj.to_mesh(scene, False, 'PREVIEW')

        # Store name and matrix, then fully delete the old object
        name = obj.name
        matrix = obj.matrix_world
        bpy.data.objects.remove(obj)

        # Attach the new mesh to a new object with the old name
        new_object = bpy.data.objects.new(name, new_mesh)
        new_object.matrix_world = matrix

        # Make a new vertex group from all vertices on this mesh
        vertex_group = new_object.vertex_groups.new(name=name)
        vertex_group.add(range(len(new_mesh.vertices)), 1.0, 'ADD')

        vertex_groups.append(vertex_group)

        # Link our new object
        link_object(new_object)

        # Add it to our list
        mesh_objects.append(new_object)

    # Make a new context
    ctx = context.copy()

    # Select our objects in the context
    ctx['active_object'] = mesh_objects[0]
    ctx['selected_objects'] = mesh_objects
    if bpy.app.version >= (2, 80):
        ctx['selected_editable_objects'] = mesh_objects
    else:
        ctx['selected_editable_bases'] = [scene.object_bases[o.name] for o in mesh_objects]

  
dim = 40
ydim = 100

r_axes = 0.1
r_coords = 0.025
r_sub = 0.01
r_curve = 0.064

x_color = Vector((1.,0.5,0.5,1))
y_color = Vector((0.5,1.,0.5,1))
z_color = Vector((0.5,0.5,1,1))


def get_line_material():
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links =mat.node_tree.links

    output = nodes.get("Material Output")
    output.location = Vector((800,200))
    
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=0.1  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[4].default_value=1 #metalicity
    bsdf.inputs[7].default_value=0.1 #roughness
    links.new(bsdf.outputs[0],output.inputs[0])
    
    hue = nodes.new(type='ShaderNodeHueSaturation')
    hue.location = Vector((100,300))
    hue.inputs[1].default_value=1
    hue.inputs[2].default_value=1
    hue.inputs[3].default_value=1
    hue.inputs[4].default_value=(1.0,0,0.,1.)
    links.new(hue.outputs[0],bsdf.inputs[0])
    links.new(hue.outputs[0],bsdf.inputs[17])
    
    m1 = nodes.new(type='ShaderNodeMath')
    m1.operation='ADD'
    m1.location=Vector((-100,300))
    m1.inputs[1].default_value = 0.5
    links.new(m1.outputs[0],hue.inputs[0])
    
    m2 = nodes.new(type='ShaderNodeMath')
    m2.operation='DIVIDE'
    m2.location=Vector((-300,300))
    m2.inputs[1].default_value = 2.*np.pi
    links.new(m2.outputs[0],m1.inputs[0])
    
    m3 = nodes.new(type='ShaderNodeMath')
    m3.operation='ARCTAN2'
    m3.location=Vector((-500,300))
    links.new(m3.outputs[0],m2.inputs[0])
    
    m_real = nodes.new(type='ShaderNodeMath')
    m_real.operation='MULTIPLY'
    m_real.location=Vector((-700,100))
    links.new(m_real.outputs[0],m3.inputs[1])
    
    m_im = nodes.new(type='ShaderNodeMath')
    m_im.operation='MULTIPLY'
    m_im.location=Vector((-700,500))
    links.new(m_im.outputs[0],m3.inputs[0])
    
    m_sin = nodes.new(type='ShaderNodeMath')
    m_sin.operation='SINE'
    m_sin.location=Vector((-900,-100))
    links.new(m_sin.outputs[0],m_real.inputs[0])
    
    m_cos = nodes.new(type='ShaderNodeMath')
    m_cos.operation='COSINE'
    m_cos.location=Vector((-900,100))
    links.new(m_cos.outputs[0],m_im.inputs[0])
    
    m_sinh = nodes.new(type='ShaderNodeMath')
    m_sinh.operation='SINH'
    m_sinh.location=Vector((-900,300))
    links.new(m_sinh.outputs[0],m_im.inputs[1])
    
    m_cosh = nodes.new(type='ShaderNodeMath')
    m_cosh.operation='COSH'
    m_cosh.location=Vector((-900,500))
    links.new(m_cosh.outputs[0],m_real.inputs[1])
   
    sep = nodes.new(type='ShaderNodeSeparateXYZ')
    sep.location = Vector((-1100,300))
    links.new(sep.outputs[0],m_sin.inputs[0]) 
    links.new(sep.outputs[0],m_cos.inputs[0]) 
    links.new(sep.outputs[1],m_sinh.inputs[0]) 
    links.new(sep.outputs[1],m_cosh.inputs[0]) 
    
    input = nodes.new(type='ShaderNodeTexCoord')
    input.location = Vector((-1300,300))
    links.new(input.outputs[3],sep.inputs[0])
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

def get_gridplane_material(color = (0.8,0.8,0.8,1)):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((800,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=0.1  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value =color
    bsdf.inputs[0].default_value=color
    bsdf.inputs[4].default_value=1 #metalicity
    bsdf.inputs[7].default_value=0.1 #roughness
    
    file = "zetafunction_20_55_5000_hsv.png"
    img = bpy.data.images.load("/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/"+file, check_existing=True)
    img_node = nodes.new(type='ShaderNodeTexImage')
    img_node.location=Vector((0,0))
    img_node.image = img
    
    file2 = "zeta_function_20_55_5000_bumpmap.tiff"
    img2 = bpy.data.images.load("/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/"+file2, check_existing=True)
    img2_node = nodes.new(type='ShaderNodeTexImage')
    img2_node.location=Vector((1300,-300))
    img2_node.image = img2
    img2_node.extension='EXTEND'
    
    links =mat.node_tree.links
    m1 = nodes.new(type='ShaderNodeMath')
    m1.operation='MULTIPLY'
    m1.location=Vector((1100,-300))
    m1.inputs[1].default_value = 4
    links.new(img2_node.outputs[0],m1.inputs[0])
    
    m2 = nodes.new(type='ShaderNodeMath')
    m2.operation='ADD'
    m2.location=Vector((900,-300))
    m2.inputs[1].default_value = -2
    links.new(m1.outputs[0],m2.inputs[0])
    
    m3 = nodes.new(type='ShaderNodeMath')
    m3.operation='POWER'
    m3.location=Vector((700,-300))
    m3.inputs[0].default_value = 10
    links.new(m2.outputs[0],m3.inputs[1])
    
    
    disp = nodes.new(type='ShaderNodeDisplacement')
    disp.location=Vector((600,-300))
    disp.inputs[1].default_value = 0
    disp.inputs[2].default_value = 0
    links.new(m3.outputs[0],disp.inputs[0])
    
    bsdf2 = nodes.new("ShaderNodeBsdfPrincipled") 
    bsdf2.inputs[18].default_value=0.1  #increase emission strength
    bsdf2.location = Vector((300,0))
    bsdf2.inputs[4].default_value=0 #metalicity
    bsdf2.inputs[7].default_value=0.5 #roughness
    
    mix_shader =  nodes.new(type='ShaderNodeMixShader')
    mix_shader.location=Vector((600,200))
    mix_shader.inputs[0].default_value = 0
    mix_shader_plane = None
    
    #make plane invisible initially
    
    links.new(img_node.outputs[0],bsdf2.inputs[0])
    links.new(img_node.outputs[0],bsdf2.inputs[17])
    links.new(bsdf.outputs[0],mix_shader.inputs[1])
    links.new(bsdf2.outputs[0],mix_shader.inputs[2])
    links.new(mix_shader.outputs[0],output.inputs[0])
    
    links.new(disp.outputs[0],output.inputs[2])
    return mat

def stereo_proj(object):


    verts = object.data.vertices

    sk_basis = object.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    object.data.shape_keys.use_relative = True


    sk = object.shape_key_add(name='Deform')
    sk.interpolation = 'KEY_LINEAR'

    # position each vert
    for i in range(len(verts)):
        x,y,z=sk.data[i].co[:]
        r2 = x*x + y*y
        nv = 1./(r2 + 1) * Vector((2*x, 2*y, (r2 - 1)))
        sk.data[i] = nv


#sin_fnc
def sin_fnc(t,offset: float = 0.0):
    return [t,0,np.sin(t+offset)]

#sin_fnc
def abs_sin_fnc(z,offset: float = 0.0):
    return [np.real(z),np.imag(z),np.abs(np.sin(z))]


def curve_to_mesh(curve):
    deg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(curve.evaluated_get(deg), depsgraph=deg)

    new_obj = bpy.data.objects.new(curve.name + "_mesh", me)
    bpy.context.collection.objects.link(new_obj)

    for o in bpy.context.selected_objects:
        o.select_set(False)

    new_obj.matrix_world = curve.matrix_world
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj

 
bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
camera_empty = bpy.context.active_object

#sun
bpy.ops.object.light_add(type='SUN', radius=10, align='WORLD', location=(50, -50, 0), scale=(1, 1, 1))
sun = bpy.context.active_object
ttc = sun.constraints.new(type='TRACK_TO')
ttc.target = camera_empty

#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
 



#create path for camera
loc_0=(8,-60,5)
    
coords_list = [[0,-95,0],[0,-80,0],[42,-92,35],[8,-60,5],[8,-30,5], [8,-5,5], [0,-5,5],[-2.5,0,5],[0,5,5], [8,5,5],[8,30,5],[8,60,5],[8,80,5]]

# make a new curve
camera_path = bpy.data.curves.new('crv', 'CURVE')
camera_path.dimensions = '3D'

# make a new spline in that curve
camera_spline = camera_path.splines.new(type='NURBS')

# a spline point for each point
camera_spline.points.add(len(coords_list)-1) # theres already one point by default

# assign the point coordinates to the spline points
for p, new_co in zip(camera_spline.points, coords_list):
    p.co = (new_co + [1.0]) # (add nurbs weight)

# make a new object with the curve
camera_path = bpy.data.objects.new('camera_path', camera_path)
bpy.context.scene.collection.objects.link(camera_path)

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
    cylinder.data.materials.append(get_coordinate_material(y_color))
    xlines.append(cylinder)
    
#gridlines z 
zlines = []
for z in range(-int(dim/2),int(dim/2)+1,10):
    if z==0:
        r=r_axes
    else:
        r=r_coords
    bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=dim)
    cylinder = bpy.context.active_object
    cylinder.location[0]=0
    cylinder.location[2]=z
    cylinder.location[1]=0
    cylinder.rotation_euler[0]=3.14159/2
    cylinder.rotation_euler[2]=3.14159/2
    cylinder.data.materials.append(get_coordinate_material(x_color))
    zlines.append(cylinder)

#gridlines y 
ylines = []
for y in range(-int(dim/2),int(dim/2)+1,10):
    if y==0:
        r=r_axes
    else:
        r=r_coords
    bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=1.1*ydim)
    cylinder = bpy.context.active_object
    cylinder.location[0]=y
    cylinder.location[2]=0
    cylinder.location[1]=0
    cylinder.rotation_euler[0]=3.14159/2
    cylinder.data.materials.append(get_coordinate_material(z_color))
    ylines.append(cylinder)

#gridlines z 
zlines2 = []
for z in range(-int(ydim/2),int(ydim/2)+1,10):
    if z==0:
        r=r_axes
    else:
        r=r_coords
    bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=dim)
    cylinder = bpy.context.active_object
    cylinder.location[0]=0
    cylinder.location[1]=z
    cylinder.location[2]=0
    cylinder.rotation_euler[0]=3.14159/2
    cylinder.rotation_euler[2]=3.14159/2
    cylinder.data.materials.append(get_coordinate_material(x_color))
    zlines2.append(cylinder)


#subgridlines x  
x_sub_lines = [] 
for x in range(0,11):
    r=r_sub
    bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=10)
    cylinder = bpy.context.active_object
    cylinder.location[0]=x
    cylinder.location[1]=0
    cylinder.location[2]=5
    cylinder.scale[2]=0
    #cylinder.rotation_euler[0]=3.14159/2
    cylinder.data.materials.append(get_coordinate_material(y_color))
    x_sub_lines.append(cylinder)
    
#subgridlines y 
z_sub_lines = []
for z in range(0,11):
    r=r_sub
    bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=10)
    cylinder = bpy.context.active_object
    cylinder.location[0]=5
    cylinder.location[2]=z
    cylinder.location[1]=0
    cylinder.scale[2]=0
    cylinder.rotation_euler[0]=3.14159/2
    cylinder.rotation_euler[2]=3.14159/2
    cylinder.data.materials.append(get_coordinate_material(x_color))
    z_sub_lines.append(cylinder)

r=r_axes

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
zarrow = bpy.context.active_object
zarrow.rotation_euler[1]=3.14159/2
zarrow.scale[0]=r_axes/0.25
zarrow.scale[1]=r_axes/0.25
zarrow.scale[2]=10*r_axes/0.25
zarrow.location[2]=0
zarrow.location[1]=0
zarrow.location[0]=1.1*dim/2
zarrow.data.materials.append(get_coordinate_material(x_color))

bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))",z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
bpy.ops.object.editmode_toggle()
yarrow = bpy.context.active_object
yarrow.rotation_euler[0]=-3.14159/2
yarrow.scale[0]=r_axes/0.25
yarrow.scale[1]=r_axes/0.25
yarrow.scale[2]=10*r_axes/0.25
yarrow.location[2]=0
yarrow.location[0]=0
yarrow.location[1]=1.1*dim/2
yarrow.data.materials.append(get_coordinate_material(z_color))

r_curves = 0.05

#sin curves 

to_be_projected = []

curves_right = []
curves_left = []
curve_mat = get_line_material()
for i in range(0,5):
    curve = create_parametric_curve(sin_fnc,offset=0.0,min=i*np.pi,max=(i+1)*np.pi,use_cubic=True,iterations=20)
    curve.data.bevel_depth = r_curves
    curve.data.bevel_resolution=1
    curve.data.bevel_factor_start = 0
    curve.data.bevel_factor_end = 0
    curve.data.materials.append(curve_mat)
    curve_to_mesh(curve)
    to_be_projected.append(curve)
    curves_right.append(curve)
    curve = create_parametric_curve(sin_fnc,offset=0.0,min=-i*np.pi,max=-(i+1)*np.pi,use_cubic=True,iterations=20)
    curve.data.bevel_depth = r_curves
    curve.data.bevel_factor_start = 0
    curve.data.bevel_factor_end = 0
    curve.data.materials.append(curve_mat)
    bpy.ops.object.convert(target='MESH')
    curves_left.append(curve)
    to_be_projected.append(curve)

more_curves = []
for i in range(1,16):
    curve = create_parametric_curve(abs_sin_fnc,offset=0,min=-5*np.pi+3j*i/15,max =5*np.pi+3j*i/15,use_cubic=True,iterations = 200)
    curve.data.bevel_depth = r_curves
    curve.data.bevel_resolution=1
    curve.data.bevel_factor_start = 0
    curve.data.bevel_factor_end = 0
    curve.data.materials.append(curve_mat)
    bpy.ops.object.convert(target='MESH')
    more_curves.append(curve)
    to_be_projected.append(curve)
for i in range(1,16):
    curve = create_parametric_curve(abs_sin_fnc,offset=0,min=-5*np.pi-3j*i/15,max =5*np.pi-3j*i/15,use_cubic=True,iterations = 200)
    curve.data.bevel_depth = r_curves
    curve.data.bevel_resolution=1
    curve.data.bevel_factor_start = 0
    curve.data.bevel_factor_end = 0
    curve.data.materials.append(curve_mat)
    bpy.ops.object.convert(target='MESH')
    more_curves.append(curve)
    to_be_projected.append(curve)
   
more_curves2 = []
for i in range(-50,50):
    curve = create_parametric_curve(abs_sin_fnc,offset=0,min=i*np.pi/10-3j,max =i*np.pi/10+3j,use_cubic=True,iterations = 200)
    curve.data.bevel_depth = r_curves
    curve.data.bevel_resolution=1
    curve.data.bevel_factor_start = 0
    curve.data.bevel_factor_end = 0
    curve.data.materials.append(curve_mat)
    more_curves2.append(curve)
    to_be_projected.append(curve)

for obj in bpy.context.scene.objects:
    if obj and obj.type == 'CURVE':
        curve_to_mesh(obj)
#stereographic projection
#for curve in to_be_projected:
    #stereo_proj(curve)


#######################################
######################Animations
######################################################


# set first and last frame index
total_time = 90 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(total_time*fps)+1
pause = 15

#grow xlines in the first second
offset = 0
bpy.context.scene.frame_set(offset+1)
xarrow.location[2]=0
xarrow.scale[2]=0
xarrow.keyframe_insert("location")
xarrow.keyframe_insert("scale")
zarrow.location[0]=0
zarrow.scale[2]=0
zarrow.keyframe_insert("location")
zarrow.keyframe_insert("scale")
yarrow.location[1]=0
yarrow.scale[2]=0
yarrow.scale[1]=0
yarrow.scale[0]=0
yarrow.keyframe_insert("location")
yarrow.keyframe_insert("scale")
for line in xlines:
    line.scale[2]=0
    line.keyframe_insert("scale")  
bpy.context.scene.frame_set(offset+30)
for line in xlines:
    line.scale[2]=1
    line.keyframe_insert("scale")
xarrow.location[2]=1.1*dim/2
xarrow.keyframe_insert("location")
xarrow.scale[2]=10*r_axes/0.25
xarrow.keyframe_insert("scale")    
ttf.offset_factor=0
ttf.keyframe_insert("offset_factor")

    
#grow zlines in the second second
offset = offset+30
#30
bpy.context.scene.frame_set(offset+1)
zarrow.location[0]=0
zarrow.keyframe_insert("location")
zarrow.scale[2]=0
zarrow.keyframe_insert("scale")
for line in zlines:
    line.scale[2]=0
    line.keyframe_insert("scale")
    
bpy.context.scene.frame_set(offset+30)
#60
for line in zlines:
    line.scale[2]=1
    line.keyframe_insert("scale")
zarrow.location[0]=1.1*dim/2
zarrow.keyframe_insert("location")
zarrow.scale[2]=10*r_axes/0.25
zarrow.keyframe_insert("scale")

#zoom in

offset=offset+30
#90
bpy.context.scene.frame_set(offset+1)
camera.data.keyframe_insert("lens")

for line in x_sub_lines:
    line.scale[2]=0
    line.keyframe_insert("scale")
    
for line in z_sub_lines:
    line.scale[2]=0
    line.keyframe_insert("scale")
    
bpy.context.scene.frame_set(offset+30)
#120
camera.data.lens = 130
camera.data.keyframe_insert("lens")

for line in x_sub_lines:
    line.scale[2]=1
    line.keyframe_insert("scale")
    
for line in z_sub_lines:
    line.scale[2]=1
    line.keyframe_insert("scale")

#grow function
for i in range(0,len(curves_right)):
    bpy.context.scene.frame_set(offset)
    curves_left[i].data.bevel_factor_end=0
    curves_left[i].data.keyframe_insert(data_path="bevel_factor_end")
    curves_right[i].data.bevel_factor_end=0
    curves_right[i].data.keyframe_insert(data_path="bevel_factor_end")
    offset = offset+15
    bpy.context.scene.frame_set(offset)
    curves_left[i].data.bevel_factor_end=1
    curves_left[i].data.keyframe_insert(data_path="bevel_factor_end")
    curves_right[i].data.bevel_factor_end=1
    curves_right[i].data.keyframe_insert(data_path="bevel_factor_end")

 
#remove axis from the set of lines
remove_zlines=[]
for i,line in enumerate(zlines):
    if i!=int(len(zlines)/2):
        remove_zlines.append(line)
 
remove_xlines=[]
for i,line in enumerate(xlines):
    if i!=int(len(xlines)/2):
        remove_xlines.append(line)
        
 
offset = offset+30
bpy.context.scene.frame_set(offset+1)
camera_empty.keyframe_insert("location")

for line in x_sub_lines:
    line.keyframe_insert("scale")
for line in z_sub_lines:
    line.keyframe_insert("scale")
for line in remove_xlines:
    line.keyframe_insert("scale")
for line in remove_zlines:
    line.keyframe_insert("scale")

offset = offset+30
bpy.context.scene.frame_set(offset)
camera_empty.location[0]=-5
camera_empty.keyframe_insert("location")

for line in x_sub_lines:
    line.scale[2]=0
    line.keyframe_insert("scale")
for line in z_sub_lines:
    line.scale[2]=0
    line.keyframe_insert("scale")
for line in remove_xlines:
    line.scale[0]=0
    line.scale[1]=0
    line.keyframe_insert("scale")
for line in remove_zlines:
    line.scale[0]=0
    line.scale[1]=0
    line.keyframe_insert("scale")

offset=offset+30
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.0
ttf.keyframe_insert("offset_factor")

yarrow.location[1]=0
yarrow.scale[2]=0
yarrow.scale[1]=0
yarrow.scale[0]=0
yarrow.keyframe_insert("location")
yarrow.keyframe_insert("scale")
for line in ylines:
    line.scale[2]=0
    line.keyframe_insert("scale")
    
for line in zlines2:
    line.scale[2]=0
    line.keyframe_insert("scale")

offset=offset+30
bpy.context.scene.frame_set(offset)
for line in zlines2:
    line.scale[2]=1
    line.keyframe_insert("scale")
offset=offset+30
bpy.context.scene.frame_set(offset)
for line in ylines:
    line.scale[2]=1
    line.keyframe_insert("scale")    
yarrow.location[1]=1.05*ydim/2
yarrow.keyframe_insert("location")
yarrow.scale[2]=10*r_axes/0.25
yarrow.scale[1]=r_axes/0.25
yarrow.scale[0]=r_axes/0.25
yarrow.keyframe_insert("scale")

bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0.175
ttf.keyframe_insert("offset_factor")

for curve in curves_left:
    curve.keyframe_insert("rotation_euler")
for curve in curves_right:
    curve.keyframe_insert("rotation_euler")

offset=offset+30
bpy.context.scene.frame_set(offset)
for i in range(0,len(curves_right)):
    if i%2==1:
        curves_right[i].rotation_euler[0]=3.14159
        curves_right[i].keyframe_insert("rotation_euler")
    if i%2==0:
        curves_left[i].rotation_euler[0]=3.14159
        curves_left[i].keyframe_insert("rotation_euler")

offset=offset+pause
#grow more function
for curve in more_curves:
    bpy.context.scene.frame_set(offset)
    curve.data.bevel_factor_end=0
    curve.data.keyframe_insert(data_path="bevel_factor_end")
    
    
offset = offset+30
bpy.context.scene.frame_set(offset)
for curve in more_curves:
    curve.data.bevel_factor_end=1
    curve.data.keyframe_insert(data_path="bevel_factor_end")
    
offset = offset+pause
#grow more function
for curve in more_curves2:
    bpy.context.scene.frame_set(offset)
    curve.data.bevel_factor_end=0
    curve.data.keyframe_insert(data_path="bevel_factor_end")
    
    
offset = offset+30
bpy.context.scene.frame_set(offset)
for curve in more_curves2:
    curve.data.bevel_factor_end=1
    curve.data.keyframe_insert(data_path="bevel_factor_end")

#projection

offset = offset+pause
bpy.context.scene.frame_set(offset)
   

print(offset)

