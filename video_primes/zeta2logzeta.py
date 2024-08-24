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
#if len(bpy.context.scene.objects)>0:
#    for ob in bpy.context.scene.objects:
#        if fnmatch.fnmatch(ob.name,"XYZ Function*"):
#            ob.select_set(True)
#        if fnmatch.fnmatch(ob.name,"Cylinder*"):
#            ob.select_set(True)
#        if fnmatch.fnmatch(ob.name,"Plane*"):
#            ob.select_set(True)
#        if fnmatch.fnmatch(ob.name,"coordinate_system*"):
#            ob.select_set(True)
#        if fnmatch.fnmatch(ob.name,"Empty*"):
#            ob.select_set(True)
#        bpy.ops.object.delete()

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

    # Join them together
    bpy.ops.object.join(ctx)


def zeta_fcn(t, offset: float = 0.0):
    """
    converts the real data loaded from file between -50.01 and 49.99 in 0.02 steps to avoid the singularity at one
    """

    index = int(np.round((t+50.01)/0.02))
    return (
        zeta_real[index][0],
        0,
        zeta_real[index][1]
    )

def zeta_one_half_fcn(t, offset: float = 0.0):
    """
    converts the abs value data loaded from file between -50 and 50 in 0.02 steps 
    along the line re z = 1/2
    
    """
    index = int(np.round((t+50)/0.02))
    return (
        0.5,
        zeta_one_half[index][0],
        zeta_one_half[index][1]
    )
    
   

zetas=[1.64493, 1.20206, 1.08232, 1.03693, 1.01734, 1.00835, 1.00408,1.00201, 1.00099]
zeta_real = []
zeta_one_half = []

path = "/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/zeta_data_real.csv"  # a blender relative path
f = Path(bpy.path.abspath(path)) 

if f.exists():
    text  = f.read_text()
    lines = text.split('\n')
    for line in lines:
        data_raw = line.split(',')
        if len(data_raw)==2:
            data = [float(data_raw[0]),float(data_raw[1])]
            zeta_real.append(data)
else:
    print("No file found")
    
path = "/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/zeta_one_half_50_0.02.csv"  # a blender relative path
f = Path(bpy.path.abspath(path)) 

if f.exists():
    text  = f.read_text()
    lines = text.split('\n')
    for line in lines:
        data_raw = line.split(',')
        if len(data_raw)==2:
            data = [float(data_raw[0]),float(data_raw[1])]
            zeta_one_half.append(data)
else:
    print("No file found")
    
zeros =[]
path = "/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/zeros.dat"  # a blender relative path
f = Path(bpy.path.abspath(path)) # make a path object of abs path

if f.exists():
    text  = f.read_text()
    lines = text.split('\n')
    for line in lines:
        zeros.append(float(line))
else:
    print("No file found")

dim = 40
ydim = 100

r_axes = 0.1
r_coords = 0.025
r_sub = 0.01
r_curve = 0.064

x_color = Vector((1.,0.5,0.5,1))
y_color = Vector((0.5,1.,0.5,1))
z_color = Vector((0.5,0.5,1,1))


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
    img_node.extension='EXTEND'
    
    file2 = "zeta_function_20_55_5000_bumpmap.tiff"
    img2 = bpy.data.images.load("/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/"+file2, check_existing=True)
    img2_node = nodes.new(type='ShaderNodeTexImage')
    img2_node.location=Vector((-1300,-300))
    img2_node.image = img2
    img2_node.extension='EXTEND'
    
    
    file3 = "logzeta_20_55_5000_hsv.png"
    img3 = bpy.data.images.load("/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/"+file3, check_existing=True)
    img3_node = nodes.new(type='ShaderNodeTexImage')
    img3_node.location=Vector((0,-300))
    img3_node.image = img3
    img3_node.extension='EXTEND'
    
    file4 = "logzeta_20_55_5000_bumpmap.png"
    img4 = bpy.data.images.load("/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/"+file4, check_existing=True)
    img4_node = nodes.new(type='ShaderNodeTexImage')
    img4_node.location=Vector((-1300,-600))
    img4_node.image = img4
    img4_node.extension='EXTEND'
    
    
    links =mat.node_tree.links
    mix1_node = nodes.new(type="ShaderNodeMixRGB")
    mix1_node.location=Vector((100,-150))
    links.new(img_node.outputs[0],mix1_node.inputs[1])
    links.new(img3_node.outputs[0],mix1_node.inputs[2])
    
    m1 = nodes.new(type='ShaderNodeMath')
    m1.operation='MULTIPLY'
    m1.location=Vector((-900,-450))
    m1.inputs[1].default_value = 4
    links.new(img2_node.outputs[0],m1.inputs[0])
    
    m2 = nodes.new(type='ShaderNodeMath')
    m2.operation='ADD'
    m2.location=Vector((-700,-450))
    m2.inputs[1].default_value = -2
    links.new(m1.outputs[0],m2.inputs[0])
    
    m3 = nodes.new(type='ShaderNodeMath')
    m3.operation='POWER'
    m3.location=Vector((-500,-450))
    m3.inputs[0].default_value = 10
    links.new(m2.outputs[0],m3.inputs[1])
    
    m4 = nodes.new(type='ShaderNodeMath')
    m4.operation='MULTIPLY'
    m4.location=Vector((-500,-450))
    m4.inputs[1].default_value = 1
    links.new(m3.outputs[0],m4.inputs[0])
    
    n1 = nodes.new(type='ShaderNodeMath')
    n1.operation='MULTIPLY'
    n1.location=Vector((-900,-650))
    n1.inputs[1].default_value = 1
    links.new(img4_node.outputs[0],n1.inputs[0])
    
    n2 = nodes.new(type='ShaderNodeMath')
    n2.operation='MULTIPLY'
    n2.location=Vector((-700,-650))
    n2.inputs[1].default_value = 0
    links.new(n1.outputs[0],n2.inputs[0])
    
    max = nodes.new(type='ShaderNodeMath')
    max.operation='MAXIMUM'
    max.location=Vector((-500,-650))
    links.new(m4.outputs[0],max.inputs[0])
    links.new(n2.outputs[0],max.inputs[1])
    
    
    disp = nodes.new(type='ShaderNodeDisplacement')
    disp.location=Vector((-300,-450))
    disp.inputs[1].default_value = 0
    disp.inputs[2].default_value = 0
    links.new(max.outputs[0],disp.inputs[0])
    
    bsdf2 = nodes.new("ShaderNodeBsdfPrincipled") 
    bsdf2.inputs[18].default_value=0.1  #increase emission strength
    bsdf2.location = Vector((0,-450))
    bsdf2.inputs[4].default_value=0 #metalicity
    bsdf2.inputs[7].default_value=0.5 #roughness
    
    mix_shader =  nodes.new(type='ShaderNodeMixShader')
    mix_shader.location=Vector((300,-200))
    mix_shader.inputs[0].default_value = 0
    mix_shader_plane = None
    
    #make plane invisible initially
    
    links.new(mix1_node.outputs[0],bsdf2.inputs[0])
    links.new(mix1_node.outputs[0],bsdf2.inputs[17])
    links.new(bsdf.outputs[0],mix_shader.inputs[1])
    links.new(bsdf2.outputs[0],mix_shader.inputs[2])
    links.new(mix_shader.outputs[0],output.inputs[0])
    
    links.new(disp.outputs[0],output.inputs[2])
    return mat


def get_image_texture_material(file):
    if file != "": 
        mat = bpy.data.materials.new("material")

        img = bpy.data.images.load("/home/jmartin/Dropbox/MyBlender/PrimesAndZeta/"+file, check_existing=True)
         
        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes

        output = nodes.get("Material Output")
        output.location = Vector((600,200))
        bsdf = nodes.get("Principled BSDF") 
        bsdf.inputs[18].default_value=0.1  #increase emission strength
        bsdf.location = Vector((300,300))
        img_node = nodes.new(type='ShaderNodeTexImage')
        img_node.location=Vector((0,0))
        img_node.image = img
        
        input_coord = nodes.new(type='ShaderNodeTexCoord')
        input_coord.location=Vector((-600,0))
        
        sep = nodes.new(type='ShaderNodeSeparateXYZ')
        sep.location = Vector((-400,0))
        
        div = nodes.new(type='ShaderNodeMath')
        div.location = Vector((-200,0))
        div.operation='DIVIDE'
        div.inputs[1].default_value = 49.9
        
        bsdf.inputs[4].default_value=1 #metalicity
        bsdf.inputs[7].default_value=0.1 #roughness
      
       
        links =mat.node_tree.links
        
        links.new(input_coord.outputs[3],sep.inputs[0])
        links.new(sep.outputs[1],div.inputs[0])
        links.new(div.outputs[0],img_node.inputs[0])
        links.new(bsdf.outputs[0],output.inputs[0])
        links.new(img_node.outputs[0],bsdf.inputs[0]) #standard color
        links.new(img_node.outputs[0],bsdf.inputs[17]) #emission
        
        return mat

#remove old labels and arrows
for ob in bpy.context.scene.objects:
    if fnmatch.fnmatch(ob.name, "Latex*"):
        ob.select_set(True)
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
    if fnmatch.fnmatch(ob.name,"Bezier*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Parametric*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"camera_path*"):
        ob.select_set(True)
    bpy.ops.object.delete()



#gridplane
bpy.ops.mesh.primitive_plane_add(size=dim, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
plane = bpy.context.active_object
plane.scale[1]=1.1*ydim/dim
plane_material = get_gridplane_material((0.9,0.9,0.9,1))
#get alpha channel of plane material for hiding the plane initially
grid_alpha = plane_material.node_tree.nodes["Principled BSDF"].inputs[19]
#allow for textural displacements
plane.data.materials.append(plane_material)
bpy.context.object.active_material.cycles.displacement_method = 'DISPLACEMENT'
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
bpy.context.object.cycles.use_adaptive_subdivision = True
bpy.context.object.cycles.dicing_rate=0.5

#sun 
bpy.ops.object.light_add(type='SUN', radius=10, align='WORLD', location=(50, -50, 0), scale=(1, 1, 1))
sun = bpy.context.active_object
ttc = sun.constraints.new(type='TRACK_TO')
ttc.target = plane

#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
 
 
bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
camera_empty = bpy.context.active_object

#create path for camera
loc_0=(8,-60,5)
    
coords_list = [[8,-80,5],[8,-60,5],[8,-30,15], [8,-5,20],[8,0,25],[8,5,20] ,[8,30,15],[8,60,5],[8,80,5]]

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
camera.data.lens =30


ttf = camera.constraints.new(type='FOLLOW_PATH')
ttf.offset_factor=0
ttf.use_fixed_location=True
ttf.use_curve_follow=True
ttf.influence=1
ttf.target= camera_path
ttc = camera.constraints.new(type='TRACK_TO')
ttc.target = camera_empty


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


r=r_axes
bpy.ops.mesh.primitive_cylinder_add(radius=r,depth=ydim)
onehalfline = bpy.context.active_object
onehalfline.location[0]=0.5
onehalfline.location[2]=0
onehalfline.location[1]=0
onehalfline.rotation_euler[0]=3.14159/2
onehalfline.data.materials.append(get_coordinate_material((1.,1.,1.,1.)))


bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))",z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
bpy.ops.object.editmode_toggle()
x_arrow = bpy.context.active_object
x_arrow.rotation_euler[1]=3.14159/2
x_arrow.scale[0]=r_axes/0.25
x_arrow.scale[1]=r_axes/0.25
x_arrow.scale[2]=10*r_axes/0.25
x_arrow.location[2]=0
x_arrow.location[1]=0
x_arrow.location[0]=1.1*dim/2
x_arrow.data.materials.append(get_coordinate_material(x_color))

bpy.ops.mesh.primitive_xyz_function_surface(x_eq="cos(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))",z_eq="v",y_eq="sin(u)*((v>-2)*(v<0)*0.25+1*(v>=0)*(1-v))", range_v_min=-2, range_v_max=1, range_v_step=32)
bpy.ops.object.editmode_toggle()
y_arrow = bpy.context.active_object
y_arrow.rotation_euler[0]=-3.14159/2
y_arrow.scale[0]=r_axes/0.25
y_arrow.scale[1]=r_axes/0.25
y_arrow.scale[2]=10*r_axes/0.25
y_arrow.location[2]=0
y_arrow.location[0]=0
y_arrow.location[1]=1.1*dim/2
y_arrow.data.materials.append(get_coordinate_material(z_color))



#######################################
###    Animations
######################################################


# set first and last frame index
total_time = 50 
fps =30  # Frames per second (fps)


bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(total_time*fps)+1
camera_empty.location[0]=-5
camera_empty.location[2]=5

#and make plane visible in the first half second
offset = 1
bpy.context.scene.frame_set(offset)
x_arrow.location[0]=0
x_arrow.scale[2]=0
x_arrow.keyframe_insert("location")
x_arrow.keyframe_insert("scale")
y_arrow.location[1]=0
y_arrow.scale[2]=0
y_arrow.scale[1]=0
y_arrow.scale[0]=0
y_arrow.keyframe_insert("location")
y_arrow.keyframe_insert("scale")
onehalfline.scale[2]=0
onehalfline.keyframe_insert("scale")
grid_alpha.default_value=0
grid_alpha.keyframe_insert("default_value")

offset = offset+15
bpy.context.scene.frame_set(offset)
grid_alpha.default_value=1
grid_alpha.keyframe_insert("default_value")
x_arrow.location[0]=0
x_arrow.scale[2]=0
x_arrow.keyframe_insert("location")
x_arrow.keyframe_insert("scale")
for line in zlines2:
    line.scale[2]=0
    line.keyframe_insert("scale")


#grow zlines in the second second
offset = offset+15   
bpy.context.scene.frame_set(offset)
x_arrow.location[0]=1.1*dim/2
x_arrow.keyframe_insert("location")
x_arrow.scale[2]=10*r_axes/0.25
x_arrow.keyframe_insert("scale")
for line in zlines2:
    line.scale[2]=1
    line.keyframe_insert("scale")


y_arrow.location[1]=0
y_arrow.scale[2]=0
y_arrow.scale[1]=0
y_arrow.scale[0]=0
y_arrow.keyframe_insert("location")
y_arrow.keyframe_insert("scale")
bpy.context.scene.frame_set(offset)
for line in ylines:
    line.scale[2]=0
    line.keyframe_insert("scale")


offset=offset+15
bpy.context.scene.frame_set(offset)
for line in ylines:
    line.scale[2]=1
    line.keyframe_insert("scale")    
y_arrow.location[1]=1.2*ydim/2
y_arrow.keyframe_insert("location")
y_arrow.scale[2]=10*r_axes/0.25
y_arrow.scale[1]=r_axes/0.25
y_arrow.scale[0]=r_axes/0.25
y_arrow.keyframe_insert("scale")



#hue switcher 2
hue_switcher = plane_material.node_tree.nodes["Mix"]
hue_switcher.inputs[0].default_value=0
hue_switcher.inputs[0].keyframe_insert("default_value")


#displacement_switcher 
factor_zeta = plane_material.node_tree.nodes["Math.003"]
factor_zeta.inputs[1].default_value=1
factor_log_zeta = plane_material.node_tree.nodes["Math.005"]
factor_log_zeta.inputs[1].default_value=0
factor_zeta.inputs[1].keyframe_insert("default_value")
factor_log_zeta.inputs[1].keyframe_insert("default_value")


#switch on (plane to function plane)
mix_shader_node = plane_material.node_tree.nodes["Mix Shader"]
bpy.context.scene.frame_set(offset)
mix_shader_node.inputs[0].default_value = 0
mix_shader_node.inputs[0].keyframe_insert("default_value")
offset=offset+30
bpy.context.scene.frame_set(offset)
mix_shader_node.inputs[0].default_value = 1
mix_shader_node.inputs[0].keyframe_insert("default_value")


#turn on heights
displacement_node = plane_material.node_tree.nodes["Displacement"]
displacement_node.inputs[2].default_value=0
displacement_node.inputs[2].keyframe_insert("default_value")


offset=offset+90
bpy.context.scene.frame_set(offset)
displacement_node.inputs[2].default_value = 14
displacement_node.inputs[2].keyframe_insert("default_value")


ttf.offset_factor = 0.0
ttf.keyframe_insert("offset_factor")

#traverse zeta 
offset = offset +600
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 1
ttf.keyframe_insert("offset_factor")


#switch functions

factor_zeta.inputs[1].default_value=1
factor_log_zeta.inputs[1].default_value=0
factor_zeta.inputs[1].keyframe_insert("default_value")
factor_log_zeta.inputs[1].keyframe_insert("default_value")
hue_switcher.inputs[0].default_value=0
hue_switcher.inputs[0].keyframe_insert("default_value")

offset = offset+220
bpy.context.scene.frame_set(offset)
factor_zeta.inputs[1].default_value=0
factor_log_zeta.inputs[1].default_value=1
factor_zeta.inputs[1].keyframe_insert("default_value")
factor_log_zeta.inputs[1].keyframe_insert("default_value")
hue_switcher.inputs[0].default_value=1
hue_switcher.inputs[0].keyframe_insert("default_value")
ttf.offset_factor = 1
ttf.keyframe_insert("offset_factor")


#traverse zeta 
offset = offset +400
bpy.context.scene.frame_set(offset)
ttf.offset_factor = 0
ttf.keyframe_insert("offset_factor")

print(offset)

