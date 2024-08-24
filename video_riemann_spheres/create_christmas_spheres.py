import bpy
from mathutils import Vector
import numpy as np
import math
from numpy import cos, sin
import fnmatch
from typing import Any
import bmesh
import colorsys
import random as rnd

scene = bpy.context.scene
context = bpy.context
link_object = scene.collection.objects.link if bpy.app.version >= (2, 80) else scene.objects.link
unlink_object = scene.collection.objects.unlink if bpy.app.version >= (2, 80) else scene.objects.unlink


#final positions of the sphere
sphere_positions = [
Vector((1.9911, -9.6600, 26.2783)),
Vector((-10.2932, -4.5089, 20.5601)),
Vector((11.9972, -1.2301, 15.1555)),
Vector((4.9564, -10.3242, 17.6558)),
Vector((-7.3968, 17.5650, 10.7369)),
Vector((1.2992, -15.0748, 9.2636)),
Vector((8.0052, 7.0541, 35.5856)),
Vector((1.9245, 8.3490, 41.5682)),
Vector((2.9794, 12.6949, 29.4343)),
Vector((-6.4862, -8.4885, 28.3151)),
Vector((7.3287, -6.9125, 21.6883)),
Vector((8.9744, -10.4049, 7.9289)),
Vector((-13.6468, -5.6249, 9.1932)),
Vector((-6.5056, -9.3024, 16.3148)),
Vector((10.8952, 9.4436, 22.9665)),
Vector((3.3940, 14.6351, 16.4709)),
Vector((8.7340, 0.8688, 41.6861)),
Vector((14.8800, 2.0108, 17.5796)),
Vector((9.1790, -5.4982, 35.4352)),
Vector((11.5876, 0.8579, 29.1828)),
Vector((3.6423, -7.9365, 40.2864)),
Vector((-5.7822, 13.5562, 23.0477)),
Vector((-2.3391, -10.3783, 35.4906)),
Vector((-4.0467, 9.3353, 35.3405)),
Vector((-2.9580, -13.3012, 22.7862)),
Vector((12.5375, -6.7661, 23.9006)),
Vector((-6.9304, 4.0278, 41.3461)),
Vector((-5.6892, -4.9916, 38.6626)),
Vector((11.9932, -11.2069, 12.1509)),
Vector((-10.4530, -0.9700, 35.4584)),
Vector((-12.2415, 7.5152, 16.3888)),
Vector((-10.7763, 6.5395, 29.4453)),
Vector((5.1163, -11.9962, 29.4810)),
Vector((-9.6470, -4.9801, 29.4320)),
Vector((-19.0936, -1.6578, 10.6921)),
Vector((-7.2368, -14.5491, 10.6690)),
Vector((-13.9725, -3.3010, 23.5020)),
Vector((3.2333, -14.1935, 19.6864)),
Vector((-11.4118, -9.9777, 16.4918))
]

#remove Riemann Spheres
for ob in bpy.context.scene.objects:
    if fnmatch.fnmatch(ob.name,"Riemann*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"BezierCircle*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Camera*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Sun*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Floor*"):
        ob.select_set(True)
    if fnmatch.fnmatch(ob.name,"Parametric*"):
        ob.select_set(True)
    bpy.ops.object.delete()
   
   
##parametrics ####   
   
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
        index,
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
            f(((i - 3) / (3 * iterations)) * (max - min) + min,index, *args, **kwargs)
            for i in range((3 * (iterations + 2)) + 1)
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
        points = [f(i / iterations,index, *args, **kwargs) for i in range(iterations + 1)]

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

start_heights = []
for vec in sphere_positions:
    start_heights.append(40+20*rnd.random())

def f(t, i,offset: float = 0.0):
    start = sphere_positions[i]
    t= t
   
    phi0 = np.angle(start.x+1j*start.y)
    r0 = np.sqrt(start.x*start.x+start.y*start.y)

    r = r0+(150-r0)*t
    phi = phi0+t*2.*np.pi
    z = start.z+(start_heights[i]-start.z)*t
    return (
        r*cos(phi),
        r*sin(phi),
        z)

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
   
   
   
## Materials ####   
    
def floor_material():
    mat = bpy.data.materials.new("material")
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    
    links =mat.node_tree.links

    output = nodes.get("Material Output")
    output.location = Vector((800,200))
    
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs['Emission Strength'].default_value=0.1  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs['Metallic'].default_value=0.5 #metalicity
    bsdf.inputs['Roughness'].default_value=0.1 #roughness
        
    ramp  = nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[1].color=(0.19,0,0.16,1)
    ramp.location=Vector((0,300))
    links.new(ramp.outputs['Color'],bsdf.inputs['Base Color'])
    links.new(ramp.outputs['Color'],bsdf.inputs['Emission'])
    
    voronoi = nodes.new(type='ShaderNodeTexVoronoi')
    voronoi.location = Vector((-300,300))
    links.new(voronoi.outputs['Color'],ramp.inputs['Fac'])
    return mat

def mandelbrot(z,n):
    it = 100
    count =0
    w = 0
    threshold = 10
    blue = 0.3333
    while count<it and threshold>np.abs(w):
        w = w*w+z
        count+=1
    if count ==it:
        r=1.1
        phase = blue*np.pi
    else:
        r = 1-((it-count)/it)
        phase =r*5
        phase %=1
        phase /=4
        phase += n/4
    return r*np.exp(1j*2*np.pi*phase)


roots3 =[]
for i in range(0,3):
    roots3.append(cos(2*np.pi/3*i)+1j*sin(2*i*np.pi/3))
    
 
roots5 =[]
for i in range(0,5):
    roots5.append(cos(2*np.pi/5*i)+1j*sin(2*i*np.pi/5))
    

def newton(z,n=3):
    count =0
    eps = 0.01
    if n==3:
        roots = roots3
    elif n==5:
        roots = roots5
    
    while count<100:
        for i in range(0,n):
            if np.abs(z-roots[i])<eps:
                return count*roots[i]
        tmp = z**(n-1)
        y = tmp*z-1
        yp = n*tmp
        z = z-y/yp
        count+=1
    return 1


def sphere2plane(r,x,y,z):
    return r*x/(r-z)+1j*r*y/(r-z)


def riemann_sphere(sub_num=128, radius=1, function=np.sin):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=sub_num, ring_count=sub_num, radius=radius, enter_editmode=False, location=(0, 0, 0))  
    object = bpy.context.active_object
    object.name = "RiemannSphere"
    me = object.data  # Selects the plane's data
    bm = bmesh.new()   # Creates an empty BMesh
    bm.from_mesh(me)   # Fills it in using the plane 
    epsilon = 0.01 #rotate sphere to regularize poles
    for v in bm.verts:
        x = v.co.x
        y= v.co.y
        z= v.co.z #perform tiny rotation to regularize the poles
        rot = np.array([[np.cos(epsilon), 0,np.sin(-epsilon)],[0,1,0], [np.sin(epsilon), 0,np.cos(epsilon)]])
        vec = np.array([x,y,z])
        vec2 = np.matmul(rot,vec)
        v.co = vec2 
    bm.to_mesh(me)  # Freeing the BMesh, moving on to coloring the domain
    bm.free()
    vert_list = me.vertices  # Assigning color to the vertices:
    color_map_collection = me.vertex_colors
    if len(color_map_collection) == 0:   # Creates a new color map or replaces 
        color_map_collection.new()
    color_map = color_map_collection['Col']
    i = 0
    for poly in me.polygons:               
        for idx in poly.loop_indices:                     
            loop = me.loops[idx] # For loop used for coloring each vertex  
            v = loop.vertex_index
            z = sphere2plane(radius,vert_list[v].co.x,vert_list[v].co.y,vert_list[v].co.z)
            func = function(z)
            angle = np.angle(func)  # Returns the phase of the complex number
            final = colorsys.hsv_to_rgb((0.5*angle/np.pi)%1,1,1)
            color_map.data[i].color = (*final,0)
            i += 1
    phase_color = bpy.data.materials.new(name="Phase Color")
    phase_color.use_nodes = True
    nodes = phase_color.node_tree.nodes
    p_bsdf = nodes.get("Principled BSDF")
    p_bsdf.inputs['Emission Strength'].default_value=0.01
    p_bsdf.inputs['Metallic'].default_value = 1
    p_bsdf.inputs['Roughness'].default_value = 0 
    vert_col = nodes.new(type='ShaderNodeVertexColor')
    links = phase_color.node_tree.links
    links.new(vert_col.outputs['Color'], p_bsdf.inputs['Base Color'])
    links.new(vert_col.outputs['Color'], p_bsdf.inputs['Emission'])
    bpy.context.object.active_material = phase_color
    bpy.ops.object.mode_set(mode='OBJECT')  
    verts = me.vertices
    sk_basis = object.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    object.data.shape_keys.use_relative = True
    sk = object.shape_key_add(name='Profile')
    sk.interpolation = 'KEY_LINEAR'
    x,y,z = sk.data[0].co[:]
    r = np.sqrt(x*x+y*y+z*z)
    for i in range(len(verts)):
        x,y,z=sk.data[i].co[:]  
        z = sphere2plane(radius,x,y,z)
        func = function(z)
        mag = np.abs(func)
        l_mag = np.log(mag)/np.log(10) # Returns the phase of the complex number
        if l_mag<0:
            sk.data[i].co = 1/(1-l_mag) *sk.data[i].co
        else:
            sk.data[i].co = (1+l_mag)*sk.data[i].co     
    return {'FINISHED'}

def fcn1(z):
    return (z*z-9)/(z-1)
def fcn2(z):
    return (z-3)*(z+2)*(z+2j)/(z**2-9)
def fcn3(z):
    return np.log(np.cos(z)*np.sin(z))
def fcn4(z):
    return mandelbrot(z,2)
def fcn5(z):
    return mandelbrot(z,0)
def fcn6(z):
    return np.log(np.cosh(z))
def fcn7(z):
    return (z*z-16)/(z*z-4)
def fcn8(z):
    return (z-2)
def fcn9(z):
    return z*z/(z**2-4)
def fcn10(z):
    return np.tanh(z)
def fcn11(z):
    return np.log(np.sin(z))
def fcn12(z):
    return newton(z,5)
def fcn13(z):
    return newton(z,3)
def fcn14(z):
    return mandelbrot(z,1)

functions = [fcn1,fcn2,fcn3,fcn4,fcn5,fcn6,fcn7,fcn8,fcn9,fcn10,fcn11,fcn12,fcn13,fcn14]
    
     
#the stem of the christmas tree is the object that sets the track spot for the lights and the camera
stem = scene.objects['Cylinder002']
 
# add lights
bpy.ops.object.light_add(type='SUN', radius=12, align='WORLD', location=(50, 50, 50), scale=(1, 1, 1))
sun1 = bpy.context.active_object
bpy.ops.object.constraint_add(type='TRACK_TO')
sun1.data.energy = 20
sun1.constraints["Track To"].target = stem

bpy.ops.object.light_add(type='SUN', radius=12, align='WORLD', location=(-50, -50, 50), scale=(1, 1, 1))
sun2 = bpy.context.active_object
bpy.ops.object.constraint_add(type='TRACK_TO')
sun2.data.energy = 20
sun2.constraints["Track To"].target = stem

suns = [sun1,sun2]

bpy.ops.curve.primitive_bezier_circle_add(radius=150, enter_editmode=False, align='WORLD', location=(0,0,50), scale=(1, 1, 1))
camera_circle = bpy.context.active_object

bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0,0,0), rotation=(1.10871, 0.0132652, 1.14827), scale=(1, 1, 1))
camera  =bpy.context.active_object
bpy.ops.object.constraint_add(type='FOLLOW_PATH')
bpy.ops.object.constraint_add(type='TRACK_TO')

cc = camera.constraints["Follow Path"]
cc.target=camera_circle

cc.use_fixed_location=True
cc.use_curve_follow=True
  
cf = camera.constraints["Track To"]
cf.target = stem

#floor
bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0,0,0), scale=(1, 1, 1))
floor = bpy.context.active_object
floor.name = 'Floor'
floor.data.materials.append(floor_material())


curves = []
for i in range(0,len(sphere_positions)):
    curves.append(create_parametric_curve(i, offset=0.0, min=0.0, max=1.0, use_cubic=True, resolution_u=100, iterations=100))


riemann_spheres = []
rs_constraints = []

for c,vec in enumerate(sphere_positions):
    blob = riemann_sphere(sub_num=128, radius=1, function=functions[c%len(functions)])
    blob = bpy.context.active_object
    blob.rotation_euler[0]=np.pi*rnd.random()
    bpy.ops.object.constraint_add(type='FOLLOW_PATH')
    
    bc = blob.constraints["Follow Path"]
    rs_constraints.append(bc)
    bc.target=curves[c]

    bc.use_fixed_location=True
    bc.use_curve_follow=True
    
    bpy.ops.constraint.followpath_path_animate(constraint="Follow Path", owner='OBJECT')

    riemann_spheres.append(blob)

########################## Animation #############################

# set first and last frame index
total_time = 45 # 
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 0
end = int(total_time*fps)+1
bpy.context.scene.frame_end = end 


offset = 1
scene.frame_set(offset) 
#initialize all data
camera_circle.scale[0]=1
camera_circle.scale[1]=1
camera_circle.keyframe_insert("scale")


cc.offset_factor=0
cc.keyframe_insert('offset_factor')



#find spheres
alphas  = []
for i in range(1,40):
    i_str = str(i)
    i_str = i_str.zfill(3)
    alphas.append(scene.objects['Sphere'+i_str].data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Alpha'])

for alpha in alphas:
    alpha.default_value = 1
    alpha.keyframe_insert("default_value")

offset = 270
scene.frame_set(offset)
rs_alphas=[]
for rs in riemann_spheres:
    alpha= rs.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Alpha']
    rs_alphas.append(alpha)
    alpha.default_value = 0
    alpha.keyframe_insert('default_value')
    
for alpha in alphas:
    alpha.default_value = 0
    alpha.keyframe_insert("default_value")

#move spheres

f0 = 0
df = 2.*np.pi/270
started = []
for i in range(0,len(riemann_spheres)):
    started.append(False)
    
for i in range(0,270):
    f = f0+df*i
    scene.frame_set(offset+i)
    for c in range(0,len(riemann_spheres)):
        v = sphere_positions[c]
        phase = np.pi/2+np.angle(v.x+1j*v.y)
        if phase%1*2.*np.pi<f or i==269:
            if started[c]==False:
                started[c]=True
                rs_constraints[c].offset_factor=1
                rs_constraints[c].keyframe_insert('offset_factor')
                rs_alphas[c].default_value = 0
                rs_alphas[c].keyframe_insert('default_value')
                scene.frame_set(offset+i+10)  
                rs_alphas[c].default_value = 1
                rs_alphas[c].keyframe_insert('default_value')    
                scene.frame_set(offset+i+120)          
                rs_constraints[c].offset_factor=0
                rs_constraints[c].keyframe_insert('offset_factor')      

#show shapes

for r,rs in enumerate(riemann_spheres):
    
    rndframe = int(30*rnd.random())
    scene.frame_set(offset+800+rndframe)
    mat = rs.data.materials[0]
    sk = rs.data.shape_keys.key_blocks["Profile"]
    sk.value = 0
    sk.keyframe_insert("value")
    
    bpy.context.scene.frame_set(offset+800+rndframe+30)
    sk.value = 0.5
    sk.keyframe_insert("value")
    
    bpy.context.scene.frame_set(250)    
    emission = mat.node_tree.nodes["Principled BSDF"].inputs['Emission Strength']
    emission.default_value=0.1
    emission.keyframe_insert("default_value")
    
    bpy.context.scene.frame_set(1280)    
    emission = mat.node_tree.nodes["Principled BSDF"].inputs['Emission Strength']
    emission.default_value=1
    emission.keyframe_insert("default_value")
    

#dim light
bpy.context.scene.frame_set(1000)
for sun in suns:
    sun.data.energy = 20
    sun.data.keyframe_insert("energy")


bpy.context.scene.frame_set(1280)
for sun in suns:
    sun.data.energy = 1
    sun.data.keyframe_insert("energy")
    

#zoom in
bpy.context.scene.frame_set(end)
camera_circle.scale[0]=0.25
camera_circle.scale[1]=0.25
camera_circle.keyframe_insert("scale")

cc.offset_factor=2.5
cc.keyframe_insert('offset_factor')
   
    

    