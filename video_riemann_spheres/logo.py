import bpy
import numpy as np
import bmesh
from mathutils import Vector
import colorsys



#make objects glow
bpy.context.scene.eevee.use_bloom = True
#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# add camera and target to empty
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(-6.9,-2.5,20), rotation=(0, 0, 0), scale=(1, 1, 1))
camera = bpy.context.active_object
camera.data.lens = 45



bpy.ops.mesh.primitive_circle_add(radius=0.5, enter_editmode=False, align='WORLD', location=(0, 0.5, 0), scale=(1, 1, 1))
bpy.ops.mesh.primitive_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 1, 0), scale=(1, 1, 1))


def harmonic(z):
    sum = 0
    for k in range(1,100):
        sum = sum+z/k/(k+z)
    return sum

def newton(z,n):
    eps = 0.01
    fz = 1
    count  = 0
    xi = z
    while np.abs(fz)>eps:
        count+=1
        fz = xi**n-1
        fzp = n*xi**(n-1)
        xi -= fz/fzp
    return count/10*np.exp(1j*np.angle(xi))

def get_material(color):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[20].default_value=7.5  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[19].default_value = color


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

def sphere2plane(x,y,z):
        r = 1
        return r*x/(r-z)+1j*r*y/(r-z)      
        
def riemann(resolution = 16, radius = 1, function = "z"):
    #create uv-sphere
    bpy.ops.mesh.primitive_uv_sphere_add(segments=resolution, ring_count=resolution, radius=1, enter_editmode=False, location=(0, 0, 0))  
    object = bpy.context.active_object
    me = object.data  # Selects the plane's data

    bm = bmesh.new()   # Creates an empty BMesh
    bm.from_mesh(me)   # Fills it in using the plane 

    #rotate sphere to regularize poles
    epsilon = 0.01
    #print("Before rotation:")
    for v in bm.verts:
        x = v.co.x
        y= v.co.y
        z= v.co.z
        
        #perform tiny rotation to regularize the poles
        rot = np.array([[np.cos(epsilon), 0,np.sin(-epsilon)],[0,1,0], [np.sin(epsilon), 0,np.cos(epsilon)]])
        vec = np.array([x,y,z])
        vec2 = np.matmul(rot,vec)
        
        v.co = vec2
       
    bm.to_mesh(me)  # Freeing the BMesh, moving on to coloring the domain
    bm.free()

    # Assigning color to the vertices:
    vert_list = me.vertices
    color_map_collection = me.vertex_colors

    if len(color_map_collection) == 0:   # Creates a new color map or replaces 
        color_map_collection.new()

    color_map = color_map_collection['Col']

    i = 0
    for poly in me.polygons:               
        for idx in poly.loop_indices:
                                          # For loop used for coloring each vertex  
            loop = me.loops[idx]
            
            v = loop.vertex_index
            
            # 'z' is a complex number with the x-coordinate of the vertex being the real part
            # and the y-coordinate of the vertex the imaginary part:            
            
            X,Y,Z= vert_list[v].co[:]        
                        
            z = sphere2plane(X,Y,Z)
            # Using compile() and eval() like before for the absolute value, this time for the phase:                

            result = compile(function,'','eval')                
            func = eval(result)
            
            angle = np.angle(func)  # Returns the phase of the complex number
            final = colorsys.hsv_to_rgb(0.5+0.5*angle/np.pi,1,1)

            color_map.data[i].color = (*final,0)
            i += 1

    # Connecting the Vertex Color node output to the default Principled BSDF base color input
    # to see color in rendered view:

    phase_color = bpy.data.materials.new(name="Phase Color")
    phase_color.use_nodes = True

    nodes = phase_color.node_tree.nodes

    p_bsdf = nodes.get("Principled BSDF")
    p_bsdf.inputs['Emission Strength'].default_value=0.1 #emission
    p_bsdf.inputs['Metallic'].default_value = 1 # metallic
    p_bsdf.inputs['Roughness'].default_value = 0 #roughness
    vert_col = nodes.new(type='ShaderNodeVertexColor')

    links = phase_color.node_tree.links

    links.new(vert_col.outputs[0], p_bsdf.inputs[0])
    links.new(vert_col.outputs[0], p_bsdf.inputs[19])

    bpy.context.object.active_material = phase_color

    # Setting to object mode:
    bpy.ops.object.mode_set(mode='OBJECT')  
    
    #add shape keys for the profile
    verts = me.vertices

    sk_basis = object.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    object.data.shape_keys.use_relative = True


    sk = object.shape_key_add(name='Profile')
    sk.interpolation = 'KEY_LINEAR'

    #calculate radius
    x,y,z = sk.data[0].co[:]
    r = np.sqrt(x*x+y*y+z*z)
  
    # position each vert
    for i in range(len(verts)):
        x,y,z=sk.data[i].co[:]
        
        z = sphere2plane(x,y,z)
        
        result = compile(function,'','eval')                
        func = eval(result)
        
        mag = np.abs(func)
        l_mag = np.log(mag)/np.log(10) # Returns the phase of the complex number
     
        if l_mag<0:
            sk.data[i].co = 1/(1-l_mag) *sk.data[i].co
        else:
            sk.data[i].co = (1+l_mag)*sk.data[i].co 
       
    return object   

locations = []
radii = []
alphas = []
functions = ["z*z*z*z*z-0.5**5","harmonic(z)","np.exp(z)","np.tanh(z)","np.sin(z)","newton(z,5)"]


logo_mat = []
dim = 40
for i in range(-dim,dim+1):
    scale = 1/(1/4*i*i+1/2)
    r = 1/4*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=(i/2*scale, 3/4*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    mat = get_material((0.8, 0, 0.00318879, 1))
    logo_mat.append(mat)
    obj.data.materials.append(mat)
    if np.abs(i)<3 or i==-3:
        locations.append(Vector((i/2*scale, 3/4*scale, 0)))
        radii.append(r)
        alphas.append(mat.node_tree.nodes["Principled BSDF"].inputs[21])
    
for i in range(-dim,dim+1):
    scale = 1/(1/4*i*i-1/4*i+96/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 9/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    mat = get_material((0.00318879, 0.8, 0., 1))
    logo_mat.append(mat)
    obj.data.materials.append(mat)
    
for i in range(-dim,dim+1):
    scale = 1/(1/4*i*i-1/4*i+240/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 15/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    mat = get_material((0.00318879, 0, 0.8, 1))
    logo_mat.append(mat)
    obj.data.materials.append(mat)


res =16 #512
spheres = []
for r,location,function in zip(radii,locations,functions):
    sphere = riemann(res,1,function)
    sphere.scale[0]=r
    sphere.scale[1]=r
    sphere.scale[2]=r
    sphere.location = location
    spheres.append(sphere)
    

###########################################
########## Animation ######################
###########################################

# set first and last frame index
total_time = 20 # Animation should be 2*pi seconds long
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = int(total_time*fps)+1
bpy.context.scene.frame_set(1)

pause = 15
offset = 1
axis = 1

        
#make all alphas zero
for alpha in alphas:
    alpha.default_value = 0
    alpha.keyframe_insert("default_value")
    

new_locations  = []
Z = 5
x_spread = 3.75
y_spread = 3
x_center =-7.025
y_center =-5.5

delta_x = x_spread/3
delta_y = y_spread/2

for x in range(-1,2):
    for y in range(-1,1):
        Y = (0.5-y)*y_spread+y_center
        X = x*x_spread+x_center
        new_locations.append(Vector((X,Y,Z)))
        
rotations = 2.75
scale = 1   
 
scene = bpy.context.scene    
for i,sphere in enumerate(spheres):
    scene.frame_set(offset)
    sphere.keyframe_insert("location")
    sphere.keyframe_insert("scale")
    sphere.keyframe_insert("rotation_euler")
    alphas[i].keyframe_insert("default_value")
    
    scene.frame_set(offset+30)
    sphere.location=new_locations[i]
    sphere.scale[0]=scale
    sphere.scale[1]=scale
    sphere.scale[2]=scale
    sphere.keyframe_insert("location")
    sphere.keyframe_insert("scale")
    alphas[i].default_value = 1
    alphas[i].keyframe_insert("default_value")
    
    scene.frame_set(offset+200)
    sphere.rotation_euler[0]=rotations*np.pi*2
    sphere.keyframe_insert("rotation_euler")
    offset = offset+30
        
offset +=2*pause

rotations +=2.25


#fade away logo

for mat in logo_mat:
    scene.frame_set(offset)
    alpha = mat.node_tree.nodes["Principled BSDF"].inputs[21]
    alpha.default_value = 1
    alpha.keyframe_insert("default_value")
    scene.frame_set(offset+60)
    alpha.default_value = 0
    alpha.keyframe_insert("default_value")

#profile shape for the various functions
shapes = [0.05,0.625,0.01,0.5,0.02,0.9]
for i,sphere in enumerate(spheres):
    scene.frame_set(offset)
    sphere.keyframe_insert("rotation_euler")
    shape= sphere.data.shape_keys.key_blocks["Profile"]
    shape.value = 0
    shape.keyframe_insert("value")
    
    scene.frame_set(offset+200)
    shape.value = shapes[i]
    shape.keyframe_insert("value")
    sphere.rotation_euler[1]=rotations*np.pi*2
    sphere.keyframe_insert("rotation_euler")
    offset = offset+30
    
print(offset)