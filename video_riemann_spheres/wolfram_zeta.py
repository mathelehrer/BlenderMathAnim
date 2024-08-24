import bpy
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl,wlexpr
import numpy as np
import bmesh
from mathutils import Vector
import colorsys
import math



s = WolframLanguageSession()
  
  
def test(z):
    return (z-0.5)*(z-1)*(z-1j*2)*(z-3)*(z-1j*4)*(z-5)*(z-1j*6) 
 
def zeta(z):
    tmp=s.evaluate(wl.N(wl.Log(wl.Zeta(z))))
    if isinstance(tmp,float):
        return tmp
    else:
        return tmp[0]+1j*tmp[1]
    
    
def sphere2plane(x,y,z,R):
        return R*x/(R-z)+1j*R*y/(R-z)      
   

def riemann(resolution = 16, radius=1, function = "z"):
    #create uv-sphere
    bpy.ops.mesh.primitive_uv_sphere_add(segments=resolution, ring_count=resolution, radius=radius, enter_editmode=False, location=(0, 0, 0))  
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
                        
            z = sphere2plane(X,Y,Z,radius)
            # Using compile() and eval() like before for the absolute value, this time for the phase:                

                           
            func = function(z)
            
            angle = np.angle(func)  # Returns the phase of the complex number
            if math.isnan(angle):
                final = colorsys.hsv_to_rgb(0,0,0)
            else:
                final = colorsys.hsv_to_rgb((0.5*angle/np.pi)%1,1,1)

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

    # Setting to object mode:s
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
        
        z = sphere2plane(x,y,z,radius)
                     
        func = function(z)
        
        mag = np.abs(func)
        l_mag = np.log(mag)/np.log(10) # Returns the phase of the complex number
     
        if l_mag<0:
            sk.data[i].co = 1/(1-l_mag) *sk.data[i].co
        else:
            sk.data[i].co = (1+l_mag)*sk.data[i].co 
       
    return object   



res =1024
scale = 25
sphere = riemann(res, scale,function = zeta)
s.terminate()