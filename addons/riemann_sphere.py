# Imports:
from typing import Any

import bpy
import bmesh
import numpy as np
import colorsys

bl_info = {
    "name": "Riemann sphere",
    "description": "Plots the Riemann sphere of a complex-valued function.",
    "author": "NumberCruncher",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Add > Mesh",
    "category": "Add Mesh"
}

class MESH_OT_riemann_sphere(bpy.types.Operator):
    """Create the Riemann sphere to a complex-valued function"""   # Operator tool-tips description
    bl_idname = "mesh.riemann_sphere_add"     # Operator ID name (for typing in console)
    bl_label = "Riemann Sphere"          # Operator name (for searching using F3 shortcut)
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

# Defining adjustable properties that appear in the undo panel in the viewport:

    sub_num: bpy.props.IntProperty(
        name="Subdivisions",
        description = "Assigns the number of subdivisions for the uv-sphere.",
        default=32,
        min=0,soft_max=128)
            
    radius: bpy.props.FloatProperty(
        name="raidus R",
        description = "Determines the radius of the Riemann sphere.",
        default=1,
        min=0,soft_max=10)
        
    function: bpy.props.StringProperty(
        name="Function",
        description = "Defines the function. Variable must be 'z' and requires use of numpy operations (make sure to type 'np.' before functions)",
        default = "np.sin(z)")

    # Define the stereographic projection:

    def sphere2plane(self,x,y,z):
        r = self.radius
        return r*x/(r-z)+1j*r*y/(r-z)
    
    #Executing the code:
    def execute(self, context):
        
        #create uv-sphere
        bpy.ops.mesh.primitive_uv_sphere_add(segments=self.sub_num, ring_count=self.sub_num, radius=self.radius, enter_editmode=False, location=(0, 0, 0))  
        object = bpy.context.active_object
        me = object.data  # Selects the plane's data

        bm = bmesh.new()   # Creates an empty BMesh
        bm.from_mesh(me)   # Fills it in using the plane 
        
        function = self.function

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
                            
                z = self.sphere2plane(vert_list[v].co.x,vert_list[v].co.y,vert_list[v].co.z)
                # Using compile() and eval() like before for the absolute value, this time for the phase:                

                result = compile(function,'','eval')                
                func = eval(result)
                
                angle = np.angle(func)  # Returns the phase of the complex number
                final = colorsys.hsv_to_rgb((0.5*angle/np.pi)%1,1,1)

                color_map.data[i].color = (*final,0)
                i += 1

        # Connecting the Vertex Color node output to the default Principled BSDF base color input
        # to see color in rendered view:

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
            
            z = self.sphere2plane(x,y,z)
            
            result = compile(function,'','eval')                
            func = eval(result)
            
            mag = np.abs(func)
            l_mag = np.log(mag)/np.log(10) # Returns the phase of the complex number
         
            if l_mag<0:
                sk.data[i].co = 1/(1-l_mag) *sk.data[i].co
            else:
                sk.data[i].co = (1+l_mag)*sk.data[i].co 
             
        return {'FINISHED'}

# Adding an "add mesh" button to the UI menu:

def add_button(self,context):
    self.layout.operator(
        MESH_OT_riemann_sphere.bl_idname,
        text = "Riemann Sphere",
        icon = 'RNDCURVE')

def register():
    bpy.utils.register_class(MESH_OT_riemann_sphere)
    bpy.types.VIEW3D_MT_mesh_add.append(add_button)

def unregister():
    bpy.utils.unregister_class(MESH_OT_riemann_sphere)
    bpy.types.VIEW3D_MT_mesh_add.remove(add_button)

if __name__ == "__main__":
    register()