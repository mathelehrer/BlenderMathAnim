# Imports:
from typing import Any

import bpy
import bmesh
import numpy as np

bl_info = {
    "name": "Modular Surface",
    "description": "Plots a complex-valued function. Z coordinate represents absolute value and color represents phase.",
    "author": "Lemmaxiom",
    "version": (1, 0),
    "blender": (2, 91, 0),
    "location": "View3D > Add > Mesh",
    "category": "Add Mesh"
}

class MESH_OT_modular_surf(bpy.types.Operator):
    """Create a modular surface of a complex-valued function"""   # Operator tool-tips description
    bl_idname = "mesh.modular_surface_add"     # Operator ID name (for typing in console)
    bl_label = "Modular Surface"          # Operator name (for searching using F3 shortcut)
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

# Defining adjustable properties that appear in the undo panel in the viewport:

    sub_num: bpy.props.IntProperty(
        name="Subdivisions",
        description = "Assigns the number of subdivisions for the plane.",
        default=32,
        min=0,soft_max=128)
            
    x_size: bpy.props.FloatProperty(
        name="Size X",
        description = "Determines size of the plane in the x-direction.",
        default=1,
        min=0,soft_max=10)
        
    y_size: bpy.props.FloatProperty(
        name="Size Y",
        description = "Determines size of the plane in the y-direction.",
        default=1,
        min=0,soft_max=10)
        
    x_loc: bpy.props.FloatProperty(
        name="Location: X",
        description = "Moves plane along the x-axis.",
        default=0,
        soft_min=-20,soft_max=20)
        
    y_loc: bpy.props.FloatProperty(
        name ="Location: Y",
        description = "Moves plane along the y-axis.",
        default = 0,
        soft_min = -20,soft_max=20)
        
    alpha: bpy.props.FloatProperty(
        name = "Parameter alpha",
        description = "A free parameter that can be used in the function.",
        default = 2,
        soft_min= 0,
        soft_max = 20)
        
    function: bpy.props.StringProperty(
        name="Function",
        description = "Defines the function. Variable must be 'z' and requires use of numpy operations (make sure to type 'np.' before functions)",
        default = "np.log(z)/np.cosh(z)")

# Executing the code:

    def execute(self, context):
        
        alpha = self.alpha
        function = self.function
        function = function.replace("alpha",str(alpha))
        
        bpy.ops.mesh.primitive_plane_add()  # Adds in a plane to work with
       
        me = bpy.context.active_object.data  # Selects the plane's data

        bm = bmesh.new()   # Creates an empty BMesh
        bm.from_mesh(me)   # Fills it in using the plane 

        plane_size = [self.x_size, self.y_size, 1]  # Takes user inputs
        
        location = [self. x_loc, self.y_loc, 1]  # Takes user inputs
        
# Subdividing, scaling, and moving the plane according to user inputs:  
             
        bmesh.ops.subdivide_edges(bm,edges=bm.edges,cuts=self.sub_num,use_grid_fill=True)

        bmesh.ops.scale(bm,vec=plane_size,verts=bm.verts)
        
        bmesh.ops.translate(bm,vec=location,verts=bm.verts)

# Defining a grid of complex points and computing the user's function:
      
        for v in bm.verts:
            X,Y = np.meshgrid(v.co.x, v.co.y)
            z = X + 1j*Y

# Need to use compile() to convert the input string into a code object,
# then finally the eval() function to get a viable complex-valued output:

            
            result = compile(function,'','eval')
            func = eval(result)
            
# "v.co.z", the z-coordinate of each vertex, represents the absolute value of the function's output:

            v.co.z = np.abs(func)
            
        bm.to_mesh(me)  # Freeing the BMesh, moving on to coloring the domain
        bm.free()

# Assigning color to the vertices:

        vert_list = me.vertices
        color_map_collection = me.vertex_colors

        if len(color_map_collection) == 0:   # Creates a new color map or replaces 
                                               # the current one under name 'Col'
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
                            
                z = vert_list[v].co.x+1j*vert_list[v].co.y
                
# Using compile() and eval() like before for the absolute value, this time for the phase:                

                result = compile(function,'','eval')                
                func = eval(result)
                
                angle = np.angle(func)  # Returns the phase of the complex number
                
# Dividing the four quadrants of the complex plane and
# coloring vertices according to where they lie in each one:

            # Quad 1:
            
                if angle <= np.pi/2 and angle >= 0:
                    
                    red = (angle)+1
                    green = angle/2.2
                    blue = 0
                    
            # Quad 2:
                    
                elif angle <= np.pi and angle >= np.pi/2:
                    red = -angle+3.1
                    green = angle-0.85
                    blue = angle/4-0.9
                    
            # Quad 3:
                 
                elif angle >= -np.pi and angle <= -np.pi/2:
                    
                    red = 0
                    green = -angle-1.7
                    blue = angle+3.3
                    
            # Quad 4:
                    
                else:
                    
                    red = angle+1.65
                    green = 0
                    blue = -(angle) 
                    
                t = 0
                
                final = (red,green,blue,t)  # Final color as determined by the lines above
                
                color_map.data[i].color = final
                i += 1

# Connecting the Vertex Color node output to the default Principled BSDF base color input
# to see color in rendered view:

        phase_color = bpy.data.materials.new(name="Phase Color")
        phase_color.use_nodes = True

        nodes = phase_color.node_tree.nodes

        p_bsdf = nodes.get("Principled BSDF")

        vert_col = nodes.new(type='ShaderNodeVertexColor')

        links = phase_color.node_tree.links

        links.new(vert_col.outputs[0], p_bsdf.inputs[0])

        bpy.context.object.active_material = phase_color

# Setting to object mode:

        bpy.ops.object.mode_set(mode='OBJECT')
                
        return {'FINISHED'}

# Adding an "add mesh" button to the UI menu:

def add_button(self,context):
    self.layout.operator(
        MESH_OT_modular_surf.bl_idname,
        text = "Modular Surface",
        icon = 'RNDCURVE')

def register():
    bpy.utils.register_class(MESH_OT_modular_surf)
    bpy.types.VIEW3D_MT_mesh_add.append(add_button)

def unregister():
    bpy.utils.unregister_class(MESH_OT_modular_surf)
    bpy.types.VIEW3D_MT_mesh_add.remove(add_button)

if __name__ == "__main__":
    register()