import bpy
import numpy as np
import bmesh



def cos(z):
    return np.cos(z)


def plane(x_min,x_max,y_min,y_max,cx,cy,res,function):
    bpy.ops.mesh.primitive_plane_add()  # Adds in a plane to work with
    plane = bpy.context.active_object.data  # Selects the plane's data
    bm = bmesh.new()   # Creates an empty BMesh
    bm.from_mesh(plane)   # Fills it in using the plane 

    plane_size = [x_max-x_min, y_max-y_min, 0.1]  # Takes user inputs

    location = [cx, cy, 0]  # later there has to be an adjustment for unsymmetric ranges in x and y
    bmesh.ops.subdivide_edges(bm,edges=bm.edges,cuts=res,use_grid_fill=True)
    bmesh.ops.scale(bm,vec=plane_size,verts=bm.verts)
    bmesh.ops.translate(bm,vec=location,verts=bm.verts)

    # Defining a grid of complex points and computing the user's function:
    for v in bm.verts:
        X,Y = np.meshgrid(v.co.x, v.co.y)
        z = X + 1j*Y

        func = function(z)
        v.co.z = np.abs(func)
    
    bm.to_mesh(plane)  # Freeing the BMesh, moving on to coloring the domain
    bm.free()

    # Assigning color to the vertices:
    vert_list = plane.vertices
    color_map_collection = plane.vertex_colors

    if len(color_map_collection) == 0:   # Creates a new color map or replaces 
        color_map_collection.new()

    color_map = color_map_collection['Col'] # the current one under name 'Col'

    i = 0
    for poly in plane.polygons:               
        for idx in poly.loop_indices:
            loop = plane.loops[idx]                              # For loop used for coloring each vertex  
            v = loop.vertex_index         
            
            # 'z' is a complex number with the x-coordinate of the vertex being the real part
            # and the y-coordinate of the vertex the imaginary part:                         
            z = vert_list[v].co.x+1j*vert_list[v].co.y
            
            func = function(z)
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
    return plane


plane(-10,10,-1,1,0,0,50,cos)