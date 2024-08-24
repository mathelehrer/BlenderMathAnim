# learnt from https://www.youtube.com/watch?v=mljWBuj0Gho

import bpy # blender python api
import bmesh
from mathutils import Vector
import numpy as np


# remove all existing objects

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


#light
bpy.ops.object.light_add(type='SUN', radius=1.86, align='WORLD', location=(0,-1.4571, 0.42159), scale=(1, 1, 1))
sun = bpy.context.active_object
sun.data.energy = 1
sun.data.angle = 1.02451

bpy.ops.object.light_add(type='SPOT', radius=1.65, align='WORLD', location=(-0.003053,-0.00141,1.58), scale=(1, 1, 1))
spot = bpy.context.active_object
spot.data.spot_size=0.45204



# add camera and target to plane
camera_pos = (0,0,0)
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=camera_pos, rotation=(0,1.570795,1.570795), scale=(1, 1, 1))
camera = bpy.context.active_object

# make world black
bpy.data.worlds["World.001"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# curve for camera rotation
bpy.ops.curve.primitive_bezier_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0.,0.,0.), scale=(1, 1, 1))
curve = bpy.context.active_object
bpy.ops.transform.rotate(value=1.5708,orient_axis='Y')
bpy.ops.transform.resize(value=(1.665,1.665,1.665))

camera.constraints.new("FOLLOW_PATH").target=curve
camera.constraints["Follow Path"].target = bpy.data.objects["BezierCircle"]
camera.constraints["Follow Path"].forward_axis = 'FORWARD_X'
camera.constraints["Follow Path"].up_axis = 'UP_Y'
camera.constraints["Follow Path"].offset = 25


# plane
bpy.ops.mesh.primitive_plane_add(size=0.3, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1,1,1))
plane = bpy.context.active_object

bpy.ops.object.modifier_add(type='SOLIDIFY')
plane.modifiers["Solidify"].thickness = 0.004


# create material
mat  = bpy.data.materials.new("checker")
mat.use_nodes = True
node_tree = mat.node_tree
nodes = node_tree.nodes # create access for nodes
links =mat.node_tree.links #create access for links

output = nodes.get("Material Output")
output.location = Vector((800,200))

bsdf = nodes.get("Principled BSDF") 
bsdf.location = Vector((400,0))

checkerShader = nodes.new(type='ShaderNodeTexChecker')
checkerShader.location=Vector((200,0))
checkerShader.inputs[3].default_value=5.5
checkerShader.inputs[2].default_value = (1, 0, 0, 1)


textureInput = nodes.new(type='ShaderNodeTexCoord')
textureInput.location=Vector((0,0))

links.new(bsdf.outputs[0],output.inputs[0])
links.new(checkerShader.outputs[0],bsdf.inputs[0])
links.new(textureInput.outputs[0],checkerShader.inputs[0])


plane.data.materials.append(mat)

bpy.ops.object.editmode_toggle()  # switch to edit mode that only the mesh is moved but the plane center still stays at the origin
bpy.ops.transform.translate(value=(-0.15,0,0))
bpy.ops.transform.translate(value=(0,-0.15,0))

bpy.ops.object.editmode_toggle() # go back to object mode
bpy.ops.object.modifier_add(type='MIRROR')
bpy.context.object.modifiers["Mirror"].use_axis[1] = True

bm = bmesh.new()
me = plane.data
bm.from_mesh(me)

#subdivide
bmesh.ops.subdivide_edges(bm,edges=bm.edges,cuts=10,use_grid_fill=True,)
bm.to_mesh(me)
me.update()

# set trackers
sun.constraints.new('TRACK_TO').target=plane
spot.constraints.new('TRACK_TO').target =plane
camera.constraints.new('TRACK_TO').target = plane
camera.constraints["Track To"].target_space = 'LOCAL'
camera.constraints["Track To"].owner_space = 'LOCAL'


