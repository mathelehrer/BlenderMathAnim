import bpy
import numpy as np
import bmesh
from mathutils import Vector
from random import random
import os

#Read data from disk

path = "/home/jmartin/Dropbox/MyBlender/nPendulum/g25_40/"
try:
    os.chdir(path)
    print("Current working directory ",os.getcwd())
except OSError:
    print("Can't change the current working directory")
    print("Current working directory ",os.getcwd())

# make big Left dynamic    
file = open("pendulumBigLeft.csv", "r")
bigLeftData = file.readlines()
file.close()
 
# make big Right dynamic    
file = open("pendulumBigRight.csv", "r")
bigRightData = file.readlines()
file.close()
     
# make outer Left dynamic    
file = open("pendulumOuterLeft.csv", "r")
outerLeftData = file.readlines()
file.close()

# make outer Right dynamic    
file = open("pendulumOuterRight.csv", "r")
outerRightData = file.readlines()
file.close()
  
# make inner Left dynamic    
file = open("pendulumInnerLeft.csv", "r")
innerLeftData = file.readlines()
file.close()
  
# make inner Right dynamic    
file = open("pendulumInnerRight.csv", "r")
innerRightData = file.readlines()
file.close()

#make objects glow
bpy.context.scene.eevee.use_bloom = True
#make background black
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


camera_pos = (0,1,5.7)
# add camera and target to empty
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=camera_pos, rotation=(0.78, 0, 0), scale=(1, 1, 1))
camera = bpy.context.active_object

bpy.ops.object.empty_add(type='CIRCLE', radius=0.18, align='WORLD', location=(0, 1, 0), scale=(1, 1, 1))
empty = bpy.context.active_object
camera.constraints.new('TRACK_TO').target = empty

def get_material(color):
    mat = bpy.data.materials.new("material")

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes.get("Material Output")
    output.location = Vector((600,200))
    bsdf = nodes.get("Principled BSDF") 
    bsdf.inputs[18].default_value=7.5  #increase emission strength
    bsdf.location = Vector((300,300))
    bsdf.inputs[17].default_value = color

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


# set first and last frame index
total_time = 45
fps =30  # Frames per second (fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = int(total_time*fps)+1


cylBigLeft =[]
cylBigRight =[]
cylOuterLeft = []
cylOuterRight = []
cylInnerLeft = []
cylInnerRight = []

innerRodesLeft=[]
innerRodesRight=[]
outerRodesLeft=[]
outerRodesRight=[]

#store lengths of the pendulum
lBigLeft = []
lBigRight = []
lInnerRight = []
lOuterRight = []
lOuterLeft = []
lInnerLeft = []

#store locations of disks
locOuterLeft =[]
locOuterRight=[]
locInnerLeft = []
locInnerRight = []

bigLeftZ =0.2
bigRightZ = 0.4

location = (0,1.5,0)
pivot = Vector(location)
tmpPivot = pivot

dim = 4
for i in range(-dim,dim+1):
    scale = 1/(1/4*i*i+1/2)
    r = 1/4*scale
    location = (i/2*scale, 3/4*scale, 0)
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
    #bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    obj.data.materials.append(get_material((0.8, 0, 0.00318879, 1)))
    obj.keyframe_insert(data_path="location")
    if i < 0:
        bpy.ops.transform.translate(value=(0,0,bigLeftZ))
        cylBigLeft.insert(0,obj)
    if i > 0: # the lengths are determined for the right branch, since there the disks are created in the correct order
        bpy.ops.transform.translate(value=(0,0,bigRightZ))
        diff = Vector(location)-tmpPivot
        lBigRight.append(np.sqrt(diff.dot(diff)))
        tmpPivot = Vector(location)
        cylBigRight.append(obj)
    
lBigLeft = lBigRight
print(lBigLeft)
print(lBigRight)        
  
tmpPivot = pivot  # reset to the common pivot
for i in range(-dim+1,dim+1):
    scale = 1/(1/4*i*i-1/4*i+96/256)
    r = 1/16*scale
    if i!=0:
        location = ((2*i-1)/4*scale, 9/16*scale,0)
    else:
        location = ((2*i-1)/4*scale, 9/16*scale, 0)
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
    #bpy.ops.transform.translate(value=(0,0,0.05))
    obj = bpy.context.active_object
    obj.data.materials.append(get_material((0.00318879, 0.8, 0., 1)))
    obj.keyframe_insert(data_path="location")
    if i<=0:
        bpy.ops.transform.translate(value=(0,0,-bigLeftZ))
        cylOuterLeft.insert(0,obj)
        locOuterLeft.insert(0,location)
    if i>0:
        bpy.ops.transform.translate(value=(0,0,-bigRightZ))
        cylOuterRight.append(obj)
        locOuterRight.append(location)
        diff = Vector(location)-tmpPivot
        lOuterRight.append(np.sqrt(diff.dot(diff)))
        tmpPivot = Vector(location)
        
lOuterLeft = lOuterRight
print(lOuterLeft)
print(lOuterRight)

# create outer rodes
for i in range(0,dim-1):
    # left ones
    middle = 0.5*(Vector(locOuterLeft[i+1])+Vector(locOuterLeft[i]))
    line = outerLeftData[0].split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.003, depth=lOuterLeft[i+1], enter_editmode=False, align='WORLD', location=middle, scale=(1, 1, 1))
    obj = bpy.context.active_object
    obj.rotation_euler[0]=1.5708
    obj.rotation_euler[2]=-phis[i+1]
    obj.data.materials.append(get_material((0.00318879, 0.8, 0., 1)))
    obj.keyframe_insert(data_path="location")
    obj.keyframe_insert("rotation_euler")
    bpy.ops.transform.translate(value=(0,0,-bigLeftZ))
    outerRodesLeft.append(obj)
    #right ones
    middle = 0.5*(Vector(locOuterRight[i+1])+Vector(locOuterRight[i]))
    line = outerRightData[0].split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.003, depth=lOuterRight[i+1], enter_editmode=False, align='WORLD', location=middle, scale=(1, 1, 1))
    obj = bpy.context.active_object
    obj.rotation_euler[0]=1.5708
    obj.rotation_euler[2]=-phis[i+1]
    obj.data.materials.append(get_material((0.00318879,0.8, 0, 1)))
    obj.keyframe_insert(data_path="location")
    obj.keyframe_insert("rotation_euler")
    bpy.ops.transform.translate(value=(0,0,-bigRightZ))
    outerRodesRight.append(obj)  
    
tmpPivot = pivot # reset to the common pivot
for i in range(-dim+1,dim+1):
    scale = 1/(1/4*i*i-1/4*i+240/256)
    r = 1/16*scale
    location = ((2*i-1)/4*scale, 15/16*scale, 0)
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*r+10),depth=0.1,radius=r, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
    obj = bpy.context.active_object
    obj.data.materials.append(get_material((0.00318879, 0, 0.8, 1)))
    obj.keyframe_insert(data_path="location")
    if i<=0:
        bpy.ops.transform.translate(value=(0,0,-3*bigLeftZ))
        cylInnerLeft.insert(0,obj)
        locInnerLeft.insert(0,location)
    if i>0:
        bpy.ops.transform.translate(value=(0,0,-2*bigRightZ))
        cylInnerRight.append(obj)
        locInnerRight.append(location)
        diff = Vector(location)-tmpPivot
        lInnerRight.append(np.sqrt(diff.dot(diff)))
        tmpPivot = Vector(location)

lInnerLeft = lInnerRight
print(lInnerLeft)
print(lInnerRight)

# create inner rodes
for i in range(0,dim-1):
    # left ones
    middle = 0.5*(Vector(locInnerLeft[i+1])+Vector(locInnerLeft[i]))
    line = innerLeftData[0].split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.003, depth=lInnerLeft[i+1], enter_editmode=False, align='WORLD', location=middle, scale=(1, 1, 1))
    obj = bpy.context.active_object
    obj.rotation_euler[0]=1.5708
    obj.rotation_euler[2]=-phis[i+1]
    obj.data.materials.append(get_material((0.00318879, 0.,0.8, 1)))
    obj.keyframe_insert(data_path="location")
    obj.keyframe_insert("rotation_euler")
    bpy.ops.transform.translate(value=(0,0,-3*bigLeftZ))
    innerRodesLeft.append(obj)
    #right ones
    middle = 0.5*(Vector(locInnerRight[i+1])+Vector(locInnerRight[i]))
    line = innerRightData[0].split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.003, depth=lInnerRight[i+1], enter_editmode=False, align='WORLD', location=middle, scale=(1, 1, 1))
    obj = bpy.context.active_object
    obj.rotation_euler[0]=1.5708
    obj.rotation_euler[2]=-phis[i+1]
    obj.data.materials.append(get_material((0.00318879, 0., 0.8, 1)))
    obj.keyframe_insert(data_path="location")
    obj.keyframe_insert("rotation_euler")
    bpy.ops.transform.translate(value=(0,0,-2*bigRightZ))
    innerRodesRight.append(obj)

pause = 5*30 #start animation after 5 seconds

for (k,line) in enumerate(bigLeftData):
    bpy.context.scene.frame_set(pause+k)
    line = line.split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    #calculate positions from the angles
    X=pivot[0]
    Y=pivot[1]
    x=[]
    y=[]
    for i in range(0,4):
        X=X-lBigLeft[i]*np.sin(phis[i])
        Y=Y-lBigLeft[i]*np.cos(phis[i])
        x.append(X)
        y.append(Y)
    for (i,cyl) in enumerate(cylBigLeft):
        cyl.location = (x[i],y[i],bigLeftZ)
        cyl.keyframe_insert(data_path="location")
    
for (k,line) in enumerate(bigRightData):
    bpy.context.scene.frame_set(pause+k)
    line = line.split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    #calculate positions from the angles
    X=pivot[0]
    Y=pivot[1]
    x=[]
    y=[]
    for i in range(0,4):
        X=X-lBigRight[i]*np.sin(phis[i])
        Y=Y-lBigRight[i]*np.cos(phis[i])
        x.append(X)
        y.append(Y)
    for (i,cyl) in enumerate(cylBigRight):
        cyl.location = (x[i],y[i],bigRightZ)
        cyl.keyframe_insert(data_path="location")
        
for (k,line) in enumerate(outerLeftData):
    bpy.context.scene.frame_set(pause+k)
    line = line.split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    #calculate positions from the angles
    X=pivot[0]
    Y=pivot[1]
    x=[]
    y=[]
    for i in range(0,4):
        X=X-lOuterLeft[i]*np.sin(phis[i])
        Y=Y-lOuterLeft[i]*np.cos(phis[i])
        x.append(X)
        y.append(Y)
    for (i,cyl) in enumerate(cylOuterLeft):
        cyl.location = (x[i],y[i],-bigLeftZ)
        cyl.keyframe_insert(data_path="location")
    #make rodes dynamic
    for i in range(0,3):
        middle = Vector([x[i+1]+x[i],y[i+1]+y[i],-2*bigLeftZ])*0.5
        outerRodesLeft[i].location=middle
        outerRodesLeft[i].rotation_euler[2]=-phis[i+1]
        outerRodesLeft[i].keyframe_insert(data_path="location")
        outerRodesLeft[i].keyframe_insert("rotation_euler")

for (k,line) in enumerate(outerRightData):
    bpy.context.scene.frame_set(pause+k)
    line = line.split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    #calculate positions from the angles
    X=pivot[0]
    Y=pivot[1]
    x=[]
    y=[]
    for i in range(0,4):
        X=X-lOuterRight[i]*np.sin(phis[i])
        Y=Y-lOuterRight[i]*np.cos(phis[i])
        x.append(X)
        y.append(Y)
    for (i,cyl) in enumerate(cylOuterRight):
        cyl.location = (x[i],y[i],-bigRightZ)
        cyl.keyframe_insert(data_path="location")
    #make rodes dynamic
    for i in range(0,3):
        middle = Vector([x[i+1]+x[i],y[i+1]+y[i],-2*bigRightZ])*0.5
        outerRodesRight[i].location=middle
        outerRodesRight[i].rotation_euler[2]=-phis[i+1]
        outerRodesRight[i].keyframe_insert(data_path="location")
        outerRodesRight[i].keyframe_insert("rotation_euler")
      
for (k,line) in enumerate(innerLeftData):
    bpy.context.scene.frame_set(pause+k)
    line = line.split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    #calculate positions from the angles
    X=pivot[0]
    Y=pivot[1]
    x=[]
    y=[]
    for i in range(0,4):
        X=X-lInnerLeft[i]*np.sin(phis[i])
        Y=Y-lInnerLeft[i]*np.cos(phis[i])
        x.append(X)
        y.append(Y)
    for (i,cyl) in enumerate(cylInnerLeft):
        cyl.location = (x[i],y[i],-3*bigLeftZ)
        cyl.keyframe_insert(data_path="location")
    #make rodes dynamic
    for i in range(0,3):
        middle = Vector([x[i+1]+x[i],y[i+1]+y[i],-6*bigLeftZ])*0.5
        innerRodesLeft[i].location=middle
        innerRodesLeft[i].rotation_euler[2]=-phis[i+1]
        innerRodesLeft[i].keyframe_insert(data_path="location")
        innerRodesLeft[i].keyframe_insert("rotation_euler")
    
for (k,line) in enumerate(innerRightData):
    bpy.context.scene.frame_set(pause+k)
    line = line.split(",")
    phis=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
    #calculate positions from the angles
    X=pivot[0]
    Y=pivot[1]
    x=[]
    y=[]
    for i in range(0,4):
        X=X-lInnerRight[i]*np.sin(phis[i])
        Y=Y-lInnerRight[i]*np.cos(phis[i])
        x.append(X)
        y.append(Y)
    for (i,cyl) in enumerate(cylInnerRight):
        cyl.location = (x[i],y[i],-2*bigRightZ)
        cyl.keyframe_insert(data_path="location")
    #make rodes dynamic
    for i in range(0,3):
        middle = Vector([x[i+1]+x[i],y[i+1]+y[i],-4*bigRightZ])*0.5
        innerRodesRight[i].location=middle
        innerRodesRight[i].rotation_euler[2]=-phis[i+1]
        innerRodesRight[i].keyframe_insert(data_path="location")
        innerRodesRight[i].keyframe_insert("rotation_euler")