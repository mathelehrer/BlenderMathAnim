import bpy
import numpy as np
import bmesh
import mathutils

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_circle_add(radius=0.5, enter_editmode=False, align='WORLD', location=(0, 0.5, 0), scale=(1, 1, 1))
bpy.ops.mesh.primitive_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 1, 0), scale=(1, 1, 1))


def extrude(object):
    me = object.data
    bm = bmesh.new()
    bm.from_mesh(me)
    faces = bm.faces[:]
    for face in faces:
        r = bmesh.ops.extrude_discrete_faces(bm, faces=[face])
        bmesh.ops.translate(bm, vec=mathutils.Vector((0,0,0.1)), verts=r['faces'][0].verts)
        bm.to_mesh(me)
        me.update()

for i in range(-10,11):
    scale = 1/(1/4*i*i+1/2)
    r = 1/4*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*radius+10),depth=0.1,fill_type='NGON',radius=r, enter_editmode=False, align='WORLD', location=(i/2*scale, 3/4*scale, 0), scale=(1, 1, 1))
    bpy.ops.translate(value(0,0,0.05)
    
for i in range(-10,11):
    scale = 1/(1/4*i*i-1/4*i+96/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*radius+10),depth=0.1,fill_type='NGON',radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 9/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.translate(value(0,0,0.05)
    
for i in range(-10,11):
    scale = 1/(1/4*i*i-1/4*i+240/256)
    r = 1/16*scale
    bpy.ops.mesh.primitive_cylinder_add(vertices=int(100*radius+10),depth=0.1,fill_type='NGON',radius=r, enter_editmode=False, align='WORLD', location=((2*i-1)/4*scale, 15/16*scale, 0), scale=(1, 1, 1))
    bpy.ops.translate(value(0,0,0.05)


