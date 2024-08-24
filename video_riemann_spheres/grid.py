
import bpy, bmesh
import numpy as np 
from bpy_extras import object_utils

def apply_boolean(obj_A, obj_B, bool_type='INTERSECT'):
    
    print('+++',obj_A, obj_B)
    bpy.ops.object.select_all(action='DESELECT')
    obj_A.select_set(True)
    
    bpy.context.view_layer.objects.active = obj_A
    bpy.ops.object.modifier_add(type='BOOLEAN')

    mod = obj_A.modifiers
    mod[0].name = obj_A.name + bool_type
    mod[0].object = obj_B
    mod[0].operation = bool_type

    bpy.ops.object.modifier_apply(modifier=mod[0].name)

def chessboard(npts=24,
              size = 2):
                  
    me = bpy.data.meshes.new("Chess")
    ob = bpy.data.objects.new("Chess", me)
    print(bpy.context.scene.objects)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = [0,0,0]
    
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(me)   # fill it in from a Mesh

    xs = np.linspace(-size,size,npts)
    ys = xs[:]

    tt = []
    for x in xs:        
        tt.append([])
        for y in ys:
            tt[-1].append(bm.verts.new((x,y,0)))
        
    for i,row in enumerate(tt[:-1]):
        for j,elt in enumerate(row[:-1]):
            #skip every fourth face
            if i*j % 2 == 1: continue
            bm.faces.new([tt[i][j],tt[i][j+1],tt[i+1][j+1],tt[i+1][j]])
                
    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access
    #be polite: make active and return a reference
    bpy.context.view_layer.objects.active = ob
    return ob

def cube_slicer(cube_scale=.85,
                z_offset=1):
                    
    '''splits selected mesh into two parts 
    using boolean operations'''
    
    def clean_up(cut_off=.1):
        '''cleans up after boolean operations
        deleting any extraneous vertices out of 
        the xy-plane
        '''
        ob = bpy.context.object
        bpy.ops.object.mode_set(mode = 'EDIT')
        me = ob.data
        bm = bmesh.from_edit_mesh(me)

        verts = [v for v in bm.verts 
                if abs(v.co[2]) > cut_off]

        bmesh.ops.delete(bm, geom=verts, context='VERTS')
        bmesh.update_edit_mesh(me)
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    xx = bpy.context.active_object 
    xx.select_set(True)
    bpy.ops.object.duplicate()
    yy = bpy.context.view_layer.objects.active
    yy.location = [0,0,z_offset]

    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.mesh.primitive_cube_add(location=[0,0,0])
    cc = bpy.context.view_layer.objects.active
    cc.scale = [cube_scale,cube_scale,1]

    bpy.ops.object.duplicate()
    dd = bpy.context.view_layer.objects.active
    dd.location = [0,0,z_offset]

    apply_boolean(xx, cc, bool_type='INTERSECT')
    clean_up()
    apply_boolean(yy, dd, bool_type='DIFFERENCE')
    clean_up()

    #get rid of the cubes
    bpy.ops.object.select_all(action='DESELECT')
    for x in [cc,dd]:
        x.select_set(True)
    
    bpy.ops.object.delete() 
        

chessboard()

cube_slicer(cube_scale=.85)