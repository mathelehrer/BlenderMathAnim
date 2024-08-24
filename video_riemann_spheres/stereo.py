import bpy, bmesh
from mathutils import Vector

###################got this from mesh_looptools.py#################

# input: bmesh, output: dict with the edge-key as key and face-index as value
def dict_edge_faces(bm):
    edge_faces = dict([[edgekey(edge), []] for edge in bm.edges 
                                        if not edge.hide])
    for face in bm.faces:
        if face.hide:
            continue
        for key in face_edgekeys(face):
            edge_faces[key].append(face.index)

    return(edge_faces)


# return the edgekey ([v1.index, v2.index]) of a bmesh edge
def edgekey(edge):
    return(tuple(sorted([edge.verts[0].index, edge.verts[1].index])))

# returns the edgekeys of a bmesh face
def face_edgekeys(face):
    return([tuple(sorted([edge.verts[0].index, edge.verts[1].index])) for \
        edge in face.edges])


###################################################################


def stereo_proj(scale_factor=.9):

    # Get the active mesh
    me = bpy.context.object.data
    # Get a BMesh representation
    bm = bmesh.new()   
    bm.from_mesh(me)   

    #this gets border edges which should be extruded to make  a solid
    borders = [ x for x,y in dict_edge_faces(bm).items()
                 if len(y) < 2 ]
    
    borders.sort()             
              
    for v in bm.verts:
        x,y,z = v.co[:]
        r2 = x*x + y*y
        nv = 1./(r2 + 1) * Vector((2*x, 2*y, (r2 - 1)))
        v.co = nv
        
    offset = len(bm.verts)
    
    #take a copy as we are adding to bm.verts
    #otherwise hangs with an infinite loop
    copy_verts =  bm.verts[:]
    for pt in copy_verts:   
        bm.verts.new( scale_factor*pt.co )
    
    #hash this to make clearer
    vvs = bm.verts
    vvs.ensure_lookup_table()

    #new layer of faces concentric to original faces
    new_faces  = [ ( vvs[vv.index + offset] for vv in ff.verts[:] )
                                           for ff in bm.faces]
 
    #add faces that go between the two layers
    extruded_edge_faces =  [ (vvs[a], vvs[b], vvs[ b + offset ], vvs[a + offset])
                                                                    for a,b in borders ]
                                                                    
    #add the new faces to the mesh
    new_faces.extend(extruded_edge_faces)
    for ff in new_faces:
        bm.faces.new(ff)
           
    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  


stereo_proj()