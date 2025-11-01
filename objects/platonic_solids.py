from itertools import combinations

import numpy as np
from mathutils import Vector

from geometry_nodes.geometry_nodes_modifier import UnfoldModifier
from interface.ibpy import create_mesh
from mathematics.mathematica.mathematica import tuples, choose, partition
from objects.bobject import BObject
from objects.cube import Cube
from utils.kwargs import get_from_kwargs

pi = np.pi

def orient_face(s, vertices):
    s = list(s)
    # compute normal and switch to indices if it is not pointing outward
    center = Vector()
    for i in s:
        center += vertices[i]
    center/=len(s)

    # compute outward pointing normal
    v0 = vertices[s[0]]-center
    v1 = vertices[s[1]]-center
    v2 = vertices[s[2]]-center
    u = v1-v0
    v = v2-v0
    normal = u.cross(v)
    if normal.dot(center)<0:
        normal = -normal
    normal.normalize()

    # perturb x_vec slightly to lift degeneracies
    x_vec = vertices[s[0]]-center
    y_vec = normal.cross(x_vec)
    x_vec.normalize()
    y_vec.normalize()
    s.sort(key=lambda x: np.angle((vertices[x]-center).dot(x_vec)+1j*(vertices[x]-center).dot(y_vec)))

    return s

class Tetrahedron(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name=self.get_from_kwargs('name','Tetrahedron')
        # vertices from wikipedia

        r2 = 2**0.5
        r3 = 3**0.5
        r6 = 6**0.5

        #vertices = [[1,0,-1/r2],[-1,0,-1/r2],[0,1,1/r2],[0,-1,1/r2]]
        vertices = [
            [1,-1/r3,-1/r6],
            [-1,-1/r3,-1/r6],
            [0,2/r3,-1/r6],
            [0,0,3/r6]]
        vertices = [Vector(v) for v in vertices]

        # find shortest edges
        edges = []
        for i,v in enumerate(vertices):
            for j,w in enumerate(vertices):
                if i>j:
                    d = (v-w).length
                    if d<2.1:
                        edges.append([i,j])
        print(len(edges))
        # find faces
        faces = []
        for i in range(len(edges)):
            for j in range(i+1,len(edges)):
                for k in range(j+1,len(edges)):
                    u,v,w = edges[i],edges[j],edges[k]
                    s = set(u + v + w)
                    if len(s)==3:
                        s = orient_face(s,vertices)
                        faces.append(s)

        print(len(faces))

        super().__init__(mesh=create_mesh(vertices,edges,faces),name=self.name,**kwargs)

class Octahedron(BObject):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'Octahedron')
        # vertices from wikipedia

        r2 = 2 ** 0.5
        r3 = 3 ** 0.5
        r6 = 6 ** 0.5

        # vertices = [[1,0,-1/r2],[-1,0,-1/r2],[0,1,1/r2],[0,-1,1/r2]]
        vertices = [
            [0,0,1/r2],
            [0,0,-1/r2],
            [1/2,1/2,0],
            [-1/2,1/2,0],
            [-1/2,-1/2,0],
            [1/2,-1/2,0]
        ]
        vertices = [Vector(v) for v in vertices]

        # find shortest edges
        edges = []
        for i, v in enumerate(vertices):
            for j, w in enumerate(vertices):
                if i > j:
                    d = (v - w).length
                    if d < 1.1:
                        edges.append([i, j])
        print(len(edges))
        # find faces
        faces = []
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                for k in range(j + 1, len(edges)):
                    u, v, w = edges[i], edges[j], edges[k]
                    s = set(u + v + w)
                    if len(s) == 3:
                        s = orient_face(s, vertices)
                        faces.append(s)

        print(len(faces))

        super().__init__(mesh=create_mesh(vertices, edges, faces), name=self.name, **kwargs)

class Icosahedron(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name=self.get_from_kwargs('name','Icosahedron')
        # vertices from wikipedia

        r5 = 5**0.5
        phi = (1 + r5) / 2

        vertices = [[0,1,phi],[0,-1,phi],[0,-1,-phi],[0,1,-phi],
                  [phi,0,1],[-phi,0,1],[-phi,0,-1],[phi,0,-1],
                  [1,phi,0],[-1,phi,0],[-1,-phi,0],[1,-phi,0]]
        vertices = [Vector(v) for v in vertices]

        # find shortest edges
        edges = []
        for i,v in enumerate(vertices):
            for j,w in enumerate(vertices):
                if i>j:
                    d = (v-w).length
                    if d<2.1:
                        edges.append([i,j])
        print(len(edges))
        # find faces
        faces = []
        for i in range(len(edges)):
            for j in range(i+1,len(edges)):
                for k in range(j+1,len(edges)):
                    u,v,w = edges[i],edges[j],edges[k]
                    s = set(u + v + w)
                    if len(s)==3:
                        s = orient_face(s,vertices)
                        faces.append(s)

        print(len(faces))

        super().__init__(mesh=create_mesh(vertices,edges,faces),name=self.name,**kwargs)

class Dodecahedron(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name=self.get_from_kwargs('name','Dodecahedron')
        # vertices from wikipedia
        r5 = 5**0.5
        phi = (1 + r5) / 2
        vertices = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],
                [0,phi,1/phi],[0,-phi,1/phi],[0,-phi,-1/phi],[0,phi,-1/phi],
                   [1/phi,0,phi],[-1/phi,0,phi],[-1/phi,0,-phi],[1/phi,0,-phi],
                    [phi,1/phi,0],[-phi,1/phi,0],[-phi,-1/phi,0],[phi,-1/phi,0]]
        vertices = [Vector(v) for v in vertices]

        # find shortest edges
        edges = []
        for i,v in enumerate(vertices):
            for j,w in enumerate(vertices):
                if i>j:
                    d = (v-w).length
                    if d<1.24:
                        edges.append([i,j])
        # find faces
        faces = []
        for i in range(len(edges)):
            for j in range(i+1,len(edges)):
                for k in range(j+1,len(edges)):
                    for l in range(k+1,len(edges)):
                        for m in range(l+1,len(edges)):
                            u,v,w,x,y = edges[i],edges[j],edges[k],edges[l],edges[m]
                            s = set(u + v + w + x + y)
                            if len(s)==5:
                                s = orient_face(s,vertices)
                                faces.append(s)

        sub_divide=get_from_kwargs(kwargs,'sub_divide',0)

        if sub_divide>0:
            middle_vertices = []
            sub_edges = []
            sub_faces = []
            for e in edges:
                v1,v2 = vertices[e[0]],vertices[e[1]]
                d = (v2-v1)/(sub_divide+1)
                first_i = e[0]
                first_v = vertices[first_i]
                for i in range(1,sub_divide+1):
                    v = first_v+d
                    vertices.append(v)
                    sub_edges.append([first_i,len(vertices)-1])
                    first_i = len(vertices)-1
                    first_v = v
                sub_edges.append([first_i,e[1]])

            # this probably only works for sub_divide==2
            # find all vertices with two edges that are not parallel and extend them into parallelogram
            n_old = len(vertices)

            new_vertices = []
            count = 0
            for i in range(len(vertices)):
                neighbors = []
                for e in sub_edges:
                    if i in e:
                        if e[0]==i:
                            neighbors.append(e[1])
                        else:
                            neighbors.append(e[0])
                # corner vertices have three neighbors
                if len(neighbors)==3:
                    # deal with every pair of neighbors
                    v0 = vertices[i]
                    for t in choose(neighbors, 2):
                        u=vertices[t[0]]-v0
                        v=vertices[t[1]]-v0
                        new_vertices.append(v0+u+v)
                        sub_edges.append([t[0],len(vertices)+count])
                        sub_edges.append([t[1],len(vertices)+count])
                        sub_faces.append(orient_face([i,t[0],t[1],len(vertices)+count],vertices+new_vertices))
                        count+=1
                else:
                    middle_vertices.append(i)
            vertices +=new_vertices

            # find edges of the pentagons
            penta_edges = []
            for i in range(n_old,len(vertices)):
                for j in range(i+1,len(vertices)):
                    v= vertices[i]
                    w= vertices[j]
                    d = (v-w).length
                    if d<0.7:
                        penta_edges.append([i,j])
            # we now found 120 edges since also the distance between neighbouring pentagons
            # a regular rhombicosidodecahedron is formed with pentagons, squares and triangles
            # we can remove all edges that form triangles (angle is 60 degrees)

            bad_edges = []
            for e in penta_edges:
                for f in penta_edges:
                    if len(set(e+f))==3:
                        common = set(e).intersection(set(f))
                        e2 = list(set(e).difference(common))[0]
                        f2 = list(set(f).difference(common))[0]
                        c = list(common)[0]

                        u = vertices[e2]-vertices[c]
                        v = vertices[f2]-vertices[c]

                        if 0.45< u.dot(v)/u.length/v.length< 0.55:
                            # remove edges with angles of 60 degree
                            bad_edges.append(e)
                            break

            penta_edges = [e for e in penta_edges if e not in bad_edges]

            penta_faces = []
            for i in range(len(penta_edges)):
                for j in range(i+1,len(penta_edges)):
                    for k in range(j+1,len(penta_edges)):
                        for l in range(k+1,len(penta_edges)):
                            for m in range(l+1,len(penta_edges)):
                                    s = set(penta_edges[i]+penta_edges[j]+penta_edges[k]+penta_edges[l]+penta_edges[m])
                                    if len(s)==5:
                                        s = orient_face(s,vertices)
                                        penta_faces.append(s)

            # deal with middle_vertices:
            middle_edges=[]
            for e in sub_edges:
                if e[0] in middle_vertices and e[1] in middle_vertices:
                    middle_edges.append(e)

            # find shortest distances to middle pentagons
            middle_faces=[]
            for e in middle_edges:
                for f in penta_faces:
                    more_vertex_indices = []
                    for fi in f:
                        for ei in e:
                            length = (vertices[ei] - vertices[fi]).length
                            if length <0.5:
                                more_vertex_indices.append(fi)
                    indices = e+more_vertex_indices
                    if len(indices)==4:
                        middle_faces.append(orient_face(indices,vertices))


        if sub_divide==0:
            super().__init__(mesh=create_mesh(vertices,edges,faces),name=self.name,**kwargs)
        else:
            super().__init__(mesh=create_mesh(vertices,sub_edges+penta_edges,sub_faces+penta_faces+middle_faces),name=self.name,**kwargs)


class SubdividedPentagon(BObject):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'Pentagon')

        # vertices from wikipedia
        vertices = [Vector([np.cos(2*pi/5*i),0,-np.sin(2*pi/5*i)]) for i in range(5)]
        edges =partition(list(range(5)),2,1,1)
        vertices = [Vector(v) for v in vertices]

        middle_vertices = []
        sub_edges = []
        sub_faces = []
        middle_squeeze=get_from_kwargs(kwargs,'middle_squeeze',0)
        for e in edges:
            v1,v2 = vertices[e[0]],vertices[e[1]]
            d = (v2-v1)/3
            first_i = e[0]
            first_v = vertices[first_i]
            for i in range(1,3):
                if i==1:
                    v = first_v+(1+middle_squeeze/2)*d
                else:
                    v=first_v+(1-middle_squeeze)*d
                vertices.append(v)
                sub_edges.append([first_i,len(vertices)-1])
                first_i = len(vertices)-1
                first_v = v
            sub_edges.append([first_i,e[1]])

        # this probably only works for sub_divide==2
        # find all vertices with two edges that are not parallel and extend them into parallelogram
        n_old = len(vertices)

        new_vertices = []
        count = 0
        for i in range(len(vertices)):
            neighbors = []
            for e in sub_edges:
                if i in e:
                    if e[0]==i:
                        neighbors.append(e[1])
                    else:
                        neighbors.append(e[0])

            # corner vertices have non-aligned neighbors
            v0 = vertices[i]
            u = vertices[neighbors[0]]-v0
            v = vertices[neighbors[1]]-v0
            if abs(u.dot(v)/u.length/v.length)<0.9:
                new_vertices.append(v0+u+v)
                sub_edges.append([neighbors[0],len(vertices)+count])
                sub_edges.append([neighbors[1],len(vertices)+count])
                sub_faces.append(orient_face([i,neighbors[0],neighbors[1],len(vertices)+count],vertices+new_vertices))
                count+=1
            else:
                middle_vertices.append(i)
        vertices +=new_vertices

        # internal pentagon
        penta_edges = []
        distances = {}
        for i in range(n_old,len(vertices)):
            for j in range(i+1,len(vertices)):
                v= vertices[i]
                w= vertices[j]
                dist = (v-w).length
                distances[(i,j)] = dist

        min_dist = list(distances.values())
        min_dist.sort()
        min_dist = min_dist[0]

        for key,val in distances.items():
            if val<1.1*min_dist:
                penta_edges.append(list(key))

        penta_face = [orient_face(list(range(n_old,len(vertices))),vertices)]

        # deal with middle_vertices:
        middle_edges=[]
        for e in sub_edges:
            if e[0] in middle_vertices and e[1] in middle_vertices:
                middle_edges.append(e)

        # find shortest distances to middle pentagon
        middle_faces=[]
        for e in middle_edges:
            more_vertex_indices = []
            for pe in penta_edges:
                ve = vertices[e[1]]-vertices[e[0]]
                vp = vertices[pe[1]]-vertices[pe[0]]
                if ve.cross(vp).length<0.005:
                    more_vertex_indices.append(pe[0])
                    more_vertex_indices.append(pe[1])
                    break
            indices = e+more_vertex_indices
            middle_faces.append(orient_face(indices,vertices))

        super().__init__(mesh=create_mesh(vertices,sub_edges+penta_edges,sub_faces+penta_face+middle_faces),name=self.name,**kwargs)
