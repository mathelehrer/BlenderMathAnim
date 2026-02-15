# we generate the full coxeter group of type A3
# see https://en.wikipedia.org/wiki/Coxeter_group#Table_of_all_Coxeter_groups
import os
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from interface.ibpy import Vector
from mathematics.geometry.meshface import MeshFace

from mathematics.geometry.field_extensions import QR, FTensor, FMatrix, FVector

COXA3_SEEDS= {
    "TETRA":FVector.parse("[1, 0, -1/2*r2]"),
    "OCTA":FVector.parse("[0, 0,r2]"),
    "TRUNC_TETRA":FVector.parse("[1, 0, -3/2*r2]"),
    "CUBOCTA":FVector.parse("[1,1,-r2]"),
    "TRUNC_OCTA":FVector.parse("[1, 1,-2*r2]"),
}

COXA3_SIGNATURES= {
    "TETRA":(1,0,0),
    "OCTA":(0,0,1),
    "TRUNC_TETRA":(1,0,-1),
    "CUBOCTA":(1,1,0),
    "TRUNC_OCTA":(1,1,-1)
}

COXA3_TYPES={
    (1,0,0):"TETRA",
    (0,0,1):"OCTA",
    (1,0,-1):"TRUNC_TETRA",
    (1,1,0):"CUBOCTA",
    (1,1,-1):"TRUNC_OCTA"
}

# predefined seeds computed in video_CoxH4/mathematica/CoxH3.nb
PATH = "../mathematics/geometry/data/"
class CoxA3:
    def __init__(self,path = None):

        self.name = "coxA3"
        if path is None:
            self.path = PATH
        else:
            self.path = path

        zero = QR.from_integers(0,1,0,1,2,"r2")
        one = QR.from_integers(1,1,0,1,2,"r2")
        half = QR.from_integers(1,2,0,1,2,"r2")

        # normal vectors
        normals = [
                    FVector([one,zero,zero]),
                   FVector([zero,one,zero]),
                   FVector([half,half,QR.from_integers(0,1,1,2,2,"r2")]),
                   ]

        self.normals = normals

        # the generators of H3 are given by
        identity =  FMatrix([[one,zero,zero],[zero,one,zero],[zero,zero,one]])
        generators=[identity-(n*n)-(n*n) for n in normals]
        # cast to matrix for proper matrix multiplication
        self.generators = [FMatrix(g.components) for g in generators]
        # [print(g) for g in self.generators]

        # load or generate the group from the generators
        if not os.path.exists(os.path.join(self.path, self.name+"_elements.dat")):
            self.elements = self.generate_elements()
        else:
            self.elements = self.read_elements(self.name+"_elements.dat")

    def read_elements(self,filname):
        data = []
        with open(os.path.join(self.path,filname),"r") as f:
            for line in f:
                data.append(FMatrix.parse(line.strip()))
        return data

    def read_points(self, filname):
        data = []
        with open(os.path.join(self.path, filname), "r") as f:
            for line in f:
                data.append(FVector.parse(line.strip()))
        return data

    def read_edges(self, filename):
        edges = []
        with open(os.path.join(self.path, filename), "r") as f:
            for line in f:
                edges.append(eval(line))
        return edges

    def save(self,elements,filename):
        with open(os.path.join(self.path,filename),"w") as f:
            for e in elements:
                f.write(str(e)+"\n")

    def generate_elements(self):
        # generate the group from the generators
        filename=self.name+"_elements.dat"
        if os.path.exists(os.path.join(self.path,filename)):
            return self.read_elements(filename)

        new_elements = self.generators
        elements = set(self.generators)
        while len(new_elements) > 0:
            next_elements = []
            for h in new_elements:
                for g in self.generators:
                    element = g * h
                    if element not in elements:
                        next_elements.append(element)
                        elements.add(element)
                        print(len(elements), end=" ")
            new_elements = next_elements
            print(len(new_elements))
        self.save(elements, self.name+"_elements.dat")
        return elements

    def get_radius(self,signature=[1,1,-1]):
        return self.get_real_point_cloud(signature)[0].length

    def point_cloud(self, signature=[1, 1, 1], start=None):
        if isinstance(signature, tuple):
            signature = list(signature)
        filename = self.name+"_points" + str(signature).replace(",", "_") + ".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            if start is None:
                start = COXA3_SEEDS.get(COXA3_TYPES[tuple(signature)], None)
                if start is None:
                    raise "No start point given for point cloud generation."
            point_cloud = set()
            for element in self.elements:
                new_element = element @ start
                point_cloud.add(new_element)
            self.save(point_cloud, filename)
        else:
            point_cloud = self.read_points(filename)
            # print("point cloud read from file")
        return list(point_cloud)

    def get_real_point_cloud(self, signature=[1, 1, -1]):
        if isinstance(signature, tuple):
            signature = list(signature)
        return [p.real() for p in self.point_cloud(signature)]

    def find_edges_for_chunk(self,min_dist,point_cloud,rng):
        print("Start process for range: ",rng)
        edges=[]

        for i in range(*rng):
            for j in range(i + 1, len(point_cloud)):
                dist = (point_cloud[i] - point_cloud[j]).norm()
                if dist == min_dist:
                    edges.append([i, j])

        return edges

    def get_edges(self, signature=[1, 1, 1]):
        filename= self.name +"_edges" + str(signature).replace(",", "_") + ".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            point_cloud = self.point_cloud(signature)
            # find minimum distance between two points
            minimum = np.inf
            min_dist = None

            for j in range(1,len(point_cloud)):
                dist = (point_cloud[0]-point_cloud[j]).norm()
                dist_real = dist.real()
                if dist_real<minimum:
                    minimum = dist_real
                    min_dist=dist

            edges = []

            if len(point_cloud)>1000:
                size = int(len(point_cloud)/os.cpu_count())
                chunks = [[i,min(i+size,len(point_cloud))] for i in range(0,len(point_cloud),size)]

                worker = partial(self.find_edges_for_chunk,min_dist,point_cloud)
                with Pool(processes=os.cpu_count()) as pool:
                    for res in pool.imap_unordered(worker,chunks,chunksize=1):
                        edges = edges + res

            else:
                edges = self.find_edges_for_chunk(min_dist,point_cloud,[0,len(point_cloud)])

            self.save(edges,filename)
        else:
            edges = self.read_edges(filename)
            print("edge data read from file")

        return edges

    def is_coplanar(self,points):
        if len(points)<4:
            return True
        normal = (points[1]-points[0]).cross(points[2]-points[0])
        for i in range(3,len(points)):
            test = normal.dot(points[i] - points[0]).real()
            if test !=0:
                return False
        return True

    def walk_face(self, vertices, edge_map, path, max_len=10):
        """
        Backtracking algorithm to find all closed loops in the graph that start with `path`.

        - vertices: list/array of points; vertices[i] gives coordinates of vertex i
        - edge_map: dict {vertex: iterable_of_neighbor_vertices}
        - path: starting vertex sequence (prefix) to be extended
        - max_len: cap on path length during search
        - skip_backtrack: if True, avoid immediately returning to the previous vertex
        """
        if not path or len(path) < 1:
            return []

        start = path[0]
        all_cycles = []
        seen_signatures = set()  # to deduplicate cycles (same order or reverse)

        def record_cycle(cycle):
            # Normalize representation for deduplication:
            # All cycles start at `start`, so dedup by the ordered sequence starting at start,
            # and also consider its reverse as the same cycle.
            forward = tuple(cycle)
            backward = tuple([start] + list(reversed(cycle[1:])))
            sig = min(forward, backward)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                all_cycles.append(list(forward))

        def dfs(current_path):
            if len(current_path) > max_len:
                return

            current = current_path[-1]

            for neighbor in edge_map[current]:
                # optionally avoid immediate backtrack to previous vertex
                if len(current_path) >= 2 and neighbor == current_path[-2]:
                    continue

                if neighbor == start:
                    # Found a cycle that starts with `path[0]` and returns to start
                    if len(current_path) >= 3:
                        points = [vertices[i] for i in current_path]
                        if self.is_coplanar(points):
                            # Keep cycle as [v0, v1, ..., vk] (start not repeated at end)
                            record_cycle(current_path[:])
                    # Even after finding a cycle, continue exploring other neighbors
                    continue

                # Do not revisit vertices already on the current path (prevents sub-cycles)
                if neighbor in current_path:
                    continue

                dfs(current_path + [neighbor])

        dfs(path)
        return all_cycles

    def is_boundary(self, face,vertices):
        """
        check, whether all points of the solid lie on one side of the face
        """
        signs = set()
        signs.add(0)

        normal = (vertices[face[1]]-vertices[face[0]]).cross(vertices[face[2]]-vertices[face[0]])

        for v in vertices:
            dot = normal.dot(v - vertices[face[0]]).real()
            if dot == 0:
                signs.add(0)
            elif dot < 0:
                signs.add(-1)
            else:
                signs.add(1)
            if len(signs) > 2:
                return False
        return True


    def find_faces(self,vertices,edges):
        # undirected graph
        # construct a map that generates an empty list for each key as default
        edge_map = defaultdict(list)
        for a,b in edges:
            edge_map[a].append(b)
            edge_map[b].append(a)

        faces ={} # prepare a dictionary that stores a canonical representation of a face and the proper index ordering
        visited = set()

        success=0
        for start in range(len(vertices)):
            for neighbor in edge_map[start]:
                if (start,neighbor) in visited or (neighbor,start) in visited:
                    continue # skip the rest in this case

                # attempt to walk in a cycle from (start->neighbor)
                path = [start,neighbor]
                new_faces = self.walk_face(vertices,edge_map,path)
                for face in new_faces:
                    real_ordering = face.copy()
                    # normalize and store
                    canonical = tuple(sorted(face))
                    if self.is_boundary(canonical,vertices) and canonical not in faces:
                        faces[canonical]=real_ordering
                        success+=1

                visited.add((start,neighbor))
        print(len(faces),"faces")
        return list(faces.values())

    def get_faces(self, signature=[1, 1, -1]):
        if isinstance(signature, tuple):
            signature=list(signature)

        faces = list()
        filename = self.name+"_faces" + str(signature).replace(",", "_") + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    faces.append(eval(line))

            return faces

        point_cloud = self.point_cloud(signature)
        edges = self.get_edges(signature)
        faces = self.find_faces(point_cloud,edges)
        # rearrange indices for an outward pointing normal

        for idx in range(len(faces)):
            face = faces[idx]
            # create normal from the first three points
            # show that it is aligned with the center of the face,
            # otherwise reverse orientation of face

            normal = (point_cloud[face[1]]-point_cloud[face[0]]).cross(point_cloud[face[2]]-point_cloud[face[0]])
            center = Vector((0,0,0))
            for i in face:
                center+=point_cloud[i].real()
            center/=len(face)

            if normal.real().dot(center)<0:
                faces[idx]=face[::-1]

        self.save(faces,filename)
        return faces

    def get_normals(self):
        return [n.real() for n in self.normals]

    def get_faces_in_conjugacy_classes(self,signature=None):
        """
        sort faces into their conjugacy classes
        the representatives of the classes are the faces that contain the vertex
        with index 0
        >>> group = CoxA3("data")
        >>> cc = group.get_faces_in_conjugacy_classes(COXA3_SIGNATURES["TRUNC_OCTA"])
        >>> print(cc)
        {Face([0, 16, 6, 5, 21, 1]): {Face([3, 19, 8, 9, 23, 4]), Face([0, 16, 6, 5, 21, 1]), Face([2, 20, 22, 18, 15, 10]), Face([7, 11, 17, 14, 13, 12])}, Face([0, 1, 22, 18, 13, 14]): {Face([0, 1, 22, 18, 13, 14]), Face([2, 20, 21, 5, 9, 8]), Face([3, 19, 10, 15, 12, 7]), Face([4, 11, 17, 16, 6, 23])}, Face([0, 14, 17, 16]): {Face([2, 10, 19, 8]), Face([1, 21, 20, 22]), Face([0, 14, 17, 16]), Face([3, 7, 11, 4]), Face([5, 6, 23, 9]), Face([12, 13, 18, 15])}}

        """
        if signature is None:
            raise "function must be called with signature"
        faces = self.get_faces(signature)
        vertices = self.point_cloud(signature)

        classes = {}

        # get all faces of the first vertex
        first_faces = []
        for face in faces:
            if 0 in face:
                face=MeshFace(face)
                first_faces.append(face)
                classes[face]={face}

        for first_face in first_faces:
            for element in self.elements:
                mapped_face = []
                for face_index in first_face:
                    vertex = vertices[face_index]
                    target = element @ vertex
                    mapped_face.append(vertices.index(target))
                classes[first_face].add(MeshFace(mapped_face))

        return classes




