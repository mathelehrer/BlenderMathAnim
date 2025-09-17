# we generate the full coxeter group of type H3
# see https://en.wikipedia.org/wiki/Coxeter_group#Table_of_all_Coxeter_groups
import numpy as np
from mathutils import Vector

from mathematics.geometry.cell600 import QR5, FTensor, FMatrix, FVector


class CoxH3:
    def __init__(self):

        zero = QR5.from_integers(0,1,0,1)
        one = QR5.from_integers(1,1,0,1)
        half = QR5.from_integers(1,2,0,1)
        two = QR5.from_integers(2,1,0,1)

        # normal vectors
        normals = [
                    FVector([one,zero,zero]),
                   FVector([zero,one,zero]),
                   FVector([QR5.from_integers(1,2,0,1),QR5.from_integers(1,4,1,4),QR5.from_integers(-1,4,1,4)]),
                   ]
        # normal vectors
        normals = [
                    FVector([one,zero,zero]),
                   FVector([zero,one,zero]),
                   FVector([QR5.from_integers(1,2,0,1),QR5.from_integers(1,4,1,4),QR5.from_integers(-1,4,1,4)]),
                   ]

        self.normals = normals

        # the generators of H3 are given by
        identity =  FMatrix([[one,zero,zero],[zero,one,zero],[zero,zero,one]])
        generators=[identity-(n*n)-(n*n) for n in normals]
        # cast to matrix for proper matrix multiplication
        generators = [FMatrix(g.components) for g in generators]

        # generators = [
        #     FMatrix([[-one,zero,zero],[zero,one,zero],[zero,zero,one]]),
        #     FMatrix([[one,zero,zero],[zero,-one,zero],[zero,zero,one]]),
        #     FMatrix([[half,QR5.from_integers(-1,4,-1,4),QR5.from_integers(1,4,-1,4)],
        #              [QR5.from_integers(-1,4,-1,4),QR5.from_integers(1,4,-1,4),-half],
        #              [QR5.from_integers(1,4,-1,4),-half,QR5.from_integers(1,4,1,4)]])]

        [print(g) for g in generators]

        # generate the group from the generators
        new_elements =generators
        elements = set(generators)
        while len(new_elements)>0:
            next_elements = []
            for h in new_elements:
                for g in generators:
                    element = g*h
                    if element not in elements:
                        next_elements.append(element)
                        elements.add(element)
                        # print(len(elements),end=" ")
            new_elements = next_elements
            # print(len(new_elements))
        self.elements = elements

    def get_point_cloud(self,seed=[1,1,-1]):
        seed = FVector.from_vector(seed)
        point_cloud = [element@seed for element in self.elements]
        return point_cloud

    def get_real_point_cloud(self, seed=[1, 1, -1]):
        return [p.real() for p in self.get_point_cloud(seed)]

    def get_edges(self,seed=[1,1,-1]):
        point_cloud = self.get_point_cloud(seed)
        # find minimum distance between two points
        min = np.inf
        min_dist = None

        for i in range(len(point_cloud)):
            for j in range(i+1,len(point_cloud)):
                dist = (point_cloud[i]-point_cloud[j]).norm()
                dist_real = dist.real()
                if dist_real<min:
                    min = dist_real
                    min_dist=dist

        edges = []
        for i in range(len(point_cloud)):
            for j in range(i+1,len(point_cloud)):
                dist = (point_cloud[i]-point_cloud[j]).norm()
                if dist==min_dist:
                    edges.append([i,j])
        print(len(edges),"edges")
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

    def walk_face(self,vertices,edge_map,path):
        """
        backtracking algorithm to find a closed loop in the graph

        """
        max_len=10 # Don't walk huge paths
        visited = set(path)

        def recurse(current_path):
            if len(current_path)>max_len:
                return None

            current=current_path[-1]
            for neighbor in edge_map[current]:
                if neighbor == current_path[0] and len(current_path)>2:
                    # closed loop, check coplanarity
                    points = [vertices[i] for i in current_path]
                    if self.is_coplanar(points):
                        return current_path
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                result = recurse(current_path+[neighbor])
                if result:
                    return result
                visited.remove(neighbor) # remove neighbor that didn't lead to a loop, should not be needed in our situation, since all paths are loops
            return None

        return recurse(path)



    def find_faces(self,vertices,edges):

        # undirected graph
        from collections import defaultdict
        # construct a map that generates an empty list for each key as default
        edge_map = defaultdict(list)
        for a,b in edges:
            edge_map[a].append(b)
            edge_map[b].append(a)

        faces ={} # prepare a dictionary that stores a canonical representation of a face and the proper index ordering
        visited = set()
        count = 0
        success=0
        for start in range(len(vertices)):
            for neighbor in edge_map[start]:
                if (start,neighbor) in visited or (neighbor,start) in visited:
                    continue # skip the rest in this case

                # attempt to walk in a cycle from (start->neighbor)
                path = [start,neighbor]
                print(count,". attempt",path,end=" ")
                face = self.walk_face(vertices,edge_map,path)
                count+=1
                if face:
                    real_ordering = face.copy()
                    # normalize and store
                    canonical = tuple(sorted(face))
                    if canonical not in faces:
                        faces[canonical]=real_ordering
                        success+=1
                        print("... success: ",success);

                visited.add((start,neighbor))
        print(len(faces),"faces")
        return list(faces.values())



    def get_faces(self,seed=[1,1,-1]):
        point_cloud = self.get_point_cloud(seed)
        edges = self.get_edges(seed)
        faces = self.find_faces(point_cloud,edges)
        return faces


    def get_normals(self):
        return [n.real() for n in self.normals]



