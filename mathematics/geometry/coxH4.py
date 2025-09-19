# we generate the full coxeter group of type H4
# see https://en.wikipedia.org/wiki/Coxeter_group#Table_of_all_Coxeter_groups
import os

import numpy as np
from mathematics.geometry.field_extensions import QR5, FMatrix, FVector

PATH = "data/"
zero = QR5.from_integers(0,1,0,1)
one = QR5.from_integers(1,1,0,1)
half = QR5.from_integers(1,2,0,1)
two = QR5.from_integers(2,1,0,1)

class CoxH4:
    def __init__(self,path=None):
        """

        """
        if path is None:
            self.path=PATH
        else:
            self.path = path
        zero = QR5.from_integers(0,1,0,1)
        one = QR5.from_integers(1,1,0,1)
        half = QR5.from_integers(1,2,0,1)
        two = QR5.from_integers(2,1,0,1)

        # normal vectors: Computed in CoxH4.nb
        normals = [
                    FVector([one,zero,zero,zero]),
                    FVector([half,half,half,half]),
                    FVector([zero, one,zero, zero]),
                   FVector([zero,
                            QR5.from_integers(1,4,1,4),
                            -half,
                            QR5.from_integers(1,4,-1,4)
                            ]
                           ),
                   ]

        self.normals = normals

        # the generators of H4 are given by
        identity =  FMatrix([[one,zero,zero,zero],[zero,one,zero,zero],[zero,zero,one,zero],[zero,zero,zero,one]])
        generators=[identity-(n*n)-(n*n) for n in normals]
        # cast to matrix for proper matrix multiplication
        self.generators = [FMatrix(g.components) for g in generators]

        if not os.path.exists(os.path.join(self.path, "coxH4_elements.dat")):
            self.elements = self.generate_elements()
        else:
            self.elements = self.read_elements("coxH4_elements.dat")

    def read_elements(self,filname):
        data = []
        with open(os.path.join(self.path,filname),"r") as f:
            for line in f:
                data.append(FMatrix.parse(line.strip()))
        return data

    def read_points(self,filname):
        data = []
        with open(os.path.join(self.path,filname),"r") as f:
            for line in f:
                data.append(FVector.parse(line.strip()))
        return data

    def read_edges(self,filename):
        edges = []
        with open(os.path.join(self.path,filename),"r") as f:
            for line in f:
                edges.append(eval(line))
        return edges

    def save(self,elements,filename):
        with open(os.path.join(self.path,filename),"w") as f:
            for e in elements:
                f.write(str(e)+"\n")

    def generate_elements(self):
        # generate the group from the generators
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
        self.save(elements, "coxH4_elements.dat")
        return elements

    def get_radius(self,seed=[1,1,-1]):
        return self.get_real_point_cloud(seed)[0].length

    def point_cloud(self,signs =[1,1,1,1]):
        filename = "coxH4_points" + str(signs).replace(",", "_") + ".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            if signs==[1,1,1,1]:
                # 600 cell (120 vertices)
                seed = FVector([one,one,-one,one])
            elif signs == [-1,1,1,1]:
                # 2400 vertices
                seed = FVector([
                    QR5.from_integers(3,1,-1,1),
                    QR5.from_integers(-3,1,1,1),
                    QR5.from_integers(-1,1,-1,1),
                    QR5.from_integers(-7,1,1,1)
                ])
            elif signs==[1,-1,1,1]:
                # 7200 vertices
                seed = FVector([
                    QR5.from_integers(-3, 1, 1, 1),
                    QR5.from_integers(-3, 1, 1, 1),
                    QR5.from_integers(7, 1, -5, 1),
                    QR5.from_integers(5, 1, 1, 1)
                ])
            elif signs == [1, 1,-1, 1]:
                #  14400 vertices
                seed = FVector([
                    QR5.from_integers(-3, 1, 1, 1),
                    QR5.from_integers(3, 1, -1, 1),
                    QR5.from_integers(3, 1, 1, 1),
                    QR5.from_integers(-9, 1, -1, 1)
                ])
            elif signs == [1, 1, 1, -1]:
                #  3600 vertices
                seed = FVector([
                    QR5.from_integers(-3, 1, 1, 1),
                    QR5.from_integers(-3, 1, 1, 1),
                    QR5.from_integers(-5, 1, -1, 1),
                    QR5.from_integers(5, 1, 1, 1)
                ])

            point_cloud = set()
            for element in self.elements:
                point_cloud.add(element@seed)
            self.save(point_cloud,filename)
        else:
            point_cloud = self.read_points(filename)
            print("point cloud read from file")
        return list(point_cloud)

    def get_real_point_cloud(self, seed=[1, 1, -1]):
        return [p.real() for p in self.get_point_cloud(seed)]

    def get_edges(self,seed=[1,1,1,1]):
        filename="coxH4_edges"+str(seed).replace(",","_")+".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            point_cloud = self.point_cloud(seed)
            # find minimum distance between two points
            min = np.inf
            min_dist = None


            for j in range(1,len(point_cloud)):
                dist = (point_cloud[0]-point_cloud[j]).norm()
                dist_real = dist.real()
                if dist_real<min:
                    min = dist_real
                    min_dist=dist

            edges = []
            for i in range(len(point_cloud)):
                print(i)
                for j in range(i+1,len(point_cloud)):
                    dist = (point_cloud[i]-point_cloud[j]).norm()
                    if dist==min_dist:
                        edges.append([i,j])
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
                new_faces = self.walk_face(vertices,edge_map,path)
                for face in new_faces:
                    real_ordering = face.copy()
                    # normalize and store
                    canonical = tuple(sorted(face))
                    if canonical not in faces:
                        faces[canonical]=real_ordering
                        success+=1

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

    def get_geometry(self,seed):
        return [self.point_cloud(seed),self.get_edges(seed)]

if __name__ == '__main__':
    coxH4 = CoxH4()
    signs = [-1,1,1,1]
    print(len(coxH4.point_cloud(signs)))
    print(len(coxH4.get_edges(signs)))