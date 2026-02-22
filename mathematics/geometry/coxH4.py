# we generate the full coxeter group of type H4
# see https://en.wikipedia.org/wiki/Coxeter_group#Table_of_all_Coxeter_groups
from __future__ import annotations

import multiprocessing
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np

from mathematics.geometry.meshface import MeshFace
from mathematics.geometry.field_extensions import QR, FMatrix, FVector, EpsilonTensor
from utils.string_utils import show_inline_progress_in_terminal

COXH4_SIGNATURES= {
    "x3o3o5o":[1,0,0,0], # 120 (tetra)
    "o3o3o5x":[0,0,0,1], # 600 (dodeca)
    "o3x3o5o": [0, 1, 0, 0],  # 720 ()
    "o3o3x5o":[0,0,1,0], # 1200  ()


    "x3x3o5o":[1,-1,0,0], # 1440 ()
    "x3o3o5x":[1,0,0,-1], # 2400 ()
    "o3o3x5x":[0,0,1,-1], # 2400 ()
    "o3x3o5x":[0,1,0,1], # 3600 ()
    "o3x3x5o": [0,1,-1,0],  # 3600 ()
    "x3o3x5o": [1,0,1,0],  # 3600 ()

    "x3x3x5o":[1,-1,1,0], # 7200 ()
    "x3x3o5x":[1,-1,0,-1], # 7200 ()
    "x3o3x5x":[1,0,1,-1], # 7200 ()
    "o3x3x5x":[0,1,-1,1], # 7200 ()

    "x3x3x5x":[1,-1,1,-1] # 14400 ()
}

PATH = "data/"
zero = QR.from_integers(0,1,0,1)
one = QR.from_integers(1,1,0,1)
half = QR.from_integers(1,2,0,1)
two = QR.from_integers(2,1,0,1)
epsilon = EpsilonTensor(4)
zero4 = FVector([zero,zero,zero,zero])


def center(key, point_cloud):
    c = FVector([zero,zero,zero,zero])
    for k in key:
        c = c + point_cloud[k]
    return c.real() / len(key)

def process_element(args):
    elements, first_cell, vertices, vertex2index = args
    mapped_cells = []
    for element in elements:
        mapped_cell = []
        for cell_index in first_cell:
            vertex = vertices[cell_index]
            target = element @ vertex
            mapped_cell.append(vertex2index[target])
        mapped_cells.append(tuple(sorted(mapped_cell)))
    return set(mapped_cells)


class CoxH4:
    def __init__(self,path=None):
        """

        """
        self.name = "coxH4"
        self.size = 14400
        if path is None:
            self.path=PATH
        else:
            self.path = path
        zero = QR.from_integers(0,1,0,1)
        one = QR.from_integers(1,1,0,1)
        half = QR.from_integers(1,2,0,1)
        two = QR.from_integers(2,1,0,1)

        # normal vectors: Computed in CoxH4.nb
        normals = [
                    FVector([one,zero,zero,zero]),
                    FVector([half,half,half,half]),
                    FVector([zero, one,zero, zero]),
                   FVector([zero,
                            QR.from_integers(1,4,1,4),
                            -half,
                            QR.from_integers(1,4,-1,4)
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

    def read_faces(self,filename):
        faces = []
        with open(os.path.join(self.path,filename),"r") as f:
            for line in f:
                faces.append(eval(line))
        return faces

    def save(self,elements,filename):
        with open(os.path.join(self.path,filename),"w") as f:
            for e in elements:
                f.write(str(e)+"\n")

    def save_dictionary(self, dictionary, filename):
        with open(os.path.join(self.path, filename), "w") as f:
            for key, val in dictionary.items():
                f.write(f"{key}->{val}\n")

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
        self.save(elements, "coxH4_elements.dat")
        return elements

    def get_radius(self,seed=[1,1,-1]):
        return self.get_real_point_cloud(seed)[0].length

    def get_point_cloud(self,signature =[1,1,1,1],start = None):
        filename = "coxH4_points" + str(signature).replace(",", "_") + ".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            if start is None:
                raise "No start point given for point cloud generation."
            point_cloud = set()
            for element in self.elements:
                point_cloud.add(element@start)
            self.save(point_cloud,filename)
        else:
            point_cloud = self.read_points(filename)
            # print("point cloud read from file")
        return list(point_cloud)

    def get_real_point_cloud(self, seed=[1, 1, -1]):
        return [p.real() for p in self.get_point_cloud(seed)]

    def find_edges_for_chunk(self,min_dist,point_cloud,rng):
        print("Start process for range: ",rng)
        edges=[]

        for i in range(*rng):
            for j in range(i + 1, len(point_cloud)):
                dist = (point_cloud[i] - point_cloud[j]).norm()
                if dist == min_dist:
                    edges.append([i, j])

        return edges


    def get_edges(self,seed=[1,1,1,1]):
        filename="coxH4_edges"+str(seed).replace(",","_")+".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            point_cloud = self.point_cloud(seed)
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

            size = int(len(point_cloud)/os.cpu_count())
            chunks = [[i,min(i+size,len(point_cloud))] for i in range(0,len(point_cloud),size)]

            worker = partial(self.find_edges_for_chunk,min_dist,point_cloud)
            with Pool(processes=os.cpu_count()) as pool:
                for res in pool.imap_unordered(worker,chunks,chunksize=1):
                    edges = edges + res

            self.save(edges,filename)
        else:
            edges = self.read_edges(filename)
            print("edge data read from file")

        return edges

    def get_normal_for_points(self,vertices,path):
        if len(path)<4:
            return None
        directions = [vertices[p]-vertices[path[0]] for p in path[1:]]
        tensor = directions[0]
        for d in directions[1:]:
            tensor = tensor *d
        n = epsilon.contract(tensor,axes=[[1,2,3],[0,1,2]])
        if n==zero4:
            return None
        else:
            return n

    def check_boundary(self, normal, point_cloud, path):
        """
        The idea behind this boundary check
        if all points of the point cloud lie on one side of the cell it is a boundary
        pos = None
        neg = None

        """
        boundary_points = path.copy()

        p0 = point_cloud[path[0]]
        pos = None
        neg = None

        for i in range(len(point_cloud)):
            if i != path:
                p = point_cloud[i]
                diff = p-p0
                dot = diff.dot(normal)
                if dot ==zero:
                    boundary_points.append(i)
                elif dot.real()<0:
                    neg = -1
                else:
                    pos = 1
                if pos is not None and neg is not None:
                    return False
        return boundary_points

    def get_cells_at_origin(self,point_cloud,edges):
        """
        we start at the first point of the point cloud and follow each possible edge until
        we reach a path length of 4
        for each path we check if it is part of a boundary and
        if so we store the normal vector and the indices that belong to the cell

        """

        edge_map = self.get_edge_map(edges)

        # we need at least four vertices to define a cell
        cell_normals = {}
        cell_indices = [0]
        old_text = ""
        for i,first_neighbor in enumerate(edge_map[0]):
            cell_indices.append(first_neighbor)
            text = "Find boundaries along the line of the "+str(i)+"/"+str(len(edge_map[0]))+" neighbors"
            show_inline_progress_in_terminal(text,old_text)
            old_text = text
            for second_neighbor in edge_map[first_neighbor]:
                if second_neighbor not in cell_indices:
                    cell_indices.append(second_neighbor)
                    for third_neighbor in edge_map[second_neighbor]:
                        if third_neighbor not in cell_indices:
                            cell_indices.append(third_neighbor)
                            normal = self.get_normal_for_points(point_cloud,cell_indices)
                            if normal is not None:
                                result =  self.check_boundary(normal, point_cloud, cell_indices)
                                if result:
                                    result.sort()
                                    index_tuple = tuple(set(result))
                                    # make normal point away from origin
                                    if point_cloud[index_tuple[0]].dot(normal).real()<0:
                                        normal = -normal
                                    cell_normals[index_tuple]=normal
                                    # print("success",index_tuple)
                            # print("tested",cell_indices)
                            cell_indices=cell_indices[:-1]
                    cell_indices=cell_indices[:-1]
            cell_indices=cell_indices[:-1]
        print()
        print(len(cell_normals)," cells found")

        return cell_normals

    def get_cells_from_chunk(self,point_cloud,key,val,chunk):
        cells = {}
        for i, elem in enumerate(chunk):


            new_normal = elem @ val
            new_indices = []
            for i in key:
                new_indices.append(point_cloud.index(elem @ point_cloud[i]))
            new_indices.sort()
            index_tuple = tuple(set(new_indices))
            if index_tuple not in cells:
                # make normal point away from origin
                if point_cloud[index_tuple[0]].dot(new_normal).real() < 0:
                    new_normal = -new_normal
                cells[index_tuple] = new_normal
                # print("added",index_tuple)
                # assert((point_cloud[index_tuple[0]]-point_cloud[index_tuple[-1]]).dot(new_normal)==zero)
        return cells

    def get_cells(self,seed=[1,1,1,1]):
        """
        we find the cells at the first point of the point cloud
        These cells are mapped by the active group elements to all other cells

        the active group elements are one representative for each point

        """

        cells = {}
        filename = "coxH4_cells" + str(seed).replace(",", "_") + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    parts = line.split("->")
                    key = eval(parts[0].strip())
                    val = FVector.parse(parts[1].strip())
                    cells[key] = val
                return cells

        point_cloud = self.get_point_cloud(seed)
        edges = self.get_edges(seed)
        print("Find cells at origin... ",end="")
        cells0 = self.get_cells_at_origin(point_cloud,edges)
        print("done")

        for key,val in cells0.items():
            cells[key]=val

        active_elements = {}
        active_elements[point_cloud[0]] = FMatrix.identity(4)
        for elem in self.generate_elements():
            if len(active_elements)==len(point_cloud):
                break
            point = elem@point_cloud[0]
            if point not in active_elements:
                active_elements[point] = elem

        # print("found all active elements")
        # for key,val in active_elements.items():
        #     print(key,"->",val)

        # now map the cells at the origin to all other cells
        active_elements = list(active_elements.values())

        for i,(key,val) in enumerate(cells0.items()):
            print("map cell",i)
            old_text = ""

            # parallel processing
            cpus = os.cpu_count()
            size= int(len(active_elements)/cpus)
            chunks = []
            for i in range(0,len(active_elements),size):
                chunks.append(active_elements[i:min(len(active_elements),i+size)])

            worker =partial(self.get_cells_from_chunk,point_cloud,key,val)

            with Pool(processes=cpus) as pool:
                for res in pool.imap_unordered(worker,chunks,chunksize=1):
                    for key,val in res.items():
                        if key not in cells:
                            cells[key] = val


        with open(os.path.join(self.path,filename),"w") as f:
            for key,val in cells.items():
                f.write(f"{key}->{val}\n")
                print(key, val, center(key, point_cloud))
        return cells

    def find_faces_of_cell_chunk(self,point_cloud,edge_map,chunk):
        """
        function to find the faces of a cell chunk
        intended for parallel processing
        """
        print("Start process for a chunk of cells...")
        faces=set()
        for i, (indices, cell_normal) in enumerate(chunk.items()):
            # compute faces for each cell
            reduced_edge_map = {}
            index_set = set(indices)

            for src, dest in edge_map.items():
                if src in indices:
                    reduced_edge_map[src] = set(dest).intersection(index_set)

            cell_faces = set(self.find_faces_of_cell(point_cloud, reduced_edge_map, cell_normal))
            faces = faces.union(cell_faces)
        return faces
        return []

    def get_faces(self,seed=[1,1,1,1]):
        faces = set()
        filename = "coxH4_faces" + str(seed).replace(",", "_") + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    faces.add(eval("Mesh"+line))
        else:
            point_cloud = self.point_cloud(seed)
            edges = self.get_edges(seed)
            edge_map=self.get_edge_map(edges)
            cells = self.get_cells(seed)

            cells=dict(sorted(cells.items())) # sort to get a balanced load on the worker threads

            cpus = os.cpu_count()
            size=int(len(cells)/cpus)
            items = list(cells.items())
            cell_parts = []
            for i in range(0,len(cells),size):
                dictionary = {}
                for key,val in items[i:min(len(cells),i+size)]:
                    dictionary[key] = val
                cell_parts.append(dictionary)

            worker = partial(self.find_faces_of_cell_chunk, point_cloud, edge_map)
            faces = set()
            with Pool(processes=cpus) as pool:
                for res in pool.imap_unordered(worker,cell_parts, chunksize=1):
                    faces = faces.union(res)

            with open(os.path.join(self.path,filename),"w") as f:
                for face in faces:
                    f.write(f"{face}\n")
        return faces

    def is_boundary(self,cell_points,face_points,face_normal):
        """
        check, whether all points of the cell lie on one side of the face


        """
        signs = set()
        signs.add(0)

        for cell_point in cell_points:
            dot = face_normal.dot(cell_point-face_points[0]).real()
            if dot == 0:
                signs.add(0)
            elif dot < 0:
                signs.add(-1)
            else:
                signs.add(1)
            if len(signs) > 2:
                return False
        return True

    def is_coplanar(self,points,normal=None):
        if len(points) < 4:
            return True

        if points[0].dim==3:
            normal = (points[1]-points[0]).cross(points[2]-points[0])
            for i in range(3,len(points)):
                test = normal.dot(points[i] - points[0]).real()
                if test !=0:
                    return False
            return normal
        elif points[0].dim==4:
            if normal is None:
                raise("Normal is required in dim>3")
            directions = [p-points[0] for p in points[1:]]
            # construct second normal from the first three points
            dir0 =directions[0]
            dir1 = directions[1]
            tri = dir0*dir1*normal
            normal2 = epsilon.contract(tri,axes=[[1,2,3],[0,1,2]])
            for d in directions[2:]:
                if d.dot(normal2)!=zero:
                    return False
            return FVector(normal2.components)
        else:
            raise("Coplanar is not implemented in dim>4 yet")

    def walk_face(self, vertices, edge_map, path, max_len=10,normal=None):
        """
        Backtracking algorithm to find all closed loops in the graph that start with `path`.

        - vertices: list/array of points; vertices[i] gives coordinates of vertex i
        - edge_map: dict {vertex: iterable_of_neighbor_vertices}
        - path: starting vertex sequence (prefix) to be extended
        - max_len: cap on path length during search
        - skip_backtrack: if True, avoid immediately returning to the previous vertex
        """
        # skip unreasonable input
        if not path or len(path) < 1:
            return []

        start = path[0]
        all_cycles = []

        all_indices = list(edge_map.keys())
        cell_vertices = [vertices[i] for i in all_indices]

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
                    points = [vertices[i] for i in current_path]
                    if len(current_path) > 3:
                        normal2=self.is_coplanar(points,normal=normal)
                        if normal2 is not False:
                            if self.is_boundary(cell_vertices,points,normal2):
                                all_cycles.append(current_path)
                    else:
                        all_cycles.append(current_path)
                    # Even after finding a cycle, continue exploring other neighbors
                    continue

                # Do not revisit vertices already on the current path (prevents sub-cycles)
                if neighbor in current_path:
                    continue

                dfs(current_path + [neighbor])

        dfs(path)
        return all_cycles

    def get_edge_map(self,edges):
        edge_map = defaultdict(list)
        for a, b in edges:
            edge_map[a].append(b)
            edge_map[b].append(a)
        return edge_map

    def find_faces(self,vertices,edges,max_length=10):
        # undirected graph
        # construct a map that generates an empty list for each key as default
        edge_map = self.get_edge_map(edges)

        faces ={} # prepare a dictionary that stores a canonical representation of a face and the proper index ordering
        visited = set()
        count = 0
        success=0
        oldline = ""
        for start in range(len(vertices)):
            text = "progress: " + str(start) + " / " + str(len(vertices))
            show_inline_progress_in_terminal(text,oldline)
            oldline = text

            for neighbor in edge_map[start]:
                if (start,neighbor) in visited or (neighbor,start) in visited:
                    continue # skip the rest in this case

                # attempt to walk in a cycle from (start->neighbor)
                path = [start,neighbor]
                new_faces = self.walk_face(vertices,edge_map,path,max_length)
                for face in new_faces:
                    real_ordering = face.copy()
                    # normalize and store
                    canonical = tuple(sorted(face))
                    if canonical not in faces:
                        faces[canonical]=real_ordering
                        success+=1

                visited.add((start,neighbor))
        print() # finish progress line
        return list(faces.values())

    def find_faces_of_cell(self,point_cloud,edge_map,cell_normal,silent=True):
        faces = {} # prepare a dictionary that stores a canonical representation of a face and the proper index ordering
        visited = set()
        oldline = ""
        indices = edge_map.keys()
        for i,start in enumerate(indices):
            if not silent:
                text = "progress: " + str(i+1) + " / " + str(len(indices))
                show_inline_progress_in_terminal(text,oldline)
                oldline = text

            for neighbor in edge_map[start]:
                if (start,neighbor) in visited or (neighbor,start) in visited:
                    continue # skip the rest in this case

                # attempt to walk in a cycle from (start->neighbor)
                path = [start,neighbor]
                new_faces = self.walk_face(point_cloud,edge_map,path,10,normal=cell_normal)
                for face in new_faces:
                    real_ordering = face.copy()
                    # normalize and store
                    canonical = tuple(sorted(face))
                    if canonical not in faces:
                        faces[canonical]=MeshFace(real_ordering)

                visited.add((start,neighbor))
            if not silent:
                print()
        return list(faces.values())

    def get_faces_old(self,seed=[1,1,-1],max_length=10):
        filename = "coxH4_faces" + str(seed).replace(",", "_") + ".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            point_cloud = self.point_cloud(seed)
            edges = self.get_edges(seed)
            faces = self.find_faces(point_cloud,edges,max_length)
        else:
            faces = self.read_faces(filename)
            print("face data read from file")
        self.save(faces,filename)
        return faces

    def get_normals(self):
        return [n.real() for n in self.normals]

    def get_geometry(self,seed):
        return [self.point_cloud(seed),self.get_edges(seed),self.get_faces(seed)]


    def info(self,seed):
        vertices = self.point_cloud(seed)
        edges = self.get_edges(seed)
        faces = self.get_faces(seed)
        cells = self.get_cells(seed)

        cell_sizes = dict()
        for key, val in cells.items():
            l = len(key)
            if l in cell_sizes.keys():
                cell_sizes[l] = cell_sizes[l] + 1
            else:
                cell_sizes[l] = 1
        print("cell sizes", cell_sizes)

        face_sizes = dict()
        for face in faces:
            l = len(face)
            if l in face_sizes.keys():
                face_sizes[l] = face_sizes[l] + 1
            else:
                face_sizes[l] = 1
        print("face sizes: ", face_sizes)

        print("edges: ",len(edges))
        print("vertices: ",len(vertices))
        print("faces: ",len(faces))
        print("cells: ",len(cells))
        print("Euler: ",len(vertices)-len(edges)+len(faces)-len(cells))

    def get_cells_in_conjugacy_classes(self,signature=COXH4_SIGNATURES["x3x3x5x"]):
        """
        sort faces into their conjugacy classes
        the representatives of the classes are the faces that contain the vertex
        with index 0
        >>> group = CoxH4("data")
        >>> cc = group.get_cells_in_conjugacy_classes(COXB4_SIGNATURES["x3x3x5x"])
        >>> print(cc)
        {(0, 17, 55, 76, 82, 114, 121, 124, 129, 134, 142, 147, 156, 158, 196, 209, 233, 259, 286, 288, 298, 316, 367, 376): {(31, 40, 41, 42, 49, 67, 105, 130, 133, 172, 174, 175, 192, 202, 227, 265, 278, 280, 282, 327, 329, 330, 335, 365), (14, 22, 28, 32, 34, 37, 45, 57, 72, 79, 118, 136, 143, 176, 177, 182, 231, 240, 250, 273, 283, 305, 318, 380), (9, 20, 50, 61, 77, 81, 99, 106, 122, 141, 198, 228, 244, 249, 261, 287, 301, 307, 317, 323, 324, 358, 363, 375), (0, 17, 55, 76, 82, 114, 121, 124, 129, 134, 142, 147, 156, 158, 196, 209, 233, 259, 286, 288, 298, 316, 367, 376), (2, 26, 27, 36, 38, 46, 101, 128, 144, 152, 167, 200, 204, 215, 218, 239, 246, 294, 308, 333, 348, 362, 368, 373), (8, 15, 29, 52, 89, 97, 112, 140, 148, 185, 194, 199, 207, 248, 254, 272, 281, 299, 304, 313, 334, 356, 372, 383), (35, 44, 64, 70, 73, 74, 75, 103, 107, 127, 131, 160, 168, 212, 224, 235, 256, 297, 332, 341, 342, 345, 349, 352), (23, 39, 59, 69, 83, 90, 94, 111, 153, 165, 214, 251, 260, 262, 291, 292, 312, 319, 336, 338, 340, 343, 351, 379), (19, 30, 48, 62, 87, 110, 115, 135, 139, 150, 164, 171, 188, 189, 206, 213, 223, 230, 242, 275, 279, 296, 344, 382), (10, 21, 43, 56, 63, 78, 104, 123, 138, 146, 161, 173, 180, 190, 217, 258, 276, 277, 289, 295, 303, 311, 331, 350), (6, 16, 25, 54, 71, 86, 88, 108, 120, 154, 159, 166, 211, 219, 222, 232, 237, 241, 255, 267, 310, 364, 369, 370), (11, 12, 58, 68, 100, 109, 119, 137, 151, 191, 195, 197, 226, 238, 263, 264, 268, 269, 284, 285, 293, 302, 315, 381), (5, 13, 33, 53, 92, 113, 116, 125, 132, 162, 183, 184, 186, 203, 208, 229, 252, 257, 271, 309, 320, 326, 355, 359), (1, 18, 84, 85, 93, 95, 102, 117, 145, 179, 181, 193, 216, 225, 245, 290, 300, 314, 321, 322, 328, 357, 360, 361), (4, 7, 24, 47, 51, 65, 91, 96, 126, 155, 157, 163, 178, 187, 201, 205, 220, 253, 306, 325, 339, 366, 374, 378), (3, 60, 66, 80, 98, 149, 169, 170, 210, 221, 234, 236, 243, 247, 266, 270, 274, 337, 346, 347, 353, 354, 371, 377)}, (0, 10, 44, 50, 61, 73, 83, 90, 104, 121, 129, 132, 134, 137, 138, 147, 151, 166, 184, 191, 197, 212, 214, 222, 229, 237, 241, 251, 252, 255, 256, 259, 261, 263, 264, 267, 287, 289, 295, 303, 320, 324, 332, 349, 351, 359, 375, 379): {(3, 8, 19, 25, 32, 33, 71, 82, 86, 88, 111, 113, 114, 136, 140, 143, 165, 182, 183, 188, 194, 206, 207, 208, 209, 219, 223, 230, 234, 236, 254, 260, 262, 271, 283, 286, 292, 298, 305, 313, 319, 326, 353, 354, 364, 371, 376, 382), (2, 7, 9, 11, 21, 26, 38, 40, 41, 65, 70, 74, 81, 93, 100, 106, 107, 109, 117, 126, 131, 141, 172, 175, 180, 192, 195, 200, 202, 205, 215, 216, 217, 225, 258, 269, 276, 277, 297, 301, 302, 321, 325, 341, 361, 363, 368, 374), (0, 10, 44, 50, 61, 73, 83, 90, 104, 121, 129, 132, 134, 137, 138, 147, 151, 166, 184, 191, 197, 212, 214, 222, 229, 237, 241, 251, 252, 255, 256, 259, 261, 263, 264, 267, 287, 289, 295, 303, 320, 324, 332, 349, 351, 359, 375, 379), (29, 30, 36, 43, 47, 48, 53, 54, 56, 89, 96, 101, 108, 116, 122, 125, 139, 144, 150, 154, 155, 157, 159, 161, 173, 189, 190, 198, 211, 242, 244, 246, 248, 249, 253, 257, 272, 299, 308, 309, 317, 331, 355, 356, 358, 362, 366, 369), (12, 17, 22, 23, 31, 42, 57, 59, 60, 64, 67, 72, 76, 85, 94, 127, 133, 142, 153, 158, 160, 193, 210, 224, 231, 233, 238, 240, 243, 247, 268, 284, 285, 290, 312, 316, 318, 322, 328, 329, 337, 338, 342, 345, 347, 357, 365, 381), (1, 5, 13, 20, 28, 46, 55, 58, 62, 68, 77, 79, 84, 87, 92, 95, 99, 110, 115, 118, 119, 124, 128, 152, 156, 162, 164, 167, 177, 181, 186, 196, 203, 226, 228, 239, 245, 250, 288, 293, 296, 307, 314, 315, 323, 367, 373, 380), (4, 14, 15, 18, 24, 27, 34, 37, 45, 49, 52, 97, 98, 102, 105, 135, 145, 148, 163, 170, 171, 176, 179, 185, 187, 204, 213, 218, 220, 227, 265, 266, 270, 273, 275, 279, 294, 300, 306, 327, 330, 333, 344, 346, 348, 360, 377, 383), (6, 16, 35, 39, 51, 63, 66, 69, 75, 78, 80, 91, 103, 112, 120, 123, 130, 146, 149, 168, 169, 174, 178, 199, 201, 221, 232, 235, 274, 278, 280, 281, 282, 291, 304, 310, 311, 334, 335, 336, 339, 340, 343, 350, 352, 370, 372, 378)}, (0, 23, 76, 82, 94, 129, 142, 214, 260, 292, 298, 379): {(21, 35, 44, 74, 78, 168, 212, 277, 289, 295, 341, 350), (12, 70, 73, 107, 109, 137, 197, 224, 268, 269, 332, 342), (52, 66, 97, 112, 149, 194, 207, 236, 266, 334, 346, 371), (4, 15, 24, 157, 199, 201, 248, 253, 272, 372, 378, 383), (10, 56, 122, 141, 180, 217, 249, 287, 301, 303, 331, 375), (28, 57, 114, 118, 124, 136, 158, 196, 231, 305, 316, 376), (27, 46, 84, 93, 102, 145, 152, 200, 216, 314, 333, 368), (18, 42, 85, 172, 175, 225, 227, 329, 330, 357, 360, 361), (7, 41, 91, 105, 130, 163, 178, 187, 192, 278, 327, 374), (6, 39, 83, 88, 165, 166, 219, 232, 237, 262, 340, 351), (30, 36, 48, 115, 164, 167, 213, 218, 308, 344, 348, 373), (0, 23, 76, 82, 94, 129, 142, 214, 260, 292, 298, 379), (1, 37, 72, 176, 177, 179, 181, 300, 318, 322, 328, 380), (5, 121, 132, 134, 156, 203, 208, 209, 271, 286, 359, 367), (3, 59, 111, 221, 234, 243, 274, 291, 312, 319, 337, 343), (43, 104, 120, 123, 138, 154, 159, 161, 222, 255, 310, 311), (69, 75, 90, 153, 160, 235, 251, 256, 336, 338, 345, 349), (38, 47, 101, 126, 155, 204, 215, 220, 294, 306, 325, 362), (9, 11, 50, 77, 81, 100, 119, 191, 226, 261, 264, 323), (51, 63, 65, 96, 146, 173, 190, 205, 258, 276, 339, 366), (22, 32, 45, 98, 182, 210, 240, 270, 273, 347, 353, 354), (33, 62, 110, 116, 139, 162, 183, 186, 189, 206, 223, 257), (13, 20, 61, 92, 99, 184, 198, 309, 320, 324, 355, 358), (49, 60, 67, 80, 133, 169, 170, 174, 247, 265, 335, 377), (17, 55, 58, 147, 151, 233, 238, 259, 263, 285, 288, 293), (25, 53, 113, 125, 211, 229, 241, 252, 267, 326, 364, 369), (2, 26, 106, 128, 144, 228, 239, 244, 246, 307, 317, 363), (31, 40, 64, 103, 127, 131, 202, 280, 282, 297, 352, 365), (8, 16, 29, 54, 71, 86, 108, 254, 281, 304, 356, 370), (14, 34, 79, 87, 143, 188, 250, 275, 279, 283, 296, 382), (19, 89, 135, 140, 148, 150, 171, 185, 230, 242, 299, 313), (68, 95, 117, 193, 195, 245, 284, 290, 302, 315, 321, 381)}, (0, 90, 94, 137, 142, 214, 224, 233, 238, 259, 263, 268, 332, 338, 345, 349): {(5, 28, 33, 62, 87, 124, 156, 162, 188, 208, 209, 223, 250, 283, 305, 376), (18, 60, 67, 98, 176, 179, 210, 227, 240, 265, 273, 318, 322, 329, 357, 377), (21, 35, 41, 63, 65, 78, 130, 178, 202, 258, 280, 297, 339, 341, 352, 374), (7, 38, 102, 163, 175, 192, 200, 204, 216, 220, 225, 325, 327, 330, 333, 360), (64, 69, 75, 103, 133, 153, 160, 169, 174, 243, 247, 274, 282, 312, 343, 365), (13, 20, 30, 36, 110, 115, 116, 139, 144, 186, 239, 244, 307, 355, 358, 373), (55, 58, 61, 92, 99, 119, 132, 134, 147, 151, 184, 203, 261, 264, 323, 367), (16, 51, 54, 120, 123, 146, 154, 161, 190, 199, 201, 248, 253, 304, 356, 366), (10, 43, 125, 138, 159, 198, 211, 249, 252, 255, 267, 309, 320, 324, 331, 375), (44, 50, 73, 74, 81, 100, 107, 109, 191, 197, 217, 277, 287, 295, 301, 303), (15, 24, 47, 48, 157, 171, 185, 213, 218, 242, 272, 294, 299, 306, 308, 362), (9, 11, 26, 68, 77, 93, 95, 106, 117, 128, 152, 195, 226, 228, 314, 368), (14, 27, 37, 46, 79, 84, 145, 164, 167, 177, 181, 279, 296, 300, 344, 348), (12, 31, 40, 42, 70, 85, 127, 131, 172, 269, 290, 302, 321, 342, 361, 381), (25, 88, 113, 121, 129, 229, 237, 241, 262, 271, 286, 292, 298, 351, 359, 379), (2, 56, 96, 101, 122, 126, 141, 155, 173, 180, 205, 215, 246, 276, 317, 363), (39, 83, 104, 166, 168, 212, 222, 232, 235, 251, 256, 289, 310, 311, 336, 350), (3, 6, 66, 71, 111, 112, 165, 194, 219, 221, 236, 254, 281, 291, 340, 370), (4, 49, 52, 80, 91, 105, 149, 170, 187, 278, 334, 335, 346, 372, 378, 383), (22, 23, 59, 76, 82, 114, 136, 182, 231, 234, 260, 316, 319, 337, 347, 353), (8, 19, 29, 53, 86, 89, 108, 150, 183, 189, 206, 257, 313, 326, 364, 369), (32, 34, 45, 97, 135, 140, 143, 148, 207, 230, 266, 270, 275, 354, 371, 382), (1, 17, 57, 72, 118, 158, 193, 196, 245, 284, 285, 288, 293, 315, 328, 380), (0, 90, 94, 137, 142, 214, 224, 233, 238, 259, 263, 268, 332, 338, 345, 349)}}

        """
        if isinstance(signature, str):
            signature = COXH4_SIGNATURES[signature]

        filename = self.name + "_classes" + str(signature).replace(",", "_") + ".dat"
        classes = self.read_classes(filename)
        if classes is None:
            cells = self.get_cells(signature)
            vertices = self.get_point_cloud(signature)

            sorted_cells = [tuple(sorted(cell)) for cell in cells.keys()]

            classes = {}
            vertex2index = {v: idx for idx, v in enumerate(vertices)}
            # get all faces of the first vertex

            while len(sorted_cells)>0:
                # find cell at 0
                for cell in sorted_cells:
                    if 0 in cell:
                        rep_cell = cell
                        classes[cell]={cell}
                        sorted_cells.remove(cell)
                        break

                with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                    # split self.elements in os.cpu_count() parts
                    total_elements = len(self.elements)
                    num_processes = min(total_elements,os.cpu_count())

                    chunk_size = max(1, total_elements // num_processes)  # Ensure at least 1 per chunk
                    chunks = [self.elements[i * chunk_size: (i + 1) * chunk_size] for i in range(num_processes)]
                    # Filter out empty chunks (in case of uneven division)
                    chunks = [chunk for chunk in chunks if chunk]
                    tasks = [(elements, rep_cell, vertices, vertex2index) for elements in chunks]
                    results = pool.map(process_element, tasks)
                    for result in results:
                        classes[rep_cell]=classes[rep_cell].union(result)
                        for cell in result:
                            if cell in sorted_cells:
                                sorted_cells.remove(cell)
                        print(len(sorted_cells))

            self.save_dictionary(classes, filename)
        return classes

    def read_classes(self, filename):
        classes = None
        try:
            with open(os.path.join(self.path, filename), "r") as f:
                classes = {}
                for line in f:
                    parts = line.split("->")
                    classes[eval(parts[0].strip())] = eval(parts[1].strip())
        except Exception as e:
            return None
        return classes


if __name__ == '__main__':
    coxH4 = CoxH4()
    signature = COXH4_SIGNATURES["x3x3x5x"]
    print(len(coxH4.get_point_cloud(signature)))
    print(len(coxH4.get_edges(signature)))
    print(coxH4.get_cells_in_conjugacy_classes(signature))