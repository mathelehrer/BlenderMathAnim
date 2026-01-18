# we generate the full coxeter group of type H4
# see https://en.wikipedia.org/wiki/Coxeter_group#Table_of_all_Coxeter_groups
from __future__ import annotations

import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np

from mathematics.geometry.field_extensions import QR, FMatrix, FVector, EpsilonTensor
from utils.string_utils import show_inline_progress_in_terminal

PATH = "data/"
zero = QR.from_integers(0,1,0,1,root_modulus=2,root_string="r2")
one = QR.from_integers(1,1,0,1,root_modulus=2,root_string="r2")
half = QR.from_integers(1,2,0,1,root_modulus=2,root_string="r2")
root2inv = QR.from_integers(0,1,1,2,root_modulus=2,root_string="r2")
epsilon = EpsilonTensor(4,root_modulus=2,root_string="r2")
zero4 = FVector([zero,zero,zero,zero])

class Face(list):
    """
    this is just a list of integers that is hashable, we want to make sure that
    the ording of the indices is preserved and cyclicly rotated to have the smallest index
    in the first position

    However, the reverse of the list should have the same hash value as the original list
    """


    def __init__(self,elements):
        """
        hashable list that is equivalent with respect to cyclic permutations
        >>> Face([1,2,3]) == Face([3,1,2])
        True
        >>> Face([1,2,3]) == Face([1,3,2])
        True

        """

        super().__init__(elements)
        smallest_index = min(self)

        index = self.index(smallest_index)

        for i in range(index):
            elements.append(elements.pop(0))

        self.reverse_elements = elements.copy()
        self.reverse_elements.reverse()
        self.reverse_elements.insert(0,self.reverse_elements.pop())
        self.elements = elements

    def __str__(self):
        return f"Face("+super().__str__()+")"

    def __rep__(self):
        return str(self)

    def __hash__(self):
        """
        """
        return hash(tuple(self.elements))*hash(tuple(self.reverse_elements))

    def __eq__(self,other):
        first = self.elements == other.elements
        second = self.elements == other.reverse_elements
        return first or second

def center(key, point_cloud):
    c = FVector([zero,zero,zero,zero])
    for k in key:
        c = c + point_cloud[k]
    return c.real() / len(key)


class CoxF4:
    def __init__(self,path=None):
        """

        """
        self.name = "coxF4"
        if path is None:
            self.path=PATH
        else:
            self.path = path

        # normal vectors: Computed in CoxF4.nb
        normals = [
                    FVector([one,zero,zero,zero]),
                    FVector([half,half,root2inv,zero]),
                    FVector([zero,root2inv,half, half]),
                   FVector([zero,zero,zero,one])
                   ]

        self.normals = normals

        # the generators of D4 are given by
        identity =  FMatrix([[one,zero,zero,zero],[zero,one,zero,zero],[zero,zero,one,zero],[zero,zero,zero,one]])
        generators=[identity-(n*n)-(n*n) for n in normals]
        # cast to matrix for proper matrix multiplication
        self.generators = [FMatrix(g.components) for g in generators]

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

    def get_radius(self,seed=[1,1,-1]):
        return self.get_real_point_cloud(seed)[0].length

    def point_cloud(self,signature =[1,1,1,1],start = None):
        filename = self.name+"_points" + str(signature).replace(",", "_") + ".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            if start is None:
                raise "No start point given for point cloud generation."
            point_cloud = set()
            for element in self.elements:
                point_cloud.add(element@start)
            self.save(point_cloud,filename)
        else:
            point_cloud = self.read_points(filename)
            print("point cloud read from file")
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
        filename=self.name+"_edges"+str(seed).replace(",","_")+".dat"
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
            if size>0:
                chunks = [[i,min(i+size,len(point_cloud))] for i in range(0,len(point_cloud),size)]
            else:
                chunks = [[0,len(point_cloud)]]
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
        filename = self.name+"_cells" + str(seed).replace(",", "_") + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    parts = line.split("->")
                    key = eval(parts[0].strip())
                    val = FVector.parse(parts[1].strip())
                    cells[key] = val
                return cells

        point_cloud = self.point_cloud(seed)
        edges = self.get_edges(seed)
        print("Find cells at origin... ",end="")
        cells0 = self.get_cells_at_origin(point_cloud,edges)
        print("done")

        for key,val in cells0.items():
            cells[key]=val

        active_elements = {}
        active_elements[point_cloud[0]] = FMatrix.identity(4,root_modulus=2,root_string="r2")
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
            if size>0:
                for i in range(0,len(active_elements),size):
                    chunks.append(active_elements[i:min(len(active_elements),i+size)])
            else:
                chunks.append(active_elements)

            worker =partial(self.get_cells_from_chunk,point_cloud,key,val)

            with Pool(processes=cpus) as pool:
                for res in pool.imap_unordered(worker,chunks,chunksize=1):
                    for key,val in res.items():
                        if key not in cells:
                            cells[key] = val


        with open(os.path.join(self.path,filename),"w") as f:
            for key,val in cells.items():
                f.write(f"{key}->{val}\n")
                print(key,val,center(key,point_cloud))

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
        filename = self.name+"_faces" + str(seed).replace(",", "_") + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    faces.add(eval(line))
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
            if size>0:
                for i in range(0,len(cells),size):
                    dictionary = {}
                    for key,val in items[i:min(len(cells),i+size)]:
                        dictionary[key] = val
                    cell_parts.append(dictionary)
            else:
                dictionary={}
                for key,val in items:
                    dictionary[key]=val
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
                        faces[canonical]=Face(real_ordering)

                visited.add((start,neighbor))
            if not silent:
                print()
        return list(faces.values())

    def get_faces_old(self,seed=[1,1,-1],max_length=10):
        filename = self.name+"_faces" + str(seed).replace(",", "_") + ".dat"
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

if __name__ == '__main__':
    group = CoxF4()
    signs = [-1,1,1,1]
    print(len(group.point_cloud(signs)))
    print(len(group.get_edges(signs)))