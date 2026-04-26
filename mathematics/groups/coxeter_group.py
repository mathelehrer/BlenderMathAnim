from __future__ import annotations

import multiprocessing
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np

from interface.ibpy import Vector
from mathematics.algebra.field_extensions import QR, FMatrix, FVector
from mathematics.geometry.meshface import MeshFace
from utils.string_utils import show_inline_progress_in_terminal


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


class CoxeterGroup:
    def __init__(self, name, normals, generators, path,signatures={}, seeds={},name_dict={},load_elements=True):
        self.name = name
        self.path = path
        self.normals = normals
        self.generators = generators

        # every Coxeter group should be given two dictionaries that relate the coxeter-dynkin-string to the signature and the seed
        self.signatures = signatures
        self.seeds = seeds
        self.name_dict = name_dict
        # a third dictionary is created for compatability that relates the signature tuple to the coxeter-dynkin-string
        self.cd_strings = {}
        for cd_string,signature in signatures.items():
            self.cd_strings[tuple(signature)] = cd_string

        if load_elements:
            if not os.path.exists(os.path.join(self.path, self.name + "_elements.dat")):
                self.elements = self.generate_elements()
            else:
                self.elements = self._read_elements(self.name + "_elements.dat")

    def point_cloud(self, signature, start=None):

        point_cloud = set()
        if start:
            for element in self.elements:
                point_cloud.add(element @ start)
            return list(point_cloud)
        else:
            cd_label = self._get_cd_label(signature)
            filename = self.name + "_points_" + cd_label + ".dat"
            one = QR.from_integers(1,1,0,1)
            if not os.path.exists(os.path.join(self.path, filename)):
                seed = self.seeds.get(cd_label)
                for element in self.elements:
                    if cd_label.startswith("s"):
                        if element.determinant() == one:
                            point_cloud.add(element @ seed)
                    else:
                        point_cloud.add(element @ seed)
                self._save(point_cloud, filename)
            else:
                point_cloud = self._read_points(filename)
            return list(point_cloud)

    def get_snub_cloud(self,seed=""):
        elements = self.generate_elements()
        snub_points = []
        one = QR.from_integers(1,1,0,1)
        start = self.seeds.get(seed)
        for element in elements:
            if element.determinant() == one:
                snub_points.append(FVector((element*start).components))
        return snub_points

    def get_real_point_cloud(self, signature):
        return [p.real() for p in self.point_cloud(signature)]

    def get_edges(self, seed):
        cd_label = self._get_cd_label(seed)
        filename = self.name + "_edges_" + self._suffix(cd_label) + ".dat"
        if not os.path.exists(os.path.join(self.path, filename)):
            point_cloud = self.point_cloud(seed)
            minimum = np.inf
            min_dist = None
            for j in range(1, len(point_cloud)):
                dist = (point_cloud[0] - point_cloud[j]).norm()
                dist_real = dist.real()
                if dist_real < minimum:
                    minimum = dist_real
                    min_dist = dist

            size = int(len(point_cloud) / os.cpu_count())
            if size > 0:
                chunks = [[i, min(i + size, len(point_cloud))] for i in range(0, len(point_cloud), size)]
            else:
                chunks = [[0, len(point_cloud)]]
            worker = partial(self._find_edges_for_chunk, min_dist, point_cloud)
            edges = []
            with Pool(processes=os.cpu_count()) as pool:
                for res in pool.imap_unordered(worker, chunks, chunksize=1):
                    edges = edges + res
            if len(edges) > 0:
                self._save(edges, filename)
        else:
            edges = self._read_edges(filename)
            print("edge data read from file")
        return edges

    def generate_elements(self):
        filename = self.name + "_elements.dat"
        if os.path.exists(os.path.join(self.path, filename)):
            return self._read_elements(filename)

        new_elements = list(self.generators)
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
        result = list(elements)
        self._save(result, filename)
        return result

    def get_radius(self, signature):
        return self.get_real_point_cloud(signature)[0].length

    def get_normals(self):
        return [n.real() for n in self.normals]

    # io stuff
    def _get_cd_label(self, signature):
        """
        robust function that returns the cd label for a given signature.
        if the signature is a string, it is returned as is.

        :param signature:
        :return:
        """
        if isinstance(signature, str):
            if signature.startswith("x") or signature.startswith("o") or signature.startswith("s"):
                return signature
            else:
                return self.name_dict[signature]
        else:
            if isinstance(signature, list):
                signature = tuple(signature)
            return self.cd_strings[signature]

    def _suffix(self, signature):
        if isinstance(signature, tuple):
            signature = list(signature)
        return str(signature).replace(",", "_")

    def _read_elements(self, filename):
        data = []
        with open(os.path.join(self.path, filename), "r") as f:
            for line in f:
                data.append(FMatrix.parse(line.strip()))
        return data

    def _read_points(self, filename):
        data = []
        with open(os.path.join(self.path, filename), "r") as f:
            for line in f:
                data.append(FVector.parse(line.strip()))
        return data

    def _read_edges(self, filename):
        edges = []
        with open(os.path.join(self.path, filename), "r") as f:
            for line in f:
                edges.append(eval(line))
        return edges

    def _read_faces(self, filename):
        faces = []
        with open(os.path.join(self.path, filename), "r") as f:
            for line in f:
                faces.append(eval(line))
        return faces

    def _save(self, elements, filename):
        with open(os.path.join(self.path, filename), "w") as f:
            for e in elements:
                f.write(str(e) + "\n")

    def _save_dictionary(self, dictionary, filename):
        with open(os.path.join(self.path, filename), "w") as f:
            for key, val in dictionary.items():
                f.write(f"{key}->{val}\n")

    # auxiliary functions

    def _find_edges_for_chunk(self, min_dist, point_cloud, rng):
        print("Start process for range: ", rng)
        edges = []
        for i in range(*rng):
            for j in range(i + 1, len(point_cloud)):
                dist = (point_cloud[i] - point_cloud[j]).norm()
                if dist == min_dist:
                    edges.append([i, j])
        return edges


class CoxeterGroup3D(CoxeterGroup):
    def __init__(self, name, normals, generators, path,signatures={},seeds={},name_dict={}, load_elements=True):
        """
           this is a subclass for three-dimensional coxeter groups. They only differ in the way they find the faces
            from their four-dimensional relatives
        """
        super().__init__(name, normals, generators, path,
                         signatures=signatures,seeds=seeds,load_elements=load_elements)

    def get_faces(self, signature):
        cd_label = self._get_cd_label(signature)
        filename = self.name + "_faces_" + self._suffix(cd_label) + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            faces = []
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    faces.append(eval(line))
            return faces

        point_cloud = self.point_cloud(signature)
        edges = self.get_edges(signature)
        faces = self._find_faces(point_cloud, edges)

        for idx in range(len(faces)):
            face = faces[idx]
            normal = (point_cloud[face[1]] - point_cloud[face[0]]).cross(
                point_cloud[face[2]] - point_cloud[face[0]])
            center = Vector((0, 0, 0))
            for i in face:
                center += point_cloud[i].real()
            center /= len(face)
            if normal.real().dot(center) < 0:
                faces[idx] = face[::-1]

        self._save(faces, filename)
        return faces

    def get_faces_in_conjugacy_classes(self, signature=None):
        if signature is None:
            raise ValueError("signature must be provided")
        faces = self.get_faces(signature)
        vertices = self.point_cloud(signature)

        classes = {}
        first_faces = []
        for face in faces:
            if 0 in face:
                face = MeshFace(face)
                first_faces.append(face)
                classes[face] = {face}

        for first_face in first_faces:
            for element in self.elements:
                mapped_face = []
                for face_index in first_face:
                    vertex = vertices[face_index]
                    target = element @ vertex
                    mapped_face.append(vertices.index(target))
                classes[first_face].add(MeshFace(mapped_face))

        return classes

    def _is_coplanar(self, points):
        if len(points) < 4:
            return True
        normal = (points[1] - points[0]).cross(points[2] - points[0])
        for i in range(3, len(points)):
            if normal.dot(points[i] - points[0]).real() != 0:
                return False
        return True

    def _walk_face(self, vertices, edge_map, path, max_len=10):
        if not path:
            return []
        start = path[0]
        all_cycles = []
        seen_signatures = set()

        def record_cycle(cycle):
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
                if len(current_path) >= 2 and neighbor == current_path[-2]:
                    continue
                if neighbor == start:
                    if len(current_path) >= 3:
                        points = [vertices[i] for i in current_path]
                        if self._is_coplanar(points):
                            record_cycle(current_path[:])
                    continue
                if neighbor in current_path:
                    continue
                dfs(current_path + [neighbor])

        dfs(path)
        return all_cycles

    def _is_boundary(self, face, vertices):
        signs = {0}
        normal = (vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]])
        for v in vertices:
            dot = normal.dot(v - vertices[face[0]]).real()
            if dot < 0:
                signs.add(-1)
            elif dot > 0:
                signs.add(1)
            if len(signs) > 2:
                return False
        return True

    def _find_faces(self, vertices, edges):
        edge_map = defaultdict(list)
        for a, b in edges:
            edge_map[a].append(b)
            edge_map[b].append(a)

        faces = {}
        visited = set()
        for start in range(len(vertices)):
            for neighbor in edge_map[start]:
                if (start, neighbor) in visited or (neighbor, start) in visited:
                    continue
                new_faces = self._walk_face(vertices, edge_map, [start, neighbor])
                for face in new_faces:
                    canonical = tuple(sorted(face))
                    if self._is_boundary(canonical, vertices) and canonical not in faces:
                        faces[canonical] = face
                visited.add((start, neighbor))
        return list(faces.values())


class CoxeterGroup4D(CoxeterGroup):

    def __init__(self, name, normals, generators, zero, epsilon, identity4, path,
                 signatures={}, seeds={}, load_elements=True):
        super().__init__(name, normals, generators, path,
                         signatures=signatures,seeds=seeds, load_elements=load_elements)
        self._zero = zero
        self._epsilon = epsilon
        self._zero4 = FVector([zero, zero, zero, zero])
        self._identity4 = identity4

    def get_edge_map(self, edges):
        edge_map = defaultdict(list)
        for a, b in edges:
            edge_map[a].append(b)
            edge_map[b].append(a)
        return edge_map

    def get_faces(self, seed):
        faces = set()
        cd_label = self._get_cd_label(seed)
        filename = self.name + "_faces_" + self._suffix(cd_label) + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    faces.add(eval(line))
        else:
            point_cloud = self.point_cloud(seed)
            edges = self.get_edges(seed)
            edge_map = self.get_edge_map(edges)
            cells = dict(sorted(self.get_cells(seed).items()))

            cpus = os.cpu_count()
            size = int(len(cells) / cpus)
            items = list(cells.items())
            cell_parts = []
            if size > 0:
                for i in range(0, len(cells), size):
                    cell_parts.append(dict(items[i:min(len(cells), i + size)]))
            else:
                cell_parts.append(dict(items))

            worker = partial(self._find_faces_of_cell_chunk, point_cloud, edge_map)
            with Pool(processes=cpus) as pool:
                for res in pool.imap_unordered(worker, cell_parts, chunksize=1):
                    faces = faces.union(res)

            if len(faces) > 0:
                with open(os.path.join(self.path, filename), "w") as f:
                    for face in faces:
                        f.write(f"{face}\n")
        return faces

    def get_cells(self, seed):
        cells = {}
        cd_label = self._get_cd_label(seed)
        filename = self.name + "_cells_" + self._suffix(cd_label) + ".dat"
        if os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    parts = line.split("->")
                    cells[eval(parts[0].strip())] = FVector.parse(parts[1].strip())
            return cells

        point_cloud = self.point_cloud(seed)
        edges = self.get_edges(seed)
        print("Find cells at origin... ", end="")
        cells0 = self._get_cells_at_origin(point_cloud, edges)
        print("done")

        cells.update(cells0)

        active_elements = {point_cloud[0]: self._identity4}
        for elem in self.elements:
            if len(active_elements) == len(point_cloud):
                break
            point = elem @ point_cloud[0]
            if point not in active_elements:
                active_elements[point] = elem
        active_elements = list(active_elements.values())

        for i, (key, val) in enumerate(cells0.items()):
            print("map cell", i)
            cpus = os.cpu_count()
            size = int(len(active_elements) / cpus)
            chunks = []
            if size > 0:
                for j in range(0, len(active_elements), size):
                    chunks.append(active_elements[j:min(len(active_elements), j + size)])
            else:
                chunks.append(active_elements)

            worker = partial(self._get_cells_from_chunk, point_cloud, key, val)
            with Pool(processes=cpus) as pool:
                for res in pool.imap_unordered(worker, chunks, chunksize=1):
                    for k, v in res.items():
                        if k not in cells:
                            cells[k] = v

        if len(cells) > 0:
            with open(os.path.join(self.path, filename), "w") as f:
                for key, val in cells.items():
                    f.write(f"{key}->{val}\n")

        # convert FTensors to FVectors
        for key, val in cells.items():
            cells[key] = FVector(val.components)
        return cells

    def get_cells_in_conjugacy_classes(self, signature):
        cd_label = self._get_cd_label(signature)
        filename = self.name + "_classes_" + self._suffix(cd_label) + ".dat"
        classes = self._read_classes(filename)
        if classes is None:
            cells = self.get_cells(signature)
            vertices = self.point_cloud(signature)
            sorted_cells = [tuple(sorted(cell)) for cell in cells.keys()]
            classes = {}
            vertex2index = {v: idx for idx, v in enumerate(vertices)}

            while len(sorted_cells) > 0:
                for cell in sorted_cells:
                    if 0 in cell:
                        rep_cell = cell
                        classes[cell] = {cell}
                        sorted_cells.remove(cell)
                        break

                total = len(self.elements)
                num_processes = min(total, os.cpu_count())
                chunk_size = max(1, total // num_processes)
                chunks = [self.elements[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
                chunks = [c for c in chunks if c]
                tasks = [(chunk, rep_cell, vertices, vertex2index) for chunk in chunks]
                with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                    for result in pool.map(process_element, tasks):
                        classes[rep_cell] = classes[rep_cell].union(result)
                        for cell in result:
                            if cell in sorted_cells:
                                sorted_cells.remove(cell)
                        print(len(sorted_cells))

            self._save_dictionary(classes, filename)
        return classes

    def info(self, seed):
        vertices = self.point_cloud(seed)
        edges = self.get_edges(seed)
        faces = self.get_faces(seed)
        cells = self.get_cells(seed)
        cell_sizes = {}
        for key in cells:
            cell_sizes[len(key)] = cell_sizes.get(len(key), 0) + 1
        print("cell sizes", cell_sizes)
        face_sizes = {}
        for face in faces:
            face_sizes[len(face)] = face_sizes.get(len(face), 0) + 1
        print("face sizes:", face_sizes)
        print("edges:", len(edges))
        print("vertices:", len(vertices))
        print("faces:", len(faces))
        print("cells:", len(cells))
        print("Euler:", len(vertices) - len(edges) + len(faces) - len(cells))

    def get_normal_for_points(self, vertices, path):
        if len(path) < 4:
            return None
        directions = [vertices[p] - vertices[path[0]] for p in path[1:]]
        tensor = directions[0]
        for d in directions[1:]:
            tensor = tensor * d
        n = self._epsilon.contract(tensor, axes=[[1, 2, 3], [0, 1, 2]])
        if n == self._zero4:
            return None
        return n

    def _check_boundary(self, normal, point_cloud, path):
        boundary_points = path.copy()
        p0 = point_cloud[path[0]]
        pos = None
        neg = None
        for i in range(len(point_cloud)):
            if i not in path:
                p = point_cloud[i]
                diff = p - p0
                dot = diff.dot(normal)
                if dot == self._zero:
                    boundary_points.append(i)
                elif dot.real() < 0:
                    neg = -1
                else:
                    pos = 1
                if pos is not None and neg is not None:
                    return False
        return boundary_points

    def _get_cells_at_origin(self, point_cloud, edges):
        edge_map = self.get_edge_map(edges)
        cell_normals = {}
        cell_indices = [0]
        old_text = ""
        for i, first_neighbor in enumerate(edge_map[0]):
            cell_indices.append(first_neighbor)
            text = "Find boundaries along the line of the " + str(i) + "/" + str(len(edge_map[0])) + " neighbors"
            show_inline_progress_in_terminal(text, old_text)
            old_text = text
            for second_neighbor in edge_map[first_neighbor]:
                if second_neighbor not in cell_indices:
                    cell_indices.append(second_neighbor)
                    for third_neighbor in edge_map[second_neighbor]:
                        if third_neighbor not in cell_indices:
                            cell_indices.append(third_neighbor)
                            normal = self.get_normal_for_points(point_cloud, cell_indices)
                            if normal is not None:
                                result = self._check_boundary(normal, point_cloud, cell_indices)
                                if result:
                                    result.sort()
                                    index_tuple = tuple(set(result))
                                    if point_cloud[index_tuple[0]].dot(normal).real() < 0:
                                        normal = -normal
                                    cell_normals[index_tuple] = normal
                            cell_indices = cell_indices[:-1]
                    cell_indices = cell_indices[:-1]
            cell_indices = cell_indices[:-1]
        print()
        print(len(cell_normals), " cells found")
        return cell_normals

    def _get_cells_from_chunk(self, point_cloud, key, val, chunk):
        cells = {}
        for elem in chunk:
            new_normal = elem @ val
            new_indices = [point_cloud.index(elem @ point_cloud[i]) for i in key]
            new_indices.sort()
            index_tuple = tuple(set(new_indices))
            if index_tuple not in cells:
                if point_cloud[index_tuple[0]].dot(new_normal).real() < 0:
                    new_normal = -new_normal
                cells[index_tuple] = new_normal
        return cells

    def _is_boundary(self, cell_points, face_points, face_normal):
        signs = {0}
        for cell_point in cell_points:
            dot = face_normal.dot(cell_point - face_points[0]).real()
            if dot < 0:
                signs.add(-1)
            elif dot > 0:
                signs.add(1)
            if len(signs) > 2:
                return False
        return True

    def _is_coplanar(self, points, normal=None):
        if len(points) < 4:
            return True
        if points[0].dim == 3:
            n = (points[1] - points[0]).cross(points[2] - points[0])
            for i in range(3, len(points)):
                if n.dot(points[i] - points[0]).real() != 0:
                    return False
            return n
        elif points[0].dim == 4:
            if normal is None:
                raise ValueError("Normal is required in dim>3")
            directions = [p - points[0] for p in points[1:]]
            tri = directions[0] * directions[1] * normal
            normal2 = self._epsilon.contract(tri, axes=[[1, 2, 3], [0, 1, 2]])
            for d in directions[2:]:
                if d.dot(normal2) != self._zero:
                    return False
            return FVector(normal2.components)
        else:
            raise ValueError("Coplanar is not implemented in dim>4")

    def _walk_face(self, vertices, edge_map, path, max_len=10, normal=None):
        if not path:
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
                if len(current_path) >= 2 and neighbor == current_path[-2]:
                    continue
                if neighbor == start:
                    points = [vertices[i] for i in current_path]
                    if len(current_path) > 3:
                        normal2 = self._is_coplanar(points, normal=normal)
                        if normal2 is not False:
                            if self._is_boundary(cell_vertices, points, normal2):
                                all_cycles.append(current_path)
                    else:
                        all_cycles.append(current_path)
                    continue
                if neighbor in current_path:
                    continue
                dfs(current_path + [neighbor])

        dfs(path)
        return all_cycles

    def _find_faces(self, vertices, edges, max_length=10):
        edge_map = self.get_edge_map(edges)
        faces = {}
        visited = set()
        oldline = ""
        for start in range(len(vertices)):
            text = "progress: " + str(start) + " / " + str(len(vertices))
            show_inline_progress_in_terminal(text, oldline)
            oldline = text
            for neighbor in edge_map[start]:
                if (start, neighbor) in visited or (neighbor, start) in visited:
                    continue
                new_faces = self._walk_face(vertices, edge_map, [start, neighbor], max_length)
                for face in new_faces:
                    canonical = tuple(sorted(face))
                    if canonical not in faces:
                        faces[canonical] = face
                visited.add((start, neighbor))
        print()
        return list(faces.values())

    def _find_faces_of_cell(self, point_cloud, edge_map, cell_normal, silent=True):
        faces = {}
        visited = set()
        oldline = ""
        indices = list(edge_map.keys())
        for i, start in enumerate(indices):
            if not silent:
                text = "progress: " + str(i + 1) + " / " + str(len(indices))
                show_inline_progress_in_terminal(text, oldline)
                oldline = text
            for neighbor in edge_map[start]:
                if (start, neighbor) in visited or (neighbor, start) in visited:
                    continue
                new_faces = self._walk_face(point_cloud, edge_map, [start, neighbor], 10, normal=cell_normal)
                for face in new_faces:
                    canonical = tuple(sorted(face))
                    if canonical not in faces:
                        faces[canonical] = MeshFace(face)
                visited.add((start, neighbor))
        return list(faces.values())

    def _find_faces_of_cell_chunk(self, point_cloud, edge_map, chunk):
        print("Start process for a chunk of cells...")
        faces = set()
        for indices, cell_normal in chunk.items():
            reduced_edge_map = {}
            index_set = set(indices)
            for src, dest in edge_map.items():
                if src in indices:
                    reduced_edge_map[src] = set(dest).intersection(index_set)
            faces = faces.union(set(self._find_faces_of_cell(point_cloud, reduced_edge_map, cell_normal)))
        return faces

    def _read_classes(self, filename):
        try:
            with open(os.path.join(self.path, filename), "r") as f:
                classes = {}
                for line in f:
                    parts = line.split("->")
                    classes[eval(parts[0].strip())] = eval(parts[1].strip())
            return classes
        except Exception:
            return None
