import numpy as np
from anytree import RenderTree
from mathutils import Vector, Quaternion

from mathematics.lin_alg.subspace import Subspace
from objects.bobject import BObject
from objects.face import Face
from objects.tex_bobject import SimpleTexBObject
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME


class Polyhedron(BObject):
    """
    Create a polyhedron from vertices, edges and faces:
    """

    def __init__(self, vertices, faces, **kwargs):
        """
        :param vertices: vertices
        :param edges: edges in terms of vertices
        :param faces: faces in terms of vertices
        :param index_one: if True -> counting starts at one
        :param kwargs:
        """
        self.counter = None  # dummy counter for growing the polyhedron
        self.root = None
        self.kwargs = kwargs
        self.faces = []
        self.vertices = vertices

        self.coordinate_system = self.get_from_kwargs('coordinate_system', None)
        if self.coordinate_system:  # adjust coordinates to the parenting coordinate system
            for i, v in enumerate(self.vertices):
                self.vertices[i] = self.coordinate_system.coords2location(v)

        self.index_base = self.get_from_kwargs('index_base', 1)
        name = self.get_from_kwargs('name', 'Polyhedron')
        self.vertex_radius = self.get_from_kwargs('vertex_radius',
                                                  0.1)  # need to access the vertex radius to place word labels properly
        if 'name' in kwargs:
            kwargs.pop('name')

        # calculate the center of the polyhedron
        center = Vector([0, 0, 0])
        for vertex in vertices:
            center += Vector(vertex)
        center /= len(vertices)
        self.center = center

        objects = []

        location = self.get_from_kwargs('location', [0, 0, 0])
        if 'location' in kwargs:
            kwargs.pop('location')

        vertex_colors = self.get_from_kwargs('vertex_colors', ['example'])
        face_colors=self.get_from_kwargs('face_colors',['drawing'])

        for i, f in enumerate(faces):
            face = Face(self.vertices, f, center, index=i, index_base=self.index_base,
                        colors=[vertex_colors, ['text'], face_colors], vertex_radius=self.vertex_radius, **kwargs)
            objects.append(face)
            self.faces.append(face)

        super().__init__(children=objects, name=name, location=location, **kwargs)

        if self.coordinate_system:
            self.coordinate_system.add_object(self)

    @classmethod
    def from_group(cls, group, start, eps=1.e-4, **kwargs):
        """
        The full polyhedron is generated from the group provided that a polyhedron is generated with all its edges
        having the same length :param group: :param start: :param eps: :return:
        """
        cls.group = group
        cls.start = start
        cls.EPS = eps

        vertices = []
        cls.word_vertex_dict = {}
        for element in group.elements:
            # ugly conversion between numpy and mathutils
            vertex = Vector((element.matrix @ np.matrix([start]).transpose()).transpose().tolist()[0])
            # calculate distance to all verticex in the list
            # only append vertex if it is different from all others
            different = True
            for v in vertices:
                diff = v-vertex
                if diff.dot(diff)<0.1:
                    different = False
                    break

            if different:
                vertices.append(vertex)
                cls.word_vertex_dict[str(element)] = vertex

        # for v in vertices:
        #     print(v)

        # calculate edges
        min_distance = np.Infinity
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                distance = (vertices[i] - vertices[j]).length
                if distance < min_distance:
                    min_distance = distance

        edges = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                distance = (vertices[i] - vertices[j]).length
                if np.abs(distance - min_distance) < cls.EPS:
                    edges.append(set([i, j]))

        # find at least three edges in a plane to form a face
        face_subspaces = set()
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                e1 = edges[i]
                e2 = edges[j]
                e12 = e1.union(e2)
                # two edges with common vertex
                if len(e12) == 3:
                    plane = Subspace(*tuple(e12), vertices, eps=cls.EPS)
                    for m in range(len(vertices)):
                        if m not in e12:
                            plane.add(m)
                    face_subspaces = face_subspaces.union({plane})

        face_subspaces = sorted(face_subspaces)
        faces = []
        for subspace in face_subspaces:
            faces.append(list(subspace.indices))

        return Polyhedron(vertices, faces, index_base=0, **kwargs)

    def __repr__(self):
        if self.group:
            f"{self.__class__.__name__}(group={self.group})"
        else:
            f"{self.__class__.__name__}(vertices={self.vertices})"

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        for face in self.faces:
            face.appear(begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def grow(self, index=0, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, show_faces=True):
        super().appear(begin_time=begin_time, transition_time=0)  # needed for linking
        self.create_face_tree(index)
        self.counter = 0  # counts the number of faces that have been grown already, (unskillful hack with a global variable)
        max_level = self.max_level(self.root, 0)
        dt = transition_time / len(self.faces)  # time per face
        self.grow_recursively(self.root, begin_time=begin_time, transition_time=dt, show_faces=show_faces)
        return begin_time+transition_time

    def grow_without_faces(self, index=0, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        super().grow(index, begin_time, transition_time, faces=False)

    def grow_recursively(self, face_node, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, show_faces=True):
        face_node.grow(begin_time=begin_time + self.counter * transition_time, transition_time=transition_time)
        self.counter += 1
        for child in face_node.children:
            self.grow_recursively(child, begin_time=begin_time, transition_time=transition_time, show_faces=show_faces)

    def max_level(self, tree, level):
        if not tree.children:
            return level
        else:
            tmp = 0
            for child in tree.children:
                c_level = self.max_level(child, level + 1)
                if tmp < c_level:
                    tmp = c_level
            return tmp

    def create_face_tree(self, root_face_index):
        face_indices = []
        for i in range(len(self.faces)):
            face_indices.append(i)

        self.root = self.faces[root_face_index]
        face_indices.remove(root_face_index)
        self.root.parent = None

        current_level_faces = [self.root]
        while len(current_level_faces) > 0:
            next_level_faces = []
            for face in current_level_faces:
                to_be_removed = []
                for index in face_indices:
                    if face.is_neighbour_of(self.faces[index]):
                        neighbour = self.faces[index]
                        next_level_faces.append(neighbour)
                        neighbour.parent = face
                        to_be_removed.append(index)
                for tbr in to_be_removed:  # remove face indices that have been identified as neighbours
                    face_indices.remove(tbr)
            current_level_faces = next_level_faces.copy()

    def show_face_tree(self):
        for pre, fill, node in RenderTree(self.root):
            tree_str = u"%s%s" % (pre, str(node))
            print(tree_str.ljust(8))

    def unfold(self, fraction=1, begin_time=0, resolution=10, transition_time=OBJECT_APPEARANCE_TIME):
        dt = transition_time / resolution
        for child in self.root.children:
            angle, axis, center = child.get_unfolding_parameters(self.index_base)

            for r in range(resolution + 1):
                d_angle = angle / resolution * fraction
                alpha = d_angle * r
                quaternion_axis = axis * np.sin(alpha / 2)
                quaternion = Quaternion([np.cos(alpha / 2), *quaternion_axis[:]])
                translation = center - quaternion @ center  # translation that compensates the rotation around a
                # center different from the origin
                self.recursively_unfold(child, quaternion, translation, begin_time + r * dt, dt,
                                        fraction * r / resolution)
        return begin_time+transition_time

    def recursively_unfold(self, face, quaternion, translation, begin_time, transition_time, fraction):
        face.rotate(rotation_quaternion=quaternion, begin_time=begin_time, transition_time=transition_time)
        face.move_to(translation, begin_time=begin_time, transition_time=transition_time)

        for child in face.children:
            angle, axis, center = child.get_unfolding_parameters(self.index_base)

            # transform axis and center of rotation to the new values according to the transformation of the parents
            axis = quaternion @ axis  # adjust axis of rotation to the transformed face, only rotations need to be
            # considered, since the position of the axis is unimportant
            center = quaternion @ center + translation  # adjust the center of rotation
            alpha = angle * fraction
            quaternion2_axis = axis * np.sin(alpha / 2)
            quaternion2 = Quaternion([np.cos(alpha / 2), *quaternion2_axis[:]])

            # the local transformation of this child with respect to its parent is a rotation of quaternion2 around
            # the center center the transformation is composed with the transformation provided by the parent and
            # grand parents
            self.recursively_unfold(child, quaternion2 @ quaternion, center + quaternion2 @ (translation - center),
                                    begin_time, transition_time, fraction)

    def disappear_faces(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME
                        ):
        for face in self.faces:
            face.disappear_polygon(begin_time=begin_time, transition_time=transition_time)

    def write_word(self, word, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, modus='top', **kwargs):
        if modus == 'top':
            shift = Vector((0, 0, 1))
        if modus == 'right':
            shift = Vector((1, 0, 0))
        if modus == 'front':
            shift = Vector((0, -1, 0))

        if word in self.group.words:
            vertex = self.word_vertex_dict[word]
            location = vertex + self.vertex_radius * shift
            if self.coordinate_system:
                location = self.coordinate_system.coords2location(location)

            if 'scale' in kwargs:
                scale = kwargs['scale']
                kwargs.pop('scale')
            else:
                scale = 2

            if 'thickness' in kwargs:
                thickness = kwargs['thickness']
                kwargs.pop('thickness')
            else:
                thickness = 3

            bword = SimpleTexBObject(word, location=location, scale=scale, thickness=thickness,
                                     **kwargs)
            self.coordinate_system.add_object(bword)
            bword.write(begin_time=begin_time, transition_time=transition_time)

    def change_emission(self,from_value=0,to_value=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for face in self.faces:
            face.change_emission(from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)
