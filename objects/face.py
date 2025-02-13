from collections import OrderedDict

import numpy as np
from anytree import NodeMixin
from mathutils import Vector, Quaternion

from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.polygon import Polygon
from objects.geometry.sphere import Sphere
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME


class Face(BObject, NodeMixin):
    """
    Create a face for a polyhedron
    """

    def __init__(self, vertices, face, origin, index=0, index_base=1, colors=['example', 'text', 'drawing'], **kwargs):
        """
        :param vertices: a list of vertices
        :param face: the vertices of the list, that define the face
        :param origin: the origin of the corresponding polyhedron
        :param index:
        :param index_base:
        :param kwargs:
        """

        self.kwargs = kwargs
        self.face = face
        self.vertices = vertices
        self.normal = None
        self.center = None

        self.name = self.get_from_kwargs('name', 'face_' + str(index))
        if 'name' in kwargs:
            kwargs.pop('name')

        # sort vertices to obtain the correct orientation of the face
        self.center = Vector([0, 0, 0])
        for vert_index in face:
            self.center += Vector(vertices[vert_index - index_base])
        self.center /= len(face)

        dic = {}
        self.normal = (self.center - origin).normalized()
        ref_index = face[0]
        # create local coordinate system
        x_vec = (Vector(vertices[ref_index]) - self.center).normalized()
        y_vec = self.normal.cross(x_vec)
        for index in face:
            v = Vector(vertices[index - index_base]) - self.center
            x = v.dot(x_vec)
            y = v.dot(y_vec)
            phi = np.angle(x + 1j * y)
            phi *= 10
            phi = int(phi)
            dic[phi] = index

        sorted_dic = OrderedDict(sorted(dic.items()))

        poly_vertices = []
        mapping = {}
        poly_face = []
        n = 0
        for v, i in sorted_dic.items():
            poly_vertices.append(vertices[i - index_base])
            mapping[i] = n
            poly_face.append(n)
            n += 1

        poly_edges = []
        n = len(poly_vertices)
        for i in range(len(poly_vertices)):
            poly_edges.append([i, (i + 1) % n])

        sphere_colors, edge_colors, face_colors = colors

        self.polygon = Polygon(poly_vertices, poly_edges, index=i, color=face_colors[0], transmission=1, roughness=0,
                               brightness=1, thickness=0.02)
        self.vertex_spheres = []

        radius = self.get_from_kwargs('vertex_radius', 0.2)

        if 'vertex_radius' in kwargs:
            kwargs.pop('vertex_radius')

        for i, vertex_index in enumerate(face):
            if vertex_index<len(sphere_colors):
                color = sphere_colors[vertex_index]
            else:
                color = sphere_colors[-1]
            index = self.vertices.index(vertices[vertex_index - index_base])
            self.vertex_spheres.append(Sphere(radius, resolution=3, location=vertices[vertex_index - index_base],
                                              name="vertex_" + str(index), color=color, brightness=1,
                                              roughness=0, **kwargs, smooth=2))

        radius = self.get_from_kwargs('edge_radius', 0.02)
        if 'edge_radius' in kwargs:
            kwargs.pop('edge_radius')
        self.edge_cylinders = []

        for i, edge in enumerate(poly_edges):
            self.edge_cylinders.append(
                Cylinder.from_start_to_end(start=poly_vertices[edge[0]], end=poly_vertices[edge[1]], radius=radius,
                         name="edge_" + str(i), color=edge_colors[0], metallic=1, roughness=0, brighness=1))

        super().__init__(children=[self.polygon, *self.vertex_spheres, *self.edge_cylinders], name=self.name, **kwargs)

    def get_location_of_vertex(self,idx):
        """
        returns the location of the idx vertex of the face
        :param idx:
        :return:
        """
        return self.vertices[self.face[idx]]

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        for vertex_sphere in self.vertex_spheres:
            vertex_sphere.appear(begin_time=begin_time, transition_time=transition_time)
        for edge_cylinder in self.edge_cylinders:
            edge_cylinder.appear(begin_time=begin_time, transition_time=transition_time)
        self.polygon.appear(begin_time=begin_time, transition_time=transition_time)

    def disappear_polygon(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME
                          ):
        self.polygon.disappear(begin_time=begin_time, transition_time=transition_time)

    def grow(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME,show_face=True):
        super().appear(begin_time=begin_time, transition_time=0)  # need for linking
        half = transition_time / 2
        n = len(self.vertex_spheres)
        dt = half / n
        for i in range(n):
            self.vertex_spheres[i].grow(begin_time=begin_time + i * dt, transition_time=dt, modus='from_center')
            if i > 0:
                self.edge_cylinders[i - 1].grow(begin_time=begin_time + i * dt, transition_time=dt)
        self.edge_cylinders[-1].grow(begin_time=begin_time + half, transition_time=dt)
        if show_face:
            self.polygon.appear(begin_time=begin_time + half, transition_time=half)

    def change_color(self,new_color, begin_time=0,transition_time=0):
        self.polygon.change_color(new_color,begin_time=begin_time,transition_time=transition_time)

    def is_neighbour_of(self, other):
        """
        two faces are neighbours if they share at least one vertex
        :param other:
        :return:
        """
        return len(list(set(self.face) & set(other.face))) > 0

    def common_edge(self, other, index_base):
        """
        return the center and the unit direction of the edge that is common to self and other
        :param index_base:
        :param other: other polygon
        :return: None, if there is no common_edge, otherwise [center, direction]
        """
        common_points = list(set(self.face) & set(other.face))
        if len(common_points) < 2:
            return None
        else:
            center = Vector([0, 0, 0])
            for i in range(2):
                center += Vector(self.vertices[common_points[i] - index_base])
            center /= 2
            diff = (Vector(self.vertices[common_points[1] - index_base]) - Vector(
                self.vertices[common_points[0] - index_base])).normalized()
            print(diff)
            return [center, diff]

    def get_unfolding_parameters(self, index_base):
        if self.parent is None:
            return Quaternion()
        else:
            [center, axis] = self.common_edge(self.parent, index_base)
            sign = self.parent.normal.cross(axis).dot(center - self.parent.center)
            if np.abs(sign) > 0:
                sign /= np.abs(sign)
            else:
                sign = 1
            angle = sign * np.arccos(self.parent.normal.dot(self.normal))
            return [angle, axis, center]

    def change_emission(self,from_value=0, to_value=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for sphere in self.vertex_spheres:
            sphere.change_emission(from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)
        for edge in self.edge_cylinders:
            edge.change_emission(from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)

    def __str__(self):
        return str(self.face)
