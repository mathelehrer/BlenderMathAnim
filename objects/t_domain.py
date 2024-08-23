import bmesh
import numpy as np
import bpy

from appearance.textures import make_complex_function_material, make_conformal_transformation_material
from interface import ibpy
from objects.bobject import BObject
from objects.plane import Plane
from objects.plane_complex import ComplexPlane
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE


class TDomain(BObject):
    """
    Create a disc with a descent mesh and with four discs removed:
    """

    def __init__(self, radius=10, resolution=10, removed_circles=[[1 + 1j, 1], [-1 - 1j, 1]], **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name', 'T')

        if 'conformal_transformations' in kwargs:
            conformal_transformations = kwargs['conformal_transformations']
        else:
            conformal_transformations = None

        mesh = self.create_mesh2(r_max=radius, res=resolution, removed_circles=removed_circles)
        super().__init__(name=name, mesh=mesh, **kwargs)

        # only call this after the mesh has been created
        if conformal_transformations is not None:
            self.mixer_dialers = make_conformal_transformation_material(self, conformal_transformations, name=name)
            self.current_mapping = 0

    def next_shape(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        """
        transition to the next shape of the plane
        :param begin_time:
        :param transition_time:
        :return:
        """
        appear_frame = begin_time * FRAME_RATE
        dialer_index = self.current_mapping - 1
        if len(self.mixer_dialers)>0 and len(self.mixer_dialers)>dialer_index:
            current_dialer = self.mixer_dialers[dialer_index]
            ibpy.insert_keyframe(current_dialer, 'default_value', appear_frame)
            current_dialer.default_value = 1
            ibpy.insert_keyframe(current_dialer, 'default_value', appear_frame + transition_time * FRAME_RATE)

        ibpy.morph_to_next_shape(self.ref_obj, self.current_mapping, appear_frame, transition_time * FRAME_RATE)
        self.current_mapping += 1
        print("Next shape at " + str(begin_time) + " with transition time " + str(transition_time))

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time,**kwargs)
        ibpy.morph_to_next_shape(self.ref_obj, 0, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        self.current_mapping+=1

    def create_mesh2(self, r_max=10, res=10, removed_circles=None):
        """
        the main idea of this algorithm is that the first and last vertex that is removed by a disc is shifted back along the coordinate line to the boundary of the disc
        :param d_r:
        :param res:
        :param removed_circles:
        :return:
        """
        dr = r_max / res
        dphi = 2 * np.pi / res

        index_matrix = []  # a rectangular arrangement of the vertex indices to find neighbours more easily
        # the values along a ray of constant phy are stored inside a row of the matrix
        connection_matrix = []  # stores whether a vertex is connectable or not

        vertices = []
        edges = []
        faces = []

        null_vertex = (0, 0, 0)
        vertices.append(null_vertex)
        for row in range(0, res + 1):
            phi = dphi * row
            row_vertices = []
            row_connection = []
            for col in range(0, res + 1):
                # rows are lines of constant r
                r = dr * col
                if r == 0:
                    row_vertices.append(0)  # the same vertex is added for all positions to access neighbours more
                    # easily
                    row_connection.append(1)
                else:
                    x = r * np.cos(phi)
                    y = r * np.sin(phi)
                    vertex = (x, y, 0)
                    row_vertices.append(len(vertices))
                    row_connection.append(1)
                    vertices.append(vertex)
            index_matrix.append(row_vertices)
            connection_matrix.append(row_connection)

        # disconnect vertices inside discs
        for disc in removed_circles:
            d_center, d_r = disc
            for row, cols in enumerate(index_matrix):
                for col in range(len(cols)):
                    c_index = connection_matrix[row][col]
                    index = index_matrix[row][col]
                    if c_index > 0:
                        vertex = vertices[index]
                        zv = vertex[0] + 1j * vertex[1]
                        if np.abs(d_center - zv) < d_r:
                            connection_matrix[row][col] = 0

            # move first and last disconnected vertex of each row onto the circle and reconnect it
            zm2 = np.abs(d_center) ** 2
            dr2 = d_r * d_r
            for row in range(0, res + 1):
                c_last = 1
                for col in range(0, res + 1):
                    c = connection_matrix[row][col]
                    if c_last > 0 and c == 0:
                        c_last = 0
                        r = dr * col
                        r2 = r * r
                        dr2 = d_r * d_r
                        vertex = vertices[index_matrix[row][col]]
                        v = vertex[0] + 1j * vertex[1]  # old pos
                        p_half = -np.real(d_center * np.conj(v)) / r2
                        q = (zm2 - dr2) / r2
                        discriminant = p_half * p_half - q
                        if discriminant >= 0:
                            l = -p_half - np.sqrt(discriminant)
                            v_new = l * v
                            vertices[index_matrix[row][col]] = [np.real(v_new), np.imag(v_new), 0]
                            connection_matrix[row][col] = 1
                    if c_last == 0 and c > 0:
                        r = dr * (col - 1)
                        r2 = r * r
                        vertex = vertices[index_matrix[row][col - 1]]
                        v = vertex[0] + 1j * vertex[1]  # old pos
                        p_half = -np.real(d_center * np.conj(v)) / r2
                        q = (zm2 - dr2) / r2
                        discriminant = p_half * p_half - q
                        if discriminant >= 0:
                            l = -p_half + np.sqrt(discriminant)
                            print(row, col, l)
                            v_new = l * v
                            vertices[index_matrix[row][col - 1]] = [np.real(v_new), np.imag(v_new), 0]
                            connection_matrix[row][col - 1] = 1
                        break

            # move first and last disconnected vertex of each col onto the circle and reconnect i
            for col in range(1, res + 1):
                c_last = 1
                r = dr * col
                r2 = r * r
                for row in range(0, res + 1):
                    phi = dphi * row
                    c = connection_matrix[row][col]
                    if c_last > 0 and c == 0:
                        c_last = 0
                        discriminant = 2 * (r2 + dr2) / zm2 - ((r2 - dr2) / zm2) ** 2 - 1
                        if discriminant > 0:
                            sqrt_d = np.sqrt(discriminant)
                            sol1 = 0.5 * (1 + (r2 - dr2) / zm2) * d_center - 0.5 * sqrt_d * 1j * d_center
                            sol2 = 0.5 * (1 + (r2 - dr2) / zm2) * d_center + 0.5 * sqrt_d * 1j * d_center

                            phi1 = np.angle(sol1)
                            if phi1 < 0:
                                phi1 += 2 * np.pi
                            phi2 = np.angle(sol2)
                            if phi2 < 0:
                                phi2 += 2 * np.pi

                            if phi2 < phi1:
                                phi1, phi2 = phi2, phi1
                            print(phi1, phi2, phi)
                            vertices[index_matrix[row][col]] = (r * np.cos(phi1), r * np.sin(phi1), 0)
                            connection_matrix[row][col] = 1
                    if c_last == 0 and c > 0:
                        discriminant = 2 * (r2 + dr2) / zm2 - ((r2 - dr2) / zm2) ** 2 - 1
                        if discriminant > 0:
                            sqrt_d = np.sqrt(discriminant)
                            sol1 = 0.5 * (1 + (r2 - dr2) / zm2) * d_center - 0.5 * sqrt_d * 1j * d_center
                            sol2 = 0.5 * (1 + (r2 - dr2) / zm2) * d_center + 0.5 * sqrt_d * 1j * d_center

                            phi1 = np.angle(sol1)
                            if phi1 < 0:
                                phi1 += 2 * np.pi
                            phi2 = np.angle(sol2)
                            if phi2 < 0:
                                phi2 += 2 * np.pi

                            if phi2 < phi1:
                                phi1, phi2 = phi2, phi1
                            print(phi1, phi2, phi)
                            vertices[index_matrix[row][col - 1]] = (r * np.cos(phi2), r * np.sin(phi2), 0)
                            connection_matrix[row][col - 1] = 1
                        break

            # mark remaining vertices inside the disc as deleted
            # in this way they are not considered in the calculations of other discs
            for row, cols in enumerate(index_matrix):
                for col in range(len(cols)):
                    c_index = connection_matrix[row][col]
                    if c_index == 0:
                        connection_matrix[row][col] = -1

        # create edges
        cm = connection_matrix
        im = index_matrix
        for col in range(1, res + 1):
            for row in range(1, res + 1):
                if cm[row][col] > 0:
                    if cm[row][col - 1] > 0:
                        edges.append((im[row][col - 1], im[row][col]))
                    if cm[row - 1][col] > 0:
                        edges.append((im[row][col], im[row - 1][col]))

        # create faces
        for row in range(0, res):
            if cm[0][0] > 0 and cm[row + 1][1] > 0 and cm[row][1] > 0:
                faces.append((im[0][0], im[row + 1][1], im[row][1]))

        for col in range(1, res):
            for row in range(0, res):
                if cm[row][col] > 0 and cm[row + 1][col + 1] > 0 and cm[row + 1][col] > 0 and cm[row][col + 1] > 0:
                    faces.append((im[row][col], im[row + 1][col], im[row + 1][col + 1], im[row][col + 1]))

        new_mesh = bpy.data.meshes.new('t_domain_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()

        return new_mesh

    def create_mesh2(self, r_max=10, res=10, removed_circles=None):
        """
        This algorithm is a lot simpler:
        Every first and last point of the coordinate line that is inside the disc is shifted to the boundary of the disc along the shortest distance

        :param d_r:
        :param res:
        :param removed_circles:
        :return:
        """
        dr = r_max / res
        dphi = 2 * np.pi / res

        index_matrix = []  # a rectangular arrangement of the vertex indices to find neighbours more easily
        # the values along a ray of constant phy are stored inside a row of the matrix
        connection_matrix = []  # stores whether a vertex is connectable or not

        vertices = []
        edges = []
        faces = []

        null_vertex = [0., 0., 0.]
        vertices.append(null_vertex)
        for row in range(0, res + 1):
            phi = dphi * row
            row_vertices = []
            row_connection = []
            for col in range(0, res + 1):
                # rows are lines of constant r
                r = dr * col
                if r == 0:
                    row_vertices.append(0)  # the same vertex is added for all positions to access neighbours more
                    # easily
                    row_connection.append(1)
                else:
                    x = r * np.cos(phi)
                    y = r * np.sin(phi)
                    vertex = [x, y, 0.]
                    row_vertices.append(len(vertices))
                    row_connection.append(1)
                    vertices.append(vertex)
            index_matrix.append(row_vertices)
            connection_matrix.append(row_connection)

        # disconnect vertices inside discs
        for disc in removed_circles:

            d_center, d_r = disc
            for row, cols in enumerate(index_matrix):
                for col in range(len(cols)):
                    c_index = connection_matrix[row][col]
                    index = index_matrix[row][col]
                    if c_index > 0:
                        vertex = vertices[index]
                        zv = vertex[0] + 1j * vertex[1]
                        if np.abs(d_center - zv) < d_r:
                            connection_matrix[row][col] = 0

            # move first and last disconnected vertex of each row onto the circle and reconnect it
            for row in range(0, res + 1):
                c_last = 1
                for col in range(0, res + 1):
                    c = connection_matrix[row][col]
                    if c_last > 0 and c == 0:
                            c_last = 0
                            vertex = vertices[index_matrix[row][col]]
                            v = vertex[0] + 1j * vertex[1]
                            direction = v - d_center
                            direction /= np.abs(direction)
                            v_new = d_center + d_r * direction
                            vertices[index_matrix[row][col]] = [float(np.real(v_new)), float(np.imag(v_new)), 0.]
                            connection_matrix[row][col] = 1
                    if c_last == 0 and c > 0:
                        vertex = vertices[index_matrix[row][col - 1]]
                        v = vertex[0] + 1j * vertex[1]  # old pos
                        direction = v - d_center
                        length = np.abs(direction)
                        if length > 0:
                            direction /= length
                        v_new = d_center + d_r * direction
                        vertices[index_matrix[row][col - 1]] = [float(np.real(v_new)), float(np.imag(v_new)), 0.]
                        connection_matrix[row][col - 1] = 1
                        break

            # move first and last disconnected vertex of each col onto the circle and reconnect i
            for col in range(1, res + 1):
                c_last = 1
                for row in range(0, res + 1):
                    c = connection_matrix[row][col]
                    if c_last > 0 and c == 0:
                        c_last = 0
                        vertex = vertices[index_matrix[row][col]]
                        v = vertex[0] + 1j * vertex[1]
                        direction = v - d_center
                        length = np.abs(direction)
                        if length>0:
                             direction /= length
                        v_new = d_center + d_r * direction
                        vertices[index_matrix[row][col]] = [float(np.real(v_new)), float(np.imag(v_new)), 0.]
                        connection_matrix[row][col] = 1
                    if c_last == 0 and c > 0:
                        vertex = vertices[index_matrix[row][col - 1]]
                        v = vertex[0] + 1j * vertex[1]  # old pos
                        direction = v - d_center
                        direction /= np.abs(direction)
                        v_new = d_center + d_r * direction
                        vertices[index_matrix[row][col - 1]] = [float(np.real(v_new)), float(np.imag(v_new)), 0.]
                        connection_matrix[row][col - 1] = 1
                        break

            # mark remaining vertices inside the disc as deleted
            # in this way they are not considered in the calculations of other discs
            for row, cols in enumerate(index_matrix):
                for col in range(len(cols)):
                    c_index = connection_matrix[row][col]
                    if c_index ==0:
                        connection_matrix[row][col] = -1

        # create edges
        cm = connection_matrix
        im = index_matrix
        for col in range(1, res + 1):
            for row in range(1, res + 1):
                if cm[row][col] > 0:
                    if cm[row][col - 1] > 0:
                        edges.append((im[row][col - 1], im[row][col]))
                    if cm[row - 1][col] > 0:
                        edges.append((im[row][col], im[row - 1][col]))

        # create faces
        for row in range(0, res):
            if cm[0][0] > 0 and cm[row + 1][1] > 0 and cm[row][1] > 0:
                faces.append((im[0][0], im[row + 1][1], im[row][1]))

        for col in range(1, res):
            for row in range(0, res):
                if cm[row][col] > 0 and cm[row + 1][col + 1] > 0 and cm[row + 1][col] > 0 and cm[row][col + 1] > 0:
                    faces.append((im[row][col], im[row + 1][col], im[row + 1][col + 1], im[row][col + 1]))

        # # remove unused vertices
        connected_vertices = []
        ind = []

        for col in range(0, res+1):
            for row in range(0, res+1):
                c = connection_matrix[row][col]
                index = index_matrix[row][col]
                if c > -1:
                    ind.append(index)

        index_set = set(ind)

        for index in index_set:
            connected_vertices.append(vertices[index])

        # create dictionary
        index_list=list(index_set)
        print(index_list[-1])
        mapping={}
        for i in range(len(vertices)):
            if i in index_list:
                position = index_list.index(i)
                mapping.update({str(i): position})

        # re-index edges and faces
        new_edges = []
        for edge in edges:
            new_edge=[]
            for e in edge:
                new_edge.append(mapping[str(e)])
            new_edges.append(new_edge)

        new_faces = []
        for face in faces:
            new_face = []
            for e in face:
                new_face.append(mapping[str(e)])
            new_faces.append(new_face)

        new_mesh = bpy.data.meshes.new('t_domain_mesh')
        new_mesh.from_pydata(connected_vertices, new_edges, new_faces)
        new_mesh.update()

        return new_mesh
