import bmesh
import numpy as np
import bpy
from mathutils import Vector

from appearance.textures import make_complex_function_material, make_transformations_and_complex_material
from interface import ibpy
from objects.bobject import BObject
from objects.plane import Plane
from objects.plane_complex import ComplexPlane
from objects.transformable_objects.transformable_object import TransBObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE
from utils.utils import to_vector


class Disc2(TransBObject):
    """
        creates a disc with a decent mesh. The radius and the center of the disc are incorporated into the vertex coordinates

        :param r:
        :param center:
        :param resolution:
        :param transformations:
        :param kwargs:
    """

    def __init__(self, r=1, center=Vector(), resolution=10, transformations=None, **kwargs):
        self.kwargs = kwargs
        center = to_vector(center)

        if not isinstance(resolution, list):
            resolution = [resolution, resolution]

        self.name = self.get_from_kwargs('name', 'Disc')
        mesh = self.create_mesh(r, center, resolution)
        super().__init__(name=self.name, mesh=mesh, transformations=transformations, **kwargs)

    def create_mesh(self, r, center, resolution=10):
        vertices = []
        edges = []
        faces = []

        res_r = int(resolution[0])
        res_phi = int(resolution[1])

        dr = r / res_r
        for i in range(0, int(res_r + 1)):
            r = i * dr
            # create vertices
            if r == 0:
                vertices.append([r + center.x, center.y, center.z])
            else:
                dphi = np.pi * 2 / res_phi
                for j in range(0, int(res_phi)):
                    phi = j * dphi
                    vertices.append([r * np.cos(phi) + center.x, r * np.sin(phi) + center.y, center.z])
                    v = len(vertices)
                    if j > 0:
                        edges.append([v - 1, v - 2])
                        if i > 1:
                            edges.append([v - 1, v - res_phi - 1])
                            if j > 0:
                                faces.append([v - 2, v - 1, v - 1 - res_phi, v - 2 - res_phi])
                        else:
                            edges.append([v - 1, 0])
                            if j > 0:
                                faces.append([v - 2, v - 1, 0])
                edges.append([v - 1, v - res_phi])
                # connect to
                if i > 1:
                    faces.append([v - 1, v - res_phi, v - 2 * res_phi, v - 1 - res_phi])
                else:
                    faces.append([v - 1, v - res_phi, 0])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh


class Annulus2(TransBObject):
    """
    create an annulus with a decent mesh which is not related to a complex plane
    :param r = [1,2]
    :param phi = [0,2*np.pi]
    The location of the annulus can be customized, default
    :param location =[0,0,0]
    The mesh resolution can be chosen, default
    :param resolution = 10
    :param transformations possible transformations of the vertices
    """

    def __init__(self, r=[0.5, 1], phi=[0, 2 * np.pi], resolution=10,
                 transformations=None, **kwargs):
        self.kwargs = kwargs
        self.phi = phi
        self.resolution = resolution
        self.r = r
        self.name = self.get_from_kwargs('name', 'Annulus')

        # start with a plane

        self.plane = ibpy.add_plane()
        self.create_mesh()
        super().__init__(obj=self.plane, transformations=transformations, name=self.name, **kwargs)

    # def create_mesh(self):
    #     """
    #     overrides the mesh creation function of the super class
    #     :return:
    #     """
    #     bm = bmesh.new()  # Creates an empty BMesh
    #     bm.from_mesh(self.plane.data)  # Fills it in using the plane
    #     plane_size = [np.abs(self.u[1] - self.u[0]) / 2, np.abs(self.v[1] - self.v[0]) / 2, 1]  # Takes user inputs
    #     # Subdividing, scaling, and moving the plane according to user inputs:
    #     bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=self.resolution, use_grid_fill=True)
    #     try_to_make_mesh_uniform(bm, self.u, self.v)
    #     bmesh.ops.scale(bm, vec=plane_size, verts=bm.verts)
    #     bmesh.ops.translate(bm, vec=Vector([0.5 * (self.u[1] + self.u[0]), 0, 0]), verts=bm.verts)
    #     map_to_annulus(bm)
    #     bm.to_mesh(self.plane.data)
    #     bm.free()

    def create_mesh(self):
        '''
        overrides the mesh creation function of the super class
        :return:
        '''

        # completely rewrite the mesh
        vertices = []
        edges = []
        faces = []

        singular = False
        dr = (self.r[1] - self.r[0]) / self.resolution
        for i in range(0, self.resolution + 1):
            r = self.r[0] + i * dr
            # create vertices
            if r == 0:
                vertices.append((r, 0, 0))
                singular = True
            else:
                dphi = (self.phi[1]-self.phi[0]) / self.resolution
                for j in range(0, self.resolution):
                    phi = self.phi[0]+j * dphi
                    vertices.append((r * np.cos(phi), r * np.sin(phi), 0))
                    v = len(vertices)
                    if j > 0:
                        edges.append([v - 1, v - 2])
                        # make connection to vertices with r-dr
                        if i > 0:
                            if not singular:
                                edges.append([v - 1, v - self.resolution - 1])
                                if j > 0:
                                    faces.append([v - 2, v - 1, v - 1 - self.resolution, v - 2 - self.resolution])
                            else:
                                edges.append([v - 1, 0])
                                if j > 0:
                                    faces.append([v - 2, v - 1, 0])

                # close loop
                if self.phi[0]==0 and self.phi[1]==2*np.pi:
                    edges.append([v - 1, v - self.resolution])
                    if i > 0:
                        if not singular:
                            faces.append([v - 1, v - self.resolution, v - 2 * self.resolution, v - 1 - self.resolution])
                        else:
                            faces.append([v - 1, v - self.resolution, 0])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        self.plane.data=new_mesh


def try_to_make_mesh_uniform(bm, u, v):
    """
    if u and v coordinates differ in extend, there are additional mesh edges added to make the mesh more uniform

    for instance if u ranges from [-10,10] and v ranges from [-1,1] there are 2^3 additional lines added between the lines of constant u
    :param u:
    :param v:
    :return:
    """
    # try to make the distance between mesh lines uniform in both dimensions u and v
    lu = u[1] - u[0]
    lv = v[1] - v[0]
    ratio = lu / lv

    if ratio > 1:
        selected_edges = find_edges_of_constant_v(bm)
        bmesh.ops.subdivide_edges(bm, edges=selected_edges, cuts=int(np.round(ratio)), use_grid_fill=True)
    else:
        ratio = lv / lu
        selected_edges = find_edges_of_constant_u(bm)
        bmesh.ops.subdivide_edges(bm, edges=selected_edges, cuts=int(np.round(ratio)), use_grid_fill=True)


def approx(a, precision):
    return np.round(a * 10 ** precision) / 10 ** precision


def find_edges_of_constant_u(bm):
    edge_list = []
    for e in bm.edges:
        if approx(e.verts[0].co[0] - e.verts[1].co[0], 2) == 0:
            edge_list.append(e)
    return edge_list


def find_edges_of_constant_v(bm):
    edge_list = []
    for e in bm.edges:
        if approx(e.verts[0].co[1] - e.verts[1].co[1], 2) == 0:
            edge_list.append(e)
    return edge_list


class Annulus(ComplexPlane):
    """
    Create a annulus with a descent mesh:
    The uv-coordinates of the annulus can be customized, default
    :param r = [1,2]
    :param phi = [0,2*np.pi]
    The location of the annulus can be customized, default
    :param location =[0,0,0]
    The mesh resolution can be chosen, default
    :param resolution = 10
    """

    def __init__(self, r=[1, 2], phi=[0, 2 * np.pi], location=None, resolution=10, coordinate_system=None,
                 functions=None, **kwargs):
        self.kwargs = kwargs
        if location is None:
            location = (0.5 * (r[1] + r[0]), 0, 0)

        self.phi = phi
        self.resolution = resolution
        self.location = location
        self.r = r

        self.name = self.get_from_kwargs('name', 'Annulus')

        # if coordinate_system is None:
        #     raise "Annulus must be created inside a coordinate system"
        super().__init__(coordinate_system, functions, u=r, v=phi, location=location, resolution=resolution, name=self.name,
                         **kwargs)

    def create_mesh(self):
        '''
        overrides the mesh creation function of the super class
        :return:
        '''

        # completely rewrite the mesh
        vertices = []
        edges = []
        faces = []

        singular = False
        if isinstance(self.resolution,list):
            resolution=self.resolution[0]
        else:
            resolution=self.resolution
        dr = (self.r[1] - self.r[0]) / resolution
        for i in range(0, resolution + 1):
            r = self.r[0] + i * dr
            # create vertices
            if r == 0:
                vertices.append((r, 0, 0))
                singular = True
            else:
                dphi = np.pi * 2 / resolution
                for j in range(0, resolution):
                    phi = j * dphi
                    vertices.append((r * np.cos(phi), r * np.sin(phi), 0))
                    v = len(vertices)
                    if j > 0:
                        edges.append([v - 1, v - 2])
                        # make connection to vertices with r-dr
                        if i > 0:
                            if not singular:
                                edges.append([v - 1, v - resolution - 1])
                                if j > 0:
                                    faces.append([v - 2, v - 1, v - 1 - resolution, v - 2 - resolution])
                            else:
                                edges.append([v - 1, 0])
                                if j > 0:
                                    faces.append([v - 2, v - 1, 0])

                # close loop
                edges.append([v - 1, v - resolution])
                if i > 0:
                    if not singular:
                        faces.append([v - 1, v - resolution, v - 2 * resolution, v - 1 - resolution])
                    else:
                        faces.append([v - 1, v - resolution, 0])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        self.plane.data=new_mesh


def map_to_annulus(bm):
    for vertex in bm.verts:
        r, phi = vertex.co[0:2]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        vertex.co.x = x
        vertex.co.y = y
