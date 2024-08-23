import bmesh
import numpy as np
from interface import ibpy
from objects.bobject import BObject
from utils.utils import quaternion_from_normal


class Plane(BObject):
    """
    Create a plane with a descent mesh:

    Since the standard blender plane only consists of 4 mesh points, the mesh is constructed manually.
    This is important, if the plane is subject to transformations

    The uv-coordinates of the plane can be customized, default
    :param u=[-1,1]
    :param v=[-1,1]
    The location of the plane can be customized, default
    :param location =[0,0,0]
    The mesh resolution can be chosen, default
    :param resolution = 10
    """

    def __init__(self, u=[-1, 1], v=[-1, 1], normal =None, resolution=10, uniformization=True, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', "Plane")

        self.plane = ibpy.add_plane(name=self.name+"Mesh")
        self.apply_scale=self.get_from_kwargs('apply_scale',True)
        self.apply_location=self.get_from_kwargs('apply_location',True)

        if self.apply_scale:
            scale = self.get_from_kwargs('scale', [1, 1, 1])
            if isinstance(scale,float) or isinstance(scale,int):
                scale=[scale]*3
            self.u = [scale[0]*u[0],scale[0]*u[1]]
            self.v = [scale[1]*v[0],scale[1]*v[1]]
        else:
            self.u=u
            self.v=v

        if isinstance(resolution,list):
            self.resolution=resolution
        else:
            self.resolution=[resolution,resolution]

        if self.apply_location:
            location = self.get_from_kwargs('location',None)
            if location is None:
                location = ((u[1] + u[0]) / 2, (v[1] + v[0]) / 2, 0)
            self.location = location

        self.uniformization = uniformization

        self.subdivide_boundary =self.get_from_kwargs('subdivide_boundary',False)
        self.create_mesh()


        if normal:
            quat = quaternion_from_normal(normal)
            super().__init__(obj=self.plane, name=self.name,rotation_quaternion=quat,**kwargs)
        else:
            super().__init__(obj=self.plane, name=self.name,**kwargs)

    def create_mesh(self):
        bm = bmesh.new()  # Creates an empty BMesh
        bm.from_mesh(self.plane.data)  # Fills it in using the plane
        plane_size = [np.abs(self.u[1] - self.u[0]) / 2, np.abs(self.v[1] - self.v[0]) / 2, 1]  # Takes user inputs

        # Subdividing, scaling, and moving the plane according to user inputs:
        if self.resolution[0]>0:
            edges = find_edges_of_constant_u(bm)
            bmesh.ops.subdivide_edges(bm, edges=edges, cuts=self.resolution[0], use_grid_fill=True)

        if self.resolution[1] > 0:
            edges = find_edges_of_constant_v(bm)
            bmesh.ops.subdivide_edges(bm, edges=edges, cuts=self.resolution[1], use_grid_fill=True)

        if self.apply_scale:
            bmesh.ops.scale(bm, vec=plane_size, verts=bm.verts)
        if self.apply_location:
            bmesh.ops.translate(bm, vec=self.location, verts=bm.verts)
        if self.uniformization:
            try_to_make_mesh_uniform(bm,self.u,self.v)
        if self.subdivide_boundary:
            subdivide_boundary(bm,self.u,self.v)

        bm.to_mesh(self.plane.data)
        bm.free()


def find_edges_of_constant_u(bm):
    edge_list = []
    for e in bm.edges:
        if approx(e.verts[0].co[0] - e.verts[1].co[0], 0):
            edge_list.append(e)
    return edge_list


def find_edges_of_constant_v(bm):
    edge_list = []
    for e in bm.edges:
        if approx(e.verts[0].co[1] - e.verts[1].co[1], 0):
            edge_list.append(e)
    return edge_list


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
        selected_edges = find_edges_of_constant_u(bm)
        bmesh.ops.subdivide_edges(bm, edges=selected_edges, cuts=int(np.round(ratio)), use_grid_fill=True)
    else:
        ratio = lv/lu
        selected_edges = find_edges_of_constant_v(bm)
        bmesh.ops.subdivide_edges(bm, edges=selected_edges, cuts=int(np.round(ratio)), use_grid_fill=True)


def subdivide_boundary(bm,u,v):
    """
    this function introduces loop cuts at the boundary

    :param bm:
    :param u:
    :param v:
    :return:
    """
    # left boundary
    u_min = u[0]
    for i in range(4):
        selected=[edge for edge in bm.edges if (edge.verts[0].co[0]==u_min or edge.verts[1].co[0]==u_min) and np.abs(edge.verts[0].co[1]-edge.verts[1].co[1])<0.0000001]
        # for edge in selected:
        #     print(edge.verts[0].co,edge.verts[1].co)
        # print(len(selected))
        bmesh.ops.subdivide_edges(bm, edges=selected, cuts=1, use_grid_fill=True)
    # right boundary
    u_max = u[1]
    for i in range(4):
        selected = [edge for edge in bm.edges if
                    (edge.verts[0].co[0] == u_max or edge.verts[1].co[0] == u_max) and np.abs(
                        edge.verts[0].co[1] - edge.verts[1].co[1]) < 0.0000001]
        # for edge in selected:
        #     print(edge.verts[0].co,edge.verts[1].co)
        # print(len(selected))
        bmesh.ops.subdivide_edges(bm, edges=selected, cuts=1, use_grid_fill=True)

    # bottom boundary
    v_min = v[0]
    for i in range(4):
        selected = [edge for edge in bm.edges if
                    (edge.verts[0].co[1] == v_min or edge.verts[1].co[1] == v_min) and np.abs(
                        edge.verts[0].co[0] - edge.verts[1].co[0]) < 0.0000001]
        # for edge in selected:
        #     print(edge.verts[0].co,edge.verts[1].co)
        # print(len(selected))
        bmesh.ops.subdivide_edges(bm, edges=selected, cuts=1, use_grid_fill=True)

    # top boundary
    v_max = v[1]
    for i in range(4):
        selected = [edge for edge in bm.edges if
                    (edge.verts[0].co[1] == v_max or edge.verts[1].co[1] == v_max) and np.abs(
                        edge.verts[0].co[0] - edge.verts[1].co[0]) < 0.0000001]
        # for edge in selected:
        #     print(edge.verts[0].co,edge.verts[1].co)
        # print(len(selected))
        bmesh.ops.subdivide_edges(bm, edges=selected, cuts=1, use_grid_fill=True)


def close(a, b):
    return np.abs(a - b) == 0


def approx(a, precision):
    return np.round(a * 10 ** precision) / 10 ** precision
