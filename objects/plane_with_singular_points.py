import bmesh
import numpy as np
from objects.plane import Plane, approx, close
from objects.plane_complex import ComplexPlane


class PlaneWithSingularPoints(Plane):
    def __init__(self, special_x=[0.5], special_y=[-0.5, 0.5], u=[-1, 1], v=[-1, 1], location=[0, 0, 0],
                 resolution=10, detail=5,
                 **kwargs):
        """
        Create a plane whose mesh is more dense at the special values of u and v
        :param special_u:
        :param special_v:
        :param u:
        :param v:
        :param location:
        :param resolution: mesh resolution, number of cuts in u and v direction
        :param detail:be careful with larger than 10, the grid might be spoiled due to rounding issues
        :param kwargs:
        """

        self.detail = detail
        self.special_x = special_x
        self.special_y = special_y

        super().__init__(u, v, location, resolution, **kwargs)

        # create a special mesh that is concentrated at singular points and zeros
        bm = bmesh.new()
        bm.from_mesh(self.ref_obj.data)


class ComplexPlaneWithSingularPoints(ComplexPlane):
    def __init__(self, coord, functions, special_x=[0.5], special_y=[-0.5, 0.5], u=[-1, 1], v=[-1, 1],
                 resolution=10, detail=5,
                 **kwargs):
        """
        Create a plane whose mesh is more dense at the special values of u and v
        :param special_x:
        :param special_y:
        :param u:
        :param v:
        :param location:
        :param resolution: mesh resolution, number of cuts in u and v direction
        :param detail:be careful with larger than 10, the grid might be spoiled due to rounding issues
        :param kwargs:
        """
        self.detail = detail
        self.special_x = special_x
        self.special_y = special_y
        super().__init__(coord, functions, u, v, resolution, **kwargs)

    def create_mesh(self):
        super().create_mesh()
        # create a special mesh that is concentrated at singular points and zeros
        bm = bmesh.new()
        bm.from_mesh(self.plane.data)

        for i in range(self.detail):
            for x in self.special_x:
                bmesh.ops.subdivide_edges(bm, edges=get_special_edges(x, bm.edges, 0, i), cuts=1,
                                          use_grid_fill=True)
            for y in self.special_y:
                bmesh.ops.subdivide_edges(bm, edges=get_special_edges(y, bm.edges, 1, i), cuts=1,
                                          use_grid_fill=True)

        # write mesh back to the plane
        bm.to_mesh(self.plane.data)  # Freeing the BMesh, moving on to coloring the domain
        print("Complex plane created with " + str(len(bm.verts)) + " vertices.")
        bm.free()


def get_special_edges(s, edges, direction, iteration):
    """
    find all edges whose vertices enclose the special coordinate value

    there is a subtle  situation when the vertex coordinate coincides with a singular point
    one has to take care that no edges are connected across the singular line
    to achieve symmetric meshes the edges are cut alternatingly below and above the singular line

    :param s: the singular coordinate value around which the mesh lines are concentrated
    :param edges: the edges of the mesh
    :param direction: 0 for u and 1 for v coordinates
    :param iteration: a simple counter to make sure that the mesh lines are distributed evenly around the singular line
    :return:
    """
    selected = []
    for e in edges:
        c0 = e.verts[0].co[direction]
        c1 = e.verts[1].co[direction]

        # round values to a precision of 10**4
        c0 = approx(c0, 4)
        c1 = approx(c1, 4)

        if c1 < c0:
            tmp = c0
            c0 = c1
            c1 = tmp
        if close(s, c0) or close(s, c1):
            if iteration % 2 == 0:
                if close(c0, s) and s < c1:
                    selected.append(e)
            else:
                if c0 < s and close(s, c1):
                    selected.append(e)
        else:
            if c0 < s < c1 or c1 < s < c0:
                selected.append(e)
    return selected