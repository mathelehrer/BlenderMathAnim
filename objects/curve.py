from functools import partial

import bpy
import math
import numpy as np

from appearance.textures import get_texture
from interface import ibpy
from interface.ibpy import add_shape_key, morph_to_next_shape, Vector
from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.geometry.geo_bobject import GeoBObject
from objects.plane import Plane
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs
from utils.utils import to_vector, z2vec
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import InputValue, InputInteger, CurveLine, ResampleCurve, SetPosition, \
    CurveToMesh, CurveCircle, SceneTime, InputMaterial, SetMaterial, create_geometry_line, make_function, Index, \
    DeleteGeometry


def mapping_from_data(data, x):
    n = len(data)
    index = np.round(n * x)
    index %= n

    v = data[int(index)]
    if isinstance(v, Vector):
        return v
    elif isinstance(v, complex):
        return z2vec(v)
    else:
        raise "Don't know how to convert data into vector"


class MeshCurve(BObject):
    """
    A curve that is a mesh object.
    The mesh is constructed with the Frenet-Serret-formula.
    Have a look at Frenet_Serret_Formula.nb
    """

    def __init__(self, f, df, d2f, **kwargs):
        """Create a curve as a true tube mesh, using analytic derivatives for
        the Frenet-Serret frame.

        At each sampled parameter ``s``, the tangent comes from ``df(s)`` and
        the curvature direction from ``d2f(s)``. A circle of radius
        ``thickness * 0.01`` is swept along the curve in the normal/binormal
        plane to form the tube.

        Args:
            f: Callable ``s -> Vector(x, y, z)`` -- the curve's position.
            df: Callable ``s -> Vector`` -- the first derivative of ``f``.
            d2f: Callable ``s -> Vector`` -- the second derivative of ``f``.
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``name`` (str): Defaults to ``'DataCurve'``.
                * ``thickness`` (float): Tube radius multiplier (final
                  radius ``= thickness * 0.01``). Defaults to 1.
                * ``closed`` (bool): If ``True``, the curve is treated as
                  closed (first vertex == last vertex). Defaults to ``False``.
                * ``resolution`` (list[int]): ``[u, v]`` -- ``u`` samples
                  along the curve, ``v`` segments around the tube.
                  Defaults to ``[100, 10]``.
                * ``domain`` (list[float]): ``[s_min, s_max]``. Defaults to
                  ``[0, 1]``.
                * Standard BObject kwargs (``color``, ``location``, ...).
        """
        # defined in create_messh
        self.s1 = None
        self.v = None
        self.u = None
        self.s0 = None
        self.closed = None
        self.thickness = None
        self.old_sk = None

        self.shape = 0
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'DataCurve')

        mesh = self.create_mesh(f, df, d2f)
        super().__init__(mesh=mesh, name=self.name, **kwargs)

    def create_mesh(self, f, df, d2f):
        self.thickness = self.get_from_kwargs('thickness', 1)
        self.closed = self.get_from_kwargs('closed', False)
        self.u, self.v = self.get_from_kwargs('resolution', [100, 10])
        self.s0, self.s1 = self.get_from_kwargs('domain', [0, 1])

        thickness = self.thickness
        closed = self.closed
        u = self.u
        v = self.v
        s0 = self.s0
        s1 = self.s1

        # central points of the curve
        if self.closed:
            # assume that the first and last vertex are the same
            verts = [f(self.s0 + (self.s1 - self.s0) / self.u * i) for i in range(0, u)]
        else:
            verts = [f(self.s0 + (self.s1 - self.s0) / self.u * i) for i in range(0, u + 1)]
        # calculate normal n and binormal b and span circle around every point
        vertices = []
        thickness *= 0.01

        for i, vert in enumerate(verts):
            s = s0 + (s1 - s0) / u * i
            t = df(s)
            a = d2f(s)
            n = t.dot(t) * a - a.dot(t) * t
            n.normalize()
            b = n.cross(t)
            b.normalize()
            vertices += [vert + n * thickness * np.cos(2 * np.pi * j / v) + b * thickness * np.sin(2 * np.pi * j / v)
                         for j in range(v)]

        #edges:
        #calculate edges along v direction
        edges = []
        for i in range(0, len(vertices), v):
            edges += [[i + j, i + (j + 1) % v] for j in range(v)]

        #calculate edges along u direction
        if closed:
            lim = u
        else:
            lim = u - 1
        for j in range(v):
            edges += [[i * v + j, (i + 1) % u * v + j] for i in range(lim)]

        #faces:
        faces = []
        for i in range(u - 1):
            for j in range(v):
                faces.append([i * v + (j + 1) % v, i * v + j, i * v + j + v, i * v + (j + 1) % v + v])
        if closed:
            for j in range(v):
                i = u - 1
                faces.append([(i * v + (j + 1) % v) % (u * v), (i * v + j) % (u * v), (i * v + j + v) % (u * v),
                              (i * v + (j + 1) % v + v) % (u * v)])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh

    def transform_mesh(self, f, df, d2f, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        # recalculate all vertices
        thickness = self.thickness
        closed = self.closed
        u = self.u
        v = self.v
        s0 = self.s0
        s1 = self.s1

        # central points of the curve
        if self.closed:
            # assume that the first and last vertex are the same
            verts = [f(self.s0 + (self.s1 - self.s0) / self.u * i) for i in range(0, u)]
        else:
            verts = [f(self.s0 + (self.s1 - self.s0) / self.u * i) for i in range(0, u + 1)]
        # calculate normal n and binormal b and span circle around every point
        vertices = []
        thickness *= 0.01

        for i, vert in enumerate(verts):
            s = s0 + (s1 - s0) / u * i
            t = df(s)
            a = d2f(s)
            n = t.dot(t) * a - a.dot(t) * t
            n.normalize()
            b = n.cross(t)
            b.normalize()
            vertices += [vert + n * thickness * np.cos(2 * np.pi * j / v) + b * thickness * np.sin(2 * np.pi * j / v)
                         for j in range(v)]

        # transform shape_keys
        if self.old_sk is None:
            self.old_sk = add_shape_key(self, 'Basis')

        self.shape += 1
        self.old_sk = add_shape_key(self, 'Shape' + str(self.shape), self.old_sk)
        for v, sk in zip(vertices, self.old_sk.data):
            sk.co = v

        morph_to_next_shape(self, self.shape - 1, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        return begin_time + transition_time


class NumericMeshCurve(BObject):
    """
    A curve that is a mesh object.
    The mesh is constructed with the Frenet-Serret-formula.
    Have a look at Frenet_Serret_Formula.nb
    However, only the spinal points are computed from the function.
    The tangent, normal and bi-normal are computed numerically.
    We need one point before the first point and one after the last point, or periodic boundary conditions (closed curves).
    """

    def __init__(self, f, **kwargs):
        """Create a curve as a tube mesh, with numeric Frenet-Serret frames.

        Like :class:`MeshCurve`, but only the spinal positions are computed
        from ``f``. Tangents, normals, and binormals are estimated from
        finite differences of those samples -- useful when analytic
        derivatives are unavailable.

        Args:
            f: Callable ``s -> Vector(x, y, z)`` -- the curve's position.
                For non-closed curves, ``f`` is also evaluated at one extra
                point on each side so finite differences work at the boundary.
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``name`` (str): Defaults to ``'DataCurve'``.
                * ``thickness`` (float): Tube radius multiplier
                  (``radius = thickness * 0.01``). Defaults to 1.
                * ``closed`` (bool): Treat the curve as closed.
                  Defaults to ``False``.
                * ``resolution`` (list[int]): ``[u, v]`` -- spinal samples
                  and circumferential segments. Defaults to ``[100, 10]``.
                * ``domain`` (list[float]): ``[s_min, s_max]``. Defaults to
                  ``[0, 1]``.
                * ``force_normal`` (bool): If ``True``, flip normals/binormals
                  so they point away from the origin / consistently upward.
                  Defaults to ``False``.
                * Standard BObject kwargs.
        """
        # defined in create_messh
        self.s1 = None
        self.v = None
        self.u = None
        self.s0 = None
        self.closed = None
        self.thickness = None
        self.old_sk = None

        self.shape = 0
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'DataCurve')
        # forces normal to point away from the origin
        self.force_normal = self.get_from_kwargs('force_normal', False)

        mesh = self.create_mesh(f)
        super().__init__(mesh=mesh, name=self.name, **kwargs)

    def create_mesh(self, f):
        self.thickness = self.get_from_kwargs('thickness', 1)
        self.closed = self.get_from_kwargs('closed', False)
        self.u, self.v = self.get_from_kwargs('resolution', [100, 10])
        self.s0, self.s1 = self.get_from_kwargs('domain', [0, 1])

        thickness = self.thickness
        closed = self.closed
        u = self.u
        s0 = self.s0
        s1 = self.s1

        # central (spinal) points of the curve (the backbone)
        # the backbones are from 0 to u-1 for closed curves (0=u)
        # additional points for non-closed curves -1, u+1
        if self.closed:
            # assume that the first and last vertex are the same
            verts = [f(self.s0 + (self.s1 - self.s0) / self.u * i) for i in range(0, u)]
        else:
            verts = [f(self.s0 + (self.s1 - self.s0) / self.u * i) for i in range(-1, u + 2)]

        # calculate normal n and binormal b and span circle around every point
        vertices = []
        thickness *= 0.01
        tangents = []
        normals = []
        data = []
        if self.closed:
            for i in range(0, self.u):
                tangents.append((verts[(i + 1) % self.u] - verts[i]).normalized())
                normals.append((verts[(i + 1) % self.u] + verts[i - 1] - 2 * verts[i]).normalized())
                if self.force_normal:
                    if normals[-1].dot(verts[(i + 1) % self.u]) < 0:
                        normals[-1] = -normals[-1]
            for i in range(0, self.u):
                binormal = -(tangents[i].cross(normals[i])).normalized()
                if self.force_normal:
                    if binormal.dot(Vector((0, 0, 1))) < 0:
                        binormal *= -1
                data.append([verts[i], normals[i], binormal])
        else:
            for i in range(0, self.u + 2):
                v = Vector()

                diff = (verts[(i + 1)] - verts[i]).normalized()
                tangents.append(diff)
            for i in range(1, self.u + 2):
                diff = (verts[i + 1] + verts[i - 1] - 2 * verts[i]).normalized()
                normals.append(diff)
                if self.force_normal:
                    if normals[-1].dot(verts[i + 1]) < 0:
                        normals[-1] = -normals[-1]
            for i in range(0, self.u + 1):
                binormal = -tangents[i + 1].cross(normals[i]).normalized()
                if self.force_normal:
                    if binormal.dot(Vector((0, 0, 1))) < 0:
                        binormal *= -1
                data.append([verts[i + 1], normals[i], binormal])

        for v, n, b in data:
            vertices += [
                v + n * thickness * np.cos(2 * np.pi * j / self.v) + b * thickness * np.sin(2 * np.pi * j / self.v) for
                j in range(self.v)]

        v = self.v
        #edges:
        #calculate edges along v direction
        edges = []
        for i in range(0, len(vertices), v):
            edges += [[i + j, i + (j + 1) % v] for j in range(v)]

        #calculate edges along u direction
        if closed:
            lim = u
        else:
            lim = u - 1
        for j in range(v):
            edges += [[i * v + j, (i + 1) % u * v + j] for i in range(lim)]

        #faces:
        faces = []
        for i in range(u - 1):
            for j in range(v):
                faces.append([i * v + (j + 1) % v, i * v + j, i * v + j + v, i * v + (j + 1) % v + v])
        if closed:
            for j in range(v):
                i = u - 1
                faces.append([(i * v + (j + 1) % v) % (u * v), (i * v + j) % (u * v), (i * v + j + v) % (u * v),
                              (i * v + (j + 1) % v + v) % (u * v)])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh


class SimpleLoopMeshCurve(BObject):
    """
    A curve that is a mesh object.
    We just grow the backbone of a curve. Vertices connected with lines
    """

    def __init__(self, f, **kwargs):
        """Create a 1D backbone curve (vertices connected by edges, no tube).

        Args:
            f: Callable ``s -> Vector(x, y, z)`` -- the curve's spinal position.
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``name`` (str): Defaults to ``'BackBoneCurve'``.
                * ``closed`` (bool): If ``True``, the last vertex links back
                  to the first. Defaults to ``False``.
                * ``resolution`` (int): Number of vertices along the curve.
                  Defaults to 100.
                * ``domain`` (list[float]): ``[s_min, s_max]``. Defaults to
                  ``[0, 1]``.
                * Standard BObject kwargs.
        """
        # defined in create_messh
        self.s1 = None
        self.u = None
        self.s0 = None
        self.closed = None
        self.thickness = None
        self.old_sk = None

        self.shape = 0
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'BackBoneCurve')

        mesh = self.create_mesh(f)
        super().__init__(mesh=mesh, name=self.name, **kwargs)

    def create_mesh(self, f):
        self.closed = self.get_from_kwargs('closed', False)
        self.u = self.get_from_kwargs('resolution', 100)
        self.s0, self.s1 = self.get_from_kwargs('domain', [0, 1])

        closed = self.closed
        u = self.u
        s0 = self.s0
        s1 = self.s1

        # central (spinal) points of the curve (the backbone)
        # the backbones are from 0 to u-1 for closed curves (0=u)
        # additional points for non-closed curves -1, u+1
        if self.closed:
            # assume that the first and last vertex are the same
            verts = [f(s0 + (s1 - s0) / u * i) for i in range(0, u)]
        else:
            verts = [f(s0 + (s1 - s0) / u * i) for i in range(0, u + 1)]

        #edges:
        #calculate edges along v direction
        edges = [[i, i + 1] for i in range(0, len(verts) - 1)]
        if closed:
            edges.append([len(verts) - 1, 0])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(verts, edges, [])
        new_mesh.update()
        return new_mesh


class DataCurve(BObject):
    '''
    A curve based on a mesh constructed from data points.
    Morphing is possible, but the curve cannot be grown from one side to the other
    It can only appear
    '''

    def __init__(self, data, closed=True, **kwargs):
        """Create a strip-mesh curve from a list of points.

        Each datapoint becomes a pair of vertices (one at the base level,
        one offset by ``+Z * 0.01*thickness``), connected into a thin
        flat ribbon. Suitable for 2D plots that should appear flat but solid.

        Args:
            data: Iterable of points -- each entry may be a complex
                number, a :class:`Vector`, or a list (interpreted by
                :func:`z2vec`).
            closed: If ``True``, the strip closes back to the first point.
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``name`` (str): Defaults to ``'DataCurve'``.
                * ``thickness`` (float): Vertical separation of the two
                  ribbon edges (visual thickness). Defaults to 1.
                * Standard BObject kwargs (``color``, ``location``, ...).
        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'DataCurve')
        thickness = self.get_from_kwargs('thickness', 1)
        mesh = self.create_mesh(data, closed, thickness=thickness)
        super().__init__(mesh=mesh,
                         # solid=0.02, bevel=0.001,
                         **kwargs)

    def create_mesh(self, data, closed, thickness=1):

        vertices = [z2vec(d) for d in data]
        vertices2 = [z2vec(d) + Vector([0, 0, 0.01 * thickness]) for d in data]
        n = len(vertices)
        edges = [[i, i + 1] for i in range(n - 1)]
        edges2 = [[n + i, n + i + 1] for i in range(n - 1)]
        if closed:
            edges.append([n - 1, 0])
            edges.append([2 * n - 1, n])
        edges3 = [[i, n + i] for i in range(n)]  # vertical edges

        vertices += vertices2
        edges += edges2
        edges += edges3

        faces = [[i + 1, n + i + 1, n + i, i] for i in range(n - 1)]
        if closed:
            faces.append([0, n, 2 * n - 1, n - 1])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh


class DataCurveRefined(BObject):
    '''
    This one gets a set of data sets
    '''

    def __init__(self, data_set, closed=True, **kwargs):
        """Create a multi-resolution :class:`DataCurve` that can morph between
        coarse and fine sample sets.

        The mesh is built from the highest-resolution data set; lower-resolution
        copies are added as shape keys so the curve can animate between
        smooth and coarse representations.

        Args:
            data_set: List of data lists -- each inner list is a sequence
                of complex numbers / vectors describing the curve at a
                given resolution.
            closed: If ``True``, each data list is treated as a closed loop.
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``name`` (str): Defaults to ``'DataCurve'``.
                * ``thickness`` (float): Visual thickness of the ribbon.
                  Defaults to 1.
                * ``solid`` (float): Solidify modifier thickness.
                  Defaults to 0.02.
                * Standard BObject kwargs.
        """
        self.current_shape = None
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'DataCurve')
        thickness = self.get_from_kwargs('thickness', 1)
        solid = self.get_from_kwargs('solid', 0.02)

        # make the data with the most data points first to create the mesh  from
        data_set2 = sorted(data_set, key=len, reverse=True)

        # create a dictionary that keeps track of the old order
        old_order = {}
        for i in range(len(data_set)):
            data = data_set[i]
            for j, data2 in enumerate(data_set2):
                if data2 == data:
                    old_order[i] = j
                    break

        mesh = self.create_mesh(data_set2[0], closed, thickness=thickness)
        super().__init__(mesh=mesh, name=self.name, solid=solid, bevel=0.001, **kwargs)
        # create the shape keys for the remaining data_sets

        # create shape key for vertical displacement
        ref = self.ref_obj
        old_sk = ibpy.add_shape_key(ref, 'Basis')

        for s in range(1, len(data_set2)):
            set = data_set2[s]
            # find indices of matching data points
            indices = []
            last = 0
            for data in set:
                for i in range(last, len(old_sk.data)):
                    v = old_sk.data[i].co
                    if math.isclose((z2vec(data) - v).length, 0, abs_tol=0.0000001):
                        indices.append(i)
                        last = i
                        break

            for data in set:
                for i in range(last, len(old_sk.data)):
                    v = old_sk.data[i].co
                    if math.isclose((z2vec(data) + Vector([0, 0, 0.01 * thickness]) - v).length, 0, abs_tol=0.0000001):
                        indices.append(i)
                        last = i
                        break

            old_sk = ibpy.add_shape_key(ref, name='Profile' + str(s), previous=old_sk)
            # project vertices that do not belong to the low-res curve onto the low-res curve
            for interval in range(len(indices) - 1):
                start = old_sk.data[indices[interval + 1]].co
                end = old_sk.data[indices[interval]].co
                dir = end - start
                dir /= dir.length

                low = indices[interval]
                high = indices[interval + 1]
                for i in range(low, high):
                    pos = old_sk.data[i].co
                    diff = pos - start
                    final = start + dir * diff.dot(dir)  # projection with the dot product
                    old_sk.data[i].co = final

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        '''
        make sure that the last shape key appears first
        :param begin_time:
        :param transition_time:
        :return:
        '''
        super().appear(begin_time=begin_time, transition_time=transition_time)
        self.current_shape = ibpy.set_to_last_shape(self, begin_time * FRAME_RATE)

    def next_shape(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.current_shape -= 1
        ibpy.morph_to_previous_shape(self, self.current_shape, begin_time * FRAME_RATE, transition_time * FRAME_RATE)

    def create_mesh(self, data_set, closed, thickness=1):
        if closed:
            data_set.append(data_set[0])

        vertices = [z2vec(data) for data in data_set]
        vertices2 = [z2vec(data) + Vector([0, 0, 0.01 * thickness]) for data in data_set]
        n = len(vertices)
        edges = [[i, i + 1] for i in range(n - 1)]
        edges2 = [[n + i, n + i + 1] for i in range(n - 1)]
        edges3 = [[i, n + i] for i in range(n)]

        vertices += vertices2
        edges += edges2
        edges += edges3

        faces = [[i, n + i, n + i + 1, i + 1] for i in range(n - 1)]

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()

        return new_mesh


class Curve(GeoBObject):
    """
    Create a parametric curve, based on a bezier curve
    which is not restricted to a coordinate system

    example:
    Function([lambda phi: [phi, 0, abs_func(phi)],
              lambda phi: [phi, 0, func(phi)]],
             domain=[-2, 2],
             num_points=100,
             color='white',
             name='frame')

    remarks:
    There is an complicated algorithm to calculate the control points of the Bezier curve.
    It only works, when the data is finer than the resolution of the curve.

    known issues:
    * it is important that each function has a unique name string,
      otherwise it will not be found and animated correctly

    """

    def __init__(self, mappings,
                 domain=None,
                 num_points=100,
                 color='example',
                 **kwargs):
        """Create a parametric bezier curve in world space (no coordinate system).

        Equivalent to :class:`Function` in ``mode='PARAMETRIC'``, but without
        the coordinate-system dependency. Each mapping is a callable
        ``t -> Vector(x, y, z)``; multiple mappings are stored as shape keys
        so the curve can morph between them.

        Args:
            mappings: A single callable ``t -> Vector`` (or list type for
                Vector conversion) or a list of such callables.
            domain: ``[t_min, t_max]`` parameter range. Defaults to ``[-1, 1]``.
            num_points: Number of bezier control points sampled across
                ``domain``. Higher = smoother.
            color: Color name forwarded to the underlying material
                (e.g. ``'example'``, ``'drawing'``, ``'important'``).
            **kwargs: Forwarded to :class:`GeoBObject`. Notable keys:
                * ``name`` (str): Defaults to ``'Curve'``.
                * ``thickness`` (float): Curve bevel depth scaled by 0.1.
                  Defaults to 1.
                * ``extrude`` (float): Curve extrude scaled by 0.01.
                  Defaults to 1.
                * ``bevel_resolution`` (int): Bevel cross-section resolution.
                  Defaults to 4.
                * ``bevel_caps`` (bool): Close the bevel caps. Defaults to ``True``.
                * ``bevel_factor_mapping_end`` (str): Bevel factor mapping.
                  Defaults to ``'RESOLUTION'``.
                * Standard BObject kwargs.
        """
        if not isinstance(mappings, list):
            mappings = [mappings]

        self.current_mapping = None  # keeps track of the shown mapping, when more than one mapping is defined for the
        # bobject
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'Curve')
        self.num_points = num_points
        # constants
        self.eps = 0.1

        self.mappings = mappings
        if domain is None:
            self.domain = [-1, 1]
        else:
            self.domain = domain

        self.curve = ibpy.get_new_curve(self.name, num_points)

        list_of_points = []
        self.x_range = self.domain[1] - self.domain[0]
        self.x_min = self.domain[0]

        for mapping in self.mappings:
            list_of_points.append(self.create_list_of_points_for_mapping(mapping))

        self.ref_obj = ibpy.new_curve_object(self.name, self.curve)
        self.dialer = []  # dial between different hue colors

        self.map_count = 0
        self.map_sk_dict = {}
        for points in list_of_points:
            self.create_shape_key_for_points(points)

        super().__init__(obj=self.ref_obj, color=color, name=self.name, **kwargs)

        thickness = self.get_from_kwargs("thickness", 1)
        self.bevel_depth = thickness * 0.1
        self.bevel_res = self.get_from_kwargs('bevel_resolution', 4)
        self.bevel_caps = self.get_from_kwargs('bevel_caps', True)
        # the following synchronizes the progress of the curve with the curve parameter
        self.bevel_factor_mapping_end = self.get_from_kwargs('bevel_factor_mapping_end', "RESOLUTION")

        extrude = self.get_from_kwargs("extrude", 1)
        self.extrude = extrude * 0.01
        # it can be linked without hesitation. The curve only appears once a bevel depth is set,
        # which only happens in the show bobject
        ibpy.link(self.ref_obj)

    @classmethod
    def from_data(cls, data, color='example', res=1, resolution=1, **kwargs):
        if not isinstance(data[0], list):
            data = [data]

        mappings = [lambda x, d=d: partial(mapping_from_data, d)(x) for d in data]

        num_points = 10
        for d in data:
            num_points = np.maximum(num_points, len(d))

        num_points *= res  # kept for backward compatibility
        num_points *= resolution
        return Curve(mappings, domain=[0, 1], num_points=num_points, color=color, **kwargs)

    def create_shape_key_for_points(self, points):
        if self.map_count == 0:
            for i in range(1, self.num_points + 1):
                ibpy.set_bezier_point_of_curve(self.curve, i - 1, points[3 * i], points[3 * i - 1],
                                               points[3 * i + 1])
            self.map_sk_dict[self.map_count] = ibpy.add_shape_key(self.ref_obj, 'Basis')
        else:
            self.map_sk_dict[self.map_count] = ibpy.add_shape_key(self.ref_obj,
                                                                  self.name + str(self.map_count),
                                                                  self.map_sk_dict[self.map_count - 1]
                                                                  )
            sk = self.map_sk_dict[self.map_count]
            for i in range(1, self.num_points + 1):
                ibpy.reset_bezier_point(sk.data[i - 1], points[3 * i], points[3 * i - 1],
                                        points[3 * i + 1])
        self.map_count += 1
        return (self.map_count - 1)

    def create_list_of_points_for_mapping(self, mapping):
        points = []
        for i in range(-1, self.num_points + 1):
            for j in range(0, 3):
                x = self.x_min + (3 * i + j) * self.x_range / self.num_points / 3
                # evaluate the bobject at num_points+2 points and two sub-steps for each interval,
                # starting one main interval before t_min and finishing one main interval after t_max
                points.append(to_vector(mapping(x)))
                # add one last point to finish the additional interval at the end for the calculation of the handles
        x = self.x_min + (self.num_points + 1) * self.x_range / self.num_points
        points.append(to_vector(mapping(x)))

        # print("number of points: "+str(len(points)))
        # convert data points to bezier control points
        for p in range(0, len(points) - 1, 3):
            # print(p)
            xs = [points[p + j].x for j in range(0, 4)]
            ys = [points[p + j].y for j in range(0, 4)]
            zs = [points[p + j].z for j in range(0, 4)]

            handle_one_x, handle_two_x = derive_bezier_handles(*xs, 1 / 3, 2 / 3)
            handle_one_y, handle_two_y = derive_bezier_handles(*ys, 1 / 3, 2 / 3)
            handle_one_z, handle_two_z = derive_bezier_handles(*zs, 1 / 3, 2 / 3)

            points[p + 1] = Vector([handle_one_x, handle_one_y, handle_one_z])
            points[p + 2] = Vector([handle_two_x, handle_two_y, handle_two_z])

        return points

    def grow(self,
             begin_time=0,
             transition_time=OBJECT_APPEARANCE_TIME, inverted=False, start_factor=0, end_factor=1,
             **kwargs):
        if start_factor == 0:
            # this is done only once
            # if the growing is continued, it would cause flickering
            super().appear(begin_time=begin_time, transition_time=0)
            ibpy.set_use_path(self, True)

        # if bevel doesn't work you have to give the curve a unique name
        ibpy.set_bevel(self, self.bevel_depth, caps=self.bevel_caps, res=self.bevel_res)
        ibpy.set_extrude(self, self.extrude)
        ibpy.set_bevel_factor_mapping(self, self.bevel_factor_mapping_end)

        appear_frame = begin_time * FRAME_RATE
        ibpy.grow_curve(self, appear_frame, transition_time, inverted, start_factor=start_factor, end_factor=end_factor)

        self.current_mapping = 0
        return begin_time + transition_time

    def shrink(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME, inverted=False,
               **kwargs):

        appear_frame = begin_time * FRAME_RATE
        ibpy.shrink_curve(self, appear_frame, transition_time, inverted)

        self.current_mapping = 0
        return begin_time + transition_time

    def next(self,
             begin_time=0,
             transition_time=OBJECT_APPEARANCE_TIME):
        """
            morph from one mapping to the next

            :param begin_time:
            :param transition_time:
            :return:
        """

        begin_frame = begin_time * FRAME_RATE

        if len(self.dialer) > self.current_mapping:
            dialer_index = self.current_mapping
            current_dialer = self.dialer[dialer_index]
            ibpy.insert_keyframe(current_dialer, 'default_value', begin_frame)
            current_dialer.default_value = 1
            ibpy.insert_keyframe(current_dialer, 'default_value', begin_frame + transition_time * FRAME_RATE)

        ibpy.morph_to_next_shape(self.ref_obj, self.current_mapping, begin_frame, transition_time * FRAME_RATE)
        self.current_mapping += 1
        return begin_time + transition_time

    def previous(self,
                 begin_time=None,
                 transition_time=OBJECT_APPEARANCE_TIME):
        """
            morph from one mapping to the next

            :param begin_time:
            :param transition_time:
            :return:
        """

        begin_frame = begin_time * FRAME_RATE
        dialer_index = self.current_mapping
        if len(self.dialer) > dialer_index:
            current_dialer = self.dialer[dialer_index]
            current_dialer.default_value = 1
            ibpy.insert_keyframe(current_dialer, 'default_value', begin_frame)
            current_dialer.default_value = 0
            ibpy.insert_keyframe(current_dialer, 'default_value', begin_frame + transition_time * FRAME_RATE)
        ibpy.morph_to_previous_shape(self.ref_obj, self.current_mapping, begin_frame, transition_time * FRAME_RATE)
        self.current_mapping += 0

    def save_eval(self, mapping, x):
        try:
            y = mapping(x)
        except ZeroDivisionError:
            y = mapping(x + self.eps)
        return y

    def appear(self,
               begin_time=0,
               transition_time=0,
               **kwargs):
        ibpy.set_bevel(self, self.bevel_depth)
        ibpy.fade_in(self, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        self.current_mapping = 0

    def transform(self, transformation, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        mapping = lambda x: transformation(self.mappings[self.current_mapping](x))
        points = self.create_list_of_points_for_mapping(mapping)
        sk_of_interest = self.create_shape_key_for_points(points)

        sk_transitions = sk_of_interest - self.current_mapping
        dt = transition_time / sk_transitions

        for i in range(sk_transitions):
            self.next(begin_time=begin_time, transition_time=dt)
            begin_time += dt

    def change_thickness(self, new_thickness=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.change_bevel(self, self.bevel_depth, new_thickness * 0.1, begin_time * FRAME_RATE,
                          transition_time * FRAME_RATE)
        self.bevel_depth = new_thickness * 0.1
        return begin_time + transition_time


class BezierDataCurve(BObject):
    """
    Creates a Bezier curve from data points
    The control points are calculated by blender default method
    """

    def __init__(self, data, **kwargs):
        """Create a Blender bezier curve directly from data points.

        Control-point handles are auto-computed by Blender's ``AUTOMATIC``
        handle type, giving a smooth interpolation through the data.

        Args:
            data: List of points (each entry consumable by
                ``ibpy.get_new_curve(..., data=...)``).
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``name`` (str): Defaults to ``'BezierDataCurve'``.
                * ``thickness`` (float): Bevel depth (``thickness * 0.05``).
                  Defaults to 1.
                * ``extrude`` (float): Extrude depth (``extrude * 0.01``).
                  Defaults to 1.
                * Standard BObject kwargs.
        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'BezierDataCurve')
        curve = ibpy.get_new_curve(name=self.name, num_points=len(data), data=data, **kwargs)
        self.ref_obj = ibpy.new_curve_object(self.name, curve)
        thickness = self.get_from_kwargs('thickness', 1)
        extrude = self.get_from_kwargs("extrude", 1)
        self.extrude = extrude * 0.01
        self.bevel_caps = True
        self.bevel_depth = thickness * 0.05
        self.bevel_res = 4

        super().__init__(obj=self.ref_obj, **kwargs)

        # Link object to the scene
        ibpy.link(self.ref_obj)
        # Toggle handle type (faster than doing it point by point)
        ibpy.set_active(self)
        bpy.ops.object.editmode_toggle()
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.handle_type_set(type='AUTOMATIC')
        bpy.ops.object.editmode_toggle()

    def grow(self,
             begin_time=0,
             transition_time=OBJECT_APPEARANCE_TIME, inverted=False, start_factor=0, end_factor=1,
             **kwargs):
        if start_factor == 0:
            # this is done only once
            # if the growing is continued, it would cause flickering
            super().appear(begin_time=begin_time, transition_time=0)
            ibpy.set_use_path(self, True)
        # TODO: Create parameter for thickness
        ibpy.set_bevel(self, self.bevel_depth, caps=self.bevel_caps, res=self.bevel_res)
        ibpy.set_extrude(self, self.extrude)

        appear_frame = begin_time * FRAME_RATE
        ibpy.grow_curve(self, appear_frame, transition_time, inverted, start_factor=start_factor, end_factor=end_factor)

        self.current_mapping = 0
        return begin_time + transition_time


class CurveModifier(GeometryNodesModifier):
    """
    Geometry-node setup that turns a poly line into a tube mesh.

    Rebuilt from ``video_hat_tile/tmp.xml``. The nodes are placed explicitly at
    the (integer-rounded) grid locations stored in the xml.

    The two ``GROUP`` nodes from the xml -- which displace and select the
    resampled points -- are intentionally left out. They are meant to be
    produced with :func:`make_function` and wired into the Set Position node's
    ``Selection`` and ``Position`` inputs afterwards; those inputs are therefore
    left unconnected here.
    """

    def __init__(self, name='CurveModifier', **kwargs):
        super().__init__(name=name, automatic_layout=False, group_output=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        out.location = (6 * 200, 0)
        links = tree.links

        function = get_from_kwargs(kwargs, "function", ["t", "t", "t"])
        pnts = get_from_kwargs(kwargs, "points", 100)
        material_string = get_from_kwargs(kwargs,"color", "drawing")
        resolution = get_from_kwargs(kwargs,"resolution",4)
        thickness = get_from_kwargs(kwargs,"thickness",0.01)
        if isinstance(material_string, str):
            material = get_texture(material_string,**kwargs)
        else:
            material = material_string

        # --- input value nodes (left column) ---   
        points = InputInteger(tree, location=(-2, 0), integer=pnts, name='Integer', label='points')
        thickness = InputValue(tree, location=(-2, -0.5), value=thickness, name='Thickness', label='thickness')
        minimum = InputValue(tree, location=(-2, -1), value=0, name='Value.003', label='min')
        maximum = InputValue(tree, location=(-2, -1.5), value=math.tau, name='Value.004', label='max')
        index = Index(tree, location=(-2, 0.5), value=0, name='Index')
        scene_time = SceneTime(tree, location=(-2, -3.5))
        t0 = InputValue(tree, location=(-2, -2), value=0, name='Value', label='t0')
        duration = InputValue(tree, location=(-2, -2.5), value=0, name='Value.001', label='duration')

        # --- curve backbone: a line resampled into 'points' segments ---
        curve_line = CurveLine(tree, location=(-1, 2), hide=True)
        resample = ResampleCurve(tree, location=(0, 1),
                                 count=points.std_out, mode='Count', hide=True)

        # --- displace the resampled points; Selection and Position are fed by

        position = make_function(tree, name="tFunction",
                                 aux_functions={
                                     "t": "mini,i,n,1,-,/,maxi,mini,-,*,+"
                                 },
                                 functions={
                                     "pos": function
                                 }, inputs=["i", "n", "mini", "maxi"], outputs=["pos"],
                                 scalars=["t", "i", "n", "mini", "maxi"], vectors=["pos"], hide=True,
                                 location=(0, -0.5))
        links.new(index.std_out, position.inputs["i"])
        links.new(points.std_out, position.inputs["n"])
        links.new(minimum.std_out, position.inputs["mini"])
        links.new(maximum.std_out, position.inputs["maxi"])

        selector = make_function(tree, name="GrowFunction",
                                 functions={
                                    "select": "i,t,t0,-,duration,/,n,*,>"
                                 }, inputs=["i","n","t","t0","duration"], outputs=["select"],
                                 scalars=["i","n","t","t0","duration","select"], hide=True,
                                 location=(0,-1.5))
        links.new(index.std_out, selector.inputs["i"])
        links.new(points.std_out, selector.inputs["n"])
        links.new(scene_time.std_out,selector.inputs["t"])
        links.new(t0.std_out,selector.inputs["t0"])
        links.new(duration.std_out,selector.inputs["duration"])

        del_geo = DeleteGeometry(tree, location=(1, 1), domain="POINT", selection=selector.outputs["select"], hide=True, name="Grow")

        set_position = SetPosition(tree, location=(2, 1), position=position.outputs["pos"], hide=True)

        # --- profile circle swept along the curve to build the tube ---
        circle = CurveCircle(tree, location=(2, 0), radius=thickness.std_out, resolution=resolution, mode='RADIUS')
        curve_to_mesh = CurveToMesh(tree, location=(3, 1), fill_caps=False,
                                    profile_curve=circle.geometry_out, hide=True)

        # --- material ---
        self.materials.append(material)
        material_node = InputMaterial(tree, location=(3, 0),material=material)
        set_material = SetMaterial(tree, location=(5, 1), material=material_node.std_out, hide=True)

        # --- output ---
        create_geometry_line(tree, [curve_line, resample,del_geo, set_position, curve_to_mesh, set_material], out=out.inputs[0])


class GeoCurve(BObject):
    def __init__(self, **kwargs):
        points = get_from_kwargs(kwargs, "points", 100)
        name = get_from_kwargs(kwargs, "name", "CurveFromGeometryNodes")
        carrier = Plane(name=name)
        carrier.appear(begin_time=0, transition_time=0)

        self.mod = CurveModifier(points=points,**kwargs)
        carrier.add_mesh_modifier(type='NODES', node_modifier=self.mod)
        super().__init__(obj=carrier, **kwargs)

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center', pivot=None,
             initial_scale=0, alpha=1):

        t0_node = ibpy.get_geometry_node_from_modifier(self.mod,label="t0")
        duration_node = ibpy.get_geometry_node_from_modifier(self.mod,label="duration")

        ibpy.change_value(t0_node,from_value=0,to_value=begin_time,begin_time=0,transition_time=0)
        ibpy.change_value(duration_node,from_value=0,to_value=transition_time,begin_time=0,transition_time=0)

        return begin_time+transition_time

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, slot=None, children=True,
                  **kwargs):
        for mat in self.mod.materials:
            ibpy.change_alpha_of_material(mat,from_value=1,to_value=alpha,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

########################
## Static functions ####
########################

def derive_bezier_handles(a, b, c, d, tb, tc):
    """
    TODO: for speed up, this can be optimized, when tb and tc are fixed and the calculations simplify

    Derives bezier handles by using the start and end of the curve with 2 intermediate
    points to use for interpolation.
    :param a:
        The start point.
    :param b:
        The first mid-point, located at `tb` on the bezier segment, where 0 < `tb` < 1.
    :param c:
        The second mid-point, located at `tc` on the bezier segment, where 0 < `tc` < 1.
    :param d:
        The end point.
    :param tb:
        The position of the first point in the bezier segment.
    :param tc:
        The position of the second point in the bezier segment.
    :return:
        A tuple of the two intermediate handles, that is, the right handle of the start point
        and the left handle of the end point.
    """

    # Calculate matrix coefficients
    matrix_a = 3 * math.pow(1 - tb, 2) * tb
    matrix_b = 3 * (1 - tb) * math.pow(tb, 2)
    matrix_c = 3 * math.pow(1 - tc, 2) * tc
    matrix_d = 3 * (1 - tc) * math.pow(tc, 2)

    # Calculate the matrix determinant
    matrix_determinant = 1 / ((matrix_a * matrix_d) - (matrix_b * matrix_c))

    # Calculate the components of the target position vector
    final_b = b - (math.pow(1 - tb, 3) * a) - (math.pow(tb, 3) * d)
    final_c = c - (math.pow(1 - tc, 3) * a) - (math.pow(tc, 3) * d)

    # Multiply the inversed matrix with the position vector to get the handle points
    bezier_b = matrix_determinant * ((matrix_d * final_b) + (-matrix_b * final_c))
    bezier_c = matrix_determinant * ((-matrix_c * final_b) + (matrix_a * final_c))

    # Return the handle points
    return bezier_b, bezier_c
