import bmesh
import numpy as np
import bpy
from mathutils import Vector, Quaternion

from interface import ibpy
from interface.ibpy import add_shape_key, keyframe_shape_key, set_shape_key_to_value
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import quaternion_from_normal, to_vector


class Disc(BObject):
    """
    Create a disc with descent mesh:
    """
    def __init__(self, radius=1, resolution=10,location =Vector(), **kwargs):
        """Create a flat disc lying in the XY plane.

        Args:
            radius: Outer radius of the disc.
            resolution: Either an int (used for both radial and angular
                segments) or ``[res_r, res_phi]``. Higher values produce a
                smoother circular boundary.
            location: Centre of the disc in world coordinates. The location
                is baked into the mesh vertices (the object's origin remains
                at the world origin).
            **kwargs: Forwarded to :class:`BObject`. Notable keys:
                ``name`` (defaults to ``'Disc'``), ``color``, ``scale``,
                ``smooth``, ``rotation_euler``, etc.
        """
        self.kwargs = kwargs
        self.radius = radius
        if isinstance(resolution,list) and len(list)>1:
            self.res_r = resolution[0]
            self.resolution=resolution[1]
        else:
            self.res_r =resolution
            self.resolution = resolution
        self.location = to_vector(location)
        self.name = self.get_from_kwargs('name', 'Disc')
        mesh = self.create_mesh()
        super().__init__(name=self.name, mesh=mesh, **kwargs)

    def create_verts_edges_faces(self):
        vertices = []
        edges = []
        faces = []
        res = self.resolution
        first_row = True
        dr = self.radius / self.res_r
        for i in range(0, self.res_r + 1):
            r = i * dr
            # create vertices
            if r == 0:
                vertices.append((r, 0, 0))
            else:
                dphi = np.pi * 2 / res
                for j in range(0, res):
                    phi = j * dphi
                    vertices.append((r * np.cos(phi), r * np.sin(phi), 0))
                    v = len(vertices)
                    if j > 0:
                        edges.append([v - 1, v - 2])
                        if i > 1:
                            edges.append([v - 1, v - res - 1])
                            if j > 0:
                                faces.append([v - 2, v - 1, v - 1 - res, v - 2 - res])
                        else:
                            edges.append([v - 1, 0])
                            if j > 0:
                                faces.append([v - 2, v - 1, 0])
                edges.append([v - 1, v - res])
                if i > 1:
                    faces.append([v - 1, v - res, v - 2 * res, v - 1 - res])
                else:
                    faces.append([v - 1, v - res, 0])

        vertices2 = []
        for v in vertices:
            pos = Vector(list(v)) + self.location
            vertices2.append((pos.x, pos.y, pos.z))

        return vertices2, edges, faces

    def create_mesh(self):
        vertices, edges, faces = self.create_verts_edges_faces()
        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh


class TransformedDisc(Disc):
    """A :class:`Disc` whose vertices are deformed by an arbitrary mapping."""

    def __init__(self,transformation=lambda x:x,**kwargs):
        """Create a disc whose vertices are pushed through ``transformation``.

        Args:
            transformation: Callable ``Vector -> Vector`` applied to every
                vertex of the underlying disc mesh. The default identity
                produces a regular flat disc.
            **kwargs: Forwarded to :class:`Disc` (``radius``, ``resolution``,
                ``location``, ``name``, etc.).
        """
        self.transformation = transformation
        super().__init__(**kwargs)

    def create_mesh(self):
        vertices, edges, faces = self.create_verts_edges_faces()
        vertices2 = []
        for vertex in vertices:
            pos = to_vector(vertex)
            pos2 = self.transformation(pos)
            vertices2.append((pos2.x,pos2.y,pos2.z))

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices2, edges, faces)
        new_mesh.update()
        return new_mesh


class Annulus(BObject):
    """
    Create a annulus with descent mesh:
    """

    def __init__(self, r_in=0, r_out=1, resolution=10, **kwargs):
        """Create a flat annulus (ring) lying in the XY plane.

        Args:
            r_in: Inner radius. Use ``0`` for a full disc.
            r_out: Outer radius.
            resolution: Number of subdivisions used for both the radial and
                angular directions.
            **kwargs: Forwarded to :class:`BObject` (``name`` defaults to
                ``'Annulus'``; also ``color``, ``location``, ``scale``, etc.).
        """
        self.kwargs = kwargs

        self.name = self.get_from_kwargs('name', 'Annulus')
        mesh = self.create_mesh(r_in, r_out, resolution)
        super().__init__(name=self.name, mesh=mesh, **kwargs)

    def create_mesh(self, r_in=0, r_out=1, resolution=10):
        vertices = []
        edges = []
        faces = []

        dr = (r_out - r_in) / resolution
        for i in range(0, resolution + 1):
            r = r_in + i * dr
            dphi = np.pi * 2 / resolution
            for j in range(0, resolution):
                phi = j * dphi
                vertices.append((r * np.cos(phi), r * np.sin(phi), 0))
                v = len(vertices)
                if j > 0:
                    edges.append([v - 1, v - 2])
                if i > 0:
                    edges.append([v - 1, v - resolution - 1])
                    if j > 0:
                        faces.append([v - 2, v - 1, v - 1 - resolution, v - 2 - resolution])

            edges.append([v - 1, v - resolution])
            if i > 0:
                faces.append([v - 1, v - resolution, v - 2 * resolution, v - 1 - resolution])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh


class DynamicAnnulus(BObject):
    """
    Create a annulus with descent mesh that can interpolate between a disc and a circle:
    """

    def __init__(self, radius=1, resolution=10, **kwargs):
        """Create an annulus that can morph between a full disc and a thin ring.

        A chain of shape keys is generated, one per discrete inner radius from
        ``0`` to ``radius``. Use :meth:`set_inner_radius` to animate towards
        any of those values.

        Args:
            radius: Outer radius of the annulus.
            resolution: Number of subdivisions in both directions. Also
                determines the number of shape-key snapshots (one per step).
            **kwargs: Forwarded to :class:`BObject` (``name`` defaults to
                ``'Annulus'``; standard color/transform kwargs).
        """
        self.kwargs = kwargs
        self.shape_keys = []
        self.name = self.get_from_kwargs('name', 'Annulus')
        mesh = self.create_mesh(radius, resolution)

        super().__init__(name=self.name, mesh=mesh, **kwargs)
        # create shape keys for various states of the interpolation
        old_sk = add_shape_key(self, 'Basis')
        self.shape_keys.append('Basis')
        r_in_old = 0.001
        for i in range(1, resolution + 1):
            dr = radius / resolution
            r_in = np.round(dr * i * 100) / 100

            add_shape_key(self, str(r_in_old), previous=old_sk)
            self.shape_keys.append(str(r_in_old))
            for v in range(len(old_sk.data)):
                x, y, z = old_sk.data[v].co[:]
                r = np.sqrt(x * x + y * y)
                # map r from [0,radius] to [r_in,radius]
                r_new = (r - r_in_old) / (radius - r_in_old) * (radius - r_in) + r_in
                old_sk.data[v].co[0] *= r_new / r
                old_sk.data[v].co[1] *= r_new / r

            r_in_old = r_in

    def create_mesh(self, radius=1, resolution=10):
        vertices = []
        edges = []
        faces = []
        epsilon = 0.001  # avoid degeneracy at the origin of the disc
        dr = radius / resolution
        for i in range(0, resolution + 1):
            r = epsilon + i * dr
            dphi = np.pi * 2 / resolution
            for j in range(0, resolution):
                phi = j * dphi
                vertices.append((r * np.cos(phi), r * np.sin(phi), 0))
                v = len(vertices)
                if j > 0:
                    edges.append([v - 1, v - 2])
                if i > 0:
                    edges.append([v - 1, v - resolution - 1])
                    if j > 0:
                        faces.append([v - 2, v - 1, v - 1 - resolution, v - 2 - resolution])

            edges.append([v - 1, v - resolution])
            if i > 0:
                faces.append([v - 1, v - resolution, v - 2 * resolution, v - 1 - resolution])

        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh

    def set_inner_radius(self, r,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        # find best shape key
        sk = self.shape_keys[0]
        minimum = np.inf
        min_i=0
        for i in range(1, len(self.shape_keys)):
            dist= np.abs(r - float(self.shape_keys[i]))
            if dist< minimum:
                minimum = dist
                sk = self.shape_keys[i]
                min_i=i

        # keyframe all shape_keys to the current value
        for i in range(1,len(self.shape_keys)):
            frame0 = begin_time * FRAME_RATE
            # keyframe_shape_key(self, self.shape_keys[i], frame0)
            if i!=min_i:
                value = 0
            else:
                value=1

            frame1 = begin_time * FRAME_RATE + np.maximum(1, transition_time * FRAME_RATE)
            set_shape_key_to_value(self, self.shape_keys[i], value, frame1)

