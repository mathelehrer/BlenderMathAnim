import bmesh
import bpy
import mathutils
import numpy as np

from interface.ibpy import add_sphere, add_shape_key, Vector
from objects.bobject import BObject
from objects.geometry.geo_bobject import GeoBObject

from random import uniform

from utils.utils import to_vector


class Sphere(GeoBObject):
    """
    a sphere is created as a uv-mesh or an iso-sphere
    depending on the choice of the attribute 'mesh_type'
    mesh_type = 'uv'
    mesh_type = 'iso'
    """

    def __init__(self, r, **kwargs):
        """
        :param r:  radius of the sphere
        :param kwargs:
        """
        self.kwargs = kwargs
        mesh_type = self.get_from_kwargs('mesh_type', 'iso')

        if mesh_type == 'uv':
            default_name = 'UV-Sphere'
        else:
            default_name = 'ISO-Sphere'

        smooth = self.get_from_kwargs('smooth',True)

        name = self.get_from_kwargs('name', default_name)

        res = self.get_from_kwargs('resolution', 4)
        sphere = add_sphere(radius=r, mesh_type=mesh_type, resolution=res, location=(0, 0, 0), scale=(1, 1, 1),
                            enter_editmode=False,
                            align='WORLD',smooth=smooth)

        super().__init__(obj=sphere, name=name, **kwargs)
        self.label_sep = 2 * r  # override default label separation


class HalfSphere(BObject):
    """
    Create half a sphere from scratch:
    """

    def __init__(self, radius=1, resolution=10, location=Vector(), **kwargs):
        self.kwargs = kwargs
        self.radius = radius
        self.resolution = resolution
        self.location = to_vector(location)

        self.name = self.get_from_kwargs('name', 'HalfSphere')
        mesh = self.create_mesh()

        super().__init__(name=self.name, mesh=mesh, **kwargs)

    def create_mesh(self):
        vertices = []
        edges = []
        faces = []

        first_row = True
        dtheta = np.pi / self.resolution
        for i in range(0, self.resolution + 1):
            theta = dtheta*i
            # create vertices
            r = self.radius * np.sin(theta)
            if i == 0 or i==self.resolution:
                pos = Vector([0, 0, self.radius*np.cos(theta)]) + self.location
                vertices.append((pos.x,pos.y,pos.z))
            else:
                dphi = np.pi / self.resolution
                for j in range(0, self.resolution+1):
                    phi = j * dphi
                    r = self.radius*np.sin(theta)
                    pos = self.location+Vector([r * np.cos(phi), r * np.sin(phi), self.radius*np.cos(theta)])
                    vertices.append((pos.x,pos.y,pos.z))
                    v = len(vertices)

                    if j > 0:
                        # edge to the previous
                        edges.append([v - 1, v - 2])
                        if i == 1:
                            # edge to the first
                            edges.append([v - 1, 0])
                            if j > 0:
                                # triangle to the first
                                faces.append([v - 2, v - 1, 0])
                        else:
                            # edge to the one below
                            edges.append([v - 1, v - self.resolution - 2])
                            if j > 0:
                                # face below
                                faces.append([v - 2, v - 1, v -2- self.resolution, v - 3 - self.resolution])

        # connect last point
        v= len(vertices)
        for i in range(1,self.resolution+2):
            edges.append([v-1,v-1-i])
            if i<self.resolution+1:
                faces.append([v-1,v-i-1,v-i-2])
        new_mesh = bpy.data.meshes.new(self.name + '_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        return new_mesh

class StackOfSpheres(BObject):
    """
       a stack of spheres is created with a very simple slope definition.
       There is no brain put into it. The parameter dim controls the height of the stack
       You can make appear the spheres from bottom to top by adding incrementals
       or you can shrink the spheres from top to bottom by removing incrementals
       """

    def __init__(self, radius=1, dim=5, number_of_spheres=None, **kwargs):
        self.kwargs = kwargs
        self.spheres = []
        self.visible_spheres = 0
        self.first_time = True

        self.name = self.get_from_kwargs('name', "StackOfSpheres")
        if number_of_spheres:
            if number_of_spheres > 114:
                raise Exception("class not configured for such a big number of spheres, find the appropriate dimension "
                                "first")
            if number_of_spheres <= 11:
                self.dim = 3
            elif number_of_spheres <= 39:
                self.dim = 4
            elif number_of_spheres <= 42:
                self.dim = 5
            elif number_of_spheres <= 90:
                self.dim = 6
            elif number_of_spheres <= 115:
                self.dim = 7
        else:
            number_of_spheres = np.Infinity
            self.dim = dim
        r3 = np.sqrt(3)
        r6 = np.sqrt(6)
        h = self.dim
        w = self.dim + 4
        self.stack_location = self.get_from_kwargs('location', [0, 0, 0])
        self.stack_scale = self.get_from_kwargs('scale', [1, 1, 1])
        count = 0
        for k in range(self.dim):
            for i in range(-self.dim + k, self.dim - k):
                for j in range(-self.dim + k, self.dim - k):
                    x = 2 * i + ((j + k) % 2)
                    y = r3 * (j + 0.333 * (k % 2))
                    z = 2 * r6 / 3 * k
                    if x * x + y * y < (h - h / w * z) ** 2:
                        x *= radius
                        y *= radius
                        z *= radius
                        self.spheres.append(
                            Sphere(radius, location=[x, y, z], name=self.name + "_" + str(count), **self.kwargs))
                        count += 1
                        if count == number_of_spheres:
                            super().__init__(children=self.spheres, location=self.stack_location, name=self.name,
                                             scale=self.stack_scale,
                                             **kwargs)
                            return
        super().__init__(children=self.spheres, location=self.stack_location, scale=self.stack_scale, name=self.name,
                         **kwargs)

    def appear(self,
               count=0,
               incr=1,
               begin_time=0,
               transition_time=0,
               **kwargs):
        """
        :param count: total number of spheres that are supposed to be visible
        :param incr: increase or decrease the number of  visible spheres by this incr
        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        """
        if self.first_time:
            super().appear(begin_time=begin_time, transition_time=0)  # link stack
            self.first_time = False

        if count > 0:
            self.appear(count=0, incr=count - self.visible_spheres, begin_time=begin_time,
                        transition_time=transition_time)
        else:
            if incr > 0:
                if self.visible_spheres + incr < len(self.spheres):
                    upper_limit = self.visible_spheres + incr
                else:
                    upper_limit = len(self.spheres)
                dt = transition_time / incr
                for s in range(self.visible_spheres, upper_limit):
                    self.spheres[s].grow(begin_time=begin_time, transition_time=dt)
                    self.visible_spheres += 1
                    begin_time += dt
            else:
                if self.visible_spheres + incr >= 0:
                    lower_limit = self.visible_spheres + incr
                else:
                    lower_limit = 0
                dt = transition_time / np.abs(incr)
                for s in range(self.visible_spheres - 1, lower_limit - 1, -1):
                    self.spheres[s].shrink(begin_time=begin_time, transition_time=dt)
                    begin_time += dt
                    self.visible_spheres -= 1


def invert(m, r):
    '''
    see mathematica notebook MapOfCirclesUnderInversion.nb
    :param m: center of source circle
    :param r: radius of source circle
    :return: M,R center and radius of the image circle
    '''

    m2 = m * np.conj(m)
    r2 = r * r
    i = 1j
    delta = r2 - 1 - m2 + i * (m - np.conj(m))
    M = (-np.conj(m) + i * m2 - i * r2) / delta
    R = r / np.abs(delta)
    return M, R


def get_polar_coordinates(vertex, curvature, location):
    r = 1 / curvature
    loc = mathutils.Vector([np.real(location), np.imag(location), 0])
    v = vertex - loc
    theta = np.arccos(np.maximum(np.minimum(1, v.z / r), -1))
    phi = np.arctan2(v.y, v.x)
    return theta, phi


class MultiSphere(BObject):
    """
       a mesh with lots of spheres is created
       """

    def __init__(self, locations, curvatures, **kwargs):
        """

        :param locations:
        :param curvatures:
        :param kwargs: remaining parameters the same as for sphere
        """

        self.kwargs = kwargs

        name = self.get_from_kwargs('name', 'MultiSphere')
        mesh_type = self.get_from_kwargs('mesh_type', 'iso')
        max_curvature = self.get_from_kwargs('max_curvature', 100)

        # create an empty mesh object and add it to the scene
        sphereMesh = bpy.data.meshes.new('AllSpheres')
        sphereObj = bpy.data.objects.new('AllSpheres', sphereMesh)

        bm = bmesh.new()
        color_layer = bm.loops.layers.color.new("curvature_color")

        old_faces_length = 0
        old_verts_length = 0

        inverted_coords = []

        for loc, curv in zip(locations, curvatures):
            locMatrix = mathutils.Matrix.Translation([np.real(loc), np.imag(loc), 0])
            scaleMatrix = mathutils.Matrix.Scale(1 / curv, 4)

            if curv < max_curvature:
                if curv < 10:
                    res = 4
                elif curv < 100:
                    res = 3
                else:
                    res = 2

                r = curv / max_curvature
                alpha = 1 - 1.9 / curv  # the highest curvature 2 is almost transparent
                color = [1, 1, 1 - r ** 2, 1 - alpha]  # the alpha channel is used to encode the transmission

                if mesh_type == 'uv':
                    bmesh.ops.create_uvsphere(bm, u_segments=2 ** res, v_segments=2 ** res, radius=1,
                                              matrix=locMatrix @ scaleMatrix)
                else:
                    bmesh.ops.create_icosphere(bm, radius=1, subdivisions=res, matrix=locMatrix @ scaleMatrix)

                # attach appropriate color to vertices of new faces
                bm.faces.ensure_lookup_table()
                for f in range(old_faces_length, len(bm.faces)):
                    face = bm.faces[f]
                    face.smooth = True
                    for loop in face.loops:
                        loop[color_layer] = color
                old_faces_length = len(bm.faces)

                # prepare shape_key coordinates
                bm.verts.ensure_lookup_table()
                for v in range(old_verts_length, len(bm.verts)):
                    vertex = bm.verts[v]
                    # get polar coordinates
                    theta, phi = get_polar_coordinates(vertex.co, curv, loc)
                    M, R = invert(loc, 1 / curv)
                    sintheta = np.sin(theta)
                    inverted_coords.append(
                        [R * sintheta * np.cos(phi) + np.real(M), R * sintheta * np.sin(phi) + np.imag(M),
                         R * np.cos(theta)])
                old_verts_length = len(bm.verts)

        bm.to_mesh(sphereMesh)
        bm.free()
        super().__init__(obj=sphereObj, name=name, **kwargs, roughness=0, ior=1, metallic=0.2)

        # prepare shape keys
        old_sk = add_shape_key(self, 'Basis')
        add_shape_key(self, 'inversion', previous=old_sk)
        for v in range(len(old_sk.data)):
            old_sk.data[v].co = inverted_coords[v]
