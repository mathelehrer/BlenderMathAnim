from collections import OrderedDict

import bpy
import numpy as np
from mathutils import Vector

from interface import ibpy
from mathematics.mathematica.mathematica import find_closest
from mathematics.zeros import chop
from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import to_vector


class Polygon(BObject):
    """
    Create a polygon from vertices and edges
    The normal by default points away from the origin:
    """

    def __init__(self, vertices, obj=None, initial_function=None, edges=None, index=0, reordering=True,**kwargs):
        """
        :param vertices: a list of vertices
        :param face: the vertices of the list, that define the face
        :param origin: the origin of the corresponding polyhedron
        :param index:
        :param index_base:
        :param kwargs:
        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', str(len(vertices)) + "gon_" + str(index))

        # reorder vertices to make a convex face
        if reordering:
            vertices = [to_vector(v) for v in vertices]

        center = Vector()
        for v in vertices:
            center+=v
        center =1/len(vertices) *center

        # create a base
        a  =vertices[0]-center
        a.normalize()
        i = 1
        while i<len(vertices) and to_vector(chop(a.cross(vertices[i]-center))).length==0: # find the first non-parallel
            i+=1

        if i<len(vertices):
            b = vertices[i]-center
            n = a.cross(b)
            n.normalize()
            b = a.cross(n)
            b.normalize()
           
            vertices=sorted(vertices, key=lambda x: np.arctan2(a.dot(x-center),b.dot(x-center)))

        self.vertices0 = vertices  # store initial positions to have later access for it
        if initial_function is None:
            self.vertices = [to_vector(v) for v in vertices]
        else:
            self.vertices = [Vector(initial_function(v)) for v in vertices]

        if obj is None:
            new_mesh = bpy.data.meshes.new(self.name + "_mesh_" + str(index))
            face = []
            for i, v in enumerate(vertices):
                face.append(i)

            if edges is None:
                edges = []
                for i in range(len(vertices) - 1):
                    edges.append([i, i + 1])
                edges.append([len(vertices) - 1, 0])
            new_mesh.from_pydata(self.vertices, edges, [face])
            new_mesh.update()
            super().__init__(name=self.name, mesh=new_mesh, **kwargs)
        else:
            # the mesh can be created in a batch in the class method batch_create
            super().__init__(name=self.name, obj=obj, **kwargs)

        self.last_shapekey = 0

        if 'thickness' in kwargs:
            thickness = self.get_from_kwargs('thickness', 0.01)
            ibpy.add_solidify_modifier(self, thickness)

    @classmethod
    def batch_create(cls, name="PolygonBatch", vertices_list=[], colors=['drawing'], initial_function=None, link=False,
                     **kwargs):
        objs = []
        i = 0
        for vertices in vertices_list:
            obj_name = name + "_" + str(i)
            if initial_function is not None:
                verts = [initial_function(v) for v in vertices]
            else:
                verts = vertices
            mesh = ibpy.create_mesh(verts, [], faces=[list(range(len(verts)))])
            obj = ibpy.create(mesh, obj_name, location=Vector())
            objs.append(obj)
            if link:
                ibpy.link(obj)
            i += 1

        bobs = []
        for i, obj in enumerate(objs):
            if i >= len(colors):
                col = colors[-1]
            else:
                col = colors[i]
            bob = Polygon(name=obj.name, vertices=vertices_list[i], initial_function=initial_function, index=i, obj=obj,
                          color=col, **kwargs)
            bobs.append(bob)

        return bobs

    def morph_to(self, projector=lambda v: v, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        name = 'projection' + str(self.last_shapekey)
        old_sk = ibpy.add_shape_key(self, name=name, relative=False)
        ibpy.set_shape_key_eval_time(self, self.last_shapekey * 10, begin_time * FRAME_RATE)
        self.last_shapekey += 1

        # ibpy.keyframe_shape_key(self,name, frame=begin_time * FRAME_RATE)
        # always apply the projection to the initial state
        for old_sk, vertex0 in zip(old_sk.data, self.vertices0):
            old_sk.co = projector(vertex0)
        ibpy.set_shape_key_eval_time(self, self.last_shapekey * 10, (begin_time + transition_time) * FRAME_RATE)
        # ibpy.set_shape_key_to_value(self,name,1, frame=(begin_time+transition_time) * FRAME_RATE)
        return begin_time+transition_time
    def morph_to2(self, new_vertices=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
               the function is buggy, the relative shape keys don't work for more than one transformation
               :param projector:
               :param begin_time:
               :param transition_time:
               :return:
        """
        self.last_shapekey += 1
        name = 'projection' + str(self.last_shapekey)
        if new_vertices is None:
            new_vertices = self.vertices

        old_sk = ibpy.add_shape_key(self, name=name)
        ibpy.keyframe_shape_key(self, name, frame=begin_time * FRAME_RATE)
        for i in range(len(old_sk.data)):
            old_sk.data[i].co = new_vertices[i]
        ibpy.set_shape_key_to_value(self, name, 1, frame=(begin_time + transition_time) * FRAME_RATE)
        return begin_time + transition_time

    def grow(self, scale=None, alpha=1,begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center', pivot=None,
             initial_scale=0):
        if pivot is None:
            # grow from first vertex as default
            pivot = self.vertices[0]
        return super().grow(alpha=alpha,scale=scale, begin_time=begin_time, transition_time=transition_time, modus=modus,
                                pivot=pivot)

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        pivot = self.vertices[0]
        if alpha==0:
            return super().grow(initial_scale=1,scale=0, begin_time=begin_time, transition_time=transition_time, pivot=pivot)
        else:
            return super().disappear(alpha=alpha,begin_time=begin_time,
                                     transition_time=transition_time)

class Triangle(BObject):
    """
        Create an equilateral triangel
        The default center is in the center of mass located at (0,0,0)
        """

    def __init__(self, name='Triangle', **kwargs):
        """
        :param vertices: a list of vertices
        :param face: the vertices of the list, that define the face
        :param origin: the origin of the corresponding polyhedron
        :param index:
        :param index_base:
        :param kwargs:
        """
        self.kwargs = kwargs
        new_mesh = bpy.data.meshes.new(name + "_mesh")

        face = []
        r3 = np.sqrt(3)
        center = Vector([1 / 2, 1 / 2 / r3, 0])

        vertices = [Vector(), Vector([1 / 2, 0, 0]), Vector([1, 0, 0]), Vector([1 / 4, r3 / 4, 0]),
                    Vector([3 / 4, r3 / 4, 0]), Vector([1 / 2, r3 / 2, 0])]
        vertices = [v - center for v in vertices]
        edges = [[0, 1], [1, 2], [0, 3], [1, 3], [1, 4], [2, 4], [3, 4], [1, 5], [4, 5]]
        faces = [[0, 1, 3], [1, 4, 3], [1, 2, 4], [5, 3, 4]]
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()

        super().__init__(name=name, mesh=new_mesh, **kwargs)
        thickness = self.get_from_kwargs('thickness', 0.1)
        offset = self.get_from_kwargs('offset', 0)
        ibpy.add_solidify_modifier(self, thickness, offset=offset)

        col_dic = {
            0: [1, 0, 0, 1],
            1: [1, 1, 0, 1],
            2: [0, 1, 0, 1],
            3: [1, 0, 1, 1],
            4: [0, 1, 1, 1],
            5: [0, 0, 1, 1]
        }

        # when the vertex colors are assigned, the algorithm loops through all faces and assignes the vertex colors for each face vertex separately
        colors = [
            col_dic[0],
            col_dic[1],
            col_dic[3],
            col_dic[1],
            col_dic[4],
            col_dic[3],
            col_dic[1],
            col_dic[2],
            col_dic[4],
            col_dic[5],
            col_dic[3],
            col_dic[4]
        ]
        ibpy.set_vertex_colors(self, colors)
