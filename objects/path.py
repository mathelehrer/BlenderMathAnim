import numpy as np
from mathutils import Vector, Matrix

from interface import ibpy
from interface.ibpy import add_cone, add_cylinder
from objects.circle import BezierCircle
from objects.cone import Cone
from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.derived_objects.p_arrow import PArrow
from objects.geometry.sphere import Sphere
from objects.tex_bobject import SimpleTexBObject
from objects.torus import Torus
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import to_vector


class Path(BObject):
    """
    Create a path for the computation of the Onsager solution (a la Feynman, Biswanathan)

    """

    def __init__(self, location=Vector(), path_indices=[0, 0, 0, 0], **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name', 'Path')
        path = Vector([0, 1, 0])
        self.vertices = []
        self.edges = []
        for p in path_indices:
            old_location = location
            path = self.next_path(path, p)
            location = old_location + path
            sphere = Sphere(0.15, location=location, color='magnet',direction='Y')
            self.vertices.append(sphere)
            if np.random.random()<0.5:
                v=-1
            else:
                v=1
            sphere.shader_value(old_value=0,new_value=v,transition_time=0)

            if p ==0:
                color = 'text'
            elif p==1:
                color = 'drawing'
            else:
                color = 'gray_3'
            self.edges.append(Cylinder.from_start_to_end(start=old_location, end=location, thickness=0.5,color=color))
        super().__init__(children=[*self.vertices, *self.edges], name=name, **kwargs)

    def next_path(self, current, direction):
        if direction == 0:
            path = current
        elif direction == 1:
            path = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ current
        elif direction == 2:
            path = Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) @ current
        return path

    def appear(self, alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, clear_data=False, silent=False):
        super().appear(begin_time=begin_time, transition_time=transition_time)

        [sphere.appear(begin_time=begin_time, transition_time=transition_time) for sphere in self.vertices]
        [edge.appear(begin_time=begin_time, transition_time=transition_time) for edge in self.edges]

        return begin_time + transition_time

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        super().disappear(begin_time=begin_time, transition_time=transition_time)
        [sphere.disappear(begin_time=begin_time, transition_time=transition_time) for sphere in self.vertices]
        [edge.disappear(begin_time=begin_time, transition_time=transition_time) for edge in self.edges]

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center', pivot=None,
             initial_scale=0):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        dt = transition_time / len(self.vertices)
        t0 = begin_time
        for vertex, edge in zip(self.vertices, self.edges):
            t0 = edge.grow(begin_time=t0, transition_time=dt / 2)
            t0 = vertex.grow(begin_time=t0, transition_time=dt / 2)

        return begin_time + transition_time


def vector2tuple(vec):
    return int(vec.x), int(vec.y), int(vec.z)


class BiswanathanPath(Path):
    """
    create a closed path, that uses every edge at most once
    """

    def __init__(self,**kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name','Biswanathan')

        success = False

        while not success:

            points = []
            edges = []
            path_code = []



            path = Vector([0, 1, 0])
            start = Vector()
            old_index = 0
            points.append(vector2tuple(start))
            next_point = None

            while next_point != Vector() and len(points)<300:
                code = np.random.randint(0, 3)
                new_path = super().next_path(path, code)
                next_point = start + new_path
                next_tuple = vector2tuple(next_point)
                if next_tuple not in points:
                    points.append(next_tuple)
                    #print("new point: ",next_tuple)
                    new_edge = (old_index, len(points) - 1)
                    edges.append(new_edge)
                    #print("new edge: ",new_edge)
                    old_index = len(points)-1
                    start = next_point
                    path= new_path
                    path_code.append(code)
                else:
                    index = points.index(next_tuple)
                    new_edge = (old_index, index)
                    if new_edge in edges or (index, old_index) in edges:
                        pass
                    else:
                        #print("index of old point: ", index, next_tuple)
                        #print(points)
                        edges.append(new_edge)
                        #print("new edge: ",new_edge)
                        #print("edges: ",edges)
                        old_index = index
                        start = next_point
                        path=new_path
                        path_code.append(code)
            if next_point == Vector():
                success = True
                print(path_code)

        super().__init__(path_indices=path_code,name=name,**self.kwargs)
