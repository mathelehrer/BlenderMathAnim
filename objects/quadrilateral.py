import numpy as np
from mathutils import Vector

from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier
from geometry_nodes.geometry_nodes_modifier import QuadModifier
from objects.bobject import BObject
from objects.cube import Cube
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.utils import to_vector

pi = np.pi

class BQuadrilateral(BObject):
    def __init__(self, vertices,name="Quadrilateral",resolution=100,geo_rotation=None, **kwargs):
        """
        make a rectangular frame defined by its corner vertices. If length and width are switched,
        just cycle the vertices by one vertex
        """
        self.vertices = [to_vector(v) for v in vertices]

        a = (self.vertices[0]-self.vertices[1])
        b = (self.vertices[0]-self.vertices[3])
        normal = a.cross(b)
        normal.normalize()

        sum = Vector()
        for v in vertices:
            sum = sum +to_vector(v)
        center = sum / len(vertices)
        cube = Cube()
        self.quad_modifier =QuadModifier(name=name,
                                    height=b.length,
                                    width=a.length,
                                    normal = normal,
                                    geo_location=center,
                                    resolution=resolution,
                                    **kwargs)

        cube.add_mesh_modifier(type="NODES", node_modifier=self.quad_modifier)
        super().__init__(obj=cube.ref_obj,name=name,**kwargs)

    def grow(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):

        grow_node = get_geometry_node_from_modifier(self.quad_modifier,label="Grow")
        ibpy.change_default_value(grow_node,from_value=-0.01,to_value=1,begin_time=begin_time,
                                  transition_time=transition_time)
        return begin_time+transition_time

    def disappear(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        grow_node = get_geometry_node_from_modifier(self.quad_modifier, label="Grow")
        ibpy.change_default_value(grow_node, from_value=1, to_value=-0.01, begin_time=begin_time,
                                  transition_time=transition_time)

        return begin_time + transition_time
