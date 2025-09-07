import numpy as np

from geometry_nodes.geometry_nodes_modifier import LogoModifier
from interface import ibpy
from interface.ibpy import create_mesh
from objects.bobject import BObject
from objects.circle import Circle, Circle2
from objects.coordinate_system import CoordinateSystem
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.geometry.sphere import Sphere
from objects.text import Text
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs



class DynkinDiagram(BObject):

    def __init__(self, dim=3,labels=[], **kwargs):
        """
        example:

        DynkinDiagram(location=[0, 0, 0],dim=3,labels=["4",""])
        """
        self.name = get_from_kwargs(kwargs,"name","DynkinDiagram")
        center = 2*(dim-1) /2
        self.spheres = [Sphere(r=0.25, location=[-center+ i * 2, 0, 0], color='plastic_example') for i in range(dim)]
        self.cylinders = [Cylinder.from_start_to_end(start=[-center+i*2, 0, 0], end=[-center+(i+1)*2, 0,0], thickness=0.5, color='plastic_text') for i in range(dim-1)]
        self.labels = []
        for i,l in enumerate(labels):
            if len(l)>0:
                self.labels.append(Text(l,location=[-center+0.8+i*2,0,0.5],text_size="Large",aligned="center"))

        super().__init__(children=self.spheres+self.cylinders+self.labels,name=self.name, **kwargs)

    def appear(self,
               begin_time=0,
               transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):

        super().appear(begin_time=begin_time, transition_time=0,children=False)

        dt = transition_time / len(self.spheres)
        for i,sphere in enumerate(self.spheres):
            sphere.grow(begin_time=begin_time+i*dt, transition_time=dt)

        dt = transition_time / len(self.cylinders)
        for i,cylinder in enumerate(self.cylinders):
            cylinder.grow(begin_time=begin_time+i*dt, transition_time=dt)

        dt = transition_time / len(self.labels)
        for i,label in enumerate(self.labels):
            label.write(begin_time=begin_time+i*dt, transition_time=dt)

        return begin_time+transition_time

