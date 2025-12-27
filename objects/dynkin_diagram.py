import numpy as np

from objects.bobject import BObject
from objects.circle import Circle2
from objects.cylinder import Cylinder
from objects.geometry.sphere import Sphere
from objects.text import Text
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs

pi = np.pi

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
        self.rings = [Circle2(center=[-center+i*2,0,0],radius=0.5,num_points=20,color="plastic_example",thickness=1,rotation_euler=[pi/2,0,0]) for i in range(dim)]
        self.labels = []
        for i,l in enumerate(labels):
            if len(l)>0:
                self.labels.append(Text(l,location=[-center+0.8+i*2,0,0.5],text_size="Large",aligned="center"))

        super().__init__(children=self.spheres+self.cylinders+self.labels+self.rings,name=self.name, **kwargs)

    def appear_customized(self,nodes = [], labels =[],rings = [],begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if not self.appeared:
            super().appear(begin_time=begin_time, transition_time=0,children=False)

        for node in nodes:
            self.spheres[node].grow(begin_time=begin_time, transition_time=transition_time)
        for label in labels:
            self.labels[label].write(begin_time=begin_time, transition_time=transition_time)
            self.cylinders[label].grow(begin_time=begin_time, transition_time=transition_time)
        for ring in rings:
            self.rings[ring].grow(begin_time=begin_time, transition_time=transition_time)

        return begin_time+transition_time

    def disappear_customized(self,nodes = [], labels =[],rings = [],begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if not self.appeared:
            super().appear(begin_time=begin_time, transition_time=0,children=False)

        for node in nodes:
            self.spheres[node].shrink(begin_time=begin_time, transition_time=transition_time)
        for label in labels:
            self.labels[label].disappear(begin_time=begin_time, transition_time=transition_time)
            self.cylinders[label].shrink(begin_time=begin_time, transition_time=transition_time)
        for ring in rings:
            self.rings[ring].shrink(begin_time=begin_time, transition_time=transition_time)

        return begin_time+transition_time


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

