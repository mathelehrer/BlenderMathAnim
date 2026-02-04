import numpy as np
from anytree import Node

from objects.bobject import BObject
from objects.circle import Circle2
from objects.cylinder import Cylinder
from objects.geometry.sphere import Sphere
from objects.text import Text
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs
from interface.ibpy import Vector

pi = np.pi


class DynkinDiagram(BObject):

    def __init__(self, dim=3, labels=[], graph=None, **kwargs):
        """
        example:

        DynkinDiagram(location=[0, 0, 0],dim=3,labels=["4",""])
        """
        self.name = get_from_kwargs(kwargs, "name", "DynkinDiagram")
        self.dim = dim

        if graph is not None:
            size = 2
            self.spheres = []
            self.cylinders = []
            self.labels = []
            self.rings = []
            root = graph
            children = root.children
            directions = []
            n = len(children)
            root_location = Vector([0, 0, 0])
            root_sphere = Sphere(r=0.25, location=root_location, color='plastic_example')
            if root.name[1] == 1:
                self.rings.append(Circle2(center=[root_location.x,root_location.z], radius=0.5, num_points=20,
                                          color="plastic_example",thickness=1, rotation_euler=[0, 0, 0],mode="XZ"))
            self.spheres.append(root_sphere)
            for i, child in enumerate(children):
                u = np.sin(2 * pi * i / n)
                v = np.cos(2 * pi * i / n)
                child_direction = Vector([u, 0, v])
                directions.append(child_direction)
                child_location = root_location + size * child_direction
                child_sphere = Sphere(r=0.25, location=child_location, color='plastic_example')
                self.spheres.append(child_sphere)
                if child.name[1] == 1:
                    self.rings.append(Circle2(center=[child_location.x,child_location.z], radius=0.5,
                                              num_points=20, color="plastic_example",
                                              thickness=1, rotation_euler=[0, 0, 0],mode="XZ"))
                if child.weight > 2:
                    edge = Cylinder.from_start_to_end(start=root_location, end=child_location,
                                                      thickness=0.5, color='plastic_text')
                    self.cylinders.append(edge)

            # TODO extend for children of children

        else:  # linear diagram
            center = 2 * (dim - 1) / 2
            self.spheres = [Sphere(r=0.25, location=[-center + i * 2, 0, 0], color='plastic_example') for i in
                            range(dim)]
            self.cylinders = [
                Cylinder.from_start_to_end(start=[-center + i * 2, 0, 0], end=[-center + (i + 1) * 2, 0, 0],
                                           thickness=0.5, color='plastic_text')
                for i in range(len(labels))
            ]
            self.rings = [
                Circle2(center=[-center + i * 2, 0, 0], radius=0.5, num_points=20, color="plastic_example",
                        thickness=1, rotation_euler=[pi / 2, 0, 0])
                for i in range(dim)
            ]
            self.labels_param = labels
            self.labels = []
            self.without_threes = kwargs.get("without_threes", False)
            for i, l in enumerate(labels):
                if len(l) > 0:
                    if not self.without_threes:
                        self.labels.append(
                            Text(l, location=[-center + 0.8 + i * 2, 0, 0.5], text_size="Large", aligned="center"))
                    elif self.labels_param[i] != '3':
                        self.labels.append(
                            Text(l, location=[-center + 0.8 + i * 2, 0, 0.5], text_size="Large", aligned="center"))

        super().__init__(children=self.spheres + self.cylinders + self.labels + self.rings, name=self.name, **kwargs)

    def appear_customized(self, nodes=[], labels=[], rings=[], edges=[], begin_time=0,
                          transition_time=DEFAULT_ANIMATION_TIME):
        if not self.appeared:
            super().appear(begin_time=begin_time, transition_time=0, children=False)

        for node in nodes:
            self.spheres[node].grow(begin_time=begin_time, transition_time=transition_time)
        for label in labels:
            self.labels[label].write(begin_time=begin_time, transition_time=transition_time)
            self.cylinders[label].grow(begin_time=begin_time, transition_time=transition_time)
        for ring in rings:
            self.rings[ring].grow(begin_time=begin_time, transition_time=transition_time)
        for edge in edges:
            self.cylinders[edge].grow(begin_time=begin_time, transition_time=transition_time)

        return begin_time + transition_time

    def disappear_customized(self, nodes=[], labels=[], rings=[], edges=[], begin_time=0,
                             transition_time=DEFAULT_ANIMATION_TIME):
        if not self.appeared:
            super().appear(begin_time=begin_time, transition_time=0, children=False)

        for node in nodes:
            self.spheres[node].shrink(begin_time=begin_time, transition_time=transition_time)
        for label in labels:
            self.labels[label].disappear(begin_time=begin_time, transition_time=transition_time)
            self.cylinders[label].shrink(begin_time=begin_time, transition_time=transition_time)
        for ring in rings:
            self.rings[ring].shrink(begin_time=begin_time, transition_time=transition_time)
        for edge in edges:
            self.cylinders[edge].shrink(begin_time=begin_time, transition_time=transition_time)

        return begin_time + transition_time

    def appear(self,
               begin_time=0,
               transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):

        super().appear(begin_time=begin_time, transition_time=0, children=False)
        if len(self.spheres) > 0:
            dt = transition_time / len(self.spheres)
            for i, sphere in enumerate(self.spheres):
                sphere.grow(begin_time=begin_time + i * dt, transition_time=dt)

        if len(self.cylinders) > 0:
            dt = transition_time / len(self.cylinders)
            for i, cylinder in enumerate(self.cylinders):
                cylinder.grow(begin_time=begin_time + i * dt, transition_time=dt)

        if len(self.labels) > 0:
            dt = transition_time / len(self.labels)
            for i, label in enumerate(self.labels):
                label.write(begin_time=begin_time + i * dt, transition_time=dt)

        return begin_time + transition_time

    def copy(self):
        return DynkinDiagram(dim=self.dim, labels=self.labels_param,
                             location=self.ref_obj.location.copy(), scale=self.ref_obj.scale)

    @classmethod
    def d4(cls, **kwargs):
        graph = Node((0, 1))
        for i in range(3):
            node = Node((i, 1))
            node.parent = graph
            node.weight = 3
        return DynkinDiagram(dim=4, graph=graph, **kwargs)
