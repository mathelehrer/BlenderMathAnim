import numpy as np
from anytree import Node

from objects.bobject import BObject
from objects.circle import Circle2
from objects.cylinder import Cylinder
from objects.geometry.sphere import Sphere
from objects.text import Text
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs
from interface.ibpy import Vector, to_vector

pi = np.pi


class DynkinDiagram(BObject):

    def __init__(self, dim=3, labels=[], graph=None, move_to_center=False, **kwargs):
        """
        example:

        DynkinDiagram(location=[0, 0, 0],dim=3,labels=["4",""])
        """
        self.name = get_from_kwargs(kwargs, "name", "DynkinDiagram")
        self.dim = dim
        self.locations = []
        self.scale = get_from_kwargs(kwargs, "scale", [1, 1, 1])
        self.text_size = get_from_kwargs(kwargs, "text_size", "Large")
        self.text_offset = Vector([-1.2, 0, 0.5])
        if self.text_size == "Huge":
            self.text_offset = Vector([-1.5, 0, 0.85])
        if isinstance(self.scale, (int, float)):
            self.scale = [self.scale] * 3

        if graph is not None:
            self.spheres = []
            self.cylinders = []
            self.labels = []
            self.rings = []
            root = graph
            root_location = Vector([0, 0, 0])
            self.locations.append(root_location)
            root_sphere = Sphere(r=0.25, location=root_location, color='plastic_example')
            if root.name[1] == 1:
                self.rings.append(Circle2(center=[root_location.x, root_location.z], radius=0.5, num_points=20,
                                          color="plastic_example", thickness=1, rotation_euler=[0, 0, 0], mode="XZ"))
            self.spheres.append(root_sphere)
            self._place_children(root, root_location)

        else:  # linear diagram
            center = 2 * (dim - 1) / 2
            for i in range(dim):
                child_location = Vector([-center + i * 2, 0, 0])
                self.spheres.append(Sphere(r=0.25, location=child_location, color='plastic_example'))
                self.locations.append(child_location)
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
                            Text(l, location=Vector([-center + (i + 1) * 2, 0, 0]) + self.text_offset,
                                 text_size=self.text_size, aligned="center"))
                    elif self.labels_param[i] != '3':
                        self.labels.append(
                            Text(l, location=Vector([-center + 0.8 + (i + 1) * 2, 0, 0.5]) + self.text_offset,
                                 text_size=self.text_size, aligned="center"))

        if move_to_center:
            # compute the center of the diagram
            center = sum(self.locations, Vector()) / len(self.locations)
            location = to_vector(get_from_kwargs(kwargs, "location", Vector())) - Vector(
                [c * s for c, s in zip(center, self.scale)])

            super().__init__(children=self.spheres + self.cylinders + self.labels + self.rings, name=self.name,
                             location=location, scale=self.scale, **kwargs)
        else:
            super().__init__(children=self.spheres + self.cylinders + self.labels + self.rings, name=self.name,
                             scale=self.scale, **kwargs)

        self.ring_tags = [False]*self.dim

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
            if not self.ring_tags[ring]:
                self.rings[ring].grow(begin_time=begin_time, transition_time=transition_time)
                self.ring_tags[ring]=True
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
        t0 = begin_time
        for ring in rings:
            dt = transition_time/len(rings)
            if self.ring_tags[ring]:
                self.rings[ring].shrink(begin_time=begin_time, transition_time=transition_time)
                self.ring_tags[ring]=False
                if ring<self.dim-1:
                    # adjust edge to the right of the ring
                    self.cylinders[ring].rescale(rescale=[1,1,1.25],begin_time=t0,transition_time=dt)
                    self.cylinders[ring].move(direction=[-0.25,0,0],begin_time=t0,transition_time=dt)
                if ring>0:
                    # adjust edge to the left of the ring
                    self.cylinders[ring-1].rescale(rescale=[1,1,1.25],begin_time=t0,transition_time=dt)
                    self.cylinders[ring-1].move(direction=[0.25,0,0],begin_time=t0,transition_time=dt)
                t0 = t0+dt
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

        if len(self.rings) > 0:
            dt = transition_time / len(self.rings)
            for i, ring in enumerate(self.rings):
                ring.grow(begin_time=begin_time + i * dt, transition_time=dt)
                self.ring_tags[i]=True

        return begin_time + transition_time

    def copy(self):
        return DynkinDiagram(dim=self.dim, labels=self.labels_param,
                             location=self.ref_obj.location.copy(), scale=self.ref_obj.scale)

    @classmethod
    def d4(cls, **kwargs):
        rings = get_from_kwargs(kwargs, "rings", [1, 1, 1, 1])
        graph = Node((0, rings[0]))
        for i in range(1,4):
            node = Node((i, rings[i]))
            node.parent = graph
            node.weight = 3
        return DynkinDiagram(dim=4, graph=graph, **kwargs)

    @classmethod
    def from_string(cls, dynkin_string, **kwargs):
        if dynkin_string=="x3x3x *b3x":
            return DynkinDiagram.d4(rings=[1,1,1,1], **kwargs)
        elif dynkin_string=="o3x3x *b3x":
            return DynkinDiagram.d4(rings=[0, 1, 1,1], **kwargs)
        elif dynkin_string=="x3o3x *b3x":
            return DynkinDiagram.d4(rings=[1,0,1,1], **kwargs)
        elif dynkin_string=="o3x3o *b3x":
            return DynkinDiagram.d4(rings=[0,1,0,1], **kwargs)
        elif dynkin_string=="o3o3x *b3x":
            return DynkinDiagram.d4(rings=[1,0,1,0], **kwargs)
        elif dynkin_string=="o3o3o *b3x":
            return DynkinDiagram.d4(rings=[0,0,0,1], **kwargs)
        elif dynkin_string=="o3x3o *b3o":
            return DynkinDiagram.d4(rings=[0,1,0,0],**kwargs)
        # deal with linear diagrams only
        dim = 0
        last = None
        first = None
        weight = None
        for char in dynkin_string:
            if char == 'x' or char == 'o':
                if char == 'x':
                    node = Node((dim, 1))
                else:
                    node = Node((dim, 0))
                if last is not None:
                    node.parent = last
                else:
                    first = node
                last = node
                dim += 1
                if weight is not None:
                    node.weight = weight
                    weight = None
            else:
                weight = int(char)
        return DynkinDiagram(dim=dim, graph=first, move_to_center=True, **kwargs)

    def _place_children(self, node, node_location):
        children = node.children

        if len(children) == 1:
            # linear diagram
            child = children[0]
            child_direction = Vector([1, 0, 0])
            child_location = node_location + 2 * child_direction
            self.locations.append(child_location)
            child_sphere = Sphere(r=0.25, location=child_location, color='plastic_example')
            self.spheres.append(child_sphere)
            if child.name[1] == 1:
                self.rings.append(Circle2(center=[child_location.x, child_location.z], radius=0.5,
                                          num_points=20, color="plastic_example",
                                          thickness=1, rotation_euler=[0, 0, 0], mode="XZ"))
            if hasattr(child, "weight"):
                if child.weight > 2:
                    ring_size = 0.5
                    # avoid intersection between edges and rings
                    if child.name[1] == 1:
                        end_location = child_location - ring_size * Vector([1, 0, 0])
                    else:
                        end_location = child_location
                    if child.parent.name[1] == 1:
                        start_location = node_location + ring_size * Vector([1, 0, 0])
                    else:
                        start_location = node_location
                    edge = Cylinder.from_start_to_end(start=start_location, end=end_location,
                                                      thickness=0.5, color='plastic_text')
                    self.cylinders.append(edge)
                    if child.weight > 3:
                        self.labels.append(
                            Text(str(child.weight), location=child_location + self.text_offset,
                                 text_size=self.text_size, aligned="center")
                        )

            self._place_children(child, child_location)
        else:
            directions = []
            n = len(children)
            for i, child in enumerate(children):
                u = np.sin(2 * pi * i / n)
                v = np.cos(2 * pi * i / n)
                child_direction = Vector([u, 0, v])
                directions.append(child_direction)
                child_location = node_location + 2 * child_direction
                self.locations.append(child_location)
                child_sphere = Sphere(r=0.25, location=child_location, color='plastic_example')
                self.spheres.append(child_sphere)
                if child.name[1] == 1:
                    self.rings.append(Circle2(center=[child_location.x, child_location.z], radius=0.5,
                                              num_points=20, color="plastic_example",
                                              thickness=1, rotation_euler=[0, 0, 0], mode="XZ"))
                if hasattr(child, "weight"):
                    if child.weight > 2:
                        ring_size = 0.5
                        # avoid intersection between edges and rings
                        if child.name[1] == 1:
                            end_location = child_location - ring_size * Vector(child_direction)
                        else:
                            end_location = child_location
                        if child.parent.name[1] == 1:
                            start_location = node_location + ring_size * Vector(child_direction)
                        else:
                            start_location = node_location
                        edge = Cylinder.from_start_to_end(start=start_location, end=end_location,
                                                          thickness=0.5, color='plastic_text')
                        self.cylinders.append(edge)
                        if child.weight > 3:
                            self.labels.append(
                                Text(str(child.weight), location=child_location + self.text_offset,
                                     text_size=self.text_size, aligned="center")
                            )

                self._place_children(child, child_location)
