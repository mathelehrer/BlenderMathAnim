import numpy as np
from anytree import RenderTree
from mathutils import Vector

from objects.bobject import BObject
from objects.tree.node import Node
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME


class Tree(BObject):
    """
    Generates a tree object
    example

    """
    def __init__(self, root_element, width, height,**kwargs):
        self.kwargs = kwargs
        self.root_element = root_element
        self.root_node = None
        self.width = width
        self.height = height
        self.max_level = self.get_max_level()
        self.elements_node_map = {}
        self.level_shrink = 0.7
        self.node_thickness = self.get_from_kwargs('node_thickness', 0.2)

        self.nodes = []
        self.lines = []
        self.name = self.get_from_kwargs('name', 'tree')
        location = self.get_from_kwargs('location', Vector([0, 0, 0]))
        self.scale = self.get_from_kwargs('scale', 1)
        self.node_circles = self.get_from_kwargs('node_circles', True)
        self.direction=self.get_from_kwargs('direction','up_down')
        self.level_positions=self.get_from_kwargs('level_positions',None)
        self.build_tree()
        self.get_from_kwargs('bevel', 0)  # remove bevel otherwise it'll interfere with the mesh-objects
        self.root_node = self.nodes[0]


        super().__init__(children=[*self.nodes, *self.lines], name=self.name, location=location, **kwargs)

    def appear(self,
               begin_time=0,
               transition_time=0, mode='linear',
               **kwargs):
        super().appear(begin_time=begin_time, transition_time=0)

        if mode == 'linear':
            dt = transition_time / len(self.nodes)
            for i, node in enumerate(self.nodes):
                node.appear(begin_time=begin_time + (i + 0.5) * dt, transition_time=dt / 2)
            for i, line in enumerate(self.lines):
                line.grow(begin_time=begin_time + (i + 1) * dt, transition_time=dt / 2, modus="from_top")
            t=begin_time+transition_time
        if mode == 'level_wise':
            max_level = self.get_max_level()
            dt_level = transition_time / max_level
            t = begin_time
            for level in range(max_level):
                level_nodes = self.get_nodes_of_level(level)
                dt = dt_level / len(level_nodes)
                dt = np.minimum(1,dt) # at most 1 second per node drawing, if there are only a few nodes per level
                for node in level_nodes:
                    if node.line:
                        node.line.grow(begin_time=t + dt / 2, transition_time=dt / 2)
                        t += dt / 2
                    node.appear(begin_time=t + dt / 2, transition_time=dt / 2)
                    t += dt / 2
                t += 0.1 * transition_time  # level break

        return t

    def appear_set(self, node_set={0}, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        super().appear(begin_time=begin_time, transition_time=0)
        dt = transition_time / len(node_set)

        for c,i in enumerate(node_set):
            self.nodes[i].appear(begin_time=begin_time + (c+ 1) * dt, transition_time=dt / 2)
            if self.nodes[i].line:
                self.nodes[i].line.grow(begin_time=begin_time + (c + 0.5) * dt, transition_time=dt / 2, modus="from_top")

        return begin_time+transition_time

    def appear_words(self,set={},begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        dt = transition_time/len(set)

        for i,w in enumerate(set):
            node = self.node(w)
            node.appear(begin_time+(i+1)*dt,transition_time=dt/2)
            if node.line:
                node.line.grow(begin_time=begin_time+(i+0.5)*dt,transition_time=dt/2,modus="from_top")

        return begin_time+transition_time

    def build_tree(self):
        dh = self.height / self.max_level
        dw = self.width / self.max_level

        for pos in range(self.get_max_level()):
            level_elements = self.get_elements_of_level(pos)
            z0 = self.height - pos * dh
            dx = self.width / len(level_elements)
            scale = np.maximum(1,self.scale * self.level_shrink ** pos)
            dz = self.height/(len(level_elements)+1)
            for i, element in enumerate(level_elements):
                if self.direction=='up_down':
                    x = (i + 0.5) * dx
                    z = z0 + (-1) ** ((i + 1) % 2) * 0.15 * dh
                else:
                    if self.level_positions:
                        x=self.level_positions[pos]*self.width
                    else:
                        x = pos*dw
                    z = self.height-(i+1)*dz

                if element.parent:
                    # establish parent child relation between node_bobjects and store relation inside a dictionary
                    parent_node = self.elements_node_map[element.parent]
                    node_bobject = Node(element, parent=parent_node, location=[x, 0, z], thickness=self.node_thickness,
                                        scale=scale,
                                        name=self.name+"_node_"+str(len(self.nodes)),
                                        node_circles=self.node_circles, **self.kwargs)
                    self.lines.append(node_bobject.line)
                else:
                    node_bobject = Node(element, location=[x, 0, z], thickness=self.node_thickness, scale=scale,
                                        name=self.name+"node_"+str(len(self.nodes)),node_circles=self.node_circles, **self.kwargs)
                self.nodes.append(node_bobject)
                self.elements_node_map[element] = node_bobject

    def get_max_level(self):
        return self.scan_recursively(self.root_element, 0)

    def scan_recursively(self, tree, level):
        if len(tree.children) == 0:
            return level + 1
        maximum = 0
        for child in tree.children:
            maximum = np.maximum(maximum, self.scan_recursively(child, level + 1))
        return maximum

    def get_elements_of_level(self, level):
        return self.get_nodes_of_level_recursively(self.root_element, level, 0)

    def get_nodes_of_level(self, level):
        return self.get_nodes_of_level_recursively(self.root_node, level, 0)

    def get_nodes_of_level_recursively(self, tree, level, current_level):
        if level == current_level:
            return [tree]
        else:
            nodes = []
            for child in tree.children:
                for node in self.get_nodes_of_level_recursively(child, level, current_level + 1):
                    nodes.append(node)
            return nodes

    def add_node(self, node_element, parent, level, shift):
        if isinstance(parent, str):
            parent_node = self.find_node(self.root_node, parent)
            node_element.parent = parent_node.element
            location = self.calculate_location_of(parent_node, shift, vertical_scale=0.75)
            node = Node(node_element, parent=parent_node, location=location, thickness=self.node_thickness,
                        scale=np.maximum(1,self.scale * self.level_shrink ** level),
                        name="added_node_"+str(len(self.nodes)),**self.kwargs)
            self.nodes.append(node)
            self.lines.append(node.line)
            self.add_child(node)
            self.add_child(node.line)

    def calculate_location_of(self, parent, shift, vertical_scale=1):
        max_level = self.get_max_level()
        dh = self.height / max_level*vertical_scale
        z = parent.location.z - dh
        x = parent.location.x
        x += shift
        return Vector([x, 0, z])

    def appear_node(self, word, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        node = self.find_node(self.root_node, word)
        if node:
            self.appear_recursively(node, begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time

    def appear_recursively(self, node, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        node.appear(begin_time=begin_time, transition_time=transition_time)
        if node.line:
            node.line.grow(modus='from_start',begin_time=begin_time, transition_time=transition_time/4)
        for child in node.children:
            self.appear_recursively(child, begin_time=begin_time+transition_time/4, transition_time=3*transition_time/4)

    def disappear_node(self, word, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        node = self.find_node(self.root_node, word)
        if node:
            self.disappear_recursively(node, begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time

    def disappear_recursively(self, node, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        node.disappear(begin_time=begin_time, transition_time=transition_time)
        node.line.disappear(begin_time=begin_time, transition_time=transition_time)
        # self.nodes.remove(node) # do not remove node
        for child in node.children:
            self.disappear_recursively(child, begin_time=begin_time, transition_time=transition_time)

    def find_node(self, tree, word):
        if word.strip()=='':
            return self.root_node
        if tree.element.word == word:
            return tree
        else:
            for child in tree.children:
                result = self.find_node(child, word)
                if result:
                    return result
            return None

    def node(self,word):
        if word.strip()=='':
            return self.root_node
        return self.find_node(self.root_node,word)

    def __str__(self):
        for pre, fill, node in RenderTree(self):
            print("%s%s" % (pre, node.name))
