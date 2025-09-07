from anytree import Node, RenderTree, AsciiStyle
from mathutils import Vector

from interface import ibpy
from interface.ibpy import create_mesh, get_geometry_node_from_modifier
from new_stuff.geometry_nodes_modifier import WordTreeModifier
from objects.bobject import BObject
from objects.cube import Cube
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs

class TreeNode(Node):
    """
    Just a container that collects all the data provided inside a node"
    """
    def __init__(self,word="",parent=None):
        self.word = word
        self.parent = parent
        self.x = 0
        self.y = 0
        super().__init__(word,parent=parent)

    def __str__(self):
        return self.word+"("+str(self.x)+","+str(self.y)+")"

    def __repr__(self):
        return str(self)

def fill_levels(node,levels=[],level=0):
    if len(levels)<=level:
        levels.append([])
    levels[level].append(node)
    for child in node.children:
        levels = fill_levels(child,levels,level+1)
    return levels


def create_mesh_from_word_list(word_list, dimensions = [240,108]):
    # create nodes
    root = TreeNode(word=" ")
    nodes = []
    nodes.append(root)

    for i in range(1,4):
        nodes.append(TreeNode(word=word_list[i],parent=root))

    for word in word_list[4:]:
        for node in nodes:
            if word[1:]==node.word:
                parent = node
                break
        nodes.append(TreeNode(word=word,parent=parent))

    print(RenderTree(root,style=AsciiStyle()).by_attr("word"))

    # compute level sizes
    level_sizes = []
    for word in word_list:
        l = len(word.strip())

        if word == r'\varepsilon':
            level_sizes.append(1)
        else:
            while len(level_sizes) <= l:
                level_sizes.append(0)
            level_sizes[l] += 1

    print(level_sizes)

    # compute coordinates
    dx = dimensions[0] / (len(level_sizes))


    levels = fill_levels(root)

    for j in range(7):
        for i, node in enumerate(levels[j]):
            if len(levels[j])==1:
                node.y = dimensions[1]/2
            else:
                dy = dimensions[1] / (len(levels[j]) - 1)
                node.y= int(i*dy)
            node.x = dx * j
            if j==6:
                while True:
                    if len(node.children)==0:
                        break
                    else:
                        child = node.children[0]
                        child.x=int(node.x+dx)
                        child.y=int(node.y)
                    node = child

    # # start with level 4
    # for i,node in enumerate(levels[4]):
    #     node.x = int(4*dx)
    #     node.y = int(i*dy)
    #
    # # repeat the same for level 3 and level 5 and level 6
    # for i, node in enumerate(levels[3]):
    #     node.x = int(3 * dx)
    #     node.y = int(i * dy)
    #
    # for i, node in enumerate(levels[5]):
    #     node.x = int(5 * dx)
    #     node.y = int(i * dy)
    #
    # for i, node in enumerate(levels[6]):
    #     node.x = int(6 * dx)
    #     node.y = int(i * dy)
    #     # fill everything after level six
    #     while True:
    #         if len(node.children)==0:
    #             break
    #         else:
    #             child = node.children[0]
    #             child.x=int(node.x+dx)
    #             child.y=int(node.y)
    #         node = child
    #
    # # work back to level 0
    # # the y position is the average y-position of all children
    # for level in range(2,-1,-1):
    #     for node in levels[level]:
    #         node.x=int(level*dx)
    #         y=0
    #         for child in node.children:
    #             y+=child.y
    #         if len(node.children)==0:
    #             node.y=5
    #         else:
    #             y/=len(node.children)
    #             node.y=int(y)

    print(levels)

    vertices = []
    edges = []
    node_map = {}
    for i,node in enumerate(nodes):
        node_map[node]=i
        vertices.append([node.x,0,node.y])
        if node.parent is not None:
            edges.append((node_map[node.parent],i))

    print(vertices)
    print(edges)
    return vertices,edges,levels


class WordTree(BObject):
    def __init__(self,word_list=[],instances=[], **kwargs):
        """
        A tree that is used for visualizing the isometries CoxB3
        We want to display the words, the permutations and the 3d representation in a tree-like structure

        """

        vertices,edges,levels=create_mesh_from_word_list(word_list)
        tree = BObject(mesh = create_mesh(vertices=vertices,edges=edges,faces=[],name="GroupTreeMesh"),name="GroupTree")

        self.name= get_from_kwargs(kwargs,'name',"WordTree")
        location = get_from_kwargs(kwargs,'location',[0,0,0])
        self.tree_mod = WordTreeModifier(name="CoxB3Modifier",words = word_list, instances =instances,location =location, **kwargs)
        self.instances = instances

        tree.add_mesh_modifier(type="NODES", node_modifier=self.tree_mod)
        super().__init__(obj=tree.ref_obj, name=self.name, **kwargs)
        self.level_counts = [len(level) for level in levels]
        print(self.level_counts)
        self.level = -1

    def get_level_instances(self):
        old_instances = sum(self.level_counts[:self.level])
        new_instances = self.level_counts[self.level]
        return self.instances[old_instances:old_instances + new_instances]


    def appear_next_level(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if self.level==-1:
            super().appear(begin_time=begin_time,transition_time=0)
        self.level = self.level + 1
        old_instances = sum(self.level_counts[:self.level])
        new_instances = self.level_counts[self.level]
        for i in range(old_instances, old_instances + new_instances):
            if i < len(self.instances):
                instance = self.instances[i]
                instance.grow(begin_time=begin_time, transition_time=transition_time)
                # instance.appear_corner_labels(begin_time,transition_time=transition_time)
                instance.appear_face_colors(begin_time=begin_time,transition_time=transition_time)

        for instance in self.get_level_instances():
            instance.grow(begin_time=begin_time,transition_time=transition_time)
        # make instances appear
        max_instance = ibpy.get_geometry_node_from_modifier(self.tree_mod,label="MaxInstance")
        ibpy.change_default_integer(max_instance,from_value=old_instances,to_value=old_instances+new_instances,begin_time=begin_time,transition_time=0)

        # make tree appear
        progress = ibpy.get_geometry_node_from_modifier(self.tree_mod,label="Progress")
        fraction = len(self.level_counts)-1
        ibpy.change_default_value(progress,from_value=(self.level-1)/fraction,to_value=self.level/fraction,begin_time=begin_time,transition_time=transition_time)

        return begin_time+transition_time

    def rotate_instances(self, rotation_euler=Vector(),begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for instance in self.instances:
            instance.rotate(rotation_euler=rotation_euler,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time