from collections import OrderedDict

import numpy as np

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import create_from_xml
from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier
from objects.cube import Cube
from perform.scene import Scene
from utils.kwargs import get_from_kwargs
from utils.utils import print_time_report

pi = np.pi
tau = 2*pi

class ExampleModifier(GeometryNodesModifier):
    def __init__(self, **kwargs):
        super().__init__(get_from_kwargs(kwargs, 'name', "ExampleModifier"),
                         group_input=False, group_output=False, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        create_from_xml(tree, "Example_nodes", **kwargs)


class Examples(Scene):
    """
    This scene contains examples for complex animations for reference
    """
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('geometry_from_xml', {'duration': 10}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def geometry_from_xml(self):
        t0  = 0

        # the original blend file is stored in files/xml_generators
        cube = Cube()
        modifier = ExampleModifier()
        cube.add_mesh_modifier(type="NODES", node_modifier=modifier)

        slider_node=get_geometry_node_from_modifier(modifier,"Slider")
        t0  = 0.5 + ibpy.change_default_value(slider_node,from_value=0,to_value=1,begin_time=t0,transition_time=10)


        self.t0 = t0




if __name__ == '__main__':
    try:
        example = Examples()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        if len(dictionary) == 1:
            selected_scene = dictionary[0]
        else:
            choice = input("Choose scene:")
            if len(choice) == 0:
                choice = 0
            print("Your choice: ", choice)
            selected_scene = dictionary[int(choice)]

        example.create(name=selected_scene, resolution=[1920, 1080], start_at_zero=True)

        # example.render(debug=True)
        # doesn't work
        # example.final_render(name=selected_scene,debug=False)
    except:
        print_time_report()
        raise ()
