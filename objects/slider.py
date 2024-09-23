from geometry_nodes.geometry_nodes_modifier import SliderModifier
from interface import ibpy
from objects.bobject import BObject
from objects.cube import Cube
from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME


class BSlider(BObject):
    def __init__(self,label:str,**kwargs):
        """
        possible customizations
        range: [-1,1]
        orientation: "horizontal"|"vertical"
        dimension: [1,0.25,0.25]
        location: [0,0,0]
        shape: "cubic" | "cylinder"

        """

        self.slider = Cube()
        slider_geometry = SliderModifier(**kwargs)
        self.slider.add_mesh_modifier(type="NODES", node_modifier=slider_geometry)
        self.label = SimpleTexBObject(label,aligned="right",location=slider_geometry.label_position)

        self.slider_value_slot = ibpy.get_geometry_node_from_modifier(slider_geometry,label="SliderValue").outputs[0]


        super().__init__(children=[self.slider,self.label],**kwargs)
