import numpy as np

from geometry_nodes.geometry_nodes_modifier import SliderModifier
from interface import ibpy
from interface.ibpy import Vector
from objects.bobject import BObject
from objects.cube import Cube
from objects.number_line import NumberLine2
from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs


class BSlider(BObject):
    def __init__(self,label:str,**kwargs):
        """
        possible customizations
        range: [-1,1]
        orientation: "HORIZONTAL"|"VERTICAL"
        dimension: [1,0.25,0.25]
        location: [0,0,0]
        side_segments: 2
        shape: "cubic" | "cylinder"

        """
        self.dimensions=get_from_kwargs(kwargs,"dimensions",[0.25,0.25,1.5])
        self.slider = Cube()
        slider_geometry = SliderModifier(dimensions=self.dimensions,**kwargs)

        self.slider.add_mesh_modifier(type="NODES", node_modifier=slider_geometry)
        self.label = SimpleTexBObject(label,aligned="right",location=slider_geometry.label_position)
        children = [self.slider, self.label]

        numberline = get_from_kwargs(kwargs,"numberline",None)
        if numberline:
            numberline_kwargs = kwargs.copy()
            orientation = get_from_kwargs(kwargs,"orientation","HORIZONTAL")
            if orientation=='HORIZONTAL':
                numberline_kwargs["location"]=[-self.dimensions[2],0,0]
            else:
                numberline_kwargs["location"]=[0,0,-self.dimensions[2]]
            self.numberline = NumberLine2(length=2*self.dimensions[2],radius = 0,axis_label="",tic_label_shift=[2*self.dimensions[0],0,0],**numberline_kwargs)
            children.append(self.numberline)
        else:
            self.numberline = None
        self.slider_value_slot = ibpy.get_geometry_node_from_modifier(slider_geometry,label="SliderValue").outputs[0]
        self.growth_slot = ibpy.get_geometry_node_from_modifier(slider_geometry,label="Growth").outputs[0]

        super().__init__(children=children,name="Slider_"+label,**kwargs)

    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               clear_data=False, silent=False,linked=False, nice_alpha=False,**kwargs):
        super().appear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,silent=silent,
                       linked=linked,nice_alpha=nice_alpha)
        # use only one quarter of the time to write the label
        self.label.write(begin_time=begin_time+0.75*transition_time,transition_time=0.25*transition_time)
        l = max(list(self.dimensions))
        if self.numberline:
            self.numberline.grow(begin_time=begin_time,transition_time=transition_time)
        return ibpy.change_default_value(self.growth_slot,from_value=-1.1*l,to_value=1.1*l,begin_time=begin_time,transition_time=transition_time)

