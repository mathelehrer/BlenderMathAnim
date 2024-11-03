from interface import ibpy
from interface.ibpy import Vector
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME


class TextScaleBox(BObject):
    def __init__(self, children=None,name="ScaleBox"):
        location = Vector()
        rotation = Vector()
        grandparent = None
        if children is None:
            children = []
        if len(children) > 0:
            parent = ibpy.get_parent(children[0])
            grandparent = ibpy.get_parent(parent)

            location = ibpy.get_location(parent)
            rotation = ibpy.get_rotation(parent)
            scale = ibpy.get_scale(parent)

        # find average location for nice scalings
        average = Vector()
        for child in children:
            average += ibpy.get_location(child)
        average /= len(children)
        #location += average

        super().__init__(children=children, location=location, rotation_euler=rotation,scale=scale,name=name)
        if grandparent:
            ibpy.set_parent(self, grandparent)  # if it is text located inside a display, the display will be parent of
            # the ScaleBox

    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               clear_data=False, silent=False,linked=False, nice_alpha=False,**kwargs):
        super().appear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,
                       clear_data=clear_data,silent=silent,linked=linked,nice_alpha=nice_alpha,**kwargs)
        for child in self.b_children:
            child.appear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,clear_data=clear_data,silent=silent,linked=linked,nice_alpha=nice_alpha)