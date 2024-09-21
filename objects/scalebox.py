from interface import ibpy
from interface.ibpy import Vector
from objects.bobject import BObject


class ScaleBox(BObject):
    def __init__(self, children=None):
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

        super().__init__(children=children, location=location, rotation_euler=rotation,scale=scale)
        if grandparent:
            ibpy.set_parent(self, grandparent)  # if it is text located inside a display, the display will be parent of
            # the ScaleBox
