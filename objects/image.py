
from interface import ibpy
from interface.ibpy import add_cone
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import to_vector


class ReferenceImage(BObject):
    """
    Create a cube:

    """

    def __init__(self,name,**kwargs):
        self.kwargs =kwargs

        img = ibpy.add_reference_image(name)

        if 'name' in kwargs:
            pass
        else:
            kwargs['name']='ReferenceImage'

        location = self.get_from_kwargs('location',[0,0,0])
        location=to_vector(location)
        origin = self.get_from_kwargs('origin', location)

        shift = to_vector(location)-to_vector(origin)
        super().__init__(obj=img, location=shift,**kwargs)

        apply_location = self.get_from_kwargs('apply_location', False)
        if shift.length>0:
            apply_location=True
        apply_scale = self.get_from_kwargs('apply_scale', False)
        apply_rotation = self.get_from_kwargs('apply_rotation', False)
        ibpy.apply(self, location=apply_location, rotation=apply_rotation, scale=apply_scale)
        self.ref_obj.location=location-shift

def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
             initial_scale=0,pivot=None):
        super().appear(begin_time=begin_time,transition_time=0)
        """
        grow an object from 0 to
        :param scale: the final scale
        :param begin_time: starting time
        :param transition_time: duration
        :param modus: can be 'from_center', 'from_left', 'from_right', 'from_top', 'from_bottom', 'from_front', 'from_back'
        :return:
        """
        if scale is None:
            scale = self.intrinsic_scale
        if not pivot:
            pivot = self.ref_obj.location
        ibpy.grow_from(self,pivot, begin_time * FRAME_RATE, transition_time * FRAME_RATE)