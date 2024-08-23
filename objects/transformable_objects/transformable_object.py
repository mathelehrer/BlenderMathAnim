import numpy as np

from appearance.textures import make_transformations_and_complex_material
from interface import ibpy
from objects.bobject import BObject
from utils.constants import FRAME_RATE, OBJECT_APPEARANCE_TIME


class TransBObject(BObject):
    def __init__(self,transformations = None, **kwargs):
        super().__init__(**kwargs)

        # only call this after the mesh has been created
        if transformations:
            self.mixer_dialers = make_transformations_and_complex_material(self, transformations, name=self.ref_obj.name)
            self.current_transformation = 0  # this corresponds to the basis shape key
            self.current_dialer_index = 0

    def next_shape_key(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        appear_frame = begin_time * FRAME_RATE
        ibpy.morph_to_next_shape(self.ref_obj, self.current_transformation, appear_frame,
                                 np.maximum(1, transition_time * FRAME_RATE))
        self.current_transformation += 1

    def next_color_map(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        appear_frame = begin_time * FRAME_RATE
        dialer_index = self.current_dialer_index  # the ith dialer activates the colors for the (i+1)th profile
        if len(self.mixer_dialers) > 0 and len(self.mixer_dialers) > dialer_index:
            current_dialer = self.mixer_dialers[dialer_index]
            current_dialer.default_value = 0
            ibpy.insert_keyframe(current_dialer, 'default_value', appear_frame)
            current_dialer.default_value = 1
            ibpy.insert_keyframe(current_dialer, 'default_value',
                                 appear_frame + np.maximum(1, transition_time * FRAME_RATE))
        self.current_dialer_index += 1

    def next_shape(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        """
        transition to the next transformation of the annulus
        :param begin_time:
        :param transition_time:
        :return:
        """
        self.next_shape_key(begin_time=begin_time, transition_time=transition_time)
        self.next_color_map(begin_time=begin_time, transition_time=transition_time)
        print("Next transformation at " + str(begin_time) + " with transition time " + str(transition_time))

    def previous_shape(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        """
        transition to the next transformation of the annulus
        :param begin_time:
        :param transition_time:
        :return:
        """
        appear_frame = begin_time * FRAME_RATE
        dialer_index = self.current_transformation - 1
        if len(self.mixer_dialers) > 0 and len(self.mixer_dialers) > dialer_index:
            current_dialer = self.mixer_dialers[dialer_index]
            ibpy.insert_keyframe(current_dialer, 'default_value', appear_frame)
            current_dialer.default_value = 0
            ibpy.insert_keyframe(current_dialer, 'default_value', appear_frame + transition_time * FRAME_RATE)
        ibpy.morph_to_previous_shape(self.ref_obj, self.current_transformation, appear_frame,
                                     transition_time * FRAME_RATE)
        self.current_transformation -= 1
        print("Next transformation at " + str(begin_time) + " with transition time " + str(transition_time))
