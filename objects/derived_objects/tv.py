import numpy as np

from interface import ibpy
from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME


class TV(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['drawing','screen'])
        bobs = BObject.from_file("TV", objects=["TV","Screen"],colors=colors,**kwargs)
        self.tv = bobs[0]
        self.screen = bobs[1]
        super().__init__(children=bobs,name="Television",rotation_euler=rotation,location=location,)

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, clear_data=False, silent=False):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        self.tv.appear(begin_time=begin_time,transition_time=transition_time)
        self.screen.appear(begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def set_movie(self,src,duration):
        ibpy.set_movie_to_material(self.screen,src,duration=duration)

    def start_movie(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        begin_frame=begin_time*FRAME_RATE
        ibpy.set_movie_start(self.screen, begin_frame)
        ibpy.mix_color(self.screen,from_value=0,to_value=1,begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)
        return begin_time+transition_time

    def stop_movie(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.mix_color(self.screen, from_value=1, to_value=0, begin_frame=begin_time * FRAME_RATE,
                       frame_duration=transition_time * FRAME_RATE)
        return begin_time + transition_time