import numpy as np

from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME


class Pin(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name=self.get_from_kwargs('name','Pin')
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['drawing'])
        emission = self.get_from_kwargs('emission',0)
        colors[0]='glass_'+colors[0]
        colors.append('metal_0.5')

        bobs = BObject.from_file("Pin", objects=["Head", "Needle"],colors=colors,name=self.name,emission=emission)
        self.head = bobs[0]
        self.needle = bobs[1]
        super().__init__(children=[self.head, self.needle], name=self.name, rotation_euler=rotation, location=location,**kwargs)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time,**kwargs)
        self.head.appear(begin_time=begin_time, transition_time=transition_time,**kwargs)
        self.needle.appear(begin_time=begin_time, transition_time=transition_time,**kwargs)

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        super().disappear(begin_time=begin_time,transition_time=transition_time,**kwargs)
        self.head.disappear(begin_time=begin_time, transition_time=transition_time,**kwargs)
        self.needle.disappear(begin_time=begin_time, transition_time=transition_time,**kwargs)