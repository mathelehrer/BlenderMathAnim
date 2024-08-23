import numpy as np

from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME


class InfoPanel(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['drawing'])
        bobs = BObject.from_file("InfoPanel_wlog", objects=["InfoPanel", "Text"],colors=colors)
        self.info = bobs[0]
        self.info.ref_obj.scale = [0.2,0.2,1.5]
        self.info.ref_obj.location=[-2,0,1.5]
        super().__init__(children=[self.info],name="InfoPanel",rotation_euler=rotation,location=location)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        self.info.appear(begin_time=begin_time,transition_time=transition_time)