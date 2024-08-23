import numpy as np

from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME


class WallWithDoor(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['drawing'])
        bobs = BObject.from_file("WallWithDoor", objects=["Wall", "Door"],colors=colors)
        self.wall = bobs[0]
        self.wall.ref_obj.scale = [3.8,0.2,1.8]
        self.wall.ref_obj.location=[0,0,1.9]
        self.door = bobs[1]
        self.door.ref_obj.scale = [1.056,0.1,1.056]
        self.door.ref_obj.location=[-1.05,-0.1,1.23]
        super().__init__(children=[self.wall,self.door],name="WallWithDoor",rotation_euler=rotation,location=location)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        self.door.appear(begin_time=begin_time,transition_time=transition_time)
        self.wall.appear(begin_time=begin_time,transition_time=transition_time)

    def open_door(self,begin_time=0,transition_time=OBJECT_APPEARANCE_TIME):
        self.door.rotate(begin_time=begin_time,transition_time=transition_time,rotation_euler=[0,0,np.pi/2])