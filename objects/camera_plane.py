from interface import ibpy
from objects.empties import EmptyAxes


class CameraPlane(EmptyAxes):
    def __init__(self,**kwargs):
        super().__init__(name="CameraPlane",**kwargs)

    def add(self,bob):
        ibpy.set_parent(bob,self)