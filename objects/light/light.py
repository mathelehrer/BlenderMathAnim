from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class LightProbe(BObject):
    """
    A light probe to highlight reflections in Eevee
    """

    def __init__(self, **kwargs):
        self.kwargs=kwargs
        name= self.get_from_kwargs('name',"Probe")
        self.rotation_euler = self.get_from_kwargs('rotation_euler',[0,0,0])
        probe = ibpy.add_light_probe(**kwargs)

        super().__init__(obj=probe, name=name,rotation_euler=self.rotation_euler, **self.kwargs)


class SpotLight(BObject):
    """
    Spotlight
    target to bob
    specify location, radius,scale,energy
    """
    def __init__(self,target = None, **kwargs):
        self.kwargs = kwargs
        name=self.get_from_kwargs('name','SpotLight')
        light = ibpy.add_spot_light(**kwargs)

        # default arguments that are used in the construction
        # they are removed that they are not used again in the construction of the wrapper
        self.location=self.get_from_kwargs('location',Vector())
        self.radius=self.get_from_kwargs('radius',1)
        self.scale=self.get_from_kwargs('scale',[1]*3)
        self.energy=self.get_from_kwargs('energy',10)

        super().__init__(obj=light,name=name,**self.kwargs)

        if target:
            ibpy.set_track(self,target)

    def appear(self,
               begin_time=0,
               transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):
        self.on(begin_time=begin_time,transition_time=transition_time)

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        self.off(begin_time=begin_time,transition_time=transition_time)

    def on(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.switch_on(self, begin_frame=begin_time * FRAME_RATE, frame_duration=transition_time * FRAME_RATE)

    def off(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.switch_off(self,begin_frame = begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)


class AreaLight(BObject):
    """
    Arealight
    target to bob
    specify location, radius,scale,energy
    """
    def __init__(self,target = None, **kwargs):
        self.kwargs = kwargs
        # default arguments that are used in the construction
        # they are removed that they are not used again in the construction of the wrapper
        # self.location = self.get_from_kwargs('location', Vector())
        # self.radius = self.get_from_kwargs('radius', 1)
        # self.scale = self.get_from_kwargs('scale', [1] * 3)
        self.energy = self.get_from_kwargs('energy', 10)
        self.color = self.get_from_kwargs('color', 'text')
        self.shape =self.get_from_kwargs('shape','SQUARE')
        self.size=self.get_from_kwargs('size',1)
        self.size_y=self.get_from_kwargs('size_y',1)
        self.diffuse_factor=self.get_from_kwargs('diffuse_factor',1)
        self.specular_factor=self.get_from_kwargs('specular_factor',1)
        self.volume_factor=self.get_from_kwargs('volume_factor',1)

        name=self.get_from_kwargs('name','AreaLight')
        light = ibpy.add_area_light(energy=self.energy,color=self.color,shape=self.shape,
                                    size=self.size,size_y=self.size_y,diffuse_factor=self.diffuse_factor,
                                    specular_factor=self.specular_factor,volume_factor=self.volume_factor)


        super().__init__(obj=light,name=name,**self.kwargs)

        if target:
            ibpy.set_track(self,target)

    def appear(self,
               begin_time=0,
               transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):
        return self.on(begin_time=begin_time,transition_time=transition_time)

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        return self.off(begin_time=begin_time,transition_time=transition_time)

    def on(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.switch_on(self, begin_frame=begin_time * FRAME_RATE, frame_duration=transition_time * FRAME_RATE)
        return begin_time+transition_time

    def off(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.switch_off(self,begin_frame = begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)
        return begin_time + transition_time