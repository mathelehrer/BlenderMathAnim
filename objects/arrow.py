from mathutils import Vector

from interface import ibpy
from interface.ibpy import add_cone, add_cylinder
from objects.cone import Cone
from objects.bobject import BObject
from objects.cylinder import Cylinder
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Arrow(BObject):
    """
    Create a cylinder with a descent mesh:
    """
    def __init__(self,children=None, location=[0, 0, 0], rotation_euler=[0, 0, 0], length=1, radius=0.1, **kwargs):
        if children:
            super().__init__(children=children,**kwargs)
            self.cyl = children[0]
            self.tip = children[1]
            self.start = kwargs['start']
            self.end = kwargs['end'] # it is the end of the stem not the end of the arrow, used for grow function, to grow tip from its base
        else:
            self.kwargs = kwargs
            name = self.get_from_kwargs('name', 'Arrow')
            if 'name' in kwargs:
                kwargs.pop('name')
            self.cyl = Cylinder(length=length / 2,
                                location= Vector([0, 0, -3*length / 4]),
                                radius=radius,
                                name=name + "_stem",
                                **kwargs)
            self.tip = Cone(length=length / 4,
                            radius=radius * 2,
                            location= Vector([0, 0, -length / 4]),
                            name=name + "_tip",
                            **kwargs)
            self.start=Vector([0, 0, -3*length / 4])
            self.end =  Vector([0, 0, -length / 4])
            super().__init__(children=[self.cyl, self.tip], location=location, rotation_euler=rotation_euler, name=name,
                             **kwargs)

    @classmethod
    def from_start_to_end(cls,start =[0,0,0],end=[0,0,1],radius=0.1,**kwargs):
        if 'name' in kwargs:
            name=kwargs['name']
            kwargs.pop('name')
        else:
            name='Arrow'
        if isinstance(start,list):
            start=Vector(start)
        if isinstance(end,list):
            end=Vector(end)

        diff = 0.8*(end-start)
        length=diff.length

        cyl = Cylinder.from_start_to_end(start=start,end=start+diff ,
                            radius=radius,
                            name=name + "_stem",
                            **kwargs)
        rotation=cyl.rotation_quaternion
        tip = Cone(length=length / 8, # one fifth of the full length, since length is only 4/5 of the full length
                        radius=radius * 2,
                        location=start+diff*9/8,
                        name=name + "_tip",
                        rotation_quaternion=rotation,
                        **kwargs)
        kwargs['start'] = start # transport start and final value to object
        kwargs['end'] = start+diff
        return Arrow(children=[cyl,tip],name=name,**kwargs)

    def rescale(self,s=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        """
        rescale the length of the arrow by some scalar s
        :param s:
        :param begin_time:
        :param transition_time:
        :return:
        """
        ibpy.rescale(self.cyl,re_scale=[1,1,s],begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)
        ibpy.rescale(self.tip,re_scale=[1,1,s],begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)


    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center',order="stem_first"):
        super().appear(begin_time=begin_time, transition_time=0)

        if modus == 'from_bottom' or modus=='from_start':
            if order=='stem_first':
                self.cyl.grow(scale=self.cyl.intrinsic_scale, begin_time=begin_time, transition_time=3*transition_time / 4, modus=modus)
                self.tip.grow(scale=self.tip.intrinsic_scale,pivot=self.end, begin_time=begin_time + transition_time / 2, transition_time=transition_time / 2)
            else:
                self.cyl.grow(scale=self.cyl.intrinsic_scale, begin_time=begin_time + transition_time / 4,
                              transition_time=3*transition_time / 4, modus=modus)
                self.tip.grow(scale=self.tip.intrinsic_scale,pivot=self.end, begin_time=begin_time,
                              transition_time=transition_time / 2)
        elif modus == 'from_top' or modus == 'from_end':
            if order=='stem_first':
                self.cyl.grow(scale=self.cyl.intrinsic_scale, begin_time=begin_time, transition_time=transition_time / 2, modus=modus)
                self.tip.grow(scale=self.tip.intrinsic_scale,pivot=self.end, begin_time=begin_time + transition_time / 2, transition_time=transition_time / 2)
            else:
                self.cyl.grow(scale=self.cyl.intrinsic_scale, begin_time=begin_time+ transition_time / 2,
                              transition_time=transition_time / 2, modus=modus)
                self.tip.grow(scale=self.tip.intrinsic_scale,pivot=self.end, begin_time=begin_time ,
                              transition_time=transition_time / 2)
        else:
            self.cyl.grow(scale=self.cyl.intrinsic_scale, begin_time=begin_time, transition_time=transition_time, modus=modus)
            self.tip.grow(scale=self.tip.intrinsic_scale,pivot=self.end, begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time