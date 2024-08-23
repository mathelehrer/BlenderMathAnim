import numpy as np
from mathutils import Vector

from interface import ibpy
from interface.ibpy import add_cone, add_cylinder
from objects.circle import BezierCircle
from objects.cone import Cone
from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.derived_objects.p_arrow import PArrow
from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Clock(BObject):
    """
    Create a clock with Roman numerals:
    """
    def __init__(self,radius=6, location=[0, 0, 0], rotation_euler=[0, 0, 0],colors=['drawing','important'], **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name','Clock')
        r = 6
        self.circle = BezierCircle(radius=r, rotation_euler=[np.pi / 2, 0, 0], thickness=10, color=colors[0], resolution=100)
        self.arrow = PArrow(start=Vector(), end=0.825 * r * Vector([0, 0, 1]), color=colors[1], thickness=5)

        number_strings = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
        self.numbers = []
        angle = np.pi / 2
        dt = 1 / 12
        for number_string in number_strings:
            angle -= np.pi / 6
            number = SimpleTexBObject(r"\text{" + number_string + "}", aligned="center", color='drawing',
                                      location=[0.875 * r * np.cos(angle), 0, 0.9 * r * np.sin(angle)],
                                      text_size='huge')
            self.numbers.append(number)


        super().__init__(children=[self.circle, self.arrow,*self.numbers], location=location, rotation_euler=rotation_euler, name=name,
                             **kwargs)

    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, clear_data=False, silent=False):
        super().appear(begin_time=begin_time,transition_time=transition_time)

        self.circle.appear(begin_time=begin_time,transition_time= transition_time)
        self.arrow.grow(begin_time=begin_time,transition_time=transition_time)

        dt = transition_time/2/12
        t0=begin_time
        for number in self.numbers:
            number.write(begin_time=t0,transition_time=transition_time/2)
            t0+=dt

        return begin_time+transition_time

    def progress(self,begin_time=0,number_of_minutes=0,number_of_seconds=0):
        self.arrow.rotate(rotation_euler=[0,2*np.pi*number_of_minutes+2*np.pi/60*number_of_seconds,0],begin_time=begin_time,transition_time=number_of_minutes*60+number_of_seconds/2)
        ibpy.set_linear_fcurves(self.arrow)

    def reset(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        self.arrow.rotate(rotation_euler=[0,0,0],begin_time=begin_time,transition_time=transition_time)