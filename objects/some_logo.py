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
from objects.torus import Torus
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import to_vector


class SoME(BObject):
    """
    Create the summer of Math exposition logo:
    """
    def __init__(self, location=[0, 0, 0], rotation_euler=[0, 0, 0], **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name','SoMELogo')
        r = 6

        self.torus = Torus(rotation_euler=[np.pi / 2, 0, 0],
                      name='Circle', major_radius=1.92,
                      minor_radius=0.075, major_segments=96,
                      minor_segments=24, color='gradient',
                      colors=['some_logo_blue', 'some_logo_green'],
                      coordinate_type='Object', coordinate='Y',
                      ramp_positions=[0, 0.01], emission=0.1)
        self.some = SimpleTexBObject(r"\text{SoME}", aligned='center', typeface='times',
                                scale=5.5, location=[-0.011, 0, 0.053],
                                emission=0.1, thickness=1.5, bevel=1,
                                     color='plastic_text')

        top = Vector([0.48122, 0, 1.866])
        left = Vector([-1.88, 0, 0])
        right = Vector([1.88, 0, 0])
        bottom = Vector([-0.538, 0, -1.843])

        self.cylinders = [
            Cylinder.from_start_to_end(start=left, end=right, thickness=0.5, color='some_logo_green'),
            Cylinder.from_start_to_end(start=left, end=top, thickness=0.5, color='some_logo_green'),
            Cylinder.from_start_to_end(start=right, end=top, thickness=0.5, color='some_logo_green'),
            Cylinder.from_start_to_end(start=right, end=bottom, thickness=0.5, color='dashed',
                                       colors=['some_logo_blue'],
                                       dash_scale=1.8, phase_offset=1.7),
            Cylinder.from_start_to_end(start=left, end=bottom, thickness=0.5, color='dashed', colors=['some_logo_blue'],
                                       dash_scale=1.4, phase_offset=2.1)
        ]


        super().__init__(children=[self.torus,self.some,*self.cylinders], location=location, rotation_euler=rotation_euler, name=name,
                             **kwargs)

    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, clear_data=False, silent=False):
        super().appear(begin_time=begin_time,transition_time=transition_time)

        dt1 = 2*transition_time/3
        dt2 = 1*transition_time/3
        self.some.write(begin_time=begin_time,transition_time=dt1)
        self.torus.appear(begin_time=begin_time+dt2,transition_time=dt2)
        for cyl in self.cylinders:
            cyl.grow(begin_time=begin_time+2*dt2,transition_time=dt2)

        return begin_time+transition_time

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        super().disappear(begin_time=begin_time, transition_time=transition_time)
        for cyl in self.cylinders:
            cyl.shrink(begin_time=begin_time,transition_time=transition_time)
