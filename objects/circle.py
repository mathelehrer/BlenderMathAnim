import numpy as np
from mathutils import Vector

from interface import ibpy
from interface.ibpy import add_cone, add_circle
from objects.bobject import BObject
from objects.curve import Curve
from objects.function import Function
from objects.geometry.sphere import Sphere
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import to_vector


class Circle(Function):
    """
    Create a circle, based on a bezier curve

    which is a curve and it has to be associated with an
    existing coordinate system:

    """

    def __init__(self, coordinate_system=None,
                 center=[0, 0],
                 radius=1,
                 num_points=100,
                 color='example',
                 **kwargs):
        """

        :param coordinate_system:
        :param center:
        :param num_points:
        :param color:
        :param mode:
        :param kwargs:
        """
        super().__init__(lambda x: Vector([center[0] + radius * np.cos(x), center[1] + radius * np.sin(x)]), coordinate_system,
                         [0, 2.1 * np.pi], num_points=num_points, color=color, mode='PARAMETRIC', **kwargs)


class Circle2(Curve):
    """
    Create a circle, based on a bezier curve

    which is a curve
    """

    def __init__(self,
                 center=[0, 0],
                 radius=1,
                 num_points=100,
                 color='example',
                 **kwargs):
        """

        :param center:
        :param num_points:
        :param color:
        :param mode:
        :param kwargs:
        """

        self.center = center
        self.radius = radius
        self.kwargs = kwargs

        self.mode = self.get_from_kwargs('mode','XY')
        domain = self.get_from_kwargs('domain',[0,2*np.pi*(1+1/num_points)])

        super().__init__(lambda x: self.parametric_function(x), domain,
                         num_points=num_points, color=color, mode='PARAMETRIC', **self.kwargs)

    def parametric_function(self, x):
        if self.mode == 'XY':
            X = self.center[0] + self.radius * np.cos(x)
            Y = self.center[1] + self.radius * np.sin(x)
            Z = 0
        elif self.mode == 'XZ':
            X = self.center[0] + self.radius * np.cos(x)
            Z = self.center[2] + self.radius * np.sin(x)
            Y = 0
        return [X, Y, Z]

    def get_location_at_angle(self, phi):
        return Vector(self.parametric_function(phi))


class BezierCircle(BObject):
    def __init__(self,radius=1,location=Vector(),rotation_euler=Vector(),**kwargs):
        self.kwargs = kwargs
        self.start = [0, 0, -0.5]
        name=self.get_from_kwargs('name','BezierCircle')
        color = self.get_from_kwargs('color','drawing')
        brighter=self.get_from_kwargs('brighter',0)
        thickness =self.get_from_kwargs('thickness',0)
        resolution=self.get_from_kwargs('resolution',12)
        circle = add_circle(radius=radius, location=location)
        super().__init__(obj=circle,color=color,rotation_euler=rotation_euler,name=name,brighter=brighter,**kwargs)
        if thickness>0:
            self.ref_obj.data.bevel_depth=thickness/100
        self.ref_obj.data.resolution_u=resolution


class RightAngle(BObject):
    def __init__(self,radius=1,location=Vector(), **kwargs):
        self.kwargs = kwargs
        mode = self.get_from_kwargs('mode', 'XY')
        name= self.get_from_kwargs('name','RightAngle')
        rotation_euler = self.get_from_kwargs('rotation_euler',[0,0,0])
        self.arc = CircleArc(center=Vector(), radius=radius, start_angle=0,
                             end_angle=np.pi / 2, **self.kwargs, mode=mode,name=name+"_Arc")
        dot_pos = 0.4
        if mode == 'XZ':
            dot_location =  Vector([radius *dot_pos, 0, radius *dot_pos])
        else:
            dot_location =  Vector([radius *dot_pos, radius *dot_pos, 0])

        thickness = self.get_from_kwargs('thickness', 0.2) # keep it after the construction of the circle arc since it also needs the parameter thickness
        self.dot = Sphere(thickness/5, location=dot_location,name=name+"_dot" ,**kwargs)
        super().__init__(children=[self.arc, self.dot],location=location,rotation_euler=rotation_euler, name=name,**kwargs)

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, clear_data=False, silent=False,**kwargs):
        super().appear(begin_time=begin_time,transition_time=0)
        self.arc.grow(begin_time=begin_time, transition_time=transition_time)
        self.dot.grow(begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time


class CircleArc(Circle2):
    def __init__(self, center=Vector(), radius=1, start_angle=0, end_angle=np.pi * 2, **kwargs):
        self.kwargs = kwargs
        self.start = start_angle
        self.end = end_angle
        self.radius = radius
        self.center = to_vector(center)
        self.factor = 1
        self.factor0 = 0
        self.label_offset = Vector()
        if 'name' in kwargs:
            name = kwargs['name']
            kwargs.pop('name')
        else:
            name = 'Arc'
        super().__init__(center=self.center, radius=radius, domain=[start_angle, end_angle], name=name, **kwargs)

    def grow(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        appear_frame = begin_time * FRAME_RATE
        ibpy.grow_curve(self, appear_frame, transition_time, start_factor=0, end_factor=1)

    def get_center(self):
        location = super().get_location_at_angle(0.5 * (self.start + self.factor0 * self.end + self.factor * self.end))
        return self.center + 0.7 * (location - self.center)

    def extend_to(self, factor, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.factor = factor
        ibpy.set_curve_range(self, factor, begin_time=begin_time, transition_time=transition_time)
        if self.label:
            self.label.move_to(target_location=self.get_center(), begin_time=begin_time,
                               transition_time=transition_time)

    def extend_from_to(self, factor1, factor2, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.factor = factor2
        self.factor0 = factor1
        ibpy.set_curve_full_range(self, factor1, factor2, begin_time=begin_time, transition_time=transition_time)
        if self.label:
            self.label.move_to(target_location=self.get_center() + self.label_offset, begin_time=begin_time,
                               transition_time=transition_time)
