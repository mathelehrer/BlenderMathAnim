import numpy as np

from interface import ibpy
from objects.curve import Curve
from utils.utils import to_vector


class Arc(Curve):
    """
    Create an arc from the start to the end around a center
    Always the shorter arc is taken


    known issues:
    * it is important that each arc has a unique name string,
      otherwise it will not be found and animated correctly

    """

    def __init__(self, start, end, center=[0, 0, 0], **kwargs):
        """
        """

        self.kwargs = kwargs
        name = self.get_from_kwargs('name', 'Arc')
        num_points = self.get_from_kwargs('num_points', 10)
        # mapping

        start = to_vector(start)
        end = to_vector(end)
        center = to_vector(center)

        # deal with the antipodal point, which is singular
        if (end+start).length<0.0001:
            end.x*=1.01
            end.y*=0.99
        d = end - start

        r2 = (end - center).length
        r1 = (start - center).length

        super().__init__([
            lambda t: center + (r1 + t * (r2 - r1)) / (start + t * d - center).length* (start + t * d - center),
        ], domain=[0, 1], name=name, num_points=num_points, **kwargs)


class Arc2(Curve):
    """
    Create an arc from the start to the end around a center
    Always the shorter arc is taken


    known issues:
    * it is important that each arc has a unique name string,
      otherwise it will not be found and animated correctly

    """

    def __init__(self, center=[0,0,0],start_point=[0,0,1], start_angle=0,
                 end_angle=np.pi, normal=[0,0,1], pivot = None,**kwargs):
        """
        """

        self.kwargs = kwargs
        self.arc_start=start_point

        name = self.get_from_kwargs('name', 'Arc')
        num_points = self.get_from_kwargs('num_points', 50)
        # mapping

        start_point= to_vector(start_point)
        center = to_vector(center)
        normal=to_vector(normal)
        if pivot is None:
            pivot = center

        x=start_point-center
        r = x.length
        y=normal.cross(x)
        self.arc_end = center +x*np.cos(end_angle)+y*np.sin(end_angle)

        super().__init__([
            lambda t: center +x*np.cos(t)+y*np.sin(t)],
            domain=[start_angle, (1+1/num_points)*end_angle], name=name,
            num_points=num_points, **kwargs)
        ibpy.set_pivot(self,pivot)
