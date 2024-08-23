import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.curve import Curve
from objects.function import Function
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.utils import to_vector


class Ellipse(Curve):
    """
    Create an ellipse, based on a bezier curve
    """

    def __init__(self,
                 center=[0, 0],
                 radius=1,
                 ratio=2,
                 num_points=20,
                 color='example',
                 **kwargs):
        """

        :param center:
        :param num_points:
        :param ratio:
        :param color:
        :param mode:
        :param kwargs:
        """

        self.ratio = ratio
        self.center = center
        self.radius = radius

        if 'type' in kwargs:
            self.mode = kwargs['type']
            kwargs.pop('type')
        else:
            self.mode = 'XY'

        if 'domain' in kwargs:
            domain = kwargs['domain']
            kwargs.pop('domain')
        else:
            domain = [0, 2 * np.pi]

        super().__init__(lambda x: self.parametric_function(x), domain,
                         num_points=num_points, color=color, mode='PARAMETRIC', **kwargs)

    def parametric_function(self, x):
        if self.mode == 'XY':
            X = self.center[0] + self.ratio*self.radius * np.cos(x)
            Y = self.center[1] + self.radius * np.sin(x)
            Z = 0
        elif self.mode == 'XZ':
            X = self.center[0] + self.radius * np.cos(x)
            Z = self.center[2] + self.radius * np.sin(x)
            Y = 0
        return [X, Y, Z]

    def get_location_at_angle(self, phi):
        return Vector(self.parametric_function(phi))
