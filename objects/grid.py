from mathutils import Vector

from objects.bobject import BObject
from objects.cylinder import Cylinder
from utils.constants import DEFAULT_ANIMATION_TIME


class DefaultGrid(BObject):
    """A fixed-size 24x13 reference grid drawn in the world XZ plane,
    bounded by a 4-cylinder frame. Used as a backdrop for 2D scenes."""

    def __init__(self):
        """Construct the default grid (no parameters).

        Geometry is hard-coded: frame at ``x in [-12, 12]``, ``z in [-6.5, 6.5]``,
        with one horizontal cylinder per integer ``z`` in ``[-6, 6]`` and one
        vertical cylinder per integer ``x`` in ``[-11, 11]``.
        """
        xmin = -12
        xmax = 12
        zmin = -6.5
        zmax = 6.5

        top = Cylinder.from_start_to_end(start=Vector([xmin, 0, zmax]), end=Vector([xmax, 0, zmax]))
        bottom = Cylinder.from_start_to_end(start=Vector([xmax, 0, zmin]), end=Vector([xmin, 0, zmin]))
        left = Cylinder.from_start_to_end(start=Vector([xmin, 0, zmin]), end=Vector([xmin, 0, zmax]))
        right = Cylinder.from_start_to_end(start=Vector([xmax, 0, zmax]), end=Vector([xmax, 0, zmin]))

        self.frame = [top, right, bottom, left]

        self.horizontal_grid = []
        self.vertical_grid = []

        for z in range(-6, 7):
            self.horizontal_grid.append(
                Cylinder.from_start_to_end(start=Vector([xmin, 0, z]), end=Vector([xmax, 0, z]), thickness=0.1,
                                           color='text'))
        for x in range(-11, 12):
            self.vertical_grid.append(
                Cylinder.from_start_to_end(start=Vector([x, 0, zmin]), end=Vector([x, 0, zmax]), thickness=0.1,
                                           color='text'))

    def show(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time / 4
        for i, line in enumerate(self.frame):
            line.grow(begin_time=begin_time + i * dt, transition_time=dt)

        dt = transition_time / len(self.horizontal_grid) / 2
        for i, line in enumerate(self.horizontal_grid):
            line.grow(begin_time=begin_time + i * dt, transition_time=transition_time / 2)

        dt = transition_time / len(self.vertical_grid) / 2
        for i, line in enumerate(self.vertical_grid):
            line.grow(begin_time=begin_time + i * dt, transition_time=transition_time / 2)
