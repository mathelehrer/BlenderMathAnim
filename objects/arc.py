from mathutils import Vector
from objects.curve import Curve


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

        if not isinstance(start, Vector):
            start = Vector(start)
        if not isinstance(end, Vector):
            end = Vector(end)
        if not isinstance(center, Vector):
            center = Vector(center)

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
