
from interface.ibpy import Vector
from objects.bobject import BObject
from objects.cube import Cube


class BoundingBox(BObject):
    """A :class:`Cube` that fills a given axis-aligned bounding box.

    Used to wrap arbitrary geometry's :meth:`get_text_bounding_box`-like
    output back into a visualisable solid -- handy for debugging layout.
    """

    def __init__(self,bounding_box):
        """Build a cube that exactly fills the given AABB.

        Args:
            bounding_box: ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
        """
        xmin,ymin,zmin,xmax,ymax,zmax = bounding_box
        center = Vector([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
        scale=0.5*Vector([xmax-xmin,ymax-ymin,zmax-zmin])
        cube = Cube(location=center,scale=scale)
        super().__init__(cube)