import bmesh

from interface import ibpy
from interface.ibpy import add_cone
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Cone(BObject):
    """
    Create a cone with a descent mesh:

    """

    def __init__(self, location=[0, 0, 0], length=1, radius=0.1, **kwargs):
        """Create a cone aligned with the local Z axis.

        The base sits at the bottom (``-Z``) and the tip at the top (``+Z``).
        Length is encoded as ``scale.z = length`` and the base radius as
        ``scale.x = scale.y = radius``.

        Args:
            location: World location of the cone's origin. Defaults to ``[0, 0, 0]``.
            length: Height of the cone (base to tip) along the local Z axis.
            radius: Base radius of the cone.
            **kwargs: Forwarded to :class:`BObject`. Notable keys:
                * ``name`` (str): Defaults to ``'Cone'``.
                * ``loop_cuts`` (int): Number of subdivisions along the height
                  (scaled by ``length``). Useful for smooth deformations.
                * ``color``, ``smooth``, ``bevel``, ``rotation_euler``, etc.
                  forwarded to :class:`BObject`.
        """
        self.kwargs =kwargs
        self.start=[0,0,-0.5]
        cone = add_cone()
        loop_cuts = self.get_from_kwargs('loop_cuts', 0)
        if loop_cuts > 0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(cone.data)  # Fills it in using the cylinder
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=int(loop_cuts*length), use_grid_fill=True)
            bm.to_mesh(cone.data)
            bm.free()
        if 'name' in kwargs:
            pass
        else:
            kwargs['name']='Cone'
        super().__init__(obj=cone, location=location, scale=[radius, radius, length], **kwargs)
        ibpy.un_link(self.ref_obj,
                     collection="Collection")  # unlink the object, since it is linked automatically from the mesh creation process

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, initial_scale=0,pivot=None):
        super().appear(begin_time=begin_time,transition_time=0)
        """
        grow an object from 0 to
        :param scale: the final scale
        :param begin_time: starting time
        :param transition_time: duration
        :param modus: can be 'from_center', 'from_left', 'from_right', 'from_top', 'from_bottom', 'from_front', 'from_back'
        :return:
        """
        if scale is None:
            scale = self.intrinsic_scale
        if not pivot:
            pivot = self.ref_obj.location
        ibpy.grow_from(self,pivot, begin_time * FRAME_RATE, transition_time * FRAME_RATE)


    def shrink(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, initial_scale=1,pivot=None):
        """
        """
        if scale is None:
            scale = self.intrinsic_scale
        if not pivot:
            pivot = self.ref_obj.location
        ibpy.shrink_from(self,pivot, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
