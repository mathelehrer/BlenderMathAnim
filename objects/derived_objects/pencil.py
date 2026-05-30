import numpy as np

from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME


class Pencil(BObject):
    """A pencil model (wood + mine) loaded from a bundled .blend asset."""

    def __init__(self,**kwargs):
        """Load a pencil from the bundled asset library.

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``location`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``rotation_euler`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``colors`` (list[str]): One color per part
                  (``Wood``, ``Mine``). Defaults to ``['drawing']``.
                  Object name is fixed to ``'Pencil'``.
        """
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['drawing'])
        bobs = BObject.from_file("Pencil", objects=["Wood", "Mine"],colors=colors,**kwargs)
        self.wood = bobs[0]
        # self.wood.ref_obj.scale = [3.8, 0.2, 1.8]
        # self.wood.ref_obj.location=[0, 0, 1.9]
        self.mine = bobs[1]
        # self.mine.ref_obj.scale = [1.056, 0.1, 1.056]
        # self.mine.ref_obj.location=[-1.05, -0.1, 1.23]
        super().__init__(children=[self.wood, self.mine], name="Pencil", rotation_euler=rotation, location=location)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        self.mine.appear(begin_time=begin_time, transition_time=transition_time)
        self.wood.appear(begin_time=begin_time, transition_time=transition_time)

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        self.mine.disappear(begin_time=begin_time,transition_time=transition_time)
        self.wood.disappear(begin_time=begin_time,transition_time=transition_time)
