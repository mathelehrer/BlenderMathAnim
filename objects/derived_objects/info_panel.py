import numpy as np

from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME


class InfoPanel(BObject):
    """A small information-panel asset loaded from the bundled .blend file.

    Carries an embedded text mesh and a backing panel; both can be
    revealed together via :meth:`appear`.
    """

    def __init__(self,**kwargs):
        """Load and place an info panel.

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``location`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``rotation_euler`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``colors`` (list[str]): Per-sub-object colors.
                  Defaults to ``['drawing']``. Object name is fixed to
                  ``'InfoPanel'``.
        """
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['drawing'])
        # The InfoPanel_wlog.blend asset contains a single object ("InfoPanel");
        # only bobs[0] is used below.
        bobs = BObject.from_file("InfoPanel_wlog", objects=["InfoPanel"],colors=colors)
        self.info = bobs[0]
        self.info.ref_obj.scale = [0.2,0.2,1.5]
        self.info.ref_obj.location=[-2,0,1.5]
        super().__init__(children=[self.info],name="InfoPanel",rotation_euler=rotation,location=location)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        self.info.appear(begin_time=begin_time,transition_time=transition_time)