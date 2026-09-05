import numpy as np

from interface import ibpy
from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME


class Whistle(BObject):
    """
    A whistle model loaded from a bundled .blend asset.
    """

    def __init__(self,**kwargs):
        """Load a whistle model.

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``location`` (list[float]): Left at the imported pose when
                  it is not given.
                * ``rotation_euler`` (list[float]): ditto.
                * ``original_material`` (bool): Keep the ``whistle_texture``
                  material that comes with ``Whistle.blend`` instead of
                  painting the model with a palette colour. Defaults to
                  ``True``. Pass ``False`` (together with ``color``/``colors``,
                  if a particular colour is wanted) to recolour the whistle
                  the way the other library primitives are recoloured.
        """
        self.kwargs = kwargs
        original_material = self.get_from_kwargs('original_material',True)

        # Only the transform the caller actually asked for is handed on, so
        # that the asset keeps whatever pose it is imported with - passing a
        # default of [0,0,0] would overwrite it instead.
        transform = {key:kwargs.pop(key) for key in
                     ('location','rotation_euler','rotation_quaternion') if key in kwargs}

        # Appending the object brings its material along, so keeping it is only
        # a matter of not painting over it. The import is therefore always done
        # with no_material, and any color is left to the wrapper below - whose
        # apply_material reaches through to the very same blender object and
        # would otherwise overwrite whatever the import had just painted.
        bobs = BObject.from_file("Whistle", objects=["Whistle"],no_material=True,**kwargs)

        # obj=bobs[0].ref_obj, not bobs[0]: wrapping the wrapper makes BObject
        # write location and rotation onto a python attribute of the inner
        # BObject, where they have no effect on the blender object at all
        if original_material:
            super().__init__(obj=bobs[0].ref_obj,name="Whistle",no_material=True,**transform)
        else:
            if 'colors' not in kwargs:
                kwargs.setdefault('color','drawing')  # the library default
            super().__init__(obj=bobs[0].ref_obj,name="Whistle",**kwargs,**transform)

