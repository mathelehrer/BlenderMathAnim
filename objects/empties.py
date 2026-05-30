import bmesh
import numpy as np
import bpy
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject


class EmptyCube(BObject):
    """
    Creates an empty as bounding box for a curve
    """
    def __init__(self, **kwargs):
        """Create a cube-shaped empty (no mesh, useful as a bounding box).

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``name`` (str): Defaults to ``'Empty_cube'``.
                * ``location`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``scale`` (list[float]): Defaults to ``[0, 0, 0]``
                  (zero-sized; caller usually overrides).
                * ``apply_location``, ``apply_scale``, ``apply_rotation``
                  (bool): Bake transforms into the empty. Default ``False``.
        """
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        scale = self.get_from_kwargs('scale',[0,0,0])
        name = self.get_from_kwargs('name',"Empty_cube")

        bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=location, scale=scale)
        empty = bpy.context.object

        super().__init__(obj=empty, name=name, no_material=True, **kwargs)
        apply_location = self.get_from_kwargs('apply_location', False)
        apply_scale = self.get_from_kwargs('apply_scale', False)
        apply_rotation = self.get_from_kwargs('apply_rotation', False)
        ibpy.apply(self, location=apply_location, rotation=apply_rotation, scale=apply_scale)


class EmptyArrow(BObject):
    """
    Creates an empty arrow
    """
    def __init__(self, **kwargs):
        """Create a single-arrow empty (Blender's ``SINGLE_ARROW`` type).

        Useful as a constraint target or as a visual placeholder.

        Args:
            **kwargs: Forwarded to :class:`BObject`. ``name`` defaults
                to ``'Empty_arrow'``. Standard ``apply_*`` flags supported.
        """
        self.kwargs = kwargs
        name = self.get_from_kwargs('name',"Empty_arrow")

        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD')
        empty = bpy.context.object

        super().__init__(obj=empty, name=name,**kwargs)
        apply_location = self.get_from_kwargs('apply_location', False)
        apply_scale = self.get_from_kwargs('apply_scale', False)
        apply_rotation = self.get_from_kwargs('apply_rotation', False)
        ibpy.apply(self, location=apply_location, rotation=apply_rotation, scale=apply_scale)


class EmptyAxes(BObject):
    """
    Creates an empty as axes
    """
    def __init__(self, **kwargs):
        """Create a plain-axes empty (Blender's ``PLAIN_AXES`` type).

        Args:
            **kwargs: Forwarded to :class:`BObject`. ``name`` defaults
                to ``'axes_empty'``. Standard ``apply_*`` flags supported.
        """
        self.kwargs = kwargs
        name = self.get_from_kwargs('name',"axes_empty")

        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD')
        empty = bpy.context.object

        super().__init__(obj=empty, name=name,**kwargs)
        apply_location = self.get_from_kwargs('apply_location', False)
        apply_scale = self.get_from_kwargs('apply_scale', False)
        apply_rotation = self.get_from_kwargs('apply_rotation', False)
        ibpy.apply(self, location=apply_location, rotation=apply_rotation, scale=apply_scale)

