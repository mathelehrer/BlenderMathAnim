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
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        scale = self.get_from_kwargs('scale',[0,0,0])
        name = self.get_from_kwargs('name',"Empty_cube")

        bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=location, scale=scale)
        empty = bpy.context.object

        super().__init__(obj=empty, name=name, **kwargs)
        apply_location = self.get_from_kwargs('apply_location', False)
        apply_scale = self.get_from_kwargs('apply_scale', False)
        apply_rotation = self.get_from_kwargs('apply_rotation', False)
        ibpy.apply(self, location=apply_location, rotation=apply_rotation, scale=apply_scale)


class EmptyArrow(BObject):
    """
    Creates an empty arrow
    """
    def __init__(self, **kwargs):
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
        self.kwargs = kwargs
        name = self.get_from_kwargs('name',"axes_empty")

        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD')
        empty = bpy.context.object

        super().__init__(obj=empty, name=name,**kwargs)
        apply_location = self.get_from_kwargs('apply_location', False)
        apply_scale = self.get_from_kwargs('apply_scale', False)
        apply_rotation = self.get_from_kwargs('apply_rotation', False)
        ibpy.apply(self, location=apply_location, rotation=apply_rotation, scale=apply_scale)

