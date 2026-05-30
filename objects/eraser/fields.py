import bmesh
import numpy as np
import bpy
from mathutils import Vector

from interface.ibpy import add_wind, add_turbulence, add_force, decay_force
from objects.bobject import BObject
from utils.constants import FRAME_RATE, DEFAULT_ANIMATION_TIME


class Wind(BObject):
    """
    creates a wind field
    """
    def __init__(self, **kwargs):
        """Create a wind force field.

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``location`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``rotation_euler`` (list[float]): Wind direction.
                  Defaults to ``[0, 0, 0]``.
                * ``strength`` (float): Wind strength. Defaults to 1.
                * ``name`` (str): Defaults to ``'Wind'``.
        """
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        strength=self.get_from_kwargs('strength',1)
        name = self.get_from_kwargs('name',"Wind")
        wind = add_wind(location=location)
        super().__init__(obj=wind,name=name,rotation_euler=rotation, **kwargs)
        wind.field.strength = strength


class Turbulence(BObject):
    """
    creates a turbulence field
    """
    def __init__(self,**kwargs):
        """Create a turbulence force field.

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``location`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``strength`` (float): Turbulence intensity. Defaults to 1.
                * ``name`` (str): Defaults to ``'Turbulence'``.
        """
        self.kwargs=kwargs
        location = self.get_from_kwargs('location', [0, 0, 0])
        strength = self.get_from_kwargs('strength', 1)
        name = self.get_from_kwargs('name', "Turbulence")
        turbulence = add_turbulence(location=location)
        super().__init__(obj=turbulence, name=name, **kwargs)
        turbulence.field.strength = strength


class Force(BObject):
    """
    creates a turbulence field
    """
    def __init__(self,**kwargs):
        """Create a directional force field.

        Despite the class's brief docstring, ``Force`` is a generic
        push/pull force field (not specifically turbulence). The
        :meth:`disappear` method linearly decays its strength to 0.

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``location`` (list[float]): Defaults to ``[0, 0, 0]``.
                * ``strength`` (float): Force magnitude. Defaults to 1.
                * ``name`` (str): Defaults to ``'Force'``.
        """
        self.kwargs=kwargs
        location = self.get_from_kwargs('location', [0, 0, 0])
        self.strength = self.get_from_kwargs('strength', 1)
        name = self.get_from_kwargs('name', "Force")
        forcefield = add_force(location=location)
        super().__init__(obj=forcefield, name=name, **kwargs)
        forcefield.field.strength = self.strength

    def disappear(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        decay_force(self,self.strength,0,begin_frame=int(begin_time*FRAME_RATE),frame_duration=transition_time*FRAME_RATE)