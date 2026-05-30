import bmesh
import numpy as np
import bpy

from appearance.textures import make_complex_function_material, make_conformal_transformation_material
from interface import ibpy
from objects.bobject import BObject
from objects.plane import Plane
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE


class ComplexPlane(Plane):
    """
    Create a complex plane with a descent mesh:
    Since the standard blender plane only consists of 4 mesh points, the mesh is constructed manually.
    This is important, if the plane is subject to transformations

    example:

    final_plane = ComplexPlane(coord,
                                   [func, func],
                                   u=[-2, 2], v=[-2, 2],
                                   resolution=100,
                                   alpha=0.5,
                                   metallic=0,
                                   roughness=0.7,
                                   smooth=2)

    if a list of functions is given, the profile of the plane can be changed
    with the function next_shape. It provides a transition from one set of shape keys
    to the next. Only 8 functions are allowed in total (blender limitations)

    with the parameter smooth a subdivision surface modifier is added and the value
    level given by the parameter

    """

    def __init__(self, coordinate_system, functions, u=[-1, 1], v=[-1, 1], resolution=10, location=None, **kwargs):
        """Create a plane mapped by one or more complex functions.

        Each function in ``functions`` is baked into a shape key, so the
        plane's surface can morph between the functions via
        :meth:`next_shape`. Blender limits the chain to 8 functions.

        Args:
            coordinate_system: :class:`CoordinateSystem` to add the plane
                to (or ``None`` for world space).
            functions: Either a single ``complex -> complex`` callable
                or a list of such callables. If ``None``, no shader is
                applied (raw plane). Pass an empty list to fall back to
                the identity map.
            u: ``[u_min, u_max]`` extent along the real axis.
                Defaults to ``[-1, 1]``.
            v: ``[v_min, v_max]`` extent along the imaginary axis.
                Defaults to ``[-1, 1]``.
            resolution: Mesh subdivisions per axis.
            location: Plane location in the coordinate system. If ``None``,
                defaults to ``Plane``'s computed centre.
            **kwargs: Forwarded to :class:`Plane`. Supported keys:
                * ``name`` (str): Defaults to ``'ComplexPlane'``.
                * ``shape`` (bool): If ``True``, also deform vertex
                  positions when registering shape keys; if ``False``,
                  only the shader is affected. Defaults to ``True``.
                * ``conformal_transformations`` (list): Optional list of
                  conformal transformations; mutually exclusive with
                  ``functions`` (drives a transformation-specific shader).
                * ``alpha``, ``metallic``, ``roughness``, ``smooth``, etc.
                  -- standard appearance kwargs.
        """
        self.kwargs = kwargs
        if functions is not None:
            if not isinstance(functions, list):
                functions = [functions]
                self.only_one_function = True
            else:
                self.only_one_function = False
            if len(functions) == 0:
                functions = [lambda z: z]
        self.functions = functions

        name = self.get_from_kwargs('name', 'ComplexPlane')

        # mesh properties
        self.u = u
        self.v = v
        self.resolution = resolution

        super().__init__(u=u, v=v, location=location, resolution=resolution, name=name, **kwargs)
        if coordinate_system:
            coordinate_system.add_object(self)

        if 'conformal_transformations' in kwargs:
            conformal_transformations = kwargs['conformal_transformations']
        else:
            conformal_transformations = None

        # only call this after the mesh has been created
        if conformal_transformations is not None:
            self.mixer_dialers = make_conformal_transformation_material(self, conformal_transformations, name=name)
            self.current_mapping = 0
        elif functions is not None:
            shape = self.get_from_kwargs('shape', True)
            self.mixer_dialers = make_complex_function_material(self, functions, shape=shape, name=name, **kwargs)
            self.current_mapping = 0

    def next_shape(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        """
        transition to the next shape of the plane
        :param begin_time:
        :param transition_time:
        :return:
        """
        appear_frame = begin_time * FRAME_RATE
        dialer_index = self.current_mapping
        if len(self.mixer_dialers) > 0 and len(self.mixer_dialers) > dialer_index:
            current_dialer = self.mixer_dialers[dialer_index]
            ibpy.insert_keyframe(current_dialer, 'default_value', appear_frame)
            current_dialer.default_value = 1
            ibpy.insert_keyframe(current_dialer, 'default_value', appear_frame + transition_time * FRAME_RATE)

        ibpy.morph_to_next_shape(self.ref_obj, self.current_mapping, appear_frame, transition_time * FRAME_RATE)
        self.current_mapping += 1
        print("Next shape at " + str(begin_time) + " with transition time " + str(transition_time))
        return begin_time+transition_time

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time, transition_time=transition_time, **kwargs)
        if self.functions is not None:
            ibpy.morph_to_next_shape(self.ref_obj, 0, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
            self.current_mapping += 1
        return begin_time+transition_time
