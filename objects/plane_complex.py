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
        """
        :param coordinate_system: the coordinate system, where the plane is located in
        :param u: =[-1,1]
        :param v: =[-1,1]
        :param resolution: The mesh resolution can be chosen, default 19
        :param kwargs:
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
