import os

import bpy
import numpy as np

from appearance.textures import apply_material
from interface.ibpy import link, set_bevel_factor_and_keyframe, set_alpha_and_keyframe, fade_in, fade_out
from objects.bobject import BObject
from objects.svg_bobject import SVGBObject
from objects.tex_bobject import reindex_to_useful_writing_start
from utils.constants import RES_SVG_DIR, FRAME_RATE


class Paws(SVGBObject):
    def __init__(self, steps=2, **kwargs):
        self.kwargs = kwargs
        scale = self.get_from_kwargs('scale', 1)
        scale *= 0.05
        aligned = self.get_from_kwargs('aligned', 'left')
        super().__init__(os.path.join(RES_SVG_DIR, 'paw_prints'), aligned=aligned, scale=scale, **kwargs)

        self.thickness = self.get_from_kwargs('thickness', 1)
        self.parts = []
        self.parts.append(self.rendered_objects)

        steps = int((steps + 1) / 2)
        self.parts = []
        for bob in self.rendered_objects:
            self.parts.append(bob)

        for i in range(1, steps):
            for bob in self.rendered_objects:
                ref = bob.ref_obj
                curve_copy = ref.copy()
                curve_copy.data = ref.data.copy()

                copy = BObject(obj=curve_copy,
                               name='copy_' + str(i) + '_of_' + ref.name)  # name='hand_writing_of_'+ref.name)
                copy.ref_obj.location = ref.location.copy()
                copy.ref_obj.location.x = i * 20 + ref.location.x
                copy.ref_obj.parent = ref.parent
                copy.ref_obj.rotation_euler = ref.rotation_euler
                self.parts.append(copy)

        # create parent child relation for the letters
        for part in self.parts:
            part.ref_obj.parent = self.ref_obj

        self.shadow = self.get_from_kwargs('shadow', True)
        if not self.shadow:
            for c in self.ref_obj.children:
                bpy.data.objects[c.name].visible_shadow = False

    def appear(self,
               begin_time=0,
               transition_time=0, writing=True,
               **kwargs):
        self.appeared=True
        link(self.ref_obj)  # linking is necessary here, since no super functions are called
        # This can be improved for the fade-in and fade-out part at least

        # unlike all other objects, the appearance of text is controlled completely in this
        # method and not outsourced to ibpy yet

        appear_frame = begin_time * FRAME_RATE
        n_frames = transition_time * FRAME_RATE
        frames_per_shape = n_frames / len(self.parts)

        frame = appear_frame

        for i, part in enumerate(self.parts):
            curve = part.ref_obj
            c1_data = curve.data
            c1_data.extrude = self.thickness
            c1_data.fill_mode = 'BOTH'

            # make open_path curve
            b_curve2 = self.open_path_object_from(part, re_index=True)
            curve2 = b_curve2.ref_obj
            curve2_data = curve2.data
            curve2_data.use_path = True
            curve2_data.bevel_depth = 0.0025
            curve2_data.fill_mode = 'NONE'
            curve2.scale[2] *= self.thickness
            for s in curve2_data.splines:
                s.use_cyclic_u = False

            if isinstance(self.default_color, list):
                if i < len(self.default_color):
                    apply_material(curve, self.default_color[i], emission=0.5)
                    apply_material(curve2, self.default_color[i], emission=5)
                else:
                    apply_material(curve, self.default_color[-1], emission=0.5)
                    apply_material(curve2, self.default_color[-1], emission=5)
            else:
                apply_material(curve, self.default_color, emission=0.5)
                apply_material(curve2, self.default_color, emission=5)

            link(curve)
            if writing:
                link(curve2)
                if int(frames_per_shape) > 0:
                    set_bevel_factor_and_keyframe(curve2_data, 0, frame)
                    frame = appear_frame + np.maximum(1, int((i + 0.9) * frames_per_shape))
                    # 90% of the time for fifty percent of the curve
                    set_alpha_and_keyframe(curve, 0, frame)
                    set_bevel_factor_and_keyframe(curve2_data, 0.5, frame)
                    frame1 = appear_frame + np.maximum(1, int((i + 1) * frames_per_shape))
                    duration = np.maximum(5, frame1 - frame)
                    fade_in(part, frame, duration)
                    frame = frame1
                    set_bevel_factor_and_keyframe(curve2_data, 1, frame)
                    fade_out(b_curve2, frame, duration, handwriting=True)
                else:
                    duration = FRAME_RATE / 6
                    fade_in(part, frame, 1)
                    set_bevel_factor_and_keyframe(curve2_data, 0, frame)
                    set_bevel_factor_and_keyframe(curve2_data, 1, frame + 1)
                    fade_out(b_curve2, frame, duration, handwriting=True)
            else:
                if frames_per_shape > 0:
                    set_alpha_and_keyframe(curve, 0, frame)
                    fade_in(part, frame, duration)
                else:
                    fade_in(part, frame, 1)
        return begin_time+transition_time

    def open_path_object_from(self, b_obj, re_index=True):
        ref = b_obj.ref_obj
        curve_copy = ref.copy()
        curve_copy.data = ref.data.copy()

        b_obj = BObject(obj=curve_copy, name='hand_writing')  # name='hand_writing_of_'+ref.name)
        b_obj.ref_obj.location = ref.location
        b_obj.ref_obj.parent = b_obj.ref_obj.parent
        b_obj.ref_obj.rotation_euler = b_obj.ref_obj.rotation_euler

        if re_index:
            reindex_to_useful_writing_start(b_obj.ref_obj)

        return b_obj
