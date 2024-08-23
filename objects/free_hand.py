import os.path

import bpy
import numpy as np
from mathutils import Vector

from appearance.textures import apply_material
from interface import ibpy
from objects.bobject import BObject
from utils.constants import DATA_DIR, DEFAULT_ANIMATION_TIME, FRAME_RATE


def load_data(filename):
    '''
    load point data from file
    the data is stored in lines
    each line corresponds to one part of the curve
    (x_1,y_1),(x_2,y_2),...,(x_n,y_n)

    :param filename:
    :return:
    '''
    path = os.path.join(DATA_DIR, filename + '.text')
    file = open(path, "r")
    parts = []
    for line in file:
        line = line[1:len(line) - 2]  # remove first and last bracket
        line_parts = line.split('),(')
        if len(line_parts) > 1:
            parts.append([Vector([float(d) for d in data.split(',')] + [0]) for data in line_parts])
    return parts


class FreeHandText(BObject):
    def __init__(self, filename, eps=0.01, **kwargs):
        self.kwargs = kwargs
        self.parts = load_data(filename)
        self.clean_data(eps)
        self.align_data(**kwargs)
        self.letters = []

        bevel_depth = self.get_from_kwargs('bevel_depth', 0.01)

        for i, part in enumerate(self.parts):
            curve = ibpy.get_new_curve('curve' + str(i), num_points=len(part) - 1, data=part)
            if bevel_depth > 0:
                curve.bevel_depth = bevel_depth
                curve.use_fill_caps = True
            letter = ibpy.new_curve_object('letter' + str(i), curve)

            # link
            bpy.context.scene.collection.objects.link(letter)
            letter.select_set(True)
            bpy.context.view_layer.objects.active = letter
            bpy.ops.object.editmode_toggle()
            bpy.ops.curve.select_all(action='SELECT')
            bpy.ops.curve.handle_type_set(type='AUTOMATIC')
            bpy.ops.object.editmode_toggle()

            self.letters.append(letter)

        col = self.get_from_kwargs('color', 'text') # get color from kwargs for later
        super().__init__(children=self.letters, **kwargs)

        for letter in self.letters:
            apply_material(letter,col,**kwargs)

    def length(self, list_of_points):
        length = 0
        start = None
        for p in list_of_points:
            if start is not None:
                length += np.sqrt((start[0] - p[0]) ** 2 + (start[1] - p[1]) ** 2)
            start = p
        return length

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center'):
        super().appear(begin_time=begin_time, transition_time=0)

        total_length = 0
        lengths = []
        t0=begin_time

        for part in self.parts:
            l = self.length(part)
            total_length += l
            lengths.append(l)

        if len(self.parts) > 1:
            break_time = 0.1 * transition_time
            transition_time *= 0.9
            bt = break_time / (len(self.parts) - 1)

        for i,letter in enumerate(self.letters):
            dt = lengths[i]/total_length*transition_time
            ibpy.set_bevel_factor_and_keyframe(letter.data, 0, t0*FRAME_RATE)
            ibpy.set_bevel_factor_and_keyframe(letter.data, 1, t0*FRAME_RATE+np.maximum(1,dt*FRAME_RATE))
            t0+=dt
            if i<len(self.letters)-1:
                t0+=bt

    def align_data(self, **kwargs):
        if 'aligned' in kwargs:
            aligned = kwargs.pop('aligned')

        if aligned == 'center':
            counter = 0
            center = Vector()
            for part in self.parts:
                for point in part:
                    center += point
                    counter += 1
            center /= counter

            for i in range(len(self.parts)):
                for j in range(len(self.parts[i])):
                    self.parts[i][j] -= center
                    self.parts[i][j] -= center
        # TODO: implement when needed
        elif aligned == 'left':
            pass
        elif aligned == 'right':
            pass

    def clean_data(self, eps):
        for i in range(len(self.parts)):
            part = self.parts[i]
            if len(part) > 50:  # avoid to clean i-dots
                clean_part = []
                old = part[0]
                clean_part.append(old)
                for j in range(1, len(part)):
                    p = part[j]
                    if (old[0] - p[0]) ** 2 + (old[1] - p[1]) ** 2 > eps:
                        clean_part.append(p)
                        old = p
                self.parts[i] = clean_part

    def change_emission(self, from_value=0, to_value=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for letter in self.letters:
            ibpy.change_emission(letter,from_value=from_value,to_value=to_value,begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)
        return begin_time+transition_time
