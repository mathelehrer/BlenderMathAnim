import collections
import hashlib
import os
import time
from copy import deepcopy
from functools import partial

import bpy
import numpy as np
from mathutils import Vector, Matrix
from numpy import sort

from appearance.textures import apply_material
from interface import ibpy
from interface.ibpy import link, set_bevel_factor_and_keyframe, set_alpha_and_keyframe, fade_out, get_location
from objects.bobject import BObject
from objects.empties import EmptyCube
from objects.svg_bobject import SVGBObject, equalize_spline_count, new_null_curve
from utils.constants import FRAME_RATE, TEMPLATE_TEX_FILE, TEX_DIR, TEX_TEXT_TO_REPLACE, SVG_DIR, \
    OBJECT_APPEARANCE_TIME, CONTROL_POINTS_PER_SPLINE, DEFAULT_ANIMATION_TIME, TEMPLATE_TEXT_FILE
from utils.kwargs import get_from_kwargs
from utils.utils import to_vector


class TexBObject(BObject):
    """
        This class combines many expressions into a single blender object
    """

    def __init__(self, *expressions, **kwargs):
        self.kwargs = kwargs
        typeface = self.get_from_kwargs('typeface', 'default')
        name = self.get_from_kwargs('name', 'Extended_Tex')
        self.text_only = self.get_from_kwargs('text_only', False)
        location = self.get_from_kwargs('location',Vector())
        rotation_euler = self.get_from_kwargs('rotation_euler',[0,0,0])
        if 'rotation_quaternion' or 'rotation_euler' in kwargs:
            pass
        else:
            # set default rotation
            kwargs['rotation_euler'] = [np.pi / 2, 0, 0]

        if 'colors' in self.kwargs:
            self.colors = self.get_from_kwargs('colors', ['text'])
        else:
            self.colors = [self.get_from_kwargs('color', 'text')]

        scale = self.get_from_kwargs('scale', 1)

        self.objects = []
        for i, expression in enumerate(expressions):
            if len(self.colors) > i:
                color = self.colors[i]
            else:
                color = self.colors[-1]
            self.objects.append(
                SimpleTexBObject(expression,
                                 color=color,
                                 typeface=typeface,
                                 scale=scale,
                                 name=name + "_" + expression + '_' + str(i),
                                 **kwargs)
            )

        self.get_from_kwargs('bevel',0) # remove bevel to not interfer with mesh-objects
        super().__init__(children=self.objects,location=location,rotation_euler=rotation_euler, **kwargs)
        for sub_objects in self.objects:
            sub_objects.ref_obj.parent = self.ref_obj

        self.morph_counter = 0  # controls the position that is morphed to
        self.created_null_curves = self.objects[0].created_null_curves # create reference to newly created letters

    def get_number_of_expressions(self):
        return len(self.objects)

    def get_part(self, index, end_index=-1):
        if end_index < index:
            return self.objects[index]
        else:
            return self.objects[index:end_index]

    def disappear(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, **kwargs):
        for obj in self.objects:
            obj.disappear(begin_time=begin_time, transition_time=transition_time, **kwargs)

    def write(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        """
        The write animation consists of two parts. First the outline of the letter is drawn and
        eventually the shape of the letter appears on the screen

        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        """

        return self.write_index(0, begin_time=begin_time, transition_time=transition_time)


    def write_index(self, expression_index, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        ibpy.link(self.ref_obj)
        self.objects[expression_index].write(begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time

    def move(self, direction, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for obj in self.objects:
            obj.move(direction, begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time

    def next(self, letter_range=None, img_letter_range=None,begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        """
        the first object in self.objects remains the main object
        the next object in self.objects is always converted into a shape key transformation for the first object
        :param letter_range:
        :param begin_time:
        :param transition_time:
        :return:
        """

        if letter_range is None:
            letter_range = [0, len(self.objects[0].letters)]
        if img_letter_range is None:
            img_letter_range = [0, len(self.objects[1].letters)]

        # all data of the morph is stored in one single data struct
        if isinstance(self.objects[0].color,list):
            color_src = self.objects[0].color
        else:
            color_src=[self.objects[0].color]

        if isinstance(self.objects[1].color,list):
            color_img = self.objects[1].color
        else:
            color_img=[self.objects[1].color]

        self.objects[0].add_to_morph_chain(self.objects[1],
                                           letter_range, img_letter_range,
                                           [1] * 3, [0] * 3,
                                           color_src, color_img,
                                           begin_time=begin_time,
                                           transition_time=transition_time)

        # # transfer name of the destination to the src
        # dest_name=self.objects[self.morph_counter+1].ref_obj.name
        self.objects.pop(1)
        self.morph_counter += 1  # is needed for the correct change of colors
        return begin_time+transition_time

    def perform_morphing(self):
        for obj in self.objects:
            obj.perform_morphing()

    def morph_and_move(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):

        bb1 = self.get_first_text_bounding_box()
        bb2 = self.get_text_bounding_box()
        bb3 = self.get_second_text_bounding_box()

        # it's important to first measure bb1 and bb2 since morphing merges the objects
        self.next(begin_time=begin_time, transition_time=transition_time)
        self.perform_morphing()
        # compensate the change in bounding box by a motion of the object to keep it centered in the table column

        #shift = -0.5 * Vector([bb2[3] - bb2[0] - bb1[3] + bb1[0], 0, 0]) # for penrose this has to be used, has to be done more wisely
        shift = -0.5 * Vector([bb3[3] - bb3[0] - bb1[3] + bb1[0], 0, 0])

        self.move(direction=shift, begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time

    def get_location(self, letter=0):
        return self.ref_obj.location

    def get_text_bounding_box(self):
        '''
        designated for SimpleTexBOjects
        it works on the level of letters
        :param texBObject:
        :return:
        '''

        bounds =[part.get_text_bounding_box() for part in self.objects]
        x_min=min([x[0] for x in bounds])
        y_min=min([x[1] for x in bounds])
        z_min=min([x[2] for x in bounds])
        x_max=max([x[3] for x in bounds])
        y_max=max([x[4] for x in bounds])
        z_max=max([x[5] for x in bounds])
        return [x_min,y_min,z_min,x_max,y_max,z_max]

    def get_first_text_bounding_box(self):
        return self.objects[0].get_text_bounding_box()

    def get_second_text_bounding_box(self):
        return self.objects[1].get_text_bounding_box()

class FastTexBObject(SVGBObject):
    """
        Tries to combine all splines into one curve
        TODO Trouble at overlaps
        """

    def __init__(self, expression, **kwargs):

        start = time.perf_counter()
        self.kwargs = kwargs
        self.centered = self.get_from_kwargs('centered', False)
        self.typeface = self.get_from_kwargs('typeface', 'default')
        self.rotation_euler = self.get_from_kwargs('rotation_euler', [np.pi / 2, 0, 0])
        self.color = self.get_from_kwargs('color', ['text'])
        self.min_length = 1  # don't know what this is for yet
        self.reindex_points_before_morph = True
        self.text_only = self.get_from_kwargs('text_only', False)
        self.recreate = self.get_from_kwargs('recreate',False)

        if 'vert_align_centers' not in kwargs:
            kwargs['vert_align_centers'] = True

        if 'name' not in kwargs:
            kwargs['name'] = 'tex'

        if 'rotation_euler' in kwargs:
            kwargs.pop('rotation_euler')

        super().__init__(expression, rotation_euler=self.rotation_euler, color=self.color, **kwargs)
        self.annotations = []
        self.letters = self.rendered_objects

        # add all splines to first letter
        full_letter = ibpy.merge_splines(self.letters)
        ibpy.link(full_letter.ref_obj)
        full_letter.ref_obj.parent = self.ref_obj

        if 'thickness' in kwargs:
            thickness = kwargs['thickness']
            self.thickness = thickness
            for letter in self.letters:
                letter.ref_obj.scale[2] *= thickness
        else:
            self.thickness = 1
        ende = time.perf_counter()
        # print("construction of "+self.ref_obj.name+" took "+str(ende-start)+" s.")

    def get_file_path(self, expression):
        # Replaces the svg_b_object method
        if self.text_only:
            template = deepcopy(TEMPLATE_TEXT_FILE)
        else:
            template = deepcopy(TEMPLATE_TEX_FILE)
        if self.typeface != 'default':
            template = template[:-4]  # chop off the .tex
            template += '_' + self.typeface + '.tex'
            if not os.path.exists(template):
                raise Warning(r'Can\'t find template tex file for that font.')

        self.path = tex_to_svg_file(expression, template, self.typeface, self.text_only,self.recreate)

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

    def get_figure_curves(self, fig):
        """
        returns the imported svg data without the leading 'H' that was added for alignment vertically
        :param fig:
        :return:
        """
        if fig is None:
            return self.imported_svg_data[fig]['curves']
        else:
            return self.imported_svg_data[fig]['curves'][1:]

    def align_figure(self, fig):
        curve_list = super().align_figure(fig)
        # Get rid of reference H after alignment is done
        if fig is None:
            self.imported_svg_data[fig]['curves'] = curve_list
        else:
            self.imported_svg_data[fig]['curves'] = curve_list[1:]


class SimpleTexBObject(SVGBObject):
    """
    This is just a container that provides the tex functionality, ie file operations
    In the end, tex is reduced to
    svg data and the morphing, transformation and alignment is provided in the super class

    color can be a single color or a list of colors
    """

    def __init__(self, expression, **kwargs):
        # store for copies
        self.expression_copy = expression
        self.kwargs_copy = kwargs.copy()

        self.kwargs = kwargs
        self.centered = self.get_from_kwargs('centered', False)
        self.typeface = self.get_from_kwargs('typeface', 'default')
        self.text_only = self.get_from_kwargs('text_only', False)
        self.recreate = self.get_from_kwargs('recreate',False)
        self.center_letters_origin = self.get_from_kwargs('center_letters_origin',False)
        if 'rotation_quaternion' in kwargs:
            pass
        elif 'rotation_euler' not in kwargs:
            kwargs['rotation_euler'] = [np.pi / 2, 0, 0]
            # set default rotation

        self.color = self.get_from_kwargs('color', ['text'])
        self.brighter = self.get_from_kwargs('brighter', 0)
        self.emission = self.get_from_kwargs('emission', 0.5)
        self.thickness = self.get_from_kwargs('thickness', 1)
        self.thickness *= 0.01

        self.bevel = self.get_from_kwargs('bevel', 0)
        self.bevel *= 0.005
        self.min_length = 1  # don't know what this is for yet
        self.reindex_points_before_morph = True

        # variables for morphing more than once
        self.morph_obj_chain = []
        self.max_spline_count = 0
        self.max_point_count = 0

        if 'vert_align_centers' not in kwargs:
            kwargs['vert_align_centers'] = True
        self.name = self.get_from_kwargs('name', 'tex')
        self.shadow = self.get_from_kwargs('shadow', True)  # catch shadow property to apply it to letters individually
        super().__init__(expression, color=self.color, name=self.name, **kwargs)

        self.annotations = []
        self.letters = self.rendered_objects
        self.created_null_curves = []
        self.copies_of_letters = []

        # create parent child relation for the letters
        for letter in self.letters:
            letter.ref_obj.parent = self.ref_obj

        # for letter in self.letters:
        #     letter.ref_obj.scale[2] *= self.thickness
        self.copies_of_letters = []
        # take care of shadows
        for c in self.ref_obj.children:
            ibpy.set_shadow_of_object(c,self.shadow)

        ende = time.perf_counter()
        # print("construction of "+self.ref_obj.name+" took "+str(ende-start)+" s.")
        self.copies_of_letters = []  # needed for transformations
        self.outlined = self.get_from_kwargs('outlined', None)

    def convert_to_mesh(self, **kwargs):
        ibpy.convert_to_mesh(self.letters, **kwargs)

    def get_letter(self, index):
        return self.letters[index]

    def get_bounding_box(self):
        return self.get_text_bounding_box()

    def get_text_bounding_box(texBObject):
        '''
        designated for SimpleTexBOjects
        it works on the level of letters
        :param texBObject:
        :return:
        '''

        x_min = np.Infinity
        x_max = -np.Infinity
        y_min = np.Infinity
        y_max = -np.Infinity
        z_min = np.Infinity
        z_max = -np.Infinity

        for letter in texBObject.letters:
            box = ibpy.get_bounding_box_for_letter(letter)
            if box[0] < x_min:
                x_min = box[0]
            if box[3] > x_max:
                x_max = box[0]
            if box[1] < y_min:
                y_min = box[1]
            if box[4] > y_max:
                y_max = box[4]
            if box[2] < z_min:
                z_min = box[2]
            if box[5] > z_max:
                z_max = box[5]
        return [x_min, y_min, z_min, x_max, y_max, z_max]


    def get_letters(self):
        return self.letters

    def copy(self, letter_range=None):
        if 'name' in self.kwargs_copy:
            name = "Copy_of_"+self.kwargs_copy.pop('name')
        else:
            name="Copy_of_"+self.name
        c = SimpleTexBObject(self.expression_copy,name = name,**self.kwargs_copy)
        return c

    def remove_letter(self, letter, ipby=None):
        if letter in self.letters:
            ibpy.unlink(letter.ref_obj)
            self.letters.pop(letter)

    def set_scale(self, scale=[1] * 3):
        self.ref_obj.scale = scale

    ###########################
    # additional Animations ###
    ###########################

    def to_second_shape(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for letter in self.letters:
            # shape keys
            ref_char1 = letter.ref_obj
            start_frame=begin_time*FRAME_RATE
            end_frame=(begin_time+transition_time)*FRAME_RATE
            if len(ref_char1.data.shape_keys.key_blocks)>1:
                eval_time = ref_char1.data.shape_keys.key_blocks[-2].frame
                ref_char1.data.shape_keys.eval_time = eval_time
                ref_char1.data.shape_keys.keyframe_insert(data_path='eval_time', frame=start_frame)

                eval_time = ref_char1.data.shape_keys.key_blocks[-1].frame
                ref_char1.data.shape_keys.eval_time = eval_time
                ref_char1.data.shape_keys.keyframe_insert(data_path='eval_time', frame=end_frame)
                ref_char1.data.shape_keys.eval_time = 0

    def to_first_shape(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for letter in self.letters:
            # shape keys
            ref_char1 = letter.ref_obj
            start_frame = begin_time * FRAME_RATE
            end_frame = (begin_time + transition_time) * FRAME_RATE
            if len(ref_char1.data.shape_keys.key_blocks) > 1:
                eval_time = ref_char1.data.shape_keys.key_blocks[-1].frame
                ref_char1.data.shape_keys.eval_time = eval_time
                ref_char1.data.shape_keys.keyframe_insert(data_path='eval_time', frame=start_frame)

                eval_time = ref_char1.data.shape_keys.key_blocks[-2].frame
                ref_char1.data.shape_keys.eval_time = eval_time
                ref_char1.data.shape_keys.keyframe_insert(data_path='eval_time', frame=end_frame)
                ref_char1.data.shape_keys.eval_time = 0

    def shader_value(self,old_value,new_value,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for letter in self.letters:
            letter.shader_value(old_value=old_value,new_value=new_value,begin_time=begin_time,transition_time=transition_time)

    def transform_mesh(self, transformation, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):

        self.convert_to_mesh(apply_transform=True)
        for letter in self.letters:
            letter.transform_mesh(transformation=transformation, begin_time=begin_time, transition_time=transition_time)

    def transform_mesh_to_next_shape(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        for letter in self.letters:
            letter.transform_mesh_to_next_shape(begin_time=begin_time, transition_time=transition_time, **kwargs)
        return begin_time + transition_time

    def transform_mesh_to_previous_shape(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for letter in self.letters:
            letter.transform_mesh_to_previous_shape(begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def scale(self, initial_scale=0, final_scale=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for letter in self.letters:
            letter.scale(initial_scale=initial_scale, final_scale=final_scale, begin_time=begin_time,
                         transition_time=transition_time)

    def shift(self, shift=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for letter in self.letters:
            letter.shift(shift=shift, begin_time=begin_time,
                         transition_time=transition_time)

    def write(self, letter_set=None, letter_range=None, letters=None, begin_time=0,
              transition_time=OBJECT_APPEARANCE_TIME, alpha=1,
              writing=True, sorting=None, order=None,verbose=False):
        """
        write expression, just the outline is drawn for each letter then the letter fades in
        :param letter_set:
        :param order:
        :param writing:
        :param sorting:
        :param letter_range: define the number of letters that are written
        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        """

        if sorting=='natural':
            for render in self.rendered_objects:
                ibpy.link(render)
                ibpy.set_origin(render, type='ORIGIN_GEOMETRY')

        if transition_time == 0:  # disable writing without transition time
            writing = False

        link(self.ref_obj)  # linking is necessary here, since no super functions are called
        # This can be improved for the fade-in and fade-out part at least

        # unlike all other objects, the appearance of text is controlled completely in this
        # method and not outsourced to ibpy yet
        appear_frame = begin_time * FRAME_RATE
        n_frames = transition_time * FRAME_RATE

        if order is not None:
            letter_set = []
            for o in order:
                for i, l in enumerate(self.letters):
                    if str(o) in l.ref_obj.name:
                        letter_set.append(i)
                        break

        if letter_range is not None:
            start = letter_range[0]
            end = letter_range[1]
            letter_set = list(range(start, end, 1))

        if letter_set is None:
            letter_set = list(range(0, len(self.rendered_objects)))

        frames_per_shape = int(n_frames / len(letter_set))

        if letters:
            selected_letters = letters
        else:
            selected_letters = [self.rendered_objects[i] for i in letter_set]

        if sorting == 'natural':
            # sort letters into rows
            rows = []
            # extended letters like fraction lines, backets or borderlines are captured as special objects
            specials = []
            extension_parameter = 15.3 # it is chosen such that a minus sign is still not a special letter
            for i, letter in enumerate(selected_letters):
                bb = ibpy.get_bounding_box_for_letter(letter)
                ratio = (bb[3] - bb[0]) / (bb[4] - bb[1])
                if ratio < 1:
                    ratio = 1 / ratio
                if ratio > extension_parameter:
                    specials.append(i)
                else:
                    inserted = False
                    for row in rows:
                        if row.contains(letter):
                            row.insert(letter, i)
                            inserted = True
                            break
                    if not inserted:
                        row = Row(letter, i)
                        rows.append(row)
                # print(i,ratio)
            letter_set = []

            rows.sort(key=lambda x: -x.max)

            for row in rows:
                letter_set += row.indices

            # add special characters at the end
            if len(specials) > 0:
                letter_set += specials

            selected_letters = [selected_letters[i] for i in letter_set]

        for i, letter in enumerate(selected_letters):
            frame = appear_frame + int(i * frames_per_shape)
            render = letter
            curve = render.ref_obj
            c1_data = curve.data
            c1_data.extrude = self.thickness
            c1_data.bevel_depth = self.bevel
            if hasattr(c1_data.fill_mode, 'BOTH'):  # 2D curves
                c1_data.fill_mode = 'BOTH'
            elif hasattr(c1_data.fill_mode, 'FULL'):  # 3D curves
                c1_data.fill_mode = 'FULL'

            # make open_path curve
            b_curve2 = self.open_path_object_from(render, re_index=True, index=i)
            curve2 = b_curve2.ref_obj
            curve2_data = curve2.data
            curve2_data.use_path = True
            curve2_data.bevel_depth = 0.005
            # if hasattr(curve2_data.fill_mode,'NONE'):
            #     curve2_data.fill_mode = 'NONE'
            curve2.scale[2] *= self.thickness
            for spline in curve2_data.splines:
                spline.use_cyclic_u = False

            # make closed open curve if needed
            if self.outlined:
                b_curve3 = self.closed_path_object_from(render, re_index=True, index=i)
                curve3 = b_curve3.ref_obj
                curve3_data = curve3.data
                curve3_data.use_path = True
                curve3_data.bevel_depth = 0.005
                curve3_data.fill_mode = 'NONE'
                # curve3.scale[2] *= self.thickness
                for spline in curve3_data.splines:
                    spline.use_cyclic_u = True
                apply_material(curve3, self.outlined, emission=0)

                if not self.shadow:
                    curve3.visible_shadow = False

            apply_material(curve, self.color_map[letter], emission=self.emission, brighter=self.brighter,**self.kwargs)
            apply_material(curve2, self.color_map[letter], emission=50, brighter=self.brighter,**self.kwargs)
            ibpy.set_shadow_of_object(curve2, self.shadow)

            if writing:
                link(curve2)
                if self.center_letters_origin:
                    ibpy.set_origin(b_curve2,type='ORIGIN_GEOMETRY')
                if int(frames_per_shape) > 0:
                    set_bevel_factor_and_keyframe(curve2_data, 0, frame - 1)
                    frame1 = frame + np.maximum(1, int(0.9 * frames_per_shape))
                    # 75% of the time for fifty percent of the curve
                    set_bevel_factor_and_keyframe(curve2_data, 0.75, frame1)
                    frame2 = frame + np.maximum(1, int(frames_per_shape))
                    duration = np.maximum(5, frame2 - frame1)
                    render.appear(alpha=alpha,begin_time=frame1 / FRAME_RATE, transition_time=duration / FRAME_RATE, silent=True)
                    set_bevel_factor_and_keyframe(curve2_data, 1, frame2)
                    fade_out(b_curve2, frame2, duration, handwriting=True)
                else:
                    render.appear(alpha=alpha,begin_time=frame / FRAME_RATE, transition_time=1 / FRAME_RATE, silent=True)
                    set_bevel_factor_and_keyframe(curve2_data, 0, frame)
                    set_bevel_factor_and_keyframe(curve2_data, 1, frame + 1)
                    fade_out(b_curve2, frame, 10, handwriting=True)
            else:
                if int(frames_per_shape) > 0:
                    set_alpha_and_keyframe(curve, 0, frame)
                    render.appear(alpha=alpha,begin_time=frame / FRAME_RATE, transition_time=frames_per_shape / FRAME_RATE,
                                  silent=True)
                else:
                    render.appear(alpha=alpha,begin_time=frame / FRAME_RATE, transition_time=0, silent=True)

            if self.outlined:
                b_curve3.appear(alpha=alpha,begin_time=frame / FRAME_RATE, transition_time=frames_per_shape / FRAME_RATE,
                                silent=True)
                if self.center_letters_origin:
                    ibpy.set_origin(b_curve3,type='ORIGIN_GEOMETRY')
        if verbose:
            print("Wrote: "+self.name+" at "+str(begin_time))
        return begin_time + transition_time

    def appear(self, letter_set=None, letter_range=None, begin_time=0, transition_time=0, **kwargs):
        """
        appear without writing
        :param letter_set:
        :param letter_range:
        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        """
        writing = get_from_kwargs(kwargs,'writing',True)
        self.write(letter_set=letter_set, letter_range=letter_range, begin_time=begin_time,
                   transition_time=transition_time, writing=writing)

    def change_alpha(self,alpha=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        [letter.change_alpha(alpha=alpha,begin_time=begin_time,transition_time=transition_time) for letter in self.letters]
        return begin_time+transition_time


    def align(self, other, char_index=0, other_char_index=0):
        # align with other
        diff = other.ref_obj.location[0] - self.ref_obj.location[0]
        self.ref_obj.location[0] += diff
        # align the chars
        diff = other.letters[other_char_index].ref_obj.location[0] - self.letters[char_index].ref_obj.location[0]
        for letter in self.letters:
            letter.ref_obj.location[0] += diff


    def batch_replace(self, b_tex_obj, src_ranges=None, img_ranges=None, shifts=[[0, 0]], begin_time=0,
                      transition_time=DEFAULT_ANIMATION_TIME):
        count = 0
        for src, img in zip(src_ranges, img_ranges):
            if count < len(shifts):
                shift = shifts[count]
            else:
                shift = shifts[-1]
            count += 1

            self.replace(b_tex_obj, src_letter_range=src, img_letter_range=img, shift=shift, begin_time=begin_time,
                         transition_time=transition_time, morphing=False)
        self.perform_morphing()


    def replace(self, b_tex_obj, src_letter_range=None, img_letter_range=None, rescale=[1, 1, 1], shift=[0, 0],
                begin_time=0,
                transition_time=OBJECT_APPEARANCE_TIME, morphing=True, keep_color=False,in_place=False):
        """
        this is supposed to be a one-time replacement
        the replaced object cannot be changed again.
        When more then one change is needed use morph
        :param in_place: if true, splines can be moved while morphing
        :param keep_color:
        :param morphing:
        :param b_tex_obj:
        :param src_letter_range:
        :param img_letter_range:
        :param rescale:
        :param shift:
        :param begin_time:
        :param transition_time:
        :return:
        """

        if src_letter_range is None:
            src_letter_range = [0, len(self.letters)]
        if img_letter_range is None:
            img_letter_range = [0, len(b_tex_obj.letters)]

        if isinstance(self.color, str):
            src_color = [self.color]
        else:
            src_color = self.color
        if isinstance(b_tex_obj.color, str):
            img_color = [b_tex_obj.color]
        else:
            img_color = b_tex_obj.color

        self.add_to_morph_chain(b_tex_obj, src_letter_range, img_letter_range, rescale, shift,
                                src_color, img_color,
                                begin_time=begin_time, transition_time=transition_time, keep_color=keep_color,in_place=in_place)
        if morphing:
            self.perform_morphing()
        if begin_time is not None:
            return begin_time + transition_time


    def replace2(self, b_tex_obj, src_letter_range=None, img_letter_range=None, rescale=[1, 1, 1], shift=[0, 0],
                 begin_time=0,
                 transition_time=OBJECT_APPEARANCE_TIME, morphing=True):
        """
        this is supposed to be a one-time replacement
        the replaced object cannot be changed again.
        When more then one change is needed use morph
        In this function the replace is physically replaced by the replacement after morphing
        :param morphing:
        :param b_tex_obj:
        :param src_letter_range:
        :param img_letter_range:
        :param rescale:
        :param shift:
        :param begin_time:
        :param transition_time:
        :return:
        """

        full_src = False
        if src_letter_range is None:
            src_letter_range = [0, len(self.letters)]
            full_src = True
        if img_letter_range is None:
            img_letter_range = [0, len(b_tex_obj.letters)]

        if isinstance(self.color, str):
            src_color = [self.color]
        else:
            src_color = self.color
        if isinstance(b_tex_obj.color, str):
            img_color = [b_tex_obj.color]
        else:
            img_color = b_tex_obj.color

        self.add_to_morph_chain(b_tex_obj, src_letter_range, img_letter_range, rescale, shift,
                                src_color, img_color,
                                begin_time=begin_time, transition_time=transition_time)
        if morphing:
            self.perform_morphing()

        b_tex_obj.write(letter_set=range(img_letter_range[0], img_letter_range[1]),
                        begin_time=begin_time + transition_time, transition_time=0)
        if full_src:
            self.disappear(begin_time=begin_time + transition_time, transition_time=0)
        else:
            for i in range(src_letter_range[0], src_letter_range[1]):
                self.letters[i].disappear(begin_time=begin_time + transition_time, transition_time=0)


    def grow_letter(self, index, final_scale=1, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):
        letter = self.letters[index]
        letter.appear(begin_time=begin_time, transition_time=transition_time)
        letter.grow(final_scale, begin_time=begin_time, transition_time=transition_time)
        self.write(letter_set={index}, begin_time=begin_time * 1.02, transition_time=0)


    def move_to_match_letter(self, target=None, src_letter_index=0, target_letter_index=0, begin_time=0,
                             transition_time=DEFAULT_ANIMATION_TIME):
        if not target:
            target = self

        src_letter = self.letters[src_letter_index]
        target_letter = target.letters[target_letter_index]
        shift = ibpy.get_location(target_letter) - ibpy.get_location(src_letter)
        scale = self.ref_obj.scale
        target_location = ibpy.get_location(target)
        for i in range(3):
            shift[i] *= scale[i]
        target_location += shift

        ibpy.set_location(self, location=target_location)


    def move_copy_of_letter_to(self,index,target_location,begin_time=0,rescale=1,transition_time=DEFAULT_ANIMATION_TIME):
        if self.copies_of_letters is None:
            self.copies_of_letters=[]

        letter_copy = self.letters[index].copy(clear_animation_data=True)
        self.copies_of_letters.append(letter_copy)
        letter_copy.appear(begin_time=begin_time,transition_time=0)
        if rescale!=1:
            letter_copy.rescale(rescale=rescale,begin_time=begin_time,transition_time=transition_time)
        return letter_copy.move_to(target_location=target_location,begin_time=begin_time,transition_time=transition_time)


    def move_letters_to(self, target=None, src_letter_indices=[], target_letter_indices=[], begin_time=0,
                        offsets=[[0, 0, 0]],
                        transition_time=DEFAULT_ANIMATION_TIME):
        if not target:
            target = self

        scale = self.ref_obj.scale
        shift = ibpy.get_location(target) - ibpy.get_location(self)
        for i in range(3):
            shift[i] /= scale[i]

        count = 0
        for src_index, target_index in zip(src_letter_indices, target_letter_indices):
            letter = self.letters[src_index]
            if len(offsets) > count:
                offset = offsets[count]
            else:
                offset = offsets[-1]
            offset = to_vector(offset)
            letter.move_to(target_location=shift + ibpy.get_location(target.letters[target_index]) + offset,
                           begin_time=begin_time, transition_time=transition_time)
            count += 1


    def move_null_curves_to(self, target=None, null_indices=[], target_letter_indices=[], begin_time=0,
                            offsets=[[0, 0, 0]],
                            transition_time=DEFAULT_ANIMATION_TIME):
        if not target:
            target = self

        scale = self.ref_obj.scale
        shift = ibpy.get_location(target) - ibpy.get_location(self)
        for i in range(3):
            shift[i] /= scale[i]

        count = 0
        for src_index, target_index in zip(null_indices, target_letter_indices):
            letter = self.created_null_curves[src_index]
            if len(offsets) > count:
                offset = offsets[count]
            else:
                offset = offsets[-1]
            offset = to_vector(offset)
            letter.move_to(target_location=shift + ibpy.get_location(target.letters[target_index]) + offset,
                           begin_time=begin_time, transition_time=transition_time)
            count += 1


    def move_copy_to_and_remove(self, target=None, src_letter_indices=[], target_letter_indices=[], begin_time=0,
                                rescale=None,
                                new_color=None, offset=[0, 0, -0.001], transition_time=DEFAULT_ANIMATION_TIME,
                                remove_time=None, detour=None):

        """
        the default offset shifts the copy behind the replacing target, therefore there will be no disturbation during the transition
        for transformations of the backside the sign has to be altered

        :param target:
        :param src_letter_indices:
        :param target_letter_indices:
        :param begin_time:
        :param rescale:
        :param new_color:
        :param offset:
        :param transition_time:
        :param remove_time:
        :return:
        """
        if not target:
            target = self

        if not remove_time:
            remove_time = begin_time + transition_time

        # set default range
        if len(src_letter_indices) == 0:
            src_letter_indices = list(range(len(self.letters)))
        if len(target_letter_indices) == 0:
            target_letter_indices = list(range(len(target.letters)))

        self.move_copy_to(target=target, src_letter_indices=src_letter_indices,
                          target_letter_indices=target_letter_indices,
                          begin_time=begin_time, new_color=new_color, offset=offset, transition_time=transition_time,
                          detour=detour)
        # replace with target letter after moving
        for copy in self.copies_of_letters:
            if rescale is not None:
                ibpy.rescale(copy, re_scale=rescale, begin_frame=begin_time * FRAME_RATE,
                             frame_duration=transition_time * FRAME_RATE)
            copy.disappear(begin_time=remove_time, transition_time=transition_time / 2)
        target.write(letter_set=target_letter_indices, begin_time=remove_time, transition_time=0)
        return begin_time+transition_time


    def move_copy_of_null_to_and_remove(self, target=None, null_indices=[], target_letter_indices=[], begin_time=0,
                                        rescale=None,
                                        new_color=None, offset=[0, 0, 0], transition_time=DEFAULT_ANIMATION_TIME,
                                        remove_time=None):

        if not target:
            target = self

        if not remove_time:
            remove_time = begin_time + transition_time

        self.move_copy_of_null_to(target=target, src_letter_indices=null_indices,
                                  target_letter_indices=target_letter_indices,
                                  begin_time=begin_time, new_color=new_color, offset=offset,
                                  transition_time=transition_time)
        # replace with target letter after moving
        for copy in self.copies_of_letters:
            if rescale is not None:
                ibpy.rescale(copy, re_scale=rescale, begin_frame=begin_time * FRAME_RATE,
                             frame_duration=transition_time * FRAME_RATE)
            copy.disappear(begin_time=remove_time, transition_time=transition_time / 2)
        target.write(letter_set=target_letter_indices, begin_time=begin_time + transition_time, transition_time=0)

    def move_copy_to(self, target=None, src_letter_indices=[], target_letter_indices=[], begin_time=0, new_color=None,
                     offset=[0, 0, 0],
                     transition_time=DEFAULT_ANIMATION_TIME, detour=None):
        if not target:
            target = self

        scale = self.ref_obj.scale
        shift = ibpy.get_location(target) - ibpy.get_location(self)
        # the shift has to be rotated by the rotation of the target and source
        # TODO there is a big problem that all letters are centered vertically and the vertical shift has to be introduced manually by an offset
        rotation = self.ref_obj.matrix_world.copy()
        rotation.invert()
        shift = rotation.to_3x3()@shift
        for i in range(3):
            shift[i] /= scale[i]
            shift[i] += offset[i]

        #set default range
        if len(src_letter_indices)==0:
            src_letter_indices=list(range(len(self.letters)))
        if len(target_letter_indices)==0:
            target_letter_indices=list(range(len(target.letters)))

        for src_index, target_index in zip(src_letter_indices, target_letter_indices):
            letter_copy = self.letters[src_index].copy(clear_animation_data=True)
            self.copies_of_letters.append(letter_copy)
            letter_copy.appear(begin_time=begin_time, transition_time=0, clear_data=True)  # make copy appear
            if detour:
                detour = to_vector(detour)
                letter_copy.move_to(target_location=get_location(letter_copy) + detour, begin_time=begin_time,
                                    transition_time=transition_time / 4)
                letter_copy.move_to(target_location=detour + shift + get_location(target.letters[target_index]),
                                    begin_time=begin_time + transition_time / 4, transition_time=transition_time / 2)
                letter_copy.move_to(target_location=+shift + get_location(target.letters[target_index]),
                                    begin_time=begin_time + transition_time * 0.75, transition_time=transition_time / 4)
            else:
                letter_copy.move_to(target_location=shift + ibpy.get_location(target.letters[target_index]),
                                    begin_time=begin_time, transition_time=transition_time)
            if new_color:
                letter_copy.change_color(new_color=new_color, begin_time=begin_time + transition_time / 2,
                                         transition_time=transition_time / 2)
        return begin_time+transition_time



    def move_copy_of_null_to(self, target=None, null_indices=[], target_letter_indices=[], begin_time=0, new_color=None,
                             offset=[0, 0, 0],
                             transition_time=DEFAULT_ANIMATION_TIME):
        """
        moves copies of null curves that are created during morphing
        :param target:
        :param null_indices:
        :param target_letter_indices:
        :param begin_time:
        :param new_color:
        :param offset:
        :param transition_time:
        :return:
        """
        if not target:
            target = self

        self.copies_of_letters = []
        scale = self.ref_obj.scale
        shift = ibpy.get_location(target) - ibpy.get_location(self)
        for i in range(3):
            shift[i] /= scale[i]
            shift[i] += offset[i]

        for src_index, target_index in zip(null_indices, target_letter_indices):
            letter_copy = self.created_null_curves[src_index].copy()
            print(self.letters[src_index].ref_obj.name, " copied")
            self.copies_of_letters.append(letter_copy)
            letter_copy.appear(begin_time=begin_time, transition_time=0, clear_data=True)  # make copy appear
            letter_copy.move_to(target_location=shift + ibpy.get_location(target.letters[target_index]),
                                begin_time=begin_time, transition_time=transition_time)
            if new_color:
                letter_copy.change_color(new_color=new_color, begin_time=begin_time - transition_time / 2,
                                         transition_time=transition_time)


    def disappear_copies(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, display=None):
        for letter_copy in self.copies_of_letters:
            letter_copy.disappear(begin_time=begin_time, transition_time=transition_time)
            # if display:
            #     display.hide(letter_copy,begin_time=begin_time,transition_time=transition_time)
            letter_copy.toggle_hide(begin_time=(begin_time + transition_time))
        return begin_time+transition_time


    def hide(self, begin_time=0):
        for letter in self.letters:
            letter.toggle_hide(begin_time=begin_time)


    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        super().disappear(begin_time=begin_time, transition_time=transition_time, **kwargs)
        half = transition_time/2
        delta = half/np.maximum(1,len(self.letters))
        for i,l in enumerate(self.letters):
            l.disappear(begin_time=begin_time+i*delta, transition_time=half, **kwargs)
        for i,l in enumerate(self.created_null_curves):
            l.appeared=True # make sure it is said to true otherwise the null curves won't disappear
            l.disappear(begin_time=begin_time+i*delta,transition_time=half,**kwargs)
        return begin_time + transition_time

    def change_color(self, new_color, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        if isinstance(new_color, list):
            for letter, color in zip(self.letters, new_color):
                letter.change_color(new_color=color, begin_time=begin_time, transition_time=transition_time)
        else:
            for letter in self.letters:
                letter.change_color(new_color=new_color, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def change_color_of_letters(self,indices,new_color,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for i in indices:
            self.change_color_of_letter(i,new_color=new_color,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time
    def change_color_of_letter(self, index, new_color, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.letters[index].change_color(new_color=new_color, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def change_color_with_dictionary(self, dict, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for key, value in dict.items():
            color = key
            if isinstance(value, list):
                for v in value:
                    for letter in self.letters:
                        if str(v) in letter.ref_obj.name:
                            letter.change_color(new_color=color, begin_time=begin_time, transition_time=transition_time)
                            break
            else:
                for letter in self.letters:
                    if str(value) in letter.ref_obj.name:
                        letter.change_color(new_color=color, begin_time=begin_time, transition_time=transition_time)
                        break
        return begin_time + transition_time

    ###########
    ## tools ##
    ###########

    def add_empties_around_splines(self):
        for l, letter in enumerate(self.letters):
            for spline in letter.ref_obj.data.splines:
                mi = [np.inf, np.inf, np.inf]
                ma = [-np.inf, -np.inf, -np.inf]
                for p in spline.bezier_points:
                    for i, v in enumerate(p.co):
                        if mi[i] > v:
                            mi[i] = v
                        if ma[i] < v:
                            ma[i] = v

                minimum = Vector(mi)
                maximum = Vector(ma)
                center = 0.5 * (minimum + maximum)
                scale = maximum - center

                bbox = EmptyCube(location=center + letter.ref_obj.location, scale=scale,
                                 name=letter.ref_obj.name + "_box_" + str(l))
                bbox.ref_obj.parent = self.ref_obj
                ibpy.link(bbox.ref_obj)

    def add_to_morph_chain(self, simple_tex_obj, src_letter_range, img_letter_range, rescale, shift, src_colors,
                           img_colors,
                           begin_time=0,
                           transition_time=OBJECT_APPEARANCE_TIME, keep_color=False,in_place=True):
        self.morph_obj_chain.append(
            [simple_tex_obj, src_letter_range, img_letter_range,
             rescale, shift, src_colors, img_colors, begin_time,
             transition_time, keep_color,in_place])

    def perform_morphing(self):

        if len(self.morph_obj_chain) == 1:
            self.morph_to(*self.morph_obj_chain[0])
            return

        # determine the maximum number of splines and points
        self.calculate_max_spline_and_point_number()
        # since it is not clear apriori, which spline will be morphed into which
        # simply all curves get the same number of splines and with the same number of points
        # equalize all curve objects to have the same number of splines and points

        morph_to_objs = [self]
        for morph_obj in self.morph_obj_chain:
            morph_to_objs.append(morph_obj[0])

        for obj in morph_to_objs:
            for curve in obj.letters:
                curve = curve.ref_obj
                equalize_spline_count(curve, self.max_spline_count)

        for obj in morph_to_objs:
            for curve in obj.letters:
                curve = curve.ref_obj
                add_points_to_curve_spline(curve, 'all', self.max_point_count)

        # for obj in morph_to_objs:
        #     for curve in obj.letters:
        #         curve = curve.ref_obj
        #         for index, spline in enumerate(ibpy.get_splines(curve)):
        #             print(index, ": ", len(spline.bezier_points))

        # perform morph to each element in the chain
        # this is used in DigitalNumber for instance

        for element in self.morph_obj_chain:
            self.morph_to(*element)

    def morph_to(self, simple_tex_object, src_letter_range, img_letter_range, rescale, shift,
                 color_src, color_img, begin_time=0,
                 transition_time=OBJECT_APPEARANCE_TIME, keep_color=False,in_place=False):
        """
        In this method the morph chain will be written to blender. It is important to
        first get the entire morph chain since the number of splines and points have to be held
        fix along the morph chain, and they can only be determined,
        once the entire chain is known
        :return:
        """
        if 'e_0' in self.ref_obj.name:
            print("morph with e")

        print(
            "morph to:" + simple_tex_object.ref_obj.name + " from range " + str(src_letter_range) + " to range " + str(
                img_letter_range))
        morph_chains = []
        if begin_time is not None:
            start_frame = begin_time * FRAME_RATE
            end_frame = (begin_time + transition_time) * FRAME_RATE
        else:
            start_frame = None


        from_curves = self.rendered_objects[src_letter_range[0]:src_letter_range[1]]
        to_curves = simple_tex_object.rendered_objects[img_letter_range[0]:img_letter_range[1]]

        initial = from_curves
        final = to_curves

        # look for the same letters in source and target and try to match them
        destinations = self.find_lazy_morph_plan(initial, final,in_place=in_place)
        destinations2 = self.find_bounding_box_morph_plan(initial, final)

        # fill destinations into destinations2
        for pos, d in enumerate(destinations2):
            if d is None:
                destinations2[pos] = destinations[pos]

        sources = []
        for j in range(len(final)):
            if j in destinations2:
                sources.append(destinations2.index(j))
            else:
                sources.append(None)

        # print('Destinations and sources before pairing:')
        # print(' Destinations', destinations2)
        # print(' Sources', sources)
        # print()
        #
        # print("  Adding curves to chains")
        for j, (cur, dest) in enumerate(zip(initial, destinations2)):
            if dest is not None:
                morph_chains.append([cur, final[dest]])
            else:
                k = j
                # curves without a destination will look forward to try to
                # pair with a curve that has no source, but won't jump past
                # other curves with destinations.
                while k < len(sources):
                    # Don't jump past a char with a destination
                    if k < len(destinations2):  # Doing this so the next line works
                        if destinations2[k] is not None: break
                    if sources[k] is None:
                        morph_chains.append([cur, final[k]])
                        sources[k] = j
                        dest = destinations2[j] = k
                        break
                    k += 1

        # print('Destinations and sources after dest -> source match:')
        # print(' Destinations', destinations2)
        # print(' Sources', sources)
        # print()
        # print_morph_chain(morph_chains)

        for j, (cur, src) in enumerate(zip(final, sources)):
            if src is None:
                k = j
                # curves without a source will look forward to try to
                # pair with a curve that has no source, but won't jump past
                # other curves with sources.
                # max_index = min(len(destinations), len(sources))
                while k < len(destinations2):
                    # Don't jump past a char with a destination

                    if k < len(sources):  # Doing this so the next line works
                        if sources[k] is not None:
                            break
                    if destinations2[k] is None:
                        morph_chains.append([initial[k], cur])
                        sources[j] = k
                        dest = destinations2[k] = j
                        break
                    k += 1
            # bpy.context.scene.update()

        # print('Destinations and sources after source -> dest match:')
        # print(' Destinations', destinations2)
        # print(' Sources', sources)
        # print()
        # print_morph_chain(morph_chains)
        #
        # print("  Adding null curves for destination-less curves")
        # If dest is still None after trying to pair it with a source,
        # just insert a zero-size curve for cur to morph to.
        # This section is pretty hacky
        for j, dest in enumerate(destinations2):
            if dest is None:
                cur = initial[j]
                if j > 0:
                    k = j
                    while k >= len(final):
                        k -= 1
                    loc_cur = final[k]
                else:
                    loc_cur = final[j]

                null_curve = new_null_curve(
                    parent=final[0].ref_obj.parent,
                    location=loc_cur.ref_obj.location,
                    rotation=loc_cur.ref_obj.rotation_euler
                    # reuse_object = self.reusable_empty_curve
                )
                self.created_null_curves.append(null_curve)
                morph_chains.append([cur, null_curve])
        # print_morph_chain(morph_chains)
        #
        # print("  Adding null curves for sourceless curves")
        # If sources[j] is still None after trying to pair final[j] with
        # a source, just insert a zero-size curve for final[j] to morph from.
        for j, src in enumerate(sources):
            if src is None:
                cur = final[j]
                if j > 0:
                    k = j
                    while k >= len(initial):
                        k -= 1
                    loc_cur = initial[k]
                else:
                    loc_cur = initial[j]

                # Make the null curve if i == 1, because that means the curve
                # to morph from is one that is actually rendered. Otherwise,
                # reuse the reusable empty curve.
                '''if i == 1:
                    reuse = None
                else:
                    pass'''
                # reuse = self.reusable_empty_curve
                # bpy.context.scene.update()
                null_curve = new_null_curve(
                    parent=initial[0].ref_obj.parent,
                    location=cur.ref_obj.location,
                    rotation=cur.ref_obj.rotation_euler
                    # reuse_object = reuse
                )
                self.created_null_curves.append(null_curve)
                data = null_curve.ref_obj.data
                data.extrude = 0.01
                data.fill_mode = 'BOTH'

                # colors = self.default_color # old version
                colors = color_img  # new version 2022-12-2022 to be able to set colors to newly created curves

                if isinstance(colors, list):
                    if src is None:
                        if len(colors) > j:
                            color = colors[j]
                        else:
                            color = colors[-1]
                    else:
                        color = colors[src + src_letter_range[0]]
                else:
                    color = colors

                apply_material(null_curve.ref_obj, color, emission=0.5)
                equalize_spline_count(null_curve.ref_obj, self.max_spline_count)
                add_points_to_curve_spline(null_curve.ref_obj, 'all', self.max_point_count)

                morph_chains.append([null_curve, cur])
        #
        # print_morph_chain(morph_chains)
        # print("  Okay, done with that chain")
        #
        # print("  Adding null curves to extend chains")

        # Print chain info
        # for i, chain in enumerate(morph_chains):
        #     print(
        #         "Chain " + str(i + 1) + " of " + str(len(morph_chains)) + \
        #         " which are each of length " + str(len(chain))
        #     )
        #     chain = [x.ref_obj.name for x in chain]
        #     print(chain)
        #
        # print_morph_chain(morph_chains)
        # process morph chain and insert key_frames
        morph_pairs = []

        for source, image in morph_chains:
            if not source in self.rendered_objects:
                # self.rendered_objects.append(source)
                ibpy.link(source.ref_obj)

            morph_pairs.append([source, image])

        for char1, char2 in morph_pairs:
            ref_char1 = char1.ref_obj
            ref_char2 = char2.ref_obj

            if keep_color:
                ref_char2.material_slots[0].material = ref_char1.material_slots[0].material

            # add final transformations
            char2.ref_obj.scale = rescale
            char2.ref_obj.location[0] += shift[0]
            char2.ref_obj.location[1] += shift[1]

            # print("Info before morphing: ")
            # print_curve_info(ref_char1)
            # here the actual morphing takes place
            self.add_morph_shape_keys(ref_char1, ref_char2)

            if start_frame is not None:
                # key frames
                # character location relative to parent
                # this ensures preservation of overall expression arrangement
                ibpy.insert_keyframe(ref_char1, "location", frame=start_frame)
                ref_char1.location = ref_char2.location
                ibpy.insert_keyframe(ref_char1, "location", end_frame)

                ibpy.insert_keyframe(ref_char1, "rotation_euler", frame=start_frame)
                ref_char1.rotation_euler = ref_char2.rotation_euler
                ibpy.insert_keyframe(ref_char1, "rotation_euler", end_frame)

                # shape keys
                eval_time = ref_char1.data.shape_keys.key_blocks[-2].frame
                ref_char1.data.shape_keys.eval_time = eval_time
                ref_char1.data.shape_keys.keyframe_insert(data_path='eval_time', frame=start_frame)

                eval_time = ref_char1.data.shape_keys.key_blocks[-1].frame
                ref_char1.data.shape_keys.eval_time = eval_time
                ref_char1.data.shape_keys.keyframe_insert(data_path='eval_time', frame=end_frame)
                ref_char1.data.shape_keys.eval_time = 0

                # compare colors of the objects
                if char1 in self.letters:
                    src_index = self.letters.index(char1)
                else:
                    src_index = -1

                if char2 in simple_tex_object.letters:
                    img_index = simple_tex_object.letters.index(char2)
                else:
                    img_index = -1

                if -1 < src_index < len(color_src):
                    color1 = color_src[src_index]
                else:
                    color1 = color_src[-1]

                if -1 < img_index < len(color_img):
                    color2 = color_img[img_index]
                else:
                    color2 = color_img[-1]

                if color1 != color2 and not keep_color:
                    # create color transition
                    material = char1.ref_obj.material_slots[0].material
                    dialer = ibpy.create_color_mixing(material, color1, color2)
                    dialer.default_value = 0
                    ibpy.insert_keyframe(dialer, 'default_value', frame=start_frame)
                    dialer.default_value = 1
                    ibpy.insert_keyframe(dialer, 'default_value', frame=end_frame)

    def find_bounding_box_morph_plan(self, expr1, expr2):
        target_bbs = []
        source_bbs = []
        destinations = []
        sources = []
        for t in expr2:
            target_bbs.append(ibpy.get_spline_bounding_box(t))
        for s in expr1:
            source_bbs.append(ibpy.get_spline_bounding_box(s))

        for source_bb in source_bbs:
            max_overlap = 0
            best_index = -1
            for t, target_bb in enumerate(target_bbs):
                overlap = target_bb.overlap(source_bb)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_index = t
            if best_index == -1:
                destinations.append(None)
            else:
                destinations.append(best_index)

        for target_bb in target_bbs:
            max_overlap = 0
            best_index = -1
            for s, source_bb in enumerate(source_bbs):
                overlap = source_bb.overlap(target_bb)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_index = s
            if best_index == -1:
                sources.append(None)
            else:
                sources.append(best_index)

        # erase destinations that are not confirmed by sources
        for pos, d in enumerate(destinations):  # source at pos is assigned for target d
            if d is not None and sources[d] != pos:  # target d is not confirmed for source at pos
                destinations[pos] = None

        return destinations

    def find_lazy_morph_plan(self, expr1, expr2, min_length=None,in_place=False):
        # max length of substring we bother keeping
        # Increments if shared is still too long
        if min_length is None:
            min_length = self.min_length  # Default = 1

        if in_place:
            max_shared = 0
        else:
            max_shared = 10  # 8! is 40320

        shared = get_shared_substrings(expr1, expr2)

        for i in range(len(shared)):
            if shared[-i][2] < min_length:
                shared[-i] = None
        shared = [sub for sub in shared if sub is not None]

        while len(shared) > max_shared:
            min_length += 1
            removed = 0
            for i in range(len(shared)):
                if len(shared) - removed <= max_shared:
                    break
                if shared[-i][2] <= min_length:
                    shared[-i] = None
                    removed += 1

            shared = [sub for sub in shared if sub is not None]
        # raise Warning("Shit's cray")
        combos = get_substring_combos(shared)

        best_option = [[0, 0, 0]]
        highest_total = 0
        for combo in combos:
            total = 0
            for substring in combo:
                total += substring[2] ** 2
            if total > highest_total:
                highest_total = total
                best_option = combo

        destinations = []
        for j in range(len(expr1)):
            destination = None
            for plan in best_option:
                if j in range(plan[0], plan[0] + plan[2]):
                    destination = j + plan[1] - plan[0]

            destinations.append(destination)

        # print(best_option)
        # print("Here's the plan:")
        # print(destinations)

        return destinations

    ############################################
    # Functions for importing svg into blender #
    ############################################

    def get_file_path(self, expression):
        # Replaces the svg_b_object method
        if self.text_only:
            template = deepcopy(TEMPLATE_TEXT_FILE)
        else:
            template = deepcopy(TEMPLATE_TEX_FILE)
        if self.typeface != 'default':
            template = template[:-10]  # chop off the _arial.tex
            template += '_' + self.typeface + '.tex'
            if not os.path.exists(template):
                raise Warning(r'Can\'t find template tex file for that font.')

        self.path = tex_to_svg_file(expression, template, self.typeface, self.text_only,self.recreate)

    def open_path_object_from(self, b_obj, re_index=True, index=0):
        ref = b_obj.ref_obj
        curve_copy = ref.copy()
        curve_copy.data = ref.data.copy()

        # split unique identifier
        name = ref.name[ref.name.find('_') + 1:]
        b_obj = BObject(obj=curve_copy, name='hw_' + str(index) + "_" + name)  # name='hand_writing_of_'+ref.name)
        b_obj.ref_obj.location = ref.location
        b_obj.ref_obj.parent = b_obj.ref_obj.parent
        b_obj.ref_obj.rotation_euler = b_obj.ref_obj.rotation_euler

        if re_index:
            reindex_to_useful_writing_start(b_obj.ref_obj)

        return b_obj

    def closed_path_object_from(self, b_obj, re_index=True, index=0):
        ref = b_obj.ref_obj
        curve_copy = ref.copy()
        curve_copy.data = ref.data.copy()

        # split unique identifier
        name = ref.name[ref.name.find('_') + 1:]
        b_obj = BObject(obj=curve_copy, name='outline_' + str(index) + "_" + name)
        b_obj.ref_obj.location = ref.location
        b_obj.ref_obj.parent = b_obj.ref_obj.parent
        b_obj.ref_obj.rotation_euler = b_obj.ref_obj.rotation_euler

        if re_index:
            reindex_to_useful_writing_start(b_obj.ref_obj)

        return b_obj

    def get_figure_curves(self, fig):
        """
        returns the imported svg data without the leading 'H' that was added for alignment vertically
        :param fig:
        :return:
        """
        if fig is None:
            return self.imported_svg_data[fig]['curves']
        else:
            return self.imported_svg_data[fig]['curves'][1:]

    def align_figure(self, fig):
        curve_list = super().align_figure(fig)
        # Get rid of reference H after alignment is done
        if fig is None:
            self.imported_svg_data[fig]['curves'] = curve_list
        else:
            self.imported_svg_data[fig]['curves'] = curve_list[1:]

    def add_morph_shape_keys(self, initial, final):
        """
        :param initial: blender object
        :param final: blender object
        :return:
        """
        equalize_spline_and_point_count(initial, final)
        # find splines that are mapped together according to the length ranking
        # This might be a bit confusing, caused by the fact that I mixed up
        # length rank and index in my original names and implementation.
        # Could probably reimplement or at least change names.
        # TODO: This should be improved. Not only length but also the center of mass location should be considered
        initial_spline_length_ranks = get_list_of_spline_length_ranks(initial)
        final_spline_length_ranks = get_list_of_spline_length_ranks(final)

        # equalize the points on the corresponding splines
        # this has to be done before the shape keys are added

        for i in range(len(initial.data.splines)):
            initial_points = initial.data.splines[i].bezier_points
            initial_length_rank = initial_spline_length_ranks[i]
            final_index = final_spline_length_ranks.index(initial_length_rank)
            final_points = final.data.splines[final_index].bezier_points
            # Assign final_points values to initial_points
            if len(initial_points) != len(final_points):
                equalize_point_count(initial, i, final, final_index)

        was_hidden = False
        if ibpy.is_hidden(initial):
            was_hidden = True
        ibpy.un_hide(initial)
        ibpy.set_active(initial)
        ibpy.set_object_mode()
        # If absolute shape keys exist, set eval_time to zero
        try:
            initial.data.shape_keys.eval_time = 0
        except:
            pass
        # basis shape key
        bpy.ops.object.shape_key_add(from_mix=False)
        initial.data.shape_keys.use_relative = False
        # For some reason, the default 'CARDINAL' interpolation setting caused
        # bouncing, which would occasionally enlarge splines that should have
        # been size zero, messing with the fill.
        initial.data.shape_keys.key_blocks[-1].interpolation = 'KEY_LINEAR'
        # bpy.ops.object.shape_key_retime()

        # If there's only one shape key, it's the basis shape key.
        if len(initial.data.shape_keys.key_blocks) == 1:
            # We should add another shape key, which will get a keyframe
            bpy.ops.object.shape_key_add(from_mix=False)
            initial.data.shape_keys.key_blocks[-1].interpolation = 'KEY_LINEAR'
            # initial.data.shape_keys.use_relative = False
            # bpy.ops.object.shape_key_retime()

        ibpy.set_edit_mode()

        for i in range(len(initial.data.splines)):
            initial_points = initial.data.splines[i].bezier_points
            initial_length_rank = initial_spline_length_ranks[i]
            final_index = final_spline_length_ranks.index(initial_length_rank)
            final_points = final.data.splines[final_index].bezier_points

            if self.reindex_points_before_morph:
                reindex_to_the_least_deviation(initial_points, final_points)

            for j in range(len(initial_points)):
                # print(str(initial_points[j].co) + "->" + str(final_points[j].co))
                initial_points[j].co = final_points[j].co
                initial_points[j].handle_left = final_points[j].handle_left
                initial_points[j].handle_right = final_points[j].handle_right

        ibpy.set_object_mode()
        if was_hidden:
            ibpy.hide(initial)

    def calculate_max_spline_and_point_number(self):
        """
        simple walk through all splines
        find the maximum number of splines in one curve
        find the maximum number of bezier points in one spline
        :return:
        """
        self.max_spline_count = 0
        self.max_point_count = 0

        morph_to_objs = [self]
        for morph_obj in self.morph_obj_chain:
            morph_to_objs.append(morph_obj[0])
        for obj in morph_to_objs:
            for curve in obj.letters:
                curve = curve.ref_obj
                splines = ibpy.get_splines(curve)
                self.max_spline_count = np.maximum(self.max_spline_count, len(splines))
                for spline in splines:
                    self.max_point_count = np.maximum(self.max_point_count, len(ibpy.get_bezier_points(spline)))

class MultiLineTexBObject(SimpleTexBObject):
    """
    multiple lines in one BObject,
    try to make it as light_weighted as possible no writing just fading in and fading out lines of text
    """

    def __init__(self, expression, **kwargs):
        self.kwargs = kwargs
        self.expression = expression
        self.centered = self.get_from_kwargs('centered', False)
        self.typeface = self.get_from_kwargs('typeface', 'default')
        self.text_only = self.get_from_kwargs('text_only', False)
        if 'rotation_quaternion' in kwargs:
            pass
        elif 'rotation_euler' not in kwargs:
            kwargs['rotation_euler'] = [np.pi / 2, 0, 0]
            # set default rotation

        self.color = self.get_from_kwargs('color', ['text'])
        self.brighter = self.get_from_kwargs('brighter', 0)
        self.emission = self.get_from_kwargs('emission', 0.5)

        self.bevel = self.get_from_kwargs('bevel', 0)
        self.bevel *= 0.005
        self.min_length = 1  # don't know what this is for yet

        if 'vert_align_centers' not in kwargs:
            kwargs['vert_align_centers'] = False
        self.name = self.get_from_kwargs('name', 'tex')
        self.shadow = self.get_from_kwargs('shadow', True)  # catch shadow property to apply it to letters individually
        n_lines = self.count_number_of_lines()
        super().__init__(expression, color=self.color, name=self.name, number_of_lines=n_lines, **kwargs)

        self.annotations = []
        self.letters = self.rendered_objects[0:]
        self.lines = []
        self.group_in_lines(n_lines)

        # create parent child relation for the letters
        self.thickness = self.get_from_kwargs('thickness', 1)
        self.thickness *= 0.01

        for letter in self.letters:
            letter.ref_obj.parent = self.ref_obj
            ibpy.set_extrude(letter, self.thickness)

        if not self.shadow:
            for c in self.ref_obj.children:
                bpy.data.objects[c.name].visible_shadow = False

    #######################
    # animation functions #
    #######################

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        if not self.appeared:
            ibpy.link(self)
            self.appeared = True
        t0 = begin_time
        if len(self.lines) > 0:
            dt = transition_time / len(self.lines)
            for line in self.lines:
                if len(line) > 0:
                    ddt = dt / len(line)
                    for letter in line:
                        letter.appear(begin_time=t0, transition_time=ddt, **kwargs)
                        t0 += ddt

    def write_line(self,line_index,letters=None,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        if not self.appeared:
            ibpy.link(self)
        line=self.lines[line_index]
        if len(line)>0:
            # sort letters from left to right
            line.sort(key=lambda x:x.ref_obj.location[0])
            if letters is not None:
                selected = [line[i] for i in letters]
            else:
                selected = line
            self.write(letters=selected,begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def appear_line(self, line_index, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        if not self.appeared:
            ibpy.link(self)
            self.appeared = True
        t0 = begin_time
        line = self.lines[line_index]
        if len(line) > 0:
            dt = transition_time / len(line)
            for letter in line:
                letter.appear(begin_time=t0, transition_time=dt, **kwargs)
                t0 += dt
        return begin_time+transition_time

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        t0 = begin_time
        if len(self.lines) > 0:
            dt = transition_time / len(self.lines)
            for line in self.lines:
                if len(line) > 0:
                    ddt = dt / len(line)
                    for letter in line:
                        letter.disappear(alpha=alpha, begin_time=t0, transition_time=ddt, **kwargs)
                        t0 += ddt

    def disappear_line(self, line_index, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        t0 = begin_time
        line = self.lines[line_index]
        if len(line) > 0:
            dt = transition_time / len(line)
            for letter in line:
                letter.disappear(alpha, begin_time=t0, transition_time=dt, **kwargs)
                t0 += dt

    def disappear_line_at_once(self, line_index, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
                               **kwargs):
        t0 = begin_time
        line = self.lines[line_index]

        for letter in line:
            letter.disappear(alpha, begin_time=t0, transition_time=transition_time, quick=True, **kwargs)

    #######################
    # auxiliary functions #
    #######################

    def find_gaps(self):
        both = []

        for letter in self.letters:
            x_min, x_max, y_min, y_max, z_min, z_max = ibpy.analyse_bound_box(letter)
            both.append((y_max, ')'))
            both.append((y_min, '('))

        both.sort(key=lambda x: x[0])
        stack = []
        gaps = []
        end = True
        for (key, value) in both:
            if value == '(':
                if end:
                    gaps.append(key)
                    end = False
                stack.append(value)
            else:
                stack.pop()
                if len(stack) == 0:
                    gaps.append(key)
                    end = True
        return gaps

    def group_in_lines(self, n):
        """
        arrange letters in lines according to their bounding boxes
        the full bounding box is divided into n regions and the letters are distributed according to their center of
        their individual bounding boxes

        param n:
        return:
        """

        y_bins = self.find_gaps()

        # create n empty lines
        for i in range(n):
            self.lines.append([])

        for letter in self.letters:
            x_min, x_max, y_min, y_max, z_min, z_max = ibpy.analyse_bound_box(letter)
            for i in range(n - 1, -1, -1):  # look in reversed order, since the upper lines have higher y-values
                y_center = (y_max + y_min) / 2
                if y_bins[2 * i + 1] > y_center > y_bins[2 * i]: #(change made 2023-10-09 penrose video)
                # if y_bins[i + 1] > y_center > y_bins[ i]:
                    self.lines[n - i - 1].append(letter)

    def count_number_of_lines(self):
        lines = self.expression.split(r"\\")
        return len(lines)

    def get_file_path(self, expression):
        # Replaces the svg_b_object method
        if self.text_only:
            template = deepcopy(TEMPLATE_TEXT_FILE)
        else:
            template = deepcopy(TEMPLATE_TEX_FILE)
        if self.typeface != 'default':
            template = template[:-4]  # chop off the .tex
            template += '_' + self.typeface + '.tex'
            if not os.path.exists(template):
                raise Warning(r'Can\'t find template tex file for that font.')

        self.path = tex_to_svg_file(expression, template, self.typeface, self.text_only,self.recreate)


class Row:
    def __init__(self, letter, index):
        self.indices = [index]
        bb = ibpy.get_bounding_box_for_letter(letter)
        self.min = bb[1]
        self.max = bb[4]

    def contains(self, letter):
        bb = ibpy.get_bounding_box_for_letter(letter)
        center = 0.5 * (bb[1] + bb[4])
        if self.min <= center <= self.max:
            return True
        else:
            return False

    def insert(self, letter, index):
        self.indices.append(index)
        bb = ibpy.get_bounding_box_for_letter(letter)
        if self.min > bb[1]:
            self.min = bb[1]
        if self.max < bb[4]:
            self.max = bb[4]


#######################
# static functions    #
#######################


def tex_to_svg_file(expression, template_tex_file, typeface, text_only,recreate=False):
    path = os.path.join(
        SVG_DIR,
        # tex_title(expression, typeface)
        hashed_tex(expression, typeface)
    ) + ".svg"
    if not recreate and  os.path.exists(path):
        return path

    tex_file = generate_tex_file(expression, template_tex_file, typeface, text_only,recreate)
    dvi_file = tex_to_dvi(tex_file,recreate)
    return dvi_to_svg(dvi_file,recreate)


def get_null():
    if os.name == "nt":
        return "NUL"
    return "/dev/null"


def dvi_to_svg(dvi_file,recreate):
    """
    Converts a dvi, which potentially has multiple slides, into a
    directory full of enumerated svgs corresponding with these slides.
    Returns a list of PIL Image objects for these images sorted as they
    where in the dvi
    """

    result = dvi_file.replace(".dvi", ".svg")
    result = result.replace("tex", "svg")  # change directory for the svg file
    print('svg: ', result)
    if recreate or not os.path.exists(result):
        commands = [
            "dvisvgm",
            dvi_file,
            "-n",
            "-v",
            "3",
            "-o",
            result
            # Not sure what these are for, and it seems to work without them
            # so commenting out for now
            # ,
            # ">",
            # get_null()
        ]
        os.system(" ".join(commands))
    return result


def tex_title(expression, typeface):
    """
       bobject that regularizes the latex content to get but most likely unique filenames
            | '\' -> '_bs_'
            | '(' -> '_rob_'
            | ')' -> '_rcb_'
            | '^' -> '_**_'
            | '{' -> '_cob_'
            | '}' -> '_ccb_'
            | '&' -> '_amp_'
            | '$' -> '_dol_'
            | '^' ->'_cflex_'
            | ',' ->'_c_'
       :param typeface:
       :param expression:
       :return:
    """

    name = expression
    to_delete = ['/', '~', '\'', '\"', ' ']
    # Replace these rather than deleting them. These are characters that I've
    # wanted as lone expressions. (Which are also off limits in file names)
    to_replace = {
        '<': 'lessthan',
        '>': 'greaterthan',
        '?': 'questionmark',
        '.': 'point',
        ':': 'colon',
        '%': 'percent',
        '|': 'vbar',
        '\\': '_bs_',
        '(': '_rob_',
        ')': '_rcb_',
        '{': '_cob_',
        '}': '_ccb_',
        '^': '_**_',
        '=': '_eq_',
        '&': '_amp_',
        '$': '_dol_',
        '^': '_cflex_',
        ',': '_c_',
    }
    for char in name:
        if char in to_delete:
            name = name.replace(char, "")
    for char in name:
        if char in to_replace.keys():
            name = name.replace(char, to_replace[char])
    # name = str(name) + '_'
    if typeface != 'default':
        name += '_' + typeface

    if len(name) > 200:  # make sure that the filename doesn't get too long
        name = name[0:200]
    return str(name)

def hashed_tex(expression, typeface):
    string = expression + typeface
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]

def generate_tex_file(expression, template_tex_file, typeface, text_only,recreate):
    result = os.path.join(
        TEX_DIR,
        # tex_title(expression, typeface)
        hashed_tex(expression, typeface)
    ) + ".tex"

    if recreate or not os.path.exists(result):
        print("Writing \"%s\" to %s" % (
            "".join(expression), result
        ))
        with open(template_tex_file, "r") as infile:
            body = infile.read()
            # I add an H to every expression to give a common reference point
            # for all expressions, then hide the H character. This is necessary
            # for consistent alignment of tex curves in blender, because
            # blender's import svg bobject sets the object's origin depending
            # on the expression itself, not according to a typesetting reference
            # frame.
            if text_only:
                expression = 'H ' + expression
            else:
                expression = '\\text{H} ' + expression
            body = body.replace(TEX_TEXT_TO_REPLACE, expression)
        with open(result, "w") as outfile:
            outfile.write(body)
    return result


def tex_to_dvi(tex_file,recreate):
    result = tex_file.replace(".tex", ".dvi")
    if recreate or not os.path.exists(result):
        commands = [
            "latex",
            "-interaction=batchmode",
            "-halt-on-error",
            "-output-directory=" + TEX_DIR,
            tex_file  # ,
            # ">",
            # get_null()
        ]
        exit_code = os.system(" ".join(commands))
        if exit_code != 0:
            latex_output = ''
            log_file = tex_file.replace(".tex", ".log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    latex_output = f.read()
            raise Exception(
                "Latex error converting to dvi. "
                "See log output above or the log file: %s" % log_file)
    return result


###########################
# Auxiliaries for morphing #
###########################


def reindex_to_the_least_deviation(source_points, target_points):
    """
    the order of the target points are cycled through (including a change in orientation)
    until the least current_deviation to the source points is found
    The configuration of least current_deviation is implemented in the target points

    :param target_points:
    :param source_points:
    :param target:
    :return:
    """

    smallest_deviation = np.inf
    smallest_index = 0
    flip = False

    for i in range(len(target_points)):
        dev = deviation(source_points, target_points, shift=i)
        if dev < smallest_deviation:
            smallest_index = i
            smallest_deviation = dev
            # print(smallest_deviation, " ", smallest_index)

    for i in range(len(target_points)):
        dev = deviation(source_points, target_points, shift=i, flip=True)
        if dev < smallest_deviation:
            smallest_index = i
            smallest_deviation = dev
            # print(smallest_deviation, " ", smallest_index, " ", True)
            flip = True

    # copy point data to lists
    positions = []
    left_handles = []
    right_handles = []
    for point in target_points:
        positions.append(deepcopy(point.co))
        left_handles.append(deepcopy(point.handle_left))
        right_handles.append(deepcopy(point.handle_right))

    # re-index copied lists, cycle through until index_highest is the first point
    for i in range(smallest_index):
        positions.append(positions.pop(0))
        left_handles.append(left_handles.pop(0))
        right_handles.append(right_handles.pop(0))

    if not flip:
        # assign values to blender bezier points
        for i in range(len(target_points)):
            target_points[i].co = positions[i]
            target_points[i].handle_left = left_handles[i]
            target_points[i].handle_right = right_handles[i]
    else:
        for i in range(len(target_points)):
            target_points[i].co = positions[-i]
            target_points[i].handle_right = left_handles[-i]
            target_points[i].handle_left = right_handles[-i]

    return flip


def deviation(source_points, target_points, shift=0, flip=False):
    dev = 0
    n = len(source_points)
    m = len(target_points)
    if n != m:
        raise "different length of source and target points, no morphing is possible"
    for i in range(n):
        src_point = source_points[i]
        if not flip:
            target_point = target_points[(i + shift) % n]
        else:
            target_point = target_points[(-i + shift) % n]
        dev += (src_point.co - target_point.co).length
    return dev


def reindex_to_useful_writing_start(obj):
    """
    Writing usually starts at a upper point of the the letter with high curvature
    Look for the smallest gaps (less than 10 percent of the largest gap) and take the one that is highest (highest y coordinate)
    as starting point
    :param obj:
    :return:
    """

    for spline in obj.data.splines:
        points = spline.bezier_points
        smallest_gap = np.inf
        largest_gap = 0

        for i in range(len(points)):
            a = points[i - 1].co
            b = points[i].co
            l = (a - b).length
            if smallest_gap > l:
                smallest_gap = l
            if largest_gap < l:
                largest_gap = l

        # print(smallest_gap, " ", largest_gap, " ", largest_gap / smallest_gap)

        # find all indices that have gaps to the previous in the range of
        #  smallest_gap to 0.1 * largest_gap

        list_of_indices = []

        for i in range(len(points)):
            a = points[i - 1].co
            b = points[i].co
            l = (a - b).length
            if smallest_gap <= l <= 0.1 * largest_gap:
                list_of_indices.append(i)

        # print(list_of_indices)

        # select index with highest y-coordinate

        highest_y = 0
        selected_index = 0
        for index in list_of_indices:
            if points[index].co[1] > highest_y:
                selected_index = index

        # print(selected_index)

        # start anti-clockwise at the selected index
        positions = []
        left_handles = []
        right_handles = []
        for point in points:
            positions.append(deepcopy(point.co))
            left_handles.append(deepcopy(point.handle_left))
            right_handles.append(deepcopy(point.handle_right))

        # cyclic rearrangement and the anti-clock-wise drawing is converted into
        # clock-wise drawing, which is more intuitive. Therefore, left and right handles have
        # to interchange
        shift = len(points) - selected_index
        for i in range(len(points)):
            points[i].co = positions[-shift + i]
            points[i].handle_left = left_handles[-shift + i]
            points[i].handle_right = right_handles[-shift + i]


def reindex_to_smallest_gap(obj):
    """
    the order of bezier points is cycled in such a way that the smallest length gap is between the
    first and the last point. In this way the curve starts drawing at the sharpest edge normally
    :param obj:
    :return:
    """

    for spline in obj.data.splines:
        # Make it so the highest control point is at index 0
        # This eliminates net rotation of points around the curve as they transition
        # from the starting char to the target char
        # Rotation would be fine, but they actually just go in a straight line,
        # causing the curve to sometimes fold on itself
        points = spline.bezier_points
        index_of_smallest_gap = 0
        gap = np.inf
        for i in range(len(points)):
            a = points[i - 1].co
            b = points[i].co
            l = (a - b).length
            if gap > l:
                gap = l
                index_of_smallest_gap = i

        positions = []
        left_handles = []
        right_handles = []
        for point in points:
            positions.append(deepcopy(point.co))
            left_handles.append(deepcopy(point.handle_left))
            right_handles.append(deepcopy(point.handle_right))

        # cyclic rearrangement and the anti-clock-wise drawing is converted into
        # clock-wise drawing, which is more intuitive. Therefore, left and right handles have
        # to interchange
        for i in range(len(points)):
            points[i].co = positions[index_of_smallest_gap - 1 - i]
            points[i].handle_right = left_handles[index_of_smallest_gap - 1 - i]
            points[i].handle_left = right_handles[index_of_smallest_gap - 1 - i]


def reindex_to_top_point(spline):
    # Make it so the highest control point is at index 0
    # This eliminates net rotation of points around the curve as they transition
    # from the starting char to the target char
    # Rotation would be fine, but they actually just go in a straight line,
    # causing the curve to sometimes fold on itself
    points = spline.bezier_points
    # Find index of highest point in curve
    index_highest = 0
    for i in range(len(points)):
        if points[i].co[1] > points[index_highest].co[1]:  # Compare y values
            index_highest = i
    # copy point data to lists
    positions = []
    left_handles = []
    right_handles = []
    for point in points:
        positions.append(deepcopy(point.co))
        left_handles.append(deepcopy(point.handle_left))
        right_handles.append(deepcopy(point.handle_right))

    # re-index copied lists, cycle through until index_highest is the first point
    for i in range(index_highest):
        positions.append(positions.pop(0))
        left_handles.append(left_handles.pop(0))
        right_handles.append(right_handles.pop(0))

    # assign values to blender bezier points
    for i in range(len(points)):
        points[i].co = positions[i]
        points[i].handle_left = left_handles[i]
        points[i].handle_right = right_handles[i]


def get_shared_substrings(expr1, expr2):
    # not actually strings, but a series of curves that represent letters, mostly
    curves1 = expr1
    curves2 = expr2

    shared = []
    for i in range(len(curves1)):
        j = 0
        for j in range(len(curves2)):
            length = 0
            length = get_match_length(length, i, j, curves1, curves2)

            if length > 0:
                candidate = [i, j, length]

                # Check whether candidate is redundant with a substring we
                # already found. E.g., without this, comparing '01' with '012'
                # would find the '01' and '1' substrings. We just want the longer
                # one.
                redundant = False
                '''
                #Actually, think we want redundancy, at least until speed becomes
                #an issue. Without redundancy, morphing '123' to '1223' would
                #result in the shared '3' being discarded, since it's redundant
                #with the shared '23'. This is usually good, but because the
                #original '2' is in two shared substrings, one (the '23') is
                #discarded. In this case, the '3' won't be captured, even though
                #it's not redundant with any substring that actually gets used.
                #The truly redundant substrings will get tossed later when
                #choosing the highest-scoring set to actually morph.
                #Only smaller redundant substrings toward the right of larger
                #substrings will be preserved. That's okay, because when
                #substrings overlap, the left-most overlapping string is used.
                #Since the left-most strings are never tossed, no redundancy is
                #needed for backup.
                '''
                '''
                Aaaaaaaactually, fuck redundancy. Things got slow.
                '''

                for substring in shared:
                    start1_diff = candidate[0] - substring[0]
                    start2_diff = candidate[1] - substring[1]
                    length_diff = candidate[2] - substring[2]
                    if start1_diff == start2_diff == -length_diff:
                        redundant = True

                if redundant is False:
                    shared.append(candidate)

    return shared


def are_chars_same(char1, char2):
    '''
    Compare to characters on the level of splines
    :param char1:
    :param char2:
    :return:
    '''
    splines1 = char1.data.splines
    splines2 = char2.data.splines
    if len(splines1) != len(splines2):
        return False
    for spline1, spline2 in zip(splines1, splines2):
        points1 = spline1.bezier_points
        points2 = spline2.bezier_points
        for point1, point2 in zip(points1, points2):
            for coord1, coord2 in zip(point1.co, point2.co):
                if round(coord1, 3) == round(coord2, 3):
                    # When the svg is imported, coords are stored to many decimal
                    # points. Even in characters we'd call equivalent, there is
                    # some fluctuation in the less significant digits, so
                    # rounding here yields the desired behavior.
                    pass
                else:
                    return False
    return True


def get_match_length(length, char1_index, char2_index, curves1, curves2):
    if are_chars_same(curves1[char1_index].ref_obj, curves2[char2_index].ref_obj):
        length += 1
        char1_index += 1
        char2_index += 1

        try:
            length = get_match_length(length, char1_index, char2_index, curves1, curves2)

            return length
        except:
            return length
    else:
        if length > 0:
            pass
        return length


def get_substring_combos(substrings):
    combos = []
    combo_in_progress = []
    combos = add_non_overlapping_substrings(combo_in_progress, combos, substrings)

    return combos


def add_non_overlapping_substrings(combo_in_progress, combos, substrings):
    if len(combo_in_progress) > 0:
        # Start checking substrings with the one after the last one added.
        starting_index = substrings.index(combo_in_progress[-1]) + 1
        # starting_index = 0
    else:
        starting_index = 0

    for i in range(starting_index, len(substrings)):
        # check if substring works
        candidate = substrings[i]
        no_overlap = True
        # check if substring overlaps with any substring alredy
        # in combo_in_progress. If so, don't add it to combos.
        for sub in combo_in_progress:
            # E.g., sub = [0, 0, 1] and candidate = [3, 0, 1] overlap
            no_overlap_in_1 = candidate[0] >= sub[0] + sub[2] or \
                              candidate[0] + candidate[2] <= sub[0]
            no_overlap_in_2 = candidate[1] >= sub[1] + sub[2] or \
                              candidate[1] + candidate[2] <= sub[1]

            no_overlap = (no_overlap_in_1 and no_overlap_in_2)

            if not no_overlap:
                break

        if no_overlap:
            new_combo = deepcopy(combo_in_progress)
            new_combo.append(candidate)
            combos.append(new_combo)
            combos = add_non_overlapping_substrings(new_combo, combos, substrings)

    return combos


def add_points_to_curve_spline(
        curve,
        index,
        total_points=CONTROL_POINTS_PER_SPLINE,
        closed_loop=True
):
    """
    adds a total_points number of points to the spline curve.data.splines[index]
    :param curve:
    :param index:
    :param total_points:
    :param closed_loop:
    :return:
    """
    was_hidden = False
    if ibpy.is_hidden(curve):
        was_hidden = True
    ibpy.un_hide(curve)
    previously_active = ibpy.get_active()
    if previously_active:
        previous_mode = previously_active.mode
    else:
        previous_mode = None
    is_linked = ibpy.is_linked(curve)
    if not is_linked:
        ibpy.link(curve)
    ibpy.set_active(curve)
    ibpy.set_edit_mode()

    if index == 'all':
        for spline in curve.data.splines:
            ibpy.add_points_to_spline(spline, total_points, closed_loop)
    else:
        ibpy.add_points_to_spline(curve.data.splines[index], total_points, closed_loop)

    if previous_mode == 'OBJECT':
        ibpy.set_object_mode()
    if previously_active:
        ibpy.set_active(previously_active)
    if was_hidden:
        ibpy.hide(curve)
    if not is_linked:
        ibpy.un_link(curve)


def get_list_of_spline_length_ranks(curve):
    splines = curve.data.splines
    curve_splines_ranked_by_length = []

    # get a list of splines and sort them by length
    # we have to do this because 'splines' is a bpy_prop_collection, not a list
    # meaning it doesn't have list methods.
    for spline in splines:
        curve_splines_ranked_by_length.append(spline)
    curve_splines_ranked_by_length.sort(key=lambda x: get_spline_length(x), \
                                        reverse=True)

    list_of_length_ranks = []
    for spline in splines:
        rank = curve_splines_ranked_by_length.index(spline)
        list_of_length_ranks.append(rank)

    return list_of_length_ranks


def get_spline_length(spline):
    points = spline.bezier_points
    length = 0
    for j in range(len(points)):
        k = (j + 1) % len(points)
        sep = points[k].co - points[j].co
        length += sep.length
    return length


def equalize_spline_and_point_count(one, two):
    """
    this bobject adds splines or/and points to either one or two in such a way that both of them have the
    same number of splines and in each spline the same number of points

    :param one:
    :param two:
    :return:
    """
    if len(one.data.splines) == len(two.data.splines):
        # print("no adjustment was necessary, curves had the same number of splines")
        return
    elif len(one.data.splines) < len(two.data.splines):
        equalize_spline_count(one, len(two.data.splines))
    elif len(one.data.splines) > len(two.data.splines):
        equalize_spline_count(two, len(one.data.splines))

    # print("number of splines was adjusted for ", one, " and ", two,
    #       ". They now have ", len(one.data.splines),
    #       " splines. The difference is ",
    #       len(one.data.splines) - len(two.data.splines))


def equalize_point_count(curve1, index1, curve2, index2):
    spline1 = curve1.data.splines[index1]
    spline2 = curve2.data.splines[index2]
    if len(spline1.bezier_points) < len(spline2.bezier_points):
        add_points_to_curve_spline(curve1, index1, len(spline2.bezier_points))
    elif len(spline1.bezier_points) > len(spline2.bezier_points):
        ibpy.link(curve2)
        add_points_to_curve_spline(curve2, index2, len(spline1.bezier_points))
        ibpy.un_link(curve2)


def print_curve_info(obj):
    for i, spline in enumerate(obj.data.splines):
        print("information for spline ", i)
        for point in spline.bezier_points:
            print(point.co[0], ",", point.co[1])


def print_morph_chain(chain):
    for c in chain:
        print(c[0].ref_obj.name, "->", c[1].ref_obj.name)
