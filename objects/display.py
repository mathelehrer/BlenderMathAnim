import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.plane import Plane
from objects.tex_bobject import SimpleTexBObject, TexBObject
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME, BACK, DOWN, UP


class Display(Plane):
    """
    Create a plane with a glass like color that is used to write text on it
    """

    def __init__(self, location=[12, 0, 0],
                 scales=[4, 6],
                 rotation_euler=[np.pi / 2, 0, 0],
                 number_of_lines=5, columns=1,
                 shadow=True, show_lines=False,
                 roughness=1, apply_scale=False,
                 apply_location=False, **kwargs):
        """

        :param location:
        :param scales:
        :param rotation_euler:
        :param number_of_lines:
        :param columns:
        :param flat:
        :param shadow:
        :param show_lines:
        :param kwargs:
        """

        self.kwargs = kwargs
        self.flat = self.get_from_kwargs('flat', False)
        self.debugging = 0

        self.front_text = []
        self.back_text = []

        # the scale determines the extension of the display need for the font adjustment in get_scale
        self.scale_x = scales[0]
        self.scale_y = scales[1]
        if len(scales) == 3:
            self.scale_z = scales[2]
        else:
            self.scale_z = scales[0]

        # introduce manim like parameters for the display.
        # due to the rotation the z-direction and the y-direction are interchanged

        self.up = Vector([0, 0.167, 0])
        self.down = -self.up
        self.left = Vector([-0.2, 0, 0])
        self.right = -self.left

        # the line spaceing is enforced by the number of lines that have to fit
        # the font size is adjusted accordingly

        self.top = Vector([0, 1, 0])
        self.left_edge = Vector([-1, 0, 0])
        self.right_edge = Vector([1, 0, 0])
        self.bottom = Vector([0, -1, 0])

        self.line_spacing = 2 / number_of_lines * self.bottom
        self.number_of_lines = number_of_lines

        if self.debugging:
            print("right: ", self.right)
            print("top: ", self.top)
            print("left_edge: ", self.left_edge)
            print("right_edge: ", self.right_edge)
            print("scale_x: ", self.scale_x)
            print("scale_y: ", self.scale_y)

        self.title_line = self.top + 0.6 * self.line_spacing

        self.sep = Vector([0.03, 0, 0])

        self.text_start = []
        self.text_start_back = []
        self.cursor = []
        self.line = []

        for col in range(columns):
            self.text_start.append(self.title_line + self.line_spacing + self.left_edge + (
                    self.right_edge - self.left_edge) * col / columns)
            self.text_start_back.append(self.title_line + self.line_spacing + self.right_edge + (
                    self.left_edge - self.right_edge) * col / columns)
            self.cursor.append(self.text_start[-1].copy())
            self.line.append(0)

        if self.flat is True:
            if rotation_euler[2] != 0 and rotation_euler[2] != np.pi:
                rotation_euler[2] = 0
        else:
            if self.flat is False:
                rotation_euler[2] = -np.pi / 15
            else:
                rotation_euler[2] = -np.pi / 15 * self.flat

        name = 'Display'
        if 'name' in kwargs:
            name = kwargs['name']
            kwargs.pop('name')

        color = self.get_from_kwargs('color', 'background')

        # add scale to dictionary
        kwargs['scale'] = [self.scale_x, self.scale_y, self.scale_z]
        super().__init__(name=name,
                         location=location,
                         rotation_euler=rotation_euler,
                         color=color,
                         roughness=roughness,
                         apply_scale=apply_scale,
                         apply_location=apply_location,
                         **kwargs)

        plane = self.plane
        solid = self.get_from_kwargs('solid', 0.05)
        modifier_solid = plane.modifiers.new(name='solid', type='SOLIDIFY')
        modifier_solid.thickness = solid
        modifier_solid.offset = 0

        ibpy.set_shadow_of_object(self, shadow)
        self.show_lines = show_lines
        self.lines = []
        if show_lines:
            self.draw_lines()

        self.standard_shift = self.get_from_kwargs('standard_shift',
                                                   [0, 0, 0.025])  # shift that raises the text above the display

    def draw_lines(self):
        for i in range(int(self.number_of_lines)):
            cyl = Cylinder(length=2, cyl_radii=[0.1 / self.scale_x, 0.1 / self.scale_y],
                           location=self.top + (i + 1) * self.line_spacing, rotation_euler=[0, np.pi / 2, 0])
            self.add_child(cyl)
            self.lines.append(cyl)

    def add_bob(self, bob, line=0, indent=0, column=1, scale=1, shift=[0, 0, 0]):
        self.add_child(bob)
        # lift bob to make it appear on top of the display
        # bob.move(direction=[0,0,0.025],begin_time=0,transition_time=0)
        # compensate the rotation of the display
        bob.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=0, transition_time=0)
        # compensate the scale of the display
        if not isinstance(scale, list):
            scale = 3 * [scale]
        new_scale = [1 / x * y for x, y in zip(self.ref_obj.scale, scale)]
        bob.ref_obj.scale = new_scale
        location = self.text_start[column - 1] + line * self.line_spacing + self.right * indent + UP * shift[
            2] + Vector(
            [shift[0], shift[1], 0]) + Vector([0, 0, 0.025])
        bob.ref_obj.location = location

    def add_line(self, start=[0, 0], end=[0, 0], **kwargs):
        start_point = self.top + self.left_edge + start[0] * self.right + start[1] * self.down
        end_point = self.top + self.left_edge + end[0] * self.right + end[1] * self.down
        line = Cylinder.from_start_to_end(start=start_point, end=end_point, **kwargs)
        self.add_child(line)
        return line

    def get_scale(self, scale=1):
        """
        this is empirically adjusted to make the font scale properly with the size of the display and
        the number of lines as far as regular text is concerned

        if you ever change the size of the tex import, adjustments will be needed here as well
        :param scale:
        :return:
        """
        return scale * 1.25 * 5 / self.number_of_lines * Vector([self.scale_y / self.scale_x, 1, 1 / scale])

    def set_title(self, title, scale=1, shift=None):
        """
        define SimpleTexObject as title and put it at the designated location

        :param shift: the third component accounts for the thickness of the display
        :param scale:
        :param title:
        :return:
        """
        if shift is None:
            shift = self.standard_shift

        if len(shift) == 2:
            shift += [0.025]
        title.ref_obj.parent = self.ref_obj
        title.ref_obj.location = self.title_line + shift[0] * self.up + shift[1] * self.right + UP * shift[2]
        title.ref_obj.name = 'Title'
        title.ref_obj.rotation_euler = [0, 0, 0]
        title.ref_obj.scale = self.get_scale(scale)

    def write_title(self, title, scale=1, shift=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        if shift is None:
            shift = self.standard_shift
        self.set_title(title, scale=scale, shift=self.standard_shift)
        title.write(begin_time=begin_time, transition_time=transition_time, **kwargs)
        self.front_text.append(title)
        return begin_time + transition_time + 0.5

    def write_back_title(self, title, scale=1, shift=None, begin_time=0,
                         transition_time=DEFAULT_ANIMATION_TIME):
        if shift is None:
            shift = self.standard_shift
        self.set_title_back(title, scale=scale, shift=shift)
        title.write(begin_time=begin_time, transition_time=transition_time)
        self.back_text.append(title)
        return begin_time + transition_time + 0.5

    def set_title_back(self, title, scale=1, shift=None):
        """
        define SimpleTexObject as title and put it at the designated location

        :param shift:
        :param scale:
        :param title:
        :return:
        """
        if shift is None:
            shift = self.standard_shift

        if len(shift) == 2:
            shift += [0.025]
        title.ref_obj.parent = self.ref_obj
        title.ref_obj.location = self.title_line + shift[0] * self.up + shift[1] * self.left + DOWN * shift[2]
        title.ref_obj.name = 'Title'
        title.ref_obj.rotation_euler = [0, np.pi, 0]
        title.ref_obj.scale = self.get_scale(scale)

    def add_text(self, tex_object, scale=0.7, indent=None, column=1, location=None, shift=None):
        """
        :param column:
        :param tex_object:
        :param scale:
        :param indent:
        :return:
        """
        if shift is None:
            shift = self.standard_shift

        # reindex column from userfriendly to python friendly
        column -= 1
        adjusted_scale = self.get_scale(scale)
        # print("before adding: ",self.cursor)
        if indent is not None:
            self.cursor[column] += indent * self.right

        if isinstance(tex_object, TexBObject):
            for s, sub_tex_object in enumerate(tex_object.objects):
                sub_tex_object.ref_obj.parent = self.ref_obj
                sub_tex_object.ref_obj.rotation_euler = [0, 0, 0]
                sub_tex_object.ref_obj.location = self.cursor[column]
                sub_tex_object.set_scale(adjusted_scale)
                self.cursor[column] = self.cursor[column] + self.sep + sub_tex_object.length * Vector(
                    [scale, 0, 0]) + UP * 0.025
                print(self.cursor[column])
                if self.cursor[column].x > self.right.x:
                    self.set_cursor_to_start_of_next_line(indent=indent, column=column)
        elif isinstance(tex_object, SimpleTexBObject):
            # single object
            tex_object.ref_obj.parent = self.ref_obj
            tex_object.ref_obj.rotation_euler = [0, 0, 0]
            if location is not None:
                tex_object.ref_obj.location = location + shift[1] * self.up + UP * shift[2]
            else:
                tex_object.ref_obj.location = self.cursor[column] + self.sep + UP * 0.025
            tex_object.set_scale(adjusted_scale)
            raw_length = tex_object.get_text_bounding_box()[3]
            self.cursor[column] += raw_length / 2 * Vector([scale, 0, 0])
            # print(raw_length)
        # print("after adding: ",self.cursor)

    def write_text(self, tex_object, scale=0.7, indent=None, column=1, location=None, shift=None, begin_time=0,
                   transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        if shift is None:
            shift = self.standard_shift
        self.add_text(tex_object, scale=scale, indent=indent, column=column, location=location, shift=shift)
        tex_object.write(begin_time=begin_time, transition_time=transition_time, **kwargs)
        self.front_text.append(tex_object)
        return begin_time + transition_time

    def add_text_in(self, tex_object, scale=0.7, line=1, indent=0, column=1, hidden=False, shift=None):
        if shift is None:
            shift = self.standard_shift
        column -= 1
        adjusted_scale = self.get_scale(scale)
        tex_object.ref_obj.parent = self.ref_obj
        tex_object.ref_obj.rotation_euler = [0, 0, 0]
        tex_object.ref_obj.location = self.text_start[column] + \
                                      line * self.line_spacing + self.right * indent + UP * shift[2] + Vector(
            [shift[0], shift[1], 0])
        if hidden:
            tex_object.ref_obj.location += Vector([0, 0, -0.1])
        tex_object.ref_obj.scale = adjusted_scale

    def write_text_in(self, tex_object, scale=0.7, line=1, indent=0, column=1, hidden=False, begin_time=0,
                      transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        if 'shift' in kwargs:
            shift = kwargs.pop('shift')
        else:
            shift = self.standard_shift
        self.add_text_in(tex_object, scale=scale, line=line, indent=indent, column=column, hidden=hidden, shift=shift)
        tex_object.write(begin_time=begin_time, transition_time=transition_time, **kwargs)
        self.front_text.append(tex_object)
        return begin_time + transition_time

    def write_text_in_back(self, tex_object, scale=0.7, line=1, indent=0, column=1, hidden=False, begin_time=0,
                           transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        self.add_text_in_back(tex_object, scale=scale, line=line, indent=indent, column=column, hidden=hidden)
        tex_object.write(begin_time=begin_time, transition_time=transition_time, **kwargs)
        self.back_text.append(tex_object)
        self.back_text.append(tex_object)
        return begin_time + transition_time

    def add_text_in_back(self, tex_object, scale=0.7, line=1, indent=0, column=1, hidden=False):
        column -= 1
        adjusted_scale = self.get_scale(scale)
        tex_object.ref_obj.parent = self.ref_obj
        tex_object.ref_obj.rotation_euler = [0, np.pi, 0]
        tex_object.ref_obj.location = self.text_start_back[column] + \
                                      line * self.line_spacing + self.left * indent + DOWN * self.standard_shift[2]
        if hidden:
            tex_object.ref_obj.location += Vector([0, 0, -0.1])
        tex_object.ref_obj.scale = adjusted_scale

    def clear_front_text(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        clear front page
        :param begin_time:
        :param transition_time:
        :return:
        """

        n = len(self.front_text)
        t0 = begin_time
        if n > 0:
            dt = transition_time / n
            for line in self.front_text:
                line.disappear(begin_time=t0, transition_time=DEFAULT_ANIMATION_TIME)
                t0 += dt

        self.front_text = []
        return t0 + DEFAULT_ANIMATION_TIME

    def clear_back_text(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        clear front page
        :param begin_time:
        :param transition_time:
        :return:
        """

        n = len(self.back_text)
        t0 = begin_time
        if n > 0:
            dt = transition_time / n
            for line in self.back_text:
                line.disappear(begin_time=t0, transition_time=DEFAULT_ANIMATION_TIME)
                t0 += dt
        self.back_text = []
        return t0 + DEFAULT_ANIMATION_TIME

    def set_cursor_to_start_of_next_line(self, indent=None, column=1):
        column -= 1

        self.line[column] += 1
        if indent is None:
            par_indent = 0 * self.left_edge
        else:
            par_indent = indent * self.right
        self.cursor[column] = par_indent + self.text_start[column] + self.line[column] * self.line_spacing

    def set_cursor_to_start_of_line(self, line, indent=None, column=1):
        column -= 1

        self.line[column] = line
        if indent is None:
            par_indent = 0 * self.left_edge
        else:
            par_indent = indent * self.right
        self.cursor[column] = par_indent + self.text_start[column] + self.line[column] * self.line_spacing

    def to_top(self, column=1):
        column -= 1
        self.line[column] = -1
        self.set_cursor_to_start_of_next_line(indent=0, column=column)

    def appear(self,
               begin_time=0,
               transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        if self.show_lines:
            for line in self.lines:
                line.grow(begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def hide(self, b_obj, letter_range=None, letter_set=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        unhide text that was added hidden initially

        :param letter_range:
        :param b_obj:
        :param begin_time:
        :param transition_time:
        :return:
        """

        if hasattr(b_obj, 'letters'):
            if not letter_set:
                if letter_range:
                    letter_set = set(range(letter_range[0], letter_range[1]))
                else:
                    letter_set = set(range(0, len(b_obj.letters)))

            for l in letter_set:
                b_obj.letters[l].move(direction=Vector([0, 0, -self.standard_shift[2]]), begin_time=begin_time,
                                      transition_time=transition_time)
        else:
            b_obj.move(direction=Vector([0, 0, -0.1]), begin_time=begin_time, transition_time=transition_time)

    def un_hide(self, b_obj, letter_range=None, letter_set=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        unhide text that was added hidden initially

        :param letter_range:
        :param b_obj:
        :param begin_time:
        :param transition_time:
        :return:
        """
        if not letter_set:
            if letter_range:
                letter_set = set(range(letter_range[0], letter_range[1]))
            else:
                letter_set = set(range(0, len(b_obj.letters)))

        for l in letter_set:
            b_obj.letters[l].move(direction=self.standard_shift, begin_time=begin_time,
                                  transition_time=transition_time)

    def turn(self, flat=False, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, reverse=False):
        if not reverse:
            if not flat:
                self.rotate(rotation_euler=[np.pi / 2, 0, np.pi - np.pi / 15], begin_time=begin_time,
                            transition_time=transition_time)
            else:
                self.rotate(rotation_euler=[np.pi / 2, 0, np.pi], begin_time=begin_time,
                            transition_time=transition_time)
        else:
            if not flat:
                self.rotate(rotation_euler=[np.pi / 2, 0, -np.pi / 15], begin_time=begin_time,
                            transition_time=transition_time)
            else:
                self.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=begin_time,
                            transition_time=transition_time)
        return begin_time + transition_time


class InfinityDisplay(Display):
    def __init__(self, location=[12, 0, 0], scales=[4, 6, 4], rotation_euler=[np.pi / 2, 0, 0], number_of_lines=5,
                 columns=1,
                 flat=False, shadow=True, show_lines=False, **kwargs):
        super().__init__(location=location, scales=scales, rotation_euler=rotation_euler,
                         number_of_lines=number_of_lines, columns=columns, flat=flat, show_lines=show_lines,
                         shadow=shadow, **kwargs)
        self.elements = []

    def add_text(self, tex_object, scale=0.7, indent=None,
                 begin_time=0, transition_time=OBJECT_APPEARANCE_TIME):

        if len(self.elements) == self.number_of_lines - 1:
            first = self.elements[0]
            # first.disappear(begin_time=begin_time, transition_time=transition_time)
            first.move(direction=Vector([0, 0, -0.1]), begin_time=begin_time + transition_time / 2,
                       transition_time=transition_time / 2)
            self.elements.pop(0)

            for element in self.elements:
                element.move(direction=-self.line_spacing, begin_time=begin_time, transition_time=transition_time)

        super().add_text_in(tex_object, scale=scale, indent=indent, column=1, line=len(self.elements), hidden=True)

        self.elements.append(tex_object)
        if len(self.elements) < self.number_of_lines - 1:
            self.set_cursor_to_start_of_next_line()

        tex_object.write(begin_time=begin_time, transition_time=transition_time)
        tex_object.move(direction=Vector([0, 0, 0.01]), begin_time=begin_time, transition_time=0)


class DisplayOld(BObject):
    """
    Create a plane with a glass like color that is used to write text on it


    """

    def __init__(self, location=[12, 0, 0],
                 scales=[4, 6, 4],
                 rotation_euler=[np.pi / 2, 0, 0],
                 line_spacing=1, columns=1,
                 flat=False, shadow=False, **kwargs):

        plane = ibpy.add_plane()

        modifier_solid = plane.modifiers.new(name='solid', type='SOLIDIFY')
        modifier_solid.thickness = 0.05

        # introduce manim like parameters for the display.
        # due to the rotation the z-direction and the y-direction are interchanged

        self.scale_x = scales[0]
        self.scale_y = scales[1]
        self.scale_z = scales[2]

        self.up = 1 / self.scale_y * Vector([0, 1, 0])
        self.down = -self.up
        self.left = 1. / self.scale_x * Vector([-1, 0, 0])
        self.right = -self.left

        self.line_spacing = line_spacing * self.down
        self.title_line = self.scale_y * self.up + 0.75 * self.line_spacing

        self.sep = Vector([0.03, 0, 0])
        self.col_sep = 0.1 * self.scale_x

        self.text_start = []
        self.cursor = []
        self.line = []
        for col in range(columns):
            self.text_start.append(self.title_line + 0.5 * self.line_spacing + self.scale_x * 0.8 * self.left + col * (
                    self.scale_x / columns + self.col_sep) * self.right)
            self.cursor.append(self.text_start[-1].copy())
            self.line.append(0)

        if not flat:
            rotation_euler[2] -= np.pi / 15

        super().__init__(obj=plane,
                         name='Display',
                         location=location,
                         scale=[self.scale_x, self.scale_y, self.scale_z],
                         rotation_euler=rotation_euler,
                         color='background',
                         **kwargs)

        ibpy.set_shadow_of_object(self, shadow)

    def set_title(self, title, scale=0.4, shift=[0, 0]):
        """
        define SimpleTexObject as title and put it at the designated location

        :param scale:
        :param title:
        :return:
        """
        title.ref_obj.parent = self.ref_obj
        title.ref_obj.location = self.title_line + shift[0] * self.up + shift[1] * self.right
        title.ref_obj.name = 'Title'
        title.ref_obj.rotation_euler = [0, 0, 0]
        title.ref_obj.scale = scale * Vector([1, self.scale_x / self.scale_y, 1 / scale])

    def add_text(self, tex_object, scale=0.3, indent=None, custom_scales=None, column=1):
        """

        :param tex_object:
        :param scale:
        :param indent:
        :param custom_scales: can be used to add text without length, eg when the text is only used for morphing into
        :return:
        """
        # reindex column from userfriendly to python friendly
        column -= 1

        # print("before adding: ",self.cursor)
        if indent is not None:
            self.cursor[column] += indent * self.right
        if isinstance(tex_object, SimpleTexBObject):
            # single object
            tex_object.ref_obj.parent = self.ref_obj
            tex_object.ref_obj.rotation_euler = [0, 0, 0]
            tex_object.ref_obj.location = self.cursor[column]
            tex_object.ref_obj.scale = scale * Vector([1, self.scale_x / self.scale_y, 1 / scale])
            self.cursor[column] = self.cursor[column] + tex_object.length * Vector([scale, 0, 0]) + self.sep
            # print(tex_object.length*scale)
        elif isinstance(tex_object, TexBObject):
            for s, sub_tex_object in enumerate(tex_object.objects):
                sub_tex_object.ref_obj.parent = self.ref_obj
                sub_tex_object.ref_obj.rotation_euler = [0, 0, 0]
                # print(sub_tex_object.length * scale)
                if (self.cursor[column] + sub_tex_object.length * Vector([scale, 0, 0])).x > 3 * self.right.x:
                    self.set_cursor_to_start_of_next_line(indent, column)
                sub_tex_object.ref_obj.location = self.cursor[column]
                sub_tex_object.ref_obj.scale = scale * Vector([1, self.scale_x / self.scale_y, 1 / scale])
                custom_scale = 1
                if custom_scales is not None:
                    if len(custom_scales) > s:
                        custom_scale = custom_scales[s]
                self.cursor[column] = self.cursor[column] + self.sep + custom_scale * sub_tex_object.length * Vector(
                    [scale, 0, 0])
                if self.cursor[column].x > 3 * self.right.x:
                    self.set_cursor_to_start_of_next_line(self.line, column)
        # print("after adding: ",self.cursor)

    def set_cursor_to_start_of_next_line(self, indent=None, column=1):
        column -= 1

        self.line[column] += 1
        if indent is None:
            par_indent = self.right
        else:
            par_indent = indent * self.right
        self.cursor[column] = par_indent + self.text_start[column] + self.line[column] * self.line_spacing

    def toTop(self, column=1):
        column -= 1
        self.line[column] = -1
        self.set_cursor_to_start_of_next_line(indent=0, column=column)


class CodeDisplay(Display):
    def __init__(self, code_parser, class_index=0, **kwargs):
        self.kwargs = kwargs
        self.code_parser = code_parser
        cls = self.code_parser.classes[class_index]
        n_lines = self.get_from_kwargs('number_of_lines', None)
        if n_lines is None:
            n_lines = cls.number_of_lines()

        super().__init__(number_of_lines=n_lines, **kwargs)
        self.current_line = 0

    def write_text(self, text, **kwargs):
        t0 = super().write_text_in(text, line=self.current_line, **kwargs)
        self.current_line += 1
        return t0

    def write_text_in_back(self, text, **kwargs):
        t0 = super().write_text_in_back(text, line=self.current_line, **kwargs)
        self.current_line += 1
        return t0

    def add_empty_line(self):
        self.current_line += 1

    def turn(self, flat=False, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, reverse=False):
        self.current_line = 0  # start from the top again
        return super().turn(flat=flat, begin_time=begin_time, transition_time=transition_time, reverse=reverse)


class GlassDisplay(Display):
    """
    the material is preset to 'fake_glass_text'
    there is a predefined shift that moves all text inside the display

    the roughness=0 and metallic=1 is set with the override_material flag
    """

    def __init__(self, location=[12, 0, 0],
                 scales=[4, 6],
                 rotation_euler=[np.pi / 2, 0, 0],
                 number_of_lines=5, columns=1,
                 shadow=True, show_lines=False,
                 apply_scale=False,
                 apply_location=False, shift=[0, 0, 0], metallic=0, color='text', ior = 1.45, **kwargs):
        super().__init__(location=location, scales=scales, number_of_lines=number_of_lines, columns=columns,
                         rotation_euler=rotation_euler, shadow=shadow, roughness=0,
                         show_lines=show_lines, apply_location=apply_location, apply_scale=apply_scale,
                         color='fake_glass_' + color, ior=ior, standard_shift=shift, metallic=metallic,
                         override_material=True,subdivide_boundary=True,
                         **kwargs)
