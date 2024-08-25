import os
import bpy
import math
import numpy
import numpy as np

from appearance.textures import apply_material
from interface import ibpy
from interface.ibpy import get_splines, add_bezier_spline, link, is_hidden, un_hide, set_active, set_edit_mode, \
    set_object_mode, hide, un_link, select, un_select
from objects.bobject import BObject
from utils.constants import TEX_LOCAL_SCALE_UP, SVG_DIR, DEFAULT_ANIMATION_TIME, IMPORTED_OBJECTS, IMPORTED_CURVES
from utils.utils import add_lists_by_element


def sort(curve):
    """
    sort elements of the curve from left to right and up to down
    simple bubble sort algorith

    TODO make it smarter in such a way that objects which are relatively close are sorted from top to bottom

    :param curve:
    :return:
    """
    n = len(curve)
    if n == 1:
        return

    # determine the box of all locations
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    xs = []

    # the location captures the reference of the curve, and all the locations are more or less lined up in y-direction
    # the true position of the curve is determined with its bound boxes
    bounds = []
    for c in curve:
        x_min_loc = np.inf
        x_max_loc = -np.inf
        y_min_loc = np.inf
        y_max_loc = -np.inf
        for b in c.bound_box:
            if b[0] < x_min_loc:
                x_min_loc = b[0]
            if b[0] > x_max_loc:
                x_max_loc = b[0]
            if b[1] < y_min_loc:
                y_min_loc = b[1]
            if b[1] > y_max_loc:
                y_max_loc = b[1]
        bounds.append([x_min_loc + c.location[0], y_min_loc + c.location[1], x_max_loc + c.location[0],
                       y_max_loc + c.location[1]])
        xs.append(c.location[0])
        if c.location[0] < x_min:
            x_min = c.location[0]
        if c.location[0] > x_max:
            x_max = c.location[0]
        if c.location[1] < y_min:
            y_min = c.location[1]
        if c.location[1] > y_max:
            y_max = c.location[1]

    list.sort(xs)
    # group letters in bins in x-direction. Letters closer than dx belong to the same bin
    dx = (x_max - x_min) / 2 / len(curve)
    bins = []
    for i in range(0, len(xs) - 1):
        if xs[i + 1] > xs[i] + dx:
            bins.append(xs[i])

    for i in range(0, n - 1):
        for j in range(0, n - 1 - i):
            loc_j = curve[j].location[0]
            loc_jn = curve[j + 1].location[0]

            # first check, whether both letters belong to the same bin, within the same bin they are sorted from top
            # to bottom
            for x_bin in bins:
                if loc_j <= x_bin and loc_jn <= x_bin:
                    # the same bin
                    mid_y = 0.5 * (bounds[j][1] + bounds[j][3])
                    mid_y_n = 0.5 * (bounds[j + 1][1] + bounds[j + 1][3])
                    if mid_y_n > mid_y:
                        curve[j + 1], curve[j] = curve[j], curve[j + 1]
                        bounds[j + 1], bounds[j] = bounds[j], bounds[j + 1]
                    break
                elif loc_j <= x_bin < loc_jn:
                    break
                elif loc_jn <= x_bin < loc_j:
                    curve[j + 1], curve[j] = curve[j], curve[j + 1]
                    bounds[j + 1], bounds[j] = bounds[j], bounds[j + 1]
                    break

    return tuple(curve)


class SVGBObject(BObject):
    """
    This is the simple version of an svg-object that does not contain morphing.
    Simply the data is imported and two arrays are generated,
    | one with curves of open splines
    | one with curves of closed spline

    """

    def __init__(self, filename, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'SVG')
        super().__init__(name=self.name, **kwargs)

        self.length = None
        self.rendered_objects = None  # initialized in make_rendered_curve_b_objects
        self.prep_log = None  # initialized in process_morph_chains
        # self.max_point_count = None  # initialized in process_morph_chains
        # self.max_spline_count = None  # initialized in process_morph_chains
        self.path = None  # initialized in get_path
        self.list_of_copies = None  # initialized in make_morph_chains
        self.imported_svg_data = None  # initialized in import_svg_data

        if 'vert_align_centers' in kwargs:
            self.vert_align_centers = kwargs['vert_align_centers']
        else:
            self.vert_align_centers = 'x_and_y'
        if 'aligned' in kwargs:
            self.aligned = kwargs['aligned']
        else:
            self.aligned = 'left'
        if 'color' in kwargs:
            self.default_color = kwargs['color']
        else:
            self.default_color = 'drawing'
        if 'text_size' in kwargs:
            self.text_size = kwargs['text_size']
        else:
            self.text_size = 'normal'

        scale = self.get_from_kwargs('scale', 1)

        # for tex objects the svg data is just intermediate, which is removed eventually
        self.remove_svg_data = self.get_from_kwargs('remove_svg_data', False)
        # self.reindex_points_before_morph = self.get_from_kwargs('reindex_points_before_morph', True)
        # self.lazy_morph = self.get_from_kwargs('lazy_morph', True)
        # self.min_length = self.get_from_kwargs('min_length', 1)
        # self.write_first = self.get_from_kwargs('write_first', True)
        self.get_file_path(filename)
        self.import_svg_data(text_size=self.text_size, scale=scale)
        self.align_figures()
        # self.count_splines_and_points()
        self.make_rendered_objects()
        bevel = self.get_from_kwargs('bevel', 0)

        # create a letter color dictionary
        self.color_map = {}
        if isinstance(self.default_color, list):
            for i,obj in enumerate(self.rendered_objects):
                if i<len(self.default_color):
                    self.color_map[obj]=self.default_color[i]
                else:
                    self.color_map[obj]=self.default_color[-1]
        else:
            for obj in self.rendered_objects:
                self.color_map[obj]=self.default_color

        # modify rendered objects for simple svg objects
        # text objects get modified elsewhere
        for rendered in self.rendered_objects:
            curve = rendered.ref_obj
            curve.parent = self.ref_obj
            curve.data.bevel_depth = bevel * 0.01

        print("SVG BObject initialized " + str(filename))

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        dt = transition_time / len(self.rendered_objects)
        for i, rendered in enumerate(self.rendered_objects):
            rendered.appear(begin_time=begin_time + i * dt, transition_time=dt,**kwargs)

    def get_file_path(self, filename):
        """
        The latex expression is converted into a filename.
        :param filename:
        :return:
        """
        self.path = os.path.join(SVG_DIR, filename) + ".svg"
        if not os.path.exists(self.path):
            raise Warning("Could not find " + filename + ".svg")

    # def count_splines_and_points(self):
    #     """
    #     find the highest number of splines and points in a curve
    #     :return:
    #     """
    #     self.max_spline_count = 0
    #     self.max_point_count = CONTROL_POINTS_PER_SPLINE
    #     for curve in self.imported_svg_data[self.path]['curves']:
    #         spline_count = len(curve.ref_obj.data.splines)
    #         self.max_spline_count = max(spline_count, self.max_spline_count)
    #         for spline in curve.ref_obj.data.splines:
    #             point_count = len(spline.bezier_points)
    #             self.max_point_count = max(point_count, self.max_point_count)
    #
    #     print("highest number of splines: ", self.max_spline_count)
    #     print("highest points in single spline: ", self.max_point_count)

    def make_rendered_objects(self):
        self.rendered_objects = []

        for i, curve in enumerate(self.imported_svg_data[self.path]['curves']):
            c = curve.ref_obj

            #  this is not heavily needed in this place but it makes it more convenient for morphing later on
            # equalize_spline_count(c, self.max_spline_count)
            # link(c)
            #  TODO: maybe it is better, to set a fixed number of points to make sure that there are no big gaps
            # in the open path
            # add_points_to_curve_splines(c, total_points=self.max_point_count)
            # un_link(c)

            # apply color, if there is a list of colors, the colors are assigned to the individual curves
            # if the end of the list of colors is reached, the last color is applied to the remaining letters
            color = None
            if isinstance(self.default_color, list):
                if i < len(self.default_color):
                    color = self.default_color[i]
                    # apply_material(c, self.default_color[i])
                else:
                    color = self.default_color[-1]
                    # apply_material(c, self.default_color[-1])
            else:
                color = self.default_color
                # apply_material(c, self.default_color)

            rendered_curve = BObject(obj=c, color=color, name="ren_"+c.name,
                                     emission=0.5,**self.kwargs)
            rendered_curve.ref_obj.location = c.location
            rendered_curve.ref_obj.rotation_euler = c.rotation_euler
            self.rendered_objects.append(rendered_curve)

        return

    def import_svg_data(self, text_size='normal', scale=1):
        self.imported_svg_data = {}  # Build dictionary of imported svgs to use
        # shape keys later and to avoid duplicate
        # imports
        path = self.path
        # Import svg and get list of new curves in Blender
        if path not in self.imported_svg_data.keys():
            self.imported_svg_data[path] = {'curves': []}
            # This is a dict of dicts for metadata, e.g., center and length
            # of tex expressions
            if path is None:
                null = new_null_curve()
                cur = null.ref_obj.children[0]
                equalize_spline_count(cur, 1)
                self.imported_svg_data[path]['curves'].append(cur)

            # start = time.perf_counter()
            previous_curves = [x for x in bpy.data.objects if x.type == 'CURVE']
            bpy.ops.import_curve.svg(filepath=path)  # here the import takes place
            # all curves are imported into a separate collection with the name path
            new_curves = [x for x in bpy.data.objects if \
                          x.type == 'CURVE' and x not in previous_curves]
            # ende = time.perf_counter()
            # print("time fresser? "+str(ende-start))
            # Arrange new curves relative to tex object's ref_obj
            if text_size == 'normal':
                scale_up = TEX_LOCAL_SCALE_UP
            elif text_size == 'medium':
                scale_up = TEX_LOCAL_SCALE_UP * 1.25
            elif text_size == 'large':
                scale_up = 1.5 * TEX_LOCAL_SCALE_UP
            elif text_size == 'Large':
                scale_up = 2 * TEX_LOCAL_SCALE_UP
            elif text_size == 'Small':
                scale_up = 0.8 *TEX_LOCAL_SCALE_UP
            elif text_size == 'small':
                scale_up = 0.7 * TEX_LOCAL_SCALE_UP
            elif text_size =='xs':
                scale_up = 0.65* TEX_LOCAL_SCALE_UP
            elif text_size=='tiny':
                scale_up = 0.5 *TEX_LOCAL_SCALE_UP
            elif text_size=='huge':
                scale_up = 3 * TEX_LOCAL_SCALE_UP
            elif text_size=='Huge':
                scale_up = 5 * TEX_LOCAL_SCALE_UP
            else:  # a number
                scale_up = text_size * TEX_LOCAL_SCALE_UP

            # in the following lines the location of the curve is determined from the geometry and the
            # y-coordinates of the locations are all aligned with the reference 'H'
            for curve in new_curves:
                for spline in curve.data.splines:
                    for point in spline.bezier_points:
                        point.handle_left_type = 'FREE'
                        point.handle_right_type = 'FREE'
                    # This needs to be in a separate loop because moving points before
                    # they're all 'Free' type makes the shape warp.
                    # It makes a cool "disappear in the wind" visual, though.
                    for point in spline.bezier_points:
                        for i in range(len(point.co)):
                            point.co[i] *= (scale * scale_up)
                            point.handle_left[i] *= (scale * scale_up)
                            point.handle_right[i] *= (scale * scale_up)

                ibpy.set_origin(curve)
                # This part is just meant for tex_objects
                if self.vert_align_centers:
                    loc = curve.location
                    new_y = new_curves[0].location[1]  # reference location of the 'H'
                    ibpy.set_pivot(curve, [loc[0], new_y, loc[2]])
                un_select(curve)

                # find the collection of the curve and unlink object
                for collection in bpy.context.scene.collection.children:
                    for o in collection.objects:
                        if o == curve:
                            un_link(curve, collection.name)
                            break

            self.imported_svg_data[path]['curves'] = new_curves

        # replace imported curves by bobjects

        curve = self.imported_svg_data[path]['curves']
        sort(curve)  # introduced by p6majo to make the morphing less random

        for i, curve in enumerate(self.imported_svg_data[path]['curves']):
            curve_bobj = BObject(obj=curve,
                                 location=curve.location,
                                 rotation_euler=curve.rotation_euler,
                                 name='imp_'+str(len(IMPORTED_CURVES)).zfill(5))
            IMPORTED_CURVES.append(curve_bobj)
            self.imported_svg_data[path]['curves'][i] = curve_bobj

        bpy.context.view_layer.update()

    def get_file_paths(self, filename):
        """
        The latex expression is converted into a filename.
        To avoid problems, all backslashes and brackets have to be replaced

        :param filename: name of file to import
        :return:
        """
        self.path = os.path.join(SVG_DIR, filename) + ".svg"
        if not os.path.exists(self.path):
            raise Warning("Could not find " + filename + ".svg")

    def get_figure_curves(self, fig):
        """
        returns the curves
        this method is overriden by the TexBObject to strip the leading H
        :param fig:
        :return:
        """
        if fig is None:
            return self.imported_svg_data[fig]['curves']
        else:
            return self.imported_svg_data[fig]['curves'][0:]

    def calc_lengths(self):
        """
        The dimension of the object is calculated
        The parameters are stored in

        | imported_svg_data:
        | 'length'
        | 'height'
        | 'centerx'
        | 'centery'
        | 'beginning'
        | 'end'
        | 'top'
        | 'bottom'

        :return:
        """
        for expr in self.imported_svg_data:
            curves = self.get_figure_curves(expr)  # the H is stripped for latex formulas

            max_vals = [-math.inf, -math.inf]
            min_vals = [math.inf, math.inf]

            directions = [0, 1]  # 0 horizontal
            # 1 vertical

            for direction in directions:
                for char in curves:
                    # char is a b_object, so reassign to the contained curve
                    char = char.ref_obj
                    for spline in char.data.splines:
                        for point in spline.bezier_points:
                            candidate = char.matrix_local.translation[direction] + point.co[direction] * char.scale[
                                direction]  # +char.parent.matrix_local.translation[direction]
                            if max_vals[direction] < candidate:
                                max_vals[direction] = candidate
                            if min_vals[direction] > candidate:
                                min_vals[direction] = candidate

            right_most_x, top_most_y = max_vals
            left_most_x, bottom_most_y = min_vals

            length = right_most_x - left_most_x
            center = left_most_x + length / 2

            self.imported_svg_data[expr]['length'] = length * self.intrinsic_scale[0]
            self.imported_svg_data[expr]['centerx'] = center
            self.imported_svg_data[expr]['beginning'] = left_most_x  # Untested
            self.imported_svg_data[expr]['end'] = right_most_x

            height = top_most_y - bottom_most_y
            center = bottom_most_y + height / 2

            self.imported_svg_data[expr]['top'] = top_most_y
            self.imported_svg_data[expr]['bottom'] = bottom_most_y
            self.imported_svg_data[expr]['height'] = height * self.intrinsic_scale[1]
            self.imported_svg_data[expr]['centery'] = center

            self.length = length * self.intrinsic_scale[0]

    def align_figures(self):
        self.calc_lengths()
        for fig in self.imported_svg_data:
            self.align_figure(fig)

    def align_figure(self, fig):
        data = self.imported_svg_data
        curve_list = data[fig]['curves']
        offset = list(curve_list[0].ref_obj.location)

        # positioning of the first element of the list
        if self.aligned == 'right':
            offset[0] = data[fig]['end']
            offset[1] = data[fig]['centery']
        if self.aligned == 'right_bottom':
            offset[0] = data[fig]['end']
            offset[1] = data[fig]['bottom']
        elif self.aligned == 'top_centered':
            offset[0] = data[fig]['centerx']
            offset[1] = data[fig]['top']
        elif self.aligned == 'left_bottom':
            offset[0] = data[fig]['beginning']
            offset[1] = data[fig]['bottom']
        elif self.aligned == 'left_top':
            offset[0] = data[fig]['beginning']
            offset[1] = data[fig]['top']
        elif self.aligned == 'x_and_y':
            offset[0] = data[fig]['centerx']
            offset[1] = data[fig]['centery']
        elif self.aligned == 'left':
            offset[0] = data[fig]['beginning']
            offset[1] = data[fig]['centery']
        elif self.aligned == 'center':
            cen = data[fig]['centerx']
            offset[0] = cen
            offset[1] = data[fig]['centery']

        for i in range(len(curve_list)):
            loc = list(curve_list[i].ref_obj.location)
            new_loc = add_lists_by_element(loc, offset, subtract=True)
            curve_list[i].ref_obj.location = new_loc

        return curve_list  # needed by sub class TexBObject


def new_null_curve(parent=None, location=(0, 0, 0), rotation=(0, 0, 0),
                   # color = 'color5',
                   reuse_object=None
                   ):
    # print("Adding null curve")
    # if reuse_object == None:
    data = bpy.data.curves.new(name='no_curve_data', type='CURVE')
    obj = bpy.data.objects.new(name='no_curve', object_data=data)
    # else:
    #    print('Reusing object!!!!!!')
    #    obj = reuse_object
    bobj = BObject(obj=obj, name='null')
    # obj.parent = bobj.ref_obj
    # bobj.objects.append(obj)
    bobj.ref_obj.parent = parent
    # print(matrix_local)
    bobj.ref_obj.location = location
    bobj.ref_obj.rotation_euler = rotation
    # print(bobj.ref_obj.matrix_local)
    # bpy.context.scene.objects.link(new_null)
    # if reuse_object == None:
    #    bobj.add_to_blender(animate = False)
    # apply_material(obj, color)
    # bpy.data.scenes[0].update()
    # print('    Done adding null curve')
    return bobj


def equalize_spline_count(curve1, target):
    splines1 = ibpy.get_splines(curve1)

    target_is_number = False
    if isinstance(target, int) or isinstance(target, np.int32) or isinstance(target, numpy.int64):
        target_is_number = True
        spline_count = target
    else:
        splines2 = ibpy.get_splines(target)
        spline_count = max(len(splines1), len(splines2))

    while len(splines1) < spline_count:
        new_spline = splines1.new('BEZIER')
        new_spline.bezier_points.add(count=2)
        new_spline.use_cyclic_u = True

    if not target_is_number:
        while len(splines2) < spline_count:
            new_spline = splines2.new('BEZIER')
            new_spline.bezier_points.add(count=2)
            new_spline.use_cyclic_u = True
