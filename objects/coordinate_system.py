import mathutils
from mathutils import Vector
import numpy as np


from interface import ibpy

from objects.cylinder import Cylinder
from objects.function import Function, MeshFunction
from objects.number_line import NumberLine, DynamicNumberLine, NumberLine2
from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs
from utils.utils import to_vector

class CoordinateSystem2(BObject):
    def __init__(self,**kwargs):
        """
            creates a coordinate system

            use dim=2 for two-dimensional and dim=3 for three-dimensional coordinate systems
            the length of the arrays should correspond to the number of dimensions

            for each dimension an Numberline is created
            in geometry nodes
            """

        self.kwargs = kwargs
        self.origin = self.get_from_kwargs('origin', [0, 0])
        self.dimension = self.get_from_kwargs('dim', 2)
        self.location = self.get_from_kwargs('location', Vector([0, 0]))
        self.lengths = self.get_from_kwargs('lengths', [7,7])
        self.radii = self.get_from_kwargs('radii', [0.05, 0.05])
        self.domains = self.get_from_kwargs("domains",[[0,10],[0,10]])
        self.include_zeros =self.get_from_kwargs("include_zeros",[False,False])
        self.colors = self.get_from_kwargs('colors',['drawing','drawing'])

        self.axes = []
        if self.dimension ==2:
            names= ["xAxis","yAxis"]
            directions = ["HORIZONTAL","VERTICAL"]
            for i in range(2):
                self.axes.append(NumberLine2(name=names[i],direction=directions[i],domain=self.domains[i],
                                         include_zero=self.include_zeros[i],length=self.lengths[i],
                                 color=self.colors[i],**kwargs))


        super().__init__(children=self.axes,name=str(self.dimension)+"D-CoordinateSystem",location=self.location,**kwargs)

    def appear(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,alpha=1):
        super().appear(scale=scale,begin_time=begin_time,transition_time=transition_time,alpha=alpha)
        for axis in self.axes:
            axis.grow(scale=scale,begin_time=begin_time,transition_time=transition_time,alpha=alpha)
        return begin_time+transition_time

class CoordinateSystem(BObject):
    """
    creates a coordinate system

    use dim=2 for two-dimensional and dim=3 for three-dimensional coordinate systems
    the length of the arrays should correspond to the number of dimensions

    for each dimension an Numberline is created

    example:

    coord = CoordinateSystem(dim=3,
                             lengths=[10, 10, 10],
                             radii=[0.03, 0.03, 0.03],
                             domains=[[-2, 2], [-2, 2], [-1, 3]],
                             all_n_tics=[4, 4, 4],
                             location_of_origin=[0, 0, 0],
                             labels=["{\\mathbb R}",
                                     "i{\\mathbb R}",
                                     "|z|"],
                             all_tic_labels=[np.arange(-2, 2.1, 1),
                                             np.arange(-2, 2.1, 1),
                                             np.arange(-1, 3.1, 1)],
                             colors=['drawing', 'drawing', 'drawing']
                             label_colors=['drawing','drawing','example'],
                             text_size='large',
                             label_digits=[0,1],
                             label_positions=['left','left','left'],
                             )

    """

    def __init__(self, **kwargs):
        """
        Constructor for CoordinateSystem
        """
        self.kwargs = kwargs
        self.origin = self.get_from_kwargs('origin', [0, 0, 0])
        self.dimensions = self.get_from_kwargs('dim', 2)
        self.location_of_origin = self.get_from_kwargs('location_of_origin', Vector([0, 0, 0]))
        self.lengths = self.get_from_kwargs('lengths', [2, 2, 2])
        self.radii = self.get_from_kwargs('radii', [0.05, 0.05, 0.05])
        self.dynamic = self.get_from_kwargs('dynamic',False)
        rotation = self.get_from_kwargs('rotation_euler', [0, 0, 0])
        if 'rotation_euler' in kwargs:
            kwargs.pop('rotation_euler')

        if self.dimensions == 2:
            # configure coordinate system for x- and y-axis only (y-axis pointing into z-direction)
            labels = self.get_from_kwargs('labels', ['x', 'y'])
            directions = ['horizontal', 'vertical']
        else:
            # configure coordinate system for x-, y- and z-axis (z-axis pointing into z-direction)
            labels = self.get_from_kwargs('labels', ['x', 'y', 'z'])
            directions = ['horizontal', 'deep', 'vertical']

        self.domains = self.get_from_kwargs('domains', [[-1, 1], [-1, 1], [-1, 1]])

        if self.dimensions > 3 or self.dimensions < 1:
            raise str(self.dimensions) + " dimensions are not applicable for the class " + CoordinateSystem
        all_n_tics = self.get_from_kwargs('all_n_tics', [10, 10, 10])
        all_tic_labels = self.get_from_kwargs('all_tic_labels', ['AUTO', 'AUTO', 'AUTO'])

        self.axes = []
        kwargs['location_of_origin'] = [0, 0,
                                        0]  # reset the location of the origin for the numberlines relative to the parent

        shadings = self.get_from_kwargs('shadings', [None, None, None])
        colors = self.get_from_kwargs('colors', ['text', 'text', 'text'])
        tip_lengths = self.get_from_kwargs('tip_lengths', [None, None, None])
        label_colors = self.get_from_kwargs('label_colors', colors)
        label_digits = self.get_from_kwargs('label_digits', [1, 1, 1])
        label_units = self.get_from_kwargs('label_units', ['', '', ''])
        include_zeros = self.get_from_kwargs('include_zeros', [True, False, False])
        label_positions = self.get_from_kwargs('label_positions', ['left', 'left', 'left'])
        label_closenesses = self.get_from_kwargs('label_closenesses', [1, 1, 1])
        axis_label_closenesses = self.get_from_kwargs('axis_label_closenesses', [1, 1, 1])
        label_sizes = self.get_from_kwargs('label_sizes',['normal']*self.dimensions)

        if not self.dynamic:
            for i in range(self.dimensions):
                axis = NumberLine(length=self.lengths[i], radius=self.radii[i],
                                  domain=self.domains[i],
                                  n_tics=all_n_tics[i],
                                  label=labels[i],
                                  tic_labels=all_tic_labels[i],
                                  direction=directions[i],
                                  include_zero=include_zeros[i],
                                  origin=self.origin[i],
                                  color=colors[i],
                                  shading=shadings[i],
                                  label_digit=label_digits[i],
                                  label_unit=label_units[i],
                                  tip_length=tip_lengths[i],
                                  label_color=label_colors[i],
                                  label_position=label_positions[i],
                                  label_closeness=label_closenesses[i],
                                  axis_label_closeness=axis_label_closenesses[i],
                                  axis_label_size=label_sizes[i],
                                  tic_label_size=label_sizes[i],
                                  **kwargs)
                self.axes.append(axis)
        else:
            ranges = self.get_from_kwargs('ranges', None)
            for i in range(self.dimensions):
                if ranges is not None:
                    rng = ranges[i]
                else:
                    rng = list(range(0,10))
                axis = DynamicNumberLine(length=self.lengths[i], radius=self.radii[i],
                                  domain=self.domains[i],
                                  n_tics=all_n_tics[i],
                                  label=labels[i],
                                  tic_labels=all_tic_labels[i],
                                  direction=directions[i],
                                  include_zero=include_zeros[i],
                                  origin=self.origin[i],
                                  color=colors[i],
                                  shading=shadings[i],
                                  label_digit=label_digits[i],
                                  label_unit=label_units[i],
                                  tip_length=tip_lengths[i],
                                  label_color=label_colors[i],
                                  label_position=label_positions[i],
                                  label_closeness=label_closenesses[i],
                                  axis_label_closeness=axis_label_closenesses[i],
                                    range=rng,
                                  **kwargs)
                self.axes.append(axis)

        self.name=self.get_from_kwargs('name','CoordinateSystem')
        super().__init__(children=self.axes, name=self.name, rotation_euler=rotation, location=self.location_of_origin,
                         **kwargs)

    def get_scales(self):
        scales = []
        for axis in self.axes:
            scales.append(axis.get_scale())
        if len(scales) == 2:
            scales.insert(2, scales[0])  # insert the same x-scale for the unimportant y-direction in the 2d case
        return scales

    def coords2location(self, coordinates):
        """
        returns the location (X,Y,Z) in the world for given coordinates (x,y,z) of the coordinate system
            |
            | all little complication arises from the fact that the coordinate mapping is as follows

            | x->X
            | y->Z
            | z->y
            |
        :param coordinates: (x,y,z)
        :return: (X,Y,Z)
        """

        if isinstance(coordinates, tuple):
            coordinates = list(coordinates)
        dX = []
        for i in range(self.dimensions):
            dX.append(self.lengths[i] / (self.domains[i][1] - self.domains[i][0]))

        X = []
        for i in range(self.dimensions):
            X.append((coordinates[i] - self.origin[i]) * dX[i])

        # this is a bit rushed to have it compatible with
        # video_primes2.complex_functions
        # and
        # video_primes2.real_functions
        # any changes should be checked against the performance of the two sequences

        if self.dimensions == 3:
            if len(X) < 3:
                X.append(0)
            X = Vector(X)
        else:
            if len(X) < 3:
                X.append(0)
            # scales = self.get_scales()
            # X[0] /= scales[0]
            # X[1] /= scales[1]

            X[1], X[2] = X[2], X[1]
            X = Vector(X)  # + Vector(self.location_of_origin)
        return X

    def coords2location_relative2coordinate_system(self, coordinates,apply_origin_shift=False):
        if apply_origin_shift:
            return self.coords2location(coordinates) + to_vector(self.location_of_origin)
        else:
            return self.coords2location(coordinates)

    def location2coords(self, location):
        """
            returns the coordinates (x,y,z) of the coordinate system for a given position in the world
            |
            | all little complication arises from the fact that the coordinate mapping is as follows
            |
            | X->x
            | Y->z
            | Z->y
            |

        :param location: (X,Y,Z)
        :return: (x,y,z)
        """
        dx = []
        for i in range(self.dimensions):
            dx.append((self.domains[i][1] - self.domains[i][0]) / self.lengths[i])

        x = []
        if self.dimensions == 2:
            for index, i in enumerate([0, 2]):
                x.append(self.origin[i] + (location[i] - self.location_of_origin[i]) * dx[index])
        else:
            for i in range(self.dimensions):
                x.append(self.origin[i] + (location[i] - self.location_of_origin[i]) * dx[i])

        return x

    def get_domain(self):
        if self.dimensions == 2:
            return self.domains[0]
        else:
            return self.domains[0:2]

    def appear_individually(self, begin_times=[0, 0, 0],
                            transition_times=[OBJECT_APPEARANCE_TIME, OBJECT_APPEARANCE_TIME, OBJECT_APPEARANCE_TIME]):
        """
        allow each axis to appear independently

        :param begin_times:
        :param transition_times:
        :return:
        """
        super().appear()
        for i, axis in enumerate(self.axes):
            axis.appear(begin_time=begin_times[i], transition_time=transition_times[i])

    def appear(self,alpha=1,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME, empty=False,**kwargs
               ):

        t0 = begin_time
        super().appear(alpha=alpha,begin_time=t0, transition_time=transition_time,**kwargs)
        if not empty:
            for axis in self.axes:
                axis.appear(alpha=alpha, begin_time=t0, transition_time=transition_time,**kwargs)

        return t0+transition_time

    def disappear_axes(self, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME
                       ):
        for axis in self.axes:
            axis.disappear(begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time

    def disappear_grid(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time / (len(self.x_lines) + len(self.y_lines))
        counter = 0
        for line in self.x_lines:
            line.disappear(begin_time=begin_time + counter * dt, transition_time=0.9 * transition_time)
            counter += 1
        for line in self.y_lines:
            line.disappear(begin_time=begin_time + counter * dt, transition_time=0.9 * transition_time)
            counter += 1
        return begin_time + transition_time

    def zoom(self, zoom=2, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        Scale coordinate system, where the labels and lines keep there thickness

        :param zoom:
        :param begin_time:
        :param transition_time:
        :return:
        """

        self.rescale(rescale=[zoom, zoom, zoom], begin_time=begin_time, transition_time=transition_time)

        # counter-act the overall scaling by indivually rescaling the number lines
        for axis in self.axes:
            axis.compensate_zoom(zoom=zoom, begin_time=begin_time, transition_time=transition_time)

    def rotate(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, pivot=None, interpolation='BEZIER',
               compensate=False, **kwargs):
        '''
        rotate coordinate system, with the option to compensate the rotation on the labels.
        :param begin_time:
        :param transition_time:
        :param rotation_euler:
        :param pivot:
        :param interpolation:
        :param compensate:
        :param kwargs:
        :return:
        '''
        super().rotate(begin_time=begin_time, transition_time=transition_time, pivot=pivot, interpolation=interpolation,
                       **kwargs)

        if not isinstance(compensate,list):
            compensate=[compensate]*3

        if 'rotation_euler' in kwargs:
            rotation_euler = kwargs.pop('rotation_euler')
            # calculate the delta compensation
            old_rotation_euler = ibpy.get_rotation_at_frame(self, begin_time * FRAME_RATE)
            delta = mathutils.Euler(rotation_euler).to_matrix() @ mathutils.Euler(
                old_rotation_euler).to_matrix().inverted()

            for comp, axis in zip(compensate,self.axes):
                if comp:
                    axis.compensate_rotation(begin_time=begin_time, transition_time=transition_time,
                                             delta_rotation=delta)
        return begin_time+transition_time

    def draw_grid_lines(self, colors=['drawing', 'drawing'], sub_grid=0, begin_time=0,
                        transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        self.grid_colors=colors

        x_axis = self.axes[0]
        y_axis = self.axes[1]

        x_min_tic = np.Infinity
        x_max_tic = -np.Infinity
        y_min_tic = np.Infinity
        y_max_tic = -np.Infinity
        for x in x_axis.tic_values:
            if x < x_min_tic:
                x_min_tic = x
            if x > x_max_tic:
                x_max_tic = x
        for y in y_axis.tic_values:
            if y < y_min_tic:
                y_min_tic = y
            if y > y_max_tic:
                y_max_tic = y

        self.x_lines = []
        first = True
        for x in x_axis.tic_values:
            if first:
                x_last = x
                first = False
            else:
                if sub_grid > 0:
                    dx = (x - x_last) / sub_grid
                    for i in range(1, sub_grid):
                        xi = x_last + i * dx
                        self.x_lines.append(Cylinder.from_start_to_end(start=self.coords2location([xi, y_min_tic]),
                                                                       end=self.coords2location([xi, y_max_tic]),
                                                                       color=colors[0], radius=self.radii[0] * 0.125,
                                                                       name='sub_grid_line_x_' + str(xi), **kwargs))
                        self.add_object(self.x_lines[-1])
                    x_last = x
            self.x_lines.append(Cylinder.from_start_to_end(start=self.coords2location([x, y_min_tic]),
                                                           end=self.coords2location([x, y_max_tic]),
                                                           color=colors[0], radius=self.radii[0] * 0.5,
                                                           name='grid_line_x_' + str(x), **kwargs))
            self.add_object(self.x_lines[-1])

        self.y_lines = []
        first = True
        for y in y_axis.tic_values:
            if first:
                y_last = y
                first = False
            else:
                if sub_grid > 0:
                    dy = (y - y_last) / sub_grid
                    for i in range(1, sub_grid):
                        yi = y_last + i * dy
                        self.y_lines.append(Cylinder.from_start_to_end(start=self.coords2location([x_min_tic, yi]),
                                                                       end=self.coords2location([x_max_tic, yi]),
                                                                       color=colors[0], radius=self.radii[0] * 0.125,
                                                                       name='sub_grid_line_y_' + str(yi), **kwargs))
                        self.add_object(self.y_lines[-1])
                    y_last = y
            self.y_lines.append(Cylinder.from_start_to_end(start=self.coords2location([x_min_tic, y]),
                                                           end=self.coords2location([x_max_tic, y]),
                                                           color=colors[1], radius=self.radii[1] * 0.5,
                                                           name='grid_line_y_' + str(y), **kwargs))
            self.add_object(self.y_lines[-1])

        offset = transition_time / 4 / len(self.x_lines)
        for i, line in enumerate(self.x_lines):
            line.grow(begin_time=begin_time + i * offset, transition_time=transition_time / 2, modus='from_start')

        offset = transition_time / 4 / len(self.y_lines)
        for i, line in enumerate(self.y_lines):
            line.grow(begin_time=begin_time + i * offset + transition_time / 2, transition_time=transition_time / 2,
                      modus='from_start')
        return begin_time+transition_time

    def grid_next_transform(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        for xline in self.x_lines:
            xline.next(begin_time=begin_time, transition_time=transition_time)
        for yline in self.y_lines:
            yline.next(begin_time=begin_time, transition_time=transition_time)
        return begin_time+transition_time

    def grid_change_color(self, new_color='text',begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for xline in self.x_lines:
            xline.change_color(new_color=new_color,begin_time=begin_time, transition_time=transition_time)
        for yline in self.y_lines:
            yline.change_color(new_color=new_color,begin_time=begin_time, transition_time=transition_time)

    def draw_transformable_grid(self, transformations=[], colors=['drawing', 'drawing'], sub_grid=0, begin_time=0,
                                transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        '''
        draw gridlines that can be transformed

        :param transformations:
        :param colors:
        :param sub_grid:
        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        '''

        thickness = get_from_kwargs(kwargs, 'thickness', 1)

        x_axis = self.axes[0]
        y_axis = self.axes[1]

        x_min_tic = np.Infinity
        x_max_tic = -np.Infinity
        y_min_tic = np.Infinity
        y_max_tic = -np.Infinity
        for x in x_axis.tic_values:
            if x < x_min_tic:
                x_min_tic = x
            if x > x_max_tic:
                x_max_tic = x
        for y in y_axis.tic_values:
            if y < y_min_tic:
                y_min_tic = y
            if y > y_max_tic:
                y_max_tic = y

        self.x_lines = []
        first = True
        for x in x_axis.tic_values:
            if first:
                x_last = x
                first = False
            else:
                if sub_grid > 0:
                    dx = (x - x_last) / sub_grid
                    for i in range(1, sub_grid):
                        xi = x_last + i * dx
                        start = Vector([xi, 0, y_min_tic])
                        end = Vector([xi, 0, y_max_tic])
                        self.x_lines.append(
                            MeshFunction(
                                lambda t: start + (end - start) * t,
                                lambda t: (end - start),
                                lambda t: Vector(),
                                transformations=transformations,
                                         domain=[0, 1],
                                         thickness=0.25 * thickness,
                                         name='sub_grid_line_x_' + str(xi),
                                         scale=self.get_scales(), **kwargs)
                        )
                    x_last = x
            start = Vector([x, 0, y_min_tic])
            end = Vector([x, 0, y_max_tic])
            self.x_lines.append(
                MeshFunction(
                    lambda t: start + (end - start) * t,
                    lambda t: (end - start),
                    lambda t: Vector(),
                    transformations=transformations,
                             domain=[0, 1],
                             thickness=thickness,
                             name='grid_line_x_' + str(x),
                             scale=self.get_scales(), **kwargs)
            )

        self.y_lines = []
        first = True
        for y in y_axis.tic_values:
            if first:
                y_last = y
                first = False
            else:
                if sub_grid > 0:
                    dy = (y - y_last) / sub_grid
                    for i in range(1, sub_grid):
                        yi = y_last + i * dy
                        start = Vector([x_min_tic, 0, yi])
                        end = Vector([x_max_tic, 0, yi])
                        self.y_lines.append(
                            MeshFunction(
                                lambda t: start + (end - start) * t,
                                lambda t: (end - start),
                                lambda t: Vector(),
                                transformations=transformations,
                                         domain=[0, 1],
                                         thickness=0.25*thickness,
                                         name='sub_grid_line_y_' + str(yi),
                                         scale=self.get_scales(), **kwargs)
                        )
                    y_last = y
            start = Vector([x_min_tic, 0, y])
            end = Vector([x_max_tic, 0, y])
            self.y_lines.append(
                MeshFunction(
                    lambda t: start + (end - start) * t,
                    lambda t: (end - start),
                    lambda t: Vector(),
                    transformations=transformations,
                             domain=[0, 1],
                             thickness=thickness,
                             name='grid_line_y_' + str(y),
                             scale=self.get_scales(), **kwargs)
            )

        offset = transition_time / 4 / len(self.x_lines)
        for i, line in enumerate(self.x_lines):
            self.add_object(line)
            line.grow(begin_time=begin_time + i * offset, transition_time=transition_time / 2, modus='from_start')

        offset = transition_time / 4 / len(self.y_lines)
        for i, line in enumerate(self.y_lines):
            self.add_object(line)
            line.grow(begin_time=begin_time + i * offset + transition_time / 2, transition_time=transition_time / 2,
                      modus='from_start')
        return begin_time+transition_time

    def draw_transformable_polar_grid(self, transformations, twists=[0], sub_grid=0,
                                      begin_time=0,center=Vector(),
                                      transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        '''
        draw gridlines that can be transformed

        :param twists:
        :param center:
        :param transformations:
        :param colors:
        :param sub_grid:
        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        '''

        thickness = get_from_kwargs(kwargs, 'thickness', 1)

        x_axis = self.axes[0]
        y_axis = self.axes[1]

        x_min_tic = np.Infinity
        x_max_tic = -np.Infinity
        y_min_tic = np.Infinity
        y_max_tic = -np.Infinity
        for x in x_axis.tic_values:
            if x < x_min_tic:
                x_min_tic = x
            if x > x_max_tic:
                x_max_tic = x
        for y in y_axis.tic_values:
            if y < y_min_tic:
                y_min_tic = y
            if y > y_max_tic:
                y_max_tic = y

        ext_x = x_max_tic - x_min_tic
        ext_y = y_max_tic - y_min_tic
        r_max = int(np.round(np.minimum(ext_x/2,ext_y/2)))

        n = int(np.maximum(len(x_axis.tic_values), len(y_axis.tic_values)))
        dr = r_max / n
        self.r_lines = []
        for i in range(1, n + 1):
            r = dr * i
            embedding = lambda t: Vector([r * np.cos(t), 0, r * np.sin(t)])+center
            velocity = lambda t: Vector([-r * np.sin(t), 0, r * np.cos(t)])
            acceleration = lambda t: Vector([-r * np.cos(t), 0, -r * np.sin(t)])
            self.r_lines.append(
                MeshFunction(embedding, velocity, acceleration, transformations=transformations, twists=twists,
                             domain=[0, 1.999 * np.pi],
                             thickness=thickness, name='grid_line_r' + str(r),
                             scale=self.get_scales(),
                             **kwargs)
            )
            for j in range(1,sub_grid):
                rs = r - dr / sub_grid * j
                embedding = lambda t: Vector([rs * np.cos(t), 0, rs * np.sin(t)])+center
                velocity = lambda t: Vector([-rs * np.sin(t), 0, rs * np.cos(t)])
                acceleration = lambda t: Vector([-rs * np.cos(t), 0, -rs * np.sin(t)])
                self.r_lines.append(
                    MeshFunction(embedding, velocity, acceleration, transformations=transformations, twists=twists,
                                 domain=[0, 1.99 * np.pi],
                                 thickness=0.25 * thickness, name='sub_grid_line_r' + str(rs), scale=self.get_scales(),
                                 **kwargs)
                )

        self.phi_lines = []
        dphi = 1.999 * np.pi / n
        for i in range(0, n):
            phi = dphi * i
            embedding = lambda t: t * Vector([r_max * np.cos(phi), 0, r_max * np.sin(phi)])+center
            velocity = lambda t: Vector([r_max * np.cos(phi), 0, r_max * np.sin(phi)])
            acceleration = lambda t: Vector()

            self.phi_lines.append(
                MeshFunction(embedding, velocity, acceleration, transformations=transformations, twists=twists,
                             domain=[0.001, 1],
                             thickness=thickness, name='grid_line_phi' + str(phi), scale=self.get_scales(),
                             **kwargs)
            )

            for j in range(1,sub_grid):
                phis = phi - j * dphi / sub_grid
                embedding = lambda t: t * Vector([r_max * np.cos(phis), 0, r_max * np.sin(phis)])+center
                velocity = lambda t: Vector([r_max * np.cos(phis), 0, r_max * np.sin(phis)])
                acceleration = lambda t: Vector()

                self.phi_lines.append(
                    MeshFunction(embedding, velocity, acceleration, transformations=transformations, twists=twists,
                                 domain=[0.001, 1],
                                 thickness=0.25 * thickness, name='sub_grid_line_phi' + str(phis),
                                 scale=self.get_scales(),
                                 **kwargs)
                )

        offset = transition_time / 4 / len(self.r_lines)
        scaled_center = Vector([
            center.x*self.get_scales()[0],
            center.y*self.get_scales()[1],
            center.z*self.get_scales()[2]])
        for i, line in enumerate(self.r_lines):
            self.add_object(line)
            line.grow(begin_time=begin_time + i * offset, transition_time=transition_time / 2,pivot=scaled_center)

        offset = transition_time / 4 / len(self.phi_lines)
        for i, line in enumerate(self.phi_lines):
            self.add_object(line)
            line.grow(begin_time=begin_time + i * offset + transition_time / 2, transition_time=transition_time / 2,
                      pivot=scaled_center)

        # for compatibility
        self.x_lines = self.r_lines
        self.y_lines = self.phi_lines

    def add_object(self, bobject):
        """ a bobject or plane is added to the coordinate system simply by establishing a blender
            parent-child relation.
        """
        obj = bobject.ref_obj
        obj.parent = self.ref_obj

    def add_objects(self, *bobjects):
        """
        more objects are added
        """
        if len(bobjects) == 1:  # make sure tuple and lists are accepted equally well
            bobjects = bobjects[0]
        else:
            bobjects = [*bobjects]
        if not isinstance(bobjects, list):
            bobjects = [bobjects]

        for bob in bobjects:
            self.add_object(bob)


class DynamicCoordinateSystem(CoordinateSystem):
    """
        Coordinate system, where the number label are digital ranges,
        otherwise it's a regular coordinate system
    """
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name','DynamicCoordinateSystem')
        super().__init__(dynamic=True,name=self.name, **kwargs)

    def scale_labels(self,scale=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for axis in self.axes:
            axis.scale_labels(scale,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def shift_labels(self,shift=[0,0],begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for s,axis in zip(shift,self.axes):
            axis.shift_labels(s,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def adjust_grid_for_shifting(self, scale=1, shift = [0,0], begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        # grid lines that move out on one line have to move in on the other side
        x_vals = self.axes[0].tic_values
        y_vals = self.axes[1].tic_values
        # calculate the shift back to the original scale
        shift_x = shift[0]
        shift_y = shift[1]

        max_x = max(x_vals)
        min_x = min(x_vals)
        max_y = max(y_vals)
        min_y = min(y_vals)

        direction = -self.coords2location(Vector([shift_x/scale, 0, 0]))
        self.axes[1].move(direction = direction,begin_time=begin_time,transition_time=transition_time)

        for val,xline  in zip(x_vals,self.x_lines):
            if val-shift_x>max_x:
                offset_t = transition_time*(val-max_x)/shift_x
                fraction = offset_t/transition_time
                xline.move(direction=fraction*direction, begin_time=begin_time, transition_time=fraction*transition_time)
                xline.toggle_hide(begin_time=begin_time+offset_t-1/FRAME_RATE)
                # I'm sorry to give up to solve this in general. Since it is only needed for this particular task
                # the y-coordinate is hard-coded, it should be the middle between y_max and y_min, however, due to scaling things might have changed
                jump_position = self.coords2location([min_x/scale, (min_y + max_y) / 2 / scale, 0])
                xline.move_to(target_location=jump_position, begin_time=begin_time + offset_t + 1 / FRAME_RATE, transition_time=0)
                xline.toggle_hide(begin_time=begin_time+offset_t+2/FRAME_RATE)
                rest = (transition_time - offset_t) / transition_time
                xline.move_to(target_location=jump_position+Vector(rest*direction),begin_time=begin_time+offset_t,transition_time=transition_time-offset_t)
            else:
                xline.move(direction=direction, begin_time=begin_time, transition_time=transition_time)

        # add last grid line
        new_line = Cylinder.from_start_to_end(start=self.coords2location([min_x/scale, min_y/scale]),
                                              end=self.coords2location([min_x/scale, max_y/scale]), color=self.grid_colors[0],
                                              radius=self.radii[0] * 0.5, name='grid_line_x_' + str(min_x+shift_x))
        # appear new line at the appropriate moment
        new_line.appear(begin_time=begin_time +transition_time, transition_time=0)
        self.x_lines.append(new_line)
        self.add_object(self.x_lines[-1])

        direction = -self.coords2location(Vector([0, shift_y/scale]))
        self.axes[0].move(direction=direction, begin_time=begin_time, transition_time=transition_time)

        for val,yline in zip(y_vals,self.y_lines):
            if val - shift_y > max_y:
                offset_t = transition_time * (val - max_y) / shift_y
                fraction = offset_t / transition_time
                yline.move(direction=fraction * direction, begin_time=begin_time,
                           transition_time=fraction * transition_time)
                yline.toggle_hide(begin_time=begin_time + offset_t - 1 / FRAME_RATE)
                # I'm sorry to give up to solve this in general. Since it is only needed for this particular task
                # the y-coordinate is hard-coded, it should be the middle between y_max and y_min, however, due to scaling things might have changed
                jump_position = self.coords2location([(min_x+max_x)/2/scale, min_y / scale, 0])
                yline.move_to(target_location=jump_position, begin_time=begin_time + offset_t + 1 / FRAME_RATE,
                              transition_time=0)
                yline.toggle_hide(begin_time=begin_time + offset_t + 2 / FRAME_RATE)
                rest = (transition_time - offset_t) / transition_time
                yline.move_to(target_location=jump_position + Vector(rest * direction),
                              begin_time=begin_time + offset_t, transition_time=transition_time - offset_t)
            else:
                yline.move(direction=direction, begin_time=begin_time, transition_time=transition_time)

        # add last grid line
        new_line = Cylinder.from_start_to_end(start=self.coords2location([min_x/scale, min_y / scale]),
                                              end=self.coords2location([max_x/scale, min_y / scale]),
                                              color=self.grid_colors[0],
                                              radius=self.radii[0] * 0.5, name='grid_line_y' + str(min_y+shift_y))
        # appear new line at the appropriate moment
        new_line.appear(begin_time=begin_time + transition_time, transition_time=0)
        self.y_lines.append(new_line)
        self.add_object(self.y_lines[-1])

        return begin_time+transition_time

    def adjust_grid_for_scaling(self,scale=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):

        x_vals = self.axes[0].tic_values
        for line,val in zip(self.x_lines, x_vals):
            direction = self.coords2location(Vector([val,0,0])-Vector([val/scale,0,0]))
            line.move(direction=-direction,begin_time=begin_time,transition_time=transition_time)
        y_vals = self.axes[1].tic_values
        for line, val in zip(self.y_lines, y_vals):
            direction = self.coords2location(Vector([0,val,  0]) - Vector([0,val / scale, 0]))
            line.move(direction=-direction, begin_time=begin_time, transition_time=transition_time)

        max_x = max(x_vals)
        min_x = min(x_vals)
        max_y = max(y_vals)
        min_y = min(y_vals)
        delta_x = x_vals[1]-x_vals[0]
        delta_y = y_vals[1]-y_vals[0]

        new_x_lines = int((max_x*scale-max_x)/delta_x)
        new_y_lines = int((max_y*scale-max_y)/delta_y)

        for i in range(new_x_lines):
            x = max_x+(i+1)*delta_x
            self.axes[0].tic_values.append(x)  # add values to keep track of the pre-image of each line
            new_line = Cylinder.from_start_to_end(start=self.coords2location([max_x, min_y]),
                                             end=self.coords2location([max_x, max_y]), color=self.grid_colors[0],
                                             radius=self.radii[0] * 0.5, name='grid_line_x_' + str(x))
            # appear new line at the appropriate moment
            offset_t = transition_time * (i + 1) / new_x_lines
            new_line.appear(begin_time=begin_time + offset_t, transition_time=1)
            # move new line to the appropriate position
            direction =  self.coords2location(Vector([max_x,0,  0]) - Vector([x / scale,0, 0]))
            new_line.move(direction=-direction,begin_time=begin_time+offset_t,transition_time=(transition_time-offset_t))
            self.x_lines.append(new_line)
            self.add_object(self.x_lines[-1])

        for i in range(new_y_lines):
            y = max_y + (i + 1) * delta_y
            self.axes[1].tic_values.append(y) # add values to keep track of the pre-image of each line
            new_line = Cylinder.from_start_to_end(start=self.coords2location([min_x, max_y]),
                                                  end=self.coords2location([max_x, max_y]), color=self.grid_colors[1],
                                                  radius=self.radii[0] * 0.5, name='grid_line_y_' + str(x))
            # appear new line at the appropriate moment
            offset_t = transition_time * (i + 1) / new_y_lines
            new_line.appear(begin_time=begin_time + offset_t, transition_time=1)
            # move new line to the appropriate position
            direction = self.coords2location(Vector([0, max_y, 0]) - Vector([0, y / scale, 0]))
            new_line.move(direction=-direction, begin_time=begin_time + offset_t,
                          transition_time=(transition_time - offset_t))
            self.y_lines.append(new_line)
            self.add_object(self.y_lines[-1])

        return begin_time+transition_time