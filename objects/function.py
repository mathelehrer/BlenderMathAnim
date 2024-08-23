import math
import bpy

import numpy as np
from mathutils import Vector

from appearance.textures import phase2rgb, make_complex_function_material, make_colorful_bezier_curve, \
    make_colorscript_bezier_curve, make_voronoi_bezier_curve, make_conformal_transformation_material, phase2rgb2
from interface import ibpy
from interface.ibpy import set_extrude, set_bevel, set_use_path, convert_to_mesh
from objects.bobject import BObject
from objects.cylinder import Cylinder
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME

class Function(BObject):
    """
    Create a function, based on a bezier curve

    which is a curve and it has to be associated with an
    existing coordinate system:

    example:
    Function([lambda phi: [phi, 0, abs_func(phi)],
              lambda phi: [phi, 0, func(phi)]],
             coord,
             domain=[-2, 2],
             num_points=100,
             color='hue_color',
             hue_functions=["x,x,*,y,y,*,-,1,-", "x,y,*,2,*"],
             type='PARAMETRIC',
             name='RealPart')

    remarks:


    * the combination color='hue_color' and hue_functions=[...] provide
      the possibility to create a complex hue-shader for the functions

    * alternatively use a color value from ['example','text','important'] to
      match the color scheme of the presentation

    * special points split the curve into pieces between the special points and the domain boundaries

    known issues:
    * it is important that each function has a unique name string,
      otherwise it will not be found and animated correctly

    """

    def __init__(self, mappings, coordinate_system,
                 domain=None,
                 num_points=100,
                 color='example',
                 colors=None,
                 script='',
                 hue_functions=["x", "y"],
                 color_mode='simple_color',
                 mode='2D',
                 **kwargs):
        """

        :param mappings:
        :param coordinate_system:
        :param domain:
        :param num_points:
        :param color:
        :param colors:
        :param script:
        :param hue_functions:
        :param color_mode: 'simple_color', 'hue_colors', 'script', 'voronoi'
        :param mode:
        :param kwargs:
        """
        self.kwargs = kwargs
        if not isinstance(mappings, list):
            mappings = [mappings]

        self.current_mapping = None  # keeps track of the shown mapping, when more than one mapping is defined for the
        # bobject
        self.kwargs = kwargs
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            kwargs['name'] = 'Function'
            name = 'Function'

        # constants
        self.eps = 0.1
        self.bevel_factor_mapping_end = self.get_from_kwargs('bevel_factor_mapping_end', "RESOLUTION")
        self.mappings = mappings
        self.coordinate_system = coordinate_system

        if domain is None:
            # TODO this might cause troubles since a three dimensional coordinate system might return a
            # TODO two-dimensional domain
            self.domain = coordinate_system.get_domain()
        else:
            self.domain = domain
        self.num_points = num_points
        self.curve = ibpy.get_new_curve(name, num_points)

        list_of_points = []
        x_range = self.domain[1] - self.domain[0]
        x_min = self.domain[0]
        for mapping in self.mappings:
            points = []
            for i in range(-1, num_points + 1):
                for j in range(0, 3):
                    x = x_min + (3 * i + j) * x_range / num_points / 3
                    # evaluate the bobject at num_points+2 points and two sub-steps for each interval,
                    # starting one main interval before t_min and finishing one main interval after t_max
                    if mode == 'PARAMETRIC':
                        vec = mapping(x)
                        # points.append(self.coordinate_system.coords2location_relative2coordinate_system(vec)+Vector(self.coordinate_system.location_of_origin))
                        if self.coordinate_system:
                            points.append(self.coordinate_system.coords2location_relative2coordinate_system(vec))
                        else:
                            points.append(vec)
                    elif mode == '2D':
                        y = self.save_eval(mapping, x)
                        if self.coordinate_system:
                            points.append(self.coordinate_system.coords2location_relative2coordinate_system([x, y]))
                        else:
                            points.append([x, y])

            # add one last point to finish the additional interval at the end for the calculation of the handles
            x = x_min + (num_points + 1) * x_range / num_points
            if mode == 'PARAMETRIC':
                vec = mapping(x)
                if self.coordinate_system:
                    points.append(self.coordinate_system.coords2location_relative2coordinate_system(vec))
                else:
                    points.append(vec)
            elif mode == '2D':
                y = self.save_eval(mapping, x)
                if self.coordinate_system:
                    points.append(self.coordinate_system.coords2location_relative2coordinate_system([x, y]))
                else:
                    points.append([x, y])

            # print("number of points: "+str(len(points)))
            # convert data points to bezier control points
            for p in range(0, len(points) - 1, 3):
                # print(p)
                xs = [points[p + j].x for j in range(0, 4)]
                ys = [points[p + j].y for j in range(0, 4)]

                handle_one_x, handle_two_x = derive_bezier_handles(*xs, 1 / 3, 2 / 3)
                handle_one_y, handle_two_y = derive_bezier_handles(*ys, 1 / 3, 2 / 3)

                points[p + 1] = Vector([handle_one_x, handle_one_y, points[p + 1].z])
                points[p + 2] = Vector([handle_two_x, handle_two_y, points[p + 2].z])

            list_of_points.append(points)

        ref_obj = ibpy.new_curve_object(name, self.curve)
        self.dialer = []  # dial between different hue colors
        for map_count, points in enumerate(list_of_points):
            if map_count == 0:
                for i in range(1, num_points + 2):
                    ibpy.set_bezier_point_of_curve(self.curve, i - 1, points[3 * i], points[3 * i - 1],
                                                   points[3 * i + 1])
                if len(list_of_points) > 1:
                    old_shape_key = ibpy.add_shape_key(ref_obj, 'Basis')
            else:
                old_shape_key = ibpy.add_shape_key(ref_obj, name + str(map_count), old_shape_key)
                for i in range(1, num_points + 1):
                    ibpy.reset_bezier_point(old_shape_key.data[i - 1], points[3 * i], points[3 * i - 1],
                                            points[3 * i + 1])

        if color_mode == 'script':
            super().__init__(obj=ref_obj, **kwargs)
            make_colorscript_bezier_curve(self, script, scale=coordinate_system.get_scales(), emission_strength=0.3)
        elif color_mode == 'hue_color':
            super().__init__(obj=ref_obj, **kwargs)
            if 'input' in kwargs:
                input = kwargs.pop('input')
            else:
                input='geometry_position'
            self.dialer = make_colorful_bezier_curve(self, hue_functions, scale=coordinate_system.get_scales(),
                                                     emission_strength=0.3,input=input)
        elif color_mode == 'simple_color':
            super().__init__(obj=ref_obj, color=color, **kwargs)
        elif color_mode == 'voronoi':
            super().__init__(obj=ref_obj, **kwargs)
            self.dialer = make_voronoi_bezier_curve(self, colors, scale=coordinate_system.get_scales(),
                                                    emission_strength=0.3)

        thickness = super().get_from_kwargs("thickness", 1)
        self.bevel_depth = 0.05 * thickness
        extrude = self.get_from_kwargs('extrude', 0.005)
        self.extrude = extrude

        # it can be linked without hesitation. The curve only appears once a bevel depth is set,
        # which only happens in the show bobject
        ibpy.link(self.ref_obj)
        if self.coordinate_system:
            self.coordinate_system.add_object(self)

    def grow(self, start_factor=0, end_factor=1,
             begin_time=0,
             transition_time=OBJECT_APPEARANCE_TIME, inverted=False,
             **kwargs):
        super().appear(begin_time=begin_time, transition_time=OBJECT_APPEARANCE_TIME)
        set_use_path(self, True)
        # TODO: Create parameter for thickness
        set_bevel(self, self.bevel_depth, caps=True)
        set_extrude(self, self.extrude)
        appear_frame = begin_time * FRAME_RATE
        ibpy.grow_curve(self, appear_frame, transition_time, inverted, end_factor=end_factor, start_factor=start_factor,bevel_factor_mapping_end=self.bevel_factor_mapping_end)
        self.current_mapping = 0
        return begin_time+transition_time

    def next(self,
             begin_time=None,
             transition_time=OBJECT_APPEARANCE_TIME):
        """
            morph from one mapping to the next

            :param begin_time:
            :param transition_time:
            :return:
        """

        begin_frame = begin_time * FRAME_RATE

        if len(self.dialer) > self.current_mapping:
            dialer_index = self.current_mapping
            current_dialer = self.dialer[dialer_index]
            ibpy.insert_keyframe(current_dialer, 'default_value', begin_frame)
            current_dialer.default_value = 1
            ibpy.insert_keyframe(current_dialer, 'default_value', begin_frame + transition_time * FRAME_RATE)

        ibpy.morph_to_next_shape(self.ref_obj, self.current_mapping, begin_frame, transition_time * FRAME_RATE)
        self.current_mapping += 1
        return begin_time+transition_time

    def save_eval(self, mapping, x):
        try:
            y = mapping(x)
        except ZeroDivisionError:
            y = mapping(x + self.eps)
        return y

    def appear(self,
               begin_time=0,
               transition_time=0,
               **kwargs):
        ibpy.set_bevel(self, self.bevel_depth)
        ibpy.fade_in(self, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        self.current_mapping = 0


class SimpleFunction(BObject):
    def __init__(self, mappings, coordinate_system,
                 domain=None,
                 num_points=100,
                 radius=0.1,
                 color='example',
                 mode='2D',
                 **kwargs):
        if not isinstance(mappings, list):
            mappings = [mappings]

        self.current_mapping = None  # keeps track of the shown mapping, when more than one mapping is defined for the
        # bobject
        self.kwargs = kwargs
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            kwargs['name'] = 'Function'
            name = 'Function'

        # constants
        self.eps = 0.1

        self.mappings = mappings
        self.coordinate_system = coordinate_system

        if domain is None:
            # TODO this might cause troubles since a three dimensional coordinate system might return a
            # TODO two-dimensional domain
            self.domain = coordinate_system.get_domain()
        else:
            self.domain = domain
        self.num_points = num_points
        self.curve = ibpy.get_new_curve(name, num_points)

        list_of_points = []
        x_range = self.domain[1] - self.domain[0]
        x_min = self.domain[0]
        for mapping in self.mappings:
            points = []
            for i in range(0, num_points + 1):
                x = x_min + i * x_range / num_points
                # evaluate the bobject at num_points+2 points and two sub-steps for each interval,
                # starting one main interval before t_min and finishing one main interval after t_max
                if mode == '2D':
                    y = self.save_eval(mapping, x)
                    points.append(self.coordinate_system.coords2location_relative2coordinate_system([x, y]))
                elif mode == 'PARAMETRIC':
                    vec = mapping(x)
                    points.append(self.coordinate_system.coords2location_relative2coordinate_system(vec))

            # print("number of points: "+str(len(points)))
            # convert data points to bezier control points
            self.pieces = []
            for p in range(1, len(points)):
                start = points[p - 1]
                end = points[p]

                self.pieces.append(
                    Cylinder(start=start, end=end, radius=radius, color=color, name=name + "_piece_" + str(p)))

        super().__init__(children=self.pieces, **kwargs)
        self.coordinate_system.add_object(self)

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, inverted=False):
        def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, inverted=False):
            if inverted:
                delta = transition_time / len(self.pieces)
                for i, piece in enumerate(reversed(self.pieces)):
                    piece.grow(begin_time=begin_time + i * delta, transition_time=delta, inverted=inverted)
            else:
                delta = transition_time / len(self.pieces)
                for i, piece in enumerate(self.pieces):
                    piece.grow(begin_time=begin_time + i * delta, transition_time=delta, inverted=inverted)

    def save_eval(self, mapping, x):
        try:
            y = mapping(x)
        except ZeroDivisionError:
            y = mapping(x + self.eps)
        return y


class PieceWiseFunction(BObject):
    def __init__(self, mappings, coordinate_system,
                 singular_points,
                 domain=[0, 1],
                 num_points=100,
                 color='example',
                 script='',
                 hue_functions=["x", "y"],
                 mode='2D', EPS=0.001,
                 **kwargs):

        self.kwargs = kwargs
        name = self.get_from_kwargs('name', 'Piecewise_Function')
        if 'name' in kwargs:
            kwargs.pop('name')

        self.singular_points = singular_points
        self.domain = domain

        self.singular_points.append(self.domain[0])
        self.singular_points.append(self.domain[1])

        self.singular_points.sort()
        relevant_points = []
        for i, p in enumerate(self.singular_points):
            if self.domain[0] <= p <= self.domain[1]:
                relevant_points.append(p)

        self.singular_points = relevant_points

        self.pieces = []
        for i in range(1, len(self.singular_points)):
            start = self.singular_points[i - 1] + EPS
            end = self.singular_points[i] - EPS

            self.pieces.append(Function(mappings, coordinate_system,
                                        domain=[start, end],
                                        num_points=num_points,
                                        color=color,
                                        script=script,
                                        hue_functions=hue_functions,
                                        mode=mode, name=name + "_piece_" + str(i),
                                        **kwargs))

            super().__init__(objects=self.pieces)

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, inverted=False):
        if inverted:
            delta = transition_time / len(self.pieces)
            for i, piece in enumerate(reversed(self.pieces)):
                piece.grow(begin_time=begin_time + i * delta, transition_time=delta, inverted=inverted)
        else:
            delta = transition_time / len(self.pieces)
            for i, piece in enumerate(self.pieces):
                piece.grow(begin_time=begin_time + i * delta, transition_time=delta, inverted=inverted)


class MeshFunction(BObject):
    def __init__(self, embedding=lambda x: Vector([x, 0, 0]),
                 velocity=lambda x: Vector([1, 0, 0]),
                 acceleration=lambda x: Vector(),
                 transformations=[],color=None,
                 domain=[0, 1], num_points=100, thickness=1, smooth=True, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'MeshFunction')
        self.embedding = embedding
        self.velocity = velocity
        self.acceleration= acceleration

        self.vertex_colors = []
        mesh = self.create_mesh(resolution=num_points, radius=0.1 * thickness, domain=domain)
        super().__init__(mesh=mesh, name=self.name, **self.kwargs)
        if color is None:
            ibpy.create_color_map_for_mesh(self, self.vertex_colors, self.name, **self.kwargs)

        if smooth:
            ibpy.smooth_mesh(self)

        ## create shape keys for the different transformations

        if len(transformations) > 0:
            old_sk = ibpy.add_shape_key(self, 'Basis')
            count=1
            for t in transformations:
                old_sk = ibpy.add_shape_key(self, name='Profile' + str(count), previous=old_sk)
                self.create_mesh(resolution=num_points, radius=0.1 * thickness, domain=domain, trafo=t,
                                  shape_keys=old_sk)
                count+=1

            self.current_mapping = 0

    def frame(self, t, pos, velocity, acceleration,trafo=None):
        v0 = velocity(t)
        if trafo:
            v = Vector(trafo.d_trafo(pos).dot(v0))
        else:
            v = v0
        tau = v.normalized()
        s = v.length  # speed
        a0 = acceleration(t)
        if trafo:
            a = Vector(trafo.d_trafo(pos).dot(a0)+trafo.d2_trafo(pos).dot(v0).dot(v0))
        else:
            a=a0
        if s==0:
            i=0
        n = a - tau.dot(a) * tau
        c = n.length  # curvature
        if c == 0:  # no curvature (straight line) n and b can be chosen arbitrary, as long as they are orthogonal to v
            h = Vector([0, 1, 0])
            n = tau.cross(h)
            if np.isclose(n.length, 0):
                h = Vector([0, 0, 1])
                n = tau.cross(h)
            n = n.normalized()
            b = tau.cross(n)
            b = b.normalized()
        else:
            n /= c
            b = tau.cross(n)
            b = b.normalized()
        return tau, n, b

    def create_mesh(self, resolution=100, radius=1, domain=[0, 1], trafo=None, shape_keys=None):
        t_max = domain[1]
        t_min = domain[0]
        dt = (t_max - t_min) / resolution

        if not shape_keys:
            vertices = []
            edges = []
            faces = []

        r = radius
        count = 0
        for i in range(0, resolution + 1):
            t = i * dt+t_min
            v0 = self.embedding(t)
            if trafo:
                v0=trafo.trafo(v0)
            z = v0.x + 1j * v0.z
            color = phase2rgb(np.angle(z))
            t, n, b = self.frame(t, v0, self.velocity, self.acceleration,trafo=trafo)
            # create vertices
            res2 = np.maximum(8, int(resolution * radius))
            d_phi = np.pi * 2 / res2

            # twist basis if necessary
            # use Rodrigues' formula
            psi = 0
            n2 = n*np.cos(psi)+t.cross(n)*np.sin(psi)+t.dot(n)*t*(1-np.cos(psi))
            b2 = b * np.cos(psi) + t.cross(b) * np.sin(psi) + t.dot(b) * t*(1-np.cos(psi))

            for j in range(0, res2):
                phi = j * d_phi
                v = v0 + n2 * (r * np.cos(phi)) + b2 * (r * np.sin(phi))
                if not shape_keys:
                    vertices.append(v)
                    self.vertex_colors.append(color)
                    index = len(vertices)
                    if j > 0:
                        edges.append([index - 1, index - 2])
                    if i > 0:
                        edges.append([index - 1, index - res2 - 1])
                    if i > 0 and j > 0:
                        faces.append([index - 1, index - 2, index - 2 - res2, index - 1 - res2])
                else:
                    shape_keys.data[count].co = v
                    count += 1

            if not shape_keys:
                edges.append([index - 1, index - res2])
                if i > 0:
                    faces.append([index - res2, index - 1, index - 1 - res2, index - 2 * res2])

        if not shape_keys:
            new_mesh = bpy.data.meshes.new(self.name + '_mesh')
            new_mesh.from_pydata(vertices, edges, faces)
            new_mesh.update()
            return new_mesh

    def next(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        appear_frame = begin_time * FRAME_RATE
        ibpy.morph_to_next_shape(self.ref_obj, self.current_mapping, appear_frame, transition_time * FRAME_RATE)
        self.current_mapping += 1
        print("Next shape at " + str(begin_time) + " with transition time " + str(transition_time))

########################
## Static functions ####
########################


def derive_bezier_handles(a, b, c, d, tb, tc):
    """
    TODO: for speed up, this can be optimized, when tb and tc are fixed and the calculations simplify

    Derives bezier handles by using the start and end of the curve with 2 intermediate
    points to use for interpolation.
    :param a:
        The start point.
    :param b:
        The first mid-point, located at `tb` on the bezier segment, where 0 < `tb` < 1.
    :param c:
        The second mid-point, located at `tc` on the bezier segment, where 0 < `tc` < 1.
    :param d:
        The end point.
    :param tb:
        The position of the first point in the bezier segment.
    :param tc:
        The position of the second point in the bezier segment.
    :return:
        A tuple of the two intermediate handles, that is, the right handle of the start point
        and the left handle of the end point.
    """

    # Calculate matrix coefficients
    matrix_a = 3 * math.pow(1 - tb, 2) * tb
    matrix_b = 3 * (1 - tb) * math.pow(tb, 2)
    matrix_c = 3 * math.pow(1 - tc, 2) * tc
    matrix_d = 3 * (1 - tc) * math.pow(tc, 2)

    # Calculate the matrix determinant
    matrix_determinant = 1 / ((matrix_a * matrix_d) - (matrix_b * matrix_c))

    # Calculate the components of the target position vector
    final_b = b - (math.pow(1 - tb, 3) * a) - (math.pow(tb, 3) * d)
    final_c = c - (math.pow(1 - tc, 3) * a) - (math.pow(tc, 3) * d)

    # Multiply the inversed matrix with the position vector to get the handle points
    bezier_b = matrix_determinant * ((matrix_d * final_b) + (-matrix_b * final_c))
    bezier_c = matrix_determinant * ((-matrix_c * final_b) + (matrix_a * final_c))

    # Return the handle points
    return bezier_b, bezier_c
