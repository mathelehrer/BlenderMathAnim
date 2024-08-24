from collections import OrderedDict

import numpy as np
from mathutils import Vector
from mpmath import mp

from interface import ibpy
from mathematics.polynomial import Polynomial
from objects.arrow import Arrow
from objects.logo import Logo
from objects.coordinate_system import CoordinateSystem
from objects.cylinder import Cylinder
from objects.digital_number import DigitalNumber
from objects.display import DisplayOld, Display
from objects.empties import EmptyCube
from objects.function import Function, PieceWiseFunction
from objects.plane_complex import ComplexPlane
from objects.plane_with_singular_points import ComplexPlaneWithSingularPoints
from objects.geometry.sphere import Sphere
from objects.tex_bobject import SimpleTexBObject, TexBObject
from objects.updateable_text import UpdateableTextWithNumbers
from perform.scene import Scene
from utils.constants import UP, SMALL_BUFF, RIGHT, FRAME_RATE
from utils.utils import print_time_report


def flatten(colors):
    colors_flat = [col for sublist in colors for col in sublist]
    return colors_flat


def prime_factors(i):
    factors = []
    while i > 1:
        for d in range(2, i + 1):
            if i % d == 0:
                factors.append(d)
                i /= d
                i = int(i)
                break
    return factors


def get_color(line):
    if 'cdot' in line:
        return ['text']
    else:
        pos = line.index('=')
        colors = [['text'] * (pos+1), ['important']]
        return flatten(colors)


def get_prime_string(i):
    factors = prime_factors(i)
    if len(factors) == 1:
        return str(i) + "=" + str(i)
    else:
        out_string = str(i) + "="
        for f, factor in enumerate(factors):
            if f == 0:
                out_string += str(factor)
            else:
                out_string += r"\cdot" + str(factor)
        return out_string


def zeta_sum(z, n):
    result = 1
    for i in range(2, n):
        result += 1 / i ** z
    return result


def product(x, n):
    """
    product expansion formula for the 1/pi*sin(pi*x)
    :param x:
    :param n:
    :return:
    """
    result = Polynomial([1])
    if n == 0:
        return result.eval(x)
    result = result * Polynomial([0, 1])
    if n == 1:
        return result.eval(x)
    for i in range(1, n):
        if i % 2 == 0:
            s = 1
        else:
            s = -1
        result *= Polynomial([1, s * 1 / np.floor((i + 1) / 2)])
    return result.eval(x)


def product2(x, n):
    """
    product expansion formula for the 1/pi*sin(pi*x), where the similar linear terms are combined into quadratic terms
    :param x:
    :param n:
    :return:
    """
    result = Polynomial([1])
    if n == 0:
        return result.eval(x)
    result = result * Polynomial([0, 1])
    if n == 1:
        return result.eval(x)
    for i in range(1, n):
        result *= Polynomial([1, 0, - 1 / i / i])
    return result.eval(x)


class Primes2(Scene):
    def __init__(self):
        self.primes = None

        self.sub_scenes = OrderedDict([
            ('decomposition', {'duration': 0}),
            ('decomposition_complex', {'duration': 0}),
            ('real_functions', {'duration': 0}),
            ('complex_functions', {'duration': 60}),
            ('zeta_sum', {'duration': 0}),
            ('zeta_continuation', {'duration': 0}),
            ('decomposition_poles', {'duration': 0}),
            ('show_zeros', {'duration': 0}),
            ('result', {'duration': 0}),
            ('psi_contributions', {'duration': 0}),
            ('psi_contributions_cumulative', {'duration': 0}),
            ('prime_counting_functions', {'duration': 0}),
            ('overview', {'duration': 0}),
            ('zeta_decomposition', {'duration': 0}),
            ('algebra', {'duration': 0}),
            ('contour_integral', {'duration': 0}),
            ('intro', {'duration': 0}),
            ('branding', {'duration': 0}),
        ])
        # in the super constructor the timing is set for all scenes
        super().__init__(light_energy=2)

    def real_functions(self):
        cues = self.sub_scenes['real_functions']
        t0 = 0  # cues['start']
        digits = 1  # 0 for testing
        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[6, -20, 0])
        display = DisplayOld()
        display.appear(begin_time=t0 + 0, transition_time=1)

        #####################
        # coordinate system #
        #####################
        coord = CoordinateSystem(dim=2, lengths=[10, 10], radii=[0.03, 0.03], domains=[[-2, 2], [-1, 3]],
                                 all_n_tics=[4, 4], location_of_origin=[0, 0, -3],
                                 labels=["x", "y"], all_tic_labels=[np.arange(-2, 2.1, 1), np.arange(-1, 3.1, 1)],
                                 materials=['drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=t0 + 0, transition_time=5)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{Real functions}", color='important', aligned='center')
        display.set_title(title)
        title.write(begin_time=1, transition_time=5)

        text = SimpleTexBObject(r"x \mapsto y=x^2-1", color=['text', 'important', 'important', 'example'],
                                aligned='left')
        display.add_text(text, indent=1.3)
        text.write(begin_time=6, transition_time=3)

        lines = [SimpleTexBObject(r"-2 \mapsto +3", color=['text', 'text', 'important', 'important', 'example'],
                                  aligned='left'),
                 SimpleTexBObject(r"-1 \mapsto +0", color=['text', 'text', 'important', 'important', 'example'],
                                  aligned='left'),
                 SimpleTexBObject(r" 0 \mapsto -1", color=['text', 'important', 'important', 'example'],
                                  aligned='left'),
                 SimpleTexBObject(r"+1 \mapsto +0", color=['text', 'text', 'important', 'important', 'example'],
                                  aligned='left'),
                 SimpleTexBObject(r"+2 \mapsto +3", color=['text', 'text', 'important', 'important', 'example'],
                                  aligned='left')]

        for i, line in enumerate(lines):
            display.set_cursor_to_start_of_next_line()
            if i == 2:
                indent = 0.3
            else:
                indent = 0
            display.add_text(line, indent=indent)
            line.write(begin_time=10 + i / (len(lines) - 1) * 10, transition_time=1)

        ######################
        # coordinate spheres #
        ######################
        t0 += 10
        r = 0.1

        sphere_x = Sphere(r, resolution=5, mesh_type="uv", color='text', location=coord.coords2location([-2, 0]))
        sphere_x.appear(begin_time=t0, transition_time=1)

        sphere_y = Sphere(r, resolution=5, mesh_type="uv", color='example', location=coord.coords2location([0, 3]))
        sphere_y.appear(begin_time=t0, transition_time=1)

        xPos = DigitalNumber(-2, number_of_digits=digits, color='text')
        xPos.write(begin_time=t0, transition_time=1)
        xPos.next_to(sphere_x, UP, buff=SMALL_BUFF, shift=0.2 * RIGHT)

        sphere_x.move(Vector(coord.coords2location([2, 0]) - coord.coords2location([-2, 0])), begin_time=t0 + 2,
                      transition_time=10)
        xPos.update_value(lambda frm: coord.location2coords(sphere_x.get_location_at_frame(frm))[0], begin_time=t0 + 2,
                          transition_time=10)

        yPos = DigitalNumber(3, number_of_digits=digits, color='example')
        yPos.write(begin_time=t0, transition_time=1)
        yPos.next_to(sphere_y, RIGHT, buff=SMALL_BUFF, shift=0.4 * RIGHT)

        sphere_y.update_position(lambda frm: coord.coords2location(
            [0, coord.location2coords(sphere_x.get_location_at_frame(frm))[0] ** 2 - 1]), begin_time=t0 + 2,
                                 transition_time=10)
        yPos.update_value(lambda frm: coord.location2coords(sphere_x.get_location_at_frame(frm))[0] ** 2 - 1,
                          begin_time=t0 + 2,
                          transition_time=10)

        ############
        # Function #
        ############

        t0 += 15

        func_tracer = Sphere(0.1, resolution=5, mesh_type='uv', color='important',
                             location=coord.coords2location([2, 3]))
        func_tracer.appear(begin_time=t0 + 1, transition_time=1)

        t0 += 5
        function = Function(lambda x: x ** 2 - 1, coord, domain=[-2, 2], color='important')
        function.grow(begin_time=t0, transition_time=10, inverted=True)

        sphere_x.move(Vector(coord.coords2location([-2, 0]) - coord.coords2location([2, 0])), begin_time=t0,
                      transition_time=10)
        xPos.update_value(lambda frm: coord.location2coords(sphere_x.get_location_at_frame(frm))[0],
                          begin_time=t0,
                          transition_time=10)

        sphere_y.update_position(lambda frm: coord.coords2location(
            [0, coord.location2coords(sphere_x.get_location_at_frame(frm))[0] ** 2 - 1]), begin_time=t0,
                                 transition_time=10)
        yPos.update_value(lambda frm: coord.location2coords(sphere_x.get_location_at_frame(frm))[0] ** 2 - 1,
                          begin_time=t0,
                          transition_time=10)

        func_tracer.update_position(lambda frm:
                                    [sphere_x.get_location_at_frame(frm)[0], 0, sphere_y.get_location_at_frame(frm)[2]],
                                    begin_time=t0, transition_time=10
                                    )

        x_old = 3
        label_count = 0
        limits = [2, 1, 0, -1, -1.99, -3]
        labels = ["(2|3)", "(1|0)", "(0|-1)", "(-1|0)", "(-2|3)"]
        label_shifts = [[0.1, 0], [0.1, 0.1], [-0.15, -0.125], [0.05, 0.09], [0.1, 0]]
        # this is a bit dirty. The labels are displayed when the sphere is in the right place.
        # this should be solved more elegantly later
        for frame in range(t0 * FRAME_RATE, (t0 + 10) * FRAME_RATE + 1):
            x = coord.location2coords(sphere_x.get_location_at_frame(frame))[0]
            y = coord.location2coords(sphere_y.get_location_at_frame(frame))[1]
            if x < limits[label_count]:
                sphere = Sphere(r, resolution=5, mesh_type="uv", color='important',
                                location=coord.coords2location([np.round(x), np.round(y)]))
                label = SimpleTexBObject(labels[label_count], color='important', aligned="left",
                                         location=coord.coords2location(
                                             [x + label_shifts[label_count][0], y + label_shifts[label_count][1]]))
                label.write(begin_time=frame / FRAME_RATE, transition_time=1)
                sphere.appear(begin_time=frame / FRAME_RATE, transition_time=0.1)
                label_count += 1

    def decomposition(self):
        cues = self.sub_scenes['decomposition']
        t0 = cues['start']

        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[6, -20, 0])
        display = DisplayOld()
        display.appear(begin_time=0, transition_time=1)

        #####################
        # coordinate system #
        #####################
        coord = CoordinateSystem(dim=2, lengths=[10, 10], radii=[0.03, 0.03], domains=[[-8, 8], [-2, 2]],
                                 all_n_tics=[8, 8],
                                 labels=["x", "y"], all_tic_labels=[np.arange(-8, 9, 2), np.arange(-2, 2.1, 0.5)],
                                 materials=['drawing', 'drawing']
                                 )
        coord.appear(begin_time=0, transition_time=5)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{Product composition of functions}", color='important', aligned='center')
        display.set_title(title)
        title.write(begin_time=1, transition_time=3)

        text_bits = ["f(x)=1", r"\cdot x"]
        for i in range(1, 6):
            text_bits.append(r"\cdot \left(1-{x\over " + str(i) + r"}\right)")
            text_bits.append(r"\cdot \left(1+{x\over " + str(i) + r"}\right)")

        text_bits.append(r"\cdot \dotsc ")
        text_bits.append(r"\cdot\left(1+{x\over 200}\right)\cdot\left(1+{x\over 200}\right)")
        f1 = TexBObject(*text_bits, aligned='left')
        display.add_text(f1, indent=1.45)

        for i in range(len(text_bits) - 2):
            f1.write(expression_index=i, begin_time=5 + 3 * i, transition_time=2)

        f1.write(expression_index=len(text_bits) - 2, begin_time=47, transition_time=2)
        f1.write(expression_index=len(text_bits) - 1, begin_time=70, transition_time=2)

        sine_text = SimpleTexBObject(r"f(x)={1\over \pi}\sin(\pi x)", color='important', aligned='left')
        display.set_cursor_to_start_of_next_line(indent=0)

        display.add_text(sine_text)
        sine_text.write(begin_time=75, transition_time=2)

        lambdas = [lambda x, i=i: product(x, i) for i in range(
            12)]  # remark: if written with just i instead of i=i only the last value of i is used for all lambdas

        # functions

        # all products of the sine(pi*x)  in the interval [-5,5]
        bobject = Function(
            lambdas, coord, color='text', num_points=50,
            name='Function_one')

        bobject.grow(begin_time=5, transition_time=2)
        for i in range(11):
            bobject.next(begin_time=8 + 3 * i, transition_time=2)

        # continue with zeros beyond

        limit = 200
        lambdas2 = [lambda x, i=i: product(x, i) for i in range(11,
                                                                limit)]
        # remark: if written with just i instead of i=i only the last value of i is used for all lambdas
        function2 = Function(lambdas2, coord, color='text', num_points=50, name='Function_two')
        function2.grow(begin_time=44, transition_time=1)
        bobject.disappear(begin_time=45, transition_time=1)

        time = 45
        for i in range(1, limit - 12):
            delta = 4 / i
            function2.next(begin_time=time, transition_time=delta)
            time += 2 * delta
        print("final time" + str(time))

        sine = Function(lambda x: np.sin(x * np.pi) / np.pi, coord, color='important', num_points=150, name='Sine')
        sine.appear(begin_time=65, transition_time=20)

    def decomposition_complex(self):
        cues = self.sub_scenes['decomposition_complex']
        t0 = cues['start']

        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[4, -20, 5])
        display = DisplayOld(location=[11, 0, 5])
        display.appear(begin_time=0, transition_time=1)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{Composition of complex functions}", color='important',
                                 aligned='center', name='title')
        display.set_title(title)
        title.write(begin_time=1, transition_time=3)

        text_bits = [r"f(z)=\pi", r"\cdot z"]
        for i in range(1, 6):
            text_bits.append(
                r"\cdot \left(1-{z\over " + str(i) + r"}\right)\cdot \left(1+{z\over " + str(i) + r"}\right)")
        text_bits.append(r"\cdot \dotsc ")
        f1 = TexBObject(*text_bits, aligned='left', name='lines1')
        display.add_text(f1, indent=0)

        for i in range(len(text_bits)):
            f1.write(expression_index=i, begin_time=5 + 3 * i, transition_time=2)

        sine_text = SimpleTexBObject(r"f(z)=\sin(\pi z)", color='important', aligned='left', text_size='large',
                                     name='sine')
        display.set_cursor_to_start_of_next_line(indent=0)

        display.add_text(sine_text)
        sine_text.write(begin_time=27, transition_time=2)

        lambdas = [lambda z, i=i: np.pi * product2(z, i) for i in range(
            7)]  # remark: if written with just i instead of i=i only the last value of i is used for all lambdas

        lambdas.append(lambda z: np.sin(np.pi * z))
        # functions

        #####################
        # coordinate system #
        #####################

        coord = CoordinateSystem(dim=3, lengths=[10, 10, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-5, 5], [-1, 1], [0, 10]],
                                 all_n_tics=[5, 4, 5], location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-5, 5.1, 2),
                                                 np.arange(-1, 1.1, 0.5),
                                                 np.arange(0, 10.1, 2)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=0, transition_time=5)

        complex_function = ComplexPlaneWithSingularPoints(coord,
                                                          lambdas,
                                                          u=[-5, 5], v=[-1, 1],
                                                          special_x=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                                          special_y=[0],
                                                          detail=10,
                                                          resolution=100,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          smooth=2, name='Function_one')
        complex_function.appear(begin_time=5, transition_time=2)

        delay = 0
        for i in range(8):
            if i + 1 == 8:
                delay = 3
            complex_function.next_shape(begin_time=5 + delay + 3 * i, transition_time=2)

        ######################
        # real sine function #
        ######################

        t0 = 30
        real_part = Function([lambda x: [x, 0, np.abs(np.sin(np.pi * x))], lambda x: [x, 0, np.sin(np.pi * x)]], coord,
                             domain=[-5, 5], num_points=100, color_mode='hue_color',
                             hue_functions=["pi,x,*,sin", "y"], mode='PARAMETRIC', name='RealPart',
                             thickness=0.1)
        real_part.grow(begin_time=t0, transition_time=5)
        real_part.next(begin_time=t0 + 7, transition_time=5)

        #################
        # Camera motion #
        #################

        # only do this, once all objects are created, since the keyframe of the camera empty is disturbed by the
        # creation of other objects. No idea, why it is.

        empty = EmptyCube(location=[4, 0, 5], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        t0 = 45
        ibpy.camera_move([0, 20, 20], begin_time=t0, transition_time=5)
        display.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=5)

        t0 = 50
        f1.move([0, 0, -1], begin_time=t0, transition_time=5)
        sine_text.disappear(begin_time=t0, transition_time=3)
        complex_function.disappear(begin_time=t0, transition_time=5)
        real_part.disappear(begin_time=t0, transition_time=5)

        ##################################
        # Replay animation without shape #
        ##################################

        t0 = 55
        complex_function2 = ComplexPlaneWithSingularPoints(coord,
                                                           lambdas,
                                                           u=[-5, 5], v=[-1, 1],
                                                           special_x=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                                           special_y=[0],
                                                           detail=10,
                                                           resolution=100,
                                                           alpha=0.95,
                                                           metallic=0,
                                                           roughness=0.7,
                                                           emission_strength=0.1,
                                                           smooth=2, name='Function_two',
                                                           shape=False)
        complex_function2.appear(begin_time=t0, transition_time=5)

        delay = 0
        for i in range(8):
            if i + 1 == 8:
                delay = 3
            complex_function2.next_shape(begin_time=t0 + 5 + delay + 3 * i, transition_time=2)

        text_bits2 = [r"f(z)=\pi", r"\cdot z"]
        for i in range(1, 6):
            text_bits2.append(
                r"\cdot \left(1-{z\over " + str(i) + r"}\right)\cdot \left(1+{z\over " + str(i) + r"}\right)")
        text_bits2.append(r"\cdot \dotsc ")

        display.toTop()
        f2 = TexBObject(*text_bits2, aligned='left', name='lines2')
        display.add_text(f2, indent=0)

        # move things out of the way, when they are not needed to avoid interference
        # it would be nicer, if it was possible to make objects re-appear once they've disappeared
        f2.move([0, 0, -1], begin_time=0, transition_time=0)
        f2.move([0, 0, 1], begin_time=t0, transition_time=0)

        for i in range(len(text_bits2)):
            f2.write(expression_index=i, begin_time=t0 + 5 + 3 * i, transition_time=2)

        sine_text2 = SimpleTexBObject(r"f(z)=\sin(\pi z)", color='important', aligned='left', name='sine2',
                                      text_size='large')

        display.set_cursor_to_start_of_next_line(indent=0)
        display.add_text(sine_text2)
        sine_text2.write(begin_time=84, transition_time=2)

    def complex_functions(self):
        cues = self.sub_scenes['complex_functions']
        t0 = 0  # cues['start']
        digits = 0  # 0 for testing

        func = lambda z: z * z - 1
        abs_func = lambda z: np.abs(z * z - 1)

        ibpy.set_shadow(False)

        #####################
        # coordinate system #
        #####################

        coord = CoordinateSystem(dim=3, lengths=[10, 10, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-2, 2], [-2, 2], [-1, 3]],
                                 all_n_tics=[4, 4, 4], location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-2, 2.1, 1),
                                                 np.arange(-2, 2.1, 1),
                                                 np.arange(-1, 3.1, 1)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear_individually(begin_times=[t0 + 0, t0 + 15, t0 + 50], transition_times=[5, 5, 5])
        ibpy.set_camera_location(location=[0, -20, 0])
        # ibpy.set_camera_rotation([np.pi / 2, 0, 0])

        # for some unknown reason the coordinate system has to be set up before the empty,
        # otherwise the keyframes of the empties are disturbed
        coord2 = CoordinateSystem(dim=3, lengths=[10, 10, 10], radii=[0.03, 0.03, 0.03],
                                  domains=[[-2, 2], [-1, 1], [-1, 3]],
                                  all_n_tics=[4, 4, 4], location_of_origin=[13, 0, 0],
                                  labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                  all_tic_labels=[np.arange(-2, 2.1, 1),
                                                  np.arange(-2, 2.1, 1),
                                                  np.arange(-1, 3.1, 1)],
                                  materials=['drawing', 'drawing', 'drawing'],
                                  text_size='large'
                                  )

        coord2.appear_individually(begin_times=[t0 + 20, t0 + 20, t0 + 200], transition_times=[5, 5, 5])

        ########################
        # Display and function #
        ########################
        display = DisplayOld(location=[6, 0, 6], scales=[4, 1, 4], flat=True)
        display.appear(begin_time=30, transition_time=1)

        title = SimpleTexBObject(r"\text{Complex functions}", color='important', aligned='center')
        display.set_title(title)
        title.write(begin_time=31, transition_time=5)

        text = SimpleTexBObject(r"z \mapsto w=z^2-1", color=['text', 'important', 'important', 'example'],
                                aligned='left')
        display.add_text(text, indent=1.3)
        text.write(begin_time=36, transition_time=3)

        coord.rotate(rotation_euler=[0, 0, -np.pi / 4], begin_time=25, transition_time=5)
        coord2.rotate(rotation_euler=[0, 0, -np.pi / 4], begin_time=25, transition_time=5)

        src = Sphere(0.1, mesh_type="uv", resolution=5, color='text', location=coord.coords2location([1, 0, 0]))
        coord.add_object(src)
        src.appear(begin_time=30, transition_time=1)

        updater_duration = 15
        frame_duration = updater_duration * FRAME_RATE
        label = UpdateableTextWithNumbers(lambda x: r"{\rm e}^{" + str(x) + r"\pi i}",
                                          lambda frm: 2 * frm / frame_duration, color="text", aligned="left",
                                          text_size='large', number_of_digits=1)  # use number_of_digits=0 for debugging
        label.write(begin_time=30, transition_time=1)

        src.rotate(rotation_euler=[0, 0, 2 * np.pi], begin_time=35, transition_time=updater_duration, pivot=[0, 0, 0],
                   interpolation='LINEAR')
        label.update_value(begin_time=35, transition_time=updater_duration)
        label.update_position(lambda frm: -coord.coords2location([np.cos(frm / frame_duration * 2 * np.pi),
                                                                  np.sin(frm / frame_duration * 2 * np.pi),
                                                                  -0.2]),
                              begin_time=35, transition_time=updater_duration)

        img = Sphere(0.1, resolution=5, mesh_type="uv", color='example', location=[0, 0, 0])
        coord2.add_object(img)
        img.appear(begin_time=30, transition_time=1)

        label2 = UpdateableTextWithNumbers(lambda x, y: str(x) + str(y) + "i",
                                           lambda frm: [np.real(func(np.exp(2 * frm / frame_duration * np.pi * 1j))),
                                                        np.imag(func(np.exp(2 * frm / frame_duration * np.pi * 1j)))],
                                           color="example",
                                           aligned="center",
                                           text_size='large',
                                           rotation_euler=[np.pi / 2, 0, np.pi / 4],
                                           number_of_digits=1);
        label2.next_to(img, direction=UP)
        label2.write(begin_time=30, transition_time=1)
        label2.update_value(begin_time=35, transition_time=updater_duration)

        img.update_position(lambda frm: map_between_coordinate_systems(coord, coord2, src, frm, func),
                            begin_time=35, transition_time=updater_duration)

        ibpy.set_origin(img.ref_obj)

        t0 = 50
        label.disappear(begin_time=t0 + 5, transition_time=1)
        label2.disappear(begin_time=t0 + 5, transition_time=1)
        display.disappear(begin_time=t0 + 5, transition_time=5)

        #######################
        #  complex plane      #
        #######################

        c_plane = ComplexPlane(coord2, u=[-2, 2], v=[-2, 2], resolution=50)
        c_plane.appear(begin_time=t0 + 3, transition_time=1)

        coord.rotate(rotation_euler=[0, 0, 0], begin_time=t0 + 6, transition_time=5)
        coord2.rotate(rotation_euler=[0, 0, 0], begin_time=t0 + 6, transition_time=5)

        t0 += 11

        func_3d = Function(lambda phi: [np.cos(phi), np.sin(phi), np.abs(func(np.cos(phi) + 1j * np.sin(phi)))],
                           coordinate_system=coord,
                           domain=[0, 2 * np.pi], color_mode='hue_color',
                           hue_functions=["x,x,*,y,y,*,-,1,-", "x,y,*,2,*"],
                           num_points=100,
                           mode='PARAMETRIC', name="Trace1")
        func_3d.grow(begin_time=t0, transition_time=10)

        src.rotate(rotation_euler=[0, 0, 4 * np.pi], begin_time=t0, transition_time=10, pivot=[0, 0, 0])
        img.update_position(lambda frm: map_between_coordinate_systems(coord, coord2, src, frm, func),
                            begin_time=t0, transition_time=10)
        ibpy.set_origin(img.ref_obj)

        ######################
        # trace a few points #
        ######################

        t0 += 11
        vals = [0.5, 1.5, 2]

        for val in vals:
            src = Sphere(0.1, mesh_type="uv", resolution=5, color='text', location=coord.coords2location([val, 0, 0]))
            coord.add_object(src)
            src.appear(begin_time=t0, transition_time=1)
            src.rotate(rotation_euler=[0, 0, 2 * np.pi], begin_time=t0 + 1, transition_time=5, pivot=[0, 0, 0])

            img = Sphere(0.1, resolution=5, mesh_type="uv", color='example', location=[val * val - 1, 0, 0])
            coord2.add_object(img)
            img.appear(begin_time=t0, transition_time=1)

            img.update_position(lambda frm: map_between_coordinate_systems(coord, coord2, src, frm, func),
                                begin_time=t0 + 1, transition_time=5)
            ibpy.set_origin(img.ref_obj)

            func_next = Function(lambda phi: [val * np.cos(phi), val * np.sin(phi),
                                              np.abs(func(val * (np.cos(phi) + 1j * np.sin(phi))))],
                                 coordinate_system=coord,
                                 domain=[0, 2 * np.pi], color_mode='hue_color',
                                 hue_functions=["x,x,*,y,y,*,-,1,-", "x,y,*,2,*"], num_points=100,
                                 mode='PARAMETRIC', name="Trace" + str(val))
            func_next.grow(begin_time=t0 + 1, transition_time=5)
            t0 += 6

        ########################
        # grow complex surface #
        ########################

        final_plane = ComplexPlane(coord,
                                   [func, func],
                                   u=[-2, 2], v=[-2, 2],
                                   resolution=100,
                                   alpha=0.95,
                                   metallic=0,
                                   roughness=0.7,
                                   emission_strength=0.1,
                                   smooth=2)
        final_plane.appear(begin_time=t0, transition_time=5)

        t0 += 5
        final_plane.next_shape(begin_time=t0, transition_time=5)

        t0 += 5  # 100
        coord2.disappear(begin_time=t0, transition_time=5)

        t0 += 10
        real_part = Function([lambda phi: [phi, 0, abs_func(phi)], lambda phi: [phi, 0, func(phi)]], coord,
                             domain=[-2, 2], num_points=100, color_mode='hue_color',
                             hue_functions=["x,x,*,y,y,*,-,1,-", "x,y,*,2,*"], mode='PARAMETRIC', name='RealPart',
                             thickness=0.1)
        real_part.grow(begin_time=t0, transition_time=5)

        real_part.next(begin_time=t0 + 7, transition_time=5)

        #################
        # Camera motion #
        #################

        # only do this, once all objects are created, since the keyframe of the camera empty is disturbed by the
        # creation of other objects. No idea, why it is.

        empty = EmptyCube(location=[0, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        ibpy.camera_move([7, 0, 10], begin_time=10, transition_time=5)
        empty.move(direction=[7, 0., 0.], begin_time=10, transition_time=5)
        empty.move([0, 0, 4], begin_time=78, transition_time=10)

        ibpy.camera_move([-7, 0, -10], begin_time=100, transition_time=5)
        empty.move([-7, 0, -4], begin_time=100, transition_time=5)

        t0 += 12
        ibpy.camera_move([0, 0, 3], begin_time=t0, transition_time=5)
        empty.move([0, 0, 3], begin_time=t0, transition_time=5)
        ibpy.camera_move([0, 10, 20], begin_time=t0 + 5, transition_time=5)
        empty.move([0, 0, -3], begin_time=t0 + 5, transition_time=5)
        ibpy.camera_move([0, 10, 0], begin_time=t0 + 10, transition_time=10)

    def zeta_sum(self):
        cues = self.sub_scenes['zeta_sum']
        t0 = cues['start']

        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[4, -25, 5])
        display = DisplayOld(location=[13, 0, 7], scales=[5, 5, 5])
        display.appear(begin_time=0, transition_time=1)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{The }\zeta\text{-function definition}", color='important',
                                 aligned='center', name='title', text_size='large')
        display.set_title(title)
        title.write(begin_time=1, transition_time=3)

        text_bits = [r"\zeta(z)=1"]
        for i in range(2, 6):
            text_bits.append(
                r"+{1\over " + str(i) + r"^z}")
        text_bits.append(r"+ \dotsc ")
        f1 = TexBObject(*text_bits, aligned='left', name='lines1')
        display.set_cursor_to_start_of_next_line(indent=0)

        display.add_text(f1, indent=0)

        for i in range(len(text_bits)):
            f1.write(expression_index=i, begin_time=5 + 3 * i, transition_time=2)

        zeta_text = SimpleTexBObject(r"\zeta(z)=\sum_{n=1}^\infty {1\over n^z}", color='important', aligned='left',
                                     text_size='large', name='sine')
        display.set_cursor_to_start_of_next_line(indent=0)
        display.set_cursor_to_start_of_next_line(indent=0)
        display.add_text(zeta_text)
        zeta_text.write(begin_time=27, transition_time=2)

        lambdas = [lambda z, i=i: zeta_sum(z, i) for i in range(1, 8)]
        # remark: if written with just i instead of i=i only the last value of i is used for all lambdas
        lambdas.append(lambda z: mp.zeta(z))
        print("Anzahl der Funktionen: ", len(lambdas))

        # functions

        #####################
        # coordinate system #
        #####################

        debug = 1  # put to 1
        coord = CoordinateSystem(dim=3, lengths=[20, 80, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-10, 10], [-40, 40], [0, 10]],
                                 all_n_tics=[int(10 / debug), int(40 / debug), int(5 / debug)],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-10, 10.1, 2 * debug),
                                                 np.arange(-40, 40.1, 2 * debug),
                                                 np.arange(0, 10.1, 2 * debug)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=0, transition_time=5)

        complex_function = ComplexPlaneWithSingularPoints(coord,
                                                          lambdas,
                                                          u=[1, 10], v=[-40, 40],
                                                          special_x=[-10, -8, -6, -4, -2, 0.5, 1],
                                                          special_y=[0],
                                                          detail=10,
                                                          resolution=100,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          smooth=2, name='zeta_sum')
        complex_function.appear(begin_time=5, transition_time=2)

        delay = 0
        for i in range(8):
            if i + 1 == 8:
                delay = 3
            complex_function.next_shape(begin_time=5 + delay + 3 * i, transition_time=2)

        #################
        # Camera motion #
        #################

        # only do this, once all objects are created, since the keyframe of the camera empty is disturbed by the
        # creation of other objects. No idea, why it is.

        empty = EmptyCube(location=[4, 0, 5], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        t0 = 45
        ibpy.camera_move([0, 20, 20], begin_time=t0, transition_time=5)
        display.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=5)

        t0 = 50
        f1.move([0, 0, -1], begin_time=t0, transition_time=5)

    def zeta_continuation(self):
        cues = self.sub_scenes['zeta_continuation']
        t0 = cues['start']

        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[4, -25, 5])
        display = DisplayOld(location=[13, 0, 7], scales=[4, 5, 4], line_spacing=1.5)
        display.appear(begin_time=0, transition_time=0)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{The }\zeta(z)\text{ continuation}", color='important',
                                 aligned='center', name='title', text_size='large')
        display.set_title(title)
        title.write(begin_time=1, transition_time=3)

        text_bits = [r"\left(1-{2\over 2^z}\right)",
                     r"\zeta(z)=\sum_{n=1}^\infty {1\over n^z}",
                     r"\left(1-{2\over 2^z}\right)"]
        text_bits2 = [r"=\sum_{n=1}^\infty \left({1\over n^z}-{2\over (2n)^z}\right)",
                      r"=\sum_{n=1}^\infty (-1)^{n+1}{1\over n^z}",
                      r"\zeta(z)=\left(1-2^{-z+1}\right)^{-1}\sum_{n=1}^\infty (-1)^{n+1}{1\over n^z}",
                      ]

        functional_equation = SimpleTexBObject(
            r"\pi^{-{z\over 2}}\Gamma\left({z\over 2}\right)\zeta(z)=\pi^{-{1-z\over 2}}\Gamma\left({1-z\over 2}\right)\zeta(1-z)",
            aligned='left', name='functional', color='example')

        f1 = TexBObject(*text_bits, aligned='left', name='lines1')
        display.add_text(f1, indent=0)

        lines = []

        for i, text in enumerate(text_bits2):
            if i == 2:
                lines.append(SimpleTexBObject(text, color='important', align='left',
                                              name='more_lines_' + str(i)))
            else:
                lines.append(SimpleTexBObject(text, align='left', name='more_lines_' + str(i), color='text'))

        t0 = 5
        animation_time = 2
        f1.write(expression_index=1, begin_time=t0, transition_time=animation_time)
        t0 += 3
        f1.write(expression_index=0, begin_time=t0, transition_time=animation_time)
        f1.write(expression_index=2, begin_time=t0, transition_time=animation_time)

        ref = f1.get_part(1)
        ref_chars = [4, 4, 0]
        for char, line in zip(ref_chars, lines):
            t0 += 3
            display.set_cursor_to_start_of_next_line(indent=0)
            display.add_text(line)
            line.align(ref, char_index=0, other_char_index=char)
            line.write(begin_time=t0, transition_time=animation_time)

        t0 += 3

        # #####################
        # # coordinate system #
        # #####################

        debug = 1  # put to 1
        coord = CoordinateSystem(dim=3, lengths=[20, 80, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-10, 10], [-40, 40], [0, 10]],
                                 all_n_tics=[int(10 / debug), int(40 / debug), int(5 / debug)],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-10, 10.1, 2 * debug),
                                                 np.arange(-40, 40.1, 2 * debug),
                                                 np.arange(0, 10.1, 2 * debug)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=0, transition_time=0)

        mp.pretty = True  # convert the clumpsy mp notation into standard python notation

        zeros = [0]
        zero = 0
        n = 1
        while zero < 40:
            zero = mp.zetazero(n).imag
            if zero < 40:
                zeros.append(zero)
                zeros.append(-zero)
            n += 1

        zeros.sort()

        complex_function = ComplexPlaneWithSingularPoints(coord,
                                                          lambda z: mp.zeta(z),
                                                          u=[1, 10], v=[-40, 40],
                                                          special_x=[-10, -8, -6, -4, -2, 0, 0.5, 1],
                                                          special_y=zeros,
                                                          detail=5,
                                                          resolution=50,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          smooth=2, name='zeta')

        complex_function.appear(begin_time=0, transition_time=0)

        #####################
        # Grow continuation #
        #####################
        # number of lines
        ny = int(500 / debug)
        t0 += 2
        yrange = coord.get_domain()[1]
        yspan = yrange[1] - yrange[0]
        dy = yspan / ny

        print("GPU render upto ", t0 * FRAME_RATE)
        remove = []
        for i in range(0, ny + 1):
            y = yrange[0] + dy * i

            end_domain = 1.01
            if y == 0:
                end_domain = 0.999

            line = Function(lambda x: [x, y, np.abs(mp.zeta(x + 1j * y))], coord, domain=[0, end_domain], num_points=20,
                            color='script', script='zeta_function', mode='PARAMETRIC', name='extension_x_' + str(i),
                            thickness=0.01)
            line.grow(begin_time=t0, transition_time=5, inverted=True)
            remove.append(line)

        nx = int(20 / debug)
        xrange = [0, 0.999]
        xspan = xrange[1] - xrange[0]
        dx = xspan / nx

        for i in range(0, nx + 1):
            x = xrange[0] + dx * i
            line = Function(lambda y: [x, y, np.abs(mp.zeta(x + 1j * y))], coord, domain=[-40, 40], num_points=1000,
                            color='script', script="zeta_function", mode='PARAMETRIC', name='extension_y' + str(i),
                            thickness=0.01)
            line.grow(begin_time=t0, transition_time=5)
            remove.append(line)

        t0 += 7
        display.set_cursor_to_start_of_next_line(indent=0)
        display.add_text(functional_equation, indent=0, scale=0.33)
        functional_equation.align(f1.get_part(0), char_index=0, other_char_index=0)
        functional_equation.write(begin_time=t0, transition_time=animation_time)

        t0 += animation_time + 3

        complex_function2 = ComplexPlaneWithSingularPoints(coord,
                                                           lambda z: mp.zeta(z),
                                                           u=[-10, 1], v=[-40, 40],
                                                           special_x=[-10, -8, -6, -4, -2, 0, 0.5, 1],
                                                           special_y=zeros,
                                                           detail=5,
                                                           resolution=50,
                                                           alpha=0.95,
                                                           metallic=0,
                                                           roughness=0.7,
                                                           emission_strength=0.1,
                                                           smooth=2, name='zeta_full')

        complex_function2.appear(begin_time=t0, transition_time=animation_time)

        # #################
        # # Camera motion #
        # #################
        #
        # only do this, once all objects are created, since the keyframe of the camera empty is disturbed by the
        # creation of other objects. No idea, why it is.

        empty = EmptyCube(location=[4, 0, 5], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        t0 = t0 + animation_time + 2
        # remove lines that one can switch back to GPU rendering
        print("GPU render after: ", t0 * FRAME_RATE + 1)
        for rem in remove:
            rem.disappear(begin_time=t0, transition_time=0)

        empty.move(direction=[-4, 0, -5], begin_time=t0, transition_time=5)
        ibpy.camera_move([-4, 25, 50], begin_time=t0, transition_time=5)
        display.move(direction=[0, 0, 0, ], begin_time=t0 - 1, transition_time=0)
        display.move(direction=[30, 0, 0], begin_time=t0, transition_time=0)
        coord.rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=t0, transition_time=5)

    def decomposition_poles(self):
        cues = self.sub_scenes['decomposition_poles']
        t0 = cues['start']

        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[4, -20, 5])
        display = DisplayOld(location=[11, 0, 5])
        display.appear(begin_time=0, transition_time=1)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{Functions with poles}", color='important',
                                 aligned='center', name='title')
        display.set_title(title)
        title.write(begin_time=1, transition_time=3)

        function = TexBObject("f(x)=1", "f(z)=1", aligned='left', name='function')
        display.add_text(function, custom_scales=[1, 0])

        #####################
        # coordinate system #
        #####################
        t0 = 0
        coord = CoordinateSystem(dim=3, lengths=[10, 10, 8], radii=[0.03, 0.03, 0.03],
                                 domains=[[-5, 5], [-5, 5], [0, 8]],
                                 all_n_tics=[5, 5, 4], location_of_origin=[0, 0, 2],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-5, 5.1, 2),
                                                 np.arange(-5, 5.1, 2),
                                                 np.arange(0, 8.1, 2)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=t0, transition_time=5)
        t0 += 6
        function.write(0, begin_time=t0, transition_time=1)
        t0 += 4

        factors = [TexBObject(r"\cdot {1\over \left (1-{x\over 2}\right)}", r"\cdot \left(1-{z\over2}\right)^{-1}",
                              aligned='left', name='factor1'),
                   SimpleTexBObject(r"\cdot \left(1-{z\over 2i}\right)^{-1}", aligned='left', name='factor2',
                                    color='text'),
                   SimpleTexBObject(r"\cdot \left(1+{z\over 1+i}\right)", aligned='left', name='factor2', color='text'),
                   ]
        for factor in factors:
            display.add_text(factor, custom_scales=[1, 0])

        factors[0].write(0, begin_time=t0, transition_time=1)

        real_part1 = Function(
            [lambda x: [x, 0, 3], lambda x: [x, 0, 2 + 1 / (1 - x / 2)], lambda x: [x, 0, 2 + np.abs(1 / (1 - x / 2))]],
            coordinate_system=coord,
            domain=[-5., 1.9999],
            num_points=50,
            singular_points=[2],
            color_mode='hue_color',
            mode='PARAMETRIC',
            hue_functions=["1,0,x,*,+", "y", "1,1,x,2,/,-,/", "y", "1,1,x,2,/,-,/", "y"],
            name='real_part')

        real_part2 = Function(
            [lambda x: [x, 0, 3], lambda x: [x, 0, 2 + 1 / (1 - x / 2)], lambda x: [x, 0, 2 + np.abs(1 / (1 - x / 2))]],
            coordinate_system=coord,
            domain=[2.0001, 5],
            num_points=50,
            singular_points=[2],
            color_mode='hue_color',
            mode='PARAMETRIC',
            hue_functions=["1,0,x,*,+", "y", "1,1,x,2,/,-,/", "y", "1,1,x,2,/,-,/", "y"],
            name='real_part')

        real_part2.grow(begin_time=t0 - 3, transition_time=2 * 0.7)
        real_part1.grow(begin_time=t0 - 3 + 2 * 0.7, transition_time=2 * 0.3)

        real_part2.next(begin_time=t0, transition_time=3)
        real_part1.next(begin_time=t0, transition_time=3)

        t0 += 6
        function.next(0, 1, begin_time=t0, transition_time=2)
        factors[0].next(0, 1, begin_time=t0, transition_time=2)

        t0 += 2
        real_part2.next(begin_time=t0, transition_time=3)
        real_part1.next(begin_time=t0, transition_time=3)

        t0 += 4
        start_camera_motion = t0
        lambdas = [lambda z: 1 / (1 - z / 2), lambda z: 1 / (1 - z / 2) / (1 - z / 2j),
                   lambda z: 1 / (1 - z / 2) / (1 - z / 2j) * (1 + z / (1 + 1j))]
        complex_function = ComplexPlaneWithSingularPoints(coord,
                                                          lambdas,
                                                          u=[-5, 5], v=[-5, 5],
                                                          special_x=[2, -1],
                                                          special_y=[0, 2, -1],
                                                          detail=3,
                                                          resolution=100,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          smooth=2, name='complex_function')
        complex_function.appear(begin_time=t0, transition_time=5)
        complex_function.next_shape(begin_time=t0 + 1, transition_time=4)

        t0 += 6
        real_part2.disappear(begin_time=t0, transition_time=1)
        real_part1.disappear(begin_time=t0, transition_time=1)

        factors[1].write(begin_time=t0, transition_time=1)
        complex_function.next_shape(begin_time=t0 + 1, transition_time=4)

        t0 += 6

        factors[2].write(begin_time=t0, transition_time=1)
        complex_function.next_shape(begin_time=t0 + 1, transition_time=4)

        t0 += 2
        start_camera_motion2 = t0 + 5
        # add a second and a third pole
        #################
        # Camera motion #
        #################

        # only do this, once all objects are created, since the keyframe of the camera empty is disturbed by the
        # creation of other objects. No idea, why it is.

        empty = EmptyCube(location=[4, 0, 5], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        ibpy.camera_move([0, 3, 16], begin_time=start_camera_motion, transition_time=5)
        display.rotate(rotation_euler=[np.pi / 4, 0, 0], begin_time=start_camera_motion, transition_time=5)

        ibpy.camera_move([0, -3, -16], begin_time=start_camera_motion2, transition_time=5)
        display.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=start_camera_motion2, transition_time=5)

        ibpy.camera_move([0, 0, -3], begin_time=start_camera_motion2 + 3, transition_time=4)
        empty.move([0, 0, -3], begin_time=start_camera_motion2 + 3, transition_time=4)
        display.move([0, 0, -3], begin_time=start_camera_motion2 + 3, transition_time=4)

        # display.disappear(begin_time=start_camera_motion2+7,transition_time=1)
        display.move([10, 0, 0], begin_time=start_camera_motion2 + 8, transition_time=1)
        empty.move([-4, 0, 0], begin_time=start_camera_motion2 + 8, transition_time=4)
        ibpy.camera_move([-4, 0, 0], begin_time=start_camera_motion2 + 8, transition_time=4)

        ibpy.camera_move([0, +8, 0], begin_time=start_camera_motion2 + 13, transition_time=4)

    def show_zeros(self):
        cues = self.sub_scenes['show_zeros']
        t0 = cues['start']

        ibpy.set_shadow(True)

        ###########
        # display #
        # #########
        ibpy.set_sun_light(location=[20, -5, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[0, 0, 55])
        display = DisplayOld(rotation_euler=[0, 0, np.pi / 2], location=[12.25, -5, 40], scales=[4, 6, 4], flat=True,
                             line_spacing=0.6, columns=3)
        display.appear(begin_time=0, transition_time=0)
        setup_time = 2

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{Special points of }\zeta(z)", color='important', aligned='center',
                                 name='title')
        display.set_title(title)
        t0 = setup_time + 1
        title.write(begin_time=t0, transition_time=3)

        lines = [r"\text{Poles:}",
                 "z=1",
                 r"\text{Zeros:}",
                 r"\text{trivial:}",
                 r"z=-2",
                 "z=-4",
                 "z=-6",
                 "z=-8",
                 r'\cdots',
                 r"\text{More zeros:}",
                 r"\text{non-trivial:}",
                 r"z\approx{\tfrac{1}{2}\pm 14.13i}",
                 r"z\approx\tfrac{1}{2}\pm 21.03i",
                 r"z\approx\tfrac{1}{2}\pm 25.01i",
                 r"z\approx\tfrac{1}{2}\pm 30.42i",
                 r"z\approx\tfrac{1}{2}\pm 32.94i",
                 r"z\approx\tfrac{1}{2}\pm 37.59i",
                 r"\cdots"
                 ]
        indents = [0, 0.5, 0, 0.5, 1, 1, 1, 1, 1, 0, 0.5, 1, 1, 1, 1, 1, 1, 1]
        colors = ['text', 'text', 'white', 'example', 'example', 'example', 'example', 'example', 'example', 'white',
                  'important', 'important', 'important', 'important', 'important', 'important', 'important',
                  'important', 'important', 'important']

        text = []

        arrow_colors = ['text', 'example', 'example', 'example', 'example', 'important', 'important',
                        'important', 'important', 'important', 'important', 'important', 'important', 'important',
                        'important', 'important', 'important', 'important', 'important', 'important', 'important',
                        'important', 'important']
        arrow_times = []
        for i in range(len(lines)):
            if i < 2:
                col = 1
            elif i < 9:
                col = 2
            else:
                col = 3

            display.set_cursor_to_start_of_next_line(indent=indents[i], column=col)
            text.append(SimpleTexBObject(lines[i], color=colors[i], name='line_' + str(i), aligned='left'))
            display.add_text(text[-1], column=col)
            t0 += 3
            text[-1].write(begin_time=t0, transition_time=2)

            if i not in {0, 2, 3, 8, 9, 10, 17}:
                arrow_times.append(t0)
            if i > 10 and i < 17:
                arrow_times.append(t0)  # add twice for the complex conjugate

        # #####################
        # # coordinate system #
        # #####################

        debug = 1  # put to 1
        coord = CoordinateSystem(dim=3, lengths=[20, 80, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-10, 10], [-40, 40], [0, 10]],
                                 all_n_tics=[int(10 / debug), int(40 / debug), int(5 / debug)],
                                 rotation_euler=[0, 0, 0],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-10, 10.1, 2 * debug),
                                                 np.arange(-40, 40.1, 2 * debug),
                                                 np.arange(0, 10.1, 2 * debug)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large',
                                 tip_lengths=[1, 1, 1],
                                 )
        coord.appear(begin_time=0, transition_time=0)

        mp.pretty = True  # convert the clumpsy mp notation into standard python notation

        zeros = [0]
        pos_zeros = []
        zero = 0
        n = 1
        while zero < 40:
            zero = mp.zetazero(n).imag
            if zero < 40:
                pos_zeros.append(zero)
                zeros.append(zero)
                zeros.append(-zero)
            n += 1

        zeros.sort()
        print(zeros)

        complex_function = ComplexPlaneWithSingularPoints(coord,
                                                          lambda z: mp.zeta(z),
                                                          u=[-10, 10], v=[-40, 40],
                                                          special_x=[-10, -8, -6, -4, -2, 0, 0.5, 1],
                                                          special_y=zeros,
                                                          detail=int(5 / debug),
                                                          resolution=50 / debug,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          smooth=2, name='zeta_full')

        complex_function.appear(begin_time=0, transition_time=0)

        # arrows
        i = 0
        length = 4
        radius = 0.3
        offset = Vector([length / 1.41, length / 1.41, 0])
        for color, time in zip(arrow_colors, arrow_times):

            if i == 0:
                arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                              location=coord.coords2location([1, 0, 7.5]), color=color, name='arrow_' + str(i))
            elif i in {1, 2, 3, 4}:
                arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                              location=coord.coords2location([-2 * (i), 0, 1]),
                              color=color, name='arrow_' + str(i))
            else:
                if i % 2 == 0:
                    arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                                  location=coord.coords2location([0.5, pos_zeros[int((i - 5) / 2)], 1]),
                                  color=color, name='arrow_' + str(i))
                else:
                    arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                                  location=coord.coords2location([0.5, -pos_zeros[int((i - 5) / 2)], 1]),
                                  color=color, name='arrow_' + str(i))

            if i in {0, 1, 2, 3, 4}:
                arrow.rotate_by(rotation_euler=[0, 0, 3 * np.pi / 4])
                arrow.rotate_by(rotation_euler=[-np.pi / 4, 0, 0])
            else:
                arrow.rotate_by(rotation_euler=[0, 0, 5 * np.pi / 4])
                arrow.rotate_by(rotation_euler=[np.pi / 4, 0, 0])

            arrow.grow(begin_time=time, transition_time=2, modus="from_top")

            i += 1

        #################
        # Camera motion #
        #################

        ibpy.set_camera_view_to(rotation_euler=[0, 0, np.pi / 2])
        # setup
        ibpy.camera_move([7.5, 0, 0], begin_time=0.5, transition_time=setup_time)

        ##################
        # remove display #
        ##################

        display.move(direction=[12, 0, 0], begin_time=t0 + 3, transition_time=2)

    def result(self):
        cues = self.sub_scenes['result']
        t0 = cues['start']

        ibpy.set_shadow(True)

        ###########
        # display #
        # #########
        setup_time = 2
        ibpy.set_sun_light(location=[20, -5, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[7.5, 0, 55])
        display = Display(rotation_euler=[np.pi / 4, 0, np.pi / 2], location=[9, 0, 35], scales=[20, 7, 4], flat=True,
                          number_of_lines=7, columns=1)

        t0 = 2 * setup_time + 1
        display.appear(begin_time=t0, transition_time=1)

        ###############
        #   text    ###
        ###############
        t0 += 1
        title = SimpleTexBObject(r"\text{The prime counting function }\psi(x)", color='important', aligned='center',
                                 name='title')
        display.set_title(title, scale=0.9)
        title.write(begin_time=t0, transition_time=3)
        display.set_cursor_to_start_of_next_line()

        # set color for each letter
        colors = [['drawing'] * 13, ['text'] * 2, ['example'] * 14, ['important']]
        colors_flat = [col for sublist in colors for col in sublist]

        psi = SimpleTexBObject(
            r"\psi(x)=-\log(2\pi)+x-\sum_{n=1}^{\infty}{x^{-2n}\over -2n}-\sum_{\rho}{x^\rho\over{\rho}}",
            color=colors_flat, aligned='left')
        display.add_text(psi, indent=3)

        # set color for each letter
        colors = [['white'] * 17, ['text'] * 7, ['example'] * 10, ['important']]
        colors_flat = [col for sublist in colors for col in sublist]

        final_line = SimpleTexBObject(
            r"\text{contribution from:\hspace{0.25em} the pole \hspace{0.25em} triv. zeros \hspace{0.25em} non-triv. zeros}",
            color=colors_flat, aligned='left')
        display.add_text_in(final_line, line=5, indent=0.5)

        t0 += 4
        psi.write(begin_time=t0, transition_time=5)

        arrow_colors = ['text', 'example', 'example', 'example', 'example', 'important', 'important',
                        'important', 'important', 'important', 'important', 'important', 'important', 'important',
                        'important', 'important', 'important', 'important', 'important', 'important', 'important',
                        'important', 'important']

        # #####################
        # # coordinate system #
        # #####################

        debug = 1  # put to 1
        coord = CoordinateSystem(dim=3, lengths=[20, 80, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-10, 10], [-40, 40], [0, 10]],
                                 all_n_tics=[int(10 / debug), int(40 / debug), int(5 / debug)],
                                 rotation_euler=[0, 0, 0],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-10, 10.1, 2 * debug),
                                                 np.arange(-40, 40.1, 2 * debug),
                                                 np.arange(0, 10.1, 2 * debug)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large',
                                 tip_lengths=[1, 1, 1],
                                 )
        coord.appear(begin_time=0, transition_time=0)

        mp.pretty = True  # convert the clumpsy mp notation into standard python notation
        zeros = [0]
        pos_zeros = []
        zero = 0
        n = 1
        while zero < 40:
            zero = mp.zetazero(n).imag
            if zero < 40:
                pos_zeros.append(zero)
                zeros.append(zero)
                zeros.append(-zero)
            n += 1

        zeros.sort()

        complex_function = ComplexPlaneWithSingularPoints(coord,
                                                          lambda z: mp.zeta(z),
                                                          u=[-10, 10], v=[-40, 40],
                                                          special_x=[-10, -8, -6, -4, -2, 0, 0.5, 1],
                                                          special_y=zeros,
                                                          detail=int(5 / debug),
                                                          resolution=50 / debug,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          smooth=2, name='zeta_full')

        complex_function.appear(begin_time=0, transition_time=0)
        arrows = []
        # arrows
        length = 4
        radius = 0.3
        offset = Vector([length / 1.41, length / 1.41, 0])
        for i in range(17):
            if i == 0:
                arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                              location=coord.coords2location([1, 0, 7.5]), color=arrow_colors[i],
                              name='arrow_' + str(i))
                arrows.append(arrow)
            elif i in {1, 2, 3, 4}:
                arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                              location=coord.coords2location([-2 * (i), 0, 1]),
                              color=arrow_colors[i], name='arrow_' + str(i))
                arrows.append(arrow)
            else:
                if i % 2 == 0:
                    arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                                  location=coord.coords2location([0.5, pos_zeros[int((i - 5) / 2)], 1]),
                                  color=arrow_colors[i], name='arrow_' + str(i))
                    arrows.append(arrow)
                else:
                    arrow = Arrow(length=length, radius=radius, rotation_euler=[0, np.pi / 2, 0],
                                  location=coord.coords2location([0.5, -pos_zeros[int((i - 5) / 2)], 1]),
                                  color=arrow_colors[i], name='arrow_' + str(i))
                    arrows.append(arrow)

            if i in {0, 1, 2, 3, 4}:
                arrow.rotate_by(rotation_euler=[0, 0, 3 * np.pi / 4])
                arrow.rotate_by(rotation_euler=[-np.pi / 4, 0, 0])
            else:
                arrow.rotate_by(rotation_euler=[0, 0, 5 * np.pi / 4])
                arrow.rotate_by(rotation_euler=[np.pi / 4, 0, 0])

            arrow.grow(begin_time=0, transition_time=0, modus="from_bottom")

        t0 += 6
        final_line.write(letter_range=[0, 17], begin_time=t0, transition_time=1)
        print("motion of pole arrow: ", t0)
        ###############
        # move arrows #
        ###############

        arrow = arrows[0]
        arrow.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0, transition_time=2)
        loc1 = psi.get_letter(13).get_world_location()
        loc2 = psi.get_letter(14).get_world_location()
        loc = 0.5 * (loc1 + loc2)
        # add shift manually
        loc.x += 1.75
        loc.z -= 1.75
        arrow.move_to(loc, begin_time=t0, transition_time=3)
        final_line.write(letter_range=[17, 24], begin_time=t0 + 3, transition_time=1)

        t0 += 5
        print("motion of trivial arrow: ", t0)
        for i in range(1, 5):
            arrow = arrows[i]
            arrow.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0, transition_time=2)
            loc = psi.get_letter(21).get_world_location()

            # add shift manually
            loc.x += 1.75
            loc.z -= 1.75
            arrow.move_to(loc, begin_time=t0 + (i - 1) / 5, transition_time=3)
        final_line.write(letter_range=[24, 34], begin_time=t0 + 2, transition_time=1)

        t0 += 5
        print("motion of non-trivial arrow: ", t0)
        for i in range(5, 17):
            arrow = arrows[i]
            arrow.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0, transition_time=2)
            loc = psi.get_letter(31).get_world_location()

            # add shift manually
            loc.x += 1.75
            loc.z -= 1.75
            arrow.move_to(loc, begin_time=t0 + (i - 5) / 12, transition_time=3)
        l = len(final_line.get_letters())
        final_line.write(letter_range=[34, l], begin_time=t0 + 3, transition_time=1)

        #################
        # Camera motion #
        #################
        empty = EmptyCube(location=[7.49, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        empty.move(direction=[-7.5, 0, 20], begin_time=setup_time, transition_time=setup_time)
        ibpy.set_camera_view_to(target=empty)

        ibpy.set_camera_view_to(rotation_euler=[0, 0, np.pi / 2])
        # setup
        ibpy.camera_move([29.5, 0, 2], begin_time=setup_time, transition_time=setup_time)

    def psi_contributions(self):

        cues = self.sub_scenes['psi_contributions']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        ######################
        # mathematical preps #
        ######################

        # mp.pretty = True  # convert the clumpsy mp notation into standard python notation
        pos_zeros = []
        # n = 0
        # while n < 100:
        #     zero = mp.zetazero(n + 1).imag
        #     pos_zeros.append(zero)
        #     n += 1
        # quickly calculated from mathematica
        # first 100 zeta zeros
        pos_zeros = [14.1347, 21.022, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703,
                     56.4462, 59.347, 60.8318,
                     65.1125, 67.0798, 69.5464, 72.0672, 75.7047, 77.1448, 79.3374, 82.9104, 84.7355, 87.4253, 88.8091,
                     92.4919, 94.6513, 95.8706,
                     98.8312, 101.318, 103.726, 105.447, 107.169, 111.03, 111.875, 114.32, 116.227, 118.791, 121.37,
                     122.947, 124.257, 127.517, 129.579,
                     131.088, 133.498, 134.757, 138.116, 139.736, 141.124, 143.112, 146.001, 147.423, 150.054, 150.925,
                     153.025, 156.113, 157.598,
                     158.85, 161.189, 163.031, 165.537, 167.184, 169.095, 169.912, 173.412, 174.754, 176.441, 178.377,
                     179.916, 182.207, 184.874,
                     185.599, 187.229, 189.416, 192.027, 193.08, 195.265, 196.876, 198.015, 201.265, 202.494, 204.19,
                     205.395, 207.906, 209.577,
                     211.691, 213.348, 214.547, 216.17, 219.068, 220.715, 221.431, 224.007, 224.983, 227.421, 229.337,
                     231.25, 231.987, 233.693, 236.524, 237.77, 239.555, 241.049, 242.823, 244.071, 247.137, 248.102,
                     249.574, 251.015, 253.07, 255.306, 256.381, 258.61, 259.874, 260.805,
                     263.574, 265.558, 266.615, 267.922, 269.97, 271.494, 273.46, 275.587,
                     276.452, 278.251, 279.229, 282.465, 283.211, 284.836, 286.667,
                     287.912, 289.58, 291.846, 293.558, 294.965, 295.573, 297.979, 299.84,
                     301.649, 302.697, 304.864, 305.729, 307.219, 310.109, 311.165,
                     312.428, 313.985, 315.476, 317.735, 318.853, 321.16, 322.145,
                     323.467, 324.863, 327.444, 329.033, 329.953, 331.474, 333.645,
                     334.211, 336.842, 338.34, 339.858, 341.042, 342.055, 344.662,
                     346.348, 347.273, 349.316, 350.408, 351.879, 353.489, 356.018,
                     357.151, 357.953, 359.744, 361.289, 363.331, 364.736, 366.213,
                     367.994, 368.968, 370.051, 373.062, 373.865, 375.826, 376.324,
                     378.437, 379.873, 381.484, 383.444, 384.956, 385.861, 387.223,
                     388.846, 391.456, 392.245, 393.428, 395.583, 396.382]

        # for i,zero in enumerate(pos_zeros):
        #     print(i," ",zero," ",float(zero))

        self.primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101,
            # 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
            # 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
            # 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
            # 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
            # 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
        }

        def psi(x):
            if x <= 1:
                return 0
            s = 0
            for p in self.primes:
                if p <= x:
                    s += np.log(p)
                else:
                    break
            for power in range(2, int(np.ceil((np.log2(x))) + 1)):
                for p in self.primes:
                    if p ** power <= x:
                        s += np.log(p)
                    else:
                        break
            return s

        def non_trivial(x, n):
            s = 0
            for i in range(n):
                r = float(pos_zeros[i])
                s += 1 / (0.25 + r * r) * (0.5 * np.cos(r * np.log(x)) + r * np.sin(r * np.log(x)))

            return -2 * np.sqrt(x) * s

        # for i in range(1, 101):
        #     print(i, " ", non_trivial(i, 100))

        ###########
        # display #
        # #########

        setup_time = 2
        ibpy.set_sun_light(location=[37, -10, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[37, 0, 57])

        empty = EmptyCube(location=[-0.01, 0, 20], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(rotation_euler=[np.pi / 4, 0, np.pi / 2], location=[9, 0, 35], scales=[20, 7, 4], flat=True,
                          number_of_lines=7, columns=1)

        display.appear(begin_time=t0, transition_time=0)

        ###############
        #   text    ###
        ###############
        title = SimpleTexBObject(r"\text{The prime counting function }\psi(x)", color='important', aligned='center',
                                 name='title')
        display.set_title(title, scale=0.9)
        title.write(begin_time=0, transition_time=0)
        display.set_cursor_to_start_of_next_line()

        # set color for each letter
        colors = [['drawing'] * 13, ['text'] * 2, ['example'] * 14, ['important']]
        colors_flat = [col for sublist in colors for col in sublist]

        psi_text = SimpleTexBObject(
            r"\psi(x)=-\log(2\pi)+x-\sum_{n=1}^{\infty}{x^{-2n}\over -2n}-\sum_{\rho}{x^\rho\over{\rho}}",
            color=colors_flat, aligned='left', name='psi_text')
        display.add_text(psi_text, indent=3)

        # set color for each letter
        colors = [['white'] * 17, ['text'] * 7, ['example'] * 10, ['important']]
        colors_flat = [col for sublist in colors for col in sublist]

        final_line = SimpleTexBObject(r"\text{contribution from:\hspace{0.25em} the pole \hspace{0.25em} triv. zeros "
                                      r"\hspace{0.25em} non-triv. zeros}", color=colors_flat, aligned='left',
                                      name='final_line')
        display.add_text_in(final_line, line=5, indent=0.5)

        psi_text.write(begin_time=0, transition_time=0)
        final_line.write(begin_time=0, transition_time=0)

        log_sub = SimpleTexBObject(r"\psi(x)=-\log(2\pi)+x+\log\tfrac{x}{\sqrt{x^2-1}}", color='example',
                                   aligned='left', name='log_repl')

        ###############
        # arrows #
        ###############

        arrow_colors = ['text', 'example', 'important']
        arrows = []

        locations = []
        loc1 = psi_text.get_letter(13).get_world_location()
        loc2 = psi_text.get_letter(14).get_world_location()
        loc = 0.5 * (loc1 + loc2)
        locations.append(loc)
        locations.append(psi_text.get_letter(21).get_world_location())
        locations.append(psi_text.get_letter(32).get_world_location())
        for loc in locations:
            loc.x += 1.75
            loc.z -= 1.75

        length = 4
        radius = 0.3
        for i, color in enumerate(arrow_colors):
            arrow = Arrow(length=length, radius=radius, rotation_euler=[0, -np.pi / 4, 0],
                          location=locations[i], color=color,
                          name='arrow_' + str(i))
            arrow.grow(begin_time=0, transition_time=0, modus='from_top')
            arrows.append(arrow)

        # #####################
        # # coordinate system #
        # #####################

        t0 += setup_time

        coord = CoordinateSystem(dim=2, lengths=[20, 20], radii=[0.03, 0.03],
                                 domains=[[0, 40], [0, 40]],
                                 all_n_tics=[8, 8],
                                 rotation_euler=[0, 0, np.pi / 2],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"x", r"\psi(x)"],
                                 all_tic_labels=[np.arange(0, 40.1, 5),
                                                 np.arange(0, 40.1, 5), ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large',
                                 tip_lengths=[1, 1],
                                 )
        t0 += 2
        coord.appear(begin_time=t0, transition_time=5)

        #################
        # Camera motion #
        #################

        empty.move(direction=[0.01, 20, -10], begin_time=setup_time, transition_time=setup_time)

        ibpy.set_camera_view_to(rotation_euler=[0, 0, np.pi / 2])
        # setup
        ibpy.camera_move([0, 20, -47], begin_time=setup_time, transition_time=setup_time)
        display.rotate(rotation_euler=[np.pi / 2, 0, np.pi / 2], begin_time=setup_time, transition_time=setup_time)
        for arrow in arrows:
            arrow.rotate(rotation_euler=[0, 0, 0], begin_time=setup_time, transition_time=setup_time)
            arrow.move([-20, 35, -30], begin_time=setup_time, transition_time=setup_time)

        display.move(direction=[-20, 35, -30], begin_time=setup_time, transition_time=setup_time)

        ############
        # psi func #
        ############

        functions = []

        psi_plot = PieceWiseFunction(lambda x: psi(x), coord, domain=[1, 40], numpoints=100,
                                     singular_points=[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32,
                                                      37, 39, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71,
                                                      73, 79], color='white', name="func_psi", shadow=False)

        t0 += 6
        psi_plot.grow(begin_time=t0, transition_time=10)

        log_plot = Function(lambda x: -np.log(2 * np.pi), coord, domain=[1, 40],
                            num_points=10,
                            color='drawing', name='func_log', shadow=False)
        t0 += 12
        log_plot.grow(begin_time=t0, transition_time=3)
        functions.append(log_plot)

        x_plot = Function([lambda x: x, lambda x: x - np.log(2 * np.pi)], coord, domain=[1, 40],
                          num_points=10,
                          color='text', name='func_x', shadow=False)
        t0 += 5
        x_plot.grow(begin_time=t0, transition_time=3)
        functions.append(x_plot)

        trivial_plot = Function(lambda x: -0.5 * np.log(1 - 1 / x ** 2), coord, domain=[1.0001, 40], num_points=1000,
                                color='example', name='func_trivial', shadow=False)
        t0 += 5
        trivial_plot.grow(begin_time=t0, transition_time=3)
        psi_text.replace(log_sub, src_letter_range=[15, 29], img_letter_range=[15, 27], shift=[0, 0.0], begin_time=t0,
                         transition_time=3)
        functions.append(trivial_plot)

        lambdas = []
        for i in range(1, 101):
            lambdas.append(lambda x, i=i: non_trivial(x, i))
        non_trivial_plot = Function(lambdas, coord, domain=[1, 40],
                                    num_points=2000, color='important', name='funct_non_trivial', shadow=False)
        functions.append(non_trivial_plot)
        t0 += 5
        non_trivial_plot.grow(begin_time=t0, transition_time=10)
        t0 += 11
        for i in range(2, 101):
            delta = 0.05
            non_trivial_plot.next(begin_time=t0 + i * delta, transition_time=delta)

        riemann_bound = Function(lambda x: np.sqrt(x), coord, color='white', domain=[1, 40], name='bound1',
                                 shadow=False)
        riemann_bound2 = Function(lambda x: -np.sqrt(x), coord, color='white', domain=[1, 40], name='bound1',
                                  shadow=False)

        t0 += 12
        riemann_bound.grow(begin_time=t0, transition_time=1)
        riemann_bound2.grow(begin_time=t0, transition_time=1)
        functions.append(riemann_bound)
        functions.append(riemann_bound2)

        t0 += 2
        for fcn in functions:
            fcn.disappear(begin_time=t0, transition_time=1)

    def psi_contributions_cumulative(self):

        cues = self.sub_scenes['psi_contributions_cumulative']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        ######################
        # mathematical preps #
        ######################

        pos_zeros = [14.1347, 21.022, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703,
                     56.4462, 59.347, 60.8318,
                     65.1125, 67.0798, 69.5464, 72.0672, 75.7047, 77.1448, 79.3374, 82.9104, 84.7355, 87.4253, 88.8091,
                     92.4919, 94.6513, 95.8706,
                     98.8312, 101.318, 103.726, 105.447, 107.169, 111.03, 111.875, 114.32, 116.227, 118.791, 121.37,
                     122.947, 124.257, 127.517, 129.579,
                     131.088, 133.498, 134.757, 138.116, 139.736, 141.124, 143.112, 146.001, 147.423, 150.054, 150.925,
                     153.025, 156.113, 157.598,
                     158.85, 161.189, 163.031, 165.537, 167.184, 169.095, 169.912, 173.412, 174.754, 176.441, 178.377,
                     179.916, 182.207, 184.874,
                     185.599, 187.229, 189.416, 192.027, 193.08, 195.265, 196.876, 198.015, 201.265, 202.494, 204.19,
                     205.395, 207.906, 209.577,
                     211.691, 213.348, 214.547, 216.17, 219.068, 220.715, 221.431, 224.007, 224.983, 227.421, 229.337,
                     231.25, 231.987, 233.693, 236.524, 237.77, 239.555, 241.049, 242.823, 244.071, 247.137, 248.102,
                     249.574, 251.015, 253.07, 255.306, 256.381, 258.61, 259.874, 260.805,
                     263.574, 265.558, 266.615, 267.922, 269.97, 271.494, 273.46, 275.587,
                     276.452, 278.251, 279.229, 282.465, 283.211, 284.836, 286.667,
                     287.912, 289.58, 291.846, 293.558, 294.965, 295.573, 297.979, 299.84,
                     301.649, 302.697, 304.864, 305.729, 307.219, 310.109, 311.165,
                     312.428, 313.985, 315.476, 317.735, 318.853, 321.16, 322.145,
                     323.467, 324.863, 327.444, 329.033, 329.953, 331.474, 333.645,
                     334.211, 336.842, 338.34, 339.858, 341.042, 342.055, 344.662,
                     346.348, 347.273, 349.316, 350.408, 351.879, 353.489, 356.018,
                     357.151, 357.953, 359.744, 361.289, 363.331, 364.736, 366.213,
                     367.994, 368.968, 370.051, 373.062, 373.865, 375.826, 376.324,
                     378.437, 379.873, 381.484, 383.444, 384.956, 385.861, 387.223,
                     388.846, 391.456, 392.245, 393.428, 395.583, 396.382]

        self.primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101,
        }

        def psi(x):
            if x <= 1:
                return 0
            s = 0
            for p in self.primes:
                if p <= x:
                    s += np.log(p)
                else:
                    break
            for power in range(2, int(np.ceil((np.log2(x))) + 1)):
                for p in self.primes:
                    if p ** power <= x:
                        s += np.log(p)
                    else:
                        break
            return s

        def non_trivial(x, n):
            s = 0
            for i in range(n):
                r = float(pos_zeros[i])
                s += 1 / (0.25 + r * r) * (0.5 * np.cos(r * np.log(x)) + r * np.sin(r * np.log(x)))

            return -2 * np.sqrt(x) * s

        ###########
        # display #
        # #########

        setup_time = 2
        ibpy.set_sun_light(location=[37, -10, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[37, 20, 10])

        empty = EmptyCube(location=[0, 20, 10], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(rotation_euler=[np.pi / 2, 0, np.pi / 2], location=[-11, 35, 5], scales=[20, 7, 4], flat=True,
                          number_of_lines=7, columns=1)

        display.appear(begin_time=0, transition_time=0)

        ###############
        #   text    ###
        ###############
        title = SimpleTexBObject(r"\text{The prime counting function }\psi(x)", color='important', aligned='center',
                                 name='title')
        display.set_title(title, scale=0.9)
        title.write(begin_time=0, transition_time=0)
        display.set_cursor_to_start_of_next_line()

        # set color for each letter
        colors = [['drawing'] * 13, ['text'] * 2, ['example'] * 11, ['important']]
        colors_flat = [col for sublist in colors for col in sublist]

        psi_text = SimpleTexBObject(
            r"\psi(x)=-\log(2\pi)+x+\log\tfrac{x}{x^2-1}-\sum_{\rho}{x^\rho\over{\rho}}",
            color=colors_flat, aligned='left', name='psi_text')
        display.add_text(psi_text, indent=3)

        # set color for each letter
        colors = [['white'] * 17, ['text'] * 7, ['example'] * 10, ['important']]
        colors_flat = [col for sublist in colors for col in sublist]

        final_line = SimpleTexBObject(r"\text{contribution from:\hspace{0.25em} the pole \hspace{0.25em} triv. zeros "
                                      r"\hspace{0.25em} non-triv. zeros}", color=colors_flat, aligned='left',
                                      name='final_line')
        display.add_text_in(final_line, line=5, indent=0.5)

        psi_text.write(begin_time=0, transition_time=0)
        final_line.write(begin_time=0, transition_time=0)

        ###############
        # arrows #
        ###############

        arrow_colors = ['text', 'example', 'important']
        arrows = []

        locations = []
        loc1 = psi_text.get_letter(13).get_world_location()
        loc2 = psi_text.get_letter(14).get_world_location()
        loc = 0.5 * (loc1 + loc2)
        locations.append(loc)
        locations.append(psi_text.get_letter(21).get_world_location())
        locations.append(psi_text.get_letter(29).get_world_location())
        for loc in locations:
            loc.x += 1.75
            loc.x -= 1.56
            loc.z -= 1.75
            loc.z -= 0.7878

        length = 4
        radius = 0.3
        for i, color in enumerate(arrow_colors):
            arrow = Arrow(length=length, radius=radius, rotation_euler=[0, 0, 0],
                          location=locations[i], color=color,
                          name='arrow_' + str(i))
            arrow.grow(begin_time=0, transition_time=0, modus='from_top')
            arrows.append(arrow)

        # #####################
        # # coordinate system #
        # #####################

        t0 += setup_time

        coord = CoordinateSystem(dim=2, lengths=[20, 20], radii=[0.03, 0.03],
                                 domains=[[0, 40], [0, 40]],
                                 all_n_tics=[8, 8],
                                 rotation_euler=[0, 0, np.pi / 2],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"x", r"\psi(x)"],
                                 all_tic_labels=[np.arange(0, 40.1, 5),
                                                 np.arange(0, 40.1, 5), ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large',
                                 tip_lengths=[1, 1],
                                 )
        coord.appear(begin_time=0, transition_time=0)

        #################
        # Camera motion #
        #################

        ############
        # psi func #
        ############

        functions = []

        psi_plot = PieceWiseFunction(lambda x: psi(x), coord, domain=[1, 40], numpoints=100,
                                     singular_points=[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32,
                                                      37, 39, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71,
                                                      73, 79], color='white', name="func_psi", shadow=False)
        psi_plot.grow(begin_time=0, transition_time=0)

        t0 += 2

        lambdas = [lambda x: x,
                   lambda x: x - np.log(2 * np.pi),
                   lambda x: x - np.log(2 * np.pi) + np.log(x / np.sqrt(x ** 2 - 1)),
                   ]

        for i in range(0, 100, 5):
            lambdas.append(lambda x, i=i: x - np.log(2 * np.pi) + np.log(x / np.sqrt(x * x - 1)) + non_trivial(x, i))

        colors = [
            ['text'],
            ['text', 'drawing'],
            ['text', 'drawing', 'example'],
            ['text', 'drawing', 'example', 'important'],
        ]

        psi_composition = Function(lambdas, coord, domain=[1.01, 40], num_points=1000, colors=colors,
                                   color_mode='voronoi',
                                   name='func_psi_composition', shadow=False)

        psi_composition.grow(begin_time=t0, transition_time=3)
        t0 += 4
        psi_composition.next(begin_time=t0, transition_time=1)
        t0 += 2
        psi_composition.next(begin_time=t0, transition_time=1)
        t0 += 2
        delta = 10 / (len(lambdas) - 3)
        for i in range(2, len(lambdas) - 1):
            psi_composition.next(begin_time=t0 + (i - 2) * delta, transition_time=delta)

    def prime_counting_functions(self):

        cues = self.sub_scenes['prime_counting_functions']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        ######################
        # mathematical preps #
        ######################

        self.primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101,
            # 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
            # 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
            # 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
            # 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
            # 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
        }

        def pi(x):
            if x <= 1:
                return 0
            s = 0
            for p in self.primes:
                if p <= x:
                    s += 1
                else:
                    break
            return s

        def riemann(x):
            if x <= 1:
                return 0
            s = 0
            for power in range(1, int(np.ceil((np.log2(x))) + 1)):
                for p in self.primes:
                    if p ** power <= x:
                        s += 1 / power
                    else:
                        break
            return s

        def psi(x):
            if x <= 1:
                return 0
            s = 0
            for power in range(1, int(np.ceil((np.log2(x))) + 1)):
                for p in self.primes:
                    if p ** power <= x:
                        s += np.log(p)
                    else:
                        break
            return s

        ###########
        # display #
        # #########

        ibpy.set_sun_light(location=[10, -37, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[20, -37, 10])

        empty = EmptyCube(location=[20, 0, 10], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(flat=False,
                          location=[37, 5, 18], scales=[6, 5, 4], number_of_lines=8, columns=1)

        display.appear(begin_time=0, transition_time=2)

        ###############
        #   text    ###
        ###############
        t0 += 3
        title = SimpleTexBObject(r"\text{Prime counting functions }", color='important', aligned='center',
                                 name='title')
        display.set_title(title, scale=0.9)
        title.write(begin_time=t0, transition_time=1)
        display.set_cursor_to_start_of_next_line()

        pi_function = SimpleTexBObject(r"\pi(x)=\sum_{p\le x} 1", color='text', aligned='left', name="pi_text")
        display.add_text(pi_function, indent=1)

        riemann_function = SimpleTexBObject(r"\Pi(x)=\sum_{n}\sum_{p^n\le x}{1\over n}", color='example',
                                            aligned='left',
                                            name="Pi_text")
        display.set_cursor_to_start_of_next_line()
        display.set_cursor_to_start_of_next_line()
        display.add_text(riemann_function, indent=1)

        psi_function = SimpleTexBObject(r"\psi(x)=\sum_{n}\sum_{p^n\le x}\log p", color='important',
                                        aligned='left',
                                        name="psi_text")
        display.set_cursor_to_start_of_next_line()
        display.set_cursor_to_start_of_next_line()
        display.add_text(psi_function, indent=1)
        # #####################
        # # coordinate system #
        # #####################

        coord = CoordinateSystem(dim=2, lengths=[40, 20], radii=[0.03, 0.03],
                                 domains=[[0, 80], [0, 40]],
                                 all_n_tics=[16, 8],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"x", r"f(x)"],
                                 all_tic_labels=[np.arange(0, 80.1, 5),
                                                 np.arange(0, 40.1, 5), ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large',
                                 tip_lengths=[1, 1],
                                 )
        coord.appear(begin_time=0, transition_time=2)

        prime_texts = []
        for p in self.primes:
            if p < 80:
                prime_texts.append(SimpleTexBObject(str(p), color='text', aligned='center', name="prime_" + str(p),
                                                    location=coord.coords2location([p, -2, 0]), scale=2))

        #################
        # Camera motion #
        #################

        #############
        # functions #
        #############

        t0 += 2
        pi_function.write(begin_time=t0, transition_time=1)

        pi_duration = 10
        pi_plot = PieceWiseFunction(lambda x: pi(x), coord, domain=[1, 80], numpoints=100,
                                    singular_points=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                                                     37, 39, 41, 43, 47, 53, 59, 61, 67, 71,
                                                     73, 79], color='text', name="func_pi", shadow=False)
        pi_plot.grow(begin_time=t0, transition_time=pi_duration)

        delta = pi_duration / len(prime_texts)
        for i, txt in enumerate(prime_texts):
            txt.write(begin_time=t0 + i * delta, transition_time=delta)

        t0 += pi_duration + 1
        riemann_function.write(begin_time=t0, transition_time=1)
        riemann_plot = PieceWiseFunction(lambda x: riemann(x), coord, domain=[1, 80], numpoints=100,
                                         singular_points=[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31,
                                                          32,
                                                          37, 39, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71,
                                                          73, 79], color='example', name="func_Pi", shadow=False)
        riemann_plot.grow(begin_time=t0, transition_time=pi_duration)

        prime_powers = [SimpleTexBObject(r"2^2", color='example', aligned='center', name="prime_power_" + str(4),
                                         location=coord.coords2location([4, -4, 0]), scale=2),
                        SimpleTexBObject(r"2^3", color='example', aligned='center', name="prime_power_" + str(8),
                                         location=coord.coords2location([8, -4, 0]), scale=2),
                        SimpleTexBObject(r"3^2", color='example', aligned='center', name="prime_power_" + str(9),
                                         location=coord.coords2location([9, -4, 0]), scale=2),
                        SimpleTexBObject(r"2^4", color='example', aligned='center', name="prime_power_" + str(16),
                                         location=coord.coords2location([16, -4, 0]), scale=2),
                        SimpleTexBObject(r"5^2", color='example', aligned='center', name="prime_power_" + str(25),
                                         location=coord.coords2location([25, -4, 0]), scale=2),
                        SimpleTexBObject(r"3^3", color='example', aligned='center', name="prime_power_" + str(27),
                                         location=coord.coords2location([27, -4, 0]), scale=2),
                        SimpleTexBObject(r"2^5", color='example', aligned='center', name="prime_power_" + str(32),
                                         location=coord.coords2location([32, -4, 0]), scale=2),
                        SimpleTexBObject(r"7^2", color='example', aligned='center', name="prime_power_" + str(49),
                                         location=coord.coords2location([49, -4, 0]), scale=2),
                        SimpleTexBObject(r"2^6", color='example', aligned='center', name="prime_power_" + str(64),
                                         location=coord.coords2location([64, -4, 0]), scale=2)]

        delta = pi_duration / len(prime_powers)
        for i, txt in enumerate(prime_powers):
            txt.write(begin_time=t0 + i * 0.7 * delta, transition_time=delta)

        t0 += pi_duration + 1
        psi_function.write(begin_time=t0, transition_time=1)
        psi_plot = PieceWiseFunction(lambda x: psi(x), coord, domain=[1, 80], numpoints=100,
                                     singular_points=[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32,
                                                      37, 39, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71,
                                                      73, 79], color='important', name="func_psi", shadow=False)
        psi_plot.grow(begin_time=t0, transition_time=pi_duration)

        ###############
        # explanation #
        ###############
        t0 += pi_duration + 1
        psi1 = psi(30.9)
        psi2 = psi(31.1)
        dist = Cylinder(start=coord.coords2location([31, psi1]), end=coord.coords2location([31, psi2]), color='white')
        dist.grow(begin_time=t0, transition_time=1, modus='from_center')

        log31 = SimpleTexBObject(r"\log 31", color='white', location=coord.coords2location([30, 0.5 * (psi1 + psi2)]),
                                 aligned='right', scale=2)
        log31.write(begin_time=t0, transition_time=1)

        t0 += 2
        psi1 = psi(31.9)
        psi2 = psi(32.1)
        dist = Cylinder(start=coord.coords2location([32, psi1]), end=coord.coords2location([32, psi2]), color='white')
        dist.grow(begin_time=t0, transition_time=1, modus='from_center')

        log2 = SimpleTexBObject(r"\log 2", color='white', location=coord.coords2location([31, 0.5 * (psi1 + psi2)]),
                                aligned='right', scale=2)
        log2.write(begin_time=t0, transition_time=1)

        t0 += 2

        display2 = Display(flat=False, name='Display2',
                           location=[30, 5, 3], scales=[12, 5, 4], number_of_lines=7, columns=1)

        display2.appear(begin_time=t0, transition_time=2)
        remark = SimpleTexBObject(r"{\rm e}^{\psi(x_+)-\psi(x_-)}=\left\{\text{\begin{tabular}{l l}$x$ & if $x$ "
                                  r"is prime\\\\$p=\log_nx$ & if $x$ is a prime power\\\\$1$ & otherwise\end{"
                                  r"tabular}}\right.", color='white', aligned='left')
        display2.add_text_in(remark, line=2, indent=3)
        remark.write(letter_range=[0, 28], begin_time=t0, transition_time=2)
        t0 += 3
        remark.write(letter_set=[30, 37, 40, 43, 46, 49, 52, 56, 59, 62, 65], begin_time=t0, transition_time=2)
        t0 += 3
        remark.write(
            letter_set=[28, 31, 32, 33, 34, 35, 36, 38, 39, 41, 44, 47, 50, 54, 57, 60, 63, 64, 66, 67, 68, 69, 70, 71],
            begin_time=t0, transition_time=2)
        t0 += 3
        remark.write(letter_set=[29, 39, 42, 45, 48, 51, 53, 55, 58, 61], begin_time=t0, transition_time=2)

    def overview(self):
        cues = self.sub_scenes['overview']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        locations = [[6.5, 5, 17.5], [33.5, 5, 17.5], [6.5, 5, 2.85], [33.5, 5, 2.85]]
        titles = [r"\text{I: Prime counting functions}", r"\text{II: Product representations}",
                  r"\text{III: Riemann's } "
                  r"\,\,\zeta\text{--function}",
                  r"\text{IV: Fill in the gaps}"]
        ibpy.set_sun_light(location=[10, -37, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[20, -40, 10])

        empty = EmptyCube(location=[20, 0, 10], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        t0 = 1
        for i in range(4):
            display = Display(flat=True,
                              location=locations[i], scales=[13.3, 7, 13.3], number_of_lines=8, columns=1,
                              name="Display_" + str(i))

            display.appear(begin_time=t0, transition_time=2)
            title = SimpleTexBObject(titles[i], color='important')
            display.set_title(title, scale=0.5)
            title.write(begin_time=t0 + 3, transition_time=1)
            t0 += 5

    def zeta_decomposition(self):
        cues = self.sub_scenes['zeta_decomposition']
        t0 = 0
        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[20, -25, 5])
        display = Display(location=[5, 1, 9], scales=[15, 4, 4], number_of_lines=8, flat=True)
        display.appear(begin_time=1, transition_time=2)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{The }\zeta(z)\text{ product representation}", color='important',
                                 aligned='center', name='title', text_size='large')
        display.set_title(title, scale=0.7)
        title.write(begin_time=1, transition_time=3)
        colors = [['drawing'] * 4, ['text'] * 6,
                  ['drawing'], ['important'] * 8, ['drawing'], ['important'] * 9,
                  ['drawing'], ['important'] * 8, ['drawing'], ['important'] * 9,
                  ['drawing'], ['important'] * 8, ['drawing'], ['important'] * 9,
                  ['drawing'], ['important'] * 3, ['drawing'],
                  ]
        colors = [col for sublist in colors for col in sublist]

        colors2 = [
            ['drawing'], ['example'] * 7,
            ['drawing'], ['example'] * 7,
            ['drawing'], ['example'] * 7,
            ['drawing'], ['example'] * 7,
            ['drawing'], ['example'] * 3, ['drawing'],
        ]
        colors2 = [col for sublist in colors2 for col in sublist]

        zeta_text = SimpleTexBObject(r"\zeta(z)={1\over z-1}\cdot \left(1-\tfrac{z}{\rho_1}\right)"
                                     r"\cdot\left(1-\tfrac{z}{{\overline{\rho_1}}}\right)"
                                     r"\cdot\left(1-\tfrac{z}{{\rho_2}}\right)"
                                     r"\cdot\left(1-\tfrac{z}{{\overline{\rho_2}}}\right)"
                                     r"\cdot\left(1-\tfrac{z}{{\rho_3}}\right)"
                                     r"\cdot\left(1-\tfrac{z}{{\overline{\rho_3}}}\right)"
                                     r"\cdot\dots\cdot",
                                     color=colors, aligned='left')
        display.add_text_in(zeta_text, line=1, scale=0.7, indent=1)
        zeta_text2 = SimpleTexBObject(
            r"\cdot \left(1+\tfrac{s}{2}\right)"
            r"\cdot \left(1+\tfrac{s}{4}\right)"
            r"\cdot \left(1+\tfrac{s}{6}\right)"
            r"\cdot \left(1+\tfrac{s}{8}\right)"
            r"\cdot \dots \cdot",
            color=colors2, aligned='left')
        display.add_text_in(zeta_text2, line=3, scale=0.7, indent=3)

        zeta_text3 = SimpleTexBObject(
            r"\cdot \left(1+1\right)^{-\tfrac{z}{2}}"
            r"\cdot \left(1+\tfrac{1}{2}\right)^{-\tfrac{z}{2}}"
            r"\cdot \left(1+\tfrac{1}{3}\right)^{-\tfrac{z}{2}}"
            r"\cdot \left(1+\tfrac{1}{4}\right)^{-\tfrac{z}{2}}"
            r"\cdot \dots \cdot \tfrac{\pi^{\tfrac{z}{2}}}{2}",
            color='white', aligned='left')
        display.add_text_in(zeta_text3, line=5, scale=0.6, indent=3)

        zeros_text = SimpleTexBObject(r"\rho_1\approx\tfrac{1}{2}+14.1347i", color='white', aligned='left')
        zeros_text2 = SimpleTexBObject(r"\rho_2\approx\tfrac{1}{2}+21.0220i", color='white', aligned='left')
        zeros_text3 = SimpleTexBObject(r"\rho_3\approx\tfrac{1}{2}+25.0109i", color='white', aligned='left')
        display.add_text_in(zeros_text, line=0, scale=0.5, indent=5)
        display.add_text_in(zeros_text2, line=0, scale=0.5, indent=10.5)
        display.add_text_in(zeros_text3, line=0, scale=0.5, indent=16)

        # #####################
        # # coordinate system #
        # #####################

        debug = 1  # put to 1
        coord = CoordinateSystem(dim=3, lengths=[20, 80, 5], radii=[0.03, 0.03, 0.03],
                                 domains=[[-10, 10], [-40, 40], [0, 5]],
                                 all_n_tics=[int(10 / debug), int(20 / debug), int(5 / debug)],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                 all_tic_labels=[np.arange(-10, 10.1, 2 * debug),
                                                 np.arange(-40, 40.1, 4 * debug),
                                                 np.arange(0, 5.1, 1 * debug)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=1, transition_time=5)

        mp.pretty = True  # convert the clumpsy mp notation into standard python notation

        zeros = [0]
        zero = 0
        roots = []
        n = 1
        while zero < 250 / debug:
            root = mp.zetazero(n)
            roots.append(root)
            zero = root.imag
            if zero < 40:
                zeros.append(zero)
                zeros.append(-zero)
            n += 1

        zeros.sort()

        print(roots)

        polynomials = []
        pol = Polynomial([1])
        for i in range(int(11 * 5 / debug)):
            pol *= Polynomial([1, -1 / roots[i]])
            pol *= Polynomial([1, -1 / np.conj(roots[i])])
            polynomials.append(pol.copy())

        t0 += 4
        complex_function = ComplexPlaneWithSingularPoints(coord,
                                                          [
                                                              lambda z: 1 / (z - 1),
                                                              lambda z: polynomials[0].eval(z) / (z - 1),
                                                              lambda z: polynomials[1].eval(z) / (z - 1),
                                                              lambda z: polynomials[2].eval(z) / (z - 1),
                                                              lambda z: polynomials[-1].eval(z) / (
                                                                      z - 1),
                                                          ],
                                                          u=[-10, 10], v=[-40, 40],
                                                          special_x=[0.5, 1],
                                                          special_y=zeros,
                                                          detail=int(5 / debug),
                                                          resolution=50 / debug,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          smooth=2, name='zeta')

        complex_function.appear(begin_time=t0, transition_time=3)
        zeta_text.write(letter_range=[0, 10], begin_time=t0, transition_time=3)

        t0 += 5
        complex_function.next_shape(begin_time=t0, transition_time=3)
        zeta_text.write(letter_range=[10, 29], begin_time=t0, transition_time=3)
        zeros_text.write(begin_time=t0, transition_time=1)

        t0 += 5
        complex_function.next_shape(begin_time=t0, transition_time=3)
        zeta_text.write(letter_range=[29, 48], begin_time=t0, transition_time=3)
        zeros_text2.write(begin_time=t0, transition_time=1)

        t0 += 5
        complex_function.next_shape(begin_time=t0, transition_time=3)
        zeta_text.write(letter_range=[48, 67], begin_time=t0, transition_time=3)
        zeros_text3.write(begin_time=t0, transition_time=1)

        t0 += 5
        complex_function.next_shape(begin_time=t0, transition_time=3)
        zeta_text.write(letter_range=[67, 72], begin_time=t0, transition_time=3)

        polynomials2 = []
        pol = Polynomial([1])
        for i in range(1, int(12 * 5 / debug)):
            pol *= Polynomial([1, 1 / 2 / i])
            polynomials2.append(pol.copy())

        polynomials3 = []
        pol = Polynomial([1])
        for i in range(1, int(12 * 5 / debug)):
            pol *= Polynomial([1, 1 / i])
            polynomials3.append(pol.copy())

        complex_function2 = ComplexPlaneWithSingularPoints(coord,
                                                           [
                                                               lambda z: polynomials[-1].eval(z) / (z - 1),
                                                               lambda z: polynomials[-1].eval(z) * polynomials2[0].eval(
                                                                   z) * polynomials3[0].eval(1) ** (-z / 2) / (z - 1),
                                                               lambda z: polynomials[-1].eval(z) * polynomials2[1].eval(
                                                                   z) * polynomials3[1].eval(1) ** (-z / 2) / (z - 1),
                                                               lambda z: polynomials[-1].eval(z) * polynomials2[2].eval(
                                                                   z) * polynomials3[2].eval(1) ** (-z / 2) / (z - 1),
                                                               lambda z: polynomials[-1].eval(z) * polynomials2[3].eval(
                                                                   z) * polynomials3[3].eval(1) ** (-z / 2) / (z - 1),
                                                               lambda z: polynomials[-1].eval(z) * polynomials2[
                                                                   -1].eval(z) * polynomials3[-1].eval(1) ** (
                                                                                 -z / 2) * np.pi ** (z / 2) / 2 / (
                                                                                 z - 1),
                                                           ],
                                                           u=[-10, 10], v=[-40, 40],
                                                           special_x=[0.5, 1, -2, -4, -6, -8, -10],
                                                           special_y=zeros,
                                                           detail=int(5 / debug),
                                                           resolution=50 / debug,
                                                           alpha=0.95,
                                                           metallic=0,
                                                           roughness=0.7,
                                                           emission_strength=0.1,
                                                           smooth=2, name='zeta2')

        t0 += 5
        camera_turn_time = t0
        complex_function2.appear(begin_time=t0, transition_time=0.1)
        complex_function.disappear(begin_time=t0, transition_time=0.1)

        t0 += 4
        complex_function2.next_shape(begin_time=t0, transition_time=3)
        zeta_text2.write(letter_range=[0, 8], begin_time=t0, transition_time=3)
        zeta_text3.write(letter_range=[0, 10], begin_time=t0, transition_time=3)

        t0 += 4
        complex_function2.next_shape(begin_time=t0, transition_time=3)
        zeta_text2.write(letter_range=[8, 16], begin_time=t0, transition_time=3)
        zeta_text3.write(letter_range=[10, 22], begin_time=t0, transition_time=3)

        t0 += 4
        complex_function2.next_shape(begin_time=t0, transition_time=3)
        zeta_text2.write(letter_range=[16, 24], begin_time=t0, transition_time=3)
        zeta_text3.write(letter_range=[22, 34], begin_time=t0, transition_time=3)

        t0 += 4
        complex_function2.next_shape(begin_time=t0, transition_time=3)
        zeta_text2.write(letter_range=[24, 32], begin_time=t0, transition_time=3)
        zeta_text3.write(letter_range=[34, 46], begin_time=t0, transition_time=3)

        t0 += 4
        complex_function2.next_shape(begin_time=t0, transition_time=3)
        zeta_text2.write(letter_range=[32, 37], begin_time=t0, transition_time=3)
        zeta_text3.write(letter_range=[46, len(zeta_text3.get_letters())], begin_time=t0, transition_time=3)

        # #################
        # # Camera motion #
        # #################
        #
        # only do this, once all objects are created, since the keyframe of the camera empty is disturbed by the
        # creation of other objects. No idea, why it is.

        empty = EmptyCube(location=[5, 0, 5], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        empty.move([-5, 0, -5], begin_time=camera_turn_time, transition_time=5)
        ibpy.camera_move([20, 25, 20], begin_time=camera_turn_time, transition_time=5)
        display.rotate([np.pi / 4, 0, np.pi / 2], begin_time=camera_turn_time, transition_time=5)
        display.move_to([10.5, 0, 4.5], begin_time=camera_turn_time, transition_time=5)

    def algebra(self):
        cues = self.sub_scenes['overview']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        locations = [[6.5, 5, 12.775], [33.5, 5, 18.775]]

        ibpy.set_sun_light(location=[0, -37, 35])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[20, -40, 10])

        empty = EmptyCube(location=[20, 0, 10], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        t0 = 0
        displays = []
        for i in range(2):
            display = Display(flat=True,
                              location=locations[i], scales=[13.3, (2 - i) * 6, 13.3], number_of_lines=7 * (2 - i),
                              columns=1,
                              name="Display_" + str(i))

            display.appear(begin_time=t0, transition_time=2)
            displays.append(display)

        display_new, display_old = displays

        colors = [['drawing'] * 13, ['example'] * 4, ['drawing']]
        colors = [col for sublist in colors for col in sublist]

        colors2 = [['example'] * 4, ['drawing']]
        colors2 = [col for sublist in colors2 for col in sublist]

        lines_old = [
            SimpleTexBObject(r"\log\zeta(z)=-z\int\limits_0^\infty \Pi(x)x^{-z}\tfrac{{\rm d}x}{x}",
                             color=colors, aligned='left', name='old1'),
            SimpleTexBObject(r"\Pi(x)=-\tfrac{1}{2\pi i}\int\limits_{a-i\infty}^{a+i \infty} \log\zeta(z)x^{z}\tfrac{"
                             r"{\rm d}z}{z}", color=colors2, aligned='left', name='old2'),
        ]

        lines_old[1].align(lines_old[0], char_index=4, other_char_index=7)

        diagram_show_time = t0
        for i, line in enumerate(lines_old):
            display_old.add_text_in(line, line=3 * i + 1, indent=3)
            line.write(begin_time=t0, transition_time=2)
            t0 += 3

        colors = [['drawing'] * 16, ['important'] * 4, ['drawing']]
        colors = [col for sublist in colors for col in sublist]

        colors2 = [['important'] * 4, ['drawing']]
        colors2 = [col for sublist in colors2 for col in sublist]

        colors3 = [['drawing'] * 13, ['text'] * 5, ['drawing'] * 6, ['drawing', 'example'] * 3, ['drawing'] * 4,
                   ['example'] * 3,
                   ['drawing'], ['example'] * 2, ['drawing'] * 18, ['important']]
        colors3 = [col for sublist in colors3 for col in sublist]

        colors4 = [['drawing'] * 22, ['text'] * 6, ['drawing'] * 6, ['example'] * 13, ['drawing'] * 3, ['important']]
        colors4 = [col for sublist in colors4 for col in sublist]

        lines_new = [
            SimpleTexBObject(r"\frac{\zeta'(z)}{\zeta(z)}=-z\int\limits_0^\infty \psi(x)x^{-z}\tfrac{{\rm d}x}{x}",
                             color=colors, aligned='left', name='new1'),
            SimpleTexBObject(
                r"\psi(x)=-\tfrac{1}{2\pi i}\int\limits_{a-i\infty}^{a+i \infty} \tfrac{\zeta'(z)}{\zeta(z)}x^{"
                r"z}\tfrac{{\rm d}z}{z}",
                color=colors2, aligned='left', name='new2'),
            SimpleTexBObject(r"\zeta(z)=\tfrac{1}{2} \pi^{\tfrac{z}{2}}\cdot\frac{1}{z-1}\cdot\prod\limits_{"
                             r"n=1}^\infty\frac{\left(1+\tfrac{z}{2n}\right)}{\left(1+\tfrac{1}{n}\right)^{\tfrac{z}{"
                             r"2}}}\cdot\!\!\!\!\!\!\!\prod\limits_{\text{\begin{tabular}{r l}$\zeta("
                             r"\rho)\!\!\!\!\!\!\!\!$&$=0$\\ Re$\,\rho\!\!\!\!\!\!\!\!$&$=\tfrac{1}{2}$\end{"
                             r"tabular}}}\!\!\!\!\!\!\!\left(1-\tfrac{z}{\rho}\right)",
                             color=colors3, aligned='left', name='new3'),
            SimpleTexBObject(r"\frac{\zeta'(z)}{\zeta(z)}=\frac{\rm d}{{\rm d}z} \log\zeta(z)",
                             color='drawing', aligned='left', name='new4'),
            SimpleTexBObject(r"\frac{\zeta'(z)}{\zeta(z)}-\frac{\zeta'(0)}{\zeta(0)}=-\frac{z}{z-1}+\sum_{n=1}^\infty "
                             r"\frac{z}{(-2n)(z+2n)}+\sum_\rho \frac{ z}{\rho(z-\rho)}", color=colors4, aligned='left',
                             name='new5')
        ]

        lines_new[1].align(lines_new[0], char_index=4, other_char_index=10)
        # lines_new[1].align(lines_new[0], char_index=4, other_char_index=10)

        for i, line in enumerate(lines_new):
            if i > 1:
                scale = 0.5
            else:
                scale = 0.7
            if i > 1:
                offset = 2
                spc = 2
            else:
                offset = 0
                spc = 3
            display_new.add_text_in(line, line=spc * i + 1 + offset, indent=3, scale=scale)
            line.write(begin_time=t0, transition_time=2)
            t0 += 3

        ############################
        # prime counting functions #
        ############################

        self.primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101,
        }

        def pi(x):
            if x <= 1:
                return 0
            s = 0
            for p in self.primes:
                if p <= x:
                    s += 1
                else:
                    break
            return s

        def riemann(x):
            if x <= 1:
                return 0
            s = 0
            for power in range(1, int(np.ceil((np.log2(x))) + 1)):
                for p in self.primes:
                    if p ** power <= x:
                        s += 1 / power
                    else:
                        break
            return s

        def psi(x):
            if x <= 1:
                return 0
            s = 0
            for power in range(1, int(np.ceil((np.log2(x))) + 1)):
                for p in self.primes:
                    if p ** power <= x:
                        s += np.log(p)
                    else:
                        break
            return s

        # #####################
        # # coordinate system #
        # #####################

        coord = CoordinateSystem(dim=2, lengths=[20, 10], radii=[0.03, 0.03],
                                 domains=[[0, 40], [0, 40]],
                                 all_n_tics=[8, 8],
                                 location_of_origin=[21, 0, -1],
                                 labels=[r"x", r"f(x)"],
                                 all_tic_labels=[np.arange(0, 40.1, 5),
                                                 np.arange(0, 40.1, 5), ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large',
                                 tip_lengths=[1, 1],
                                 )
        coord.appear(begin_time=diagram_show_time, transition_time=2)

        prime_texts = []
        for p in self.primes:
            if p < 40:
                prime_texts.append(SimpleTexBObject(str(p), color='text', aligned='center', name="prime_" + str(p),
                                                    location=coord.coords2location([p, -4, 0]), scale=1.5))
        #############
        # functions #
        #############

        pi_duration = 1
        pi_plot = PieceWiseFunction(lambda x: pi(x), coord, domain=[1, 40], numpoints=100,
                                    singular_points=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                                                     37, 39], color='text', name="func_pi", shadow=False)
        pi_plot.grow(begin_time=diagram_show_time, transition_time=pi_duration)

        delta = pi_duration / len(prime_texts)
        for i, txt in enumerate(prime_texts):
            txt.write(begin_time=diagram_show_time + i * delta, transition_time=delta)

        diagram_show_time += pi_duration + 1
        riemann_plot = PieceWiseFunction(lambda x: riemann(x), coord, domain=[1, 40], numpoints=100,
                                         singular_points=[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31,
                                                          32,
                                                          37, 39], color='example', name="func_Pi", shadow=False)
        riemann_plot.grow(begin_time=diagram_show_time, transition_time=pi_duration)

        diagram_show_time += pi_duration + 1
        psi_plot = PieceWiseFunction(lambda x: psi(x), coord, domain=[1, 40], numpoints=100,
                                     singular_points=[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32,
                                                      37, 39], color='important', name="func_psi", shadow=False)
        psi_plot.grow(begin_time=diagram_show_time, transition_time=pi_duration)

        t0 += 2

        display_old.disappear(begin_time=t0, transition_time=1)
        coord.disappear(begin_time=t0, transition_time=1)

        lines_new[0].disappear(begin_time=t0, transition_time=1)
        lines_new[2].disappear(begin_time=t0, transition_time=1)
        lines_new[3].disappear(begin_time=t0, transition_time=1)

        for i, text in enumerate(prime_texts):
            text.disappear(begin_time=t0 + 0.1 * i, transition_time=1)

    def contour_integral(self):
        cues = self.sub_scenes['overview']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        locations = [6.5, 5, 12.775]

        ibpy.set_sun_light(location=[0, -37, 35])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[20, -40, 10])

        empty = EmptyCube(location=[20, 0, 10], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        t0 += 0

        display = Display(flat=True, location=[6.5, 5, 12.775], scales=[13.3, 12, 13.3], number_of_lines=14, columns=1,
                          name="Display")
        display.appear(begin_time=t0, transition_time=0)

        colors = [['drawing'] * 16, ['important'] * 4, ['drawing']]
        colors = [col for sublist in colors for col in sublist]

        colors2 = [['important'] * 4, ['drawing']]
        colors2 = [col for sublist in colors2 for col in sublist]

        colors3 = [['drawing'] * 13, ['text'] * 5, ['drawing'] * 6, ['drawing', 'example'] * 3, ['drawing'] * 4,
                   ['example'] * 3, ['drawing'], ['example'] * 2, ['drawing'] * 18, ['important']]
        colors3 = [col for sublist in colors3 for col in sublist]

        colors4 = [['drawing'] * 22, ['text'] * 6, ['drawing'] * 6, ['example'] * 13, ['drawing'] * 3, ['important']]
        colors4 = [col for sublist in colors4 for col in sublist]

        lines_new = [
            SimpleTexBObject(r"\frac{\zeta'(z)}{\zeta(z)}=-z\int\limits_0^\infty \psi(x)x^{-z}\tfrac{{\rm d}x}{x}",
                             color=colors, aligned='left'),
            SimpleTexBObject(
                r"\psi(x)=-\tfrac{1}{2\pi i}\int\limits_{a-i\infty}^{a+i \infty} \tfrac{\zeta'(z)}{\zeta(z)}x^{"
                r"z}\tfrac{{\rm d}z}{z}",
                color=colors2, aligned='left'),
            SimpleTexBObject(r"\zeta(z)=\tfrac{1}{2} \pi^{\tfrac{z}{2}}\cdot\frac{1}{z-1}\cdot\prod\limits_{"
                             r"n=1}^\infty\frac{\left(1+\tfrac{z}{2n}\right)}{\left(1+\tfrac{1}{n}\right)^{\tfrac{z}{"
                             r"2}}}\cdot\!\!\!\!\!\prod\limits_{\text{\begin{tabular}{r l}$\zeta("
                             r"\rho)\!\!\!\!\!$&$=0$\\ Re$\,\rho\!\!\!\!\!\!\!\!\!\!$&$=\tfrac{1}{2}$\end{"
                             r"tabular}}}\!\!\!\!\!\!\!\!\!\!\left(1-\tfrac{z}{\rho}\right)",
                             color=colors3, aligned='left'),
            SimpleTexBObject(r"\frac{\zeta'(z)}{\zeta(z)}=\frac{\rm d}{{\rm d}z} \log\zeta(z)",
                             color='drawing', aligned='left'),
            SimpleTexBObject(r"\frac{\zeta'(z)}{\zeta(z)}-\frac{\zeta'(0)}{\zeta(0)}=-\frac{z}{z-1}+\sum_{n=1}^\infty "
                             r"\frac{z}{(-2n)(z+2n)}+\sum_\rho \frac{ z}{\rho(z-\rho)}", color=colors4, aligned='left')
        ]

        lines_new[1].align(lines_new[0], char_index=4, other_char_index=10)

        for i, line in enumerate(lines_new):
            if i > 1:
                scale = 0.5
            else:
                scale = 0.7
            if i > 1:
                offset = 2
                spc = 2
            else:
                offset = 0
                spc = 3
            if i == 1 or i == 4:
                display.add_text_in(line, line=spc * i + 1 + offset, indent=3, scale=scale)
                line.write(begin_time=t0, transition_time=0)

        first = lines_new[1]
        second = lines_new[4]

        t0 += 1

        first.move(-4 * display.line_spacing, begin_time=1, transition_time=1)
        second.move(-8 * display.line_spacing, begin_time=1, transition_time=1)

        t0 += 2
        second.replace(SimpleTexBObject(r"=\frac{\zeta'(0)}{\zeta(0)}=\log 2\pi", color='drawing',
                                        aligned='left'), src_letter_range=[10, 22], img_letter_range=[11, 17],
                       shift=[-0.1, 0.05],
                       begin_time=t0, transition_time=1)

        colors5 = [['important'] * 4, ['drawing'] * 24, ['text'] * 6, ['drawing'] * 6, ['example'] * 13,
                   ['drawing'] * 3, ['important'] * 8, ['drawing']]
        colors5 = [col for sublist in colors5 for col in sublist]
        t0 += 2
        third = SimpleTexBObject(
            r"\psi(x)=-\tfrac{1}{2\pi i}\int\limits_{a-i\infty}^{a+i \infty} \left(\log 2\pi-\frac{z}{z-1}+\sum_{n=1}^\infty "
            r"\frac{z}{(-2n)(z+2n)}+\sum_\rho \frac{ z}{\rho(z-\rho)}\right)x^{"
            r"z}\tfrac{{\rm d}z}{z}", aligned='left', color=colors5)

        display.add_text_in(third, line=5, scale=0.45, indent=3)
        third.write(begin_time=t0, transition_time=2)

        t0 += 3

        # set color for each letter
        colors = [['important'] * 4, ['drawing'] * 9, ['text'] * 2, ['example'] * 14, ['important']]
        colors_flat = [col for sublist in colors for col in sublist]

        fourth = SimpleTexBObject(
            r"\psi(x)=-\log(2\pi)+x-\sum_{n=1}^{\infty}{x^{-2n}\over -2n}-\sum_{\rho}{x^\rho\over{\rho}}",
            color=colors_flat, aligned='left')

        display.add_text_in(fourth, line=8, indent=3)
        fourth.write(begin_time=t0, transition_time=2)

        t0 += 3

        display2 = Display(flat=True, location=[40.55, 5, 12.775], scales=[6, 12, 6], number_of_lines=14, columns=1,
                           name="Display")
        display2.appear(begin_time=t0, transition_time=1)

        line = SimpleTexBObject(
            r"\tfrac{1}{2\pi i}\int\limits_{a-i\infty}^{a+i\infty}\frac{x^z}{\rho\left(z-\rho\right)}{\rm d}z=",
            color='white', aligned='left')
        display2.add_text(line, indent=1)
        line.write(begin_time=t0 + 1, transition_time=1)

        t0 += 3
        line2 = SimpleTexBObject(r"z\rightarrow s+\rho", color='text', aligned='left')
        display2.set_cursor_to_start_of_next_line()
        display2.set_cursor_to_start_of_next_line()
        display2.add_text(line2, indent=3)
        line2.write(begin_time=t0, transition_time=1)

        t0 += 2
        colors = [['white'] * 8, ['text'], ['white'] * 3, ['text'], ['white'] * 11, ['text'] * 2, ['white']]
        colors = [col for sublist in colors for col in sublist]
        line3 = SimpleTexBObject(
            r"\tfrac{1}{2\pi i}\int\limits_{a-\rho-i\infty}^{a-\rho+i\infty}\frac{x^{s+\rho}}{\rho\cdot s}{\rm d}s=",
            color=colors, aligned='left')
        display2.set_cursor_to_start_of_next_line()
        display2.set_cursor_to_start_of_next_line()
        display2.add_text(line3, indent=1)
        line3.write(begin_time=t0, transition_time=1)

        t0 += 2
        colors = [['white'] * 5, ['important'] * 4, ['text'] * 13, ['example'] * 4, ['white']]
        colors = [col for sublist in colors for col in sublist]
        line4 = SimpleTexBObject(
            r"\tfrac{1}{2\pi i}\frac{x^\rho}{\rho}\int\limits_{a-\rho-i\infty}^{a-\rho+i\infty}\frac{x^{s}}{s}{\rm d}s=",
            color=colors, aligned='left')
        display2.set_cursor_to_start_of_next_line()
        display2.set_cursor_to_start_of_next_line()
        display2.set_cursor_to_start_of_next_line()
        display2.add_text(line4, indent=1)
        line4.write(begin_time=t0, transition_time=1)

        t0 += 2
        colors = [['white'] * 5, ['important'] * 4, ['text'] * 13, ['example'] * 8, ['white'] * 16, ['important']]
        colors = [col for sublist in colors for col in sublist]
        line5 = SimpleTexBObject(
            r"\tfrac{1}{2\pi i}\frac{x^\rho}{\rho}\int\limits_{a-\rho-i\infty}^{a-\rho+i\infty}\frac{{\rm e}^{s \log x}}{s}{\rm d}s=\left\{\text{\begin{tabular}{c}$0$ \\ \\\\$\frac{x^\rho}{\rho}$ \end{tabular}}\right.",
            color=colors, aligned='left')
        display2.set_cursor_to_start_of_next_line()
        display2.set_cursor_to_start_of_next_line()
        display2.set_cursor_to_start_of_next_line()
        display2.add_text(line5, indent=1, scale=0.5)

        line5.write(letter_range=[0, 33], begin_time=t0, transition_time=1)

        t0 += 2
        #####################
        # coordinate system #
        #####################

        coord = CoordinateSystem(dim=3, lengths=[10, 10, 6], radii=[0.03, 0.03, 0.03],
                                 domains=[[-5, 5], [-5, 5], [0, 6]],
                                 all_n_tics=[5, 5, 3],
                                 location_of_origin=[7.1, 0, 0],
                                 rotation_euler=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", r"\left|\frac{\tfrac{1}{2}^s}{s}\right|"],
                                 all_tic_labels=[np.arange(-5, 5.1, 2),
                                                 np.arange(-5, 5.1, 2),
                                                 np.arange(0, 6.1, 2)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 label_colors=['drawing', 'drawing', 'example'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=t0, transition_time=2)

        t0 += 3
        function1 = ComplexPlaneWithSingularPoints(coord,
                                                   [lambda z: 0.5 ** z / z],
                                                   u=[-5, 5], v=[-5, 5],
                                                   detail=8,
                                                   special_x=[0],
                                                   special_y=[0],
                                                   resolution=30,
                                                   alpha=0.95,
                                                   metallic=0,
                                                   roughness=0.7,
                                                   emission_strength=0.1,
                                                   color='example',
                                                   name='first',
                                                   smooth=2)
        function1.appear(begin_time=t0, transition_time=2)

        t0 += 3
        coord2 = CoordinateSystem(dim=3, lengths=[10, 10, 6], radii=[0.03, 0.03, 0.03],
                                  domains=[[-5, 5], [-5, 5], [0, 6]],
                                  all_n_tics=[5, 5, 3],
                                  location_of_origin=[27.68, 0, 0],
                                  rotation_euler=[0, 0, 0],
                                  labels=[r"{\mathbb R}", r"i{\mathbb R}", r"\left|\frac{2^s}{s}\right|"],
                                  all_tic_labels=[np.arange(-5, 5.1, 2),
                                                  np.arange(-5, 5.1, 2),
                                                  np.arange(0, 6.1, 2)],
                                  materials=['drawing', 'drawing', 'drawing'],
                                  label_colors=['drawing', 'drawing', 'example'],
                                  text_size='large'
                                  )
        coord2.appear(begin_time=t0, transition_time=2)

        t0 += 3
        function2 = ComplexPlaneWithSingularPoints(coord2,
                                                   [lambda z: 2 ** z / z],
                                                   u=[-5, 5], v=[-5, 5],
                                                   detail=8,
                                                   special_x=[0],
                                                   special_y=[0],
                                                   resolution=30,
                                                   alpha=0.95,
                                                   metallic=0,
                                                   roughness=0.7,
                                                   emission_strength=0.1,
                                                   color='example',
                                                   name='second',
                                                   smooth=2)
        function2.appear(begin_time=t0, transition_time=2)

        t0 += 3

        path1 = Function(lambda x: [1, x, np.abs(0.5 ** (1 + 1j * x) / (1 + 1j * x))],
                         coord, domain=[-10, 10], num_points=100, color='text', mode='PARAMETRIC', name='path1',
                         thickness=0.1)
        path1.grow(begin_time=t0, transition_time=2)
        t0 += 3

        path2 = Function(lambda phi: [10 * np.cos(phi) + 1, 10 * np.sin(phi), np.abs(
            0.5 ** (10 * np.cos(phi) + 1 + 1j * 10 * np.sin(phi)) / (10 * np.cos(phi) + 1 + 1j * 10 * np.sin(phi)))],
                         coord, domain=[np.pi / 2, -np.pi / 2],
                         num_points=100, color='text', mode='PARAMETRIC', name='path2', thickness=0.1)
        path2.grow(begin_time=t0, transition_time=2)
        line5.write(letter_range=[33, 46], begin_time=t0 + 1, transition_time=1)

        line5.write(letter_set=[48], begin_time=t0 + 2, transition_time=1)

        t0 += 4
        path3 = Function(lambda x: [1, x, np.abs(2 ** (1 + 1j * x) / (1 + 1j * x))],
                         coord2, domain=[-10, 10], num_points=100, color='text', mode='PARAMETRIC', name='path3',
                         thickness=0.1)
        path3.grow(begin_time=t0, transition_time=2)
        t0 += 3

        path4 = Function(lambda phi: [10 * np.cos(phi) + 1, 10 * np.sin(phi), np.abs(
            2 ** (10 * np.cos(phi) + 1 + 1j * 10 * np.sin(phi)) / (10 * np.cos(phi) + 1 + 1j * 10 * np.sin(phi)))],
                         coord2, domain=[np.pi / 2, 3 * np.pi / 2],
                         num_points=100, color='text', mode='PARAMETRIC', name='path4', thickness=0.1)
        path4.grow(begin_time=t0, transition_time=2)
        line5.write(letter_set=[46, 47, 49, 50], begin_time=t0, transition_time=1)

    def intro(self):
        cues = self.sub_scenes['intro']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        ######################
        # mathematical preps #
        ######################

        pos_zeros = [14.1347, 21.022, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271, 48.0052, 49.7738, 52.9703,
                     56.4462, 59.347, 60.8318,
                     65.1125, 67.0798, 69.5464, 72.0672, 75.7047, 77.1448, 79.3374, 82.9104, 84.7355, 87.4253, 88.8091,
                     92.4919, 94.6513, 95.8706,
                     98.8312, 101.318, 103.726, 105.447, 107.169, 111.03, 111.875, 114.32, 116.227, 118.791, 121.37,
                     122.947, 124.257, 127.517, 129.579,
                     131.088, 133.498, 134.757, 138.116, 139.736, 141.124, 143.112, 146.001, 147.423, 150.054, 150.925,
                     153.025, 156.113, 157.598,
                     158.85, 161.189, 163.031, 165.537, 167.184, 169.095, 169.912, 173.412, 174.754, 176.441, 178.377,
                     179.916, 182.207, 184.874,
                     185.599, 187.229, 189.416, 192.027, 193.08, 195.265, 196.876, 198.015, 201.265, 202.494, 204.19,
                     205.395, 207.906, 209.577,
                     211.691, 213.348, 214.547, 216.17, 219.068, 220.715, 221.431, 224.007, 224.983, 227.421, 229.337,
                     231.25, 231.987, 233.693, 236.524, 237.77, 239.555, 241.049, 242.823, 244.071, 247.137, 248.102,
                     249.574, 251.015, 253.07, 255.306, 256.381, 258.61, 259.874, 260.805,
                     263.574, 265.558, 266.615, 267.922, 269.97, 271.494, 273.46, 275.587,
                     276.452, 278.251, 279.229, 282.465, 283.211, 284.836, 286.667,
                     287.912, 289.58, 291.846, 293.558, 294.965, 295.573, 297.979, 299.84,
                     301.649, 302.697, 304.864, 305.729, 307.219, 310.109, 311.165,
                     312.428, 313.985, 315.476, 317.735, 318.853, 321.16, 322.145,
                     323.467, 324.863, 327.444, 329.033, 329.953, 331.474, 333.645,
                     334.211, 336.842, 338.34, 339.858, 341.042, 342.055, 344.662,
                     346.348, 347.273, 349.316, 350.408, 351.879, 353.489, 356.018,
                     357.151, 357.953, 359.744, 361.289, 363.331, 364.736, 366.213,
                     367.994, 368.968, 370.051, 373.062, 373.865, 375.826, 376.324,
                     378.437, 379.873, 381.484, 383.444, 384.956, 385.861, 387.223,
                     388.846, 391.456, 392.245, 393.428, 395.583, 396.382]

        self.primes = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101,
        }

        def psi(x):
            if x <= 1:
                return 0
            s = 0
            for p in self.primes:
                if p <= x:
                    s += np.log(p)
                else:
                    break
            for power in range(2, int(np.ceil((np.log2(x))) + 1)):
                for p in self.primes:
                    if p ** power <= x:
                        s += np.log(p)
                    else:
                        break
            return s

        def non_trivial(x, n):
            s = 0
            for i in range(n):
                r = float(pos_zeros[i])
                s += 1 / (0.25 + r * r) * (0.5 * np.cos(r * np.log(x)) + r * np.sin(r * np.log(x)))

            return -2 * np.sqrt(x) * s

        ###########
        # display #
        # #########

        ibpy.set_sun_light(location=[10, -37, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[20, -37, 10])

        empty = EmptyCube(location=[20, 0, 10], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(flat=False,
                          location=[31.5, 5, 18], scales=[12, 5, 4], number_of_lines=8, columns=1)

        display.appear(begin_time=0, transition_time=2)

        ###############
        #   text    ###
        ###############
        t0 += 3
        title = SimpleTexBObject(r"\text{The prime counting function }\psi", color='important', aligned='center',
                                 name='title')
        display.set_title(title, scale=0.9)
        title.write(begin_time=t0, transition_time=1)
        display.set_cursor_to_start_of_next_line()

        colors = [['drawing'] * 5, ['text'], ['drawing'] * 7, ['important'] * 6, ['drawing'], ['example']]
        colors_flat = [col for sublist in colors for col in sublist]

        psi_text = SimpleTexBObject(r"\psi(x)=x-\log 2\pi+\sum_\rho {x^\rho\over \rho}+\log\left(\tfrac{x}{\sqrt{x^{"
                                    r"2}-1}}\right)", color=colors_flat, aligned='left', name="pi_text")
        display.add_text(psi_text, indent=1, scale=1.1)

        # #####################
        # # coordinate system #
        # #####################

        coord = CoordinateSystem(dim=2, lengths=[20, 20], radii=[0.03, 0.03],
                                 domains=[[0, 40], [0, 40]],
                                 all_n_tics=[8, 8],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"x", r"\psi(x)"],
                                 all_tic_labels=[np.arange(0, 40.1, 5),
                                                 np.arange(0, 40.1, 5), ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large',
                                 tip_lengths=[1, 1],
                                 )
        coord.appear(begin_time=0, transition_time=2)

        prime_texts = []
        for p in self.primes:
            if p < 40:
                prime_texts.append(SimpleTexBObject(str(p), color='text', aligned='center', name="prime_" + str(p),
                                                    location=coord.coords2location([p, -2, 0]), scale=2))

        prime_powers = [SimpleTexBObject(r"2^2", color='example', aligned='center', name="prime_power_" + str(4),
                                         location=coord.coords2location([4, -4, 0]), scale=2),
                        SimpleTexBObject(r"2^3", color='example', aligned='center', name="prime_power_" + str(8),
                                         location=coord.coords2location([8, -4, 0]), scale=2),
                        SimpleTexBObject(r"3^2", color='example', aligned='center', name="prime_power_" + str(9),
                                         location=coord.coords2location([9, -4, 0]), scale=2),
                        SimpleTexBObject(r"2^4", color='example', aligned='center', name="prime_power_" + str(16),
                                         location=coord.coords2location([16, -4, 0]), scale=2),
                        SimpleTexBObject(r"5^2", color='example', aligned='center', name="prime_power_" + str(25),
                                         location=coord.coords2location([25, -4, 0]), scale=2),
                        SimpleTexBObject(r"3^3", color='example', aligned='center', name="prime_power_" + str(27),
                                         location=coord.coords2location([27, -4, 0]), scale=2),
                        SimpleTexBObject(r"2^5", color='example', aligned='center', name="prime_power_" + str(32),
                                         location=coord.coords2location([32, -4, 0]), scale=2),
                        ]

        #############
        # functions #
        #############

        pi_duration = 10
        delta = pi_duration / len(prime_texts)
        for i, txt in enumerate(prime_texts):
            txt.write(begin_time=t0 + i * delta, transition_time=delta)

        delta = pi_duration / len(prime_powers)
        for i, txt in enumerate(prime_powers):
            txt.write(begin_time=t0 + i * delta, transition_time=delta)

        psi_plot = PieceWiseFunction(lambda x: psi(x), coord, domain=[1, 40], numpoints=100,
                                     singular_points=[2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32,
                                                      37, 39], color='important', name="func_psi")
        psi_plot.grow(begin_time=t0, transition_time=pi_duration)

        t0 += 1 + pi_duration

        lambdas = [lambda x: x,
                   lambda x: x - np.log(2 * np.pi),
                   ]

        for i in range(0, 100, 5):
            lambdas.append(lambda x, i=i: x - np.log(2 * np.pi) + non_trivial(x, i))

        lambdas.append(lambda x: x - np.log(2 * np.pi) + non_trivial(x, 100) + np.log(x / np.sqrt(x ** 2 - 1)))

        colors = [
            ['text'],
            ['text', 'drawing'],
        ]

        for i in range(20):
            colors.append(['text', 'drawing', 'important'])

        colors.append(['text', 'drawing', 'important', 'example'])

        psi_composition = Function(lambdas, coord, domain=[1.01, 40], num_points=1000, colors=colors,
                                   color_mode='voronoi',
                                   name='func_psi_composition', shadow=False)

        psi_composition.grow(begin_time=t0, transition_time=3)
        psi_text.write(letter_range=[0, 6], begin_time=t0, transition_time=1)
        t0 += 4
        psi_composition.next(begin_time=t0, transition_time=1)
        psi_text.write(letter_range=[6, 12], begin_time=t0, transition_time=2)
        t0 += 3

        delta = 10 / (len(lambdas) - 3)
        for i in range(2, len(lambdas) - 1):
            psi_composition.next(begin_time=t0 + (i - 2) * delta, transition_time=delta)
        psi_text.write(letter_range=[12, 19], begin_time=t0, transition_time=3)

        t0 += 11
        psi_composition.next(begin_time=t0, transition_time=1)
        psi_text.write(letter_range=[19, len(psi_text.get_letters())], begin_time=t0, transition_time=1)

        ###############
        # arrows #
        ###############
        arrow_colors = ['text', 'example', 'important']
        arrows = []

        locations = []
        loc = psi_text.get_letter(5).get_world_location()
        locations.append(loc)
        locations.append(psi_text.get_letter(23).get_world_location())
        locations.append(psi_text.get_letter(15).get_world_location())
        for loc in locations:
            loc.x += 1.75
            loc.x -= 1.56
            loc.z -= 1.75
            loc.z -= 0.7878

        length = 4
        radius = 0.3
        for i, color in enumerate(arrow_colors):
            arrow = Arrow(length=length, radius=radius, rotation_euler=[0, 0, 0],
                          location=locations[i], color=color,
                          name='arrow_' + str(i), shadow=False)
            arrows.append(arrow)

        t0 += 1

        ###############################
        # zeta and the special points #
        ###############################
        debug = 1  # put to 1
        coord2 = CoordinateSystem(dim=3, lengths=[5, 20, 10], radii=[0.03, 0.03, 0.03],
                                  domains=[[-10, 10], [-40, 40], [0, 40]],
                                  all_n_tics=[int(4 / debug), int(8 / debug), int(5 / debug)],
                                  rotation_euler=[0, 0, -np.pi / 2],
                                  location_of_origin=[34, 10, -2.69],
                                  labels=[r"{\mathbb R}", r"i{\mathbb R}", "|z|"],
                                  all_tic_labels=[np.arange(-10, 10.1, 5 * debug),
                                                  np.arange(-40, 40.1, 10 * debug),
                                                  np.arange(0, 40.1, 8 * debug)],
                                  materials=['drawing', 'drawing', 'drawing'],
                                  text_size='large',
                                  tip_lengths=[1, 1, 1],
                                  )
        coord2.appear(begin_time=t0, transition_time=5)

        mp.pretty = True  # convert the clumpsy mp notation into standard python notation
        zeros = [0]
        pos_zeros = []
        zero = 0
        n = 1
        while zero < 40:
            zero = mp.zetazero(n).imag
            if zero < 40:
                pos_zeros.append(zero)
                zeros.append(zero)
                zeros.append(-zero)
            n += 1

        zeros.sort()
        print(zeros)

        t0 += 6
        complex_function = ComplexPlaneWithSingularPoints(coord2,
                                                          lambda z: mp.zeta(z),
                                                          u=[-10, 10], v=[-40, 40],
                                                          special_x=[-10, -8, -6, -4, -2, 0, 0.5, 1],
                                                          special_y=zeros,
                                                          detail=int(5 / debug),
                                                          resolution=50 / debug,
                                                          alpha=0.95,
                                                          metallic=0,
                                                          roughness=0.7,
                                                          emission_strength=0.1,
                                                          scale=[0.25, 0.25, 0.25],
                                                          smooth=2, name='zeta_full')

        complex_function.appear(begin_time=t0, transition_time=2)

        t0 += 2
        arrows[0].grow(begin_time=t0, transition_time=2, modus='from_bottom')
        arrows[1].grow(begin_time=t0 + 3, transition_time=2, modus='from_bottom')
        arrows[2].grow(begin_time=t0 + 6, transition_time=2, modus='from_bottom')

        # arrows
        i = 0
        length = 2
        radius = 0.1

        arrow_colors = ['text', 'example', 'example', 'example', 'example', 'important', 'important',
                        'important', 'important', 'important', 'important', 'important', 'important', 'important',
                        'important', 'important', 'important']
        arrow_times = [[t0], [t0 + 3] * 4, [t0 + 6] * 12]
        arrow_times = [time for sublist in arrow_times for time in sublist]

        for color, time in zip(arrow_colors, arrow_times):

            if i == 0:
                arrow = Arrow(length=length, radius=radius, rotation_euler=[0, 0, 0],
                              location=coord2.coords2location([1, 0, 7.5]), color=color, name='arrow_' + str(i))
            elif i in {1, 2, 3, 4}:
                arrow = Arrow(length=length, radius=radius, rotation_euler=[0, 0, 0],
                              location=coord2.coords2location([-2 * (i), 0, 1]),
                              color=color, name='arrow_' + str(i))
            else:
                if i % 2 == 0:
                    arrow = Arrow(length=length, radius=radius, rotation_euler=[0, 0, 0],
                                  location=coord2.coords2location([0.5, pos_zeros[int((i - 5) / 2)], 1]),
                                  color=color, name='arrow_' + str(i))
                else:
                    arrow = Arrow(length=length, radius=radius, rotation_euler=[0, 0, 0],
                                  location=coord2.coords2location([0.5, -pos_zeros[int((i - 5) / 2)], 1]),
                                  color=color, name='arrow_' + str(i))

            coord2.add_object(arrow)
            if i == 0:
                arrow.rotate_by(rotation_euler=[np.pi * 3 / 4, 0, 0])
            elif i in {1, 2, 3, 4}:
                # arrow.rotate_by(rotation_euler=[0, 0, 3 * np.pi / 4])
                arrow.rotate_by(rotation_euler=[5 * np.pi / 4, 0, 0])
            else:
                # arrow.rotate_by(rotation_euler=[0, 0, 5 * np.pi / 4])
                arrow.rotate_by(rotation_euler=[3 * np.pi / 4, 0, 0])

            arrow.grow(begin_time=time, transition_time=2, modus="from_top")
            i += 1

    def branding(self):
        cues = self.sub_scenes['intro']
        t0 = 0  # cues['start']
        ibpy.set_shadow(True)

        ###########
        # display #
        # #########

        prime_occurance_times=[6, 8, 12,16, 24, 28, 36, 40, 50, 54]
        logo = Logo(location=[7, 0, 0], length=10, colors=['important', 'drawing', 'drawing'],thickness=0.1,details=50)
        logo.appear(begin_times=prime_occurance_times, transition_times=[1])

        ibpy.set_sun_light(location=[10, -37, 45])  # maybe change location to [20,0,35] to match shadows
        ibpy.set_camera_location(location=[20, -37, 10])
        empty = EmptyCube(location=[20, 0, 10], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(flat=True,
                                  location=[20, 5, 18], scales=[12, 5, 4], number_of_lines=8, columns=3)

        display.appear(begin_time=0, transition_time=2)

        t0 += 3
        colors=[['important']*12,['text']*3,['example']]
        title = SimpleTexBObject(r"\text{Prime numbers and the }\zeta\text{--function}", color=flatten(colors))
        display.set_title(title)
        title.write(begin_time=t0, transition_time=2)

        t0 += 3
        count = 0
        hiders =[]
        offsets = [-4.73,-3.29,-3.29,-1.7,-2,-1.1,-1.1,-0.75,-0.8]
        for i in range(2, 20):
            l = (i-2)%6+1
            col = int((i-2)/6)+1
            factors = prime_factors(i)
            line = get_prime_string(i)
            colors = get_color(line)

            b_object = SimpleTexBObject(line, color=colors, aligned='left')
            display.add_text_in(b_object, indent=1,column=col,line = l)
            b_object.write(begin_time=t0, transition_time=1)

            if col==1:
                hiders.append(b_object)

            if len(factors) == 1:
                offset = int(len(line)/2)
                print("write "+line+" at time "+str(t0))
                copy = b_object.copy(letter_range=[offset+1,len(line)])
                copy.ref_obj.scale = b_object.ref_obj.scale
                center = logo.get_center(count)
                if count<len(offsets):
                    center[0]+=offsets[count]
                else:
                    center[0]+=offsets[-1]
                copy.move_to(center,
                             begin_time=prime_occurance_times[count]+1,
                             transition_time=1,
                             global_system=True)
                sc = 6*logo.get_scale(count)
                copy.scale(
                    initial_scale=copy.ref_obj.scale,
                    final_scale=sc*copy.ref_obj.scale,
                    begin_time=prime_occurance_times[count]+1,
                    transition_time=1)
                count += 1
            t0 += 2

        for hider in hiders:
            hider.move(direction=[0,0,-0.1],begin_time=t0,transition_time=1)

        t0+=2

        for i in range(20,26):
            col = 1
            l = i-19
            factors = prime_factors(i)
            line = get_prime_string(i)
            colors = get_color(line)

            b_object = SimpleTexBObject(line, color=colors, aligned='left')
            display.add_text_in(b_object, indent=1, column=col, line=l,hidden=True)
            b_object.write(begin_time=t0, transition_time=1)
            display.un_hide(b_object, begin_time=t0, transition_time=0.1)

            if len(factors) == 1:
                offset = int(len(line) / 2)
                print("write " + line + " at time " + str(t0))
                copy = b_object.copy(letter_range=[offset + 1, len(line)])
                copy.ref_obj.scale = b_object.ref_obj.scale
                center = logo.get_center(count)
                if count < len(offsets):
                    center[0] += offsets[count]
                else:
                    center[0] += offsets[-1]
                copy.move_to(center,
                             begin_time=prime_occurance_times[count] + 1,
                             transition_time=1,
                             global_system=True)
                sc = 6 * logo.get_scale(count)
                copy.scale(
                    initial_scale=copy.ref_obj.scale,
                    final_scale=sc * copy.ref_obj.scale,
                    begin_time=prime_occurance_times[count] + 1,
                    transition_time=1)
                count += 1
            t0+=2

        # now the display is full
        # hide entries in the first column

        coord = CoordinateSystem(dim=2, lengths=[12, 12], radii=[0.03, 0.03], domains=[[-4, 4], [-4, 4]],
                                 all_n_tics=[8, 8], location_of_origin=[30, 0, 4.5],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", ],
                                 all_tic_labels=[np.arange(-4, 4.1, 1), np.arange(-4, 4.1, 1)],
                                 materials=['drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=0, transition_time=5)

        mp.pretty = True  # convert the clumpsy mp notation into standard python notation

        zeta = Function([lambda x: [np.real(mp.zeta(0.5 + 1j * x)), np.imag(mp.zeta(0.5 + 1j * x)), 0]], coord,
                        domain=[0, 100], num_points=1000, color='example', mode='PARAMETRIC', name='zeta',
                        thickness=0.05)
        zeta.grow(begin_time=6, transition_time=50)

    def play(self):
        super().play()
        # self.decomposition()
        # self.decomposition_complex()
        # self.real_functions()
        # self.complex_functions()
        # self.zeta_sum()
        # self.zeta_continuation()
        # self.decomposition_poles()
        # self.show_zeros()
        # self.result()
        # self.psi_contributions()
        # self.psi_contributions_cumulative()
        # self.prime_counting_functions()
        # self.overview()
        # self.zeta_decomposition()
        # self.algebra()
        # self.contour_integral()
        # self.intro()
        self.branding()


def map_between_coordinate_systems(coord, coord2, src, frame, func):
    x, y, z = coord.location2coords(src.get_location_at_frame(frame))
    Z = x + 1j * y
    w = func(Z)
    X = np.real(w)
    Y = np.imag(w)
    Z = z
    return coord2.coords2location([X, Y, Z])


if __name__ == '__main__':
    try:
        example = Primes2()
        example.create(name="branding")
        # example.create(name="algebra")
        # example.render(
    except:
        print_time_report()
        raise ()
