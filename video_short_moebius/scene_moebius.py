from collections import OrderedDict

import numpy as np

from interface import ibpy
from objects.coordinate_system import CoordinateSystem
from objects.display import Display
from objects.function import Function, PieceWiseFunction
from objects.geometry.sphere import Sphere
from objects.tex_bobject import SimpleTexBObject
from perform.scene import Scene
from utils.utils import print_time_report


def flatten(colors):
    colors_flat = [col for sublist in colors for col in sublist]
    return colors_flat


def moebius(a,b,c,d,z):
    return (a*z+b)/(c*z+d)


def riemann(x, y):
    d = 1 + x ** 2 + y ** 2
    X = x / d
    Y = y / d
    Z = (d - 1) / d
    return [X, Y, Z]


def rotation(vec):
    [X, Y, Z] = vec
    Z -= 0.5
    x = -Z
    Z = X
    X = x
    Z += 0.5
    return [X, Y, Z]


class Moebius(Scene):
    def __init__(self):
        self.primes = None

        self.sub_scenes = OrderedDict([
            ('title', {'duration': 5}),
            ('real_function', {'duration': 12}),
            ('complex_function', {'duration': 20}),
            ('riemann_sphere0', {'duration': 5}),
            ('riemann_sphere', {'duration': 26}),

        ])
        # in the super constructor the timing is set for all scenes
        super().__init__(light_energy=2)

    def title(self):
        cues = self.sub_scenes['title']
        t0 = 0  # cues['start']
        digits = 1  # 0 for testing
        ###########
        # display #
        # #########
        ibpy.set_camera_location(location=[0, -70, -20])
        display = Display(flat=True, show_lines=False, number_of_lines=2, location=[0, 0, 0], scales=[40, 4, 1])
        display.appear(begin_time=t0 + 0, transition_time=1)

        ###############
        #   text    ###
        ###############

        colors = [['text'] * 18, ['important']]
        colors = flatten(colors)
        title = SimpleTexBObject(r"\text{What's the meaning of }z\mapsto \tfrac{z+1}{z-1}", color=colors,
                                 aligned='center', thickness=0.25)
        display.set_title(title, shift=[-2, 0])
        title.write(begin_time=1, transition_time=3)

    def real_function(self):
        cues = self.sub_scenes['real_function']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[0, -20, 0])

        func = lambda z: (z + 1) / (z - 1)

        #####################
        # coordinate system #
        #####################

        coord = CoordinateSystem(dim=2, lengths=[10, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-5, 5], [-5, 5]],
                                 all_n_tics=[5, 10], location_of_origin=[0, 0, 0],
                                 labels=[r"z", r"{f(z)}"],
                                 all_tic_labels=[
                                     np.arange(-5, 5.1, 2),
                                     np.arange(-5, 5.1, 1),
                                 ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=t0, transition_time=5)

        display = Display(flat=True, show_lines=False, number_of_lines=1, location=[-6.5, 0, 6], scales=[5, 0.5, 1])
        display.appear(begin_time=t0 + 0, transition_time=1)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{Real function}", color='example',
                                 aligned='center', thickness=0.25)
        display.set_title(title, shift=[-0, 0])
        title.write(begin_time=1, transition_time=3)

        function = PieceWiseFunction(func, coord, singular_points=[1], domain=[-5, 5], num_points=50, color='example')
        t0 += 6
        function.grow(begin_time=t0, transition_time=5)

    def complex_function(self):
        cues = self.sub_scenes['complex_function']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6.5, -25, 0])

        func = lambda z: (z + 1) / (z - 1)

        #####################
        # coordinate system #
        #####################

        coord = CoordinateSystem(dim=2, lengths=[10, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-5, 5], [-5, 5]],
                                 all_n_tics=[5, 5], location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}"],
                                 all_tic_labels=[
                                     np.arange(-5, 5.1, 2),
                                     np.arange(-5, 5.1, 2),
                                 ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=t0, transition_time=5)

        display = Display(flat=True, show_lines=False, number_of_lines=1, location=[9, 0, 7.7], scales=[5, 0.5, 1])
        display.appear(begin_time=t0 + 0, transition_time=1)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{Complex function}", color='example',
                                 aligned='center', thickness=0.25)
        display.set_title(title, shift=[-0, 0])
        title.write(begin_time=1, transition_time=3)

        grid_r=[]
        grid_phi=[]
        n_phi = 9
        radii = []
        for i in range(-3, 4):
            radii.append(2 ** i)

        for r in radii:
            grid_line = Function([lambda t: [r * np.cos(t), r * np.sin(t), -0],
                                  ], coord, domain=[0, 2 * np.pi], num_points=100,
                                 hue_functions=['x', 'z', 'x', 'z'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_r' + str(r), thickness=0.01)
            grid_r.append(grid_line)

        for i in range(0, n_phi + 1):
            phi = 2 * i / n_phi * np.pi
            grid_line = Function([lambda t: [t * np.cos(phi), t * np.sin(phi), -0],
                                  ], coord, domain=[0, 8], num_points=100,
                                 hue_functions=['x', 'z', 'x', 'z'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_phi' + str(phi), thickness=0.01)
            grid_phi.append(grid_line)

        delay = 3
        t0 += 6
        for i, line in enumerate(grid_r):
            line.grow(begin_time=t0 + i / len(grid_r) * delay, transition_time=3)

        t0 += 5
        for i, line in enumerate(grid_phi):
            line.grow(begin_time=t0 + i / len(grid_phi), transition_time=3)

        # t0 += 5
        # for line in grid_r:
        #     line.next(begin_time=t0, transition_time=3)
        #
        # for line in grid_phi:
        #     line.next(begin_time=t0, transition_time=3)

    def complex_function2(self):
        cues = self.sub_scenes['complex_function']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6.5, -25, 0])

        func = lambda z: (z + 1) / (z - 1)

        #####################
        # coordinate system #
        #####################

        coord = CoordinateSystem(dim=2, lengths=[10, 10], radii=[0.03, 0.03, 0.03],
                                 domains=[[-5, 5], [-5, 5]],
                                 all_n_tics=[5, 5], location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}"],
                                 all_tic_labels=[
                                     np.arange(-5, 5.1, 2),
                                     np.arange(-5, 5.1, 2),
                                 ],
                                 materials=['drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=t0, transition_time=5)

        ###############
        #   text    ###
        ###############

        grid_r = []
        grid_phi = []
        n_phi = 9
        radii = []
        for i in range(-3, 4):
            radii.append(2 ** i)

        for r in radii:
            grid_line = Function([lambda t: [r * np.cos(t), r * np.sin(t), -0],
                                  lambda t: [
                                      np.real(moebius(1,1,1,-1,r * np.cos(t)+ r * np.sin(t)*1j)),
                                      np.imag(moebius(1,1,1,-1,r * np.cos(t)+ r * np.sin(t)*1j)),
                                      0
                                  ]
                                  ], coord, domain=[0, 2 * np.pi], num_points=100,
                                 hue_functions=['x', 'z', 'x', 'z'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_r' + str(r), thickness=0.01)
            grid_r.append(grid_line)

        for i in range(0, n_phi + 1):
            phi = 2 * i / n_phi * np.pi
            grid_line = Function([lambda t: [t * np.cos(phi), t * np.sin(phi), -0],
                                  lambda t: [
                                      np.real(moebius(1, 1, 1, -1, t * np.cos(phi) + t * np.sin(phi) * 1j)),
                                      np.imag(moebius(1, 1, 1, -1, t * np.cos(phi) + t * np.sin(phi) * 1j)),
                                      0
                                  ]
                                  ], coord, domain=[0, 8], num_points=100,
                                 hue_functions=['x', 'z', 'x', 'z'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_phi' + str(phi), thickness=0.01)
            grid_phi.append(grid_line)

        delay = 3
        t0 += 6
        for i, line in enumerate(grid_r):
            line.grow(begin_time=t0 + i / len(grid_r) * delay, transition_time=3)

        t0 += 5
        for i, line in enumerate(grid_phi):
            line.grow(begin_time=t0 + i / len(grid_phi), transition_time=3)

        t0 += 5
        for line in grid_r:
            line.next(begin_time=t0, transition_time=3)

        for line in grid_phi:
            line.next(begin_time=t0, transition_time=3)

    def riemann_sphere0(self):
        cues = self.sub_scenes['riemann_sphere0']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6.5, -25, 0])

        display = Display(flat=True, show_lines=False, number_of_lines=1, location=[9, 0, 7.7], scales=[5, 0.5, 1])
        display.appear(begin_time=t0 + 0, transition_time=1)

        ###############
        #   text    ###
        ###############

        title = SimpleTexBObject(r"\text{M\"obius transformation}", color='example',
                                 aligned='center', thickness=0.25)
        display.set_title(title, shift=[-0, 0])
        title.write(begin_time=1, transition_time=3)

    def riemann_sphere(self):
        cues = self.sub_scenes['complex_function']
        t0 = 0  # cues['start']


        func = lambda z: (z + 1) / (z - 1)

        ibpy.set_camera_location(location=[4.8, -4.09, 5.5])
        ibpy.set_camera_rotation(rotation=[np.pi / 180*52, 0,np.pi / 180*48])
        # #####################
        # # coordinate system #
        # #####################

        coord = CoordinateSystem(dim=3, lengths=[10, 10, 2], radii=[0.03, 0.03, 0.03],
                                 domains=[[-5, 5], [-5, 5], [0, 2]],
                                 all_n_tics=[10, 10, 2],
                                 location_of_origin=[0, 0, 0],
                                 labels=[r"{\mathbb R}", r"i{\mathbb R}", "{\mathbb R}"],
                                 all_tic_labels=[np.arange(-5.0, 5.1, 1),
                                                 np.arange(-5.0, 5.1, 1),
                                                 np.arange(0.0, 2.1, 1)],
                                 materials=['drawing', 'drawing', 'drawing'],
                                 text_size='large'
                                 )
        coord.appear(begin_time=0, transition_time=5)

        grid_r = []
        grid_phi = []
        n_phi = 9
        radii = []
        radii = []
        for i in range(-3, 4):
            radii.append(2 ** i)

        for r in radii:
            grid_line = Function([lambda t: [r * np.cos(t), r * np.sin(t), -0],
                                  lambda t: riemann(r * np.cos(t), r * np.sin(t)),
                                  lambda t: [np.real(moebius(1, 1, 1, -1, r * np.cos(t) + 1j * r * np.sin(t))),
                                             np.imag(moebius(1, 1, 1, -1, r * np.cos(t) + 1j * r * np.sin(t))),
                                             -0]
                                  ], coord, domain=[0, 2 * np.pi], num_points=100,
                                 hue_functions=['x', 'y', 'x', 'y'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_r' + str(i), thickness=0.01)
            grid_r.append(grid_line)

        for i in range(0, n_phi + 1):
            phi = 2 * i / n_phi * np.pi
            grid_line = Function([lambda t: [t * np.cos(phi), t * np.sin(phi), -0],
                                  lambda t: riemann(t * np.cos(phi), t * np.sin(phi)),
                                  lambda t: [
                                      np.real(moebius(1, 1, 1, -1, t * np.cos(phi) + 1j * t * np.sin(phi))),
                                      np.imag(moebius(1, 1, 1, -1, t * np.cos(phi) + 1j * t * np.sin(phi))), -0]
                                  ], coord, domain=[0, 2 ** 3], num_points=100,
                                 hue_functions=['x', 'y', 'x', 'y'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_phi' + str(phi), thickness=0.01)
            grid_phi.append(grid_line)

        delay = 2
        t0 += 6
        for i, line in enumerate(grid_r):
            line.grow(begin_time=t0 + i / len(grid_r) * delay, transition_time=3)

        t0 += 5
        for i, line in enumerate(grid_phi):
            line.grow(begin_time=t0 + i / len(grid_phi), transition_time=3)

        t0 += 5
        for line in grid_r:
            line.next(begin_time=t0, transition_time=3)

        for line in grid_phi:
            line.next(begin_time=t0, transition_time=3)

        sphere = Sphere(0.5, location=[0, 0, 0.5], mesh_type='uv', resolution=5, color='example', transmission=0.9,
                        ior=1, roughness=0, smooth=True)

        sphere.appear(begin_time=t0, transition_time=1.5)

        t0 += 4
        sphere.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=t0, transition_time=1)
        for line in grid_r:
            line.rotate(rotation_euler=[0, np.pi / 2, 0], pivot=[0, 0, 0.5], begin_time=t0, transition_time=1)
        for line in grid_phi:
            line.rotate(rotation_euler=[0, np.pi / 2, 0], pivot=[0, 0, 0.5], begin_time=t0, transition_time=1)

        grid_phi2 = []
        grid_r2 = []

        for r in radii:
            grid_line = Function([lambda t: rotation(riemann(r * np.cos(t), r * np.sin(t))),
                                  lambda t: [np.real(moebius(1, 1, 1, -1, r * np.cos(t) + 1j * r * np.sin(t))),
                                             np.imag(moebius(1, 1, 1, -1, r * np.cos(t) + 1j * r * np.sin(t))),
                                             -0]
                                  ], coord, domain=[0, 2 * np.pi], num_points=100,
                                 hue_functions=['x', 'y', 'x', 'y'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_r' + str(i), thickness=0.01)
            grid_r2.append(grid_line)

        for i in range(0, n_phi + 1):
            phi = 2 * i / n_phi * np.pi
            grid_line = Function([lambda t: rotation(riemann(t * np.cos(phi), t * np.sin(phi))),
                                  lambda t: [
                                      np.real(moebius(1, 1, 1, -1, t * np.cos(phi) + 1j * t * np.sin(phi))),
                                      np.imag(moebius(1, 1, 1, -1, t * np.cos(phi) + 1j * t * np.sin(phi))),
                                      -0]
                                  ], coord, domain=[0, 2 ** 3], num_points=100,
                                 hue_functions=['x', 'y', 'x', 'y'], color_mode='hue_color',
                                 mode='PARAMETRIC', name='grid_phi' + str(phi), thickness=0.01)
            grid_phi2.append(grid_line)

        t0 += 2
        for line in grid_r2:
            line.grow(begin_time=t0, transition_time=0)

        for line in grid_phi2:
            line.grow(begin_time=t0, transition_time=0)

        for line in grid_r2:
            line.next(begin_time=t0, transition_time=3)

        for line in grid_phi2:
            line.next(begin_time=t0, transition_time=3)

        ibpy.camera_move(shift=[-0.3, -0.21, -2.5], begin_time=5, transition_time=t0 - 10)
        ibpy.camera_rotate_to(rotation_euler=[np.pi / 3, 0, np.pi / 4], begin_time=5, transition_time=t0 - 10)

    def play(self):
        super().play()
        # self.title()
        # self.real_function()
        # self.complex_function()
        # self.complex_function2()
        self.riemann_sphere0()
        # self.riemann_sphere()


if __name__ == '__main__':
    try:
        example = Moebius()
        # example.create(name="title")
        example.create(name='riemann_sphere0')
        # example.render(
    except:
        print_time_report()
        raise ()
