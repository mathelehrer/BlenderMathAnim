import os

import bpy
import mpmath as mp
import numpy as np

from appearance.textures import make_complex_function_material
from objects.cylinder import Cylinder
from objects.plane_complex import ComplexPlane
from objects.coordinate_system import CoordinateSystem
from objects.number_line import NumberLine
from objects.plane_with_singular_points import PlaneWithSingularPoints
from objects.geometry.sphere import Sphere
from objects_old.graph_bobject import GraphBObjectOld
from objects.tex_bobject import SimpleTexBObject
from objects_old.tex_bobject_old import TexBObjectOld
from utils.constants import BLEND_DIR
from utils.utils import print_time_report, finish_noise


def create_zeta():
    z1 = mp.im(mp.zetazero(1))
    z2 = mp.im(mp.zetazero(2))
    z3 = mp.im(mp.zetazero(3))
    z4 = mp.im(mp.zetazero(4))
    z5 = mp.im(mp.zetazero(5))
    z6 = mp.im(mp.zetazero(6))
    z7 = mp.im(mp.zetazero(7))
    z8 = mp.im(mp.zetazero(8))
    z9 = mp.im(mp.zetazero(9))
    z10 = mp.im(mp.zetazero(10))

    plane1 = PlaneWithSingularPoints(u=[2, 51], v=[0, 51], resolution=50, location=[26.5, -25.5, 0])
    make_complex_function_material(plane1, mp.zeta)
    print("two finished")

    plane2 = PlaneWithSingularPoints(u=[2, 51], v=[-51, 0], resolution=50, location=[26.5, -25.5, 0])
    make_complex_function_material(plane2, mp.zeta)
    print("two finished")

    res = 32
    fine = 1
    plane3 = PlaneWithSingularPoints(
        special_x=[-50, -48, -46, -44, -42, -40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12,
                   -10, -8, -6, -4, -2, 0.5, 1], special_y=[0, -z1, -z2, -z3, -z4, -z5, -z6, -z7, -z8, -z9, -z10],
        u=[-51, 2], v=[-51, 2],
        resolution=res, detail=fine, location=[-24.5, -24.5, 0])
    make_complex_function_material(plane3, mp.zeta)
    print("three finished")

    plane4 = PlaneWithSingularPoints(
        special_x=[-50, -48, -46, -44, -42, -40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12,
                   -10, -8, -6, -4, -2, 0.5, 1], special_y=[0, z1, z2, z3, z4, z5, z6, z7, z8, z9, -z10],
        u=[-51, 2], v=[-2, 51],
        resolution=res, detail=fine, location=[-24.5, 24.5, 0])
    make_complex_function_material(plane4, mp.zeta)
    print("three finished")


def sample_scene():
    for i in range(-51, 51):
        plane = PlaneWithSingularPoints(special_x=[2], u=[2, 51], v=[i, i + 1], resolution=7, detail=14,
                                        location=[26.5, i + 0.5, 0])
        make_complex_function_material(plane, mp.zeta)
        print(str(i) + " finished")
    print("one finished")


def tex_scene():
    line = TexBObjectOld(
        r'\zeta(s) = \sum\limits_{n=1}^\infty {1\over n^s}',
        location=[0, 0, 6.75],
        rotation_euler=[74 * np.pi / 180, 0, 0],
        scale=1.5,
        centered=True,
        name="definition"
    )
    line.add_to_blender(begin_time=1)
    # line.disappear(begin_time=60)


def tex_scene2():
    line = TexBObjectOld(r'\zeta(s) = \sum\limits_{n=1}^\infty {1\over n^s}',
                         r'\zeta(s) = \prod\limits_{p}^\infty (1-p^{-s})^{-1}',
                         location=[0, 0, 5],
                         rotation_euler=[0, 0, 0],
                         scale=2,
                         centered='left',
                         name="euler_relation")
    line.add_to_blender(appear_frame=30, appear_mode='per_curve')
    line.morph_figure('next', start_time=2)


def writing():
    line = TexBObjectOld(
        r'\zeta(s) = \sum\limits_{n=1}^\infty {1\over n^s}',
        location=[0, 0, 6.75],
        rotation_euler=[74 * np.pi / 180, 0, 0],
        scale=1.5,
        centered=True,
        name="i_m_writing"
    )
    line.add_to_blender(begin_time=1, transition_time=5)
    # line.disappear(begin_time=60)


def hugo_und_lucy():
    line = TexBObjectOld(
        r'\text{Hugo Martin}',
        r'\text{Lucy Martin}',
        location=[0, 0, 6.75],
        rotation_euler=[74 * np.pi / 180, 0, 0],
        scale=15,
        centered='right',
        name="i_m_writing"
    )
    line.add_to_blender(begin_time=1, transition_time=5)
    line.morph_figure('next', start_time=7)


def hugo_und_lucy2():
    line = TexBObjectOld(
        r'\text{Hugo}',
        r'\text{Lucy}',
        location=[0, 0, 6.75],
        rotation_euler=[74 * np.pi / 180, 0, 0],
        scale=15,
        centered='right',
        name="i_m_writing",
    )
    line.add_to_blender(begin_time=1, transition_time=5)
    line.morph_figure('next', start_time=7)


def func(x): return 2 ** x


def graphing():
    g = GraphBObjectOld(
        func,
        x_range=4,
        location=[0, -2, 0],
        centered=True,
        tick_step=[100, 100],
        width=24,
        x_label=r'\text{Time}',
        y_label=r'\text{Population}',
        y_label_rot=True,
    )
    g.add_to_blender(begin_time=0)


def simple_moving_sphere():
    sphere = Sphere(1, location=[0, 0, 0], mesh_type='uv')
    sphere.move_to(0, 10, new_location=[1, 0, 0])


def cylinder():
    l = 10
    r = 0.1
    cyl = Cylinder(length=l / 2, location=[l / 2, 0, 0], radius=r, rotation_euler=[0, np.pi / 2, 0])
    cyl.add_to_blender(begin_time=0)


def numberline2():
    l = 10
    r = 0.05
    nl = NumberLine(l, r)
    nl.appear(begin_time=0, transition_time=10)


def numberline3():
    l = 10
    r = 0.01
    # nl = NumberLine2(l,r,domain=[-10,10],n_tics = 10, label="x", tic_labels='AUTO')
    nl = NumberLine(l, r, domain=[-10, 10], n_tics=10, label="x", tic_labels=np.arange(-10, 11, 2))
    nl.appear(begin_time=0, transition_time=2)


def coordinate_system3d():
    l = 10
    r = 0.03
    coord = CoordinateSystem(dim=3, lengths=[l, l, l], radii=[r, r, r], domains=[[-10, 10], [-10, 10], [0, 20]],
                             all_n_tics=[10, 5, 10], labels=["x", "y", "z"], location_of_origin=[0, 0, 0],
                             all_tic_labels=[np.arange(-10, 11, 2), np.arange(-10, 11, 4), np.arange(0, 20, 2)],
                             materials=['drawing', 'drawing', 'drawing'],
                             shadings=['redder', 'greener', 'bluer'])
    coord.appear(begin_time=0, transition_time=5)


def simple():
    line = SimpleTexBObject(
        r'\text{Hugo}',
        location=[0, 0, 1],
        rotation_euler=[74 * np.pi / 180, 0, 0],
        scale=15,
        centered='left',
        name="i_m_writing",
    )
    line.appear(begin_time=1, writing_time=5)


def numberline_unrotated():
    l = 10
    r = 0.01
    # nl = NumberLine2(l,r,domain=[-10,10],n_tics = 10, label="x", tic_labels='AUTO')
    nl = NumberLine(l, r, domain=[0, 10],
                    n_tics=10,
                    label="x",
                    tic_labels=np.arange(0, 11, 1),
                    direction='deep',
                    origin=0,
                    location_of_origin=[1, 1, 1])
    nl.appear(begin_time=0, transition_time=2)


def plane():
    coords = CoordinateSystem(dim=3, lengths=[20, 20, 10], radii=[0.03, 0.03, 0.03],
                              domains=[[-1, 1], [-1, 1], [0, 1]],
                              all_n_tics=[4, 4, 2], labels=[r"\mathbb{R}", r"i\mathbb{R}", "|f(z)|"], location_of_origin=[0, 0, 0],
                              all_tics_labels=[np.arange(-1, 1.1, 0.5), np.arange(-1, 1.1, 0.5), np.arange(0, 1.1, 0.5)],
                              materials=['white', 'white', 'white'], shadings=['redder', 'greener', 'bluer']
                              )
    plane = ComplexPlane(coords,[lambda z:z,lambda z:z**2],u=[-1,1],v=[-1,1],resolution=100,alpha=0.3,metalicity=1)

    coords.appear(begin_time=0, transition_time=5)
    plane.appear(begin_time=5, transition_time=1)


def main():
    """
    Use this to test scene
    """
    initialize_blender(light_location=(-10,0,35),light_energy=10)

    # sample_scene()
    # tex_scene()
    # tex_scene2()
    # writing()
    # hugo_und_lucy()
    # hugo_und_lucy()
    # graphing()
    # simple_moving_sphere()
    # cylinder()
    # numberline()
    # numberline2()
    # numberline3()
    # simple()
    # coordinate_system3d()
    # numberline_unrotated()

    # display_with_text()
    plane()
    bpy.ops.wm.save_mainfile(filepath=os.path.join(BLEND_DIR, "complex_plane.blend"))


if __name__ == "__main__":
    try:
        main()
        print_time_report()
    except:
        print_time_report()
        finish_noise(error=True)
        raise ()
