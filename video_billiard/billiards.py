from collections import OrderedDict

import numpy as np

from compositions.compositions import create_glow_composition
from interface import ibpy
from interface.ibpy import Vector, Quaternion, set_parent, camera_move
from interface.interface_constants import BLENDER_EEVEE
from objects.billiards_objects import ScoreTable, f, BilliardsTable, BilliardsBall, BilliardPaper, \
    ReflectableBilliardPaper, BilliardBallReal

from objects.quadrilateral import BQuadrilateral
from objects.text2 import Text
from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.empties import EmptyCube
from objects.logo import LogoFromInstances
from objects.number_line import NumberLine2
from objects.plane import Plane
from objects.some_logo import SoME
from objects.table import Table
from objects.tex_bobject import SimpleTexBObject
from perform.scene import Scene
from utils.constants import FRAME_RATE
from utils.utils import print_time_report, flatten
from video_billiard.auxiliaries import random_prime

pi = np.pi
tau=2*pi


class Billard(Scene):
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('prime_factors',{'duration':35}),
            ('classroom',{'duration':55}),
            ('benford',{'duration':66}),
            ('mo_math',{'duration':30}),
            ('result',{'duration':15}),
            ('logo',{'duration':17}),
            ('pattern3',{'duration':35}),
            ('pattern2',{'duration':27}),
            ('pattern',{'duration':35}),
            ('mirroring',{'duration': 60}),
            ('reflexion',{'duration': 35}),
            ('reducible',{'duration': 30}),
            ('diagonal',{'duration': 17}),
            ('introduction',{'duration': 72}),
            ('score_table',{'duration': 30}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def prime_factors(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        create_glow_composition(threshold=1, type='BLOOM', size=4)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        # create connection to previous slide
        shift = 5
        x_axis = NumberLine2(length=9, location=[-4.5 - shift, 0, -5.25], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{Width}[w]", tip_length=0)

        y_axis = NumberLine2(length=9, location=[-5.25 - shift, 0, -4.5], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.1, 0, 0],
                             label_position="right",
                             axis_label_location=[-1.15, 0, 9.5],
                             axis_label=r"\text{Height}[h]", tip_length=0, direction="VERTICAL")
        y_axis.grow(begin_time=t0, transition_time=0)
        x_axis.grow(begin_time=t0, transition_time=0)

        score_table = ScoreTable(width=10, height=10, background_material="background",
                                 ur_material="custom1", ul_material="joker", br_material="important",
                                 font_material="text", location=[-shift, 0, 0])
        t0 = 0.5 + score_table.appear(begin_time=t0, transition_time=0)

        factors =["1=1","2=2","3=3","4=2\cdot 2","5=5","6=2\cdot 3","7=7","8=2\cdot 2\cdot 2","9=3\cdot 3","10=2\cdot 5"]
        colors = [
            ["text","text","text"],
            ["text","text","joker"],
            ["text","text","text"],
            ["text", "text", "joker","text","joker"],
            ["text", "text", "text"],
            ["text", "text", "joker", "text", "text"],
            ["text", "text", "text"],
            ["text", "text", "joker", "text", "joker","text","joker"],
            ["text", "text", "text", "text", "text"],
            ["text","text", "text", "joker", "text", "text"],
        ]

        locations = [[3,0,4-i] for i in range(10)]
        locations[9]=[2.704,0,4-9]
        lines =[SimpleTexBObject(factor,color=color,location=location,text_size="Large") for (factor,color,location) in zip(factors,colors,locations)]
        for line in lines:
            t0 = 0.5 + line.write(begin_time=t0, transition_time=0.3)

        t0 +=0.5

        # fourth column
        col = [[4,i] for i in range(1,11)]

        box = BQuadrilateral(vertices=[(2.5,0,0.5),(5.5,0,0.5),(5.5,0,1.5),(2.5,0,1.5)],color="example")
        box.grow(begin_time=t0, transition_time=1)

        t0 = 0.5 + score_table.turn_on(col,begin_time=t0,transition_time=3)

        entries = [
            [
                SimpleTexBObject(r"\text{Width}"),
                SimpleTexBObject(r"\text{Height}"),
                SimpleTexBObject(r"\text{Pocket}")
            ]
            ,
            [
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{UR}", color="custom1"),
            ]
            ,
            [
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{even}"),
                SimpleTexBObject(r"\text{UL}", color="joker"),
            ]
            ,
            [
                SimpleTexBObject(r"\text{even}"),
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{BR}", color="important"),
            ]
        ]

        table = Table(entries, location=[8, 0, 5], scale=1.4)
        for i in range(4):
            t0 = 0.5 + table.write_row(i, begin_time=t0, transition_time=0.5, verbose=True)

        box.disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + score_table.turn_off(col, begin_time=t0, transition_time=1)

        # eight row
        row = [[i,8] for i in range(1, 11)]

        box = BQuadrilateral(vertices=[(2.5, 0, -3.5), (6.25, 0, -3.5), (6.25, 0, -2.5), (2.5, 0, -2.5)], color="joker")
        box.grow(begin_time=t0, transition_time=1)

        t0 = 0.5 + score_table.turn_on(row, begin_time=t0, transition_time=3)

        t0+=1

        box.disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + score_table.turn_off(row, begin_time=t0, transition_time=1)

        self.t0 = t0

    def classroom(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        create_glow_composition(threshold=1, type='BLOOM', size=4)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -26, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)


        paper1 = BilliardPaper(width=9, height=4, rotation_quaternion=Quaternion([1, 1, 0, 0]), location=[-6.5, 0, 2],
                               paper_color="billiards_cloth_material", pivot="LOWER_LEFT")
        paper1.grow(begin_time=t0, transition_time=1)

        paper2 = BilliardPaper(width=3, height=7, rotation_quaternion=Quaternion([1, 1, 0, 0]),
                               paper_color="billiards_cloth_material", location=[-0.5, 0, -6])
        paper2.grow(begin_time=t0, transition_time=1)

        paper3 = BilliardPaper(width=8, height=10, rotation_quaternion=Quaternion([1, 1, 0, 0]),
                               paper_color="billiards_cloth_material", location=[3, 0, -4])
        t0 = 0.5 + paper3.grow(begin_time=t0, transition_time=1)

        paper1.show_ball(begin_time=t0, transition_time=1)
        paper3.show_ball(begin_time=t0, transition_time=1)
        paper2.show_ball(begin_time=t0, transition_time=1)


        papers = [paper2,paper1,  paper3]
        times = [23.67, 38.67, 42.67]
        [paper.disappear_ball(begin_time=time, transition_time=0.5) for paper, time in zip(papers, times)]
        t0 = max(times) +1

        shift = 5
        x_axis = NumberLine2(length=9, location=[-4.5 - shift, 0.15, -5.25], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{Width}[w]", tip_length=0)

        y_axis = NumberLine2(length=9, location=[-5.25 - shift, 0.15, -4.5], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.1, 0, 0],
                             label_position="right",
                             axis_label_location=[-1.15, 0, 9.5],
                             axis_label=r"\text{Height}[h]", tip_length=0, direction="VERTICAL")
        y_axis.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + x_axis.grow(begin_time=t0, transition_time=1)

        score_table = ScoreTable(width=10, height=10, background_material="background",
                                 ur_material="custom1", ul_material="joker", br_material="important",
                                 font_material="text", location=[-shift, 0.15,0])

        t0 = score_table.appear(begin_time=t0, transition_time=1, offset_for_slots=[0, 0, 0, 0.75])

        # turn billiard tables into scores
        square = Plane(location=[0, 0, 0], color="custom1", name="URSquare")
        square.rotate(rotation_euler=[pi / 2, 0, 0], begin_time=t0, transition_time=0)
        square.move_to(target_location=[2, 0, 0.5], begin_time=t0, transition_time=0)
        t0 = 0.5 + square.grow(begin_time=t0, transition_time=0.3, scale=0.5)

        t0 = square.move_to(target_location=[-7.5, 0, 1.5], begin_time=t0, transition_time=1)
        square.disappear(begin_time=t0 - 0.3, transition_time=0.6)
        paper2.disappear(begin_time=t0 - 0.3, transition_time=0.6)
        score_table.turn_on([[3, 7]], begin_time=t0, transition_time=0)
        t0 += 0.5

        square = Plane(location=[0, 0, 0], color="joker", name="ULSquare")
        square.rotate(rotation_euler=[pi / 2, 0, 0], begin_time=t0, transition_time=0)
        square.move_to(target_location=[-6, 0, 5.5], begin_time=t0, transition_time=0)
        t0 = 0.5 + square.grow(begin_time=t0, transition_time=0.3, scale=0.5)

        t0 = square.move_to(target_location=[-1.5, 0, -1.5], begin_time=t0, transition_time=1)
        square.disappear(begin_time=t0 - 0.3, transition_time=0.6)
        paper1.disappear(begin_time=t0 - 0.3, transition_time=0.6)
        score_table.turn_on([[9,4]], begin_time=t0,transition_time=0)
        t0 += 0.5

        square = Plane(location=[0, 0, 0], color="important", name="BRSquare")
        square.rotate(rotation_euler=[pi/2,0,0], begin_time=t0, transition_time=0)
        square.move_to(target_location=[10.5,0,-3.5], begin_time=t0, transition_time=0)
        t0 =0.5 + square.grow(begin_time=t0, transition_time=0.3, scale=0.5)

        t0 = square.move_to(target_location=[-2.5,0,4.5], begin_time=t0, transition_time=1)
        square.disappear(begin_time=t0-0.3, transition_time=0.6)
        paper3.disappear(begin_time=t0-0.3, transition_time=0.6)
        score_table.turn_on([[8,10]],begin_time=t0,transition_time=0)

        t0+=0.5

        score_table.turn_off([[9,4],[3,7],[8,10]],begin_time=t0,transition_time=0)
        t0+=0.5

        self.t0 = t0

    def mo_math(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 52 / 180 * pi])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)


        title_plane = Plane(u=[-5.5,5.5],v=[-2,2],color="image",src="momath.png")
        title_plane.move_to(target_location=[0, 0, 4], begin_time=0, transition_time=0)
        title_plane.rotate(rotation_euler=[pi / 2, 0, 0], begin_time=0, transition_time=0)
        t0 = 0.5 + title_plane.appear(begin_time=t0, transition_time=1)

        lesson_plane = Plane(u=[-9,9],v=[-2.5,2.5],color="image",src="lesson.png")
        lesson_plane.appear(begin_time=t0, transition_time=0)
        lesson_plane.move_to(target_location=[0,0,-2], begin_time=t0, transition_time=0)
        t0 =  0.5 + lesson_plane.rotate(rotation_euler=[pi / 2, 0, 0], begin_time=t0, transition_time=1)

        title_plane.rescale(rescale=0.5,begin_time=t0,transition_time=1)
        title_plane.move(direction=[-6,0,0.5],begin_time=t0,transition_time=1)
        lesson_plane.move(direction=[-6,0,4],begin_time=t0,transition_time=1)
        t0 = 0.5 + lesson_plane.rescale(rescale=0.5,begin_time=t0,transition_time=1)

        anotherroof = Plane(u=[-6,6],v=[-3.4,3.4],color="image",src="anotherroof.png")
        anotherroof.rotate(rotation_euler=[pi / 2, 0, 0],begin_time=t0,transition_time=0)
        anotherroof.move_to(target_location=[5,0,-2.5],begin_time=t0,transition_time=0)
        t0 = 0.5 + anotherroof.appear(begin_time=t0, transition_time=1)

        anotherroof.move(direction=[0,0,5.5],begin_time=t0,transition_time=1)
        t0 = 0.5 + anotherroof.rescale(rescale=0.7,begin_time=t0,transition_time=1)
        self.t0 = t0

    def result(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 52 / 180 * pi])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=0.6, type='BLOOM', size=1)

        colors = flatten([["text"]*2,["joker"],["text"]])
        w_line = SimpleTexBObject(r"w=2^{p_2}\cdot 3^{p_3}\cdot 5^{p_5}\cdot \dots",text_size="Large",aligned="left",location=[-10,0,1.5],color=colors)
        h_line = SimpleTexBObject(r"h=2^{q_2}\cdot 3^{q_3}\cdot 5^{q_5}\cdot \dots",text_size="Large",aligned="left",location=[-10,0,0],color=colors)
        colors = flatten([["text"] * 10,["custom1"], ["joker"],["important"],["custom1"],["joker"],["important"], ["text"]*9,["custom1"], ["joker"],["important"],["text"]])


        result = SimpleTexBObject(r"f(w,h)=\left\{\begin{array}{l r}\text{UR}, & p_2=q_2\\\text{UL}, & p_2<q_2\\\text{BR}, & p_2>q_2\end{array}\right.",
                         text_size="Huge",aligned="left",location=[-10,0,-3],color=colors)
        t0 = 0.5 + w_line.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + h_line.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + result.write(letter_set=list(range(0,10)),begin_time=t0, transition_time=0.5)
        t0 = 0.5 + result.write(letter_set=list(range(10,32,3)),begin_time=t0, transition_time=0.5)
        t0 = 0.5 + result.write(letter_set=list(range(12,34,3)),begin_time=t0, transition_time=0.5)
        t0 = 0.5 + result.write(letter_set=list(range(11,33,3)),begin_time=t0, transition_time=0.5)
        self.t0 =t0

    def logo(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 52/180*pi])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=0.6, type='BLOOM', size=1)

        kwargs = {"rotation_euler":[-pi/2,7/180*pi,0]}
        kwargs_red = kwargs|{"color":"red","solid":True,"number":random_prime}
        kwargs_green=kwargs|{"color":"green","solid":False,"number":random_prime}
        kwargs_blue=kwargs|{"color":"blue","solid":False,"number":random_prime}
        logo = LogoFromInstances(instance=BilliardBallReal, rotation_euler=[pi / 2, 0, 0],
                                 scale=[5] * 3, location=[-4, 0, -5],
                                 details=15,kwargs_blue=kwargs_blue,kwargs_green=kwargs_green,kwargs_red=kwargs_red)
        t0 = 0.5 + logo.grow(begin_time=t0, transition_time=2)

        some = SoME(location = [6.25,0,0],scale=2)
        t0 = 0.5 + some.appear(begin_time=t0, transition_time=2)

        some.move_to(target_location=[-4,0,-2.5],begin_time=t0,transition_time=2)
        t0  = 2 + some.rescale(rescale=0.5,begin_time=t0,transition_time=2)

        lines = [
            Text(r"\text{\ding{43} Greatest Common Divisor}",location=[2,0,1.5],
                 text_size="Large",color="example",aligned="left"),
            Text(r"\text{\ding{43} Least Common Multiple}",location=[2,0,0],
                 text_size="Large",color="example",aligned="left"),
            Text(r"\text{\ding{43} Prime Factorization}",location=[2,0,-1.5],
                 text_size="Large",color="example",aligned="left")
        ]

        for line in lines:
            t0 = 0.5 + line.write(begin_time=t0, transition_time=1)

        t0 = 1+t0

        title = Text(r"\text{Billiard and Number Theory}", text_size="huge", color="example",
                     outline_color="joker", location=[-1.8, 0, 4.7])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1)

        self.t0 = t0

    def pattern3(self):
        t0 =  0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        shift = 5
        x_axis = NumberLine2(length=9, location=[-4.5 - shift, 0, -5.25], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{Width}[w]", tip_length=0)

        y_axis = NumberLine2(length=9, location=[-5.25 - shift, 0, -4.5], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.1, 0, 0],
                             label_position="right",
                             axis_label_location=[-1.15, 0, 9.5],
                             axis_label=r"\text{Height}[h]", tip_length=0, direction="VERTICAL")
        y_axis.grow(begin_time=t0, transition_time=0)
        x_axis.grow(begin_time=t0, transition_time=0)

        score_table = ScoreTable(width=10, height=10, background_material="background",
                                 ur_material="custom1", ul_material="joker", br_material="important",
                                 font_material="text", location=[-shift, 0, 0])
        t0 = 0.5 + score_table.appear(begin_time=t0, transition_time=0)

        entries =[
            [
            SimpleTexBObject(r"\text{Width}"),
                SimpleTexBObject(r"\text{Height}"),
                SimpleTexBObject(r"\text{Pocket}")
        ]
            ,
            [
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{UR}",color="custom1"),
            ]
            ,
            [
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{even}"),
                SimpleTexBObject(r"\text{UL}", color="joker"),
            ]
            ,
            [
                SimpleTexBObject(r"\text{even}"),
                SimpleTexBObject(r"\text{odd}"),
                SimpleTexBObject(r"\text{BR}", color="important"),
            ]
            ,
            [
                SimpleTexBObject(r"\text{even}"),
                SimpleTexBObject(r"\text{even}"),
                SimpleTexBObject(r"\text{???}",color=["custom1", "joker", "important"])
            ]
        ]

        table = Table(entries,location=[6,0,2],scale=2)
        t0 =  0.5 + table.write_row(0,begin_time=t0,transition_time=1,verbose=True)

        for row in range(1,11,2):
            for col in range(1,11,2):
                t0  = score_table.single_turn_on([row,col], begin_time=t0 )+0.1

        t0  = 1 + table.write_row(1,begin_time=t0 -0.5,transition_time=1,verbose=True)

        for col in range(1,11,2):
            for row in range(1,11,2):
                t0  = score_table.single_turn_off([row,col], begin_time=t0 )+0.05

        t0 +=0.5

        for row in range(2,11,2):
            for col in range(1,11,2):
                t0  = score_table.single_turn_on([col,row], begin_time=t0 )+0.1

        t0 = 1 + table.write_row(2, begin_time=t0 - 0.5, transition_time=1, verbose=True)

        for col in range(1,11,2):
            for row in range(2, 11, 2):
                t0  = score_table.single_turn_off([col,row], begin_time=t0 )+0.05

        t0 +=0.5

        for row in range(1, 11, 2):
            for col in range(2, 11, 2):
                t0 = score_table.single_turn_on([col, row], begin_time=t0) + 0.1

        t0 = 1 + table.write_row(3, begin_time=t0 - 0.5, transition_time=1, verbose=True)


        for col in range(2, 11, 2):
            for row in range(1, 11, 2):
                t0 = score_table.single_turn_off([col, row], begin_time=t0) + 0.05

        t0 += 0.5

        for row in range(2, 11, 2):
            for col in range(2, 11, 2):
                t0 = score_table.single_turn_on([col, row], begin_time=t0) + 0.1

        t0 = 1 + table.write_row(4, begin_time=t0 - 0.5, transition_time=1)

        lines = [
            SimpleTexBObject(r"f\left(10\over 6\right) = f\left(5\over 3\right)\rightarrow \text{UR}",color=flatten([["text"]*15,["custom1"]*2]),text_size="large",location=[3.75,0,-2.5]),
            SimpleTexBObject(r"f\left(6\over 8\right) = f\left(3\over 4\right)\rightarrow \text{UL}",color=flatten([["text"]*14,["joker"]*2]),text_size="large",location=[3.75,0,-4]),
            SimpleTexBObject(r"f\left(8\over 10\right) = f\left(4\over 5\right)\rightarrow \text{BR}",color=flatten([["text"]*15,["important"]*2]),text_size="large",location=[3.75,0,-5.5]),
        ]

        box = BQuadrilateral(vertices=[[-1,0,0],[-1,0,1],[0,0,1],[0,0,0]],color="example",thickness=3)

        for i,line in enumerate(lines):
            if i ==0:
                t0 = 0.5 + box.grow(begin_time=t0, transition_time=1)
            elif i==1:
                t0 = 0.5 + box.move(direction=[-4,0,2],begin_time=t0,transition_time=1)
            else:
                t0 = 0.5 + box.move(direction=[2,0,2],begin_time=t0,transition_time=1)
            t0 = 0.5 + line.write(begin_time=t0,transition_time=0.5)

        t0 = t0+0.5
        t0 = box.disappear(begin_time=t0, transition_time=0.5)

        for col in range(2, 11, 2):
            for row in range(2, 11, 2):
                t0 = score_table.single_turn_off([col, row], begin_time=t0) + 0.05

        [line.disappear(begin_time=t0+0.1*i, transition_time=0.5) for i,line in enumerate(lines)]
        t0 = 0.5 + table.disappear(begin_time=t0-0.5, transition_time=1)
        # score_table.disappear(begin_time=t0-0.5, transition_time=1)
        # x_axis.disappear(begin_time=t0-0.5, transition_time=1)
        # t0  = 0.5 + y_axis.disappear(begin_time=t0-0.5, transition_time=1)

        self.t0 = t0

    def pattern2(self):
        t0 =  0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        shift = 5
        x_axis = NumberLine2(length=9, location=[-4.5-shift, 0, -5.25], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{Width}[w]", tip_length=0)

        y_axis = NumberLine2(length=9, location=[-5.25-shift, 0, -4.5], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.1, 0, 0],
                             label_position="right",
                             axis_label_location=[-1.15, 0, 9.5],
                             axis_label=r"\text{Height}[h]", tip_length=0, direction="VERTICAL")
        y_axis.grow(begin_time=t0, transition_time=0)
        x_axis.grow(begin_time=t0, transition_time=0)

        score_table = ScoreTable(width=10, height=10, background_material="background",
                                 ur_material="custom1", ul_material="joker", br_material="important",
                                 font_material="text",location = [-shift,0,0])
        t0 = score_table.appear(begin_time=t0, transition_time=0)

        first_col = [[1, j] for j in range(1, 11)]

        papers = []
        balls=[]
        for i in range(10):
            paper = ReflectableBilliardPaper(width=1, height=i+1,
                                             paper_color="billiards_cloth_material",
                                             target_square_scale=0.33)
            paper.move_to(target_location=[0.5+1.1*i,0,-4.5 + 0.5 * i ],begin_time=t0,transition_time=0)
            paper.rotate(rotation_euler=[pi/2,0,0],begin_time=t0,transition_time=0)
            paper.appear(begin_time=t0, transition_time=1)
            papers.append(paper)
            ball = BilliardsBall(width=1, height=1+i, start_time=t0 + 1.5, grid_radius_ratio=1, radius=0.25,
                                 ball_material="billiards_ball_material", speed=0.49999, trace_material="custom1",
                                 name="BilliardBall")
            ball.appear(begin_time=t0, transition_time=1)
            set_parent(ball,paper)
            ball.ball_disappear(begin_time=t0+1.5+2*(i+1),transition_time=1)

            balls.append(ball)

        t0 = t0 +2.5

        for cell in first_col:
            score_table.turn_on([cell], begin_time=t0 , transition_time=1)
            t0+=2

        t0 =  t0 + 1.5
        for i,paper in enumerate(papers):
            paper.change_alpha(alpha=0,begin_time=t0+0.1*i, transition_time=0.5) # only make paper transparent to make not change alpha of children
            balls[i].line_disappear(begin_time=t0+0.1*i, transition_time=0.5)
            score_table.single_turn_off(first_col[i],begin_time=t0+0.1*i-0.5)

        t0 = t0+2
        self.t0 = t0

    def pattern(self):
        t0 =  0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        shift = 5
        x_axis = NumberLine2(length=9, location=[-4.5-shift, 0, -5.25], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{Width}[w]", tip_length=0)

        y_axis = NumberLine2(length=9, location=[-5.25-shift, 0, -4.5], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.1, 0, 0],
                             label_position="right",
                             axis_label_location=[-1.15, 0, 9.5],
                             axis_label=r"\text{Height}[h]", tip_length=0, direction="VERTICAL")
        y_axis.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + x_axis.grow(begin_time=t0, transition_time=1)

        score_table = ScoreTable(width=10, height=10, background_material="background",
                                 ur_material="custom1", ul_material="joker", br_material="important",
                                 font_material="text",location = [-shift,0,0])
        t0 = score_table.appear(begin_time=t0, transition_time=1, offset_for_slots=[0, 0, 0, 0.75])
        diagonal = [[i, i] for i in range(1, 11)]
        t0 = 0.5 + score_table.turn_on(diagonal, begin_time=t0, transition_time=1)

        t0  = 0.5 + score_table.turn_off(diagonal, begin_time=t0, transition_time=1)

        first_row = [[j, 1] for j in range(1, 11)]


        papers = []
        balls=[]
        for i in range(10):
            paper = ReflectableBilliardPaper(width=i+1, height=1,
                                             paper_color="billiards_cloth_material",
                                             target_square_scale=0.33)
            paper.move_to(target_location=[1.5+0.5*i,0,4.4-1.1*i],begin_time=t0,transition_time=0)
            paper.rotate(rotation_euler=[pi/2,0,0],begin_time=t0,transition_time=0)
            paper.appear(begin_time=t0, transition_time=1)
            papers.append(paper)
            ball = BilliardsBall(width=i+1, height=1, start_time=t0 + 1.5, grid_radius_ratio=1, radius=0.25,
                                 ball_material="billiards_ball_material", speed=0.49999, trace_material="custom1",
                                 name="BilliardBall")
            ball.appear(begin_time=t0, transition_time=1)
            set_parent(ball,paper)
            ball.ball_disappear(begin_time=t0+1.5+2*(i+1),transition_time=1)

            balls.append(ball)

        t0 = t0 +2.5

        for cell in first_row:
            score_table.turn_on([cell], begin_time=t0 , transition_time=1)
            t0+=2

        t0 =  t0 + 1.5
        for i,paper in enumerate(papers):
            paper.change_alpha(alpha=0,begin_time=t0+0.1*i, transition_time=0.5) # only make paper transparent to make not change alpha of children
            balls[i].line_disappear(begin_time=t0+0.1*i, transition_time=0.5)
            score_table.single_turn_off(first_row[i],begin_time=t0+0.1*i-0.5)

        t0 = t0+2
        self.t0 = t0

    def mirroring(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        create_glow_composition(threshold=1, type='BLOOM', size=4)

        camera_empty = EmptyCube(location=Vector([4.7,6, 0]))
        camera_location = [4.7,6,23]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=30)

        # 5x3
        paper = ReflectableBilliardPaper(width=5,height=3,paper_color="billiards_cloth_material")
        t0 = paper.appear(begin_time=t0,transition_time=1)

        ball = BilliardsBall(width=5, height=3 , start_time=t0 + 1.5,grid_radius_ratio=1,radius=0.25,
                             ball_material="billiards_ball_material", speed=1, trace_material="custom1",name="BilliardBall")
        t0  = 0.5 + ball.appear(begin_time=t0, transition_time=1)

        balls = [ball]
        reflection_frames = [330,450,512,692,751,873]
        initial_positions = [[3,0,0.25],[0,2,0.25],[1,0,0.25],[4,0,0.25],[0,1,0.25],[2,0,0.25]] # positions inside the pool-table
        location=[[0,3,0],[5,3,0],[5,6,0],[5,9,0],[10,9,0],[10,12,0]] # locations in the world
        for i,f in enumerate(reflection_frames):
            ball= BilliardsBall(width=5,height=3,start_time=f/FRAME_RATE,grid_radius_ratio=1,radius=0.25,initial_position=initial_positions[i],location=location[i],
                                ball_material="billiards_ball_material", speed=1, trace_material="custom1",name="BilliardBall"+str(i))
            balls.append(ball)
            ball.appear(begin_time=f/FRAME_RATE,transition_time=0)
            paper.unfold(begin_time=f/FRAME_RATE-1.2,transition_time=1)

        [ball.ball_disappear(begin_time=1160/FRAME_RATE,transition_time=1) for ball in balls]
        t0  = 1160/FRAME_RATE +1.5

        # make frame
        lcm_frame = BQuadrilateral(vertices=[[-2.5,-1.5,0],[12.5,-1.5,0],[12.5,13.5,0],[-2.5,13.5,0]],
                                   color="text", name="LCM_Frame")
        t0 = 0.5 + lcm_frame.grow(begin_time=t0, transition_time=1)
        lcm_text = Text(r"\text{lcm}(w,h)=3\times 5 =15",color="important",location=[2,0,-1],
                        text_size="huge",emission=0,emission_outline=0,rotation_euler=[-pi/2,0,0],
                        aligned="left")
        lcm_text2 = Text(r"\text{lcm}(w,h)=5\times 3 =15", color="important", location=[6,0,1.75],
                        text_size="Large", emission=0, emission_outline=0, rotation_euler=[-pi / 2, 0, pi/2],
                        aligned="left")
        lcm_text2.write(begin_time=t0,transition_time=0.5)
        t0 = lcm_text.write(begin_time=t0,transition_time=0.5)

        movers = BObject(children=[paper]+balls+[lcm_frame,lcm_text,lcm_text2])
        movers.appear(begin_time=t0,transition_time=0)
        movers.rescale(rescale=0.75,begin_time=t0,transition_time=1)
        t0 =  0.5 + movers.move(direction=[-7,3,0],begin_time=t0,transition_time=1)

        t0 = 24.33333333
        # second scenario
        # 4x3
        paper = ReflectableBilliardPaper(width=4, height=3, paper_color="billiards_cloth_material")
        paper.move(direction=[5,0,0],begin_time=t0,transition_time=0)
        t0 = paper.appear(begin_time=t0, transition_time=1)

        ball = BilliardsBall(width=4, height=3, start_time=t0 + 1.5, grid_radius_ratio=1, radius=0.25,
                             ball_material="billiards_ball_material", speed=1, trace_material="custom1",
                             name="BilliardBall",location=[5,0,0])
        t0 = 0.5 + ball.appear(begin_time=t0, transition_time=1)

        balls = [ball]
        reflection_frames = [1790, 1850, 1971, 2091, 2151]
        initial_positions = [
            [3, 0, 0.25],
            [0, 1, 0.25],
            [2, 0, 0.25],
            [0, 2, 0.25],
            [1, 0, 0.25]
        ]  # positions inside the pool-table
        location = [
            [5, 3, 0],
            [9, 3, 0],
            [9, 6, 0],
            [13, 6, 0],
            [13, 9, 0]
        ]  # locations in the world
        for i, f in enumerate(reflection_frames):
            ball = BilliardsBall(width=4, height=3, start_time=f / FRAME_RATE, grid_radius_ratio=1, radius=0.25,
                                 initial_position=initial_positions[i], location=location[i],
                                 ball_material="billiards_ball_material", speed=1, trace_material="custom1",
                                 name="BilliardBall" + str(i))
            balls.append(ball)
            ball.appear(begin_time=f / FRAME_RATE, transition_time=0)
            paper.unfold(begin_time=f / FRAME_RATE - 1.2, transition_time=1)

        [ball.ball_disappear(begin_time=2360 / FRAME_RATE, transition_time=1) for ball in balls]
        t0 = 2360 / FRAME_RATE + 1.5

        # make frame
        lcm_frame = BQuadrilateral(vertices=[[3, -1.5, 0], [15, -1.5, 0], [15, 10.5, 0], [3, 10.5, 0]],
                                   color="text", name="LCM_Frame")
        t0 = 0.5 + lcm_frame.grow(begin_time=t0, transition_time=1)
        lcm_text = Text(r"\text{lcm}(w,h)=3\times 4 =12", color="important", location=[7.5, 0, -1],
                        text_size="Large", emission=0, emission_outline=0, rotation_euler=[-pi / 2, 0, 0],
                        aligned="left")
        lcm_text2 = Text(r"\text{lcm}(w,h)=4\times 3 =12", color="important", location=[5, 0, -3.5],
                         text_size="large", emission=0, emission_outline=0, rotation_euler=[-pi / 2, 0, pi / 2],
                         aligned="left")
        lcm_text2.write(begin_time=t0, transition_time=0.5)
        t0 = lcm_text.write(begin_time=t0, transition_time=0.5)

        movers = BObject(children=[paper] + balls + [lcm_frame, lcm_text, lcm_text2])
        movers.appear(begin_time=t0, transition_time=0)
        movers.rescale(rescale=0.75, begin_time=t0, transition_time=1)
        t0 = 0.5 + movers.move(direction=[7, 5.5, 0], begin_time=t0, transition_time=1)

        t0 = 44.333333
        # third scenario
        # 3x2
        paper = ReflectableBilliardPaper(width=3, height=2, paper_color="billiards_cloth_material",
                                         target_square_scale=0.5)
        paper.move(direction=[4.3, 0, 0], begin_time=t0, transition_time=0)
        t0 = paper.appear(begin_time=t0, transition_time=1)

        ball = BilliardsBall(width=3, height=2, start_time=t0 + 1.5, grid_radius_ratio=1, radius=0.25,
                             ball_material="billiards_ball_material", speed=0.75, trace_material="custom1",
                             name="BilliardBall", location=[4.3, 0, 0])
        t0 = 0.5 + ball.appear(begin_time=t0, transition_time=1)

        balls = [ball]
        reflection_frames = [ 2970, 3049, 3132]
        initial_positions = [
            [2, 0, 0.25],
            [0, 1, 0.25],
            [1, 0, 0.25]
        ]  # positions inside the pool-table
        location = [
            [4.25, 2, 0],
            [7.25, 2, 0],
            [7.25, 4, 0]
        ]  # locations in the world
        for i, f in enumerate(reflection_frames):
            ball = BilliardsBall(width=3, height=2, start_time=f / FRAME_RATE, grid_radius_ratio=1, radius=0.25,
                                 initial_position=initial_positions[i], location=location[i],
                                 ball_material="billiards_ball_material", speed=0.75, trace_material="custom1",
                                 name="BilliardBall" + str(i))
            balls.append(ball)
            ball.appear(begin_time=f / FRAME_RATE, transition_time=0)
            paper.unfold(begin_time=f / FRAME_RATE - 1.2, transition_time=1)

        [ball.ball_disappear(begin_time=3260 / FRAME_RATE, transition_time=1) for ball in balls]
        t0 = 3260 / FRAME_RATE + 1.5

        # make frame
        lcm_frame = BQuadrilateral(vertices=[[2.8, -1, 0], [8.8, -1, 0], [8.8, 5, 0], [2.8, 5, 0]],
                                   color="text", name="LCM_Frame")
        t0 = 0.5 + lcm_frame.grow(begin_time=t0, transition_time=1)
        lcm_text = Text(r"2\times 3 =6", color="important", location=[5.6, 0, -0.5],
                        text_size="Large", emission=0, emission_outline=0, rotation_euler=[-pi / 2, 0, 0],
                        aligned="left")
        lcm_text2 = Text(r"3\times 2 =6", color="important", location=[2.7, 0, -3.05],
                         text_size="large", emission=0, emission_outline=0, rotation_euler=[-pi / 2, 0, pi / 2],
                         aligned="left")
        lcm_text2.write(begin_time=t0, transition_time=0.5)
        t0 = lcm_text.write(begin_time=t0, transition_time=0.5)

        self.t0  = t0

    def reflexion(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        create_glow_composition(threshold=1, type='BLOOM', size=4)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -18, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        result = SimpleTexBObject(r"f\left({ w\over h}\right)\rightarrow f\left({h\over w}\right)",
                                  location=[0, 0, 3.5],
                                  text_size='Large', color="text",
                                  aligned="center", emission=0)
        t0 = 0.5 + result.write(begin_time=0, transition_time=0.5)

        paper1 = BilliardPaper(width=4,height=3,rotation_quaternion=Quaternion([1,1,0,0]),location=[-7,0,-4],
                               paper_color="billiards_cloth_material",pivot="LOWER_LEFT")
        t0 = 0.5 + paper1.grow(begin_time=t0,transition_time=1)

        paper2 = BilliardPaper(width=5, height=3, rotation_quaternion=Quaternion([1,1,0, 0]),
                               paper_color="billiards_cloth_material",location=[-2.5, 0, -4])
        t0 = 0.5 + paper2.grow(begin_time=t0, transition_time=1)

        paper3 = BilliardPaper(width=3, height=4, rotation_quaternion=Quaternion([1,1, 0, 0]),
                               paper_color="billiards_cloth_material",location=[3, 0, -4])
        t0 = 0.5 + paper3.grow(begin_time=t0, transition_time=1)

        paper1.show_ball(begin_time=t0,transition_time=1)
        paper3.show_ball(begin_time=t0,transition_time=1)
        t0 = 0.5 + paper2.show_ball(begin_time=t0,transition_time=1)

        t0 = 22
        papers = [paper1, paper2, paper3]
        times = [19.33,21.5,19.33]
        [paper.disappear_ball(begin_time=time, transition_time=0.5) for paper,time in zip(papers,times)]
        t0  = t0 + 1



        # reflect first paper
        square = Plane(location=[0, 0, 0], color="important", name="BRSquare")
        set_parent(square, papers[0])
        square.move_to(target_location=[3.5, 0.5, 0.01], begin_time=t0, transition_time=0)
        square.grow(begin_time=t0, transition_time=0.3, scale=0.5)
        br = Text(r"\text{BR}", text_size="huge", color="important", location=[-7.1, 0, 1], aligned="center",
                  emission_outline=0, emission=0)
        t0 = 0.5 + br.write(begin_time=t0, transition_time=0.3)

        r2 = 1/np.sqrt(2)
        t0 = 0.5 + paper1.rotate(rotation_quaternion=Quaternion([1,1,0,0])@Quaternion([0,r2,r2,0]),begin_time=t0,transition_time=1)

        square = Plane(location=[0, 0, 0], color="joker", name="ULSquare")
        set_parent(square, papers[0])
        square.move_to(target_location=[3.5,0.5,-0.01], begin_time=t0, transition_time=0)
        square.grow(begin_time=t0, transition_time=0.3, scale=0.475)
        ul = Text(r"\text{UL}", text_size="huge", color="joker", location=[-4.5, 0, 1], aligned="center",
                  emission_outline=0, emission=0)
        arrow = Text(r"\rightarrow", text_size="huge",color="text",location=[-5.95,0,1],aligned="center",
                     emission_outline=0, emission=0)
        t0  = arrow.write(begin_time=t0, transition_time=0.1)
        t0 = 0.5 + ul.write(begin_time=t0, transition_time=0.3)

        # reflect second paper
        square = Plane(location=[0, 0, 0], color="custom1", name="URSquare")
        set_parent(square, papers[1])
        square.move_to(target_location=[4.5, 2.5, 0.01], begin_time=t0, transition_time=0)
        square.grow(begin_time=t0, transition_time=0.3, scale=0.5)
        ur = Text(r"\text{UR}", text_size="huge", color="custom1", location=[-2.7, 0, 2], aligned="center",
                  emission_outline=0, emission=0)
        t0 = 0.5 + ur.write(begin_time=t0, transition_time=0.3)

        t0 = 0.5 + paper2.rotate(rotation_quaternion=Quaternion([1, 1, 0, 0]) @ Quaternion([0, r2, r2, 0]),
                                 begin_time=t0, transition_time=1)

        square.move(direction=[0,0, -0.02], begin_time=t0, transition_time=0.1)
        ur_copy = ur.copy()
        ur_copy.appear(begin_time=t0, transition_time=0)
        ur_copy.move(direction=[2.6,0,0],begin_time=t0, transition_time=1)
        arrow = Text(r"\rightarrow", text_size="huge", color="text", location=[-1.55, 0, 2], aligned="center",
                     emission_outline=0, emission=0)
        t0 =0.5+ arrow.write(begin_time=t0, transition_time=0.1)

        # reflect third paper
        square = Plane(location=[0, 0, 0], color="joker", name="ULSquare2")
        set_parent(square, papers[2])
        square.move_to(target_location=[0.5, 3.5, 0.01], begin_time=t0, transition_time=0)
        square.grow(begin_time=t0, transition_time=0.3, scale=0.5)
        ul = Text(r"\text{UL}", text_size="huge", color="joker", location=[3.25, 0, 0.5], aligned="center",
                  emission_outline=0, emission=0)
        t0 = 0.5 + ul.write(begin_time=t0, transition_time=0.3)

        t0 = 0.5 + paper3.rotate(rotation_quaternion=Quaternion([1, 1, 0, 0]) @ Quaternion([0, r2, r2, 0]),
                                 begin_time=t0, transition_time=1)

        square = Plane(location=[0, 0, 0], color="important", name="BRSquare2")
        set_parent(square, papers[2])
        square.move_to(target_location=[0.5, 3.5, -0.01], begin_time=t0, transition_time=0)
        square.grow(begin_time=t0, transition_time=0.3, scale=0.475)
        br = Text(r"\text{BR}", text_size="huge", color="important", location=[6.05, 0, 0.5], aligned="center",
                  emission_outline=0, emission=0)
        arrow = Text(r"\rightarrow", text_size="huge", color="text", location=[4.5, 0, 0.5], aligned="center",
                     emission_outline=0, emission=0)
        t0 = arrow.write(begin_time=t0, transition_time=0.1)
        t0 = 0.5 + br.write(begin_time=t0, transition_time=0.3)
        self.t0 = t0

    def reducible(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        create_glow_composition(threshold=1, type='BLOOM', size=4)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -33, 25]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        one = -7.5
        two = -8.5
        lines = [
            SimpleTexBObject("w=2",color=["text","text","example"],location=[-15.4,-4.5,one],text_size='Large',aligned="left",emission=0),
            SimpleTexBObject("h=3",color=["text","text","example"],location=[-15.5,-4.5,two],text_size='Large',aligned="left",emission=0),
            SimpleTexBObject("w=4=2\cdot 2",color=[*(["text"]*4),"drawing","text","example"],location=[-5.5,-4.5,one],text_size='Large',aligned="left",emission=0),
            SimpleTexBObject("h=6=2\cdot 3",color=[*(["text"]*4),"drawing","text","example"],location=[-5.5,-4.5,two],text_size='Large',aligned="left",emission=0),
            SimpleTexBObject("w=6=3\cdot 2",color=[*(["text"]*4),"drawing","text","example"],location=[8.5,-4.5,one],text_size='Large',aligned="left",emission=0),
            SimpleTexBObject("h=9=3\cdot 3",color=[*(["text"]*4),"drawing","text","example"],location=[8.77,-4.5,two],text_size='Large',aligned="left",emission=0),
        ]
        [line.rotate(rotation_euler=[ibpy.camera_alignment_euler(line,camera_location)[0],0,0] ,begin_time=t0, transition_time=0) for line in lines]

        location = [[-14, 0, -5], [-4, 3, -5], [10, 6, -5]]
        scale = [0.5, 0.5, 0.5]
        tables = []
        for i in range(3):
            table = BilliardsTable(width=2*(i+1), height=3*(i+1), location=location[i], scale=scale[i])
            t0 = table.appear(begin_time=t0, transition_time=1, offset_for_slots=[0, 0, 0, 1])
            lines[2*i].write(letter_set=[0,1,2],begin_time=t0,transition_time=0.1)
            lines[2*i+1].write(letter_set=[0,1,2],begin_time=t0,transition_time=0.1)
            t0 += 0.5
            tables.append(table)

        balls = []
        speed=[0.5,1,1.5]
        for i in range(3):
            ball = BilliardsBall(width=2*(i+1), height=3*(i+1), start_time=t0 + 1.5,
                                 ball_material="billiards_ball_material", speed=speed[i],trace_material="custom1")
            ball.appear(begin_time=t0, transition_time=1)
            set_parent(ball, tables[i])
            balls.append(ball)

        t0 = t0 + 14

        for i in range(1,3):
            lines[2 * i].write(letter_set=[3,4,5,6], begin_time=t0, transition_time=0.1)
            t0  = 0.5+lines[2 * i + 1].write(letter_set=[3,4,5,6], begin_time=t0, transition_time=0.1)

        three = -10
        gcds = [
            SimpleTexBObject(r"\gcd(2,3)=1",color="drawing",location=[-17.7,-4.5,three],text_size='Large',aligned="left",emission=0),
            SimpleTexBObject(r"\gcd(4,6)=2",color="drawing",location=[-6.48,-4.5,three],text_size='Large',aligned="left",emission=0),
            SimpleTexBObject(r"\gcd(6,9)=3",color="drawing",location=[8.1,-4.5,three],text_size='Large',aligned="left",emission=0),
        ]
        [line.rotate(rotation_euler=[ibpy.camera_alignment_euler(line, camera_location)[0], 0, 0], begin_time=t0,
                     transition_time=0) for line in gcds]

        for i in range(3):
            balls[i].ball_disappear(begin_time=t0, transition_time=0.25)
            t0 = gcds[i].write(begin_time=t0,transition_time=0.25)

        for i in range(3):
            cyl = Cylinder.from_start_to_end(start=[-4*(i+1),-2*(i+1),0],end=[0,-2*(i+1),0],radius=0.3,color="drawing",name="GCD"+str(i+1))
            t0 = 0.5 + cyl.grow(modus="from_start",begin_time=t0, transition_time=0.5*i+0.5)
            ibpy.set_parent(cyl, tables[i])


        result = SimpleTexBObject(r"f(q\cdot w,q\cdot h)=f(w,h)=f\left({w\over h}\right)",location=[0,0,8.3],text_size='Huge',color=[*["text"]*2,"drawing",*["text"]*3,"drawing","text"],aligned="center",emission=0)
        result.rotate(rotation_euler=[ibpy.camera_alignment_euler(result, camera_location)[0], 0, 0], begin_time=t0,transition_time=0)

        t0  = 0.5 + result.write(letter_set=list(range(17)),begin_time=t0,transition_time=1)
        t0  = 0.5 + result.write(letter_set=list(range(17,24)),begin_time=t0,transition_time=1)
        self.t0 = t0

    def diagonal(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        create_glow_composition(threshold=1, type='BLOOM', size=4)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -33, 25]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        location = [[-13,0,-5],[-2.5,0,-5],[10,0,-5]]
        scale = [0.5,0.5,0.5]
        tables = []
        for i in range(3):
            table = BilliardsTable(width=i+3, height=i+3, location=location[i],scale=scale[i])
            t0 = table.appear(begin_time=t0, transition_time=1, offset_for_slots=[0, 0, 0, 1])
            tables.append(table)

        balls = []
        for i in range(3):
            ball = BilliardsBall(width=i+3, height=i+3, start_time=t0 + 1.5,ball_material="billiards_ball_material", trace_material="custom1")
            t0  = ball.appear(begin_time=t0,transition_time=1)
            set_parent(ball,tables[i])
            balls.append(ball)

        t0 = t0+7

        function = Text(r"f(m,m) = \hspace{2em} (m \in \mathbb{N})", text_size="Huge", color="example",
                        location=[1,0, 5],aligned="center", emission_outline=0, emission=0)
        function.rotate(rotation_euler=[ibpy.camera_alignment_euler(function, camera_location)[0]-pi/2, 0, 0], begin_time=t0,
                      transition_time=0)

        rotation = ibpy.camera_alignment_euler(function, camera_location)[0]
        t0 = 0.1 + function.write(begin_time=t0, transition_time=2)
        ur = Text(r"\text{UR}", text_size="Huge", color="custom1", location=[0, 0, 5],rotation_euler=[rotation-pi/2,0,0], aligned="left",
                  emission_outline=0, emission=0)

        t0 = 0.5 + ur.write(begin_time=t0-1.25, transition_time=0.3)

        self.t0 = t0

    def introduction(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        create_glow_composition(threshold=1, type='BLOOM', size=4)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -33, 25]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        width = 3
        height= 4
        location = [-5.5,0,0]
        papers = []

        table = BilliardsTable(width=width,height=height,location=location)
        t0  = 0.5 + table.appear(begin_time=t0,transition_time=1,offset_for_slots=[0,0,0,1])

        paper = Plane(u=[-2, 2], v=[-2.5, 2.5], color="gray_1", name="Paper3x4")
        papers.append(paper)
        paper.rotate(rotation_euler=ibpy.camera_alignment_euler(paper, camera_location), begin_time=t0,
                     transition_time=0)
        paper.move_to(target_location=[12, 0, 0], begin_time=t0, transition_time=0)
        t0  = 0.5  + paper.grow(begin_time=t0, transition_time=1, scale=3)

        lines = [
            (-1.5,-2,1.5,-2),
            (-1.5,-1,1.5,-1),
            (-1.5,0,1.5,0),
            (-1.5,1,1.5,1),
            (-1.5,2,1.5,2),
            (-1.5,2,-1.5,-2),
            (-0.5,2,-0.5,-2),
            (0.5,2,0.5,-2),
            (1.5,2,1.5,-2)
        ]
        cylinders = []
        for line in lines:
            cylinders.append(Cylinder.from_start_to_end(
                start=Vector([line[0],line[1],0]),end=Vector([line[2],line[3],0]),radius=0.01))
            cylinders[-1].grow(begin_time=t0,transition_time=1)
            ibpy.set_parent(cylinders[-1],paper)
            t0 +=0.1

        ball = BilliardsBall(width=width,height=height,start_time=t0+1.5,speed = 4,
                             ball_material="billiards_ball_material",trace_material="custom1",location=location)

        ball2 = BilliardsBall(width=width, height=height, start_time=t0 + 1.5,grid_radius_ratio=1,radius=0.333,
                             ball_material="billiards_ball_material", trace_material="custom1", location=[0,0,0])

        set_parent(ball2,paper)

        ball2.appear(begin_time=t0, transition_time=1)
        t0 = 0.5+ball.appear(begin_time=t0,transition_time=1)

        t0 = 19
        ball.disappear(begin_time=t0,transition_time=1)
        ball2.ball_disappear(begin_time=t0,transition_time=1)
        t0 =0.5 + table.disappear(begin_time=t0,transition_time=1)

        # two more examples
        width = 4
        height = 3
        tables = []
        table_balls = []
        location=[-8,0,6]
        paper = Plane(u=[-2.2, 2.2], v=[-2.75, 2.75], color="gray_1",name="Paper4x3")
        papers.append(paper)
        paper.rotate(rotation_euler=ibpy.camera_alignment_euler(paper, camera_location), begin_time=t0,
                     transition_time=0)
        paper.move_to(target_location=[1.5, 1, 5.5], begin_time=t0, transition_time=0)
        paper.grow(begin_time=t0, transition_time=1, scale=1.5)

        table = BilliardsTable(width=width, height=height, location=location,grid_radius_ratio=4,rotation_euler=[pi/2/18,0,0])
        table.rescale(rescale=0.5,begin_time=t0,transition_time=0)
        tables.append(table)
        t0 = table.appear(begin_time=t0, transition_time=1, offset_for_slots=[0, 0, 0, 1])

        lines = [
            (-1.5, -2, 1.5, -2),
            (-1.5, -1, 1.5, -1),
            (-1.5, 0, 1.5, 0),
            (-1.5, 1, 1.5, 1),
            (-1.5, 2, 1.5, 2),
            (-1.5, 2, -1.5, -2),
            (-0.5, 2, -0.5, -2),
            (0.5, 2, 0.5, -2),
            (1.5, 2, 1.5, -2)
        ]
        cylinders = []
        for line in lines:
            cylinders.append(Cylinder.from_start_to_end(
                start=Vector([line[1], line[0], 0]), end=Vector([line[3], line[2], 0]), radius=0.01))
            cylinders[-1].grow(begin_time=t0, transition_time=1)
            ibpy.set_parent(cylinders[-1], paper)
            t0 += 0.1

        ball = BilliardsBall(width=width, height=height, start_time=t0 + 1.5,rotation_euler=[pi/36,0,0],
                             ball_material="billiards_ball_material", trace_material="custom1", location=location)
        ball.rescale(rescale=0.5,begin_time=t0,transition_time=0)
        table_balls.append(ball)

        ball2 = BilliardsBall(width=width, height=height, start_time=t0 + 1.5, grid_radius_ratio=1, radius=0.333,
                              ball_material="billiards_ball_material", trace_material="custom1", location=[0, 0, 0])

        set_parent(ball2, paper)

        ball2.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + ball.appear(begin_time=t0, transition_time=1)

        t0 = 37
        t0 = 0.5 + ball2.ball_disappear(begin_time=t0, transition_time=1)

        width = 3
        height = 5
        location = [-11.5, 0, -6]
        paper = Plane(u=[-2.2, 2.2], v=[-2.75, 2.75], color="gray_1", name="Paper3x5")
        papers.append(paper)

        paper.rotate(rotation_euler=ibpy.camera_alignment_euler(paper, camera_location), begin_time=t0,
                     transition_time=0)
        paper.move_to(target_location=[1.5,-4.5,-2.5], begin_time=t0, transition_time=0)
        paper.grow(begin_time=t0, transition_time=1, scale=1.5)

        table = BilliardsTable(width=width, height=height, location=location, grid_radius_ratio=4,
                               rotation_euler=[pi / 2 / 18, 0, 0])
        table.rescale(rescale=0.5, begin_time=t0, transition_time=0)
        tables.append(table)
        t0 = table.appear(begin_time=t0, transition_time=1, offset_for_slots=[0, 0, 0, 1])

        lines = [
            (-1.5, -2.5, 1.5, -2.5),
            (-1.5, -1.5, 1.5, -1.5),
            (-1.5, -0.5, 1.5, -0.5),
            (-1.5, 0.5, 1.5, 0.5),
            (-1.5, 1.5, 1.5, 1.5),
            (-1.5, 2.5, 1.5, 2.5),
            (-1.5, 2.5, -1.5, -2.5),
            (-0.5, 2.5, -0.5, -2.5),
            (0.5, 2.5, 0.5, -2.5),
            (1.5, 2.5, 1.5, -2.5)
        ]
        cylinders = []
        for line in lines:
            cylinders.append(Cylinder.from_start_to_end(
                start=Vector([line[0], line[1], 0]), end=Vector([line[2], line[3], 0]), radius=0.01))
            cylinders[-1].grow(begin_time=t0, transition_time=1)
            ibpy.set_parent(cylinders[-1], paper)
            t0 += 0.1

        ball = BilliardsBall(width=width, height=height, start_time=t0 + 1.5,rotation_euler=[pi/36,0,0],
                             ball_material="billiards_ball_material", trace_material="custom1", location=location)
        ball.rescale(rescale=0.5, begin_time=t0, transition_time=0)
        table_balls.append(ball)
        ball2 = BilliardsBall(width=width, height=height, start_time=t0 + 1.5, grid_radius_ratio=1, radius=0.333,
                              ball_material="billiards_ball_material", trace_material="custom1", location=[0, 0, 0])

        set_parent(ball2, paper)

        ball2.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + ball.appear(begin_time=t0, transition_time=1)

        t0 = 59
        t0 = 0.5 + ball2.ball_disappear(begin_time=t0, transition_time=1)

        # make tables disappear and arrange papers in a row
        [object.disappear(begin_time=t0,transition_time=1) for object in tables+table_balls]
        t0  = 1.5+t0

        duration = 3
        center = papers[2]
        center.rescale(rescale=1.81,begin_time=t0,transition_time=duration)
        center.move_to(target_location=[0,0,0],begin_time=t0,transition_time=duration)

        left = papers[1]
        left.rescale(rescale=1.81,begin_time=t0,transition_time=duration)

        [paper.rotate(rotation_euler=[pi/2,0,0],begin_time=t0,transition_time=duration) for paper in papers]
        ibpy.camera_move(shift=[0,-7,-25],begin_time=t0,transition_time=duration)
        t0 = left.move_to(target_location=[-12,0,0],begin_time=t0,transition_time=duration)

        camera_move(shift=[0,0,-2.5],begin_time=t0,transition_time=1)
        t0  = 0.5 + camera_empty.move(direction=[0,0,-2.5],begin_time=t0,transition_time=1)

        # function and squares

        function = Text(r"f(w,h) = \left\{\rule{0em}{5ex} \right.",text_size="huge",color="example",location =[-7,0,-10],emission_outline=0.2,emission=0)
        t0 = 0.5 + function.write(begin_time=t0,transition_time=1)

        square = Plane(location=[0,0,0],color="important",name="BRSquare")
        set_parent(square,papers[1])
        square.move_to(target_location=[1.5,-1,0],begin_time=t0,transition_time=0)
        square.grow(begin_time=t0,transition_time=0.3,scale=0.5)
        br = Text(r"\text{BR}",text_size="huge",color="important",location =[0,0,-8],aligned="center",emission_outline=0,emission=0)
        t0 = 0.5 + br.write(begin_time=t0,transition_time=0.3)

        square = Plane(location=[0, 0, 0], color="joker", name="ULSquare")
        set_parent(square, papers[0])
        square.move_to(target_location=[-1,1.5, 0], begin_time=t0, transition_time=0)
        square.grow(begin_time=t0, transition_time=0.3, scale=0.475)
        ul = Text(r"\text{UL}", text_size="huge", color="joker", location=[0, 0, -10], aligned="center",emission_outline=0,emission=0)
        t0 = 0.5 + ul.write(begin_time=t0, transition_time=0.3)

        square = Plane(location=[0, 0, 0], color="custom1", name="URSquare")
        set_parent(square, papers[2])
        square.move_to(target_location=[1, 2
            , 0], begin_time=t0, transition_time=0)
        square.grow(begin_time=t0, transition_time=0.3, scale=0.475)
        ur = Text(r"\text{UR}", text_size="huge", color="custom1", location=[0, 0, -12], aligned="center",emission_outline=0,emission=0)
        t0 = 0.5 + ur.write(begin_time=t0, transition_time=0.3)

        self.t0 = t0

    def score_table(self):
        t0 = 1.5

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -26, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        shift = 5
        x_axis = NumberLine2(length=9, location=[-4.5 - shift, 0.15, -5.25], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{Width}[w]", tip_length=0)

        y_axis = NumberLine2(length=9, location=[-5.25 - shift, 0.15, -4.5], domain=[1, 10], n_tics=9, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.1, 0, 0],
                             label_position="right",
                             axis_label_location=[-1.15, 0, 9.5],
                             axis_label=r"\text{Height}[h]", tip_length=0, direction="VERTICAL")
        y_axis.grow(begin_time=t0, transition_time=0)
        x_axis.grow(begin_time=t0, transition_time=0)

        score_table = ScoreTable(width=10, height=10, background_material="background",
                                 ur_material="custom1", ul_material="joker", br_material="important",
                                 font_material="text", location=[-shift, 0.15, 0])
        t0 = 0.5 + score_table.appear(begin_time=t0, transition_time=0)

        movers =[x_axis,y_axis,score_table]
        [mover.move(direction=(shift,-0.15,0),begin_time=t0,transition_time=1) for mover in movers]
        ibpy.camera_move(shift=[0,1,0],begin_time=t0,transition_time=1)

        t0+=1.5

        diagonal = [[i, i] for i in range(1, 11)]
        t0 = 0.5 + score_table.turn_on(diagonal, begin_time=t0, transition_time=1)

        zero_data = [[i, j] for i in range(1, 11) for j in range(1, 11) if f(i, j) == 0 and i!=j]
        zero_data.sort(key=lambda x: x[0]+x[1])
        one_data = [[i, j] for i in range(1, 11) for j in range(1, 11) if f(i, j) == 1]
        two_data = [[i, j] for j in range(1, 11) for i in range(1, 11) if f(i, j) == 2]

        t0 = 0.5 + score_table.turn_on(zero_data, begin_time=t0, transition_time=1)

        t0 = 0.5 + score_table.turn_off(zero_data+diagonal,begin_time=t0,transition_time=0.1)

        score_table.turn_on(one_data,begin_time=t0,transition_time=5)
        t0  = 0.5 + score_table.turn_on(two_data,begin_time=t0,transition_time=5)

        t0 = 0.5 + score_table.turn_on(diagonal+zero_data,begin_time=t0,transition_time=1)

        self.t0 = t0



if __name__ == '__main__':
    try:
        example = Billard()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        if len(dictionary) == 1:
            selection=0
        else:
            selection = input("Choose scene:")
            if len(selection) == 0:
                selection = 0
        print("Your choice: ", selection)
        selected_scene = dictionary[int(selection)]

        example.create(name=selected_scene, resolution=[1920,1080], start_at_zero=True)

    except:
        print_time_report()
        raise ()
