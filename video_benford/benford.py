import os
from collections import OrderedDict

import numpy as np
from scipy.constants import golden_ratio
from scipy.stats import arcsine

from appearance.textures import highlighting_for_material, box_highlighting_for_material
from compositions.compositions import create_glow_composition
from interface import ibpy
from interface.ibpy import Vector, set_parent, get_geometry_node_from_modifier, \
    change_default_value, camera_zoom, Quaternion
from interface.interface_constants import BLENDER_EEVEE, CYCLES
from geometry_nodes.geometry_nodes_modifier import PowerOfTwoModifier, FibonacciModifier
from objects.benford import BenfordDiagram, BenfordFibonacciDiagram, BenfordInterval, BenfordFilesDiagram
from objects.billiards_new_objects import BilliardsTableRound, BilliardBallRound
from objects.file_explorer import FileExplorer
from objects.functions import SimpleFunction
from objects.billiards_objects import BilliardBallReal
from objects.book import Book
from objects.codeparser import CodeParser
from objects.display import CodeDisplay
from objects.light.light import SpotLight
from objects.logo import LogoFromInstances
from objects.plane import Plane
from objects.text2 import Text, MorphText, MorphText2, MorphText3
from objects.arc import Arc2
from objects.bobject import BObject
from objects.circle import Circle2
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.empties import EmptyCube
from objects.geometry.sphere import Sphere
from objects.number_line import NumberLine2
from objects.table import Table, Table2
from objects.tex_bobject import SimpleTexBObject
from perform.scene import Scene
from utils.constants import FRAME_RATE, DATA_DIR, LOC_FILE_DIR
from utils.utils import print_time_report, flatten
from video_billiard.auxiliaries import random_prime

pi = np.pi
tau=2*pi


class Benford(Scene):
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('code_display',{'duration':30}),
            ('fib_theory',{'duration':10}),
            ('probabilities',{'duration':10}),
            ('powers_of_five',{'duration':12}),
            ('intervals',{'duration':int(212)}),
            ('title_movie',{'duration':5}),
            ('branding',{'duration':15}),
            ('logo_movie',{'duration':15}),
            ('intro_table',{'duration':15}),
            ('computations2',{'duration':16}),
            ('computations',{'duration':30}),
            ('logarithms',{'duration':17}),
            ('file_explorer',{'duration':62}),
            ('benford_explainer',{'duration':435}),
            ('billiard_irrational',{'duration':int(9999/FRAME_RATE)}),
            ('billiard_stars',{'duration':30}),
            ('billiard_deflection_angle',{'duration':30}),
            ('billiard_polygons',{'duration':22}),
            ('fibonacci',{'duration':37}),
            ('power_of_two',{'duration':35}),
            ('benford3',{'duration':95}),
            ('benford2',{'duration':95}),
            ('benford',{'duration':66}),

        ])
        super().__init__(light_energy=2, transparent=False)

    def code_display(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        filename = os.path.join(LOC_FILE_DIR, "benford_files.py")
        cp = CodeParser(filename, emission=0, recreate=False)

        display = CodeDisplay(cp, location=Vector([0, 0, 0]), scales=[10,6], number_of_lines=30,emission=0, flat=True)
        t0 = display.appear(begin_time=t0)

        t0 = 0.5 + cp.write(display, class_index=0, function=None, begin_time=t0, transition_time=10, indent=0.5)

        self.t0 = t0

    def fib_theory(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        theory = Text(r"\text{Fib}_n=\frac{\varphi^n}{\sqrt{5}}-\frac{(-\varphi)^{-n}}{\sqrt{5}}",
                      text_size="large",location=[0,0,0],aligned="left")

        t0 =0.5+ theory.write(begin_time=t0,transition_time=0.5)
        golden_ratio=Text(r"\varphi=\frac{1+\sqrt{5}}{2}",
                          text_size="large",location=[6,0,0],aligned="left")

        t0 = 0.5 + golden_ratio.write(begin_time=t0, transition_time=0.5)

        golden_ratio.move(direction=[-2.75,0,0],begin_time=t0,transition_time=0.5)
        t0 = 0.5 + theory.unwrite(letters=11,begin_time=t0, transition_time=0.5)

        theta = SimpleTexBObject(r"\theta = {\log_{10}\varphi \over 2\pi}", location=[0.9, 0,-2],
                color="example",text_size="Large",aligned="left")

        t0 = 0.5 + theta.write(begin_time=t0, transition_time=0.5)
        self.t0 = t0

    def probabilities(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        left = Text("p(d)=",aligned="right",text_size="large",location=[0,0,5])
        right = MorphText3(r"\log_{10}(d+1)-\log_{10} d",r"\log_{10}\tfrac{d+1}{d}",aligned="left",text_size="large",location=[0,0,5],src_fix=[0,1,2,3,4],target_fix=[0,1,2,3,4])
        t0 = left.write(begin_time=t0,transition_time=0.1)
        t0 = 0.5 + right.write(begin_time=t0,transition_time=0.5)
        t0 = 0.5 + right.morph(begin_time=t0,transition_time=0.25)

        numbers = [i for i in range(1, 10)]
        alg = [r"\log_{10}\tfrac{"+str(i+1)+"}{"+str(i)+"}" for i in range(1, 10)]
        probs = [round(np.log((i + 1)/i) / np.log(10) * 100, 1) for i in range(1, 10)]
        number_objects =[SimpleTexBObject(r"d",text_size="large",aligned="center",color="important")]+ [SimpleTexBObject(str(number), text_size="large", aligned="center",color="important") for number in numbers]
        alg_objects = [SimpleTexBObject(r"\phantom{d}\log_{10}\tfrac{d+1}{d}",color=flatten([["text"]*5,["important"]]),text_size="large",aligned="left")]+[SimpleTexBObject(alg[i], text_size="large", aligned="left") for i in range(len(alg))]
        prob_objects = [SimpleTexBObject(r"p(d)",text_size="large",aligned="center",color=["text","text","important","text"])]+[SimpleTexBObject(str(prob) + "\%", text_size="large", aligned="center") for prob in probs]
        data = np.array([number_objects,alg_objects, prob_objects])
        data=  data.transpose()

        table = Table(data, bufferx=1,buffery=0.4, location=[-1.15, 0, 0])
        t0 = table.write_row(0, begin_time=t0, transition_time=1)
        lines = []
        line = Cylinder.from_start_to_end(start=[-3, 0, 2.25], end=[3, 0, 2.25], radius=0.02)
        xs = [0.95, -1.7]
        for i in range(len(xs)):
            vert = Cylinder.from_start_to_end(start=[xs[i], 0, 3], end=[xs[i], 0, -5.5], radius=0.02)
            vert.grow(begin_time=t0 + i * 0.1, transition_time=0.1)
            lines.append(vert)

        t0 = 0.5 + line.grow(begin_time=t0, transition_time=1)
        for i in range(1,len(data)):
            t0 =  table.write_row(i, begin_time=t0, transition_time=0.25)

        self.t0 = t0


    def powers_of_five(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)


        for i in range(11):
            power = 5**i
            if i<10:
                text = SimpleTexBObject(r"5^{\phantom{1}"+str(i)+"}="+str(power),aligned="left",text_size="large",location=[0,0,-5+0.75*i])
            else:
                text = SimpleTexBObject(r"5^{"+str(i)+"}="+str(power),aligned="left",text_size="large",location=[0,0,-5+0.75*i])
            t0 = 0.5 + text.write(begin_time=t0,transition_time=0.5)

        self.t0 = t0

    def intervals(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -17, 17]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)


        floats = [1./(1+np.sqrt(5)),1./(1+np.sqrt(5))]
        balls = []
        tables = []
        for i in range(2):
            if i==0:
                table = BilliardsTableRound(radius=3, location=[-5 + 2*i * 5,-2, 0], scale=0.95)
                table.appear(begin_time=t0 + i * 0.5, transition_time=1)
            else:
                table = Sphere(r=0,location=[-5 + 2 * i * 5, -2, 0], scale=0.95)
                table.appear(begin_time=t0,transition_time=0)
            tables.append(table)
            ball = BilliardBallRound(ratio=floats[i], number=15, radius=7,  scale=0.5)
            balls.append(ball)
            set_parent(ball, table)

        t0 = t0 + 3

        title = SimpleTexBObject(r"\theta = {1 \over 1+\sqrt{5}}\pi", location=[-6, 0,5.5], rotation_euler=[pi / 4, 0, 0],
                             text_size="huge", color="example", aligned="center")

        title.write(begin_time=t0 , transition_time=0.5)
        t0 += 1


        balls[0].appear(begin_time=t0,transition_time=1)
        balls[0].start(begin_time=t0, transition_time=1039/2)
        balls[0].change_trace_thickness(from_value=1, to_value=0.05, begin_time=t0, transition_time=40)
        balls[1].change_trace_thickness(from_value=1, to_value=0.05, begin_time=t0, transition_time=40)

        t0 = t0 + 5

        for table in tables:
            table.rotate(rotation_euler=[pi / 4, 0, 0], begin_time=t0, transition_time=2)

        t0 = t0 + 2.5
        tables[0].disappear(begin_time=t0, transition_time=1, children=False)
        t0 += 1.5

        colors = [
            flatten(["text",["important"]*5,"text"]),
            flatten(["text",["important"]*9,"text"])
        ]
        locations=[
            [0, 0, 5],[-1,0,4],[-1,0,3],
        ]
        lines = [
            Text(r"\text{Reflection Points:}",aligned="center",text_size="large",color="joker"),
            Text(r"\text{\ding{43} {\bf dense} and}",color=colors[0],text_size="large"),
            Text(r"\text{\ding{43} {\bf uniformly} distributed}",color=colors[1],text_size="large"),
        ]

        display = BObject(rotation_euler=[-pi/4,0,0],location=[-1.7,0,0])
        display.appear(begin_time=t0,transition_time=0)

        balls[0].grow_reflection_points(begin_time=t0, transition_time=25)
        for i, line in enumerate(lines):
            set_parent(line,display)
            line.move_to(target_location=locations[i],begin_time=t0,transition_time=0)
            t0 = 0.5 + line.write(begin_time=t0, transition_time=0.5)


        balls[1].appear(begin_time=t0,transition_time=1)
        balls[1].start(begin_time=t0,transition_time=1039);

        # create arcs

        r5 = np.sqrt(5)
        r=4
        theta=0
        dt = 1

        # first level
        dtheta = 2*pi*balls[1].ratio.numerator/balls[1].ratio.denominator
        for i in range(3):
            arc = Arc2(start_point=[r,0,0],start_angle=theta+0.025,end_angle=theta+dtheta-0.025,normal = [0,0,1],thickness=0.1,color="joker")
            theta=theta+dtheta
            t0 = arc.grow(begin_time=t0,transition_time=dt)
            set_parent(arc,tables[1])
            r=r

        # second level
        theta_start = theta
        d2theta = (tau-theta)

        r=3.9
        for i in range(13):
            arc = Arc2(start_point=[r,0,0],start_angle=theta+0.0125,end_angle=theta+d2theta-0.0125,normal=[0,0,1],thickness=0.1,color="important",num_points=100)
            theta = theta + dtheta
            t0 = dt*0.9+arc.grow(begin_time=t0,transition_time=dt*0.1)
            set_parent(arc,tables[1])

        # third level
        r = 3.8
        d3theta = theta_start%tau-theta%tau
        theta_start = theta
        for i in range(55):
            arc = Arc2(start_point=[r,0,0],start_angle=theta-0.00625,end_angle=theta+d3theta+0.00625,normal=[0,0,1],thickness=0.1,color="custom1",num_points=100)
            theta = theta+dtheta
            t0 = dt*0.9+arc.grow(begin_time=t0,transition_time=dt*0.1)
            set_parent(arc,tables[1])

        # fourth level
        r = 3.7
        d4theta = theta_start%tau-theta%tau
        for i in range(123):
            arc = Arc2(start_point=[r, 0, 0], start_angle=theta +0.003125, end_angle=theta + d4theta - 0.003125,
                       normal=[0, 0, 1], thickness=0.1, color="text", num_points=100)
            theta = theta + dtheta
            t0 = dt * 0.9 + arc.grow(begin_time=t0, transition_time=dt * 0.1)
            set_parent(arc, tables[1])
        self.t0 = t0

    def title_movie(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 52 / 180 * pi])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -12, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=1, type="BLOOM", size=1)

        title = Text(r"\text{Benford's Law}",text_size="Huge",location=[-0.5,0,0],
                     outline_color="text",color="joker",outline_emission=1,aligned="center")
        t0 = title.write(begin_time=t0,transition_time=3)

        self.t0 = t0

    def branding(self):
        t0 = 0

        ibpy.set_volume_scatter_background(density=0.03)

        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=1024,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -12, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)


        logo_sphere = Sphere(r=4.5, location=[4.75,4.9,2.6],mesh_type="ico",name="LogoSphere",smooth=4)
        logo_sphere.grow(begin_time=t0,transition_time=0)

        title_plane = Plane(u=[-3.5,3.5],v=[-3.5,3.5],normal=[-1,1,1],name="TitlePlane")
        title_plane.move_to(target_location=[-3.5,4,2],begin_time=t0,transition_time=0)
        title_plane.grow(begin_time=t0,transition_time=0)

        logo_light = SpotLight(location=Vector([0,-2,-2]),color="movie",src="logo_movie.mp4",
                               target= logo_sphere,name="LogoLight",
                               coordinates="UV",duration=15,energy=5000)
        title_light = SpotLight(location=Vector([0,-2,-2]),target=title_plane,name="TitleLight",
                                color="movie",src="title_movie.mp4",coordinates="UV",duration=5,begin_time=10,
                                energy=5000)

        self.t0 = t0

    def logo_movie(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 52 / 180 * pi])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False,shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -12, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=0.6, type='BLOOM', size=1)

        r2 = np.sqrt(2)
        quat = Quaternion([1/r2,-1/r2,0,0])
        # quat = Quaternion()
        kwargs ={"rotation_quaternion":quat}
        kwargs_red = kwargs | {"color": "red", "solid": True, "number": random_prime}
        kwargs_green = kwargs | {"color": "green", "solid": False, "number": random_prime}
        kwargs_blue = kwargs | {"color": "blue", "solid": False, "number": random_prime}
        logo = LogoFromInstances(instance=BilliardBallReal,rotation_euler=[pi/2,0,0],
                                 scale=[5] * 3, location=[0, 0, -5],
                                 details=15, kwargs_blue=kwargs_blue, kwargs_green=kwargs_green, kwargs_red=kwargs_red)

        rot2 = Quaternion([0,1,0],pi)
        rot3 = Quaternion([0,-1,0],pi)
        for instance in logo.red_instances:
            instance.rotate(rotation_quaternion=rot2@quat,begin_time=0,transition_time=5)
            instance.rotate(rotation_quaternion=rot2@rot2@ quat, begin_time=5, transition_time=5)
        for instance in logo.green_instances:
            instance.rotate(rotation_quaternion=rot3@quat,begin_time=0,transition_time=5)
            instance.rotate(rotation_quaternion=rot3@rot3@quat, begin_time=5, transition_time=5)
        for instance in logo.blue_instances:
            instance.rotate(rotation_quaternion=rot3@quat,begin_time=0,transition_time=5)
            instance.rotate(rotation_quaternion=rot3@rot3@quat, begin_time=5, transition_time=5)

        t0 = 0.5 + logo.grow(begin_time=t0, transition_time=10)
        self.t0 = t0

    def intro_table(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        year = Text("1881", text_size="Huge", aligned="center", location=[-10, 0, 0])
        newcomb = Plane(color="image", src="newcomb.png", u=[-1, 1], v=[-1.4, 1.4], rotation_euler=[pi / 2, 0, 0],
                        location=[-6.7, -2.5, 0])
        newcomb.rescale(rescale=1.4,begin_time=t0,transition_time=0)
        year.write(begin_time=t0, transition_time=0.1)
        t0 = 0.5 + newcomb.appear(begin_time=t0, transition_time=1)

        newcomb.disappear(begin_time=t0, transition_time=1)
        t0 = 0.5 + year.unwrite(begin_time=t0, transition_time=1)

        year2 = Text("1938", text_size="Huge", aligned="center", location=[-10, 0, 0])
        benford = Plane(color="image", src="benford.png", u=[-1, 1], v=[-1.4, 1.4], rotation_euler=[pi / 2, 0, 0],
                        location=[-6.7, -2.5, 0])
        benford.rescale(rescale=1.4, begin_time=t0, transition_time=0)
        year2.write(begin_time=t0, transition_time=0.1)
        t0 = 0.5 + benford.appear(begin_time=t0, transition_time=1)

        benford.disappear(begin_time=t0, transition_time=1)
        t0 = 0.5 + year2.unwrite(begin_time=t0, transition_time=1)

        numbers  = [i for i in range(1,10)]
        probs = [round((np.log(i+1)/np.log(10)-np.log(i)/np.log(10))*100,1) for i in range(1,10)]
        number_objects = [SimpleTexBObject(str(number),text_size="Huge",aligned="center") for number in numbers]
        prob_objects = [SimpleTexBObject(str(prob)+"\%",text_size="Large",aligned="center") for prob in probs]
        data = np.array([number_objects,prob_objects])

        table = Table(data,bufferx=1,location=[-4,0,0])
        t0 = table.write_row(0,begin_time=t0,transition_time=1)
        lines =[]
        line = Cylinder.from_start_to_end(start=[-9,0,-1],end=[10,0,-1],radius=0.02)
        xs = [0,-7,-4.8,-2.8,-0.7,1.1,3,4.9,6.8]
        for i in range(1,9):
            vert = Cylinder.from_start_to_end(start=[xs[i],0,1],end=[xs[i],0,-3],radius=0.02)
            vert.grow(begin_time=t0+i*0.1,transition_time=0.1)
            lines.append(vert)

        t0 = 0.5 + line.grow(begin_time=t0,transition_time=1)
        t0 = 0.5 + table.write_row(1,begin_time=t0,transition_time=1)

        for i,l in enumerate(lines[::-1]):
            l.shrink(begin_time=t0+i*0.1,transition_time=0.1)

        line.shrink(begin_time=t0,transition_time=1)
        t0 =0.5 + table.disappear(begin_time=t0,transition_time=1)


        self.t0 = t0

    def computations2(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)


        text = Text(r"5^{10}\approx",text_size="Huge",location=(-12,0,5))
        t0 = 0.5 + text.write(begin_time=t0,transition_time=0.5)

        # look up first number
        image = Plane(color="image",src="log_table-pages025.png",name="Mantissa1",
                      location=[5.5,0,0],scale=[4.5,6.5,0],brightness=-0.5, contrast = 0.8,rotation_euler=[pi/2,0,0])
        t0 = 0.5 + image.grow(begin_time=t0,transition_time=1)

        image_material = image.ref_obj.material_slots[0].material
        mixers = box_highlighting_for_material(image_material, data={
            (0.10,0.370,0.240,0.385): ('joker', 0.75),
        })

        t0 = 0.5 + change_default_value(mixers[0].factor, from_value=1, to_value=0.5, begin_time=t0,
                                  transition_time=1)

        image.rescale(rescale=2.431,begin_time=t0,transition_time=1)
        image.move_to(target_location=[-3.514,0,2.3627],begin_time=t0,transition_time=1)

        colors = ["important","text","joker"]
        mantissa = SimpleTexBObject(r"0.69897",text_size="Large",color=colors,location=(-9,0,3))
        t0 =0.5 + mantissa.write(begin_time=t0,transition_time=1)
        t0 = 0.5 + image.disappear(begin_time=t0,transition_time=1)


        colors= ["text","important"]
        mantissa2 = SimpleTexBObject(r"\times 10", text_size="Large", color=colors, location=(-8.12, 0, 2))
        t0 = 0.5 + mantissa2.write(begin_time=t0, transition_time=1)


        line = Cylinder.from_start_to_end(start=[-9.75,0,1.5],end=[-6.75,0,1.5],radius=0.05,color="joker")
        t0 =0.5 + line.grow(begin_time=t0, transition_time=1,modus="from_start")

        colors =["important","text","joker"]
        result = SimpleTexBObject(str(6.9897)+"0",color=colors,text_size="Large",location=(-9,0,1))
        t0 = 0.5 + result.write(begin_time=t0, transition_time=0.25)

        # look up 3rd number

        image3 = Plane(color="image", src="log_table-pages043.png", name="Mantissa3",
                       location=[5.5, 0, 0], scale=[4.5, 6.5, 0], brightness=-0.5, contrast=0.8,
                       rotation_euler=[pi / 2, 0, 0])
        t0 = 0.5 + image3.grow(begin_time=t0, transition_time=1)

        image_material3 = image3.ref_obj.material_slots[0].material
        mixers3 = highlighting_for_material(image_material3, direction='Y', data={
            (0.495, 0.508): ('drawing', 0.75, 'Y'),
            (0.63, 0.69): ('joker', 0.75, 'X'),
        })

        change_default_value(mixers3[1].factor, from_value=1, to_value=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + change_default_value(mixers3[0].factor, from_value=1, to_value=0.5, begin_time=t0,
                                        transition_time=1)
        t0 = 0.5 +image3.move(direction=[1.5,0,0],begin_time=t0, transition_time=1)

        result2 = SimpleTexBObject(r"9766000",text_size="Huge",color="text",location=(-6.5,0,5))
        t0 = 0.5 + result2.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def computations(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)


        text = Text(r"46 \times 345=",text_size="Huge",location=(-12,0,5))
        t0 = 0.5 + text.write(begin_time=t0,transition_time=0.5)

        # look up first number
        image = Plane(color="image",src="log_table-pages033.png",name="Mantissa1",
                      location=[5.5,0,0],scale=[4.5,6.5,0],brightness=-0.5, contrast = 0.8,rotation_euler=[pi/2,0,0])
        t0 = 0.5 + image.grow(begin_time=t0,transition_time=1)

        image_material = image.ref_obj.material_slots[0].material
        mixers = highlighting_for_material(image_material, direction='Y', data={
            (0.740,0.755): ('drawing', 0.75,'Y'),
            (0.170,0.250): ('joker',0.75,'X'),
        })

        change_default_value(mixers[1].factor,from_value=1,to_value=0.5,begin_time=t0,transition_time=1)
        t0 = 0.5 + change_default_value(mixers[0].factor, from_value=1, to_value=0.5, begin_time=t0,
                                  transition_time=1)

        image.rescale(rescale=2.431,begin_time=t0,transition_time=1)
        image.move_to(target_location=[-4.274,0,-4.5513],begin_time=t0,transition_time=1)

        colors = ["important","text","joker"]
        mantissa = SimpleTexBObject(r"1.66276",text_size="Large",color=colors,location=(-9,0,3))
        t0 =0.5 + mantissa.write(begin_time=t0,transition_time=1)
        t0 = 0.5 + image.disappear(begin_time=t0,transition_time=1)

        # look up second number
        image2 = Plane(color="image", src="log_table-pages030.png",name="Mantissa2",
                      location=[5.5, 0, 0], scale=[4.5, 6.5, 0], brightness=-0.5, contrast=0.8,
                      rotation_euler=[pi / 2, 0, 0])
        t0 = 0.5 + image2.grow(begin_time=t0, transition_time=1)

        image_material2 = image2.ref_obj.material_slots[0].material
        mixers2 = highlighting_for_material(image_material2, direction='Y', data={
            (0.1925, 0.208): ('drawing', 0.75, 'Y'),
            (0.170, 0.250): ('joker', 0.75, 'X'),
        })

        change_default_value(mixers2[1].factor, from_value=1, to_value=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + change_default_value(mixers2[0].factor, from_value=1, to_value=0.5, begin_time=t0,
                                        transition_time=1)

        image2.rescale(rescale=2.431, begin_time=t0, transition_time=1)
        image2.move_to(target_location=[-4.133, 0, 5.5767], begin_time=t0, transition_time=1)

        mantissa2 = SimpleTexBObject(r"2.53782", text_size="Large", color=colors, location=(-9, 0, 2))
        t0 = 0.5 + mantissa2.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + image2.disappear(begin_time=t0, transition_time=1)

        line = Cylinder.from_start_to_end(start=[-9.75,0,1.5],end=[-6.75,0,1.5],radius=0.05,color="joker")
        line.grow(begin_time=t0, transition_time=1,modus="from_start")
        plus = Text(r"+",text_size="Large", color='joker', location=(-9.75,0,2),aligned="center")
        t0 = 1+plus.write(begin_time=t0+0.5, transition_time=0.1)

        result = SimpleTexBObject(str(1.66276+2.53782),color=colors,text_size="Large",location=(-9,0,1))
        t0 = 0.5 + result.write(begin_time=t0, transition_time=0.25)

        # look up 3rd number

        image3 = Plane(color="image", src="log_table-pages027.png", name="Mantissa3",
                       location=[5.5, 0, 0], scale=[4.5, 6.5, 0], brightness=-0.5, contrast=0.8,
                       rotation_euler=[pi / 2, 0, 0])
        t0 = 0.5 + image3.grow(begin_time=t0, transition_time=1)

        image_material3 = image3.ref_obj.material_slots[0].material
        mixers3 = highlighting_for_material(image_material3, direction='Y', data={
            (0.790, 0.808): ('drawing', 0.75, 'Y'),
            (0.700, 0.760): ('joker', 0.75, 'X'),
        })

        change_default_value(mixers3[1].factor, from_value=1, to_value=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + change_default_value(mixers3[0].factor, from_value=1, to_value=0.5, begin_time=t0,
                                        transition_time=1)
        t0 = 0.5 +image3.move(direction=[1.5,0,0],begin_time=t0, transition_time=1)

        result2 = SimpleTexBObject(r"15870",text_size="Huge",color="text",location=(-2.9,0,5))
        t0 = 0.5 + result2.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def logarithms(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False,shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -4,9]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=1, type="BLOOM", size=1)

        # paper=Plane(color="mix_texture",material1="image",material2="old_paper",src="log_table-pages001.png",separation=4)
        # t0  = 0.5  + paper.grow(begin_time=t0,transition_time=1)

        book = Book(pages=110,scale=Vector([1.2,1.9,0.05]))

        book.set_cover_image("log_table-pages001.png",coordinates="UV",
                             scale=[-3.7,-3.9,1],location=[1.41,2.96,0])
        for i in range(2,105):
            factor = 0.115
            bump_scale=0.05
            if i < 10:
                number = "00" + str(i)
            elif i < 100:
                number = "0" + str(i)
            else:
                number = str(i)
            print("last page", i)
            if i<24:
                wear=0
            else:
                wear = 0.0025
            if i==25:
                wear = 0.05
                bump_scale=0.1
                factor = 0.5
            if i>25 and i<48:
                wear = 0.05/(i-25)**0.8
                factor = 0.115+0.6/(i-25)**0.4
                bump_scale=0.0+0.25/(i-25)**0.8
                print(i,wear,factor)
            if i%2==0:
                book.set_page_image_with_background_color(i,"log_table-pages"+number+".png",coordinates="UV",extension="REPEAT",
                                    scale=[-4,-3.9,1],location=[1.47,2.96,0],
                                                          uv_shift=[-0.625,-0.5,0],uv_scale=[4,4,1],wear=wear,
                                                          factor =factor,darkness=0.4,bump_scale=bump_scale)
            else:
                book.set_page_image_with_background_color(i, "log_table-pages" + number + ".png", coordinates="UV", extension="REPEAT",
                                    scale=[-4, -3.9, 1], location=[1.55, 2.96, 0],
                                                          uv_shift=[-0.625,-0.5,0],uv_scale=[4,4,1],right_side=-1,
                                                          wear=wear,factor = factor,darkness=0.4,bump_scale=bump_scale)


        t0  = 0.5  + book.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + book.open(begin_time=t0,transition_time=2)

        for i in range(12):
            book.turn_page(i,begin_time=t0-2)
            t0+=0.25

        for i in range(12,21):
            book.turn_page(i,begin_time=t0)
            t0+=1

        t0+=0.5


        self.t0 = t0

    def file_explorer(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -20, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=1,type="BLOOM",size=1)

        file_explorer = FileExplorer(dir="/usr",max_length=35)
        file_explorer.appear(begin_time=t0, transition_time=1)

        t0 = 0.5 + file_explorer.scroll(begin_time=t0,transition_time=60)

        self.t0 =  t0

    def benford_explainer(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False,shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -20, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        colors1 = flatten([["important"],["text"]*3,["joker"], ["text"],["drawing"],["example"],["text"],["important"]*5,["text"]*3,["joker"]])
        colors2 = flatten(
            [["text"]*5,["important"], ["text"] , ["joker"],["text"],  ["example"],["text"]*6, ["drawing"], ["text"]*6, ["important"]*5,
             ["text"], ["joker"]])
        colors3 = flatten([
            ["example"],["text"]*6,["drawing"],["text"]*2,["joker"],["example"],["text"],["joker"],["text"]*2,["joker"],["text"],["joker"]*2,["text"]*2,["joker"],["drawing"],["text"],["joker"],["text"]*2,["example"],["text"]*6,["drawing"],["text"]
        ])
        colors4 = flatten(
            [["text"] * 5, ["important"],  ["text"]*2, ["example"], ["text"] * 6, ["drawing"],
             ["text"] * 7, ["important"] * 5])
        lines = [
            SimpleTexBObject(r"d\cdot 10^k\le 2^n< (d+1)\cdot 10^k",text_size="huge",color = colors1,aligned="center"),
            SimpleTexBObject(r"\log_{10}d+k\le n\cdot \log_{10}2<\log_{10}(d+1)+k",text_size="huge",color = colors2,aligned="center"),
            SimpleTexBObject(r"n\cdot \log_{10}2=\underbrace{\lfloor n\cdot \log_{10}2\rfloor}_k+\left\{n\cdot \log_{10}2\right\}",text_size="large",
                             color = colors3,aligned="center"),
            SimpleTexBObject(r"\log_{10}d\le \left\{n\cdot \log_{10}2\right\}<\log_{10}(d+1)", text_size="huge", color=colors4,
                             aligned="center"),
        ]

        # first line (10<=16<20)
        morph1 = SimpleTexBObject(r"16",text_size="huge",color = "text",aligned="center")
        t0 = 0.5 +morph1.write(begin_time=t0,transition_time=0.1)
        repl1  =SimpleTexBObject("\le 2^4 <",text_size="huge",color=["text","drawing","example","text"],aligned="center")
        t0 =  0.5 + morph1.replace2(repl1,src_letter_range=[0,2],img_letter_range=[1,3], shift=[0,0.0],begin_time=t0,transition_time=1,morphing=True)
        t0 = 0.5 + repl1.write(letter_set=[0,3],begin_time=t0,transition_time=0.1)
        morph2 = SimpleTexBObject(r"10",text_size="huge",color = "text",aligned="center")
        morph2.move(direction=[-0.798-0.899-0.94,0,-0.114],begin_time=0,transition_time=0)
        t0 = morph2.write(begin_time=t0,transition_time=0.1)
        cols = flatten([["important"], ["text"] * 3, ["joker"]])
        repl2 = SimpleTexBObject(r"1\cdot 10^1", text_size="huge", color=cols, aligned="center")
        repl2.move(direction=[-0.798 - 0.899 - 0.94, 0, -0.114], begin_time=0, transition_time=0)
        morph3 = SimpleTexBObject(r"20",text_size="huge",color = "text",aligned="center")
        morph3.move(direction=[0.8234+0.937+0.49, 0, -0.0633], begin_time=0, transition_time=0)
        t0 = 0.5 +morph3.write(begin_time=t0,transition_time=0.1)
        repl3 = SimpleTexBObject(r"2\cdot 10^1", text_size="huge", color=cols, aligned="center")
        repl3.move(direction=[0.8234 + 0.937 + 0.49, 0, -0.0633], begin_time=0, transition_time=0)

        morph2.replace2(repl2,
                        shift=[-0.40,0],
                        begin_time=t0,transition_time=0.5,morphing=True)
        t0 =  0.5 + morph3.replace2(repl3,
                                    shift=[0.77,0.0],
                                    begin_time=t0,transition_time=0.5,morphing=True)

        text_container = BObject(children=[morph1,morph2,morph3,repl1,repl2,repl3],location=[0,0,4.25],name="TextContainer")
        text_container.appear(begin_time=t0,transition_time=0,children=False)
        [set_parent(l, text_container) for i,l in enumerate(lines) if i<2]

        # formal version
        lines[0].move(direction=[1.2,0,-1.34],begin_time=t0,transition_time=0.0)
        t0 = 0.5 + lines[0].write(begin_time=t0,transition_time=0.75)

        # digit  log digit table
        data = [
            [SimpleTexBObject(r"d",color="important"),SimpleTexBObject(r"\log_{10}d",color=flatten([["text"]*5,["important"]]))],
        ]
        for d in range(1,10):
            if d==1:
                data.append([SimpleTexBObject(str(d), color="important"),
                             SimpleTexBObject(r"0", color="text")])
            else:
                string = f"{np.log(d)/np.log(10):.2}"
                while len(string)<4:
                    string+="0"
                data.append([SimpleTexBObject(str(d),color="important"),SimpleTexBObject(string,color="text")])



        # # apply log magic
        lines[1].move(direction=[0,0,-2.66],begin_time=t0,transition_time=0)
        t0 = 0.5 + lines[1].write(begin_time=t0,transition_time=1)

        tab_lines = [
            Cylinder.from_start_to_end(start=[-8.75,0,-0.75],end=[-7.20,0,-0.75],color="text",radius=0.01),
            Cylinder.from_start_to_end(start=[-8.33,0,-0.15],end=[-8.33,0,-4.7],color="text",radius=0.01),
        ]

        # explain floor part

        explain = SimpleTexBObject(r"3.14 = \,\,\,\,\,\,\,3\,\,\,\,\,\, + \,\,0.14\,\,\,\,",text_size="large",color="text",aligned="center")
        explain2 = SimpleTexBObject(r"3.14=\lfloor 3.14\rfloor + \{3.14\}",text_size="large",color="text",aligned="center",location=[0,0,-1])
        explain_container = BObject(children=[explain,explain2],location=[0,0,-0.5],name="ExplainContainer")
        explain_container.appear(begin_time=t0,transition_time=0,children=False)

        t0 =0.5 + explain.write(begin_time=t0,transition_time=0.5)
        t0 =0.5 + explain2.write(begin_time=t0,transition_time=0.5)

        t0 = 0.5 + explain_container.move(direction=[6.5,0,-3],begin_time=t0,transition_time=1)

        lines[2].move(direction=[2, 0, -0.66], begin_time=t0, transition_time=0)
        t0 = 0.5 + lines[2].write(letter_set=list(range(0,10))+[11,12,14,15,17,20,21,23,24]+list(range(26,37)),begin_time=t0, transition_time=1)

        table = Table(data, alignment=['r', 'r'], location=[-8.5, 0, -1.5], buffer_x=0.6)
        [table.write_entry(r, 0, begin_time=t0, transition_time=0.25) for r in range(0, 10)]
        t0 += 0.5
        [l.grow(begin_time=t0, transition_time=0.5) for l in tab_lines]
        [table.write_entry(r, 1, begin_time=t0, transition_time=0.5) for r in range(0, 10)]
        t0 += 1

        lines[2].change_color_of_letters([9,11,12,14,15,17,20,21,23,24],new_color="joker",begin_time=t0,transition_time=0.5)
        t0 = 0.5 + lines[2].write(letter_set=[10,13,16,18,19,22,25],begin_time=t0, transition_time=1)
        t0 = 0.5 + lines[2].move(direction=[-3.5,0,-3.5],begin_time=t0,transition_time=1)

        # result
        lines[3].move(direction=[0,0,0.1],begin_time=t0,transition_time=0)
        t0 = 0.5 + lines[3].write(begin_time=t0,transition_time=1)

        # stay=SimpleTexBObject(r"2^n",color=["drawing","example"],text_size="huge",aligned="center")
        # stay.move(direction=[0,0,2.946],begin_time=t0,transition_time=0)
        # t0 = stay.write(begin_time=t0,transition_time=0)
        #
        # stay.move(direction=[-8.25,0,4.5],begin_time=t0,transition_time=1)
        # stay.rescale(rescale=1.5,begin_time=t0,transition_time=1)

        explain_container.disappear(begin_time=t0,transition_time=1,children=True)
        lines[2].disappear(begin_time=t0,transition_time=1)
        lines[3].move(direction=[0.22,0,4.22],begin_time=t0+0.5,transition_time=0.5)
        t0 = 0.5 + text_container.shrink(begin_time=t0,transition_time=1)


        benford_interval = BenfordInterval(scale=3,location=[-2,0,1.5])
        t0  = 0.5+ benford_interval.appear(begin_time=t0,transition_time=0.5)

        copies = []
        for i in range(1,10):
            copy= data[i][1].copy()
            copy.appear(begin_time=t0+0.3*(i-1),transition_time=0)
            copy.move_to(target_location=[-2.1+np.log(i)/np.log(10)*9,0,1.9+i//6*i%2*0.4],begin_time=t0+0.3*(i-1),transition_time=0.3)
            copies.append(copy)
        t0 = 0.5 + benford_interval.show_tics(from_value=0,to_value=9,begin_time=t0,transition_time=3)


        t0 = 0.5 + benford_interval.show_labels(from_value=0,to_value=9,begin_time=t0,transition_time=1)

        [copy.disappear(begin_time=t0+i*0.1,transition_time=0.5) for i,copy in enumerate(copies)]
        t0 += 1.5
        t0 =0.5 +  benford_interval.transform_interval(from_value=0,to_value=1,shift = [1,0,-0.9],begin_time=t0,transition_time=3)

        theta = SimpleTexBObject(r"\theta = {\log_{10}2\over 2\pi}",location=[1,-0.5,-1.15],
                                 color=flatten([["example"]*2,["text"]*3,["example"],["text"],["example"]*2,["text"],["drawing"]]),text_size="large",aligned="center")
        t0=0.5+theta.write(letter_set=[0,1,2,3,4,5,8,9,6,7,10],begin_time=t0,transition_time=0.5)
        table = BilliardsTableRound(radius=3, location=[1,0,-1.15], scale=0.59,rotation_euler=[pi/2,0,0])
        ball = BilliardBallRound(ratio=np.log(2)/np.log(10), number=9, radius=7, scale=0.5,prime=419)
        table.appear(begin_time=t0, transition_time=0.3)
        t0 = ball.appear(begin_time=t0, transition_time=0.3)
        set_parent(ball, table)
        ball.start(begin_time=t0, transition_time=419)
        t0=table.disappear(begin_time=t0+5, transition_time=0.3,children=False)
        ball.change_trace_thickness(from_value=1,to_value=0.05,begin_time=t0,transition_time=5)
        ball.rescale(rescale=1.4,begin_time=t0,transition_time=1)

        self.t0 =t0

    def billiard_irrational(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -17, 17]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)


        floats = [1./np.sqrt(2),2./(1+np.sqrt(5))]
        balls = []
        tables = []
        for i in range(2):
            table = BilliardsTableRound(radius=3, location=[-5 + 2*i * 5,-2, 0], scale=0.95)
            table.appear(begin_time=t0 + i * 0.5, transition_time=1)
            tables.append(table)
            ball = BilliardBallRound(ratio=floats[i], number=14+i, radius=7, scale=0.5)
            balls.append(ball)
            set_parent(ball, table)

        t0 = t0 + 3

        texts = [
            SimpleTexBObject(r"\theta = {1 \over \sqrt{2}}\pi", location=[-8, 6, 0], rotation_euler=[pi / 4, 0, 0],
                             text_size="huge", color="example", aligned="center"),
            SimpleTexBObject(r"\theta = {2 \over 1+\sqrt{5}}\pi", location=[8, 6, 0], rotation_euler=[pi / 4, 0, 0],
                             text_size="huge", color="example", aligned="center"),
        ]


        for i in range(2):
            texts[i].write(begin_time=t0 , transition_time=0.5)

        t0 += 1

        for i, ball in enumerate(balls):
            ball.appear(begin_time=t0, transition_time=1)
            ball.start(begin_time=t0, transition_time=200)
            ball.change_trace_thickness(from_value=1,to_value=0.05,begin_time=t0,transition_time=40)
        t0 = t0 + 5

        for table in tables:
            table.rotate(rotation_euler=[pi / 4, 0, 0], begin_time=t0, transition_time=2)

        t0 = t0 + 2.5
        for table in tables:
            table.disappear(begin_time=t0, transition_time=1, children=False)
        t0 += 1.5

        colors = [
            flatten(["text",["important"]*5,"text"]),
            flatten(["text",["important"]*9,"text"])
        ]
        locations=[
            [0, 0, 5],[-1,0,4],[-1,0,3],
        ]
        lines = [
            Text(r"\text{Reflection Points:}",aligned="center",text_size="large",color="joker"),
            Text(r"\text{\ding{43} {\bf dense} and}",color=colors[0],text_size="large"),
            Text(r"\text{\ding{43} {\bf uniformly} distributed}",color=colors[1],text_size="large"),
        ]

        display = BObject(rotation_euler=[-pi/4,0,0],location=[-1.7,0,0])
        display.appear(begin_time=t0,transition_time=0)

        t0 = 25

        for i, line in enumerate(lines):
            set_parent(line,display)
            line.move_to(target_location=locations[i],begin_time=t0,transition_time=0)
            t0 = 0.5 + line.write(begin_time=t0, transition_time=0.5)

        for ball in balls:
            ball.grow_reflection_points(begin_time=t0,transition_time=5)

        t0 = t0+5.5

        t0 = 3 + camera_zoom(lens=115,begin_time=t0,transition_time=3)
        t0 = 3 + camera_zoom(lens=40,begin_time=t0,transition_time=3)
        self.t0 = t0

    def billiard_stars(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -17, 17]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)


        numerators = [1,2,3]
        denominator=7
        balls = []
        tables = []
        for i in range(3):
            table = BilliardsTableRound(radius=2, location=[-6 + i * 6, -2 + i % 2 * 6, 0], scale=1)
            table.appear(begin_time=t0 + i * 0.5, transition_time=1)
            tables.append(table)
            ball = BilliardBallRound(ratio=(numerators[i], denominator), number=7, radius=5, scale=0.5)
            balls.append(ball)
            set_parent(ball, table)

        t0 = t0 + 3

        texts = [
            SimpleTexBObject(r"\theta = {1 \over 7}\pi", location=[-8, 6, 0], rotation_euler=[pi / 4, 0, 0],
                             text_size="huge", color="example", aligned="center"),
            SimpleTexBObject(r"\theta = {2 \over 7}\pi", location=[0, -4, 0], rotation_euler=[pi / 4, 0, 0],
                             text_size="huge", color="example", aligned="center"),
            SimpleTexBObject(r"\theta = {3 \over 7}\pi", location=[8, 6, 0], rotation_euler=[pi / 4, 0, 0],
                             text_size="huge", color="example", aligned="center"),
        ]

        # mark reflection points with spheres
        bobs = [BObject(name="construction" + str(i), location=[-6 + 6 * i, -2 + i % 2 * 6, 1],
                        rotation_euler=[0, 0, 0]) for i in range(3)]
        [bob.appear(begin_time=t0, transition_time=0) for bob in bobs]

        for j in range(7):
            for i in range(3):
                sphere =Sphere(r=0.1,location=3*Vector([np.cos(tau*j/7),np.sin(tau*j/7),0]),color="important",name="Sphere"+str(i)+str(j))
                sphere.grow(begin_time=t0+0.1*j,transition_time=1)
                if j==0:
                    texts[i].write(begin_time=t0 , transition_time=0.5)
                set_parent(sphere,bobs[i])
        t0 += 2

        for i, ball in enumerate(balls):
            ball.appear(begin_time=t0 + 0.5 * i - 1, transition_time=1)
            ball.start(begin_time=t0 + 0.5 * i, transition_time=3)

        t0 = t0 + 5

        for bob,table in zip(bobs,tables):
            table.rotate(rotation_euler=[pi / 4, 0, 0], begin_time=t0, transition_time=2)
            bob.rotate(rotation_euler=[pi / 4, 0, 0], begin_time=t0, transition_time=2)
            bob.move(direction=[0,0,-1],begin_time=t0,transition_time=2)

        t0 = t0 + 2.5
        for table in tables:
            table.disappear(begin_time=t0, transition_time=1, children=False)
        t0 += 1.5



        self.t0 = t0

    def billiard_polygons(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0,0, 0]))
        camera_location = [0,-17,17]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        tables = []
        ratios = [3,5,8]
        balls = []
        tables = []
        for i in range(3):
            table = BilliardsTableRound(radius=2, location=[-6+i*6,-2+i%2*6,0], scale=1)
            table.appear(begin_time=t0+i*0.5, transition_time=1)
            tables.append(table)
            ball = BilliardBallRound(ratio = ratios[i],number=ratios[i],radius=5,scale=0.5)
            balls.append(ball)
            set_parent(ball, table)
            tables.append(table)

        t0 = t0+3

        texts = [
            SimpleTexBObject(r"\theta = {\pi \over 3}",location = [-8,6,0],rotation_euler=[pi/4,0,0],text_size="huge",color="example",aligned="center"),
            SimpleTexBObject(r"\theta = {\pi \over 5}",location = [0,-4,0],rotation_euler=[pi/4,0,0],text_size="huge",color="example",aligned="center"),
            SimpleTexBObject(r"\theta = {\pi \over 8}",location = [8,6,0],rotation_euler=[pi/4,0,0],text_size="huge",color="example",aligned="center"),
        ]

        for i,ball in enumerate(balls):
            ball.appear(begin_time=t0+0.5*i-1, transition_time=1)
            texts[i].write(begin_time=t0+0.5*i+0.5, transition_time=0.5)
            ball.start(begin_time=t0+0.5*i, transition_time=3)

        t0 = t0+5

        for table in tables:
            table.rotate(rotation_euler=[pi/4,0,0],begin_time=t0,transition_time=2)

        t0 = t0 + 2.5
        for table in tables:
            table.disappear(begin_time=t0, transition_time=1,children=False)
        t0 +=1.5

        # make the construction in the x-y plane for table in a different object, move and rotate the object in the end
        bobs=[BObject(name="construction"+str(i),location = [-6+6*i,-2+i%2*6,0],rotation_euler=[pi/4,0,0] ) for i in range(3)]
        [bob.appear(begin_time=t0, transition_time=0) for bob in bobs]
        circles=[]
        for i in range(3):
            circle = Circle2(radius =2.5,location=[0,0,-0.2],color="drawing",thickness=0.5,name="Circle"+str(i))
            circle.grow(begin_time=t0,transition_time=1)
            circles.append(circle)
            set_parent(circle,bobs[i])
        t0 +=1.5

        tp = Vector([2.5,0,0])
        lines = []
        for i in range(3):
            line = Cylinder.from_start_to_end(start=[2.5,-3,-0.2],end=[2.5,3,-0.2],thickness=0.5,color="drawing",name="Line"+str(i))
            line.grow(begin_time=t0,transition_time=1,modus="from_center")
            lines.append(line)
            set_parent(line,bobs[i])
        t0 +=1.5

        for i in range(3):
            circle_points = [2.5*Vector([np.cos(tau*j/ratios[i]),np.sin(tau*j/ratios[i]),0]) for j in range(ratios[i])]
            cn = circle_points[-1] # the last circle point
            p_ext = tp-2.5*(cn-tp).normalized()# extension point
            line = Cylinder.from_start_to_end(start=tp,end=p_ext,thickness=0.5,color="dashed",colors =["example"],name="Line"+str(i))
            line.grow(begin_time=t0,transition_time=1,modus="from_center")
            set_parent(line,bobs[i])
        t0 +=1.5

        labels =[
            SimpleTexBObject(r"{\pi\over 3}",aligned="center",color="example",rotation_euler=Vector(),text_size="large"),
            SimpleTexBObject(r"{60^\circ}",aligned="center",color="example",rotation_euler=Vector(),text_size="large"),
            SimpleTexBObject(r"{\pi\over 5}",aligned="center",color="example",rotation_euler=Vector(),text_size="large"),
            SimpleTexBObject(r"{36^\circ}",aligned="center",color="example",rotation_euler=Vector(),text_size="large"),
            SimpleTexBObject(r"{\pi\over 8}",aligned="center",color="example",rotation_euler=Vector(),text_size="large"),
            SimpleTexBObject(r"{{45\over 2}^\circ}",aligned="center",color="example",rotation_euler=Vector(),text_size="large",scale=0.8),
            ]

        for i in range(3):
            circles[i].disappear(alpha=0.15,begin_time=t0,transition_time=0.5)
            lines[i].disappear(alpha=0.15,begin_time=t0,transition_time=0.5)
            arc = Arc2(center = tp,radius=2,start_point = tp+Vector([0,2-i*0.1,0]),start_angle=0,end_angle=pi/ratios[i],thickness=0.5,
                       color="dashed",colors=["example","drawing"],name="Arc"+str(i))
            arc.grow(begin_time=t0,transition_time=1)
            labels[2*i].move_to(target_location=tp+1.25*Vector([-np.sin(pi/ratios[i]*.5),np.cos(pi/ratios[i]*.5),0]),begin_time=t0,transition_time=0)
            labels[2*i].write(begin_time=t0+0.5,transition_time=0.5)
            set_parent(arc,bobs[i])
            set_parent(labels[2*i],bobs[i])
        t0 +=1.5

        for i in range(3):
            arc = Arc2(center = tp,radius=2,start_point = tp+Vector([0,2-i*0.1,0]),start_angle=0,end_angle=-pi/ratios[i],thickness=0.5,
                       color="dashed",colors=["example","drawing"],name="Arc"+str(i))
            arc.grow(begin_time=t0,transition_time=1)
            labels[2*i+1].move_to(target_location=tp+1.25*Vector([np.sin(pi/ratios[i]*.5),np.cos(pi/ratios[i]*.5),0]),begin_time=t0,transition_time=0)
            if i==2:
                labels[2*i+1].write(letter_set=[0,3,1,2,4],begin_time=t0+0.5,transition_time=0.5)
            else:
                labels[2*i+1].write(begin_time=t0+0.5,transition_time=0.5)
            set_parent(arc,bobs[i])
            set_parent(labels[2*i+1],bobs[i])
        t0 +=1.5
        self.t0 = t0

    def billiard_deflection_angle(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -17, 17]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)



        table = BilliardsTableRound(radius=2, location=[0, 4, 0], scale=1)
        table.appear(begin_time=t0, transition_time=1)
        ball = BilliardBallRound(ratio=3, number=3, radius=5, scale=0.5)
        set_parent(ball, table)

        t0 = t0 + 3

        ball.appear(begin_time=t0 -0.5, transition_time=1)
        ball.start(begin_time=t0 + 0.5, transition_time=3)

        t0 = t0 + 5
        table.rotate(rotation_euler=[pi / 4, 0, 0], begin_time=t0, transition_time=2)
        t0 = t0 + 2.5
        table.disappear(begin_time=t0, transition_time=1, children=False)
        t0 += 1.5

        # make the construction in the x-y plane for table in a different object, move and rotate the object in the end
        bob = BObject(name="construction", location=[0, 4, 0],
                        rotation_euler=[pi / 4, 0, 0])
        bob.appear(begin_time=t0, transition_time=0)

        circle = Circle2(radius=2.5, location=[0, 0, -0.2], color="drawing", thickness=0.5, name="Circle" )
        circle.grow(begin_time=t0, transition_time=1)
        set_parent(circle, bob)
        t0 += 1.5

        tp = Vector([2.5, 0, 0])
        line = Cylinder.from_start_to_end(start=[2.5, -3, -0.2], end=[2.5, 3, -0.2], thickness=0.5, color="drawing",
                                          name="Line" )
        line.grow(begin_time=t0, transition_time=1, modus="from_center")
        set_parent(line, bob)
        t0 += 1.5

        circle_points = [2.5 * Vector([np.cos(tau * j / 3), np.sin(tau * j /3), 0]) for j in
                         range(3)]
        cn = circle_points[-1]  # the last circle point
        p_ext = tp - 2.5 * (cn - tp).normalized()  # extension point
        line = Cylinder.from_start_to_end(start=tp, end=p_ext, thickness=0.5, color="dashed", colors=["example"],
                                          name="Line")
        line.grow(begin_time=t0, transition_time=1, modus="from_center")
        set_parent(line, bob)

        t0 += 1.5

        labels = [
            SimpleTexBObject(r"{\theta}", aligned="center", color="example", rotation_euler=Vector(),
                             text_size="large"),
            SimpleTexBObject(r"{\theta}", aligned="center", color="example", rotation_euler=Vector(),
                             text_size="large"),
        ]


        circle.disappear(alpha=0.15, begin_time=t0, transition_time=0.5)
        line.disappear(alpha=0.15, begin_time=t0, transition_time=0.5)
        arc = Arc2(center=tp, start_point=tp + Vector([0, 2 , 0]), start_angle=0,
                   end_angle=pi / 3, thickness=0.5,
                   color="dashed", colors=["example", "drawing"], name="Arc" )
        arc.grow(begin_time=t0, transition_time=1)
        labels[0].move_to(
            target_location=tp + 1.25 * Vector([-np.sin(pi / 3 * .5), np.cos(pi / 3 * .5), 0]),
            begin_time=t0, transition_time=0)
        labels[0].write(begin_time=t0 + 0.5, transition_time=0.5)
        set_parent(arc, bob)
        set_parent(labels[0], bob)
        t0 += 1.5

        arc = Arc2(center=tp, start_point=tp + Vector([0, 2 , 0]), start_angle=0,
                   end_angle=-pi / 3, thickness=0.5,
                   color="dashed", colors=["example", "drawing"], name="Arc2" )
        arc.grow(begin_time=t0, transition_time=1)
        labels[1].move_to(
            target_location=tp + 1.25 * Vector([np.sin(pi / 3 * .5), np.cos(pi /3 * .5), 0]),
            begin_time=t0, transition_time=0)
        labels[1].write(begin_time=t0 + 0.5, transition_time=0.5)
        set_parent(arc, bob)
        set_parent(labels[ 1], bob)

        t0 += 1.5
        self.t0 = t0

    def fibonacci(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[60/180*pi, 20/180*pi, 52 / 180 * pi])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([2,-0.5, 0]))
        camera_location = [2,-0.5, 10]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=1, type='STREAKS', size=4)

        cube = Cube()
        level = 13
        modifier = FibonacciModifier(level=level)
        cube.add_mesh_modifier(type="NODES", node_modifier=modifier)

        scale_box = BObject()
        scale_box.appear(begin_time=t0, transition_time=0)

        r5 = np.sqrt(5)
        phi = (1+r5)/2
        psi = (1-r5)/2
        fibs = [int((phi**n-psi**n)/r5) for n in range(1,level+2)]
        print(fibs)

        t0 = 0.5 + cube.appear(begin_time=t0, transition_time=1)

        sliders = []
        for i in range(1,level+1):
            sliders.append(get_geometry_node_from_modifier(modifier, label="Slider" + str(i)))
            if i==6:
                camera_empty.move(direction=[0,-0.5,0],begin_time=t0,transition_time=3)
                ibpy.camera_move(shift=[0,-0.5,0],begin_time=t0,transition_time=3)
            if i==10:
                camera_empty.move(direction=[0,-0.1,0],begin_time=t0,transition_time=3)
                ibpy.camera_move(shift=[0,-0.1,0],begin_time=t0,transition_time=3)
            if i%4==2 and i>4:
                # t0  =0.5 + cube.rescale(rescale=1/phi**4, begin_time=t0, transition_time=3)
                cube.rescale(rescale=1/phi**4, begin_time=t0, transition_time=3)
            t0 = 0.5 + ibpy.change_default_value(sliders[-1], from_value=0, to_value=1, begin_time=t0,
                                                 transition_time=1)

        camera_empty.move(direction=[-0.7, 0, 0], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + ibpy.camera_move(shift=[-0.7, 0, 0], begin_time=t0, transition_time=0.5)
        # t0 = 0.5 + change_default_value(sliders[-1],from_value=1, to_value=0, begin_time=t0, transition_time=2)

        font_centers = []
        font_scales = []
        for i in range(1,level+1):
            # move numbers, rescale them and change from center to right alignment
            font_centers.append(get_geometry_node_from_modifier(modifier, label="FontCenter"+str(i)))
            font_scales.append(get_geometry_node_from_modifier(modifier, label="FontScale"+str(i)))
            ra = get_geometry_node_from_modifier(modifier, label="RightAligned"+str(i))
            ibpy.change_default_value(ra,from_value=1,to_value=0,begin_time=t0,transition_time=1)
            ibpy.change_default_value(font_scales[-1],from_value=fibs[i-1]*0.9**(len(str(fibs[i-1]))-1),to_value=34*0.9,begin_time=t0,transition_time=1)
            ibpy.change_default_vector(font_centers[-1],from_value=modifier.centers[i-1],to_value=Vector([-110,75-18*i,0]),begin_time=t0,transition_time=1)

        # draw spiral on top of it
        spiral_growth_node = get_geometry_node_from_modifier(modifier, label="SpiralGrowthFactor")
        # start from slightly negative value to avoid drawing the start of the curve from the beginning
        t0 = 0.5 + change_default_value(spiral_growth_node,from_value=-0.1,to_value=1.1,begin_time=t0,transition_time=5)

        self.t0 = t0

    def power_of_two(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[60/180*pi, 20/180*pi, 52 / 180 * pi])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0,0,10]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        cube = Cube()
        level = 10
        modifier=PowerOfTwoModifier(level=level)
        cube.add_mesh_modifier(type="NODES",node_modifier=modifier)

        scale_box = BObject()
        scale_box.appear(begin_time=t0,transition_time=0)

        powers = [2**i for i in range(level+1)]
        bobs = []
        for i,p in enumerate(powers):
            bobs.append(
                SimpleTexBObject(r""+str(p),aligned="right",color="text",text_size="large",location=[-1,-i*0.5,0],rotation_euler=[0,0,0])
            )
            set_parent(bobs[-1],scale_box)

        bobs[0].write(begin_time=t0+0.9,transition_time=0.1)
        t0 = 0.5 + cube.appear(begin_time=t0, transition_time=1)

        sliders=[]
        for i in range(level):
            sliders.append(get_geometry_node_from_modifier(modifier,label="Slider"+str(i)))
            bobs[i+1].write(begin_time=t0,transition_time=0.1*(i+1))
            if i == 3:
                camera_empty.move_to(target_location=[0,-1,0],begin_time=t0,transition_time=1)
                ibpy.camera_move(shift=[0,-1,0],begin_time=t0,transition_time=1)
            elif i==4:
                camera_empty.move_to(target_location=[2,-1, 0], begin_time=t0, transition_time=1)
                cube.rescale(rescale=0.88,begin_time=t0,transition_time=1)
                ibpy.camera_move(shift=[2, 0, 0], begin_time=t0, transition_time=1)
            elif i==5:
                camera_empty.move_to(target_location=[2, -2.2, 0], begin_time=t0, transition_time=1)
                ibpy.camera_move(shift=[0,-1.2,0],begin_time=t0,transition_time=1)
                cube.rescale(rescale=0.69,begin_time=t0,transition_time=1)
                scale_box.rescale(rescale=0.9,begin_time=t0,transition_time=1)
            elif i==6:
                cube.rescale(rescale=0.69,begin_time=t0,transition_time=1)
            elif i>6:
                cube.rescale(rescale=0.7,begin_time=t0,transition_time=1)
            t0  = 0.5 + ibpy.change_default_value(sliders[-1],from_value=0,to_value=1,begin_time=t0,transition_time=1)



        self.t0  = t0

    def benford3(self):
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
        ibpy.set_camera_lens(lens=38)

        # /usr/src up to 10000 seconds is possible but only small files
        # /usr/share is the same
        benford = BenfordFilesDiagram(transition_frames=30, rotation_euler=[pi / 2, 0, 0], location=[7, 0, 4],
                                          number_material="joker", bar_material="custom3", start_time=2.5,
                                      path="/usr")
        benford.rotate(rotation_euler=[pi / 2, 0, 0], begin_time=t0, transition_time=0)
        benford.move_to(target_location=[7, 0, 4], begin_time=t0, transition_time=0)

        x_axis = NumberLine2(length=12, location=[-5.5, 0, -5.25], domain=[1, 9], n_tics=8, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{first digit}", tip_length=0)

        y_axis = NumberLine2(length=5, location=[-7, 0, -4.65], domain=[0, 50], tic_label_suffix="\,\%", n_tics=5,
                             tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.4, 0, 0], direction="VERTICAL",
                             axis_label=r"\text{frequency}",
                             axis_label_location=[-1.5, 0, 5.7])
        y_axis.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + x_axis.grow(begin_time=t0, transition_time=1)

        benford.appear(begin_time=t0, transition_time=1)

        t0 = 40
        log = SimpleFunction(function=lambda x: np.log10((x + 1) / x), domain=[1, 9], color="example",
                             scale=[1.5, 10, 10], location=[-7, -0.35, -4.65])
        # title = SimpleTexBObject(r"\text{Benford's Law}", text_size="Huge", color="example", location=[-6, 0, 5.5],
        #                          aligned="center")
        # title.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + log.grow(begin_time=t0, transition_time=1)

        self.t0 = t0

    def benford2(self):
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
        ibpy.set_camera_lens(lens=38)

        data = []
        with open(os.path.join(DATA_DIR,"fibonacci.dat"),"r") as f:
            for line in f.readlines():
                data.append(int(line))

        benford = BenfordFibonacciDiagram(transition_frames=30,rotation_euler=[pi / 2, 0, 0],location=[7,0,4],number_material="joker",bar_material="custom3",start_time=2.5)
        benford.add_data(data)
        benford.rotate(rotation_euler=[pi / 2, 0, 0],begin_time=t0,transition_time=0)
        benford.move_to(target_location=[7,0,4],begin_time=t0,transition_time=0)

        x_axis = NumberLine2(length=12, location=[-5.5, 0, -5.25], domain=[1, 9], n_tics=8, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{first digit}", tip_length=0)

        y_axis = NumberLine2(length=5, location=[-7, 0, -4.65], domain=[0,50],tic_label_suffix="\,\%", n_tics=5, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.4, 0, 0],direction="VERTICAL",axis_label=r"\text{frequency}",
                             axis_label_location=[-1.5,0,5.7])
        y_axis.grow(begin_time=t0, transition_time=1)
        t0  = 0.5 + x_axis.grow(begin_time=t0, transition_time=1)

        benford.appear(begin_time=t0, transition_time=1)
        log = SimpleFunction(function=lambda x: np.log10((x+1)/x),domain=[1,9],color="example",scale=[1.5,10,10],location=[-7,-0.35,-4.65])
        title = SimpleTexBObject(r"\text{Fibonacci Numbers: } f(n)=f(n-1)+f(n-2)",text_size="Large",color="example",location=[0,0,5.5],
                                 aligned="center")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1)
        t0  = 0.5  + log.grow(begin_time=t0, transition_time=1)

        self.t0 = t0

    def benford(self):
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

        benford = BenfordDiagram(transition_frames=30,rotation_euler=[pi / 2, 0, 0],location=[7,0,4],number_material="joker",bar_material="custom3",start_time=2.5)
        benford.rotate(rotation_euler=[pi / 2, 0, 0],begin_time=t0,transition_time=0)
        benford.move_to(target_location=[7,0,4],begin_time=t0,transition_time=0)

        x_axis = NumberLine2(length=12, location=[-5.5, 0, -5.25], domain=[1, 9], n_tics=8, tic_label_digits=0,
                             tic_label_aligned="center", tic_label_shift=[0, 0, -0.3],
                             label_position="right",
                             axis_label=r"\text{first digit}", tip_length=0)

        y_axis = NumberLine2(length=5, location=[-7, 0, -4.65], domain=[0,50],tic_label_suffix="\,\%", n_tics=5, tic_label_digits=0,
                             tic_label_aligned="right", tic_label_shift=[-1.4, 0, 0],direction="VERTICAL",axis_label=r"\text{frequency}",
                             axis_label_location=[-1.5,0,5.7])
        y_axis.grow(begin_time=t0, transition_time=1)
        t0  = 0.5 + x_axis.grow(begin_time=t0, transition_time=1)


        benford.appear(begin_time=t0, transition_time=1)
        log = SimpleFunction(function=lambda x: np.log10((x+1)/x),domain=[1,9],color="example",scale=[1.5,10,10],location=[-7,-0.35,-4.65])
        title = SimpleTexBObject(r"\text{Powers of Two: } f(n)=2^n", text_size="Large", color="example",
                                 location=[0, 0, 5.5],
                                 aligned="center")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1)
        t0  = 0.5  + log.grow(begin_time=t0, transition_time=1)

        self.t0 = t0



if __name__ == '__main__':
    try:
        example = Benford()
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

        if selected_scene =='logo_movie':
            resolution = [1080]*2
        else:
            resolution = [1920,1080]

        example.create(name=selected_scene, resolution=resolution, start_at_zero=True)

    except:
        print_time_report()
        raise ()
