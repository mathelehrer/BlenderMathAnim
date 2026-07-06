from collections import OrderedDict

import numpy as np

from compositions.compositions import create_glow_composition
from geometry_nodes.geometry_nodes_modifier import LegendrePolynomials, AssociatedLegendreP, \
    PlmSurface, YlmSurface, YlmSurfaceReference, YlmSurface_200, UnfoldModifier, CustomOpDemoModifier
from interface import ibpy
from interface.ibpy import Vector, Quaternion
from interface.interface_constants import BLENDER_EEVEE

from objects.bderivation import BDerivation
from objects.book import Book
from objects.coordinate_system import CoordinateSystem2
from objects.cube import Cube
from objects.derived_objects.rubiks_cube import BRubiksCube
from objects.display import Display
from objects.empties import EmptyCube
from objects.logo import GeometryLogo
from objects.number_line import NumberLine2
from objects.plane import Plane
from objects.platonic_solids import SubdividedPentagon, Icosahedron, Dodecahedron
from objects.polyhedron import Polyhedron
from objects.slider import BSlider
from objects.tex_bobject import SimpleTexBObject
from objects.text import Text, MorphText
from objects.textscalebox import TextScaleBox
from perform.scene import Scene
from utils.utils import print_time_report

pi = np.pi
tau = 2*pi


class Examples(Scene):
    """
    This scene contains examples for complex animations for reference
    """
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('solids', {'duration': 10}),
            ('icosahedron_and_dodecahedron', {'duration': 10}),
            ('logo', {'duration': 10}),
            ('slider', {'duration': 10}),
            ('spherical_harmonics', {'duration': 65}),
            ('coordinate_system_3d', {'duration': 50}),
            ('associatedLegendreP', {'duration': 50}),
            ('legendreP', {'duration': 50}),
            ('numberline', {'duration': 5}),
            ('book', {'duration': 15}),
            ('move_letters_and_move_copies_of_letters',{'duration': 15}),
            ('custom_op_demo',{'duration':15}),
            ('bderivation_complete_square', {'duration': 25}),
            ('bderivation_geometric_sum', {'duration': 35}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def solids(self):
        t0 = 0
        polyhedron = Polyhedron.from_points(solid_type="TRUNC_ICOSA")
        polyhedron.appear(begin_time=t0,transition_time=0)
        self.t0 = t0

    def icosahedron_and_dodecahedron(self):
        t0  = 0
        pentagon = SubdividedPentagon(location=[0, 0, 0], sub_divide=2)
        pentagon.grow(begin_time=t0, transition_time=1)
        ico = Icosahedron(location = [-3,0,0])
        dodeca = Dodecahedron(location = [3,0,0])
        dodeca.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + ico.grow(begin_time=t0, transition_time=1)

        self.t0 = t0


    def logo(self):
        t0  = 0
        logo = GeometryLogo()
        t0  =0.5 + logo.appear(begin_time=t0,transition_time=1)
        self.t0 = t0

    def slider(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -20, 0])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        slider = BSlider(label="a",domain=[0,1])
        t0  = 0.5+ slider.appear(begin_time=t0,transition_time=1)
        t0  =0.5 + ibpy.change_default_value(slider.slider_value_slot,from_value=0,to_value=1,begin_time=t0,transition_time=1)
        t0  =0.5 + ibpy.change_default_value(slider.slider_value_slot,from_value=1,to_value=-1,begin_time=t0,transition_time=2)
        t0  =0.5 + ibpy.change_default_value(slider.slider_value_slot,from_value=-1,to_value=0,begin_time=t0,transition_time=2)
        self.t0 = t0

    def spherical_harmonics(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[10, -20, 10])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # create comparison in 3D coordinate system  between regular spherical harmonics node and recursive spherical harmonics node
        coords = CoordinateSystem2(dimension=3, location=[0, 0, 0], name="CoordinateSystem3D",
                                   lengths=[15, 15, 7.5], colors=["text", "text", "text"],
                                   domains=[[-pi, pi], [0, pi], [-1, 1]], tic_label_digits=[1, 2, 1],
                                   tic_labels=[
                                       {r"-\pi": -pi, r"\pi": pi},
                                       {r"\,\tfrac{1}{2}\pi": 0.5*pi, r"\pi": pi},
                                       {"-1": -1, r"-\tfrac{1}{2}": -0.5, r"\,\tfrac{1}{2}": 0.5, "1": 1},
                                   ],
                                   axes_labels={r"\phi": [-0.5, 0, 15.5], r"\theta": [0.25, 0, 7.5], "Y_l^m": [0.25, 0, 7.75]})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        l = 50
        m = 25
        data = YlmSurface(domain=[[-pi, pi], [0, pi]], l=l, m=m, scale=[7.5 / pi, 15 / pi, 3.75], thickness=0.5)

        cube = Cube()
        cube.add_mesh_modifier(type="NODES", node_modifier=data)
        coords.add_data(cube)

        data2 = YlmSurfaceReference(domain=[[-pi, pi], [0.01 * pi, 0.99 * pi]], l=l, m=-m,
                                    scale=[7.5 / pi, 15 / pi, 3.75])
        cube2 = Cube()
        cube2.add_mesh_modifier(type="NODES", node_modifier=data2)
        coords.add_data(cube2)

        self.t0 = t0

    def spherical_harmonics_l_200(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[10, -20, 10])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # create comparison in 3D coordinate system  between regular spherical harmonics node and recursive spherical harmonics node
        coords = CoordinateSystem2(dimension=3, location=[0, 0, 0], name="CoordinateSystem3D",
                                   lengths=[15, 15, 7.5], colors=["text", "text", "text"],
                                   domains=[[-pi, pi], [0, pi], [-1, 1]], tic_label_digits=[1, 2, 1],
                                   tic_labels=[
                                       {r"-\pi": -pi, r"\pi": pi},
                                       {r"\,\tfrac{1}{2}\pi": 0.5 * pi, r"\pi": pi},
                                       {"-1": -1, r"-\tfrac{1}{2}": -0.5, r"\,\tfrac{1}{2}": 0.5, "1": 1},
                                   ],
                                   axes_labels={r"\phi": [-0.5, 0, 15.5], r"\theta": [0.25, 0, 7.5],
                                                "Y_l^m": [0.25, 0, 7.75]})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        m=150
        data = YlmSurface_200(domain=[[-pi, pi], [0, pi]], l=200, m=m, scale=[7.5 / pi, 15 / pi, 3.75], thickness=0.5)

        cube = Cube()
        cube.add_mesh_modifier(type="NODES", node_modifier=data)
        coords.add_data(cube)

        self.t0 = t0

    def coordinate_system_3d(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[10, -20, 10])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        coords = CoordinateSystem2(dimension=3, location=[0, 0, 0], name="CoordinateSystem3D",
                                   lengths=[15, 7.5, 7.5], colors=["text", "text","text"],
                                   domains=[[-1, 1],[0,1], [-1, 1]], tic_label_digits=[1, 2,1],
                                   tic_labels=[
                                       {"-1": -1, "-0.5": -0.5, "0.5": 0.5, "1": 1},
                                       {"0.25": 0.25, "0.5": 0.5, "0.75":0.75, "1": 1},
                                       {"-1":-1,"-0.5": -0.5, "0.5": 0.5,"1":1},
                                   ],
                                   axes_labels={"x": [-0.5, 0, 15.5], "y":[0.25,0,7.5], "P_l^m": [0.25, 0, 7.75]})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        data = PlmSurface(domain=[[-1,1],[0,1]],l=2,m=1,scale=[7.5,7.5,3.75])

        cube = Cube()
        cube.add_mesh_modifier(type="NODES",node_modifier=data)
        coords.add_data(cube)

        self.t0 = t0

    def associatedLegendreP(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -20, 0])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        coords = CoordinateSystem2(location=[-5,0,0],name="CoordinateSystemAssociatedLegendreFunctions",
                                   lengths=[4,6],colors=["text","text"],
                                   domains=[[-1,1],[-3,3]],tic_label_digits=[0,0],
                                   tic_labels=[
                                        "AUTO",
                                        "AUTO",
                                   ],
                                   n_tics=[1,1],
                                   axes_labels={"x":[-0.25,0,4.25],"P_l":[0.25,0,6.5]})
        t0 = 0.5 + coords.appear(begin_time=t0,transition_time=5)

        cube = Cube(name="AssociatedLegendreFunctions")
        modifier = AssociatedLegendreP(l_range=range(1,6),m=1,begin_time=t0,transition_time=10,scale=[2,1,1])
        cube.add_mesh_modifier(type="NODES", node_modifier=modifier)
        ibpy.set_parent(cube,coords)
        cube.appear(begin_time=0,transition_time=0)
        t0 = t0+10.5

        coords = CoordinateSystem2(location=[0, 0, 0], name="CoordinateSystemAssociatedLegendreFunctions",
                                   lengths=[4, 6], colors=["text", "text"],
                                   domains=[[-1, 1], [-20, 20]], tic_label_digits=[0, 0],
                                   tic_labels=[
                                       "AUTO",
                                       "AUTO",
                                   ],
                                   n_tics=[1,1],
                                   axes_labels={"x": [-0.25, 0, 4.25], "P_l": [0.25, 0, 6.5]})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        cube = Cube(name="AssociatedLegendreFunctions")
        modifier = AssociatedLegendreP(l_range=range(2, 7), m=2, begin_time=t0, transition_time=10, scale=[2, 1, 6/40])
        cube.add_mesh_modifier(type="NODES", node_modifier=modifier)
        ibpy.set_parent(cube, coords)
        cube.appear(begin_time=0, transition_time=0)
        t0 = t0+10.5

        coords = CoordinateSystem2(location=[5, 0, 0], name="CoordinateSystemAssociatedLegendreFunctions",
                                   lengths=[4, 6], colors=["text", "text"],
                                   domains=[[-1, 1], [-0.5, 0.5]], tic_label_digits=[0, 1],
                                   tic_labels=[
                                       "AUTO",
                                       "AUTO",
                                   ],
                                   n_tics=[1,1],
                                   axes_labels={"x": [-0.25, 0, 4.25], "P_{-l}": [0.25, 0, 6.5]})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        cube = Cube(name="AssociatedLegendreFunctions")
        modifier = AssociatedLegendreP(l_range=range(2, 6), m=-2, begin_time=t0, transition_time=10,
                                       scale=[2, 1,12])
        cube.add_mesh_modifier(type="NODES", node_modifier=modifier)
        ibpy.set_parent(cube, coords)
        cube.appear(begin_time=0, transition_time=0)
        t0 = t0+10.5

        self.t0 = t0

    def legendreP(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -20, 0])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        coords = CoordinateSystem2(location=[0,0,0],name="CoordinateSystemLegendrePolynomials",
                                   lengths=[15,7.5],colors=["text","text"],
                                   domains=[[-1,1],[-0.5,0.5]],tic_label_digits=[1,1],
                                   tic_labels=[
                                        {"-1":-1,"-0.5":-0.5,"0.5":0.5,"1":1},
                                        {"-0.5":-0.5,"0.5":0.5},
                                   ],
                                   axes_labels={"x":[-0.5,0,10.5],"P_l":[0.25,0,7.75]})
        t0 = 0.5 + coords.appear(begin_time=t0,transition_time=5)

        cube = Cube(name="LegendrePolynomials")
        modifier = LegendrePolynomials(l_range =range(50,201,50),begin_time=t0,transition_time=10)
        cube.add_mesh_modifier(type="NODES", node_modifier=modifier)
        ibpy.set_parent(cube,coords)
        cube.appear(begin_time=0,transition_time=0)

        self.t0 = t0

    def numberline(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -20, 0])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        numberline = NumberLine2(location=[0, 0, -4],
                                 length=7,
                                 emission=0.3,
                                 include_zero=False,
                                 tic_label_digits=0,
                                 aligned="center")

        t0 = 0.5 + numberline.grow(begin_time=t0,transition_time=3)
        self.t0 = t0

    def book(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        t0 = ibpy.set_hdri_strength(0.1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -20, 0])
        ibpy.set_camera_lens(lens=45)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # title = SimpleTexBObject(r"\text{A simple book animation}", aligned='center',
        #                          text_size='large',
        #                          color='example', location=[0, 0, 4])
        # t0 = 0.5 + title.write(begin_time=t0, transition_time=1)
        book =Book(pages=10)
        t0=0.5+book.grow(begin_time=0,transition_time=5)
        t0=0.5+book.open(begin_time=t0,transition_time=1)
        for i in range(5):
            t0  = 0.5  + book.turn_page(i,begin_time=t0,transition_time=1)
        self.t0 = t0

    def move_letters_and_move_copies_of_letters(self):
        t0 = 0
        display = Display(flat=True, scales=[3, 3], location=[4.5, 0, -1.25],
                          number_of_lines=20)
        t0 = 0.5+ display.appear(begin_time=t0, transition_time=1, nice_alpha=True)

        rec = SimpleTexBObject(r"{T_\text{rec}\over T_0}={a_0\over a_\text{rec}}\approx {2.725\,K\over 3000\,"
                               r"K}\approx {1\over 1100}", color='text', center_letters_origin=True)
        t0 = 0.5 + display.write_text_in(rec, line=12.5, begin_time=t0, transition_time=1, indent=1,
                                         letter_set=[0, 1, 4, 6, 2, 3, 5, 7, 9, 12, 10, 8, 11, 13, 14])

        rec.move_letters_to(rec, [11, 13, 14, 1, 4, 6], [1, 4, 6, 11, 13, 14], begin_time=t0, transition_time=1)
        t0 = 0.5 + rec.move_letters_to(rec, [0, 8], [8, 0], offsets=[[0, 0.076, 0], [0, 0, 0]], begin_time=t0,
                                       transition_time=1)
        rec.move_letters_to(rec, [3, 9], [9, 3], offsets=[[0, 0.076, 0], [0, 0, 0]], begin_time=t0, transition_time=1)
        t0 = 0.5 + rec.move_letters_to(rec, [5, 12], [12, 5], begin_time=t0, transition_time=1)

        t0 = 0.5 + rec.write(
            letter_set=[15, 16, 18, 19, 21, 24, 26, 22, 17, 20, 23, 25, 27, 28, 31, 32, 29, 30, 33, 34], begin_time=t0,
            transition_time=1)

        line = SimpleTexBObject(
            r"t_\text{rec} = \int\limits_0{{\rm d} a\over \sqrt{{\Omega_m\over a}+\Omega_\Lambda a +\Omega_r "
            r"a^2}}H_0^{-1}")
        # time of recombination
        letter_set=[0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 7, 16, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 26, 28]

        t0 = 0.5 + display.write_text_in(line, letter_set=letter_set, line=4, indent=0.5,
                                             begin_time=t0, transition_time=1)

        copies = rec.get_copy_of_letters(letter_list=[29, 30, 31, 32, 33, 34])
        scale_box = TextScaleBox(children=copies,name="ScaleBox")
        scale_box.appear(begin_time=t0, transition_time=0)
        scale_box.move_to(target_location=[-1.2472, 0.617, 0.025], begin_time=t0, transition_time=1)
        t0 = 0.5 + scale_box.rescale(rescale=0.7, begin_time=t0, transition_time=1)

        self.t0 =t0

    def bderivation_complete_square(self):
        """
        Solve x^2+6x-7=0 by completing the square -- a BDerivation feature tour:

        * a term flies across the equals sign and flips its sign
          (``map={"-7": "7"}`` merges the two glyphs into one),
        * ``+9`` is inserted on BOTH sides as fresh material
          (``map={None: "+9@all"}`` -- forced appearance, ``@all`` selects
          every occurrence),
        * arithmetic is evaluated IN PLACE (``mode='replace'`` morphs
          ``7+9`` into ``16`` without leaving the line),
        * the square is completed in place as a group morph,
        * highlights flash the term that is about to move.
        """
        t0 = 0
        ibpy.set_camera_location(location=[0, -16, 0])
        empty = EmptyCube(location=Vector((0, 0, -0.4)))
        ibpy.set_camera_view_to(empty)

        title = SimpleTexBObject(r"\text{Completing the square}", aligned='center',
                                 color='example', text_size='large', location=[0, 0, 3.6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1)

        d = BDerivation(r"x^2+6\cdot x-7 = 0", location=[-2.5, 0, 2],
                        line_spacing=Vector((0, 0, -1)), name="CompleteSquare")
        t0 = 0.5 + d.write(begin_time=t0, transition_time=1)

        # -7 flies over the equals sign and becomes +7 (the 0 vanishes)
        t0 = 0.5 + d.step(r"x^2+6\cdot x = 7", map={"-7": "7"}, highlight="-7",
                          begin_time=t0, transition_time=1.5)

        # add 9 to both sides: two fresh terms appear simultaneously
        t0 = 0.5 + d.step(r"x^2+6\cdot x+9 = 7+9", map={None: "+9@all"},
                          begin_time=t0, transition_time=1.5)

        # evaluate the right side in place
        t0 = 0.5 + d.step(r"x^2+6\cdot x+9 = 16", mode='replace',
                          map={"7+9": "16"}, highlight="7+9",
                          begin_time=t0, transition_time=1)

        # complete the square in place: the whole left side morphs as a group
        t0 = 0.5 + d.step(r"(x+3)^2 = 16", mode='replace',
                          map={r"x^2+6\cdot x+9": "(x+3)^2"},
                          begin_time=t0, transition_time=1.5)

        # take the square root onto a new line
        t0 = 0.5 + d.step(r"x+3 = \pm 4",
                          map={"(x+3)^2": "x+3", "16": r"\pm 4"},
                          begin_time=t0, transition_time=1.5)

        # move the 3 across: it arcs over the equals sign and flips its sign
        t0 = 0.5 + d.step(r"x = -3\pm 4", map={"+3": "-3"}, highlight="+3",
                          begin_time=t0, transition_time=1.5)

        self.t0 = t0

    def bderivation_geometric_sum(self):
        """
        Derive the closed form of the geometric sum on a Display -- the
        showcase for pinned flights and CANCELLATION:

        * multiplying by q: ``map={"1": "q@1"}`` pins one flight, the
          identical powers travel automatically,
        * subtracting the two lines: occurrence selectors (``"q@1"``) steer
          every power of q into its NEGATIVE counterpart,
        * the telescoping step uses ``cancel=[...]``: three +/- pairs fly to
          their joint midpoints and annihilate while 1 and -q^4 carry over,
        * factoring out S happens in place, and the final division produces
          a FRACTION (the bar is handled like any other glyph).
        """
        t0 = 0
        display = Display(flat=True, scales=[7, 4], location=[0, 0, -0.5],
                          number_of_lines=14)
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1, nice_alpha=True)

        title = SimpleTexBObject(r"\text{The geometric sum}", aligned='center',
                                 color='example', text_size='large', location=[0, 0, 3])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1)

        d = BDerivation(r"S = 1+q+q^2+q^3", display=display, line=2, line_step=2,
                        indent=2.5, scale=0.7, name="GeometricSum")
        t0 = 0.5 + d.write(begin_time=t0, transition_time=1)

        # multiply by q: the 1 becomes the new lone q, the powers slide on
        t0 = 0.5 + d.step(r"q\cdot S = q+q^2+q^3+q^4",
                          map={"1": "q@1"},
                          begin_time=t0, transition_time=1.5)

        # subtract from the first line: pin every power onto its negative twin
        # note: "-q@0" would be the -q inside "S-q\cdot S"; the bare negative
        # term is occurrence 1
        t0 = 0.5 + d.step(r"S-q\cdot S = 1+q+q^2+q^3-q-q^2-q^3-q^4",
                          map={"q@1": "-q@1",
                               "+q^2": "-q^2",
                               "+q^3": "-q^3",
                               "+q^4": "-q^4"},
                          begin_time=t0, transition_time=2)

        # telescope: the three +/- pairs meet mid-air and annihilate
        t0 = 0.5 + d.step(r"S-q\cdot S = 1-q^4",
                          cancel=[("+q@0", "-q@1"),
                                  ("+q^2", "-q^2"),
                                  ("+q^3", "-q^3")],
                          begin_time=t0, transition_time=2)

        # factor out S in place
        t0 = 0.5 + d.step(r"S\cdot (1-q) = 1-q^4", mode='replace',
                          map={r"S-q\cdot S": r"S\cdot (1-q)"},
                          begin_time=t0, transition_time=1.5)

        # divide: the (1-q) dives under the fraction bar
        t0 = 0.5 + d.step(r"S = {1-q^4\over 1-q}",
                          map={"1-q^4": "1-q^4", "1-q@0": "1-q@1"},
                          highlight="(1-q)",
                          begin_time=t0, transition_time=2)

        self.t0 = t0

    def custom_op_demo(self):
        """
        Blended wave surfaces via a custom MixNode RPN op.
        The make_function factory of the geometry nodes is extended to include custom operators and auxiliary
        expressions. The left grid uses the binary "mix05" operator with a fixed factor of 0.5:
        z = mix(sin(x), cos(y), 0.5). The right grid is a radial ripple built with the ternary
        "mix" operator and two auxiliary functions that are each used more than once:
        r = sqrt(x^2+y^2) feeds both wave operands, and damp = 1/(1+r) serves as both the
        mix factor and the amplitude: z = damp * mix(sin(2r), cos(2r), damp).
        """
        ibpy.set_hdri_background(filename="kloofendal_misty_morning_puresky_4k",
                                 rotation_euler=[pi / 3 * 2, 0, 0], simple=True,
                                 transparent=True)
        ibpy.set_render_engine(
            denoising=False, transparent=True,
            resolution_percentage=100,
            engine='BLENDER_EEVEE', taa_render_samples=128, frame_start=1, exposure=1)
        ibpy.set_camera_location(location=[0, 0, 14])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        carrier = Plane(name='WaveCarrier')
        carrier.appear(begin_time=0, transition_time=0)

        mod = CustomOpDemoModifier()
        carrier.add_mesh_modifier(type='NODES', node_modifier=mod)

        self.t0 = 0

if __name__ == '__main__':
    try:
        example = Examples()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        if len(dictionary) == 1:
            selected_scene = dictionary[0]
        else:
            choice = input("Choose scene:")
            if len(choice) == 0:
                choice = 0
            print("Your choice: ", choice)
            selected_scene = dictionary[int(choice)]

        example.create(name=selected_scene, resolution=[1920, 1080], start_at_zero=True)

        # example.render(debug=True)
        # doesn't work
        # example.final_render(name=selected_scene,debug=False)
    except:
        print_time_report()
        raise ()
