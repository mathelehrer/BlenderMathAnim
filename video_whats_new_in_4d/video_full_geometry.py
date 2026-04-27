import os
from collections import OrderedDict
from time import time

import numpy as np

import interface.ibpy
from addons.solids import get_solid_data
from appearance.textures import get_texture
from compositions.compositions import create_glow_composition, set_alpha_composition
from geometry_nodes.geometry_nodes_modifier import CustomUnfoldModifier, PolyhedronViewModifier, Show120MatricesModifier
from interface import ibpy
from interface.ibpy import Vector, Quaternion, Matrix, set_origin
from interface.interface_constants import CYCLES, BLENDER_EEVEE
from mathematics.groups.coxA3 import CoxA3, COXA3_SIGNATURES
from mathematics.groups.coxA4 import CoxA4
from mathematics.groups.coxB3 import CoxB3, COXB3_SIGNATURES
from mathematics.groups.coxB4 import CoxB4
from mathematics.groups.coxD4 import CoxD4
from mathematics.groups.coxF4 import CoxF4
from mathematics.groups.coxH3 import CoxH3, COXH3_SIGNATURES
from mathematics.groups.coxH4 import CoxH4
from mathematics.geometry.coxeter.diagram_to_matrix import CoxeterDynkinDiagram
from mathematics.mathematica.mathematica import tuples
from geometry_nodes.geometry_nodes_modifier import ExplosionModifier, CrystalModifier
from mathematics.geometry.unfolder import Unfolder, NetMorpher, Unfolder4D2, Morpher4D, NetMorpher4D
from objects.bobject import BObject
from objects.codeparser import CodeParser
from objects.coordinate_system import CoordinateSystem2
from objects.cylinder import Cylinder
from objects.data import Data3D
from objects.display import CodeDisplay
from objects.dynkin_diagram import DynkinDiagram
from objects.empties import EmptyCube
from objects.floor import Floor
from objects.geometry.sphere import Sphere
from objects.logo import LogoFromInstances
from objects.number_line import NumberLine2
from objects.plane import Plane
from objects.polygon import Polygon
from objects.polyhedron import PolyhedronWithModifier, Polyhedron
from objects.text import Text
from perform.scene import Scene
from utils.constants import DEFAULT_ANIMATION_TIME, LOC_FILE_DIR
from utils.utils import print_time_report, to_vector, flatten

pi = np.pi
tau: float = 2 * pi
r2 = np.sqrt(2)
r5 = np.sqrt(5)
EPS = 1e-6

cd_solid_type_dict = {
    "x3x3x": "trunc_octa",
    "o3x3x": "trunc_tetra",
    "x3x3o": "trunc_tetra",
    "x3o3x": "cubocta",
    "o3o3x": "tetra",
    "x3o3o": "tetra",
    "o3x3o": "octa",
    "x3x.x": "prism6",
    "x.x3x": "prism6",
    "o3x.x": "prism3",
    "x.o3x": "prism3",
    "x3o.x": "prism3",
    "x.x3o": "prism3",
    "x3x4x": "trunc_cubocta",
    "o3x4x": "trunc_cube",
    "x3x4o": "trunc_octa",
    "x3o4x": "rhombicubocta",
    "o3o4x": "cube",
    "x3o4o": "octa",
    "o3x4o": "cubocta",
    "x4x.x": "prism8",
    "x.x4x": "prism8",
    "o4x.x": "prism4",
    "x.o4x": "prism4",
    "x4o.x": "prism4",
    "x.x4o": "prism4",
    "x3x5x": "trunc_icosidodeca",
    "o3x5x": "trunc_dodeca",
    "x3x5o": "trunc_icosa",
    "x3o5x": "rhombicosidodeca",
    "o3o5x": "dodeca",
    "x3o5o": "icosa",
    "o3x5o": "icosidodeca",
    "x5x.x": "prism10",
    "x.x5x": "prism10",
    "o5x.x": "prism5",
    "x.o5x": "prism5",
    "x5o.x": "prism5",
    "x.x5o": "prism5",
    "x.x.x": "prism4",
}

def normalize_to_unity(vertices):
    l = to_vector(vertices[0]).length
    return [to_vector(vert) / l for vert in vertices]


def get_z_location(x_range, z_range, x, offset=0):
    return z_range[0] + (z_range[1] - z_range[0]) * np.log(float(x / x_range[0])) / np.log(
        float(x_range[1] / x_range[0])) + offset


def get_z_location_lin(x_range, z_range, x, offset=0):
    return z_range[0] + (z_range[1] - z_range[0]) / (x_range[1] - x_range[0]) * (x - x_range[0]) + offset


def make_exploding_shell(druse, outer_cell, start_time=0, crack_density=1,
                         src="rock3.jpg", crack_scale=0.01, crack_limit=0, crack_seed=0, center=Vector()):
    # set outer shell
    if druse is not None:
        shell = ibpy.get_child_with_name(druse, "Cell" + str(outer_cell), starts_with=True)
        shell.rescale(rescale=1.01, begin_time=start_time, transition_time=0)
    else:
        shell = outer_cell
    ibpy.apply_scale(shell)
    shell.add_mesh_modifier(type="SOLIDIFY", thickness=0.1)
    ibpy.apply_modifiers(shell)
    explosion_modifier = ExplosionModifier(unpublished=False, begin_time=int(np.ceil(start_time)),
                                           elements_scale=0.999, crack_density=crack_density, crack_seed=crack_seed,
                                           crack_scale=crack_scale,
                                           material=get_texture("image", src=src,
                                                                projection="BOX", projection_blend=0.01))
    shell.add_mesh_modifier(type="NODES", node_modifier=explosion_modifier)
    ibpy.apply_modifiers(shell)

    ibpy.separate(shell, type="LOOSE")

    start = time()
    actives = []
    passives = []
    if druse is not None:
        for i, child in enumerate(druse.ref_obj.children):
            ibpy.set_origin(child, type="ORIGIN_GEOMETRY")
            location = ibpy.get_location(child)
            if location.dot(Vector([0, 0, 1])) > crack_limit:
                actives.append(child)
            else:
                passives.append(child)
    else:
        objects = ibpy.get_objects_from_name(outer_cell.name)
        for o in objects:
            ibpy.set_origin(o, type="ORIGIN_GEOMETRY")
            location = ibpy.get_location(o)
            if (location - center).dot(Vector([0, 0, 1])) > crack_limit:
                actives.append(o)
            else:
                passives.append(o)

    # # add light
    # for passive in passives:
    #     if "Cell" + str(outer_cell) in child.name:
    #         pass
    #     else:
    #         ibpy.add_point_light(location=location, energy=100)

    # set rigid bodies
    ibpy.make_rigid_bodies(actives, type="ACTIVE", mass=100, collision_shape="MESH", friction=0.5)
    ibpy.make_rigid_bodies(passives, type="PASSIVE", mass=100, collision_shape="MESH", friction=0.5)
    if druse:
        print("made explosion shell for", druse.name)
    else:
        print("made explosion shell for ", outer_cell.name)


def show_panel(elements, scales=1, begin_time=0,
               transition_time=DEFAULT_ANIMATION_TIME,
               show_time=9,
               poly_shift=Vector(),
               dynkin_shift=Vector(),
               global_shift=Vector(),
               dynkin_scale=1, columns=1, **kwargs):
    # building blocks panel
    t0 = begin_time
    global_shift = to_vector(global_shift)
    poly_shift = to_vector(poly_shift)
    dynkin_shift = to_vector(dynkin_shift)
    if isinstance(elements, str):
        elements = [elements]

    if isinstance(scales, float):
        scales = [scales] * len(elements)

    panel = Plane(u=[-1.75, 1.75], v=[-6.9, 6.9], normal=Vector([0, -1, 0]), color="frosty", solid=0.1, offset=-1)
    panel.appear(begin_time=t0, transition_time=0)
    panel.move(direction=[-15, -0.5, 0], begin_time=0, transition_time=0)

    # determine the spacing of the elements
    dy = 13.5 / (len(elements) // columns + 1)
    if len(elements) == 0:
        dy = 7
    if columns > 1:
        dx = 3 / columns
    else:
        dx = 0

    elements_per_column = len(elements) // columns
    for i, element in enumerate(elements):

        dia = DynkinDiagram.from_string(element, location=global_shift + Vector(
            [-0.5 + dx * (i // elements_per_column), 8.5 - 1.03 * dy * (i % elements_per_column + 1),
             0]) + dynkin_shift,
                                        rotation_euler=[-pi / 2, 0, 0],
                                        scale=0.25 * dynkin_scale, no_threes=True, **kwargs)
        dia.appear(begin_time=t0, transition_time=0)
        ibpy.set_parent(dia, panel)

        if len(element) == 1:
            vertices = [
                Sphere(r=0.1, mesh_type="ico", color="red", resolution=1, location=Vector([x, 14 - dy * (i + 1), 0]))
                for x in [-1, 1]]
            [v.appear(begin_time=t0, transition_time=0) for v in vertices]
            [ibpy.set_parent(v, panel) for v in vertices]

            edge = Cylinder.from_start_to_end(start=Vector([-1, 0, 0]), end=Vector([1, 0, 0]), color="example",
                                              thickness=0.5)
            edge.grow(begin_time=t0, transition_time=0)
            ibpy.set_parent(edge, panel)
        if len(element) == 3:
            location = global_shift + Vector([1.5 + dx * (i // elements_per_column),
                                              7.5 - dy * (i % elements_per_column + 1),
                                              0.05]) + poly_shift
            # polygons here
            if element == "x3x":
                poly = Polygon(vertices=[Vector([np.sin(-tau / 6 * i), np.cos(tau / 6 * i), 0]) for i in range(6)],
                               location=location, solid=0.1,
                               color="hexagon")
            elif element in ["o3x", "x3o"]:
                poly = Polygon(vertices=[Vector([np.sin(-tau / 3 * i), np.cos(tau / 3 * i), 0]) for i in range(3)],
                               location=location, color="triangle")
            elif element in ["x4x"]:
                poly = Polygon(vertices=[Vector([np.sin(-tau / 8 * i), np.cos(tau / 8 * i), 0]) for i in range(8)],
                               location=location, color="octagon")
            elif element in ["x.x"]:
                poly = Polygon(vertices=[Vector([np.sin(-tau / 4 * i), np.cos(tau / 4 * i), 0]) for i in range(4)],
                               location=location, color="square")
            elif element in ["x4o", "o4x"]:
                poly = Polygon(vertices=[Vector([np.sin(-tau / 4 * i), np.cos(tau / 4 * i), 0]) for i in range(4)],
                               location=location, color="tetragon")
            poly.appear(begin_time=t0, transition_time=transition_time)
            ibpy.set_parent(poly, panel)
        elif len(element) > 3:
            # three-dimensional objects

            solid_type = cd_solid_type_dict[element]
            poly = Polyhedron.from_points(solid_type=solid_type,
                                          name="Ledgend" + str(i),
                                          location=global_shift + Vector([1.5 + dx * (i // elements_per_column),
                                                                          7.5 - dy * (i % elements_per_column + 1),
                                                                          1]) + poly_shift,
                                          color=solid_type, scale=scales[i])
            poly.appear(begin_time=0, transition_time=0)
            poly.rotate(rotation_euler=[0, show_time / 10 * tau, 0], begin_time=t0, transition_time=show_time)
            ibpy.set_parent(poly, panel)

    t0 = show_time + panel.move(direction=[4, 0, 0], begin_time=t0, transition_time=transition_time)
    panel.move(direction=[-4, 0, 0], begin_time=t0, transition_time=transition_time)
    return begin_time + show_time + 2 * transition_time


class VideoFullGeometry(Scene):
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('almost_platonic', {'duration': 15}),
            ('trailer_h4_panel', {'duration': 15}),
            ('trailer_v1440', {'duration': 15}),
            ('trailer_v120', {'duration': 15}),
            ('trailer_net_merger', {'duration': 15}),
            ('trailer_overlay', {'duration': 60}),
            ('trailer_c120', {'duration': 15}),
            ('crystal_texture', {'duration': 10}),
            ('h4_net_fun', {'duration': 60}),
            ('platonic_explosion', {'duration': 20}),
            ('druse_explosion', {'duration': 15}),
            ('a_family', {'duration': 15}),
            ('family_a1', {'duration': 12}),
            ('family_a2', {'duration': 35}),
            ('family_b2', {'duration': 40}),
            ('family_h2', {'duration': 45}),
            ('family_a1xa1', {'duration': 25}),
            ('family_a1xa1xa1', {'duration': 30}),
            ('family_a2xa1', {'duration': 32}),
            ('family_a4_nets', {'duration': 60}),
            ('family_b4_nets', {'duration': 85}),
            ('family_h4_nets_a', {'duration': 85}),
            ('no_crystals', {'duration': 60}),
            ('no_crystals2', {'duration': 60}),
            ('crystals', {'duration': 42}),
            ('crystals2', {'duration': 42}),
            ('family_b4', {'duration': 70}),
            ('family_a4', {'duration': 50}),
            ('family_h3_net', {'duration': 25}),
            ('family_h3', {'duration': 43}),
            ('family_b3_net', {'duration': 24}),
            ('family_b3', {'duration': 43}),
            ('family_f4_nets', {'duration': 50}),
            ('family_f4', {'duration': 50}),
            ('family_d4_nets', {'duration': 40}),
            ('family_d4_and_b4', {'duration': 5}),
            ('demi_tesseract', {'duration': 15}),
            ('tesseract_to_16_cell', {'duration': 30}),
            ('family_d4', {'duration': 40}),
            ('family_h4', {'duration': 85}),
            ('family_a3', {'duration': 55}),
            ('family_a3', {'duration': 55}),
            ('family_a3_outtake', {'duration': 35}),
            ('show_panel_a3', {'duration': 20}),
            ('show_panel_b3', {'duration': 20}),
            ('morphing_stereo_3d', {'duration': 90}),
            ('logo', {'duration': 60}),
            ('documentation', {'duration': 35}),
            ('documentation2', {'duration': 35}),
            ('documentation3', {'duration': 35}),
            ('short_dynkin_magic', {'duration': 42}),
            ('short_h4_family', {'duration': 42}),
            ('lifting_cubes', {'duration': 36}),
            ('dimensions_intro', {'duration': 20}),
            ('a_series_intro', {'duration': 25}),
            ('b_series_intro', {'duration': 25}),
            ('d_series_intro', {'duration': 10}),
            ('h_family_intro', {'duration': 25}),
            ('f4_intro', {'duration': 5}),
            ('a4_normal_constraints', {'duration': 32}),
            ('a4_normal_explicit', {'duration': 12}),
            ('a4_python_script', {'duration': 35}),
            ('a4_image_generation', {'duration': 5}),
            ('a4_seed_point', {'duration': 15}),
            ('a4_cell_normal', {'duration': 25}),
            ('a4_cell_projection', {'duration': 10}),
            ('a4_display_cell', {'duration': 15}),
            ('a4_cell_equation', {'duration': 30}),
            ('a4_neighbors', {'duration': 5}),
            ('cd_wizardry', {'duration': 20}),
            ('cd_wizardry2', {'duration': 15}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def almost_platonic(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)
        set_alpha_composition()

        dia1 = DynkinDiagram.from_string("o3x4x3o", location=[-8, 0, 6], no_threes=True)
        dia1.appear(begin_time=t0, transition_time=1)

        dia2 = DynkinDiagram.from_string("o3x3x3o", location=[8, 0, 6], no_threes=True)
        t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=1)

        unfolder1 = Unfolder4D2(CoxF4, "o3x4x3o")
        ap1 = unfolder1.create_net(scale_elements=0.99, scale=0.4,
                                   location=[-6, 0, -0.75])
        ap1.appear(begin_time=t0, transition_time=5, sequentially=True)
        ap1.rotate(rotation_euler=[0, 0, tau], begin_time=t0, transition_time=10)

        unfolder2 = Unfolder4D2(CoxA4, "o3x3x3o")
        ap2 = unfolder2.create_net(scale_elements=0.99, scale=0.45,
                                   location=[6, 0, 0])
        ap2.appear(begin_time=t0, transition_time=5, sequentially=True)
        t0 = 0.5 + ap2.rotate(rotation_euler=[0, 0, tau], begin_time=t0, transition_time=10)

        self.t0 = t0

    def trailer_h4_panel(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        t0 = 0.5 + show_panel([
            "x3x5x", "x3x5o", "x3o5x", "o3x5x", "o3o5x", "o3x5o", "x3o5o",
            "x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o",
            "x3x.x", "x3o.x", "x5x.x", "x5o.x"
        ],
            scales=[0.096, 0.09, 0.122, 0.146, 0.303, 0.332, 0.303, 0.24,
                    0.13, 0.4, 0.377, 0.445,
                    0.42, 0.65, 0.34, 0.65],
            dynkin_scale=0.75, columns=2,
            show_time=9,
            begin_time=2, transition_time=1,
            dynkin_shift=Vector([-0.15, -1, 0]),
            poly_shift=Vector([-1.55, -0.9, 0]), dynkin_label_shift=Vector([-0, 0, 0.6]))

        self.t0 = t0

    def trailer_v1440(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -50, 0])
        ibpy.set_camera_location(location=camera_location)

        v1440_unfolder = Unfolder4D2(CoxH4, "x3x3o5o", rotate_to_south_pole=False)
        net = v1440_unfolder.create_net(scale_elements=0.99)
        net.rotate(rotation_euler=[-pi / 2, 0, 0], begin_time=0, transition_time=0)
        net.rotate(rotation_euler=[-pi / 2, 0, 2 * pi], begin_time=t0 + 1.5, transition_time=13.5)
        t0 = 0.5 + net.appear(begin_time=t0, transition_time=15, sequentially=True)
        self.t0 = t0

    def trailer_v120(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -50, 0])
        ibpy.set_camera_location(location=camera_location)

        v120_unfolder = Unfolder4D2(CoxA4, "x3x3x3x", rotate_to_south_pole=False)
        net = v120_unfolder.create_net(scale_elements=0.99, cells_sorted=True)

        quat_0 = Quaternion([-0.421, 0.47, -797, 0.138])

        net.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.5 + net.appear(begin_time=t0, transition_time=15, sequentially=True)
        quat_rot = Quaternion(Vector([0, 0, 1]), 0.99 * pi)
        t0 = net.rotate(rotation_quaternion=quat_rot @ quat_0, begin_time=t0, transition_time=10)
        t0 = net.rotate(rotation_quaternion=quat_rot @ quat_rot @ quat_0, begin_time=t0, transition_time=10)
        self.t0 = t0

    def trailer_net_merger(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -50, 0])
        ibpy.set_camera_location(location=camera_location)

        morpher = NetMorpher4D(CoxH4, ["x3x3x5x", "x3x3x5o", "x3x3o5o"], name="MorpherH4",
                               scale=1)

        morpher.morph_sequence(begin_time=t0, transition_time=2, pause=1, scale_elements=0.99, sequentially=True)

        self.t0 = t0

    def trailer_c120(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -45, 0])
        ibpy.set_camera_location(location=camera_location)

        v120_unfolder = Unfolder4D2(CoxH4, "o3o3o5x", color="dodeca")
        net = v120_unfolder.create_net(scale_elements=0.99)
        net.rotate(rotation_euler=[0, 0, 1.9 * pi], begin_time=t0 + 1.5, transition_time=14)
        t0 = 0.5 + net.appear(begin_time=t0, transition_time=15, sequentially=True)

        self.t0 = t0

    def crystal_texture(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)
        icosahedron = Polyhedron.from_points(solid_type="ICOSA", color="crystal_icosa")
        crystal = CrystalModifier(unpublished=False)
        icosahedron.add_mesh_modifier(type="NODES", node_modifier=crystal)
        crystal.transfer_material_from(icosahedron)
        t0 = 0.5 + icosahedron.appear(begin_time=t0, transition_time=1)
        self.t0 = t0

    def h4_net_fun(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)
        morphing_start_time = t0
        morpher_center = [0, 0, 0]

        # second morph sequence
        morpher = NetMorpher4D(CoxH4, ["x3x3x5x", "x3x3o5x", "x3o3o5x", "x3o3o5o"], name="MorpherH4_b",
                               scale=1, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=10, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))

        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 50, transition_time=1)
        print(morphing_start_time)

        self.t0 = t0

    def platonic_explosion(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -60, 0])
        ibpy.set_camera_location(location=camera_location)

        five_cell = Unfolder4D2(group=CoxA4, coxeter_dynkin_label="o3o3o3x", mode="LARGEST_CELL", crystal=True,
                                cracks_count=5)
        five_cell_outer_cell = 3
        five_cell_bob = five_cell.create_stereo_bob(half_way=False, add_cells=[five_cell_outer_cell],
                                                    location=[-25, 0, 10], scale=1.5, limit=0)
        five_cell_bob.grow(begin_time=t0, transition_time=1)

        eight_cell = Unfolder4D2(group=CoxB4, coxeter_dynkin_label="o3o3o4x", mode="LARGEST_CELL", crystal=True,
                                 cracks_count=5)
        eight_cell_outer_cell = 5
        eight_cell_bob = eight_cell.create_stereo_bob(half_way=True, add_cells=[eight_cell_outer_cell],
                                                      location=[-8, 0, 10],
                                                      scale=0.9)
        eight_cell_bob.grow(begin_time=t0, transition_time=1)

        sixteen_cell = Unfolder4D2(group=CoxB4, coxeter_dynkin_label="x3o3o4o", mode="LARGEST_CELL", crystal=True,
                                   cracks_count=5)
        sixteen_cell_outer_cell = 11
        sixteen_cell_bob = sixteen_cell.create_stereo_bob(half_way=True, add_cells=[sixteen_cell_outer_cell],
                                                          location=[9, 0, 10], scale=0.9)
        sixteen_cell_bob.grow(begin_time=t0, transition_time=1)

        c120 = Unfolder4D2(group=CoxH4, coxeter_dynkin_label="o3o3o5x", mode="LARGEST_CELL", crystal=True,
                           cracks_count=5)

        c120_outer_cell = 117
        c120_bob = c120.create_stereo_bob(half_way=True, add_cells=[c120_outer_cell], location=[0, 0, -9],
                                          scale=0.3)
        c120_bob.grow(begin_time=t0, transition_time=1)

        shell_empty = EmptyCube(location=[17, 0, -9])
        c600_shell = Polyhedron.from_points(solid_type="ICOSA", scale=4)
        ibpy.set_parent(c600_shell, shell_empty)
        c600_shell.appear(begin_time=t0, transition_time=1)

        c600 = Unfolder4D2(group=CoxH4, coxeter_dynkin_label="x3o3o5o", mode="LARGEST_CELL", crystal=True,
                           cracks_count=5, south_pole=2, external_rotation=Matrix(np.identity(4, float)))
        cell_removals = [295, 330, 36, 598]
        c600_bob = c600.create_stereo_bob(half_way=True,
                                          location=[17, 0, -9], cell_removals=cell_removals, scale=0.49)
        c600_bob.grow(begin_time=t0, transition_time=1)
        shell_empty.grow(begin_time=t0, transition_time=1)

        c24 = Unfolder4D2(group=CoxF4, coxeter_dynkin_label="o3o4o3x", mode="LARGEST_CELL", crystal=True,
                          cracks_count=5)
        c24_outer_cell = 20
        c24_bob = c24.create_stereo_bob(half_way=True, add_cells=[c24_outer_cell],
                                        location=[25, 0, 10], scale=0.6)
        t0 = 0.5 + c24_bob.grow(begin_time=t0, transition_time=1)

        # prepare dynkin diagrams

        dias = ["o3o3o3x", "o3o3o4x", "x3o3o4o", "o3o3o *b3x", "o3x3o4o", "o3o4o3x", "o3x3o *b3o", "o3o3o5x", "x3o3o5o"]
        locations = [
            Vector([-25, 0, 2]),
            Vector([-8, 0, 2]),
            Vector([9, 0, 2]),
            Vector([9, 0, -2]),
            Vector([26, 0, 2]),
            Vector([26, 0, 0]),
            Vector([26, 0, -4]),
            Vector([0, 0, -18]),
            Vector([17, 0, -18]),

        ]
        for dia, loc in zip(dias, locations):
            diagram = DynkinDiagram.from_string(dia, location=loc, scale=1, no_threes=True)
            diagram.appear(begin_time=t0, transition_time=1)

        t0 += 1.5

        make_exploding_shell(five_cell_bob, five_cell_outer_cell, start_time=t0, src="rock3.jpg", crack_seed=20,
                             crack_density=100, crack_scale=0.05)
        make_exploding_shell(eight_cell_bob, eight_cell_outer_cell, start_time=t0, src="rock2.jpg", crack_density=10,
                             crack_scale=0.02)
        make_exploding_shell(sixteen_cell_bob, sixteen_cell_outer_cell, start_time=t0, src="rock1.jpg",
                             crack_density=20,
                             crack_scale=0.02, crack_seed=20)
        make_exploding_shell(c120_bob, c120_outer_cell, start_time=t0, src="rock2.jpg", crack_density=1,
                             crack_scale=0.05,
                             crack_seed=15)
        make_exploding_shell(None, c600_shell, start_time=t0, src="rock1.jpg", crack_density=10, crack_scale=0.05,
                             crack_seed=15, center=Vector())
        make_exploding_shell(c24_bob, c24_outer_cell, start_time=t0, src="rock3.jpg", crack_density=2)

        ibpy.set_simulation(begin_time=5, transition_time=15)
        t0 += 5

        # rotate objects into view
        druses = [five_cell_bob, eight_cell_bob, sixteen_cell_bob, c120_bob, c600_bob, c24_bob]
        rotations = [
            [2 * pi / 5, 0, pi / 10],
            [2 * pi / 5, 0, 0],
            [2 * pi / 5, 0, 0],
            [2 * pi / 5, 0, 0],
            [2 * pi / 5, 0, -pi / 10],
            [2 * pi / 5, 0, -pi / 10]
        ]

        for i, (druse, rotation) in enumerate(zip(druses, rotations)):
            if i == 4:
                shell_empty.rotate(rotation_euler=rotation, begin_time=t0, transition_time=1)
            t0 = 0.5 + druse.rotate(rotation_euler=rotation, begin_time=t0, transition_time=1)
        self.t0 = t0

    def druse_explosion(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -60, 0])
        ibpy.set_camera_location(location=camera_location)

        v120 = Unfolder4D2(group=CoxA4, coxeter_dynkin_label="x3x3x3x", mode="LARGEST_CELL", crystal=True,
                           cracks_count=5)
        # add cell
        v120_outer_cell = 5
        v120_bob = v120.create_stereo_bob(half_way=True, add_cells=[v120_outer_cell], location=[-25, 0, 10], scale=0.75)
        v120_bob.grow(begin_time=t0, transition_time=1)

        v384 = Unfolder4D2(group=CoxB4, coxeter_dynkin_label="x3x3o4x", mode="LARGEST_CELL", crystal=True,
                           cracks_count=5)
        # add cell
        v384_outer_cell = 9
        v384_bob = v384.create_stereo_bob(half_way=True, add_cells=[v384_outer_cell], location=[-0, 0, 10], scale=0.6)
        v384_bob.grow(begin_time=t0, transition_time=1)

        v14400 = Unfolder4D2(group=CoxH4, coxeter_dynkin_label="o3x3x5x", mode="LARGEST_CELL", crystal=True,
                             cracks_count=5)

        v14400_outer_cell = 30
        v14400_bob = v14400.create_stereo_bob(half_way=True, add_cells=[v14400_outer_cell], location=[25, 0, 10],
                                              scale=0.3)
        v14400_bob.grow(begin_time=t0, transition_time=1)

        v192 = Unfolder4D2(group=CoxD4, coxeter_dynkin_label="x3x3x *b3x", mode="LARGEST_CELL", crystal=True,
                           cracks_count=5)
        v192_outer_cell = 1
        v192_bob = v192.create_stereo_bob(half_way=True, add_cells=[v192_outer_cell],
                                          location=[-13, 0, -10], scale=0.6)
        v192_bob.grow(begin_time=t0, transition_time=1)

        v1152 = Unfolder4D2(group=CoxF4, coxeter_dynkin_label="x3o4x3x", mode="LARGEST_CELL", crystal=True,
                            cracks_count=5)
        v1152_outer_cell = 26
        v1152_bob = v1152.create_stereo_bob(half_way=True, add_cells=[v1152_outer_cell],
                                            location=[13, 0, -10], scale=0.4)
        t0 = 0.5 + v1152_bob.grow(begin_time=t0, transition_time=1)

        # prepare dynkin diagrams

        dias = ["x3x3x3x", "x3x3o4x", "o3x3x5x", "x3x3x *b3x", "x3o4x3x"]
        locations = [
            Vector([-25, 0, 0]),
            Vector([0, 0, 0]),
            Vector([25, 0, 0]),
            Vector([-30, 0, -12]),
            Vector([30, 0, -12]),
        ]
        for dia, loc in zip(dias, locations):
            diagram = DynkinDiagram.from_string(dia, location=loc, scale=1)
            diagram.appear(begin_time=t0, transition_time=1)

        t0 += 1.5

        make_exploding_shell(v120_bob, v120_outer_cell, start_time=t0, src="rock3.jpg", crack_density=2)
        make_exploding_shell(v384_bob, v384_outer_cell, start_time=t0, src="rock1.jpg", crack_density=2)
        make_exploding_shell(v192_bob, v192_outer_cell, start_time=t0, src="rock2.jpg", crack_density=2)
        make_exploding_shell(v14400_bob, v14400_outer_cell, start_time=t0, src="rock1.jpg", crack_density=2)
        make_exploding_shell(v1152_bob, v1152_outer_cell, start_time=t0, src="rock3.jpg", crack_density=2)

        ibpy.set_simulation(begin_time=5, transition_time=30)
        t0 += 5

        # rotate objects into view
        druses = [v120_bob, v384_bob, v14400_bob, v192_bob, v1152_bob]
        rotations = [
            [2 * pi / 5, 0, pi / 10],
            [2 * pi / 5, 0, 0],
            [2 * pi / 5, 0, 0],
            [2 * pi / 5, 0, 0],
            [2 * pi / 5, 0, -pi / 10]
        ]
        for druse, rotation in zip(druses, rotations):
            t0 = 1 + druse.rotate(rotation_euler=rotation, begin_time=t0, transition_time=1)
        self.t0 = t0

    def family_a1(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21.5, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }A_1", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[0, 2],
                           tic_labels={"0": 0, "1": 1, "2": 2}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)

        family = [["x"], ["o"], ]
        dias = []
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                dx = 16 / (max(2, level_count) - 1)
                x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location_lin([0, 2], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
                dias.append(dia2)

        # tell simple story about the one-dimensional case
        mirror_center = Vector([6, 0, 3])
        src_location = mirror_center - Vector([2, 0, 0])
        vertex = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=src_location)
        size = 0.5
        mirror = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_A", normal=Vector([1, 0, 0]),
                       roughness=0.05,
                       shadow=False, solid=0.1, solidify_mode="SIMPLE",
                       smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror.move_to(target_location=mirror_center, begin_time=0, transition_time=0)
        t0 = 0.5 + vertex.grow(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + mirror.grow(begin_time=t0, transition_time=0.5)

        image_location = mirror_center - Vector([-2, 0, 0])
        image = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=image_location)
        t0 = 0.5 + image.grow(begin_time=t0, transition_time=0.5)

        edge = Cylinder.from_start_to_end(start=src_location, end=image_location, color="example", thickness=0.5)
        t0 = 0.5 + edge.grow(begin_time=t0, transition_time=0.5)

        copies = []
        copies.append(vertex.copy())
        copies.append(image.copy())
        copies.append(edge.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copy.move(direction=[-10, 0, 2], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        # degenerate case
        vertex.move(direction=[2, 0, 0], begin_time=t0, transition_time=1)
        image.move(direction=[-2, 0, 0], begin_time=t0, transition_time=1)
        edge.move(direction=[2, 0, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + edge.rescale(rescale=[1, 1, 0], begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_a2(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21.5, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }A_2", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[2, 6],
                           tic_labels={"2": 2, "3": 3, "4": 4, "5": 5, "6": 6}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)

        family = [["x3x"], ["o3x"], ]
        dias = []
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                dx = 16 / (max(2, level_count) - 1)
                x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location_lin([2, 6], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
                dias.append(dia2)

        # tell simple story about the one-dimensional case
        mirror_center = Vector([6, 0, 2])
        size = 4
        mirror_a = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A", normal=Vector([1, 0, 0]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_a.move_to(target_location=mirror_center, begin_time=0, transition_time=0)

        mirror_b = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A",
                         normal=Vector([np.cos(pi / 3), 0, np.sin(pi / 3)]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_b.move_to(target_location=mirror_center, begin_time=0, transition_time=0)
        mirrors = [mirror_a, mirror_b]

        src_location = 2 * Vector([np.sin(pi / 6), 0, - np.cos(pi / 6)])
        vertex = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - src_location)
        for mirror in mirrors:
            mirror.grow(begin_time=t0, transition_time=1)
        t0 = 1.5 + t0
        t0 = 0.5 + vertex.grow(begin_time=t0, transition_time=0.5)

        appear_order = [5, 1, 4, 3, 2]
        src_locations = [
            2 * Vector([np.sin(pi / 6 + i * pi / 3), 0, -np.cos(pi / 6 + i * pi / 3)]) for i in appear_order]

        sources = [vertex]
        for location in src_locations:
            sphere = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - location)
            t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=1)
            sources.append(sphere)

        src_locations.insert(0, src_location)

        edges = [Cylinder.from_start_to_end(
            start=mirror_center - 2 * Vector([np.sin(pi / 6 + i * pi / 3), 0, -np.cos(pi / 6 + i * pi / 3)]),
            end=mirror_center - 2 * Vector(
                [np.sin(pi / 6 + (i + 1) % 6 * pi / 3), 0, -np.cos(pi / 6 + (i + 1) % 6 * pi / 3)]),
            color="example", thickness=0.5) for i in range(6)]

        for edge in edges:
            t0 = 0.5 + edge.grow(begin_time=t0, transition_time=1)

        # reindex src_locations
        src_locations2 = [
            2 * Vector([np.sin(pi / 6 + i * pi / 3), 0, -np.cos(pi / 6 + i * pi / 3)]) for i in range(6)]
        hexagon = Polygon(vertices=[mirror_center - src_location for src_location in src_locations2], color="hexagon")
        t0 = 0.5 + hexagon.appear(begin_time=t0, transition_time=1)

        to_copies = sources + edges + [hexagon]
        copies = []
        for to_copy in to_copies:
            copies.append(to_copy.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copy.move(direction=[-10, 0, 2], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        pairs = [[0, 2], [1, 3], [4, 5]]
        pair_locations = [[0, 1], [5, 4], [3, 2]]
        new_source_locations = [Vector()] * 6
        for pair, loc in zip(pairs, pair_locations):
            center = mirror_center - (
                    Vector([np.sin(pi / 6 + loc[0] * pi / 3), 0, -np.cos(pi / 6 + loc[0] * pi / 3)]) +
                    Vector([np.sin(pi / 6 + loc[1] * pi / 3), 0, -np.cos(pi / 6 + loc[1] * pi / 3)])
            )
            sources[pair[0]].move_to(target_location=center, begin_time=t0, transition_time=1)
            sources[pair[1]].move_to(target_location=center, begin_time=t0, transition_time=1)
            new_source_locations[pair[0]] = center
            new_source_locations[pair[1]] = center

        # transform edges
        r = float(np.sqrt(3))
        for i in [0, 2, 4]:
            edge = edges[i]
            new_start = mirror_center - r * Vector([np.sin(pi / 3 + i * pi / 3), 0, -np.cos(pi / 3 + i * pi / 3)])
            new_end = mirror_center - r * Vector([np.sin(pi / 3 + i * pi / 3), 0, -np.cos(pi / 3 + i * pi / 3)])
            # lift the degeneracy between new_start and new_end to avoid rotations
            new_start = new_start + 0.001 * (edge.start - new_start)
            new_end = new_end + 0.001 * (edge.end - new_end)
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        for i in [1, 3, 5]:
            edge = edges[i]
            l = i // 2
            new_start = mirror_center - r * Vector(
                [np.sin(pi / 3 + 2 * l * pi / 3), 0, -np.cos(pi / 3 + 2 * l * pi / 3)])
            new_end = mirror_center - r * Vector(
                [np.sin(pi / 3 + (l + 1) % 3 * 2 * pi / 3), 0, -np.cos(pi / 3 + (l + 1) % 3 * 2 * pi / 3)])
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        # transform polygon
        trafo = lambda i: mirror_center - r * Vector(
            [np.sin(pi / 3 + (i // 2 + 1) * 2 * pi / 3), 0, -np.cos(pi / 3 + (i // 2 + 1) * 2 * pi / 3)])
        hexagon.change_color(new_color="triangle", begin_time=t0, transition_time=1)
        t0 = 0.5 + hexagon.index_transform_mesh(trafo, begin_time=t0, transition_time=1)

        to_copies = sources + edges + [hexagon]
        copies = []
        for to_copy in to_copies:
            copies.append(to_copy.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copies[-1].change_color(new_color="triangle", begin_time=t0, transition_time=0)
            copy.move(direction=[-10, 0, -5], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        # building blocks panel

        t0 = 0.5 + show_panel("x", begin_time=t0, transition_time=1)
        self.t0 = t0

    def family_b2(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21.5, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }B_2", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[3, 8],
                           tic_labels={"3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)

        family = [["x4x"], ["o4x"], ]
        dias = []
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                dx = 16 / (max(2, level_count) - 1)
                x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location_lin([3, 8], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
                dias.append(dia2)

        # tell simple story about the one-dimensional case
        mirror_center = Vector([6, 0, 2])
        size = 4
        mirror_a = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A", normal=Vector([1, 0, 0]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_a.move_to(target_location=mirror_center, begin_time=0, transition_time=0)

        mirror_b = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A",
                         normal=Vector([np.cos(pi / 4), 0, np.sin(pi / 4)]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_b.move_to(target_location=mirror_center, begin_time=0, transition_time=0)
        mirrors = [mirror_a, mirror_b]

        src_location = 2 * Vector([np.sin(pi / 8), 0, - np.cos(pi / 8)])
        vertex = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - src_location)
        for mirror in mirrors:
            mirror.grow(begin_time=t0, transition_time=1)
        t0 = 1.5 + t0
        t0 = 0.5 + vertex.grow(begin_time=t0, transition_time=0.5)

        appear_order = [1, 7, 6, 2, 3, 4, 5]
        src_locations = [
            2 * Vector([np.sin(pi / 8 + i * pi / 4), 0, -np.cos(pi / 8 + i * pi / 4)]) for i in appear_order]

        sources = [None] * 8
        sources[0] = vertex
        for idx, location in zip(appear_order, src_locations):
            sphere = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - location)
            t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=1)
            sources[idx] = sphere

        src_locations.insert(0, src_location)

        edges = [Cylinder.from_start_to_end(
            start=mirror_center - 2 * Vector([np.sin(pi / 8 + i * pi / 4), 0, -np.cos(pi / 8 + i * pi / 4)]),
            end=mirror_center - 2 * Vector(
                [np.sin(pi / 8 + (i + 1) % 8 * pi / 4), 0, -np.cos(pi / 8 + (i + 1) % 8 * pi / 4)]),
            color="example", thickness=0.5) for i in range(8)]

        for edge in edges:
            t0 = 0.5 + edge.grow(begin_time=t0, transition_time=1)

        # reindex src_locations
        src_locations2 = [
            2 * Vector([np.sin(pi / 8 + i * pi / 4), 0, -np.cos(pi / 8 + i * pi / 4)]) for i in range(8)]
        octagon = Polygon(vertices=[mirror_center - src_location for src_location in src_locations2], color="octagon")
        t0 = 0.5 + octagon.appear(begin_time=t0, transition_time=1)

        to_copies = sources + edges + [octagon]
        copies = []
        for to_copy in to_copies:
            copies.append(to_copy.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copy.move(direction=[0, 0.1, 0], begin_time=0, transition_time=0)
            copy.move(direction=[-10, 0, 2], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]
        new_source_locations = [Vector()] * 8
        for pair in pairs:
            loc = pair
            center = mirror_center - (
                    Vector([np.sin(pi / 8 + loc[0] * pi / 4), 0, -np.cos(pi / 8 + loc[0] * pi / 4)]) +
                    Vector([np.sin(pi / 8 + loc[1] * pi / 4), 0, -np.cos(pi / 8 + loc[1] * pi / 4)])
            )
            sources[pair[0]].move_to(target_location=center, begin_time=t0, transition_time=1)
            sources[pair[1]].move_to(target_location=center, begin_time=t0, transition_time=1)
            new_source_locations[pair[0]] = center
            new_source_locations[pair[1]] = center

        # transform edges
        r = 2 * float(np.cos(pi / 8))
        for i in [0, 2, 4, 6]:
            edge = edges[i]
            new_start = mirror_center - r * Vector([np.sin(pi / 4 + i * pi / 4), 0, -np.cos(pi / 4 + i * pi / 4)])
            new_end = mirror_center - r * Vector([np.sin(pi / 4 + i * pi / 4), 0, -np.cos(pi / 4 + i * pi / 4)])
            # lift the degeneracy between new_start and new_end to avoid rotations
            new_start = new_start + 0.001 * (edge.start - new_start)
            new_end = new_end + 0.001 * (edge.end - new_end)
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        for i in [1, 3, 5, 7]:
            edge = edges[i]
            l = i // 2
            new_start = mirror_center - r * Vector(
                [np.sin(pi / 4 + 2 * l * pi / 4), 0, -np.cos(pi / 4 + 2 * l * pi / 4)])
            new_end = mirror_center - r * Vector(
                [np.sin(pi / 4 + (l + 1) % 4 * 2 * pi / 4), 0, -np.cos(pi / 4 + (l + 1) % 4 * 2 * pi / 4)])
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        # transform polygon
        trafo = lambda i: mirror_center - r * Vector(
            [np.sin(pi / 4 + (i // 2 + 1) * 2 * pi / 4), 0, -np.cos(pi / 4 + (i // 2 + 1) * 2 * pi / 4)])
        octagon.change_color(new_color="tetragon", begin_time=t0, transition_time=1)
        t0 = 0.5 + octagon.index_transform_mesh(trafo, begin_time=t0, transition_time=1)

        to_copies = sources + edges + [octagon]
        copies = []
        for to_copy in to_copies:
            copies.append(to_copy.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copy.move(direction=[0, 0.1, 0], begin_time=0, transition_time=0)
            copies[-1].change_color(new_color="tetragon", begin_time=t0, transition_time=0)
            copy.move(direction=[-10, 0, -5], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        # building blocks panel

        t0 = 0.5 + show_panel("x", begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_h2(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21.5, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }H_2", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[4, 10],
                           tic_labels={"10": 10, "9": 9, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)

        family = [["x5x"], ["o5x"], ]
        dias = []
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                dx = 16 / (max(2, level_count) - 1)
                x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location_lin([4, 10], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
                dias.append(dia2)

        # tell simple story about the one-dimensional case
        mirror_center = Vector([6, 0, 2])
        size = 4
        mirror_a = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A", normal=Vector([1, 0, 0]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_a.move_to(target_location=mirror_center, begin_time=0, transition_time=0)

        mirror_b = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A",
                         normal=Vector([np.cos(pi / 5), 0, np.sin(pi / 5)]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_b.move_to(target_location=mirror_center, begin_time=0, transition_time=0)
        mirrors = [mirror_a, mirror_b]

        src_location = 2 * Vector([np.sin(pi / 10), 0, - np.cos(pi / 10)])
        vertex = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - src_location)
        for mirror in mirrors:
            mirror.grow(begin_time=t0, transition_time=1)
        t0 = 1.5 + t0
        t0 = 0.5 + vertex.grow(begin_time=t0, transition_time=0.5)

        appear_order = [1, 9, 2, 8, 7, 3, 4, 5, 6]
        src_locations = [
            2 * Vector([np.sin(pi / 10 + i * pi / 5), 0, -np.cos(pi / 10 + i * pi / 5)]) for i in appear_order]

        sources = [None] * 10
        sources[0] = vertex
        for idx, location in zip(appear_order, src_locations):
            sphere = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - location)
            t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=1)
            sources[idx] = sphere

        src_locations.insert(0, src_location)

        edges = [Cylinder.from_start_to_end(
            start=mirror_center - 2 * Vector([np.sin(pi / 10 + i * pi / 5), 0, -np.cos(pi / 10 + i * pi / 5)]),
            end=mirror_center - 2 * Vector(
                [np.sin(pi / 10 + (i + 1) % 10 * pi / 5), 0, -np.cos(pi / 10 + (i + 1) % 10 * pi / 5)]),
            color="example", thickness=0.5) for i in range(10)]

        for edge in edges:
            t0 = 0.5 + edge.grow(begin_time=t0, transition_time=1)

        # reindex src_locations
        src_locations2 = [
            2 * Vector([np.sin(pi / 10 + i * pi / 5), 0, -np.cos(pi / 10 + i * pi / 5)]) for i in range(10)]
        decagon = Polygon(vertices=[mirror_center - src_location for src_location in src_locations2], color="decagon")
        t0 = 0.5 + decagon.appear(begin_time=t0, transition_time=1)

        to_copies = sources + edges + [decagon]
        copies = []
        for to_copy in to_copies:
            copies.append(to_copy.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copy.move(direction=[0, 0.1, 0], begin_time=0, transition_time=0)
            copy.move(direction=[-10, 0, 2], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        new_source_locations = [Vector()] * 10
        for pair in pairs:
            loc = pair
            center = mirror_center - (
                    Vector([np.sin(pi / 10 + loc[0] * pi / 5), 0, -np.cos(pi / 10 + loc[0] * pi / 5)]) +
                    Vector([np.sin(pi / 10 + loc[1] * pi / 5), 0, -np.cos(pi / 10 + loc[1] * pi / 5)])
            )
            sources[pair[0]].move_to(target_location=center, begin_time=t0, transition_time=1)
            sources[pair[1]].move_to(target_location=center, begin_time=t0, transition_time=1)
            new_source_locations[pair[0]] = center
            new_source_locations[pair[1]] = center

        # transform edges
        r = 2 * float(np.cos(pi / 10))
        for i in [0, 2, 4, 6, 8]:
            edge = edges[i]
            new_start = mirror_center - r * Vector([np.sin(pi / 5 + i * pi / 5), 0, -np.cos(pi / 5 + i * pi / 5)])
            new_end = mirror_center - r * Vector([np.sin(pi / 5 + i * pi / 5), 0, -np.cos(pi / 5 + i * pi / 5)])
            # lift the degeneracy between new_start and new_end to avoid rotations
            new_start = new_start + 0.001 * (edge.start - new_start)
            new_end = new_end + 0.001 * (edge.end - new_end)
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        for i in [1, 3, 5, 7, 9]:
            edge = edges[i]
            l = i // 2
            new_start = mirror_center - r * Vector(
                [np.sin(pi / 5 + 2 * l * pi / 5), 0, -np.cos(pi / 5 + 2 * l * pi / 5)])
            new_end = mirror_center - r * Vector(
                [np.sin(pi / 5 + (l + 1) % 5 * 2 * pi / 5), 0, -np.cos(pi / 5 + (l + 1) % 5 * 2 * pi / 5)])
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        # transform polygon
        decagon.change_color(new_color="pentagon", begin_time=t0, transition_time=1)
        t0 = 0.5 + decagon.index_transform_mesh(
            lambda idx: mirror_center - r * Vector(
                [np.sin(pi / 5 + ((idx + 1) // 2 + 1) * 2 * pi / 5), 0,
                 -np.cos(pi / 5 + ((idx + 1) // 2 + 1) * 2 * pi / 5)]),
            begin_time=t0, transition_time=1)

        to_copies = sources + edges + [decagon]
        copies = []
        for to_copy in to_copies:
            copies.append(to_copy.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copy.move(direction=[0, 0.1, 0], begin_time=0, transition_time=0)
            copies[-1].change_color(new_color="pentagon", begin_time=t0, transition_time=0)
            copy.move(direction=[-10, 0, -5], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        # building blocks panel
        t0 = 0.5 + show_panel("x", begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_a1xa1(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21.5, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }A_1\times A_1", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[-1, 4],
                           tic_labels={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)

        family = [["x2x"], ["o2x"], ]
        dias = []
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                dx = 16 / (max(2, level_count) - 1)
                x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location_lin([-1, 4], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
                dias.append(dia2)

        mirror_center = Vector([6, 0, 2])
        size = 4
        mirror_a = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A", normal=Vector([1, 0, 0]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_a.move_to(target_location=mirror_center, begin_time=0, transition_time=0)

        mirror_b = Plane(u=[-size, size], v=[-0.5, 0.5], color='mirror', name="Mirror_A",
                         normal=Vector([0, 0, 1]),
                         roughness=0.05,
                         shadow=False, solid=0.1, solidify_mode="SIMPLE",
                         smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        mirror_b.move_to(target_location=mirror_center, begin_time=0, transition_time=0)
        mirrors = [mirror_a, mirror_b]

        src_location = 2 * Vector([np.sin(pi / 4), 0, - np.cos(pi / 4)])
        vertex = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - src_location)
        for mirror in mirrors:
            mirror.grow(begin_time=t0, transition_time=1)
        t0 = 1.5 + t0
        t0 = 0.5 + vertex.grow(begin_time=t0, transition_time=0.5)

        appear_order = [1, 2, 3]
        src_locations = [
            2 * Vector([np.sin(pi / 4 + i * pi / 2), 0, -np.cos(pi / 4 + i * pi / 2)]) for i in appear_order]

        sources = [vertex]
        for location in src_locations:
            sphere = Sphere(r=0.25, mesh_type="ico", color="red", resolution=1, location=mirror_center - location)
            t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=1)
            sources.append(sphere)

        src_locations.insert(0, src_location)

        edges = [Cylinder.from_start_to_end(
            start=mirror_center - 2 * Vector([np.sin(pi / 4 + i * pi / 2), 0, -np.cos(pi / 4 + i * pi / 2)]),
            end=mirror_center - 2 * Vector(
                [np.sin(pi / 4 + (i + 1) % 4 * pi / 2), 0, -np.cos(pi / 4 + (i + 1) % 4 * pi / 2)]),
            color="example", thickness=0.5) for i in range(4)]

        for edge in edges:
            t0 = 0.5 + edge.grow(begin_time=t0, transition_time=1)

        # reindex src_locations
        src_locations2 = [
            2 * Vector([np.sin(pi / 4 + i * pi / 2), 0, -np.cos(pi / 4 + i * pi / 2)]) for i in range(6)]
        square = Polygon(vertices=[mirror_center - src_location for src_location in src_locations2], color="square")
        t0 = 0.5 + square.appear(begin_time=t0, transition_time=1)

        to_copies = sources + edges + [square]
        copies = []
        for to_copy in to_copies:
            copies.append(to_copy.copy())

        for copy in copies:
            copy.appear(begin_time=t0, transition_time=0)
            copy.move(direction=[-10, 0, 2], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        pairs = [[0, 1], [2, 3]]
        pair_locations = [[0, 1], [3, 2]]
        new_source_locations = [Vector()] * 6
        for pair, loc in zip(pairs, pair_locations):
            center = mirror_center - (
                    Vector([np.sin(pi / 4 + loc[0] * pi / 2), 0, -np.cos(pi / 4 + loc[0] * pi / 2)]) +
                    Vector([np.sin(pi / 4 + loc[1] * pi / 2), 0, -np.cos(pi / 4 + loc[1] * pi / 2)])
            )
            sources[pair[0]].move_to(target_location=center, begin_time=t0, transition_time=1)
            sources[pair[1]].move_to(target_location=center, begin_time=t0, transition_time=1)
            new_source_locations[pair[0]] = center
            new_source_locations[pair[1]] = center

        # transform edges

        r = float(r2)
        for i in [0, 2]:
            edge = edges[i]
            new_start = mirror_center - r * Vector([np.sin(pi / 2 + i // 2 * pi), 0, -np.cos(pi / 2 + i // 2 * pi)])
            new_end = mirror_center - r * Vector([np.sin(pi / 2 + i // 2 * pi), 0, -np.cos(pi / 2 + i // 2 * pi)])
            # lift the degeneracy between new_start and new_end to avoid rotations
            new_start = new_start + 0.001 * (edge.start - new_start)
            new_end = new_end + 0.001 * (edge.end - new_end)
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        for i in [1, 3]:
            edge = edges[i]
            l = i // 2
            new_start = mirror_center - r * Vector([np.sin(pi / 2 + 2 * l * pi / 3), 0, -np.cos(pi / 2 + 2 * l * pi)])
            new_end = mirror_center - r * Vector(
                [np.sin(pi / 2 + (l + 1) % 2 * 2 * pi / 3), 0, -np.cos(pi / 2 + (l + 1) % 2 * 2 * pi)])
            edge.move_to_new_start_and_end_point(
                start=new_start,
                end=new_end,
                begin_time=t0, transition_time=1
            )

        # transform polygon
        trafo = lambda i: mirror_center - r * Vector(
            [np.sin(pi / 2 + (i // 2) * 2 * pi / 2), 0, -np.cos(pi / 2 + (i // 2) * 2 * pi / 2)])
        square.change_color(new_color="square", begin_time=t0, transition_time=1)
        t0 = 0.5 + square.index_transform_mesh(trafo, begin_time=t0, transition_time=1)

        t0 = 0.5 + show_panel("x", begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_a1xa1xa1(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), no_transmission_ray=False,
                                 reflections=False, reflection_color=[0.05, 0.05, 0.05, 1])
        t0 = ibpy.set_hdri_strength(1.5, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        rotatables = []
        title = Text(r"\text{Coxeter Group }A_1\times A_1\times A_1", color="example", text_size="Large",
                     aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)
        set_origin(title)
        rotatables.append(title)

        axis = NumberLine2(length=12, domain=[1, 8],
                           tic_labels={"8": 8}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        rotatables.append(axis)

        family = [["x2x2x"]]

        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = -7
                else:
                    dx = 16 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location_lin([1, 8], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5, no_threes=True)
                rotatables.append(dia2)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        # three orthogonal mirrors for A1×A1×A1
        size = 4
        n_a = Vector([1, 0, 0])
        n_b = Vector([0, 0, 1])
        n_c = Vector([0, 1, 0])

        origin = Vector([2, 0, 0])
        plane_a = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_A", normal=n_a, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_b = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_B", normal=n_b, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_c = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_C", normal=n_c, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)

        camera_shift = Vector([-2, 0, 3.5])
        camera_location += camera_shift
        ibpy.camera_move(shift=camera_shift, begin_time=t0, transition_time=1)

        for rotatable in rotatables:
            rotatable.rotate(
                rotation_quaternion=ibpy.camera_alignment_quaternion(camera_empty, camera_location, default=(0, -1, 0)),
                begin_time=t0, transition_time=1)

        floor = Floor(u=[-size, size], v=[-size, size], checker_scale=10, coords="Generated")
        floor.move_to(target_location=origin + Vector([0, 0, -4]), begin_time=0, transition_time=0)
        t0 = 0.5 + floor.appear(begin_time=t0, transition_time=1)

        mirrors = [plane_a, plane_b, plane_c]
        for mirror in mirrors:
            mirror.move_to(target_location=origin, begin_time=t0, transition_time=0)
            t0 = 0.5 + mirror.grow(begin_time=t0, transition_time=1)

        # cube: the main polytope for A1×A1×A1
        vertices, faces = get_solid_data("CUBE")
        cube = BObject(mesh=ibpy.create_mesh(vertices, faces=faces),
                       location=origin, scale=1.5, color="text")
        mod = CustomUnfoldModifier(face_materials=["text"],
                                   edge_material="example", vertex_material="red",
                                   edge_radius=0.025, face_types=[4],
                                   vertex_radius=0.051, sorting=False)
        cube.add_mesh_modifier(type="NODES", node_modifier=mod)
        cube.appear(begin_time=t0, transition_time=0)
        t0 = 0.5 + mod.grow(begin_time=t0, transition_time=1.5, max_faces=6)

        # build a separate copy at scale 1.49 so the color change below does not affect it
        cube_copy = BObject(mesh=ibpy.create_mesh(vertices, faces=faces),
                            location=origin, scale=1.45, color="text")
        mod_copy = CustomUnfoldModifier(face_materials=["text"],
                                        edge_material="example", vertex_material="red",
                                        edge_radius=0.025, face_types=[4],
                                        vertex_radius=0.051, sorting=False)
        cube_copy.add_mesh_modifier(type="NODES", node_modifier=mod_copy)
        cube_copy.appear(begin_time=t0, transition_time=0)
        t0 = mod_copy.grow(begin_time=t0, transition_time=0, max_faces=6)

        t0 = 0.5 + cube_copy.move_to(target_location=[-6, 0, 2.5], begin_time=t0, transition_time=1)
        cube_copy.rotate(rotation_euler=[1.25 * tau, 0, 0], begin_time=t0, transition_time=10)

        # define a transformation for the vertices
        # cube vertices are at (±1, ±1, ±1); collapse y→0 to obtain a square in the XZ plane (x2o2x)

        def transform(v):
            if v[1] < 0:
                return Vector([v[0], -0.01, v[2]])
            else:
                return Vector([v[0], 0.01, v[2]])

        t0 = 0.5 + cube.transform_mesh(transform, begin_time=t0, transition_time=1)

        [mirror.disappear(begin_time=t0, transition_time=1) for mirror in mirrors]
        floor.disappear(begin_time=t0, transition_time=1)

        camera_location -= camera_shift
        ibpy.camera_move(shift=-camera_shift, begin_time=t0, transition_time=1)

        for rotatable in rotatables:
            rotatable.rotate(
                rotation_quaternion=ibpy.camera_alignment_quaternion(camera_empty, camera_location, default=(0, -1, 0)),
                begin_time=t0, transition_time=1)

        t0 = 1.5 + t0

        t0 = 0.5 + show_panel(["x.x"],
                              scales=[0.5, 0.5], poly_shift=Vector([-0.7, -0.25, 0]),
                              global_shift=[0, -1, 0],
                              begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_a2xa1(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), no_transmission_ray=False,
                                 reflections=False, reflection_color=[0.05, 0.05, 0.05, 1])
        t0 = ibpy.set_hdri_strength(1.5, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)
        origin = Vector([2, 0, 0])

        rotatables = []
        title = Text(r"\text{Coxeter Group }A_2\times A_1", color="example", text_size="Large",
                     aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)
        set_origin(title)
        rotatables.append(title)

        axis = NumberLine2(length=12, domain=[1, 12],
                           tic_labels={"6": 6, "12": 12}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        # axis.to_log(begin_time=0, transition_time=0)
        rotatables.append(axis)

        family = [["x3x2x"], ["x3o2x"]]

        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = -7
                else:
                    dx = 16 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location_lin([1, 12], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5, no_threes=True)
                rotatables.append(dia2)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        # mirrors for A2×A1: two A2 mirrors at 60°, one A1 mirror perpendicular
        size = 4
        n_a = Vector([1, 0, 0])
        n_b = Vector([0, 0, 1])
        n_c = Vector([np.cos(pi / 3), np.sin(pi / 3), 0])

        plane_a = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_A", normal=n_a, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_b = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_B", normal=n_b, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_c = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_C", normal=n_c, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)

        camera_shift = Vector([-2, 0, 10])
        camera_location += camera_shift
        ibpy.camera_move(shift=camera_shift, begin_time=t0, transition_time=1)

        for rotatable in rotatables:
            rotatable.rotate(
                rotation_quaternion=ibpy.camera_alignment_quaternion(camera_empty, camera_location, default=(0, -1, 0)),
                begin_time=t0, transition_time=1)

        floor = Floor(u=[-size, size], v=[-size, size], checker_scale=10, coords="Generated")
        floor.move_to(target_location=origin + Vector([0, 0, -4]), begin_time=0, transition_time=0)
        t0 = 0.5 + floor.appear(begin_time=t0, transition_time=1)

        mirrors = [plane_a, plane_b, plane_c]
        for mirror in mirrors:
            mirror.move_to(target_location=origin, begin_time=t0, transition_time=0)
            t0 = 0.5 + mirror.grow(begin_time=t0, transition_time=1)

        # hexagonal prism: the main polytope for A2×A1
        vertices, faces = get_solid_data("PRISM6")
        prism6 = BObject(mesh=ibpy.create_mesh(vertices, faces=faces),
                         location=origin, scale=1.5, color="example")
        mod = CustomUnfoldModifier(face_materials=["square", "hexagon"],
                                   edge_material="example", vertex_material="red",
                                   edge_radius=0.025, face_types=[4, 6],
                                   vertex_radius=0.051, sorting=False)
        prism6.add_mesh_modifier(type="NODES", node_modifier=mod)
        prism6.appear(begin_time=t0, transition_time=0)
        t0 = 0.5 + mod.grow(begin_time=t0, transition_time=1.5, max_faces=8)

        # make prism again a tiny bit smaller (a copy will transform the color as well, what we dont want)
        prism6_copy = BObject(mesh=ibpy.create_mesh(vertices, faces=faces),
                              location=origin, scale=1.49, color="example")
        mod = CustomUnfoldModifier(face_materials=["square", "hexagon"],
                                   edge_material="example", vertex_material="red",
                                   edge_radius=0.025, face_types=[4, 6],
                                   vertex_radius=0.051, sorting=False)
        prism6_copy.add_mesh_modifier(type="NODES", node_modifier=mod)
        prism6_copy.appear(begin_time=t0, transition_time=0)
        ibpy.set_origin(prism6_copy)
        t0 = mod.grow(begin_time=t0, transition_time=0, max_faces=8)

        t0 = 0.5 + prism6_copy.move_to(target_location=[-4, 0, 4.5], begin_time=t0, transition_time=1)

        prism6_copy.rotate(rotation_euler=[1.25 * tau, 0, 0], begin_time=t0, transition_time=10)

        # define a transformation for the vertices
        # they are generated with (r*cos(tau/6*i),r*sin(tau/6*i),z)

        def transform(v):
            # use radius, such that side length is equal to 1
            r = 1.0 / 2 / np.sin(np.pi / 3)
            phi = np.arctan2(v[1], v[0])

            i = int(np.round(phi / (pi / 3)))
            # odd move forward by pi/6
            # even move backward by pi/6
            if i % 2 == 0:
                phi -= pi / 6
            else:
                phi += pi / 6
            return r * Vector([np.cos(phi), np.sin(phi), v[2]])

        prism6.change_color(new_color="triangle", begin_time=t0, transition_time=1, slot=2)
        t0 = 0.5 + prism6.transform_mesh(transform, begin_time=t0, transition_time=1)

        copy_prism3 = prism6.copy()
        copy_prism3.appear(begin_time=t0, transition_time=0)
        copy_prism3.rescale(rescale=0.99, begin_time=0, transition_time=0)
        t0 = 0.5 + copy_prism3.move_to(target_location=[-4, 0, -1.5], begin_time=t0, transition_time=1)
        copy_prism3.rotate(rotation_euler=[1.25 * tau, 0, 0], begin_time=t0, transition_time=10)

        [mirror.disappear(begin_time=t0, transition_time=1) for mirror in mirrors]
        floor.disappear(begin_time=t0, transition_time=1)
        # prism6.disappear(begin_time=t0, transition_time=1)

        camera_location -= camera_shift
        ibpy.camera_move(shift=-camera_shift, begin_time=t0, transition_time=1)

        for rotatable in rotatables:
            rotatable.rotate(
                rotation_quaternion=ibpy.camera_alignment_quaternion(camera_empty, camera_location, default=(0, -1, 0)),
                begin_time=t0, transition_time=1)

        t0 = 1.5 + t0

        t0 = 0.5 + show_panel(["x3x", "x3o", "x.x"],
                              scales=[0.5, 0.5, 0.5], poly_shift=Vector([-0.7, -0.25, 0]),
                              begin_time=t0, transition_time=1)

        self.t0 = t0

    def a_family(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        title = Text(r"\text{A--Family}", color=example, text_size="Large", aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        labels = ["x", "x3x", "x3x3x", "x3x3x3x"]

        dynkin_diagrams = []
        for i, label in enumerate(labels):
            dynkin = DynkinDiagram.from_string(label, location=[-8 + 1 * i, 0, 4 - 3 * i], scale=1)
            t0 = 0.5 + dynkin.appear(begin_time=t0, transition_time=0.5)
            dynkin_diagrams.append(dynkin)

        current_alphas = [1] * len(dynkin_diagrams)
        for i in range(4):
            for j in range(4):
                if i != j:
                    dynkin_diagrams[j].change_alpha(from_value=current_alphas[j], to_value=0.1, begin_time=t0,
                                                    transition_time=0.5, children=True)
                    current_alphas[j] = 0.1
                else:
                    dynkin_diagrams[j].change_alpha(from_value=current_alphas[j], to_value=1, begin_time=t0,
                                                    transition_time=0.5, children=True)
                    current_alphas[j] = 1
            t0 += 1

        self.t0 = t0

    def family_a4_nets(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }A_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[4, 120],
                           tic_labels={"5": 5, "10": 10, "20": 20, "30": 30, "60": 60, "120": 120}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        family = [["x3x3x3x"], ["x3o3x3x", "o3x3x3x"], ["o3o3x3x", "x3o3o3x", "o3x3x3o", "o3x3o3x"],
                  ["o3o3o3x", "o3x3o3o"]]
        # dias = []
        # for level in family:
        #     level_count = len(level)
        #     for i, cd_string in enumerate(level):
        #         if level_count == 1:
        #             x = 0
        #         else:
        #             dx = 16 / (level_count - 1)
        #             x = -8 + i * dx
        #         dia = CoxeterDynkinDiagram(cd_string)
        #         z = get_z_location([4, 120], [-6, 6], dia.get_vertex_count(), offset=-1)
        #         dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25)
        #         t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
        #         dias.append(dia2)

        morphing_start_time = t0
        morpher_center = [3, 0, -3.1]

        # first morph sequence
        morpher = NetMorpher4D(CoxA4, ["x3x3x3x", "x3o3x3x", "o3o3x3x", "o3o3o3x"], name="MorpherA4",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=t0, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 6,12, 18 are back morphing times
        appear_times = [0, 3, 6, 9]
        locations = [
            Vector([4.5, 0, 4.7]),
            Vector([-7.7, 0, 4.7]),
            Vector([-8, 0, 0.13]),
            Vector([-7.4, 0, -3.9]),

        ]

        states = [morpher.get_state(i) for i in range(4)]
        scales = [0.75, 0.5, 0.5, 0.75]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # second morph sequence

        morpher = NetMorpher4D(CoxA4, ["x3x3x3x", "x3x3x3o", "x3o3x3o", "o3x3o3o"], name="MorpherA4",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=t0, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        # rot_box = BObject(children=[morpher.bob]+rays,location=morpher_center,name="rotation_container")
        # rot_box.appear(begin_time=t0,transition_time=0,children=False)
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 6,12, 18 are back morphing times
        appear_times = [3, 6, 9]
        locations = [
            Vector([9.3, 0, 4.47]),
            Vector([9.6, 0, -1.3]),
            Vector([10.1, 0, -5]),

            # Vector([-2.6, 0, 0.5]),
        ]

        states = [morpher.get_state(i) for i in range(1, 4)]
        scales = [0.5, 0.5, 0.75]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # third morph sequence

        morpher = NetMorpher4D(CoxA4, ["x3x3x3x", "x3o3o3x"], name="MorpherA4",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=1)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)

        # make individual projections appear at 3 are back morphing times
        appear_times = [3]
        locations = [
            Vector([-3, 0, 1.5]),
        ]

        states = [morpher.get_state(i) for i in range(1, 2)]
        scales = [0.75]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 6, transition_time=1)

        # fourth morph sequence

        morpher = NetMorpher4D(CoxA4, ["x3x3x3x", "o3x3x3o"], name="MorpherA4",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=1)

        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)

        morpher.bob.rescale(rescale=0.7, begin_time=morphing_start_time + 6, transition_time=1)
        morphing_start_time = 0.5 + morpher.bob.move(direction=[0, 0, 0.5], begin_time=morphing_start_time + 6,
                                                     transition_time=1)

        t0 = 0.5 + show_panel(["x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o", "x3x.x", "o3x.x"],
                              scales=[0.35, 0.25, 0.5, 0.6, 0.6, 0.5, 0.8],
                              begin_time=morphing_start_time, transition_time=1)
        self.t0 = t0

    def family_b4_nets(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }B_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = title.write(begin_time=t0, transition_time=0)

        axis = NumberLine2(length=12, domain=[4, 384],
                           tic_labels={"8": 8, "16": 16, "24": 24, "32": 32, "48": 48, "64": 64, "96": 96, "192": 192,
                                       "384": 384}, tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = axis.grow(begin_time=t0, transition_time=0)
        axis.to_log(begin_time=0, transition_time=0)

        dias = []
        family = [["x3x3x4x"], ["o3x3x4x", "x3x3o4x", "x3x3x4o", "x3o3x4x"],
                  ["o3o3x4x", "x3o3o4x", "x3x3o4o", "o3x3x4o", "o3x3o4x", "x3o3x4o"],
                  ["o3o3o4x", "x3o3o4o", "o3x3o4o", "o3o3x4o"]]
        for l, level in enumerate(family):
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    if l < 3:
                        dx = 16 / (level_count - 1)
                    else:
                        dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([4, 384], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge")
                dias.append(dia2)
                t0 = dia2.appear(begin_time=t0, transition_time=0)

        morphing_start_time = t0
        morpher_center = [7, 0, -2.6]

        # first morph sequence
        morpher = NetMorpher4D(CoxB4, ["x3x3x4x", "o3x3x4x", "o3o3x4x", "o3o3o4x"], name="MorpherB4_a",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=t0, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 6,12, 18 are back morphing times
        appear_times = [0, 3, 6, 9]
        locations = [
            Vector([-5.1, 0, 4.9]),
            Vector([-7.7, 0, 1.7]),
            Vector([-8, 0, -1.34]),
            Vector([-8, 0, -5]),

        ]

        states = [morpher.get_state(i) for i in range(4)]
        scales = [0.65, 0.5, 0.5, 0.5]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # second morph sequence

        morpher = NetMorpher4D(CoxB4, ["x3x3x4x", "x3x3o4x", "x3o3o4x", "x3o3o4o"], name="MorpherB4_b",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 6,12, 18 are back morphing times
        appear_times = [3, 6, 9]
        locations = [
            Vector([-2.0, 0, 1.95]),
            Vector([-4.37, 0, -1.522]),
            Vector([-4.7, 0, -5.21]),

            # Vector([-2.6, 0, 0.5]),
        ]

        states = [morpher.get_state(i) for i in range(1, 4)]
        scales = [0.5, 0.5, 0.7]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        dias[2].move(direction=[0, 0, 0.3], begin_time=morphing_start_time + 3 + 1, transition_time=0.5)
        dias[12].move(direction=[0, 0, 1.7], begin_time=morphing_start_time + 9 + 1, transition_time=0.5)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # third morph sequence
        # skip first net
        morpher = NetMorpher4D(CoxB4, ["x3x3x4x", "x3x3o4x", "x3x3o4o", "o3x3o4o"], name="MorpherB4_c",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time - 3,
                               appear_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 3 are back morphing times
        appear_times = [3, 6]
        locations = [
            Vector([-1.38, 0, -1.96]),
            Vector([-1.32, 0, -5.35]),
        ]

        states = [morpher.get_state(i) for i in range(2, 4)]
        scales = [0.5, 0.5]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        dias[13].move(direction=[0, 0, -1.5], begin_time=morphing_start_time + 3, transition_time=0.5)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # fourth morph sequence

        morpher = NetMorpher4D(CoxB4, ["x3x3x4x", "x3x3x4o", "o3x3x4o", "o3o3x4o"], name="MorpherB4_e",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=1)

        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 3 are back morphing times
        appear_times = [3, 6, 9]
        locations = [
            Vector([4.07, 0, 4.855]),
            Vector([2.897, 0, 2.1]),
            Vector([2.14, 0, -0.08]),
        ]

        states = [morpher.get_state(i) for i in range(1, 4)]
        scales = [0.5, 0.5, 0.5]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        dias[3].move(direction=[-0.55, 0, 0.5], begin_time=morphing_start_time + 6, transition_time=0.5)
        dias[8].move(direction=[-1.1, 0, -0.45], begin_time=morphing_start_time + 6, transition_time=0.5)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # fifth morph sequence

        morpher = NetMorpher4D(CoxB4, ["x3x3x4x", "x3o3x4x", "x3o3x4o"], name="MorpherB4_e",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=1)

        # rotate main axis into z-direction
        final_direction = Vector([0, 0, 1])
        omega = Vector(main_axis).cross(final_direction)
        quat_0 = Quaternion(omega, np.arccos(main_axis.dot(final_direction)))
        quat_1 = Quaternion(Vector([0, 0, 1]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)

        # make individual projections appear at 3 are back morphing times
        appear_times = [3, 6]
        locations = [
            Vector([10.05, 0, 5.2488]),
            Vector([10.55, 0, 1.1693]),
        ]

        states = [morpher.get_state(i) for i in range(1, 3)]
        scales = [0.5, 0.5]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 9, transition_time=1)

        # final morph sequence
        morpher = NetMorpher4D(CoxB4, ["x3x3x4x", "o3x3o4x"], name="MorpherB4_f",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time,
                               transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=1)
        # rotate main axis into z-direction
        final_direction = Vector([0, 0, -1])
        omega = Vector(main_axis).cross(final_direction)
        quat_0 = Quaternion(omega, np.arccos(main_axis.dot(final_direction)))
        quat_1 = Quaternion(final_direction, pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        morphing_start_time = 0.5 + morpher.bob.move(direction=[-0.8, 0, 0.5], begin_time=morphing_start_time + 6,
                                                     transition_time=1)

        t0 = 0.5 + show_panel(["x3x4x", "x3x4o", "x3o4x", "o3x4x", "o3o4x", "o3x4o", "x3o4o",
                               "x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o",
                               "x3x.x", "x3o.x", "x4x.x", "x4o.x"],
                              scales=[0.096, 0.225, 0.19, 0.4, 0.4, 0.38, 0.5, 0.24,
                                      0.13, 0.4, 0.377, 0.445,
                                      0.42, 0.65, 0.34, 0.65],
                              dynkin_scale=0.5, columns=2,
                              begin_time=morphing_start_time, transition_time=1,
                              dynkin_shift=Vector([-0.15, -1, 0]),
                              poly_shift=Vector([-1.55, -0.9, 0]), dynkin_label_shift=Vector([-0.5, 0, 0.6]))

        self.t0 = t0

    def family_h4_nets_a(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }H_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = title.write(begin_time=t0, transition_time=0)

        axis = NumberLine2(length=12, domain=[60, 14400],
                           tic_labels={"120": 120, "600": 600, "720": 720, "1200": 1200, "1440": 1440,
                                       "2400": 2400, "3600": 3600, "7200": 7200, "14400": 14400},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1.3, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=0)
        axis.to_log(begin_time=0, transition_time=0)

        dias = []
        family = [["x3x3x5x"], ["o3x3x5x", "x3x3o5x", "x3x3x5o", "x3o3x5x"],
                  ["o3o3x5x", "x3o3o5x", "x3x3o5o", "o3x3x5o", "o3x3o5x", "x3o3x5o"],
                  ["o3o3o5x", "x3o3o5o", "o3x3o5o", "o3o3x5o"]]
        for l, level in enumerate(family):
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    if l < 3:
                        dx = 16 / (level_count - 1)
                    else:
                        dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([60, 14400], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge")
                dias.append(dia2)
                t0 = dia2.appear(begin_time=t0, transition_time=0)

        morphing_start_time = t0

        morpher_center = [7, 0, -2.6]

        # first morph sequence
        morpher = NetMorpher4D(CoxH4, ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x"], name="MorpherH4_a",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=t0, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 6,12, 18 are back morphing times
        appear_times = [0, 3, 6, 9]
        locations = [
            Vector([-5.1, 0, 4.9]),
            Vector([-7.9, 0, 2.64]),
            Vector([-8, 0, -0.4625]),
            Vector([-8, 0, -5]),

        ]

        states = [morpher.get_state(i) for i in range(4)]
        scales = [0.675, 0.5, 0.5, 0.5]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale * 0.8, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        dias[1].move(direction=[0, 0, 0.7], begin_time=morphing_start_time + 3, transition_time=1)
        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)
        print(morphing_start_time)
        self.t0 = t0

    def trailer_overlay(self):
        t0 = 0

        ibpy.set_hdri_background("belfast_sunset_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 270]))
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)
        # ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)

        sequence = ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x",
                    "x3x3x5x", "x3x3o5x", "x3o3o5x", "x3o3o5o",
                    "x3x3o5x", "x3x3o5o", "o3x3o5o",
                    "x3x3x5x", "x3x3x5o", "o3x3o5x", "o3o3x5o",
                    "x3x3x5o", "o3x3x5o",
                    "x3x3x5x", "x3o3x5x", "x3o3x5o"
                    ]

        panel = Plane(u=[-2, 2], v=[-0.875, 0.875], normal=Vector([0, -1, 0]), color="frosty", solid=0.1, offset=-1)
        dia = DynkinDiagram.from_string(sequence[0], no_threes=True, scale=0.5, location=[14.96, 0, -6])

        panel.move_to(target_location=Vector([15, 0.1, -6]), begin_time=0, transition_time=0)
        dia.appear(begin_time=0, transition_time=0)
        dia.move(direction=[-5, 0, 0], begin_time=2, transition_time=1)
        t0 = 1 + panel.move(direction=[-5, 0, 0], begin_time=2, transition_time=1)

        for i in range(1, len(sequence)):
            t0 += 1
            t0 = 1 + dia.change_state(from_state=sequence[i - 1], to_state=sequence[i], begin_time=t0 + 0.5,
                                      transition_time=0.5)

        t0 = 0.5 + show_panel([
            "x3x5x", "x3x5o", "x3o5x", "o3x5x", "o3o5x", "o3x5o", "x3o5o",
            "x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o",
            "x3x.x", "x3o.x", "x5x.x", "x5o.x"
        ],
            scales=[0.096, 0.09, 0.122, 0.146, 0.303, 0.332, 0.303, 0.24,
                    0.13, 0.4, 0.377, 0.445,
                    0.42, 0.65, 0.34, 0.65],
            dynkin_scale=0.75, columns=2,
            show_time=55,
            begin_time=2, transition_time=1,
            dynkin_shift=Vector([-0.15, -1, 0]),
            poly_shift=Vector([-1.55, -0.9, 0]), dynkin_label_shift=Vector([-0, 0, 0.6]))

        self.t0 = t0

    def no_crystals(self):
        t0 = 0

        ibpy.set_hdri_background("belfast_sunset_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 270]))
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)
        ibpy.camera_zoom(lens=90, begin_time=0, transition_time=0)
        set_alpha_composition()
        # ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)

        morpher = Morpher4D(CoxH4, ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x",
                                    "x3x3x5x", "x3x3o5x", "x3o3o5x", "x3o3o5o",
                                    "x3x3o5x", "x3x3o5o", "o3x3o5o",
                                    "x3x3x5x", "x3x3x5o", "o3x3o5x", "o3o3x5o",
                                    "x3x3x5o", "o3x3x5o",
                                    "x3x3x5x", "x3o3x5x", "x3o3x5o"
                                    ],
                            rotation_euler=[-pi / 2, 173 / 180 * pi, 0], scale=0.255,
                            location=[0, 0, 0],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1, south_pole=1,
                            mode="CELL_SIZE", cell_size=20, skip=0, scale_elements=0.99,
                            cell_removals=[1858, 1981, 2333, 2177, 2269, 2166, 2235, 1032, 813, 660, 139, 492, 358],
                            crystal=False)

        t0 = 0.5 + morpher.morph(begin_time=t0, transition_time=2, pause=1)

        t0 = 60
        self.t0 = t0

    def no_crystals2(self):
        t0 = 0

        ibpy.set_hdri_background("belfast_sunset_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 270]))
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)
        ibpy.camera_zoom(lens=90, begin_time=0, transition_time=0)
        set_alpha_composition()
        # ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)

        morpher = Morpher4D(CoxF4, ["x3x4x3x", "o3x4x3x", "o3o4x3x", "o3o4x3o",
                                    "o3x4x3x", "o3x4x3o",
                                    "x3x4x3x", "x3x4o3x", "x3o4o3x",
                                    "x3x4o3x", "o3x4o3x", "o3o4o3x"
                                    ],
                            rotation_euler=[-pi / 2, 135 / 180 * pi, 0], scale=0.255,
                            location=[0, 0, 0],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1, south_pole=1,
                            mode="CELL_SIZE", cell_size=48, skip=0, scale_elements=0.99,
                            cell_removals=[11],
                            crystal=False)

        t0 = 0.5 + morpher.morph(begin_time=t0, transition_time=2, pause=1)

        t0 = 60
        self.t0 = t0

    def crystals(self):
        t0 = 0
        # don't forget to put on the light colors before running this script
        ibpy.set_hdri_background("belfast_sunset_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 270]))
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)

        ibpy.camera_zoom(lens=90, begin_time=0, transition_time=2)

        set_alpha_composition()
        ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)

        morpher = Morpher4D(CoxH4, ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x",
                                    "x3x3x5x", "x3x3o5x", "x3o3o5x", "x3o3o5o",
                                    "x3x3o5x", "x3x3o5o", "o3x3o5o",
                                    "x3x3x5x", "x3x3x5o", "o3x3o5x", "o3o3x5o",
                                    "x3x3x5o", "o3x3x5o",
                                    "x3x3x5x", "x3o3x5x", "x3o3x5o"
                                    ],
                            rotation_euler=[-pi / 2, 173 / 180 * pi, 0], scale=0.255,
                            location=[0, 0, 0],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1, south_pole=1,
                            mode="CELL_SIZE", cell_size=20, skip=0, scale_elements=0.99,
                            cell_removals=[1858, 1981, 2333, 2177, 2269, 2166, 2235, 1032, 813, 660, 139, 492],
                            crystal=True)

        t0 = 0.5 + morpher.morph(begin_time=t0, transition_time=2, pause=1)
        t0 = 60
        self.t0 = t0

    def crystals2(self):
        t0 = 0
        # don't forget to put on the light colors before running this script
        ibpy.set_hdri_background("belfast_sunset_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 270]))
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)

        ibpy.camera_zoom(lens=90, begin_time=0, transition_time=2)

        set_alpha_composition()
        ibpy.light_path_settings(total=32, diffuse=32, glossy=32, transmission=32, volume=8, transparency=32)

        morpher = Morpher4D(CoxF4, ["x3x4x3x", "o3x4x3x", "o3o4x3x", "o3o4x3o",
                                    "o3x4x3x", "o3x4x3o",
                                    "x3x4x3x", "x3x4o3x", "x3o4o3x",
                                    "x3x4o3x", "o3x4o3x", "o3o4o3x"
                                    ],
                            rotation_euler=[-pi / 2, 135 / 180 * pi, 0], scale=0.255,
                            location=[0, 0, 0],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1, south_pole=1,
                            mode="CELL_SIZE", cell_size=48, skip=0, scale_elements=0.99,
                            cell_removals=[11],
                            crystal=True)

        t0 = 0.5 + morpher.morph(begin_time=t0, transition_time=2, pause=1)
        t0 = 60
        self.t0 = t0

    def short_dynkin_magic(self):
        t0 = 0

        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True, transparent=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 59]))
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)

        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)

        set_alpha_composition()

        sequence = ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x",
                    "x3x3x5x", "x3x3o5x", "x3o3o5x", "x3o3o5o",
                    "x3x3o5x", "x3x3o5o", "o3x3o5o",
                    "x3x3x5x", "x3x3x5o", "o3x3o5x", "o3o3x5o",
                    "x3x3x5o", "o3x3x5o",
                    "x3x3x5x", "x3o3x5x", "x3o3x5o"
                    ]

        diagram = DynkinDiagram.from_string(sequence[0], location=[0, 0, 8])
        t0 = 1 + diagram.appear(begin_time=0, transition_time=1)

        title = Text(r"\text{4D Magic}", color="example", text_size="Large", aligned="center", outline_color="joker")
        title.write(begin_time=t0, transition_time=1)

        for s in range(1, len(sequence)):
            t0 = 1 + diagram.change_state(from_state=sequence[s - 1], to_state=sequence[s], begin_time=t0,
                                          transition_time=1)

        t0 = 40
        t0 = 1 + title.unwrite(begin_time=t0, transition_time=1)
        self.t0 = t0

    def short_h4_family(self):
        t0 = 0

        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True, transparent=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 59]))
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)

        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=False)

        set_alpha_composition()

        morpher = Morpher4D(CoxH4, ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x",
                                    "x3x3x5x", "x3x3o5x", "x3o3o5x", "x3o3o5o",
                                    "x3x3o5x", "x3x3o5o", "o3x3o5o",
                                    "x3x3x5x", "x3x3x5o", "o3x3o5x", "o3o3x5o",
                                    "x3x3x5o", "o3x3x5o",
                                    "x3x3x5x", "x3o3x5x", "x3o3x5o"
                                    ],
                            rotation_euler=[-pi / 2, 0, 0], scale=0.075,
                            location=[0, 0, 0],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1, south_pole=1,
                            mode="CELL_SIZE", cell_size=20, skip=0)
        t0 = 0.5 + morpher.morph(begin_time=t0, transition_time=1, pause=1, grow_sequentially=True,
                                 cell_removal_list=[1858])

        # camera zooming
        ibpy.camera_zoom(lens=250, begin_time=0, transition_time=0)
        ibpy.camera_zoom(lens=30, begin_time=0.1, transition_time=2)
        ibpy.camera_zoom(lens=50, begin_time=4, transition_time=1)
        ibpy.camera_zoom(lens=65, begin_time=6, transition_time=1)
        ibpy.camera_zoom(lens=30, begin_time=8, transition_time=1)
        ibpy.camera_zoom(lens=55, begin_time=12, transition_time=1)
        ibpy.camera_zoom(lens=150, begin_time=14, transition_time=1)
        ibpy.camera_zoom(lens=30, begin_time=15.5, transition_time=1.5)
        ibpy.camera_zoom(lens=110, begin_time=18.5, transition_time=0.5)
        ibpy.camera_zoom(lens=94, begin_time=20, transition_time=1)
        ibpy.camera_zoom(lens=30, begin_time=22, transition_time=1)
        ibpy.camera_zoom(lens=46, begin_time=36, transition_time=4)

        self.t0 = t0

    def family_a4(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }A_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[4, 120],
                           tic_labels={"5": 5, "10": 10, "20": 20, "30": 30, "60": 60, "120": 120}, tic_label_digits=0,
                           direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        family = [["x3x3x3x"], ["x3o3x3x", "o3x3x3x"], ["o3o3x3x", "x3o3o3x", "o3x3x3o", "o3x3o3x"],
                  ["o3o3o3x", "o3o3x3o"]]
        dias = []
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 16 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([4, 120], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
                dias.append(dia2)

        # perform morphing x3x3x3x -> x3o3x3x -> x3o3o3x -> x3o3x3x -> o3o3o3x
        morphing_start_time = t0

        # customized rotation that centers one cell around y-pole
        external_rotation = Matrix(((1.0, 0, 0, 0),
                                    (0, 0.9341726303100586, -0.35682204365730286, 0),
                                    (0, 0.35682204365730286, 0.9341723918914795, 0),
                                    (0, 0, 0, 1.0)))

        # make individual projections appear at 6,12, 18 are back morphing times
        appear_times = [0, 2, 4, 8, 10, 14, 16, 18, 22]

        cd_strings = ["x3x3x3x", "x3o3x3x", "x3o3o3x", "o3o3x3x", "o3o3o3x", "o3x3x3x", "o3x3x3o", "o3o3x3o", "o3x3o3x"]
        unfolders = [Unfolder4D2(CoxA4, cd_string, external_rotation=external_rotation) for cd_string in cd_strings]
        locations = [
            Vector([3.3, 0, 3.4]),
            Vector([-7, 0, 4.7]),
            Vector([-2.6, 0, 0.5]),
            Vector([-7.5, 0, 0.13]),
            Vector([-8.1, 0, -4.7]),

            Vector([9.3, 0, 4.47]),
            Vector([4.8, 0, -1.9]),
            Vector([9.9, 0, -5.2]),
            Vector([9.6, 0, -1.3]),

        ]
        cell_removal_lists = [
            [17, 27],
            [19, 1],
            [17, 8, 26],
            [3],
            [],

            [5],
            [8],
            [9],
            [6]
        ]
        explode_scales = [
            2, 2, 3, 2, 4,
            2, 2, 4, 2
        ]

        remove_bobs = []
        for unfolder, time, location, signature, cell_removals, explode_scale in zip(unfolders, appear_times,
                                                                                     locations,
                                                                                     cd_strings, cell_removal_lists,
                                                                                     explode_scales):
            bob = unfolder.create_stereo_bob(location=location, scale=0.1, name=signature, half_way=False, limit=0,
                                             limit_direction=Vector([0, 0, 1]),
                                             rotation_euler=pi / 2 * Vector([1, 0, 0]), cell_removals=cell_removals)
            bob.move_to(target_location=location, begin_time=0, transition_time=0)
            bob.appear(begin_time=time + morphing_start_time, transition_time=1, children=True, sequentially=True)
            bob.explode(explode_scale=explode_scale, begin_time=time + morphing_start_time + 1, transition_time=1)
            remove_bobs.append(bob)

        morpher = Morpher4D(CoxA4, ["x3x3x3x", "x3o3x3x", "x3o3o3x", "x3o3x3x", "o3o3x3x", "o3o3o3x",
                                    "x3x3x3x", "o3x3x3x", "o3x3x3o", "o3o3x3o", "o3x3x3x", "o3x3o3x"],
                            rotation_euler=[pi / 2, 0, 0], scale=0.2, location=[2.9, 0, -3.1], name="Morpher",
                            limit=0.01
                            )
        t0 = 0.5 + morpher.morph(begin_time=morphing_start_time, transition_time=1, pause=1, scale_elements=0.95)

        # additional motion creating space for popups
        morpher.bob.move(direction=[-2, 0, -1], begin_time=morphing_start_time + 15, transition_time=1)
        morpher.bob.rescale(rescale=0.5, begin_time=morphing_start_time + 15, transition_time=1)
        morpher.bob.rescale(rescale=2, begin_time=morphing_start_time + 18, transition_time=1)
        morpher.bob.rescale(rescale=0.5, begin_time=morphing_start_time + 21, transition_time=1)

        morpher.bob.disappear(begin_time=morphing_start_time + 25, transition_time=1)
        dias[2].move(direction=[-0.6, 0, -0.3], begin_time=morphing_start_time + 15, transition_time=1)
        dias[6].move(direction=[-0.5, 0, 0.37], begin_time=morphing_start_time + 21, transition_time=1)

        t0 = show_panel(["x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o", "x3x.x", "o3x.x"],
                        scales=[0.35, 0.25, 0.5, 0.6, 0.6, 0.5, 0.8],
                        global_shift=Vector([0, -1, 0]),
                        begin_time=morphing_start_time + 22, transition_time=1)

        for bob in remove_bobs:
            bob.disappear(begin_time=t0, transition_time=1)

        dias[2].move(direction=[0.6, 0, 0.3], begin_time=t0, transition_time=1)
        dias[6].move(direction=[0.5, 0, 0.37], begin_time=t0, transition_time=1)
        t0 = 1.5 + t0

        self.t0 = t0

    def family_b4(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }B_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[4, 384],
                           tic_labels={"8": 8, "16": 16, "24": 24, "32": 32, "48": 48, "64": 64, "96": 96, "192": 192,
                                       "384": 384}, tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        dias = []
        family = [["x3x3x4x"], ["o3x3x4x", "x3x3o4x", "x3x3x4o", "x3o3x4x"],
                  ["o3o3x4x", "x3o3o4x", "x3x3o4o", "o3x3x4o", "o3x3o4x", "x3o3x4o"],
                  ["o3o3o4x", "x3o3o4o", "o3x3o4o", "o3o3x4o"]]
        for l, level in enumerate(family):
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    if l < 3:
                        dx = 16 / (level_count - 1)
                    else:
                        dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([4, 384], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge")
                dias.append(dia2)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        morphing_start_time = t0

        morpher = Morpher4D(CoxB4, ["x3x3x4x", "o3x3x4x", "o3o3x4x", "o3o3o4x",
                                    "x3x3x4x", "x3x3o4x", "x3o3o4x", "x3o3o4o",
                                    "x3x3o4x", "x3x3o4o", "o3x3o4o",
                                    "x3x3x4x", "x3x3x4o", "o3x3x4o", "o3o3x4o",
                                    "x3x3x4o", "o3x3o4x",
                                    "x3x3x4x", "x3o3x4x", "x3o3x4o"
                                    ],
                            rotation_euler=[0, 0, -118 / 180 * pi], scale=0.15, location=[8.26, 0, -2.6],
                            name="Morpher", limit_direction=Vector([-1, 0, 0]), limit=0.1)
        t0 = 0.5 + morpher.morph(begin_time=morphing_start_time, transition_time=1, pause=1, scale_elements=0.95)

        external_rotation = Matrix(((1.0, 0, 0.0, 0),
                                    (0, 1.0, 0.0, -0),
                                    (0.0, 0.0, 1.0, 0.0),
                                    (0, 0, 0.0, 1.0))
                                   )
        appear_times = [0, 2, 4, 6, 10, 12, 14, 18, 20, 24, 26, 28, 32, 36, 38]
        cd_labels = ["x3x3x4x", "o3x3x4x", "o3o3x4x", "o3o3o4x",
                     "x3x3o4x", "x3o3o4x", "x3o3o4o",
                     "x3x3o4o", "o3x3o4o",
                     "x3x3x4o", "o3x3x4o", "o3o3x4o",
                     "o3x3o4x",
                     "x3o3x4x", "x3o3x4o"
                     ]
        unfolders = [Unfolder4D2(CoxB4, cd_label, external_rotation=external_rotation) for cd_label in
                     cd_labels]
        locations = [
            Vector([2.8, 0, 5.4]),
            Vector([-7.7, 0, 5.3]),
            Vector([-7.9, 0, 1.6]),
            Vector([-8, 0, -1.76]),

            Vector([-3.6, 0, 5.1]),
            Vector([-4.8, 0, 1.7]),
            Vector([-4.8, 0, -3.8]),

            Vector([-1.5, 0, 1.2]),
            Vector([-1.5, 0, -4.2]),

            Vector([5, 0, 3.6]),
            Vector([1.73, 0, 0]),
            Vector([1.8, 0, -3.6]),

            Vector([4.8, 0, -0.4]),

            Vector([10.6, 0, 3.6]),
            Vector([10.6, 0, 0.6]),
        ]
        cell_removal_lists = [
            [76, 19],
            [9],
            [5],
            [5],

            [64, 9],
            [57, 74, 61, 52, 55, 72],
            [],

            [],
            [],

            [23, 31],
            [23],
            [1],

            [20],  # o3x3o4x

            [8, 73, 6],
            [21, 42],
        ]
        explode_scales = [
            1.5, 1.5, 1.5, 3, 2, 2, 6, 4, 4, 1.5, 1.5, 1.5, 1.5, 1.5, 2
        ]

        removals = []
        for unfolder, time, location, signature, cell_removals, explode_scale in zip(unfolders, appear_times, locations,
                                                                                     cd_labels, cell_removal_lists,
                                                                                     explode_scales):
            bob = unfolder.create_stereo_bob(location=location, scale=0.05, name=signature, half_way=False, limit=0,
                                             limit_direction=Vector([0, 0, 1]),
                                             rotation_euler=Vector([0, 0, pi / 2]), cell_removals=cell_removals)
            bob.move_to(target_location=location, begin_time=0, transition_time=0)
            bob.appear(begin_time=time + morphing_start_time, transition_time=1, children=True, sequentially=True)
            bob.explode(explode_scale=explode_scale, begin_time=time + morphing_start_time + 1, transition_time=1)
            removals.append(bob)

        # additional motion creating space for popups
        title.move(direction=[9, 0, 0], begin_time=morphing_start_time - 1, transition_time=1)
        # morpher.bob.move(direction=[-2, 0, -1], begin_time=morphing_start_time + 15, transition_time=1)
        morpher.bob.rescale(rescale=0.8, begin_time=morphing_start_time + 26, transition_time=1)
        #morpher.bob.rescale(rescale=2, begin_time=morphing_start_time + 21, transition_time=1)
        # morpher.bob.disappear(begin_time=morphing_start_time + 25, transition_time=1)

        dias[8].move(direction=[0, 0, 0.3], begin_time=morphing_start_time + 26, transition_time=1)

        t0 = 0.5 + show_panel(["x3x4x", "x3x4o", "x3o4x", "o3x4x", "o3o4x", "o3x4o", "x3o4o",
                               "x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o",
                               "x3x.x", "x3o.x", "x4x.x", "x4o.x"],
                              scales=[0.096, 0.225, 0.19, 0.4, 0.4, 0.38, 0.5, 0.24,
                                      0.13, 0.4, 0.377, 0.445,
                                      0.42, 0.65, 0.34, 0.65],
                              dynkin_scale=0.5, columns=2,
                              begin_time=t0, transition_time=1,
                              dynkin_shift=Vector([-0.15, -1, 0]),
                              poly_shift=Vector([-1.55, -0.9, 0]), dynkin_label_shift=Vector([-0.5, 0, 0.6]))

        for bob in removals:
            bob.disappear(begin_time=t0, transition_time=1)

        title.move(direction=[-9, 0, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + morpher.bob.disappear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_h4(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }H_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[60, 14400],
                           tic_labels={"120": 120, "600": 600, "720": 720, "1200": 1200, "1440": 1440,
                                       "2400": 2400, "3600": 3600, "7200": 7200, "14400": 14400},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1.3, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        dias = []
        family = [["x3x3x5x"], ["o3x3x5x", "x3x3o5x", "x3x3x5o", "x3o3x5x"],
                  ["o3o3x5x", "x3o3o5x", "x3x3o5o", "o3x3x5o", "o3x3o5x", "x3o3x5o"],
                  ["o3o3o5x", "x3o3o5o", "o3x3o5o", "o3o3x5o"]]
        for l, level in enumerate(family):
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    if l < 3:
                        dx = 16 / (level_count - 1)
                    else:
                        dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([60, 14400], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge")
                dias.append(dia2)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        morphing_start_time = 18.5  #t0

        morpher = Morpher4D(CoxH4, ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x",
                                    "x3x3x5x", "x3x3o5x", "x3o3o5x", "x3o3o5o",
                                    "x3x3o5x", "x3x3o5o", "o3x3o5o",
                                    "x3x3x5x", "x3x3x5o", "o3x3x5o", "o3o3x5o",
                                    "x3x3x5o", "o3x3o5x",
                                    "x3x3x5x", "x3o3x5x", "x3o3x5o"
                                    ],
                            rotation_euler=[-pi / 2, 173 / 180 * pi, -21 / 180 * pi], scale=0.05,
                            location=[7.82, 0, -2.48],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1, south_pole=1,
                            mode="CELL_SIZE", cell_size=20, skip=0, cell_removals=[1858])
        t0 = 0.5 + morpher.morph(begin_time=morphing_start_time, transition_time=1, pause=1)

        external_rotation = Matrix(((0.0000, -1.0000, 0.0000, -0.0000),
                                    (-0.5878, 0.0000, 0.4253, -0.6882),
                                    (0.0000, 0.0000, 0.8507, 0.5257),
                                    (0.8090, 0.0000, 0.3090, -0.5000)))

        appear_times = [0, 2, 4, 6, 10, 12, 14, 18, 20, 24, 26, 28, 32, 36, 38]
        signatures = ["x3x3x5x", "o3x3x5x", "o3o3x5x", "o3o3o5x",
                      "x3x3o5x", "x3o3o5x", "x3o3o5o",
                      "x3x3o5o", "o3x3o5o",
                      "x3x3x5o", "o3x3x5o", "o3o3x5o",
                      "o3x3o5x",
                      "x3o3x5x", "x3o3x5o"]

        unfolders = [Unfolder4D2(CoxH4, sig, external_rotation=external_rotation, no_tree=True) for sig in signatures]
        locations = [
            Vector([2.8, 0, 5.4]),
            Vector([-7.7, 0, 5.3]),
            Vector([-7.9, 0, -0.2]),
            Vector([-8, 0, -3.96]),

            Vector([-3.6, 0, 5.36]),
            Vector([-4.8, 0, -0.7]),
            Vector([-4.8, 0, -3.8]),

            Vector([-1.5, 0, 1.7]),
            Vector([-1.5, 0, -4.2]),

            Vector([5, 0, 3.6]),
            Vector([1.73, 0, 0]),
            Vector([1.8, 0, -3.6]),

            Vector([5.7, 0, -0.4]),

            Vector([10.6, 0, 3.6]),
            Vector([10.6, 0, 0.20]),
        ]
        cell_removal_lists = [
            [1981, 1858, 1972, 1990, 2005],
            [11, 39, 72],
            [46, 38, 85],
            [11, 106],

            [511, 512, 2016, 1936],
            [2541, 2472, 2572, 1858],
            [],

            [32],
            [5],

            [76, 1268],
            [13, ],  # o3x3o4x
            [18, 95],

            [15],

            [1955, 1209, 1637],
            [1362, 1077],
        ]
        explode_scales = [
            1.5, 1.5, 1.5, 5, 2, 2.6, 6, 5.2, 5.2, 1.5, 1.9, 2, 1.5, 2, 2.6
        ]

        removables = []
        for i, (unfolder, time, location, signature, cell_removals, explode_scale) in enumerate(
                zip(unfolders, appear_times, locations,
                    signatures, cell_removal_lists,
                    explode_scales)):
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                bob = unfolder.create_stereo_bob(location=location, scale=0.015, name=signature, half_way=False,
                                                 limit=0,
                                                 limit_direction=Vector([0, 0, 1]),
                                                 rotation_euler=Vector([pi / 2, 0, 0]), cell_removals=cell_removals)
                bob.move_to(target_location=location, begin_time=0, transition_time=0)
                bob.appear(begin_time=time + morphing_start_time, transition_time=1, children=True, sequentially=True)
                bob.explode(explode_scale=explode_scale, begin_time=time + morphing_start_time + 1, transition_time=1)
                removables.append(bob)

        # additional motion creating space for popups
        title.move(direction=[9, 0, 0], begin_time=morphing_start_time, transition_time=1)
        morpher.bob.move_to(target_location=[9.2, 0, -3.4], begin_time=morphing_start_time + 32, transition_time=1)
        morpher.bob.rescale(rescale=0.6, begin_time=morphing_start_time + 26, transition_time=1)
        # morpher.bob.rescale(rescale=1.25, begin_time=morphing_start_time + 20, transition_time=1)
        # morpher.bob.disappear(begin_time=morphing_start_time + 25, transition_time=1)
        dias[-1].move(direction=[0, 0, -1.48], begin_time=morphing_start_time + 33, transition_time=1)
        # dias[6].move(direction=[-0.5, 0, 0.37], begin_time=morphing_start_time + 21, transition_time=1)
        removables.append(morpher.bob)

        show_panel([
            "x3x5x", "x3x5o", "x3o5x", "o3x5x", "o3o5x", "o3x5o", "x3o5o",
            "x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o",
            "x3x.x", "x3o.x", "x5x.x", "x5o.x"
        ],
            scales=[0.096, 0.09, 0.122, 0.146, 0.303, 0.332, 0.303, 0.24,
                    0.13, 0.4, 0.377, 0.445,
                    0.42, 0.65, 0.34, 0.65],
            dynkin_scale=0.75, columns=2,
            show_time=9,
            begin_time=2, transition_time=1,
            dynkin_shift=Vector([-0.15, -1, 0]),
            poly_shift=Vector([-1.55, -0.9, 0]), dynkin_label_shift=Vector([-0, 0, 0.6]))

        last_camera_location = Vector([0, -21, 0])
        ibpy.camera_zoom(lens=100, begin_time=t0, transition_time=2)
        for location in locations:
            camera_empty.move_to(target_location=location, begin_time=t0, transition_time=1)
            vector_location = Vector(location) + Vector([0, -21, 0])
            t0 = 0.5 + ibpy.camera_move(shift=vector_location - last_camera_location, begin_time=t0, transition_time=1)
            last_camera_location = vector_location

        ibpy.camera_move(shift=-last_camera_location + Vector([0, -21, 0]), begin_time=t0, transition_time=1)
        ibpy.camera_zoom(lens=30, begin_time=t0, transition_time=1)
        t0 = 0.5 + camera_empty.move_to(target_location=Vector(), begin_time=t0, transition_time=1)

        # reset everything
        title.move(direction=[-9, 0, 0], begin_time=t0, transition_time=1)
        dias[-1].move(direction=[0, 0, 1.48], begin_time=t0, transition_time=1)
        for bob in removables:
            bob.disappear(begin_time=t0, transition_time=1)
        t0 += 1.5

        self.t0 = t0

    def family_f4_nets(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }F_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = title.write(begin_time=t0, transition_time=0)

        axis = NumberLine2(length=12, domain=[20, 1152],
                           tic_labels={"24": 24, "96": 96, "144": 144, "192": 192, "288": 288,
                                       "576": 576, "1152": 1152},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = axis.grow(begin_time=t0, transition_time=0)
        axis.to_log(begin_time=0, transition_time=0)

        dias = []
        family = [["x3x4x3x"], ["o3x4x3x", "x3x4o3x"],
                  ["o3o4x3x", "o3x4x3o", "x3o4o3x", "o3x4o3x"],
                  ["o3o4x3o", "o3o4o3x"]]
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 16 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([20, 1152], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge",
                                                 no_threes=True)
                dias.append(dia2)
                t0 = dia2.appear(begin_time=t0, transition_time=0)

        morphing_start_time = t0
        morpher_center = [-1.21, 0, -3]

        # first morph sequence
        morpher = NetMorpher4D(CoxF4, ["x3x4x3x", "o3x4x3x", "o3o4x3x", "o3o4x3o"], name="MorpherF4_a",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=t0, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear
        appear_times = [0, 3, 6, 9]
        locations = [
            Vector([-5.1, 0, 4.9]),
            Vector([-6.9, 0, 1.34]),
            Vector([-5.74, 0, -1.36]),
            Vector([-6.91, 0, -4.88]),

        ]

        states = [morpher.get_state(i) for i in range(4)]
        scales = [0.65, 0.5, 0.5, 0.7]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # second morph sequence

        morpher = NetMorpher4D(CoxF4, ["o3x4x3x", "o3x4x3o"], name="MorpherF4_b",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=1)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear
        appear_times = [3]
        locations = [
            Vector([-1.85, 0, 2.99]),
        ]

        states = [morpher.get_state(i) for i in range(1, 2)]
        scales = [0.5]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        # dias[2].move(direction=[0, 0, 0.3], begin_time=morphing_start_time + 3 + 1, transition_time=0.5)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 6, transition_time=1)

        # third morph sequence

        morpher = NetMorpher4D(CoxF4, ["x3x4x3x", "x3o4x3x", "x3o4o3x"], name="MorpherF4_c",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=1)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear
        appear_times = [3, 6]
        locations = [
            Vector([9.998, 0, 4.35]),
            Vector([2.52, 0, 1.69]),
        ]

        states = [morpher.get_state(i) for i in range(1, 3)]
        scales = [0.7, 0.7]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        dias[2].move(direction=[0, 0, -0.5], begin_time=morphing_start_time + 3, transition_time=0.5)
        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 9, transition_time=1)

        # fourth morph sequence

        morpher = NetMorpher4D(CoxF4, ["x3x4x3x", "o3x4o3x", "o3o4o3x"], name="MorpherF4_d",
                               scale=0.25, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=2)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        t0 = morphing_start_time
        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear
        appear_times = [3, 6]
        locations = [
            Vector([9.07, 0, -1.9]),
            Vector([4.85, 0, -4.35]),
        ]

        states = [morpher.get_state(i) for i in range(1, 3)]
        scales = [0.75, 0.75]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 9, transition_time=1)

        t0 = 0.5 + show_panel(["x3x4x", "x3x4o", "x3o4x", "o3x4x", "o3o4x", "o3x4o", "x3o4o",
                               "x3x.x", "x3o.x"],
                              scales=[0.096, 0.225, 0.19, 0.4, 0.4, 0.38, 0.5,
                                      0.42, 0.65],
                              global_shift=Vector([0, -1, 0]), dynkin_shift=Vector([-0.15, 0, 0]),
                              poly_shift=Vector([0, 0, 0]),
                              dynkin_label_shift=Vector([-0.2, 0, 0.2]),
                              begin_time=morphing_start_time, transition_time=1)

        self.t0 = t0

    def family_f4(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }F_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[20, 1152],
                           tic_labels={"24": 24, "96": 96, "144": 144, "192": 192, "288": 288,
                                       "576": 576, "1152": 1152},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        dias = []
        family = [["x3x4x3x"], ["o3x4x3x", "x3x4o3x"],
                  ["o3o4x3x", "o3x4x3o", "x3o4o3x", "o3x4o3x"],
                  ["o3o4x3o", "o3o4o3x"]]
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 16 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([20, 1152], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge",
                                                 no_threes=True)
                dias.append(dia2)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        morphing_start_time = t0

        morpher = Morpher4D(CoxF4, ["x3x4x3x", "o3x4x3x", "x3x4o3o", "o3x4o3o",
                                    "o3x4x3x", "o3x4x3o",
                                    "x3x4x3x", "x3o4x3x", "x3o4o3x",
                                    "x3o4x3x", "x3o4x3o", "x3o4o3o", "x3x4x3x"],
                            rotation_euler=[-pi / 2, 0, 0], scale=0.1, location=[3.3, 0, -4.2],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1)
        t0 = 0.5 + morpher.morph(begin_time=morphing_start_time, transition_time=1, pause=1, scale_elements=0.95)

        external_rotation = Matrix(((1.0, 0, 0, 0),
                                    (0, 1.0, 0, 0),
                                    (0, 0, 1.0, 0),
                                    (0, 0, 0, 1.0)))

        appear_times = [0, 2, 4, 6, 10, 14, 16, 20, 22]
        cd_strings = ["x3x4x3x", "o3x4x3x",
                      "x3x4o3o", "o3x4o3o", "o3x4x3o", "x3o4x3x", "x3o4o3x",
                      "o3x4o3x", "o3o4o3x"]
        unfolders = [Unfolder4D2(CoxF4, cd_string, external_rotation=external_rotation) for cd_string in cd_strings]
        locations = [
            Vector([4.17, 0, 5.05]),
            Vector([-6.5, 0, 5.1]),
            Vector([-7.3, 0, 1.2]),
            Vector([-7.4, 0, -4.5]),

            Vector([-2.65, 0, -2.2]),

            Vector([9.4, 0, 4.5]),
            Vector([1.5, 0, 1.5]),

            Vector([8.9, 0, -1.5]),
            Vector([10, 0, -5.2]),
        ]
        cell_removal_lists = [
            [20, 132],
            [1, 128],
            [14, 36],
            [19, 35],

            [24, 41],

            [26, 231],
            [190],

            [27, 142],
            [],
        ]
        explode_scales = [
            2, 2.5, 2, 3, 4, 2, 3, 3.5, 6
        ]

        remove_bobs = []
        for unfolder, time, location, signature, cell_removals, explode_scale in zip(unfolders, appear_times,
                                                                                     locations,
                                                                                     cd_strings, cell_removal_lists,
                                                                                     explode_scales):
            bob = unfolder.create_stereo_bob(location=location, scale=0.05, name=signature, half_way=False, limit=0,
                                             limit_direction=Vector([0, 0, 1]),
                                             rotation_euler=pi / 180 * Vector([45, 0, 0]), cell_removals=cell_removals)
            bob.move_to(target_location=location, begin_time=0, transition_time=0)
            bob.appear(begin_time=time + morphing_start_time, transition_time=1, children=True, sequentially=True)
            bob.explode(explode_scale=explode_scale, begin_time=time + morphing_start_time + 1, transition_time=1)
            remove_bobs.append(bob)

        # additional motion creating space for popups
        dias[2].move(direction=[-1.3, 0, 0], begin_time=morphing_start_time + 14, transition_time=1)
        morpher.bob.rotate(rotation_euler=[-pi / 2, 0, tau], begin_time=morphing_start_time + 25, transition_time=5)

        t0 = show_panel(["x3x4x", "x3x4o", "x3o4x", "o3x4x", "o3o4x", "o3x4o", "x3o4o",
                         "x3x.x", "o3x.x"],
                        scales=[0.096, 0.225, 0.19, 0.4, 0.4, 0.38, 0.5,
                                0.42, 0.65], global_shift=Vector([0, -1, 0]), dynkin_shift=Vector([-0.15, -1, 0]),
                        dynkin_label_shift=Vector([-0.2, 0, 0.2]),
                        begin_time=morphing_start_time + 22, transition_time=1)

        for bob in remove_bobs:
            bob.disappear(begin_time=t0, transition_time=1)
        morpher.bob.disappear(begin_time=t0, transition_time=1)
        dias[2].move(direction=[1.3, 0, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        self.t0 = t0

    def family_d4_nets(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }D_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = title.write(begin_time=t0, transition_time=0)

        axis = NumberLine2(length=12, domain=[6, 192], tic_labels={"8": 8, "24": 24, "32": 32, "48": 48, "96": 96,
                                                                   "192": 192},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = axis.grow(begin_time=t0, transition_time=0)
        axis.to_log(begin_time=0, transition_time=0)

        dias = []
        family = [["x3x3x *b3x"], ["x3o3x *b3x", "o3x3x *b3x"],
                  ["o3o3x *b3x", "o3x3o *b3x"],
                  ["o3o3o *b3x", "o3x3o *b3o"]]
        for l, level in enumerate(family):
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 18 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([6, 192], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge",
                                                 no_threes=True)
                dias.append(dia2)
                t0 = dia2.appear(begin_time=t0, transition_time=0)

        morphing_start_time = t0
        title_move_time = t0
        morpher_center = [1.0, 0, -2.5]

        # first morph sequence
        morpher = NetMorpher4D(CoxD4, ["x3x3x *b3x", "x3o3x *b3x", "o3o3x *b3x", "o3o3o *b3x"], name="MorpherD4_a",
                               scale=0.325, location=morpher_center)

        morpher.morph_sequence(begin_time=t0, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=3)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 6,12, 18 are back morphing times
        appear_times = [0, 3, 6, 9]
        locations = [
            Vector([3.83, 0, 3.788]),
            Vector([-6.11, 0, 4]),
            Vector([-5.4, 0, -0.587]),
            Vector([-5.2, 0, -5.63]),

        ]

        states = [morpher.get_state(i) for i in range(4)]
        scales = [0.75, 0.65, 0.65, 0.65]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        # second morph sequence

        morpher = NetMorpher4D(CoxD4, ["x3x3x *b3x", "o3x3x *b3x", "o3x3o *b3x", "o3x3o *b3o"], name="MorpherD4_b",
                               scale=0.325, location=morpher_center)

        morpher.morph_sequence(begin_time=morphing_start_time, transition_time=2, pause=1, sequentially=True)
        main_axis = morpher.get_main_axis(state=3)
        # rotate main axis into x-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(omega, np.arccos(main_axis[0]))
        quat_1 = Quaternion(Vector([1, 0, 0]), pi)

        morpher.bob.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_0, begin_time=t0, transition_time=5)
        t0 = 0.01 + morpher.bob.rotate(rotation_quaternion=quat_1 @ quat_1 @ quat_1 @ quat_0, begin_time=t0,
                                       transition_time=5)

        # make individual projections appear at 3 are back morphing times
        appear_times = [3, 6, 9]
        locations = [
            Vector([9.5, 0, 3.97]),
            Vector([7, 0, -0.75]),
            Vector([9.6, 0, -4.48]),

        ]

        states = [morpher.get_state(i) for i in range(1, 4)]
        scales = [0.65, 0.65, 0.68]

        for time, scale, state, location in zip(appear_times, scales, states, locations):
            state.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
            state.move_to(target_location=location, begin_time=0, transition_time=0)
            state.rescale(rescale=scale, begin_time=0, transition_time=0)
            state.appear(begin_time=time + morphing_start_time + 1, transition_time=1, children=True, sequentially=True)

        title.move(direction=[-1.2, 0, 0], begin_time=title_move_time, transition_time=1)

        dias[2].move(direction=[1.75, 0, -0.48], begin_time=morphing_start_time + 3, transition_time=0.5)
        dias[6].move(direction=[1.7, 0, 0], begin_time=morphing_start_time + 9, transition_time=0.5)

        morphing_start_time = 0.5 + morpher.bob.disappear(begin_time=morphing_start_time + 12, transition_time=1)

        t0 = 0.5 + show_panel(["x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o", "x.x.x"],
                              scales=[0.33, 0.23, 0.5, 0.5, 0.6, 0.5, 0.8, 0.8], global_shift=Vector([0, -1, 0]),
                              dynkin_shift=Vector([-0.15, -1, 0]),
                              dynkin_label_shift=Vector([-0.2, 0, 0.2]),
                              begin_time=morphing_start_time, transition_time=1)

        self.t0 = t0

    def family_d4_and_b4(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        cd_d4 = DynkinDiagram.from_string("x3x3x *b3x", location=[-7, 0, 0.5], no_threes=True)
        cd_d4.appear(begin_time=t0, transition_time=1)

        unfolder = Unfolder4D2(CoxD4, "x3x3x *b3x")
        d4_net = unfolder.create_net(location=[4.65, 0, 0.62], scale=0.55, scale_elements=0.99)
        t0 = d4_net.appear(begin_time=t0, transition_time=1, sequentially=True)

        removal = [cd_d4, d4_net]
        for ob in removal:
            ob.disappear(begin_time=t0, transition_time=1)

        t0 += 1.5

        cd_b4 = DynkinDiagram.from_string("x3x3x4o", location=[8.3, 0, 0], no_threes=True)
        cd_b4.appear(begin_time=t0, transition_time=1)

        unfolder = Unfolder4D2(CoxB4, "x3x3x4o")
        b4_net = unfolder.create_net(scale=0.55, location=[-3.18, 0, 0], scale_elements=0.99)
        t0 = b4_net.appear(begin_time=t0, transition_time=1, sequentially=True)

        removal = [cd_b4, b4_net]
        for ob in removal:
            ob.disappear(begin_time=t0, transition_time=1)

        t0 += 1.5

        self.t0 = t0

    def demi_tesseract(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        unfolder = Unfolder4D2(CoxB4, "o3o3o4x")

        external_rotation = Matrix([
            [0.7071, 0.7071, 0.0000, 0.0000],
            [0.7071, 0.7071, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ])

        pc = unfolder.point_cloud
        print(pc)
        points_real = [v.real() for v in pc]

        edges = []
        min_dist = np.inf
        for i in range(len(pc)):
            for j in range(i + 1, len(pc)):
                dist = (points_real[i] - points_real[j]).length
                if dist < 2.1:
                    edges.append([i, j])

        stereo_points = unfolder.stereo
        print(stereo_points)

        mesh = BObject(mesh=ibpy.create_mesh(stereo_points, edges=edges))
        modifier = PolyhedronViewModifier()
        mesh.add_mesh_modifier(type="NODES", node_modifier=modifier)
        t0 = 0.5 + mesh.appear(begin_time=t0, transition_time=3)

        # parity: remove all points with one or three negative entries

        survivors = []
        pc_loc = [p for p in pc]
        #take first point and remove all direct neighbors then add next point and iterate

        while len(pc_loc) > 0:
            survivors.append(pc_loc[0])
            excluded = []

            for i, p in enumerate(pc_loc):
                if (p - survivors[-1]).real().length < 2.1:
                    excluded.append(i)

            while len(excluded) > 0:
                exclude = excluded.pop()
                pc_loc.remove(pc_loc[exclude])

        # now there should be 24 edges connecting the remaining 8 vertices
        new_edges = []
        for i in range(len(survivors)):
            for j in range(i + 1, len(survivors)):
                dist = (survivors[i] - survivors[j]).real().length
                if dist < 2.9:
                    new_edges.append([i, j])

        print(len(new_edges))

        # identify survived indices:
        survived_indices = [pc.index(survivor) for survivor in survivors]

        print(survived_indices)

        # create 16-cell mesh

        stereo_points2 = [stereo_points[idx] for idx in survived_indices]
        mesh2 = BObject(mesh=ibpy.create_mesh(stereo_points2, edges=new_edges))
        modifier = PolyhedronViewModifier(edge_color="joker")
        mesh2.add_mesh_modifier(type="NODES", node_modifier=modifier)
        t0 = 0.5 + mesh2.appear(begin_time=t0, transition_time=3)
        mesh.disappear(alpha=0.1, begin_time=t0, transition_time=1)

        container = BObject(children=[mesh, mesh2])
        container.appear(begin_time=0, transition_time=0)
        container.rotate(rotation_euler=[0, 0, tau], begin_time=0, transition_time=10)
        # stereo_bob = unfolder.create_stereo_bob()
        # t0 = 0.5 + stereo_bob.appear(begin_time=t0,transition_time=3)
        self.t0 = t0

    def family_d4(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }D_4", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        axis = NumberLine2(length=12, domain=[6, 192], tic_labels={"8": 8, "24": 24, "32": 32, "48": 48, "96": 96,
                                                                   "192": 192},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        family = [["x3x3x *b3x"], ["x3o3x *b3x", "o3x3x *b3x"],
                  ["o3o3x *b3x", "o3x3o *b3x"],
                  ["o3o3o *b3x", "o3x3o *b3o"]]
        dias = []

        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 18 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([6, 192], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.25, text_size="Huge",
                                                 no_threes=True)
                dias.append(dia2)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)
        morphing_start_time = t0

        # customized rotation that centers one cell around y-pole
        external_rotation = Matrix(
            ((0.5000, 0.5000, 0.5000, 0.5000),
             (0.7071, 0.0000, -0.7071, 0.0000),
             (0.2887, -0.8660, 0.2887, 0.2887),
             (0.4082, 0.0000, 0.4082, -0.8165)))

        appear_times = [0, 2, 4, 6, 10, 12, 14]

        cd_strings = ["x3x3x *b3x", "x3o3x *b3x", "o3o3x *b3x", "o3o3o *b3x", "o3x3x *b3x", "o3x3o *b3x", "o3x3o *b3o"]
        unfolders = [Unfolder4D2(CoxD4, cd_string, external_rotation=external_rotation) for cd_string in cd_strings]
        locations = [
            Vector([4.88, 0, 4.85]),
            Vector([-5.24, 0, 4.49]),
            Vector([-5.88, 0, -0.82]),
            Vector([-6, 0, -4.48]),

            Vector([9.73, 0, 4.76]),
            Vector([6.4, 0, -0.5]),
            Vector([7.3, 0, -4.75]),

        ]
        cell_removal_lists = [
            [5, 16],
            [45],
            [5, 21],
            [11, 14],

            [2, 20],
            [7, 21],
            [8, 13],

        ]
        explode_scales = [
            2, 2, 2, 4, 2, 2, 3
        ]

        remove_bobs = []
        for unfolder, t, location, signature, cell_removals, explode_scale in zip(unfolders, appear_times, locations,
                                                                                  cd_strings, cell_removal_lists,
                                                                                  explode_scales):
            bob = unfolder.create_stereo_bob(location=location, scale=0.1, name=signature, half_way=False, limit=0,
                                             limit_direction=Vector([0, 0, 1]),
                                             rotation_euler=pi / 2 * Vector([1, 0, 0]), cell_removals=cell_removals)
            bob.move_to(target_location=location, begin_time=0, transition_time=0)
            bob.appear(begin_time=t + morphing_start_time, transition_time=1, children=True, sequentially=True)
            bob.explode(explode_scale=explode_scale, begin_time=t + morphing_start_time + 1, transition_time=1)
            remove_bobs.append(bob)

        morpher = Morpher4D(CoxD4, ["x3x3x *b3x", "x3o3x *b3x", "o3o3x *b3x", "o3o3o *b3x",
                                    "x3x3x *b3x", "o3x3x *b3x", "o3x3o *b3x", "o3x3o *b3o"],
                            rotation_euler=[-pi / 2, 0 / 180 * pi, 0 / 180 * pi], scale=0.25,
                            location=[0, 0, -2.25],
                            name="Morpher", limit_direction=Vector([0, 0, 1]), limit=0.1, south_pole=1,
                            mode="CELL_SIZE", cell_size=24, skip=0, cell_removals=[5], scale_elements=0.99)
        t0 = 0.5 + morpher.morph(begin_time=morphing_start_time, transition_time=1, pause=1)

        # additional motion creating space for popups
        morpher.bob.disappear(begin_time=morphing_start_time + 17, transition_time=1)
        dias[2].move(direction=[1.5, 0, 0], begin_time=morphing_start_time + 10, transition_time=1)
        dias[3].move(direction=[-0.75, 0, 0], begin_time=morphing_start_time + 4, transition_time=1)

        t0 = show_panel(["x3x3x", "x3x3o", "x3o3x", "o3o3x", "o3x3o", "x.x.x"],
                        scales=[0.33, 0.23, 0.5, 0.5, 0.6, 0.5, 0.8, 0.8], global_shift=Vector([0, -1, 0]),
                        begin_time=morphing_start_time + 14, transition_time=1)

        for bob in remove_bobs:
            bob.disappear(begin_time=t0, transition_time=1)
        dias[2].move(direction=[-1.5, 0, 0], begin_time=t0, transition_time=1)
        dias[3].move(direction=[0.75, 0, 0], begin_time=t0, transition_time=1)

        t0 += 1.5

        self.t0 = t0

    def family_a3(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), no_transmission_ray=False,
                                 reflections=False, reflection_color=[0.05, 0.05, 0.05, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        rotatables = []
        title = Text(r"\text{Coxeter Group }A_3", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)
        set_origin(title)
        rotatables.append(title)

        axis = NumberLine2(length=12, domain=[3, 24], tic_labels={"4": 4, "6": 6, "12": 12, "24": 24},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)
        rotatables.append(axis)

        family = [["x3x3x"], ["x3o3x", "o3x3x"], ["o3o3x", "o3x3o"]]

        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 16 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([3, 24], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                rotatables.append(dia2)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        group = CoxA3()
        (n_a, n_b, n_c) = [group.normals[i].real() for i in range(3)]

        size = 4
        plane_a = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_A", normal=n_a, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_c = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_C", shadow=False, normal=n_c,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_b = Plane(u=[-size, size], v=[-size, size], normal=n_b, color='mirror', name="Mirror_B", shadow=False,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)

        camera_shift = Vector([-2, 0, 10])
        camera_location += camera_shift
        ibpy.camera_move(shift=camera_shift, begin_time=t0, transition_time=1)

        for rotatable in rotatables:
            rotatable.rotate(
                rotation_quaternion=ibpy.camera_alignment_quaternion(camera_empty, camera_location, default=(0, -1, 0)),
                begin_time=t0, transition_time=1)

        floor = Floor(u=[-size, size], v=[-size, size], checker_scale=10, coords="Generated")
        floor.move_to(target_location=[0, 0, -4], begin_time=0, transition_time=0)
        t0 = 0.5 + floor.appear(begin_time=t0, transition_time=1)

        mirrors = [plane_a, plane_b, plane_c]
        for mirror in mirrors:
            t0 = 0.5 + mirror.grow(begin_time=t0, transition_time=1)

        trunc_octa = PolyhedronWithModifier.from_group_signature(CoxA3, COXA3_SIGNATURES["TRUNC_OCTA"],
                                                                 name="TruncatedOctahedron", color="color_dict")
        t0 = 0.5 + trunc_octa.grow(begin_time=t0, transition_time=1)

        copy_trunc_octa = trunc_octa.copy()
        copy_trunc_octa.appear(begin_time=t0, transition_time=0)
        copy_trunc_octa.rescale(rescale=0.6, begin_time=t0, transition_time=1)
        t0 = 0.5 + copy_trunc_octa.move_to(target_location=[5.75, 0, 4.5], begin_time=t0, transition_time=1)

        # # perform morphing of the solids
        derived_polyhedra = ["TRUNC_TETRA", "OCTA", "CUBOCTA", "TETRA"]
        signatures = [COXA3_SIGNATURES[poly] for poly in derived_polyhedra]

        src_vertices = group.get_real_point_cloud(signature=COXA3_SIGNATURES["TRUNC_OCTA"])
        src_radius = src_vertices[0].length

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_scale_factors = [src_radius / vertices[0].length for vertices in all_vertices]
        all_target_vertices = [[v * scale_factor for v in vertices] for scale_factor, vertices in
                               zip(all_scale_factors, all_vertices)]
        polyhedra = [PolyhedronWithModifier.from_group_signature(CoxA3, signatures[i], name=derived_polyhedra[i],
                                                                 color="color_dict", scale=all_scale_factors[i]) for i
                     in range(4)]

        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx, src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx, target_v in enumerate(target_vertices):
                    if (src_v - target_v).length < dist:
                        dist = (src_v - target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        locations = {
            "TRUNC_TETRA": [10, 0, -0.5],
            "CUBOCTA": [-7.8, 0, 3.3],
            "OCTA": [10, 0, -5],
            "TETRA": [-7.8, 0, -3.25],
        }

        transformations = []
        for j in range(4):
            transformations.append(lambda i, j=j: all_target_vertices[j][all_maps[j][i]])
            # reset after two tranformations
            if j == 1:
                transformations.append(lambda i: src_vertices[i])

        trunc_octa.index_transform_mesh(transformations=transformations, begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_colors(shape_key=1,
                                               face_classes=group.get_faces_in_conjugacy_classes(
                                                   COXA3_SIGNATURES["TRUNC_OCTA"]),
                                               begin_time=t0, transition_time=1)

        count = 0
        for i in range(4):
            if i != 2:
                polyhedra[count].appear(begin_time=t0, transition_time=0)
                polyhedra[count].rescale(rescale=0.5, begin_time=t0, transition_time=1)
                t0 = 0.5 + polyhedra[count].move_to(target_location=locations[derived_polyhedra[count]], begin_time=t0,
                                                    transition_time=1)
                count += 1

            trunc_octa.transform_colors(shape_key=i + 2, face_classes=group.get_faces_in_conjugacy_classes(
                COXA3_SIGNATURES["TRUNC_OCTA"]), begin_time=t0, transition_time=1)
            t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)
        polyhedra[count].appear(begin_time=t0, transition_time=0)
        polyhedra[count].rescale(rescale=0.6, begin_time=t0, transition_time=1)

        [mirror.disappear(begin_time=t0, transition_time=1) for mirror in mirrors]
        floor.disappear(begin_time=t0, transition_time=1)

        camera_location -= camera_shift
        ibpy.camera_move(shift=-camera_shift, begin_time=t0, transition_time=1)

        for rotatable in rotatables:
            rotatable.rotate(
                rotation_quaternion=ibpy.camera_alignment_quaternion(camera_empty, camera_location, default=(0, -1, 0)),
                begin_time=t0, transition_time=1)

        t0 = 0.5 + polyhedra[count].move_to(target_location=locations[derived_polyhedra[count]], begin_time=t0,
                                            transition_time=1)

        t0 = 0.5 + trunc_octa.disappear(begin_time=t0, transition_time=1)

        # morph from trunc_octa to trunc_tetra to octa
        # first round

        start_time = t0
        morpher = NetMorpher(group=CoxA3,
                             sequence=[COXA3_SIGNATURES[_] for _ in ["TRUNC_OCTA", "TRUNC_TETRA", "OCTA"]],
                             scale=0.3, location=Vector([-3.1, 0, -3.4]),
                             )
        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 1
        locations = [Vector([10.0, 0, 4.3]),
                     Vector([5.33, 0, 0.4]),
                     Vector([5.65, 0, -4.8])]

        colors = [
            ["hexagon", "hexagon", "square"],
            ["hexagon", "triangle", "square"],
            ["triangle", "triangle", "square"]
        ]
        for i in range(3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.5, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        # second morphing trunc_octa to cubocta to tetra

        start_time = t0
        morpher = NetMorpher(group=CoxA3,
                             sequence=[COXA3_SIGNATURES[_] for _ in ["TRUNC_OCTA", "CUBOCTA", "TETRA"]],
                             scale=0.3, location=Vector([-2.9, 0, 2.4]),
                             )
        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 3
        locations = [Vector([5.0, 0, 5]),
                     Vector([-3.9, 0, 3.2]),
                     Vector([-3.8, 0, -3.6])]

        colors = [
            ["square", "hexagon"],
            ["square", "triangle"],
            ["triangle", "hexagon"]
        ]
        for i in range(1, 3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.5, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_a3_outtake(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), no_transmission_ray=False,
                                 reflections=False, reflection_color=[0.05, 0.05, 0.05, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        group = CoxA3()
        (n_a, n_b, n_c) = [group.normals[i].real() for i in range(3)]

        size = 4
        plane_a = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_A", normal=n_a, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_c = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_C", shadow=False, normal=n_c,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_b = Plane(u=[-size, size], v=[-size, size], normal=n_b, color='mirror', name="Mirror_B", shadow=False,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)

        camera_shift = Vector([-2, 0, 10])
        camera_location += camera_shift
        ibpy.camera_move(shift=camera_shift, begin_time=t0, transition_time=1)

        floor = Floor(u=[-size, size], v=[-size, size], checker_scale=10, coords="Generated")
        floor.move_to(target_location=[0, 0, -4], begin_time=0, transition_time=0)
        t0 = 0.5 + floor.appear(begin_time=t0, transition_time=1)

        mirrors = [plane_a, plane_b, plane_c]
        for mirror in mirrors:
            t0 = 0.5 + mirror.grow(begin_time=t0, transition_time=1)

        trunc_octa = PolyhedronWithModifier.from_group_signature(CoxA3, COXA3_SIGNATURES["TRUNC_OCTA"],
                                                                 name="TruncatedOctahedron", color="color_dict")
        t0 = 0.5 + trunc_octa.grow(begin_time=t0, transition_time=1)

        # # perform morphing of the solids
        derived_polyhedra = ["TRUNC_TETRA", "OCTA", "CUBOCTA", "TETRA"]
        signatures = [COXA3_SIGNATURES[poly] for poly in derived_polyhedra]

        src_vertices = group.get_real_point_cloud(signature=COXA3_SIGNATURES["TRUNC_OCTA"])
        src_radius = src_vertices[0].length

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_scale_factors = [src_radius / vertices[0].length for vertices in all_vertices]
        all_target_vertices = [[v * scale_factor for v in vertices] for scale_factor, vertices in
                               zip(all_scale_factors, all_vertices)]
        polyhedra = [PolyhedronWithModifier.from_group_signature(CoxA3, signatures[i], name=derived_polyhedra[i],
                                                                 color="color_dict", scale=all_scale_factors[i]) for i
                     in range(4)]

        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx, src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx, target_v in enumerate(target_vertices):
                    if (src_v - target_v).length < dist:
                        dist = (src_v - target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        transformations = []
        for j in range(4):
            transformations.append(lambda i, j=j: all_target_vertices[j][all_maps[j][i]])
            # reset after two tranformations
            if j == 1:
                transformations.append(lambda i: src_vertices[i])

        trunc_octa.index_transform_mesh(transformations=transformations, begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_colors(shape_key=1,
                                               face_classes=group.get_faces_in_conjugacy_classes(
                                                   COXA3_SIGNATURES["TRUNC_OCTA"]),
                                               begin_time=t0, transition_time=1)

        count = 0
        for i in range(4):
            trunc_octa.transform_colors(shape_key=i + 2, face_classes=group.get_faces_in_conjugacy_classes(
                COXA3_SIGNATURES["TRUNC_OCTA"]), begin_time=t0, transition_time=1)
            t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        self.t0 = t0

    def show_panel_a3(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), no_transmission_ray=False,
                                 reflections=False, reflection_color=[0.05, 0.05, 0.05, 1])
        t0 = ibpy.set_hdri_strength(1.5, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        t0 = 0.5
        t0 = 0.5 + show_panel(["x3x", "o3x", "x.x"], poly_shift=Vector([-0.7, -0.25, 0]),
                              begin_time=t0, transition_time=1, show_time=10, dynkin_scale=2)

        self.t0 = t0

    def show_panel_b3(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), no_transmission_ray=False,
                                 reflections=False, reflection_color=[0.05, 0.05, 0.05, 1])
        t0 = ibpy.set_hdri_strength(1.5, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        t0 = 0.5
        t0 = 0.5 + show_panel(["x4x", "x4o", "x3x", "o3x", "x.x"], poly_shift=Vector([-0.7, -0.25, 0]),
                              begin_time=t0, transition_time=1, show_time=10, dynkin_scale=2)

        self.t0 = t0

    def family_b3_net(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        # ibpy.empty_blender_view3d()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }B_3", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        title.write(begin_time=t0, transition_time=0)
        set_origin(title)

        axis = NumberLine2(length=12, domain=[5, 48], tic_labels={"6": 6, "8": 8, "12": 12, "24": 24, "48": 48},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        axis.grow(begin_time=t0, transition_time=0)
        axis.to_log(begin_time=0, transition_time=0)

        family = [["x3x4x"], ["x3x4o", "x3o4x", "o3x4x"], ["x3o4o", "o3o4x", "o3x4o"]]
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([5, 48], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                dia2.appear(begin_time=t0, transition_time=0)

        group = CoxB3()

        # morph from trunc_cubocta to trunc_octa to octa
        start_time = t0
        morpher = NetMorpher(group=CoxB3,
                             sequence=[COXB3_SIGNATURES[_] for _ in ["TRUNC_CUBOCTA", "TRUNC_OCTA", "OCTA"]],
                             scale=0.3, location=Vector([11.5, 1.5, -0.5]),
                             )
        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 1
        locations = [Vector([5.0, 0, 5]),
                     Vector([-7.7, 0, 3.5]),
                     Vector([-8.2, 0, -4.3])]

        colors = [
            ["square", "octagon", "hexagon"],
            ["hexagon", "triangle", "tetragon"],
            ["triangle", "octagon", "hexagon"]
        ]
        for i in range(3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.5, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        # morph from trunc_cubocta to rhombicubocta to cube
        start_time = t0
        morpher = NetMorpher(group=CoxB3,
                             sequence=[COXB3_SIGNATURES[_] for _ in ["TRUNC_CUBOCTA", "RHOMBICUBOCTA", "CUBE"]],
                             rotation_quaternion=Quaternion([0, -1, 0], pi / 4),
                             location=Vector([8.1, 0, -4.1]), scale=0.275)

        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 1
        locations = [Vector([9.5, 0, 4.6]),
                     Vector([-3.5, 0, 3.13]),
                     Vector([-3.2, 0, -2.4])]
        colors = [
            ["square", "octagon", "hexagon"],
            ["tetragon", "square", "triangle"],
            ["tetragon", "octagon", "hexagon"]
        ]
        for i in range(3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.5, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        # morph from trunc_cubocta to trunc_cube to cube
        start_time = t0
        morpher = NetMorpher(group=CoxB3,
                             sequence=[COXB3_SIGNATURES[_] for _ in ["TRUNC_CUBOCTA", "TRUNC_CUBE", "CUBOCTA"]],
                             rotation_quaternion=Quaternion(),
                             location=Vector([3.5, 0, -0.4]), scale=0.3)

        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 3
        locations = [Vector([6.5, 0, 1.6]),
                     Vector([2.1, 0, 0]),
                     Vector([2.1, 0, -4.1])]
        colors = [
            ["square", "octagon", "hexagon"],
            ["triangle", "octagon", "tetragon"],
            ["triangle", "tetragon"]
        ]
        for i in range(1, 3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.4, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        # t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_h3_net(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        # ibpy.empty_blender_view3d()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }H_3", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        title.write(begin_time=t0, transition_time=0)
        set_origin(title)

        axis = NumberLine2(length=12, domain=[10, 120], tic_labels={"12": 12, "20": 20, "30": 30, "60": 60, "120": 120},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        axis.grow(begin_time=t0, transition_time=0)
        axis.to_log(begin_time=0, transition_time=0)

        family = [["x3x5x"], ["o3x5x", "x3o5x", "x3x5o"], ["x3o5o", "o3o5x", "o3x5o"]]
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([10, 120], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                dia2.appear(begin_time=t0, transition_time=0)

        group = CoxH3()

        # morph from trunc_icosidodeca to trunc icosa to icosa
        start_time = t0
        morpher = NetMorpher(group=CoxH3,
                             sequence=[COXH3_SIGNATURES[_] for _ in ["TRUNC_ICOSIDODECA", "TRUNC_ICOSA", "ICOSA"]],
                             scale=0.275, location=Vector([2.2, 0, -5.3]),
                             )
        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 1
        locations = [Vector([5.0, 0, 4.2]),
                     Vector([-7.7, 0, 3.5]),
                     Vector([-8.2, 0, -3.5])]

        colors = [
            ["decagon", "hexagon", "square"],
            ["pentagon", "hexagon", "tetragon"],
            ["triangle", "octagon", "hexagon"]
        ]
        for i in range(3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.5, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        # morph from trunc_icosidodeca to rhombicosidodeca to dodeca
        start_time = t0
        morpher = NetMorpher(group=CoxH3,
                             sequence=[COXH3_SIGNATURES[_] for _ in ["TRUNC_ICOSIDODECA",
                                                                     "RHOMBICOSIDODECA", "DODECA"]],
                             rotation_quaternion=Quaternion(),
                             location=Vector([5.0, 0, 0.4]), scale=0.25)

        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 1
        locations = [Vector([9.5, 0, 4.6]),
                     Vector([-2.8, 0, 3.63]),
                     Vector([-2.7, 0, -1.7])]
        colors = [
            ["decagon", "hexagon", "square"],
            ["square", "triangle", "pentagon"],
            ["pentagon", "octagon", "hexagon"]
        ]
        for i in range(3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.5, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        # morph from trunc_icosidodeca to trunc_dodeca to icosidodeca
        start_time = t0
        morpher = NetMorpher(group=CoxH3,
                             sequence=[COXH3_SIGNATURES[_] for _ in
                                       ["TRUNC_ICOSIDODECA", "TRUNC_DODECA", "ICOSIDODECA"]],
                             rotation_quaternion=Quaternion([0, -1, 0], 110 / 180 * pi),
                             location=Vector([11.2, 0, 0.2]), scale=0.3)

        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)
        t0 = 0.5 + morpher.unmorph_sequence(begin_time=t0, transition_time=1)

        t0 = start_time + 3
        locations = [Vector([6.5, 0, 1.6]),
                     Vector([2.1, 0, 0]),
                     Vector([1.75, 0, -3.6])]
        colors = [
            ["decagon", "hexagon", "square"],
            ["decagon", "triangle", "triangle"],
            ["pentagon", "triangle", "triangle"]
        ]
        for i in range(1, 3):
            state = morpher.get_state(i, colors=colors[i])
            state.appear(begin_time=t0, transition_time=1)
            state.rescale(rescale=0.4, begin_time=t0, transition_time=1)
            t0 = 1 + state.move_to(target_location=locations[i], begin_time=t0, transition_time=1)

        # t0 = 0.5 + morpher.disappear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def family_b3(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }B_3", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)
        set_origin(title)

        axis = NumberLine2(length=12, domain=[5, 48], tic_labels={"6": 6, "8": 8, "12": 12, "24": 24, "48": 48},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-1, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        family = [["x3x4x"], ["x3x4o", "x3o4x", "o3x4x"], ["x3o4o", "o3o4x", "o3x4o"]]
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([5, 48], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        group = CoxB3()
        (n_a, n_b, n_c) = [group.normals[i].real() for i in range(3)]

        size = 4
        plane_a = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_A", normal=n_a, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_c = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_C", shadow=False, normal=n_c,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_b = Plane(u=[-size, size], v=[-size, size], normal=n_b, color='mirror', name="Mirror_B", shadow=False,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)

        center = Vector([8.2, 0, -3.25])
        floor = Floor(u=[-size, size], v=[-size, size], normal=Vector([0, 0, 1]), checker_scale=10, coords="Generated")

        # define objects
        scale = 0.8
        trunc_cubocta = PolyhedronWithModifier.from_group_signature(CoxB3, COXB3_SIGNATURES["TRUNC_CUBOCTA"],
                                                                    name="TruncatedCuboctahedron", color="color_dict",
                                                                    scale=0.8)

        floor.move(direction=Vector([0, 0, -4]), begin_time=0, transition_time=0)
        t0 = 0.5 + floor.grow(begin_time=t0, transition_time=1)

        mirrors = [plane_a, plane_b, plane_c]
        for mirror in mirrors:
            t0 = 0.5 + mirror.grow(begin_time=t0, transition_time=1)

        t0 = 0.5 + trunc_cubocta.grow(begin_time=t0, transition_time=1)

        copy_trunc_cubocta = trunc_cubocta.copy()
        copy_trunc_cubocta.rescale(rescale=scale, begin_time=t0, transition_time=0)
        copy_trunc_cubocta.appear(begin_time=t0, transition_time=0)
        copy_trunc_cubocta.rescale(rescale=0.6, begin_time=t0, transition_time=1)
        t0 = 0.5 + copy_trunc_cubocta.move_to(target_location=[0.25, 7, 0], begin_time=t0, transition_time=1)

        # # perform morphing of the solids
        derived_polyhedra = ["TRUNC_OCTA", "OCTA", "RHOMBICUBOCTA", "CUBE", "TRUNC_CUBE", "CUBOCTA"]
        signatures = [COXB3_SIGNATURES[poly] for poly in derived_polyhedra]

        src_vertices = group.get_real_point_cloud(signature=COXB3_SIGNATURES["TRUNC_CUBOCTA"])
        src_radius = src_vertices[0].length

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_scale_factors = [src_radius / vertices[0].length for vertices in all_vertices]
        all_target_vertices = [[v * scale_factor for v in vertices] for scale_factor, vertices in
                               zip(all_scale_factors, all_vertices)]

        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx, src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx, target_v in enumerate(target_vertices):
                    if (src_v - target_v).length < dist:
                        dist = (src_v - target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        polyhedra = []
        for j, sig in enumerate(signatures):
            if j < len(
                    signatures) - 1:  # skip the last signature, we don't need to create the copy for the last polyhedron

                _tmp = PolyhedronWithModifier.from_group_signature(CoxB3, COXB3_SIGNATURES["TRUNC_CUBOCTA"],
                                                                   scale=0.8,
                                                                   color="color_dict")
                _tmp.index_transform_mesh(lambda i, j=j: all_target_vertices[j][all_maps[j][i]], begin_time=0,
                                          transition_time=0)
                _tmp.transform_colors(shape_key=1, face_classes=group.get_faces_in_conjugacy_classes(
                    COXB3_SIGNATURES["TRUNC_CUBOCTA"]), begin_time=0, transition_time=0)
                polyhedra.append(_tmp)

        locations = {
            "TRUNC_CUBOCTA": [-2.2, 4, 0],
            "TRUNC_OCTA": [-16.2, 7.2, 0],
            "OCTA": [-16.2, 0.2, 0],
            "RHOMBICUBOCTA": [-11.1, 7.2, 0],
            "CUBE": [-11.2, 1.7, 0],
            "TRUNC_CUBE": [-5, 6.89, 0],
            "CUBOCTA": [-6.1, -1.6, 0],
        }

        transformations = []
        for j in range(6):
            transformations.append(lambda i, j=j: all_target_vertices[j][all_maps[j][i]])
            # reset after two tranformations
            if j == 1 or j == 3:
                transformations.append(lambda i: src_vertices[i])

        trunc_cubocta.index_transform_mesh(transformations=transformations, begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_colors(shape_key=1,
                                                  face_classes=group.get_faces_in_conjugacy_classes(
                                                      COXB3_SIGNATURES["TRUNC_CUBOCTA"]),
                                                  begin_time=t0, transition_time=1)

        count = 0
        for i in range(7):
            if not (i == 2 or i == 5):
                polyhedra[count].appear(begin_time=t0, transition_time=0)
                polyhedra[count].rescale(rescale=0.5, begin_time=t0, transition_time=1)
                t0 = 0.5 + polyhedra[count].move_to(target_location=locations[derived_polyhedra[count]], begin_time=t0,
                                                    transition_time=1)
                count += 1

            trunc_cubocta.transform_colors(shape_key=i + 2, face_classes=group.get_faces_in_conjugacy_classes(
                COXB3_SIGNATURES["TRUNC_CUBOCTA"]), begin_time=t0, transition_time=1)
            t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)
        # if i<6:
        #     polyhedra[count].appear(begin_time=t0, transition_time=0)
        #     polyhedra[count].rescale(rescale=0.5, begin_time=t0, transition_time=1)

        [mirror.disappear(begin_time=t0, transition_time=1) for mirror in mirrors]
        floor.disappear(begin_time=t0, transition_time=1)

        trunc_cubocta.rescale(rescale=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.move_to(target_location=locations[derived_polyhedra[count]], begin_time=t0,
                                         transition_time=1)

        [poly.shrink(begin_time=t0, transition_time=0.5) for poly in polyhedra]
        trunc_cubocta.shrink(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + copy_trunc_cubocta.shrink(begin_time=t0, transition_time=0.5)
        self.t0 = t0

        container_rotation = Quaternion([1, 0, 0], pi / 2)
        container = BObject(children=[plane_a, plane_b, plane_c, floor, trunc_cubocta, copy_trunc_cubocta] + polyhedra,
                            location=center,
                            rotation_quaternion=container_rotation, name="RotationContainer")
        container.appear(begin_time=0, transition_time=0)

        self.t0 = t0

    def family_h3(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{Coxeter Group }H_3", color="example", text_size="Large", aligned="center",
                     location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)
        set_origin(title)

        axis = NumberLine2(length=12, domain=[10, 120], tic_labels={"12": 12, "20": 20, "30": 30, "60": 60, "120": 120},
                           tic_label_digits=0, direction="VERTICAL",
                           axis_label=r"\text{Number of Vertices}",
                           location=[-11.5, 0, -7], name="VertexNumberAxis", axis_label_location=[-0.5, 0, 13.5])
        t0 = 0.5 + axis.grow(begin_time=t0, transition_time=2)
        axis.to_log(begin_time=0, transition_time=0)

        family = [["x3x5x"], ["o3x5x", "x3o5x", "x3x5o"], ["x3o5o", "o3o5x", "o3x5o"]]
        for level in family:
            level_count = len(level)
            for i, cd_string in enumerate(level):
                if level_count == 1:
                    x = 0
                else:
                    dx = 10 / (level_count - 1)
                    x = -8 + i * dx
                dia = CoxeterDynkinDiagram(cd_string)
                z = get_z_location([10, 120], [-6, 6], dia.get_vertex_count(), offset=-1)
                dia2 = DynkinDiagram.from_string(cd_string, location=[x, 0, z], scale=0.5)
                t0 = 0.5 + dia2.appear(begin_time=t0, transition_time=0.5)

        group = CoxH3()
        (n_a, n_b, n_c) = [group.normals[i].real() for i in range(3)]

        size = 4
        plane_a = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_A", normal=n_a, roughness=0.05,
                        shadow=False, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_c = Plane(u=[-size, size], v=[-size, size], color='mirror', name="Mirror_C", shadow=False, normal=n_c,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)
        plane_b = Plane(u=[-size, size], v=[-size, size], normal=n_b, color='mirror', name="Mirror_B", shadow=False,
                        roughness=0.05, solid=0.1, solidify_mode="SIMPLE",
                        smooth=4, subdivision_type="SIMPLE", use_rim=True, use_rim_only=True, offset=0)

        center = Vector([8.1, 0, -3.09])
        floor = Floor(u=[-size, size], v=[-size, size], normal=Vector([0, 0, 1]), checker_scale=10, coords="Generated")

        # define objects
        scale = 0.5
        trunc_icosidodeca = PolyhedronWithModifier.from_group_signature(CoxH3, COXH3_SIGNATURES["TRUNC_ICOSIDODECA"],
                                                                        name="TruncatedIcosidodecahedron",
                                                                        color="color_dict",
                                                                        scale=scale)

        floor.move(direction=Vector([0, 0, -4]), begin_time=0, transition_time=0)
        t0 = 0.5 + floor.grow(begin_time=t0, transition_time=1)

        mirrors = [plane_a, plane_b, plane_c]
        for mirror in mirrors:
            t0 = 0.5 + mirror.grow(begin_time=t0, transition_time=1)

        t0 = 0.5 + trunc_icosidodeca.grow(begin_time=t0, transition_time=1)

        copy_trunc_cubocta = trunc_icosidodeca.copy()
        copy_trunc_cubocta.appear(begin_time=t0, transition_time=0)
        copy_trunc_cubocta.rescale(rescale=scale, begin_time=0, transition_time=0)
        copy_trunc_cubocta.rescale(rescale=0.6, begin_time=t0, transition_time=1)
        t0 = 0.5 + copy_trunc_cubocta.move_to(target_location=[0.25, 7, 0], begin_time=t0, transition_time=1)

        # # perform morphing of the solids
        derived_polyhedra = ["TRUNC_ICOSA", "ICOSA", "RHOMBICOSIDODECA", "DODECA", "TRUNC_DODECA", "ICOSIDODECA"]
        signatures = [COXH3_SIGNATURES[poly] for poly in derived_polyhedra]

        src_vertices = group.get_real_point_cloud(signature=COXH3_SIGNATURES["TRUNC_ICOSIDODECA"])
        src_radius = src_vertices[0].length

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_scale_factors = [src_radius / vertices[0].length for vertices in all_vertices]
        all_target_vertices = [[v * scale_factor for v in vertices] for scale_factor, vertices in
                               zip(all_scale_factors, all_vertices)]

        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx, src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx, target_v in enumerate(target_vertices):
                    if (src_v - target_v).length < dist:
                        dist = (src_v - target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        polyhedra = []
        for j, sig in enumerate(signatures):
            if j < len(
                    signatures) - 1:  # skip the last signature, we don't need to create the copy for the last polyhedron

                _tmp = PolyhedronWithModifier.from_group_signature(CoxH3, COXH3_SIGNATURES["TRUNC_ICOSIDODECA"],
                                                                   scale=scale,
                                                                   color="color_dict")
                _tmp.index_transform_mesh(lambda i, j=j: all_target_vertices[j][all_maps[j][i]], begin_time=0,
                                          transition_time=0)
                _tmp.transform_colors(shape_key=1, face_classes=group.get_faces_in_conjugacy_classes(
                    COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]), begin_time=0, transition_time=0)
                polyhedra.append(_tmp)

        locations = {
            "TRUNC_ICOSIDODECA": [-2.2, 4, 0],
            "TRUNC_ICOSA": [-16.2, 7.2, 0],
            "ICOSA": [-16.2, -0.59, 0],
            "RHOMBICOSIDODECA": [-11.1, 7.06, 0],
            "DODECA": [-11.2, 1.7, 0],
            "TRUNC_DODECA": [-5, 6.89, 0],
            "ICOSIDODECA": [-6.56, -0.95, 0],
        }

        transformations = []
        for j in range(6):
            transformations.append(lambda i, j=j: all_target_vertices[j][all_maps[j][i]])
            # reset after two tranformations
            if j == 1 or j == 3:
                transformations.append(lambda i: src_vertices[i])

        trunc_icosidodeca.index_transform_mesh(transformations=transformations, begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_icosidodeca.transform_colors(shape_key=1,
                                                      face_classes=group.get_faces_in_conjugacy_classes(
                                                          COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]),
                                                      begin_time=t0, transition_time=1)

        count = 0
        for i in range(7):
            if not (i == 2 or i == 5):
                polyhedra[count].appear(begin_time=t0, transition_time=0)
                polyhedra[count].rescale(rescale=0.5, begin_time=t0, transition_time=1)
                t0 = 0.5 + polyhedra[count].move_to(target_location=locations[derived_polyhedra[count]], begin_time=t0,
                                                    transition_time=1)
                count += 1

            trunc_icosidodeca.transform_colors(shape_key=i + 2, face_classes=group.get_faces_in_conjugacy_classes(
                COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]), begin_time=t0, transition_time=1)
            t0 = 0.5 + trunc_icosidodeca.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)
        # if i<6:
        #     polyhedra[count].appear(begin_time=t0, transition_time=0)
        #     polyhedra[count].rescale(rescale=0.5, begin_time=t0, transition_time=1)

        [mirror.disappear(begin_time=t0, transition_time=1) for mirror in mirrors]
        floor.disappear(begin_time=t0, transition_time=1)

        trunc_icosidodeca.rescale(rescale=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_icosidodeca.move_to(target_location=locations[derived_polyhedra[count]], begin_time=t0,
                                             transition_time=1)

        [poly.shrink(begin_time=t0, transition_time=0.5) for poly in polyhedra]
        trunc_icosidodeca.shrink(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + copy_trunc_cubocta.shrink(begin_time=t0, transition_time=0.5)
        self.t0 = t0

        container_rotation = Quaternion([1, 0, 0], pi / 2)
        container = BObject(
            children=[plane_a, plane_b, plane_c, floor, trunc_icosidodeca, copy_trunc_cubocta] + polyhedra,
            location=center,
            rotation_quaternion=container_rotation, name="RotationContainer")
        container.appear(begin_time=0, transition_time=0)

        self.t0 = t0

    def logo(self):
        t0 = 0
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        # logo
        vertices, faces = get_solid_data("TRUNC_ICOSIDODECA")
        vertices2, faces2 = get_solid_data("TRUNC_ICOSA")
        vertices3, faces3 = get_solid_data("RHOMBICOSIDODECA")

        vertices = normalize_to_unity(vertices)
        vertices2 = normalize_to_unity(vertices2)
        vertices3 = normalize_to_unity(vertices3)

        mod1 = CustomUnfoldModifier(face_materials=["red", "red", "red"],
                                    edge_material="example", vertex_material="red", edge_radius=0.01,
                                    face_types=[4, 6, 10],
                                    vertex_radius=0.02, sorting=False, max_faces=120, emission=.5)

        mod2 = CustomUnfoldModifier(face_materials=["blue", "blue"],
                                    edge_material="example", vertex_material="red", edge_radius=0.0251,
                                    face_types=[5, 6],
                                    vertex_radius=0.051, sorting=False, max_faces=32, emission=.5)

        mod3 = CustomUnfoldModifier(face_materials=["green", "green", "green"],
                                    edge_material="example", vertex_material="red", edge_radius=0.0251,
                                    face_types=[3, 4, 5],
                                    vertex_radius=0.051, sorting=False, max_faces=62, emission=.5)

        kwargs_red = {"mesh": ibpy.create_mesh(vertices, faces=faces), "color": "red",
                      "geo_node_modifier": mod1}
        kwargs_green = {"mesh": ibpy.create_mesh(vertices3, faces=faces3), "color": "green",
                        "geo_node_modifier": mod3}
        kwargs_blue = {"mesh": ibpy.create_mesh(vertices2, faces=faces2), "color": "blue",
                       "geo_node_modifier": mod2}

        logo = LogoFromInstances(instance=BObject, rotation_euler=[pi / 2, 0, 0],
                                 scale=[10] * 3, location=[-9, 20, -9],
                                 details=15, kwargs_blue=kwargs_blue, kwargs_green=kwargs_green, kwargs_red=kwargs_red)

        t0 = 0.5 + logo.grow(begin_time=t0, transition_time=2)
        for instance in logo.get_instances():
            instance.rotate(rotation_euler=[0, tau, 0], begin_time=0, transition_time=30)

        self.t0 = t0

    def documentation(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        ibpy.empty_blender_view3d()

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=0.6, type='BLOOM', size=1)

        # vertex map between two polyhedra
        poly = PolyhedronWithModifier.from_group_signature(CoxB3, COXB3_SIGNATURES["TRUNC_OCTA"], radius=5)
        poly.grow(begin_time=t0, transition_time=0)
        poly2 = PolyhedronWithModifier.from_group_signature(CoxB3, COXB3_SIGNATURES["OCTA"], radius=5)
        poly2.grow(begin_time=t0, transition_time=0)

        # nets with different root face
        # trunc_octa_unfolder = Unfolder(CoxB3, COXB3_SIGNATURES["TRUNC_OCTA"])
        # net = trunc_octa_unfolder.create_net()
        # net.appear(begin_time=t0, transition_time=1)
        #
        # trunc_octa_unfolder2 = Unfolder(CoxB3, COXB3_SIGNATURES["TRUNC_OCTA"],mode="SMALLEST_FACE")
        # net2 = trunc_octa_unfolder2.create_net()
        # net2.appear(begin_time=t0, transition_time=1)

        # Face tree illustration
        # trunc_octa_unfolder = Unfolder(CoxB3, COXB3_SIGNATURES["TRUNC_OCTA"])
        # trunc_octa = trunc_octa_unfolder.create_bob()
        # trunc_octa.grow(begin_time=t0, transition_time=1)
        #
        # trunc_octa_net = trunc_octa_unfolder.create_net()
        # trunc_octa_net.appear(begin_time=t0, transition_time=1)
        #
        # trunc_octa_unfolder.show_node_tree()
        # trunc_octa_unfolder.show_index_structure()
        #
        # print(trunc_octa_unfolder.unfolded2vertex_map)
        # print(trunc_octa_unfolder.vertex2unfolded_map)
        #trunc_octa_unfolder.display_net_data()

        self.t0 = t0

    def documentation2(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        ibpy.empty_blender_view3d()

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        octahedron_unfolder = Unfolder(CoxB3, COXB3_SIGNATURES["OCTA"])
        trunc_octa_unfolder = Unfolder(CoxB3, COXB3_SIGNATURES["TRUNC_OCTA"])

        src2target = trunc_octa_unfolder.create_map(octahedron_unfolder)
        target2src = octahedron_unfolder.create_map(trunc_octa_unfolder)

        print(src2target)
        print(target2src)

        self.t0 = t0

    def documentation3(self):
        t0 = 0
        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()
        ibpy.empty_blender_view3d()

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        trunc_octa = PolyhedronWithModifier.from_group_signature(CoxB3, COXB3_SIGNATURES["TRUNC_OCTA"],
                                                                 location=[-7, 0, -2.5], radius=2.5)
        t0 = 0.5 + trunc_octa.grow(begin_time=t0, transition_time=1)

        octa = PolyhedronWithModifier.from_group_signature(CoxB3, COXB3_SIGNATURES["OCTA"],
                                                           location=[7, 0, -2.5], radius=2.5)
        t0 = 0.5 + octa.grow(begin_time=t0, transition_time=1)

        morpher = NetMorpher(group=CoxA3,
                             sequence=[COXA3_SIGNATURES[_] for _ in ["TRUNC_OCTA", "OCTA"]],
                             scale=0.3, location=Vector([-2.9, 0, 2.4]))

        t0 = 0.5 + morpher.morph_sequence(begin_time=t0, transition_time=1, pause=1)

        morpher2 = NetMorpher(group=CoxA3,
                              sequence=[COXA3_SIGNATURES[_] for _ in ["TRUNC_OCTA", "OCTA"]],
                              scale=0.3, location=Vector([-2.9, 0, 2.4]))

        t0 = 0.5 + morpher2.morph_sequence(begin_time=t0, transition_time=1, pause=1000)

        self.t0 = t0

    def lifting_cubes(self):
        """Animation showing cubes and spheres from 1D to 4D"""
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -30, 0])
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_lens(lens=40)

        # === CUBES SECTION ===
        cube_title = Text(r"\text{Cubes in Various Dimensions}", color="example", text_size="Large", aligned="center",
                          location=[0, 0, 6])
        t0 = 0.5 + cube_title.write(begin_time=t0, transition_time=0.5)
        t0 += 1

        # 1D Cube - Line segment
        label_1d = Text(r"\text{1D: Line Segment}", color="text", text_size="normal", aligned="center",
                        location=[-10, 0, 4.5])
        t0 = 0.5 + label_1d.write(begin_time=t0, transition_time=0.5)

        # Create vertices at 0 and 1
        v0_location = Vector([-10, 0, 0])
        v1_location = Vector([-10, 0, 2])
        v0 = Sphere(r=0.15, mesh_type="ico", color="red", resolution=1, location=v0_location)
        v1 = Sphere(r=0.15, mesh_type="ico", color="red", resolution=1, location=v1_location)
        edge_1d = Cylinder.from_start_to_end(start=v0_location, end=v1_location, color="example", thickness=0.2)

        t0 = 0.5 + v0.grow(begin_time=t0, transition_time=0.3)
        t0 = 0.5 + v1.grow(begin_time=t0, transition_time=0.3)
        t0 = 0.5 + edge_1d.grow(begin_time=t0, transition_time=0.5)

        coord_0 = Text(r"(0)", color="joker", emission=1, aligned="center",
                       location=[-10.5, 0, 0])
        coord_1 = Text(r"(1)", color="joker", emission=1, aligned="center",
                       location=[-10.5, 0, 2])
        t0 = 0.5 + coord_0.write(begin_time=t0, transition_time=0.3)
        t0 = 0.5 + coord_1.write(begin_time=t0, transition_time=0.3)
        t0 += 1

        # 2D Cube - Square
        label_2d = Text(r"\text{2D: Square}", color="text", text_size="normal", aligned="center",
                        location=[-3, 0, 4.5])
        t0 = 0.5 + label_2d.write(begin_time=t0, transition_time=0.5)

        # Create 4 vertices
        square_verts = [
            Vector([-3, 0, 0]),  # (0,0)
            Vector([-1, 0, 0]),  # (1,0)
            Vector([-1, 0, 2]),  # (1,1)
            Vector([-3, 0, 2])  # (0,1)
        ]
        square_spheres = []
        for v in square_verts:
            s = Sphere(r=0.12, mesh_type="ico", color="red", resolution=1, location=v)
            t0 = 0.2 + s.grow(begin_time=t0, transition_time=0.2)
            square_spheres.append(s)

        # Create edges
        square_edges = [
            Cylinder.from_start_to_end(square_verts[0], square_verts[1], color="example", thickness=0.2),
            Cylinder.from_start_to_end(square_verts[1], square_verts[2], color="example", thickness=0.2),
            Cylinder.from_start_to_end(square_verts[2], square_verts[3], color="example", thickness=0.2),
            Cylinder.from_start_to_end(square_verts[3], square_verts[0], color="example", thickness=0.2)
        ]
        for edge in square_edges:
            t0 = 0.2 + edge.grow(begin_time=t0, transition_time=0.3)

        coord_labels_2d = [
            Text(r"(0,0)", color="joker", emission=1, aligned="center", location=[-3.5, 0, -0.3]),
            Text(r"(1,0)", color="joker", emission=1, aligned="center", location=[-0.5, 0, -0.3]),
            Text(r"(1,1)", color="joker", emission=1, aligned="center", location=[-0.5, 0, 2.3]),
            Text(r"(0,1)", color="joker", emission=1, aligned="center", location=[-3.5, 0, 2.3])
        ]
        for coord in coord_labels_2d:
            t0 = 0.2 + coord.write(begin_time=t0, transition_time=0.2)
        t0 += 1

        # 3D Cube
        label_3d = Text(r"\text{3D: Cube}", color="text", text_size="normal", aligned="center",
                        location=[4, 0, 4.5])
        t0 = 0.5 + label_3d.write(begin_time=t0, transition_time=0.5)

        # Build 3D cube with explicit vertices and edges (same color scheme as 1D/2D)
        cube_pivot = EmptyCube(location=[4.5, 0, 1], name="CubePivot3D")
        half = 1.0
        cube_vert_locals = [
            Vector([-half, -half, -half]),  # (0,0,0)
            Vector([half, -half, -half]),  # (1,0,0)
            Vector([half, half, -half]),  # (1,1,0)
            Vector([-half, half, -half]),  # (0,1,0)
            Vector([-half, -half, half]),  # (0,0,1)
            Vector([half, -half, half]),  # (1,0,1)
            Vector([half, half, half]),  # (1,1,1)
            Vector([-half, half, half]),  # (0,1,1)
        ]
        coord_strs_3d = [
            r"(0,0,0)", r"(1,0,0)", r"(1,1,0)", r"(0,1,0)",
            r"(0,0,1)", r"(1,0,1)", r"(1,1,1)", r"(0,1,1)"
        ]
        cube_3d_spheres = []
        for v in cube_vert_locals:
            s = Sphere(r=0.12, mesh_type="ico", color="red", resolution=1, location=v)
            ibpy.set_parent(s, cube_pivot)
            cube_3d_spheres.append(s)
        cube_edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]
        cube_3d_edges = []
        for i, j in cube_edge_pairs:
            cyl = Cylinder.from_start_to_end(
                start=cube_vert_locals[i], end=cube_vert_locals[j],
                color="example", thickness=0.2
            )
            ibpy.set_parent(cyl, cube_pivot)
            cube_3d_edges.append(cyl)
        # Labels parented to their sphere so they follow the vertex during rotation
        cube_3d_labels = []
        for k, (v, lbl) in enumerate(zip(cube_vert_locals, coord_strs_3d)):
            offset = Vector([v.x * 0.4, 0, v.z * 0.4])
            t_label = Text(lbl, color="joker", emission=1, text_size="small", aligned="center",
                           location=offset, name="CubeLabel3D" + str(k))
            ibpy.set_parent(t_label, cube_3d_spheres[k])
            cube_3d_labels.append(t_label)

        for s in cube_3d_spheres:
            s.grow(begin_time=t0, transition_time=0.5)
        t0 += 0.5 + 0.3
        for e in cube_3d_edges:
            e.grow(begin_time=t0, transition_time=0.5)
        t0 += 0.5 + 0.3
        for tl in cube_3d_labels:
            tl.appear(begin_time=t0, transition_time=0)
            tl.write(begin_time=t0, transition_time=0.3)
        t0 += 0.3 + 0.5

        # Rotate cube; labels counter-rotate to always face observer
        cube_pivot.rotate(rotation_euler=[0, 0, 12 / 11 * tau], begin_time=t0, transition_time=12)
        for tl in cube_3d_labels:
            tl.rotate(rotation_euler=[0, 0, -12 / 11 * tau], begin_time=t0, transition_time=12)
        t0 += 3 + 0.5

        # Pattern explanation
        pattern_text = Text(r"\text{Pattern: All combinations of 0s and 1s}", color="important",
                            text_size="Large", aligned="center", location=[0, 0, -3])
        t0 = 0.5 + pattern_text.write(begin_time=t0, transition_time=0.5)

        # label all coordinates
        coord_texts = []
        shift = Vector([5, 0, -4])
        coord_list = tuples([0, 1], 4)
        for i, coords in enumerate(coord_list):
            coord_texts.append(Text(
                "(" + str(coords[0]) + "," + str(coords[1]) + "," + str(coords[2]) + "," + str(coords[3]) + ")",
                color="joker", text_size="normal", aligned="center",
                location=shift + Vector([2 * (i % 4), 0, -(i // 4)])))
            t0 = 0.1 + coord_texts[-1].write(begin_time=t0, transition_time=0.1)

        # 4D Cube - Tesseract
        label_4d = Text(r"\text{4D: Tesseract}", color="text", text_size="normal",
                        aligned="center", location=[10, 0, 4.5])
        t0 = 0.5 + label_4d.write(begin_time=t0, transition_time=0.5)

        # Use 4D projection via CoxB4
        unfolder = Unfolder4D2(CoxB4, "o3o3o4x", south_pole=1, mode="LARGEST_CELL", skip=0)
        net = unfolder.create_net(location=[10, 0, 1], scale=0.3, name="Tesseract", color="transparent")
        mod = PolyhedronViewModifier(edge_color="example", vertex_color="red",
                                     vertex_radius=0.4, edge_radius=0.1)
        [child.add_mesh_modifier(type="NODES", node_modifier=mod) for child in net.b_children]
        main_axis = unfolder.get_main_axis(choice="MINIMUM")
        # rotate main axis into z-direction
        omega = Vector(main_axis).cross(Vector([1, 0, 0]))
        quat_0 = Quaternion(Vector([1, 0, 0]), pi) @ Quaternion(omega, np.arccos(main_axis.dot(omega)))
        net.rotate(rotation_quaternion=quat_0, begin_time=0, transition_time=0)
        quat_rot = Quaternion(Vector([0, 0, 1]), 0.99 * pi)

        t0 = 0.5 + net.appear(children=True, begin_time=t0, transition_time=1.5, sequentially=True)
        t0 = net.rotate(rotation_quaternion=quat_rot @ quat_0, begin_time=t0, transition_time=3)
        net.rotate(rotation_quaternion=quat_rot @ quat_rot @ quat_0, begin_time=t0, transition_time=3)
        t0 += 2

        # Clean up cube section
        t0 += 0.5
        cube_title.disappear(begin_time=t0, transition_time=0.5)
        label_1d.disappear(begin_time=t0, transition_time=0.5)
        label_2d.disappear(begin_time=t0, transition_time=0.5)
        label_3d.disappear(begin_time=t0, transition_time=0.5)
        label_4d.disappear(begin_time=t0, transition_time=0.5)
        pattern_text.disappear(begin_time=t0, transition_time=0.5)
        v0.disappear(begin_time=t0, transition_time=0.5)
        v1.disappear(begin_time=t0, transition_time=0.5)
        edge_1d.disappear(begin_time=t0, transition_time=0.5)
        coord_0.disappear(begin_time=t0, transition_time=0.5)
        coord_1.disappear(begin_time=t0, transition_time=0.5)
        for s in square_spheres:
            s.disappear(begin_time=t0, transition_time=0.5)
        for edge in square_edges:
            edge.disappear(begin_time=t0, transition_time=0.5)
        for coord in coord_labels_2d:
            coord.disappear(begin_time=t0, transition_time=0.5)
        for s in cube_3d_spheres:
            s.disappear(begin_time=t0, transition_time=0.5)
        for e in cube_3d_edges:
            e.disappear(begin_time=t0, transition_time=0.5)
        for tl in cube_3d_labels:
            tl.disappear(begin_time=t0, transition_time=0.5)
        for coord_text in coord_texts:
            coord_text.disappear(begin_time=t0, transition_time=0.5)
        net.disappear(begin_time=t0, transition_time=0.5)
        t0 += 1

        self.t0 = t0

    def dimensions_intro(self):
        """
        Two-column overview: Three dimensions (left) vs Four dimensions (right),
        showing Coxeter-Dynkin diagrams in each column.

        Timing:
          1. "Three dimensions" title appears
          2. "Four dimensions" title appears
          3. 3D diagrams: x3x3x, x3x4x, x3x5x
          4. 4D diagrams: x3x3x3x, x3x3x4x, x3x3x5x, x3x4x3x, x3x3x *b3x
        """
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        # --- Titles (sequential) ---
        title_3d = Text(r"\text{Three dimensions}", color="example", text_size="Large", aligned="center",
                        location=[-5, 0, 6])
        t0 = 0.5 + title_3d.write(begin_time=t0, transition_time=0.5)

        title_4d = Text(r"\text{Four dimensions}", color="example", text_size="Large", aligned="center",
                        location=[5, 0, 6])
        t0 = 0.5 + title_4d.write(begin_time=t0, transition_time=0.5)

        # --- 3D Coxeter-Dynkin diagrams (left column) ---
        diagrams_3d = ["x3x3x", "x3x4x", "x3x5x"]
        z_positions_3d = [3.5, 1.5, -0.5]
        for cd_string, z in zip(diagrams_3d, z_positions_3d):
            dia = DynkinDiagram.from_string(cd_string, location=[-5, 0, z], scale=0.5, no_threes=True)
            t0 = 0.5 + dia.appear(begin_time=t0, transition_time=0.5)

        # --- 4D Coxeter-Dynkin diagrams (right column) ---
        diagrams_4d = ["x3x3x3x", "x3x3x4x", "x3x3x5x", "x3x4x3x", "x3x3x *b3x"]
        z_positions_4d = [3.5, 1.5, -0.5, -2.5, -5]
        for cd_string, z in zip(diagrams_4d, z_positions_4d):
            dia = DynkinDiagram.from_string(cd_string, location=[5, 0, z], scale=0.5, no_threes=True)
            t0 = 0.5 + dia.appear(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def a_series_intro(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{A-Series}", color="example", text_size="Large", aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        labels = ["x", "x3x", "x3x3x", "x3x3x3x"]
        z_positions = [3.0, 0.5, -2.0, -4.5]

        dynkin_diagrams = []
        for label, z in zip(labels, z_positions):
            dia = DynkinDiagram.from_string(label, location=[0, 0, z], scale=0.7, no_threes=True)
            t0 = 0.5 + dia.appear(begin_time=t0, transition_time=0.5)
            dynkin_diagrams.append(dia)

        # All four have appeared — dim the last three; the first stays fully bright
        dim_tr = 0.5
        for dia in dynkin_diagrams[1:]:
            dia.change_alpha(from_value=1.0, to_value=0.05, begin_time=t0, transition_time=dim_tr, children=True)
        t0 += dim_tr

        # Cycle: every 5 seconds move the spotlight to the next diagram
        pause = 5
        flash_in = 0.5  # emission ramps up
        flash_out = 0.75  # emission fades back to zero
        current_alphas = [1.0, 0.05, 0.05, 0.05]

        for active in range(1, 4):
            prev = active - 1
            cycle_t = t0 + pause

            # Dim the previously highlighted diagram
            dynkin_diagrams[prev].change_alpha(from_value=current_alphas[prev], to_value=0.05,
                                               begin_time=cycle_t, transition_time=dim_tr, children=True)
            current_alphas[prev] = 0.05

            # Restore the now-active diagram to full alpha
            dynkin_diagrams[active].change_alpha(from_value=current_alphas[active], to_value=1.0,
                                                 begin_time=cycle_t, transition_time=dim_tr, children=True)
            current_alphas[active] = 1.0

            # Brief emission flash on the node spheres of the active diagram
            for sphere in dynkin_diagrams[active].spheres:
                sphere.change_emission(from_value=0, to_value=1,
                                       begin_time=cycle_t, transition_time=flash_in)
                sphere.change_emission(from_value=1, to_value=0,
                                       begin_time=cycle_t + flash_in, transition_time=flash_out)
                for edge in dynkin_diagrams[active].cylinders:
                    edge.change_emission(from_value=0, to_value=1,
                                         begin_time=cycle_t, transition_time=flash_in)
                    edge.change_emission(from_value=1, to_value=0,
                                         begin_time=cycle_t + flash_in, transition_time=flash_out)
                for ring in dynkin_diagrams[active].rings:
                    ring.change_emission(from_value=0, to_value=1,
                                         begin_time=cycle_t, transition_time=flash_in)
                    ring.change_emission(from_value=1, to_value=0,
                                         begin_time=cycle_t + flash_in, transition_time=flash_out)

            t0 = cycle_t

        self.t0 = t0

    def b_series_intro(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{B-Series}", color="example", text_size="Large", aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        labels = ["x", "x4x", "x3x4x", "x3x3x4x"]
        z_positions = [3.0, 0.5, -2.0, -4.5]

        dynkin_diagrams = []
        for label, z in zip(labels, z_positions):
            dia = DynkinDiagram.from_string(label, location=[0, 0, z], scale=0.7, no_threes=True)
            t0 = 0.5 + dia.appear(begin_time=t0, transition_time=0.5)
            dynkin_diagrams.append(dia)

        # All four have appeared — dim the last three; the first stays fully bright
        dim_tr = 0.5
        for dia in dynkin_diagrams[1:]:
            dia.change_alpha(from_value=1.0, to_value=0.05, begin_time=t0, transition_time=dim_tr, children=True)
        t0 += dim_tr

        # Cycle: every 5 seconds move the spotlight to the next diagram
        pause = 5
        flash_in = 0.5  # emission ramps up
        flash_out = 0.75  # emission fades back to zero
        current_alphas = [1.0, 0.05, 0.05, 0.05]

        for active in range(1, 4):
            prev = active - 1
            cycle_t = t0 + pause

            # Dim the previously highlighted diagram
            dynkin_diagrams[prev].change_alpha(from_value=current_alphas[prev], to_value=0.05,
                                               begin_time=cycle_t, transition_time=dim_tr, children=True)
            current_alphas[prev] = 0.05

            # Restore the now-active diagram to full alpha
            dynkin_diagrams[active].change_alpha(from_value=current_alphas[active], to_value=1.0,
                                                 begin_time=cycle_t, transition_time=dim_tr, children=True)
            current_alphas[active] = 1.0

            # Brief emission flash on the node spheres of the active diagram
            for sphere in dynkin_diagrams[active].spheres:
                sphere.change_emission(from_value=0, to_value=1,
                                       begin_time=cycle_t, transition_time=flash_in)
                sphere.change_emission(from_value=1, to_value=0,
                                       begin_time=cycle_t + flash_in, transition_time=flash_out)
                for edge in dynkin_diagrams[active].cylinders:
                    edge.change_emission(from_value=0, to_value=1,
                                         begin_time=cycle_t, transition_time=flash_in)
                    edge.change_emission(from_value=1, to_value=0,
                                         begin_time=cycle_t + flash_in, transition_time=flash_out)
                for ring in dynkin_diagrams[active].rings:
                    ring.change_emission(from_value=0, to_value=1,
                                         begin_time=cycle_t, transition_time=flash_in)
                    ring.change_emission(from_value=1, to_value=0,
                                         begin_time=cycle_t + flash_in, transition_time=flash_out)

            t0 = cycle_t

        self.t0 = t0

    def d_series_intro(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{D-Series}", color="example", text_size="Large", aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        labels = ["x3x3x *b3x", "x3x3x3x *b3x", "x3x3x3x3x *b3x"]
        z_positions = [3.0, -0.5, -4]

        dynkin_diagrams = []
        for label, z in zip(labels, z_positions):
            dia = DynkinDiagram.from_string(label, location=[0, 0, z], scale=0.7, no_threes=True)
            t0 = 0.5 + dia.appear(begin_time=t0, transition_time=0.5)
            dynkin_diagrams.append(dia)

        dynkin_diagrams[0].rotate(rotation_euler=[0, -pi / 6, 0], begin_time=0, transition_time=0)

        # All four have appeared — dim the last three; the first stays fully bright
        dim_tr = 0.5
        for dia in dynkin_diagrams[1:]:
            dia.change_alpha(from_value=1.0, to_value=0.05, begin_time=t0, transition_time=dim_tr, children=True)
        t0 += dim_tr

        self.t0 = t0

    def f4_intro(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{F}_4", color="example", text_size="Large", aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        labels = ["x3x4x3x"]
        z_positions = [0]

        dynkin_diagrams = []
        for label, z in zip(labels, z_positions):
            dia = DynkinDiagram.from_string(label, location=[0, 0, z], scale=1.7, no_threes=True)
            t0 = 0.5 + dia.appear(begin_time=t0, transition_time=0.5)
            dynkin_diagrams.append(dia)

        self.t0 = t0

    def h_family_intro(self):
        t0 = 0

        ibpy.set_hdri_background("kloofendal_misty_morning_puresky_4k", 'exr', simple=True, transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)
        set_alpha_composition()

        empty_location = Vector([0, 0, 0])
        camera_empty = EmptyCube(location=empty_location)
        ibpy.set_camera_view_to(camera_empty)
        camera_location = Vector([0, -21, 0])
        ibpy.set_camera_location(location=camera_location)

        title = Text(r"\text{H-Family}", color="example", text_size="Large", aligned="center", location=[0, 0, 6])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        labels = ["x", "x5x", "x3x5x", "x3x3x5x"]
        z_positions = [3.0, 0.5, -2.0, -4.5]

        dynkin_diagrams = []
        for label, z in zip(labels, z_positions):
            dia = DynkinDiagram.from_string(label, location=[0, 0, z], scale=0.7, no_threes=True)
            t0 = 0.5 + dia.appear(begin_time=t0, transition_time=0.5)
            dynkin_diagrams.append(dia)

        # All four have appeared — dim the last three; the first stays fully bright
        dim_tr = 0.5
        for dia in dynkin_diagrams[1:]:
            dia.change_alpha(from_value=1.0, to_value=0.05, begin_time=t0, transition_time=dim_tr, children=True)
        t0 += dim_tr

        # Cycle: every 5 seconds move the spotlight to the next diagram
        pause = 5
        flash_in = 0.5  # emission ramps up
        flash_out = 0.75  # emission fades back to zero
        current_alphas = [1.0, 0.05, 0.05, 0.05]

        for active in range(1, 4):
            prev = active - 1
            cycle_t = t0 + pause

            # Dim the previously highlighted diagram
            dynkin_diagrams[prev].change_alpha(from_value=current_alphas[prev], to_value=0.05,
                                               begin_time=cycle_t, transition_time=dim_tr, children=True)
            current_alphas[prev] = 0.05

            # Restore the now-active diagram to full alpha
            dynkin_diagrams[active].change_alpha(from_value=current_alphas[active], to_value=1.0,
                                                 begin_time=cycle_t, transition_time=dim_tr, children=True)
            current_alphas[active] = 1.0

            # Brief emission flash on the node spheres of the active diagram
            for sphere in dynkin_diagrams[active].spheres:
                sphere.change_emission(from_value=0, to_value=1,
                                       begin_time=cycle_t, transition_time=flash_in)
                sphere.change_emission(from_value=1, to_value=0,
                                       begin_time=cycle_t + flash_in, transition_time=flash_out)
                for edge in dynkin_diagrams[active].cylinders:
                    edge.change_emission(from_value=0, to_value=1,
                                         begin_time=cycle_t, transition_time=flash_in)
                    edge.change_emission(from_value=1, to_value=0,
                                         begin_time=cycle_t + flash_in, transition_time=flash_out)
                for ring in dynkin_diagrams[active].rings:
                    ring.change_emission(from_value=0, to_value=1,
                                         begin_time=cycle_t, transition_time=flash_in)
                    ring.change_emission(from_value=1, to_value=0,
                                         begin_time=cycle_t + flash_in, transition_time=flash_out)

            t0 = cycle_t

        self.t0 = t0

    def a4_display_cell(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [20, -30, 10]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        vertices = [(1.0, 1.7320508509874344, 2.4494897425174713), (1.0, -1.7320506572723389, -2.4494895339012146),
                    (2.0, 0.0, -2.4494893550872803), (1.0, -1.7320508509874344, 2.4494899213314056),
                    (-2.0, 0.0, -2.4494893550872803), (-1.0, 2.886751413345337, 0.8164966106414795),
                    (-3.0, 0.5773502290248871, -0.8164963275194168), (-1.0, 1.732050821185112, -2.4494894444942474),
                    (-2.0, 0.0, 2.4494898319244385), (-2.0, 2.3094011545181274, -0.8164966404438019),
                    (-2.0, -2.3094010055065155, 0.8164969086647034), (2.0, 0.0, 2.4494898319244385),
                    (1.0, -2.8867510557174683, -0.8164962530136108), (-1.0, -1.7320508509874344, 2.4494899213314056),
                    (3.0, -0.5773502290248871, 0.8164967894554138), (-1.0, -2.8867510557174683, -0.8164962530136108),
                    (3.0, 0.5773502290248871, -0.8164963275194168), (2.0, 2.3094011545181274, -0.8164966404438019),
                    (-1.0, 1.7320508509874344, 2.4494897425174713), (-3.0, -0.5773502290248871, 0.8164967894554138),
                    (1.0, 1.732050821185112, -2.4494894444942474), (-1.0, -1.7320506572723389, -2.4494895339012146),
                    (1.0, 2.886751413345337, 0.8164966106414795), (2.0, -2.3094010055065155, 0.8164969086647034)
                    ]
        vertices = [Vector(v) for v in vertices]

        coords = CoordinateSystem2(dimension=3, lengths=[12, 12, 12], colors=['text', 'text', 'text'],
                                   domains=[[-3, 3], [-3, 3], [-3, 3]], tic_label_digits=[0, 0, 0],
                                   origin=[0, 0, 0], labels=[r"\phantom{x}", r"\phantom{y}", r"\phantom{z}"],
                                   include_zeros=[False] * 3,
                                   tic_labels=[{"-2": -2, "-1":-1, "1": 1, "2": 2}] * 3
                                   )
        coords.appear(begin_time=t0, transition_time=0)
        v24 = Data3D(data=vertices, coordinate_system=coords, name="Vertices", material="red", emission=0.5,
                     pointsize=4,subdivisions=1,)
        t0 = 0.5 + v24.appear(begin_time=t0, transition_time=10)

        trunc_octa = PolyhedronWithModifier.from_group_signature(CoxA3, COXA3_SIGNATURES["TRUNC_OCTA"],
                                                                 scale=2,rotation_euler=Vector([55,0,60])/180*pi)
        t0 = 0.5 + trunc_octa.appear(begin_time=t0, transition_time=3)


        container = BObject(children=[trunc_octa,coords])
        container.appear(begin_time=0,transition_time=0,children=False)
        container.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=15)

        self.t0 = t0

    def cd_wizardry(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # Main A3 diagram: x3x3x
        dia = DynkinDiagram.from_string("x3x3x", location=[0, 0, 4], no_threes=True)
        t0 = dia.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + dia.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=0.5)

        # Truncated octahedron appears next to the main diagram and rotates throughout

        trunc_octa_main = PolyhedronWithModifier.from_group_signature(CoxA3, COXA3_SIGNATURES["TRUNC_OCTA"],
                                                                      location=[6.5, 0, 3.25], radius=2.5)
        trunc_octa_main.appear(begin_time=t0, transition_time=0)
        rot0 = t0
        trunc_octa_main.rotate(rotation_euler=[0, 0, tau], begin_time=rot0, transition_time=15)

        # # --- copy 1: remove node 2 → leaves x3x (first two nodes) ---
        copy1 = dia.move_copy(direction=[-6, 0, -4], begin_time=t0, transition_time=1, scale=0.99)
        t0 = 1 + copy1.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=0)
        t0 = 0.5 + copy1.disappear_customized(nodes=[2], rings=[2], edges=[1],
                                              begin_time=t0, transition_time=0.5)
        # hexagon: face of x3x (A2)
        hex1 = Polygon(
            vertices=[2 * Vector([np.sin(tau / 6 * i), 0, np.cos(tau / 6 * i)]) for i in range(6)],
            location=[-7, 0, -3.5], solid=0.1, color="hexagon")
        t0 = 0.5 + hex1.appear(begin_time=t0, transition_time=0.5)

        # --- copy 2: remove node 1 → leaves x.x (two isolated nodes) ---
        copy2 = dia.move_copy(direction=[-1, 0, -4], begin_time=t0, transition_time=1, scale=0.99)
        t0 = 1 + copy2.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=0)

        t0 = copy2.disappear_customized(nodes=[1], edges=[0, 1], rings=[1],
                                        begin_time=t0, transition_time=0.5)
        copy2.rings[0].move(direction=[2, 0, 0], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + copy2.spheres[0].move(direction=[2, 0, 0], begin_time=t0, transition_time=0.5)
        # square: face of x.x (A1 × A1)
        sq = Polygon(
            vertices=[2 * Vector([np.sin(tau / 4 * i), 0, np.cos(tau / 4 * i)]) for i in range(4)],
            location=[0, 0, -3.5], solid=0.1, color="text")
        t0 = 0.5 + sq.appear(begin_time=t0, transition_time=0.5)

        # --- copy 3: remove node 0 → leaves x3x (last two nodes) ---
        copy3 = dia.move_copy(direction=[6, 0, -4], begin_time=t0, transition_time=1)
        t0 = 1 + copy3.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=0)

        t0 = 0.5 + copy3.disappear_customized(nodes=[0], edges=[0], rings=[0],
                                              begin_time=t0, transition_time=0.5)
        # hexagon: face of x3x (A2)
        hex2 = Polygon(
            vertices=[2 * Vector([np.sin(tau / 6 * i), 0, np.cos(tau / 6 * i)]) for i in range(6)],
            location=[7, 0, -3.5], solid=0.1, color="hexagon")
        t0 = 0.5 + hex2.appear(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def cd_wizardry2(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # Main A3 diagram: x3x3x
        dia = DynkinDiagram.from_string("x3x3o", location=[0, 0, 4], no_threes=True)
        t0 = dia.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + dia.appear_customized(rings=[0, 1], begin_time=t0, transition_time=0.5)

        # Truncated octahedron appears next to the main diagram and rotates throughout

        trunc_octa_main = PolyhedronWithModifier.from_group_signature(CoxA3, COXA3_SIGNATURES["TRUNC_TETRA"],
                                                                      location=[6.5, 0, 3.25], radius=2.5)
        trunc_octa_main.appear(begin_time=t0, transition_time=0)
        rot0 = t0
        trunc_octa_main.rotate(rotation_euler=[0, 0, tau], begin_time=rot0, transition_time=15)

        # # --- copy 1: remove node 2 → leaves x3x (first two nodes) ---
        copy1 = dia.move_copy(direction=[-6, 0, -4], begin_time=t0, transition_time=1, scale=0.99)

        t0 = 1 + copy1.appear_customized(rings=[0, 1], begin_time=t0, transition_time=0)
        t0 = 0.5 + copy1.disappear_customized(nodes=[2], edges=[1],
                                              begin_time=t0, transition_time=0.5)
        # hexagon: face of x3x (A2)
        hex1 = Polygon(
            vertices=[2 * Vector([np.sin(tau / 6 * i), 0, np.cos(tau / 6 * i)]) for i in range(6)],
            location=[-7, 0, -3.5], solid=0.1, color="hexagon")
        t0 = 0.5 + hex1.appear(begin_time=t0, transition_time=0.5)

        # --- copy 2: remove node 1 → leaves x.x (two isolated nodes) ---
        copy2 = dia.move_copy(direction=[-1, 0, -4], begin_time=t0, transition_time=1, scale=0.99)
        t0 = 1 + copy2.appear_customized(rings=[0, 1], begin_time=t0, transition_time=0)
        copy2.rings[0].move(direction=[2, 0, 0], begin_time=t0, transition_time=0.5)
        copy2.spheres[0].move(direction=[2, 0, 0], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + copy2.disappear_customized(nodes=[1], edges=[0, 1], rings=[1],
                                              begin_time=t0, transition_time=0.5)

        # --- copy 3: remove node 0 → leaves x3o (last two nodes) ---
        copy3 = dia.move_copy(direction=[6, 0, -4], begin_time=t0, transition_time=1)
        t0 = 1 + copy3.appear_customized(rings=[0, 1], begin_time=t0, transition_time=0)

        t0 = 0.5 + copy3.disappear_customized(nodes=[0], edges=[0], rings=[0],
                                              begin_time=t0, transition_time=0.5)
        # hexagon: face of x3x (A2)
        tri = Polygon(
            vertices=[2 * Vector([np.sin(tau / 3 * i), 0, np.cos(tau / 3 * i)]) for i in range(3)],
            location=[7, 0, -3.5], solid=0.1, color="triangle")
        t0 = 0.5 + tri.appear(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def a4_normal_constraints(self):
        """
        Show how the A4 Coxeter-Dynkin diagram encodes constraint equations
        for the four mirror normal vectors: orthogonality for disconnected
        pairs and 60-degree angles for connected pairs.
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # --- A4 Coxeter-Dynkin Diagram: x--x--x--x ---
        dia = DynkinDiagram.from_string("x3x3x3x",
                                        scale=2, no_threes=True)
        t0 = 0.5 + dia.appear(begin_time=t0, transition_time=1)

        dia.rescale(rescale=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + dia.move_to(target_location=[-2.94, 0, 3.5], begin_time=t0, transition_time=1)

        unwritten = []

        # --- Title ---
        title = Text(r"\text{Computing Normals and Reflections for }A_4",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        # --- Node labels: n_x, n_y, n_z, n_w below each node ---
        # With location=[0,0,4.5] and scale=1, nodes sit at x = -3, -1, 1, 3
        label_names = [r"n_x", r"n_y", r"n_z", r"n_w"]
        x_pos = [-3, -1, 1, 3]
        for name, x in zip(label_names, x_pos):
            lbl = Text(name, color="text", text_size="large",
                       aligned="center", location=[x, 0, 2.5])
            t0 = 0.25 + lbl.write(begin_time=t0, transition_time=0.3)

        t0 += 1  # pause while speech explains "describes the orientation of mirrors"

        relations = [
            r"\angle(n_x,n_w)=90^\circ",
            r"\angle(n_y,n_x)=60^\circ",
            r"\angle(n_y,n_z)=60^\circ",
            r"\angle(n_y,n_w)=90^\circ",
            r"\angle(n_z,n_x)=90^\circ",
            r"\angle(n_z,n_w)=60^\circ",
        ]

        appear_times = [t0, t0 + 4, t0 + 5, t0 + 6, t0 + 7, t0 + 8]
        for i, rel in enumerate(relations):
            if i < 3:
                location = [-9.2, 0, 4.5 - i]
            else:
                location = [5.5, 0, 7.5 - i]
            text = Text(rel, location=location, text_size="large")
            text.write(begin_time=appear_times[i], transition_time=0.5)
            unwritten.append(text)

        t0 += 1
        # --- Explicit normal vectors ---
        normals = [
            r"n_x = \left(\begin{array}{c} 1\\0\\ 0\\ 0\end{array}\right)",
            r"n_w = \left(\begin{array}{c} 0\\0\\ 0\\ 1\end{array}\right)",
            # r"n_y = {1\over 4}  \left(\begin{array}{c}2\\\sqrt{5}-1\\ \sqrt{5}+1\\ 0\end{array}\right)",
            # r"n_z = {1\over 4}  \left(\begin{array}{c}0\\\sqrt{5}+1\\ \sqrt{5}-1\\ 2\end{array}\right)",
        ]

        x_positions = [-7.5, -2.5, 2.5, 7.5]
        for nrm_str, x in zip(normals, x_positions):
            nrm = Text(nrm_str, color="text", text_size="large",
                       aligned="center", location=[x, 0, 0.5])
            t0 = 0.75 + nrm.write(begin_time=t0, transition_time=0.5)

        t0 += 9
        # --- Left column: disconnected node pairs -> orthogonal (90 degrees) ---
        header_orth = Text(r"\text{Disconnected}\rightarrow 90°\text{:}",
                           color="drawing", text_size="large", aligned="left",
                           location=[-9, 0, -1])
        unwritten.append(header_orth)
        t0 = 0.5 + header_orth.write(begin_time=t0, transition_time=0.5)

        orth_eqs = [
            r"n_x \cdot n_z = 0",
            r"n_x \cdot n_w = 0",
            r"n_y \cdot n_w = 0",
        ]
        for i, eq_str in enumerate(orth_eqs):
            eq = Text(eq_str, color="text", text_size="large",
                      aligned="left", location=[-8.5, 0, -2.3 - i * 1.2])
            unwritten.append(eq)
            t0 = 0.5 + eq.write(begin_time=t0, transition_time=0.5)

        # --- Right column: connected node pairs -> 60 degrees ---
        header_angle = Text(r"\text{Connected}\rightarrow 60°\text{:}",
                            color="drawing", text_size="large", aligned="left",
                            location=[2, 0, -1])
        unwritten.append(header_angle)
        t0 = 0.5 + header_angle.write(begin_time=t0, transition_time=0.5)

        angle_eqs = [
            r"n_x \cdot n_y = \cos(\pi/3)",
            r"n_y \cdot n_z = \cos(\pi/3)",
            r"n_z \cdot n_w = \cos(\pi/3)",
        ]
        for i, eq_str in enumerate(angle_eqs):
            eq = Text(eq_str, color="text", text_size="large",
                      aligned="left", location=[2.5, 0, -2.3 - i * 1.2])
            unwritten.append(eq)
            t0 = 0.5 + eq.write(begin_time=t0, transition_time=0.5)

        # --- Normalization ---
        colors = flatten([["drawing"] * 14, ["text"]])
        norm_eq = Text(r"\text{Normalization:}\,\,\,\,\,n_i \cdot n_i = 1 \quad \forall\, i",
                       color=colors, text_size="large", aligned="center",
                       location=[0, 0, -5.5])
        unwritten.append(norm_eq)
        t0 = 0.5 + norm_eq.write(begin_time=t0, transition_time=0.5)

        t0 += 1  #

        # unwrite

        for unw in unwritten:
            unw.unwrite(begin_time=t0, transition_time=0.3)

        t0 += 1
        self.t0 = t0

    def a4_normal_explicit(self):
        """
        Show that the constraint system is under-determined, present explicit
        normal vector values, the reflection matrix formula, and a glimpse of
        the A4 polychora that can be generated.
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # --- Title ---
        title = Text(r"\text{Computing Normals and Reflections for }A_4",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5])
        t0 = title.write(begin_time=t0, transition_time=0)

        unwritten = []
        # --- A4 Coxeter-Dynkin Diagram: x--x--x--x ---
        dia = DynkinDiagram.from_string("x3x3x3x", location=[0, 0, 3.5],
                                        scale=1, no_threes=True)
        t0 = dia.appear(begin_time=t0, transition_time=0)
        t0 = dia.appear_customized(rings=[0, 1, 2, 3],
                                   begin_time=t0, transition_time=0)

        # --- Node labels: n_x, n_y, n_z, n_w below each node ---
        # With location=[0,0,4.5] and scale=1, nodes sit at x = -3, -1, 1, 3
        label_names = [r"n_x", r"n_y", r"n_z", r"n_w"]
        x_pos = [-3, -1, 1, 3]
        for name, x in zip(label_names, x_pos):
            lbl = Text(name, color="text", text_size="large",
                       aligned="center", location=[x, 0, 2.5])
            t0 = lbl.write(begin_time=t0, transition_time=0)

        # --- Explicit normal vectors ---
        normals = [
            r"n_x = \left(\begin{array}{c} 1\\0\\ 0\\ 0\end{array}\right)",
            r"n_w = \left(\begin{array}{c} 0\\0\\ 0\\ 1\end{array}\right)",
            r"n_y = {1\over 4}  \left(\begin{array}{c}2\\\sqrt{5}-1\\ \sqrt{5}+1\\ 0\end{array}\right)",
            r"n_z = {1\over 4}  \left(\begin{array}{c}0\\\sqrt{5}+1\\ \sqrt{5}-1\\ 2\end{array}\right)",
        ]

        x_positions = [-7.5, -2.5, 2.5, 7.5]
        for i, (nrm_str, x) in enumerate(zip(normals, x_positions)):
            nrm = Text(nrm_str, color="text", text_size="large",
                       aligned="center", location=[x, 0, 0.5])
            if i < 2:
                nrm.write(begin_time=0, transition_time=0)
            else:
                t0 = 0.75 + nrm.write(begin_time=t0, transition_time=0.5)

        t0 += 1  # pause

        # --- Reflection matrix formula ---
        formula = Text(r"R = I - 2\, n \otimes n^T",
                       color="drawing", text_size="large", aligned="center",
                       location=[0, 0, -2])
        t0 = 0.5 + formula.write(begin_time=t0, transition_time=0.5)

        t0 += 1

        matrices = [
            r"R_1=\left(\begin{array}{r r r r}-1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&1\end{array}\right)",
            r"R_2=\left(\begin{array}{r r r r}\frac{1}{2}&\frac{1}{4}-\frac{1}{4}\sqrt{5}&-\frac{1}{4}-\frac{1}{4}\sqrt{5}&0\\\frac{1}{4}-\frac{1}{4}\sqrt{5}&\frac{1}{4}+\frac{1}{4}\sqrt{5}&-\frac{1}{2}&0\\-\frac{1}{4}-\frac{1}{4}\sqrt{5}&-\frac{1}{2}&\frac{1}{4}-\frac{1}{4}\sqrt{5}&0\\0&0&0&1\end{array}\right)",
            r"R_3=\left(\begin{array}{r r r r}1&0&0&0\\0&\frac{1}{4}-\frac{1}{4}\sqrt{5}&-\frac{1}{2}&-\frac{1}{4}-\frac{1}{4}\sqrt{5}\\0&-\frac{1}{2}&\frac{1}{4}+\frac{1}{4}\sqrt{5}&\frac{1}{4}-\frac{1}{4}\sqrt{5}\\0&-\frac{1}{4}-\frac{1}{4}\sqrt{5}&\frac{1}{4}-\frac{1}{4}\sqrt{5}&\frac{1}{2}\end{array}\right)",
            r"R_4=\left(\begin{array}{r r r r}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&-1\end{array}\right)"
        ]

        x_positions = [-9, -3.6, 3.65, 8.95]
        for matrix_str, x in zip(matrices, x_positions):
            nrm = Text(matrix_str, color="text", text_size="normal",
                       aligned="center", location=[x, 0, -3.5])
            t0 = 0.75 + nrm.write(begin_time=t0, transition_time=0.5)

        t0 += 1  # pause

        self.t0 = t0

    def a4_python_script(self):
        """
        Show the Python script from computation_for_presentation.py that builds
        the four reflection matrices from the mirror normals and multiplies them
        until the full A4 group of 120 elements is generated.
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        title = Text(r"\text{Generating the Reflection Group }A_4",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        filename = os.path.join(LOC_FILE_DIR, "computation_for_presentation.py")
        cp = CodeParser(filename, recreate=False)
        display = CodeDisplay(cp, location=Vector([4, 0, -0.5]),
                              scales=[6, 5], number_of_lines=30, flat=True)
        t0 = display.appear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + cp.write(display, class_index=0,
                            begin_time=t0, transition_time=25, indent=0.3)

        self.t0 = t0

    def a4_matrix_parade(self):
        """
        Display the 120 elements of A4 as a scrolling parade of 4×4 matrices,
        reading coefficients from matrix_data.csv via the Show120MatricesModifier.
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -12, 2])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        title = Text(r"\text{The 120 Elements of }A_4",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        carrier = BObject(mesh=ibpy.create_mesh([(0, 0, 0)], []),
                          location=[0, 0, 0])
        mod = Show120MatricesModifier()

        # Set the CSV path to the absolute location of matrix_data.csv
        csv_node = ibpy.get_geometry_node_from_modifier(mod, "Import CSV")
        if csv_node is not None:
            csv_node.inputs["Path"].default_value = os.path.join(LOC_FILE_DIR, "matrix_data.csv")

        counter_node = ibpy.get_geometry_node_from_modifier(mod, "Counter")

        carrier.add_mesh_modifier(type="NODES", node_modifier=mod)
        carrier.appear(begin_time=t0, transition_time=0)

        if counter_node is not None:
            t0 = ibpy.change_default_integer(counter_node, from_value=0, to_value=119,
                                             begin_time=t0, transition_time=15)

        self.t0 = t0

    def a4_seed_point(self):
        """
        Find the seed point q that is equally distant from all four mirrors.
        Present the abstract conditions n_i . p = +/-1 for the omnitruncated
        pentachoron, the explicit linear system after substituting the normals,
        and the solution p = (1, 3, -3, 1) that will generate 120 image points.
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # --- Title ---
        title = Text(r"\text{Seed Point $q$ for }A_4",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5.5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        # --- Subtitle: equally distant from all mirrors ---
        subtitle = Text(r"\text{equally distant from all mirrors: } n_i \cdot q = \pm 1",
                        color="drawing", text_size="large", aligned="center",
                        location=[0, 0, 4.7])
        t0 = 0.5 + subtitle.write(begin_time=t0, transition_time=0.5)

        # --- Left column: abstract dot-product conditions for the omnitruncated pentachoron ---
        dot_header = Text(r"\text{Omnitruncated Pentachoron}",
                          color="drawing", text_size="large", aligned="center",
                          location=[-6, 0, 3.3])
        t0 = 0.5 + dot_header.write(begin_time=t0, transition_time=0.5)

        dot_eqs = [
            r"n_x \cdot q = +1",
            r"n_y \cdot q = -1",
            r"n_z \cdot q = +1",
            r"n_w \cdot q = -1",
        ]
        for i, eq_str in enumerate(dot_eqs):
            eq = Text(eq_str, color="text", text_size="large",
                      aligned="center", location=[-6, 0, 1.8 - i * 1.2])
            t0 = 0.25 + eq.write(begin_time=t0, transition_time=0.4)

        # --- Note about replacing ones with zeros for smaller representations ---
        note = Text(r"\text{replace ones by zeros} \rightarrow \text{smaller polychora}",
                    color="joker", text_size="large", aligned="center",
                    location=[-6, 0, -3.8])
        t0 = 0.5 + note.write(begin_time=t0, transition_time=0.5)

        # --- Right column: explicit linear system after substituting the larges ---
        explicit_header = Text(r"\text{Explicit linear system}",
                               color="drawing", text_size="large", aligned="center",
                               location=[6, 0, 3.3])
        t0 = 0.5 + explicit_header.write(begin_time=t0, transition_time=0.5)

        explicit_eqs = [
            r"q_x = 1",
            r"\tfrac{1}{2}q_x + \tfrac{1}{4}(\sqrt{5}-1)\, q_y + \tfrac{1}{4}(\sqrt{5}+1)\, q_z = -1",
            r"\tfrac{1}{4}(\sqrt{5}+1)\, q_y + \tfrac{1}{4}(\sqrt{5}-1)\, q_z + \tfrac{1}{2}q_w = +1",
            r"q_w = -1",
        ]
        for i, eq_str in enumerate(explicit_eqs):
            eq = Text(eq_str, color="text", text_size="large",
                      aligned="center", location=[6, 0, 1.8 - i * 1.2])
            t0 = 0.25 + eq.write(begin_time=t0, transition_time=0.5)

        # --- Solution ---
        solution = Text(r"\text{Solution:}\,\,\,\, q = (1,\,3,\,-3,\,-1)",
                        color="example", text_size="large", aligned="center",
                        location=[0, 0, -5.3])
        t0 = 0.5 + solution.write(begin_time=t0, transition_time=0.7)

        # # --- Closing line about the 120 images ---
        # closing = Text(r"\text{120 matrices} \rightarrow \text{120 image points}",
        #                color="drawing", text_size="large", aligned="center",
        #                location=[0, 0, -6.7])
        # t0 = 0.5 + closing.write(begin_time=t0, transition_time=0.5)

        t0 += 1
        self.t0 = t0

    def a4_image_generation(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        title = Text(r"\text{Generate The 120 Reflection Images}",
                     color="example", text_size="large", location=[-10.25, 0, 5.5], text_aligned="center")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        formula = Text(r"M_i\cdot q = p_i", color="green", text_size="Huge",
                       location=[-10.62, 0, 1.2])
        t0 = 0.5 + formula.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def a4_cell_normal(self):
        """
        Turn the 120 image points into geometry: starting from the seed point q,
        collect its four nearest neighbours, show how edges/faces/cells arise
        from singles/pairs/triples of them, and then compute the normal of one
        concrete cell using the generalized cross product (epsilon tensor).
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # ============================================================
        # Phase 1: seed point, nearest neighbours and their combinatorics
        # ============================================================
        phase1 = []

        title = Text(r"\text{From Points to Geometry}",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5.5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)
        phase1.append(title)

        seed = Text(r"\text{home base:}\,\,\, q = (1,\,3,\,-3,\,-1)",
                    color="drawing", text_size="large", aligned="center",
                    location=[0, 0, 4.5])
        phase1.append(seed)
        t0 = 0.5 + seed.write(begin_time=t0, transition_time=0.5)

        # --- Left column: four nearest neighbours ---
        nb_header = Text(r"\text{nearest neighbours of }q",
                         color="drawing", text_size="large", aligned="center",
                         location=[-6, 0, 3])
        phase1.append(nb_header)
        t0 = 0.5 + nb_header.write(begin_time=t0, transition_time=0.5)

        neighbours = [
            r"p_1 = (-1,\,3,\,-3,\,-1)",
            r"p_2 = (2,\,\tfrac{5+\sqrt{5}}{2},\,\tfrac{\sqrt{5}-5}{2},\,-1)",
            r"p_3 = (1,\,\tfrac{5-\sqrt{5}}{2},\,\tfrac{-5-\sqrt{5}}{2},\,-2)",
            r"p_4 = (1,\,3,\,-3,\,1)",
        ]

        for i, nb_str in enumerate(neighbours):
            nb = Text(nb_str, color="text", text_size="large",
                      aligned="left", location=[-6, 0, 1.6 - i * 1.2])
            phase1.append(nb)
            t0 = 0.25 + nb.write(begin_time=t0, transition_time=0.5)

        # --- Right column: singles / pairs / triples -> edges / faces / cells ---
        comb_header = Text(r"\text{combinatorics}",
                           color="drawing", text_size="large", aligned="center",
                           location=[6, 0, 3])
        phase1.append(comb_header)
        t0 = 0.5 + comb_header.write(begin_time=t0, transition_time=0.5)

        comb_rules = [
            r"\text{single neighbour} \rightarrow \text{edge}",
            r"\text{pair of neighbours} \rightarrow \text{face}",
            r"\text{triple of neighbours} \rightarrow \text{cell}",
        ]
        for i, rule_str in enumerate(comb_rules):
            rule = Text(rule_str, color="text", text_size="large",
                        aligned="center", location=[6, 0, 1.4 - i * 1.4])
            phase1.append(rule)
            t0 = 0.5 + rule.write(begin_time=t0, transition_time=0.5)

        bridge = Text(r"\text{pick three neighbours} \rightarrow \text{compute the cell normal}",
                      color="joker", text_size="large", aligned="center",
                      location=[0, 0, -5])
        phase1.append(bridge)
        t0 = 1 + bridge.write(begin_time=t0, transition_time=0.5)

        # Clear phase 1 before showing the normal computation
        for obj in phase1:
            obj.unwrite(begin_time=t0, transition_time=0.3)
        t0 += 0.8

        # ============================================================
        # Phase 2: compute the cell normal via the epsilon tensor
        # ============================================================
        title2 = Text(r"\text{Cell Normal}",
                      color="example", text_size="large", aligned="center",
                      location=[0, 0, 5.5])
        t0 = 0.5 + title2.write(begin_time=t0, transition_time=0.5)

        # The three chosen neighbours
        chosen_header = Text(r"\text{chosen triple:}",
                             color="drawing", text_size="large", aligned="left",
                             location=[-9, 0, 4.5])
        t0 = 0.25 + chosen_header.write(begin_time=t0, transition_time=0.4)

        chosen = [
            r"p_1 = (-1,\,3,\,-3,\,-1)",
            r"p_2 = (2,\,\tfrac{5+\sqrt{5}}{2},\,\tfrac{\sqrt{5}-5}{2},\,-1)",
            r"p_3 = (1,\,\tfrac{5-\sqrt{5}}{2},\,\tfrac{-5-\sqrt{5}}{2},\,-2)",
        ]
        for i, p_str in enumerate(chosen):
            p = Text(p_str, color="text", text_size="large",
                     aligned="left", location=[-9, 0, 3.3 - i * 1.1])
            t0 = 0.25 + p.write(begin_time=t0, transition_time=0.5)

        # Constraint equations
        constraint = Text(r"n \cdot (p_i - q) = 0, \quad i = 1,2,3",
                          color="drawing", text_size="large", aligned="center",
                          location=[0, 0, -0.3])
        t0 = 0.5 + constraint.write(begin_time=t0, transition_time=0.5)

        # Generalized cross product via epsilon tensor
        u_def = Text(r"u = p_1 - q,\,\, v = p_2 - q,\,\, w = p_3 - q",
                     color="text", text_size="large", aligned="center",
                     location=[0, 0, -1.8])
        t0 = 0.5 + u_def.write(begin_time=t0, transition_time=0.5)

        eps_formula = Text(r"n^{\,i} = \epsilon^{ijkl}\, u_j\, v_k\, w_l",
                           color="drawing", text_size="large", aligned="center",
                           location=[0, 0, -3.3])
        t0 = 0.5 + eps_formula.write(begin_time=t0, transition_time=0.7)

        # Result
        result = Text(r"n = (0,\,-1-\sqrt{5},\,-1+\sqrt{5},\,2\sqrt{5})",
                      color="example", text_size="large", aligned="center",
                      location=[0, 0, -5.1])
        t0 = 0.75 + result.write(begin_time=t0, transition_time=0.8)

        remark = Text(r"\text{(no normalization needed)}",
                      color="joker", text_size="large", aligned="center",
                      location=[7.6, 0, -5.1])
        t0 = 0.5 + remark.write(begin_time=t0, transition_time=0.4)

        t0 += 1
        self.t0 = t0

    def a4_cell_projection(self):
        """
        Final scene: orthonormalize (u, v, w) into a basis for the cell,
        project each of the 24 cell vertices into that basis, and display the
        resulting truncated octahedron that forms one facet of the 120-cell.
        """
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # --- Title ---
        title = Text(r"\text{A Facet of the 120-Cell}",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        # --- Projection explanation (top-left) ---
        step1 = Text(r"(u,\,v,\,w) \xrightarrow{\,\text{Gram--Schmidt}\,} (e_1,\,e_2,\,e_3)",
                     color="drawing", text_size="large", aligned="center",
                     location=[0, 0, 3.8])
        t0 = 0.5 + step1.write(begin_time=t0, transition_time=0.5)

        details = [
            r"e_1 = \frac{u}{\sqrt{u\cdot u}}",
            r"v_\perp = v-(v\cdot e_1) e_1",
            r"e_2 = \frac{v_\perp}{\sqrt{v_\perp\cdot v_\perp}}",
            r"w_\perp = w-(w\cdot e_1) e_1-(w\cdot e_2) e_2",
            r"e_3 = \frac{w_\perp}{\sqrt{w_\perp\cdot w_\perp}}"
        ]

        for i,detail in enumerate(details):
            text = Text(detail, color="text",location = [-0,0,2.8-i])
            t0 = 0.5 + text.write(begin_time=t0, transition_time=0.5)

        step2 = Text(r"p \mapsto (p\cdot e_1,\,p\cdot e_2,\,p\cdot e_3)",
                     color="drawing", text_size="large", aligned="center",
                     location=[-0, 0, -3.5])
        t0 = 0.5 + step2.write(begin_time=t0, transition_time=0.5)

        step3 = Text(r"\text{24 three-dimensional coordinates}",
                     color="text", text_size="large", aligned="center",
                     location=[-0, 0, -4.5])
        t0 = 0.5 + step3.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def a4_neighbors(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # --- Title ---
        title = Text(r"\text{Find nearest Neighbours of q}",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        formula = Text(r"\text{Distance function:}",color="joker",text_size="large",location = [-4,0,3])
        t0 =0.5 + formula.write(begin_time=t0,transition_time=0.5)

        dist = Text(r"d(p_i,q)=\sqrt{(p_i-q)\cdot (p_i-q)}",color="text",text_size="large",location = [-4,0,1.5])
        t0 = 0.5 + dist.write(begin_time=t0,transition_time=0.5)
        self.t0 = t0


    def a4_cell_equation(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=64, motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        ibpy.set_camera_location(location=[0, -30, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # --- Title ---
        title = Text(r"\text{The Cell Equation}",
                     color="example", text_size="large", aligned="center",
                     location=[0, 0, 5])
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        lines = [r"\text{Normal: } n = (0,\,-1-\sqrt{5},\,-1+\sqrt{5},\,2\sqrt{5})",
                 r"\text{Seed: } q = (1,3,-3,-1)",
                 r"\text{Cell Equation: } n\cdot (p_i-q)=0"
                 ]

        for i,line in enumerate(lines):
            text = Text(line,color="text",text_size="large",location = [-10,0,3.5-i])
            t0 = 0.5 + text.write(begin_time=t0,transition_time=0.5)

        sat = Text(r"\text{Satisfied by the following 24 points: }",color="joker",location = [1,0,4],text_size="large")

        vertices = ['p_{1}=\\left(-1,-\\tfrac{1}{2}+\\tfrac{1}{2}\\sqrt{5},\\tfrac{1}{2}+\\tfrac{1}{2}\\sqrt{5},-4\\right)', 'p_{2}=\\left(-1,3,-3,-1\\right)', 'p_{3}=\\left(-2,\\tfrac{5}{2}+\\tfrac{1}{2}\\sqrt{5},-\\tfrac{5}{2}+\\tfrac{1}{2}\\sqrt{5},-1\\right)', 'p_{4}=\\left(-1,\\tfrac{1}{2}-\\tfrac{1}{2}\\sqrt{5},-\\tfrac{1}{2}-\\tfrac{1}{2}\\sqrt{5},-4\\right)', 'p_{5}=\\left(2,\\tfrac{5}{2}+\\tfrac{1}{2}\\sqrt{5},-\\tfrac{5}{2}+\\tfrac{1}{2}\\sqrt{5},-1\\right)', 'p_{6}=\\left(1,+1\\sqrt{5},+1\\sqrt{5},-3\\right)', 'p_{7}=\\left(3,\\tfrac{3}{2}+\\tfrac{1}{2}\\sqrt{5},-\\tfrac{3}{2}+\\tfrac{1}{2}\\sqrt{5},-2\\right)', 'p_{8}=\\left(1,2+1\\sqrt{5},-2+1\\sqrt{5},-1\\right)', 'p_{9}=\\left(2,0,0,-4\\right)', 'p_{10}=\\left(2,1+1\\sqrt{5},-1+1\\sqrt{5},-2\\right)', 'p_{11}=\\left(2,\\tfrac{3}{2}-\\tfrac{1}{2}\\sqrt{5},-\\tfrac{3}{2}-\\tfrac{1}{2}\\sqrt{5},-3\\right)', 'p_{12}=\\left(-2,0,0,-4\\right)', 'p_{13}=\\left(-1,\\tfrac{5}{2}-\\tfrac{1}{2}\\sqrt{5},-\\tfrac{5}{2}-\\tfrac{1}{2}\\sqrt{5},-2\\right)', 'p_{14}=\\left(1,\\tfrac{1}{2}-\\tfrac{1}{2}\\sqrt{5},-\\tfrac{1}{2}-\\tfrac{1}{2}\\sqrt{5},-4\\right)', 'p_{15}=\\left(-3,1,-1,-3\\right)', 'p_{16}=\\left(1,\\tfrac{5}{2}-\\tfrac{1}{2}\\sqrt{5},-\\tfrac{5}{2}-\\tfrac{1}{2}\\sqrt{5},-2\\right)', 'p_{17}=\\left(-3,\\tfrac{3}{2}+\\tfrac{1}{2}\\sqrt{5},-\\tfrac{3}{2}+\\tfrac{1}{2}\\sqrt{5},-2\\right)', 'p_{18}=\\left(-2,1+1\\sqrt{5},-1+1\\sqrt{5},-2\\right)', 'p_{19}=\\left(1,-\\tfrac{1}{2}+\\tfrac{1}{2}\\sqrt{5},\\tfrac{1}{2}+\\tfrac{1}{2}\\sqrt{5},-4\\right)', 'p_{20}=\\left(3,1,-1,-3\\right)', 'p_{21}=\\left(-1,2+1\\sqrt{5},-2+1\\sqrt{5},-1\\right)', 'p_{22}=\\left(1,3,-3,-1\\right)', 'p_{23}=\\left(-1,+1\\sqrt{5},+1\\sqrt{5},-3\\right)',
                    'p_{24}=\\left(-2,\\tfrac{3}{2}-\\tfrac{1}{2}\\sqrt{5},-\\tfrac{3}{2}-\\tfrac{1}{2}\\sqrt{5},-3\\right)']
        t0 = 0.5 + sat.write(begin_time=t0, transition_time=0.5)

        for i,v in enumerate(vertices):
            text = Text(v,color="text",text_size="small",location = [1+5*(i//12),0,3-0.75*(i%12)])
            t0 = 0.5 + text.write(begin_time=t0,transition_time=0.5)



        self.t0 = t0

if __name__ == '__main__':
    try:
        example = VideoFullGeometry()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        if len(dictionary) == 1:
            selection = 0
        else:
            selection = input("Choose scene:")
            if len(selection) == 0:
                selection = 0
        print("Your choice: ", selection)
        selected_scene = dictionary[int(selection)]

        if "short" in selected_scene:
            resolution = [1080, 1920]
        else:
            resolution = [1920, 1080]
        example.create(name=selected_scene, resolution=resolution, start_at_zero=True)

    except:
        print_time_report()
        raise
