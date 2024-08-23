import time
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations
from random import shuffle
from timeit import default_timer as timer

import numpy as np
from anytree import NodeMixin

from appearance.textures import get_alpha_of_material
from geometry_nodes.geometry_nodes import create_hexagon_tilings, create_z5, de_bruijn, penrose_3D_analog, create_z3
from geometry_nodes.geometry_nodes_modifier import Penrose2D, Penrose2DIntro, Penrose2DVoronoi, \
    ConvexHull2D
from interface import ibpy
from interface.ibpy import Vector, Matrix, Quaternion, get_geometry_node_from_modifier, change_default_value, \
    apply_modifier, separate, make_rigid_body, convert_to_mesh, \
    set_simulation, set_material, change_default_boolean, change_default_vector
from interface.interface_constants import BLENDER_EEVEE
from mathematics.geometry.convex_hull import ConvexHull
from mathematics.geometry.polytope import Simplex
from mathematics.mathematica.mathematica import tuples, unit_tuples, random_points, mean, convex_hull, \
    negative_unit_tuples
from mathematics.zeros import chop
from objects.bmatrix import BMatrix
from objects.bobject import BObject
from objects.circle import BezierCircle
from objects.coordinate_system import CoordinateSystem
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.derived_objects.p_arrow import PArrow
from objects.digital_number import DigitalRange
from objects.display import Display, GlassDisplay
from objects.empties import EmptyCube
from objects.geometry.sphere import Sphere
from objects.light.light import AreaLight
from objects.plane import Plane
from objects.polygon import Polygon
from objects.some_logo import SoME
from objects.table import Table
from objects.tex_bobject import SimpleTexBObject, TexBObject
from objects.tree.tree import Tree
from perform.scene import Scene
from utils.constants import FRAME_RATE
from utils.utils import print_time_report, to_vector, flatten, pi


class Tuple(NodeMixin):
    def __init__(self, element):
        self.element = element

    def __str__(self):
        return '(' + '|'.join(map(str, self.element)) + ')'


def compose(lambda1, lambda2):
    return lambda v: lambda1(lambda2(v))


def tensor(a, b):
    return np.tensordot(a, b, axes=0)


def wedge(a, b):
    return tensor(a, b) - tensor(b, a)


def matrix_norm(a):
    return 0.5 * np.trace(np.tensordot(a, np.transpose(a), axes=1))


def projector(u, v):
    return (u.dot(u) * tensor(u, u) + v.dot(v) * tensor(v, v) - u.dot(v) * (tensor(u, v) + tensor(v, u))) / matrix_norm(
        wedge(u, v))


def ortho(u, v):
    dim = len(u)
    identity = np.identity(dim)
    return identity - projector(u, v)


def face_directions(dim):
    return [(a, b) for a in range(dim) for b in range(a + 1, dim)]


def plane_for_center_and_directions(center, units, trafo=lambda v: v):
    center = to_vector(center)
    units = [to_vector(u) for u in units]
    positions = [center, center + units[0], center + units[0] + units[1], center + units[1]]
    return [trafo(pos) for pos in positions]


def voronoi(dim):
    return tuples([0.5, -0.5], dim)


def close_points_3d(normal, iterations):
    roots = [Vector(v) for v in unit_tuples(3)] + [-1 * Vector(v) for v in unit_tuples(3)]
    selZ3 = {(0, 0, 0)}
    oldZ3 = deepcopy(selZ3)
    count = 0
    voronoi_points = voronoi(3)
    projected_voronoi_points = [normal.dot(Vector(a)) for a in voronoi_points]
    min_z = min(projected_voronoi_points)
    max_z = max(projected_voronoi_points)
    while count < iterations:
        count += 1
        newZ3 = list(filter(lambda v: min_z <= v.dot(normal) <= max_z,
                            [Vector(s) + r for s in oldZ3 for r in roots]
                            ))
        oldZ3 = {tuple(v) for v in newZ3}.difference(selZ3)
        selZ3 = selZ3.union({tuple(v) for v in newZ3})
    return selZ3


def close_points_5d(ortho, iterations):
    roots = [Vector(v) for v in unit_tuples(5)] + [-1 * Vector(v) for v in unit_tuples(5)]
    selZ5 = {(0, 0, 0, 0, 0)}
    oldZ5 = deepcopy(selZ5)
    count = 0
    voronoi_points = voronoi(5)
    cell_points = [np.dot(ortho, Vector(v)) for v in voronoi_points]
    ch = ConvexHull(cell_points)

    while count < iterations:
        count += 1
        newZ5 = list(filter(lambda v: ch.is_inside(np.dot(ortho, v)),
                            [Vector(s) + r for s in oldZ5 for r in roots]
                            ))
        oldZ5 = {tuple(v) for v in newZ5}.difference(selZ5)
        selZ5 = selZ5.union({tuple(v) for v in newZ5})
    return selZ5


def find_base_points(point_set, dim):
    faces = face_directions(dim)
    base_points = []
    roots = [Vector(u) for u in unit_tuples(dim)]
    for face in faces:
        face_base_points = []
        for p in point_set:
            p0 = p
            p1 = tuple(Vector(p) + roots[face[0]])
            p2 = tuple(Vector(p) + roots[face[0]] + roots[face[1]])
            p3 = tuple(Vector(p) + roots[face[1]])
            if {p0, p1, p2, p3}.issubset(point_set):
                face_base_points.append(p)
        base_points.append(face_base_points)
    return base_points


def baserep(number, base=3, length=5):
    digits = []
    power = base ** length
    for j in range(length):
        power /= base
        digits.append(int(np.floor(number / power)))
        number -= power * digits[-1]
    return tuple(digits)


def interpol2(s):
    """
     This interpolating matrix is calculated in
    RotationInterpolation.nb. It interchanges the two rotation planes for the Penrose tiling
    :param s:
    :return:
    """
    theta = s * np.pi / 2
    c = np.cos(theta)
    c2 = c * c
    s = np.sin(theta)
    s2 = s * s
    cs = c * s

    return np.array(
        [[c2, -cs, s2, cs, 0], [cs, c2, -cs, s2, 0], [s2, cs, c2, -cs, 0], [-cs, s2, cs, c2, 0], [0, 0, 0, 0, 1]])


def interpol(s):
    """
    This interpolating matrix is calculated in
    RotationInterpolation.nb. Basically, the initial and final rotation matrix
    are converted into Lie-algebra elements. Inbetween the two different algebra
    elements is linearly interpolated and then the result is lifted back into the
    rotation group.
    """
    c1 = np.cos(0.418476 * s)
    c2 = np.cos(2.76138 * s)
    s1 = np.sin(0.418476 * s)
    s2 = np.sin(2.76138 * s)
    return np.array([[
        0.773155 + 0.0682199 * c1 + 0.158625 * c2,
        -0.142662 - 0.145103 * c1 + 0.287766 * c2 + 0.0275851 * s1 - 0.144617 * s2,
        0.248295 - 0.0659743 * c1 - 0.182321 * c2 - 0.0814771 * s1 - 0.295261 * s2,
        -0.280672 + 0.148834 * c1 + 0.131838 * c2 - 0.165017 * s1 - 0.101279 * s2,
        0.120857 - 0.126518 * c1 + 0.00566077 * c2 - 0.183275 * s1 + 0.200685 * s2
    ],
        [-0.142662 - 0.145103 * c1 + 0.287766 * c2 - 0.0275851 * s1 + 0.144617 * s2,
         0.026324 + 0.319788 * c1 + 0.653888 * c2,
         -0.0458153 + 0.107381 * c1 - 0.0615658 * c2 + 0.199978 * s1 - 0.70186 * s2,
         0.0517896 - 0.383295 * c1 + 0.331506 * c2 + 0.290808 * s1 - 0.0635378 * s2,
         -0.0223005 + 0.194994 * c1 - 0.172693 * c2 + 0.440983 * s1 + 0.369228 * s2
         ],
        [
            0.248295 - 0.0659743 * c1 - 0.182321 * c2 + 0.0814771 * s1 + 0.295261 * s2,
            -0.0458153 + 0.107381 * c1 - 0.0615658 * c2 - 0.199978 * s1 + 0.70186 * s2,
            0.0797388 + 0.161113 * c1 + 0.759148 * c2,
            -0.0901366 + 0.0531498 * c1 + 0.0369868 * c2 + 0.337343 * s1 + 0.361809 * s2,
            0.0388126 + 0.341244 * c1 - 0.380057 * c2 + 0.0261384 * s1 - 0.220127 * s2
        ],
        [
            -0.280672 + 0.148834 * c1 + 0.131838 * c2 + 0.165017 * s1 + 0.101279 * s2,
            0.0517896 - 0.383295 * c1 + 0.331506 * c2 - 0.290808 * s1 + 0.0635378 * s2,
            -0.0901366 + 0.0531498 * c1 + 0.0369868 * c2 - 0.337343 * s1 - 0.361809 * s2,
            0.10189 + 0.72387 * c1 + 0.174239 * c2,
            -0.0438737 + 0.167303 * c1 - 0.123429 * c2 - 0.705882 * s1 + 0.170409 * s2
        ],
        [
            0.120857 - 0.126518 * c1 + 0.00566077 * c2 + 0.183275 * s1 - 0.200685 * s2,
            -0.0223005 + 0.194994 * c1 - 0.172693 * c2 - 0.440983 * s1 - 0.369228 * s2,
            0.0388126 + 0.341244 * c1 - 0.380057 * c2 - 0.0261384 * s1 + 0.220127 * s2,
            -0.0438737 + 0.167303 * c1 - 0.123429 * c2 + 0.705882 * s1 - 0.170409 * s2,
            0.0188919 + 0.727009 * c1 + 0.254099 * c2
        ]])


r2 = np.sqrt(2)
r3 = np.sqrt(3)
r5 = np.sqrt(5)

pi = np.pi


class Penrose(Scene):
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('transition', {'duration': 91}),
            ('flatsmiley', {'duration': 10}),
            ('some', {'duration': 10}),
            ('convex_hull_2d', {'duration': 30}),
            ('projection_2d_voronoi', {'duration': 140}),
            ('projection_2d_intro', {'duration': 40}),
            ('projection_2d', {'duration': 30}),
            ('three_d_rotations', {'duration': 50}),
            ('five_d_rotations', {'duration': 70}),
            ('dynamical_convex_hull', {'duration': 60}),
            ('title', {'duration': 10}),
            ('branding', {'duration': 25}),
            ('geo_penrose', {'duration': 9}),
            ('hexagons', {'duration': 9}),
            ('hexagons2', {'duration': 18}),
            ('intro', {'duration': 95}),
            ('intro_penrose', {'duration': 30}),
            ('intro_overlay', {'duration': 30}),
            ('hyper_cube', {'duration': 23}),
            ('hyper_cube_data', {'duration': 75}),
            ('hyper_cube_data2', {'duration': 45}),
            ('trivial_projection_3d', {'duration': 70}),
            ('trivial_projection_3d2', {'duration': 45}),
            ('trivial_projection_3d3', {'duration': 45}),
            ('trivial_projection_3d4', {'duration': 21}),
            ('plane_rotation_3d', {'duration': 36}),
            ('plane_rotation_5d', {'duration': 77}),
            ('convex_hull_3d', {'duration': 50}),
            ('maybe_penrose', {'duration': 65}),
            ('maybe_penrose2', {'duration': 65}),
            ('penrose_rotation', {'duration': 38}),
            ('penrose_rotation2', {'duration': 25}),
            ('penrose_tiling', {'duration': 225}),
            ('penrose_tiling2', {'duration': 95}),
            ('penrose_merger', {'duration': 15}),
            ('penrose_merger2', {'duration': 15}),
            ('optimize_object_creation', {'duration': 15}),
            ('optimize_object_creation2', {'duration': 15}),
            ('hide_test', {'duration': 11}),
            ('citation', {'duration': 40}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def transition(self):

        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[-2, -9, 0])
        empty = EmptyCube(location=Vector((-2, 0, 0)))
        ibpy.set_camera_view_to(empty)

        plane = Plane(resolution=[10, 10], uniformization=False)
        plane.move(direction=[0, 0, -0.5], begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)

        z5_nodes = create_z5(n=5, base="STANDARD",
                             final_rotation=[pi / 2, 0, 0],
                             final_scale=[0.25] * 3,
                             final_translation=[-4.5, 0, 0])
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)

        ts = [ibpy.get_geometry_node_from_modifier(z5_nodes, 't_' + str(i)) for i in range(10)]

        plane.appear(begin_time=t0, transition_time=0)

        # computations of the angles are performed in GeneralizedEulerAngles5D.nb

        change_default_value(ts[0].outputs[0], to_value=0.553574, from_value=-1.01722, begin_time=t0,
                                        transition_time=90)
        change_default_value(ts[1].outputs[0], to_value=-0.48402, from_value=-2.43672, begin_time=t0,
                                        transition_time=90)
        change_default_value(ts[2].outputs[0], to_value=2.49621, from_value=-2.7607, begin_time=t0,
                                        transition_time=90)
        change_default_value(ts[4].outputs[0], to_value=0.74951, from_value=0.913563, begin_time=t0,
                                        transition_time=90)
        change_default_value(ts[5].outputs[0], to_value=0.321336, from_value=-0.791708, begin_time=t0,
                                        transition_time=90)
        change_default_value(ts[6].outputs[0], to_value=2.45687, from_value=2.45687, begin_time=t0,
                                        transition_time=90)
        change_default_value(ts[7].outputs[0], to_value=-1.0582, from_value=0.397177, begin_time=t0,
                                        transition_time=90)
        change_default_value(ts[8].outputs[0], to_value=-2.18628, from_value=-0.955317, begin_time=t0,
                                        transition_time=90)
        t0 =0.5+ change_default_value(ts[9].outputs[0], to_value=0, from_value=-pi, begin_time=t0, transition_time=90)

        self.t0 = t0
    def flatsmiley(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=False,
                                 transparent=True, no_transmission_ray=True, background='drawing',
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=1024)

        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        frownie = TexBObject(r"\text{\frownie{}}", r"\text{\smiley{}}", location=[2.25, 0, -5], text_size='Huge',color='important')
        t0 = 0.5+ frownie.write(begin_time=t0,transition_time=1)

        t0 = 0.5 + frownie.move(direction=[-4.5,0,0],begin_time=t0,transition_time=1)

        frownie.next(begin_time=t0,transition_time=1)
        frownie.perform_morphing()
        t0 = 0.5 + frownie.change_color(new_color='joker',begin_time=t0,transition_time=1)

        self.t0 = t0

    def some(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=False,
                                 transparent=True, no_transmission_ray=True, background='drawing',
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=1024)

        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        logo = SoME(location=[0,0, 0], rotation_euler=[0, 0, 0], scale=3)
        t0 = logo.appear(begin_time=t0, transition_time=5)
        self.t0 = t0

    def convex_hull_2d(self):

        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=False,
                                 transparent=True,no_transmission_ray=True,background='drawing',
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=1024)

        ibpy.set_camera_location(location=[0, 0, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        cube = Cube(rotation_euler=[pi / 2, 0, 0])
        geo_nodes = ConvexHull2D(size=5)
        cube.add_mesh_modifier(type='NODES', node_group=geo_nodes.get_node_tree())

        # rotate camera during build up

        camera_circle = BezierCircle(location=[0, 0, 1], radius=20)

        ibpy.set_camera_view_to(empty)
        ibpy.set_camera_follow(camera_circle)

        ibpy.camera_follow(camera_circle,initial_value=0.425,final_value=0.675,begin_time=0,transition_time=20)

        # change normal to show different convex hull configurations
        radius = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "rMax").outputs[0]
        t0  =0.5 + ibpy.change_default_value(radius,from_value=0,to_value=6*r3,begin_time=0.25,transition_time=2)

        # grow projection line
        length = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(),"lMax").outputs[0]
        t0 = 0.5 + ibpy.change_default_value(length,from_value=0,to_value=1, begin_time=t0,transition_time=2)


        # make voronoi appear
        voronoi_scale = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(),"voronoiScale").outputs[0]
        t0 = 0.5+ibpy.change_default_value(voronoi_scale,from_value=0,to_value=1,begin_time=t0,transition_time= 2)

        # make projection appear
        ch_max_mat = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(),"ConvexHullMaterialMax")
        ch_min_mat = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(),"ConvexHullMaterialMin")
        ibpy.change_alpha_of_material(ch_min_mat,from_value=0,to_value=1,begin_time=t0,transition_time=2)
        t0 = 0.5+ ibpy.change_alpha_of_material(ch_max_mat,from_value=0,to_value=1,begin_time=t0,transition_time=2)

        # make zone appear
        p_max = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(),"pMax").outputs[0]
        ibpy.change_default_value(p_max,from_value=0,to_value=1,begin_time=t0, transition_time=3)
        ico_max = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(),"icoMax").outputs[0]
        ibpy.change_default_value(ico_max, from_value=0, to_value=1, begin_time=t0, transition_time=3)
        zone_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "ZoneMaterial")
        t0 = 0.5+ibpy.change_alpha_of_material(zone_material,from_value=0,to_value=1,begin_time=t0,transition_time=3)


        p_max = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "pMax").outputs[0]
        ibpy.change_default_value(p_max, to_value=0, from_value=1, begin_time=t0, transition_time=6)
        ico_max = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "icoMax").outputs[0]
        ibpy.change_default_value(ico_max, to_value=0, from_value=1, begin_time=t0, transition_time=6)
        zone_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "ZoneMaterial")
        ibpy.camera_zoom(lens=150,begin_time=t0,transition_time=6)
        t0 = 0.5 + ibpy.change_alpha_of_material(zone_material, to_value=0, from_value=1, begin_time=t0,
                                                 transition_time=6)

        # move projections from back to end and vice-versa
        normal = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(),"Normal")
        t0 = 0.5 + ibpy.change_default_vector(normal,from_value=Vector([1,0,0]),to_value=Vector([1,1,1]),begin_time=t0,transition_time=3)


        p_max = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "pMax").outputs[0]
        ibpy.change_default_value(p_max, from_value=0, to_value=1, begin_time=t0, transition_time=6)
        ico_max = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "icoMax").outputs[0]
        ibpy.change_default_value(ico_max, from_value=0, to_value=1, begin_time=t0, transition_time=6)
        zone_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "ZoneMaterial")
        ibpy.camera_zoom(lens=30, begin_time=t0, transition_time=4)
        print(t0 * FRAME_RATE)
        t0 = 0.5 + ibpy.change_alpha_of_material(zone_material, from_value=0, to_value=1, begin_time=t0,
                                                 transition_time=6)

        self.t0 = t0

    def projection_2d_voronoi(self):

        t0 = 0
        #ibpy.set_hdri_background("forest", 'exr', simple=True,
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=1024)

        ibpy.set_camera_location(location=[0, -20, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        cube = Cube(rotation_euler=[pi / 2, 0, 0])
        geo_nodes = Penrose2DVoronoi()
        cube.add_mesh_modifier(type='NODES', node_group=geo_nodes.get_node_tree())

        radius = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "GrowGrid")
        length = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "GrowLine")
        angle = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "Angle")
        shift = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "VoronoiShift")
        offset = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "GridShift")
        projection = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "Projector")
        cell_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "VoronoiCellMaterial")
        icosphere_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "IcosphereMaterial")
        grid_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "GridMaterial")
        lattice_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "LatticeMaterial")
        max_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "MaxMaterial")
        min_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "MinMaterial")
        projection_line_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "ProjectionMaterial")
        horizontal_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "HorizontalMaterial")
        vertical_material = ibpy.get_material_from_modifier(geo_nodes.get_node_tree(), "VerticalMaterial")

        materials = [cell_material, icosphere_material, grid_material, lattice_material, max_material, min_material,
                     projection_line_material, horizontal_material, vertical_material]

        # grow voronoi zone

        # make cell appear
        t0 = 0.5 + ibpy.change_alpha_of_material(cell_material, from_value=0, to_value=1, begin_time=t0,
                                                 transition_time=1)

        # rotate projection line
        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=pi / 2, to_value=3 * pi / 5, begin_time=t0,
                                             transition_time=5)

        # make zone boundary appear
        t0 = 0.5 + ibpy.change_default_value(length.outputs[0], from_value=0, to_value=30, begin_time=t0,
                                             transition_time=5)

        # make a full turn
        ibpy.change_default_value(radius.outputs[0], from_value=0, to_value=30, begin_time=t0 + 10, transition_time=2)
        ibpy.change_alpha_of_material(icosphere_material, from_value=0, to_value=1, begin_time=t0 + 2,
                                      transition_time=2)
        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=3 * pi / 5, to_value=13 * pi / 5,
                                             begin_time=t0,
                                             transition_time=20)

        # shift Voronoi cell
        t0 = 0.5 + ibpy.change_default_vector(shift, from_value=Vector(), to_value=Vector([1.1, 3.7, 0]),
                                              begin_time=t0, transition_time=5)

        # shift grid

        t0 = ibpy.change_default_vector(offset, from_value=Vector(), to_value=Vector([0.7, 0, 0]), begin_time=t0,
                                        transition_time=3)
        t0 = ibpy.change_default_vector(offset, from_value=Vector([0.7, 0, 0]), to_value=Vector([0.7, -0.7, 0]),
                                        begin_time=t0, transition_time=3)
        t0 = 0.5 + ibpy.change_default_vector(offset, from_value=Vector([0.7, -0.7, 0]), to_value=Vector([0, 0, 0]),
                                              begin_time=t0, transition_time=3)

        t0 = 0.5 + ibpy.change_default_value(projection.outputs[0], from_value=0, to_value=1, begin_time=t0,
                                             transition_time=5)

        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=13 * pi / 5, to_value=11 * pi / 5,
                                             begin_time=t0,
                                             transition_time=15)

        #reset everything

        t0 = ibpy.change_default_value(projection.outputs[0], from_value=1, to_value=0, begin_time=t0,
                                       transition_time=3)
        t0 = ibpy.change_default_value(angle.outputs[0], from_value=11 * pi / 5, to_value=5 * pi / 2, begin_time=t0,
                                       transition_time=3)
        t0 = ibpy.change_default_vector(offset, from_value=Vector(), to_value=Vector([0, -0.001, 0]), begin_time=t0,
                                        transition_time=3)
        t0 = 0.5 + ibpy.change_default_vector(shift, from_value=Vector([1.1, 3.7, 0]), to_value=Vector(), begin_time=t0,
                                              transition_time=3)

        fade = 0.03125
        for material in materials:
            ibpy.change_alpha_of_material(material, from_value=1, to_value=fade, begin_time=t0 + 2,
                                          transition_time=0.5)
        t0 += 1

        # display and coordinate system

        coords = CoordinateSystem(dim=2, lengths=[20, 10], domains=[[-10, 10], [-5, 5]],
                                  all_n_tics=[10, 5],
                                  all_tic_lablls=[np.arange(-10, 10.1, 2), np.arange(-5, 5.1, 2)],
                                  label_digits=[0, 0],
                                  radii=[0.015, 0.015, 0.015],
                                  include_zeros=[False] * 2
                                  )
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=3)
        coords.disappear(alpha=0.5, begin_time=t0, transition_time=0.5)

        empty.move(direction=[5, 0, 0], begin_time=t0, transition_time=1)
        ibpy.camera_move(shift=[5, 0, 0], begin_time=t0, transition_time=1)
        display = Display(flat=True, number_of_lines=22, location=[11.69, -2.1, 0])
        cube.rescale(rescale=3, begin_time=t0, transition_time=1)
        coords.rescale(rescale=3, begin_time=t0, transition_time=1)
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1)

        # explain grid
        colors = flatten([['gray_5'] * 8, ['custom3'] * 6, ['gray_5']])
        grid = SimpleTexBObject(r"\text{grid with offset:}", color=colors)
        ibpy.change_alpha_of_material(grid_material, from_value=0.1, to_value=1, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + display.write_text_in(grid, letter_set=[0, 1, 2, 3], line=0, indent=0.5, begin_time=t0,
                                         transition_time=0.3)

        arrow = PArrow(start=Vector(), end=coords.coords2location([2, -1]), name="GridExample", color='gray_5')
        label = SimpleTexBObject(r"\left(\!\!\!\begin{array}{c} 2\\-1\end{array}\!\!\!\right)",
                                 location=coords.coords2location([1.5, -1.5]), aligned='left', color='gray_5')
        coords.add_objects([arrow, label])
        label.write(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + arrow.grow(begin_time=t0, transition_time=0.5)

        colors = flatten([['gray_5'] * 8, ['custom3'] * 2, ['gray_5'] * 13, ['custom3'] * 9, ['gray_5']])
        grid2 = SimpleTexBObject(
            r"P=\left\{\left . \left(\begin{array}{c} k\\ l \end{array}\right) +\vec{\sigma}\,\,\right |\,\, (k,l)\in \mathbb{Z}^2,\vec{\sigma} \in [0,1)^2\right\}",
            recreate=False, color=colors)
        t0 = 0.5 + display.write_text_in(grid2,
                                         letter_set=[0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                     21, 32], line=2, indent=0.5, begin_time=t0, transition_time=1)

        dx = 0.3
        dy = -0.1
        arrow2 = PArrow(start=Vector(), end=coords.coords2location([dx, dy]), color='custom3', name="GridShiftArrow")
        coords.add_object(arrow2)
        arrow.move(direction=coords.coords2location([dx, dy]), begin_time=t0, transition_time=2)
        label.move(direction=coords.coords2location([dx, dy]), begin_time=t0, transition_time=2)
        grid2.write(letter_set=[7, 9, 8, 23, 22, 24, 25, 26, 27, 28, 29, 30, 31], begin_time=t0, transition_time=0.5)
        ibpy.change_default_vector(offset, from_value=Vector([0, -0.001, 0]), to_value=Vector([dx, dy, 0]),
                                   begin_time=t0, transition_time=2)
        grid.write(letter_set=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + arrow2.grow(begin_time=t0, transition_time=2)

        ibpy.change_alpha_of_material(grid_material, from_value=1, to_value=fade, begin_time=t0, transition_time=0.5)
        arrow.disappear(alpha=fade, begin_time=t0, transition_time=0.5)
        arrow2.disappear(alpha=fade, begin_time=t0, transition_time=0.5)
        label.disappear(alpha=fade, begin_time=t0, transition_time=0.5)

        # explain normal
        normal = SimpleTexBObject(r"\text{normal:}", color="custom1")
        ibpy.change_alpha_of_material(projection_line_material, from_value=0.1, to_value=1, begin_time=t0,
                                      transition_time=0.5)
        t0 = 0.5 + display.write_text_in(normal, line=4, indent=0.35, begin_time=t0, transition_time=0.3)

        arrow3 = PArrow(start=Vector(), end=coords.coords2location([0, 1]), color='custom1', name="NormalArrow")
        coords.add_object(arrow3)

        normal_comp = SimpleTexBObject(r"\vec{n}=\left(\!\!\begin{array}{c} 0\\1\end{array}\!\!\right)",
                                       color='custom1')
        normal_comp2 = SimpleTexBObject(
            r"\vec{n}=\left(\!\!\begin{array}{c} -\sin\varphi\\\cos\varphi\end{array}\!\!\right)", color='custom1')
        display.write_text_in(normal_comp, line=6, indent=0.35, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + arrow3.grow(begin_time=t0, transition_time=1.5)

        normal_comp.replace(normal_comp2, begin_time=t0, transition_time=1)
        arrow3.rotate(rotation_euler=[0, -pi / 10, 0], begin_time=t0, transition_time=2)
        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=5 * pi / 2, to_value=13 * pi / 5,
                                             begin_time=t0, transition_time=2)

        # explain voronoi
        arrow3.disappear(alpha=fade, begin_time=t0, transition_time=0.5)
        ibpy.change_alpha_of_material(projection_line_material, from_value=1, to_value=fade, begin_time=t0,
                                      transition_time=0.5)

        unit_cube = SimpleTexBObject(r"\text{unit cube:}", color='drawing')
        display.write_text_in(unit_cube, line=8, indent=0.35, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + ibpy.change_alpha_of_material(cell_material, from_value=0.1, to_value=1, begin_time=t0,
                                                 transition_time=1)

        corners = [[0, 0], [1, 0], [0, 1], [1, 1]]
        corner_labels = [r"\vec{d}_l", r"\vec{d}_r", r"\vec{u}_l", r"\vec{u}_r"]
        corner_spheres = []
        corner_texts = []
        i = 0
        for corner, label in zip(corners, corner_labels):
            corner_texts.append(SimpleTexBObject(
                label + r"=\left(\!\!\!\begin{array}{c}" + str(corner[0]) + r"\\" + str(
                    corner[1]) + r"\end{array}\!\!\!\right)", color='drawing'))
            corner_spheres.append(Sphere(0.05, location=coords.coords2location(corner), color='drawing'))
            display.write_text_in(corner_texts[-1], line=10 + 2 * int(i / 4), indent=0.35 + 2.5 * (i % 4),
                                  begin_time=t0, transition_time=0.5)
            t0 = 0.5 + corner_spheres[-1].grow(begin_time=t0, transition_time=1)
            i = i + 1
        coords.add_objects(corner_spheres)

        # selection zone
        arrow3.disappear(alpha=1, begin_time=t0, transition_time=0.5)
        ibpy.change_alpha_of_material(projection_line_material, from_value=fade, to_value=1, begin_time=t0,
                                      transition_time=0.5)
        zone = SimpleTexBObject(r"\text{selection zone:}", color='drawing')
        t0 = 0.5 + display.write_text_in(zone, line=12, indent=0.35, begin_time=t0, transition_time=0.5)

        colors = flatten(
            [['drawing'] * 9, ['custom1'] * 2, ['drawing'] * 5, ['custom1'] * 2, ['drawing'] * 5, ['custom1'] * 2,
             ['drawing'] * 5, ['custom1'] * 2, ['drawing']])
        max_text = SimpleTexBObject(
            r"d_\text{max}=\max\{\vec{n}\cdot \vec{d}_l,\vec{n}\cdot \vec{d}_r,\vec{n}\cdot \vec{u}_l,\vec{n}\cdot \vec{u}_r\}",
            color=colors)
        display.write_text_in(max_text, line=13, indent=0.35, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + ibpy.change_alpha_of_material(max_material, from_value=fade, to_value=1, begin_time=t0,
                                                 transition_time=1)

        min_text = SimpleTexBObject(
            r"d_\text{min}=\,\min\,\{\vec{n}\cdot \vec{d}_l,\vec{n}\cdot \vec{d}_r,\vec{n}\cdot \vec{u}_l,\vec{n}\cdot \vec{u}_r\}",
            color=colors)
        min_text.align(max_text, char_index=4, other_char_index=4)
        display.write_text_in(min_text, line=14, indent=0.35, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + ibpy.change_alpha_of_material(min_material, from_value=fade, to_value=1, begin_time=t0,
                                                 transition_time=1)

        # selection
        for corner in corner_spheres:
            corner.disappear(alpha=fade, begin_time=t0, transition_time=0.5)
        ibpy.change_alpha_of_material(cell_material, from_value=1, to_value=fade, begin_time=t0, transition_time=0.5)
        ibpy.change_alpha_of_material(grid_material, from_value=fade, to_value=1, begin_time=t0,
                                      transition_time=0.5)
        selection_text = SimpleTexBObject(r"\text{selection:}", color='example')
        t0 = 0.5 + display.write_text_in(selection_text, line=16, indent=0.35, begin_time=t0, transition_time=0.5)
        colors = flatten([['example'] * 4, ['gray_5'] * 4, ['example'], ['drawing'] * 5, ['custom1'] * 2, ['drawing'],
                          ['gray_5'] * 2, ['drawing'] * 5, ['example']])
        selection_formula = SimpleTexBObject(
            r"P^\star=\left\{ \vec{p}\in P\,\,| \,\,d_\text{min} < \vec{n}\cdot \vec{p}\le d_\text{max}\right\}",
            color=colors)
        ibpy.change_alpha_of_material(icosphere_material, from_value=0.1, to_value=1, begin_time=t0, transition_time=1)
        t0 = 0.5 + display.write_text_in(selection_formula, line=17, indent=0.35, begin_time=t0, transition_time=1)

        # projection
        ibpy.change_alpha_of_material(horizontal_material, from_value=0.1, to_value=1, begin_time=t0,
                                      transition_time=0.5)
        ibpy.change_alpha_of_material(vertical_material, from_value=0.1, to_value=1, begin_time=t0, transition_time=0.5)
        ibpy.change_alpha_of_material(grid_material, from_value=1, to_value=fade, begin_time=t0, transition_time=0.5)
        ibpy.change_alpha_of_material(max_material, from_value=1, to_value=fade, begin_time=t0, transition_time=0.5)
        ibpy.change_alpha_of_material(min_material, from_value=1, to_value=fade, begin_time=t0, transition_time=0.5)
        projection_text = SimpleTexBObject(r"\text{projection:}")
        t0 = 0.5 + display.write_text_in(projection_text, line=19, begin_time=t0, transition_time=0.5, indent=0.35)

        colors = flatten(
            [['text'] * 4, ['example'] * 2, ['text'] * 2, ['custom1'] * 2, ['text'], ['gray_5'] * 2, ['text'],
             ['custom1'] * 2, ['text'], ['example'] * 5, ['text']])
        projection_formula = SimpleTexBObject(
            r"P_\parallel = \left\{\vec{p}-(\vec{n}\cdot \vec{p})\vec{n}\,\,|\,\, \vec{p} \in P^\star \right\}",
            color=colors)
        display.write_text_in(projection_formula, line=20, begin_time=t0, transition_time=1, indent=0.35)
        t0 = 0.5 + ibpy.change_default_value(projection.outputs[0], from_value=0, to_value=1, begin_time=t0,
                                             transition_time=2)

        coords.disappear(begin_time=t0, transition_time=0.5)
        arrow3.disappear(begin_time=t0, transition_time=0.5)

        for mat in set(materials) - {horizontal_material, vertical_material, icosphere_material}:
            ibpy.change_alpha_of_material(mat, from_value=fade, to_value=0, begin_time=t0, transition_time=0.5)

        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=13 * pi / 5, to_value=8 * pi / 5,
                                             begin_time=t0,
                                             transition_time=23)

        # display.disappear(begin_time=t0-5,transition_time=1)

        self.t0 = t0

    def projection_2d_intro(self):

        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -20, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        coords = CoordinateSystem(dim=2, lengths=[20, 10], domains=[[-10, 10], [-5, 5]],
                                  all_n_tics=[10, 5],
                                  all_tic_lablls=[np.arange(-10, 10.1, 2), np.arange(-5, 5.1, 2)],
                                  label_digits=[0, 0],
                                  radii=[0.015, 0.015, 0.015],
                                  include_zeros=[False] * 2
                                  )
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=3)

        cube = Cube(rotation_euler=[pi / 2, 0, 0])
        geo_nodes = Penrose2DIntro()
        cube.add_mesh_modifier(type='NODES', node_group=geo_nodes.get_node_tree())

        radius = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "GrowGrid")
        t0 = 0.5 + ibpy.change_default_value(radius.outputs[0], from_value=-1, to_value=20, begin_time=t0,
                                             transition_time=5)

        thickness = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "Thickness")
        length = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "GrowLine")
        angle = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "Angle")
        projection = ibpy.get_geometry_node_from_modifier(geo_nodes.get_node_tree(), "Projector")

        t0 = ibpy.change_default_value(thickness.outputs[0], from_value=0, to_value=0.02, begin_time=t0,
                                       transition_time=1)
        coords.disappear(alpha=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.change_default_value(thickness.outputs[0], from_value=0.02, to_value=0.01, begin_time=t0,
                                             transition_time=1)

        t0 = 0.5 + ibpy.change_default_value(length.outputs[0], from_value=0, to_value=20, begin_time=t0,
                                             transition_time=5)

        coords.disappear(begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=pi / 2, to_value=4 * pi / 5, begin_time=t0,
                                             transition_time=5)
        t0 = 0.5 + ibpy.change_default_value(projection.outputs[0], from_value=0, to_value=1, begin_time=t0,
                                             transition_time=3)

        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=4 * pi / 5, to_value=3 * pi / 4,
                                             begin_time=t0, transition_time=0.5)
        t0 = 0.5 + ibpy.change_default_value(angle.outputs[0], from_value=3 * pi / 4, to_value=pi / 2, begin_time=t0,
                                             transition_time=3.5)

        t0 = 0.5 + ibpy.change_default_value(projection.outputs[0], from_value=1, to_value=0, begin_time=t0,
                                             transition_time=3)
        t0 = 0.5 + ibpy.change_default_value(thickness.outputs[0], from_value=0.01, to_value=0, begin_time=t0,
                                             transition_time=1)

        self.t0 = t0

    def projection_2d(self):

        t0 = 0

        text = 'gray_1'
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[0, -20, 0])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        cube = Cube(rotation_euler=[pi / 2, 0, 0])
        geo_nodes = Penrose2D()
        cube.add_mesh_modifier(type='NODES', node_group=geo_nodes.get_node_tree())

        self.t0 = t0

    def three_d_rotations(self):

        t0 = 0

        text = 'gray_1'
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[-15/180*pi, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[-2, -8, 0])
        empty = EmptyCube(location=Vector((-2, 0, 0)))
        ibpy.set_camera_view_to(empty)

        plane = Plane(resolution=[10, 10], uniformization=False)
        plane.move(direction=[0, 0, -0.5], begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)

        z3_nodes = create_z3(n=5, base="STANDARD",
                             final_rotation=[pi / 2, 0, 0],
                             final_scale=[0.25] * 3,
                             final_translation=[-4.5, 0, 0.05])
        plane.add_mesh_modifier(type='NODES', node_group=z3_nodes)
        set_material(plane, "Penrose", slot=0)

        ts = [ibpy.get_geometry_node_from_modifier(z3_nodes, 't_' + str(i)) for i in range(3)]

        plane.appear(begin_time=t0, transition_time=0)

        #rotations computed with mathematica : EulerAngles3D.nb
        values = [0, -np.arctan(r2),  -pi/4]
        values1 = [-pi/2,-pi,0]
        values2 = [-1.54,-2.35,-2.267] #angle computed in GeneralizedEulerAngles3D.nb


        view_rotation=ibpy.get_geometry_node_from_modifier(z3_nodes,label='ViewRotation')

        growth_3d = ibpy.get_geometry_node_from_modifier(z3_nodes, label='GrowthScale3D')
        t0 = 0.5 + ibpy.change_default_value(growth_3d.outputs[0], to_value=10,
                                              from_value=0,
                                              begin_time=t0, transition_time=2)

        plane_scale = ibpy.get_geometry_node_from_modifier(z3_nodes,label="PlaneScale")
        t0 = 0.5 + ibpy.change_default_value(plane_scale.outputs[0], to_value=10,
                                             from_value=0,
                                             begin_time=t0, transition_time=2)

        ortho_line_scale=ibpy.get_geometry_node_from_modifier(z3_nodes,label="OrthoLineScale")
        t0 =0.5+ibpy.change_default_value(ortho_line_scale.outputs[0],to_value=1,from_value=0,
                                          begin_time=t0,transition_time=1)

        voronoi_scale = ibpy.get_geometry_node_from_modifier(z3_nodes, label="VoronoiScale")
        t0 = 0.5 + ibpy.change_default_value(voronoi_scale.outputs[0], to_value=1, from_value=0,
                                             begin_time=t0, transition_time=1)

        convex_hull_scale = ibpy.get_geometry_node_from_modifier(z3_nodes, label="ConvexHullScale")
        t0 = 0.5 + ibpy.change_default_value(convex_hull_scale.outputs[0], to_value=1, from_value=0,
                                             begin_time=t0, transition_time=1)
        #
        # t0 =0.5+ibpy.change_default_value(ortho_line_scale.outputs[0],from_value=1,to_value=0,
        #                                   begin_time=t0,transition_time=1)

        selected_scale=ibpy.get_geometry_node_from_modifier(z3_nodes,label='SelectedScale')
        ortho_projection_scale = ibpy.get_geometry_node_from_modifier(z3_nodes, label='OrthoProjectionScale')
        change_default_value(ortho_projection_scale.outputs[0], from_value=0, to_value=1, begin_time=t0,
                                        transition_time=1)
        t0 = 0.5+ibpy.change_default_value(selected_scale.outputs[0],to_value=0.15,from_value=0,
                                           begin_time=t0,transition_time=1)

        rot1 = Quaternion([1,0,0],pi/2)
        t0 =0.5 +ibpy.change_default_vector(view_rotation, to_value=rot1.to_euler(),
                                              from_value=Vector(),
                                              begin_time=t0, transition_time=5)

        # show faces
        face_scale = ibpy.get_geometry_node_from_modifier(z3_nodes, label='FaceScale')
        change_default_value(face_scale.outputs[0], from_value=0, to_value=0.95, begin_time=t0,
                                        transition_time=2)
        projection_scale = ibpy.get_geometry_node_from_modifier(z3_nodes, label='ScaleProjection')
        t0 = 0.5 + change_default_value(projection_scale.outputs[0], from_value=0, to_value=0.95, begin_time=t0,
                                        transition_time=2)


        rot2 = Quaternion([1,0,0],pi/4)
        t0 = 0.5 + ibpy.change_default_vector(view_rotation, to_value=rot2.to_euler(),
                                              from_value=rot1.to_euler(),
                                              begin_time=t0, transition_time=2.5)

        rotation_panel_switch = ibpy.get_geometry_node_from_modifier(z3_nodes, label='RotationPanelSwitch')
        t0 = 0.5 + ibpy.change_default_boolean(rotation_panel_switch, from_value=True,to_value=False, begin_time=t0)

        # first turn
        quat1 = Quaternion([0,0,1],-values[2])
        rot3 = rot2@quat1

        ibpy.change_default_vector(view_rotation, to_value=rot3.to_euler(),from_value=rot2.to_euler(),
                                   begin_time=t0, transition_time=5)
        t0 = 0.5 + change_default_value(ts[2].outputs[0], from_value=0, to_value=values[2], begin_time=t0,transition_time=5)

        rot4 = rot1@quat1
        t0 = 0.5 +change_default_vector(view_rotation, to_value=rot4.to_euler(),from_value=rot3.to_euler(),
                                   begin_time=t0, transition_time=2.5)

        rot5 = quat1
        t0 = 0.5 + change_default_vector(view_rotation, to_value=rot5.to_euler(), from_value=rot4.to_euler(),
                                         begin_time=t0, transition_time=5)

        quat2 = Quaternion([0, 1, 0], -values[1])
        rot6 = quat2@quat1

        change_default_vector(view_rotation, to_value=rot6.to_euler(), from_value=rot5.to_euler(),
                                         begin_time=t0, transition_time=5)
        t0 = 0.5 + change_default_value(ts[1].outputs[0], from_value=0, to_value=values[1], begin_time=t0,transition_time=5)

        final = [1.529,-0.805,0.959]
        t0 = 0.5+ change_default_vector(view_rotation, to_value=Vector(final), from_value=rot6.to_euler(),
                              begin_time=t0, transition_time=2.5)

        self.t0 = t0

    def five_d_rotations(self):

        t0 = 0

        text = 'gray_1'
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64)

        ibpy.set_camera_location(location=[-2, -9, 0])
        empty = EmptyCube(location=Vector((-2, 0, 0)))
        ibpy.set_camera_view_to(empty)

        plane = Plane(resolution=[10, 10], uniformization=False)
        plane.move(direction=[0, 0, -0.5], begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)

        z5_nodes = create_z5(n=5, base="STANDARD",
                             final_rotation=[pi / 2, 0, 0],
                             final_scale=[0.25] * 3,
                             final_translation=[-4, 0, 0])
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)

        ts = [ibpy.get_geometry_node_from_modifier(z5_nodes, 't_' + str(i)) for i in range(10)]

        plane.appear(begin_time=t0, transition_time=0)

        # computations of the angles are performed in GeneralizedEulerAngles5D.nb

        t0 = 0.5 + change_default_value(ts[0].outputs[0], from_value=0, to_value=-1.01722, begin_time=t0,
                                        transition_time=10)
        t0 = 0.5 + change_default_value(ts[1].outputs[0], from_value=0, to_value=-2.43672, begin_time=t0,
                                        transition_time=10)
        t0 = 0.5 + change_default_value(ts[2].outputs[0], from_value=0, to_value=-2.7607, begin_time=t0,
                                        transition_time=10)
        t0 = 0.5 + change_default_value(ts[4].outputs[0], from_value=0, to_value=0.913563, begin_time=t0,
                                        transition_time=10)
        t0 = 0.5 + change_default_value(ts[5].outputs[0], from_value=0, to_value=-0.791708, begin_time=t0,
                                        transition_time=10)
        t0 = 0.5 + change_default_value(ts[6].outputs[0], from_value=0, to_value=2.45687, begin_time=t0,
                                        transition_time=10)
        t0 = 0.5 + change_default_value(ts[7].outputs[0], from_value=0, to_value=0.397177, begin_time=t0,
                                        transition_time=1)
        t0 = 0.5 + change_default_value(ts[8].outputs[0], from_value=0, to_value=-0.955317, begin_time=t0,
                                        transition_time=1)
        t0 = 0.5 + change_default_value(ts[9].outputs[0], from_value=0, to_value=-pi, begin_time=t0, transition_time=1)

        self.t0 = t0

    def dynamical_convex_hull(self):
        t0 = 0

        text = 'gray_1'
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[0, -20, 20])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        plane = Plane(resolution=[10, 10], uniformization=False)
        plane.move(direction=[0, 0, -0.7], begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)

        z5_nodes = create_z5(n=5)
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)

        plane.appear(begin_time=t0, transition_time=0)

        self.t0 = t0

    def intro(self):
        t0 = 0
        size = 5
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 no_transmission_ray='True',
                                 rotation_euler=[0, 0, 260 * pi / 180])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine=BLENDER_EEVEE,
                               taa_render_samples=1024  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(Vector())
        empty = EmptyCube(location=[0, 0, 0])

        camera_circle = BezierCircle(location=[0, 0, 6.3], radius=r2 * 13)

        ibpy.set_camera_view_to(empty)
        ibpy.set_camera_follow(camera_circle)

        coords = CoordinateSystem(dim=3, lengths=[10, 10, 10], domains=[[-5, 5], [-5, 5], [-5, 5]],
                                  all_n_tics=[10, 10, 10],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-5, 5.1, 1), np.arange(-5, 5.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015],
                                  include_zeros=[False] * 3
                                  )
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        lattice_node = penrose_3D_analog(size=50, radius=0.05, color='plastic_example')
        lattice = Cube()
        # coords.add_object(lattice)

        lattice.add_mesh_modifier(type='NODES', node_group=lattice_node)

        x_max = get_geometry_node_from_modifier(lattice_node, label='xMax')
        x_min = get_geometry_node_from_modifier(lattice_node, label='xMin')
        y_max = get_geometry_node_from_modifier(lattice_node, label='yMax')
        y_min = get_geometry_node_from_modifier(lattice_node, label='yMin')
        z_min = get_geometry_node_from_modifier(lattice_node, label='zMin')
        z_max = get_geometry_node_from_modifier(lattice_node, label='zMax')

        max_material = ibpy.get_material_from_modifier(lattice_node, "MaxMaterial")
        min_material = ibpy.get_material_from_modifier(lattice_node, "MinMaterial")
        cube_material = ibpy.get_material_from_modifier(lattice_node, "CubeMaterial")
        icosphere_material = ibpy.get_material_from_modifier(lattice_node, "IcoSphereMaterial")
        materials = [max_material, min_material, cube_material]
        [ibpy.change_alpha_of_material(mat, from_value=0, to_value=0, begin_time=0, transition_time=0) for mat in
         materials]

        # trip to infinity
        r_slider = get_geometry_node_from_modifier(lattice_node, label='rMax')
        t0 = 0.5 + change_default_value(r_slider.outputs[0], from_value=0, to_value=50, begin_time=t0,
                                        transition_time=10)

        # restrict to box
        change_default_value(x_max.outputs[0], from_value=50, to_value=5, begin_time=t0, transition_time=5)
        change_default_value(x_min.outputs[0], from_value=-50, to_value=-5, begin_time=t0, transition_time=5)
        change_default_value(y_max.outputs[0], from_value=50, to_value=5, begin_time=t0, transition_time=5)
        change_default_value(y_min.outputs[0], from_value=-50, to_value=-5, begin_time=t0, transition_time=5)
        change_default_value(z_min.outputs[0], from_value=-50, to_value=-5, begin_time=t0, transition_time=5)
        t0 = 0.5 + change_default_value(z_max.outputs[0], from_value=50, to_value=5, begin_time=t0, transition_time=5)

        camera_circle.move_to(target_location=[0, 0, 1], begin_time=0, transition_time=t0 + 5)
        t0 = -5 + ibpy.camera_follow(camera_circle, initial_value=0, final_value=-0.5, begin_time=0,
                                     transition_time=t0 + 5)

        # show voronoi cell
        t0 = 0.5 + ibpy.change_alpha_of_material(cube_material, from_value=0, to_value=1, begin_time=t0,
                                                 transition_time=1)
        # show voronoi zone
        ibpy.change_alpha_of_material(min_material, from_value=0, to_value=1, begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.change_alpha_of_material(max_material, from_value=0, to_value=1, begin_time=t0,
                                                 transition_time=1)

        coords.disappear(begin_time=t0, transition_time=1)
        # rotate plane and show selected grid points
        rotatePlane = get_geometry_node_from_modifier(lattice_node, label='rotatePlane')

        ibpy.change_alpha_of_material(icosphere_material, from_value=0, to_value=1, begin_time=t0, transition_time=2)

        t0 = change_default_vector(rotatePlane, from_value=Vector([0, 0, 0]), to_value=Vector([0, pi, 0]),
                                   begin_time=t0, transition_time=10)

        t0 = 0.5 + change_default_vector(rotatePlane, from_value=Vector([0, pi, 0]),
                                         to_value=Vector([0, np.pi / 2 - np.arctan(1 / r2), np.pi / 4]),
                                         begin_time=t0, transition_time=9)

        # make cube disappear
        ibpy.camera_follow(camera_circle, initial_value=-0.5, final_value=-0.375, begin_time=t0,
                           transition_time=5)
        cube_scale = ibpy.get_geometry_node_from_modifier(lattice_node, label='CubeScale').outputs[0]
        t0 = 0.5 + ibpy.change_default_value(cube_scale, from_value=1, to_value=0, begin_time=t0, transition_time=1)

        # show faces
        ranges = [
            get_geometry_node_from_modifier(lattice_node, label='xyRange').outputs[0],
            get_geometry_node_from_modifier(lattice_node, label='xzRange').outputs[0],
            get_geometry_node_from_modifier(lattice_node, label='yzRange').outputs[0],
        ]
        for r in ranges:
            t0 = 0.5 + change_default_value(r, from_value=0, to_value=6, begin_time=t0, transition_time=1)

        t0 = change_default_vector(rotatePlane, from_value=Vector([0, np.pi / 2 - np.arctan(1 / r2), np.pi / 4]),
                                   to_value=Vector(), begin_time=t0, transition_time=10)
        ibpy.camera_follow(camera_circle, initial_value=-0.375, final_value=-0.625, begin_time=t0, transition_time=10)
        t0 = 0.5 + change_default_vector(rotatePlane, from_value=Vector(),
                                         to_value=Vector([0, np.pi / 2 - np.arctan(1 / r2), np.pi / 4]),
                                         begin_time=t0, transition_time=10)

        # show plane
        showPlane = get_geometry_node_from_modifier(lattice_node, label='showPlane')
        plane_material = get_geometry_node_from_modifier(lattice_node, label='planeMaterial').inputs[
            'Material'].default_value

        change_default_boolean(showPlane, from_value=False, to_value=True, begin_time=t0, transition_time=0.1)
        ibpy.change_alpha_of_material(min_material, from_value=1, to_value=0, begin_time=t0, transition_time=1)
        ibpy.change_alpha_of_material(max_material, from_value=1, to_value=0, begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.change_alpha_of_material(plane_material, from_value=0, to_value=1, begin_time=t0,
                                                 transition_time=1)

        # projection
        coords.disappear(begin_time=t0 - 1, transition_time=1)
        cubie_scale = get_geometry_node_from_modifier(lattice_node, label='CubieScale').outputs[0]
        projection = get_geometry_node_from_modifier(lattice_node, label='Projection').outputs[0]
        change_default_value(cubie_scale, from_value=0.05, to_value=0, begin_time=t0, transition_time=0.3)

        t0 = 0.5 + change_default_value(projection, from_value=0, to_value=1, begin_time=t0, transition_time=3)

        t0 = 0.5 + ibpy.camera_follow(camera_circle, initial_value=-0.625, final_value=-0.375, begin_time=t0,
                                      transition_time=5)
        ibpy.change_alpha_of_material(plane_material, from_value=1, to_value=0.25, begin_time=t0, transition_time=0.1)
        t0 = change_default_vector(rotatePlane, from_value=Vector([0, np.pi / 2 - np.arctan(1 / r2), np.pi / 4]),
                                   to_value=Vector(), begin_time=t0, transition_time=3)
        t0 = 0.5 + change_default_vector(rotatePlane, from_value=Vector(),
                                         to_value=Vector([0, np.pi / 2 - np.arctan(1 / r2), np.pi / 4]),
                                         begin_time=t0, transition_time=3)

        # remove points
        change_default_value(x_max.outputs[0], from_value=5, to_value=-5, begin_time=t0, transition_time=1)
        ibpy.change_alpha_of_material(plane_material, from_value=0.25, to_value=0, begin_time=t0, transition_time=0.1)

        ibpy.camera_zoom(lens=85, begin_time=t0, transition_time=5)
        camera_circle.move_to(target_location=[0, 0, 13.2], begin_time=t0, transition_time=8)

        t0 += 0.5 + 3
        self.t0 = t0

    def intro_old(self):
        t0 = 0
        # sphere = Sphere(0.1, location=coords.coords2location([-size, -size, -size]), color='example', name="vertex")
        # sphere.add_mesh_modifier(type='ARRAY', count=2 * size + 1, relative_offset_displace=[0, 5, 0])
        # sphere.add_mesh_modifier(type='ARRAY', count=2 * size + 1, relative_offset_displace=[0, 0, 5])
        # sphere.add_mesh_modifier(type='ARRAY', count=2 * size + 1, relative_offset_displace=[5, 0, 0])
        # ibpy.apply_modifiers(sphere)
        # ibpy.set_origin(sphere, type='ORIGIN_CENTER_OF_VOLUME')
        # ibpy.separate(sphere, type='LOOSE')
        # coords.add_object(sphere)
        #
        # positions = []
        # for y in range(-size, size + 1):
        #     for z in range(-size, size + 1):
        #         for x in range(-size, size + 1):
        #             positions.append(Vector((x, y, z)))  # use tuples to be able to construct sets
        #
        # num_spheres = (2 * size + 1) ** 3
        # names = ['vertex.' + str(i).zfill(3) for i in range(1, np.minimum(1000, num_spheres))]
        # names.insert(0, 'vertex')
        # if num_spheres > 1000:
        #     names2 = ['vertex.' + str(i) for i in range(1000, num_spheres)]
        #     names = names + names2
        # spheres = [BObject.from_name(name, scale=[1, 1, 1], color='example') for name in names]
        # for sphere in spheres:
        #     sphere.appear(begin_time=t0, transition_time=0.5)
        #     coords.add_object(sphere)
        #     t0 += 0.01
        #
        # t0 += 0.5
        # t0 = 0.5 + coords.disappear_axes(begin_time=t0,
        #                                  transition_time=1)  # make it disappear but  remain present for turning
        #
        # r2 = np.sqrt(2)
        # r34 = np.sqrt(3 / 4)
        # plane = Plane(rotation_euler=[0, np.pi / 2 - np.arctan(1 / r2), np.pi / 4], scale=[5, 5, 5], transmission=0.9,
        #               color="fake_glass_joker")
        # t0 = 0.5 + plane.appear(begin_time=t0, transition_time=1)
        # normal = Vector([1, 1, 1])
        # normal.normalize()
        # coords.add_object(plane)
        #
        # # delete all far away points
        #
        # sel_spheres = []
        # sel_positions = []
        # for sphere, pos in zip(spheres, positions):
        #     if np.abs(normal.dot(pos)) > r34:
        #         sphere.disappear(begin_time=t0, transition_time=0.5)
        #         t0 += 0.01
        #     else:
        #         sel_spheres.append(sphere)
        #         sel_positions.append(pos)
        #
        # t0 += 1
        #
        # # collect relevant faces
        # faces = [(0, 1), (0, 2), (1, 2)]
        # units = [Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])]
        #
        # positions_set = [tuple(pos) for pos in sel_positions]
        #
        # colors = ['plastic_drawing', 'plastic_important', 'plastic_custom2']
        # count = 0
        # col = 0
        # for face, color in zip(faces, colors):
        #     base_points = []
        #     for pos in sel_positions:
        #         v0 = pos
        #         v1 = pos + units[face[0]]
        #         v2 = pos + units[face[0]] + units[face[1]]
        #         v3 = pos + units[face[1]]
        #         if {tuple(v0), tuple(v1), tuple(v2), tuple(v3)}.issubset(positions_set):
        #             base_points.append(v0)
        #             verts = [v0, v1, v2, v3]
        #             if col == 1:
        #                 verts = [v0, v3, v2, v1]
        #             poly = Polygon(vertices=verts, color=color, index=(col * 1000 + count))
        #             poly.appear(begin_time=t0, transition_time=0.5)
        #             coords.add_object(poly)
        #             count += 1
        #             t0 += 0.05
        #     col += 1
        #     t0 += 0.5
        #
        # [sphere.disappear(begin_time=t0, transition_time=0.5) for sphere in sel_spheres]
        #

        # ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=1)
        # plane.disappear(alpha=0.3, begin_time=t0 - 10, transition_time=10)
        # # empty.move(direction=[0,0,-1],begin_time=0,transition_time=60)

    def title(self):
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 rotation_euler=[np.pi / 2, np.pi, 0])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine=BLENDER_EEVEE,
                               taa_render_samples=256  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0.5)

        ibpy.set_camera_location(location=[0, -13, 0])
        camera_empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(camera_empty)

        title = SimpleTexBObject(r"\text{Penrose's Echos from Five Dimensions}", color='plastic_example',
                                 location=[-7, 0, 4], text_size='large')
        t0 = 2 + title.write(begin_time=t0, transition_time=1)

        subtitle = SimpleTexBObject(r"\text{featuring Dugan Hammock}", color='plastic_example', location=[7, 0, -4],
                                    aligned='right', text_size='large')
        t0 = 1 + subtitle.write(begin_time=t0, transition_time=1)

        self.t0 = t0

    def geo_penrose(self):
        """
        setup for a different version of penrose tilings directly from 5D
        it misses the implementation of the convex hull and the check whether a 5D point projection into the orthogonal direction
        is contained in the convex hull

        :return:
        """
        ibpy.set_hdri_background("forest", 'exr', background='gradient')
        t0 = ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, taa_render_samples=256)
        t0 = 0

        ibpy.set_camera_location(location=[0, -12, 5])
        camera_empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(camera_empty)

        floor = Plane(location=[0, 0, -0.1], scale=100, name='Floor', color='background')
        t0 = 0.5 + floor.appear(begin_time=0, transition_time=0)
        ibpy.make_rigid_body(floor, type='PASSIVE')

        # add plane, whose mesh vertices will be the centers of the tilings
        plane = Plane(resolution=[10, 10], uniformization=False)
        z5_nodes = create_z5(n=3)
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        t0 = 0.5 + plane.appear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def branding(self):
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 rotation_euler=[np.pi / 2, np.pi, 0])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine=BLENDER_EEVEE,
                               taa_render_samples=256  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0)

        ibpy.set_camera_location(location=[0, -12, 5])
        camera_empty = EmptyCube(location=[0, 0, 0])
        camera_angle = np.arctan2(12, 5)
        ibpy.set_camera_view_to(camera_empty)

        title = SimpleTexBObject(r'\text{Penrose Tilings}', location=[0, 0, 0], aligned='center',
                                 thickness=20, bevel=10, color='plastic_example', text_size='Huge',
                                 rotation_euler=[camera_angle, 0, 0], emission=0)
        title.write(begin_time=0, transition_time=0)

        # add plane, whose mesh vertices will be the centers of the tilings
        plane = Plane(resolution=[10, 10], uniformization=False)
        z5_nodes = de_bruijn(k=30, base_color='joker', tile_separation=0.05, radius=300, emission=0.3)
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)
        plane.add_mesh_modifier('SOLIDIFY', thickness=0.12, offset=0)
        #plane.add_mesh_modifier('BEVEL', amount=0.05, segments=2)

        title.move(direction=[0, 0, 0.75], begin_time=t0, transition_time=1)
        plane.appear(begin_time=t0, transition_time=0)

        tile_size = get_geometry_node_from_modifier(z5_nodes, 'tile_size')
        change_default_value(tile_size.inputs['Scale'], from_value=0, to_value=0.95, begin_time=0, transition_time=2)
        stretcher = get_geometry_node_from_modifier(z5_nodes, 'stretcher')
        t0 = 0.5 + change_default_value(stretcher.inputs['Scale'], from_value=10, to_value=1, begin_time=0,
                                        transition_time=8.5)

        title.disappear(begin_time=t0, transition_time=3)
        t0 = ibpy.camera_move(shift=[0, 12, 8], begin_time=t0, transition_time=3)

        ibpy.make_image_material(src='logo.png', name='PenroseImage')

        mixer = ibpy.get_nodes_of_material(plane, name_part='Mix')[0]
        change_default_value(mixer.inputs[0], from_value=0, to_value=1, begin_time=t0, transition_time=5)

        t0 = ibpy.camera_move(shift=[0, 0, 217], begin_time=t0, transition_time=10)
        self.t0 = t0

    def hexagons(self):
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 rotation_euler=[np.pi / 2, np.pi, 0])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine=BLENDER_EEVEE,
                               taa_render_samples=256  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0)

        ibpy.set_camera_location(location=[0, -12, 5])
        camera_empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(camera_empty)

        floor = Plane(location=[0, 0, -0.1], scale=100, name='Floor', color='background')
        t0 = 0.5 + floor.appear(begin_time=0, transition_time=0)
        ibpy.make_rigid_body(floor, type='PASSIVE')

        # add plane, whose mesh vertices will be the centers of the tilings
        plane = Plane(resolution=[10, 10], uniformization=False)
        hexagon_tilings = create_hexagon_tilings(material='joker', level=3, emission=0.3)
        plane.add_mesh_modifier(type='NODES', node_group=hexagon_tilings)
        plane.add_mesh_modifier('BEVEL', amount=0.05, segments=2)
        t0 = 0.5 + plane.appear(begin_time=t0, transition_time=1)

        scale = get_geometry_node_from_modifier(hexagon_tilings, 'scale')
        t0 = 0.5 + change_default_value(scale.inputs['Scale'], from_value=3000, to_value=50, begin_time=0,
                                        transition_time=8.5)

        # ibpy.make_rigid_body(plane)

        self.t0 = t0

    def hexagons2(self):
        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 rotation_euler=[np.pi / 2, np.pi, 0])
        ibpy.set_render_engine(motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100,
                               engine=BLENDER_EEVEE,
                               taa_render_samples=256  # for good alpha display
                               )
        t0 = ibpy.set_hdri_strength(1, transition_time=0)

        ibpy.set_camera_location(location=[0, -12, 5])
        camera_angle = np.arctan2(12, 5)
        camera_empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(camera_empty)

        floor = Plane(location=[0, 0, -0.05], scale=100, name='Floor', color='background')
        t0 = floor.appear(begin_time=0, transition_time=0)
        ibpy.make_rigid_body(floor, type='PASSIVE', use_margin=True, collision_margin=0.001, friction=0.3)

        # add plane, whose mesh vertices will be the centers of the tilings
        plane = Plane(resolution=[10, 10], uniformization=False)
        hexagon_tilings = create_hexagon_tilings(material='joker', level=3, scale=50, emission=0.3)
        plane.add_mesh_modifier(type='NODES', node_group=hexagon_tilings)
        plane.add_mesh_modifier(type='BEVEL', amount=0.05, segments=2)
        simulation_start = t0
        t0 = plane.appear(begin_time=t0, transition_time=0)

        apply_modifier(plane, type='modifier_NODES')
        separate(plane, type='LOOSE')
        make_rigid_body(plane, all_similar_objects=True, use_margin=True,
                        collision_margin=0.001, friction=0.5)
        # ibpy.make_rigid_body(plane)

        title = SimpleTexBObject(r'\text{Penrose Tilings}', location=[0, 0, -1], aligned='center',
                                 thickness=20, bevel=10, color='plastic_example', text_size='Huge',
                                 rotation_euler=[camera_angle, 0, 0], emission=0)
        title.write(begin_time=0, transition_time=0)
        convert_to_mesh(title.letters)

        make_rigid_body(title.letters[0], all_similar_objects=True, type='ACTIVE', kinematic=True, friction=0)
        floor.disappear(begin_time=t0, transition_time=2)
        t0 = title.move(direction=[0, 0, 1], begin_time=t0, transition_time=2)
        t0 = 0.5 + title.rotate(rotation_euler=[pi / 2 * 0.95, 0, 2 * pi], begin_time=t0, transition_time=5)

        t0 = 0.5 + set_simulation(begin_time=simulation_start, transition_time=15)
        t0 = 0.5 + plane.disappear(begin_time=t0, transition_time=1, slot=1)
        self.t0 = t0

    def optimize_object_creation(self):
        t0 = 0
        # test state of the art
        start = time.time()

        # try a simple batch creation of a single mesh
        coords = [[-8 / 16.0, 0 / 16.0, 0], [-4 / 16.0, 7 / 16.0, 0], [4 / 16.0, 7 / 16.0, 0], [8 / 16.0, 0 / 16.0, 0],
                  [4 / 16.0, -7 / 16.0, 0], [-4 / 16.0, -7 / 16.0, 0]]
        faces = [[0, 5, 4, 3], [3, 2, 1, 0]]

        mesh = ibpy.create_mesh(coords, [], faces)

        # create locations
        locations = []
        x_factor = 24 / 16.0
        x_offset = 0.75
        y_factor = 14 / 16.0
        y_offset = 0.875 / 2
        for x_tiles in range(48):
            for y_tiles in range(80):
                locations.append(Vector([x_tiles * x_factor, y_tiles * y_factor, 0.0]))
                locations.append(Vector([x_tiles * x_factor + x_offset, y_tiles * y_factor + y_offset, 0.0]))

        l = len(locations)
        colors = flatten([['plastic_drawing'] * int(l / 2), ['plastic_important'] * int(l / 2 + 1)])
        shuffle(colors)
        bobjects = BObject.batch_create_from_single_mesh(mesh, name="hexagon", locations=locations, colors=colors,
                                                         solid=0.1, bevel=0.1)
        [bob.appear(begin_time=t0, transition_time=0.1) for bob in bobjects]
        end = time.time()
        print(str(end - start) + " seconds")

        self.t0 = t0

    def optimize_object_creation2(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr',
                                 simple=True,
                                 transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 0]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        rr5p = np.sqrt(5 + r5)
        rr5m = np.sqrt(5 - r5)

        # data
        u = 1 / 2 / r5 / r2 * Vector([1 + r5, 1 - r5, 1 - r5, 1 + r5, -4])
        v = 1 / 2 / r5 * Vector([rr5m, -rr5p, rr5p, -rr5m, 0])
        w3 = 1 / r5 * Vector([1, 1, 1, 1, 1])
        w1 = 1 / r2 / 2 / r5 * Vector([1 - r5, 1 + r5, 1 + r5, 1 - r5, -4])
        w2 = 1 / 2 / r5 * Vector([rr5p, rr5m, -rr5m, -rr5p, 0])

        # initial configuration of eigen vectors that generate the first penrose tiling
        rot_start = [u, v, w1, w2, w3]

        # create data (points and faces)
        size = 4
        points = tuples(range(-size, size + 1), 5)
        units = [Vector(v) for v in np.identity(5)]

        unit_cube = tuples({-0.5, 0.5}, 5)

        orientations = face_directions(5)
        faces = {orientation: {point + orientation:
            plane_for_center_and_directions(
                point, [units[orientation[0]], units[orientation[1]]])
            for point in points} for orientation in orientations}

        colors = ['drawing', 'important', 'custom1', 'text', 'custom2', 'joker', 'custom3', 'custom4', 'example',
                  'gray_4']
        colors = ['plastic_' + col for col in colors]

        steps = 10
        ds = 1 / (steps - 1)
        dt = 0.25

        face_map = {}  # this map keeps track of all the faces that have appeared already
        visibility_dict = {}

        for i in range(steps):
            s = i * ds
            geometry = interpol2(s).dot(rot_start)
            ortho = np.array(geometry[2:])
            para = geometry[0:2]
            para = np.array([para[0], [0, 0, 0, 0, 0], para[1]])  # add zero line for 3d representation of 2d plane
            p_unit_cube = [ortho @ Vector(tup) for tup in unit_cube]
            hull = ConvexHull(p_unit_cube)
            counter = 0
            vertices_list = []
            key_list = []
            color_list = []

            start = time.time()
            for dict_key, face_dict in faces.items():
                color = colors[counter]
                # print("Investigating in step ",i," the faces ",dict_key," with a dictionary length of ",len(face_dict)," for faces iwith the color ",color)
                for key, value in face_dict.items():
                    if all(hull.is_inside(ortho.dot(p)) for p in value):
                        if key in face_map:
                            # update existing face
                            if not visibility_dict[key]:
                                face_map[key].toggle_hide(begin_time=t0)
                                face_map[key].change_alpha(alpha=1, begin_time=t0, transition_time=dt)
                                face_map[key].move_fast(direction=[0, 0.02, 0], begin_time=t0,
                                                        transition_time=0)  # push up from behind to lift the degeneracy of overlapping faces
                                face_map[key].move_fast(direction=[0, -0.01, 0], begin_time=t0 + 2 / FRAME_RATE,
                                                        transition_time=dt)
                                visibility_dict[key] = True
                                print("reappearing of face", key)
                            face_map[key].morph_to(projector=lambda v: para.dot(v), begin_time=t0, transition_time=dt)
                        else:
                            # collect data and postpone creation of the polygons until the full batch is collected
                            vertices_list.append(value)
                            key_list.append(key)
                            color_list.append(color)
                    else:
                        if key in visibility_dict:
                            if visibility_dict[key]:
                                face_map[key].move_fast(direction=[0, -0.01, 0], begin_time=t0, transition_time=dt)
                                face_map[key].change_alpha(alpha=0.01, begin_time=t0, transition_time=dt)
                                face_map[key].toggle_hide(
                                    begin_time=t0 + dt)  # hide plane that is not visible any longer
                                visibility_dict[key] = False
                                # print("hide face ",key)

                counter += 1
            # create new faces of the batch
            if len(vertices_list) > 0:
                polygons = Polygon.batch_create(name="PolyBatch" + str(i), vertices_list=vertices_list,
                                                colors=color_list,
                                                initial_function=lambda v: para.dot(v), solid=0.09, bevel=0.025,
                                                offset=0, link=True)

                for key, polygon in zip(key_list, polygons):
                    polygon.move_fast(direction=[0, 0.01, 0], begin_time=t0,
                                      transition_time=0)  # push up from behind
                    polygon.move_fast(direction=[0, -0.01, 0], begin_time=t0 + 2 / FRAME_RATE, transition_time=dt)
                    face_map[key] = polygon
                    polygon.appear(begin_time=t0 + 3 / FRAME_RATE, transition_time=dt, linked=True)
                    visibility_dict[key] = True
            t0 += dt
            end = time.time()
            print("at ", i, (end - start), " seconds.")
        self.t0 = t0

    def penrose_merger2(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 0]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        r2 = np.sqrt(2)
        r3 = np.sqrt(3)
        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        rr5p = np.sqrt(5 + r5)
        rr5m = np.sqrt(5 - r5)

        # data
        u = 1 / 2 / r5 / r2 * Vector([1 + r5, 1 - r5, 1 - r5, 1 + r5, -4])
        v = 1 / 2 / r5 * Vector([rr5m, -rr5p, rr5p, -rr5m, 0])
        w3 = 1 / r5 * Vector([1, 1, 1, 1, 1])
        w1 = 1 / r2 / 2 / r5 * Vector([1 - r5, 1 + r5, 1 + r5, 1 - r5, -4])
        w2 = 1 / 2 / r5 * Vector([rr5p, rr5m, -rr5m, -rr5p, 0])

        # initial configuration of eigen vectors that generate the first penrose tiling
        rot_start = [u, v, w1, w2, w3]

        # create data (points and faces)
        size = 4
        points = tuples(range(-size, size + 1), 5)
        units = [Vector(v) for v in np.identity(5)]

        unit_cube = tuples({-0.5, 0.5}, 5)

        orientations = face_directions(5)
        faces = {orientation: {point + orientation:
            plane_for_center_and_directions(
                point, [units[orientation[0]], units[orientation[1]]])
            for point in points} for orientation in orientations}

        colors = ['drawing', 'important', 'custom1', 'text', 'custom2', 'joker', 'custom3', 'custom4', 'example',
                  'gray_4']
        colors = ['plastic_' + col for col in colors]

        steps = 100
        ds = 1 / (steps - 1)
        dt = 0.25

        face_map = {}  # this map keeps track of all the faces that have appeared already

        face_counter = 0
        visibility_dict = {}

        for i in range(steps):
            s = i * ds
            geometry = interpol2(s).dot(rot_start)
            ortho = np.array(geometry[2:])
            para = geometry[0:2]
            para = np.array([para[0], [0, 0, 0, 0, 0], para[1]])  # add zero line for 3d representation of 2d plane
            p_unit_cube = [ortho @ Vector(tup) for tup in unit_cube]
            hull = ConvexHull(p_unit_cube)

            for face_dict, color in zip(faces.values(), colors):
                for key, value in face_dict.items():
                    if all(hull.is_inside(ortho.dot(p)) for p in value):
                        if key in face_map:
                            # update existing face
                            if not visibility_dict[key]:
                                face_map[key].toggle_hide(begin_time=t0)
                                face_map[key].change_alpha(alpha=1, begin_time=t0, transition_time=dt)
                                polygon.move(direction=[0, 0.02, 0], begin_time=t0,
                                             transition_time=0)  # push up from behind to lift the degeneracy of overlapping faces
                                polygon.move(direction=[0, -0.01, 0], begin_time=t0 + 2 / FRAME_RATE,
                                             transition_time=dt)
                                visibility_dict[key] = True
                                print("reappearing of face", key)
                            face_map[key].morph_to(projector=lambda v: para.dot(v), begin_time=t0, transition_time=dt)
                        else:
                            # create new face
                            polygon = Polygon(vertices=value, initial_function=lambda v: para.dot(v), color=color,
                                              name="poly_" + str(face_counter), solid=0.09, bevel=0.025)
                            polygon.move(direction=[0, 0.01, 0], begin_time=t0,
                                         transition_time=0)  # push up from behind
                            polygon.move(direction=[0, -0.01, 0], begin_time=t0 + 2 / FRAME_RATE, transition_time=dt)
                            face_counter += 1
                            face_map[key] = polygon
                            polygon.appear(begin_time=t0 + 3 / FRAME_RATE, transition_time=dt)
                            # print("show face ",key)
                            visibility_dict[key] = True
                    else:
                        if key in visibility_dict:
                            if visibility_dict[key]:
                                face_map[key].move(direction=[0, -0.01, 0], begin_time=t0, transition_time=dt)
                                face_map[key].change_alpha(alpha=0.01, begin_time=t0, transition_time=dt)
                                face_map[key].toggle_hide(
                                    begin_time=t0 + dt)  # hide plane that is not visible any longer
                                visibility_dict[key] = False
                                # print("hide face ",key)
            print(i, ": ", face_counter)
            t0 += dt

        self.t0 = t0

    def penrose_merger(self):
        t0 = 0

        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 0]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        r2 = np.sqrt(2)
        r3 = np.sqrt(3)
        # initial configuration of eigen vectors that generates the three-dimensional picture
        rot_start = [[1 / r2, -1 / r2, 0, 0, 0],
                     [1 / r2 / r3, 1 / r2 / r3, -r2 / r3, 0, 0],
                     [1 / r3, 1 / r3, 1 / r3, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]]

        # create data (points and faces)
        size = 4
        points = tuples(range(-size, size + 1), 5)
        units = [Vector(v) for v in np.identity(5)]

        unit_cube = tuples({-0.5, 0.5}, 5)

        orientations = face_directions(5)
        faces = {orientation: {point + orientation:
            plane_for_center_and_directions(
                point, [units[orientation[0]], units[orientation[1]]])
            for point in points} for orientation in orientations}

        print(orientations)
        print(len(faces), len(faces[(0, 1)]))

        colors = ['drawing', 'important', 'custom1', 'text', 'custom2', 'joker', 'custom3', 'custom4', 'example',
                  'gray_4']
        colors = ['plastic_' + col for col in colors]

        steps = 100
        ds = 1 / (steps - 1)
        dt = 0.2

        face_map = {}  # this map keeps track of all the faces that have appeared already

        face_counter = 0
        visibility_dict = {}

        for i in range(steps):
            s = i * ds
            geometry = interpol(s).dot(rot_start)
            ortho = np.array(geometry[2:])
            para = geometry[0:2]
            para = np.array([para[0], [0, 0, 0, 0, 0], para[1]])  # add zero line for 3d representation of 2d plane
            p_unit_cube = [ortho @ Vector(tup) for tup in unit_cube]
            hull = ConvexHull(p_unit_cube)

            ## show convex hull for debugging
            # p_locations = hull.points
            # hull_polygons = [
            #     Polygon(vertices=[Vector(p_locations[i]) for i in face], color='plastic_text') for
            #     face in hull.faces]
            # [polygon.appear(alpha=0.25, begin_time=t0 + 0.1 * i, transition_time=0.5) for i, polygon in
            #  enumerate(hull_polygons)]

            for face_dict, color in zip(faces.values(), colors):
                for key, value in face_dict.items():
                    if all(hull.is_inside(ortho.dot(p)) for p in value):
                        if key in face_map:
                            # update existing face
                            if not visibility_dict[key]:
                                face_map[key].toggle_hide(begin_time=t0)
                                face_map[key].change_alpha(alpha=1, begin_time=t0, transition_time=dt)
                                polygon.move(direction=[0, 0.02, 0], begin_time=t0,
                                             transition_time=0)  # push up from behind to lift the degeneracy of overlapping faces
                                polygon.move(direction=[0, -0.01, 0], begin_time=t0 + 2 / FRAME_RATE,
                                             transition_time=dt)
                                visibility_dict[key] = True
                                print("reappearing of face", key)
                            face_map[key].morph_to(projector=lambda v: para.dot(v), begin_time=t0, transition_time=dt)
                        else:
                            # create new face
                            polygon = Polygon(vertices=value, initial_function=lambda v: para.dot(v), color=color,
                                              name="poly_" + str(face_counter), solid=0.09, bevel=0.025)
                            polygon.move(direction=[0, 0.01, 0], begin_time=t0,
                                         transition_time=0)  # push up from behind
                            polygon.move(direction=[0, -0.01, 0], begin_time=t0 + 2 / FRAME_RATE, transition_time=dt)
                            face_counter += 1
                            face_map[key] = polygon
                            polygon.appear(begin_time=t0 + 3 / FRAME_RATE, transition_time=dt)
                            # print("show face ",key)
                            visibility_dict[key] = True
                    else:
                        if key in visibility_dict:
                            if visibility_dict[key]:
                                face_map[key].move(direction=[0, -0.01, 0], begin_time=t0, transition_time=dt)
                                face_map[key].change_alpha(alpha=0.01, begin_time=t0, transition_time=dt)
                                face_map[key].toggle_hide(
                                    begin_time=t0 + dt)  # hide plane that is not visible any longer
                                visibility_dict[key] = False
                                # print("hide face ",key)
            print(i, ": ", face_counter)
            t0 += dt

        self.t0 = t0

    def maybe_penrose(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        display = Display(flat=True, number_of_lines=30, color='gray_2', location=[4.5, 0, 0], scales=[6.5, 6],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1)

        # repeat analysis from 3d
        # don't be a never-nester
        tupels = []
        for x1 in range(-2, 3):
            for x2 in range(-2, 3):
                for x3 in range(-2, 3):
                    for x4 in range(-2, 3):
                        for x5 in range(-2, 3):
                            tupels.append((x1, x2, x3, x4, x5))

        # now we have 3125 points

        # calculate projections points
        r3 = np.sqrt(3)
        r5 = np.sqrt(5)
        tupels2 = tuples([0.5, -0.5], 5)
        ortho = np.array(
            [[0.5 / r3, 0.5 / r3, 0.5 / r3, -r3 / 2, 0], [0.5 / r5, 0.5 / r5, 0.5 / r5, 0.5 / r5, -2 / r5],
             [1 / r5] * 5])
        r2 = np.sqrt(2)
        para = np.array([
            [1 / r2, -1 / r2, 0, 0, 0],
            [1 / r3, 1 / r3, -2 / r3, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        p_locations = [ortho @ Vector(tup) for tup in tupels2]
        hull = ConvexHull(p_locations)

        selected_points = [tup for tup in tupels if hull.is_inside(ortho @ Vector(tup))]

        print(len(selected_points), " of 3125 selected")

        # construct all faces
        units = [Vector(u) for u in unit_tuples(5)]
        directions = face_directions(5)

        faces = {}
        for dir in directions:
            d0 = dir[0]
            d1 = dir[1]
            for pos in tupels:
                if pos[d0] < 2 and pos[d1] < 2:
                    faces[(pos, dir)] = plane_for_center_and_directions(pos, [units[d0], units[d1]])

        colors = ['text', 'example', 'custom1', 'drawing', 'joker', 'custom2', 'custom3', 'custom4', 'gray2', 'gray4']
        colors = ['plastic_' + col for col in colors]

        i = 0
        selected_faces = {key: val for key, val in faces.items()
                          if all([hull.is_inside(ortho @ point) for point in val])}

        polygons = []

        for key, val in selected_faces.items():
            pos, dir = key
            color = colors[directions.index(dir)]
            polygons.append(Polygon(vertices=[para @ v for v in val], color=color))

        for i, polygon in enumerate(polygons):
            polygon.appear(begin_time=t0 + 0.1 * i, transition_time=0.5)

        t0 += len(polygons) * 0.1 + 1
        self.t0 = t0

    def penrose_tiling2(self):
        t0 = 0
        r3 = np.sqrt(3)

        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # projection plane

        plane = Plane(color='plastic_important')
        plane_center = Vector([6, 0, -1.5])
        plane.move_to(target_location=plane_center)
        plane.rescale(rescale=[5] * 3, begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                             camera_location[1]), 0, 0], begin_time=t0,
                     transition_time=0)
        t0 = 0.5 + plane.appear(alpha=0.5, begin_time=t0, transition_time=1)

        # orthogonal space
        ortho_origin = Vector([-6.5, 0, 0])
        coords3d = CoordinateSystem(dim=3, lengths=[10, 10, 10],
                                    domains=[[-2, 2], [-2, 2], [-2, 2]],
                                    all_n_tics=[4, 4, 4],
                                    all_tic_lablls=[np.arange(-2, 2.1, 1), np.arange(-2, 2.1, 1),
                                                    np.arange(-2, 2.1, 1)],
                                    label_digits=[0, 0, 0],
                                    radii=[0.03, 0.03, 0.03],
                                    labels=[r'\vec{w}_1', r'\vec{w}_2', r'\vec{w}_3'],
                                    colors=['plastic_drawing', 'plastic_drawing', 'plastic_drawing'],
                                    location_of_origin=ortho_origin,
                                    )

        t0 = 0.5 + coords3d.appear(begin_time=t0, transition_time=5)

        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        rr5p = np.sqrt(5 + r5)
        rr5m = np.sqrt(5 - r5)

        # data
        u = 1 / 2 / r5 / r2 * Vector([1 + r5, 1 - r5, 1 - r5, 1 + r5, -4])
        v = 1 / 2 / r5 * Vector([rr5m, -rr5p, rr5p, -rr5m, 0])
        w1 = 1 / r5 * Vector([1, 1, 1, 1, 1])
        w2 = 1 / r2 / 2 / r5 * Vector([1 - r5, 1 + r5, 1 + r5, 1 - r5, -4])
        w3 = 1 / 2 / r5 * Vector([rr5p, rr5m, -rr5m, -rr5p, 0])

        tuples2 = tuples([0.5, -0.5], 5)
        ortho = np.array([
            w1[0:5], w2[0:5], w3[0:5]
        ])
        para = np.array([u[0:5], [0] * 5, v[0:5]])

        p_locations = [ortho @ Vector(tup) for tup in tuples2]
        hull = ConvexHull(p_locations)

        hull_polygons = [
            Polygon(vertices=[coords3d.coords2location(p_locations[i]) for i in face], color='plastic_text') for
            face in hull.faces]
        [polygon.appear(alpha=0.25, begin_time=t0 + 0.1 * i, transition_time=0.5) for i, polygon in
         enumerate(hull_polygons)]
        coords3d.add_objects(hull_polygons)
        t0 += 0.1 * len(hull_polygons) + 1

        plane.disappear(begin_time=t0, transition_time=5)

        b = 5
        power = b ** 5
        count = 0
        selected_tuples = []
        for i in range(power):
            dt = 1 / (count / 10 + 1)

            digits = baserep(i, base=b, length=5)
            digits = [-int(np.floor(b / 2)) + d for d in digits]

            ortho_pos = ortho @ Vector(digits)

            if hull.is_inside(ortho_pos):
                blue = Sphere(0.25 * 0.4, location=coords3d.coords2location(ortho_pos), color='plastic_joker')
                blue.grow(begin_time=t0, transition_time=0.5)

                coords3d.add_object(blue)
                p = Vector(para @ Vector(digits))
                projection = Vector([p[0], p[1], p[2]])
                location = plane_center + projection

                red = Sphere(0.25 * 0.4, location=location, color='plastic_important')
                red.grow(begin_time=t0, transition_time=dt)
                selected_tuples.append(tuple(digits))

        t0 += 0.5
        coords3d.rotate(rotation_euler=[0, 0, 4 * np.pi], begin_time=t0, transition_time=80)

        selected_tuples = set(selected_tuples)
        colors = ['text', 'example', 'custom1', 'drawing', 'joker', 'custom2', 'custom3', 'custom4', 'gray_4', 'gray_6']
        colors = ['plastic_' + col for col in colors]

        locations = [[i, 0, 5] for i in range(7, -10, -4)]
        locations2 = [[-1, 0, j] for j in [1.5, -1.75, -5]]
        locations3 = [[i, 0, -5] for i in range(-5, -13, -4)]
        locations = locations + locations2 + locations3
        len(locations)
        units = unit_tuples(5)
        neg_units = negative_unit_tuples(5)

        coords = []
        for i, dir in enumerate(face_directions(5)):
            coord = CoordinateSystem(dim=2, lengths=[2, 2], domains=[[-1, 1], [-1, 1]],
                                     all_n_tics=[2, 2],
                                     all_tic_labels=[np.arange(-1, 1.1, 1), np.arange(-1, 1.1, 1)],
                                     label_digits=[0, 0],
                                     radii=[0.03, 0.03],
                                     labels=[r'x_' + str(dir[0] + 1), r'x_' + str(dir[1] + 1)],
                                     colors=[colors[i]] * 2,
                                     location_of_origin=locations[i]
                                     )

            coords.append(coord)
            plane_symbol = Plane(u=[0, 1], v=[0, 1], color=colors[i], rotation_euler=[np.pi / 2, 0, 0],
                                 solid=0.05, offset=0, bevel=0.05)
            coord.add_object(plane_symbol)
            coord.axes[1].axis_label.move(direction=[-0.2, 0, -0.6], begin_time=t0, transition_time=0)
            coord.axes[0].axis_label.move(direction=[0, 0, -0.45], begin_time=t0, transition_time=0)
            coord.axes[0].labels[1].move(direction=[0, 0, -0.1], begin_time=t0, transition_time=0)
            plane_symbol.appear(begin_time=t0 + 1, tt0=1)
            t0 = 0.5 + coord.appear(begin_time=t0, transition_time=5)

            for sel_point in selected_tuples:
                points = plane_for_center_and_directions(Vector(sel_point), {units[dir[0]], units[dir[1]]})
                pro_points = [Vector(ortho @ point) for point in points]

                if all([hull.is_inside(point) for point in pro_points]):
                    poly1 = Polygon(vertices=[coords3d.coords2location(point) for point in pro_points], color=colors[i])
                    coords3d.add_object(poly1)
                    poly2 = Polygon(vertices=[Vector(plane_center) + Vector(para @ point) for point in points],
                                    color=colors[i], solid=0.2, offset=0, bevel=0.05)

                    poly1.appear(alpha=0.25, begin_time=t0, transition_time=1)
                    poly2.appear(begin_time=t0, transition_time=1)
                    t0 += 0.1

            if i == 5:
                # hide convex hull
                [polygon.disappear(begin_time=t0, transition_time=10) for polygon in hull_polygons]

        for i, dir in enumerate(face_directions(5)):

            plane_symbol = Plane(u=[0, 1], v=[-1, 0], color=colors[i], rotation_euler=[np.pi / 2, 0, 0],
                                 solid=0.05, offset=0, bevel=0.05)
            coords[i].add_object(plane_symbol)
            plane_symbol.appear(begin_time=t0, tt0=1)

            for sel_point in selected_tuples:
                points = plane_for_center_and_directions(Vector(sel_point), {neg_units[dir[0]], units[dir[1]]})
                pro_points = [Vector(ortho @ point) for point in points]

                if all([hull.is_inside(point) for point in pro_points]):
                    poly1 = Polygon(vertices=[coords3d.coords2location(point) for point in pro_points], color=colors[i])
                    coords3d.add_object(poly1)
                    poly2 = Polygon(vertices=[Vector(plane_center) + Vector(para @ point) for point in points],
                                    color=colors[i], solid=0.2, offset=0, bevel=0.05)

                    poly1.appear(alpha=0.25, begin_time=t0, transition_time=1)
                    poly2.appear(alpha=1, begin_time=t0, transition_time=1)
                    t0 += 0.04

        for i, dir in enumerate(face_directions(5)):

            plane_symbol = Plane(u=[-1, 0], v=[0, 1], color=colors[i], rotation_euler=[np.pi / 2, 0, 0],
                                 solid=0.05, offset=0, bevel=0.05)
            coords[i].add_object(plane_symbol)
            plane_symbol.appear(begin_time=t0, tt0=1)

            for sel_point in selected_tuples:
                points = plane_for_center_and_directions(Vector(sel_point), {units[dir[0]], neg_units[dir[1]]})
                pro_points = [Vector(ortho @ point) for point in points]

                if all([hull.is_inside(point) for point in pro_points]):
                    poly1 = Polygon(vertices=[coords3d.coords2location(point) for point in pro_points], color=colors[i])
                    coords3d.add_object(poly1)
                    poly2 = Polygon(vertices=[Vector(plane_center) + Vector(para @ point) for point in points],
                                    color=colors[i], solid=0.2, offset=0, bevel=0.05)

                    poly1.appear(alpha=0.25, begin_time=t0, transition_time=1)
                    poly2.appear(alpha=1, begin_time=t0, transition_time=1)
                    t0 += 0.02
        t0 += 1
        self.t0 = t0

    def penrose_tiling(self):
        t0 = 0
        r3 = np.sqrt(3)

        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # running tuple

        display = Display(flat=True, scales=[3, 1], number_of_lines=3, background='gray_2', location=[7, 0, 5],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1)

        dim1 = DigitalRange(list(range(-3, 4)), digits=0, aligned="center", signed=True, color="example")
        dim2 = DigitalRange(list(range(-3, 4)), digits=0, aligned="center", signed=True, color="example")
        dim3 = DigitalRange(list(range(-3, 4)), digits=0, aligned="center", signed=True, color="example")
        dim4 = DigitalRange(list(range(-3, 4)), digits=0, aligned="center", signed=True, color="example")
        dim5 = DigitalRange(list(range(-3, 4)), digits=0, aligned="center", signed=True, color="example")

        digfigs = [dim1, dim2, dim3, dim4, dim5]

        display.add_text_in(dim1, scale=0.7, line=0, indent=2)
        display.add_text_in(dim2, scale=0.7, line=0, indent=3.5)
        display.add_text_in(dim3, scale=0.7, line=0, indent=5)
        display.add_text_in(dim4, scale=0.7, line=0, indent=6.5)
        display.add_text_in(dim5, scale=0.7, line=0, indent=8)

        count = 0
        comma_indents = [2.2, 3.7, 5.2, 6.7]
        for dim in digfigs:
            dim.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=t0, transition_time=0)
            if count == 0:
                t0 = display.write_text_in(SimpleTexBObject(r"(", scale=1.5), line=0, indent=1.3, begin_time=t0 - 0.1,
                                           transition_time=0.1)
            t0 = dim.show(0, begin_time=t0, transition_time=0.1)
            if count < 4:
                t0 = display.write_text_in(SimpleTexBObject(r",", scale=2, aligned="bottom"), line=0,
                                           indent=comma_indents[count],
                                           begin_time=t0, transition_time=0.1)
            else:
                t0 = 0.5 + display.write_text_in(SimpleTexBObject(r")", scale=1.5), line=0, indent=8.6, begin_time=t0,
                                                 transition_time=0.1)
            count += 1

        sphere_loc = [3.5, 0, 5]
        breader = Sphere(0.4, location=sphere_loc, color='plastic_example')
        t0 = 0.5 + breader.grow(begin_time=t0, transition_time=1)
        # projection plane

        plane = Plane(color='plastic_important')
        plane_center = Vector([6, 0, -1.5])
        plane.move_to(target_location=plane_center)
        plane.rescale(rescale=[5] * 3, begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                             camera_location[1]), 0, 0], begin_time=t0,
                     transition_time=0)
        t0 = 0.5 + plane.appear(alpha=0.5, begin_time=t0, transition_time=1)

        # orthogonal space
        ortho_origin = Vector([-6.5, 0, 0])
        coords = CoordinateSystem(dim=3, lengths=[10, 10, 10],
                                  domains=[[-2, 2], [-2, 2], [-2, 2]],
                                  all_n_tics=[4, 4, 4],
                                  all_tic_lablls=[np.arange(-2, 2.1, 1), np.arange(-2, 2.1, 1),
                                                  np.arange(-2, 2.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.03, 0.03, 0.03],
                                  labels=[r'\vec{w}_1', r'\vec{w}_2', r'\vec{w}_3'],
                                  colors=['plastic_drawing', 'plastic_drawing', 'plastic_drawing'],
                                  location_of_origin=ortho_origin,
                                  )

        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        rr5p = np.sqrt(5 + r5)
        rr5m = np.sqrt(5 - r5)

        # data
        u = 1 / 2 / r5 / r2 * Vector([1 + r5, 1 - r5, 1 - r5, 1 + r5, -4])
        v = 1 / 2 / r5 * Vector([rr5m, -rr5p, rr5p, -rr5m, 0])
        w1 = 1 / r5 * Vector([1, 1, 1, 1, 1])
        w2 = 1 / r2 / 2 / r5 * Vector([1 - r5, 1 + r5, 1 + r5, 1 - r5, -4])
        w3 = 1 / 2 / r5 * Vector([rr5p, rr5m, -rr5m, -rr5p, 0])

        tuples2 = tuples([0.5, -0.5], 5)
        ortho = np.array([
            w1[0:5], w2[0:5], w3[0:5]
        ])
        para = np.array([u[0:5], [0] * 5, v[0:5]])

        print(ortho)
        print(para)

        p_locations = [ortho @ Vector(tup) for tup in tuples2]
        hull = ConvexHull(p_locations)

        polygons = [Polygon(vertices=[coords.coords2location(p_locations[i]) for i in face], color='plastic_text') for
                    face in hull.faces]
        [polygon.appear(alpha=0.25, begin_time=t0 + 0.1 * i, transition_time=0.5) for i, polygon in enumerate(polygons)]
        coords.add_objects(polygons)
        t0 += 0.1 * len(polygons) + 1
        unit = coords.coords2location([1, 0, 0])[0]

        b = 5
        power = b ** 5
        count = 0
        for i in range(power):
            dt = 1 / (count / 10 + 1)

            digits = baserep(i, base=b, length=5)
            digits = [-int(np.floor(b / 2)) + d for d in digits]

            ortho_pos = ortho @ Vector(digits)
            if ortho_pos.dot(ortho_pos) < 2.25:
                for val, fig in zip(digits, digfigs):
                    t0 = fig.show(val, begin_time=t0, transition_time=dt)

                blue = breader.copy(color='plastic_drawing')
                blue.grow(begin_time=t0, transition_time=0)
                blue.move_to(target_location=ortho_origin + unit * Vector(ortho_pos), begin_time=t0, transition_time=dt)

                if hull.is_inside(ortho_pos):
                    blue.change_color(new_color='plastic_joker', begin_time=t0, transition_time=dt)
                    blue.rescale(rescale=[0.25, 0.25, 0.25], begin_time=t0 + np.maximum(0.05, dt / 2),
                                 transition_time=dt)
                    red = breader.copy(color='plastic_important')
                    red.grow(begin_time=t0, transition_time=0)
                    red.rescale(rescale=[0.25, 0.25, 0.25], begin_time=t0 + np.maximum(0.05, dt / 2),
                                transition_time=dt)

                    p = Vector(para @ Vector(digits))
                    projection = Vector([p[0], p[1], p[2]])
                    location = plane_center + projection
                    red.move_to(target_location=location, begin_time=t0, transition_time=dt)
                else:
                    blue.rescale(rescale=[0.1, 0.1, 0.1], begin_time=t0 + np.maximum(0.05, dt / 2), transition_time=dt)
                    blue.change_color(new_color='custom1', begin_time=t0, transition_time=dt)

                count += 1
        t0 += 0.5
        self.t0 = t0

    def penrose_rotation(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100, taa_render_samples=64)

        camera_location = [0, -21, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        display = Display(flat=True, number_of_lines=20, color='gray_1', location=[5, 0, 0], scales=[6.5, 6],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])

        coords = CoordinateSystem(dim=3, lengths=[5, 5, 5],
                                  domains=[[-1, 1], [-1, 1], [-1, 1]],
                                  all_n_tics=[0, 0, 0],
                                  all_tic_labels=[np.arange(0, 0, 1), np.arange(0, 0, 1),
                                                  np.arange(0, 0, 1)],
                                  label_digits=[0, 0, 0],
                                  include_zeros=[False, False, False],
                                  radii=[0.03, 0.03, 0.03],
                                  labels=[r'x', r'y', r'z'],
                                  colors=['plastic_text', 'plastic_text', 'plastic_text'],
                                  axis_label_closenesses=[0.5] * 3,
                                  location_of_origin=[0, 0, -1.75],
                                  )

        coords.rescale(rescale=2, begin_time=0, transition_time=0)
        coords.rotate(rotation_quaternion=Quaternion(Vector([0, 0, 1]), -25 / 180 * pi), begin_time=0,
                      transition_time=0)
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=2)
        coords.rotate(rotation_quaternion=Quaternion(Vector([0, 0, 1]), - 115 * np.pi / 180), begin_time=t0,
                      transition_time=1)

        unit = coords.coords2location(Vector([1, 0, 0]))[0]

        lattice_node = penrose_3D_analog(size=2, radius=0.05, plane_size=2,colors=["x12_color","x13_color","x23_color"])
        lattice = Cube()
        lattice.add_mesh_modifier(type='NODES', node_group=lattice_node)
        coords.add_object(lattice)

        max_material = ibpy.get_material_from_modifier(lattice_node, "MaxMaterial")
        min_material = ibpy.get_material_from_modifier(lattice_node, "MinMaterial")
        cube_material = ibpy.get_material_from_modifier(lattice_node, "CubeMaterial")
        cube_scale = ibpy.get_geometry_node_from_modifier(lattice_node, label='CubeScale').outputs[0]
        ibpy.change_default_value(cube_scale, from_value=0, to_value=0, begin_time=0, transition_time=0)
        materials = [max_material, min_material, cube_material]
        [ibpy.change_alpha_of_material(mat, from_value=0, to_value=0, begin_time=0, transition_time=0) for mat in
         materials]

        projection = get_geometry_node_from_modifier(lattice_node, label='Projection').outputs[0]
        change_default_value(projection, from_value=1, to_value=1, begin_time=0, transition_time=0)

        r_slider = get_geometry_node_from_modifier(lattice_node, label='rMax')

        scale = get_geometry_node_from_modifier(lattice_node, label='ScaleElements').outputs[0]
        extrude = get_geometry_node_from_modifier(lattice_node, label='ExtrudeElements').outputs[0]
        ibpy.change_default_value(scale, from_value=0.95, to_value=0.95, begin_time=0, transition_time=0)
        ibpy.change_default_value(extrude, from_value=0.01, to_value=0.01, begin_time=0, transition_time=0)

        # hide lattice points
        cubie_scale = ibpy.get_geometry_node_from_modifier(lattice_node, label='CubieScale').outputs[0]
        ibpy.change_default_value(cubie_scale, from_value=0, to_value=0, begin_time=0, transition_time=0)

        showPlane = get_geometry_node_from_modifier(lattice_node, label='showPlane')
        plane_material = get_geometry_node_from_modifier(lattice_node, label='planeMaterial').inputs[
            'Material'].default_value
        change_default_boolean(showPlane, from_value=False, to_value=True, begin_time=t0, transition_time=0.1)
        change_default_value(r_slider.outputs[0], from_value=0, to_value=3, begin_time=t0, transition_time=2)

        # normal
        normal = PArrow(start=[0, 0, 0], end=[0, 0, unit],
                        color='plastic_drawing', thickness=2, name="normal_arrow")
        normal.grow(begin_time=t0, transition_time=1)
        coords.add_object(normal)
        ibpy.change_alpha_of_material(plane_material, from_value=0, to_value=1, begin_time=t0,
                                      transition_time=1)

        # show faces
        ranges = [
            get_geometry_node_from_modifier(lattice_node, label='xyRange').outputs[0],
            get_geometry_node_from_modifier(lattice_node, label='xzRange').outputs[0],
            get_geometry_node_from_modifier(lattice_node, label='yzRange').outputs[0],
        ]
        for r in ranges:
            change_default_value(r, from_value=0, to_value=6, begin_time=t0, transition_time=1)

        normal.rotate(rotation_euler=[-np.arcsin(np.sqrt(2 / 3)), 0, -np.arccos(np.sqrt(1 / 2))], begin_time=t0,
                      transition_time=3)
        rotatePlane = get_geometry_node_from_modifier(lattice_node, label='rotatePlane')
        coords.move_to(target_location=[0, 0, 0.6], begin_time=t0, transition_time=2)
        t0 = 0.5 + change_default_vector(rotatePlane, to_value=Vector([0, np.pi / 2 - np.arctan(1 / r2), np.pi / 4]),
                                         from_value=Vector(), begin_time=t0, transition_time=3)

        coords.move_to(target_location=[-6.5, 0, -1.5], begin_time=t0, transition_time=2)
        coords.rotate(rotation_quaternion=Quaternion(Vector([0, 0, 1]), - 90 * np.pi / 180), begin_time=t0,
                      transition_time=2)
        t0 = 0.5 + coords.rescale(rescale=[0.707] * 3, begin_time=t0, transition_time=2)
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=2)

        line = -1
        normal_text0 = SimpleTexBObject(r"\text{Normal vector:}")
        t0 = 0.5 + display.write_text_in(normal_text0, line=line, indent=1, begin_time=t0, transition_time=0.25)

        normal_text = SimpleTexBObject(
            r"\vec{n} = \frac{1}{\sqrt{3}}\left(\begin{array}{c} 1 \\  1 \\1 \end{array}\right)",
            color='drawing', aligned="center")
        t0 = 0.5 + display.write_text_in(normal_text, letter_set=[1, 0, 2, 4, 5, 7, 3, 6, 8, 9, 10, 11, 12, 13, 14],
                                         line=line + 2, indent=5, begin_time=t0, transition_time=1)

        # rotation
        rotation_text = SimpleTexBObject(r"\text{Rotation:}")
        t0 = 0.5 + display.write_text_in(rotation_text, line=line + 4, indent=1, begin_time=t0, transition_time=0.25)

        rot_text = SimpleTexBObject(r"R = \left(\begin{array}{c c c} 0 & 0 & 1\\1& 0 &0 \\0& 1&0\end{array}\right)",
                                    color='example', aligned="center")
        display.write_text_in(rot_text, letter_set=[0, 1, 2, 3, 4, 7, 10, 5, 8, 11, 6, 9, 12, 13, 14],
                              line=line + 6, indent=5, begin_time=t0 + 1, transition_time=1)

        t0 = 0.5 + coords.rotate(
            rotation_quaternion=Quaternion(Vector([1, -1, 1]), 2 / 3 * np.pi) @ Quaternion(Vector([0, 0, 1]),
                                                                                           - 90 * np.pi / 180),
            begin_time=t0,
            transition_time=3)
        t0 = 0.5 + coords.rotate(
            rotation_quaternion=Quaternion(Vector([1, -1, 1]), 0) @ Quaternion(Vector([0, 0, 1]), - 90 * np.pi / 180),
            begin_time=t0,
            transition_time=3)

        eigenvalue_text = SimpleTexBObject(r"\text{Eigenvalues:}")
        t0 = 0.5 + display.write_text_in(eigenvalue_text, line=line + 8, begin_time=t0,
                                         transition_time=0.5, indent=1)
        colors = flatten([["drawing"], ["text"], ["custom1"] * 11, ["text"], ["gray_6"] * 11, ["text"]])
        eigenvalues_text = SimpleTexBObject(r"1,-\tfrac{1}{2}-i\tfrac{\sqrt{3}}{2},-\tfrac{1}{2}+i\tfrac{\sqrt{3}}{2}",
                                            color=colors, aligned='center')
        t0 = 0.5 + display.write_text_in(eigenvalues_text,
                                         letter_set=list(range(0, 8)) + [12, 9, 10, 8, 11] + list(range(13, 20)) + [24,
                                                                                                                    21,
                                                                                                                    22,
                                                                                                                    20,
                                                                                                                    23],
                                         line=line + 9, indent=5, begin_time=t0, transition_time=1)

        eigenvector_text = SimpleTexBObject(r"\text{Eigenvectors:}")
        t0 = 0.5 + display.write_text_in(eigenvector_text, indent=1,
                                         line=line + 11, begin_time=t0,
                                         transition_time=0.5)
        colors = flatten([["drawing"] * 7, ["text"], ["custom1"] * 31, ["text"], ["gray_6"] * 31, ["text"]])

        eigenvectors_text = SimpleTexBObject(r"\left(\begin{array}{c}1\\1\\1\end{array}\right), "
                                             r"\left(\begin{array}{c} -\tfrac{1}{2}-i\tfrac{\sqrt{3}}{2}\\\rule{0em}{3ex}-\tfrac{1}{2}+i\tfrac{\sqrt{3}}{2}\\\rule{0em}{2ex}1\end{array}\right),"
                                             r"\left(\begin{array}{c} -\tfrac{1}{2}+i\tfrac{\sqrt{3}}{2}\\\rule{0em}{3ex}-\tfrac{1}{2}-i\tfrac{\sqrt{3}}{2}\\\rule{0em}{2ex}1\end{array}\right)",
                                             color=colors, recreate=False, aligned="center")

        t0 = 0.5 + display.write_text_in(eigenvectors_text,
                                         letter_set=list(range(0, 8)) +
                                                    [10, 8, 9, 11, 12, 14, 15, 16, 20, 23, 30, 26, 27, 25, 29, 13, 17,
                                                     18, 19, 21, 24, 34, 31, 32, 28, 33, 22, 35, 37, 38, 36] +
                                                    [39, 42, 40, 41, 43, 44, 46, 47, 48, 52, 55, 62, 58, 59, 57, 61, 45,
                                                     49, 50, 51, 53, 56, 66, 63, 64, 60, 65, 54, 67, 69, 70, 68],
                                         line=line + 13, indent=5, begin_time=t0, transition_time=3)

        colors = flatten([["custom1"] * 18, ["text"], ["custom1"]])
        uv_text = SimpleTexBObject(
            r"\vec{u} = \left(\begin{array}{c}-\tfrac{1}{2}\\-\rule{0em}{3ex}\tfrac{1}{2}\\\rule{0em}{2ex}1\end{array}\right), \vec{v} = \left(\begin{array}{c}-\tfrac{\sqrt{3}}{2}\\\tfrac{\sqrt{3}}{2}\\\rule{0em}{2ex}0\end{array}\right) ",
            color=colors, aligned="center")

        display.write_text_in(uv_text, letter_set=[1, 0, 2, 4, 3, 5, 6, 9, 10, 11, 7, 12, 13, 14, 8, 15, 17, 16] +
                                                  [18, 19, 20, 21, 23, 22, 24, 25, 36, 28, 29, 27, 35, 31, 32, 33, 26,
                                                   30, 34, 37, 39, 38],
                              line=line + 17, indent=5, begin_time=t0, transition_time=2)

        u = PArrow(start=[0, 0, 0], end=unit * Vector([-0.5, -0.5, 1]), color='plastic_custom1', thickness=2,
                   name="u_arrow")
        v = PArrow(start=[0, 0, 0], end=unit * Vector([-r3 / 2, r3 / 2, 0]), color='plastic_custom1', thickness=2,
                   name="v_arrow")
        coords.add_objects([u, v])

        u.grow(begin_time=t0, transition_time=3)
        v.grow(begin_time=t0, transition_time=3)
        change_default_value(r_slider.outputs[0], from_value=3, to_value=0, begin_time=t0, transition_time=2)

        t0 = coords.rotate(
            rotation_quaternion=Quaternion(Vector([1, -1, 1]), 2 / 3 * np.pi) @ Quaternion(Vector([0, 0, 1]),
                                                                                           - np.pi / 2), begin_time=t0,
            transition_time=3)
        # remove faces
        for r in ranges:
            change_default_value(r, from_value=6, to_value=0, begin_time=t0, transition_time=1)
        t0 += 0.5

        t0 = 0.5 + coords.rotate(
            rotation_quaternion=Quaternion(Vector([1, -1, 1]), 0) @ Quaternion(Vector([0, 0, 1]), - np.pi / 2),
            begin_time=t0,
            transition_time=3)

        coords.rescale(rescale=[0] * 3, begin_time=t0, transition_time=1)
        normal_text0.disappear(begin_time=t0, transition_time=1)
        normal_text.disappear(begin_time=t0 + 0.15, transition_time=1)
        # rotation_text.disappear(begin_time=t0+0.3,transition_time=1)
        # rot_text.disappear(begin_time=t0+0.6,transition_time=1)
        eigenvalue_text.disappear(begin_time=t0 + 0.3, transition_time=1)
        eigenvalues_text.disappear(begin_time=t0 + 0.45, transition_time=1)
        eigenvector_text.disappear(begin_time=t0 + 0.6, transition_time=1)
        eigenvectors_text.disappear(begin_time=t0 + 0.75, transition_time=1)
        t0 = 1 + uv_text.disappear(begin_time=t0 + 0.9, transition_time=1)

        self.t0 = t0

    def penrose_rotation2(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100, taa_render_samples=1024)

        camera_location = [0, -21, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        display = Display(flat=True, number_of_lines=20, color='gray_1', location=[5, 0, 0], scales=[6.5, 6],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])
        t0 = display.appear(begin_time=t0, transition_time=0)

        line = -1
        # rotation
        rotation_text = SimpleTexBObject(r"\text{Rotation:}")
        display.write_text_in(rotation_text, line=line + 4, indent=1, begin_time=0, transition_time=0)

        rot_text = SimpleTexBObject(r"R = \left(\begin{array}{c c c} 0 & 0 & 1\\1& 0 &0 \\0& 1&0\end{array}\right)",
                                    color='example', aligned="center")
        display.write_text_in(rot_text, letter_set=[0, 1, 2, 3, 4, 7, 10, 5, 8, 11, 6, 9, 12, 13, 14],
                              line=line + 6, indent=5, begin_time=0, transition_time=0)


        rotation_text.move(direction=[0, 0.4, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + rot_text.move(direction=[0, 0.4, 0], begin_time=t0, transition_time=1)

        rot_text2 = SimpleTexBObject(r"R = \left(\begin{array}{c c c c c}"
                                     r" 0 & 0 & 0 & 0 & 1\\"
                                     r" 1 & 0 & 0 & 0 & 0\\"
                                     r" 0 & 1 & 0 & 0 & 0\\"
                                     r" 0 & 0 & 1 & 0 & 0\\"
                                     r" 0 & 0 & 0 & 1 & 0\end{array}\right)",
                                     color='example', aligned="center")
        t0 = 0.5 + rot_text.replace(rot_text2, begin_time=t0, transition_time=1)

        # eigenvalues
        t0 = 0.5 + display.write_text_in(SimpleTexBObject(r"\text{Eigenvalues:}"),
                                         line=line+5, begin_time=t0,
                                         transition_time=0.5,
                                         indent=1)
        colors = flatten([["drawing"], ["text"], ["custom1"] * 21, ["text"], ["drawing"] * 21, ["text"]])
        eigenvalues_text = SimpleTexBObject(r"1, "
                                            r"-\tfrac{\sqrt{5}+1}{4}\pm i\sqrt{\tfrac{5}{8}-\tfrac{\sqrt{5}}{8}}, "
                                            r"\tfrac{\sqrt{5}-1}{4}\pm i\sqrt{\tfrac{5}{8}+\tfrac{\sqrt{5}}{8}}",
                                            color=colors, aligned='center', recreate=False)

        t0 = 0.5 + display.write_text_in(eigenvalues_text,
                                         letter_set=[0,1,2,5,3,4,8,9,6,7,10,11,13,14,15,16,22,19,20,18,21,12,17,23]+
                                         [26,24,25,29,30,27,28,31,32,34,35,36,37,43,40,41,39,42,33,38],
                                         line=line+6, indent=5,
                                         begin_time=t0,
                                         transition_time=1)
        # eigenvectors
        t0 = 0.5 + display.write_text_in(SimpleTexBObject(r"\text{Eigenvectors:}"),
                                         indent=1, line=line+8, begin_time=t0,
                                         transition_time=0.5)
        colors = flatten(
            [["drawing"] * 22, ["text"], ["custom1"] * 41, ["text"], ["custom1"] * 57, ["text"], ["drawing"] * 41,
             ["text"], ["drawing"]])

        eigenvectors_text = SimpleTexBObject(
            r"\tfrac{1}{\sqrt{5}}\left(\begin{array}{c}1\\1\\1\\1\\1\end{array}\right), "
            r"\frac{1}{2\sqrt{10}}\left(\begin{array}{c} 1+\sqrt{5}\\1-\sqrt{5}\\1-\sqrt{5}\\1+\sqrt{5}\\-4\end{array}\right),"
            r"\frac{1}{2\sqrt{5}}\left(\begin{array}{c}\sqrt{5-\sqrt{5}}\\\rule{0em}{3ex}-\sqrt{5+\sqrt{5}}\\\rule{0em}{3ex}\sqrt{5+\sqrt{5}}\\\rule{0em}{3ex}-\sqrt{5-\sqrt{5}}\\\rule{0em}{2ex}0\end{array}\right), "
            r"\frac{1}{2\sqrt{10}}\left(\begin{array}{c} 1-\sqrt{5}\\1+\sqrt{5}\\1+\sqrt{5}\\1-\sqrt{5}\\-4\end{array}\right),"
            r"\frac{1}{2\sqrt{5}}\left(\begin{array}{c}\sqrt{5+\sqrt{5}}\\\rule{0em}{3ex}\sqrt{5-\sqrt{5}}\\\rule{0em}{3ex}-\sqrt{5-\sqrt{5}}\\\rule{0em}{3ex}-\sqrt{5+\sqrt{5}}\\\rule{0em}{2ex}0\end{array}\right) ",
            color=colors, recreate=False, aligned="center",text_size='xs')

        t0 = 0.5 + display.write_text_in(eigenvectors_text,
                                        letter_set= [
                                            1,2,4,0,3,9,5,6,7,8,10,11,12,13,14,15,16,18,19,20,21,17,22
                                                    ]+[
                                            25,26,23,27,29,24,28,34,30,31,32,33,35,36,40,51,46,50,37,41,53,47,52,38,42,55,48,54,39,43,57,49,56,44,45,58,60,61,62,63,59,64
                                        ]+[
                                            66,67,69,70,68,65,79,71,72,73,74,75,76,77,78,80,85,91,103,98,102,83,94,81,89,96,109,104,108,86,99,87,92,106,100,105,84,95,82,90,97,111,107,110,88,101,93,112]+list(range(114,122))+[113,122]
                                         +[
                                        125,126,123,127,129,124,128,134,130,131,132,133,135,136,140,151,146,150,137,141,153,147,152,138,142,155,148,154,139,143,157,149,156,144,145,158,160,161,162,163,159,164
                                                    ]+[
                                            166,167,165,170,168,169,179]+list(range(171,179))+[180,185,191,203,198,202,183,194,186,192,205,199,204,184,195,181,189,196,209,206,208,187,200,182,190,197,211,207,210,188,201,193,212]+list(range(214,222))+[213],
                                         line=line+11, indent=5, begin_time=t0, transition_time=4)

        p2 = SimpleTexBObject(r"\text{Projection plane:}", color='custom1')
        t0 = 0.5 + display.write_text_in(p2, line=line+14, indent=1, begin_time=t0, transition_time=0.5)

        u_arrow = PArrow(name=r"\vec{u}", start=[-8, 0, -4], end=[-6, 0, -2], color="plastic_custom1", thickness=2,label_rotation=[pi/2,0,0])
        u_arrow.write_name_as_label(begin_time=t0, transition_time=0.5, modus="up", aligned='bottom',
                                    scale_compensate=False)
        u_arrow.grow(begin_time=t0 + 0.8, transition_time=0.2)

        v_arrow = PArrow(name=r"\vec{v}", start=[-8, 0, -4], end=[-10, 0, -2], color="plastic_custom1", thickness=2,label_rotation=[pi/2,0,0])
        v_arrow.write_name_as_label(begin_time=t0, transition_time=0.5, modus="up", aligned='bottom',
                                    scale_compensate=False)
        v_arrow.grow(begin_time=t0 + 0.8, transition_time=0.2)

        plane = Plane(color='plastic_custom1')
        plane.move_to(target_location=[-8, 0, -4], begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[np.pi / 2, np.pi / 4, 0], begin_time=t0, transition_time=0)
        plane.rescale(rescale=[2] * 3, begin_time=t0, transition_time=0)
        plane.appear(alpha=0.5, begin_time=t0, transition_time=2)

        u = SimpleTexBObject(r"\vec{u}", color="custom1", text_size="large")
        v = SimpleTexBObject(r"\vec{v}", color="custom1", text_size="large")
        t0 = display.write_text_in(u, line=line+15, indent=2.25, begin_time=t0, transition_time=0.2)
        t0 = 0.5 + display.write_text_in(v, line=line+15, indent=4.5, begin_time=t0, transition_time=0.2)

        p2 = SimpleTexBObject(r"\text{Orthogonal Space:}", color='drawing')
        t0 = 0.5 + display.write_text_in(p2, line=line+17, indent=1, begin_time=t0, transition_time=0.5)

        w1_arrow = PArrow(name=r"\vec{w}_1", start=[0, 0, 0], end=[2, 0, 0], color="plastic_drawing", thickness=2,label_rotation=[pi/2,0,0])
        w1_arrow.write_name_as_label(begin_time=t0, transition_time=0.25, modus="up", aligned='bottom',
                                     scale_compensate=False)
        w1_arrow.grow(begin_time=t0 + 0.8, transition_time=0.2)

        w2_arrow = PArrow(name=r"\vec{w}_2", start=[0, 0, 0], end=[0, 2, 0], color="plastic_drawing", thickness=2,label_rotation=[pi/2,0,0])
        w2_arrow.write_name_as_label(begin_time=t0, transition_time=0.25, modus="up", aligned='bottom',
                                     scale_compensate=False)
        w2_arrow.grow(begin_time=t0 + 0.8, transition_time=0.2)

        w3_arrow = PArrow(name=r"\vec{w}_3", start=[0, 0, 0], end=[0, 0, 2], color="plastic_drawing", thickness=2,label_rotation=[pi/2,0,0])
        w3_arrow.write_name_as_label(begin_time=t0, transition_time=0.25, modus="up", aligned='bottom',
                                     scale_compensate=False)
        w3_arrow.grow(begin_time=t0 + 0.8, transition_time=0.2)

        rot_box = BObject(children=[w1_arrow, w2_arrow, w3_arrow], location=[-8, 0, 2],
                          rotation_euler=[np.pi / 4, -np.pi / 9, 0])
        rot_box.appear(begin_time=t0, transition_time=0)

        w1 = SimpleTexBObject(r"\vec{w}_1", color="drawing", text_size="large")
        w2 = SimpleTexBObject(r"\vec{w}_2", color="drawing", text_size="large")
        w3 = SimpleTexBObject(r"\vec{w}_3", color="drawing", text_size="large")
        t0 = display.write_text_in(w1, line=line+18, indent=0.75, begin_time=t0, transition_time=0.2)
        t0 = display.write_text_in(w2, line=line+18, indent=6.5, begin_time=t0, transition_time=0.2)
        t0 = 0.5 + display.write_text_in(w3, line=line+18, indent=8.75, begin_time=t0, transition_time=0.2)

        sphere = Sphere(0.125,mesh_type="iso" ,smooth=False,resolution=2,location=[-8, 0, 0], color='plastic_example', name=r"\vec{x}", scale=[2] * 3,label_rotation=[pi/2,0,0])
        t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + sphere.write_name_as_label(scale_compensate=False, modus="right", begin_time=t0, transition_time=0.3,
                                              aligned="left")

        sphere2 = Sphere(0.12, mesh_type="iso",smooth=False,resolution=2,location=[-8, 0, 0], scale=[2] * 3, color='plastic_custom1',
                         name=r"(\vec{u}\cdot \vec{x},\vec{v}\cdot \vec{x})",label_rotation=[pi/2,0,0])
        sphere2.grow(begin_time=t0, transition_time=0)
        sphere2.move(direction=[0.8, 0, -5.7], begin_time=t0, transition_time=2)
        sphere2.write_name_as_label(scale_compensate=False, modus="right", begin_time=t0 + 1, transition_time=1,
                                    aligned="left")

        sphere3 = Sphere(0.12,mesh_type="iso",smooth=False, resolution=2,location=[-8, 0, 0], scale=[2] * 3, color='plastic_drawing',
                         name=r"(\vec{w}_1\cdot \vec{x},\vec{w}_2\cdot \vec{x},\vec{w}_3\cdot \vec{x})",label_rotation=[pi/2,0,0])
        sphere3.grow(begin_time=t0, transition_time=0)
        sphere3.move(direction=[-1.5, 2, 4.7], begin_time=t0, transition_time=2)
        t0 = 0.5 + sphere3.write_name_as_label(scale_compensate=False, modus="right", begin_time=t0 + 1,
                                               transition_time=1, aligned="left")

        self.t0 = t0

    def maybe_penrose2(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        display = Display(flat=True, number_of_lines=30, color='gray_2', location=[4.5, 0, 0], scales=[6.5, 6],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1)

        # repeat analysis from 3d
        # don't be a never-nester
        tupels = []
        for x1 in range(-2, 3):
            for x2 in range(-2, 3):
                for x3 in range(-2, 3):
                    for x4 in range(-2, 3):
                        for x5 in range(-2, 3):
                            tupels.append((x1, x2, x3, x4, x5))

        # now we have 3125 points

        # calculate projections points
        r3 = np.sqrt(3)
        r5 = np.sqrt(5)
        tupels2 = tuples([0.5, -0.5], 5)
        ortho = np.array([
            [1 / r3, 1 / r3, 1 / r3, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        r2 = np.sqrt(2)
        para = np.array([
            [1 / r2, -1 / r2, 0, 0, 0],
            [1 / r2 / r3, 1 / r2 / r3, -r2 / r3, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        p_locations = [ortho @ Vector(tup) for tup in tupels2]
        hull = ConvexHull(p_locations)

        selected_points = [tup for tup in tupels if hull.is_inside(ortho @ Vector(tup))]

        print(len(selected_points), " of 3125 selected")

        # construct all faces
        units = [Vector(u) for u in unit_tuples(5)]
        directions = face_directions(5)

        faces = {}
        for dir in directions:
            d0 = dir[0]
            d1 = dir[1]
            for pos in tupels:
                if pos[d0] < 2 and pos[d1] < 2:
                    faces[(pos, dir)] = plane_for_center_and_directions(pos, [units[d0], units[d1]])

        print(len(faces))

        colors = ['text', 'example', 'custom1', 'drawing', 'joker', 'custom2', 'custom3', 'custom4', 'gray2', 'gray4']
        colors = ['plastic_' + col for col in colors]

        print(colors)
        i = 0
        selected_faces = {key: val for key, val in faces.items()
                          if all([hull.is_inside(ortho @ point) for point in val])}

        polygons = []

        for key, val in selected_faces.items():
            pos, dir = key
            color = colors[directions.index(dir)]
            polygons.append(Polygon(vertices=[para @ v for v in val], color=color))

        for i, polygon in enumerate(polygons):
            polygon.appear(begin_time=t0 + 0.1 * i, transition_time=0.5)

        t0 += len(polygons) * 0.1 + 1
        self.t0 = t0

    def convex_hull_3d(self):
        t0 = 0
        size = 4
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=False,
                                 transparent=True, rotation_euler=[0, 0, 0])
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=1024)

        camera_location = [0, -20, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        display = Display(flat=True, number_of_lines=20, color='gray_2', location=[4, 0, 0], scales=[7.5, 6],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])
        display.appear(begin_time=t0, transition_time=1)

        # recall 3d first

        cube_coords = tuples([0,1],dim=3)
        n = 1/r3*Vector([1,1,1])
        a = 1/r2* Vector([-1,1,0])
        b = 1/r2/r3 * Vector([1,1,-2])
        projector = np.array([
            a[0:3], b[0:3]
        ])
        projection = lambda x:(projector@(x-offset))[0]*a+(projector@(x-offset))[1]*b+offset

        locations = [Vector(tup) for tup in cube_coords]
        offset = Vector([-9,0,3])
        spheres = [Sphere(0.1,location=2.5*Vector(location)+offset,color='plastic_drawing') for location in locations]
        [sphere.grow(begin_time=t0,transition_time=0.1) for sphere in spheres]

        spheres2 = [Sphere(0.099, location=2.5 * Vector(location) + offset, color='plastic_joker') for location in
                   locations]
        [sphere.grow(begin_time=t0, transition_time=0.1) for sphere in spheres2]

        lines = []
        pairs = list(combinations(locations,2))
        pairs = [pair for pair in pairs if (pair[0]-pair[1]).length==1]

        for pair in pairs:
            lines.append(Cylinder.from_start_to_end(start=2.5*pair[0]+offset,end=2.5*pair[1]+offset,color='plastic_drawing'))
            lines[-1].grow(begin_time=t0,transition_time=1,modus='from_start')

        t0 +=1.5

        lines2 = []

        pairs = list(combinations(locations, 2))
        pairs = [pair for pair in pairs if (pair[0] - pair[1]).length == 1]

        for pair in pairs:
            lines2.append(Cylinder.from_start_to_end(start=2.5 * pair[0] + offset, end=2.5 * pair[1] + offset,
                                                    color='plastic_joker',radius=0.099))
            lines2[-1].grow(begin_time=t0, transition_time=0, modus='from_start')

        projection_line = Cylinder.from_start_to_end(start=-2.5*n+offset,end=5*n+offset,color='custom1')
        t0 =0.5+ projection_line.grow(begin_time=t0,transition_time=1,modus='from_start')

        for line in lines2:
            line.move_to_new_start_and_end_point(start=projection(line.start),end=projection(line.end),begin_time=t0,transition_time=2)
        for loc,sphere in zip(locations,spheres2):
            sphere.move_to(target_location=projection(2.5*loc+offset),begin_time=t0,transition_time=2)
        t0 +=2.5

        removals = [line for line in lines2 if chop((to_vector(line.start)-offset).length)==0 or chop((to_vector(line.end)-offset).length)==0]
        for r in removals:
            r.disappear(begin_time=t0,transition_time=2)
        point_set = set([tuple(chop(list(projection(2.5*loc+offset)))) for loc in locations])-{tuple(offset)}
        points = [to_vector(tup) for tup in point_set]
        polygon = Polygon(points,color='plastic_joker')
        t0=0.5+polygon.grow(alpha=0.75,begin_time=t0,transition_time=2)

        title = SimpleTexBObject(r"\text{The Convex Hull For Penrose Tilings}", aligned='center', color='example')
        t0 = 0.5 + display.write_title(title, begin_time=t0, transition_time=1)

        projection = SimpleTexBObject(
            r"\text{Apply }P_{\perp}=\left(\begin{array}{c}\vec{w}_1\\\vec{w}_2\\\vec{w}_3\end{array}\right)="
            r"\frac{1}{\sqrt{5}}\left(\begin{array}{c c c c c} 1&1&1&1&1\\"
            r' \rule{0em}{2.5ex}\frac{1+\sqrt{5}}{2\sqrt{2}}& \frac{1-\sqrt{5}}{2\sqrt{2}}& \frac{1-\sqrt{5}}{2\sqrt{2}}&\frac{1+\sqrt{5}}{2\sqrt{2}}& -\frac{2}{\sqrt{2}}\\'
            r' \rule{0em}{2.5ex}\frac{1}{2}\sqrt{5+\sqrt{5}}&\frac{1}{2}\sqrt{5-\sqrt{5}}& -\frac{1}{2}\sqrt{5-\sqrt{5}} & -\frac{1}{2}\sqrt{5+\sqrt{5}}&0\end{array}\right) ',
            color='joker', text_size='Small')

        t0 = 0.5 + display.write_text_in(projection, line=16, indent=0.5, begin_time=t0, transition_time=3)

        cube_coords = tuples([0,1], dim=5)
        cube_strings = [f"({c[0]}|{c[1]}|{c[2]}|{c[3]}|{c[4]})" for c in cube_coords]
        b_cubes = np.array([SimpleTexBObject(cube_string,color='drawing') for cube_string in cube_strings])
        table = Table(b_cubes.reshape((8, 4)), bufferx=0.4, buffery=0.4)
        table.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=0, transition_time=0)
        display.add_text_in(table, line=5, indent=4)
        [table.write_row(i, begin_time=t0 + i * 1, transition_time=1) for i in range(8)]
        t0 += 0.5 + 8 * 1

        pairs = list(combinations(cube_coords,2))
        line_pairs = []
        for pair in pairs:
            if(to_vector(pair[0])-to_vector(pair[1])).length==1:
                line_pairs.append(pair)

        # calculate projections points
        rr5p = np.sqrt(5 + r5)
        rr5m = np.sqrt(5 - r5)

        # data
        u = 1 / r2 / 2 / r5 * Vector([1 + r5, 1 - r5, 1 - r5, 1 + r5, -4])
        v = 1 / 2 / r5 * Vector([rr5m, -rr5p, rr5p, -rr5m, 0])
        w1 = 1 / r5 * Vector([1, 1, 1, 1, 1])
        w2 = 1 / r2 / 2 / r5 * Vector([1 + r5, 1 - r5, 1 - r5, 1 + r5, -4])
        w3 = 1 / 2 / r5 * Vector([rr5p, rr5m, -rr5m, -rr5p, 0])

        coords = CoordinateSystem(dim=3, lengths=[5, 5, 5],
                                  domains=[[0, 2], [-1, 1], [-1, 1]],
                                  all_n_tics=[2, 2, 2],
                                  all_tic_lablls=[np.arange(0, 2.1, 1), np.arange(-1, 1.1, 1),
                                                  np.arange(-1, 1.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.03, 0.03, 0.03],
                                  labels=[r'\vec{w}_1', r'\vec{w}_2', r'\vec{w}_3'],
                                  colors=['plastic_drawing', 'plastic_drawing', 'plastic_drawing'],
                                  location_of_origin=[-2.5, 0, 0],
                                  )


        rot_box = BObject(children=[coords],location=[-8,0,-3.5])
        rot_box.appear()
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=1)
        rot_box.rotate(rotation_euler=[0, 0, -2 * np.pi - np.pi / 6], begin_time=t0-1.5, transition_time=30,pivot=coords.coords2location([1,0,0]))

        tuples2 = tuples([0,1], 5)
        projector = np.array([
            w1[0:5], w2[0:5], w3[0:5]
        ])

        p_locations = [projector @ Vector(tup) for tup in tuples2]
        projections = [Sphere(0.1, mesh_type="iso",smooth=False,resolution=2,location=coords.coords2location(loc), color="plastic_joker") for loc in p_locations]
        coords.add_objects(projections)
        for i ,projection in enumerate(projections):
            row = int(i/4)
            col = i%4
            projection.grow(begin_time=t0 + i * 0.1, transition_time=0.5)
            table.get_entry(row,col).disappear(begin_time=t0+i*0.1,transition_time=0.5)
        t0 += len(projections) * 0.1 + 1

        center = Vector()
        for loc in p_locations:
            center+=to_vector(loc)
        center =1/len(p_locations)*center

        lines = []
        for i,pair in enumerate(line_pairs):
            start = projector@to_vector(pair[0])
            end = projector@to_vector(pair[1])
            line = Cylinder.from_start_to_end(start=coords.coords2location(start),end=coords.coords2location(end),radius=0.05,color='plastic_joker')
            line.grow(begin_time=t0+i*0.1,transition_time=0.5,modus='from_start')
            lines.append(line)
            coords.add_object(line)

        t0 +=len(line_pairs)*0.1+1

        hull = ConvexHull(p_locations)

        sorted_p_locs = sorted(p_locations, key = lambda x: to_vector((x-center)).length)
        inside = sorted_p_locs[0:10]

        removals =[]
        for i,pair in enumerate(line_pairs):
            start = projector@to_vector(pair[0])
            end = projector@to_vector(pair[1])
            test_start =any([chop(to_vector(start-x).length)==0 for x in inside])
            test_end =any([chop(to_vector(end-x).length)==0 for x in inside])
            if test_start or test_end:
                removals.append(i)

        removers = []
        for i,line in enumerate(lines):
            if i in removals:
                removers.append(line)

        for i,remover in enumerate(removers):
            remover.disappear(begin_time=t0+i*0.1,transition_time=0.5)

        t0 +=len(removers)*0.1+1

        polygons = [Polygon(vertices=[coords.coords2location(p_locations[i]) for i in face], color='plastic_joker') for
                    face in hull.faces]
        [polygon.appear(alpha=0.75, begin_time=t0 + 0.1 * i, transition_time=0.5) for i, polygon in enumerate(polygons)]
        coords.add_objects(polygons)
        t0 += 0.1 * len(polygons) + 1

        self.t0 = t0

    def plane_rotation_5d(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # setup coordinate system
        coords = CoordinateSystem(dim=3, lengths=[7.5, 7.5, 5],
                                  domains=[[0, 3], [0, 3], [-1, 1]],
                                  all_n_tics=[3, 3, 2],
                                  all_tic_lablls=[np.arange(-0, 3.1, 1), np.arange(-0, 3.1, 1),
                                                  np.arange(-1, 1.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015],
                                  labels=[r'x_1', r'x_2', r'x_3,x_4,x_5']
                                  )

        rotation_center = coords.coords2location([2, 2, 0])
        coords.move(direction=-rotation_center, begin_time=0, transition_time=0)
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        unit = coords.coords2location(Vector([1, 0, 0]))[0]

        projection_plane = Plane(location=[0, 0, 0], rotation_euler=[0, 0, 0], scale=[2 * unit] * 3,
                                 name="Plane_Projection",
                                 color="plastic_drawing")

        t0 = 0.5 + projection_plane.appear(begin_time=t0, transition_time=1, alpha=0.5)
        coords.add_object(projection_plane)

        # normal
        normal = PArrow(start=[0, 0, 0], end=[0, 0, unit], color='plastic_drawing', thickness=2)
        coords.axes[2].axis_label.change_color_of_letters(range(6, 8), new_color='drawing', begin_time=t0,
                                                          transition_time=1)
        t0 = 0.5 + normal.grow(begin_time=t0, transition_time=1)
        coords.add_object(normal)

        coords.rotate(rotation_euler=[0, 0, -np.pi / 6], begin_time=t0,
                      transition_time=2, compensate=[True, False, True])
        coords.move(direction=Vector((0, 0, -2)), begin_time=t0, transition_time=2)
        t0 = 0.5 + coords.rescale(rescale=[0.5] * 3, begin_time=t0, transition_time=2)

        display = Display(flat=True, number_of_lines=28,
                          color='gray_2', location=[0, 0, 0], scales=[12, 6],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1)

        title = SimpleTexBObject(r"\text{Projection plane in 5D}", color='example', aligned="center")
        t0 = 0.5 + display.write_title(title, begin_time=t0, transition_time=3)

        rot1 = BMatrix(recreate=False,
                       entries=np.array([[1, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 0, np.sqrt(1 / 5), 2 * np.sqrt(1 / 5)],
                                         [0, 0, 0, -2 * np.sqrt(1 / 5), np.sqrt(1 / 5)]]),
                       pre_word=r"R_{x_4 x_5}=",
                       mapping={np.sqrt(1 / 5): r"\tfrac{1}{\sqrt{3}}", 1: "1", 0: "0",
                                2 * np.sqrt(1 / 5): r"\tfrac{\sqrt{4}}{\sqrt{5}}",
                                -2 * np.sqrt(1 / 5): r"-\tfrac{\sqrt{4}}{\sqrt{5}}"})
        t0 = 0.5 + display.write_text_in(rot1, indent=0.25, line=3, begin_time=t0, transition_time=3)

        colors = flatten([['text'] * 6, ['drawing'] * 17, ['text'], ['important']])
        trans1 = SimpleTexBObject(
            r"R_{x_4 x_5}\cdot \left(\begin{array}{c}0\\0\\0\\0\\1\end{array}\right) =\tfrac{1}{\sqrt{5}}\left(\begin{array}{c}0\\0\\0\\2\\1\end{array}\right)",
            color=colors)

        normal.change_color(new_color='plastic_important', begin_time=t0, transition_time=2)
        coords.axes[2].axis_label.change_color_of_letters([3, 4, 6, 7], new_color='plastic_important', begin_time=t0,
                                                          transition_time=2)
        projection_plane.change_color(new_color='plastic_important', begin_time=t0, transition_time=2)
        t0 = 0.5 + display.write_text_in(trans1, indent=0.5, line=8, begin_time=t0, transition_time=2)

        rot2 = BMatrix(recreate=False,
                       entries=np.array([[1, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0],
                                         [0, 0, np.sqrt(1 / 4), np.sqrt(3 / 4), 0],
                                         [0, 0, -np.sqrt(3 / 4), np.sqrt(1 / 4), 0],
                                         [0, 0, 0, 0, 1]]),
                       pre_word=r"R_{x_3 x_4}=",
                       mapping={np.sqrt(1 / 4): r"\tfrac{1}{\sqrt{4}}", 1: "1", 0: "0",
                                np.sqrt(3 / 4): r"\tfrac{\sqrt{3}}{\sqrt{4}}",
                                -np.sqrt(3 / 4): r"-\tfrac{\sqrt{3}}{\sqrt{4}}"})
        t0 = 0.5 + display.write_text_in(rot2, indent=2.75, line=3, begin_time=t0, transition_time=3)

        colors = flatten([['text'] * 6, ['important'] * 22, ['text'], ['custom1']])
        trans2 = SimpleTexBObject(
            r"R_{x_3 x_4}\cdot \tfrac{1}{\sqrt{5}}\left(\begin{array}{c}0\\0\\0\\2\\1\end{array}\right) =\tfrac{1}{\sqrt{5}}\left(\begin{array}{c}0\\0\\\sqrt{3}\\1\\1\end{array}\right)",
            color=colors)

        normal.change_color(new_color='plastic_custom1', begin_time=t0, transition_time=2)
        coords.axes[2].axis_label.change_color_of_letters([0, 1, 3, 4, 6, 7], new_color='plastic_custom1',
                                                          begin_time=t0,
                                                          transition_time=2)
        projection_plane.change_color(new_color='plastic_custom1', begin_time=t0, transition_time=2)
        t0 = 0.5 + display.write_text_in(trans2, indent=2.75, line=8, begin_time=t0, transition_time=2)

        rot3 = BMatrix(recreate=False,
                       entries=np.array([[1, 0, 0, 0, 0],
                                         [0, np.sqrt(1 / 3), np.sqrt(2 / 3), 0, 0],
                                         [0, -np.sqrt(2 / 3), np.sqrt(1 / 3), 0, 0],
                                         [0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 1]]),
                       pre_word=r"R_{x_2 x_3}=",
                       mapping={np.sqrt(1 / 3): r"\tfrac{1}{\sqrt{3}}", 1: "1", 0: "0",
                                np.sqrt(2 / 3): r"\tfrac{\sqrt{3}}{\sqrt{4}}",
                                -np.sqrt(2 / 3): r"-\tfrac{\sqrt{3}}{\sqrt{4}}"})
        t0 = 0.5 + display.write_text_in(rot3, indent=5.25, line=3, begin_time=t0, transition_time=3)

        colors = flatten([['text'] * 6, ['custom1'] * 24, ['text'], ['example']])
        trans3 = SimpleTexBObject(
            r"R_{x_2 x_3}\cdot \tfrac{1}{\sqrt{5}}\left(\begin{array}{c}0\\0\\\sqrt{3}\\1\\1\end{array}\right) =\tfrac{1}{\sqrt{5}}\left(\begin{array}{c}0\\\sqrt{2}\\1\\1\\1\end{array}\right)",
            color=colors)

        normal.rotate(rotation_euler=[-np.arccos(np.sqrt(2 / 3)), 0, 0], begin_time=t0, transition_time=2)
        normal.change_color(new_color='plastic_example', begin_time=t0, transition_time=2)
        coords.axes[2].axis_label.change_color_of_letters([0, 1, 3, 4, 6, 7], new_color='plastic_example',
                                                          begin_time=t0,
                                                          transition_time=2)
        coords.axes[1].axis_label.change_color(new_color='example', begin_time=t0, transition_time=2)
        projection_plane.change_color(new_color='plastic_example', begin_time=t0, transition_time=2)
        projection_plane.rotate(rotation_euler=[-np.arccos(np.sqrt(2 / 3)), 0, 0], begin_time=t0, transition_time=2)
        t0 = 0.5 + display.write_text_in(trans3, indent=5.25, line=8, begin_time=t0, transition_time=2)

        rot4 = BMatrix(recreate=False,
                       entries=np.array([
                           [np.sqrt(1 / 2), np.sqrt(1 / 2), 0, 0, 0],
                           [-np.sqrt(1 / 2), np.sqrt(1 / 2), 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]]),
                       pre_word=r"R_{x_2 x_3}=",
                       mapping={np.sqrt(1 / 2): r"\tfrac{1}{\sqrt{2}}", 1: "1", 0: "0",
                                -np.sqrt(1 / 2): r"-\tfrac{1}{\sqrt{2}}"})
        t0 = 0.5 + display.write_text_in(rot4, indent=7.75, line=3, begin_time=t0, transition_time=3)

        colors = flatten([['text'] * 6, ['example'] * 24, ['text'], ['joker']])
        trans4 = SimpleTexBObject(
            r"R_{x_2 x_3}\cdot \tfrac{1}{\sqrt{5}}\left(\begin{array}{c}0\\\sqrt{2}\\1\\1\\1\end{array}\right) =\tfrac{1}{\sqrt{5}}\left(\begin{array}{c}1\\1\\1\\1\\1\end{array}\right)",
            color=colors)

        normal.rotate(rotation_euler=[-np.arccos(np.sqrt(2 / 3)), 0, -np.pi / 4], begin_time=t0, transition_time=2)
        normal.change_color(new_color='plastic_joker', begin_time=t0, transition_time=2)
        coords.axes[2].axis_label.change_color_of_letters([0, 1, 3, 4, 6, 7], new_color='plastic_joker',
                                                          begin_time=t0,
                                                          transition_time=2)
        coords.axes[1].axis_label.change_color(new_color='joker', begin_time=t0, transition_time=2)
        coords.axes[0].axis_label.change_color(new_color='joker', begin_time=t0, transition_time=2)
        projection_plane.change_color(new_color='plastic_joker', begin_time=t0, transition_time=2)
        projection_plane.rotate(rotation_euler=[-np.arccos(np.sqrt(2 / 3)), 0, -np.pi / 4], begin_time=t0,
                                transition_time=2)
        t0 = 0.5 + display.write_text_in(trans4, indent=7.75, line=8, begin_time=t0, transition_time=2)

        full_rot = SimpleTexBObject(r"R=R_{x_1 x_2}\cdot R_{x_2 x_3} \cdot R_{x_3 x_4} \cdot R_{x_4 x_5}")
        t0 = 0.5 + display.write_text_in(full_rot, line=11, indent=0.25, begin_time=t0, transition_time=3, scale=1)

        faders = [rot1, rot2, rot3, rot4, title, trans1, trans2, trans3, trans4]

        colors = flatten(
            [['text'], ['joker'] * 2, ['text'], ['joker'] * 2, ['text'], ['important'] * 3, ['text'], ['important'] * 3,
             ['text'], ['important'] * 3, ['text']])
        fivebeinlabels = SimpleTexBObject(
            r"(\vec{u},\vec{v},\vec{w}_1,\vec{w}_2,\vec{w}_3)=(R\cdot \vec{x}_1,R\cdot \vec{x}_2,R\cdot \vec{x}_3,R\cdot \vec{x}_4,R\cdot \vec{x}_5)",
            color=colors)

        t0 = 0.5 + display.write_text_in(fivebeinlabels, scale=1, line=13, indent=0.25, begin_time=t0,
                                         transition_time=3)

        # setup coordinate system
        fivebein = CoordinateSystem(dim=3, lengths=[10, 10, 5],
                                    domains=[[-1, 1], [-1, 1], [-1, 1]],
                                    all_n_tics=[0, 0, 0],
                                    all_tic_labels=[None, None, None],
                                    label_digits=[0, 0, 0],
                                    radii=[0.03, 0.03, 0.03],
                                    labels=[r'\vec{u}', r'\vec{v}', r'\vec{w}_1,\vec{w}_2,\vec{w}_3'],
                                    colors=['joker', 'joker', 'important'],
                                    name='FiveBein',
                                    location_of_origin=[-5, -5, -2],
                                    )

        fivebein.rotate(rotation_euler=[-np.arccos(np.sqrt(2 / 3)), 0, -np.arccos(np.sqrt(1 / 2)) - np.pi / 6],
                        begin_time=t0 + 0.1,
                        transition_time=0)
        fivebein.rescale(rescale=[0.5] * 3, begin_time=t0, transition_time=0)
        fivebein.axes[0].axis_label.rotate(rotation_euler=[0, 0, np.pi / 2], begin_time=t0 + 0.2, transition_time=0)
        fivebein.axes[1].axis_label.rotate(rotation_euler=[np.pi / 2, -np.pi / 2, 0], begin_time=t0 + 0.2,
                                           transition_time=0)
        fivebein.axes[2].axis_label.rotate(rotation_euler=[-np.pi / 2, np.pi, -np.pi / 2], begin_time=t0 + 0.2,
                                           transition_time=0)
        [coords.axes[i].axis_label.change_color(new_color='text', begin_time=t0, transition_time=5) for i in range(3)]
        normal.change_color(new_color='plastic_important', begin_time=t0, transition_time=5)
        t0 = 0.5 + fivebein.appear(begin_time=t0, transition_time=5)

        projectors = SimpleTexBObject(r'\text{Projectors:}', aligned='center', color='example')
        [fader.change_alpha(alpha=0.25, begin_time=t0, transition_time=3) for fader in faders]
        t0 = 0.5 + display.write_text_in(projectors, line=11, indent=5, scale=1, begin_time=t0, transition_time=3)

        coords.rescale(rescale=[2] * 3, begin_time=t0, transition_time=2)
        coords.move(direction=[-1, 0, 1], begin_time=t0, transition_time=2)
        fivebein.move(direction=[-1, 0, 1], begin_time=t0, transition_time=2)
        fivebein.rescale(rescale=[2] * 3, begin_time=t0, transition_time=2)
        t0 = 0.5 + display.move(direction=[0, 0, 5.6], begin_time=t0, transition_time=2)

        para = SimpleTexBObject(
            r'P_{\parallel}=\left(\begin{array}{c} \vec{u}^T\\ \vec{v}^T\end{array}\right) =\frac{1}{\sqrt{2}}\left(\begin{array}{c c c c c} 1 &-1 & 0 & 0 & 0\\ \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} & - \frac{2}{\sqrt{3}} & 0 & 0\end{array}\right) ',
            color='joker')
        t0 = 0.5 + display.write_text_in(para, line=14, indent=5, scale=1, begin_time=t0, transition_time=6)

        ortho = SimpleTexBObject(
            r'P_{\perp}=\left(\begin{array}{c} \vec{w}_1^T\\ \vec{w}_2^T\\ \vec{w}_3^T\end{array}\right) =\left(\begin{array}{c c c c c} \frac{1}{2\sqrt{3}} & \frac{1}{2\sqrt{3}}&\frac{1}{2\sqrt{3}} & -\frac{\sqrt{3}}{2} & 0 \\'
            r' \rule{0em}{2.5ex}\frac{1}{2\sqrt{5}}& \frac{1}{2\sqrt{5}}& \frac{1}{2\sqrt{5}}&\frac{1}{2\sqrt{5}}& -\frac{2}{\sqrt{5}}\\'
            r' \rule{0em}{2.5ex}\frac{1}{\sqrt{5}}&\frac{1}{\sqrt{5}}& \frac{1}{\sqrt{5}} & \frac{1}{\sqrt{5}}&\frac{1}{\sqrt{5}}\end{array}\right) ',
            color='important', recreate=False)
        t0 = 0.5 + display.write_text_in(ortho, line=19, indent=5, scale=1, begin_time=t0, transition_time=6)

        example = SimpleTexBObject(
            r'P_{\perp}\cdot \tfrac{1}{\sqrt{5}}\left(\begin{array}{c} 1\\1\\1\\1\\1\end{array}\right)=\left(\begin{array}{c} 0\\0\\1\end{array}\right)'
            , color='text')
        t0 = 0.5 + display.write_text_in(example, line=24, indent=5, begin_time=t0, transition_time=3)

        self.t0 = t0

    def plane_rotation_3d(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        camera_location = [0, -20, 1.5]
        ibpy.set_camera_location(location=camera_location)
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # setup coordinate system
        coords = CoordinateSystem(dim=3, lengths=[7.5, 7.5, 5], domains=[[0, 3], [0, 3], [-1, 1]],
                                  all_n_tics=[3, 3, 2],
                                  all_tic_lablls=[np.arange(-0, 3.1, 1), np.arange(-0, 3.1, 1),
                                                  np.arange(-1, 1.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015]
                                  )

        rotation_center = coords.coords2location([2, 2, 0])
        coords.move(direction=-rotation_center, begin_time=0, transition_time=0)
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        unit = coords.coords2location(Vector([1, 0, 0]))[0]

        projection_plane = Plane(location=[0, 0, 0], rotation_euler=[0, 0, 0], scale=[2 * unit] * 3,
                                 name="Plane_Projection",
                                 color="plastic_drawing")

        t0 = 0.5 + projection_plane.appear(begin_time=t0, transition_time=1, alpha=0.5)
        coords.add_object(projection_plane)

        # normal
        normal = PArrow(start=[0, 0, 0], end=[0, 0, unit], color='plastic_drawing', thickness=2)
        t0 = 0.5 + normal.grow(begin_time=t0, transition_time=1)
        coords.add_object(normal)

        display = Display(flat=True, number_of_lines=18,
                          color='gray_2', location=[6, 0, 0],
                          rotation_euler=[-np.pi / 2 + np.arctan2(camera_location[2],
                                                                  camera_location[1]), 0, 0])
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1)

        title = SimpleTexBObject(r"\text{Projection plane in 3D}", color='example', aligned="center")
        t0 = 0.5 + display.write_title(title, begin_time=t0, transition_time=3)

        t0 = 0.5 + coords.rotate(rotation_euler=[0, 0, -np.pi / 4],
                                 compensate=[True, False, True], begin_time=t0, transition_time=1)

        rot1 = BMatrix(
            entries=np.array([[1, 0, 0], [0, np.sqrt(1 / 3), np.sqrt(2 / 3)], [0, -np.sqrt(2 / 3), np.sqrt(1 / 3)]]),
            pre_word=r"R_{yz}=",
            mapping={np.sqrt(1 / 3): r"\tfrac{1}{\sqrt{3}}",
                     1: "1",
                     0: "0",
                     np.sqrt(2 / 3): r"\tfrac{\sqrt{2}}{\sqrt{3}}",
                     -np.sqrt(2 / 3): r"-\tfrac{\sqrt{2}}{\sqrt{3}}"})
        t0 = 0.5 + display.write_text_in(rot1, indent=0.5, line=2, begin_time=t0, transition_time=3)

        colors = flatten([['text'] * 4, ['drawing'] * 7, ['text'], ['important']])
        trans1 = SimpleTexBObject(
            r"R_{yz}\cdot \left(\begin{array}{c}0\\0\\1\end{array}\right) =\tfrac{1}{\sqrt{3}}\left(\begin{array}{c}0\\\sqrt{2}\\1\end{array}\right)",
            color=colors)

        normal.rotate(rotation_euler=[-np.arccos(1 / np.sqrt(3)), 0, 0], begin_time=t0, transition_time=2)
        normal.change_color(new_color='plastic_important', begin_time=t0, transition_time=2)
        projection_plane.rotate(rotation_euler=[-np.arccos(1 / np.sqrt(3)), 0, 0], begin_time=t0, transition_time=2)
        projection_plane.change_color(new_color='plastic_important', begin_time=t0, transition_time=2)
        t0 = 0.5 + display.write_text_in(trans1, indent=1, line=6, begin_time=t0, transition_time=2)

        r2 = np.sqrt(2)
        rot2 = BMatrix(
            entries=np.array([[1, 0, 0], [0, 1 / r2, 1 / r2], [0, -1 / r2, 1 / r2]]),
            pre_word=r"R_{xy}=",
            mapping={1 / r2: r"\tfrac{1}{\sqrt{2}}",
                     -1 / r2: r"\tfrac{1}{\sqrt{2}}",
                     1: "1",
                     0: "0",
                     })
        t0 = 0.5 + display.write_text_in(rot2, indent=0.5, line=10, begin_time=t0, transition_time=3)

        colors = flatten([['text'] * 4, ['important'] * 14, ['text'], ['joker']])
        trans2 = SimpleTexBObject(
            r"R_{xy}\cdot \tfrac{1}{\sqrt{3}}\left(\begin{array}{c}0\\\sqrt{2}\\1\end{array}\right)=\tfrac{1}{\sqrt{3}}\left(\begin{array}{c}1\\1\\1\end{array}\right) ",
            color=colors, recreate=False)

        normal.rotate(rotation_euler=[-np.arccos(1 / np.sqrt(3)), 0, -np.arccos(1 / np.sqrt(2))], begin_time=t0,
                      transition_time=2)
        normal.change_color(new_color='plastic_joker', begin_time=t0, transition_time=2)
        projection_plane.rotate(rotation_euler=[-np.arccos(1 / np.sqrt(3)), 0, -np.arccos(1 / np.sqrt(2))],
                                begin_time=t0, transition_time=2)
        projection_plane.change_color(new_color='plastic_joker', begin_time=t0, transition_time=2)
        t0 = 0.5 + display.write_text_in(trans2, indent=1, line=14, begin_time=t0, transition_time=2)

        t0 = 0.5 + coords.rotate(rotation_euler=[0, 0, -3 / 4 * np.pi], begin_time=t0, transition_time=5)

        self.t0 = t0

    def trivial_projection_3d4(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[0, -20, 1.5])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # setup coordinate system
        coords = CoordinateSystem(dim=3, lengths=[10, 10, 10], domains=[[0, 4], [0, 4], [-2, 2]],
                                  all_n_tics=[4, 4, 4],
                                  all_tic_lablls=[np.arange(-0, 4.1, 1), np.arange(-0, 4.1, 1), np.arange(-2, 2.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015]
                                  )

        rotation_center = coords.coords2location([2, 2, 0])
        coords.move(direction=-rotation_center, begin_time=0, transition_time=0)

        t0 = coords.appear(begin_time=t0, transition_time=0)

        sphere = Sphere(0.1, location=coords.coords2location([0, 0, -size / 2]), color='example', name="vertex")
        spacing = 12.5
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, spacing, 0])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, 0, spacing])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[spacing, 0, 0])
        ibpy.apply_modifiers(sphere)
        ibpy.set_origin(sphere, type='ORIGIN_CENTER_OF_VOLUME')
        ibpy.separate(sphere, type='LOOSE')
        coords.add_object(sphere)

        positions = []
        for x in range(0, size + 1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for y in range(0, size + 1):
                    positions.append((x, y, z))  # use tuples to be able to construct sets

        num_spheres = (size + 1) ** 3
        names = ['vertex.' + str(i).zfill(3) for i in range(1, np.minimum(1000, num_spheres))]
        names.insert(0, 'vertex')
        if num_spheres > 1000:
            names2 = ['vertex.' + str(i) for i in range(1000, num_spheres)]
            names = names + names2
        spheres = [BObject.from_name(name, scale=[1, 1, 1], color='example') for name in names]
        for sphere in spheres:
            sphere.appear(begin_time=t0, transition_time=0)
            coords.add_object(sphere)

        [sphere.rename(name="pos_" + str(pos[0:3])) for sphere, pos in zip(spheres, positions)]

        # create map between position and sphere, this is a bit tricky, since the spheres are created and located by the
        # array modifier
        position_sphere_map = {pos: sphere for pos, sphere in zip(positions, spheres)}

        unit = coords.coords2location((1, 0, 0))[0]
        projection_plane = Plane(location=[2 * unit, 2 * unit, 0], rotation_euler=[0, 0, 0], scale=[2 * unit] * 3,
                                 name="Plane_Projection",
                                 color="plastic_joker")
        # projection_plane.move(direction=[-2*unit,-2*unit,0],begin_time=0,transition_time=0)
        t0 = projection_plane.appear(begin_time=t0, transition_time=0, alpha=0.5)
        coords.add_object(projection_plane)
        coords.disappear_axes(begin_time=0, transition_time=1)

        # select spheres
        for y in range(size, -1, -1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for x in range(0, size + 1):
                    pos = (x, y, z)
                    sphere = position_sphere_map[pos]
                    if pos[2] == 0:
                        sphere.change_color(new_color='plastic_drawing', begin_time=t0, transition_time=0)
                    else:
                        sphere.change_color(new_color='plastic_important', begin_time=t0, transition_time=0)
                        sphere.change_alpha(begin_time=t0, transition_time=0, alpha=0.05)

        [sphere.disappear(begin_time=0.1, transition_time=1) for sphere in spheres]
        # create all faces
        units = [Vector(u) for u in unit_tuples(3)]

        projected_faces = []
        for pos in positions:
            if pos[0] < 4 and pos[1] < 4:
                projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[1]]),
                                               color='plastic_drawing', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
                                               scale=[unit] * 3, name="projection_01_" + str(pos)))

        for pos in positions:
            if pos[0] < 4 and pos[2] < 2:
                projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[2]]),
                                               color='plastic_important', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
                                               scale=[unit] * 3, name="projection_02_" + str(pos)))

        for pos in positions:
            if pos[1] < 4 and pos[2] < 2:
                projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[1], units[2]]),
                                               color='plastic_custom2', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
                                               scale=[unit] * 3, name="projection_12_" + str(pos)))

        [face.appear(begin_time=t0, transition_time=0.5) for face in projected_faces]
        [ibpy.set_parent(face, projection_plane) for face in projected_faces]

        # dynamically adjust points in the zone
        count = 0
        face_visibility_map = {}
        for projected_face in projected_faces:
            projected_face.appear(begin_time=t0, transition_time=0)

            if all([v.z == 0 for v in projected_face.vertices0]):
                face_visibility_map[projected_face] = True
                print("face: ", projected_face.name, " displayed from the beginning")
                count += 1
            else:
                face_visibility_map[projected_face] = False
                projected_face.toggle_hide(begin_time=t0)
                print("face: ", projected_face.name, " hidden from the beginning")

        map_selected_positions = {pos: pos[2] == 0 for pos in positions}

        voronoi_cell = tuples([0.5, -0.5], 3)
        steps = 60
        alpha_y = np.arcsin(np.sqrt(2 / 3))
        alpha_z = np.arcsin(np.sqrt(1 / 2))
        duration = 20

        delta_y = alpha_y / steps
        delta_z = alpha_z / steps
        delta_t = duration / steps

        for s in range(0, steps + 1):
            alpha_y = s * delta_y
            alpha_z = s * delta_z
            t = t0 + s * delta_t
            rot1 = Matrix([[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
            rot2 = Matrix([[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
            rot = rot2 @ rot1

            # locale base
            base = [Vector(rot @ u) for u in units]

            projected_voronoi = [base[2].dot(v) for v in voronoi_cell]
            inf = min(projected_voronoi)
            sup = max(projected_voronoi)

            new_map_selected_positions = {pos: inf <= base[2].dot(Vector(pos) - Vector([2, 2, 0])) <= sup for pos in
                                          positions}
            print(len({k: v for k, v in new_map_selected_positions.items() if v}))

            map_selected_positions = new_map_selected_positions
            active_point_set = set([key for key, val in map_selected_positions.items() if val])

            # adjust the projection of all faces
            plane_center = Vector([2, 2, 0])
            [poly.morph_to(projector=lambda v: (
                    Vector([base[0].dot(v - plane_center), base[1].dot(v - plane_center), 0]) + plane_center),
                           begin_time=t, transition_time=delta_t) for poly in projected_faces]

            # adjust visibility if necessary
            for projected_face in projected_faces:
                if set([tuple(v) for v in projected_face.vertices0]).issubset(active_point_set):
                    if not face_visibility_map[projected_face]:
                        projected_face.toggle_hide(begin_time=t)
                        face_visibility_map[projected_face] = True
                        print(projected_face.name, " appears")

                else:
                    if face_visibility_map[projected_face]:
                        projected_face.toggle_hide(begin_time=t)
                        face_visibility_map[projected_face] = False
                        print(projected_face.name, " disappears")

        t0 += 0.5 + duration
        rot_box2 = BObject(children=[coords], name="CoordinateSystem_box", location=rotation_center)
        rot_box2.appear(begin_time=0, transition_time=0)
        rot_box2.move(direction=-rotation_center + Vector(), begin_time=0, transition_time=0)
        rot_box2.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=0, transition_time=1)

        self.t0 = t0

    def trivial_projection_3d3(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[0, -20, 1.5])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # setup coordinate system
        coords = CoordinateSystem(dim=3, lengths=[10, 10, 10], domains=[[0, 4], [0, 4], [-2, 2]],
                                  all_n_tics=[4, 4, 4],
                                  all_tic_lablls=[np.arange(-0, 4.1, 1), np.arange(-0, 4.1, 1), np.arange(-2, 2.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015]
                                  )

        rotation_center = coords.coords2location([2, 2, 0])
        coords.move(direction=-rotation_center, begin_time=0, transition_time=0)

        t0 = coords.appear(begin_time=t0, transition_time=0)

        sphere = Sphere(0.1, location=coords.coords2location([0, 0, -size / 2]), color='example', name="vertex")
        spacing = 12.5
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, spacing, 0])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, 0, spacing])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[spacing, 0, 0])
        ibpy.apply_modifiers(sphere)
        ibpy.set_origin(sphere, type='ORIGIN_CENTER_OF_VOLUME')
        ibpy.separate(sphere, type='LOOSE')
        coords.add_object(sphere)

        positions = []
        for x in range(0, size + 1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for y in range(0, size + 1):
                    positions.append((x, y, z))  # use tuples to be able to construct sets

        num_spheres = (size + 1) ** 3
        names = ['vertex.' + str(i).zfill(3) for i in range(1, np.minimum(1000, num_spheres))]
        names.insert(0, 'vertex')
        if num_spheres > 1000:
            names2 = ['vertex.' + str(i) for i in range(1000, num_spheres)]
            names = names + names2
        spheres = [BObject.from_name(name, scale=[1, 1, 1], color='example') for name in names]
        for sphere in spheres:
            sphere.appear(begin_time=t0, transition_time=0)
            coords.add_object(sphere)

        [sphere.rename(name="pos_" + str(pos[0:3])) for sphere, pos in zip(spheres, positions)]

        # create map between position and sphere, this is a bit tricky, since the spheres are created and located by the
        # array modifier
        position_sphere_map = {pos: sphere for pos, sphere in zip(positions, spheres)}

        unit = coords.coords2location((1, 0, 0))[0]
        projection_plane = Plane(rotation_euler=[0, 0, 0], scale=[2 * unit] * 3, name="Plane_Projection",
                                 color="plastic_joker")
        t0 = projection_plane.appear(begin_time=t0, transition_time=0, alpha=0.5)
        coords.add_object(projection_plane)

        # select spheres
        for y in range(size, -1, -1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for x in range(0, size + 1):
                    pos = (x, y, z)
                    sphere = position_sphere_map[pos]
                    if pos[2] == 0:
                        sphere.change_color(new_color='plastic_drawing', begin_time=t0, transition_time=0.5)
                    else:
                        sphere.change_color(new_color='plastic_important', begin_time=t0, transition_time=0.5)
                        sphere.change_alpha(begin_time=t0, transition_time=0, alpha=0.05)

        # create all faces
        units = [Vector(u) for u in unit_tuples(3)]

        projected_faces = []
        for pos in positions:
            if pos[0] < 4 and pos[1] < 4:
                projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[1]]),
                                               color='plastic_drawing', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
                                               scale=[unit] * 3, name="projection_01_" + str(pos)))

        for pos in positions:
            if pos[0] < 4 and pos[2] < 2:
                projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[2]]),
                                               color='plastic_important', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
                                               scale=[unit] * 3, name="projection_02_" + str(pos)))

        for pos in positions:
            if pos[1] < 4 and pos[2] < 2:
                projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[1], units[2]]),
                                               color='plastic_custom2', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
                                               scale=[unit] * 3, name="projection_12_" + str(pos)))

        [face.appear(begin_time=t0, transition_time=0.5) for face in projected_faces]
        coords.add_objects(projected_faces)

        # rotate plane
        rotation_duration = 20
        rot_box = BObject(children=[projection_plane],
                          location=[2 * unit, 2 * unit, 0])
        rot_box.appear(begin_time=t0, transition_time=0)
        coords.add_object(rot_box)
        rot_box.rotate(rotation_euler=[0, np.arcsin(np.sqrt(2 / 3)), np.pi / 4], begin_time=t0,
                       transition_time=rotation_duration,
                       pivot=Vector([2 * unit, 2 * unit, 0]))
        # adjust for diagonal stretching
        rot_box.rescale(rescale=[1.5, 1.5, 1], begin_time=t0, transition_time=rotation_duration)

        # dynamically adjust points in the zone

        count = 0
        face_visibility_map = {}
        for projected_face in projected_faces:
            projected_face.appear(begin_time=t0, transition_time=0)

            if all([v.z == 0 for v in projected_face.vertices0]):
                face_visibility_map[projected_face] = True
                print("face: ", projected_face.name, " displayed from the beginning")
                count += 1
            else:
                face_visibility_map[projected_face] = False
                projected_face.toggle_hide(begin_time=t0)
                print("face: ", projected_face.name, " hidden from the beginning")

        map_selected_positions = {pos: pos[2] == 0 for pos in positions}

        voronoi_cell = tuples([0.5, -0.5], 3)
        delta = 1

        for frame in range(int(t0 * FRAME_RATE), int(t0 + rotation_duration) * FRAME_RATE + 1, delta):
            tstart = frame / FRAME_RATE
            duration = delta / FRAME_RATE
            # get rotation angle
            rotation = ibpy.get_rotation_at_frame(rot_box, frame + delta)
            # calculate_normal
            alpha_y = rotation[1]
            alpha_z = rotation[2]
            rot1 = Matrix([[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
            rot2 = Matrix([[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
            rot = rot2 @ rot1

            # locale base
            base = [Vector(rot @ u) for u in units]

            projected_voronoi = [base[2].dot(v) for v in voronoi_cell]
            inf = min(projected_voronoi)
            sup = max(projected_voronoi)
            print(frame, ": ", inf, sup)
            new_map_selected_positions = {pos: inf <= base[2].dot(Vector(pos) - Vector([2, 2, 0])) <= sup for pos in
                                          positions}
            print(len({k: v for k, v in new_map_selected_positions.items() if v}))
            for pos in positions:
                if new_map_selected_positions[pos] and map_selected_positions[pos]:
                    pass
                elif new_map_selected_positions[pos]:
                    # make position visible and turn blue
                    position_sphere_map[pos].change_alpha(alpha=1, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_drawing', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned visible")
                elif map_selected_positions[pos]:
                    # make position invisible and red
                    position_sphere_map[pos].change_alpha(alpha=0.05, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_important', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned invisible")

            map_selected_positions = new_map_selected_positions
            active_point_set = set([key for key, val in map_selected_positions.items() if val])

            # adjust the projection of all faces
            plane_center = Vector([2, 2, 0])
            [poly.morph_to(projector=lambda v: (
                    base[0].dot(v - plane_center) * base[0] + base[1].dot(v - plane_center) * base[
                1] + plane_center),
                           begin_time=tstart, transition_time=duration) for poly in projected_faces]

            # adjust visibility if necessary
            for projected_face in projected_faces:
                if set([tuple(v) for v in projected_face.vertices0]).issubset(active_point_set):
                    if not face_visibility_map[projected_face]:
                        projected_face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[projected_face] = True
                        print(projected_face.name, " appears")

                else:
                    if face_visibility_map[projected_face]:
                        projected_face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[projected_face] = False
                        print(projected_face.name, " disappears")
        t0 += 0.5 + rotation_duration

        rot_box.rotate(rotation_euler=[0, 0, 0], begin_time=t0,
                       transition_time=rotation_duration,
                       pivot=Vector([2 * unit, 2 * unit, 0]))
        # adjust for diagonal stretching
        rot_box.rescale(rescale=[1 / 1.5, 1 / 1.5, 1], begin_time=t0, transition_time=rotation_duration)

        for frame in range(int(t0 * FRAME_RATE), int(t0 + rotation_duration) * FRAME_RATE + 1, delta):
            tstart = frame / FRAME_RATE
            duration = delta / FRAME_RATE
            # get rotation angle
            rotation = ibpy.get_rotation_at_frame(rot_box, frame + delta)
            # calculate_normal
            alpha_y = rotation[1]
            alpha_z = rotation[2]
            rot1 = -1 * Matrix(
                [[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
            rot2 = -1 * Matrix(
                [[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
            rot = rot2 @ rot1

            # locale base
            base = [Vector(rot @ u) for u in units]

            projected_voronoi = [base[2].dot(v) for v in voronoi_cell]
            inf = min(projected_voronoi)
            sup = max(projected_voronoi)
            print(frame, ": ", inf, sup)
            new_map_selected_positions = {pos: inf <= base[2].dot(Vector(pos) - Vector([2, 2, 0])) <= sup for pos in
                                          positions}

            for pos in positions:
                if new_map_selected_positions[pos] and map_selected_positions[pos]:
                    pass
                elif new_map_selected_positions[pos]:
                    # make position visible and turn blue
                    position_sphere_map[pos].change_alpha(alpha=1, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_drawing', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned visible")
                elif map_selected_positions[pos]:
                    # make position invisible and red
                    position_sphere_map[pos].change_alpha(alpha=0.05, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_important', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned invisible")

            map_selected_positions = new_map_selected_positions
            active_point_set = set([key for key, val in map_selected_positions.items() if val])
            print("active points: ", len(active_point_set))

            plane_center = Vector([2, 2, 0])
            [poly.morph_to(projector=lambda v: (
                    base[0].dot(v - plane_center) * base[0] + base[1].dot(v - plane_center) * base[
                1] + plane_center),
                           begin_time=tstart, transition_time=duration) for poly in projected_faces]

            # adjust visibility if necessary
            for projected_face in projected_faces:
                if set([tuple(v) for v in projected_face.vertices0]).issubset(active_point_set):
                    if not face_visibility_map[projected_face]:
                        projected_face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[projected_face] = True
                        print(projected_face.name, " appears")

                else:
                    if face_visibility_map[projected_face]:
                        projected_face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[projected_face] = False
                        print(projected_face.name, " disappears")

        t0 += 0.5 + rotation_duration

        rot_box2 = BObject(children=[coords], name="CoordinateSystem_box", location=rotation_center)
        rot_box2.appear(begin_time=0, transition_time=0)
        rot_box2.move(direction=-rotation_center + Vector(), begin_time=0, transition_time=0)
        # rot_box2.move(direction=[5, 0, 0], begin_time=coords_move_back_time, transition_time=2)

        rot_box2.rotate(rotation_euler=[0, 0, -2 * np.pi], begin_time=0, transition_time=t0)

        self.t0 = t0

    def trivial_projection_3d2(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[0, -20, 1.5])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # setup coordinate system
        coords = CoordinateSystem(dim=3, lengths=[10, 10, 10], domains=[[0, 4], [0, 4], [-2, 2]],
                                  all_n_tics=[4, 4, 4],
                                  all_tic_lablls=[np.arange(-0, 4.1, 1), np.arange(-0, 4.1, 1), np.arange(-2, 2.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015]
                                  )

        rotation_center = coords.coords2location([2, 2, 0])
        coords.move(direction=-rotation_center, begin_time=0, transition_time=0)

        # empty.move(direction=[0,0,1.5],begin_time=t0,transition_time=5)
        t0 = coords.appear(begin_time=t0, transition_time=0)

        sphere = Sphere(0.1, location=coords.coords2location([0, 0, -size / 2]), color='example', name="vertex")
        spacing = 12.5
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, spacing, 0])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, 0, spacing])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[spacing, 0, 0])
        ibpy.apply_modifiers(sphere)
        ibpy.set_origin(sphere, type='ORIGIN_CENTER_OF_VOLUME')
        ibpy.separate(sphere, type='LOOSE')
        coords.add_object(sphere)

        positions = []
        for x in range(0, size + 1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for y in range(0, size + 1):
                    positions.append((x, y, z))  # use tuples to be able to construct sets

        num_spheres = (size + 1) ** 3
        names = ['vertex.' + str(i).zfill(3) for i in range(1, np.minimum(1000, num_spheres))]
        names.insert(0, 'vertex')
        if num_spheres > 1000:
            names2 = ['vertex.' + str(i) for i in range(1000, num_spheres)]
            names = names + names2
        spheres = [BObject.from_name(name, scale=[1, 1, 1], color='example') for name in names]
        for sphere in spheres:
            sphere.appear(begin_time=t0, transition_time=0)
            coords.add_object(sphere)

        [sphere.rename(name="pos_" + str(pos[0:3])) for sphere, pos in zip(spheres, positions)]

        # create map between position and sphere, this is a bit tricky, since the spheres are created and located by the
        # array modifier
        position_sphere_map = {pos: sphere for pos, sphere in zip(positions, spheres)}

        unit = coords.coords2location((1, 0, 0))[0]
        projection_plane = Plane(rotation_euler=[0, 0, 0], scale=[2 * unit] * 3, name="Plane_Projection",
                                 color="plastic_joker")
        t0 = projection_plane.appear(begin_time=t0, transition_time=0, alpha=0.5)
        coords.add_object(projection_plane)

        # select spheres
        for y in range(size, -1, -1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for x in range(0, size + 1):
                    pos = (x, y, z)
                    sphere = position_sphere_map[pos]
                    if pos[2] == 0:
                        sphere.change_color(new_color='plastic_drawing', begin_time=t0, transition_time=0)
                    else:
                        sphere.change_color(new_color='plastic_important', begin_time=t0, transition_time=0)
                        sphere.change_alpha(begin_time=t0, transition_time=0, alpha=0.05)

        t0 += 2.5

        # create all faces
        units = [Vector(u) for u in unit_tuples(3)]

        unprojected_faces = []
        for pos in positions:
            if pos[0] < 4 and pos[1] < 4:
                unprojected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[1]]),
                                                 transmission=0, color='plastic_drawing', name="plane_01_" + str(pos),
                                                 scale=[unit] * 3))

        for pos in positions:
            if pos[0] < 4 and pos[2] < 2:
                unprojected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[2]]),
                                                 transmission=0, color='plastic_important', name="plane_02_" + str(pos),
                                                 scale=[unit] * 3))

        for pos in positions:
            if pos[1] < 4 and pos[2] < 2:
                unprojected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[1], units[2]]),
                                                 transmission=0, color='plastic_custom2', name="plane_12_" + str(pos),
                                                 scale=[unit] * 3))

        [face.appear(begin_time=t0, transition_time=0.5) for face in unprojected_faces]
        [face.change_alpha(alpha=0.7, begin_time=t0 + 0.1, transition_time=0.5) for face in unprojected_faces]
        coords.add_objects(unprojected_faces)

        # rotate plane
        rotation_duration = 20
        rot_box = BObject(children=[projection_plane],
                          location=[2 * unit, 2 * unit, 0])
        rot_box.appear(begin_time=t0, transition_time=0)
        coords.add_object(rot_box)
        rot_box.rotate(rotation_euler=[0, np.arcsin(np.sqrt(2 / 3)), np.pi / 4], begin_time=t0,
                       transition_time=rotation_duration,
                       pivot=Vector([2 * unit, 2 * unit, 0]))
        # adjust for diagonal stretching
        rot_box.rescale(rescale=[1.5, 1.5, 1], begin_time=t0, transition_time=rotation_duration)

        # dynamically adjust points in the zone
        count = 0
        face_visibility_map = {}
        for face in unprojected_faces:
            face.appear(begin_time=t0, transition_time=0.1)

            if all([v.z == 0 for v in face.vertices0]):
                face_visibility_map[face] = True
                print("face: ", face.name, " displayed from the beginning")
                count += 1
            else:
                face_visibility_map[face] = False
                face.toggle_hide(begin_time=t0)
                print("face: ", face.name, " hidden from the beginning")

        map_selected_positions = {pos: pos[2] == 0 for pos in positions}

        voronoi_cell = tuples([0.5, -0.5], 3)
        delta = 10

        for frame in range(int(t0 * FRAME_RATE), int(t0 + rotation_duration) * FRAME_RATE + 1, delta):
            tstart = frame / FRAME_RATE
            duration = delta / FRAME_RATE
            # get rotation angle
            rotation = ibpy.get_rotation_at_frame(rot_box, frame + delta)
            # calculate_normal
            alpha_y = rotation[1]
            alpha_z = rotation[2]
            rot1 = Matrix([[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
            rot2 = Matrix([[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
            rot = rot2 @ rot1

            # locale base
            base = [Vector(rot @ u) for u in units]

            projected_voronoi = [base[2].dot(v) for v in voronoi_cell]
            inf = min(projected_voronoi)
            sup = max(projected_voronoi)
            print(frame, ": ", inf, sup)
            new_map_selected_positions = {pos: inf <= base[2].dot(Vector(pos) - Vector([2, 2, 0])) <= sup for pos in
                                          positions}
            print(len({k: v for k, v in new_map_selected_positions.items() if v}))
            for pos in positions:
                if new_map_selected_positions[pos] and map_selected_positions[pos]:
                    pass
                elif new_map_selected_positions[pos]:
                    # make position visible and turn blue
                    position_sphere_map[pos].change_alpha(alpha=1, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_drawing', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned visible")
                elif map_selected_positions[pos]:
                    # make position invisible and red
                    position_sphere_map[pos].change_alpha(alpha=0.05, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_important', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned invisible")

            map_selected_positions = new_map_selected_positions
            active_point_set = set([key for key, val in map_selected_positions.items() if val])
            print("active points: ", len(active_point_set))
            # adjust the projection of all faces

            plane_center = Vector([2, 2, 0])

            # adjust visibility if necessary
            for face in unprojected_faces:
                if set([tuple(v) for v in face.vertices0]).issubset(active_point_set):
                    if not face_visibility_map[face]:
                        face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[face] = True
                        print(face.name, " appears")

                else:
                    if face_visibility_map[face]:
                        face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[face] = False
                        print(face.name, " disappears")
        t0 += 0.5 + rotation_duration

        rot_box.rotate(rotation_euler=[0, 0, 0], begin_time=t0,
                       transition_time=rotation_duration,
                       pivot=Vector([2 * unit, 2 * unit, 0]))
        # adjust for diagonal stretching
        rot_box.rescale(rescale=[1 / 1.5, 1 / 1.5, 1], begin_time=t0, transition_time=rotation_duration)

        for frame in range(int(t0 * FRAME_RATE), int(t0 + rotation_duration) * FRAME_RATE + 1, delta):
            tstart = frame / FRAME_RATE
            duration = delta / FRAME_RATE
            # get rotation angle
            rotation = ibpy.get_rotation_at_frame(rot_box, frame + delta)
            # calculate_normal
            alpha_y = rotation[1]
            alpha_z = rotation[2]
            rot1 = -1 * Matrix(
                [[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
            rot2 = -1 * Matrix(
                [[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
            rot = rot2 @ rot1

            # locale base
            base = [Vector(rot @ u) for u in units]

            projected_voronoi = [base[2].dot(v) for v in voronoi_cell]
            inf = min(projected_voronoi)
            sup = max(projected_voronoi)
            print(frame, ": ", inf, sup)
            new_map_selected_positions = {pos: inf <= base[2].dot(Vector(pos) - Vector([2, 2, 0])) <= sup for pos in
                                          positions}
            print(len({k: v for k, v in new_map_selected_positions.items() if v}))
            for pos in positions:
                if new_map_selected_positions[pos] and map_selected_positions[pos]:
                    pass
                elif new_map_selected_positions[pos]:
                    # make position visible and turn blue
                    position_sphere_map[pos].change_alpha(alpha=1, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_drawing', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned visible")
                elif map_selected_positions[pos]:
                    # make position invisible and red
                    position_sphere_map[pos].change_alpha(alpha=0.05, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_important', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned invisible")

            map_selected_positions = new_map_selected_positions
            active_point_set = set([key for key, val in map_selected_positions.items() if val])
            print("active points: ", len(active_point_set))

            # adjust visibility if necessary
            for face in unprojected_faces:
                if set([tuple(v) for v in face.vertices0]).issubset(active_point_set):
                    if not face_visibility_map[face]:
                        face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[face] = True
                        print(face.name, " appears")

                else:
                    if face_visibility_map[face]:
                        face.toggle_hide(begin_time=frame / FRAME_RATE)
                        face_visibility_map[face] = False
                        print(face.name, " disappears")

        t0 += 0.5 + rotation_duration

        rot_box2 = BObject(children=[coords], name="CoordinateSystem_box", location=rotation_center,
                           rotation_euler=[0, 0, -np.pi])
        rot_box2.appear(begin_time=0, transition_time=0)
        rot_box2.move(direction=-rotation_center + Vector(), begin_time=0, transition_time=0)

        rot_box2.rotate(rotation_euler=[0, 0, -4 * np.pi], begin_time=0, transition_time=t0)

        self.t0 = t0

    def trivial_projection_3d(self):
        t0 = 0
        size = 4
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, np.pi / 2, -np.pi / 9])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False, denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[0, -20, 1.5])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # setup coordinate system
        coords = CoordinateSystem(dim=3, lengths=[10, 10, 10], domains=[[0, 4], [0, 4], [-2, 2]],
                                  all_n_tics=[4, 4, 4],
                                  all_tic_lablls=[np.arange(-0, 4.1, 1), np.arange(-0, 4.1, 1), np.arange(-2, 2.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015]
                                  )

        rotation_center = coords.coords2location([2, 2, 0])
        coords.move(direction=-rotation_center, begin_time=0, transition_time=0)

        # empty.move(direction=[0,0,1.5],begin_time=t0,transition_time=5)
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=5)

        sphere = Sphere(0.1, location=coords.coords2location([0, 0, -size / 2]), color='example', name="vertex")
        spacing = 12.5
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, spacing, 0])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[0, 0, spacing])
        sphere.add_mesh_modifier(type='ARRAY', count=size + 1, relative_offset_displace=[spacing, 0, 0])
        ibpy.apply_modifiers(sphere)
        ibpy.set_origin(sphere, type='ORIGIN_CENTER_OF_VOLUME')
        ibpy.separate(sphere, type='LOOSE')
        coords.add_object(sphere)

        positions = []
        for x in range(0, size + 1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for y in range(0, size + 1):
                    positions.append((x, y, z))  # use tuples to be able to construct sets

        num_spheres = (size + 1) ** 3
        names = ['vertex.' + str(i).zfill(3) for i in range(1, np.minimum(1000, num_spheres))]
        names.insert(0, 'vertex')
        if num_spheres > 1000:
            names2 = ['vertex.' + str(i) for i in range(1000, num_spheres)]
            names = names + names2
        spheres = [BObject.from_name(name, scale=[1, 1, 1], color='example') for name in names]
        for sphere in spheres:
            sphere.appear(begin_time=t0, transition_time=0.5)
            coords.add_object(sphere)
            t0 += 0.01

        [sphere.rename(name="pos_" + str(pos[0:3])) for sphere, pos in zip(spheres, positions)]

        t0 += 0.5
        # create map between position and sphere, this is a bit tricky, since the spheres are created and located by the
        # array modifier
        position_sphere_map = {pos: sphere for pos, sphere in zip(positions, spheres)}

        unit = coords.coords2location((1, 0, 0))[0]
        projection_plane = Plane(rotation_euler=[0, 0, 0], scale=[2 * unit] * 3, name="Plane_Projection",
                                 color="plastic_joker")
        t0 = 0.5 + projection_plane.appear(begin_time=t0, transition_time=1, alpha=0.5)
        coords.add_object(projection_plane)

        scatter_cube = Cube(scale=[2 * unit, 2 * unit, 0.5 * unit],
                            color='scatter_volume', name='ScatterCube')
        ibpy.set_volume_scatter(scatter_cube, value=0.3, begin_time=t0)
        ibpy.set_volume_absorption(scatter_cube, value=0, begin_time=t0)

        scatter_cube.appear(begin_time=t0, transition_time=0)
        coords.add_object(scatter_cube)

        light_zone1 = AreaLight(location=[-2 * unit, 0, 0], name="light_zone1",
                                rotation_euler=[np.pi / 2, 0, -np.pi / 2], shape='RECTANGLE',
                                color="plastic_joker", diffuse_factor=0, energy=100, specular_factor=0, size=4 * unit,
                                size_y=1.5 * unit)
        light_zone1.appear(begin_time=t0, transition_time=1)

        light_zone2 = AreaLight(location=[0, -2 * unit, 0], name="light_zone2", rotation_euler=[0, 0, 0],
                                color="plastic_joker", shape='RECTANGLE', diffuse_factor=0, energy=100,
                                specular_factor=0, size=4 * unit, size_y=1.5 * unit)
        light_zone2.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=0, transition_time=0, pivot=[2 * unit, 0, 0])
        light_zone2.appear(begin_time=t0, transition_time=1)

        light_zone3 = AreaLight(location=[2 * unit, 0, 0], name="light_zone3", rotation_euler=[np.pi / 2, 0, np.pi / 2],
                                shape='RECTANGLE',
                                color="plastic_joker", diffuse_factor=0, energy=100, specular_factor=0, size=4 * unit,
                                size_y=1.5 * unit)
        t0 = 0.5 + light_zone3.appear(begin_time=t0, transition_time=1)

        coords.add_objects([light_zone1, light_zone2, light_zone3])

        # select spheres
        for y in range(size, -1, -1):
            for z in range(-int(size / 2), int(size / 2) + 1):
                for x in range(0, size + 1):
                    pos = (x, y, z)
                    sphere = position_sphere_map[pos]
                    if pos[2] == 0:
                        sphere.change_color(new_color='plastic_drawing', begin_time=t0, transition_time=0.5)
                    else:
                        sphere.change_color(new_color='plastic_important', begin_time=t0, transition_time=0.5)
                        sphere.change_alpha(begin_time=t0 + 1, transition_time=1, alpha=0.05)
                    t0 += 0.1

        t0 += 2.5

        # example faces
        signs = [-1, -1, 1]
        signed_units = [sign * Vector(u) for u, sign in zip(unit_tuples(3), signs)]
        projected_faces = face_directions(3)

        center = Vector((2, 2, 0.01))  # lift it slightly above the plane
        colors = ["plastic_drawing", "plastic_important", "plastic_custom2"]

        # t0+=2.5
        coords_move_back_time = t0

        t0 += 2.5

        # show projection
        # show_plane = Plane(location=[2,2,0],rotation_euler=[np.pi/2,0,0],scale=[2,2,0],name="Plane_Show",color='plastic_joker')
        # show_plane.move_to(target_location=[-11.5,0,-4.5],begin_time=0,transition_time=0)
        # show_plane.rescale(rescale=unit,begin_time=0,transition_time=0)
        # show_plane.appear(begin_time=t0,transition_time=0.5)
        #
        # selected_pos = [pos for pos in positions if pos[0] > 0 and pos[1] > 0 and pos[2] == 0]
        # projected_polygons =[Polygon(vertices=plane_for_center_and_directions(pos,[signed_units[0],signed_units[1]]),
        #                              color='plastic_drawing',solid=0.1,offset=0,bevel=0.1) for pos in selected_pos]
        # [poly.appear(begin_time=t0+i*0.1,transition_time=0.5) for i,poly in enumerate(projected_polygons)]
        # [ibpy.set_parent(poly,show_plane) for poly in projected_polygons]
        # duration = 0.5 + 0.1 * len(projected_polygons)
        # t0+=0.5+duration

        # create all faces
        units = [Vector(u) for u in unit_tuples(3)]

        # projected_faces = []
        # for pos in positions:
        #     if pos[0] < 4 and pos[1] < 4:
        #         projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[1]]),
        #                                        color='plastic_drawing', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
        #                                        scale=[unit] * 3,name="projection_01_"+str(pos)))
        #
        # for pos in positions:
        #     if pos[0] < 4 and pos[2] < 2:
        #         projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[0], units[2]]),
        #                                        color='plastic_important', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
        #                                        scale=[unit] * 3,name="projection_02_"+str(pos)))
        #
        # for pos in positions:
        #     if pos[1] < 4 and pos[2] < 2:
        #         projected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos, [units[1], units[2]]),
        #                                        color='plastic_custom2', solid=0.1 / unit, offset=0, bevel=0.1 / unit,
        #                                        scale=[unit] * 3,name="projection_12_"+str(pos)))
        #
        # [face.appear(begin_time=t0, transition_time=0.5) for face in projected_faces]
        # coords.add_objects(projected_faces)
        #
        # unprojected_faces=[]
        # for pos in positions:
        #     if pos[0]<4 and pos[1]<4:
        #         unprojected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos,[units[0],units[1]]),
        #                              transmission=0,color='plastic_drawing',name="plane_01_"+str(pos),scale=[unit]*3))
        #
        # for pos in positions:
        #     if pos[0]<4 and pos[2]<2:
        #         unprojected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos,[units[0],units[2]]),
        #                              transmission=0,color='plastic_important',name="plane_02_"+str(pos),scale=[unit]*3))
        #
        # for pos in positions:
        #     if pos[1]<4 and pos[2]<2:
        #         unprojected_faces.append(Polygon(vertices=plane_for_center_and_directions(pos,[units[1],units[2]]),
        #                             transmission=0,color='plastic_custom2',name="plane_12_"+str(pos),scale=[unit]*3))
        #
        # [face.appear(begin_time=t0,transition_time=0.5) for face in unprojected_faces]
        # [face.change_alpha(alpha=0.2,begin_time=t0+0.1,transition_time=0.5) for face in unprojected_faces]
        # coords.add_objects(unprojected_faces)

        # rotate plane
        rotation_duration = 20
        rot_box = BObject(children=[scatter_cube, projection_plane, light_zone3, light_zone2, light_zone1],
                          location=[2 * unit, 2 * unit, 0])
        rot_box.appear(begin_time=t0, transition_time=0)
        coords.add_object(rot_box)
        rot_box.rotate(rotation_euler=[0, np.arcsin(np.sqrt(2 / 3)), np.pi / 4], begin_time=t0,
                       transition_time=rotation_duration,
                       pivot=Vector([2 * unit, 2 * unit, 0]))
        # adjust for diagonal stretching
        rot_box.rescale(rescale=[1.5, 1.5, 1], begin_time=t0, transition_time=rotation_duration)

        # normal_vec = Cylinder.from_start_to_end(start=Vector([2 * unit, 2 * unit,0]),end=Vector([2*unit,2*unit,1*unit]), color='example')
        # normal_vec.appear(begin_time=t0, transition_time=0.1)
        #
        # base_x_vec = Cylinder.from_start_to_end(start=Vector([2*unit,2*unit,0]),end=Vector([3*unit,2*unit,0]),color='important')
        # base_y_vec = Cylinder.from_start_to_end(start=Vector([2*unit,2*unit,0]),end=Vector([2*unit,3*unit,0]),color='drawing')
        # base_x_vec.appear(begin_time=t0,transition_time=0.1)
        # base_y_vec.appear(begin_time=t0,transition_time=0.1)
        #
        # coords.add_objects([normal_vec,base_x_vec,base_y_vec])

        # dynamically adjust points in the zone

        # count = 0
        # face_visibility_map={}
        # for face,projected_face in zip(projected_faces,unprojected_faces):
        #     face.appear(begin_time=t0,transition_time=0.1)
        #     projected_face.appear(begin_time=t0,transition_time=0.1)
        #
        #     if all([v.z==0 for v in face.vertices0]):
        #         face_visibility_map[face]=True
        #         face_visibility_map[projected_face]=True
        #         print("face: ",face.name," displayed from the beginning")
        #         count+=1
        #     else:
        #         face_visibility_map[face]=False
        #         face_visibility_map[projected_face]=False
        #         face.toggle_hide(begin_time=t0)
        #         projected_face.toggle_hide(begin_time=t0)
        #         print("face: ",face.name," hidden from the beginning")

        map_selected_positions = {pos: pos[2] == 0 for pos in positions}

        voronoi_cell = tuples([0.5, -0.5], 3)
        delta = 10

        for frame in range(int(t0 * FRAME_RATE), int(t0 + rotation_duration) * FRAME_RATE + 1, delta):
            tstart = frame / FRAME_RATE
            duration = delta / FRAME_RATE
            # get rotation angle
            rotation = ibpy.get_rotation_at_frame(rot_box, frame + delta)
            # calculate_normal
            alpha_y = rotation[1]
            alpha_z = rotation[2]
            rot1 = Matrix([[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
            rot2 = Matrix([[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
            rot = rot2 @ rot1

            # locale base
            base = [Vector(rot @ u) for u in units]

            projected_voronoi = [base[2].dot(v) for v in voronoi_cell]
            inf = min(projected_voronoi)
            sup = max(projected_voronoi)
            print(frame, ": ", inf, sup)
            new_map_selected_positions = {pos: inf <= base[2].dot(Vector(pos) - Vector([2, 2, 0])) <= sup for pos in
                                          positions}
            print(len({k: v for k, v in new_map_selected_positions.items() if v}))
            for pos in positions:
                if new_map_selected_positions[pos] and map_selected_positions[pos]:
                    pass
                elif new_map_selected_positions[pos]:
                    # make position visible and turn blue
                    position_sphere_map[pos].change_alpha(alpha=1, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_drawing', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned visible")
                elif map_selected_positions[pos]:
                    # make position invisible and red
                    position_sphere_map[pos].change_alpha(alpha=0.05, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_important', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned invisible")

            map_selected_positions = new_map_selected_positions
            active_point_set = set([key for key, val in map_selected_positions.items() if val])
            print("active points: ", len(active_point_set))
            # adjust the projection of all faces

            # plane_center = Vector([2,2,0])
            # [poly.morph_to(projector=lambda v:(base[0].dot(v-plane_center)*base[0]+ base[1].dot(v-plane_center)*base[1]+plane_center),
            #                begin_time=tstart,transition_time=duration) for poly in projected_faces]
            #
            # #adjust visibility if necessary
            # for face,projected_face in zip(projected_faces,unprojected_faces):
            #     if set([tuple(v) for v in face.vertices0]).issubset(active_point_set):
            #         if not face_visibility_map[face]:
            #             face.toggle_hide(begin_time=frame/FRAME_RATE)
            #             projected_face.toggle_hide(begin_time=frame/FRAME_RATE)
            #             face_visibility_map[face]=True
            #             face_visibility_map[projected_face]=True
            #             print(face.name," appears")
            #
            #     else:
            #         if face_visibility_map[face]:
            #             face.toggle_hide(begin_time=frame/FRAME_RATE)
            #             projected_face.toggle_hide(begin_time=frame/FRAME_RATE)
            #             face_visibility_map[face]=False
            #             face_visibility_map[projected_face]=False
            #             print(face.name," disappears")
        t0 += 0.5 + rotation_duration

        rot_box.rotate(rotation_euler=[0, 0, 0], begin_time=t0,
                       transition_time=rotation_duration,
                       pivot=Vector([2 * unit, 2 * unit, 0]))
        # adjust for diagonal stretching
        rot_box.rescale(rescale=[1 / 1.5, 1 / 1.5, 1], begin_time=t0, transition_time=rotation_duration)

        for frame in range(int(t0 * FRAME_RATE), int(t0 + rotation_duration) * FRAME_RATE + 1, delta):
            tstart = frame / FRAME_RATE
            duration = delta / FRAME_RATE
            # get rotation angle
            rotation = ibpy.get_rotation_at_frame(rot_box, frame + delta)
            # calculate_normal
            alpha_y = rotation[1]
            alpha_z = rotation[2]
            rot1 = -1 * Matrix(
                [[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
            rot2 = -1 * Matrix(
                [[np.cos(alpha_z), -np.sin(alpha_z), 0], [np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
            rot = rot2 @ rot1

            # locale base
            base = [Vector(rot @ u) for u in units]

            projected_voronoi = [base[2].dot(v) for v in voronoi_cell]
            inf = min(projected_voronoi)
            sup = max(projected_voronoi)
            print(frame, ": ", inf, sup)
            new_map_selected_positions = {pos: inf <= base[2].dot(Vector(pos) - Vector([2, 2, 0])) <= sup for pos in
                                          positions}
            print(len({k: v for k, v in new_map_selected_positions.items() if v}))
            for pos in positions:
                if new_map_selected_positions[pos] and map_selected_positions[pos]:
                    pass
                elif new_map_selected_positions[pos]:
                    # make position visible and turn blue
                    position_sphere_map[pos].change_alpha(alpha=1, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_drawing', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned visible")
                elif map_selected_positions[pos]:
                    # make position invisible and red
                    position_sphere_map[pos].change_alpha(alpha=0.05, begin_time=tstart, transition_time=duration)
                    position_sphere_map[pos].change_color(new_color='plastic_important', begin_time=tstart,
                                                          transition_time=duration)
                    print(pos, " turned invisible")

            map_selected_positions = new_map_selected_positions
            active_point_set = set([key for key, val in map_selected_positions.items() if val])
            print("active points: ", len(active_point_set))

        t0 += 0.5 + rotation_duration

        rot_box2 = BObject(children=[coords], name="CoordinateSystem_box", location=rotation_center)
        rot_box2.appear(begin_time=0, transition_time=0)
        rot_box2.move(direction=-rotation_center + Vector(), begin_time=0.5, transition_time=5)
        # rot_box2.move(direction=[5, 0, 0], begin_time=coords_move_back_time, transition_time=2)

        rot_box2.rotate(rotation_euler=[0, 0, -3 * np.pi], begin_time=0, transition_time=t0)

        self.t0 = t0

    def hyper_cube_data2(self):
        t0 = 0

        text = 'gray_1'
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[0, -20, 20])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=0)
        # setup penrose background
        plane = Plane(resolution=[10, 10], uniformization=False)
        plane.move(direction=[0, 0, -0.7], begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)

        z5_nodes = de_bruijn(k=35, base_color='gray_8', tile_separation=0.0, radius=300, emission=0, contrast=0.1)
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)

        plane.appear(begin_time=t0, transition_time=0)
        alpha = get_alpha_of_material("Penrose")

        tile_size = get_geometry_node_from_modifier(z5_nodes, 'tile_size')
        change_default_value(tile_size.inputs['Scale'], from_value=0, to_value=0.99, begin_time=0, transition_time=0)
        stretcher = get_geometry_node_from_modifier(z5_nodes, 'stretcher')
        change_default_value(stretcher.inputs['Scale'], from_value=1, to_value=1, begin_time=0, transition_time=0)

        t0 = change_default_value(alpha, from_value=0, to_value=1, begin_time=0, transition_time=0)

        display = GlassDisplay(flat=True, scales=[11, 6], color='drawing', location=(-8, 2, 4), number_of_lines=8,ior=1.1,volume_absorption=0.1)
        display.rotate(rotation_euler=[pi / 4, 0, 0], begin_time=0, transition_time=0)
        t0 = 0.5 + display.appear(begin_time=0, transition_time=0)

        title = SimpleTexBObject(r"\text{Cubes in various dimensions}", color='plastic_important', aligned="center")
        t0 = 0.5 + display.write_title(title, begin_time=0, transition_time=0)

        table = Table(np.array([
            [
                SimpleTexBObject(r"\text{Dim: } n", color=text),
                SimpleTexBObject(r"P=2^n", color=text),
                TexBObject("E", r"E=\tfrac{n\cdot P}{2}", color=text),
                TexBObject("F", r"F=\tfrac{(n-1)\cdot E}{4}", color=text),
                TexBObject("V_3", r"V_3=\tfrac{(n-2)\cdot F}{6}", color=text),
                TexBObject("V_4", r"V_4=\tfrac{(n-3)\cdot V_3}{8}", color=text),
                SimpleTexBObject(r"V_5", color=text)
            ],
            [SimpleTexBObject("0", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None,
             None,
             None,
             None
             ],
            [SimpleTexBObject("1", color=text),
             SimpleTexBObject("2", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None,
             None,
             None
             ],
            [SimpleTexBObject("2", color=text),
             SimpleTexBObject(" 4", color=text),
             SimpleTexBObject("4", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None,
             None
             ],
            [SimpleTexBObject("3", color=text),
             SimpleTexBObject("8", color=text),
             SimpleTexBObject("12", color=text),
             SimpleTexBObject("6", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None
             ],
            [SimpleTexBObject("4", color=text),
             SimpleTexBObject("16", color=text),
             SimpleTexBObject("32", color=text),
             SimpleTexBObject("24", color=text),
             SimpleTexBObject("8", color=text),
             SimpleTexBObject("1", color=text),
             None
             ],
            [SimpleTexBObject("5", color=text),
             SimpleTexBObject("32", color=text),
             SimpleTexBObject("80", color=text),
             SimpleTexBObject("80", color=text),
             SimpleTexBObject("40", color=text),
             SimpleTexBObject("10", color=text),
             SimpleTexBObject("1", color=text)
             ]
        ]))

        display.add_bob(table, line=1, indent=3.55, scale=[2, 4, 4])

        times = [0, 0, 0, 0, 0, 4, 5]
        for r in range(5):
            table.write_row(r, begin_time=0, transition_time=times[r])

        display.rescale(rescale=[2 / 3, 2 / 3, 2 / 3], begin_time=t0, transition_time=0)

        [table.write_entry(5, col, begin_time=0, transition_time=0) for col in range(2)]
        [table.write_entry(6, col, begin_time=0, transition_time=0) for col in range(2)]

        for row in range(5, 7):
            t0 = table.write_entry(row, row, begin_time=t0, transition_time=0.3)

        t0 += 0.5

        # derive formula for E
        # dim 1
        pos1 = Vector([6, 8, 0])
        sphere1 = Sphere(0.3, location=pos1, color='plastic_example')
        pos2 = Vector([10, 8, 0])
        sphere2 = Sphere(0.29, location=pos1, color='plastic_example')
        edge = Cylinder.from_start_to_end(start=Vector(pos1), end=Vector(pos2), color='plastic_drawing')
        coords = [
            SimpleTexBObject(r"(0)", color=text, location=pos1 + Vector([-0.5, 0, 0]), aligned="right",
                             text_size='large', rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1)", color=text, location=pos2 + Vector([0.5, 0, 0]), aligned="left", text_size='large',
                             rotation_euler=[pi / 4, 0, 0])
        ]

        removers = [sphere1, sphere2, edge, *coords]
        [sphere.grow(begin_time=t0, transition_time=1) for sphere in {sphere1, sphere2}]
        t0 = 0.5 + coords[0].write(begin_time=t0, transition_time=1)

        sphere2.move_to(target_location=pos2, begin_time=t0, transition_time=1)
        coords[1].write(begin_time=t0, transition_time=1)
        t0 = 0.5 + edge.grow(begin_time=t0, transition_time=1)

        # dim 2
        pos2 = [Vector([6, 0, 0]), Vector([10, 0, 0]), Vector([6, 4, 0])]
        # make the first sphere a tiny bit bigger
        spheres2 = [Sphere(0.3 - 0.01 * i, location=pos2[0], color='plastic_example') for i in range(3)]
        edges2 = [
            Cylinder.from_start_to_end(start=pos2[0], end=pos2[1], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=pos2[0], end=pos2[2], color='plastic_drawing'),
        ]

        coords2 = [
            SimpleTexBObject(r"(0|0)", color=text, location=pos2[0] + Vector([-0.5, 0, 0]), aligned="right",
                             text_size='large', rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1|0)", color=text, location=pos2[1] + Vector([0.5, 0, 0]), aligned="left",
                             text_size='large', rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(0|1)", color=text, location=pos2[2] + Vector([-0.5, 0, 0]), aligned="right",
                             text_size='large', rotation_euler=[pi / 4, 0, 0]),
        ]

        [sphere.grow(begin_time=t0, transition_time=1) for sphere in spheres2]
        coords2[0].write(begin_time=t0, transition_time=1)
        t0 += 1.5

        [spheres2[i].move_to(target_location=pos2[i], begin_time=t0, transition_time=1) for i in range(1, 3)]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges2]

        [coord.write(begin_time=t0, transition_time=1) for coord in coords2[1:]]
        t0 += 1.5

        removers += spheres2
        removers += edges2
        removers += coords2

        # dim 3
        pos3 = [Vector([6, -8, 0]), Vector([10, -8, 0]), Vector([6, -4, 0]), Vector([6, -8, 4])]
        # make the first sphere a tiny bit bigger
        spheres3 = [Sphere(0.3 - 0.01 * i, location=pos3[0], color='plastic_example') for i in range(4)]
        edges3 = [
            Cylinder.from_start_to_end(start=pos3[0], end=pos3[1], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=pos3[0], end=pos3[2], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=pos3[0], end=pos3[3], color='plastic_drawing'),
        ]

        coords3 = [
            SimpleTexBObject(r"(0|0|0)", location=pos3[0] + Vector([-0.5, 0, 0]), color=text,
                             rotation_euler=[pi / 4, 0, 0], text_size='large', aligned="right"),
            SimpleTexBObject(r"(1|0|0)", location=pos3[1] + Vector([+0.5, 0, 0]), color=text,
                             rotation_euler=[pi / 4, 0, 0], text_size='large', aligned="left"),
            SimpleTexBObject(r"(0|1|0)", location=pos3[2] + Vector([-0.5, 0, 0]), color=text,
                             rotation_euler=[pi / 4, 0, 0], text_size='large', aligned="right"),
            SimpleTexBObject(r"(0|0|1)", location=pos3[3] + Vector([-0.5, 0, 0]), color=text,
                             rotation_euler=[pi / 4, 0, 0], text_size='large', aligned="right"),
        ]

        [sphere.grow(begin_time=t0, transition_time=1) for sphere in spheres3]
        coords3[0].write(begin_time=t0, transition_time=1)
        t0 += 1.5

        [spheres3[i].move_to(target_location=pos3[i], begin_time=t0, transition_time=1) for i in range(1, 4)]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges3]

        [coord.write(begin_time=t0, transition_time=1) for coord in coords3[1:]]
        t0 += 1.5

        removers += spheres3
        removers += edges3
        removers += coords3

        t0 = 0.5 + table.get_entry(0, 2).morph_and_move(begin_time=t0, transition_time=1)
        [table.write_entry(5 + i, 2, begin_time=t0 + i * 0.5, transition_time=0.3) for i in range(2)]
        t0 += 1

        [remover.disappear(begin_time=t0, transition_time=1) for remover in removers]
        t0 += 1.5

        removers = []
        # derive formula for F

        # dim 2
        pos2 = [Vector([6, 0, 0]), Vector([10, 0, 0]), Vector([6, 4, 0]), Vector([10, 4, 0])]
        coords2 = [
            SimpleTexBObject(r"(0|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos2[0] + Vector([-0.5, 0, 0]), aligned="right"),
            SimpleTexBObject(r"(1|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos2[1] + Vector([0.5, 0, 0]), aligned="left"),
            SimpleTexBObject(r"(1|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos2[3] + Vector([0.5, 0, 0]), aligned="left"),
            SimpleTexBObject(r"(0|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos2[2] + Vector([-0.5, 0, 0]), aligned="right"),
        ]
        # make the first sphere a tiny bit bigger
        spheres2 = [Sphere(0.3 - 0.01 * i, location=pos2[0], color='plastic_example') for i in range(4)]
        edges2 = [
            Cylinder.from_start_to_end(start=Vector(pos2[0]), end=Vector(pos2[1]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos2[0]), end=Vector(pos2[1]), radius=0.099,
                                       color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos2[0]), end=Vector(pos2[2]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos2[1]), end=Vector(pos2[3]), color='plastic_drawing'),
        ]

        [sphere.grow(begin_time=t0, transition_time=1) for sphere in spheres2]
        t0 = 0.5 + coords2[0].write(begin_time=t0, transition_time=1)

        [spheres2[i].move_to(target_location=pos2[1], begin_time=t0, transition_time=1) for i in [1, 3]]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges2[0:2]]
        face = Polygon(vertices=[*pos2[0:2], pos2[1], pos2[0]], color="fake_glass_joker", solid=0.2, offset=0)
        face.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + coords2[1].write(begin_time=t0, transition_time=1)

        [spheres2[i].move_to(target_location=pos2[i], begin_time=t0, transition_time=1) for i in [2, 3]]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges2[2:4]]
        edges2[1].move(direction=[0, 4, 0], begin_time=t0, transition_time=1)
        [coord.write(begin_time=t0, transition_time=1) for coord in coords2[2:4]]
        t0 = 0.5 + face.morph_to2(new_vertices=[*pos2[0:2], pos2[3], pos2[2]], begin_time=t0, transition_time=1)

        removers.append(face)
        removers += spheres2
        removers += edges2
        removers += coords2

        # dim 3
        pos3 = [Vector([6, -8, 0]), Vector([10, -8, 0]), Vector([6, -4, 0]), Vector([10, -4, 0]), Vector([6, -8, 4]),
                Vector([10, -8, 4])]
        coords3 = [
            SimpleTexBObject(r"(0|0|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[0] + Vector([-0.5, 0, 0]), aligned="right"),
            SimpleTexBObject(r"(1|0|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[1] + Vector([0.5, 0, 0]), aligned="left"),
            SimpleTexBObject(r"(0|1|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[2] + Vector([-0.5, 0, 0]), aligned="right"),
            SimpleTexBObject(r"(1|1|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[3] + Vector([0.5, 0, 0]), aligned="left"),
            SimpleTexBObject(r"(0|0|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[4] + Vector([-0.5, 0, 0]), aligned="right"),
            SimpleTexBObject(r"(1|0|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[5] + Vector([0.5, 0, 0]), aligned="left"),
        ]
        # make the first sphere a tiny bit bigger
        spheres3 = [Sphere(0.3 - 0.01 * i, location=pos3[0], color='plastic_example') for i in range(6)]
        edges3 = [
            Cylinder.from_start_to_end(start=Vector(pos3[0]), end=Vector(pos3[1]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[0]), end=Vector(pos3[1]), radius=0.099,
                                       color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[0]), end=Vector(pos3[1]), radius=0.098,
                                       color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[0]), end=Vector(pos3[2]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[1]), end=Vector(pos3[3]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[0]), end=Vector(pos3[4]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[1]), end=Vector(pos3[5]), color='plastic_drawing'),
        ]

        [sphere.grow(begin_time=t0, transition_time=1) for sphere in spheres3]
        t0 = 0.5 + coords3[0].write(begin_time=t0, transition_time=1)

        [sphere.move_to(target_location=pos3[1], begin_time=t0, transition_time=1) for sphere in
         {spheres3[1], spheres3[3], spheres3[5]}]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges3[0:3]]
        faces3 = [
            Polygon(vertices=[*pos3[0:2], pos3[1], pos3[0]], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[*pos3[0:2], pos3[1], pos3[0]], color='fake_glass_joker', solid=0.2, offset=0)
        ]
        [face.appear(begin_time=t0, transition_time=1) for face in faces3]
        t0 = 0.5 + coords3[1].write(begin_time=t0, transition_time=1)

        [spheres3[i].move_to(target_location=pos3[i], begin_time=t0, transition_time=1) for i in range(2, 6)]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges3[3:]]
        [edge.move(direction=[0, (1 - i) * 4, i * 4], begin_time=t0, transition_time=1) for i, edge in
         enumerate(edges3[1:3])]
        [coord.write(begin_time=t0, transition_time=1) for coord in coords3[2:]]
        faces3[0].morph_to2(new_vertices=[*pos3[0:2], pos3[3], pos3[2]], begin_time=t0, transition_time=1)
        t0 = 0.5 + faces3[1].morph_to2(new_vertices=[*pos3[0:2], pos3[5], pos3[4]], begin_time=t0, transition_time=1)

        removers += spheres3
        removers += edges3
        removers += coords3
        removers += faces3

        t0 = 0.5 + table.get_entry(0, 3).morph_and_move(begin_time=t0, transition_time=1)
        [table.write_entry(5 + i, 3, begin_time=t0 + i * 0.5, transition_time=0.3) for i in range(2)]
        t0 += 1

        [remover.disappear(begin_time=t0, transition_time=1) for remover in removers]
        t0 += 1.5

        removers = []
        # derive formula for V3

        # dim 3
        pos3 = [Vector([6, -8, 0]), Vector([10, -8, 0]), Vector([10, -4, 0]), Vector([6, -4, 0])]
        coords3 = [
            SimpleTexBObject(r"(0|0|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[0] + Vector([-0.5, 0, 0]), aligned="right"),
            SimpleTexBObject(r"(1|0|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[1] + Vector([0.5, 0, 0]), aligned="left"),
            SimpleTexBObject(r"(0|1|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[3] + Vector([-0.5, 0, 0]), aligned="right"),
            SimpleTexBObject(r"(1|1|0)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[2] + Vector([0.5, 0, 0]), aligned="left"),
            SimpleTexBObject(r"(0|0|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[0] + Vector([-0.5, 0, 4]), aligned="right"),
            SimpleTexBObject(r"(1|0|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[1] + Vector([0.5, 0, 4]), aligned="left"),
            SimpleTexBObject(r"(0|1|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[3] + Vector([-0.5, 0, 4]), aligned="right"),
            SimpleTexBObject(r"(1|1|1)", color=text, text_size='large', rotation_euler=[pi / 4, 0, 0],
                             location=pos3[2] + Vector([0.5, 0, 4]), aligned="left"),
        ]
        # make the first sphere a tiny bit bigger
        spheres3 = [Sphere(0.3, location=pos, color='plastic_example') for pos in pos3]
        edges3 = [
            Cylinder.from_start_to_end(start=Vector(pos3[0]), end=Vector(pos3[1]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[1]), end=Vector(pos3[2]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[2]), end=Vector(pos3[3]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(pos3[3]), end=Vector(pos3[0]), color='plastic_drawing'),
        ]
        face = Polygon(vertices=pos3, color="fake_glass_joker", solid=0.2, offset=0)
        face.grow(begin_time=t0, transition_time=1)

        [sphere.grow(begin_time=t0, transition_time=1) for sphere in spheres3]
        [coord.write(begin_time=t0, transition_time=1) for coord in coords3[0:4]]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges3]

        t0 += 1.5

        cube = Cube(location=(8, -6, 2), scale=2, color='plastic_text')
        cube.appear(alpha=0.8, begin_time=t0, transition_time=1)

        [coord.write(begin_time=t0, transition_time=1) for coord in coords3[4:]]

        removers += spheres3
        removers += edges3
        removers += coords3
        removers.append(face)
        removers.append(cube)
        t0 += 1.5

        t0 = 0.5 + table.get_entry(0, 4).morph_and_move(begin_time=t0, transition_time=1)
        [table.write_entry(5 + i, 4, begin_time=t0 + i * 0.5, transition_time=0.3) for i in range(2)]
        t0 += 1

        [remover.disappear(begin_time=t0, transition_time=1) for remover in removers]
        t0 += 1.5

        # formula for V4
        display.rescale(rescale=[3 / 2, 3 / 2, 3 / 2], begin_time=t0, transition_time=1)
        t0 = 0.5 + display.move(direction=[8, -5, 0], begin_time=t0, transition_time=1)

        t0 = 0.5 + table.get_entry(0, 5).morph_and_move(begin_time=t0, transition_time=1)
        table.write_entry(6, 5, begin_time=t0, transition_time=0.3)
        t0 += 1.5

        # additional remarks

        consistency = SimpleTexBObject("32-80+80-40+10-1=1", color='drawing')
        t0 = 1 + display.write_text_in(consistency, line=6, indent=1.5, begin_time=t0, transition_time=3)

        # there is a bug, when display disappear is called somehow the morphing part disappears immediately
        # t0 = 0.5 + display.disappear(begin_time=t0, transition_time=1)
        self.t0 = t0

    def hyper_cube_data(self):
        t0 = 0
        roughness = 0.1
        text = 'gray_1'
        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[0, -20, 20])
        empty = EmptyCube(location=Vector((0, 0, 0)))
        ibpy.set_camera_view_to(empty)

        # ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=0)
        # setup penrose background
        plane = Plane(resolution=[10, 10], uniformization=False)
        plane.move(direction=[0, 0, -0.7], begin_time=t0, transition_time=0)
        plane.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)

        z5_nodes = de_bruijn(k=35, base_color='gray_8', tile_separation=0.0, radius=300, emission=0, contrast=0.1)
        plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        set_material(plane, "Penrose", slot=0)

        plane.appear(begin_time=t0, transition_time=0)
        alpha = get_alpha_of_material("Penrose")

        tile_size = get_geometry_node_from_modifier(z5_nodes, 'tile_size')
        change_default_value(tile_size.inputs['Scale'], from_value=0, to_value=0.99, begin_time=0, transition_time=0)
        stretcher = get_geometry_node_from_modifier(z5_nodes, 'stretcher')
        change_default_value(stretcher.inputs['Scale'], from_value=1, to_value=1, begin_time=0, transition_time=0)

        t0 = change_default_value(alpha, from_value=0, to_value=1, begin_time=0.1, transition_time=1)

        display = GlassDisplay(flat=True, scales=[11, 6], color='drawing', location=Vector(), number_of_lines=8,ior=1.1,volume_scatter=0.5)
        display.rotate(rotation_euler=[pi / 4, 0, 0], begin_time=0, transition_time=0)
        display.move(direction=[0, -4,4], begin_time=t0, transition_time=0)
        t0 = 0.5 + display.appear(begin_time=t0, transition_time=1)

        title = SimpleTexBObject(r"\text{Cubes in various dimensions}", color='plastic_important', aligned="center")
        t0 = 0.5 + display.write_title(title, begin_time=t0, transition_time=1)

        table = Table(np.array([
            [
                SimpleTexBObject(r"\text{Dim: } n", color=text),
                TexBObject(r"P", r"P=2^n", color=text),
                TexBObject("E", r"E=\tfrac{n\cdot P}{2}", color=text),
                TexBObject("F", r"F=\tfrac{(n-1)\cdot E}{4}", color=text),
                TexBObject("V_3", r"V_3=\tfrac{(n-2)\cdot F}{6}", color=text),
                TexBObject("V_4", r"V_4=\tfrac{(n-3)\cdot V_3}{8}", color=text),
                SimpleTexBObject(r"V_5", color=text)
            ],
            [SimpleTexBObject("0", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None,
             None,
             None,
             None
             ],
            [SimpleTexBObject("1", color=text),
             SimpleTexBObject("2", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None,
             None,
             None
             ],
            [SimpleTexBObject("2", color=text),
             SimpleTexBObject(" 4", color=text),
             SimpleTexBObject("4", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None,
             None
             ],
            [SimpleTexBObject("3", color=text),
             SimpleTexBObject("8", color=text),
             SimpleTexBObject("12", color=text),
             SimpleTexBObject("6", color=text),
             SimpleTexBObject("1", color=text),
             None,
             None
             ],
            [SimpleTexBObject("4", color=text),
             SimpleTexBObject("16", color=text),
             SimpleTexBObject("32", color=text),
             SimpleTexBObject("24", color=text),
             SimpleTexBObject("8", color=text),
             SimpleTexBObject("1", color=text),
             None
             ],
            [SimpleTexBObject("5", color=text),
             SimpleTexBObject("32", color=text),
             SimpleTexBObject("80", color=text),
             SimpleTexBObject("80", color=text),
             SimpleTexBObject("40", color=text),
             SimpleTexBObject("10", color=text),
             SimpleTexBObject("1", color=text)
             ]
        ]))

        display.add_bob(table, line=1, indent=3.55, scale=[2, 4, 4])

        times = [1, 1, 1, 1, 1, 4, 5]
        t0 = 0.5 + table.write_row(0, begin_time=t0, transition_time=times[0])

        display.rescale(rescale=[5 / 6, 5 / 6, 5 / 6], begin_time=t0, transition_time=1)
        t0 = 0.5 + display.move(direction=[-4, 0, 0], begin_time=t0, transition_time=1)

        # zero dimensions
        pos0 = Vector([9.5, -7, 0])
        shift_normal = Vector([0, 1, 0])
        sphere0 = Sphere(0.3, location=pos0, color='plastic_example')
        sphere0.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + table.write_row(1, begin_time=t0, transition_time=1)

        # one dimensions

        t0 = 0.5 + sphere0.move(direction=shift_normal * 4, begin_time=t0, transition_time=1)

        pos1 = Vector([7.5, -7, 0])
        sphere = Sphere(0.3, location=pos1, color='plastic_example')
        pos2 = Vector([11.5, -7, 0])
        sphere2 = Sphere(0.3, location=pos2, color='plastic_example')
        edge = Cylinder.from_start_to_end(start=Vector(pos1), end=Vector(pos2), color='plastic_drawing')

        removers = [sphere0, sphere, sphere2, edge]

        [sphere.grow(begin_time=t0, transition_time=1) for sphere in {sphere, sphere2}]
        table.write_row(2, begin_time=t0, transition_time=times[2])
        t0 = 0.5 + edge.grow(begin_time=t0, transition_time=1)

        coords = [
            SimpleTexBObject(r"(0)", location=pos1 + Vector([-0.5, 0.5, 0]), aligned="right",
                             rotation_euler=[pi / 4, 0, 0],
                             text_size='large', color=text),
            SimpleTexBObject(r"(1)", location=pos2 + Vector([0.5, 0.5, 0]), aligned="left",
                             rotation_euler=[pi / 4, 0, 0],
                             text_size='large', color=text)
        ]

        for c in coords:
            t0 = c.write(begin_time=t0, transition_time=0.5)

        t0 += 0.5

        removers += coords

        # two dimensions

        [mover.move(direction=shift_normal * 5, begin_time=t0, transition_time=1) for mover in removers]
        t0 += 1.5

        positions = [Vector([7.5, -9, 0]), Vector([11.5, -9, 0]), Vector([11.5, -5, 0]), Vector([7.5, -5, 0])]
        spheres2 = [Sphere(0.3, location=p, color='plastic_example') for p in positions]
        edges2 = [
            Cylinder.from_start_to_end(start=Vector(positions[0]), end=Vector(positions[1]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(positions[0]), end=Vector(positions[3]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(positions[1]), end=Vector(positions[2]), color='plastic_drawing'),
            Cylinder.from_start_to_end(start=Vector(positions[2]), end=Vector(positions[3]), color='plastic_drawing'),
        ]
        [sphere.grow(begin_time=t0, transition_time=1) for sphere in spheres2]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges2]
        t0 += 1.5

        face = Polygon(vertices=[Vector(p) for p in positions], color='fake_glass_joker', solid=0.2, offset=0)
        table.write_row(3, begin_time=t0, transition_time=times[3])
        t0 = 0.5 + face.grow(begin_time=t0, transition_time=1)

        removers += spheres2
        removers += edges2
        removers.append(face)

        coords2 = [
            SimpleTexBObject(r"(0|0)", location=positions[0] + Vector([-0.5, 0, 0]), aligned="right", text_size='large',
                             color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1|0)", location=positions[1] + Vector([0.5, 0, 0]), aligned="left", text_size='large',
                             color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(0|1)", location=positions[3] + Vector([-0.5, 0, 0]), aligned="right", text_size='large',
                             color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1|1)", location=positions[2] + Vector([0.5, 0, 0]), aligned="left", text_size='large',
                             color=text, rotation_euler=[pi / 4, 0, 0]),
        ]
        removers += coords2

        for c in coords2:
            t0 = c.write(begin_time=t0, transition_time=0.5)
        t0 += 0.5

        # three dimensions
        [mover.move(direction=shift_normal * 10, begin_time=t0, transition_time=1) for mover in removers]
        display.move(direction=[-4, 6, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + display.rescale(rescale=[4 / 5, 4 / 5, 4 / 5], begin_time=t0, transition_time=1)

        coords = CoordinateSystem(dim=3, lengths=[5, 5, 5], domains=[[0, 1.1], [0, 1.1], [0, 1.1]],
                                  all_n_tics=[1, 1, 1],
                                  all_tic_lablls=[np.arange(-1, 1.1, 1), np.arange(-1, 1.1, 1), np.arange(-1, 1.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015],
                                  rotation_euler=[0,0,-pi/4],
                                  location_of_origin=[2,-4,0]
                                  )

        # zero dimensional cube
        coords.axes[2].axis_label.move(direction=[0, 0, -0.35], begin_time=t0, transition_time=0)
        sphere = Sphere(0.2, color="plastic_example", roughness=roughness)
        coords.add_object(sphere)
        t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=0.5)

        # one dimensional cube
        first = t0

        # grow axis
        t0 += 1.5
        sphere2 = Sphere(0.199, color="plastic_example", roughness=roughness)
        sphere2.appear(begin_time=t0, transition_time=0)
        coords.add_object(sphere2)
        sphere2.move(direction=coords.coords2location((1, 0, 0)), begin_time=t0, transition_time=2)

        edge = Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((1, 0, 0)),
                                          color='plastic_drawing', roughness=roughness)
        t0 = 0.5 + edge.grow(begin_time=t0, transition_time=2)
        coords.add_object(edge)
        # grow second axis
        second = t0
        t0 += 1.5

        movers = [
            Sphere(0.199, color="plastic_example", roughness=roughness),
            Sphere(0.198, location=coords.coords2location((1, 0, 0)), color="plastic_example", roughness=roughness)
        ]
        coords.add_objects(movers)
        [mover.appear(begin_time=t0, transition_time=0) for mover in movers]
        [mover.move(direction=coords.coords2location((0, 1, 0)), begin_time=t0, transition_time=2) for mover in movers]

        edges = [
            Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((0, 1, 0)), color="plastic_drawing",
                                       roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 0, 0)), end=coords.coords2location((1, 1, 0)),
                                       color="plastic_drawing", roughness=roughness),
        ]
        coords.add_objects(edges)
        [edge.grow(begin_time=t0, transition_time=2) for edge in edges]

        copy = Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((1, 0, 0)),
                                          color="plastic_drawing", roughness=0.3, radius=0.09)
        copy.appear(begin_time=t0, transition_time=0)
        copy.move(direction=coords.coords2location((0, 1, 0)), begin_time=t0, transition_time=2
                  )

        coords.add_object(copy)
        face = Polygon(vertices=[
            Vector(),
            coords.coords2location((1, 0, 0)),
            coords.coords2location((1, 0, 0)),
            Vector()
        ], color='fake_glass_joker', solid=0.2, offset=0)
        face.appear(begin_time=t0, transition_time=0, alpha=1)
        coords.add_object(face)
        t0 = 0.5 + face.morph_to2(new_vertices=[
            Vector(),
            coords.coords2location((1, 0, 0)),
            coords.coords2location((1, 1, 0)),
            coords.coords2location((0, 1, 0))
        ],
            begin_time=t0,
            transition_time=2
        )

        # grow third axis
        third = t0
        t0 += 1.5

        movers = [
            Sphere(0.197, location=Vector(), color='plastic_example', roughness=roughness),
            Sphere(0.197, location=coords.coords2location((1, 0, 0)), color='plastic_example', roughness=roughness),
            Sphere(0.197, location=coords.coords2location((1, 1, 0)), color='plastic_example', roughness=roughness),
            Sphere(0.197, location=coords.coords2location((0, 1, 0)), color='plastic_example', roughness=roughness),
        ]

        coords.add_objects(movers)
        [mover.appear(begin_time=t0, transition_time=0) for mover in movers]
        [mover.move(direction=coords.coords2location((0, 0, 1)), begin_time=t0, transition_time=2) for mover in movers]

        edges = [
            Cylinder.from_start_to_end(start=coords.coords2location((0, 0, 0)), end=coords.coords2location((0, 0, 1)),
                                       color='plastic_drawing', roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 0, 0)), end=coords.coords2location((1, 0, 1)),
                                       color='plastic_drawing', roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((0, 1, 0)), end=coords.coords2location((0, 1, 1)),
                                       color='plastic_drawing', roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 1, 0)), end=coords.coords2location((1, 1, 1)),
                                       color='plastic_drawing', roughness=roughness),
        ]
        coords.add_objects(edges)
        [edge.grow(begin_time=t0, transition_time=2) for edge in edges]

        faces = [
            Polygon(vertices=[
                Vector(),
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 0, 0)),
                Vector()
            ],color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[
                Vector(),
                coords.coords2location((0, 1, 0)),
                coords.coords2location((0, 1, 0)),
                Vector()
            ], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 0, 0)),
            ], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[
                coords.coords2location((0, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((0, 1, 0)),
            ], color='fake_glass_joker', solid=0.2, offset=0)
        ]

        new_vertices = [
            [
                Vector(),
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 0, 1)),
                coords.coords2location((0, 0, 1)),
            ],
            [
                Vector(),
                coords.coords2location((0, 1, 0)),
                coords.coords2location((0, 1, 1)),
                coords.coords2location((0, 0, 1)),
            ],
            [
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 1)),
                coords.coords2location((1, 0, 1)),
            ],
            [
                coords.coords2location((0, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 1)),
                coords.coords2location((0, 1, 1)),
            ]

        ]
        coords.add_objects(faces)
        [face.appear(begin_time=t0, transition_time=0, alpha=0.5) for face in faces]
        [face.morph_to2(new_vertices=verts, begin_time=t0, transition_time=2) for face, verts in
         zip(faces, new_vertices)]

        children = [
            Polygon(vertices=[
                coords.coords2location((0, 0, -0.001)),
                coords.coords2location((1, 0, -0.001)),
                coords.coords2location((1, 1, -0.001)),
                coords.coords2location((0, 1, 0.01))], color='fake_glass_joker', solid=0.2, offset=0),
            Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((1, 0, 0)), color="plastic_drawing",
                                       roughness=roughness, radius=0.098),
            Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((0, 1, 0)), color="plastic_drawing",
                                       roughness=roughness, radius=0.098),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 0, 0)), end=coords.coords2location((1, 1, 0)),
                                       color="plastic_drawing", roughness=roughness, radius=0.098),
            Cylinder.from_start_to_end(start=coords.coords2location((0, 1, 0)), end=coords.coords2location((1, 1, 0)),
                                       color="plastic_drawing", roughness=roughness, radius=0.098),
        ]

        [child.appear(begin_time=t0, transition_time=0) for child in children]

        # scatter_cube = Cube(name="volume_cube", color='scatter_volume',
        #                     location=coords.coords2location((0.5, 0.5, 0.5)),
        #                     scale=0.49 * coords.coords2location((1, 0, 0))[0])
        # coords.add_object(scatter_cube)
        #
        # ibpy.set_volume_scatter(scatter_cube, value=0.0, begin_time=0)
        # ibpy.set_volume_absorption(scatter_cube, value=0.0, begin_time=0)
        # ibpy.change_volume_scatter(scatter_cube, final_value=0.15, begin_time=t0 + 1, transition_time=2)
        # ibpy.change_volume_absorption(scatter_cube, final_value=0.15, begin_time=t0 + 1, transition_time=2)

        lit = BObject(children=children, name='lit')
        lit.appear(begin_time=t0, transition_time=0)
        coords.add_object(lit)
        lit.move(direction=coords.coords2location((0, 0, 1)), begin_time=t0, transition_time=2)
        t0 = 0.5 + coords.rotate(rotation_euler=[0, 0, 2 * np.pi], begin_time=t0, transition_time=8)

        coords.appear_individually(begin_times=[first, second, third], transition_times=[1] * 3)

        # remove all objects
        t0 = 0.5 + coords.disappear(begin_time=t0, transition_time=1)





        positions3 = [
            Vector([5.5, -9, 0]),
            Vector([9.5, -9, 0]),
            Vector([9.5, -5, 0]),
            Vector([5.5, -5, 0]),
            Vector([5.5, -9, 4]),
            Vector([9.5, -9, 4]),
            Vector([9.5, -5, 4]),
            Vector([5.5, -5, 4])
        ]
        center = Vector()
        for pos in positions3:
            center = center + pos
        center *= 0.125

        for i in range(len(positions3)):
            positions3[i] = positions3[i] - center

        cube = []

        spheres3 = [Sphere(0.3, location=p, color='plastic_example') for p in positions3]

        cube += spheres3
        edges3 = [
            Cylinder.from_start_to_end(start=positions3[0], end=positions3[1], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[0], end=positions3[3], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[1], end=positions3[2], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[2], end=positions3[3], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[4], end=positions3[5], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[4], end=positions3[7], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[5], end=positions3[6], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[6], end=positions3[7], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[0], end=positions3[4], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[1], end=positions3[5], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[2], end=positions3[6], color='plastic_drawing'),
            Cylinder.from_start_to_end(start=positions3[3], end=positions3[7], color='plastic_drawing'),
        ]
        [sphere.grow(begin_time=t0, transition_time=1) for sphere in spheres3]
        [edge.grow(begin_time=t0, transition_time=1) for edge in edges3]
        cube += edges3

        t0 += 1.5

        faces3 = [
            Polygon(vertices=[positions3[p] for p in [0, 1, 2, 3]], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[positions3[p] for p in [3, 7, 6, 2]], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[positions3[p] for p in [1, 2, 6, 5]], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[positions3[p] for p in [0, 4, 7, 3]], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[positions3[p] for p in [0, 1, 5, 4]], color='fake_glass_joker', solid=0.2, offset=0),
            Polygon(vertices=[positions3[p] for p in [4, 5, 6, 7]], color='fake_glass_joker', solid=0.2, offset=0),
        ]
        table.write_row(4, begin_time=t0, transition_time=times[4])

        for face in faces3:
            t0 = face.grow(begin_time=t0, transition_time=0.5)
        t0 += 0.5
        cube += faces3

        coords3 = [
            SimpleTexBObject(r"(0|0|0)", location=positions3[0] + Vector([-0.5, 0, 0]), aligned="right",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1|0|0)", location=positions3[1] + Vector([0.5, 0, 0]), aligned="left",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(0|1|0)", location=positions3[3] + Vector([-0.5, 0, 0]), aligned="right",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1|1|0)", location=positions3[2] + Vector([0.5, 0, 0]), aligned="left",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(0|0|1)", location=positions3[4] + Vector([-0.5, 0, 0]), aligned="right",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1|0|1)", location=positions3[5] + Vector([0.5, 0, 0]), aligned="left",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(0|1|1)", location=positions3[7] + Vector([-0.5, 0, 0]), aligned="right",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
            SimpleTexBObject(r"(1|1|1)", location=positions3[6] + Vector([0.5, 0, 0]), aligned="left",
                             text_size="large", color=text, rotation_euler=[pi / 4, 0, 0]),
        ]
        cube += coords3
        removers += spheres3
        removers += edges3
        removers += faces3
        removers += coords3

        bcube = BObject(children=cube, name="Cube")
        bcube.move(direction=center, begin_time=0, transition_time=0)
        bcube.appear(begin_time=0, transition_time=0)

        for coord in coords3:
            t0 = coord.write(begin_time=t0, transition_time=0.3)

        t0 += 0.7

        t0 = bcube.rotate(rotation_euler=[0, 0, -pi / 4], begin_time=t0, transition_time=3)
        t0 = 0.5 + bcube.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=3)

        # generate tree elements

        root = Tuple(element=())
        nodes = [root]
        new_nodes = [root]
        for i in range(0, 4):
            brand_new_nodes = []
            for node in new_nodes:
                child = Tuple(node.element + (0,))
                brand_new_nodes.append(child)
                child.parent = node
                child = Tuple(node.element + (1,))
                brand_new_nodes.append(child)
                child.parent = node
            nodes += brand_new_nodes
            new_nodes = brand_new_nodes

        tree = Tree(root, 9, 8, color_function=lambda x: [text], direction='left_right', name='t1',
                    display_mode='word', bevel=1, node_circles=False, level_positions=[0, 0.1, 0.3, 0.5, 0.8])

        tree.move(direction=[-12, -10, -0.7], begin_time=0, transition_time=0)
        tree.rotate(rotation_euler=[-np.pi / 4, 0, 0], begin_time=t0, transition_time=0)
        t0 = tree.appear(begin_time=t0, transition_time=10) + 0.5

        table.get_entry(0, 1).morph_and_move(begin_time=t0, transition_time=1)
        for entry in range(2):
            t0 = table.write_entry(5, entry, begin_time=t0, transition_time=0.3)
        t0 += 1
        for entry in range(2):
            t0 = table.write_entry(6, entry, begin_time=t0, transition_time=0.3)
        t0 += 1

        tree.disappear(begin_time=t0, transition_time=1)
        [remover.disappear(begin_time=t0, transition_time=1) for remover in removers]
        t0 += 1.5

        self.t0 = t0

    def hyper_cube(self):
        t0 = 0

        roughness = 0.1

        ibpy.set_hdri_background("forest", 'exr', simple=True, transparent=True,
                                 rotation_euler=[0, 0, 0])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True,
                               resolution_percentage=100)

        ibpy.set_camera_location(location=[10, -10, 7])
        empty = EmptyCube(location=Vector((0, 0, 0.67)))
        ibpy.set_camera_view_to(empty)

        # ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=0)

        # plane = Plane(resolution=[10, 10], uniformization=False)
        # plane.move(direction=[0, 0, -0.7], begin_time=t0, transition_time=0)
        # z5_nodes = de_bruijn(k=35, base_color='gray_3', tile_separation=0.0, radius=300, emission=0.1)
        # plane.add_mesh_modifier(type='NODES', node_group=z5_nodes)
        # set_material(plane, "Penrose", slot=0)
        # plane.add_mesh_modifier('SOLIDIFY', thickness=0.05, offset=0)
        # plane.add_mesh_modifier('BEVEL', amount=0.05, segments=1)

        # t0 = plane.appear(begin_time=t0, transition_time=0)
        # alpha = get_alpha_of_material("Penrose")
        # change_default_value(alpha, from_value=0, to_value=1, begin_time=t0 + 0.1, transition_time=1)

        # tile_size = get_geometry_node_from_modifier(z5_nodes, 'tile_size')
        # change_default_value(tile_size.inputs['Scale'], from_value=0, to_value=0.99, begin_time=0, transition_time=2)
        # stretcher = get_geometry_node_from_modifier(z5_nodes, 'stretcher')
        # t0 = 0.5 + change_default_value(stretcher.inputs['Scale'], from_value=1, to_value=1, begin_time=0,
        #                                 transition_time=0)

        coords = CoordinateSystem(dim=3, lengths=[5, 5, 5], domains=[[0, 1.1], [0, 1.1], [0, 1.1]],
                                  all_n_tics=[1, 1, 1],
                                  all_tic_lablls=[np.arange(-1, 1.1, 1), np.arange(-1, 1.1, 1), np.arange(-1, 1.1, 1)],
                                  label_digits=[0, 0, 0],
                                  radii=[0.015, 0.015, 0.015]
                                  )

        # zero dimensional cube
        coords.axes[2].axis_label.move(direction=[0, 0, -0.35], begin_time=t0, transition_time=0)
        sphere = Sphere(0.2, color="plastic_example", roughness=roughness)
        coords.add_object(sphere)
        t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=0.5)

        # one dimensional cube
        first = t0

        # grow axis
        t0 += 1.5
        sphere2 = Sphere(0.199, color="plastic_example", roughness=roughness)
        sphere2.appear(begin_time=t0, transition_time=0)
        coords.add_object(sphere2)
        sphere2.move(direction=coords.coords2location((1, 0, 0)), begin_time=t0, transition_time=2)

        edge = Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((1, 0, 0)),
                                          color='plastic_drawing', roughness=roughness)
        t0 = 0.5 + edge.grow(begin_time=t0, transition_time=2)
        coords.add_object(edge)
        # grow second axis
        second = t0
        t0 += 1.5

        movers = [
            Sphere(0.199, color="plastic_example", roughness=roughness),
            Sphere(0.198, location=coords.coords2location((1, 0, 0)), color="plastic_example", roughness=roughness)
        ]
        coords.add_objects(movers)
        [mover.appear(begin_time=t0, transition_time=0) for mover in movers]
        [mover.move(direction=coords.coords2location((0, 1, 0)), begin_time=t0, transition_time=2) for mover in movers]

        edges = [
            Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((0, 1, 0)), color="plastic_drawing",
                                       roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 0, 0)), end=coords.coords2location((1, 1, 0)),
                                       color="plastic_drawing", roughness=roughness),
        ]
        coords.add_objects(edges)
        [edge.grow(begin_time=t0, transition_time=2) for edge in edges]

        copy = Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((1, 0, 0)),
                                          color="plastic_drawing", roughness=0.3, radius=0.09)
        copy.appear(begin_time=t0, transition_time=0)
        copy.move(direction=coords.coords2location((0, 1, 0)), begin_time=t0, transition_time=2
                  )

        coords.add_object(copy)
        face = Polygon(vertices=[
            Vector(),
            coords.coords2location((1, 0, 0)),
            coords.coords2location((1, 0, 0)),
            Vector()
        ], color="joker", solid=0.2, offset=0,alpha=0.05)
        face.appear(begin_time=t0, transition_time=0, alpha=1)
        coords.add_object(face)
        t0 = 0.5 + face.morph_to2(new_vertices=[
            Vector(),
            coords.coords2location((1, 0, 0)),
            coords.coords2location((1, 1, 0)),
            coords.coords2location((0, 1, 0))
        ],
            begin_time=t0,
            transition_time=2
        )

        # grow third axis
        third = t0
        t0 += 1.5

        movers = [
            Sphere(0.197, location=Vector(), color='plastic_example', roughness=roughness),
            Sphere(0.197, location=coords.coords2location((1, 0, 0)), color='plastic_example', roughness=roughness),
            Sphere(0.197, location=coords.coords2location((1, 1, 0)), color='plastic_example', roughness=roughness),
            Sphere(0.197, location=coords.coords2location((0, 1, 0)), color='plastic_example', roughness=roughness),
        ]

        coords.add_objects(movers)
        [mover.appear(begin_time=t0, transition_time=0) for mover in movers]
        [mover.move(direction=coords.coords2location((0, 0, 1)), begin_time=t0, transition_time=2) for mover in movers]

        edges = [
            Cylinder.from_start_to_end(start=coords.coords2location((0, 0, 0)), end=coords.coords2location((0, 0, 1)),
                                       color='plastic_drawing', roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 0, 0)), end=coords.coords2location((1, 0, 1)),
                                       color='plastic_drawing', roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((0, 1, 0)), end=coords.coords2location((0, 1, 1)),
                                       color='plastic_drawing', roughness=roughness),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 1, 0)), end=coords.coords2location((1, 1, 1)),
                                       color='plastic_drawing', roughness=roughness),
        ]
        coords.add_objects(edges)
        [edge.grow(begin_time=t0, transition_time=2) for edge in edges]

        faces = [
            Polygon(vertices=[
                Vector(),
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 0, 0)),
                Vector()
            ], color="joker", solid=0.2, offset=0,alpha=0.05),
            Polygon(vertices=[
                Vector(),
                coords.coords2location((0, 1, 0)),
                coords.coords2location((0, 1, 0)),
                Vector()
            ], color="joker", solid=0.2, offset=0,alpha=0.05),
            Polygon(vertices=[
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 0, 0)),
            ], color="joker", solid=0.2, offset=0,alpha=0.05),
            Polygon(vertices=[
                coords.coords2location((0, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((0, 1, 0)),
            ], color="joker", solid=0.2, offset=0,alpha=0.05)
        ]

        new_vertices = [
            [
                Vector(),
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 0, 1)),
                coords.coords2location((0, 0, 1)),
            ],
            [
                Vector(),
                coords.coords2location((0, 1, 0)),
                coords.coords2location((0, 1, 1)),
                coords.coords2location((0, 0, 1)),
            ],
            [
                coords.coords2location((1, 0, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 1)),
                coords.coords2location((1, 0, 1)),
            ],
            [
                coords.coords2location((0, 1, 0)),
                coords.coords2location((1, 1, 0)),
                coords.coords2location((1, 1, 1)),
                coords.coords2location((0, 1, 1)),
            ]

        ]
        coords.add_objects(faces)
        [face.appear(begin_time=t0, transition_time=0, alpha=0.5) for face in faces]
        [face.morph_to2(new_vertices=verts, begin_time=t0, transition_time=2) for face, verts in
         zip(faces, new_vertices)]

        children = [
            Polygon(vertices=[
                coords.coords2location((0, 0, -0.001)),
                coords.coords2location((1, 0, -0.001)),
                coords.coords2location((1, 1, -0.001)),
                coords.coords2location((0, 1, 0.01))], color="joker", solid=0.2, offset=0,alpha=0.05),
            Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((1, 0, 0)), color="plastic_drawing",
                                       roughness=roughness, radius=0.098),
            Cylinder.from_start_to_end(start=Vector(), end=coords.coords2location((0, 1, 0)), color="plastic_drawing",
                                       roughness=roughness, radius=0.098),
            Cylinder.from_start_to_end(start=coords.coords2location((1, 0, 0)), end=coords.coords2location((1, 1, 0)),
                                       color="plastic_drawing", roughness=roughness, radius=0.098),
            Cylinder.from_start_to_end(start=coords.coords2location((0, 1, 0)), end=coords.coords2location((1, 1, 0)),
                                       color="plastic_drawing", roughness=roughness, radius=0.098),
        ]

        [child.appear(begin_time=t0, transition_time=0) for child in children]

        # scatter_cube = Cube(name="volume_cube", color='scatter_volume',
        #                     location=coords.coords2location((0.5, 0.5, 0.5)),
        #                     scale=0.49 * coords.coords2location((1, 0, 0))[0])
        # coords.add_object(scatter_cube)
        # 
        # ibpy.set_volume_scatter(scatter_cube, value=0.0, begin_time=0)
        # ibpy.set_volume_absorption(scatter_cube, value=0.0, begin_time=0)
        # ibpy.change_volume_scatter(scatter_cube, final_value=0.15, begin_time=t0 + 1, transition_time=2)
        # ibpy.change_volume_absorption(scatter_cube, final_value=0.15, begin_time=t0 + 1, transition_time=2)

        lit = BObject(children=children, name='lit')
        lit.appear(begin_time=t0, transition_time=0)
        coords.add_object(lit)
        lit.move(direction=coords.coords2location((0, 0, 1)), begin_time=t0, transition_time=2)
        t0 = 0.5 + coords.rotate(rotation_euler=[0, 0, 2 * np.pi], begin_time=t0, transition_time=8)

        coords.appear_individually(begin_times=[first, second, third], transition_times=[1] * 3)

        # remove all objects
        t0 = 0.5 + coords.disappear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def intro_overlay(self):
        t0 = 0
        size = 5

        ibpy.set_hdri_background("forest", 'exr', background='black', no_transmission_ray='True',
                                 rotation_euler=[0, 0, 3 * np.pi / 2])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE, motion_blur=False)
        ibpy.set_camera_location(Vector([13, 13, 13]))
        empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(empty)
        ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=0)

        # create 3d picture
        normal = Vector([1, 1, 1])
        normal.normalize()
        sel_points = close_points_3d(normal, 12)
        base_points = find_base_points(sel_points, 3)
        faces3 = face_directions(3)
        units3 = unit_tuples(3)
        [print(len(b)) for b in base_points]

        f = 0
        r2 = np.sqrt(2)
        r6 = np.sqrt(6)
        u = Vector((1 / r2, -1 / r2, 0))
        v = Vector((1 / r6, 1 / r6, -2 / r6))
        p = projector(u, v)
        para32 = np.dot(np.array([u, v]), p)
        polygons = []
        polygon_dict = {}

        colors = ["plastic_drawing", "plastic_important", "plastic_custom2"]

        for bps, face in zip(base_points, faces3):
            c = 0
            for pt in bps:
                v0 = Vector(pt)
                v1 = Vector(pt) + Vector(units3[face[0]])
                v2 = Vector(pt) + Vector(units3[face[0]]) + Vector(units3[face[1]])
                v3 = Vector(pt) + Vector(units3[face[1]])
                if f == 1:
                    verts = [p @ v0, p @ v3, p @ v2, p @ v1]
                else:
                    verts = [p @ v0, p @ v1, p @ v2, p @ v3]
                poly = Polygon(vertices=verts, index=f * 1000 + c, color=colors[f])
                poly.appear(transition_time=0)
                polygons.append(poly)
                polygon_dict[(f, c)] = poly
                c += 1
            f += 1

        t0 += 0.5

        colors5 = ["plastic_drawing", "plastic_important",
                   "plastic_custom2", "plastic_custom1",
                   "plastic_custom3", "plastic_custom4",
                   "plastic_joker", "plastic_gray_5",
                   "plastic_text", "plastic_gray_8"]

        # create 5d picture
        # define plane and orthogonal space
        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        u1 = -0.5 * np.sqrt(3 / 5 + 1 / r5)
        u2 = (-1 + r5) / 2 / r2 / r5
        u = (u1, u2, u2, u1, r2 / r5)
        v1 = 0.5 * np.sqrt(1 - 1 / r5)
        v2 = -0.5 * np.sqrt(1 + 1 / r5)
        v = (v1, v2, -v2, -v1, 0)

        ortho3 = np.array([[u2, u1, u1, u2, r2 / r5],
                           [-v2, v1, -v1, v2, 0],
                           [1 / r5, 1 / r5, 1 / r5, 1 / r5, 1 / r5]])

        p2 = np.array([u, v])

        p5 = projector(Vector(u), Vector(v))
        ortho = np.dot(ortho3, np.identity(5) - p5)
        para = np.dot(p2, p5)

        selected_points = close_points_5d(ortho, 15)
        base_points5 = find_base_points(selected_points, 5)
        faces = face_directions(5)
        units = unit_tuples(5)

        r6 = np.sqrt(6)
        u3 = Vector((1 / r2, -1 / r2, 0))
        v3 = Vector((1 / r6, 1 / r6, -2 / r6))
        n = Vector((1, 1, 1))

        polygons5 = []
        b5 = 0
        for bps5, face5 in zip(base_points5, faces):
            i5 = 0
            for q in bps5:
                v51 = np.dot(para, Vector(q))
                v52 = np.dot(para, Vector(q) + Vector(units[face5[0]]))
                v54 = np.dot(para, Vector(q) + Vector(units[face5[1]]))
                v53 = np.dot(para, Vector(q) + Vector(units[face5[0]]) + Vector(units[face5[1]]))
                if b5 in {1, 3, 5, 8}:
                    v54, v52 = v52, v54
                polygons5.append(
                    Polygon(vertices=[
                        0.0 * n + u3 * v51[0] + v3 * v51[1],
                        0.0 * n + u3 * v52[0] + v3 * v52[1],
                        0.0 * n + u3 * v53[0] + v3 * v53[1],
                        0.0 * n + u3 * v54[0] + v3 * v54[1]
                    ], color=colors5[b5], index=b5 * 1000 + i5))
                i5 += 1
            b5 += 1

        t0 += 2

        [poly.disappear(begin_time=t0, transition_time=2) for poly in polygons]
        [poly.appear(begin_time=t0, transition_time=2) for poly in polygons5]

        t0 += 2.5
        self.t0 = t0

    def intro_penrose(self):
        t0 = 0
        size = 5

        ibpy.set_hdri_background("forest", 'exr', background='black', no_transmission_ray='True',
                                 rotation_euler=[0, 0, 3 * np.pi / 2])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE)
        ibpy.set_camera_location(Vector([13, 13, 13]))
        empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(empty)
        ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=0)

        # define plane and orthogonal space
        r5 = np.sqrt(5)
        r2 = np.sqrt(2)
        u1 = -0.5 * np.sqrt(3 / 5 + 1 / r5)
        u2 = (-1 + r5) / 2 / r2 / r5
        u = (u1, u2, u2, u1, r2 / r5)
        v1 = 0.5 * np.sqrt(1 - 1 / r5)
        v2 = -0.5 * np.sqrt(1 + 1 / r5)
        v = (v1, v2, -v2, -v1, 0)

        ortho3 = np.array([[u2, u1, u1, u2, r2 / r5],
                           [-v2, v1, -v1, v2, 0],
                           [1 / r5, 1 / r5, 1 / r5, 1 / r5, 1 / r5]])

        p2 = np.array([u, v])

        p5 = projector(Vector(u), Vector(v))
        ortho = np.dot(ortho3, np.identity(5) - p5)
        para = np.dot(p2, p5)

        # visualizing the complex hull
        # voronoiCell = tuples([0.5,-0.5],5)
        #
        # cell_points = [np.dot(ortho, Vector(v)) for v in voronoiCell]
        # ch = ConvexHull(cell_points)

        # spheres = [Sphere(0.025, location=point, color='plastic_joker') for point in ch.points]
        # for sphere in spheres:
        #     sphere.grow(begin_time=t0, transition_time=0.5)
        #     t0 += 0.1
        #
        # faces = [Polygon(vertices=[Vector(v) for v in ch.points[list(indices)]], color='fake_glass_joker',transmission=0.5,roughness=0.5) for indices
        #          in ch.faces.keys()]
        # for face in faces:
        #     face.appear(begin_time=t0, transition_time=0.5)
        #     t0 += 0.1

        # test inside function

        # points = random_points(dim=3,n=100,domain=1)
        # spheres2 = [Sphere(r=0.05,location=point, color='plastic_joker') for point in points if ch.is_inside(point)]
        # spheres3 = [Sphere(r=0.05,location=point, color='plastic_important') for point in points if not ch.is_inside(point)]
        #
        # for sphere in spheres2+spheres3:
        #     sphere.grow(begin_time=t0,transition_time=0.5)
        #     t0+=0.1

        selected_points = close_points_5d(ortho, 25)
        base_points = find_base_points(selected_points, 5)
        faces = face_directions(5)
        units = unit_tuples(5)

        polygons = []
        colors = ["plastic_drawing", "plastic_important",
                  "plastic_custom2", "plastic_custom1",
                  "plastic_custom3", "plastic_custom4",
                  "plastic_joker", "plastic_gray_5",
                  "plastic_text", "plastic_gray_8"]
        f = 0

        r6 = np.sqrt(6)
        u3 = Vector((1 / r2, -1 / r2, 0))
        v3 = Vector((1 / r6, 1 / r6, -2 / r6))

        print([len(b) for b in base_points])
        for bps, face in zip(base_points, faces):
            c = 0
            for pt in bps:
                p0 = Vector(pt)
                p1 = Vector(pt) + Vector(units[face[0]])
                p2 = Vector(pt) + Vector(units[face[0]]) + Vector(units[face[1]])
                p3 = Vector(pt) + Vector(units[face[1]])

                verts = [np.dot(para, p0), np.dot(para, p1), np.dot(para, p2), np.dot(para, p3)]
                verts = [v[0] * u3 + v[1] * v3 for v in verts]
                poly = Polygon(vertices=verts, index=f * 1000 + c, color=colors[f])
                poly.appear(transition_time=0)
                polygons.append(poly)
                c += 1
            f += 1

        self.t0 = t0

    def test_simplex_convex_hull(self):
        t0 = 0
        size = 5

        ibpy.set_hdri_background("forest", 'exr', background='black', no_transmission_ray='True',
                                 rotation_euler=[0, 0, 3 * np.pi / 2])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE)
        ibpy.set_camera_location(Vector([13, 13, 13]))
        empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(empty)
        ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=0)

        simplex = Simplex(dim=3, vertices=random_points(dim=3, n=4, domain=10))
        spheres = [Sphere(0.1, location=Vector(v), color='example') for v in simplex.structure[0]]
        [sphere.appear(begin_time=t0, transition_time=0.5) for sphere in spheres]

        t0 += 1.5
        polygons = [Polygon(vertices=simplex.structure[0][list(face)], color='plastic_drawing') for face in
                    simplex.structure[2]]
        for polygon in polygons:
            t0 = polygon.appear(begin_time=t0, transition_time=0.5)

        normals = []
        n = 0
        for face in simplex.structure[2]:
            face_vertices = simplex.structure[0][list(face)]
            center = mean(face_vertices)
            normal = simplex.structure[2][face][0]
            p_arrow = PArrow(start=Vector(center), end=Vector(center) + Vector(normal), color='important',
                             name="Normal" + str(n))
            normals.append(p_arrow)
            p_arrow.grow(begin_time=t0, transition_time=0.5)
            n += 1

        t0 += 1

        points = random_points(dim=3, n=100, domain=30)
        ch = convex_hull(points)

        spheres = [Sphere(0.3, location=point, color='plastic_important') for point in points]
        for sphere in spheres:
            sphere.grow(begin_time=t0, transition_time=0.5)
            t0 += 0.1

        faces = [Polygon(vertices=[Vector(v) for v in ch.points[indices]], color='plastic_joker', alpha=0.3) for indices
                 in ch.simplices]
        for face in faces:
            face.appear(begin_time=t0, transition_time=0.5)
            t0 += 0.1

        self.t0 = t0

    def intro_morph(self):
        t0 = 0
        size = 5

        ibpy.set_hdri_background("forest", 'exr', background='black', no_transmission_ray='True',
                                 rotation_euler=[0, 0, 3 * np.pi / 2])
        ibpy.set_hdri_strength(1, transition_time=0)
        ibpy.set_render_engine(engine=BLENDER_EEVEE)
        ibpy.set_camera_location(Vector([13, 13, 13]))
        empty = EmptyCube(location=[0, 0, 0])
        ibpy.set_camera_view_to(empty)
        ibpy.camera_zoom(lens=45, begin_time=t0, transition_time=0)

        normal = Vector([1, 1, 1])
        normal.normalize()
        start = timer()
        sel_points = close_points_3d(normal, 11)
        end = timer()
        print((end - start), "s", sel_points)
        base_points = find_base_points(sel_points, 3)
        faces = face_directions(3)
        units = unit_tuples(3)

        f = 0
        c = 0
        r2 = np.sqrt(2)
        r6 = np.sqrt(6)
        u = Vector((1 / r2, -1 / r2, 0))
        v = Vector((1 / r6, 1 / r6, -2 / r6))
        p = projector(u, v)
        polygons = []
        colors = ["plastic_drawing", "plastic_important", "plastic_custom2"]
        for bps, face in zip(base_points, faces):
            for pt in bps:
                v0 = Vector(pt)
                v1 = Vector(pt) + Vector(units[face[0]])
                v2 = Vector(pt) + Vector(units[face[0]]) + Vector(units[face[1]])
                v3 = Vector(pt) + Vector(units[face[1]])
                if f == 1:
                    verts = [v0, v3, v2, v1]
                else:
                    verts = [v0, v1, v2, v3]
                poly = Polygon(vertices=verts, index=f * 1000 + c, color=colors[f])
                poly.appear(transition_time=0)
                polygons.append(poly)
                c += 1
            f += 1

        rot_box = BObject(name="Box", children=polygons)
        rot_box.appear(begin_time=t0, transition_time=0)

        t0 += 0.5

        t0 = 0.5 + rot_box.rotate(rotation_euler=[0, 0, np.pi / 6], begin_time=t0, transition_time=1)

        for poly in polygons:
            poly.morph_to(projector=lambda v: p @ v, begin_time=t0, transition_time=1)
        t0 += 1.5

        t0 = 0.5 + rot_box.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=1)
        self.t0 = t0

    def citation(self):
        t0 = 0
        ibpy.set_render_engine(engine=BLENDER_EEVEE)
        one = Plane(color='image', src='cite_book.png', location=[-6, 3, 0],
                    scale=[5, 2.5, 2], emission=1, solid=0.2, rotation_euler=[np.pi / 2, 0, 0])
        t0 = one.appear(begin_time=t0)

        two = Plane(color='image', src='cite_dice.png', location=[6, 3, 0],
                    scale=[5, 2.5, 2], emission=1, solid=0.2, rotation_euler=[np.pi / 2, 0, 0])
        t0 = two.appear(begin_time=t0)

        three = Plane(color='image', src='cite_mirror.png', location=[-6, -3, 0],
                      scale=[5, 2.5, 2], emission=1, solid=0.2, rotation_euler=[np.pi / 2, 0, 0])
        t0 = three.appear(begin_time=t0)

        four = Plane(color='image', src='cite_glas.png', location=[6, -3, 0],
                     scale=[5, 2.5, 2], emission=1, solid=0.2, rotation_euler=[np.pi / 2, 0, 0])
        t0 = four.appear(begin_time=t0)

        t0 += 30
        t0 = one.disappear(begin_time=t0)
        t0 = two.disappear(begin_time=t0)
        t0 = three.disappear(begin_time=t0)
        t0 = four.disappear(begin_time=t0)

        self.t0 = t0


if __name__ == '__main__':
    try:
        example = Penrose()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        choice = input("Choose scene:")
        print("Your choice: ", choice)
        selected_scene = dictionary[int(choice)]
        # selected_scene = "plane_rotation_5d"
        # selected_scene = "maybe_penrose2"
        # selected_scene = "penrose_tiling"
        # selected_scene = "penrose_tiling2"
        # selected_scene = "penrose_merger"
        # selected_scene = "penrose_merger2"
        # selected_scene = "optimize_object_creation"
        # selected_scene = "optimize_object_creation2"

        #selected_scene = "hexagons"
        # selected_scene = "hexagons2"

        # scenes with geometry nodes!!!
        # selected_scene = "geo_penrose"
        # selected_scene = "title"

        # selected_scene = "hyper_cube"
        # selected_scene = "branding"
        # selected_scene = "hyper_cube_data"
        # selected_scene = "dynamical_convex_hull"
        # selected_scene = "five_d_rotations"
        # selected_scene = "projection_2d"
        # selected_scene = "projection_2d_intro"
        # selected_scene = "projection_2d_voronoi"
        # selected_scene = "intro"
        # selected_scene = "penrose_rotation"
        # selected_scene = "five_d_rotations2"
        # selected_scene = "convex_hull_2d"
        # selected_scene = "penrose_rotation2"
        # selected_scene = "convex_hull_3d"
        # selected_scene = "five_d_rotations"
        # selected_scene = "three_d_rotations"

        example.create(name=selected_scene, resolution=[1920, 1080], start_at_zero=True)

        # example.render(debug=True)
        # doesn't work
        # example.final_render(name=selected_scene,debug=False)
    except:
        print_time_report()
        raise ()
