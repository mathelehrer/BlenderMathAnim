import os
from collections import OrderedDict
from math import tau

import bpy
import numpy as np
from numpy import pi, sqrt, cos, sin

from compositions.compositions import create_glow_composition, set_alpha_composition
from geometry_nodes.geometry_nodes_modifier import FundamentalDomainCoverModifier, \
    TriangularGridModifier, AnalyserModifier, SubstitutionModifier, SubstitutionModifierFull, HexagonalTilingModifier
from geometry_nodes.geometry_nodes_modifier import MakeFrameModifier, BarnsleyFernModifier, \
    SierpinskiTriangleModifier, ApollonianGasketModifier
from interface import ibpy
from interface.ibpy import Vector, Matrix, Quaternion, get_node_from_shader, get_geometry_node_from_modifier, \
    create_mesh, change_default_integer
from objects.bderivation import BDerivation
from objects.bobject import BObject
from objects.circle import BezierCircle
from objects.coordinate_system import CoordinateSystem2
from objects.cube import Cube
from objects.curve import GeoCurve, Curve, BezierDataCurve
from objects.cylinder import Cylinder
from objects.derived_objects.flag import Flag
from objects.derived_objects.p_arrow import PArrow
from objects.derived_objects.pencil import Pencil
from objects.derived_objects.person_with_cape import PersonWithCape
from objects.derived_objects.pin import Pin
from objects.digital_number import DigitalRange
from objects.display import Display
from objects.empties import EmptyCube
from objects.floor import Floor
from objects.geometry.sphere import Sphere
from objects.hat_tile import HatTile, LabbeSelingerModifier, LabbeSelingerOptimizedModifier, \
    LabbeSelingerColorModifier, \
    HatTileSubstitutionModifier, HatClusterCsvModifier, DeadEndClusterModifier, _hat_vertices14, \
    CommandmentTableModifier, create_instruction_table
from objects.light.light import SpotLight, PointLight
from objects.plane import Plane
from objects.polygon import Polygon
from objects.quadrilateral import BQuadrilateral
from objects.tex_bobject import SimpleTexBObject
from objects.text import Text
from perform.scene import Scene
from shader_nodes.shader_nodes import (TextureCoordinate, SeparateXYZ, InputValue, MathNode,
                                       CombineXYZ, TextureImage, VectorMathNode, PrincipledBSDF)
from tools.images import ImageCreator
from utils.constants import FRAME_RATE, DATA_DIR, DEFAULT_ANIMATION_TIME, IMG_DIR
from utils.utils import print_time_report, flatten
from video_hat_tile.substitution_explainer import (
    play_substitution, H1_TILE, H2_TILE, H7_TILE, H8_TILE, SUPER_H7_TILE, SUPER_H8_TILE)

r3 = sqrt(3)
r5 = sqrt(5)
phi = 0.5 * (r5 + 1)


# ===========================================================================
# Hat-tile geometry helpers (14-point version)
#
# For the 14-point hat tile (dir_in=0, ref=False, pt=0, scale=1) every vertex
# coordinate is exact:   x = Bx * sqrt(3) / 2   and   y = Ay / 2
# with integer Bx, Ay. This lets us label vertices with exact LaTeX coordinates
# (sqrt(3) and fractions) instead of floats.
# ===========================================================================

def _hat14_symbolic():
    """Return the 14-point hat vertices as integer pairs ``(Ax, By)`` where
    ``y = By*sqrt(3)/2`` and ``x = Ax/2``."""
    verts = _hat_vertices14(dir_in=-1, ref=False, pt=0, scale=1.0)
    out = []
    for x, y in verts:
        by = int(round(2 * y / r3))
        ax = int(round(2 * x))
        # the 14-point hat is built from sqrt(3)/1/2 edges at 30-degree angles,
        # so this reconstruction is exact -- guard against silent drift.
        assert abs(by * r3 / 2 - y) < 1e-6 and abs(ax / 2 - x) < 1e-6, (x, y)
        out.append((ax, by))
    return out


def _fmt_rational_half(n):
    """LaTeX for ``n/2``."""
    if n == 0:
        return "0"
    if n % 2 == 0:
        return str(n // 2)
    sign = "-" if n < 0 else ""
    return r"%s\frac{%d}{2}" % (sign, abs(n))


def _fmt_sqrt3_half(n):
    """LaTeX for ``(n/2)*sqrt(3)``."""
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    m = abs(n)
    if m % 2 == 0:
        c = m // 2
        coeff = "" if c == 1 else str(c)
        return r"%s%s\sqrt{3}" % (sign, coeff)
    num = r"\sqrt{3}" if m == 1 else r"%d\sqrt{3}" % m
    return r"%s\frac{%s}{2}" % (sign, num)


def _coord_flag_text(ax, by):
    """Boxed LaTeX ``(x, y)`` label for a hat-tile vertex flag."""
    ys = _fmt_sqrt3_half(by)
    xs = _fmt_rational_half(ax)
    return r"\text{\fbox{$\rule{0em}{2ex}\left(" + xs + r",\;" + ys + r"\right)$}}"


def _hat_kites(scale=1):
    """The eight kites whose union is the 14-point hat tile.

    The hat is a *polykite*: eight congruent kites of the deltoidal
    trihexagonal (``kite``) grid, obtained by overlaying a triangular lattice
    ``U`` (edge ``2*sqrt(3)``) with its dual.  Every kite is
    ``[outer_vertex, edge_midpoint, triangle_centre, edge_midpoint]``.  We
    rebuild that grid around the hat outline and keep the eight kites whose
    centroid falls inside the outline -- their areas sum exactly to the hat
    area and none pokes outside.  The kites are returned ordered
    counter-clockwise about the hat centroid, so a rainbow colour sweep runs
    smoothly around the tile.

    Returns ``(kites, centre)`` with ``kites`` a list of eight ``(4, 2)``
    float arrays and ``centre`` the hat centroid as a length-2 array.
    """
    hat = np.array(_hat_vertices14(dir_in=-1, ref=False, pt=0, scale=scale))
    base = np.array([1.5, -r3 / 2]) * scale
    e1 = np.array([3.0, r3]) * scale
    e2 = np.array([0.0, 2 * r3]) * scale

    def triangle_kites(P, Q, R):
        C = (P + Q + R) / 3.0
        return [np.array([A, (A + b) / 2, C, (A + c) / 2])
                for A, b, c in ((P, Q, R), (Q, P, R), (R, P, Q))]

    def inside(pt, poly):
        x, y = pt
        n = len(poly)
        ins = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                ins = not ins
            j = i
        return ins

    kites = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = base + i * e1 + j * e2
            for tri in ((p, p + e1, p + e2), (p + e1, p + e2, p + e1 + e2)):
                for k in triangle_kites(*tri):
                    c = k.mean(axis=0)
                    if inside(c, hat) and not any(np.allclose(c, s.mean(axis=0), atol=1e-4) for s in kites):
                        kites.append(k)
    centre = hat.mean(axis=0)
    kites.sort(key=lambda k: np.arctan2(k.mean(axis=0)[1] - centre[1],
                                        k.mean(axis=0)[0] - centre[0]))
    return kites, centre


# ===========================================================================
# Shared helpers
# ===========================================================================


def _setup_camera(distance=20.0):
    ibpy.set_camera_location(location=[0, 0, distance])
    empty = EmptyCube(location=Vector((0, 0, 0)))
    ibpy.set_camera_view_to(empty)
    return empty


def _setup_standard_camera(distance=20, shift_x=0):
    ibpy.set_camera_location(location=[shift_x, -distance, 0])
    empty = EmptyCube(location=Vector((shift_x, 0, 0)))
    ibpy.set_camera_view_to(empty)
    return empty


def _setup_tilted_camera(center=(0, 0, 0), distance=55.0, tilt=0.25):
    """Camera mostly top-down but shifted toward -y so the extruded tiles show
    their 3-dimensionality.  ``tilt`` is the side offset as a fraction of
    ``distance``."""
    center = Vector((center[0], center[1], 0)) if len(center) == 2 else Vector(center)
    ibpy.set_camera_location(location=center + Vector((0, -tilt * distance, distance)))
    empty = EmptyCube(location=center)
    ibpy.set_camera_view_to(empty)
    return empty


def _setup_render(hdri="kloofendal_misty_morning_puresky_4k",
                  engine="BLENDER_EEVEE",
                  transparent=True,
                  exposure=1):
    ibpy.set_hdri_background(hdri,
                             'exr', rotation_euler=[pi / 3 * 2, 0, 0], simple=True,
                             transparent=transparent)
    ibpy.set_render_engine(
        denoising=False, transparent=transparent,
        resolution_percentage=100,
        engine=engine, taa_render_samples=128, frame_start=1, exposure=exposure
    )


def _rgba_socket(collection, name):
    """A Mix node's RGBA-typed 'A'/'B'/'Result' socket, disambiguated by type.

    ``ShaderNodeMix`` keeps one same-named socket per ``data_type`` (VALUE,
    VECTOR, RGBA, ROTATION) side by side in ``.inputs``/``.outputs`` at once;
    a plain ``inputs['A']`` name lookup silently returns the FLOAT one (first
    in the collection), so a colour link lands on an inactive socket and the
    node keeps mixing its default black/white instead.
    """
    return next(s for s in collection if s.name == name and s.type == 'RGBA')


def _make_jump_slide_material(red_image, blue_image, window, y_center,
                              name="JumpSlideMaterial"):
    """Two flat tiling snapshots that slide toward each other; wherever their
    footprints overlap, show their colour difference instead of either image.

    Built for :meth:`HatTileScene.fractal_jump`: ``red_image``/``blue_image``
    (files in ``IMG_DIR``) are the same pre-rendered tilings used by
    ``fractal_jump_images.py``. The plane carrying this material must sit at
    the origin with its mesh offset via ``u``/``v`` (as with the
    ``hat_tile_fractal`` shader elsewhere in this scene) so its "Object"
    texture coordinate equals world position; each texture's own UV is then
    ``uv = (world_xy - centre) / window + 0.5``, and ``Extension='CLIP'``
    makes every fragment outside the resulting [0,1]^2 square fully
    transparent, regardless of the source image's own (opaque) alpha. Each
    texture's ``centre.x`` is exposed as an animatable Value node (returned
    alongside the material) so the caller can slide it from its home window
    to the shared centre with ``ibpy.change_value``; ``centre.y`` is fixed
    since both windows sit on one row.

    :return: (material, red_centre_x_node, blue_centre_x_node)
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    out = nodes.get("Material Output")
    for n in list(nodes):
        if n != out:
            nodes.remove(n)

    coord = TextureCoordinate(tree, location=(-14, 0), std_out='Object', hide=True)
    xyz = SeparateXYZ(tree, location=(-13, 0), vector=coord.std_out, hide=True)

    def texture_uv(image_name, center_x_value, row):
        cx = InputValue(tree, location=(-13, row), value=center_x_value, hide=True,
                        name=image_name + "_CenterX")
        u = MathNode(tree, location=(-12, row), operation='SUBTRACT',
                    input0=xyz.std_out_x, input1=cx.std_out, hide=True)
        u = MathNode(tree, location=(-11, row), operation='DIVIDE',
                    input0=u.std_out, input1=float(window), hide=True)
        u = MathNode(tree, location=(-10, row), operation='ADD',
                    input0=u.std_out, input1=0.5, hide=True)
        v = MathNode(tree, location=(-12, row - 0.5), operation='SUBTRACT',
                    input0=xyz.std_out_y, input1=float(y_center), hide=True)
        v = MathNode(tree, location=(-11, row - 0.5), operation='DIVIDE',
                    input0=v.std_out, input1=float(window), hide=True)
        v = MathNode(tree, location=(-10, row - 0.5), operation='ADD',
                    input0=v.std_out, input1=0.5, hide=True)
        uv = CombineXYZ(tree, location=(-9, row), x=u.std_out, y=v.std_out, z=0, hide=True)
        tex = TextureImage(tree, location=(-8, row), image_name=image_name,
                           extension='CLIP', vector=uv.std_out, hide=True)
        return cx, tex

    red_cx, red_tex = texture_uv(red_image, -window, 2)
    blue_cx, blue_tex = texture_uv(blue_image, window, -2)
    red_alpha = red_tex.node.outputs["Alpha"]
    blue_alpha = blue_tex.node.outputs["Alpha"]

    overlap = MathNode(tree, location=(-6, 0), operation='MULTIPLY',
                       input0=red_alpha, input1=blue_alpha, hide=True, label="Overlap")
    only_red = MathNode(tree, location=(-6, 1), operation='SUBTRACT',
                        input0=red_alpha, input1=overlap.std_out, hide=True)
    only_blue = MathNode(tree, location=(-6, -1), operation='SUBTRACT',
                         input0=blue_alpha, input1=overlap.std_out, hide=True)

    diff_node = nodes.new(type="ShaderNodeMix")
    diff_node.data_type, diff_node.blend_type, diff_node.hide = 'RGBA', 'DIFFERENCE', True
    diff_node.location = (-6 * 200, 3 * 100)
    diff_node.inputs[0].default_value = 1.0  # uniform Factor (the VALUE one)
    links.new(red_tex.std_out, _rgba_socket(diff_node.inputs, 'A'))
    links.new(blue_tex.std_out, _rgba_socket(diff_node.inputs, 'B'))
    diff_out = _rgba_socket(diff_node.outputs, 'Result')

    red_part = VectorMathNode(tree, location=(-4, 1), operation='SCALE',
                              input0=red_tex.std_out, scale=only_red.std_out, hide=True)
    blue_part = VectorMathNode(tree, location=(-4, -1), operation='SCALE',
                               input0=blue_tex.std_out, scale=only_blue.std_out, hide=True)
    diff_part = VectorMathNode(tree, location=(-4, 3), operation='SCALE',
                               input0=diff_out, scale=overlap.std_out, hide=True)

    color = VectorMathNode(tree, location=(-2, 1), operation='ADD',
                           input0=red_part.std_out, input1=blue_part.std_out, hide=True)
    color = VectorMathNode(tree, location=(-1, 1), operation='ADD',
                           input0=color.std_out, input1=diff_part.std_out, hide=True)

    alpha = MathNode(tree, location=(-4, 2), operation='ADD',
                     input0=red_alpha, input1=blue_alpha, hide=True)
    alpha = MathNode(tree, location=(-3, 2), operation='SUBTRACT',
                     input0=alpha.std_out, input1=overlap.std_out, hide=True)

    bsdf = PrincipledBSDF(tree, location=(2, 0), base_color=color.std_out,
                         alpha=alpha.std_out, hide=True)
    links.new(bsdf.node.outputs['BSDF'], out.inputs['Surface'])

    return mat, red_cx.node, blue_cx.node


class HatTileScene(Scene):
    def __init__(self):
        self.t0 = 0
        self.sub_scenes = OrderedDict([
            ('short', {"duration": 70}),
            ('naked_triangular_grid', {"duration": 30}),
            ("ifs", {"duration": 5}),
            ('inv9', {"duration": 5}),
            ('fib4', {"duration": 5}),
            ('toc', {"duration": 15}),
            ('intro_fib', {"duration": 15}),
            ('intro_fib_overlay', {"duration": 15}),
            ('intro_algebra_overlay', {"duration": 30}),
            ('recursion_to_quadratic', {"duration": 30}),
            ('outro_fibonacci_matrix', {"duration": 60}),
            ('boring_tilings', {"duration": 20}),
            ('fractal_shader', {"duration": 15}),
            ('the_background_fractal', {"duration": 35}),
            ('fundamental_domain', {"duration": 40}),
            ('fundamental_domain_overlay', {"duration": 30}),
            ('labbe_selinger', {"duration": 60}),
            ('labbe_selinger_cheap', {"duration": 60}),
            ('fractal_jump', {"duration": 30}),
            ('simple_hat', {"duration": 15}),
            ('hat_from_code', {"duration": 15}),
            ('fundamental_domain_cover', {"duration": 15}),
            ('trapezoid_analyser', {'duration': 22}),
            ('barnsley_fern', {'duration': 20}),
            ('sierpinski_triangle', {'duration': 30}),
            ('apollonian_gasket', {'duration': 28}),
            ('substitution_intro', {'duration': 50}),
            ('substitution_vs_ifs', {'duration': 20}),
            ('substitution', {'duration': 15}),
            ('substitution_hat_overlay', {'duration': 15}),
            ('substitution_hat', {'duration': 25}),
            ('substitution_explainer', {'duration': 26}),
            ('substitution_overlay', {'duration': 15}),
            ('substitution_overlay2', {'duration': 20}),
            ('substitution_explainer2', {'duration': 18}),
            ('substitution_explainer3', {'duration': 20}),
            ('hat_layout', {'duration': 15}),
            ('hat_introduction', {'duration': 15}),
            ('hat_tile_basics', {'duration': 45}),
            ('hat_from_kites', {'duration': 20}),
            ('commandments', {'duration': 35}),
            ('title', {'duration': 50}),
            ('overlay', {'duration': 65}),
            ('rot_sym_intro', {'duration': 50}),
            ('rot_sym', {'duration': 45}),
            ('dead_end_leaves', {'duration': 75}),
            ('show_orientations', {'duration': 12}),
        ])
        super().__init__(light_energy=1, transparent=False)

    def ifs(self):
        t0 = 0.3
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        camera_empty = _setup_camera(0)
        ibpy.set_camera_location(location=[0, -10, 0])
        ibpy.set_camera_view_to(camera_empty)
        create_glow_composition(threshold=1)

        ifs = Text(r"\text{IFS}", location=[-1.5, 0, 0], color="example", outline_color="joker",
                   emission_outline=1, scale=3)

        t0 = 0.5 + ifs.write(begin_time=t0, transition_time=1.5)
        t0 = 0.5 + ifs.unwrite(begin_time=t0, transition_time=1.5)

        self.t0 = t0

    def fib4(self):
        t0 = 0.3
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        camera_empty = _setup_camera(0)
        ibpy.set_camera_location(location=[0, -10, 0])
        ibpy.set_camera_view_to(camera_empty)

        numbers = [1, 8, 55, 377, 2584]
        x = [-5.5, -4.5, -3.5, -0.75, 2.5]
        for i, number in enumerate(numbers):
            if i < 2:
                number_str = str(number)
                colors = ["example"]
            else:
                number_str = str(number) + r"\mbox{\footnotesize $=\!7\!\times\!" + str(numbers[i - 1]) + r"-\!" + str(
                    numbers[i - 2]) + r"$}"
                colors = flatten(
                    [["example"] * len(str(number)), ["text"] * 3, ["example"] * len(str(numbers[i - 1])), ["text"],
                     ["example"]])
            bob = SimpleTexBObject(number_str, location=[x[i], 0, 0], text_size="large", color=colors, aligned="left")
            t0 = 0.5 + bob.write(begin_time=t0, transition_time=0.1 * len(str(number)))

        self.t0 = t0

    def inv9(self):
        t0 = 0.3
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        camera_empty = _setup_camera(0)
        ibpy.set_camera_location(location=[0, -10, 0])
        ibpy.set_camera_view_to(camera_empty)

        lines = [
            SimpleTexBObject(str(x) + r"^2-7\times" + str(x) + r"\times" + str(y) + "+" + str(y) + r"^2=9",
                             aligned="right",
                             location=[3, 0, 3 - i], text_size="large",
                             color=flatten(
                                 [["example"] * len(str(x)), ["text"] * 4, ["example"] * len(str(x)), ["text"],
                                  ["example"] * len(str(y)), ["text"], ["example"] * len(str(y)), ["text"]]))
            for i, (x, y) in enumerate([(8, 1), (55, 8), (377, 55)])]

        for line in lines:
            t0 = 0.5 + line.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def short(self):
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        camera_empty = _setup_camera(0)
        camera_circle = BezierCircle(location=Vector(), radius=20, rotation_euler=[0, pi / 2, 0])
        ibpy.set_camera_follow(camera_circle)
        ibpy.camera_follow(camera_circle, initial_value=0.25, final_value=0.25, begin_time=0, transition_time=0)

        t0 = 0
        create_glow_composition(threshold=0.9, type="BLOOM", size=4)

        fractal = Plane(name="FractalPlane", u=[-20, 20], v=[-20, 20], color="hat_tile_fractal", roughness=0.1,
                        metallic=0.9)
        fractal.grow(begin_time=t0, transition_time=0)
        fractal.move(direction=[0, 0, -0.01], begin_time=t0, transition_time=0)

        # setup initial state
        depth_node = get_node_from_shader(ibpy.get_material_at_slot(fractal, 0), label="Depth")
        rot_zero = Quaternion()
        rot_quat = Quaternion([1, 0, 0], -pi / 4)
        fractal.rotate(rotation_quaternion=rot_quat @ rot_zero, begin_time=0, transition_time=0)
        camera_empty.move_to(target_location=(0, 35, 0), begin_time=0, transition_time=0)

        hat_empty = EmptyCube(location=Vector((0, 0, 0)), hame="HatTileEmpty")

        t0 = camera_empty.move_to(target_location=(0, 0.1, 0), begin_time=0, transition_time=5)
        fractal.rotate(rotation_quaternion=rot_zero, begin_time=t0, transition_time=5)
        ibpy.camera_zoom(lens=100, begin_time=t0, transition_time=5)
        ibpy.camera_zoom(lens=2500, begin_time=t0 + 5, transition_time=5)
        t0 = ibpy.change_value(depth_node, from_value=0, to_value=10, begin_time=t0, transition_time=10)

        ibpy.camera_zoom(lens=230, begin_time=t0, transition_time=5)
        t0 = camera_circle.rescale(rescale=2, begin_time=t0, transition_time=5)

        hat_plane = Plane(name='Carrier')
        hat_plane.appear(begin_time=0, transition_time=0)

        mod = LabbeSelingerModifier(roughness=0.1,
                                    frame_driven_displacement_function=[
                                        "2,t,*,cos,t,cos,*,1,*",
                                        "2,t,*,cos,t,sin,*,3,*",
                                        "0"
                                    ],
                                    grid_size=100,
                                    domain=[0, tau],
                                    begin_time=t0, transition_time=25)

        hat_plane.add_mesh_modifier(type='NODES', node_modifier=mod)
        t0 = hat_plane.grow(begin_time=t0, transition_time=0)
        #
        # rot_quad = Quaternion([1,0,0],-pi/4)

        # start with a single hat tile
        gs_node = get_geometry_node_from_modifier(mod, "GridFilter")
        hat_scale_node = get_geometry_node_from_modifier(mod, "HatScale")
        grid_radius_node = get_geometry_node_from_modifier(mod, "GridRadius")
        extrude_node = get_geometry_node_from_modifier(mod, "ExtrudeScale")
        ibpy.change_value(hat_scale_node, from_value=0, to_value=0.25, begin_time=t0, transition_time=0.5)
        ibpy.change_value(grid_radius_node, from_value=0.025, to_value=0.04, begin_time=0, transition_time=0)

        for mat in mod.hat_materials:
            ibpy.change_alpha_of_material(mat, from_value=0, to_value=0.5, begin_time=t0, transition_time=0.5)

        curve = GeoCurve(function=["2,t,*,cos,t,cos,*,0.1,+", "3,2,t,*,cos,*,t,sin,*,0.01,+", "0.0"],
                         domain=[0, 1.0 * tau], points=1000,
                         thickness=0.005, resolution=8, color="red")

        # triangular grid

        tri_carrier = Plane(name='FDTriGrid')
        tri_mod = TriangularGridModifier(grid_n=12, frame_driven_displacement_function=[
            "2,t,*,cos,t,cos,*,1,*",
            "2,t,*,cos,t,sin,*,3,*",
            "0"
        ], colors=['red', 'joker'], edge_thickness=0.005, shift=Vector([0.1, 0.01, 0]),
                                         show_fundamental_plane=False, dot_size=0.025, domain=[0, tau],
                                         begin_time=t0, transition_time=25)
        tri_carrier.add_mesh_modifier(type='NODES', node_modifier=tri_mod)
        tri_carrier.appear(begin_time=t0, transition_time=1.0)

        curve_begin = t0
        print("curve_begin", curve_begin)
        # drive hat_empty so it rides the growing tip of the curve, synchronized
        # with the growth percentage (t - curve_begin) / 15
        ibpy.add_driver(hat_empty,
                        functions=["2,t,*,cos,t,cos,*,0.1,+", "3,2,t,*,cos,*,t,sin,*,0.01,+", "0.05"],
                        domain=[0, 1.0 * tau], begin_time=curve_begin, transition_time=25)
        ibpy.camera_set_damped_track(hat_empty, influence=0, begin_time=curve_begin)
        ibpy.camera_change_track_influence(hat_empty, start=0, end=1, begin_time=curve_begin, transition_time=5)
        ibpy.camera_zoom(lens=500, begin_time=curve_begin, transition_time=5)
        ibpy.camera_zoom(lens=230, begin_time=curve_begin + 15, transition_time=5)
        ibpy.camera_change_track_influence(hat_empty, start=1, end=0, begin_time=curve_begin + 15, transition_time=5)

        ibpy.change_value(hat_scale_node, from_value=0.25, to_value=0.99, begin_time=curve_begin + 15,
                          transition_time=5)
        for mat in mod.hat_frame_materials:
            ibpy.change_emission_of_material(mat, from_value=0, to_value=1, begin_frame=(curve_begin + 20) * FRAME_RATE,
                                             frame_duration=6)
            ibpy.change_emission_of_material(mat, from_value=1, to_value=0, begin_frame=(curve_begin + 21) * FRAME_RATE,
                                             frame_duration=6)
        ibpy.change_default_integer(gs_node, from_value=0, to_value=100, begin_time=curve_begin + 15, transition_time=5)
        ibpy.change_default_integer(gs_node, from_value=100, to_value=200, begin_time=curve_begin + 20,
                                    transition_time=5)
        ibpy.camera_zoom(lens=75, begin_time=curve_begin + 20, transition_time=5)

        print("last change: ", FRAME_RATE * (curve_begin + 25))
        latest = 0.5 + ibpy.change_default_integer(gs_node, from_value=200, to_value=2000, begin_time=curve_begin + 25,
                                                   transition_time=20)
        ibpy.change_default_value(extrude_node, from_value=0.01, to_value=0.5, begin_time=curve_begin + 25,
                                  transition_time=5)
        ibpy.change_default_value(hat_scale_node, from_value=0.99, to_value=0.975, begin_time=curve_begin + 25,
                                  transition_time=1)

        for mat in mod.hat_materials:
            ibpy.change_alpha_of_material(mat, from_value=0.5, to_value=0.75, begin_time=curve_begin + 20,
                                          transition_time=5)
            ibpy.change_alpha_of_material(mat, from_value=0.75, to_value=1, begin_time=curve_begin + 25,
                                          transition_time=20)
        curve.disappear(alpha=0.001, begin_time=curve_begin + 20, transition_time=0.5)

        rot_zero = Quaternion([0, 0, 1], pi / 4)
        hat_plane.rotate(rotation_quaternion=rot_zero, begin_time=curve_begin + 20, transition_time=5)
        fractal.rotate(rotation_quaternion=rot_zero, begin_time=curve_begin + 20, transition_time=5)
        curve.rotate(rotation_quaternion=rot_zero, begin_time=curve_begin + 20, transition_time=5)
        tri_carrier.rotate(rotation_quaternion=rot_zero, begin_time=curve_begin + 20, transition_time=5)
        ibpy.camera_follow(camera_circle, initial_value=0.25, final_value=0.11, begin_time=curve_begin + 22,
                           transition_time=5)

        t0 = 0.5 + curve.grow(begin_time=curve_begin, transition_time=25)

        self.t0 = latest

    def naked_triangular_grid(self):
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        camera_empty = _setup_camera(20)
        ibpy.set_camera_view_to(camera_empty)
        t0 = 0

        # triangular grid

        tri_carrier = Plane(name='FDTriGrid')
        tri_mod = TriangularGridModifier(grid_n=0, colors=['important', 'joker'], edge_thickness=0.005, shift=Vector([0, 0, 0]),
                                         show_fundamental_plane=False, dot_size=0.025, domain=[0, tau],
                                         begin_time=t0, transition_time=25)
        tri_carrier.add_mesh_modifier(type='NODES', node_modifier=tri_mod)
        t0 = tri_carrier.appear(begin_time=t0, transition_time=1.0)

        l_node = get_geometry_node_from_modifier(tri_mod,label="Lambda")
        grid_size_node = get_geometry_node_from_modifier(tri_mod, label="GridSize")
        edge_node = get_geometry_node_from_modifier(tri_mod, label="EdgeThickness")
        dot_node = get_geometry_node_from_modifier(tri_mod, label="DotRadius")

        t0 = 0.5 + ibpy.change_default_integer(grid_size_node,from_value=0,to_value=40,begin_time=t0,transition_time=3)

        ibpy.change_default_value(edge_node,from_value=0.005,to_value=0.0,begin_time=t0,transition_time=1)
        ibpy.change_default_value(dot_node,from_value=0.1,to_value=0.1,begin_time=t0,transition_time=1)
        t0 = 0.5 + ibpy.change_default_value(l_node, from_value=0, to_value=1, begin_time=t0, transition_time=15)

        self.t0 = t0

    def toc(self):
        t0 = 0

        _setup_render()
        # the camera always points at the tracked empty; its world position is
        # driven by smooth Bezier fly-paths instead of straight waypoint hops.
        ibpy.set_camera_location(location=[0, -20, 0])
        ibpy.set_camera_lens(lens=30)

        lines = [
            SimpleTexBObject(r"\text{Fibonacci cluster}", color="example", aligned="left", text_size="Large",
                             location=[-8, 0, 3]),
            SimpleTexBObject(r"\text{Labbé--Selinger construction}", color="example", aligned="left", text_size="Large",
                             location=[-8, 0, 1]),
            SimpleTexBObject(r"\text{Secret connection}", color="example", aligned="left", text_size="Large",
                             location=[-8, 0, -1]),
            SimpleTexBObject(r"\text{Rotationally Symmetric Tilings}", color="example", aligned="left",
                             text_size="Large", location=[-8, 0, -3]),
        ]

        for line in lines:
            t0 = 2.5 + line.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def intro_fib(self):
        t0 = 0
        set_alpha_composition()
        _setup_render()
        # the camera always points at the tracked empty; its world position is
        # driven by smooth Bezier fly-paths instead of straight waypoint hops.
        camera_empty = _setup_camera(30)
        ibpy.set_camera_lens(lens=30, clip_end=100000)
        set_alpha_composition()

        # prepare the first four instances of the substitution modifier
        positions = [
            Vector([-5, -5, 0]),
            Vector([-5, 5, 0]),
            Vector([7.5, 5, 0]),
            Vector([7.5, -5, 0]),
        ]

        dialers = []
        carriers = []
        for i in range(4):
            carrier = Plane(name='SubstitutionCarrier')
            carrier.move_to(target_location=positions[i], begin_time=0, transition_time=0)
            carrier.rotate(rotation_euler=[0, 0, 50 / 180 * pi], begin_time=0, transition_time=0)
            mod = HatTileSubstitutionModifier(level=0, color_scheme=1, extrude_scale=1, hat_scale=0.975)
            dialers.append(ibpy.get_geometry_node_from_modifier(mod, label="Level"))
            carrier.add_mesh_modifier(type='NODES', node_modifier=mod)
            carriers.append(carrier)
            print("1 should appear", t0)
            t0 = 0.5 + carrier.appear(begin_time=0, transition_time=1)

        ibpy.camera_move(shift=[0, 0, 6], begin_time=t0, transition_time=1)
        new_positions = [
            Vector([-5, -5, 0]),
            Vector([-5 + 1.75, 5 + 4.75, 0]),
            Vector([7.5 + 6.5, 5 + 4.75, 0]),
            Vector([7.5 + 6.5, -5 + 3.4, 0]),
        ]

        for i in range(0, 4):
            carriers[i].move_to(target_location=new_positions[i], begin_time=t0, transition_time=1)
            if i > 0:
                print("8 should appear", t0)
                ibpy.change_default_integer(dialers[i], from_value=0, to_value=1, begin_time=t0 + 1, transition_time=0)

        t0 += 1.5

        ibpy.camera_move(shift=[0, 0, 44], begin_time=t0, transition_time=1)

        new_positions = [
            Vector([-31.5, -17, 0]),
            Vector([-6.1, -12, 0]),
            Vector([-14, 13.5, 0]),
            Vector([31, 15.5, 0]),
        ]

        for i in range(0, 4):
            carriers[i].move_to(target_location=new_positions[i], begin_time=t0, transition_time=1)
            if i > 1:
                print("55 should appear", t0)
                ibpy.change_default_integer(dialers[i], from_value=1, to_value=2, begin_time=t0 + 1, transition_time=0)

        t0 += 1.5

        ibpy.camera_move(shift=[0, 0, 80], begin_time=t0, transition_time=1)

        new_positions = [
            Vector([-68.2, -43.1, 0]),
            Vector([-42.8, -38.1, 0]),
            Vector([-50.75, -12.6, 0]),
            Vector([46.65, 22.7, 0]),
        ]

        for i in range(0, 4):
            carriers[i].move_to(target_location=new_positions[i], begin_time=t0, transition_time=1)
            if i > 2:
                print("377 should appear", t0)
                ibpy.change_default_integer(dialers[i], from_value=2, to_value=3, begin_time=t0 + 1, transition_time=0)

        t0 += 1.5

        self.t0 = t0

    def intro_fib_overlay(self):
        t0 = 0

        _setup_render()
        # the camera always points at the tracked empty; its world position is
        # driven by smooth Bezier fly-paths instead of straight waypoint hops.

        ibpy.set_camera_location(location=[0, -10, 0])
        ibpy.set_camera_lens(lens=30, clip_end=100000)

        numbers = [1, 8, 55, 377]
        appear_times = [0, 1.5, 3, 4.5]

        for i, (number, time) in enumerate(zip(numbers, appear_times)):
            n_bob = SimpleTexBObject(f"{number}", location=[-5.5 + 0.75 * (4 * i + 2), 0, -3],
                                     color="example", text_size="large")
            n_bob.write(begin_time=time, transition_time=0.1)

        t0 = 0.5 + time

        fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        for i, fib in enumerate(fibs):
            if (i + 2) % 4 != 0:
                fib_bob = SimpleTexBObject(f"{fib}", location=[-5.5 + 0.75 * i, 0, -3])
                t0 = 0.1 + fib_bob.write(begin_time=t0, transition_time=0.1)

        self.t0 = t0

    def intro_algebra_overlay(self):
        t0 = 0
        _setup_render()
        ibpy.set_camera_location(location=[0.91, -5, -2])
        ibpy.camera_zoom(lens=23, begin_time=0, transition_time=0)

        deriv = BDerivation(r"(x+y)^2 = 9\cdot (1+x\cdot y)",
                            name="AlgebraDerivation")
        t0 = 0.5 + deriv.write(begin_time=t0, transition_time=0.5)

        # y -> 55, in place: each y morphs into one 5, its twin is written in
        # (the @n pins are required, y occurs twice)
        t0 = 0.5 + deriv.step(r"(x+55)^2 = 9\cdot (1+x\cdot 55)", mode='replace',
                              map={"y@0": "55@0", "y@1": "55@1"},
                              begin_time=t0, transition_time=0.5)

        # expand both sides onto a new line
        t0 = 0.5 + deriv.step(r"x^2+2\cdot 55\cdot x+55^2 = 9+9\cdot 55\cdot x",
                              map={"(x+55)^2": r"x^2+2\cdot 55\cdot x+55^2",
                                   r"9\cdot (1+x\cdot 55)": r"9+9\cdot 55\cdot x"},
                              begin_time=t0, transition_time=1)

        # evaluate the products in place: 2*55 -> 110, 55^2 -> 3025, 9*55 -> 495
        t0 = 0.5 + deriv.step(r"x^2+110\cdot x+3025 = 9+495\cdot x", mode='replace',
                              map={r"2\cdot 55": "110",
                                   "55^2": "3025",
                                   r"9\cdot 55": "495"},
                              begin_time=t0, transition_time=1)

        # collect everything on the left by moving the right-hand terms across
        # the '=': +9 -> -9 merges into 3025 -> 3016, and +495x -> -495x merges
        # into 110x -> -385x (each mover lifts, crosses, flips sign and sinks in)
        t0 = 0.5 + deriv.step(r"x^2-385\cdot x+3016 = 0", mode='add_subtract',
                              map={r"110\cdot x": r"-385\cdot x",
                                   r"495\cdot x": r"-385\cdot x",
                                   "3025": "3016",
                                   "9@0": "3016"},
                              lift=[0.3, 0.7], begin_time=t0, transition_time=3)

        t0 = 0.5 + deriv.step(r"x_{1,2}=\tfrac{385}{2}\pm\sqrt{\left(\tfrac{385^2}{4}-3016\right)}",
                              mode='new_line', begin_time=t0, transition_time=1)

        t0 = 0.5 + deriv.step(r"x_{1,2}=\left\{\parbox{0.1\linewidth}{$8$\\\\$377$}\right.",
                              mode='new_line', begin_time=t0, transition_time=1)

        # move copies of 8, 55, 377
        shifts = [
            Vector([-1.56, -1.33, 0]),
            Vector([1.67, -3.97, 0]),
            Vector([2.34, -0.57, 0]),
        ]

        # send copies of the Fibonacci numbers out to their tiles: the 8 and
        # the 377 from the last line, the first 55 from the second line; each
        # copy turns 'example' in flight and has faded out on arrival
        flights = [(deriv.lines[-1], "8", shifts[0]),
                   (deriv.lines[1], "55@0", shifts[1]),
                   (deriv.lines[-1], "377", shifts[2])]
        flight_time = 1
        for k, (line, spec, shift) in enumerate(flights):
            start = t0 + 0.5 * k
            scale = line.ref_obj.scale
            local_shift = Vector([shift[i] / scale[i] for i in range(3)])
            for idx in line.find_letters(spec):
                target = line.letters[idx].ref_obj.location + local_shift
                line.move_copy_of_letter_to(idx, target, begin_time=start,
                                            transition_time=flight_time)
                copy = line.copies_of_letters[-1]
                copy.change_color(new_color='example', begin_time=start,
                                  transition_time=0.5 * flight_time)
                copy.disappear(begin_time=start + 0.7 * flight_time,
                               transition_time=0.3 * flight_time)
        t0 = t0 + 0.5 * (len(flights) - 1) + flight_time

        self.t0 = t0

    def recursion_to_quadratic(self):
        """Derive the quadratic invariant of the Fibonacci-like sequence
        a_{n+1}=7 a_n - a_{n-1} directly from the recursion.

        The chain of transformations is animated with :class:`BDerivation`
        (see :meth:`intro_algebra_overlay` for the same idioms):

          a_{n+1}=7a_n-a_{n-1}
          a_{n+1}+a_{n-1}=7a_n                      (symmetric form)
          (a_{n+1}+a_{n-1})(a_{n+1}-a_{n-1})=7a_n(a_{n+1}-a_{n-1})
                                                    (multiply by the blue factor)
          a_{n+1}^2-a_{n-1}^2=7a_n a_{n+1}-7a_n a_{n-1}
                                                    (difference of squares)
          a_{n+1}^2-7a_n a_{n+1}=a_{n-1}^2-7a_n a_{n-1}
                                                    (collect n+1 left, n-1 right)
          a_{n+1}^2-7a_n a_{n+1}+a_n^2 = a_{n-1}^2-7a_n a_{n-1}+a_n^2
                                                    (add a_n^2, green)

        Two ``\\underbrace{...}_{Q_n}`` labels then slide up from below onto the
        two sides; each side collapses into its label giving Q_n=Q_{n-1}, i.e.
        the quantity is constant.  Its value 9 (from the seed a_0=0, a_1=3)
        turns the invariant into the target equation
        (a_n+a_{n+1})^2 = 9(1+a_n a_{n+1}).

        The camera pans down smoothly so the active line stays framed; the row
        positions, brace offsets and pan targets are the tunable constants
        collected at the top of the method.
        """
        t0 = 0
        _setup_render()

        # ---- tunable layout / framing ------------------------------------
        top_z = 1.6  # z of the first derivation row
        row_dz = -0.95  # vertical spacing between rows
        txt = 0.7  # font scale of the derivation (wide lines must fit)
        brace_drop = 0.5  # how far below a row's baseline the brace sits
        brace_rise = 0.8  # distance a brace travels as it slides up in

        # the camera always looks along +y; we pan it in z to follow the
        # write head.  cam[2] is kept in sync by pan().
        cam = [0.35, -9.0, 1.6]
        ibpy.set_camera_location(location=list(cam))
        ibpy.camera_zoom(lens=42, begin_time=0, transition_time=0)

        # objects that ride along with the camera: a sticky title bar keeps its
        # place on screen while every pan shifts it by the same dz as the camera
        sticky = []

        def pan(z_target, begin_time, transition_time=1.0):
            dz = z_target - cam[2]
            if abs(dz) > 1e-4:
                ibpy.camera_move(shift=[0, 0, dz], begin_time=begin_time,
                                 transition_time=transition_time)
                for obj in sticky:
                    obj.move(direction=Vector((0, 0, dz)),
                             begin_time=begin_time, transition_time=transition_time)
                cam[2] = z_target
            return begin_time + transition_time

        # ---- heading on a sticky display fixed at the top of the frame: it
        #      holds its place while the derivation scrolls up and vanishes
        #      behind the (opaque) display panel ------------------------------
        head_display = Display(scales=[4.5, 0.45], location=[0.35, -0.6, 3.45],
                               number_of_lines=1, flat=True, shadow=False,
                               name="RQ_head_display")
        head_display.appear(begin_time=t0, transition_time=0.5)
        head = SimpleTexBObject(r"\text{From Recursion to Invariant}",
                                color="example", text_size=0.8, aligned="center",
                                location=[0.35, -0.72, 3.3], name="RQ_head")
        t0 = 0.4 + head.write(begin_time=t0 + 0.3, transition_time=0.8)
        sticky.extend([head_display, head])

        # ---- the derivation chain ----------------------------------------
        deriv = BDerivation(r"a_{n+1}=7a_n-a_{n-1}", name="RecToQuad",
                            color="text", text_size=txt,
                            location=Vector((-0.6, 0, top_z)),
                            line_spacing=Vector((0, 0, row_dz)))
        t0 = 0.5 + deriv.write(begin_time=t0, transition_time=0.8)

        # symmetric form: -a_{n-1} swaps across the '=' to the left, its '-'
        # morphing into '+', in place; a_{n+1} is pinned so it stays whole (its
        # subscript '+' must not be mistaken for the operator '+' and torn off)
        t0 = 0.5 + deriv.step(r"a_{n+1}+a_{n-1}=7a_n", mode="swap",
                              map={r"a_{n+1}": r"a_{n+1}", r"a_{n-1}": r"a_{n-1}"},
                              begin_time=t0, transition_time=1.2, lift=0.35)

        # multiply both sides by (a_{n+1}-a_{n-1}) -- the fresh factor is written
        # in on both sides, the old terms fly into the left factor / the 7a_n
        t0 = 0.5 + deriv.step(r"(a_{n+1}+a_{n-1})(a_{n+1}-a_{n-1})=7a_n(a_{n+1}-a_{n-1})",
                              mode="new_line_copy",
                              map={r"a_{n+1}": r"a_{n+1}@0", r"a_{n-1}": r"a_{n-1}@0",
                                   "7": "7", r"a_n": r"a_n"},
                              begin_time=t0, transition_time=1.4)
        # paint the two copies of the key factor blue
        factor_idx = deriv.current.find_letters(r"(a_{n+1}-a_{n-1})@all")
        deriv.current.change_color_of_letters(factor_idx, "drawing",
                                              begin_time=t0 - 1, transition_time=0.4)
        t0 += 0.6
        t0 = pan(0.6, begin_time=t0, transition_time=1.0)

        # difference of squares (left) and distribute (right)
        t0 = 0.5 + deriv.step(r"a_{n+1}^2-a_{n-1}^2=7a_na_{n+1}-7a_na_{n-1}",
                              mode="replace",
                              map={r"(a_{n+1}+a_{n-1})(a_{n+1}-a_{n-1})":
                                       r"a_{n+1}^2-a_{n-1}^2",
                                   r"7a_n(a_{n+1}-a_{n-1})":
                                       r"7a_na_{n+1}-7a_na_{n-1}"},
                              begin_time=t0, transition_time=1.6)

        # collect: a_{n-1}^2 and 7a_na_{n+1} swap sides across the '=' at two
        # heights (they cross in opposite directions); the '-' sticks to the
        # leaving a_{n-1}^2 and flips to '+' as it crosses, while the arriving
        # 7a_na_{n+1} grows a fresh '-' over the '='
        t0 = 0.5 + deriv.step(r"a_{n+1}^2-7a_na_{n+1}=a_{n-1}^2-7a_na_{n-1}",
                              mode="swap",
                              map={r"a_{n+1}^2": r"a_{n+1}^2",
                                   r"7a_na_{n+1}": r"7a_na_{n+1}",
                                   r"a_{n-1}^2": r"a_{n-1}^2",
                                   r"7a_na_{n-1}": r"7a_na_{n-1}"},
                              lift=[0.3, 0.6],
                              begin_time=t0, transition_time=1.6)
        t0 = pan(-0.4, begin_time=t0, transition_time=1.0)

        # add a_n^2 to both sides; the two fresh terms flash green
        t0 = 0.5 + deriv.step(r"a_{n+1}^2-7a_na_{n+1}+a_n^2=a_{n-1}^2-7a_na_{n-1}+a_n^2",
                              mode="new_line_copy",
                              map={r"a_{n+1}^2-7a_na_{n+1}": r"a_{n+1}^2-7a_na_{n+1}",
                                   r"a_{n-1}^2-7a_na_{n-1}": r"a_{n-1}^2-7a_na_{n-1}"},
                              begin_time=t0, transition_time=1.4)
        line5 = deriv.current
        added_idx = line5.find_letters(r"+a_n^2@all")
        line5.change_color_of_letters(added_idx, "joker",
                                      begin_time=t0 - 1.5, transition_time=0.4)
        t0 += 0.6

        # ---- underbraces slide up from below onto the two sides ----------
        def world_bbox(line, spec):
            loc = line.ref_obj.location
            xs = [loc[0] + line.letters[i].ref_obj.location[0]
                  for i in line.find_letters(spec)]
            return min(xs), max(xs)

        lhs_min, _ = world_bbox(line5, r"a_{n+1}^2-7a_na_{n+1}+a_n^2")
        rhs_min, _ = world_bbox(line5, r"a_{n-1}^2-7a_na_{n-1}+a_n^2")
        base_z = line5.ref_obj.location[2]
        brace_z = base_z - brace_drop

        brace_specs = [
            (r"\underbrace{\phantom{a_{n+1}^2-7a_na_{n+1}+a_n^2}}_{\;Q_n}",
             lhs_min, "RQ_Qn"),
            (r"\underbrace{\phantom{a_{n-1}^2-7a_na_{n-1}+a_n^2}}_{\;Q_{n-1}}",
             rhs_min, "RQ_Qm"),
        ]
        braces = []
        for k, (tex, x0, nm) in enumerate(brace_specs):
            b = SimpleTexBObject(tex, color="example", text_size=txt,
                                 location=[x0, 0, brace_z - brace_rise],
                                 name=nm)
            start = t0 + 0.35 * k
            b.write(begin_time=start, transition_time=0.7)
            b.move(direction=Vector((0, 0, brace_rise)),
                   begin_time=start, transition_time=0.8)
            braces.append(b)
        t0 += 0.35 * (len(braces) - 1) + 0.8 + 0.5

        # the brace labels themselves rise into the new line -> Q_n = Q_{n-1}
        # (no copies fly out of the equation; both sides are pinned away)
        t0 = 0.5 + deriv.step(r"Q_n=Q_{n-1}", mode="new_line",
                              map={r"a_{n+1}^2-7a_na_{n+1}+a_n^2": None,
                                   r"a_{n-1}^2-7a_na_{n-1}+a_n^2": None},
                              sources=[(braces[0], "Q_n", "Q_n"),
                                       (braces[1], "Q_{n-1}", "Q_{n-1}")],
                              begin_time=t0, transition_time=1.4)
        # the arrived labels stay green and simply ARE the new line; the
        # pristine letters (recoloured to match) take over invisibly when
        # the next step starts
        q_idx = (deriv.current.find_letters("Q_n")
                 + deriv.current.find_letters("Q_{n-1}"))
        deriv.current.change_color_of_letters(q_idx, "example",
                                              begin_time=t0 - 0.5,
                                              transition_time=0)
        for b in braces:
            b.disappear(begin_time=t0 - 1.2, transition_time=0.6)
        t0 = pan(-1.4, begin_time=t0, transition_time=1.0)

        # the invariant is constant: value fixed by the seed a_0=0, a_1=3
        t0 = 0.5 + deriv.step(r"Q_n=Q_{n-1}=\dots=Q_0=9", mode="replace",
                              map={"Q_n": "Q_n", "Q_{n-1}": "Q_{n-1}"},
                              begin_time=t0, transition_time=1.2)

        # spell the invariant out ...
        t0 = 0.5 + deriv.step(r"a_n^2+a_{n+1}^2-7a_na_{n+1}=9", mode="new_line",
                              map={"Q_n": r"a_n^2+a_{n+1}^2-7a_na_{n+1}", "9": "9"},
                              begin_time=t0, transition_time=1.4)

        # split -7a_na_{n+1} into +2a_na_{n+1}-9a_na_{n+1}: the +2 term completes
        # the square a_n^2+2a_na_{n+1}+a_{n+1}^2=(a_n+a_{n+1})^2, making the next
        # step transparent, while -9a_na_{n+1} is the remainder that crosses the
        # '=' to build 9(1+a_na_{n+1})
        # pin each square and the moving product whole (a_{n+1}^2 must not be
        # torn apart -- its subscript '+' must not be grabbed by the fresh '+2'
        # operator); the -7 term morphs into the -9 term (7->9), +2a_na_{n+1}
        # is written in fresh, '9@1' keeps the right-hand 9 distinct from it
        t0 = 0.5 + deriv.step(
            r"a_n^2+a_{n+1}^2+2a_na_{n+1}-9a_na_{n+1}=9", mode="replace",
            map={r"a_n^2": r"a_n^2",
                 r"a_{n+1}^2": r"a_{n+1}^2",
                 r"7a_na_{n+1}": r"9a_na_{n+1}",
                 "9": "9@1"},
            begin_time=t0, transition_time=1.4)
        t0 = pan(-2.6, begin_time=t0, transition_time=1.0)

        # ... and factor it into the target equation (the completed square folds
        # into (a_n+a_{n+1})^2; '9@1' pins the right-hand 9, since the line now
        # carries a second 9 in the -9a_na_{n+1} term)
        t0 = 0.5 + deriv.step(r"(a_n+a_{n+1})^2=9\,(1+a_na_{n+1})",
                              mode="new_line",
                              map={r"a_n^2+a_{n+1}^2+2a_na_{n+1}":
                                       r"(a_n+a_{n+1})^2",
                                   "9@1": "9"},
                              begin_time=t0, transition_time=1.6)
        # final flourish: flash the two 9's that make the equation tick
        deriv.current.change_color_of_letters(deriv.current.find_letters("9"),
                                              "important",
                                              begin_time=t0, transition_time=0.5)
        t0 += 1.0

        self.t0 = t0

    def barnsley_fern(self):
        """The Barnsley fern as an iterated function system, grown live by a
        geometry-node chaos game (:class:`BarnsleyFernModifier`).

        A carrier plane on the right hosts the modifier; two exposed node
        values are animated -- ``PointCount`` climbs so the fern fills in from
        a sparse scatter, and ``PointSize`` is dialed to show the dots are
        tunable.  The four affine maps that define the IFS stand as matrices
        (with their probabilities) to the left of the plane.
        """
        t0 = 0
        _setup_render()
        ibpy.set_camera_location(location=[-0.6, -12.5, 0.6])
        ibpy.camera_zoom(lens=36, begin_time=0, transition_time=0)

        # ---- carrier plane hosting the fern modifier ---------------------
        fern_plane = Plane(name="FernCarrier", location=[2.5, 0, -0.1])
        fern_plane.appear(begin_time=0, transition_time=0)
        mod = BarnsleyFernModifier(point_count=40000, point_size=0.02,
                                   iterations=50, scale=0.55)
        fern_plane.add_mesh_modifier(type='NODES', node_modifier=mod)
        t0 = fern_plane.grow(begin_time=0, transition_time=0)
        fern_plane.move(direction=[-1.75, 0, 0], begin_time=t0, transition_time=0)
        count_node = ibpy.get_geometry_node_from_modifier(mod, "PointCount")
        size_node = ibpy.get_geometry_node_from_modifier(mod, "PointSize")

        # ---- title -------------------------------------------------------
        title = SimpleTexBObject(r"\text{The Barnsley Fern}", color="example",
                                 text_size="large", aligned="center",
                                 location=[-3, 0, 3.8], name="RF_title")
        t0 = 0.4 + title.write(begin_time=t0, transition_time=0.8)

        subtitle = SimpleTexBObject(r"\text{four maps, iterated forever}",
                                    color="text", text_size="small",
                                    aligned="center", location=[-3, 0, 3.1],
                                    name="RF_subtitle")
        t0 = 0.3 + subtitle.write(begin_time=t0, transition_time=0.6)

        # ---- the four defining affine maps, beside the plane -------------
        # each map is drawn in the colour of the fern region it generates, so
        # the matrix and the points it paints share a colour (see the
        # per-point map_index colouring in BarnsleyFernModifier)
        maps = [
            r"\omega_1=\begin{pmatrix}0&0\\0&0.16\end{pmatrix}\mathbf{x}",
            r"\omega_2=\begin{pmatrix}0.85&0.04\\-0.04&0.85\end{pmatrix}\mathbf{x}"
            r"+\begin{pmatrix}0\\1.6\end{pmatrix}",
            r"\omega_3=\begin{pmatrix}0.2&-0.26\\0.23&0.22\end{pmatrix}\mathbf{x}"
            r"+\begin{pmatrix}0\\1.6\end{pmatrix}",
            r"\omega_4=\begin{pmatrix}-0.15&0.28\\0.26&0.24\end{pmatrix}\mathbf{x}"
            r"+\begin{pmatrix}0\\0.44\end{pmatrix}",
        ]
        probs = [r"p=0.01", r"p=0.85", r"p=0.07", r"p=0.07"]
        rows = [2.4, 0.9, -0.7, -2.3]
        # matches BarnsleyFernModifier.colors (omega_1..omega_4)
        map_colors = ["important", "joker", "drawing", "example"]
        for tex, p, z, col in zip(maps, probs, rows, map_colors):
            m = SimpleTexBObject(tex, color=col, text_size=0.7,
                                 aligned="left", location=[-6.6, 0, 0.9 * z],
                                 name="RF_map")
            p_obj = SimpleTexBObject(p, color=col, text_size=0.7,
                                     aligned="left", location=[-5, 0, 0.9 * z - 0.6],
                                     name="RF_prob")
            m.write(begin_time=t0, transition_time=0.7)
            p_obj.write(begin_time=t0 + 0.3, transition_time=0.4)
            t0 += 0.55
        t0 += 0.4

        # --- grow the number of iterations

        iter_count_node = get_geometry_node_from_modifier(mod, label="IterationCount")
        t0 = 0.5 + ibpy.change_default_integer(iter_count_node, from_value=0, to_value=50,
                                               begin_time=t0, transition_time=2)

        # ---- grow the fern: sparse -> dense, dots shrinking as it fills --
        ibpy.change_default_integer(count_node, from_value=4000, to_value=40000,
                                    begin_time=t0, transition_time=7)
        ibpy.change_default_value(size_node, from_value=0.01, to_value=0.005,
                                  begin_time=t0, transition_time=7)
        t0 += 8.0

        self.t0 = t0

    def sierpinski_triangle(self):
        """The Sierpinski triangle as an iterated function system, built
        end-to-end inside :class:`SierpinskiTriangleModifier`.

        The three corners are geometry-node *input vectors* and everything is
        derived from them, so the whole picture is one consistent construction:

        1. the corners appear as three fat coloured markers (``MarkerSize``);
        2. a *single* point plays the chaos game by hand -- growing
           ``TrajectorySteps`` walks it hop by hop, always half-way toward a
           randomly chosen corner, leaving a dot and a segment in that corner's
           colour;
        3. the same game then runs on 40000 points at once (``PointCount`` /
           ``PointSize``) and the gasket floods in over the hand-played walk;
        4. finally a corner is *moved*: because the maps, the cloud, the
           trajectory and the markers all read the same input vector, the whole
           fractal shears to follow it.
        """
        t0 = 0
        _setup_render()
        ibpy.set_camera_location(location=[-0.6, -12.5, 0.6])
        ibpy.camera_zoom(lens=36, begin_time=0, transition_time=0)

        # ---- carrier plane hosting the whole construction ----------------
        # every visible size starts at 0 / 0 steps, so the scene opens empty
        # and each beat is dialed in by animating one exposed node
        carrier = Vector([2.5, 0, -0.1])
        tri_plane = Plane(name="SierpinskiCarrier", location=carrier)
        tri_plane.move(direction=[2.5, 0, -0.2], begin_time=t0, transition_time=0)
        tri_plane.appear(begin_time=0, transition_time=0)
        n_steps = 20
        mod = SierpinskiTriangleModifier(point_count=40000, point_size=0.0,
                                         iterations=50, marker_size=0.0,
                                         trajectory_length=n_steps,
                                         trajectory_steps=0, dot_size=0.0)
        tri_plane.add_mesh_modifier(type='NODES', node_modifier=mod)
        t0 = tri_plane.grow(begin_time=0, transition_time=0)

        count_node = ibpy.get_geometry_node_from_modifier(mod, "PointCount")
        size_node = ibpy.get_geometry_node_from_modifier(mod, "PointSize")
        steps_node = ibpy.get_geometry_node_from_modifier(mod, "TrajectorySteps")
        dot_node = ibpy.get_geometry_node_from_modifier(mod, "TrajectoryDotSize")
        marker_node = ibpy.get_geometry_node_from_modifier(mod, "MarkerSize")
        corner_nodes = [ibpy.get_geometry_node_from_modifier(mod, "Corner%d" % i)
                        for i in (1, 2, 3)]

        # corner labels live in world space; the modifier's corners are local
        # to the carrier, so world = carrier + local (one source of truth)
        corner_cols = ["important", "joker", "drawing"]
        label_off = [Vector([-0.35, 0, -0.35]),  # C_1 bottom-left
                     Vector([0.35, 0, -0.35]),  # C_2 bottom-right
                     Vector([0.0, 0, 0.25])]  # C_3 top

        # ---- title -------------------------------------------------------
        title = SimpleTexBObject(r"\text{The Sierpinski Triangle}",
                                 color="example", text_size="large",
                                 aligned="center", location=[2.5, 0, 3.75],
                                 name="ST_title")
        t0 = 0.4 + title.write(begin_time=t0, transition_time=0.8)

        subtitle = SimpleTexBObject(r"\text{halfway to a random corner}",
                                    color="text", text_size="small",
                                    aligned="left", location=[-1.0, 0, 3.1],
                                    name="ST_subtitle")
        t0 = 0.3 + subtitle.write(begin_time=t0, transition_time=0.6)

        # ---- the three midpoint maps, beside the plane -------------------
        # each map is drawn in the colour of the corner it generates, matching
        # SierpinskiTriangleModifier.colors (w_1..w_3)
        maps = [
            r"w_1(\mathbf{x})=\tfrac12\left(\mathbf{x}+C_1\right)",
            r"w_2(\mathbf{x})=\tfrac12\left(\mathbf{x}+C_2\right)",
            r"w_3(\mathbf{x})=\tfrac12\left(\mathbf{x}+C_3\right)",
        ]
        probs = [r"p=\tfrac13", r"p=\tfrac13", r"p=\tfrac13"]
        rows = [2.5, 1.6, 0.6]
        for tex, p, z, col in zip(maps, probs, rows, corner_cols):
            m = SimpleTexBObject(tex, color=col,
                                 aligned="left", location=[-1, 0, z],
                                 name="ST_map")
            p_obj = SimpleTexBObject(p, color=col,
                                     aligned="left", location=[0, 0, z - 0.5],
                                     name="ST_prob")
            m.write(begin_time=t0, transition_time=0.7)
            p_obj.write(begin_time=t0 + 0.3, transition_time=0.4)
            t0 += 0.55
        t0 += 0.3

        # ---- the three input corners C_1, C_2, C_3 -----------------------
        ibpy.change_default_value(marker_node, from_value=0.0, to_value=0.15,
                                  begin_time=t0, transition_time=0.5)
        labels = []
        for i, (cn, col, off) in enumerate(zip(corner_nodes, corner_cols,
                                               label_off)):
            lbl = SimpleTexBObject(r"C_%d" % (i + 1), color=col,
                                   aligned="center",
                                   location=carrier + Vector(cn.vector) + off,
                                   name="ST_cornerlabel%d" % (i + 1))
            lbl.write(begin_time=t0 + 0.2, transition_time=0.4)
            labels.append(lbl)
        t0 += 1.0

        # ---- single trajectory: play the chaos game by hand --------------
        # growing TrajectorySteps walks the point hop by hop; the dots and the
        # connecting segments are generated by the same node graph as the cloud
        ibpy.change_default_value(dot_node, from_value=0.0, to_value=0.07,
                                  begin_time=t0, transition_time=0.4)
        ibpy.change_default_integer(steps_node, from_value=0, to_value=n_steps,
                                    begin_time=t0, transition_time=6.0)
        t0 += 6.5

        # ---- now run the game on 40000 points at once --------------------
        go = SimpleTexBObject(r"\text{now 40000 points at once}", color="text",
                              text_size="small", aligned="left",
                              location=[-1.0, 0, 3.1], name="ST_go")
        subtitle.disappear(begin_time=t0 - 0.4, transition_time=0.4)
        go.write(begin_time=t0, transition_time=0.5)

        # the hand-played dots stay put and simply become part of the cloud
        radius_node = ibpy.get_geometry_node_from_modifier(mod, "PathRadius")
        ibpy.change_default_integer(count_node, from_value=800, to_value=40000,
                                    begin_time=t0, transition_time=6)
        ibpy.change_default_value(size_node, from_value=0.0, to_value=0.005,
                                  begin_time=t0, transition_time=6)
        ibpy.change_default_value(radius_node, from_value=0.005, to_value=0.0001, begin_time=t0, transition_time=6)
        t0 += 6.8

        # ---- move a corner: the whole construction follows ----------------
        # C_3 is only an input vector, and the maps, the cloud, the trajectory
        # and the markers all read it -- so the gasket shears as it moves
        go2 = SimpleTexBObject(r"\text{move a corner, the fractal follows}",
                               color="text", text_size="small",
                               aligned="left", location=[-1.0, 0, 3.1],
                               name="ST_go2")
        go.disappear(begin_time=t0 - 0.4, transition_time=0.4)
        go2.write(begin_time=t0, transition_time=0.5)

        # both ends of every leg are keyframed explicitly: with no key at
        # begin_time the corner would hold its final value for the whole scene
        c3_home = Vector(corner_nodes[2].vector)
        c3_away = Vector([1.6, 0, 1.5])
        for source, target in ((c3_home, c3_away), (c3_away, c3_home)):
            ibpy.change_default_vector(corner_nodes[2], from_value=source,
                                       to_value=target, begin_time=t0,
                                       transition_time=2.5)
            labels[2].move_to(carrier + target + label_off[2], begin_time=t0,
                              transition_time=2.5)
            t0 += 3.0

        self.t0 = t0

    def apollonian_gasket(self):
        """The Apollonian gasket as the limit set of four Moebius maps,
        generated live by the DFS tree inside :class:`ApollonianGasketModifier`.

        The node graph *is* the depth-first search of ``video_apollonian``:
        every point is a word in the generators; per repeat-zone iteration
        each unfinished branch spawns its three children and a branch freezes
        as soon as the image of its Schottky disc has radius below ``Epsilon``.
        The scene tells that story through the one dial that matters:

        1. the four generators appear as complex 2x2 matrices, coloured like
           the branches of the tree they root (``first_letter`` colouring);
        2. the gasket pops in coarse (``epsilon = 0.4`` -- a few hundred
           points, each standing in for a whole subtree);
        3. epsilon is dialed down in two legs; the tree grows deeper wherever
           the discs are still large and the curve refines live, with the
           dots shrinking as ~200000 points fill in the fractal.
        """
        t0 = 0
        _setup_render()
        ibpy.set_camera_location(location=[-0.6, -12.5, 0.6])
        ibpy.camera_zoom(lens=36, begin_time=0, transition_time=0)

        # ---- carrier plane hosting the modifier --------------------------
        # the gasket occupies a 5x5 box in the carrier's x-z plane; epsilon
        # starts coarse and the radius at 0 so the scene opens empty
        gasket_plane = Plane(name="GasketCarrier", location=[2.5, 0, 0.1])
        gasket_plane.appear(begin_time=0, transition_time=0)
        mod = ApollonianGasketModifier(epsilon=0.4, max_level=45,
                                       point_radius=0.0, color_by="first_letter")
        gasket_plane.add_mesh_modifier(type='NODES', node_modifier=mod)
        t0 = gasket_plane.grow(begin_time=0, transition_time=0)
        gasket_plane.move(direction=[-2, 0, 0], begin_time=t0, transition_time=0)

        eps_node = ibpy.get_geometry_node_from_modifier(mod, "Epsilon")
        radius_node = ibpy.get_geometry_node_from_modifier(mod, "PointRadius")

        # ---- title -------------------------------------------------------
        title = SimpleTexBObject(r"\text{The Apollonian Gasket}",
                                 color="example", text_size="large",
                                 aligned="center", location=[-3, 0, 3.8],
                                 name="AG_title")
        t0 = 0.4 + title.write(begin_time=t0, transition_time=0.8)

        subtitle = SimpleTexBObject(r"\text{the limit set of four M\"obius maps}",
                                    color="text", text_size="small",
                                    aligned="center", location=[-3, 0, 3.1],
                                    name="AG_subtitle")
        t0 = 0.3 + subtitle.write(begin_time=t0, transition_time=0.6)

        # ---- the four generators, coloured like their subtrees -----------
        # matches ApollonianGasketModifier.colors (first_letter a, b, A, B);
        # displayed in the original coordinates of video_apollonian -- the
        # modifier only conjugates them once to frame the picture
        maps = [
            r"a=\begin{pmatrix}1&0\\-2i&1\end{pmatrix}",
            r"b=\begin{pmatrix}1-i&1\\1&1+i\end{pmatrix}",
            r"A=\begin{pmatrix}1&0\\2i&1\end{pmatrix}",
            r"B=\begin{pmatrix}1+i&-1\\-1&1-i\end{pmatrix}",
        ]
        rows = [2.2, 0.9, -0.4, -1.7]
        map_colors = ["text", "joker", "important", "custom1"]
        for tex, z, col in zip(maps, rows, map_colors):
            m = SimpleTexBObject(tex, color=col, text_size=0.7,
                                 aligned="left", location=[-6.4, 0, z],
                                 name="AG_gen")
            m.write(begin_time=t0, transition_time=0.6)
            t0 += 0.45
        t0 += 0.4

        # ---- coarse gasket: every dot is a whole subtree -----------------
        # with epsilon = 0.4 the DFS stops after a few levels; the ~700
        # points are the leaves of that shallow tree.
        # the readout is a single DigitalRange: instead of writing a whole
        # new LaTeX expression at each step (BarnsleyFern/Sierpinski style),
        # its digits morph via shape keys -- see objects/digital_number.py,
        # used the same way for the temperature/magnetization dials in
        # video_ising/scene_ising.py. The value list is log-spaced (epsilon
        # shrinks multiplicatively) and sorted descending so the shape-key
        # chain ticks the number down monotonically as epsilon is dialed.
        leg1 = np.geomspace(0.4, 0.05, 9)
        leg2 = np.geomspace(0.05, 0.008, 9)[1:]
        eps_values = sorted(set(np.round(np.concatenate([leg1, leg2]), 3)),
                            reverse=True)
        eps_range = DigitalRange(eps_values, digits=3, signed=False,
                                 aligned="left", color="text", rotation_euler=[pi / 2, 0, 0],
                                 prefix=r"\varepsilon=", sorting='DESC',
                                 location=[-6.4, 0, -2.6], name="AG_epsilon")
        eps_range.write(begin_time=t0, transition_time=0.4)
        t0 = 0.5 + ibpy.change_default_value(radius_node, from_value=0.0,
                                             to_value=0.06, begin_time=t0,
                                             transition_time=0.7)

        # ---- refine: dial epsilon down, the tree grows where needed ------
        eps_range.show(from_value=0.4, to_value=0.05, begin_time=t0,
                       transition_time=6)
        ibpy.change_default_value(eps_node, from_value=0.4, to_value=0.05,
                                  begin_time=t0, transition_time=6)
        ibpy.change_default_value(radius_node, from_value=0.06, to_value=0.03,
                                  begin_time=t0, transition_time=6)
        t0 += 6.5

        eps_range.show(from_value=0.05, to_value=0.008, begin_time=t0,
                       transition_time=8)
        ibpy.change_default_value(eps_node, from_value=0.05, to_value=0.008,
                                  begin_time=t0, transition_time=8)
        ibpy.change_default_value(radius_node, from_value=0.03, to_value=0.012,
                                  begin_time=t0, transition_time=8)
        t0 += 8.5

        self.t0 = t0

    def outro_fibonacci_matrix(self):
        """Outro: the Q-matrix secret behind 1, 8, 55, 377.

        Storyboard fitted to the outro recap paragraph of script_claude.md
        ("And those numbers from the very beginning? 1, 8, 55, 377. You now
        know all their secrets: ... they stride along Fibonacci four steps
        at a time because each generation scales by phi^4; and the
        intimidating quadratic was just 'seven times this, minus the last
        one' -- wearing a disguise."):

        A  the numbers: the Fibonacci row with F_2, F_6, F_10, F_14
           = 1, 8, 55, 377 lighting up -- the four-step stride.
        B  the machine: Q = ((1,1),(1,0)) generates Fibonacci, so a stride
           of four is one power of Q^4 = ((5,3),(3,2)).
        C  the scale factor: det(Q^4-lambda I)=0 is ground out to
           lambda^2-7 lambda+1=0, whose roots are phi^{+-4} -- each
           generation scales by phi^4, and 7 = phi^4 + phi^{-4}.
        D  the recursion: Cayley-Hamilton turns the characteristic equation
           into Q^8 = 7Q^4 - I; multiplying by Q^{4(n-1)+2} and reading the
           off-diagonal entries gives a_{n+1} = 7a_n - a_{n-1} for
           a_n = F_{4n+2}.
        E  the check on the video's own numbers, 55 = 7*8-1 and
           377 = 7*55-8: "seven times this, minus the last one" -- the
           intimidating quadratic, unmasked.

        Layout: a title line, the Fibonacci row as a persistent golden
        thread underneath, and three equation slots below that every beat
        reuses.  All derivation lines share one '=' spine: BDerivation
        aligns consecutive lines on '=' (align_char), the second chain is
        tied to the first with a manual align, and in-place morph steps
        (replace/swap) keep at most three equations on stage.
        """
        t0 = 0
        _setup_render()
        set_alpha_composition()
        ibpy.set_camera_location(location=[0.3, -9.5, 0.5])
        ibpy.camera_zoom(lens=40, begin_time=0, transition_time=0)
        # visible frame at the y=0 text plane: x in [-4.0, 4.6], z in [-1.9, 2.9]

        # colour roles echoing recursion_to_quadratic
        KEY, FIB, HOT, GRN = "drawing", "example", "important", "joker"

        # ---- layout constants ---------------------------------------------
        z_title = 2.42  # title line
        z_row = 1.95  # the Fibonacci thread row
        z_lab = 1.50  # F_2, F_6, ... labels under the lit numbers
        r_top = 0.65  # top equation slot (first derivation row)
        row_dz = -0.85  # slot spacing; slots at 0.65 / -0.20 / -1.05
        z_cap = -1.55  # caption line below the slots
        td = 0.7  # derivation font scale
        mm = 0.85  # matrix strip font scale

        def paint(o, spec, color, begin, tt=0.4):
            o.change_color_of_letters(o.find_letters(spec), color,
                                      begin_time=begin, transition_time=tt)

        def epulse(o, spec, begin, peak=4.0, rise=0.3, fall=0.45):
            """Emission pulse on a substring (colors stay untouched)."""
            for i in o.find_letters(spec):
                letter = o.letters[i]
                ibpy.change_emission_to(letter, peak, begin_time=begin,
                                        transition_time=rise)
                ibpy.change_emission_to(letter, 0.5,
                                        begin_time=begin + rise + 0.1,
                                        transition_time=fall)
            return begin + rise + 0.1 + fall

        def group_center_x(o, spec):
            """World x of the midpoint of a letter group (for underlabels)."""
            idx = o.find_letters(spec)
            xs = [o.letters[i].ref_obj.location[0] for i in idx]
            return o.ref_obj.location[0] + 0.5 * (min(xs) + max(xs)) + 0.07

        # =================================================================
        # Beat A -- 1, 8, 55, 377 stride along Fibonacci four steps apart
        # =================================================================
        title = SimpleTexBObject(r"\text{The last secret of }1,\ 8,\ 55,\ 377",
                                 color=KEY, text_size="large", aligned="center",
                                 location=[0.3, 0, z_title], name="OF_title")
        t0 = title.write(begin_time=t0, transition_time=1.2)
        for spec in ("1", "8", "55", "377"):
            paint(title, spec, FIB, begin=t0 - 0.6, tt=0.5)

        row = SimpleTexBObject(
            r"1,\ 1,\ 2,\ 3,\ 5,\ 8,\ 13,\ 21,\ 34,\ 55,\ 89,\ 144,\ 233,\ 377,\ \dots",
            color="text", text_size=0.75, aligned="center",
            location=[0.3, 0, z_row], name="OF_row")
        t0 = 0.1 + row.write(begin_time=t0 + 0.3, transition_time=1.8)

        # every fourth entry lights up, its Fibonacci index appearing below
        stride = [("1@1", r"F_2"), ("8@0", r"F_6"),
                  ("55", r"F_{10}"), ("377", r"F_{14}")]
        strides = [s for s, _ in stride]
        labels = []
        for k, (spec, lab) in enumerate(stride):
            start = t0 + 0.55 * k
            paint(row, spec, FIB, begin=start, tt=0.4)
            epulse(row, spec, begin=start)
            label = SimpleTexBObject(lab, color=KEY, text_size="small",
                                     aligned="center",
                                     location=[group_center_x(row, spec), 0, z_lab],
                                     name="OF_lab_%d" % k)
            label.write(begin_time=start + 0.15, transition_time=0.5)
            labels.append(label)
        t0 += 0.55 * 3 + 0.85

        # =================================================================
        # Beat B -- the machine: Q generates Fibonacci, the stride is Q^4
        # =================================================================
        Q1 = r"Q=\begin{pmatrix}1&1\\1&0\end{pmatrix}"
        Q2 = r"Q^2=\begin{pmatrix}2&1\\1&1\end{pmatrix}"
        Q3 = r"Q^3=\begin{pmatrix}3&2\\2&1\end{pmatrix}"
        Q4 = r"Q^4=\begin{pmatrix}5&3\\3&2\end{pmatrix}"
        Qn = r"Q^n=\begin{pmatrix}F_{n+1}&F_n\\F_n&F_{n-1}\end{pmatrix}"

        # the four matrices appear side by side; as each lands its
        # off-diagonal Fibonacci entry lights up in yellow
        powers = []
        for k, (tex, x, fib_spec) in enumerate([
                (Q1, -3.05, None), (Q2, -1.05, "1@0"),
                (Q3, 0.95, "2@0"), (Q4, 3.0, "3@0")]):
            o = SimpleTexBObject(tex, color="text", text_size=mm,
                                 aligned="center", location=[x, 0, 0.45],
                                 name="OF_Q%d" % k)
            t0 = o.write(begin_time=t0 + (0.25 if k else 0.4),
                         transition_time=0.9)
            if fib_spec:  # both off-diagonal copies of F_k
                paint(o, fib_spec, FIB, begin=t0 - 0.3)
                paint(o, fib_spec[:-1] + "1", FIB, begin=t0 - 0.3)
            powers.append(o)
        q4 = powers[-1]

        law = SimpleTexBObject(Qn, color="text", text_size=mm,
                               aligned="center", location=[0.3, 0, -0.95],
                               name="OF_law")
        t0 = 0.1 + law.write(begin_time=t0 + 0.4, transition_time=1.1)
        paint(law, r"F_n@all", FIB, begin=t0, tt=0.5)
        t0 += 0.5

        capB = SimpleTexBObject(
            r"\text{four steps at a time}\;\Longrightarrow\;\text{powers of }Q^4",
            color=KEY, text_size="small", aligned="center",
            location=[0.3, 0, z_cap], name="OF_capB")
        t0 = 0.6 + capB.write(begin_time=t0 + 0.1, transition_time=0.9)

        # let the caption breathe, then Q^4 -- the star -- glides to the
        # left anchor over a cleared stage (the rest fades out first, so
        # nothing is crossed mid-fade)
        t0 += 0.7
        for o in powers[:-1] + [law, capB]:
            o.disappear(begin_time=t0, transition_time=0.5)
        q4.move(direction=Vector((-5.95, 0, 0.2)), begin_time=t0 + 0.55,
                transition_time=1.0)
        t0 += 1.75

        # =================================================================
        # Beat C -- the scale factor: char. equation of Q^4 and phi^4
        # =================================================================
        cderiv = BDerivation(r"\det(Q^4-\lambda\,I)=0",
                             name="OF_char", color="text", text_size=td,
                             location=Vector((-0.9, 0, r_top)),
                             line_spacing=Vector((0, 0, row_dz)))
        t0 = 0.4 + cderiv.write(begin_time=t0, transition_time=1.0)
        paint(cderiv.current, r"\lambda", KEY, begin=t0 - 0.4)

        # plug in the entries of Q^4 (they flash in the anchored matrix) ...
        epulse(q4, "5", begin=t0 + 0.2)
        epulse(q4, "3@all", begin=t0 + 0.35)
        epulse(q4, "2", begin=t0 + 0.5)
        t0 = 0.4 + cderiv.step(
            r"\det\begin{pmatrix}5-\lambda&3\\3&2-\lambda\end{pmatrix}=0",
            mode="replace",
            map={r"\det": r"\det", r"\lambda": r"\lambda@all", r"=0": r"=0",
                 r"Q^4": None, r"I": None},
            begin_time=t0, transition_time=1.3)

        # ... grind out the 2x2 determinant: the two 3s merge into the 9
        t0 = 0.4 + cderiv.step(r"(5-\lambda)(2-\lambda)-9=0", mode="replace",
                               map={r"5-\lambda": r"5-\lambda",
                                    r"2-\lambda": r"2-\lambda",
                                    r"3@all": "9", r"=0": r"=0",
                                    r"\det": None},
                               begin_time=t0, transition_time=1.3)

        t0 = 0.3 + cderiv.step(r"\lambda^2-7\,\lambda+1=0", mode="replace",
                               map={r"(5-\lambda)(2-\lambda)-9":
                                        r"\lambda^2-7\,\lambda+1",
                                    r"=0": r"=0"},
                               begin_time=t0, transition_time=1.3)
        chareq = cderiv.current
        paint(chareq, "7", HOT, begin=t0, tt=0.4)
        epulse(chareq, "7", begin=t0)
        t0 += 0.7

        # its roots are the fourth powers of the golden ratio
        t0 = 0.4 + cderiv.step(r"\lambda_\pm=\varphi^{\pm 4}", mode="new_line",
                               map={r"\lambda@1": r"\lambda"},
                               begin_time=t0, transition_time=1.2)
        paint(cderiv.current, r"\varphi^{\pm 4}", GRN, begin=t0 - 0.5)

        # and their sum is the 7: the trace identity
        t0 = 0.3 + cderiv.step(r"7=\varphi^{4}+\varphi^{-4}", mode="new_line",
                               map={r"\varphi^{\pm 4}":
                                        r"\varphi^{4}+\varphi^{-4}"},
                               begin_time=t0, transition_time=1.2)
        trace = cderiv.current
        paint(trace, "7", HOT, begin=t0 - 0.5)
        paint(trace, r"\varphi^{4}", GRN, begin=t0 - 0.5)
        paint(trace, r"\varphi^{-4}", GRN, begin=t0 - 0.5)

        capC = SimpleTexBObject(
            r"\text{each generation scales by }\varphi^{4}",
            color=GRN, text_size="small", aligned="center",
            location=[0.3, 0, z_cap], name="OF_capC")
        t0 = 0.2 + capC.write(begin_time=t0 + 0.1, transition_time=0.9)

        # the thread hops: 1 -> 8 -> 55 -> 377, one factor phi^4 each
        for k, spec in enumerate(strides):
            epulse(row, spec, begin=t0 + 0.35 * k)
        t0 += 0.35 * 3 + 0.9

        # keep the characteristic equation; everything else steps aside
        eigen1, eigen2 = cderiv.lines[-2], cderiv.lines[-1]
        for o in (eigen1, eigen2, capC, q4):
            o.disappear(begin_time=t0, transition_time=0.6)
        t0 += 0.8

        # =================================================================
        # Beat D -- Cayley-Hamilton: insert Q^4 for lambda, multiply up
        # =================================================================
        deriv = BDerivation(r"(Q^4)^2-7\,Q^4+I=0",
                            name="OF_rec", color="text", text_size=td,
                            location=Vector((-1.1, 0, r_top + row_dz)),
                            line_spacing=Vector((0, 0, row_dz)))
        # both chains share one '=' spine
        m1 = deriv.current
        m1.align(chareq, char_index=m1.find_letters('=')[0],
                 other_char_index=chareq.find_letters('=')[0])
        arrow = SimpleTexBObject(r"\lambda\;\longrightarrow\;Q^4",
                                 color=KEY, text_size="small", aligned="center",
                                 location=[3.0, 0, r_top + 0.5 * row_dz],
                                 name="OF_arrow")
        arrow.write(begin_time=t0, transition_time=0.7)
        t0 = 0.2 + deriv.write(begin_time=t0 + 0.3, transition_time=1.2)
        epulse(chareq, r"\lambda@all", begin=t0 - 0.5)
        epulse(m1, r"Q^4@all", begin=t0 - 0.3)
        paint(m1, "7", HOT, begin=t0 - 0.3)
        t0 += 0.4

        # rearrange: the matrix identity behind the recursion
        t0 = 0.4 + deriv.step(r"Q^8=7\,Q^4-I", mode="swap",
                              map={r"(Q^4)^2": r"Q^8", r"7\,Q^4": r"7\,Q^4",
                                   r"I": r"I"},
                              lift=0.45, begin_time=t0, transition_time=1.5)
        paint(deriv.current, "7", HOT, begin=t0 - 0.4, tt=0.3)

        # the lambda-line has served its purpose; the identity moves up
        chareq.disappear(begin_time=t0, transition_time=0.6)
        arrow.disappear(begin_time=t0, transition_time=0.6)
        t0 = deriv.move(direction=Vector((0, 0, -row_dz)), begin_time=t0 + 0.3,
                        transition_time=0.9)

        mult = SimpleTexBObject(r"\times\;Q^{4(n-1)+2}",
                                color=KEY, text_size="small", aligned="center",
                                location=[3.55, 0, r_top + 0.42],
                                name="OF_mult")
        t0 = 0.2 + mult.write(begin_time=t0, transition_time=0.7)

        # multiply by Q^{4(n-1)+2}: the whole Fibonacci strand through F_2
        t0 = 0.4 + deriv.step(r"Q^{4(n+1)+2}=7\,Q^{4n+2}-Q^{4(n-1)+2}",
                              mode="replace",
                              map={r"Q^8": r"Q^{4(n+1)+2}",
                                   r"7\,Q^4": r"7\,Q^{4n+2}",
                                   r"I": r"Q^{4(n-1)+2}"},
                              begin_time=t0, transition_time=1.5)
        paint(deriv.current, "7", HOT, begin=t0 - 0.4, tt=0.3)
        mult.disappear(begin_time=t0 - 0.3, transition_time=0.5)

        # read the off-diagonal entries: they are Fibonacci numbers
        t0 = 0.3 + deriv.step(r"F_{4(n+1)+2}=7\,F_{4n+2}-F_{4(n-1)+2}",
                              mode="new_line",
                              map={r"Q^{4(n+1)+2}": r"F_{4(n+1)+2}",
                                   r"Q^{4n+2}": r"F_{4n+2}",
                                   r"Q^{4(n-1)+2}": r"F_{4(n-1)+2}"},
                              begin_time=t0, transition_time=1.5)
        frec = deriv.current
        for spec in (r"F_{4(n+1)+2}", r"F_{4n+2}", r"F_{4(n-1)+2}"):
            paint(frec, spec, FIB, begin=t0 - 0.6)
        paint(frec, "7", HOT, begin=t0 - 0.6, tt=0.3)
        note = SimpleTexBObject(r"\text{(the off-diagonal entries)}",
                                color=FIB, text_size="small", aligned="center",
                                location=[1.2, 0, r_top + 1.45 * row_dz],
                                name="OF_note")
        t0 = 0.2 + note.write(begin_time=t0, transition_time=0.7)

        # rename the strand: a_n := F_{4n+2} = 1, 8, 55, 377, ...
        deflab = SimpleTexBObject(r"a_n:=F_{4n+2}",
                                  color=FIB, text_size="small", aligned="center",
                                  location=[3.5, 0, r_top + 1.5 * row_dz],
                                  name="OF_def")
        deflab.write(begin_time=t0, transition_time=0.8)
        t0 = 0.4 + deriv.step(r"a_{n+1}=7\,a_n-a_{n-1}", mode="new_line",
                              map={r"F_{4(n+1)+2}": r"a_{n+1}",
                                   r"F_{4n+2}": r"a_n",
                                   r"F_{4(n-1)+2}": r"a_{n-1}"},
                              begin_time=t0 + 0.5, transition_time=1.4)
        arec = deriv.current
        paint(arec, "7", HOT, begin=t0 - 0.5)
        t0 += 0.3

        # =================================================================
        # Beat E -- the check on 1, 8, 55, 377 and the unmasked quadratic
        # =================================================================
        line_q = deriv.lines[-3]  # the Q-strand line still in the top slot
        for o in (line_q, frec, note, deflab):
            o.disappear(begin_time=t0, transition_time=0.6)
        t0 = deriv.move(direction=Vector((0, 0, -2 * row_dz)),
                        begin_time=t0 + 0.3, transition_time=0.9)

        # 55 = 7*8 - 1 : the ingredients flash in the thread row above
        epulse(row, "8@0", begin=t0)
        epulse(row, "1@1", begin=t0 + 0.15)
        t0 = 0.2 + deriv.step(r"55=7\cdot 8-1", mode="new_line",
                              map={"7": "7"},
                              begin_time=t0, transition_time=1.2)
        for spec in ("55", "8", "1"):
            paint(deriv.current, spec, FIB, begin=t0 - 0.5)
        paint(deriv.current, "7", HOT, begin=t0 - 0.5)
        epulse(row, "55", begin=t0 - 0.3)
        t0 += 0.3

        # 377 = 7*55 - 8
        epulse(row, "55", begin=t0)
        epulse(row, "8@0", begin=t0 + 0.15)
        t0 = 0.2 + deriv.step(r"377=7\cdot 55-8\;\;\checkmark", mode="new_line",
                              map={"7": "7@2"},
                              begin_time=t0, transition_time=1.2)
        chk = deriv.current
        for spec in ("377", "55", "8"):
            paint(chk, spec, FIB, begin=t0 - 0.5)
        paint(chk, "7@2", HOT, begin=t0 - 0.5)
        paint(chk, r"\checkmark", GRN, begin=t0, tt=0.4)
        epulse(row, "377", begin=t0 - 0.3)
        t0 += 0.4

        # the script's punch line: the quadratic, unmasked
        quote = SimpleTexBObject(
            r"\text{``seven times this, minus the last one''}",
            color=HOT, text_size=0.75, aligned="center",
            location=[0.3, 0, -1.42], name="OF_quote")
        t0 = 0.1 + quote.write(begin_time=t0, transition_time=1.3)
        unmask = SimpleTexBObject(
            r"\text{--- the intimidating quadratic, unmasked}",
            color=GRN, text_size="small", aligned="center",
            location=[0.3, 0, -1.76], name="OF_unmask")
        t0 = 0.3 + unmask.write(begin_time=t0, transition_time=0.9)

        # final bow: the recursion's 7 and the four numbers glow once more
        epulse(arec, "7", begin=t0)
        for k, spec in enumerate(strides):
            epulse(row, spec, begin=t0 + 0.2 + 0.25 * k)
        t0 += 0.2 + 0.25 * 3 + 0.9

        self.t0 = t0

    def boring_tilings(self):
        """Two 'boring' periodic tilings, shown side by side and built in
        lock-step: both fundamental domains appear together, are marked
        together, are then shifted together along u, along v and along u+v, and
        finally both halves fill in over the same time span.

        Left:  the Pythagorean tiling -- two different rectangles (a big square
               in ``drawing``, a small one in ``joker``) whose two-tile motif
               repeats along the lattice vectors u=(a,b), v=(-b,a).
        Right: the trihexagonal (3.6.3.6) tiling -- regular hexagons (``orange``)
               with the gaps filled by equilateral triangles (``yellow``),
               repeating along a triangular lattice u, v.

        In each case the fundamental domain is marked by an outlined
        parallelogram together with the two lattice-translation arrows.
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        set_alpha_composition()

        # top-down camera looking onto the x-y plane
        ibpy.set_camera_location(location=[0, -1, 22])
        cam_empty = EmptyCube(location=[0, -1, 0], name='BoringCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=cam_empty)

        r32 = r3 / 2
        nm = [0]  # running counter for unique polygon names

        # ---- shared helpers -------------------------------------------------
        def grow_poly(verts_xy, color, tb, tt=0.4, move=None):
            """One filled, outlined tile.  ``verts_xy`` are (x, y) pairs in the
            z=0 plane; optional ``move`` (Vector) animates a translation."""
            # force counter-clockwise winding so the face normal points +z
            # (toward the top-down camera); otherwise back-facing tiles -- the
            # outward triangles in particular -- render dark under the light.
            vs = list(verts_xy)
            area = sum(vs[i][0] * vs[(i + 1) % len(vs)][1] -
                       vs[(i + 1) % len(vs)][0] * vs[i][1] for i in range(len(vs)))
            if area < 0:
                vs = vs[::-1]
            p = Polygon([Vector((x, y, 0.0)) for (x, y) in vs],
                        color=color, name='btile_%d' % nm[0], reordering=False)
            nm[0] += 1
            p.add_mesh_modifier(type='NODES',
                                node_modifier=MakeFrameModifier(color='background',
                                                                thickness=0.03))
            if move is None:
                p.grow(begin_time=tb, transition_time=tt)
            else:
                p.grow(begin_time=tb, transition_time=0.01)
                p.move(direction=move, begin_time=tb, transition_time=tt)
            return p

        def mark_fundamental_domain(corners, c0, uv, tex, tb, tag, edge_color='important'):
            """Red outline of the parallelogram ``corners`` plus the two lattice
            arrows u, v drawn from ``c0`` with LaTeX labels ``tex=(u_tex,v_tex)``.
            Returns the time when the marking is finished."""
            t = tb
            for i in range(len(corners)):
                a, b = corners[i], corners[(i + 1) % len(corners)]
                edge = Cylinder.from_start_to_end(
                    start=Vector((a.x, a.y, 0.15)), end=Vector((b.x, b.y, 0.15)),
                    thickness=0.35, color=edge_color)
                edge.grow(modus='from_start', begin_time=t, transition_time=0.3)
                t += 0.3
            u_vec, v_vec = uv
            au = PArrow(start=Vector((c0.x, c0.y, 0.2)),
                        end=Vector((c0.x + u_vec.x, c0.y + u_vec.y, 0.2)),
                        color='red', thickness=2, name='%s_u' % tag)
            av = PArrow(start=Vector((c0.x, c0.y, 0.2)),
                        end=Vector((c0.x + v_vec.x, c0.y + v_vec.y, 0.2)),
                        color='red', thickness=2, name='%s_v' % tag)
            au.grow(begin_time=t, transition_time=0.5)
            av.grow(begin_time=t + 0.5, transition_time=0.5)
            t += 1.0
            # u_lab = SimpleTexBObject(tex[0], color='red', text_size='large', aligned='center',
            #                          location=Vector((c0.x + 0.5 * u_vec.x + 0.35,
            #                                           c0.y + 0.5 * u_vec.y - 0.4, 0.3)),
            #                          name='%s_ulab' % tag)
            # v_lab = SimpleTexBObject(tex[1], color='red', text_size='large', aligned='center',
            #                          location=Vector((c0.x + 0.5 * v_vec.x - 0.55,
            #                                           c0.y + 0.5 * v_vec.y, 0.3)),
            #                          name='%s_vlab' % tag)
            # u_lab.write(begin_time=t, transition_time=0.4)
            # v_lab.write(begin_time=t + 0.4, transition_time=0.4)
            return t + 0.9

        # =====================================================================
        # Geometry -- tiling 1: two rectangles (Pythagorean), on the left
        # =====================================================================
        a1, b1l = 2.4, 1.2
        U1 = Vector((a1, b1l, 0))
        V1 = Vector((-b1l, a1, 0))
        c0_1 = Vector((-8.0, -1.0, 0))
        big = [(0, 0), (a1, 0), (a1, a1), (0, a1)]
        small = [(a1, 0), (a1 + b1l, 0), (a1 + b1l, b1l), (a1, b1l)]
        motif1_colors = ('drawing', 'joker')  # big rectangle, small rectangle

        def motif1(m, n):
            off = c0_1 + m * U1 + n * V1
            return off, [[(off.x + x, off.y + y) for (x, y) in big],
                         [(off.x + x, off.y + y) for (x, y) in small]]

        corners1 = [c0_1, c0_1 + U1, c0_1 + U1 + V1, c0_1 + V1]
        shifts1 = [U1, V1, U1 + V1]

        # cells filling the left half, minus the four drawn as FD + shifted FDs
        done1 = {(0, 0), (1, 0), (0, 1), (1, 1)}
        cells1 = []
        for m in range(-6, 6):
            for n in range(-6, 6):
                off, _ = motif1(m, n)
                cx, cy = off.x + a1 * 0.5, off.y + a1 * 0.5
                if -12.5 <= cx <= -0.8 and -7.3 <= cy <= 5.2 and (m, n) not in done1:
                    cells1.append((m, n, off))
        cells1.sort(key=lambda c: (c[2] - c0_1).length)

        # =====================================================================
        # Geometry -- tiling 2: triangles + hexagons (trihexagonal 3.6.3.6)
        # =====================================================================
        s2 = 1.4
        L2 = 2 * s2
        # pointy-top hexagons (vertices at 30+60k deg): the whole tiling is the
        # 30 deg-rotated 3.6.3.6, so the triangles point left/right
        B1 = Vector((L2 * r32, L2 / 2, 0))
        B2 = Vector((0, L2, 0))
        c0_2 = Vector((8.0, -1.0, 0))
        hexv = [Vector((s2 * cos(pi / 6 + k * pi / 3), s2 * sin(pi / 6 + k * pi / 3), 0))
                for k in range(6)]

        def hexagon(center):
            return [(center.x + v.x, center.y + v.y) for v in hexv]

        def triangles_of(center):
            """The six outward equilateral triangles on a hexagon's edges;
            apex = 2*midpoint - center."""
            verts = [center + v for v in hexv]
            out = []
            for k in range(6):
                a, b = verts[k], verts[(k + 1) % 6]
                mid = 0.5 * (a + b)
                apex = 2 * mid - center
                out.append(((a + b + apex) / 3,
                            [(a.x, a.y), (b.x, b.y), (apex.x, apex.y)]))
            return out

        # collect hexagons and (deduplicated) triangles inside the right half
        def in_bounds2(x, y):
            return 1.0 <= x <= 12.5 and -7.3 <= y <= 5.0

        hex_cells, tri_map = [], {}
        for m in range(-5, 6):
            for n in range(-5, 6):
                c = c0_2 + m * B1 + n * B2
                if in_bounds2(c.x, c.y):
                    hex_cells.append(c)
                for cen, tri in triangles_of(c0_2 + m * B1 + n * B2):
                    if in_bounds2(cen.x, cen.y):
                        tri_map[(round(cen.x, 2), round(cen.y, 2))] = (cen, tri)
        tris = list(tri_map.values())

        # the fundamental domain is the rhombus centred on the seed hexagon,
        # |a|<=1/2, |b|<=1/2 in the {B1, B2} basis; the two triangles inside it
        # are the ones that appear first (with the hexagon)
        det = B1.x * B2.y - B1.y * B2.x

        def in_fd(p):
            dx, dy = p.x - c0_2.x, p.y - c0_2.y
            a = (dx * B2.y - dy * B2.x) / det
            b = (B1.x * dy - B1.y * dx) / det
            return -0.5 - 1e-6 <= a <= 0.5 + 1e-6 and -0.5 - 1e-6 <= b <= 0.5 + 1e-6

        # the two triangles inside the FD appear first, together with the hexagon
        seed_tris = sorted([ct for ct in tris if in_fd(ct[0])],
                           key=lambda ct: (ct[0] - c0_2).length)
        seed_tri_verts = [tri for _, tri in seed_tris]

        # the fundamental domain: rhombus centred on the seed hexagon
        half = 0.5 * (B1 + B2)
        corners2 = [c0_2 - half, c0_2 + B1 - half, c0_2 + B1 + B2 - half, c0_2 + B2 - half]
        shifts2 = [B1, B2, B1 + B2]

        # everything left to fill, minus the FD motif and its three shifted copies
        shifts0 = [Vector((0, 0, 0))] + shifts2
        shown_hex = [c0_2 + sh for sh in shifts0]
        shown_tri_keys = {(round((cen + sh).x, 2), round((cen + sh).y, 2))
                          for cen, _ in seed_tris for sh in shifts0}
        rest_hex = sorted([c for c in hex_cells
                           if all((c - h).length > 0.1 for h in shown_hex)],
                          key=lambda c: (c - c0_2).length)
        rest_tri = sorted([ct for key, ct in tri_map.items() if key not in shown_tri_keys],
                          key=lambda ct: (ct[0] - c0_2).length)
        # interleave hexagons and triangles by radius so the tiling grows evenly
        events = ([('h', (c - c0_2).length, c) for c in rest_hex] +
                  [('t', (ct[0] - c0_2).length, ct[1]) for ct in rest_tri])
        events.sort(key=lambda e: e[1])

        # =====================================================================
        # Animation -- both tilings advance in lock-step
        # =====================================================================
        # 1) the fundamental domain of each tiling
        _, seed_shapes = motif1(0, 0)
        for shp, col in zip(seed_shapes, motif1_colors):
            grow_poly(shp, col, t0, tt=0.8)
        grow_poly(hexagon(c0_2), 'orange', t0, tt=0.8)
        for cen, tri in seed_tris:
            grow_poly(tri, 'yellow', t0, tt=0.8)
        t0 += 1.1

        # 2) mark both fundamental domains (equal duration, so they end together)
        t_fd1 = mark_fundamental_domain(corners1, c0_1, (U1, V1),
                                        (r"\vec{u}", r"\vec{v}"), t0, 'fd1')
        t_fd2 = mark_fundamental_domain(corners2, c0_2 - half, (B1, B2),
                                        (r"\vec{u}", r"\vec{v}"), t0, 'fd2',
                                        edge_color='joker')
        t0 = max(t_fd1, t_fd2) + 0.3

        # 3) shift both fundamental domains along u, then v, then u+v
        for sh1, sh2 in zip(shifts1, shifts2):
            _, shapes = motif1(0, 0)
            for shp, col in zip(shapes, motif1_colors):
                grow_poly(shp, col, t0, tt=0.7, move=sh1)
            grow_poly(hexagon(c0_2), 'orange', t0, tt=0.7, move=sh2)
            for tri in seed_tri_verts:
                grow_poly(tri, 'yellow', t0, tt=0.7, move=sh2)
            t0 += 0.9
        t0 += 0.4

        # 4) fill both halves.  The per-tile spacing is scaled by the tile count
        #    so that the two sweeps start and finish at the same time.
        fill_time = 7.0
        step1 = fill_time / max(1, len(cells1))
        for i, (m, n, off) in enumerate(cells1):
            _, shapes = motif1(m, n)
            for shp, col in zip(shapes, motif1_colors):
                grow_poly(shp, col, t0 + i * step1, tt=0.4)
        step2 = fill_time / max(1, len(events))
        for i, (kind, _, data) in enumerate(events):
            if kind == 'h':
                grow_poly(hexagon(data), 'orange', t0 + i * step2, tt=0.4)
            else:
                grow_poly(data, 'yellow', t0 + i * step2, tt=0.4)
        t0 += fill_time + 0.4 + 0.6

        self.t0 = t0

    def fractal_shader(self):
        """Improved version: default-true fix so un-terminated points are bright (in-fractal)."""
        _setup_render()
        _setup_camera(distance=18)

        t0 = 0
        plane = Plane(u=[-10, 10], v=[-10, 10], color="hat_tile_fractal")
        t0 = 0.5 + plane.appear(begin_time=t0, transition_time=0.5)
        self.t0 = t0

    def the_background_fractal(self):
        """Introduce the fundamental trapezoid of the background fractal.

        Camera looks along +y onto the x-z plane.  A default (x-z) coordinate
        system from :class:`CoordinateSystem2` carries the trapezoid (taken from
        ``mathematica/FractalBackground.nb``), which is drawn edge by edge.
        Spheres then pop up at the four vertices while -- in parallel -- their
        exact coordinates are written next to the trapezoid, using the
        golden-ratio abbreviation ``phi``, fractions and roots (no decimals).
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)

        # camera straight along +y, looking onto the x-z plane
        ibpy.set_camera_location(location=[0, -20, 0])
        cam_empty = EmptyCube(location=[0, 0, 0], name='FractalCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=cam_empty)

        # --- title ---
        title = SimpleTexBObject(r"\text{The Background Fractal}", color="example",
                                 text_size="Large", aligned="center",
                                 location=Vector([0, 0, 6.25]), name='FractalTitle')

        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        # --- golden-ratio abbreviation ---
        phi_text = SimpleTexBObject(r"\text{The Golden Ratio}", color="important", text_size="large", aligned="left",
                                    location=Vector([6, 0, 4.5]), name="GoldenRatio")
        phi_def = SimpleTexBObject(r"\varphi=\frac{1+\sqrt{5}}{2}", color="important",
                                   text_size="large", aligned="left",
                                   location=Vector([6, 0, 3.5]), name='GoldenRatioDef')
        t0 = phi_text.write(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + phi_def.write(begin_time=t0, transition_time=0.3)

        # --- coordinate system in the x-z plane (default directions) ---
        s = 1.5  # blender units per math unit
        coord_loc = Vector([0, -0.1, 0])
        x_max, y_max = 3.5, 3.35
        coords = CoordinateSystem2(
            dimension=2,
            location=coord_loc,
            radii=[0.025] * 2,
            lengths=[2 * s * x_max, 2 * s * y_max], domains=[[-x_max, x_max], [-y_max, y_max]],
            n_tics=[int(x_max) * 2, int(y_max) * 2], tic_labels=['AUTO', 'AUTO'], tic_label_digits=[0, 0],
            include_zeros=[False, False], colors=['drawing', 'drawing'],
            tic_label_shifts=[Vector([0, 0, -0.5]), Vector([0, 0, 0])],
            name='FractalCoords', axes_labels={'x': Vector([0.5, 0, s * 7]), 'y': Vector([-0.5, 0, s * 7])})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=2)

        def to_world(mx, my):
            """math vector (x, y) -> world translation in the x-z plane."""
            return Vector((s * mx, 0.0, s * my))

        r32 = r3 / 2
        u = to_world(phi + 1.5, r32)
        v = to_world(phi / 2, r32 * (2 + phi))

        def to_loc(x, y):
            """Math point (x, y) -> world location in the x-z plane."""
            return coord_loc + Vector((s * x, 0.0, s * y))

        origin = Vector([0, 0, 0])
        # red arrows from the origin
        arrow_u = PArrow(start=origin, end=u, color='red', thickness=2, name='LatticeU')
        arrow_v = PArrow(start=origin, end=v, color='red', thickness=2, name='LatticeV')
        arrow_u.grow(begin_time=t0, transition_time=0.6)
        t0 = arrow_v.grow(begin_time=t0, transition_time=0.6) + 0.2

        u_label = SimpleTexBObject(r"\vec{u}", color='red', text_size='large', aligned='center',
                                   location=Vector([4.122, -0.2, 0.64]), name='FDULabel')
        v_label = SimpleTexBObject(r"\vec{v}", color='red', text_size='large', aligned='center',
                                   location=Vector([0.54, -0.2, 4.17]), name='FDVLabel')

        shift_in_front = [arrow_u, arrow_v, u_label, v_label]
        for obj in shift_in_front:
            obj.move(direction=Vector((0, -0.2, 0)), begin_time=0, transition_time=0)

        u_label.write(begin_time=t0, transition_time=0.4)
        t0 = v_label.write(begin_time=t0, transition_time=0.4) + 0.3

        # --- u and v coordinates, written where the golden ratio used to be,
        #     headed by "Fundamental Domain" (important color) ---
        fd_header = SimpleTexBObject(r"\text{Fundamental Domain}", color="important", text_size="large",
                                     aligned="left", location=Vector([-11.5, 0, 4.5]), name="FundamentalDomain")
        u_def = SimpleTexBObject(r"\vec{u}=\left(\varphi+\tfrac{3}{2},\,\tfrac{\sqrt{3}}{2}\right)",
                                 color="important", text_size="large", aligned="left",
                                 location=Vector([-11.5, 0, 3.5]), name="LatticeUDef")
        v_def = SimpleTexBObject(
            r"\vec{v}=\left(\tfrac{\varphi}{2},\,\tfrac{\sqrt{3}}{2}\left(2+\varphi\right)\right)",
            color="important", text_size="large", aligned="left",
            location=Vector([-11.5, 0, 2.6]), name="LatticeVDef")
        t0 = fd_header.write(begin_time=t0, transition_time=0.5)
        u_def.write(begin_time=t0, transition_time=0.4)
        t0 = v_def.write(begin_time=t0, transition_time=0.4) + 0.3

        # --- trapezoid vertices (exact: golden ratio + roots, no decimals) ---
        # FractalBackground.nb: {{0,0},{phi^2,0},{phi^2-1/2,sqrt(3)/2},{1/2,sqrt(3)/2}}

        verts_math = [
            (0, 0),
            (phi ** 2, 0),
            (phi ** 2 - 0.5, r32),
            (0.5, r32),
        ]
        labels_tex = [
            r"\left(0,\,0\right)",
            r"\left(\varphi^{2},\,0\right)",
            r"\left(\varphi^{2}-\tfrac{1}{2},\,\tfrac{\sqrt{3}}{2}\right)",
            r"\left(\tfrac{1}{2},\,\tfrac{\sqrt{3}}{2}\right)",
        ]
        # push each label outward, away from the trapezoid centre
        label_offsets = [
            0.5 * Vector([-1.4, 0, -1.5]),  # (0, 0)               lower-left
            0.5 * Vector([1.6, 0, -1.5]),  # (phi^2, 0)           lower-right
            0.5 * Vector([2.4, 0, 1.0]),  # (phi^2-1/2, r3/2)    upper-right
            0.5 * Vector([1, 0, 1.0]),  # (1/2, r3/2)          upper-left
        ]
        verts = [to_loc(x, y) for (x, y) in verts_math]
        n = len(verts)

        # --- draw the trapezoid edge by edge ---
        edge_dt = 1.0
        edges = []
        for i in range(n):
            edge = Cylinder.from_start_to_end(start=verts[i], end=verts[(i + 1) % n],
                                              thickness=0.5, color='joker')
            edge.grow(modus='from_start', begin_time=t0, transition_time=edge_dt)
            edges.append(edge)
            t0 += edge_dt
        t0 += 0.5

        # --- spheres at the vertices + coordinate labels (in parallel) ---
        dots, labels = [], []
        for i in range(n):
            dot = Sphere(0.2, location=verts[i], color='important', name='FractalVertex%d' % i)
            dot.grow(begin_time=t0, transition_time=0.5)
            dots.append(dot)

            label = SimpleTexBObject(labels_tex[i], color="text", text_size="normal",
                                     aligned="center", name='FractalCoordLabel%d' % i,
                                     location=verts[i] + label_offsets[i])
            label.write(begin_time=t0, transition_time=1.0)
            labels.append(label)
            t0 += 1.2
        t0 += 0.5

        # --- fill the trapezoid in as a solid polygon, then drop the scaffolding
        #     (edges, vertices, coordinate labels) and the golden-ratio note ---
        trapezoid = Polygon(verts, color='joker', name='FractalTrapezoid')
        trapezoid.add_mesh_modifier(type='NODES',
                                    node_modifier=MakeFrameModifier(color='important', thickness=0.02))
        trapezoid.grow(begin_time=t0, transition_time=1.0)

        for obj in edges + dots + labels + [phi_text, phi_def]:
            obj.disappear(begin_time=t0, transition_time=1.0)
        t0 += 1.5

        # --- five copies, each rotated by i*60 deg about the origin (the
        #     trapezoid's 60 deg corner sits at V0=(0,0,0)), so the six tiles
        #     tile into the hexagonal generator of the fractal.  Animate the
        #     copies successively, one swinging into its slot after another. ---

        rot_dt = 0.8
        copies = []
        for i in range(1, 6):
            copy = Polygon(verts, color='joker', name='FractalTrapezoid%d' % i)
            copy.add_mesh_modifier(type='NODES',
                                   node_modifier=MakeFrameModifier(color='important', thickness=0.02))
            copy.grow(begin_time=t0, transition_time=0)
            copy.rotate(rotation_euler=[0, -i * pi / 3, 0], pivot=origin, begin_time=t0, transition_time=rot_dt)
            copies.append(copy)
            t0 += rot_dt
        t0 += 0.5

        # --- introduce the two lattice vectors u and v (red) and translate a
        #     copy of the whole six-trapezoid flower along u, v, -u, -v, u+v.
        #     u, v are the fundamental translations from FractalBackground.nb:
        #       u = (phi + 3/2, sqrt(3)/2),  v = (phi/2, sqrt(3)/2 (2 + phi)) ---

        # make room: drop the title and pull the camera back so all copies fit
        # title.disappear(begin_time=t0, transition_time=0.5)
        # ibpy.camera_move(shift=Vector([0, -12, 0]), begin_time=t0, transition_time=1.0)
        t0 += 1.2

        # the six petals with their 60 deg rotations baked into the vertices,
        # so a translated flower can be rebuilt directly at the right position
        def _ry(p, theta):
            c, sn = cos(theta), sin(theta)
            return Vector((p[0] * c + p[2] * sn, 0.0, -p[0] * sn + p[2] * c))

        petal_verts = [[_ry(w, -i * pi / 3) for w in verts] for i in range(6)]

        def make_flower_copy(tag):
            polys = []
            for i, pv in enumerate(petal_verts):
                p = Polygon(pv, color='joker', name='FlowerCopy_%s_%d' % (tag, i))
                p.add_mesh_modifier(type='NODES',
                                    node_modifier=MakeFrameModifier(color='important', thickness=0.02))
                polys.append(p)
            return polys

        translations = [('u', u), ('v', v), ('mu', -u), ('mv', -v), ('uv', u + v), ('-uv', -u - v), ('u-v', u - v),
                        ('v-u', v - u)]
        move_dt = 0.8
        all_tiles = [trapezoid] + copies
        for tag, shift in translations:
            for p in make_flower_copy(tag):
                p.grow(begin_time=t0, transition_time=0.01)
                p.move(direction=shift, begin_time=t0, transition_time=move_dt)
                all_tiles.append(p)
            t0 += move_dt
        t0 += 0.5

        # --- fade every trapezoid down to alpha=0.25 and bring the fractal
        #     background back.  appear() is a no-op here (self.appeared is
        #     already True), so the alpha is driven directly via
        #     disappear(alpha=...): 0.25 for the tiles, 1 for the fractal. ---
        for tile in all_tiles:
            tile.disappear(alpha=0.25, begin_time=t0, transition_time=1.0)

        # fade out the title and all remaining text with the same animation
        for txt in [title, fd_header, u_def, v_def, u_label, v_label]:
            txt.disappear(begin_time=t0, transition_time=1.0)
        t0 += 1.0

        self.t0 = t0 + 1

    def fundamental_domain(self):
        """Fold the triangular lattice into the fundamental domain.

        Re-introduces the same x-z coordinate system and the red lattice
        vectors u, v from :meth:`the_background_fractal`, overlays the
        triangular lattice (:class:`TriangularGridModifier`) and animates the
        modifier's ``l`` dial 0 -> 1.  That dial drives the modifier's
        back-translation, which maps every grid vertex into the (u, v)
        fundamental parallelogram.  The reduction formula is shown alongside.
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        set_alpha_composition()
        # camera straight along +y, looking onto the x-z plane (as before)
        ibpy.set_camera_location(location=[0, -20, 0])
        cam_empty = EmptyCube(location=[0, 0, 0], name='FDCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=cam_empty)

        title = SimpleTexBObject(r"\text{Fundamental Domain}", color="example", text_size="Large",
                                 aligned="center", location=Vector([-8.5, 0, 6.25]), name='FDTitle')
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        # --- same coordinate system as the_background_fractal ---
        s = 1.5  # blender units per math unit
        coord_loc = Vector([0, -0.1, 0])
        x_max, y_max = 3.5, 3.35
        coords = CoordinateSystem2(
            dimension=2,
            location=coord_loc,
            radii=[0.025] * 2,
            lengths=[2 * s * x_max, 2 * s * y_max], domains=[[-x_max, x_max], [-y_max, y_max]],
            n_tics=[int(x_max) * 2, int(y_max) * 2], tic_labels=['AUTO', 'AUTO'], tic_label_digits=[0, 0],
            include_zeros=[False, False], colors=['drawing', 'drawing'],
            tic_label_shifts=[Vector([0, 0, -0.5]), Vector([0, 0, 0])],
            name='FDCoords', axes_labels={'x': Vector([0.5, 0, s * 7]), 'y': Vector([-0.5, 0, s * 7])})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=1.5)

        # --- red lattice vectors u and v (same as before) ---
        phi = (1 + sqrt(5)) / 2
        r32 = r3 / 2

        def to_world(mx, my):
            return Vector((s * mx, 0.0, s * my))

        origin = Vector([0, 0, 0])
        u = to_world(phi + 1.5, r32)
        v = to_world(phi / 2, r32 * (2 + phi))
        arrow_u = PArrow(start=origin, end=u, color='red', thickness=2, name='FDVecU')
        arrow_v = PArrow(start=origin, end=v, color='red', thickness=2, name='FDVecV')
        # lift the arrows slightly toward the camera so they read over the grid
        for a in [arrow_u, arrow_v]:
            a.move(direction=Vector((0, -0.2, 0)), begin_time=0, transition_time=0)
        arrow_u.grow(begin_time=t0, transition_time=0.5)
        t0 = arrow_v.grow(begin_time=t0, transition_time=0.5) + 0.2

        u_label = SimpleTexBObject(r"\vec{u}", color='red', text_size='large', aligned='center',
                                   location=Vector([4.122, -0.2, 0.64]), name='FDULabel')
        v_label = SimpleTexBObject(r"\vec{v}", color='red', text_size='large', aligned='center',
                                   location=Vector([0.54, -0.2, 4.17]), name='FDVLabel')
        u_label.write(begin_time=t0, transition_time=0.4)
        t0 = v_label.write(begin_time=t0, transition_time=0.4) + 0.3

        # --- transformation formula (important color) ---
        f_head = SimpleTexBObject(r"\text{Mapping into the domain}", color="important", text_size="large",
                                  aligned="left", location=Vector([-11.5, 0, 5.0 + 0.5]), name='FDFormulaHead')
        f_p = SimpleTexBObject(r"\vec p=a\,\vec u+b\,\vec v", color="important", text_size="large",
                               aligned="left", location=Vector([-11.5, 0, 4.0 + 0.5]), name='FDFormulaP')
        f_t = SimpleTexBObject(r"T(\vec p)=\{a\}\,\vec u+\{b\}\,\vec v", color="important", text_size="large",
                               aligned="left", location=Vector([-11.5, 0, 3.0 + 0.5]), name='FDFormulaT')
        f_frac = SimpleTexBObject(r"\{x\}=x-\lfloor x\rfloor", color="important", text_size="large",
                                  aligned="left", location=Vector([-11.5, 0, 2.1 + 0.5]), name='FDFormulaFrac')
        f_head.write(begin_time=t0, transition_time=0.5)
        f_p.write(begin_time=t0 + 0.4, transition_time=0.4)
        f_t.write(begin_time=t0 + 0.8, transition_time=0.4)
        t0 = f_frac.write(begin_time=t0 + 1.2, transition_time=0.4) + 0.4

        # --- triangular lattice in the x-z plane, scaled to match the system.
        #     The modifier builds the grid in the x-y plane, so rotate the
        #     carrier by pi/2 about x (y -> z) and rescale by s. ---
        tri_carrier = Plane(name='FDTriGrid')
        tri_mod = TriangularGridModifier(grid_n=6, colors=['important', 'joker'], edge_thickness=0.01,
                                         show_fundamental_plane=True, dot_size=0.1)
        tri_carrier.add_mesh_modifier(type='NODES', node_modifier=tri_mod)
        tri_carrier.rotate(rotation_euler=[pi / 2, 0, 0], begin_time=t0, transition_time=0)
        tri_carrier.rescale(rescale=s, begin_time=t0, transition_time=0)
        t0 = 0.5 + tri_carrier.appear(begin_time=t0, transition_time=1.0)

        # --- play the mapping: the "l" dial folds every vertex into the cell ---
        l_node = get_geometry_node_from_modifier(tri_mod, "Lambda")
        t0 = ibpy.change_value(l_node, from_value=0, to_value=1, begin_time=t0, transition_time=4)
        t0 += 1.5

        # --- increase the grid size and shrink the edge_thickness and dot_size at the same time

        grid_size_node = get_geometry_node_from_modifier(tri_mod, "GridSize")
        dot_size_node = get_geometry_node_from_modifier(tri_mod, "DotRadius")
        edge_size_node = get_geometry_node_from_modifier(tri_mod, "EdgeThickness")
        toggle_plane = get_geometry_node_from_modifier(tri_mod, "FundamentalDomainFlag")

        cam_empty.move_to(target_location=(3.25, 0, 2.7), begin_time=t0, transition_time=1)
        t0 = 0.1 + ibpy.camera_zoom(lens=60, begin_time=t0, transition_time=1)
        ibpy.change_default_boolean(toggle_plane, from_value=False, to_value=True, begin_time=t0)

        ibpy.change_default_value(dot_size_node, from_value=0.1, to_value=0.04, begin_time=t0, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.04, to_value=0.03, begin_time=t0 + 1, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.03, to_value=0.02, begin_time=t0 + 2, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.02, to_value=0.01, begin_time=t0 + 3, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.01, to_value=0.006125, begin_time=t0 + 4,
                                  transition_time=6)
        ibpy.change_default_value(edge_size_node, from_value=0.01, to_value=0.000125, begin_time=t0,
                                  transition_time=3.5)

        t0 = 0.5 + ibpy.change_default_integer(grid_size_node, from_value=12, to_value=200, begin_time=t0,
                                               transition_time=10)

        cam_empty.move_to(target_location=(0, 0, 0), begin_time=t0, transition_time=1)
        t0 = 0.1 + ibpy.camera_zoom(lens=30, begin_time=t0, transition_time=1)

        # --- six trapezoids (same as the_background_fractal rosette) ----------
        #     Each copy i is pre-rotated by i*60° about the math origin so it
        #     arrives already in its slot.  The y-coordinate is pushed slightly
        #     toward the camera (-0.05 extra) so the polygons read over the grid.
        verts_math_fd = [(0.0, 0.0), (phi ** 2, 0.0), (phi ** 2 - 0.5, r32), (0.5, r32)]

        def _trap_verts_fd(i):
            c, sn = cos(i * pi / 3), sin(i * pi / 3)
            return [
                coord_loc + Vector((s * (c * xm - sn * ym), -0.05, s * (sn * xm + c * ym)))
                for xm, ym in verts_math_fd
            ]

        appear_dt = 0.5
        fd_traps = []
        for i in range(6):
            trap = Polygon(_trap_verts_fd(i), color='joker', name='FDRosetteTrap%d' % i)
            trap.add_mesh_modifier(type='NODES',
                                   node_modifier=MakeFrameModifier(color='important', thickness=0.02))
            trap.grow(begin_time=t0, transition_time=appear_dt)
            fd_traps.append(trap)
            t0 += appear_dt
        t0 += 0.5

        # --- slide each copy into the fundamental domain ----------------------
        #     Shifts derived so the tile centre lands inside the domain [0,1)²
        #     in (a, b) coordinates where p = a·u + b·v:
        #       i=1 → +u    i=2 → +u    i=3 → +(u+v)    i=4 → +v    i=5 → +v
        shifts_fd = [None, u, u, u + v, v, v]
        move_dt = 1.0
        for i in range(1, 6):
            fd_traps[i].move(direction=shifts_fd[i], begin_time=t0, transition_time=move_dt)
            t0 += 0.4
        t0 += move_dt + 0.5

        # --- four corner-quadrant fills: appear at origin, slide to corners ---
        #     BL quadrant: i=1, no shift
        #     BR quadrant: i=3, +u
        #     TL quadrant: i=0, +v
        #     TR quadrant: i=4, +(u+v)
        extra_rots_shifts = [(1, None), (3, u), (0, v), (4, u + v)]
        extra_traps = []
        for i, shift in extra_rots_shifts:
            trap = Polygon(_trap_verts_fd(i), color='joker', name='FDCornerTrap%d' % i)
            trap.add_mesh_modifier(type='NODES',
                                   node_modifier=MakeFrameModifier(color='important', thickness=0.02))
            trap.grow(begin_time=t0, transition_time=appear_dt)
            extra_traps.append((trap, shift))
            t0 += appear_dt
        t0 += 0.5

        for trap, shift in extra_traps:
            if shift is not None:
                trap.move(direction=shift, begin_time=t0, transition_time=move_dt)
                t0 += 0.4
        t0 += move_dt + 0.5

        self.t0 = t0 + 1

    def fundamental_domain_overlay(self):
        """Fold the triangular lattice into the fundamental domain.

        Re-introduces the same x-z coordinate system and the red lattice
        vectors u, v from :meth:`the_background_fractal`, overlays the
        triangular lattice (:class:`TriangularGridModifier`) and animates the
        modifier's ``l`` dial 0 -> 1.  That dial drives the modifier's
        back-translation, which maps every grid vertex into the (u, v)
        fundamental parallelogram.  The reduction formula is shown alongside.
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)

        # camera straight along +y, looking onto the x-z plane (as before)
        ibpy.set_camera_location(location=[0, -20, 0])
        cam_empty = EmptyCube(location=[0, 0, 0], name='FDCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=cam_empty)

        # --- transformation formula (important color) ---
        f_head = SimpleTexBObject(r"\text{Map $T$ into the domain}", color="important", text_size="large",
                                  aligned="left", location=Vector([6.64, 0, -3.1]), name='FDFormulaHead')
        f_p = SimpleTexBObject(r"\vec p=a\,\vec u+b\,\vec v", color="important", text_size="large",
                               aligned="left", location=Vector([6.64, 0, -4.2]), name='FDFormulaP')
        f_t = SimpleTexBObject(r"T(\vec p)=\{a\}\,\vec u+\{b\}\,\vec v", color="important", text_size="large",
                               aligned="left", location=Vector([6.64, 0, -5.2]), name='FDFormulaT')
        f_frac = SimpleTexBObject(r"\{x\}=x-\lfloor x\rfloor", color="important", text_size="large",
                                  aligned="left", location=Vector([6.64, 0, -6.3]), name='FDFormulaFrac')
        f_head.write(begin_time=t0, transition_time=0.5)
        f_p.write(begin_time=t0 + 0.4, transition_time=0.4)
        f_t.write(begin_time=t0 + 0.8, transition_time=0.4)
        t0 = f_frac.write(begin_time=t0 + 1.2, transition_time=0.4) + 0.4

        # --- triangular lattice in the x-z plane, scaled to match the system.
        #     The modifier builds the grid in the x-y plane, so rotate the
        #     carrier by pi/2 about x (y -> z) and rescale by s. ---
        tri_carrier = Plane(name='FDTriGrid')
        tri_mod = TriangularGridModifier(grid_n=6, colors=['important', 'joker'], edge_thickness=0.01,
                                         show_fundamental_plane=True, dot_size=0.1)
        tri_carrier.add_mesh_modifier(type='NODES', node_modifier=tri_mod)
        tri_carrier.rotate(rotation_euler=[pi / 2, 0, 0], begin_time=t0, transition_time=0)
        tri_carrier.rescale(rescale=1.5, begin_time=t0, transition_time=0)
        t0 = 0.5 + tri_carrier.appear(begin_time=t0, transition_time=1.0)

        # --- play the mapping: the "l" dial folds every vertex into the cell ---
        l_node = get_geometry_node_from_modifier(tri_mod, "Lambda")
        t0 = ibpy.change_value(l_node, from_value=0, to_value=1, begin_time=t0, transition_time=4)
        t0 += 1.5

        # --- increase the grid size and shrink the edge_thickness and dot_size at the same time

        grid_size_node = get_geometry_node_from_modifier(tri_mod, "GridSize")
        dot_size_node = get_geometry_node_from_modifier(tri_mod, "DotRadius")
        edge_size_node = get_geometry_node_from_modifier(tri_mod, "EdgeThickness")
        toggle_plane = get_geometry_node_from_modifier(tri_mod, "FundamentalDomainFlag")

        ibpy.change_default_boolean(toggle_plane, from_value=False, to_value=True, begin_time=t0)

        ibpy.change_default_value(dot_size_node, from_value=0.1, to_value=0.04, begin_time=t0, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.04, to_value=0.03, begin_time=t0 + 1, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.03, to_value=0.02, begin_time=t0 + 2, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.02, to_value=0.01, begin_time=t0 + 3, transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.01, to_value=0.006125, begin_time=t0 + 4,
                                  transition_time=6)
        ibpy.change_default_value(edge_size_node, from_value=0.01, to_value=0.000125, begin_time=t0,
                                  transition_time=3.5)

        t0 = 0.5 + ibpy.change_default_integer(grid_size_node, from_value=11, to_value=200, begin_time=t0,
                                               transition_time=10)

        self.t0 = t0 + 1

    def labbe_selinger(self):
        t0 = 0

        _setup_render()
        _setup_camera(distance=18)

        carrier = Plane(name='Carrier')
        carrier.appear(begin_time=0, transition_time=0)

        mod = LabbeSelingerModifier(roughness=0.1)
        carrier.add_mesh_modifier(type='NODES', node_modifier=mod)

        self.t0 = t0

    def labbe_selinger_cheap(self):
        t0 = 0

        _setup_render()
        _setup_camera(distance=18)

        carrier = Plane(name='Carrier')
        carrier.appear(begin_time=0, transition_time=0)
        mod = LabbeSelingerOptimizedModifier(roughness=0.1)
        carrier.add_mesh_modifier(type='NODES', node_modifier=mod)

        fractal = Plane(name="FractalPlane", u=[-200, 200], v=[-200, 200], color="hat_tile_fractal", roughness=0.1)
        fractal.appear(begin_time=t0, transition_time=0)

        self.t0 = t0

    def fractal_jump(self):
        """Four windows connect the Labbé--Selinger fractal to the tilings it
        parameterizes.

        Bottom: a strip of the background fractal with a red grid of dots on
        top (the sample points of one tiling).  Top left/right: the two large
        hat tilings that belong to the red and the blue grid position (live
        ``LabbeSelingerColorModifier`` carriers, framed red and blue).  A blue
        copy of the grid becomes visible and shifts slightly against the red
        one; the right tiling jumps accordingly.  A third window in the middle
        finally overlays the two tilings as a colour difference map
        (pre-rendered pictures; run ``fractal_jump_images.py`` once to create
        them in ``media/raster``).

        Alignment: the fractal shader samples the fractal in the fractal
        plane's *object* coordinates, and the plane's mesh is offset via its
        u/v ranges while the object itself stays at the origin, so object
        coordinates equal world coordinates.  With the shader inputs Scale=k
        and Shift=-F/k a world point w shows the fractal at uv=(w-F)/k.  The
        two dot carriers are scaled by k and moved to F, which puts their
        lattice point p at w=F+k*p, i.e. onto the fractal pixel uv=p --
        exactly the point whose region determines the hat orientation in the
        tiling windows (their carriers sample at object coordinates p, too).
        """

        RED_SHIFT = Vector([0.01, 0.01, 0])
        BLUE_SHIFT = Vector([0.015, 0.015, 0])
        GRID_SIZE = 100
        GRID_FILTER = 5000
        DOT_SIZE = 0.005
        WINDOW = 4.5  # backdrop edge length of an upper window (world units)
        CARRIER_SCALE = 0.05  # live tiling carrier object scale


        t0 = 0
        _setup_render()

        labels =[
            "Tiling for the red",
            "Difference",
            "Tiling for the blue",
        ]

        locations = [
            Vector([-4.6, -1,0]),
            Vector([0, -1,0]),
            Vector([4.6, -1,0]),

        ]

        # double distance and lens: same framing as distance=20 with the
        # default 50mm lens, but much flatter perspective in the side windows
        _setup_camera(distance=40)
        ibpy.camera_zoom(lens=100, begin_time=0, transition_time=0)

        # --- layout -------------------------------------------------------
        win = WINDOW  # backdrop edge length of the three upper windows
        left_center = Vector([-4.6, 1.5, 0])
        right_center = Vector([4.6, 1.5, 0])
        mid_center = Vector([0, 1.5, 0])
        f_center = Vector([0, -2.65, 0])  # centre of the fractal strip
        fw, fh = 9.0, 2.7  # size of the fractal strip
        k = 20  # zoom of the fractal window (1 lattice unit = k world units)
        s = CARRIER_SCALE# object scale of the tiling carriers
        red_shift = RED_SHIFT
        blue_shift = BLUE_SHIFT  # keep in sync with fractal_jump_images.py

        def window_backdrop(center, w, h, frame_color, name):
            hw, hh = w / 2, h / 2
            verts = [center + Vector([-hw, -hh, -0.05]), center + Vector([hw, -hh, -0.05]),
                     center + Vector([hw, hh, -0.05]), center + Vector([-hw, hh, -0.05])]
            poly = Polygon(verts, color='background', reordering=False, name=name)
            poly.add_mesh_modifier(type='NODES',
                                   node_modifier=MakeFrameModifier(color=frame_color, thickness=0.04))
            return poly

        left_backdrop = window_backdrop(left_center, win, win, 'red', 'LeftWindow')
        right_backdrop = window_backdrop(right_center, win, win, 'blue', 'RightWindow')
        bottom_backdrop = window_backdrop(f_center, fw + 0.1, fh + 0.1, 'text', 'FractalWindow')
        for backdrop in (left_backdrop, right_backdrop, bottom_backdrop):
            backdrop.grow(begin_time=t0, transition_time=1)
        t0 += 1

        # --- bottom window: the fractal strip ------------------------------
        fractal = Plane(name="FractalPlane",
                        u=[f_center.x - fw / 2, f_center.x + fw / 2],
                        v=[f_center.y - fh / 2, f_center.y + fh / 2],
                        color="hat_tile_fractal", roughness=0.1)
        fractal.grow(begin_time=t0, transition_time=0)
        fractal.move(direction=[0, 0, -0.01], begin_time=t0, transition_time=0)
        fractal_material = ibpy.get_material_at_slot(fractal, 0)
        for label, value in (("Scale", k),
                             ("ShiftX", -f_center.x / k),
                             ("ShiftY", -f_center.y / k)):
            node = get_node_from_shader(fractal_material, label=label)
            ibpy.change_value(node, from_value=value, to_value=value, begin_time=0, transition_time=0)

        # two dot-only carriers on top of the same fractal strip: the second
        # one shows no fractal background of its own, only its blue grid
        def dot_grid(name, dot_color, z):
            carrier = Plane(name=name)
            # dot_window crops the dots to the fractal strip (local coords)
            mod = LabbeSelingerColorModifier(color_scheme=0, shift=red_shift, grid_size=12,
                                             grid_filter=16, dot_color=dot_color,dot_radius=0.001,
                                             dot_window=(0, 0, (fw / 2 - 0.15) / k, (fh / 2 - 0.15) / k))
            carrier.add_mesh_modifier(type='NODES', node_modifier=mod)
            # dots only: collapse the hats, no wireframe, no extrusion
            for label in ("HatScale", "ExtrudeScale", "GridRadius"):
                node = get_geometry_node_from_modifier(mod, label=label)
                ibpy.change_default_value(node, from_value=0, to_value=0, begin_time=0, transition_time=0)
            wireframe = get_geometry_node_from_modifier(mod, label="Wireframe")
            ibpy.change_default_boolean(wireframe, from_value=False, to_value=False, begin_time=0)
            carrier.rescale(rescale=[k, k, k], begin_time=0, transition_time=0)
            carrier.move_to(target_location=Vector([f_center.x, f_center.y, z]),
                            begin_time=0, transition_time=0)
            carrier.appear(begin_time=0, transition_time=0)
            return mod

        red_grid = dot_grid("RedGrid", "red", 0.0)
        blue_grid = dot_grid("BlueGrid", "blue", 0.02)

        red_radius = get_geometry_node_from_modifier(red_grid, label="GridRadius")
        t0 = 0.5 + ibpy.change_default_value(red_radius, from_value=0, to_value=DOT_SIZE,
                                             begin_time=t0, transition_time=1)

        # --- the two big tiling windows ------------------------------------
        def tiling_window(name, center, shift):
            carrier = Plane(name=name)
            mod = LabbeSelingerColorModifier(color_scheme=0, shift=shift, grid_size=GRID_SIZE, grid_filter=0)
            carrier.add_mesh_modifier(type='NODES', node_modifier=mod)
            radius = get_geometry_node_from_modifier(mod, label="GridRadius")
            ibpy.change_default_value(radius, from_value=0, to_value=0, begin_time=0, transition_time=0)
            carrier.rescale(rescale=[s, s, s], begin_time=0, transition_time=0)
            carrier.move_to(target_location=center + Vector([0, 0, 0.05]),
                            begin_time=0, transition_time=0)
            carrier.appear(begin_time=0, transition_time=0)
            return mod

        left_mod = tiling_window("LeftTiling", left_center, red_shift)
        right_mod = tiling_window("RightTiling", right_center, red_shift)

        for mod in (left_mod, right_mod):
            grid_filter = get_geometry_node_from_modifier(mod, label="GridFilter")
            ibpy.change_default_integer(grid_filter, from_value=-1, to_value=GRID_FILTER,
                                        begin_time=t0, transition_time=4)

        start = t0
        for label, location in zip(labels, locations):
            lbl = SimpleTexBObject(r"\text{"+label+"}",aligned="center",rotation_euler=[0,0,0],location=location)
            start  = lbl.write(begin_time=start,transition_time=0.3)

        t0 += 4.5

        # --- the blue grid appears and shifts, the right tiling jumps ------
        blue_radius = get_geometry_node_from_modifier(blue_grid, label="GridRadius")
        t0 = 0.5 + ibpy.change_default_value(blue_radius, from_value=0, to_value=DOT_SIZE,
                                             begin_time=t0, transition_time=1)

        jump_time = 3
        for mod in (blue_grid, right_mod):
            shift_x = get_geometry_node_from_modifier(mod, label="ShiftX")
            shift_y = get_geometry_node_from_modifier(mod, label="ShiftY")
            ibpy.change_default_value(shift_x, from_value=red_shift.x, to_value=blue_shift.x,
                                      begin_time=t0, transition_time=jump_time)
            ibpy.change_default_value(shift_y, from_value=red_shift.y, to_value=blue_shift.y,
                                      begin_time=t0, transition_time=jump_time)
        t0 += jump_time + 1

        # --- third window: the red/blue snapshots slide from their live
        #     windows toward the centre, showing their colour difference
        #     wherever the two footprints overlap ---------------------------
        mid_backdrop = window_backdrop(mid_center, win, win, 'important', 'DiffWindow')
        t0 = mid_backdrop.grow(begin_time=t0, transition_time=1)

        images = ["hat_fractal_jump_red.png", "hat_fractal_jump_blue.png"]
        if all(os.path.exists(os.path.join(IMG_DIR, image)) for image in images):
            # one plane spans all three windows (object at the origin, mesh
            # offset via u/v -- same alignment trick as FractalPlane -- so its
            # "Object" coordinate equals world position for the shader)
            slide_material, red_cx, blue_cx = _make_jump_slide_material(
                images[0], images[1], win, left_center.y, name="JumpSlideMaterial")
            slide_plane = Plane(u=[left_center.x - win / 2, right_center.x + win / 2],
                                v=[left_center.y - win / 2, left_center.y + win / 2],
                                name="JumpSlidePlane")
            ibpy.set_material(slide_plane, slide_material, slot=0)
            # a hair above the live tiling carriers (z=0.05): at start it
            # exactly covers them -- the same tiling, just a snapshot --
            # before sliding away and revealing them again underneath
            slide_plane.move(direction=Vector([0, 0, 0.06]), begin_time=0, transition_time=0)
            slide_plane.appear(begin_time=t0, transition_time=0)

            slide_time = 2.5
            ibpy.change_value(red_cx, from_value=left_center.x, to_value=mid_center.x,
                              begin_time=t0 + 0.5, transition_time=slide_time)
            t0 = ibpy.change_value(blue_cx, from_value=right_center.x, to_value=mid_center.x,
                                   begin_time=t0 + 0.5, transition_time=slide_time)
        else:
            print("fractal_jump: pictures missing in " + IMG_DIR +
                  " -- run video_hat_tile/fractal_jump_images.py first")

        self.t0 = t0 + 2

    def simple_hat(self):
        t0 = 0
        _setup_render()
        _setup_camera(distance=18)

        colors = ["hat00", "hat01", "hat02", "hat03", "hat04", "hat05"]
        colors2 = ["hat10", "hat11", "hat12", "hat13", "hat14", "hat15"]

        for i in range(6):
            hat_tile = HatTile(rotation=i, location=Vector([-4.5 + 2.75 * cos(pi / 3 * i), 2.75 * sin(pi / 3 * i), 0]),
                               solid=0.02, color=colors[i])
            t0 = hat_tile.appear(begin_time=t0, transition_time=1)

            hat_tile = HatTile(rotation=i, reflection=True,
                               location=Vector([4.5 - 3 * cos(pi / 3 * i), 3 * sin(pi / 3 * i), 0]), solid=0.02,
                               color=colors2[i])
            t0 = hat_tile.appear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def hat_from_code(self):
        t0 = 0
        _setup_render()
        _setup_camera(distance=18)

        colors = ["hat00", "hat01", "hat02", "hat03", "hat04", "hat05"]
        colors2 = ["hat10", "hat11", "hat12", "hat13", "hat14", "hat15"]

        x = 0
        y = 0
        count = 0
        for reflection in [False, True]:
            for pivot in range(13):
                for rot in range(6):
                    location = (x + 3 * (count // 6) - 40 * reflection, y + 3 * (count % 6) - 20 * reflection)
                    if reflection:
                        color = colors2[rot]
                    else:
                        color = colors[rot]

                    hat_tile = HatTile.from_code(code=(location, rot, reflection, pivot), color=color)
                    t0 = hat_tile.appear(begin_time=t0, transition_time=0.1)
                    count += 1

        self.t0 = t0

    def show_orientations(self):
        """Show the twelve orientations of the hat tile: six unreflected hats
        (``hat00``..``hat05``) arranged on a left circle and their six
        reflected counterparts (``hat10``..``hat15``) on a right circle, each
        pair appearing together at mirror-symmetric positions.  The anchor
        point of every hat is highlighted by a red sphere."""
        t0 = 0.5
        _setup_render(hdri="cayley_interior_4k", transparent=True)
        _setup_camera(distance=18)

        colors = ["hat00", "hat01", "hat02", "hat03", "hat04", "hat05"]
        colors2 = ["hat10", "hat11", "hat12", "hat13", "hat14", "hat15"]

        radius = 2.25
        left_center = Vector([-5, 0, 0])
        right_center = Vector([5, 0, 0])

        # first all six unreflected hats on the left circle ...
        for i in range(6):
            angle = pi / 3 * i
            anchor = left_center + radius * Vector([cos(angle), sin(angle), 0])
            hat_tile = HatTile(rotation=i, location=anchor, hat_scale=1.5, solid=0.2,
                               color=colors[i])
            t0 = hat_tile.appear(begin_time=t0, transition_time=0.5)
            dot = Sphere(0.12, location=anchor, color='red', name='AnchorDot%d' % i)
            t0 = 0.25 + dot.grow(begin_time=t0, transition_time=0.3)

        # ... then their reflected partners at the mirrored positions on the right circle
        for i in range(6):
            angle = pi / 3 * i
            anti_anchor = right_center + radius * Vector([-cos(angle), sin(angle), 0])
            anti_hat_tile = HatTile(rotation=i, reflection=True, location=anti_anchor,
                                    hat_scale=1.5, solid=0.02, color=colors2[i])
            t0 = anti_hat_tile.appear(begin_time=t0, transition_time=0.5)
            anti_dot = Sphere(0.12, location=anti_anchor, color='red',
                              name='AntiAnchorDot%d' % i)
            t0 = 0.25 + anti_dot.grow(begin_time=t0, transition_time=0.3)

        self.t0 = t0 + 1

    def fundamental_domain_cover(self):
        """Show the triangular lattice: dots at vertices and wireframe edges."""
        _setup_render()
        _setup_camera(distance=18)

        carrier = Plane(name='DomainCarrier')
        carrier.appear(begin_time=0, transition_time=0)

        mod = FundamentalDomainCoverModifier()
        carrier.add_mesh_modifier(type='NODES', node_modifier=mod)

        self.t0 = 0

    def trapezoid_analyser(self):
        """Show the lattice's fundamental parallelogram and its 10 trapezoid regions."""
        _setup_render()
        cam_empty = _setup_camera(distance=16)
        ibpy.set_camera_location(location=(1.375, 0, 16))
        cam_empty.move_to(target_location=(1.375, 0.363, 0), begin_time=0, transition_time=0)
        ibpy.camera_zoom(lens=166, begin_time=0, transition_time=0)
        t0 = 0
        set_alpha_composition()
        # --- Background triangular grid (faded) ---
        grid_carrier = Plane(name='TrapezoidAnalyserCarrier')
        grid_carrier.appear(begin_time=t0, transition_time=1)
        analyser_modifier = AnalyserModifier(point_radius=0.005)
        grid_carrier.add_mesh_modifier(type='NODES', node_modifier=analyser_modifier)

        point_count_node = get_geometry_node_from_modifier(analyser_modifier, label="PointCount")
        step_node = get_geometry_node_from_modifier(analyser_modifier, label="Steps")

        t0 = 0.5 + ibpy.change_default_integer(point_count_node, from_value=0, to_value=100000, begin_time=t0,
                                               transition_time=10)
        t0 = 0.5 + ibpy.change_default_integer(step_node, from_value=0, to_value=10, begin_time=t0, transition_time=10)

        self.t0 = t0

    def substitution_vs_ifs(self):
        """Two-panel point-location demo driven by a single SubstitutionModifierFull.

        The modifier draws BOTH panels itself (left deflation, right inflation),
        so the scene only has to:

        * create the external ``Pin`` object that the modifier's Object Info
          nodes instance,
        * feed the pin's math-coordinate ``Position`` and the animated
          ``Iterations`` to the modifier, and
        * frame the result.

        Left panel (deflation): the forward subdivision refines only the
        sub-quadrilateral that holds the pin and stops once the pin lands in an
        exit face; meanwhile it records the transform address ``TranfoSequence``.
        Right panel (inflation): a single subdivision is shown as background, and
        the pin is walked out of the ribbon by the inverse maps decoded from that
        address.  Both panels advance in lock-step as ``Iterations`` grows.

        The whole tiling lives in math coordinates (left trapezoid x in
        [0, phi+1], right panel shifted by ``right_shift``); the carrier is then
        rescaled by ``S``.
        """
        t0 = 0
        _setup_render(engine="CYCLES")

        S = 2.5  # world units per math unit (carrier scale)
        right_shift = 2.75  # math-unit x offset of the right panel (modifier default)
        P = (1.481, 0.373, 0.0)  # the random interior point, in math coordinates
        max_iter = 6  # final subdivision depth (enough to resolve the pin)

        # framing: math x in [0, right_shift + (phi+1)], y in [0, r3/2]
        span_x = right_shift + (phi + 1)
        cx = 0.5 * S * span_x
        empty = _setup_standard_camera(distance=12, shift_x=cx)

        # --- external pin object: geometry source for the modifier's Object Info
        #     nodes (the modifier instances + places it; it is not shown on its
        #     own, so it is created but never made to appear) ---
        pin = Pin(name='Pin', colors=["important"])
        pin.appear(alpha=0, begin_time=0, transition_time=0)
        # activate this line for the final render to get rid of the source pin
        pin.toggle_hide(begin_time=0.1)

        # --- single carrier; the modifier renders both panels ---
        carrier = Plane(name='PointLocationPanels')
        carrier.rotate(rotation_euler=[pi / 3, 0, 0], begin_time=t0, transition_time=0)
        carrier.appear(begin_time=0, transition_time=0)

        mod = SubstitutionModifierFull(iterations=0, position=P,
                                       pin_object=pin.needle,
                                       pin_head_object=pin.head,
                                       right_shift=right_shift)
        carrier.add_mesh_modifier(type='NODES', node_modifier=mod)
        carrier.rescale(rescale=S, begin_time=0, transition_time=0)

        # the modifier exposes ``Position`` and ``Iterations`` as group inputs;
        # set them on the modifier instance.
        ibpy.set_socket_data_for_geometry_node_modifier(
            carrier, {"Position": Vector(P), "Iterations": 0})
        gnmod = ibpy.get_geometry_nodes_modifier(carrier)
        iter_socket = ibpy.get_socket_names_from_modifier(gnmod)["Iterations"]

        # --- panel titles ---
        left_title = SimpleTexBObject(r"\text{Subdivide ($T_1$, $T_2$, $T_3$, $P_1$)}", color="example",
                                      text_size="large", aligned="center",
                                      location=Vector([3, 3.5, 4]),
                                      name='PLLeftTitle')
        right_title = SimpleTexBObject(r"\text{IFS ($T_1^{-1}$, $T_2^{-1}$, $T_3^{-1}$, $P_1^{-1}$)}", color="example",
                                       text_size="large", aligned="center",
                                       location=Vector([11.4, 3.5, 4]),
                                       name='PLRightTitle')
        left_title.write(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + right_title.write(begin_time=t0, transition_time=0.5)

        # --- drive both panels in lock-step by deepening the subdivision: the
        #     left refines around the pin, the right re-decodes the (now longer)
        #     address and walks the pin one inverse map further out.  ``Iterations``
        #     is a geometry-node group input, so keyframe the modifier socket
        #     directly (stepwise jumps over a single frame). ---
        beat = 1.5
        old_text = None
        trafos = ["P_1^{-1}", "P_1^{-1}", "T_3^{-1}", "P_1^{-1}"]
        for step in range(0, max_iter):
            frame = int(t0 * FRAME_RATE)
            gnmod[iter_socket] = step
            if old_text is not None:
                old_text.disappear(begin_time=t0, transition_time=0.1)
            gnmod.keyframe_insert(data_path='["%s"]' % iter_socket, frame=frame)
            gnmod[iter_socket] = step
            gnmod.keyframe_insert(data_path='["%s"]' % iter_socket, frame=frame + 1)
            if 0 < step < 5:
                trafo_bob = SimpleTexBObject(trafos[step - 1], color="example", text_size="large", location=[10, 0, -2])
                trafo_bob.write(begin_time=t0, transition_time=0.5)
                old_text = trafo_bob
            t0 += beat

        empty.move_to(target_location=[3.7, 0.34, 0.78], begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.camera_zoom(lens=1965, begin_time=t0, transition_time=3)

        empty.move_to(target_location=Vector([cx, 0, 0]), begin_time=t0 + 2, transition_time=1)
        t0 = 0.5 + ibpy.camera_zoom(lens=30, begin_time=t0, transition_time=3)
        self.t0 = t0 + 1

    def substitution_intro(self):
        """Introduce the trapezoid substitution rule (cf. SubstitutionModifier).

        A coordinate system carries the fundamental trapezoid
        (0,0) -> (phi+1,0) -> (phi+1/2, sqrt3/2) -> (1/2, sqrt3/2) with its
        vertices labelled by their coordinate pairs (text colour).  Three copies
        are scaled and rotated into place following the trapezoid transformations
        of the SubstitutionModifier (recoloured joker), and the remaining gap is
        filled by a custom1-coloured parallelogram (the fourth transformation).

        Camera looks along +y onto the x-z plane, so math (x, y) -> world
        (s*x, 0, s*y).
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        set_alpha_composition()
        s = 2.0  # blender units per math unit
        coord_loc = Vector([0, 0, 0])

        def to_loc(x, y):
            """Math point (x, y) -> world location in the x-z plane."""
            return coord_loc + Vector((s * x, 0.0, s * y))

        # --- camera straight along +y, centred on the trapezoid ---
        center_math = (3, r3 / 4.0)
        cam_empty = EmptyCube(location=to_loc(*center_math), name='SubIntroCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_location(location=to_loc(*center_math) + Vector([0, -16, 0]))
        ibpy.set_camera_view_to(target=cam_empty)

        # --- title ---
        title = SimpleTexBObject(r"\text{Substitutions}", color="example",
                                 text_size="Large", aligned="center",
                                 location=to_loc(*center_math) + Vector([0, 0, 5.0]),
                                 name='SubIntroTitle')
        t0 = 0.5 + title.write(begin_time=t0, transition_time=0.5)

        # --- coordinate system in the x-z plane ---
        x_max, y_max = 3.0, 2.0
        coords = CoordinateSystem2(
            dimension=2, location=coord_loc, radii=[0.02] * 2,
            lengths=[s * (x_max), s * (y_max)], domains=[[0, x_max], [0, y_max]],
            n_tics=[int(x_max), int(y_max)], tic_labels=['AUTO', 'AUTO'], tic_label_digits=[0, 0],
            include_zeros=[False, False], colors=['drawing', 'drawing'],
            name='SubIntroCoords',
            axes_labels={'x': Vector([0.25, 0, 2.5 * s]), 'y': Vector([-0.5, 0, 2 * s])})
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=1.5)

        # --- the fundamental trapezoid (canonical xi = (1/2, sqrt3/2)) ---
        r32 = r3 / 2
        trap_math = [(0, 0), (phi + 1, 0), (phi + 0.5, r32), (0.5, r32)]
        trap_world = [to_loc(x, y) for (x, y) in trap_math]
        trapez = Polygon(trap_world, color="text", name='SubIntroTrapez', reordering=False)
        trapez.add_mesh_modifier(type='NODES',
                                 node_modifier=MakeFrameModifier(color='important', thickness=0.02))
        t0 = 0.5 + trapez.grow(begin_time=t0, transition_time=1.0)

        # --- label the vertices with their coordinate pairs (text colour) ---
        labels_tex = [
            r"A=\left(0,\,0\right)",
            r"B=\left(\varphi+1,\,0\right)",
            r"C=\left(\varphi+\tfrac{1}{2},\,\tfrac{\sqrt{3}}{2}\right)",
            r"D=\left(\tfrac{1}{2},\,\tfrac{\sqrt{3}}{2}\right)",
        ]
        label_offsets = [
            Vector([-0.8, 0, -0.7]),  # (0,0)            lower-left
            Vector([0.9, 0, -0.7]),  # (phi+1,0)        lower-right
            Vector([1.6, 0, 0.6]),  # (phi+1/2, r3/2)  upper-right
            Vector([0.2, 0, 0.7]),  # (1/2, r3/2)      upper-left
        ]
        dots, labels = [], []
        for i, (mx, my) in enumerate(trap_math):
            dot = Sphere(0.12, location=to_loc(mx, my), color='important', name='SubIntroV%d' % i)
            dot.grow(begin_time=t0, transition_time=0.4)
            dots.append(dot)
            lbl = SimpleTexBObject(labels_tex[i], color="text", text_size="normal", aligned="center",
                                   location=to_loc(mx, my) + label_offsets[i], name='SubIntroLabel%d' % i)
            lbl.write(begin_time=t0, transition_time=0.8)
            labels.append(lbl)
            t0 += 1.0
        t0 += 0.5

        # --- substitution transforms (read from SubstitutionModifier) ---
        # vertex roles: v1=a, v2=b, v3=c, v4=d; scale factor k=(phi-1)/phi.
        # Each piece = pivot + k*R(theta)*(p-pivot) + trans.  The Polygon object
        # origin is world (0,0,0), so rescale/rotate act about it and the final
        # object location bakes in the pivot:  L = pivot + trans - k*R(theta)*pivot.
        k = (phi - 1) / phi
        a, b, c, d = trap_math
        v1, v2, v3, v4 = a, b, c, d

        def rot2(theta, vx, vy):
            cc, ss = cos(theta), sin(theta)
            return (cc * vx - ss * vy, ss * vx + cc * vy)

        def xf(p, theta, trans, pivot):
            rx, ry = rot2(theta, p[0] - pivot[0], p[1] - pivot[1])
            return (pivot[0] + k * rx + trans[0], pivot[1] + k * ry + trans[1])

        def sub(u, w):
            return (u[0] - w[0], u[1] - w[1])

        def scl(u, f):
            return (u[0] * f, u[1] * f)

        def _show_text(text, line, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
            bob = SimpleTexBObject(text, location=Vector([-3, 0, line]), aligned="left", color="joker")
            bob.write(begin_time=begin_time, transition_time=transition_time)

        # Transf1, Transf2, Transf3 -> three trapezoids.
        # All three pivot about vertex B (shift=1 puts B first in the list).
        # The translation is given explicitly, but described in the text purely
        # by trapezoid vertex vectors:
        #   Transf1: rotate -120 about B, scale, then translate by -AB  (B -> A)
        #   Transf2: rotate  180 about B, scale, then translate by  BD  (B -> D)
        #   Transf3: rotate    0 about B, scale, no translation        (B fixed)
        trap_transforms = [
            (-2.0 / 3.0 * pi, (-phi - 1, 0), 1),  # Transf1
            (pi, (-phi - 0.5, r32), 1),  # Transf2  (-phi-1/2, r3/2) = D-B
            (0.0, (0, 0), 1),  # Transf3
        ]
        texts = [
            [r"R_B\left(-120^\circ\right)", r"S_B\left(\tfrac{1}{\varphi^2}\right)", r"T_{-\overrightarrow{AB}}"],
            [r"R_B\left(180^\circ\right)", r"S_B\left(\tfrac{1}{\varphi^2}\right)", r"T_{\overrightarrow{BD}}"],
            [r"R_B\left(0^\circ\right)", r"S_B\left(\tfrac{1}{\varphi^2}\right)", r"T_{\vec{0}}"],
        ]

        def _shift(vertices, n):
            vs = [vertices[(i + n) % 4] for i in range(4)]
            return vs

        dt = 1.5
        for j, ((theta, trans, shift), text) in enumerate(zip(trap_transforms, texts)):
            copy = Polygon(_shift(trap_world, shift), color="text", name='SubIntroCopy%d' % j, reordering=False)
            copy.add_mesh_modifier(type='NODES',
                                   node_modifier=MakeFrameModifier(color='important', thickness=0.02))
            copy.grow(begin_time=t0, transition_time=0)
            copy.change_color(new_color="joker", begin_time=t0, transition_time=dt)
            _show_text(text[0], line=5 - 0.8 * (4 * j), begin_time=t0, transition_time=dt)
            t0 = 0.1 + copy.rotate(rotation_euler=[0, -theta, 0], begin_time=t0, transition_time=dt)
            _show_text(text[1], line=5 - 0.8 * (4 * j + 1), begin_time=t0, transition_time=dt)
            t0 = 0.1 + copy.rescale(rescale=k, begin_time=t0 + 0.1, transition_time=dt)
            _show_text(text[2], line=5 - 0.8 * (4 * j + 2), begin_time=t0, transition_time=dt)
            t0 = 0.1 + copy.move(direction=to_loc(*trans), begin_time=t0, transition_time=dt)

            t0 = 0.5 + t0 + dt

        coord_loc2 = Vector([8, 0, 0])

        def to_loc2(x, y):
            """Math point -> world location in the second coordinate system."""
            return coord_loc2 + Vector((s * x, 0.0, s * y))

        def to_dir(x, y):
            """World translation for a math direction (no coordinate offset)."""
            return Vector((s * x, 0.0, s * y))

        coords2 = CoordinateSystem2(
            dimension=2, location=coord_loc2, radii=[0.02] * 2,
            lengths=[s * (x_max), s * (y_max)], domains=[[0, x_max], [0, y_max]],
            n_tics=[int(x_max), int(y_max)], tic_labels=['AUTO', 'AUTO'], tic_label_digits=[0, 0],
            include_zeros=[False, False], colors=['drawing', 'drawing'],
            name='SubIntroCoords2',
            axes_labels={'x': Vector([0.25, 0, 2.5 * s]), 'y': Vector([-0.5, 0, 2 * s])})
        t0 = 0.5 + coords2.appear(begin_time=t0, transition_time=1.5)

        # the parallelogram: A -> (1, 0), the other three vertices unchanged
        para_math = [(1, 0), (phi + 1, 0), (phi + 0.5, r32), (0.5, r32)]
        para_world2 = [to_loc2(x, y) for (x, y) in para_math]
        para_tile = Polygon(para_world2, color="text", name='SubIntroParaTile', reordering=False)
        para_tile.add_mesh_modifier(type='NODES',
                                    node_modifier=MakeFrameModifier(color='important', thickness=0.02))
        t0 = 0.5 + para_tile.grow(begin_time=t0, transition_time=1.0)

        # --- label the parallelogram vertices (A' moved, B/C/D unchanged) ---
        para_labels_tex = [
            r"A'=\left(1,\,0\right)",
            r"B=\left(\varphi+1,\,0\right)",
            r"C=\left(\varphi+\tfrac{1}{2},\,\tfrac{\sqrt{3}}{2}\right)",
            r"D=\left(\tfrac{1}{2},\,\tfrac{\sqrt{3}}{2}\right)",
        ]
        para_label_offsets = [
            Vector([-0.3, 0, -0.7]),  # A'=(1,0)         lower
            Vector([0.9, 0, -0.7]),  # B=(phi+1,0)      lower-right
            Vector([1.6, 0, 0.6]),  # C                upper-right
            Vector([0.1, 0, 0.7]),  # D                upper-left
        ]
        for i, (mx, my) in enumerate(para_math):
            dot = Sphere(0.12, location=to_loc2(mx, my), color='important', name='SubIntroParaV%d' % i)
            dot.grow(begin_time=t0, transition_time=0.4)
            lbl = SimpleTexBObject(para_labels_tex[i], color="text", text_size="normal", aligned="center",
                                   location=to_loc2(mx, my) + para_label_offsets[i], name='SubIntroParaLabel%d' % i)
            lbl.write(begin_time=t0, transition_time=0.8)
            t0 += 1.0
        t0 += 0.5

        # transformation (pivot B): R_B(-60), S_B(1/phi^2), translate by
        # -AB/phi^2 = (-1, 0).  Transformation text on the right of the screen.
        def _show_text_right(text, line, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
            bob = SimpleTexBObject(text, location=Vector([13, 0, line]), aligned="left", color="custom1")
            bob.write(begin_time=begin_time, transition_time=transition_time)

        para_texts = [r"R_B\left(-60^\circ\right)",
                      r"S_B\left(\tfrac{1}{\varphi^2}\right)",
                      r"T_{-\overrightarrow{AB}/\varphi^2}"]

        para_copy = Polygon(_shift(para_world2, 1), color="text", name='SubIntroParaCopy', reordering=False)
        para_copy.add_mesh_modifier(type='NODES',
                                    node_modifier=MakeFrameModifier(color='important', thickness=0.02))
        para_copy.grow(begin_time=t0, transition_time=0)
        para_copy.change_color(new_color="custom1", begin_time=t0, transition_time=dt)
        _show_text_right(para_texts[0], line=5 - 0.8 * 0, begin_time=t0, transition_time=dt)
        t0 = 0.1 + para_copy.rotate(rotation_euler=[0, pi / 3, 0], begin_time=t0, transition_time=dt)
        _show_text_right(para_texts[1], line=5 - 0.8 * 1, begin_time=t0, transition_time=dt)
        t0 = 0.1 + para_copy.rescale(rescale=k, begin_time=t0 + 0.1, transition_time=dt)
        _show_text_right(para_texts[2], line=5 - 0.8 * 2, begin_time=t0, transition_time=dt)
        t0 = 0.1 + para_copy.move(direction=to_dir(-1, 0), begin_time=t0, transition_time=dt)
        t0 = 0.5 + t0 + dt

        # --- Transf4 -> fill the gap with a custom1 parallelogram, and at the
        #     same time grow the parallelogram's two sub-trapezoids (which reuse
        #     transforms 2 & 3, pivot B) inside the second coordinate system ---
        fill_t = t0
        # gap-filling parallelogram inside the trapezoid (first system)
        aP = (v4[0] + v2[0] - v3[0], v4[1] + v2[1] - v3[1])
        para_base = [aP, b, c, d]
        para_final = [xf(p, -pi / 3.0, scl(sub(v1, v2), 1.0 / phi ** 2), v2) for p in para_base]
        para_world = [to_loc(x, y) for (x, y) in para_final]
        para = Polygon(para_world, color="custom1", name='SubIntroParallelogram')
        para.add_mesh_modifier(type='NODES',
                               node_modifier=MakeFrameModifier(color='important', thickness=0.02))
        para.grow(begin_time=fill_t, transition_time=1.0)

        # the parallelogram's two sub-trapezoids (transforms 2 & 3) in 2nd system.
        # The source must be a trapezoid (A' = (1,0) moved back to A = (0,0)),
        # otherwise the transformed pieces stay parallelograms.
        for tag, (theta, trans) in [("2", (pi, (-phi - 0.5, r32))), ("3", (0.0, (0, 0)))]:
            piece_world = [to_loc2(*xf(p, theta, trans, v2)) for p in trap_math]
            tri = Polygon(piece_world, color="joker", name='SubIntroParaTrap%s' % tag, reordering=False)
            tri.add_mesh_modifier(type='NODES',
                                  node_modifier=MakeFrameModifier(color='important', thickness=0.02))
            tri.grow(begin_time=fill_t, transition_time=1.0)
        t0 = 0.5 + fill_t + 1.0

        # --- column headings on top of each triple of transformations,
        #     same colour/size as the title heading ---
        heading_kw = dict(color="example", text_size="Large", aligned="left")
        SimpleTexBObject(r"T_1:", location=Vector([-3, 0, 5.7]), name='SubIntroHeadT1',
                         **heading_kw).write(begin_time=t0, transition_time=0.5)
        SimpleTexBObject(r"T_2:", location=Vector([-3, 0, 2.5]), name='SubIntroHeadT2',
                         **heading_kw).write(begin_time=t0, transition_time=0.5)
        SimpleTexBObject(r"T_3:", location=Vector([-3, 0, -0.7]), name='SubIntroHeadT3',
                         **heading_kw).write(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + SimpleTexBObject(r"P_1:", location=Vector([13, 0, 5.7]), name='SubIntroHeadP1',
                                    **heading_kw).write(begin_time=t0, transition_time=0.5)

        # --- title ---
        colors = flatten([["text"] * 32, ["joker"] * 10, ["text"] * 3, ["custom1"]])
        outle = SimpleTexBObject(r"\text{Repeat substitutions for the smaller trapezoids and parallelograms}",
                                 color=colors,
                                 text_size="Large", aligned="left",
                                 location=Vector([-3.4, 0, -3.9]),
                                 name='Outro')
        t0 = 0.5 + outle.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def substitution(self):
        """Animate a trapezoid splitting into its substitution sub-pieces."""
        _setup_render()
        _setup_camera(distance=3)
        set_alpha_composition()
        t0 = 0

        base_center = Vector(((phi + 1) / 2, -0.125, 0))
        ibpy.camera_move(shift=base_center, begin_time=t0, transition_time=0)
        empty = EmptyCube(location=base_center)
        ibpy.set_camera_view_to(empty)

        carrier = Plane(name='SubsitutionCarrierTrap')
        mod = SubstitutionModifier(colors=["joker", "custom1", "background"])
        carrier.add_mesh_modifier(type='NODES', node_modifier=mod)

        carrier2 = Plane(name='SubsitutionCarrierTrap')
        carrier2.move(direction=[0, -1, 0], begin_time=t0, transition_time=0)
        mod2 = SubstitutionModifier(colors=["joker", "custom1", "background"])
        carrier2.add_mesh_modifier(type='NODES', node_modifier=mod2)

        toggle = get_geometry_node_from_modifier(mod2, "TrapezParallelogramToggle")
        ibpy.change_default_boolean(toggle, from_value=True, to_value=False, begin_time=t0)

        level = get_geometry_node_from_modifier(mod, label="Iterations")
        level2 = get_geometry_node_from_modifier(mod2, label="Iterations")

        levels = [level, level2]
        [ibpy.change_default_integer(level, from_value=0, to_value=0, begin_time=t0, transition_time=0) for level in
         levels]

        carrier.appear(begin_time=0, transition_time=1)
        t0 = 0.5 + carrier2.appear(begin_time=0, transition_time=1)

        for l in range(1, 9):
            [ibpy.change_default_integer(level, from_value=l - 1, to_value=l, begin_time=t0, transition_time=1) for
             level in levels]
            t0 += 1.5

        t0 = 0.5 + carrier2.move(direction=[0.5, 0, 0], begin_time=t0, transition_time=0.5)

        fractal = Plane(u=[-0.5, 0.5], v=[-0.5, 0.5], name="LabbeSelingerCarrier", color="hat_tile_fractal")
        fractal.move(direction=[0.25, -0.6, 0], begin_time=t0, transition_time=0)
        scale_node = get_node_from_shader(ibpy.get_material_at_slot(fractal, 0), label="Scale")
        ibpy.change_default_value(scale_node, from_value=1, to_value=0.25, begin_time=t0, transition_time=0)

        fractal.appear(begin_time=t0, transition_time=1)
        self.t0 = t0

    def substitution_hat_overlay(self):
        t0 = 0
        self.t0 = t0

        t0 = 0
        _setup_render()
        # the camera always points at the tracked empty; its world position is
        # driven by smooth Bezier fly-paths instead of straight waypoint hops.
        camera_empty = _setup_camera(distance=20)
        ibpy.set_camera_location(location=[0, -20, 0])
        ibpy.set_camera_lens(lens=30)
        set_alpha_composition()

        counts = [1, 8, 55, 377, 2584, 17711, 121393, 832040]

        for i in range(len(counts)):
            if i == 0:
                text = SimpleTexBObject(r"1\text{ hat}", color="example", text_size="Large", alinged="left")
            else:
                text = SimpleTexBObject(str(counts[i]) + r"\text{ hats}", color="example", text_size="Large",
                                        alinged="left")

            t0 = 0.8 + text.write(begin_time=t0, transition_time=0.1)
            t0 = text.disappear(begin_time=t0, transition_time=0.1)

    def substitution_hat(self):
        """Hat tiling built by substitution (H7H8Supertiles.nb).
            Grow cluster to the limit
            0 1 hat
            1 8 hats
            2 55 hats
            3 377 hats
            4 2584 hats
            5 17711 hats
            6 121393 hats
            7 832040 hats

            conseccutive numbers satisfy (x+y)**2/(1+x*y)=9


        """
        t0 = 0
        _setup_render()
        # the camera always points at the tracked empty; its world position is
        # driven by smooth Bezier fly-paths instead of straight waypoint hops.
        camera_empty = _setup_camera(0)
        ibpy.set_camera_lens(lens=30, clip_end=100000)
        set_alpha_composition()

        # helper: cumulative-chord-length offset_factor of each control point
        def _offsets(points):
            acc = [0.0]
            for j in range(1, len(points)):
                acc.append(acc[-1] + (points[j] - points[j - 1]).length)
            tot = acc[-1]
            return [a / tot for a in acc]

        # === PHASE 1: growth zoom-out ==========================================
        # One control point per substitution level, climbing out along the
        # diagonal as the cluster explodes toward its full 832040-hat extent.
        #   index 0 -> levels 0/1 (1 & 8 hats) ... index 6 -> level 7
        growth_points = [
            Vector([9, 0, 9]),  # 1 / 8 hats
            Vector([25, 0, 18]),  # 55 hats
            Vector([80, 0, 50]),  # 377 hats
            Vector([214, 47, 115]),  # 2584 hats
            Vector([555, 130, 285]),  # 17711 hats
            Vector([1342, 321, 989]),  # 121393 hats
            Vector([3433, 900, 2950]),  # 832040 hats (full extent)
        ]

        # === PHASE 2: fly-over of the level-7 cluster ==========================
        # The 832040-hat cluster (substitution algorithm in SubstitutionRules2.nb)
        # lies in the z=0 plane, grown from the seed hat at the origin toward
        # -x / +y:   x in [-4089, 1221], y in [-2509, 2914], centroid ~ (-1135, 293).
        # Explicit camera path: drop from the highest (growth-end) overview point,
        # skim low across the cluster, and end past the far edge.
        fly_points = [
            growth_points[-1],  # highest point (832040-hat overview)
            Vector([2200, -700, 800]),  # mid-descent: keeps the Bezier above the surface
            Vector([500, -2500, 250]),
            Vector([-500, 500, 250]),
            Vector([-4100, 500, 250]),
        ]
        # Where the camera looks at each fly point (z=0 plane, ~32 deg down into
        # the cluster).  On the last segment the camera overtakes its aim point,
        # so the view swings forward -> straight down -> looking back *before*
        # reaching the final point, avoiding a stare into the empty far edge.
        aim_points = [
            Vector([0, 0, 3]),  # overview aim (continues from the growth)
            Vector([120, -870, 0]),  # mid-descent aim (matches the descent point)
            Vector([247, -1741, 0]),
            Vector([-1300, 500, 0]),
            Vector([-3300, 500, 0]),  # camera flies past this -> looks back
        ]

        # Build the two invisible guide curves the camera rides (never grown, so
        # they don't render).  make_pieces=False keeps the widely-spaced control
        # points as a single continuous spline.
        cam_path = BezierDataCurve(data=[Vector(p) for p in growth_points],
                                   name='SubstitutionGrowthPath', make_pieces=False)
        fly_path = BezierDataCurve(data=[Vector(p) for p in fly_points],
                                   name='SubstitutionFlyPath', make_pieces=False)
        ibpy.set_camera_follow(cam_path)
        ibpy.set_camera_follow(fly_path)
        # the paths provide the absolute position; orientation stays locked on the
        # tracked empty, so suppress curve-tangent banking on both
        for c in ibpy.get_camera().constraints:
            if c.type == 'FOLLOW_PATH':
                c.use_curve_follow = False
        # park the camera at the start of the growth path from the very first frame
        ibpy.camera_follow(cam_path, initial_value=0, final_value=0, begin_time=0, transition_time=0)

        carrier = Plane(name='SubstitutionCarrier')
        mod = HatTileSubstitutionModifier(level=0, color_scheme=1, extrude_scale=1, hat_scale=0.975)
        carrier.add_mesh_modifier(type='NODES', node_modifier=mod)
        t0 = 0.5 + carrier.appear(begin_time=0, transition_time=1)

        # lift the look-target so the off-origin level-1 cluster stays in frame
        t0 = 0.5 + camera_empty.move(direction=[0, 0, 3], begin_time=t0, transition_time=1)

        level_node = get_geometry_node_from_modifier(mod, "Level")

        # --- growth: one level per beat; constant time per segment makes the
        #     camera accelerate as the cluster spreads, selling the growth ------
        g_off = _offsets(growth_points)
        t0 = 0.5 + ibpy.change_default_integer(level_node, from_value=0, to_value=1,
                                               begin_time=t0, transition_time=0)
        seg_time = 1.25
        for i in range(1, 7):
            t0 = ibpy.camera_follow(cam_path, initial_value=g_off[i - 1], final_value=g_off[i],
                                    begin_time=t0, transition_time=seg_time)
            t0 = 0.5 + ibpy.change_default_integer(level_node, from_value=i, to_value=i + 1,
                                                   begin_time=t0, transition_time=0)

        # --- fly-over: descend from the overview point, skim low across the
        #     cluster, then overtake the aim point so the view turns back -------
        f_off = _offsets(fly_points)
        # segment durations: descent (split in two), skim, turn-back-and-look-back
        seg_times = [1.5, 1.5, 3.0, 4.0]
        for k in range(1, len(fly_points)):
            dt = seg_times[k - 1]
            t0 = ibpy.camera_follow(fly_path, initial_value=f_off[k - 1], final_value=f_off[k],
                                    begin_time=t0, transition_time=dt)
            camera_empty.move_to(target_location=aim_points[k], begin_time=t0 - dt, transition_time=dt)
        ibpy.camera_zoom(lens=18, begin_time=t0 - 4, transition_time=3)
        self.t0 = t0

    def substitution_explainer(self):
        """Assemble H7Tile and H8Tile from copies of the H2Tile/H1Tile seeds.

        Each tile is an individually extruded, keyframed mesh (geometry nodes
        can't keyframe the per-tile motion), coloured blue (direct) / yellow
        (reflected).  kloppenheim HDRI lights a transparent background; the
        camera is tilted so the extrusion is visible.
        """
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        _setup_tilted_camera(center=(0, 0), distance=61, tilt=0)

        t0 = 0

        floor = Floor(u=[-30, 30], v=[-19, 15], color="checker", coords="Object")
        floor.move(direction=[0, 0, -0.1], begin_time=t0, transition_time=0)
        t0 = 0.5 + floor.grow(begin_time=t0, transition_time=1.0)

        left = SimpleTexBObject(r"\text{Base Cluster}", color="example", text_size="Huge", aligned="center",
                                location=Vector([-12, 18, 0]))
        left.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0.5)
        left.write(begin_time=t0, transition_time=0.5)

        right = SimpleTexBObject(r"\text{Addon Cluster}", color="example", text_size="Huge", aligned="center",
                                 location=Vector([12, 18, 0]))
        right.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + right.write(begin_time=t0, transition_time=0.5)

        t0 = play_substitution(H2_TILE, H1_TILE, t0=t0, scale=1, solid=0.4, rescale=0.5)

        self.t0 = t0

    def substitution_overlay(self):
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        ibpy.set_camera_location(location=(0, -20, 0))

        # ref = Plane(u=[-1.92,1.92],v=[-1.08,1.080],color="image",src="substitution_ref.png",rotation_euler=[pi/2,0,0])
        # ref.appear(begin_time=0,transition_time=0)
        # ref.rescale(rescale=6.26,begin_time=0,transition_time=0)
        t0 = 0

        line = SimpleTexBObject(r"5\times 1 +2 =7", text_size="huge", aligned="left", color="blue",
                                location=[-8.25, 0, -5.75])
        t0 = 0.5 + line.write(begin_time=t0, transition_time=0.5)

        line2 = SimpleTexBObject(r"6\times 1 +2 =8", text_size="huge", aligned="left", color="blue",
                                 location=[-1, 0, -5.75])
        t0 = 0.5 + line2.write(begin_time=t0, transition_time=0.5)

        line3 = SimpleTexBObject(r"7+1=8", text_size="huge", aligned="left", color="blue",
                                 location=[6.05, 0, -5.75])
        t0 = 0.5 + line3.write(begin_time=t0, transition_time=0.5)

        lines = [line, line2, line3]

        for i, line in enumerate(lines):
            line.disappear(begin_time=t0 + 0.1 * i, transition_time=0.5)

        t0 += 1

        line = SimpleTexBObject(r"5\times 8 +7 =47", text_size="Large", aligned="left", color="blue",
                                location=[-8.25, 0, -6.1])
        t0 = 0.5 + line.write(begin_time=t0, transition_time=0.5)

        line2 = SimpleTexBObject(r"6\times 8 +7 =55", text_size="Large", aligned="left", color="blue",
                                 location=[0.3, 0, -6.1])
        t0 = 0.5 + line2.write(begin_time=t0, transition_time=0.5)

        line3 = SimpleTexBObject(r"47+8=55", text_size="Large", aligned="left", color="blue",
                                 location=[5.75, 0, -6.1])
        t0 = 0.5 + line3.write(begin_time=t0, transition_time=0.5)

        lines = [line, line2, line3]

        for i, line in enumerate(lines):
            line.disappear(begin_time=t0 + 0.1 * i, transition_time=0.5)

        t0 += 1

        line = SimpleTexBObject(r"5\times 55 +47 =322", text_size="Large", aligned="left", color="blue",
                                location=[-8.25, 0, -6.1])
        t0 = 0.5 + line.write(begin_time=t0, transition_time=0.5)

        line2 = SimpleTexBObject(r"6\times 55 +47 =377", text_size="Large", aligned="left", color="blue",
                                 location=[0.3, 0, -6.1])
        t0 = 0.5 + line2.write(begin_time=t0, transition_time=0.5)

        line3 = SimpleTexBObject(r"322+55=377", text_size="Large", aligned="left", color="blue",
                                 location=[5.75, 0, -6.1])
        t0 = 0.5 + line3.write(begin_time=t0, transition_time=0.5)

        lines = [line, line2, line3]

        for i, line in enumerate(lines):
            line.disappear(begin_time=t0 + 0.1 * i, transition_time=0.5)

        t0 += 1

        self.t0 = t0

    def substitution_overlay2(self):
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        ibpy.set_camera_location(location=(0, -20, 0))
        create_glow_composition(threshold=1, size=1)

        t0 = 0

        base = SimpleTexBObject(r"\text{Base}", color="joker", location=[-7, 0, 5.75], text_size="Large")
        t0 = 0.1 + base.write(begin_time=t0, transition_time=0.5)

        addon = SimpleTexBObject(r"\text{Add-On}", color="custom1", location=[1.5, 0, 5.75], text_size="Large")
        t0 = 0.5 + addon.write(begin_time=t0, transition_time=0.5)

        left_column = [
            r"5\times 1 +2 =7",
            r"5\times 8 +7 =47",
            r"5\times 55 +47 =322",
            r"5\times 377 +322 =2207",
        ]

        l_shifts = [0, 0, -0.65, -1.27]
        l_colors = [
            flatten([["text"] * 2, ["custom1"] * 1, ["text"], ["joker"] * 1, ["text"], ["joker"]]),
            flatten([["text"] * 2, ["custom1"] * 1, ["text"], ["joker"] * 1, ["text"], ["joker"]]),
            flatten([["text"] * 2, ["custom1"] * 2, ["text"], ["joker"] * 2, ["text"], ["joker"]]),
            flatten([["text"] * 2, ["custom1"] * 3, ["text"], ["joker"] * 3, ["text"], ["joker"]]),
        ]

        center_column = [
            r"6\times 1 +2 =8",
            r"6\times 8 +7 =55",
            r"6\times 55 +47 =377",
            r"6\times 377 +322 =2584",
        ]

        c_shifts = [1.30, 1.30, 0.64, 0]
        c_colors = [
            flatten([["text"] * 2, ["custom1"] * 1, ["text"], ["joker"] * 1, ["text"], ["custom1"]]),
            flatten([["text"] * 2, ["custom1"] * 1, ["text"], ["joker"] * 1, ["text"], ["custom1"]]),
            flatten([["text"] * 2, ["custom1"] * 2, ["text"], ["joker"] * 2, ["text"], ["custom1"]]),
            flatten([["text"] * 2, ["custom1"] * 3, ["text"], ["joker"] * 3, ["text"], ["custom1"]]),
        ]

        right_column = [
            r"7+1=8",
            r"47+8=55",
            r"322+55=377",
            r"2207+377=2584",
        ]

        r_shifts = [1.60, 1.26, 0.64, 0]
        r_colors = [
            flatten([["joker"] * 1, ["text"], ["custom1"] * 1, ["text"], ["custom1"] * 1]),
            flatten([["joker"] * 2, ["text"], ["custom1"] * 1, ["text"], ["custom1"] * 2]),
            flatten([["joker"] * 3, ["text"], ["custom1"] * 2, ["text"], ["custom1"] * 3]),
            flatten([["joker"] * 4, ["text"], ["custom1"] * 3, ["text"], ["custom1"] * 4]),
        ]

        r_derivations = []
        c_derivations = []
        for i in range(4):
            left = SimpleTexBObject(left_column[i], text_size="Large", location=[-8.25 + l_shifts[i], 0, 4 - 2 * i],
                                    color=l_colors[i])
            t0 = 0.1 + left.write(begin_time=t0, transition_time=0.5)

            center = BDerivation(center_column[i], color=c_colors[i], text_size="Large",
                                 location=[-1.5 + c_shifts[i], 0, 4 - 2 * i], )
            t0 = 0.1 + center.write(begin_time=t0, transition_time=0.5)
            c_derivations.append(center)

            right = BDerivation(right_column[i], color=r_colors[i], text_size="Large",
                                location=[6 + r_shifts[i], 0, 4 - 2 * i])
            r_derivations.append(right)
            t0 = 0.5 + right.write(begin_time=t0, transition_time=0.5)

        # switch terms in right column

        r_column_new = [
            r"7=8-1",
            r"47=55-8",
            r"322=377-55",
            r"2207=2584-377",
        ]

        r_maps = [
            {"7": "7", "8": "8", "1": "1"},
            {"47": "47", "55": "55", "8": "8"},
            {"322": "322", "377": "377", "55": "55"},
            {"2207": "2207", "2584": "2584", "377": "377"},
        ]

        # frame add-on column

        rect = BQuadrilateral(vertices=[[-2, 0, -3], [5, 0, -3], [5, 0, 6.5], [-2, 0, 6.5]], color="custom1",
                              name="AddOnFrame", resolution=1000, thickness=4)
        t0 = 0.5 + rect.grow(begin_time=t0, transition_time=1)

        # highlight base cluster
        substring_highlights = ["2", "7", "47", "322"]

        for i, derivations in enumerate(c_derivations):
            derivations.highlight(substring_highlights[i], color="joker", emission=10, begin_time=t0,
                                  transition_time=1.5, flash_time=0.5)

        t0 = 0.5 + base.change_emission(from_value=0, to_value=5, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + base.change_emission(from_value=5, to_value=0, begin_time=t0, transition_time=0.5)

        # per-glyph colors for the reordered lines (color lists are positional)
        r_colors_new = [
            flatten([["joker"] * 1, ["text"], ["custom1"] * 1, ["text"], ["custom1"] * 1]),
            flatten([["joker"] * 2, ["text"], ["custom1"] * 2, ["text"], ["custom1"] * 1]),
            flatten([["joker"] * 3, ["text"], ["custom1"] * 3, ["text"], ["custom1"] * 2]),
            flatten([["joker"] * 4, ["text"], ["custom1"] * 4, ["text"], ["custom1"] * 3]),
        ]
        for i, derivation in enumerate(r_derivations):
            derivation.step(r_column_new[i], mode="swap", map=r_maps[i], color=r_colors_new[i],
                            begin_time=t0, transition_time=1.2, lift=0.5)
            derivation.move(direction=[-1.725, 0, 0], begin_time=t0, transition_time=1.25)

        t0 += 1.5
        c_column_new = [
            r"6\times 1 +2 =8",
            r"6\times 8 +8-1 =55",
            r"6\times 55 +55-8 =377",
            r"6\times 377 +377-55 =2584",
        ]

        rect.rescale(rescale=[1.15, 1, 1], begin_time=t0, transition_time=1)
        rect.move(direction=[-0.9, 0, 0], begin_time=t0, transition_time=1)

        c_colors_new = [
            flatten([["text"] * 2, ["custom1"] * 1, ["text"], ["custom1"] * 1, ["text"], ["custom1"]]),
            flatten([["text"] * 2, ["custom1"] * 1, ["text"], ["custom1"] * 1, ["text"], ["custom1"] * 1, ["text"],
                     ["custom1"]]),
            flatten([["text"] * 2, ["custom1"] * 2, ["text"], ["custom1"] * 2, ["text"], ["custom1"] * 1, ["text"],
                     ["custom1"]]),
            flatten([["text"] * 2, ["custom1"] * 3, ["text"], ["custom1"] * 3, ["text"], ["custom1"] * 2, ["text"],
                     ["custom1"]]),
        ]

        for i, derivation in enumerate(c_derivations):
            derivation.step(c_column_new[i], mode="replace", begin_time=t0, transition_time=1.2,
                            color=c_colors_new[i])

        # flights

        copies = ["8-1", "55-8", "377-55"]
        shifts = [
            Vector([-7.2, -1.95, 0]),
            Vector([-7.6, -1.95, 0]),
            Vector([-8.2, -1.95, 0]),
        ]
        # send copies of the Fibonacci numbers out to their tiles: the 8 and
        # the 377 from the last line, the first 55 from the second line; each
        # copy turns 'example' in flight and has faded out on arrival
        flights = [(r_derivation.lines[-1], copy, shift) for r_derivation, copy, shift in
                   zip(r_derivations, copies, shifts)]

        flight_time = 1.2
        t0 += 0.5

        for k, (line, spec, shift) in enumerate(flights):
            scale = line.ref_obj.scale
            local_shift = Vector([shift[i] / scale[i] for i in range(3)])
            for idx in line.find_letters(spec):
                target = line.letters[idx].ref_obj.location + local_shift
                line.move_copy_of_letter_to(idx, target, begin_time=t0,
                                            transition_time=flight_time)
                copy = line.copies_of_letters[-1]
                copy.disappear(begin_time=t0 + 0.99 * flight_time,
                               transition_time=0.01 * flight_time)

        t0 += 2

        recursion = SimpleTexBObject(r"a_{n+1}=7\times a_{n}-a_{n-1}", color="example", text_size="huge",
                                     location=[0, 0, -5], aligned="center")
        t0 = 0.5 + recursion.write(begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def substitution_explainer2(self):
        """Continue: now H7Tile/H8Tile play the role of H2/H1 and substitute
        into the second-level supertiles (superH7Tile/superH8Tile)."""
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        _setup_tilted_camera(center=(0, 0), distance=60, tilt=0)

        t0 = 0

        floor = Floor(u=[-30, 30], v=[-19, 15], color="checker", coords="Object")
        floor.move(direction=[0, 0, -0.1], begin_time=t0, transition_time=0)
        floor.grow(begin_time=t0, transition_time=0)

        left = SimpleTexBObject(r"\text{Base Cluster}", color="example", text_size="Huge", aligned="center",
                                location=Vector([-12, 18, 0]))
        left.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)
        left.write(begin_time=t0, transition_time=0)

        right = SimpleTexBObject(r"\text{Addon Cluster}", color="example", text_size="Huge", aligned="center",
                                 location=Vector([12, 18, 0]))
        right.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)
        right.write(begin_time=t0, transition_time=0)

        t0 = play_substitution(H7_TILE, H8_TILE, scale=0.5, solid=0.18, rescale=0.4)
        self.t0 = t0

    def substitution_explainer3(self):
        """Continue once more: superH7Tile/superH8Tile substitute into the
        third-level super-supertiles."""
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        _setup_tilted_camera(center=(0, 0), distance=60, tilt=0)

        t0 = 0

        floor = Floor(u=[-30, 30], v=[-19, 15], color="checker", coords="Object")
        floor.move(direction=[0, 0, -0.1], begin_time=t0, transition_time=0)
        floor.grow(begin_time=t0, transition_time=0)

        left = SimpleTexBObject(r"\text{Base Cluster}", color="example", text_size="Huge", aligned="center",
                                location=Vector([-12, 18, 0]))
        left.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)
        left.write(begin_time=t0, transition_time=0)

        right = SimpleTexBObject(r"\text{Addon Cluster}", color="example", text_size="Huge", aligned="center",
                                 location=Vector([12, 18, 0]))
        right.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)
        right.write(begin_time=t0, transition_time=0)

        t0 = play_substitution(SUPER_H7_TILE, SUPER_H8_TILE, anchor7=(-15, -7.5), scale=0.2, solid=0.07, rescale=0.4)
        self.t0 = t0

    def overlay(self):
        """
        Match the substitution algorithm with the Labbe Selinger Algorithm
        Run it low - poly, no extrusion on substitution, now wireframe on the Labbe Selinger
        It is important to show how the location in the fractal moves level by level
        """
        _setup_render(hdri="kloppenheim_06_puresky_4k",engine="CYCLES", transparent=True)
        _setup_tilted_camera(center=(0, 0), distance=60, tilt=0)
        set_alpha_composition()

        t0 = 0

        scales = [1, 1, 0.55, 0.25]
        locations = [
            Vector([-21.7, 12.9, 0]),
            Vector([-2.31, 13.4, 0]),
            Vector([24.5, 12.6, 0]),
            Vector([-10.5, -6.2, 0])
        ]
        tiles = ["H1_TILE","H8_TILE", "SUPER_H8_TILE", "SUPER_SUPER_H8_TILE","SUPER_SUPER_SUPER_H8_TILE"]
        carriers = []
        mismatch_carriers = []
        modifiers = []
        mismatch_modifiers = []
        for i, (tile, loc, scale) in enumerate(zip(tiles, locations, scales)):
            hat_carrier = Plane(name=tile + "Carrier")
            hat_carrier.rotate(rotation_euler=[0, 0, pi / 6], begin_time=0, transition_time=0)
            hat_carrier.rescale(rescale=scale, begin_time=00, transition_time=0)
            hat_carrier.move(direction=loc, begin_time=0, transition_time=0)
            carriers.append(hat_carrier)
            hat_mod = HatClusterCsvModifier(file_path=os.path.join(DATA_DIR, tile + ".dat"),
                                            color_scheme=1, extrude_scale=1)
            modifiers.append(hat_mod)
            hat_carrier.add_mesh_modifier(type='NODES', node_modifier=hat_mod)
            if 0<i <3:
                hat_carrier.appear(begin_time=t0, transition_time=1)
                hat_carrier = Plane(name=tile + "MissMatchCarrier")
                hat_carrier.rotate(rotation_euler=[0, 0, pi / 6], begin_time=0, transition_time=0)
                hat_carrier.rescale(rescale=scale, begin_time=0, transition_time=0)
                hat_carrier.move(direction=loc+Vector([0,0,-0.001]), begin_time=0, transition_time=0)
                mismatch_carriers.append(hat_carrier)
                hat_mod = HatClusterCsvModifier(file_path=os.path.join(DATA_DIR, tile + ".dat"),
                                                color_scheme=1, extrude_scale=1)
                mismatch_modifiers.append(hat_mod)
                hat_carrier.add_mesh_modifier(type='NODES', node_modifier=hat_mod)
            t0 = 0.5 + hat_carrier.appear(begin_time=t0, transition_time=1)

        initial_scale = 5
        fractal = Plane(name="FractalPlane", u=[-35 / 2, 35 / 2], v=[-10, 10], color="hat_tile_fractal", roughness=0.1,
                        depth=0)
        t0 = 0.5 + fractal.appear(begin_time=t0, transition_time=0.5)

        scale_node = get_node_from_shader(ibpy.get_material_at_slot(fractal, 0), label="Scale")
        depth_node = get_node_from_shader(ibpy.get_material_at_slot(fractal, 0), label="Depth")

        ibpy.change_default_value(scale_node, from_value=1, to_value=initial_scale, begin_time=t0, transition_time=3)
        t0 = 0.5 + ibpy.change_default_value(depth_node, from_value=0, to_value=10, begin_time=t0, transition_time=3)

        labbe_selinger_carrier = Plane(name="LabbeSelinger")
        labbe_selinger_carrier.rescale(rescale=initial_scale, from_scale=1, begin_time=0, transition_time=0)
        labbe_selinger_modifier = LabbeSelingerColorModifier(color_scheme=0, shift=Vector(), grid_filter=-1)

        grid_radius = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridRadius")
        grid_size = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridSize")
        ibpy.change_default_integer(grid_size, from_value=10, to_value=2, begin_time=0, transition_time=0)
        labbe_selinger_carrier.add_mesh_modifier(type='NODES', node_modifier=labbe_selinger_modifier)
        t0 = 0.5 + labbe_selinger_carrier.appear(alpha=0.75, begin_time=t0, transition_time=0.5)

        pin = Pin(location=Vector(), colors=["important"])
        pin.grow(begin_time=t0, transition_time=1)

        xi = Vector([0.5, r3 / 2, 0])
        shift = Vector([8/5,-0.3,0])

        def curve_func(t):
            if t < 1:
                return t * ((xi + shift) - Vector([0.5 / 3, 0, 0]))
            else:
                return (xi + shift) + float(0.5 / 3 / sqrt(t)) * Vector([-cos(t - 1), sin(t - 1), 0])

        curve = Curve(lambda t: initial_scale * curve_func(t), domain=[0, 15], color="joker",
                      thickness=0.001)
        curve.appear(alpha=0, begin_time=t0, transition_time=0)

        container = BObject(children=[fractal, curve, labbe_selinger_carrier], location=[35 / 2, -10, 0])
        container.appear(begin_time=0, transition_time=0)
        container.rotate(rotation_euler=[-pi / 4, 0, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.change_default_value(grid_radius, from_value=0, to_value=0.075, begin_time=t0,
                                             transition_time=1)

        # set the shift in the Labbe-Selinger-Modifier corresponding to the value of the curve
        shift_x_node = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="ShiftX")
        shift_y_node = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="ShiftY")

        # pin.follow's Follow Path constraint uses use_fixed_location=True, so
        # its offset_factor (0..1, default_smooth-eased below) is an ARC-LENGTH
        # fraction along the curve -- not a fraction of the raw parameter t.
        # This curve is nowhere near arc-length parametrized (the spiral's
        # radius shrinks like 1/sqrt(t), so most of the curve's length sits in
        # a small slice of small t); "15 * fraction" put the sample at the
        # wrong t entirely (e.g. fraction=0.5 -> true arc-length midpoint is
        # t~=0.9, not t=7.5). Build a dense arc-length table once and invert
        # it, so the segment endpoints land on the same points along the
        # curve that the pin itself passes through at the same times.
        _al_samples = np.linspace(0, 15, 2000)
        _al_points = np.array([[curve_func(float(t)).x, curve_func(float(t)).y] for t in _al_samples])
        _al_cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(_al_points, axis=0), axis=1))])
        _al_fraction = _al_cum / _al_cum[-1]

        def t_at_arclength_fraction(s):
            return float(np.interp(s, _al_fraction, _al_samples))

        for i in range(18):
            ti = t_at_arclength_fraction(ibpy.default_smooth(i / 18))
            te = t_at_arclength_fraction(ibpy.default_smooth((i + 1) / 18))
            ibpy.change_default_value(shift_x_node, from_value=curve_func(ti).x,
                                      to_value=curve_func(te).x, begin_time=t0 + 3 * i / 18, transition_time=3 / 18)
            ibpy.change_default_value(shift_y_node, from_value=curve_func(ti).y,
                                      to_value=curve_func(te).y, begin_time=t0 + 3 * i / 18, transition_time=3 / 18)

        t0 = 0.5 + pin.follow(curve, initial_value=0, final_value=1,
                              use_curve_radius=False, use_curve_follow=False,
                              begin_time=t0, transition_time=3)

        carriers[0].rescale(rescale=1.448 / 1, begin_time=t0, transition_time=1)
        carriers[0].rotate(rotation_euler=[-pi / 4, 0, pi / 6], rotation_mode="ZYX", begin_time=t0, transition_time=1)
        t0 = carriers[0].move_to(target_location=[31.739,-9.3514,-0.63], begin_time=t0, transition_time=1)
        color_node = get_geometry_node_from_modifier(modifiers[0], label="ColorScheme")
        ibpy.change_default_integer(color_node, from_value=1, to_value=0, begin_time=t0 - 0.1, transition_time=0.1)
        wireframe_node = get_geometry_node_from_modifier(modifiers[0], label="Wireframe")
        extrude_node = get_geometry_node_from_modifier(modifiers[0], label="ExtrudeScale")
        ibpy.change_default_value(extrude_node, from_value=1, to_value=0.25, begin_time=t0, transition_time=1)
        ibpy.change_default_boolean(wireframe_node, from_value=False, to_value=True, begin_time=t0)
        wireframe_radius = get_geometry_node_from_modifier(modifiers[0], label="WireframeRadius")
        ibpy.change_default_value(wireframe_radius, from_value=0.01, to_value=0.05, begin_time=t0, transition_time=0)
        grid_center_node = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridCenter")
        ibpy.change_default_vector(grid_center_node, from_value=Vector(),
                                   to_value=Vector([curve_func(15).x, curve_func(15).y, 0]), begin_time=t0,
                                   transition_time=0.1)

        t0 += 0.5

        grid_filter = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridFilter")
        t0 = ibpy.change_default_integer(grid_filter, from_value=-1, to_value=0, begin_time=t0, transition_time=0.1)
        t0 += 0.5
        ibpy.change_default_integer(grid_filter, from_value=0, to_value=-1, begin_time=t0, transition_time=0.2)
        t0 = 0.5 + carriers[0].disappear(begin_time=t0, transition_time=0.5)

        # show mismatch of the next cluster
        next_scale = 3
        mismatch_carrier = mismatch_carriers[0]
        mismatch_modifier= mismatch_modifiers[0]

        # rescale
        ibpy.change_default_value(scale_node, from_value=initial_scale, to_value=next_scale,
                                  begin_time=t0, transition_time=1)
        labbe_selinger_carrier.rescale(rescale=next_scale / initial_scale, begin_time=t0, transition_time=1)
        t0 = 0.5 + curve.rescale(rescale=next_scale / initial_scale, begin_time=t0, transition_time=1)

        # show mismatch of the next cluster
        mismatch_carrier.rotate(rotation_euler=[-pi / 4, 0, pi / 6], rotation_mode="ZYX", begin_time=t0,
                                transition_time=1)
        mismatch_carrier.rescale(rescale=1.448 / 1*3/5, from_scale=1, begin_time=t0, transition_time=1)
        mismatch_carrier.move_to(target_location=Vector([26.04,-9.6,-0.43]), begin_time=t0, transition_time=1)
        mismatch_wireframe_radius = get_geometry_node_from_modifier(mismatch_modifier, label="WireframeRadius")
        ibpy.change_default_value(mismatch_wireframe_radius, from_value=0.01, to_value=0.05, begin_time=t0,
                                             transition_time=1)
        wireframe_node = get_geometry_node_from_modifier(mismatch_modifier, label="Wireframe")
        extrude_node = get_geometry_node_from_modifier(mismatch_modifier, label="ExtrudeScale")
        ibpy.change_default_value(extrude_node, from_value=1, to_value=0.55, begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.change_default_boolean(wireframe_node, from_value=False, to_value=True, begin_time=t0)

        grid_filter = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridFilter")
        fractal.change_alpha(from_value=1,to_value=0.1,begin_time=t0-0.5,transition_time=0.5)
        t0 = ibpy.change_default_integer(grid_size, from_value=2, to_value=6, begin_time=t0, transition_time=0.1)
        t0 = ibpy.change_default_integer(grid_filter, from_value=-1, to_value=7, begin_time=t0, transition_time=2)

        t0 += 1  # hold on the grown mismatch so it registers before the fix

        ibpy.change_default_integer(grid_filter, from_value=7, to_value=-1, begin_time=t0, transition_time=0.2)
        fractal.change_alpha(from_value=0.1,to_value=1,begin_time=t0,transition_time=0.5)
        t0 = 0.5 + mismatch_carrier.disappear(begin_time=t0, transition_time=0.5)

        # move the pin
        dx = -1.007
        dy = 0.140
        for i in range(10):
            ti = ibpy.default_smooth(i / 10)
            te = ibpy.default_smooth((1 + i) / 10)

            x_i = dx * ti  # avoid clash with variable xi
            yi = dy * ti
            xe = dx * te
            ye = dy * te
            ibpy.change_default_value(shift_x_node, from_value=curve_func(15).x + x_i, to_value=curve_func(15).x + xe,
                                      begin_time=t0 + i / 10, transition_time=1 / 10)
            ibpy.change_default_value(shift_y_node, from_value=curve_func(15).y + yi, to_value=curve_func(15).y + ye,
                                      begin_time=t0 + i / 10, transition_time=1 / 10)

        curve_location = Vector([-3, 0.42, 0])
        t0 = 0.5 + curve.move(direction=curve_location, begin_time=t0, transition_time=1)

        carriers[1].rescale(rescale=1.448/5*3, begin_time=t0, transition_time=1)
        carriers[1].rotate(rotation_euler=[-pi / 4, 0, pi / 6], rotation_mode="ZYX", begin_time=t0, transition_time=1)
        t0 = carriers[1].move_to(target_location=[23.009, -9.3381, -0.67762], begin_time=t0, transition_time=1)
        color_node = get_geometry_node_from_modifier(modifiers[1], label="ColorScheme")
        ibpy.change_default_integer(color_node, from_value=1, to_value=0, begin_time=t0 - 0.1, transition_time=0.1)
        wireframe_node = get_geometry_node_from_modifier(modifiers[1], label="Wireframe")
        wireframe_radius = get_geometry_node_from_modifier(modifiers[1], label="WireframeRadius")
        ibpy.change_default_value(wireframe_radius, from_value=0.01, to_value=0.05, begin_time=t0, transition_time=0)

        extrude_node = get_geometry_node_from_modifier(modifiers[1], label="ExtrudeScale")
        ibpy.change_default_value(extrude_node, from_value=1, to_value=0.5, begin_time=t0, transition_time=1)
        ibpy.change_default_boolean(wireframe_node, from_value=False, to_value=True, begin_time=t0)
        t0 += 0.5

        grid_center_node = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridCenter")
        ibpy.change_default_vector(grid_center_node,from_value=Vector(),to_value=Vector(),begin_time=t0,transition_time=0)
        grid_filter = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridFilter")

        fractal.change_alpha(from_value=1, to_value=0.1, begin_time=t0 - 0.5, transition_time=0.5)
        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=-1, to_value=10, begin_time=t0, transition_time=2)
        t0 = ibpy.change_default_integer(grid_filter, from_value=10, to_value=-1, begin_time=t0, transition_time=2)
        fractal.change_alpha(from_value=0.1, to_value=1, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + carriers[1].disappear(begin_time=t0, transition_time=0.5)


        fractal2 = Plane(name="FractalPlane", u=[-35 / 2, 35 / 2], v=[-10, 10], color="hat_tile_fractal", roughness=0.1,
                         depth=10)
        t0 = 0.5 + fractal2.appear(begin_time=t0, transition_time=0.5)

        shader = ibpy.get_material_at_slot(fractal2, 0)
        scale_node2 = get_node_from_shader(shader, label="Scale")
        shader_shift_x_node = get_node_from_shader(shader, label="ShiftX")
        shader_shift_y_node = get_node_from_shader(shader, label="ShiftY")

        ibpy.change_default_value(scale_node2, from_value=initial_scale, to_value=3 * initial_scale, begin_time=t0,
                                  transition_time=1)
        ibpy.change_default_value(shader_shift_x_node, from_value=0, to_value=1, begin_time=t0, transition_time=1)
        ibpy.change_default_value(shader_shift_y_node, from_value=0, to_value=0.5, begin_time=t0, transition_time=1)

        pin2 = Pin(location=0.6*Vector([1.7156,6.18, 0]), colors=["important"], name="Pin2")

        container2 = BObject(children=[fractal2, pin2], location=[-35 / 2, 11.5, 0])
        container2.appear(begin_time=0, transition_time=0)
        container2.rotate(rotation_euler=[-pi / 4, 0, 0], begin_time=t0, transition_time=1)

        t0 = 0.5 + container.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + pin2.grow(begin_time=t0, transition_time=0.5)

        next_scale2 = 1.5
        ibpy.change_default_value(scale_node, from_value=next_scale, to_value=next_scale2,
                                  begin_time=t0, transition_time=1)
        ibpy.change_default_integer(grid_size, from_value=6, to_value=14, begin_time=t0 + 0.5, transition_time=0)
        labbe_selinger_carrier.rescale(rescale=next_scale2 / next_scale, begin_time=t0, transition_time=1)
        curve.move_to(target_location=next_scale2 / next_scale * curve_location, begin_time=t0, transition_time=1)
        t0 = 0.5 + curve.rescale(rescale=next_scale2 / next_scale, begin_time=t0, transition_time=1)

        # show mismatch of the next cluster
        mismatch_carrier=mismatch_carriers[1]
        mismatch_modifier=mismatch_modifiers[1]

        mismatch_carrier.rescale(rescale=1.448 / 0.55 * 1.5 / 5, from_scale=0.55, begin_time=t0, transition_time=1)
        mismatch_carrier.move_to(target_location=Vector([20.214, -9.48, 0]), begin_time=t0, transition_time=1)
        mismatch_wireframe_radius = get_geometry_node_from_modifier(mismatch_modifier, label="WireframeRadius")
        ibpy.change_default_value(mismatch_wireframe_radius, from_value=0.01, to_value=0.05, begin_time=t0,
                                  transition_time=1)
        wireframe_node = get_geometry_node_from_modifier(mismatch_modifier, label="Wireframe")
        extrude_node = get_geometry_node_from_modifier(mismatch_modifier, label="ExtrudeScale")
        ibpy.change_default_value(extrude_node, from_value=1, to_value=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.change_default_boolean(wireframe_node, from_value=False, to_value=True, begin_time=t0)

        grid_filter = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridFilter")
        fractal.change_alpha(from_value=1, to_value=0.1, begin_time=t0 - 0.5, transition_time=0.5)
        # t0 = ibpy.change_default_integer(grid_size, from_value=2, to_value=6, begin_time=t0, transition_time=0.1)

        t0 = ibpy.change_default_integer(grid_filter, from_value=-1, to_value=30, begin_time=t0, transition_time=2)

        t0 += 1  # hold on the grown mismatch so it registers before the fix

        ibpy.change_default_integer(grid_filter, from_value=30, to_value=-1, begin_time=t0, transition_time=0.2)
        fractal.change_alpha(from_value=0.1, to_value=1, begin_time=t0, transition_time=0.5)
        t0 = 0.5 + mismatch_carrier.disappear(begin_time=t0, transition_time=0.5)

        # move everything
        dx_old = -1.007
        dy_old = 0.140
        dx = -0.344
        dy = 0.07
        for i in range(10):
            ti = ibpy.default_smooth(i / 10)
            te = ibpy.default_smooth((1 + i) / 10)

            x_i = dx * ti  # avoid clash with variable xi
            yi = dy * ti
            xe = dx * te
            ye = dy * te
            ibpy.change_default_value(shift_x_node, from_value=curve_func(15).x+dx_old + x_i,
                                      to_value=curve_func(15).x +dx_old + xe,
                                      begin_time=t0 + i / 10, transition_time=1 / 10)
            ibpy.change_default_value(shift_y_node, from_value=curve_func(15).y+dy_old + yi,
                                      to_value=curve_func(15).y +dy_old + ye,
                                      begin_time=t0 + i / 10, transition_time=1 / 10)

        pin2.move_to(target_location=Vector([-3.88,4.82, 0]), begin_time=t0, transition_time=1)
        curve_location = Vector([-2.0152,0.32012, 0])
        t0 = 0.5 + curve.move_to(target_location=curve_location, begin_time=t0, transition_time=1)

        carriers[2].rescale(rescale=0.429 / 0.55*1.0107, begin_time=t0, transition_time=1)
        t0 = carriers[2].move_to(target_location=Vector([19.735, -9.3763, 0]), begin_time=t0, transition_time=1)
        color_node = get_geometry_node_from_modifier(modifiers[1], label="ColorScheme")
        ibpy.change_default_integer(color_node, from_value=1, to_value=1, begin_time=t0 - 0.1, transition_time=0.1)
        wireframe_node = get_geometry_node_from_modifier(modifiers[2], label="Wireframe")
        extrude_node = get_geometry_node_from_modifier(modifiers[2], label="ExtrudeScale")
        extrude = get_geometry_node_from_modifier(modifiers[2], label="ExtrudeSelector")
        ibpy.change_default_boolean(extrude, from_value=True, to_value=False, begin_time=t0 + 1)
        wireframe_radius = get_geometry_node_from_modifier(modifiers[2], label="WireframeRadius")
        ibpy.change_default_value(wireframe_radius, from_value=0.01, to_value=0.125, begin_time=t0, transition_time=0)
        ibpy.change_default_value(extrude_node, from_value=1, to_value=0.1, begin_time=t0, transition_time=1)
        ibpy.change_default_boolean(wireframe_node, from_value=False, to_value=True, begin_time=t0)
        t0 += 0.5

        t0 = 0.5 + fractal.change_alpha(alpha=0.1, begin_time=t0, transition_time=0.5)

        labbe_selinger_carrier2 = Plane(name="LabbeSelinger2")
        labbe_selinger_carrier2.move_to(target_location=[17.5, -10, 0], begin_time=t0, transition_time=0)
        labbe_selinger_carrier2.rescale(rescale=1.5, begin_time=t0, transition_time=0)
        labbe_selinger_modifier2 = LabbeSelingerColorModifier(color_scheme=0, shift=Vector([0.08, -0.03, 0]),
                                                              grid_filter=-1)
        labbe_selinger_carrier2.add_mesh_modifier(type='NODES', node_modifier=labbe_selinger_modifier2)
        labbe_selinger_carrier2.appear(alpha=1, begin_time=t0, transition_time=0)

        hat_scale = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier2, label="HatScale")
        grid_radius = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier2, label="GridRadius")
        grid_size2 = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier2, label="GridSize")
        grid_filter2 = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier2, label="GridFilter")
        ibpy.change_default_integer(color_node, from_value=0, to_value=1, begin_time=t0, transition_time=0)
        ibpy.change_default_value(hat_scale, from_value=0.95, to_value=0.99, begin_time=t0, transition_time=0)
        ibpy.change_default_value(grid_radius, from_value=0.025, to_value=0, begin_time=t0, transition_time=0)
        ibpy.change_default_integer(grid_size2, from_value=16, to_value=16, begin_time=t0, transition_time=0)
        t0 = 0.5 + ibpy.change_default_integer(grid_filter2, from_value=-1, to_value=281, begin_time=t0,
                                               transition_time=5)

        labbe_selinger_carrier2.change_alpha(alpha=0, begin_time=t0, transition_time=1)
        ibpy.change_default_integer(grid_filter2, from_value=281, to_value=-1, begin_time=t0, transition_time=0.1)
        carriers[2].disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + fractal.change_alpha(alpha=1, begin_time=t0, transition_time=0.5)




        # move everything
        pin2.move_to(target_location=Vector([-6.26, 5.246, 0]), begin_time=t0, transition_time=1)

        dx_old = -1.007-0.344
        dy_old = 0.140+0.07
        dx = -0.15
        dy = 0.03
        for i in range(10):
            ti = ibpy.default_smooth(i / 10)
            te = ibpy.default_smooth((1 + i) / 10)

            x_i = dx * ti  # avoid clash with variable xi
            yi = dy * ti
            xe = dx * te
            ye = dy * te
            ibpy.change_default_value(shift_x_node, from_value=curve_func(15).x + dx_old + x_i,
                                      to_value=curve_func(15).x + dx_old + xe,
                                      begin_time=t0 + i / 10, transition_time=1 / 10)
            ibpy.change_default_value(shift_y_node, from_value=curve_func(15).y + dy_old + yi,
                                      to_value=curve_func(15).y + dy_old + ye,
                                      begin_time=t0 + i / 10, transition_time=1 / 10)

        curve_location = Vector([-2.2322, 0.3621, 0])
        t0 = 0.5 + curve.move_to(target_location=curve_location, begin_time=t0, transition_time=1)

        next_scale3 = 0.75
        ibpy.change_default_value(scale_node, from_value=next_scale2, to_value=next_scale3,
                                  begin_time=t0, transition_time=1)
        ibpy.change_default_integer(grid_size, from_value=14, to_value=50, begin_time=t0 + 0.5, transition_time=0)
        labbe_selinger_carrier.rescale(rescale=next_scale3 / next_scale2, begin_time=t0, transition_time=1)
        curve.move_to(target_location=next_scale3 / next_scale2 * curve_location, begin_time=t0, transition_time=1)
        pin.rescale(rescale=next_scale3 / next_scale2, begin_time=t0, transition_time=1)
        t0 = 0.5 + curve.rescale(rescale=next_scale3 / next_scale2, begin_time=t0, transition_time=1)


        shift_x_node2 = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier2, label="ShiftX")
        shift_y_node2 = ibpy.get_geometry_node_from_modifier(labbe_selinger_modifier2, label="ShiftY")

        ibpy.change_default_value(shift_x_node2, from_value=0.08, to_value=0.08 - 0.037+0.06, begin_time=t0,
                                  transition_time=1)
        ibpy.change_default_value(shift_y_node2, from_value=-0.03, to_value=-0.03 + 0.011, begin_time=t0,
                                  transition_time=1)

        labbe_selinger_carrier2.change_alpha(alpha=1, begin_time=t0 - 0.5, transition_time=0.1)
        labbe_selinger_carrier2.rescale(rescale=0.5, begin_time=t0 - 0.5, transition_time=0.1)
        ibpy.change_default_integer(grid_size2, from_value=16, to_value=50, begin_time=t0 - 0.5, transition_time=0.1)
        ibpy.change_default_value(grid_radius, from_value=0.025, to_value=0.0, begin_time=t0, transition_time=0)

        carriers[3].rescale(rescale=0.217 / 0.25*0.9968, begin_time=t0, transition_time=1)
        ibpy.camera_zoom(lens=29, begin_time=t0, transition_time=0.5)
        t0 = carriers[3].move_to(target_location=Vector([18.522, -9.6908, 0]), begin_time=t0, transition_time=1)
        wireframe_node = get_geometry_node_from_modifier(modifiers[3], label="Wireframe")
        wireframe_radius = get_geometry_node_from_modifier(modifiers[3], label="WireframeRadius")
        extrude_node = get_geometry_node_from_modifier(modifiers[3], label="ExtrudeScale")
        extrude = get_geometry_node_from_modifier(modifiers[3], label="ExtrudeSelector")
        ibpy.change_default_boolean(extrude, from_value=True, to_value=False, begin_time=t0 + 1)

        ibpy.change_default_value(extrude_node, from_value=1, to_value=0, begin_time=t0, transition_time=1)
        ibpy.change_default_value(wireframe_radius, from_value=0.01, to_value=0.25, begin_time=t0, transition_time=0)
        ibpy.change_default_boolean(wireframe_node, from_value=False, to_value=True, begin_time=t0)
        t0 = 0.5 + fractal.change_alpha(alpha=0.1, begin_time=t0, transition_time=0.5)

        t0 = 0.5 + ibpy.change_default_integer(grid_filter2, from_value=-1, to_value=2175, begin_time=t0,
                                               transition_time=5)

        self.t0 = t0

    def rot_sym_intro(self):
        """
        combine three hats into a rotationally symmetric clsuter
        """
        t0 = 0.5
        _setup_render(hdri="kloppenheim_06_puresky_4k", engine="CYCLES", transparent=True)
        shift = Vector([phi / 2 + 1 / 2, r3 / 2 + phi / 2 / r3, 0])
        center = Vector()
        camera_circle = BezierCircle(center=shift, radius=10, rotation_euler=[0, pi / 2, 0])
        camera_empty = EmptyCube(location=center)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_follow(camera_circle)
        ibpy.set_camera_location(location=Vector())
        ibpy.camera_follow(camera_circle, initial_value=0.24, final_value=0.24, begin_time=0, transition_time=0)

        shift = Vector([phi / 2 + 1 / 2, r3 / 2 + phi / 2 / r3, 0])
        fractal = Plane(name="FractalPlane", u=[-50, 50], v=[-50, 50], color="hat_tile_fractal", roughness=0.1,
                        depth=10)
        t0 = 0.5 + fractal.appear(begin_time=t0, transition_time=0.5)

        tri_shift = Vector([-(phi - 1) / 2, -(phi - 1) / 2 / r3, 0])

        ibpy.camera_zoom(lens=50, begin_time=t0, transition_time=1)
        camera_circle.move_to(target_location=-shift, begin_time=t0, transition_time=1)
        t0 = 0.5 + camera_empty.move_to(target_location=-shift, begin_time=t0, transition_time=1)

        tri_carrier = Plane(name="TriPlane")
        tri_mod = TriangularGridModifier(grid_n=8, colors=["red", "important"],
                                         show_fundamental_plane=False,
                                         dot_size=0.1 / 2, edge_thickness=0.005)
        tri_carrier.add_mesh_modifier(type='NODES', node_modifier=tri_mod)
        tri_carrier.rotate(rotation_euler=[0, 0, 0])
        tri_carrier.rescale(rescale=1, begin_time=t0, transition_time=0)
        t0 = 0.5 + tri_carrier.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + tri_carrier.move(direction=tri_shift, begin_time=t0, transition_time=1)

        t0 = 0.5 + ibpy.camera_follow(camera_circle, initial_value=0.24, final_value=0.125, begin_time=t0,
                                      transition_time=1)

        center = Pin(location=-shift, colors=["important"], scale=0.5)
        t0 = 0.5 + center.grow(begin_time=t0, transition_time=0.5)

        labbe_selinger_carrier = Plane(name="LabbeSelinger")
        labbe_selinger_modifier = LabbeSelingerColorModifier(color_scheme=0, shift=-shift - Vector([0, 1 / r3, 0]),
                                                             grid_filter=20)
        labbe_selinger_carrier.add_mesh_modifier(type='NODES', node_modifier=labbe_selinger_modifier)
        wireframe = get_geometry_node_from_modifier(labbe_selinger_modifier, label="Wireframe")
        grid_center = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridCenter")
        # make sure that the hats around the center of rotation are shown first and grown circularly
        ibpy.change_default_vector(grid_center, from_value=Vector(), to_value=Vector([-1.4, -1.16, 0]), begin_time=0,
                                   transition_time=0)
        ibpy.change_default_boolean(wireframe, from_value=True, to_value=False, begin_time=0)
        hat_scale = get_geometry_node_from_modifier(labbe_selinger_modifier, label="HatScale")
        ibpy.change_default_value(hat_scale, from_value=0.95, to_value=0.99, begin_time=0, transition_time=0)
        labbe_selinger_carrier.appear(alpha=1, begin_time=t0, transition_time=0)
        grid_filter = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridFilter")

        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=0, to_value=5, begin_time=t0, transition_time=3)
        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=5, to_value=12, begin_time=t0, transition_time=3)

        ibpy.camera_follow(camera_circle, initial_value=0.125, final_value=0.249, begin_time=t0, transition_time=1)
        ibpy.camera_zoom(lens=5, begin_time=t0, transition_time=1)
        fractal.disappear(begin_time=t0, transition_time=1)

        grid_size = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridSize")
        ibpy.change_default_integer(grid_size, from_value=10, to_value=50, begin_time=t0, transition_time=0)
        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=12, to_value=2000, begin_time=t0,
                                               transition_time=10)

        color_switch = get_geometry_node_from_modifier(labbe_selinger_modifier, label="ColorScheme")
        t0 = 0.5 + ibpy.change_default_integer(color_switch, from_value=0, to_value=1, begin_time=t0, transition_time=0)

        grid_radius = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridRadius")
        ibpy.change_default_value(grid_radius, from_value=0.025, to_value=0, begin_time=t0, transition_time=0.5)
        tri_carrier.disappear(begin_time=t0, transition_time=0.5)
        extrude = get_geometry_node_from_modifier(labbe_selinger_modifier, label="ExtrudeScale")
        t0 = ibpy.change_default_value(extrude, from_value=0.15, to_value=0, begin_time=t0, transition_time=0.5)
        wireframe_radius = get_geometry_node_from_modifier(labbe_selinger_modifier, label="WireRadius")
        ibpy.change_default_value(wireframe_radius, from_value=0.01, to_value=0.05, begin_time=t0, transition_time=0.5)
        wireframe = get_geometry_node_from_modifier(labbe_selinger_modifier, label="Wireframe")
        solid = get_geometry_node_from_modifier(labbe_selinger_modifier, label="Solid")
        ibpy.change_default_boolean(solid, from_value=True, to_value=False, begin_time=t0)
        t0 = 0.5 + ibpy.change_default_boolean(wireframe, from_value=False, to_value=True, begin_time=t0)

        ibpy.camera_follow(camera_circle, initial_value=0.249, final_value=0.125, begin_time=t0, transition_time=10)
        t0 = 0.5 + ibpy.camera_zoom(lens=40, begin_time=t0, transition_time=10)

        pivot = get_geometry_node_from_modifier(labbe_selinger_modifier, label="Pivot")
        ibpy.change_default_vector(pivot, from_value=Vector(), to_value=-shift, begin_time=0, transition_time=0)
        rotation = get_geometry_node_from_modifier(labbe_selinger_modifier, label="FinalRotation")

        t0 = 0.5 + ibpy.change_default_vector(rotation, from_value=Vector(), to_value=Vector([0, 0, tau / 3]),
                                              begin_time=t0, transition_time=5)

        ibpy.camera_follow(camera_circle, initial_value=0.125, final_value=0.249, begin_time=t0, transition_time=5)
        ibpy.change_default_vector(rotation, from_value=Vector([0, 0, tau / 3]), to_value=Vector([0, 0, 2 * tau / 3]),
                                   begin_time=t0,
                                   transition_time=5)
        t0 = 0.5 + ibpy.camera_zoom(lens=10, begin_time=t0, transition_time=5)

        self.t0 = t0

    def rot_sym(self):
        """
        combine three hats into a rotationally symmetric cluster
        """
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        _setup_tilted_camera(center=(0, 0), distance=10, tilt=0)

        t0 = 0
        shift = Vector([phi / 2 + 1 / 2, r3 / 2 + phi / 2 / r3, 0])
        # fractal = Plane(name="FractalPlane", u=[-50, 50], v=[-50, 50], color="hat_tile_fractal", roughness=0.1,
        #                 depth=10)
        # fractal.move(direction=shift, begin_time=0, transition_time=0)
        # t0 = 0.5 + fractal.appear(begin_time=t0, transition_time=0.5)

        t0 = 1

        wireframe_dials = []
        rot_suffixes = ["000", "240", "120", ]
        rot_matrices = [Matrix([[-np.sin(a), np.cos(a), 0], [np.cos(a), np.sin(a), 0], [0, 0, 1]]) for a in
                        [0, 2 * pi / 3, 4 * pi / 3]]
        for rot, suffix in zip(rot_matrices, rot_suffixes):
            hat_carrier = Plane(name="H1Carrier")
            hat_carrier.rotate(rotation_euler=[0, 0, pi / 6], begin_time=t0, transition_time=0)
            hat_carrier.rescale(rescale=1 / 2 / r3, begin_time=t0, transition_time=0)
            hat_carrier.move(direction=Vector(), begin_time=t0, transition_time=0)
            data_path = os.path.join(os.path.dirname(__file__), "data", f"h1_{suffix}.csv")

            hat_mod = HatClusterCsvModifier(file_path=data_path, color_scheme=1)
            hat_carrier.add_mesh_modifier(type='NODES', node_modifier=hat_mod)
            hat_carrier.appear(begin_time=t0, transition_time=0)
            translation = get_geometry_node_from_modifier(hat_mod, label="Translation")
            wireframe_dials.append(get_geometry_node_from_modifier(hat_mod, label="Wireframe"))

            ibpy.change_default_boolean(wireframe_dials[-1], from_value=True, to_value=False, begin_time=t0)
            ibpy.change_default_vector(translation, from_value=rot @ Vector([0, -25, 0]), to_value=Vector(),
                                       begin_time=t0, transition_time=3)

        t0 += 3.5

        [ibpy.change_default_boolean(wireframe, from_value=False, to_value=True, begin_time=t0 + i * 0.45) for
         i, wireframe in enumerate(wireframe_dials)]
        t0 += 1.5

        labbe_selinger_carrier = Plane(name="LabbeSelinger")
        labbe_selinger_carrier.move_to(target_location=shift, begin_time=t0, transition_time=0)
        labbe_selinger_modifier = LabbeSelingerColorModifier(color_scheme=0, shift=-shift - Vector([0, 1 / r3, 0]),
                                                             grid_filter=20)
        labbe_selinger_carrier.add_mesh_modifier(type='NODES', node_modifier=labbe_selinger_modifier)
        wireframe = get_geometry_node_from_modifier(labbe_selinger_modifier, label="Wireframe")
        grid_center = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridCenter")
        grid_size = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridSize")
        ibpy.change_default_integer(grid_size, from_value=10, to_value=200, begin_time=t0, transition_time=0)
        # make sure that the hats around the center of rotation are shown first and grown circularly
        ibpy.change_default_vector(grid_center, from_value=Vector(), to_value=Vector([-1.4, -1.16, 0]), begin_time=0,
                                   transition_time=0)
        ibpy.change_default_boolean(wireframe, from_value=True, to_value=False, begin_time=0)
        hat_scale = get_geometry_node_from_modifier(labbe_selinger_modifier, label="HatScale")
        ibpy.change_default_value(hat_scale, from_value=0.95, to_value=0.99, begin_time=0, transition_time=0)
        labbe_selinger_carrier.appear(alpha=1, begin_time=t0, transition_time=0)
        grid_radius = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridRadius")
        ibpy.change_default_value(grid_radius, from_value=0.025, to_value=0.075, begin_time=t0, transition_time=3)
        grid_filter = get_geometry_node_from_modifier(labbe_selinger_modifier, label="GridFilter")
        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=0, to_value=5, begin_time=t0, transition_time=3)

        wireframe_dials = []
        rot_suffixes = ["240", "120", "000", ]
        rot_matrices = [Matrix([[-np.sin(a), np.cos(a), 0], [np.cos(a), np.sin(a), 0], [0, 0, 1]]) for a in
                        [0, 2 * pi / 3, 4 * pi / 3]]
        for rot, suffix in zip(rot_matrices, rot_suffixes):
            hat_carrier = Plane(name="H8Carrier")
            hat_carrier.rotate(rotation_euler=[0, 0, pi / 6], begin_time=t0, transition_time=0)
            hat_carrier.rescale(rescale=1 / 2 / r3, begin_time=t0, transition_time=0)
            hat_carrier.move(direction=Vector(), begin_time=t0, transition_time=0)
            data_path = os.path.join(os.path.dirname(__file__), "data", f"h8_{suffix}.csv")

            hat_mod = HatClusterCsvModifier(file_path=data_path, color_scheme=1)
            hat_carrier.add_mesh_modifier(type='NODES', node_modifier=hat_mod)
            hat_carrier.appear(begin_time=t0, transition_time=0)
            translation = get_geometry_node_from_modifier(hat_mod, label="Translation")
            wireframe_dials.append(get_geometry_node_from_modifier(hat_mod, label="Wireframe"))

            ibpy.change_default_boolean(wireframe_dials[-1], from_value=True, to_value=False, begin_time=t0)
            ibpy.change_default_vector(translation, from_value=rot @ Vector([0, -25, 0]), to_value=Vector(),
                                       begin_time=t0, transition_time=3)

        t0 += 3.5

        [ibpy.change_default_boolean(wireframe, from_value=False, to_value=True, begin_time=t0 + i * 0.45) for
         i, wireframe in enumerate(wireframe_dials)]
        # fractal.disappear(begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.camera_move(shift=[0, 0, 10], begin_time=t0, transition_time=1)
        # t0 =0.5 + ibpy.camera_zoom(lens=15,begin_time=t0,transition_time=1)

        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=5, to_value=26, begin_time=t0, transition_time=1)

        wireframe_dials = []
        rot_suffixes = ["240", "120", "000", ]
        rot_matrices = [Matrix([[-np.sin(a), -np.cos(a), 0], [np.cos(a), -np.sin(a), 0], [0, 0, 1]]) for a in
                        [0, 2 * pi / 3, 4 * pi / 3]]
        for rot, suffix in zip(rot_matrices, rot_suffixes):
            hat_carrier = Plane(name="SuperH8Carrier")
            hat_carrier.rotate(rotation_euler=[0, 0, pi / 6], begin_time=t0, transition_time=0)
            hat_carrier.rescale(rescale=1 / 2 / r3, begin_time=t0, transition_time=0)
            hat_carrier.move(direction=Vector(), begin_time=t0, transition_time=0)
            data_path = os.path.join(os.path.dirname(__file__), "data", f"super_h8_{suffix}.csv")

            hat_mod = HatClusterCsvModifier(file_path=data_path, color_scheme=1)
            hat_carrier.add_mesh_modifier(type='NODES', node_modifier=hat_mod)
            hat_carrier.appear(begin_time=t0, transition_time=0)
            translation = get_geometry_node_from_modifier(hat_mod, label="Translation")
            wireframe_dials.append(get_geometry_node_from_modifier(hat_mod, label="Wireframe"))

            ibpy.change_default_boolean(wireframe_dials[-1], from_value=True, to_value=False, begin_time=t0)
            ibpy.change_default_vector(translation, from_value=rot @ Vector([0, -50, 0]), to_value=Vector(),
                                       begin_time=t0, transition_time=3)

        t0 = 0.5 + ibpy.camera_move(shift=[0, 0, 20 * 0.5], begin_time=t0, transition_time=3)
        # t0 = 0.5 + ibpy.camera_zoom(lens=10,begin_time=t0,transition_time=3)

        [ibpy.change_default_boolean(wireframe, from_value=False, to_value=True, begin_time=t0 + i * 0.45) for
         i, wireframe in enumerate(wireframe_dials)]
        t0 += 1.5

        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=26, to_value=123, begin_time=t0,
                                               transition_time=3)

        t0 = 0.5 + ibpy.camera_move(shift=[0, 0, 20 * 1.5 * 1.5], begin_time=t0, transition_time=3)
        # t0 = 0.5 + ibpy.camera_zoom(lens=4, begin_time=t0, transition_time=3)

        wireframe_dials = []
        rot_suffixes = ["", "_120", "_240"]
        rot_matrices = [Matrix([[-np.sin(a), np.cos(a), 0], [np.cos(a), np.sin(a), 0], [0, 0, 1]]) for a in
                        [0, 2 * pi / 3, 4 * pi / 3]]
        for rot, suffix in zip(rot_matrices, rot_suffixes):
            hat_carrier = Plane(name="SuperSuperH8Carrier")
            hat_carrier.rotate(rotation_euler=[0, 0, pi / 6], begin_time=t0, transition_time=0)
            hat_carrier.rescale(rescale=1 / 2 / r3, begin_time=t0, transition_time=0)
            hat_carrier.move(direction=Vector(), begin_time=t0, transition_time=0)
            data_path = os.path.join(os.path.dirname(__file__), "data", f"SUPER_SUPER_H8{suffix}.csv")

            hat_mod = HatClusterCsvModifier(file_path=data_path, color_scheme=1)
            hat_carrier.add_mesh_modifier(type='NODES', node_modifier=hat_mod)
            hat_carrier.appear(begin_time=t0, transition_time=0)
            translation = get_geometry_node_from_modifier(hat_mod, label="Translation")
            wireframe_dials.append(get_geometry_node_from_modifier(hat_mod, label="Wireframe"))

            ibpy.change_default_boolean(wireframe_dials[-1], from_value=True, to_value=False, begin_time=t0)
            ibpy.change_default_vector(translation, from_value=rot @ Vector([0, -200, 0]), to_value=Vector(),
                                       begin_time=t0, transition_time=3)

        t0 += 3.5

        [ibpy.change_default_boolean(wireframe, from_value=False, to_value=True, begin_time=t0 + i * 0.45) for
         i, wireframe in enumerate(wireframe_dials)]
        t0 += 1.5

        t0 = 0.5 + ibpy.change_default_integer(grid_filter, from_value=123, to_value=9500, begin_time=t0,
                                               transition_time=10)

        self.t0 = t0

    def dead_end_leaves(self):
        """Draw all 21 leaves (dead corners) of the 120-degree-symmetric
        growth tree around the 3-hat pinwheel (data/dead_end_sym_tree.svg),
        one after another, from data/dead_end_all_cluster.csv.

        A single ``DeadEndClusterModifier`` holds the union of all cluster
        hats; the animated float ``ClusterTime`` walks through the states
        (0 = empty, 1 = seed pinwheel, 2.. = the leaves in file order).
        Between two leaves only the difference is animated: hats missing from
        the next leaf shrink away in the first half of a transition, new hats
        grow radially in the second half, and common hats stay put — the
        three growth-blocking hats of every leaf just fade to red.  The seed
        pinwheel never moves and turns red only in the final leaf, where the
        dead corner sits at the pinwheel itself; the camera pulls back while
        the clusters grow and comes back in for that finale."""
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        tilt = 0.22
        distance = 26.0
        ibpy.set_camera_location(location=[0, -tilt * distance, distance])
        cam_empty = EmptyCube(location=Vector((0, 0, 0)), name="DeadEndCamEmpty")
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(cam_empty)

        data_path = os.path.join(os.path.dirname(__file__), "data", "dead_end_all_cluster.csv")
        carrier = Plane(name="DeadEndCarrier")
        modifier = DeadEndClusterModifier(file_path=data_path)
        carrier.add_mesh_modifier(type='NODES', node_modifier=modifier)
        carrier.appear(begin_time=0, transition_time=0)

        time_dial = get_geometry_node_from_modifier(modifier, label="ClusterTime")
        sizes = modifier.transition_sizes
        radii = modifier.state_radii

        t0 = 0.5
        for k, (removed, added) in enumerate(sizes):
            # transition k -> k+1; bigger diffs get more time
            duration = min(3.6, max(1.6, 1.2 + 0.045 * (removed + added)))
            final = (k == len(sizes) - 1)

            # pull back whenever the next cluster outgrows the view, come
            # back in for the small red pinwheel at the very end
            target = min(68.0, max(26.0, 3.1 * radii[k + 1] + 7))
            if (target > distance + 2 or (final and abs(target - distance) > 2)) and k > 1:
                shift = target - distance
                ibpy.camera_move(shift=[0, -tilt * shift, shift], begin_time=t0,
                                 transition_time=duration + (2 if final else 0))
                distance = target

            hold = 1.2 if k > 0 else 0.6  # a beat to take in each dead end
            t0 = hold + ibpy.change_default_value(time_dial, from_value=k, to_value=k + 1,
                                                  begin_time=t0, transition_time=duration)

        self.t0 = t0 + 2

    def hat_tile_basics(self):
        """Introduce the geometry of the hat tile.

        A philosopher reads a list of walking instructions (three columns,
        synchronised with a pencil) and draws the 14-point hat tile at the
        origin of a geometry-nodes coordinate system, overlaid -- together with
        a hexagonal and a triangular grid -- on a plane of sand.  Waving flags
        label every vertex with its exact ``(x, y)`` coordinate (sqrt(3) and
        fractions).  Finally a real, thick ``joker``-coloured hat tile appears
        inside the drawing, pops out of the sand (rotates into the x-z plane)
        and spins about the z-axis.
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)

        # camera: front-top view so the final pop-out is clearly visible
        ibpy.set_camera_location(location=[0, -50, 30])
        ibpy.set_camera_lens(lens=85)
        cam_empty = EmptyCube(location=[0, 0, 2], name='HatBasicsCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=cam_empty)

        # --- plane of sand ---
        sand = Plane(u=[-10, 10], v=[-10, 10], color='sand', name='SandPlane', emission=0.3)
        sand.appear(begin_time=t0, transition_time=1)
        ibpy.add_modifier(sand, type='COLLISION')
        sand.move(direction=[0, 0, -0.1], begin_time=t0, transition_time=0)

        coords = CoordinateSystem2(
            dimension=2, location=[0, 0, 0.02],
            lengths=[16, 16], domains=[[-4, 4], [-4, 4]],
            n_tics=[8, 8], tic_labels=['AUTO', 'AUTO'], tic_label_digits=[0, 0],
            include_zeros=[False, False],
            colors=['custom1', 'custom1'],
            labels=[r"\phantom{x}", r"\phantom{y}"],
            tic_label_shifts=[Vector([-0.44, -1.04, -0.8]), Vector([-0.6, 0, 0])],
            name='HatCoords',
            directions=["HORIZONTAL", "DEEP"],
            label_rotations=[Vector(), Vector()],
            origin=[0, 0],

            axes_labels={"x": [0.1, 1, 16], "y": Vector([-1.3, 0, 16])}
        )
        t0 = coords.appear(begin_time=t0, transition_time=2)

        philosopher = PersonWithCape(location=[-14, -6, 0.1], rotation_euler=[0, 0, np.pi / 3],
                                     colors=['gray_8', 'joker'],
                                     simulation_start=0, simulation_duration=60, name='HatPhilosopher')
        philosopher.appear(begin_time=t0, transition_time=1)
        t0 += 1.5

        # --- 14-point hat geometry (exact coordinates) ---
        sym = _hat14_symbolic()  # [(Ax, By), ...]
        verts = [Vector((ax, by * r3, 0.05)) for (ax, by) in sym]
        n = len(verts)  # 14 boundary vertices

        # --- pencil draws the hat, edge by edge, synced with the instructions ---
        pencil = Pencil(location=verts[0], colors=['wood', 'joker'], name='HatPencil')
        pencil.appear(begin_time=t0, transition_time=1)
        pencil.rotate(rotation_euler=[-np.pi / 9, 0, 0], begin_time=t0, transition_time=1)
        philosopher.move_to(target_location=Vector([-1.4, 0, 0]), begin_time=t0, transition_time=1)
        philosopher.rotate(rotation_euler=[0, 0, np.atan2(-0.7, 0)])
        t0 += 1.2

        edge_dt = 1.1
        drawing = []
        first_dot = Sphere(0.08, location=verts[0], color='joker', name='HatVertex0')
        first_dot.grow(begin_time=t0, transition_time=0.3)
        drawing.append(first_dot)

        drawing_starts = t0
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]

            edge = Cylinder.from_start_to_end(start=a, end=b, thickness=0.8, color='joker')
            edge.grow(modus='from_start', begin_time=t0, transition_time=edge_dt * 0.8)
            pencil.move_to(target_location=b, begin_time=t0, transition_time=edge_dt * 0.8)
            philosopher.move_to(
                target_location=Vector([-0.7, 0, -0.8]) if b.length == 0 else b + 0.85 * b.normalized() + Vector(
                    [0, 0, -0.8]),
                begin_time=t0, transition_time=edge_dt * 0.8)
            philosopher.rotate(rotation_euler=[0, 0, np.atan2(verts[i][0], verts[i][1])], begin_time=t0,
                               transition_time=edge_dt * 0.8)
            drawing.append(edge)

            if (i + 1) % n != 0:  # don't re-draw the closing vertex
                dot = Sphere(0.16, location=b, color='joker', name='HatVertex%d' % (i + 1))
                dot.grow(begin_time=t0 + edge_dt * 0.8, transition_time=0.3)
                drawing.append(dot)

            t0 += edge_dt

        pencil.disappear(begin_time=t0, transition_time=0.5)
        t0 += 0.7
        drawing_ends = t0

        # --- waving flags carrying each vertex's exact coordinate ---
        flags = []
        for i in range(n):
            ax, by = sym[i]
            fl = Flag(colors=['custom1', 'joker'], name='HatFlag%d' % i,
                      location=verts[i] + Vector((0, 0, 10.02)),
                      simulation_start=drawing_starts, simulation_duration=20)
            fl.appear(begin_time=drawing_starts, transition_time=0)
            image = ImageCreator(_coord_flag_text(ax, by), i, prefix=fl.name).get_image_path()
            fl.add_image_texture(image, int((drawing_starts + 0.1) * FRAME_RATE))
            fl.move(direction=[0, 0, -10], begin_time=drawing_starts + i + 1, transition_time=2)
            flags.append(fl)

        philosopher.disappear(begin_time=drawing_ends, transition_time=0.5)
        t0 = drawing_ends + 2

        # --- the real, solid hat tile (joker colour, near-zero roughness) ---
        tile_verts = [(v.x, v.y, 0.0) for v in verts]
        real_hat = BObject(mesh=create_mesh(tile_verts, faces=[list(range(n))]),
                           name='RealHatTile', color='joker',
                           solid=0.18, roughness=0.05, metallic=0.2,
                           location=Vector((0, 0, 0.05)))
        real_hat.grow(begin_time=t0, transition_time=1.5)
        t0 += 2

        # --- pop out of the sand: rotate into the x-z plane, then spin about z ---
        real_hat.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=2)
        real_hat.move(direction=[0, 0, 3], begin_time=t0, transition_time=2)
        [obj.disappear(begin_time=t0, transition_time=0.5) for obj in [sand, coords] + flags + drawing]
        cam_empty.move_to(target_location=Vector([0, 0, 6]), begin_time=t0, transition_time=2)
        ibpy.camera_move(shift=Vector([0, 0, -24]), begin_time=t0, transition_time=2)

        t0 += 2.3
        real_hat.rotate(rotation_euler=[np.pi / 2, 0, 2 * np.pi], begin_time=t0, transition_time=7)
        t0 += 7

        # --- overlays on the sand: hexagonal grid + triangular grid (geometry
        #     nodes) and a geometry-nodes coordinate system ---

        tri_carrier = Plane(name='TriGridCarrier', location=[0, 0, 3], rotation_euler=[pi / 2, pi / 6, 0])
        tri_mod = TriangularGridModifier(grid_n=8, colors=["red", "important"], show_fundamental_plane=False,
                                         dot_size=0.1 / 2)
        tri_carrier.add_mesh_modifier(type='NODES', node_modifier=tri_mod)
        tri_carrier.rescale(rescale=4 * r3, begin_time=t0, transition_time=0)
        tri_carrier.move(direction=[3, 0, 4 / 3], begin_time=t0, transition_time=0)
        t0 = 0.5 + tri_carrier.appear(begin_time=t0, transition_time=1)

        sphere = Sphere(0.6, location=Vector([-3, 0, 4.8]), color='red')
        t0 = 0.5 + sphere.grow(begin_time=t0, transition_time=1)

        hex_carrier = Plane(name='HexGridCarrier', location=[0, 0, 6], rotation_euler=[pi / 2, pi / 6, 0])
        hex_mod = HexagonalTilingModifier(iterations=30, with_edges=True, edge_length=1.0, colors=["joker", "example"],
                                          dot_size=0)
        hex_carrier.add_mesh_modifier(type='NODES', node_modifier=hex_mod)
        hex_carrier.appear(begin_time=t0, transition_time=1)
        hex_carrier.rescale(rescale=8, begin_time=t0, transition_time=0)
        hex_carrier.move(direction=[3, 8, 4 / 3], begin_time=t0, transition_time=0)

        self.t0 = t0

    def hat_from_kites(self):
        """Reveal the hat tile as a polykite of eight kites.

        The scene picks up exactly where :meth:`hat_tile_basics` ends: the hat
        stands upright in the x-z plane, elevated in front of a head-on camera
        and overlaid with the triangular and hexagonal grids (plus the little
        red marker sphere).  This time the hat is not a single ``joker``
        polygon but a composition of the *eight kites* it is really made of
        (see :func:`_hat_kites`).

        The kites first drift slightly apart along their radial directions --
        with a small twist -- and turn back to exactly where they started,
        making the eight-piece decomposition legible.  Finally each kite fades
        into its own colour, sweeping a rainbow around the tile.
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", transparent=True)
        set_alpha_composition()

        # camera: same head-on framing the previous scene finishes on
        ibpy.set_camera_location(location=[0, -50, 6])
        ibpy.set_camera_lens(lens=85)
        cam_empty = EmptyCube(location=[0, 0, 6], name='HatKitesCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=cam_empty)

        # --- overlays inherited from the end of hat_tile_basics ---
        tri_carrier = Plane(name='KiteTriGridCarrier', location=[0, 0, 3],
                            rotation_euler=[pi / 2, pi / 6, 0])
        tri_mod = TriangularGridModifier(grid_n=8, colors=["red", "important"],
                                         show_fundamental_plane=False, dot_size=0.1 / 2)
        tri_carrier.add_mesh_modifier(type='NODES', node_modifier=tri_mod)
        tri_carrier.rescale(rescale=4 * r3, begin_time=0, transition_time=0)
        tri_carrier.move(direction=[3, 0, 4 / 3], begin_time=0, transition_time=0)
        tri_carrier.appear(begin_time=0, transition_time=0)

        sphere = Sphere(0.6, location=Vector([-3, 0, 4.8]), color='red')
        sphere.grow(begin_time=0, transition_time=0)

        hex_carrier = Plane(name='KiteHexGridCarrier', location=[0, 0, 6],
                            rotation_euler=[pi / 2, pi / 6, 0])
        hex_mod = HexagonalTilingModifier(iterations=30, with_edges=True, edge_length=1.0,
                                          colors=["joker", "example"], dot_size=0)
        hex_carrier.add_mesh_modifier(type='NODES', node_modifier=hex_mod)
        hex_carrier.rescale(rescale=8, begin_time=0, transition_time=0)
        hex_carrier.move(direction=[3, 8, 4 / 3], begin_time=0, transition_time=0)
        hex_carrier.appear(begin_time=0, transition_time=0)

        # --- the hat, this time built from its eight kites ---
        # standing pose = 90 deg about x then lifted, i.e. flat (x, y) -> (x, 0, y + Z0)
        Z0 = 3.05
        # shift = Vector([-r3,0,7])
        shift = Vector()
        kites, centre = _hat_kites(scale=2)
        kite_objs = []
        radial_dirs = []
        face = [0, 1, 2, 3]
        inv_face = [3, 2, 1, 0]
        for idx, k in enumerate(kites):
            kc = k.mean(axis=0)
            local = [(float(v[0] - kc[0]), 0.0, float(v[1] - kc[1])) for v in k]
            if idx in [0, 2, 4, 5]:
                f = inv_face
            else:
                f = face
            obj = BObject(mesh=create_mesh(local, faces=[f]),
                          name='HatKite%d' % idx, color='joker',
                          solid=0.12, roughness=0.05, metallic=0.2,
                          location=Vector((float(kc[0]), 0.0, float(kc[1] + Z0))) + shift, scale=1)
            obj.appear(begin_time=0, transition_time=0)
            kite_objs.append(obj)
            mx = 0
            max_dir = Vector()
            for p in local:
                for q in local:
                    dir = Vector(p) - Vector(q)
                    if dir.length > mx:
                        mx = dir.length
                        max_dir = dir.normalized()
            radial = max_dir
            radial_dirs.append(radial)
        t0 += 1.5

        # shrink sphere and grid thickness to concentrate on the kites
        edge_thickness_node = get_geometry_node_from_modifier(hex_mod, label="EdgeThickness")
        ibpy.change_default_value(edge_thickness_node, from_value=0.018, to_value=0.001, begin_time=t0,
                                  transition_time=1)
        edge_thickness_node2 = get_geometry_node_from_modifier(tri_mod, label="EdgeThickness")
        dot_size_node = get_geometry_node_from_modifier(tri_mod, label="DotRadius")
        ibpy.change_default_value(edge_thickness_node2, from_value=0.018, to_value=0.001, begin_time=t0,
                                  transition_time=1)
        ibpy.change_default_value(dot_size_node, from_value=0.05, to_value=0.01, begin_time=t0, transition_time=1)
        t0 = 0.5 + sphere.shrink(scale=0.1, begin_time=t0, transition_time=1)

        # --- rotate each kite around its longest axis ---

        for obj, d in zip(kite_objs, radial_dirs):
            obj.rotate(rotation_quaternion=Quaternion(d, pi), begin_time=t0, transition_time=2)

        # --- fade each kite into its own colour: a rainbow around the tile ---
        rainbow = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'important']
        for i, obj in enumerate(kite_objs):
            obj.change_color(rainbow[i], begin_time=t0 + 1, transition_time=1)

        self.t0 = t0 + 3

    def commandments(self):
        """Carve the 14 walking instructions into a self-built marble panel.

        ``create_instruction_table()`` is rendered to SVG and carved into a
        marble info panel by a :class:`CommandmentTableModifier` (geometry-nodes
        Mesh Boolean *difference*).  The panel is generated inside the modifier,
        so the modifier is simply applied to a plain carrier cube.

        The glyphs are revealed line-wise through the modifier's integer
        ``CurveCount`` input: the left half of the table (steps 1-7) top-to-
        bottom first, then the right half (steps 8-14).  Each line carves in
        0.1 s and a new line starts every second, while a spotlight tracks the
        line currently being carved.
        """
        t0 = 0
        _setup_render(hdri="kloppenheim_06_puresky_4k", engine="CYCLES", transparent=True, exposure=0.75)
        ibpy.set_camera_location(location=[0, -31, 0])
        cam_empty = EmptyCube(location=[0, 0, 0], name='CmdCamEmpty')
        cam_empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=cam_empty)

        set_alpha_composition()

        # ---- carrier cube: its geometry is ignored, the panel is built in the
        #      modifier; passing node_modifier registers the marble material ----
        cube = Cube(name='CommandmentPanel', rotation_euler=[pi / 2, 0, 0])
        mod = CommandmentTableModifier(create_instruction_table(), thickness=0.05, dark=0.13)
        cube.add_mesh_modifier(type='NODES', node_modifier=mod)
        t0 = 0.5 + cube.appear(begin_time=0, transition_time=0.5)
        curve_count = get_geometry_node_from_modifier(mod, "CurveCount")

        # ---- spotlight tracking the freshly carved line ----
        spot_target = EmptyCube(location=[-3, 0, 3], name='CmdSpotTarget')
        spot_target.appear(begin_time=0, transition_time=0)
        spot = SpotLight(location=[0, -13, 0], energy=8000,
                         target=spot_target, name='CmdSpot', spot_size=pi / 9, color="red", scale=[1.5, 0.2, 1])
        t0 = spot.appear(begin_time=t0, transition_time=0.5)
        t0 += 0.5

        # --- add further light ---
        point_empty = EmptyCube(location=[0, -1, 20], name="CmdPointTarget")
        point_empty.appear(begin_time=0, transition_time=0)
        light = PointLight(location=[-15, -20, 8], target=point_empty, exposure=2, energy=15000)
        light.appear(begin_time=0, transition_time=0)

        # ---- reveal line-wise: 8 lines left half, then 8 lines right half ----

        splines = [14, 24, 34, 42, 51, 62, 73, 81, 96, 107, 118, 131, 140, 149, 156, 166, 167]
        appear_times = [0.2, 1, 2, 3, 4, 5, 6, 7, 7.2, 8, 9, 10, 11, 12, 13, 14, 15]
        cube_positions = [
            Vector([-3.1, 0, 2.5]),
            Vector([-3.1, 0, 1.77]),
            Vector([-3.1, 0, 0.9]),
            Vector([-3.1, 0, 0.15]),
            Vector([-3.1, 0, -0.68]),
            Vector([-3.1, 0, -1.525]),
            Vector([-3.1, 0, -2.4]),
            Vector([-3.1, 0, -3.1]),
            Vector([3.1, 0, 2.5]),
            Vector([3.1, 0, 1.77]),
            Vector([3.1, 0, 0.9]),
            Vector([3.1, 0, 0.15]),
            Vector([3.1, 0, -0.68]),
            Vector([3.1, 0, -1.525]),
            Vector([3.1, 0, -2.4]),
            Vector([3.1, 0, -3.1]),
            Vector([0, 0, -3.8])
        ]

        last_count = 0
        for spline, time, pos in zip(splines, appear_times, cube_positions):
            change_default_integer(curve_count, from_value=last_count, to_value=spline,
                                   begin_time=t0 + time, transition_time=0.1)
            spot_target.move_to(target_location=pos, begin_time=t0 + time, transition_time=0.1)
            last_count = spline
        t0 += time + 0.5

        self.t0 = t0 + 1


if __name__ == '__main__':
    try:
        example = HatTileScene()
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

        resolution = [1920, 1080]
        if selected_scene == "intro_to_background":
            resolution = [1080, 1920]
        example.create(name=selected_scene, resolution=resolution, start_at_zero=True)
    except Exception:
        print_time_report()
        raise
