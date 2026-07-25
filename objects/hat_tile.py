import csv

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, sqrt, zeros, cumsum, array, cos, sin, roll

from appearance.textures import get_texture
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (
    Grid, InputValue, InstanceOnPoints, JoinGeometry,
    create_geometry_line, Position, SetPosition,
    SetMaterial, NamedAttribute, WireFrame,
    InputVector,
    TransformGeometry, RealizeInstances,
    CompareNode, InputInteger,
    MathNode, ExtrudeMesh, SampleNearest,
    SampleIndex, Reroute, ComplexMathNode, PointsToVertices, Points, Quadrilateral,
    FillCurve, StoredNamedAttribute, ScaleElements,
    GeometryToInstance, RotateInstances, TranslateInstances, ForEachZone, Index, SeparateGeometry, RepeatZone,
    InsidePolygon, IndexSwitch, Frame,
    MeshCircle, MeshToPoints, CombineXYZ, SceneTime, make_function, Switch, IcoSphere, InputMaterial, VectorMath,
    SortElements, Coord, OnRightNode,
    Points, MeshLine, GeometryToInstance, RealizeInstances, RotateInstances,
    InputBoolean, DeleteGeometry, BooleanMath,
    MeshToCurve, CurveCircle, CurveToMesh,
    CollectionInfo, SeparateXYZ, MeshBoolean,
    BoundingBox, SetShadeSmooth, CubeMesh, CylinderMesh, ResampleCurve, FilletCurve,
    AttributeStatistic,
)
from interface.ibpy import create_mesh, Vector, Matrix, get_material
from objects.bobject import BObject
from objects.polygon import Polygon
from objects.text import generate_expression, hashed_tex
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs
from utils.utils import to_vector

pi = pi
r3 = sqrt(3)
r2 = np.sqrt(2)

HAT_SCALE = 1.0 / (2.0 * r3)

custom_ops = {
    "ifv": {
        "type": Switch,
        "class_kwargs": {"input_type": "VECTOR"},
        "inputs": ("True", "False", "Switch"),
        "output": "Output",
        "label": "ifv",
    },
    "iff": {
        "type": Switch,
        "class_kwargs": {"input_type": "FLOAT"},
        "inputs": ("True", "False", "Switch"),
        "output": "Output",
        "label": "iff",
    },
    "ifi": {
        "type": Switch,
        "class_kwargs": {"input_type": "INT"},
        "inputs": ("True", "False", "Switch"),
        "output": "Output",
        "label": "ifi",
    },
}


# ---------------------------------------------------------------------------
# Hat tile vertex computation (tileVertexList from Mathematica)
# ---------------------------------------------------------------------------

def _hat_vertices(dir_in=0, ref=False, pt=9, scale=1.0):
    """Return (13, 2) float array of hat-tile vertices.

    Parameters
    ----------
    dir_in : int, 0-5 — orientation (multiples of 60°)
    ref    : bool     — True for the reflected (anti-hat) variant
    pt     : int      — vertex-sequence offset (9 keeps anchor at origin)
    scale  : float    — uniform scale factor
    """
    d = 2 * dir_in

    # 13-edge walk around the hat polykite
    raw = [
        (r3, d - 1),
        (r3, d + 1),
        (1.0, d + 4),
        (1.0, d + 6),
        (r3, d + 3),
        (r3, d + 5),
        (1.0, d + 8),
        (1.0, d + 6),
        (r3, d + 9),
        (r3, d + 7),
        (1.0, d + 10),
        (2.0, d + 12),
        (1.0, d + 14),
    ]
    # RotateLeft by pt (Mathematica: first pt elements go to end)
    edges_rot = raw[pt:] + raw[:pt]

    if ref:
        # Reverse and negate direction indices (mod 12)
        edges = [(edge_length, int((-rotation) % 12)) for edge_length, rotation in reversed(edges_rot)]
    else:
        edges = [(edge_length, int(rotation % 12)) for edge_length, rotation in edges_rot]

    # e[len, x] → len · (cos π(x+1)/6, sin π(x+1)/6)
    displacements = array([
        [edge_length * cos(pi * (rotation + 1) / 6.0),
         edge_length * sin(pi * (rotation + 1) / 6.0)]
        for edge_length, rotation in edges
    ])

    # Cumulative sum from origin → 14 vertices
    vertices = zeros((14, 2))
    vertices[1:] = cumsum(displacements, axis=0)

    # Most[vertices] (remove closing duplicate) → 13 vertices
    vertices = vertices[:-1]

    # RotateRight by pt (last pt elements go to front)
    vertices = roll(vertices, pt, axis=0)

    return vertices * scale


def _hat_vertices14(dir_in=0, ref=False, pt=0, scale=1.0):
    """
    Return (14, 2) float array of hat-tile vertices. A fourteenth vertex is added for convenience
    This choice of vertices is adjusted for the substitution computations from H7H8 supertiles
    Parameters
    ----------
    dir_in : int, 0-11 — orientation (multiples of 30°)
    ref    : bool     — True for the reflected (anti-hat) variant
    pt     : int      — vertex-sequence offset (0 keeps anchor at origin)
    scale  : float    — uniform scale factor
    """
    d = dir_in

    # 14-edge walk around the hat polykite
    raw = [
        (r3, d - 1),
        (r3, d + 1),
        (1.0, d + 4),
        (1.0, d + 6),
        (r3, d + 3),
        (r3, d + 5),
        (1.0, d + 8),
        (1.0, d + 6),
        (r3, d + 9),
        (r3, d + 7),
        (1.0, d + 10),
        (1.0, d + 12),
        (1.0, d + 12),
        (1.0, d + 14),
    ]
    # RotateLeft by pt (Mathematica: first pt elements go to end)
    edges_rot = raw[pt:] + raw[:pt]

    if ref:
        # Reverse and negate direction indices (mod 12)
        edges = [(edge_length, int((-rotation) % 12)) for edge_length, rotation in reversed(edges_rot)]
    else:
        edges = [(edge_length, int(rotation % 12)) for edge_length, rotation in edges_rot]

    # e[len, x] → len · (cos π(x+1)/6, sin π(x+1)/6)
    displacements = array([
        [edge_length * cos(pi * (rotation + 1) / 6.0),
         edge_length * sin(pi * (rotation + 1) / 6.0)]
        for edge_length, rotation in edges
    ])

    # Cumulative sum from origin → 14 vertices
    vertices = zeros((15, 2))
    vertices[1:] = cumsum(displacements, axis=0)

    # Most[vertices] (remove closing duplicate) → 14 vertices
    vertices = vertices[:-1]

    # RotateRight by pt (last pt elements go to front)
    vertices = roll(vertices, pt, axis=0)

    return vertices * scale


def _hat_vertices_3d(rotation=0, ref=False, hat_scale=HAT_SCALE, pivot=0):
    """Return list of Vector-like (x, y, 0) tuples for use as Blender vertices."""
    verts2d = _hat_vertices(dir_in=rotation, ref=ref, scale=hat_scale, pt=(9 + pivot) % 13)
    return [(v[0], v[1], 0.0) for v in verts2d]


def code_to_hat14(direction, ref, pt):
    """Translate a substitution/notebook code into ``_hat_vertices14`` arguments.

    The Mathematica notebook (and ``video_hat_tile/substitution_explainer.py``,
    plus the ``.dat``/``.csv`` exports) labels a hat by ``(pos, dir, ref, pt)``
    with ``dir`` in units of 30 degrees and a **1-indexed** pivot ``pt``.
    ``_hat_vertices14`` takes a **0-indexed** pivot and a ``dir_in`` that is
    offset by one 30-degree step -- with the sign of that offset depending on
    ``ref``, because the two conventions mirror about different axes (this
    module mirrors the edge-direction indices, i.e. about the 30-degree line;
    the notebook mirrors ``x``, i.e. about the y-axis).

    Returns ``(dir_in, pt0)``.
    """
    dir_in = (direction + 1) % 12 if ref else (direction - 1) % 12
    return dir_in, pt - 1


def hat_vertices_from_code(pos=(0, 0), direction=0, ref=False, pt=1, scale=1.0):
    """The 14 absolute ``(x, y)`` vertices of the hat given by a notebook code."""
    dir_in, pt0 = code_to_hat14(direction, ref, pt)
    origin = array(pos, dtype=float)
    return [(v + origin) * scale
            for v in _hat_vertices14(dir_in=dir_in, ref=ref, pt=pt0)]


# ---------------------------------------------------------------------------
# Basic BObject
# ---------------------------------------------------------------------------

class HatTile(BObject):
    def __init__(self, name="HatTile", **kwargs):
        vertices2d = get_from_kwargs(kwargs, "vertices2d", None)
        rotation = get_from_kwargs(kwargs, "rotation", 0)
        reflection = get_from_kwargs(kwargs, "reflection", False)
        hat_scale = get_from_kwargs(kwargs, "hat_scale", 1)
        location = get_from_kwargs(kwargs, "location", (0, 0))
        pivot = get_from_kwargs(kwargs, "pivot", 0)
        hat_scale = HAT_SCALE * hat_scale
        if len(location) == 2:
            location = Vector([*location, 0])
        if vertices2d is None:
            verts = _hat_vertices_3d(rotation, reflection, hat_scale, pivot)
        else:
            verts = [(float(v[0]), float(v[1]), 0.0) for v in vertices2d]
        super().__init__(
            mesh=create_mesh(verts, faces=[list(range(len(verts)))]),
            name=name, location=location, **kwargs)

    @classmethod
    def from_code(cls, code=((0, 0), 0, False, 0), **kwargs):
        """
        Compatability method that lays out hat tiles according to this module's
        own 13-vertex code: ``(location, rotation, reflection, pivot)`` with
        ``rotation`` in units of 60 degrees and a 0-indexed ``pivot`` (0..12).

        For the codes produced by ``substitution_explainer`` (30-degree
        ``dir``, 1-indexed ``pt``) use :meth:`from_substitution_code`.
        """
        return HatTile(location=code[0], rotation=code[1], reflection=code[2], pivot=code[3], **kwargs)

    @classmethod
    def from_substitution_code(cls, code=((0, 0), 0, False, 1), scale=1.0, **kwargs):
        """Build a hat from a ``substitution_explainer`` code ``(pos, dir, ref, pt)``.

        ``dir`` is in units of 30 degrees and ``pt`` is 1-indexed, exactly as
        emitted by ``substitute()`` and by the ``.dat``/``.csv`` exports.  The
        mesh carries the 14-vertex outline anchored at its pivot and the object
        sits at the (scaled) code position, so the tile stays independently
        movable.  ``scale`` is applied to the outline *and* the position, so a
        whole cluster scales rigidly.
        """
        pos, direction, ref, pt = code
        dir_in, pt0 = code_to_hat14(direction, ref, pt)
        verts = _hat_vertices14(dir_in=dir_in, ref=ref, pt=pt0, scale=scale)
        location = Vector((float(pos[0]) * scale, float(pos[1]) * scale, 0.0))
        return cls(vertices2d=verts, location=location, **kwargs)


# --------------------------------------------
# Modifier
# --------------------------------------------

class LabbeSelingerModifier(GeometryNodesModifier):
    """
    This modifier implements the IFS approach for tiling  the plane with hats put forward in
    https://arxiv.org/abs/2604.20964
    _generate_hats(): All twelve possible orientations of the hat are generated
    """

    def __init__(self, **kwargs):
        super().__init__('LabbeSelingerModifier', automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        links = tree.links
        out = self.group_outputs
        grid_size = get_from_kwargs(kwargs, "grid_size", 0)
        # move group output far away
        right = 65
        out.location = (right * 200, 0)

        # Coord(tree, min=[30, -20], max=[right, 20])
        hats = self._generate_hats(tree, location=(43, 3), **kwargs)
        tri_grid, tri_grid_vertices = self._generate_grid(tree, grid_size, location=(-20, 10), **kwargs)

        # prepare fundamental domain and trapezoid cover
        (one, zero, xi_function, phi_function, u_function, v) = self._create_constants(tree, location=(-20, 3))
        g_trapzoid = self._create_trapezoid(tree, zero, one, xi_function, phi_function, location=(-18, 3))
        cover, trapezoids = self._make_cover(tree, g_trapzoid, u_function, v, location=(-14, 2))

        # setup backtranslation
        back = self._back_translation(tree, tri_grid, u_function, v, location=(-13, 5))

        sample_points = self._sample_points_in_cover(tree, cover, back, location=(-9, 5))

        for_each = ForEachZone(tree, location=(3, 3), node_width=57, geometry=sample_points.geometry_out)

        # trapezoid selector
        trap_selector, trap_index = self._trapezoid_selector(tree, trapezoids, for_each.element, location=(4, 3))
        # prepare orientations
        oriented_points = self._setup_orientations(tree, for_each.element, trap_index, location=(7, 8))
        # trapezoid substitution
        g_substitution_trap, trap1, trap2, trap3, para = self._substitution_trapezoid(tree, trap_selector.std_out,
                                                                                      phi_function, location=(7, 1))
        # sample sub structure of trapezoid
        sample_repeat = self._sample_substructure(tree, oriented_points, g_substitution_trap, location=(16, 10))
        # inverse transformations
        inverse_transforms = self._inverse_transforms(tree, trap1, trap2, para, trap3, location=(26, 2))
        # ifs
        ifs_result = self._apply_ifs(tree, sample_repeat, inverse_transforms, g_substitution_trap, location=(33, 10))
        # place hats
        hat_config = self._hat_config(tree, ifs_result, hats, location=(48, 7))

        # ifs_result and outer repeat
        links.new(hat_config.geometry_out, for_each.foreach_output.inputs[1])

        final_join = self._finalize(tree, hat_config, location=(52, 0))

        grid_points = self._grid_point_visualization(tree, tri_grid_vertices, location=(53, 5))
        create_geometry_line(tree, [grid_points, final_join])

        links.new(final_join.geometry_out, for_each.foreach_output.inputs["Geometry"])

        create_geometry_line(tree, [for_each], out=out.inputs[0])
    def _grid_point_visualization(self, tree, ins, location):
        (x, y) = location

        grid_radius = InputValue(tree, location=(x, y - 2), hide=False, name="GridRadius", value=0.025)
        compare = CompareNode(tree,location=(x+0.5,y-3),hide=True,data_type="FLOAT",inputs0=grid_radius.std_out,inputs1=0,inputs2=0.001)
        del_geo = DeleteGeometry(tree,location=(x+0.5,y-2),hide=True,selection=compare.std_out)

        sphere = IcoSphere(tree, location=(x, y - 1), name="GridSphere", radius=grid_radius.std_out, subdivisions=3,hide=True)
        iop = InstanceOnPoints(tree, location=(x + 1, y), hide=True, instance=sphere.geometry_out)

        mat = get_texture(material="red", roughness=0.1, emission=1)
        self.materials.append(mat)
        set_material = SetMaterial(tree, location=(x + 2, y), hide=True, material=mat)

        create_geometry_line(tree, [ins,del_geo, iop, set_material])

        frame = Frame(tree, location=location, name="Grid Point Visualization")
        frame.add([grid_radius, compare, del_geo, sphere, iop, set_material])
        return set_material

    def _finalize(self, tree, ins, location):
        (x, y) = location
        links = tree.links
        center = NamedAttribute(tree, location=(x, y - 2), hide=True, name="OldPosition", data_type="FLOAT_VECTOR")
        hat_scale = InputValue(tree, location=(x, y - 3), hide=True, value=0.95, name="HatScale")
        extrude_scale = InputValue(tree,location=(x,y-4),hide=True,value=0.01,name="ExtrudeScale")
        scale = ScaleElements(tree, location=(x, y), hide=True, scale=hat_scale.std_out, center=center.std_out)
        extrude = ExtrudeMesh(tree, location=(x + 1, y), mode="FACES", offset_scale=extrude_scale.std_out, hide=True)

        join = JoinGeometry(tree, location=(x + 2, y), hide=True)

        # --- IndexSwitch Material: 12 slots ---
        orientation = NamedAttribute(tree, location=(x + 2, y - 1), name="Orientation",
                                     data_type="INT", hide=True)
        sample_index = SampleIndex(tree, location=(x + 2, y), name="SampleOrientation",
                                   hide=True, value=orientation.std_out, data_type="INT",
                                   geometry=ins.geometry_out, domain="INSTANCE")
        isw = IndexSwitch(tree, data_type='MATERIAL',
                          location=(x + 3, y), index=sample_index.std_out)
        for _ in range(10):
            isw.new_item()
        count = 0
        colors = []
        self.hat_frame_materials = []
        for i in range(0, 2):
            for k in range(0, 6):
                material = get_texture(material="hat" + str(i) + str(k), emission=0.01)
                self.materials.append(material)
                self.hat_frame_materials.append(material)
                color = InputMaterial(tree, location=(x + 2, y - 2 - i * 3 - k * 0.5),
                                      material=material, hide=True)

                colors.append(color)
                links.new(color.std_out, isw.slots[count + 1])
                count += 1

        wireframe = WireFrame(tree, location=(x + 4, y), radius=0.01, hide=True)
        mat = SetMaterial(tree, material=isw.std_out,
                          hide=True, location=(x + 5, y))
        join2 = JoinGeometry(tree, location=(x + 6, y), hide=True)
        create_geometry_line(tree, [ins, scale, join])
        create_geometry_line(tree, [scale, extrude, join])
        create_geometry_line(tree, [join, wireframe, mat, join2])
        create_geometry_line(tree, [join, join2])
        frame = Frame(tree, location=(x, y), name="Finalize")
        frame.add(colors + [sample_index, isw, orientation, scale, extrude, join, wireframe, mat, hat_scale, center])
        return join2

    def _hat_config(self, tree, point, hats, location):
        (x, y) = location
        links = tree.links

        exit_type = NamedAttribute(tree, location=(x, y), data_type="INT", name="Type", hide=True)
        bottom = NamedAttribute(tree, location=(x, y - 1), data_type="INT", name="Bottom", hide=True)
        top = NamedAttribute(tree, location=(x, y - 2), data_type="INT", name="Top", hide=True)

        sample_type = SampleIndex(tree, location=(x + 1, y), data_type="INT", geometry=point.geometry_out,
                                  value=exit_type.std_out, hide=True, index=0)
        sample_bottom = SampleIndex(tree, location=(x + 1, y - 1), data_type="INT", geometry=point.geometry_out,
                                    value=bottom.std_out, hide=True, index=0)
        sample_top = SampleIndex(tree, location=(x + 1, y - 2), data_type="INT", geometry=point.geometry_out,
                                 value=top.std_out, hide=True, index=0)

        orientation_switch = IndexSwitch(tree, location=(x + 1, y), data_type="INT", name="Orientation", hide=True,
                                         index=sample_type.std_out)
        orientation_switch.new_item()
        links.new(sample_bottom.std_out, orientation_switch.slots[2])
        links.new(sample_top.std_out, orientation_switch.slots[3])

        # link orientation to hat switch
        links.new(orientation_switch.std_out, hats.inputs[0])
        store_orientation = StoredNamedAttribute(tree, location=(x + 2, y), hide=True,
                                                 name="Orientation", domain="POINT",
                                                 value=orientation_switch.std_out)
        iop = InstanceOnPoints(tree, location=(x + 3, y), instance=hats.std_out)
        create_geometry_line(tree, [point, store_orientation, iop])
        frame = Frame(tree, location=(x, y), name="Hat Placement")
        frame.add([store_orientation, exit_type, bottom, top, orientation_switch, iop])
        return iop

    def _apply_ifs(self, tree, g_sample, inverse_transforms, g_trapez, location=(0, 0)):
        (x, y) = location
        links = tree.links

        rr_trap = Reroute(tree, location=(x, y - 3), ins=g_trapez.geometry_out, hide=True, name="Trapez")
        repeat = RepeatZone(tree, location=(x, y), node_width=13, iterations=10, geometry=g_sample.geometry_out)
        # transform point according to region
        set_pos = SetPosition(tree, location=(x + 1, y), geometry=repeat.repeat_input.outputs["Geometry"],
                              position=inverse_transforms.std_out)

        # adjust region of point according to new position
        # Find the nearest analyser face for the current point.
        position = Position(tree, location=(x + 1, y - 3), hide=True)
        sample_nearest = SampleNearest(
            tree, location=(x + 2, y - 2), domain="FACE",
            geometry=rr_trap.geometry_out,
            sample_position=position.std_out)

        polygon_attr = NamedAttribute(
            tree, location=(x + 3, y - 2),
            data_type="INT", name="PolygonType")

        sample_index = SampleIndex(
            tree, location=(x + 4, y - 1),
            data_type="INT", domain="FACE",
            geometry=rr_trap.geometry_out,
            value=polygon_attr.std_out,
            index=sample_nearest.std_out)

        store_type = StoredNamedAttribute(
            tree, location=(x + 5, y),
            data_type="INT", domain="POINT",
            name="Type", value=sample_index.std_out)

        # Track Bottom/Top/Left transform labels through each repeat iteration
        type_attr_ifs = NamedAttribute(tree, location=(x + 5, y - 1.9), data_type="INT", name="Type", hide=True)
        old_top_attr_ifs = NamedAttribute(tree, location=(x + 5, y - 0.5), data_type="INT", name="OldTop", hide=True)
        old_left_attr_ifs = NamedAttribute(tree, location=(x + 5.5, y - 1.2), data_type="INT", name="OldLeft",
                                           hide=True)
        old_bottom_attr_ifs = NamedAttribute(tree, location=(x + 5.5, y - 0.1), data_type="INT", name="OldBottom",
                                             hide=True)

        bottom_transform_ifs = IndexSwitch(tree, location=(x + 7, y - 2.1), data_type="INT",
                                           index=type_attr_ifs.std_out)
        for _ in range(5):
            bottom_transform_ifs.new_item()
        for slot in [1, 2, 3, 6, 7]:
            links.new(old_bottom_attr_ifs.std_out, bottom_transform_ifs.slots[slot])
        links.new(old_left_attr_ifs.std_out, bottom_transform_ifs.slots[4])
        links.new(old_top_attr_ifs.std_out, bottom_transform_ifs.slots[5])

        top_transform_ifs = IndexSwitch(tree, location=(x + 8.5, y - 1.8), data_type="INT",
                                        index=type_attr_ifs.std_out)
        for _ in range(5):
            top_transform_ifs.new_item()
        for slot in [1, 2, 3, 6, 7]:
            links.new(old_top_attr_ifs.std_out, top_transform_ifs.slots[slot])
        links.new(old_bottom_attr_ifs.std_out, top_transform_ifs.slots[4])
        links.new(old_bottom_attr_ifs.std_out, top_transform_ifs.slots[5])

        left_transform_ifs = IndexSwitch(tree, location=(x + 10, y - 1.6), data_type="INT",
                                         index=type_attr_ifs.std_out)
        for _ in range(5):
            left_transform_ifs.new_item()
        for slot in [1, 2, 3, 6]:
            links.new(old_left_attr_ifs.std_out, left_transform_ifs.slots[slot])
        links.new(old_bottom_attr_ifs.std_out, left_transform_ifs.slots[4])
        links.new(old_bottom_attr_ifs.std_out, left_transform_ifs.slots[5])
        links.new(old_top_attr_ifs.std_out, left_transform_ifs.slots[7])

        store_bottom_ifs = StoredNamedAttribute(tree, location=(x + 6.5, y - 3.9), data_type="INT",
                                                domain="POINT", name="Bottom",
                                                value=bottom_transform_ifs.std_out, hide=False)
        store_top_ifs = StoredNamedAttribute(tree, location=(x + 7.5, y - 3.7), data_type="INT",
                                             domain="POINT", name="Top",
                                             value=top_transform_ifs.std_out, hide=False)
        store_left_ifs = StoredNamedAttribute(tree, location=(x + 8.5, y - 3.8), data_type="INT",
                                              domain="POINT", name="Left",
                                              value=left_transform_ifs.std_out, hide=False)

        # Read back and store as OldBottom/OldTop/OldLeft for next iteration
        bottom_attr2_ifs = NamedAttribute(tree, location=(x + 8.5, y - 4.9), data_type="INT",
                                          name="Bottom", hide=True)
        top_attr2_ifs = NamedAttribute(tree, location=(x + 9.5, y - 4.8), data_type="INT",
                                       name="Top", hide=True)
        left_attr2_ifs = NamedAttribute(tree, location=(x + 10.5, y - 4.9), data_type="INT",
                                        name="Left", hide=True)
        store_old_bottom_ifs = StoredNamedAttribute(tree, location=(x + 9.5, y - 3.9), data_type="INT",
                                                    domain="POINT", name="OldBottom",
                                                    value=bottom_attr2_ifs.std_out, hide=False)
        store_old_top_ifs = StoredNamedAttribute(tree, location=(x + 10.5, y - 4.0), data_type="INT",
                                                 domain="POINT", name="OldTop",
                                                 value=top_attr2_ifs.std_out, hide=False)
        store_old_left_ifs = StoredNamedAttribute(tree, location=(x + 11.5, y - 3.9), data_type="INT",
                                                  domain="POINT", name="OldLeft",
                                                  value=left_attr2_ifs.std_out, hide=False)

        frame = Frame(tree, location=(x, y), name="Iterated Function System")
        repeat.create_geometry_line([set_pos, store_type,
                                     store_bottom_ifs, store_top_ifs, store_left_ifs,
                                     store_old_bottom_ifs, store_old_top_ifs, store_old_left_ifs])

        attr_old_pos = NamedAttribute(tree, location=(x + 13, y), name="OldPosition",
                                      data_type="FLOAT_VECTOR", hide=True)
        reset_position = SetPosition(tree, location=(x + 14, y), position=attr_old_pos.std_out, hilde=True,
                                     name="ResetPosition")
        create_geometry_line(tree, [repeat, reset_position])
        frame.add(
            [rr_trap, repeat, set_pos, polygon_attr, sample_nearest, sample_index, store_type,
             type_attr_ifs, old_top_attr_ifs, old_left_attr_ifs, old_bottom_attr_ifs,
             bottom_transform_ifs, top_transform_ifs, left_transform_ifs,
             store_bottom_ifs, store_top_ifs, store_left_ifs,
             bottom_attr2_ifs, top_attr2_ifs, left_attr2_ifs,
             store_old_bottom_ifs, store_old_top_ifs, store_old_left_ifs,
             attr_old_pos, reset_position])

        return reset_position

    def _inverse_transforms(self, tree, trap1, trap2, para, trap3, location=(0, 0)):
        (x, y) = location
        links = tree.links

        # vertex sampling
        pos = Position(tree, location=(x, y - 5), hide=True)
        rr_trap1 = Reroute(tree, location=(x + 0.5, y - 1), name="trap1", ins=trap1.geometry_out)
        rr_trap2 = Reroute(tree, location=(x + 0.5, y - 3), name="trap2", ins=trap2.geometry_out)
        rr_para = Reroute(tree, location=(x + 0.5, y - 5), name="para", ins=para.geometry_out)
        rr_trap3 = Reroute(tree, location=(x + 0.5, y - 7), name="trap3", ins=trap3.geometry_out)

        connections = [rr_trap1, rr_trap2, rr_para, rr_trap3]
        vertex_sample_nodes = []
        for c, con in enumerate(connections):
            for i in range(4):
                si = SampleIndex(tree, location=(x + 1, y - 2 * c - 0.5 * i), name="v" + str(i + 1), index=i,
                                 value=pos.std_out,
                                 domain="POINT", data_type="FLOAT_VECTOR", hide=True, geometry=con.geometry_out)
                vertex_sample_nodes.append(si)

        (r_60, r_120, r_180) = self._create_rotations(tree, location=(x + 2, y - 1))
        (f_xi, f_phi) = self._create_constants2(tree, location=(x + 2, y - 3))

        f_k = make_function(
            tree, name="k-Scale",
            functions={"k": "phi,1,-,phi,/"},
            inputs=["phi"], outputs=["k"],
            scalars=["phi", "k"],
            hide=True, location=(x + 3, y - 6))
        links.new(f_phi.outputs["phi"], f_k.inputs["phi"])

        # compare with notes in IFS_Shortcut.nb
        transforms = {
            3: "rot120,1,k,/,scale,p,v4,csub,cmul,v1,cadd",
            4: "p,v4,csub,v2,v1,csub,1,phi,phi,*,/,scale,csub,-1,k,/,scale,v1,cadd",
            5: "rot60,1,k,/,scale,p,v2,csub,v1,v2,csub,1,phi,/,scale,csub,cmul,v2,cadd",
            6: "p,v1,v2,csub,1,phi,/,scale,cadd,v1,csub,1,k,/,scale,v1,cadd",
        }

        position = Position(tree, location=(x + 4, y + 1))
        trafos = {}
        t = 0
        for key, val in transforms.items():
            trafo = make_function(tree, name="invTrafo" + str(key),
                                  functions={"p": val},
                                  inputs=["p", "one", "phi", "xi", "rot60", "rot120", "k", "v1", "v2", "v3", "v4"],
                                  outputs=["p"], scalars=["k", "phi"],
                                  vectors=["p", "rot60", "rot120", "v1", "v2", "v3", "v4"],
                                  location=(x + 5, y - 2 * (key - 3)))
            links.new(f_phi.outputs["phi"], trafo.inputs["phi"])
            links.new(r_60.outputs["rot60"], trafo.inputs["rot60"])
            links.new(r_120.outputs["rot120"], trafo.inputs["rot120"])
            links.new(position.std_out, trafo.inputs["p"])
            links.new(f_k.outputs["k"], trafo.inputs["k"])
            trafos[key] = trafo.outputs["p"]
            for i in range(4):
                links.new(vertex_sample_nodes[i + 4 * t].std_out, trafo.inputs["v" + str(i + 1)])
            t += 1

        attr_type = NamedAttribute(tree, location=(x + 6, y + 1), data_type="INT", domain="POINT", name="Type",
                                   hide=True)
        switch = IndexSwitch(tree, location=(x + 6, y), data_type="VECTOR", index=attr_type.std_out)

        # the first three slots remain untransformed
        for _ in range(7 - 2):
            switch.new_item()
        slots = [position.std_out, position.std_out, position.std_out, trafos[3], trafos[4], trafos[5], trafos[6]]
        for slot_idx, slot in enumerate(slots):
            links.new(slot, switch.slots[slot_idx + 1])  # the zero slot is for the index variable

        frame = Frame(tree, location=(x, y), name="InverseTransformations")
        frame.add(
            [r_60, r_120, r_180] + vertex_sample_nodes + [pos, rr_trap1, rr_trap2, rr_trap3, rr_para, f_xi, f_phi, f_k,
                                                          r_60, r_120, r_180, position, *trafos.values(), attr_type,
                                                          switch])

        return switch

    def _setup_orientations(self, tree, sample, trapezoid, location):
        # depending on the choice of the trapezoid three different hat orientations are setup

        (x, y) = location
        links = tree.links

        bottom_switch = IndexSwitch(tree, location=(x + 1, y - 1), data_type="INT", domain="POINT",
                                    name="BottomOrientation", hide=True, index=trapezoid.std_out)
        top_switch = IndexSwitch(tree, location=(x + 3, y - 1), data_type="INT", domain="POINT", name="TopOrientation",
                                 hide=True, index=trapezoid.std_out)
        left_switch = IndexSwitch(tree, location=(x + 5, y - 1), data_type="INT", domain="POINT",
                                  name="LeftOrientation", hide=True, index=trapezoid.std_out)
        old_bottom = StoredNamedAttribute(tree, location=(x + 2, y), name="OldBottom", data_type="INT", domain="POINT",
                                          hide=True, value=bottom_switch.std_out)
        old_top = StoredNamedAttribute(tree, location=(x + 4, y), name="OldTop", data_type="INT", domain="POINT",
                                       hide=True, value=top_switch.std_out)
        old_left = StoredNamedAttribute(tree, location=(x + 6, y), name="OldLeft", data_type="INT", domain="POINT",
                                        hide=True, value=left_switch.std_out)
        switches = [bottom_switch, top_switch, left_switch]
        orientations = {
            "BottomOrientation": [1, 2, 3, 4, 5, 0, 1, 2, 4, 5],
            "TopOrientation": [4, 5, 0, 1, 2, 3, 4, 5, 1, 2],
            "LeftOrientation": [10, 11, 6, 7, 8, 9, 10, 11, 7, 8]
        }

        for switch in switches:
            for _ in range(8):
                switch.new_item()
            for s in range(10):
                switch.slots[s + 1].default_value = orientations[switch.name][s]

        frame = Frame(tree, location=(x, y), name="Setup Orientation")
        frame.add([old_bottom, old_top, old_left, bottom_switch, left_switch, top_switch])
        create_geometry_line(tree, [old_bottom, old_top, old_left], ins=sample)
        last = old_left

        return last

    def _sample_substructure(self, tree, sample, g_substructure_trap, location):
        (x, y) = location
        links = tree.links

        frame = Frame(tree, location=(x, y), name="Sample Substructure of Trapezoid")

        position = Position(tree, location=(x, y + 0.5), hide=True)
        foreach = ForEachZone(tree, location=(x, y), domain="POINT",
                              node_width=11, geometry=sample.geometry_out)
        foreach.add_socket(socket_type="VECTOR", name="Position",
                           value=position.std_out, for_input=True)

        rr_substitution = Reroute(tree, location=(x + 1, y - 4), name="substitution",
                                  ins=g_substructure_trap.geometry_out)
        # Find the nearest analyser face for the current point.
        sample_nearest = SampleNearest(
            tree, location=(x + 1, y - 5), domain="FACE",
            geometry=rr_substitution.geometry_out,
            sample_position=foreach.foreach_input.outputs["Position"], hide=True)

        polygon_attr = NamedAttribute(
            tree, location=(x + 1, y - 3),
            data_type="INT", name="PolygonType", hide=True)

        sample_index = SampleIndex(
            tree, location=(x + 2, y - 4),
            data_type="INT", domain="FACE",
            geometry=rr_substitution.geometry_out,
            value=polygon_attr.std_out,
            index=sample_nearest.std_out, hide=True)

        store_type = StoredNamedAttribute(
            tree, location=(x + 3, y),
            data_type="INT", domain="POINT",
            name="Type", value=sample_index.std_out, hide=True)

        # Track Bottom/Top/Left transform labels per point through the IFS iteration
        type_attr_fs = NamedAttribute(tree, location=(x + 2, y - 2.2), data_type="INT", name="Type", hide=True)
        old_bottom_attr_fs = NamedAttribute(tree, location=(x + 2, y - 0.1), data_type="INT", name="OldBottom",
                                            hide=True)
        old_top_attr_fs = NamedAttribute(tree, location=(x + 1.6, y - 0.8), data_type="INT", name="OldTop", hide=True)
        old_left_attr_fs = NamedAttribute(tree, location=(x + 1.8, y - 1.5), data_type="INT", name="OldLeft", hide=True)

        # BottomTransform: type→which-old-slot becomes the new Bottom label
        bottom_transform_fs = IndexSwitch(tree, location=(x + 3.6, y - 2.4), data_type="INT",
                                          index=type_attr_fs.std_out)
        for _ in range(5):
            bottom_transform_fs.new_item()
        for slot in [1, 2, 3, 6, 7]:
            links.new(old_bottom_attr_fs.std_out, bottom_transform_fs.slots[slot])
        links.new(old_left_attr_fs.std_out, bottom_transform_fs.slots[4])
        links.new(old_top_attr_fs.std_out, bottom_transform_fs.slots[5])

        # TopTransform
        top_transform_fs = IndexSwitch(tree, location=(x + 4.8, y - 2.0), data_type="INT",
                                       index=type_attr_fs.std_out, hide=True)
        for _ in range(5):
            top_transform_fs.new_item()
        for slot in [1, 2, 3, 6, 7]:
            links.new(old_top_attr_fs.std_out, top_transform_fs.slots[slot])
        links.new(old_bottom_attr_fs.std_out, top_transform_fs.slots[4])
        links.new(old_bottom_attr_fs.std_out, top_transform_fs.slots[5])

        # LeftTransform
        left_transform_fs = IndexSwitch(tree, location=(x + 6.1, y - 1.8), data_type="INT",
                                        index=type_attr_fs.std_out, hide=True)
        for _ in range(5):
            left_transform_fs.new_item()
        for slot in [1, 2, 3, 6]:
            links.new(old_left_attr_fs.std_out, left_transform_fs.slots[slot])
        links.new(old_bottom_attr_fs.std_out, left_transform_fs.slots[4])
        links.new(old_bottom_attr_fs.std_out, left_transform_fs.slots[5])
        links.new(old_top_attr_fs.std_out, left_transform_fs.slots[7])

        store_bottom_fs = StoredNamedAttribute(tree, location=(x + 5.1, y - 3.9), data_type="INT",
                                               domain="POINT", name="Bottom",
                                               value=bottom_transform_fs.std_out, hide=True)
        store_top_fs = StoredNamedAttribute(tree, location=(x + 6.1, y - 3.8), data_type="INT",
                                            domain="POINT", name="Top",
                                            value=top_transform_fs.std_out, hide=True)
        store_left_fs = StoredNamedAttribute(tree, location=(x + 7.0, y - 3.8), data_type="INT",
                                             domain="POINT", name="Left",
                                             value=left_transform_fs.std_out, hide=True)

        # Read back the new Bottom/Top/Left and store as OldBottom/OldTop/OldLeft for next iteration
        bottom_attr2_fs = NamedAttribute(tree, location=(x + 7.0, y - 4.8), data_type="INT",
                                         name="Bottom", hide=True)
        top_attr2_fs = NamedAttribute(tree, location=(x + 7.8, y - 4.7), data_type="INT",
                                      name="Top", hide=True)
        left_attr2_fs = NamedAttribute(tree, location=(x + 8.7, y - 4.7), data_type="INT",
                                       name="Left", hide=True)
        store_old_bottom_fs = StoredNamedAttribute(tree, location=(x + 7.8, y - 3.8), data_type="INT",
                                                   domain="POINT", name="OldBottom",
                                                   value=bottom_attr2_fs.std_out, hide=True)
        store_old_top_fs = StoredNamedAttribute(tree, location=(x + 8.6, y - 3.8), data_type="INT",
                                                domain="POINT", name="OldTop",
                                                value=top_attr2_fs.std_out, hide=True)
        store_old_left_fs = StoredNamedAttribute(tree, location=(x + 9.5, y - 3.8), data_type="INT",
                                                 domain="POINT", name="OldLeft",
                                                 value=left_attr2_fs.std_out, hide=True)

        foreach.create_geometry_line([store_type, store_bottom_fs, store_top_fs, store_left_fs,
                                      store_old_bottom_fs, store_old_top_fs, store_old_left_fs])

        frame.add([position, foreach, sample_nearest, polygon_attr, sample_index,
                   store_type, type_attr_fs, old_bottom_attr_fs, old_top_attr_fs, old_left_attr_fs,
                   bottom_transform_fs, top_transform_fs, left_transform_fs,
                   store_bottom_fs, store_top_fs, store_left_fs,
                   bottom_attr2_fs, top_attr2_fs, left_attr2_fs,
                   store_old_bottom_fs, store_old_top_fs, store_old_left_fs])

        return foreach

    def _trapezoid_selector(self, tree, trapezoids, sample_point, location):
        (x, y) = location
        links = tree.links
        frame = Frame(tree, location=(x, y), name="Trapezoid Selector")

        trap_id = NamedAttribute(tree, location=(x, y - 1), data_type="INT", name="TrapezoidId", hide=True)
        sample_index = SampleIndex(tree, location=(x + 1, y + 1), name="TrapezoidIndex", geometry=sample_point,
                                   value=trap_id.std_out, hide=True, data_type="INT")
        index_switch = IndexSwitch(tree, location=(x + 2, y - 2), hide=True, index=sample_index.std_out,
                                   data_type="GEOMETRY")
        for _ in range(len(trapezoids) - 2):
            index_switch.new_item()
        for slot in range(len(trapezoids)):
            links.new(trapezoids[slot].geometry_out, index_switch.slots[slot + 1])
        frame.add([trap_id, sample_index, index_switch])

        return index_switch, sample_index

    def _substitution_trapezoid(self, tree, g_trapez, f_phi, location):
        (x, y) = location
        links = tree.links

        frame = Frame(tree, location=(x, y), name="Trapezoid Substitution")
        pos = Position(tree, location=(x, y - 5))
        rr_phi = Reroute(tree, location=(x, y - 8), name="phi", ins=f_phi.outputs["phi"], hide=False)
        rr_trap1 = Reroute(tree, location=(x + 0.5, y - 4.5), name="trap", ins=g_trapez, hide=False)
        rr_trap2 = Reroute(tree, location=(x + 4.5, y - 6), name="trap", ins=g_trapez, hide=False)

        sample_vs = []
        for i in range(0, 4):
            sample_vs.append(SampleIndex(tree, location=(x + 1, y - 4 - 0.5 * i),
                                         geometry=rr_trap1.std_out,
                                         data_type="FLOAT_VECTOR", domain="POINT",
                                         name="v" + str(i + 1), index=i,
                                         value=pos.std_out, hide=True))

        # --- Four transformations (PolygonType: 1 = trapezoid sub-piece,
        #                                       2 = parallelogram sub-piece) ---
        transformations = {
            1: ["Transf1", {
                "scaling": ["k", "k", "k"],
                "rotation": ["0", "0", "-2,3,/,pi,*"],
                "translation": "v4,v1,sub",
                "pivot": "v1"
            }, 3],
            2: ["Transf2", {
                "scaling": ["k", "k", "k"],
                "rotation": ["0", "0", "pi"],
                "translation": "v4,v1,sub,v2,v1,sub,1,phi,phi,*,/,scale,add",
                "pivot": "v1"
            }, 4],
            3: ["Transf3", {
                "scaling": ["k", "k", "k"],
                "rotation": ["0", "0", "0"],
                "translation": "v2,v1,sub,1,phi,/,scale",
                "pivot": "v1"
            }, 6],
            4: ["Transf4", {
                "scaling": ["k", "k", "k"],
                "rotation": ["0", "0", "pi,-3,/"],
                "translation": "v1,v2,sub,1,phi,phi,*,/,scale",
                "pivot": "v2"
            }, 5],
        }

        scale_function = make_function(
            tree, name="k-Scale",
            functions={"k": "phi,1,-,phi,/"},
            inputs=["phi"], outputs=["k"],
            scalars=["phi", "k"],
            hide=True, location=(x + 1, y - 2.5))
        links.new(rr_phi.std_out, scale_function.inputs["phi"])

        join_inner = JoinGeometry(tree, location=(x + 18, y - 2))

        vertex_labels = ["v1", "v2", "v3", "v4"]
        # Track the realized output of each transformation for the
        # leftover-quad sampling below.
        realizes = {}
        frame_nodes = []

        for i, [name, function, polygon_type] in transformations.items():
            row_y = y - 2.0 * i

            if i == 4:
                v1_new = make_function(
                    tree, name="v1New",
                    functions={"v1": "v4,v2,v3,sub,add"},
                    inputs=vertex_labels[1:4], outputs=["v1"],
                    vectors=vertex_labels,
                    hide=True, location=(x + 2, row_y))
                for k in range(1, 4):
                    links.new(sample_vs[k].std_out,
                              v1_new.inputs[vertex_labels[k]])

                index = Index(tree, location=(x + 3, row_y),
                              domain="POINT", hide=True)
                index_select = CompareNode(
                    tree, location=(x + 4, row_y), data_type="INT",
                    inputs0=index.std_out, inputs1=0, operation="EQUAL",
                    hide=True, name="=")
                set_pos_i = SetPosition(
                    tree, location=(x + 5, row_y),
                    selection=index_select.std_out, hide=True,
                    position=v1_new.outputs["v1"])
                frame_nodes += [v1_new, index, index_select, set_pos_i]
            transformation = make_function(
                tree, functions=function,
                name=name, location=(x + 5, row_y - 0.5), hide=True,
                inputs=["phi", "k"] + vertex_labels,
                outputs=["rotation", "scaling", "translation", "pivot"],
                scalars=["phi", "k"],
                vectors=vertex_labels + ["rotation", "scaling",
                                         "translation", "pivot"])
            links.new(rr_phi.std_out,
                      transformation.inputs["phi"])
            links.new(scale_function.outputs["k"],
                      transformation.inputs["k"])
            for sample, label in zip(sample_vs, vertex_labels):
                links.new(sample.std_out, transformation.inputs[label])

            geom_to_inst = GeometryToInstance(
                tree, location=(x + 6, row_y))
            scale_elem = ScaleElements(
                tree, location=(x + 7, row_y), domain="FACE",
                scale=transformation.outputs["scaling"],
                center=transformation.outputs["pivot"], hide=True)
            rotate_inst = RotateInstances(
                tree, location=(x + 8, row_y),
                instances=scale_elem.geometry_out,
                rotation=transformation.outputs["rotation"],
                pivot_point=transformation.outputs["pivot"],
                local_space=False, hide=True)
            translate_inst = TranslateInstances(
                tree, location=(x + 9, row_y),
                instances=rotate_inst.geometry_out,
                translation=transformation.outputs["translation"],
                local_space=False, hide=True)
            realize_inst = RealizeInstances(
                tree, location=(x + 10, row_y), hide=True)
            links.new(translate_inst.geometry_out,
                      realize_inst.geometry_in)
            realizes[i] = realize_inst

            store_kind = StoredNamedAttribute(
                tree, location=(x + 11, row_y),
                data_type="INT", domain="FACE",
                name="PolygonType", value=polygon_type, hide=True)
            frame_nodes += [transformation, geom_to_inst, scale_elem, rotate_inst, translate_inst, store_kind]
            if i == 4:
                create_geometry_line(tree,
                                     [set_pos_i, geom_to_inst, scale_elem,
                                      rotate_inst, translate_inst, realize_inst,
                                      store_kind, join_inner], ins=rr_trap2.geometry_out)
            else:
                create_geometry_line(tree,
                                     [geom_to_inst, scale_elem, rotate_inst,
                                      translate_inst, realize_inst, store_kind,
                                      join_inner], ins=rr_trap2.geometry_out)

        # create left-over faces
        y -= 2
        rr_sub1 = Reroute(tree, location=(x + 12, y + 1), hide=True, name="sub1", ins=realizes[1].geometry_out)
        rr_sub3 = Reroute(tree, location=(x + 12, y + 1.5), hide=True, name="sub3", ins=realizes[3].geometry_out)
        rr_sub4 = Reroute(tree, location=(x + 12, y + 2), hide=True, name="sub4", ins=realizes[4].geometry_out)

        position = Position(tree, location=(x + 13, y + 2), hide=True)

        sample_indices_below = [
            SampleIndex(tree, location=(x + 14, y + 1), domain="POINT", geometry=rr_sub1.geometry_out,
                        value=position.std_out, index=2, hide=True),
            SampleIndex(tree, location=(x + 14, y + 1.5), domain="POINT", geometry=rr_sub4.geometry_out,
                        value=position.std_out, index=1, hide=True),
            SampleIndex(tree, location=(x + 14, y + 2), domain="POINT", geometry=rr_sub4.geometry_out,
                        value=position.std_out, index=0, hide=True),
            SampleIndex(tree, location=(x + 14, y + 2.5), domain="POINT", geometry=rr_sub1.geometry_out,
                        value=position.std_out, index=3, hide=True),
        ]

        sample_indices_above = [
            SampleIndex(tree, location=(x + 14, y + 3), domain="POINT", geometry=rr_sub3.geometry_out,
                        value=position.std_out, index=3, hide=True),
            SampleIndex(tree, location=(x + 14, y + 3.5), domain="POINT", geometry=rr_sub3.geometry_out,
                        value=position.std_out, index=2, hide=True),
            sample_vs[2],
            SampleIndex(tree, location=(x + 14, y + 4), domain="POINT", geometry=rr_sub4.geometry_out,
                        value=position.std_out, index=3, hide=True),
        ]

        quad_below = Quadrilateral(tree, location=(x + 15, y + 1), mode="POINTS",
                                   name="QuadBelow",
                                   hide=True)
        below_fill = FillCurve(tree, location=(x + 16, y + 1), hide=True)
        below_attr = StoredNamedAttribute(tree, location=(x + 17, y + 1), data_type="INT", domain="FACE",
                                          name="PolygonType", value=1, hide=True)
        quad_above = Quadrilateral(tree, location=(x + 15, y + 2), mode="POINTS", hide=True)
        above_fill = FillCurve(tree, location=(x + 16, y + 2), hide=True)
        above_attr = StoredNamedAttribute(tree, location=(x + 17, y + 2), data_type="INT", domain="FACE",
                                          name="PolygonType", value=2, hide=True)

        for i in range(4):
            links.new(sample_indices_below[i].std_out, quad_below.inputs["Point " + str(i + 1)])
            links.new(sample_indices_above[i].std_out, quad_above.inputs["Point " + str(i + 1)])

        create_geometry_line(tree, [quad_below, below_fill, below_attr, join_inner])
        create_geometry_line(tree, [quad_above, above_fill, above_attr, join_inner])
        frame.add(sample_vs + [pos, rr_phi, rr_trap1, rr_trap2, scale_function, join_inner] + list(realizes.values()) +
                  [rr_sub1, rr_sub4, rr_sub3] +
                  frame_nodes + [position, quad_above, quad_below, below_fill, above_fill, below_attr, above_attr] +
                  sample_indices_above + sample_indices_below)
        return join_inner, *(realizes.values())

    def _back_translation(self, tree, points, u, v, location=(0, 0)):
        (x, y) = location
        links = tree.links
        frame = Frame(tree, location=(x, y), name="BackTranslation")

        pos = Position(tree, location=(x, y), name="Position", hide=True)

        back_translation_prep = make_function(tree, name="BackTranslationPreparation",
                                              functions={"u*v": "u,cconj,v,cmul",
                                                         "uv*": "u,v,cconj,cmul",
                                                         "pu*": "p,u,cconj,cmul",
                                                         "pv*": "p,v,cconj,cmul",
                                                         },
                                              inputs=["u", "v", "p"], outputs=["u*v", "uv*", "pu*", "pv*"],
                                              vectors=["u*v", "uv*", "u", "v", "pu*", "pv*", "p"], hide=True,
                                              location=(x + 1, y + 0.5)
                                              )
        links.new(u.outputs["u"], back_translation_prep.inputs["u"])
        links.new(v.std_out, back_translation_prep.inputs["v"])
        links.new(pos.std_out, back_translation_prep.inputs["p"])

        l_value = InputValue(tree, value=1, name="l", location=(x + 1, y - 0.5), hide=False)
        back_translation = make_function(tree, name="BackTranslation", functions={
            "pos": "pos,1,l,-,scale,u,pv*_y,uv*_y,/,frac,l,*,scale,add,v,pu*_y,u*v_y,/,frac,l,*,scale,add"
        },
                                         inputs=["u*v", "uv*", "pu*", "pv*", "l", "u", "v", "pos"],
                                         outputs=["pos"],
                                         vectors=["u*v", "uv*", "pos", "pu*", "pv*", "u", "v"], scalars=["l"],
                                         location=(x + 2, y), hide=True)
        links.new(pos.std_out, back_translation.inputs["pos"])
        links.new(u.outputs["u"], back_translation.inputs["u"])
        links.new(v.std_out, back_translation.inputs["v"])
        links.new(back_translation_prep.outputs["u*v"], back_translation.inputs["u*v"])
        links.new(back_translation_prep.outputs["uv*"], back_translation.inputs["uv*"])
        links.new(back_translation_prep.outputs["pu*"], back_translation.inputs["pu*"])
        links.new(back_translation_prep.outputs["pv*"], back_translation.inputs["pv*"])
        links.new(l_value.std_out, back_translation.inputs["l"])

        set_pos = SetPosition(tree, location=(x + 3, y), hide=True, position=back_translation.outputs["pos"])
        create_geometry_line(tree, [points, set_pos])

        frame.add([pos, back_translation_prep, back_translation, set_pos, l_value])

        return set_pos

    def _generate_grid(self, tree, grid_size=10, location=(0, 0), **kwargs):
        (x, y) = location
        links = tree.links

        frame = Frame(tree, location=(x, y), name="GridGeneration")

        grid_size_node = InputInteger(tree, location=(x + 1, y + 2), integer=grid_size, name='GridSize')
        plus_one = MathNode(
            tree, location=(x + 2, y + 1.5), operation='ADD',
            inputs0=grid_size_node.std_out, inputs1=1,
            name='+1', hide=True
        )

        # --- Square grid -> triangular lattice ---
        grid = Grid(
            tree, location=(x + 3, y + 2),
            size_x=grid_size_node.std_out,
            size_y=grid_size_node.std_out,
            vertices_x=plus_one.std_out,
            vertices_y=plus_one.std_out,
            name='Grid',
            hide=True
        )

        pos = Position(tree, location=(x, y + 1.5))
        shift_x = InputValue(tree, location=(x, y), value=0.1, name="ShiftX", hide=True)
        shift_y = InputValue(tree, location=(x, y), value=0.01, name="ShiftX", hide=True)
        combine = CombineXYZ(tree, location=(x + 1, y - 1), x=shift_x.std_out, y=shift_y.std_out, name="GridShift",
                             hide=True)

        # Shear: x' = x + y/2, y' = y*sqrt(3)/2
        grid_function = make_function(
            tree,
            functions={
                'out': [
                    'pos_x,pos_y,2.0,/,+',
                    f'pos_y,{r3},*,2.0,/',
                    '0',
                ],
            },
            inputs=['pos'],
            outputs=['out'],
            vectors=['pos', 'out'],
            name='TriShear',
            location=(x + 1, y + 1.5),
            hide=True,
        )
        links.new(pos.std_out, grid_function.inputs['pos'])

        frame_driven_displacement_function = get_from_kwargs(kwargs, "frame_driven_displacement_function", "")
        time = t_value = driver = None
        if frame_driven_displacement_function != "":
            t0 = get_from_kwargs(kwargs, "begin_time", 0)
            duration = get_from_kwargs(kwargs, "transition_time", DEFAULT_ANIMATION_TIME)
            [mini, maxi] = get_from_kwargs(kwargs, "domain", [0, 1])
            # prepare t value
            time = SceneTime(tree, location=(x + 1, y + 3), std_out="Seconds", hide=True, name="Time")

            t_value = make_function(tree, custom_ops=custom_ops, name="tValue", functions={
                "t": f"time,{t0},{duration},+,>,{maxi},time,{t0},<,{mini},{mini},{maxi},{mini},-,{duration},/,time,{t0},-,*,+,iff,iff"
            }, inputs=["time"], outputs=["t"], scalars=["time", "t"],
                                    location=(x + 2, y + 3), hide=True)
            links.new(time.std_out, t_value.inputs["time"])

            driver = make_function(tree, name="FrameDrivenDisplacement",
                                   aux_functions={
                                       "driver": frame_driven_displacement_function
                                   },
                                   functions={
                                       "pos": "driver,shift,add"
                                   },
                                   inputs=["t", "shift"], outputs=["pos"],
                                   scalars=["t"], vectors=["pos", "shift", "driver"], hide=True,
                                   location=[x + 3, y + 3])
            links.new(t_value.outputs["t"], driver.inputs["t"])
            links.new(combine.std_out, driver.inputs["shift"])

            offset = driver.outputs["pos"]
        else:
            offset = combine.std_out

        index = Index(tree, location=(x + 2, y - 3), hide=True)
        grid_filter = InputInteger(tree, location=(x + 2, y - 3.5), hide=True, integer=0, name="GridFilter")
        length = VectorMath(tree, operation="LENGTH", location=(x + 1, y - 3), hide=True, inputs0=pos.std_out)
        compare = CompareNode(tree, data_type="INT", operation="LESS_EQUAL",
                              inputs0=index.std_out, inputs1=grid_filter.std_out, location=(x + 3, y - 3.5),
                              hide=True)
        sort_elements = SortElements(tree, location=(x + 3, y), hide=True, domain="POINT", sort_weight=length.std_out)
        sep_geo = SeparateGeometry(tree, domain="POINT", location=(x + 3, y - 0.5), hide=True,
                                   selection=compare.std_out)

        set_pos = SetPosition(
            tree, location=(x + 4, y + 2),
            position=grid_function.outputs["out"],
            name='Set Position',
            offset=offset,
            hide=True
        )

        mesh2points = MeshToPoints(tree, location=(x + 5, y + 2), hide=True)

        pos2 = Position(tree, location=(x + 6, y + 0.5), hide=True)
        old_pos = StoredNamedAttribute(tree, location=(x + 6, y), name="OldPosition", data_type="FLOAT_VECTOR",
                                       domain="POINT",
                                       hide=True, value=pos2.std_out)

        create_geometry_line(tree, [grid, sort_elements, sep_geo, set_pos, mesh2points, old_pos])

        driven_nodes = [time, t_value, driver] if frame_driven_displacement_function != "" else []
        frame.add(driven_nodes + [index, grid_filter, length, compare, sort_elements, sep_geo] +
                  [shift_x, shift_y, combine, grid_size_node, plus_one, grid, pos, pos2, grid_function, set_pos,
                   mesh2points,
                   old_pos])
        return old_pos, set_pos

    def _generate_hats(self, tree, location, **kwargs):
        links = tree.links
        (x, y) = location
        frame = Frame(tree, location=(x, y), name="Hats")

        direct_vertices = _hat_vertices_3d(rotation=0, ref=False)
        n = 13  # number of vertices of the hat tile
        scale = InputValue(tree, value=1, location=(x, y - 5), hide=False)

        circle_d = MeshCircle(tree, vertices=13, fill_type='NGON', location=(x + 1, y))
        index_d = Index(tree, location=(x, y - 1))
        vsw_d = IndexSwitch(tree, data_type='VECTOR', location=(x + 1, y - 1), hide=True)
        links.new(index_d.std_out, vsw_d.index)
        for _ in range(n - 2):
            vsw_d.new_item()
        for i, v in enumerate(direct_vertices):
            vsw_d.slots[i + 1].default_value = list(v)
        set_pos_d = SetPosition(tree, location=(x + 2, y),
                                geometry=circle_d.geometry_out,
                                position=vsw_d.std_out)

        # --- Reflected prototile ---
        reflect_verts = _hat_vertices_3d(rotation=0, ref=True)
        circle_r = MeshCircle(tree, vertices=n, fill_type='NGON', location=(x + 1, y - 3))
        index_r = Index(tree, location=(x, y - 4))
        vsw_r = IndexSwitch(tree, data_type='VECTOR', location=(x + 1, y - 4), hide=True)
        links.new(index_r.std_out, vsw_r.index)
        for _ in range(n - 2):
            vsw_r.new_item()
        for i, v in enumerate(reflect_verts):
            vsw_r.slots[i + 1].default_value = list(v)
        set_pos_r = SetPosition(tree, location=(x + 2, y - 3),
                                geometry=circle_r.geometry_out,
                                position=vsw_r.std_out)

        # --- 12 variants: direct k=5..0, then reflected k=5..0 ---
        # Each slot in the final IndexSwitch receives the G2I output in that order.
        g2i_outputs = []

        rr_unreflected = Reroute(tree, location=(x + 3, y - 2), name="unreflected", ins=set_pos_d.geometry_out)
        rr_reflected = Reroute(tree, location=(x + 3, y - 6), name="reflected", ins=set_pos_r.geometry_out)

        frame_nodes = []
        self.hat_materials = []
        for i, k in enumerate(range(0, 6)):
            angle = k * pi / 3.0
            xd = TransformGeometry(tree, location=(x + 4, y - i), hide=True,
                                   rotation=[0, 0, angle], scale=scale.std_out)
            g2i_d = GeometryToInstance(tree, location=(x + 5, y - i))
            material = get_texture(material="hat0" + str(i), **kwargs)
            self.materials.append(material)
            self.hat_materials.append(material)
            color = SetMaterial(tree, location=(x + 6, y - i), material=material, hide=True)

            create_geometry_line(tree, [xd, g2i_d, color], ins=rr_unreflected.geometry_out)
            g2i_outputs.append(color)
            frame_nodes += [xd, g2i_d, color]

        rot_map = {0: 2, 1: 3, 2: 4, 3: 5, 4: 0, 5: 1}
        for i in range(6):
            angle = rot_map[i] * pi / 3.0
            xr = TransformGeometry(tree, location=(x + 4, y - i - 6.5), hide=True,
                                   rotation=[0, 0, angle], scale=scale.std_out)
            g2i_r = GeometryToInstance(tree, location=(x + 5, y - i - 6.5))
            material = get_texture(material="hat1" + str(i), **kwargs)
            self.hat_materials.append(material)
            self.materials.append(material)
            color = SetMaterial(tree, location=(x + 6, y - i - 6.5), material=material, hide=True)
            create_geometry_line(tree, [xr, g2i_r, color], ins=rr_reflected.geometry_out)
            g2i_outputs.append(color)
            frame_nodes += [xr, g2i_r, color]

        # --- IndexSwitch GEOMETRY: 12 slots ---
        isw = IndexSwitch(tree, data_type='GEOMETRY', location=(x + 7, y))
        for _ in range(10):
            isw.new_item()
        for i, g2i in enumerate(g2i_outputs):
            links.new(g2i.geometry_out, isw.slots[i + 1])

        frame.add(
            [scale, circle_d, index_d, vsw_d, set_pos_d, circle_r, index_r, vsw_r, set_pos_r, rr_unreflected,
             rr_reflected,
             isw] + frame_nodes)
        return isw

    def _sample_points_in_cover(self, tree, cover, points, location):
        (x, y) = location
        links = tree.links
        frame = Frame(tree, location=(x, y), name="SamplePointsInCover")

        # Convert point cloud to mesh vertices
        p2v = PointsToVertices(tree, location=(x, y))
        store_trapezoid = StoredNamedAttribute(tree, location=(x + 1, y), name="TrapezoidId", data_type="INT",
                                               domain="POINT", value=-1, hide=True)
        links.new(points.geometry_out, p2v.geometry_in)

        # Position field feeds the ForEachZone per-element position socket
        pos_field = Position(tree, location=(x + 2, y + 1))

        # ForEachZone iterating over the sample point mesh
        for_each_zone = ForEachZone(
            tree, location=(x + 3, y + 2),
            domain="POINT", node_width=6,
            geometry=p2v.geometry_out,
            name="ForEachSamplePoints")
        for_each_zone.add_socket(socket_type="VECTOR", name="Position",
                                 value=pos_field.std_out, for_input=True)

        # Inside the zone: nearest face of g_full at the current element position
        sample_nearest = SampleNearest(tree, location=(x + 4, y + 1), domain='FACE',
                                       geometry=cover.geometry_out,
                                       sample_position=for_each_zone.outputs["Position"])

        # Index == face index field; Compare selects the one matching face from g_full
        index_node = Index(tree, location=(x + 4, y + 0))
        compare = CompareNode(tree, location=(x + 5, y + 1), data_type="INT", operation="EQUAL",
                              inputs0=index_node.std_out, inputs1=sample_nearest.std_out)
        sep_geom = SeparateGeometry(tree, location=(x + 6, y + 1), domain="FACE",
                                    selection=compare.std_out)
        links.new(cover.geometry_out, sep_geom.geometry_in)

        # InsidePolygon: is the sample point truly inside the selected face?
        inside_test = InsidePolygon(tree, location=(x + 7, y + 2),
                                    target_geometry=sep_geom.geometry_out,
                                    source_position=for_each_zone.outputs["Position"])

        store_trapezoid2 = StoredNamedAttribute(tree, location=(x + 8, y + 1), name="TrapezoidId", data_type="INT",
                                                domain="POINT", value=sample_nearest.std_out,
                                                selection=inside_test.std_out)

        for_each_zone.create_geometry_line([store_trapezoid2])

        trap_index = NamedAttribute(tree, location=(x + 10, y + 1), name="TrapezoidId", hide=True)
        compare = CompareNode(tree, location=(x + 10, y), data_type="INT", operation="GREATER_EQUAL",
                              inputs0=trap_index.std_out,
                              inputs1=0, hide=True)
        sep_geo = SeparateGeometry(tree, location=(x + 11, y + 2), domain="POINT", selection=compare.std_out)
        create_geometry_line(tree, [p2v, store_trapezoid, for_each_zone, sep_geo])
        frame.add(
            [p2v, store_trapezoid, pos_field, for_each_zone, sample_nearest, index_node, compare, sep_geom, inside_test,
             store_trapezoid2, trap_index, compare, sep_geo])
        return sep_geo

    def _make_fundamental_domain(self, tree, xi, phi, u_function, v, location):
        (x, y) = location

        frame = Frame(tree, location=(x, y), name="Fundamental Domain")
        point = Points(tree, location=(x, y), position=Vector([0, 0, 0.1]), count=1, hide=True)
        p2mesh = PointsToVertices(tree, location=(x + 1, y), hide=True)
        extrude = ExtrudeMesh(tree, location=(x + 2, y), mesh=point.geometry_out, mode="VERTICES",
                              offset=u_function.outputs["u"], hide=True)
        extrude2 = ExtrudeMesh(tree, location=(x + 3, y), mesh=point.geometry_out, mode="EDGES", offset=v.std_out,
                               hide=True)
        wireframe = WireFrame(tree, location=(x + 4, y), hide=True)

        join_inner = JoinGeometry(tree, location=(x + 5, y), hide=True)
        create_geometry_line(tree, [point, p2mesh, extrude, extrude2, wireframe, join_inner])
        frame.add([point, p2mesh, extrude, extrude2, wireframe, join_inner])
        return join_inner

    def _create_constants(self, tree, location):
        (x, y) = location
        links = tree.links

        frame = Frame(tree, location=(x, y), name="Constants")
        one = InputVector(tree, value=[1, 0, 0], name="one",
                          location=(x, y), hide=True)
        zero = InputVector(tree, value=[0, 0, 0], name="zero",
                           location=(x, y - 0.5), hide=True)
        xi_function = make_function(tree, name="xi", functions={
            "xi": ["1,2,/", "3,sqrt,2,/", "0"]
        }, outputs=["xi"], vectors=["xi"],
                                    location=(x, y - 1), hide=True)
        phi_function = make_function(tree, name="phi", functions={
            "phi": "1,5,sqrt,+,2,/"
        }, outputs=["phi"], scalars=["phi"],
                                     location=(x, y - 1.5), hide=True)

        u_function = make_function(tree, name="u =phi + 1 + xi", functions={
            "u": ["phi,1,+,xi_x,+", "xi_y", "0"]
        }, inputs=["xi", "phi"], outputs=["u"], vectors=["xi", "u"], scalars=["phi"],
                                   location=(x, y - 2), hide=True)
        links.new(xi_function.outputs["xi"], u_function.inputs["xi"])
        links.new(phi_function.outputs["phi"], u_function.inputs["phi"])

        v = ComplexMathNode(tree, name="v=xi u", z=u_function.outputs["u"],
                            w=xi_function.outputs["xi"], operation="MUL",
                            location=(x, y - 2.5),
                            hide=True)
        frame.add([one, zero, xi_function, phi_function, u_function, v])
        return one, zero, xi_function, phi_function, u_function, v

    def _create_constants2(self, tree, location):
        (x, y) = location

        frame = Frame(tree, location=(x, y), name="Constants xi and phi")

        xi_function = make_function(tree, name="xi", functions={
            "xi": ["1,2,/", "3,sqrt,2,/", "0"]
        }, outputs=["xi"], vectors=["xi"],
                                    location=(x, y - 1), hide=True)
        phi_function = make_function(tree, name="phi", functions={
            "phi": "1,5,sqrt,+,2,/"
        }, outputs=["phi"], scalars=["phi"],
                                     location=(x, y - 1.5), hide=True)

        frame.add([xi_function, phi_function])
        return xi_function, phi_function

    def _create_rotations(self, tree, location):
        (x, y) = location

        rots = []

        for i in range(0, 3):
            r = (i + 1) * 60
            input_vector = InputVector(tree, value=[0, (i + 1) * pi / 3, 0], location=(x, y - 0.5 * i),
                                       name="rot" + str(r), hide=True)
            rots.append(
                make_function(tree, name="rot" + str(r), functions={
                    "rot" + str(r): "v,cexp"
                }, outputs=["rot" + str(r)], vectors=["v", "rot" + str(r)],
                              inputs=["v"],
                              location=(x + 1, y - 0.5 * i), hide=True)
            )
            tree.links.new(input_vector.std_out, rots[-1].inputs["v"])
        return rots

    def _create_trapezoid(self, tree, zero, one, f_xi, f_phi, location):
        (x, y) = location
        frame = Frame(tree, location=(x, y), name="Trapezoid")

        params_in = ["zero", "one", "xi", "phi"]
        params_out = ["a", "b", "c", "d"]
        links = tree.links

        vertices = make_function(tree, name="VerticesTrapez", functions={
            "a": "zero",
            "b": "one,phi,1,+,scale",
            "c": "one,phi,scale,xi,cadd",
            "d": "xi",
        }, inputs=params_in, outputs=params_out,
                                 vectors=params_out + params_in,
                                 location=(x, y), hide=True)
        links.new(zero.std_out, vertices.inputs["zero"])
        links.new(one.std_out, vertices.inputs["one"])
        links.new(f_xi.outputs["xi"], vertices.inputs["xi"])
        links.new(f_phi.outputs["phi"], vertices.inputs["phi"])

        trapez = Quadrilateral(tree, location=(x + 1, y), mode="POINTS",
                               name="Trapez", hide=True)
        links.new(vertices.outputs["a"], trapez.inputs["Point 1"])
        links.new(vertices.outputs["b"], trapez.inputs["Point 2"])
        links.new(vertices.outputs["c"], trapez.inputs["Point 3"])
        links.new(vertices.outputs["d"], trapez.inputs["Point 4"])

        fill = FillCurve(tree, location=(x + 2, y), hide=True)

        # Mark the trapezoid face with PolygonType=0.
        store_trapez = StoredNamedAttribute(
            tree, location=(x + 3, y),
            data_type="INT", domain="FACE",
            name="PolygonType", label="MarkTrapez",
            value=0, hide=True)
        create_geometry_line(tree, [trapez, fill, store_trapez])

        frame.add([vertices, trapez, fill, store_trapez])
        return store_trapez

    def _make_cover(self, tree, g_trapezoid, f_u, v, location):
        (x, y) = location
        links = tree.links

        frame = Frame(tree, location=(x, y), name="Fundamental Domain Cover")
        join = JoinGeometry(tree, location=(x + 3, y))

        rotations = [Vector([0, 0, pi / 3 * i]) for i in [0, 1, 2, 3, 4, 5, 0, 1, 3, 4]]
        uv = make_function(tree, name="u+v", functions={
            "uv": "u,v,cadd"
        }, inputs=["u", "v"], outputs=["uv"], vectors=["uv", "u", "v"], location=(x, y - 5), hide=True)
        links.new(f_u.outputs["u"], uv.inputs["u"])
        links.new(v.std_out, uv.inputs["v"])

        translations = [
            [0, 0, 0],
            [0, 0, 0],
            f_u.outputs["u"],
            f_u.outputs["u"],
            v.std_out,
            v.std_out,
            v.std_out,
            f_u.outputs["u"],
            uv.outputs["uv"],
            uv.outputs["uv"]

        ]
        frame_nodes = []
        trapezoids = []
        for i, (rotation, translation) in enumerate(zip(rotations, translations)):
            trafo = TransformGeometry(tree, location=(x + 2, y - 0.5 * i), geometry=g_trapezoid.geometry_out,
                                      rotation=rotation, translation=translation, hide=True)
            trapezoids.append(trafo)
            frame_nodes.append(trafo)

        # important that the joining happens in reverse order, since it determines the value of the variable TrapezoidId
        for trafo in reversed(trapezoids):
            create_geometry_line(tree, [g_trapezoid, trafo, join])

        frame.add([uv, join] + frame_nodes)
        return join, trapezoids

# ---------------------------------------------------------------------------
# Cheap modifier: arithmetic inRegion algorithm (no SampleNearest ray casts)
# ---------------------------------------------------------------------------

class LabbeSelingerOptimizedModifier(GeometryNodesModifier):
    """
    Standalone hat-tiling modifier (extends GeometryNodesModifier directly).

    Keeps only the GridGeneration, Hats, Fractal Index, GridpointVisualization
    and Finalize frames.  Each grid point's hat orientation index (0-11) is
    computed directly by porting the ``hat_tile_fractal`` shader into geometry
    nodes (see ``_fractal_index`` and the ``InRegion`` group in
    geometry_nodes/nodes.py) rather than via the trapezoid / fundamental-domain
    cover used by ``LabbeSelingerModifier``.

    The frame builders ``_grid_point_visualization``, ``_finalize``,
    ``_generate_grid`` and ``_generate_hats`` are copied from
    ``LabbeSelingerModifier`` so this class no longer depends on it.
    """

    def __init__(self, **kwargs):
        GeometryNodesModifier.__init__(self, 'LabbeSelingerModifierCheap',
                                       automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    # Frame builders copied from LabbeSelingerModifier so this class is
    # self-contained (extends GeometryNodesModifier directly).
    # ------------------------------------------------------------------

    def _grid_point_visualization(self, tree, ins, location, **kwargs):
        (x, y) = location

        dot_color = get_from_kwargs(kwargs, "dot_color", "red")
        dot_radius = get_from_kwargs(kwargs, "dot_radius", 0.1)
        # optional rectangular crop of the visualized grid points, given as
        # (center_x, center_y, half_width, half_height) in object coordinates;
        # the full (unfiltered) grid is shown when it is omitted
        dot_window = get_from_kwargs(kwargs, "dot_window", None)
        grid_radius = InputValue(tree, location=(x, y - 2), hide=False, name="GridRadius", value=dot_radius)
        compare = CompareNode(tree, location=(x + 0.5, y - 3), hide=True, data_type="FLOAT",operation="LESS_THAN",
                              inputs0=grid_radius.std_out, inputs1=0.00001)
        del_geo = DeleteGeometry(tree, location=(x + 0.5, y - 2), hide=True, selection=compare.std_out)

        sphere = IcoSphere(tree, location=(x, y - 1), name="GridSphere", radius=grid_radius.std_out, hide=True)
        iop = InstanceOnPoints(tree, location=(x + 1, y), hide=True, instance=sphere.geometry_out)

        mat = get_texture(material=dot_color, roughness=0.1, emission=1)
        self.materials.append(mat)
        set_material = SetMaterial(tree, location=(x + 2, y), hide=True, material=mat)

        window_nodes = []
        if dot_window is not None:
            cx, cy, hw, hh = dot_window
            pos = Position(tree, location=(x + 0.2, y - 4), hide=True)
            outside = make_function(tree, name="DotWindowFilter", functions={
                "out": f"pos_x,{cx},-,abs,{hw},>,pos_y,{cy},-,abs,{hh},>,or"
            }, inputs=["pos"], outputs=["out"], scalars=["out"], vectors=["pos"],
                                    location=(x + 0.5, y - 4), hide=True)
            tree.links.new(pos.std_out, outside.inputs["pos"])
            del_outside = DeleteGeometry(tree, location=(x + 0.8, y - 2), hide=True,
                                         selection=outside.outputs["out"])
            window_nodes = [pos, outside, del_outside]
            create_geometry_line(tree, [ins, del_geo, del_outside, iop, set_material])
        else:
            create_geometry_line(tree, [ins, del_geo, iop, set_material])

        frame = Frame(tree, location=location, name="Grid Point Visualization")
        frame.add([grid_radius, compare, del_geo, sphere, iop, set_material] + window_nodes)
        return set_material

    def _finalize(self, tree, ins, location):
        (x, y) = location
        links = tree.links
        center = NamedAttribute(tree, location=(x, y - 2), hide=True, name="OldPosition", data_type="FLOAT_VECTOR")
        hat_scale = InputValue(tree, location=(x, y - 3), hide=True, value=0.95, name="HatScale")
        scale = ScaleElements(tree, location=(x, y), hide=True, scale=hat_scale.std_out, center=center.std_out)
        extrude = ExtrudeMesh(tree, location=(x + 1, y), mode="FACES", offset_scale=0.15, hide=True)

        join = JoinGeometry(tree, location=(x + 2, y), hide=True)

        # --- IndexSwitch Material: 12 slots ---
        orientation = NamedAttribute(tree, location=(x + 2, y - 1), name="Orientation",
                                     data_type="INT", hide=True)
        sample_index = SampleIndex(tree, location=(x + 2, y), name="SampleOrientation",
                                   hide=True, value=orientation.std_out, data_type="INT",
                                   geometry=ins.geometry_out, domain="INSTANCE")
        isw = IndexSwitch(tree, data_type='MATERIAL',
                          location=(x + 3, y), index=sample_index.std_out)
        for _ in range(10):
            isw.new_item()
        count = 0
        colors = []
        self.hat_frame_materials = []
        for i in range(0, 2):
            for k in range(0, 6):
                material = get_texture(material="hat" + str(i) + str(k), emission=0.01)
                self.materials.append(material)
                self.hat_frame_materials.append(material)
                color = InputMaterial(tree, location=(x + 2, y - 2 - i * 3 - k * 0.5),
                                      material=material, hide=True)

                colors.append(color)
                links.new(color.std_out, isw.slots[count + 1])
                count += 1

        toggle_wireframe = InputBoolean(tree, location=(x, y - 4), value=True, name="Wireframe")
        not_wire = BooleanMath(tree, location=(x + 1, y - 4), operation="NOT",
                               inputs0=toggle_wireframe.std_out, hide=True)
        wireframe = WireFrame(tree, location=(x + 4, y), radius=0.01, hide=True)
        del_wire = DeleteGeometry(tree, location=(x + 4.5, y), domain="POINT", mode="ALL",
                                  geometry=wireframe.geometry_out, selection=not_wire.std_out, hide=True)
        mat = SetMaterial(tree, material=isw.std_out,
                          hide=True, location=(x + 5, y))
        join2 = JoinGeometry(tree, location=(x + 6, y), hide=True)
        create_geometry_line(tree, [ins, scale, join])
        create_geometry_line(tree, [scale, extrude, join])
        create_geometry_line(tree, [join, wireframe, del_wire, mat, join2])
        create_geometry_line(tree, [join, join2])
        frame = Frame(tree, location=(x, y), name="Finalize")
        frame.add(colors + [sample_index, isw, orientation, scale, extrude, join, wireframe, del_wire,
                            mat, hat_scale, center, toggle_wireframe, not_wire])
        return join2

    def _generate_grid(self, tree, grid_size=10, location=(0, 0), **kwargs):
        (x, y) = location
        links = tree.links

        shift = get_from_kwargs(kwargs,"shift",Vector([0.01,0.01,0]))
        grid_filter = get_from_kwargs(kwargs,"grid_filter",10)
        frame = Frame(tree, location=(x, y), name="GridGeneration")

        grid_size_node = InputInteger(tree, location=(x + 1, y + 2), integer=grid_size, name='GridSize')
        plus_one = MathNode(
            tree, location=(x + 2, y + 1.5), operation='ADD',
            inputs0=grid_size_node.std_out, inputs1=1,
            name='+1', hide=True
        )

        # --- Square grid -> triangular lattice ---
        grid = Grid(
            tree, location=(x + 3, y + 2),
            size_x=grid_size_node.std_out,
            size_y=grid_size_node.std_out,
            vertices_x=plus_one.std_out,
            vertices_y=plus_one.std_out,
            name='Grid',
            hide=True
        )

        pos = Position(tree, location=(x, y + 1.5))
        shift_x = InputValue(tree, location=(x, y), value=shift[0], name="ShiftX", hide=True)
        shift_y = InputValue(tree, location=(x, y-0.5), value=shift[1], name="ShiftY", hide=True)
        combine = CombineXYZ(tree, location=(x + 1, y - 1), x=shift_x.std_out, y=shift_y.std_out, name="GridShift",
                             hide=True)

        # Shear: x' = x + y/2, y' = y*sqrt(3)/2
        grid_function = make_function(
            tree,
            functions={
                'out': [
                    'pos_x,pos_y,2.0,/,+',
                    f'pos_y,{r3},*,2.0,/',
                    '0',
                ],
            },
            inputs=['pos'],
            outputs=['out'],
            vectors=['pos', 'out'],
            name='TriShear',
            location=(x + 1, y + 1.5),
            hide=True,
        )
        links.new(pos.std_out, grid_function.inputs['pos'])

        frame_driven_displacement_function = get_from_kwargs(kwargs, "frame_driven_displacement_function", "")
        time = t_value = driver = None
        if frame_driven_displacement_function != "":
            t0 = get_from_kwargs(kwargs, "begin_time", 0)
            duration = get_from_kwargs(kwargs, "transition_time", DEFAULT_ANIMATION_TIME)
            [mini, maxi] = get_from_kwargs(kwargs, "domain", [0, 1])
            # prepare t value
            time = SceneTime(tree, location=(x + 1, y + 3), std_out="Seconds", hide=True, name="Time")

            t_value = make_function(tree, custom_ops=custom_ops, name="tValue", functions={
                "t": f"time,{t0},{duration},+,>,{maxi},time,{t0},<,{mini},{mini},{maxi},{mini},-,{duration},/,time,{t0},-,*,+,iff,iff"
            }, inputs=["time"], outputs=["t"], scalars=["time", "t"],
                                    location=(x + 2, y + 3), hide=True)
            links.new(time.std_out, t_value.inputs["time"])

            driver = make_function(tree, name="FrameDrivenDisplacement",
                                   aux_functions={
                                       "driver": frame_driven_displacement_function
                                   },
                                   functions={
                                       "pos": "driver,shift,add"
                                   },
                                   inputs=["t", "shift"], outputs=["pos"],
                                   scalars=["t"], vectors=["pos", "shift", "driver"], hide=True,
                                   location=[x + 3, y + 3])
            links.new(t_value.outputs["t"], driver.inputs["t"])
            links.new(combine.std_out, driver.inputs["shift"])

            offset = driver.outputs["pos"]
        else:
            offset = combine.std_out

        index = Index(tree, location=(x + 2, y - 3), hide=True)

        grid_filter = InputInteger(tree, location=(x + 2, y - 3.5), hide=True, integer=grid_filter, name="GridFilter")
        grid_center = InputVector(tree, location=(x, y - 2.5), hide=True,
                                  value=get_from_kwargs(kwargs, "grid_center", Vector([0, 0, 0])),
                                  name="GridCenter")
        offset_from_center = VectorMath(tree, operation="SUBTRACT", location=(x + 0.5, y - 3), hide=True,
                                        inputs0=pos.std_out, inputs1=grid_center.std_out)
        length = VectorMath(tree, operation="LENGTH", location=(x + 1, y - 3), hide=True,
                            inputs0=offset_from_center.std_out)  # distance from GridCenter
        compare = CompareNode(tree, data_type="INT", operation="LESS_EQUAL",
                              inputs0=index.std_out, inputs1=grid_filter.std_out, location=(x + 3, y - 3.5),
                              hide=True)
        sort_elements = SortElements(tree, location=(x + 3, y), hide=True, domain="POINT", sort_weight=length.std_out)
        sep_geo = SeparateGeometry(tree, domain="POINT", location=(x + 3, y - 0.5), hide=True,
                                   selection=compare.std_out)

        set_pos = SetPosition(
            tree, location=(x + 4, y + 2),
            position=grid_function.outputs["out"],
            name='Set Position',
            offset=offset,
            hide=True
        )

        mesh2points = MeshToPoints(tree, location=(x + 5, y + 2), hide=True)

        pos2 = Position(tree, location=(x + 6, y + 0.5), hide=True)
        old_pos = StoredNamedAttribute(tree, location=(x + 6, y), name="OldPosition", data_type="FLOAT_VECTOR",
                                       domain="POINT",
                                       hide=True, value=pos2.std_out)

        create_geometry_line(tree, [grid,set_pos,  sort_elements, sep_geo, mesh2points, old_pos])

        driven_nodes = [time, t_value, driver] if frame_driven_displacement_function != "" else []
        frame.add(driven_nodes + [index, grid_filter, grid_center, offset_from_center, length, compare,
                                  sort_elements, sep_geo] +
                  [shift_x, shift_y, combine, grid_size_node, plus_one, grid, pos, pos2, grid_function, set_pos,
                   mesh2points,
                   old_pos])
        return old_pos, set_pos

    def _generate_hats(self, tree, location, **kwargs):
        links = tree.links
        (x, y) = location
        frame = Frame(tree, location=(x, y), name="Hats")

        direct_vertices = _hat_vertices_3d(rotation=0, ref=False)
        n = 13  # number of vertices of the hat tile
        scale = InputValue(tree, value=1, location=(x, y - 5), hide=False)

        circle_d = MeshCircle(tree, vertices=13, fill_type='NGON', location=(x + 1, y))
        index_d = Index(tree, location=(x, y - 1))
        vsw_d = IndexSwitch(tree, data_type='VECTOR', location=(x + 1, y - 1), hide=True)
        links.new(index_d.std_out, vsw_d.index)
        for _ in range(n - 2):
            vsw_d.new_item()
        for i, v in enumerate(direct_vertices):
            vsw_d.slots[i + 1].default_value = list(v)
        set_pos_d = SetPosition(tree, location=(x + 2, y),
                                geometry=circle_d.geometry_out,
                                position=vsw_d.std_out)

        # --- Reflected prototile ---
        reflect_verts = _hat_vertices_3d(rotation=0, ref=True)
        circle_r = MeshCircle(tree, vertices=n, fill_type='NGON', location=(x + 1, y - 3))
        index_r = Index(tree, location=(x, y - 4))
        vsw_r = IndexSwitch(tree, data_type='VECTOR', location=(x + 1, y - 4), hide=True)
        links.new(index_r.std_out, vsw_r.index)
        for _ in range(n - 2):
            vsw_r.new_item()
        for i, v in enumerate(reflect_verts):
            vsw_r.slots[i + 1].default_value = list(v)
        set_pos_r = SetPosition(tree, location=(x + 2, y - 3),
                                geometry=circle_r.geometry_out,
                                position=vsw_r.std_out)

        # --- 12 variants: direct k=5..0, then reflected k=5..0 ---
        # Each slot in the final IndexSwitch receives the G2I output in that order.
        g2i_outputs = []

        rr_unreflected = Reroute(tree, location=(x + 3, y - 2), name="unreflected", ins=set_pos_d.geometry_out)
        rr_reflected = Reroute(tree, location=(x + 3, y - 6), name="reflected", ins=set_pos_r.geometry_out)

        frame_nodes = []
        self.hat_materials = []
        for i, k in enumerate(range(0, 6)):
            angle = k * pi / 3.0
            xd = TransformGeometry(tree, location=(x + 4, y - i), hide=True,
                                   rotation=[0, 0, angle], scale=scale.std_out)
            g2i_d = GeometryToInstance(tree, location=(x + 5, y - i))
            material = get_texture(material="hat0" + str(i), **kwargs)
            self.materials.append(material)
            self.hat_materials.append(material)
            color = SetMaterial(tree, location=(x + 6, y - i), material=material, hide=True)

            create_geometry_line(tree, [xd, g2i_d, color], ins=rr_unreflected.geometry_out)
            g2i_outputs.append(color)
            frame_nodes += [xd, g2i_d, color]

        rot_map = {0: 2, 1: 3, 2: 4, 3: 5, 4: 0, 5: 1}
        for i in range(6):
            angle = rot_map[i] * pi / 3.0
            xr = TransformGeometry(tree, location=(x + 4, y - i - 6.5), hide=True,
                                   rotation=[0, 0, angle], scale=scale.std_out)
            g2i_r = GeometryToInstance(tree, location=(x + 5, y - i - 6.5))
            material = get_texture(material="hat1" + str(i), **kwargs)
            self.hat_materials.append(material)
            self.materials.append(material)
            color = SetMaterial(tree, location=(x + 6, y - i - 6.5), material=material, hide=True)
            create_geometry_line(tree, [xr, g2i_r, color], ins=rr_reflected.geometry_out)
            g2i_outputs.append(color)
            frame_nodes += [xr, g2i_r, color]

        # --- IndexSwitch GEOMETRY: 12 slots ---
        isw = IndexSwitch(tree, data_type='GEOMETRY', location=(x + 7, y))
        for _ in range(10):
            isw.new_item()
        for i, g2i in enumerate(g2i_outputs):
            links.new(g2i.geometry_out, isw.slots[i + 1])

        frame.add(
            [scale, circle_d, index_d, vsw_d, set_pos_d, circle_r, index_r, vsw_r, set_pos_r, rr_unreflected,
             rr_reflected,
             isw] + frame_nodes)
        return isw

    def create_node(self, tree, **kwargs):
        """
        Recreated cheap modifier: only four frames remain — GridGeneration,
        Hats, Fractal Index, GridpointVisualization, Finalize.  All the
        trapezoid / fundamental-domain-cover machinery is gone; the per-point
        hat orientation index (0-11) is computed directly by porting the
        ``hat_tile_fractal`` shader into geometry nodes (see ``_fractal_index``,
        which uses the ``InRegion`` node group from geometry_nodes/nodes.py).
        """
        links = tree.links
        out = self.group_outputs
        # Coord(tree, [-5, -20], [50, 20])
        grid_size = get_from_kwargs(kwargs, "grid_size", 10)
        right = 46

        hats = self._generate_hats(tree, location=(24, -3), **kwargs)
        tri_grid, tri_grid_vertices = self._generate_grid(tree, grid_size, location=(-5, -4), **kwargs)

        # iterate over the grid points (single-element context per point)
        position = Position(tree, location=(2, -1), hide=True)
        for_each = ForEachZone(tree, location=(3, 1), node_width=41, geometry=tri_grid.geometry_out)
        for_each.add_socket(socket_type="VECTOR", value=position.std_out, for_input=True, name="Position")

        # prepare coordinates and base
        torus, base_node = self._prepare_coordinate(tree, for_each, location=(5, 1))
        # per-point hat orientation index 0-11 from the ported shader pipeline
        index = self._fractal_index(tree, torus, base_node, location=(10, 5), **kwargs)

        hat_config = self._place_hats(tree, for_each.element, hats, index, location=(33, -1))
        links.new(hat_config.geometry_out, for_each.foreach_output.inputs[1])

        finalized_out = self._finalize(tree, hat_config, location=(37, -6))
        grid_points = self._grid_point_visualization(tree, tri_grid_vertices, location=(33, -6), **kwargs)
        links.new(finalized_out.geometry_out, for_each.foreach_output.inputs["Geometry"])

        geo_instance = GeometryToInstance(tree,location=(right,0),hide=True)
        rotation = InputVector(tree,location=(right,-1),vector=Vector(),name="FinalRotation",hide=True)
        pivot = InputVector(tree,location=(right,-1.5),vector=Vector(),name="Pivot",hide=True)
        rotate_instance = RotateInstances(tree,location=(right+1,-1),rotation=rotation.std_out,pivot_point=pivot.std_out,hide=True)
        final_join = JoinGeometry(tree, location=(right + 2, 0), name="FinalJoin")
        realize_instances = RealizeInstances(tree,location=(right+3,0),hide=True)
        out.location = ((right + 4) * 200, 0)
        create_geometry_line(tree, [grid_points, final_join])

        create_geometry_line(tree, [for_each,geo_instance,final_join,realize_instances], out=out.inputs[0])
        create_geometry_line(tree,[geo_instance,rotate_instance,final_join])

    def _prepare_coordinate(self, tree, for_each, location, **kwargs):
        (x, y) = location
        links = tree.links

        r32 = r3 / 2.0
        # torus span vectors / similarity transform (verbatim from _prepare_coordinates)
        U1, U2 = 3.118033988749895, r32
        V1, V2 = 0.8090169943749476, 3.1333093460129504
        B00, B01 = 0.3454915028125263, -0.0892055224432725
        B10, B11 = -0.09549150281252629, 0.34380717944894684

        # world -> uv (fundamental cell): B-matrix + frac per component
        world_to_uv = make_function(tree, name="WorldToUV", functions={
            "puv": [f"p_x,{B00},*,p_y,{B01},*,+,frac",
                    f"p_x,{B10},*,p_y,{B11},*,+,frac", "0"]
        }, inputs=["p"], outputs=["puv"], vectors=["p", "puv"],
                                    location=(x, y), hide=True)
        links.new(for_each.outputs["Position"], world_to_uv.inputs["p"])

        # lower-triangle selector
        triangle_selector = make_function(tree, name="LowerTriangleSelector", functions={
            "lower": "1,puv_x,puv_y,+,1,>,-"
        }, inputs=["puv"], outputs=["lower"], scalars=["lower"], vectors=["puv"],
                                          location=(x + 1, y), hide=True)
        links.new(world_to_uv.outputs["puv"], triangle_selector.inputs["puv"])

        # torus map (raw + customized)
        torus_raw = make_function(tree, name="UVToTorusRaw", functions={
            "uv": [f"puv_x,{U1},*,puv_y,{V1},*,+", f"puv_x,{U2},*,puv_y,{V2},*,+", "0"]
        }, inputs=["puv", "lower"], outputs=["uv"], scalars=["lower"],
                                  vectors=["puv", "uv"], location=(x + 2, y + 1), hide=True)
        links.new(world_to_uv.outputs["puv"], torus_raw.inputs["puv"])
        links.new(triangle_selector.outputs["lower"], torus_raw.inputs["lower"])

        torus = make_function(tree, name="TorusTRCustomized", functions={
            "uv": [f"lower,uv_x,*,1,lower,-,-1,uv_x,*,{U1},+,{V1},+,*,+",
                   f"lower,uv_y,*,1,lower,-,-1,uv_y,*,{U2},+,{V2},+,*,+", "0"]
        }, inputs=["uv", "lower"], outputs=["uv"], scalars=["lower"], vectors=["uv"],
                              location=(x + 3, y), hide=True)
        links.new(torus_raw.outputs["uv"], torus.inputs["uv"])
        links.new(triangle_selector.outputs["lower"], torus.inputs["lower"])

        base_constant = make_function(tree, name="baseConstant", functions={
            "base": "1,lower,-,10,*,lower,1,*,+"
        }, inputs=["lower"], outputs=["base"], scalars=["base", "lower"],
                                      location=(x + 3, y - 1), hide=True)
        links.new(triangle_selector.outputs["lower"], base_constant.inputs["lower"])

        frame = Frame(tree, location, name="Prepare Coordinates")
        frame.add([world_to_uv, base_constant, torus, torus_raw])
        return torus, base_constant

    def _make_in_region(self, tree, outer_repeat, location, **kwargs):
        links = tree.links
        (x, y) = location

        phi = (1 + sqrt(5)) / 2
        phi2 = phi + 1
        phim1 = 1.0 / phi
        r32 = r3 / 2.0

        # fractal membership from the initial uv (computed once).
        # The shader uses a ShaderNodeMix custom op with inputs=("B","A","Factor");
        # combined with make_function's rightmost-operand->"right"(=B->A swap)
        # fill order, the shader's "f,X,Y,mix" evaluates to A*(1-f)+B*f with
        # A=Y, B=X, i.e.  Y*(1-f) + X*f   (== algorithm.txt mix[a,b,f]=a(1-f)+bf
        # for "f,b,a,mix").  f is always a constant here, so reproduce that exact
        # expression with scale/add.  (A previous version returned X*(1-f)+Y*f,
        # which swapped X/Y and diverged from the shader on ~11% of points.)
        def _mix(f, X, Y):
            return f"{Y},{1.0 - f},scale,{X},{f},scale,add"

        # --- custom operator table (shader IfNode -> geometry Switch) ---------
        custom_ops = {
            "onRight": {
                "type": OnRightNode,
                "inputs": ("A", "B", "Position"),
                "output": "Result",
                "label": "onRight",
            },
            "ifv": {"type": Switch, "class_kwargs": {"input_type": "VECTOR"},
                    "inputs": ("True", "False", "Switch"), "output": "Output", "label": "ifv"},
            "ifb": {"type": Switch, "class_kwargs": {"input_type": "BOOLEAN"},
                    "inputs": ("True", "False", "Switch"), "output": "Output", "label": "ifb"},
        }

        # --- initial trapezoid corners (constants) ----------------------------
        vec_a = InputVector(tree, hide=True, value=Vector([0.0, 0.0, 0.0]), location=(x, y + 4), name="a")
        vec_b = InputVector(tree, hide=True, value=Vector([phi2, 0.0, 0]), location=(x, y + 3.5), name="b")
        vec_c = InputVector(tree, hide=True, value=Vector([phi2 - 0.5, r32, 0]), location=(x, y + 3), name="c")
        vec_d = InputVector(tree, hide=True, value=Vector([0.5, r32, 0]), location=(x, y + 2.5), name="d")

        # --- repeat zone carrying the IFS state -------------------------------
        depth = InputInteger(tree, location=(x, y + 2), name="Depth", integer=10)

        repeat = RepeatZone(tree, location=(x + 1, y + 3), node_width=6,
                            iterations=depth.std_out)
        # isTrap / result / exit are effectively booleans -> BOOLEAN sockets
        for socket_type, nm in [("VECTOR", "uv"), ("BOOLEAN", "isTrap"), ("BOOLEAN", "result"),
                                ("VECTOR", "a"), ("VECTOR", "b"), ("VECTOR", "c"),
                                ("VECTOR", "d"), ("BOOLEAN", "exit")]:
            repeat.add_socket(socket_type, nm)

        links.new(outer_repeat.outputs["uv"], repeat.repeat_input.inputs["uv"])
        repeat.repeat_input.inputs["isTrap"].default_value = True
        repeat.repeat_input.inputs["result"].default_value = False
        links.new(vec_a.std_out, repeat.repeat_input.inputs["a"])
        links.new(vec_b.std_out, repeat.repeat_input.inputs["b"])
        links.new(vec_c.std_out, repeat.repeat_input.inputs["c"])
        links.new(vec_d.std_out, repeat.repeat_input.inputs["d"])
        repeat.repeat_input.inputs["exit"].default_value = False

        rin = repeat.repeat_input.outputs  # values available inside the body
        rout = repeat.repeat_output.inputs  # values fed back into the loop

        inputs = ["a", "b", "c", "d"]
        aux1 = ["e1", "e2", "f1", "f2", "g1"]
        aux2 = ["a1", "b1", "c1", "d1", "k1", "l1", "m1", "n1",
                "a2", "b2", "c2", "d2", "k2", "l2"]
        conds = ["cond" + str(i + 1) for i in range(14)]
        variables_vector = ["a", "b", "c", "d"]
        variables_scalar = ["isTrap", "result", "exit"]
        update_vectors = ["new_a", "new_b", "new_c", "new_d"]
        update_scalars = ["new_isTrap", "new_result", "new_exit"]
        updates = update_scalars + update_vectors

        # aux1 -- primary subdivision points
        make_aux1 = make_function(tree, name="PrimarySubdivision", custom_ops=custom_ops,
                                  aux_functions={
                                      "e1": _mix(1.0 / phi2, "b", "a"),
                                      "e2": _mix(phim1, "c", "d"),
                                  }, functions={
                "e1": "e1",
                "e2": "e2",
                "f1": "b,a,e1,sub,add",
                "g1": "d,e1,a,sub,add",
                "f2": "b,d,e2,sub,add",
            }, inputs=inputs, outputs=aux1,
                                  vectors=inputs + aux1, location=(x + 2, y))
        for v in inputs:
            links.new(rin[v], make_aux1.inputs[v])

        # aux2 -- sub-region boundary points
        make_aux2 = make_function(tree, name="SubRegionBoundaryPoints", custom_ops=custom_ops, functions={
            "a1": _mix(phim1, "a", "e1"),
            "b1": _mix(phim1, "d", "e1"),
            "c1": _mix(phim1, "d", "e1"),
            "d1": _mix(phim1, "g1", "e1"),
            "k1": _mix(phim1, "g1", "e1"),
            "l1": _mix(phim1, "f1", "c"),
            "m1": _mix(phim1, "b", "c"),
            "n1": _mix(phim1, "f1", "c"),
            "a2": _mix(phim1, "d", "a"),
            "b2": _mix(phim1, "e2", "a"),
            "c2": _mix(phim1, "e2", "a"),
            "d2": _mix(phim1, "f2", "c"),
            "k2": _mix(phim1, "f2", "c"),
            "l2": _mix(phim1, "b", "c"),
        }, inputs=inputs + aux1, outputs=aux2,
                                  vectors=inputs + aux1 + aux2, location=(x + 3, y))
        for v in inputs:
            links.new(rin[v], make_aux2.inputs[v])
        for v in aux1:
            links.new(make_aux1.outputs[v], make_aux2.inputs[v])

        make_cond = make_function(tree, name="Conditions", custom_ops=custom_ops, functions={
            "cond1": "uv,d,e1,onRight",
            "cond2": "uv,a1,b1,onRight",
            "cond3": "uv,g1,e1,onRight",
            "cond4": "uv,c1,d1,onRight",
            "cond5": "uv,c,f1,onRight",
            "cond6": "uv,k1,f1,onRight",
            "cond7": "uv,l1,g1,onRight",
            "cond8": "uv,m1,n1,onRight",
            "cond9": "uv,e2,a,onRight",
            "cond10": "uv,a2,b2,onRight",
            "cond11": "uv,c,f2,onRight",
            "cond12": "uv,c2,f2,onRight",
            "cond13": "uv,d2,e2,onRight",
            "cond14": "uv,l2,k2,onRight",
        }, inputs=inputs + aux1 + aux2 + ["uv"], outputs=conds,
                                  vectors=inputs + aux1 + aux2 + ["uv"], booleans=conds,
                                  location=(x + 4, y))
        for v in inputs:
            links.new(rin[v], make_cond.inputs[v])
        for v in aux1:
            links.new(make_aux1.outputs[v], make_cond.inputs[v])
        for v in aux2:
            links.new(make_aux2.outputs[v], make_cond.inputs[v])
        links.new(rin["uv"], make_cond.inputs["uv"])

        # update -- per-branch new state (ifv switches vectors, ifb switches
        # booleans; "1,x,-" boolean negations rewritten as "x,not")
        update_values = make_function(tree, name="updateValues", custom_ops=custom_ops, functions={
            "new_a": "isTrap,cond1,cond2,a,d,ifv,cond3,cond4,a,g1,ifv,cond5,cond6,a,k1,ifv,cond8,a,f1,ifv,ifv,ifv,"
                     "ifv,cond9,cond10,a,e2,ifv,cond11,cond12,a,cond13,a,c2,ifv,ifv,cond14,a,f2,ifv,ifv,ifv,ifv",
            "new_b": "isTrap,cond1,cond2,b,a,ifv,cond3,cond4,b,d,ifv,cond5,cond6,b,f1,ifv,cond8,b,b,ifv,ifv,ifv,ifv,"
                     "cond9,cond10,b,d,ifv,cond11,cond12,b,cond13,b,f2,ifv,ifv,cond14,b,b,ifv,ifv,ifv,ifv",
            "new_c": "isTrap,cond1,cond2,c,a1,ifv,cond3,cond4,c,c1,ifv,cond5,cond6,c,l1,ifv,cond8,c,m1,ifv,ifv,ifv,"
                     "ifv,cond9,cond10,c,a2,ifv,cond11,cond12,c,cond13,c,d2,ifv,ifv,cond14,c,l2,ifv,ifv,ifv,ifv",
            "new_d": "isTrap,cond1,cond2,d,b1,ifv,cond3,cond4,d,d1,ifv,cond5,cond6,d,g1,ifv,cond8,d,n1,ifv,ifv,ifv,"
                     "ifv,cond9,cond10,d,b2,ifv,cond11,cond12,d,cond13,d,e2,ifv,ifv,cond14,d,k2,ifv,ifv,ifv,ifv",
            "new_isTrap": "isTrap,cond1,cond2,isTrap,1,ifb,cond3,cond4,isTrap,1,ifb,cond5,cond6,isTrap,0,ifb,cond8,"
                          "isTrap,1,ifb,ifb,ifb,ifb,cond9,cond10,isTrap,1,ifb,cond11,cond12,isTrap,cond13,isTrap,0,"
                          "ifb,ifb,cond14,isTrap,1,ifb,ifb,ifb,ifb",
            "new_result": "isTrap,cond1,result,not,cond3,result,not,cond5,cond6,result,not,result,ifb,result,ifb,ifb,"
                          "ifb,cond9,result,not,cond11,cond12,result,not,result,ifb,result,ifb,ifb,ifb",
            "new_exit": "isTrap,cond1,cond2,1,exit,ifb,cond3,cond4,1,exit,ifb,cond5,cond6,1,cond7,1,exit,ifb,ifb,"
                        "cond8,1,exit,ifb,ifb,ifb,ifb,cond9,cond10,1,exit,ifb,cond11,cond12,1,cond13,1,exit,ifb,ifb,"
                        "cond14,1,exit,ifb,ifb,ifb,ifb",
        }, inputs=variables_scalar + variables_vector + aux1 + aux2 + conds, outputs=updates,
                                      vectors=inputs + aux1 + aux2 + update_vectors + variables_vector,
                                      booleans=conds + update_scalars + variables_scalar,
                                      location=(x + 5, y + 4))
        for v in variables_scalar + variables_vector:
            links.new(rin[v], update_values.inputs[v])
        for v in aux1:
            links.new(make_aux1.outputs[v], update_values.inputs[v])
        for v in aux2:
            links.new(make_aux2.outputs[v], update_values.inputs[v])
        for v in conds:
            links.new(make_cond.outputs[v], update_values.inputs[v])

        # exit-check gate: freeze state once exit fired ("1,exit,-" -> "exit,not")
        variables = variables_scalar + variables_vector
        exit_check = make_function(tree, name="exitCheck", custom_ops=custom_ops, functions={
            "a": "exit,not,new_a,a,ifv",
            "b": "exit,not,new_b,b,ifv",
            "c": "exit,not,new_c,c,ifv",
            "d": "exit,not,new_d,d,ifv",
            "isTrap": "exit,not,new_isTrap,isTrap,ifb",
            "result": "exit,not,new_result,result,ifb",
            "exit": "exit,not,new_exit,exit,ifb",
        }, inputs=variables + updates, outputs=variables,
                                   vectors=variables_vector + update_vectors,
                                   booleans=variables_scalar + update_scalars,
                                   location=(x + 6, y))
        for v in variables:
            links.new(rin[v], exit_check.inputs[v])
            links.new(exit_check.outputs[v], rout[v])
        for v in updates:
            links.new(update_values.outputs[v], exit_check.inputs[v])
        # uv is carried unchanged through the loop
        links.new(rin["uv"], rout["uv"])

        # membership readout: If[exit == 0, True, result]  ==  exit ? result : True
        membership = make_function(tree, name="membership", custom_ops=custom_ops, functions={
            "m": "exit,result,1,ifb"
        }, inputs=["exit", "result"], outputs=["m"],
                                   booleans=["exit", "result", "m"], location=(x + 8, y - 1))
        links.new(repeat.repeat_output.outputs["exit"], membership.inputs["exit"])
        links.new(repeat.repeat_output.outputs["result"], membership.inputs["result"])

        frame = Frame(tree, location=location, name="InFractal")
        frame.add(
            [depth, repeat, vec_a, vec_b, vec_c, vec_d, membership, exit_check, update_values, make_cond, make_aux1,
             make_aux2])

        return membership.outputs["m"]

    def _fractal_index(self, tree, torus, base_node, location, **kwargs):
        """
        Per-point hat orientation index (0-11), ported 1:1 from the
        ``hat_tile_fractal`` shader (new_stuff/textures.py).

        Reuses the current element's ``Position`` as the coordinate (no
        TextureCoordinate node).  Pipeline:
          1. ``_prepare_coordinates`` math: world -> uv (B-matrix + frac) ->
             triangular selector -> torus map, plus the per-triangle ``base``.
          2. ``InRegion`` node group (geometry_nodes/nodes.py) computes the
             fractal membership ``inFractal`` once from the initial uv.
          3. Outer RepeatZone (3 iterations) accumulates ``result`` via the
             ``TriangleSelection`` check while ``Ruv`` rotates uv each step.
          4. ``prims`` -> ``Prim2ColorIndex`` -> integer color index = the hat
             orientation index.

        Note: the COLS / hat-material order is [hat00..hat05, hat10..hat15];
        ramp stop k/12 (k=1..12) maps to COLS[k-1], so the orientation index is
        ``round(col*12) - 1``.  If the result comes out shifted, this -1 offset
        (and the onRight ``<``/``>`` sense inside InRegion) are the knobs to tune.
        """
        (x, y) = location
        links = tree.links
        frame = Frame(tree, location=(x, y), name="Fractal Index")
        fnodes = []

        r32 = r3 / 2.0
        # torus span vectors / similarity transform (verbatim from _prepare_coordinates)
        U1, U2 = 3.118033988749895, r32

        # R transform (rotation plus shift by u), from hat_tile_fractal
        R00, R01 = -0.5, -r32
        R10, R11 = r32, -0.5

        # outer repeat zone: 3 iterations accumulating the triangle result
        rz = RepeatZone(tree, location=(x + 1, y), node_width=14, iterations=3)
        rz.add_socket("VECTOR", "uv")
        rz.add_socket("INT", "base")
        rz.add_socket("INT", "result")
        links.new(torus.outputs["uv"], rz.repeat_input.inputs["uv"])
        links.new(base_node.outputs["base"], rz.repeat_input.inputs["base"])
        rz.repeat_input.inputs["result"].default_value = 0

        in_region = self._make_in_region(tree, rz, location=(x + 3, y + 5), **kwargs)

        # cond/cond2/cond3/inFractal are booleans.  The mask-multiplications of
        # the original sum (cond2*cond*... terms) become a single if:
        #   result = (cond2 and cond) ? base+3*i + (inFractal ? 1 : cond3 ? 2 : 0)
        #                             : result
        # cond2 == (result == 0) marks the first triangle hit.  The cond AND of
        # the two boundary comparisons uses the boolean `and` operator.
        check = make_function(tree, name="TriangleSelection", custom_ops={
            "ifi": {"type": Switch, "class_kwargs": {"input_type": "INT"},
                    "inputs": ("True", "False", "Switch"), "output": "Output", "label": "ifi"},
        }, aux_functions={
            "cond": f"uv_y,{r32},<,uv_y,{r3},uv_x,*,<,and",
            "cond2": "result,0,=",
            "cond3": f"uv_y,{r3},1,uv_x,-,*,>",
        }, functions={
            "result": "cond2,cond,and,"
                      "base,3,i,*,+,inFractal,1,cond3,2,0,ifi,ifi,+,"
                      "result,ifi"
        }, inputs=["uv", "result", "base", "i", "inFractal"], outputs=["result"],
                              integers=["i", "result", "base"],
                              booleans=["cond", "cond2", "cond3", "inFractal"],
                              vectors=["uv"], location=(x + 13, y), hide=True)
        links.new(rz.iteration, check.inputs["i"])
        links.new(rz.repeat_input.outputs["uv"], check.inputs["uv"])
        links.new(rz.repeat_input.outputs["base"], check.inputs["base"])
        links.new(rz.repeat_input.outputs["result"], check.inputs["result"])
        links.new(in_region, check.inputs["inFractal"])
        links.new(check.outputs["result"], rz.repeat_output.inputs["result"])
        links.new(rz.repeat_input.outputs["base"], rz.repeat_output.inputs["base"])

        ruv = make_function(tree, name="Ruv", functions={
            "uv": [f"uv_x,{R00},*,uv_y,{R01},*,+,{U1},+",
                   f"uv_x,{R10},*,uv_y,{R11},*,+,{U2},+", "1"]
        }, inputs=["uv"], outputs=["uv"], vectors=["uv"], location=(x + 13, y + 1), hide=True)
        links.new(rz.repeat_input.outputs["uv"], ruv.inputs["uv"])
        links.new(ruv.outputs["uv"], rz.repeat_output.inputs["uv"])

        # result (0-18) -> prim value, via an IndexSwitch lookup table
        prim_values = [0, -5, 2, 5, -3, 6, 3, -1, 4, 1, -2, 5, 2, -6, 3, 6, -4, 1, 4]
        prims = IndexSwitch(tree, data_type="INT", location=(x + 16, y),
                            index=rz.repeat_output.outputs["result"])
        for _ in range(len(prim_values) - 2):  # IndexSwitch starts with 2 slots
            prims.new_item()
        for k, val in enumerate(prim_values):
            prims.slots[k + 1].default_value = val

        to_color = make_function(tree, name="Prim2ColorIndex", functions={
            "col": "prim,0,<,-2,*,1,+,prim,*,prim,0,<,6,*,+,1,-"
        }, inputs=["prim"], outputs=["col"], scalars=["col", "prim"],
                                 location=(x + 17, y), hide=True)
        links.new(prims.std_out, to_color.inputs["prim"])

        frame.add(fnodes + [rz, check, ruv, prims, to_color])
        return to_color.outputs["col"]

    def _place_hats(self, tree, element, hats, index_socket, location):
        """
        Drive the Hats IndexSwitch with the per-point orientation index, store it
        as the ``Orientation`` attribute (read by ``_finalize`` for the material
        switch), and instance the selected hat on the point.
        """
        (x, y) = location
        links = tree.links

        links.new(index_socket, hats.inputs[0])
        store = StoredNamedAttribute(tree, location=(x + 1, y), hide=True,
                                     name="Orientation", domain="POINT", value=index_socket)

        # Store the center of each placed hat as a "Center" attribute: the point
        # position ("OldPosition") plus the centroid (mean vertex position) of
        # the selected prototile.  The prototile is realized so the attribute
        # statistic can average over its vertices.
        realize = RealizeInstances(tree, location=(x, y - 2), geometry=hats.std_out)
        position = Position(tree, location=(x, y - 3))
        centroid = AttributeStatistic(tree, location=(x + 1, y - 2),
                                      data_type="FLOAT_VECTOR", domain="POINT",
                                      geometry=realize.geometry_out,
                                      attribute=position.std_out, std_out="Mean")
        old_position = NamedAttribute(tree, location=(x + 1, y - 1),
                                      data_type="FLOAT_VECTOR", name="OldPosition")
        center = VectorMath(tree, location=(x + 2, y - 1), operation="ADD",
                            inputs0=old_position.std_out, inputs1=centroid.std_out)
        store_center = StoredNamedAttribute(tree, location=(x + 2, y),
                                            name="Center", data_type="FLOAT_VECTOR",
                                            domain="POINT", value=center.std_out)

        iop = InstanceOnPoints(tree, location=(x + 3, y), instance=hats.std_out)
        create_geometry_line(tree, [store, store_center, iop], ins=element)
        frame = Frame(tree, location=(x, y), name="Hat Placement")
        frame.add([store, store_center, iop, realize, position, centroid,
                   old_position, center])
        return iop


class LabbeSelingerColorModifier(LabbeSelingerOptimizedModifier):
    """
    Copy of :class:`LabbeSelingerOptimizedModifier` with centralized color
    management.

    In ``LabbeSelingerOptimizedModifier`` the color is set in two different
    places: the hat geometry is initialized with a per-orientation material in
    ``_generate_hats`` and the wireframe ("frame") gets its color additionally
    in ``_finalize``.  Here the first color setting is removed; the instanced
    prototiles carry no material and the color is applied once, in
    ``_finalize``, to the *full* geometry (hat faces together with the
    wireframe).

    Two color schemes are offered, switched by the ``ColorScheme`` input
    (compare ``HatTileSubstitutionModifier._apply_colors``):

    * ``color_scheme=0`` -- one of the 12 orientation materials
      (``hat00``..``hat05``, ``hat10``..``hat15``); every rotation state gets
      its own color.
    * ``color_scheme=1`` -- the rotation state is ignored and only reflected
      and unreflected hats are distinguished (``blue`` for the 6 unreflected
      orientations 0-5, ``yellow`` for the 6 reflected orientations 6-11).
    """

    def __init__(self, color_scheme=0, **kwargs):
        self._color_scheme = color_scheme
        GeometryNodesModifier.__init__(self, 'LabbeSelingerColor',
                                       automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    # Hats without color: the per-orientation material is no longer baked
    # into the instanced prototiles; coloring happens once in _finalize.
    # ------------------------------------------------------------------
    def _generate_hats(self, tree, location, **kwargs):
        links = tree.links
        (x, y) = location
        frame = Frame(tree, location=(x, y), name="Hats")

        direct_vertices = _hat_vertices_3d(rotation=0, ref=False)
        n = 13  # number of vertices of the hat tile
        scale = InputValue(tree, value=1, location=(x, y - 5), hide=False)

        circle_d = MeshCircle(tree, vertices=13, fill_type='NGON', location=(x + 1, y))
        index_d = Index(tree, location=(x, y - 1))
        vsw_d = IndexSwitch(tree, data_type='VECTOR', location=(x + 1, y - 1), hide=True)
        links.new(index_d.std_out, vsw_d.index)
        for _ in range(n - 2):
            vsw_d.new_item()
        for i, v in enumerate(direct_vertices):
            vsw_d.slots[i + 1].default_value = list(v)
        set_pos_d = SetPosition(tree, location=(x + 2, y),
                                geometry=circle_d.geometry_out,
                                position=vsw_d.std_out)

        # --- Reflected prototile ---
        reflect_verts = _hat_vertices_3d(rotation=0, ref=True)
        circle_r = MeshCircle(tree, vertices=n, fill_type='NGON', location=(x + 1, y - 3))
        index_r = Index(tree, location=(x, y - 4))
        vsw_r = IndexSwitch(tree, data_type='VECTOR', location=(x + 1, y - 4), hide=True)
        links.new(index_r.std_out, vsw_r.index)
        for _ in range(n - 2):
            vsw_r.new_item()
        for i, v in enumerate(reflect_verts):
            vsw_r.slots[i + 1].default_value = list(v)
        set_pos_r = SetPosition(tree, location=(x + 2, y - 3),
                                geometry=circle_r.geometry_out,
                                position=vsw_r.std_out)

        # --- 12 variants: direct k=5..0, then reflected k=5..0 ---
        # Each slot in the final IndexSwitch receives the instanced prototile in
        # that order.  Unlike the optimized modifier, no material is set here.
        g2i_outputs = []

        rr_unreflected = Reroute(tree, location=(x + 3, y - 2), name="unreflected", ins=set_pos_d.geometry_out)
        rr_reflected = Reroute(tree, location=(x + 3, y - 6), name="reflected", ins=set_pos_r.geometry_out)

        frame_nodes = []
        self.hat_materials = []
        for i, k in enumerate(range(0, 6)):
            angle = k * pi / 3.0
            xd = TransformGeometry(tree, location=(x + 4, y - i), hide=True,
                                   rotation=[0, 0, angle], scale=scale.std_out)
            g2i_d = GeometryToInstance(tree, location=(x + 5, y - i))
            create_geometry_line(tree, [xd, g2i_d], ins=rr_unreflected.geometry_out)
            g2i_outputs.append(g2i_d)
            frame_nodes += [xd, g2i_d]

        rot_map = {0: 2, 1: 3, 2: 4, 3: 5, 4: 0, 5: 1}
        for i in range(6):
            angle = rot_map[i] * pi / 3.0
            xr = TransformGeometry(tree, location=(x + 4, y - i - 6.5), hide=True,
                                   rotation=[0, 0, angle], scale=scale.std_out)
            g2i_r = GeometryToInstance(tree, location=(x + 5, y - i - 6.5))
            create_geometry_line(tree, [xr, g2i_r], ins=rr_reflected.geometry_out)
            g2i_outputs.append(g2i_r)
            frame_nodes += [xr, g2i_r]

        # --- IndexSwitch GEOMETRY: 12 slots ---
        isw = IndexSwitch(tree, data_type='GEOMETRY', location=(x + 7, y))
        for _ in range(10):
            isw.new_item()
        for i, g2i in enumerate(g2i_outputs):
            links.new(g2i.geometry_out, isw.slots[i + 1])

        frame.add(
            [scale, circle_d, index_d, vsw_d, set_pos_d, circle_r, index_r, vsw_r, set_pos_r, rr_unreflected,
             rr_reflected,
             isw] + frame_nodes)
        return isw

    # ------------------------------------------------------------------
    # Single, full-geometry coloring with two selectable schemes.
    # ------------------------------------------------------------------
    def _finalize(self, tree, ins, location):
        (x, y) = location
        links = tree.links
        center = NamedAttribute(tree, location=(x, y - 2), hide=True, name="OldPosition", data_type="FLOAT_VECTOR")
        hat_scale = InputValue(tree, location=(x, y - 3), hide=True, value=0.95, name="HatScale")
        scale = ScaleElements(tree, location=(x, y), hide=True, scale=hat_scale.std_out, center=center.std_out)
        extrude_scale = InputValue(tree, location=(x, y - 2.5), value=0.15, name="ExtrudeScale", hide=True)

        extrude = ExtrudeMesh(tree, location=(x + 1, y), mode="FACES", offset_scale=extrude_scale.std_out, hide=True)

        extrude_compare = CompareNode(tree,location=(x+1,y-1),operation="LESS_EQUAL",data_type="FLOAT",inputs0=extrude_scale.std_out,inputs1=0.0001,hide=True)
        del_extrude = DeleteGeometry(tree,location=(x+2,y-1),selection=extrude_compare.std_out,hide=True)
        join = JoinGeometry(tree, location=(x + 2, y), hide=True)

        # per-point orientation (0-11), sampled from the single instance in the
        # current for-each element.
        orientation = NamedAttribute(tree, location=(x + 2, y - 1), name="Orientation",
                                     data_type="INT", hide=True)
        sample_index = SampleIndex(tree, location=(x + 2, y), name="SampleOrientation",
                                   hide=True, value=orientation.std_out, data_type="INT",
                                   geometry=ins.geometry_out, domain="INSTANCE")

        # --- color scheme 0: 12 orientation materials -------------------------
        isw = IndexSwitch(tree, data_type='MATERIAL',
                          location=(x + 3, y), index=sample_index.std_out)
        for _ in range(10):
            isw.new_item()
        count = 0
        colors = []
        self.hat_frame_materials = []
        for i in range(0, 2):
            for k in range(0, 6):
                material = get_texture(material="hat" + str(i) + str(k), emission=0.01)
                self.materials.append(material)
                self.hat_frame_materials.append(material)
                color = InputMaterial(tree, location=(x + 2, y - 2 - i * 3 - k * 0.5),
                                      material=material, hide=True)

                colors.append(color)
                links.new(color.std_out, isw.slots[count + 1])
                count += 1

        # --- color scheme 1: only reflected vs unreflected --------------------
        # orientations 0-5 are unreflected (blue), 6-11 reflected (yellow).
        blue_mat = get_texture(material="blue", emission=0.01)
        yellow_mat = get_texture(material="yellow", emission=0.01)
        self.materials += [blue_mat, yellow_mat]
        blue = InputMaterial(tree, location=(x + 2, y - 8), material=blue_mat, hide=True)
        yellow = InputMaterial(tree, location=(x + 2, y - 8.5), material=yellow_mat, hide=True)
        is_reflected = CompareNode(tree, data_type="INT", operation="GREATER_EQUAL",
                                   inputs0=sample_index.std_out, inputs1=6,
                                   location=(x + 2, y - 9), hide=True)
        scheme1 = Switch(tree, location=(x + 3, y - 8), input_type="MATERIAL",
                         switch=is_reflected.std_out, false=blue.std_out, true=yellow.std_out)

        # choose the active scheme (0 -> orientation, !=0 -> reflected/unreflected)
        color_scheme = InputInteger(tree, location=(x, y - 5), integer=self._color_scheme, name="ColorScheme")
        material_switch = Switch(tree, location=(x + 4, y), input_type="MATERIAL",
                                 switch=color_scheme.std_out, false=isw.std_out, true=scheme1.std_out)

        # --- wireframe ("frame") ----------------------------------------------
        wire_radius = InputValue(tree,location=(x,y-2),value=0.01,name="WireRadius",hide=True)
        toggle_solid = InputBoolean(tree, location=(x, y - 3), value=True, name="Solid",hide=True)
        toggle_wireframe = InputBoolean(tree, location=(x, y - 4), value=True, name="Wireframe",hide=True)
        not_solid = BooleanMath(tree, location=(x + 1, y - 3.5), operation="NOT", inputs0=toggle_solid.std_out, hide=True)
        not_wire = BooleanMath(tree, location=(x + 1, y - 4), operation="NOT", inputs0=toggle_wireframe.std_out, hide=True)
        wire_radius_function = make_function(tree,name="WireRadiusFunction",
                    functions={
                        "r":"r,1,reflected,0.3,*,+,*"
                    },inputs=["r","reflected"],outputs=["r"],
                    scalars=["r","reflected"],vectors=[],location=(x+3,y-7))
        links.new(wire_radius.std_out,wire_radius_function.inputs["r"])
        links.new(is_reflected.std_out,wire_radius_function.inputs["reflected"])

        wireframe = WireFrame(tree, location=(x + 4, y-6), radius=wire_radius_function.outputs["r"], hide=True)
        del_solid = DeleteGeometry(tree, location=(x + 4.5, y - 5), domain="POINT", mode="ALL",name="DeleteSolid",
                                  geometry=wireframe.geometry_out, selection=not_solid.std_out, hide=True)

        del_wire = DeleteGeometry(tree, location=(x + 4.5, y-6), domain="POINT", mode="ALL",name="DeleteWireframe",
                                  geometry=wireframe.geometry_out, selection=not_wire.std_out, hide=True)

        # combine hat faces and wireframe, then color the full geometry once
        join2 = JoinGeometry(tree, location=(x + 6, y), hide=True)
        mat = SetMaterial(tree, material=material_switch.std_out,
                          hide=True, location=(x + 7, y))

        create_geometry_line(tree, [ins, scale, join])
        create_geometry_line(tree, [scale, extrude, del_extrude,join,del_solid])
        create_geometry_line(tree, [join, wireframe, del_wire, join2])
        create_geometry_line(tree, [del_solid, join2,mat])

        frame = Frame(tree, location=(x, y), name="Finalize")
        frame.add(colors + [blue, yellow, is_reflected, scheme1, color_scheme, material_switch,
                            sample_index, isw, orientation, scale, extrude, join, wireframe, del_wire,
                            mat, hat_scale, center, toggle_wireframe, not_wire, join2])
        return mat


class HatTileSubstitutionModifier(GeometryNodesModifier):
    """
    Hat tiling built by *substitution* (port of video_hat_tile/mathematica/
    H7H8Supertiles.nb), as opposed to the index-vector method of
    ``LabbeSelingerOptimizedModifier``.

    Starting from the pair (t1, t2) = (H2, H1), a RepeatZone applies
    ``substitute`` ``level`` times; the result ``t1`` is a cluster of hats, each
    carrying ``dir``/``ref``/``pt`` attributes.  The per-hat orientation is kept
    as a stored ``Orientation`` attribute.  Two switchable color schemes:
    ``color_scheme=0`` colors by orientation (the 12 hat materials of
    ``LabbeSelingerOptimizedModifier``); ``color_scheme=1`` is the notebook
    scheme (light blue for direct hats, yellow for reflected ones).  A second
    geometry line shows the final control points (red spheres) and the cyclic
    line through them.
    """

    def __init__(self, level=3, color_scheme=0, tile_scale=1.0, extrude=True,
                 tile_selector=True, hat_scale=0.9,extrude_scale=31.59, show_control=False, **kwargs):
        self._level = level
        self._color_scheme = color_scheme
        self._tile_scale = tile_scale
        # control-frame toggles (booleans switch parts of the geometry on/off)
        self._extrude = extrude  # extrude the hat faces into 3D
        self._tile_selector = tile_selector  # False -> supertile t1, True -> t2
        self._extrude_scale = extrude_scale  # extrusion height
        self._show_control = show_control  # show the control-point line
        self._hat_scale = hat_scale  # show the control-point line
        GeometryNodesModifier.__init__(self, 'HatTileSubstitutionModifier',
                                       automatic_layout=False, **kwargs)

    # boolean ternary Switch op for make_function
    @staticmethod
    def _ifb_ops():
        return {"ifb": {"type": Switch, "class_kwargs": {"input_type": "BOOLEAN"},
                        "inputs": ("True", "False", "Switch"), "output": "Output", "label": "ifb"}}

    @staticmethod
    def _ifi_ops():
        return {"ifi": {"type": Switch, "class_kwargs": {"input_type": "INT"},
                        "inputs": ("True", "False", "Switch"), "output": "Output", "label": "ifi"}}

    # distinct (ref, pt) variants present in the H1/H2 seed (preserved by
    # substitution): (False,2), (False,6), (True,10)
    def _hat_variants(self):
        return [(False, 2), (False, 6), (True, 10)]

    @staticmethod
    def _variant_rpn(variants):
        """RPN mapping the ``(ref, pt)`` attributes to the variant index:
        sum over i of (pt == pt_i) * (ref == ref_i) * i.  Keying on ``ref`` as
        well as ``pt`` keeps the lookup unambiguous when the same ``pt`` occurs
        with both reflection flags (as in arbitrary cluster data)."""
        terms = []
        for i, (ref, pt) in enumerate(variants):
            ref_match = "ref,1,0,ifi" if ref else "ref,0,1,ifi"
            terms.append(f"pt,{pt},=,{ref_match},*,{i},*")
        rpn = terms[0]
        for term in terms[1:]:
            rpn += "," + term + ",+"
        return rpn

    def create_node(self, tree, **kwargs):
        links = tree.links
        out = self.group_outputs
        out.location = (28 * 200, 0)

        # --- control frame: all user-facing parameters bundled together --------
        level = InputInteger(tree, location=(-15.5, -2.5), integer=self._level, name="Level")
        extrude_sel = InputBoolean(tree, location=(-15.5, -3), value=self._extrude, name="ExtrudeSelector")
        color_scheme = InputInteger(tree, location=(-15.5, -3.5), integer=self._color_scheme, name="ColorScheme")
        tile_sel = InputBoolean(tree, location=(-15.5, -4), value=self._tile_selector, name="TileSelector")
        extrude_scale = InputValue(tree, location=(-15.5, -4.5), value=self._extrude_scale, name="ExtrudeScale")
        show_control = InputBoolean(tree, location=(-15.5, -5), value=self._show_control, name="ShowControl")
        hat_scale = InputValue(tree, location=(-15.5, -5.5), value=self._hat_scale, name="HatScale")
        Frame(tree, location=(-16, -2.5), name="ControlFrame").add(
            [level, extrude_sel, color_scheme, tile_sel, extrude_scale, show_control, hat_scale])

        # --- seed: (t1, t2) = (H2, H1) -----------------------------------------
        base_code = self._make_code_cloud(tree, H2_TILE["code"], location=(-22, 11), name="H2 Tile")
        base_cp = self._make_cp_cloud(tree, H2_TILE["cp"], location=(-18, -1), name="H2 ControlPoints")
        addon_code = self._make_code_cloud(tree, H1_TILE["code"], location=(-22, 9), name="H1 Tile")
        addon_cp = self._make_cp_cloud(tree, H1_TILE["cp"], location=(-18, 2), name="H1 ControlPoints")

        # --- substitution repeat zone -----------------------------------------
        repeat = RepeatZone(tree, location=(-14, 5), node_width=13, iterations=level.std_out)
        for nm in ["base_code", "addon_code", "base_cp", "addon_cp"]:
            repeat.add_socket("GEOMETRY", nm)
        links.new(base_code.geometry_out, repeat.repeat_input.inputs["base_code"])
        links.new(addon_code.geometry_out, repeat.repeat_input.inputs["addon_code"])
        links.new(base_cp.geometry_out, repeat.repeat_input.inputs["base_cp"])
        links.new(addon_cp.geometry_out, repeat.repeat_input.inputs["addon_cp"])
        repeat.del_socket(name="Geometry")
        # the repeat zone main geometry socket is unused (carry-through)

        self._substitute_gn(tree, repeat, location=(-12, 5))

        # --- TileSelector: pick which supertile of the final pair to show ------
        #     False -> t1 (base), True -> t2 (addon); drives both the hats and
        #     the control-point geometry.
        code_switch = Switch(tree, location=(0.5, 3.5), input_type="GEOMETRY", switch=tile_sel.std_out,
                             false=repeat.repeat_output.outputs["base_code"],
                             true=repeat.repeat_output.outputs["addon_code"])
        cp_switch = Switch(tree, location=(0.5, 1.5), input_type="GEOMETRY", switch=tile_sel.std_out,
                           false=repeat.repeat_output.outputs["base_cp"],
                           true=repeat.repeat_output.outputs["addon_cp"])
        final_code = code_switch.std_out
        final_cp = cp_switch.std_out

        # --- hats -----------------------------------------------------
        hats = self._instance_hats(tree, final_code, hat_scale.std_out,
                                   extrude_sel.std_out, extrude_scale.std_out, location=(8, 6), **kwargs)

        # bundle the hat geometry before it fans out to the two color schemes
        rr = Reroute(tree, location=(13, 0), ins=hats.geometry_out, name="Reroute")
        colored = self._apply_colors(tree, rr, color_scheme.std_out, location=(15, 0), **kwargs)

        # --- control point visualization (second geometry line) ----------------
        cp_viz = self._control_point_viz(tree, final_cp, location=(8, -4))
        # ShowControl toggles the control-point geometry (delete it when off)
        not_show = BooleanMath(tree, location=(24, -0.5), operation="NOT", inputs0=show_control.std_out)
        del_ctrl = DeleteGeometry(tree, location=(25, 0), domain="POINT", mode="ALL",
                                  geometry=cp_viz.geometry_out, selection=not_show.std_out)

        join = JoinGeometry(tree, location=(26.5, 0), name="Join Geometry.003")
        create_geometry_line(tree, [del_ctrl,join])
        create_geometry_line(tree, [colored,join], out=out.inputs[0])

    # ----- base-tile point clouds -----------------------------------------------

    def _make_cp_cloud(self, tree, cp_list, location, name="cp"):
        node, _ = self._point_cloud(tree, cp_list, location, name=name)
        return node

    def _make_code_cloud(self, tree, code_list, location, name="code"):
        positions = [h[0] for h in code_list]
        attrs = {
            "dir": ("INT", [h[1] for h in code_list]),
            "ref": ("BOOLEAN", [h[2] for h in code_list]),
            "pt": ("INT", [h[3] for h in code_list]),
        }
        node, _ = self._point_cloud(tree, positions, location, name=name, attrs=attrs)

        return node

    def _point_cloud(self, tree, positions, location, name="cloud", attrs=None):
        """Point cloud with one point per entry in ``positions`` (2D), positions
        set from the list and optional per-point attributes."""
        (x, y) = location
        links = tree.links
        s = self._tile_scale
        n = len(positions)
        pts = Points(tree, location=(x, y), count=n)
        idx = Index(tree, location=(x, y - 1), hide=True)
        psw = IndexSwitch(tree, data_type="VECTOR", location=(x + 1, y), index=idx.std_out, name="PosSwitch", hide=True)
        for _ in range(max(0, n - 2)):
            psw.new_item()
        for i, p in enumerate(positions):
            psw.slots[i + 1].default_value = [p[0] * s, p[1] * s, 0.0]
        last = SetPosition(tree, location=(x + 2, y), geometry=pts.geometry_out, position=psw.std_out, hide=True)
        nodes = [pts, idx, psw, last]
        if attrs:
            j = 0
            for attr_name, (dt, vals) in attrs.items():
                asw = IndexSwitch(tree, data_type=dt, location=(x + 3 + j, y - 0.5), name=attr_name + "Switch",
                                  index=idx.std_out, hide=True)
                for _ in range(max(0, n - 2)):
                    asw.new_item()
                for i, v in enumerate(vals):
                    asw.slots[i + 1].default_value = v
                store = StoredNamedAttribute(tree, location=(x + 4 + j, y), name=attr_name, label=attr_name + "Store",
                                             data_type=dt,
                                             value=asw.std_out, hide=True)
                links.new(last.geometry_out, store.geometry_in)
                last = store
                nodes += [asw, store]
                j += 1
        frame = Frame(tree, location=location, name=name)
        frame.add(nodes)
        return last, nodes

    # ----- substitution body ----------------------------------------------------

    def _substitute_gn(self, tree, repeat, location):
        """RepeatZone body implementing substitute[t1,t2] -> (t1', t2')."""
        (x, y) = location
        links = tree.links
        rin = repeat.repeat_input.outputs
        rout = repeat.repeat_output.inputs

        pos = Position(tree, location=(x, y + 3), hide=True)

        # bundle the repeat-input geometry sockets through reroutes before they
        # fan out to the many samples / placements
        rr_base_cp = Reroute(tree, location=(x - 1, y + 4.5), ins=rin["base_cp"], name="Reroute.003")
        rr_addon_cp = Reroute(tree, location=(x - 1, y + 1.5), ins=rin["addon_cp"], name="Reroute.004")
        rr_addon_code = Reroute(tree, location=(x + 2.5, y + 1.5), ins=rin["addon_code"], name="Reroute.005")

        def sample(cp_geo, i, yy, name):
            return SampleIndex(tree, location=(x + 1, yy), data_type="FLOAT_VECTOR", name=name,
                               domain="POINT", geometry=cp_geo, value=pos.std_out, index=i, hide=True)

        position_frame = Frame(tree, location=(x, y), name="PositionFrame")
        cp1 = sample(rr_base_cp.std_out, 0, y + 5.0, "base_cp1")
        cp4 = sample(rr_base_cp.std_out, 3, y + 4.5, "base_cp4")
        cp5 = sample(rr_base_cp.std_out, 4, y + 4.0, "base_cp5")
        cp6 = sample(rr_base_cp.std_out, 5, y + 3.5, "base_cp6")
        q2 = sample(rr_addon_cp.std_out, 1, y + 2.0, "addon_cp2")
        q3 = sample(rr_addon_cp.std_out, 2, y + 1.5, "addon_cp3")
        q4 = sample(rr_addon_cp.std_out, 3, y + 1.0, "addon_cp4")

        def add(a, b, yy):
            return VectorMath(tree, operation="ADD", location=(x + 2, yy), hide=True,
                              inputs0=a.std_out, inputs1=b.std_out)

        off5 = add(cp4, q2, y + 2.0)
        off6 = add(cp4, q3, y + 1.5)
        off7 = add(cp4, q4, y + 1.0)

        position_frame.add([cp1, cp4, cp5, cp6, q2, q3, q4, off5, off6, off7])

        # placements (source, offset_socket, dir-constant); T1 uses base_code
        # directly, T2..T7 share the addon_code reroute
        T1 = self._place_cluster(tree, rin["base_code"], cp1.std_out, 0, (x + 3, y + 5), "T1")
        T2 = self._place_cluster(tree, rr_addon_code.std_out, cp4.std_out, 0, (x + 3, y + 4), "T2")
        T3 = self._place_cluster(tree, rr_addon_code.std_out, cp6.std_out, 4, (x + 3, y + 3), "T3")
        T4 = self._place_cluster(tree, rr_addon_code.std_out, cp5.std_out, 2, (x + 3, y + 2), "T4")
        T5 = self._place_cluster(tree, rr_addon_code.std_out, off5.std_out, 8, (x + 3, y + 1), "T5")
        T6 = self._place_cluster(tree, rr_addon_code.std_out, off6.std_out, 10, (x + 3, y + 0), "T6")
        T7 = self._place_cluster(tree, rr_addon_code.std_out, off7.std_out, 0, (x + 3, y - 1), "T7")

        new_base = JoinGeometry(tree, location=(x + 7, y + 4))
        for T in [T1, T2, T3, T4, T5, T6]:
            links.new(T.geometry_out, new_base.geometry_in)
        new_t2 = JoinGeometry(tree, location=(x + 7, y + 1))
        for T in [T1, T2, T3, T4, T5, T6, T7]:
            links.new(T.geometry_out, new_t2.geometry_in)

        new_base_cp, new_t2_cp = self._new_control_points(
            tree, cp1, cp4, cp5, cp6, q2, q3, q4, location=(x + 7, y - 3))

        links.new(new_base.geometry_out, rout["base_code"])
        links.new(new_t2.geometry_out, rout["addon_code"])
        links.new(new_base_cp.geometry_out, rout["base_cp"])
        links.new(new_t2_cp.geometry_out, rout["addon_cp"])

    def _place_cluster(self, tree, geo, off_socket, d, location, name):
        """placeCluster: rotate each hat pos by d*30deg, translate by off, and set
        dir <- (ref ? dir-d : dir+d) mod 12 (ref/pt carried through)."""
        (x, y) = location
        links = tree.links
        ang = d * pi / 6.0
        c, s = cos(ang), sin(ang)
        pos = Position(tree, location=(x, y), hide=True)
        rot = make_function(tree, name="placePos", functions={
            "p": [f"pos_x,{c},*,pos_y,{s},*,-,off_x,+",
                  f"pos_x,{s},*,pos_y,{c},*,+,off_y,+", "0"]
        }, inputs=["pos", "off"], outputs=["p"], vectors=["pos", "off", "p"],
                            location=(x + 1, y), hide=True)
        links.new(pos.std_out, rot.inputs["pos"])
        links.new(off_socket, rot.inputs["off"])
        setpos = SetPosition(tree, location=(x + 2, y), geometry=geo, position=rot.outputs["p"], hide=True, name=name)

        dir_a = NamedAttribute(tree, location=(x, y - 0.5), name="dir", data_type="INT", hide=True)
        ref_a = NamedAttribute(tree, location=(x, y - 1.0), name="ref", data_type="BOOLEAN", hide=True)
        # NOTE: "%" is the scalar MODULO (ShaderNodeMath); "mod" is the VECTOR one.
        # (dir±d)+12 is always >0 so truncated "%" 12 equals the floored result.
        ndir = make_function(tree, name="placeDir", custom_ops=self._ifi_ops(), functions={
            "nd": f"ref,dir,{d},-1,*,+,dir,{d},+,ifi,12,+,12,%"
        }, inputs=["dir", "ref"], outputs=["nd"], integers=["dir", "nd", "ref"],
                             location=(x + 1, y - 0.7), hide=True)
        links.new(dir_a.std_out, ndir.inputs["dir"])
        links.new(ref_a.std_out, ndir.inputs["ref"])
        store = StoredNamedAttribute(tree, location=(x + 3, y), name="dir", data_type="INT",
                                     value=ndir.outputs["nd"], hide=True)
        links.new(setpos.geometry_out, store.geometry_in)
        return store

    def _new_control_points(self, tree, cp1, cp4, cp5, cp6, q2, q3, q4, location):
        """Compute the 6 new control points (newPt = q4) for t1 and t2."""
        (x, y) = location
        links = tree.links

        def rotsum(units, base, *adds):
            ang = units * pi / 6.0
            c, s = cos(ang), sin(ang)
            xexpr = f"{base}_x,{c},*,{base}_y,{s},*,-" + "".join(f",{a}_x,+" for a in adds)
            yexpr = f"{base}_x,{s},*,{base}_y,{c},*,+" + "".join(f",{a}_y,+" for a in adds)
            return [xexpr, yexpr, "0"]

        invars = ["cp1", "cp4", "cp5", "cp6", "q2", "q3", "q4"]
        f = make_function(tree, name="newControlPoints", functions={
            "c1": rotsum(0, "cp1"),
            "c2": rotsum(8, "q4", "cp4", "q2"),
            "c3": rotsum(10, "q4", "cp4", "q3"),
            "c4": rotsum(0, "q4", "cp4"),  # t1 control point 4
            "c5": rotsum(2, "q4", "cp5"),
            "c6": rotsum(4, "q4", "cp6"),
            "c4b": rotsum(0, "q4", "cp4", "q4"),  # t2 control point 4 (newPt+cp4+q4)
        }, inputs=invars, outputs=["c1", "c2", "c3", "c4", "c5", "c6", "c4b"],
                          vectors=invars + ["c1", "c2", "c3", "c4", "c5", "c6", "c4b"],
                          location=(x, y), hide=True)
        for nm, src in [("cp1", cp1), ("cp4", cp4), ("cp5", cp5), ("cp6", cp6),
                        ("q2", q2), ("q3", q3), ("q4", q4)]:
            links.new(src.std_out, f.inputs[nm])

        t1cp = self._cp_from_sockets(tree, [f.outputs["c1"], f.outputs["c2"], f.outputs["c3"],
                                            f.outputs["c4"], f.outputs["c5"], f.outputs["c6"]],
                                     location=(x + 1, y + 2), name="New Control Points Supertile 1")
        t2cp = self._cp_from_sockets(tree, [f.outputs["c1"], f.outputs["c2"], f.outputs["c3"],
                                            f.outputs["c4b"], f.outputs["c5"], f.outputs["c6"]],
                                     location=(x + 1, y - 1), name="New Control Points Supertile 2")
        return t1cp, t2cp

    def _cp_from_sockets(self, tree, sockets, location, name="cp"):
        (x, y) = location
        links = tree.links
        n = len(sockets)
        pts = Points(tree, location=(x, y), count=n)
        idx = Index(tree, location=(x, y - 1), hide=True)
        psw = IndexSwitch(tree, data_type="VECTOR", location=(x + 1, y), index=idx.std_out, hide=True)
        for _ in range(max(0, n - 2)):
            psw.new_item()
        for i, soc in enumerate(sockets):
            links.new(soc, psw.slots[i + 1])
        sp = SetPosition(tree, location=(x + 2, y), geometry=pts.geometry_out, position=psw.std_out, hide=True)
        Frame(tree, location=location, name=name).add([pts, idx, psw, sp])
        return sp

    # ----- hats + colors --------------------------------------------------------
    # this method is used in two modifiers
    # HatTileSubsitutionModifier and HatClusterCsvModifier
    def _instance_hats(self, tree, code_geo, hat_scale_socket,
                       extrude_sel_socket, extrude_scale_socket, location):
        (x, y) = location
        links = tree.links

        variants = self._hat_variants()
        instances = []
        for vi, (ref, pt) in enumerate(
                reversed(variants)):  # the reversed is necessary that they are joined in the correct order

            g2i = self._hat_mesh(tree, hat_scale_socket, ref, pt, location=(x, y - 2.5 * vi))
            instances.append(g2i)
        # core hat-layout nodes are gathered into the "HatLayout and Extrusion" frame
        layout = []
        join_inst = JoinGeometry(tree, location=(x + 4, y), name="variantsJoin")
        for g in instances:
            links.new(g.geometry_out, join_inst.geometry_in)
        layout.append(join_inst)

        # per-hat variant index from (ref, pt)
        ref_a = NamedAttribute(tree, location=(x, y + 2), name="ref", data_type="BOOLEAN", hide=True)
        pt_a = NamedAttribute(tree, location=(x, y + 1.5), name="pt", data_type="INT", hide=True)
        dir_a = NamedAttribute(tree, location=(x, y + 1), name="dir", data_type="INT", hide=True)
        var = make_function(tree, name="variant", custom_ops=self._ifi_ops(), functions={
            "v": self._variant_rpn(variants)
        }, inputs=["pt", "ref"], outputs=["v"], integers=["pt", "v"], booleans=["ref"],
                            location=(x + 2, y + 2), hide=True)
        links.new(pt_a.std_out, var.inputs["pt"])
        links.new(ref_a.std_out, var.inputs["ref"])

        # orientation index 0-11 (dir//2 + 6 if reflected) for the orientation scheme
        orient = make_function(tree, name="orientation", custom_ops=self._ifi_ops(), functions={
            "o": "ref,dir,2,/,6,+,dir,2,/,ifi"
        }, inputs=["dir", "ref"], outputs=["o"], integers=["dir", "o"], booleans=["ref"],
                               location=(x + 2, y + 1), hide=True)
        links.new(dir_a.std_out, orient.inputs["dir"])
        links.new(ref_a.std_out, orient.inputs["ref"])
        store_o = StoredNamedAttribute(tree, location=(x + 4, y + 2), name="Orientation", data_type="INT",
                                       value=orient.outputs["o"], hide=True)
        links.new(code_geo, store_o.geometry_in)

        # instance hats, rotate each by dir*30deg about Z
        iop = InstanceOnPoints(tree, location=(x + 6, y), instance=join_inst.geometry_out, pick_instance=True,
                               instance_index=var.outputs["v"])
        links.new(store_o.geometry_out, iop.geometry_in)
        rot = make_function(tree, name="hatRot", custom_ops=self._ifi_ops(), functions={
            "r": ["0", "0", f"dir,{pi / 6.0},*,ref,-1,1,ifi,*"]
        }, inputs=["dir", "ref"], outputs=["r"], integers=["dir"], booleans=["ref"], vectors=["r"],
                            location=(x + 6, y + 1), hide=True)
        links.new(dir_a.std_out, rot.inputs["dir"])
        links.new(ref_a.std_out, rot.inputs["ref"])
        rotinst = RotateInstances(tree, location=(x + 7, y), instances=iop.geometry_out,
                                  rotation=rot.outputs["r"])
        realize = RealizeInstances(tree, location=(x + 8, y))
        links.new(rotinst.geometry_out, realize.geometry_in)
        layout += [ref_a, pt_a, dir_a, var, orient, store_o, iop, rot, rotinst, realize]

        # --- extrusion (toggled by ExtrudeSelector) ----------------------------
        # extrude the realized faces; the duplicated extrusion geometry is
        # deleted again when ExtrudeSelector is off (NOT-gate on the selection).
        extrude = ExtrudeMesh(tree, location=(x + 9, y), mode="FACES", mesh=realize.geometry_out,
                              offset_scale=extrude_scale_socket, hide=True)
        extrude.node.inputs["Individual"].default_value = True
        not_ext = BooleanMath(tree, location=(x + 9, y - 1), operation="NOT", inputs0=extrude_sel_socket, hide=True)
        del_ext = DeleteGeometry(tree, location=(x + 10, y - 0.5), domain="POINT", mode="ALL",
                                 geometry=extrude.geometry_out, selection=not_ext.std_out, hide=True)
        ext_join = JoinGeometry(tree, location=(x + 11, y), name="Join Geometry.004")
        links.new(realize.geometry_out, ext_join.geometry_in)
        links.new(del_ext.geometry_out, ext_join.geometry_in)
        layout += [extrude, not_ext, del_ext, ext_join]
        Frame(tree, location=location, name="HatLayout and Extrusion").add(layout)
        return ext_join

    def _hat_mesh(self, tree, hat_scale_socket, ref, pt, location):
        (x, y) = location
        links = tree.links
        dir_in = -1
        if ref:
            dir_in = 1
        verts = _hat_vertices14(dir_in=dir_in, ref=ref, pt=_pt_to_pivot14(pt), scale=self._tile_scale)
        vert_vectors = [Vector([v[0], v[1], 0]) for v in verts]
        center = sum(vert_vectors, Vector()) / 14
        center_node = InputVector(tree, location=(x, y), value=center, name="Center",
                                  hide=True)
        n = len(verts)
        circle = MeshCircle(tree, vertices=n, fill_type='NGON', location=(x, y), hide=True)
        idx = Index(tree, location=(x, y - 1), hide=True)
        vsw = IndexSwitch(tree, data_type="VECTOR", location=(x + 1, y), index=idx.std_out, hide=True)
        for _ in range(max(0, n - 2)):
            vsw.new_item()
        for i, v in enumerate(vert_vectors):
            vsw.slots[i + 1].default_value = v
        scale_function = make_function(tree, name="ScaleFunction",
                                       functions={
                                           "pos": "pos,center,sub,s,scale,center,add"
                                       }, inputs=["pos", "center", "s"], outputs=["pos"],
                                       scalars=["s"], vectors=["pos", "center"], location=(x + 1.5, y - 1),
                                       hide=True)
        links.new(vsw.std_out, scale_function.inputs["pos"])
        links.new(center_node.std_out, scale_function.inputs["center"])
        links.new(hat_scale_socket, scale_function.inputs["s"])

        sp = SetPosition(tree, location=(x + 2, y), geometry=circle.geometry_out,
                         position=scale_function.outputs["pos"], hide=True)
        g2i = GeometryToInstance(tree, location=(x + 3, y))
        links.new(sp.geometry_out, g2i.geometry_in)
        Frame(tree, location=location, name=f"hat_{int(ref)}_{pt}").add(
            [circle, idx, vsw, sp, scale_function, center_node, g2i])
        return g2i

    def _apply_colors(self, tree, geo, color_scheme_socket, location, **kwargs):
        """Two color schemes, switched by ``color_scheme`` (0=orientation, 1=notebook)."""
        (x, y) = location
        links = tree.links

        # scheme 0: 12 orientation materials (hat00..hat05, hat10..hat15)
        orient_attr = NamedAttribute(tree, location=(x, y + 3), name="Orientation", data_type="INT", hide=True)
        a = geo
        idx = 0
        scheme0_nodes = [orient_attr]
        self.materials = getattr(self, "materials", [])
        for grp in range(2):
            for k in range(6):
                o = grp * 6 + k
                mat = get_texture(material="hat" + str(grp) + str(k), **kwargs)
                self.materials.append(mat)
                cmp = CompareNode(tree, location=(x + 1, y + 3 - o * 0.3), data_type="INT", operation="EQUAL",
                                  inputs0=orient_attr.std_out, inputs1=o, hide=True)
                sm = SetMaterial(tree, location=(x + 2, y + 3 - o * 0.3), material=mat,
                                 selection=cmp.std_out, hide=True)
                links.new(a.geometry_out, sm.geometry_in)
                a = sm
                scheme0_nodes += [cmp, sm]
                idx += 1
        scheme0 = a
        Frame(tree, location=(x, y + 3), name="LabbeSelingerColors").add(scheme0_nodes)

        # scheme 1: notebook colors (blue=direct, yellow=reflected)
        ref_attr = NamedAttribute(tree, location=(x, y - 3), name="ref", data_type="BOOLEAN", hide=True)
        blue = get_texture(material="blue", **kwargs)
        yellow = get_texture(material="yellow", **kwargs)
        self.materials += [blue, yellow]
        not_ref = make_function(tree, name="notRef", functions={"nr": "ref,not"}, inputs=["ref"], outputs=["nr"],
                                booleans=["nr", "ref"], location=(x, y - 3.5), hide=True)
        links.new(ref_attr.std_out, not_ref.inputs["ref"])
        sm_blue = SetMaterial(tree, location=(x + 2, y - 3), material=blue, selection=not_ref.outputs["nr"], hide=True)
        links.new(geo.geometry_out, sm_blue.geometry_in)
        sm_yellow = SetMaterial(tree, location=(x + 3, y - 3), material=yellow, selection=ref_attr.std_out, hide=True)
        links.new(sm_blue.geometry_out, sm_yellow.geometry_in)
        scheme1 = sm_yellow
        Frame(tree, location=(x, y - 3), name="SubstitutionColors").add([ref_attr, not_ref, sm_blue, sm_yellow])

        switch = Switch(tree, location=(x + 5, y), input_type="GEOMETRY",
                        switch=color_scheme_socket, false=scheme0.geometry_out, true=scheme1.geometry_out)

        return switch

    # ----- control point visualization -----------------------------------------

    def _control_point_viz(self, tree, cp_geo, location):
        (x, y) = location
        links = tree.links
        red = get_texture(material="red", roughness=0.1, emission=1)
        self.materials = getattr(self, "materials", [])
        self.materials.append(red)

        # bundle the incoming control-point geometry: one reroute inside the
        # frame feeds the spheres, a second (outside) feeds the line samples
        rr_in = Reroute(tree, location=(x + 0.5, y), ins=cp_geo, name="Reroute.002")
        rr_line = Reroute(tree, location=(x + 0.5, y - 5.5), ins=rr_in.std_out, name="Reroute.001")

        # red spheres on the 6 control points
        radius = InputValue(tree, location=(x, y - 1), value=0.15 * self._tile_scale, name="CPRadius", hide=True)
        sphere = IcoSphere(tree, location=(x, y - 2), radius=radius.std_out, hide=True)
        iop = InstanceOnPoints(tree, location=(x + 1, y), instance=sphere.geometry_out)
        links.new(rr_in.std_out, iop.geometry_in)
        spheres = SetMaterial(tree, location=(x + 2, y), material=red, hide=True)
        links.new(iop.geometry_out, spheres.geometry_in)

        # cyclic polyline through the 6 control points: a 7-vertex MeshLine whose
        # vertices are repositioned to cp[0..5, 0]
        line = MeshLine(tree, location=(x, y - 4), count=7, mode="END_POINTS")
        lidx = Index(tree, location=(x, y - 5), hide=True)
        lsw = IndexSwitch(tree, data_type="VECTOR", location=(x + 1, y - 4), index=lidx.std_out, hide=True)
        pos = Position(tree, location=(x, y - 6), hide=True)
        samp = [SampleIndex(tree, location=(x + 1, y - 5 - 0.3 * i), data_type="FLOAT_VECTOR", domain="POINT",
                            geometry=rr_line.std_out, value=pos.std_out, index=(i % 6), hide=True) for i in range(7)]
        for _ in range(7 - 2):
            lsw.new_item()
        for i in range(7):
            links.new(samp[i].std_out, lsw.slots[i + 1])
        lpos = SetPosition(tree, location=(x + 2, y - 4), geometry=line.geometry_out, position=lsw.std_out, hide=True)
        # tube along the edges (InstanceOnEdges expanded inline so the three
        # sub-nodes can be parented to the Control Points frame)
        mesh2curve = MeshToCurve(tree, location=(x + 3, y - 3.5), mesh=lpos.geometry_out)
        profile = CurveCircle(tree, location=(x + 3, y - 4.5), mode="RADIUS", resolution=8,
                              radius=0.04 * self._tile_scale, name="InstanceOnEdgesCircle")
        tubes = CurveToMesh(tree, location=(x + 4, y - 4), curve=mesh2curve.geometry_out,
                            profile_curve=profile.geometry_out)
        line_mat = SetMaterial(tree, location=(x + 5, y - 4), material=red, hide=True)
        links.new(tubes.geometry_out, line_mat.geometry_in)

        join = JoinGeometry(tree, location=(x + 6, y - 2))
        links.new(spheres.geometry_out, join.geometry_in)
        links.new(line_mat.geometry_out, join.geometry_in)
        frame = Frame(tree, location=location, name="Control Points")
        frame.add([join, mesh2curve, profile, tubes, line_mat, lpos, pos, lsw, line, lidx,
                   radius, sphere, iop, spheres, rr_in, rr_line] + samp)
        return join


class HatClusterCsvModifier(HatTileSubstitutionModifier):
    """
    Hat tiling read from a data file with one ``x,y,dir,ref,pt`` row per hat,
    e.g. ``video_hat_tile/data/H7_TILE.dat`` exported by
    ``scene_hat_tile.overlay`` from ``substitution_explainer.H7_TILE["code"]``.

    ``pt`` follows the notebook's 1-indexed pivot convention and is converted
    on load to the 0-indexed convention of ``_hat_vertices14``.  The hats are
    instanced like in ``HatTileSubstitutionModifier`` with the same two color
    schemes: ``color_scheme=0`` colors by orientation with the 12 hat
    materials of ``LabbeSelingerModifier``; ``color_scheme=1`` is the notebook
    scheme (blue direct / yellow reflected).
    """

    def __init__(self, file_path, color_scheme=0, tile_scale=1.0, extrude=True,
                 extrude_scale=0.15,wireframe=False, **kwargs):
        self._codes = self._read_codes(file_path)
        self._color_scheme = color_scheme
        self._tile_scale = tile_scale
        self._extrude = extrude
        self._extrude_scale = extrude_scale
        self._wireframe = wireframe
        GeometryNodesModifier.__init__(self, 'HatTileCsvModifier',
                                       automatic_layout=False, **kwargs)

    @staticmethod
    def _read_codes(file_path):
        """Parse ``x,y,dir,ref,pt`` rows into [(pos2d, dir, ref, pt)]; the
        1-indexed notebook ``pt`` becomes the 0-indexed pivot used here."""
        codes = []
        with open(file_path, newline="") as f:
            for row in csv.DictReader(f):
                codes.append((
                    (float(row["x"]), float(row["y"])),
                    int(row["dir"]),
                    row["ref"].strip().lower() in ("true", "1"),
                    (int(row["pt"]) - 1) % 14,
                ))
        return codes

    def _hat_variants(self):
        # (ref, pt) keys the variant lookup (see _variant_rpn), so the same pt
        # may safely appear with both reflection flags in the cluster data.
        return sorted({(ref, pt) for _, _, ref, pt in self._codes}, key=lambda v: (v[1], v[0]))

    def create_node(self, tree, **kwargs):
        links = tree.links

        # --- control frame ------------------------------------------------
        extrude_sel = InputBoolean(tree, location=(-9.5, -3), value=self._extrude, name="ExtrudeSelector")
        color_scheme = InputInteger(tree, location=(-9.5, -3.5), integer=self._color_scheme, name="ColorScheme")
        extrude_scale = InputValue(tree, location=(-9.5, -4), value=self._extrude_scale, name="ExtrudeScale")
        hat_scale = InputValue(tree, location=(-9.5, -4.5), value=0.99, name="HatScale")
        toggle_wireframe = InputBoolean(tree,location=(-9.5,-5),value=self._wireframe,name="Wireframe")
        wireframe_radius = InputValue(tree,location=(-9.5,-5.5),value=0.01,name="WireframeRadius")
        Frame(tree, location=(-10, -3), name="ControlFrame").add(
            [extrude_sel, color_scheme, extrude_scale, hat_scale,wireframe_radius,toggle_wireframe])

        # --- one point per data row, carrying dir/ref/pt attributes --------
        code_cloud = self._make_code_cloud(tree, self._codes, location=(-9, 5), name="DataCodes")

        # --- hats + colors (reused from HatTileSubstitutionModifier) -------
        hats = self._instance_hats(tree, code_cloud.geometry_out,hat_scale.std_out, extrude_sel.std_out,
                                   extrude_scale.std_out, location=(0, 5), **kwargs)

        wireframe = WireFrame(tree,location=(20,-3),radius=wireframe_radius.std_out,geometry=hats.geometry_out,hilde=True)
        wireframe_switch = Switch(tree,location=(21,-3),switch=toggle_wireframe.std_out,false=hats.geometry_out,
                                  true=wireframe.geometry_out,hide=True)

        colored = self._apply_colors(tree, wireframe_switch, color_scheme.std_out, location=(22, 0), **kwargs)

        trans = InputVector(tree,location=(31,-3),name="Translation",hide=True)
        transform_geo = TransformGeometry(tree,location=(32,0),translation=trans.std_out,hide=True)

        out = self.group_outputs
        out.location = (33 * 200, 0)
        create_geometry_line(tree,[colored,transform_geo],out=out.inputs[0])


# ---------------------------------------------------------------------------
# Dead-end cluster animation (dead_end_all_cluster.csv / dead_end_sym_tree.svg)
# ---------------------------------------------------------------------------
#
# The csv holds the 21 leaves ("dead corners") of the exhaustive
# 120-degree-symmetric growth tree around the 3-hat pinwheel.  Every cluster
# starts with the same three seed hats; the ``consistent`` column flags the
# three hats at the dead corner (drawn red in the svg).  The helpers below are
# deliberately bpy-free so the RPN strings and the mask bookkeeping can be
# unit-checked outside Blender.


def _read_dead_end_clusters(file_path):
    """Parse ``cluster,x,y,dir,ref,pt,consistent`` rows into a list of
    clusters (in file order); each cluster is a list of
    ``(x, y, dir, ref, pt, red)`` hats.  Clusters are separated by ``#``
    lines (the cluster id changing is what actually splits them) and the
    1-indexed notebook ``pt`` becomes the 0-indexed pivot used here
    (cf. ``HatClusterCsvModifier``)."""
    clusters = []
    current_id = None
    with open(file_path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "cluster":
                continue
            cid = int(row[0])
            if cid != current_id:
                clusters.append([])
                current_id = cid
            clusters[-1].append((
                float(row[1]), float(row[2]), int(row[3]),
                row[4].strip().lower() in ("true", "1"),
                (int(row[5]) - 1) % 14,
                row[6].strip().lower() in ("true", "1"),
            ))
    return clusters


def _dead_end_states(clusters):
    """Union the hats of all clusters into one code list plus per-hat state
    bitmasks.

    States: bit 0 = empty start state, bit 1 = seed pinwheel (the first three
    hats, identical in every cluster), bit ``c+2`` = cluster ``c`` in file
    order.  Returns ``(codes, member, red)`` where ``codes`` is
    ``[(pos2d, dir, ref, pt), ...]`` in first-appearance order and ``member``
    / ``red`` are parallel bitmask lists (which states contain the hat /
    flag it as a growth-blocking hat)."""

    index, codes, member, red = {}, [], [], []

    def slot(h):
        k = (round(h[0], 6), round(h[1], 6), h[2], h[3], h[4])
        if k not in index:
            index[k] = len(codes)
            codes.append(((h[0], h[1]), h[2], h[3], h[4]))
            member.append(0)
            red.append(0)
        return index[k]

    for h in clusters[0][:3]:  # seed pinwheel = state 1
        member[slot(h)] |= 1 << 1
    for c, cluster in enumerate(clusters):
        for h in cluster:
            s = slot(h)
            member[s] |= 1 << (c + 2)
            if h[5]:
                red[s] |= 1 << (c + 2)
    return codes, member, red


def _dead_end_rpn(n_states, r_min, r_span):
    """RPN strings (aux + main) for the per-hat presence/redness driven by
    the float ``ClusterTime`` input ``t``.

    ``k0 = floor(t)`` / ``k1 = k0+1`` select the adjacent states and
    ``f = t - k0`` is the transition progress; the per-hat state flags are
    bits ``k0``/``k1`` of the ``m`` (member) and ``r`` (red) masks — exact in
    float math since ``n_states <= 24``.  During a transition, hats missing
    from the next state shrink away in the first half and new hats grow in
    the second half, both radially staggered by the normalized anchor radius
    ``d``; hats present in both states keep presence 1 and only crossfade
    their redness mid-transition."""
    last = n_states - 1
    aux = {
        "k0": f"t,floor,0,max,{last},min",
        "f": "t,k0,-,0,max,1,min",
        "k1": f"k0,1,+,{last},min",
        "p0": "m,2,k0,**,/,floor,2,%",
        "p1": "m,2,k1,**,/,floor,2,%",
        "r0": "r,2,k0,**,/,floor,2,%",
        "r1": "r,2,k1,**,/,floor,2,%",
        "d": f"pos,length,{r_min},-,{r_span},/,0,max,1,min",
        # shrink-out ramp inside f in [0, 0.5), grow-in ramp inside [0.5, 1]
        "so": "f,0.05,d,0.30,*,+,-,0.12,/,0,max,1,min",
        "gi": "f,0.55,d,0.30,*,+,-,0.12,/,0,max,1,min",
        "sos": "so,so,*,3,2,so,*,-,*",
        "gis": "gi,gi,*,3,2,gi,*,-,*",
        # color crossfade around the middle of the transition
        "fc": "f,0.40,-,0.20,/,0,max,1,min",
    }
    functions = {
        "presence": "p0,p1,*,p0,1,p1,-,*,1,sos,-,*,+,1,p0,-,p1,*,gis,*,+",
        "redness": "p0,p1,*,r0,r1,r0,-,fc,*,+,*,p0,1,p1,-,*,r0,*,+,1,p0,-,p1,*,r1,*,+",
        "seed": "m,2,/,floor,2,%",
    }
    return aux, functions


def _dead_end_color_rpn(seed_color, blue_color, red_color):
    """Vector RPN for the per-hat color: seed hats (``sd`` = 1) use
    ``seed_color``, all other hats ``blue_color``; the result is faded to
    ``red_color`` by the redness ``rn``.  Colors are (r, g, b) in linear
    space."""
    comps = []
    for sc, bc, rc in zip(seed_color, blue_color, red_color):
        comps.append(f"sd,{sc},{bc},iff,1,rn,-,*,{rc},rn,*,+")
    return {"col": comps}


def _hex_to_linear(hex_color):
    """``#rrggbb`` (sRGB) -> (r, g, b) linear floats, rounded for RPN use."""
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(
        round(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, 6)
        for c in rgb)


class DeadEndClusterModifier(HatClusterCsvModifier):
    """
    Animated visualization of ``data/dead_end_all_cluster.csv`` — the 21
    leaves (dead corners) of the 120-degree-symmetric growth tree around the
    3-hat pinwheel (``data/dead_end_sym_tree.svg``).

    All hats of all clusters are united into one point cloud; per-hat
    ``Member``/``Red`` bitmasks record for each animation state (0 = empty,
    1 = seed pinwheel, 2.. = the leaves in file order) whether the hat exists
    and whether it is one of the hats that block further growth.  A single
    animatable float input ``ClusterTime`` (= state index + transition
    progress) drives the whole drawing: hats absent from the next state
    shrink away in the first half of a transition, new hats grow in the
    second half (both staggered radially, so clusters are drawn from the
    center outwards), and hats present in both states stay untouched — only
    their color crossfades (light blue <-> red for the blocking hats).  The
    seed pinwheel is present in every state, keeps a single color and turns
    red only in the final leaf, where the dead corner sits at the pinwheel
    itself.

    Per-hat ``Presence`` scales the faces around per-hat ``HatCenter``
    attributes (and the extrusion height along with it); ``HatColor`` is read
    by the modifier's single attribute-driven material.
    """

    SEED_COLOR = "#f39c12"  # orange, like the svg seed pinwheel
    BLUE_COLOR = "#a8d8ea"  # added hats
    RED_COLOR = "#e74c3c"   # hats at the dead corner

    def __init__(self, file_path, tile_scale=1.0, extrude=True,
                 extrude_scale=0.15, emission=0.02, **kwargs):
        self._clusters = _read_dead_end_clusters(file_path)
        self._codes, self._member, self._red = _dead_end_states(self._clusters)
        self._n_states = len(self._clusters) + 2
        self._tile_scale = tile_scale
        self._extrude = extrude
        self._extrude_scale = extrude_scale
        self._emission = emission
        GeometryNodesModifier.__init__(self, 'DeadEndClusterModifier',
                                       automatic_layout=False, **kwargs)

    # ---- data for the scene to pace the animation --------------------------

    @property
    def n_states(self):
        return self._n_states

    @property
    def transition_sizes(self):
        """[(removed, added)] hat counts for each transition s -> s+1."""
        sizes = []
        for s in range(self._n_states - 1):
            removed = added = 0
            for m in self._member:
                b0, b1 = (m >> s) & 1, (m >> (s + 1)) & 1
                removed += b0 & ~b1 & 1
                added += b1 & ~b0 & 1
            sizes.append((removed, added))
        return sizes

    @property
    def state_radii(self):
        """Maximal (scaled) anchor radius per state, 0 for the empty state."""
        radii = [0.0] * self._n_states
        for (pos, _, _, _), m in zip(self._codes, self._member):
            r = (pos[0] ** 2 + pos[1] ** 2) ** 0.5 * self._tile_scale
            for st in range(self._n_states):
                if (m >> st) & 1:
                    radii[st] = max(radii[st], r)
        return radii

    # ---- node graph ---------------------------------------------------------

    def create_node(self, tree, **kwargs):
        links = tree.links

        # --- control frame --------------------------------------------------
        cluster_time = InputValue(tree, location=(-12, -3), value=0.0, name="ClusterTime")
        extrude_sel = InputBoolean(tree, location=(-12, -3.5), value=self._extrude, name="ExtrudeSelector")
        extrude_scale = InputValue(tree, location=(-12, -4), value=self._extrude_scale, name="ExtrudeScale")
        hat_scale = InputValue(tree, location=(-12, -4.5), value=0.97, name="HatScale")
        Frame(tree, location=(-12.5, -3), name="ControlFrame").add(
            [cluster_time, extrude_sel, extrude_scale, hat_scale])

        # --- one point per union hat, carrying codes + animation masks -------
        positions = [p for p, _, _, _ in self._codes]
        attrs = {
            "dir": ("INT", [d for _, d, _, _ in self._codes]),
            "ref": ("BOOLEAN", [r for _, _, r, _ in self._codes]),
            "pt": ("INT", [p for _, _, _, p in self._codes]),
            "Member": ("INT", list(self._member)),
            "Red": ("INT", list(self._red)),
        }
        cloud, _ = self._point_cloud(tree, positions, location=(-11, 5),
                                     name="DeadEndCodes", attrs=attrs)

        # --- per-hat presence / redness / seed flag --------------------------
        s = self._tile_scale
        radii = [(px * s) ** 2 + (py * s) ** 2 for (px, py), _, _, _ in self._codes]
        r_min = min(radii) ** 0.5
        r_span = max(max(radii) ** 0.5 - r_min, 1.0e-6)
        aux, funcs = _dead_end_rpn(self._n_states, round(r_min, 6), round(r_span, 6))

        pos = Position(tree, location=(-5, 2), hide=True)
        member_a = NamedAttribute(tree, location=(-5, 1.5), name="Member", data_type="INT", hide=True)
        red_a = NamedAttribute(tree, location=(-5, 1), name="Red", data_type="INT", hide=True)
        state = make_function(tree, name="StateFunction", aux_functions=aux, functions=funcs,
                              inputs=["pos", "m", "r", "t"],
                              outputs=["presence", "redness", "seed"],
                              scalars=["t", "presence", "redness", "seed"] + list(aux.keys()),
                              integers=["m", "r"], vectors=["pos"],
                              location=(-3.5, 2), hide=True)
        links.new(pos.std_out, state.inputs["pos"])
        links.new(member_a.std_out, state.inputs["m"])
        links.new(red_a.std_out, state.inputs["r"])
        links.new(cluster_time.std_out, state.inputs["t"])

        # --- per-hat color ----------------------------------------------------
        color = make_function(tree, name="ColorFunction",
                              functions=_dead_end_color_rpn(_hex_to_linear(self.SEED_COLOR),
                                                            _hex_to_linear(self.BLUE_COLOR),
                                                            _hex_to_linear(self.RED_COLOR)),
                              inputs=["sd", "rn"], outputs=["col"],
                              scalars=["sd", "rn"], vectors=["col"],
                              custom_ops={"iff": custom_ops["iff"]},
                              location=(-2, 3), hide=True)
        links.new(state.outputs["seed"], color.inputs["sd"])
        links.new(state.outputs["redness"], color.inputs["rn"])

        # --- per-hat center (anchor + rotated local centroid) ----------------
        # the union data has a single pivot (csv pt=1 -> 0), so one local
        # centroid per reflection flag suffices; dir_in matches _hat_mesh.
        pivot = self._codes[0][3]
        c_false = _hat_vertices14(dir_in=-1, ref=False, pt=pivot, scale=s).mean(axis=0)
        c_true = _hat_vertices14(dir_in=1, ref=True, pt=pivot, scale=s).mean(axis=0)
        dir_a = NamedAttribute(tree, location=(-5, 0.5), name="dir", data_type="INT", hide=True)
        ref_a = NamedAttribute(tree, location=(-5, 0), name="ref", data_type="BOOLEAN", hide=True)
        center = make_function(tree, name="HatCenterFunction",
                               custom_ops={"iff": custom_ops["iff"], "ifi": custom_ops["ifi"]},
                               aux_functions={
                                   "sgn": "ref,-1,1,ifi",
                                   "th": f"dir,{pi / 6.0},*,sgn,*",
                                   "cx": f"ref,{round(c_true[0], 9)},{round(c_false[0], 9)},iff",
                                   "cy": f"ref,{round(c_true[1], 9)},{round(c_false[1], 9)},iff",
                               },
                               functions={
                                   "center": ["cx,th,cos,*,cy,th,sin,*,-,pos_x,+",
                                              "cx,th,sin,*,cy,th,cos,*,+,pos_y,+",
                                              "0"]
                               },
                               inputs=["pos", "dir", "ref"], outputs=["center"],
                               scalars=["sgn", "th", "cx", "cy"], integers=["dir"], booleans=["ref"],
                               vectors=["pos", "center"],
                               location=(-3.5, 0.5), hide=True)
        links.new(pos.std_out, center.inputs["pos"])
        links.new(dir_a.std_out, center.inputs["dir"])
        links.new(ref_a.std_out, center.inputs["ref"])

        # --- store animation attributes, drop fully absent hats ---------------
        store_presence = StoredNamedAttribute(tree, location=(-2, 5), name="Presence", data_type="FLOAT",
                                              domain="POINT", value=state.outputs["presence"], hide=True)
        store_color = StoredNamedAttribute(tree, location=(-1, 5), name="HatColor", data_type="FLOAT_COLOR",
                                           domain="POINT", value=color.outputs["col"], hide=True)
        store_center = StoredNamedAttribute(tree, location=(0, 5), name="HatCenter", data_type="FLOAT_VECTOR",
                                            domain="POINT", value=center.outputs["center"], hide=True)
        presence_a = NamedAttribute(tree, location=(0.5, 4), name="Presence", data_type="FLOAT", hide=True)
        absent = CompareNode(tree, location=(1, 4.5), data_type="FLOAT", operation="LESS_THAN",
                             inputs0=presence_a.std_out, inputs1=1.0e-4, hide=True)
        delete = DeleteGeometry(tree, location=(2, 5), domain="POINT", mode="ALL", selection=absent.std_out)
        create_geometry_line(tree, [cloud, store_presence, store_color, store_center, delete])
        Frame(tree, location=(-2, 5), name="Animation State").add(
            [pos, member_a, red_a, state, color, dir_a, ref_a, center,
             store_presence, store_color, store_center, presence_a, absent, delete])

        # --- hats --------------------------------------------------------------
        hats = self._instance_hats_animated(tree, delete.geometry_out, hat_scale.std_out,
                                            extrude_sel.std_out, extrude_scale.std_out,
                                            location=(4, 5))

        # --- material + output -------------------------------------------------
        mat = self._hat_color_material(emission=self._emission)
        self.materials = getattr(self, "materials", [])
        self.materials.append(mat)
        set_mat = SetMaterial(tree, location=(18, 5), material=mat, hide=True)

        trans = InputVector(tree, location=(19, 4), name="Translation", hide=True)
        transform_geo = TransformGeometry(tree, location=(20, 5), translation=trans.std_out, hide=True)

        out = self.group_outputs
        out.location = (21 * 200, 0)
        create_geometry_line(tree, [hats, set_mat, transform_geo], out=out.inputs[0])

    def _instance_hats_animated(self, tree, code_geo, hat_scale_socket,
                                extrude_sel_socket, extrude_scale_socket, location):
        """Like ``_instance_hats`` but scales every hat face by its
        ``Presence`` attribute around its ``HatCenter`` (the grow/shrink
        animation) and scales the extrusion height along with it."""
        (x, y) = location
        links = tree.links

        variants = self._hat_variants()
        instances = []
        for vi, (ref, pt) in enumerate(
                reversed(variants)):  # reversed so they are joined in the correct order
            g2i = self._hat_mesh(tree, hat_scale_socket, ref, pt, location=(x, y - 2.5 * vi))
            instances.append(g2i)
        layout = []
        join_inst = JoinGeometry(tree, location=(x + 4, y), name="variantsJoin")
        for g in instances:
            links.new(g.geometry_out, join_inst.geometry_in)
        layout.append(join_inst)

        # per-hat variant index from (ref, pt)
        ref_a = NamedAttribute(tree, location=(x, y + 2), name="ref", data_type="BOOLEAN", hide=True)
        pt_a = NamedAttribute(tree, location=(x, y + 1.5), name="pt", data_type="INT", hide=True)
        dir_a = NamedAttribute(tree, location=(x, y + 1), name="dir", data_type="INT", hide=True)
        var = make_function(tree, name="variant", custom_ops=self._ifi_ops(), functions={
            "v": self._variant_rpn(variants)
        }, inputs=["pt", "ref"], outputs=["v"], integers=["pt", "v"], booleans=["ref"],
                            location=(x + 2, y + 2), hide=True)
        links.new(pt_a.std_out, var.inputs["pt"])
        links.new(ref_a.std_out, var.inputs["ref"])

        # instance hats, rotate each by dir*30deg about Z
        iop = InstanceOnPoints(tree, location=(x + 6, y), instance=join_inst.geometry_out, pick_instance=True,
                               instance_index=var.outputs["v"])
        links.new(code_geo, iop.geometry_in)
        rot = make_function(tree, name="hatRot", custom_ops=self._ifi_ops(), functions={
            "r": ["0", "0", f"dir,{pi / 6.0},*,ref,-1,1,ifi,*"]
        }, inputs=["dir", "ref"], outputs=["r"], integers=["dir"], booleans=["ref"], vectors=["r"],
                            location=(x + 6, y + 1), hide=True)
        links.new(dir_a.std_out, rot.inputs["dir"])
        links.new(ref_a.std_out, rot.inputs["ref"])
        rotinst = RotateInstances(tree, location=(x + 7, y), instances=iop.geometry_out,
                                  rotation=rot.outputs["r"])
        realize = RealizeInstances(tree, location=(x + 8, y))
        links.new(rotinst.geometry_out, realize.geometry_in)
        layout += [ref_a, pt_a, dir_a, var, iop, rot, rotinst, realize]

        # grow/shrink: scale the flat hat faces around their per-hat centers
        presence_a = NamedAttribute(tree, location=(x + 8, y - 1), name="Presence", data_type="FLOAT", hide=True)
        center_a = NamedAttribute(tree, location=(x + 8, y - 1.5), name="HatCenter",
                                  data_type="FLOAT_VECTOR", hide=True)
        scale_faces = ScaleElements(tree, location=(x + 9, y), domain="FACE",
                                    scale=presence_a.std_out, center=center_a.std_out, hide=True)
        links.new(realize.geometry_out, scale_faces.geometry_in)
        layout += [presence_a, center_a, scale_faces]

        # --- extrusion (toggled by ExtrudeSelector), height follows presence ---
        ext_height = MathNode(tree, location=(x + 9, y - 2), operation="MULTIPLY",
                              inputs0=extrude_scale_socket, inputs1=presence_a.std_out, hide=True)
        extrude = ExtrudeMesh(tree, location=(x + 10, y), mode="FACES", mesh=scale_faces.geometry_out,
                              offset_scale=ext_height.std_out, hide=True)
        extrude.node.inputs["Individual"].default_value = True
        not_ext = BooleanMath(tree, location=(x + 10, y - 1), operation="NOT", inputs0=extrude_sel_socket, hide=True)
        del_ext = DeleteGeometry(tree, location=(x + 11, y - 0.5), domain="POINT", mode="ALL",
                                 geometry=extrude.geometry_out, selection=not_ext.std_out, hide=True)
        ext_join = JoinGeometry(tree, location=(x + 12, y), name="extrudeJoin")
        links.new(scale_faces.geometry_out, ext_join.geometry_in)
        links.new(del_ext.geometry_out, ext_join.geometry_in)
        layout += [ext_height, extrude, not_ext, del_ext, ext_join]
        Frame(tree, location=location, name="HatLayout and Extrusion").add(layout)
        return ext_join

    @staticmethod
    def _hat_color_material(attr_name="HatColor", **kwargs):
        """Material whose base and emission color come from the per-hat
        ``HatColor`` attribute written by the modifier."""
        import bpy
        from interface.ibpy import customize_material
        from interface.interface_constants import EMISSION
        from shader_nodes.shader_nodes import AttributeNode
        mat = bpy.data.materials.new(name="DeadEndHatColor")
        mat.use_nodes = True
        customize_material(mat, **kwargs)
        shader_tree = mat.node_tree
        bsdf = shader_tree.nodes.get("Principled BSDF")
        attr = AttributeNode(shader_tree, location=(-3, 0), attribute_name=attr_name)
        shader_tree.links.new(attr.std_out, bsdf.inputs["Base Color"])
        shader_tree.links.new(attr.std_out, bsdf.inputs[EMISSION])
        return mat


# ---------------------------------------------------------------------------
# Substitution construction (H7H8Supertiles.nb)
# ---------------------------------------------------------------------------
#
# A cluster/supertile = {controlPoints: 6 points, code: list of hats}, where
# each hat = (pos2d, dir, ref, pt) with dir in {0,2,..,10} (always even),
# ref a reflection flag and pt a pivot vertex index.  Base tiles (r = sqrt3):


# H1: 1 hat ; H2: 2 hats.  cp = 6 control points, code = [ (pos, dir, ref, pt) ]
H1_TILE = {
    "cp": [[0, 0], [-3 / 2, 5 * r3 / 2], [-3, 2 * r3], [-3, r3], [-9 / 2, -r3 / 2], [-3, 0]],
    "code": [([0, 0], 8, False, 6)],
}
H2_TILE = {
    "cp": [[0, 0], [-3 / 2, 5 * r3 / 2], [-3, 2 * r3], [-6, 0], [-9 / 2, -r3 / 2], [-3, 0]],
    "code": [([-9 / 2, -r3 / 2], 8, False, 2), ([-9 / 2, -r3 / 2], 2, True, 10)],
}


# notebook pt (1-indexed pivot) -> the `pt` argument of `_hat_vertices` (which is
# itself the port of the notebook's TileVertices).  Assumed identical mod 13;
# this is the main orientation/anchor knob to verify in Blender against the
# notebook screenshots (docs/Reitenbusch/Images/superTiles_*).
def _pt_to_pivot(pt):
    return pt % 13

def _pt_to_pivot14(pt):
    return pt % 14


class Turtle:
    def __init__(self, start=Vector()):
        self.start = to_vector(start)
        self.points = [self.start]
        self.direction = 0

    def step(self, length, direction):
        self.direction += direction
        rotation = Matrix([[np.cos(self.direction), -np.sin(self.direction), 0],
                           [np.sin(self.direction), np.cos(self.direction), 0], [0, 0, 1]]);
        rod = Vector([length, 0, 0])
        rot_rod = rotation @ rod
        self.points.append(self.points[-1] + rot_rod)


# walking instruction for the hat tile
instructions = {
    0: (1, -pi / 6),
    1: (1, pi / 3),
    2: (0, pi / 2),
    3: (0, pi / 3),
    4: (1, -pi / 2),
    5: (1, pi / 3),
    6: (0, pi / 2),
    7: (0, -pi / 3),
    8: (1, pi / 2),
    9: (1, -pi / 3),
    10: (0, pi / 2),
    11: (0, pi / 3),
    12: (0, 0),
    13: (0, pi / 3)
}


def turtle_vertices(lengths, code):
    '''
    generate the vertices of a hat tile
    code = [origin:Vector, rotation: int]
    '''
    turtle = Turtle(Vector([*code[0], 0]))
    for step, instruction in instructions.items():
        if step == 0:
            turtle.step(lengths[instruction[0]], instruction[1] + code[1] * pi / 6)
        else:
            turtle.step(lengths[instruction[0]], instruction[1])
    return turtle.points


def create_instruction_table():
    # generate walking instructions

    line = r"\begin{tabular}{r r r r r r} "
    line += r"Step & Turn & Walk & Step & Turn & Walk\\"
    # line += r"\hline "
    istrs = list(instructions.values())
    for i in range(7):
        line += f"{i + 1}& {round(180 / pi * istrs[i][1])}" + r"$^\circ$ & " + f"{"1" if bool(not istrs[i][0]) else r"$\sqrt{3}$"} & "
        line += f"{i + 8}& {round(180 / pi * istrs[i + 7][1])}" + r"$^\circ$ & " + f"{"1" if bool(not istrs[i + 7][0]) else r"$\sqrt{3}$"}" + r"\\"
    line += r"\hline"
    line += r"\end{tabular}"

    return line


class CommandmentTableModifier(GeometryNodesModifier):
    """Carve a LaTeX instruction table into a self-generated marble info panel.

    The whole panel (a flat slab plus two rounded side rails) is built *inside*
    the modifier, so it can be applied to any carrier mesh -- a plain cube will
    do; its incoming geometry is ignored.  The ``expression`` is rendered to SVG
    via the geo-fonts pipeline, imported as glyph curves, and the letters are
    sorted into reading order, progressively revealed, extruded to solid glyphs
    and subtracted from the panel with a Mesh Boolean (DIFFERENCE).

    Reveal order is established with a helper grid laid over the centred text:
    the grid faces are sorted top-to-bottom for the left half of the table
    (steps 1-7) and then top-to-bottom for the right half (steps 8-14), and each
    glyph inherits the order of its nearest grid face (plus a small x tie-break).
    An integer ``CurveCount`` input then keeps only the first *n* glyphs, so
    animating ``CurveCount`` from 0 upward carves the commandments line-wise,
    half by half.

    ``create_node`` is split into one private method per node-editor frame so the
    Python mirrors the ``tmp.xml`` layout: :meth:`_info_panel`,
    :meth:`_fill_curves_and_center_text`, :meth:`_lay_grid_over_text`,
    :meth:`_sort_grid_faces`, :meth:`_sort_text_points`, :meth:`_select_text`
    and :meth:`_extrude_text`.  The final ``Transform Geometry`` carries the
    optional ``location``/``rotation`` placement, and the marble material is
    registered in ``self.materials`` so it is added to / removed from the carrier
    object's slots automatically.
    """

    def __init__(self, expression, thickness=0.025,
                 location=(0.0, 0.0, 0.0125), rotation=(0.0, 0.0, 0.0),
                 final_scale=2.0, half_offset=1000.0, material='marble', **kwargs):
        self.collection_name = None
        self.expression = expression
        self.thickness = thickness
        self.location = location
        self.rotation = rotation
        self.final_scale = final_scale
        self.half_offset = half_offset
        self.material_name = material
        self.number_of_letters = 0
        self.curve_count = None
        # the panel is generated inside the tree, so the carrier mesh is ignored
        # (no group geometry input).  Every node is placed at an explicit
        # location matching ``tmp.xml``, so the automatic layout is disabled.
        kwargs.setdefault("automatic_layout", False)
        super().__init__(name="CommandmentTable",
                         group_input=False, group_output=True, **kwargs)

    # ------------------------------------------------------------------ frames
    #
    # Each frame is reproduced from ``video_hat_tile/tmp.xml``: nodes are placed
    # at ``frame_origin + xml_relative_location`` and grouped with ``Frame.add``,
    # mirroring the other modifiers in this module.  Reroute nodes are created
    # with the :class:`Reroute` wrapper (``ins=`` wires the upstream socket).
    def _info_panel(self, tree, location):
        """Frame "InfoPanel": flat slab plus two rounded side rails."""
        x, y = location
        cylinder = CylinderMesh(tree, location=(x + 0.1, y - 0.8), radius=0.2, depth=7.0)
        left_rail = TransformGeometry(tree, location=(x + 1.6, y - 0.5), hide=True,
                                      translation=[-4.0, -1.5, 0.0], rotation=[pi / 2, 0.0, 0.0],
                                      scale=[1.0, 1.0, 1.0])
        right_rail = TransformGeometry(tree, location=(x + 1.6, y - 1.0), hide=True,
                                       translation=[4.0, -1.5, 0.0], rotation=[pi / 2, 0.0, 0.0],
                                       scale=[1.0, 1.0, 1.0])
        join_rail = JoinGeometry(tree,location=(x+2.6,y-1))
        smooth = SetShadeSmooth(tree, location=(x + 3.6, y - 1))
        create_geometry_line(tree,[cylinder,left_rail,join_rail])
        create_geometry_line(tree,[cylinder,right_rail,join_rail,smooth])

        cube = CubeMesh(tree, location=(x + 0.1, y - 2.4), size=[1.0, 1.0, 1.0])
        slab = TransformGeometry(tree, location=(x + 1.6, y - 1.7),
                                 translation=[0.0, -0.5, 0.0], rotation=[pi / 2, 0.0, 0.0],
                                 scale=[8.0, 0.05, 4.0])
        shifted = TransformGeometry(tree, location=(x + 3.4, y - 0.4),
                                    translation=[0.0, 0.0, -0.01])
        create_geometry_line(tree,[cube,slab,shifted])

        Frame(tree, location=location, name="InfoPanel").add(
            [cylinder, left_rail, right_rail, cube, slab, shifted, smooth])
        return shifted,smooth

    def _fill_curves_and_center_text(self, tree, location):
        """Frame "Fill Curves And Center Text": import glyphs and centre them.

        Returns ``(uncentered, centered)`` sockets: the bounding box for the
        helper grid is taken from the uncentered text (size is translation
        invariant) while the sorting downstream uses the centred text.
        """
        x, y = location
        instances = CollectionInfo(tree, location=(x + 0.2, y - 0.7),
                                   collection_name=self.collection_name,
                                   separate_children=True, reset_children=False)
        realized = RealizeInstances(tree, location=(x + 1.1, y - 0.5),
                                    geometry=instances.geometry_out)
        reroute = Reroute(tree, location=(x + 5.2, y - 0.7), name="Reroute.002",
                          ins=realized.geometry_out)
        # centre the text block on the origin: offset every point by -mean(position)
        position = Position(tree, location=(x + 3.8, y - 2.3))
        mean = AttributeStatistic(tree, location=(x + 6.0, y - 1.3),
                                  data_type="FLOAT_VECTOR", domain="POINT",
                                  geometry=reroute.std_out, attribute=position.std_out,
                                  std_out="Mean")
        negated = VectorMath(tree, location=(x + 7.0, y - 1.2), operation="SCALE",
                             inputs0=mean.std_out, float_input=-1.0)
        centered = SetPosition(tree, location=(x + 8.7, y - 0.5),
                               geometry=reroute.std_out, offset=negated.std_out)
        Frame(tree, location=location, name="Fill Curves And Center Text").add(
            [instances, realized, reroute, position, mean, negated, centered])
        return reroute.std_out, centered.geometry_out

    def _lay_grid_over_text(self, tree, text, location):
        """Frame "Lay Grid over Text": a grid scaled to the text bounding box."""
        x, y = location
        bbox = BoundingBox(tree, location=(x + 0.1, y - 0.4), geometry=text)
        size = VectorMath(tree, location=(x + 1.3, y - 0.4), operation="SUBTRACT",
                          inputs0=bbox.node.outputs["Max"], inputs1=bbox.node.outputs["Min"])
        scaled = VectorMath(tree, location=(x + 2.2, y - 0.4), operation="MULTIPLY",
                            inputs0=size.std_out, inputs1=[0.8, 0.69, 0.0])
        sep = SeparateXYZ(tree, location=(x + 3.1, y - 0.3), vector=scaled.std_out)
        grid = Grid(tree, location=(x + 4.0, y - 0.1), size_x=sep.x, size_y=sep.y,
                    vertices_x=7, vertices_y=10)
        placed = TransformGeometry(tree, location=(x + 4.9, y - 0.1), geometry=grid.geometry_out,
                                   translation=[-0.3, -0.33, 0.0])
        Frame(tree, location=location, name="Lay Grid over Text").add(
            [bbox, size, scaled, sep, grid, placed])
        return placed.geometry_out

    def _sort_grid_faces(self, tree, grid, location):
        """Frame "Sort Grid Faces In Text Display Order": -y + half_offset*(x>0)."""
        x, y = location
        position = Position(tree, location=(x + 0.1, y - 0.4))
        sep = SeparateXYZ(tree, location=(x + 1.4, y - 1.2), vector=position.std_out)
        is_right = CompareNode(tree, location=(x + 2.1, y - 0.3), operation="GREATER_THAN",
                               data_type="FLOAT", inputs0=sep.x, inputs1=0.0)
        neg_y = MathNode(tree, location=(x + 2.4, y - 1.4), operation="MULTIPLY",
                         inputs0=sep.y, inputs1=-1.0)
        half = MathNode(tree, location=(x + 3.1, y - 0.1), operation="MULTIPLY",
                        inputs0=is_right.std_out, inputs1=self.half_offset)
        weight = MathNode(tree, location=(x + 3.4, y - 1.2), operation="ADD",
                          inputs0=neg_y.std_out, inputs1=half.std_out)
        ordered = SortElements(tree, location=(x + 5.4, y - 0.4), domain="FACE",
                               geometry=grid, sort_weight=weight.std_out)
        Frame(tree, location=location, name="Sort Grid Faces In Text Display Order").add(
            [position, sep, is_right, neg_y, half, weight, ordered])
        return ordered.geometry_out

    def _sort_text_points(self, tree, text, sorted_grid, location):
        """Frame "Sort Text Points": order glyphs by their nearest grid face."""
        x, y = location
        reroute = Reroute(tree, location=(x + 2.0, y - 2.0), name="Reroute.001", ins=text)
        position = Position(tree, location=(x + 0.1, y - 0.9))
        index = Index(tree, location=(x + 0.2, y - 1.7))
        # representative position of each glyph curve (sampled at the curve index)
        sampled = SampleIndex(tree, location=(x + 2.4, y - 1.0), data_type="FLOAT_VECTOR",
                              domain="CURVE", geometry=reroute.std_out,
                              value=position.std_out, index=index.std_out)
        nearest = SampleNearest(tree, location=(x + 3.3, y - 0.1), domain="FACE",
                                geometry=sorted_grid, sample_position=sampled.std_out)
        sep = SeparateXYZ(tree, location=(x + 1.0, y - 0.4), hide=True, vector=position.std_out)
        tie_break = MathNode(tree, location=(x + 1.9, y - 0.5), hide=True, operation="MULTIPLY",
                             inputs0=sep.x, inputs1=0.01)
        weight = MathNode(tree, location=(x + 4.1, y - 1.0), operation="ADD",
                          inputs0=nearest.std_out, inputs1=tie_break.std_out)
        ordered = SortElements(tree, location=(x + 5.0, y - 1.5), domain="CURVE",
                               geometry=reroute.std_out, sort_weight=weight.std_out)
        Frame(tree, location=location, name="Sort Text Points").add(
            [reroute, position, index, sampled, nearest, sep, tie_break, weight, ordered])
        return ordered.geometry_out

    def _select_text(self, tree, ordered_text, location):
        """Frame "Select Text": keep only the first ``CurveCount`` glyph curves."""
        x, y = location
        self.curve_count = InputInteger(tree, location=(x + 0.1, y - 0.1),
                                        integer=0, name="CurveCount")
        index = Index(tree, location=(x + 0.1, y - 1.0))
        # delete every curve whose sorted index exceeds CurveCount (CurveCount < index)
        beyond = CompareNode(tree, location=(x + 1.1, y - 0.4), operation="LESS_THAN",
                             data_type="INT", inputs0=self.curve_count.std_out,
                             inputs1=index.std_out)
        kept = DeleteGeometry(tree, location=(x + 3.6, y - 1.6), domain="CURVE", mode="ALL",
                              geometry=ordered_text, selection=beyond.std_out)
        Frame(tree, location=location, name="Select Text").add(
            [self.curve_count, index, beyond, kept])
        return kept.geometry_out

    def _extrude_text(self, tree, curves, location):
        """Frame "Extrude Text": fill the glyph curves and extrude to solids."""
        x, y = location
        resampled = ResampleCurve(tree, location=(x + 0.1, y - 0.2), mode="Count",
                                  curve=curves, count=30)
        filleted = FilletCurve(tree, location=(x + 1.0, y - 0.2), mode="Bézier",
                               curve=resampled.geometry_out, radius=0.01, count=1)
        filled = FillCurve(tree, location=(x + 1.9, y - 0.1), mode="N-gons",
                           curve=filleted.geometry_out)
        faces = Reroute(tree, location=(x + 3.0, y - 0.4), name="Reroute", ins=filled.geometry_out)
        extruded = ExtrudeMesh(tree, location=(x + 3.3, y - 0.6), mode="FACES",
                               mesh=faces.std_out, offset_scale=abs(self.thickness))
        extruded.node.inputs["Individual"].default_value = True
        solids = JoinGeometry(tree, location=(x + 4.3, y - 0.1),
                              geometry=[extruded.geometry_out, faces.std_out])
        Frame(tree, location=location, name="Extrude Text").add(
            [resampled, filleted, filled, faces, extruded, solids])
        return solids.geometry_out

    # --------------------------------------------------------------- assembly
    def create_node(self, tree, **kwargs):
        links = tree.links

        # render the table -> hidden collection of glyph curves
        self.number_of_letters = generate_expression(self.expression)
        self.collection_name = hashed_tex(self.expression)

        uncentered, centered = self._fill_curves_and_center_text(tree, location=(-3.7, 4.7))
        grid = self._lay_grid_over_text(tree, uncentered, location=(2.4, 7.0))
        sorted_grid = self._sort_grid_faces(tree, grid, location=(5.2, 9.5))
        ordered_text = self._sort_text_points(tree, centered, sorted_grid, location=(8.9, 7.1))
        revealed = self._select_text(tree, ordered_text, location=(14.9, 5.8))
        cutter = self._extrude_text(tree, revealed, location=(19.7, 4.4))
        panel,rail = self._info_panel(tree, location=(18.2, 8.5))

        # carve the revealed glyphs out of the panel
        boolean = MeshBoolean(tree, location=(24.7, 7.9), operation="DIFFERENCE", solver="EXACT",
                              mesh_1=panel.geometry_out, mesh_2=cutter,
                              self_intersection=True, hole_tolerant=True)

        # marble material -- registered in self.materials so it follows the object
        material_marble = get_texture(self.material_name, **kwargs)
        material_wood = get_texture("wood",**kwargs)

        self.materials.append(material_marble)
        self.materials.append(material_wood)
        mat = SetMaterial(tree, location=(27.1, 7.7),material=material_marble)
        mat2 = SetMaterial(tree,location=(27.1,8.7),material=material_wood)
        final_join= JoinGeometry(tree, location=( 28, 8))
        # final placement (carries the optional location / rotation kwargs)
        placed = TransformGeometry(tree, location=(29.0, 7.4),
                                   translation=list(self.location), rotation=list(self.rotation),
                                   scale=[self.final_scale] * 3)
        create_geometry_line(tree,[boolean,mat,final_join])
        create_geometry_line(tree,[rail,mat2,final_join,placed],out=self.group_output.inputs[0])
        self.group_output.location = (30 * 200, 7.7 * 100)


if __name__ == '__main__':
    """
    create hat tile graphics with matplotlib
    from the vertices generated by tile_vertices
    """

    vertices = turtle_vertices([1, r3], [[0, 0], 0])
    vertices.append(vertices[0])

    x, y, z = np.transpose(vertices)
    plt.plot(x, y, marker='o')
    plt.gca().set_aspect('equal')
    plt.show()

    print(create_instruction_table())

