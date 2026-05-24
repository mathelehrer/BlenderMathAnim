from mathutils import Vector
from numpy import pi, sqrt, zeros, cumsum, array, cos, sin, roll

from appearance.textures import get_texture
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (
    Grid, InputValue, InstanceOnPoints, JoinGeometry,
    create_geometry_line, Position, make_function, SetPosition,
    SetMaterial, NamedAttribute, WireFrame,
    InputVector,
    TransformGeometry, RealizeInstances,
    CompareNode, InputInteger,
    MathNode, ExtrudeMesh, SampleNearest,
    SampleIndex, Reroute, ComplexMathNode, PointsToVertices, Points, Quadrilateral,
    FillCurve, StoredNamedAttribute, ScaleElements,
    GeometryToInstance, RotateInstances, TranslateInstances, ForEachZone, Index, SeparateGeometry, RepeatZone,
    UVSphere, InsidePolygon, IndexSwitch, Frame,
    MeshCircle, MeshToPoints, CombineXYZ, BevelNode,
)
from interface.ibpy import create_mesh
from objects.bobject import BObject
from utils.kwargs import get_from_kwargs

pi = pi
r3 = sqrt(3)

HAT_SCALE = 1.0 / (2.0 * r3)


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


def _hat_vertices_3d(rotation=0, ref=False, hat_scale=HAT_SCALE, pivot=0):
    """Return list of Vector-like (x, y, 0) tuples for use as Blender vertices."""
    verts2d = _hat_vertices(dir_in=rotation, ref=ref, scale=hat_scale, pt=(9 + pivot) % 13)
    return [(v[0], v[1], 0.0) for v in verts2d]


# ---------------------------------------------------------------------------
# Basic BObject
# ---------------------------------------------------------------------------

class HatTile(BObject):
    def __init__(self, name="HatTile", **kwargs):
        rotation = get_from_kwargs(kwargs, "rotation", 0)
        reflection = get_from_kwargs(kwargs, "reflection", False)
        hat_scale = get_from_kwargs(kwargs, "hat_scale", 1)
        location = get_from_kwargs(kwargs, "location", (0, 0))
        pivot = get_from_kwargs(kwargs, "pivot", 0)
        hat_scale = HAT_SCALE * hat_scale
        if len(location) == 2:
            location = Vector([*location, 0])
        super().__init__(
            mesh=create_mesh(_hat_vertices_3d(rotation, reflection, hat_scale, pivot), faces=[list(range(13))]),
            name=name, location=location, **kwargs)

    @classmethod
    def from_code(cls, code=((0, 0), 0, False, 0), **kwargs):
        """
        Compatability method that lays out hat tiles according to the code used in the mathematica notebooks
        the first tuple holds the location
        the amount of rotation
        the reflection
        the pivot
        """
        return HatTile(location=code[0], rotation=code[1], reflection=code[2], pivot=code[3], **kwargs)


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

    def create_node(self, tree, grid_size=10, **kwargs):
        links = tree.links
        out = self.group_outputs
        # move group output far away
        right = 60
        out.location = (right * 200, 0)

        # Coord(tree, min=[30, -20], max=[right, 20])
        hats = self._generate_hats(tree, location=(43,3),**kwargs)
        tri_grid = self._generate_grid(tree, grid_size, location=(-20, 5))

        # prepare fundamental domain and trapezoid cover
        (one, zero, xi_function, phi_function, u_function, v) = self._create_constants(tree, location=(-20, 3))
        g_trapzoid = self._create_trapezoid(tree, zero, one, xi_function, phi_function, location=(-18, 3))
        cover, trapezoids = self._make_cover(tree, g_trapzoid, u_function, v, location=(-14, 2))

        # setup backtranslation
        back = self._back_translation(tree, tri_grid, u_function, v, location=(-13, 5))

        sample_points = self._sample_points_in_cover(tree, cover, back, location=(-9, 5))

        for_each = ForEachZone(tree, location=(3, 3), node_width=49, geometry=sample_points.geometry_out)

        # trapezoid selector
        trap_selector, trap_index = self._trapezoid_selector(tree, trapezoids, for_each.element,location=(4, 3))
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

        final = self._finalize(tree,for_each,location=(55,0))

        create_geometry_line(tree, [for_each,final], out=out.inputs[0])

    def _finalize(self, tree, ins, location):
        (x, y) = location

        scale = ScaleElements(tree,location=(x,y),hide=True,scale=0.95)
        extrude = ExtrudeMesh(tree,location=(x+1,y),mode="FACES",offset_scale=0.3,hide=True)

        join = JoinGeometry(tree,location=(x+2,y),hide=True)
        wireframe = WireFrame(tree,location=(x+3,y),radius=0.01,hide=True)
        mat = SetMaterial(tree,material=get_texture(material="gray_8",roughness=0.1),hide=True,location=(x+4,y))
        join2 = JoinGeometry(tree, location=(x + 5, y), hide=True)
        create_geometry_line(tree,[ins,scale,join])
        create_geometry_line(tree,[scale,extrude,join])
        create_geometry_line(tree,[join,wireframe,mat,join2])
        create_geometry_line(tree,[join,join2])
        frame = Frame(tree, location=(x, y), name="Finalize")
        frame.add([scale,extrude,join,wireframe,mat,join2])
        return join2

    def _hat_config(self, tree, point, hats, location):
        (x, y) = location
        links = tree.links


        exit_type = NamedAttribute(tree, location=(x, y), data_type="INT", name="Type", hide=True)
        bottom = NamedAttribute(tree, location=(x, y - 1), data_type="INT", name="Bottom", hide=True)
        top = NamedAttribute(tree, location=(x, y - 2), data_type="INT", name="Top", hide=True)

        sample_type = SampleIndex(tree,location=(x+1,y),data_type="INT",geometry=point.geometry_out,
                                  value=exit_type.std_out,hide=True,index=0)
        sample_bottom = SampleIndex(tree,location=(x+1,y-1),data_type="INT",geometry=point.geometry_out,
                                    value=bottom.std_out,hide=True,index=0)
        sample_top = SampleIndex(tree,location=(x+1,y-2),data_type="INT",geometry=point.geometry_out,
                                  value=top.std_out,hide=True,index=0)

        orientation_switch = IndexSwitch(tree, location=(x + 1, y), data_type="INT", name="Orientation", hide=True,
                                         index=sample_type.std_out)
        orientation_switch.new_item()
        links.new(sample_bottom.std_out, orientation_switch.slots[2])
        links.new(sample_top.std_out, orientation_switch.slots[3])

        # link orientation to hat switch
        links.new(orientation_switch.std_out, hats.inputs[0])
        iop = InstanceOnPoints(tree, location=(x + 3, y),instance=hats.std_out)
        create_geometry_line(tree, [point, iop])
        frame = Frame(tree, location=(x, y), name="Hat Placement")
        frame.add([exit_type, bottom, top, orientation_switch, iop])
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
        frame.add([ old_bottom, old_top, old_left, bottom_switch, left_switch, top_switch])
        create_geometry_line(tree, [ old_bottom, old_top, old_left], ins=sample)
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

    def _trapezoid_selector(self, tree, trapezoids, sample_point,  location):
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

    def _generate_grid(self, tree, grid_size=10, location=(0, 0)):
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
        shift_x = InputValue(tree,location=(x,y),value=0.1,name="ShiftX",hide=True)
        shift_y = InputValue(tree,location=(x,y),value=0.01,name="ShiftX",hide=True)
        combine = CombineXYZ(tree,location=(x+1,y-1),x=shift_x.std_out,y=shift_y.std_out,name="GridShift",hide=True)

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

        set_pos = SetPosition(
            tree, location=(x + 4, y + 2),
            position=grid_function.outputs["out"],
            name='Set Position',
            offset=combine.std_out,
            hide=True
        )

        mesh2points = MeshToPoints(tree, location=(x + 5, y + 2), hide=True)

        pos2 = Position(tree, location=(x+6, y + 0.5), hide=True)
        old_pos = StoredNamedAttribute(tree, location=(x+6, y), name="OldPosition", data_type="FLOAT_VECTOR",
                                       domain="POINT",
                                       hide=True, value=pos2.std_out)

        create_geometry_line(tree, [grid, set_pos, mesh2points,old_pos])

        frame.add([shift_x,shift_y,combine,grid_size_node, plus_one, grid, pos,pos2, grid_function, set_pos, mesh2points,old_pos])
        return old_pos

    def _generate_hats(self, tree, location, **kwargs):
        links = tree.links
        (x, y) = location
        frame = Frame(tree, location=(x, y), name="Hats")

        direct_vertices = _hat_vertices_3d(rotation=0, ref=False)
        n = 13  # number of vertices of the hat tile
        scale = InputValue(tree,value=1,location=(x,y-5),hide=False)

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
        for i, k in enumerate(range(0,6)):
            angle = k * pi / 3.0
            xd = TransformGeometry(tree, location=(x + 4, y - i), hide=True,
                                   rotation=[0, 0, angle],scale=scale.std_out)
            g2i_d = GeometryToInstance(tree, location=(x + 5, y - i))
            material = get_texture(material="hat0" + str(i), **kwargs)
            self.materials.append(material)
            color = SetMaterial(tree, location=(x + 6, y - i), material=material, hide=True)

            create_geometry_line(tree, [xd, g2i_d, color], ins=rr_unreflected.geometry_out)
            g2i_outputs.append(color)
            frame_nodes += [xd, g2i_d, color]

        rot_map={0:2,1:3,2:4,3:5,4: 0,5:1}
        for i in range(6):
            angle = rot_map[i] * pi / 3.0
            xr = TransformGeometry(tree, location=(x + 4, y - i - 6.5), hide=True,
                                   rotation=[0, 0, angle],scale=scale.std_out)
            g2i_r = GeometryToInstance(tree, location=(x + 5, y - i - 6.5))
            material = get_texture(material="hat1" + str(i), **kwargs)
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
            [scale,circle_d, index_d, vsw_d, set_pos_d, circle_r, index_r, vsw_r, set_pos_r, rr_unreflected, rr_reflected,
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
