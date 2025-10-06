import hashlib
import math
import os
from copy import deepcopy

import numpy as np

from appearance.textures import phase2hue_material, gradient_from_attribute, z_gradient, double_gradient, get_texture
from extended_math_nodes.generic_nodes import AssociatedLegendrePolynomial, \
    SphericalHarmonicsRekursive, SphericalHarmonics200
from geometry_nodes.nodes import layout, Points, InputValue, CurveCircle, InstanceOnPoints, JoinGeometry, \
    create_geometry_line, RealizeInstances, Position, make_function, ObjectInfo, SetPosition, Index, SetMaterial, \
    RandomValue, RepeatZone, StoredNamedAttribute, NamedAttribute, VectorMath, CurveToMesh, PointsToCurve, Grid, \
    TransformGeometry, InputVector, DeleteGeometry, IcoSphere, MeshLine, InstanceOnEdges, CubeMesh, \
    EdgeVertices, BooleanMath, SetShadeSmooth, RayCast, WireFrame, ConvexHull, InsideConvexHull, ExtrudeMesh, \
    ScaleElements, UVSphere, SceneTime, Simulation, MathNode, PointsToVertices, CombineXYZ, Switch, MeshToPoints, \
    SubdivideMesh, CollectionInfo, CylinderMesh, ConeMesh, InputRotation, InvertRotation, RotateRotation, \
    Frame, SeparateXYZ, DualMesh, WireFrameRectangle, SplitEdges, VectorRotate, EvaluateOnDomain, InputNormal, \
    create_from_xml, ForEachZone, AxesToRotation, UnfoldMeshNode, Quadrilateral, InputInteger, ResampleCurve, TrimCurve, \
    SeparateGeometry, CurveWireFrame, AlignRotationToVector, ScaleInstances, ValueToString, StringToCurves, IndexSwitch, \
    CycleNode, FillCurve, AttributeStatistic, CoxeterReflectionNode, QuaternionToRotation, RotateVector, \
    SubdivisionSurface, FilletCurve, FaceArea, SortElements, InputBoolean, BeveledCubeNode, CompareNode, \
    GeometryToInstance, InputMaterial, RotateInstances, SimpleRubiksCubeNode, CornersOfFace
from interface import ibpy
from interface.ibpy import make_new_socket, Vector, get_node_tree, get_material
from mathematics.parsing.parser import ExpressionConverter
from mathematics.spherical_harmonics import SphericalHarmonics
from objects.derived_objects.p_arrow import PArrow
from utils.constants import FRAME_RATE, TEMPLATE_TEXT_FILE, SVG_DIR, TEX_DIR, TEX_TEXT_TO_REPLACE, TEMPLATE_TEX_FILE, \
    TEX_LOCAL_SCALE_UP, DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs

pi = np.pi
tau = 2 * pi
r2 = np.sqrt(2)
r3 = np.sqrt(3)


face_color_dict = {
    1: "white",
    2: "white",
    3: "white",
    4: "white",
    5: "white",
    6: "white",
    7: "white",
    8: "white",
    9: "orange",
    10: "orange",
    11: "orange",
    12: "orange",
    13: "orange",
    14: "orange",
    15: "orange",
    16: "orange",
    17: "green",
    18: "green",
    19: "green",
    20: "green",
    21: "green",
    22: "green",
    23: "green",
    24: "green",
    25: "red",
    26: "red",
    27: "red",
    28: "red",
    29: "red",
    30: "red",
    31: "red",
    32: "red",
    33: "blue",
    34: "blue",
    35: "blue",
    36: "blue",
    37: "blue",
    38: "blue",
    39: "blue",
    40: "blue",
    41: "yellow",
    42: "yellow",
    43: "yellow",
    44: "yellow",
    45: "yellow",
    46: "yellow",
    47: "yellow",
    48: "yellow"
}


############################
## GeometryNodesModifier ###
## provide more versality ##
############################


class GeometryNodesModifier:
    """
    base class that organizes the boilerplate code for the creation of a geometry nodes modifier
    """

    def __init__(self, name='DefaultGeometryNodeGroup', automatic_layout=True,mode="Sugiyama", group_output=True, group_input=False, **kwargs):
        tree = get_node_tree(name=name, type='GeometryNodeTree')

        # if  materials are created inside the geometry node,
        # they are stored inside the following array
        # and will be added to the material slots of the blender object
        self.materials = []
        # create output nodes
        if group_output:
            self.group_outputs = tree.nodes.new('NodeGroupOutput')
            make_new_socket(tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')
        if group_input:
            self.group_inputs = tree.nodes.new('NodeGroupInput')
            make_new_socket(tree, name='Geometry', io='INPUT', type='NodeSocketGeometry')
        self.group_output = tree.nodes.get("Group Output")

        self.tree = tree
        self.nodes = self.tree.nodes  # needed for ibpy.get_geometry_node_from_modifier

        self.create_node(tree, **kwargs)
        # automatically layout nodes
        if automatic_layout:
            layout(tree,mode = mode)


    def create_node(self, tree, **kwargs):
        """
        this method is to be overridden to customize your geometry node modifier
        :param tree:
        :return:
        """

        pass

    def get_node_tree(self):
        return self.tree

##################
## Applications ##
##################

class SpherePreImage(GeometryNodesModifier):
    """
    theta-phi-domain that gets mapped into a sphere

    """

    def __init__(self):
        super().__init__(name="SpherePreImage", automatic_layout=False)

    def create_node(self, tree):
        out = tree.nodes.get("Group Output")
        links = tree.links

        left = -15
        reset_left = left
        # vertical grid lines
        line = MeshLine(tree, location=(left, 2), count=20, start_location=Vector(), end_location=Vector([tau, 0, 0]))
        left += 1
        mesh2points = MeshToPoints(tree, location=(left, 2))
        left += 1
        points2verts = PointsToVertices(tree, location=(left, 2))
        index = Index(tree, location=(left, 1))
        left += 1
        attr = StoredNamedAttribute(tree, location=(left, 2), name="Index", data_type="INT", value=index.std_out)
        pi_offset = InputVector(tree, location=(left, 1), value=[0, 0, pi])
        left += 1
        extrude_mesh = ExtrudeMesh(tree, location=(left, 2), offset=pi_offset.std_out)
        left += 1
        sub_div = SubdivideMesh(tree, location=(left, 1), level=5)
        left += 1
        lmda = InputValue(tree, location=(left - 1, 4), value=0, name="lambda")
        position = Position(tree, location=(left - 1, 3.5))
        shift = InputVector(tree, location=(left - 1, 3), value=[-4.5, 0, -1.5], name="shift")
        radius = InputValue(tree, location=(left - 1, 2.5), value=2.5, name="r")
        selected_index = InputValue(tree, location=(left - 1, 2), value=0, name="idx0")
        named_attr = NamedAttribute(tree, location=(left - 1, 1.5), data_type="INT", name="Index")

        # mega trafo
        # (r cos\phi sin(\theta-pi) + shift_x)*(idx<idx_0)+[(r cos\phi sin(\theta-pi) + shift_x)*lambda+pos_x*(1-lambda)]*(idx=idx_0)+pos_x*(idx>idx_0)
        # similar for the other components
        x = "r,pos_x,cos,*,pos_z,pi,-,sin,*,shift_x,+"
        y = "r,pos_x,sin,*,pos_z,pi,-,sin,*,shift_y,+"
        z = "r,pos_z,pi,-,cos,*,shift_z,+"
        less = "index,idx0,>"
        equal = "index,idx0,="
        more = "index,idx0,<"

        trafo = make_function(tree, functions={
            "position": [
                x + "," + more + ",*," + x + ",lambda,*,pos_x,1,lambda,-,*,+," + equal + ",*,+,pos_x," + less + ",*,+",
                y + "," + more + ",*," + y + ",lambda,*,pos_y,1,lambda,-,*,+," + equal + ",*,+,pos_y," + less + ",*,+",
                z + "," + more + ",*," + z + ",lambda,*,pos_z,1,lambda,-,*,+," + equal + ",*,+,pos_z," + less + ",*,+",
            ]
        }, inputs=["lambda", "pos", "shift", "r", "idx0", "index"], outputs=["position"],
                              scalars=["lambda", "r", "idx0", "index"], vectors=["pos", "position", "shift"],
                              name="Map2Sphere", location=(left, 1.5), hide=True)
        links.new(lmda.std_out, trafo.inputs["lambda"])
        links.new(position.std_out, trafo.inputs["pos"])
        links.new(radius.std_out, trafo.inputs["r"])
        links.new(shift.std_out, trafo.inputs["shift"])
        links.new(named_attr.std_out, trafo.inputs["index"])
        links.new(selected_index.std_out, trafo.inputs["idx0"])
        left += 1
        set_pos = SetPosition(tree, location=(left, 1), position=trafo.outputs["position"])
        left += 1
        wireframe = WireFrame(tree, location=(left, 1))
        left += 1
        material = gradient_from_attribute(name="Index",
                                           function="fac,20,/",
                                           attr_name="Index",
                                           gradient={0: [1, 0, 0, 1], 1: [0.8, 0, 1, 1]})
        mat = SetMaterial(tree, location=(left, 1), material=material)
        self.materials.append(material)
        left += 1
        join = JoinGeometry(tree, location=(left, 0))
        create_geometry_line(tree,
                             [line, mesh2points, points2verts, attr, extrude_mesh, sub_div, set_pos, wireframe, mat,
                              join])

        # horizontal grid lines
        left = reset_left

        line = MeshLine(tree, location=(left, -2), count=10, start_location=Vector([0, 0, pi]),
                        end_location=Vector([0, 0, 0]))
        left += 1
        mesh2points = MeshToPoints(tree, location=(left, -2))
        left += 1
        points2verts = PointsToVertices(tree, location=(left, -2))
        index = Index(tree, location=(left, 1))
        left += 1
        attr = StoredNamedAttribute(tree, location=(left, -2), name="Index2", data_type="INT", value=index.std_out)
        tau_offset = InputVector(tree, location=(left, -1), value=[tau, 0, 0])
        left += 1
        extrude_mesh = ExtrudeMesh(tree, location=(left, -2), offset=tau_offset.std_out)
        left += 1
        sub_div = SubdivideMesh(tree, location=(left, -1), level=5)
        left += 1
        lmda = InputValue(tree, location=(left - 1, -1.5), value=0, name="lambda2")
        position = Position(tree, location=(left - 1, -2))
        shift = InputVector(tree, location=(left - 1, -2.5), value=[-4.5, 0, -1.5], name="shift")
        radius = InputValue(tree, location=(left - 1, -3), value=2.5, name="r")
        selected_index = InputValue(tree, location=(left - 1, -3.5), value=0, name="idx02")
        named_attr = NamedAttribute(tree, location=(left - 1, -4), data_type="INT", name="Index2")

        # mega trafo
        # (r cos\phi sin(\theta-pi) + shift_x)*(idx<idx_0)+[(r cos\phi sin(\theta-pi) + shift_x)*lambda+pos_x*(1-lambda)]*(idx=idx_0)+pos_x*(idx>idx_0)
        # similar for the other components
        x = "r,pos_x,cos,*,pos_z,pi,-,sin,*,shift_x,+"
        y = "r,pos_x,sin,*,pos_z,pi,-,sin,*,shift_y,+"
        z = "r,pos_z,pi,-,cos,*,shift_z,+"
        less = "index,idx0,>"
        equal = "index,idx0,="
        more = "index,idx0,<"

        trafo = make_function(tree, functions={
            "position": [
                x + "," + more + ",*," + x + ",lambda,*,pos_x,1,lambda,-,*,+," + equal + ",*,+,pos_x," + less + ",*,+",
                y + "," + more + ",*," + y + ",lambda,*,pos_y,1,lambda,-,*,+," + equal + ",*,+,pos_y," + less + ",*,+",
                z + "," + more + ",*," + z + ",lambda,*,pos_z,1,lambda,-,*,+," + equal + ",*,+,pos_z," + less + ",*,+",
            ]
        }, inputs=["lambda", "pos", "shift", "r", "idx0", "index"], outputs=["position"],
                              scalars=["lambda", "r", "idx0", "index"], vectors=["pos", "position", "shift"],
                              name="Map2Sphere", location=(left, -1.5), hide=True)
        links.new(lmda.std_out, trafo.inputs["lambda"])
        links.new(position.std_out, trafo.inputs["pos"])
        links.new(radius.std_out, trafo.inputs["r"])
        links.new(shift.std_out, trafo.inputs["shift"])
        links.new(named_attr.std_out, trafo.inputs["index"])
        links.new(selected_index.std_out, trafo.inputs["idx0"])
        left += 1
        set_pos = SetPosition(tree, location=(left, -1), position=trafo.outputs["position"])
        left += 1
        wireframe = WireFrame(tree, location=(left, -1))
        left += 1
        material = gradient_from_attribute(name="Index2",
                                           function="fac,10,/",
                                           attr_name="Index2",
                                           gradient={0: [0, 1, 0.95, 1], 1: [1, 1, 0, 1]})
        mat = SetMaterial(tree, location=(left, 1), material=material)
        self.materials.append(material)
        left += 1
        rot = InputVector(tree, location=(left, -1), name='rotation')
        trafo1 = TransformGeometry(tree, location=(left, 0), translation=Vector([4.5, 0, 1.5]))
        left += 1
        trafo2 = TransformGeometry(tree, location=(left, 0), rotation=rot.std_out)
        left += 1
        trafo3 = TransformGeometry(tree, location=(left, 0), translation=Vector([-4.5, 0, -1.5]))
        left += 1

        create_geometry_line(tree,
                             [line, mesh2points, points2verts, attr, extrude_mesh, sub_div, set_pos, wireframe, mat,
                              join, trafo1, trafo2, trafo3], out=out.inputs["Geometry"])

class MathematicalSurface(GeometryNodesModifier):
    """
    geometry node setup to display a mathematical surface in explicit form
    """

    def __init__(self, function="x,y,+,2,**", name="(x+y)**2", resolution=100, automatic_layout=False, **kwargs):
        self.function = function
        self.resolution = resolution
        self.name = name
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=automatic_layout)

    def create_node(self, tree):
        out = tree.nodes.get("Group Output")
        links = tree.links

        first_left = -10
        left = first_left
        # create grid, the (x,y) domain
        grid = Grid(tree, location=(left, 1), size_x=10, size_y=10, vertices_x=self.resolution,
                    vertices_y=self.resolution)

        # read in the original location of each vertex (the x,y coordinates are kept and the z coordinate is adjusted
        # depending on the value of the function f(x,y)
        pos = Position(tree, location=(left, 0))

        left += 1

        # create the function, here is your function $\left(2(x-3)^2+(y-2)^2-1\right) \cdot(x-1)(y-1)+3$
        surface = make_function(tree, functions={
            "position": [
                "pos_x",  # x stays unchanged
                "pos_y",  # y stays unchanged
                self.function.replace('x', 'pos_x').replace('y', 'pos_y')  # f(x,y)
            ]
        }, name="Surface_" + self.name, location=(left, 0), inputs=["pos"], outputs=["position"],
                                vectors=["pos", "position"])

        links.new(pos.std_out, surface.inputs["pos"])
        left += 1

        # window: -5<=f(x,y)<=5
        selector = make_function(tree, functions={
            "deselect": "pos_z,-5,>,pos_z,5,<,and,not"
        }, name="Window", location=(left, 1), inputs=["pos"], outputs=["deselect"], vectors=["pos"],
                                 scalars=["deselect"])
        links.new(surface.outputs["position"], selector.inputs["pos"])

        left += 1
        del_geo = DeleteGeometry(tree, location=(left, 0), selection=selector.outputs["deselect"])

        left += 1
        # update the z-value for each mesh point
        set_pos = SetPosition(tree, location=(left, 0), position=surface.outputs["position"])

        left += 1
        smooth = SetShadeSmooth(tree, location=(left, 0))

        left += 1

        # custom material that is a gradient in z-direction
        gradient_material = z_gradient(**self.kwargs)
        self.materials.append(gradient_material)
        mat = SetMaterial(tree, location=(left, 0), material=gradient_material)

        # connect all geometry nodes
        create_geometry_line(tree, [grid, del_geo, set_pos, smooth, mat], out=out.inputs["Geometry"])

class PendulumModifierSmall(GeometryNodesModifier):
    def __init__(self, min_frame=0,max_frame=100,name="SmallPendulum", automatic_layout=False):
        self.min_frame=min_frame
        self.max_frame=max_frame
        super().__init__(name=name, automatic_layout=automatic_layout)

    def create_node(self, tree):
        out = tree.nodes.get("Group Output")
        links = tree.links

        ##### first pendulum ####
        first_left = -18
        left = first_left
        theta = InputValue(tree, location=(left, 0.25), name="theta", value=np.pi * 0.95)
        b = InputValue(tree, location=(left, 0.5), name="friction",
                       value=0.0225)  # damping to compensate accumulating errors
        omega = InputValue(tree, location=(left, 0.75), name="omega", value=0)
        origin = Points(tree, location=(left, 4), name="Origin")
        point = Points(tree, location=(left, 0), name="PendulumMass")

        frame = SceneTime(tree, location=(left, -0.75), std_out="Frame")
        min_frame = InputValue(tree, location=(left, - 1), name="minFrame", value=self.min_frame)
        max_frame = InputValue(tree, location=(left, - 1.25), name="maxFrame", value=self.max_frame)
        t0 = InputValue(tree, location=(left, -0.5), value=-120 / FRAME_RATE)

        left += 1

        simulation = Simulation(tree, location=(left, 0))
        simulation.add_socket(socket_type='FLOAT', name="t")  # elongation
        simulation.add_socket(socket_type='FLOAT', name="theta")  # elongation
        simulation.add_socket(socket_type='FLOAT', name="omega")  # angular velocity
        links.new(omega.std_out, simulation.simulation_input.inputs["omega"])
        links.new(t0.std_out, simulation.simulation_input.inputs["t"])

        length = InputValue(tree, location=(left - 1, -4), value=2.5, name="Length")

        left += 2
        time = MathNode(tree, location=(left, +1), inputs0=simulation.simulation_input.outputs["t"],
                        inputs1=simulation.simulation_input.outputs["Delta Time"])

        start_sim = make_function(tree, functions={
            "theta": "frame,start,=,theta0,*,th,+"  # initialize theta at a particular frame
        }, name="thetaInitializer", inputs=["frame", "start", "theta0", "th"], outputs=["theta"],
                                  scalars=["frame", "start", "theta0", "th", "theta"],
                                  location=(left - 1, -0.5), hide=True)
        links.new(theta.std_out, start_sim.inputs["theta0"])
        links.new(frame.std_out, start_sim.inputs["frame"])
        links.new(min_frame.std_out, start_sim.inputs["start"])
        links.new(simulation.simulation_input.outputs["theta"], start_sim.inputs["th"])

        update_omega = make_function(tree, functions={
            "omega": "o,9.81,l,/,theta,sin,*,b,o,*,+,dt,*,-"
        }, name="updateOmega", location=(left, -0.5), hide=True,
                                     outputs=["omega"], inputs=["dt", "theta", "o", "l", "b"],
                                     scalars=["omega", "o", "l", "theta", "dt", "b"])

        links.new(length.std_out, update_omega.inputs["l"])
        links.new(b.std_out, update_omega.inputs["b"])
        links.new(start_sim.outputs["theta"], update_omega.inputs["theta"])
        links.new(simulation.simulation_input.outputs["Delta Time"], update_omega.inputs["dt"])
        links.new(update_omega.outputs["omega"], simulation.simulation_output.inputs["omega"])
        links.new(time.std_out, simulation.simulation_output.inputs["t"])
        links.new(simulation.simulation_input.outputs["omega"], update_omega.inputs["o"])

        update_theta = make_function(tree, functions={
            "theta": "th,omega,dt,*,+"
        }, name="updateTheta", location=(left, -1.5), hide=True,
                                     outputs=["theta"], inputs=["dt", "th", "omega"],
                                     scalars=["theta", "th", "dt", "omega"])

        links.new(start_sim.outputs["theta"], update_theta.inputs["th"])
        links.new(simulation.simulation_input.outputs["Delta Time"], update_theta.inputs["dt"])
        links.new(update_theta.outputs["theta"], simulation.simulation_output.inputs["theta"])
        links.new(simulation.simulation_input.outputs["omega"], update_theta.inputs["omega"])
        left += 4

        # convert angle into position
        converter = make_function(tree, functions={
            "position": ["theta,sin,l,*", "0", "theta,cos,l,-1,*,*"]
        }, name="Angle2Position", inputs=["theta", "l"], outputs=["position"], scalars=["l", "theta"],
                                  vectors=["position"],
                                  location=(left, -1))
        links.new(simulation.simulation_output.outputs["theta"], converter.inputs["theta"])
        links.new(length.std_out, converter.inputs["l"])
        left += 1

        set_pos = SetPosition(tree, position=converter.outputs["position"], location=(left, -0.5))
        left += 1
        join = JoinGeometry(tree, location=(left, 0))
        create_geometry_line(tree, [origin, join])

        left += 1
        point2mesh = PointsToVertices(tree, location=(left, 0))
        left += 1
        convex_hull = ConvexHull(tree, location=(left, 0))
        left += 1
        wireframe = WireFrame(tree, location=(left, 0))
        left += 1
        material = get_material('joker')
        self.materials.append(material)
        mat = SetMaterial(tree, location=(left, 0), material=material)
        left += 1
        join2 = JoinGeometry(tree, location=(left, -0.5))
        left += 1
        trafo = TransformGeometry(tree, location=(left, 0), translation=[5, 0, 1])
        left += 1
        join_full = JoinGeometry(tree, location=(left, 0))
        create_geometry_line(tree, [point, simulation, set_pos, join, point2mesh,
                                    convex_hull, wireframe, mat, join2, trafo, join_full], out=out.inputs["Geometry"])

        # create branch for the mass
        left -= 6
        pos = Position(tree, location=(left, -1))
        left += 1
        lengthBool = VectorMath(tree, location=(left, -1), operation='LENGTH',
                                inputs0=pos.std_out)  # one vertex is at the origin
        # only the vertex away from the origin has a non-zero length, this value is used to select the correct vertex for the instance on points
        ico = IcoSphere(tree, location=(left, -2), radius=0.3, subdivisions=1)
        left += 1
        iop = InstanceOnPoints(tree, location=(left, -1), selection=lengthBool.std_out,
                               instance=ico.geometry_out)
        left += 1
        material = get_material('plastic_example')
        self.materials.append(material)
        mat2 = SetMaterial(tree, location=(left, -2), material=material)

        create_geometry_line(tree, [point2mesh, iop, mat2, join2])

        # create angle visualization

        left = first_left + 6
        arc_length_factor = InputValue(tree, location=(left, -5.5), value=0.5, name="ArcFactor")
        res = InputValue(tree, location=(left, -6), value=100, name="Resolution")
        idx = Index(tree, location=(left, -5))
        left += 1
        arc = Points(tree, location=(left, -6), count=res.std_out)
        left += 1

        compute_positions = make_function(tree, functions={
            "position": ["l,fac,*,theta,res,/,idx,*,sin,*", "0", "l,fac,*,-1,*,theta,res,/,idx,*,cos,*"]
        }, name="arcPosition", inputs=["l", "fac", "idx", "res", "theta"], outputs=["position"],
                                          scalars=["l", "fac", "idx", "res", "theta"],
                                          vectors=["position"], location=(left, -3))

        links.new(res.std_out, compute_positions.inputs["res"])
        links.new(idx.std_out, compute_positions.inputs["idx"])
        links.new(length.std_out, compute_positions.inputs["l"])
        links.new(arc_length_factor.std_out, compute_positions.inputs["fac"])
        links.new(simulation.simulation_output.outputs["theta"], compute_positions.inputs["theta"])
        left += 1
        set_arc_pos = SetPosition(tree, location=(left, -6), position=compute_positions.outputs["position"])
        left += 1
        join3 = JoinGeometry(tree, location=(left, -6))
        left += 1
        arc_fill = ConvexHull(tree, location=(left, -6))
        left += 1
        theta_attr = StoredNamedAttribute(tree, location=(left, -6), name="thetaStorage",
                                          value=simulation.simulation_output.outputs["theta"])
        left += 1
        material = gradient_from_attribute(attr_name="thetaStorage",
                                           roughness=0.1, metallic=0.5, emission=0.75)
        self.materials.append(material)
        arc_mat = SetMaterial(tree, location=(left, -6), material=material)
        create_geometry_line(tree, [arc, set_arc_pos, join3, arc_fill, theta_attr, arc_mat, join2])
        create_geometry_line(tree, [origin, join3])

        # trace motion
        left = first_left + 4
        y = -10
        graph = MeshLine(tree, location=(left, y + 1), count=1)

        skip_frame = InputValue(tree, location=(left, y - 0.5), name="skipFrame", value=7)
        left += 1
        freeze_frames = make_function(tree, location=(left, y / 2),
                                      functions={
                                          "switch": "frame,skip,%,0,=,frame,end,<,and,frame,start,>,and"
                                      }, inputs=["frame", "skip", "start", "end"], outputs=["switch"],
                                      scalars=["frame", "skip", "start", "end", "switch"], name="freezer")
        links.new(frame.std_out, freeze_frames.inputs["frame"])
        links.new(skip_frame.std_out, freeze_frames.inputs["skip"])
        links.new(max_frame.std_out, freeze_frames.inputs["end"])
        links.new(min_frame.std_out, freeze_frames.inputs["start"])

        comb_xyz = CombineXYZ(tree, location=(left, y + 1),
                              x=simulation.simulation_output.outputs["t"],
                              z=simulation.simulation_output.outputs["theta"])

        left += 1

        set_graph = SetPosition(tree, location=(left, y + 1), position=comb_xyz.std_out)
        switch = Switch(tree, location=(left, y), switch=freeze_frames.outputs["switch"])
        left += 1
        attr2 = StoredNamedAttribute(tree, location=(left, y), name='thetaStorage2',
                                     value=simulation.simulation_output.outputs["theta"])
        left += 1
        simulation2 = Simulation(tree, location=(left, y + 2))
        left += 1
        join_freezes = JoinGeometry(tree, location=(left, y + 1))
        links.new(simulation2.simulation_input.outputs["Geometry"], join_freezes.geometry_in)
        links.new(join_freezes.geometry_out, simulation2.simulation_output.inputs["Geometry"])
        links.new(switch.std_out, attr2.geometry_in)
        links.new(attr2.geometry_out, join_freezes.geometry_in)
        left += 3
        graph_sphere = IcoSphere(tree, location=(left, y + 1), radius=0.1)
        left += 1
        graph_iop = InstanceOnPoints(tree, location=(left, y), instance=graph_sphere.geometry_out)
        left += 1
        graph_trafo = TransformGeometry(tree, location=(left, y),
                                        translation=Vector([-24.7, 0,- 0.5]))
        left += 1

        graph_mat = SetMaterial(tree, location=(left, y), material_list=self.materials,
                                material=gradient_from_attribute, attr_name="thetaStorage2", attr_type="INSTANCER",
                                emission=0.75)

        create_geometry_line(tree, [graph, set_graph], out=switch.true)
        create_geometry_line(tree, [simulation2, graph_iop, graph_trafo, graph_mat, join_full])

class PendulumModifierLarge(GeometryNodesModifier):
    def __init__(self, min_frame=0,max_frame=10,name="Pendulum", automatic_layout=False):
        self.min_frame=min_frame
        self.max_frame=max_frame
        super().__init__(name=name, automatic_layout=automatic_layout)

    def create_node(self, tree):
        out = tree.nodes.get("Group Output")
        links = tree.links

        # ##### first pendulum ####
        first_left = -38
        left = first_left
        # theta = InputValue(tree, location=(left, 0.25), name="theta", value=np.pi * 0.95)
        # b = InputValue(tree, location=(left, 0.5), name="friction",
        #                value=0.0225)  # damping to compensate accumulating errors
        # omega = InputValue(tree, location=(left, 0.75), name="omega", value=0)
        # origin = Points(tree, location=(left, 4), name="Origin")
        # point = Points(tree, location=(left, 0), name="PendulumMass")
        #
        frame = SceneTime(tree, location=(left, -0.75), std_out="Frame")
        min_frame = InputValue(tree, location=(left, - 1), name="maxFrame", value=self.min_frame)
        max_frame = InputValue(tree, location=(left, - 1.25), name="maxFrame", value=self.max_frame)
        t0 = InputValue(tree, location=(left, -0.5), value=-120 / FRAME_RATE)

        ##### second pendulum ####

        first_left = -38
        left = first_left
        theta = InputValue(tree, location=(left, 0.25), name="theta", value=0.2)
        b = InputValue(tree, location=(left, 0.5), name="friction",
                       value=0.0225)  # damping to compensate accumulating errors
        omega = InputValue(tree, location=(left, 0.75), name="omega", value=0)
        origin = Points(tree, location=(left, 4), name="Origin")
        point = Points(tree, location=(left, 0), name="PendulumMass")

        left += 1

        simulation = Simulation(tree, location=(left, 0))
        simulation.add_socket(socket_type='FLOAT', name="t")  # elongation
        simulation.add_socket(socket_type='FLOAT', name="theta")  # elongation
        simulation.add_socket(socket_type='FLOAT', name="omega")  # angular velocity
        links.new(omega.std_out, simulation.simulation_input.inputs["omega"])
        links.new(t0.std_out, simulation.simulation_input.inputs["t"])

        length = InputValue(tree, location=(left - 1, -4), value=15.75 / 2, name="Length")

        left += 2
        time = MathNode(tree, location=(left, +1), inputs0=simulation.simulation_input.outputs["t"],
                        inputs1=simulation.simulation_input.outputs["Delta Time"])

        start_sim = make_function(tree, functions={
            "theta": "frame,start,=,theta0,*,th,+"  # initialize theta at a particular frame
        }, name="thetaInitializer", inputs=["frame", "start", "theta0", "th"], outputs=["theta"],
                                  scalars=["frame", "start", "theta0", "th", "theta"],
                                  location=(left - 1, -0.5), hide=True)
        links.new(theta.std_out, start_sim.inputs["theta0"])
        links.new(frame.std_out, start_sim.inputs["frame"])
        links.new(min_frame.std_out, start_sim.inputs["start"])
        links.new(simulation.simulation_input.outputs["theta"], start_sim.inputs["th"])

        update_omega = make_function(tree, functions={
            "omega": "o,4.905,l,/,theta,sin,*,b,o,*,+,dt,*,-"
            # reduce gravity for matching period without over-length pendulum
        }, name="updateOmega", location=(left, -0.5), hide=True,
                                     outputs=["omega"], inputs=["dt", "theta", "o", "l", "b"],
                                     scalars=["omega", "o", "l", "theta", "dt", "b"])

        links.new(length.std_out, update_omega.inputs["l"])
        links.new(b.std_out, update_omega.inputs["b"])
        links.new(start_sim.outputs["theta"], update_omega.inputs["theta"])
        links.new(simulation.simulation_input.outputs["Delta Time"], update_omega.inputs["dt"])
        links.new(update_omega.outputs["omega"], simulation.simulation_output.inputs["omega"])
        links.new(time.std_out, simulation.simulation_output.inputs["t"])
        links.new(simulation.simulation_input.outputs["omega"], update_omega.inputs["o"])

        update_theta = make_function(tree, functions={
            "theta": "th,omega,dt,*,+"
        }, name="updateTheta", location=(left, -1.5), hide=True,
                                     outputs=["theta"], inputs=["dt", "th", "omega"],
                                     scalars=["theta", "th", "dt", "omega"])

        links.new(start_sim.outputs["theta"], update_theta.inputs["th"])
        links.new(simulation.simulation_input.outputs["Delta Time"], update_theta.inputs["dt"])
        links.new(update_theta.outputs["theta"], simulation.simulation_output.inputs["theta"])
        links.new(simulation.simulation_input.outputs["omega"], update_theta.inputs["omega"])
        left += 4

        # convert angle into position
        converter = make_function(tree, functions={
            "position": ["theta,sin,l,*", "0", "theta,cos,l,-1,*,*"]
        }, name="Angle2Position", inputs=["theta", "l"], outputs=["position"], scalars=["l", "theta"],
                                  vectors=["position"],
                                  location=(left, -1))
        links.new(simulation.simulation_output.outputs["theta"], converter.inputs["theta"])
        links.new(length.std_out, converter.inputs["l"])
        left += 1

        set_pos = SetPosition(tree, position=converter.outputs["position"], location=(left, -0.5))
        left += 1
        join = JoinGeometry(tree, location=(left, 0))
        create_geometry_line(tree, [origin, join])

        left += 1
        point2mesh = PointsToVertices(tree, location=(left, 0))
        left += 1
        convex_hull = ConvexHull(tree, location=(left, 0))
        left += 1
        wireframe = WireFrame(tree, location=(left, 0))
        left += 1
        material = get_material('joker')
        self.materials.append(material)
        mat = SetMaterial(tree, location=(left, 0), material=material)
        left += 1
        join2 = JoinGeometry(tree, location=(left, -0.5))
        left += 1
        trafo = TransformGeometry(tree, location=(left, 0), translation=[-6, 0, 2.5],scale = [1,1,0.8])
        left += 1

        join_full = JoinGeometry(tree, location=(left, 0))

        create_geometry_line(tree, [point, simulation, set_pos, join, point2mesh,
                                    convex_hull, wireframe, mat, join2, trafo, join_full], out=out.inputs["Geometry"])

        # create branch for the mass
        left -= 6
        pos = Position(tree, location=(left, -1))
        left += 1
        lengthBool = VectorMath(tree, location=(left, -1), operation='LENGTH',
                                inputs0=pos.std_out)  # one vertex is at the origin
        # only the vertex away from the origin has a non-zero length, this value is used to select the correct vertex for the instance on points
        cube = CubeMesh(tree, location=(left, -2), size=0.4)
        left += 1
        iop = InstanceOnPoints(tree, location=(left, -1), selection=lengthBool.std_out,
                               instance=cube.geometry_out)
        left += 1
        material = get_material('plastic_example')
        self.materials.append(material)
        mat2 = SetMaterial(tree, location=(left, -2), material=material)

        create_geometry_line(tree, [point2mesh, iop, mat2, join2])

        # create angle visualization

        left = first_left + 6
        arc_length_factor = InputValue(tree, location=(left, -5.5), value=0.5, name="ArcFactor")
        res = InputValue(tree, location=(left, -6), value=100, name="Resolution")
        idx = Index(tree, location=(left, -5))
        left += 1
        arc = Points(tree, location=(left, -6), count=res.std_out)
        left += 1

        compute_positions = make_function(tree, functions={
            "position": ["l,fac,*,theta,res,/,idx,*,sin,*", "0", "l,fac,*,-1,*,theta,res,/,idx,*,cos,*"]
        }, name="arcPosition", inputs=["l", "fac", "idx", "res", "theta"], outputs=["position"],
                                          scalars=["l", "fac", "idx", "res", "theta"],
                                          vectors=["position"], location=(left, -3))

        links.new(res.std_out, compute_positions.inputs["res"])
        links.new(idx.std_out, compute_positions.inputs["idx"])
        links.new(length.std_out, compute_positions.inputs["l"])
        links.new(arc_length_factor.std_out, compute_positions.inputs["fac"])
        links.new(simulation.simulation_output.outputs["theta"], compute_positions.inputs["theta"])
        left += 1
        set_arc_pos = SetPosition(tree, location=(left, -6), position=compute_positions.outputs["position"])
        left += 1
        join3 = JoinGeometry(tree, location=(left, -6))
        left += 1
        arc_fill = ConvexHull(tree, location=(left, -6))
        left += 1
        theta_attr = StoredNamedAttribute(tree, location=(left, -6), name="thetaStorage",
                                          value=simulation.simulation_output.outputs["theta"])
        left += 1
        material = gradient_from_attribute(attr_name="thetaStorage",
                                           roughness=0.1, metallic=0.5, emission=0.5)
        self.materials.append(material)
        arc_mat = SetMaterial(tree, location=(left, -6), material=material)
        create_geometry_line(tree, [arc, set_arc_pos, join3, arc_fill, theta_attr, arc_mat, join2])
        create_geometry_line(tree, [origin, join3])

        # trace motion
        left = first_left + 4
        y = -10
        graph = MeshLine(tree, location=(left, y + 1), count=1)

        skip_frame = InputValue(tree, location=(left, y - 0.5), name="skipFrame", value=10)
        left += 1
        freeze_frames = make_function(tree, location=(left, y / 2),
                                      functions={
                                          "switch": "frame,skip,%,0,=,frame,end,<,and,frame,start,>,and"
                                      }, inputs=["frame", "skip", "start", "end"], outputs=["switch"],
                                      scalars=["frame", "skip", "start", "end", "switch"], name="freezer")
        links.new(frame.std_out, freeze_frames.inputs["frame"])
        links.new(skip_frame.std_out, freeze_frames.inputs["skip"])
        links.new(max_frame.std_out, freeze_frames.inputs["end"])
        links.new(min_frame.std_out, freeze_frames.inputs["start"])

        comb_xyz = CombineXYZ(tree, location=(left, y + 1),
                              x=simulation.simulation_output.outputs["t"],
                              z=simulation.simulation_output.outputs["theta"])

        left += 1

        set_graph = SetPosition(tree, location=(left, y + 1), position=comb_xyz.std_out)
        switch = Switch(tree, location=(left, y), switch=freeze_frames.outputs["switch"])
        left += 1
        attr2 = StoredNamedAttribute(tree, location=(left, y), name='thetaStorage2',
                                     value=simulation.simulation_output.outputs["theta"])
        left += 1
        simulation2 = Simulation(tree, location=(left, y + 2))
        left += 1
        join_freezes = JoinGeometry(tree, location=(left, y + 1))
        links.new(simulation2.simulation_input.outputs["Geometry"], join_freezes.geometry_in)
        links.new(join_freezes.geometry_out, simulation2.simulation_output.inputs["Geometry"])
        links.new(switch.std_out, attr2.geometry_in)
        links.new(attr2.geometry_out, join_freezes.geometry_in)
        left += 3
        graph_cube = CubeMesh(tree, location=(left, y + 1), size=0.1)
        left += 1
        graph_iop = InstanceOnPoints(tree, location=(left, y), instance=graph_cube.geometry_out)
        left += 1
        graph_trafo = TransformGeometry(tree, location=(left, y),
                                        translation=Vector([-9, 0, -0.5]))
        left += 1

        graph_mat = SetMaterial(tree, location=(left, y), material_list=self.materials,
                                material=gradient_from_attribute, attr_name="thetaStorage2", attr_type="INSTANCER",
                                emission=0.5)

        create_geometry_line(tree, [graph, set_graph], out=switch.true)
        final_tranformation = TransformGeometry(tree)

        create_geometry_line(tree, [simulation2, graph_iop, graph_trafo, graph_mat, join_full])

class VectorLogo(GeometryNodesModifier):
    def __init__(self, name='VectorLogo', n=10, colors=['important', 'example', 'drawing']):
        self.n = n
        self.colors = colors
        super().__init__(name)

    def create_node(self, tree):
        # arrow object with origin at the tip
        arrow = PArrow(name="ArrowObject")
        ibpy.set_pivot(arrow, Vector([0, 0, 1]))

        # parameters for animations
        res = InputValue(tree, name='res', value=10)
        n = InputValue(tree, name='N', value=self.n)
        idx = Index(tree)
        growth = InputValue(tree, name='growth', value=0)

        object_info = ObjectInfo(tree, transform_space='ORIGINAL', object=arrow.ref_obj)
        pos = Position(tree)

        # function for the red circles
        red_circles = make_function(tree.nodes, functions={
            "center": ["1,2,index,N,2,/,floor,-,2,**,+,/,2,*,index,N,2,/,floor,-,*",
                       "1,2,index,N,2,/,floor,-,2,**,+,/,3,*", "0"],
            "radius": "1,2,index,N,2,/,floor,-,2,**,+,/"
        }, inputs=['N', 'index'], outputs=['center', 'radius'],
                                    scalars=['N', 'index', 'radius'], vectors=['center'], name="RedCircles")
        tree.links.new(n.std_out, red_circles.inputs['N'])
        tree.links.new(idx.std_out, red_circles.inputs['index'])

        cloud = Points(tree, name='PointCloud', position=red_circles.outputs["center"], count=n.std_out)

        circle = CurveCircle(tree, resolution=res.std_out, radius=1)
        mul_red = make_function(tree.nodes, functions={
            "s": "radius,growth,*"
        }, inputs=['radius', 'growth'], outputs=['s'], scalars=['radius', 'growth', 's'])
        tree.links.new(growth.std_out, mul_red.inputs['growth'])
        tree.links.new(red_circles.outputs['radius'], mul_red.inputs['radius'])
        instance = InstanceOnPoints(tree, instance=circle.geometry_out, scale=mul_red.outputs['s'])

        rotation = make_function(tree.nodes, functions={
            "Rotation": "pos,e_z,cross,pi,-2,/,axis_rot,rot2euler"
        }, name="Rotation", inputs=["pos"], outputs=["Rotation"], vectors=["pos", "Rotation"])
        instance2 = InstanceOnPoints(tree, instance=object_info.geometry_out)
        tree.links.new(rotation.outputs['Rotation'], instance2.inputs['Rotation'])
        tree.links.new(pos.std_out, rotation.inputs['pos'])
        material = SetMaterial(tree, material=self.colors[0])
        join = JoinGeometry(tree)
        create_geometry_line(tree, [cloud, instance, instance2, material, join],
                             out=self.group_outputs.inputs['Geometry'])

        # yellow circles
        n_string = "index,N,2,/,floor,-"
        r_string = "1,6,4," + n_string + ",*," + n_string + ",1,-,*,+,/"
        yellow_circles = make_function(tree.nodes, functions={
            "center": ["8," + n_string + ",*,4,-," + r_string + ",*", "9," + r_string + ",*", "0"],
            "radius": r_string
        }, inputs=['N', 'index'], outputs=['center', 'radius'],
                                       scalars=['N', 'index', 'radius'], vectors=['center'], name="YellowCircles")
        tree.links.new(n.std_out, yellow_circles.inputs['N'])
        tree.links.new(idx.std_out, yellow_circles.inputs['index'])

        cloud = Points(tree, name='PointCloud', position=yellow_circles.outputs["center"], count=n.std_out)
        circle = CurveCircle(tree, resolution=res.std_out, radius=1)
        mul_yellow = make_function(tree.nodes, functions={
            "s": "radius,growth,*"
        }, inputs=['radius', 'growth'], outputs=['s'], scalars=['radius', 'growth', 's'])
        tree.links.new(growth.std_out, mul_yellow.inputs['growth'])
        tree.links.new(yellow_circles.outputs['radius'], mul_yellow.inputs['radius'])
        instance = InstanceOnPoints(tree, instance=circle.geometry_out, scale=mul_yellow.outputs['s'])

        # rotation = make_function(group_tree.nodes, functions={
        #     "Rotation": "pos,e_z,cross,pi,-2,/,axis_rot,rot2euler"
        # }, name="Rotation", inputs=["pos"], outputs=["Rotation"], vectors=["pos", "Rotation"])
        instance2 = InstanceOnPoints(tree, instance=object_info.geometry_out)
        tree.links.new(rotation.outputs['Rotation'], instance2.inputs['Rotation'])
        tree.links.new(pos.std_out, rotation.inputs['pos'])
        material = SetMaterial(tree, material=self.colors[1])
        create_geometry_line(tree, [cloud, instance, instance2, material, join],
                             out=self.group_outputs.inputs['Geometry'])

        # blue circles
        n_string = "index,N,2,/,floor,-"
        r_string = "1,15,4," + n_string + ",*," + n_string + ",1,-,*,+,/"
        blue_circles = make_function(tree.nodes, functions={
            "center": ["8," + n_string + ",*,4,-," + r_string + ",*", "15," + r_string + ",*", "0"],
            "radius": r_string
        }, inputs=['N', 'index'], outputs=['center', 'radius'],
                                     scalars=['N', 'index', 'radius'], vectors=['center'], name="BlueCircles")
        tree.links.new(n.std_out, blue_circles.inputs['N'])
        tree.links.new(idx.std_out, blue_circles.inputs['index'])

        cloud = Points(tree, name='PointCloud', position=blue_circles.outputs["center"], count=n.std_out)
        circle = CurveCircle(tree, resolution=res.std_out, radius=1)
        mul_blue = make_function(tree.nodes, functions={
            "s": "radius,growth,*"
        }, inputs=['radius', 'growth'], outputs=['s'], scalars=['radius', 'growth', 's'])
        tree.links.new(growth.std_out, mul_blue.inputs['growth'])
        tree.links.new(blue_circles.outputs['radius'], mul_blue.inputs['radius'])
        instance = InstanceOnPoints(tree, instance=circle.geometry_out, scale=mul_blue.outputs['s'])

        # rotation = make_function(group_tree.nodes, functions={
        #     "Rotation": "pos,e_z,cross,pi,-2,/,axis_rot,rot2euler"
        # }, name="Rotation", inputs=["pos"], outputs=["Rotation"], vectors=["pos", "Rotation"])
        instance2 = InstanceOnPoints(tree, instance=object_info.geometry_out)
        tree.links.new(rotation.outputs['Rotation'], instance2.inputs['Rotation'])
        tree.links.new(pos.std_out, rotation.inputs['pos'])
        material = SetMaterial(tree, material=self.colors[2])
        create_geometry_line(tree, [cloud, instance, instance2, material, join],
                             out=self.group_outputs.inputs['Geometry'])

        self.arrow = arrow

    def get_arrow_object(self):
        return self.arrow

class LorentzAttractorNode(GeometryNodesModifier):
    def __init__(self, name='LorentzAttractor', iterations=15000, a=0.4):
        self.iterations = iterations
        self.a = a
        super().__init__(name)

    def create_node(self, tree):
        random_value = RandomValue(tree, data_type='FLOAT_VECTOR', min=-0.5 * Vector([1, 1, 1]),
                                   max=0.5 * Vector([1, 1, 1]))
        position = Position(tree)
        a_value = InputValue(tree, value=1.4)
        points = Points(tree, position=random_value.std_out)
        repeat = RepeatZone(tree, iterations=self.iterations, hide=False)
        repeat.add_socket(socket_type="GEOMETRY", name="Joined")
        repeat.geometry_out = repeat.repeat_output.outputs[
            "Joined"]  # make the second socket the default geometry output
        repeat.join_in_geometries(out_socket_name='Joined')

        namedAttribute = NamedAttribute(tree, data_type='FLOAT_VECTOR', name='pos')

        storedNamedAttr = StoredNamedAttribute(tree, data_type='FLOAT_VECTOR', name='pos')
        transformation = make_function(tree.nodes, functions={
            "position": [
                "old_pos_x,a,pos_x,*,-,4,pos_y,*,-,4,pos_z,*,-,pos_y,2,**,-",
                "old_pos_y,a,pos_y,*,-,4,pos_z,*,-,4,pos_x,*,-,pos_z,2,**,-",
                "old_pos_z,a,pos_z,*,-,4,pos_x,*,-,4,pos_y,*,-,pos_x,2,**,-",
            ]
        }, inputs=["pos", "old_pos", 'a'], outputs=["position"], vectors=["pos", "old_pos", "position"], scalars=['a'])
        tree.links.new(namedAttribute.std_out, transformation.inputs['old_pos'])
        tree.links.new(position.std_out, transformation.inputs['pos'])
        tree.links.new(transformation.outputs['position'], storedNamedAttr.inputs["Value"])
        tree.links.new(a_value.std_out, transformation.inputs['a'])
        namedAttribute2 = NamedAttribute(tree, data_type='FLOAT_VECTOR', name='pos')
        scale = VectorMath(tree, operation='SCALE', float_input=0.005, inputs0=namedAttribute2.std_out)
        setPosition = SetPosition(tree, position=scale.std_out)

        repeat.create_geometry_line([storedNamedAttr, setPosition])

        points2curve = PointsToCurve(tree)
        circle = CurveCircle(tree, resolution=8, radius=0.3)
        curve2mesh = CurveToMesh(tree, profile_curve=circle.geometry_out)

        create_geometry_line(tree, [points, repeat, points2curve, curve2mesh],
                             out=self.group_outputs.inputs['Geometry'])
        self.repeat = repeat

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]

# Penrose videos
class Penrose2DIntro(GeometryNodesModifier):
    def __init__(self, name='Penrose2D'):
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        angle = InputValue(tree, value=pi / 2, name="Angle")
        lattice_thickness = InputValue(tree, value=0.1, name="Thickness")
        visible_radius = InputValue(tree, value=0, name="GrowGrid")
        visible_length = InputValue(tree, value=0, name="GrowLine")

        position = Position(tree)
        projector = InputValue(tree, value=0, name='Projector')

        # create voronoi zone
        normal = make_function(tree, functions={
            "normal": ["angle,cos", "angle,sin", "0"]
        }, inputs=['angle'], outputs=['normal'], scalars=['angle'], vectors=['normal'],
                               name='Normal', hide=True)
        tree.links.new(angle.std_out, normal.inputs['angle'])

        # projection
        projection = make_function(tree, functions={
            "location": "position,normal,normal,position,dot,projector,*,scale,sub"
        }, inputs=['position', 'projector', 'normal'], outputs=['location'], scalars=['projector'],
                                   vectors=['normal', 'position', 'location'], name="Projection", hide=True)
        tree.links.new(normal.outputs['normal'], projection.inputs['normal'])
        tree.links.new(position.std_out, projection.inputs['position'])
        tree.links.new(projector.std_out, projection.inputs['projector'])

        # grid

        mesh = Grid(tree, size_x=30, size_y=20, vertices_y=21, vertices_x=31)
        join_geometry = JoinGeometry(tree)

        cube = CubeMesh(tree, size=[0.1] * 3)
        instance_on_points = InstanceOnPoints(tree, instance=cube.geometry_out)
        selector = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorGrid", inputs=['pos', 'r'], outputs=['select'], scalars=['r', 'select'], vectors=['pos'])
        tree.links.new(visible_radius.std_out, selector.inputs['r'])
        tree.links.new(position.std_out, selector.inputs['pos'])

        set_projection = SetPosition(tree, position=projection.outputs['location'])

        delete_geom = DeleteGeometry(tree, selection=selector.outputs['select'])
        create_geometry_line(tree, [mesh, delete_geom, set_projection, instance_on_points, join_geometry],
                             out=self.group_outputs.inputs[0])

        instance_on_edges = InstanceOnEdges(tree, radius=lattice_thickness.std_out, name="Lattice")
        mat = SetMaterial(tree, material='joker')
        create_geometry_line(tree, [set_projection, instance_on_edges, mat, join_geometry])

        # mesh line branch
        rotation = make_function(tree.nodes, functions={
            "rotation": ["pi,2,/", "0", "angle"]
        }, inputs=['angle'], outputs=['rotation'], scalars=['angle'], vectors=['rotation'], name="Rotation", hide=True)
        tree.links.new(angle.std_out, rotation.inputs['angle'])

        selector2 = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorLine", inputs=['pos', 'r'], outputs=['select'], scalars=['r', 'select'], vectors=['pos'])
        tree.links.new(visible_length.std_out, selector2.inputs['r'])
        tree.links.new(position.std_out, selector2.inputs['pos'])

        mesh_line = MeshLine(tree, name="ProjectionLine", count=1000,
                             start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        delete_line = DeleteGeometry(tree, selection=selector2.outputs['select'])
        instance_on_edge = InstanceOnEdges(tree, radius=0.025, resolution=8)
        transform_geometry2 = TransformGeometry(tree, rotation=rotation.outputs['rotation'])
        mat_line = SetMaterial(tree, material='plastic_custom1')

        create_geometry_line(tree,
                             [mesh_line, delete_line, instance_on_edge, transform_geometry2, mat_line, join_geometry])

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]

class Penrose2DVoronoi(GeometryNodesModifier):
    def __init__(self, name='Penrose2D'):
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        angle = InputValue(tree, value=pi / 2, name="Angle")
        lattice_thickness = InputValue(tree, value=0, name="Thickness")
        visible_radius = InputValue(tree, value=0, name="GrowGrid")
        visible_length = InputValue(tree, value=0, name="GrowLine")

        dl = InputVector(tree, value=Vector([0, 0, 0]), name="DL")
        dr = InputVector(tree, value=Vector([0, 1, 0]), name="DR")
        ul = InputVector(tree, value=Vector([1, 0, 0]), name="UL")
        ur = InputVector(tree, value=Vector([1, 1, 0]), name='UR')

        position = Position(tree)
        projector = InputValue(tree, value=0, name='Projector')

        shift = InputVector(tree, value=Vector(), name='VoronoiShift')
        offset = InputVector(tree, value=Vector(), name='GridShift')

        # normal
        normal = make_function(tree, functions={
            "normal": ["angle,cos", "angle,sin", "0"]
        }, inputs=['angle'], outputs=['normal'], scalars=['angle'], vectors=['normal'],
                               name='Normal', hide=True)
        tree.links.new(angle.std_out, normal.inputs['angle'])

        # projection
        projection = make_function(tree, functions={
            "location": "position,normal,normal,position,dot,projector,*,scale,sub"
        }, inputs=['position', 'projector', 'normal'], outputs=['location'], scalars=['projector'],
                                   vectors=['normal', 'position', 'location'], name="Projection", hide=True)
        tree.links.new(normal.outputs['normal'], projection.inputs['normal'])
        tree.links.new(position.std_out, projection.inputs['position'])
        tree.links.new(projector.std_out, projection.inputs['projector'])

        # grid

        mesh = Grid(tree, size_x=40, size_y=20, vertices_y=21, vertices_x=41)
        offset_transform = TransformGeometry(tree, translation=offset.std_out, name="GridShift")

        join_geometry = JoinGeometry(tree)

        cube = CubeMesh(tree, size=[0.1] * 3)
        instance_on_points = InstanceOnPoints(tree, instance=cube.geometry_out)

        set_projection = SetPosition(tree, position=projection.outputs['location'])

        instance_on_edges = InstanceOnEdges(tree, radius=lattice_thickness.std_out, name="Lattice")
        mat = SetMaterial(tree, material='joker', name="LatticeMaterial")

        create_geometry_line(tree, [set_projection, instance_on_edges, mat, join_geometry])

        # mesh line branch
        rotation = make_function(tree.nodes, functions={
            "rotation": ["pi,2,/", "0", "angle"]
        }, inputs=['angle'], outputs=['rotation'], scalars=['angle'], vectors=['rotation'],
                                 name="Rotation", hide=True)
        tree.links.new(angle.std_out, rotation.inputs['angle'])

        selector2 = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorLine", hide=True, inputs=['pos', 'r'], outputs=['select'],
                                  scalars=['r', 'select'], vectors=['pos'])
        tree.links.new(visible_length.std_out, selector2.inputs['r'])
        tree.links.new(position.std_out, selector2.inputs['pos'])

        mesh_line = MeshLine(tree, name="ProjectionLine", count=30,
                             start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        instance_on_edge = InstanceOnEdges(tree, radius=0.025, resolution=8)
        transform_geometry2 = TransformGeometry(tree, rotation=rotation.outputs['rotation'])
        mat_line = SetMaterial(tree, material='plastic_custom1', name="ProjectionMaterial")

        create_geometry_line(tree, [mesh_line, instance_on_edge, transform_geometry2, mat_line, join_geometry])

        # create voronoi zone

        voronoi = make_function(tree, functions={
            "min": [
                "dl,shift,add,normal,dot,dr,shift,add,normal,dot,min,ul,shift,add,normal,dot,min,ur,shift,add,normal,dot,min"],
            "max": [
                "dl,shift,add,normal,dot,dr,shift,add,normal,dot,max,ul,shift,add,normal,dot,max,ur,shift,add,normal,dot,max"],
        }, inputs=['dl', 'dr', 'ul', 'ur', 'normal', 'shift'], outputs=['min', 'max'],
                                scalars=['min', 'max'], vectors=['shift', 'dl', 'dr', 'ul', 'ur', 'normal'],
                                name='Voronoi',
                                hide=True)
        tree.links.new(normal.outputs['normal'], voronoi.inputs['normal'])
        tree.links.new(dl.std_out, voronoi.inputs['dl'])
        tree.links.new(dr.std_out, voronoi.inputs['dr'])
        tree.links.new(ur.std_out, voronoi.inputs['ur'])
        tree.links.new(ul.std_out, voronoi.inputs['ul'])
        tree.links.new(shift.std_out, voronoi.inputs['shift'])

        # implement min< n*pos <= max
        close_to = make_function(tree, functions={
            "is": ["position,normal,dot,minimum,>,position,normal,dot,maximum,>,not,*"],
            "is_not": ["position,normal,dot,minimum,>,not,position,normal,dot,maximum,>,+"]
        }, name="CloseTo", inputs=['position', 'minimum', 'maximum', 'normal'], outputs=['is', 'is_not'],
                                 scalars=['minimum', 'maximum', 'is', 'is_not'],
                                 vectors=['position', 'normal'], hide=True)
        tree.links.new(position.std_out, close_to.inputs['position'])
        tree.links.new(normal.outputs['normal'], close_to.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], close_to.inputs['minimum'])
        tree.links.new(voronoi.outputs['max'], close_to.inputs['maximum'])

        # finish grid line

        set_pos_for_inside_points = SetPosition(tree, selection=close_to.outputs["is"],
                                                position=projection.outputs["location"])
        grid_material = SetMaterial(tree, material='gray_5', name="GridMaterial")
        create_geometry_line(tree,
                             [mesh, offset_transform, set_pos_for_inside_points, instance_on_points, grid_material,
                              join_geometry],
                             out=self.group_outputs.inputs[0])

        # voronoi cell
        voronoi_cell = Grid(tree, name='VoronoiCell', size_x=1, size_y=1, vertices_x=2, vertices_y=2)
        voronoi_shift = make_function(tree, functions={
            "translation": "e_x,0.5,scale,e_y,0.5,scale,add,shift,add"
        }, inputs=['shift'], outputs=['translation'], vectors=['shift', 'translation'], hide=True, name='VoronoiShift')
        tree.links.new(shift.std_out, voronoi_shift.inputs['shift'])
        transform_geometry3 = TransformGeometry(tree, translation=voronoi_shift.outputs['translation'])
        mat_cell = SetMaterial(tree, material='plastic_drawing', name="VoronoiCellMaterial")
        create_geometry_line(tree, [voronoi_cell, transform_geometry3, mat_cell, join_geometry])

        # voronoi zone

        mesh_line_max = MeshLine(tree, name='SelectionZoneMax', count=1000,
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        mesh_line_min = MeshLine(tree, name='SelectionZoneMin', count=1000,
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))

        instance_on_edge_max = InstanceOnEdges(tree, resolution=4, radius=0.01)
        instance_on_edge_min = InstanceOnEdges(tree, resolution=4, radius=0.01)

        translation_max = make_function(tree, functions={
            "translation": "normal,maximum,scale"
        }, inputs=['normal', 'maximum'], outputs=['translation'], scalars=['maximum'],
                                        vectors=['normal', 'translation'], name='TransMax', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_max.inputs['normal'])
        tree.links.new(voronoi.outputs['max'], translation_max.inputs['maximum'])

        translation_min = make_function(tree, functions={
            "translation": "normal,minimum,scale"
        }, inputs=['normal', 'minimum'], outputs=['translation'], scalars=['minimum'],
                                        vectors=['normal', 'translation'], name='TransMin', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_min.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], translation_min.inputs['minimum'])

        transform_geometry_max = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_max.outputs['translation'])
        transform_geometry_min = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_min.outputs['translation'])
        mat_max = SetMaterial(tree, material='plastic_drawing', name="MaxMaterial")
        mat_min = SetMaterial(tree, material='plastic_drawing', name="MinMaterial")

        delete_line_max = DeleteGeometry(tree, selection=selector2.outputs['select'])
        delete_line_min = DeleteGeometry(tree, selection=selector2.outputs['select'])
        create_geometry_line(tree,
                             [mesh_line_max, delete_line_max, instance_on_edge_max, transform_geometry_max, mat_max,
                              join_geometry])
        create_geometry_line(tree,
                             [mesh_line_min, delete_line_min, instance_on_edge_min, transform_geometry_min, mat_min,
                              join_geometry])

        # highlight inside zone:
        # inside line
        selector = make_function(tree, functions={
            'select': "pos,length,r,>"
        }, name="SelectorGrid", inputs=['pos', 'r'], outputs=['select'],
                                 scalars=['r', 'select'], vectors=['pos'], hide=True)
        tree.links.new(visible_radius.std_out, selector.inputs['r'])
        tree.links.new(position.std_out, selector.inputs['pos'])

        delete_geometry = DeleteGeometry(tree, selection=close_to.outputs["is_not"])
        delete_geometry2 = DeleteGeometry(tree, selection=selector.outputs["select"])

        icosphere = IcoSphere(tree, radius=0.1, subdivisions=2, name="SelectedPoints")
        instance_on_points = InstanceOnPoints(tree, instance=icosphere.geometry_out, selection=close_to.outputs["is"])
        set_pos_icos = SetPosition(tree, position=projection.outputs['location'])
        icos_mat = SetMaterial(tree, material='plastic_example', name="IcosphereMaterial")
        create_geometry_line(tree, [offset_transform, delete_geometry,
                                    instance_on_points, set_pos_icos, icos_mat, join_geometry],
                             out=self.group_outputs.inputs[0])

        # split horizontal and vertical lines
        edge_vertices = EdgeVertices(tree)
        horizontal_tester = make_function(tree, functions={
            "is": "pos1,pos2,sub,e_x,dot,abs,0,>",
        }, inputs=["pos1", "pos2"], outputs=["is", "is_not"], vectors=["pos1", "pos2"], scalars=["is"],
                                          name="HorizontalTester", hide=True)
        tree.links.new(edge_vertices.outputs["Position 1"], horizontal_tester.inputs["pos1"])
        tree.links.new(edge_vertices.outputs["Position 2"], horizontal_tester.inputs["pos2"])

        stored_attribute = StoredNamedAttribute(tree, data_type='BOOLEAN', domain='EDGE', name='horizontalAttribute',
                                                value=horizontal_tester.outputs[0])

        named_attribute = NamedAttribute(tree, data_type='BOOLEAN', name='horizontalAttribute')
        not_node = BooleanMath(tree, operation='NOT', inputs0=named_attribute.std_out, name="Not")

        # horizontal
        instance_on_edges_horizontal = InstanceOnEdges(tree, selection=named_attribute.std_out, resolution=8,
                                                       radius=0.05, name="Horizontal")
        set_pos_horizontal = SetPosition(tree, position=projection.outputs['location'])
        horizontal_mat = SetMaterial(tree, selection=None, material='plastic_joker', name="HorizontalMaterial")

        # vertical
        instance_on_edges_vertical = InstanceOnEdges(tree, selection=not_node.std_out, resolution=8, radius=0.05,
                                                     name="Vertical")
        set_pos_vertical = SetPosition(tree, position=projection.outputs['location'])
        vertical_mat = SetMaterial(tree, selection=None, material='plastic_important', name="VerticalMaterial")

        create_geometry_line(tree, [offset_transform, delete_geometry, stored_attribute, delete_geometry2,
                                    set_pos_horizontal,
                                    instance_on_edges_horizontal, horizontal_mat, join_geometry],
                             out=self.group_outputs.inputs[0])
        create_geometry_line(tree, [stored_attribute, delete_geometry2, set_pos_vertical, instance_on_edges_vertical,
                                    vertical_mat,
                                    join_geometry])

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]

class Penrose2D(GeometryNodesModifier):
    def __init__(self, name='Penrose2D'):
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        angle = InputValue(tree, value=0, name="angle")
        offset = InputVector(tree, value=Vector([0.2, 0.2, 0]), name='Offset')
        dl = InputVector(tree, value=Vector([0, 0, 0]), name="DL")
        dr = InputVector(tree, value=Vector([0, 1, 0]), name="DR")
        ul = InputVector(tree, value=Vector([1, 0, 0]), name="UL")
        ur = InputVector(tree, value=Vector([1, 1, 0]), name='UR')
        position = Position(tree)
        projector = InputValue(tree, value=0, name='Projector')

        # create voronoi zone
        normal = make_function(tree, functions={
            "normal": ["angle,cos", "angle,sin", "0"]
        }, inputs=['angle'], outputs=['normal'], scalars=['angle'], vectors=['normal'],
                               name='Normal', hide=True)
        tree.links.new(angle.std_out, normal.inputs['angle'])

        voronoi = make_function(tree, functions={
            "min": ["dl,normal,dot,dr,normal,dot,min,ul,normal,dot,min,ur,normal,dot,min"],
            "max": ["dl,normal,dot,dr,normal,dot,max,ul,normal,dot,max,ur,normal,dot,max"],
        }, inputs=['dl', 'dr', 'ul', 'ur', 'normal'], outputs=['min', 'max'],
                                scalars=['min', 'max'], vectors=['dl', 'dr', 'ul', 'ur', 'normal'], name='Voronoi',
                                hide=True)
        tree.links.new(normal.outputs['normal'], voronoi.inputs['normal'])
        tree.links.new(dl.std_out, voronoi.inputs['dl'])
        tree.links.new(dr.std_out, voronoi.inputs['dr'])
        tree.links.new(ur.std_out, voronoi.inputs['ur'])
        tree.links.new(ul.std_out, voronoi.inputs['ul'])

        close_to = make_function(tree, functions={
            "is": ["position,normal,dot,minimum,>,position,normal,dot,maximum,<,*"],
            "is_not": ["position,normal,dot,minimum,>,position,normal,dot,maximum,<,*,not"]
        }, name="CloseTo", inputs=['position', 'minimum', 'maximum', 'normal'], outputs=['is', 'is_not'],
                                 scalars=['minimum', 'maximum', 'is', 'is_not'],
                                 vectors=['position', 'normal'], hide=True)
        tree.links.new(position.std_out, close_to.inputs['position'])
        tree.links.new(normal.outputs['normal'], close_to.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], close_to.inputs['minimum'])
        tree.links.new(voronoi.outputs['max'], close_to.inputs['maximum'])

        # projection
        projection = make_function(tree, functions={
            "location": "position,normal,normal,position,dot,projector,*,scale,sub"
        }, inputs=['position', 'projector', 'normal'], outputs=['location'], scalars=['projector'],
                                   vectors=['normal', 'position', 'location'], name="Projection", hide=True)
        tree.links.new(normal.outputs['normal'], projection.inputs['normal'])
        tree.links.new(position.std_out, projection.inputs['position'])
        tree.links.new(projector.std_out, projection.inputs['projector'])

        # inside line

        mesh = Grid(tree, size_x=20, size_y=20, vertices_y=21, vertices_x=21)
        transform_geometry = TransformGeometry(tree, translation=offset.std_out)
        delete_geometry = DeleteGeometry(tree, selection=close_to.outputs["is_not"])

        join_geometry = JoinGeometry(tree)

        # split horizontal and vertical lines
        edge_vertices = EdgeVertices(tree)
        horizontal_tester = make_function(tree, functions={
            "is": "pos1,pos2,sub,e_x,dot,abs,0,>",
        }, inputs=["pos1", "pos2"], outputs=["is", "is_not"], vectors=["pos1", "pos2"], scalars=["is"],
                                          name="HorizontalTester", hide=True)
        tree.links.new(edge_vertices.outputs["Position 1"], horizontal_tester.inputs["pos1"])
        tree.links.new(edge_vertices.outputs["Position 2"], horizontal_tester.inputs["pos2"])

        stored_attribute = StoredNamedAttribute(tree, data_type='BOOLEAN', domain='EDGE', name='horizontalAttribute',
                                                value=horizontal_tester.outputs[0])

        named_attribute = NamedAttribute(tree, data_type='BOOLEAN', name='horizontalAttribute')
        not_node = BooleanMath(tree, operation='NOT', inputs0=named_attribute.std_out, name="Not")

        # horizontal
        instance_on_edges_horizontal = InstanceOnEdges(tree, selection=named_attribute.std_out, resolution=4,
                                                       radius=0.05, name="Horizontal")
        set_pos_horizontal = SetPosition(tree, position=projection.outputs['location'])
        horizontal_mat = SetMaterial(tree, selection=None, material='plastic_joker')

        # vertical
        instance_on_edges_vertical = InstanceOnEdges(tree, selection=not_node.std_out, resolution=4, radius=0.05,
                                                     name="Vertical")
        set_pos_vertical = SetPosition(tree, position=projection.outputs['location'])
        vertical_mat = SetMaterial(tree, selection=None, material='plastic_important')

        create_geometry_line(tree, [mesh, transform_geometry, delete_geometry, stored_attribute, set_pos_horizontal,
                                    instance_on_edges_horizontal, horizontal_mat, join_geometry],
                             out=self.group_outputs.inputs[0])
        create_geometry_line(tree, [stored_attribute, set_pos_vertical, instance_on_edges_vertical, vertical_mat,
                                    join_geometry])

        # outside
        cube = CubeMesh(tree, size=[0.1] * 3)
        instance_on_points2 = InstanceOnPoints(tree, selection=close_to.outputs["is_not"], instance=cube.geometry_out)
        create_geometry_line(tree, [transform_geometry, instance_on_points2, join_geometry])

        # sphere branch
        icosphere = IcoSphere(tree, radius=0.1, subdivisions=2, name="SelectedPoints")
        instance_on_points = InstanceOnPoints(tree, instance=icosphere.geometry_out, selection=close_to.outputs["is"])
        set_pos2 = SetPosition(tree, position=projection.outputs['location'])
        cube_mat = SetMaterial(tree, material='plastic_example')
        create_geometry_line(tree, [transform_geometry, instance_on_points, set_pos2, cube_mat, join_geometry])

        # mesh line branch
        rotation = make_function(tree.nodes, functions={
            "rotation": ["pi,2,/", "0", "angle"]
        }, inputs=['angle'], outputs=['rotation'], scalars=['angle'], vectors=['rotation'], name="Rotation", hide=True)
        tree.links.new(angle.std_out, rotation.inputs['angle'])

        mesh_line = MeshLine(tree, name="ProjectionLine",
                             start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        instance_on_edge = InstanceOnEdges(tree, radius=0.025, resolution=8)
        transform_geometry2 = TransformGeometry(tree, rotation=rotation.outputs['rotation'])
        mat_line = SetMaterial(tree, material='plastic_custom1')

        create_geometry_line(tree, [mesh_line, instance_on_edge, transform_geometry2, mat_line, join_geometry])

        # voronoi cell
        voronoi_cell = Grid(tree, name='VoronoiCell', size_x=1, size_y=1, vertices_x=2, vertices_y=2)
        transform_geometry3 = TransformGeometry(tree, translation=[0.5, 0.5, 0])
        mat_cell = SetMaterial(tree, material='plastic_drawing')
        create_geometry_line(tree, [voronoi_cell, transform_geometry3, mat_cell, join_geometry])

        # voronoi zone

        mesh_line_max = MeshLine(tree, name='SelectionZoneMax',
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))
        mesh_line_min = MeshLine(tree, name='SelectionZoneMin',
                                 start_location=Vector([0, 0, -15]), end_location=Vector([0, 0, 15]))

        instance_on_edge_max = InstanceOnEdges(tree, resolution=4, radius=0.01)
        instance_on_edge_min = InstanceOnEdges(tree, resolution=4, radius=0.01)

        translation_max = make_function(tree, functions={
            "translation": "normal,maximum,scale"
        }, inputs=['normal', 'maximum'], outputs=['translation'], scalars=['maximum'],
                                        vectors=['normal', 'translation'], name='TransMax', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_max.inputs['normal'])
        tree.links.new(voronoi.outputs['max'], translation_max.inputs['maximum'])

        translation_min = make_function(tree, functions={
            "translation": "normal,minimum,scale"
        }, inputs=['normal', 'minimum'], outputs=['translation'], scalars=['minimum'],
                                        vectors=['normal', 'translation'], name='TransMin', hide=True
                                        )
        tree.links.new(normal.outputs['normal'], translation_min.inputs['normal'])
        tree.links.new(voronoi.outputs['min'], translation_min.inputs['minimum'])

        transform_geometry_max = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_max.outputs['translation'])
        transform_geometry_min = TransformGeometry(tree, rotation=rotation.outputs['rotation'],
                                                   translation=translation_min.outputs['translation'])
        mat_max = SetMaterial(tree, material='plastic_drawing')
        mat_min = SetMaterial(tree, material='plastic_drawing')
        create_geometry_line(tree,
                             [mesh_line_max, instance_on_edge_max, transform_geometry_max, mat_max, join_geometry])
        create_geometry_line(tree,
                             [mesh_line_min, instance_on_edge_min, transform_geometry_min, mat_min, join_geometry])

    def get_iteration_socket(self):
        return self.repeat.repeat_input.inputs[0]

class ConvexHull2D(GeometryNodesModifier):
    def __init__(self, name='ConvexHull2D', size=10):
        self.size = size
        super().__init__(name)

    def create_node(self, tree):
        # parameters
        size = self.size

        # input nodes
        position = Position(tree)
        cube_scale = InputValue(tree, value=1, name="CubeScale")
        r_max = InputValue(tree, hide=True, name='rMax', value=1.1 * size * r3)
        u = InputVector(tree, value=Vector([1, 1, 1]), name="Normal")
        l_max = InputValue(tree, hide=True, name='lMax', value=0)
        p_max = InputValue(tree, hide=True, name='pMax', value=0)
        ico_max = InputValue(tree, hide=True, name='icoMax', value=0)
        voronoi_scale = InputValue(tree, name="voronoiScale", value=0)
        # output nodes
        out = tree.nodes.get("Group Output")

        # grid

        mesh_line = MeshLine(tree, count=2 * size + 1, start_location=Vector([0, 0, -size]),
                             end_location=Vector([0, 0, size]))
        grid = Grid(tree, size_x=2 * size, size_y=2 * size, vertices_x=2 * size + 1, vertices_y=2 * size + 1)
        instance_on_points = InstanceOnPoints(tree, instance=grid.outputs['Mesh'])
        realize_instances = RealizeInstances(tree)
        cubie_scale = InputValue(tree, value=0.05, name="CubieScale")
        cubies = CubeMesh(tree, size=cubie_scale.std_out)
        set_material = SetMaterial(tree, material='text')
        nodes = [r_max, position]
        all_labels = ['r_max', 'position']

        selector = make_function(tree.nodes, functions={
            'selector': ['position,length,r_max,<']
        }, name='Limits', inputs=all_labels,
                                 outputs=['selector'],
                                 vectors=['position'],
                                 scalars=['r_max', 'selector'], hide=True)

        for node, label in zip(nodes, all_labels):
            tree.links.new(node.std_out, selector.inputs[label])

        create_geometry_line(tree, [cubies, set_material])

        instance_on_points2 = InstanceOnPoints(tree, selection=selector.outputs['selector'],
                                               instance=set_material.geometry_out)

        join = JoinGeometry(tree)
        shade_smooth = SetShadeSmooth(tree)
        join2 = JoinGeometry(tree)

        # voronoi cell

        cube = CubeMesh(tree, name="VoronoiCell", size=voronoi_scale.std_out)
        wire_frame = WireFrame(tree, resolution=8, radius=0.025)
        transform_geo = TransformGeometry(tree, translation=[0.5] * 3, scale=cube_scale.std_out)
        drawing = SetMaterial(tree, material='plastic_drawing', roughness=0.25, name="CubeMaterial")

        # projection line

        normalizer = make_function(tree, functions={
            'normalized': 'normal,normalize',
            'neg_normalized': 'normal,normalize,-1,scale',
            'u': 'e_z,normal,cross,normalize,e_z,normal,cross,length,0,>,scale,e_x,normal,cross,normalize,1,e_z,normal,cross,length,0,>,-,scale,add',
            'v': 'normal,e_z,normal,cross,cross,normalize,e_z,normal,cross,length,0,>,scale,normal,e_x,normal,cross,cross,normalize,1,e_z,normal,cross,length,0,>,-,scale,add',
            'euler': 'e_z,normal,normalize,cross,normal,normalize,e_z,dot,acos,axis_angle_euler',
        }, name="Normalizer", inputs=['normal'], outputs=['normalized', 'neg_normalized', 'u', 'v', 'euler'],
                                   vectors=['normal', 'normalized', 'neg_normalized', 'u', 'v', 'euler'], hide=True)
        tree.links.new(u.std_out, normalizer.inputs['normal'])

        ## length of projection line with ray-cast

        boundary_cube = CubeMesh(tree, size=[2 * size] * 3, name="BoundaryCube")

        positive_ray = RayCast(tree, data_type='FLOAT_VECTOR', target_geometry=boundary_cube.geometry_out,
                               ray_direction=normalizer.outputs['normalized'],
                               source_position=normalizer.outputs['normalized'])
        negative_ray = RayCast(tree, data_type='FLOAT_VECTOR', target_geometry=boundary_cube.geometry_out,
                               ray_direction=normalizer.outputs['neg_normalized'],
                               source_position=normalizer.outputs['neg_normalized'])
        scaler_pos = VectorMath(tree, operation='SCALE', inputs0=positive_ray.outputs['Hit Position'],
                                float_input=l_max.std_out, name="LineScaling")
        scaler_neg = VectorMath(tree, operation='SCALE', inputs0=negative_ray.outputs['Hit Position'],
                                float_input=l_max.std_out, name="LineScaling")
        projection_line = MeshLine(tree, name="ProjectionLine", count=30,
                                   start_location=scaler_neg.std_out, end_location=scaler_pos.std_out)
        instance_on_edge = InstanceOnEdges(tree, radius=0.0225, resolution=8)
        mat_line = SetMaterial(tree, material='plastic_custom1', name="ProjectionMaterial")

        # orthogonal plane
        # orthogonal_grid = Grid(group_tree,size_x=2*size,size_y=2*size,vertices_x=2*size+1,vertices_y=2*size+1,name="OrthogonalPlane")
        # rotate_orthogonal_grid = TransformGeometry(group_tree,rotation=normalizer.outputs['euler'])
        # orthogonal_wire_frame = WireFrame(group_tree,name="OrthogonalPlaneWire")

        # projector

        projector = make_function(tree, functions={
            'projection': "u,pos,u,dot,scale,v,pos,v,dot,scale,add"
        }, inputs=["pos", "u", "v"], outputs=["projection"], vectors=["pos", "u", "v", "projection"],
                                  name="Projector", hide=True)
        tree.links.new(position.std_out, projector.inputs['pos'])
        tree.links.new(normalizer.outputs['u'], projector.inputs['u'])
        tree.links.new(normalizer.outputs['v'], projector.inputs['v'])

        # projected cube
        scale_min = make_function(tree, functions={"pos_min": "hit_pos,p_min,scale"}, inputs=["p_min", "hit_pos"],
                                  outputs=["pos_min"], scalars=["p_min"], vectors=["hit_pos", "pos_min"],
                                  name="moveMin", hide=True)
        scale_max = make_function(tree, functions={"pos_max": "hit_pos,p_max,scale"}, inputs=["p_max", "hit_pos"],
                                  outputs=["pos_max"], scalars=["p_max"], vectors=["hit_pos", "pos_max"],
                                  name="moveMax", hide=True)
        tree.links.new(p_max.std_out, scale_min.inputs["p_min"])
        tree.links.new(p_max.std_out, scale_max.inputs["p_max"])
        tree.links.new(positive_ray.outputs['Hit Position'], scale_max.inputs["hit_pos"])
        tree.links.new(negative_ray.outputs['Hit Position'], scale_min.inputs["hit_pos"])
        set_projected_pos = SetPosition(tree, position=projector.outputs['projection'], name="Projection")
        move_to_max = TransformGeometry(tree, translation=scale_max.outputs["pos_max"])
        wire_frame_max = WireFrame(tree, resolution=16, radius=0.025, name="WireFrameMax")
        sphere_max = IcoSphere(tree, radius=0.025, subdivisions=3)
        iop_max = InstanceOnPoints(tree, instance=sphere_max.geometry_out)
        sphere_max_mat = SetMaterial(tree, material='plastic_joker')
        convex_hull_max_material = SetMaterial(tree, material='plastic_joker', name='ConvexHullMaterialMax')
        move_to_min = TransformGeometry(tree, translation=scale_min.outputs["pos_min"])
        wire_frame_min = WireFrame(tree, resolution=16, radius=0.025, name="WireFrameMin")
        sphere_min = IcoSphere(tree, radius=0.025, subdivisions=3)
        iop_min = InstanceOnPoints(tree, instance=sphere_max.geometry_out)
        sphere_min_mat = SetMaterial(tree, material='plastic_joker')
        convex_hull_min_material = SetMaterial(tree, material='plastic_joker', name='ConvexHullMaterialMin')

        # convex hull
        convex_hull = ConvexHull(tree)
        scale_element = ScaleElements(tree, scale=1.0001)  # to include boundary cases
        extrude = ExtrudeMesh(tree, mode="FACES")  # give finite thickness to convex hull to make ray tracing feasible

        # select inside convex hull
        icosphere = IcoSphere(tree, subdivisions=2, name="SelectedSphere")
        selector = make_function(tree, functions={
            "is_selected": "ico_max,ray_hit,length,*,pos,length,>"
        }, inputs=["pos", "ray_hit", "ico_max"], outputs=["is_selected"], vectors=["pos", "ray_hit"],
                                 scalars=["is_selected", "ico_max"],
                                 hide=True, name="SelectorFunction")
        tree.links.new(ico_max.std_out, selector.inputs['ico_max'])
        tree.links.new(position.std_out, selector.inputs['pos'])
        tree.links.new(positive_ray.outputs["Hit Position"], selector.inputs['ray_hit'])

        convex_hull_test = InsideConvexHull(tree, source_position=projector.outputs['projection'],
                                            ray_direction=normalizer.outputs['u'])

        and_node = BooleanMath(tree, operation="AND", inputs0=selector.outputs["is_selected"],
                               inputs1=convex_hull_test.std_out)
        instance_on_grid_points = InstanceOnPoints(tree, selection=and_node.std_out, instance=icosphere.geometry_out)
        selected_material = SetMaterial(tree, material='plastic_example', name='SelectedMaterial')

        # create voronoi zone

        product_geometry = InstanceOnPoints(tree, instance=convex_hull.geometry_out)
        realize = RealizeInstances(tree)
        hull_hull = ConvexHull(tree)
        zone_mat = SetMaterial(tree, material="fake_glass_joker", ior=1.01, name="ZoneMaterial")

        # create geometry lines
        # lattice grid
        create_geometry_line(tree, [
            mesh_line,
            instance_on_points,
            realize_instances,
            instance_on_points2,
            join,
            shade_smooth,
            join2
        ], out=out.inputs['Geometry'])
        # voronoi cell
        create_geometry_line(tree, [cube, transform_geo, wire_frame, drawing, join2])
        # projection line
        create_geometry_line(tree, [projection_line, instance_on_edge, mat_line, join])
        # orthogonal space
        # create_geometry_line(group_tree,[orthogonal_grid,rotate_orthogonal_grid,orthogonal_wire_frame,join])
        # display convex hull geometry
        create_geometry_line(tree,
                             [transform_geo, set_projected_pos, move_to_max, wire_frame_max, convex_hull_max_material,
                              join])
        create_geometry_line(tree, [set_projected_pos, move_to_min, wire_frame_min, convex_hull_min_material, join])
        create_geometry_line(tree, [move_to_min, iop_min, sphere_min_mat, join])
        create_geometry_line(tree, [move_to_max, iop_max, sphere_max_mat, join])
        # inside convex hull test as selector
        create_geometry_line(tree, [set_projected_pos, convex_hull, scale_element, extrude, convex_hull_test])
        # selected grid points
        create_geometry_line(tree, [realize_instances, instance_on_grid_points, selected_material, join2])
        # voronoi zone
        create_geometry_line(tree, [projection_line, product_geometry, realize, hull_hull, zone_mat, join2])

# CMB video
class SphericalHarmonicsNode(GeometryNodesModifier):
    def __init__(self, l=0, m=0, name='SphericalHarmonics', resolution=5, **kwargs):
        """
        creates an object that represents a spherical harmonics
        with input value lambda the magnitude of the real value is imprinted into the geometry
        :param name:
        :param l: orbital quantum number
        :param m: magnetic quantum number
        :param kwargs:
        """
        self.l = l
        self.m = m
        self.resolution = resolution
        self.kwargs = kwargs
        super().__init__(name)

    def create_node(self, tree):
        # recalculate position of vertices
        position = Position(tree, location=(-6, 0))
        lmbda = InputValue(tree, name='lambda', location=(-6, 1))

        # convert to polar coordinates
        cart2polar = make_function(tree, name="PolarCoordinates", hide=True, location=(-5, 0),
                                   functions={
                                       "r": "position,length",
                                       "theta": "position_z,acos",
                                       "phi": "position_y,position_x,atan2"
                                   }, inputs=["position"], outputs=["r", "theta", "phi"],
                                   vectors=["position"], scalars=["r", "theta", "phi"])

        tree.links.new(position.std_out, cart2polar.inputs["position"])
        # create spherical harmonics terms
        y_lm = SphericalHarmonics(self.l, self.m, "theta", "phi")

        real_part = ExpressionConverter(y_lm.real()).postfix()
        imag_part = ExpressionConverter(y_lm.imag()).postfix()

        if len(imag_part.strip()) == 0:
            imag_part = "0"
        print("real: ", real_part)
        print("imag: ", imag_part)

        compute_y_lm = make_function(tree, name="Y_lm", functions={
            "re": real_part,
            "im": imag_part
        },
                                     inputs=["theta", "phi"],
                                     outputs=["re", "im"], scalars=["re", "im", "theta", "phi"], hide=True,
                                     location=(-4, 0))

        tree.links.new(cart2polar.outputs["theta"], compute_y_lm.inputs["theta"])
        tree.links.new(cart2polar.outputs["phi"], compute_y_lm.inputs["phi"])

        complex_analyser = make_function(tree, name="Analyser", hide=True, functions={
            "re": "re",
            "im": "im",
            "absolute": "re,re,*,im,im,*,+,sqrt",
            "phase": "im,re,atan2"
        }, inputs=["re", "im"], outputs=["re", "im", "absolute", "phase"], scalars=["re", "im", "absolute", "phase"],
                                         location=(-3, 0))

        tree.links.new(compute_y_lm.outputs["re"], complex_analyser.inputs["re"])
        tree.links.new(compute_y_lm.outputs["im"], complex_analyser.inputs["im"])

        # transform position
        vals = ["re", "im", "absolute", "phase"]
        trafo = make_function(tree, name="Transformation", hide=True, functions={
            "position": "position,re,abs,lambda,*,1,lambda,-,+,scale"
        }, inputs=["lambda", "position"] + vals, outputs=["position"],
                              vectors=["position"],
                              scalars=["lambda"] + vals, location=(-2, 0))

        tree.links.new(position.std_out, trafo.inputs["position"])
        for val in vals:
            tree.links.new(complex_analyser.outputs[val], trafo.inputs[val])
        tree.links.new(lmbda.std_out, trafo.inputs["lambda"])
        # create default spherical geometry
        sphere = UVSphere(tree, location=(-3, 1), rings=2 ** self.resolution, segments=2 ** (self.resolution + 1),
                          hide=False)
        set_pos = SetPosition(tree, location=(-2, 1), position=trafo.outputs["position"])

        # store the phase for coloring
        attr = StoredNamedAttribute(tree, location=(-1, -2), name="Phase", value=complex_analyser.outputs["phase"])
        material = phase2hue_material(attribute_names=["Phase"], **self.kwargs)
        self.materials.append(material)
        color = SetMaterial(tree, location=(0, -1), material=material)

        smooth = SetShadeSmooth(tree, location=(1, -1))

        create_geometry_line(tree, [sphere, set_pos, attr, color, smooth], out=self.group_outputs.inputs[0])

class SphericalHarmonicsNode2(GeometryNodesModifier):
    def __init__(self, l=0, m=0, name='SphericalHarmonics', resolution=5, **kwargs):
        """
        creates an object that morphs from a sphere to the represention of a spherical harmonics
        specialized node setup that is used in the video_cmb
        From a wireframe sphere into a spherical harmonics
        :param name:
        :param l: orbital quantum number
        :param m: magnetic quantum number
        :param kwargs:
        """
        self.l = l
        self.m = m
        self.resolution = resolution
        self.kwargs = kwargs
        super().__init__(name, automatic_layout=False)

    def create_node(self, tree):

        # create colorful wireframe of sphere
        left = -20
        out = tree.nodes.get("Group Output")
        links = tree.links

        reset_left = left
        # vertical grid lines
        line = MeshLine(tree, location=(left, 2), count=self.resolution * 4, start_location=Vector(),
                        end_location=Vector([tau, 0, 0]))
        left += 1
        mesh2points = MeshToPoints(tree, location=(left, 2))
        left += 1
        points2verts = PointsToVertices(tree, location=(left, 2))
        index = Index(tree, location=(left, 1))
        left += 1
        attr = StoredNamedAttribute(tree, location=(left, 2), name="Index", data_type="INT", value=index.std_out)
        pi_offset = InputVector(tree, location=(left, 1), value=[0, 0, pi])
        left += 1
        extrude_mesh = ExtrudeMesh(tree, location=(left, 2), offset=pi_offset.std_out)
        left += 1
        sub_div = SubdivideMesh(tree, location=(left, 1), level=5)
        left += 1
        position = Position(tree, location=(left - 1, 3.5))

        # trafo
        # similar for the other components
        x = "pos_x,cos,pos_z,pi,-,sin,*"
        y = "pos_x,sin,pos_z,pi,-,sin,*"
        z = "pos_z,pi,-,cos"

        trafo = make_function(tree, functions={
            "position": [
                x,
                y,
                z,
            ]
        }, inputs=["pos"], outputs=["position"],
                              vectors=["pos", "position"],
                              name="Map2Sphere", location=(left, 1.5), hide=True)
        links.new(position.std_out, trafo.inputs["pos"])
        left += 1
        set_pos = SetPosition(tree, location=(left, 1), position=trafo.outputs["position"])
        left += 1
        wireframe = WireFrame(tree, radius=0.01, location=(left, 1))
        left += 1
        material = gradient_from_attribute(name="Index",
                                           function="fac," + str(self.resolution * 4) + ",/",
                                           attr_name="Index",
                                           gradient={0: [1, 0, 0, 1], 1: [0.8, 0, 1, 1]},
                                           alpha_function={
                                               "Alpha": "1,alpha,alpha,*,-"})  # fading in from -1 to 0 fading out from 0 to 1
        mat = SetMaterial(tree, location=(left, 1), material=material)
        self.materials.append(material)
        left += 1
        join = JoinGeometry(tree, location=(left, 0))
        create_geometry_line(tree,
                             [line, mesh2points, points2verts, attr, extrude_mesh, sub_div, set_pos, wireframe, mat,
                              join])

        # horizontal grid lines
        left = reset_left

        line = MeshLine(tree, location=(left, -2), count=self.resolution * 2, start_location=Vector([0, 0, pi]),
                        end_location=Vector([0, 0, 0]))
        left += 1
        mesh2points = MeshToPoints(tree, location=(left, -2))
        left += 1
        points2verts = PointsToVertices(tree, location=(left, -2))
        index = Index(tree, location=(left, 1))
        left += 1
        attr = StoredNamedAttribute(tree, location=(left, -2), name="Index2", data_type="INT", value=index.std_out)
        tau_offset = InputVector(tree, location=(left, -1), value=[tau, 0, 0])
        left += 1
        extrude_mesh = ExtrudeMesh(tree, location=(left, -2), offset=tau_offset.std_out)
        left += 1
        sub_div = SubdivideMesh(tree, location=(left, -1), level=5)
        left += 1
        position = Position(tree, location=(left - 1, -2))

        # trafo

        trafo = make_function(tree, functions={
            "position": [
                x,
                y,
                z,
            ]
        }, inputs=["pos"], outputs=["position"],
                              vectors=["pos", "position", "shift"],
                              name="Map2Sphere2", location=(left, -1.5), hide=True)
        links.new(position.std_out, trafo.inputs["pos"])
        left += 1
        set_pos = SetPosition(tree, location=(left, -1), position=trafo.outputs["position"])
        left += 1
        wireframe = WireFrame(tree, radius=0.01, location=(left, -1))
        left += 1
        material = gradient_from_attribute(name="Index2",
                                           function="fac," + str(self.resolution * 2) + ",/",
                                           attr_name="Index2",
                                           gradient={0: [0, 1, 0.95, 1], 1: [1, 1, 0, 1]},
                                           alpha_function={
                                               "Alpha": "1,alpha,alpha,*,-"})  # fading in from -1 to 0 fading out from 0 to 1
        mat = SetMaterial(tree, location=(left, 1), material=material)
        self.materials.append(material)
        left += 1

        create_geometry_line(tree,
                             [line, mesh2points, points2verts, attr, extrude_mesh, sub_div, set_pos, wireframe, mat,
                              join], out=out.inputs["Geometry"])

        # recalculate position of vertices
        sphere_position = Position(tree, location=(left, 2))
        lambda_node = InputValue(tree, name='lambda', location=(left, 1.5), value=-1)

        # create default spherical geometry

        sphere = UVSphere(tree, location=(left, 1), rings=2 ** self.resolution, segments=2 ** (self.resolution + 1),
                          hide=False)
        sphere_node_pos = left
        left += 1
        # convert to polar coordinates
        cart2polar = make_function(tree, name="PolarCoordinates", hide=True, location=(left, -2),
                                   functions={
                                       "r": "position,length",
                                       "theta": "position_z,acos",
                                       "phi": "position_y,position_x,atan2"
                                   }, inputs=["position"], outputs=["r", "theta", "phi"],
                                   vectors=["position"], scalars=["r", "theta", "phi"])

        tree.links.new(sphere_position.std_out, cart2polar.inputs["position"])
        left += 1

        # create spherical harmonics terms
        y_lm = SphericalHarmonics(self.l, self.m, "theta", "phi")

        real_part = ExpressionConverter(y_lm.real()).postfix()
        imag_part = ExpressionConverter(y_lm.imag()).postfix()

        if len(imag_part.strip()) == 0:
            imag_part = "0"
        print("real: ", real_part)
        print("imag: ", imag_part)

        compute_y_lm = make_function(tree, name="Y_lm", functions={
            "re": real_part,
            "im": imag_part
        },
                                     inputs=["theta", "phi"],
                                     outputs=["re", "im"], scalars=["re", "im", "theta", "phi"], hide=True,
                                     location=(left, -2))

        tree.links.new(cart2polar.outputs["theta"], compute_y_lm.inputs["theta"])
        tree.links.new(cart2polar.outputs["phi"], compute_y_lm.inputs["phi"])
        left += 1

        complex_analyser = make_function(tree, name="Analyser", hide=True, functions={
            "re": "re",
            "im": "im",
            "absolute": "re,re,*,im,im,*,+,sqrt",
            "phase": "im,re,atan2"
        }, inputs=["re", "im"], outputs=["re", "im", "absolute", "phase"], scalars=["re", "im", "absolute", "phase"],
                                         location=(left, -2))

        tree.links.new(compute_y_lm.outputs["re"], complex_analyser.inputs["re"])
        tree.links.new(compute_y_lm.outputs["im"], complex_analyser.inputs["im"])
        left += 1

        if self.m >= 0:
            selection = "re"
        else:
            selection = "im"

        # transform position
        vals = ["re", "im", "absolute", "phase"]
        trafo = make_function(tree, name="Transformation", hide=True, functions={
            "position": "position," + selection + ",abs,lambda,lambda,0,>,*,*,1,lambda,lambda,0,>,*,-,+,scale"
            # function is only active for lambda>0
        }, inputs=["lambda", "position"] + vals, outputs=["position"],
                              vectors=["position"],
                              scalars=["lambda"] + vals, location=(left, -2))

        tree.links.new(sphere_position.std_out, trafo.inputs["position"])
        for val in vals:
            tree.links.new(complex_analyser.outputs[val], trafo.inputs[val])
        tree.links.new(lambda_node.std_out, trafo.inputs["lambda"])
        left += 1
        join_full = JoinGeometry(tree, location=(left, 0))
        left += 1
        alpha_attr = StoredNamedAttribute(tree, location=(left, 0), name="Alpha", value=lambda_node.std_out)
        left += 1
        set_pos = SetPosition(tree, location=(left, 0), position=trafo.outputs["position"])
        left += 1
        # store the phase for coloring
        attr = StoredNamedAttribute(tree, location=(sphere_node_pos + 1, 1), name="Phase",
                                    value=complex_analyser.outputs["phase"])
        material = phase2hue_material(attribute_names=["Phase"], alpha_function={"Alpha": "alpha"}, **self.kwargs)
        self.materials.append(material)

        color = SetMaterial(tree, location=(sphere_node_pos + 2, 1), material=material)

        smooth = SetShadeSmooth(tree, location=(left, 0))

        create_geometry_line(tree, [sphere, attr, color, join_full, alpha_attr, set_pos, smooth],
                             out=self.group_outputs.inputs[0])
        create_geometry_line(tree, [join, join_full])

class NodeFromCollection(GeometryNodesModifier):
    def __init__(self, name='NodeFromCollection', collection="Collection", translation=Vector(), rotation=Vector(),
                 scale=Vector([1, 1, 1]), **kwargs):
        """
        create object from collection, which can be rotated
        """

        self.kwargs = kwargs
        self.collection = collection
        self.translation = translation
        self.rotation = rotation
        self.scale = scale

        super().__init__(name, automatic_layout=True)

    def create_node(self, tree):
        out = tree.nodes.get("Group Output")
        links = tree.links

        translation = InputVector(tree, value=self.translation, name="Translation")
        rotation = InputVector(tree, value=self.rotation, name="Rotation")
        scale = InputVector(tree, value=self.scale, name="Scale")
        collectionInfo = CollectionInfo(tree, collection_name=self.collection)
        trafo = TransformGeometry(tree, rotation=rotation.std_out, translation=translation.std_out, scale=scale.std_out)

        create_geometry_line(tree, [collectionInfo,
                                    trafo
                                    ], out=self.group_outputs.inputs[0])

class SliderModifier(GeometryNodesModifier):
    def __init__(self, name='SliderModifier', **kwargs):
        """
        create a wireframe for a given geometry
        """

        self.kwargs = kwargs
        self.label_position = Vector()
        super().__init__(name, group_input=False, automatic_layout=False)

    def create_node(self, tree):
        out = self.group_outputs
        links = tree.links

        left = -10
        slider_value = InputValue(tree, location=(left, -1), name="SliderValue")

        # dimensions
        dimensions = get_from_kwargs(self.kwargs, "dimensions", [0.25, 0.25, 2])
        scale = InputVector(tree, location=(left, -1.5), value=Vector(dimensions), name="Scale")

        #orientation
        orientation = get_from_kwargs(self.kwargs, "orientation", "HORIZONTAL")
        # position
        position = get_from_kwargs(self.kwargs, "position", [0, 0, 0])
        pos = InputVector(tree, location=(left, -2), value=Vector(position), name="Position")
        # shape
        shape = get_from_kwargs(self.kwargs, "shape", "cylinder")
        domain = get_from_kwargs(self.kwargs,"domain",[-1,1])
        side_segments = get_from_kwargs(self.kwargs,"side_segments",2)
        min = max(domain[0],0)

        delta = domain[1]-domain[0]

        if shape == 'cubic':
            geometry = CubeMesh(tree, size=dimensions, location=(left, 0))
        else:
            geometry = CylinderMesh(tree, radius=1, vertices=16, depth=2, fill_type='NGON', side_segments=side_segments,
                                    location=(left, 0))
            left += 1
            l = dimensions[2]
            if orientation == "HORIZONTAL":
                rotation = [0, pi / 2, 0]
                p0 = "pos_x,"+str(min)+",2,*,"+  str(domain[1]) + ",-,"+str(domain[0])+",-," + str(delta) + ",/,s_z,*,+"
                self.label_position = Vector(position) - Vector([1.1 * l, 0, 0])
                position_function = [p0+",value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,not,*,value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,*,+,s_z,*,+", "pos_y", "pos_z"]
                scale_function = ["s_x", "s_y", "value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,not,*,value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,*,+,s_z,*"]
            else:
                p0 = "pos_z," +str(min)+",2,*,"+  str(domain[1]) + ",-,"+str(domain[0])+",-," + str(delta) + ",/,s_z,*,+"
                rotation = [0, 0, 0]
                self.label_position = Vector(position) - Vector([0, 0, 1.1 * l])
                position_function = ["pos_x", "pos_y", p0+",value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,not,*,value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,*,+,s_z,*,+"]
                scale_function = ["s_x", "s_y", "value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,not,*,value,"+str(min)+",-,"+str(delta)+",/,value,"+str(min)+",<,*,+,s_z,*"]
        left += 1

        slider_trafo = make_function(tree, location=(left, -1),
                                     functions={
                                         "position": position_function,
                                         "scale": scale_function,
                                     },
                                     inputs=["value", "pos", "s"], outputs=["position", "scale"],
                                     scalars=["value"], vectors=["position", "scale", "pos", "s"])

        links.new(slider_value.std_out, slider_trafo.inputs["value"])
        links.new(pos.std_out, slider_trafo.inputs["pos"])
        links.new(scale.std_out, slider_trafo.inputs["s"])

        left += 1
        transformation = TransformGeometry(tree, location=(left, 0), translation=pos.std_out, rotation=rotation,
                                           scale=scale.std_out)
        inside_transformation = TransformGeometry(tree, location=(left, -1),
                                                  translation=slider_trafo.outputs["position"], rotation=rotation,
                                                  scale=slider_trafo.outputs["scale"])
        left += 1
        wireframe = WireFrame(tree, radius=0.0025, location=(left, 0))
        left += 1
        grid_material = SetMaterial(tree, location=(left, 0), material='text', emission=1)
        left += 1
        sub_div = SubdivideMesh(tree, level=5, location=(left, 0))

        color = get_from_kwargs(self.kwargs,'color',None)
        if color:
            material = get_material(color,**self.kwargs)
            self.materials.append(material)
        else:
            material = double_gradient(functions={"uv": ["uv_x,0.5,-,2,*", "uv_y,0.5,-,2,*", "uv_z,0.5,-,2,*"],
                                                       "abs_uv": ["uv_x,0.5,-,2,*,abs", "uv_y,0.5,-,2,*,abs",
                                                                  "uv_z,0.5,-,2,*,abs"]})
            self.materials.append(material)
        material_node = SetMaterial(tree, location=(left, -1), material=material)
        pos = Position(tree, location=(left, -2))
        growth = InputValue(tree, location=(left, -3), value=-l, name="Growth")
        left += 1
        join = JoinGeometry(tree, location=(left, 0))

        if orientation == "HORIZONTAL":
            sel_pos = "pos_x"
        else:
            sel_pos = "pos_z"
        sel_fcn = make_function(tree, location=(left, -1), name="GrowthSelector",
                                functions={"selection": sel_pos + ",growth,>"
                                           }, inputs=["pos", "growth"], outputs=["selection"],
                                scalars=["growth", "selection"], vectors=["pos"])
        links.new(pos.std_out, sel_fcn.inputs["pos"])
        links.new(growth.std_out, sel_fcn.inputs["growth"])
        left += 1

        del_geo = DeleteGeometry(tree, location=(left, 0), selection=sel_fcn.outputs["selection"])

        create_geometry_line(tree, [geometry, inside_transformation, material_node, join])
        create_geometry_line(tree, [geometry, transformation, wireframe, grid_material, sub_div, join, del_geo], out=out.inputs[0])

class NumberLineModifier(GeometryNodesModifier):
    def __init__(self, name='NumberLineModifier', **kwargs):
        """
        create geometry for the number line
        """

        super().__init__(name, group_input=False, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        # input parameters
        domain = get_from_kwargs(kwargs, 'domain', [0, 10])
        tic_labels = get_from_kwargs(kwargs, 'tic_labels', 'AUTO')
        tic_label_digits = get_from_kwargs(kwargs, 'tic_label_digits', 1)
        tic_label_shift = get_from_kwargs(kwargs, 'tic_label_shift', [0,0,0])
        max_length = get_from_kwargs(kwargs, 'length', 5)
        radius = get_from_kwargs(kwargs, 'radius', 0.025)
        tip_length = get_from_kwargs(kwargs, 'tip_length', 0.1)
        n_tics = get_from_kwargs(kwargs, 'n_tics', 5)
        direction = get_from_kwargs(kwargs,'direction','HORIZONTAL')
        axis_label=get_from_kwargs(kwargs,'axis_label',"x")
        axis_label_location=get_from_kwargs(kwargs,'axis_label_location','AUTO')

        if direction =="VERTICAL":
            global_rotation = Vector()
        elif direction == "HORIZONTAL":
            global_rotation=Vector([0,pi/2,0])
        elif direction == "DEEP":
            global_rotation=Vector([-pi/2,0,0])
        else:
            global_rotation = get_from_kwargs(kwargs,"rotation_euler",[0,0,0])

        # setup rotation
        # the labels have to be rotated with inverse rotation to stay in the readable orientation
        rotation_frame=Frame(tree,name="Rotations")
        left = -15
        in_rotation = InputRotation(tree,location=(left,5),rotation=global_rotation,name="GlobalRotation")
        inv_rotation = InvertRotation(tree,location=(left+1,4.75),in_rotation=in_rotation.std_out)

        label_rotation0 = InputRotation(tree,location=(left,4.5),rotation=[pi/2,0,0],name="InitialLabelRotation")
        label_rotation = RotateRotation(tree,location=(left+2,4),rotation=label_rotation0.std_out,rotate_by=inv_rotation.std_out)

        rotation_frame.add([in_rotation,inv_rotation,label_rotation,label_rotation0])

        if tic_labels == 'AUTO':
            tic_labels = {}
            # create tic_labels
            x0 = domain[0]
            dx = (domain[1] - domain[0]) / n_tics
            p = 10 ** tic_label_digits  # power for rounding
            for i in range(0, n_tics + 1):
                if p == 1:
                    rounded_val = str(int(round(x0+dx * i)))
                else:
                    rounded_val = str(round((x0+dx * i) * p) / p)
                tic_labels[rounded_val] = [x0+dx * i]

        tic_labels, axis_label = generate_labels(tic_labels,axis_label, **kwargs)
        downshift=-5
        in_min = InputValue(tree,location=(left,downshift), name='Minimum', value=domain[0])
        in_max = InputValue(tree,location=(left,downshift+0.25), name='Maximum', value=domain[1])
        in_length = InputValue(tree,location=(left,downshift+0.5), name='AxisLength', value=max_length)
        in_radius = InputValue(tree,location=(left,downshift+0.75), name='Radius', value=radius)
        in_log = InputValue(tree,location=(left,downshift+1),name='Log',value=0)
        in_label_scale = InputValue(tree,location=(left,downshift+1),name='LabelScale',value=0)

        # create geometry
        # axis
        left+=2
        cyl = CylinderMesh(tree,location=(left,0), depth=in_length.std_out, radius=in_radius.std_out,name="Axis")
        translation = make_function(tree,location=(left-1,1) ,name="AxisTranslation",
                                    functions={"translation": ["0", "0", "l,2,/"]},
                                    inputs=['l'], outputs=["translation"],
                                    scalars=['l'], vectors=["translation"],
                                    )
        links.new(in_length.std_out, translation.inputs['l'])
        cyl_transformation = TransformGeometry(tree,location=(left+1,0), translation=translation.outputs["translation"])

        # tip
        tip_scaling = make_function(tree,location=(left-1,-1), name="TipScaling",
                                    functions={
                                        "tipLength": "l," + str(tip_length) + ",*",
                                        "tipRadius": "r,2,*"
                                    },
                                    inputs=["r", "l"],
                                    outputs=["tipLength", "tipRadius"],
                                    scalars=["r", "l"] + ["tipLength", "tipRadius"])
        for node, label in zip([in_radius, in_length], ["r", "l"]):
            links.new(node.std_out, tip_scaling.inputs[label])

        cone = ConeMesh(tree,location=(left,-0.5), radius_bottom=tip_scaling.outputs["tipRadius"],
                        depth=tip_scaling.outputs["tipLength"],name="Tip")
        cone_transformation = TransformGeometry(tree,location=(left+1,-0.5), translation_z=in_length)
        left+=3

        # tics
        # increase by one to include tic at zero
        # the tic position is given by x0+(i+1-includeZero)*(x1-x0)/n mapped by f**(-1)
        # this position is stored in the coordinate of the point associated with the tic

        tic_function = make_function(tree, location=(left-2,0),name="ticsFunction",
                                     functions={
                                                "depth": "l,500,/",
                                                "r": "r,3,*",
                                                },
                                     inputs=[ "l", "r"],
                                     outputs=[ "r", "depth"],
                                     scalars=[ "l", "r", "depth"])

        links.new(in_length.std_out, tic_function.inputs["l"])
        links.new(in_radius.std_out, tic_function.inputs["r"])
        left+=1

        # join parts
        join = JoinGeometry(tree,location=(left+4,0))

        label_frame =  Frame(tree,name="LabelFrame")
        # add tic_labels
        count=0
        shift=InputVector(tree,location=(left-3,downshift+0.5),name="LabelShift",value=tic_label_shift)
        for key, val in tic_labels.items():
            coll_info = CollectionInfo(tree,location=(left-3,downshift-0.5*count), collection_name=str(key), name=str(key))
            value = InputValue(tree,location=(left-3,downshift-0.5*(count+1/2)), name="LabelValue=" + str(val[0]), value=val[0])
            label_frame.add(value)
            transform_geo = TransformGeometry(tree,location=(left-2,downshift+0.5+0.25*count), rotation=label_rotation.std_out, translation=[0,0,0])
            label_frame.add(transform_geo)
            tic_label_function = make_function(tree,location=(left-1,downshift+0.25*count), name="LabelPositionVisibility" + str(key),
                                               functions={
                                                   "position_tic":["0","0", "val,x0,-,x1,x0,-,/,1,lambda,-,*,val,lg,x0,lg,-,x1,lg,x0,lg,-,/,lambda,*,+," + str(max_length) + ",*"],
                                                   "position_label":["shift_x","shift_y", "shift_z,val,x0,-,x1,x0,-,/,1,lambda,-,*,val,lg,x0,lg,-,x1,lg,x0,lg,-,/,lambda,*,+," + str(max_length) + ",*,+"],
                                                   "invisible": "l,val,x0,-,x1,x0,-,/," + str(max_length) + ",*,<"
                                               }, inputs=["shift","val", "x0", "x1", "l","lambda"], outputs=["position_tic","position_label", "invisible"],
                                               scalars=["val", "x0", "x1", "invisible", "l","lambda"], vectors=["shift","position_tic","position_label"])

            links.new(in_length.std_out, tic_label_function.inputs["l"])
            links.new(in_log.std_out, tic_label_function.inputs["lambda"])
            links.new(in_min.std_out, tic_label_function.inputs["x0"])
            links.new(in_max.std_out, tic_label_function.inputs["x1"])
            links.new(value.std_out, tic_label_function.inputs["val"])
            links.new(shift.std_out,tic_label_function.inputs["shift"])
            label_frame.add(tic_label_function)
            tic_mesh = CylinderMesh(tree, location=(left, downshift+0.25*count+3), vertices=16, radius=tic_function.outputs["r"],depth=tic_function.outputs['depth'])
            set_tic_pos = TransformGeometry(tree, location=(left + 1,  downshift+0.25*count+3),translation=tic_label_function.outputs["position_tic"])
            local_join = JoinGeometry(tree,location=(left+2, downshift+0.25*count+1.5))
            del_label_geo = DeleteGeometry(tree, location=(left+3,downshift+0.25*count),selection=tic_label_function.outputs["invisible"])
            label_frame.add(del_label_geo)
            create_geometry_line(tree,[tic_mesh, set_tic_pos, local_join])
            set_label_pos = SetPosition(tree, location=(left+1, downshift + 0.25 * count),
                                        position=tic_label_function.outputs["position_label"],
                                        offset=Vector([12 * radius, 0, 0]))
            label_frame.add(set_label_pos)
            create_geometry_line(tree, [coll_info, transform_geo,  set_label_pos, local_join,del_label_geo,join])
            count+=1

        # take care of the axis label
        if axis_label_location=='AUTO':
            axis_label_location=[2*radius,0,max_length]

        if axis_label:
            label_location=InputVector(tree,location=(left-3,downshift+0.25*count+2),name="LabelLocation",value=axis_label_location)
            label_info = CollectionInfo(tree, location=(left - 3, downshift+0.25*count+3), collection_name=axis_label,
                                       name="AxisLabel")
        else:
            label_info=None
        if label_info:
            label_trafo= TransformGeometry(tree, location=(left - 2, downshift + 0.5 + 0.25 * count),
                                              rotation=label_rotation.std_out, translation=label_location.std_out,scale=in_label_scale.std_out)
            create_geometry_line(tree,[label_info,label_trafo,join])

        # finalize
        left+=5
        global_transformation = TransformGeometry(tree, location=(left,0),rotation=in_rotation.std_out, name="GlobalTransformation")
        left += 1
        pre_material = get_from_kwargs(kwargs, 'color', 'drawing')
        material = get_material(pre_material, **kwargs)
        self.materials.append(material)
        set_material = SetMaterial(tree, location=(left, 0), material=material)
        left += 1
        auto_smooth = get_from_kwargs(kwargs, 'auto_smooth', True)
        if auto_smooth:
            smooth = SetShadeSmooth(tree,location=(left,0))
            main_line = [cyl, cyl_transformation, join, global_transformation, set_material, smooth]
        else:
            main_line = [cyl, cyl_transformation, join, global_transformation, set_material]

        create_geometry_line(tree, [cone, cone_transformation, join])
        create_geometry_line(tree, main_line, out=out.inputs[0])

class DataModifier(GeometryNodesModifier):
    def __init__(self, name='DataModifier', **kwargs):
        """
        create geometry to display a point cloud of data points
        """

        super().__init__(name, group_input=True, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        ins = self.group_inputs
        links = tree.links

        # input parameters
        x_domain = get_from_kwargs(kwargs, 'x_domain', [0, 10])
        y_domain = get_from_kwargs(kwargs, 'y_domain', [0, 10])
        width = get_from_kwargs(kwargs,"width",10)
        height= get_from_kwargs(kwargs,"height",7)

        pointsize = get_from_kwargs(kwargs, "pointsize", None)
        linesize = get_from_kwargs(kwargs, "linesize", None)
        if pointsize:
            in_pointsize = InputValue(tree,name='PointSize',value=0.05*pointsize)
        else:
            in_pointsize=None

        in_width = InputValue(tree, name='Width', value=width)
        in_height = InputValue(tree, name='Height', value=height)
        in_x_min = InputValue(tree, name='X0', value=x_domain[0])
        in_x_max = InputValue(tree, name='X1', value=x_domain[1])
        in_y_min = InputValue(tree, name='Y0', value=y_domain[0])
        in_y_max = InputValue(tree, name='Y1', value=y_domain[1])
        in_time = SceneTime(tree,std_out="Seconds")
        in_log = InputValue(tree,name='Log',value=0)

        in_time_min = InputValue(tree,name='T0',value=0)
        in_time_max = InputValue(tree,name='T1',value=DEFAULT_ANIMATION_TIME)
        in_pos = Position(tree)

        # this function takes care of
        # * positioning the data points
        # * deleting points outside the view window
        # * deleting points outside the display time
        position_function=make_function(tree,name="DataDisplayFunction",
                    functions={
                        "position":["pos_x,x0,-,x1,x0,-,/,1,l,-,*,pos_x,x0,/,lg,x1,x0,/,lg,/,l,*,+,w,*","0","pos_z,y0,-,y1,y0,-,/,h,*"],
                        "invisible":["pos_x,x0,<,pos_x,x1,>,or,pos_z,y0,<,or,pos_z,y1,>,or,t,t0,<,or,pos_x,x0,-,x1,x0,-,/,t,t0,-,t1,t0,-,/,>,or"]
                    },
                    inputs=["pos","x0","x1","y0","y1","w","h","t0","t1","t","l"],outputs=["position","invisible"],
                    scalars=["x0","x1","y0","y1","w","h","invisible","t0","t1","t","l"],vectors=["pos","position"])
        links.new(in_pos.std_out,position_function.inputs["pos"])
        links.new(in_width.std_out,position_function.inputs["w"])
        links.new(in_height.std_out,position_function.inputs["h"])
        links.new(in_x_min.std_out,position_function.inputs["x0"])
        links.new(in_x_max.std_out,position_function.inputs["x1"])
        links.new(in_y_min.std_out, position_function.inputs["y0"])
        links.new(in_y_max.std_out, position_function.inputs["y1"])
        links.new(in_time_min.std_out, position_function.inputs["t0"])
        links.new(in_time_max.std_out, position_function.inputs["t1"])
        links.new(in_time.std_out,position_function.inputs["t"])
        links.new(in_log.std_out,position_function.inputs["l"])

        del_geo = DeleteGeometry(tree,selection=position_function.outputs["invisible"])
        set_pos = SetPosition(tree,position=position_function.outputs["position"])
        if pointsize:
            point = IcoSphere(tree,subdivisions=2,radius=in_pointsize.std_out)
            iop = InstanceOnPoints(tree,instance=point.geometry_out)
        if linesize:
            wireframe = WireFrame(tree,radius=0.02*linesize,**kwargs)

        mat = get_from_kwargs(kwargs,"material",None)
        if mat:
            mat = get_material(mat,**kwargs)
            self.materials.append(mat)
        set_material = SetMaterial(tree,material=mat)
        if pointsize:
            create_geometry_line(tree, [del_geo,set_pos,iop,set_material], ins = ins.outputs[0],out=out.inputs[0])
        if linesize:
            create_geometry_line(tree, [del_geo,set_pos,wireframe,set_material], ins = ins.outputs[0],out=out.inputs[0])

class LegendrePolynomials(GeometryNodesModifier):
    def __init__(self,l_range=range(10), name='LegendrePolynomials', **kwargs):
        """
        geometry nodes that turn a set of mesh lines into Legendre Polynomials
        """
        self.l_range = l_range
        if "begin_time" in kwargs:
            self.begin_time = kwargs.pop("begin_time")
        else:
            self.begin_time = 0
        if "transition_time" in kwargs:
            self.transition_time = kwargs.pop("transition_time")
        else:
            self.transition_time = 0

        super().__init__(name, group_input=False, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        transform=TransformGeometry(tree,scale=[7]*3)
        join = JoinGeometry(tree)
        time = SceneTime(tree)
        dt = self.transition_time/len(list(self.l_range))

        colors = ["drawing","joker","important","custom1","custom2","custom3","custom4","gray_4"]
        count = 0
        for l in self.l_range:
            mesh_line = MeshLine(tree,count=l*20+10,start_location=[-1,0,0],end_location=[1,0,0])
            position = Position(tree)
            separate = SeparateXYZ(tree,vector=position.std_out)
            pl = AssociatedLegendrePolynomial(tree,l=l,x=separate.x)
            appear_function = make_function(tree, name="AppearFunction",
                                            functions={
                                                "selection": "t," + str(self.begin_time) + ",-," + str(dt) + ",/,"+str(count)+",<"
                                            },
                                inputs=["t"],outputs=["selection"],scalars=["t","selection"])
            links.new(time.std_out,appear_function.inputs["t"])
            combine = CombineXYZ(tree,x=separate.x,y=0,z=pl.std_out)
            set_pos = SetPosition(tree,position=combine.std_out)
            wireframe = WireFrame(tree,radius=0.0025)
            del_geo = DeleteGeometry(tree,selection=appear_function.outputs["selection"])

            if count<len(colors):
                color=colors[count]
            else:
                color=colors[-1]
            set_mat = SetMaterial(tree,material=color)
            create_geometry_line(tree,[mesh_line,set_pos,del_geo,wireframe,set_mat,join])
            count+=1

        create_geometry_line(tree,[join,transform],out=out.inputs[0])

class AssociatedLegendreP(GeometryNodesModifier):
    def __init__(self,l_range=range(0,5),m=1, name='AssociatedLegendrePolynomials', **kwargs):
        """
        geometry nodes that turn a set of mesh lines into associated Legendre Polynomials
        """
        self.m=m
        self.l_range=l_range
        if begin_time :=kwargs.pop("begin_time"):
            self.begin_time=begin_time
        else:
            self.begin_time=0
        if transition_time :=kwargs.pop("transition_time"):
            self.transition_time=transition_time
        else:
            self.transition_time=0
        if 'scale' in kwargs:
            self.scale=kwargs.pop('scale')
        else:
            self.scale = [1]*3

        super().__init__(name, group_input=False, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        join = JoinGeometry(tree)
        time = SceneTime(tree)
        dt = self.transition_time / len(list(self.l_range))

        colors = ["drawing", "joker", "important", "custom1", "custom2", "custom3", "custom4", "gray_4"]
        count = 0
        for l in self.l_range:
            mesh_line = MeshLine(tree, count=l * 100 + 10, start_location=[-0.999, 0, 0], end_location=[0.999, 0, 0]) # avoid numerical singularity at \pm 1
            position = Position(tree)
            y = make_function(tree, name="y",
                              functions={
                                  "y": "x,acos,sin"
                              }, inputs=["x"], outputs=["y"],
                              scalars=["x", "y"])

            separate = SeparateXYZ(tree, vector=position.std_out)
            links.new(separate.x, y.inputs["x"])
            alp = AssociatedLegendrePolynomial(tree, l=l, m=self.m, x=separate.x, y=y.outputs[0])

            appear_function = make_function(tree, name="AppearFunction",
                                            functions={
                                                "selection": "t," + str(self.begin_time) + ",-," + str(
                                                    dt) + ",/," + str(count) + ",<"
                                            },
                                            inputs=["t"], outputs=["selection"], scalars=["t", "selection"])
            links.new(time.std_out, appear_function.inputs["t"])
            combine = CombineXYZ(tree, x=separate.x, y=0, z=alp.std_out)
            set_pos = SetPosition(tree, position=combine.std_out)
            transform = TransformGeometry(tree, scale=self.scale)
            wireframe = WireFrame(tree, radius=0.02)
            del_geo = DeleteGeometry(tree, selection=appear_function.outputs["selection"])

            if count < len(colors):
                color = colors[count]
            else:
                color = colors[-1]
            set_mat = SetMaterial(tree, material=color)
            create_geometry_line(tree, [mesh_line, set_pos, del_geo, transform,wireframe, set_mat, join])
            count += 1

        create_geometry_line(tree, [join], out=out.inputs[0])

class PlmSurface(GeometryNodesModifier):
    def __init__(self, domain = [[-1,1],[0,1]], l=6,m=3, name='AssociatedLegendrePolynomialSurface',
                 begin_time=0,transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """
        geometry nodes that turn a set of mesh lines into Legendre Polynomials
        """
        self.domain = domain
        self.l = l
        self.m = m
        if "begin_time" in kwargs:
            self.begin_time=kwargs.pop("begin_time")
        else:
            self.begin_time =0
        if "transition_time" in kwargs:
            self.transition_time=kwargs.pop("transition_time")
        else:
            self.transition_time = 0

        super().__init__(name, group_input=False, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        size_x = self.domain[0][1]-self.domain[0][0]
        size_y = self.domain[1][1]-self.domain[1][0]
        mesh = Grid(tree,size_x=size_x,size_y=size_y,vertices_y=100,vertices_x=100)
        translation_x=(self.domain[0][1]+self.domain[0][0])/2
        translation_y=(self.domain[1][1]+self.domain[1][0])/2
        # move mesh to the coordinates of its middle
        transform = TransformGeometry(tree,translation_x=translation_x,translation_y=translation_y)
        position = Position(tree)
        sep = SeparateXYZ(tree,vector=position.std_out)
        function  = AssociatedLegendrePolynomial(tree,l=self.l,m=self.m,x=sep.x,y=sep.y)
        combine = CombineXYZ(tree,x=sep.x,y=sep.y,z=function.std_out)
        set_pos = SetPosition(tree,position=combine.std_out)
        if 'scale' in kwargs:
            scale = kwargs.pop('scale')
            transform2 = TransformGeometry(tree,scale=scale)
        else:
            transform2 = TransformGeometry(tree)
        wireframe=WireFrame(tree)

        create_geometry_line(tree, [mesh,transform,set_pos,transform2,wireframe], out=out.inputs[0])

class YlmSurface(GeometryNodesModifier):
    def __init__(self, domain = [[-pi,pi],[0,pi]], l=3,m=2, name="SphericalHarmonicsSurface",
                  **kwargs):
        """
        geometry nodes that turn a grid into a representation of a spherical harmonics function
        """
        self.domain = domain
        self.l = l
        self.m = m
        if "begin_time" in kwargs:
            self.begin_time=kwargs.pop("begin_time")
        else:
            self.begin_time =0
        if "transition_time" in kwargs:
            self.transition_time=kwargs.pop("transition_time")
        else:
            self.transition_time = 0
        self.thickness = get_from_kwargs(kwargs,"thickness",1)

        super().__init__(name, group_input=False, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        size_x = self.domain[0][1]-self.domain[0][0]
        size_y = self.domain[1][1]-self.domain[1][0]
        mesh = Grid(tree,size_x=size_x,size_y=size_y,vertices_y=self.l*10+20,vertices_x=self.l*10+20)

        translation_x=(self.domain[0][1]+self.domain[0][0])/2
        translation_y=(self.domain[1][1]+self.domain[1][0])/2
        # move mesh to the coordinates of its middle
        transform = TransformGeometry(tree,translation_x=translation_x,translation_y=translation_y)
        position = Position(tree)
        sep = SeparateXYZ(tree,vector=position.std_out)
        function  = SphericalHarmonicsRekursive(tree,l=self.l,m=self.m,phi=sep.x,theta=sep.y)
        combine = CombineXYZ(tree,x=sep.x,y=sep.y,z=function.re)
        set_pos = SetPosition(tree,position=combine.std_out)
        if 'scale' in kwargs:
            scale = kwargs.pop('scale')
            transform2 = TransformGeometry(tree,scale=scale)
        else:
            transform2 = TransformGeometry(tree)
        wireframe=WireFrame(tree,radius=0.02*self.thickness)

        create_geometry_line(tree, [mesh,transform,set_pos,transform2,wireframe], out=out.inputs[0])

class YlmSurface_200(GeometryNodesModifier):
    def __init__(self, domain = [[-pi,pi],[0,pi]], l=3,m=2, name="SphericalHarmonicsSurface",
                  **kwargs):
        """
        geometry nodes that turn a grid into a representation of a spherical harmonics function
        """
        self.domain = domain
        self.l = l
        self.m = m
        if "begin_time" in kwargs:
            self.begin_time=kwargs.pop("begin_time")
        else:
            self.begin_time =0
        if "transition_time" in kwargs:
            self.transition_time=kwargs.pop("transition_time")
        else:
            self.transition_time = 0
        self.thickness = get_from_kwargs(kwargs,"thickness",1)

        super().__init__(name, group_input=False, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        size_x = self.domain[0][1]-self.domain[0][0]
        size_y = self.domain[1][1]-self.domain[1][0]
        mesh = Grid(tree,size_x=size_x,size_y=size_y,vertices_y=self.l*10+20,vertices_x=self.l*10+20)

        translation_x=(self.domain[0][1]+self.domain[0][0])/2
        translation_y=(self.domain[1][1]+self.domain[1][0])/2
        # move mesh to the coordinates of its middle
        transform = TransformGeometry(tree,translation_x=translation_x,translation_y=translation_y)
        position = Position(tree)
        sep = SeparateXYZ(tree,vector=position.std_out)
        function  = SphericalHarmonics200(tree,m=self.m,phi=sep.x,theta=sep.y)
        combine = CombineXYZ(tree,x=sep.x,y=sep.y,z=function.re)
        set_pos = SetPosition(tree,position=combine.std_out)
        if 'scale' in kwargs:
            scale = kwargs.pop('scale')
            transform2 = TransformGeometry(tree,scale=scale)
        else:
            transform2 = TransformGeometry(tree)
        wireframe=WireFrame(tree,radius=0.02*self.thickness)

        create_geometry_line(tree, [mesh,transform,set_pos,transform2,wireframe], out=out.inputs[0])

class YlmSurfaceReference(GeometryNodesModifier):
    def __init__(self, domain = [[-pi,pi],[0,pi]], l=3,m=2, name="SphericalHarmonicsSurfaceReference",
                  **kwargs):
        """
        geometry nodes that turn a grid into a representation of a spherical harmonics function
        """
        self.domain = domain
        self.l = l
        self.m = m
        if "begin_time" in kwargs:
            self.begin_time=kwargs.pop("begin_time")
        else:
            self.begin_time =0
        if "transition_time" in kwargs:
            self.transition_time=kwargs.pop("transition_time")
        else:
            self.transition_time = 0

        super().__init__(name, group_input=False, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        size_x = self.domain[0][1]-self.domain[0][0]
        size_y = self.domain[1][1]-self.domain[1][0]
        mesh = Grid(tree,size_x=size_x,size_y=size_y,vertices_y=self.l*10+20,vertices_x=self.l*10+20)

        translation_x=(self.domain[0][1]+self.domain[0][0])/2
        translation_y=(self.domain[1][1]+self.domain[1][0])/2
        # move mesh to the coordinates of its middle
        transform = TransformGeometry(tree,translation_x=translation_x,translation_y=translation_y)
        position = Position(tree)
        sep = SeparateXYZ(tree,vector=position.std_out)
        # create spherical harmonics terms

        y_lm = SphericalHarmonics(self.l, self.m, "theta", "phi")

        real_part = ExpressionConverter(y_lm.real()).postfix()
        imag_part = ExpressionConverter(y_lm.imag()).postfix()

        if imag_part=='':
            imag_part='0'

        compute_y_lm = make_function(tree, name="Y_lm", functions={
            "re": real_part,
            "im": imag_part
        },
                                     inputs=["theta", "phi"],
                                     outputs=["re", "im"], scalars=["re", "im", "theta", "phi"], hide=True,
        )

        tree.links.new(sep.y, compute_y_lm.inputs["theta"])
        tree.links.new(sep.x, compute_y_lm.inputs["phi"])

        combine = CombineXYZ(tree,x=sep.x,y=sep.y,z=compute_y_lm.outputs["re"])
        set_pos = SetPosition(tree,position=combine.std_out)
        if 'scale' in kwargs:
            scale = kwargs.pop('scale')
            transform2 = TransformGeometry(tree,scale=scale)
        else:
            transform2 = TransformGeometry(tree)
        material = SetMaterial(tree,material=z_gradient)

        create_geometry_line(tree, [mesh,transform,set_pos,transform2,material], out=out.inputs[0])

class LogoModifier(GeometryNodesModifier):
    def __init__(self, name="LogoModifier",
                  **kwargs):
        """

        GeometryNodes representation of the NumberCruncher logo. The data is stored in a point cloud (x,y,r)
        Where (x,y) are the centers of the circles and r is the radius. There are three types of meshes, corresponding to the three different classes of circles

        """
        super().__init__(name, group_input=False, automatic_layout=True, **kwargs)

    def create_node(self, tree, sphere_colors =["important","joker","drawing"],**kwargs):
        out = self.group_outputs
        links = tree.links

        circles = ["RedCircles","GreenCircles","BlueCircles"]
        center_names=["CenterRed","CenterGreen","CenterBlue"]
        theta_names=["ThetaRed","ThetaGreen","ThetaBlue"]
        phi_names=["PhiRed","PhiGreen","PhiBlue"]
        join_geometry = JoinGeometry(tree)
        segments = [64,32,32]
        rings = [32,16,16]
        radii = [1,1,1]
        for i,circle in enumerate(circles):
            object = ibpy.get_obj_from_name(circle)
            object_info = ObjectInfo(tree, object=object)
            pos = Position(tree)
            get_pos_and_scale = make_function(tree,name="PositionAndScale",
                        functions={"position":["pos_x","pos_y","0"],
                                   "scale":["pos_z"]*3
                        },inputs=["pos"],outputs=["position","scale"],
                        scalars=[],vectors=["pos","position","scale"])
            links.new(pos.std_out,get_pos_and_scale.inputs["pos"])
            store_center=StoredNamedAttribute(tree,data_type='FLOAT_VECTOR',name=center_names[i],value=pos.std_out)
            set_pos = SetPosition(tree,position=get_pos_and_scale.outputs["position"])
            uv_sphere=UVSphere(tree,segments=segments[i],rings=rings[i],radius=radii[i])
            transform_geometry = TransformGeometry(tree,geometry=uv_sphere.geometry_out,translation=[0,0,-1])
            iop = InstanceOnPoints(tree,instance=transform_geometry.geometry_out,scale=get_pos_and_scale.outputs["scale"])
            realize_instance = RealizeInstances(tree)

            center_attr = NamedAttribute(tree,name=center_names[i],data_type="FLOAT_VECTOR")
            get_loc_on_sphere = make_function(tree,name="LocalSphereCoords",
                        functions={
                            "r":"pos,center,sub",
                            "length":"pos,center,sub,length"
                        },inputs=["pos","center"],outputs=["r","length"],
                        scalars=["length"],vectors=["pos","center","r"])
            links.new(pos.std_out,get_loc_on_sphere.inputs["pos"])
            links.new(center_attr.std_out,get_loc_on_sphere.inputs["center"])

            get_theta_phi = make_function(tree,name="UVComputation",
                        functions={
                        "theta":"r_z,l,/,acos",
                        "phi":"r_x,r_y,atan2"
                        },inputs=["r","l"],outputs=["theta","phi"],
                        scalars=["l","theta","phi"],vectors=["r"])
            links.new(get_loc_on_sphere.outputs["r"],get_theta_phi.inputs["r"])
            links.new(get_loc_on_sphere.outputs["length"],get_theta_phi.inputs["l"])

            store_theta=StoredNamedAttribute(tree,name=theta_names[i],value=get_theta_phi.outputs["theta"])
            store_phi=StoredNamedAttribute(tree,name=phi_names[i],value=get_theta_phi.outputs["phi"])

            mat=get_material(sphere_colors[i],**kwargs)
            self.materials.append(mat)
            material_node = SetMaterial(tree,material=mat)
            create_geometry_line(tree,green_nodes=[object_info,set_pos,store_center,
                                                   iop,realize_instance,
                                                   store_theta,store_phi,material_node,join_geometry])

        transform_geometry = TransformGeometry(tree,translation=[0,0,-5.5],rotation=[pi/2,0,0],scale=[5.5]*3)
        shade_smooth = SetShadeSmooth(tree)
        create_geometry_line(tree,[join_geometry,transform_geometry,shade_smooth],out=out.inputs["Geometry"])

class VoronoiModifier(GeometryNodesModifier):
    def __init__(self, name="VoronoiModifier",begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,
                  **kwargs):
        """

        GeometryNodes to convert a PreVoronoi mesh into a voronoi mesh.

        """
        self.t0 = begin_time
        self.dt = transition_time

        super().__init__(name, group_input=True, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        ins = self.group_inputs
        links = tree.links
        # show points

        sphere = IcoSphere(tree,radius=0.01)
        iop = InstanceOnPoints(tree,instance = sphere.geometry_out)
        sphere_mat = get_material("plastic_joker")
        self.materials.append(sphere_mat)
        set_mat = SetMaterial(tree,material=sphere_mat)
        join_geometry = JoinGeometry(tree)
        create_geometry_line(tree, [iop, set_mat, join_geometry], ins=ins.outputs["Geometry"])

        # create voronoi mesh
        time = SceneTime(tree)

        iterator = make_function(tree,name="IterationTimer",
                    functions={
                        "iterations":"t,t0,-,dt,/,25,*,t,t0,>,*"
                    },inputs=["t","t0","dt"],outputs=["iterations"],
                    scalars=["t","t0","dt","iterations"])

        links.new(time.std_out,iterator.inputs["t"])
        iterator.inputs["t0"].default_value=self.t0
        iterator.inputs["dt"].default_value=self.dt

        index = Index(tree)
        dual_mesh = DualMesh(tree,keep_boundaries = True)
        store_index = StoredNamedAttribute(tree,data_type="INT",domain="FACE",name="Index",value=index.std_out)
        split_edges = SplitEdges(tree)
        wireframe_material = get_material("gold")
        wireframe=WireFrameRectangle(tree, node_width=0.02, node_height=0.02)
        self.materials.append(wireframe_material)
        set_material_wireframe=SetMaterial(tree,material=wireframe_material)
        scale_elements=ScaleElements(tree,scale=0.99)

        # start fractalizing

        # rotate tilings
        position = Position(tree)
        index_attr = NamedAttribute(tree,data_type="INT",name="Index")
        normal = InputNormal(tree)
        angle = InputValue(tree,name="Angle",value=0.1)

        direction = make_function(tree,name="RotationDirection",
                    functions={
                        "angle":"-1,idx,**,angle,*"
                    },inputs=["angle","idx"],outputs=["angle"],
                    scalars=["angle","idx"],vectors=[])
        links.new(index_attr.std_out,direction.inputs["idx"])
        links.new(angle.std_out,direction.inputs["angle"])

        eval_domain = EvaluateOnDomain(tree,data_type="FLOAT_VECTOR",domain="FACE",value=position.std_out)
        rotate_vector = VectorRotate(tree, rotation_type="AXIS_ANGLE", center=eval_domain.std_out, axis=normal.std_out,
                                     vector=position.std_out,angle = direction.outputs["angle"])

        repeat= RepeatZone(tree,iteration=5,geometry=scale_elements.geometry_out)
        repeat_join = JoinGeometry(tree)
        repeat_scale = ScaleElements(tree,scale=0.9)
        repeat_set_pos = SetPosition(tree,offset=[0,-0.025,0],position=rotate_vector.std_out)
        repeat.create_geometry_line([repeat_scale,repeat_set_pos,repeat_join])
        links.new(scale_elements.geometry_out,repeat_join.geometry_in)
        links.new(iterator.outputs["iterations"],repeat.inputs["Iterations"])

        # after processing
        extrude_mesh = ExtrudeMesh(tree,mode="FACES",offset_scale=1,offset=[0,-0.025,0])
        tile_material = get_material("text",emission=0.5)
        self.materials.append(tile_material)
        set_material_tiles=SetMaterial(tree,material=tile_material)

        local_join = JoinGeometry(tree)
        # create_geometry_line(tree,[split_edges,local_join]) # cover extrude of face on the other side
        create_geometry_line(tree,[dual_mesh,split_edges,store_index,scale_elements,repeat],ins=ins.outputs["Geometry"])
        create_geometry_line(tree,[repeat, extrude_mesh,set_material_tiles,local_join,join_geometry],out=out.inputs["Geometry"])
        create_geometry_line(tree,[repeat,wireframe,set_material_wireframe,local_join])

class UnfoldModifier(GeometryNodesModifier):
    def __init__(self, name="UnfoldModifier",**kwargs):
        super().__init__(name, automatic_layout=False,group_output=False,group_input=False, **kwargs)

    def create_node(self,tree,**kwargs):
        create_from_xml(tree,"unfolding_node",**kwargs)

# Rubik's cube videos

class RubiksCubeModifier(GeometryNodesModifier):
    def __init__(self, name="RubiksCubeModifier", **kwargs):
        super().__init__(name, automatic_layout=False, **kwargs)

    def loc(self, block_x=0, block_y=0, x=0, y=0, absolute=False):
        block_width = 7
        block_height = 5
        if absolute:
            return (200 * (block_x * block_width + x), 200 * (block_y * block_height + y))
        else:
            return (block_x * block_width + x, block_y * block_height + y)

    def create_node(self, tree, **kwargs):
        '''
        we need to create the following structure

        labels for cubies (0,5)



        3x3x3 grid (-1,2)

        beveled cube (-1,1)     selected faces (0,1)          selection for animation(1,1)

        beveled face (-1,0)     face coloring  (0,0)          transformation (1,0)            output (2,0)

                                labels for cubies (0,-2)

                                labels for faces (0,-3)
        '''
        # output block
        out = self.group_outputs
        out.location = self.loc(2, 0, 0, 0, absolute=True)
        links = tree.links

        # 3x3x3 block
        separation = InputValue(tree, name="Separation", value=1, location=self.loc(-1, 2, 0, 0), hide=False)
        size = InputValue(tree, name="Cube Size", value=0.89, location=self.loc(-1, 2, 0, -1), hide=False)
        line_function = make_function(tree, name="LineFunction",
                                      functions={
                                          "start": ["0", "0", "-1,separation,*"],
                                          "offset": ["0", "0", "1,separation,*"]
                                      }, inputs=["separation"], outputs=["start", "offset"],
                                      scalars=["separation"], vectors=["start", "offset"],
                                      location=self.loc(-1, 2, 1, 0), hide=False)
        links.new(separation.std_out, line_function.inputs["separation"])
        mesh_line = MeshLine(tree, size=size.std_out, mode='OFFSET', count_mode="TOTAL", count=3,
                             start_location=line_function.outputs["start"],
                             offset=line_function.outputs["offset"], location=self.loc(-1, 2, 2, 0), hide=False
                             )
        grid = Grid(tree, size_x=2, size_y=2, vertices_y=3, vertices_x=3, location=self.loc(-1, 2, 2, -1.5), hide=False)
        iop = InstanceOnPoints(tree, instance=grid.geometry_out, scale=separation.std_out, hide=False,
                               location=self.loc(-1, 2, 3, 0))
        realize_instance = RealizeInstances(tree, location=self.loc(-1, 2, 4, 0), hide=False)
        index = Index(tree, location=self.loc(-1, 2, 4, -1.5), hide=False)
        # increase index by one, since index 0 cannot be selected (bug, probably)
        inc = make_function(tree, name="Function",
                            functions={
                                "idx": "idx,1,+"
                            }, inputs=["idx"], outputs=["idx"],
                            scalars=["idx"])
        links.new(index.std_out, inc.inputs["idx"])
        pos = Position(tree, location=self.loc(-1, 2, 5, -2), hide=False)
        cubie_index = StoredNamedAttribute(tree, domain='POINT', data_type='INT',
                                           name="CubieIndex", value=inc.outputs["idx"], hide=False,
                                           location=self.loc(-1, 2, 5, 0))
        cubie_pos = StoredNamedAttribute(tree, domain='POINT', data_type='FLOAT_VECTOR',
                                         name="CubiePosition", value=pos.std_out, hide=False,
                                         location=self.loc(-1, 2, 6, 0))
        create_geometry_line(tree, [mesh_line, iop, realize_instance, cubie_index, cubie_pos])

        # beveled cube
        bevel = InputValue(tree, name="Bevel", value=6.3, location=self.loc(-1, 1, -0.5, 1), hide=False)
        bevel_function = make_function(tree, name="BevelFunction", location=self.loc(-1, 1, 0, 0),
                                       functions={
                                           "bevel": "size,bevel,/"
                                       }, inputs=["size", "bevel"], outputs=["bevel"],
                                       scalars=["size", "bevel"], hide=False)
        links.new(size.std_out, bevel_function.inputs["size"])
        links.new(bevel.std_out, bevel_function.inputs["bevel"])

        cube = CubeMesh(tree, size=size.std_out, location=self.loc(-1, 1, 1, 0), hide=False)
        cube2 = CubeMesh(tree, size=bevel_function.outputs["bevel"], location=self.loc(-1, 1, 1, -2), hide=False)
        subsurf = SubdivisionSurface(tree, level=3, mesh=cube2.geometry_out, hide=False,
                                     location=self.loc(-1, 1, 2, -2))
        iop2 = InstanceOnPoints(tree, scale=size.std_out, instance=subsurf.geometry_out, hide=False,
                                location=self.loc(-1, 1, 3, 0))
        realize_instance2 = RealizeInstances(tree, location=self.loc(-1, 1, 4, 0), hide=False)
        convex_hull = ConvexHull(tree, location=self.loc(-1, 1, 5, -1), hide=False)

        create_geometry_line(tree, [cube, iop2, realize_instance2, convex_hull])

        # beveled face(-1, 0)

        quadri = Quadrilateral(tree, location=self.loc(-1, 0, 0, 0),
                               width=size.std_out, height=size.std_out, hide=False)
        fillet_curve = FilletCurve(tree, radius=bevel_function.outputs["bevel"], count=10,
                                   location=self.loc(-1, 0, 1, 0), hide=False)
        fill_cuve = FillCurve(tree, mode='NGONS', location=self.loc(-1, 0, 2, 0), hide=False)
        extrude_face = ExtrudeMesh(tree, mode='FACES', offset_scale=0.01, offset=None, location=self.loc(-1, 0, 3, 0),
                                   hide=False)
        create_geometry_line(tree, [quadri, fillet_curve, fill_cuve, extrude_face])

        # selected faces(0, 0)
        block_y = 2.5
        iop3 = InstanceOnPoints(tree, instance=convex_hull.geometry_out, hide=False,
                                location=self.loc(0, block_y, 0, 0))
        realize_cube = RealizeInstances(tree, location=self.loc(0, block_y, 1, 0), hide=False)
        cube_material = SetMaterial(tree, location=self.loc(0, block_y, 2, 0), material="plastic_background")
        shade_smooth = SetShadeSmooth(tree, location=self.loc(0, block_y, 3, 0))
        create_geometry_line(tree, [cubie_pos, iop3, realize_cube, cube_material, shade_smooth])

        face_area = FaceArea(tree, location=self.loc(0, block_y, 0, -3.5), hide=False)
        position2 = Position(tree, location=self.loc(0, block_y, 0, -4.5), hide=False)
        abs = VectorMath(tree, operation="ABSOLUTE", inputs0=position2.std_out, location=self.loc(0, block_y, 1, -4),
                         hide=False)

        # the basic algorithm behind the face selection is the following
        # for faces where the x coordinate is absolutely the largest coordinate, we only select the face
        # when the x>0.5*cubesize+spacing or x<-(0.5*cubesize+spacing)
        # additionally only faces with an area larger than 0.5*cubesize**2 are selected

        outside_faces = make_function(tree, name="OutSideFaces",
                                      functions={
                                          "select": "pos_x,pos_y,>,pos_x,pos_z,>,and,pos_x,0.5,size,*,spc,+,>,and,"
                                                    "pos_y,pos_x,>,pos_y,pos_z,>,and,pos_y,0.5,size,*,spc,+,>,and,or,"
                                                    "pos_z,pos_x,>,pos_z,pos_y,>,and,pos_z,0.5,size,*,spc,+,>,and,or,"
                                                    "area,0.5,size,2,**,*,>,and"
                                      }, inputs=["pos", "area", "size", "spc"], outputs=["select"],
                                      scalars=["area", "select", "size", "spc"], vectors=["pos"], hide=False,
                                      location=self.loc(0, block_y, 1, -1.75))

        links.new(abs.std_out, outside_faces.inputs["pos"])
        links.new(face_area.std_out, outside_faces.inputs["area"])
        links.new(size.std_out, outside_faces.inputs["size"])
        links.new(separation.std_out, outside_faces.inputs["spc"])

        # set cubie faces

        sep_geo = SeparateGeometry(tree, location=self.loc(0, 1, 0, 0), domain='FACE', hide=False,
                                   selection=outside_faces.outputs["select"])
        attr_cubie_index = NamedAttribute(tree, location=self.loc(0, 1, -1, -1), name="CubieIndex",
                                          data_type='INT', hide=False)
        normal = InputNormal(tree, location=self.loc(0, 1, -1, -2), hide=False)
        store_cubie_index_at_face = StoredNamedAttribute(tree, location=self.loc(0, 1, 1, 0), data_type='INT',
                                                         domain='FACE', name="CubieIndexAtFace",
                                                         value=attr_cubie_index.std_out,
                                                         hide=False)
        store_face_normal = StoredNamedAttribute(tree, location=self.loc(0, 1, 2, 0), data_type='FLOAT_VECTOR',
                                                 domain='FACE', name="FaceNormal", value=normal.std_out, hide=False)
        dual_mesh = DualMesh(tree, location=self.loc(0, 1, 3, 0), hide=False)
        position3 = Position(tree, location=self.loc(0, 1, -1, -3), hide=False)
        abs_2 = VectorMath(tree, operation="ABSOLUTE", inputs0=position3.std_out,
                           location=self.loc(0, 1, 0, -3))
        index3 = Index(tree, location=self.loc(0, 1, -1, -5), hide=False)

        # the aim of this function is to re-arrange the indices of the faces,
        # such that a side of the Rubik's cube
        # has nine faces with con secutive index values
        index_function = make_function(tree, name="IndexFunction",
                                       functions={
                                           "weight": "apos_x,apos_y,>,apos_x,apos_z,>,and,10,*,pos_x,sgn,*,"
                                                     "apos_y,apos_x,>,apos_y,apos_z,>,and,5,*,pos_y,sgn,*,+,"
                                                     "apos_z,apos_x,>,apos_z,apos_y,>,and,1,*,pos_z,sgn,*,+,"
                                                     "index,100,/,+"
                                       }, inputs=["index", "pos", "apos"], outputs=["weight"],
                                       scalars=["index", "weight"], vectors=["pos", "apos"],
                                       location=self.loc(0, 1, 3, -1),
                                       hide=False)

        links.new(index3.std_out, index_function.inputs["index"])
        links.new(abs_2.std_out, index_function.inputs["apos"])
        links.new(position3.std_out, index_function.inputs["pos"])

        sort = SortElements(tree, location=self.loc(0, 1, 4, 0), hide=False, domain='POINT',
                            sort_weight=index_function.outputs["weight"])

        attr_face_normal = NamedAttribute(tree, location=self.loc(0, 1, -1, -4),
                                          name="FaceNormal", data_type='FLOAT_VECTOR', hide=False)

        store_face_index = StoredNamedAttribute(tree, location=self.loc(0, 1, 5, 0), hide=False,
                                                domain='POINT', data_type='INT', name="FaceIndex", value=index3.std_out)
        align_rotation = AxesToRotation(tree, location=self.loc(0, 1, 5, -2), hide=False,
                                        primary_axis='Z', secondary_axis='Y',
                                        primary_direction=attr_face_normal.std_out,
                                        secondary_direction=Vector([0, 0, 1]))
        # align_rotation=AlignRotationToVector(tree,location=self.loc(0,1,5,-2),hide=False,
        #                                      vector=attr_face_normal.std_out)
        # pipe in beveled cubie face
        iop4 = InstanceOnPoints(tree, instance=extrude_face.geometry_out, hide=False,
                                rotation=align_rotation.std_out, location=self.loc(0, 1, 6, 0))

        realize_faces = RealizeInstances(tree, location=self.loc(0, 1, 7, 0), hide=False)
        create_geometry_line(tree, [realize_cube, sep_geo, store_cubie_index_at_face,
                                    store_face_normal, dual_mesh, sort, store_face_index, iop4, realize_faces])

        # face coloring
        block_y = -0.25
        block_x = -0.25
        attr_face_index = NamedAttribute(tree, location=self.loc(block_x, block_y, 0, -3), name="FaceIndex", hide=False)

        red_selector = make_function(tree, name="RedSelector", location=self.loc(block_x, block_y, 1, -1.5),
                                     functions={
                                         "select": "face_index,44,>,face_index,54,<,and,switch,and"
                                     }, inputs=["switch", "face_index"], outputs=["select"],
                                     scalars=["switch", "face_index", "select"], hide=False)
        red_selector_switch = InputBoolean(tree, name="RedSelectorSwitch", value=True,
                                           location=self.loc(block_x, block_y, 1, -2.5), hide=False)

        blue_selector = make_function(tree, name="BlueSelector", location=self.loc(block_x, block_y, 2, -1.5),
                                      functions={
                                          "select": "face_index,35,>,face_index,45,<,and,switch,and"
                                      }, inputs=["switch", "face_index"], outputs=["select"],
                                      scalars=["switch", "face_index", "select"], hide=False)
        blue_selector_switch = InputBoolean(tree, name="BlueSelectorSwitch", value=True,
                                            location=self.loc(block_x, block_y, 2, -2.5), hide=False)

        yellow_selector = make_function(tree, name="YellowSelector", location=self.loc(block_x, block_y, 3, -1.5),
                                        functions={
                                            "select": "face_index,17,>,face_index,27,<,and,switch,and"
                                        }, inputs=["switch", "face_index"], outputs=["select"],
                                        scalars=["switch", "face_index", "select"], hide=False)
        yellow_selector_switch = InputBoolean(tree, name="YellowSelectorSwitch", value=True,
                                              location=self.loc(block_x, block_y, 3, -2.5), hide=False)

        orange_selector = make_function(tree, name="OrangeSelector", location=self.loc(block_x, block_y, 4, -1.5),
                                        functions={
                                            "select": "face_index,-1,>,face_index,9,<,and,switch,and"
                                        }, inputs=["switch", "face_index"], outputs=["select"],
                                        scalars=["switch", "face_index", "select"], hide=False)
        orange_selector_switch = InputBoolean(tree, name="OrangeSelectorSwitch", value=True,
                                              location=self.loc(block_x, block_y, 4, -2.5), hide=False)

        green_selector = make_function(tree, name="GreenSelector", location=self.loc(block_x, block_y, 5, -1.5),
                                       functions={
                                           "select": "face_index,8,>,face_index,18,<,and,switch,and"
                                       }, inputs=["switch", "face_index"], outputs=["select"],
                                       scalars=["switch", "face_index", "select"], hide=False)
        green_selector_switch = InputBoolean(tree, name="GreenSelectorSwitch", value=True,
                                             location=self.loc(block_x, block_y, 5, -2.5), hide=False)

        white_selector = make_function(tree, name="WhiteSelector", location=self.loc(block_x, block_y, 6, -1.5),
                                       functions={
                                           "select": "face_index,26,>,face_index,36,<,and,switch,and"
                                       }, inputs=["switch", "face_index"], outputs=["select"],
                                       scalars=["switch", "face_index", "select"], hide=False)
        white_selector_switch = InputBoolean(tree, name="WhiteSelectorSwitch", value=True,
                                             location=self.loc(block_x, block_y, 6, -2.5), hide=False)

        edge_selector = make_function(tree, name="EdgeSelector", location=self.loc(block_x, block_y, 7, -1.5),
                                      functions={
                                          "select": "idx,19,=,idx,21,=,or,idx,23,=,or,"
                                                    "idx,25,=,or,idx,34,=,or,idx,28,=,or,idx,39,=,or,"
                                                    "idx,41,=,or,idx,12,=,or,idx,14,=,or,idx,30,=,or,idx,32,=,or,switch,and"
                                      }, inputs=["switch", "idx"], outputs=["select"],
                                      scalars=["switch", "idx", "select"], hide=False)

        edge_selector_switch = InputBoolean(tree, name="EdgeSelectorSwitch", value=False,
                                            location=self.loc(block_x, block_y, 7, -2.5), hide=False)

        corner_selector = make_function(tree, name="CornerSelector", location=self.loc(block_x, block_y, 8, -1.5),
                                        functions={
                                            "select": "idx,15,=,idx,17,=,or,idx,11,=,or,"
                                                      "idx,9,=,or,idx,44,=,or,idx,42,=,or,idx,38,=,or,"
                                                      "idx,36,=,or,switch,and"
                                        }, inputs=["switch", "idx"], outputs=["select"],
                                        scalars=["switch", "idx", "select"], hide=False)

        corner_selector_switch = InputBoolean(tree, name="CornerSelectorSwitch", value=False,
                                              location=self.loc(block_x, block_y, 8, -2.5), hide=False)

        links.new(attr_face_index.std_out, red_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, blue_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, yellow_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, orange_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, green_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, white_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, edge_selector.inputs["idx"])
        links.new(attr_face_index.std_out, corner_selector.inputs["idx"])
        links.new(red_selector_switch.std_out, red_selector.inputs["switch"])
        links.new(white_selector_switch.std_out, white_selector.inputs["switch"])
        links.new(green_selector_switch.std_out, green_selector.inputs["switch"])
        links.new(blue_selector_switch.std_out, blue_selector.inputs["switch"])
        links.new(orange_selector_switch.std_out, orange_selector.inputs["switch"])
        links.new(yellow_selector_switch.std_out, yellow_selector.inputs["switch"])
        links.new(edge_selector_switch.std_out, edge_selector.inputs["switch"])
        links.new(corner_selector_switch.std_out, corner_selector.inputs["switch"])

        material_black = SetMaterial(tree, location=self.loc(block_x, block_y, 0, 0), material="plastic_gray_1")

        material_red = SetMaterial(tree, location=self.loc(block_x, block_y, 1, 0),
                                   selection=red_selector.outputs["select"], material="plastic_red",
                                   hide=False)
        material_blue = SetMaterial(tree, location=self.loc(block_x, block_y, 2, 0),
                                    selection=blue_selector.outputs["select"], material="plastic_blue",
                                    hide=False)
        material_yellow = SetMaterial(tree, location=self.loc(block_x, block_y, 3, 0),
                                      selection=yellow_selector.outputs["select"], material="plastic_yellow",
                                      hide=False)
        material_orange = SetMaterial(tree, location=self.loc(block_x, block_y, 4, 0),
                                      selection=orange_selector.outputs["select"], material="plastic_orange",
                                      hide=False)
        material_green = SetMaterial(tree, location=self.loc(block_x, block_y, 5, 0),
                                     selection=green_selector.outputs["select"], material="plastic_green",
                                     hide=False)
        material_white = SetMaterial(tree, location=self.loc(block_x, block_y, 6, 0),
                                     selection=white_selector.outputs["select"], material="plastic_text",
                                     hide=False)

        material_edge = SetMaterial(tree, location=self.loc(block_x, block_y, 7, 0),
                                    selection=edge_selector.outputs["select"], material="plastic_joker",
                                    hide=False)

        material_corner = SetMaterial(tree, location=self.loc(block_x, block_y, 8, 0),
                                      selection=corner_selector.outputs["select"],
                                      material="plastic_custom1", hide=False)

        last_material = material_corner
        create_geometry_line(tree, [realize_faces, material_black, material_red, material_blue, material_yellow,
                                    material_orange, material_green, material_white, material_edge, material_corner])

        # selection block
        block_y = 2
        block_x = 1.1
        join = JoinGeometry(tree, location=self.loc(block_x, block_y, 0, 0), hide=False)
        create_geometry_line(tree, [last_material, join])
        create_geometry_line(tree, [shade_smooth, join])
        attr_cubie_index2 = NamedAttribute(tree, location=self.loc(block_x, block_y, 0, -1), name="CubieIndex",
                                           hide=False, data_type="INT")
        attr_face_index2 = NamedAttribute(tree, location=self.loc(block_x, block_y, 0, -2),
                                          name="CubieIndexAtFace", data_type="INT", hide=False)

        # now we need to separate each cubie and prepare its own transformation panel.

        join2 = JoinGeometry(tree, location=self.loc(block_x, block_y, 4, 0), hide=False)

        # the index for the cubies had to be artificially ranged from 1 to 27, since index 0 caused trouble
        # in the geometry selection process. For the rest of the python interface the labels are ranging from 0 to 26 again
        for i in range(1, 28):
            cubie_selection = make_function(tree, name="CubieSelection" + str(i - 1),
                                            location=self.loc(block_x, block_y, 1, -i + 1), hide=True,
                                            functions={
                                                "select": "idx1," + str(i) + ",=,idx2," + str(i) + ",=,or"
                                            }, inputs=["idx1", "idx2"], outputs=["select"],
                                            scalars=["idx1", "idx2", "select"])
            links.new(attr_cubie_index2.std_out, cubie_selection.inputs["idx1"])
            links.new(attr_face_index2.std_out, cubie_selection.inputs["idx2"])

            sep_geo = SeparateGeometry(tree, location=self.loc(block_x, block_y, 2, -i + 1), domain='FACE',
                                       hide=True, selection=cubie_selection.outputs["select"], geometry_out="Selection")

            rotation = QuaternionToRotation(tree, name="CubieRotation_" + str(i - 1),
                                            location=self.loc(block_x, block_y, 2, -i), hide=False)

            transform = TransformGeometry(tree, location=self.loc(block_x, block_y, 3, -i),
                                          rotation=rotation.std_out,
                                          hide=True)
            create_geometry_line(tree, [join, sep_geo, transform, join2], out=out.inputs[0])

        # the labels for the orientation
        block_x = 0
        block_y = 4

        size_function = make_function(tree, name="SizeFunction", location=self.loc(block_x, block_y, -1, 0), hide=True,
                                      functions={"size": "3,size,*"},
                                      inputs=["size"], outputs=["size"], scalars=["size"])

        links.new(separation.std_out, size_function.inputs["size"])

        edge_centers = CubeMesh(tree, location=self.loc(block_x, block_y, 0, 0),
                                vertices_x=4, vertices_y=4, vertices_z=4, size=size_function.outputs["size"])
        normal2 = InputNormal(tree, location=self.loc(block_x, block_y, 1, -1))
        store_face_normal2 = StoredNamedAttribute(tree, location=self.loc(block_x, block_y, 1, 0),
                                                  name="FaceNormal2", value=normal2.std_out, domain='FACE',
                                                  data_type="FLOAT_VECTOR", hide=False)
        dual_mesh = DualMesh(tree, location=self.loc(block_x, block_y, 2, 0), )
        mesh_to_points = MeshToPoints(tree, location=self.loc(block_x, block_y, 3, 0), hide=True)

        # create the selection function, depending on the index of the face
        idx = Index(tree, location=self.loc(block_x, block_y, 1, 1), hide=True)
        edges = InputBoolean(tree, name="EdgeSwitch", location=self.loc(block_x, block_y, 1, 1.5), value=False,
                             hide=True)
        corners = InputBoolean(tree, name="CornerSwitch", location=self.loc(block_x, block_y, 1, 2), value=False,
                               hide=True)
        index_function = make_function(tree, name="index_function",
                                       functions={
                                           "zero": "idx,19,=,idx,25,=,or,idx,21,=,or,idx,23,=,or,idx,1,=,or,idx,3,=,or,"
                                                   "idx,5,=,or,idx,7,=,or,idx,12,=,or,idx,14,=,or,idx,30,=,or,idx,32,=,or,edges,and,"
                                                   "idx,15,=,idx,17,=,or,idx,9,=,or,idx,11,=,or,idx,35,=,or,idx,33,=,or,idx,29,=,or,idx,27,=,or,corners,and,or",
                                           "one": "idx,46,=,idx,48,=,or,idx,50,=,or,idx,52,=,or,idx,37,=,or,idx,39,=,or,idx,41,=,or,idx,43,=,or,"
                                                  "idx,10,=,or,idx,16,=,or,idx,28,=,or,idx,34,=,or,edges,and,"
                                                  "idx,42,=,idx,20,=,or,idx,45,=,or,idx,0,=,or,idx,53,=,or,idx,24,=,or,idx,38,=,or,idx,8,=,or,corners,and,or",
                                           "two": "idx,18,=,idx,51,=,or,idx,2,=,or,idx,36,=,or,idx,26,=,or,idx,44,=,or,idx,6,=,or,idx,47,=,or,corners,and"
                                       }, inputs=["idx", "edges", "corners"], outputs=["zero", "one", "two"],
                                       scalars=["idx", "zero", "one", "two", "edges", "corners"],
                                       vectors=[], location=self.loc(block_x, block_y, 2, 1), hide=True)
        links.new(idx.std_out, index_function.inputs["idx"])
        links.new(edges.std_out, index_function.inputs["edges"])
        links.new(corners.std_out, index_function.inputs["corners"])

        # align font to face normal
        attr_face_normal2 = NamedAttribute(tree, location=self.loc(block_x, block_y, 2, -1), name="FaceNormal2",
                                           data_type="FLOAT_VECTOR", hide=False)
        align_rotation2 = AxesToRotation(tree, location=self.loc(block_x, block_y, 3, -1),
                                         primary_axis='Z', secondary_axis='Y',
                                         primary_direction=attr_face_normal2.std_out,
                                         secondary_direction=Vector([0, 0, 1]), hide=False)

        # create three geometry lines for the numbers 0, 1, 2
        create_geometry_line(tree, [edge_centers, store_face_normal2, dual_mesh, mesh_to_points])
        for i in range(3):
            text = StringToCurves(tree, location=self.loc(block_x, block_y, 2, 4 - i), string=str(i), hide=True)
            iop = InstanceOnPoints(tree, location=self.loc(block_x, block_y, 4, 4 - i), hide=True,
                                   instance=text.geometry_out, rotation=align_rotation2.std_out,
                                   selection=index_function.outputs[i])
            fill_curve = FillCurve(tree, location=self.loc(block_x, block_y, 5, 4 - i), hide=True)
            extrude = ExtrudeMesh(tree, location=self.loc(block_x, block_y, 6, 4 - i), mode='FACES', offset=None,
                                  hide=True, offset_scale=0.03)
            create_geometry_line(tree, [mesh_to_points, iop, fill_curve, extrude, join2])

        # the labels for cubies
        block_x = -0.5
        block_y = -2

        cubie_label = InputBoolean(tree, name="CubieSwitch", location=self.loc(block_x, block_y, 0, 1), value=False,
                                   hide=True)

        for_each_cubie = ForEachZone(tree, location=self.loc(block_x, block_y, 0, -2), hide=True,
                                     geometry=sort.geometry_out)

        cubie_index = make_function(tree, name="CubieIndexFunction",
                                    functions={
                                        "idx": "idx,15,=,idx,6,=,or,idx,27,=,or,1,*,"
                                               "idx,16,=,idx,30,=,or,2,*,+,"
                                               "idx,17,=,idx,33,=,or,idx,51,=,or,3,*,+,"
                                               "idx,3,=,idx,12,=,or,4,*,+,"
                                               "idx,13,=,5,*,+,"
                                               "idx,14,=,idx,48,=,or,6,*,+,"
                                               "idx,0,=,idx,9,=,or,idx,18,=,or,7,*,+,"
                                               "idx,10,=,idx,21,=,or,8,*,+,"
                                               "idx,11,=,idx,24,=,or,idx,45,=,or,9,*,+,"
                                               "idx,28,=,idx,7,=,or,10,*,+,"
                                               "idx,31,=,11,*,+,"
                                               "idx,34,=,idx,52,=,or,12,*,+,"
                                               "idx,29,=,idx,42,=,or,idx,8,=,or,13,*,+,"
                                               "idx,32,=,idx,43,=,or,14,*,+,"
                                               "idx,44,=,idx,35,=,or,idx,53,=,or,15,*,+,"

                                               "idx,49,=,17,*,+,"
                                               "idx,50,=,idx,41,=,or,16,*,+,"
                                               "idx,46,=,idx,25,=,or,18,*,+,"
                                               "idx,47,=,idx,38,=,or,idx,26,=,or,19,*,+,"
                                               "idx,40,=,20,*,+,"
                                               "idx,37,=,idx,23,=,or,26,*,+,"

                                               "idx,4,=,23,*,+,"
                                               "idx,5,=,idx,39,=,or,22,*,+,"
                                               "idx,1,=,idx,19,=,or,24,*,+,"
                                               "idx,2,=,idx,20,=,or,idx,36,=,or,25,*,+,"
                                               "idx,22,=,21,*,+"

                                    }, inputs=["idx"], outputs=["idx"], scalars=["idx"],
                                    location=self.loc(block_x, block_y, 0, -1), hide=True)
        links.new(for_each_cubie.outputs["Index"], cubie_index.inputs["idx"])
        val_to_string = ValueToString(tree, value=cubie_index.outputs["idx"], location=self.loc(block_x, block_y, 0, 0),
                                      hide=True)

        strings_to_curve2 = StringToCurves(tree, string=val_to_string.std_out,
                                           location=self.loc(block_x, block_y, 1, 0))
        fil_curve2 = FillCurve(tree, location=self.loc(block_x, block_y, 2, -1))
        extrude2 = ExtrudeMesh(tree, location=self.loc(block_x, block_y, 2, 0), mode='FACES', offset=None, hide=True,
                               offset_scale=0.03)
        label_on_points = InstanceOnPoints(tree, location=self.loc(block_x, block_y, 3, 0),
                                           rotation=align_rotation.std_out, scale=[0.5] * 3, hide=False,
                                           selection=cubie_label.std_out, instance=extrude2.geometry_out)
        string_attr = StoredNamedAttribute(tree, location=self.loc(block_x, block_y, 4, 0),
                                           name="CubieIndexString", data_type="INT", domain="POINT",
                                           value=cubie_index.outputs["idx"], hide=False)
        for_each_cubie.create_geometry_line([label_on_points, string_attr])
        create_geometry_line(tree, [strings_to_curve2, fil_curve2, extrude2])
        realize_fonts = RealizeInstances(tree, location=self.loc(block_x, block_y, 6, 0), hide=True)
        cubie_font_idx = NamedAttribute(tree, location=self.loc(block_x, block_y, 7, -2), data_type="INT",
                                        name="CubieIndexString", hide=False)
        edge_filter = make_function(tree, name="EdgeFilter",
                                    functions={
                                        "selection": "idx,2,%,0,="
                                    }, inputs=["idx"], outputs=["selection"],
                                    scalars=["idx", "selection"], location=self.loc(block_x, block_y, 7, -1), hide=True)
        links.new(cubie_font_idx.std_out, edge_filter.inputs["idx"])
        material_edge_font = SetMaterial(tree, material="plastic_joker", location=self.loc(block_x, block_y, 8, 0),
                                         hide=True)
        material_corner_font = SetMaterial(tree, material="plastic_custom1", location=self.loc(block_x, block_y, 8, -1),
                                           hide=True)
        separate_geo = SeparateGeometry(tree, location=self.loc(block_x, block_y, 7, 0),
                                        selection=edge_filter.outputs["selection"], hide=True)
        create_geometry_line(tree, [for_each_cubie, realize_fonts, separate_geo])
        links.new(separate_geo.outputs["Selection"], material_edge_font.geometry_in)
        links.new(separate_geo.outputs["Inverted"], material_corner_font.geometry_in)
        create_geometry_line(tree, [material_edge_font, join])
        create_geometry_line(tree, [material_corner_font, join])

        # the labels for faces
        block_x = -0.5
        block_y = -3.5

        face_label = InputBoolean(tree, name="FaceSwitch", location=self.loc(block_x, block_y, 0, 1), value=False,
                                  hide=True)

        for_each_face = ForEachZone(tree, location=self.loc(block_x, block_y, 0, -2), hide=True,
                                    geometry=store_face_index.geometry_out)

        face_index = make_function(tree, name="FaceIndexFunction",
                                   functions={
                                       "idx":
                                           "idx,29,=,1,*,idx,32,=,2,*,+,idx,35,=,3,*,+,idx,28,=,4,*,+,idx,34,=,6,*,+,idx,27,=,5,*,+,idx,30,=,8,*,+,idx,33,=,7,*,+,"
                                           "idx,8,=,9,*,+,idx,7,=,10,*,+,idx,6,=,11,*,+,idx,5,=,12,*,+,idx,3,=,14,*,+,idx,2,=,13,*,+,idx,1,=,16,*,+,idx,0,=,15,*,+,"
                                           "idx,51,=,25,*,+,idx,52,=,26,*,+,idx,53,=,27,*,+,idx,48,=,28,*,+,idx,50,=,30,*,+,idx,45,=,29,*,+,idx,46,=,32,*,+,idx,47,=,31,*,+,"
                                           "idx,15,=,17,*,+,idx,16,=,18,*,+,idx,17,=,19,*,+,idx,12,=,20,*,+,idx,14,=,22,*,+,idx,9,=,21,*,+,idx,10,=,24,*,+,idx,11,=,23,*,+,"
                                           "idx,44,=,33,*,+,idx,43,=,34,*,+,idx,42,=,35,*,+,idx,41,=,36,*,+,idx,39,=,38,*,+,idx,38,=,37,*,+,idx,37,=,40,*,+,idx,36,=,39,*,+,"
                                           "idx,18,=,41,*,+,idx,21,=,42,*,+,idx,24,=,43,*,+,idx,19,=,44,*,+,idx,25,=,46,*,+,idx,20,=,45,*,+,idx,23,=,48,*,+,idx,26,=,47,*,+"
                                   }, inputs=["idx"], outputs=["idx"], scalars=["idx"],
                                   location=self.loc(block_x, block_y, 0, -1), hide=True)
        links.new(for_each_face.outputs["Index"], face_index.inputs["idx"])
        val_to_string = ValueToString(tree, value=face_index.outputs["idx"], location=self.loc(block_x, block_y, 0, 0),
                                      hide=True)

        strings_to_curve3 = StringToCurves(tree, string=val_to_string.std_out,
                                           location=self.loc(block_x, block_y, 1, 0))
        fil_curve3 = FillCurve(tree, location=self.loc(block_x, block_y, 2, -1))
        extrude3 = ExtrudeMesh(tree, location=self.loc(block_x, block_y, 2, 0), mode='FACES', offset=None, hide=True,
                               offset_scale=0.03)
        label_on_points2 = InstanceOnPoints(tree, location=self.loc(block_x, block_y, 3, 0),
                                            rotation=align_rotation.std_out, scale=[0.5] * 3, hide=False,
                                            selection=face_label.std_out, instance=extrude3.geometry_out)
        string_attr2 = StoredNamedAttribute(tree, location=self.loc(block_x, block_y, 4, 0),
                                            name="FaceIndexString", data_type="INT", domain="POINT",
                                            value=face_index.outputs["idx"], hide=False)
        for_each_face.create_geometry_line([label_on_points2, string_attr2])
        create_geometry_line(tree, [strings_to_curve3, fil_curve3, extrude3])
        realize_fonts2 = RealizeInstances(tree, location=self.loc(block_x, block_y, 6, 0), hide=True)
        face_font_idx = NamedAttribute(tree, location=self.loc(block_x, block_y, 7, -2), data_type="INT",
                                       name="FaceIndexString", hide=False)
        edge_filter2 = make_function(tree, name="FaceFilter",
                                     functions={
                                         "selection": "idx,2,%,0,="
                                     }, inputs=["idx"], outputs=["selection"],
                                     scalars=["idx", "selection"], location=self.loc(block_x, block_y, 7, -1),
                                     hide=True)
        links.new(face_font_idx.std_out, edge_filter2.inputs["idx"])
        material_edge_font2 = SetMaterial(tree, material="plastic_joker", location=self.loc(block_x, block_y, 8, 0),
                                          hide=True)
        material_corner_font2 = SetMaterial(tree, material="plastic_custom1",
                                            location=self.loc(block_x, block_y, 8, -1),
                                            hide=True)
        separate_geo2 = SeparateGeometry(tree, location=self.loc(block_x, block_y, 7, 0),
                                         selection=edge_filter2.outputs["selection"], hide=True)
        create_geometry_line(tree, [for_each_face, realize_fonts2, separate_geo2])
        # don't display face labels with zero index
        separate_geo3 = SeparateGeometry(tree, selection=face_font_idx.std_out,
                                         location=self.loc(block_x, block_y, 8, 0), hide=True)
        links.new(separate_geo2.outputs["Selection"], separate_geo3.geometry_in)
        links.new(separate_geo3.outputs["Selection"], material_edge_font2.geometry_in)
        links.new(separate_geo2.outputs["Inverted"], material_corner_font2.geometry_in)
        create_geometry_line(tree, [material_edge_font2, join])
        create_geometry_line(tree, [material_corner_font2, join])

class RubiksCubeUnfolded(GeometryNodesModifier):
    def __init__(self, name="RubiksCubeUnfolded", **kwargs):
        # define a mapping between geometry indices and face indices at are used in permutations
        indices = [37, 36, 33, 40, 0, 34, 39, 38, 35, 43, 46, 47, 42, 0, 48, 41, 44, 45, 21, 20, 17, 24, 0, 18, 23, 22,
                   19, 5, 4, 1, 8, 0, 2, 7, 6, 3, 15, 16, 13, 14, 0, 12, 11, 10, 9, 25, 26, 27, 28, 0, 30, 29, 32, 31,
                   54]
        self.index_to_face_index = {i: indices[i] for i in range(55)}

        super().__init__(name, automatic_layout=True, group_output=True, group_input=False, **kwargs)

    def loc(self, block_x=0, block_y=0, x=0, y=0, absolute=False):
        block_width = 7
        block_height = 5
        if absolute:
            return (200 * (block_x * block_width + x), 200 * (block_y * block_height + y))
        else:
            return (block_x * block_width + x, block_y * block_height + y)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs

        links = tree.links
        size = InputValue(tree, name="Size", value=3)
        cube1 = CubeMesh(tree, size=size.std_out, vertices_x=2, vertices_y=2, vertices_z=2)
        r2_factor = MathNode(tree, operation="MULTIPLY", inputs0=size.std_out, inputs1=2 / 3)

        unfold_node = UnfoldMeshNode(tree, progression=1.3, root_index=2, **kwargs)

        dual_mesh = DualMesh(tree)
        grid = Grid(tree, size_x=size.std_out, size_y=size.std_out, vertices_x=4, vertices_y=4)
        split_edges = SplitEdges(tree)

        face_rotations = NamedAttribute(tree, data_type='FLOAT_VECTOR', name='FRot')
        iop = InstanceOnPoints(tree, rotation=face_rotations.std_out)
        realize_instances = RealizeInstances(tree)
        normal = InputNormal(tree)
        store_normal = StoredNamedAttribute(tree, data_type="FLOAT_VECTOR", domain="FACE", name="FaceNormal",
                                            value=normal.std_out)
        dual_mesh2 = DualMesh(tree)
        index = Index(tree)
        store_face_index = StoredNamedAttribute(tree, data_type="INT", domain="POINT", name="FaceIndex",
                                                value=index.std_out)

        create_geometry_line(tree, [grid, split_edges], out=iop.inputs["Instance"])
        create_geometry_line(tree, [cube1, unfold_node, dual_mesh, iop])

        # add faces
        size = InputValue(tree, name="FaceSize", value=0.89, location=self.loc(-1, 2, 0, -1), hide=False)
        bevel = InputValue(tree, name="Bevel", value=6.3, location=self.loc(-1, 1, -0.5, 1), hide=False)
        bevel_function = make_function(tree, name="BevelFunction", location=self.loc(-1, 1, 0, 0),
                                       functions={
                                           "bevel": "size,bevel,/"
                                       }, inputs=["size", "bevel"], outputs=["bevel"],
                                       scalars=["size", "bevel"], hide=False)
        links.new(size.std_out, bevel_function.inputs["size"])
        links.new(bevel.std_out, bevel_function.inputs["bevel"])
        quadri = Quadrilateral(tree, location=self.loc(-1, 0, 0, 0),
                               width=size.std_out, height=size.std_out, hide=False)
        fillet_curve = FilletCurve(tree, radius=bevel_function.outputs["bevel"], count=10,
                                   location=self.loc(-1, 0, 1, 0), hide=False)
        fill_cuve = FillCurve(tree, mode='NGONS', location=self.loc(-1, 0, 2, 0), hide=False)
        extrude_face = ExtrudeMesh(tree, mode='FACES', offset_scale=1, offset=Vector([0, 0, 0.05]),
                                   location=self.loc(-1, 0, 3, 0),
                                   hide=False)
        create_geometry_line(tree, [quadri, fillet_curve, fill_cuve, extrude_face])

        # instance faces on dual mesh
        face_normal = NamedAttribute(tree, data_type="FLOAT_VECTOR", name="FaceNormal")
        align = AlignRotationToVector(tree, vector=face_normal.std_out)
        face_selector = IndexSwitch(tree, index=index.std_out, name="FaceSelectorSwitch", data_type="INT", hide=False)
        for i in range(55):
            face_selector.add_item(socket=1)
        iop2 = InstanceOnPoints(tree, name="FaceInstances", selection=face_selector.std_out,
                                instance=extrude_face.geometry_out, rotation=align.std_out)
        ri2 = RealizeInstances(tree)

        # add face colors

        # face coloring
        block_y = -0.25
        block_x = -0.25
        attr_face_index = NamedAttribute(tree, location=self.loc(block_x, block_y, 0, -3), name="FaceIndex", hide=False)

        red_selector = make_function(tree, name="RedSelector", location=self.loc(block_x, block_y, 1, -1.5),
                                     functions={
                                         "select": "face_index,44,>,face_index,54,<,and,switch,and"
                                     }, inputs=["switch", "face_index"], outputs=["select"],
                                     scalars=["switch", "face_index", "select"], hide=False)
        red_selector_switch = InputBoolean(tree, name="RedSelectorSwitch", value=True,
                                           location=self.loc(block_x, block_y, 1, -2.5), hide=False)

        orange_selector = make_function(tree, name="OrangeSelector", location=self.loc(block_x, block_y, 2, -1.5),
                                        functions={
                                            "select": "face_index,35,>,face_index,45,<,and,switch,and"
                                        }, inputs=["switch", "face_index"], outputs=["select"],
                                        scalars=["switch", "face_index", "select"], hide=False)
        orange_selector_switch = InputBoolean(tree, name="OrangeSelectorSwitch", value=True,
                                              location=self.loc(block_x, block_y, 2, -2.5), hide=False)

        green_selector = make_function(tree, name="GreenSelector", location=self.loc(block_x, block_y, 3, -1.5),
                                       functions={
                                           "select": "face_index,17,>,face_index,27,<,and,switch,and"
                                       }, inputs=["switch", "face_index"], outputs=["select"],
                                       scalars=["switch", "face_index", "select"], hide=False)
        green_selector_switch = InputBoolean(tree, name="GreenSelectorSwitch", value=True,
                                             location=self.loc(block_x, block_y, 3, -2.5), hide=False)

        blue_selector = make_function(tree, name="BlueSelector", location=self.loc(block_x, block_y, 4, -1.5),
                                      functions={
                                          "select": "face_index,-1,>,face_index,9,<,and,switch,and"
                                      }, inputs=["switch", "face_index"], outputs=["select"],
                                      scalars=["switch", "face_index", "select"], hide=False)
        blue_selector_switch = InputBoolean(tree, name="BlueSelectorSwitch", value=True,
                                            location=self.loc(block_x, block_y, 4, -2.5), hide=False)

        yellow_selector = make_function(tree, name="YellowSelector", location=self.loc(block_x, block_y, 5, -1.5),
                                        functions={
                                            "select": "face_index,8,>,face_index,18,<,and,switch,and"
                                        }, inputs=["switch", "face_index"], outputs=["select"],
                                        scalars=["switch", "face_index", "select"], hide=False)
        yellow_selector_switch = InputBoolean(tree, name="YellowSelectorSwitch", value=True,
                                              location=self.loc(block_x, block_y, 5, -2.5), hide=False)

        white_selector = make_function(tree, name="WhiteSelector", location=self.loc(block_x, block_y, 6, -1.5),
                                       functions={
                                           "select": "face_index,26,>,face_index,36,<,and,switch,and"
                                       }, inputs=["switch", "face_index"], outputs=["select"],
                                       scalars=["switch", "face_index", "select"], hide=False)
        white_selector_switch = InputBoolean(tree, name="WhiteSelectorSwitch", value=True,
                                             location=self.loc(block_x, block_y, 6, -2.5), hide=False)

        edge_selector = make_function(tree, name="EdgeSelector", location=self.loc(block_x, block_y, 7, -1.5),
                                      functions={
                                          "select": "idx,1,=,idx,7,=,or,idx,10,=,or,idx,12,=,or,idx,14,=,or,idx,16,=,or,"
                                                    "idx,19,=,or,idx,25,=,or,idx,28,=,or,idx,30,=,or,idx,32,=,or,idx,34,=,or,switch,and",
                                      }, inputs=["switch", "idx"], outputs=["select"],
                                      scalars=["switch", "idx", "select"], hide=False)

        edge_selector_switch = InputBoolean(tree, name="EdgeSelectorSwitch", value=False,
                                            location=self.loc(block_x, block_y, 7, -2.5), hide=False)

        corner_selector = make_function(tree, name="CornerSelector", location=self.loc(block_x, block_y, 8, -1.5),
                                        functions={
                                            "select": "idx,20,=,idx,26,=,or,idx,18,=,or,"
                                                      "idx,24,=,or,idx,6,=,or,idx,0,=,or,idx,8,=,or,"
                                                      "idx,2,=,or,switch,and"
                                        }, inputs=["switch", "idx"], outputs=["select"],
                                        scalars=["switch", "idx", "select"], hide=False)

        corner_selector_switch = InputBoolean(tree, name="CornerSelectorSwitch", value=False,
                                              location=self.loc(block_x, block_y, 8, -2.5), hide=False)

        links.new(attr_face_index.std_out, red_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, blue_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, yellow_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, orange_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, green_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, white_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, edge_selector.inputs["idx"])
        links.new(attr_face_index.std_out, corner_selector.inputs["idx"])
        links.new(red_selector_switch.std_out, red_selector.inputs["switch"])
        links.new(white_selector_switch.std_out, white_selector.inputs["switch"])
        links.new(green_selector_switch.std_out, green_selector.inputs["switch"])
        links.new(blue_selector_switch.std_out, blue_selector.inputs["switch"])
        links.new(orange_selector_switch.std_out, orange_selector.inputs["switch"])
        links.new(yellow_selector_switch.std_out, yellow_selector.inputs["switch"])
        links.new(edge_selector_switch.std_out, edge_selector.inputs["switch"])
        links.new(corner_selector_switch.std_out, corner_selector.inputs["switch"])

        material_black = SetMaterial(tree, location=self.loc(block_x, block_y, 0, 0), material="gray_1")

        material_red = SetMaterial(tree, location=self.loc(block_x, block_y, 1, 0),
                                   selection=red_selector.outputs["select"], material="red",
                                   hide=False)
        material_blue = SetMaterial(tree, location=self.loc(block_x, block_y, 2, 0),
                                    selection=blue_selector.outputs["select"], material="blue",
                                    hide=False)
        material_yellow = SetMaterial(tree, location=self.loc(block_x, block_y, 3, 0),
                                      selection=yellow_selector.outputs["select"], material="yellow",
                                      hide=False)
        material_orange = SetMaterial(tree, location=self.loc(block_x, block_y, 4, 0),
                                      selection=orange_selector.outputs["select"], material="orange",
                                      hide=False)
        material_green = SetMaterial(tree, location=self.loc(block_x, block_y, 5, 0),
                                     selection=green_selector.outputs["select"], material="green",
                                     hide=False)
        material_white = SetMaterial(tree, location=self.loc(block_x, block_y, 6, 0),
                                     selection=white_selector.outputs["select"], material="text",
                                     hide=False)
        material_edge = SetMaterial(tree, location=self.loc(block_x, block_y, 7, 0),
                                    selection=edge_selector.outputs["select"], material="joker",
                                    hide=False)

        material_corner = SetMaterial(tree, location=self.loc(block_x, block_y, 8, 0),
                                      selection=corner_selector.outputs["select"],
                                      material="custom1", hide=False)
        mat_nodes = [material_corner, material_edge, material_green, material_orange, material_red, material_white,
                     material_black, material_blue, material_yellow, material_blue]
        [self.materials.append(mat_node.material) for mat_node in mat_nodes]

        last_color = material_corner
        join_before_transformation = JoinGeometry(tree, name="FaceGeometry")
        create_geometry_line(tree, [last_color, join_before_transformation])

        # join geometry
        join = JoinGeometry(tree, label="JoinGeometry")

        # cubie branch
        # create a smooth cubies
        show_cubies = InputBoolean(tree, value=True, label="ShowCubiesFlag")
        cube2 = CubeMesh(tree, size=r2_factor.std_out, vertices_x=3, vertices_y=3, vertices_z=3)
        store_cubie_index = StoredNamedAttribute(tree, name="CubieIndex", data_type="INT", domain="POINT")
        # position cubies to the instances
        cube_size = InputValue(tree, value=1, label="CubeSize")
        cubie = BeveledCubeNode(tree, size=cube_size.std_out, bevel=6.03)
        iop_cubies = InstanceOnPoints(tree, instance=cubie.geometry_out, selection=show_cubies.std_out)
        black = SetMaterial(tree, material="background")
        create_geometry_line(tree, [cube2, store_cubie_index, iop_cubies, black, join])

        create_geometry_line(tree, [iop, realize_instances, store_normal, dual_mesh2, store_face_index, iop2, ri2,
                                    material_black, material_red, material_blue, material_yellow, material_orange,
                                    material_green,
                                    material_white, material_edge, material_corner])

        # prepare selection for each face make index offset by 1

        attr_face_index2 = NamedAttribute(tree, data_type="INT", name="FaceIndex")
        add_one = MathNode(tree, operation="ADD", inputs0=attr_face_index2.std_out, inputs1=1)

        # convert face geometry to instances
        for i in range(1, 55):
            compare = CompareNode(tree, operation="EQUAL", inputs0=add_one.std_out, inputs1=i)
            select_face = SeparateGeometry(tree, selection=compare.std_out, label="Face" + str(i))
            geo_to_instance = GeometryToInstance(tree, label="InstanceOfFace" + str(i))
            create_geometry_line(tree, [join_before_transformation, select_face, geo_to_instance, join])

        # add fonts

        # create the selection function, depending on the index of the face
        idx = Index(tree, location=self.loc(block_x, block_y, 1, 1), hide=True)
        edges = InputBoolean(tree, name="EdgeSwitch", location=self.loc(block_x, block_y, 1, 1.5), value=False,
                             hide=True)
        corners = InputBoolean(tree, name="CornerSwitch", location=self.loc(block_x, block_y, 1, 2), value=False,
                               hide=True)
        index_function = make_function(tree, name="index_function",
                                       functions={
                                           "zero": "idx,1,=,idx,7,=,or,idx,10,=,or,idx,12,=,or,idx,14,=,or,idx,16,=,or,"
                                                   "idx,19,=,or,idx,25,=,or,idx,28,=,or,idx,30,=,or,idx,32,=,or,idx,34,=,or,edges,and,"
                                                   "idx,20,=,idx,26,=,or,idx,18,=,or,idx,24,=,or,idx,6,=,or,idx,0,=,or,idx,8,=,or,idx,2,=,or,corners,and,or",
                                           "one": "idx,3,=,idx,5,=,or,idx,21,=,or,idx,23,=,or,idx,37,=,or,idx,39,=,or,idx,41,=,or,idx,43,=,or,"
                                                  "idx,46,=,or,idx,48,=,or,idx,50,=,or,idx,52,=,or,edges,and,"
                                                  "idx,42,=,idx,33,=,or,idx,51,=,or,idx,15,=,or,idx,38,=,or,idx,11,=,or,idx,29,=,or,idx,47,=,or,corners,and,or",
                                           "two": "idx,36,=,idx,27,=,or,idx,9,=,or,idx,45,=,or,idx,53,=,or,idx,17,=,or,idx,44,=,or,idx,35,=,or,corners,and"
                                       }, inputs=["idx", "edges", "corners"], outputs=["zero", "one", "two"],
                                       scalars=["idx", "zero", "one", "two", "edges", "corners"],
                                       vectors=[], location=self.loc(block_x, block_y, 2, 1), hide=True)
        links.new(idx.std_out, index_function.inputs["idx"])
        links.new(edges.std_out, index_function.inputs["edges"])
        links.new(corners.std_out, index_function.inputs["corners"])

        # create three geometry lines for the numbers 0, 1, 2
        dual_mesh3 = DualMesh(tree)
        for i in range(3):
            text = StringToCurves(tree, string=str(i), hide=True)
            iop = InstanceOnPoints(tree, hide=True,
                                   instance=text.geometry_out, selection=index_function.outputs[i])
            set_label = SetPosition(tree, offset=Vector([0, 0, 0.05]))
            fill_curve = FillCurve(tree, hide=True)
            extrude = ExtrudeMesh(tree, mode='FACES', offset=None, hide=True, offset_scale=0.03)
            create_geometry_line(tree, [store_normal, dual_mesh3, iop, set_label, fill_curve, extrude, join])

        # create face indices

        # prepare number color and outline color depending on the parity of the number

        number_colors = get_from_kwargs(kwargs, "number_colors", ["joker", "custom1"])
        number_outline_colors = get_from_kwargs(kwargs, "number_outline_colors", ["joker", "custom1"])
        material_number_even = get_material(number_colors[0])
        material_number_odd = get_material(number_colors[1])
        material_outline_even = get_material(number_outline_colors[0])
        material_outline_odd = get_material(number_outline_colors[1])
        mats = [material_number_odd, material_number_even, material_outline_even, material_outline_odd]
        labels = ["MaterialNumberOdd", "MaterialNumberEven", "MaterialOutlineOdd", "MaterialOutlineEven"]

        [mat_number_odd, mat_number_even, mat_outline_odd, mat_outline_even] = \
            [InputMaterial(tree, material=m, name=l) for m, l in zip(mats, labels)]

        dual_mesh4 = DualMesh(tree)
        index = Index(tree)
        store_face_index2 = StoredNamedAttribute(tree, name="FaceIndex", data_type="INT", domain="POINT",
                                                 value=index.std_out)
        face_switch = InputBoolean(tree, name="FaceSwitch", value=True)
        for_each_index = ForEachZone(tree, name="ForEachFace", hide=True)
        switch_index = IndexSwitch(tree, name="FaceLabelSwitch", data_type="INT", index=for_each_index.outputs["Index"],
                                   hide=False)

        indices = list(self.index_to_face_index.values())
        selected_face_indices = get_from_kwargs(kwargs, "selected_face_indices", list(range(1, 49)))
        for i in range(len(indices)):
            if indices[i] not in selected_face_indices:
                indices[i] = 0

        for i in range(0, 55):
            switch_index.add_item(socket=indices[i])

        value_to_string = ValueToString(tree, value=switch_index.std_out, hide=True)
        string_to_curve = StringToCurves(tree, font="Arial Black", string=value_to_string.outputs["String"], size=0.7,
                                         hide=True)
        fill_curve = FillCurve(tree, mode="NGONS", hide=True)
        extrude_face = ExtrudeMesh(tree, mode="FACES", offset=None, hide=True, offset_scale=0.07)
        iop_face = InstanceOnPoints(tree, hide=True, instance=string_to_curve.geometry_out,
                                    selection=face_switch.std_out
                                    )

        parity_selector = make_function(tree, functions={
            "even": "idx,2,%,0,="
        }, inputs=["idx"], outputs=["even"], scalars=["even", "idx"], name="ParitySelector", hide=True)
        links.new(switch_index.std_out, parity_selector.inputs["idx"])

        # outline of the face label
        join_string = JoinGeometry(tree, name="FaceLabelJoin", hide=True)

        curve_wire_frame = CurveWireFrame(tree, radius=0.02, resolution=8)
        set_pos = SetPosition(tree, offset=[0, 0, 0.05], hide=True)
        mat_outline_switch = Switch(tree, input_type="MATERIAL", name="FontOutlineSwitch", hide=True,
                                    switch=parity_selector.outputs["even"], true=mat_outline_even.std_out,
                                    false=mat_outline_odd.std_out)
        mat_outline = SetMaterial(tree, name="LabelOutlineMaterial", material=mat_outline_switch.std_out)
        create_geometry_line(tree, [string_to_curve, curve_wire_frame, set_pos, mat_outline, join_string])

        # inside of the face label
        material_switch = Switch(tree, name="FontMaterialSwitch", input_type="MATERIAL", hide=True,
                                 switch=parity_selector.outputs["even"],
                                 false=mat_number_even.std_out, true=mat_number_odd.std_out)
        set_material_font = SetMaterial(tree, material=material_switch.std_out, name="LabelMaterial")

        store_face_index_string = StoredNamedAttribute(tree, name="FaceIndexString", data_type="INT", domain="POINT",
                                                       value=switch_index.std_out, hide=True)

        for_each_index.create_geometry_line([iop_face, store_face_index_string])
        create_geometry_line(tree, [string_to_curve, fill_curve, extrude_face, set_material_font, join_string],
                             out=iop_face.inputs["Instance"])
        realize_face_indices = RealizeInstances(tree, hide=True)
        attribute_face_index = NamedAttribute(tree, data_type="INT", name="FaceIndexString", hide=True)

        even_face_selector = make_function(tree, name="EvenFaceSelector", hide=True,
                                           functions={
                                               "selection": "idx,2,%,0,="
                                           }, inputs=["idx"], outputs=["selection"],
                                           scalars=["idx", "selection"], vectors=[])
        tree.links.new(attribute_face_index.std_out, even_face_selector.inputs["idx"])
        sep_geo = SeparateGeometry(tree, selection=even_face_selector.outputs["selection"], hide=True)
        select_non_zero = SeparateGeometry(tree, selection=attribute_face_index.std_out, hide=True)

        self.materials += mats

        create_geometry_line(tree, [join_before_transformation], ins=sep_geo.outputs["Inverted"])
        create_geometry_line(tree, [store_normal, dual_mesh4, store_face_index2, for_each_index,
                                    realize_face_indices, sep_geo, select_non_zero, join_before_transformation])

        # create global rotation
        gl_rotation = InputVector(tree, name="FinalRotation", value=[0, 0, 0], hide=True)
        gl_pivot = InputVector(tree, name="FinalPivot", value=[0, 0, 0], hide=True)
        global_transform = RotateInstances(tree, name="FinalRotation", local_space=False, rotation=gl_rotation.std_out,
                                           pivot_point=gl_pivot.std_out, hide=True)

        # connect to output
        create_geometry_line(tree, [join, global_transform], out=out.inputs["Geometry"])

class RubiksCubeTower(GeometryNodesModifier):
    def __init__(self, name="RubiksCubeTower", **kwargs):
        super().__init__(name, automatic_layout=True, group_output=True, group_input=False, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links
        # show points

        count = InputInteger(tree, label="NumberOfCubes", integer=2500)
        visible_count = InputInteger(tree, label="Visible", integer=0)
        index = Index(tree)

        greater_than = CompareNode(tree, operation="GREATER_THAN", data_type="INT",
                                   inputs0=index.std_out, inputs1=visible_count.std_out)

        del_geo = DeleteGeometry(tree, selection=greater_than.std_out)

        size = InputValue(tree, label="Size", value=1.54)

        points = Points(tree, count=count.std_out)
        foreachzone = ForEachZone(tree, geometry=points.geometry_out)

        position_function = make_function(tree, name="PositionFunction",
                                          functions={
                                              "offset": ["0", "0", "size,2,*,value,*"]},
                                          inputs=["size", "value"], outputs=["offset"],
                                          scalars=["size", "value"], vectors=["offset"])
        links.new(size.std_out, position_function.inputs["size"])
        links.new(foreachzone.index, position_function.inputs["value"])

        simpleCube = SimpleRubiksCubeNode(tree, seed=foreachzone.index,
                                          hide=True)
        set_pos = SetPosition(tree,
                              offset=position_function.outputs["offset"])
        iop = InstanceOnPoints(tree, instance=simpleCube.geometry_out)
        join = JoinGeometry(tree)
        foreachzone.create_geometry_line([set_pos, iop, join])
        join2 = JoinGeometry(tree)
        create_geometry_line(tree, [points, del_geo, foreachzone, join2], out=out.inputs["Geometry"])

class RubiksSphereModifier(GeometryNodesModifier):
    def __init__(self, name="RubiksSphereModifier", **kwargs):
        super().__init__(name, automatic_layout=False, **kwargs)

    def loc(self, block_x=0, block_y=0, x=0, y=0, absolute=False):
        block_width = 7
        block_height = 5
        if absolute:
            return (200 * (block_x * block_width + x), 200 * (block_y * block_height + y))
        else:
            return (block_x * block_width + x, block_y * block_height + y)

    def create_node(self, tree, **kwargs):
        '''
        we need to create the following structure

        3x3x3 grid (-1,2)

                                selected faces (0,1)          selection for animation(1,1)

        beveled face (-1,0)     face coloring  (0,0)          transformation (1,0)            output (2,0)

        '''
        # output block
        out = self.group_outputs
        out.location = self.loc(2, 0, 0, 0, absolute=True)
        links = tree.links

        # 3x3x3 block
        separation = InputValue(tree, name="Separation", value=1, location=self.loc(-1, 2, 0, 0), hide=False)
        size = InputValue(tree, name="Cube Size", value=0.9, location=self.loc(-1, 2, 0, -1), hide=False)
        line_function = make_function(tree, name="LineFunction",
                                      functions={
                                          "start": ["0", "0", "-1,separation,*"],
                                          "offset": ["0", "0", "1,separation,*"]
                                      }, inputs=["separation"], outputs=["start", "offset"],
                                      scalars=["separation"], vectors=["start", "offset"],
                                      location=self.loc(-1, 2, 1, 0), hide=False)
        links.new(separation.std_out, line_function.inputs["separation"])
        mesh_line = MeshLine(tree, size=size.std_out, mode='OFFSET', count_mode="TOTAL", count=3,
                             start_location=line_function.outputs["start"],
                             offset=line_function.outputs["offset"], location=self.loc(-1, 2, 2, 0), hide=False
                             )
        grid = Grid(tree, size_x=2, size_y=2, vertices_y=3, vertices_x=3, location=self.loc(-1, 2, 2, -1.5), hide=False)
        iop = InstanceOnPoints(tree, instance=grid.geometry_out, scale=separation.std_out, hide=False,
                               location=self.loc(-1, 2, 3, 0))
        realize_instance = RealizeInstances(tree, location=self.loc(-1, 2, 4, 0), hide=False)
        index = Index(tree, location=self.loc(-1, 2, 4, -1.5), hide=False)
        # increase index by one, since index 0 cannot be selected (bug, probably)
        inc = make_function(tree, name="Function",
                            functions={
                                "idx": "idx,1,+"
                            }, inputs=["idx"], outputs=["idx"],
                            scalars=["idx"])
        links.new(index.std_out, inc.inputs["idx"])
        pos = Position(tree, location=self.loc(-1, 2, 5, -2), hide=False)
        cubie_index = StoredNamedAttribute(tree, domain='POINT', data_type='INT',
                                           name="CubieIndex", value=inc.outputs["idx"], hide=False,
                                           location=self.loc(-1, 2, 5, 0))
        cubie_pos = StoredNamedAttribute(tree, domain='POINT', data_type='FLOAT_VECTOR',
                                         name="CubiePosition", value=pos.std_out, hide=False,
                                         location=self.loc(-1, 2, 6, 0))
        create_geometry_line(tree, [mesh_line, iop, realize_instance, cubie_index, cubie_pos])

        # beveled cube
        bevel = InputValue(tree, name="Bevel", value=12.6, location=self.loc(-1, 1, -0.5, 1), hide=False)
        bevel_function = make_function(tree, name="BevelFunction", location=self.loc(-1, 1, 0, 0),
                                       functions={
                                           "bevel": "size,bevel,/"
                                       }, inputs=["size", "bevel"], outputs=["bevel"],
                                       scalars=["size", "bevel"], hide=False)
        links.new(size.std_out, bevel_function.inputs["size"])
        links.new(bevel.std_out, bevel_function.inputs["bevel"])

        # beveled face(-1, 0)

        quadri = Quadrilateral(tree, location=self.loc(-1, 0, 0, 0),
                               width=size.std_out, height=size.std_out, hide=False)
        fillet_curve = FilletCurve(tree, radius=bevel_function.outputs["bevel"], count=10,
                                   location=self.loc(-1, 0, 1, 0), hide=False)
        fill_curve = FillCurve(tree, mode='NGONS', location=self.loc(-1, 0, 2, 0), hide=False)

        # make sphere
        block_x = -1
        block_y = -0.5
        pos_sphere = Position(tree, location=self.loc(block_x, block_y, 0, -2), hide=False)

        radius = InputValue(tree, location=self.loc(block_x, block_y, -1, -1.5), hide=False, name="Radius", value=1)
        mul = MathNode(tree, operation="MULTIPLY", location=self.loc(block_x, block_y, -1, 0),
                       inputs0=radius.std_out, inputs1=0.95, hide=True)
        projection = make_function(tree, name="Projection",
                                   functions={
                                       "pos": "pos,radius,pos,length,/,scale"
                                   }, inputs=["pos", "radius"], outputs=["pos"],
                                   scalars=["radius"], vectors=["pos"],
                                   location=self.loc(block_x, block_y, 0, 0),
                                   hide=False)
        links.new(pos_sphere.std_out, projection.inputs["pos"])
        links.new(mul.std_out, projection.inputs["radius"])
        make_round = SetPosition(tree, location=self.loc(block_x, block_y, 1, 0), position=projection.outputs["pos"],
                                 hide=False)
        shade_smooth_face = SetShadeSmooth(tree, location=self.loc(block_x, block_y, 2, 0), hide=False)
        extrude_face = ExtrudeMesh(tree, mode='FACES', offset_scale=0.06, offset=None,
                                   location=self.loc(block_x, block_y, 3, 0), hide=False)
        create_geometry_line(tree, [quadri, fillet_curve, fill_curve])

        # selected faces(0, 0)
        block_y = 2.5
        cube = CubeMesh(tree, size=[0.97] * 3, location=self.loc(0, block_y, -1, 0), hide=False)
        iop3 = InstanceOnPoints(tree, instance=cube.geometry_out, hide=False, location=self.loc(0, block_y, 0, 0))
        realize_cube = RealizeInstances(tree, location=self.loc(0, block_y, 1, 0), hide=False)

        # extract all vertices larger than 0.9 to make them round
        pos_sphere_cubie = Position(tree, location=self.loc(0, block_y + 0.5, 0, -1), hide=False)

        projection_cubie = make_function(tree, name="Projection",
                                         functions={
                                             "pos": "pos,radius,pos,length,/,scale",
                                             "sel": "pos,length,0.9,>"
                                         }, inputs=["pos", "radius"], outputs=["pos", "sel"],
                                         scalars=["radius", "sel"], vectors=["pos"],
                                         location=self.loc(0, block_y, 1, 1),
                                         hide=False)
        links.new(pos_sphere_cubie.std_out, projection_cubie.inputs["pos"])
        links.new(radius.std_out, projection_cubie.inputs["radius"])
        make_round_cubie = SetPosition(tree, location=self.loc(0, block_y + 0.5, 2, 1),
                                       selection=projection_cubie.outputs["sel"],
                                       position=projection_cubie.outputs["pos"],
                                       hide=False)
        cube_material = SetMaterial(tree, location=self.loc(0, block_y, 2, 0), material="background")
        create_geometry_line(tree, [cubie_pos, iop3, realize_cube, make_round_cubie, cube_material])

        face_area = FaceArea(tree, location=self.loc(0, block_y, 0, -3.5), hide=False)
        position2 = Position(tree, location=self.loc(0, block_y, 0, -4.5), hide=False)
        abs = VectorMath(tree, operation="ABSOLUTE", inputs0=position2.std_out, location=self.loc(0, block_y, 1, -4),
                         hide=False)

        # the basic algorithm behind the face selection is the following
        # for faces where the x coordinate is absolutely the largest coordinate, we only select the face
        # when the x>0.5*cubesize+spacing or x<-(0.5*cubesize+spacing)
        # additionally only faces with an area larger than 0.5*cubesize**2 are selected

        outside_faces = make_function(tree, name="OutSideFaces",
                                      functions={
                                          "select": "pos_x,pos_y,>,pos_x,pos_z,>,and,pos_x,0.5,size,*,spc,+,>,and,"
                                                    "pos_y,pos_x,>,pos_y,pos_z,>,and,pos_y,0.5,size,*,spc,+,>,and,or,"
                                                    "pos_z,pos_x,>,pos_z,pos_y,>,and,pos_z,0.5,size,*,spc,+,>,and,or,"
                                                    "area,0.5,size,2,**,*,>,and"
                                      }, inputs=["pos", "area", "size", "spc"], outputs=["select"],
                                      scalars=["area", "select", "size", "spc"], vectors=["pos"], hide=False,
                                      location=self.loc(0, block_y, 1, -1.75))

        links.new(abs.std_out, outside_faces.inputs["pos"])
        links.new(face_area.std_out, outside_faces.inputs["area"])
        links.new(size.std_out, outside_faces.inputs["size"])
        links.new(separation.std_out, outside_faces.inputs["spc"])

        # set cubie faces

        sep_geo = SeparateGeometry(tree, location=self.loc(0, 1, 0, 0), domain='FACE', hide=False,
                                   selection=outside_faces.outputs["select"])
        attr_cubie_index = NamedAttribute(tree, location=self.loc(0, 1, -1, -1), name="CubieIndex",
                                          data_type='INT', hide=False)
        normal = InputNormal(tree, location=self.loc(0, 1, -1, -2), hide=False)
        store_cubie_index_at_face = StoredNamedAttribute(tree, location=self.loc(0, 1, 1, 0), data_type='INT',
                                                         domain='FACE', name="CubieIndexAtFace",
                                                         value=attr_cubie_index.std_out,
                                                         hide=False)
        store_face_normal = StoredNamedAttribute(tree, location=self.loc(0, 1, 2, 0), data_type='FLOAT_VECTOR',
                                                 domain='FACE', name="FaceNormal", value=normal.std_out, hide=False)
        dual_mesh = DualMesh(tree, location=self.loc(0, 1, 3, 0), hide=False)
        position3 = Position(tree, location=self.loc(0, 1, -1, -3), hide=False)
        abs_2 = VectorMath(tree, operation="ABSOLUTE", inputs0=position3.std_out,
                           location=self.loc(0, 1, 0, -3))
        index3 = Index(tree, location=self.loc(0, 1, -1, -5), hide=False)

        # the aim of this function is to re-arrange the indices of the faces,
        # such that a side of the Rubik's cube
        # has nine faces with con secutive index values
        index_function = make_function(tree, name="IndexFunction",
                                       functions={
                                           "weight": "apos_x,apos_y,>,apos_x,apos_z,>,and,10,*,pos_x,sgn,*,"
                                                     "apos_y,apos_x,>,apos_y,apos_z,>,and,5,*,pos_y,sgn,*,+,"
                                                     "apos_z,apos_x,>,apos_z,apos_y,>,and,1,*,pos_z,sgn,*,+,"
                                                     "index,100,/,+"
                                       }, inputs=["index", "pos", "apos"], outputs=["weight"],
                                       scalars=["index", "weight"], vectors=["pos", "apos"],
                                       location=self.loc(0, 1, 3, -1),
                                       hide=False)

        links.new(index3.std_out, index_function.inputs["index"])
        links.new(abs_2.std_out, index_function.inputs["apos"])
        links.new(position3.std_out, index_function.inputs["pos"])

        sort = SortElements(tree, location=self.loc(0, 1, 4, 0), hide=False, domain='POINT',
                            sort_weight=index_function.outputs["weight"])

        attr_face_normal = NamedAttribute(tree, location=self.loc(0, 1, -1, -4),
                                          name="FaceNormal", data_type='FLOAT_VECTOR', hide=False)

        store_face_index = StoredNamedAttribute(tree, location=self.loc(0, 1, 5, 0), hide=False,
                                                domain='POINT', data_type='INT', name="FaceIndex", value=index3.std_out)
        align_rotation = AxesToRotation(tree, location=self.loc(0, 1, 5, -2), hide=False,
                                        primary_axis='Z', secondary_axis='Y',
                                        primary_direction=attr_face_normal.std_out,
                                        secondary_direction=Vector([0, 0, 1]))
        # align_rotation=AlignRotationToVector(tree,location=self.loc(0,1,5,-2),hide=False,
        #                                      vector=attr_face_normal.std_out)
        # pipe in beveled cubie face
        iop4 = InstanceOnPoints(tree, instance=fill_curve.geometry_out, hide=False,
                                rotation=align_rotation.std_out, location=self.loc(0, 1, 6, 0))

        realize_faces = RealizeInstances(tree, location=self.loc(0, 1, 7, 0), hide=False)
        create_geometry_line(tree, [realize_cube, sep_geo, store_cubie_index_at_face,
                                    store_face_normal, dual_mesh, sort, store_face_index, iop4, realize_faces])

        # face coloring
        block_y = -0.25
        block_x = -0.25
        attr_face_index = NamedAttribute(tree, location=self.loc(block_x, block_y, 0, -3), name="FaceIndex", hide=False)

        red_selector = make_function(tree, name="RedSelector", location=self.loc(block_x, block_y, 1, -1.5),
                                     functions={
                                         "select": "face_index,44,>,face_index,54,<,and,switch,and"
                                     }, inputs=["switch", "face_index"], outputs=["select"],
                                     scalars=["switch", "face_index", "select"], hide=False)
        red_selector_switch = InputBoolean(tree, name="RedSelectorSwitch", value=True,
                                           location=self.loc(block_x, block_y, 1, -2.5), hide=False)

        blue_selector = make_function(tree, name="BlueSelector", location=self.loc(block_x, block_y, 2, -1.5),
                                      functions={
                                          "select": "face_index,35,>,face_index,45,<,and,switch,and"
                                      }, inputs=["switch", "face_index"], outputs=["select"],
                                      scalars=["switch", "face_index", "select"], hide=False)
        blue_selector_switch = InputBoolean(tree, name="BlueSelectorSwitch", value=True,
                                            location=self.loc(block_x, block_y, 2, -2.5), hide=False)

        yellow_selector = make_function(tree, name="YellowSelector", location=self.loc(block_x, block_y, 3, -1.5),
                                        functions={
                                            "select": "face_index,17,>,face_index,27,<,and,switch,and"
                                        }, inputs=["switch", "face_index"], outputs=["select"],
                                        scalars=["switch", "face_index", "select"], hide=False)
        yellow_selector_switch = InputBoolean(tree, name="YellowSelectorSwitch", value=True,
                                              location=self.loc(block_x, block_y, 3, -2.5), hide=False)

        orange_selector = make_function(tree, name="OrangeSelector", location=self.loc(block_x, block_y, 4, -1.5),
                                        functions={
                                            "select": "face_index,-1,>,face_index,9,<,and,switch,and"
                                        }, inputs=["switch", "face_index"], outputs=["select"],
                                        scalars=["switch", "face_index", "select"], hide=False)
        orange_selector_switch = InputBoolean(tree, name="OrangeSelectorSwitch", value=True,
                                              location=self.loc(block_x, block_y, 4, -2.5), hide=False)

        green_selector = make_function(tree, name="GreenSelector", location=self.loc(block_x, block_y, 5, -1.5),
                                       functions={
                                           "select": "face_index,8,>,face_index,18,<,and,switch,and"
                                       }, inputs=["switch", "face_index"], outputs=["select"],
                                       scalars=["switch", "face_index", "select"], hide=False)
        green_selector_switch = InputBoolean(tree, name="GreenSelectorSwitch", value=True,
                                             location=self.loc(block_x, block_y, 5, -2.5), hide=False)

        white_selector = make_function(tree, name="WhiteSelector", location=self.loc(block_x, block_y, 6, -1.5),
                                       functions={
                                           "select": "face_index,26,>,face_index,36,<,and,switch,and"
                                       }, inputs=["switch", "face_index"], outputs=["select"],
                                       scalars=["switch", "face_index", "select"], hide=False)
        white_selector_switch = InputBoolean(tree, name="WhiteSelectorSwitch", value=True,
                                             location=self.loc(block_x, block_y, 6, -2.5), hide=False)

        links.new(attr_face_index.std_out, red_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, blue_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, yellow_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, orange_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, green_selector.inputs["face_index"])
        links.new(attr_face_index.std_out, white_selector.inputs["face_index"])
        links.new(red_selector_switch.std_out, red_selector.inputs["switch"])
        links.new(white_selector_switch.std_out, white_selector.inputs["switch"])
        links.new(green_selector_switch.std_out, green_selector.inputs["switch"])
        links.new(blue_selector_switch.std_out, blue_selector.inputs["switch"])
        links.new(orange_selector_switch.std_out, orange_selector.inputs["switch"])
        links.new(yellow_selector_switch.std_out, yellow_selector.inputs["switch"])

        emissions = get_from_kwargs(kwargs, "emissions", {})

        material_black = SetMaterial(tree, location=self.loc(block_x, block_y, 0, 0), material="gray_1")

        material_red = SetMaterial(tree, location=self.loc(block_x, block_y, 1, 0),
                                   selection=red_selector.outputs["select"], material="red",
                                   hide=False)

        emission_blue = emissions.get("blue", 0)
        blue = get_material("blue", emission=emission_blue, **kwargs)
        self.materials.append(blue)
        material_blue = SetMaterial(tree, location=self.loc(block_x, block_y, 2, 0),
                                    selection=blue_selector.outputs["select"], material=blue,
                                    hide=False, emission=emission_blue)
        material_yellow = SetMaterial(tree, location=self.loc(block_x, block_y, 3, 0),
                                      selection=yellow_selector.outputs["select"], material="yellow",
                                      hide=False)
        emission_orange = emissions.get("orange", 0)
        orange = get_material("orange", emission=emission_orange, **kwargs)
        self.materials.append(orange)
        material_orange = SetMaterial(tree, location=self.loc(block_x, block_y, 4, 0),
                                      selection=orange_selector.outputs["select"], material=orange,
                                      hide=False, emission=emission_orange)
        emission_green = emissions.get("green", 0)
        green = get_material("green", emission=emission_green, **kwargs)
        self.materials.append(green)
        material_green = SetMaterial(tree, location=self.loc(block_x, block_y, 5, 0),
                                     selection=green_selector.outputs["select"], material=green,
                                     hide=False, emission=emissions.get("green", 0))
        material_white = SetMaterial(tree, location=self.loc(block_x, block_y, 6, 0),
                                     selection=white_selector.outputs["select"], material="text",
                                     hide=False)

        last_material = material_white
        create_geometry_line(tree, [realize_faces, make_round, shade_smooth_face, extrude_face,
                                    material_black, material_red, material_blue, material_yellow, material_orange,
                                    material_green, material_white])

        # selection block
        block_y = 2
        block_x = 1.1
        join = JoinGeometry(tree, location=self.loc(block_x, block_y, 0, 0), hide=False)
        create_geometry_line(tree, [last_material, join])
        create_geometry_line(tree, [cube_material, join])
        attr_cubie_index2 = NamedAttribute(tree, location=self.loc(block_x, block_y, 0, -1), name="CubieIndex",
                                           hide=False, data_type="INT")
        attr_face_index2 = NamedAttribute(tree, location=self.loc(block_x, block_y, 0, -2),
                                          name="CubieIndexAtFace", data_type="INT", hide=False)

        # now we need to separate each cubie and prepare its own transformation panel.

        join2 = JoinGeometry(tree, location=self.loc(block_x, block_y, 4, 0), hide=False)

        # the index for the cubies had to be artificially ranged from 1 to 27, since index 0 caused trouble
        # in the geometry selection process. For the rest of the python interface the labels are ranging from 0 to 26 again
        for i in range(1, 28):
            cubie_selection = make_function(tree, name="CubieSelection" + str(i - 1),
                                            location=self.loc(block_x, block_y, 1, -i + 1), hide=True,
                                            functions={
                                                "select": "idx1," + str(i) + ",=,idx2," + str(i) + ",=,or"
                                            }, inputs=["idx1", "idx2"], outputs=["select"],
                                            scalars=["idx1", "idx2", "select"])
            links.new(attr_cubie_index2.std_out, cubie_selection.inputs["idx1"])
            links.new(attr_face_index2.std_out, cubie_selection.inputs["idx2"])

            sep_geo = SeparateGeometry(tree, location=self.loc(block_x, block_y, 2, -i + 1), domain='FACE',
                                       hide=True, selection=cubie_selection.outputs["select"], geometry_out="Selection")

            rotation = QuaternionToRotation(tree, name="CubieRotation_" + str(i - 1),
                                            location=self.loc(block_x, block_y, 2, -i), hide=False)

            transform = TransformGeometry(tree, location=self.loc(block_x, block_y, 3, -i),
                                          rotation=rotation.std_out,
                                          hide=True)
            create_geometry_line(tree, [join, sep_geo, transform, join2], out=out.inputs[0])

class RubiksCubeGroupStabChain(GeometryNodesModifier):
    def __init__(self, name="RubiksCubeGroupStabChainModifier", **kwargs):
        super().__init__(name, automatic_layout=True, group_input=True, **kwargs)

    def create_node(self, tree, **kwargs):
        """
        Creates a geometry node setup in a Blender node tree.

        The final result is a graphical representation of the stabilizer chain for the Rubik's cube group.


        """
        # output block
        out = self.group_outputs
        ins = self.group_inputs
        links = tree.links

        # extract first element of each orbit from the y-coordiate of the stab chain
        pos = Position(tree)
        index = Index(tree)
        first_orbit_element = make_function(tree, name="FirstOrbitElement",
                                            functions={
                                                "first": "pos_y",
                                                "clean_position": ["pos_x", "0", "pos_z"]
                                            }, inputs=["pos"], outputs=["first", "clean_position"],
                                            scalars=["first"], vectors=["pos", "clean_position"], hide=True)
        links.new(pos.std_out, first_orbit_element.inputs["pos"])
        store_attr_pos = StoredNamedAttribute(tree, data_type="FLOAT_VECTOR", name="PrimaryPosition", value=pos.std_out,
                                              hide=True)
        store_attr_idx = StoredNamedAttribute(tree, data_type="INT", name="ElementIndex", value=index.std_out,
                                              hide=True)

        store_attr_first = StoredNamedAttribute(tree, data_type="BOOLEAN", name="First",
                                                value=first_orbit_element.outputs["first"], hide=True)
        in_geometry = SetPosition(tree, position=first_orbit_element.outputs["clean_position"], hide=True)

        # do the coloring
        material_red = get_material("red")
        material_blue = get_material("blue")
        material_yellow = get_material("yellow")
        material_orange = get_material("orange")
        material_green = get_material("green")
        material_white = get_material("text")

        number_colors = get_from_kwargs(kwargs, "number_colors", ["background", "text"])
        number_outline_colors = get_from_kwargs(kwargs, "number_outline_colors", ["text", "background"])
        material_number_even = get_material(number_colors[0])
        material_number_odd = get_material(number_colors[1])
        material_outline_even = get_material(number_outline_colors[0])
        material_outline_odd = get_material(number_outline_colors[1])

        labels = ["MaterialRed", "MaterialBlue", "MaterialYellow", "MaterialGreen", "MaterialWhite", "MaterialOrange",
                  "MaterialNumberEven", "MaterialNumberOdd", "MaterialOutlineEven", "MaterialOutlineOdd"]

        self.materials = [material_red, material_blue, material_yellow, material_green, material_orange, material_white]
        self.materials.append(material_number_even)
        self.materials.append(material_number_odd)
        self.materials.append(material_outline_even)
        self.materials.append(material_outline_odd)

        input_mats = [InputMaterial(tree, material=m, name=l) for m, l in zip(self.materials, labels)]
        material_dict = {"red": input_mats[0],
                         "blue": input_mats[1],
                         "yellow": input_mats[2],
                         "green": input_mats[3],
                         "orange": input_mats[4],
                         "white": input_mats[5]
                         }

        mat_number_odd = input_mats[6]
        mat_number_even = input_mats[7]
        mat_outline_odd = input_mats[8]
        mat_outline_even = input_mats[9]

        material_first_frame = get_material("custom1")
        self.materials.append(material_first_frame)

        # add faces
        size = InputValue(tree, name="FaceSize", value=0.89, hide=False)
        bevel = InputValue(tree, name="Bevel", value=6.3, hide=False)
        bevel_function = make_function(tree, name="BevelFunction", functions={
            "bevel": "size,bevel,/"
        }, inputs=["size", "bevel"], outputs=["bevel"],
                                       scalars=["size", "bevel"], hide=False)
        links.new(size.std_out, bevel_function.inputs["size"])
        links.new(bevel.std_out, bevel_function.inputs["bevel"])
        quadri = Quadrilateral(tree, width=size.std_out, height=size.std_out, hide=False)
        fillet_curve = FilletCurve(tree, radius=bevel_function.outputs["bevel"], count=10, hide=False)
        fill_cuve = FillCurve(tree, mode='NGONS', hide=False)
        extrude_face = ExtrudeMesh(tree, mode='FACES', offset_scale=1, offset=Vector([0, 0, 0.05]),
                                   hide=False)
        create_geometry_line(tree, [quadri, fillet_curve, fill_cuve, extrude_face])

        # instance faces on dual mesh
        iop = InstanceOnPoints(tree, instance=extrude_face.geometry_out, rotation=[pi / 2, 0, 0])
        ri = RealizeInstances(tree)

        # show label
        pos = Position(tree)
        sep_xyz = SeparateXYZ(tree, vector=pos.std_out)

        range_min = InputInteger(tree, name="DisplayRangeMin", value=-1)
        range_max = InputInteger(tree, name="DisplayRangeMax", value=-1)
        selection_function = make_function(tree, name="SelectionFunction", functions={
            "selection": "idx,rangemax,<,idx,rangemin,>,*"
        }, inputs=["idx", "rangemin", "rangemax"], outputs=["selection"],
                                           scalars=["idx", "rangemax", "rangemin", "selection"], hide=True)
        tree.links.new(index.std_out, selection_function.inputs["idx"])
        tree.links.new(range_max.std_out, selection_function.inputs["rangemax"])
        tree.links.new(range_min.std_out, selection_function.inputs["rangemin"])

        select_geo = SeparateGeometry(tree, selection=selection_function.outputs["selection"])
        foreach_number = ForEachZone(tree, domain="POINT", name="ForEachNumber", hide=False)
        stat = AttributeStatistic(tree, attribute=sep_xyz.std_out, std_out="Max",
                                  geometry=foreach_number.outputs["Element"])

        value2string = ValueToString(tree, value=stat.std_out, hide=True)
        string2curves = StringToCurves(tree, font="Arial Black", string=value2string.std_out, size=0.85, hide=True)
        iop2 = InstanceOnPoints(tree, instance=string2curves.geometry_out, rotation=[pi / 2, 0, 0], hide=True)
        normal = InputNormal(tree)
        fill = FillCurve(tree, mode="NGONS")
        extrude = ExtrudeMesh(tree, offset=normal.std_out, offset_scale=0.1, mode="FACES")

        parity_function = make_function(tree, name="ParityFunction",
                                        functions={
                                            "odd": "idx,2,%"
                                        }, inputs=["idx"], outputs=["odd"],
                                        scalars=["idx", "odd"], vectors=[], hide=True)
        links.new(stat.std_out, parity_function.inputs["idx"])

        material_number = Switch(tree, input_type="MATERIAL", switch=parity_function.outputs["odd"],
                                 true=mat_number_odd.std_out, false=mat_number_even.std_out, hide=True)
        set_mat_number = SetMaterial(tree, material=material_number.std_out, hide=True)
        join = JoinGeometry(tree)
        foreach_number.create_geometry_line([iop2, fill, extrude, set_mat_number, join])

        wireframe = CurveWireFrame(tree, radius=0.02, resolution=8)
        set_pos = SetPosition(tree, offset=Vector([0, -0.09, 0]))
        material_outline = Switch(tree, input_type="MATERIAL", switch=parity_function.outputs["odd"],
                                  true=mat_outline_odd.std_out, false=mat_outline_even.std_out)
        set_mat_outline = SetMaterial(tree, material=material_outline.std_out, hide=True)
        create_geometry_line(tree, [iop2, wireframe, set_pos, set_mat_outline, join])

        minus_one = MathNode(tree, name="MinusOne", operation="SUBTRACT", inputs0=stat.std_out, inputs1=1, hide=False)
        index_switch = IndexSwitch(tree, data_type="MATERIAL", index=minus_one.std_out)
        for idx, color in face_color_dict.items():
            index_switch.add_item(socket=material_dict[color].std_out)

        set_material = SetMaterial(tree, material=index_switch.std_out, hide=False)
        to_instance = GeometryToInstance(tree)
        foreach_number.create_geometry_line([iop, ri, set_material, join, to_instance])

        # create geometry for first element frames
        first = NamedAttribute(tree, name="First", data_type="BOOLEAN", hide=True)
        frame = Quadrilateral(tree, width=1.25, height=1.25, hide=True)
        frame_wire = CurveWireFrame(tree, radius=0.12, resolution=8, hide=True)
        iop_first = InstanceOnPoints(tree, selection=first.std_out, rotation=[pi / 2, 0, 0], hide=True)
        set_frame_material = SetMaterial(tree, material=material_first_frame, hide=True)

        create_geometry_line(tree, [frame, frame_wire], out=iop_first.inputs["Instance"])
        create_geometry_line(tree, [iop_first, set_frame_material, join], ins=foreach_number.outputs["Element"])

        # include rotation of individual element

        attr_idx = NamedAttribute(tree, name="ElementIndex", data_type="INT", hide=True)
        attr_pos = NamedAttribute(tree, name="PrimaryPosition", data_type="FLOAT_VECTOR", hide=True)
        select_element = InputInteger(tree, name="ElementIndexSelector", value=-1)
        rot_angle = InputValue(tree, name="RotationAngle", data_type="FLOAT", hide=True)
        combine_xyz = CombineXYZ(tree, z=rot_angle.std_out, hide=True)

        selection_and_pivot_function = make_function(tree, name="SelectionPivotFunction",
                                                     functions={
                                                         "selection": "elemIdx,idx,=",
                                                         "pivot": ["prim_x,0.5,-", "prim_y", "prim_z"]
                                                     }, inputs=["prim", "elemIdx", "idx"],
                                                     outputs=["selection", "pivot"],
                                                     scalars=["elemIdx", "idx", "selection"], vectors=["pivot", "prim"],
                                                     hide=True)
        tree.links.new(attr_idx.std_out, selection_and_pivot_function.inputs["elemIdx"])
        tree.links.new(attr_pos.std_out, selection_and_pivot_function.inputs["prim"])
        tree.links.new(select_element.std_out, selection_and_pivot_function.inputs["idx"])

        rot_instance = RotateInstances(tree, selection=selection_and_pivot_function.outputs["selection"],
                                       hide=True, pivot_point=selection_and_pivot_function.outputs["pivot"],
                                       rotation=combine_xyz.std_out)

        translate = TranslateToCenterNode(tree)
        links.new(in_geometry.outputs["Geometry"], translate.inputs["SourceGeometry"])
        create_geometry_line(tree,
                             [store_attr_first, in_geometry, store_attr_pos, store_attr_idx, select_geo, foreach_number,
                              rot_instance, translate], ins=ins.outputs["Geometry"], out=out.inputs[0])

class TransformationModifier(GeometryNodesModifier):
    def __init__(self, name="TransformationModifier", label=None, **kwargs):
        """
        A geometry node setup that visualizes the isometries of a cube

        """
        self.label = label
        super().__init__(name, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        # setup basic geometry
        cube_node = CubeMesh(tree)

        wireframe = WireFrame(tree)
        edge_color = get_from_kwargs(kwargs, 'edge_material', 'drawing')
        edge_material = get_material(edge_color, **kwargs)
        self.materials.append(edge_material)
        set_edge_material = SetMaterial(tree, material=edge_material)
        pos = Position(tree)

        join = JoinGeometry(tree)

        # setup vertex labels
        selected_corners = get_from_kwargs(kwargs, 'selected_corners', None)
        rotate_corners = get_from_kwargs(kwargs, 'corner_rotation_euler', [0, 0, 0])
        corner_rotation = InputVector(tree, name="CornerRotation", value=rotate_corners, hide=True)

        if selected_corners is not None:
            index = Index(tree)
            index_switch = IndexSwitch(tree, data_type="INT", index=index.std_out, name="LabelSwitch")
            for i in range(8):
                if i + 1 in selected_corners:
                    index_switch.add_item(socket=1)
                else:
                    index_switch.add_item(socket=0)

            foreach = ForEachZone(tree, geometry=cube_node.geometry_out, selection=index_switch.std_out, hide=True)
        else:
            foreach = ForEachZone(tree, geometry=cube_node.geometry_out, hide=True)

        addOne = MathNode(tree, operation="ADD", inputs0=foreach.outputs["Index"], inputs1=1, hide=True)
        value2String = ValueToString(tree, value=addOne.std_out, hide=True)
        string2Curves = StringToCurves(tree, name="CornerLabels", string=value2String.std_out, size=0.3, hide=True)
        fill_curve = FillCurve(tree, mode="NGONS", hide=True)
        normal = InputNormal(tree, hide=True)
        extrude = ExtrudeMesh(tree, mode="FACES", offset=normal.std_out, offset_scale=0.05)

        axes2Rotation = AxesToRotation(tree, primary_axis='Y', secondary_axis='X',
                                       primary_direction=[0, 0, 1], secondary_direction=[1, 0, 0],
                                       hide=True)  # make fonts up and from left  to right
        join_string_mesh = JoinGeometry(tree, hide=True)
        font_color = get_from_kwargs(kwargs, 'font_material', 'text')
        font_material = get_material(font_color, **kwargs)
        self.materials.append(font_material)
        set_font_material = SetMaterial(tree, material=font_material)

        join_string_and_backplate = JoinGeometry(tree, hide=True)
        iop = InstanceOnPoints(tree, rotation=axes2Rotation.std_out, hide=True)
        create_geometry_line(tree, [join_string_mesh, set_font_material, join_string_and_backplate],
                             out=iop.inputs["Instance"])
        create_geometry_line(tree, [fill_curve, join_string_mesh])
        create_geometry_line(tree, [string2Curves, fill_curve, extrude, join_string_mesh])

        circle = CurveCircle(tree, radius=0.15, resolution=16, hide=True)
        fill_circle = FillCurve(tree, mode="NGONS", hide=True)
        circle_material = get_material("gray_1", **kwargs)
        self.materials.append(circle_material)
        set_circle_material = SetMaterial(tree, material=circle_material)
        create_geometry_line(tree, [circle, fill_circle, set_circle_material, join_string_and_backplate])

        scale_value = InputValue(tree, name="LabelPos", value=1.25, hide=True)
        scale = VectorMath(tree, operation="SCALE", inputs0=pos.std_out, float_input=scale_value.std_out, hide=True)
        set_pos = SetPosition(tree, position=scale.std_out, hide=True)
        rotate_instance = RotateInstances(tree, pivot_point=scale.std_out, hide=True, rotation=corner_rotation.std_out,
                                          local_space=False)
        foreach.create_geometry_line([iop, set_pos, rotate_instance])
        create_geometry_line(tree, [foreach, join])

        # create faces
        index = Index(tree)
        store_face_index = StoredNamedAttribute(tree, data_type="INT", domain="FACE", name="FaceIndex",
                                                value=index.std_out, hide=True)
        face_color = get_from_kwargs(kwargs, 'face_material', 'drawing')
        face_material = get_material(face_color, **kwargs)
        set_face_material = SetMaterial(tree, material=face_material)
        self.materials.append(face_material)
        create_geometry_line(tree, [cube_node, store_face_index, set_face_material, join])

        # transformation
        # 10 reflections
        last_pos = pos.std_out
        for r in range(10):
            normal = InputVector(tree, value=[0, 0, 0], name="Normal" + str(r), hide=True)
            coxeter = CoxeterReflectionNode(tree, position=last_pos, normal=normal.std_out,
                                            name="CoxeterReflection" + str(r), hide=True)
            last_pos = coxeter.std_out

        # 10 rotations
        for r in range(10):
            quaternion = QuaternionToRotation(tree, name="Rotation" + str(r), hide=True)
            rotateVector = RotateVector(tree, vector=last_pos, rotation=quaternion.std_out,
                                        name="RotateVector" + str(r), hide=True)
            last_pos = rotateVector.std_out

        set_pos = SetPosition(tree, position=last_pos)
        create_geometry_line(tree, [cube_node, wireframe, set_edge_material, join])
        join2 = JoinGeometry(tree, hide=True)  # additional join for labels that are added after transformation
        create_geometry_line(tree, [join, set_pos, join2])

        # create mirror grid
        radius = InputValue(tree, name="Radius", value=0, hide=True)
        grid = Grid(tree, size_x=1.75, size_y=1.75, vertices_x=15, vertices_y=15)
        grid_color = get_from_kwargs(kwargs, 'grid_material', 'custom1')
        grid_material = get_material(grid_color, **kwargs)
        self.materials.append(grid_material)
        wireframe2 = WireFrame(tree, radius=radius.std_out)
        set_grid_material = SetMaterial(tree, material=grid_material)
        grid_rotation = InputVector(tree, name="GridRotation", value=Vector())
        grid_transform = TransformGeometry(tree, name="GridTransformation",
                                           geometry=cube_node.geometry_out,
                                           rotation=grid_rotation.std_out, hide=False)
        create_geometry_line(tree, [grid, wireframe2, set_grid_material, grid_transform, join])

        # allow global transformation
        rotation = InputVector(tree, name="GlobalRotation", value=Vector(), hide=True)
        rotation_transform = TransformGeometry(tree, name="RotationTransformation", rotation=rotation.std_out,
                                               hide=False)
        rotation_join = JoinGeometry(tree, hide=True)

        # create label
        label_sep = get_from_kwargs(kwargs, "label_separation", -0.9)
        if self.label is not None:
            label_info = ObjectInfo(tree, name="Label", object=self.label, hide=True)
            set_label_pos = SetPosition(tree, name="LabelPosition", offset=[0, 0, label_sep], hide=True)
            create_geometry_line(tree, [label_info, set_label_pos, rotation_join])

        create_geometry_line(tree, [join2, rotation_transform, rotation_join], out=out.inputs[0])

class TransformationSphereModifier(GeometryNodesModifier):
    def __init__(self, name="TransformationSphereModifier", label=None, **kwargs):
        """
        A geometry node setup that visualizes the isometries of a cube blown up to a sphere

        """
        self.label = label
        super().__init__(name, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        # setup basic geometry
        cube_node = CubeMesh(tree)
        pos = Position(tree)

        # create faces
        index = Index(tree)
        store_face_index = StoredNamedAttribute(tree, data_type="INT", domain="FACE", name="FaceIndex",
                                                value=index.std_out, hide=True)
        face_color = get_from_kwargs(kwargs, 'face_material', 'drawing')
        face_material = get_material(face_color, **kwargs)
        set_face_material = SetMaterial(tree, material=face_material)
        self.materials.append(face_material)

        # turn into sphere

        sub_div = SubdivideMesh(tree, level=4)
        blow_function = make_function(tree, functions={
            "Position": "pos,1,pos,length,/,scale"
        }, inputs=["pos"], outputs=["Position"], vectors=["pos", "Position"], hide=True, name="BlowUp")
        tree.links.new(pos.std_out, blow_function.inputs["pos"])

        set_pos2 = SetPosition(tree, position=blow_function.outputs["Position"])

        # transformation
        # 10 reflections
        last_pos = pos.std_out
        for r in range(10):
            normal = InputVector(tree, value=[0, 0, 0], name="Normal" + str(r), hide=True)
            coxeter = CoxeterReflectionNode(tree, position=last_pos, normal=normal.std_out,
                                            name="CoxeterReflection" + str(r), hide=True)
            last_pos = coxeter.std_out

        # 10 rotations
        for r in range(10):
            quaternion = QuaternionToRotation(tree, name="Rotation" + str(r), hide=True)
            rotateVector = RotateVector(tree, vector=last_pos, rotation=quaternion.std_out,
                                        name="RotateVector" + str(r), hide=True)
            last_pos = rotateVector.std_out

        set_pos = SetPosition(tree, position=last_pos)

        # allow global transformation
        rotation = InputVector(tree, name="GlobalRotation", value=Vector(), hide=True)
        rotation_transform = TransformGeometry(tree, name="RotationTransformation", rotation=rotation.std_out,
                                               hide=False)
        smooth = SetShadeSmooth(tree)

        create_geometry_line(tree, [cube_node, store_face_index, set_face_material,
                                    sub_div, set_pos2, set_pos, rotation_transform, smooth], out=out.inputs[0])

class WordTreeModifier(GeometryNodesModifier):
    def __init__(self, name="TreeModifier", words=[], instances=[], **kwargs):
        """
        A geometry node setup that visualizes the structure of a group tree

        """
        self.words = words
        self.instances = instances
        super().__init__(name, automatic_layout=True, group_input=True, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        input = self.group_inputs
        links = tree.links

        # setup basic geometry
        join = JoinGeometry(tree, hide=True)

        # create the edges
        index = Index(tree)
        edge_vertices = EdgeVertices(tree, hide=True)
        edge_length = InputValue(tree, name="EdgeLength", value=0.5, hide=True)
        edge_radius = InputValue(tree, name="EdgeRadius", value=0.25, hide=True)
        transform_edges = make_function(tree, name="TransformEdges",
                                        functions={
                                            "start": "p1,p2,add,0.5,scale,p1,p2,sub,l,2,/,scale,sub",
                                            "end": "p1,p2,sub,l,2,/,scale,p1,p2,add,0.5,scale,add",
                                            "switch": "idx,2,%"
                                        }, inputs=["p1", "p2", "l", "idx"], outputs=["start", "end", "switch"],
                                        scalars=["switch", "l", "idx"], vectors=["p1", "p2", "start", "end"], hide=True)
        links.new(index.std_out, transform_edges.inputs["idx"])
        links.new(edge_length.std_out, transform_edges.inputs["l"])
        links.new(edge_vertices.outputs["Position 1"], transform_edges.inputs["p1"])
        links.new(edge_vertices.outputs["Position 2"], transform_edges.inputs["p2"])

        for_each_edge = ForEachZone(tree, domain="EDGE", name="ForEachEdge")
        switch = Switch(tree, name="EdgeSwitch", input_type="VECTOR", switch=transform_edges.outputs["switch"],
                        true=transform_edges.outputs["start"], false=transform_edges.outputs["end"])
        set_edge_pos = SetPosition(tree, position=switch.std_out)
        for_each_edge.create_geometry_line([set_edge_pos])
        subdiv_mesh = SubdivideMesh(tree, level=4)  # increase resolution for smooth growing of the edges
        wireframe = WireFrame(tree, radius=edge_radius.std_out, resolution=10)

        # delete geometry depending on the progress
        position = Position(tree, name="MaxPosition")
        statistics = AttributeStatistic(tree, data_type="FLOAT_VECTOR", domain="POINT", name="MaxPosition",
                                        std_out="Max",
                                        geometry=input.outputs[0],
                                        attribute=position.std_out, hide=True)
        progress = InputValue(tree, name="Progress", value=0, hide=True)
        progress_selection = make_function(tree, name="ProgressSelector",
                                           functions={
                                               "selection": "pos_x,p,max_x,*,<,not"
                                           }, inputs=["p", "pos", "max"], outputs=["selection"],
                                           scalars=["p", "selection"], vectors=["pos", "max"])
        del_geo = DeleteGeometry(tree, selection=progress_selection.outputs["selection"], hide=True)
        links.new(position.std_out, progress_selection.inputs["pos"])
        links.new(statistics.std_out, progress_selection.inputs["max"])
        links.new(progress.std_out, progress_selection.inputs["p"])

        create_geometry_line(tree, [for_each_edge, subdiv_mesh, wireframe, del_geo, join], ins=input.outputs[0])

        # create instances and selector function
        object_infos = []
        for word, instance in zip(self.words, self.instances):
            object_infos.append(ObjectInfo(tree, name=word, object=instance.ref_obj))

        # create for each zone
        rotation = InputVector(tree, name="Rotation", value=Vector())
        for_each_point = ForEachZone(tree, domain="POINT", name="InstanceSelector")
        outs = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15",
                "p16", "p17", "p18",
                "p19", "p20", "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p30", "p31", "p32", "p33",
                "p34", "p35", "p36",
                "p37", "p38", "p39", "p40", "p41", "p42", "p43", "p44", "p45", "p46", "p47"]
        instance_selection = make_function(tree, name="InstanceSelection",
                                           functions={
                                               "p0": "idx,0,=", "p1": "idx,1,=", "p2": "idx,2,=", "p3": "idx,3,=",
                                               "p4": "idx,4,=", "p5": "idx,5,=", "p6": "idx,6,=", "p7": "idx,7,=",
                                               "p8": "idx,8,=", "p9": "idx,9,=", "p10": "idx,10,=", "p11": "idx,11,=",
                                               "p12": "idx,12,=", "p13": "idx,13,=", "p14": "idx,14,=",
                                               "p15": "idx,15,=",
                                               "p16": "idx,16,=", "p17": "idx,17,=", "p18": "idx,18,=",
                                               "p19": "idx,19,=", "p20": "idx,20,=", "p21": "idx,21,=",
                                               "p22": "idx,22,=", "p23": "idx,23,=",
                                               "p24": "idx,24,=", "p25": "idx,25,=", "p26": "idx,26,=",
                                               "p27": "idx,27,=", "p28": "idx,28,=", "p29": "idx,29,=",
                                               "p30": "idx,30,=", "p31": "idx,31,=",
                                               "p32": "idx,32,=", "p33": "idx,33,=", "p34": "idx,34,=",
                                               "p35": "idx,35,=", "p36": "idx,36,=", "p37": "idx,37,=",
                                               "p38": "idx,38,=", "p39": "idx,39,=",
                                               "p40": "idx,40,=", "p41": "idx,41,=", "p42": "idx,42,=",
                                               "p43": "idx,43,=", "p44": "idx,44,=", "p45": "idx,45,=",
                                               "p46": "idx,46,=", "p47": "idx,47,="
                                           }, inputs=["idx"], outputs=outs,
                                           scalars=["idx"] + outs)

        last_out = None
        for i in range(0, len(object_infos)):
            switch = Switch(tree, name="InstanceSwitch" + str(i), input_type="GEOMETRY",
                            switch=instance_selection.outputs["p" + str(i)],
                            false=last_out, true=object_infos[i].geometry_out)
            last_out = switch.std_out

        iop2 = InstanceOnPoints(tree, instance=last_out,
                                rotation=rotation.std_out, scale=[8] * 3, hide=False)
        for_each_point.create_geometry_line([iop2])
        links.new(for_each_point.outputs["Index"], instance_selection.inputs["idx"])
        location = get_from_kwargs(kwargs, "location", Vector())

        # create progress selector for instances

        max_instance = InputInteger(tree, name="MaxInstance", value=0, hide=True)
        index = Index(tree)
        instance_selection = make_function(tree, name="InstanceSelector",
                                           functions={
                                               "selection": "idx,mx,<,not"
                                           }, inputs=["idx", "mx"], outputs=["selection"],
                                           scalars=["idx", "mx", "selection"])
        links.new(max_instance.std_out, instance_selection.inputs["mx"])
        links.new(index.std_out, instance_selection.inputs["idx"])
        del_instance_geo = DeleteGeometry(tree, domain="INSTANCE", selection=instance_selection.outputs["selection"],
                                          hide=True)

        global_transform = TransformGeometry(tree, name="GlobalTransform", translation=location, hide=False)
        create_geometry_line(tree, [for_each_point, del_instance_geo, join, global_transform], out=out.inputs[0],
                             ins=input.outputs[0])

class PermutationModifier(GeometryNodesModifier):
    def __init__(self, name="PermutationModifier", n=8, number_of_cycles=5, **kwargs):
        """
        A geometry node setup that visualizes the structure of a permutation for up to 9 elements
        """
        self.n = n
        self.number_of_cycles = number_of_cycles
        super().__init__(name, automatic_layout=True, group_input=False, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        # join geometry
        join = JoinGeometry(tree, hide=True)

        # start from n points

        points = Points(tree, count=self.n, hide=True)

        # create boxes

        foreach_points = ForEachZone(tree, domain="POINT", name="ForEachPointBoxes", hide=False)

        box = Quadrilateral(tree, mode="RECTANGLE", width=1, height=1, hide=True)
        iop = InstanceOnPoints(tree, instance=box.geometry_out, hide=True)
        scale = ScaleInstances(tree, scale=[0.9] * 3, hide=True)
        box_pos = SetPosition(tree, offset=[0, 0, 0.05], hide=True)
        quad = Quadrilateral(tree, mode="RECTANGLE", width=0.1, height=0.1, hide=True)
        curve_to_mesh = CurveToMesh(tree, profile_curve=quad.geometry_out, hide=True)

        combine_xyz = CombineXYZ(tree, x=foreach_points.index, hide=True)
        transform_boxes = TransformGeometry(tree, name="TransformBoxes", translation=combine_xyz.std_out,
                                            rotation=[pi / 2, 0, 0], hide=True)
        foreach_points.create_geometry_line([iop, scale, box_pos, curve_to_mesh, transform_boxes])
        frame_color = get_from_kwargs(kwargs, "frame_color", 'drawing')
        frame_material = get_material(frame_color, **kwargs)
        self.materials.append(frame_material)
        frame_material_node = SetMaterial(tree, material=frame_material)
        create_geometry_line(tree, [points, foreach_points, frame_material_node, join])

        # create numbers and attributes

        foreach_number = ForEachZone(tree, domain="POINT", name="ForEachNumber", hide=False)
        inc = MathNode(tree, operation="ADD", inputs0=foreach_number.index, inputs1=1, hide=True)
        value2string = ValueToString(tree, value=inc.std_out, hide=True)
        string2curves = StringToCurves(tree, string=value2string.std_out, size=1, hide=True)
        iop2 = InstanceOnPoints(tree, instance=string2curves.geometry_out, hide=True)
        attr_idx = StoredNamedAttribute(tree, data_type="INT", domain="INSTANCE", name="Index", value=inc.std_out,
                                        hide=False)

        # create a unique prime identifier for each number
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        value_nodes = [InputInteger(tree, integer=primes[i]) for i in range(self.n)]

        index_switch = IndexSwitch(tree, data_type="INT", index=foreach_number.index, hide=False)
        [index_switch.add_item(value_node.std_out) for value_node in value_nodes]

        attr_prime = StoredNamedAttribute(tree, data_type="INT", domain="INSTANCE", name="Prime",
                                          value=index_switch.std_out, hide=False)

        combine_xyz2 = CombineXYZ(tree, x=foreach_number.index, hide=True)
        transform_numbers = TransformGeometry(tree, name="TransformNumbers", rotation=[pi / 2, 0, 0],
                                              translation=combine_xyz2.std_out, hide=True)
        position = Position(tree, name="Position", hide=True)
        attr_position = StoredNamedAttribute(tree, data_type="FLOAT_VECTOR", domain="INSTANCE", name="Position",
                                             value=position.std_out, hide=False)
        foreach_number.create_geometry_line([iop2, attr_idx, attr_prime, transform_numbers, attr_position])

        create_geometry_line(tree, [points, foreach_number])

        # create cycles
        cycles = []
        for i in range(self.number_of_cycles):
            cycle = InputInteger(tree, name="Cycle" + str(i + 1), integer=0, hide=True)
            cycle_length = InputInteger(tree, name="CycleLength" + str(i + 1), integer=0, hide=True)
            displacement = InputValue(tree, name="Displacement" + str(i + 1), value=0, hide=True)
            up_mover = InputInteger(tree, name="UpMover" + str(i + 1), integer=0, hide=True)
            down_mover = InputInteger(tree, name="DownMover" + str(i + 1), integer=0, hide=True)
            movers = []
            for j in range(self.n):
                movers.append(InputValue(tree, name="Mover" + str(i + 1) + "_" + str(j + 1), value=0, hide=True))

            cycles.append(CycleNode(tree, name="Cycle" + str(i + 1), max_length=self.n, hide=True))
            links.new(cycle.std_out, cycles[i].inputs["Cycle"])
            links.new(cycle_length.std_out, cycles[i].inputs["CycleLength"])
            links.new(displacement.std_out, cycles[i].inputs["Displacement"])
            links.new(up_mover.std_out, cycles[i].inputs["UpMover"])
            links.new(down_mover.std_out, cycles[i].inputs["DownMover"])
            for j in range(self.n):
                links.new(movers[j].std_out, cycles[i].inputs["Mover" + str(j + 1)])

        # extrude letters
        fill_curve = FillCurve(tree, mode="NGONS", hide=True)
        combine_xyz = CombineXYZ(tree, z=0.1, hide=True)
        extrude = ExtrudeMesh(tree, mode="FACES", offset=combine_xyz.std_out, offset_scale=1, hide=True)
        extrude_join = JoinGeometry(tree, hide=True)
        create_geometry_line(tree, [foreach_number] + cycles + [fill_curve, extrude_join])
        number_color = get_from_kwargs(kwargs, 'number_color', 'text')
        material = get_material(number_color, **kwargs)
        self.materials.append(material)
        number_material_node = SetMaterial(tree, material=material)
        centering = TransformGeometry(tree, translation=Vector([(-self.n + 1) / 2, 0, 0]))

        create_geometry_line(tree, [fill_curve, extrude, extrude_join, number_material_node,
                                    join, centering],
                             out=out.inputs[0])

class QuadModifier(GeometryNodesModifier):
    def __init__(self, name="QuadrialateralModifier", mode="RECTANGLE", **kwargs):
        """
        A geometry node setup that visualizes a quadrilateral
        """
        self.mode = mode
        super().__init__(name, automatic_layout=True, group_input=False, **kwargs)

    def create_node(self, tree, **kwargs):
        out = self.group_outputs
        links = tree.links

        # join geometry
        join = JoinGeometry(tree, hide=True)

        width = get_from_kwargs(kwargs, "width", 2)
        height = get_from_kwargs(kwargs, "height", 2)
        quad_node = Quadrilateral(tree, mode=self.mode, width=width, height=height, hide=True)

        resolution = get_from_kwargs(kwargs, "resolution", 100)
        res_node = InputInteger(tree, name="Resolution", integer=resolution, hide=True)
        idx = Index(tree)

        grow = InputValue(tree, name="Grow", value=0, hide=True)

        grow_function = make_function(tree, name="GrowFunction",
                                      functions={
                                          "selection": "idx,res,/,grow,>,not"
                                      }, inputs=["res", "idx", "grow"], outputs=["selection"],
                                      scalars=["res", "idx", "grow", "selection"], vectors=[], hide=True)
        links.new(res_node.std_out, grow_function.inputs["res"])
        links.new(idx.std_out, grow_function.inputs["idx"])
        links.new(grow.std_out, grow_function.inputs["grow"])

        resample = ResampleCurve(tree, count=resolution, hide=True)
        trim = TrimCurve(tree, hide=True)
        select_geo = SeparateGeometry(tree, selection=grow_function.outputs["selection"], hide=True)

        thickness = get_from_kwargs(kwargs, "thickness", 1)
        wireframe = CurveWireFrame(tree, radius=thickness * 0.02, hide=True)

        geo_location = get_from_kwargs(kwargs, "geo_location", Vector())
        normal = get_from_kwargs(kwargs, "normal", [0, 0, 0])
        material = get_material(get_from_kwargs(kwargs, "color", 'drawing'), **kwargs)
        self.materials.append(material)
        quad_material_node = SetMaterial(tree, material=material)
        from_normal = AlignRotationToVector(tree, vector=normal, hide=True)
        global_transform = TransformGeometry(tree, name="GlobalTransform", translation=geo_location,
                                             rotation=from_normal.std_out, hide=False)

        create_geometry_line(tree, [quad_node, resample, trim, select_geo, wireframe, join, quad_material_node,
                                    global_transform], out=out.inputs[0])

class SingmasterDisplayModifier(GeometryNodesModifier):
    def __init__(self, **kwargs):
        super().__init__(get_from_kwargs(kwargs, 'name', "SingmasterDisplayModifier"),
                         group_input=False, group_output=False, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        create_from_xml(tree, "singmaster_display_node", **kwargs)

# Billiard video

class BilliardsTableModifier(GeometryNodesModifier):
    def __init__(self, **kwargs):
        super().__init__(get_from_kwargs(kwargs, 'name', "BilliardsTableModifier"),
                         group_input=False, group_output=False, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        create_from_xml(tree, "GeoBilliardsTable_node", **kwargs)

class BilliardsBallModifier(GeometryNodesModifier):
    def __init__(self, **kwargs):
        self.subdivisions = get_from_kwargs(kwargs, "subdivisions", 5)
        super().__init__(get_from_kwargs(kwargs, 'name', "BilliardsBallModifier"),
                         group_input=False, group_output=False, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        create_from_xml(tree, "GeoBilliardsBall_node", **kwargs)
        subdivisions_node = ibpy.get_geometry_node_from_modifier(self, label="SubdivisionsValue")
        ibpy.change_default_integer(subdivisions_node, from_value=5, to_value=self.subdivisions,
                                    begin_time=0, transition_time=0)

class BilliardBallRealModifier(GeometryNodesModifier):
    def __init__(self, **kwargs):
        self.subdivisions = get_from_kwargs(kwargs, "subdivisions", 5)
        super().__init__(get_from_kwargs(kwargs, 'name', "BilliardsBallModifier"),
                         group_input=False, group_output=False, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        create_from_xml(tree, "GeoBilliardsBallReal_node", **kwargs)

class IlluminationModifier(GeometryNodesModifier):
    def __init__(self, light_instance, **kwargs):
        self.light_instance = light_instance
        super().__init__(get_from_kwargs(kwargs, 'name', "IlluminationModifier"),
                         group_input=True, group_output=True, automatic_layout=True, **kwargs)

    def create_node(self, tree, **kwargs):
        ins = self.group_inputs
        out = self.group_outputs

        light = ObjectInfo(tree, transform_space="ORIGINAL",
                           as_instance=True,
                           object=self.light_instance)

        for_each = ForEachZone(tree)
        normal = InputNormal(tree)
        axes_to_rotation = AxesToRotation(tree, primary_axis='Z', secondary_axis='X', primary_direction=normal.std_out)
        iop = InstanceOnPoints(tree, instance=light.geometry_out, rotation=axes_to_rotation.std_out)

        for_each.create_geometry_line([iop])

        join = JoinGeometry(tree)
        tree.links.new(for_each.foreach_output.outputs[0], join.geometry_in)
        create_geometry_line(tree, [for_each, join], ins=ins.outputs[0], out=out.inputs[0])

class ScoreTableModifier(GeometryNodesModifier):
    def __init__(self, **kwargs):

        super().__init__(get_from_kwargs(kwargs, 'name', "ScoreTableModifier"),
                         group_input=False, group_output=False, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):
        create_from_xml(tree, "GeoBilliardsScoreTable_node", **kwargs)

        background = get_from_kwargs(kwargs, "background_material", "background")
        background_material_node = ibpy.get_geometry_node_from_modifier(self, label="MaterialBackground")
        background_material = get_material(background, **kwargs)
        background_material_node.material = background_material
        self.materials.append(background_material)

        materials = ["background_material", "ur_material", "ul_material", "br_material", "font_material"]
        defaults = ["background", "example", "joker", "br_material", "important", "background"]
        slots = ["MaterialBackground", "MaterialUR", "MaterialUL", "MaterialBR", "MaterialFont"]
        for mat, default, slot in zip(materials, defaults, slots):
            material_string = get_from_kwargs(kwargs, mat, default)
            material_node = ibpy.get_geometry_node_from_modifier(self, label=slot)
            material = get_material(material_string, **kwargs)
            material_node.material = material
            self.materials.append(material)

        index_selector_node = ibpy.get_geometry_node_from_modifier(self, label="IndexSelector")

        width = get_from_kwargs(kwargs, "width", 3)
        height = get_from_kwargs(kwargs, "height", 3)
        for w in range(0, width):
            for h in range(0, height):
                ibpy.add_item_to_switch(index_selector_node, w * height + h, False, tree)

class ReflectableBilliardPaperModifier(GeometryNodesModifier):
    def __init__(self, width=10, height=10, **kwargs):
        self.width = width
        self.height = height
        super().__init__(get_from_kwargs(kwargs, "name", "BilliardPaperModifier"),
                         group_input=False, group_output=True, automatic_layout=False, **kwargs)

    def create_node(self, tree, **kwargs):

        next = -10

        paper = Grid(tree, location=[next, 0],
                     size_x=self.width, size_y=self.height, vertices_x=self.width + 1, vertices_y=self.height + 1)
        next += 1
        wireframe = WireFrame(tree, location=[next, 1])
        next += 1
        grid_color = get_from_kwargs(kwargs, "grid_color", "example")
        grid_material = get_texture(grid_color, emission=0.5, **kwargs)
        self.materials.append(grid_material)
        grid_material_node = SetMaterial(tree, label="MaterialGrid", material=grid_material,
                                         location=[next, 1], hide=True)
        next += 1
        join = JoinGeometry(tree, location=[next, 0])
        next += 1
        geo_to_instance = GeometryToInstance(tree, location=[next, 1], hide=True)
        next -= 2
        paper_color = get_from_kwargs(kwargs, "paper_color", "gray_1")
        material = get_texture(paper_color, **kwargs)
        self.materials.append(material)
        material_node = SetMaterial(tree, material=material, location=[next, 0], hide=True)
        next += 1

        create_geometry_line(tree, [paper, material_node, join])
        create_geometry_line(tree, [paper, wireframe, grid_material_node, join, geo_to_instance])

        # create target squares
        target_square_scale = get_from_kwargs(kwargs, "target_square_scale", 1)
        join_squares = JoinGeometry(tree, location=[next, 3], hide=True)
        create_geometry_line(tree, [join_squares, join])
        colors = ["custom1", "joker", "important"]
        position = [
            [self.width / 2 - 0.5 * target_square_scale, self.height / 2 - 0.5 * target_square_scale, 0.0125],
            [-self.width / 2 + 0.5 * target_square_scale, self.height / 2 - 0.5 * target_square_scale, 0.0125],
            [self.width / 2 - 0.5 * target_square_scale, -self.height / 2 + 0.5 * target_square_scale, 0.0125],
        ]

        for i in range(3):
            inext = next - 5
            square = Grid(tree, location=[inext, 2 + i], size_x=target_square_scale, size_y=target_square_scale,
                          vertices_x=2, vertices_y=2)
            inext += 1
            square_material = get_texture(colors[i])
            self.materials.append(square_material)
            square_material_node = SetMaterial(tree, label="MaterialSquare" + str(i), material=square_material,
                                               location=[inext, 2 + i], hide=True)
            inext += 1
            square_pos = SetPosition(tree, location=[inext, 2 + i], offset=position[i], hide=True)
            inext += 1
            square_extrude = ExtrudeMesh(tree, location=[inext, 2 + i], mode="FACES",
                                         offset=Vector([0, 0, -0.025]), hide=True)
            inext += 1
            create_geometry_line(tree, [square, square_material_node, square_pos, square_extrude, join_squares])
            create_geometry_line(tree, [square_pos, join_squares])

        # final join
        join2 = JoinGeometry(tree, location=[-next - 1, 0])
        create_geometry_line(tree, [join, join2], out=self.group_outputs.inputs["Geometry"])
        self.group_outputs.location = (-next, 0)

        # create reflections
        minimum = min(self.width, self.height)
        maximum = max(self.width, self.height)
        progress = minimum
        lcm = np.lcm(self.width, self.height)
        nw = self.width
        nh = self.height
        last = geo_to_instance
        delta = maximum - minimum  # delta captures the largest possible distance that can be moved in the next move
        # print(progress,delta,minimum)

        counter = 0
        while progress < lcm:
            inext = next + counter
            angle = InputValue(tree, value=0, label="Rotation" + str(counter), location=[inext, counter + 1.5],
                               hide=True)
            inext += 1
            counter += 1
            if nh < nw:  # reflect horizontally
                combine = CombineXYZ(tree, location=[inext, counter + 1], x=angle.std_out, y=0, z=0, hide=True)
                inext += 1
                rotate_instance = RotateInstances(tree, location=[inext, counter + 1.5], rotation=combine.std_out,
                                                  pivot_point=[0, nh - self.height / 2, 0], local_space=False)
                inext += 1
                nh += self.height
                step = min(minimum, delta)
                progress += step
                delta = max(delta - minimum, self.height - step)  # maximum travel distance in the next step

            else:  # reflect vertically
                neg = MathNode(tree, operation="MULTIPLY", inputs0=angle.std_out, inputs1=-1,
                               location=[inext, counter + 0.5])
                combine = CombineXYZ(tree, location=[inext, counter + 1], x=0, y=neg.std_out, z=0, hide=True)
                inext += 1
                rotate_instance = RotateInstances(tree, location=[inext, counter + 1.5], rotation=combine.std_out,
                                                  pivot_point=[nw - self.width / 2, 0, 0], local_space=False)
                inext += 1
                nw += self.width
                step = min(minimum, delta)
                progress += step
                delta = max(delta - minimum, self.width - step)  # maximum travel distance in the next step

            # print(progress,delta,step,nw,nh)

            create_geometry_line(tree, [last, rotate_instance, join2])
            last = rotate_instance

# new videos

class CustomUnfoldModifier(GeometryNodesModifier):
    def __init__(self, name="UnfoldModifier",**kwargs):
        super().__init__(name, automatic_layout=True,group_output=True,group_input=True, **kwargs)

    def create_node(self,tree,**kwargs):
        out = self.group_outputs
        ins = self.group_inputs
        links = tree.links

        # reindex faces
        pos = Position(tree,hide=True)
        re_index_function = make_function(tree, name="ReIndexFunction",
                                          functions={
                                              "weight":"pos_x,0.33,*,pos_z,+"
                                          },hide=True,inputs=["pos"],outputs=["weight"],scalars=["weight"],vectors=["pos"])
        links.new(pos.std_out,re_index_function.inputs["pos"])
        sort_node = SortElements(tree,sort_weight=re_index_function.outputs["weight"],hide=True)

        # prepare face selection
        face_selector = InputInteger(tree,label="FaceSelector",integer=3,hide=True)
        index = Index(tree,hide=True)

        selector_function=make_function(tree,name="SelectorFunction",functions={
            "selection":"idx,face_selector,<"
        },inputs=["idx","face_selector"],outputs=["selection"],scalars=["selection","idx","face_selector"],
                                        vectors=[],hide=True)
        links.new(index.std_out,selector_function.inputs["idx"])
        links.new(face_selector.std_out,selector_function.inputs["face_selector"])

        select_geo = SeparateGeometry(tree,domain="FACE",selection=selector_function.outputs["selection"],hide=True)

        # create unfolding
        progress = InputValue(tree, name="Progress", value=0.0, hide=True)
        unfold_node = UnfoldMeshNode(tree, name="UnfoldMeshNode",hide=True,progression=progress.std_out,scale_elements=1,**kwargs)

        create_geometry_line(tree,[sort_node,select_geo,unfold_node],ins = ins.outputs[0])

        # select types of faces
        face_types = get_from_kwargs(kwargs,"face_types",[4,6,10])
        face_materials = get_from_kwargs(kwargs,"face_materials",["example","important","joker"])
        corners_of_face = CornersOfFace(tree,std_out="Total",hide=True)
        join_geo = JoinGeometry(tree,hide=True)
        for type,material_string in zip(face_types,face_materials):
            compare_node = CompareNode(tree,hide=True,
                                       data_type="INT",operation="EQUAL",inputs0=corners_of_face.std_out,inputs1=type)
            separate_geometry = SeparateGeometry(tree,domain="FACE",selection=compare_node.outputs[0],hide=True)
            material = get_texture(material_string,**kwargs)
            self.materials.append(material)
            material_node = SetMaterial(tree,material=material,hide=True)

            create_geometry_line(tree,[unfold_node,separate_geometry,material_node,join_geo])

        create_geometry_line(tree,[join_geo],out=out.inputs[0])

# recreate the essentials to convert a latex expression into a collection of curves
# that can be further processed in geometry nodes

def generate_labels(tic_labels,axis_label, **kwargs):
    aligned = get_from_kwargs(kwargs, 'aligned', 'left')
    imported_svg_data = {}  # Build dictionary of imported svgs to use

    # shape keys later and to avoid duplicate
    tic_label_suffix = get_from_kwargs(kwargs, 'tic_label_suffix', '')
    for key, val in tic_labels.items():
        path = get_file_path(key+tic_label_suffix)
        if isinstance(val,(float,int)):
            val=[val]
        tic_labels[key] = val + [path]
        imported_svg_data=import_svg_data(imported_svg_data,path,kwargs)

    imported_svg_data = align_figures(imported_svg_data, aligned)

    # cleaning up
    for label, expr in zip(tic_labels, imported_svg_data):
        # create new collection for every label
        collection = ibpy.make_new_collection(str(label),hide_render=True,hide_viewport=True)
        for curve in imported_svg_data[expr]:
            ibpy.link(curve, collection)
        # remove import collection
        collection_name = expr.split(os.path.sep)[-1]
        old_collection = ibpy.get_collection(collection_name)
        for obj in old_collection.objects:
            ibpy.un_link(obj, old_collection.name)
        ibpy.remove_collection(old_collection)

    # prepare axis label
    if len(axis_label)>0:
        imported_svg_data={}
        path=get_file_path(axis_label)
        imported_svg_data = import_svg_data(imported_svg_data,path,kwargs)
        imported_svg_data = align_figures(imported_svg_data, "left")
        collection = ibpy.make_new_collection(axis_label,hide_render=True,hide_viewport=True)
        for curve in imported_svg_data[list(imported_svg_data.keys())[-1]]:
            ibpy.link(curve,collection)

        # replace default import collection with a more appropriately named collection
        collection_name = path.split(os.sep)[-1]
        old_collection=ibpy.get_collection(collection_name)
        for obj in old_collection.objects:
            ibpy.un_link(obj, old_collection.name)
        ibpy.remove_collection(old_collection)
        label = collection.name
    else:
        label=None

    return tic_labels, label

def import_svg_data(imported_svg_data,path,kwargs):
    default_color = get_from_kwargs(kwargs, 'color', 'text')
    text_size = get_from_kwargs(kwargs, 'text_size', 'normal')
    scale = get_from_kwargs(kwargs, 'scale', 1)
    vert_align_centers = get_from_kwargs(kwargs, 'vert_align_centers', 'x_and_y')

    # Import svg and get list of new curves in Blender
    if path not in imported_svg_data.keys():
        imported_svg_data[path] = {'curves': []}

        previous_curves = ibpy.get_all_curves()
        ibpy.import_curve(path)  # here the import takes place
        # all curves are imported into a separate collection with the name path
        new_curves = [x for x in ibpy.get_all_curves() if x not in previous_curves]

        # Arrange new curves relative to tex object's ref_obj
        if text_size == 'normal':
            scale_up = TEX_LOCAL_SCALE_UP
        elif text_size == 'medium':
            scale_up = TEX_LOCAL_SCALE_UP * 1.25
        elif text_size == 'large':
            scale_up = 1.5 * TEX_LOCAL_SCALE_UP
        elif text_size == 'Large':
            scale_up = 2 * TEX_LOCAL_SCALE_UP
        elif text_size == 'Small':
            scale_up = 0.8 * TEX_LOCAL_SCALE_UP
        elif text_size == 'small':
            scale_up = 0.7 * TEX_LOCAL_SCALE_UP
        elif text_size == 'xs':
            scale_up = 0.65 * TEX_LOCAL_SCALE_UP
        elif text_size == 'tiny':
            scale_up = 0.5 * TEX_LOCAL_SCALE_UP
        elif text_size == 'huge':
            scale_up = 3 * TEX_LOCAL_SCALE_UP
        elif text_size == 'Huge':
            scale_up = 5 * TEX_LOCAL_SCALE_UP
        else:  # a number
            scale_up = text_size * TEX_LOCAL_SCALE_UP

        # in the following lines the location of the curve is determined from the geometry and the
        # y-coordinates of the locations are all aligned with the reference 'H'
        for curve in new_curves:
            for spline in curve.data.splines:
                for point in spline.bezier_points:
                    point.handle_left_type = 'FREE'
                    point.handle_right_type = 'FREE'
                # This needs to be in a separate loop because moving points before
                # they're all 'Free' type makes the shape warp.
                # It makes a cool "disappear in the wind" visual, though.
                for point in spline.bezier_points:
                    for i in range(len(point.co)):
                        point.co[i] *= (scale * scale_up)
                        point.handle_left[i] *= (scale * scale_up)
                        point.handle_right[i] *= (scale * scale_up)

            ibpy.set_origin(curve)
            # This part is just meant for tex_objects
            if vert_align_centers:
                loc = curve.location
                new_y = new_curves[0].location[1]  # reference location of the 'H'
                ibpy.set_pivot(curve, [loc[0], new_y, loc[2]])
            ibpy.un_select(curve)

            # find the collection of the curve and unlink object
            # for collection in bpy.context.scene.collection.children:
            #     for o in collection.objects:
            #         if o == curve:
            #             ibpy.un_link(curve, collection.name)
            #             break

            imported_svg_data[path]['curves'] = new_curves
    return imported_svg_data

#### Generating functions
def get_file_path(expression, text_only=False, typeface="default"):
    # Replaces the svg_b_object method
    if text_only:
        template = deepcopy(TEMPLATE_TEXT_FILE)
    else:
        template = deepcopy(TEMPLATE_TEX_FILE)
    if typeface != 'default':
        template = template[:-10]  # chop off the _arial.tex
        template += '_' + typeface + '.tex'
        if not os.path.exists(template):
            raise Warning(r'Can\'t find template tex file for that font.')

    return tex_to_svg_file(expression, template, typeface, text_only)

def tex_to_svg_file(expression, template_tex_file, typeface, text_only):
    path = os.path.join(
        SVG_DIR,
        hashed_tex(expression, typeface)
    ) + ".svg"
    if os.path.exists(path):
        return path

    tex_file = generate_tex_file(expression, template_tex_file, typeface, text_only)
    dvi_file = tex_to_dvi(tex_file)
    return dvi_to_svg(dvi_file)

def generate_tex_file(expression, template_tex_file, typeface, text_only):
    result = os.path.join(
        TEX_DIR,
        # tex_title(expression, typeface)
        hashed_tex(expression, typeface)
    ) + ".tex"

    if not os.path.exists(result):
        print("Writing \"%s\" to %s" % (
            "".join(expression), result
        ))
        with open(template_tex_file, "r") as infile:
            body = infile.read()
            # I add an H to every expression to give a common reference point
            # for all expressions, then hide the H character. This is necessary
            # for consistent alignment of tex curves in blender, because
            # blender's import svg bobject sets the object's origin depending
            # on the expression itself, not according to a typesetting reference
            # frame.
            if text_only:
                expression = 'H ' + expression
            else:
                expression = '\\text{H} ' + expression
            body = body.replace(TEX_TEXT_TO_REPLACE, expression)
        with open(result, "w") as outfile:
            outfile.write(body)
    return result

def tex_to_dvi(tex_file):
    result = tex_file.replace(".tex", ".dvi")
    if not os.path.exists(result):
        commands = [
            "latex",
            "-interaction=batchmode",
            "-halt-on-error",
            "-output-directory=" + TEX_DIR,
            tex_file  # ,
            # ">",
            # get_null()
        ]
        exit_code = os.system(" ".join(commands))
        if exit_code != 0:
            latex_output = ''
            log_file = tex_file.replace(".tex", ".log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    latex_output = f.read()
            raise Exception(
                "Latex error converting to dvi. "
                "See log output above or the log file: %s" % log_file)
    return result

def dvi_to_svg(dvi_file):
    """
    Converts a dvi, which potentially has multiple slides, into a
    directory full of enumerated svgs corresponding with these slides.
    Returns a list of PIL Image objects for these images sorted as they
    where in the dvi
    """

    result = dvi_file.replace(".dvi", ".svg")
    result = result.replace("tex", "svg")  # change directory for the svg file
    print('svg: ', result)
    if not os.path.exists(result):
        commands = [
            "dvisvgm",
            dvi_file,
            "-n",
            "-v",
            "3",
            "-o",
            result
            # Not sure what these are for, and it seems to work without them
            # so commenting out for now
            # ,
            # ">",
            # get_null()
        ]
        os.system(" ".join(commands))
    return result

def hashed_tex(expression, typeface):
    string = expression + typeface
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]

#### Alignment functions

def get_figure_curves(imported_svg_data, fig):
    """
        returns the curves
        :param fig:
        :return:
        """

    return imported_svg_data[fig]['curves']

def calc_lengths(imported_svg_data):
    """
        The dimension of the object is calculated
        The parameters are stored in

        | imported_svg_data:
        | 'length'
        | 'height'
        | 'centerx'
        | 'centery'
        | 'beginning'
        | 'end'
        | 'top'
        | 'bottom'

        :return:
        """
    lengths = []
    for expr in imported_svg_data:
        curves = get_figure_curves(imported_svg_data, expr)  # the H is stripped for latex formulas

        max_vals = [-math.inf, -math.inf]
        min_vals = [math.inf, math.inf]

        directions = [0, 1]  # 0 horizontal
        # 1 vertical

        for direction in directions:
            for char in curves:
                for spline in char.data.splines:
                    for point in spline.bezier_points:
                        candidate = char.matrix_local.translation[direction] + point.co[direction] * char.scale[
                            direction]  # +char.parent.matrix_local.translation[direction]
                        if max_vals[direction] < candidate:
                            max_vals[direction] = candidate
                        if min_vals[direction] > candidate:
                            min_vals[direction] = candidate

        right_most_x, top_most_y = max_vals
        left_most_x, bottom_most_y = min_vals

        length = right_most_x - left_most_x
        center = left_most_x + length / 2

        imported_svg_data[expr]['length'] = length
        imported_svg_data[expr]['centerx'] = center
        imported_svg_data[expr]['beginning'] = left_most_x  # Untested
        imported_svg_data[expr]['end'] = right_most_x

        height = top_most_y - bottom_most_y
        center = bottom_most_y + height / 2

        imported_svg_data[expr]['top'] = top_most_y
        imported_svg_data[expr]['bottom'] = bottom_most_y
        imported_svg_data[expr]['height'] = height
        imported_svg_data[expr]['centery'] = center
        imported_svg_data[expr]['curves'] = curves
    return imported_svg_data

def align_figures(imported_svg_data, aligned):
    imported_svg_data = calc_lengths(imported_svg_data)
    for fig in imported_svg_data:
        imported_svg_data[fig] = align_figure(fig, imported_svg_data, aligned)
    return imported_svg_data

def align_figure(fig, imported_svg_data, aligned):
    data = imported_svg_data
    curve_list = data[fig]['curves']
    offset = list(curve_list[0].location)

    # positioning of the first element of the list
    if aligned == 'right':
        offset[0] = data[fig]['end']
        offset[1] = data[fig]['centery']
    if aligned == 'right_bottom':
        offset[0] = data[fig]['end']
        offset[1] = data[fig]['bottom']
    elif aligned == 'top_centered':
        offset[0] = data[fig]['centerx']
        offset[1] = data[fig]['top']
    elif aligned == 'left_bottom':
        offset[0] = data[fig]['beginning']
        offset[1] = data[fig]['bottom']
    elif aligned == 'left_top':
        offset[0] = data[fig]['beginning']
        offset[1] = data[fig]['top']
    elif aligned == 'x_and_y':
        offset[0] = data[fig]['centerx']
        offset[1] = data[fig]['centery']
    elif aligned == 'left':
        offset[0] = data[fig]['beginning']
        offset[1] = data[fig]['centery']
    elif aligned == 'center':
        cen = data[fig]['centerx']
        offset[0] = cen
        offset[1] = data[fig]['centery']

    for i in range(len(curve_list)):
        loc = list(curve_list[i].location)
        new_loc = add_lists_by_element(loc, offset, subtract=True)
        curve_list[i].location = new_loc

    # sort letters from left to right
    curve_list.sort(key=lambda x:x.location[0])
    # remove reference H
    return curve_list[1:]

def add_lists_by_element(list1, list2, subtract=False):
    if len(list1) != len(list2):
        raise Warning("The lists aren't the same length")
    list3 = list(deepcopy(list2))
    if subtract:
        for i in range(len(list3)):
            list3[i] *= -1
    return list(map(sum, zip(list1, list3)))
