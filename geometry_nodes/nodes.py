import os
from _ast import Compare
from multiprocessing import Value
from operator import truediv

import bpy
import numpy as np
from mathutils import Vector
from sympy import false

from grandalf.graphs import Vertex, Edge, Graph
from grandalf.layouts import SugiyamaLayout, DigcoLayout
from interface import ibpy
from interface.ibpy import get_material, make_new_socket, OPERATORS, get_obj
from interface.interface_constants import blender_version
from mathematics.groups.e8 import E8Lattice
from utils.color_conversion import get_color
from utils.constants import RES_XML, RES_XML2
from utils.kwargs import get_from_kwargs
from utils.string_utils import parse_vector

pi = np.pi

SOCKET_TYPES = {"STRING":'NodeSocketString',"BOOLEAN": 'NodeSocketBool',"MATERIAL": 'NodeSocketMaterial',
                        "VECTOR":'NodeSocketVector',"INT":'NodeSocketInt', "MENU":'NodeSocketMenu',"COLLECTION":'NodeSocketCollection',
                        "GEOMETRY":'NodeSocketGeometry', "TEXTURE":'NodeSocketTexture',"FLOAT":'NodeSocketFloat',
                        "COLOR":'NodeSocketColor',"OBJECT": 'NodeSocketObject',"ROTATION":'NodeSocketRotation',
                        "MATRIX": 'NodeSocketMatrix',"IMAGE": 'NodeSocketImage',"VALUE":'NodeSocketFloat'}

def maybe_flatten(list_of_lists):
    result = []
    for part in list_of_lists:
        if isinstance(part, list):
            result += part
        else:
            result.append(part)
    return result

def parse_location(location):
    location = location.replace("(", "")
    location = location.replace(")", "")
    coords = location.split(",")
    return (float(coords[0]), float(coords[1]))

class Node:
    def __init__(self, tree, location=(0, 0), node_width=200, node_height=100, offset_y=0, **kwargs):
        """
        TODO: I have to do this nicer
        """

        self.tree = tree
        self.location = location
        self.l, self.m = location

        self.links = tree.links

        if "hide" in kwargs:
            hide = kwargs.pop("hide")
            self.node.hide = hide
        if "name" in kwargs:
            self.name = kwargs.pop("name")
            self.node.label = self.name
            self.node.name = self.name
        else:
            self.name = "DefaultGeometryNode"
        if "label" in kwargs:
            label = kwargs.pop("label")
            self.node.label = label

        parent = get_from_kwargs(kwargs,"parent",None)
        if parent:
            self.node.parent = parent.node
            self.node.location = ((self.l+parent.l) * node_width, (self.m+parent.m) * node_height + offset_y)
        else:
            self.node.location = (self.l * node_width, self.m * node_height + offset_y)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

    @classmethod
    def from_attributes(cls,tree,attributes):
        name = attributes["name"]
        # print("Create Node from attributes: ",attributes["id"],": ", name)
        if "location" not in attributes:
            pass
        location = parse_location(attributes["location"])
        label = attributes["label"]
        if attributes["hide"]=="False":
            hide=False
        else:
            hide=True

        if attributes["mute"]=="False":
            mute=False
        else:
            mute=True

        type = attributes["type"]

        # custom groups
        if type=="GROUP":
            if label=="BevelNode":
                return BevelNode(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # io nodes
        if type=="GROUP_INPUT":
            return GroupInput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="GROUP_OUTPUT":
            return GroupOutput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # input nodes
        if type=="VALUE":
            return InputValue(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="INPUT_INT":
            integer = int(attributes["integer"])
            return InputInteger(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,integer=integer)
        if type=="INPUT_STRING":
            string = attributes["string"]
            return InputString(tree,location=location,name=name,label=label,hide=hide,mute=mute,
                               node_height=200,string=string)
        if type=="INPUT_BOOL":
            boolean = bool(attributes["boolean"])
            return InputBoolean(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,boolean=boolean)
        if type=="RANDOM_VALUE":
            data_type = attributes["data_type"]
            return RandomValue(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type)
        if type=="INPUT_VECTOR":
            vector = attributes["vector"]
            return InputVector(tree,location=location, name=name, label=label, hide=hide, mute=mute, node_height=200,
                               value=parse_vector(vector))
        if type=="INPUT_SCENE_TIME":
            return SceneTime(tree,location=location, name=name, label=label, hide=hide, mute=mute, node_height=200)

        # read nodes
        if type =="INDEX":
            return Index(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type =="POSITION":
            return Position(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="COLLECTION_INFO":
            transform_space=attributes["transform_space"]
            return CollectionInfo(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,transform_space=transform_space)
        if type=="INPUT_NORMAL":
            return InputNormal(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="INPUT_MATERIAL":
            material=attributes["material"]
            return InputMaterial(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200,
                                 material=material)

        # geometry nodes
        if type=="MESH_LINE":
            return MeshLine(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="MESH_TO_CURVE":
            return MeshToCurve(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="MESH_TO_POINTS":
            return MeshToPoints(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SUBDIVISION_SURFACE":
            return SubdivisionSurface(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SPLIT_EDGES":
            return SplitEdges(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="REALIZE_INSTANCES":
            return RealizeInstances(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SET_POSITION":
            return SetPosition(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="POINTS_TO_CURVES":
            return PointsToCurve(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SEPARATE_GEOMETRY":
            domain=attributes["domain"]
            return SeparateGeometry(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,domain=domain)
        if type=="SORT_ELEMENTS":
            domain=attributes["domain"]
            return SortElements(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,domain=domain)
        if type=="DELETE_GEOMETRY":
            domain=attributes["domain"]
            mode=attributes["mode"]
            return DeleteGeometry(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,domain=domain,mode=mode)
        if type=="JOIN_GEOMETRY":
            return JoinGeometry(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SET_MATERIAL":
            return SetMaterial(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SCALE_ELEMENTS":
            domain=attributes["domain"]
            return ScaleElements(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,domain=domain)
        if type=="TRANSFORM_GEOMETRY":
            return TransformGeometry(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="BOUNDING_BOX":
            return BoundingBox(tree, location=location, name=name, label=label, hide=hide, mute=mute,
                                     node_height=200)
        if type=="CONVEX_HULL":
            return ConvexHull(tree, location=location, name=name, label=label, hide=hide, mute=mute,
                                     node_height=200)
        if type=="SET_SHADE_SMOOTH":
            return SetShadeSmooth(tree, location=location, name=name, label=label, hide=hide, mute=mute,
                                     node_height=200)
        if type=="MESH_BOOLEAN":
            operation = attributes["operation"]
            solver = attributes["solver"]
            return MeshBoolean(tree, location=location, name=name, label=label, hide=hide, mute=mute,
                               operation=operation, solver=solver)
        if type=="MERGE_BY_DISTANCE":
            mode = attributes["mode"]
            return MergeByDistance(tree,location=location,name=name,
                                   label=label,hide=hide,mute=mute,
                                   node_height=200,
                                   mode=mode)
        if type=="OBJECT_INFO":
            transform_space=attributes["transform_space"]
            return ObjectInfo(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=400,transform_space=transform_space)
        if type=="GeometryNodeImportCSV":
            return ImportCSV(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # instances
        if type=="SCALE_INSTANCES":
            return ScaleInstances(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="GEOMETRY_TO_INSTANCE":
            return GeometryToInstance(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="INSTANCE_ON_POINTS":
            return InstanceOnPoints(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="ROTATE_INSTANCES":
            return RotateInstances(tree, location=location, name=name, label=label, hide=hide, mute=mute,
                                    node_height=200)
        # attribute nodes
        if type=="STORE_NAMED_ATTRIBUTE":
            data_type = attributes["data_type"]
            domain = attributes["domain"]
            return StoredNamedAttribute(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type,domain=domain)
        if type=="INPUT_ATTRIBUTE":
            data_type = attributes["data_type"]
            return NamedAttribute(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type)
        if type=="ATTRIBUTE_DOMAIN_SIZE":
            component = attributes["component"]
            return DomainSize(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,component=component)
        if type=="ATTRIBUTE_STATISTIC":
            data_type = attributes["data_type"]
            domain=attributes["domain"]
            return AttributeStatistic(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type,domain=domain)
        if type=="SAMPLE_INDEX":
            data_type = attributes["data_type"]
            domain=attributes["domain"]
            return SampleIndex(tree,location=location,name=name,label=label,hide=mute,node_height=200,data_type=data_type,domain=domain)

        # points
        if type=="POINTS":
            return Points(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # mesh
        if type == "EXTRUDE_MESH":
            mode = attributes["mode"]
            return ExtrudeMesh(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,mode=mode)
        if type=="DUAL_MESH":
            return DualMesh(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SHORTEST_EDGE_PATHS":
            return ShortestEdgePaths(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="EDGES_OF_CORNER":
            return EdgesOfCorner(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="FACE_OF_CORNER":
            return FaceOfCorner(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="CORNERS_OF_EDGE":
            return CornersOfEdge(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="CORNERS_OF_FACE":
            return CornersOfFace(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="OFFSET_CORNER_IN_FACE":
            return OffsetCornerInFace(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="MESH_PRIMITIVE_CUBE":
            return CubeMesh(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200)
        if type=="MESH_PRIMITIVE_ICO_SPHERE":
            return IcoSphere(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200)
        if type=="MESH_PRIMITIVE_UV_SPHERE":
            return UVSphere(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200)
        if type=="MESH_PRIMITIVE_GRID":
            return Grid(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="MESH_PRIMITIVE_LINE":
            return MeshLine(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="MESH_PRIMITIVE_CYLINDER":
            return CylinderMesh(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type == "MESH_PRIMITIVE_CONE":
            return ConeMesh(tree, location=location, name=name,
                            label=label, hide=hide, mute=mute, node_height=200,
                            )

        # curves and strings
        if type=="VALUE_TO_STRING":
            data_type = attributes["data_type"]
            return ValueToString(tree,location=location,name=name,label=label,
                                 hide=hide,mute=mute,node_height=200,
                                 data_type=data_type)
        if type=="STRING_TO_CURVES":
            font = attributes["font"]
            overflow=attributes["overflow"]
            align_x=attributes["align_x"]
            align_y=attributes["align_y"]
            pivot_mode=attributes["pivot_mode"]
            return StringToCurves(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,
                                  font=font,overflow=overflow,align_x=align_x,align_y=align_y,pivot_mode=pivot_mode)
        if type=="STRING_JOIN":
            return StringJoin(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="SLICE_STRING":
            return SliceString(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200)
        if type=="CURVE_PRIMITIVE_CIRCLE":
            mode=attributes["mode"]
            return CurveCircle(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,mode=mode)
        if type=="CURVE_PRIMITIVE_QUADRILATERAL":
            mode = attributes["mode"]
            return CurveQuadrilateral(tree, location=location, name=name, label=label, hide=hide, mute=mute,
                                      node_height=200,mode=mode)
        if type=="CURVE_PRIMITIVE_ARC":
            mode=attributes["mode"]
            return CurveArc(tree,location=location,name=name,
                            label=label,hide=hide,mute=mute,node_height=200,
                            mode=mode)

        # if type=="CURVE_TO_POINTS":
        #     mode = attributes["mode"]
        #     return CurveToPoints(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,mode=mode)
        if type=="CURVE_TO_MESH":
            return CurveToMesh(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="RESAMPLE_CURVE":
            return ResampleCurve(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="FILLET_CURVE":
            return FilletCurve(tree, location=location, name=name, label=label, hide=hide, mute=mute,node_height=200)
        if type=="FILL_CURVE":
            return FillCurve(tree, location=location, name=name, label=label, hide=hide, mute=mute,node_height=200)
        if type=="SAMPLE_CURVE":
            data_type = attributes["data_type"]
            mode = attributes["mode"]
            return SampleCurve(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type,mode=mode)

        # auxiliaries
        if type == "MATH":
            operation = attributes["operation"]
            return MathNode(tree, location=location, name=name, label=label, hide=hide, operation=operation, mute=mute,node_height=200)
        if type == "VECT_MATH":
            operation = attributes["operation"]
            return VectorMath(tree, location=location, name=name, label=label, hide=hide, operation=operation, mute=mute,node_height=200)
        if type=="SEPXYZ":
            return SeparateXYZ(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="COMBXYZ":
            return CombineXYZ(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="COMPARE":
            operation=attributes["operation"]
            data_type=attributes["data_type"]
            return CompareNode(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,operation=operation,data_type=data_type)
        if type=="MAP_RANGE":
            data_type=attributes["data_type"]
            interpolation_type=attributes["interpolation_type"]
            return MapRange(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type,interpolation_type=interpolation_type)
        if type=="MIX":
            data_type=attributes["data_type"]
            factor_mode=attributes["factor_mode"]
            cf = attributes["clamp_factor"]
            if cf =='True':
                clamp_factor=True
            else:
                clamp_factor=False
            return MixNode(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type,factor_mode=factor_mode,clamp_factor=clamp_factor)
        if type=="BOOLEAN_MATH":
            operation = attributes["operation"]
            return BooleanMath(tree, location=location, name=name, label=label, hide=hide, operation=operation, mute=mute,node_height=200)

        # rotations
        if type=="ALIGN_ROTATION_TO_VECTOR":
            axis=attributes["axis"]
            pivot_axis=attributes["pivot_axis"]
            return AlignRotationToVector(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,axis=axis,pivot_axis=pivot_axis)
        if type=="ROTATE_ROTATION":
            rotation_space = attributes["rotation_space"]
            return RotateRotation(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,rotation_space=rotation_space)
        if type=="INVERT_ROTATION":
            return InvertRotation(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="AXES_TO_ROTATION":
            primary_axis=attributes["primary_axis"]
            secondary_axis=attributes["secondary_axis"]
            return AxesToRotation(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,primary_axis=primary_axis,secondary_axis=secondary_axis)
        if type=="AXIS_ANGLE_TO_ROTATION":
            return AxisAngleToRotation(tree,location=location,name=name,label=label,hide=hide,
                                       mute=mute,node_height=200)
        if type=="EULER_TO_ROTATION":
            return EulerToRotation(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # switches
        if type=="SWITCH":
            input_type = attributes["input_type"]
            return Switch(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,input_type=input_type)
        if type=="INDEX_SWITCH":
            data_type=attributes["data_type"]
            return IndexSwitch(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type)
            # don't know the category yet
        if type=="FIELD_ON_DOMAIN":
            data_type=attributes["data_type"]
            domain=attributes["domain"]
            return EvaluateOnDomain(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type,domain=domain)

        if type=="FIELD_AT_INDEX":
            data_type=attributes["data_type"]
            domain = attributes["domain"]
            return EvaluateAtIndex(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type,domain=domain)

        if type=="REROUTE":
            return ReRoute(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="FRAME":
            return Frame(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,node_width=200)

        if type=="GROUP_OUTPUT":
            return GroupOutput(tree,location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # Zones
        if type=="FOREACH_GEOMETRY_ELEMENT_INPUT":
            return ForEachInput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="FOREACH_GEOMETRY_ELEMENT_OUTPUT":
            domain = attributes["domain"]
            return ForEachOutput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,domain=domain)

        if type=="REPEAT_INPUT":
            return RepeatInput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="REPEAT_OUTPUT":
            return RepeatOutput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        if type == "SIMULATION_INPUT":
            return SimulationInput(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200)
        if type == "SIMULATION_OUTPUT":
            return SimulationOutput(tree, location=location, name=name, label=label, hide=hide, mute=mute, node_height=200)

        # Custom Nodes
        if type=="GROUP":
            if "Position Transform" in name:
                return TransformPositionNode(tree,location=location,name=name,label=label,hide=hide,mute=mute,height=400)

    def set_parent(self,parent):
        self.node.parent=parent.node

class GroupInput(Node):
    def __init__(self,tree,location=(0,0),hide=False,mute=False,**kwargs):
        self.node = tree.nodes.new("NodeGroupInput")
        self.node.hide=hide
        self.node.mute=mute
        super().__init__(tree,location,**kwargs)

class GroupOutput(Node):
    def __init__(self,tree,location=(0,0),hide=False,mute=False,**kwargs):
        self.node = tree.nodes.new("NodeGroupOutput")
        self.node.hide=hide
        self.node.mute=mute
        super().__init__(tree,location,**kwargs)

    def add_socket(self, socket_type, socket_name):
        self.node.repeat_items.new(socket_type, socket_name)

class ReRoute(Node):
    """
    ReRoute nodes are created, when the links are split and re-routed.
    Don't use it, when you create the xml file data.
    They are buggy.
    """
    def __init__(self, tree, location=(0, 0), hide=False, mute=False, **kwargs):
        self.node = tree.nodes.new("NodeReroute")
        self.node.hide=hide
        self.node.mute=mute

        self.std_out=self.node.outputs[0]
        self.std_in=self.node.inputs[0]

        super().__init__(tree,location,**kwargs)

class Frame(Node):
    def __init__(self,tree,location=(0,0),node_width=200, node_height=200,hide=False,mute=False,**kwargs):
        self.node = tree.nodes.new(type="NodeFrame")
        self.node.hide=hide
        self.node.mute=mute
        super().__init__(tree,location,node_width=node_width,node_height=node_height,**kwargs)

    def add(self,node):
        """ make Frame parent to given node(s)  """
        if isinstance(node,list):
            for n in node:
                n.parent=self.node
        else:
            node.parent=self.node

class GreenNode(Node):
    """
    Super class for "green" geometry nodes.
    They have a standard geometry input and/or a standard geometry output.
    They can be piped together in a line.
    """

    def __init__(self, tree, location=(0, 0), **kwargs):
        super().__init__(tree, location=location, **kwargs)

        # the following ports have to be declared in the children
        # to automatically build a chain of geometry nodes
        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

class RedNode(Node):
    """
    this is the super class for a general input node

    """

    def __init__(self, tree, location=(0, 0), **kwargs):
        super().__init__(tree, location=location, **kwargs)

        self.outputs = self.node.outputs

class BlueNode(Node):
    """
    this is the super class for a general blue node

    """

    def __init__(self, tree, location=(0, 0), **kwargs):
        super().__init__(tree, location=location, **kwargs)

        self.outputs = self.node.outputs

        # changed from self.std_out=self.outputs["Value"] to accomodate Vector outputs
        self.std_out = self.outputs[0]

#   green nodes  #
# mesh primitives
class MeshLine(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mode="END_POINTS",
                 count_mode="TOTAL",  # alternative is "RESOLUTION"
                 count=10,
                 start_location=Vector([0, 0, 0]),
                 end_location=None, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodeMeshLine")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.node.mode = mode
        self.node.count_mode = count_mode

        if isinstance(count, int):
            self.node.inputs["Count"].default_value = count
        else:
            self.tree.links.new(count, self.node.inputs["Count"])
        if isinstance(start_location, (list,Vector)):
            self.node.inputs["Start Location"].default_value = start_location
        else:
            self.tree.links.new(start_location, self.node.inputs["Start Location"])
        if end_location:
            # kept for compatability
            if isinstance(end_location,(list,Vector)):
                self.node.inputs["Offset"].default_value = end_location
            else:
                self.tree.links.new(end_location, self.node.inputs["Offset"])


# mesh operations
class DualMesh(GreenNode):
    def __init__(self, tree, location=(0, 0), keep_boundaries=False, **kwargs):
        """

        :param tree:
        :param location:
        :param mode: "VERTICES, FACES, EDGES
        :param mesh:
        :param selection:
        :param offset:
        :param offset_scale:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeDualMesh")
        self.node.inputs[1].default_value=keep_boundaries

        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Dual Mesh"]
        self.geometry_in = self.node.inputs["Mesh"]

class SplitEdges(GreenNode):
    def __init__(self, tree, selection = None,location=(0, 0), **kwargs):
        """
        """
        self.node = tree.nodes.new(type="GeometryNodeSplitEdges")
        super().__init__(tree, location=location, **kwargs)

        if selection:
            tree.links.new(selection,self.node.inputs["Selection"])

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Mesh"]

class ExtrudeMesh(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mode="VERTICES",
                 mesh=None,
                 selection=None,
                 offset=None,
                 offset_scale=1, **kwargs):
        """

        :param tree:
        :param location:
        :param mode: "VERTICES, FACES, EDGES
        :param mesh:
        :param selection:
        :param offset:
        :param offset_scale:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeExtrudeMesh")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Mesh"]
        self.node.mode = mode

        if isinstance(offset_scale, (int, float)):
            self.node.inputs["Offset Scale"].default_value = offset_scale
        else:
            self.tree.links.new(offset_scale, self.node.inputs["Offset Scale"])

        if offset:
            if isinstance(offset, (Vector, list)):
                vector = InputVector(tree, location=(self.location[0] - 1, self.location[1] - 1), value=offset)
                self.tree.links.new(vector.std_out, self.node.inputs["Offset"])
            else:
                self.tree.links.new(offset, self.node.inputs["Offset"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if mesh:
            self.tree.links.new(mesh, self.node.inputs["Mesh"])

class MeshToCurve(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mesh=None,
                 selection=None,
                 **kwargs):
        """
        :param tree:
        :param location:
        :param mesh:
        :param selection:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeMeshToCurve")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Curve"]
        self.geometry_in = self.node.inputs["Mesh"]

        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if mesh:
            self.tree.links.new(mesh, self.node.inputs["Mesh"])

class MeshToPoints(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mesh=None,
                 selection=None,
                 position=None,
                 **kwargs):
        """
        :param tree:
        :param location:
        :param mesh:
        :param selection:
        :param position:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeMeshToPoints")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Points"]
        self.geometry_in = self.node.inputs["Mesh"]

        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if mesh:
            self.tree.links.new(mesh, self.node.inputs["Mesh"])
        if position:
            if isinstance(position, (Vector, list)):
                self.node.inputs["Position"].default_value = position
            else:
                self.tree.links.new(position, self.node.inputs["Position"])

class SubdivisionSurface(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mesh=None,uv_smooth="PRESERVE_BOUNDARIES",boundary_smooth="ALL",
                 level=1,
                 **kwargs):
        """
        :param tree:
        :param location:
        :param mesh:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeSubdivisionSurface")
        super().__init__(tree, location=location, **kwargs)

        self.node.uv_smooth=uv_smooth
        self.node.boundary_smooth=boundary_smooth
        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Mesh"]

        if mesh:
            self.tree.links.new(mesh, self.node.inputs["Mesh"])
        if level:
            if isinstance(level, int):
                self.node.inputs["Level"].default_value = level
            else:
                self.tree.links.new(level, self.node.inputs["Level"])

class SubdivideMesh(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mesh=None,
                 level=1,
                 **kwargs):
        """
        :param tree:
        :param location:
        :param mesh:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeSubdivideMesh")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Mesh"]

        if mesh:
            self.tree.links.new(mesh, self.node.inputs["Mesh"])
        if level:
            if isinstance(level, int):
                self.node.inputs["Level"].default_value = level
            else:
                self.tree.links.new(level, self.node.inputs["Level"])

class ShortestEdgePaths(RedNode):
    def __init__(self,tree,location=(0,0),end_vertex=None,edge_count=None,std_out="Next Vertex Index",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeInputShortestEdgePaths")
        super().__init__(tree,location=location,**kwargs)

        self.std_out=self.node.outputs[std_out]

        if end_vertex:
            if isinstance(end_vertex,int):
                self.node.inputs["End Vertex"].default_value=end_vertex
            else:
                self.tree.links.new(end_vertex,self.node.inputs["End Vertex"])
        if edge_count:
            if isinstance(edge_count,int):
                self.node.inputs["Edge Count"].default_value=edge_count
            else:
                self.tree.links.new(edge_count,self.node.inputs["Edge Count"])

class EdgesOfCorner(RedNode):
    def __init__(self,tree,location=(0,0),corner_index=None,std_out="Next Edge Index",**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeEdgesOfCorner")
        super().__init__(tree,location=location,**kwargs)

        if corner_index:
            self.tree.links.new(corner_index,self.node.inputs["Corner Index"])

        self.std_out=self.node.outputs[std_out]

class FaceOfCorner(RedNode):
    def __init__(self,tree,location=(0,0),corner_index=None,std_out="Face Index",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeFaceOfCorner")
        super().__init__(tree,location=location,**kwargs)

        if corner_index:
            self.tree.links.new(corner_index,self.node.inputs["Corner Index"])

        self.std_out=self.node.outputs[std_out]

class CornersOfEdge(RedNode):
    def __init__(self,tree,location=(0,0),edge_index=None,weights=None,sort_index=0,std_out="Corner Index",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeCornersOfEdge")
        super().__init__(tree,location=location,**kwargs)

        if weights:
            self.tree.links.new(weights,self.node.inputs["Weights"])

        if edge_index:
            self.tree.links.new(edge_index,self.node.inputs["Edge Index"])

        if isinstance(sort_index,int):
            self.node.inputs["Sort Index"].default_value=sort_index
        else:
            self.tree.links.new(sort_index,self.node.inputs["Sort Index"])

        self.std_out=self.node.outputs[std_out]

class OffsetCornerInFace(RedNode):
    def __init__(self,tree,location=(0,0),corner_index=None,offset=0,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeOffsetCornerInFace")
        super().__init__(tree,location=location,**kwargs)

        if corner_index:
            self.tree.links.new(corner_index,self.node.inputs["Corner Index"])

        if isinstance(offset,int):
            self.node.inputs["Offset"].default_value=offset
        else:
            self.tree.links.new(offset,self.node.inputs["Offset"])

        self.std_out=self.node.outputs["Corner Index"]

class CornersOfFace(RedNode):
    def __init__(self,tree,location=(0,0),edge_index=None,weights=None,sort_index=0,std_out="Corner Index",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeCornersOfFace")
        super().__init__(tree,location=location,**kwargs)

        if weights:
            self.tree.links.new(weights,self.node.inputs["Weights"])

        if edge_index:
            self.tree.links.new(edge_index,self.node.inputs["Edge Index"])

        if isinstance(sort_index,int):
            self.node.inputs["Sort Index"].default_value=sort_index
        else:
            self.tree.links.new(sort_index,self.node.inputs["Sort Index"])

        self.std_out=self.node.outputs[std_out]

# curves
class CurveLine(GreenNode):
    def  __init__(self, tree, location=(0, 0),
                  mode="POINTS",
                  start=Vector(),
                  end=Vector([0,0,1]),
                  **kwargs
                  ):
        self.node = tree.nodes.new(type="GeometryNodeCurvePrimitiveLine")
        super().__init__(tree, location=location, **kwargs)
        self.node.mode = mode
        self.geometry_out = self.node.outputs["Curve"]

        if isinstance(start, (list,Vector)):
            self.node.inputs["Start"].default_value = start
        else:
            self.tree.links.new(start, self.node.inputs["Start"])
        if isinstance(end, (list,Vector)):
            self.node.inputs["End"].default_value = end
        else:
            self.tree.links.new(end, self.node.inputs["End"])


class Quadrilateral(GreenNode):
    def  __init__(self, tree, location=(0, 0),
                  mode="RECTANGLE",
                  width=1,
                  height=1,
                  **kwargs
                  ):
        self.node = tree.nodes.new(type="GeometryNodeCurvePrimitiveQuadrilateral")
        super().__init__(tree, location=location, **kwargs)
        self.node.mode = mode
        self.geometry_out = self.node.outputs["Curve"]

        if isinstance(width, (int, float)):
            self.node.inputs["Width"].default_value = width
        else:
            self.tree.links.new(width, self.node.inputs["Width"])
        if isinstance(height, (int, float)):
            self.node.inputs["Height"].default_value = height
        else:
            self.tree.links.new(height, self.node.inputs["Height"])

class ResampleCurve(GreenNode):
    def __init__(self, tree, location=(0, 0),mode="Count",curve=None,
                 selection = None,
                 count=1,
                 limit_radius="False",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeResampleCurve")
        super().__init__(tree,location=location,**kwargs)

        self.node.inputs["Mode"].default_value=mode
        self.geometry_out=self.node.outputs["Curve"]
        self.geometry_in=self.node.inputs["Curve"]

        if curve:
            self.tree.links.new(curve,self.node.inputs["Curve"])

        if selection:
            self.tree.links.new(curve,self.node.inputs["Selection"])

        if isinstance(count, int):
            self.node.inputs["Count"].default_value = count
        else:
            self.tree.links.new(count, self.node.inputs["Count"])

class TrimCurve(GreenNode):
    def __init__(self, tree, location=(0, 0),curve=None,
                 selection = None,
                 mode='FACTOR',start=0,end=1,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeTrimCurve")
        super().__init__(tree,location=location,**kwargs)

        self.node.mode=mode
        self.geometry_out=self.node.outputs["Curve"]
        self.geometry_in=self.node.inputs["Curve"]

        if curve:
            self.tree.links.new(curve,self.node.inputs["Curve"])

        if selection:
            self.tree.links.new(curve,self.node.inputs["Selection"])

        if isinstance(start, (int,float)):
            self.node.inputs["Start"].default_value = start
        else:
            self.tree.links.new(start, self.node.inputs["Start"])

        if isinstance(end, (int,float)):
            self.node.inputs["End"].default_value = end
        else:
            self.tree.links.new(end, self.node.inputs["End"])

class FilletCurve(GreenNode):
    def __init__(self, tree, location=(0, 0),mode="Poly",radius=1,curve=None,
                 count=1,
                 limit_radius="False",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeFilletCurve")
        super().__init__(tree,location=location,**kwargs)

        self.node.inputs["Mode"].default_value=mode
        self.geometry_out=self.node.outputs["Curve"]
        self.geometry_in=self.node.inputs["Curve"]

        if curve:
            self.tree.links.new(curve,self.node.inputs["Curve"])

        if isinstance(limit_radius,bool):
            self.node.inputs["Limit Radius"].default_value=limit_radius

        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value=radius
        else:
            self.tree.links.new(radius,self.node.inputs["Radius"])

        if isinstance(count, int):
            self.node.inputs["Count"].default_value = count
        else:
            self.tree.links.new(count, self.node.inputs["Count"])

class FillCurve(GreenNode):
    def __init__(self,tree,location=(0,0),mode="N-gons",curve=None,group_id=None,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeFillCurve")
        super().__init__(tree,location=location,**kwargs)

        self.geometry_out=self.node.outputs["Mesh"]
        self.geometry_in=self.node.inputs["Curve"]
        self.node.inputs["Mode"].default_value=mode

        if curve:
            self.tree.links.new(curve,self.node.inputs["Curve"])
        if group_id:
            if isinstance(group_id,int):
                self.node.inputs["Group ID"].default_value=group_id
            else:
                self.tree.links.new(group_id,self.node.inputs["Group ID"])

class SampleCurve(GreenNode):
    def __init__(self,tree,location=(0,0),mode="FACTOR",data_type="FLOAT",all_curves=True,curves=None,value=None,factor=0,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeSampleCurve")
        super().__init__(tree,location=location,**kwargs)

        self.node.mode=mode
        self.node.data_type=data_type
        self.node.use_all_curves=all_curves
        self.geometry_in=self.node.inputs["Curves"]
        self.value_out=self.node.outputs["Value"]
        self.position_out=self.node.outputs["Position"]
        self.tangent_out=self.node.outputs["Tangent"]
        self.normal_out=self.node.outputs["Normal"]

        if curves:
            self.tree.links.new(curves,self.node.inputs["Curve"])
        if value:
            if isinstance(value, (int, float)):
                self.node.inputs["Value"].default_value=value
            else:
                tree.links.new(value,self.node.inputs["Value"])
        if factor:
            if  isinstance(factor, (int, float)):
                self.node.inputs["Factor"].default_value=factor
            else:
                tree.links.new(factor,self.node.inputs["Factor"])


# point operations
class Points(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 count=1,
                 position=Vector([0, 0, 0]),
                 radius=0.1, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodePoints")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]

        if isinstance(count, int):
            self.node.inputs["Count"].default_value = count
        else:
            self.tree.links.new(count, self.node.inputs["Count"])
        if isinstance(position, Vector):
            self.node.inputs["Position"].default_value = position
        else:
            self.tree.links.new(position, self.node.inputs["Position"])
        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs["Radius"])

class PointsToVertices(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 points=None,
                 selection=None, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodePointsToVertices")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Points"]
        if points:
            self.tree.links.new(points, self.node.inputs["Points"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])

class PointsToCurve(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mesh=None,
                 selection=None,
                 **kwargs):
        """
        :param tree:
        :param location:
        :param mesh:
        :param selection:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodePointsToCurves")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Curves"]
        self.geometry_in = self.node.inputs["Points"]

# curve primitives

class CurveCircle(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mode="RADIUS",
                 resolution=4,
                 radius=0.02, **kwargs):
        """

        :param tree:
        :param location:
        :param mode: "RADIUS", "POINTS"
        :param resolution:
        :param radius:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeCurvePrimitiveCircle")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Curve"]

        self.node.mode = mode

        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs["Radius"])
        if isinstance(resolution, int):
            self.node.inputs["Resolution"].default_value = resolution
        else:
            self.tree.links.new(resolution, self.node.inputs["Resolution"])

class CurveArc(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mode="RADIUS",
                 resolution=16,
                 radius=1,start_angle=0,sweep_angle=315,
                 connect_center=False,invert_arc=False,
                 **kwargs):
        """

        :param tree:
        :param location:
        :param mode: "RADIUS", "POINTS"
        :param resolution:
        :param radius:
        :param start_angle:
        :param sweep_angle:
        :param connect_center:
        :param invert_arc:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeCurveArc")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Curve"]
        self.node.mode = mode

        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs["Radius"])
        if isinstance(resolution, int):
            self.node.inputs["Resolution"].default_value = resolution
        else:
            self.tree.links.new(resolution, self.node.inputs["Resolution"])
        if isinstance(start_angle,(int,float)):
            self.node.inputs["Start Angle"].default_value=start_angle
        else:
            tree.links.new(start_angle,self.node.inputs["Start Angle"])
        if isinstance(sweep_angle,(int,float)):
            self.node.inputs["Sweep Angle"].default_value=sweep_angle
        else:
            tree.links.new(sweep_angle,self.node.inputs["Sweep Angle"])
        if isinstance(connect_center,bool):
            self.node.inputs["Connect Center"].default_value=connect_center
        else:
            tree.links.new(connect_center,self.node.inputs["Connect Center"])
        if isinstance(invert_arc,bool):
            self.node.inputs["Invert Arc"].default_value=invert_arc
        else:
            tree.links.new(invert_arc,self.node.inputs["Invert Arc"])

class CurveQuadrilateral(GreenNode):
    def __init__(self, tree, location=(0, 0), mode="RECTANGLE",
                 width=0.02, height=0.02, **kwargs):
        """

        :param tree:
        :param location:
        :param mode: "RECTANGLE", "..."
        :param resolution:
        :param radius:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeCurvePrimitiveQuadrilateral")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Curve"]
        self.node.mode=mode

        if mode=="RECTANGLE":
            if isinstance(width, (int, float)):
                self.node.inputs["Width"].default_value = width
            else:
                self.tree.links.new(width, self.node.inputs["Width"])
            if isinstance(height, (int, float)):
                self.node.inputs["Height"].default_value = height
            else:
                self.tree.links.new(height, self.node.inputs["Height"])

class StringToCurves(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 font="Symbola Regular", # fonts have to be loaded in scene.py
                 overflow='OVERFLOW',
                 align_x="CENTER",
                 align_y="MIDDLE",
                 pivot_mode="MIDPOINT",
                 string="0",size=1,character_spacing=1,word_spacing=1,line_spacing=1,textbox_width=0, **kwargs):
        """
        """
        self.node = tree.nodes.new(type="GeometryNodeStringToCurves")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Curve Instances"]
        self.line=self.node.outputs["Line"]
        self.pivot_point=self.node.outputs["Pivot Point"]

        self.node.overflow=overflow
        self.node.align_x=align_x
        self.node.align_y=align_y
        self.node.pivot_mode=pivot_mode
        self.node.font = bpy.data.fonts.get(font)

        if isinstance(string, str):
            self.node.inputs["String"].default_value = string
        else:
            tree.links.new(string, self.node.inputs["String"])

        if isinstance(size, (int,float)):
            self.node.inputs["Size"].default_value = size
        else:
            tree.links.new(size, self.node.inputs["Size"])

        if isinstance(character_spacing, (int,float)):
            self.node.inputs["Character Spacing"].default_value = character_spacing
        else:
            tree.links.new(character_spacing, self.node.inputs["Character Spacing"])

        if isinstance(word_spacing, (int,float)):
            self.node.inputs["Word Spacing"].default_value = word_spacing
        else:
            tree.links.new(word_spacing, self.node.inputs["Word Spacing"])

        if isinstance(line_spacing, (int,float)):
            self.node.inputs["Line Spacing"].default_value = line_spacing
        else:
            tree.links.new(line_spacing, self.node.inputs["Line Spacing"])

        if isinstance(character_spacing, (int,float)):
            self.node.inputs["Character Spacing"].default_value = character_spacing
        else:
            tree.links.new(character_spacing, self.node.inputs["Character Spacing"])

        if isinstance(textbox_width, (int,float)):
            self.node.inputs["Text Box Width"].default_value = textbox_width
        else:
            self.tree.links.new(textbox_width, self.node.inputs["Text Box Width"])

# curve operations
class CurveToMesh(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 curve=None,
                 profile_curve=None,
                 fill_caps = True,
                 **kwargs):
        """

        :param tree:
        :param location:
        :param curve:
        :param profile_curve:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeCurveToMesh")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Curve"]

        self.node.inputs["Fill Caps"].default_value = fill_caps
        if curve:
            self.tree.links.new(curve, self.node.inputs["Curve"])
        if profile_curve:
            self.tree.links.new(profile_curve, self.node.inputs["Profile Curve"])

# String operations
class ValueToString(BlueNode):
    def __init__(self,tree,location=(0,0),value=0,data_type="INT",**kwargs):
        self.node=tree.nodes.new(type="FunctionNodeValueToString")
        super().__init__(tree,location=location,**kwargs)

        self.std_out = self.node.outputs["String"]
        self.node.data_type = data_type

        if isinstance(value, (int, float)):
            self.node.inputs["Value"].default_value = value
        else:
            self.tree.links.new(value,self.node.inputs["Value"])

class StringJoin(BlueNode):
    def __init__(self,tree,location=(0,0),delimiter="",strings = None,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeStringJoin")
        super().__init__(tree,location=location,**kwargs)

        self.std_out = self.node.outputs["String"]


        if isinstance(delimiter, str):
            self.node.inputs["Delimiter"].default_value = delimiter
        else:
            self.tree.links.new(delimiter,self.node.inputs["Delimiter"])

        if strings:
            self.tree.links.new(strings,self.node.inputs["Strings"])

class SliceString(BlueNode):
    def __init__(self,tree,location=(0,0),value=0,string = None, position=None,length=None,**kwargs):
        self.node=tree.nodes.new(type="FunctionNodeSliceString")
        super().__init__(tree,location=location,**kwargs)

        self.std_out = self.node.outputs["String"]

        if position is not None:
            if isinstance(position, (int)):
                self.node.inputs["Position"].default_value = position
            else:
                self.tree.links.new(position,self.node.inputs["Position"])

        if length is not None:
            if isinstance(length, (int)):
                self.node.inputs["Length"].default_value = length
            else:
                self.tree.links.new(length,self.node.inputs["Length"])

        if string is not None:
            if isinstance(string,str):
                self.node.inputs["String"].default_value = string
            else:
                self.tree.links.new(string,self.node.inputs["String"])

# default meshes

class Grid(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 size_x=10,
                 size_y=10,
                 vertices_x=11,
                 vertices_y=11, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshGrid")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(size_x, (int, float)):
            self.node.inputs["Size X"].default_value = size_x
        else:
            self.tree.links.new(size_x, self.node.inputs["Size X"])
        if isinstance(size_y, (int, float)):
            self.node.inputs["Size Y"].default_value = size_y
        else:
            self.tree.links.new(size_y, self.node.inputs["Size Y"])

        if isinstance(vertices_x, int):
            self.node.inputs["Vertices X"].default_value = vertices_x
        else:
            self.tree.links.new(vertices_x, self.node.inputs["Vertices X"])
        if isinstance(vertices_y, int):
            self.node.inputs["Vertices Y"].default_value = vertices_y
        else:
            self.tree.links.new(vertices_y, self.node.inputs["Vertices Y"])

class UVSphere(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 radius=1,
                 segments=64, rings=32, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshUVSphere")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs["Radius"])

        if isinstance(segments, int):
            self.node.inputs["Segments"].default_value = segments
        else:
            self.tree.links.new(segments, self.node.inputs["Segments"])

        if isinstance(rings, int):
            self.node.inputs["Rings"].default_value = rings
        else:
            self.tree.links.new(rings, self.node.inputs["Rings"])

class IcoSphere(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 radius=0.1,
                 subdivisions=1, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshIcoSphere")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs["Radius"])
        if isinstance(subdivisions, int):
            self.node.inputs["Subdivisions"].default_value = subdivisions
        else:
            self.tree.links.new(subdivisions, self.node.inputs["Subdivisions"])

class CubeMesh(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 size=[1, 1, 1], vertices_x=2, vertices_y=2, vertices_z=2, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshCube")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.uv_out = self.node.outputs["UV Map"]

        if isinstance(vertices_x, int):
            self.node.inputs["Vertices X"].default_value = vertices_x
        else:
            self.node.inputs["Vertices X"] = vertices_x
        if isinstance(vertices_y, int):
            self.node.inputs["Vertices Y"].default_value = vertices_y
        else:
            self.node.inputs["Vertices Y"] = vertices_y
        if isinstance(vertices_z, int):
            self.node.inputs["Vertices Z"].default_value = vertices_z
        else:
            self.node.inputs["Vertices Z"] = vertices_z
        if isinstance(size, (int, float)):
            self.node.inputs["Size"].default_value = [size] * 3
        elif isinstance(size, (list, Vector)):
            self.node.inputs["Size"].default_value = size
        else:
            self.tree.links.new(size, self.node.inputs["Size"])

class CylinderMesh(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 fill_type="TRIANGLE_FAN", vertices=32,
                 side_segments=1, fill_segments=1,
                 radius=1, depth=2, **kwargs
                 ):
        """
        :param fill_type: ("NONE", "NGON", "TRIANGLE_FAN")

        """

        self.node = tree.nodes.new(type="GeometryNodeMeshCylinder")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.top_out = self.node.outputs["Top"]
        self.side_out = self.node.outputs["Side"]
        self.bottom_out = self.node.outputs["Bottom"]
        self.uv_out = self.node.outputs["UV Map"]

        self.node.fill_type = fill_type

        if isinstance(vertices, int):
            self.node.inputs["Vertices"].default_value = vertices
        else:
            self.links.new(vertices,self.node.inputs["Vertices"])

        if isinstance(side_segments, int):
            self.node.inputs["Side Segments"].default_value = side_segments
        else:
            self.links.new(side_segments,self.node.inputs["Side Segments"])

        if isinstance(fill_segments, int):
            self.node.inputs["Fill Segments"].default_value = fill_segments
        else:
            self.links.new(fill_segments,self.node.inputs["Fill Segments"])

        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.links.new(radius,self.node.inputs["Radius"])

        if isinstance(depth, (int, float)):
            self.node.inputs["Depth"].default_value = depth
        else:
            self.links.new(depth,self.node.inputs["Depth"])

class ConeMesh(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 fill_type="TRIANGLE_FAN", vertices=32, side_segments=1, fill_segments=1,
                 radius_top=0,radius_bottom=1, depth=2, **kwargs):
        """
        :param fill_type: ("NONE", "NGON", "TRIANGLE_FAN")

        """

        self.node = tree.nodes.new(type="GeometryNodeMeshCone")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.top_out = self.node.outputs["Top"]
        self.side_out = self.node.outputs["Side"]
        self.bottom_out = self.node.outputs["Bottom"]
        self.uv_out = self.node.outputs["UV Map"]

        self.node.fill_type = fill_type

        if isinstance(vertices, int):
            self.node.inputs["Vertices"].default_value = vertices
        else:
            self.links.new(vertices, self.node.inputs["Vertices"])

        if isinstance(side_segments, int):
            self.node.inputs["Side Segments"].default_value = side_segments
        else:
            self.links.new(side_segments, self.node.inputs["Side Segments"])

        if isinstance(fill_segments, int):
            self.node.inputs["Fill Segments"].default_value = fill_segments
        else:
            self.links.new(fill_segments, self.node.inputs["Fill Segments"])

        if isinstance(radius_top, (int, float)):
            self.node.inputs["Radius Top"].default_value = radius_top
        else:
            self.links.new(radius_top, self.node.inputs["Radius Top"])

        if isinstance(radius_bottom, (int, float)):
            self.node.inputs["Radius Bottom"].default_value = radius_bottom
        else:
            self.links.new(radius_bottom, self.node.inputs["Radius Bottom"])

        if isinstance(depth, (int, float)):
            self.node.inputs["Depth"].default_value = depth
        else:
            self.links.new(depth, self.node.inputs["Depth"])

# instances
class InstanceOnPoints(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 points=None,
                 selection=None,
                 instance=None,
                 instance_index=None,
                 rotation=Vector([0, 0, 0]),
                 scale=Vector([1, 1, 1]), **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeInstanceOnPoints")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Instances"]
        self.geometry_in = self.node.inputs["Points"]

        if isinstance(rotation, (list,Vector)):
            self.node.inputs["Rotation"].default_value = rotation
        else:
            self.tree.links.new(rotation, self.node.inputs["Rotation"])
        if isinstance(scale, (Vector,list)):
            self.node.inputs["Scale"].default_value = scale
        else:
            self.tree.links.new(scale, self.node.inputs["Scale"])

        if points:
            self.tree.links.new(points, self.node.inputs["Points"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if instance:
            self.tree.links.new(instance, self.node.inputs["Instance"])
        if instance_index:
            self.tree.links.new(instance_index, self.node.inputs["Instance Index"])

class InstanceOnEdges(GreenNode):
    def __init__(self, tree, location=(0, 0), selection=None,
                 radius=0.1, resolution=8, name="InstanceOnEdges", **kwargs
                 ):
        mesh2curve = MeshToCurve(tree, selection=selection)
        profile = CurveCircle(tree, resolution=resolution, radius=radius, name=name + "Circle")
        curve2mesh = CurveToMesh(tree, profile_curve=profile.geometry_out)
        tree.links.new(mesh2curve.geometry_out, curve2mesh.geometry_in)

        self.node = curve2mesh
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = mesh2curve.inputs["Mesh"]

class ScaleInstances(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 selection=None,
                 scale=1,
                 center=None, **kwargs
                 ):


        self.node = tree.nodes.new(type="GeometryNodeScaleInstances")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Instances"]
        self.geometry_in = self.node.inputs["Instances"]

        if isinstance(scale, (int, float)):
            self.node.inputs["Scale"].default_value = [scale]*3
        elif isinstance(scale, (list, Vector)):
            self.node.inputs["Scale"].default_value = scale
        else:
            self.tree.links.new(scale, self.node.inputs["Scale"])

        if center:
            if isinstance(center, (list, Vector)):
                self.node.inputs["Center"].default_value = center
            else:
                self.tree.links.new(center, self.node.inputs["Center"])

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])


class SetPosition(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 selection=None,
                 position=Vector([0, 0, 0]),
                 offset=Vector([0, 0, 0]), **kwargs
                 ):
        """

        :param tree:
        :param location:
        :param geometry:
        :param selection:
        :param position:
        :param offset:
        :param kwargs:
        """

        self.node = tree.nodes.new(type="GeometryNodeSetPosition")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        if isinstance(position, (Vector,list)):
            self.node.inputs["Position"].default_value = position
        else:
            self.tree.links.new(position, self.node.inputs["Position"])
        if isinstance(offset, (list,Vector)):
            self.node.inputs["Offset"].default_value = offset
        else:
            self.tree.links.new(offset, self.node.inputs["Offset"])

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])

class ScaleElements(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 domain="FACE",
                 scale_mode="UNIFORM",
                 selection=None,
                 scale=1,
                 center=None, **kwargs
                 ):
        """

        :param tree:
        :param location:
        :param geometry:
        :param domain:
        :param scale_mode:
        :param selection:
        :param scale:
        :param center:
        :param kwargs:
        """

        self.node = tree.nodes.new(type="GeometryNodeScaleElements")
        super().__init__(tree, location=location, **kwargs)

        self.node.domain = domain
        if blender_version()<(5,0):
            self.node.scale_mode = scale_mode
        else:
            if scale_mode=="UNIFORM":
                scale_mode="Uniform"
            else:
                scale_mode="Single Axis"
            self.node.inputs["Scale Mode"].default_value = scale_mode

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        if isinstance(scale, (int, float)):
            self.node.inputs["Scale"].default_value = scale
        else:
            self.tree.links.new(scale, self.node.inputs["Scale"])

        if center:
            if isinstance(center, (list, Vector)):
                self.node.inputs["Center"].default_value = center
            else:
                self.tree.links.new(center, self.node.inputs["Center"])

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])

class RealizeInstances(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,**kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeRealizeInstances")
        super().__init__(tree, location=location,**kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])

class RotateInstances(GreenNode):
    def __init__(self, tree, location=(0, 0),instances=None,
                 selection=None,
                 rotation=None,pivot_point=None,local_space=True,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeRotateInstances")
        super().__init__(tree,location=location,**kwargs)

        self.geometry_out=self.node.outputs["Instances"]
        self.geometry_in=self.node.inputs["Instances"]

        if instances:
            self.tree.links.new(instances,self.node.inputs["Instances"])
        if selection:
            if isinstance(selection,bool):
                self.node.inputs["Selection"].default_value=selection
            else:
                self.tree.links.new(selection,self.node.inputs["Selection"])
        if rotation:
            if isinstance(rotation,(Vector,list)):
                self.node.inputs["Rotation"].default_value=rotation
            else:
                self.tree.links.new(rotation,self.node.inputs["Rotation"])
        if pivot_point:
            if isinstance(pivot_point,(Vector,list)):
                self.node.inputs["Pivot Point"].default_value=pivot_point
            else:
                self.tree.links.new(pivot_point,self.node.inputs["Pivot Point"])

        if isinstance(local_space,bool):
            self.node.inputs["Local Space"].default_value=local_space
        else:
            self.tree.links.new(local_space,self.node.inputs["Local Space"])

class TranslateInstances(GreenNode):
    def __init__(self, tree, location=(0, 0), instances=None,selection=None,translation=None,
                 local_space=True,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeTranslateInstances")
        super().__init__(tree,location=location,**kwargs)

        self.geometry_out=self.node.outputs["Instances"]
        self.geometry_in=self.node.inputs["Instances"]

        if selection:
            if isinstance(selection, bool):
                self.node.inputs["Selection"].default_value = selection
            else:
                self.tree.links.new(selection, self.node.inputs["Selection"])
        if translation:
            if isinstance(translation, Vector):
                self.node.inputs["Translation"].default_value = translation
            else:
                self.tree.links.new(translation, self.node.inputs["Translation"])
        if isinstance(local_space, bool):
            self.node.inputs["Local Space"].default_value = local_space
        else:
            self.tree.links.new(local_space, self.node.inputs["Local Space"])

class JoinGeometry(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeJoinGeometry")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        if isinstance(geometry, list):
            for geo in geometry:
                self.tree.links.new(geo, self.node.inputs["Geometry"])
        elif geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])

class SeparateGeometry(GreenNode):
    def __init__(self,tree,location=(0,0),
                 domain="POINT",geometry=None,selection=None,geometry_out="Selection",**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeSeparateGeometry")
        self.node.domain = domain

        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs[geometry_out]
        self.geometry_in = self.node.inputs["Geometry"]
        self.selection = self.node.outputs["Selection"]
        self.inverse = self.node.outputs["Inverted"]

        if selection:
            self.tree.links.new(selection,self.node.inputs["Selection"])

class GeometryToInstance(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 **kwargs):
        self.node=tree.nodes.new(type="GeometryNodeGeometryToInstance")
        super().__init__(tree,location=location,**kwargs)
        self.geometry_out=self.node.outputs["Instances"]
        self.geometry_in=self.node.inputs["Geometry"]

class DeleteGeometry(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 domain="POINT",
                 mode="ALL",
                 geometry=None,
                 selection=None,
                 **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeDeleteGeometry")
        self.node.domain = domain
        self.node.mode = mode
        super().__init__(tree, location=location, **kwargs)

        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs["Geometry"]

        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])

class SortElements(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 domain="FACE",
                 geometry=None,
                 selection=None,
                 group_id=None,
                 sort_weight=None,
                 **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeSortElements")
        self.node.domain = domain

        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs["Geometry"]

        super().__init__(tree,location=location,**kwargs)

        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if group_id:
            self.node.inputs["Group ID"].default_value = group_id
        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])
        if sort_weight:
            if isinstance(sort_weight,(int,float)):
                self.node.inputs["Sort Weight"].default_value = sort_weight
            else:
                self.tree.links.new(sort_weight,self.node.inputs["Sort Weight"])

class RayCast(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 data_type="FLOAT",
                 mapping="INTERPOLATED",
                 target_geometry=None,
                 attribute=None,
                 source_position=None,
                 ray_direction=Vector([0, 0, -1]),
                 ray_length=100, **kwargs
                 ):
        """

        :param tree:
        :param location:
        :param data_type:"FLOAT", "INT", "FLOAT_VECTOR", "FLOAT_COLOR", "BYTE_COLOR", "BOOLEAN", "FLOAT2", "QUATERNION"
        :param mapping:"INTERPOLATED","NEAREST"
        :param target_geometry:
        :param attribute:
        :param source_position:
        :param ray_direction:
        :param ray_length:
        """

        self.node = tree.nodes.new(type="GeometryNodeRaycast")
        self.node.data_type = data_type
        self.node.mapping = mapping
        super().__init__(tree, location=location, **kwargs)

        self.geometry_in = self.node.inputs["Target Geometry"]

        if target_geometry:
            self.tree.links.new(target_geometry, self.node.inputs["Target Geometry"])
        if attribute:
            self.tree.links.new(attribute, self.node.inputs["Attribute"])
        if source_position:
            self.tree.links.new(source_position, self.node.inputs["Source Position"])
        if isinstance(ray_direction, (Vector, list)):
            self.node.inputs["Ray Direction"].default_value = ray_direction
        else:
            self.tree.links.new(ray_direction, self.node.inputs["Ray Direction"])
        if isinstance(ray_length, (int, float)):
            self.node.inputs["Ray Length"].default_value = ray_length
        else:
            self.tree.links.new(ray_length, self.node.inputs["Ray Length"])

class ConvexHull(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,**kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeConvexHull")
        super().__init__(tree, location=location,**kwargs)

        self.geometry_out = self.node.outputs["Convex Hull"]
        self.geometry_in = self.node.inputs["Geometry"]

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])

class BoundingBox(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeBoundBox")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Bounding Box"]
        self.geometry_in = self.node.inputs["Geometry"]

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])

class MeshBoolean(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 operation="DIFFERENCE",solver="FLOAT",
                 mesh_1=None,mesh_2=None,self_intersection=False,
                 hole_tolerant=False, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeMeshBoolean")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Mesh 1"]
        self.node.operation = operation
        self.node.solver = solver

        if mesh_1:
            self.tree.links.new(mesh_1, self.node.inputs["Mesh 1"])
        if mesh_2:
            self.tree.links.new(mesh_2, self.node.inputs["Mesh 2"])

        if solver=="EXACT":
            if isinstance(self_intersection, bool):
                self.node.inputs["Self Intersection"].default_value = self_intersection
            else:
                tree.links.new(self_intersection, self.node.inputs["Self Intersection"])

            if isinstance(hole_tolerant,bool):
                self.node.inputs["Hole Tolerant"].default_value=hole_tolerant
            else:
                tree.links.new(hole_tolerant,self.node.inputs["Hole Tolerant"])

#################
## Attributes ###
#################
class StoredNamedAttribute(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 data_type="FLOAT",
                 domain="POINT",
                 selection=None,
                 name="attribute",
                 value=None, **kwargs
                 ):
        """
           :param tree:
           :param location:
           :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "FLOAT_COLOR", "BYTE_COLOR", "BOOLEAN", "FLOAT2", "INT8", "QUATERNION", "FLOAT4X4"
           :param domain: "POINT", "FACE", "EDGE", and more
           :param name: name of the attribute
           :param value: value to store
           :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeStoreNamedAttribute")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        self.node.domain = domain
        self.node.data_type = data_type

        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])

        if isinstance(name, str):
            self.node.inputs["Name"].default_value = name
        else:
            self.tree.links.new(name, self.node.inputs["Name"])

        if value is not None:
            if isinstance(value, (int, float, Vector, list)):
                self.node.inputs["Value"].default_value = value
            else:
                self.tree.links.new(value, self.node.inputs["Value"])

class NamedAttribute(RedNode):
    def __init__(self, tree, location=(0, 0),
                 data_type="FLOAT",
                 name="attribute", **kwargs
                 ):
        """
           :param tree:
           :param location:
           :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "FLOAT_COLOR", "BYTE_COLOR", "BOOLEAN", "FLOAT2", "INT8", "QUATERNION", "FLOAT4X4"
           :param name: name of the attribute
           :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeInputNamedAttribute")
        super().__init__(tree, location=location, **kwargs)

        # the order of the following lines matters, since the output depends on the data_type
        self.node.data_type = data_type
        self.std_out = self.node.outputs["Attribute"]

        if isinstance(name, str):
            self.node.inputs["Name"].default_value = name
        else:
            self.tree.links.new(name, self.node.inputs["Name"])

class AttributeStatistic(BlueNode):
    def __init__(self, tree, location=(0, 0),data_type="FLOAT",domain="POINT",
                 geometry=None,selection=None,attribute=None,
                 std_out="Mean",**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeAttributeStatistic")
        super().__init__(tree,location=location,**kwargs)

        self.node.data_type=data_type
        self.node.domain=domain

        self.std_out = self.node.outputs[std_out]
        self.geometry_in = self.node.inputs["Geometry"]

        if geometry:
            tree.links.new(geometry,self.node.inputs["Geometry"])
        if selection:
            if isinstance(selection, bool):
                self.node.inputs["Selection"].default_value = selection
            else:
                self.tree.links.new(selection, self.node.inputs["Selection"])
        if attribute:
            self.tree.links.new(attribute, self.node.inputs["Attribute"])

class CollectionInfo(RedNode):
    def __init__(self, tree, location=(0, 0),
                 transform_space="ORIGINAL",
                 collection_name="Collection",
                 separate_children=False,
                 reset_children=False, **kwargs
                 ):
        """
           :param tree:
           :param location:
           :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "FLOAT_COLOR", "BYTE_COLOR", "BOOLEAN", "FLOAT2", "QUATERNION"
           :param name: name of the attribute
           :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeCollectionInfo")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Instances"]
        self.node.transform_space = transform_space
        self.node.inputs[0].default_value = ibpy.get_collection(collection_name)
        if isinstance(separate_children, bool):
            self.node.inputs[1].default_value = separate_children
        else:
            self.tree.links.new(separate_children, self.node.inputs[1])
        if isinstance(reset_children, bool):
            self.node.inputs[2].default_value = reset_children
        else:
            self.tree.links.new(reset_children, self.node.inputs[2])

class SetMaterial(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 selection=None,
                 material="drawing", **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeSetMaterial")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if callable(material):  # call passed function
            if "attribute_names" in kwargs:
                material = material(attribute_names=kwargs.pop("attribute_names"), **kwargs)
            else:
                material = material(**kwargs)
            self.inputs["Material"].default_value = material
        elif isinstance(material, str):  # create material from passed string
            material = get_material(material, **kwargs)
            self.inputs["Material"].default_value = material
        elif isinstance(material, bpy.types.Material):
            self.inputs["Material"].default_value = material
        else:  # link socket
            self.tree.links.new(material, self.inputs["Material"])
        self.material = material

class MergeByDistance(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 mode="ALL",
                 selection=None,
                 distance=0.001,
                 **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMergeByDistance")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        self.node.mode = mode

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if isinstance(distance, (int, float)):
            self.node.inputs["Distance"].default_value = distance
        elif distance:
            self.tree.links.new(distance, self.node.inputs["Distance"])

class SetShadeSmooth(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 selection=None,
                 shade_smooth=True, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeSetShadeSmooth")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        if geometry:
            self.tree.links.new(geometry, self.node.inputs["Geometry"])
        if selection:
            self.tree.links.new(selection, self.node.inputs["Selection"])
        if isinstance(shade_smooth, bool):
            self.inputs["Shade Smooth"].default_value = shade_smooth
        else:
            self.tree.links.new(shade_smooth, self.inputs["Shade Smooth"])

class TransformGeometry(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry = None,
                 mode = "Components",
                 translation_x=None,
                 translation_y=None,
                 translation_z=None,
                 translation=Vector(),
                 rotation=Vector(),
                 scale=Vector([1, 1, 1]), **kwargs
                 ):
        """
        :param translation_x: only an x component for the translation is given, this overrides the translation parameter:
        :param translation_y: only a y component for the translation is given, this overrides the translation parameter:
        :param translation_z: only a z component for the translation is given, this overrides the translation parameter:

        """

        self.node = tree.nodes.new(type="GeometryNodeTransform")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]
        self.geometry_in = self.node.inputs["Geometry"]

        self.node.inputs["Mode"].default_value=mode

        if geometry:
            tree.links.new(geometry, self.node.inputs["Geometry"])

        if translation_z is not None or translation_y is not None or translation_x is not None:
            if translation_z is None:
                translation_z=0
            if translation_y is None:
                translation_y=0
            if translation_x is None:
                translation_x =0
            sep = CombineXYZ(tree,location=(location[0]-1,location[1]),
                             x=translation_x,y=translation_y,z=translation_z)
            tree.links.new(sep.std_out,self.inputs["Translation"])
        elif isinstance(translation, (list, Vector)):
            self.inputs["Translation"].default_value = translation
        else:
            self.tree.links.new(translation, self.inputs["Translation"])

        if rotation is not None:
            if isinstance(rotation, (list, Vector)):
                self.inputs["Rotation"].default_value = rotation
            else:
                self.tree.links.new(rotation, self.inputs["Rotation"])

        if isinstance(scale, (list, Vector)):
            self.inputs["Scale"].default_value = scale
        else:
            self.tree.links.new(scale, self.inputs["Scale"])


class DomainSize(GreenNode):
    """
    Geometry node DomainSize
    retrieves information about the geometry that is piped into it

    possible output nodes:
    Point Count
    Edge Count
    Face Count
    Face Corner Count
    """

    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 component="MESH", **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeAttributeDomainSize")
        super().__init__(tree, location=location, **kwargs)

        self.node.component = component
        if geometry is not None:
            tree.links.new(geometry, self.node.inputs["Geometry"])

        self.geometry_in = self.node.inputs["Geometry"]

class SampleIndex(GreenNode):
    """
    Geometry node SampleIndex
    retrieves information about a specified value for a geometric object with a given index
    """

    def __init__(self, tree, location=(0, 0),
                 data_type="FLOAT_VECTOR",
                 domain="POINT",
                 geometry=None,
                 value=None,
                 index=None, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeSampleIndex")
        super().__init__(tree, location=location, **kwargs)

        self.node.data_type = data_type
        self.node.domain = domain

        if geometry is not None:
            tree.links.new(geometry, self.node.inputs["Geometry"])

        if value is not None:
            tree.links.new(value, self.node.inputs["Value"])

        if isinstance(index, int):
            self.node.inputs["Index"].default_value = index
        elif index is not None:
            tree.links.new(index, self.node.inputs["Index"])

        self.geometry_in = self.node.inputs["Geometry"]
        self.std_out = self.node.outputs["Value"]

###################
## Utility Nodes ##
###################

class EvaluateOnDomain(BlueNode):
    def __init__(self, tree, location=(0, 0), value=None,data_type="FLOAT_VECTOR",domain="FACE", **kwargs):
        """

        :param tree:
        :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "BOOLEAN"
        :param domain: "FACE","POINT","EDGE",...
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeFieldOnDomain")
        self.node.data_type=data_type
        self.node.domain=domain
        super().__init__(tree, location=location, **kwargs)

        self.std_in = self.node.inputs["Value"]
        self.std_out = self.node.outputs["Value"]

        if value is not None:
            if isinstance(value, (int, float)):
                self.node.inputs["Value"].default_value = value
            else:
                tree.links.new(value, self.node.inputs["Value"])

class EvaluateAtIndex(BlueNode):
    def __init__(self, tree, location=(0, 0), value=None,data_type="FLOAT_VECTOR",
                 domain="FACE",index=None, **kwargs):
        """

        :param tree:
        :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "BOOLEAN"
        :param domain: "FACE","POINT","EDGE",...
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeFieldAtIndex")
        self.node.data_type=data_type
        self.node.domain=domain
        super().__init__(tree, location=location, **kwargs)

        self.std_in = self.node.inputs["Value"]
        self.std_out = self.node.outputs["Value"]

        if value is not None:
            if isinstance(value, (int, float)):
                self.node.inputs["Value"].default_value = value
            else:
                tree.links.new(value, self.node.inputs["Value"])
        if index is not None:
            if isinstance(index, int):
                self.index.default_value = index
            else:
                tree.links.new(index, self.node.inputs["Index"])

#  red nodes   #

class Position(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputPosition")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Position"]

class FaceArea(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputMeshFaceArea")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Area"]

class InputNormal(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputNormal")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Normal"]

class InputMaterial(RedNode):
    def __init__(self, tree, location=(0, 0),material=None, **kwargs):
        """
        create an MaterialINput node
        :param tree:
        :param location:
        :param material:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeInputMaterial")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Material"]
        if material is not None:
            if isinstance(material, str):
                self.node.material=ibpy.get_material(material,**kwargs)
            elif isinstance(material, bpy.types.Material):
                self.node.material=material

class Index(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputIndex")
        super().__init__(tree, location=location, **kwargs)
        self.std_out = self.node.outputs["Index"]

class EdgeVertices(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputMeshEdgeVertices")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Vertex Index 1"]

# input nodes
class InputValue(RedNode):
    def __init__(self, tree, location=(0, 0), value=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeValue")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Value"]
        self.outputs["Value"].default_value = value

class InputInteger(RedNode):
    def __init__(self, tree, location=(0, 0), integer=0, **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInputInt")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Integer"]
        self.node.integer = integer

class InputString(RedNode):
    def __init__(self, tree, location=(0, 0), string="", **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInputString")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["String"]
        self.node.string=string

class InputBoolean(RedNode):
    def __init__(self, tree, location=(0, 0), value=True, **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInputBool")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Boolean"]
        self.node.boolean = value

class InputVector(RedNode):
    def __init__(self, tree, location=(0, 0), value=Vector()
                 , **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInputVector")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[0]
        self.node.vector = value

class InputRotation(RedNode):
    def __init__(self, tree, location=(0, 0), rotation=Vector()
                 , **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInputRotation")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Rotation"]
        self.node.rotation_euler = rotation

class ObjectInfo(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 transform_space="RELATIVE",
                 as_instance=False,
                 object=None, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeObjectInfo")
        super().__init__(tree, location=location, **kwargs)
        self.node.transform_space = transform_space

        if object is not None:
            self.node.inputs["Object"].default_value = get_obj(object)

        self.geometry_out = self.node.outputs["Geometry"]

        if isinstance(as_instance,bool):
            self.node.inputs["As Instance"].default_value = as_instance
        else:
            tree.links.new(as_instance,self.node.inputs["As Instance"])

class ImportCSV(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 delimiter=",",
                 path=None, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeImportCSV")
        super().__init__(tree, location=location, **kwargs)


        self.geometry_out = self.node.outputs["Point Cloud"]

        if isinstance(delimiter,str):
            self.node.inputs["Delimiter"].default_value =delimiter
        else:
            tree.links.new(delimiter,self.node.inputs["Delimiter"])

        if path is not None:
            self.node.inputs["Path"].default_value = path

class SceneTime(RedNode):
    def __init__(self, tree, location=(0, 0), std_out="Seconds", **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputSceneTime")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[std_out]

# Function Nodes #

class InvertRotation(RedNode):
    def __init__(self, tree, location=(0, 0), in_rotation=Vector()
                 , **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInvertRotation")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Rotation"]

        if isinstance(in_rotation,(list,Vector)):
            self.node.inputs["Rotation"].default_value=in_rotation
        else:
            tree.links.new(in_rotation,self.node.inputs["Rotation"])

class RotateRotation(RedNode):
    def __init__(self, tree, location=(0, 0),rotation_space="GLOBAL", rotation=Vector(), rotate_by=Vector()
                 , **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeRotateRotation")
        super().__init__(tree, location=location, **kwargs)
        self.node.rotation_space=rotation_space

        self.std_out = self.node.outputs["Rotation"]

        if isinstance(rotation,(list,Vector)):
            self.node.inputs["Rotation"].default_value=rotation
        else:
            tree.links.new(rotation,self.node.inputs["Rotation"])

        if isinstance(rotate_by,(list,Vector)):
            self.node.inputs["Rotate By"].default_value=rotate_by
        else:
            tree.links.new(rotate_by,self.node.inputs["Rotate By"])

class AlignRotationToVector(RedNode):
    def __init__(self, tree, location=(0, 0), pivot_axis="AUTO",axis="Z",rotation=None,factor=1,vector=Vector(), **kwargs):

        self.node = tree.nodes.new(type="FunctionNodeAlignRotationToVector")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Rotation"]
        self.node.pivot_axis = pivot_axis
        self.node.axis = axis
        if rotation is not None:
            if isinstance(rotation,(list,Vector)):
                self.node.inputs["Rotation"].vector=rotation
            else:
                tree.links.new(rotation,self.node.inputs["Rotation"])

        if isinstance(vector,(list,Vector)):
            self.node.inputs["Vector"].default_value=vector
        else:
            tree.links.new(vector,self.node.inputs["Vector"])
        if isinstance(factor,(int,float )):
            self.node.inputs["Factor"].default_value=factor
        else:
            tree.links.new(factor,self.node.inputs["Factor"])

class AxesToRotation(BlueNode):
    def __init__(self, tree, location=(0, 0),primary_axis='X',secondary_axis='Z',primary_direction=None,secondary_direction=None, **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeAxesToRotation")
        super().__init__(tree,location=location,**kwargs)

        self.std_out=self.node.outputs["Rotation"]
        self.node.primary_axis=primary_axis
        self.node.secondary_axis=secondary_axis

        if primary_direction is not None:
            if isinstance(primary_direction,(list,Vector)):
                self.node.inputs["Primary Axis"].default_value=primary_direction
            else:
                tree.links.new(primary_direction,self.node.inputs["Primary Axis"])
        if secondary_direction is not None:
            if isinstance(secondary_direction,(list,Vector)):
                self.node.inputs["Secondary Axis"].default_value=secondary_direction
            else:
                tree.links.new(secondary_direction,self.node.inputs["Secondary Axis"])

class EulerToRotation(BlueNode):
    def __init__(self, tree, location=(0, 0),euler = Vector(), **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeEulerToRotation")
        super().__init__(tree,location=location,**kwargs)

        self.std_out=self.node.outputs["Rotation"]

        if isinstance(euler,(list,Vector)):
            self.node.inputs["Euler"].default_value=euler
        else:
            tree.links.new(euler,self.node.inputs["Euler"])

class AxisAngleToRotation(BlueNode):
    def __init__(self, tree, location=(0, 0),axis=None,angle=0, **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeAxisAngleToRotation")
        super().__init__(tree,location=location,**kwargs)

        self.std_out=self.node.outputs["Rotation"]


        if axis is not None:
            if isinstance(axis,(list,Vector)):
                self.node.inputs["Axis"].default_value=axis
            else:
                tree.links.new(axis,self.node.inputs["Axis"])

        if isinstance(angle,(int,float)):
            self.node.inputs["Angle"].default_value=angle
        else:
            tree.links.new(angle,self.node.inputs["Angle"])


class RotateVector(BlueNode):
    def __init__(self,tree,location=(0,0),vector=Vector(),rotation=Vector(), **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeRotateVector")
        super().__init__(tree,location=location,**kwargs)

        self.std_out=self.node.outputs["Vector"]

        if isinstance(vector,(list,Vector)):
            self.node.inputs["Vector"].default_value=vector
        else:
            tree.links.new(vector,self.node.inputs["Vector"])

        if isinstance(rotation,(list,Vector)):
            self.node.inputs["Rotation"].default_value=rotation
        else:
            tree.links.new(rotation,self.node.inputs["Rotation"])

class QuaternionToRotation(BlueNode):
    def __init__(self, tree, location=(0, 0), w=1,x=0,y=0,z=0, **kwargs):

        self.node = tree.nodes.new(type="FunctionNodeQuaternionToRotation")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Rotation"]

        if isinstance(w,(int,float)):
            self.node.inputs["W"].default_value=w
        else:
            tree.links.new(w,self.node.inputs["W"])

        if isinstance(x,(int,float)):
            self.node.inputs["X"].default_value=x
        else:
            tree.links.new(x,self.node.inputs["X"])

        if isinstance(y,(int,float)):
            self.node.inputs["Y"].default_value=y
        else:
            tree.links.new(y,self.node.inputs["Y"])

        if isinstance(z,(int,float)):
            self.node.inputs["Z"].default_value=z
        else:
            tree.links.new(z,self.node.inputs["Z"])

# blue nodes
class RandomValue(BlueNode):
    def __init__(self, tree, data_type="FLOAT_VECTOR", location=(0, 0), min=-1 * Vector([1, 1, 1]),
                 max=Vector([1, 1, 1]),probability = 0.5,  seed=0, **kwargs):
        """

        :param tree:
        :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "BOOLEAN"
        :param location:
        :param min:
        :param max:
        :param seed:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="FunctionNodeRandomValue")
        super().__init__(tree, location=location, **kwargs)

        self.node.data_type = data_type

        if data_type == "FLOAT_VECTOR":
            self.std_out = self.node.outputs[0]
        else:
            self.std_out == self.node.outputs[1]

        if self.node.data_type == "FLOAT":
            if isinstance(min, (int, float)):
                self.node.inputs[2].default_value = min
            elif isinstance(min, bpy.types.NodeSocketFloat):
                tree.links.new(min,self.node.inputs[2])

        elif data_type == "FLOAT_VECTOR":
            if isinstance(min, (list, Vector)):
                self.node.inputs[0].default_value = min
            elif isinstance(min, bpy.types.NodeSocketVector):
                tree.links.new(min,self.node.inputs[0])
        elif data_type == "INT":
            if isinstance(min, (int, float)):
                self.node.inputs[4].default_value = min
            elif isinstance(min, bpy.types.NodeSocketInt):
                tree.links.new(min,self.node.inputs[4])

        if probability is not None:
            if isinstance(probability, (int, float)):
                self.node.inputs[6].default_value = probability
            else:
                tree.links.new(probability,self.node.inputs[6])

        if self.node.data_type == "FLOAT":
            if isinstance(max, (int, float)):
                self.node.inputs[3].default_value = max
            elif isinstance(max, bpy.types.NodeSocketFloat):
                tree.links.new(max,self.node.inputs[3])
        elif data_type == "FLOAT_VECTOR":
            if isinstance(max, (list, Vector)):
                self.node.inputs[1].default_value = max
            elif isinstance(max, bpy.types.NodeSocketVector):
                tree.links.new(max,self.node.inputs[1])
        elif data_type == "INT":
            if isinstance(max, (int, float)):
                self.node.inputs[5].default_value = max
            elif isinstance(max, bpy.types.NodeSocketInt):
                tree.links.new(max,self.node.inputs[5])

        if isinstance(seed, (int, float)):
            self.node.inputs["Seed"].default_value = seed
        else:
            self.node.inputs["Seed"] = seed

class SeparateXYZ(BlueNode):
    def __init__(self, tree, location=(0, 0), vector=Vector(), **kwargs):
        """

        :param tree:
        :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "BOOLEAN"
        :param location:
        :param min:
        :param max:
        :param seed:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="ShaderNodeSeparateXYZ")
        super().__init__(tree, location=location, **kwargs)

        self.std_in = self.node.inputs["Vector"]
        self.x = self.node.outputs["X"]
        self.y = self.node.outputs["Y"]
        self.z = self.node.outputs["Z"]

        if isinstance(vector, (int, float, list, Vector)):
            self.std_in.default_value = vector
        else:
            tree.links.new(vector, self.std_in)

class CombineXYZ(BlueNode):
    def __init__(self, tree, location=(0, 0), x=0, y=0, z=0, **kwargs):
        """
        :param tree:
        :param data_type: "FLOAT", "INT", "FLOAT_VECTOR", "BOOLEAN"
        :param location:
        :param min:
        :param max:
        :param seed:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="ShaderNodeCombineXYZ")
        super().__init__(tree, location=location, **kwargs)

        self.std_in = self.node.inputs
        self.std_out = self.node.outputs["Vector"]

        if isinstance(x, (int, float)):
            self.node.inputs["X"].default_value = x
        else:
            if not isinstance(x, (bpy.types.NodeSocketFloat,bpy.types.NodeSocketInt)):
                x = x.std_out
            tree.links.new(x, self.node.inputs["X"])

        if isinstance(y, (int, float)):
            self.node.inputs["Y"].default_value = y
        else:
            if not isinstance(y, (bpy.types.NodeSocketFloat,bpy.types.NodeSocketInt)):
                y = y.std_out
            tree.links.new(y, self.node.inputs["Y"])

        if isinstance(z, (int, float)):
            self.node.inputs["Z"].default_value = z
        else:
            if not isinstance(z,(bpy.types.NodeSocketFloat,bpy.types.NodeSocketInt)):
                z = z.std_out
            tree.links.new(z, self.node.inputs["Z"])

class CombineMatrix(BlueNode):
    def __init__(self, tree, location=(0, 0), col1 = Vector(),col2=Vector(),col3=Vector(), col4=Vector([0,0,0,1]), **kwargs):
        """
            creates a 4x4 matrix, that can be used for transformations
        """
        self.node = tree.nodes.new(type="FunctionNodeCombineMatrix")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Matrix"]

        if isinstance(col1, (list,Vector)):
            for i in range(len(col1)):
                socket_label = "Column 1 Row "+str(i+1)
                self.node.inputs[socket_label].default_value = col1[i]

        if isinstance(col2, (list,Vector)):
            for i in range(len(col2)):
                socket_label = "Column 2 Row "+str(i+1)
                self.node.inputs[socket_label].default_value = col2[i]

        if isinstance(col3, (list,Vector)):
            for i in range(len(col3)):
                socket_label = "Column 3 Row "+str(i+1)
                self.node.inputs[socket_label].default_value = col3[i]

        if isinstance(col4, (list,Vector)):
            for i in range(len(col4)):
                socket_label = "Column 4 Row "+str(i+1)
                self.node.inputs[socket_label].default_value = col4[i]

class TransformPoint(BlueNode):
    def __init__(self, tree, location=(0, 0), vector=Vector(),transform=None, **kwargs):
        """
            creates a transformation node that allows to apply matrices to points
        """
        self.node = tree.nodes.new(type="FunctionNodeTransformPoint")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Vector"]

        if isinstance(vector, (list,Vector)):
            self.node.inputs["Vector"].default_value = vector
        else:
            tree.links.new(vector,self.node.inputs["Vector"])

        if transform is not None:
            tree.links.new(transform,self.node.inputs["Transform"])

class MapRange(BlueNode):
    def __init__(self,tree, location=(0,0),data_type="FLOAT",interpolation_type="LINEAR",
                 from_min=0,from_max=1,to_min=0,to_max=1,value=None, **kwargs):

        self.node = tree.nodes.new(type="ShaderNodeMapRange")
        super().__init__(tree,location=location,**kwargs)

        self.std_out = self.node.outputs["Result"]
        self.node.data_type = data_type
        self.node.interpolation_type = interpolation_type

        if isinstance(from_min,(int,float)):
            self.node.inputs["From Min"].default_value = from_min
        else:
            tree.links.new(from_min,self.node.inputs["From Min"])
        if isinstance(from_max,(int,float)):
            self.node.inputs["From Max"].default_value = from_max
        else:
            tree.links.new(from_max,self.node.inputs["From Max"])
        if isinstance(to_min,(int,float)):
            self.node.inputs["To Min"].default_value = to_min
        else:
            tree.links.new(to_min,self.node.inputs["To Min"])
        if isinstance(to_max,(int,float)):
            self.node.inputs["To Max"].default_value = to_max
        else:
            tree.links.new(to_max,self.node.inputs["To Max"])
        if value is not None:
            if isinstance(value,(int,float)):
                self.node.inputs["Value"].default_value = value
            else:
                tree.links.new(value,self.node.inputs["Value"])

class MixNode(BlueNode):
    def __init__(self, tree, location=(0, 0), data_type="VECTOR",factor_mode="UNIFORM",clamp_factor=False,factor = None,input_a=None,input_b=None,**kwargs):
        self.node=tree.nodes.new("ShaderNodeMix")
        super().__init__(tree,location=location,**kwargs)

        self.node.data_type=data_type
        self.node.factor_mode=factor_mode
        self.node.clamp_factor=clamp_factor
        self.std_out=self.node.outputs["Result"]

        if factor is not None:
            if isinstance(factor,(int,float)):
                self.node.inputs["Factor"].default_value=factor
            else:
                tree.links.new(factor,self.node.inputs["Factor"])
        if input_a is not None:
            if isinstance(input_a,(int,float)):
                self.node.inputs["A"].default_value=input_a
            else:
                tree.links.new(input_a,self.node.inputs["A"])
        if input_b is not None:
            if isinstance(input_b,(int,float)):
                self.node.inputs["B"].default_value=input_b
            else:
                tree.links.new(input_b,self.node.inputs["B"])

class MathNode(BlueNode):
    def __init__(self, tree, location=(0, 0), operation="ADD", inputs0=None,
                 inputs1=None, inputs2=None, **kwargs):
        """
        :param tree:
        :param location:
        :param operation: "ADD","SUBTRACT",...
        :param inputs0:
        :param inputs1:
        :param inputs2:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="ShaderNodeMath")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[0]

        if operation:
            self.node.operation = operation
        if inputs0:
            if isinstance(inputs0, (bool, int, float)):
                self.node.inputs[0].default_value = inputs0
            else:
                tree.links.new(inputs0, self.node.inputs[0])
        if inputs1:
            if isinstance(inputs1, (bool, int, float)):
                self.node.inputs[1].default_value = inputs1
            else:
                tree.links.new(inputs1, self.node.inputs[1])
        if inputs2:
            if isinstance(inputs2, (bool, int, float)):
                self.node.inputs[2].default_value = inputs2
            else:
                tree.links.new(inputs2, self.node.inputs[2])

class CompareNode(BlueNode):
    def __init__(self, tree, location=(0, 0), operation="EQUAL",
                 data_type="FLOAT",
                 inputs0=0,
                 inputs1=0,
                 inputs2=0, **kwargs):
        """

        """
        self.node = tree.nodes.new(type="FunctionNodeCompare")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Result"]
        self.node.data_type = data_type
        self.node.operation = operation

        if data_type=="FLOAT":
            if isinstance(inputs0,(bool,int,float)):
                self.node.inputs[0].default_value=inputs0
            else:
                tree.links.new(inputs0,self.node.inputs[0])
            if isinstance(inputs1,(bool,int,float)):
                self.node.inputs[1].default_value=inputs1
            else:
                tree.links.new(inputs1,self.node.inputs[1])
            # needed for float comparison
            if inputs2:
                if isinstance(inputs2, (bool, int, float)):
                    self.node.inputs[2].default_value = inputs2
                else:
                    tree.links.new(inputs2, self.node.inputs[2])
        elif data_type=="INT":
            if isinstance(inputs0,(bool,int,float)):
                self.node.inputs[2].default_value=inputs0
            else:
                tree.links.new(inputs0,self.node.inputs[2])
            if isinstance(inputs1,(bool,int,float)):
                self.node.inputs[3].default_value=inputs1
            else:
                tree.links.new(inputs1,self.node.inputs[3])
        elif data_type=="FLOAT_VECTOR":
            if isinstance(inputs0,(list,Vector)):
                self.node.inputs[4].default_value=inputs0
            else:
                tree.links.new(inputs0,self.node.inputs[4])
            if isinstance(inputs1,(list,Vector)):
                self.node.inputs[5].default_value=inputs1
            else:
                tree.links.new(inputs1,self.node.inputs[5])
        elif data_type=="COLOR":
            if isinstance(inputs0,(list,Vector)):
                self.node.inputs[6].default_value=inputs0
            else:
                tree.links.new(inputs0,self.node.inputs[6])
            if isinstance(inputs1,(list,Vector)):
                self.node.inputs[7].default_value=inputs1
            else:
                tree.links.new(inputs1,self.node.inputs[7])
        elif data_type=="STRING":
            if inputs0==0:
                self.node.inputs[8].default_value=""
            elif isinstance(inputs0,str):
                self.node.inputs[8].default_value=inputs0
            else:
                tree.links.new(inputs0,self.node.inputs[8])
            if inputs1==0:
                self.node.inputs[9].default_value=""
            elif isinstance(inputs1,str):
                self.node.inputs[9].default_value=inputs1
            else:
                tree.links.new(inputs1,self.node.inputs[9])

class Switch(BlueNode):
    def __init__(self, tree, location=(0, 0), input_type="GEOMETRY",switch=False,false=None,true=None,**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeSwitch")
        super().__init__(tree,loation=location,**kwargs)

        self.node.input_type=input_type
        self.std_out=self.node.outputs["Output"]

        if isinstance(switch,bool):
            self.node.inputs["Switch"].default_value=switch
        else:
            tree.links.new(switch,self.node.inputs["Switch"])

        if false is not None:
            tree.links.new(false,self.node.inputs["False"])

        if true is not None:
            tree.links.new(true,self.node.inputs["True"])

class BooleanMath(BlueNode):
    def __init__(self, tree, location=(0, 0), operation="AND", inputs0=True, inputs1=True, **kwargs):
        """
        :param tree:
        :param location:
        :param operation: "AND","OR",...
        :param inputs0:
        :param inputs1:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="FunctionNodeBooleanMath")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Boolean"]

        if operation:
            self.node.operation = operation
        if isinstance(inputs0, (bool, int, float)):
            self.node.inputs[0].default_value = inputs0
        else:
            tree.links.new(inputs0, self.node.inputs[0])
        if isinstance(inputs1, (bool, int, float)):
            self.node.inputs[1].default_value = inputs1
        else:
            tree.links.new(inputs1, self.node.inputs[1])

class VectorMath(BlueNode):
    def __init__(self, tree, location=(0, 0), operation="ADD", inputs0=Vector(), inputs1=Vector(), float_input=None,
                 **kwargs):
        """
        :param tree:
        :param location:
        :param operation: "AND","OR",...
        :param inputs0:
        :param inputs1:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="ShaderNodeVectorMath")
        super().__init__(tree, location=location, **kwargs)

        if operation in ("DOT", "LENGTH"):
            self.std_out = self.node.outputs[1]
        else:
            self.std_out = self.node.outputs["Vector"]

        if operation:
            self.node.operation = operation
        if isinstance(inputs0, (Vector, list)):
            self.node.inputs[0].default_value = inputs0
        else:
            tree.links.new(inputs0, self.node.inputs[0])
        if isinstance(inputs1, (Vector, list)):
            self.node.inputs[1].default_value = inputs1
        else:
            tree.links.new(inputs1, self.node.inputs[1])
        if float_input:
            if isinstance(float_input, (float, int)):
                self.node.inputs["Scale"].default_value = float_input
            else:
                tree.links.new(float_input, self.node.inputs["Scale"])

class VectorRotate(BlueNode):
    def __init__(self, tree, location=(0, 0),rotation_type="AXIS_ANGLE", vector= None, center = None, axis = None, angle = 0,
                 **kwargs):
        """

        """
        self.node = tree.nodes.new(type="ShaderNodeVectorRotate")
        super().__init__(tree, location=location, **kwargs)


        self.std_out = self.node.outputs["Vector"]

        self.node.rotation_type=rotation_type

        if isinstance(vector, (Vector, list)):
            self.node.inputs["Vector"].default_value = vector
        else:
            tree.links.new(vector, self.node.inputs["Vector"])

        if isinstance(center, (Vector, list)):
            self.node.inputs["Center"].default_value = center
        else:
            tree.links.new(center, self.node.inputs["Center"])

        if angle:
            if isinstance(angle, (float, int)):
                self.node.inputs["Angle"].default_value = angle
            else:
                tree.links.new(angle, self.node.inputs["Angle"])

        if axis:
            if isinstance(axis, (Vector, list)):
                self.node.inputs["Axis"].default_value = axis
            else:
                tree.links.new(axis, self.node.inputs["Axis"])

class Switch(BlueNode):
    def __init__(self, tree, location=(0, 0), input_type="GEOMETRY",
                 switch=None, false=None, true=None, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodeSwitch")
        self.node.input_type=input_type
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Output"]
        self.switch = self.node.inputs["Switch"]
        self.true = self.node.inputs["True"]
        self.false = self.node.inputs["False"]

        if switch:
            tree.links.new(switch, self.switch)

        if true:
            if isinstance(true, (bpy.types.NodeSocket)):
                tree.links.new(true, self.true)
            else:
                self.true.default_value = true
        if false:
            if isinstance(false, (bpy.types.NodeSocket)):
                tree.links.new(false, self.false)
            else:
                self.false.default_value = false

class IndexSwitch(BlueNode):
    def __init__(self, tree, location=(0, 0), data_type="GEOMETRY",
                 index=None, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodeIndexSwitch")
        self.tree = tree
        self.node.data_type=data_type
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Output"]
        self.index = self.node.inputs["Index"]
        self.slots = self.node.inputs
        self.added_items = 0

        if index:
            if isinstance(index, (int, float)):
                self.node.inputs["Index"].default_value = index
            else:
                tree.links.new(index, self.index)

    def add_new_item_from_xml(self,default_value=None):
        """
        do not use this function, use add_new_item instead
        it is only used for the xml import
        """
        self.new_item()
        if self.added_items==0:
            self.added_items=2 # these are the items existing by default
        self.added_items+=1
        if default_value:
            self.slots[self.added_items].default_value=default_value
        return True

    def add_item(self,socket):
        # """
        #  only use this function, when you add switch items,
        #  it assumes that you wire the index socket independently
        # """
        # if self.added_items>len(self.slots)-3:
        #     self.new_item()
        # if socket:
        #     if isinstance(socket,int):
        #         self.slots[self.added_items+1].default_value=socket
        #     else:
        #         self.tree.links.new(socket,self.slots[self.added_items+1])

        ibpy.add_item_to_switch(self.node,self.added_items,socket,self.tree)
        self.added_items+=1

    def new_item(self):
        self.node.index_switch_items.new()

# zones
class RepeatInput(GreenNode):
    def __init__(self,tree,location=(0,0),**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeRepeatInput")
        super().__init__(tree,location=location,**kwargs)

    def pair_with_output(self,output):
        if isinstance(output,Node):
            output = output.node
        self.node.pair_with_output(output)

class RepeatOutput(GreenNode):
    def __init__(self,tree,location=(0,0),**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeRepeatOutput")
        super().__init__(tree,location=location,**kwargs)
    def add_socket(self,socket_type,socket_name):
        self.node.repeat_items.new(socket_type,socket_name)

class RepeatZone(GreenNode):
    def __init__(self, tree, location=(0, 0), node_width=5, iterations=10, geometry=None, **kwargs):
        self.repeat_output = tree.nodes.new("GeometryNodeRepeatOutput")
        self.repeat_input = tree.nodes.new("GeometryNodeRepeatInput")
        self.repeat_input.location = (location[0] * 200, location[1] * 200)
        self.repeat_output.location = (location[0] * 200 + node_width * 200, location[1] * 200)
        self.repeat_input.pair_with_output(self.repeat_output)
        self.node = self.repeat_input
        self.geometry_in = self.repeat_input.inputs["Geometry"]
        self.geometry_out = self.repeat_output.outputs["Geometry"]
        self.iteration = self.repeat_input.outputs["Iteration"]
        tree.links.new(self.repeat_input.outputs["Geometry"], self.repeat_output.inputs["Geometry"])
        super().__init__(tree, location=location, **kwargs)

        if isinstance(iterations, int):
            self.repeat_input.inputs["Iterations"].default_value = iterations
        else:
            tree.links.new(iterations, self.repeat_input.inputs["Iterations"])

        if geometry is not None:
            tree.links.new(geometry, self.repeat_input.inputs["Geometry"])

    def add_socket(self, socket_type="GEOMETRY", name="socket"):
        """
        :param socket_type: "FLOAT", "INT", "BOOLEAN", "VECTOR", "ROTATION", "STRING", "RGBA", "OBJECT", "IMAGE", "GEOMETRY", "COLLECTION", "TEXTURE", "MATERIAL"
        :param name:
        :return:
        """
        self.repeat_output.repeat_items.new(socket_type, name)

    def join_in_geometries(self, out_socket_name=None):
        join = JoinGeometry(self.tree, geometry=self.repeat_input.outputs[0:-1])
        if out_socket_name:
            self.tree.links.new(join.geometry_out, self.repeat_output.inputs[out_socket_name])

    def create_geometry_line(self, nodes,out=None):
        if len(nodes)==0:
            # just connect repeat zone
            self.tree.links.new(self.repeat_input.outputs["Geometry"],self.repeat_output.inputs["Geometry"])
        else:
            last = nodes.pop()
            if out is None:
                self.tree.links.new(last.geometry_out, self.repeat_output.inputs["Geometry"])
            else:
                self.tree.links.new(last.geometry_out, out)
            while len(nodes) > 0:
                current = nodes.pop()
                self.tree.links.new(current.geometry_out, last.geometry_in)
                last = current

            self.tree.links.new(self.repeat_input.outputs["Geometry"], last.geometry_in)


class SimulationInput(GreenNode):
    def __init__(self,tree,location=(0,0),**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeSimulationInput")
        super().__init__(tree,location=location,**kwargs)
    def pair_with_output(self,output):
        if isinstance(output,Node):
            output = output.node
        self.node.pair_with_output(output)

class SimulationOutput(GreenNode):
    def __init__(self,tree,location=(0,0),**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeSimulationOutput")
        super().__init__(tree,location=location,**kwargs)
    def add_socket(self,socket_type,socket_name):
        if socket_type=="VALUE":
            socket_type="FLOAT"
        self.node.state_items.new(socket_type,socket_name)


class Simulation(GreenNode):
    def __init__(self, tree, location=(0, 0), node_width=5, geometry=None, **kwargs):
        self.simulation_output = tree.nodes.new("GeometryNodeSimulationOutput")
        self.simulation_input = tree.nodes.new("GeometryNodeSimulationInput")
        self.simulation_input.location = (location[0] * 200, location[1] * 200)
        self.simulation_output.location = (location[0] * 200 + node_width * 200, location[1] * 100)
        self.simulation_input.pair_with_output(self.simulation_output)
        self.node = self.simulation_input
        self.geometry_in = self.simulation_input.inputs["Geometry"]
        self.geometry_out = self.simulation_output.outputs["Geometry"]
        tree.links.new(self.simulation_input.outputs["Geometry"], self.simulation_output.inputs["Geometry"])
        super().__init__(tree, location=location, **kwargs)

        if geometry is not None:
            tree.links.new(geometry, self.simulation_input.inputs["Geometry"])

    def add_socket(self, socket_type="GEOMETRY", name="socket", value=0):
        """
        :param socket_type: "FLOAT", "INT", "BOOLEAN", "VECTOR", "ROTATION", "STRING", "RGBA", "OBJECT", "IMAGE", "GEOMETRY", "COLLECTION", "TEXTURE", "MATERIAL"
        :param name:
        :return:
        """
        self.simulation_output.state_items.new(socket_type, name)
        self.simulation_input.outputs[name].default_value = value

    def join_in_geometries(self, out_socket_name=None):
        join = JoinGeometry(self.tree, geometry=self.simulation_input.outputs[0:-1])
        if out_socket_name:
            self.tree.links.new(join.geometry_out, self.simulation_output.inputs[out_socket_name])

    def create_geometry_line(self, nodes):
        last = nodes.pop()
        self.tree.links.new(last.geometry_out, self.simulation_output.inputs["Geometry"])
        while len(nodes) > 0:
            current = nodes.pop()
            self.tree.links.new(current.geometry_out, last.geometry_in)
            last = current
        self.tree.links.new(self.simulation_input.outputs["Geometry"], last.geometry_in)

class ForEachZone(GreenNode):
    def __init__(self, tree, location=(0, 0), domain="POINT",node_width=5, geometry=None,selection=None, **kwargs):
        self.foreach_output = tree.nodes.new("GeometryNodeForeachGeometryElementOutput")
        self.foreach_input = tree.nodes.new("GeometryNodeForeachGeometryElementInput")
        self.foreach_input.location = (location[0] * 200, location[1] * 200)
        self.foreach_output.location = (location[0] * 200 + node_width * 200, location[1] * 100)
        self.foreach_input.pair_with_output(self.foreach_output)
        self.foreach_output.domain = domain
        self.node = self.foreach_input
        self.index = self.foreach_input.outputs[0]
        self.geometry_in = self.foreach_input.inputs["Geometry"]
        self.geometry_out = self.foreach_output.outputs[2]
        self.element=self.foreach_input.outputs["Element"]
        # tree.links.new(self.foreach_input.outputs["Element"], self.foreach_output.inputs["Geometry"])
        super().__init__(tree, location=location, **kwargs)
        self.tree = tree
        if geometry is not None:
            tree.links.new(geometry, self.foreach_input.inputs["Geometry"])

        if selection is not None:
            tree.links.new(selection,self.foreach_input.inputs["Selection"])

    def add_socket(self, socket_type="GEOMETRY", name="socket", value=None,for_input=False):
        """
        :param socket_type: "FLOAT", "INT", "BOOLEAN", "VECTOR", "ROTATION", "STRING", "RGBA", "OBJECT", "IMAGE", "GEOMETRY", "COLLECTION", "TEXTURE", "MATERIAL"
        :param name:
        :return:
        """
        self.foreach_output.input_items.new(socket_type, name)

        if value is not None:
            if isinstance(value,(int,float)):
                self.foreach_input.outputs[name].default_value = value
            else:
                if for_input:
                    self.tree.links.new(value,self.foreach_input.inputs[name])
                else:
                    self.tree.links.new(self.foreach_input.outputs[name],value)

    def create_geometry_line(self, nodes,ins=None):
        if len(nodes)==0:
            # just pipe through the geometry line
            self.tree.links.new(self.foreach_input.outputs["Geometry"], self.foreach_output.inputs["Geometry"])
        else:
            last = nodes.pop()
            self.tree.links.new(last.geometry_out, self.foreach_output.inputs["Geometry"])
            while len(nodes) > 0:
                current = nodes.pop()
                self.tree.links.new(current.geometry_out, last.geometry_in)
                last = current
            if ins is None:
                self.tree.links.new(self.foreach_input.outputs["Element"], last.geometry_in)
            else:
                self.tree.links.new(ins, last.geometry_in)


class ForEachInput(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,selection=None, **kwargs):

        # I collected the name convention from the blender github repository directly
        # it can be found under blender/source/blender/nodes/geometry/node_geo_foreach_geometry_element.cc
        # praise open source !!!
        if blender_version()<(4,3):
            raise "No for each element node in versions below (4,3)"
        self.node = tree.nodes.new("GeometryNodeForeachGeometryElementInput")
        super().__init__(tree,location,**kwargs)


        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs[0]
        self.pair_node=None

        if geometry:
            tree.links.new(geometry,self.node.inputs["Geometry"])
        if selection:
            tree.links.new(selection,self.node.inputs["Selection"])

    def pair_with_output(self,output):
        self.pair_node = output
        if isinstance(output,Node):
            output = output.node
        self.node.pair_with_output(output)


class ForEachOutput(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None, domain = "POINT",**kwargs):

        self.node = tree.nodes.new("GeometryNodeForeachGeometryElementOutput")
        self.node.domain = domain
        super().__init__(tree,location,**kwargs)

        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs["Geometry"]

        if geometry:
            tree.links.new(geometry,self.node.inputs["Geometry"])

    def add_socket(self, socket_type="GEOMETRY", name="socket", value=None):
        """
        :param socket_type: "FLOAT", "INT", "BOOLEAN", "VECTOR", "ROTATION", "STRING", "RGBA", "OBJECT", "IMAGE", "GEOMETRY", "COLLECTION", "TEXTURE", "MATERIAL"
        :param name:
        :return:
        """
        self.node.input_items.new(socket_type, name)
        if value is not None:
            self.node.outputs[name].default_value = value

# custom composite nodes

class WireFrame(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 radius=0.02,
                 resolution=4,
                 geometry=None,
                 fill_caps=True,
                 **kwargs
                 ):

        self.fill_caps = fill_caps
        self.node = self.create_node(tree.nodes)
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Mesh"]

        if geometry:
            self.tree.links.new(geometry, self.geometry_in)
        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs["Radius"])
        if isinstance(resolution, int):
            self.node.inputs["Resolution"].default_value = resolution
        else:
            self.tree.links.new(resolution, self.node.inputs["Resolution"])

    def create_node(self, nodes, name="WireframeNode"):
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Mesh", io="INPUT", type="NodeSocketGeometry")
        make_new_socket(tree, name="Radius", io="INPUT", type="NodeSocketFloat")
        make_new_socket(tree, name="Resolution", io="INPUT", type="NodeSocketInt")

        make_new_socket(tree, name="Mesh", io="OUTPUT", type="NodeSocketGeometry")

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        mesh2curve = MeshToCurve(tree, location=(1, 0))
        curve_circle = CurveCircle(tree, location=(1, 1), resolution=group_inputs.outputs["Resolution"],
                                   radius=group_inputs.outputs["Radius"])
        curve2mesh = CurveToMesh(tree, location=(2, 0), profile_curve=curve_circle.geometry_out,fill_caps=self.fill_caps)
        create_geometry_line(tree, [mesh2curve, curve2mesh],
                             ins=group_inputs.outputs["Mesh"], out=group_outputs.inputs["Mesh"])
        return group

class CurveWireFrame(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 radius=0.02,
                 resolution=4,
                 geometry=None,
                 fill_caps=True,
                 **kwargs
                 ):

        self.fill_caps = fill_caps
        self.node = self.create_node(tree.nodes)
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Curve"]

        if geometry:
            self.tree.links.new(geometry, self.geometry_in)
        if isinstance(radius, (int, float)):
            self.node.inputs["Radius"].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs["Radius"])
        if isinstance(resolution, int):
            self.node.inputs["Resolution"].default_value = resolution
        else:
            self.tree.links.new(resolution, self.node.inputs["Resolution"])

    def create_node(self, nodes, name="WireframeNode"):
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Curve", io="INPUT", type="NodeSocketGeometry")
        make_new_socket(tree, name="Radius", io="INPUT", type="NodeSocketFloat")
        make_new_socket(tree, name="Resolution", io="INPUT", type="NodeSocketInt")

        make_new_socket(tree, name="Mesh", io="OUTPUT", type="NodeSocketGeometry")

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        curve_circle = CurveCircle(tree, location=(1, 1), resolution=group_inputs.outputs["Resolution"],
                                   radius=group_inputs.outputs["Radius"])
        curve2mesh = CurveToMesh(tree, location=(2, 0), profile_curve=curve_circle.geometry_out,fill_caps=self.fill_caps)
        create_geometry_line(tree, [curve2mesh],
                             ins=group_inputs.outputs["Curve"], out=group_outputs.inputs["Mesh"])
        return group


class WireFrameRectangle(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 node_width=0.02, node_height=0.02,
                 geometry=None,
                 **kwargs
                 ):

        self.node = self.create_node(tree.nodes)
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Mesh"]
        self.geometry_in = self.node.inputs["Mesh"]

        if geometry:
            self.tree.links.new(geometry, self.geometry_in)
        if isinstance(node_width, (int, float)):
            self.node.inputs["Width"].default_value =node_width
        else:
            self.tree.links.new(node_width, self.node.inputs["Width"])
        if isinstance(node_height, (int, float)):
            self.node.inputs["Height"].default_value =node_height
        else:
            self.tree.links.new(node_height, self.node.inputs["Height"])

    def create_node(self, nodes, name="WireframeNode"):
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Mesh", io="INPUT", type="NodeSocketGeometry")
        make_new_socket(tree, name="Width", io="INPUT", type="NodeSocketFloat")
        make_new_socket(tree, name="Height", io="INPUT", type="NodeSocketFloat")

        make_new_socket(tree, name="Mesh", io="OUTPUT", type="NodeSocketGeometry")

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        mesh2curve = MeshToCurve(tree, location=(1, 0))
        rectangle = CurveQuadrilateral(tree, location=(1, 1), width=group_inputs.outputs["Width"],
                                       height=group_inputs.outputs["Height"])
        curve2mesh = CurveToMesh(tree, location=(2, 0), profile_curve=rectangle.geometry_out)
        create_geometry_line(tree, [mesh2curve, curve2mesh],
                             ins=group_inputs.outputs["Mesh"], out=group_outputs.inputs["Mesh"])
        return group

class InsideConvexHull(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 target_geometry=None,
                 source_position=Vector(),
                 ray_direction=Vector([0, 0, 1]),
                 **kwargs
                 ):
        """

        :param tree:
        :param location:
        :param target_geometry:
        :param source_position:
        :param kwargs:
        """
        self.ray_direction = ray_direction
        self.node = self.create_node(tree.nodes)
        super().__init__(tree, location=location, **kwargs)

        self.geometry_in = self.node.inputs["Target Geometry"]
        self.std_out = self.node.outputs["Is Inside"]

        if target_geometry:
            self.tree.links.new(target_geometry, self.node.inputs["Target Geometry"])
        if isinstance(source_position, (Vector, list)):
            self.node.inputs["Source Position"].default_value = source_position
        else:
            self.tree.links.new(source_position, self.node.inputs["Source Position"])
        if isinstance(self.ray_direction, (Vector, list)):
            self.node.inputs["Ray Direction"].default_value = self.ray_direction
        else:
            self.tree.links.new(ray_direction, self.node.inputs["Ray Direction"])

    def create_node(self, nodes, name="InsideConvexHullTest"):
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Target Geometry", io="INPUT", type="NodeSocketGeometry")
        make_new_socket(tree, name="Source Position", io="INPUT", type="NodeSocketVector")
        make_new_socket(tree, name="Ray Direction", io="INPUT", type="NodeSocketVector")

        make_new_socket(tree, name="Is Inside", io="OUTPUT", type="NodeSocketBool")
        make_new_socket(tree, name="Is Outside", io="OUTPUT", type="NodeSocketBool")

        group_inputs.location = (0, 0)
        group_outputs.location = (400, 0)

        ray_cast_up = RayCast(tree, location=(2, 2),
                              target_geometry=group_inputs.outputs["Target Geometry"],
                              source_position=group_inputs.outputs["Source Position"],
                              ray_direction=group_inputs.outputs["Ray Direction"], label="RayUp")

        scale = VectorMath(tree, location=(1, -2), label="Negative", operation="SCALE",
                           inputs0=group_inputs.outputs["Ray Direction"], float_input=-1, hide=True)
        ray_direction = scale.std_out
        ray_cast_down = RayCast(tree, location=(2, -2),
                                target_geometry=group_inputs.outputs["Target Geometry"],
                                source_position=group_inputs.outputs["Source Position"],
                                ray_direction=scale.std_out, label="RayDown")

        andMath = BooleanMath(tree, location=(3, 0.5), label="And", operation="AND",
                              inputs0=ray_cast_up.outputs["Is Hit"],
                              inputs1=ray_cast_down.outputs["Is Hit"],
                              hide=True
                              )

        notAndMath = BooleanMath(tree, location=(3, -0.5), label="NotAnd", operation="NAND",
                                 inputs0=ray_cast_up.outputs["Is Hit"],
                                 inputs1=ray_cast_down.outputs["Is Hit"],
                                 hide=True
                                 )

        tree_links.new(andMath.std_out, group_outputs.inputs["Is Inside"])
        tree_links.new(notAndMath.std_out, group_outputs.inputs["Is Outside"])
        return group

class InsideConvexHull3D(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 target_geometry=None,
                 source_position=Vector(),
                 **kwargs
                 ):
        """

        :param tree:
        :param location:
        :param target_geometry:
        :param source_position:
        :param kwargs:
        """
        self.node = self.create_node(tree.nodes)
        super().__init__(tree, location=location, **kwargs)

        self.geometry_in = self.node.inputs["Target Geometry"]
        self.std_out = self.node.outputs["Is Inside"]

        if target_geometry:
            self.tree.links.new(target_geometry, self.node.inputs["Target Geometry"])
        if isinstance(source_position, (Vector, list)):
            self.node.inputs["Source Position"].default_value = source_position
        else:
            self.tree.links.new(source_position, self.node.inputs["Source Position"])

    def create_node(self, nodes, name="InsideConvexHullTest"):
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Target Geometry", io="INPUT", type="NodeSocketGeometry")
        make_new_socket(tree, name="Source Position", io="INPUT", type="NodeSocketVector")

        make_new_socket(tree, name="Is Inside", io="OUTPUT", type="NodeSocketBool")
        make_new_socket(tree, name="Is Outside", io="OUTPUT", type="NodeSocketBool")

        group_inputs.location = (0, 0)
        group_outputs.location = (400, 0)

        boundary_box = BoundingBox(tree, location=(2, -2),
                                   geometry=group_inputs.outputs["Target Geometry"], label="BBox")

        comparison = make_function(tree, functions={
            "in": ["src_z,maxx_z,>,not,src_z,minn_z,>,*"],
            "out": ["src_z,maxx_z,>,src_z,minn_z,>,not,+"]
        }, inputs=["src", "maxx", "minn"], outputs=["in", "out"], scalars=["in", "out"],
                                   vectors=["src", "minn", "maxx"], name="Comparsion")
        tree.links.new(group_inputs.outputs["Source Position"], comparison.inputs["src"])
        tree.links.new(boundary_box.outputs["Min"], comparison.inputs["minn"])
        tree.links.new(boundary_box.outputs["Max"], comparison.inputs["maxx"])

        tree_links.new(comparison.outputs["in"], group_outputs.inputs["Is Inside"])
        tree_links.new(comparison.outputs["out"], group_outputs.inputs["Is Outside"])
        return group

class E8Node(GreenNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        """
        Create the geometry required for the 4_21 Gosset polytope
        :param tree:
        :param location:
        :param target_geometry:
        :param source_position:
        :param kwargs:
        """

        self.node = self.create_node(tree.nodes)
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]

    def create_node(self, nodes, name="E8Geometry"):
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_outputs = tree_nodes.new("NodeGroupOutput")
        make_new_socket(tree, name="Geometry", io="OUTPUT", type="NodeSocketGeometry")

        group_outputs.location = (400, 0)

        join = JoinGeometry(tree)
        tree_links.new(join.outputs["Geometry"], group_outputs.inputs["Geometry"])

        # the 240 eight-dimensional coordinates are hard-coded into the node
        # create a point and a set of attributes for each root of the E8 lattice
        print("Hard-coded entry of roots ...", end="")
        for root in E8Lattice().roots:
            point = Points(tree)
            attr1 = StoredNamedAttribute(tree, data_type="FLOAT_VECTOR", name="comp123", value=list(root[0:3]))
            attr2 = StoredNamedAttribute(tree, data_type="FLOAT_VECTOR", name="comp456", value=list(root[3:6]))
            attr3 = StoredNamedAttribute(tree, data_type="FLOAT_VECTOR", name="comp78", value=list(root[6:8]) + [0])

            create_geometry_line(tree, [point, attr1, attr2, attr3, join])
        print("done")
        print("Layout of the node ...", end="")
        layout(tree)
        print("done")
        return group

# custom Node groups
class NodeGroup(Node):
    def __init__(self,tree,**kwargs):
        inputs = get_from_kwargs(kwargs,"inputs",{"Position":"VECTOR","Index":"INT"})
        outputs = get_from_kwargs(kwargs,"outputs",{"Geometry":"GEOMETRY"})
        self.create_node_group(tree,inputs,outputs,**kwargs)

        # filled by subclass
        self.fill_group_with_node(self.group_tree,**kwargs)

        auto_layout = get_from_kwargs(kwargs,"auto_layout",True)
        if auto_layout:
            layout(self.group_tree)

    def create_node_group(self,tree,inputs,outputs,**kwargs):

        # new group and inputs and outputs
        nodes = tree.nodes
        name=get_from_kwargs(kwargs,"name","DefaultNodeGroup")
        group = nodes.new(type='GeometryNodeGroup')
        node_tree = bpy.data.node_groups.new(name, type='GeometryNodeTree')
        group.node_tree = node_tree
        self.group_tree=node_tree
        nodes = node_tree.nodes
        group.name = name
        self.group_inputs = nodes.new('NodeGroupInput')
        self.group_inputs.location = (-200, 0)
        self.group_outputs = nodes.new('NodeGroupOutput')
        self.group_outputs.location = (200, 0)

        for name,type in inputs.items():
                make_new_socket(node_tree,name=name,io="INPUT",type=SOCKET_TYPES[type])
        for name,type in outputs.items():
                make_new_socket(node_tree,name=name,io="OUTPUT",type=SOCKET_TYPES[type])

        self.node = group
        super().__init__(tree,**kwargs)

    def fill_group_with_node(self,group_tree,**kwargs):
        """ filled by sub classes """
        pass

class TransformPositionNode(NodeGroup):
    def __init__(self,tree,**kwargs):
        self.name = get_from_kwargs(kwargs,"name",
                                    "TransformPositionNode")

        super().__init__(tree,inputs={"Position":"VECTOR","Location":"VECTOR","Rotation":"ROTATION","Scale":"VECTOR","Undo Transformation":"BOOLEAN"},
                         outputs={"Position":"VECTOR"},name=self.name,offset_y=-400,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

    def fill_group_with_node(self,tree,**kwargs):
        do_transform = make_function(tree, name="Do Transform",
                                     functions={
                                         "result": "position,scl,mul,rotation,rot_vec,location,add"
                                     }, inputs=["position", "scl", "rotation", "location"], outputs=["result"],
                                     vectors=["position", "scl",  "location", "result"],rotations=["rotation"],hide=False)

        tree.links.new(self.group_inputs.outputs["Position"], do_transform.inputs["position"])
        tree.links.new(self.group_inputs.outputs["Scale"], do_transform.inputs["scl"])
        tree.links.new(self.group_inputs.outputs["Rotation"], do_transform.inputs["rotation"])
        tree.links.new(self.group_inputs.outputs["Location"], do_transform.inputs["location"])

        undo_transform = make_function(tree, name="Undo Transform",
                                     functions={
                                         "result": "position,location,sub,rotation,inv_rot,rot_vec,scl,div"
                                     }, inputs=["position", "scl", "rotation", "location"], outputs=["result"],
                                     vectors=["position", "scl", "location", "result"],rotations=["rotation"],hide=False)

        tree.links.new(self.group_inputs.outputs["Position"], undo_transform.inputs["position"])
        tree.links.new(self.group_inputs.outputs["Scale"], undo_transform.inputs["scl"])
        tree.links.new(self.group_inputs.outputs["Rotation"], undo_transform.inputs["rotation"])
        tree.links.new(self.group_inputs.outputs["Location"], undo_transform.inputs["location"])

        switch = Switch(tree,input_type="VECTOR",switch=self.group_inputs.outputs["Undo Transformation"],
                        false=do_transform.outputs["result"],true=undo_transform.outputs["result"])
        tree.links.new(switch.outputs["Output"],self.group_outputs.inputs["Position"])

class BeveledCubeNode(NodeGroup):
    def __init__(self,tree,size=1,bevel=0.01,**kwargs):
        self.name = get_from_kwargs(kwargs,"name","BeveledCubeNode")
        super().__init__(tree,inputs={"Size":"FLOAT","Bevel":"FLOAT"},
                         outputs={"Mesh":"GEOMETRY"},auto_layout=True,name=self.name,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(size,(int,float)):
            self.node.inputs["Size"].default_value =size
        else:
            tree.links.new(size,self.node.inputs["Size"])

        if isinstance(bevel,(int,float)):
            self.node.inputs["Bevel"].default_value =bevel
        else:
            tree.links.new(bevel,self.node.inputs["Bevel"])

    def fill_group_with_node(self,tree,**kwargs):
        links = tree.links
        bevel_function = make_function(tree, name="BevelFunction",
                                       functions={
                                           "bevel": "size,bevel,/"
                                       }, inputs=["size", "bevel"], outputs=["bevel"],
                                       scalars=["size", "bevel"], hide=False)

        links.new(self.group_inputs.outputs["Bevel"], bevel_function.inputs["bevel"])
        links.new(self.group_inputs.outputs["Size"], bevel_function.inputs["size"])


        cube = CubeMesh(tree, size=self.group_inputs.outputs["Size"], hide=False)
        cube2 = CubeMesh(tree, size=bevel_function.outputs["bevel"], hide=False)
        subsurf = SubdivisionSurface(tree, level=3, mesh=cube2.geometry_out, hide=False,)
        iop2 = InstanceOnPoints(tree, scale=self.group_inputs.outputs["Size"], instance=subsurf.geometry_out, hide=False)
        realize_instance2 = RealizeInstances(tree, hide=False)
        convex_hull = ConvexHull(tree, hide=False)

        create_geometry_line(tree, [cube, iop2, realize_instance2, convex_hull],out=self.group_outputs.inputs["Mesh"])

class BevelNode(NodeGroup):
    def __init__(self,tree,radius=0.01,subdivisions=3,**kwargs):
        self.name = get_from_kwargs(kwargs,"name","BevelNode")
        super().__init__(tree,inputs={"Points":"GEOMETRY","Radius":"FLOAT","Subdivisions":"INT"},
                         outputs={"Bevelled Geometry":"GEOMETRY"},
                         auto_layout=True,name=self.name,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_out = self.node.outputs["Bevelled Geometry"]

        if isinstance(radius,(int,float)):
            self.node.inputs["Radius"].default_value =radius
        else:
            tree.links.new(radius,self.node.inputs["Radius"])

        if isinstance(subdivisions,int):
            self.node.inputs["Subdivisions"].default_value =subdivisions
        else:
            tree.links.new(subdivisions,self.node.inputs["Subdivisions"])

    def fill_group_with_node(self,tree,**kwargs):
        links = tree.links
        pos = Position(tree)
        normal = InputNormal(tree)
        # this function insets the position of the spheres that the bevelled object is not increased in size
        bevel_function = make_function(tree, name="BevelFunction",
                                       functions={
                                           "position": "pos,normal,1.5,radius,*,scale,sub"
                                       }, inputs=["pos","normal","radius"], outputs=["position"],
                                       scalars=["radius"],vectors=["pos","normal","position"], hide=True)

        links.new(self.group_inputs.outputs["Radius"], bevel_function.inputs["radius"])
        links.new(pos.std_out, bevel_function.inputs["pos"])
        links.new(normal.std_out, bevel_function.inputs["normal"])

        set_pos = SetPosition(tree,position=bevel_function.outputs["position"])
        ico_sphere = IcoSphere(tree,radius=self.group_inputs.outputs["Radius"],
                               subdivisions=self.group_inputs.outputs["Subdivisions"],hide=True)
        iop = InstanceOnPoints(tree,instance=ico_sphere.geometry_out,hide=True)
        realize_instance = RealizeInstances(tree,hide=True)
        convex_hull = ConvexHull(tree,hide=True)
        create_geometry_line(tree, [set_pos, iop, realize_instance, convex_hull],out=self.group_outputs.inputs["Bevelled Geometry"],
                             ins = self.group_inputs.outputs["Points"])

# custom Matrix operations #
class Rotation(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 dimension=5,
                 u=0,
                 v=1,
                 angle=pi,
                 orientation=1,
                 **kwargs
                 ):
        """
        Creates a rotation matrix node for a given vector space dimension and angle of rotation
        :param tree:
        :param location:
        :param dim: dimension of the vector space
        :param u,v: define the plane of rotation
        :param angle: the angle of rotation
        :param orientation: flips minus sign in front of the sine functions
        :param kwargs:
        """
        self.orientation = orientation
        self.node = self.create_node(tree.nodes, dimension, u, v)
        super().__init__(tree, location=location, **kwargs)

        if isinstance(dimension, int):
            self.node.inputs["Dimension"].default_value = dimension
        else:
            self.tree.links.new(dimension, self.node.inputs["Dimension"])

        if isinstance(angle, (int, float)):
            self.node.inputs["Angle"].default_value = angle
        else:
            self.tree.links.new(angle, self.node.inputs["Angle"])

        if isinstance(u, int):
            self.node.inputs["U"].default_value = u
        else:
            self.tree.links.new(u, self.node.inputs["U"])
        if isinstance(v, int):
            self.node.inputs["V"].default_value = v
        else:
            self.tree.links.new(v, self.node.inputs["V"])

    def create_node(self, nodes, dimension, u, v, name="RotationMatrix"):
        d = dimension
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Dimension", io="INPUT", type="NodeSocketInt")
        make_new_socket(tree, name="U", io="INPUT", type="NodeSocketInt")
        make_new_socket(tree, name="V", io="INPUT", type="NodeSocketInt")
        make_new_socket(tree, name="Angle", io="INPUT", type="NodeSocketFloat")

        # create the required number of vectors for each column
        outputs = []
        for i in range(d):
            for j in range(0, int(np.ceil(d / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io="OUTPUT", type="NodeSocketVector")
                outputs.append(name)

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        # create function dictionary
        func_dict = {}
        for row in range(d):
            # treat diagonal entries, equal to 1, if u and v are different then the column index
            comps = ["0"] * int(np.ceil(d / 3) * 3)
            for col in range(d):
                if col == row:
                    comps[row] = "1,u," + str(row) + ",-,abs,0,>,v," + str(row) + ",-,abs,0,>,*,*,u," + str(
                        row) + ",=,v," + str(row) + ",=,+,theta,cos,*,+"
                else:
                    if self.orientation == 1:
                        comps[col] = "u," + str(row) + ",=,v," + str(col) + ",=,*,theta,sin,*,-1,*,u," + str(
                            col) + ",=,v," + str(row) + ",=,*,theta,sin,*,+"
                    else:
                        comps[col] = "u," + str(row) + ",=,v," + str(col) + ",=,*,theta,sin,*,u," + str(
                            col) + ",=,v," + str(row) + ",=,*,theta,sin,*,-1,*,+"
            for n, p in enumerate(range(0, d, 3)):
                part = comps[p:p + 3]
                func_dict["row_" + str(row) + "_" + str(n)] = part

        rot_mat = make_function(tree_nodes, functions=func_dict, name="RotationMatrix", inputs=["u", "v", "theta"],
                                outputs=outputs, scalars=["u", "v", "theta"], vectors=outputs)
        rot_mat.location = (200, 0)
        tree_links.new(group_inputs.outputs["U"], rot_mat.inputs["u"])
        tree_links.new(group_inputs.outputs["V"], rot_mat.inputs["v"])
        tree_links.new(group_inputs.outputs["Angle"], rot_mat.inputs["theta"])

        for o in outputs:
            tree_links.new(rot_mat.outputs[o], group_outputs.inputs[o])

        return group

class Matrix(GreenNode):
    """
    create a matrix node with number entries,
    """

    def __init__(self, tree, rows=3, cols=3, entries=[], location=(0, 0), **kwargs):
        self.node = self.create_node(tree.nodes, rows, cols, entries=entries, **kwargs)
        super().__init__(tree, location=location, **kwargs)

    def create_node(self, nodes, rows, cols, entries, name="Matrix"):
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_outputs = tree_nodes.new("NodeGroupOutput")

        # create the required number of vectors for each column
        outputs = []
        for i in range(rows):
            for j in range(0, int(np.ceil(cols / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io="OUTPUT", type="NodeSocketVector")
                outputs.append(name)

        group_outputs.location = (600, 0)

        # create entries and combine them to vecors

        for r, row in enumerate(entries):
            for i in range(0, len(row), 3):
                if i + 3 < len(row):
                    comps = row[i:i + 3]
                else:
                    comps = row[i:len(row)]
                while len(comps) < 3:
                    comps = comps + [0]
                vec = InputVector(tree, value=Vector(comps), location=[-2 + r, i])
                tree_links.new(vec.std_out, group_outputs.inputs["row_" + str(r) + "_" + str(i // 3)])
        return group

class Transpose(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 dimension=5,
                 **kwargs
                 ):
        """
        :param tree:
        :param location:
        :param dim: dimension of the vector space
        :param directions: define the plane of rotation
        :param angle: the angle of rotation
        :param kwargs:
        """

        self.node = self.create_node(tree.nodes, dimension)
        super().__init__(tree, location=location, **kwargs)

        if isinstance(dimension, int):
            self.node.inputs["Dimension"].default_value = dimension
        else:
            self.tree.links.new(dimension, self.node.inputs["Dimension"])

    def create_node(self, nodes, dimension, name="TransposeMatrix"):
        d = dimension
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Dimension", io="INPUT", type="NodeSocketInt")

        # create the required number of vectors for each column
        outputs = []
        for i in range(d):
            for j in range(0, int(np.ceil(d / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io="INPUT", type="NodeSocketVector")
                make_new_socket(tree, name=name, io="OUTPUT", type="NodeSocketVector")
                outputs.append(name)

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)
        dict = {0: "x", 1: "y", 2: "z"}
        # create function dictionary
        func_dict = {}
        for row in range(d):
            # treat diagonal entries, equal to 1, if u and v are different then the column index
            comps = ["0"] * int(np.ceil(d / 3) * 3)
            for col in range(d):
                part = row // 3
                comp = dict[row % 3]
                comps[col] = "row_" + str(col) + "_" + str(part) + "_" + comp
            for n, p in enumerate(range(0, d, 3)):
                part = comps[p:p + 3]
                func_dict["row_" + str(row) + "_" + str(n)] = part

        trans_mat = make_function(tree_nodes, functions=func_dict, name="TransposeMatrix", inputs=outputs,
                                  outputs=outputs, vectors=outputs
                                  )
        trans_mat.location = (200, 0)
        for o in outputs:
            tree_links.new(group_inputs.outputs[o], trans_mat.inputs[o])
            tree_links.new(trans_mat.outputs[o], group_outputs.inputs[o])

        return group

class LinearMap(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 dimension=5,
                 **kwargs
                 ):
        """
        :param tree:
        :param location:
        :param dim: dimension of the vector space
        :param directions: define the plane of rotation
        :param kwargs:
        """

        self.node = self.create_node(tree.nodes, dimension)
        super().__init__(tree, location=location, **kwargs)

        if isinstance(dimension, int):
            self.node.inputs["Dimension"].default_value = dimension
        else:
            self.tree.links.new(dimension, self.node.inputs["Dimension"])

    def create_node(self, nodes, dimension, name="LinearMap"):
        d = dimension
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        make_new_socket(tree, name="Dimension", io="INPUT", type="NodeSocketInt")

        # create the required number of vectors for each column
        mat_inputs = []
        for i in range(d):
            for j in range(0, int(np.ceil(d / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io="INPUT", type="NodeSocketVector")
                mat_inputs.append(name)
        vec_inputs = []
        for i in range(int(np.ceil(d / 3))):
            name = "v_" + str(i)
            make_new_socket(tree, name=name, io="INPUT", type="NodeSocketVector")
            vec_inputs.append(name)

        for i in range(int(np.ceil(d / 3))):
            make_new_socket(tree, name="v_" + str(i), io="OUTPUT", type="NodeSocketVector")

        components = ["0"] * int(np.ceil(d / 3) * 3)
        comp_dict = {0: "_x", 1: "_y", 2: "_z"}
        for c in range(dimension):
            prod = ""
            first = True
            for i, v in enumerate(vec_inputs):
                if first:
                    prod = "row_" + str(c) + "_" + str(i) + ",v_" + str(i) + ",dot"
                    first = False
                else:
                    prod += ",row_" + str(c) + "_" + str(i) + ",v_" + str(i) + ",dot"
                    prod += ",+"

            components[c] = prod

        func_dict = {}
        c = 0
        for label in vec_inputs:
            func_dict[label] = components[c:c + 3]
            c += 3

        mapping = make_function(tree_nodes, functions=func_dict, name="LinearMap", inputs=mat_inputs + vec_inputs,
                                outputs=vec_inputs, vectors=mat_inputs + vec_inputs)
        mapping.location = (200, 0)

        for o in mat_inputs:
            tree.links.new(group_inputs.outputs[o], mapping.inputs[o])
        for v in vec_inputs:
            tree.links.new(group_inputs.outputs[v], mapping.inputs[v])
            tree.links.new(mapping.outputs[v], group_outputs.inputs[v])
        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        return group

class ProjectionMap(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 in_dimension=8,
                 out_dimension=2,
                 **kwargs
                 ):
        """
        Applies a linear transformation to a vector of dimension in_dimension
        It has to be combined with a matrix that provides the correct dimensions
        :param tree:
        :param location:
        :param in_dimension: dimension of the source vector space
        :param out_dimension: dimension of the target vector space
        :param kwargs:
        """

        self.node = self.create_node(tree.nodes, in_dimension, out_dimension)
        super().__init__(tree, location=location, **kwargs)

    def create_node(self, nodes, in_dimension, out_dimension, name="ProjectionMap"):
        idim = in_dimension
        odim = out_dimension
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new("NodeGroupInput")
        group_outputs = tree_nodes.new("NodeGroupOutput")

        # create the required number of vectors for each column
        mat_inputs = []
        for i in range(odim):
            for j in range(0, int(np.ceil(idim / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io="INPUT", type="NodeSocketVector")
                mat_inputs.append(name)
        vec_inputs = []
        for i in range(int(np.ceil(idim / 3))):
            name = "vi_" + str(i)
            make_new_socket(tree, name=name, io="INPUT", type="NodeSocketVector")
            vec_inputs.append(name)

        vec_outputs = []
        for i in range(int(np.ceil(odim / 3))):
            name = "vo_" + str(i)
            make_new_socket(tree, name=name, io="OUTPUT", type="NodeSocketVector")
            vec_outputs.append(name)

        components = ["0"] * int(np.ceil(idim / 3) * 3)
        for c in range(odim):
            prod = ""
            first = True
            for i, v in enumerate(vec_inputs):
                if first:
                    prod = "row_" + str(c) + "_" + str(i) + ",vi_" + str(i) + ",dot"
                    first = False
                else:
                    prod += ",row_" + str(c) + "_" + str(i) + ",vi_" + str(i) + ",dot"
                    prod += ",+"

            components[c] = prod

        func_dict = {}
        c = 0
        for label in vec_outputs:
            func_dict[label] = components[c:c + 3]
            c += 3

        mapping = make_function(tree_nodes, functions=func_dict, name="ProjectionMap", inputs=mat_inputs + vec_inputs,
                                outputs=vec_outputs, vectors=mat_inputs + vec_inputs + vec_outputs)
        mapping.location = (200, 0)

        for o in mat_inputs:
            tree.links.new(group_inputs.outputs[o], mapping.inputs[o])
        for v in vec_inputs:
            tree.links.new(group_inputs.outputs[v], mapping.inputs[v])
        for v in vec_outputs:
            tree.links.new(mapping.outputs[v], group_outputs.inputs[v])
        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        return group

#auxiliary Node groups

class TransformPositionNode(NodeGroup):
    def __init__(self,tree,**kwargs):
        self.name = get_from_kwargs(kwargs,"name",
                                    "TransformPositionNode")

        super().__init__(tree,inputs={"Position":"VECTOR","Location":"VECTOR","Rotation":"ROTATION","Scale":"VECTOR","Undo Transformation":"BOOLEAN"},
                         outputs={"Position":"VECTOR"},name=self.name,offset_y=-400,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

    def fill_group_with_node(self,tree,**kwargs):
        do_transform = make_function(tree, name="Do Transform",
                                     functions={
                                         "result": "position,scl,mul,rotation,rot_vec,location,add"
                                     }, inputs=["position", "scl", "rotation", "location"], outputs=["result"],
                                     vectors=["position", "scl",  "location", "result"],rotations=["rotation"],hide=False)

        tree.links.new(self.group_inputs.outputs["Position"], do_transform.inputs["position"])
        tree.links.new(self.group_inputs.outputs["Scale"], do_transform.inputs["scl"])
        tree.links.new(self.group_inputs.outputs["Rotation"], do_transform.inputs["rotation"])
        tree.links.new(self.group_inputs.outputs["Location"], do_transform.inputs["location"])

        undo_transform = make_function(tree, name="Undo Transform",
                                     functions={
                                         "result": "position,location,sub,rotation,inv_rot,rot_vec,scl,div"
                                     }, inputs=["position", "scl", "rotation", "location"], outputs=["result"],
                                     vectors=["position", "scl", "location", "result"],rotations=["rotation"],hide=False)

        tree.links.new(self.group_inputs.outputs["Position"], undo_transform.inputs["position"])
        tree.links.new(self.group_inputs.outputs["Scale"], undo_transform.inputs["scl"])
        tree.links.new(self.group_inputs.outputs["Rotation"], undo_transform.inputs["rotation"])
        tree.links.new(self.group_inputs.outputs["Location"], undo_transform.inputs["location"])

        switch = Switch(tree,input_type="VECTOR",switch=self.group_inputs.outputs["Undo Transformation"],
                        false=do_transform.outputs["result"],true=undo_transform.outputs["result"])
        tree.links.new(switch.outputs["Output"],self.group_outputs.inputs["Position"])

class CoxeterReflectionNode(NodeGroup):
    def __init__(self,tree,position = None,normal = None, progress = None, **kwargs):
        """
        a node that performs a Coxeter reflection on an object. The reflecion matrix is entered as
        three row vectors. A parameter scales continuously from no reflection 0 to full reflection 1
        """
        self.name = get_from_kwargs(kwargs, "name",
                                    "ReflectionNode")

        super().__init__(tree,
                         inputs={"position":"VECTOR", "normal": "VECTOR","progress": "FLOAT"},
                         outputs={"position":"VECTOR"}, name=self.name, offset_y=0, **kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs
        self.std_out = self.node.outputs["position"]

        if position:
            if isinstance(position,(list,Vector)):
                self.inputs["position"].default_value = position
            else:
                tree.links.new(position,self.inputs["position"])
        if normal:
            if isinstance(normal,(list,Vector)):
                self.inputs["normal"].default_value = normal
            else:
                tree.links.new(normal,self.inputs["normal"])
        if progress:
            if isinstance(progress,(int,float)):
                self.inputs["progress"].default_value = progress
            else:
                tree.links.new(progress,self.inputs["progress"])

    def fill_group_with_node(self,tree,**kwargs):
        """
        create tensor product 1 - 2 *n^T n
        """

        tensor = make_function(tree,name="TensorProduct",
                    functions={
                        "col1":["1,n_x,n_x,*,2,progress,*,*,-","0,n_x,n_y,*,2,progress,*,*,-","0,n_x,n_z,*,2,progress,*,*,-"],
                        "col2":["0,n_y,n_x,*,2,progress,*,*,-","1,n_y,n_y,*,2,progress,*,*,-","0,n_y,n_z,*,2,progress,*,*,-"],
                        "col3":["0,n_z,n_x,*,2,progress,*,*,-","0,n_z,n_y,*,2,progress,*,*,-","1,n_z,n_z,*,2,progress,*,*,-"]
                    },inputs=["n","progress"],outputs=["col1","col2","col3"],scalars=["progress"],vectors=["n","col1","col2","col3"],hide=True)
        tree.links.new(self.group_inputs.outputs["normal"],tensor.inputs["n"])
        tree.links.new(self.group_inputs.outputs["progress"],tensor.inputs["progress"])

        separate_col1 = SeparateXYZ(tree,vector=tensor.outputs["col1"])
        separate_col2 = SeparateXYZ(tree,vector=tensor.outputs["col2"])
        separate_col3 = SeparateXYZ(tree,vector=tensor.outputs["col3"])

        combine_matrix = CombineMatrix(tree)

        for out,inp in zip(["X","Y","Z"],["Row 1","Row 2","Row 3"]):
            tree.links.new(separate_col1.outputs[out],combine_matrix.inputs["Column 1 "+inp])
            tree.links.new(separate_col2.outputs[out],combine_matrix.inputs["Column 2 "+inp])
            tree.links.new(separate_col3.outputs[out],combine_matrix.inputs["Column 3 "+inp])

        transform_point = TransformPoint(tree,vector=self.group_inputs.outputs["position"],transform=combine_matrix.std_out)
        tree.links.new(transform_point.std_out, self.group_outputs.inputs["position"])

class BeveledCubeNode(NodeGroup):
    def __init__(self,tree,size=1,bevel=0.01,**kwargs):
        self.name = get_from_kwargs(kwargs,"name","BeveledCubeNode")
        super().__init__(tree,inputs={"Size":"FLOAT","Bevel":"FLOAT"},
                         outputs={"Mesh":"GEOMETRY"},auto_layout=True,name=self.name,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(size,(int,float)):
            self.node.inputs["Size"].default_value =size
        else:
            tree.links.new(size,self.node.inputs["Size"])

        if isinstance(bevel,(int,float)):
            self.node.inputs["Bevel"].default_value =bevel
        else:
            tree.links.new(bevel,self.node.inputs["Bevel"])

    def fill_group_with_node(self,tree,**kwargs):
        links = tree.links
        bevel_function = make_function(tree, name="BevelFunction",
                                       functions={
                                           "bevel": "size,bevel,/"
                                       }, inputs=["size", "bevel"], outputs=["bevel"],
                                       scalars=["size", "bevel"], hide=False)

        links.new(self.group_inputs.outputs["Bevel"], bevel_function.inputs["bevel"])
        links.new(self.group_inputs.outputs["Size"], bevel_function.inputs["size"])


        cube = CubeMesh(tree, size=self.group_inputs.outputs["Size"], hide=False)
        cube2 = CubeMesh(tree, size=bevel_function.outputs["bevel"], hide=False)
        subsurf = SubdivisionSurface(tree, level=3, mesh=cube2.geometry_out, hide=False,)
        iop2 = InstanceOnPoints(tree, scale=self.group_inputs.outputs["Size"], instance=subsurf.geometry_out, hide=False)
        realize_instance2 = RealizeInstances(tree, hide=False)
        convex_hull = ConvexHull(tree, hide=False)

        create_geometry_line(tree, [cube, iop2, realize_instance2, convex_hull],out=self.group_outputs.inputs["Mesh"])

class UnfoldMeshNode(NodeGroup):
    def __init__(self,tree,progression=0,range=20,root_index=0,scale_elements=0.99,**kwargs):
        self.name = get_from_kwargs(kwargs,"name","UnfoldMeshNode")
        super().__init__(tree,inputs={"Mesh":"GEOMETRY","Progression":"FLOAT","Range":"FLOAT","RootIndex":"INT","ScaleElements":"FLOAT"},
                         outputs={"Mesh":"GEOMETRY"},auto_layout=False,name=self.name,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_in = self.node.inputs["Mesh"]
        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(range,(int,float)):
            self.node.inputs["Range"].default_value =range
        else:
            tree.links.new(range,self.node.inputs["Range"])

        if isinstance(progression,(int,float)):
            self.node.inputs["Progression"].default_value =progression
        else:
            tree.links.new(progression,self.node.inputs["Progression"])

        if isinstance(root_index,int):
            self.node.inputs["RootIndex"].default_value =root_index
        else:
            tree.links.new(root_index,self.node.inputs["RootIndex"])

        if isinstance(scale_elements,(int,float)):
            self.node.inputs["ScaleElements"].default_value =scale_elements
        else:
            tree.links.new(scale_elements,self.node.inputs["ScaleElements"])

    def fill_group_with_node(self,tree,**kwargs):
        # remove any existing node
        nodes = tree.nodes
        for n in nodes:
            nodes.remove(n)

        create_from_xml(tree,"unfolding_node",**kwargs)

class SimpleRubiksCubeNode(NodeGroup):
    def __init__(self,tree,seed=0, **kwargs):
        self.name = get_from_kwargs(kwargs, "name", "SimpleRubiksCubeNode")
        super().__init__(tree,inputs={"Seed":"INT"},
                         outputs={"Geometry": "GEOMETRY"}, auto_layout=False, name=self.name, **kwargs)

        self.geometry_out = self.node.outputs["Geometry"]

        if isinstance(seed,(int,float)):
            self.node.inputs["Seed"].default_value =seed
        else:
            tree.links.new(seed,self.node.inputs["Seed"])

    def fill_group_with_node(self, tree, **kwargs):
        # remove any existing node
        nodes = tree.nodes
        for n in nodes:
            nodes.remove(n)

        create_from_xml(tree, "simple_rubikscube_node", **kwargs)

class CycleNode(NodeGroup):
    def __init__(self,tree,max_length=5,**kwargs):
        self.name = get_from_kwargs(kwargs,"name","CycleNode")
        self.max_length = max_length
        inputs={"Geometry":"GEOMETRY","Cycle":"INT","CycleLength":"INT","UpMover":"INT",
                "DownMover":"INT",
                "Displacement":"FLOAT"}
        for i in range(max_length):
            inputs["Mover"+str(i+1)]="FLOAT"

        super().__init__(tree,inputs=inputs,
                         outputs={"Geometry":"GEOMETRY"},auto_layout=True,name=self.name,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs["Geometry"]

    def fill_group_with_node(self,tree,**kwargs):
        attr_idx = NamedAttribute(tree,name="Index",data_type="INT",hide=True)
        outs = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        digits = make_function(tree, name="Digits", functions={
            "one":"cycle,10,%",
            "two":"cycle,10,/,floor,10,%",
            "three":"cycle,100,/,floor,10,%",
            "four":"cycle,1000,/,floor,10,%",
            "five":"cycle,10000,/,floor,10,%",
            "six":"cycle,100000,/,floor,10,%",
            "seven":"cycle,1000000,/,floor,10,%",
            "eight":"cycle,10000000,/,floor,10,%",
            "nine":"cycle,100000000,/,floor,10,%"
        },inputs=["cycle"], outputs=outs,scalars=["cycle"]+outs, hide=False)
        tree.links.new(self.group_inputs.outputs["Cycle"],digits.inputs["cycle"])

        selector = make_function(tree,name="Selector",functions={
            "selection":"idx,one,=,idx,two,=,or,idx,three,=,or,idx,four,=,or,idx,five,=,or,idx,six,=,or,idx,seven,=,or,idx,eight,=,or,idx,nine,=,or"
        },inputs=["idx"]+outs,outputs=["selection"],scalars=["idx","selection"]+outs,hide=False)
        tree.links.new(attr_idx.std_out,selector.inputs["idx"])
        tree.links.new(digits.outputs["one"],selector.inputs["one"])
        tree.links.new(digits.outputs["two"],selector.inputs["two"])
        tree.links.new(digits.outputs["three"],selector.inputs["three"])
        tree.links.new(digits.outputs["four"],selector.inputs["four"])
        tree.links.new(digits.outputs["five"],selector.inputs["five"])
        tree.links.new(digits.outputs["six"],selector.inputs["six"])
        tree.links.new(digits.outputs["seven"],selector.inputs["seven"])
        tree.links.new(digits.outputs["eight"],selector.inputs["eight"])
        tree.links.new(digits.outputs["nine"],selector.inputs["nine"])
        select_numbers = SeparateGeometry(tree,domain="INSTANCE",
                                          selection=selector.outputs["selection"],hide=True)

        # the weight is chosen such that the first number of the cycle has the highest weigth
        # the second the lowest
        # the third the second hightest and so on
        sort_numbers = make_function(tree,name="SortNumbers",functions={
            "weight":"l,2,-,idx,one,=,0,*,idx,two,=,1,*,+,idx,three,=,2,*,+,idx,four,=,3,*,+,idx,five,=,4,*,+,idx,six,=,5,*,+,idx,seven,=,6,*,+,idx,eight,=,7,*,+,idx,nine,=,8,*,+,-,l,+,l,%"
        },inputs=["idx","l"]+outs,outputs=["weight"],scalars=["l","idx","weight"]+outs,hide=False)
        tree.links.new(attr_idx.std_out,sort_numbers.inputs["idx"])
        tree.links.new(self.group_inputs.outputs["CycleLength"],sort_numbers.inputs["l"])
        tree.links.new(digits.outputs["one"],sort_numbers.inputs["one"])
        tree.links.new(digits.outputs["two"],sort_numbers.inputs["two"])
        tree.links.new(digits.outputs["three"],sort_numbers.inputs["three"])
        tree.links.new(digits.outputs["four"],sort_numbers.inputs["four"])
        tree.links.new(digits.outputs["five"],sort_numbers.inputs["five"])
        tree.links.new(digits.outputs["six"],sort_numbers.inputs["six"])
        tree.links.new(digits.outputs["seven"],sort_numbers.inputs["seven"])
        tree.links.new(digits.outputs["eight"],sort_numbers.inputs["eight"])
        tree.links.new(digits.outputs["nine"],sort_numbers.inputs["nine"])

        sort_elements = SortElements(tree,domain="INSTANCE",sort_weight=sort_numbers.outputs["weight"],)

        index = Index(tree)
        attr_cycle_index = StoredNamedAttribute(tree,name="CycleIndex",domain="INSTANCE",data_type="INT",value=index.std_out,hide=False)

        target_index=make_function(tree,name="TargetIndex",
                    functions={
                        "idx":"idx,l,+,1,-,l,%"
                    },inputs=["idx","l"],outputs=["idx"],
                    scalars=["idx","l"],vectors=[],hide=True)
        tree.links.new(index.std_out,target_index.inputs["idx"])
        tree.links.new(self.group_inputs.outputs["CycleLength"],target_index.inputs["l"])

        # attr_target_pos = NamedAttribute(tree,name="Position",domain="INSTANCE",data_type="FLOAT_VECTOR",hide=True)
        attr_target_pos = Position(tree,hide=False)
        target_pos = EvaluateAtIndex(tree,domain="INSTANCE",data_type="FLOAT_VECTOR", index=target_index.outputs["idx"],value=attr_target_pos.std_out,hide=True)
        attr_target_position = StoredNamedAttribute(tree, name="TargetPosition", domain="INSTANCE",
                                                    data_type="FLOAT_VECTOR", value=target_pos.std_out, hide=False)


        # move out
        attr_prime = NamedAttribute(tree,name="Prime",domain="INSTANCE",data_type="INT",hide=True)
        foreach_mover = ForEachZone(tree,domain="INSTANCE",hide=False)

        move_out_function = make_function(tree,name="MoveOutFunction",
                    functions={
                        "pos":["0","0","up,prime,%,0,=,1,*,down,prime,%,0,=,-1,*,+,displace,*"]
                    },inputs=["prime","up","down","displace"],outputs=["pos"],
                    scalars=["prime","up","down","displace"],vectors=["pos"],hide=True)
        tree.links.new(self.group_inputs.outputs["UpMover"],move_out_function.inputs["up"])
        tree.links.new(self.group_inputs.outputs["DownMover"],move_out_function.inputs["down"])
        tree.links.new(self.group_inputs.outputs["Displacement"],move_out_function.inputs["displace"])
        tree.links.new(attr_prime.std_out,move_out_function.inputs["prime"])
        mover_pos = SetPosition(tree,name="MoveOutPosition",offset=move_out_function.outputs["pos"])
        foreach_mover.create_geometry_line([mover_pos])

        #cycle implementation
        repeat =RepeatZone(tree,iterations=self.group_inputs.outputs["CycleLength"],hide=False)
        get_cycle_index=NamedAttribute(tree,name="CycleIndex",data_type="INT",hide=False)
        pair_function = make_function(tree,name="PairFunction",
                    functions={
                    "selection":"idx,iter,=,idx,iter,l,1,-,+,l,%,=,or"
                    },inputs=["idx","iter","l"],outputs=["selection"],
                    scalars=["idx","iter","l","selection"],vectors=[])
        tree.links.new(get_cycle_index.std_out,pair_function.inputs["idx"])
        tree.links.new(self.group_inputs.outputs["CycleLength"],pair_function.inputs["l"])
        tree.links.new(repeat.iteration,pair_function.inputs["iter"])
        select_cycle = SeparateGeometry(tree,domain="INSTANCE",
                                        selection=pair_function.outputs["selection"],hide=True)

        mover_function = make_function(tree,name="MoverFunction",
                    functions={
                    "selection":"idx,iter,="
                    },inputs=["idx","iter"],outputs=["selection"],
                    scalars=["idx","iter","selection"],vectors=[],hide=True)
        tree.links.new(get_cycle_index.std_out,mover_function.inputs["idx"])
        tree.links.new(repeat.iteration,mover_function.inputs["iter"])

        select_mover = SeparateGeometry(tree,domain="INSTANCE",selection=mover_function.outputs["selection"],hide=True)

        #connect mover to displacement values
        switch = IndexSwitch(tree,data_type="FLOAT",hide=False,index=repeat.iteration)
        for i in range(self.max_length):
            switch.add_item(self.group_inputs.outputs["Mover"+str(i+1)])

        attr_target = NamedAttribute(tree,name="TargetPosition",data_type="FLOAT_VECTOR",hide=True)
        # attr_pos = NamedAttribute(tree,name="Position",data_type="FLOAT_VECTOR",hide=True)
        attr_pos = Position(tree,hide=True)

        progress_function =make_function(tree,name="Progress",
                    functions={
                        "position":["target_x,p_x,-,progress,*","0","0"]
                    },inputs=["progress","target","p"],outputs=["position"],
                    scalars=["progress"],vectors=["target","p","position"],hide=True)
        tree.links.new(switch.std_out,progress_function.inputs["progress"])
        tree.links.new(attr_target.std_out,progress_function.inputs["target"])
        tree.links.new(attr_pos.std_out,progress_function.inputs["p"])

        move = SetPosition(tree,offset=progress_function.outputs["position"])
        local_join = JoinGeometry(tree,hide=True)
        repeat.create_geometry_line([select_cycle,select_mover,move,local_join])
        tree.links.new(select_cycle.outputs["Inverted"],local_join.geometry_in)
        tree.links.new(select_mover.outputs["Inverted"],local_join.geometry_in)

        # update position
        update_pos = Position(tree,hide=True)
        store_pos = StoredNamedAttribute(tree,name="Position",data_type="FLOAT_VECTOR",
                                         value=update_pos.std_out,hide=True)
        final_join = JoinGeometry(tree,hide=True)
        tree.links.new(select_numbers.outputs["Inverted"],final_join.geometry_in)
        create_geometry_line(tree,[select_numbers,sort_elements,attr_cycle_index,attr_target_position,foreach_mover,repeat,store_pos,final_join],
                             out=self.group_outputs.inputs["Geometry"],
                             ins=self.group_inputs.outputs["Geometry"])

class TranslateToCenterNode(NodeGroup):
    def __init__(self,tree,max_length=5,**kwargs):
        """
        Geometry node that translates the center of the geometry to the origin

        """
        self.name = get_from_kwargs(kwargs,"name","CenterToOrigin")
        super().__init__(tree,inputs={"SourceGeometry":"GEOMETRY","TargetGeometry":"GEOMETRY"},
                         outputs={"Geometry":"GEOMETRY"},auto_layout=True,
                         name=self.name,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_in = self.node.inputs["TargetGeometry"]
        self.geometry_out = self.node.outputs["Geometry"]

    def fill_group_with_node(self,tree,**kwargs):
        pos = Position(tree)
        stat = AttributeStatistic(tree,geometry=self.group_inputs.outputs["SourceGeometry"],
                                  data_type="FLOAT_VECTOR",attribute=pos.std_out)
        center_function = make_function(tree,name="CenterFunction",
                                        functions = {
                                            "center":"maximum,minimum,add,-0.5,scale"
                                        },inputs=["maximum","minimum"],outputs=["center"],
                                        vectors=["maximum","minimum","center"],hide=True)

        tree.links.new(stat.outputs["Max"],center_function.inputs["maximum"])
        tree.links.new(stat.outputs["Min"],center_function.inputs["minimum"])

        transform = TransformGeometry(tree,translation=center_function.outputs["center"])
        create_geometry_line(tree,[transform],out=self.group_outputs.inputs["Geometry"],ins=self.group_inputs.outputs["TargetGeometry"])

class SlicerNode(NodeGroup):
    def __init__(self,tree,scale=0.7,thickness=0.02,slicing_geometry=None,**kwargs):
        """
        A node that is used to build the MegaMinx

        """
        self.name = get_from_kwargs(kwargs,"name","SlicerNode")
        super().__init__(tree,inputs={"SlicingGeometry":"GEOMETRY","Geometry":"GEOMETRY","Scale":"FLOAT","Thickness":"FLOAT"},
                         outputs={"SlicedGeometry":"GEOMETRY"},auto_layout=True,name=self.name,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs["SlicedGeometry"]

        if isinstance(scale,(int,float)):
            self.node.inputs["Scale"].default_value =scale
        else:
            tree.links.new(scale,self.node.inputs["Scale"])

        if isinstance(thickness,(int,float)):
            self.node.inputs["Thickness"].default_value =thickness
        else:
            tree.links.new(thickness,self.node.inputs["Thickness"])
        if slicing_geometry is not None:
            tree.links.new(slicing_geometry,self.node.inputs["SlicingGeometry"])

    def fill_group_with_node(self,tree,**kwargs):
        transform = TransformGeometry(tree,scale=self.group_inputs.outputs["Scale"])
        create_geometry_line(tree,[transform],ins=self.group_inputs.outputs["SlicingGeometry"])

        index = Index(tree)
        stat = AttributeStatistic(tree,geometry=self.group_inputs.outputs["Geometry"],domain="FACE",data_type="FLOAT",
                                  attribute=index.std_out,std_out="Max")
        add_one = MathNode(tree,operation="ADD",inputs0 = stat.std_out,inputs1=1)
        repeat = RepeatZone(tree,iterations=add_one.outputs["Value"])
        # inside repeat zone
        idx = Index(tree)
        compare = CompareNode(tree,data_type="INT",operation="EQUAL",inputs0=idx.std_out,inputs1=repeat.outputs["Iteration"])
        sep_geo = SeparateGeometry(tree,domain="FACE",selection=compare.std_out)
        scale_faces = ScaleElements(tree,domain="FACE",scale=2)
        extrude = ExtrudeMesh(tree,mode="FACES",offset=None,offset_scale=self.group_inputs.outputs["Thickness"])
        join = JoinGeometry(tree)
        mesh_boolean = MeshBoolean(tree, operation="DIFFERENCE", solver="FLOAT",
                                   self_intersection=True, )
        create_geometry_line(tree,[transform,sep_geo,scale_faces,join],out=mesh_boolean.inputs["Mesh 2"])
        create_geometry_line(tree, [scale_faces,extrude,join])
        repeat.create_geometry_line([mesh_boolean])


        create_geometry_line(tree,[repeat],ins=self.group_inputs.outputs["Geometry"],out=self.group_outputs.inputs["SlicedGeometry"])

class BevelFaces(NodeGroup):
    def __init__(self,tree,radius=0.01,bevel=2,
                 preserved_attribute=None,
                 attr_domain="FACE",
                 attr_data_type="INT",**kwargs):
        self.name = get_from_kwargs(kwargs,"name","BevelFaces")
        super().__init__(tree,inputs={"Geometry":"GEOMETRY","Radius":"FLOAT","Bevel":"INT",
                                      "preserved_attribute":attr_data_type},
                         outputs={"Geometry":"GEOMETRY"},auto_layout=True,
                         name=self.name,
                         preserved_attribute=preserved_attribute,
                        attr_domain=attr_domain,
                         attr_data_type=attr_data_type,**kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_in = self.node.inputs["Geometry"]
        self.geometry_out = self.node.outputs["Geometry"]

        if isinstance(radius,(int,float)):
            self.node.inputs["Radius"].default_value =radius
        else:
            tree.links.new(radius,self.node.inputs["Radius"])

        if isinstance(bevel,int):
            self.node.inputs["Bevel"].default_value =bevel
        else:
            tree.links.new(bevel,self.node.inputs["Bevel"])

        if preserved_attribute is not None:
            tree.links.new(preserved_attribute,self.node.inputs["preserved_attribute"])

    def fill_group_with_node(self,tree,**kwargs):
        icosphere=IcoSphere(tree,radius=self.group_inputs.outputs["Radius"],
                            subdivisions=self.group_inputs.outputs["Bevel"])
        foreachface = ForEachZone(tree,domain="FACE",hide=False)
        foreachface.add_socket(socket_type=get_from_kwargs(kwargs,"attr_data_type","FLOAT"),
                               name="preserved_attribute")
        tree.links.new(self.group_inputs.outputs["preserved_attribute"],foreachface.inputs["preserved_attribute"])
        iop = InstanceOnPoints(tree, instance=icosphere.geometry_out)
        realize_geo = RealizeInstances(tree)
        convex_hull = ConvexHull(tree)
        geometry_to_instance = GeometryToInstance(tree)
        preserved_attr = get_from_kwargs(kwargs,"preserved_attribute",None)
        if preserved_attr is not None:
            store_attr = StoredNamedAttribute(tree,data_type=get_from_kwargs(kwargs,"attr_data_type","FLOAT"),
                                              domain = get_from_kwargs(kwargs,"attr_domain","FACE"),
                                              name=get_from_kwargs(kwargs,"attr_name","attribute_name"),
                                              value=foreachface.outputs["preserved_attribute"])
            foreachface.create_geometry_line([iop,realize_geo,convex_hull,geometry_to_instance,store_attr])
        else:
            foreachface.create_geometry_line([iop,realize_geo,convex_hull,geometry_to_instance])

        create_geometry_line(tree,[foreachface],ins=self.group_inputs.outputs["Geometry"],out=self.group_outputs.inputs["Geometry"])

class PolyhedronViewNode(NodeGroup):
    def __init__(self,tree,edge_material=None,edge_radius=0.05,vertex_material=None,
                 vertex_radius=0.1,highlight_root=False,root_material=None,**kwargs):
        self.name = get_from_kwargs(kwargs, "name", "PolyhedronViewNode")
        super().__init__(tree, inputs={"Mesh": "GEOMETRY","Edge Radius": "FLOAT", "Edge Material": "MATERIAL", "Vertex Radius": "FLOAT","Vertex Material": "MATERIAL",
                                       "Highlight Root":"BOOLEAN", "Root Material":"MATERIAL"},
                         outputs={"Mesh": "GEOMETRY"}, auto_layout=False, name=self.name, **kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_in = self.node.inputs["Mesh"]
        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(edge_radius,(int,float)):
            self.node.inputs["Edge Radius"].default_value =edge_radius
        else:
            tree.links.new(edge_radius,self.node.inputs["Edge Radius"])
        if isinstance(vertex_radius,(int,float)):
            self.node.inputs["Vertex Radius"].default_value =vertex_radius
        else:
            tree.links.new(vertex_radius,self.node.inputs["Vertex Radius"])

        if edge_material is not None:
            tree.links.new(edge_material,self.node.inputs["Edge Material"])
        if vertex_material is not None:
            tree.links.new(vertex_material,self.node.inputs["Vertex Material"])
        if root_material is not None:
            tree.links.new(root_material,self.node.inputs["Root Material"])

        if highlight_root:
            if isinstance(highlight_root,(bool)):
                self.node.inputs["Highlight Root"].default_value=highlight_root
            else:
                tree.links.new(highlight_root,self.node.inputs["Highlight Root"])

    def fill_group_with_node(self, tree, **kwargs):
        # remove any existing node
        nodes = tree.nodes
        for n in nodes:
            nodes.remove(n)

        create_from_xml(tree, "PolyhedronView_nodes", **kwargs)

class ShowNormalsNode(NodeGroup):
    def __init__(self,tree,thickness=0.1,length=1,material=None,**kwargs):
        self.name = get_from_kwargs(kwargs, "name", "ShowNormalsNode")
        super().__init__(tree, inputs={"Mesh": "GEOMETRY","Thickness": "FLOAT", "Length": "FLOAT","Material": "MATERIAL"},
                         outputs={"Mesh": "GEOMETRY"}, auto_layout=False, name=self.name, **kwargs)

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

        self.geometry_in = self.node.inputs["Mesh"]
        self.geometry_out = self.node.outputs["Mesh"]

        if isinstance(thickness,(int,float)):
            self.node.inputs["Thickness"].default_value =thickness
        else:
            tree.links.new(thickness,self.node.inputs["Thickness"])
        if isinstance(length,(int,float)):
            self.node.inputs["Length"].default_value =length
        else:
            tree.links.new(length,self.node.inputs["Length"])

        if material is not None:
            tree.links.new(material,self.node.inputs["Material"])

    def fill_group_with_node(self, tree, **kwargs):
        # remove any existing node
        nodes = tree.nodes
        for n in nodes:
            nodes.remove(n)

        create_from_xml(tree, "ShowNormals_nodes", **kwargs)

# aux functions #

def get_attributes(line):
    tag_label_ended=false
    leftside = True
    keys = []
    values = []
    key=""
    val=""
    value_started=False
    for letter in line:
        if letter=='<':
            pass
        elif letter=='>':
            break # ignore anything after the end
        elif letter==' ' and not value_started: # only use spaces outside of String expressions as tag separator
            if not tag_label_ended:
                tag_label_ended=True
        else:
            if not tag_label_ended:
                pass # part of the tag label, can be ignored
            else:
                if letter=='=' and not value_started:
                    leftside=False
                    value_started=False
                else:
                    if leftside:
                        key+=letter
                    else:
                        if letter=='"':
                            if value_started:
                                value_started=False
                                keys.append(key)
                                values.append(val)
                                key=""
                                val=""
                                leftside=True
                            else:
                                value_started=True
                        else:
                            val+=letter
    attributes={}
    for key,val in zip(keys,values):
        attributes[key]=val
    return attributes


def parse_default_attribute(attribute):
    """
    unfortunately, we have capture some invalid data
    math nodes like Compare have VECTOR sockets that are not used but are filled with <bpy_float[3]...> stuff that cannot be parsed
    """
    if (attribute.startswith("(") and attribute.endswith(")")) or (attribute.startswith("[") and attribute.endswith("]")):
        attribute = attribute[1:-1]
        parts = attribute.split(",")
        comps = [float(part) for part in parts]
        return tuple(comps)


def get_default_value_for_socket(attributes):
    socket_type = attributes['type']
    if socket_type == 'INT':
        return int(attributes['default_value'])
    elif socket_type == 'BOOLEAN':
        attr = attributes['default_value']
        if attr == 'True':
            attr = True
        else:
            attr = False
        return attr
    elif socket_type == 'VALUE':
        return float(attributes['default_value'])
    elif socket_type == 'VECTOR':
        parsed = parse_default_attribute(attributes['default_value'])
        # print("vector: ",parsed)
        return Vector(parsed)
    elif socket_type == 'STRING':
        return str(attributes['default_value'])
    elif socket_type =='RGBA':
        parsed = parse_default_attribute(attributes['default_value'])
        # print("color: ",list(parsed))
        return list(parsed)
    elif socket_type=='ROTATION':
        parsed = parse_default_attribute(attributes['default_value'])
        # print("rotation: ",list(parsed))
        return list(parsed)
    elif socket_type=='MATERIAL':
        color = attributes['default_value']
        if color=='None' or len(color)==0:
            return None
        return ibpy.get_material(color)
    elif socket_type=="MENU":
        return attributes['default_value']


def create_socket(tree, node, node_attributes, attributes):
    # print(node)
    # print(node_attributes)
    # print(attributes)
    if attributes['type']=='CUSTOM':
        # empty socket, nothing to do
        return False
    else:
        if node_attributes['type'] == 'INDEX_SWITCH': # just add empty socket to Index Switch
            # this is called, when IndexSwitch node is created from XML
            if "default_value" in attributes:
                default_value = get_default_value_for_socket(attributes)
                node.add_new_item_from_xml(default_value)
            else:
                node.add_new_item_from_xml()
            return True
        if node_attributes['type']=='GROUP_INPUT':
            tree.interface.new_socket(attributes['name'],description='',in_out="INPUT",socket_type=SOCKET_TYPES[attributes['type']])
            if "default_value" in attributes:
                default_value = get_default_value_for_socket(attributes)
                tree.interface.items_tree.get(attributes['name']).default_value=default_value
            return True
        elif node_attributes['type']=='GROUP_OUTPUT':
            tree.interface.new_socket(attributes['name'],description='',in_out="OUTPUT",socket_type=SOCKET_TYPES[attributes['type']])
            return True
        else:
            # create socket in case of Repeat zone or GroupOutput or Simulation Zone
            if "GeometryNodeRepeatOutput" in node.name or "GeometryNodeRepeatInput" in node.name:
                node.add_socket(SOCKET_TYPES[node_attributes["type"]], node_attributes["name"])
                return True
            elif "Simulation Input" in node.name or "Simulation Output" in node.name:
                node.add_socket(attributes["type"], attributes["name"])
                return True
            return False

def create_from_xml(tree,filename=None,**kwargs):
    """
    create a node group from an xml file
    Warning: don't use ReRoute nodes. They are buggy

    """
    node_dir={}
    name_dir={}
    parent_dir={}

    save_for_after_pairing={}

    # create a node structure tree {0: {"inputs":{1,2,3}, "outputs":{4,5,6}}, ...}
    node_structure={}

    socket_count =0
    if filename:
        unpublished = get_from_kwargs(kwargs,"unpublished",False)
        if unpublished:
            path=os.path.join(RES_XML2,filename+".xml")
        else:
            path = os.path.join(RES_XML,filename+".xml")
        with open(path) as f:
            # find node range and link range in xml file
            xml_string = f.read()
            nodes_range=[]
            links_range=[]
            xml_text = xml_string.splitlines()
            for i,line in enumerate(xml_text):
                line=line.strip()
                if line.startswith("<NODES>"):
                    nodes_range.append(i+1)
                if line.startswith("</NODES>"):
                    nodes_range.append(i)
                if line.startswith("<LINKS>"):
                    links_range.append(i+1)
                if line.startswith("</LINKS>"):
                    links_range.append(i)

            # parse node data
            for i in range(*nodes_range):
                line = xml_text[i].strip()
                if line.startswith("<NODE"):
                    node_attributes = get_attributes(line)
                    node = Node.from_attributes(tree,node_attributes)
                    node_id = int(node_attributes["id"])
                    if node_id==22:
                        pass
                    node_name = node_attributes["name"]
                    node_structure[node_id]={"name": node_name, "inputs":dict(), "outputs":dict()}
                    if node is None:
                        raise "The node "+line+" could not be created"
                    node_dir[node_id]=node
                    name_dir[node_name]=node_id
                    if node_attributes["parent"]!="None":
                        parent_dir[node_id]=node_attributes["parent"]

                    while True:
                        i=i+1
                        line = xml_text[i].strip()
                        if line.startswith("</NODE>"):
                            break
                        elif line.startswith("<INPUTS>"):
                            input_count=0
                            if node_attributes["type"] in {'REPEAT_INPUT','FOREACH_GEOMETRY_ELEMENT_INPUT',"SIMULATION_INPUT"}:
                                save_for_after_pairing[node_id]={"inputs":[],"outputs":[]}
                        elif line.startswith("<OUTPUTS>"):
                            output_count=0
                        elif line.startswith("</INPUTS>"):
                            pass
                        elif line.startswith("</OUTPUTS>"):
                            pass
                        elif line.startswith("<INPUT "):
                            if node_id == 20:
                                pass
                            input_attributes = get_attributes(line)
                            input_id = int(input_attributes["id"])
                            if node_attributes["type"] in {"REPEAT_INPUT","FOREACH_GEOMETRY_ELEMENT_INPUT","SIMULATION_INPUT"}:
                                # repeat inputs can only be initiated after pairing
                                save_for_after_pairing[node_id]["inputs"].append(input_attributes)
                            elif len(node.inputs)>input_count and node.inputs[input_count].name!="": # avoid virtual socket
                                node_structure[node_id]["inputs"][input_id] = input_count
                                node.inputs[input_count].name=input_attributes['name']
                                if 'default_value' in input_attributes:
                                    node.inputs[input_count].default_value = get_default_value_for_socket(input_attributes)
                                input_count += 1
                            else:
                                if input_attributes["type"]!="CUSTOM": #FOREACH_GEOMETRY_ELEMENT_INPUT has a custom socket inbetween proper sockets
                                    result = create_socket(tree,node,node_attributes,input_attributes)
                                    if result:
                                        node_structure[node_id]["inputs"][input_id]=input_count
                                        input_count += 1
                                    else:
                                        print("Something went wrong with socket creation for node ",node_id)
                                else:
                                    # print("Warning: unrecognized socket in ", node_id, socket_count,input_attributes["type"])
                                    node_structure[node_id]["inputs"][input_id] = -1 # take last slot (this dynamically generates new sockets for Group Input and Group Output
                                    input_count +=1 #also increase input_count, since the custom socket can between real sockets
                            socket_count+=1
                        elif line.startswith("<OUTPUT "):
                            output_attributes = get_attributes(line)
                            output_id = int(output_attributes["id"])

                            if node_attributes["type"] in{'REPEAT_INPUT','FOREACH_GEOMETRY_ELEMENT_INPUT',"SIMULATION_INPUT"}:
                                # repeat inputs can only be initiated after pairing
                                save_for_after_pairing[node_id]["outputs"].append(output_attributes)

                            elif len(node.outputs)>output_count and node.outputs[output_count].name!="": # avoid virtual socket
                                node.outputs[output_count].name=output_attributes["name"]
                                node_structure[node_id]["outputs"][output_id] = output_count
                                if 'default_value' in output_attributes:
                                    node.outputs[output_count].default_value = get_default_value_for_socket(output_attributes)
                                output_count += 1
                            else:
                                if output_attributes["type"] !="CUSTOM": # FOREACH_GEOMETRY_ELEMENT_OUTPUT has a custom socket in between proper sockets
                                    result = create_socket(tree,node,node_attributes,output_attributes)
                                    if result:
                                        node_structure[node_id]["outputs"][output_id]=output_count
                                        output_count+=1
                                    else:
                                        print("Something went wrong with creating sockets for ",node_id)
                                else:
                                    # print("Warning: unrecognized socket in ",node_id,socket_count,output_attributes["type"])
                                    node_structure[node_id]["outputs"][output_id] = -1 # take last slot (this dynamically generates new sockets for Grou
                                    output_count += 1 # also increase output_count, since the custom socket can be between real sockets
                            socket_count += 1

            # establish parent relations
            for key, val in parent_dir.items():
                if node_dir[key] is not None:
                    # before parenting is established, the node needs to be shifted to the position of the parent
                    # node_dir[key].location=(node_dir[key].location[0]+node_dir[name_dir[val]].location[0],node_dir[key].location[1]+node_dir[name_dir[val]].location[1])
                    parent = node_dir[name_dir[val]]
                    child = node_dir[key]

                    parent_loc = parent.node.location
                    child_loc = child.node.location
                    child.node.location = (parent_loc[0]+child_loc[0],parent_loc[1]+child_loc[1])
                    child.set_parent(parent)

            # check for zone pairing
            for key, val in node_dir.items():
                name = node_structure[key]["name"]
                # print(name)
                key=int(key)
                # the input sockets are only created after pairing with the output node
                # therefore the links can only be created after pairing
                if "ForEachGeometryElementInput" in name or "For Each Geometry Element Input" in name:
                    # find the corresponding output node from name
                    out_name = name.replace("Input", "Output")
                    node_dir[key].pair_with_output(node_dir[name_dir[out_name]])
                    input_count = 0
                    for attributes in save_for_after_pairing[key]["inputs"]:
                        input_id = int(attributes['id'])
                        node_structure[key]["inputs"][input_id] = input_count
                        input_count += 1
                    output_count = 0
                    for attributes in save_for_after_pairing[key]["outputs"]:
                        output_id = int(attributes['id'])
                        node_structure[key]["outputs"][output_id] = output_count
                        output_count += 1
                if "Repeat Input" in name:
                    # find the corresponding output node from name
                    out_name = name.replace("Input", "Output")
                    node_dir[key].pair_with_output(node_dir[name_dir[out_name]])
                    input_count=0
                    for  attributes in save_for_after_pairing[key]["inputs"]:
                        input_id = int(attributes['id'])
                        node_structure[key]["inputs"][input_id]=input_count
                        input_count+=1
                    output_count=0
                    for attributes in save_for_after_pairing[key]["outputs"]:
                        output_id = int(attributes['id'])
                        node_structure[key]["outputs"][output_id]=output_count
                        output_count+=1
                if "Simulation Input" in name:
                    # find the corresponding output node from name
                    out_name = name.replace("Input", "Output")
                    node_dir[key].pair_with_output(node_dir[name_dir[out_name]])
                    input_count=0
                    for  attributes in save_for_after_pairing[key]["inputs"]:
                        input_id = int(attributes['id'])
                        node_structure[key]["inputs"][input_id]=input_count
                        input_count+=1
                    output_count=0
                    for attributes in save_for_after_pairing[key]["outputs"]:
                        output_id = int(attributes['id'])
                        node_structure[key]["outputs"][output_id]=output_count
                        output_count+=1

            # parse link data
            for i in range(*links_range):
                node_attributes = get_attributes(xml_text[i])
                from_socket = int(node_attributes["from_socket"])
                to_socket= int(node_attributes["to_socket"])
                from_node = int(node_attributes["from_node"])
                to_node = int(node_attributes["to_node"])

                output_id = node_structure[from_node]["outputs"][from_socket]
                if  to_socket==157:
                    pass
                input_id = node_structure[to_node]["inputs"][to_socket]

                # print("link ",node_dir[from_node],": ",str(from_socket),"->",str(to_socket),": ",node_dir[to_node])
                if output_id<len(node_dir[from_node].outputs):
                    if input_id<len(node_dir[to_node].inputs) and output_id<len(node_dir[from_node].outputs):
                        # check for virtual ports
                        if node_dir[to_node].inputs[input_id].type=="CUSTOM":
                            node = node_dir[to_node]
                            from_socket = node_dir[from_node].outputs[output_id]
                            pair_node = node.pair_node
                            pair_node.add_socket(from_socket.type,from_socket.name)
                            pass
                        tree.links.new(node_dir[from_node].outputs[output_id],node_dir[to_node].inputs[input_id])
                    else:
                        if not input_id < len(node_dir[to_node].inputs):
                            print("Failed to connect to input "+str(input_id)+" of node "+str(to_node))
                        if not output_id<len(node_dir[from_node].outputs):
                            print("Failed to connect from output "+str(output_id)+" of node "+str(from_node))

def create_geometry_line(tree, green_nodes, out=None, ins=None):
    first = True
    if len(green_nodes)==0:
        tree.links.new(ins,out)
    else:
        for gn in green_nodes:
            if first:
                first = False
                last_gn = gn
                if ins:
                    tree.links.new(ins, last_gn.geometry_in)
            else:
                tree.links.new(last_gn.geometry_out, gn.geometry_in)
                last_gn = gn
        if out:
            tree.links.new(last_gn.geometry_out, out)

def add_locations(loc1, loc2):
    return tuple(map(sum, zip(loc1, loc2)))

class Structure:
    def __init__(self):
        self.left = None
        self.right = None
        self.out = None


def make_function(nodes_or_tree, functions={}, inputs=[], outputs=[], vectors=[], scalars=[],rotations=[],
                  node_group_type="GeometryNodes",
                  name="FunctionNode", hide=True, location=(0, 0),parent=None):
    """
    this will be the optimized prototype for a flexible function generator
    functions: a dictionary that contains a key for every output. If the key is in vectors,
     either a list of three functions is required or a function with vector output
    :return:
    """
    if hasattr(nodes_or_tree, "nodes"):
        tree = nodes_or_tree
        nodes = tree.nodes
    else:
        nodes = nodes_or_tree

    if "Shader" in node_group_type:
        tree = bpy.data.node_groups.new(type="ShaderNodeTree", name=name)
        group = nodes.new(type="ShaderNodeGroup")
    else:
        tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        group = nodes.new(type="GeometryNodeGroup")

    group.name = name
    group.node_tree = tree

    tree_nodes = tree.nodes
    tree_links = tree.links

    # create inputs and outputs
    group_inputs = tree_nodes.new("NodeGroupInput")
    group_outputs = tree_nodes.new("NodeGroupOutput")

    for ins in inputs:
        if ins in vectors:
            make_new_socket(tree, name=ins, io="INPUT", type="NodeSocketVector")
        if ins in scalars:
            make_new_socket(tree, name=ins, io="INPUT", type="NodeSocketFloat")
        if ins in rotations:
            make_new_socket(tree,name=ins,io="INPUT",type="NodeSocketRotation")

    for outs in outputs:
        if outs in vectors:
            make_new_socket(tree, name=outs, io="OUTPUT", type="NodeSocketVector")
        if outs in scalars:
            make_new_socket(tree, name=outs, io="OUTPUT", type="NodeSocketFloat")
        if outs in rotations:
            make_new_socket(tree,name=outs, io="OUTPUT", type="NodeSocketRotation")

    # create stack structure from function structure
    stacks = {}
    for key, value in functions.items():
        if isinstance(value, list):
            stacks[key] = []
            for v in value:
                stacks[key].append(v.split(","))
        else:
            stacks[key] = value.split(",")

    all_stacks = []
    all_terms = []
    for value in stacks.values():
        if isinstance(value, list):
            all_stacks += value
            all_terms += value
            # p6majo 20240801 strange behaviour for scalar functions
            # for v in value:
            #     all_terms += v
        else:
            all_stacks.append(value)
            all_terms += value

    # find the longest function and position group_inputs and group_outputs
    lengths = []
    for stack in all_stacks:
        length = 1
        for s in stack:
            if s in OPERATORS:
                length += 1
        lengths.append(length)

    length = max(lengths)
    length //= 2
    width = 200
    left = -length * width
    right = 0
    length = -1
    group_inputs.location = (left, 0)
    group_outputs.location = (width, 0)

    # prepare output channels
    out_channels = {}
    combine_counter = 0
    for key, value in functions.items():
        if key in scalars:
            out_channels[key] = group_outputs.inputs[key]
        elif key in vectors or key in rotations:
            if isinstance(functions[key], list):
                comb = tree_nodes.new(type="ShaderNodeCombineXYZ")
                comb.name = key + "Merge"
                comb.label = key + "Merge"
                comb.location = (right, combine_counter * width / 2)
                comb.hide = True
                combine_counter += 1
                out_channels[key + "_x"] = comb.inputs[0]
                out_channels[key + "_y"] = comb.inputs[1]
                out_channels[key + "_z"] = comb.inputs[2]
                tree_links.new(comb.outputs[0], group_outputs.inputs[key])
            else:
                out_channels[key] = group_outputs.inputs[key]

    # prepare inputs
    in_channels = {}
    separate_counter = 0
    length = 0

    all_terms = maybe_flatten(all_terms)
    for ins in inputs:
        if ins in scalars:
            in_channels[ins] = group_inputs.outputs[ins]
        if ins in vectors or ins in rotations:
            in_channels[ins] = group_inputs.outputs[ins]
            if ins + "_x" in all_terms or ins + "_y" in all_terms or ins + "_z" in all_terms:
                sep = tree_nodes.new(type="ShaderNodeSeparateXYZ")
                sep.name = ins + "Split"
                sep.label = ins + "Split"
                tree_links.new(group_inputs.outputs[ins], sep.inputs["Vector"])
                sep.location = (left + width, separate_counter * width / 2)
                sep.hide = True
                in_channels[ins + "_x"] = sep.outputs[0]
                in_channels[ins + "_y"] = sep.outputs[1]
                in_channels[ins + "_z"] = sep.outputs[2]
                separate_counter += 1

    # now the functions are constructed
    fcn_count = 0  # function index to get a separation in the node editor
    comps = ["x", "y", "z"]
    for key, value in functions.items():
        if isinstance(value, list):
            if len(value) == 1:
                build_function(tree, stacks[key][0], scalars=scalars, vectors=vectors,rotations=rotations, in_channels=in_channels,
                               out=out_channels[key], fcn_count=fcn_count)
                fcn_count += 1
            else:
                for i, part in enumerate(value):
                    build_function(tree, stacks[key][i], scalars=scalars, vectors=vectors, rotations = rotations, in_channels=in_channels,
                                   out=out_channels[key + "_" + comps[i]], fcn_count=fcn_count)
                    fcn_count += 1
        else:
            build_function(tree, stacks[key], scalars=scalars, vectors=vectors, rotations = rotations, in_channels=in_channels,
                           out=out_channels[key], fcn_count=fcn_count)
            fcn_count += 1

    layout(tree)
    if hide:
        group.hide = True

    if parent:
        group.parent = parent.node
        location = ((location[0] + parent.location[0]) * 200, (location[1] + parent.location[1]) * 100)
    else:
        location = (location[0]  * 200, location[1]  * 100)
    group.location = location
    return group


def build_function(tree, stack, scalars=[], vectors=[], rotations=[], in_channels={},
                   fcn_count=0, out=None, unary=None,
                   last_operator=None,
                   last_structure=None,
                   length=1, height=0, level=[0]):
    """
    recursive build of a node-group function

    there is a subtlety with VectorMath nodes, they always carry two outputs.
    The first one is "Vector" and the second one is "Value"
    there is more work to be done, to do this correctly,
    so far there is only a workaround to incorporate the "LENGTH" operation, which yields a scalar output

    :param tree: the container for the function
    :param stack: contains the computation in reverse polish notation
    :param scalars: scalar variables
    :param vectors: vector variables
    :param rotations: rotation variables
    :param in_channels: incoming connections
    :param out: one single out connection
    :param fcn_count: controls the vertical position at which the nodes of this function are displayed
    :param last_operator: holds the value for the parent operator
     :param last_structure: holds the input structure for the parent operator
    :param unary:
    :param level: used for layout
    :param length: used for layout
    :param height: used for layout
    :return:
    """

    # recursively build a group_tree structure with left and right sub-group_tree. For unary operators only the left group_tree is used
    fcn_spacing = 500

    left_empty = True
    if unary:
        right_empty = False  # no need for a right sub-group_tree
    else:
        right_empty = True

    # a variable that a new operator is assigned to
    new_node_math = None

    while (left_empty or right_empty) and len(stack) > 0:
        next_element = stack.pop()
        if next_element in OPERATORS:
            # warning not all possible operators have been implemented yet
            # always implement them on the fly when needed
            unary = False  # default case, unary operators explicitly overwrite this variable
            if next_element == "*":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "MULTIPLY"
                new_node_math.label = "*"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "mul":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "MULTIPLY"
                new_node_math.label = "*"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "%":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "MODULO"
                new_node_math.label = "%"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "mod":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "MODULO"
                new_node_math.label = "%"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "/":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "DIVIDE"
                new_node_math.label = "/"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "div":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "DIVIDE"
                new_node_math.label = "/"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "+":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "ADD"
                new_node_math.label = "+"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "add":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "ADD"
                new_node_math.label = "+"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "sub":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "SUBTRACT"
                new_node_math.label = "-"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "-":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "SUBTRACT"
                new_node_math.label = "-"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "**":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "POWER"
                new_node_math.label = "**"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "<":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "LESS_THAN"
                new_node_math.label = "<"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == ">":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "GREATER_THAN"
                new_node_math.label = ">"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "=":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "COMPARE"
                new_node_math.label = "=="
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_math.inputs[2].default_value = 0
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "min":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "MINIMUM"
                new_node_math.label = "min"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "max":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "MAXIMUM"
                new_node_math.label = "max"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "sin":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "SINE"
                new_node_math.label = "sin"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "sinh":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "SINH"
                new_node_math.label = "sinh"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "lg":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "LOGARITHM"
                new_node_math.label = "lg"
                new_node_math.inputs[1].default_value = 10
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "asin":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "ARCSINE"
                new_node_math.label = "asin"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "cos":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "COSINE"
                new_node_math.label = "cos"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "cosh":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "COSH"
                new_node_math.label = "cosh"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "acos":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "ARCCOSINE"
                new_node_math.label = "acos"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "tan":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "TANGENT"
                new_node_math.label = "tan"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "atan2":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "ARCTAN2"
                new_node_math.label = "atan2"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = False
            elif next_element == "abs":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "ABSOLUTE"
                new_node_math.label = "abs"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "sgn":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "SIGN"
                new_node_math.label = "sgn"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "round":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "ROUND"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "floor":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "FLOOR"
                new_node_math.label = "floor"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "vfloor":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "FLOOR"
                new_node_math.label = "floor"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Vector"]
                unary = True
            elif next_element == "ceil":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "CEIL"
                new_node_math.label = "ceil"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "length":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "LENGTH"
                new_node_math.label = "len"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "sqrt":
                new_node_math = tree.nodes.new(type="ShaderNodeMath")
                new_node_math.operation = "SQRT"
                new_node_math.label = "sqrt"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Value"]
                unary = True
            elif next_element == "scale":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "SCALE"
                new_node_math.label = "scale"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs["Scale"]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "cross":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "CROSS_PRODUCT"
                new_node_math.label = "x"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "dot":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "DOT_PRODUCT"
                new_node_math.label = "*"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Value"]
            elif next_element == "normalize":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorMath")
                new_node_math.operation = "NORMALIZE"
                new_node_math.label = "norm"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Vector"]
                unary = True
            elif next_element == "rot":
                new_node_math = tree.nodes.new(type="ShaderNodeVectorRotate")
                new_node_math.rotation_type = "EULER_XYZ"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs["Vector"]
                new_node_structure.right = new_node_math.inputs["Rotation"]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "axis_rot":
                new_node_math = tree.nodes.new(type="FunctionNodeAxisAngleToRotation")
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs["Axis"]
                new_node_structure.right = new_node_math.inputs["Angle"]
                new_node_structure.out = new_node_math.outputs["Rotation"]
            elif next_element == "rot2euler":
                new_node_math = tree.nodes.new(type="FunctionNodeRotationToEuler")
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs["Rotation"]
                new_node_structure.out = new_node_math.outputs["Euler"]
                unary = True
            elif next_element == "axis_angle_euler":
                """ convenient combination of axis_rot and rot2euler"""
                new_node_math = tree.nodes.new(type="FunctionNodeRotateEuler")
                new_node_math.type = "AXIS_ANGLE"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs["Axis"]
                new_node_structure.right = new_node_math.inputs["Angle"]
                new_node_structure.out = new_node_math.outputs["Rotation"]
            elif next_element == "not":
                new_node_math = tree.nodes.new(type="FunctionNodeBooleanMath")
                new_node_math.operation = "NOT"
                new_node_math.label = "NOT"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs["Boolean"]
                unary = True
            elif next_element == "and":
                new_node_math = tree.nodes.new(type="FunctionNodeBooleanMath")
                new_node_math.operation = "AND"
                new_node_math.label = "AND"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Boolean"]
            elif next_element == "or":
                new_node_math = tree.nodes.new(type="FunctionNodeBooleanMath")
                new_node_math.operation = "OR"
                new_node_math.label = "OR"
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs["Boolean"]
            elif next_element == "rot_vec":
                new_node_math = tree.nodes.new(type="FunctionNodeRotateVector")
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs["Vector"]
                new_node_structure.right = new_node_math.inputs["Rotation"]
                new_node_structure.out = new_node_math.outputs["Vector"]
            elif next_element == "inv_rot":
                new_node_math = tree.nodes.new(type="FunctionNodeInvertRotation")
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs["Rotation"]
                new_node_structure.out = new_node_math.outputs["Rotation"]
                unary = True

            # positioning is a non-trivial task, matter of improvement
            if unary:
                new_level = level + [0]
            else:
                if right_empty:
                    new_level = level + [-1]
                else:
                    new_level = level[0:-1] + [1]

            y_pos = 0
            for i, bit in enumerate(new_level):
                y_pos += bit * 500 / (i + 1)

            new_node_math.location = (length * 200, - fcn_spacing * fcn_count + y_pos)
            new_node_math.hide = True

            if last_operator is None:
                # link first operator to the output
                tree.links.new(new_node_structure.out, out)
            elif right_empty:
                # make sure that the type fits, e.g. the operator "LENGTH" first has an output of type "VECTOR"
                tree.links.new(new_node_structure.out, last_structure.right)
                # success = False
                # for o in new_node_math.outputs:
                #     if o.type == last_operator.inputs[1].type:
                #         group_tree.links.new(o, last_operator.inputs[1])
                #         success = True
                #         break
                # if not success:
                #     # try the other way round
                #     for i in range(len(last_operator.inputs) - 1, -1, -1):
                #         if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                #             group_tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                #             break
                right_empty = False
            elif left_empty:
                # make sure that the type fits, e.g. the operator "LENGTH" first has an output of type "VECTOR"
                tree.links.new(new_node_structure.out, last_structure.left)
                # success = False
                # for o in new_node_math.outputs:
                #     if o.type == last_operator.inputs[0].type:
                #         group_tree.links.new(o, last_operator.inputs[0])
                #         success = True
                #         break
                # if not success:
                #     for i in range(len(last_operator.inputs)):
                #         if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                #             group_tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                #             break
                left_empty = False

        elif next_element in scalars or next_element in vectors or next_element in rotations:
            if last_operator is None:
                tree.links.new(in_channels[next_element], out)
            elif right_empty:
                tree.links.new(in_channels[next_element], last_structure.right)
                right_empty = False
            elif left_empty:
                tree.links.new(in_channels[next_element], last_structure.left)
                left_empty = False

        # remove _x, _y, _z flag for parameter detection
        elif next_element[0:-2] in vectors or next_element[0:-2] in rotations:
            src = in_channels[next_element]
            if last_operator is None:
                tree.links.new(src, out)
            elif right_empty:
                tree.links.new(src, last_structure.right)
                right_empty = False
            elif left_empty:
                tree.links.new(src, last_structure.left)
                left_empty = False
        # check for simple numbers and unit vectors
        else:
            if next_element == "pi":
                number = np.pi
            elif next_element == "e_x":
                number = Vector([1, 0, 0])
            elif next_element == "e_y":
                number = Vector([0, 1, 0])
            elif next_element == "e_z":
                number = Vector([0, 0, 1])
            elif next_element[0] == "(":
                next_element = next_element[1:-1]
                numbers = next_element.split(" ")
                vals = []
                for i in range(len(numbers)):
                    if numbers[i] == "pi":
                        vals.append(np.pi)
                    elif number[i] == "-pi":
                        vals.append(-np.pi)
                    else:
                        vals.append(float(numbers[i]))
                number = Vector(vals)
            else:
                number = float(next_element)
            if last_operator is None:
                out.default_value = number
            elif right_empty:
                last_structure.right.default_value = number
                last_operator.label = last_operator.label + next_element
                # find the first "VALUE" input
                # for i in range(1, 3):
                #     if last_operator.inputs[i].type == "VALUE":
                #         last_operator.inputs[i].default_value = number
                #         break
                right_empty = False
            elif left_empty:
                last_structure.left.default_value = number
                last_operator.label = next_element + last_operator.label
                # last_operator.inputs[0].default_value = number
                left_empty = False
            else:
                raise "Something went wrong. The number " + next_element + " is left over."

        # if a new operator is processed the function has to be called again
        if new_node_math:
            build_function(tree, stack, scalars=scalars, vectors=vectors, rotations=rotations, in_channels=in_channels,
                           out=out, fcn_count=fcn_count, length=length - 1, unary=unary, last_operator=new_node_math,
                           last_structure=new_node_structure, height=height,
                           level=new_level)
            new_node_math = None


def layout(tree, mode="Sugiyama"):
    """
    automatic layout of the nodes
    :param tree:
    :return:
    """
    out_socket_dir = {}
    in_socket_dir = {}
    node_vertex_dir = {}
    vertices = []
    for node in tree.nodes:
        for out in node.outputs:
            out_socket_dir[out] = node
        for ins in node.inputs:
            in_socket_dir[ins] = node
        vertex = Vertex(node)
        node_vertex_dir[node] = vertex
        vertices.append(vertex)

    edges = [Edge(
        node_vertex_dir[out_socket_dir[link.from_socket]],
        node_vertex_dir[in_socket_dir[link.to_socket]]
    ) for link in tree.links]

    graph = Graph(vertices, edges)

    class DefaultView(object):
        w, h = 200, 200

    class HiddenView(object):
        w, h = 30, 200

    for vertex in vertices:
        if vertex.data.hide:
            vertex.view = HiddenView()
        else:
            vertex.view = DefaultView()

    # find roots (all nodes that only use output sockets)
    in_set = set(in_socket_dir.values())
    out_set = set(out_socket_dir.values())

    roots_nodes = out_set - out_set.intersection(in_set)
    root_vertices = [v for v in vertices if v.data in roots_nodes]
    if mode == "Sugiyama":
        layout = SugiyamaLayout(graph.C[0])
        layout.init_all(roots=root_vertices)
        layout.draw(10)
    elif mode == "Digco":
        layout = DigcoLayout(graph.C[0])
        layout.init_all()
        layout.draw()

    for v in graph.C[0].sV:
        v.data.location = (v.view.xy[1], v.view.xy[0])
