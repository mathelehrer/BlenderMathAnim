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
from interface.ibpy import get_material, make_new_socket, OPERATORS
from interface.interface_constants import blender_version
from mathematics.groups.e8 import E8Lattice
from utils.color_conversion import get_color
from utils.constants import RES_XML
from utils.kwargs import get_from_kwargs

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
        self.node.location = (self.l * node_width, self.m * node_height + offset_y)
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

        self.inputs = self.node.inputs
        self.outputs = self.node.outputs

    @classmethod
    def from_attributes(cls,tree,attributes):
        name = attributes["name"]
        # print("Create Node from attributes: ",attributes["id"],": ", name)
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
        if type=="RANDOM_VALUE":
            data_type = attributes["data_type"]
            return RandomValue(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,data_type=data_type)
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
            scale_mode=attributes["scale_mode"]
            return ScaleElements(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,domain=domain,scale_mode=scale_mode)

        # instances
        if type=="INSTANCE_ON_POINTS":
            return InstanceOnPoints(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

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
        if type=="MESH_PRIMITIVE_GRID":
            return Grid(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # curves
        if type=="CURVE_PRIMITIVE_CIRCLE":
            mode=attributes["mode"]
            return CurveCircle(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,mode=mode)
        if type=="CURVE_PRIMITIVE_QUADRILATERAL":
            return CurveQuadrilateral(tree, location=location, name=name, label=label, hide=hide, mute=mute,
                                      node_height=200)
        # if type=="CURVE_TO_POINTS":
        #     mode = attributes["mode"]
        #     return CurveToPoints(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,mode=mode)
        if type=="CURVE_TO_MESH":
            return CurveToMesh(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="RESAMPLE_CURVE":
            mode =attributes["mode"]
            return ResampleCurve(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,mode=mode)
        if type=="FILLET_CURVE":
            mode = attributes["mode"]
            return FilletCurve(tree, location=location, name=name, label=label, hide=hide, mute=mute,node_height=200,mode=mode)
        if type=="FILL_CURVE":
            mode = attributes["mode"]
            return FillCurve(tree, location=location, name=name, label=label, hide=hide, mute=mute,node_height=200,mode=mode)
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
        # switches
        if type=="SWITCH":
            input_type = attributes["input_type"]
            return Switch(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,input_type=input_type)

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
            location = parse_location(attributes["location"])
            return ReRoute(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="FRAME":
            location=parse_location(attributes["location"])
            return Frame(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        if type=="GROUP_OUTPUT":
            return GroupOutput(tree,location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # For Each
        if type=="FOREACH_GEOMETRY_ELEMENT_INPUT":
            return ForEachInput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        if type=="FOREACH_GEOMETRY_ELEMENT_OUTPUT":
            domain = attributes["domain"]
            return ForEachOutput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200,domain=domain)

        if type=="REPEAT_INPUT":
            return RepeatInput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)
        if type=="REPEAT_OUTPUT":
            return RepeatOutput(tree,location=location,name=name,label=label,hide=hide,mute=mute,node_height=200)

        # Custom Nodes
        if type=="GROUP":
            if "PositionTransform" in name:
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
    def __init__(self,tree,location=(0,0),hide=False,mute=False,**kwargs):
        self.node = tree.nodes.new(type="NodeFrame")
        self.node.hide=hide
        self.node.mute=mute
        super().__init__(tree,location,**kwargs)

    def add(self,node):
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

#################
#  green nodes  #
#################

# mesh primitives
class MeshLine(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mode="END_POINTS",
                 count_mode="TOTAL",  # alternative is "RESOLUTION"
                 count=10,
                 start_location=Vector([0, 0, 0]),
                 end_location=None,
                 offset=Vector([0,0,1]), **kwargs):

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
        if isinstance(offset,(list,Vector)):
            self.node.inputs["Offset"].default_value = offset
        else:
            self.tree.links.new(offset, self.node.inputs["Offset"])


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
                 offset=Vector(),
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
class Quadrilateral(GreenNode):
    def  __init__(self, tree, location=(0, 0),
                  mode="RECTANGLE",
                  node_width=1,
                  node_height=1,
                  **kwargs
                  ):
        self.node = tree.nodes.new(type="GeometryNodeCurvePrimitiveQuadrilateral")
        super().__init__(tree, location=location, **kwargs)
        self.node.mode = mode
        self.geometry_out = self.node.outputs["Curve"]

        if isinstance(node_width, (int, float)):
            self.node.inputs["Width"].default_value = node_width
        else:
            self.tree.links.new(node_width, self.node.inputs["Width"])
        if isinstance(node_height, (int, float)):
            self.node.inputs["Height"].default_value = node_height
        else:
            self.tree.links.new(node_height, self.node.inputs["Height"])

class ResampleCurve(GreenNode):
    def __init__(self, tree, location=(0, 0),mode="COUNT",curve=None,
                 selection = None,
                 count=1,
                 limit_radius="False",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeResampleCurve")
        super().__init__(tree,location=location,**kwargs)

        self.node.mode=mode
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

class FilletCurve(GreenNode):
    def __init__(self, tree, location=(0, 0),mode="POLY",radius=1,curve=None,
                 count=1,
                 limit_radius="False",**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeFilletCurve")
        super().__init__(tree,location=location,**kwargs)

        self.node.mode=mode
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
    def __init__(self,tree,location=(0,0),mode="NGONS",curve=None,group_id=None,**kwargs):
        self.node=tree.nodes.new(type="GeometryNodeFillCurve")
        super().__init__(tree,location=location,**kwargs)

        self.node.mode=mode
        self.geometry_out=self.node.outputs["Mesh"]
        self.geometry_in=self.node.inputs["Curve"]

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

        if isinstance(width, (int, float)):
            self.node.inputs["Width"].default_value = width
        else:
            self.tree.links.new(width, self.node.inputs["Width"])
        if isinstance(height, (int, float)):
            self.node.inputs["Height"].default_value = height
        else:
            self.tree.links.new(height, self.node.inputs["Height"])

class StringToCurves(GreenNode):
    def __init__(self, tree, location=(0, 0),overflow='OVERFLOW',align_x="CENTER",align_y="MIDDLE",pivot_mode="MIDPOINT",
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

        if isinstance(rotation, Vector):
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
        self.node.scale_mode = scale_mode
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
            if isinstance(rotation,Vector):
                self.node.inputs["Rotation"].default_value=rotation
            else:
                self.tree.links.new(rotation,self.node.inputs["Rotation"])
        if pivot_point:
            if isinstance(pivot_point,Vector):
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
    def __init__(self, tree, location=(0, 0),data_type="FLOAT",domain="POINT",geometry=None,selection=None,attribute=None,std_out="Mean",**kwargs):
        self.node = tree.nodes.new(type="GeometryNodeAttributeStatistic")
        super().__init__(tree,location=location,**kwargs)

        self.node.data_type=data_type
        self.node.domain=domain

        self.std_out = self.node.outputs[std_out]
        if geometry:
            self.geometry_in = self.node.inputs["Geometry"]
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

        if isinstance(rotation, (list, Vector)):
            self.inputs["Rotation"].default_value = rotation
        else:
            self.tree.links.new(rotation, self.inputs["Rotation"])

        if isinstance(scale, (list, Vector)):
            self.inputs["Scale"].default_value = scale
        else:
            self.tree.links.new(scale, self.inputs["Scale"])

class ObjectInfo(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 transform_space="RELATIVE",
                 object=None, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeObjectInfo")
        super().__init__(tree, location=location, **kwargs)
        self.node.transform_space = transform_space
        if object is not None:
            self.node.inputs["Object"].default_value = object

        self.geometry_out = self.node.outputs["Geometry"]

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

class SceneTime(RedNode):
    def __init__(self, tree, location=(0, 0), std_out="Seconds", **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputSceneTime")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[std_out]

# Function Nodes #
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
            if not isinstance(x, bpy.types.NodeSocketFloat):
                x = x.std_out
            tree.links.new(x, self.node.inputs["X"])

        if isinstance(y, (int, float)):
            self.node.inputs["Y"].default_value = y
        else:
            if not isinstance(y, bpy.types.NodeSocketFloat):
                y = y.std_out
            tree.links.new(y, self.node.inputs["Y"])

        if isinstance(z, (int, float)):
            self.node.inputs["Z"].default_value = z
        else:
            if not isinstance(z,bpy.types.NodeSocketFloat):
                z = z.std_out
            tree.links.new(z, self.node.inputs["Z"])

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
            if isinstance(inputs0,(list,Vector)):
                self.node.inputs[8].default_value=inputs0
            else:
                tree.links.new(inputs0,self.node.inputs[8])
            if isinstance(inputs1,(list,Vector)):
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
                self.node.inputs[3].default_value = float_input
            else:
                tree.links.new(float_input, self.node.inputs[3])

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
            tree.links.new(true, self.true)

        if false:
            tree.links.new(false, self.false)

class IndexSwitch(BlueNode):
    def __init__(self, tree, location=(0, 0), data_type="GEOMETRY",
                 index=None, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodeIndexSwitch")
        self.node.data_type=data_type
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs["Output"]
        self.index = self.node.inputs["Index"]
        self.slots = self.node.inputs

        if index:
            if isinstance(index, (int, float)):
                self.node.inputs["Index"].default_value = index
            else:
                tree.links.new(index, self.index)

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
        tree.links.new(self.repeat_input.outputs["Geometry"], self.repeat_output.inputs["Geometry"])
        super().__init__(tree, location=location, **kwargs)

        if isinstance(iterations, int):
            self.repeat_input.inputs["Iterations"].default_value = iterations
        else:
            self.repeat_input.inputs["Iteration"] = iterations

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

    def create_geometry_line(self, nodes):
        last = nodes.pop()
        self.tree.links.new(last.geometry_out, self.repeat_output.inputs["Geometry"])
        while len(nodes) > 0:
            current = nodes.pop()
            self.tree.links.new(current.geometry_out, last.geometry_in)
            last = current
        self.tree.links.new(self.repeat_input.outputs["Geometry"], last.geometry_in)

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
    def __init__(self, tree, location=(0, 0), node_width=5, geometry=None, **kwargs):
        self.foreach_output = tree.nodes.new("GeometryNodeForeachGeometryElementOutput")
        self.foreach_input = tree.nodes.new("GeometryNodeForeachGeometryElementInput")
        self.foreach_input.location = (location[0] * 200, location[1] * 200)
        self.foreach_output.location = (location[0] * 200 + node_width * 200, location[1] * 100)
        self.foreach_input.pair_with_output(self.foreach_output)
        self.node = self.foreach_input
        self.index = self.foreach_input.outputs[0]
        self.geometry_in = self.foreach_input.inputs["Geometry"]
        self.geometry_out = self.foreach_output.outputs[2]
        # tree.links.new(self.foreach_input.outputs["Element"], self.foreach_output.inputs["Geometry"])
        super().__init__(tree, location=location, **kwargs)

        if geometry is not None:
            tree.links.new(geometry, self.foreach_input.inputs["Geometry"])

    def add_socket(self, socket_type="GEOMETRY", name="socket", value=0):
        """
        :param socket_type: "FLOAT", "INT", "BOOLEAN", "VECTOR", "ROTATION", "STRING", "RGBA", "OBJECT", "IMAGE", "GEOMETRY", "COLLECTION", "TEXTURE", "MATERIAL"
        :param name:
        :return:
        """
        self.foreach_output.state_items.new(socket_type, name)
        self.foreach_input.outputs[name].default_value = value

    def create_geometry_line(self, nodes):
        last = nodes.pop()
        self.tree.links.new(last.geometry_out, self.foreach_output.inputs["Geometry"])
        while len(nodes) > 0:
            current = nodes.pop()
            self.tree.links.new(current.geometry_out, last.geometry_in)
            last = current
        self.tree.links.new(self.foreach_input.outputs["Element"], last.geometry_in)

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

        if geometry:
            tree.links.new(geometry,self.node.inputs["Geometry"])
        if selection:
            tree.links.new(selection,self.node.inputs["Selection"])

    def pair_with_output(self,output):
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


# custom composite nodes

class WireFrame(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 radius=0.02,
                 resolution=4,
                 geometry=None,
                 **kwargs
                 ):

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
        curve2mesh = CurveToMesh(tree, location=(2, 0), profile_curve=curve_circle.geometry_out)
        create_geometry_line(tree, [mesh2curve, curve2mesh],
                             ins=group_inputs.outputs["Mesh"], out=group_outputs.inputs["Mesh"])
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


# aux functions #

def get_attributes(line):
    tag_label_ended=false
    leftside = True
    keys = []
    values = []
    key=""
    val=""
    for letter in line:
        if letter=='<':
            pass
        elif letter=='>':
            break # ignore anything after the end
        elif letter==' ':
            if not tag_label_ended:
                tag_label_ended=True
        else:
            if not tag_label_ended:
                pass # part of the tag label, can be ignored
            else:
                if letter=='=':
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
        attribute = attribute[1:-2]
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
        if len(color)==0:
            return None
        return ibpy.get_material(color)


def create_socket(tree, node, node_attributes, attributes):
    # print(node)
    # print(node_attributes)
    # print(attributes)
    if attributes['type']=='CUSTOM':
        # empty socket, nothing to do
        return False
    else:
        if node_attributes['type']=='GROUP_INPUT':
            tree.interface.new_socket(attributes['name'],description='',in_out="INPUT",socket_type=SOCKET_TYPES[attributes['type']])
            if "default_value" in attributes:
                default_value = get_default_value_for_socket(attributes)
                tree.interface.items_tree.get(attributes['name']).default_value=default_value
            return True
        if node_attributes['type']=='GROUP_OUTPUT':
            tree.interface.new_socket(attributes['name'],description='',in_out="OUTPUT",socket_type=SOCKET_TYPES[attributes['type']])
            return True
        else:
            # create socket in case of Repeat zone or GroupOutput
            if "GeometryNodeRepeatOutput" in node.name:
                node.add_socket(SOCKET_TYPES[node_attributes["type"]], node_attributes["name"])
                return True
            elif "GeometryNodeRepeatInput" in node.name:
                node.add_socket(SOCKET_TYPES[node_attributes["type"]], node_attributes["name"])
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
                            if node_attributes["type"] in {'REPEAT_INPUT','FOREACH_GEOMETRY_ELEMENT_INPUT'}:
                                save_for_after_pairing[node_id]={"inputs":[],"outputs":[]}
                        elif line.startswith("<OUTPUTS>"):
                            output_count=0
                        elif line.startswith("</INPUTS>"):
                            pass
                        elif line.startswith("</OUTPUTS>"):
                            pass
                        elif line.startswith("<INPUT "):
                            input_attributes = get_attributes(line)
                            input_id = int(input_attributes["id"])
                            if node_attributes["type"] in {"REPEAT_INPUT","FOREACH_GEOMETRY_ELEMENT_INPUT"}:
                                # repeat inputs can only be initiated after pairing
                                save_for_after_pairing[node_id]["inputs"].append(input_attributes)

                            elif len(node.inputs)>input_count and node.inputs[input_count].name!="": # avoid virtual socket
                                node_structure[node_id]["inputs"][input_id] = input_count
                                node.inputs[input_count].name=input_attributes['name']
                                if node_id==14:
                                    pass
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
                                    # print("Warning: unrecognized socket in ", node_id, socket_count,input_attributes["type"])
                                    node_structure[node_id]["inputs"][input_id] = -1 # take last slot (this dynamically generates new sockets for Group Input and Group Output
                                    input_count +=1 #also increase input_count, since the custom socket can between real sockets
                            socket_count+=1
                        elif line.startswith("<OUTPUT "):
                            output_attributes = get_attributes(line)
                            output_id = int(output_attributes["id"])

                            if node_attributes["type"] in{'REPEAT_INPUT','FOREACH_GEOMETRY_ELEMENT_INPUT'}:
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
                                    # print("Warning: unrecognized socket in ",node_id,socket_count,output_attributes["type"])
                                    node_structure[node_id]["outputs"][output_id] = -1 # take last slot (this dynamically generates new sockets for Grou
                                    output_count += 1 # also increase output_count, since the custom socket can be between real sockets
                            socket_count += 1

            # establish parent relations
            for key, val in parent_dir.items():
                if node_dir[key] is not None:
                    node_dir[key].set_parent(node_dir[name_dir[val]])

            # check for zone pairing
            for key, val in node_dir.items():
                name = node_structure[key]["name"]
                # print(name)
                key=int(key)
                # the input sockets are only created after pairing with the output node
                # therefore the links can only be created after pairing
                if "ForEachGeometryElementInput" in name:
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
                if "RepeatInput" in name:
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
                input_id = node_structure[to_node]["inputs"][to_socket]

                # print("link ",node_dir[from_node],": ",str(from_socket),"->",str(to_socket),": ",node_dir[to_node])
                if output_id<len(node_dir[from_node].outputs):
                    tree.links.new(node_dir[from_node].outputs[output_id],node_dir[to_node].inputs[input_id])

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
                  name="FunctionNode", hide=True, location=(0, 0)):
    """
    this will be the optimized prototype for a flexible function generator
    functions: a dictionary that contains a key for every output. If the key is in vectors,
     either a list of three functions is required or a function with vector output
    :return:
    """
    location = (location[0] * 200, location[1] * 100)
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
