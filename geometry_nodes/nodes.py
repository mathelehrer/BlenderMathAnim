import bpy
import numpy as np
from mathutils import Vector

from grandalf.graphs import Vertex, Edge, Graph
from grandalf.layouts import SugiyamaLayout, DigcoLayout
from interface.ibpy import get_material, make_new_socket, OPERATORS
from mathematics.groups.e8 import E8Lattice

pi = np.pi

def maybe_flatten(list_of_lists):
    result = []
    for part in list_of_lists:
        if isinstance(part,list):
            result+=part
        else:
            result.append(part)
    return result

class Node:
    def __init__(self, tree, location=(0, 0), width=200, height=100, **kwargs):
        self.tree = tree
        self.location = location
        self.l, self.m = location
        self.node.location = (self.l * width, self.m * height)
        self.links = tree.links
        self.node.hide = True

        if 'hide' in kwargs:
            hide = kwargs.pop('hide')
            self.node.hide = hide
        if 'name' in kwargs:
            self.name = kwargs.pop('name')
            self.node.label = self.name
            self.node.name = self.name
        else:
            self.name="DefaultGeometryNode"
        if 'label' in kwargs:
            label = kwargs.pop('label')
            self.node.label = label


class GreenNode(Node):
    """
    Super class for "green" geometry nodes.
    They have a standard geometry input and/or a standard geometry output.
    They can be piped together in a line.
    """

    def __init__(self, tree, location=(0, 0), **kwargs):
        super().__init__(tree, location=location, **kwargs)

        # these ports have to be declared in the children
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
                 end_location=Vector([0, 0, 1]), **kwargs):

        self.node = tree.nodes.new(type="GeometryNodeMeshLine")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Mesh']
        self.node.mode = mode
        self.node.count_mode = count_mode

        if isinstance(count, int):
            self.node.inputs['Count'].default_value = count
        else:
            self.tree.links.new(count, self.node.inputs['Count'])
        if isinstance(start_location, Vector):
            self.node.inputs['Start Location'].default_value = start_location
        else:
            self.tree.links.new(start_location, self.node.inputs['Start Location'])
        if isinstance(end_location, Vector):
            self.node.inputs['Offset'].default_value = end_location
        else:
            self.tree.links.new(end_location, self.node.inputs['Offset'])


# mesh operations
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

        self.geometry_out = self.node.outputs['Mesh']
        self.geometry_in = self.node.inputs['Mesh']
        self.node.mode = mode

        if isinstance(offset_scale, (int, float)):
            self.node.inputs['Offset Scale'].default_value = offset_scale
        else:
            self.tree.links.new(offset_scale, self.node.inputs['Offset Scale'])

        if isinstance(offset, (Vector,list)):
            vector = InputVector(tree,location=(self.location[0]-1,self.location[1]-1),value=offset)
            self.tree.links.new(vector.std_out,self.node.inputs['Offset'])
        else:
            self.tree.links.new(offset, self.node.inputs['Offset'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if mesh:
            self.tree.links.new(mesh, self.node.inputs['Mesh'])


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

        self.geometry_out = self.node.outputs['Curves']
        self.geometry_in = self.node.inputs['Points']


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

        self.geometry_out = self.node.outputs['Curve']
        self.geometry_in = self.node.inputs['Mesh']

        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if mesh:
            self.tree.links.new(mesh, self.node.inputs['Mesh'])


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

        self.geometry_out = self.node.outputs['Points']
        self.geometry_in = self.node.inputs['Mesh']

        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if mesh:
            self.tree.links.new(mesh, self.node.inputs['Mesh'])
        if position:
            if isinstance(position,(Vector,list)):
                self.node.inputs['Position'].default_value=position
            else:
                self.tree.links.new(position,self.node.inputs['Position'])

# curve primitives

class CurveCircle(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 mode="RADIUS",
                 resolution=4,
                 radius=0.02, **kwargs):
        """

        :param tree:
        :param location:
        :param mode: 'RADIUS', 'POINTS'
        :param resolution:
        :param radius:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeCurvePrimitiveCircle")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Curve']

        self.node.mode = mode

        if isinstance(radius, (int, float)):
            self.node.inputs['Radius'].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs['Radius'])
        if isinstance(resolution, int):
            self.node.inputs['Resolution'].default_value = resolution
        else:
            self.tree.links.new(resolution, self.node.inputs['Resolution'])


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

        self.geometry_out = self.node.outputs['Mesh']
        self.geometry_in = self.node.inputs['Curve']

        if curve:
            self.tree.links.new(curve, self.node.inputs['Curve'])
        if profile_curve:
            self.tree.links.new(profile_curve, self.node.inputs['Profile Curve'])


class Points(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 count=1,
                 position=Vector([0, 0, 0]),
                 radius=0.1, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodePoints")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Geometry']

        if isinstance(count, int):
            self.node.inputs['Count'].default_value = count
        else:
            self.tree.links.new(count, self.node.inputs['Count'])
        if isinstance(position, Vector):
            self.node.inputs['Position'].default_value = position
        else:
            self.tree.links.new(position, self.node.inputs['Position'])
        if isinstance(radius, (int, float)):
            self.node.inputs['Radius'].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs['Radius'])


class PointsToVertices(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 points=None,
                 selection=None, **kwargs):

        self.node = tree.nodes.new(type="GeometryNodePointsToVertices")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Mesh']
        self.geometry_in = self.node.inputs['Points']
        if points:
            self.tree.links.new(points, self.node.inputs['Points'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])


class Grid(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 size_x=10,
                 size_y=10,
                 vertices_x=11,
                 vertices_y=11, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshGrid")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Mesh']

        if isinstance(size_x, (int, float)):
            self.node.inputs['Size X'].default_value = size_x
        else:
            self.tree.links.new(size_x, self.node.inputs['Size X'])
        if isinstance(size_y, (int, float)):
            self.node.inputs['Size Y'].default_value = size_y
        else:
            self.tree.links.new(size_y, self.node.inputs['Size Y'])

        if isinstance(vertices_x, int):
            self.node.inputs['Vertices X'].default_value = vertices_x
        else:
            self.tree.links.new(vertices_x, self.node.inputs['Vertices X'])
        if isinstance(vertices_y, int):
            self.node.inputs['Vertices Y'].default_value = vertices_y
        else:
            self.tree.links.new(vertices_y, self.node.inputs['Vertices Y'])


class UVSphere(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 radius=1,
                 segments=64, rings=32, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshUVSphere")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Mesh']

        if isinstance(radius, (int, float)):
            self.node.inputs['Radius'].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs['Radius'])

        if isinstance(segments, int):
            self.node.inputs['Segments'].default_value = segments
        else:
            self.tree.links.new(segments, self.node.inputs['Segments'])

        if isinstance(rings, int):
            self.node.inputs['Rings'].default_value = rings
        else:
            self.tree.links.new(rings, self.node.inputs['Rings'])


class IcoSphere(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 radius=0.1,
                 subdivisions=1, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshIcoSphere")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Mesh']

        if isinstance(radius, (int, float)):
            self.node.inputs['Radius'].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs['Radius'])
        if isinstance(subdivisions, int):
            self.node.inputs['Subdivisions'].default_value = subdivisions
        else:
            self.tree.links.new(subdivisions, self.node.inputs['Subdivisions'])


class CubeMesh(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 size=[1, 1, 1], vertices_x=2, vertices_y=2, vertices_z=2, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMeshCube")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Mesh']

        if isinstance(vertices_x, int):
            self.node.inputs['Vertices X'].default_value = vertices_x
        else:
            self.node.inputs['Vertices X'] = vertices_x
        if isinstance(vertices_y, int):
            self.node.inputs['Vertices Y'].default_value = vertices_y
        else:
            self.node.inputs['Vertices Y'] = vertices_y
        if isinstance(vertices_z, int):
            self.node.inputs['Vertices Z'].default_value = vertices_z
        else:
            self.node.inputs['Vertices Z'] = vertices_z
        if isinstance(size, (int, float)):
            self.node.inputs['Size'].default_value = [size] * 3
        elif isinstance(size, (list, Vector)):
            self.node.inputs['Size'].default_value = size
        else:
            self.tree.links.new(size, self.node.inputs['Size'])


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

        self.geometry_out = self.node.outputs['Instances']
        self.geometry_in = self.node.inputs['Points']

        if isinstance(rotation, Vector):
            self.node.inputs['Rotation'].default_value = rotation
        else:
            self.tree.links.new(rotation, self.node.inputs['Rotation'])
        if isinstance(scale, Vector):
            self.node.inputs['Scale'].default_value = scale
        else:
            self.tree.links.new(scale, self.node.inputs['Scale'])

        if points:
            self.tree.links.new(points, self.node.inputs['Points'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if instance:
            self.tree.links.new(instance, self.node.inputs['Instance'])
        if instance_index:
            self.tree.links.new(instance_index, self.node.inputs['Instance Index'])


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

        self.geometry_out = self.node.outputs['Mesh']
        self.geometry_in = mesh2curve.inputs['Mesh']


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

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        if isinstance(position, Vector):
            self.node.inputs['Position'].default_value = position
        else:
            self.tree.links.new(position, self.node.inputs['Position'])
        if isinstance(offset, Vector):
            self.node.inputs['Offset'].default_value = offset
        else:
            self.tree.links.new(offset, self.node.inputs['Offset'])

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])


class ScaleElements(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 domain='FACE',
                 scale_mode='UNIFORM',
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
        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        if isinstance(scale, (int, float)):
            self.node.inputs['Scale'].default_value = scale
        else:
            self.tree.links.new(scale, self.node.inputs['Scale'])

        if center:
            if isinstance(center, (list, Vector)):
                self.node.inputs['Center'].default_value = center
            else:
                self.tree.links.new(center, self.node.inputs['Center'])

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])


class RealizeInstances(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 ):
        self.node = tree.nodes.new(type="GeometryNodeRealizeInstances")
        super().__init__(tree, location=location)

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])


class JoinGeometry(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeJoinGeometry")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        if isinstance(geometry, list):
            for geo in geometry:
                self.tree.links.new(geo, self.node.inputs['Geometry'])
        elif geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])


class DeleteGeometry(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 domain='POINT',
                 mode='ALL',
                 geometry=None,
                 selection=None,
                 **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeDeleteGeometry")
        self.node.domain = domain
        self.node.mode = mode
        super().__init__(tree, location=location, **kwargs)

        self.geometry_in = self.node.inputs['Geometry']
        self.geometry_out = self.node.outputs['Geometry']

        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])


class RayCast(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 data_type='FLOAT',
                 mapping='INTERPOLATED',
                 target_geometry=None,
                 attribute=None,
                 source_position=None,
                 ray_direction=Vector([0, 0, -1]),
                 ray_length=100, **kwargs
                 ):
        """

        :param tree:
        :param location:
        :param data_type:'FLOAT', 'INT', 'FLOAT_VECTOR', 'FLOAT_COLOR', 'BYTE_COLOR', 'BOOLEAN', 'FLOAT2', 'QUATERNION'
        :param mapping:'INTERPOLATED','NEAREST'
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

        self.geometry_in = self.node.inputs['Target Geometry']

        if target_geometry:
            self.tree.links.new(target_geometry, self.node.inputs['Target Geometry'])
        if attribute:
            self.tree.links.new(attribute, self.node.inputs['Attribute'])
        if source_position:
            self.tree.links.new(source_position, self.node.inputs['Source Position'])
        if isinstance(ray_direction, (Vector, list)):
            self.node.inputs['Ray Direction'].default_value = ray_direction
        else:
            self.tree.links.new(ray_direction, self.node.inputs['Ray Direction'])
        if isinstance(ray_length, (int, float)):
            self.node.inputs['Ray Length'].default_value = ray_length
        else:
            self.tree.links.new(ray_length, self.node.inputs['Ray Length'])


class ConvexHull(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 ):
        self.node = tree.nodes.new(type="GeometryNodeConvexHull")
        super().__init__(tree, location=location)

        self.geometry_out = self.node.outputs['Convex Hull']
        self.geometry_in = self.node.inputs['Geometry']

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])


class BoundingBox(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None, **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeBoundBox")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Bounding Box']
        self.geometry_in = self.node.inputs['Geometry']

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])


class StoredNamedAttribute(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 data_type='FLOAT',
                 domain='POINT',
                 selection=None,
                 name='attribute',
                 value=None, **kwargs
                 ):
        """
           :param tree:
           :param location:
           :param data_type: 'FLOAT', 'INT', 'FLOAT_VECTOR', 'FLOAT_COLOR', 'BYTE_COLOR', 'BOOLEAN', 'FLOAT2', 'QUATERNION'
           :param domain: 'POINT', 'FACE', 'EDGE', and more
           :param name: name of the attribute
           :param value: value to store
           :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeStoreNamedAttribute")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        self.node.domain = domain
        self.node.data_type = data_type

        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])

        if isinstance(name, str):
            self.node.inputs['Name'].default_value = name
        else:
            self.tree.links.new(name, self.node.inputs['Name'])

        if value is not None:
            if isinstance(value, (int, float, Vector, list)):
                self.node.inputs['Value'].default_value = value
            else:
                self.tree.links.new(value, self.node.inputs['Value'])


class NamedAttribute(RedNode):
    def __init__(self, tree, location=(0, 0),
                 data_type='FLOAT',
                 name='attribute', **kwargs
                 ):
        """
           :param tree:
           :param location:
           :param data_type: 'FLOAT', 'INT', 'FLOAT_VECTOR', 'FLOAT_COLOR', 'BYTE_COLOR', 'BOOLEAN', 'FLOAT2', 'QUATERNION'
           :param name: name of the attribute
           :param kwargs:
        """
        self.node = tree.nodes.new(type="GeometryNodeInputNamedAttribute")
        super().__init__(tree, location=location, **kwargs)

        # the order of the following lines matters, since the output depends on the data_type
        self.node.data_type = data_type
        self.std_out = self.node.outputs['Attribute']

        if isinstance(name, str):
            self.node.inputs['Name'].default_value = name
        else:
            self.tree.links.new(name, self.node.inputs['Name'])


class SetMaterial(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 selection=None,
                 material='drawing', **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeSetMaterial")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if callable(material):  # call passed function
            if "attribute_names" in kwargs:
                material= material(attribute_names=kwargs.pop("attribute_names"), **kwargs)
            else:
                material = material(**kwargs)
            self.inputs['Material'].default_value = material
        elif isinstance(material, str):  # create material from passed string
            material = get_material(material, **kwargs)
            self.inputs['Material'].default_value = material
        elif isinstance(material,bpy.types.Material):
            self.inputs['Material'].default_value=material
        else:  # link socket
            self.tree.links.new(material, self.inputs['Material'])


class MergeByDistance(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 mode='ALL',
                 selection=None,
                 distance=0.001,
                 **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeMergeByDistance")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        self.node.mode = mode

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if isinstance(distance, (int, float)):
            self.node.inputs['Distance'].default_value = distance
        elif distance:
            self.tree.links.new(distance, self.node.inputs['Distance'])


class SetShadeSmooth(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 geometry=None,
                 selection=None,
                 shade_smooth=True, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeSetShadeSmooth")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        if geometry:
            self.tree.links.new(geometry, self.node.inputs['Geometry'])
        if selection:
            self.tree.links.new(selection, self.node.inputs['Selection'])
        if isinstance(shade_smooth, bool):
            self.inputs['Shade Smooth'].default_value = shade_smooth
        else:
            self.tree.links.new(shade_smooth, self.inputs['Shade Smooth'])


class TransformGeometry(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 translation=Vector(),
                 rotation=Vector(),
                 scale=Vector([1, 1, 1]), **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeTransform")
        super().__init__(tree, location=location, **kwargs)

        self.geometry_out = self.node.outputs['Geometry']
        self.geometry_in = self.node.inputs['Geometry']

        if isinstance(translation, (list, Vector)):
            self.inputs['Translation'].default_value = translation
        else:
            self.tree.links.new(translation, self.inputs['Translation'])

        if isinstance(rotation, (list, Vector)):
            self.inputs['Rotation'].default_value = rotation
        else:
            self.tree.links.new(rotation, self.inputs['Rotation'])

        if isinstance(scale, (list, Vector)):
            self.inputs['Scale'].default_value = scale
        else:
            self.tree.links.new(scale, self.inputs['Scale'])


class ObjectInfo(GreenNode):
    def __init__(self, tree, location=(0, 0),
                 transform_space='RELATIVE',
                 object=None,
                 translation=Vector(),
                 rotation=Vector(),
                 scale=Vector([1, 1, 1]), **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeObjectInfo")
        super().__init__(tree, location=location, **kwargs)
        self.node.transform_space = transform_space
        if object is not None:
            self.node.inputs['Object'].default_value = object

        self.geometry_out = self.node.outputs['Geometry']


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
                 component='MESH', **kwargs
                 ):
        self.node = tree.nodes.new(type="GeometryNodeAttributeDomainSize")
        super().__init__(tree, location=location, **kwargs)

        self.node.component = component
        if geometry is not None:
            tree.links.new(geometry, self.node.inputs['Geometry'])

        self.geometry_in = self.node.inputs['Geometry']


class SampleIndex(GreenNode):
    """
    Geometry node SampleIndex
    retrieves information about a specified value for a geometric object with a given index
    """

    def __init__(self, tree, location=(0, 0),
                 data_type='FLOAT_VECTOR',
                 domain='POINT',
                 geometry=None,
                 value=None,
                 index=None, **kwargs
                 ):

        self.node = tree.nodes.new(type="GeometryNodeSampleIndex")
        super().__init__(tree, location=location, **kwargs)

        self.node.data_type = data_type
        self.node.domain = domain

        if geometry is not None:
            tree.links.new(geometry, self.node.inputs['Geometry'])

        if value is not None:
            tree.links.new(value, self.node.inputs['Value'])

        if isinstance(index, int):
            self.node.inputs['Index'].default_value = index
        elif index is not None:
            tree.links.new(index, self.node.inputs['Index'])

        self.geometry_in = self.node.inputs['Geometry']
        self.std_out = self.node.outputs['Value']


#  red nodes   #

class Position(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputPosition")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs['Position']


class Index(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputIndex")
        super().__init__(tree, location=location, **kwargs)
        self.std_out = self.node.outputs['Index']


class EdgeVertices(RedNode):
    def __init__(self, tree, location=(0, 0), **kwargs):
        self.node = tree.nodes.new(type="GeometryNodeInputMeshEdgeVertices")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs['Vertex Index 1']


class InputValue(RedNode):
    def __init__(self, tree, location=(0, 0), value=0, **kwargs):
        self.node = tree.nodes.new(type="ShaderNodeValue")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs['Value']
        self.outputs['Value'].default_value = value


# Function Nodes #
class InputBoolean(RedNode):
    def __init__(self, tree, location=(0, 0), value=True, **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInputBool")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs['Boolean']
        self.node.boolean = value


class BooleanMath(RedNode):
    def __init__(self, tree, location=(0, 0), operation='AND', inputs0=True, inputs1=True, **kwargs):
        """
        :param tree:
        :param location:
        :param operation: 'AND','OR',...
        :param inputs0:
        :param inputs1:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="FunctionNodeBooleanMath")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs['Boolean']

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


class VectorMath(RedNode):
    def __init__(self, tree, location=(0, 0), operation='ADD', inputs0=Vector(), inputs1=Vector(), float_input=None,
                 **kwargs):
        """
        :param tree:
        :param location:
        :param operation: 'AND','OR',...
        :param inputs0:
        :param inputs1:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="ShaderNodeVectorMath")
        super().__init__(tree, location=location, **kwargs)

        if operation in ("DOT", "LENGTH"):
            self.std_out = self.node.outputs[1]
        else:
            self.std_out = self.node.outputs['Vector']

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


class InputVector(RedNode):
    def __init__(self, tree, location=(0, 0), value=Vector()
                 , **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeInputVector")
        super().__init__(tree, location=location, **kwargs)

        self.std_out = self.node.outputs[0]
        self.node.vector = value


# blue nodes
class RandomValue(BlueNode):
    def __init__(self, tree, data_type='FLOAT_VECTOR', location=(0, 0), min=-1 * Vector([1, 1, 1]),
                 max=Vector([1, 1, 1]), seed=0, **kwargs):
        """

        :param tree:
        :param data_type: 'FLOAT', 'INT', 'FLOAT_VECTOR', 'BOOLEAN'
        :param location:
        :param min:
        :param max:
        :param seed:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="FunctionNodeRandomValue")
        super().__init__(tree, location=location, **kwargs)

        self.node.data_type = data_type

        if data_type == 'FLOAT_VECTOR':
            self.std_out = self.node.outputs[0]
        else:
            self.std_out == self.node.outputs[1]

        if isinstance(min, (int, float, list, Vector)):
            self.node.inputs['Min'].default_value = min
        else:
            self.node.inputs['Min'] = min

        if isinstance(max, (int, float, list, Vector)):
            self.node.inputs['Max'].default_value = max
        else:
            self.node.inputs['Max'] = max

        if isinstance(seed, (int, float)):
            self.node.inputs['Seed'].default_value = seed
        else:
            self.node.inputs['Seed'] = seed


class SeparateXYZ(BlueNode):
    def __init__(self, tree, location=(0, 0), vector=Vector(), **kwargs):
        """

        :param tree:
        :param data_type: 'FLOAT', 'INT', 'FLOAT_VECTOR', 'BOOLEAN'
        :param location:
        :param min:
        :param max:
        :param seed:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="ShaderNodeSeparateXYZ")
        super().__init__(tree, location=location, **kwargs)

        self.std_in = self.node.inputs['Vector']
        self.x = self.node.outputs['X']
        self.y = self.node.outputs['Y']
        self.z = self.node.outputs['Z']

        if isinstance(vector, (int, float, list, Vector)):
            self.std_in.default_value = vector
        else:
            tree.links.new(vector, self.std_in)


class CombineXYZ(BlueNode):
    def __init__(self, tree, location=(0, 0), x=0, y=0, z=0, **kwargs):
        """

        :param tree:
        :param data_type: 'FLOAT', 'INT', 'FLOAT_VECTOR', 'BOOLEAN'
        :param location:
        :param min:
        :param max:
        :param seed:
        :param kwargs:
        """
        self.node = tree.nodes.new(type="ShaderNodeCombineXYZ")
        super().__init__(tree, location=location, **kwargs)

        self.std_in = self.node.inputs
        self.std_out = self.node.outputs['Vector']

        if isinstance(x, (int, float)):
            self.node.inputs['X'].default_value = x
        else:
            tree.links.new(x, self.node.inputs['X'])

        if isinstance(y, (int, float)):
            self.node.inputs['Y'].default_value = y
        else:
            tree.links.new(y, self.node.inputs['Y'])

        if isinstance(z, (int, float)):
            self.node.inputs['Z'].default_value = z
        else:
            tree.links.new(z, self.node.inputs['Z'])


# zones

class RepeatZone(GreenNode):
    def __init__(self, tree, location=(0, 0), width=5, iterations=10, geometry=None, **kwargs):
        self.repeat_output = tree.nodes.new("GeometryNodeRepeatOutput")
        self.repeat_input = tree.nodes.new("GeometryNodeRepeatInput")
        self.repeat_input.location = (location[0] * 200, location[1] * 200)
        self.repeat_output.location = (location[0] * 200 + width * 200, location[1] * 200)
        self.repeat_input.pair_with_output(self.repeat_output)
        self.node = self.repeat_input
        self.geometry_in = self.repeat_input.inputs['Geometry']
        self.geometry_out = self.repeat_output.outputs['Geometry']
        tree.links.new(self.repeat_input.outputs['Geometry'], self.repeat_output.inputs['Geometry'])
        super().__init__(tree, location=location, **kwargs)

        if isinstance(iterations, int):
            self.repeat_input.inputs['Iterations'].default_value = iterations
        else:
            self.repeat_input.inputs['Iteration'] = iterations

        if geometry is not None:
            tree.links.new(geometry, self.repeat_input.inputs['Geometry'])

    def add_socket(self, socket_type='GEOMETRY', name="socket"):
        """
        :param socket_type: 'FLOAT', 'INT', 'BOOLEAN', 'VECTOR', 'ROTATION', 'STRING', 'RGBA', 'OBJECT', 'IMAGE', 'GEOMETRY', 'COLLECTION', 'TEXTURE', 'MATERIAL'
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
        self.tree.links.new(last.geometry_out, self.repeat_output.inputs['Geometry'])
        while len(nodes) > 0:
            current = nodes.pop()
            self.tree.links.new(current.geometry_out, last.geometry_in)
            last = current
        self.tree.links.new(self.repeat_input.outputs['Geometry'], last.geometry_in)


class Simulation(GreenNode):
    def __init__(self, tree, location=(0, 0), width=5, geometry=None, **kwargs):
        self.simulation_output = tree.nodes.new("GeometryNodeSimulationOutput")
        self.simulation_input = tree.nodes.new("GeometryNodeSimulationInput")
        self.simulation_input.location = (location[0] * 200, location[1] * 200)
        self.simulation_output.location = (location[0] * 200 + width * 200, location[1] * 200)
        self.simulation_input.pair_with_output(self.simulation_output)
        self.node = self.simulation_input
        self.geometry_in = self.simulation_input.inputs['Geometry']
        self.geometry_out = self.simulation_output.outputs['Geometry']
        tree.links.new(self.simulation_input.outputs['Geometry'], self.simulation_output.inputs['Geometry'])
        super().__init__(tree, location=location, **kwargs)

        if geometry is not None:
            tree.links.new(geometry, self.simulation_input.inputs['Geometry'])

    def add_socket(self, socket_type='GEOMETRY', name="socket"):
        """
        :param socket_type: 'FLOAT', 'INT', 'BOOLEAN', 'VECTOR', 'ROTATION', 'STRING', 'RGBA', 'OBJECT', 'IMAGE', 'GEOMETRY', 'COLLECTION', 'TEXTURE', 'MATERIAL'
        :param name:
        :return:
        """
        self.simulation_output.state_items.new(socket_type, name)

    def join_in_geometries(self, out_socket_name=None):
        join = JoinGeometry(self.tree, geometry=self.simulation_input.outputs[0:-1])
        if out_socket_name:
            self.tree.links.new(join.geometry_out, self.simulation_output.inputs[out_socket_name])

    def create_geometry_line(self, nodes):
        last = nodes.pop()
        self.tree.links.new(last.geometry_out, self.simulation_output.inputs['Geometry'])
        while len(nodes) > 0:
            current = nodes.pop()
            self.tree.links.new(current.geometry_out, last.geometry_in)
            last = current
        self.tree.links.new(self.simulation_input.outputs['Geometry'], last.geometry_in)


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

        self.geometry_out = self.node.outputs['Mesh']
        self.geometry_in = self.node.inputs['Mesh']

        if geometry:
            self.tree.links.new(geometry, self.geometry_in)
        if isinstance(radius, (int, float)):
            self.node.inputs['Radius'].default_value = radius
        else:
            self.tree.links.new(radius, self.node.inputs['Radius'])
        if isinstance(resolution, int):
            self.node.inputs['Resolution'].default_value = resolution
        else:
            self.tree.links.new(resolution, self.node.inputs['Resolution'])

    def create_node(self, nodes, name='WireframeNode'):
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new('NodeGroupInput')
        group_outputs = tree_nodes.new('NodeGroupOutput')

        make_new_socket(tree, name='Mesh', io='INPUT', type='NodeSocketGeometry')
        make_new_socket(tree, name='Radius', io='INPUT', type='NodeSocketFloat')
        make_new_socket(tree, name='Resolution', io='INPUT', type='NodeSocketInt')

        make_new_socket(tree, name='Mesh', io='OUTPUT', type='NodeSocketGeometry')

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        mesh2curve = MeshToCurve(tree, location=(1, 0))
        curve_circle = CurveCircle(tree, location=(1, 1), resolution=group_inputs.outputs['Resolution'],
                                   radius=group_inputs.outputs['Radius'])
        curve2mesh = CurveToMesh(tree, location=(2, 0), profile_curve=curve_circle.geometry_out)
        create_geometry_line(tree, [mesh2curve, curve2mesh],
                             ins=group_inputs.outputs['Mesh'], out=group_outputs.inputs['Mesh'])
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

        self.geometry_in = self.node.inputs['Target Geometry']
        self.std_out = self.node.outputs['Is Inside']

        if target_geometry:
            self.tree.links.new(target_geometry, self.node.inputs['Target Geometry'])
        if isinstance(source_position, (Vector, list)):
            self.node.inputs['Source Position'].default_value = source_position
        else:
            self.tree.links.new(source_position, self.node.inputs['Source Position'])
        if isinstance(self.ray_direction, (Vector, list)):
            self.node.inputs['Ray Direction'].default_value = self.ray_direction
        else:
            self.tree.links.new(ray_direction, self.node.inputs['Ray Direction'])

    def create_node(self, nodes, name='InsideConvexHullTest'):
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new('NodeGroupInput')
        group_outputs = tree_nodes.new('NodeGroupOutput')

        make_new_socket(tree, name='Target Geometry', io='INPUT', type='NodeSocketGeometry')
        make_new_socket(tree, name='Source Position', io='INPUT', type='NodeSocketVector')
        make_new_socket(tree, name='Ray Direction', io='INPUT', type='NodeSocketVector')

        make_new_socket(tree, name='Is Inside', io='OUTPUT', type='NodeSocketBool')
        make_new_socket(tree, name='Is Outside', io='OUTPUT', type='NodeSocketBool')

        group_inputs.location = (0, 0)
        group_outputs.location = (400, 0)

        ray_cast_up = RayCast(tree, location=(2, 2),
                              target_geometry=group_inputs.outputs['Target Geometry'],
                              source_position=group_inputs.outputs['Source Position'],
                              ray_direction=group_inputs.outputs['Ray Direction'], label="RayUp")

        scale = VectorMath(tree, location=(1, -2), label="Negative", operation='SCALE',
                           inputs0=group_inputs.outputs['Ray Direction'], float_input=-1, hide=True)
        ray_direction = scale.std_out
        ray_cast_down = RayCast(tree, location=(2, -2),
                                target_geometry=group_inputs.outputs['Target Geometry'],
                                source_position=group_inputs.outputs['Source Position'],
                                ray_direction=scale.std_out, label="RayDown")

        andMath = BooleanMath(tree, location=(3, 0.5), label="And", operation='AND',
                              inputs0=ray_cast_up.outputs['Is Hit'],
                              inputs1=ray_cast_down.outputs['Is Hit'],
                              hide=True
                              )

        notAndMath = BooleanMath(tree, location=(3, -0.5), label="NotAnd", operation='NAND',
                                 inputs0=ray_cast_up.outputs['Is Hit'],
                                 inputs1=ray_cast_down.outputs['Is Hit'],
                                 hide=True
                                 )

        tree_links.new(andMath.std_out, group_outputs.inputs['Is Inside'])
        tree_links.new(notAndMath.std_out, group_outputs.inputs['Is Outside'])
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

        self.geometry_in = self.node.inputs['Target Geometry']
        self.std_out = self.node.outputs['Is Inside']

        if target_geometry:
            self.tree.links.new(target_geometry, self.node.inputs['Target Geometry'])
        if isinstance(source_position, (Vector, list)):
            self.node.inputs['Source Position'].default_value = source_position
        else:
            self.tree.links.new(source_position, self.node.inputs['Source Position'])

    def create_node(self, nodes, name='InsideConvexHullTest'):
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new('NodeGroupInput')
        group_outputs = tree_nodes.new('NodeGroupOutput')

        make_new_socket(tree, name='Target Geometry', io='INPUT', type='NodeSocketGeometry')
        make_new_socket(tree, name='Source Position', io='INPUT', type='NodeSocketVector')

        make_new_socket(tree, name='Is Inside', io='OUTPUT', type='NodeSocketBool')
        make_new_socket(tree, name='Is Outside', io='OUTPUT', type='NodeSocketBool')

        group_inputs.location = (0, 0)
        group_outputs.location = (400, 0)

        boundary_box = BoundingBox(tree, location=(2, -2),
                                   geometry=group_inputs.outputs['Target Geometry'], label="BBox")

        comparison = make_function(tree, functions={
            "in": ['src_z,maxx_z,>,not,src_z,minn_z,>,*'],
            "out": ['src_z,maxx_z,>,src_z,minn_z,>,not,+']
        }, inputs=['src', 'maxx', 'minn'], outputs=['in', 'out'], scalars=['in', 'out'],
                                   vectors=['src', 'minn', 'maxx'], name="Comparsion")
        tree.links.new(group_inputs.outputs["Source Position"], comparison.inputs["src"])
        tree.links.new(boundary_box.outputs["Min"], comparison.inputs["minn"])
        tree.links.new(boundary_box.outputs["Max"], comparison.inputs["maxx"])

        tree_links.new(comparison.outputs["in"], group_outputs.inputs['Is Inside'])
        tree_links.new(comparison.outputs["out"], group_outputs.inputs['Is Outside'])
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

        self.geometry_out = self.node.outputs['Geometry']

    def create_node(self, nodes, name='E8Geometry'):
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_outputs = tree_nodes.new('NodeGroupOutput')
        make_new_socket(tree, name='Geometry', io='OUTPUT', type='NodeSocketGeometry')

        group_outputs.location = (400, 0)

        join = JoinGeometry(tree)
        tree_links.new(join.outputs["Geometry"], group_outputs.inputs['Geometry'])

        # the 240 eight-dimensional coordinates are hard-coded into the node
        # create a point and a set of attributes for each root of the E8 lattice
        print("Hard-coded entry of roots ...", end='')
        for root in E8Lattice().roots:
            point = Points(tree)
            attr1 = StoredNamedAttribute(tree, data_type='FLOAT_VECTOR', name="comp123", value=list(root[0:3]))
            attr2 = StoredNamedAttribute(tree, data_type='FLOAT_VECTOR', name="comp456", value=list(root[3:6]))
            attr3 = StoredNamedAttribute(tree, data_type='FLOAT_VECTOR', name="comp78", value=list(root[6:8]) + [0])

            create_geometry_line(tree, [point, attr1, attr2, attr3, join])
        print("done")
        print("Layout of the node ...", end='')
        layout(tree)
        print("done")
        return group


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
            self.node.inputs['Dimension'].default_value = dimension
        else:
            self.tree.links.new(dimension, self.node.inputs['Dimension'])

        if isinstance(angle, (int, float)):
            self.node.inputs['Angle'].default_value = angle
        else:
            self.tree.links.new(angle, self.node.inputs['Angle'])

        if isinstance(u, int):
            self.node.inputs['U'].default_value = u
        else:
            self.tree.links.new(u, self.node.inputs['U'])
        if isinstance(v, int):
            self.node.inputs['V'].default_value = v
        else:
            self.tree.links.new(v, self.node.inputs['V'])

    def create_node(self, nodes, dimension, u, v, name='RotationMatrix'):
        d = dimension
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new('NodeGroupInput')
        group_outputs = tree_nodes.new('NodeGroupOutput')

        make_new_socket(tree, name='Dimension', io='INPUT', type='NodeSocketInt')
        make_new_socket(tree, name='U', io='INPUT', type='NodeSocketInt')
        make_new_socket(tree, name='V', io='INPUT', type='NodeSocketInt')
        make_new_socket(tree, name='Angle', io='INPUT', type='NodeSocketFloat')

        # create the required number of vectors for each column
        outputs = []
        for i in range(d):
            for j in range(0, int(np.ceil(d / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io='OUTPUT', type='NodeSocketVector')
                outputs.append(name)

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)

        # create function dictionary
        func_dict = {}
        for row in range(d):
            # treat diagonal entries, equal to 1, if u and v are different then the column index
            comps = ['0'] * int(np.ceil(d / 3) * 3)
            for col in range(d):
                if col == row:
                    comps[row] = '1,u,' + str(row) + ',-,abs,0,>,v,' + str(row) + ',-,abs,0,>,*,*,u,' + str(
                        row) + ',=,v,' + str(row) + ',=,+,theta,cos,*,+'
                else:
                    if self.orientation == 1:
                        comps[col] = 'u,' + str(row) + ',=,v,' + str(col) + ',=,*,theta,sin,*,-1,*,u,' + str(
                            col) + ",=,v," + str(row) + ",=,*,theta,sin,*,+"
                    else:
                        comps[col] = 'u,' + str(row) + ',=,v,' + str(col) + ',=,*,theta,sin,*,u,' + str(
                            col) + ",=,v," + str(row) + ",=,*,theta,sin,*,-1,*,+"
            for n, p in enumerate(range(0, d, 3)):
                part = comps[p:p + 3]
                func_dict["row_" + str(row) + "_" + str(n)] = part

        rot_mat = make_function(tree_nodes, functions=func_dict, name="RotationMatrix", inputs=['u', 'v', 'theta'],
                                outputs=outputs, scalars=['u', 'v', 'theta'], vectors=outputs)
        rot_mat.location = (200, 0)
        tree_links.new(group_inputs.outputs['U'], rot_mat.inputs['u'])
        tree_links.new(group_inputs.outputs['V'], rot_mat.inputs['v'])
        tree_links.new(group_inputs.outputs['Angle'], rot_mat.inputs['theta'])

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
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_outputs = tree_nodes.new('NodeGroupOutput')

        # create the required number of vectors for each column
        outputs = []
        for i in range(rows):
            for j in range(0, int(np.ceil(cols / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io='OUTPUT', type='NodeSocketVector')
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
            self.node.inputs['Dimension'].default_value = dimension
        else:
            self.tree.links.new(dimension, self.node.inputs['Dimension'])

    def create_node(self, nodes, dimension, name='TransposeMatrix'):
        d = dimension
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new('NodeGroupInput')
        group_outputs = tree_nodes.new('NodeGroupOutput')

        make_new_socket(tree, name='Dimension', io='INPUT', type='NodeSocketInt')

        # create the required number of vectors for each column
        outputs = []
        for i in range(d):
            for j in range(0, int(np.ceil(d / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io='INPUT', type='NodeSocketVector')
                make_new_socket(tree, name=name, io='OUTPUT', type='NodeSocketVector')
                outputs.append(name)

        group_inputs.location = (0, 0)
        group_outputs.location = (600, 0)
        dict = {0: 'x', 1: 'y', 2: 'z'}
        # create function dictionary
        func_dict = {}
        for row in range(d):
            # treat diagonal entries, equal to 1, if u and v are different then the column index
            comps = ['0'] * int(np.ceil(d / 3) * 3)
            for col in range(d):
                part = row // 3
                comp = dict[row % 3]
                comps[col] = 'row_' + str(col) + '_' + str(part) + '_' + comp
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
            self.node.inputs['Dimension'].default_value = dimension
        else:
            self.tree.links.new(dimension, self.node.inputs['Dimension'])

    def create_node(self, nodes, dimension, name='LinearMap'):
        d = dimension
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new('NodeGroupInput')
        group_outputs = tree_nodes.new('NodeGroupOutput')

        make_new_socket(tree, name='Dimension', io='INPUT', type='NodeSocketInt')

        # create the required number of vectors for each column
        mat_inputs = []
        for i in range(d):
            for j in range(0, int(np.ceil(d / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io='INPUT', type='NodeSocketVector')
                mat_inputs.append(name)
        vec_inputs = []
        for i in range(int(np.ceil(d / 3))):
            name = "v_" + str(i)
            make_new_socket(tree, name=name, io='INPUT', type='NodeSocketVector')
            vec_inputs.append(name)

        for i in range(int(np.ceil(d / 3))):
            make_new_socket(tree, name="v_" + str(i), io='OUTPUT', type='NodeSocketVector')

        components = ['0'] * int(np.ceil(d / 3) * 3)
        comp_dict = {0: '_x', 1: '_y', 2: '_z'}
        for c in range(dimension):
            prod = ""
            first = True
            for i, v in enumerate(vec_inputs):
                if first:
                    prod = 'row_' + str(c) + '_' + str(i) + ',v_' + str(i) + ',dot'
                    first = False
                else:
                    prod += ',row_' + str(c) + '_' + str(i) + ',v_' + str(i) + ',dot'
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

    def create_node(self, nodes, in_dimension, out_dimension, name='ProjectionMap'):
        idim = in_dimension
        odim = out_dimension
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

        group.name = name
        group.node_tree = tree

        # create inputs and outputs
        tree_nodes = tree.nodes
        tree_links = tree.links

        group_inputs = tree_nodes.new('NodeGroupInput')
        group_outputs = tree_nodes.new('NodeGroupOutput')

        # create the required number of vectors for each column
        mat_inputs = []
        for i in range(odim):
            for j in range(0, int(np.ceil(idim / 3))):
                name = "row_" + str(i) + "_" + str(j)
                make_new_socket(tree, name=name, io='INPUT', type='NodeSocketVector')
                mat_inputs.append(name)
        vec_inputs = []
        for i in range(int(np.ceil(idim / 3))):
            name = "vi_" + str(i)
            make_new_socket(tree, name=name, io='INPUT', type='NodeSocketVector')
            vec_inputs.append(name)

        vec_outputs = []
        for i in range(int(np.ceil(odim / 3))):
            name = "vo_" + str(i)
            make_new_socket(tree, name=name, io='OUTPUT', type='NodeSocketVector')
            vec_outputs.append(name)

        components = ['0'] * int(np.ceil(idim / 3) * 3)
        for c in range(odim):
            prod = ""
            first = True
            for i, v in enumerate(vec_inputs):
                if first:
                    prod = 'row_' + str(c) + '_' + str(i) + ',vi_' + str(i) + ',dot'
                    first = False
                else:
                    prod += ',row_' + str(c) + '_' + str(i) + ',vi_' + str(i) + ',dot'
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

def create_geometry_line(tree, green_nodes, out=None, ins=None):
    first = True
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


def make_function(nodes_or_tree, functions={}, inputs=[], outputs=[], vectors=[], scalars=[],
                  node_group_type='GeometryNodes',
                  name='FunctionNode', hide=False, location=(0, 0)):
    """
    this will be the optimized prototype for a flexible function generator
    functions: a dictionary that contains a key for every output. If the key is in vectors,
     either a list of three functions is required or a function with vector output
    :return:
    """
    location = (location[0] * 200, location[1] * 200)
    if hasattr(nodes_or_tree, 'nodes'):
        tree = nodes_or_tree
        nodes = tree.nodes
    else:
        nodes = nodes_or_tree

    if 'Shader' in node_group_type:
        tree = bpy.data.node_groups.new(type='ShaderNodeTree', name=name)
        group = nodes.new(type='ShaderNodeGroup')
    else:
        tree = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
        group = nodes.new(type='GeometryNodeGroup')

    group.name = name
    group.node_tree = tree

    tree_nodes = tree.nodes
    tree_links = tree.links

    # create inputs and outputs
    group_inputs = tree_nodes.new('NodeGroupInput')
    group_outputs = tree_nodes.new('NodeGroupOutput')

    for ins in inputs:
        if ins in vectors:
            make_new_socket(tree, name=ins, io='INPUT', type='NodeSocketVector')
        if ins in scalars:
            make_new_socket(tree, name=ins, io='INPUT', type='NodeSocketFloat')

    for outs in outputs:
        if outs in vectors:
            make_new_socket(tree, name=outs, io='OUTPUT', type='NodeSocketVector')
        if outs in scalars:
            make_new_socket(tree, name=outs, io='OUTPUT', type='NodeSocketFloat')

    # create stack structure from function structure
    stacks = {}
    for key, value in functions.items():
        if isinstance(value, list):
            stacks[key] = []
            for v in value:
                stacks[key].append(v.split(','))
        else:
            stacks[key] = value.split(',')

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
        elif key in vectors:
            if isinstance(functions[key], list):
                comb = tree_nodes.new(type='ShaderNodeCombineXYZ')
                comb.name = key + "Merge"
                comb.label = key + "Merge"
                comb.location = (right, combine_counter * width / 2)
                comb.hide = True
                combine_counter += 1
                out_channels[key + '_x'] = comb.inputs[0]
                out_channels[key + '_y'] = comb.inputs[1]
                out_channels[key + '_z'] = comb.inputs[2]
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
        if ins in vectors:
            in_channels[ins] = group_inputs.outputs[ins]
            if ins + '_x' in all_terms or ins + '_y' in all_terms or ins + '_z' in all_terms:
                sep = tree_nodes.new(type='ShaderNodeSeparateXYZ')
                sep.name = ins + "Split"
                sep.label = ins + "Split"
                tree_links.new(group_inputs.outputs[ins], sep.inputs['Vector'])
                sep.location = (left + width, separate_counter * width / 2)
                sep.hide = True
                in_channels[ins + '_x'] = sep.outputs[0]
                in_channels[ins + '_y'] = sep.outputs[1]
                in_channels[ins + '_z'] = sep.outputs[2]
                separate_counter += 1

    # now the functions are constructed
    fcn_count = 0  # function index to get a separation in the node editor
    comps = ['x', 'y', 'z']
    for key, value in functions.items():
        if isinstance(value, list):
            if len(value) == 1:
                build_function(tree, stacks[key][0], scalars=scalars, vectors=vectors, in_channels=in_channels,
                               out=out_channels[key], fcn_count=fcn_count)
                fcn_count += 1
            else:
                for i, part in enumerate(value):
                    build_function(tree, stacks[key][i], scalars=scalars, vectors=vectors, in_channels=in_channels,
                                   out=out_channels[key + '_' + comps[i]], fcn_count=fcn_count)
                    fcn_count += 1
        else:
            build_function(tree, stacks[key], scalars=scalars, vectors=vectors, in_channels=in_channels,
                           out=out_channels[key], fcn_count=fcn_count)
            fcn_count += 1

    layout(tree)
    if hide:
        group.hide = True

    group.location = location
    return group


def build_function(tree, stack, scalars=[], vectors=[], in_channels={}, fcn_count=0, out=None, unary=None,
                   last_operator=None,
                   last_structure=None,
                   length=1, height=0, level=[0]):
    """
    recursive build of a node-group function

    there is a subtlety with VectorMath nodes, they always carry two outputs.
    The first one is 'Vector' and the second one is 'Value'
    there is more work to be done, to do this correctly,
    so far there is only a workaround to incorporate the 'LENGTH' operation, which yields a scalar output

    :param tree: the container for the function
    :param stack: contains the computation in reverse polish notation
    :param scalars: scalar variables
    :param vectors: vector variables
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

    # recursively build a tree structure with left and right sub-tree. For unary operators only the left tree is used

    fcn_spacing = 500

    left_empty = True
    if unary:
        right_empty = False  # no need for a right sub-tree
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
            if next_element == '*':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MULTIPLY'
                new_node_math.label = '*'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'mul':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'MULTIPLY'
                new_node_math.label = '*'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == '%':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MODULO'
                new_node_math.label = '%'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'mod':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'MODULO'
                new_node_math.label = '%'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == '/':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'DIVIDE'
                new_node_math.label = '/'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'div':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'DIVIDE'
                new_node_math.label = '/'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == '+':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ADD'
                new_node_math.label = '+'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'add':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'ADD'
                new_node_math.label = '+'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == 'sub':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'SUBTRACT'
                new_node_math.label = '-'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == '-':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SUBTRACT'
                new_node_math.label = '-'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == '**':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'POWER'
                new_node_math.label = '**'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == '<':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'LESS_THAN'
                new_node_math.label = '<'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == '>':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'GREATER_THAN'
                new_node_math.label = '>'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == '=':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'COMPARE'
                new_node_math.label = '=='
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_math.inputs[2].default_value = 0
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'min':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MINIMUM'
                new_node_math.label = 'min'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'max':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'MAXIMUM'
                new_node_math.label = 'max'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'sin':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SINE'
                new_node_math.label = 'sin'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'lg':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'LOGARITHM'
                new_node_math.label = 'lg'
                new_node_math.inputs[1].default_value=10
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'asin':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ARCSINE'
                new_node_math.label = 'asin'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'cos':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'COSINE'
                new_node_math.label = 'cos'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'acos':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ARCCOSINE'
                new_node_math.label = 'acos'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'tan':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'TANGENT'
                new_node_math.label = 'tan'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'atan2':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ARCTAN2'
                new_node_math.label = 'atan2'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = False
            elif next_element == 'abs':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ABSOLUTE'
                new_node_math.label = 'abs'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'round':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'ROUND'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'floor':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'FLOOR'
                new_node_math.label = 'floor'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'vfloor':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'FLOOR'
                new_node_math.label = 'floor'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Vector']
                unary = True
            elif next_element == 'ceil':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'CEIL'
                new_node_math.label = 'ceil'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'length':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'LENGTH'
                new_node_math.label = 'len'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'sqrt':
                new_node_math = tree.nodes.new(type='ShaderNodeMath')
                new_node_math.operation = 'SQRT'
                new_node_math.label = 'sqrt'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Value']
                unary = True
            elif next_element == 'scale':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'SCALE'
                new_node_math.label = 'scale'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs['Scale']
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == 'cross':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'CROSS_PRODUCT'
                new_node_math.label = 'x'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == 'dot':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'DOT_PRODUCT'
                new_node_math.label = '*'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Value']
            elif next_element == 'normalize':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorMath')
                new_node_math.operation = 'NORMALIZE'
                new_node_math.label = 'norm'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Vector']
                unary = True
            elif next_element == 'rot':
                new_node_math = tree.nodes.new(type='ShaderNodeVectorRotate')
                new_node_math.rotation_type = 'EULER_XYZ'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs['Vector']
                new_node_structure.right = new_node_math.inputs['Rotation']
                new_node_structure.out = new_node_math.outputs['Vector']
            elif next_element == 'axis_rot':
                new_node_math = tree.nodes.new(type='FunctionNodeAxisAngleToRotation')
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs['Axis']
                new_node_structure.right = new_node_math.inputs['Angle']
                new_node_structure.out = new_node_math.outputs['Rotation']
            elif next_element == 'rot2euler':
                new_node_math = tree.nodes.new(type='FunctionNodeRotationToEuler')
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs['Rotation']
                new_node_structure.out = new_node_math.outputs['Euler']
                unary = True
            elif next_element == 'axis_angle_euler':
                """ convenient combination of axis_rot and rot2euler"""
                new_node_math = tree.nodes.new(type='FunctionNodeRotateEuler')
                new_node_math.type = 'AXIS_ANGLE'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs['Axis']
                new_node_structure.right = new_node_math.inputs['Angle']
                new_node_structure.out = new_node_math.outputs['Rotation']
            elif next_element == 'not':
                new_node_math = tree.nodes.new(type='FunctionNodeBooleanMath')
                new_node_math.operation = 'NOT'
                new_node_math.label = 'NOT'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.out = new_node_math.outputs['Boolean']
                unary = True
            elif next_element == 'and':
                new_node_math = tree.nodes.new(type='FunctionNodeBooleanMath')
                new_node_math.operation = 'AND'
                new_node_math.label = 'AND'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Boolean']
            elif next_element == 'or':
                new_node_math = tree.nodes.new(type='FunctionNodeBooleanMath')
                new_node_math.operation = 'OR'
                new_node_math.label = 'OR'
                new_node_structure = Structure()
                new_node_structure.left = new_node_math.inputs[0]
                new_node_structure.right = new_node_math.inputs[1]
                new_node_structure.out = new_node_math.outputs['Boolean']

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
                # make sure that the type fits, e.g. the operator 'LENGTH' first has an output of type 'VECTOR'
                tree.links.new(new_node_structure.out, last_structure.right)
                # success = False
                # for o in new_node_math.outputs:
                #     if o.type == last_operator.inputs[1].type:
                #         tree.links.new(o, last_operator.inputs[1])
                #         success = True
                #         break
                # if not success:
                #     # try the other way round
                #     for i in range(len(last_operator.inputs) - 1, -1, -1):
                #         if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                #             tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                #             break
                right_empty = False
            elif left_empty:
                # make sure that the type fits, e.g. the operator 'LENGTH' first has an output of type 'VECTOR'
                tree.links.new(new_node_structure.out, last_structure.left)
                # success = False
                # for o in new_node_math.outputs:
                #     if o.type == last_operator.inputs[0].type:
                #         tree.links.new(o, last_operator.inputs[0])
                #         success = True
                #         break
                # if not success:
                #     for i in range(len(last_operator.inputs)):
                #         if last_operator.inputs[i].type == new_node_math.outputs[0].type:
                #             tree.links.new(new_node_math.outputs[0], last_operator.inputs[i])
                #             break
                left_empty = False

        elif next_element in scalars or next_element in vectors:
            if last_operator is None:
                tree.links.new(in_channels[next_element], out)
            elif right_empty:
                tree.links.new(in_channels[next_element], last_structure.right)
                right_empty = False
            elif left_empty:
                tree.links.new(in_channels[next_element], last_structure.left)
                left_empty = False

        # remove _x, _y, _z flag for parameter detection
        elif next_element[0:-2] in vectors:
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
            if next_element == 'pi':
                number = np.pi
            elif next_element == 'e_x':
                number = Vector([1, 0, 0])
            elif next_element == 'e_y':
                number = Vector([0, 1, 0])
            elif next_element == 'e_z':
                number = Vector([0, 0, 1])
            elif next_element[0] == '(':
                next_element = next_element[1:-1]
                numbers = next_element.split(' ')
                vals = []
                for i in range(len(numbers)):
                    if numbers[i] == 'pi':
                        vals.append(np.pi)
                    elif number[i] == '-pi':
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
                # find the first 'VALUE' input
                # for i in range(1, 3):
                #     if last_operator.inputs[i].type == 'VALUE':
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
            build_function(tree, stack, scalars=scalars, vectors=vectors, in_channels=in_channels,
                           out=out, fcn_count=fcn_count, length=length - 1, unary=unary, last_operator=new_node_math,
                           last_structure=new_node_structure, height=height,
                           level=new_level)
            new_node_math = None


def layout(tree, mode='Sugiyama'):
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
    if mode == 'Sugiyama':
        layout = SugiyamaLayout(graph.C[0])
        layout.init_all(roots=root_vertices)
        layout.draw(10)
    elif mode == 'Digco':
        layout = DigcoLayout(graph.C[0])
        layout.init_all()
        layout.draw()

    for v in graph.C[0].sV:
        v.data.location = (v.view.xy[1], v.view.xy[0])
