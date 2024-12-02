from appearance.textures import glow_at_appearance
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import MeshLine, JoinGeometry, create_geometry_line, InstanceOnPoints, IcoSphere, Index, \
    SceneTime, InputValue, DeleteGeometry, make_function, SetMaterial, StoredNamedAttribute
from interface import ibpy
from interface.ibpy import create_mesh, get_color
from objects.bobject import BObject
from objects.geometry.sphere import Sphere
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE

from utils.kwargs import get_from_kwargs


class Point(Sphere):
    """
    Create a point, whose location on the screen is associated with a coordinate system
    """

    def __init__(self, coordinate_system, coordinates=(0,0,0), size = 1, material='example', **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name','P')
        self.kwargs.pop('name')
        self.coordinate_system=coordinate_system
        location = self.coordinate_system.coords2location(coordinates)
        super().__init__(size/10,location=location, name=name,material=material, **kwargs)
        self.coordinate_system.add_object(self)

class PointCloudModifier(GeometryNodesModifier):

    def __init__(self,size=0,**kwargs):
        self.size = size
        self.kwargs = kwargs
        super().__init__(name="PointCloudModifier",group_input=True, automatic_layout=True)


    def create_node(self, tree):
        out = tree.nodes.get("Group Output")
        ins = tree.nodes.get("Group Input")
        links = tree.links

        sphere = IcoSphere(tree)
        iop = InstanceOnPoints(tree,instance = sphere.geometry_out)
        join_geometry = JoinGeometry(tree)

        # create nodes for gradual appearance
        index = Index(tree)
        time = SceneTime(tree)

        start_frame = InputValue(tree,name="StartFrame")
        end_frame = InputValue(tree,name="EndFrame")

        # the glow function is just 1/(1+Delta)
        # where delta is a growing function, once the object has appeared.
        selector = make_function(tree,
                                 functions={
                                     "select":"frame,begin,-,end,begin,-,/,"+str(self.size)+",*,index,<",
                                     "glow":"10,1,frame,begin,-,end,begin,-,/,"+str(self.size)+",*,index,-,+,/"
                                 }, name="Selector",
                                 scalars=["index","frame","begin","end","select","glow"],inputs=["index","frame","begin","end"],
                                 outputs=["select","glow"])
        links.new(start_frame.std_out,selector.inputs["begin"])
        links.new(end_frame.std_out,selector.inputs["end"])
        links.new(time.outputs["Frame"],selector.inputs["frame"])
        links.new(index.std_out,selector.inputs["index"])

        glow_attr = StoredNamedAttribute(tree,data_type="FLOAT",name="Glow",domain="INSTANCE",value=selector.outputs["glow"])
        color = glow_at_appearance(**self.kwargs)
        set_mat = SetMaterial(tree,material=color)

        del_geo = DeleteGeometry(tree,selection=selector.outputs["select"])


        create_geometry_line(tree,[del_geo,iop,glow_attr,set_mat,join_geometry], ins = ins.outputs["Geometry"], out=out.inputs["Geometry"])



class PointCloud(BObject):
    """
    Create a point cloud using geometry nodes
    """
    def __init__(self, points = None,**kwargs):
        if points:
            name = get_from_kwargs(kwargs,"name","PointCloud")
            super().__init__(mesh=create_mesh(vertices = points),name=name,**kwargs)
            self.modifier = PointCloudModifier(size=len(points),**kwargs)
            self.add_mesh_modifier(type="NODES",node_modifier=self.modifier)

    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               clear_data=False, silent=False,linked=False, nice_alpha=False,children=True,**kwargs):
        super().appear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,clear_data=clear_data,silent=silent,nice_alpha=nice_alpha,children=children,**kwargs)
        begin_frame_node = ibpy.get_geometry_node_from_modifier(self.modifier,label="StartFrame")
        end_frame_node = ibpy.get_geometry_node_from_modifier(self.modifier,label="EndFrame")
        ibpy.change_default_value(begin_frame_node,from_value=0,to_value=begin_time*FRAME_RATE,begin_time=0,transition_time=0)
        ibpy.change_default_value(end_frame_node,from_value=0,to_value=(begin_time+transition_time)*FRAME_RATE,begin_time=0,transition_time=0)
        return begin_time+transition_time