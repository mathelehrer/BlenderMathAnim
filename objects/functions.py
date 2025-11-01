from interface.ibpy import create_mesh, get_geometry_nodes_modifier, change_default_value, \
    get_geometry_node_from_modifier
from geometry_nodes.geometry_nodes_modifier import SimpleFunctionModifier
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME


class SimpleFunction(BObject):
    def __init__(self,function=lambda x:x,domain=[0,10],num_points=100,name="SimpleFunction",**kwargs):
        vertices = []
        self.num_points = num_points

        for i in range(num_points):
            x = domain[0] + i*(domain[1]-domain[0])/num_points
            z = function(x)
            vertices.append((x,0,z))

        edges = []
        for i in range(num_points-1):
            edges.append([i,i+1])

        super().__init__(mesh=create_mesh(vertices=vertices,edges=edges),name=name,**kwargs)

        self.simple_function_modifier=SimpleFunctionModifier(**kwargs)
        self.add_mesh_modifier(type="NODES",node_modifier=self.simple_function_modifier)

    def grow(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        super().appear(begin_time=begin_time,transition_time=0)
        start_time_node = get_geometry_node_from_modifier(self.simple_function_modifier,label="StartTime")
        change_default_value(start_time_node,from_value=0,to_value=begin_time,begin_time=0,transition_time=0)
        transition_time_node = get_geometry_node_from_modifier(self.simple_function_modifier,label="TransitionTime")
        change_default_value(transition_time_node,from_value=0,to_value=transition_time,begin_time=0,transition_time=0)
        num_point_node = get_geometry_node_from_modifier(self.simple_function_modifier,label="NumPoints")
        change_default_value(num_point_node,from_value=100,to_value=self.num_points,begin_time=0,transition_time=0)
        return begin_time+transition_time
