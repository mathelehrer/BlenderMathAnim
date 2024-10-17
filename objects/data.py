from geometry_nodes.geometry_nodes_modifier import DataModifier
from interface import ibpy
from interface.ibpy import create_mesh
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Data(BObject):
    def __init__(self,data=None,coordinate_system=None,name="DataObject",**kwargs):
        """
        Two dimensional data that is converted into geometry that can be displayed in a coordinate system

        """
        # create edges, when the option linesize appears as keyword argument
        edges =[]

        if "linesize" in kwargs:
            for i in range(len(data)-1):
                edges.append([i,i+1])
        super().__init__(mesh=create_mesh(data,edges),name=name)
        self.data_modifier = None

        if coordinate_system:
            domains = coordinate_system.domains
            coordinate_system.add_data(self)
        if data:
            self.data_modifier = DataModifier(x_domain=domains[0], y_domain=domains[1],**kwargs)
            self.add_mesh_modifier(type='NODES', node_modifier=self.data_modifier)

    def change_pointsize(self,from_pointsize=1,to_pointsize=0.5,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        pointsize_node = ibpy.get_geometry_node_from_modifier(self.data_modifier,label="PointSize")
        ibpy.change_default_value(pointsize_node,from_value=0.05*from_pointsize,to_value=0.05*to_pointsize,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time


    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               clear_data=False, silent=False,linked=False, nice_alpha=False,**kwargs):
        super().appear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,
                       clear_data=clear_data,silent=silent,linked=linked,nice_alpha=nice_alpha)
        if self.data_modifier:
            t0_node = ibpy.get_geometry_node_from_modifier(self.data_modifier,"T0")
            t1_node = ibpy.get_geometry_node_from_modifier(self.data_modifier,"T1")
            ibpy.change_default_value(t0_node,from_value=0,to_value=begin_time,begin_time=0,transition_time=0)
            ibpy.change_default_value(t1_node,from_value=0,to_value=begin_time+max(2/FRAME_RATE,transition_time),begin_time=0,transition_time=0)

        return begin_time+transition_time

    def zoom_x(self,from_domain=[0,1],to_domain=[0,2],begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if self.data_modifier:
            min_node = ibpy.get_geometry_node_from_modifier(self.data_modifier, "X0")
            max_node = ibpy.get_geometry_node_from_modifier(self.data_modifier, "X1")
            ibpy.change_default_value(min_node, from_value=from_domain[0], to_value=to_domain[0], begin_time=begin_time,
                                      transition_time=transition_time)
            ibpy.change_default_value(max_node, from_value=from_domain[1], to_value=to_domain[1], begin_time=begin_time,
                                      transition_time=transition_time)
        return begin_time+transition_time

    def zoom_y(self,from_domain=[0,1],to_domain=[0,2],begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if self.data_modifier:
            min_node = ibpy.get_geometry_node_from_modifier(self.data_modifier, "Y0")
            max_node = ibpy.get_geometry_node_from_modifier(self.data_modifier, "Y1")
            ibpy.change_default_value(min_node, from_value=from_domain[0], to_value=to_domain[0], begin_time=begin_time,
                                      transition_time=transition_time)
            ibpy.change_default_value(max_node, from_value=from_domain[1], to_value=to_domain[1], begin_time=begin_time,
                                      transition_time=transition_time)
        return begin_time+transition_time

    def to_log_x(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        log_node = ibpy.get_geometry_node_from_modifier(self.data_modifier, "Log")
        return ibpy.change_default_value(log_node, from_value=0, to_value=1, begin_time=begin_time,
                                  transition_time=transition_time)