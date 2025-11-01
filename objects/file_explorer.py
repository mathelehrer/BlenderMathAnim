import os

import numpy as np

from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier, get_node_from_shader
from geometry_nodes.geometry_nodes_modifier import FileExplorerModifier
from objects.bobject import BObject
from objects.plane import Plane
from utils.constants import DEFAULT_ANIMATION_TIME, DATA_DIR
from utils.io_operations import list_files_with_sizes_recursive, convert_files_to_csv_data


class FileExplorer(BObject):
    def __init__(self, path="/usr",max_length=35,max_data=2000,**kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs("name", "FileExplorer")

        convert_files_to_csv_data( path, max_length, max_data, max_number=10000)

        self.geo = Plane(name=self.name, **kwargs)
        self.modifier = FileExplorerModifier(csv_file=os.path.join(DATA_DIR,path.replace("/","")+"_data.csv"),max_length=max_length+2)
        self.geo.add_mesh_modifier(type="NODES", node_modifier=self.modifier)
        self.scatter_material = self.modifier.materials[1]

        super().__init__(obj=self.geo, **kwargs)

    def appear(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        scatter_node=get_node_from_shader(self.scatter_material,"ScatterValue")
        ibpy.change_default_value(scatter_node,from_value=0,to_value=5,begin_time=begin_time,transition_time=transition_time)
        return super().appear(begin_time=begin_time,transition_time=transition_time)

    def scroll(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        scroll_node =get_geometry_node_from_modifier(self.modifier,"Scroll")
        ibpy.change_default_value(scroll_node,from_value=-5.5,to_value=1000,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time