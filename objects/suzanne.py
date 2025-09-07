from itertools import combinations

import numpy as np
from mathutils import Vector

from geometry_nodes.geometry_nodes_modifier import UnfoldModifier
from interface.ibpy import create_mesh
from mathematics.mathematica.mathematica import tuples, choose, partition
from objects.bobject import BObject
from objects.cube import Cube
from utils.kwargs import get_from_kwargs
from utils.string_utils import get_data_from_obj

pi = np.pi



class Suzanne(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name=self.get_from_kwargs('name','Suzanne')
        #
        vertices,edges,faces = get_data_from_obj("suzanne.obj")
        super().__init__(mesh=create_mesh(vertices,edges,faces),name=self.name,**kwargs)
