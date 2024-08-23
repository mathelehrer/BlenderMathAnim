import bmesh

import numpy as np
from mathutils import Vector, Quaternion

from interface import ibpy
from interface.ibpy import add_cylinder, add_torus
from objects.geometry.geo_bobject import GeoBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import get_rotation_quaternion_from_start_and_end, to_vector


class Torus(GeoBObject):
    """
    Create a Torus with a descent mesh:
    """
    def __init__(self, **kwargs):
        """
        example:
        torus = Torus(
        location=(0,0,0),
        rotation_euler=(np.pi/2,0,0),
        major_segments=96,
        minor_segments=24,
        major_radius=20,
        minor_radius=0.1)
        """
        self.kwargs = kwargs

        location = self.get_from_kwargs('location',Vector())
        name = self.get_from_kwargs('name', 'Torus')

        if 'rotation_euler' in kwargs:
            self.rotation_euler=kwargs.pop('rotation_euler')
        else:
            self.rotation_euler=None

        if 'rotation_quaternion' in kwargs:
            self.rotation_quaternion=kwargs.pop('rotation_quaternion')
            self.quaternion=self.rotation_quaternion.copy()
        else:
            self.rotation_quaternion=None

        loop_cuts=self.get_from_kwargs('loop_cuts',0)

        major_radius=self.get_from_kwargs('major_radius',10)
        minor_radius=self.get_from_kwargs('minor_radius',0.1)
        major_segments = self.get_from_kwargs('major_segments',96)
        minor_segments = self.get_from_kwargs('minor_segments',96)
        torus = add_torus(major_segments=major_segments,
                          minor_segments=minor_segments,
                          major_radius=major_radius,
                          minor_radius=minor_radius,
                          )

        if loop_cuts>0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(torus.data)  # Fills it in using the cylinder
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=loop_cuts, use_grid_fill=True)
            bm.to_mesh(torus.data)
            bm.free()


        label_rotation=self.get_from_kwargs('label_rotation',[np.pi/2,np.pi/2,0])
        if self.rotation_euler:
            super().__init__(obj=torus, location=location,rotation_euler=self.rotation_euler,label_rotation=label_rotation, name=name, **kwargs)
        elif self.rotation_quaternion:
            super().__init__(obj=torus, location=location,rotation_quaternion=self.rotation_quaternion,label_rotation=label_rotation, name=name, **kwargs)
        ibpy.un_link(self.ref_obj,collection="Collection") # unlink the object, since it is linked automatically from the mesh creation process


