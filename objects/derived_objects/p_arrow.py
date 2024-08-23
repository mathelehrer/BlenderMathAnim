import bmesh
import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from objects.geometry.geo_bobject import GeoBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import get_rotation_quaternion_from_start_and_end, get_save_length, to_vector




class PArrow(GeoBObject):
    '''
    example
    PArrow(start=[1,0,0], end=Vector(), name='Arrow3', color='example', thickness=1),

    It's important that every new object get's its own name, otherwise there will be conflicts during import

    '''
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.start = to_vector(self.get_from_kwargs('start',Vector()))
        self.end = to_vector(self.get_from_kwargs('end',Vector([0,0,1])))
        self.thickness= self.get_from_kwargs('thickness',1)
        length= (self.end-self.start).length
        name=self.get_from_kwargs('name','Arrow')
        arrow = BObject.from_file("Arrow",name=name)

        if 'loop_cuts' in kwargs:
            loop_cuts=kwargs['loop_cuts']
        else:
            loop_cuts=0

        if loop_cuts>0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(arrow.ref_obj.data)  # Fills it in using the cylinder
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=int(loop_cuts*length), use_grid_fill=True)
            bm.to_mesh(arrow.ref_obj.data)
            bm.free()

        length=get_save_length(self.start,self.end)
        quaternion = get_rotation_quaternion_from_start_and_end(self.start,self.end)
        super().__init__(obj=arrow.ref_obj,location=self.start,rotation_quaternion=quaternion,scale=[1,1,length],name=name,**kwargs)

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.appeared=True
        scale = self.ref_obj.scale.copy()
        self.ref_obj.scale=[0,0,0]
        ibpy.insert_keyframe(self.ref_obj,'scale',frame=int(begin_time*FRAME_RATE))
        self.ref_obj.scale=[scale[0]*self.thickness,scale[1]*self.thickness,scale[2]]
        ibpy.insert_keyframe(self.ref_obj, 'scale', frame=int((begin_time+transition_time) * FRAME_RATE))
        return begin_time+transition_time