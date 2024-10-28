import bmesh

import numpy as np
from mathutils import Vector, Quaternion

from interface import ibpy
from interface.ibpy import add_cylinder
from objects.geometry.geo_bobject import GeoBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import get_rotation_quaternion_from_start_and_end, to_vector


class Cylinder(GeoBObject):
    """
    Create a cylinder with a descent mesh:
    """
    def __init__(self, start=None,end=None,location=[0, 0, 0], length=1, radius=0.1, cyl_radii=None, **kwargs):
        """
        :param location:
        :param start:
        :param end:
        :param length:
        :param radius:
        :param cyl_radii:
        :param kwargs:
        """
        self.kwargs = kwargs
        self.start=to_vector(start)
        self.end=to_vector(end)
        self.length=length

        if 'rotation_euler' in kwargs:
            self.rotation_euler=kwargs['rotation_euler']
        else:
            self.rotation_euler=None

        if 'rotation_quaternion' in kwargs:
            self.rotation_quaternion=kwargs['rotation_quaternion']
            self.quaternion=self.rotation_quaternion.copy()
        else:
            self.rotation_quaternion=None

        loop_cuts=self.get_from_kwargs('loop_cuts',0)
        vertices = self.get_from_kwargs('vertices',32)
        cylinder = add_cylinder(vertices=vertices)

        if loop_cuts>0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(cylinder.data)  # Fills it in using the cylinder
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=int(loop_cuts*length), use_grid_fill=True)
            bm.to_mesh(cylinder.data)
            bm.free()

        name = self.get_from_kwargs('name', 'Cylinder')
        thickness = self.get_from_kwargs('thickness',10*radius)
        label_rotation=self.get_from_kwargs('label_rotation',[np.pi/2,np.pi/2,0])
        thickness*=0.1

        if cyl_radii is None:
            radius_x = thickness
            radius_y = thickness
        else:
            radius_x = cyl_radii[0]
            radius_y = cyl_radii[1]

        super().__init__(obj=cylinder, location=location,label_rotation=label_rotation, scale=[radius_x, radius_y, length / 2], name=name, **kwargs)
        ibpy.un_link(self.ref_obj,collection="Collection") # unlink the object, since it is linked automatically from the mesh creation process


    @classmethod
    def from_start_to_end(cls, start=[0,0,0], end=[0,0,1], radius=0.1, cyl_radii=None, **kwargs):
        diff = (to_vector(end) - to_vector(start))
        length=diff.copy().length
        location = 0.5 * (to_vector(end) + to_vector(start))
        quaternion = get_rotation_quaternion_from_start_and_end(start,end)
        return Cylinder(start=start,end=end,location=location,length=length,rotation_quaternion=quaternion,radius=radius,cyl_radii=cyl_radii,**kwargs)

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_start',
             initial_scale=0):
        super().appear(begin_time=begin_time,transition_time=0)

        print("Grow "+self.ref_obj.name)
        if not scale:
            scale = self.ref_obj.scale.copy()
        if modus == 'from_start' and self.start:
            ibpy.grow_from(self, self.start, begin_time*FRAME_RATE, transition_time*FRAME_RATE)
        elif modus == 'from_end' and self.end:
            ibpy.grow_from(self, self.end, begin_time*FRAME_RATE, transition_time*FRAME_RATE)
        elif modus =='from_center' and self.start and self.end:
            ibpy.grow_from(self,0.5*(self.start+self.end),begin_time*FRAME_RATE,transition_time*FRAME_RATE)
        else:
            ibpy.grow(self, scale, begin_time * FRAME_RATE, transition_time * FRAME_RATE, modus=modus,
                      initial_scale=initial_scale)
        self.appeared =True
        return begin_time+transition_time

    def shrink(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_start',
             initial_scale=0):

        print("Shrink " + self.ref_obj.name)
        if not scale:
            scale = self.ref_obj.scale.copy()
        if modus == 'from_start' and self.start:
            ibpy.shrink_from(self, self.start, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        elif modus == 'from_end' and self.end:
            ibpy.shrink_from(self, self.end, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        elif modus == 'from_center' and self.start and self.end:
            ibpy.shrink_from(self, 0.5 * (self.start + self.end), begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        else:
            ibpy.shrink(self, scale, begin_time * FRAME_RATE, transition_time * FRAME_RATE, modus=modus,
                      initial_scale=initial_scale)
        return begin_time + transition_time

    def move_end_point(self, target_location=Vector(), begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ab = (self.end - self.start)
        length = ab.length
        ab = ab.normalized()
        target_location=to_vector(target_location)
        ab_new = target_location - self.start
        l_new = ab_new.length
        ab_new = ab_new.normalized()
        scale = l_new / length
        axis = ab.cross(ab_new)
        if ab_new.length>0 and ab.dot(ab_new)<0: # significant change in direction
            scale*=-1
        if axis.length>0 and np.abs(ab.dot(ab_new))<=1:
            axis = axis.normalized()
            angle = np.arccos(ab.dot(ab_new))
            axis *= np.sin(angle / 2)
            quaternion = Quaternion([np.cos(angle / 2), *axis[:]])
            quaternion_full = quaternion @ self.quaternion
            self.rotate(rotation_quaternion=quaternion_full, begin_time=begin_time, transition_time=transition_time,
                        pivot=self.start)
        else:
            quaternion_full=self.quaternion

        self.rescale(rescale=[1, 1, scale], begin_time=begin_time, transition_time=transition_time)

        self.quaternion = quaternion_full
        self.end = target_location

    def update_rotation_free_motion(self,start=lambda x:x, end=lambda x:x, begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,resolution=1,fixed_start_pos=False):
        """
        !!!The pivot of the cylinder has to be in the center to work properly
        !!!grow cylinder with modus='from_center'

        :param start:
        :param end:
        :param begin_time:
        :param transition_time:
        :param resolution:
        :return:
        """
        self.start=to_vector(self.start)
        self.end=to_vector(self.end)
        end_frame = int((begin_time + transition_time) * FRAME_RATE)
        for frame in range(int(begin_time * FRAME_RATE), end_frame, resolution):
            new_start = to_vector(start(frame+resolution))
            new_end = to_vector(end(frame+resolution))
            new_dir = new_end-new_start
            dir = self.end-self.start


            old_center = 0.5*(self.start+self.end)
            new_center = 0.5*(new_start+new_end)
            direction = new_center-old_center
            self.move(direction=direction,begin_time=frame/FRAME_RATE,transition_time=resolution/FRAME_RATE)
            self.rescale(rescale=[1,1,new_dir.length/dir.length],begin_time=frame/FRAME_RATE,transition_time=resolution/FRAME_RATE)
            self.start= new_start
            self.end = new_end

    def move_to_new_start_and_end_point(self,start=Vector(), end=Vector(), begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        """
        :param start:
        :param end:
        :param begin_time:
        :param transition_time:
        :param resolution:
        :return:
        """
        new_dir = to_vector(end)-to_vector(start)
        dir = self.end-self.start

        direction = start-self.start
        self.move(direction=direction,begin_time=begin_time,transition_time=transition_time)
        self.rescale(rescale=[1,1,new_dir.length/dir.length],begin_time=begin_time,transition_time=transition_time)
        quaternion = get_rotation_quaternion_from_start_and_end(start, end)
        self.rotate(rotation_quaternion =quaternion,begin_time=begin_time,transition_time=transition_time)
        self.start = start
        self.end = end
