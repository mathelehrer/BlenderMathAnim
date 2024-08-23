import bpy
from mathutils import Quaternion

from appearance.textures import apply_material
from interface import ibpy
from interface.ibpy import change_emission
from utils.constants import *


class Container(object):
    """
    This is a container class that allows to group BObjects
    """

    def __init__(self, **kwargs):
        # register bpy object from sub class or create default object
        self.appeared = False
        self.kwargs = kwargs
        self.updaters = []
        self.label_sep = 1
        self.name = self.get_from_kwargs('name', 'b_object')
        if 'obj' in kwargs:
            ref_obj = ibpy.get_obj(kwargs['obj'])
            if self.name != 'b_object':
                ref_obj.name = self.name
            kwargs.pop('obj')
        else:
            raise 'Container must be connected to an existing BObject'

        self.intrinsic_scale = self.get_from_kwargs('scale', 1)
        if isinstance(self.intrinsic_scale, int) or isinstance(self.intrinsic_scale, float):
            self.intrinsic_scale = [self.intrinsic_scale] * 3

        ref_obj.scale = self.intrinsic_scale
        self.total_motion = Vector()
        # override location if it is explicitly given
        if 'location' in kwargs:
            location = self.get_from_kwargs('location', [0, 0, 0])
            ref_obj.location = location
        if 'rotation_euler' in kwargs:
            ref_obj.rotation_mode = 'XYZ'
            ref_obj.rotation_euler = self.get_from_kwargs('rotation_euler', [0, 0, 0])
        elif 'rotation_quaternion' in kwargs:
            ref_obj.rotation_mode = 'QUATERNION'
            ref_obj.rotation_quaternion = self.get_from_kwargs('rotation_quaternion', Quaternion())

        self.ref_obj = ref_obj

        super().__init__()

    ##########
    # Getter #
    ##########

    def get_world_location(self):
        if self.ref_obj.parent:
            return self.ref_obj.parent.matrix_world @ self.ref_obj.location
        else:
            return self.ref_obj.location

    def get_location(self):
        return self.ref_obj.location

    def get_from_kwargs(self, kwarg_str, default):
        if kwarg_str in self.kwargs:
            result = self.kwargs[kwarg_str]
            self.kwargs.pop(
                kwarg_str)  # change introduced on 28.5.2022 automatically remove argument, whenever it was used
            return result
        else:
            return default

    ############
    # Setter ##
    ###########
    def set_rigid_body(self, dynamic=True):
        ibpy.make_rigid_body(self, dynamic=dynamic)

    ################################
    # non-animated transformations #
    ################################

    def rotate_by(self, rotation_euler=[0, 0, 0]):
        ibpy.rotate_by(self, rotation_euler=rotation_euler)

    ##############
    # Animations #
    ##############

    def appear(self, begin_time=0, clear_data=False, silent=False):
        """
        makes the object simply fade in with in the transition time
        from alpha = 0 to alpha defined in kwargs (default 1)

        :param silent:
        :param clear_data:  this is useful for copies of objects to remove animation data from inherited from the parent
        :param begin_time:
        :return:
        """

        if not self.appeared:
            if not silent:
                print("Appear " + self.ref_obj.name)
            obj = self.ref_obj
            if obj.name not in bpy.context.scene.objects:
                ibpy.link(obj)
            if clear_data:
                ibpy.clear_animation_data(self)
            self.appeared = True

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """
        :param transition_time:
        :param begin_time:
        :return:
        """
        if self.appeared:
            disappear_frame = begin_time * FRAME_RATE
            if transition_time == 0:
                transition_frames = 1
            else:
                transition_frames = transition_time * FRAME_RATE
            ibpy.fade_out(self, disappear_frame, transition_frames, **kwargs)
            self.appeared = False

    def move(self, direction, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        move an object
        :param direction: in the direction
        :param begin_time: beginning of the motion
        :param transition_time: duration of the motion
        :return:
        """
        if isinstance(direction, list):
            direction = Vector(direction)
        ibpy.move(self, direction, begin_time * FRAME_RATE, transition_time * FRAME_RATE)

    def move_to(self, target_location, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, global_system=False):
        """
        move an object
        :param global_system: if set True the motion is performed in the world system
        :param target_location: to the target_location
        :param begin_time:
        :param transition_time:
        :return:
        """

        ibpy.move_to(self, target_location, begin_time * FRAME_RATE, transition_time * FRAME_RATE,
                     global_system=global_system)

    def move_copy(self, direction=[0, 0, 0], begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        obj_copy = self.copy()
        ibpy.clear_animation_data(obj_copy)  # remove animation data to make it appear independently of the src object
        obj_copy.appear(begin_time=begin_time, transition_time=0)  # make copy appear
        obj_copy.move(direction=direction, begin_time=begin_time, transition_time=transition_time)
        return obj_copy

    def move_copy_to(self, target_location, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        obj_copy = self.copy()
        obj_copy.appear(begin_time=begin_time, transition_time=0)  # make copy appear
        obj_copy.move_to(target_location=target_location, begin_time=begin_time, transition_time=transition_time)
        return obj_copy

    def rotate(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, pivot=None, interpolation='BEZIER',
               **kwargs):
        """
        rotate an object
        :param interpolation: CONSTANT for linear interpolation
        :param rotation_euler:
        :param rotation_quaternion:
        :param begin_time:
        :param transition_time:
        :param pivot:
        :return:
        """

        ibpy.rotate_to(self, begin_time * FRAME_RATE, transition_time * FRAME_RATE, pivot, interpolation, **kwargs)

    def scale(self, initial_scale=0, final_scale=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.grow(scale=final_scale, begin_time=begin_time, transition_time=transition_time,
                  initial_scale=initial_scale)

    def rescale(self, rescale=[1, 1, 1], begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        ibpy.rescale(self, rescale, begin_time * FRAME_RATE, np.maximum(1, transition_time * FRAME_RATE), **kwargs)

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center',pivot=None,
             initial_scale=0):
        """
        grow an object from 0 to
        :param scale: the final scale
        :param begin_time: starting time
        :param transition_time: duration
        :param modus: can be 'from_center', 'from_left', 'from_right', 'from_top', 'from_bottom', 'from_front', 'from_back'
        :return:
        """

        self.appear(begin_time=begin_time,transition_time=0,silent=True)
        if pivot:
            ibpy.set_pivot(self,pivot)
        if scale is None:
            scale = self.intrinsic_scale
        ibpy.grow(self, scale, begin_time * FRAME_RATE, transition_time * FRAME_RATE, initial_scale, modus)

    def shrink(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        shrink an object to 0
        :param begin_time: starting time
        :param transition_time: duration
        :return:
        """
        ibpy.shrink(self, begin_time, transition_time)

    def next_to(self, parent, direction=RIGHT, buff=SMALL_BUFF, shift=0 * RIGHT):
        """
        aligns self relative to parent
        directions can be {UP, DOWN, LEFT, RIGHT, FRONT, BACK} or any vector in principle
        :param shift:
        :param parent:
        :param direction:
        :param buff:
        :return:
        """
        # ibpy.set_parent(self, parent)
        if direction == UP or direction == DOWN:
            dim = 1
        elif direction == RIGHT or direction == LEFT:
            dim = 0
        else:
            dim = 2

        dist = ibpy.get_dimension(parent, dim) + ibpy.get_dimension(self, dim)
        dist /= 2
        dist *= self.ref_obj.scale[dim]
        location_parent = ibpy.get_location(self)  # ibpy.get_location(parent)
        ibpy.set_location(self, location_parent + (dist + buff) * direction + shift)

    def get_location_at_frame(self, frame):
        ibpy.set_frame(frame)
        location = ibpy.get_location(self)
        return location

    def update_position(self, location_frame_function, begin_time, transition_time=OBJECT_APPEARANCE_TIME,
                        location=None, resolution=1):
        if location is None:
            location = ibpy.get_location(self)
        start_frame = int(begin_time * FRAME_RATE)
        end_frame = int((begin_time + transition_time) * FRAME_RATE)

        ibpy.insert_keyframe(self.ref_obj, "location", start_frame)

        for frame in range(start_frame, end_frame, resolution):
            new_location = location_frame_function(frame + resolution)
            if location != new_location:
                self.ref_obj.location = new_location
                ibpy.insert_keyframe(self.ref_obj, "location", frame + resolution)
                location = new_location.copy()

    def add_child(self, b_object):
        ibpy.set_parent(b_object, self)

    def follow(self, curve, initial_value=0, final_value=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):
        if 'new_constraint' in kwargs:
            new_constraint = kwargs.pop('new_constraint')
        else:
            new_constraint = True
        if new_constraint:
            ibpy.set_follow(self, curve)
        ibpy.follow(self, curve, initial_value=initial_value, final_value=final_value, begin_time=begin_time,
                    transition_time=transition_time, **kwargs)

    def hide(self, begin_time=0):
        ibpy.hide(self, begin_time=begin_time)

    def un_hide(self, begin_time=0):
        ibpy.unhide(self, begin_time=begin_time)

    def change_emission(self, from_value=0, to_value=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        change_emission(self, from_value=from_value, to_value=to_value, begin_frame=begin_time * FRAME_RATE,
                        frame_duration=transition_time * FRAME_RATE)

    def clear_parent(self):
        ibpy.clear_parent(self)
