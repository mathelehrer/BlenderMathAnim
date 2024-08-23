import numpy as np
from mathutils import Vector,Euler

from appearance.textures import apply_material
from interface import ibpy
from objects.bobject import BObject
from objects.tex_bobject import SimpleTexBObject
from utils.constants import OBJECT_APPEARANCE_TIME, DOWN, RIGHT, UP, DEFAULT_ANIMATION_TIME, LEFT
from utils.utils import to_vector


class GeoBObject(BObject):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.label_sep = 1
        self.label=None
        self.label_rotation=self.get_from_kwargs('label_rotation',[0,0,0])
        super().__init__(**kwargs)

    def get_center(self):
        return Vector()

    def copy(self,name=None,**kwargs):
        copy = ibpy.duplicate(self)
        if not name:
            name="copy_of_" + self.ref_obj.name
        bcopy = GeoBObject(obj=copy, name= name,scale=self.intrinsic_scale,**kwargs)
        bcopy.appeared=True # necessary, otherwise the copy cannot disappear
        if self.ref_obj.data and len(self.ref_obj.data.materials) > 0:
            apply_material(bcopy.ref_obj, self.color, **self.kwargs)
        return bcopy

    def write_name_as_label(self, letter_set=None,letter_range=None, modus='down',
                            begin_time=0,transition_time=OBJECT_APPEARANCE_TIME,**kwargs):
        if modus == 'down':
            location = self.label_sep * DOWN
        elif modus == 'up':
            location = self.label_sep * UP
        elif modus == 'down_right':
            location = self.label_sep * (DOWN + RIGHT)
        elif modus == 'down_left':
            location = self.label_sep * (DOWN + LEFT)
        elif modus == 'up_right':
            location = self.label_sep * (UP + RIGHT)
        elif modus=='up_left':
            location=self.label_sep*(UP+LEFT)
        elif modus == 'center':
            location = self.get_center()
        elif modus =='right':
            location=self.label_sep*(RIGHT)
        elif modus =='left':
            location=self.label_sep*(LEFT)
        else:
            location = Vector()

        if 'euler' in kwargs:
            euler=kwargs.pop('euler')
        else:
            euler = None

        if 'offset' in kwargs:
            self.label_offset = to_vector(kwargs['offset'])
        else:
            self.label_offset = Vector()

        if 'aligned' in kwargs:
            aligned=kwargs['aligned']
            kwargs.pop('aligned')
        else:
            aligned='center'
        if 'name' in kwargs:
            label = kwargs['name']
            kwargs.pop('name')
        else:
            label=self.name
        # compensate scaling of the parent to keep the label in scale
        if 'scale_compensate' in kwargs:
            scale_compensate =kwargs.pop("scale_compensate")
        else:
            scale_compensate=True

        if scale_compensate:
            scale_compensate=Vector([1,1,1])
            scale_compensate[0] = 1 / self.intrinsic_scale[2]
            scale_compensate[1] = 1 / self.intrinsic_scale[0]
        else:
            scale_compensate = Vector([1, 1, 1])

        # if self.ref_obj.parent:
        #     location = self.ref_obj.parent.matrix_world@location
        if 'decouple_rotation' in kwargs:
            decouple_rotation = kwargs.pop('decouple_rotation')
        else:
            decouple_rotation = False


        if decouple_rotation and not euler:
            rot = self.ref_obj.matrix_world.copy()
            rot.invert()
            local_rot = Euler(self.label_rotation).to_matrix()

            local_rot = rot.to_3x3()@local_rot
            # TODO I give up at this point
            # scale = self.ref_obj.scale

            # scale = local_rot @ scale
            # scale = [np.abs(1 / scale[0]), np.abs(1 / scale[1]), np.abs(1 / scale[2])]

            euler = local_rot.to_euler()
        else:
            euler = self.label_rotation

        self.label = SimpleTexBObject(label, color=self.color,
                                      rotation_euler=euler,scale=scale_compensate,location=location+self.label_offset,aligned=aligned,**kwargs)
        self.label.ref_obj.parent = self.ref_obj

        return self.label.write(letter_set=letter_set,letter_range=letter_range,begin_time=begin_time, transition_time=transition_time)

    def disappear(self,alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        if self.label:
            self.label_disappear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,**kwargs)
        return super().disappear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,**kwargs)

    def label_disappear(self,alpha=0,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        self.label.disappear(alpha=alpha,begin_time=begin_time,transition_time=transition_time,**kwargs)
        return begin_time+transition_time