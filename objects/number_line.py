import numpy as np
from mathutils import Vector

from geometry_nodes.geometry_nodes_modifier import NumberLineModifier
from interface import ibpy
from objects.cone import Cone
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.bobject import BObject
from objects.digital_number import DigitalRange
from objects.tex_bobject import SimpleTexBObject
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME


class NumberLine2(BObject):
    def __init__(self,name='Numberline',**kwargs):
        r"""Create a number line using geometry nodes:
           :param name:
               name shown in Blender
           :type first: ``str``
           :param \**kwargs:
               See below
           : Keyword Arguments:
               * *extra* (length=1,radius=0.05,domain=[0,1],location=[0,0,0],n_tics=10,
               origin=0,
               tic_labels='AUTO',
               tic_label_digits=0,
               tic_label_aligned='center',
               tic_label_shift=[0,0,0],
               label_unit='',
               label_position='left',
               label_closeness=1,
               tip_length=0.2,
               auto_smooth=True,
               axis_label='x',
               direction="HORIZONTAL|VERTICAL|DEEP|NONE"
               tic_label_suffix=""
               )--
           """

        self.modifier = NumberLineModifier(**kwargs)
        cube = ibpy.add_cube()
        self.kwargs = kwargs
        super().__init__(obj=cube, name=name, no_material=True, **kwargs)
        super().add_mesh_modifier('NODES',node_modifier=self.modifier)

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center', pivot=None,
             initial_scale=0,alpha=1):
        super().appear(alpha=alpha, begin_time=begin_time, transition_time=0, silent=True,children=True)
        length_node = ibpy.get_geometry_node_from_modifier(self.modifier,"AxisLength")
        radius_node = ibpy.get_geometry_node_from_modifier(self.modifier,"Radius")
        length = self.get_from_kwargs("length",1)
        radius = self.get_from_kwargs("radius",0.05)
        ibpy.change_default_value(length_node,from_value=0,to_value=length,begin_time=begin_time,transition_time=transition_time)
        ibpy.change_default_value(radius_node,from_value=0,to_value=radius,begin_time=begin_time,transition_time=transition_time)
        label_scale = ibpy.get_geometry_node_from_modifier(self.modifier,"LabelScale")
        ibpy.change_default_value(label_scale,from_value=0,to_value=1,begin_time=begin_time+0.9*transition_time,transition_time=0.1*transition_time)
        return begin_time+transition_time

    def to_log(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        log_node = ibpy.get_geometry_node_from_modifier(self.modifier,"Log")
        ibpy.change_default_value(log_node,from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

class NumberLine(BObject):
    """
    Create a number line :

    """

    def __init__(self,
                 length=1,
                 radius=0.05,
                 domain=[0, 1],
                 n_tics=10,
                 label='x',
                 tic_labels='AUTO',
                 include_zero=True,
                 direction='VERTICAL',
                 origin=0,
                 label_digit=1,
                 label_unit='',
                 location_of_origin=[0, 0, 0],
                 label_position='left',
                 label_closeness=1,
                 axis_label_closeness=1,
                 tip_length=None,
                 shading=None,
                 **kwargs):
        """

        :param length:
        :param radius:
        :param domain:
        :param n_tics:
        :param label:
        :param tic_labels:
        :param include_zero:
        :param direction:
        :param origin: the value of the number line that is at the origin
        :param location_of_origin: the world position of the origin
        :param kwargs:
        """

        direction = direction.upper()
        self.kwargs = kwargs
        self.label_digit = label_digit
        self.dynamic = self.get_from_kwargs('dynamic',False)
        self.range = self.get_from_kwargs('range',list(np.arange(0,1.1,1/(10**label_digit))))
        np_min = np.min(domain)
        np_max = np.max(domain)
        if np_min == np_max:
            raise Warning("invalid domain was given " + str(domain))
        self.length = length
        self.interval = {'min': np_min, 'max': np_max}
        # this is the positioning of the labels before the axis gets rotated
        # since labels and axis are rotated in combination into the final place
        if direction == 'VERTICAL':
            label_rotation = [np.pi / 2, 0, 0]
            label_aligned='right'
            if label_position=='left':
                label_offset = Vector([-0.1, 0, -0.15])
            elif label_position=='center':
                label_offset = Vector([-0.1, 0, 0])
            elif label_position=='right':
                label_offset = Vector([-0.1, 0, 0.15])
            label_offset+=Vector([-label_closeness*0.1,0,0])
            rotation_euler = self.get_from_kwargs('rotation_euler', [0, 0, 0])
            self.grow_mode = 'from_bottom'
        elif direction == 'HORIZONTAL' or direction == 'DEEP':
            label_offset = Vector([label_closeness*0.4, 0, 0])
            label_rotation = [np.pi / 2, -np.pi / 2, 0]
            if label_position == 'left':
                label_aligned = 'right'
            if label_position == 'right':
                label_aligned = 'left'
            if label_position == 'center':
                label_aligned = 'center'
            rotation_euler = self.get_from_kwargs('rotation_euler', [0, np.pi / 2, 0])
            self.grow_mode = 'from_left'
        if direction == 'DEEP':
            self.grow_mode = 'from_front'
            rotation_euler = self.get_from_kwargs('rotation_euler', [0, np.pi / 2, np.pi / 2])

        self.grow_mode = 'from_start'

        label_color = self.get_from_kwargs('label_color', 'drawing')
        color = self.get_from_kwargs('color', 'drawing')

        axis_label_size = self.get_from_kwargs('axis_label_size', 'normal')
        tic_label_size = self.get_from_kwargs('tic_label_size', 'normal')

        if 'text_size' in self.kwargs:  # the text_size option overrides the individual options for axis label and tic labels
            text_size = self.get_from_kwargs('text_size', 'normal')
            if axis_label_size != text_size:
                axis_label_size = text_size
            if tic_label_size != text_size:
                tic_label_size != text_size

        axis_label_offset = Vector([0, 0, axis_label_closeness])
        # by default a number line will be created that is centered at the middle
        # it is possible to locate the center of the number line at a particular x-value
        self.center = Vector([0, 0, 0])
        self.up = self.center + Vector([0, 0, length / 2])
        self.down = self.center - Vector([0, 0, length / 2])

        pivot = self.get_location(origin)
        center_shift = - Vector(pivot)

        self.center = center_shift
        self.up += self.center
        self.down += self.center

        if tip_length is None:
            tip_length = length / 20

        self.name = self.get_from_kwargs('name', '')

        self.cyl = Cylinder.from_start_to_end(start=self.down,
                                              end=self.up,
                                              radius=radius,
                                              name=self.name + '_' + label + "_axis",
                                              color=color,
                                              shading=shading,
                                              **kwargs)
        self.tip = Cone(length=tip_length,
                        radius=radius * 3,
                        location=self.center + Vector([0, 0, 0.5 * length + tip_length]),
                        color=color,
                        name=self.name + '_' + label + "_axis_tip",
                        shading=shading,
                        **kwargs)

        # create default tic_values
        if n_tics > 0:
            self.tic_values = []
            for i in range(0, int(n_tics) + 1):
                tic_value = self.interval['min'] + i * (self.interval['max'] - self.interval['min']) / n_tics
                self.tic_values.append(tic_value)

            # tics
            self.tics = []
            for i in range(0,int( n_tics) + 1):
                if include_zero or (self.tic_values[i] != 0 and self.tic_values[i] != '0'):
                    tic = Cylinder(length=radius / 2,
                                   location=self.get_location(self.tic_values[i]), radius=radius * 3,
                                   name=label + "_tic_" + str(self.tic_values[i]),
                                   shading=shading, color=color,
                                   **kwargs)
                    self.tics.append(tic)
                    print("tic location for ", self.tic_values[i], " ", self.get_location(self.tic_values[i]))
            self.bTics = BObject(children=self.tics, name=label + '_tics', **kwargs)
        else:
            self.tic_values = []
            self.bTics = None

        # labels

        if len(self.tic_values) > 0:
            last_tic = self.tic_values[-1]
        else:
            last_tic = self.interval['max']
        if len(label) > 0:
            self.axis_label = SimpleTexBObject(str(label),
                                               centered=True,
                                               location=self.get_location(
                                                   last_tic) + label_offset + axis_label_offset,
                                               rotation_euler=label_rotation,
                                               aligned=label_aligned, color=label_color, text_size=axis_label_size,
                                               **kwargs)


        else:
            self.axis_label = None

        self.labels = []
        if isinstance(tic_labels, str):
            if tic_labels == 'AUTO':
                for i in range(0, n_tics + 1):
                    if include_zero or (self.tic_values[i] != 0 and self.tic_values[i] !='0'):
                        if not self.dynamic:
                            lbl = SimpleTexBObject(self.make_tic_label(self.tic_values[i], label_unit),
                                             centered=True,
                                             location=self.get_location(self.tic_values[i]) + label_offset,
                                             rotation_euler=label_rotation,
                                             color=color, text_size=tic_label_size,
                                             aligned=label_aligned, **kwargs)
                        else:
                            lbl = DigitalRange(self.range,initial_value= self.tic_values[i],
                                               digits = self.label_digit,
                                               color=color,value = self.tic_values[i],
                                               rotation_euler=label_rotation,
                                               location=self.get_location(self.tic_values[i])+label_offset,
                                               text_size=tic_label_size,
                                               aligned=label_aligned,**self.kwargs)
                        self.labels.append(lbl)

        elif tic_labels is None:
            self.labels = []
        else:
            for i, lbl in enumerate(tic_labels):
                if include_zero or (lbl != 0 and lbl !='0'):
                    if not self.dynamic:
                        ll = SimpleTexBObject(self.make_tic_label(lbl, label_unit),
                                               centered=True,
                                               location=self.get_location(self.tic_values[i]) + label_offset,
                                               rotation_euler=label_rotation,
                                               color=color, text_size=tic_label_size,
                                               aligned=label_aligned, **kwargs)
                    else:
                        ll = DigitalRange(self.range, initial_value=lbl,
                                           digits=self.label_digit,
                                           color=color, value=self.tic_values[i],
                                           rotation_euler=label_rotation,
                                           location=self.get_location(self.tic_values[i]) + label_offset,
                                           text_size=tic_label_size,
                                           aligned=label_aligned, **self.kwargs)
                    self.labels.append(ll)

        if len(self.labels) > 0:
            self.bLabels = BObject(children=self.labels, name=label + '_Labels')
        else:
            self.bLabels = None

        objects = [self.cyl, self.tip]
        if self.axis_label:
            objects.append(self.axis_label)
        if self.bTics:
            objects.append(self.bTics)
        if self.bLabels:
            objects.append(self.bLabels)

        super().__init__(children=objects,
                         name=label + '_NumberLine',
                         rotation_euler=rotation_euler,
                         location=location_of_origin, **kwargs)

    def make_tic_label(self, value, unit):
        if isinstance(value,str):
            return value+unit
        p = np.power(10, self.label_digit)
        if self.label_digit == 0:
            value = int(np.round(value))
        else:
            value = np.round(value * p) / p
        tic_label = ''
        if unit != '':
            tic_label = unit

        if value == 1:
            if len(tic_label) > 0:
                return tic_label
            else:
                return str(value)
        if value == -1:
            if len(tic_label) > 0:
                return '-' + tic_label
            else:
                return str(value)
        else:
            return str(value) + tic_label

    def get_scale(self):
        return self.length / (self.interval['max'] - self.interval['min'])

    def get_location(self, x):
        """
        returns the world coordinates for a given x-value
        :param x:
        :return:
        """
        direction = self.up - self.down
        scale = (x - self.interval['min']) / (self.interval['max'] - self.interval['min'])
        return self.down + direction * scale

    def appear(self, alpha=1,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,silent = True,**kwargs
               ):
        # distribute appearance time among the components of the number line
        cyl_time = 0.2 * transition_time
        tip_time = 0.2 * transition_time
        if len(self.labels) > 0:
            label_time = 0.6 * transition_time / len(self.labels)
        else:
            label_time = 0

        t0 = begin_time
        super().appear(alpha=alpha, begin_time=t0,silent=silent,children=False,**kwargs) # apearance of children is handled in the remainder
        self.cyl.grow(self.cyl.intrinsic_scale, transition_time=cyl_time)
        t0 += cyl_time
        self.tip.grow(self.tip.intrinsic_scale, begin_time=t0, transition_time=tip_time)
        if self.axis_label:
            self.axis_label.write(begin_time=t0, transition_time=label_time)
        t0 += tip_time
        if self.bLabels:
            self.bLabels.appear(alpha=alpha, begin_time=t0, transition_time=tip_time,children=False)
        if self.bTics:
            self.bTics.appear(alpha=alpha, begin_time=t0, transition_time=tip_time)
            for tic, label in zip(self.tics, self.labels):
                tic.appear(alpha=alpha,begin_time=t0, transition_time=label_time)
                # tic.grow(tic.intrinsic_scale, begin_time=t0, transition_time=label_time)
                # #originally tic.grow() somehow makes tics flying unnecessarily from unexpected initial positions
                label.write(alpha=alpha,begin_time=t0, transition_time=label_time)
                t0 += label_time
        return begin_time+transition_time

    def compensate_zoom(self, zoom=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.cyl.rescale(rescale=[1 / zoom, 1 / zoom, 1], begin_time=begin_time, transition_time=transition_time)
        self.tip.rescale(rescale=[1 / zoom, 1 / zoom, 1], begin_time=begin_time, transition_time=transition_time)
        self.axis_label.rescale(rescale=[1 / zoom, 1 / zoom, 1], begin_time=begin_time, transition_time=transition_time)
        for tic, label in zip(self.tics, self.labels):
            label.rescale(rescale=[1 / zoom, 1 / zoom, 1 / zoom], begin_time=begin_time,
                          transition_time=transition_time)
            tic.rescale(rescale=[1/zoom,1/zoom,1/zoom],begin_time=begin_time,transition_time=transition_time)

    def compensate_rotation(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,delta_rotation=None):
        '''
        TODO this can be done better
        so far we calculate a matrix B that undoes the rotation while acting on the local matrix

        w = p_w * l
        w object's matrix_world
        p_w parent's matrix_world
        l object's matrix_local

        we want to achieve delta**(-1)*w
        we need to find local_rotation  that satisfies

        delta**(-1)*w = p_w*local_rotation*l

        consequently:
        new_rotation = p_w**(-1)*delta**(-1)*w*l**(-1)

        :param begin_time:
        :param transition_time:
        :param delta_rotation:
        :return:
        '''
        if delta_rotation:
            delta_rotation=delta_rotation.inverted()
            # apply the patch to the label
            obj = self.axis_label.ref_obj
            parent = obj.parent
            pw = parent.matrix_world.to_3x3()
            l = obj.matrix_parent_inverse.to_3x3()
            mw = obj.matrix_world.to_3x3()

            local_rotation = pw.inverted()@delta_rotation@mw@l.inverted()
            self.axis_label.rotate(rotation_euler=local_rotation.to_euler(), begin_time=begin_time, transition_time=transition_time)

            # apply the patch to the tic labels
            for label in self.labels:
                obj = label.ref_obj
                parent = obj.parent
                pw = parent.matrix_world.to_3x3()
                l = obj.matrix_parent_inverse.to_3x3()
                mw = obj.matrix_world.to_3x3()

                local_rotation = pw.inverted() @ delta_rotation @ mw @ l.inverted()
                label.rotate(rotation_euler=local_rotation.to_euler(),begin_time=begin_time,transition_time=transition_time)

class DynamicNumberLine(NumberLine):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name','DynamicNumberline')
        super().__init__(dynamic=True,name=self.name, **kwargs)

    def scale_labels(self,scale=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for label in self.labels:
            label.scale(scale=scale,begin_time=begin_time,transition_time=transition_time)

    def shift_labels(self,shift=0,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for label in self.labels:
            label.shift(shift=shift,begin_time=begin_time,transition_time=transition_time)