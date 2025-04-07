import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from objects.tex_bobject import SimpleTexBObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME


class DigitalNumber(SimpleTexBObject):
    """
    A digital number with a given number of digits that can be updated
    """

    def __init__(self, value, prefix="", suffix="", **kwargs):
        self.value = value
        self.kwargs = kwargs
        self.prefix = prefix
        self.suffix = suffix
        self.signed = self.get_from_kwargs('signed', True)
        self.number_of_digits = self.get_from_kwargs('number_of_digits', 0)
        self.p = np.power(10, self.number_of_digits)
        self.displayed_value = np.round(self.value * self.p) / self.p
        if self.number_of_digits == 0:
            self.displayed_value = int(self.displayed_value)
        aligned = self.get_from_kwargs('aligned', 'right')
        super().__init__(self.prefix + self.number2string(self.displayed_value) + self.suffix, aligned=aligned,
                         **kwargs)

    def update_single_value(self, new_value, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, location=None,
                            morphing=True):
        if location is None:
            location = ibpy.get_location(self)
        prev_value = self.value
        target = SimpleTexBObject(self.prefix + self.number2string(new_value) + self.suffix, location=location,
                                  **self.kwargs)
        src_letter_range = [0, len(self.number2string(prev_value))]
        img_letter_range = [0, len(self.number2string(new_value))]
        self.add_to_morph_chain(target,
                                src_letter_range, img_letter_range,
                                [1] * 3, [0] * 3,
                                self.color, self.color,
                                begin_time=begin_time,
                                transition_time=np.minimum(OBJECT_APPEARANCE_TIME, transition_time))

        if morphing:
            super().perform_morphing()

    def update_value(self, value_frame_function, begin_time, transition_time=OBJECT_APPEARANCE_TIME, location=None):
        if location is None:
            location = ibpy.get_location(self)
        displayed_value = self.displayed_value
        updates = [[self.value, begin_time * FRAME_RATE]]
        for frame in range(int(begin_time * FRAME_RATE), int((begin_time + transition_time) * FRAME_RATE + 1)):
            value = value_frame_function(frame)
            value = np.round(value * self.p) / self.p
            if value != displayed_value:
                updates.append([value, frame])
                displayed_value = value

        for i in range(1, len(updates)):
            prev_val = updates[i - 1][0]
            val = updates[i][0]
            frame = updates[i][1]
            frame_duration = updates[i][1] - updates[i - 1][1]
            frame_duration = np.maximum(1,
                                        0.5 * frame_duration)  # let the transition only take part in half of the transition time
            target = SimpleTexBObject(self.prefix + self.number2string(val) + self.suffix, location=location,
                                      **self.kwargs)
            src_letter_range = [len(self.prefix), len(self.prefix) + len(self.number2string(prev_val))]
            img_letter_range = [len(self.prefix), len(self.prefix) + len(self.number2string(val))]
            self.add_to_morph_chain(target,
                                    src_letter_range, img_letter_range,
                                    [1] * 3, [0] * 3,
                                    self.color, self.color,
                                    begin_time=frame / FRAME_RATE,
                                    transition_time=np.minimum(OBJECT_APPEARANCE_TIME, frame_duration / FRAME_RATE))

        super().perform_morphing()

    def number2string(self, val):
        if self.signed:
            sgn = '+'
        else:
            sgn = ''
        if val >= 0:
            s = sgn + str(np.abs(val))
        else:
            s = str(val)
        return s


def signed_string(x, signed,digits=0):
    if x >= 0 and signed:
        out = "+"
    else:
        out = ""

    fstring = "%."+str(digits)+"f"
    out += str(fstring %rounded_number(x,digits))
    return out


def rounded_number(x,digits=0):
    if digits==0:
        return np.round(x)
    return np.round(10**digits*x)/10**digits


class DigitalRange(BObject):
    """
    This Object can morph between different values
    The morphing through all values is established during the construction

    later on the values are changed by selecting shape parameters

    example:

    dim  = DigitalRange(list(range(-3, 4)), digits=0, aligned="center", signed=True, color="example")

    """

    def __init__(self, values, **kwargs):
        self.kwargs = kwargs
        self.values = values
        self.digits = self.get_from_kwargs('digits',0)
        self.initial_value=self.get_from_kwargs('initial_value',values[0])
        name = self.get_from_kwargs('name','DigitalRange')
        signed = self.get_from_kwargs('signed', True)
        # extract location and rotation information, otherwise it will be passed to every part individually as welll
        location = self.get_from_kwargs('location',Vector())
        rotation_euler = self.get_from_kwargs('rotation_euler',Vector())
        scale = self.get_from_kwargs('scale',Vector([1,1,1]))
        sorting = self.get_from_kwargs('sorting','BY_LENGTH')
        self.prefix = self.get_from_kwargs('prefix',"")
        self.suffix = self.get_from_kwargs('suffix',"")
        if sorting=='BY_LENGTH':
            self.values.sort(key=lambda x: len(signed_string(x, signed=signed,digits =self.digits)), reverse=True)
        elif sorting=='DESC':
            self.values.sort(key=lambda x:-x)
        elif sorting=='ASC':
            self.values.sort()
        # create all TexBObjects
        # override the default rotation of the SimpleTexBObject class by setting rotation_euler to zero
        self.bobs = [SimpleTexBObject(self.prefix+signed_string(v, digits=self.digits,signed=signed)+self.suffix,rotation_euler=Vector(), **kwargs) for v in self.values]
        super().__init__(children=self.bobs,name=name,location =location,rotation_euler=rotation_euler, scale =scale, **kwargs)
        self.first = self.b_children[0]
        self.start = True
        self.dict = {np.round(10**self.digits*v)/10**self.digits:i for i,v in enumerate(self.values)}
        self.current_value = None

    def write(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        # only link and appear container of the letters, make sure that the children are not visible and linked
        self.appear(begin_time=begin_time,transition_time=0,children=False,recursively=False)
        self.current_value = self.initial_value
        self.bobs[0].write(begin_time=begin_time,transition_time=transition_time,**kwargs)
        return begin_time+transition_time


    def show(self, from_value,to_value=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):

        if self.start:
            self.start=False
            # link all other words and make them invisible
            for i in range(1,len(self.bobs)):
                bobj = self.bobs[i]
                ibpy.link(bobj)
            self.morph()
            for i in range(1,len(self.bobs)):
                bobj = self.bobs[i]
                ibpy.hide(bobj)

        if to_value is None:
            to_value=from_value

        val0 = self.dict[rounded_number(from_value,self.digits)]*10 #times 10 because the time key is changed by 10 for each morph
        val1 = self.dict[rounded_number(to_value,self.digits)]*10

        for l in self.first.letters:
            ibpy.set_shape_key_eval_time(l, val0, begin_time * FRAME_RATE)
            ibpy.set_shape_key_eval_time(l, val1, (begin_time + transition_time) * FRAME_RATE)
        self.current_value = to_value
        return begin_time+transition_time

    def morph(self):
        first = self.b_children[0]
        for i in range(1, len(self.b_children)):
            first.replace(self.b_children[i], begin_time=None, morphing=False, keep_color=True,in_place=True)
        first.perform_morphing()

    def scale(self,scale = 1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        self.show(from_value=self.current_value,to_value=scale*self.current_value,begin_time=begin_time,
                  transition_time=transition_time)

    def shift(self, shift=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.show(from_value=self.current_value, to_value=self.current_value+shift, begin_time=begin_time,
                  transition_time=transition_time)

    def get_letters(self):
        return self.bobs[0].letters