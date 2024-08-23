import numpy as np

from interface import ibpy
from objects.bobject import BObject
from objects.tex_bobject import SimpleTexBObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE


class UpdateableTextWithNumbers(SimpleTexBObject):
    def __init__(self, string_function, value_function, **kwargs):
        self.value_function = value_function
        self.string_function = string_function
        self.kwargs = kwargs
        self.number_of_digits = self.get_from_kwargs('number_of_digits', 0)
        self.p = np.power(10, self.number_of_digits)
        self.displayed_value = np.round(np.multiply(self.value_function(0) , self.p))/ self.p
        super().__init__(string_function(*number2string(self.displayed_value)), **kwargs)

    def update_value(self, begin_time, transition_time=OBJECT_APPEARANCE_TIME, location=None):
        if location is None:
            location = ibpy.get_location(self)
        displayed_value = self.displayed_value
        updates = [[displayed_value, begin_time * FRAME_RATE]]
        offset = begin_time * FRAME_RATE
        for frame in range(begin_time * FRAME_RATE, (begin_time + transition_time) * FRAME_RATE + 1):
            value = self.value_function(frame - offset)
            value = np.round(np.multiply(value,self.p)) / self.p
            if isinstance(value, np.float64):
                if value != displayed_value:
                    updates.append([value, frame])
                    displayed_value = value
            else:
                change = False
                for x, y in zip(value, displayed_value):
                    if x != y:
                        change = True
                        break
                if change:
                    updates.append([value, frame])
                    displayed_value = value

        for i in range(1, len(updates)):
            val = updates[i][0]
            frame = updates[i][1]
            frame_duration = updates[i][1] - updates[i - 1][1]
            target = SimpleTexBObject(self.string_function(*number2string(val)), location=location, **self.kwargs)
            self.add_to_morph_chain(target,
                                    begin_time=frame / FRAME_RATE,
                                    transition_time=np.minimum(OBJECT_APPEARANCE_TIME, frame_duration / FRAME_RATE))

        super().perform_morphing()


def number2string(*val):
    str_val = []

    if isinstance(*val, np.float64):
        list_values = [val[0]]
    else:
        list_values = list(*val)
    for v in list_values:
        if v >= 0:
            s = "+" + str(np.abs(v))
        else:
            s = str(v)
        str_val.append(s)
    return str_val
