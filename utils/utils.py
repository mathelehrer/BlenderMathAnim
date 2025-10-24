import functools
import inspect
import os
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
from interface.ibpy import select, link, Vector, delete, Quaternion

pi = np.pi


def get_from_dictionary(dictionary, string_list):
    objects = []
    for string in string_list:
        if string in dictionary:
            objects.append(dictionary[string])

    return objects


def retrieve_name(var):
    """
    This function can be used to create a dictionary of local variables
    An example can be found in
    experiments/variable_name_to_string.py
    bwm/scene_bwm.py
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def z2p(z):
    return [np.real(z), np.imag(z)]


def z2vec(z, z_dir=False):
    if z_dir:
        return Vector([np.real(z), 0, np.imag(z)])
    return Vector([np.real(z), np.imag(z), 0])


def p2z(p):
    return p[0] + 1j * p[1]


def re(z):
    return np.real(z)


def im(z):
    return np.imag(z)


def z2str(z):
    string = ''
    if np.real(z) == 0 and np.imag(z) == 0:
        return "0"
    if np.real(z) != 0:
        re = np.real(z)
        if re == np.round(re):
            string += str(int(re))
        else:
            string += str(re)
    if np.imag(z) == 0:
        return string
    if np.imag(z) > 0:
        string += '+'
    else:
        string += '-'
    if np.abs(np.imag(z)) == 1:
        string += 'i'
        return string
    im = np.abs(np.imag(z))
    if im == np.round(im):
        string += (str(int(im)) + 'i')
    else:
        string += (str(im) + 'i')
    return string


def vec_round(v, digits=0):
    return [round(c * 10 ** digits) / 10 ** digits for c in v]


def to_vector(z):
    if z is None:
        return z
    if not isinstance(z, Vector):
        return Vector(z)
    else:
        return z


def to_list(l):
    if isinstance(l, list):
        return l
    else:
        return [l]


def flatten(colors):
    colors_flat = [col for sublist in colors for col in sublist]
    return colors_flat

"""
    translated methods
"""


def quaternion_from_normal(normal):
    normal = to_vector(normal)
    normal.normalize()
    angle = np.arccos(normal.dot(Vector([0, 0, 1])))
    axis = Vector([0, 0, 1]).cross(normal)
    length = axis.length
    if length == 0:
        if normal.length>0:
            angle=0
            axis = Vector([0,0,1])
        else:
            raise "error: rotation axis not found"
    else:
        axis /= length
    quaternion = Quaternion([np.cos(angle / 2), *((axis * np.sin(angle / 2))[:])])
    return quaternion


def link_descendants(obj, unlink=False, top_level=True):
    # If children exist, link those too
    # Will break if imported children were linked in add_to_blender
    # (if their object name in blender is the same as the filename)

    if unlink and top_level:
        select(obj)
        # obj.select = True
    # obj_names=[x.name for x in bpy.data.objects]
    obj_names = []
    for child in obj.children:
        if not unlink:
            if child.name not in obj_names:
                link(child)
        else:
            child.select = True
        link_descendants(child, unlink=unlink, top_level=False)
    if unlink:
        delete()



def get_save_length(start, end):
    if isinstance(start, list):
        start = Vector(start)
    if isinstance(end, list):
        end = Vector(end)
    return (end - start).length


def get_rotation_quaternion_from_start_and_end(start, end):
    """
    For simplicity this works for an object that is of unit length directed in z-direction
    :param start:
    :param end:
    :return:
    """
    if isinstance(start, list):
        start = Vector(start)
    if isinstance(end, list):
        end = Vector(end)

    diff = to_vector(end - start)
    diff = diff.normalized()
    up = Vector((0, 0, 1))  # default orientation
    axis = up.cross(diff).normalized()
    if axis.length == 0:
        if diff.dot(up) > 0:
            return Quaternion()  # no rotation is needed
        else:
            return Quaternion([0, 0, 1, 0])  # rotation by 180 degrees
    angle = np.arccos(up.dot(diff))
    quaternion_axis = axis * np.sin(angle / 2)
    quaternion = Quaternion([np.cos(angle / 2), *quaternion_axis[:]])

    return quaternion




def finish_noise(error=False):
    if error:
        os.system('spd-say "your program has finished with errors"')
    else:
        os.system(
            'spd-say "your program has successfully finished"')  # https://stackoverflow.com/questions/16573051/python-sound-alarm-when-code-finishes


'''
Time measurement
'''
TIME_LIST = []
now = datetime.now()
TIME_LIST.append(now)
TIME_REPORT = []


def execute_and_time(message, *funcs):  # Not sure how this will work for more than one bobject that returns
    outputs = []
    for func in funcs:
        output = func
        if output is not None:
            outputs.append(output)

    local_now = datetime.datetime.now()
    TIME_LIST.append(local_now)  # Actually just records end time, not start and end
    # So reported value includes previous, seemingly untimed code
    diff = TIME_LIST[-1] - TIME_LIST[-2]
    TIME_REPORT.append([diff.total_seconds(), message])
    if not outputs:
        return
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def print_time_report():
    print()
    for line in TIME_REPORT:
        print(line[0], line[1])
    local_now = datetime.now()
    total = local_now - TIME_LIST[0]
    print(total.total_seconds(), "Total")


def add_lists_by_element(list1, list2, subtract=False):
    if len(list1) != len(list2):
        raise Warning("The lists aren't the same length")
    list3 = list(deepcopy(list2))
    if subtract:
        for i in range(len(list3)):
            list3[i] *= -1
    return list(map(sum, zip(list1, list3)))


def mult_lists_by_element(vec1, vec2, divide=False):
    vec3 = []
    if not divide:
        for x1, x2, in zip(vec1, vec2):
            vec3.append(x1 * x2)
    else:
        for x1, x2, in zip(vec1, vec2):
            vec3.append(x1 / x2)

    return vec3


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the bobject is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated bobject {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
