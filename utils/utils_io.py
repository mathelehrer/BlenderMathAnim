import os.path

import numpy as np

import utils
from utils.constants import DATA_DIR
from sympy.parsing import mathematica as m

from utils.utils import z2vec, to_vector


def parse(text):
    '''
    converts a mathematica list of complex numbers read from a file into  a list of python complex numbers
    :param text:
    :return:
    '''
    if '+' in text:
        # parse plus operations
        parts = text.split('+')
        return parse(parts[0]) + parse(parts[1])
    elif ' - ' in text:  # it's important that the spaces differentiate between a minus operator and a minus sign
        parts = text.split(' - ')
        return parse(parts[0]) - parse(parts[1])
    elif '/' in text:
        parts = text.split('/')
        return parse(parts[0]) / parse(parts[1])
    elif '(' and ')' in text:
        text = text.strip()
        text = text[1:len(text) - 1]
        return parse(text)
    elif '*' in text:
        parts = text.split('*')
        return parse(parts[0]) * parse(parts[1])
    elif 'I' in text:
        text = text.strip()
        if text == 'I':
            return 1j
        elif text == '-I':
            return -1j
        else:
            raise 'more complex imaginary part'
    else:
        return float(text)


def read_data(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        contents = f.read()

    str_data = contents.split(',')
    data = [parse(dat) for dat in str_data]
    return data


def read_complex_data(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        contents = f.read()

    str_data = contents.split(',')
    data = []
    for i, dat in enumerate(str_data):
        try:
            data.append(complex(dat))
        except:
            raise "something wrong with " + dat
    return data

def transform(z):
    '''
    simple scaling, translation and rotation
    :param z:
    :return:
    '''
    return np.conj(1j * z / 2 - 1j / 2)

def function_from_complex_list(lst, t, scale=1, average=1, shift=[0, 0, 0]):
    '''
    convert the values in a list into a function

    :param lst: the sequence of complex numbers
    :param t: the parameter ranging from 0 to 1
    :return:
    '''

    n = len(lst)
    i = np.round(t * n)
    i %= n
    if average == 1:
        val = transform(lst[int(i)]) * scale
    else:
        val = 0
        for j in range(average):
            val += transform(lst[int((i + j) % n)])
        val /= average
        val *= scale
    return z2vec(val) + to_vector(shift)

