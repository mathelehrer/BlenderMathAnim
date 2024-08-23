import numpy as np


def lin_map(x,x_min,x_max,y_min,y_max):
    return y_min+(x-x_min)/(x_max-x_min)*(y_max-y_min)

def get_min_max_vector(vectors):
    """ find bounding box for a list of vectors"""
    mins =[np.inf,np.inf,np.inf]
    maxs =[-np.inf,-np.inf,-np.inf]

    for v in vectors:
        for i,comp in enumerate(v):
            if comp<mins[i]:
                mins[i]=comp
            if comp>maxs[i]:
                maxs[i]=comp

    return mins,maxs
def function_from_list(lst, mini, maxi, x):
    """
    convert the values in a list into a function with periodic boundary conditions

    :param lst: list with the values
    :param mini: the minimal x value
    :param maxi: the maximal x value
    :param x: the x value
    :return:

    my first doc test
    >>> function_from_list([1,2,3],0,1,0)
    1.0

    >>> function_from_list([1,2,3],0,1,1)
    3.0

    >>> function_from_list([1,2,3],0,1,0.5)
    2.0
    """

    n = len(lst)
    pos = (x - mini) / (maxi - mini) * (n - 1)

    int_part = int(np.floor(pos))
    frac_part = pos - int_part

    left = int_part % n
    right = (int_part + 1) % n

    return interpol(lst[left], lst[right], frac_part)


def interpol(min, max, t):
    return min + t * (max - min)


def regularized(expr, epsilon):
    if np.isclose(expr, 0):
        return epsilon
    else:
        return expr


class VectorTransformation:
    def __init__(self):
        pass

    def set_transformation_function(self, trafo):
        self.trafo = trafo

    def set_first_derivative_functions(self, first_derivatives):
        self.d_trafo = first_derivatives

    def set_second_derivative_functions(self, second_derivatives):
        self.d2_trafo = second_derivatives


class Koch:
    def __init__(self, a=0, b=1, iteration=1):
        lst = [a, b]
        self.r3 = np.sqrt(3)
        while iteration > 0:
            lst = self.intersect(lst)
            iteration -= 1

        self.points = lst
        # self.points.append(0.5+1j)

    def intersect(self, lst):
        new_lst = []
        for i in range(len(lst) - 1):
            a = lst[i]
            b = lst[i + 1]
            d = b - a
            one3 = a + d / 3
            two3 = a + 2 * d / 3
            one2 = a + d / 2
            perp = d / 3 * 1j * self.r3 / 2 + one2
            new_lst.append(a)
            new_lst.append(one3)
            new_lst.append(perp)
            new_lst.append(two3)
        new_lst.append(b)
        return new_lst


if __name__ == '__main__':
    print(Koch(iteration=2).points)
