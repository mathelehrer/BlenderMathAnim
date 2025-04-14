"""
This file provides python functions that emulate some mathematica functions.
They are not symbolic but restricted to work with numbers.

"""
from itertools import combinations

import numpy as np
import sympy
from numpy.linalg import matrix_rank, solve
from numpy.linalg import LinAlgError
from scipy.spatial import ConvexHull

from interface.ibpy import Vector

def factorial(n:int)->int:
    """
    Computes the factorial of an integer
    >>> factorial(8)
    40320
    """
    if n==0 or n==1:
        return 1
    return n*factorial(n-1)

def choose(lst,choice):
    '''
    >>> choose(list(range(5)),2)
    [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    '''
    return  [list(i) for i in combinations(lst, choice)]

def tuples(lst, dim):
    """
    Creates a list of all tuples of dimension dim with the elements of the list
    :param lst: constains the elements of the tuples
    :param dim: the dimension of the tuples
    :return:
    >>> tuples([0.5, -0.5], 3)
    [(0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5)]
    """

    if dim == 1:
        return [tuple([a]) for a in lst]
    else:
        return [a + tuple([b]) for a in tuples(lst, dim - 1) for b in lst]

def partition(lst, size, step, wrap=0):
    """
    Partition a list into sub-lists of length size.
    :param lst: the list that is partitioned.
    :param size: the length of each sub-list
    :param step: the step-size
    :param wrap: wrapped to the beginning
    :return:
    >>> partition([1,2,3],2,1)
    [[1, 2], [2, 3]]
    >>> partition([1,2,3],2,1,1)
    [[1, 2], [2, 3], [3, 1]]
    """
    tmp = lst + lst[0:wrap]
    return [tmp[i:(i + size)] for i in range(0, len(tmp) - size + 1, step)]

def random_points(dim=3, n=10, domain=10, seed=None):
    """
    returns a list of tuples of random points
    :param dim:
    :param n:
    :param domain:
    :param seed:
    :return:
    >>> random_points(dim=1,n=10,domain=10,seed=1234)
    [(-6.169610992422154,), (2.4421754207966373,), (-1.2454452198577108,), (5.707171674275385,), (5.599516162376069,), (-4.548147894347167,), (-4.470714897138066,), (6.0374435507003845,), (9.162787073674103,), (7.518652694841894,)]
    >>> random_points(dim=3,n=3,domain=10,seed=1234)
    [(-6.169610992422154, 5.707171674275385, -4.470714897138066), (2.4421754207966373, 5.599516162376069, 6.0374435507003845), (-1.2454452198577108, -4.548147894347167, 9.162787073674103)]
    """
    if seed is not None:
        np.random.seed(seed)
    if dim == 1:
        return [tuple([-domain + np.random.random() * 2 * domain]) for z in range(n)]
    else:
        return [t + tuple([-domain + np.random.random() * 2 * domain]) for t in
                random_points(dim - 1, n=n, domain=domain, seed=seed)]

def unit_tuples(dim):
    return [tuple(v) for v in np.identity(dim)]

def identity_matrix(dim,unit=1,zero=0):
    """
    computes the identity matrix for a given dimensions
    :param dim:
    :return:
    >>> [list(r) for r in identity_matrix(3)]
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """
    if zero==0:
        return np.identity(dim)
    result = []
    for row in np.identity(dim):
        res_row = []
        for c in row:
            if c==1:
                res_row.append(unit)
            else:
                res_row.append(zero)
        result.append(res_row)
    return np.array(result)

def tensor_product(a,b):
    """
    computes the tensor product of two vectors a and b
    :param a:
    :param b:
    :return:

    >>> tensor_product([1,2,3],[4,5])
    array([[ 4,  5],
           [ 8, 10],
           [12, 15]])
    """
    rows = len(a)
    cols = len(b)
    return np.array([[r*c for c in b] for r in a])

def dot(a,b,zero=0):
    """
    computes the dot product between the vectors a and b
    :param a:
    :param b:
    :return:
    >>> a = [1,2,3]
    >>> b = [4,5,6]
    >>> dot(a,b)
    32

    """
    if len(a)!=len(b):
        raise "Cannot take dot product of vectors with unequal length"

    if zero == 0:
        return sum([x*y for x,y in zip(a,b)])
    else:
        res = zero
        for x,y in zip(a,b):
            res+=x*y
        return res

def negative_unit_tuples(dim):
    return [tuple(-v) for v in np.identity(dim)]

def unit_vectors(dim):
    return [Vector(v) for v in unit_tuples(dim)]

def vector_sum(vectors):
    """
    sums a list of vectors
    :param vectors:
    :return:

    >>> vector_sum([v for v in unit_vectors(5)])
    Vector((1.0, 1.0, 1.0, 1.0, 1.0))
    """
    return sum([vectors[i] for i in range(1, len(vectors))], start=vectors[0])

def mean(vectors):
    return vector_sum(vectors)/len(vectors)

def find_normal(vertices):
    """
    computes the normal of a co-dimension one subspace from a given set of vectors. It takes into account that the subspace might go through the origin
    :param vertices:
    :return:

    >>> find_normal(random_points(3,n=3,domain=10,seed=1234))
    Vector((-0.6961142420768738, 0.4300248622894287, 0.5748944282531738))
    >>> Vector(np.sqrt(3)*find_normal(unit_vectors(3)))
    Vector((1.0, 1.0, 1.0))
    >>> find_normal([Vector()]+unit_vectors(3)[0:2]) # singular case for plane through origin
    Vector((0.0, 0.0, -1.0))
    """
    dim = len(vertices[0])
    rank = matrix_rank(vertices)
    rhs = vector_sum([v for v in unit_vectors(dim)])  # vector with all ones

    if rank == dim:
        n = Vector(solve(vertices, rhs))
    else:
        # find linearly independent rows
        matrix = np.array(vertices)
        _, inds = sympy.Matrix(matrix).T.rref()
        matrix = matrix[list(inds)]
        # use each of the columns as right-hand side until a solution has been found
        success = False
        column = 0
        while not success:
            try:
                selector = [x for x in range(matrix.shape[1]) if x != column]
                # return(matrix,matrix[:,selector],matrix[:,column])
                sol = solve(matrix[:,selector], matrix[:,column])
                success = True
                n = Vector(tuple(sol[0:column]) + tuple([-1]) + tuple(sol[column + 1:dim]))
            except LinAlgError:
                success = False
            column += 1
    n.normalize()
    return n

def find_plane_equation(vertices,center):
    """
    It returns the normal and the right-hand side of the plane equations n.x=d.
    The normal vector is normalized and pointing away from the center.
    :param vertices: vertices of the co-dimension one plane
    :param center: center of the polytope, used to define the direction of the normal
    :return:
    Test two faces of a cube
    >>> find_plane_equation([Vector([1,0,0]),Vector([1,1,0]),Vector([1,1,1])],center=Vector([0.5,0.5,0.5]))
    (Vector((1.0, 0.0, 0.0)), 1.0)
    >>> find_plane_equation([Vector([0,0,0]),Vector([0,1,0]),Vector([1,0,0])],center=Vector([0.5,0.5,0.5]))
    (Vector((0.0, 0.0, -1.0)), 0.0)
    >>> find_plane_equation([Vector([1,0,0,0]),Vector([1,1,0,0]),Vector([1,1,1,0]),Vector([1,0,1,0])],center=Vector([0.5,0.5,0.5,0.5]))
    (Vector((0.0, 0.0, 0.0, -1.0)), 0.0)
    >>> find_plane_equation([Vector([1,0,0,0]),Vector([1,1,0,0]),Vector([1,1,1,0]),Vector([1,1,1,1])],center=Vector([0.5,0.5,0.5,0.5]))
    (Vector((1.0, 0.0, 0.0, 0.0)), 1.0)
    >>> find_plane_equation([Vector(p) for p in random_points(5,5,10,seed=1234)],center=Vector((0,0,0,0,0)))
    (Vector((0.5028305053710938, -0.04429202899336815, -0.7388026714324951, -0.020270254462957382, -0.44604867696762085)), 0.3806169293820858)
    """
    face_center = vector_sum(vertices)
    normal = find_normal(vertices)
    if (normal.dot(face_center-center)>0):
        return normal,normal.dot(vertices[0])
    else:
        return -1*normal,-normal.dot(vertices[0])

def convex_hull(vertices):
    return ConvexHull([Vector(v) for v in vertices])

def find_closest(point_list,point):
    """
    find the nearest point from a list of points
    :param point_list:
    :param point:
    :return:

    >>> find_closest(random_points(dim=3,n=100,domain=1,seed=1234),Vector())
    Vector((0.0737563893198967, 0.20866800844669342, 0.21335043013095856))
    """

    point = Vector(point)
    closest = Vector(point_list[0])
    dist = (closest-point).dot(closest-point)

    for i in range(1,len(point_list)):
        next = Vector(point_list[i])
        next_dist = (next-point).dot(next-point)
        if next_dist<dist:
            closest = next
            dist=next_dist

    return closest


if __name__ == '__main__':
    import doctest

    doctest.testmode()
