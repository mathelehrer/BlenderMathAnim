from itertools import combinations

import numpy as np
from numpy.linalg import matrix_rank

from interface.ibpy import Vector
from mathematics.mathematica.mathematica import partition, random_points, find_plane_equation, vector_sum


class Polytope:
    """Container for the structural data of a polytope"""
    def __init__(self,dim,vertices):
        self.dim=dim # dimension of the polytope
        self.structure= {0: np.array(vertices)} #set of points
        self.center = vector_sum([Vector(v) for v in vertices])/len(vertices)
        self.name="Polytope"

        if len(vertices) < dim + 1:
            raise Exception("It is not possible to construct a polytope in " + str(dim) + " dimensions with " + str(
                len(vertices)) + " vertices.")
        if matrix_rank(vertices) != dim:
            raise Exception("The given vertices do not define an " + str(dim) + "-dimensional polytope")

    def __str__(self):
        return self.name+" of dimension "+str(self.dim)+" with "+str(len(self.structure[0]))+" vertices with center of mass at "+str(tuple(self.center))


class Simplex(Polytope):
    def __init__(self,dim,vertices):
        super().__init__(dim,vertices)
        face_indices = list(combinations(range(len(vertices)),dim))

        # check for too many vertices. The case for too few vertices is captured in the super constructor.
        if len(vertices) > dim + 1:
            raise Exception("It is not possible to construct a simplex in " + str(dim) + " dimensions with " + str(
                len(vertices)) + " vertices.")

        equations = [find_plane_equation([Vector(vertices[i]) for i in inds],self.center) for inds in face_indices]
        faces ={}

        for indices,equation in zip(face_indices,equations):
            faces[indices]=equation
        self.structure[dim-1]=faces

        self.name="Simplex"


if __name__ == '__main__':
    Simplex(3,random_points(3,4,10))