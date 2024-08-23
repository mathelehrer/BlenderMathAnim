import numpy as np
import scipy

from mathematics.mathematica.mathematica import identity_matrix, tensor_product, dot
from mathematics.zeros import chop


class E8Lattice:
    """
    This class computes the geometric properties of the exceptional root lattice E8


    """
    def __init__(self):
        self.simple_roots=[
        [1, -1, 0, 0, 0, 0, 0, 0],
         [0, 1, -1, 0, 0, 0, 0, 0],
         [0, 0, 1, -1, 0, 0, 0, 0],
         [0, 0, 0, 1, -1, 0, 0, 0],
         [0, 0, 0, 0, 1, -1, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0],
         [-1 / 2, -1 / 2, -1 / 2, -1 / 2, -1 / 2, -1 / 2, -1 / 2, -1 / 2],
         [0, 0, 0, 0, 0, 1, -1, 0]
        ]

        self.reflections = [identity_matrix(8)-2/dot(root,root)*tensor_product(root,root) for root in self.simple_roots]
        self.generate_roots()
    def generate_roots(self):
        # add simple roots to the set of roots
        self.roots = [root for root in self.simple_roots]
        # create a list of hash codes
        hashes = set([hash(tuple(root)) for root in self.roots])
        # start with simple roots as seed for the generation of new roots
        old_roots = [root for root in self.simple_roots]
        while len(old_roots)>0:
            # generate new roots from the reflections of the simple root
            new_roots = [reflection.dot(np.array(root)) for root in old_roots for reflection in self.reflections]
            # reset old roots
            old_roots = []
            for root in new_roots:
                hashcode = hash(tuple(root))
                if hashcode not in hashes:
                    hashes.add(hashcode)
                    self.roots.append(root)
                    old_roots.append(root)
    def coxeter_element(self):
        """
        :return:

        compute the coxeter element
        >>> a = E8Lattice()
        >>> a.coxeter_element()
        array([[ 0.25,  0.25,  0.25,  0.25,  0.25, -0.75,  0.25,  0.25],
               [ 0.75, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
               [-0.25,  0.75, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
               [-0.25, -0.25,  0.75, -0.25, -0.25, -0.25, -0.25, -0.25],
               [-0.25, -0.25, -0.25,  0.75, -0.25, -0.25, -0.25, -0.25],
               [-0.25, -0.25, -0.25, -0.25,  0.75, -0.25, -0.25, -0.25],
               [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25, -0.75,  0.25],
               [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.75]])

        """
        coxeter = identity_matrix(8)
        for reflection in self.reflections:
            coxeter = coxeter.dot(reflection)

        return coxeter

    def coxeter_plane(self):
        """
        returns the projection matrix to the Coxeter plane
        >>> a = E8Lattice()
        >>> a.coxeter_plane()
        [[0.0, -0.129204, -0.209057, -0.236068, -0.209057, -0.129204, 0.0, 1.0], [-0.726543, -0.502754, -0.256993, 0, 0.256993, 0.502754, -0.105104, 0]]

        """
        coxeter_element= self.coxeter_element()
        eigvals, eigvecs = scipy.linalg.eig(coxeter_element)
        lowest = (-1) ** (1 / 15)


        u = [0., -0.129204, -0.209057, -0.236068, -0.209057, -0.129204, 0., 1.]
        v = [-0.726543, -0.502754, -0.256993, 0, 0.256993, 0.502754, -0.105104, 0]

        return [u,v]


if __name__ == '__main__':
    e8 = E8()