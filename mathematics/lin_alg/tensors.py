import itertools

import numpy as np
from sympy.combinatorics import Permutation


def epsilon(rank: int = 4) -> np.ndarray:
    """
    Computes completely antisymmetric tensor for the computation of normals
    """
    n = rank ** rank
    comps = np.array([0] * n)
    comps.shape = (rank,) * rank

    permutations = list(itertools.permutations(range(rank)))
    for permutation in permutations:
        p = Permutation(permutation)
        comps[permutation] = p.signature()
    return comps