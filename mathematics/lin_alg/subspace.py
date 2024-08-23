import numpy as np


class Subspace:
    def __init__(self, i, j, k, vertices, eps=1e-4):
        a = vertices[i]
        b = vertices[j]
        c = vertices[k]
        self.vertices = vertices
        self.indices = tuple(sorted((i, j, k)))
        self.EPS = eps
        self.normal = ((c - a).cross(b - a))
        length = self.normal.length
        if length < self.EPS:
            raise "Singular plane from vertices "+str({i,j,k})
        self.normal /= self.normal.length
        self.d = a.dot(self.normal)

    def __repr__(self):
        return f"{self.__class__.__name__}(indices = {self.indices})"

    def __hash__(self):
        return hash(self.indices)

    def __eq__(self, other):
        return self.indices == other.indices

    def __lt__(self, other):
        return self.indices < other.indices

    def __le__(self, other):
        return self.indices <= other.indices

    def __gt__(self, other):
        return self.indices > other.indices

    def __ge__(self, other):
        return self.indices >= other.indices

    def add(self, i):
        if i not in self.indices and self.is_point_of_plane(i):
            self.indices = tuple(sorted(self.indices + (i,)))

    def is_point_of_plane(self, i):
        if i in self.indices:
            return True
        else:
            return np.abs(self.vertices[i].dot(self.normal) - self.d) < self.EPS
