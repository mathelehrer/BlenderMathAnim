import numpy as np


class IndraCircle:
    """
    This class is a container for a circle, which is a geometric object in the complex plane
    :param center:
    :param radius:
    """

    def __init__(self, center, radius, p1=None, p2=None):
        """
        defines a circle in the complex plane
        :param center:
        :param radius:
        """
        self.r = radius
        self.c = center
        if p1 is None:
            self.p1 = self.c + self.r
        else:
            self.p1 = p1
        if p2 is None:
            self.p2 = self.c - self.r
        else:
            self.p2 = p2

    @classmethod
    def circle_from_three_points(cls,x, y, z):
        w = (z - x) / (y - x)
        if np.isclose(np.imag(w), 0):
            # circle is a line
            return IndraCircle(np.inf, np.inf, x, y)
        else:
            c = (y - x) * (w - np.conj(w) * w) / (2j * w.imag) + x
            r = np.abs(x - c)
            return IndraCircle(c, r)

    def area(self):
        return np.pi * self.r ** 2

    def circumference(self):
        return np.pi * self.r * 2

    def __str__(self):
        return "c=" + str(self.c) + " r=" + str(self.r)

    def __repr__(self):
        return "IndraCircle(" + str(self.c) + "," + str(self.r) + ")"


def random_circle()->IndraCircle:
    r = np.random.random() * 10 - 5
    c = np.random.random() * 10 - 5 + 1j * (np.random.random() * 10 - 5)
    return IndraCircle(c, r)
