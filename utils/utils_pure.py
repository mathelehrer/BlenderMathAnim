from mathutils import Vector


def to_vector(z):
    if z is None:
        return z
    if not isinstance(z, Vector):
        return Vector(z)
    else:
        return z
