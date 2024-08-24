import numpy as np
from video_apollonian.indras_utils.circle import IndraCircle


def flatten(matrix):
    return [v for row in matrix for v in row]


def fixed_point_of(m):
    '''
    return the attractive fixed point of a matrix
    :param m:
    :return:
    '''
    a = m[0][0]
    b = m[0][1]
    c = m[1][0]
    d = m[1][1]
    z1 = 1 / 2 / c * ((a - d) + np.sqrt((a - d) ** 2 + 4 * c * b))
    z2 = 1 / 2 / c * ((a - d) - cx_sqrt((a - d) ** 2 + 4 * c * b))

    # check for attractiveness
    z = z1 * 1.1
    z_img = moebius_on_point(m, moebius_on_point(m, moebius_on_point(m, moebius_on_point(m, z))))
    if np.abs(z - z_img) < 0.1:
        return z1
    return z2


def cx_sqrt(z):
    x = np.real(z)
    y = np.imag(z)
    u2 = 0.5 * (x + np.sqrt(x * x + y * y))

    # u2 is zero when z=-|x|, then the square root is purely imaginary
    if u2 == 0:
        return 1j * np.sqrt(np.abs(x))
    else:
        u = np.sqrt(u2)
        v = y / 2 / u
        return u + 1j * v


def moebius_on_point(m, z):
    """
    :param m: matrix representing a Moebius transformation
    :param z: complex number
    :return:
    """
    if z == np.inf:
        if m[1][0] != 0:
            return m[0][0] / m[1][0]
        else:
            return np.inf
    else:
        den = m[1][0] * z + m[1][1]
        num = m[0][0] * z + m[0][1]
        if den == 0:
            return np.inf
        else:
            return num / den


def moebius_on_circle(m, circle):
    if circle.c == np.inf:
        # convert line to circle
        return IndraCircle.circle_from_three_points(
            moebius_on_point(m, circle.p1),
            moebius_on_point(m, circle.p2),
            moebius_on_point(m, np.inf),
        )
    else:
        if m[1][0] != 0:
            denominator = np.conj(m[1][1] / m[1][0] + circle.c)
            if np.abs(denominator) != 0:
                z = circle.c - circle.r ** 2 / denominator
                if np.abs(m[1][0] * z + m[1][1]) != 0:
                    cen = moebius_on_point(m, z)
                    rad = np.abs(cen - moebius_on_point(m, circle.c + circle.r))
                    return IndraCircle(cen, rad)
                else:
                    return IndraCircle(np.inf, np.inf, p1=moebius_on_point(m, circle.c + circle.r),
                                       p2=moebius_on_point(m, circle.c + 1j * circle.r))
            else:
                cen = m[0][0] / m[1][0]
                rad = np.abs(cen - moebius_on_point(m, circle.c + circle.r))
                return IndraCircle(cen, rad)
        else:
            # just a shift and scaling of the circle
            return IndraCircle((m[0][0]*circle.c + m[0][1]) /m[1][1] , circle.r * np.abs(m[0][0] / m[1][1]))
