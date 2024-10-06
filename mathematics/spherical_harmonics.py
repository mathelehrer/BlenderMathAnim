from sympy import legendre, Symbol, assoc_legendre, Ynm, simplify, expand_complex, exp, conjugate, I, arg, factorial, \
    sqrt

z = Symbol("z")
print(expand_complex(exp(z)))


# derive the analytic expressions for the spherical harmonics

class Legendre:
    def __init__(self, order=0, var="x"):
        """
        Generates the Legendre Polynomial
        :param var: variable symbol
        :param order: degree of polynomial
        """
        x = Symbol(var)
        self.poly = legendre(order, var)

    def __str__(self):
        """
        :return: String representation of the polynomial
        >>> str(Legendre("x",5))
        '63*x**5/8 - 35*x**3/4 + 15*x/8'
        """
        return str(self.poly)


class AssociatedLegendre:
    def __init__(self, l=0, m=0, var="x"):
        """
        Generates the  associated Legendre Polynomial P^m_l
        :param var: variable symbol
        :param l: angular momentum quantum number
        :param m: magnetic quantum number
        """
        x = Symbol(var)
        self.poly = assoc_legendre(l, m, x)

    def __str__(self):
        """
        :return: String representation of the polynomial
        >>> str(AssociatedLegendre(5,3,"x"))
        '-(1 - x**2)**(3/2)*(945*x**2/2 - 105/2)'
        >>> str(AssociatedLegendre(5,3,"x").poly)
        '-3*(1 - x**2)**(3/2)*(945*x**2/2 - 105/2)'
        """
        return str(self.poly)


class SphericalHarmonics:
    def __init__(self, l=0, m=0, lat="theta", lon="phi"):
        """
        Generates the spherical harmonic Y^m_l(theta,phi)
        :param var: variable symbol
        :param order: degree of polynomial
        """
        theta = Symbol(lat, real=True)
        phi = Symbol(lon, real=True)
        self.poly = Ynm(l, m, theta, phi).expand(func=True)

    def __str__(self):
        """
        :return: String representation of the polynomial
        >>> str(SphericalHarmonics(5,3,"theta","phi"))
        'sqrt(385)*(1 - 9*cos(theta)**2)*exp(3*I*phi)*sin(theta)**3/(32*sqrt(pi))'
        """
        return str(self.poly)

    def real(self):
        """
        :return: the real part of the spherical harmonic
        >>> str(SphericalHarmonics(5,3,"theta","phi").real())
        'sqrt(385)*(1 - 9*cos(theta)**2)*sin(theta)**3*cos(3*phi)/(32*sqrt(pi))'
        """
        return expand_complex((self.poly + self.poly.conjugate()) / 2, deep=True)

    def imag(self):
        """
        :return: the real part of the spherical harmonic
        >>> str(SphericalHarmonics(5,3,"theta","phi").imag())
        'sqrt(385)*(1 - 9*cos(theta)**2)*sin(3*phi)*sin(theta)**3/(32*sqrt(pi))'
        """
        return expand_complex((self.poly - self.poly.conjugate()) / 2 / I, deep=True)

if __name__ == '__main__':
    l=5
    m=3
    print(sqrt((2*l+1)*factorial(l-m)/factorial(l+m)))