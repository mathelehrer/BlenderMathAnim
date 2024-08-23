class Polynomial(object):
    def __init__(self, coefficients):
        """
        construct a polynomial of the form
        coefficients = [2,-3,5]: 2-3x+5*x**2

        :param coefficients:
        """
        self.coefficients = coefficients
        self.EPS = 0.000001

    def almost_equal(self, d1, d2):
        return abs(d1 - d2) < self.EPS

    def first_non_zero_coefficient(self):
        for num in reversed(self.coefficients):
            if num != 0:
                return num

    def degree(self):
        return len(self.coefficients) - 1

    def coefficient(self, power):
        return self.coefficients[power]

    def eval(self, x):
        return sum([self.coefficients[power] * x ** power for power in range(self.degree() + 1)])

    def norm(self):
        return sum([c * c for c in self.coefficients])

    def to_function(self):
        """
        return polynomial as a lambda function
        :return:
        """
        return lambda x: self.eval(x)

    def copy(self):
        return Polynomial(self.coefficients.copy())

    def __neg__(self):
        return Polynomial([-c for c in self.coefficients])

    def __add__(self, other):
        new_coefficients = []
        for coefficients1, coefficients2 in zip(self.coefficients, other.coefficients):
            new_coefficients.append((coefficients1 + coefficients2))

        l1 = len(self.coefficients)
        l2 = len(other.coefficients)
        if l1 > l2:
            for i in range(l2, l1):
                new_coefficients.append(self.coefficients[i])
        elif l2 > l1:
            for i in range(l1, l2):
                new_coefficients.appendt(other.coefficients[i])

        return Polynomial(new_coefficients)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        d1 = self.degree()
        d2 = other.degree()

        new_coefficients = [0] * (d1 + d2 + 1)
        for i in range(d1 + 1):
            for j in range(d2 + 1):
                new_coefficients[i + j] += self.coefficients[i] * other.coefficients[j]

        return Polynomial(new_coefficients)

    def derivative(self):
        new_coefficients = []
        for i,c in enumerate(self.coefficients):
            if i>0:
                new_coefficients.append(i*c)
        return Polynomial(new_coefficients)

    def coeff_str(self,c,e):
        if abs(c)==1 and e>0:
            return ""
        if abs(c)==1 and e==0:
            return str(c)
        if abs(c)==0 and e>0:
            return ""
        if abs(c)==0 and e==0:
            return str(c)
        else:
            return str(abs(c))

    def signed(self, c):
        if c < 0:
            return str(c)
        else:
            return "+" + str(c)

    def only_neg_sign(self, c):
        if c<0:
            return "-"
        else:
            return ""

    def sign(self,c):
        if c<0:
            return "-"
        else:
            return "+"

    def signed_coeff_str(self,c,e):
        if abs(c)==1 and e>0:
            return self.sign(c)
        if abs(c)==1 and e==0:
            return self.signed(c)
        if abs(c)==0 and e>0:
            return ""
        else:
            return self.signed(c)

    def coeff_sign(self,c):
        if c<0:
            return "-"
        if c>0:
            return "+"
        else:
            return ""


    def __str__(self):
        out = ""
        for e, c in enumerate(self.coefficients):
            if e > 0 and c!=0:
                out += self.coeff_sign(c) + self.coeff_str(c,e) + "x**" + str(e)
            else:
                out += self.coeff_str(c,0)
        return out

    def to_tex(self, variable="x"):
        """build the string from right to left"""
        out = ""
        for e, c in enumerate(self.coefficients):
            if e==len(self.coefficients)-1 and c!=0:
                if abs(c)!=1:
                    out = str(c)+variable+"^{"+str(e)+"}"+out
                else:
                    out =self.only_neg_sign(c)+variable+"^{"+str(e)+"}"+out
            elif e > 1 and c != 0:
                out = self.signed_coeff_str(c, e) + variable + "^{" + str(e) + "}" + out
            elif e==1 and c!=0:
                out = self.signed_coeff_str(c, e) + variable  + out
            elif c != 0:
                out = self.coeff_sign(c) + self.coeff_str(c, 0)
        return out