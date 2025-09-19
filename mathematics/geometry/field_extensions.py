import itertools
from fractions import Fraction

import numpy as np
from mathutils import Vector
from sympy.combinatorics import Permutation


class QR5:
    def __init__(self,x:Fraction,y:Fraction):
        """
        >>> QR5(Fraction(1, 2), Fraction(3, 4))
        (1/2+3/4*r5)

        :param x:
        :param y:
        """
        self.x=x
        self.y=y


    @classmethod
    def parse(cls,string):
        """
        parse a string of the form "(a/b+c/d*r5)" into a QR5 object
        >>> QR5.parse("(1/2+3/4*r5)")
        (1/2+3/4*r5)
        >>> QR5.parse("1/2")
        1/2
        >>> QR5.parse("-1/2")
        -1/2
        >>> QR5.parse("0")
        0
        """
        # remove brackets
        if string[0]=="(":
            string=string[1:-1]
        if string[0]=="-":
            string=string[1:]
            sign1=-1
        else:
            sign1=1
        if "-" in string:
            parts = string.split("-")
            sign2=-1
        else:
            parts = string.split("+")
            sign2=1

        if len(parts)==1:
            if "/" in parts[0]:
                a,b=parts[0].split("/")
                return cls(Fraction(sign1*int(a),int(b)),Fraction(0,1))
            else:
                return cls(Fraction(sign1*int(parts[0]),1),Fraction(0,1))
        else:
            if "/" in parts[0]:
                a,b=parts[0].split("/")
            else:
                a=parts[0]
                b=1
            parts[1]=parts[1].replace("*r5","")
            if "/" in parts[1]:
                c,d=parts[1].split("/")
            else:
                c=parts[1]
                d=1
            return cls(Fraction(sign1*int(a),int(b)),Fraction(sign2*int(c),int(d)))



    @classmethod
    def from_integers(cls, a:int, b:int, c:int, d:int):
        """
        >>> QR5.from_integers(1,2,3,4)
        (1/2+3/4*r5)
        """

        return cls(Fraction(a,b), Fraction(c,d))

    def __str__(self):
        if self.y>0:
            return "("+str(self.x)+"+"+str(self.y)+"*r5)"
        elif self.y<0:
            return "("+str(self.x)+"-"+str(Fraction(np.abs(self.y.numerator),np.abs(self.y.denominator)))+"*r5)"
        else:
            return str(self.x)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        """
        >>> QR5(Fraction(1, 2), Fraction(3, 4))*QR5(Fraction(2,1), Fraction(4,3))
        (6+13/6*r5)

        >>> np.random.seed(1234)
        >>> z = QR5.random()
        >>> w = QR5.random()
        >>> z*w
        (70-26*r5)

        :param other:
        :return:
        """
        return QR5(self.x * other.x + 5 * self.y * other.y, self.x * other.y + self.y * other.x)

    def __add__(self,other):
        """
        >>> np.random.seed(1234)
        >>> z = QR5.random()
        >>> w = QR5.random()
        >>> z+w
        (1+11*r5)

        :param other:
        :return:
        """
        return QR5(self.x + other.x, self.y + other.y)

    def __sub__(self,other):
        """
        >>> np.random.seed(1234)
        >>> z = QR5.random()
        >>> w = QR5.random()
        >>> z-w
        (9+7*r5)

        :param other:
        :return:
        """
        return QR5(self.x - other.x, self.y - other.y)

    def conj(self):
        """
        >>> QR5(Fraction(1, 2), Fraction(3, 4)).conj()
        (1/2-3/4*r5)

        :return:
        """
        return QR5(self.x, -self.y)

    def norm(self):
        """
        >>> QR5(Fraction(1, 2), Fraction(3, 4)).norm()
        Fraction(-41, 16)

        :return:
        """
        return self.x**2-5*self.y**2

    def __neg__(self):
        """
        >>> -QR5.from_integers(-1,1,2,-3)
        (1+2/3*r5)

        :return:
        """
        return QR5(-self.x, -self.y)

    def __truediv__(self,other):
        """
        >>> np.random.seed(1234)
        >>> z = QR5.random()
        >>> w = QR5.random()
        >>> z/w
        (55/2+23/2*r5)

        >>> z = QR5.random()
        >>> w = QR5.random()
        >>> z/w*w,z
        ((5+7*r5), (5+7*r5))

        :param other:
        :return:
        """
        return QR5(1 / other.norm(), Fraction(0, 1))*(self * other.conj())

    @classmethod
    def random(cls,range=10):
        x=Fraction(np.random.randint(-range,range),1)
        y=Fraction(np.random.randint(-range,range),1)
        return QR5(x, y)

    def __eq__(self, other):
        return self.x==other.x and self.y==other.y

    def __hash__(self):
        """ custom hash function for QR5 objects
        >>> a = QR5(Fraction(1, 2), Fraction(3, 4))
        >>> hash(a)
        590899387183067792

        >>> b = QR5(Fraction(5, 6), Fraction(7, 8))
        >>> hash((a,b))
        1680000522697785275

        >>> M=FTensor([a,b])
        >>> hash(M)
        1680000522697785275

        """
        return hash((self.x.numerator,self.x.denominator,self.y.numerator,self.y.denominator))

    def real(self):
        return float(self.x)+float(self.y)*np.sqrt(5)

# the quaternions are constructed with the Cayley-Dickson construction
# we need complex numbers over QR5 and from these complex numbers the
# quaternion can be constructed as pairs of complex numbers

class FComplex:
    def __init__(self, re:QR5, im:QR5):
        self.re=re
        self.im=im

    def __str__(self):
        """
        >>> str(FComplex(QR5(Fraction(1, 2), Fraction(3, 4)), QR5(Fraction(2, 1), Fraction(4, 3))))
        '(1/2+3/4*r5)+(2+4/3*r5)*i'

        :return:
        """
        return str(self.re)+"+"+str(self.im)+"*i"

    def __repr__(self):
        return self.__str__()
    def __mul__(self, other):
        """
        >>> FComplex(QR5(Fraction(1, 2), Fraction(3, 4)), QR5(Fraction(5, 6), Fraction(7, 8))) * FComplex(QR5(Fraction(9, 8), Fraction(7, 6)), QR5(Fraction(5, 4), Fraction(3, 2)))
        (-8/3-11/12*r5)+(295/24+2099/576*r5)*i

        >>> u = FComplex.random(10)
        >>> v = FComplex.random(10)
        >>> w = FComplex.random(10)
        >>> w*(u+v)-w*u-w*v
        0+0*i

        :param other:
        :return:
        """
        return FComplex(self.re*other.re-self.im*other.im,self.re*other.im+self.im*other.re)

    def __add__(self,other):
        """
        >>> FComplex(QR5(Fraction(1, 2), Fraction(3, 4)), QR5(Fraction(5, 6), Fraction(7, 8))) + FComplex(QR5(Fraction(9, 8), Fraction(7, 6)), QR5(Fraction(5, 4), Fraction(3, 2)))
        (13/8+23/12*r5)+(25/12+19/8*r5)*i

        :param other:
        :return:
        """
        return FComplex(self.re+other.re,self.im+other.im)

    def __sub__(self,other):
        """
        >>> FComplex(QR5(Fraction(1, 2), Fraction(3, 4)), QR5(Fraction(5, 6), Fraction(7, 8))) - FComplex(QR5(Fraction(9, 8), Fraction(7, 6)), QR5(Fraction(5, 4), Fraction(3, 2)))
        (-5/8-5/12*r5)+(-5/12-5/8*r5)*i

        :param other:
        :return:
        """
        return FComplex(self.re-other.re,self.im-other.im)


    def conj(self):
        return FComplex(self.re,-self.im)


    def norm(self):
        """
        >>> FComplex(QR5(Fraction(1, 2), Fraction(3, 4)), QR5(Fraction(5, 6), Fraction(7, 8))).norm()
        Fraction(10998241, 331776)

        :return:
        """
        return (self*self.conj()).re.norm()

    def __neg__(self):
        return FComplex(-self.re,-self.im)

    def __truediv__(self,other):
        """
        >>> u = FComplex.random(10)
        >>> v = FComplex.random(10)
        >>> w = FComplex.random(10)
        >>> (u-v)/w-u/w+v/w
        0+0*i

        :param other:
        :return:
        """
        return FComplex(QR5(1/other.norm(),Fraction(0,1)),QR5.from_integers(0,1,0,1))*(self*other.conj())

    def __eq__(self, other):
        return self.re==other.re and self.im==other.im

    @classmethod
    def random(cls,range=10):
        x=QR5.random(range)
        y=QR5.random(range)
        return FComplex(x,y)

    def __hash__(self):
        return hash((self.re,self.im))

class FQuaternion:
    def __init__(self, a:FComplex, b:FComplex):
        self.a = a
        self.b = b

    def __str__(self):
        return str(self.a.re)+"+"+str(self.a.im)+"*i+"+str(self.b.re)+"*j"+"+"+str(self.b.im)+"*k"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_vector(cls,a:QR5,b:QR5,c:QR5,d:QR5):
        return FQuaternion(FComplex(a,b),FComplex(c,d))

    def __add__(self,other):
        return FQuaternion(self.a+other.a,self.b+other.b)

    def __sub__(self,other):
        return FQuaternion(self.a-other.a,self.b-other.b)

    def __mul__(self,other):
        """
        Caley-Dickson construction

        :param other:
        :return:
        """
        return FQuaternion(self.a*other.a-other.b.conj()*self.b,other.b*self.a+self.b*other.a.conj())

    def __neg__(self):
        return FQuaternion(-self.a,-self.b)

    def __eq__(self, other):
        """
        Check algebra of quaternions

        >>> one = QR5.from_integers(1,1,0,1)
        >>> zero = QR5.from_integers(0,1,0,1)
        >>> q_one=FQuaternion.from_vector(one,zero,zero,zero)
        >>> q_i=FQuaternion.from_vector(zero,one,zero,zero)
        >>> q_j=FQuaternion.from_vector(zero,zero,one,zero)
        >>> q_k=FQuaternion.from_vector(zero,zero,zero,one)
        >>> assert(q_i*q_j==q_j*q_i.conj())
        >>> assert(q_i*q_k==q_k*q_i.conj())
        >>> assert(q_j*q_k==q_k*q_j.conj())
        >>> assert(q_i*q_i==-q_one)
        >>> assert(q_j*q_j==-q_one)
        >>> assert(q_k*q_k==-q_one)

        :param other:
        :return:
        """
        return self.a==other.a and self.b==other.b

    def conj(self):
        return FQuaternion(self.a.conj(),-self.b)

    def __hash__(self):
        return hash((self.a,self.b))

    def norm(self):
        return (self*self.conj()).a.re

    def to_vector(self):
        return [self.a.re,self.a.im,self.b.re,self.b.im]

# after the generation of the 120 vertices of the 600 cells, we have to
# identify edges and faces and cells

# Once the 600 cells have been identified and the 4D - geometry is established,
# we need to find the normal vector for one
# cell of the 600 cell, which the entire polytope will be projected into

# For the computation of the normal vector, we compute the tri-vector from the
# tensor product of the three basis vectors and finally compute the dual vector
# To work with this machinery, we need to define a tensor class over the field of QR5

class FTensor:
    def __init__(self,components):
        components = np.array(components)
        self.components = components
        self.rank = self.get_rank()
        self.dims = self.get_dimensions()

    def get_dimensions(self):
        """
        >>> FTensor([1,2,3]).dims
        [3]

        >>> FTensor([[[1,2],[3,4]],[[5,6],[7,8]]]).dims
        [2, 2, 2]

        >>> FTensor(5).dims
        []

        :return:
        """
        return self._get_dimensions(self.components.tolist())

    def get_rank(self):
        """
        >>> FTensor([[1,2],[3,4]]).rank
        2
        >>> FTensor([[[1,2],[3,4]],[[5,6],[7,8]]]).rank
        3
        >>> FTensor(5).rank
        0

        :param components:
        :return:
        """
        return self._rank_rec(self.components)

    def _rank_rec(self,components):
        if isinstance(components,list):
            return self._rank_rec(components[0])+1
        elif isinstance(components,np.ndarray):
            return len(components.shape)
        else:
            return 0

    def _get_dimensions(self,components):
        if isinstance(components,list):
            result = self._get_dimensions(components[0])
            result.append(len(components))
            return result
        else:
            return list()

    def scale(self,alpha:QR5):
        """
        >>> np.random.seed(1234)
        >>> comps = np.array([QR5.random() for i in range(12)])
        >>> components = comps.reshape(3,4)

        >>> tensor = FTensor(components)
        >>> tensor.components.tolist()
        [[(5+9*r5), (-4+2*r5), (5+7*r5), (-1+1*r5)], [(2+6*r5), (-5+6*r5), (-1+5*r5), (8+6*r5)], [(2-5*r5), (-8-4*r5), (-7-3*r5), (1-10*r5)]]

        >>> tensor = tensor.scale(QR5(Fraction(2,1),Fraction(0,1)))
        >>> tensor.components.tolist()
        [[(10+18*r5), (-8+4*r5), (10+14*r5), (-2+2*r5)], [(4+12*r5), (-10+12*r5), (-2+10*r5), (16+12*r5)], [(4-10*r5), (-16-8*r5), (-14-6*r5), (2-20*r5)]]

        :param alpha:
        :return:
        """
        return FTensor(self.components*alpha)

    def __mul__(self,other):
        """
        tensor product between two tensors
        >>> np.random.seed(1234)
        >>> comps = np.array([QR5.random() for i in range(12)])
        >>> components = comps.reshape(3,4)
        >>> comps2 = np.array([QR5.random() for i in range(12)])
        >>> components2 = comps2.reshape(3,2,2)
        >>> product = FTensor(components)*FTensor(components2)
        >>> product.components.tolist()
        [[[[[(40-4*r5), (-285+19*r5)], [(365-27*r5), (-395-27*r5)]], [[(410+54*r5), (320+44*r5)], [(425+81*r5), (-15-27*r5)]], [[(185+29*r5), (170+78*r5)], [(40-80*r5), (290-10*r5)]]], [[[(14-6*r5), (-94+40*r5)], [(122-52*r5), (-98+40*r5)]], [[(86-34*r5), (66-26*r5)], [(74-28*r5), (12-6*r5)]], [[(36-14*r5), (2+2*r5)], [(60-28*r5), (90-38*r5)]]], [[[(30-2*r5), (-215+7*r5)], [(275-11*r5), (-305-31*r5)]], [[(320+52*r5), (250+42*r5)], [(335+73*r5), (-15-21*r5)]], [[(145+27*r5), (140+64*r5)], [(20-60*r5), 220]]], [[[(6-2*r5), (-41+13*r5)], [(53-17*r5), (-47+11*r5)]], [[(44-8*r5), (34-6*r5)], [(41-5*r5), (3-3*r5)]], [[(19-3*r5), (8+4*r5)], [(20-12*r5), (40-12*r5)]]]], [[[[(28-4*r5), (-198+22*r5)], [(254-30*r5), (-266-6*r5)]], [[(272+24*r5), (212+20*r5)], [(278+42*r5), (-6-18*r5)]], [[(122+14*r5), (104+48*r5)], [(40-56*r5), (200-16*r5)]]], [[[(35-11*r5), (-240+71*r5)], [(310-93*r5), (-280+57*r5)]], [[(265-39*r5), (205-29*r5)], [(250-21*r5), (15-18*r5)]], [[(115-14*r5), (55+27*r5)], [(110-70*r5), (235-65*r5)]]], [[[(26-6*r5), (-181+37*r5)], [(233-49*r5), (-227+19*r5)]], [[(224-4*r5), (174-2*r5)], [(221+11*r5), (3-15*r5)]], [[(99+1*r5), (68+32*r5)], [(60-52*r5), (180-32*r5)]]], [[[(22+2*r5), (-162-20*r5)], [(206+24*r5), (-254-60*r5)]], [[(278+78*r5), (218+62*r5)], [(302+96*r5), (-24-18*r5)]], [[(128+38*r5), (146+66*r5)], [(-20-44*r5), (170+26*r5)]]]], [[[[(-27+7*r5), (187-44*r5)], [(-241+58*r5), (229-28*r5)]], [[(-223+13*r5), (-173+9*r5)], [(-217-2*r5), (-6+15*r5)]], [[(-98+3*r5), (-61-29*r5)], [(-70+54*r5), (-185+39*r5)]]], [[[(-12-4*r5), (92+32*r5)], [(-116-40*r5), (164+64*r5)]], [[(-188-76*r5), (-148-60*r5)], [(-212-88*r5), (24+12*r5)]], [[(-88-36*r5), (-116-52*r5)], [(40+24*r5), (-100-36*r5)]]], [[[(-8-4*r5), (63+31*r5)], [(-79-39*r5), (121+57*r5)]], [[(-142-66*r5), (-112-52*r5)], [(-163-75*r5), (21+9*r5)]], [[(-67-31*r5), (-94-42*r5)], [(40+16*r5), (-70-34*r5)]]], [[[(-51+11*r5), (356-67*r5)], [(-458+89*r5), (452-29*r5)]], [[(-449-1*r5), (-349-3*r5)], [(-446-31*r5), (-3+30*r5)]], [[(-199-6*r5), (-143-67*r5)], [(-110+102*r5), (-355+57*r5)]]]]]
        >>> product.dims
        [2, 2, 3, 4, 3]

        :param other:
        :return:
        """
        # # try to convert possible scalar into tensor
        # if not isinstance(other,FTensor):
        #     other = FTensor([other])
        return FTensor(np.tensordot(self.components,other.components,axes=0))

    def contract(self,other,axes=[]):
        return FTensor(np.tensordot(self.components,other.components,axes=axes))

    def __neg__(self):
        return FTensor(-self.components)

    def __str__(self):
        return str(self.components.tolist())
    def __repr__(self):
        return str(self.components.tolist())

    def __sub__(self,other):
        return FTensor(self.components-other.components)

    def __add__(self,other):
        return FTensor(self.components+other.components)

    def __eq__(self,other):
        return np.array_equal(self.components,other.components)
    def __hash__(self):
        """
        flatten array to list and convert it to tuple to generate the default hash value of a tuple

        >>> m = np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
        >>> m.shape
        (3, 2, 2)
        >>> hash(tuple(m.flatten().tolist()))
        8321287486536459339

        :return:
        """
        return hash(tuple(self.components.flatten().tolist()))

class FVector(FTensor):
    def __init__(self, components:list):
        self.dim = len(components)
        super().__init__(components)

    @classmethod
    def parse(cls, string):
        """
        parse matrix from string
        >>> FVector.parse("[-2, 0, 0, 0]")
        [-2, 0, 0, 0]
        >>> FVector.parse("[1, (1/2-1/2*r5), 0, (1/2+1/2*r5)]")
        [1, (1/2-1/2*r5), 0, (1/2+1/2*r5)]
        >>> FVector.parse("[(1/2-1/2*r5), (1/2+1/2*r5), 0, 1]")
        [(1/2-1/2*r5), (1/2+1/2*r5), 0, 1]
        """
        # remove outer brackets
        string = string[1:-1]
        parts = string.split(", ")

        components = []
        for part in parts:
            components.append(QR5.parse(part))
        return cls(components)

    @classmethod
    def from_vector(cls,vector, *args):
        if isinstance(vector,Vector):
            vector=vector[0:3]

        components = []
        for v in vector:
            if isinstance(v,int):
                components.append(QR5.from_integers(v,1,0,1))
            elif isinstance(v,Fraction):
                components.append(QR5.from_integers(v.numerator,v.denominator,0,1))
            else:
                components.append(v)
        return FVector(components)

    def dot(self,other)->QR5:
        """
        >>> a = FVector([QR5.from_integers(1,1,0,1), QR5.from_integers(1,2,2,1)])
        >>> a.dot(a)
        (85/4+2*r5)


        :param other:
        :return:
        """
        return np.tensordot(self.components,other.components,axes=1).tolist()

    def cross(self,other: 'FVector' = None):
        """
        Calculates the cross product of the current vector with another vector.

        The cross product is only defined for 3D vectors. If the current vector or the
        other vector is not in 3D, an exception will be raised.

        Parameters:
            other (FVector, optional): The other vector to calculate the cross product
                with. Must also be a 3D vector.

        Returns:
            FVector: A new vector resulting from the cross product of the current
            vector and the other vector.

        Raises:
            Exception: If the cross product is attempted on vectors not in 3D.

        >>> one = QR5.from_integers(1,1,0,1)
        >>> zero = QR5.from_integers(0,1,0,1)
        >>> x = FVector([one,zero,zero])
        >>> y = FVector([zero,one,zero])
        >>> z = FVector([zero,zero,one])
        >>> x.cross(y)
        [0, 0, 1]
        >>> y.cross(z)
        [1, 0, 0]
        >>> z.cross(x)
        [0, 1, 0]

        """
        if self.dim==3:
            return FVector([self.components[1]*other.components[2]-self.components[2]*other.components[1],
                            self.components[2]*other.components[0]-self.components[0]*other.components[2],
                            self.components[0]*other.components[1]-self.components[1]*other.components[0]])
        else:
            raise Exception("cross product only defined for 3D vectors")

    def norm(self):
        return self.dot(self)

    def __sub__(self,other):
        return FVector(self.components-other.components)

    def __add__(self,other):
        return FVector(self.components+other.components)

    def real(self):
        return Vector([self.components[i].real() for i in range(self.dim)])

class FMatrix(FTensor):
    def __init__(self, components:list):
        super().__init__(components)

    @classmethod
    def parse(cls,string):
        """
        parse matrix from string
        >>> FMatrix.parse("[[1/2, 1/2, 1/2, 1/2], [1/2, 0, (-1/4-1/4*r5), (-1/4+1/4*r5)], [-1/2, (1/4+1/4*r5), (1/4-1/4*r5), 0], [-1/2, (1/4-1/4*r5), 0, (1/4+1/4*r5)]]")
        [[1/2, 1/2, 1/2, 1/2], [1/2, 0, (-1/4-1/4*r5), (-1/4+1/4*r5)], [-1/2, (1/4+1/4*r5), (1/4-1/4*r5), 0], [-1/2, (1/4-1/4*r5), 0, (1/4+1/4*r5)]]
        >>> FMatrix.parse("[[-1/2, (-1/4+1/4*r5), 0, (-1/4-1/4*r5)], [1/2, -1/2, -1/2, -1/2], [1/2, (1/4+1/4*r5), (1/4-1/4*r5), 0], [-1/2, 0, (-1/4-1/4*r5), (-1/4+1/4*r5)]]")
        [[-1/2, (-1/4+1/4*r5), 0, (-1/4-1/4*r5)], [1/2, -1/2, -1/2, -1/2], [1/2, (1/4+1/4*r5), (1/4-1/4*r5), 0], [-1/2, 0, (-1/4-1/4*r5), (-1/4+1/4*r5)]]
        >>> FMatrix.parse("[[(-1/4+1/4*r5), 1/2, (1/4+1/4*r5), 0], [0, 0, 0, -1], [-1/2, (1/4+1/4*r5), (1/4-1/4*r5), 0], [(1/4+1/4*r5), (-1/4+1/4*r5), -1/2, 0]]")
        [[(-1/4+1/4*r5), 1/2, (1/4+1/4*r5), 0], [0, 0, 0, -1], [-1/2, (1/4+1/4*r5), (1/4-1/4*r5), 0], [(1/4+1/4*r5), (-1/4+1/4*r5), -1/2, 0]]

        """
        # remove outer brackets
        string = string[1:-1]
        rows = string.split("], [")
        # remove last brackets
        rows[0]=rows[0][1:]
        rows[-1]=rows[-1][:-1]

        components = []
        for row in rows:
            comp_row = []
            parts = row.split(", ")
            for part in parts:
                comp_row.append(QR5.parse(part))
            components.append(comp_row)
        return cls(components)


    def __mul__(self,other):
        """
        >>> I = FMatrix([[0,1],[1,0]])
        >>> I*I
        [[1, 0], [0, 1]]

        :param other:
        :return:
        """
        return FMatrix(np.tensordot(self.components,other.components,axes=[[1],[0]]))

    def __matmul__(self,other:FVector):
        return FVector(np.tensordot(self.components,other.components,axes=[[1],[0]]))

class EpsilonTensor(FTensor):
    def __init__(self,rank):
        n = rank**rank
        comps = []
        for i in range(n):
            comps.append(QR5.from_integers(0,1,0,1))
        comps = np.array(comps)
        comps.shape = (rank,)*rank
        permutations =  list(itertools.permutations(range(rank)))
        for permutation in permutations:
            p = Permutation(permutation)
            comps[permutation] = QR5.from_integers(p.signature(),1,0,1)

        super().__init__(comps.tolist())
