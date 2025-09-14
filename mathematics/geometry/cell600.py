# the 600 cell is generated from two quaternions
# omega = (-1/2+1/2i+1/2j+1/2k)
# q = (0+1/2i+1/4*(1+r5)j+1/4*(-1+r5)k
import itertools
import os
from fractions import Fraction

import numpy as np
from interface.ibpy import Vector, Matrix
from sympy.combinatorics import Permutation

from mathematics.zeros import chop
from utils.constants import DATA_DIR


# for the generation of the finite group, it is best to work with
# exact values for r5 = 5**0.5
# therefore, the field extension QR5 is implemented

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
        return hash((self.x,self.y))

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

class FVector(FTensor):
    def __init__(self, components:list):
        self.dim = len(components)
        super().__init__(components)

    def dot(self,other)->QR5:
        """
        >>> a = FVector([QR5.from_integers(1,1,0,1), QR5.from_integers(1,2,2,1)])
        >>> a.dot(a)
        (85/4+2*r5)


        :param other:
        :return:
        """
        return np.tensordot(self.components,other.components,axes=1).tolist()

    def norm(self):
        return self.dot(self)

    def __sub__(self,other):
        return FVector(self.components-other.components)

    def __add__(self,other):
        return FVector(self.components+other.components)

    def real(self):
        return Vector([self.components[i].real() for i in range(self.dim)])

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

# with the normal vector, we can project and rotate the vertices into our
# three-dimensional world

def generate_group():
    # generate 120 elements of the 600 cell
    omega=gen_a = FQuaternion.from_vector(QR5.from_integers(-1,2,0,1),
                                          QR5.from_integers(1,2,0,1),
                                          QR5.from_integers(1,2,0,1),
                                          QR5.from_integers(1,2,0,1))
    q=gen_b = FQuaternion.from_vector(QR5.from_integers(0,1,0,1),
                                      QR5.from_integers(1,2,0,1),
                                      QR5.from_integers(1,4,1,4),
                                      QR5.from_integers(-1,4,1,4))
    # check group constraints
    one = QR5.from_integers(1, 1, 0, 1)
    zero = QR5.from_integers(0, 1, 0, 1)
    q_one = FQuaternion.from_vector(one, zero, zero, zero)
    assert(q*q*q*q==q_one)
    assert(omega*omega*omega==q_one)
    assert(q*omega*q*omega*q*omega*q*omega*q*omega==q_one)

    elements = {gen_a, gen_b}
    new_elements = {gen_a, gen_b}
    generators = {gen_a, gen_b}

    while len(new_elements)>0:
        next_elements = set()
        for e in new_elements:
            for g in generators:
                element = e*g
                if element not in elements:
                    next_elements.add(element)
                    elements.add(element)
        new_elements = next_elements

    assert(len(elements)==120)
    return elements

def get_4D_vectors():
    elements = generate_group()
    vectors = []
    for element in elements:
        vectors.append(element.to_vector())
    return vectors

def detect_edges(elements):
    edges = []
    min_dist = np.inf
    min = Fraction(0,1)
    for i in range(len(elements)):
        for j in range(i+1,len(elements)):
            norm = (elements[i]-elements[j]).norm()
            dist = norm.real()
            if dist<min_dist:
                min_dist = dist
                min = norm

    print("minimal edge length: ",min)

    for i in range(len(elements)):
        for j in range(i+1,len(elements)):
            dist = (elements[i]-elements[j]).norm().real()
            if dist==min_dist:
                edges.append((i,j))

    return edges

def detect_cells(faces,edges):
    # find to faces with a common edge
    pairs = []
    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            common = set(faces[i]).intersection(set(faces[j]))
            if len(common)==1:
                pairs.append((i,j))

    print(len(pairs),"pairs")

    # merge to pairs of faces into a cell
    cells = set()
    for i in range(len(pairs)):
        for j in range(i+1,len(pairs)):
            pair_faces = set(pairs[i]+pairs[j])
            if len(pair_faces)==4:
                cell_edges = set()
                for face_index in pair_faces:
                     cell_edges= cell_edges.union(set([i for i in faces[face_index]]))

                if len(cell_edges)==6:
                    cells.add(tuple(sorted(pair_faces)))

    print(len(cells),"cells")
    return list(cells)

def detect_faces(edges):
    faces = []
    for i in range(len(edges)):
        for j in range(i+1,len(edges)):
            for k in range(j+1,len(edges)):
                edge1 = set(edges[i])
                edge2 = set(edges[j])
                edge3 = set(edges[k])
                all = edge1.union(edge2,edge3)
                if len(all)==3:
                    faces.append([i,j,k])
    return faces

def save(data,filename):
    with open(filename,"w") as f:
        for d in data:
            f.write(str(d)+"\n")

def read(filename):
    with open(filename,"r") as f:
        data = []
        for line in f:
            data.append(eval(line))
    return data

def compute_equation(cell,faces,edges,vectors):
    """
    The normal vector is computed as the dual of the tri-vector that is spanned by the four vertices of the cell.
    Then the hyper-plane is given by the equation n.x=n.x1

    :param cell:
    :param faces:
    :param edges:
    :param vectors:
    :return:
    """
    # grap the four vertices of the cell
    cell_faces = [faces[i] for i in cell]
    cell_edges = [edges[j] for c in cell_faces for j in c]
    cell_vertex_indices = set([i for e in cell_edges for i in e])
    cell_vertices = set([tuple(vectors[j]) for e in cell_edges for j in e])

    fvectors = []
    for vertex in cell_vertices:
        fvectors.append(FVector(vertex))

    fbasis =[]
    for i in range(1,len(fvectors)):
        fbasis.append(fvectors[i]-fvectors[0])

    # create tensor product from the basis
    tensor = FTensor(fbasis[0].components)
    tensor *=FTensor(fbasis[1].components)
    tensor *=FTensor(fbasis[2].components)

    epsilon = EpsilonTensor(4)

    n = epsilon.contract(tensor,axes=[[1,2,3],[0,1,2]])

    # consistency check
    # print("Consistency check for normal vector:")
    # print(cell_vertex_indices)
    # print((n.contract(fbasis[0],axes=[[0],[0]])).components)
    # print((n.contract(fbasis[1],axes=[[0],[0]])).components)
    # print((n.contract(fbasis[2],axes=[[0],[0]])).components)

    # make sure that the normal vector is pointing outwards
    for i in range(len(vectors)):
        if i not in cell_vertex_indices:
            point = FVector(vectors[i])
            if n.contract(point,axes=[[0],[0]]).components.tolist().real() < 0:
                n =-n
            break

    # we cannot normalize the normal vector within the field, we store its length
    # go to float coordinates

    return [n.components.tolist(),n.contract(fvectors[0],axes=[[0],[0]]).components.tolist()]

def get_rotation_matrix(normal):
    """
    WARNING: llm-generated!!!!
    this function computes the rotation matrix that rotates a generic four-dimensional normal vector to the form (0,0,0,1)

    Note:
    - The matrix is an orthonormal rotation (determinant +1).
    - Rotations preserve length. Therefore, this maps the direction of (a,b,c,d) to the +w-axis.
      The image of (a,b,c,d) will be (0,0,0,||(a,b,c,d)||). If you need exactly (0,0,0,1),
      pass a unit-length vector.

    :param normal: iterable of 4 scalars; each item may be a number or provide .real()
    :return: 4x4 numpy array representing the rotation matrix
    """

    # Convert to floats, supporting objects that implement .real()
    def to_float(x):
        # Objects in this project often provide a .real() method returning a float
        if hasattr(x, "real") and not isinstance(x, (int, float)):
            return float(x.real())
        return float(x)

    v = np.array([to_float(x) for x in normal], dtype=float).reshape(4)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Cannot build a rotation for the zero vector.")

    # Build rotation as a product of Givens rotations that zero a, b, c in order,
    # leaving all steps as proper rotations (det = +1).
    R = np.eye(4, dtype=float)

    def givens(i, j, x, y):
        """
        Return a 4x4 Givens rotation acting on coordinates (i, j) that maps [y, x] -> [0, hypot(x,y)].
        Specifically, it zeroes the 'y' component while preserving norm.
        """
        r = (x * x + y * y) ** 0.5
        if r == 0.0:
            return np.eye(4, dtype=float)
        c = x / r
        s = -y / r
        G = np.eye(4, dtype=float)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = s
        G[j, i] = -s
        return G

    # Step 1: zero c using plane (2,3) acting on (c,d)
    G1 = givens(2, 3, v[3], v[2])
    v = G1 @ v
    R = G1 @ R

    # Step 2: zero b using plane (1,3) acting on (b,d')
    G2 = givens(1, 3, v[3], v[1])
    v = G2 @ v
    R = G2 @ R

    # Step 3: zero a using plane (0,3) acting on (a,d'')
    G3 = givens(0, 3, v[3], v[0])
    v = G3 @ v
    R = G3 @ R

    # Ensure the final d component is positive (rotate by pi in (0,3) plane if needed).
    if v[3] < 0.0:
        G4 = np.eye(4, dtype=float)
        G4[0, 0] = -1.0
        G4[3, 3] = -1.0
        v = G4 @ v
        R = G4 @ R

    # R @ original_vector == [0,0,0, ||original_vector||]
    test = R@normal
    test = [round(comp,6) for comp in test]
    if test!= [0,0,0,1]:
        print(test)
        raise ValueError("Rotation matrix does not map normal vector to (0,0,0,1). "
                         "Maybe there is an error in the llm-generated code of this function")

    return R

def compute_projection(vectors,equation,cell, faces,edges,offset):
    """
    here the projection onto the cell is performed.
    This repeats computations that I did for the video of the 600 cell.



    :param vectors:
    :param equation:
    :param offset:
    :return:
    """

    # turn into floats and normalize
    # warning the built-in function doesn't normalize the fourth component
    normal = Vector([c.real() for c in equation[0]])
    n = normal.dot(normal)**0.5
    normal = normal/n

    # grap the four vertices of the cell
    cell_faces = [faces[i] for i in cell]
    cell_edges = [edges[j] for c in cell_faces for j in c]
    cell_vertex_indices = set([j for e in cell_edges for j in e])
    cell_vertices = set([tuple(vectors[j]) for e in cell_edges for j in e])
    cell_vertices = [Vector([float(c.real())  for c in vector]) for vector in cell_vertices]

    cell_center = sum(cell_vertices,Vector([0,0,0,0]))/len(cell_vertices)
    # construct focal point
    focus = cell_center+normal*(offset*(cell_center-cell_vertices[0]).length)

    transformed_vectors = []
    # iterate over all points to transform
    for i in range(len(vectors)):
        if i not in cell_vertex_indices:  # skip vectors of projection cell
            v = Vector([v.real() for v in vectors[i]])
            alpha = (cell_center-v).dot(normal)/(focus-v).dot(normal)
            transformed_vector = (focus*alpha)+(v*(1-alpha))
            transformed_vectors.append(transformed_vector)
        else:
            v = Vector([v.real() for v in vectors[i]])
            transformed_vectors.append(v)

    print("consistency check, all the following values should be of equal size")
    print(all([float(chop((t-cell_center).dot(normal),1e-6))==0 for i,t in enumerate(transformed_vectors)]))

    # shift center of the cell to the origin
    shifted_vertices = [t-cell_center for t in transformed_vectors]

    # rotate all vectors into a 3D coordinate sub space
    matrix = get_rotation_matrix(normal)
    print(matrix@normal)

    rotated_vertices = [matrix@v for v in shifted_vertices]
    return rotated_vertices

def get_face_indices(face,edges,vertices):
    """ express the face in terms of vertex indices properly oriented"""
    face_edge_indices = [edges[idx] for idx in face]
    face_vertex_indices = list(set([e[0] for e in face_edge_indices]+[e[1] for e in face_edge_indices]))
    vectors = [vertices[i] for i in face_vertex_indices]
    # the faces are triangles, we compute the normal and if the center of the triangle dotted with the normal is negative, we switch two vertices
    # to change the orientation

    base1 = vectors[1]-vectors[0]
    base2 = vectors[2]-vectors[0]
    center = sum(vectors,Vector([0,0,0]))/len(vectors)
    normal = base1.cross(base2)
    if (center.dot(normal)<0):
        face_vertex_indices[1],face_vertex_indices[2]=face_vertex_indices[2],face_vertex_indices[1]
    return face_vertex_indices

def get_4D_geometry():
    vectors = get_4D_vectors()

    # edges in terms of vertices

    # edges = detect_edges(elements)
    # save(edges,"edges.data")
    edges = read("edges.data")
    print("number of edges ", len(edges), edges)

    # faces in terms of edges

    # faces = detect_faces(edges)
    # save(faces,"faces.data")
    faces = read("faces.data")
    print("number of faces ", len(faces), faces)

    # cells in terms of faces
    # cells = detect_cells(faces,edges)
    # save(cells,"cells.data")
    cells = read("cells.data")

    return [vectors,edges,faces]

def get_3D_geometry(offset=0.5):
    elements = list(generate_group())
    vectors = get_4D_vectors()

    # edges in terms of vertices

    # edges = detect_edges(elements)
    # save(edges,"edges.data")
    edges = read("edges.data")
    print("number of edges ",len(edges),edges)

    # faces in terms of edges

    # faces = detect_faces(edges)
    # save(faces,"faces.data")
    faces = read("faces.data")
    print("number of faces ",len(faces),faces)

    # cells in terms of faces
    # cells = detect_cells(faces,edges)
    # save(cells,"cells.data")
    cells = read("cells.data")

    # compute the equation for the cell that the polytope is projected onto
    equation = compute_equation(cells[0],faces,edges,vectors)

    projected_vectors = compute_projection(vectors,equation,cells[0],faces,edges,offset)
    vertices = [Vector(v[0:3]) for v in projected_vectors]

    # blender wants to have the faces in terms of the vertices and not in terms of the edges
    # the faces have to be re-indexed and the order of the vertices is important to create a loop that points away from the origin

    faces = [get_face_indices(face,edges,vertices) for face in faces]


    return [vertices,edges,faces]

def export_csv_data(dir=DATA_DIR,filename=""):
    """
        >>> export_csv_data(filename="poly600cell.csv")
        True

    """

    if filename!="":
        dir = os.path.join(dir,filename)
        vectors = get_4D_vectors()
        edges = read("edges.data")
        faces = read("faces.data")
        cells = read("cells.data")

        # find smallest z-value of a cell-center for each vertex
        w_cell_centers = []
        for i in range(120):
            w_cell_centers.append(np.inf)
        for cell in cells:
            cell_faces = [faces[i] for i in cell]
            cell_edges = [edges[j] for c in cell_faces for j in c]
            cell_vertex_indices = set([i for e in cell_edges for i in e])
            cell_vertices = set([tuple(vectors[j]) for e in cell_edges for j in e])
            cell_vertices = [Vector([float(comp.real()) for comp in v]) for v in cell_vertices]
            center = sum(cell_vertices,Vector([0,0,0,0]))/len(cell_vertices)

            for i in cell_vertex_indices:
                if center[3]<w_cell_centers[i]:
                    w_cell_centers[i]=center[3]

        print(set(w_cell_centers))

        with open(dir,"w") as f:
            f.write("x,y,z,w,c\n")
            for v,w_center in zip(vectors,w_cell_centers):
                f.write(f"{v[0].real()},{v[1].real()},{v[2].real()},{v[3].real()},{w_center}\n")
        return True
    return False

def get_base_cell():
    elements = list(generate_group())
    vectors = get_4D_vectors()

    edges = read("edges.data")
    faces = read("faces.data")
    cells = read("cells.data")

    # grap the four vertices of one cell
    cell_faces = [faces[i] for i in cells[0]]
    cell_edges = [edges[j] for c in cell_faces for j in c]
    cell_vertex_indices = set([j for e in cell_edges for j in e])
    cell_vertices = set([tuple(vectors[j]) for e in cell_edges for j in e])
    cell_vertices = [Vector([float(c.real()) for c in vector]) for vector in cell_vertices]
    cell_center = sum(cell_vertices, Vector([0, 0, 0, 0])) / len(cell_vertices)

    equation = compute_equation(cells[0],faces, edges, vectors)

    normal = Vector([comp.real() for comp in  equation[0]])
    n=normal.dot(normal)**0.5
    normal = normal/n

    matrix = get_rotation_matrix(normal)
    print(matrix @ normal)

    vertices = [v-cell_center for v in cell_vertices]
    vertices = [matrix@v for v in vertices]
    print(vertices)
    vertices = [v[0:3] for v in vertices]
    print(vertices)
    return [vertices,cell_edges,cell_faces]

if __name__ == '__main__':
    export_csv_data()