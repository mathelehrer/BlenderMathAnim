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

PATH = "mathematics/geometry/data/"

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

    # print("minimal edge length: ",min)

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

    # print(len(pairs),"pairs")

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

    # print(len(cells),"cells")
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
    with open(os.path.join(PATH,filename),"w") as f:
        for d in data:
            f.write(str(d)+"\n")

def read(filename,pre_path=""):
    path = os.path.join(pre_path,PATH)
    with open(os.path.join(path,filename),"r") as f:
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
    """
    >>> get_4D_geometry()
    120  vertices
    720  edges
    1200  faces
    600  cells
    [[[0, (1/4+1/4*r5), (1/4-1/4*r5), 1/2], [(1/4+1/4*r5), (-1/4+1/4*r5), 0, -1/2], [1/2, (-1/4-1/4*r5), 0, (1/4-1/4*r5)], [(1/4-1/4*r5), (-1/4-1/4*r5), 1/2, 0], [1/2, (1/4+1/4*r5), 0, (1/4-1/4*r5)], [0, (-1/4-1/4*r5), (1/4-1/4*r5), -1/2], [(-1/4+1/4*r5), (-1/4-1/4*r5), -1/2, 0], [(1/4-1/4*r5), (1/4+1/4*r5), 1/2, 0], [(1/4+1/4*r5), -1/2, (1/4-1/4*r5), 0], [-1/2, -1/2, 1/2, 1/2], [(1/4-1/4*r5), 0, (-1/4-1/4*r5), 1/2], [0, 0, 1, 0], [0, 1/2, (1/4+1/4*r5), (1/4-1/4*r5)], [(1/4+1/4*r5), -1/2, (-1/4+1/4*r5), 0], [-1/2, 1/2, 1/2, -1/2], [1/2, -1/2, -1/2, 1/2], [(-1/4-1/4*r5), 1/2, (1/4-1/4*r5), 0], [0, (1/4-1/4*r5), -1/2, (1/4+1/4*r5)], [-1/2, -1/2, -1/2, -1/2], [-1/2, 1/2, -1/2, -1/2], [0, 0, -1, 0], [(-1/4-1/4*r5), 1/2, (-1/4+1/4*r5), 0], [-1/2, (1/4-1/4*r5), (1/4+1/4*r5), 0], [1/2, 1/2, -1/2, -1/2], [0, -1, 0, 0], [1/2, 0, (-1/4+1/4*r5), (1/4+1/4*r5)], [1/2, (-1/4-1/4*r5), 0, (-1/4+1/4*r5)], [(1/4-1/4*r5), (-1/4-1/4*r5), -1/2, 0], [1/2, (1/4+1/4*r5), 0, (-1/4+1/4*r5)], [-1/2, (-1/4+1/4*r5), (-1/4-1/4*r5), 0], [(1/4-1/4*r5), (1/4+1/4*r5), -1/2, 0], [-1/2, 0, (-1/4+1/4*r5), (-1/4-1/4*r5)], [0, -1/2, (-1/4-1/4*r5), (-1/4+1/4*r5)], [(1/4+1/4*r5), 0, 1/2, (1/4-1/4*r5)], [(1/4-1/4*r5), 0, (1/4+1/4*r5), -1/2], [1/2, 0, (1/4-1/4*r5), (-1/4-1/4*r5)], [0, 1/2, (1/4+1/4*r5), (-1/4+1/4*r5)], [(-1/4+1/4*r5), 0, (-1/4-1/4*r5), -1/2], [(1/4+1/4*r5), 0, -1/2, (1/4-1/4*r5)], [1/2, (-1/4+1/4*r5), (1/4+1/4*r5), 0], [(-1/4-1/4*r5), 0, -1/2, (1/4-1/4*r5)], [1/2, -1/2, 1/2, -1/2], [(1/4+1/4*r5), 1/2, (1/4-1/4*r5), 0], [-1/2, (1/4+1/4*r5), 0, (1/4-1/4*r5)], [0, 0, 0, 1], [0, (-1/4+1/4*r5), 1/2, (1/4+1/4*r5)], [-1/2, 1/2, 1/2, 1/2], [(1/4+1/4*r5), 1/2, (-1/4+1/4*r5), 0], [-1/2, -1/2, -1/2, 1/2], [0, (-1/4+1/4*r5), -1/2, (1/4+1/4*r5)], [1/2, 1/2, 1/2, 1/2], [1/2, (1/4-1/4*r5), (-1/4-1/4*r5), 0], [(-1/4-1/4*r5), (1/4-1/4*r5), 0, 1/2], [0, (1/4-1/4*r5), 1/2, (-1/4-1/4*r5)], [(-1/4-1/4*r5), (1/4-1/4*r5), 0, -1/2], [1/2, 1/2, -1/2, 1/2], [(1/4-1/4*r5), 1/2, 0, (-1/4-1/4*r5)], [-1, 0, 0, 0], [(-1/4+1/4*r5), -1/2, 0, (-1/4-1/4*r5)], [0, (1/4+1/4*r5), (-1/4+1/4*r5), 1/2], [-1/2, 0, (1/4-1/4*r5), (1/4+1/4*r5)], [0, -1/2, (-1/4-1/4*r5), (1/4-1/4*r5)], [1/2, 0, (-1/4+1/4*r5), (-1/4-1/4*r5)], [0, (-1/4-1/4*r5), (-1/4+1/4*r5), -1/2], [(1/4+1/4*r5), 0, -1/2, (-1/4+1/4*r5)], [-1/2, (1/4+1/4*r5), 0, (-1/4+1/4*r5)], [-1/2, (-1/4-1/4*r5), 0, (1/4-1/4*r5)], [0, (-1/4-1/4*r5), (1/4-1/4*r5), 1/2], [(-1/4-1/4*r5), 0, 1/2, (1/4-1/4*r5)], [(-1/4+1/4*r5), 0, (-1/4-1/4*r5), 1/2], [(-1/4+1/4*r5), 0, (1/4+1/4*r5), 1/2], [(-1/4-1/4*r5), (-1/4+1/4*r5), 0, 1/2], [(-1/4-1/4*r5), (-1/4+1/4*r5), 0, -1/2], [(1/4-1/4*r5), -1/2, 0, (1/4+1/4*r5)], [(1/4-1/4*r5), 0, (-1/4-1/4*r5), -1/2], [1/2, -1/2, -1/2, -1/2], [(-1/4-1/4*r5), -1/2, (1/4-1/4*r5), 0], [(-1/4+1/4*r5), (1/4+1/4*r5), 1/2, 0], [1/2, (-1/4+1/4*r5), (-1/4-1/4*r5), 0], [-1/2, (-1/4+1/4*r5), (1/4+1/4*r5), 0], [0, (-1/4+1/4*r5), 1/2, (-1/4-1/4*r5)], [-1/2, 1/2, -1/2, 1/2], [(-1/4-1/4*r5), -1/2, (-1/4+1/4*r5), 0], [0, 1, 0, 0], [(-1/4+1/4*r5), 1/2, 0, (1/4+1/4*r5)], [0, (1/4-1/4*r5), -1/2, (-1/4-1/4*r5)], [1, 0, 0, 0], [(1/4+1/4*r5), 0, 1/2, (-1/4+1/4*r5)], [-1/2, (-1/4-1/4*r5), 0, (-1/4+1/4*r5)], [1/2, 0, (1/4-1/4*r5), (1/4+1/4*r5)], [0, (-1/4-1/4*r5), (-1/4+1/4*r5), 1/2], [(-1/4-1/4*r5), 0, 1/2, (-1/4+1/4*r5)], [0, -1/2, (1/4+1/4*r5), (1/4-1/4*r5)], [(1/4+1/4*r5), (1/4-1/4*r5), 0, 1/2], [0, 0, 0, -1], [(1/4+1/4*r5), (1/4-1/4*r5), 0, -1/2], [-1/2, (1/4-1/4*r5), (-1/4-1/4*r5), 0], [0, (1/4+1/4*r5), (1/4-1/4*r5), -1/2], [-1/2, 0, (1/4-1/4*r5), (-1/4-1/4*r5)], [(-1/4-1/4*r5), 0, -1/2, (-1/4+1/4*r5)], [1/2, (1/4-1/4*r5), (1/4+1/4*r5), 0], [(-1/4+1/4*r5), (-1/4-1/4*r5), 1/2, 0], [1/2, -1/2, 1/2, 1/2], [(-1/4+1/4*r5), (1/4+1/4*r5), -1/2, 0], [(1/4-1/4*r5), 0, (1/4+1/4*r5), 1/2], [0, 1/2, (-1/4-1/4*r5), (1/4-1/4*r5)], [0, (1/4-1/4*r5), 1/2, (1/4+1/4*r5)], [-1/2, -1/2, 1/2, -1/2], [(1/4-1/4*r5), 1/2, 0, (1/4+1/4*r5)], [(1/4-1/4*r5), -1/2, 0, (-1/4-1/4*r5)], [0, 1/2, (-1/4-1/4*r5), (-1/4+1/4*r5)], [1/2, 1/2, 1/2, -1/2], [(-1/4+1/4*r5), -1/2, 0, (1/4+1/4*r5)], [0, -1/2, (1/4+1/4*r5), (-1/4+1/4*r5)], [-1/2, 0, (-1/4+1/4*r5), (1/4+1/4*r5)], [0, (-1/4+1/4*r5), -1/2, (-1/4-1/4*r5)], [0, (1/4+1/4*r5), (-1/4+1/4*r5), -1/2], [(-1/4+1/4*r5), 1/2, 0, (-1/4-1/4*r5)], [(-1/4+1/4*r5), 0, (1/4+1/4*r5), -1/2], [(1/4+1/4*r5), (-1/4+1/4*r5), 0, 1/2]], [(0, 28), (0, 30), (0, 49), (0, 55), (0, 59), (0, 65), (0, 81), (0, 83), (0, 84), (0, 103), (0, 108), (0, 110), (1, 4), (1, 23), (1, 33), (1, 35), (1, 38), (1, 42), (1, 47), (1, 62), (1, 86), (1, 95), (1, 111), (1, 117), (2, 5), (2, 6), (2, 8), (2, 13), (2, 24), (2, 26), (2, 41), (2, 58), (2, 63), (2, 75), (2, 95), (2, 101), (3, 9), (3, 22), (3, 24), (3, 63), (3, 66), (3, 82), (3, 88), (3, 90), (3, 92), (3, 101), (3, 107), (3, 113), (4, 23), (4, 28), (4, 42), (4, 47), (4, 77), (4, 83), (4, 97), (4, 103), (4, 111), (4, 116), (4, 117), (5, 6), (5, 18), (5, 24), (5, 27), (5, 58), (5, 61), (5, 63), (5, 66), (5, 75), (5, 85), (5, 109), (6, 8), (6, 15), (6, 24), (6, 26), (6, 27), (6, 32), (6, 51), (6, 61), (6, 67), (6, 75), (7, 12), (7, 14), (7, 21), (7, 36), (7, 43), (7, 46), (7, 59), (7, 65), (7, 77), (7, 79), (7, 83), (7, 116), (8, 13), (8, 15), (8, 26), (8, 38), (8, 51), (8, 64), (8, 75), (8, 86), (8, 93), (8, 95), (9, 22), (9, 52), (9, 73), (9, 82), (9, 88), (9, 90), (9, 91), (9, 104), (9, 106), (9, 113), (9, 114), (10, 17), (10, 20), (10, 29), (10, 32), (10, 48), (10, 49), (10, 60), (10, 69), (10, 81), (10, 96), (10, 99), (10, 110), (11, 12), (11, 22), (11, 34), (11, 36), (11, 39), (11, 70), (11, 79), (11, 92), (11, 100), (11, 104), (11, 113), (11, 118), (12, 14), (12, 34), (12, 36), (12, 39), (12, 77), (12, 79), (12, 80), (12, 111), (12, 116), (12, 118), (13, 26), (13, 33), (13, 41), (13, 86), (13, 87), (13, 93), (13, 95), (13, 100), (13, 101), (13, 102), (14, 21), (14, 31), (14, 34), (14, 43), (14, 56), (14, 68), (14, 72), (14, 79), (14, 80), (14, 116), (15, 17), (15, 26), (15, 32), (15, 51), (15, 64), (15, 67), (15, 69), (15, 89), (15, 93), (15, 112), (16, 19), (16, 21), (16, 29), (16, 30), (16, 40), (16, 43), (16, 57), (16, 65), (16, 71), (16, 72), (16, 81), (16, 99), (17, 32), (17, 44), (17, 48), (17, 49), (17, 60), (17, 67), (17, 69), (17, 73), (17, 89), (17, 112), (18, 27), (18, 40), (18, 54), (18, 61), (18, 66), (18, 74), (18, 76), (18, 85), (18, 96), (18, 98), (18, 109), (19, 29), (19, 30), (19, 40), (19, 43), (19, 56), (19, 72), (19, 74), (19, 97), (19, 98), (19, 105), (19, 115), (20, 29), (20, 32), (20, 37), (20, 51), (20, 61), (20, 69), (20, 74), (20, 78), (20, 96), (20, 105), (20, 110), (21, 43), (21, 46), (21, 57), (21, 65), (21, 68), (21, 71), (21, 72), (21, 79), (21, 91), (22, 34), (22, 68), (22, 79), (22, 82), (22, 91), (22, 92), (22, 104), (22, 107), (22, 113), (23, 35), (23, 37), (23, 38), (23, 42), (23, 78), (23, 97), (23, 103), (23, 105), (23, 115), (23, 117), (24, 26), (24, 27), (24, 63), (24, 66), (24, 67), (24, 88), (24, 90), (24, 101), (25, 44), (25, 45), (25, 50), (25, 70), (25, 84), (25, 87), (25, 89), (25, 93), (25, 102), (25, 106), (25, 112), (25, 119), (26, 67), (26, 90), (26, 93), (26, 101), (26, 102), (26, 112), (27, 32), (27, 48), (27, 61), (27, 66), (27, 67), (27, 76), (27, 88), (27, 96), (28, 42), (28, 47), (28, 50), (28, 55), (28, 59), (28, 77), (28, 83), (28, 84), (28, 103), (28, 119), (29, 30), (29, 40), (29, 74), (29, 81), (29, 96), (29, 99), (29, 105), (29, 110), (30, 43), (30, 65), (30, 81), (30, 83), (30, 97), (30, 103), (30, 105), (30, 110), (31, 34), (31, 53), (31, 54), (31, 56), (31, 68), (31, 72), (31, 80), (31, 94), (31, 98), (31, 107), (31, 109), (32, 48), (32, 51), (32, 61), (32, 67), (32, 69), (32, 96), (33, 39), (33, 41), (33, 47), (33, 62), (33, 86), (33, 87), (33, 95), (33, 100), (33, 111), (33, 118), (34, 53), (34, 68), (34, 79), (34, 80), (34, 92), (34, 107), (34, 118), (35, 37), (35, 38), (35, 58), (35, 62), (35, 75), (35, 85), (35, 94), (35, 95), (35, 115), (35, 117), (36, 39), (36, 45), (36, 46), (36, 50), (36, 59), (36, 70), (36, 77), (36, 79), (36, 104), (37, 38), (37, 51), (37, 61), (37, 74), (37, 75), (37, 78), (37, 85), (37, 105), (37, 115), (38, 42), (38, 51), (38, 64), (38, 75), (38, 78), (38, 86), (38, 95), (39, 47), (39, 50), (39, 70), (39, 77), (39, 87), (39, 100), (39, 111), (39, 118), (40, 54), (40, 57), (40, 72), (40, 74), (40, 76), (40, 96), (40, 98), (40, 99), (41, 53), (41, 58), (41, 62), (41, 63), (41, 92), (41, 95), (41, 100), (41, 101), (41, 118), (42, 47), (42, 55), (42, 64), (42, 78), (42, 86), (42, 103), (42, 119), (43, 56), (43, 65), (43, 72), (43, 83), (43, 97), (43, 116), (44, 45), (44, 49), (44, 60), (44, 73), (44, 84), (44, 89), (44, 106), (44, 108), (44, 112), (44, 114), (45, 46), (45, 50), (45, 59), (45, 70), (45, 84), (45, 104), (45, 106), (45, 108), (45, 114), (46, 59), (46, 65), (46, 71), (46, 79), (46, 91), (46, 104), (46, 108), (46, 114), (47, 50), (47, 77), (47, 86), (47, 87), (47, 111), (47, 119), (48, 52), (48, 60), (48, 67), (48, 73), (48, 76), (48, 88), (48, 96), (48, 99), (49, 55), (49, 60), (49, 69), (49, 81), (49, 84), (49, 89), (49, 108), (49, 110), (50, 59), (50, 70), (50, 77), (50, 84), (50, 87), (50, 119), (51, 61), (51, 64), (51, 69), (51, 75), (51, 78), (52, 57), (52, 60), (52, 71), (52, 73), (52, 76), (52, 82), (52, 88), (52, 91), (52, 99), (52, 114), (53, 58), (53, 62), (53, 63), (53, 80), (53, 92), (53, 94), (53, 107), (53, 109), (53, 118), (54, 57), (54, 66), (54, 68), (54, 72), (54, 76), (54, 82), (54, 98), (54, 107), (54, 109), (55, 64), (55, 69), (55, 78), (55, 84), (55, 89), (55, 103), (55, 110), (55, 119), (56, 72), (56, 80), (56, 94), (56, 97), (56, 98), (56, 115), (56, 116), (56, 117), (57, 68), (57, 71), (57, 72), (57, 76), (57, 82), (57, 91), (57, 99), (58, 62), (58, 63), (58, 75), (58, 85), (58, 94), (58, 95), (58, 109), (59, 65), (59, 77), (59, 83), (59, 84), (59, 108), (60, 71), (60, 73), (60, 81), (60, 99), (60, 108), (60, 114), (61, 74), (61, 75), (61, 85), (61, 96), (62, 80), (62, 94), (62, 95), (62, 111), (62, 117), (62, 118), (63, 66), (63, 92), (63, 101), (63, 107), (63, 109), (64, 69), (64, 78), (64, 86), (64, 89), (64, 93), (64, 119), (65, 71), (65, 81), (65, 83), (65, 108), (66, 76), (66, 82), (66, 88), (66, 107), (66, 109), (67, 73), (67, 88), (67, 90), (67, 112), (68, 72), (68, 79), (68, 82), (68, 91), (68, 107), (69, 78), (69, 89), (69, 110), (70, 87), (70, 100), (70, 102), (70, 104), (70, 106), (70, 113), (71, 81), (71, 91), (71, 99), (71, 108), (71, 114), (72, 98), (73, 88), (73, 90), (73, 106), (73, 112), (73, 114), (74, 85), (74, 96), (74, 98), (74, 105), (74, 115), (75, 85), (75, 95), (76, 82), (76, 88), (76, 96), (76, 99), (77, 83), (77, 111), (77, 116), (78, 103), (78, 105), (78, 110), (79, 91), (79, 104), (80, 94), (80, 111), (80, 116), (80, 117), (80, 118), (81, 99), (81, 108), (81, 110), (82, 88), (82, 91), (82, 107), (83, 97), (83, 103), (83, 116), (84, 89), (84, 108), (84, 119), (85, 94), (85, 98), (85, 109), (85, 115), (86, 87), (86, 93), (86, 95), (86, 119), (87, 93), (87, 100), (87, 102), (87, 119), (88, 90), (89, 93), (89, 112), (89, 119), (90, 101), (90, 102), (90, 106), (90, 112), (90, 113), (91, 104), (91, 114), (92, 100), (92, 101), (92, 107), (92, 113), (92, 118), (93, 102), (93, 112), (93, 119), (94, 98), (94, 109), (94, 115), (94, 117), (96, 99), (97, 103), (97, 105), (97, 115), (97, 116), (97, 117), (98, 109), (98, 115), (100, 101), (100, 102), (100, 113), (100, 118), (101, 102), (101, 113), (102, 106), (102, 112), (102, 113), (103, 105), (103, 110), (104, 106), (104, 113), (104, 114), (105, 110), (105, 115), (106, 112), (106, 113), (106, 114), (107, 109), (108, 114), (111, 116), (111, 117), (111, 118), (115, 117), (116, 117)], [[0, 3, 297], [0, 4, 298], [0, 7, 300], [0, 8, 301], [0, 9, 302], [1, 5, 313], [1, 6, 314], [1, 7, 315], [1, 9, 317], [1, 11, 319], [2, 3, 468], [2, 6, 471], [2, 8, 472], [2, 10, 474], [2, 11, 475], [3, 8, 518], [3, 9, 520], [3, 11, 521], [4, 5, 545], [4, 7, 547], [4, 8, 548], [4, 10, 549], [5, 6, 578], [5, 7, 579], [5, 10, 580], [6, 10, 640], [6, 11, 641], [7, 9, 646], [8, 10, 649], [9, 11, 704], [12, 13, 48], [12, 17, 50], [12, 18, 51], [12, 22, 56], [12, 23, 58], [13, 15, 250], [13, 16, 252], [13, 17, 253], [13, 23, 259], [14, 18, 339], [14, 19, 340], [14, 20, 341], [14, 21, 343], [14, 22, 345], [15, 16, 355], [15, 19, 357], [15, 21, 361], [15, 23, 363], [16, 17, 382], [16, 20, 387], [16, 21, 388], [17, 18, 414], [17, 20, 418], [18, 20, 456], [18, 22, 458], [19, 21, 562], [19, 22, 563], [19, 23, 564], [20, 21, 657], [22, 23, 716], [24, 25, 59], [24, 28, 61], [24, 31, 63], [24, 32, 65], [24, 33, 67], [25, 26, 70], [25, 28, 72], [25, 29, 73], [25, 33, 79], [26, 27, 92], [26, 29, 94], [26, 33, 98], [26, 34, 101], [27, 29, 147], [27, 30, 149], [27, 34, 153], [27, 35, 155], [28, 29, 260], [28, 32, 262], [28, 35, 267], [29, 35, 283], [30, 31, 406], [30, 32, 408], [30, 34, 410], [30, 35, 412], [31, 32, 539], [31, 33, 540], [31, 34, 543], [32, 35, 568], [33, 34, 621], [36, 37, 102], [36, 41, 105], [36, 42, 106], [36, 43, 107], [36, 47, 111], [37, 41, 244], [37, 44, 246], [37, 46, 248], [37, 47, 249], [38, 39, 262], [38, 40, 263], [38, 42, 265], [38, 43, 266], [38, 45, 267], [39, 40, 566], [39, 44, 567], [39, 45, 568], [39, 46, 569], [40, 41, 582], [40, 42, 583], [40, 46, 584], [41, 42, 642], [41, 46, 644], [42, 43, 663], [43, 45, 667], [43, 47, 671], [44, 45, 675], [44, 46, 676], [44, 47, 677], [45, 47, 699], [48, 50, 253], [48, 54, 255], [48, 55, 256], [48, 58, 259], [49, 50, 294], [49, 51, 295], [49, 52, 299], [49, 53, 300], [49, 55, 302], [50, 51, 414], [50, 55, 419], [51, 52, 455], [51, 56, 458], [52, 53, 626], [52, 56, 627], [52, 57, 628], [53, 54, 645], [53, 55, 646], [53, 57, 647], [54, 55, 687], [54, 57, 690], [54, 58, 691], [56, 57, 715], [56, 58, 716], [57, 58, 719], [59, 61, 72], [59, 62, 74], [59, 64, 77], [59, 67, 79], [60, 62, 199], [60, 64, 202], [60, 66, 203], [60, 68, 206], [60, 69, 209], [61, 62, 261], [61, 65, 262], [61, 66, 263], [62, 64, 288], [62, 66, 289], [63, 65, 539], [63, 67, 540], [63, 68, 541], [63, 69, 544], [64, 67, 557], [64, 68, 558], [65, 66, 566], [65, 69, 570], [66, 69, 585], [67, 68, 620], [68, 69, 653], [70, 71, 93], [70, 73, 94], [70, 76, 96], [70, 79, 98], [71, 73, 168], [71, 75, 169], [71, 76, 170], [71, 78, 172], [72, 73, 260], [72, 74, 261], [72, 78, 264], [73, 78, 280], [74, 75, 286], [74, 77, 288], [74, 78, 290], [75, 76, 332], [75, 77, 333], [75, 78, 334], [76, 77, 482], [76, 79, 485], [77, 79, 557], [80, 81, 137], [80, 83, 139], [80, 88, 141], [80, 89, 142], [80, 91, 145], [81, 82, 157], [81, 84, 160], [81, 89, 164], [81, 91, 166], [82, 84, 232], [82, 85, 233], [82, 87, 235], [82, 89, 239], [83, 85, 366], [83, 86, 368], [83, 88, 370], [83, 89, 371], [84, 87, 422], [84, 90, 424], [84, 91, 426], [85, 86, 446], [85, 87, 447], [85, 89, 449], [86, 87, 545], [86, 88, 546], [86, 90, 547], [87, 90, 579], [88, 90, 626], [88, 91, 628], [90, 91, 647], [92, 94, 147], [92, 99, 150], [92, 100, 152], [92, 101, 153], [93, 94, 168], [93, 96, 170], [93, 97, 171], [93, 100, 175], [94, 100, 282], [95, 96, 383], [95, 97, 384], [95, 98, 385], [95, 99, 387], [95, 101, 388], [96, 97, 483], [96, 98, 485], [97, 99, 573], [97, 100, 575], [98, 101, 621], [99, 100, 656], [99, 101, 657], [102, 105, 244], [102, 108, 245], [102, 109, 247], [102, 111, 249], [103, 104, 490], [103, 105, 492], [103, 106, 493], [103, 108, 494], [103, 112, 496], [104, 106, 610], [104, 107, 611], [104, 110, 612], [104, 112, 614], [105, 106, 642], [105, 108, 643], [106, 107, 663], [107, 110, 669], [107, 111, 671], [108, 109, 672], [108, 112, 673], [109, 110, 705], [109, 111, 706], [109, 112, 707], [110, 111, 711], [110, 112, 712], [113, 116, 189], [113, 117, 191], [113, 118, 192], [113, 119, 193], [113, 120, 195], [114, 115, 221], [114, 116, 222], [114, 120, 226], [114, 122, 229], [114, 124, 231], [115, 121, 307], [115, 122, 308], [115, 123, 309], [115, 124, 311], [116, 117, 331], [116, 120, 335], [116, 122, 336], [117, 119, 461], [117, 122, 466], [117, 123, 467], [118, 119, 469], [118, 120, 470], [118, 121, 471], [118, 124, 475], [119, 121, 552], [119, 123, 553], [120, 124, 597], [121, 123, 639], [121, 124, 641], [122, 123, 686], [125, 127, 138], [125, 128, 139], [125, 129, 140], [125, 131, 142], [125, 136, 146], [126, 127, 241], [126, 131, 243], [126, 132, 246], [126, 134, 247], [126, 135, 249], [127, 131, 349], [127, 132, 351], [127, 136, 353], [128, 129, 364], [128, 130, 369], [128, 131, 371], [128, 134, 372], [129, 130, 391], [129, 133, 394], [129, 136, 396], [130, 133, 599], [130, 134, 601], [130, 135, 603], [131, 134, 633], [132, 133, 674], [132, 135, 677], [132, 136, 678], [133, 135, 696], [133, 136, 697], [134, 135, 706], [137, 138, 159], [137, 142, 164], [137, 143, 165], [137, 145, 166], [138, 142, 349], [138, 143, 350], [138, 146, 353], [139, 140, 364], [139, 141, 370], [139, 142, 371], [140, 141, 392], [140, 144, 395], [140, 146, 396], [141, 144, 627], [141, 145, 628], [143, 144, 635], [143, 145, 636], [143, 146, 638], [144, 145, 715], [144, 146, 717], [147, 152, 282], [147, 155, 283], [147, 156, 284], [148, 149, 338], [148, 150, 341], [148, 151, 342], [148, 153, 343], [148, 154, 344], [149, 153, 410], [149, 154, 411], [149, 155, 412], [150, 151, 655], [150, 152, 656], [150, 153, 657], [151, 152, 659], [151, 154, 660], [151, 156, 661], [152, 156, 679], [154, 155, 694], [154, 156, 695], [155, 156, 698], [157, 160, 232], [157, 162, 236], [157, 163, 238], [157, 164, 239], [158, 159, 320], [158, 161, 323], [158, 162, 324], [158, 163, 325], [158, 165, 326], [159, 162, 348], [159, 164, 349], [159, 165, 350], [160, 161, 421], [160, 163, 423], [160, 166, 426], [161, 163, 523], [161, 165, 524], [161, 166, 529], [162, 163, 590], [162, 164, 591], [165, 166, 636], [167, 169, 189], [167, 172, 194], [167, 173, 195], [167, 174, 197], [167, 176, 198], [168, 172, 280], [168, 175, 282], [168, 176, 285], [169, 170, 332], [169, 172, 334], [169, 173, 335], [170, 171, 483], [170, 173, 484], [171, 173, 571], [171, 174, 574], [171, 175, 575], [172, 176, 589], [173, 174, 596], [174, 175, 664], [174, 176, 665], [175, 176, 680], [177, 179, 210], [177, 180, 211], [177, 181, 212], [177, 182, 213], [177, 186, 215], [178, 182, 232], [178, 183, 234], [178, 184, 235], [178, 185, 237], [178, 186, 238], [179, 180, 304], [179, 181, 305], [179, 187, 307], [179, 188, 309], [180, 182, 312], [180, 184, 313], [180, 187, 314], [181, 183, 398], [181, 186, 399], [181, 188, 404], [182, 184, 422], [182, 186, 423], [183, 185, 532], [183, 186, 533], [183, 188, 537], [184, 185, 577], [184, 187, 578], [185, 187, 604], [185, 188, 606], [187, 188, 639], [189, 191, 331], [189, 194, 334], [189, 195, 335], [190, 192, 428], [190, 193, 429], [190, 196, 430], [190, 197, 432], [190, 198, 435], [191, 193, 461], [191, 194, 462], [191, 196, 463], [192, 193, 469], [192, 195, 470], [192, 197, 473], [193, 196, 551], [194, 196, 586], [194, 198, 589], [195, 197, 596], [196, 198, 613], [197, 198, 665], [199, 202, 288], [199, 203, 289], [199, 205, 291], [199, 207, 293], [200, 201, 397], [200, 204, 400], [200, 205, 401], [200, 207, 402], [200, 208, 403], [201, 203, 507], [201, 205, 510], [201, 208, 512], [201, 209, 514], [202, 204, 556], [202, 206, 558], [202, 207, 559], [203, 205, 581], [203, 209, 585], [204, 206, 615], [204, 207, 616], [204, 208, 617], [205, 207, 624], [206, 208, 652], [206, 209, 653], [208, 209, 692], [210, 211, 304], [210, 212, 305], [210, 216, 306], [210, 219, 310], [211, 213, 312], [211, 217, 316], [211, 219, 318], [212, 215, 399], [212, 216, 400], [212, 218, 403], [213, 214, 421], [213, 215, 423], [213, 217, 425], [214, 215, 523], [214, 217, 526], [214, 218, 527], [214, 220, 528], [215, 218, 609], [216, 218, 617], [216, 219, 618], [216, 220, 619], [217, 219, 688], [217, 220, 689], [218, 220, 693], [219, 220, 709], [221, 227, 306], [221, 229, 308], [221, 230, 310], [221, 231, 311], [222, 224, 332], [222, 225, 333], [222, 226, 335], [222, 229, 336], [223, 224, 374], [223, 225, 375], [223, 227, 376], [223, 228, 378], [223, 230, 380], [224, 225, 482], [224, 226, 484], [224, 228, 486], [225, 227, 556], [225, 229, 559], [226, 228, 595], [226, 231, 597], [227, 229, 616], [227, 230, 618], [228, 230, 630], [228, 231, 631], [230, 231, 708], [232, 235, 422], [232, 238, 423], [233, 235, 447], [233, 237, 448], [233, 239, 449], [233, 240, 450], [234, 236, 531], [234, 237, 532], [234, 238, 533], [234, 240, 536], [235, 237, 577], [236, 238, 590], [236, 239, 591], [236, 240, 593], [237, 240, 605], [239, 240, 632], [241, 242, 348], [241, 243, 349], [241, 246, 351], [241, 248, 352], [242, 243, 591], [242, 244, 592], [242, 245, 593], [242, 248, 594], [243, 245, 632], [243, 247, 633], [244, 245, 643], [244, 248, 644], [245, 247, 672], [246, 248, 676], [246, 249, 677], [247, 249, 706], [250, 251, 354], [250, 252, 355], [250, 258, 362], [250, 259, 363], [251, 252, 373], [251, 254, 378], [251, 257, 380], [251, 258, 381], [252, 253, 382], [252, 254, 386], [253, 254, 417], [253, 256, 419], [254, 256, 629], [254, 257, 630], [255, 256, 687], [255, 257, 688], [255, 258, 689], [255, 259, 691], [256, 257, 703], [257, 258, 709], [258, 259, 718], [260, 264, 280], [260, 266, 281], [260, 267, 283], [261, 263, 289], [261, 264, 290], [261, 265, 292], [262, 263, 566], [262, 267, 568], [263, 265, 583], [264, 265, 587], [264, 266, 588], [265, 266, 663], [266, 267, 667], [268, 269, 427], [268, 272, 431], [268, 274, 432], [268, 277, 433], [268, 278, 435], [269, 270, 438], [269, 271, 440], [269, 272, 441], [269, 277, 443], [270, 271, 477], [270, 272, 479], [270, 273, 480], [270, 279, 481], [271, 273, 598], [271, 276, 600], [271, 277, 602], [272, 274, 648], [272, 279, 650], [273, 275, 659], [273, 276, 661], [273, 279, 662], [274, 275, 664], [274, 278, 665], [274, 279, 666], [275, 276, 679], [275, 278, 680], [275, 279, 681], [276, 277, 700], [276, 278, 701], [277, 278, 710], [280, 281, 588], [280, 285, 589], [281, 283, 667], [281, 284, 668], [281, 285, 670], [282, 284, 679], [282, 285, 680], [283, 284, 698], [284, 285, 701], [286, 287, 331], [286, 288, 333], [286, 290, 334], [286, 293, 336], [287, 290, 462], [287, 291, 464], [287, 292, 465], [287, 293, 466], [288, 293, 559], [289, 291, 581], [289, 292, 583], [290, 292, 587], [291, 292, 623], [291, 293, 624], [294, 295, 414], [294, 297, 415], [294, 302, 419], [294, 303, 420], [295, 296, 454], [295, 299, 455], [295, 303, 459], [296, 298, 476], [296, 299, 478], [296, 301, 479], [296, 303, 481], [297, 301, 518], [297, 302, 520], [297, 303, 522], [298, 299, 546], [298, 300, 547], [298, 301, 548], [299, 300, 626], [300, 302, 646], [301, 303, 650], [304, 307, 314], [304, 310, 318], [304, 311, 319], [305, 306, 400], [305, 308, 402], [305, 309, 404], [306, 308, 616], [306, 310, 618], [307, 309, 639], [307, 311, 641], [308, 309, 686], [310, 311, 708], [312, 313, 422], [312, 315, 424], [312, 316, 425], [313, 314, 578], [313, 315, 579], [314, 319, 641], [315, 316, 645], [315, 317, 646], [316, 317, 687], [316, 318, 688], [317, 318, 703], [317, 319, 704], [318, 319, 708], [320, 321, 347], [320, 324, 348], [320, 326, 350], [320, 329, 352], [321, 326, 500], [321, 327, 502], [321, 329, 503], [321, 330, 504], [322, 324, 508], [322, 325, 509], [322, 328, 512], [322, 329, 513], [322, 330, 514], [323, 325, 523], [323, 326, 524], [323, 327, 525], [323, 328, 527], [324, 325, 590], [324, 329, 594], [325, 328, 609], [326, 327, 634], [327, 328, 682], [327, 330, 683], [328, 330, 692], [329, 330, 713], [331, 334, 462], [331, 336, 466], [332, 333, 482], [332, 335, 484], [333, 336, 559], [337, 339, 389], [337, 342, 393], [337, 344, 394], [337, 345, 395], [337, 346, 396], [338, 340, 407], [338, 343, 410], [338, 344, 411], [338, 346, 413], [339, 341, 456], [339, 342, 457], [339, 345, 458], [340, 343, 562], [340, 345, 563], [340, 346, 565], [341, 342, 655], [341, 343, 657], [342, 344, 660], [344, 346, 697], [345, 346, 717], [347, 350, 500], [347, 351, 501], [347, 352, 503], [347, 353, 505], [348, 349, 591], [348, 352, 594], [350, 353, 638], [351, 352, 676], [351, 353, 678], [354, 355, 373], [354, 358, 377], [354, 359, 379], [354, 362, 381], [355, 358, 385], [355, 361, 388], [356, 357, 538], [356, 358, 540], [356, 359, 541], [356, 360, 542], [356, 361, 543], [357, 360, 561], [357, 361, 562], [357, 363, 564], [358, 359, 620], [358, 361, 621], [359, 360, 651], [359, 362, 654], [360, 362, 684], [360, 363, 685], [362, 363, 718], [364, 367, 390], [364, 369, 391], [364, 370, 392], [365, 366, 437], [365, 367, 438], [365, 368, 439], [365, 369, 440], [365, 372, 442], [366, 368, 446], [366, 371, 449], [366, 372, 451], [367, 368, 476], [367, 369, 477], [367, 370, 478], [368, 370, 546], [369, 372, 601], [371, 372, 633], [373, 374, 383], [373, 377, 385], [373, 378, 386], [374, 375, 482], [374, 377, 485], [374, 378, 486], [375, 376, 556], [375, 377, 557], [375, 379, 558], [376, 379, 615], [376, 380, 618], [376, 381, 619], [377, 379, 620], [378, 380, 630], [379, 381, 654], [380, 381, 709], [382, 384, 416], [382, 386, 417], [382, 387, 418], [383, 384, 483], [383, 385, 485], [383, 386, 486], [384, 386, 572], [384, 387, 573], [385, 388, 621], [387, 388, 657], [389, 390, 454], [389, 392, 455], [389, 393, 457], [389, 395, 458], [390, 391, 477], [390, 392, 478], [390, 393, 480], [391, 393, 598], [391, 394, 599], [392, 395, 627], [393, 394, 660], [394, 396, 697], [395, 396, 717], [397, 398, 506], [397, 399, 509], [397, 401, 510], [397, 403, 512], [398, 399, 533], [398, 401, 534], [398, 404, 537], [399, 403, 609], [400, 402, 616], [400, 403, 617], [401, 402, 624], [401, 404, 625], [402, 404, 686], [405, 406, 497], [405, 407, 498], [405, 408, 499], [405, 409, 501], [405, 413, 505], [406, 407, 538], [406, 408, 539], [406, 410, 543], [407, 410, 562], [407, 413, 565], [408, 409, 567], [408, 412, 568], [409, 411, 674], [409, 412, 675], [409, 413, 678], [411, 412, 694], [411, 413, 697], [414, 418, 456], [414, 420, 459], [415, 416, 515], [415, 417, 517], [415, 419, 520], [415, 420, 522], [416, 417, 572], [416, 418, 573], [416, 420, 576], [417, 419, 629], [418, 420, 658], [421, 423, 523], [421, 425, 526], [421, 426, 529], [422, 424, 579], [424, 425, 645], [424, 426, 647], [425, 426, 690], [427, 431, 441], [427, 433, 443], [427, 434, 444], [427, 436, 445], [428, 429, 469], [428, 431, 472], [428, 432, 473], [428, 434, 474], [429, 430, 551], [429, 434, 554], [429, 436, 555], [430, 433, 612], [430, 435, 613], [430, 436, 614], [431, 432, 648], [431, 434, 649], [432, 435, 665], [433, 435, 710], [433, 436, 712], [434, 436, 714], [437, 439, 446], [437, 442, 451], [437, 444, 452], [437, 445, 453], [438, 439, 476], [438, 440, 477], [438, 441, 479], [439, 441, 548], [439, 444, 549], [440, 442, 601], [440, 443, 602], [441, 444, 649], [442, 443, 705], [442, 445, 707], [443, 445, 712], [444, 445, 714], [446, 447, 545], [446, 452, 549], [447, 448, 577], [447, 452, 580], [448, 450, 605], [448, 452, 607], [448, 453, 608], [449, 450, 632], [449, 451, 633], [450, 451, 672], [450, 453, 673], [451, 453, 707], [452, 453, 714], [454, 455, 478], [454, 457, 480], [454, 459, 481], [455, 458, 627], [456, 457, 655], [456, 459, 658], [457, 459, 662], [460, 461, 488], [460, 463, 490], [460, 464, 491], [460, 465, 493], [460, 467, 495], [461, 463, 551], [461, 467, 553], [462, 463, 586], [462, 465, 587], [463, 465, 610], [464, 465, 623], [464, 466, 624], [464, 467, 625], [466, 467, 686], [468, 470, 516], [468, 472, 518], [468, 473, 519], [468, 475, 521], [469, 471, 552], [469, 474, 554], [470, 473, 596], [470, 475, 597], [471, 474, 640], [471, 475, 641], [472, 473, 648], [472, 474, 649], [476, 478, 546], [476, 479, 548], [477, 480, 598], [479, 481, 650], [480, 481, 662], [482, 485, 557], [483, 484, 571], [483, 486, 572], [484, 486, 595], [487, 489, 532], [487, 491, 534], [487, 492, 535], [487, 494, 536], [487, 495, 537], [488, 489, 550], [488, 490, 551], [488, 495, 553], [488, 496, 555], [489, 494, 605], [489, 495, 606], [489, 496, 608], [490, 493, 610], [490, 496, 614], [491, 492, 622], [491, 493, 623], [491, 495, 625], [492, 493, 642], [492, 494, 643], [494, 496, 673], [497, 498, 538], [497, 499, 539], [497, 502, 542], [497, 504, 544], [498, 500, 560], [498, 502, 561], [498, 505, 565], [499, 501, 567], [499, 503, 569], [499, 504, 570], [500, 502, 634], [500, 505, 638], [501, 503, 676], [501, 505, 678], [502, 504, 683], [503, 504, 713], [506, 508, 531], [506, 509, 533], [506, 510, 534], [506, 511, 535], [507, 510, 581], [507, 511, 582], [507, 513, 584], [507, 514, 585], [508, 509, 590], [508, 511, 592], [508, 513, 594], [509, 512, 609], [510, 511, 622], [511, 513, 644], [512, 514, 692], [513, 514, 713], [515, 516, 571], [515, 517, 572], [515, 519, 574], [515, 522, 576], [516, 517, 595], [516, 519, 596], [516, 521, 597], [517, 520, 629], [517, 521, 631], [518, 519, 648], [518, 522, 650], [519, 522, 666], [520, 521, 704], [523, 527, 609], [524, 525, 634], [524, 529, 636], [524, 530, 637], [525, 527, 682], [525, 528, 684], [525, 530, 685], [526, 528, 689], [526, 529, 690], [526, 530, 691], [527, 528, 693], [528, 530, 718], [529, 530, 719], [531, 533, 590], [531, 535, 592], [531, 536, 593], [532, 536, 605], [532, 537, 606], [534, 535, 622], [534, 537, 625], [535, 536, 643], [538, 542, 561], [538, 543, 562], [539, 544, 570], [540, 541, 620], [540, 543, 621], [541, 542, 651], [541, 544, 653], [542, 544, 683], [545, 547, 579], [545, 549, 580], [546, 547, 626], [548, 549, 649], [550, 552, 604], [550, 553, 606], [550, 554, 607], [550, 555, 608], [551, 555, 614], [552, 553, 639], [552, 554, 640], [554, 555, 714], [556, 558, 615], [556, 559, 616], [557, 558, 620], [560, 561, 634], [560, 563, 635], [560, 564, 637], [560, 565, 638], [561, 564, 685], [563, 564, 716], [563, 565, 717], [566, 569, 584], [566, 570, 585], [567, 568, 675], [567, 569, 676], [569, 570, 713], [571, 572, 595], [571, 574, 596], [573, 575, 656], [573, 576, 658], [574, 575, 664], [574, 576, 666], [575, 576, 681], [577, 578, 604], [577, 580, 607], [578, 580, 640], [581, 582, 622], [581, 583, 623], [582, 583, 642], [582, 584, 644], [584, 585, 713], [586, 587, 610], [586, 588, 611], [586, 589, 613], [587, 588, 663], [588, 589, 670], [591, 593, 632], [592, 593, 643], [592, 594, 644], [595, 597, 631], [598, 599, 660], [598, 600, 661], [599, 600, 695], [599, 603, 696], [600, 602, 700], [600, 603, 702], [601, 602, 705], [601, 603, 706], [602, 603, 711], [604, 606, 639], [604, 607, 640], [605, 608, 673], [607, 608, 714], [610, 611, 663], [611, 612, 669], [611, 613, 670], [612, 613, 710], [612, 614, 712], [615, 617, 652], [615, 619, 654], [617, 619, 693], [618, 619, 709], [622, 623, 642], [624, 625, 686], [626, 628, 647], [627, 628, 715], [629, 630, 703], [629, 631, 704], [630, 631, 708], [632, 633, 672], [634, 637, 685], [635, 636, 715], [635, 637, 716], [635, 638, 717], [636, 637, 719], [645, 646, 687], [645, 647, 690], [648, 650, 666], [651, 652, 682], [651, 653, 683], [651, 654, 684], [652, 653, 692], [652, 654, 693], [655, 656, 659], [655, 658, 662], [656, 658, 681], [659, 661, 679], [659, 662, 681], [660, 661, 695], [664, 665, 680], [664, 666, 681], [667, 668, 698], [667, 671, 699], [668, 669, 700], [668, 670, 701], [668, 671, 702], [669, 670, 710], [669, 671, 711], [672, 673, 707], [674, 675, 694], [674, 677, 696], [674, 678, 697], [675, 677, 699], [679, 680, 701], [682, 683, 692], [682, 684, 693], [684, 685, 718], [687, 688, 703], [688, 689, 709], [689, 691, 718], [690, 691, 719], [694, 695, 698], [694, 696, 699], [695, 696, 702], [698, 699, 702], [700, 701, 710], [700, 702, 711], [703, 704, 708], [705, 706, 711], [705, 707, 712], [715, 716, 719]]]

    """
    vectors = get_4D_vectors()
    print(len(vectors)," vertices")

    elements = [FVector(v) for v in vectors]
    # edges in terms of vertices
    if not os.path.exists(os.path.join(PATH,"edges.data")):
        edges = detect_edges(elements)
        save(edges,"edges.data")
    edges = read("edges.data")
    print(len(edges)," edges")

    # faces in terms of edges
    if not os.path.exists(os.path.join(PATH,"faces.data")):
        faces = detect_faces(edges)
        save(faces,"faces.data")
    faces = read("faces.data")
    print(len(faces)," faces")

    #cells in terms of faces
    if not os.path.exists(os.path.join(PATH,"cells.data")):
        cells = detect_cells(faces,edges)
        save(cells,"cells.data")
    cells = read("cells.data")
    print(len(cells)," cells")

    return [vectors,edges,faces]

def get_3D_geometry(pre_path="",offset=0.5):
    elements = list(generate_group())
    vectors = get_4D_vectors()

    # edges in terms of vertices
    edges = read("edges.data",pre_path)
    print("number of edges ",len(edges),edges)

    # faces in terms of edges

    faces = read("faces.data",pre_path)
    print("number of faces ",len(faces),faces)

    # cells in terms of faces

    cells = read("cells.data",pre_path)

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

    if filename != "":
        dir = os.path.join(dir, filename)
        vectors = get_4D_vectors()
        edges = read("edges.data")
        faces = read("faces.data")
        cells = read("cells.data")

        # find smallest z-value of a cell-center for each vertex
        w_cell_centers = []
        w_cells = []
        for i in range(120):
            w_cell_centers.append(-np.inf)
            w_cells.append(-1)

        for k, cell in enumerate(cells):
            cell_faces = [faces[i] for i in cell]
            cell_edges = [edges[j] for c in cell_faces for j in c]
            cell_vertex_indices = set([i for e in cell_edges for i in e])
            cell_vertices = set([tuple(vectors[j]) for e in cell_edges for j in e])
            cell_vertices = [Vector([float(comp.real()) for comp in v]) for v in cell_vertices]
            center = max([v[3] for v in cell_vertices])

            for i in cell_vertex_indices:
                if center > w_cell_centers[i]:
                    w_cell_centers[i] = center
                    w_cells[i] = k

        # print(set(w_cell_centers))

        with open(dir, "w") as f:
            f.write("x,y,z,w,center_w,cell\n")
            for v, w_center, cell in zip(vectors, w_cell_centers, w_cells):
                f.write(f"{v[0].real()},{v[1].real()},{v[2].real()},{v[3].real()},{w_center},{cell}\n")
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