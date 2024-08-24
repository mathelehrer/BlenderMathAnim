import numpy as np
from anytree import NodeMixin
from matplotlib import pyplot
from numpy.linalg import inv, det

from video_apollonian.indras_utils.circle import IndraCircle
from video_apollonian.indras_utils.mymath import moebius_on_point, moebius_on_circle, cx_sqrt


class KissingSchottky:
    def __init__(self, y=1, k=1):
        self.y = y
        self.k = k

    def get_fixed_points(self):
        return [1j * self.k, 1, -1j * self.k, -1]

    def get_generators(self):
        gens = []

        i = 1j
        y = self.y
        k = self.k
        x = np.sqrt(1 + y * y)
        v = 2 / y / (k + 1 / k)
        u = np.sqrt(1 + v * v)

        gens.append(np.array([[u, i * k * v], [-i * v / k, u]]))
        a = gens[-1]
        gens.append(np.array([[x, y], [y, x]]))
        b = gens[-1]
        gens.append(np.linalg.inv(a))
        A = gens[-1]
        gens.append(np.linalg.inv(b))
        B = gens[-1]

        return [a, b, A, B]

    def get_circles(self):
        i = 1j
        y = self.y
        k = self.k
        x = np.sqrt(1 + y * y)
        v = 2 / y / (k + 1 / k)
        u = np.sqrt(1 + v * v)

        return [
            IndraCircle(i * k * u / v, k / v),
            IndraCircle(x / y, 1 / y),
            IndraCircle(-i * k * u / v, k / v),
            IndraCircle(-x / y, 1 / y)
        ]


def k_of_s(s):
    '''
    Condition for having Shottky discs of equal size within a Shottky pair

    :param s:
    :return:
    '''
    return (1 + s - 2 * np.sqrt(s)) / (s - 1)


class BlenderModel:
    """
    This is the model used for the blender simulations
    It was derived heuristically from rotations of the Riemann sphere
    """

    def __init__(self):
        # setup initial conditions
        R1 = 1.5
        R2 = R1 * 3.247639506  # touching condition calculated in asymmetric_touching_discs.mw
        s = 4  # scaling
        k = k_of_s(s)

        a = np.array([[s + 1, 2 * R2 * (1 - s)], [(1 - s) / 2 / R2, s + 1]])
        A = np.array([[s + 1, 2 * R2 * (s - 1)], [(s - 1) / 2 / R2, s + 1]])
        b = np.array([[s + 1, 2 * R1 * 1j * (1 - s)], [(1 - s) / 2 / R1 / 1j, s + 1]])
        B = np.array([[s + 1, 2 * R1 * 1j * (s - 1)], [(s - 1) / 2 / R1 / 1j, s + 1]])
        self.gens = [a, b, A, B]
        self.labels = ['a', 'b', 'A', 'B']
        self.colors = ['custom1', 'text', 'custom4', 'custom3']

        # create discs
        # vertically the smaller disc
        rs = R1 * (1 / k - k)
        cs = R1 * 1j * (1 / k + k)
        # horizontally the bigger disc
        rb = R2 * (1 / k - k)
        cb = R2 * (1 / k + k)

        c_a = IndraCircle(-cb, rb)
        c_A = IndraCircle(cb, rb)
        c_b = IndraCircle(-cs, rs)
        c_B = IndraCircle(cs, rs)

        self.dics = [c_a, c_b, c_A, c_B]
        self.fixed_points = [fixed_point_of(g) for g in self.gens]
        self.commutators = self.get_commutators()

    def get_generators(self):
        return self.gens

    def get_circles(self):
        return self.dics

    def get_fixed_points(self):
        return self.fixed_points

    def get_commutators(self):
        id = np.array([[1, 0], [0, 1]])
        commutators = []
        for i in range(1, 5):
            left_most = id
            left_word = ''
            for j in range(4):
                left_word += self.labels[(i - j) % 4]
            right_most = id
            right_word = ''
            for j in range(4):
                right_word += self.labels[(i + j) % 4]
            print(left_word)
            print(right_word)
        return commutators


class ApollonianModel:
    """
    This model is an approximation to the setup that will generate the Apollonian Gasket
    """

    def __init__(self):
        # setup initial conditions
        R1 = 10000
        R2 = 1  # touching condition calculated in asymmetric_touching_discs.mw

        a = np.array([[1, 0], [-2j, 1]])
        A = np.array([[1, 0], [2j, 1]])
        b = np.array([[1 - 1j, 1], [1, 1 + 1j]])
        B = np.array([[1 + 1j, -1], [-1, 1 - 1j]])
        self.gens = [a, b, A, B]
        self.labels = ['a', 'b', 'A', 'B']
        self.colors = ['gray_2', 'joker', 'important', 'custom1']
        # self.colors = ['custom1', 'text', 'custom4', 'custom3']

        c_a = IndraCircle(R1 * 1j, R1)
        c_A = IndraCircle(-0.25j, 0.25)
        c_b = IndraCircle(1 - 1j, R2)
        c_B = IndraCircle(-1 - 1j, R2)

        self.discs = [c_a, c_b, c_A, c_B]
        self.circle_map = {'a': c_a, 'b': c_b, 'A': c_A, 'B': c_B}
        self.color_map = {'a': self.colors[0], 'b': self.colors[1], 'A': self.colors[2], 'B': self.colors[3]}
        self.fixed_points = [fixed_point_of(g) for g in self.gens]
        self.commutators = self.get_commutators()
        self.commutator_fixed_points = [fixed_point_of(c) for c in self.commutators]
        self.inverses = [2, 3, 0, 1]

    def get_labels(self):
        return self.labels

    def get_commutator_fixed_points(self):
        return self.commutator_fixed_points

    def get_generators(self):
        return self.gens

    def get_circles(self):
        return self.discs

    def get_fixed_points(self):
        return self.fixed_points

    def get_commutators(self):
        '''
        build the generators

        a-: BAba
        a+: bABa
        b-: aBAb
        b+: ABab
        A-: baBA
        A+: BabA
        B-: AbaB
        B+: abAB

        :return:
        '''
        id = np.array([[1, 0], [0, 1]])
        commutators = []
        for i in range(1, 5):
            left_most = id
            left_word = ''
            for j in range(4):
                left_word += self.labels[(2 + i - j) % 4]
                left_most = left_most.dot(self.gens[(2 + i - j) % 4])
            right_most = id
            right_word = ''
            for j in range(4):
                right_word += self.labels[(i + j) % 4]
                right_most = right_most.dot(self.gens[(i + j) % 4])
            commutators.append(left_most)
            commutators.append(right_most)
            # print(left_word)
            # print(right_word)
        return commutators

    def get_inverses(self):
        return self.inverses


class ThetaModel:
    """
       This model is from Indra's Pearl on p. 118

    """

    def __init__(self, theta=np.pi / 4):
        # setup initial conditions
        r = np.tan(theta)
        m = 1 / np.cos(theta)

        s = np.sin(theta)
        c = np.cos(theta)

        a = 1 / s * np.array([[1, c * 1j], [-c * 1j, 1]])
        b = 1 / s * np.array([[1, c], [c, 1]])
        B = 1 / s * np.array([[1, -c], [-c, 1]])
        A = 1 / s * np.array([[1, -c * 1j], [c * 1j, 1]])
        self.gens = [a, b, A, B]
        self.labels = ['a', 'b', 'A', 'B']
        self.colors = ['custom1', 'text', 'custom4', 'custom3']

        c_a = IndraCircle(1j * m, r)
        c_A = IndraCircle(-1j * m, r)
        c_b = IndraCircle(m, r)
        c_B = IndraCircle(-m, r)

        self.discs = [c_a, c_b, c_A, c_B]
        self.circle_map = {'a': c_a, 'b': c_b, 'A': c_A, 'B': c_B}
        self.color_map = {'a': self.colors[0], 'b': self.colors[1], 'A': self.colors[2], 'B': self.colors[3]}
        self.fixed_points = [fixed_point_of(g) for g in self.gens]
        self.commutators = self.get_commutators()
        self.commutator_fixed_points = [fixed_point_of(c) for c in self.commutators]
        self.inverses = [2, 3, 0, 1]

    def get_labels(self):
        return self.labels

    def get_commutator_fixed_points(self):
        return self.commutator_fixed_points

    def get_generators(self):
        return self.gens

    def get_circles(self):
        return self.discs

    def get_fixed_points(self):
        return self.fixed_points

    def get_commutators(self):
        '''
        build the generators

        a-: BAba
        a+: bABa
        b-: aBAb
        b+: ABab
        A-: baBA
        A+: BabA
        B-: AbaB
        B+: abAB

        :return:
        '''
        id = np.array([[1, 0], [0, 1]])
        commutators = []
        for i in range(1, 5):
            left_most = id
            left_word = ''
            for j in range(4):
                left_word += self.labels[(2 + i - j) % 4]
                left_most = left_most.dot(self.gens[(2 + i - j) % 4])
            right_most = id
            right_word = ''
            for j in range(4):
                right_word += self.labels[(i + j) % 4]
                right_most = right_most.dot(self.gens[(i + j) % 4])
            commutators.append(left_most)
            commutators.append(right_most)
            # print(left_word)
            # print(right_word)
        return commutators

    def get_inverses(self):
        return self.inverses


class SchottkyFamily:
    """
        This model is from Indra's Pearl on p. 170
    """

    def __init__(self, y, k):
        # setup initial conditions
        i = 1j
        x = np.sqrt(1 + y * y)
        v = 2 / y / (k * 1 / k)
        x = np.sqrt(1 + y * y)
        v = 2 / y / (k + 1 / k)
        u = np.sqrt(1 + v * v)

        a = np.array([[u, i * k * v], [-i * v / k, u]])
        b = np.array([[x, y], [y, x]])
        B = inv(b)
        A = inv(a)

        self.gens = [a, b, A, B]
        self.labels = ['a', 'b', 'A', 'B']
        self.colors = ['custom1', 'text', 'custom4', 'custom3']

        c_a = IndraCircle(i * k * u / v, k / v)
        c_A = IndraCircle(-i * k * u / v, k / v)
        c_b = IndraCircle(x / y, 1 / y)
        c_B = IndraCircle(-x / y, 1 / y)

        self.discs = [c_a, c_b, c_A, c_B]
        self.circle_map = {'a': c_a, 'b': c_b, 'A': c_A, 'B': c_B}
        self.color_map = {'a': self.colors[0], 'b': self.colors[1], 'A': self.colors[2], 'B': self.colors[3]}
        self.fixed_points = [fixed_point_of(g) for g in self.gens]
        self.commutators = self.get_commutators()
        self.commutator_fixed_points = [fixed_point_of(c) for c in self.commutators]
        self.inverses = [2, 3, 0, 1]

    def get_labels(self):
        return self.labels

    def get_commutator_fixed_points(self):
        return self.commutator_fixed_points

    def get_generators(self):
        return self.gens

    def get_circles(self):
        return self.discs

    def get_fixed_points(self):
        return self.fixed_points

    def get_commutators(self):
        '''
        build the generators

        a-: BAba
        a+: bABa
        b-: aBAb
        b+: ABab
        A-: baBA
        A+: BabA
        B-: AbaB
        B+: abAB

        :return:
        '''
        id = np.array([[1, 0], [0, 1]])
        commutators = []
        for i in range(1, 5):
            left_most = id
            left_word = ''
            for j in range(4):
                left_word += self.labels[(2 + i - j) % 4]
                left_most = left_most.dot(self.gens[(2 + i - j) % 4])
            right_most = id
            right_word = ''
            for j in range(4):
                right_word += self.labels[(i + j) % 4]
                right_most = right_most.dot(self.gens[(i + j) % 4])
            commutators.append(left_most)
            commutators.append(right_most)
            # print(left_word)
            # print(right_word)
        return commutators

    def get_inverses(self):
        return self.inverses


class TransformedModel:
    def __init__(self,model=ApollonianModel,transformation=np.array([[1,0],[0,1]]),**kwargs):
        self.model = model(**kwargs)
        self.colors = self.model.colors
        ti = inv(transformation)
        t=transformation
        self.gens =[t@g@ti for g in self.model.get_generators()]
        self.labels = self.model.get_labels()
        self.circles = [moebius_on_circle(t,c) for c in self.model.discs]
        self.fixed_points = [moebius_on_point(t,p) for p in self.model.fixed_points]

    def get_generators(self):
        return self.gens

    def get_fixed_points(self):
        return self.fixed_points

    def get_circles(self):
        return self.circles

class GrandMasRecipe:
    """
    This model is from Indra's Pearl on p. 229
    :param ta
    :param tb
    two complex numbers that scan through the restricted parameter space of connected Kleinian fractals up to conjugation
    """

    def __init__(self, ta=2, tb=2,truncation=True):
        # setup initial conditions
        self.ta = ta
        self.tb = tb
        self.truncation=truncation

        tab = ta * tb / 2 - cx_sqrt(ta * ta * tb * tb / 4 - ta * ta - tb * tb)
        z0 = (tab - 2) * tb / (tb * tab - 2 * ta + 2j * tab)

        a = np.array([[ta / 2, (ta * tab - 2 * tb + 4j) / (2 * tab + 4) / z0],
                      [(ta * tab - 2 * tb - 4j) * z0 / (2 * tab - 4), ta / 2]])
        b = 0.5 * np.array([[tb - 2j, tb], [tb, tb + 2j]])
        B = inv(b)
        A = inv(a)

        self.gens = [a, b, A, B]
        self.labels = ['a', 'b', 'A', 'B']
        self.colors = ['custom1', 'text', 'custom4', 'custom3']

        self.discs = self.get_circles()
        [c_a, c_b, c_A, c_B] = self.discs
        self.circle_map = {'a': c_a, 'b': c_b, 'A': c_A, 'B': c_B}
        self.color_map = {'a': self.colors[0], 'b': self.colors[1], 'A': self.colors[2], 'B': self.colors[3]}
        self.fixed_points = [fixed_point_of(g) for g in self.gens]
        self.commutators = self.get_commutators()
        self.commutator_fixed_points = [fixed_point_of(c) for c in self.commutators]
        self.inverses = [2, 3, 0, 1]

    def get_circles(self):
        """
        this function only works for a very limited range of parameters so far
        2.01<ta<2.1 and 2.01<tb<6.5
        or
        2.01<ta<6.5 and 2.01<tb<2.1

        if complex traces are passed, an approximation for real traces are calculated.
        if trunctation ==True, the imaginary part is dropped, otherwise the absolute value of the trace is used

        :return:
        """
        [a,b,A,B]=self.gens

        kb = k_of(b)
        gb = g_of(b)
        gbi = inv(gb)

        c_b = moebius_on_circle(gbi, IndraCircle(0, np.sqrt(kb)))
        c_B = moebius_on_circle(gbi, IndraCircle(0, 1 / np.sqrt(kb)))

        ga = g_of(a)
        gai = inv(ga)
        image = moebius_on_circle(ga, c_b)
        ka = np.abs(image.c) + image.r

        c_a = IndraCircle(10000j, 10000)
        c_A = moebius_on_circle(A,c_a)

        return [c_a, c_b, c_A, c_B]

    def get_labels(self):
        return self.labels

    def get_commutator_fixed_points(self):
        return self.commutator_fixed_points

    def get_generators(self):
        return self.gens

    def get_fixed_points(self):
        return self.fixed_points

    def get_commutators(self):
        '''
        build the generators

        a-: BAba
        a+: bABa
        b-: aBAb
        b+: ABab
        A-: baBA
        A+: BabA
        B-: AbaB
        B+: abAB

        :return:
        '''
        id = np.array([[1, 0], [0, 1]])
        commutators = []
        for i in range(1, 5):
            left_most = id
            left_word = ''
            for j in range(4):
                left_word += self.labels[(2 + i - j) % 4]
                left_most = left_most.dot(self.gens[(2 + i - j) % 4])
            right_most = id
            right_word = ''
            for j in range(4):
                right_word += self.labels[(i + j) % 4]
                right_most = right_most.dot(self.gens[(i + j) % 4])
            commutators.append(left_most)
            commutators.append(right_most)
            # print(left_word)
            # print(right_word)
        return commutators

    def get_inverses(self):
        return self.inverses


class GroupTree(NodeMixin):
    def __init__(self, element, circle=None, level=0, color='drawing', tag=None, word=''):
        self.element = element
        self.circle = circle
        self.level = level
        self.tag = tag
        self.color = color
        if word == '':
            self.word = tag
        else:
            self.word = word

    def __str__(self):
        if self.tag:
            return self.tag
        else:
            return 'id'

    def mathematica_string(self):
        return "{{" + str(self.element[0][0]) + "," + str(self.element[0][1]) + "},{" + str(
            self.element[1][0]) + "," + str(self.element[1][1]) + "}}"

    def __repr__(self):
        return str(self) + " " + self.mathematica_string()

    def get_element(self):
        return self.element

class DepthFirstSearchByLevel:
    def __init__(self, family, max_level=None, **kwargs):
        self.lev = None
        self.max_level = max_level
        self.old_point = None
        model = family(**kwargs)

        self.colors = model.colors
        self.gens = model.get_generators()
        self.labels = model.labels
        self.fixed_points = model.get_fixed_points()
        self.circles = model.get_circles()

        self.inv = [2, 3, 0, 1]

        self.word = None
        self.tags = None
        self.points = []  # field to store the limit points
        self.circs = []  # field to store all circles

        id = np.array([[1, 0], [0, 1]])
        self.tree = GroupTree(id)

    def start_simple(self):
        '''
        print the four discs and nothing else
        :return:
        '''

        return self.circles

    def generate_tree(self):
        '''
        generate the tree
        :return:
        '''
        self.generate_recursively(self.tree)
        return self.tree

    def generate_next_level(self, level, leaves):
        '''
        Here new discs are created by multiplication from the left or from the right
        :param level:
        :param leaves:
        :return:
        '''
        # from the right
        new_leaves = []
        words = []
        for leave in leaves:
            tag = leave.word[-1]
            if tag == 'a':
                children = [3, 0, 1]
            elif tag == 'b':
                children = [0, 1, 2]
            elif tag == 'A':
                children = [1, 2, 3]
            else:
                children = [2, 3, 0]
            for c in children:
                element = np.dot(leave.element, self.gens[c])
                new_circle = moebius_on_circle(leave.element, self.circles[c])
                word = leave.word + self.labels[c]
                if not word in words:
                    node = GroupTree(element, circle=new_circle, level=level, color=leave.color, tag=self.labels[c],
                                     word=word)
                    new_leaves.append(node)
                    words.append(word)
        return new_leaves

    def generate_recursively(self, tree):
        if tree.level == 0:
            # start with the first level
            for gen, circle, tag, color in zip(self.gens, self.circles, self.labels, self.colors):
                node = GroupTree(gen, circle=circle, level=1, tag=tag, color=color)
                node.parent = tree
                self.generate_recursively(node)
        elif self.go_deeper(tree):
            parent_tag = tree.tag
            if parent_tag[-1] == 'a':
                children = [3, 0, 1]
            elif parent_tag[-1] == 'b':
                children = [0, 1, 2]
            elif parent_tag[-1] == 'A':
                children = [1, 2, 3]
            else:
                children = [2, 3, 0]
            for c in children:
                new_circle = moebius_on_circle(self.gens[c], tree.circle)
                tag = parent_tag + self.labels[c]
                element = np.dot(tree.element, self.gens[c])
                node = GroupTree(element, circle=new_circle, level=tree.level + 1, color=tree.color, tag=tag)
                node.parent = tree
                self.generate_recursively(node)

    def go_deeper(self, tree):
        '''

        check the condition for further recursion,
        either the max_level hasn't been reached
        or the size of the discs is not small enough yet

        :param tree:
        :return:
        '''
        if tree.level < self.max_level:
            return True
        return False

    def get_leaves(self):
        '''
        returns the discs at the leaves of the tree
        :return:
        '''
        leaves = []
        self.collect_leaves_recursively(self.tree, leaves)
        return leaves

    def collect_leaves_recursively(self, tree, leaves):
        if not tree.children:
            leaves.append(tree)
        else:
            for child in tree.children:
                self.collect_leaves_recursively(child, leaves)


class DepthFirstSearchWithFixedPoints:
    def __init__(self, family, max_level=None, max_sep=None, **kwargs):
        # needed for the adaptive generation of the tree
        self.final_point = None
        self.old_point = None
        self.max_sep = max_sep
        self.points = []

        # needed for the level-dependent generation of the tree
        self.lev = None
        self.max_level = max_level

        model = family(**kwargs)

        self.colors = model.colors
        self.gens = model.get_generators()
        self.labels = model.get_labels()
        self.fixed_points = model.get_commutator_fixed_points()
        self.circles = model.get_circles()
        self.inv = model.get_inverses()

        self.word = None
        self.tags = None
        self.points = []  # field to store the limit points

        id = np.array([[1, 0], [0, 1]])
        self.tree = GroupTree(id)

    def start_simple(self):
        '''
        print the four discs and nothing else
        :return:
        '''

        return self.fixed_points

    def analyse_fixed_points(self):
        for g, label in zip(self.gens, self.labels):
            print(label, ":")
            print(g)
            node = GroupTree(g, circle=None, level=1, color='drawing', tag=label)
            print(self.get_left_most_point(node))
            print(self.get_right_most_point(node))

    def generate_tree(self):
        '''
        generate the tree
        :return:
        '''

        # simulated end node
        node = GroupTree

        # start with the first level
        for gen, tag, color in zip(self.gens, self.labels, self.colors):
            node = GroupTree(gen, circle=None, level=1, tag=tag, color=color)
            node.parent = self.tree

        self.old_point = self.get_right_most_point(self.tree.children[-1])  # obtain the last point of the tree
        self.final_point = self.old_point
        self.points.append(self.old_point)

        for c in self.tree.children:
            self.generate_recursively(c)
        return self.tree

    def generate_recursively(self, tree):
        if self.go_deeper(tree):
            parent_tag = tree.tag
            if parent_tag[-1] == 'a':
                children = [3, 0, 1]
            elif parent_tag[-1] == 'b':
                children = [0, 1, 2]
            elif parent_tag[-1] == 'A':
                children = [1, 2, 3]
            else:
                children = [2, 3, 0]
            for c in children:
                tag = parent_tag + self.labels[c]
                element = np.dot(tree.element, self.gens[c])
                node = GroupTree(element, circle=None, level=tree.level + 1, color=tree.color, tag=tag)
                node.parent = tree
                self.generate_recursively(node)

    def go_deeper(self, tree):
        '''

        check the condition for further recursion,
        either the max_level hasn't been reached
        or the size of the discs is not small enough yet

        :param tree:
        :return:
        '''
        if self.max_level:
            if tree.level < self.max_level:
                return True
        elif self.max_sep:
            new_right_point = self.get_right_most_point(tree)
            new_left_point = self.get_left_most_point(tree)
            if np.abs(
                    self.old_point - new_right_point) > self.max_sep:  # and np.abs(new_left_point-self.final_point)>self.max_sep:
                return True
            else:
                self.old_point = new_right_point
                self.points.append(self.old_point)
        return False

    def get_leaves(self):
        '''
        returns the discs at the leaves of the tree
        :return:
        '''
        leaves = []
        self.collect_leaves_recursively(self.tree, leaves)
        return leaves

    def collect_leaves_recursively(self, tree, leaves):
        if not tree.children:
            leaves.append(tree)
        else:
            for child in tree.children:
                self.collect_leaves_recursively(child, leaves)

    def pre_order_traversal(self):
        '''
        returns all nodes of the tree in pre-order traversal
        :return:
        '''
        nodes = []
        self.pre_order_recursively(self.tree, nodes)
        return nodes

    def pre_order_recursively(self, tree, nodes):
        if tree != self.tree:
            nodes.append(tree)
        for c in tree.children:
            self.pre_order_recursively(c, nodes)

    def count_nodes(self):
        return self.count_nodes_recursively(self.tree, 0)

    def count_nodes_recursively(self, tree, node_counter):
        node_counter += 1
        for c in tree.children:
            node_counter = self.count_nodes_recursively(c, node_counter)
        return node_counter

    def collect_points(self):
        # points = []
        # for l in self.get_leaves():
        #     last_label = l.tag[-1]
        #     index = self.labels.index(last_label)
        #
        #     fp_left = self.fixed_points[2 * index]
        #     fp_right = self.fixed_points[2 * index + 1]
        #
        #     element = l.get_element()
        #     z_left = moebius_on_point(element, fp_left)
        #     points.append(z_left)
        #     z_right = moebius_on_point(element, fp_right)
        #     points.append(z_right)

        points = []
        for c in self.tree.children:
            points = self.collect_points_recursively(c, points)
        return points

    def collect_points_recursively(self, tree, points):
        last_label = tree.tag[-1]
        index = self.labels.index(last_label)

        fp_left = self.fixed_points[2 * index]
        fp_right = self.fixed_points[2 * index + 1]

        element = tree.get_element()
        z_left = moebius_on_point(element, fp_left)
        points.append(z_left)
        for c in tree.children:
            points = self.collect_points_recursively(c, points)
        z_right = moebius_on_point(element, fp_right)
        points.append(z_right)
        return points

    def get_right_most_point(self, tree):
        last_label = tree.tag[-1]
        index = self.labels.index(last_label)

        fp_right = self.fixed_points[2 * index + 1]
        z_right = moebius_on_point(tree.element, fp_right)
        return z_right

    def get_left_most_point(self, tree):
        last_label = tree.tag[-1]
        index = self.labels.index(last_label)

        fp_left = self.fixed_points[2 * index]
        z_left = moebius_on_point(tree.element, fp_left)
        return z_left


def quick_fixed_points_of(m):
    f1 = m[0][0] - m[1][1] - np.sqrt(m[0][0] ** 2 + 4 * m[0][1] * m[1][0] - 2 * m[0][0] * m[1][1] + m[1][1] ** 2)
    f1 /= (2 * m[1][0])
    f2 = m[0][0] - m[1][1] + np.sqrt(m[0][0] ** 2 + 4 * m[0][1] * m[1][0] - 2 * m[0][0] * m[1][1] + m[1][1] ** 2)
    f2 /= (2 * m[1][0])
    return [f1,f2]

def k_of(m):
    [f1,f2] = quick_fixed_points_of(m)
    k=(m[0][0] - m[1][0] * f1) / (m[0][0] - m[1][0] * f2)
    if np.abs(k)<1:
        return 1/k
    else:
        return k


def g_of(m):
    [f1,f2]=quick_fixed_points_of(m)
    k = k_of(m)
    if np.abs(k) < 1:
        f1, f2 = f2, f1
    return np.array([[1, -f1], [1, -f2]])


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


class DepthFirstSearchOriginal:
    def __init__(self, family, eps=0.1, **kwargs):
        self.lev = None
        self.old_point = None

        self.gens = family(**kwargs).get_generators()
        self.fixed_points = family(**kwargs).get_fixed_points()
        self.circles = family(**kwargs).get_circles()

        self.begin_pt_generators = self.create_begin_point_generators()
        self.end_pt_generators = self.create_end_point_generators()

        self.inv = [2, 3, 0, 1]

        self.word = None
        self.tags = None
        self.epsilon = eps

        self.setup_start()
        self.points = []  # field to store the limit points
        self.circs = []  # field to store all circles

        self.breaking_length = None

    def setup_start(self,
                    begin_tag=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                    end_tag=[1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 1, 0, 3, 2]):
        '''
        This function can be called before run() to setup the begin_tag and end_tag
        :param begin_tag: [0,0,0,1,2]
        :param end_tag: [0,3,3,0,1]
        :return:
        '''

        self.tags = []
        self.word = []
        for t in begin_tag:
            self.tags.append(t)
            if len(self.word) == 0:
                self.word.append(self.gens[t])
            else:
                self.word.append(np.dot(self.word[-1], self.gens[t]))
        self.lev = len(begin_tag) - 1
        self.old_point = moebius_on_point(self.word[-1], self.begin_pt_generators[self.tags[-1]])

        self.end_tag = end_tag

    def run(self, close_curve=False):
        while self.lev != -1 or not self.check_end():
            while not self.branch_termination() and not self.check_end():
                self.go_forward()
            while True:  # do ... while loop
                self.go_backward()
                if self.lev == -1 or self.available_turn():
                    break
            self.turn_and_go_forward()
        if close_curve:
            self.points.append(self.points[0])

    def go_forward(self):
        self.lev += 1
        new_tag = (self.tags[
                       self.lev - 1] + 1) % 4  # start with the rightmost branch: enter with a-> b,a,B; enter with b -> A,b,a; ...
        if len(self.tags) > self.lev:
            self.tags[self.lev] = new_tag
        else:
            self.tags.append(new_tag)

        new_word = np.dot(self.word[self.lev - 1], self.gens[self.tags[self.lev]])
        if len(self.word) > self.lev:
            self.word[self.lev] = new_word
        else:
            self.word.append(new_word)

    def go_backward(self):
        self.lev -= 1

    def available_turn(self):
        if (self.tags[self.lev + 1] + 3) % 4 == (self.tags[self.lev] + 2) % 4:
            self.tags.pop()
            return False
        else:
            return True

    def turn_and_go_forward(self):
        if (self.lev != -1 or self.tags[
            0] != 1) and not self.check_end():  # stop turning when the zeroth level has cycled all four values
            self.tags[self.lev + 1] = (self.tags[self.lev + 1] - 1) % 4  # go to the next left branch
            if self.lev == -1:
                self.word[0] = self.gens[self.tags[0]]
            else:
                self.word[self.lev + 1] = np.dot(self.word[self.lev], self.gens[self.tags[self.lev + 1]])
            self.lev += 1

    def check_end(self, test=None):
        '''
        if self.tags is shorter than end_tag and starts with the same sequence as end_tag,
        it is returned false until it reached a shorter version that deviates

        :param test:
        :return:
        '''
        if test is not None:
            self.tags = test
        for a, b in zip(self.tags, self.end_tag):
            if a != b:
                if self.breaking_length is None:
                    return False
                elif len(self.tags) > self.breaking_length:
                    return False

        # check the cases with equal start
        # easy case, both tags agree
        if len(self.tags) == len(self.end_tag):
            return True
        # self.tag is longer, but starts with the self.end_tag
        if len(self.tags) > len(self.end_tag):
            return True
        # self.tag is shorter than self.end_tag but they have a common start
        # breaking_length captures the length of this common start
        else:
            # self.tags equals to the beginning of self.end_tag
            if self.breaking_length is None:
                self.breaking_length = len(self.tags)
                return False
            # self.tags has become shorter than the previous agreement
            if self.breaking_length > len(self.tags):
                return True
            else:
                # the common start has grown longer
                if self.breaking_length < len(self.tags):
                    self.breaking_length = len(self.tags)
                return False

    def branch_termination_1(self):
        new_circ = moebius_on_circle(self.word[self.lev - 1], self.circles[
            self.tags[self.lev]])  # this is rather smart, the last tag is used to cycle through the relevant discs
        self.circs.append(new_circ)
        if new_circ.r < self.epsilon:
            self.points.append(moebius_on_point(self.word[self.lev - 1], self.fixed_points[self.tags[self.lev]]))
            # print("point added for tag: ", self.tags)
            return True
        else:
            return False

    def branch_termination_2(self):
        new_circ = moebius_on_circle(self.word[self.lev - 1], self.circles[
            self.tags[self.lev]])  # this is rather smart, the last tag is used to cycle through the relevant discs
        self.circs.append(new_circ)
        if new_circ.r < self.epsilon:
            self.points.append(moebius_on_point(self.word[self.lev], self.fixed_points[self.tags[self.lev]]))
            # print("point added for tag: ", self.tags)
            return True
        else:
            return False

    def branch_termination_3(self):
        """
        this branch_termination terminates, when there is not enough change between different levels of the tree
        :return:
        """
        new_point = moebius_on_point(self.word[self.lev], self.fixed_points[self.tags[self.lev]])

        if np.abs(new_point - self.old_point) < self.epsilon:
            self.points.append(new_point)
            self.old_point = new_point
            return True
        else:
            self.old_point = new_point
            return False

    def branch_termination(self):
        """
        now only images of commutator fixed points are plotted.
        They can be approached symmetrically from either side.
        This creates the most symmetric plots of the limit set

        :return: True, when the required accuracy is reached.
        """
        new_point = moebius_on_point(self.word[self.lev], self.end_pt_generators[self.tags[self.lev]])
        # print(self.tags2string(), new_point, self.old_point)
        # if self.tags == [1, 0, 3, 2]:
        #     i = 0
        #     i = i + 1
        if np.abs(new_point - self.old_point) < self.epsilon:
            self.points.append(new_point)
            self.old_point = new_point
            return True
        else:
            return False

    def fixed_point_of_commutator(self, a, b, c, d):
        m = np.dot(np.dot(np.dot(self.gens[a], self.gens[b]), self.gens[c]), self.gens[d])
        return fixed_point_of(m)

    def create_begin_point_generators(self):
        end_points = [self.fixed_point_of_commutator(1, 2, 3, 0),
                      self.fixed_point_of_commutator(2, 3, 0, 1),
                      self.fixed_point_of_commutator(3, 0, 1, 2),
                      self.fixed_point_of_commutator(0, 1, 2, 3)]
        return end_points

    def create_end_point_generators(self):
        begin_points = [self.fixed_point_of_commutator(3, 2, 1, 0),
                        self.fixed_point_of_commutator(0, 3, 2, 1),
                        self.fixed_point_of_commutator(1, 0, 3, 2),
                        self.fixed_point_of_commutator(2, 1, 0, 3)]
        return begin_points

    def tags2string(self):
        out = ''
        for t in self.tags:
            if t == 0:
                out += 'a'
            elif t == 1:
                out += 'b'
            elif t == 2:
                out += 'A'
            else:
                out += 'B'
        return out


class BreadFirstSearch:
    def __init__(self, a, b, A, B, level_max):
        self.level_max = level_max
        self.gens = [a, b, A, B]
        self.inv = [2, 3, 0, 1]
        self.a = a
        self.A = A
        self.b = b
        self.B = B

        self.group = []

        self.level = 0
        self.tags = []
        self.cols = []

        self.num = []

        for i in range(0, 4):
            self.group.append(self.gens[i])
            self.tags.append(i)

        self.num.append(0)  # index, where level 0 begins
        self.num.append(4)  # index, where level 1 begins

        self.generate_levels()

    def generate_levels(self):
        for level in range(1, self.level_max):
            i_new = self.num[level]
            for i_old in range(self.num[level - 1], self.num[level]):
                for j in range(0, 4):
                    if self.inv[self.tags[i_old]] != j:
                        self.group.append(np.dot(self.group[i_old], self.gens[j]))
                        self.tags.append(j)
                        i_new += 1
            self.num.append(i_new)

    def all_elements(self):
        return sum(self.num[i] for i in range(0, self.level_max))


if __name__ == '__main__':
    dfs = DepthFirstSearchByLevel(GrandMasRecipe, max_level=9, ta=2, tb=2)
    dfs.generate_tree()
    circles = [leave.circle for leave in dfs.get_leaves()]
    print(len(circles))
    pyplot.clf()
    pycircles = [pyplot.Circle((np.real(circle.c), np.imag(circle.c)), circle.r) for circle in circles]

    pyplot.gca().set_aspect('equal')
    pyplot.gca().set_xlim((-1.5, 1.5))
    pyplot.gca().set_ylim((-1.5, 1.5))
    [pyplot.gca().add_patch(circle) for circle in pycircles]
    pyplot.show()
