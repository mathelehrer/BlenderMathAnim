import numpy as np
from anytree import NodeMixin
from interface.ibpy import Vector

primes = [104729, 224737, 350377, 479909, 611953, 746773, 882377, 1020379, 1159523, 1299709]


class Element(NodeMixin):
    '''
    shortest_word is the word of an equivalent element that has been generated earlier in the tree of elements
    '''

    def __init__(self, matrix, word, shortest_word=None,group=None, **kwargs):
        self.group=group
        self.kwargs = kwargs
        self.matrix = matrix
        self.word = word
        if shortest_word:
            self.shortest_word = shortest_word
        else:
            self.shortest_word = str(self)
        self.name = str(self)

    def __mul__(self, other):
        return Element(self.matrix @ other.matrix, self.word + other.word,group=self.group,**self.kwargs)

    def apply(self, vector):

        if isinstance(vector, list):
            mode = 'list'
            v = np.matrix(vector).transpose()
        elif isinstance(vector, np.matrix):
            mode = 'matrix'
            v = vector
        elif isinstance(vector, tuple):
            mode = 'tuple'
            v = np.matrix(list(vector)).transpose()
        else:
            mode = 'vector'
            v = np.matrix(vector[:]).transpose()

        result = self.matrix * v
        if mode == 'list':
            return result.transpose().tolist()[0]
        if mode == 'matrix':
            return result
        if mode == 'tuple':
            return tuple(result.transpose().tolist()[0])
        if mode == 'vector':
            return Vector(result.transpose().tolist()[0])

    def __hash__(self):
        hash_value = 0
        count = 0
        rows, cols = self.matrix.shape
        for row in range(rows):
            for col in range(cols):
                hash_value += int(self.matrix[row, col] * 1000+0.5) * primes[count]
                count += 1
        return hash_value

    def compare(self, other):
        return hash(self) - hash(other)

    def __str__(self):
        if len(self.word) == 0:
            if self.group:
                return self.group.unit_string
            else:
                return 'e'
        else:
            return self.word
