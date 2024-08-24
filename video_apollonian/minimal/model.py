import numpy as np
from anytree import NodeMixin, RenderTree
from matplotlib import pyplot


class ApollonianModel:
    def __init__(self):
        a = np.array([[1, 0], [-2j, 1]])
        A = np.array([[1, 0], [2j, 1]])
        b = np.array([[1 - 1j, 1], [1, 1 + 1j]])
        B = np.array([[1 + 1j, -1], [-1, 1 - 1j]])

        self.gens = {'a': a, 'b': b, 'A': A, 'B': B}
        self.labels = ['a', 'b', 'A', 'B']
        self.fix_points = {'al': -0.2 - 0.4j, 'ar': 0.2 - 0.4j,
                           'bl': -1, 'br': -0.2 - 0.4j,
                           'Al': 1, 'Ar': -1,
                           'Bl': 0.2 - 0.4j, 'Br': 1}

    def next_label(self, parent_label, current_label):
        if parent_label == 'I':
            if not current_label:
                return self.labels[0]
            else:
                index = self.labels.index(current_label)
                if 0 <= index < 3:
                    return self.labels[index + 1]
                else:
                    return None
        else:
            parent_index = self.labels.index(parent_label)
            next_label = self.labels[(parent_index + 1) % 4]
            previous_label = self.labels[(parent_index + 3) % 4]
            if not current_label:
                return previous_label
            elif current_label == previous_label:
                return parent_label
            elif current_label == parent_label:
                return next_label
            else:
                return None


class Node(NodeMixin):
    def __init__(self, label=None, value=None):
        super().__init__()
        self.value = value
        self.label = label


class FreeGroupTree:
    def __init__(self, max_level=2):
        self.am = ApollonianModel()
        self.root = Node(label='I', value=np.identity(2))
        old_nodes = [self.root]
        for i in range(max_level):
            new_nodes = []
            for parent in old_nodes:
                parent_label = parent.label
                current_label = self.am.next_label(parent_label, None)
                while current_label:
                    next_node = Node(current_label, parent.value @ self.am.gens[current_label])
                    next_node.parent = parent
                    new_nodes.append(next_node)
                    current_label = self.am.next_label(parent_label, current_label)
            old_nodes = new_nodes
        self.leaves = new_nodes

    def get_points(self):
        points = []
        for leave in self.leaves:
            fix_point_label = leave.label + 'l'
            z = self.am.fix_points[fix_point_label]
            m = leave.value
            p = (m[0, 0] * z + m[0, 1]) / (m[1, 0] * z + m[1, 1])
            points.append(p)
        points.append(points[0])
        return points


if __name__ == '__main__':
    am = ApollonianModel()
    fg = FreeGroupTree(12)
    ps = fg.get_points()

    pyplot.clf()
    pyplot.plot(np.real(ps), np.imag(ps))
    pyplot.gca().set_aspect('equal')
    pyplot.show()
