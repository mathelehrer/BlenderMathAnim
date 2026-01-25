from __future__ import annotations

import typing
from itertools import combinations

import numpy as np
from anytree import Node, RenderTree
import networkx as nx
import matplotlib.pyplot as plt
from sympy import factorial, subsets
import re

### Warning
# this only works for linear Dynkin diagrams so far
import doctest

from sympy.multipledispatch.conflict import edge

DEBUG = True
logging = []
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']


class Diagram:
    def __init__(self, diagram_string):
        # convert diagram_string into graph
        self.diagram_string = diagram_string
        # list of letters to match the position of the branches in the notation
        self.letters = letters

        self.graph = nx.Graph()
        self.create_graph()

    @classmethod
    def from_graph(cls,graph:nx.Graph)->Diagram:
        """
        create a diagram from a graph
        this is useful to convert sub-graphs into diagrams
        TODO: This needs to be improved
        """
        diagram_string = ""
        # create diagram string from graph
        nodes = list(graph.nodes)
        n = max([node[1] for node in nodes])
        for i in range(n+1):
            diagram_string+=". "
        diagram_string=diagram_string[:-1]

        # deal with branches
        connection_count = {}
        for edge in nx.edges(graph):
            connection_count[edge[0][1]] = connection_count.get(edge[0][1], 0) + 1
            connection_count[edge[1][1]] = connection_count.get(edge[1][1], 0) + 1

        weights = nx.get_edge_attributes(graph, 'weight')

        new_nodes = nodes
        new_weights = weights

        # relocated branch node, when the diagram becomes linear (put the nodes with the lowest connectivity at the start and the end
        if all(val<=2 for key,val in connection_count.items()):
            positions = list(connection_count.keys())
            positions.sort() # make sure that the branch point is at the end
            if len(positions)>0:
                if connection_count[positions[0]]==2 and connection_count[positions[-1]]==1:
                    connection_count[positions[0]-1]=connection_count[positions[-1]]
                    connection_count.pop(positions[-1])

                    new_nodes = []
                    for node in nodes:
                        if node[1]==positions[-1]:
                            new_nodes.append((node[0],positions[0]-1))
                        else:
                            new_nodes.append(node)

                    new_edges = []
                    for edge in graph.edges:
                        if edge[0][1]==positions[-1]:
                            new_edge=((edge[0][0],positions[0]-1),edge[1])
                        elif edge[1][1]==positions[-1]:
                            new_edge=(edge[0],(edge[1][0],positions[0]-1))
                        else:
                            new_edge = edge
                        if new_edge[0][1]>new_edge[1][1]:
                            new_edge = (new_edge[1],new_edge[0])
                        new_edges.append(new_edge)

                    new_weights={}
                    for edge,weight in weights.items():
                        if edge in new_edges:
                            new_weights[edge]=weight
                        else:
                            if edge[0][1]==positions[-1]:
                                new_weights[((edge[0][0],positions[0]-1),edge[1])]=weight
                            else:
                                new_weights[(((edge[1][0],positions[0]-1)),edge[0])]=weight

        for node in new_nodes:
            label, pos = node
            diagram_string = diagram_string[:2*pos] + label + diagram_string[2*pos+1:]

        for edge,weight in new_weights.items():
            if connection_count[edge[0][1]]<3 and connection_count[edge[1][1]]<3:
                diagram_string = diagram_string[:2*edge[0][1]+1]+str(weight)+diagram_string[2*edge[0][1]+2:]
            else:
                if connection_count[edge[0][1]]>2:
                    branch_from =edge[0][1]
                    branch_to = edge[1][1]
                else:
                    branch_from = edge[1][1]
                    branch_to = edge[0][1]
                l = letters[branch_from]

                diagram_string = diagram_string[:2*branch_to+2]+"*"+l+str(weight)+diagram_string[2*branch_to+2:]

        return cls(diagram_string)


    def create_graph(self):
        """
        convert Coxeter diagram string representation into a graph
        """

        # detect branches

        diagram_string = self.diagram_string.replace(" *", "*")
        diagram_string = diagram_string.replace(" ", "2")
        parts = diagram_string.split("*")

        node_count = 0

        for part in parts:
            follow_digits = False
            weight = 0
            branch_index = None
            branch_height = None
            branch_pos = None
            for l in part:
                if l.isdigit():
                    if not follow_digits:
                        weight = int(l)
                        follow_digits = True
                    else:
                        weight = weight * 10 + int(l)
                else:
                    follow_digits = False
                    # missing node
                    if l == '.':
                        node_count += 1
                    elif l == 'o' or l == 'x':
                        # create node and position
                        if branch_pos is not None:
                            pos = (branch_pos, branch_height)
                            branch_height += 1
                        else:
                            pos = (node_count, 0)
                        self.graph.add_node((l, node_count), pos=pos)
                        # add edges
                        if weight > 2:
                            if branch_index is not None:
                                # search for branching node
                                for node in self.graph.nodes:
                                    if node[1] == branch_index:
                                        self.graph.add_edge((l, node_count), node, weight=weight)
                                        break
                                branch_index = None
                            else:
                                # connect to previous node
                                self.graph.add_edge(list(self.graph.nodes.keys())[-2], (l, node_count), weight=weight)
                        node_count += 1
                    else:
                        branch_index = self.letters.index(l)
                        branch_pos = branch_index
                        branch_height = 1

    def show_graph(self):
        """
        display the graph
        >>> d=Diagram("x3o . x3x3x3x *c3x")
        >>> d.show_graph()
        """
        plt.figure(figsize=(10, 2))

        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, font_weight='bold', with_labels=True)
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)

        # for cc in nx.connected_components(G):
        #     print([labels[c] for c in cc])

        plt.show()

    def get_dim(self) -> int:
        return len(self.graph.nodes)

    def get_diagrams_from_connected_components(self) -> [str]:
        """
        create the diagram for each connected component of the graph
        >>> d=Diagram("x3x3x3x3x *c3x")
        >>> d.get_diagrams_from_connected_components()
        ['x3x3x3x3x *c3x']

        >>> d=Diagram("x3x5x")
        >>> d.get_diagrams_from_connected_components()
        ['x3x5x']

        >>> d=Diagram("x . o5x")
        >>> d.get_diagrams_from_connected_components()
        ['x', 'o5x']

        >>> d=Diagram("x3x . x3x *c3x")
        >>> d.get_diagrams_from_connected_components()
        ['x3x', 'x3x', 'x']

        """
        comp_strings = []
        for comp in nx.connected_components(self.graph):
            # sort components
            comp = list(comp)
            comp.sort(key=lambda x: x[1])
            id2label = {}
            id2node = {}
            node2pos = {}
            pos = 0  # the position of the connected components might differ from the original larger diagram
            for label, node_id in comp:
                id2label[node_id] = label
                id2node[node_id] = self.graph.nodes[(label, node_id)]
                node2pos[(label, node_id)] = pos
                pos += 1

            edges = nx.edges(self.graph)

            diagram = ""
            for val in id2label.values():
                diagram += val + " "

            diagram = diagram[:-1]

            branching = ""
            already_connected = []
            for edge in edges:
                if edge[0] in comp and edge[1] in comp:
                    a = edge[0][1]
                    b = edge[1][1]
                    pos = node2pos[edge[0]]
                    connection = a
                    if a > b:
                        connection = b
                        pos = node2pos[edge[1]]
                    w = self.graph.get_edge_data(*edge)["weight"]
                    if connection in already_connected:
                        alt_pos = node2pos[edge[1]]
                        branching = "*" + self.letters[connection] + str(w) + diagram[2 * alt_pos]
                        diagram = diagram[:2 * alt_pos]
                    else:
                        diagram = diagram[:2 * pos + 1] + str(w) + diagram[2 * pos + 2:]
                    already_connected.append(connection)
            diagram = diagram + branching
            comp_strings.append(diagram)

        return comp_strings

    def get_complement(self) -> Diagram:
        """
        return a diagram that has the same structure as the unringed nodes of
        the given diagram
        >>> d = Diagram("x3x3o5o")
        >>> d.get_complement().diagram_string
        '. . x5x'
        >>> d=Diagram("o3o3o3o3x3x *c3o")
        >>> d.get_complement().diagram_string
        'x3x3x3x . . *c3x'
        """

        complement_str = self.diagram_string.replace("x", ".").replace("o", "x")
        # remove connections to dots
        dot_position = 0
        while dot_position > -1:
            try:
                dot_position = complement_str.index(".", dot_position)
            except ValueError:
                break
            if dot_position == 0:
                complement_str = ". " + complement_str[2:]
            elif dot_position == len(complement_str) - 1:
                complement_str = complement_str[:-2] + " ."
            else:
                complement_str = complement_str[:dot_position - 1] + " . " + complement_str[dot_position + 2:]
            dot_position += 1
        return Diagram(complement_str)

    def is_connected(self) -> bool:
        """
        returns true, when a graph is connected, i.e. there is only one connected component
        and false otherwise
        >>> d = Diagram("x3x3o5o")
        >>> d.is_connected()
        True
        >>> d=Diagram("x3o3xo . . x3x")
        >>> d.is_connected()
        False

        """
        comps = nx.connected_components(self.graph)
        count = 0
        for comp in comps:
            count += 1
        return count == 1

    def is_linear_graph(self):
        """
        >>> d = Diagram("x3x3o5o")
        >>> d.is_linear_graph()
        True
        >>> d=Diagram("x3o3xo . . x3x")
        >>> d.is_linear_graph()
        False

        """
        connected = self.is_connected()
        if connected:
            # check for branchings, ie. more than two connections at a single node
            connection_counts = {}
            for edge in nx.edges(self.graph):
                connection_counts[edge[0]] = connection_counts.get(edge[0], 0) + 1
                connection_counts[edge[1]] = connection_counts.get(edge[1], 0) + 1
            max_connection = max(connection_counts.values())
            return max_connection <= 2
        else:
            return False

    def __get_vertex_count_for_graph_component(self, comp):
        """
        This function evaluates the vertex count for a given connected component of a graph.

        """

        nodes = list(comp)
        nodes.sort(key=lambda x: x[1])
        if len(nodes) == 1:
            if nodes[0][0] == 'x':
                return 2
            if nodes[0][0] == 'o':
                return 0

        weights = nx.get_edge_attributes(self.graph, 'weight')
        weight_list = list(weights.values())
        if len(nodes) == 2:
            w = weights[(nodes[0], nodes[1])]
            if nodes[0][0] == 'x' and nodes[1][0] == 'x':
                return 2 * w
            elif (nodes[0][0] == 'x' and nodes[1][0] == 'o') or (nodes[1][0] == 'x' and nodes[0][0] == 'o'):
                if w > 2:
                    return w
                else:
                    return 0
            else:
                return 0

        # compute complement to divide out inactive nodes
        dim_reduction = 1
        if all(node[0] == 'o' for node in nodes):
            return 0

        if not all(node[0] == 'x' for node in nodes):
            dim_reduction = self.get_complement().get_vertex_count()

        connection_counts = {}
        for edge in nx.edges(self.graph):
            connection_counts[edge[0]] = connection_counts.get(edge[0], 0) + 1
            connection_counts[edge[1]] = connection_counts.get(edge[1], 0) + 1

        counts = list(connection_counts.values())
        n = len(nodes)
        if all(c <= 2 for c in counts):
            # deal with A series
            threes = weight_list.count(3)
            if all(w == 3 for w in weights.values()):
                return factorial(n + 1) // dim_reduction

            # deal with B series
            fours = weight_list.count(4)
            if fours == 1 and threes == len(weights) - 1 and (
                    weights[(nodes[0], nodes[1])] == 4 or weights[(nodes[-2], nodes[-1])] == 4):
                return factorial(n) * 2 ** n // dim_reduction

            # deal with F4
            elif len(nodes) == 4 and weights[(nodes[1], nodes[2])] == 4:
                return 1152 // dim_reduction

            # deal with H series
            fives = weight_list.count(5)
            if fives == 1 and threes == len(weights) - 1 and (
                    weights[(nodes[0], nodes[1])] == 5 or weights[(nodes[-2], nodes[-1])] == 5):
                if len(nodes) == 3:
                    return 120 // dim_reduction
                elif len(nodes) == 4:
                    return 14400 // dim_reduction
        else:
            if all(w == 3 for w in weights.values()):
                three = [c for c in counts if c == 3]  # just one branch
                if len(three) == 1:
                    # deal with D series
                    if counts[1] == 3 or counts[-3] == 3:
                        return 2 ** (n - 1) * factorial(n) // dim_reduction

                    # deal with E6, E7, E8
                    # Returns dimension for E6, E7, or E8 Dynkin diagram
                    if len(nodes) == 6:
                        if counts[2] == 3:
                            return 51840 // dim_reduction
                    elif len(nodes) == 7:
                        if counts[2] == 3 or counts[3] == 3:
                            return 2903040 // dim_reduction
                    elif len(nodes) == 8:
                        if counts[2] == 3 or counts[4] == 3:
                            return 696729600 // dim_reduction
        raise "not implemented yet"

    def get_vertex_count(self):
        """
        This function is the core of the class. It returns the vertex count for a given Coxeter-Dynkin diagram.
        The evaluation includes all compoments and when the algorithm fails it gives a warning

        # start with the simple cases
        >>> d=Diagram("x o")
        >>> d.is_connected()
        False
        >>> d=Diagram("o3o")
        >>> d.get_vertex_count()
        0
        >>> d.get_vertex_count()
        0
        >>> d = Diagram("x3x")
        >>> d.get_vertex_count()
        6
        >>> d=Diagram("x3o")
        >>> d.get_vertex_count()
        3
        >>> d=Diagram("x3x3x3x")
        >>> d.get_vertex_count()
        120
        >>> d=Diagram("x4x3x3x")
        >>> d.get_vertex_count()
        384
        >>> d=Diagram("x3x3x4x")
        >>> d.get_vertex_count()
        384
        >>> d=Diagram("x3x4x3x")
        >>> d.get_vertex_count()
        1152
        >>> d=Diagram("x3x3x5x")
        >>> d.get_vertex_count()
        14400
        >>> d=Diagram("x3x3o5o")
        >>> d.get_vertex_count()
        1440
        >>> d=Diagram("o3x4x5o")
        >>> d.get_vertex_count()
        288
        >>> d=Diagram("x3x3x3x3x *c3x")
        >>> d.get_vertex_count()
        51840
        >>> d=Diagram("o3o3o3x3x *c3x")
        >>> d.get_vertex_count()
        2160
        >>> d=Diagram("x3o3o3o3o *c3o")
        >>> d.get_vertex_count()
        27
        >>> d=Diagram("o3o3o3o3x *c3x")
        >>> d.get_vertex_count()
        432
        >>> d=Diagram("x3x3x3x3x3x *c3x")
        >>> d.get_vertex_count()
        2903040
        >>> d=Diagram("x3x3x3x3x3x *d3x")
        >>> d.get_vertex_count()
        2903040
        >>> d=Diagram("x3x3x3x3x3x3x *c3x")
        >>> d.get_vertex_count()
        696729600
        >>> d=Diagram("x3x3x3x3x3x3x *e3x")
        >>> d.get_vertex_count()
        696729600

        # check disconnected graphs
        >>> d=Diagram("x3x . x")
        >>> d.get_vertex_count()
        12
        >>> d=Diagram("o3x . x")
        >>> d.get_vertex_count()
        6

        # check D series
        >>> d = Diagram("x3x3x *b3x")
        >>> d.get_vertex_count()
        192
        """

        comps = nx.connected_components(self.graph)
        count = 1
        for comp in comps:
            count *= self.__get_vertex_count_for_graph_component(comp)
        return count

    def get_subdiagrams(self):
        """
        Generate all possible sub diagrams of a given diagram by omitting nodes and their connections
        >>> d=Diagram("x3o3o3o3o *c3o")
        >>> rows,subs,dimensions = d.get_subdiagrams()
        >>> [ddd.diagram_string for dd in subs.values() for ddd in dd],dimensions
        (['. . . . . .', 'x . . . . *c .', 'x3o . . . *c .', 'x3o3o . . *c .', 'x3o3o3o . *c .', 'x3o3o . . *c3o', 'x3o3o3o3o *c .', 'x3o3o3o . *c3o'], [1, 1, 1, 1, 2, 2])
        >>> d.get_vertex_count()
        27

        >>> d=Diagram("x3x3x")
        >>> rows, subs,dimensions = d.get_subdiagrams()
        >>> [ddd.diagram_string for dd in subs.values() for ddd in dd],dimensions
        (['. . .', 'x . .', '. x .', '. . x', 'x3x .', 'x . x', '. x3x'], [1, 3, 3])

        """
        dimensions = []

        sub_diagrams = {}
        rows = []
        n = len(self.graph.nodes)
        # for i in range(n):
        #     if i > 0:
        #         first_row += " ."
        #     else:
        #         first_row += "."
        # rows.append(first_row)
        # sub_diagrams[0] = [Diagram(first_row)]

        for d in range(0, n):
            count = 0
            sub_diagrams[d] = []
            combs = list(combinations(list(range(n)), d))
            for comb in combs:
                n = 0
                row = ""
                for b in self.diagram_string:
                    if b == 'x' or b == 'o':
                        if n not in comb:
                            row += "."
                        else:
                            row += b
                        n = n + 1
                    else:
                        row += b

                row = re.sub(r"\d{1}\.\d{1}", " . ", row)
                row = re.sub(r"\d{1}\.", " .", row)
                row = re.sub(r"\.\d{1}", ". ", row)

                # check that the vertex count is larger than 0
                row_diagram = Diagram(row)
                vertex_count = row_diagram.get_vertex_count()
                if vertex_count > 0:
                    count += 1
                    rows.append(row)
                    sub_diagrams[d].append(row_diagram)
            dimensions.append(count)
        return rows, sub_diagrams, dimensions

    def contains(self,diagram):
        return nx.is_isomorphic(self.graph.subgraph(diagram.graph.nodes),diagram.graph)

    def get_maximal_diagram_string(self):
        return self.diagram_string.replace("o","x")

    def get_maximal_orthogonal_contraction(self,diagram)->int:
        """
        computation of the factor, by which a diagonal element of the incidence matrix is contracted in comparsion to the full incidence matrix
        We need to find the vertex count of the subgraph that only contains "o" and is not connected to the 'x' of the diagram

        especially difficult case: a branching point becomes linear
        >>> d = Diagram("x3o3o3o3o *c3o")
        >>> sub_rows,subs,sub_dimensions = d.get_subdiagrams()
        >>> d.get_maximal_orthogonal_contraction(subs[4][1])
        6
        >>> d = Diagram("x3x3o3o")
        >>> sub_rows,subs,sub_dimensions = d.get_subdiagrams()
        >>> d.get_maximal_orthogonal_contraction(subs[1][0])
        6
        >>> d = Diagram("x3x3o3o")
        >>> sub_rows,subs,sub_dimensions = d.get_subdiagrams()
        >>> d.get_maximal_orthogonal_contraction(subs[1][1])
        2
        >>> d = Diagram("o5x")
        >>> sub_rows,subs,sub_dimensions = d.get_subdiagrams()
        >>> d.get_maximal_orthogonal_contraction(subs[1][0])
        1
        """

        nodes = list(self.graph.nodes)
        sub_nodes = list(diagram.graph.nodes)

        complement_nodes = [node for node in nodes if node[0]=='o']
        edges = list(self.graph.edges)

        for edge in edges:
            if edge[0] in sub_nodes and edge[1] in complement_nodes:
                complement_nodes.remove(edge[1])
            if edge[0] in complement_nodes and edge[1] in sub_nodes:
                complement_nodes.remove(edge[0])

        if len(complement_nodes)==0:
            return 1
        sub_graph = self.graph.subgraph(complement_nodes)
        sub_diagram = Diagram.from_graph(sub_graph)
        return sub_diagram.get_complement().get_vertex_count()


    def __str__(self):
        return self.diagram_string

    def __repr__(self):
        return "Diagram("+self.diagram_string+")"

class IncidenceMatrix:
    def __init__(self, diagram_string):
        self.diagram = Diagram(diagram_string)
        self.max_diagram = Diagram(self.diagram.get_maximal_diagram_string())

        self.max_matrix,self.max_dimensions = self.compute_largest_incidence_matrix()
        self.max_rows,self_max_sub_diagrams,self.max_dimensions = self.max_diagram.get_subdiagrams()

        if not 'o' in diagram_string:
            self.rows=self.max_rows
            self.matrix = self.max_matrix
            self.dimensions = self.max_dimensions
        else:
            self.rows, self.sub_diagrams, self.dimensions = self.diagram.get_subdiagrams()
            self.matrix,self.dimension = self.compute_incidence_matrix()

    def print_largest_table(self):
        """
       Prints formatted incidence matrix of the largest polytope, where each unringed node is replaced by a ringed node, to console
       :param matrix: the numerical data of the matrix
       :param rows: the strings of the first column
       :param dimensions: the number of different edges, faces, etc.
       """

        array = np.array(self.max_matrix)
        # create column matrix to determine the custom width of each column
        col_matrix = array.transpose()
        widths = [max([len(row) for row in self.max_rows])] + [max([len(str(entry)) for entry in column]) for column in
                                                           col_matrix]

        top = "+" + "-" * widths[0] + "+"
        sep = "|" + "-" * widths[0] + "+"
        count = 0
        for d in self.max_dimensions:  # dimensions capture the size for each section of the table (that belongs to one particular part, eg vertices, edges, faces, etc.)
            for i in range(count, count + d):
                top += "-" * widths[i + 1] + " "
                sep += "-" * widths[i + 1] + " "
            top = top[:-1]
            top += "+"
            sep = sep[:-1]
            sep += "+"
            count += d
        sep = sep[:-1] + "|"

        table = top + ("\n")
        dim = self.max_dimensions[0]
        dim_sel = 1
        for r in range(len(self.max_matrix)):
            row_string = "|"
            row_string += self.max_rows[r] + "|"
            count = 0
            for d in self.max_dimensions:  # dimensions capture the size for each section of the table (that belongs to one particular part, eg vertices, edges, faces, etc.)
                for c in range(count, count + d):
                    entry = self.max_matrix[r][c]
                    if entry == -1:
                        entry = "*"
                    else:
                        entry = str(entry)
                    width = widths[c + 1]
                    while len(entry) < width:
                        entry = " " + entry  # padding for right alignment
                    row_string += entry + " "
                row_string = row_string[:-1] + "|"
                count += d
            table += row_string + "\n"
            if r == dim - 1:
                if dim_sel < len(self.max_dimensions):
                    table += sep + "\n"
                    dim += self.max_dimensions[dim_sel]
                    dim_sel += 1
                else:
                    table += top

        print(table)

    def print_table(self):
        """
        Prints formatted incidence matrix to console
        :param matrix: the numerical data of the matrix
        :param rows: the strings of the first column
        :param dimensions: the number of different edges, faces, etc.
        """

        array = np.array(self.matrix)
        # create column matrix to determine the custom width of each column
        col_matrix = array.transpose()
        widths = [max([len(row) for row in self.rows])] + [max([len(str(entry)) for entry in column]) for column in
                                                      col_matrix]

        top = "+" + "-" * widths[0] + "+"
        sep = "|" + "-" * widths[0] + "+"
        count = 0
        for d in self.dimensions: # dimensions capture the size for each section of the table (that belongs to one particular part, eg vertices, edges, faces, etc.)
            for i in range(count, count + d):
                top += "-" * widths[i + 1] + " "
                sep += "-" * widths[i + 1] + " "
            top = top[:-1]
            top += "+"
            sep = sep[:-1]
            sep += "+"
            count += d
        sep = sep[:-1] + "|"

        table = top + ("\n")
        dim = self.dimensions[0]
        dim_sel = 1
        for r in range(len(self.matrix)):
            row_string = "|"
            row_string += self.rows[r] + "|"
            count = 0
            for d in self.dimensions:  # dimensions capture the size for each section of the table (that belongs to one particular part, eg vertices, edges, faces, etc.)
                for c in range(count, count + d):
                    entry = self.matrix[r][c]
                    if entry==-1:
                        entry = "*"
                    else:
                        entry = str(entry)
                    width = widths[c+1]
                    while len(entry)<width:
                        entry=" "+entry # padding for right alignment
                    row_string+=entry+" "
                row_string=row_string[:-1]+"|"
                count+=d
            table += row_string+ "\n"
            if r == dim - 1:
                if dim_sel < len(self.dimensions):
                    table += sep + "\n"
                    dim += self.dimensions[dim_sel]
                    dim_sel += 1
                else:
                    table += top

        print(table)

    def print_latex_table(self):
        """
        Generates and prints a LaTeX table based on the provided matrix, row labels, dimensions, and diagram label.

        """
        array = np.array(self.matrix)
        col_matrix = array.transpose()
        widths = [max([len(row) for row in self.rows])] + [max([len(str(entry)) for entry in column]) for column in
                                                      col_matrix]
        lines = []

        lines.append(r"\fbox{\tt " + self.diagram.diagram_string + r"}\\")
        format = "|c|"
        for d in self.dimensions:
            for c in range(d):
                format += "r "
            format += "|"

        lines.append(r"\begin{tabular}{" + format + "}")

        sep_pos = [sum(self.dimensions[:i]) for i in range(len(self.dimensions))]
        for i, row in enumerate(self.matrix):
            if i in sep_pos:
                lines.append(r"\hline")
            row_string = r"{\tt " + self.rows[i] + "}&"
            for j, (width, entry) in enumerate(zip(widths[1:], row)):
                if entry == -1:
                    entry = "*"
                entry_string = str(entry)
                row_string += "$" + entry_string + "$" + "&"
            lines.append(row_string[:-1] + r"\\")

        lines.append(r"\hline")
        lines.append("\end{tabular}")
        for line in lines:
            print(line)

    def compute_largest_incidence_matrix(self):
        """
        compute the incidence matrix under the assumption that all nodes are active (ringed)
        >>> im = IncidenceMatrix("x3x")
        >>> im.print_table()
        +---+-+-- --+
        |. .|6| 1  1|
        |---+-+-- --|
        |x .|2| 3  *|
        |. x|2| *  3|
        +---+-+-- --+

        >>> im = IncidenceMatrix("x3x3x5x")
        >>> im.print_table()
        +-------+-----+---- ---- ---- ----+---- ---- ---- ---- ---- ----+--- ---- --- ---+
        |. . . .|14400|   1    1    1    1|   1    1    1    1    1    1|  1    1   1   1|
        |-------+-----+---- ---- ---- ----+---- ---- ---- ---- ---- ----+--- ---- --- ---|
        |x . . .|    2|7200    *    *    *|   1    1    1    0    0    0|  1    1   1   0|
        |. x . .|    2|   * 7200    *    *|   1    0    0    1    1    0|  1    1   0   1|
        |. . x .|    2|   *    * 7200    *|   0    1    0    1    0    1|  1    0   1   1|
        |. . . x|    2|   *    *    * 7200|   0    0    1    0    1    1|  0    1   1   1|
        |-------+-----+---- ---- ---- ----+---- ---- ---- ---- ---- ----+--- ---- --- ---|
        |x3x . .|    6|   3    3    0    0|2400    *    *    *    *    *|  1    1   0   0|
        |x . x .|    4|   2    0    2    0|   * 3600    *    *    *    *|  1    0   1   0|
        |x . . x|    4|   2    0    0    2|   *    * 3600    *    *    *|  0    1   1   0|
        |. x3x .|    6|   0    3    3    0|   *    *    * 2400    *    *|  1    0   0   1|
        |. x . x|    4|   0    2    0    2|   *    *    *    * 3600    *|  0    1   0   1|
        |. . x5x|   10|   0    0    5    5|   *    *    *    *    * 1440|  0    0   1   1|
        |-------+-----+---- ---- ---- ----+---- ---- ---- ---- ---- ----+--- ---- --- ---|
        |x3x3x .|   24|  12   12   12    0|   4    6    0    4    0    0|600    *   *   *|
        |x3x . x|   12|   6    6    0    6|   2    0    3    0    3    0|  * 1200   *   *|
        |x . x5x|   20|  10    0   10   10|   0    5    5    0    0    2|  *    * 720   *|
        |. x3x5x|  120|   0   60   60   60|   0    0    0   20   30   12|  *    *   * 120|
        +-------+-----+---- ---- ---- ----+---- ---- ---- ---- ---- ----+--- ---- --- ---+

        >>> im = IncidenceMatrix("x3x5x")
        >>> im.print_table()
        +-----+---+-- -- --+-- -- --+
        |. . .|120| 1  1  1| 1  1  1|
        |-----+---+-- -- --+-- -- --|
        |x . .|  2|60  *  *| 1  1  0|
        |. x .|  2| * 60  *| 1  0  1|
        |. . x|  2| *  * 60| 0  1  1|
        |-----+---+-- -- --+-- -- --|
        |x3x .|  6| 3  3  0|20  *  *|
        |x . x|  4| 2  0  2| * 30  *|
        |. x5x| 10| 0  5  5| *  * 12|
        +-----+---+-- -- --+-- -- --+

        >>> im = IncidenceMatrix("x3x3x3x3x *c3x")
        >>> im.print_table()
        +--------------+-----+----- ----- ----- ----- ----- -----+---- ----- ----- ----- ----- ---- ----- ----- ----- ---- ----- ---- ---- ----- -----+---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----+--- ---- --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ---+-- -- --- --- --- --+
        |. . . . . *c .|51840|    1     1     1     1     1     1|   1     1     1     1     1    1     1     1     1    1     1    1    1     1     1|   1    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1|  1    1   1    1    1    1    1    1    1    1   1   1    1    1   1| 1  1   1   1   1  1|
        |--------------+-----+----- ----- ----- ----- ----- -----+---- ----- ----- ----- ----- ---- ----- ----- ----- ---- ----- ---- ---- ----- -----+---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----+--- ---- --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ---+-- -- --- --- --- --|
        |x . . . . *c .|    2|25920     *     *     *     *     *|   1     1     1     1     1    0     0     0     0    0     0    0    0     0     0|   1    1    1    1    1    1    1    1    1    1    0    0    0    0    0    0    0    0    0    0|  1    1   1    1    1    1    1    1    1    1   0   0    0    0   0| 1  1   1   1   1  0|
        |. x . . . *c .|    2|    * 25920     *     *     *     *|   1     0     0     0     0    1     1     1     1    0     0    0    0     0     0|   1    1    1    1    0    0    0    0    0    0    1    1    1    1    1    1    0    0    0    0|  1    1   1    1    1    1    0    0    0    0   1   1    1    1   0| 1  1   1   1   0  1|
        |. . x . . *c .|    2|    *     * 25920     *     *     *|   0     1     0     0     0    1     0     0     0    1     1    1    0     0     0|   1    0    0    0    1    1    1    0    0    0    1    1    1    0    0    0    1    1    1    0|  1    1   1    0    0    0    1    1    1    0   1   1    1    0   1| 1  1   1   0   1  1|
        |. . . x . *c .|    2|    *     *     * 25920     *     *|   0     0     1     0     0    0     1     0     0    1     0    0    1     1     0|   0    1    0    0    1    0    0    1    1    0    1    0    0    1    1    0    1    1    0    1|  1    0   0    1    1    0    1    1    0    1   1   1    0    1   1| 1  1   0   1   1  1|
        |. . . . x *c .|    2|    *     *     *     * 25920     *|   0     0     0     1     0    0     0     1     0    0     1    0    1     0     1|   0    0    1    0    0    1    0    1    0    1    0    1    0    1    0    1    1    0    1    1|  0    1   0    1    0    1    1    0    1    1   1   0    1    1   1| 1  0   1   1   1  1|
        |. . . . . *c3x|    2|    *     *     *     *     * 25920|   0     0     0     0     1    0     0     0     1    0     0    1    0     1     1|   0    0    0    1    0    0    1    0    1    1    0    0    1    0    1    1    0    1    1    1|  0    0   1    0    1    1    0    1    1    1   0   1    1    1   1| 0  1   1   1   1  1|
        |--------------+-----+----- ----- ----- ----- ----- -----+---- ----- ----- ----- ----- ---- ----- ----- ----- ---- ----- ---- ---- ----- -----+---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----+--- ---- --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ---+-- -- --- --- --- --|
        |x3x . . . *c .|    6|    3     3     0     0     0     0|8640     *     *     *     *    *     *     *     *    *     *    *    *     *     *|   1    1    1    1    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0|  1    1   1    1    1    1    0    0    0    0   0   0    0    0   0| 1  1   1   1   0  0|
        |x . x . . *c .|    4|    2     0     2     0     0     0|   * 12960     *     *     *    *     *     *     *    *     *    *    *     *     *|   1    0    0    0    1    1    1    0    0    0    0    0    0    0    0    0    0    0    0    0|  1    1   1    0    0    0    1    1    1    0   0   0    0    0   0| 1  1   1   0   1  0|
        |x . . x . *c .|    4|    2     0     0     2     0     0|   *     * 12960     *     *    *     *     *     *    *     *    *    *     *     *|   0    1    0    0    1    0    0    1    1    0    0    0    0    0    0    0    0    0    0    0|  1    0   0    1    1    0    1    1    0    1   0   0    0    0   0| 1  1   0   1   1  0|
        |x . . . x *c .|    4|    2     0     0     0     2     0|   *     *     * 12960     *    *     *     *     *    *     *    *    *     *     *|   0    0    1    0    0    1    0    1    0    1    0    0    0    0    0    0    0    0    0    0|  0    1   0    1    0    1    1    0    1    1   0   0    0    0   0| 1  0   1   1   1  0|
        |x . . . . *c3x|    4|    2     0     0     0     0     2|   *     *     *     * 12960    *     *     *     *    *     *    *    *     *     *|   0    0    0    1    0    0    1    0    1    1    0    0    0    0    0    0    0    0    0    0|  0    0   1    0    1    1    0    1    1    1   0   0    0    0   0| 0  1   1   1   1  0|
        |. x3x . . *c .|    6|    0     3     3     0     0     0|   *     *     *     *     * 8640     *     *     *    *     *    *    *     *     *|   1    0    0    0    0    0    0    0    0    0    1    1    1    0    0    0    0    0    0    0|  1    1   1    0    0    0    0    0    0    0   1   1    1    0   0| 1  1   1   0   0  1|
        |. x . x . *c .|    4|    0     2     0     2     0     0|   *     *     *     *     *    * 12960     *     *    *     *    *    *     *     *|   0    1    0    0    0    0    0    0    0    0    1    0    0    1    1    0    0    0    0    0|  1    0   0    1    1    0    0    0    0    0   1   1    0    1   0| 1  1   0   1   0  1|
        |. x . . x *c .|    4|    0     2     0     0     2     0|   *     *     *     *     *    *     * 12960     *    *     *    *    *     *     *|   0    0    1    0    0    0    0    0    0    0    0    1    0    1    0    1    0    0    0    0|  0    1   0    1    0    1    0    0    0    0   1   0    1    1   0| 1  0   1   1   0  1|
        |. x . . . *c3x|    4|    0     2     0     0     0     2|   *     *     *     *     *    *     *     * 12960    *     *    *    *     *     *|   0    0    0    1    0    0    0    0    0    0    0    0    1    0    1    1    0    0    0    0|  0    0   1    0    1    1    0    0    0    0   0   1    1    1   0| 0  1   1   1   0  1|
        |. . x3x . *c .|    6|    0     0     3     3     0     0|   *     *     *     *     *    *     *     *     * 8640     *    *    *     *     *|   0    0    0    0    1    0    0    0    0    0    1    0    0    0    0    0    1    1    0    0|  1    0   0    0    0    0    1    1    0    0   1   1    0    0   1| 1  1   0   0   1  1|
        |. . x . x *c .|    4|    0     0     2     0     2     0|   *     *     *     *     *    *     *     *     *    * 12960    *    *     *     *|   0    0    0    0    0    1    0    0    0    0    0    1    0    0    0    0    1    0    1    0|  0    1   0    0    0    0    1    0    1    0   1   0    1    0   1| 1  0   1   0   1  1|
        |. . x . . *c3x|    6|    0     0     3     0     0     3|   *     *     *     *     *    *     *     *     *    *     * 8640    *     *     *|   0    0    0    0    0    0    1    0    0    0    0    0    1    0    0    0    0    1    1    0|  0    0   1    0    0    0    0    1    1    0   0   1    1    0   1| 0  1   1   0   1  1|
        |. . . x3x *c .|    6|    0     0     0     3     3     0|   *     *     *     *     *    *     *     *     *    *     *    * 8640     *     *|   0    0    0    0    0    0    0    1    0    0    0    0    0    1    0    0    1    0    0    1|  0    0   0    1    0    0    1    0    0    1   1   0    0    1   1| 1  0   0   1   1  1|
        |. . . x . *c3x|    4|    0     0     0     2     0     2|   *     *     *     *     *    *     *     *     *    *     *    *    * 12960     *|   0    0    0    0    0    0    0    0    1    0    0    0    0    0    1    0    0    1    0    1|  0    0   0    0    1    0    0    1    0    1   0   1    0    1   1| 0  1   0   1   1  1|
        |. . . . x *c3x|    4|    0     0     0     0     2     2|   *     *     *     *     *    *     *     *     *    *     *    *    *     * 12960|   0    0    0    0    0    0    0    0    0    1    0    0    0    0    0    1    0    0    1    1|  0    0   0    0    0    1    0    0    1    1   0   0    1    1   1| 0  0   1   1   1  1|
        |--------------+-----+----- ----- ----- ----- ----- -----+---- ----- ----- ----- ----- ---- ----- ----- ----- ---- ----- ---- ---- ----- -----+---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----+--- ---- --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ---+-- -- --- --- --- --|
        |x3x3x . . *c .|   24|   12    12    12     0     0     0|   4     6     0     0     0    4     0     0     0    0     0    0    0     0     0|2160    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *|  1    1   1    0    0    0    0    0    0    0   0   0    0    0   0| 1  1   1   0   0  0|
        |x3x . x . *c .|   12|    6     6     0     6     0     0|   2     0     3     0     0    0     3     0     0    0     0    0    0     0     0|   * 4320    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *|  1    0   0    1    1    0    0    0    0    0   0   0    0    0   0| 1  1   0   1   0  0|
        |x3x . . x *c .|   12|    6     6     0     0     6     0|   2     0     0     3     0    0     0     3     0    0     0    0    0     0     0|   *    * 4320    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *|  0    1   0    1    0    1    0    0    0    0   0   0    0    0   0| 1  0   1   1   0  0|
        |x3x . . . *c3x|   12|    6     6     0     0     0     6|   2     0     0     0     3    0     0     0     3    0     0    0    0     0     0|   *    *    * 4320    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *|  0    0   1    0    1    1    0    0    0    0   0   0    0    0   0| 0  1   1   1   0  0|
        |x . x3x . *c .|   12|    6     0     6     6     0     0|   0     3     3     0     0    0     0     0     0    2     0    0    0     0     0|   *    *    *    * 4320    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *|  1    0   0    0    0    0    1    1    0    0   0   0    0    0   0| 1  1   0   0   1  0|
        |x . x . x *c .|    8|    4     0     4     0     4     0|   0     2     0     2     0    0     0     0     0    0     2    0    0     0     0|   *    *    *    *    * 6480    *    *    *    *    *    *    *    *    *    *    *    *    *    *|  0    1   0    0    0    0    1    0    1    0   0   0    0    0   0| 1  0   1   0   1  0|
        |x . x . . *c3x|   12|    6     0     6     0     0     6|   0     3     0     0     3    0     0     0     0    0     0    2    0     0     0|   *    *    *    *    *    * 4320    *    *    *    *    *    *    *    *    *    *    *    *    *|  0    0   1    0    0    0    0    1    1    0   0   0    0    0   0| 0  1   1   0   1  0|
        |x . . x3x *c .|   12|    6     0     0     6     6     0|   0     0     3     3     0    0     0     0     0    0     0    0    2     0     0|   *    *    *    *    *    *    * 4320    *    *    *    *    *    *    *    *    *    *    *    *|  0    0   0    1    0    0    1    0    0    1   0   0    0    0   0| 1  0   0   1   1  0|
        |x . . x . *c3x|    8|    4     0     0     4     0     4|   0     0     2     0     2    0     0     0     0    0     0    0    0     2     0|   *    *    *    *    *    *    *    * 6480    *    *    *    *    *    *    *    *    *    *    *|  0    0   0    0    1    0    0    1    0    1   0   0    0    0   0| 0  1   0   1   1  0|
        |x . . . x *c3x|    8|    4     0     0     0     4     4|   0     0     0     2     2    0     0     0     0    0     0    0    0     0     2|   *    *    *    *    *    *    *    *    * 6480    *    *    *    *    *    *    *    *    *    *|  0    0   0    0    0    1    0    0    1    1   0   0    0    0   0| 0  0   1   1   1  0|
        |. x3x3x . *c .|   24|    0    12    12    12     0     0|   0     0     0     0     0    4     6     0     0    4     0    0    0     0     0|   *    *    *    *    *    *    *    *    *    * 2160    *    *    *    *    *    *    *    *    *|  1    0   0    0    0    0    0    0    0    0   1   1    0    0   0| 1  1   0   0   0  1|
        |. x3x . x *c .|   12|    0     6     6     0     6     0|   0     0     0     0     0    2     0     3     0    0     3    0    0     0     0|   *    *    *    *    *    *    *    *    *    *    * 4320    *    *    *    *    *    *    *    *|  0    1   0    0    0    0    0    0    0    0   1   0    1    0   0| 1  0   1   0   0  1|
        |. x3x . . *c3x|   24|    0    12    12     0     0    12|   0     0     0     0     0    4     0     0     6    0     0    4    0     0     0|   *    *    *    *    *    *    *    *    *    *    *    * 2160    *    *    *    *    *    *    *|  0    0   1    0    0    0    0    0    0    0   0   1    1    0   0| 0  1   1   0   0  1|
        |. x . x3x *c .|   12|    0     6     0     6     6     0|   0     0     0     0     0    0     3     3     0    0     0    0    2     0     0|   *    *    *    *    *    *    *    *    *    *    *    *    * 4320    *    *    *    *    *    *|  0    0   0    1    0    0    0    0    0    0   1   0    0    1   0| 1  0   0   1   0  1|
        |. x . x . *c3x|    8|    0     4     0     4     0     4|   0     0     0     0     0    0     2     0     2    0     0    0    0     2     0|   *    *    *    *    *    *    *    *    *    *    *    *    *    * 6480    *    *    *    *    *|  0    0   0    0    1    0    0    0    0    0   0   1    0    1   0| 0  1   0   1   0  1|
        |. x . . x *c3x|    8|    0     4     0     0     4     4|   0     0     0     0     0    0     0     2     2    0     0    0    0     0     2|   *    *    *    *    *    *    *    *    *    *    *    *    *    *    * 6480    *    *    *    *|  0    0   0    0    0    1    0    0    0    0   0   0    1    1   0| 0  0   1   1   0  1|
        |. . x3x3x *c .|   24|    0     0    12    12    12     0|   0     0     0     0     0    0     0     0     0    4     6    0    4     0     0|   *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    * 2160    *    *    *|  0    0   0    0    0    0    1    0    0    0   1   0    0    0   1| 1  0   0   0   1  1|
        |. . x3x . *c3x|   24|    0     0    12    12     0    12|   0     0     0     0     0    0     0     0     0    4     0    4    0     6     0|   *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    * 2160    *    *|  0    0   0    0    0    0    0    1    0    0   0   1    0    0   1| 0  1   0   0   1  1|
        |. . x . x *c3x|   12|    0     0     6     0     6     6|   0     0     0     0     0    0     0     0     0    0     3    2    0     0     3|   *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    * 4320    *|  0    0   0    0    0    0    0    0    1    0   0   0    1    0   1| 0  0   1   0   1  1|
        |. . . x3x *c3x|   12|    0     0     0     6     6     6|   0     0     0     0     0    0     0     0     0    0     0    0    2     3     3|   *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    *    * 4320|  0    0   0    0    0    0    0    0    0    1   0   0    0    1   1| 0  0   0   1   1  1|
        |--------------+-----+----- ----- ----- ----- ----- -----+---- ----- ----- ----- ----- ---- ----- ----- ----- ---- ----- ---- ---- ----- -----+---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----+--- ---- --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ---+-- -- --- --- --- --|
        |x3x3x3x . *c .|  120|   60    60    60    60     0     0|  20    30    30     0     0   20    30     0     0   20     0    0    0     0     0|   5   10    0    0   10    0    0    0    0    0    5    0    0    0    0    0    0    0    0    0|432    *   *    *    *    *    *    *    *    *   *   *    *    *   *| 1  1   0   0   0  0|
        |x3x3x . x *c .|   48|   24    24    24     0    24     0|   8    12     0    12     0    8     0    12     0    0    12    0    0     0     0|   2    0    4    0    0    6    0    0    0    0    0    4    0    0    0    0    0    0    0    0|  * 1080   *    *    *    *    *    *    *    *   *   *    *    *   *| 1  0   1   0   0  0|
        |x3x3x . . *c3x|  120|   60    60    60     0     0    60|  20    30     0     0    30   20     0     0    30    0     0   20    0     0     0|   5    0    0   10    0    0   10    0    0    0    0    0    5    0    0    0    0    0    0    0|  *    * 432    *    *    *    *    *    *    *   *   *    *    *   *| 0  1   1   0   0  0|
        |x3x . x3x *c .|   36|   18    18     0    18    18     0|   6     0     9     9     0    0     9     9     0    0     0    0    6     0     0|   0    3    3    0    0    0    0    3    0    0    0    0    0    3    0    0    0    0    0    0|  *    *   * 1440    *    *    *    *    *    *   *   *    *    *   *| 1  0   0   1   0  0|
        |x3x . x . *c3x|   24|   12    12     0    12     0    12|   4     0     6     0     6    0     6     0     6    0     0    0    0     6     0|   0    2    0    2    0    0    0    0    3    0    0    0    0    0    3    0    0    0    0    0|  *    *   *    * 2160    *    *    *    *    *   *   *    *    *   *| 0  1   0   1   0  0|
        |x3x . . x *c3x|   24|   12    12     0     0    12    12|   4     0     0     6     6    0     0     6     6    0     0    0    0     0     6|   0    0    2    2    0    0    0    0    0    3    0    0    0    0    0    3    0    0    0    0|  *    *   *    *    * 2160    *    *    *    *   *   *    *    *   *| 0  0   1   1   0  0|
        |x . x3x3x *c .|   48|   24     0    24    24    24     0|   0    12    12    12     0    0     0     0     0    8    12    0    8     0     0|   0    0    0    0    4    6    0    4    0    0    0    0    0    0    0    0    2    0    0    0|  *    *   *    *    *    * 1080    *    *    *   *   *    *    *   *| 1  0   0   0   1  0|
        |x . x3x . *c3x|   48|   24     0    24    24     0    24|   0    12    12     0    12    0     0     0     0    8     0    8    0    12     0|   0    0    0    0    4    0    4    0    6    0    0    0    0    0    0    0    0    2    0    0|  *    *   *    *    *    *    * 1080    *    *   *   *    *    *   *| 0  1   0   0   1  0|
        |x . x . x *c3x|   24|   12     0    12     0    12    12|   0     6     0     6     6    0     0     0     0    0     6    4    0     0     6|   0    0    0    0    0    3    2    0    0    3    0    0    0    0    0    0    0    0    2    0|  *    *   *    *    *    *    *    * 2160    *   *   *    *    *   *| 0  0   1   0   1  0|
        |x . . x3x *c3x|   24|   12     0     0    12    12    12|   0     0     6     6     6    0     0     0     0    0     0    0    4     6     6|   0    0    0    0    0    0    0    2    3    3    0    0    0    0    0    0    0    0    0    2|  *    *   *    *    *    *    *    *    * 2160   *   *    *    *   *| 0  0   0   1   1  0|
        |. x3x3x3x *c .|  120|    0    60    60    60    60     0|   0     0     0     0     0   20    30    30     0   20    30    0   20     0     0|   0    0    0    0    0    0    0    0    0    0    5   10    0   10    0    0    5    0    0    0|  *    *   *    *    *    *    *    *    *    * 432   *    *    *   *| 1  0   0   0   0  1|
        |. x3x3x . *c3x|  192|    0    96    96    96     0    96|   0     0     0     0     0   32    48     0    48   32     0   32    0    48     0|   0    0    0    0    0    0    0    0    0    0    8    0    8    0   24    0    0    8    0    0|  *    *   *    *    *    *    *    *    *    *   * 270    *    *   *| 0  1   0   0   0  1|
        |. x3x . x *c3x|   48|    0    24    24     0    24    24|   0     0     0     0     0    8     0    12    12    0    12    8    0     0    12|   0    0    0    0    0    0    0    0    0    0    0    4    2    0    0    6    0    0    4    0|  *    *   *    *    *    *    *    *    *    *   *   * 1080    *   *| 0  0   1   0   0  1|
        |. x . x3x *c3x|   24|    0    12     0    12    12    12|   0     0     0     0     0    0     6     6     6    0     0    0    4     6     6|   0    0    0    0    0    0    0    0    0    0    0    0    0    2    3    3    0    0    0    2|  *    *   *    *    *    *    *    *    *    *   *   *    * 2160   *| 0  0   0   1   0  1|
        |. . x3x3x *c3x|  120|    0     0    60    60    60    60|   0     0     0     0     0    0     0     0     0   20    30   20   20    30    30|   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    5    5   10   10|  *    *   *    *    *    *    *    *    *    *   *   *    *    * 432| 0  0   0   0   1  1|
        |--------------+-----+----- ----- ----- ----- ----- -----+---- ----- ----- ----- ----- ---- ----- ----- ----- ---- ----- ---- ---- ----- -----+---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----+--- ---- --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ---+-- -- --- --- --- --|
        |x3x3x3x3x *c .|  720|  360   360   360   360   360     0| 120   180   180   180     0  120   180   180     0  120   180    0  120     0     0|  30   60   60    0   60   90    0   60    0    0   30   60    0   60    0    0   30    0    0    0|  6   15   0   20    0    0   15    0    0    0   6   0    0    0   0|72  *   *   *   *  *|
        |x3x3x3x . *c3x| 1920|  960   960   960   960     0   960| 320   480   480     0   480  320   480     0   480  320     0  320    0   480     0|  80  160    0  160  160    0  160    0  240    0   80    0   80    0  240    0    0   80    0    0| 16    0  16    0   80    0    0   40    0    0   0  10    0    0   0| * 27   *   *   *  *|
        |x3x3x . x *c3x|  240|  120   120   120     0   120   120|  40    60     0    60    60   40     0    60    60    0    60   40    0     0    60|  10    0   20   20    0   30   20    0    0   30    0   20   10    0    0   30    0    0   20    0|  0    5   2    0    0   10    0    0   10    0   0   0    5    0   0| *  * 216   *   *  *|
        |x3x . x3x *c3x|   72|   36    36     0    36    36    36|  12     0    18    18    18    0    18    18    18    0     0    0   12    18    18|   0    6    6    6    0    0    0    6    9    9    0    0    0    6    9    9    0    0    0    6|  0    0   0    2    3    3    0    0    0    3   0   0    0    3   0| *  *   * 720   *  *|
        |x . x3x3x *c3x|  240|  120     0   120   120   120   120|   0    60    60    60    60    0     0     0     0   40    60   40   40    60    60|   0    0    0    0   20   30   20   20   30   30    0    0    0    0    0    0   10   10   20   20|  0    0   0    0    0    0    5    5   10   10   0   0    0    0   2| *  *   *   * 216  *|
        |. x3x3x3x *c3x| 1920|    0   960   960   960   960   960|   0     0     0     0     0  320   480   480   480  320   480  320  320   480   480|   0    0    0    0    0    0    0    0    0    0   80  160   80  160  240  240   80   80  160  160|  0    0   0    0    0    0    0    0    0    0  16  10   40   80  16| *  *   *   *   * 27|
        +--------------+-----+----- ----- ----- ----- ----- -----+---- ----- ----- ----- ----- ---- ----- ----- ----- ---- ----- ---- ---- ----- -----+---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----+--- ---- --- ---- ---- ---- ---- ---- ---- ---- --- --- ---- ---- ---+-- -- --- --- --- --+

        >>> im = IncidenceMatrix("x3o3x")
        >>> im.print_largest_table()
        +-----+--+-- -- --+-- -- --+
        |. . .|24| 1  1  1| 1  1  1|
        |-----+--+-- -- --+-- -- --|
        |x . .| 2|12  *  *| 1  1  0|
        |. x .| 2| * 12  *| 1  0  1|
        |. . x| 2| *  * 12| 0  1  1|
        |-----+--+-- -- --+-- -- --|
        |x3x .| 6| 3  3  0| 4  *  *|
        |x . x| 4| 2  0  2| *  6  *|
        |. x3x| 6| 0  3  3| *  *  4|
        +-----+--+-- -- --+-- -- --+
        """

        dim = self.max_diagram.get_dim()
        if dim == 1:
            return np.array([[2]]), [1]

        row_strings, sub_diagrams, dimensions = self.max_diagram.get_subdiagrams()

        matrix = np.array([[0] * sum(dimensions)] * sum(dimensions))
        order = self.max_diagram.get_vertex_count()

        # first column
        row_counter = 0
        for d, sub_dias in sub_diagrams.items():
            for sub in sub_dias:
                if row_counter == 0:
                    matrix[0][0]=order
                else:
                    matrix[row_counter][0]=sub.get_vertex_count()
                row_counter+=1

        # prepare lower-off-diagonal parts (only relevant for dim>2)
        for d in range(dim, 2, -1):
            row_start = sum(dimensions[:d-1])
            subs = sub_diagrams[d-1]
            for r, sub in enumerate(subs):
                sub_im = IncidenceMatrix(sub.diagram_string)
                sub_matrix, sub_dimensions = sub_im.compute_largest_incidence_matrix()
                # we need the entries of the last diagonal and need to place them, where the corresponding sub_row fits into the row of the current row
                # for example x3x3x
                sub_rows, subsub_dias, sub_dimensions = sub.get_subdiagrams()
                for sub_dim,subsubs in subsub_dias.items():
                    if sub_dim>0: # we can start with edges, vertices are done already
                        sub_start = sum(sub_dimensions[:sub_dim])
                        for i, subsub in enumerate(subsubs):
                            position = row_strings.index(subsub.diagram_string)
                            matrix[row_start+r][position] = sub_matrix[sub_start+i, sub_start+i]

        # compute last diagonal part
        # it uses the fact that the total vertex count is related to the product of [fnfn]_{ii}*[fnf(n-1)]_{ij}*[f(n-1)f(n-2)]_{jk}*...*2
        # but you always have to select non-zero elements
        start = sum(dimensions[:dim - 1])
        end = sum(dimensions[:dim])

        for row in range(start, end):
            for col in range(start, end):
                if row != col:
                    matrix[row][col] = "-1"
                else:
                    value = order
                    back_row = row
                    final_back_row_end = sum(dimensions[:1])
                    back_step = 1
                    while back_row >= final_back_row_end:
                        back_col_start = sum(dimensions[:dim - back_step - 1])
                        back_col_stop = sum(dimensions[:dim - back_step])
                        idx = 0
                        for c in range(back_col_start, back_col_stop):
                            m = matrix[back_row][c]
                            if m != 0:
                                value = value / m
                                break
                            else:
                                idx += 1
                        back_step += 1
                        back_row = sum(dimensions[:dim - back_step]) + idx
                    matrix[row][col] = value

        # work on upper half
        for sub_dim in range(dim - 1, 0, -1):
            start_diagonal = sum(dimensions[:sub_dim - 1])
            start_off_diagonal = sum(dimensions[:sub_dim])
            end_off_diagonal = sum(dimensions[:sub_dim + 1])
            subs = sub_diagrams[sub_dim]
            subsubs = sub_diagrams[sub_dim - 1]
            # compute off-diagonal part
            for  sub in subs:
                col = row_strings.index(sub.diagram_string)
                for subsub in subsubs:
                    row = row_strings.index(subsub.diagram_string)
                    if sub.contains(subsub):
                        matrix[row][col] = 1

            # compute diagonal part
            if sub_dim > 1:
                for i in range(len(subsubs)):
                    for j in range(len(subsubs)):
                        if i == j:
                            idx = 0
                            for r in range(start_off_diagonal, end_off_diagonal):
                                m = matrix[r][start_diagonal + i]
                                if m != 0:
                                    break
                                else:
                                    idx += 1
                            # get value from next diagonal part
                            value = m * matrix[start_off_diagonal + idx][start_off_diagonal + idx]
                            idx = 0
                            for c in range(start_off_diagonal, end_off_diagonal):
                                m = matrix[start_diagonal + i][c]
                                if m != 0:
                                    break
                                else:
                                    idx += 1
                            value = value / m
                            matrix[start_diagonal + i][start_diagonal + j] = value
                        else:
                            matrix[start_diagonal + i][start_diagonal + j] = -1

            # compute far-off-diagonal part
            for sub_dim_row in range(dim - 1, -1, -1):
                for sub_dim_col in range(sub_dim_row + 2, dim):
                    row_subs = sub_diagrams[sub_dim_row]
                    col_subs = sub_diagrams[sub_dim_col]
                    for row_sub in row_subs:
                        row = row_strings.index(row_sub.diagram_string)
                        for col_sub in col_subs:
                            col = row_strings.index(col_sub.diagram_string)
                            if col_sub.contains(row_sub):
                                matrix[row][col] = 1
        return matrix, dimensions

    def compute_incidence_matrix(self):
        """
        compute the incidence matrix under the assumption that all nodes are active (ringed)
        It is not successful yet. I need a smarter way to convert original branched diagrams into linear ones
        >>> im = IncidenceMatrix("o5x")
        >>> im.print_table()
        +---+-+-+
        |. .|5|2|
        |---+-+-|
        |. x|2|5|
        +---+-+-+
        >>> im = IncidenceMatrix("o3o3o5x")
        >>> im.print_table()
        +-------+---+----+---+---+
        |. . . .|600|   4|  6|  4|
        |-------+---+----+---+---|
        |. . . x|  2|1200|  3|  3|
        |-------+---+----+---+---|
        |. . o5x|  5|   5|720|  2|
        |-------+---+----+---+---|
        |. o3o5x| 20|  30| 12|120|
        +-------+---+----+---+---+
        >>> im = IncidenceMatrix("x3o3o3o3o *c3o")
        >>> im.print_table()
        +--------------+--+---+---+----+--- ---+-- --+
        |. . . . . *c .|27| 16| 80| 160| 80  40|16 10|
        |--------------+--+---+---+----+--- ---+-- --|
        |x . . . . *c .| 2|216| 10|  30| 20  10| 5  5|
        |--------------+--+---+---+----+--- ---+-- --|
        |x3o . . . *c .| 3|  3|720|   6|  6   3| 2  3|
        |--------------+--+---+---+----+--- ---+-- --|
        |x3o3o . . *c .| 4|  6|  4|1080|  2   1| 1  2|
        |--------------+--+---+---+----+--- ---+-- --|
        |x3o3o3o . *c .| 5| 10| 10|   5|432   *| 1  1|
        |x3o3o . . *c3o| 5| 10| 10|   5|  * 216| 0  2|
        |--------------+--+---+---+----+--- ---+-- --|
        |x3o3o3o3o *c .| 6| 15| 20|  15|  6   0|72  *|
        |x3o3o3o . *c3o|10| 40| 80|  80| 16  16| * 27|
        +--------------+--+---+---+----+--- ---+-- --+

        >>> im = IncidenceMatrix("x3x3o3o")
        >>> im.print_table()
        +-------+--+-- --+-- --+-- --+
        |. . . .|20| 1  3| 3  3| 3  1|
        |-------+--+-- --+-- --+-- --|
        |x . . .| 2|10  *| 3  0| 3  0|
        |. x . .| 2| * 30| 1  2| 2  1|
        |-------+--+-- --+-- --+-- --|
        |x3x . .| 6| 3  3|10  *| 2  0|
        |. x3o .| 3| 0  3| * 20| 1  1|
        |-------+--+-- --+-- --+-- --|
        |x3x3o .|12| 6 12| 4  4| 5  *|
        |. x3o3o| 4| 0  6| 0  4| *  5|
        +-------+--+-- --+-- --+-- --+


        >>> im = IncidenceMatrix("o3o5x")
        >>> im.print_largest_table()

        >>> im = IncidenceMatrix("x3o")
        >>> im.print_table()
        +---+-+-+
        |. .|3|1|
        |---+-+-|
        |x .|2|3|
        +---+-+-+
        """

        dim = self.diagram.get_dim()
        row_strings, sub_diagrams, dimensions = self.diagram.get_subdiagrams()
        matrix = self.max_matrix.copy()
        max_rows = self.max_rows.copy()
        surviving_row_patterns = [row.replace('o','x') for row in self.rows]
        # delete unvalid rows and columns
        rows_to_remove = []
        for r,row in enumerate(self.max_rows):
            if not row in surviving_row_patterns:
                rows_to_remove.append(r)

        # remove from large indices to smaller indices
        for r in reversed(rows_to_remove):
            matrix = np.delete(matrix,r,axis=0)
            matrix = np.delete(matrix,r,axis=1)
            max_rows.remove(self.max_rows[r])


        # for each o, we have to half the number of elements in the diagonal once for each active orthogonal dimension
        sub_rows,sub_diagrams,sub_dimensions = self.diagram.get_subdiagrams()
        sub_diagrams.pop(0)

        matrix[0][0]=self.diagram.get_vertex_count()

        for sub_dim,subs in sub_diagrams.items():
            for sub in subs:
                row = self.rows.index(sub.diagram_string)
                # remaining diagonals
                if row>0:
                    matrix[row][0] = sub.get_vertex_count()  # first column
                    matrix[row][row] = matrix[row][row] //self.diagram.get_maximal_orthogonal_contraction(sub)

        # now, we repeat the analysis from the large_incidence_matrix
        # start with the lower left part of the matrix  (they are taken from sub-diagrams)

        # Populates matrix from subdiagram incidence matrices, that are computed recursively from dimension 2
        for d in range(dim, 2, -1):
            subs = sub_diagrams[d-1]
            for sub in subs:
                sub_im = IncidenceMatrix(sub.diagram_string)
                sub_matrix, sub_dimensions = sub_im.compute_incidence_matrix()
                # we need the entries of the last diagonal and need to place them,
                # where the corresponding sub_row fits into the row of the current row
                # for example x3x3x
                row = row_strings.index(sub.diagram_string)
                sub_rows, subsub_dias, sub_dimensions = sub.get_subdiagrams()
                for sub_dim,subsubs in subsub_dias.items():
                    if sub_dim>0: # we can start with edges, vertices are done already
                        sub_start = sum(sub_dimensions[:sub_dim])
                        for i, subsub in enumerate(subsubs):
                            position = row_strings.index(subsub.diagram_string)
                            matrix[row][position] = sub_matrix[sub_start+i, sub_start+i]

        # compute the upper part of the matrix from the lower part
        sub_rows, sub_diagrams, sub_dimensions = self.diagram.get_subdiagrams()
        # work on upper half (next to the diagonal)
        # compute the far off-diagonal terms
        for sub_dim_row in range(dim - 1, -1, -1):
            for sub_dim_col in range(sub_dim_row + 1, dim):
                row_subs = sub_diagrams[sub_dim_row]
                col_subs = sub_diagrams[sub_dim_col]
                for row_sub in row_subs:
                    row = row_strings.index(row_sub.diagram_string)
                    for col_sub in col_subs:
                        col = row_strings.index(col_sub.diagram_string)
                        if col_sub.contains(row_sub):
                            matrix[row][col] = matrix[col][col]*matrix[col][row]//matrix[row][row]




        return matrix, dimensions



if __name__ == '__main__':
    solids = ("o3o3o3x","o3o3x3o","o3o3x3x","x3o3o3x","o3x3o3x","o3x3x3o","o3x3x3x","x3o3x3x","x3x3x3x")
    for solid in solids:
        im = IncidenceMatrix(solid)
        im.print_table()
    for solid in solids:
        im = IncidenceMatrix(solid)
        im.print_latex_table()
