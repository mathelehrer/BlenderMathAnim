from __future__ import annotations

import typing
from itertools import combinations

import numpy as np
from anytree import Node, RenderTree
import networkx as nx
import matplotlib.pyplot as plt
from sympy import factorial, subsets

### Warning
# this only works for linear Dynkin diagrams so far
import doctest

DEBUG=True
logging =[]

class Diagram:

    def __init__(self,diagram_string):
        # convert diagram_string into graph
        self.diagram_string = diagram_string
        # list of letters to match the position of the branches in the notation
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']

        self.graph = nx.Graph()
        self.create_graph()


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
                            pos=(branch_pos,branch_height)
                            branch_height+=1
                        else:
                            pos = (node_count, 0)
                        self.graph.add_node((l,node_count),pos=pos)
                        # add edges
                        if weight>2:
                            if branch_index is not None:
                                # search for branching node
                                for node in self.graph.nodes:
                                    if node[1]==branch_index:
                                        self.graph.add_edge((l,node_count),node,weight=weight)
                                        break
                                branch_index = None
                            else:
                                # connect to previous node
                                self.graph.add_edge(list(self.graph.nodes.keys())[-2],(l,node_count),weight=weight)
                        node_count += 1
                    else:
                        branch_index =self.letters.index(l)
                        branch_pos = branch_index
                        branch_height = 1


    def show_graph(self):
        """
        display the graph
        >>> d=Diagram("x3o . x3x3x3x *c3x")
        >>> d.show_graph()
        """
        plt.figure(figsize=(10,2))

        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos,font_weight='bold' ,with_labels=True)
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph,pos, edge_labels=labels)

        # for cc in nx.connected_components(G):
        #     print([labels[c] for c in cc])

        plt.show()

    def find_node(self,root,index)->Node:
        if root.name[1]==index:
            return root
        else:
            for node in root.children:
                return self.find_node(node,index)

    def parse_diagram(self):
        """
        convert Coxeter diagram into a graph

        >>> d = Diagram("x3o . x3o *c3x")
        >>> d.show_tree()
        ('x', 0)
        └── ('o', 1)
        ('x', 3)
        └── ('o', 4)
        ('x', 5)

        >>> d = Diagram("o3x3x . x *c3o")
        >>> d.show_tree()
        ('o', 0)
        └── ('x', 1)
            └── ('x', 2)
                └── ('o', 5)
        ('x', 4)

        >>> d = Diagram("x3x3x3x3x3 *c3x")
        >>> d.show_tree()
        ('x', 0)
        └── ('x', 1)
            └── ('x', 2)
                ├── ('x', 3)
                │   └── ('x', 4)
                └── ('x', 5)
        """

        # list of letters to match the position of the branches in the notation
        letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
        roots = []

        # detect branches
        diagram_string=self.diagram_string.replace(" *","*")
        diagram_string=diagram_string.replace(" ","2")
        parts = diagram_string.split("*")

        node_count = 0
        # deal with diagram part by part
        for part in parts:
            follow_digits = False
            weight =0
            root = None
            last = None
            for i,l in enumerate(part):
                if l.isdigit():
                    if not follow_digits:
                        weight = int(l)
                        follow_digits = True
                    else:
                        weight = weight*10+int(l)
                else:
                    follow_digits=False
                    if l=='.':
                        node_count+=1
                        if root:
                            roots.append(root)
                        root = None
                        last = None
                    elif l=='o' or l=='x':
                        if last is not None:
                            if weight>2:
                                last = Node((l,node_count),parent=last,weight=weight)
                            else:
                                roots.append(root)
                                root = Node((l,node_count),weight=weight)
                                last = root
                        else:
                            root = Node((l,node_count),weight=weight)
                            last = root
                        node_count+=1
                    else:
                        last_index = letters.index(l)
                        # try to find a node to connect to
                        for root in roots:
                            last = self.find_node(root,last_index)
                            if last:
                                break
            if root:
                if root not in roots:
                    roots.append(root)

        return roots

    def __get_positions(self,node)->[]:
        positions = [node.name[1]]
        for child in node.children:
            positions+=self.__get_positions(child)

        return positions

    def get_positions(self)->[]:
        """
        return a list of integers that correspond to the indices of the existing nodes

        >>> d = Diagram("x3x3x5x")
        >>> d.get_positions()
        [0, 1, 2, 3]
        >>> d = Diagram("x x3o4x")
        >>> d.get_positions()
        [0, 1, 2, 3]
        >>> d = Diagram("x . x . o *c3x")
        >>> d.get_positions()
        [0, 2, 4, 5]
        """
        positions = []
        for root in self.roots:
            positions+=self.get_positions_recursively(root)

        positions.sort()
        return positions

    def is_linear(self,node)->bool:
        """
       Checks whether a Coxeter tree is linear

       >>> diagram_strings = ["x3x5x","x3x3x3x3x *c3x"]
       >>> diagrams = [Diagram(s) for s in diagram_strings]
       >>> [d.is_linear(d.roots[0]) for d in diagrams]
       [True, False]
        """
        if len(node.children)>1:
            return False
        elif len(node.children)==0:
            return True
        else:
            return self.is_linear(node.children[0])

    def get_diagrams_from_connected_components(self)->[str]:
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
            id2label={}
            id2node = {}
            node2pos = {}
            pos= 0 # the position of the connected components might differ from the original larger diagram
            for label,node_id in comp:
                id2label[node_id]=label
                id2node[node_id]=self.graph.nodes[(label,node_id)]
                node2pos[(label,node_id)]=pos
                pos+=1

            edges = nx.edges(self.graph)

            diagram = ""
            for val in id2label.values():
                diagram+=val+" "

            diagram = diagram[:-1]

            branching = ""
            already_connected =[]
            for edge in edges:
                if edge[0] in comp and edge[1] in comp:
                    a = edge[0][1]
                    b = edge[1][1]
                    pos = node2pos[edge[0]]
                    connection = a
                    if a>b:
                        connection = b
                        pos = node2pos[edge[1]]
                    w = self.graph.get_edge_data(*edge)["weight"]
                    if connection in already_connected:
                        alt_pos = node2pos[edge[1]]
                        branching = "*"+self.letters[connection]+str(w)+diagram[2*alt_pos]
                        diagram = diagram[:2*alt_pos]
                    else:
                        diagram = diagram[:2*pos+1]+str(w)+diagram[2*pos+2:]
                    already_connected.append(connection)
            diagram = diagram + branching
            comp_strings.append(diagram)

        return comp_strings

    def get_complement(self)->Diagram:
        """
        return a diagram that has the same structure as the unringed nodes of
        the given diagram
        >>> d = Diagram("x3x3o5o")
        >>> d.get_complement().diagram_string
        '. . x5x'
        """

        complement_str = self.diagram_string.replace("x",".").replace("o","x")
        # remove connections to dots
        dot_position = 0
        while dot_position>-1:
            try:
                dot_position = complement_str.index(".",dot_position)
            except ValueError:
                break
            if dot_position == 0:
                complement_str = ". "+complement_str[2:]
            elif dot_position == len(complement_str)-1:
                complement_str = complement_str[:-2]+" ."
            else:
                complement_str = complement_str[:dot_position-1]+" . "+complement_str[dot_position+2:]
            dot_position+=1
        return Diagram(complement_str)

    def get_weights(self,node)->[]:
        """
        The function assumes that the node is linear
        >>> diagram_strings = ["x3x5x","x17x","x3o","x3x4x3x"]
        >>> diagrams = [Diagram(string) for string in diagram_strings]
        >>> [diagram.get_weights(diagram.roots[0]) for diagram in diagrams]
        [[3, 5], [17], [3], [3, 4, 3]]
        """
        weights = []
        if node.weight:
            weights.append(node.weight)

        for child in node.children:
            weights+=self.get_weights(child)

        return weights

    def __get_vertex_count(self,node):
        """
        call this only with connected components
        use get_vertex_count() for the usage

        """
        if (len(node.children)==0):
            if node.name[0]=='o':
                return 0
            if node.name[0]=='x':
                return 2

        if self.is_linear(node):
            # n=2, A(n), B(n), H3, H4
            # TODO implement missing cases

            weights = self.get_weights(node)
            if len(weights)==1:
                size = weights[0]
                if node.name[0]=='x' and node.children[0].name[0]=='x':
                    return 2*size
                elif node.name[0]=='x' or node.children[0].name[0]=='x':
                    return size
                else:
                    return 0


            pass
        else:
            # E6,
            # TODO lots of diagrams missing
            pass
        weights = []
        return 0

    def get_vertex_count(self):
        """
        compute the number of vertices that the corresponding diagram generates
        >>> diagram_strings = ["x","o","x x","x o","x x x x","x17o","x5x"]
        >>> diagrams = [Diagram(string) for string in diagram_strings]
        >>> [diagram.get_vertex_count() for diagram in diagrams]
        [2, 0, 4, 0, 16, 17, 10]
        """

        n = 1
        for root in self.roots:
            n*=self.__get_vertex_count(root)

        return n

def create_diagram(edges,nodes):
    """
    create visual Dynkin diagram from internal representation of edges and nodes
    >>> create_diagram([3,3,5],['x','x','x','x'])
    'x3x3x5x'
    >>> create_diagram([2,3,4],['x','x','o','x'])
    'x x3o4x'
    >>> create_diagram(*get_complement(*parse_diagram("x3o3x3x3o4o")))
    'x x4x'
    """
    diagram = ""
    for i in range(len(nodes)):
        if i>0:
            d = ' '
            if edges[i-1]>2:
                d=str(edges[i-1])
            diagram+=d
            diagram+=nodes[i]
        else:
            diagram+=nodes[i]
    return diagram

def parse_diagram(diagram):
    """
    1. replace spaces with 2
    2. read out numbers and replace them with ','
    3. split diagram with respect to ','
    return list of edges and list of nodes

    >>> parse_diagram("x3x3x5x")
    ([3, 3, 5], ['x', 'x', 'x', 'x'])
    >>> parse_diagram("x x3o4x")
    ([2, 3, 4], ['x', 'x', 'o', 'x'])
    """

    diagram = diagram.replace(" ","2")

    # remove some notation extensions that are used to indicate the sub structure
    while '2.2' in diagram:
        diagram = diagram.replace("2.2","2")

    # deal with end and front parts
    diagram = diagram.replace(".2","")
    diagram = diagram.replace("2.","")

    edges = []
    diagram2 = ""
    follow_digits = False
    for i in range(len(diagram)):
        d = diagram[i]
        if diagram[i].isdigit():
            if not follow_digits:
                edges.append(int(d))
                diagram2+=','
                follow_digits = True
            else:
                edges[-1]=edges[-1]*10+int(d)
        else:
            follow_digits = False
            diagram2+=d


    nodes = diagram2.split(",")
    return edges,nodes

def compute_order_recursive(edges,nodes):

    n = len(nodes)

    # trivial case of an edge
    if n == 1:
        if nodes[0] =='o':
            return 0
        if nodes[0]=='x':
            return 2

    if n == 2:
        size= edges[0]
        if nodes[0] == 'x' and nodes[1] == 'x':
            return 2 * size
        elif nodes[0] == 'x' or nodes[1] == 'x':
            if size<3: # avoid lines "o2x" or "x2o"
                return 0
            else:
                return size
        else:
            return 0

    # deal with disconnected diagrams
    try :
        pos2=edges.index(2)
    except ValueError:
        pos2=-1
    if pos2>-1:
        return (compute_order_recursive(edges[:pos2],nodes[:pos2+1])*
                compute_order_recursive(edges[pos2+1:],nodes[pos2+1:]))

    n = len(nodes)
    dim_reduction = 1
    if all(node=='o' for node in nodes):
        return 0

    if not all(node=='x' for node in nodes):
        edges_comp,nodes_comp = get_complement(edges,nodes)
        dim_reduction = compute_order_recursive(edges_comp,nodes_comp)
        if DEBUG:
            logging.append(
                  create_diagram(edges,nodes)+"->"+create_diagram(*get_complement(edges,nodes)))
    # deal with A series
    if all(d==3 for d in edges):
        return factorial(n+1)//dim_reduction

    # deal with B series
    fours = edges.count(4)
    threes = edges.count(3)
    if fours==1 and threes==len(edges)-1 and (edges[0]==4 or edges[-1]==4):
       return factorial(n)*2**n//dim_reduction

    # deal with F4
    elif n==4 and edges[1]==4:
        return 1152//dim_reduction

    # deal with H series:
    fives = edges.count(5)
    threes = edges.count(3)
    if fives==1 and threes==len(edges)-1 and (edges[0]==5 or edges[-1]==5):
        if n==3:
            return 120 // dim_reduction
        if n==4:
            return 14400// dim_reduction

    raise "not implemented yet"


def compute_order(diagram):
    """
    This algorithm computes the order of a linear Coxeter diagram.

    1. parse diagram
    2. determine group
    3. return dimension
    TODO implement algorithm for diagrams with branches

    # tests for polygons
    >>> compute_order("x175x")
    350
    >>> compute_order("x3o")
    3
    >>> compute_order("o91x")
    91
    >>> compute_order("o5o")
    0

    # test for prisms
    >>> compute_order("x x3o")
    6
    >>> compute_order("x x x x")
    16

    # test for H-series
    >>> compute_order("x3x5x")
    120
    >>> compute_order("x3x3x5x")
    14400

    # test for B and A-series
    >>> compute_order("x3x4x")
    48

    # check for os
    >>> compute_order("o3x3x")
    12
    >>> compute_order("x3o4x3x")
    576
    >>> compute_order("x3o3x5x")
    7200
    >>> compute_order("o3x3o5x")
    3600
    >>> compute_order("o3o3o5x")
    600
    >>> compute_order("x3o3o5o")
    120
    >>> compute_order("x3o o")
    0

    # >>> print(logging)
    """

    edges,nodes = parse_diagram(diagram)
    return compute_order_recursive(edges,nodes)

def check_validity(row):
    """
    check is sub diagrams are non-trivial
    >>> check_validity(". x3o .")
    True
    >>> check_validity("o . x3x")
    False
    >>> check_validity("x . o3x")
    True
    >>> check_validity(". x . o")
    False
    """

    size = compute_order(row)
    if size>0:
        return True
    return False

def get_subdiagrams(diagram):
    dimensions=[1]
    edges,nodes = parse_diagram(diagram)
    sub_diagrams = {}
    rows = []
    first_row = ""
    for i in range(len(nodes)):
        if i>0:
            first_row+=" ."
        else:
            first_row+="."
    rows.append(first_row)
    sub_diagrams[0]=[first_row]

    for d in range(1,len(nodes)):
        count = 0
        sub_diagrams[d]=[]
        combs = list(combinations(list(range(len(nodes))),d))
        for comb in combs:
            n = 0
            row = ""
            first = True
            edge  =" "
            for b in diagram:
                if b=='x' or b=='o':
                    if n in comb:
                        if not first:
                            row+=str(edge)
                            row+=b
                        else:
                            if len(row)>0:
                                row+=" "
                            row+=b
                            first = False
                    else:
                        if len(row)>0:
                            row+=" "
                        row+="." # replace node by deselector
                        first=True
                    n+=1
                    edge =""
                else:
                    edge+=b

            # check validity
            valid = check_validity(row)
            if valid:
                count+=1
                rows.append(row)
                sub_diagrams[d].append(row)
        dimensions.append(count)
    return sub_diagrams, dimensions

def print_table(rows,matrix,dimensions):
    array = np.array(matrix)
    col_matrix = array.transpose()
    widths=[max([len(row) for row in rows])]+[max([len(str(entry)) for entry in column]) for column in col_matrix]

    top ="+"+"-"*widths[0]+"+"
    sep ="|"+"-"*widths[0]+"+"
    count = 0
    for d in dimensions:
        for i in range(count,count+d):
            top+="-"*widths[i+1]+" "
            sep+="-"*widths[i+1]+" "
        top = top[:-1]
        top+="+"
        sep = sep[:-1]
        sep+="+"
        count+=d
    sep=sep[:-1]+"|"

    table = top+("\n")
    dim = dimensions[0]
    dim_sel=1
    for i,row in enumerate(matrix):
        row_string = "|"
        row_string+=rows[i]+"|"
        col_index=1
        col_dim_sel=1
        for j,(width,entry) in enumerate(zip(widths[1:],row)):
            if entry==-1:
                entry="*"
            entry_string = str(entry)
            while len(entry_string)<width:
                entry_string=" "+entry_string
            if j == col_index-1:
                row_string += entry_string + "|"
                if col_index<len(dimensions):
                    col_index += dimensions[col_dim_sel]
                col_dim_sel += 1
            else:
                row_string += entry_string + " "
        table+=row_string[:-1]+"|\n"
        if i==dim-1:
            if dim_sel<len(dimensions):
                table+=sep+"\n"
                dim+=dimensions[dim_sel]
                dim_sel+=1
            else:
                table+=top

    print(table)

def print_latex_table(rows,matrix,dimensions,diagram):
    array = np.array(matrix)
    col_matrix = array.transpose()
    widths=[max([len(row) for row in rows])]+[max([len(str(entry)) for entry in column]) for column in col_matrix]
    lines  = []

    lines.append(r"\fbox{\tt "+diagram+r"}\\")
    format = "|c|"
    for d in dimensions:
        for c in range(d):
            format+="r "
        format+="|"

    lines.append(r"\begin{tabular}{"+format+"}")


    sep_pos = [sum(dimensions[:i]) for i in range(len(dimensions))]
    for i,row in enumerate(matrix):
        if i in sep_pos:
            lines.append(r"\hline")
        row_string=r"{\tt "+rows[i]+"}&"
        for j,(width,entry) in enumerate(zip(widths[1:],row)):
            if entry==-1:
                entry="*"
            entry_string = str(entry)
            row_string += "$"+entry_string+"$" + "&"
        lines.append(row_string[:-1]+r"\\")


    lines.append(r"\hline")
    lines.append("\end{tabular}")
    for line in lines:
        print(line)

def get_positions(diagram):
    diagram = diagram.replace(" ","2")
    indices = []
    positions = []
    for l in diagram:
        if l=='x':
            indices.append(1)
        if l=='o':
            indices.append(-1)
        if l=='.':
            indices.append(0)
    for i,idx in enumerate(indices):
        if idx==1:
            positions.append(i)
    return positions


def compute_largest_incidence_matrix(diagram):
    edges,nodes = parse_diagram(diagram)
    full_diagram = diagram # TODO this has to be modified in the general case
    dim = len(nodes)

    if dim==1:
        if nodes[0]=='x':
            return np.array([[2]]),[1]
        raise "Wrong input in compute_largest_incidence_matrix"

    if dim>1:
        sub_diagrams,dimensions = get_subdiagrams(diagram)
        row_strings = []
        for sub_dias in sub_diagrams.values():
            for sub in sub_dias:
                row_strings.append(sub)

        matrix = np.array([[0]*sum(dimensions)]*sum(dimensions))
        matrix[0][0]=compute_order(diagram)

        order = compute_order(full_diagram)
        row_counter = 0

        # first column
        for d,sub_dias in sub_diagrams.items():
            for sub in sub_dias:
                if row_counter==0:
                    matrix[0][0] = order
                else:
                    matrix[row_counter][0]=compute_order(sub)
                row_counter+=1

        # prepare lower-off-diagonal parts (only relevant for dim>2)
        for d in range(dim,2,-1):
            row_start = sum(dimensions[:dim - 1])
            for sub_dim in range(dim-1,1,-1):
                col_start = sum(dimensions[:sub_dim-1])
                subs = sub_diagrams[sub_dim]
                for r,sub in enumerate(subs):
                    sub_matrix, sub_dimensions = compute_largest_incidence_matrix(sub)
                    subsub_dias,dimensions = get_subdiagrams(sub)
                    subsubs = subsub_dias[sub_dim-1]
                    sub_start = sum(sub_dimensions[:sub_dim - 1])
                    for i,subsub in enumerate(subsubs):
                        position = row_strings.index(subsub)
                        matrix[row_start+r][position]=sub_matrix[sub_start+i,sub_start+i]

        # compute last diagonal part
        start = sum(dimensions[:dim-1])
        end = sum(dimensions[:dim])

        for row in range(start,end):
            for col in range(start,end):
                if row!=col:
                    matrix[row][col]="-1"
                else:
                    value=order
                    back_row=row
                    final_back_row_end=sum(dimensions[:1])
                    back_step = 1
                    while back_row>=final_back_row_end:
                        back_col_start = sum(dimensions[:dim-back_step-1])
                        back_col_stop = sum(dimensions[:dim-back_step])
                        idx = 0
                        for c in range(back_col_start,back_col_stop):
                            m = matrix[back_row][c]
                            if m!=0:
                                value=value/m
                                break
                            else:
                                idx+=1
                        back_step+=1
                        back_row = sum(dimensions[:dim-back_step])+idx
                    matrix[row][col]=value

        if dim>2:
            # work on upper half
            for sub_dim in range(dim-1,1,-1):
                start_diagonal = sum(dimensions[:sub_dim-1])
                start_off_diagonal = sum(dimensions[:sub_dim])
                end_off_diagonal = sum(dimensions[:sub_dim+1])
                subs = sub_diagrams[sub_dim]
                subsubs = sub_diagrams[sub_dim-1]
                # compute off-diagonal part
                for c,sub in enumerate(subs):
                    positions = get_positions(sub)
                    for r,subsub in enumerate(subsubs):
                        sub_positions=get_positions(subsub)
                        if set(sub_positions)<set(positions):
                            matrix[start_diagonal+r][start_off_diagonal+c]=1
                        elif len(sub_positions)==0: # first row
                            matrix[start_diagonal+r][start_off_diagonal+c]=1
                # compute diagonal part
                if sub_dim>1:
                    for i in range(len(subsubs)):
                        for j in range(len(subsubs)):
                            if i==j:
                                idx = 0
                                for r in range(start_off_diagonal,end_off_diagonal):
                                    m = matrix[r][start_diagonal+i]
                                    if m!=0:
                                        break
                                    else:
                                        idx+=1
                                # get value from next diagonal part
                                value = m*matrix[start_off_diagonal+idx][start_off_diagonal+idx]
                                idx = 0
                                for c in range(start_off_diagonal,end_off_diagonal):
                                    m = matrix[start_diagonal+i][c]
                                    if m!=0:
                                        break
                                    else:
                                        idx+=1
                                value = value/m
                                matrix[start_diagonal+i][start_diagonal+j]=value
                            else:
                                matrix[start_diagonal+i][start_diagonal+j]=-1

            # compute far-off-diagonal part
            for sub_dim_row in range(dim - 1, -1, -1):
                for sub_dim_col in range(sub_dim_row+2,dim):
                    row_start= sum(dimensions[:sub_dim_row])
                    col_start = sum(dimensions[:sub_dim_col])
                    row_subs = sub_diagrams[sub_dim_row]
                    col_subs = sub_diagrams[sub_dim_col]
                    for r, row_sub in enumerate(row_subs):
                        for c,col_sub in enumerate(col_subs):
                            if len(get_positions(row_sub))==0: # zero dimensions
                                matrix[row_start+r][col_start+c]=1

        return matrix,dimensions

def compute_incidence_matrix(diagram):
    full_diagram = diagram.replace('o','x')
    group_order=compute_order(full_diagram)
    vertices = compute_order(diagram)

    sub_diagrams,dimensions = get_subdiagrams(diagram)
    rows = []
    for sub_dias in sub_diagrams.values():
        for sub in sub_dias:
            rows.append(sub)

    matrix, dimensions = compute_largest_incidence_matrix(full_diagram)
    return rows,matrix,dimensions



if __name__ == '__main__':
    # diagram = "x3x5x"
    # print(diagram)
    # print_table(rows,matrix,dimensions)

    # diagram = "x3x4x"
    # print(diagram)
    # compute_incidence_matrix(diagram)
    # diagram = "x4x3x"
    # print(diagram)
    # compute_incidence_matrix(diagram)
    # diagram = "x3x3x"
    # print(diagram)
    # compute_incidence_matrix(diagram)
    diagram = "x3x5x"
    compute_incidence_matrix(diagram)

    diagram = ["x3x3x","x3x4x","x3x5x","x x3x","x x4x","x x5x"]
    for d in diagram:
        # print_table(*compute_incidence_matrix(diagram))
        print_latex_table(*compute_incidence_matrix(d),d)



