# the 600 cell is generated from two quaternions
# omega = (-1/2+1/2i+1/2j+1/2k)
# q = (0+1/2i+1/4*(1+r5)j+1/4*(-1+r5)k
import os
from fractions import Fraction

import numpy as np

from interface.ibpy import Vector
from mathematics.geometry.field_extensions import FQuaternion, QR5, FTensor, EpsilonTensor, FVector
from mathematics.zeros import chop
from utils.constants import DATA_DIR


# for the generation of the finite group, it is best to work with
# exact values for r5 = 5**0.5
# therefore, the field extension QR5 is implemented

PATH = "mathematics/geometry/data/"

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
        >>> export_csv_data(dir=PATH,filename="poly600cell_test.csv")
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
    export_csv_data(dir=PATH,filename="poly600cell_test.csv")