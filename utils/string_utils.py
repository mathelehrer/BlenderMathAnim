import os
import re

from mathematics.mathematica.mathematica import partition
from utils.constants import OBJ_DIR


# parse obj data
def get_data_from_obj(obj_file):
    """
    Parses data from an OBJ file and extracts vertices, edges, and faces.

    This function reads an OBJ file, processes its contents to extract
    geometry information such as vertices, edges, and faces. It determines
    edges by identifying connections between vertices specified by the
    triangle/quad definitions within the face data of the OBJ file.

    Parameters:
    obj_file: str
        The name of the OBJ file located in the predefined OBJ_DIR directory.

    Returns:
    tuple[list, list, list]
        A tuple containing:
        - vertices (list): A list of vertex coordinates derived from the OBJ file.
        - edges (list): A list of edges represented as connections between
          vertex indices.
        - faces (list): A list of faces where each face is defined by indices
          of its vertices.

    Raises:
    FileNotFoundError
        If the specified OBJ file cannot be located in the OBJ_DIR directory.

    """
    vertices = []
    faces =[]
    with open(os.path.join(OBJ_DIR, obj_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("v "):
                vertices.append(parse_obj_vertex(line))
            if line.startswith("f "):
                faces.append([f - 1 for f in parse_obj_face(line)])

    edge_set = set()
    for face in faces:
        partitions = partition(face, 2, 1, 1)
        for part in partitions:
            if part[0] > part[1]:
                tup = (part[1], part[0])
            else:
                tup = (part[0], part[1])
        edge_set.add(tup)

    edges = [[e[0], e[1]] for e in edge_set]

    return vertices, edges, faces

def parse_obj_vertex(vertex_str):
    vertex_str = vertex_str.replace("v ", "")
    vertex_str = vertex_str.replace("\n", "")
    return tuple(map(float, vertex_str.split()))

def parse_obj_face(face_str):
    face_str = face_str.replace("f ", "")
    face_str = face_str.replace("\n", "")
    parts = face_str.split()
    indices = []
    for part in parts:
        indices.append(int(part.split("/")[0]))
    return indices

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

def remove_digits(name):
    pattern = r'[0-9]'
    # Match all digits in the string and replace them with an empty string
    return re.sub(pattern, '', name)
    print(new_string)

def remove_punctuation(name):
    pattern = r'[.!?]'
    return re.sub(pattern, '', name)

def parse_vector(vector_str):
    vector_str = vector_str.replace("[", "")
    vector_str = vector_str.replace("]", "")
    coords = vector_str.split(",")
    return [float(coord) for coord in coords]

if __name__ == '__main__':
    print(list(find_all('spam spam spam spam', 'spam')))