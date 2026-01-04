import os
from collections import defaultdict

from mathematics.geometry.coxH4 import CoxH4
from mathematics.geometry.field_extensions import FVector
from utils.constants import DATA_DIR

def header_from_labels(label_list):
    line = ""
    for i in range(len(label_list)):
        if i>0:
            line+=","
        line += label_list[i]
    return line+"\n"

def read_start_vector(signature,grp_string="CoxH4"):
    with open(os.path.join(os.path.join(DATA_DIR,"../../mathematics/geometry/data"),grp_string+"_points"+str(signature).replace(",","_")+".dat"), "r") as f:
        lines = f.readlines()
        print("start vector: ",lines[0][:-1])
        return lines[0][:-1]

def find_cells_of_face(face,cell_vertex_map):
    cells_of_face = []
    for key,val in cell_vertex_map.items():
        if set(face).issubset(val):
            cells_of_face.append(key)
            cells_of_face.append(len(val))

    return cells_of_face

def create_csv_file_from_polytope_data(signature,group=CoxH4):
    grp = group(path="../mathematics/geometry/data")

    start = FVector.parse(read_start_vector(signature,grp_string=grp.name))
    vertices = grp.point_cloud(signature, start)
    cells = grp.get_cells(signature)

    vertex_cell_map=defaultdict(list)
    for i,(indices,normals) in enumerate(cells.items()):
        for index in indices:
            vertex_cell_map[index].append(i)

    cell_vertex_map = defaultdict(list)
    for cell_idx,cell in enumerate(cells.keys()):
        cell_vertex_map[cell_idx] = set(cell)

    cells_per_vertex = len(vertex_cell_map[0])
    print(cells_per_vertex," cells per vertex")

    faces = grp.get_faces(signature)
    vertex_face_map=defaultdict(list)
    for face_idx,face in enumerate(faces):
        for v_idx in face:
            vertex_face_map[v_idx].append(face_idx)

    faces_per_vertex = len(vertex_face_map[0])
    print(faces_per_vertex," faces per vertex")

    labels = ["x","y","z","w"]
    for i in range(cells_per_vertex):
        labels.append("cell"+str(i))
    for i in range(faces_per_vertex):
        labels.append("face"+str(i))

    filename = "vertex_topology_"+str(signature).replace(", ","_")+".csv"
    with open(os.path.join(DATA_DIR, filename), "w") as f:
        print(f"Write vertex data to file {filename}... ")
        f.write(header_from_labels(labels))
        for i,vertex in enumerate(vertices):
            vertex = vertex.real()
            line = str(vertex[0]) + "," + str(vertex[1]) + ", " + str(vertex[2]) + ", " + str(vertex[3])
            for j in vertex_cell_map[i]:
                line+=", "+str(j)
            for j in vertex_face_map[i]:
                line+=","+str(j)
            line+= "\n"
            f.write(line)
        print("done!")

    face_vertex_map=defaultdict(list)
    largest_face = 0
    for face_idx,face in enumerate(faces):
        face_vertex_map[face_idx]=face
        if len(face)>largest_face:
            largest_face=len(face)

    labels = ["n","cell0","cell_size0","cell1","cell_size1"]
    for i in range(largest_face):
        labels.append("vertex"+str(i))

    filename="face_topology_"+str(signature).replace(", ","_")+".csv"
    with open(os.path.join(DATA_DIR, filename), "w") as f:
        print(f"Write face data to file {filename}")
        f.write(header_from_labels(labels))
        for index,face in enumerate(faces):
            line = str(len(face))
            cells_of_face = find_cells_of_face(face,cell_vertex_map)
            for cell in cells_of_face:
                line+=","+str(cell)
            for i in range(largest_face):
                if i<len(face):
                    line+=(","+str(face[i]))
                else:
                    line+=","
            line+="\n"
            f.write(line)
        print("done!")


if __name__ == '__main__':
    create_csv_file_from_polytope_data([1,-1,1,-1])