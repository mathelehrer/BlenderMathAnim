import os

from matplotlib import pyplot as plt

from mathematics.geometry.coxH3 import CoxH3
from mathematics.geometry.field_extensions import FVector


def read_signatures(filename):
    with open(filename, "r") as f:
        data = []
        lines = f.readlines()
        for line in lines:
            # remove new line
            if line[-1] == "\n":
                line = line[0:-1]
            parts = line.split("->")
            signature = eval(parts[0])
            # print(signature)
            # print(parts[1])
            p0 = FVector.parse(parts[1])
            data.append((signature, p0))
    return data

def append_to_good(text, filename):
    with open(os.path.join(filename), "a") as f:
        f.write(str(text) + "\n")


def generate_data():
    """
    this function computes the data for all solids that belong to that symmetry group
    It requires the pre-computation of a signature file as it is done, eg in video_CoxH4/mathematica/coxH3.nb
    """
    t0 = 0
    path = "data"
    group = CoxH3(path=path)

    data = read_signatures(os.path.join(path,"CoxH3_signatures.dat"))
    for signature, start in data:
        vertices = group.point_cloud(signature=signature,start = start)
        vertices =[v.real() for v in vertices]
        edges = group.get_edges(signature)
        print(len(vertices), len(edges))

        # create 3d plot from vertices and edges using matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract x, y, z (ignore w for 3D projection)
        xs = [v[0] for v in vertices]  # Assuming vertices are lists/arrays; adjust if using custom Vector class
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]

        # Plot vertices as scatter points
        ax.scatter(xs, ys, zs, color='blue', s=20)  # Adjust color/size as needed

        # Plot edges as lines
        for i, j in edges:
            ax.plot([vertices[i][0], vertices[j][0]],
                    [vertices[i][1], vertices[j][1]],
                    [vertices[i][2], vertices[j][2]], color='black')

        # Set labels and limits (optional, tune based on data)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([min(xs), max(xs)])
        ax.set_ylim([min(ys), max(ys)])
        ax.set_zlim([min(zs), max(zs)])

        plt.show()

    # the plots reveal the following dictionary
    """
    0->dodecahedron
    1->rhombicosidodecahedron
    2->truncated icosidodecahedron
    3->icosahedron
    4->icosahedron
    5->truncated icosahedron (soccer ball)
    6->rhombicosidodecahedron
    7->dodecahedron
    8->
    9->icosahedron
    10->dodecahedron
    11->truncated dodecahedron
    12->icosidodecahedron
    """

if __name__ == '__main__':
    generate_data()