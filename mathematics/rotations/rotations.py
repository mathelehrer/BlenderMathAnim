from numpy import pi, cos, sin

from mathematics.mathematica.mathematica import identity_matrix


def rotation(dim:int, directions:[],angle:float):
    rotation = identity_matrix(dim)
    rotation[directions[0],directions[0]]=cos(angle)
    rotation[directions[1],directions[1]]=cos(angle)
    rotation[directions[0],directions[1]]=-sin(angle)
    rotation[directions[1],directions[0]]=sin(angle)
    return rotation


if __name__ == '__main__':
    print(rotation(5,[0,1],pi/2))