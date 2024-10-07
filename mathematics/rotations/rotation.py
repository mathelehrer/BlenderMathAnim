from numpy import pi, cos, sin

from mathematics.mathematica.mathematica import identity_matrix

def rotation(dim,direction=[0,1],angle=pi/2):
    rot = identity_matrix(dim)
    rot[direction[0],direction[0]]=cos(angle)
    rot[direction[0],direction[1]]=-sin(angle)
    rot[direction[1],direction[1]]=cos(angle)
    rot[direction[1],direction[0]]=sin(angle)
    return rot

if __name__ == '__main__':
    print(rotation(5,[2,4]))