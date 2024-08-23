import numpy as np
from mathutils import Vector, Quaternion


class Embedding:
    def __init__(self,X,Y,Z,dXdu,dXdv,dYdu,dYdv,dZdu,dZdv):
        self.X = X
        self.Y = Y
        self.Z = Z

        self.dXdu = dXdu
        self.dXdv = dXdv
        self.dYdu = dYdu
        self.dYdv = dYdv
        self.dZdu = dZdu
        self.dZdv = dZdv

    def embedding(self,u,v):
        return Vector([self.X(u,v),self.Y(u,v),self.Z(u,v)])

    def unit_u(self,u,v):
        vec = Vector([self.dXdu(u,v),self.dYdu(u,v),self.dZdu(u,v)])
        if vec.length!=0:
            return vec.normalized()
        else:
            return vec

    def unit_v(self,u,v):
        vec = Vector([self.dXdv(u,v),self.dYdv(u,v),self.dZdv(u,v)])
        if vec.length!=0:
            return vec.normalized()
        else:
            return vec

    def unit_n(self,u,v):
        vec = self.unit_u(u,v).cross(self.unit_v(u,v))
        if vec.length!=0:
            return vec.normalized()
        else:
            return vec

    def local_basis(self,u,v):
        '''
        more efficient than calculating each vector independently
        :param u:
        :param v:
        :return:
        '''

        unit_u = self.unit_u(u,v)
        unit_v = self.unit_v(u,v)
        normal = unit_u.cross(unit_v)
        if normal.length!=0:
            unit_n = normal.normalized()
        else:
            unit_n = normal
        return unit_u,unit_v,unit_n

    def rotation_matrix(self,u,v):
        loc = self.local_basis(u,v)
        return np.array([
            [loc[0].x,loc[1].x,loc[2].x],
            [loc[0].y,loc[1].y,loc[2].y],
            [loc[0].z,loc[1].z,loc[2].z],
        ])

    def local_frame_quaternion(self,u,v):
        '''

            algorithm taken from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

            :param u:
            :param v: local coordinates
            :return: the corresponding quaternion
        '''

        m = self.rotation_matrix(u,v)
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0:
            s = 2 * np.sqrt(tr + 1)
            w = s / 4
            x = (m[1][2] - m[2][1]) / s
            y = (m[2][0] - m[0][2]) / s
            z = (m[0][1] - m[1][0]) / s
        else:
            if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
                s = 2 * np.sqrt(1 + m[0][0] - m[1][1] - m[2][2])
                x = s / 4
                y = (m[1][0] + m[0][1]) / s
                z = (m[0][2] + m[2][0]) / s
                w = (m[1][2] - m[2][1]) / s
            elif m[1][1] > m[2][2]:
                s = 2 * np.sqrt(1 + m[1][1] - m[0][0] - m[2][2])
                x = (m[1][0] + m[0][1]) / s
                y = s / 4
                z = (m[1][2] + m[2][1]) / s
                w = (m[2][0] - m[0][2]) / s
            else:
                s = 2 * np.sqrt(1 + m[2][2] - m[0][0] - m[1][1])
                x = (m[2][0] + m[0][2]) / s
                y = (m[1][2] + m[2][1]) / s
                z = s / 4
                w = (m[0][1] - m[1][0]) / s
        return Quaternion([w, -x, -y, -z])


class BoundingBox:
    def __init__(self, *ranges):
        self.ranges = []
        for [minimum,maximum] in ranges:
            if minimum<maximum:
                self.ranges.append([minimum,maximum])
        self.dim = len(self.ranges)

    def __str__(self):
        out = ''
        for i, [minimum, maximum] in enumerate(self.ranges):
            if i > 0:
                out += 'x'
            out += '[' + str(minimum) + ',' + str(maximum) + ']'
        return out

    @property
    def volume(self):
        lengths = []
        for [minimum, maximum] in self.ranges:
            lengths.append(maximum - minimum)
        vol = 1
        for l in lengths:
            if l > 0:
                vol *= l
        return vol

    def overlap(self, other):
        if self.dim != other.dim:
            raise "overlap of bounding box with different dimension cannot be calculated"
        deltas = []
        for i in range(self.dim):
            min1 = self.ranges[i][0]
            max1 = self.ranges[i][1]
            min2 = other.ranges[i][0]
            max2 = other.ranges[i][1]

            if max2 <= min1 or max1 <= min2:
                deltas.append(0)
            elif min1 >= min2 and max1 <= max2:
                deltas.append(max1 - min1)
            elif min1 <= min2 and max1 >= max2:
                deltas.append(max2 - min2)
            elif min1 <= min2 <= max1:
                deltas.append(max1 - min2)
            elif min1 <= max2 <= max2:
                deltas.append(max2 - min1)
            else:
                raise "overlap configuration has not considered all possible situations"

        overlap = 1
        for delta in deltas:
            overlap *= delta
        return overlap / np.maximum(self.volume, other.volume)


if __name__ == '__main__':
    bbox = BoundingBox([-0.5, 0.5], [-2, 2])
    bbox2 = BoundingBox([-1, -0.5], [-2, 2])
    print(bbox.overlap(bbox2))
