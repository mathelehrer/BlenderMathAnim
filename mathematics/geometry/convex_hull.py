from copy import deepcopy

from interface.ibpy import Vector
from mathematics.mathematica.mathematica import mean, find_normal, convex_hull, vector_sum, tuples
from utils.utils import to_vector
import numpy as np

EPS = 1e-6

class ConvexHull:
    def __init__(self, points):
        """
        Create the convex hull for a cube
        >>> a = ConvexHull(tuples([0,1],3))
        >>> print(a.faces)
        {(6, 0, 2): (Vector((0.0, 0.0, -1.0)), 0.0), (6, 4, 0): (Vector((0.0, 0.0, -1.0)), 0.0), (5, 4, 0): (Vector((0.0, -1.0)), 0.0), (5, 1, 0): (Vector((0.0, -1.0)), 0.0), (5, 4, 6): (Vector((1.0, 0.0, 0.0)), 1.0), (5, 6, 7): (Vector((1.0, 0.0, 0.0)), 1.0), (3, 2, 0): (Vector((-1.0, 0.0)), 0.0), (3, 1, 0): (Vector((-1.0, 0.0)), 0.0), (3, 6, 2): (Vector((0.0, 1.0, 0.0)), 1.0), (3, 7, 6): (Vector((0.0, 1.0, 0.0)), 1.0), (3, 1, 5): (Vector((0.0, 0.0, 1.0)), 1.0), (3, 5, 7): (Vector((0.0, 0.0, 1.0)), 1.0)}

        :param points:
        """
        self.ch = convex_hull([Vector(v) for v in points])
        self.points = self.ch.points
        self.center  = mean(self.points)
        self.simplices = self.ch.simplices
        self.faces = {tuple(inds): self.calc_equation(self.points[inds]) for inds in self.ch.simplices}
        self.dim = len(points[0])
        if self.dim==3:
            self.orient_faces()

    def calc_equation(self,points):
        normal = find_normal(points)
        face_center = mean(points)
        if normal.dot(face_center - self.center) > 0:
            return normal, normal.dot(face_center)
        else:
            return -1 * normal, -normal.dot(face_center)

    def orient_faces(self):
        """ for three dimensions the faces are oriented such that the outside of the face is displayed correctly in blender"""
        new_faces={}
        for indices,equation in self.faces.items():
            p0,p1,p2 = self.points[list(indices)]
            normal = Vector(p1-p0).cross(Vector(p2-p0))
            if len(equation[0])==3 and normal.dot(equation[0])<0:
                new_faces[indices[0],indices[2],indices[1]]=deepcopy(self.faces[indices])
            else:
                new_faces[indices]=deepcopy(self.faces[indices])
        self.faces = new_faces


    def is_inside(self,point):
        point =to_vector(point)
        return all([eqn[0].dot(point)-eqn[1]<=0 for eqn in self.faces.values()])

    def ray_cast(self,source,direction=Vector([0,0,1])):
        """
        Calcuates the ray from the source and finds the face and location, where the ray hits the convex hull

        :param source:
        :param direction:
        :return:

        >>> cube = ConvexHull(tuples([0,1],3))
        >>> cube.ray_cast(Vector((0.5,0.5,0.5)))
        {0.5: ((6, 4, 0), Vector((0.5, 0.5, 1.0))), -0.5: ((3, 5, 7), Vector((0.5, 0.5, 0.0)))}

        The result states that two triangular faces of the cube are hit by the ray. One triangle in forward direction and one in backward direction
        algorithm from:
        https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        """
        source = to_vector(source)
        hit_faces={}
        for face,equation in self.faces.items():
            vertices = [Vector(self.points[idx]) for idx in face]

            edge1 = vertices[1]-vertices[0]
            edge2 = vertices[2]-vertices[0]

            h = direction.cross(edge2)
            a = edge1.dot(h)  # negative determinant of the system of equation
            # exclude parallel ray
            if (np.abs(a)>EPS):
                    f = 1/a
                    s = source-vertices[0]
                    u = f*(s.dot(h))
                    if 0<=u<=1:
                        q=s.cross(edge1)
                        v=f*direction.dot(q)
                        if 0<=v<=1-u:
                            t = q.dot(edge2)*f
                            hit_faces[t]=(face,source+t*direction)
        return hit_faces
