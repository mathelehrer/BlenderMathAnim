import math
import os
from collections import OrderedDict
from functools import partial

import numpy as np
from anytree import NodeMixin, RenderTree
from matplotlib.pyplot import subplots

from interface import ibpy
from interface.ibpy import set_hdri_background, set_hdri_strength, Vector, Quaternion, set_exposure, \
    set_camera_view_to, value_of_action_fcurve_at_frame
from mathematics.groups.element import Element
from mathematics.groups.group import Group
from objects.annulus import Annulus, Annulus2, Disc2
from objects.bmatrix import BMatrix
from objects.bobject import AnimBObject, BObject
from objects.circle import BezierCircle, Circle2, CircleArc
from objects.codeparser import CodeParser
from objects.container import Container
from objects.coordinate_system import CoordinateSystem
from objects.curve import Curve, DataCurve, DataCurveRefined, BezierDataCurve
from objects.cylinder import Cylinder
from objects.derived_objects.p_arrow import PArrow
from objects.derived_objects.pencil import Pencil
from objects.derived_objects.person_with_cape import PersonWithCape
from objects.derived_objects.pin import Pin
from objects.digital_number import DigitalNumber
from objects.disc import TransformedDisc
from objects.display import Display, CodeDisplay
from objects.empties import EmptyCube
from objects.free_hand import FreeHandText
from objects.function import Function, MeshFunction
from objects.geometry.sphere import MultiSphere, invert, Sphere, HalfSphere
from objects.grid import DefaultGrid
from objects.number_line import NumberLine
from objects.plane import Plane
from objects.polygon import Triangle
from objects.polyhedron import Polyhedron
from objects.rope import Rope
from objects.solid_of_revolution import RevolutionSolid
from objects.tex_bobject import SimpleTexBObject, MultiLineTexBObject
from objects.tree.tree import Tree
from perform.scene import Scene
from utils import utils
from utils.constants import FRAME_RATE, DEFAULT_ANIMATION_TIME, LOC_FILE_DIR
from utils.geometry import Embedding
from utils.mathematics import interpol, VectorTransformation, regularized, Koch
from utils.utils import print_time_report, z2vec, flatten, lin_map
# some functions created on demand
from utils.utils_io import read_data, read_complex_data
from video_apollonian.indras_utils.indra_generating_algorithms import BreadFirstSearch, \
    BlenderModel, DepthFirstSearchByLevel, ApollonianModel, DepthFirstSearchWithFixedPoints, ThetaModel, GrandMasRecipe, \
    SchottkyFamily, TransformedModel
from video_apollonian.indras_utils.circle import IndraCircle
from video_apollonian.indras_utils.mymath import moebius_on_circle, moebius_on_point, fixed_point_of
from video_bwm.scene_bwm import z2loc


class TreeNode(NodeMixin):
    """
     Just a container that is a tree node and can contain
    """

    def __init__(self, content, bob, connection=None, color='text', u=0, v=0):
        """
        :param content:
        :param bob:
        :param connection:
        :param color:
        :param u:
        :param v:
        """
        self.content = content
        self.bob = bob
        self.connection = connection
        self.color = color
        self.u = u
        self.v = v

    def connect(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.connection.grow(begin_time=begin_time, transition_time=transition_time)



def get_brightness(leave):
    brighter = 0.15 / leave.circle.r
    if leave.color == 'custom1':
        brighter *= 2
    if leave.color == 'custom2':
        brighter *= 0.5
    if leave.color == 'text':
        brighter *= 0.5
    return brighter


def kleinian1(face,begin_time,duration):

    disc_counter = 0
    fn = face.normal
    q = get_quaternion(face)
    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)

    # introduce small shift towards vertex_0
    shift = 0.05 * (face.get_location_of_vertex(0) - face.center)

    max_level = 5
    n_discs = 2 * (pow(3, max_level + 1) - 1)
    dt = duration / n_discs
    t0 = begin_time
    for level in range(2, max_level + 1):
        dfs = DepthFirstSearchByLevel(BlenderModel, max_level=level)
        dfs.generate_tree()
        leaves = dfs.get_leaves()

        scale = 0.2
        for leave in leaves:
            circle = leave.circle
            center = np.real(circle.c) * loc_x + np.imag(circle.c) * loc_y
            center *= scale
            brighter = get_brightness(leave)
            disc = disc_from_circle(circle, location=center + face.center + fn * 0.05 * 1.01 * (level-1) + shift,
                                    color=leave.color, name=leave.word, rotation_quaternion=q,
                                    brighter=0.1 / leave.circle.r, height=0.05,
                                    scale=[scale, scale, 1],
                                    transmission=1 - np.minimum(1, 0.01 * brighter),
                                    roughness=0.01 * brighter)
            ibpy.set_parent(disc,face)
            t0=disc.appear(begin_time=t0, transition_time=dt)
            disc_counter+=1


def get_quaternion(face):
    """
    calculate the rotation quaternion from the face normal
    the quaternion has to rotate the z-unit vector into the face normal

    :param face:
    :return:
    """
    z_unit = Vector([0, 0, 1])
    fn = face.normal.normalized()
    cos = fn.dot(z_unit)
    axis = -fn.cross(z_unit).normalized()
    return Quaternion(axis, np.arccos(cos))


def kleinian2(face,begin_time,duration):
    t0=begin_time
    disc_counter = 0
    fn = face.normal

    # create local coordinates
    loc_x = (face.get_location_of_vertex(0)-face.center).normalized()
    loc_y = fn.cross(loc_x)

    # introduce small shift towards vertex_3
    shift = 0.675 * (face.vertices[1] - face.center)+ 0.25 * (face.vertices[4] - face.center)

    all_points = []
    all_curvatures = []
    max_curvature = 200
    # create cylinders up to a given resolution
    N = 100
    # import the data of centers and curvatures for a unit cell of the strip of circles
    points = read_data("centers.dat")
    curvatures = read_data("curvatures.dat")
    scale =5

    # calculate circles under inversion
    for center, curvature in zip(points, curvatures):
        for i in set(range(-N, N)):
            M, R = invert(center + i, 1 / curvature)
            if 1 / R < max_curvature:
                all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                all_curvatures.append(1 / R / scale)
                if np.imag(center) != 0.5:
                    M, R = invert(np.real(center) + i + 1j - np.imag(center) * 1j, 1 / curvature)
                    if 1 / R < max_curvature:
                        all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                        all_curvatures.append(1 / R / scale)

    # at center sphere
    M, R = invert(points[0], 1 / curvatures[0])

    all_points.append((M + 1j) * scale)
    all_curvatures.append(1 / R / scale)

    # make cylinders appear depending on the position of the spider
    dt = duration/len(all_points)
    # remove first
    all_points.pop(0)
    all_curvatures.pop(0)
    for point, curvature in zip(all_points, all_curvatures):
        start = Vector(np.real(point)*loc_x+np.imag(point)*loc_y)
        pos = face.center+shift
        cyl = Cylinder.from_start_to_end(start=pos+start,
                                         end=pos+start + 0.1*fn * np.sqrt(curvature),
                                         thickness=10 / curvature,
                                         color='text',
                                         metallic=0.5,
                                         roughness=0,
                                         ior=1.5,
                                         smooth=False,
                                         transmission=0.25,
                                         vertices=int(4 * np.sqrt(max_curvature / curvature)))
        t0=cyl.grow(begin_time=t0, transition_time=dt)
        ibpy.set_parent(cyl,face)
        disc_counter+=1


def kleinian3(face,begin_time,duration):
    q = get_quaternion(face)
    disc_counter=0
    fn = face.normal
    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)
    #introduce small shift towards the edge with vertices 8 and 9
    shift = -0.2*(face.vertices[3]-face.center)-0.075*(face.vertices[9]-face.center)

    max_level = 4
    n_discs  = 2*(pow(3,max_level+1)-1)
    dt = duration/n_discs
    t0=begin_time
    for level in range(2, max_level+1):
        dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=level)
        dfs.generate_tree()
        leaves = dfs.get_leaves()
        brighter = 2 * (level - 1)
        scale = 1.25
        for leave in leaves:
            circle = leave.circle

            center = np.real(circle.c) * loc_x + np.imag(circle.c) * loc_y
            center *= scale
            disc = disc_from_circle(circle, location=center + face.center + fn * 0.05 * 1.01 * (level - 1) + shift,
                                    color=leave.color, name=leave.word, rotation_quaternion=q,
                                    brighter=0.1 / leave.circle.r, height=0.05,
                                    scale=[scale, scale, 1],
                                    transmission=1 - np.minimum(1, 0.01 * brighter),
                                    roughness=0.01 * brighter)
            ibpy.set_parent(disc,face)
            t0=disc.appear(begin_time=t0, transition_time=dt)
            disc_counter+=1


def kleinian4(face,begin_time,duration):
    q = get_quaternion(face)
    disc_counter=0
    fn = face.normal
    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)

    #introduce small shift towards the edge with vertices 8 and 9
    shift = 0.06*(face.vertices[1]-face.center)

    max_level = 5
    n_discs  = 2*(pow(3,max_level+1)-1)
    dt = duration/n_discs
    t0=begin_time
    for level in range(1, max_level+1):
        dfs = DepthFirstSearchByLevel(ThetaModel, max_level=level,theta=0.9*np.pi/4)
        dfs.generate_tree()
        leaves = dfs.get_leaves()

        scale = 1.25
        for leave in leaves:
            circle = leave.circle
            center = np.real(circle.c) * loc_x + np.imag(circle.c) * loc_y
            center *= scale
            brighter = get_brightness(leave)
            disc = disc_from_circle(circle, location=center + face.center + fn * 0.05 * 1.01 * (level) + shift,
                                    color=leave.color, name=leave.word, rotation_quaternion=q,
                                    brighter=0.1 / leave.circle.r, height=0.05,
                                    scale=[scale, scale, 1],
                                    transmission=1 - np.minimum(1, 0.01 * brighter),
                                    roughness=0.01 * brighter)
            ibpy.set_parent(disc,face)
            t0=disc.appear(begin_time=t0, transition_time=dt)
            disc_counter+=1


def kleinian5(face, begin_time, duration):
    q = get_quaternion(face)
    disc_counter = 0
    fn = face.normal
    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)

    # introduce small shift towards the edge with vertices 8 and 9
    shift = 0.0 * (0.5 * (face.vertices[8] + face.vertices[9]) - face.center)

    max_level = 5
    n_discs = 2 * (pow(3, max_level + 1) - 1)
    dt = duration / n_discs
    t0 = begin_time
    for level in range(2, max_level + 1):
        dfs = DepthFirstSearchByLevel(ThetaModel, max_level=level, theta=np.pi / 4)
        dfs.generate_tree()
        leaves = dfs.get_leaves()
        brighter = 2 * (level - 1)
        scale = 1.75
        for leave in leaves:
            circle = leave.circle

            center = np.real(circle.c)*loc_x+np.imag(circle.c)*loc_y
            center*=scale
            disc = disc_from_circle(circle, location=center + face.center + fn * 0.05 * 1.01 * (level - 1) + shift,
                                    color=leave.color, name=leave.word, rotation_quaternion=q,
                                    brighter=0.1/leave.circle.r, height=0.05,
                                    scale=[scale, scale, 1],
                                    transmission=1 - np.minimum(1, 0.01 * brighter),
                                    roughness=0.01 * brighter)
            ibpy.set_parent(disc,face)
            t0=disc.appear(begin_time=t0, transition_time=dt)
            disc_counter+=1


def disc_from_circle(circle, color, height=0.1,alpha=1,brighter=0,name="Disc",**kwargs):
    n = np.sqrt(circle.r)
    if math.isnan(n) or math.isinf(n):
        n=1000
    resolution = [2, np.minimum(500, np.maximum(10, int(80 * n)))]

    if color == 'example' or 'gray' in color:
        bright_factor = 1.5
    else:
        bright_factor = 3

    disc = Disc2(r=circle.r,
                 resolution=resolution, solid=height, color=color, name=name, alpha=alpha,
                 brighter=bright_factor * brighter,**kwargs)
    return disc


def kleinian6(face,begin_time,duration):
    q = get_quaternion(face)
    fn = face.normal

    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)

    t0=0
    count=0
    discs = 431
    shift=0.2*(face.vertices[9]-face.center)+0.1*(face.vertices[15]-face.center)
    scale=1.2

    for level in range(1, 30):
        dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=level)
        if level == 1:
            dfs.generate_tree()
            leaves = dfs.get_leaves()
        else:
            leaves = dfs.generate_next_level(level, leaves)
        leaves = list(filter(lambda x: x.circle.r > 0.03, leaves))
        if level > 1:
            if len(leaves) > 0:
                dt = duration / discs
                for leave in leaves:
                    circle=leave.circle
                    center = np.real(circle.c)*loc_x+np.imag(circle.c)*loc_y
                    center*=scale
                    brighter=get_brightness(leave)
                    disc = disc_from_circle(circle,location=center+face.center+fn*0.05*1.01*(level-2)+shift,
                                                 color=leave.color, name=leave.word,rotation_quaternion=q,
                                                  brighter=0.1 / leave.circle.r,
                                            height=0.05,
                                            scale=[scale,scale,1],
                                            transmission = 1 - np.minimum(1, 0.02 * brighter),
                                            roughness = np.minimum(0.5, 0.02 * brighter)
                                            )
                    disc.appear(begin_time=begin_time+t0, transition_time=dt)
                    ibpy.set_parent(disc, face)
                    t0 += dt
                    count+=1
            else:
                print("last level: ", level)
                break
    if discs!=count:
        raise "Number of discs: "+str(count)+" update dt calculation for precise duration."


def kleinian7(face,begin_time,duration):
    q = get_quaternion(face)
    fn = face.normal

    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)
    max = 6
    min = 2
    discs = 0

    for c in range(min, max):
        discs += 4 * pow(3, c - 1)

    t0=begin_time
    tb = 2.501
    ta = 2.001
    shift = -0.2*(face.vertices[6]-face.center)+ 0.05*(face.vertices[4]-face.center)
    scale=1.35

    for level in range(min, max):
        dfs = DepthFirstSearchByLevel(GrandMasRecipe, ta=ta, tb=tb, max_level=level)
        dfs.generate_tree()
        leaves = dfs.get_leaves()
        dt = duration / discs
        for leave in leaves:
            circle = leave.circle
            center = np.real(circle.c) * loc_x + np.imag(circle.c) * loc_y
            center *= scale
            brighter = 2*(level-1)
            disc = disc_from_circle(circle, location=center + face.center + fn * 0.05 * 1.01 * (level - 2) + shift,
                                    color=leave.color, name=leave.word, rotation_quaternion=q,
                                    offset=level, brighter=0.1 / leave.circle.r, height=0.05,
                                    scale=[scale,scale,1],  transmission=1-0.01*brighter,
                                    roughness=0.01*brighter)
            t0 = disc.appear(begin_time=t0,
                             transition_time=0) + dt  # disc has to appear instantly, otherwise alpha will destroy transparency
            ibpy.set_parent(disc,face)


def kleinian8(face,begin_time,duration):
    q = get_quaternion(face)
    fn = face.normal

    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)

    t0=0
    count=0
    discs = 906
    shift=0.05*(face.vertices[5]-face.center)
    scale=0.25

    for level in range(1, 30):
        dfs = DepthFirstSearchByLevel(BlenderModel, max_level=level)
        if level == 1:
            dfs.generate_tree()
            leaves = dfs.get_leaves()
        else:
            leaves = dfs.generate_next_level(level, leaves)
        leaves = list(filter(lambda x: x.circle.r > 0.05, leaves))
        if level > 2:
            if len(leaves) > 0:
                dt = duration / discs
                for leave in leaves:
                    circle=leave.circle
                    center = np.real(circle.c)*loc_x+np.imag(circle.c)*loc_y
                    center*=scale
                    brighter =get_brightness(leave)
                    disc = disc_from_circle(circle,location=center+face.center+fn*0.05*1.01*(level-2)+shift,
                                                 color=leave.color, name=leave.word,rotation_quaternion=q,
                                                  brighter=brighter, height=0.05,
                                            scale=[scale,scale,1],
                                            transmission=1 - np.minimum(1, 0.01 * brighter),
                                            roughness=np.minimum(0.5, 0.01 * brighter)
                    )
                    disc.appear(begin_time=begin_time+t0, transition_time=dt)
                    ibpy.set_parent(disc, face)
                    t0 += dt
                    count+=1
            else:
                print("last level: ", level)
                break
    if discs!=count:
        raise "Number of discs: "+str(count)+" update dt calculation for precise duration."


def kleinian9(face,begin_time,duration):
    q = get_quaternion(face)
    fn = face.normal

    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)

    t0=0
    count=0
    discs = 532
    shift=Vector()#0.05*(face.vertices[5]-face.center)
    scale=1#0.25
    max_level=30

    for level in range(1, max_level):
        dfs = DepthFirstSearchByLevel(SchottkyFamily,k=1,y=1, max_level=level)
        if level == 1:
            dfs.generate_tree()
            leaves = dfs.get_leaves()
        else:
            leaves = dfs.generate_next_level(level, leaves)
        leaves = list(filter(lambda x: x.circle.r > 0.01, leaves))
        if level > 0:
            if len(leaves) > 0:
                dt = duration / discs
                for leave in leaves:
                    circle=leave.circle
                    center = np.real(circle.c)*loc_x+np.imag(circle.c)*loc_y
                    center*=scale
                    brighter = get_brightness(leave)
                    disc = disc_from_circle(circle,location=center+face.center+fn*0.05*1.01*(level)+shift,
                                                 color=leave.color, name=leave.word,rotation_quaternion=q,
                                                  brighter=brighter, height=0.05,scale=[scale,scale,1],
                                            transmission=1-np.minimum(1,0.01*brighter),
                                            roughness=np.minimum(0.5,0.01*brighter))
                    disc.appear(begin_time=begin_time+t0, transition_time=dt)
                    ibpy.set_parent(disc, face)
                    t0 += dt
                    count+=1
            else:
                print("last level: ", level)
                break
    if discs!=count:
        print("Number of discs: "+str(count)+" update dt calculation for precise duration.")


def kleinian10(face,begin_time,duration):
    q = get_quaternion(face)
    fn = face.normal

    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)
    max = 6
    min = 3
    discs = 0

    for c in range(min, max):
        discs += 4 * pow(3, c - 1)

    t0=begin_time

    shift = 0.05*(face.vertices[8]-face.center)
    scale=2.2
    height=0.1

    for level in range(min, max):
        dfs = DepthFirstSearchByLevel(SchottkyFamily, y=0.5,k=0.1, max_level=level)
        dfs.generate_tree()
        leaves = dfs.get_leaves()
        dt = duration / discs
        for leave in leaves:
            circle = leave.circle
            center = np.real(circle.c) * loc_x + np.imag(circle.c) * loc_y
            center *= scale
            brighter = 2*(level-2)
            disc = disc_from_circle(circle, location=center + face.center + fn * height * 1.01 * (level-2) + shift,
                                    color=leave.color, name=leave.word, rotation_quaternion=q,
                                    brighter=0.1 / leave.circle.r, height=height,
                                    scale=[scale,scale,1],
                                    transmission=1-0.01*brighter,
                                    roughness=0.01*brighter)
            t0 = disc.appear(begin_time=t0,
                             transition_time=0) + dt  # disc has to appear instantly, otherwise alpha will destroy transparency
            ibpy.set_parent(disc,face)


def kleinian11(face,begin_time,duration):
    q = get_quaternion(face)
    fn = face.normal

    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)
    max = 7
    min = 3
    discs = 0
    counter = 0
    discs = 696

    t0=begin_time

    shift = 0.3*(face.vertices[12]-face.center)
    scale=0.75
    height=0.1
    t = np.array([[1,-0.99j],[1,0.99j]])
    for level in range(min, max):
        dfs = DepthFirstSearchByLevel(TransformedModel,transformation=t,model=ApollonianModel, max_level=level)
        dfs.generate_tree()
        leaves = dfs.get_leaves()
        dt = duration / discs
        for leave in leaves:
            circle = leave.circle
            center = np.real(circle.c) * loc_x + np.imag(circle.c) * loc_y
            center *= scale
            brighter = 2*(level-2)
            if not ('bb' in leave.word or 'BB' in leave.word):
                disc = disc_from_circle(circle, location=center + face.center + fn * height * 1.01 * (level - 2) + shift,
                                        color=leave.color, name=leave.word, rotation_quaternion=q,
                                        brighter=0.1 / leave.circle.r, height=height,
                                        scale=[scale,scale,1],  transmission=1-0.01*brighter,
                                        roughness=0.01*brighter)
                t0 = disc.appear(begin_time=t0,
                                 transition_time=0) + dt  # disc has to appear instantly, otherwise alpha will destroy transparency
                ibpy.set_parent(disc,face)
                counter+=1

    print("number of discs in kleinian 11: ",counter)


def kleinian12(face,begin_time,duration):
    q = get_quaternion(face)
    fn = face.normal

    loc_x = (face.get_location_of_vertex(0) - face.center).normalized()
    loc_y = fn.cross(loc_x)

    t0=0
    count=0
    discs = 402
    shift= 0.2*(face.vertices[13]-face.center)
    scale=0.75
    t = np.array([[1, -0.99j], [1, 0.99j]])
    for level in range(1, 30):
        dfs = DepthFirstSearchByLevel(TransformedModel,transformation=t,model=ApollonianModel, max_level=level)
        if level == 1:
            dfs.generate_tree()
            leaves = dfs.get_leaves()
        else:
            leaves = dfs.generate_next_level(level, leaves)
        leaves = list(filter(lambda x: x.circle.r > 0.05, leaves))
        leaves =list(filter(lambda x: 'bb' not in x.word,leaves))
        leaves =list(filter(lambda x: 'BB' not in x.word,leaves))
        if level > 2:
            if len(leaves) > 0:
                dt = duration / discs
                for leave in leaves:
                    circle=leave.circle
                    center = np.real(circle.c)*loc_x+np.imag(circle.c)*loc_y
                    center*=scale
                    brighter=get_brightness(leave)
                    disc = disc_from_circle(circle,location=center+face.center+fn*0.05*1.01*(level-2)+shift,
                                                 color=leave.color, name=leave.word,rotation_quaternion=q,
                                                  brighter=brighter, height=0.05,scale=[scale,scale,1],
                                                    transmission = 1 - np.minimum(1, 0.01 * brighter),
                                                    roughness = np.minimum(0.5, 0.01 * brighter)
                    )
                    disc.appear(begin_time=begin_time+t0, transition_time=dt)
                    ibpy.set_parent(disc, face)
                    t0 += dt
                    count+=1
            else:
                print("last level: ", level)
                break
    if discs!=count:
        raise "Number of discs: "+str(count)+" update dt calculation for precise duration."



def get_group(label):
    if label is None:
        r2 = np.sqrt(2)
        a = Element(np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'a')
        c = Element(np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]]), 'c')
        b = Element(-0.5 * np.matrix([[-1, r2, 1], [r2, 0, r2], [1, r2, -1]]), 'b')
        group = Group(a, b, c)
        return group
    else:
        return Group.from_label(label)


def color_function(expr):
    colors = []
    expr = expr.strip()
    for c in expr:
        if c == '1':
            colors.append('text')
        elif c == 'a':
            colors.append('joker')
        elif c == 'b':
            colors.append('important')
    return colors


def k_of_s(s):
    '''
    Condition for having Shottky discs of equal size within a Shottky pair

    :param s:
    :return:
    '''
    return (1 + s - 2 * np.sqrt(s)) / (s - 1)


def a(v):
    z = v.x + 1j * v.y
    w = (2.999 * z + 1) / (z + 3)
    return Vector([np.real(w), np.imag(w), 0])


def moebius_vec(a, v):
    z = v.x + 1j * v.y
    w = (a[0][0] * z + a[0][1]) / (a[1][0] * z + a[1][1])
    return Vector([np.real(w), np.imag(w), 0])


def moebius(a, w):
    return (a[0][0] * w + a[0][1]) / (a[1][0] * w + a[1][1])


def riemann(v):
    return riemann_general(v, 1 / 2)


def riemann_general(v, r):
    '''
    projection to the riemann sphere with radius 1/2 and center at (0,0,0.5)
    :param x:
    :param y:
    :return:
    '''
    d = 1 + (v.x ** 2 + v.y ** 2) / 4 / r ** 2
    X = v.x / d
    Y = v.y / d
    Z = 2 * r * (d - 1) / d
    return Vector([X, Y, Z])


def inv_riemann(v):
    return riemann_general_inv(v, 1 / 2)


def riemann_general_inv(v, r, eps=0.001):
    '''
    back projection from the riemann sphere with radius 1/2 and center at (0,0,0.5)
    :param v:
    :return:
    '''

    denominator = (1 - v.z / 2 / r)
    if denominator == 0:
        denominator = eps
    x = v.x / denominator
    y = v.y / denominator

    return Vector([x, y, 0])


def rotate(v):
    return Vector([v.z - 0.5, v.y, -v.x + 0.5])


def rotate2(v, r):
    return Vector([v.x, v.z - r, -v.y + r])


def output(tree):
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.content))


def previous_letter(letter):
    '''
    cyclic counting of the letters a,b,A,B

    :param letter:
    :return:
    '''
    if letter == 'a':
        return 'B'
    if letter == 'b':
        return 'a'
    if letter == 'A':
        return 'b'
    if letter == 'B':
        return 'A'


def next_letter(letter):
    '''
    cyclic counting of the letters a,b,A,B

    :param letter:
    :return:
    '''
    if letter == 'a':
        return 'b'
    if letter == 'b':
        return 'A'
    if letter == 'A':
        return 'B'
    if letter == 'B':
        return 'a'


def revolution(x):
    if x == 1:
        return 1.5
    else:
        return (np.sqrt(x) - x ** 2) / (1 - x)


def d_revolution(x):
    r = np.sqrt(x)
    return 1 / (1 + 1 / (r + 2 * x + x * r))


def embedding(x, phi):
    r = revolution(x)
    return Vector([r * np.cos(phi), r * np.sin(phi), -x])


def local_basis(x, phi):
    f = revolution(x)
    df = d_revolution(x)
    sin = np.sin(phi)
    cos = np.cos(phi)

    X = Vector([df * cos, df * sin, -1])
    Y = Vector([-f * sin, f * cos, 0])
    return -X.normalized(), Y.normalized()  # minus sign makes the radial vector pointing to the tip


def rotation_matrix(x, phi):
    b = local_basis(x, phi)
    n = normal(x, phi)
    return [[b[1].x, b[0].x, n.x], [b[1].y, b[0].y, n.y], [b[1].z, b[0].z, n.z]]


def determinant(m):
    return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] + m[0][2] * m[1][0] * m[2][1] - m[0][0] * m[1][2] * \
           m[2][1] - m[0][1] * m[1][0] * m[2][2] - m[0][2] * m[1][1] * m[2][0]


def normal(x, phi):
    df = d_revolution(x)
    n = Vector([np.cos(phi), np.sin(phi), df])
    return n.normalized()


def local_basis_quaternion(x, phi):
    '''
    x -> dphi
    y -> dx
    z -> n

    m =[dphi,dx,n] is the matrix of rotation, it converts the standard basis into the local basis

    algorithm taken from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/


    :param x:
    :param phi:
    :return: the corresponding quaternion
    '''

    m = rotation_matrix(x, phi)

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


def resolution_function(curvature, scale=1):
    curv = curvature * scale
    if curv < 10:
        res = 4
    elif curv < 100:
        res = 3
    else:
        res = 2
    return res


def matrix_print(m):
    for r in m:
        out = ''
        for c in r:
            out += str(np.round(c * 100) / 100) + " "
        print(out)


def transform(z):
    '''
    simple scaling, translation and rotation
    :param z:
    :return:
    '''
    return np.conj(1j * z / 2 - 1j / 2)


def piecewise_function_from_point_data(lst, t, scale=1):
    '''
    convert a list of point lists into a piecewise function
    :param lst:
    :param t:
    :param scale:
    :return:
    '''
    pass


def function_from_complex_list(lst, t, scale=1, average=1, shift=[0, 0, 0]):
    '''
    convert the values in a list into a function

    :param lst: the sequence of complex numbers
    :param t: the parameter ranging from 0 to 1
    :return:
    '''

    n = len(lst)
    i = np.round(t * n)
    i %= n
    if average == 1:
        val = transform(lst[int(i)]) * scale
    else:
        val = 0
        for j in range(average):
            val += transform(lst[int((i + j) % n)])
        val /= average
        val *= scale
    return z2vec(val) + utils.to_vector(shift)


def fill_transforms(R, k, m, c_unit, coords, t0, label='Disc', remove_tic=None, disc_collector=[]):
    r1 = 2 * R * 1 / k * c_unit
    r2 = 2 * R * k * c_unit
    transformations = [
        lambda w: moebius_vec(m, w),
    ]
    for i in range(1, 5):

        r = 1 / 2 * np.abs(np.abs(r1) - np.abs(r2))
        center = 1 / 2 * (r1 + r2)
        Y = Vector([0, 1, 0])
        disc_next = Disc2(r=r, center=Vector([np.real(center), np.imag(center), 0]), resolution=[int(50 / i), 50],
                          rotation_euler=[np.pi / 2, 0, 0],
                          offset=0, bevel=0.1, solid=0.1, smooth=2,
                          location=0 * Y, scale=[1, 1, 1], transformations=transformations,
                          name=label + str(i))
        disc_collector.append(disc_next)
        disc_next.appear(begin_time=t0, transition_time=0)
        coords.add_object(disc_next)
        disc_next.next_shape(begin_time=t0)
        disc_next.move(direction=-0.1 * i * Y, begin_time=t0, transition_time=0)
        if remove_tic:
            coords.axes[remove_tic[0]].tics[remove_tic[1]].disappear(begin_time=t0)
        r1 = moebius(m, r1)
        r2 = moebius(m, r2)
        t0 += 1.5
    return t0


def hull2(x, amp=10, sigma=10):
    return amp * np.exp(-x * x / sigma)


def d_hull2(x, amp=10, sigma=10):
    return -2 * amp * x / sigma * np.exp(-x * x / sigma)


def hull(x, radius=1, arc=0.99 * np.pi / 2):
    '''
    create a circular arc with radius and arc and continue with a straight line afterwards
    :param x:
    :param radius:
    :return:
    '''
    if x < radius * np.sin(arc):
        return -radius + np.sqrt(radius * radius - x * x)
    else:
        return -np.tan(arc) * x + radius * (1 / np.cos(arc) - 1)


def d_hull(x, radius=1, arc=0.99 * np.pi / 2):
    if x < radius * np.sin(arc):
        return -x / np.sqrt(radius * radius - x * x)
    else:
        return -np.tan(arc)


def comp(pair):
    return pair[1]


def construct_circle(center=Vector(), radius=1, time=0, duration=1, index=0):
    dt = duration / 5
    sphere_c = Sphere(0.05, location=center, color='important',
                      name="Sphere_" + str(index))
    rope = Rope(start=center, end=center, length=radius * 0.975, folding_direction=[0, 1, 0],
                resolution=10, name='Rope_' + str(index), color='gray_8')
    pin = Pin(location=center, colors=['important', 'metal_0.5'], scale=0.5,
              name='Pin_' + str(index))
    pencil = Pencil(location=center, colors=['wood', 'important'],
                    name='Pencil_' + str(index))
    arc = CircleArc(center=center, radius=radius, start_angle=0,
                    end_angle=2 * np.pi,
                    color='important', thickness=0.25, name="Arc_" + str(index))
    rope.attach_to(sphere_c, pencil)

    sphere_c.grow(begin_time=time, transition_time=0)
    pencil.appear(begin_time=time, transition_time=dt)
    ibpy.set_follow(pencil, arc)
    time += dt

    rope.appear(begin_time=time, transition_time=dt)
    rope.set_dynamic(begin_time=time, transition_time=4 * dt)
    time += dt

    pin.appear(begin_time=time + 0.1 * dt)
    ibpy.change_follow_influence(pencil, arc, 0, 1, begin_time=time, transition_time=dt)
    pencil.move_to(target_location=Vector(), begin_time=time, transition_time=dt)
    time += dt

    ibpy.follow(pencil, arc, 0, 1, begin_time=time, transition_time=1.5 * dt)
    arc.grow(begin_time=time, transition_time=1.5 * dt)
    time += 1.5 * dt

    pin.disappear(begin_time=time, transition_time=0.5 * dt)
    pencil.disappear(begin_time=time, transition_time=0.5 * dt)
    rope.disappear(begin_time=time, transition_time=0.5 * dt)
    time += 0.5 * dt

    return time


def get_brighter(color, brighter):
    if color == 'example' or 'gray' in color:
        bright_factor = 1.5
    else:
        bright_factor = 3
    return bright_factor * brighter


def quaternion_power(quaternion, power):
    q = Quaternion()
    for i in range(0, power):
        q = quaternion @ q
    return q


def lin_map(m, v, b):
    x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + b.x
    y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + b.y
    z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + b.z
    return Vector([x, y, z])


def rot(degree):
    rad = degree / 180 * np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def extension_for_node(connections, labels, parent, max_level, level, extension_length, level_node_map, polar,
                       paraboloid, q0,
                       t0, dt, pauses, res, tree_parts, morph_connections, leaves):
    if not isinstance(pauses, list):
        pauses = [pauses]

    # extension
    delta_length = 0.2
    old_nodes = level_node_map[level]
    dphi = 2 * np.pi / len(old_nodes)
    d3phi = dphi / 3
    level += 1

    phi = parent.v
    rel_positions = [-1, 1]
    first = True
    second = True
    parents = [parent, parent]
    pause_counter = 0
    for l in range(level, level + extension_length):
        level_nodes = []
        letters = [previous_letter(parents[0].content), next_letter(parents[1].content)]
        for letter, pos, parent in zip(letters, rel_positions, parents):
            phi = parent.v
            if first or second:
                if not first and second:
                    second = False
                if first:
                    first = False
                phi_j = phi + pos * d3phi
            else:
                phi_j = parent.v
            u = 2 * max_level + delta_length * (l - max_level)
            location = paraboloid.embedding(u, phi_j)
            normal = paraboloid.unit_n(u, phi_j)
            location += 0.01 * normal
            q = paraboloid.local_frame_quaternion(u, phi_j)
            label = SimpleTexBObject(letter, aligned='center',
                                     location=location,
                                     rotation_quaternion=q @ q0, color='text')
            label.write(begin_time=t0, transition_time=dt)
            labels.append(label)
            if l == level:
                connection = Curve([
                    lambda t: paraboloid.embedding(interpol((u - delta_length),
                                                            u - 0.01, t),
                                                   interpol(phi, phi_j, t)),
                    lambda t: polar.embedding(interpol((u - delta_length),
                                                       u - 0.01, t),
                                              interpol(phi, phi_j, t)),
                ],
                    domain=[0.15, 0.9],
                    num_points=res,
                    color='text', thickness=0.25,
                    name='Connection_' + str(len(connections)))
                morph_connections.append(connection)
            else:
                connection = Curve([
                    lambda t: paraboloid.embedding(interpol((u - delta_length),
                                                            u - 0.01, t),
                                                   interpol(phi, phi_j, t))
                ],
                    domain=[0.15, 0.9],
                    num_points=res,
                    color='text', thickness=0.25,
                    name='Connection_' + str(len(connections)))
            connections.append(connection)
            node = TreeNode(letter, label, connection=connection, color='text', u=u, v=phi_j)
            if l == level:
                leaves.append(node)
            node.parent = parent
            level_nodes.append(node)
            tree_parts.append(label)
            tree_parts.append(connection)
            node.connect(begin_time=t0, transition_time=dt)
            t0 += dt + pauses[pause_counter]
            if pause_counter < len(pauses) - 1:
                pause_counter += 1  # switch to next pause if available
        if l not in level_node_map:
            level_node_map[l] = level_nodes
        else:
            level_node_map[l] += level_nodes
        parents = level_nodes
    return t0


def get_fixed_point_from(letter, first):
    if letter == 'a':
        if first:
            return 'f_1'
        else:
            return 'f_2'
    elif letter == 'b':
        if first:
            return 'f_3'
        else:
            return 'f_1'
    elif letter == 'A':
        if first:
            return 'f_4'
        else:
            return 'f_3'
    elif letter == 'B':
        if first:
            return 'f_2'
        else:
            return 'f_4'


def extension_for_node2(connections, labels, parent, max_level, level, extension_length, level_node_map, polar, q0,
                        t0, dt, pauses, res, tree_parts, morph_connections, leaves):
    if not isinstance(pauses, list):
        pauses = [pauses]

    # extension
    delta_length = 0.2
    old_nodes = level_node_map[level]
    dphi = 2 * np.pi / len(old_nodes)
    d3phi = dphi / 3
    level += 1

    phi = parent.v
    rel_positions = [-1, 1]
    first = True
    second = True
    parents = [parent, parent]
    pause_counter = 0
    for l in range(level, level + extension_length):
        level_nodes = []
        letters = [previous_letter(parents[0].content), next_letter(parents[1].content)]
        for letter, pos, parent in zip(letters, rel_positions, parents):
            phi = parent.v
            fp = get_fixed_point_from(parent.content, first)
            if first or second:
                if not first and second:
                    second = False
                if first:
                    first = False
                phi_j = phi + pos * d3phi
            else:
                phi_j = parent.v
            u = 2 * max_level + delta_length * (l - max_level)
            location = polar.embedding(u, phi_j)
            normal = polar.unit_n(u, phi_j)
            location += 0.01 * normal
            q = polar.local_frame_quaternion(u, phi_j)

            label = SimpleTexBObject(fp, aligned='center',
                                     location=location,
                                     rotation_quaternion=q @ q0, color='example')
            label.write(begin_time=t0, transition_time=dt)
            labels.append(label)
            connection = Curve([
                lambda t: polar.embedding(interpol((u - delta_length),
                                                   u - 0.01, t),
                                          interpol(phi, phi_j, t)),
            ],
                domain=[0.25, 0.8],
                num_points=res,
                color='drawing', thickness=0.25,
                name='Connection_' + str(len(connections)))
            morph_connections.append(connection)
            connections.append(connection)
            node = TreeNode(letter, label, connection=connection, color='text', u=u, v=phi_j)
            if l == level:
                leaves.append(node)
            node.parent = parent
            level_nodes.append(node)
            tree_parts.append(label)
            tree_parts.append(connection)
            node.connect(begin_time=t0, transition_time=dt)
            t0 += dt + pauses[pause_counter]
            if pause_counter < len(pauses) - 1:
                pause_counter += 1  # switch to next pause if available
        if l not in level_node_map:
            level_node_map[l] = level_nodes
        else:
            level_node_map[l] += level_nodes
        parents = level_nodes
    return t0


def first_path_evaluation(display, coords, gen1, gen2, fp, t0=0):
    print("fill display and coordinate system for ", gen1, gen2, fp)

    am = ApollonianModel()
    a = am.get_generators()[0]
    b = am.get_generators()[1]
    A = am.get_generators()[2]
    B = am.get_generators()[3]

    gen_dict = {'a': a, 'b': b, 'A': A, 'B': B}

    comm_dict = {1: B @ A @ b @ a, 2: b @ A @ B @ a, 3: a @ B @ A @ b, 4: b @ a @ B @ A}

    calculation2 = BMatrix(np.array([["1", "0"], ["-2i", "1"]]), pre_word=r"a\cdot B=", after_word="\cdot")
    calculation3 = BMatrix(np.array([["1+i", "-1"], ["-1", "1-i"]]))
    t0 = 0.5 + display.write_text_in_back(calculation2, line=1, begin_time=t0, indent=1)
    t0 = 0.5 + display.write_text_in_back(calculation3, line=1, begin_time=t0, indent=5)
    calculation4 = BMatrix(np.array([["1+i", "-1"], ["1-2i", "1+i"]]), pre_word=r"a\cdot B=")
    t0 = 0.5 + display.write_text_in_back(calculation4, line=3, begin_time=t0, indent=1)
    colors = flatten([['text'] * 3, ['example'] * 2, ['text'] * 16, ['example'] * 4, ['text'] * 7, ['example'] * 2])
    calculation = SimpleTexBObject(
        r"aB(f_2)={(1+i)\cdot f_2-1\over (1-2i)\cdot f_2+1+i}=-1", color=colors)
    display.write_text_in_back(calculation, line=5, begin_time=t0, indent=1)

    t0 = coords.appear(begin_time=t0, transition_time=2)
    coords.axes[1].axis_label.move(direction=[0, 0, -0.5], transition_time=0)
    coords.axes[0].axis_label.move(direction=[0, 0, -0.5], transition_time=0)

    comm1 = comm_dict[fp]
    image = moebius_on_point(gen_dict[gen1] @ gen_dict[gen2], fixed_point_of(comm1))

    print("for savety aB", a @ B)
    print("f2: ", image)

    p = Sphere(r=0.1, location=Vector([4 * np.real(image), 0, 4 * np.imag(image)]), color='example')
    coords.add_object(p)
    t0 = 0.5 + p.grow(begin_time=t0)
    return t0


def create_next_point(coords, gen1, gen2, fp, t0=0):
    print("fill display and coordinate system for ", gen1, gen2, fp)
    am = ApollonianModel()
    a = am.get_generators()[0]
    b = am.get_generators()[1]
    A = am.get_generators()[2]
    B = am.get_generators()[3]
    gen_dict = {'a': a, 'b': b, 'A': A, 'B': B}
    comm_dict = {1: B @ A @ b @ a, 2: b @ A @ B @ a, 3: a @ B @ A @ b, 4: b @ a @ B @ A}
    comm1 = comm_dict[fp]
    image = moebius_on_point(gen_dict[gen1] @ gen_dict[gen2], fixed_point_of(comm1))
    print("next point for ", gen1, gen2, " at: ", image)
    p = Sphere(r=0.1, location=Vector([4 * np.real(image), 0, 4 * np.imag(image)]), color='example')
    coords.add_object(p)
    t0 = 0.5 + p.grow(begin_time=t0, transition_time=0.5)

    return t0


def create_next_point2(coords, gen1, gen2, gen3, fp, t0=0):
    print("fill display and coordinate system for ", gen1, gen2, fp)
    am = ApollonianModel()
    a = am.get_generators()[0]
    b = am.get_generators()[1]
    A = am.get_generators()[2]
    B = am.get_generators()[3]
    gen_dict = {'a': a, 'b': b, 'A': A, 'B': B}
    comm_dict = {1: B @ A @ b @ a, 2: b @ A @ B @ a, 3: a @ B @ A @ b, 4: b @ a @ B @ A}
    comm1 = comm_dict[fp]
    image = moebius_on_point(gen_dict[gen1] @ gen_dict[gen2]@gen_dict[gen3], fixed_point_of(comm1))
    print("next point for ", gen1, gen2,gen3, " at: ", image)
    p = Sphere(r=0.1, location=Vector([4 * np.real(image), 0, 4 * np.imag(image)]), color='example')
    coords.add_object(p)
    t0 = 0.5 + p.grow(begin_time=t0, transition_time=0.5)

    return t0

class ApollonianFractal(Scene):
    def __init__(self):
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('title', {'duration': 6}),
            ('title2', {'duration': 6}),
            ('title3', {'duration': 6}),
            ('title4', {'duration': 6}),
            ('title5', {'duration': 6}),
            ('epsilon', {'duration': 6}),
            ('intro', {'duration': 120}),
            ('intro2', {'duration': 120}),
            ('curvature_gasket', {'duration': 30}),
            ('zoom', {'duration': 16}),
            ('one_minute_trafo_text', {'duration': 31}),
            ('one_minute_trafo_one', {'duration': 12}),
            ('one_minute_trafo_two', {'duration': 12}),
            ('one_minute_trafo_three', {'duration': 12}),
            ('one_minute_trafo_four', {'duration': 12}),
            ('one_minute_tree', {'duration': 81}),
            ('flyby', {'duration': 0}),
            ('create_fundamental_domain', {'duration': 10}),
            ('coding', {'duration': 80}),
            ('towards_apollonian', {'duration': 40}),
            ('apollonian_adaptive', {'duration': 60}),
            ('finale', {'duration': 81}),
            ('groups1b', {'duration': 57}),
            ('groups2', {'duration': 33}),
            ('groups3', {'duration': 65}),
            ('geometric_construction', {'duration': 0}),
            ('induced_transformation', {'duration': 28}),
            ('on_the_sphere', {'duration': 30}),
            ('rotate_sphere', {'duration': 12}),
            ('algebra', {'duration': 18}),
            ('summary', {'duration': 60}),
            ('variant', {'duration': 24}),
            ('doubling', {'duration': 37}),
            ('fixed_points', {'duration': 35}),
            ('fixed_points2', {'duration': 5}),
            ('fixed_points3', {'duration': 87}),
            ('pre_full_picture', {'duration': 30}),
            ('full_picture_start', {'duration': 17}),
            ('full_picture', {'duration': 88}),
            ('full_picture_spider', {'duration': 41}),
            ('groups', {'duration': 65}),
            ('discs_and_group', {'duration': 49}),
            ('tree_extension', {'duration': 179}),
            ('tree_extension2', {'duration': 130}),
            ('apollonian', {'duration': 38}),
            ('apollonian_full', {'duration': 74}),
            ('commutators', {'duration': 30}),
            ('commutators_adaptive', {'duration': 90}),
            ('commutators_adaptive_ii', {'duration': 90}),
            ('geogebra', {'duration': 12}),
            ('pattern', {'duration': 12}),
            ('families', {'duration': 32}),
            ('ancient', {'duration': 50}),
            ('curves1', {'duration': 10}),
            ('curves2', {'duration': 15}),
            ('curves3', {'duration': 15}),
            ('curves3b', {'duration': 6}),
            ('curves3c', {'duration': 7}),
            ('curves3d', {'duration': 16}),
            ('curves4', {'duration': 15}),
            ('curves4b', {'duration': 12}),
            ('curves5', {'duration': 33}),
            ('curves6', {'duration': 48}),
            ('clock', {'duration': 120}),
            ('scaling', {'duration': 30}),
            ('theory', {'duration': 30}),
            ('actor', {'duration': 6}),
            ('indentations', {'duration': 20}),
            ('further_reading', {'duration': 0}),
            ('linear_transformations', {'duration': 16}),
            ('actors', {'duration': 6}),
            ('sl2c', {'duration': 8}),
            ('indentations', {'duration': 20}),
            ('inverses', {'duration': 10}),
        ])
        # in the super constructor the timing is set for all scenes
        super().__init__(light_energy=2, transparent=False)

    def title(self):
        cues = self.sub_scenes['title']
        t0 = 0.5  # cues['start']

        title = FreeHandText("title1", eps=0.006, color='example',
                             rotation_euler=[-np.pi / 2, 0, 0], bevel_depth=0.05,
                             aligned='center')
        # title = FreeHandText("title2",eps=0.02)

        title.grow(begin_time=t0, transition_time=5)
        t0 += 5
        print("time elapsed ", t0)

    def title2(self):
        cues = self.sub_scenes['title2']
        t0 = 0.5  # cues['start']

        title = FreeHandText("continued", eps=0.005, color='example',
                             rotation_euler=[-np.pi / 2, 0, 0], bevel_depth=0.05,
                             aligned='center')
        # title = FreeHandText("title2",eps=0.02)

        title.grow(begin_time=t0, transition_time=5)
        t0 += 5
        print("time elapsed ", t0)

    def title3(self):
        cues = self.sub_scenes['title3']
        t0 = 0.5  # cues['start']

        title = FreeHandText('titel3', eps=0.005, color='example',
                             rotation_euler=[-np.pi / 2, 0, 0], bevel_depth=0.05,
                             aligned='center')

        title.grow(begin_time=t0, transition_time=5)
        t0 += 5
        print("time elapsed ", t0)

    def epsilon(self):
        cues = self.sub_scenes['epsilon']
        t0 = 0.5  # cues['start']

        title = FreeHandText('1671834913667', eps=0.005, color='example',
                             rotation_euler=[-np.pi / 2, 0, 0], bevel_depth=0.05,
                             aligned='center')

        title.grow(begin_time=t0, transition_time=5)
        t0 += 5
        print("time elapsed ", t0)

    def title4(self):
        cues = self.sub_scenes['epsilon']
        t0 = 0.5  # cues['start']

        title = FreeHandText('drawing_curves', eps=0.005, color='drawing',
                             rotation_euler=[-np.pi / 2, 0, 0], bevel_depth=0.05,
                             aligned='center')

        title.grow(begin_time=t0, transition_time=5)
        t0 += 5
        print("time elapsed ", t0)

    def title5(self):
        cues = self.sub_scenes['title5']
        t0 = 0.5  # cues['start']

        title = FreeHandText('1672669822416', eps=0.005, color='custom1',
                             rotation_euler=[-np.pi / 2, 0, 0], bevel_depth=0.05,
                             aligned='center')

        title.grow(begin_time=t0, transition_time=5)
        t0 += 5
        print("time elapsed ", t0)

    def intro(self):
        cues = self.sub_scenes['intro']
        t0 = 0  # cues['start']

        # hdri background
        set_hdri_background('kloppenheim_06_puresky_4k', ext='exr', transparent=True)
        set_hdri_strength(1)
        # after effects
        set_exposure(1.5)

        eps = 0.0001
        average = 100
        scale = 100
        curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)
        fx2 = lambda x: function_from_complex_list(curve, x, scale=scale, average=average)

        fractal = Function([fx], None, domain=[0, 1], mode='PARAMETRIC', name='full',
                           color='silk', extrude=0.05, thickness=0.3, num_points=resolution, emission=0.1)

        average_fractal = Function([fx2], None, domain=[0, 1], mode='PARAMETRIC',
                                   name='average', color='example', extrude=0, thickness=0,
                                   numpoints=resolution / average)

        t0 += 0.5
        duration = 120
        fractal.grow(begin_time=t0, transition_time=duration)

        spider = AnimBObject('Spider', object_name='widow', armature='Armature', color='joker', scale=1,
                             location=[0, 0, -1], emission=0.75)
        spider.appear(begin_time=t0, transition_time=2)

        spider.armature.follow(average_fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')

        # set up camera
        ibpy.set_camera_location(location=[0, 30, 0])
        camera_circle = BezierCircle(radius=50, location=[0, 50, 10])
        empty_circle = BezierCircle(radius=50, location=[0, 50, 0])
        empty_circle.rescale(rescale=[0.7, 0.7, 0.7], begin_time=t0 + 10, transition_time=10)
        camera_empty = EmptyCube()
        camera_empty.follow(empty_circle, begin_time=t0, transition_time=duration)
        ibpy.set_camera_follow(camera_circle)
        ibpy.camera_follow(camera_circle, initial_value=0, final_value=1, begin_time=t0, transition_time=duration)
        ibpy.set_camera_view_to(camera_empty)
        camera_circle.move(direction=[0, 0, -5], begin_time=t0, transition_time=22)
        camera_circle.move(direction=[0, 0, 145], begin_time=t0 + 23, transition_time=duration - 23 - t0)

        camera_empty2 = EmptyCube(name='CameraEmpty2')  # create empty that overtakes control in the end
        camera_empty2.move(direction=[0, 50, 0], begin_time=t0 + 110, transition_time=2.5)
        set_camera_view_to(camera_empty2)

        ibpy.set_track_influence(ibpy.get_camera(), camera_empty, 1, begin_time=t0 + 110.5)
        ibpy.set_track_influence(ibpy.get_camera(), camera_empty, 0, begin_time=t0 + 116)
        ibpy.set_track_influence(ibpy.get_camera(), camera_empty2, 0, begin_time=t0 + 110.5)
        ibpy.set_track_influence(ibpy.get_camera(), camera_empty2, 0.75, begin_time=t0 + 115.5)

        # zooming

        ibpy.camera_zoom(lens=13, begin_time=22, transition_time=4)
        ibpy.camera_zoom(lens=40, begin_time=32, transition_time=5)
        ibpy.camera_zoom(lens=27, begin_time=38)

        # prepare spheres

        all_points = []
        all_curvatures = []
        max_curvature = 1500
        # create spheres up to a given resolution
        N = 100
        # import the data of centers and curvatures for a unit cell of the strip of circles
        points = read_data("centers.dat")
        curvatures = read_data("curvatures.dat")

        # calculate circles under inversion
        for center, curvature in zip(points, curvatures):
            for i in set(range(-N, 0)):
                M, R = invert(center + i, 1 / curvature)
                if 1 / R < max_curvature:
                    all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                    all_curvatures.append(1 / R / scale)
                if np.imag(center) != 0.5:
                    M, R = invert(np.real(center) + i + 1j - np.imag(center) * 1j, 1 / curvature)
                    if 1 / R < max_curvature:
                        all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                        all_curvatures.append(1 / R / scale)

        # at center sphere
        M, R = invert(points[0], 1 / curvatures[0])
        all_points.append((M + 1j) * scale)
        all_curvatures.append(1 / R / scale)

        # make spheres appear depending on the position of the spider
        offset = int(t0 * FRAME_RATE)
        # figure out the correct action with the following function
        # ibpy.print_actions()
        # figure out the correct fcurve
        # ibpy.print_fcurves_of_action("ArmatureAction")

        for frm in range(offset, offset + duration * FRAME_RATE):
            spider_pos = fx2(value_of_action_fcurve_at_frame("ArmatureAction", 0, frm))
            distance = spider_pos.length
            angle = np.angle(spider_pos.x + 1j * spider_pos.y)
            to_be_removed = []
            counter = 0
            for point, curvature in zip(all_points, all_curvatures):
                if curvature < 50:
                    smooth = 2
                else:
                    smooth = None
                if np.abs(point) + 1 / curvature < 1.1 * distance and angle < np.angle(point):
                    sphere = Sphere(1 / curvature, location=[np.real(point), np.imag(point), 0],
                                    resolution=resolution_function(curvature, scale), color='text',
                                    transmission=0.75 + 1 / (scale * curvature),
                                    roughness=0, ior=1.33, metallic=0, smooth=smooth)
                    sphere.appear(begin_time=frm / FRAME_RATE, transition_time=1)
                    to_be_removed.append(counter)
                counter += 1

            to_be_removed.sort()
            for i in reversed(to_be_removed):
                all_points.pop(i)
                all_curvatures.pop(i)

    def intro2(self):
        cues = self.sub_scenes['intro']
        t0 = 0  # cues['start']

        # hdri background
        set_hdri_background('kloppenheim_06_puresky_4k', ext='exr', transparent=False)
        set_hdri_strength(0.8)
        # after effects
        set_exposure(1.5)

        eps = 0.0001
        average = 10
        scale = 100
        curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)
        fx2 = lambda x: function_from_complex_list(curve, x, scale=scale, average=average)

        fractal = Function([fx], None, domain=[0, 1], mode='PARAMETRIC', name='full', location=[0, 0, 0.2],
                           color='silk', extrude=0.075, thickness=0.3, num_points=resolution, emission=0.1)

        # average_fractal = Function([fx2], None, domain=[0, 1], type='PARAMETRIC',
        #                            name='average', color='example', extrude=0, thickness=0,
        #                            numpoints=resolution / average)

        t0 += 0.5
        duration = 120
        fractal.grow(begin_time=t0, transition_time=duration)
        ibpy.set_linear_fcurves(fractal)
        spider = AnimBObject('Spider', object_name='widow', armature='Armature', color='joker', scale=0.25,
                             location=[0, 0, 0], emission=0.75)
        spider.appear(begin_time=t0, transition_time=2)
        spider.armature.follow(fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')
        ibpy.set_linear_fcurves(spider.armature)

        # set up camera

        ibpy.set_camera_location(location=[0, 0, 0])
        ibpy.set_camera_rotation(rotation=[np.pi / 18 * 8, 0, -np.pi / 2])
        camera_circle = BezierCircle(radius=107, location=[0, 50, 30])
        ibpy.set_camera_follow(camera_circle)
        ibpy.camera_follow(camera_circle, initial_value=0, final_value=1, begin_time=t0, transition_time=duration)

        camera_circle.move(direction=[0, 0, 65.5], begin_time=duration - 10, transition_time=8)
        ibpy.camera_zoom(38, begin_time=duration - 5, transition_time=6)
        ibpy.camera_rotate_to(rotation_euler=[np.pi / 4, 0, -np.pi / 2], begin_time=duration - 10, transition_time=8)

        # prepare spheres

        all_points = []
        all_curvatures = []
        max_curvature = 1500
        # create spheres up to a given resolution
        N = 100
        # import the data of centers and curvatures for a unit cell of the strip of circles
        points = read_data("centers.dat")
        curvatures = read_data("curvatures.dat")

        # calculate circles under inversion
        for center, curvature in zip(points, curvatures):
            for i in set(range(-N, N)):
                M, R = invert(center + i, 1 / curvature)
                if 1 / R < max_curvature:
                    all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                    all_curvatures.append(1 / R / scale)
                if np.imag(center) != 0.5:
                    M, R = invert(np.real(center) + i + 1j - np.imag(center) * 1j, 1 / curvature)
                    if 1 / R < max_curvature:
                        all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                        all_curvatures.append(1 / R / scale)

        # at center sphere
        # M, R = invert(points[0], 1 / curvatures[0])
        #
        # all_points.append((M + 1j) * scale)
        # all_curvatures.append(1 / R / scale)

        # make cylinders appear depending on the position of the spider
        offset = int(t0 * FRAME_RATE)

        for frm in range(offset, offset + duration * FRAME_RATE):
            spider_pos = fx2(value_of_action_fcurve_at_frame("ArmatureAction", 0, frm))
            distance = spider_pos.length
            z_spider = spider_pos.x + 1j * spider_pos.y
            angle = np.angle(z_spider)
            to_be_removed = []
            counter = 0
            for point, curvature in zip(all_points, all_curvatures):
                if np.abs(point - z_spider) < 1.1 / curvature and angle < np.angle(point):
                    start = Vector([np.real(point), np.imag(point), 0])
                    cyl = Cylinder.from_start_to_end(start=start, end=start + Vector([0, 0, 1]) * np.sqrt(curvature),
                                                     thickness=10 / curvature, color='text', metallic=0.5, roughness=0,
                                                     ior=1.5, smooth=False, transmission=0.25,
                                                     vertices=int(4 * np.sqrt(max_curvature / curvature)))
                    cyl.grow(begin_time=frm / FRAME_RATE, transition_time=1)
                    to_be_removed.append(counter)
                counter += 1

            to_be_removed.sort()
            for i in reversed(to_be_removed):
                all_points.pop(i)
                all_curvatures.pop(i)

    def curvature_gasket(self):
        cues = self.sub_scenes['curvature_gasket']
        t0 = 0  # cues['start']

        # hdri background
        set_hdri_background('kloppenheim_06_puresky_4k', ext='exr', transparent=True)
        set_hdri_strength(1)

        scale = 100

        # set up camera
        ibpy.set_camera_location(location=[0, 50, 175])
        ibpy.set_camera_rotation(rotation=Vector())
        # camera_circle = BezierCircle(radius=107, location=[0, 50, 30])
        # ibpy.set_camera_follow(camera_circle)
        # ibpy.camera_follow(camera_circle, initial_value=0, final_value=1, begin_time=t0, transition_time=duration)
        #
        # camera_circle.move(direction=[0,0,65.5],begin_time=duration-10,transition_time=8)
        # ibpy.camera_zoom(38,begin_time=duration-5,transition_time=6)
        # ibpy.camera_rotate_to(rotation_euler=[np.pi/4,0,-np.pi/2],begin_time=duration-10,transition_time=8)

        # prepare spheres

        all_points = []
        all_curvatures = []
        max_curvature = 300
        # create spheres up to a given resolution
        N = 100
        # import the data of centers and curvatures for a unit cell of the strip of circles
        points = read_data("centers.dat")
        curvatures = read_data("curvatures.dat")

        # calculate circles under inversion
        for center, curvature in zip(points, curvatures):
            for i in set(range(-N, N)):
                M, R = invert(center + i, 1 / curvature)
                if 1 / R < max_curvature:
                    all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                    all_curvatures.append(1 / R / scale)
                if np.imag(center) != 0.5:
                    M, R = invert(np.real(center) + i + 1j - np.imag(center) * 1j, 1 / curvature)
                    if 1 / R < max_curvature:
                        all_points.append((M + 1j) * scale)  # translate to different sectors of the band
                        all_curvatures.append(1 / R / scale)

        # at center sphere
        M, R = invert(points[0], 1 / curvatures[0])
        all_points.append((M + 0.5j) * scale)
        all_curvatures.append(1 / R / scale)

        # make cylinders appear depending on the position of the spider
        offset = int(t0 * FRAME_RATE)
        duration = 25

        # create dictonary to sort according to curvatures
        circles = {"50j": -0.02}
        for point, curvature in zip(all_points, all_curvatures):
            circles[point] = curvature

        dt = duration / len(all_points)
        counter = 0
        for point, curvature in sorted(circles.items(), key=lambda item: item[1]):
            if curvature < max_curvature:
                if 100 * curvature > 50:
                    thickness = 10
                else:
                    thickness = 20
                cyl = BezierCircle(location=z2vec(complex(point)), radius=np.abs(1 / curvature), metallic=0.5,
                                   roughness=0, thickness=thickness, color='important')
                if counter < 8:
                    dt2 = 0.5
                    dt1 = 0.5
                else:
                    dt2 = 2.5 * dt
                    dt1 = dt
                if 1 > curvature > 0:  # only labels upto 100
                    c = int(curvature * 100)
                    curve_string = SimpleTexBObject(r"\text{" + str(c) + "}", aligned='center',
                                                    location=z2vec(complex(point)), rotation_euler=[0, 0, 0],
                                                    scale=4 / curvature, color='joker')
                    curve_string.write(begin_time=t0 + dt2, transition_time=dt2)
                    counter += 1
                cyl.grow(begin_time=t0, transition_time=dt2)
                counter += 1
                t0 += dt1

        print("finished", t0)

    def one_minute_trafo_text(self):
        cues = self.sub_scenes['one_minute_trafo_text']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20)
        display.appear(begin_time=t0)
        t0 += 1.5

        title = SimpleTexBObject(r"\text{Two complex transformations}", color='example', aligned='center')
        display.write_title(title, begin_time=t0)
        t0 += 1.5

        matrix_a = BMatrix(np.array([['1', '0'], ['-2i', '1']]), color='joker')
        transform = SimpleTexBObject(r'a: z\mapsto {z\over -2 i z+1}')
        display.set_cursor_to_start_of_line(1)
        display.add_text(transform, indent=0.5)
        t0 += 1
        transform.write(letter_set={0, 1, 2, 3, 4}, begin_time=t0, transition_time=0.2)
        t0 += 0.3
        transform.write(letter_set={8}, begin_time=t0, transition_time=0.15)
        t0 += 0.50
        transform.write(letter_set={9}, begin_time=t0, transition_time=0.1)
        t0 += 0.15
        transform.write(letter_set={5, 6, 7, 10, 11, 12}, begin_time=t0, transition_time=0.45)
        t0 += 0.95

        transform = SimpleTexBObject(r'b: z\mapsto {(1-i)z+1\over  z+(1+i)}')
        display.set_cursor_to_start_of_line(8)
        display.add_text(transform, indent=0.5)
        t0 += 1

        transform.write(letter_set={0, 1, 2, 3, 4}, begin_time=t0, transition_time=0.2)
        t0 += 0.3
        transform.write(letter_set={5, 6, 8, 10, 12, 15, 17, 20}, begin_time=t0, transition_time=1)
        t0 += 1.50
        transform.write(letter_set={13}, begin_time=t0, transition_time=0.1)
        t0 += 0.15
        transform.write(letter_set={7, 9, 11, 14, 16, 18, 19}, begin_time=t0, transition_time=0.95)
        t0 += 1.45

        fix_a = SimpleTexBObject(r"\text{single fixed point: } z_f=0 ")
        display.write_text_in(fix_a, line=3, indent=1, begin_time=t0)
        t0 += 1.5

        inf_a = SimpleTexBObject(r"a:\infty\mapsto \tfrac{i}{2}")
        display.write_text_in(inf_a, line=4, indent=1, begin_time=t0)
        t0 += 1.5

        spec_a = SimpleTexBObject(r"a:-\tfrac{i}{2}\mapsto \infty")
        display.write_text_in(spec_a, line=5, indent=1, begin_time=t0)
        t0 += 1.5

        zero_a = SimpleTexBObject(r"a:0\mapsto 0")
        display.write_text_in(zero_a, line=6, indent=1, begin_time=t0)
        t0 += 1.5

        matrix_b = BMatrix(np.array([['1-i', '1'], ['1', '1+i']]), color='joker')

        fix_b = SimpleTexBObject(r"\text{single fixed point: } z_f=-i ")
        display.write_text_in(fix_b, line=10, indent=1, begin_time=t0)
        t0 += 1.5

        inf_b = SimpleTexBObject(r"b:\infty\mapsto 1-i")
        display.write_text_in(inf_b, line=11, indent=1, begin_time=t0)
        t0 += 1.5

        spec_b = SimpleTexBObject(r"b:-1-i\mapsto \infty")
        display.write_text_in(spec_b, line=12, indent=1, begin_time=t0)
        t0 += 1.5

        zero_b = SimpleTexBObject(r"b:0\mapsto \tfrac{1}{2}-\tfrac{i}{2}")
        display.write_text_in(zero_b, line=13, indent=1, begin_time=t0)
        t0 += 1.5

        display.write_text_in(matrix_a, line=1, indent=6, begin_time=t0)
        t0 += 2.5

        display.write_text_in(matrix_b, line=8, indent=6, begin_time=t0)
        t0 += 1.5

        matrix_A = BMatrix(np.array([['1', '0'], ['2i', '1']]), pre_word=r"A=a^{-1}=", color='joker')
        display.write_text_in(matrix_A, line=15, indent=0.5, begin_time=t0)
        t0 += 1.5

        matrix_B = BMatrix(np.array([['1+i', '-1'], ['-1', '1-i']]), pre_word=r"B=b^{-1}=", color='joker')
        display.write_text_in(matrix_B, line=17, indent=0.5, begin_time=t0)
        t0 += 1.5

        print("finished at ", t0)

    def one_minute_trafo_one(self):
        cues = self.sub_scenes['one_minute_trafo_one']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 2], [-2, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[8, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 0.5), np.arange(-2, 2.1, .5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane1')
        coords.appear(begin_time=t0, transition_time=2)
        t0 += 2.5

        a_trafo = VectorTransformation()
        # look at the mathematica worksheet "VectorTransformations" to see how the expressions are generated
        a_trafo.set_transformation_function(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) * Vector(
                [v.x, 0, 2 * v.x ** 2 + v.z + 2 * v.z ** 2]))
        a_trafo.set_first_derivative_functions(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) ** 2 * np.array(
                [[-4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0, 4 * v.x * (1 + 2 * v.z)], [0, 0, 0],
                 [-4 * v.x * (1 + 2 * v.z), 0, -4 * v.x ** 2 + (1 + 2 * v.z) ** 2]]
            ))
        a_trafo.set_second_derivative_functions(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) ** 3 * np.array(
                [[[8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2), 0,
                   4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2)], [0, 0, 0],
                  [-4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2), 0,
                   8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2)]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[-4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2), 0,
                   8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2)], [0, 0, 0],
                  [8 * v.x * (-4 * v.x ** 2 + 3 * (1 + 2 * v.z) ** 2), 0,
                   -4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2)]]
                 ]))

        coords.draw_transformable_grid(transformations=[a_trafo], twists=[0, 0],
                                       begin_time=t0, transition_time=3, sub_grid=5,
                                       thickness=0.1, num_points=1000)
        t0 += 3.5
        coords.grid_next_transform(begin_time=t0, transition_time=3)
        t0 += 3.5

        print("finished at ", t0)

    def one_minute_trafo_two(self):
        cues = self.sub_scenes['one_minute_trafo_two']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 2], [-2, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[8, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 0.5), np.arange(-2, 2.1, .5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane2')
        coords.appear(begin_time=t0, transition_time=2)
        t0 += 2.5

        a_trafo = VectorTransformation()
        # look at the mathematica worksheet "VectorTransformations" to see how the expressions are generated
        a_trafo.set_transformation_function(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) * Vector(
                [v.x, 0, 2 * v.x ** 2 + v.z + 2 * v.z ** 2]))
        a_trafo.set_first_derivative_functions(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) ** 2 * np.array(
                [[-4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0, 4 * v.x * (1 + 2 * v.z)], [0, 0, 0],
                 [-4 * v.x * (1 + 2 * v.z), 0, -4 * v.x ** 2 + (1 + 2 * v.z) ** 2]]
            ))
        a_trafo.set_second_derivative_functions(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) ** 3 * np.array(
                [[[8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2), 0,
                   4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2)], [0, 0, 0],
                  [-4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2), 0,
                   8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2)]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[-4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2), 0,
                   8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2)], [0, 0, 0],
                  [8 * v.x * (-4 * v.x ** 2 + 3 * (1 + 2 * v.z) ** 2), 0,
                   -4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2)]]
                 ]))

        coords.draw_transformable_polar_grid(transformations=[a_trafo], twists=[0, 0],
                                             begin_time=t0, transition_time=3, sub_grid=5,
                                             thickness=0.1, num_points=2500)
        t0 += 3.5
        coords.grid_next_transform(begin_time=t0, transition_time=3)
        t0 += 3.5

        print("finished at ", t0)

    def one_minute_trafo_three(self):
        cues = self.sub_scenes['one_minute_trafo_three']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 2], [-2, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[8, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 0.5), np.arange(-2, 2.1, .5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane2')
        coords.appear(begin_time=t0, transition_time=2)
        t0 += 2.5

        b_trafo = VectorTransformation()
        # look at the mathematica worksheet "VectorTransformations" to see how the expressions are generated
        b_trafo.set_transformation_function(
            lambda v: 1 / regularized((v.x + 1) ** 2 + (v.z + 1) ** 2, 0.001) * Vector(
                [(v.x + 1) * (v.x + v.z + 1) + (v.z - v.x) * (v.z + 1), 0,
                 (v.z - v.x) * (v.x + 1) - (v.x + v.z + 1) * (v.z + 1)])
        )
        b_trafo.set_first_derivative_functions(
            lambda v: 1 / regularized(2 * v.x * (v.x + 2) + v.z * (v.z + 2), 0.001) ** 2 * np.array(
                [[(v.x - v.z) * (2 + v.x + v.z), 0, -2 * (1 + v.x) * (1 + v.z)], [0, 0, 0],
                 [2 * (1 + v.x) * (1 + v.z), 0, (v.x - v.z) * (2 + v.x + v.z)]]
            ))
        b_trafo.set_second_derivative_functions(
            lambda v: 1 / regularized(2 * v.x * (v.x + 2) + v.z * (v.z + 2), 0.001) ** 3 * np.array(
                [[[-2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z)), 0,
                   -2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z))], [0, 0, 0],
                  [2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z)), 0,
                   -2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z))]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z)), 0,
                   -2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z))], [0, 0, 0],
                  [2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z)), 0,
                   2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z))]]
                 ]))

        coords.draw_transformable_grid(transformations=[b_trafo], twists=[0, 0],
                                       begin_time=t0, transition_time=3, sub_grid=5,
                                       thickness=0.1, num_points=1000)
        t0 += 3.5
        coords.grid_next_transform(begin_time=t0, transition_time=3)
        t0 += 3.5

        print("finished at ", t0)

    def one_minute_trafo_four(self):
        cues = self.sub_scenes['one_minute_trafo_four']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 2], [-3, 1]], radii=[0.03, 0.03],
                                  all_n_tics=[8, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 0.5), np.arange(-3, 1.1, .5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 2],
                                  name='ComplexPlane2')
        coords.appear(begin_time=t0, transition_time=2)
        t0 += 2.5

        b_trafo = VectorTransformation()
        # look at the mathematica worksheet "VectorTransformations" to see how the expressions are generated
        b_trafo.set_transformation_function(
            lambda v: 1 / ((v.x + 1) ** 2 + (v.z + 1) ** 2) * Vector(
                [(v.x + 1) * (v.x + v.z + 1) + (v.z - v.x) * (v.z + 1), 0,
                 (v.z - v.x) * (v.x + 1) - (v.x + v.z + 1) * (v.z + 1)])
        )
        b_trafo.set_first_derivative_functions(lambda v: 1 / (2 * v.x * (v.x + 2) + v.z * (v.z + 2)) ** 2 * np.array(
            [[(v.x - v.z) * (2 + v.x + v.z), 0, -2 * (1 + v.x) * (1 + v.z)], [0, 0, 0],
             [2 * (1 + v.x) * (1 + v.z), 0, (v.x - v.z) * (2 + v.x + v.z)]]
        ))
        b_trafo.set_second_derivative_functions(lambda v: 1 / (2 * v.x * (v.x + 2) + v.z * (v.z + 2)) ** 3 * np.array(
            [[[-2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z)), 0,
               -2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z))], [0, 0, 0],
              [2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z)), 0,
               -2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z))]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z)), 0,
               -2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z))], [0, 0, 0],
              [2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z)), 0,
               2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z))]]
             ]))

        coords.draw_transformable_polar_grid(transformations=[b_trafo], twists=[0, 0],
                                             begin_time=t0, transition_time=3, sub_grid=5,
                                             thickness=0.1, num_points=1000, center=Vector([0, 0, -1]))
        t0 += 3.5
        coords.grid_next_transform(begin_time=t0, transition_time=3)
        t0 += 3.5

        print("finished at ", t0)

    def one_minute_tree(self):
        cues = self.sub_scenes['one_minute_tree']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[0, 0, 0])
        ibpy.set_camera_rotation(rotation=[np.pi / 6, 0, 0])
        camera_empty = EmptyCube(location=[1.5, 0, -0.5])
        camera_radius = Vector([0.02954, -3.1279, 0])
        camera_circle = BezierCircle(radius=np.sqrt(camera_radius.dot(camera_radius)), location=[1.5, 0, 4.9261])

        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_follow(camera_circle)

        level_node_map = {}
        flat_levels = 3

        polar = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: 0,  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: 0,
            lambda u, v: 0
        )

        paraboloid = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: hull(u, radius=np.sqrt(8), arc=np.pi / 4),  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: d_hull(u, radius=np.sqrt(8), arc=np.pi / 4),
            lambda u, v: 0
        )

        identity = SimpleTexBObject(".", location=embedding(0, 0), rotation_euler=[np.pi / 2, 0, 0], aligned='center',
                                    color='drawing')
        identity.write(begin_time=t0, transition_time=0.1)
        root = TreeNode("I", identity)
        level_node_map[0] = [root]

        quat_start = Quaternion(Vector([0, 0, 1]), np.pi / 2)
        tree_container = Container(obj=identity, rotation_quaternion=quat_start,
                                   location=[2.5, -0.2, 0.5],
                                   scale=[0.5, 0.5, 0.5])
        tree_container.appear(begin_time=t0)

        res = 20
        phi0 = -np.pi
        d_phi = np.pi / 2

        # level = 1
        level_letters = []
        letter = 'a'
        for i in range(4):
            level_letters.append(letter)
            letter = next_letter(letter)

        nodes = []

        duration = 3
        dt = 0.8 * duration / 4
        r2 = np.sqrt(2)
        ir2 = 1 / r2
        q0 = Quaternion([ir2, 0, 0, ir2])

        model = ApollonianModel()

        morph_connections = []

        # first four nodes
        dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=1)
        dfs.generate_tree()
        leaves = dfs.get_leaves()
        self.disc_counter = 0

        for i, l in enumerate(level_letters):
            phi = phi0 + i * d_phi
            q = polar.local_frame_quaternion(2, phi)
            location = polar.embedding(2, phi)
            label = SimpleTexBObject(l, aligned='center',
                                     location=location,
                                     rotation_quaternion=q0 @ q,
                                     name=l + '_1', color=model.colors[i])
            ibpy.set_parent(label, tree_container)
            label.write(begin_time=t0, transition_time=dt)
            connection = Curve([
                lambda t: partial(polar.embedding, v=phi)(2 * t),
                lambda t: partial(paraboloid.embedding, v=phi)(2 * t),
            ], domain=[0.0, 0.9],
                num_points=res, color='text',
                thickness=0.125, name='Connection_' + l + '_1')
            ibpy.set_parent(connection, tree_container)
            morph_connections.append(connection)
            node = TreeNode(l, label, connection, color=model.colors[i], u=2, v=phi)
            node.connect(begin_time=t0, transition_time=dt)
            node.parent = root
            nodes.append(node)

            disc = self.disc_from_circle(leaves[i].circle, begin_time=t0, transition_time=dt,
                                         color=leaves[i].color,
                                         offset=1, brighter=0)
            if i == 0:
                disc0 = disc
            elif i == 1:
                disc1 = disc

            t0 += dt * 1.25
        level_node_map[1] = nodes
        t0 += 1

        # make big discs more transparent for the tree
        disc0.disappear(alpha=0.5, begin_time=t0)
        disc1.disappear(alpha=0.5, begin_time=t0)
        # regular tree to level flat_level

        for level in range(2, flat_levels + 1):
            self.disc_counter = 0
            dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=level)
            dfs.generate_tree()
            leaves = dfs.get_leaves()
            duration = 3 * np.sqrt(level)
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 0.8 * duration / len(old_nodes) / 3
            d_phi = 2 * np.pi / len(old_nodes)
            d3phi = d_phi / 3
            for i, parent in enumerate(old_nodes):
                phi = parent.v
                letter = parent.content
                letter = previous_letter(letter)
                for j in range(-1, 2):
                    phi_j = phi + d3phi * j
                    location = polar.embedding(level + 1, phi_j)
                    n = polar.unit_n(level + 1, phi_j)
                    location += 0.01 * n
                    q = polar.local_frame_quaternion(level + 1, phi_j)
                    color = leaves[self.disc_counter].color
                    label = SimpleTexBObject(letter, aligned='center',
                                             location=location,
                                             rotation_quaternion=q0 @ q, color=parent.color,
                                             brighter=get_brighter(color, level - 1))
                    ibpy.set_parent(label, tree_container)
                    label.write(begin_time=t0, transition_time=dt)
                    self.disc_from_circle(leaves[self.disc_counter].circle, begin_time=t0, transition_time=dt,
                                          color=color,
                                          offset=level, brighter=level - 1)
                    connection = Curve([
                        lambda t: polar.embedding(interpol((level - 1) + 1,
                                                           (level - 1) + 1.9, t),
                                                  interpol(phi, phi_j, t)),
                        lambda t: paraboloid.embedding(interpol((level - 1) + 1,
                                                                (level - 1) + 1.9, t),
                                                       interpol(phi, phi_j, t)),

                    ],
                        domain=[0.1, 0.95],
                        num_points=res,
                        color='text', thickness=0.125,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    ibpy.set_parent(connection, tree_container)
                    morph_connections.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color, u=level + 1, v=phi_j)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt * 1.25
                    letter = next_letter(letter)
            level_node_map[level] = nodes
            t0 += 1

        # morph tree
        for connection in morph_connections:
            connection.next(begin_time=t0)

        for level_nodes in level_node_map.values():
            for node in level_nodes:
                node.bob.move_to(target_location=paraboloid.embedding(node.u, node.v), begin_time=t0)
                q = paraboloid.local_frame_quaternion(node.u, node.v)
                node.bob.rotate(rotation_quaternion=q @ q0, begin_time=t0)

        rotation_state = 0
        quat_tilt = Quaternion(Vector([1, 0, 0]), -np.pi / 4)
        quat_next = quat_tilt @ quat_start

        tree_container.rotate(rotation_quaternion=quat_next, begin_time=t0)
        tree_container.move_to(target_location=[3.4, -0.25, 0.25], begin_time=t0)
        tree_container.rescale(rescale=[0.5, 0.5, 0.5], begin_time=t0)
        t0 += 1.5

        # next levels directly on the morphed tree

        axis_rot = Vector([0, 1, 1]).normalized()
        rotation_state = np.pi
        quat_rot = Quaternion(axis_rot, rotation_state)

        power = 0
        for level in range(flat_levels + 1, flat_levels + 3):
            self.disc_counter = 0
            dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=level)
            dfs.generate_tree()
            leaves = dfs.get_leaves()
            duration = 3 * np.sqrt(level)
            tree_container.rotate(rotation_quaternion=quaternion_power(quat_rot, power + 1) @ quat_tilt @ quat_start,
                                  begin_time=t0,
                                  transition_time=duration / 2)
            tree_container.rotate(rotation_quaternion=quaternion_power(quat_rot, power + 2) @ quat_tilt @ quat_start,
                                  begin_time=t0 + duration / 2,
                                  transition_time=duration / 2)
            power += 2
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 0.8 * duration / len(old_nodes) / 3
            d_phi = 2 * np.pi / len(old_nodes)
            d3phi = d_phi / 3
            if level == flat_levels + 1:
                text_size = 'small'
            else:
                text_size = 'tiny'
            for i, parent in enumerate(old_nodes):
                phi = parent.v
                letter = parent.content
                letter = previous_letter(letter)
                for j in range(-1, 2):
                    phi_j = phi + d3phi * j
                    location = paraboloid.embedding(level + 1, phi_j)
                    n = paraboloid.unit_n(level + 1, phi_j)
                    location += 0.01 * n
                    q = paraboloid.local_frame_quaternion(level + 1, phi_j)
                    color = leaves[self.disc_counter].color
                    label = SimpleTexBObject(letter, aligned='center',
                                             location=location,
                                             rotation_quaternion=q @ q0, color=parent.color,
                                             brighter=get_brighter(color, level - 1), text_size=text_size)
                    ibpy.set_parent(label, tree_container)
                    label.write(begin_time=t0, transition_time=np.maximum(0.1, dt))
                    self.disc_from_circle(leaves[self.disc_counter].circle, begin_time=t0, transition_time=dt,
                                          color=color,
                                          offset=level, brighter=level - 1)
                    connection = Curve([
                        lambda t: paraboloid.embedding(interpol((level - 1) + 1,
                                                                (level - 1) + 1.9, t),
                                                       interpol(phi, phi_j, t)),
                    ],
                        domain=[0.1, 0.95],
                        num_points=int(res / 5),
                        color='text', thickness=0.0625, extrude=0,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    ibpy.set_parent(connection, tree_container)
                    node = TreeNode(letter, label, connection=connection, color=parent.color, u=level + 1, v=phi_j)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt * 1.25
                    letter = next_letter(letter)
            level_node_map[level] = nodes
            t0 += 1

        tree_container.disappear(begin_time=t0)
        camera_empty.move_to(target_location=[0, 0, 0], begin_time=t0)
        camera_circle.move_to(target_location=[0, 0, 4.9261], begin_time=t0)

        for level in range(flat_levels + 3, 8):
            dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=level)
            dfs.generate_tree()
            leaves = dfs.get_leaves()
            duration = 3 * np.sqrt(level)
            dt = 0.8 * duration / len(leaves)
            for leave in leaves:
                self.disc_from_circle(leave.circle, begin_time=t0,
                                      color=leave.color,
                                      offset=level, brighter=level - 1)

                t0 += dt * 1.25
            t0 += 1

        # spider action

        t0 += 0.5
        duration = 30

        eps = 0.001
        average = 100
        scale = 2
        curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)
        fx2 = lambda x: function_from_complex_list(curve, x, scale=scale, average=average)

        fractal = Function([fx], None, domain=[0, 1], mode='PARAMETRIC', name='a_full',
                           location=[1, 0, 0.78],
                           rotation_euler=[0, 0, np.pi / 2],
                           color='important', extrude=0.01, thickness=0.01, metallic=1, roughness=0,
                           num_points=resolution, emission=1)

        average_fractal = Function([fx2], None, domain=[0, 1], mode='PARAMETRIC',
                                   location=[1, 0, 0.76],
                                   rotation_euler=[0, 0, np.pi / 2],
                                   name='average', color='example', extrude=0, thickness=0,
                                   numpoints=resolution / average)

        camera_circle.move(direction=[0, 0, -2], begin_time=t0, transition_time=duration / 2)
        camera_circle.move(direction=[0, 0, 2], begin_time=t0 + duration / 2, transition_time=duration / 2)
        ibpy.camera_follow(camera_circle, begin_time=t0, transition_time=duration, initial_value=0, final_value=1)
        fractal.grow(begin_time=t0, transition_time=duration)

        spider = AnimBObject('Spider', object_name='widow', armature='Armature', color='joker', scale=1,
                             emission=0.75)
        spider.armature.ref_obj.scale = [0.02, 0.02, 0.02]
        spider.appear(begin_time=t0, transition_time=2)

        spider.armature.follow(average_fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')

        t0 += duration

        print("finished at ", t0)

    def flyby(self):
        cues = self.sub_scenes['flyby']
        t0 = 0  # cues['start']

        eps = 0.001
        curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x)
        fx_conj = lambda x: Vector([fx(x).x, -fx(x).y, 0])

        # curve_length = length_of_curve(fx, domain=[0, 1], resolution=resolution)

        # hdri background
        set_hdri_background('Carina_nircam_final', ext='tif', transparent=True)
        set_hdri_strength(0)
        set_hdri_strength(1, begin_time=0.5, transition_time=1)

        # after effects
        set_exposure(1.5)
        fractal = Function([fx_conj], None, domain=[0, 1], mode='PARAMETRIC',
                           color='example', extrude=0.0005, thickness=0.0001, num_points=resolution, emission=0.3)
        t0 += 0.5
        duration = 60
        fractal.grow(begin_time=t0, transition_time=duration)

        tracer = Sphere(0.005, color='joker', emission=3)
        tracer.grow(begin_time=t0 - 0.5, transition_time=0.5)
        tracer.follow(fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                      forward_axis='TRACK_NEGATIVE_Y')

        tracer.disappear(begin_time=t0 + 8 / 9 * duration, transition_time=1 / 9 * duration)

        # setup camera motion
        # set rotation to have start of motion at the origin and counter-clockwise direction
        circle = BezierCircle(radius=0.5, location=Vector([0, 0, 0.5]), rotation_euler=[0, np.pi, np.pi],
                              name='CameraPath')
        circle.appear(transition_time=0)
        ibpy.set_camera_follow(target=circle)
        ibpy.camera_follow(circle, 0, 1, begin_time=t0, transition_time=duration)
        ibpy.set_camera_location(location=Vector())

        # setup empty and circle for the empty
        circle_empty = BezierCircle(radius=0.35, location=Vector([0, -0.5, 0]), rotation_euler=[0, np.pi, np.pi],
                                    name='EmptyPath')
        circle_empty.appear(transition_time=0)

        empty = EmptyCube(name='CameraEmpty', scale=[0.01, 0.01, 0.01])
        empty.appear(transition_time=0)
        ibpy.set_follow(empty, circle_empty)
        ibpy.follow(empty, circle_empty, 0, 1, begin_time=t0, transition_time=duration)
        set_camera_view_to(empty, targetZ=True)

        ibpy.set_camera_lens(65)
        ibpy.camera_zoom(10, begin_time=t0 + 4 / 15 * duration, transition_time=1 / 9 * duration)
        ibpy.camera_zoom(60, begin_time=t0 + 11 / 18 * duration, transition_time=1 / 9 * duration)
        ibpy.camera_zoom(37, begin_time=t0 + 7 / 9 * duration, transition_time=2 / 9 * duration)

        circle.move(direction=[0, -0.64, 1.5], begin_time=t0 + 7 / 9 * duration, transition_time=2 / 9 * duration)
        circle_empty.move(direction=[0, -0.25, 0], begin_time=t0 + 7 / 9 * duration, transition_time=2 / 9 * duration)

        # show spheres
        all_points = []
        all_curvatures = []
        max_curvature = 100
        # create spheres up to a given resolution
        N = 10
        points = read_data("centers.dat")
        curvatures = read_data("curvatures.dat")

        for center, curvature in zip(points, curvatures):
            for i in set(range(0, N + 1)) | set(range(-N, 0)):
                M, R = invert(center + i, 1 / curvature)
                if curvature < max_curvature and 1 / R < max_curvature:
                    all_points.append(center + i)  # translate to different sectors of the band
                    all_curvatures.append(curvature)
                    if np.imag(center) != 0.5:
                        all_points.append(np.real(center) + i + 1j - np.imag(center) * 1j)  # mirror along the half line
                        all_curvatures.append(curvature)

        multi_sphere = MultiSphere(all_points, all_curvatures, name="ApollonianPackage", mesh_type='ico',
                                   color='vertex_color', max_curvature=max_curvature)
        multi_sphere.appear(begin_time=t0 + 6 / 9 * duration, transition_time=3 / 9 * duration)

    def zoom(self):
        cues = self.sub_scenes['zoom']
        t0 = 0

        ibpy.set_camera_location(location=[-30, 0, 150])
        ibpy.set_camera_lens(lens=27)

        duration = 15
        camera_empty = EmptyCube(location=[0, 50, 0])
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_track_influence(ibpy.get_camera(), camera_empty, influence=0.75, begin_time=t0 + 0.2)

        ibpy.set_camera_rotation(rotation=[0, 0, np.pi / 2])
        eps = 0.0000625
        scale = 100
        curve = read_complex_data("apollonian_partial.dat")
        res = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)

        fractal = Function([fx], None, domain=[0, 0.99], mode='PARAMETRIC', name='full',
                           color='example', extrude=0.1, thickness=0.01, num_points=res, emission=1)

        fractal.grow(begin_time=t0, transition_time=0)

        t0 += 0.5
        camera_empty.move_to(target_location=[7.14, 50.01, 0.0], begin_time=t0, transition_time=duration / 10)
        ibpy.camera_move(shift=[37.08, 48.5, 0], begin_time=t0, transition_time=duration / 10)
        ibpy.camera_zoom(lens=2000, begin_time=t0, transition_time=duration)
        t0 += duration

        print("final time: ", t0)

    def create_fundamental_domain(self):
        cues = self.sub_scenes['create_fundamental_domain']
        t0 = 0  # cues['start']

        sphere = Sphere(1 / 2, mesh_type="uv", resolution=10, location=[0, 0, 1 / 2])
        t0 = 0.5 + sphere.grow(begin_time=t0)

        half_sphere = HalfSphere(1 / 2, location=[0, 0, 1 / 2], resolution=100, solid=0.01, offset=0)
        t0 = half_sphere.appear(begin_time=t0)

        rs = [1, 1, 0.25]
        centers = [Vector([1, -1, 0]), Vector([-1, -1, 0]), Vector([0, -0.25, 0])]

        for r, c in zip(rs, centers):
            td = TransformedDisc(solid=0.01, offset=0, radius=r, location=c, resolution=100,
                                 transformation=lambda x: riemann(x))
            td.appear(begin_time=t0)

    def coding(self):
        cues = self.sub_scenes['coding']
        t0 = 0  # cues['start']

        filename = os.path.join(LOC_FILE_DIR,"minimal/model.py")
        cp = CodeParser(filename)

        display = CodeDisplay(cp,class_index=0,flat=True,location=Vector([-7,0,0]),scales=[4.1,6])
        t0=0.5+cp.write(display,class_index=0, begin_time=t0,transition_time=30,indent=0.25)

        display = CodeDisplay(cp, class_index=1, flat=True, location=Vector([1,0,4.5]), scales=[3.5, 1.5],number_of_lines=8)
        t0 = 0.5 + cp.write(display, class_index=1, begin_time=t0, transition_time=15, indent=0.25)

        display = CodeDisplay(cp, class_index=2, flat=True, location=Vector([3, 0, -2]), scales=[5.75, 4])
        t0 = 0.5 + cp.write(display, class_index=2, begin_time=t0, transition_time=15, indent=0.25)

        display = CodeDisplay(cp, class_index=3, flat=True, location=Vector([7.25, 0, 4.5]), scales=[2.5, 1.5])
        t0 = 0.5 + cp.write(display, class_index=3, begin_time=t0, transition_time=15, indent=0.25)

        print("finished ",t0)

    def towards_apollonian(self):
        cues = self.sub_scenes['towards_apollonian']
        self.disc_counter=0
        t0 = 40  # cues['start']

        ibpy.set_camera_location(location=[-1.64,-1.08,0.67])
        ibpy.set_camera_rotation(rotation=[np.pi/180*70,0,304*np.pi/180])

        # hdri background
        set_hdri_background('belfast_sunset_puresky_4k', ext='exr', transparent=False)
        set_hdri_strength(1, begin_time=t0, transition_time=1)

        duration = 0
        max = 7
        min = 2
        discs = 0
        all_discs = []
        old_circles = []
        for c in range(min,max):
            discs +=4*pow(3,c-1)

        tb = 2.001
        ta = 2.001

        for level in range(min, max):
            dfs = DepthFirstSearchByLevel(GrandMasRecipe,ta=ta,tb=tb, max_level=level)
            dfs.generate_tree()
            leaves = dfs.get_leaves()
            dt = duration / discs
            for leave in leaves:
                disc = self.disc_from_circle2(leave.circle,
                                             color=leave.color,
                                             offset=level, brighter=level-1,height=0.02)
                t0 = disc.appear(begin_time=t0, transition_time=0)+dt # disc has to appear instantly, otherwise alpha will destroy transparency
                all_discs.append(disc)
                old_circles.append(leave.circle)

        delta = 0.5
        step_duration=2
        while tb>2.002:
            tb-=delta
            new_old_circles= []
            self.disc_counter=0
            if tb==2.001:
                    step_duration=5
            for level in range(min, max):
                dfs = DepthFirstSearchByLevel(GrandMasRecipe, ta=ta, tb=tb, max_level=level)
                dfs.generate_tree()
                leaves = dfs.get_leaves()
                for leave in leaves:
                    old_circle=old_circles[self.disc_counter]
                    new_circle =leave.circle
                    scale = new_circle.r/old_circle.r
                    move = new_circle.c-old_circle.c
                    move = Vector([np.real(move),np.imag(move),0])
                    disc = all_discs[self.disc_counter]
                    disc.rescale(rescale=[scale,scale,1],begin_time=t0,transition_time=step_duration)
                    disc.move(direction=move,begin_time=t0,transition_time=step_duration)
                    new_old_circles.append(new_circle)
                    self.disc_counter+=1
            old_circles = new_old_circles
            t0+=step_duration
            print(tb,"finished at ",t0)

        # eps = 0.001
        # average = 100
        # scale = 2
        # curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        # resolution = int(10 / eps)
        #
        # height=(max-1)*0.022
        # fx = lambda x: function_from_complex_list(curve, x, scale=scale)
        # fx2 = lambda x: function_from_complex_list(curve, x, scale=scale, average=average)
        #
        # fractal = Function([fx], None, domain=[0, 1], type='PARAMETRIC', name='a_full',
        #                    location=[1, 0, height],
        #                    rotation_euler=[0, 0, np.pi / 2],
        #                    color='important', extrude=0.005, thickness=0.01, metallic=1, roughness=0,
        #                    num_points=resolution, emission=0)
        #
        # average_fractal = Function([fx2], None, domain=[0, 1], type='PARAMETRIC',
        #                            location=[1, 0, height],
        #                            rotation_euler=[0, 0, np.pi / 2],
        #                            name='average', color='example', extrude=0, thickness=0,
        #                            numpoints=resolution / average)
        #
        # t0 += 0.5
        # duration = 15
        # fractal.grow(begin_time=t0, transition_time=duration)
        #
        # spider = AnimBObject('Spider', object_name='widow', armature='Armature', color='joker', scale=1,
        #                      emission=0.75)
        # spider.armature.ref_obj.scale = [0.02, 0.02, 0.02]
        # spider.appear(begin_time=t0, transition_time=2)
        #
        # spider.armature.follow(average_fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
        #                        forward_axis='TRACK_NEGATIVE_Y')
        #
        # t0 += duration
        # spider.disappear(begin_time=t0,transition_time=0.1)
        # t0+=2
        #
        # fractal.disappear(begin_time=t0,transition_time=1)

        delta = 0.5
        while ta < 6.002:
            ta += delta
            new_old_circles = []
            self.disc_counter = 0
            if ta == 2.001:
                step_duration = 5
            else:
                step_duration=2
            for level in range(min, max):
                dfs = DepthFirstSearchByLevel(GrandMasRecipe, ta=ta, tb=tb, max_level=level)
                dfs.generate_tree()
                leaves = dfs.get_leaves()
                for leave in leaves:
                    old_circle = old_circles[self.disc_counter]
                    new_circle = leave.circle
                    scale = new_circle.r / old_circle.r
                    move = new_circle.c - old_circle.c
                    move = Vector([np.real(move), np.imag(move), 0])
                    disc = all_discs[self.disc_counter]
                    disc.rescale(rescale=[scale, scale, 1], begin_time=t0, transition_time=step_duration)
                    disc.move(direction=move, begin_time=t0, transition_time=step_duration)
                    new_old_circles.append(new_circle)
                    self.disc_counter += 1
            old_circles = new_old_circles
            t0 += step_duration
            print(ta, "finished at ", t0)
        print("finished at ", t0)

    def apollonian_adaptive(self):
        cues = self.sub_scenes['apollonian_adaptive']
        t0=0

        self.disc_counter = 0
        cues = self.sub_scenes['apollonian_full']
        t0 = 0.5  # cues['start']
        ibpy.set_camera_location(location=[0, 0, 0])
        ibpy.set_camera_rotation(rotation=[np.pi / 6, 0, 0])
        camera_empty = EmptyCube(location=[0, 0, 0])
        camera_radius = 1.4*Vector([0.02954, -3.1279, 0])
        camera_circle = BezierCircle(radius=np.sqrt(camera_radius.dot(camera_radius)), rotation_euler=[0,np.pi/2,0])

        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_follow(camera_circle)

        for level in range(1,60):
            dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=level)
            if level==1:
                dfs.generate_tree()
                leaves = dfs.get_leaves()
            else:
                leaves = dfs.generate_next_level(level,leaves)
            leaves = list(filter(lambda x: x.circle.r>0.01, leaves))
            duration =30
            if level>1:
                if len(leaves)>0:
                    dt = duration / 2000
                    for leave in leaves:
                        disc = self.disc_from_circle(leave.circle, begin_time=t0,
                                                     color=leave.color,name = leave.word,
                                                     offset=level, brighter=0.1/leave.circle.r,height=0.05)
                        disc.appear(begin_time=t0, transition_time=dt)
                        t0 += dt
                else:
                    print("last level: ",level)
                    break

        ibpy.camera_follow(camera_circle,initial_value=0.125,final_value=0.2499,begin_time=0.5*duration,transition_time=0.8*duration)
        ibpy.camera_zoom(lens=40,begin_time=1.3*duration,transition_time = 0.5*duration)

        print("finished at ",t0," after ",self.disc_counter," Discs")
        # bring the spider in action


    def finale(self):
        cues = self.sub_scenes['finale']
        t0 = 0  # cues['start']

        duration = 30
        ibpy.set_camera_location(location=Vector())
        camera_circle = BezierCircle(radius=100,location=[-98,0,0])
        ibpy.set_camera_follow(camera_circle)
        camera_empty=EmptyCube()
        ibpy.set_camera_view_to(camera_empty)

         # hdri background
        set_hdri_background('belfast_sunset_puresky_4k', ext='exr', transparent=False)
        set_hdri_strength(1, begin_time=t0, transition_time=1)

        group = get_group("CoxH3")
        r5 = np.sqrt(5)
        starting_point = Vector([0,np.sqrt(1/6*(3+r5)),np.sqrt(1/6*(3-r5))])

        dodecahedron = Polyhedron.from_group(group, starting_point*5.2, coordinate_system=None, edge_radius=0.01,vertex_colors=['gold'])
        t0=1+dodecahedron.grow(0, begin_time=t0, transition_time=3)

        dodecahedron.rotate(rotation_euler=[np.pi/2,-np.pi/180*31.8,194*np.pi/180], begin_time=t0,transition_time=duration)
        kleinian_functions = [kleinian2,kleinian1,kleinian3,kleinian4,kleinian5,
                              kleinian6,kleinian7,kleinian8,kleinian10,kleinian9,kleinian11,kleinian12]

        i=0
        for face in dodecahedron.faces:
            if i in range(12):
                kleinian_functions[i](face,t0,duration)
            i=i+1

        t0=t0+duration
        dodecahedron.move(direction=[0, 0, -6.51], begin_time=t0, transition_time=duration/2)
        t0=1+dodecahedron.unfold(fraction=1,begin_time=t0-duration/2,transition_time=duration)

        ibpy.camera_follow(camera_circle, initial_value=-0.215, final_value=-0.25, begin_time=t0, transition_time=duration)
        camera_empty.move(direction=[0,0,-6.51],begin_time=t0+duration/2,transition_time=duration/2)
        camera_circle.move(direction=[0,0,10],begin_time=t0+duration/2,transition_time=duration/2)
        t0=dodecahedron.rotate(rotation_euler=[np.pi/2,-31.8*np.pi/180,233*np.pi/180],begin_time=t0+duration/2,transition_time=duration/2)
        print("finished ",t0)

    def groups1b(self):
        cues = self.sub_scenes['groups1b']
        t0 = 0  # cues['start']

        level_node_map = {}

        am = ApollonianModel()

        polar = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: 0,  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: 0,
            lambda u, v: 0
        )

        tree_objects = []

        shift = Vector()
        identity = SimpleTexBObject("1", location=embedding(0, 0),
                                    aligned='center',
                                    color='text',
                                    rotation_euler=[0, 0, 0], text_size='huge')
        identity.write(begin_time=t0 + 0.5, transition_time=0.5)
        tree_objects.append(identity)
        root = TreeNode("I", identity)
        level_node_map[0] = [root]
        t0 += 1.5

        res = 20
        phi0 = -np.pi
        dphi = np.pi / 2

        # level = 1
        level_letters = []
        letter = 'a'
        for i in range(4):
            level_letters.append(letter)
            letter = next_letter(letter)

        nodes = []

        dt = 1
        r2 = np.sqrt(2)
        ir2 = 1 / r2
        q0 = Quaternion([ir2, 0, 0, ir2])

        for i, l in enumerate(level_letters):
            phi = phi0 + i * dphi
            q = polar.local_frame_quaternion(2, phi)
            location = polar.embedding(2, phi) + shift
            label = SimpleTexBObject(l, aligned='center',
                                     location=location,
                                     rotation_quaternion=q0 @ q,
                                     name=l + '_1', color=am.colors[i],
                                     text_size='huge')
            label.write(begin_time=t0, transition_time=dt)
            tree_objects.append(label)
            connection = Curve([lambda t: partial(polar.embedding, v=phi)(2 * t)], domain=[0.2, 0.8],
                               num_points=res, color=am.colors[i], thickness=0.35, name='Connection_' + l + '_1',
                               location=shift)
            tree_objects.append(connection)
            node = TreeNode(l, label, connection, color=am.colors[i])
            node.connect(begin_time=t0, transition_time=dt)
            node.parent = root
            nodes.append(node)
            t0 += (dt + 0.5)

        level_node_map[1] = nodes

        offset = phi0
        for level in range(2, 3):
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 1
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = i * dphi
                letter = parent.content
                letter = previous_letter(letter)
                index = level_letters.index(letter)
                for j in range(-1, 2):
                    phi_j = offset + phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += (0.01 * normal + shift)
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    label = SimpleTexBObject(letter + parent.content, aligned='center',
                                             location=location, name=letter + parent.content + "_2",
                                             rotation_quaternion=q0 @ q, color=[am.colors[index], parent.color],
                                             text_size='huge')
                    tree_objects.append(label)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: polar.embedding(interpol(2 * (level - 1) + 0.2,
                                                           2 * (level - 1) + 1.8, t),
                                                  interpol(offset + phi, phi_j, t))
                    ],
                        domain=[0.1, 0.9],
                        num_points=res,
                        color=am.colors[index], thickness=0.25, location=shift,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    tree_objects.append(connection)
                    node = TreeNode(letter + parent.content, label, connection=connection,
                                    color=[am.colors[index], parent.color])
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += (dt + 0.5)
                    letter = next_letter(letter)
                    index = level_letters.index(letter)
            offset -= d3phi
            level_node_map[level] = nodes

        tree = BObject(children=tree_objects, location=[0, 0, 0], rotation_euler=[np.pi / 2, 0, 0], scale=1)
        tree.appear()

        t0 += 0.5

        for level in range(3, 4):
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 1 / 3
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = i * dphi
                letter = parent.content[0]
                letter = previous_letter(letter)
                index = level_letters.index(letter)
                for j in range(-1, 2):
                    phi_j = offset + phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += (0.01 * normal + shift)
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    label = SimpleTexBObject(letter + parent.content, aligned='center',
                                             location=location, name=letter + parent.content + "_2",
                                             rotation_quaternion=q0 @ q,
                                             color=flatten([[am.colors[index]], parent.color]))
                    tree_objects.append(label)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: polar.embedding(interpol(2 * (level - 1) + 0.2,
                                                           2 * (level - 1) + 1.8, t),
                                                  interpol(offset + phi, phi_j, t))
                    ],
                        domain=[0.1, 0.9],
                        num_points=res,
                        color=am.colors[index], thickness=0.25, location=shift,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    tree_objects.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += (dt + 0.5)
                    letter = next_letter(letter)
                    index = level_letters.index(letter)
            offset -= d3phi
            level_node_map[level] = nodes

        tree = BObject(children=tree_objects, location=[0, 0, 0], rotation_euler=[np.pi / 2, 0, 0], scale=1)
        tree.appear()

        print("finished ", t0)

    def groups2(self):
        cues = self.sub_scenes['groups2']
        t0 = 0  # cues['start']

        ibpy.set_sun_light(location=[-3, 20, 35])
        ibpy.set_camera_location(location=Vector())
        circle = BezierCircle(location=[0.85, 0, 4], radius=4)
        empty = EmptyCube(location=[0.85, 0, 0])
        ibpy.set_camera_view_to(empty)
        ibpy.set_camera_follow(circle)

        coords = CoordinateSystem(dim=2, lengths=[4, 4], domains=[[-2, 2], [-2, 2]], radii=[0.015, 0.015],
                                  all_n_tics=[4, 4],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 1), np.arange(-2, 2.1, 1)],
                                  colors=['drawing', 'drawing'], label_colors=['drawing', 'drawing'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane')
        coords.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=t0, transition_time=0)
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=2)
        coords.axes[0].labels[2].ref_obj.location = [0.4, 0, -0.20651]

        am = ApollonianModel()
        a = am.get_generators()[0]
        b = am.get_generators()[1]
        A = am.get_generators()[2]
        B = am.get_generators()[3]

        trafos = [
            lambda x: inv_riemann(x + Vector([0, 0, 1 / 2])) - Vector([0, 0, 1 / 2]),
            lambda x: moebius_vec(a, x),
            lambda x: moebius_vec(b, x),
            lambda x: moebius_vec(A, x),
            lambda x: moebius_vec(B, x)
        ]

        fundamental_domain = BObject.from_file(filename='FundamentalDomain', color='text', location=[0, 0, 1 / 2],
                                               solid=0.04, offset=0)
        fundamental_domain.appear(begin_time=t0)

        t0 = 0.5 + ibpy.camera_follow(circle, initial_value=0, final_value=1, begin_time=t0, transition_time=3)
        start_tree = t0
        t0 = 0.5 + fundamental_domain.transform_mesh(trafos[0], begin_time=t0)

        fundamental_domain_flat = BObject.from_file(filename='FundamentalDomainFlat', color='text', solid=0.038,
                                                    offset=0)
        fundamental_domain_flat.appear(begin_time=t0, transition_time=0)
        fundamental_domain.disappear(begin_time=t0, transition_time=0)

        copies = [
            fundamental_domain_flat.copy(color='text', hidden=True, clear_animation_data=True),
            fundamental_domain_flat.copy(color='text', hidden=True, clear_animation_data=True),
            fundamental_domain_flat.copy(color='text', hidden=True, clear_animation_data=True),
            fundamental_domain_flat.copy(color='text', hidden=True, clear_animation_data=True)
        ]

        for i in range(4):
            copies[i].appear(begin_time=t0)
            copies[i].change_color(new_color=am.colors[i], begin_time=t0)
            copies[i].move(direction=[0, 0, 0.03], begin_time=t0)
            t0 = 0.5 + copies[i].transform_mesh(trafos[1 + i], begin_time=t0)

        coords.disappear(begin_time=t0)

        for i in range(4):
            copies2 = [
                copies[i].copy(color=am.colors[i], hidden=True, clear_animation_data=True),
                copies[i].copy(color=am.colors[i], hidden=True, clear_animation_data=True),
                copies[i].copy(color=am.colors[i], hidden=True, clear_animation_data=True)
            ]
            for j in range(3):
                if j == 1:
                    index = i
                elif j == 0:
                    index = i - 1
                    if index < 0:
                        index += 4
                else:
                    index = (i + 1) % 4

                copies2[j].appear(begin_time=t0)
                copies2[j].move(direction=[0, 0, 0.03], begin_time=t0)
                copies2[j].change_color(new_color=am.colors[i], begin_time=t0)
                t0 = 0.5 + copies2[j].transform_mesh(
                    lambda x: moebius_vec(am.get_generators()[index], x),
                    begin_time=t0,
                )

        display = Display(scales=[1, 1], location=[3.05, 0.41, 0.69], rotation_euler=[np.pi / 4, 0, 0], flat=True)
        t0 = start_tree - 1.5
        display.appear(begin_time=t0)

        t0 = start_tree
        level_node_map = {}

        polar = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: 0,  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: 0,
            lambda u, v: 0
        )

        tree_objects = []

        shift = Vector()
        identity = SimpleTexBObject("1", location=embedding(0, 0),
                                    aligned='center',
                                    color='text',
                                    rotation_euler=[0, 0, 0], text_size='huge')
        identity.write(begin_time=t0 + 0.5, transition_time=0.5)
        tree_objects.append(identity)
        root = TreeNode("I", identity)
        level_node_map[0] = [root]
        t0 += 1.5

        res = 20
        phi0 = -np.pi
        dphi = np.pi / 2

        # level = 1
        level_letters = []
        letter = 'a'
        for i in range(4):
            level_letters.append(letter)
            letter = next_letter(letter)

        nodes = []

        dt = 1
        r2 = np.sqrt(2)
        ir2 = 1 / r2
        q0 = Quaternion([ir2, 0, 0, ir2])
        colors = BlenderModel().colors
        for i, l in enumerate(level_letters):
            phi = phi0 + i * dphi
            q = polar.local_frame_quaternion(2, phi)
            location = polar.embedding(2, phi) + shift
            label = SimpleTexBObject(l, aligned='center',
                                     location=location,
                                     rotation_quaternion=q0 @ q,
                                     name=l + '_1', color=am.colors[i],
                                     outlined='text',
                                     text_size='huge')
            label.write(begin_time=t0, transition_time=dt)
            tree_objects.append(label)
            connection = Curve([lambda t: partial(polar.embedding, v=phi)(2 * t)], domain=[0.2, 0.8],
                               num_points=res, color=am.colors[i], thickness=0.35, name='Connection_' + l + '_1',
                               location=shift)
            tree_objects.append(connection)
            node = TreeNode(l, label, connection, color=am.colors[i])
            node.connect(begin_time=t0, transition_time=dt)
            node.parent = root
            nodes.append(node)
            t0 += (dt + 0.5)

        level_node_map[1] = nodes

        offset = phi0
        for level in range(2, 3):
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 1
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = i * dphi
                letter = parent.content
                letter = previous_letter(letter)
                index = level_letters.index(letter)
                for j in range(-1, 2):
                    phi_j = offset + phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += (0.01 * normal + shift)
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    label = SimpleTexBObject(letter + parent.content, aligned='center',
                                             location=location,
                                             rotation_quaternion=q0 @ q, color=[am.colors[index], parent.color],
                                             outlined='text', text_size='huge')
                    tree_objects.append(label)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: polar.embedding(interpol(2 * (level - 1) + 0.2,
                                                           2 * (level - 1) + 1.8, t),
                                                  interpol(offset + phi, phi_j, t))
                    ],
                        domain=[0.1, 0.9],
                        num_points=res,
                        color=am.colors[index], thickness=0.25, location=shift,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    tree_objects.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += (dt + 0.5)
                    letter = next_letter(letter)
                    index = level_letters.index(letter)
            offset -= d3phi
            level_node_map[level] = nodes

        tree = BObject(children=tree_objects, location=[3.05, 0.4, 0.72], rotation_euler=[np.pi / 4, 0, 0], scale=0.2)
        tree.appear()

        print("finished ", t0)

    def groups3(self):
        cues = self.sub_scenes['groups3']
        t0 = 0.5  # cues['start']

        # hdri background
        # set_hdri_background('belfast_sunset_puresky_4k', ext='exr', transparent=True)
        # set_hdri_strength(1, begin_time=t0, transitions_time=0.1)

        ibpy.set_sun_light(location=[-3, 20, 35])
        ibpy.set_camera_location(location=Vector())
        circle = BezierCircle(location=[0.85, 0, 4], radius=4)
        empty = EmptyCube(location=[0.85, 0, 0])
        ibpy.set_camera_view_to(empty)
        ibpy.set_camera_follow(circle)

        am = ApollonianModel()
        a = am.get_generators()[0]
        b = am.get_generators()[1]
        A = am.get_generators()[2]
        B = am.get_generators()[3]

        start_t0 = t0
        circles = am.get_circles()
        discs = []
        labels = []
        # first level of discs
        for i, circle in enumerate(circles):
            disc = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[i], solid=0.04,
                         resolution=np.maximum(500, np.minimum(10, 1000 * circle.r)),
                         name='Disc_' + str(i) + '_x_x', metallic=0.75, roughness=0.5)
            disc.grow(begin_time=t0, pivot=z2vec(circle.c))
            t0 = disc.move(direction=[0, 0, 0.03], begin_time=t0)
            discs.append(disc)
            labels.append(am.labels[i])

        t0 += 0.5  # pause

        for i in range(4):
            for j in range(3):
                if j == 1:
                    index = i
                elif j == 0:
                    index = i - 1
                    if index < 0:
                        index += 4
                else:
                    index = (i + 1) % 4

                circle = moebius_on_circle(am.get_generators()[index], am.get_circles()[i])
                disc_new = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[i], solid=0.04,
                                 resolution=np.maximum(100, np.minimum(10, 1000 * circle.r)),
                                 name='Disc_' + str(i) + '_' + str(index) + '_x', metallic=0.75, roughness=0.5)
                disc_new.grow(begin_time=t0, pivot=z2vec(circle.c))
                t0 = disc_new.move(direction=[0, 0, 0.06], begin_time=t0)

        t0 += 0.5  # pause
        level_three_start_time = t0

        # grow discs for level 3
        for i in range(4):

            for j in range(3):
                if j == 1:
                    index = i
                elif j == 0:
                    index = i - 1
                    if index < 0:
                        index += 4
                else:
                    index = (i + 1) % 4
                for k in range(3):
                    if k == 1:
                        index2 = index
                    elif k == 0:
                        index2 = index - 1
                        if index2 < 0:
                            index2 += 4
                    else:
                        index2 = (index + 1) % 4

                    circle = moebius_on_circle(am.get_generators()[index2],
                                               moebius_on_circle(am.get_generators()[index], am.get_circles()[i]))
                    disc_new = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[i], solid=0.04,
                                     resolution=np.maximum(100, np.minimum(10, 1000 * circle.r)), metallic=0.75,
                                     roughness=0.5,
                                     name='Disc_' + str(i) + '_' + str(index) + '_' + str(index2))
                    disc_new.grow(begin_time=t0, pivot=z2vec(circle.c), transition_time=0.3)
                    t0 = disc_new.move(direction=[0, 0, 0.09], begin_time=t0, transition_time=0.3)

        t0 += 0.5

        level_four_start_time = t0
        # grow discs for level 4
        for i in range(4):

            for j in range(3):
                if j == 1:
                    index = i
                elif j == 0:
                    index = i - 1
                    if index < 0:
                        index += 4
                else:
                    index = (i + 1) % 4
                for k in range(3):
                    if k == 1:
                        index2 = index
                    elif k == 0:
                        index2 = index - 1
                        if index2 < 0:
                            index2 += 4
                    else:
                        index2 = (index + 1) % 4
                    for m in range(3):
                        if m == 1:
                            index3 = index2
                        elif m == 0:
                            index3 = index2 - 1
                            if index3 < 0:
                                index3 += 4
                        else:
                            index3 = (index2 + 1) % 4

                        circle = moebius_on_circle(am.get_generators()[index3],
                                                   moebius_on_circle(am.get_generators()[index2],
                                                                     moebius_on_circle(am.get_generators()[index],
                                                                                       am.get_circles()[i])))
                        disc_new = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[i], solid=0.04,
                                         resolution=np.maximum(100, np.minimum(10, 1000 * circle.r)), metallic=0.75,
                                         roughness=0.5,
                                         name='Disc_' + str(i) + '_' + str(index) + '_' + str(index2) + '_' + str(
                                             index3))
                        disc_new.grow(begin_time=t0, pivot=z2vec(circle.c), transition_time=0.1)
                        disc_new.move(direction=[0, 0, 0.12], begin_time=t0, transition_time=0.05)
                        t0 += 0.1
        level_four_end_time = t0

        t0 += 0.5

        # grow discs for level 5
        for i in range(4):

            for j in range(3):
                if j == 1:
                    index = i
                elif j == 0:
                    index = i - 1
                    if index < 0:
                        index += 4
                else:
                    index = (i + 1) % 4
                for k in range(3):
                    if k == 1:
                        index2 = index
                    elif k == 0:
                        index2 = index - 1
                        if index2 < 0:
                            index2 += 4
                    else:
                        index2 = (index + 1) % 4
                    for m in range(3):
                        if m == 1:
                            index3 = index2
                        elif m == 0:
                            index3 = index2 - 1
                            if index3 < 0:
                                index3 += 4
                        else:
                            index3 = (index2 + 1) % 4

                        for n in range(3):
                            if n == 1:
                                index4 = index3
                            elif n == 0:
                                index4 = index3 - 1
                                if index4 < 0:
                                    index4 += 4
                            else:
                                index4 = (index3 + 1) % 4

                            circle = moebius_on_circle(am.get_generators()[index4],
                                                       moebius_on_circle(am.get_generators()[index3],
                                                                         moebius_on_circle(am.get_generators()[index2],
                                                                                           moebius_on_circle(
                                                                                               am.get_generators()[
                                                                                                   index],
                                                                                               am.get_circles()[i]))))
                            disc_new = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[i], solid=0.04,
                                             resolution=np.maximum(100, np.minimum(10, 1000 * circle.r)), metallic=0.75,
                                             roughness=0.5,
                                             name='Disc_' + str(i) + '_' + str(index) + '_' + str(index2) + '_' + str(
                                                 index3) + "_" + str(index4))
                            disc_new.grow(begin_time=t0, pivot=z2vec(circle.c), transition_time=0.04)
                            disc_new.move(direction=[0, 0, 0.15], begin_time=t0, transition_time=0.02)
                            t0 += 0.04
        t0 += 0.5

        # grow discs for level 6
        for i in range(4):

            for j in range(3):
                if j == 1:
                    index = i
                elif j == 0:
                    index = i - 1
                    if index < 0:
                        index += 4
                else:
                    index = (i + 1) % 4
                for k in range(3):
                    if k == 1:
                        index2 = index
                    elif k == 0:
                        index2 = index - 1
                        if index2 < 0:
                            index2 += 4
                    else:
                        index2 = (index + 1) % 4
                    for m in range(3):
                        if m == 1:
                            index3 = index2
                        elif m == 0:
                            index3 = index2 - 1
                            if index3 < 0:
                                index3 += 4
                        else:
                            index3 = (index2 + 1) % 4

                        for n in range(3):
                            if n == 1:
                                index4 = index3
                            elif n == 0:
                                index4 = index3 - 1
                                if index4 < 0:
                                    index4 += 4
                            else:
                                index4 = (index3 + 1) % 4

                            for o in range(3):
                                if o == 1:
                                    index5 = index4
                                elif o == 0:
                                    index5 = index4 - 1
                                    if index5 < 0:
                                        index5 += 4
                                else:
                                    index5 = (index4 + 1) % 4

                                circle = moebius_on_circle(am.get_generators()[index5],
                                                           moebius_on_circle(am.get_generators()[index4],
                                                                             moebius_on_circle(
                                                                                 am.get_generators()[index3],
                                                                                 moebius_on_circle(
                                                                                     am.get_generators()[index2],
                                                                                     moebius_on_circle(
                                                                                         am.get_generators()[
                                                                                             index],
                                                                                         am.get_circles()[i])))))
                                disc_new = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[i], solid=0.04,
                                                 resolution=np.maximum(100, np.minimum(10, 1000 * circle.r)),
                                                 metallic=0.75,
                                                 roughness=0.5,
                                                 name='Disc_' + str(i) + '_' + str(index) + '_' + str(
                                                     index2) + '_' + str(
                                                     index3) + "_" + str(index4) + "_" + str(index5))
                                disc_new.grow(begin_time=t0, pivot=z2vec(circle.c), transition_time=0.01)
                                disc_new.move(direction=[0, 0, 0.18], begin_time=t0, transition_time=0)
                                t0 += 0.01
        t0 += 0.5
        finish_time = t0

        # create tree
        level_node_map = {}
        t0 = start_t0
        polar = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: 0,  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: 0,
            lambda u, v: 0
        )

        bell = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: hull2(u),  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: d_hull2(u),
            lambda u, v: 0
        )

        tree_objects = []

        shift = Vector()
        identity = SimpleTexBObject("1", location=embedding(0, 0),
                                    aligned='center',
                                    color='text',
                                    rotation_euler=[0, 0, 0],
                                    text_size='huge')
        # identity.write(begin_time=0, transition_time=0.5)
        tree_objects.append(identity)
        root = TreeNode("I", identity)
        level_node_map[0] = [root]

        res = 20
        phi0 = -np.pi
        dphi = np.pi / 2

        # level = 1
        level_letters = []
        letter = 'a'
        for i in range(4):
            level_letters.append(letter)
            letter = next_letter(letter)

        nodes = []

        dt = 1
        r2 = np.sqrt(2)
        ir2 = 1 / r2
        q0 = Quaternion([ir2, 0, 0, ir2])

        morph_connections = []
        for i, l in enumerate(level_letters):
            phi = phi0 + i * dphi
            q = polar.local_frame_quaternion(2, phi)
            location = polar.embedding(2, phi) + shift
            label = SimpleTexBObject(l, aligned='center',
                                     location=location,
                                     rotation_quaternion=q0 @ q,
                                     name=l + '_1', color=am.colors[i],
                                     outlined='text',
                                     shadow=False,
                                     text_size='huge')
            label.write(begin_time=t0, transition_time=dt)
            tree_objects.append(label)
            connection = Curve([
                lambda t: partial(polar.embedding, v=phi)(2 * t),
                lambda t: partial(bell.embedding, v=phi)(2 * t),
            ], domain=[0.2, 0.8],
                num_points=res, color=am.colors[i], thickness=0.35, name='Connection_' + l + '_1',
                shadow=False,
                location=shift)
            morph_connections.append(connection)
            tree_objects.append(connection)
            node = TreeNode(l, label, connection, color=am.colors[i], u=2, v=phi)
            node.connect(begin_time=t0, transition_time=dt)
            node.parent = root
            nodes.append(node)
            t0 += dt

        level_node_map[1] = nodes
        t0 += 0.5

        offset = phi0
        for level in range(2, 3):
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 1
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = i * dphi
                letter = parent.content
                letter = previous_letter(letter)
                index = level_letters.index(letter)
                for j in range(-1, 2):
                    phi_j = offset + phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += (0.01 * normal + shift)
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    label = SimpleTexBObject(letter, aligned='center',
                                             location=location,
                                             rotation_quaternion=q0 @ q, color=[parent.color],
                                             text_size='huge',
                                             shadow=False,
                                             outlined='text'
                                             )
                    tree_objects.append(label)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: polar.embedding(interpol(2 * (level - 1) + 0.2,
                                                           2 * (level - 1) + 1.8, t),
                                                  interpol(offset + phi, phi_j, t)),
                        lambda t: bell.embedding(interpol(2 * (level - 1) + 0,
                                                          2 * (level - 1) + 2.05, t),
                                                 interpol(offset + phi, phi_j, t)),
                    ],
                        domain=[0.1, 0.9],
                        num_points=res,
                        color=am.colors[index], thickness=0.25, location=shift,
                        shadow=False,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    tree_objects.append(connection)
                    morph_connections.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color, u=2 * level, v=phi_j)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt
                    letter = next_letter(letter)
                    index = level_letters.index(letter)
            offset -= d3phi
            level_node_map[level] = nodes

        t0 += 0.5
        discs[0].disappear(alpha=0.1, begin_time=t0)
        # morph tree
        for connection in morph_connections:
            connection.next(begin_time=t0)

        for level_nodes in level_node_map.values():
            for node in level_nodes:
                node.bob.move_to(target_location=bell.embedding(node.u, node.v), begin_time=t0)
                q = bell.local_frame_quaternion(node.u, node.v)
                node.bob.rotate(rotation_quaternion=q @ q0, begin_time=t0)

        tree_rotate_time = t0
        t0 += 2
        # level 3
        t0 = level_three_start_time
        for level in range(3, 4):
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 0.3
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = i * dphi
                letter = parent.content
                letter = previous_letter(letter)
                index = level_letters.index(letter)
                for j in range(-1, 2):
                    phi_j = offset + phi + d3phi * j
                    location = bell.embedding(level + 2.4, phi_j)
                    normal = bell.unit_n(level + 2.4, phi_j)
                    location += (0.01 * normal + shift)
                    q = bell.local_frame_quaternion(level + 2.4, phi_j)
                    label = SimpleTexBObject(letter, aligned='center',
                                             location=location,
                                             rotation_quaternion=q @ q0,
                                             color=[parent.color],
                                             text_size='huge',
                                             shadow=False,
                                             outlined='text',
                                             )
                    tree_objects.append(label)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: bell.embedding(interpol((level - 1) + 2.05,
                                                          level + 2.05, t),
                                                 interpol(offset + phi, phi_j, t)),
                    ],
                        domain=[0.2, 1],
                        num_points=res,
                        color=am.colors[index], thickness=0.25, location=shift,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level),
                        shadow=False
                    )
                    tree_objects.append(connection)
                    morph_connections.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color, u=2 * level, v=phi_j)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt
                    letter = next_letter(letter)
                    index = level_letters.index(letter)
            offset -= d3phi
            level_node_map[level] = nodes
        level_three_end_time = t0

        t0 = level_four_start_time
        for level in range(4, 5):
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 0.1
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = i * dphi
                letter = parent.content
                letter = previous_letter(letter)
                index = level_letters.index(letter)
                for j in range(-1, 2):
                    phi_j = offset + phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += (0.01 * normal + shift)
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    label = SimpleTexBObject(letter, aligned='center',
                                             location=location,
                                             rotation_quaternion=q @ q0,
                                             color=[parent.color],
                                             text_size='large',
                                             shadow=False,
                                             outlined='text',
                                             )
                    tree_objects.append(label)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: bell.embedding(interpol(2 * (level - 1) + 0,
                                                          2 * (level - 1) + 2.05, t),
                                                 interpol(offset + phi, phi_j, t)),
                    ],
                        domain=[0, 0.9],
                        num_points=res,
                        color=am.colors[index], thickness=0.25, location=shift,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level),
                        shadow=False,
                    )
                    tree_objects.append(connection)
                    morph_connections.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color, u=2 * level, v=phi_j)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt
                    letter = next_letter(letter)
                    index = level_letters.index(letter)
            offset -= d3phi
            level_node_map[level] = nodes

        tree = BObject(children=tree_objects, location=[3.05, 0.4, 0.72], rotation_euler=[np.pi / 4, 0, 0], scale=0.2)
        tree.appear()
        tree.move_to(target_location=[3, 0, 0], begin_time=tree_rotate_time)
        tree.rotate(rotation_euler=[0, 0, 0], begin_time=tree_rotate_time)

        dt = level_three_end_time - level_three_start_time
        tree.rotate(rotation_euler=[0, 0, -6 * np.pi / 4], begin_time=level_three_start_time + 1 / 3 * dt,
                    transition_time=3 / 4 * dt)
        tree.move_to(target_location=[3, 0, 0.2], begin_time=level_three_start_time, transition_time=dt)

        dt = level_four_end_time - level_four_start_time
        tree.rotate(rotation_euler=[0, 0, -13 * np.pi / 4], begin_time=level_four_start_time,
                    transition_time=dt)

        print("finished ", finish_time)

    def induced_transformation(self):
        cues = self.sub_scenes['induced_transformation']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20)
        display.appear(begin_time=t0)
        t0 += 1

        title = SimpleTexBObject(r"\text{Transformations in the complex plane}", color='example', aligned='center',
                                 text_size='small')
        display.set_title(title)
        title.write(begin_time=t0, transition_time=1.5)
        t0 += 2

        matrix = BMatrix(np.array([['a', 'b'], ['c', 'd']]))
        transform = SimpleTexBObject(r'z\mapsto {a z+b\over c z+d}')
        display.set_cursor_to_start_of_line(2)
        display.add_text(matrix, indent=0.5)
        display.add_text(transform)
        matrix.write(begin_time=t0)
        t0 += 1
        transform.write(letter_set={0, 1, 2, 3, 5, 8, 11}, begin_time=t0, transition_time=0.45)
        t0 += 0.45
        transform.write(letter_set={7}, begin_time=t0, transition_time=0.1)
        t0 += 0.1
        transform.write(letter_set={4, 6, 9, 10}, begin_time=t0, transition_time=0.45)
        t0 += 0.95

        title3 = SimpleTexBObject(r"\text{Scaling:}", color='joker')
        display.write_text_in(title3, indent=0.5, line=4, begin_time=t0)
        t0 += 0.5

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-5, 5], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 10],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-5, 5.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=2)

        t0 += 2.5

        transformations = [
            lambda z: 2 * z,
            lambda z: z,
            lambda z: 2 * z,
            lambda z: 4 * z,
            lambda z: z,
        ]

        annulus = Annulus(r=[0.5, 1], phi=[0, 2 * np.pi], color='joker',
                          coordinate_system=coords,
                          conformal_transformations=transformations,
                          rotation_euler=[np.pi / 2, 0, 0])
        annulus.appear(begin_time=t0 - 1.5)

        matrix = BMatrix(np.array([['2', '0'], ['0', '1']]))
        transform = SimpleTexBObject(r' z\mapsto 2z', name='transform')
        display.set_cursor_to_start_of_line(6)
        display.write_text(matrix, indent=0.5, begin_time=t0)
        t0 += 1
        display.write_text(transform, begin_time=t0)
        t0 += 1.5
        annulus.next_shape(begin_time=t0)
        t0 += 1.5

        title4 = SimpleTexBObject(r"\text{Inverse:}", color='joker')
        display.write_text_in(title4, indent=0.5, line=7.5, begin_time=t0)
        t0 += 0.5

        matrix = BMatrix(np.array([[2, 0], [0, 1]]), after_word=r"^{-1}=", name='m1')
        matrix2 = BMatrix(np.array([[r'{1\over 2}', '0'], ['0', '1']]), name='m2')
        display.set_cursor_to_start_of_line(9)
        display.write_text(matrix, indent=0.5, begin_time=t0)
        t0 += 1.5
        display.write_text(matrix2, begin_time=t0, indent=-3)
        transform2 = SimpleTexBObject(r' z\mapsto {1\over 2}z', name='transform2')
        display.write_text_in(transform2, line=11, begin_time=t0)
        transform2.align(transform, char_index=0, other_char_index=0)
        t0 += 1
        annulus.next_shape(begin_time=t0)
        t0 += 1.5

        title5 = SimpleTexBObject(r"\text{Composition:}", color='joker')
        display.write_text_in(title5, indent=0.5, line=13.5, begin_time=t0)
        t0 += 0.5
        matrix = BMatrix(np.array([[2, 0], [0, 1]]), after_word=r"\cdot", name='m3')
        matrix2 = BMatrix(np.array([[2, 0], [0, 1]]), after_word=r"=", name='m4')
        matrix3 = BMatrix(np.array([[4, 0], [0, 1]]), name='m4')
        display.set_cursor_to_start_of_line(15)
        display.write_text(matrix, indent=0.5, begin_time=t0)
        t0 += 1
        display.write_text(matrix2, begin_time=t0, indent=-2.5)
        t0 += 1
        display.write_text(matrix3, begin_time=t0, indent=-3)
        t0 += 1.5
        transform3 = SimpleTexBObject(r' z\mapsto 2\cdot(2z)', name='transform3')
        display.write_text_in(transform3, line=17, begin_time=t0)
        transform3.align(transform2, char_index=0, other_char_index=0)

        t0 += 1.5
        annulus.next_shape(begin_time=t0)
        t0 += 1.1
        annulus.next_shape(begin_time=t0)
        t0 += 2

        display.turn(begin_time=t0)
        annulus.next_shape(begin_time=t0)
        t0 += 1.5

        print("finished at ", t0)

    def on_the_sphere(self):
        cues = self.sub_scenes['on_the_sphere']
        t0 = 0  # cues['start']

        # hdri background
        set_hdri_background('qwantani_puresky_4k', ext='exr', transparent=True)
        set_hdri_strength(2, begin_time=t0, transition_time=0.1)

        display = Display(number_of_lines=15, shadow=False, location=[-6, 0, 2], scales=[4, 4], flat=True)
        display.appear(begin_time=t0)

        projection_display = Display(scales=[2, 2], location=[6, 0, 3], number_of_lines=2, shadow=False)

        coords = CoordinateSystem(dim=2, lengths=[10, 8], domains=[[-5, 5], [-4, 4]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-4, 4.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[5, 0, -2],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)
        # adjust position of the zero label manually
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        transformations = [
            lambda z: z,
            lambda z: 1 / 2 * z,
            lambda z: 1 / 4 * z,
            lambda z: 1 / 8 * z,
        ]

        annulus = Annulus(r=[0.5, 1], phi=[0, 2 * np.pi], color='joker',
                          coordinate_system=coords,
                          conformal_transformations=transformations,
                          rotation_euler=[np.pi / 2, 0, 0], resolution=40, solid=0.025, offset=0, smooth=2)
        annulus.appear(begin_time=t0, transition_time=0)
        annulus.next_shape(begin_time=t0, transition_time=0)
        t0 += 0.5

        coords.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=t0, compensate=True)
        coords.zoom(4, begin_time=t0)
        coords.move(direction=[0, 0, -2], begin_time=t0)
        t0 += 1.5

        sphere = Sphere(r=1.99, location=[5, 0, -2], resolution=5, color='fake_glass_text', roughness=0.01,
                        metallic=0.2)
        sphere.grow(begin_time=t0)
        t0 += 2.5

        stereo = SimpleTexBObject(r'\text{Stereographic projection}', color='example', emission=1, aligned='center')
        display.write_title(stereo, begin_time=t0)
        t0 += 1.5

        projection = BMatrix(np.array([[r'{\Re{z}\over 1+z\bar{z}}'], [r'\rule{0em}{3ex}{\Im{z}\over 1+ z\bar{z}}'],
                                       [r'\rule{0em}{3ex}{z\bar{z}\over 1+z\bar{z}}']]), pre_word=r'P\colon z \mapsto ',
                             color='example')
        display.write_text_in(projection, line=3, indent=3, begin_time=t0)

        transformations = [
            lambda v: riemann(v),
            lambda v: riemann(1 / 2 * v),
            lambda v: riemann(1 / 4 * v),
            lambda v: riemann(1 / 8 * v),
        ]

        end_points = [
            Vector([9, 0, -4]),
            Vector([7, 0, -4]),
            Vector([3, 0, -4]),
            Vector([1, 0, -4]),
        ]
        end_points2 = [
            0.5 * Vector([4, 0, -4]) + Vector([5, 0, 0]),
            0.8 * Vector([2, 0, -4]) + Vector([5, 0, 0]),
            0.8 * Vector([-2, 0, -4]) + Vector([5, 0, 0]),
            0.5 * Vector([-4, 0, -4]) + Vector([5, 0, 0]),
        ]
        rays = []
        for end_point in end_points:
            ray = Cylinder.from_start_to_end(start=Vector([5, 0, 0]), end=end_point, color='example', thickness=0.1,
                                             emission=1)
            ray.grow(begin_time=t0)
            rays.append(ray)
        t0 += 1.5

        projections = [
            SimpleTexBObject("P", color=['example'], aligned='center', text_size='small', name='P1'),
            SimpleTexBObject("PS", color=['example', 'text', 'text'], aligned='center', text_size='small', name='P2'),
        ]

        number = DigitalNumber(number_of_digits=0, value=2, signed=False)
        projection_display.add_text_in(number, line=-1, scale=0.5, indent=9.5)

        projection_display.appear(begin_time=t0)
        for i in range(0, len(projections)):
            projection_display.add_text_in(projections[i], line=-0.5, indent=5, scale=1.5)
        projections[0].write(begin_time=t0, transition_time=0.3)

        annulus2 = Annulus2(transformations=transformations, scale=[4, 4, 4], location=[5, 0, -4], name="Annulus2",
                            resolution=10, solid=0.02, offset=-2)
        annulus2.appear(begin_time=t0, transition_time=0)
        annulus2.next_shape(begin_time=t0)
        for end_point, ray in zip(end_points2, rays):
            ray.move_end_point(target_location=end_point, begin_time=t0)
            ray.disappear(begin_time=t0 + 1, transition_time=0.5)
        t0 += 1.5

        scaling = [
            SimpleTexBObject(r'S\colon z\mapsto \tfrac{1}{2}z', color='text'),
            SimpleTexBObject(r'S^2\colon z\mapsto \tfrac{1}{4} z', color='text'),
            SimpleTexBObject(r'S^3\colon z\mapsto \tfrac{1}{8}z', color='text'),
        ]
        for i in range(0, 3):
            display.write_text_in(scaling[i], line=6 + i, indent=3, begin_time=t0)
            if i > 0:
                scaling[i].align(scaling[0], char_index=2, other_char_index=1)
            annulus.next_shape(begin_time=t0)
            if i == 0:
                projections[i].replace2(projections[i + 1], morphing=True, begin_time=t0)
            annulus2.next_shape(begin_time=t0)
            if i == 1:
                number.write(begin_time=t0, transition_time=0.3)
            if i == 2:
                number.update_single_value(new_value=3, begin_time=t0, transition_time=0.3)
            t0 += 2

        t0 = projection_display.turn(begin_time=t0) + 0.5

        # repeat everything for the inverse mapping
        projections[0].disappear(begin_time=t0)
        projections[1].disappear(begin_time=t0)
        number.disappear(begin_time=t0)

        projections2 = [
            SimpleTexBObject("P", color=['example'], aligned='center', text_size='small', name='P1'),
            SimpleTexBObject("PS", color=['example', 'text', 'text'], aligned='center', text_size='small', name='P2'),
        ]

        transformations = [
            lambda z: z,
            lambda z: 2 * z,
            lambda z: 4 * z,
            lambda z: 8 * z,
            lambda z: 16 * z,
        ]

        annulus3 = Annulus(r=[0.5, 1], phi=[0, 2 * np.pi], color='joker', name='Annulus3',
                           coordinate_system=coords,
                           conformal_transformations=transformations,
                           rotation_euler=[np.pi / 2, 0, 0], resolution=40, solid=0.025, offset=0, smooth=2)
        annulus3.appear(begin_time=t0, transition_time=0)
        annulus3.next_shape(begin_time=t0, transition_time=0)
        t0 += 0.5

        transformations = [
            lambda v: riemann(v),
            lambda v: riemann(2 * v),
            lambda v: riemann(4 * v),
            lambda v: riemann(8 * v),
            lambda v: riemann(16 * v),
        ]

        end_points = [
            Vector([9, 0, -4]),
            Vector([7, 0, -4]),
            Vector([3, 0, -4]),
            Vector([1, 0, -4]),
        ]
        end_points2 = [
            0.5 * Vector([4, 0, -4]) + Vector([5, 0, 0]),
            0.8 * Vector([2, 0, -4]) + Vector([5, 0, 0]),
            0.8 * Vector([-2, 0, -4]) + Vector([5, 0, 0]),
            0.5 * Vector([-4, 0, -4]) + Vector([5, 0, 0]),
        ]
        rays = []
        for end_point in end_points:
            ray = Cylinder.from_start_to_end(start=Vector([5, 0, 0]), end=end_point, color='example', thickness=0.1,
                                             emission=1)
            ray.grow(begin_time=t0)
            rays.append(ray)
        t0 += 1.5

        number2 = DigitalNumber(number_of_digits=0, value=-1, signed=False)
        projection_display.add_text_in_back(number2, line=-1, scale=0.5, indent=10)

        for i in range(0, len(projections)):
            projection_display.add_text_in_back(projections2[i], line=-0.5, indent=3.5, scale=1.5)
        projections2[0].write(begin_time=t0, transition_time=0.3)

        annulus4 = Annulus2(transformations=transformations, scale=[4, 4, 4], location=[5, 0, -4], name="Annulus4",
                            resolution=10, solid=0.02, offset=-1)
        annulus4.appear(begin_time=t0, transition_time=0)
        annulus4.next_shape(begin_time=t0)
        for end_point, ray in zip(end_points2, rays):
            ray.move_end_point(target_location=end_point, begin_time=t0)
            ray.disappear(begin_time=t0 + 1, transition_time=0.5)
        t0 += 1.5

        scaling2 = [
            SimpleTexBObject(r'S^{-1}\colon z\mapsto 2z', color='text'),
            SimpleTexBObject(r'S^{-2}\colon z\mapsto 4z', color='text'),
            SimpleTexBObject(r'S^{-3}\colon z\mapsto 8z', color='text'),
            SimpleTexBObject(r'S^{-4}\colon z\mapsto 16z', color='text'),
        ]

        for i in range(0, 4):
            display.write_text_in(scaling2[i], line=9 + i, indent=3, begin_time=t0)
            scaling2[i].align(scaling[0], char_index=3, other_char_index=1)
            annulus3.next_shape(begin_time=t0)
            if i == 0:
                projections2[i].replace2(projections2[i + 1], morphing=True, begin_time=t0)
            annulus4.next_shape(begin_time=t0)
            if i == 0:
                number2.write(begin_time=t0, transition_time=0.3)
            if i == 1:
                number2.update_single_value(new_value=-2, begin_time=t0, transition_time=0.3, morphing=False)
            if i == 2:
                number2.update_single_value(new_value=-3, begin_time=t0, transition_time=0.3, morphing=False)
            if i == 3:
                number2.update_single_value(new_value=-4, begin_time=t0, transition_time=0.3)
            t0 += 2

        display.disappear(begin_time=t0)
        projection_display.disappear(begin_time=t0)
        annulus.disappear(begin_time=t0)
        t0 = annulus3.disappear(begin_time=t0) + 0.5

        print("finished at ", t0)

    def rotate_sphere(self):
        cues = self.sub_scenes['on_the_sphere']
        t0 = 0  # cues['start']

        # hdri background
        set_hdri_background('qwantani_puresky_4k', ext='exr', transparent=True)
        set_hdri_strength(2, begin_time=t0, transition_time=0)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-5, 5], [-4, 4]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-4, 4.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[5, 0, -2],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)
        # adjust position of the zero label manually
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        t0 += 0.1
        coords.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=t0, transition_time=0.1, compensate=True)
        coords.zoom(4, begin_time=t0, transition_time=0.1)
        coords.move(direction=[0, 0, -2], begin_time=t0, transition_time=0)

        sphere = Sphere(r=1.99, location=[5, 0, -2], resolution=5, color='fake_glass_text', roughness=0.01,
                        metallic=0.2)
        sphere.grow(begin_time=t0, transition_time=0)

        transformations = [
            lambda v: riemann(16 * v),
            lambda v: rotate(riemann(16 * v)),
            lambda v: inv_riemann(rotate(riemann(16 * v))),
            lambda v: inv_riemann(rotate(riemann(8 * v))),
            lambda v: inv_riemann(rotate(riemann(4 * v))),
            lambda v: inv_riemann(rotate(riemann(2.001 * v))),
        ]

        annulus2 = Annulus2(transformations=transformations, scale=[4, 4, 4], location=[5, 0, -2], name="Annulus2",
                            resolution=50,
                            solid=0.01, offset=-1)  # the effect of scaling and location works on top of all the
        # transformations that are preformed with the vertices in the standard coordinate frame

        annulus2.appear(begin_time=t0, transition_time=0)
        annulus2.next_shape(begin_time=t0, transition_time=0.1)

        transformations2 = [
            lambda v: riemann(1 / 8 * v),
            lambda v: rotate(riemann(1 / 8 * v)),
            lambda v: inv_riemann(rotate(riemann(1 / 8 * v))),
            lambda v: inv_riemann(rotate(riemann(1 / 4 * v))),
            lambda v: inv_riemann(rotate(riemann(1 / 2 * v))),
            lambda v: inv_riemann(rotate(riemann(1.001 * v))),
        ]

        annulus = Annulus2(transformations=transformations2, scale=[4, 4, 4], location=[5, 0, -2], name="Annulus",
                           resolution=50, offset=-1,
                           solid=0.01)  # the effect of scaling and location works on top of all the transformations
        # that are preformed with the vertices in the standard coordinate frame

        annulus.appear(begin_time=t0, transition_time=0)
        annulus.next_shape(begin_time=t0, transition_time=0.1)

        # move everything back to the center
        coords.move_to(target_location=[0, 0, -4], begin_time=t0)
        annulus.move(direction=[-5, 0, 0], begin_time=t0)
        sphere.move(direction=[-5, 0, 0], begin_time=t0)
        t0 = annulus2.move(direction=[-5, 0, 0], begin_time=t0) + 0.5

        # rotate
        t0 += 0.5
        rot_box = BObject(children=[annulus, annulus2], location=[0, 0, -2])
        rot_box.appear(begin_time=t0, transition_time=0)
        rot_box.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=t0)
        # annulus2.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=t0, pivot=Vector([0, 0, 0]))
        # annulus.rotate(rotation_euler=[0,np.pi/2,0],begin_time=t0,pivot=Vector([0,0,0]))
        annulus2.next_color_map(begin_time=t0)
        annulus.next_color_map(begin_time=t0)
        t0 = sphere.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=t0)

        # replace the rotation with an instantaneous transformation to have the vertices in the right place for
        # further transformations
        annulus2.next_shape_key(begin_time=t0, transition_time=0)
        annulus2.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)
        annulus.next_shape_key(begin_time=t0, transition_time=0)
        annulus.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)
        rot_box.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0)
        t0 += 0.5

        points = [
            Vector([-16, 0, 0]),
            Vector([-8, 0, 0]),
            Vector([8, 0, 0]),
            Vector([16, 0, 0])
        ]

        points2 = [
            Vector([-1 / 16, 0, 0]),
            Vector([-1 / 8, 0, 0]),
            Vector([1 / 8, 0, 0]),
            Vector([1 / 16, 0, 0])
        ]

        projected_points = [riemann(point) for point in points]
        projected_points2 = [riemann(point) for point in points2]
        rotated_points = [Vector([v.z - 0.5, v.y, -v.x + 0.5]) for v in projected_points]
        rotated_points2 = [Vector([v.z - 0.5, v.y, -v.x + 0.5]) for v in projected_points2]
        back_projected_points = [inv_riemann(rotated_point) for rotated_point in rotated_points]
        back_projected_points2 = [inv_riemann(rotated_point) for rotated_point in rotated_points2]

        transformed_rotated_points = [p * 4 + Vector([0, 0, -4]) for p in rotated_points]
        transformed_rotated_points2 = [p * 4 + Vector([0, 0, -4]) for p in rotated_points2]
        transformed_back_projected_points = [p * 4 + Vector([0, 0, -4]) for p in back_projected_points]
        transformed_back_projected_points2 = [p * 4 + Vector([0, 0, -4]) for p in back_projected_points2]

        end_points = transformed_rotated_points
        end_points2 = transformed_back_projected_points
        end_points3 = transformed_rotated_points2
        end_points4 = transformed_back_projected_points2
        rays = []
        rays2 = []
        for end_point in end_points:
            ray = Cylinder.from_start_to_end(start=Vector(), end=end_point, color='example', thickness=0.1, emission=1)
            ray.grow(begin_time=t0)
            rays.append(ray)
        for end_point in end_points3:
            ray = Cylinder.from_start_to_end(start=Vector(), end=end_point, color='example', thickness=0.1, emission=1)
            ray.grow(begin_time=t0)
            rays2.append(ray)
        t0 += 1

        # create copy of annulus that remains on the sphere
        transformations3 = [
            lambda z: rotate(riemann(16 * z)),
            lambda z: rotate(riemann(8 * z)),
            lambda z: rotate(riemann(4 * z)),
            lambda z: rotate(riemann(2.001 * z)),
        ]

        annulus3 = Annulus2(transformations=transformations3, resolution=10, solid=0.01, offset=-1, scale=[4, 4, 4],
                            location=[0, 0, -4])
        annulus3.appear(begin_time=t0, transition_time=0)
        annulus3.next_shape(begin_time=t0, transition_time=0)

        transformations4 = [
            lambda z: rotate(riemann(1 / 8 * z)),
            lambda z: rotate(riemann(1 / 4 * z)),
            lambda z: rotate(riemann(1 / 2 * z)),
            lambda z: rotate(riemann(0.999 * z)),
        ]
        annulus4 = Annulus2(transformations=transformations4, resolution=10, solid=0.01, offset=-1, scale=[4, 4, 4],
                            location=[0, 0, -4])
        annulus4.appear(begin_time=t0, transition_time=0)
        annulus4.next_shape(begin_time=t0, transition_time=0)

        # back projection
        for end_point, ray in zip(end_points2, rays):
            ray.move_end_point(target_location=end_point, begin_time=t0)
            ray.disappear(begin_time=t0 + 1, transition_time=0.5)
        for end_point, ray in zip(end_points4, rays2):
            ray.move_end_point(target_location=end_point, begin_time=t0)
            ray.disappear(begin_time=t0 + 1, transition_time=0.5)
        annulus2.next_shape(begin_time=t0)
        annulus.next_shape(begin_time=t0)
        t0 += 1.5

        # change camera perspective

        empty = EmptyCube()
        set_camera_view_to(empty)
        ibpy.camera_move(shift=[0, 0, 10], begin_time=t0)
        t0 += 1.5

        # perform remaining rescalings
        for i in range(2):
            annulus.next_shape(begin_time=t0)
            annulus2.next_shape(begin_time=t0)
            annulus3.next_shape(begin_time=t0)
            annulus4.next_shape(begin_time=t0)
            t0 += 1.5

        print("finished: ", t0)

    def algebra(self):
        cues = self.sub_scenes['algebra']
        t0 = 0.5  # cues['start']

        display = Display(scales=[10, 6], flat=True, location=[0, 0, 0], number_of_lines=15)
        t0 = display.appear(begin_time=t0) + 0.5

        title = SimpleTexBObject(r'\text{M\"obius transformations}', color='example', aligned='center')
        t0 = 0.5 + display.write_title(title, begin_time=t0, transition_time=3)

        colors = flatten([['text'] * 3, ['joker'] * 2, ['text'] * 5, ['joker'] * 2])
        trafo = SimpleTexBObject(r"z\mapsto {az+b\over cz+d}", color=colors)
        t0 = 0.5 + display.write_text_in(trafo, line=1, indent=1.5, begin_time=t0)

        colors = flatten([['text'], ['joker'] * 4, ['text']])
        matrix = SimpleTexBObject(r"\left(\begin{array}{c c}a & b \\ c & d\end{array}\right)", color=colors)
        t0 = 0.5 + display.write_text_in(matrix, line=1, indent=5, begin_time=t0)

        colors = flatten([['text'] * 5, ['joker'], ['text']])
        trafo2 = SimpleTexBObject(r"S\colon z\mapsto 2z", color=colors)
        trafo2.align(trafo, char_index=2, other_char_index=0)
        t0 = 0.5 + display.write_text_in(trafo2, line=4, indent=1.5, begin_time=t0)

        colors = flatten([['text'], ['joker'] * 4, ['text']])
        matrix2 = SimpleTexBObject(r"\left(\begin{array}{c c}2 & 0 \\ 0 & 1\end{array}\right)", color=colors)
        t0 = 0.5 + display.write_text_in(matrix2, line=4, indent=5, begin_time=t0)

        colors = flatten([['text'] * 6, ['joker'], ['text'], ['joker'], ['text'] * 3, ['joker'], ['text']])
        trafo3 = SimpleTexBObject(r"S^2\colon z\mapsto 2(2z)=4z", color=colors)
        trafo3.align(trafo, char_index=3, other_char_index=0)
        t0 = 0.5 + display.write_text_in(trafo3, line=7, indent=1.5, begin_time=t0)

        colors = flatten([['text'], ['joker'] * 4, ['text'] * 3, ['joker'] * 4, ['text'] * 3, ['joker'] * 4, ['text']])
        matrix3 = SimpleTexBObject(
            r"\left(\begin{array}{c c}2 & 0 \\ 0 & 1\end{array}\right)\cdot \left(\begin{array}{c c}2 & 0 \\ 0 & 1\end{array}\right)=\left(\begin{array}{c c}4 & 0 \\ 0 & 1\end{array}\right)",
            color=colors)
        t0 = 0.5 + display.write_text_in(matrix3, line=7, indent=5, begin_time=t0)

        colors = flatten([['text'] * 20, ['example']])
        matrix4 = SimpleTexBObject(
            r"\left(\begin{array}{c c}2 & 0 \\ 0 & 1\end{array}\right)\sim\left(\begin{array}{c c}\sqrt{2} & 0 \\ 0 & \tfrac{1}{\sqrt{2}}\end{array}\right)\in SL_2(\mathbb{C})",
            color=colors)
        t0 = 0.5 + display.write_text_in(matrix4, line=10, indent=5, begin_time=t0)

        print("finished: ", t0)

    def summary(self):
        cues = self.sub_scenes['summary']
        t0 = 0  # cues['start']

        # grid = DefaultGrid()
        # grid.show()

        # hdri background
        set_hdri_background('qwantani_puresky_4k', ext='exr', transparent=True)
        set_hdri_strength(1, begin_time=t0, transition_time=0)

        center_ul = Vector([-8, 0, 3])
        center_ur = Vector([8, 0, 3])
        center_dr = Vector([8, 0, -3])
        center_dl = Vector([-8, 0, -3])
        t0 += 0.5

        coords = CoordinateSystem(dim=2, lengths=[6, 6], domains=[[-3, 3], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-3, 3.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=center_ul,
                                  name='ComplexPlaneUL')
        coords.appear(begin_time=t0, transition_time=3)
        zplane = SimpleTexBObject(r"\fbox{\text{$z$--plane}}", color='text', aligned='center',
                                  location=[-5, -0.05, 5.5])
        zplane.write(begin_time=t0 + 2)
        # manually shifting the location of the axis label
        loc = coords.axes[1].axis_label.ref_obj.location
        coords.axes[1].axis_label.ref_obj.location = Vector([loc.x - 0.5, loc.y, loc.z - 0.5])

        transformations = [
            lambda z: riemann(z),
        ]
        annulus = Annulus2(r=[1.25, 2.5], location=center_ul + Vector([0, -0.001, 0]),
                           transformations=[lambda z: z],
                           rotation_euler=[np.pi / 2, 0, 0],
                           resolution=20, smooth=2, roughness=0.25,
                           solid=0.1, bevel=0.1, offset=0, name="AnnulusUL")
        t0 += 3.5

        annulus.appear(begin_time=t0)
        display = Display(scales=[3.5, 2.5], location=0.5 * (center_ul + center_ur), flat=True, number_of_lines=8)
        display.appear(begin_time=t0)
        t0 += 1

        title = BMatrix(entries=np.array([[1, -1], [1, 1]]), pre_word=r"P^{-1}RP\Longleftrightarrow", color='text',
                        aligned="center")
        display.set_title(title, shift=[-2, 0], scale=0.9)

        sphere = Sphere(0.495, location=center_ul + Vector([0, -0.5, 0]), resolution=5, color='fake_glass_text',
                        roughness=0.1, metallic=0.2, transmission=0.9)
        sphere.grow(begin_time=t0)

        t0 += 1.5
        title.write(letter_set={4}, begin_time=t0)
        transformations = [
            lambda z: riemann(z),
            lambda z: rotate(riemann(z))
        ]
        mover = Annulus2(r=[1.25, 2.5], location=center_ul,
                         transformations=transformations,
                         solid=0.05, bevel=0.1, offset=0,
                         resolution=40, smooth=2, roughness=0.25,
                         rotation_euler=[np.pi / 2, 0, 0], name='MoverAnnulus')
        mover.appear(begin_time=t0, transition_time=0)
        t0 += 1
        mover.next_shape(begin_time=t0)
        t0 += 1.5

        origin2 = center_ur - Vector([2, 0, 0])
        coords_w = CoordinateSystem(dim=2, lengths=[6, 6], domains=[[-1, 5], [-3, 3]], radii=[0.03, 0.03],
                                    all_n_tics=[6, 6],
                                    labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                    all_tic_labels=[np.arange(-1, 5.1, 1), np.arange(-3, 3.1, 1)],
                                    colors=['text', 'text'], label_colors=['text', 'text'],
                                    label_digits=[0, 0], label_units=['', 'i'],
                                    axis_label_size='medium',
                                    tic_label_size='small',
                                    location_of_origin=origin2,
                                    name='ComplexPlaneUR')
        coords_w.appear(begin_time=t0, transition_time=3)
        w_plane = SimpleTexBObject(r"\fbox{\text{$w$--plane}}", color='text', aligned='center',
                                   location=[11, -0.05, 5.5])
        w_plane.write(begin_time=t0 + 2)
        # manually shifting the location of the axis label
        loc = coords_w.axes[1].axis_label.ref_obj.location
        coords_w.axes[1].axis_label.ref_obj.location = Vector([loc.x - 0.5, loc.y, loc.z - 1])
        t0 += 3.5

        # move sphere with annulus
        shift = center_ur - center_ul - Vector([2, 0, 0])
        sphere.move(direction=shift, begin_time=t0)
        mover.move(direction=shift, begin_time=t0)
        t0 += 1.5

        # create local annulus for rotation
        transformations3 = [
            lambda z: riemann(z),
            lambda z: rotate(riemann(z)),
            lambda z: inv_riemann(rotate(riemann(z)))
        ]
        annulus2 = Annulus2(r=[1.25, 2.5], resolution=200, location=origin2,
                            transformations=transformations3,
                            rotation_euler=[np.pi / 2, 0, 0], roughness=0.25,
                            solid=0.1, bevel=0.1, offset=0, smooth=2,
                            name="AnnulusUR")
        annulus2.appear(begin_time=t0, transition_time=0)
        annulus2.next_shape(begin_time=t0, transition_time=0)
        mover.disappear(begin_time=t0, transition_time=0)

        # rotate sphere
        title.write(letter_set={3}, begin_time=t0)
        t0 += 1
        sphere.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=t0)
        annulus2.rotate(rotation_euler=[np.pi / 2, 0, np.pi / 2], begin_time=t0, pivot=origin2 - Vector([0, 0.5, 0]))
        annulus2.next_color_map(begin_time=t0)
        t0 += 1.5
        annulus2.next_shape_key(begin_time=t0, transition_time=0)
        annulus2.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=0)

        # project back
        title.write(letter_set={0, 1, 2}, begin_time=t0)
        t0 += 1.5

        annulus2.next_shape(begin_time=t0)
        sphere.disappear(begin_time=t0)
        t0 += 1.5

        # wrap up first transformation
        title.write(letter_set={5, 6}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        title.write(letter_set={7, 13}, begin_time=t0, transition_time=0.3)
        t0 += 0.3
        title.write(letter_set={8, 10, 12}, begin_time=t0, transition_time=0.3)
        t0 += 0.3
        title.write(letter_set={9, 11}, begin_time=t0, transition_time=0.3)
        t0 += 0.8

        trafo = SimpleTexBObject(r"M\colon z\mapsto w={z-1\over z+1}")
        display.add_text_in(trafo, line=4, indent=0.5, scale=1)
        trafo.write(letter_set={0, 1, 2, 3, 4, 5, 6, 7, 9, 12}, begin_time=t0)
        t0 += 1
        trafo.write(letter_set={10}, begin_time=t0, transition_time=0.3)
        t0 += 0.3
        trafo.write(letter_set={8, 11, 13}, begin_time=t0, transition_time=0.3)
        t0 += 0.8

        # repeat everything
        origin3 = Vector([-10, 0, -5])
        coords_z2 = CoordinateSystem(dim=2, lengths=[6, 6], domains=[[-1, 5], [-1, 5]], radii=[0.03, 0.03],
                                     all_n_tics=[6, 6],
                                     labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                     all_tic_labels=[np.arange(-1, 5.1, 1), np.arange(-1, 5.1, 1)],
                                     colors=['text', 'text'], label_colors=['text', 'text'],
                                     label_digits=[0, 0], label_units=['', 'i'],
                                     axis_label_size='medium',
                                     tic_label_size='small',
                                     location_of_origin=origin3,
                                     name='ComplexPlaneDL')
        coords_z2.appear(begin_time=t0, transition_time=3)
        zplane2 = SimpleTexBObject(r"\fbox{\text{$z$--plane}}", color='text', aligned='center',
                                   location=[-5, -0.05, -1])
        zplane2.write(begin_time=t0 + 2)
        # manually shifting the location of the axis label
        loc = coords_z2.axes[1].axis_label.ref_obj.location
        coords_z2.axes[1].axis_label.ref_obj.location = Vector([loc.x - 0.5, loc.y, loc.z - 0.5])

        transformations = [
            lambda z: 2 * z,
        ]

        annulus3 = Annulus2(r=[1.25, 2.5], location=origin3 - Vector([0, -0.001, 0]),
                            transformations=transformations,
                            resolution=40, smooth=2,
                            solid=0.1, bevel=0.1, offset=0, roughness=0.25,
                            rotation_euler=[np.pi / 2, 0, 0], name="AnnulusDL")
        t0 += 3.5

        annulus3.appear(begin_time=t0)

        # scaling
        display2 = Display(scales=[2.5, 0.5], location=0.5 * (center_ul + center_dl) + Vector([2, -0.05, 0]), flat=True,
                           number_of_lines=2, shadow=False)
        title2 = SimpleTexBObject(r"S\colon z\mapsto 2z", aligned='center')
        display2.appear(begin_time=t0, transition_time=0.5)
        t0 += 0.5
        display2.write_title(title2, begin_time=t0)
        t0 += 1

        transformations = [
            lambda z: 2 * z,
            lambda z: riemann(2 * z),
            lambda z: rotate(riemann(2 * z))
        ]
        mover2 = Annulus2(r=[1.25, 2.5], location=origin3, transformations=transformations,
                          rotation_euler=[np.pi / 2, 0, 0], offset=0, bevel=0.1, solid=0.05,
                          name='MoverAnnulus', resolution=40, smooth=2, roughness=0.25, )
        mover2.appear(begin_time=t0, transition_time=0)
        mover2.next_shape(begin_time=t0)
        annulus3.next_shape(begin_time=t0)
        t0 += 1.5

        display3 = Display(scales=[3.5, 2.5], location=0.5 * (center_dl + center_dr), flat=True, number_of_lines=8)
        display3.appear(begin_time=t0)

        title3 = BMatrix(entries=np.array([[1, -1], [1, 1]]), pre_word=r"P^{-1}RP\Longleftrightarrow", color='text',
                         aligned="center")
        display3.set_title(title3, shift=[-2, 0], scale=0.9)

        sphere2 = Sphere(0.495, location=origin3 + Vector([0, -0.5, 0]), resolution=5, color='fake_glass_text',
                         roughness=0.1, metallic=0.2)
        sphere2.grow(begin_time=t0)
        t0 += 1.5

        title3.write(letter_set={4}, begin_time=t0)
        t0 += 1
        mover2.next_shape(begin_time=t0)
        t0 += 1.5

        origin2 = center_dr - Vector([2, 0, 0])
        coords_w2 = CoordinateSystem(dim=2, lengths=[6, 5], domains=[[-1, 5], [-3, 2]], radii=[0.03, 0.03],
                                     all_n_tics=[6, 5],
                                     labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                     all_tic_labels=[np.arange(-1, 5.1, 1), np.arange(-3, 2.1, 1)],
                                     colors=['text', 'text'], label_colors=['text', 'text'],
                                     label_digits=[0, 0], label_units=['', 'i'],
                                     axis_label_size='medium',
                                     tic_label_size='small',
                                     location_of_origin=origin2,
                                     name='ComplexPlaneUL')
        coords_w2.appear(begin_time=t0, transition_time=3)
        w_plane = SimpleTexBObject(r"\fbox{\text{$w$--plane}}", color='text', aligned='center',
                                   location=[11, -0.05, -1])
        w_plane.write(begin_time=t0 + 2)
        # manually shifting the location of the axis label
        loc = coords_w2.axes[1].axis_label.ref_obj.location
        coords_w2.axes[1].axis_label.ref_obj.location = Vector([loc.x - 0.5, loc.y, loc.z - 1])
        t0 += 3.5

        # move sphere with annulus
        shift = center_dr - origin3 - Vector([2, 0, 0])
        sphere2.move(direction=shift, begin_time=t0)
        mover2.move(direction=shift, begin_time=t0)
        t0 += 1.5

        # create local annulus for rotation
        transformations3 = [
            lambda z: riemann(z),
            lambda z: rotate(riemann(z)),
            lambda z: inv_riemann(rotate(riemann(z)))
        ]
        annulus4 = Annulus2(r=[2.5, 5], location=origin2, transformations=transformations3,
                            rotation_euler=[np.pi / 2, 0, 0],
                            solid=0.1, bevel=0.1, offset=0,
                            resolution=40, smooth=2, roughness=0.25,
                            name='AnnulusDR')
        annulus4.appear(begin_time=t0, transition_time=0)
        annulus4.next_shape(begin_time=t0, transition_time=0)
        mover2.disappear(begin_time=t0, transition_time=0)

        # rotate sphere
        title3.write(letter_set={3}, begin_time=t0)
        t0 += 1
        sphere2.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=t0)
        annulus4.rotate(rotation_euler=[np.pi / 2, 0, np.pi / 2], begin_time=t0, pivot=origin2 - Vector([0, 0.5, 0]))
        annulus4.next_color_map(begin_time=t0)
        t0 += 1.5
        annulus4.next_shape_key(begin_time=t0, transition_time=0)
        annulus4.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=0)

        # project back
        title3.write(letter_set={0, 1, 2}, begin_time=t0)
        t0 += 1.5

        annulus4.next_shape(begin_time=t0)
        sphere2.disappear(begin_time=t0)
        t0 += 1.5

        # wrap up second transformation
        title3.write(letter_set={5, 6}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        title3.write(letter_set={7, 13}, begin_time=t0, transition_time=0.3)
        t0 += 0.3
        title3.write(letter_set={8, 10, 12}, begin_time=t0, transition_time=0.3)
        t0 += 0.3
        title3.write(letter_set={9, 11}, begin_time=t0, transition_time=0.3)
        t0 += 0.8

        trafo = SimpleTexBObject(r"M\colon z\mapsto w={z-1\over z+1}")
        display3.add_text_in(trafo, line=4, indent=0.5, scale=1)
        trafo.write(letter_set={0, 1, 2, 3, 4, 5, 6, 7, 9, 12}, begin_time=t0)
        t0 += 1
        trafo.write(letter_set={10}, begin_time=t0, transition_time=0.3)
        t0 += 0.3
        trafo.write(letter_set={8, 11, 13}, begin_time=t0, transition_time=0.3)
        t0 += 0.8

        # final transformation
        display4 = Display(scales=[2.5, 1.5], location=0.5 * (center_ur + center_dr) + Vector([-2, -0.05, 0]),
                           flat=True,
                           number_of_lines=4, shadow=False)
        title4 = SimpleTexBObject(r"a= MSM^{-1} ", aligned='center')
        display4.appear(begin_time=t0, transition_time=0.5)
        display4.set_title(title4)
        t0 += 1

        title4.write(letter_set={0, 1}, begin_time=t0 + 0.75, transition_time=0.25)
        arrow1 = PArrow(start=Vector([11, -.25, 3]), end=Vector([11, -.25, -3]), color='joker', thickness=3,
                        name='ArrowUDR')
        arrow1.grow(begin_time=t0)
        t0 += 1.5
        title4.write(letter_set={4, 5, 6}, begin_time=t0 + 0.75, transition_time=0.25)
        arrow2 = PArrow(start=Vector([4, -.25, 3]), end=Vector([-4, -.25, 3]), color='joker', thickness=3,
                        name='ArrowRLU')
        arrow2.grow(begin_time=t0)
        t0 += 1.5

        title4.write(letter_set={3}, begin_time=t0 + 0.9, transition_time=0.1)
        arrow3 = PArrow(start=Vector([-11, -.25, 3]), end=Vector([-11, -.25, -3]), color='joker', thickness=3,
                        name='ArrowUDL')
        arrow3.grow(begin_time=t0)
        t0 += 1.5

        title4.write(letter_set={2}, begin_time=t0 + 0.9, transition_time=0.1)
        arrow4 = PArrow(start=Vector([-4, -.25, -3]), end=Vector([4, -.25, -3]), color='joker', thickness=3,
                        name='ArrowLRD')
        arrow4.grow(begin_time=t0)
        t0 += 1.5

        # final remarks

        result = SimpleTexBObject(r"a\colon w\mapsto {3z+1\over z+3}")
        display4.add_text_in(result, line=1.5, indent=0.25)
        result.write(letter_set={0, 1, 2, 3, 4, 5, 7, 10, 12}, begin_time=t0)
        t0 += 1
        result.write(letter_set={8}, begin_time=t0, transition_time=0.3)
        t0 += 0.3
        result.write(letter_set={6, 9, 11}, begin_time=t0, transition_time=0.1)
        t0 += 0.6

        print('finish: ', t0)

    def fixed_points(self):
        cues = self.sub_scenes['fixed_points']
        t0 = 0.1  # cues['start']

        # hdri background
        # set_hdri_background('sunset_in_the_chalk_quarry_4k', ext='exr', transparent=True)
        # set_hdri_strength(1, begin_time=t0, transitions_time=0.1)

        # camera settings
        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # display
        display = Display(number_of_lines=15, flat=True, location=[12, -1, 0], shadow=False)
        display.appear(begin_time=t0)
        t0 += 0.5

        origin = Vector([-5, 0, -1])
        coords = CoordinateSystem(dim=2, lengths=[20, 12], domains=[[-5, 5], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=3)
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        title = SimpleTexBObject(r"\text{Fixed points}", aligned='center', color='example')
        display.write_title(title, begin_time=t0)
        t0 += 1.5

        trafo = BMatrix(entries=np.array([[3, 1], [1, 3]]),
                        pre_word=r"a\colon w\mapsto {3w+1\over w+3}\Longleftrightarrow")
        display.write_text_in(trafo, line=1, begin_time=t0, scale=0.5, indent=0.5)
        t0 += 1.5

        a = np.array([[3.001, 1], [1, 2.999]])
        transformations = [
            lambda w: w,
        ]

        Y = Vector([0, 1, 0])
        # the disc is readily obtained from transforming a disc with radius 3/2 centered around the origin
        # by M 3/2->1/5 and -3/2 -> 5, which gives the radius =(5-1/5)/2 and the x coordinate of the center (5+1/5)/2
        disc = Disc2(r=12 / 5, center=[13 / 5, 0, 0],
                     resolution=[100, 100], rotation_euler=[np.pi / 2, 0, 0],
                     location=0 * Y, scale=[2, 2, 2],
                     solid=0.1, bevel=0.1, smooth=2,
                     transformations=transformations, name='Disc_a')
        disc.appear(begin_time=t0)
        coords.add_object(disc)
        t0 += 1.5

        # ibpy.camera_move(shift=[0,0,-34],begin_time=t0)
        # ibpy.camera_zoom(lens=60,begin_time=t0)
        # display.rotate(rotation_euler=[np.pi/2+np.pi/3,0,0],begin_time=t0)

        coords.rotate(rotation_euler=[-np.pi / 3, 0, 0], begin_time=t0, compensate=True)
        t0 += 1.5

        r1 = 1 / 5
        r2 = 5
        for i in range(1, 6):
            transformations = [
                lambda w: moebius_vec(a, w),
            ]
            disc2 = Disc2(r=(r2 - r1) / 2, center=[(r2 + r1) / 2, 0, 0],
                          resolution=[int(100 / i), 100],
                          rotation_euler=[np.pi / 2, 0, 0],
                          location=.02 * Y, scale=[2, 2, 2],
                          transformations=transformations,
                          solid=0.1, bevel=0.1, smooth=2, name='disc_a_' + str(i))
            disc2.appear(begin_time=t0, transition_time=0)
            coords.add_object(disc2)
            disc2.next_shape(begin_time=t0)
            disc2.move(direction=-0.2 * i * Y, begin_time=t0, transition_time=0)

            r1 = moebius_vec(a, Vector([r1, 0, 0])).x
            r2 = moebius_vec(a, Vector([r2, 0, 0])).x
            t0 += 1.5

        lines = [
            SimpleTexBObject(r"\text{Fixed point equation:} "),
            SimpleTexBObject(r"w=a(w)"),
            SimpleTexBObject(r"w={3w+1\over w+3}"),
            SimpleTexBObject(r"\text{two solutions:}"),
            SimpleTexBObject(r"w_+=1"),
            SimpleTexBObject(r"w_-=-1"),
        ]

        indents = [0.5, 1, 1, 2.5, 3, 3]
        rows = [3, 4, 6, 8, 9, 10]
        for l, line in enumerate(lines):
            display.write_text_in(line, line=rows[l], begin_time=t0, indent=indents[l])
            t0 += 1.5

        coords.move(direction=[5, 0, 0], begin_time=t0)

        r1 = 1 / 5
        r2 = 5
        A = np.array([[3, -1], [-1, 3]])

        r1 = moebius_vec(A, Vector([r1, 0, 0])).x
        r2 = moebius_vec(A, Vector([r2, 0, 0])).x
        t0 += 1.5

        disc = Disc2(r=np.abs(r1 - r2) / 2, center=[(r2 + r1) / 2, 0, 0],
                     resolution=[100, 100],
                     rotation_euler=[np.pi / 2, 0, 0],
                     location=0 * Y,
                     scale=[2, 2, 2], transformations=transformations,
                     solid=0.1, bevel=0.1, smooth=2, name='DiscA')

        disc.appear(begin_time=t0)
        coords.add_object(disc)
        t0 += 1.5

        trafo2 = BMatrix(entries=np.array([[3, -1], [-1, 3]]),
                         pre_word=r"A\colon w\mapsto {3w-1\over -w+3}\Longleftrightarrow")
        display.write_text_in(trafo2, line=12, begin_time=t0, scale=0.5, indent=0.5)

        for i in range(1, 6):
            transformations = [
                lambda w: moebius_vec(A, w),
            ]
            disc2 = Disc2(r=np.abs(r1 - r2) / 2, center=[(r2 + r1) / 2, 0, 0],
                          resolution=[int(100 / i), 100],
                          rotation_euler=[np.pi / 2, 0, 0],
                          location=0.02 * Y, scale=[2, 2, 2],
                          transformations=transformations,
                          solid=0.1, bevel=0.1, smooth=2, name='DiscA_' + str(i))
            disc2.appear(begin_time=t0, transition_time=0)
            coords.add_object(disc2)
            disc2.next_shape(begin_time=t0)
            disc2.move(direction=-0.2 * i * Y, begin_time=t0, transition_time=0)

            r1 = moebius_vec(A, Vector([r1, 0, 0])).x
            r2 = moebius_vec(A, Vector([r2, 0, 0])).x
            t0 += 1.5

        print("finished at time: ", t0)

    def fixed_points2(self):
        cues = self.sub_scenes['fixed_points2']
        t0 = 0.1  # cues['start']
        display = Display(flat=True, location=[0, 0, 0], scales=[6, 2])

        text = SimpleTexBObject(r"\forall z\in \mathbb{C} \setminus \{-1\}: \,\,\, aaa\dots aaa(z)=\bar{a}(z)=1",
                                color='important')
        text2 = SimpleTexBObject(r"\forall z\in \mathbb{C} \setminus \{1\}: \,\,\, AAA\dots AAA(z)=\bar{A}(z)=-1 ",
                                 color='drawing')

        t0 = 0.5 + display.write_text_in(text, line=0, indent=1, begin_time=t0)
        t0 = 0.5 + display.write_text_in(text2, line=2, indent=1, begin_time=t0)

        print("finished: ", t0)

    def fixed_points3(self):
        cues = self.sub_scenes['fixed_points3']
        t0 = 0.5

        factor = 10  # set to 10 for final render
        n_discs = 10  # 10  for final render

        am = ApollonianModel()
        colors = am.colors

        matrix_a = BMatrix(np.array([['1', '0'], ['-2i', '1']]), pre_word='a=', color=colors[0], outlined='text')
        matrix_b = BMatrix(np.array([['1-i', '1'], ['1', '1+i']]), pre_word='b=', color=colors[1])
        display = Display(flat=True, number_of_lines=10, location=[-8.5, 0, 0], scales=[3, 5], shadow=False)
        display2 = Display(flat=True, number_of_lines=10, location=[8.5, 0, 0], scales=[3, 5], shadow=False)

        t0 = 0.5 + display.appear(begin_time=t0)
        t0 = 0.5 + display.write_text_in(matrix_a, line=0, indent=1, begin_time=t0)

        lines = [
            SimpleTexBObject(r"z=a(z)", color=colors[0], outlined='text'),
            SimpleTexBObject(r"z={z\over 2i z+1}", color=colors[0], outlined='text'),
            SimpleTexBObject(r"0=2iz^2", color=colors[0], outlined='text'),
            SimpleTexBObject(r"\text{one solution:}", color=colors[0], outlined='text'),
            SimpleTexBObject(r"z=0", color=colors[0], outlined='text'),
        ]

        rows = [2, 4, 6, 7, 8]
        idx = [1, 1, 1, 2, 1]
        for l, line in enumerate(lines):
            if l > 0:
                lines[l].align(lines[0], char_index=idx[l], other_char_index=1)
            t0 = 0.5 + display.write_text_in(line, line=rows[l], begin_time=t0, indent=1)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 2], [-2, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[8, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 0.5), np.arange(-2, 2.1, .5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane2')
        coords.appear(begin_time=t0, transition_time=2)
        t0 += 2.5

        a_trafo = VectorTransformation()
        # look at the mathematica worksheet "VectorTransformations" to see how the expressions are generated
        a_trafo.set_transformation_function(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) * Vector(
                [v.x, 0, 2 * v.x ** 2 + v.z + 2 * v.z ** 2]))
        a_trafo.set_first_derivative_functions(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) ** 2 * np.array(
                [[-4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0, 4 * v.x * (1 + 2 * v.z)], [0, 0, 0],
                 [-4 * v.x * (1 + 2 * v.z), 0, -4 * v.x ** 2 + (1 + 2 * v.z) ** 2]]
            ))
        a_trafo.set_second_derivative_functions(
            lambda v: 1 / regularized(4 * v.x ** 2 + (1 + 2 * v.z) ** 2, 0.001) ** 3 * np.array(
                [[[8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2), 0,
                   4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2)], [0, 0, 0],
                  [-4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2), 0,
                   8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2)]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[-4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2), 0,
                   8 * v.x * (4 * v.x ** 2 - 3 * (1 + 2 * v.z) ** 2)], [0, 0, 0],
                  [8 * v.x * (-4 * v.x ** 2 + 3 * (1 + 2 * v.z) ** 2), 0,
                   -4 * (1 + 2 * v.z) * (-12 * v.x ** 2 + (1 + 2 * v.z) ** 2)]]
                 ]))

        coords.draw_transformable_polar_grid(transformations=[a_trafo], twists=[0, 0],
                                             begin_time=t0, transition_time=3, sub_grid=5,
                                             thickness=0.1, num_points=factor * 250)
        t0 += 3.5
        coords.grid_next_transform(begin_time=t0, transition_time=3)
        t0 += 3.5

        t0 = 0.5 + display2.appear(begin_time=t0)
        t0 = 0.5 + display2.write_text_in(matrix_b, line=0, indent=0.5, begin_time=t0, scale=0.6)

        lines = [
            SimpleTexBObject(r"z=b(z)", color=colors[1]),
            SimpleTexBObject(r"z={(1-i)z+1\over z+1+i}", color=colors[1]),
            SimpleTexBObject(r"0=z^2+2iz-1", color=colors[1]),
            SimpleTexBObject(r"\text{one solution:}", color=colors[1]),
            SimpleTexBObject(r"z=-i", color=colors[1]),
        ]

        for l, line in enumerate(lines):
            if l > 0:
                lines[l].align(lines[0], char_index=idx[l], other_char_index=1)
            t0 = 0.5 + display2.write_text_in(line, line=rows[l], begin_time=t0, indent=0.5)

        coords.disappear(begin_time=t0 - 1.5)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 2], [-3, 1]], radii=[0.03, 0.03],
                                  all_n_tics=[8, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 0.5), np.arange(-3, 1.1, .5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 2],
                                  name='ComplexPlane2')
        coords.appear(begin_time=t0, transition_time=2)
        t0 += 2.5

        b_trafo = VectorTransformation()
        # look at the mathematica worksheet "VectorTransformations" to see how the expressions are generated
        b_trafo.set_transformation_function(
            lambda v: 1 / ((v.x + 1) ** 2 + (v.z + 1) ** 2) * Vector(
                [(v.x + 1) * (v.x + v.z + 1) + (v.z - v.x) * (v.z + 1), 0,
                 (v.z - v.x) * (v.x + 1) - (v.x + v.z + 1) * (v.z + 1)])
        )
        b_trafo.set_first_derivative_functions(lambda v: 1 / (2 * v.x * (v.x + 2) + v.z * (v.z + 2)) ** 2 * np.array(
            [[(v.x - v.z) * (2 + v.x + v.z), 0, -2 * (1 + v.x) * (1 + v.z)], [0, 0, 0],
             [2 * (1 + v.x) * (1 + v.z), 0, (v.x - v.z) * (2 + v.x + v.z)]]
        ))
        b_trafo.set_second_derivative_functions(lambda v: 1 / (2 * v.x * (v.x + 2) + v.z * (v.z + 2)) ** 3 * np.array(
            [[[-2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z)), 0,
               -2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z))], [0, 0, 0],
              [2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z)), 0,
               -2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z))]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z)), 0,
               -2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z))], [0, 0, 0],
              [2 * (1 + v.x) * (-2 + v.x * (2 + v.x) - 3 * v.z * (2 + v.z)), 0,
               2 * (1 + v.z) * (-2 - 3 * v.x * (2 + v.x) + v.z * (2 + v.z))]]
             ]))

        coords.draw_transformable_polar_grid(transformations=[b_trafo], twists=[0, 0],
                                             begin_time=t0, transition_time=3, sub_grid=5,
                                             thickness=0.1, num_points=factor * 100, center=Vector([0, 0, -1]))
        t0 += 3.5
        coords.grid_next_transform(begin_time=t0, transition_time=3)
        t0 += 3.5
        t0 = 0.5 + coords.disappear(begin_time=t0)

        coords = CoordinateSystem(dim=2, lengths=[8, 8], domains=[[-2, 2], [-2, 2]], radii=[0.015, 0.015],
                                  all_n_tics=[4, 4],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 1), np.arange(-2, 2.1, 1)],
                                  colors=['drawing', 'drawing'], label_colors=['drawing', 'drawing'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane')
        coords.rotate(rotation_euler=[-np.pi / 4, 0, 0], begin_time=t0, transition_time=0)
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=2)
        coords.axes[0].labels[2].ref_obj.location = [0.4, 0, -0.20651]

        am = ApollonianModel()
        a = am.get_generators()[0]
        b = am.get_generators()[1]
        A = am.get_generators()[2]
        B = am.get_generators()[3]

        circle = am.get_circles()[0]
        discs = []
        for i in range(n_discs):
            disc = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[0], solid=0.04,
                         resolution=np.maximum(500, np.minimum(10, 1000 * circle.r)),
                         name='Disc_' + str(i), metallic=0.25, roughness=0.5)
            discs.append(disc)
            disc.grow(begin_time=t0, pivot=z2vec(circle.c))
            disc.move(direction=[0, 0, 0.04 * i], begin_time=t0, transition_time=0.1)
            t0 += 1
            circle = moebius_on_circle(a, circle)

        container = BObject(children=discs, rotation_euler=[np.pi / 4, 0, 0], scale=2)
        container.appear()

        t0 += 0.5  # pause

        circle = am.get_circles()[2]
        discs = []
        for i in range(n_discs):
            disc = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[2], solid=0.04,
                         resolution=np.maximum(500, np.minimum(10, 1000 * circle.r)),
                         name='Disc_' + str(i), metallic=0.25, roughness=0.5)
            discs.append(disc)
            disc.grow(begin_time=t0, pivot=z2vec(circle.c))
            disc.move(direction=[0, 0, 0.04 * i], begin_time=t0, transition_time=0.1)
            t0 += 1
            circle = moebius_on_circle(A, circle)

        container = BObject(children=discs, rotation_euler=[np.pi / 4, 0, 0], scale=2)
        container.appear()

        t0 += 0.5  # pause

        circle = am.get_circles()[1]
        discs = []
        for i in range(n_discs):
            disc = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[1], solid=0.04,
                         resolution=np.maximum(500, np.minimum(10, 1000 * circle.r)),
                         name='Disc_' + str(i), metallic=0.25, roughness=0.5)
            disc.grow(begin_time=t0, pivot=z2vec(circle.c))
            disc.move(direction=[0, 0, 0.04 * i], begin_time=t0, transition_time=0.1)
            discs.append(disc)
            t0 += 1
            circle = moebius_on_circle(b, circle)

        container = BObject(children=discs, rotation_euler=[np.pi / 4, 0, 0], scale=2)
        container.appear()

        t0 += 0.5  # pause

        circle = am.get_circles()[3]
        discs = []
        for i in range(n_discs):
            disc = Disc2(r=circle.r, center=z2vec(circle.c), color=am.colors[3], solid=0.04,
                         resolution=np.maximum(500, np.minimum(10, 1000 * circle.r)),
                         name='Disc_' + str(i), metallic=0.25, roughness=0.5)
            discs.append(disc)
            disc.grow(begin_time=t0, pivot=z2vec(circle.c))
            disc.move(direction=[0, 0, 0.04 * i], begin_time=t0, transition_time=0.1)
            t0 += 1
            circle = moebius_on_circle(B, circle)

        container = BObject(children=discs, rotation_euler=[np.pi / 4, 0, 0], scale=2)
        container.appear()

        t0 += 0.5  # pause

        print("finished: ", t0)

    def variant(self):
        cues = self.sub_scenes['variant']
        t0 = 0  # cues['start']

        set_hdri_background('qwantani_puresky_4k', ext='exr', transparent=True)
        set_hdri_strength(1, begin_time=t0, transition_time=0.1)
        # perform projection on a sphere with larger radius

        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-5, 5], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 10],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-5, 5.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=3)
        # adjust position of the zero label manually
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        t0 = 3.5

        r = 2
        s = (1 + np.sqrt(2)) ** 2
        k = (1 + s - 2 * np.sqrt(s)) / (s - 1)
        rd = 2 * r * (1 - k) / (1 + k)

        transformations = [
            lambda v: riemann_general(v, r),
            lambda v: rotate2(riemann_general(v, r), r)
        ]

        Y = Vector([0, 1, 0])
        disc = Disc2(r=rd, center=[0, 0, 0], resolution=[50, 100],
                     rotation_euler=[np.pi / 2, 0, 0], offset=0,
                     location=-0.05 * Y, transformations=transformations,
                     solid=0.1, bevel=0.1, smooth=2, name='Disc_b')
        disc.appear(begin_time=t0)
        coords.rotate(rotation_euler=[-np.pi / 4, 0, 0], begin_time=t0, compensate=True)
        t0 += 1.5

        sphere = Sphere(r, location=Vector([0, -r, 0]), resolution=5,
                        color='fake_glass_text', transmission=1, ior=1,
                        roughness=0.01, metallic=0.2)
        sphere.grow(begin_time=t0)
        t0 += 1.5

        disc.next_shape(begin_time=t0)
        t0 += 1.5

        disc.rotate(rotation_euler=[0, 0, 0], begin_time=t0, pivot=[0, -r, 0])
        sphere.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0)
        disc.next_color_map(begin_time=t0)
        t0 += 1.

        transformations2 = [
            lambda v: rotate2(riemann_general(v, r), r),
            lambda v: riemann_general_inv(rotate2(riemann_general(v, r), r), r)
        ]

        disc2 = Disc2(r=rd, center=[0, 0, 0], resolution=[20, 100],
                      rotation_euler=[np.pi / 2, 0, 0], offset=0,
                      location=-0.05 * Y, transformations=transformations2,
                      solid=0.1, bevel=0.1, smooth=2, name='Disc_b')
        disc2.next_shape(begin_time=t0, transition_time=0)
        t0 += 0.5
        disc2.appear(begin_time=t0, transition_time=0)
        disc.disappear(begin_time=t0, transition_time=0)
        t0 += 1.5
        disc2.next_shape(begin_time=t0)
        sphere.disappear(begin_time=t0)
        t0 += 1.5

        r1 = -2 * r * k
        r2 = -2 * r / k
        print(r1, r2)

        b = np.array([[s + 1, 2 * r * (1 - s) * 1j], [1j / 2 / r * (s - 1), s + 1]])

        for i in range(1, 4):
            transformations = [
                lambda w: moebius_vec(b, w),
            ]
            disc_next = Disc2(r=np.abs(r1 - r2) / 2, center=[0, (r2 + r1) / 2, 0], resolution=[int(100 / i), 100],
                              rotation_euler=[np.pi / 2, 0, 0], offset=0,
                              location=-0.05 * Y, scale=[1, 1, 1],
                              transformations=transformations,
                              solid=0.1, bevel=0.1, smooth=2, name='Disc_b_' + str(i))
            disc_next.appear(begin_time=t0, transition_time=0)
            coords.add_object(disc_next)
            disc_next.next_shape(begin_time=t0)
            disc_next.move(direction=-0.1 * i * Y, begin_time=t0, transition_time=0)
            if i == 3:
                coords.axes[1].tics[1].disappear(begin_time=t0)
            r1 = moebius_vec(b, Vector([0, r1, 0])).y
            r2 = moebius_vec(b, Vector([0, r2, 0])).y
            print(r1, r2)
            t0 += 1.5

        B = np.array([[s + 1, 2 * r * (s - 1) * 1j], [1j / 2 / r * (1 - s), s + 1]])
        # r1 = moebius(B,moebius(B,Vector([0,-2*r*k,0]))).y
        # r2 = moebius(B,moebius(B,Vector([0,-2*r/k,0]))).y
        r1 = moebius_vec(B, Vector([0, -2 * r * k, 0])).y
        r2 = moebius_vec(B, Vector([0, -2 * r / k, 0])).y
        print(r1, r2)
        disc3 = Disc2(r=np.abs(r1 - r2) / 2, center=[0, (r1 + r2) / 2, 0], resolution=[100, 100],
                      rotation_euler=[np.pi / 2, 0, 0], offset=0,
                      location=-0.07 * Y, transformations=transformations2,
                      solid=0.1, bevel=0.1, smmoth=2, name='Disc_B')

        disc3.appear(begin_time=t0)

        t0 += 1.5

        for i in range(1, 4):
            transformations = [
                lambda w: moebius_vec(B, w),
            ]
            disc_next2 = Disc2(r=np.abs(r1 - r2) / 2, center=[0, (r2 + r1) / 2, 0], resolution=[int(100 / i), 100],
                               rotation_euler=[np.pi / 2, 0, 0], offset=0,
                               location=-0.05 * Y, scale=[1, 1, 1], transformations=transformations,
                               solid=0.1, bevel=0.1, smooth=2, name='Disc_B_' + str(i))
            disc_next2.appear(begin_time=t0, transition_time=0)
            coords.add_object(disc_next2)
            disc_next2.next_shape(begin_time=t0)
            disc_next2.move(direction=-0.1 * i * Y, begin_time=t0, transition_time=0)
            if i == 2:
                coords.axes[1].tics[8].disappear(begin_time=t0)
            r1 = moebius_vec(B, Vector([0, r1, 0])).y
            r2 = moebius_vec(B, Vector([0, r2, 0])).y
            print(r1, r2)
            t0 += 1.5

        coords.add_objects([sphere, disc, disc2, disc3])
        print("finished at time: ", t0)

    def doubling(self):
        cues = self.sub_scenes['doubling']
        t0 = 0.5  # cues['start']

        # setup initial conditions
        R1 = 1.5
        R2 = R1 * 3.247639506  # touching condition calculated in asymmetric_touching_discs.mw
        s = 4  # scaling
        k = k_of_s(s)

        a = np.array([[s + 1, 2 * R2 * (1 - s)], [(1 - s) / 2 / R2, s + 1]])
        A = np.array([[s + 1, 2 * R2 * (s - 1)], [(s - 1) / 2 / R2, s + 1]])
        b = np.array([[s + 1, 2 * R1 * 1j * (1 - s)], [(1 - s) / 2 / R1 / 1j, s + 1]])
        B = np.array([[s + 1, 2 * R1 * 1j * (s - 1)], [(s - 1) / 2 / R1 / 1j, s + 1]])

        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=2, lengths=[20, 10], domains=[[-10, 10], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 10],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-10, 10.1, 2), np.arange(-5, 5.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=3)
        # adjust position of the zero label manually
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        t0 = 3.5
        # vertically the smaller disc
        rs = R1 * (1 / k - k)
        cs = R1 * 1j * (1 / k + k)
        # horizontally the bigger disc
        rb = R2 * (1 / k - k)
        cb = R2 * (1 / k + k)

        remove_discs = []

        # draw initial discs
        trafos = [lambda z: z]
        disc1 = Disc2(r=rs, center=Vector([0, -np.abs(cs), 0]),
                      rotation_euler=[np.pi / 2, 0, 0],
                      resolution=[100, 100],
                      offset=0, bevel=0.1, solid=0.1, smooth=2,
                      transformations=trafos)
        disc1.appear(begin_time=t0)
        coords.rotate(rotation_euler=[-np.pi / 4, 0, 0], begin_time=t0, compensate=True)

        t0 += 1.5
        t0 = fill_transforms(R1, k, b, -1j, coords, t0, label='disc_b', remove_tic=[1, 2], disc_collector=remove_discs)

        disc4 = Disc2(r=rs, center=Vector([0, np.abs(cs), 0]), resolution=[50, 50], rotation_euler=[np.pi / 2, 0, 0],
                      offset=0, bevel=0.1, solid=0.1, smooth=2,
                      transformations=trafos)
        disc4.appear(begin_time=t0)
        t0 = fill_transforms(R1, k, B, 1j, coords, t0, label='disc_B', remove_tic=[1, 7], disc_collector=remove_discs)

        disc2 = Disc2(r=rb, center=Vector([-cb, 0, 0]), resolution=[100, 200], rotation_euler=[np.pi / 2, 0, 0],
                      offset=0, bevel=0.1, solid=0.1, smooth=2,
                      transformations=trafos)
        disc2.appear(begin_time=t0)
        t0 += 1.5
        t0 = fill_transforms(R2, k, a, -1, coords, t0, label='disc_a', disc_collector=remove_discs)

        disc3 = Disc2(r=rb, center=Vector([cb, 0, 0]), resolution=[50, 100], rotation_euler=[np.pi / 2, 0, 0],
                      offset=0, bevel=0.1, solid=0.1, smooth=2,
                      transformations=trafos)
        disc3.appear(begin_time=t0)
        t0 = fill_transforms(R2, k, A, 1, coords, t0, label='disc_A', disc_collector=remove_discs)

        coords.add_objects([disc1, disc2, disc3, disc4])

        t0 += 1
        duration = 5
        dt = duration / 2 / len(remove_discs)
        for i, disc in enumerate(remove_discs):
            disc.disappear(begin_time=t0 + i * dt, transition_time=duration / 2)

        t0 += duration
        print("finished at time: ", t0)

    def pre_full_picture(self):
        cues = self.sub_scenes['pre_full_picture']
        t0 = 0  # cues['start']

        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=2, lengths=[20, 10], domains=[[-10, 10], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 10],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-10, 10.1, 2), np.arange(-5, 5.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        t0 = coords.appear(begin_time=t0, transition_time=1) + 0.5
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        # setup initial conditions
        R1 = 1.5
        R2 = R1 * 3.247639506  # touching condition calculated in asymmetric_touching_discs.mw
        s = 4  # scaling
        k = k_of_s(s)

        A = np.array([[s + 1, 2 * R2 * (1 - s)], [(1 - s) / 2 / R2, s + 1]])
        a = np.array([[s + 1, 2 * R2 * (s - 1)], [(s - 1) / 2 / R2, s + 1]])
        b = np.array([[s + 1, 2 * R1 * 1j * (1 - s)], [(1 - s) / 2 / R1 / 1j, s + 1]])
        B = np.array([[s + 1, 2 * R1 * 1j * (s - 1)], [(s - 1) / 2 / R1 / 1j, s + 1]])

        # vertically the smaller disc
        rs = R1 * (1 / k - k)
        cs = -R1 * 1j * (1 / k + k)
        # horizontally the bigger disc
        rb = R2 * (1 / k - k)
        cb = -R2 * (1 / k + k)

        circles = [
            IndraCircle(-cb, rb),  # a
            IndraCircle(cb, rb),  # A
            IndraCircle(-cs, rs),  # b
            IndraCircle(cs, rs),  # B
        ]

        bfs = BreadFirstSearch(a, b, A, B, 6)
        colors = BlenderModel().colors

        colors = [colors[2], colors[0], colors[3], colors[1]]
        duration = 3
        thickness = 0.1
        coords.rotate(rotation_euler=[-np.pi / 4, 0, 0], begin_time=t0, transition_time=duration / 4, compensate=True)
        removes = []
        dt = duration / len(circles)

        for i, circle in enumerate(circles):
            disc_next = Disc2(circle.r, center=Vector([np.real(circle.c), np.imag(circle.c), 0]), color=colors[i],
                              resolution=[100, 100], rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            t0 = disc_next.appear(begin_time=t0, transition_time=dt)
            removes.append(disc_next)
            coords.add_object(disc_next)

        current = a
        label = 'a'
        Y = Vector([0, -thickness, 0])
        for i in range(4):
            img = moebius_on_circle(current, circles[0])
            disc_next = Disc2(img.r, name='Disc_' + label,
                              center=Vector([np.real(img.c), np.imag(img.c), 0]),
                              color=colors[0],
                              resolution=[np.maximum(2, int(20 * img.r)), np.maximum(10, int(100 * img.r))],
                              rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            disc_next.appear(begin_time=t0, transition_time=1)
            t0 = disc_next.move(direction=(i + 1) * Y, begin_time=t0, transition_time=dt) + 0.5
            coords.add_object(disc_next)
            removes.append(disc_next)
            current = current @ a
            label += 'a'

        t0 += 0.5
        current = A
        label = 'A'
        Y = Vector([0, -thickness, 0])
        for i in range(4):
            img = moebius_on_circle(current, circles[1])
            disc_next = Disc2(img.r, name='Disc_' + label,
                              center=Vector([np.real(img.c), np.imag(img.c), 0]),
                              color=colors[1],
                              resolution=[np.maximum(2, int(20 * img.r)), np.maximum(10, int(100 * img.r))],
                              rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            disc_next.appear(begin_time=t0, transition_time=1)
            t0 = disc_next.move(direction=(i + 1) * Y, begin_time=t0, transition_time=dt) + 0.5
            coords.add_object(disc_next)
            removes.append(disc_next)
            current = current @ A
            label += 'A'

        t0 += 0.5
        current = B
        label = 'b'
        Y = Vector([0, -thickness, 0])
        for i in range(4):
            img = moebius_on_circle(current, circles[2])
            disc_next = Disc2(img.r, name='Disc_' + label,
                              center=Vector([np.real(img.c), np.imag(img.c), 0]),
                              color=colors[2],
                              resolution=[np.maximum(2, int(20 * img.r)), np.maximum(10, int(100 * img.r))],
                              rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            disc_next.appear(begin_time=t0, transition_time=1)
            t0 = disc_next.move(direction=(i + 1) * Y, begin_time=t0, transition_time=dt) + 0.5
            coords.add_object(disc_next)
            removes.append(disc_next)
            current = current @ B
            label += 'B'

        t0 += 0.5
        current = b
        label = 'B'
        Y = Vector([0, -thickness, 0])
        for i in range(4):
            img = moebius_on_circle(current, circles[3])
            disc_next = Disc2(img.r, name='Disc_' + label,
                              center=Vector([np.real(img.c), np.imag(img.c), 0]),
                              color=colors[3],
                              resolution=[np.maximum(2, int(20 * img.r)), np.maximum(10, int(100 * img.r))],
                              rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            disc_next.appear(begin_time=t0, transition_time=1)
            t0 = disc_next.move(direction=(i + 1) * Y, begin_time=t0, transition_time=dt) + 0.5
            coords.add_object(disc_next)
            removes.append(disc_next)
            current = current @ b
            label += 'B'

        t0 += 0.5

        t0 = coords.rotate(rotation_euler=[0, 0, 0], compensate=True, begin_time=t0) + 0.5
        for remove in removes:
            remove.disappear(begin_time=t0)

        t0 += 1.5

        print("finished ", t0)

    def full_picture_start(self):
        cues = self.sub_scenes['full_picture_start']
        t0 = 0  # cues['start']

        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=2, lengths=[20, 10], domains=[[-10, 10], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 10],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-10, 10.1, 2), np.arange(-5, 5.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        # setup initial conditions
        R1 = 1.5
        R2 = R1 * 3.247639506  # touching condition calculated in asymmetric_touching_discs.mw
        s = 4  # scaling
        k = k_of_s(s)

        a = np.array([[s + 1, 2 * R2 * (1 - s)], [(1 - s) / 2 / R2, s + 1]])
        A = np.array([[s + 1, 2 * R2 * (s - 1)], [(s - 1) / 2 / R2, s + 1]])
        b = np.array([[s + 1, 2 * R1 * 1j * (1 - s)], [(1 - s) / 2 / R1 / 1j, s + 1]])
        B = np.array([[s + 1, 2 * R1 * 1j * (s - 1)], [(s - 1) / 2 / R1 / 1j, s + 1]])

        # vertically the smaller disc
        rs = R1 * (1 / k - k)
        cs = R1 * 1j * (1 / k + k)
        # horizontally the bigger disc
        rb = R2 * (1 / k - k)
        cb = R2 * (1 / k + k)

        circles = [
            IndraCircle(-cb, rb),  # a
            IndraCircle(-cs, rs),  # b
            IndraCircle(cb, rb),  # A
            IndraCircle(cs, rs),  # B
        ]

        bfs = BreadFirstSearch(a, b, A, B, 6)
        colors = BlenderModel().colors
        colors = [colors[2], colors[0], colors[3], colors[1]]

        duration = 40
        thickness = 0.1

        indices = [2, 0, 3, 1]
        for i in range(4):
            circle = circles[indices[i]]
            disc_next = Disc2(circle.r, center=Vector([np.real(circle.c), np.imag(circle.c), 0]), color=colors[i],
                              resolution=[20, 100], rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            t0 = disc_next.appear(begin_time=t0, transition_time=1)
            coords.add_object(disc_next)

        t0 += 0.5

        Y = Vector([0, -thickness, 0])
        labels = ['a', 'b', 'A', 'B']
        indices = [8, 11, 6, 12, 3, 9, 1, 4, 10, 5, 7, 2]
        c_index = [0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3]
        a = bfs.gens[2]
        A = bfs.gens[0]
        b = bfs.gens[3]
        B = bfs.gens[1]
        gens = [a, b, B, b, A, a, A, B, b, B, A, a]
        discs = [2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1]
        for i in range(12):
            img = moebius_on_circle(gens[i], circles[discs[i]])
            disc_next = Disc2(img.r, name='Disc_' + str(i), location=-0.001 * Y,
                              center=Vector([np.real(img.c), np.imag(img.c), 0]),
                              color=colors[c_index[i]],
                              resolution=[np.maximum(2, int(20 * img.r)), np.maximum(10, int(100 * img.r))],
                              rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            disc_next.appear(begin_time=t0, transition_time=1)
            disc_next.move(direction=(1) * Y, begin_time=t0, transition_time=1)
            coords.add_object(disc_next)
            t0 += 1

        print("finished ", t0)

    def full_picture(self):
        cues = self.sub_scenes['full_picture']
        t0 = 0  # cues['start']

        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=2, lengths=[20, 10], domains=[[-10, 10], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 10],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-10, 10.1, 2), np.arange(-5, 5.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        # setup initial conditions
        R1 = 1.5
        R2 = R1 * 3.247639506  # touching condition calculated in asymmetric_touching_discs.mw
        s = 4  # scaling
        k = k_of_s(s)

        a = np.array([[s + 1, 2 * R2 * (1 - s)], [(1 - s) / 2 / R2, s + 1]])
        A = np.array([[s + 1, 2 * R2 * (s - 1)], [(s - 1) / 2 / R2, s + 1]])
        b = np.array([[s + 1, 2 * R1 * 1j * (1 - s)], [(1 - s) / 2 / R1 / 1j, s + 1]])
        B = np.array([[s + 1, 2 * R1 * 1j * (s - 1)], [(s - 1) / 2 / R1 / 1j, s + 1]])

        # vertically the smaller disc
        rs = R1 * (1 / k - k)
        cs = R1 * 1j * (1 / k + k)
        # horizontally the bigger disc
        rb = R2 * (1 / k - k)
        cb = R2 * (1 / k + k)

        circles = [
            IndraCircle(-cb, rb),  # a
            IndraCircle(-cs, rs),  # b
            IndraCircle(cb, rb),  # A
            IndraCircle(cs, rs),  # B
        ]

        bfs = BreadFirstSearch(a, b, A, B, 6)
        colors = BlenderModel().colors

        duration = 40
        thickness = 0.1
        coords.rotate(rotation_euler=[-np.pi / 4, 0, 0], begin_time=t0, transition_time=duration / 4, compensate=True)
        removes = []
        dt = duration / bfs.all_elements()

        for i, circle in enumerate(circles):
            disc_next = Disc2(circle.r, center=Vector([np.real(circle.c), np.imag(circle.c), 0]), color=colors[i],
                              resolution=[20, 100], rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness)
            disc_next.appear(begin_time=t0, transition_time=1)
            removes.append(disc_next)
            coords.add_object(disc_next)
            t0 += dt

        Y = Vector([0, -thickness, 0])
        labels = ['a', 'b', 'A', 'B']
        for l in range(0, bfs.level_max - 1):
            for i in range(bfs.num[l], bfs.num[l + 1]):
                for j in range(0, 4):
                    if bfs.inv[bfs.tags[i]] != j:
                        img = moebius_on_circle(bfs.group[i], circles[j])
                        disc_next = Disc2(img.r, name='Disc_' + labels[bfs.tags[i]] + '_' + labels[j],
                                          center=Vector([np.real(img.c), np.imag(img.c), 0]),
                                          color=colors[j],
                                          resolution=[np.maximum(2, int(20 * img.r)), np.maximum(10, int(100 * img.r))],
                                          rotation_euler=[np.pi / 2, 0, 0],
                                          solid=thickness)
                        disc_next.appear(begin_time=t0, transition_time=1)
                        disc_next.move(direction=(l + 1) * Y, begin_time=t0, transition_time=1)
                        if l < bfs.level_max - 2:
                            removes.append(disc_next)
                        coords.add_object(disc_next)
                        t0 += dt

        coords.axes[0].disappear(begin_time=10)
        coords.axes[1].disappear(begin_time=10)

        t0 = coords.rotate(rotation_euler=[0, 0, 0], begin_time=t0)

        duration = 5
        dt = duration / len(removes)
        while len(removes) > 0:
            removes.pop(0).disappear(begin_time=t0, transition_time=0)
            t0 += dt

        t0 += 0.5

        print("finished at ", t0)

    def full_picture_spider(self):
        cues = self.sub_scenes['full_picture_spider']
        t0 = 0  # cues['start']

        ibpy.set_hdri_background("belfast_sunset_puresky_4k", 'exr', transparent=False)
        ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0.1)

        empty = EmptyCube(location=[0, 0, -10])
        ibpy.set_camera_view_to(empty)
        ibpy.set_camera_location(location=[0, 0, 0])
        camera_circle = BezierCircle(radius=20)
        ibpy.set_camera_follow(camera_circle)

        origin = Vector([0, 0, -10])
        coords = CoordinateSystem(dim=2, lengths=[20, 10], domains=[[-10, 10], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 10],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-10, 10.1, 2), np.arange(-5, 5.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)
        t0 += 0.1
        coords.disappear(begin_time=t0, transition_time=0.1)
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        # setup initial conditions
        R1 = 1.5
        R2 = R1 * 3.247639506  # touching condition calculated in asymmetric_touching_discs.mw
        s = 4  # scaling
        k = k_of_s(s)

        a = np.array([[s + 1, 2 * R2 * (1 - s)], [(1 - s) / 2 / R2, s + 1]])
        A = np.array([[s + 1, 2 * R2 * (s - 1)], [(s - 1) / 2 / R2, s + 1]])
        b = np.array([[s + 1, 2 * R1 * 1j * (1 - s)], [(1 - s) / 2 / R1 / 1j, s + 1]])
        B = np.array([[s + 1, 2 * R1 * 1j * (s - 1)], [(s - 1) / 2 / R1 / 1j, s + 1]])

        # vertically the smaller disc
        rs = R1 * (1 / k - k)
        cs = R1 * 1j * (1 / k + k)
        # horizontally the bigger disc
        rb = R2 * (1 / k - k)
        cb = R2 * (1 / k + k)

        circles = [
            IndraCircle(-cb, rb),  # a
            IndraCircle(-cs, rs),  # b
            IndraCircle(cb, rb),  # A
            IndraCircle(cs, rs),  # B
        ]

        bfs = BreadFirstSearch(a, b, A, B, 6)
        colors = BlenderModel().colors

        t0 = coords.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=t0, transition_time=1, compensate=True) + 0.5

        duration = 4
        thickness = 0.1

        dt = duration / bfs.all_elements()

        for i, circle in enumerate(circles):
            disc_next = Disc2(circle.r, center=Vector([np.real(circle.c), np.imag(circle.c), 0]), color=colors[i],
                              resolution=[50, 100], rotation_euler=[np.pi / 2, 0, 0],
                              solid=thickness, bevel=0.025, smooth=1, roughness=0.1, metallic=1)
            disc_next.appear(begin_time=t0, transition_time=1)
            coords.add_object(disc_next)
            t0 += dt

        Y = Vector([0, -thickness, 0])
        labels = ['a', 'b', 'A', 'B']
        for l in range(0, bfs.level_max - 1):
            for i in range(bfs.num[l], bfs.num[l + 1]):
                for j in range(0, 4):
                    if bfs.inv[bfs.tags[i]] != j:
                        img = moebius_on_circle(bfs.group[i], circles[j])
                        disc_next = Disc2(img.r, name='Disc_' + labels[bfs.tags[i]] + '_' + labels[j],
                                          center=Vector([np.real(img.c), np.imag(img.c), 0]),
                                          color=colors[j],
                                          resolution=[np.maximum(2, int(20 * img.r)), np.maximum(10, int(100 * img.r))],
                                          rotation_euler=[np.pi / 2, 0, 0],
                                          solid=thickness, bevel=0.025, smooth=1, roughness=0.1, metallic=1)
                        disc_next.appear(begin_time=t0, transition_time=1)
                        disc_next.move(direction=(l + 1) * Y, begin_time=t0, transition_time=1)
                        coords.add_object(disc_next)
                        t0 += dt

        eps = 0.001
        average = 100
        scale = 2
        duration = 30
        height = 0.53
        curve = read_complex_data("blender_model_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)
        fx2 = lambda x: function_from_complex_list(curve, x, scale=scale, average=average)

        fractal = Function([fx], None, domain=[0, 1], mode='PARAMETRIC', name='a_full',
                           location=[1, 0, -10 + height],
                           rotation_euler=[0, 0, np.pi / 2],
                           color='important', extrude=0.025, thickness=0.2, metallic=1, roughness=0,
                           num_points=resolution, emission=1)

        average_fractal = Function([fx2], None, domain=[0, 1], mode='PARAMETRIC',
                                   location=[1, 0, -10 + height],
                                   rotation_euler=[0, 0, np.pi / 2],
                                   name='average', color='example', extrude=0, thickness=0,
                                   numpoints=resolution / average)

        fractal.grow(begin_time=t0, transition_time=duration)

        spider = AnimBObject('Spider', object_name='widow', armature='Armature', color='joker', scale=1,
                             emission=0.75)
        spider.armature.ref_obj.scale = [0.1, 0.1, 0.1]
        spider.appear(begin_time=t0, transition_time=1)

        spider.armature.follow(average_fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')

        ibpy.camera_follow(camera_circle, begin_time=t0, transition_time=duration, initial_value=0, final_value=1)

        t0 += duration
        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5

        print("finished at ", t0)

    def groups(self):
        cues = self.sub_scenes['discs_and_group']
        t0 = 0.5  # cues['start']

        display = Display(location=[-7, 0, -3], flat=True, scales=[4, 3], number_of_lines=10)
        t0 = display.appear(begin_time=t0) + 0.5

        r3 = np.sqrt(3)
        mapping = {r3 / 2: r"\tfrac{\sqrt{3}}{2}",
                   -r3 / 2: r"-\tfrac{\sqrt{3}}{2}",
                   0.5: r"\tfrac{1}{2}",
                   -0.5: r"-\tfrac{1}{2}",
                   -1: "-1",
                   1: "1",
                   0: "0",
                   }

        a_matrix = np.matrix([[-1, 0], [0, 1]])
        b_matrix = np.matrix([[1 / 2, r3 / 2], [r3 / 2, -1 / 2]])

        a_bob = BMatrix(a_matrix, pre_word=r'a=', mapping=mapping)
        b_bob = BMatrix(b_matrix, pre_word=r'b=', mapping=mapping)

        t0 = 0.5 + display.write_text_in(a_bob, line=0, indent=1, begin_time=t0, transition_time=2)
        t0 = 0.5 + display.write_text_in(b_bob, line=2, indent=1, begin_time=t0, transition_time=2)

        a = Element(a_matrix, 'a', mapping=mapping)
        b = Element(b_matrix, 'b', mapping=mapping)

        group = Group(a, b, dim=2, mode='free', level=3, action='left', mapping=mapping, unit_string='1')

        tree = Tree(group.elements[0], 7, 6, color_function=color_function, action='left', name='t1',
                    display_mode='word')
        tree.move(direction=[-11, 0, 0], begin_time=0, transition_time=0)
        t0 = tree.appear(begin_time=t0, transition_time=10) + 0.5

        tree2 = Tree(group.elements[0], 15, 15, color_function=color_function, action='left', name='t2',
                     node_circles=False, display_mode='element')
        tree2.move(direction=[-4, 0, -9], begin_time=0, transition_time=0)

        t0 = 0.5 + tree2.appear_set(node_set={0, 1, 2, 3}, begin_time=t0, transition_time=4)

        colors1 = flatten([['joker'] * 2, ['text']])
        colors2 = flatten([['important'] * 2, ['text']])
        lines = [
            SimpleTexBObject(r"a^2=1", aligned='left', color=colors1),
            SimpleTexBObject(r"b^2=1", aligned='left', color=colors2),
            SimpleTexBObject(r"bab=aba", aligned='left',
                             color=['important', 'joker', 'important', 'text', 'joker', 'important', 'joker']),
            SimpleTexBObject(r"D_3=\{a^2=b^2=(ab)^3=1\}", aligned='left'),
        ]

        t0 = 0.5 + display.write_text_in(lines[0], line=4, indent=1, begin_time=t0)

        tree.node("aa").next(begin_time=t0)
        t0 = 0.5 + tree.node("aab").next(begin_time=t0)

        tree.disappear_node("aa", begin_time=t0)
        tree.disappear_node("aab", begin_time=t0)

        t0 = 0.5 + tree2.disappear_node("aa", begin_time=t0)

        t0 = 0.5 + tree2.appear_words(set={"bb"}, begin_time=t0)
        t0 = 0.5 + display.write_text_in(lines[1], line=5, indent=1, begin_time=t0)
        tree.node("bba").next(begin_time=t0)
        t0 = 0.5 + tree.node("bb").next(begin_time=t0)

        tree.disappear_node("bb", begin_time=t0)
        tree.disappear_node("bba", begin_time=t0)

        t0 = 0.5 + tree2.disappear_node("bb", begin_time=t0)
        t0 = 0.5 + tree2.appear_words(set=["ba", "ab", "aba", "bab"], begin_time=t0, transition_time=4)

        t0 = 0.5 + display.write_text_in(lines[2], line=6, indent=1, begin_time=t0)
        t0 = 0.5 + tree.node("bab").next(begin_time=t0)
        tree.disappear_node("bab", begin_time=t0)
        t0 = 0.5 + tree2.disappear_node("bab", begin_time=t0)

        t0 = 0.5 + display.write_text_in(lines[3], line=8, indent=1, begin_time=t0)

        # create the triangle

        tree2.root_node.disappear(alpha=0.05, begin_time=t0)
        triangle = Triangle(offset=0, color='vertex_color', rotation_euler=[np.pi / 2, 0, 0], emission=0.5)
        triangle.appear(begin_time=t0)
        r3 = np.sqrt(3)
        one = SimpleTexBObject("1", aligned='center', location=1.5 * Vector([-0.5, 0, -r3 / 6]))
        one.write(begin_time=t0)

        two = SimpleTexBObject("2", aligned='center', location=1.5 * Vector([0.5, 0, -r3 / 6]))
        two.write(begin_time=t0)

        three = SimpleTexBObject("3", aligned='center', location=1.5 * Vector([0, 0, r3 / 3]))
        t0 = 0.5 + three.write(begin_time=t0)

        labelled_triangle = BObject(children=[triangle, one, two, three], location=[3.46, 0, 5.42])
        labelled_triangle.appear(begin_time=t0)

        # visualize the action on the triangle
        # the corresponding rotations have been found out experimentally in the blender file

        pi = np.pi
        a_quat = Quaternion(Vector([0, 0, 1]), pi)
        b_quat = Quaternion(Vector([r3 / 2, 0, 1 / 2]), pi)

        copy1 = labelled_triangle.copy()
        copy1.appear(begin_time=t0)
        tree2.node("a").disappear(alpha=0.05, begin_time=t0)
        t0 = 0.5 + copy1.move_to(target_location=[-0.205, 0, 1.67], begin_time=t0)
        t0 = 0.5 + copy1.rotate(rotation_quaternion=a_quat, begin_time=t0)

        copy2 = labelled_triangle.copy()
        copy2.appear(begin_time=t0)
        tree2.node("b").disappear(alpha=0.05, begin_time=t0)
        t0 = 0.5 + copy2.move_to(target_location=[7.26, 0, 2.81], begin_time=t0)
        t0 = 0.5 + copy2.rotate(rotation_quaternion=b_quat, begin_time=t0)

        copy3 = labelled_triangle.copy()
        copy3.appear(begin_time=t0)
        tree2.node("ba").disappear(alpha=0.05, begin_time=t0)
        t0 = 0.5 + copy3.move_to(target_location=[1.63, 0, -0.93], begin_time=t0)
        t0 = copy3.rotate(rotation_quaternion=a_quat, begin_time=t0)
        t0 = 0.5 + copy3.rotate(rotation_quaternion=b_quat @ a_quat, begin_time=t0)

        copy4 = labelled_triangle.copy()
        copy4.appear(begin_time=t0)
        tree2.node("ab").disappear(alpha=0.05, begin_time=t0)
        t0 = 0.5 + copy4.move_to(target_location=[5.39, 0, -2.10], begin_time=t0)
        t0 = copy4.rotate(rotation_quaternion=b_quat, begin_time=t0)
        t0 = 0.5 + copy4.rotate(rotation_quaternion=a_quat @ b_quat, begin_time=t0)

        copy5 = labelled_triangle.copy()
        copy5.appear(begin_time=t0)
        tree2.node("aba").disappear(alpha=0.05, begin_time=t0)
        t0 = 0.5 + copy5.move_to(target_location=[0.704, 0, -5.85], begin_time=t0)
        t0 = copy5.rotate(rotation_quaternion=a_quat, begin_time=t0)
        t0 = copy5.rotate(rotation_quaternion=b_quat @ a_quat, begin_time=t0)
        t0 = 0.5 + copy5.rotate(rotation_quaternion=a_quat @ b_quat @ a_quat, begin_time=t0)

        print('finished ', t0)

    def discs_and_group(self):
        self.disc_counter = 0
        cues = self.sub_scenes['discs_and_group']
        t0 = 0.5  # cues['start']

        camera_empty = EmptyCube(location=[6.5, 0, 0])
        ibpy.set_camera_location(location=[6.5, 0, 32])
        ibpy.set_camera_rotation(rotation=[0, 0, 0])
        set_camera_view_to(camera_empty)
        ibpy.set_sun_light(location=[0, 34, 35])
        remove_discs = []

        level_node_map = {}

        polar = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: 0,  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: 0,
            lambda u, v: 0
        )
        shift = Vector([19, 0, 0.125])
        identity = SimpleTexBObject(".", location=embedding(0, 0) + shift, rotation_euler=[np.pi / 2, 0, 0],
                                    aligned='center',
                                    color='drawing')
        identity.write(begin_time=t0, transition_time=0.1)
        root = TreeNode("I", identity)
        level_node_map[0] = [root]

        res = 20
        phi0 = -np.pi
        dphi = np.pi / 2

        # level = 1
        level_letters = []
        letter = 'a'
        for i in range(4):
            level_letters.append(letter)
            letter = next_letter(letter)

        nodes = []

        duration = 3
        dt = 0.8 * duration / 4
        r2 = np.sqrt(2)
        ir2 = 1 / r2
        q0 = Quaternion([ir2, 0, 0, ir2])
        colors = BlenderModel().colors
        for i, l in enumerate(level_letters):
            phi = phi0 + i * dphi
            q = polar.local_frame_quaternion(2, phi)
            location = polar.embedding(2, phi) + shift
            label = SimpleTexBObject(l, aligned='center',
                                     location=location,
                                     rotation_quaternion=q0 @ q,
                                     name=l + '_1', color=colors[i],
                                     outlined='background',
                                     text_size='huge')
            label.write(begin_time=t0, transition_time=dt)
            connection = Curve([lambda t: partial(polar.embedding, v=phi)(2 * t)], domain=[0.0, 0.8],
                               num_points=res, color='drawing', thickness=0.35, name='Connection_' + l + '_1',
                               location=shift)
            node = TreeNode(l, label, connection, color=colors[i])
            node.connect(begin_time=t0, transition_time=dt)
            node.parent = root
            nodes.append(node)
            t0 += dt * 1.25

        level_node_map[1] = nodes
        t0 += 1

        offset = phi0
        for level in range(2, 4):
            duration = 3 * level
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 0.8 * duration / len(old_nodes) / 3
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = i * dphi
                letter = parent.content
                letter = previous_letter(letter)
                for j in range(-1, 2):
                    phi_j = offset + phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += (0.01 * normal + shift)
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    label = SimpleTexBObject(letter, aligned='center',
                                             location=location,
                                             rotation_quaternion=q0 @ q, color=parent.color,
                                             outlined='background', text_size='huge')
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: polar.embedding(interpol(2 * (level - 1) + 0.15,
                                                           2 * (level - 1) + 1.8, t),
                                                  interpol(offset + phi, phi_j, t))
                    ],
                        domain=[0.1, 0.9],
                        num_points=res,
                        color='drawing', thickness=0.25, location=shift,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    node = TreeNode(letter, label, connection=connection, color=parent.color)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt * 1.25
                    letter = next_letter(letter)
            offset -= d3phi
            level_node_map[level] = nodes
            t0 += 1

        # create the discs
        t0 = 0.5
        ibpy.camera_move(shift=[0, -21.5, -7], begin_time=10, transition_time=10)
        # alphas = [1,1,0.01,1] # dim the third disc to make the tree visible
        # alpha_counter = 0
        for level in range(1, 6):
            dfs = DepthFirstSearchByLevel(BlenderModel, max_level=level)
            dfs.generate_tree()
            leaves = dfs.get_leaves()

            if level < 4:
                duration = 3 * level
            else:
                duration = 9

            dt = 0.8 * duration / len(leaves)
            for leave in leaves:
                disc = self.disc_from_circle(leave.circle, begin_time=t0,
                                             color=leave.color,
                                             offset=level)  # ,alpha=alphas[alpha_counter])
                # if alpha_counter>-1:
                #     alpha_counter+=1
                #     if alpha_counter==3:
                #         alpha_counter=-1
                if level < 5:
                    remove_discs.append(disc)
                t0 += dt * 1.25
            t0 += 1

        for disc in remove_discs:
            disc.disappear(begin_time=t0)

        ibpy.camera_move(shift=[1.5, 21.5, 7], begin_time=t0, transition_time=5)
        camera_empty.move(direction=[1.5, 0, 0], begin_time=t0, transition_time=5)
        ibpy.camera_zoom(lens=31.5, begin_time=t0, transition_time=5)
        t0 += 7

        print("finished ", t0)

    def disc_from_circle(self, circle, color, begin_time=0, offset=0,height=0.1,
                         transition_time=DEFAULT_ANIMATION_TIME, alpha=1,
                         brighter=0,name="Disc"):
        n = np.sqrt(circle.r)
        resolution = [2, np.minimum(500, np.maximum(10, int(80 * n)))]

        if color == 'example' or 'gray' in color:
            bright_factor = 1.5
        else:
            bright_factor = 3

        disc = Disc2(r=circle.r, center=Vector([np.real(circle.c), np.imag(circle.c), offset * height*1.1]),
                     resolution=resolution, solid=height,
                     rotation_euler=[0, 0, 0]
                     , color=color, name=name, alpha=alpha,
                     brighter=bright_factor * brighter)
        self.disc_counter += 1
        disc.appear(begin_time=begin_time, transition_time=transition_time)
        return disc

    def disc_from_circle2(self, circle, color,  offset=0,height=0.1,
                         alpha=1,
                         brighter=0):

        resolution = 100/np.sqrt(offset+1)

        if color=='custom1':
            b_factor=2.75
        else:
            b_factor=0.75

        disc = Disc2(r=circle.r, center=Vector(),
                     resolution=resolution, solid=height,
                     rotation_euler=[0, 0, 0]
                     , color="fake_glass_"+color, name='Disc_' + str(self.disc_counter), alpha=alpha,
                     ior=1,brighter = b_factor*offset, transmission=1-0.05*brighter,roughness=0.01*brighter,metallic=0.01*brighter)
        self.disc_counter += 1
        disc.move_to(target_location=Vector([np.real(circle.c), np.imag(circle.c), offset * height*1.1]),transition_time=0)
        return disc

    def tree_extension(self):
        cues = self.sub_scenes['tree_extension']
        t0 = 0.5  # cues['start']
        debug = 0

        camera_circle = BezierCircle(radius=12.25, location=[0, 0, 0], rotation_euler=[-np.pi / 2, 0, -np.pi / 2])
        ibpy.set_camera_follow(camera_circle)
        camera_empty = EmptyCube()
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_location(location=[0, 0, 0])
        ibpy.camera_follow(camera_circle, initial_value=0, final_value=0.01, begin_time=0, transition_time=0)

        level_node_map = {}
        max_level = 2
        tree_parts = []

        polar = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: 0,  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: 0,
            lambda u, v: 0
        )

        paraboloid = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: hull(u, radius=4, arc=0.9 * np.pi / 2),  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: d_hull(u, radius=4, arc=0.9 * np.pi / 2),
            lambda u, v: 0
        )

        identity = SimpleTexBObject(".", location=embedding(0, 0), rotation_euler=[np.pi / 2, 0, 0], aligned='center',
                                    color='drawing')
        identity.write(begin_time=t0, transition_time=0.1)
        root = TreeNode("I", identity)
        level_node_map[0] = [root]
        tree_parts.append(identity)

        res = 20
        phi0 = -np.pi
        dphi = np.pi / 2

        # level = 1
        level_letters = []
        letter = 'a'
        for i in range(4):
            level_letters.append(letter)
            letter = next_letter(letter)

        nodes = []

        duration = 3
        dt = 0.8 * duration / 4
        r2 = np.sqrt(2)
        ir2 = 1 / r2
        q0 = Quaternion([ir2, 0, 0, ir2])

        model = ApollonianModel()

        morph_connections = []

        # first four nodes
        for i, l in enumerate(level_letters):
            phi = phi0 + i * dphi
            q = polar.local_frame_quaternion(2, phi)
            location = polar.embedding(2, phi)
            if i == 0:
                label = SimpleTexBObject(l, aligned='center',
                                         location=location,
                                         rotation_quaternion=q0 @ q,
                                         name=l + '_1', color=model.colors[i],
                                         outlined='text')
            else:
                label = SimpleTexBObject(l, aligned='center',
                                         location=location,
                                         rotation_quaternion=q0 @ q,
                                         name=l + '_1', color=model.colors[i])
            label.write(begin_time=t0, transition_time=dt)
            connection = Curve([
                lambda t: partial(polar.embedding, v=phi)(2 * t),
                lambda t: partial(paraboloid.embedding, v=phi)(2 * t),
            ], domain=[0.0, 0.9],
                num_points=res, color='drawing', thickness=0.25, name='Connection_' + l + '_1')
            morph_connections.append(connection)
            node = TreeNode(l, label, connection, color=model.colors[i], u=2, v=phi)
            node.connect(begin_time=t0, transition_time=dt)
            node.parent = root
            nodes.append(node)
            tree_parts.append(label)
            tree_parts.append(connection)
            t0 += dt * 1.25
        level_node_map[1] = nodes
        t0 += 1

        # regular tree to level max_level

        for level in range(2, max_level + 1):
            duration = 3 * level
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 0.8 * duration / len(old_nodes) / 3
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = parent.v
                letter = parent.content
                letter = previous_letter(letter)
                for j in range(-1, 2):
                    phi_j = phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += 0.01 * normal
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    if 'gray' in parent.color:
                        label = SimpleTexBObject(letter, aligned='center',
                                                 location=location,
                                                 rotation_quaternion=q0 @ q, color=parent.color, outlined='text')
                    else:
                        label = SimpleTexBObject(letter, aligned='center',
                                                 location=location,
                                                 rotation_quaternion=q0 @ q, color=parent.color)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: polar.embedding(interpol(2 * (level - 1) + 0,
                                                           2 * (level - 1) + 2, t),
                                                  interpol(phi, phi_j, t)),
                        lambda t: paraboloid.embedding(interpol(2 * (level - 1) + 0,
                                                                2 * (level - 1) + 2.05, t),
                                                       interpol(phi, phi_j, t)),

                    ],
                        domain=[0.1, 0.95],
                        num_points=res,
                        color='drawing', thickness=0.25,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    morph_connections.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color, u=2 * level, v=phi_j)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt * 1.25
                    letter = next_letter(letter)
                    tree_parts.append(label)
                    tree_parts.append(connection)
            level_node_map[level] = nodes
            t0 += 1

        # morph tree
        for connection in morph_connections:
            connection.next(begin_time=t0)

        for level_nodes in level_node_map.values():
            for node in level_nodes:
                node.bob.move_to(target_location=paraboloid.embedding(node.u - 0.05, node.v), begin_time=t0)
                q = paraboloid.local_frame_quaternion(node.u - 0.05, node.v)
                node.bob.rotate(rotation_quaternion=q @ q0, begin_time=t0)

        ibpy.camera_zoom(lens=25, begin_time=t0)

        # change camera position
        ibpy.camera_follow(camera_circle, initial_value=0.01, final_value=0.25, begin_time=t0)
        camera_empty.move(direction=[0, 0, -4], begin_time=t0)

        time_first_tree_rotation = t0

        # slow example extension for the node $a$

        for node in level_node_map[max_level]:
            if node.content == 'a' and node.parent.content == 'a':
                selected_node = node

        t0 += 1.5
        time_second_tree_rotation = t0 + 2
        time_third_tree_rotation = t0 + 6
        time_fourth_tree_rotation = t0 + 10

        if debug:
            extend = 4
        else:
            extend = 8
        ext_connections = []
        ext_labels = []
        morph_connections2 = []
        leaves = []
        t0 = extension_for_node(ext_connections, ext_labels, selected_node, max_level, level, extend, level_node_map,
                                polar, paraboloid, q0, t0, 1, [0, 3, 3, 3, 0, 0.5, 0], res, tree_parts,
                                morph_connections2, leaves)

        if debug:
            extend = 2
        else:
            extend = 4
        time_last_rotation_start = t0
        for node in level_node_map[max_level]:
            if node != selected_node:
                t0 = extension_for_node(ext_connections, ext_labels, node, max_level, level, extend, level_node_map,
                                        polar, paraboloid, q0, t0, 0.1, 0, res, tree_parts, morph_connections2, leaves)
        time_last_rotation_end = t0

        # rotate the tree
        tree_full = BObject(children=tree_parts, rotation_euler=[0, 0, -np.pi / 2])
        tree_full.appear()
        tree_full.rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=time_first_tree_rotation)
        tree_full.rotate(rotation_euler=[0, 0, 0], begin_time=time_second_tree_rotation)
        tree_full.rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=time_second_tree_rotation + 2)
        ibpy.camera_zoom(lens=15, begin_time=time_fourth_tree_rotation + 2)
        camera_empty.move(direction=[0, 0, -3.5], begin_time=time_fourth_tree_rotation + 2)
        tree_full.rotate(rotation_euler=[0, 0, -np.pi], begin_time=time_third_tree_rotation)
        tree_full.rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=time_third_tree_rotation + 2)
        tree_full.rotate(rotation_euler=[0, 0, np.pi / 2], begin_time=time_fourth_tree_rotation)
        tree_full.rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=time_fourth_tree_rotation + 2)
        delta = time_last_rotation_end - time_last_rotation_start
        tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2], begin_time=time_last_rotation_start,
                         transition_time=1.25 * delta)

        t0 = time_last_rotation_end + 0.5
        tree_full.move(direction=[-7, 0, 0], begin_time=t0, transition_time=2)
        camera_empty.move(direction=[0, 0, 1], begin_time=t0, transition_time=2)
        t0 = camera_circle.move(direction=[0, 0, -6.5], begin_time=t0, transition_time=2)

        # first fixed point
        for i in range(1, 8):
            ext_connections[2 * i].change_color(new_color='example', begin_time=t0)

        display = Display(location=[7, 0, -6.5], flat=True, number_of_lines=20, scales=[7, 8],
                          rotation_euler=[np.pi / 2, 0, np.pi])
        t0 = 0.5 + display.appear(begin_time=t0)
        title = SimpleTexBObject(r"\text{Fixed points of the commutators}", aligned='center', color='example')
        t0 = 0.5 + display.write_title(title, begin_time=t0)

        if not debug:
            c1 = BMatrix(np.array([["-5-2i", "-2i"], ["8-6i", "3+2i"]]), pre_word="BAba=", color='text')
            f1 = SimpleTexBObject(r"f_1=-\tfrac{1}{5}-\tfrac{2}{5}i", color='example')

            t0 = 0.5 + display.write_text_in(c1, line=1, indent=1, begin_time=t0, transition_time=2)
            t0 = 0.5 + display.write_text_in(f1, line=1, indent=7, begin_time=t0, transition_time=1)
            for i in range(7, 0, -1):
                t0 = ext_labels[2 * i].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[2 * i].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_1", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[0].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[0].replace(fix, begin_time=t0)

            delta = -2 * np.pi / 24
            # second fixed point
            tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + delta], begin_time=t0)
            for i in range(1, 8):
                ext_connections[2 * i + 1].change_color(new_color='example', begin_time=t0)

            c2 = BMatrix(np.array([["-5+2i", "-2i"], ["-8-6i", "3-2i"]]), pre_word="bABa=", color='text')
            f2 = SimpleTexBObject(r"f_2=\tfrac{1}{5}-\tfrac{2}{5}i", color='example')

            t0 = 0.5 + display.write_text_in(c2, line=3, indent=1, begin_time=t0, transition_time=2)
            t0 = 0.5 + display.write_text_in(f2, line=3, indent=7, begin_time=t0, transition_time=1)
            for i in range(8, 1, -1):
                t0 = ext_labels[2 * i - 1].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[2 * i - 1].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_2", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[1].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[1].replace(fix, begin_time=t0)

            # third fixed point
            idx = [26, 28, 30]
            tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + 2 * delta], begin_time=t0)
            for id in idx:
                ext_connections[id].change_color(new_color='example', begin_time=t0)

            c3 = BMatrix(np.array([["-1-2i", "-2i"], ["2i", "-1+2i"]]), pre_word="aBAb=", color='text')
            f3 = SimpleTexBObject(r"f_3=-1", color='example')

            t0 = 0.5 + display.write_text_in(c3, line=5, indent=1, begin_time=t0, transition_time=2)
            t0 = 0.5 + display.write_text_in(f3, line=5, indent=7, begin_time=t0, transition_time=1)
            for id in reversed(idx):
                t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_3", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[24].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[24].replace(fix, begin_time=t0)

            # fourth fixed point
            idx = [27, 29, 31]
            tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + 3 * delta], begin_time=t0)
            for id in idx:
                ext_connections[id].change_color(new_color='example', begin_time=t0)

            c4 = SimpleTexBObject(r"ABab = (BAba)^{-1}", color='text')
            t0 = 0.5 + display.write_text_in(c4, line=7, indent=1, begin_time=t0, transition_time=1)

            for id in reversed(idx):
                t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_1", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[25].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[25].replace(fix, begin_time=t0)

            idx_list = [
                [34, 36, 38],
                [35, 37, 39],
                [42, 44, 46],
                [43, 45, 47],
            ]
            fxps = ["f_1", "f_2", "f_3", "f_1"]

            # next four fps
            rot = 3 * delta
            for idx, fp in zip(idx_list, fxps):
                rot += delta
                tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + rot], begin_time=t0)
                for id in idx:
                    ext_connections[id].change_color(new_color='example', begin_time=t0)
                t0 += 0.5
                for id in reversed(idx):
                    t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                    t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
                fix = SimpleTexBObject(fp, color='example', aligned="center")
                s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
                ext_labels[idx[0] - 2].adopt(s)
                s.grow(begin_time=t0)
                t0 = 0.5 + ext_labels[idx[0] - 2].replace(fix, begin_time=t0)

            # next
            idx = [50, 52, 54]
            rot += delta
            tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + rot], begin_time=t0)
            for id in idx:
                ext_connections[id].change_color(new_color='example', begin_time=t0)

            c4 = BMatrix(np.array([["-1-2i", "2i"], ["-2i", "-1+2i"]]), pre_word="baBA=", color='text')
            f4 = SimpleTexBObject(r"f_4=1", color='example')

            t0 = 0.5 + display.write_text_in(c4, line=9, indent=1, begin_time=t0, transition_time=2)
            t0 = 0.5 + display.write_text_in(f4, line=9, indent=7, begin_time=t0, transition_time=1)
            for id in reversed(idx):
                t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_4", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[idx[0] - 2].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[idx[0] - 2].replace(fix, begin_time=t0)

            # next
            idx = [51, 53, 55]
            rot += delta
            tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + rot], begin_time=t0)
            for id in idx:
                ext_connections[id].change_color(new_color='example', begin_time=t0)

            c4 = SimpleTexBObject(r"BabA = (aBAb)^{-1}", color='text')
            t0 = 0.5 + display.write_text_in(c4, line=11, indent=1, begin_time=t0, transition_time=1)

            for id in reversed(idx):
                t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_3", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[idx[0] - 2].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[idx[0] - 2].replace(fix, begin_time=t0)

            idx_list = [
                [58, 60, 62],
                [59, 61, 63],
                [66, 68, 70],
                [67, 69, 71],
            ]
            fxps = ["f_3", "f_1", "f_4", "f_3"]

            # next four fps
            for idx, fp in zip(idx_list, fxps):
                rot += delta
                tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + rot], begin_time=t0)
                for id in idx:
                    ext_connections[id].change_color(new_color='example', begin_time=t0)
                t0 += 0.5
                for id in reversed(idx):
                    t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                    t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
                fix = SimpleTexBObject(fp, color='example', aligned="center")
                s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
                ext_labels[idx[0] - 2].adopt(s)
                s.grow(begin_time=t0)
                t0 = 0.5 + ext_labels[idx[0] - 2].replace(fix, begin_time=t0)

            # next
            idx = [74, 76, 78]
            rot += delta
            tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + rot], begin_time=t0)
            for id in idx:
                ext_connections[id].change_color(new_color='example', begin_time=t0)

            c4 = SimpleTexBObject(r"AbaB = (bABa)^{-1}", color='text')
            t0 = 0.5 + display.write_text_in(c4, line=13, indent=1, begin_time=t0, transition_time=1)

            for id in reversed(idx):
                t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_2", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[idx[0] - 2].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[idx[0] - 2].replace(fix, begin_time=t0)

            # next
            idx = [75, 77, 79]
            rot += delta
            tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + rot], begin_time=t0)
            for id in idx:
                ext_connections[id].change_color(new_color='example', begin_time=t0)

            c4 = SimpleTexBObject(r"abAB = (baBA)^{-1}", color='text')
            t0 = 0.5 + display.write_text_in(c4, line=15, indent=1, begin_time=t0, transition_time=1)

            for id in reversed(idx):
                t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
            fix = SimpleTexBObject(r"f_4", color='example', aligned="center")
            s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
            ext_labels[idx[0] - 2].adopt(s)
            s.grow(begin_time=t0)
            t0 = 0.5 + ext_labels[idx[0] - 2].replace(fix, begin_time=t0)

            # the remaining

            idx_list = [
                [82, 84, 86],
                [83, 85, 87],
                [90, 92, 94],
                [91, 93, 95],
                [98, 100, 102],
                [99, 101, 103],
                [18, 20, 22],
                [19, 21, 23],
            ]
            fxps = ["f_4", "f_3", "f_2", "f_4", "f_1", "f_2", "f_2", "f_4"]

            for idx, fp in zip(idx_list, fxps):
                rot += delta
                tree_full.rotate(rotation_euler=[0, 0, -5 * np.pi / 2 + rot], begin_time=t0)
                for id in idx:
                    ext_connections[id].change_color(new_color='example', begin_time=t0)
                t0 += 0.5
                for id in reversed(idx):
                    t0 = ext_labels[id].disappear(begin_time=t0, transition_time=0.1)
                    t0 = ext_connections[id].disappear(begin_time=t0, transition_time=0.1)
                fix = SimpleTexBObject(fp, color='example', aligned="center")
                s = Sphere(0.25, color="fake_glass_drawing", roughness=0, transmission=1)
                ext_labels[idx[0] - 2].adopt(s)
                s.grow(begin_time=t0)
                t0 = 0.5 + ext_labels[idx[0] - 2].replace(fix, begin_time=t0)

        # un-morph tree
        tree_full.move(direction=[-1, 0, -6], begin_time=t0)
        t0 = tree_full.rotate(rotation_euler=[-np.pi / 2, 0, -4 * np.pi], begin_time=t0, transition_time=2)
        tree_full.rescale(rescale=[1.5, 1.5, 1], begin_time=t0)

        for connection in morph_connections:
            connection.previous(begin_time=t0)

        for connection in morph_connections2:
            connection.next(begin_time=t0)

        lev = 0  # only retransform upto third level
        for level_nodes in level_node_map.values():
            if lev < 4:
                for node in level_nodes:
                    node.bob.move_to(target_location=polar.embedding(node.u, node.v), begin_time=t0)
                    q = polar.local_frame_quaternion(node.u, node.v)
                    node.bob.rotate(rotation_quaternion=q @ q0, begin_time=t0)
                lev += 1

        t0 = display.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=1)

        example = SimpleTexBObject(r"\text{Example calculation}", aligned='center', color='example')
        t0 = 0.5 + display.write_back_title(example, begin_time=t0)

        origin = Vector([8, 0.25, -9.5])
        coords = CoordinateSystem(dim=2, lengths=[8, 8], domains=[[-1, 1], [-1., 1.]], radii=[0.03, 0.03],
                                  all_n_tics=[4, 4],
                                  labels=[r"\mathbb{R}", r"i\mathbb{R}"],
                                  all_tic_labels=[np.arange(-1, 1.1, 0.5), np.arange(-1, 1.1, 0.5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', ''],
                                  location_of_origin=origin,
                                  label_positions=['center', 'center'],
                                  include_zeros=[False, False],
                                  label_closenesses=[0.5, 1],
                                  name='CoordinateSystem',
                                  rotation_euler=[0, 0, np.pi])

        # convert tree to curve
        old_i = -1
        old_j = -1
        old_k = -1
        dt = 0.1
        k_count = 0
        level_1 = ['a', 'b', 'A', 'B']
        level_2 = ['B', 'a', 'b', 'a', 'b', 'A', 'b', 'A', 'B', 'A', 'B', 'a']
        fs = [2, 4, 1, 2, 3, 1, 1, 2, 3, 1, 4, 3, 3, 1, 4, 3, 2, 4, 4, 3, 2, 4, 1, 2]

        # swap two entries
        list = morph_connections2
        list[0], list[2] = list[2], list[0]
        list[1], list[3] = list[3], list[1]

        for i in range(4):
            if old_i > -1:
                morph_connections[old_i].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                                     transition_time=dt)
                if old_j > -1:
                    morph_connections[old_j].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                         begin_time=t0, transition_time=dt)
                    old_j = -1
                    if old_k > -1:
                        morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                              begin_time=t0, transition_time=dt)
                        old_k = -1
                t0 += dt
            t0 = morph_connections[i].highlight(color='example', emission=0.5, begin_time=t0, transition_time=dt)
            old_i = i
            for j in range(3):
                new_j = 4 + 3 * i + j
                if old_j > -1:
                    morph_connections[old_j].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                         begin_time=t0, transition_time=dt)
                    if old_k > -1:
                        morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                              begin_time=t0, transition_time=dt)
                        old_k = -1
                    t0 += dt
                t0 = morph_connections[new_j].highlight(color='example', emission=0.5, begin_time=t0,
                                                        transition_time=dt)
                old_j = new_j
                for k in range(0, 2):
                    if old_k > -1:
                        t0 = morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                                   begin_time=t0, transition_time=dt)
                    t0 = morph_connections2[k_count].highlight(color='example', emission=0.5, begin_time=t0,
                                                               transition_time=dt)
                    old_k = k_count
                    k_count += 1
                    if i == 0 and j == 0 and k == 0:
                        t0 = first_path_evaluation(display, coords, level_1[i], level_2[3 * i + j],
                                                   fs[2 * (3 * i + j) + k], t0=t0)
                    else:
                        t0 = create_next_point(coords, level_1[i], level_2[3 * i + j], fs[2 * (3 * i + j) + k], t0=t0)

        morph_connections[old_i].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                             transition_time=dt)
        morph_connections[old_j].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                             transition_time=dt)
        t0 = morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                                   transition_time=dt)

        print("finished at ", t0)

    def tree_extension2(self):
        cues = self.sub_scenes['tree_extension2']
        t0 = 0.5  # cues['start']
        debug = 0

        empty_circle = BezierCircle(radius=6.25, location=[0, 0, 0], rotation_euler=[0, 0, np.pi/6])
        camera_empty = EmptyCube()
        camera_empty.set_follow(empty_circle, influence=0)
        ibpy.set_camera_copy_location(camera_empty,offset=True)
        ibpy.set_camera_location(location=[0, 0, 12.25])
        ibpy.set_camera_rotation(rotation=[0,0,0])
        ibpy.camera_zoom(lens=45, transition_time=0)

        level_node_map = {}
        max_level = 3
        tree_parts = []

        polar = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: 0,  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: 0,
            lambda u, v: 0
        )

        paraboloid = Embedding(
            lambda u, v: u * np.cos(v),  # X
            lambda u, v: u * np.sin(v),  # Y
            lambda u, v: hull(u, radius=4, arc=0.9 * np.pi / 2),  # Z
            lambda u, v: np.cos(v),  # dXdu
            lambda u, v: -u * np.sin(v),  # dXdv
            lambda u, v: np.sin(v),  # dYdu
            lambda u, v: u * np.cos(v),  # dYdv
            lambda u, v: d_hull(u, radius=4, arc=0.9 * np.pi / 2),
            lambda u, v: 0
        )

        identity = SimpleTexBObject(".", location=embedding(0, 0), rotation_euler=[np.pi / 2, 0, 0], aligned='center',
                                    color='drawing')
        identity.write(begin_time=t0, transition_time=0.1)
        root = TreeNode("I", identity)
        level_node_map[0] = [root]
        tree_parts.append(identity)

        res = 20
        phi0 = -np.pi
        dphi = np.pi / 2

        # level = 1
        level_letters = []
        letter = 'a'
        for i in range(4):
            level_letters.append(letter)
            letter = next_letter(letter)

        nodes = []

        duration = 3
        dt = 0.8 * duration / 4
        r2 = np.sqrt(2)
        ir2 = 1 / r2
        q0 = Quaternion([ir2, 0, 0, ir2])

        model = ApollonianModel()

        morph_connections = []

        # first four nodes
        for i, l in enumerate(level_letters):
            phi = phi0 + i * dphi
            q = polar.local_frame_quaternion(2, phi)
            location = polar.embedding(2, phi)
            if i == 0:
                label = SimpleTexBObject(l, aligned='center',
                                         location=location,
                                         rotation_quaternion=q0 @ q,
                                         name=l + '_1', color=model.colors[i],
                                         outlined='text')
            else:
                label = SimpleTexBObject(l, aligned='center',
                                         location=location,
                                         rotation_quaternion=q0 @ q,
                                         name=l + '_1', color=model.colors[i])
            label.write(begin_time=t0, transition_time=dt)
            connection = Curve([
                lambda t: partial(polar.embedding, v=phi)(2 * t),
                lambda t: partial(paraboloid.embedding, v=phi)(2 * t),
            ], domain=[0.0, 0.9],
                num_points=res, color='drawing', thickness=0.25, name='Connection_' + l + '_1')
            morph_connections.append(connection)
            node = TreeNode(l, label, connection, color=model.colors[i], u=2, v=phi)
            node.connect(begin_time=t0, transition_time=dt)
            node.parent = root
            nodes.append(node)
            tree_parts.append(label)
            tree_parts.append(connection)
            t0 += dt * 1.25
        level_node_map[1] = nodes
        t0 += 1

        # regular tree to level max_level
        lenses = [25, 19]
        for level in range(2, max_level + 1):
            t0 = +ibpy.camera_zoom(lenses[level - 2], begin_time=t0)
            duration = 3 * level
            nodes = []
            old_nodes = level_node_map[level - 1]
            dt = 0.8 * duration / len(old_nodes) / 3
            dphi = 2 * np.pi / len(old_nodes)
            d3phi = dphi / 3
            for i, parent in enumerate(old_nodes):
                phi = parent.v
                letter = parent.content
                letter = previous_letter(letter)
                for j in range(-1, 2):
                    phi_j = phi + d3phi * j
                    location = polar.embedding(2 * level, phi_j)
                    normal = polar.unit_n(2 * level, phi_j)
                    location += 0.01 * normal
                    q = polar.local_frame_quaternion(2 * level, phi_j)
                    if 'gray' in parent.color:
                        label = SimpleTexBObject(letter, aligned='center',
                                                 location=location,
                                                 rotation_quaternion=q0 @ q, color=parent.color, outlined='text')
                    else:
                        label = SimpleTexBObject(letter, aligned='center',
                                                 location=location,
                                                 rotation_quaternion=q0 @ q, color=parent.color)
                    label.write(begin_time=t0, transition_time=dt)
                    connection = Curve([
                        lambda t: polar.embedding(interpol(2 * (level - 1) + 0,
                                                           2 * (level - 1) + 2, t),
                                                  interpol(phi, phi_j, t)),
                    ],
                        domain=[0.1, 0.95],
                        num_points=res,
                        color='drawing', thickness=0.25,
                        name='Connection_' + str(i) + "_" + str(j) + "_" + str(level))
                    morph_connections.append(connection)
                    node = TreeNode(letter, label, connection=connection, color=parent.color, u=2 * level, v=phi_j)
                    node.parent = parent
                    nodes.append(node)
                    node.connect(begin_time=t0, transition_time=dt)
                    t0 += dt * 1.25
                    letter = next_letter(letter)
                    tree_parts.append(label)
                    tree_parts.append(connection)
            level_node_map[level] = nodes
            t0 += 0.5

        time_start = t0
        extend = 1
        ext_connections = []
        ext_labels = []
        morph_connections2 = []
        leaves = []
        for node in level_node_map[max_level]:
            t0 = extension_for_node2(ext_connections, ext_labels, node, max_level, level, extend, level_node_map,
                                     polar, q0, t0, 0.1, 0, res, tree_parts, morph_connections2, leaves)
        # time_end = t0
        # duration = time_end - time_start
        # ibpy.camera_zoom(lens=50,begin_time=time_start)
        # t0 = camera_empty.change_follow_influence(empty_circle, initial=0, final=1, begin_time=time_start - 0.5,
        #                                           transition_time=0.5)
        # camera_empty.follow(empty_circle, initial_value=0, final_value=-1, begin_time=time_start,
        #                     transition_time=duration,new_constraint=False)
        # ibpy.set_linear_action(camera_empty, 'offset_factor')
        # t0=ibpy.camera_zoom(lens=19,begin_time=time_end-1)
        # camera_empty.change_follow_influence(empty_circle,initial=1,final=0,begin_time=t0)

        t0= camera_empty.move(direction=[5,0,0],begin_time=t0)

        full_tree = BObject(children=tree_parts, rotation_euler=[0, 0, np.pi])
        full_tree.appear(transition_time=0)

        origin = Vector([11,0,0])
        coords = CoordinateSystem(dim=2, lengths=[8, 8], domains=[[-1, 1], [-1., 1.]], radii=[0.03, 0.03],
                                  all_n_tics=[4, 4],
                                  labels=[r"\mathbb{R}", r"i\mathbb{R}"],
                                  all_tic_labels=[np.arange(-1, 1.1, 0.5), np.arange(-1, 1.1, 0.5)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', ''],
                                  location_of_origin=origin,
                                  label_positions=['center', 'center'],
                                  include_zeros=[False, False],
                                  label_closenesses=[0.5, 1],
                                  name='CoordinateSystem',
                                  rotation_euler=[-np.pi/2,0,0])

        t0 = coords.appear(begin_time=t0, transition_time=2)
        coords.axes[1].axis_label.move(direction=[0, 0, -0.5], transition_time=0)
        coords.axes[0].axis_label.move(direction=[0, 0, -0.5], transition_time=0)

        # convert tree to curve
        old_h = -1
        old_i = -1
        old_j = -1
        old_k = -1
        dt = 0.1
        k_count = 0
        level_1 = ['a', 'b', 'A', 'B']
        level_2 = ['B', 'a', 'b', 'a', 'b', 'A', 'b', 'A', 'B', 'A', 'B', 'a']
        level_3 = ['A','B','a','B','a','b','a','b','A','B','a','b',
                   'a','b','A','b','A','B','a','b','A','b','A','B',
                   'A','B','a','b','A','B','A','B','a','B','a','b']
        fs = [4,3,2,4,1,2,2,4,1,2,3,1,1,2,3,1,4,3,
              2,4,1,2,3,1,1,2,3,1,4,3,3,1,4,3,2,4,
              1,2,3,1,4,3,3,1,4,3,2,4,4,3,2,4,1,2,
              3,1,4,3,2,4,4,3,2,4,1,2,2,4,1,2,3,1]

        # swap two entries
        # list = morph_connections2
        # list[0], list[2] = list[2], list[0]
        # list[1], list[3] = list[3], list[1]

        for h in range(4):
            if old_h >-1:
                morph_connections[old_h].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                                         transition_time=dt)
                if old_i > -1:
                    morph_connections[old_i].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                                         transition_time=dt)
                    old_i=-1
                    if old_j > -1:
                        morph_connections[old_j].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                             begin_time=t0, transition_time=dt)
                        old_j = -1
                        if old_k > -1:
                            morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                                  begin_time=t0, transition_time=dt)
                            old_k = -1
                t0 += dt
            t0 = morph_connections[h].highlight(color='example',emission=0.5,begin_time=t0,transition_time=dt)
            old_h = h
            for i in range(3):
                new_i = 4+3*h+i
                if old_i > -1:
                    morph_connections[old_i].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                                         transition_time=dt)
                    if old_j > -1:
                        morph_connections[old_j].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                             begin_time=t0, transition_time=dt)
                        old_j = -1
                        if old_k > -1:
                            morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                                  begin_time=t0, transition_time=dt)
                            old_k = -1
                    t0 += dt
                t0 = morph_connections[new_i].highlight(color='example', emission=0.5, begin_time=t0, transition_time=dt)
                old_i = new_i
                for j in range(3):
                    new_j = 16+9*h + 3 * i + j
                    if old_j > -1:
                        morph_connections[old_j].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                             begin_time=t0, transition_time=dt)
                        if old_k > -1:
                            morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                                  begin_time=t0, transition_time=dt)
                            old_k = -1
                        t0 += dt
                    t0 = morph_connections[new_j].highlight(color='example', emission=0.5, begin_time=t0,
                                                            transition_time=dt)
                    old_j = new_j
                    for k in range(0, 2):
                        if old_k > -1:
                            t0 = morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0,
                                                                       begin_time=t0, transition_time=dt)
                        t0 = morph_connections2[k_count].highlight(color='example', emission=0.5, begin_time=t0,
                                                                   transition_time=dt)
                        old_k = k_count
                        k_count += 1

                        t0 = create_next_point2(coords, level_1[h], level_2[3 * h + i],level_3[9*h+3*i+j], fs[2 * (9*h+3 * i + j) + k], t0=t0)

        morph_connections[old_h].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                             transition_time=dt)
        morph_connections[old_i].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                             transition_time=dt)
        morph_connections[old_j].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                             transition_time=dt)
        t0 = morph_connections2[old_k].unhighlight(color='drawing', from_emission=0.5, to_emission=0, begin_time=t0,
                                                   transition_time=dt)

        eps = 0.001
        average = 100
        scale = 2
        curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)

        fractal = Function([fx], None, domain=[0, 1], mode='PARAMETRIC', name='a_full',
                           location=[7, 0, 0],
                           rotation_euler=[np.pi, np.pi, np.pi / 2],
                           color='drawing', extrude=0.00, thickness=0.02, metallic=1, roughness=0,
                           num_points=resolution, emission=1,scale=4)

        t0=fractal.grow(begin_time=t0,transition_time=4)

        print("finished at ", t0)

    def apollonian(self):
        cues = self.sub_scenes['apollonian']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[0, -6, 0])

        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=2, lengths=[4, 2], domains=[[-2, 2], [-2., 1.]], radii=[0.015, 0.015],
                                  all_n_tics=[4, 3],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 2.1, 1), np.arange(-2, 1.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='small',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=3)

        t0 = 3.5

        model = ApollonianModel()
        resolutions = [
            [200, 1000], [20, 100], [20, 100], [20, 100]
        ]
        colors = model.colors
        basic_discs = []
        for circ, res, color in zip(model.get_circles(), resolutions, colors):
            disc = Disc2(r=circ.r, center=[np.real(circ.c), np.imag(circ.c), 0],
                         rotation_euler=[np.pi / 2, 0, 0], resolution=res, color=color)
            basic_discs.append(disc)
            coords.add_object(disc)
            disc.appear(begin_time=t0)
            t0 += 1.5

        for g in range(4):
            circle = model.get_circles()[g]
            gen = model.get_generators()[g]
            for i in range(6):
                circle = moebius_on_circle(gen, circle)
                disc = Disc2(r=circle.r, center=[np.real(circle.c), np.imag(circle.c), 0.02 * (i + 1)],
                             rotation_euler=[np.pi / 2, 0, 0], resolution=res, solid=0.02,
                             color=colors[g], brighter=i + 1)
                disc.appear(begin_time=t0)
                t0 += 1
            t0 += 1

        print("finished at ", t0)

    def apollonian_full(self):
        self.disc_counter = 0
        cues = self.sub_scenes['apollonian_full']
        t0 = 0.5  # cues['start']
        ibpy.set_camera_location(location=[0, 0, 0])
        ibpy.set_camera_rotation(rotation=[np.pi / 6, 0, 0])
        camera_empty = EmptyCube(location=[0, 0, -0.5])
        camera_radius = Vector([0.02954, -3.1279, 0])
        camera_circle = BezierCircle(radius=np.sqrt(camera_radius.dot(camera_radius)), location=[0, 0, 4.9261])

        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_follow(camera_circle)

        for level in range(1, 8):
            dfs = DepthFirstSearchByLevel(ApollonianModel, max_level=level)
            dfs.generate_tree()
            leaves = dfs.get_leaves()
            duration = 5
            dt = 0.8 * duration / len(leaves)
            for leave in leaves:
                disc = self.disc_from_circle(leave.circle, begin_time=t0,
                                             color=leave.color,
                                             offset=level, brighter=level - 1)
                disc.appear(begin_time=t0, transition_time=dt)
                t0 += dt * 1.25
            t0 += 1

        # bring the spider in action

        eps = 0.001
        average = 100
        scale = 2
        curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)
        fx2 = lambda x: function_from_complex_list(curve, x, scale=scale, average=average)

        fractal = Function([fx], None, domain=[0, 1], mode='PARAMETRIC', name='a_full',
                           location=[1, 0, 0.78],
                           rotation_euler=[0, 0, np.pi / 2],
                           color='important', extrude=0.01, thickness=0.01, metallic=1, roughness=0,
                           num_points=resolution, emission=1)

        average_fractal = Function([fx2], None, domain=[0, 1], mode='PARAMETRIC',
                                   location=[1, 0, 0.76],
                                   rotation_euler=[0, 0, np.pi / 2],
                                   name='average', color='example', extrude=0, thickness=0,
                                   numpoints=resolution / average)

        t0 += 0.5
        duration = 30
        camera_circle.move(direction=[0, 0, -2], begin_time=t0, transition_time=duration / 2)
        camera_circle.move(direction=[0, 0, 2], begin_time=t0 + duration / 2, transition_time=duration / 2)
        ibpy.camera_follow(camera_circle, begin_time=t0, transition_time=duration, initial_value=0, final_value=1)
        fractal.grow(begin_time=t0, transition_time=duration)

        spider = AnimBObject('Spider', object_name='widow', armature='Armature', color='joker', scale=1,
                             emission=0.75)
        spider.armature.ref_obj.scale = [0.02, 0.02, 0.02]
        spider.appear(begin_time=t0, transition_time=2)

        spider.armature.follow(average_fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')

        t0 += duration

        print("final ", t0)

    def commutators(self):
        cues = self.sub_scenes['commutators']
        t0 = 0.5  # cues['start']

        model = ApollonianModel()
        for l in range(1, 5):

            dfs = DepthFirstSearchWithFixedPoints(ApollonianModel, max_level=l)
            dfs.generate_tree()
            points = dfs.collect_points()

            duration = 5 * l
            last = points[-1]
            dt = duration / len(points)
            for p in points:
                sphere = Sphere(0.01, location=z2vec(p))
                sphere.grow(begin_time=t0, transition_time=dt)
                line = Cylinder.from_start_to_end(start=z2vec(last), end=z2vec(p), thickness=0.1 / l)
                line.grow(begin_time=t0, transition_time=dt)
                t0 += dt
                last = p
        t0 += 1

        print("finished at  ", t0)

    def commutators_adaptive(self):
        cues = self.sub_scenes['commutators_adaptive']
        t0 = 0.5  # cues['start']

        epsilon = [0.25, 0.125, 0.01]

        data_set = []
        for l in range(0, len(epsilon)):
            dfs = DepthFirstSearchWithFixedPoints(ApollonianModel, max_sep=epsilon[l])
            dfs.generate_tree()
            data_set.append(dfs.points)

        curve = Curve.from_data(data_set, thickness=0.1, name='Curve', res=1000)

        for l, data in enumerate(data_set):
            duration = 5 * (l + 1)
            dt = duration / len(data)
            remove_spheres = []

            if l < 2:
                for point in data:
                    sphere = Sphere(0.02, location=z2vec(point), res=2)
                    sphere.grow(begin_time=t0, transition_time=dt)
                    remove_spheres.append(sphere)
                    t0 += dt

            if l == 0:
                curve.grow(begin_time=t0, transition_time=duration)
                t0 += duration
            else:
                curve.next(begin_time=t0)

            [s.disappear(begin_time=t0) for s in remove_spheres]

        print("finished at  ", t0)

    def commutators_adaptive_ii(self):
        cues = self.sub_scenes['commutators_adaptive']
        t0 = 0.5  # cues['start']

        epsilon = [0.5, 0.25, 0.125, 0.0625]

        data_set = []
        for l in range(0, len(epsilon)):
            dfs = DepthFirstSearchWithFixedPoints(ApollonianModel, max_sep=epsilon[l])
            dfs.generate_tree()
            data_set.append(dfs.points)

        curve = DataCurveRefined(data_set, name="DataCurve", solid=0.001, smooth=0, scale=3,
                                 rotation_euler=[np.pi / 2, 0, 0])
        curve.appear(begin_time=t0)
        t0 += 2

        for l in range(len(epsilon) - 1):
            curve.next_shape(begin_time=t0)
            t0 += 2

        epsilon = [0.0625, 0.01, 0.001]
        data_set = []
        for l in range(0, len(epsilon)):
            dfs = DepthFirstSearchWithFixedPoints(ApollonianModel, max_sep=epsilon[l])
            dfs.generate_tree()
            data_set.append(dfs.points)

        curve2 = DataCurveRefined(data_set, name="DataCurve2", solid=0.001, smooth=0, scale=3,
                                  rotation_euler=[np.pi / 2, 0, 0])

        for l in range(len(epsilon)):
            if l == 0:
                curve2.appear(begin_time=t0)
                curve.disappear(begin_time=t0 + 1, transition_time=0)
            else:
                curve2.next_shape(begin_time=t0)
            t0 += 2

        print("finished at  ", t0)

    def geogebra(self):
        cues = self.sub_scenes['geogebra']
        t0 = 0.5

        ibpy.set_camera_lens(lens=50)

        display = Display(location=[0, 0, 0], flat=True, number_of_lines=19)
        display.appear(begin_time=t0)

        title = SimpleTexBObject(r"\text{Descart's theorem}", aligned='center', color='example')

        first_obervation = SimpleTexBObject(r"(2+2+8+0)^2=2\cdot\left(2^2+2^2+8^2+0^2\right)")
        display.write_text_in(first_obervation, line=2, begin_time=t0, scale=0.5, indent=1)
        t0 += 1.5

        display.write_title(title, begin_time=t0, transition_time=2)

        circle1 = BezierCircle(1, location=[-1, 0, 0],
                               rotation_euler=[np.pi / 2, 0, 0],
                               bevel=0.01, extrude=0.2)
        circle1.grow(begin_time=t0)
        circle2 = BezierCircle(1, location=[1, 0, 0],
                               rotation_euler=[np.pi / 2, 0, 0],
                               bevel=0.01, extrude=0.2)
        circle2.grow(begin_time=t0)
        line = Cylinder.from_start_to_end(start=[-3, -0.1, 1], end=[3, -0.1, 1], cyl_radii=[0.01, 0.1, 1])
        line.grow(begin_time=t0)

        circle3 = BezierCircle(radius=0.25, location=[0, 0, 0.75], rotation_euler=[np.pi / 2, 0, 0],
                               color='example', bevel=0.01, extrude=0.2)

        t0 += 1.5
        circle3.grow(begin_time=t0)
        t0 += 1.5

        rot_group = BObject(children=[circle1, circle2, line, circle3], location=[1, 0, -4.4],
                            rotation_euler=[0, -np.pi / 6, 0])
        rot_group.appear(begin_time=0)

        colors = flatten([['text'], ['example'], ['text'] * 12, ['example'], ['text']])
        colors1b = flatten([['text'], ['example'], ['text'] * 13, ['example'], ['text']])
        equation1 = SimpleTexBObject(r"(x+2+2+8)^2=2\cdot\left(x^2+2^2+2^2+8^2\right)", color=colors)
        display.write_text_in(equation1, line=5, begin_time=t0, scale=0.5, indent=1)
        t0 += 1.5
        colors2 = flatten([['example'], ['text'] * 5, ['example'], ['text']])
        solution1 = SimpleTexBObject(r"x=24\Longrightarrow r={1\over 24}", color=colors2)
        display.write_text_in(solution1, line=7, indent=3, begin_time=t0)
        t0 += 1.5
        equation2 = SimpleTexBObject(r"(x+2+8+18)^2=2\cdot\left(x^2+2^2+8^2+18^2\right)", color=colors1b)
        display.write_text_in(equation2, line=9, begin_time=t0, scale=0.5, indent=1)
        t0 += 1.5
        solution2 = SimpleTexBObject(r"x=56\Longrightarrow r={1\over 56}", color=colors2)
        display.write_text_in(solution2, line=11, indent=3, begin_time=t0)

    def pattern(self):
        cues = self.sub_scenes['pattern']
        t0 = 0.5  # cues['start']

        display = Display(location=[0, 0, 0], scales=[10, 3], number_of_lines=5, flat=True)
        display.appear(begin_time=t0)
        t0 += 1.5

        table = r"x && {1\over 2} && {1\over 3} && {1\over 4} && {1\over 5} && {1\over 6} && {1\over 7} && {p\over " \
                r"q}\\ && \\ \hline \rule{0em}{5ex} r && {1\over 8} && {1\over 18} && {1\over 32} && {1\over 50} && {" \
                r"1\over 72} && {1\over 98} && {1\over 2q^2} "
        colors = flatten([['example'] * 2, ['text'] * 20, ['example'], ['text'] * 21, ['example']])

        tabular = SimpleTexBObject(table, color=colors)

        display.add_text_in(tabular, line=1, scale=0.5)
        tabular.write(letter_set={0, 1, 22}, begin_time=t0, transition_time=0.5)
        t0 += 1
        tabular.write(letter_set={2, 3, 4}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        tabular.write(letter_set={5, 6, 7}, begin_time=t0, transition_time=0.5)
        t0 += 0.6

        tabular.write(letter_set={8, 9, 10}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        tabular.write(letter_set={11, 12, 13, 14}, begin_time=t0, transition_time=0.5)
        t0 += 0.6

        tabular.write(letter_set={15, 16, 17}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        tabular.write(letter_set={18, 19, 21, 20}, begin_time=t0, transition_time=0.5)
        t0 += 0.6

        tabular.write(letter_set={23, 24, 25}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        tabular.write(letter_set={26, 27, 29, 28}, begin_time=t0, transition_time=0.5)
        t0 += 0.6

        tabular.write(letter_set={30, 31, 32}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        tabular.write(letter_set={33, 34, 36, 35}, begin_time=t0, transition_time=0.5)
        t0 += 0.6

        tabular.write(letter_set={37, 38, 39}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        tabular.write(letter_set={40, 41, 43, 42}, begin_time=t0, transition_time=0.5)
        t0 += 0.6

        tabular.write(letter_set={49, 48, 50}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        tabular.write(letter_set={45, 46, 44, 47, 51}, begin_time=t0, transition_time=0.5)
        t0 += 0.6
        print("finished at ", t0)

    def families(self):
        cues = self.sub_scenes['families']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[0, 0, 10])
        ibpy.set_camera_rotation(rotation=[0, 0, 0])

        colors = ['joker', 'example', 'text', 'drawing', 'custom1']
        for f in range(1, 6):
            points = read_data("family_" + str(f) + "_centers.txt")
            curvatures = read_data("family_" + str(f) + "_curvatures.txt")

            pairs = [[point, curvature] for point, curvature in zip(points, curvatures)]
            pairs = sorted(pairs, key=comp)

            duration = 5
            dt = duration / len(pairs)
            bright_max = 10
            bright_min = -5

            min_curvature = pairs[0][1]
            max_curvature = pairs[-1][1]

            offset = Vector([(-1) ** f, 7 - 2.3 * f, 0])

            for pair in pairs:
                scale = min_curvature * 1.5
                brighter = int(lin_map(pair[1], min_curvature, max_curvature, bright_min, bright_max))
                circle = BezierCircle(location=z2vec(pair[0]) * scale + offset, bevel=0.0025 / min_curvature,
                                      radius=1 / pair[1], color=colors[f - 1], resolution=max_curvature * 4 / pair[1],
                                      brighter=brighter, scale=[scale, scale, scale])
                circle.grow(begin_time=t0, transition_time=dt)
                t0 += dt

            t0 += 1

        print("finished at ", t0)

    def ancient(self):
        cues = self.sub_scenes['ancient']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[0, -13, 4.5])
        ibpy.set_camera_rotation(rotation=[np.pi / 180 * 70, 0, 0])

        plane = Plane(u=[-5, 5], v=[-5, 5], color='sand',
                      location=[0, 0, -0.01])  # Watch out, the location is vertex location that gets rescaled
        plane.appear(begin_time=t0)
        ibpy.add_modifier(plane, type='COLLISION')  # make sure that the rope cannot dive underneath the plane
        t0 += 1.5

        apollonius = PersonWithCape(location=[-3.5, -3.5, 0], rotation_euler=[0, 0, -np.pi / 4],
                                    colors=['gray_8', 'important'],
                                    simulation_start=0, simulation_duration=60, name='Greek1')
        apollonius.appear(begin_time=t0)
        t0 += 1

        f = 1
        points = read_data("family_" + str(f) + "_centers.txt")
        curvatures = read_data("family_" + str(f) + "_curvatures.txt")

        pairs = [[point, curvature] for point, curvature in zip(points, curvatures)]
        pairs = sorted(pairs, key=comp)

        for i in range(0, 9):
            c = z2vec(pairs[i][0] * 3)
            r = 3 / pairs[i][1]
            t0 = construct_circle(center=c, radius=r, time=t0, duration=5, index=i)

        print("finished at ", t0)

    def curves1(self):
        cues = self.sub_scenes['curves1']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[0, -15.5, 0])

        # coordinate system
        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=2, lengths=[4, 5], domains=[[-2, 2], [-2., 3.]], radii=[0.015, 0.015],
                                  all_n_tics=[4, 5],
                                  labels=[r"x", r"y"],
                                  all_tic_labels=[np.arange(-2, 2.1, 1), np.arange(-2, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', ''],
                                  axis_label_size='small',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  label_positions=['center', 'center'],
                                  include_zeros=[False, False],
                                  label_closenesses=[0.5, 1],
                                  name='CoordinateSystem')
        coords.appear(begin_time=t0, transition_time=3)
        coords.axes[1].axis_label.move(direction=[0, 0, -0.5], transition_time=0)
        coords.axes[0].axis_label.move(direction=[0, 0, -0.5], transition_time=0)
        # display

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0)
        t0 += 1.5

        title = SimpleTexBObject(r"\text{Functions}", aligned="center", color="example")
        display.write_title(title, begin_time=t0)
        t0 += 1.5

        # spider with function thread
        duration = 5
        y = Function([lambda x: x ** 2 - 1], coords, domain=[-2, 2], color='drawing', mode='2D')
        y.grow(begin_time=t0, transition_time=duration)

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, -np.pi / 2, 0])
        spider.armature.ref_obj.scale = [0.2, 0.2, 0.2]
        spider.appear(begin_time=t0, transition_time=0.1)

        spider.armature.follow(y, initial_value=0, final_value=1, begin_time=t0,
                               transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')

        spider.disappear(begin_time=t0 + duration, transition_time=0.1)
        function = SimpleTexBObject(r"y(x)=x^2-1", aligned='center')
        display.write_text_in(function, line=7, begin_time=t0, transition_time=1, indent=5)
        t0 += 3.5

        print("finished at ", t0)

    def curves2(self):

        cues = self.sub_scenes['curves1']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[0, 0, 0])

        # coordinate system
        origin = Vector([0, 0, 0])
        coords = CoordinateSystem(dim=3, lengths=[4, 5, 5], domains=[[-2, 2], [-2., 3.], [0, 5]],
                                  radii=[0.015, 0.015, 0.015],
                                  all_n_tics=[4, 5, 5],
                                  labels=[r"x", r"y", r"z"],
                                  all_tic_labels=[np.arange(-2, 2.1, 1), np.arange(-2, 3.1, 1), np.arange(0, 5.1, 1)],
                                  colors=['text', 'text', 'text'], label_colors=['text', 'text', 'text'],
                                  label_digits=[0, 0, 0], label_units=['', '', ''],
                                  axis_label_size='small',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  label_positions=['center', 'center', 'center'],
                                  include_zeros=[False, False, False],
                                  label_closenesses=[0.5, 1, 1],
                                  name='CoordinateSystem'
                                  )
        coords.appear(begin_time=t0, transition_time=3)
        coords.rotate(rotation_euler=[np.pi / 2, 0, 0], compensate=False, transition_time=0)
        coords.axes[1].axis_label.move(direction=[0, 0, -0.5], transition_time=0)
        coords.axes[0].axis_label.move(direction=[0, 0, -0.5], transition_time=0)
        # display

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0)
        t0 += 1.5

        title = SimpleTexBObject(r"\text{Parametric Functions}", aligned="center", color="example")
        display.write_title(title, begin_time=t0)
        t0 += 1.5

        # spider with function thread
        duration = 5
        y = Function([lambda t: Vector([2 * np.cos(t), 2 * np.sin(t), t / np.pi])], coords, domain=[0, 6 * np.pi],
                     color='drawing', mode='PARAMETRIC')
        y.grow(begin_time=t0, transition_time=duration, end_factor=0.3333)

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, -np.pi / 4, 0])
        spider.armature.ref_obj.scale = [0.2, 0.2, 0.2]
        spider.appear(begin_time=t0, transition_time=0.1)

        spider.armature.follow(y, initial_value=0, final_value=0.333, begin_time=t0,
                               transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')

        x_function = SimpleTexBObject(r"x(t)=2\cos(t)")
        display.write_text_in(x_function, line=4, begin_time=t0, transition_time=1, indent=7)
        y_function = SimpleTexBObject(r"y(t)=2\sin(t)")
        display.write_text_in(y_function, line=5, begin_time=t0 + 1, transition_time=1, indent=7)

        t0 += duration + 0.5

        camera_circle = BezierCircle(radius=15.5, location=[0, 0, 0], rotation_euler=[0, np.pi / 2, 0])
        camera_empty = EmptyCube()
        ibpy.set_camera_follow(camera_circle)
        ibpy.camera_follow(camera_circle, initial_value=0, final_value=-0.125, begin_time=t0, transition_time=duration)
        ibpy.set_camera_view_to(camera_empty)
        z_function = SimpleTexBObject(r"z(t)={t\over \pi}")
        display.write_text_in(z_function, line=7, begin_time=t0, transition_time=1, indent=7)
        texts = [z_function, x_function, y_function, title]
        for i, text in enumerate(texts):
            if i == 0:
                shift = 0.055
            else:
                shift = 0.035
            text.rotate(rotation_euler=[np.pi / 8, 0, 0], begin_time=t0 + 0.1, transition_time=duration)
            text.move(direction=[0, 0, shift], begin_time=t0 + 0.1, transition_time=duration)

        camera_empty.move(direction=[0, -0.5, 0], begin_time=t0, transition_time=duration)
        spider.armature.follow(y, initial_value=0.333, final_value=1, begin_time=t0,
                               transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y', new_constraint=False)
        y.grow(start_factor=0.333, end_factor=1, begin_time=t0, transition_time=duration)

        t0 += duration
        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5

        print("finished at ", t0)

    def curves3(self):
        cues = self.sub_scenes['curves3']
        t0 = 0  # cues['start']

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0)
        t0 += 1.5

        title = SimpleTexBObject(r"\text{Turtle graphics}", aligned="center", color="example")
        display.write_title(title, begin_time=t0)
        t0 += 1.5

        colors1 = flatten([['text'] * 3, ['joker'] * 7, ['drawing'], ['important'] * 6, ['drawing'], ['text']])
        colors2 = flatten([['drawing'] * 5, ['important'] * 6, ['drawing']])
        colors3 = flatten([['joker'] * 7, ['drawing'], ['important'] * 3, ['drawing']])

        instructions = [
            SimpleTexBObject(r"\text{def pattern}(length):", color=colors1),
            SimpleTexBObject(r"\text{walk}(100)", color='drawing'),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(100)", color='drawing'),
            SimpleTexBObject(r"\text{turn}(-120)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(100)", color='drawing'),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(100)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{pattern(100)}", color=colors3)
        ]

        l = 2
        r3o2 = np.sqrt(3) / 2
        shift = -5

        functions = [
            Curve([lambda x: Vector([l * x + shift, 0, 0])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve1', thickness=0.5),
            Curve([lambda x: Vector([l + 0.5 * l * x + shift, 0, r3o2 * l * x])], domain=[0, 1], color='drawing',
                  num_points=3, name='curve2', thickness=0.5),
            Curve([lambda x: Vector([1.5 * l + 0.5 * l * x + shift, 0, r3o2 * l - r3o2 * l * x])], domain=[0, 1],
                  color='drawing', num_points=3, name='curve3', thickness=0.5),
            Curve([lambda x: Vector([2 * l + l * x + shift, 0, 0])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve4', thickness=0.5),
        ]

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, -np.pi / 2, np.pi]
                             )
        spider.armature.ref_obj.scale = [0.2, 0.2, 0.2]
        spider.appear(begin_time=t0, transition_time=0.1)

        for i, function in enumerate(functions):
            ibpy.set_follow(spider.armature, function)
            if i == 0:
                ibpy.set_follow_influence(spider.armature, function, value=1)
            else:
                ibpy.set_follow_influence(spider.armature, function, value=0)

        for i in {1, 2, 3, 4, 5, 6, 7}:
            instruction = instructions[i]
            display.write_text_in(instruction, line=i, begin_time=t0, indent=7)
            if i in {1, 3, 5, 7}:
                f = functions[int((i - 1) / 2)]
                f.grow(begin_time=t0)
                spider.armature.follow(f, begin_time=t0, new_constraint=False)
            elif i in {2, 4, 6}:
                f_old = functions[int((i - 2) / 2)]
                f_new = functions[int(i / 2)]
                # make instant transition from one curve to the next
                ibpy.change_follow_influence(spider.armature, f_old, initial=1, final=0, begin_time=t0 + 0.5,
                                             transition_time=0)
                ibpy.change_follow_influence(spider.armature, f_new, initial=0, final=1, begin_time=t0 + 0.5,
                                             transition_time=0)
            t0 += 1
        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5
        display.write_text_in(instructions[0], line=0, begin_time=t0, indent=6)
        t0 += 1.5
        instructions[1].replace(instructions[8], src_letter_range=[5, 9], img_letter_range=[5, 12], begin_time=t0,
                                morphing=True)
        instructions[3].replace(instructions[8], src_letter_range=[5, 9], img_letter_range=[5, 12], begin_time=t0,
                                morphing=True)
        instructions[5].replace(instructions[8], src_letter_range=[5, 9], img_letter_range=[5, 12], begin_time=t0,
                                morphing=True)
        instructions[7].replace(instructions[8], src_letter_range=[5, 9], img_letter_range=[5, 12], begin_time=t0,
                                morphing=True)
        t0 += 1.5

        print("finished at ", t0)

    def curves3b(self):
        cues = self.sub_scenes['curves3b']
        t0 = 0  # cues['start']

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0, transition_time=0)

        title = SimpleTexBObject(r"\text{Turtle graphics}", aligned="center", color="example")
        display.write_title(title, begin_time=t0, transition_time=0)

        colors1 = flatten([['text'] * 3, ['joker'] * 7, ['drawing'], ['important'] * 6, ['drawing'], ['text']])
        colors2 = flatten([['drawing'] * 5, ['important'] * 6, ['drawing']])
        colors3 = flatten([['joker'] * 7, ['drawing'], ['important'] * 3, ['drawing']])

        instructions = [
            SimpleTexBObject(r"\text{def pattern}(length):", color=colors1),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(-120)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{pattern(100)}", color=colors3)
        ]

        l = 2
        r3o2 = np.sqrt(3) / 2
        shift = -5
        shift2 = 2

        functions = [
            Curve([lambda x: Vector([l * x + shift, 0, shift2])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve1', thickness=0.5),
            Curve([lambda x: Vector([l + 0.5 * l * x + shift, 0, r3o2 * l * x + shift2])], domain=[0, 1],
                  color='drawing',
                  num_points=3, name='curve2', thickness=0.5),
            Curve([lambda x: Vector([1.5 * l + 0.5 * l * x + shift, 0, r3o2 * l - r3o2 * l * x + shift2])],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='curve3', thickness=0.5),
            Curve([lambda x: Vector([2 * l + l * x + shift, 0, shift2])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve4', thickness=0.5),
        ]

        for i, instruction in enumerate(instructions):
            if i == 0:
                display.write_text_in(instruction, line=i, begin_time=t0, indent=6, transition_time=0)
            elif i < 8:
                display.write_text_in(instruction, line=i, begin_time=t0, indent=7, transition_time=0)
            else:
                display.write_text_in(instruction, line=0, begin_time=t0, indent=0.5)

        t0 += 1.5

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, -np.pi / 2, np.pi]
                             )
        spider.armature.ref_obj.scale = [0.2, 0.2, 0.2]
        spider.appear(begin_time=t0, transition_time=0.1)

        for i, function in enumerate(functions):
            ibpy.set_follow(spider.armature, function)
            if i == 0:
                ibpy.set_follow_influence(spider.armature, function, value=1)
            else:
                ibpy.set_follow_influence(spider.armature, function, value=0)

        f_old = None
        for f in functions:
            if f_old:
                ibpy.change_follow_influence(spider.armature, f_old, initial=1, final=0, begin_time=t0,
                                             transition_time=0)
                ibpy.change_follow_influence(spider.armature, f, initial=0, final=1, begin_time=t0,
                                             transition_time=0)
            f.grow(begin_time=t0)
            spider.armature.follow(f, begin_time=t0, new_constraint=False)
            f_old = f
            t0 += 1

        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5

        print("finished at ", t0)

    def curves3c(self):
        cues = self.sub_scenes['curves3c']
        t0 = 0  # cues['start']

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0, transition_time=0)

        title = SimpleTexBObject(r"\text{Turtle graphics}", aligned="center", color="example")
        display.write_title(title, begin_time=t0, transition_time=0)

        colors1 = flatten([['text'] * 3, ['joker'] * 7, ['drawing'], ['important'] * 6, ['drawing'], ['text']])
        colors2 = flatten([['drawing'] * 5, ['important'] * 6, ['drawing']])
        colors3 = flatten([['joker'] * 7, ['drawing'], ['important'] * 3, ['drawing']])

        instructions = [
            SimpleTexBObject(r"\text{def pattern}(length):", color=colors1),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(-120)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{pattern(100)}", color=colors3)
        ]

        colors4 = flatten([['joker'] * 7, ['drawing'], ['important'] * 5, ['drawing']])

        instructions2 = [
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
            SimpleTexBObject(r"\text{turn}(-120)", color='drawing'),
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
        ]

        l = 2
        r3o2 = np.sqrt(3) / 2
        shift = -5
        shift2 = 2

        functions = [
            Curve([lambda x: Vector([l * x + shift, 0, shift2])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve1', thickness=0.5),
            Curve([lambda x: Vector([l + 0.5 * l * x + shift, 0, r3o2 * l * x + shift2])], domain=[0, 1],
                  color='drawing',
                  num_points=3, name='curve2', thickness=0.5),
            Curve([lambda x: Vector([1.5 * l + 0.5 * l * x + shift, 0, r3o2 * l - r3o2 * l * x + shift2])],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='curve3', thickness=0.5),
            Curve([lambda x: Vector([2 * l + l * x + shift, 0, shift2])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve4', thickness=0.5),
        ]

        for i, instruction in enumerate(instructions):
            if i == 0:
                display.write_text_in(instruction, line=i, begin_time=t0, indent=6, transition_time=0)
            elif i < 8:
                display.write_text_in(instruction, line=i, begin_time=t0, indent=7, transition_time=0)
            else:
                display.write_text_in(instruction, line=0, begin_time=t0, indent=0.5, transition_time=0)

        for f in functions:
            f.grow(begin_time=t0, transition_time=0)

        t0 += 0.5

        l3 = 2 / 3
        r3o2 = np.sqrt(3) / 2
        shift = -5
        shift2 = -0.5

        b0 = Vector([shift, 0, shift2])

        c0 = Vector([0, 0, shift2])
        c1 = Vector([l, 0, shift2])
        c2 = Vector([1.5 * l, 0, l * r3o2 + shift2])
        c3 = Vector([2 * l, 0, shift2])

        functions2 = [
            Curve([lambda x: lin_map(rot(0), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c0)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve1', thickness=0.5),
            Curve([lambda x: lin_map(rot(0),
                                     Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0, b0 + c0)],
                  domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve2', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c0)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve3', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c0)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve4', thickness=0.5),

            Curve([lambda x: lin_map(rot(60), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c1)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve5', thickness=0.5),
            Curve([lambda x: lin_map(rot(60), Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0,
                                     b0 + c1)], domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve6', thickness=0.5),
            Curve([lambda x: lin_map(rot(60), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c1)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve7', thickness=0.5),
            Curve([lambda x: lin_map(rot(60), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c1)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve8', thickness=0.5),

            Curve([lambda x: lin_map(rot(-60), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c2)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve9', thickness=0.5),
            Curve([lambda x: lin_map(rot(-60),
                                     Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0, b0 + c2)],
                  domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve10', thickness=0.5),
            Curve([lambda x: lin_map(rot(-60), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c2)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve11', thickness=0.5),
            Curve([lambda x: lin_map(rot(-60), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c2)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve12', thickness=0.5),

            Curve([lambda x: lin_map(rot(0), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c3)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve13', thickness=0.5),
            Curve([lambda x: lin_map(rot(0),
                                     Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0, b0 + c3)],
                  domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve14', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c3)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve15', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c3)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve16', thickness=0.5),
        ]

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, -np.pi / 2, np.pi]
                             )
        spider.armature.ref_obj.scale = [0.2, 0.2, 0.2]
        spider.appear(begin_time=t0, transition_time=0.1)

        for i, function in enumerate(functions2):
            ibpy.set_follow(spider.armature, function)
            if i == 0:
                ibpy.set_follow_influence(spider.armature, function, value=1)
            else:
                ibpy.set_follow_influence(spider.armature, function, value=0)

        f_old = None
        for i, f in enumerate(functions2):
            if i % 4 == 0:
                j = int(2 * i / 4)
                instruction = instructions2[j]
                display.write_text_in(instruction, line=0.89 * j + 2.5, begin_time=t0,
                                      indent=0.5, scale=0.5)
            if i % 4 == 3 and i < 12:
                j = int(2 * (i - 3) / 4) + 1
                instruction = instructions2[j]
                display.write_text_in(instruction, line=0.89 * j + 2.5, begin_time=t0 + 0.3, transition_time=0.015,
                                      scale=0.5, indent=0.5)
            if f_old:
                ibpy.change_follow_influence(spider.armature, f_old, initial=1, final=0, begin_time=t0,
                                             transition_time=0)
                ibpy.change_follow_influence(spider.armature, f, initial=0, final=1, begin_time=t0,
                                             transition_time=0)
            f.grow(begin_time=t0, transition_time=0.333)
            spider.armature.follow(f, begin_time=t0, new_constraint=False, transition_time=0.3333)
            f_old = f
            t0 += 0.3333

        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5

        print("finished at ", t0)

    def curves3d(self):
        cues = self.sub_scenes['curves3d']
        t0 = 0  # cues['start']

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0, transition_time=0)

        title = SimpleTexBObject(r"\text{Turtle graphics}", aligned="center", color="example")
        display.write_title(title, begin_time=t0, transition_time=0)

        colors1 = flatten([['text'] * 3, ['joker'] * 7, ['drawing'], ['important'] * 6, ['drawing'], ['text']])
        colors2 = flatten([['drawing'] * 5, ['important'] * 6, ['drawing']])
        colors3 = flatten([['joker'] * 7, ['drawing'], ['important'] * 3, ['drawing']])

        instructions = [
            SimpleTexBObject(r"\text{def pattern}(length):", color=colors1),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(-120)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{walk}(length)", color=colors2),
            SimpleTexBObject(r"\text{pattern(100)}", color=colors3)
        ]

        colors4 = flatten([['joker'] * 7, ['drawing'], ['important'] * 5, ['drawing']])

        instructions2 = [
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
            SimpleTexBObject(r"\text{turn}(-120)", color='drawing'),
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
            SimpleTexBObject(r"\text{turn}(60)", color='drawing'),
            SimpleTexBObject(r"\text{pattern}({100 \over 3})", color=colors4),
        ]

        l = 2
        r3o2 = np.sqrt(3) / 2
        shift = -5
        shift2 = 2

        functions = [
            Curve([lambda x: Vector([l * x + shift, 0, shift2])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve1', thickness=0.5),
            Curve([lambda x: Vector([l + 0.5 * l * x + shift, 0, r3o2 * l * x + shift2])], domain=[0, 1],
                  color='drawing',
                  num_points=3, name='curve2', thickness=0.5),
            Curve([lambda x: Vector([1.5 * l + 0.5 * l * x + shift, 0, r3o2 * l - r3o2 * l * x + shift2])],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='curve3', thickness=0.5),
            Curve([lambda x: Vector([2 * l + l * x + shift, 0, shift2])], domain=[0, 1], color='drawing', num_points=3,
                  name='curve4', thickness=0.5),
        ]

        for i, instruction in enumerate(instructions):
            if i == 0:
                display.write_text_in(instruction, line=i, begin_time=t0, indent=6, transition_time=0)
            elif i < 8:
                display.write_text_in(instruction, line=i, begin_time=t0, indent=7, transition_time=0)
            else:
                display.write_text_in(instruction, line=0, begin_time=t0, indent=0.5, transition_time=0)

        for f in functions:
            f.grow(begin_time=t0, transition_time=0)

        l3 = 2 / 3
        r3o2 = np.sqrt(3) / 2
        shift = -5
        shift2 = -0.5

        b0 = Vector([shift, 0, shift2])

        c0 = Vector([0, 0, shift2])
        c1 = Vector([l, 0, shift2])
        c2 = Vector([1.5 * l, 0, l * r3o2 + shift2])
        c3 = Vector([2 * l, 0, shift2])

        functions2 = [
            Curve([lambda x: lin_map(rot(0), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c0)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve1', thickness=0.5),
            Curve([lambda x: lin_map(rot(0),
                                     Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0, b0 + c0)],
                  domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve2', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c0)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve3', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c0)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve4', thickness=0.5),

            Curve([lambda x: lin_map(rot(60), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c1)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve5', thickness=0.5),
            Curve([lambda x: lin_map(rot(60), Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0,
                                     b0 + c1)], domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve6', thickness=0.5),
            Curve([lambda x: lin_map(rot(60), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c1)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve7', thickness=0.5),
            Curve([lambda x: lin_map(rot(60), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c1)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve8', thickness=0.5),

            Curve([lambda x: lin_map(rot(-60), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c2)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve9', thickness=0.5),
            Curve([lambda x: lin_map(rot(-60),
                                     Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0, b0 + c2)],
                  domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve10', thickness=0.5),
            Curve([lambda x: lin_map(rot(-60), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c2)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve11', thickness=0.5),
            Curve([lambda x: lin_map(rot(-60), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c2)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve12', thickness=0.5),

            Curve([lambda x: lin_map(rot(0), Vector([l3 * x + shift, 0, shift2]) - b0, b0 + c3)], domain=[0, 1],
                  color='drawing', num_points=3,
                  name='sub_curve13', thickness=0.5),
            Curve([lambda x: lin_map(rot(0),
                                     Vector([l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 * x + shift2]) - b0, b0 + c3)],
                  domain=[0, 1],
                  color='drawing',
                  num_points=3, name='sub_curve14', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector(
                [1.5 * l3 + 0.5 * l3 * x + shift, 0, r3o2 * l3 - r3o2 * l3 * x + shift2]) - b0, b0 + c3)],
                  domain=[0, 1],
                  color='drawing', num_points=3, name='sub_curve15', thickness=0.5),
            Curve([lambda x: lin_map(rot(0), Vector([2 * l3 + l3 * x + shift, 0, shift2]) - b0, b0 + c3)],
                  domain=[0, 1], color='drawing', num_points=3,
                  name='sub_curve16', thickness=0.5),
        ]

        for i, f in enumerate(functions2):
            if i % 4 == 0:
                j = int(2 * i / 4)
                instruction = instructions2[j]
                display.write_text_in(instruction, line=0.89 * j + 2.25, begin_time=t0, transition_time=0,
                                      indent=0.5, scale=0.5)
            if i % 4 == 3 and i < 12:
                j = int(2 * (i - 3) / 4) + 1
                instruction = instructions2[j]
                display.write_text_in(instruction, line=0.89 * j + 2.25, begin_time=t0, transition_time=0,
                                      scale=0.5, indent=0.5)

            f.grow(begin_time=t0, transition_time=0)

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, -np.pi, np.pi]
                             )
        spider.armature.ref_obj.scale = [0.2, 0.2, 0.2]
        spider.appear(begin_time=t0, transition_time=0.1)

        # full Koch curve
        curve = Koch(iteration=5).points
        fx = lambda x: function_from_complex_list(curve, x, scale=12)

        duration = 15

        shift = -5
        shift2 = -4

        fractal = Function([fx], None, domain=[0, 0.9995], mode='PARAMETRIC', name='full', location=[1, 0, shift2],
                           color='drawing', extrude=0.001, thickness=0.35, num_points=int(len(curve) / 2), emission=0.1,
                           rotation_euler=[0, np.pi / 2, np.pi / 2])

        fractal.grow(begin_time=t0, transition_time=duration)
        spider.armature.follow(fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                               forward_axis='TRACK_NEGATIVE_Y')

        spider.disappear(begin_time=t0 + duration, transition_time=0.1)
        t0 += duration

        print("finished at ", t0)

    def curves4(self):
        cues = self.sub_scenes['curves4']
        t0 = 0.5  # cues['start']

        points = [[1, 0], [0.31, 0.95], [-0.81, 0.59], [-0.81, -0.59], [0.31, -0.95], [1, 0]]

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0)

        title = SimpleTexBObject(r"\text{From point to point}", color='example', aligned='center')
        display.write_title(title, begin_time=t0)

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, 0, np.pi]
                             )
        spider.armature.ref_obj.scale = [0.1, 0.1, 0.1]

        functions = []
        for i in range(len(points)):
            if i < len(points) - 1:
                a = points[i]
                b = points[i + 1]
            else:
                a = points[i]
                b = points[0]
            a = Vector([a[0], 0, a[1]])
            b = Vector([b[0], 0, b[1]])
            functions.append(Curve(lambda x: a + x * (b - a), domain=[0, 1], color='drawing', num_points=3,
                                   name='curve_' + str(i), thickness=0.125, scale=[4, 4, 4], location=[-4, 0, 0]))

        for i, function in enumerate(functions):
            ibpy.set_follow(spider.armature, function)
            if i == 0:
                ibpy.set_follow_influence(spider.armature, function, value=1)
            else:
                ibpy.set_follow_influence(spider.armature, function, value=0)

        f_old = None
        point_texts = []
        align_indices = [5, 6, 6, 5, 2]
        for i, point in enumerate(points):
            point_texts.append(SimpleTexBObject(r"(" + str(point[0]) + "," + str(point[1]) + ")", color='drawing',
                                                name='text_' + str(i)))
            if i > 0:
                point_texts[i].align(point_texts[0], align_indices[i - 1], 2)
            display.write_text_in(point_texts[-1], begin_time=t0, indent=7, line=i + 1)
            if i == 0:
                spider.appear(begin_time=t0 + 0.9, transition_time=0.1)
            else:
                function = functions[i - 1]
                function.grow(begin_time=t0)
                if f_old:
                    ibpy.change_follow_influence(spider.armature, f_old, initial=1, final=0, begin_time=t0,
                                                 transition_time=0)
                    ibpy.change_follow_influence(spider.armature, function, initial=0, final=1, begin_time=t0,
                                                 transition_time=0)
                if i == 3:
                    spider.armature.rotate(rotation_euler=[0, np.pi / 2, np.pi], begin_time=t0, transition_time=0)
                if i > 3:
                    spider.armature.rotate(rotation_euler=[0, np.pi, np.pi], begin_time=t0, transition_time=0)
                spider.armature.follow(function, begin_time=t0, new_constraint=False, up_axis='UP_X')
                f_old = function
            t0 += 1
        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5
        print("finished at ", t0)

    def curves4b(self):
        cues = self.sub_scenes['curves4b']
        t0 = 0  # cues['start']

        points = [[1, 0], [0.31, 0.95], [-0.81, 0.59], [-0.81, -0.59], [0.31, -0.95], [1, 0]]

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0, transition_time=0)

        title = SimpleTexBObject(r"\text{From point to point}", color='example', aligned='center')
        display.write_title(title, begin_time=t0, transition_time=0)

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, 0, np.pi]
                             )
        spider.armature.ref_obj.scale = [0.1, 0.1, 0.1]

        functions = []
        for i in range(len(points)):
            if i < len(points) - 1:
                a = points[i]
                b = points[i + 1]
            else:
                a = points[i]
                b = points[0]
            a = Vector([a[0], 0, a[1]])
            b = Vector([b[0], 0, b[1]])
            functions.append(Curve(lambda x: a + x * (b - a), domain=[0, 1], color='drawing', num_points=3,
                                   name='curve_' + str(i), thickness=0.125, scale=[4, 4, 4], location=[-4, 0, 0]))

        point_texts = []
        align_indices = [5, 6, 6, 5, 2]
        for i, point in enumerate(points):
            point_texts.append(SimpleTexBObject(r"(" + str(point[0]) + "," + str(point[1]) + ")", color='drawing',
                                                name='text_' + str(i)))
            if i > 0:
                point_texts[i].align(point_texts[0], align_indices[i - 1], 2)
            display.write_text_in(point_texts[-1], begin_time=t0, indent=7, line=i + 1, transition_time=0)
            if i > 0:
                function = functions[i - 1]
                function.grow(begin_time=t0, transition_time=0)
        t0 += 0.5
        # reshuffle points
        old_positions = []
        for text in point_texts:
            old_positions.append(ibpy.get_location_at_frame(text, t0 * FRAME_RATE))

        new_pos_indices = [0, 3, 1, 4, 2, 5]
        for text, new_pos_index in zip(point_texts, new_pos_indices):
            text.move_to(target_location=old_positions[new_pos_index], begin_time=t0)
            text.change_color('important', begin_time=t0 + 0.5, transition_time=0.5)

        new_functions = []
        for i in range(len(points)):
            if i < len(points) - 1:
                a = points[new_pos_indices[i]]
                b = points[new_pos_indices[i + 1]]
            a = Vector([a[0], 0, a[1]])
            b = Vector([b[0], 0, b[1]])
            new_functions.append(Curve(lambda x: a + x * (b - a), domain=[0, 1], color='important', num_points=3,
                                       name='curve' + str(i), thickness=0.125, scale=[4, 4, 4], location=[-4, 0, 0]))

        for i, function in enumerate(new_functions):
            ibpy.set_follow(spider.armature, function)
            if i == 0:
                ibpy.set_follow_influence(spider.armature, function, value=1)
            else:
                ibpy.set_follow_influence(spider.armature, function, value=0)

        f_old = None
        angles = [0, np.pi, np.pi / 2, 0, np.pi, 0]
        for i, point in enumerate(points):
            if i == 0:
                spider.appear(begin_time=t0 + 0.9, transition_time=0.1)
            else:
                function = new_functions[i - 1]
                function.grow(begin_time=t0)
                if f_old:
                    ibpy.change_follow_influence(spider.armature, f_old, initial=1, final=0, begin_time=t0,
                                                 transition_time=0)
                    ibpy.change_follow_influence(spider.armature, function, initial=0, final=1, begin_time=t0,
                                                 transition_time=0)
                if i == 3:
                    spider.armature.rotate(rotation_euler=[0, np.pi / 2, np.pi], begin_time=t0, transition_time=0)
                if i > 3:
                    spider.armature.rotate(rotation_euler=[0, np.pi, np.pi], begin_time=t0, transition_time=0)
                spider.armature.rotate(rotation_euler=[0, angles[i - 1], np.pi], begin_time=t0, transition_time=0)
                spider.armature.follow(function, begin_time=t0, new_constraint=False, up_axis='UP_X')
                f_old = function
            t0 += 1
        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5

        print("finished at ", t0)

    def curves5(self):
        cues = self.sub_scenes['curves5']
        t0 = 0.25  # cues['start']

        display = Display(location=[0, 0.25, 0], scales=[9, 5], number_of_lines=10, flat=True)
        display.appear(begin_time=t0)

        title = SimpleTexBObject(r"\text{The Apollonian Gasket}", color='example', aligned='center')
        display.write_title(title, begin_time=t0)

        spider = AnimBObject('Spider', object_name='widow',
                             armature='Armature', color='joker', scale=1,
                             emission=[0, 0.75], rotation_euler=[0, 0, np.pi]
                             )
        spider.armature.ref_obj.scale = [0.1, 0.1, 0.1]

        eps = 0.001
        average = 100
        scale = 8
        curve = read_complex_data("apollonian_" + str(eps) + ".dat")
        resolution = int(10 / eps)

        origin = [-4, 0, -4.5]

        fx = lambda x: function_from_complex_list(curve, x, scale=scale)
        fx2 = lambda x: function_from_complex_list(curve, x, scale=scale, average=average)

        coords = CoordinateSystem(dim=2, lengths=[8, 8], domains=[[-4, 4], [0, 8]], radii=[0.03, 0.03],
                                  all_n_tics=[8, 8],
                                  all_tic_labels=[np.arange(-4, 4.1, 1), np.arange(-0, 8.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[1, 1], label_units=['', ''],
                                  axis_labels=['', ''],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=origin,
                                  name='coordinateSystem')
        coords.appear(begin_time=t0, transition_time=2)

        t0 += 2
        duration = 30

        lines_on_display = 8
        lps = 4
        dt = 1 / lps

        fractal = Function([fx], None, domain=[0, 1], mode='PARAMETRIC', name='a_full',
                           location=origin,
                           rotation_euler=[np.pi / 2, 0, 0],
                           color='drawing', extrude=0.001, thickness=0.1, metallic=1, roughness=0,
                           num_points=resolution, emission=0.1)

        average_fractal = Function([fx2], None, domain=[0, 1], mode='PARAMETRIC',
                                   location=origin,
                                   rotation_euler=[np.pi / 2, 0, 0],
                                   name='average', color='example', extrude=0, thickness=0,
                                   numpoints=resolution / average)

        spider.appear(begin_time=t0, transition_time=0.1)
        fractal.grow(begin_time=t0, transition_time=duration)
        spider.armature.follow(average_fractal, initial_value=0, final_value=1, begin_time=t0, transition_time=duration,
                               forward_axis='FORWARD_Y')

        t_reset = t0
        text_string = ""
        sphere_appear_times = []
        for i in range(0, int(lps * duration)):
            if i > 0:
                text_string += r"\\"
            frame = t0 * FRAME_RATE
            offset = ibpy.get_offset_factor_at_frame(spider.armature, average_fractal, frame);
            print(i, frame, offset)
            fs = fx(offset)
            x = np.round(fs[0] * 100) / 100
            y = np.round(fs[1] * 100) / 100
            sphere = Sphere(0.05, location=[x, 0, y], color='text')
            coords.add_object(sphere)
            text_string += "(" + str(x) + ",&" + str(y) + ")"
            sphere.grow(begin_time=t0, transition_time=dt / 2)
            sphere_appear_times.append(t0)
            t0 += dt

        text = MultiLineTexBObject(text_string, color='text', aligned='left_top')
        display.add_text_in(text, line=0, indent=7.5, shift=[0, 2])

        for i in range(9, len(sphere_appear_times) - 8):
            time = sphere_appear_times[i]
            text.appear_line(i, begin_time=time, transition_time=dt)

        for i in range(9, len(sphere_appear_times) - lines_on_display - 8):
            text.disappear_line_at_once(i, begin_time=sphere_appear_times[i + lines_on_display - 1], transition_time=dt)

        text.move(direction=[0, 20, 0], begin_time=sphere_appear_times[lines_on_display + 8],
                  transition_time=sphere_appear_times[-9] - sphere_appear_times[lines_on_display + 8])
        ibpy.set_linear_fcurves(text)
        spider.disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.5
        print("finished at ", t0)

    def curves6(self):
        cues = self.sub_scenes['curves6']
        t0 = 0.25  # cues['start']

        colors = ['drawing', 'important', 'joker', 'example', 'text']
        epsilons = [0.5, 0.1, 0.05, 0.01, 0.001]
        radii = [0.01, 0.0075, 0.005]
        thicknesses = [0.1, 0.1, 0.1, 0.05, 0.025]
        count = 0
        spheres = []
        for epsilon, color, thickness in zip(epsilons, colors, thicknesses):
            dfs = DepthFirstSearchWithFixedPoints(ApollonianModel, max_sep=epsilon)
            dfs.generate_tree()

            points = dfs.points
            duration = 5
            scale = 11.75

            if epsilon > 0.01:
                dt = duration / len(points)
                r = radii[count]
                for p in points:
                    sphere = Sphere(r * scale, location=z2vec(p, z_dir=True) * scale, color=color)
                    sphere.grow(begin_time=t0, transition_time=dt)
                    spheres.append(sphere)
                    t0 += dt

            t0 += 0.5
            curve = BezierDataCurve(name='Epsilon' + str(epsilon), rotation_euler=[np.pi / 2, 0, 0],
                                    scale=scale,
                                    data=[z2vec(item) for item in dfs.points], color=color, thickness=thickness)
            curve.grow(begin_time=t0, transition_time=duration)
            t0 += duration + 0.5
            if count == 3:
                for sphere in spheres:
                    sphere.disappear(alpha=0.2, begin_time=t0, transition_time=0.1)
            if count < 4:
                curve.disappear(begin_time=t0, transition_time=0.1)
            count += 1

        t0 += 2

        print("finished at ", t0)

    def clock(self):
        cues = self.sub_scenes['clock']
        t0 = 0.5  # cues['start']

        r = 6
        circle = BezierCircle(radius=r, rotation_euler=[np.pi / 2, 0, 0], thickness=10, color='drawing', resolution=100)
        circle.appear(begin_time=t0)

        arrow = PArrow(start=Vector(), end=0.825 * r * Vector([0, 0, 1]), color='important', thickness=5)
        arrow.grow(begin_time=t0)

        number_strings = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
        numbers = []
        angle = np.pi / 2
        dt = 1 / 12
        for number_string in number_strings:
            angle -= np.pi / 6
            number = SimpleTexBObject(r"\text{" + number_string + "}", aligned="center", color='drawing',
                                      location=[0.875 * r * np.cos(angle), 0, 0.9 * r * np.sin(angle)],
                                      text_size='huge')
            numbers.append(number)
            number.write(begin_time=t0, transition_time=0.3)
            t0 += dt

        t0 += 0.5
        arrow.rotate(rotation_euler=[0, np.pi * 4, 0], begin_time=t0, transition_time=120)
        ibpy.set_linear_fcurves(arrow)
        t0 += 2.5
        print("finished at ", t0)

    def scaling(self):
        cues = self.sub_scenes['scaling']
        t0 = 0.5  # cues['start']

        display = Display(location=[-6, 0, 5], number_of_lines=2, scales=[4, 1], flat=True)
        display.appear(begin_time=t0)

        colors = flatten([['drawing'], ['text'] * 2, ['important'] * 2])
        title = SimpleTexBObject(r'x\mapsto 2x', color=colors, aligned='center')

        t0 = display.write_title(title, shift=[-1, 0, 0.025], begin_time=t0) + 0.5

        nl = NumberLine(domain=[-1, 7], length=8, n_tics=8, direction='horizontal', location_of_origin=[-9, 0, 0],
                        color='text', label_color='text')
        t0 = nl.appear(begin_time=t0, transition_time=3)

        p1 = Sphere(0.25, location=[-8, 0, 0], color='drawing')
        p1.grow(begin_time=t0)

        p2 = Sphere(0.25, location=[-6, 0, 0], color='drawing')
        t0 = p2.grow(begin_time=t0) + 0.5

        cp1 = p1.copy(emission=0)
        cp1.move(direction=[1, 0, 0], begin_time=t0)
        cp1.change_color(new_color='important', begin_time=t0 + 0.5, transition_time=0.5)

        cp2 = p2.copy(emission=0)
        cp2.move(direction=[3, 0, 0], begin_time=t0)
        t0 = cp2.change_color(new_color='important', begin_time=t0 + 0.5, transition_time=0.5) + 0.5

        nl2 = NumberLine(domain=[-1, 7], length=8, n_tics=8, direction='horizontal', location_of_origin=[-9, 0, -2],
                         color='text', label_color='text')
        t0 = nl2.appear(begin_time=t0, transition_time=3)

        cyl = Cylinder.from_start_to_end(start=[-7, 0, -2], end=[-6, 0, -2], color='drawing', radius=0.25)
        t0 = cyl.grow(begin_time=t0) + 0.5

        c_cyl = cyl.copy()
        c_cyl.move(direction=[2, 0, 0], begin_time=t0)
        c_cyl.rescale(rescale=[1, 1, 2], begin_time=t0)
        t0 = c_cyl.change_color(new_color='important', begin_time=t0 + 0.5, transition_time=0.5) + 0.5

        display2 = Display(location=[6, 0, 5], number_of_lines=2, scales=[4, 1], flat=True)
        display2.appear(begin_time=t0)
        title = SimpleTexBObject(r'z\mapsto 2z', color=colors, aligned='center')
        t0 = display2.write_title(title, shift=[-1, 0, 0.025], begin_time=t0) + 0.5

        coords = CoordinateSystem(dim=2, lengths=[10, 8], domains=[[-5, 5], [-4, 4]], radii=[0.03, 0.03],
                                  all_n_tics=[10, 8],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-5, 5.1, 1), np.arange(-4, 4.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[5, 0, -2],
                                  name='ComplexPlane')
        t0 = coords.appear(begin_time=t0, transition_time=2) + 0.5
        # adjust position of the zero label manually
        coords.axes[0].labels[5].ref_obj.location = [0.4, 0, -0.20651]

        transformations = [
            lambda z: 2 * z,
            lambda z: 4 * z,
            lambda z: 2 * z,
            lambda z: z,
        ]

        annulus = Annulus(r=[0.5, 1], phi=[0, 2 * np.pi], color='joker',
                          coordinate_system=coords,
                          conformal_transformations=transformations,
                          rotation_euler=[np.pi / 2, 0, 0], resolution=20,
                          smooth=2, solid=0.1, offset=0, bevel=0.1,
                          )
        t0 = annulus.appear(begin_time=t0) + 0.5

        t0 = annulus.next_shape(begin_time=t0) + 0.5

        t0 = annulus.next_shape(begin_time=t0) + 0.5

        colors2 = flatten([['drawing'], ['text'] * 2, ['important'] * 5])
        title2 = SimpleTexBObject(r"z\mapsto \tfrac{1}{2} z", color=colors2, aligned='center')
        t0 = title.replace(title2, begin_time=t0) + 0.5

        t0 = annulus.next_shape(begin_time=t0) + 0.5

        t0 = annulus.next_shape(begin_time=t0) + 0.5

        # disappear things

        display.disappear(begin_time=t0)
        display2.disappear(begin_time=t0)
        nl.disappear(begin_time=t0)
        nl2.disappear(begin_time=t0)
        cyl.disappear(begin_time=t0)
        c_cyl.disappear(begin_time=t0)
        p1.disappear(begin_time=t0)
        p2.disappear(begin_time=t0)
        cp1.disappear(begin_time=t0)
        t0 = cp2.disappear(begin_time=t0)

    def theory(self):
        cues = self.sub_scenes['theory']
        t0 = 0.5  # cues['start']

        title = SimpleTexBObject(r"\text{The condition for a single fixed point}", color='example', aligned='center')
        display = Display(flat=True, number_of_lines=15, scales=[11.5, 6], location=[0, 0, 0])
        t0 = display.appear(begin_time=t0)

        t0 = 0.5 + display.write_title(title, begin_time=t0, transition_time=3)

        colors1 = flatten(
            [['joker'], ['text'] * 4, ['joker'] * 2, ['text'] * 5, ['joker'] * 2, ['text'] * 4, ['important'] * 7])
        moebius = SimpleTexBObject(r"m\colon z\mapsto {az+b\over cz+d} \,\,\,\text{ with } \,\,\,ad-bc=1",
                                   color=colors1)
        t0 = 0.5 + display.write_text_in(moebius, line=1, indent=1, begin_time=t0, transition_time=3)

        colors2 = flatten([['text'] * 2, ['joker'] * 2, ['text'] * 5, ['joker'] * 2])
        eq_fix = SimpleTexBObject(r"z={az+b\over cz+d}", color=colors2)
        t0 = 0.5 + display.write_text_in(eq_fix, line=3, indent=1, begin_time=t0, transition_time=2)

        colors3 = flatten(
            [['text'] * 2, ['joker'], ['text'] * 4, ['joker'], ['text'], ['joker'], ['text'] * 3, ['joker']])
        eq_fix2 = SimpleTexBObject(r"0=cz^2+(d-a)z-b", color=colors3)
        t0 = 0.5 + display.write_text_in(eq_fix2, line=5, indent=1, begin_time=t0, transition_time=2)

        colors4 = flatten([['text'] * 3, ['joker'], ['text'] * 3, ['joker'] * 2, ['text'] * 4, ['joker'],
                           ['text'] * 2, ['joker'], ['text'], ['joker'], ['text'] * 5, ['joker']])
        sol = SimpleTexBObject(r"z_\pm={a-d\over 2c}\pm{1\over 2c}\sqrt{(a-d)^2+4bc}", color=colors4)
        t0 = 0.5 + display.write_text_in(sol, line=7, indent=1, begin_time=t0, transition_time=2)

        colors5 = flatten([['text'] * 3, ['joker'], ['text'], ['joker'], ['text'] * 4, ['joker'] * 2])
        discr = SimpleTexBObject(r"0=(a-d)^2+4bc", color=colors5)
        t0 = 0.5 + display.write_text_in(discr, line=9, indent=1, begin_time=t0, transition_time=2)

        colors6 = flatten([['text'] * 3, ['joker'], ['text'], ['joker'], ['text'] * 4, ['important'] * 6])
        discr2 = SimpleTexBObject(r"0=(a-d)^2+4(ad-1)", color=colors6)
        t0 = 0.5 + discr.replace(discr2, img_letter_range=[0, 16], begin_time=t0, morphing=True)

        colors7 = flatten([['text'] * 18, ['joker'], ['text'], ['joker'], ['text']])
        discr3 = SimpleTexBObject(r"0=(a-d)^2+4(ad-1)=(a+d)^2-4", color=colors7)
        display.add_text_in(discr3, line=9, indent=1)
        t0 = 0.5 + discr3.write(letter_range=[16, 25], begin_time=t0)

        matrix = BMatrix(np.array([['a', 'b'], ['c', 'd']]), pre_word='M=', color='joker')
        t0 = 0.5 + display.write_text_in(matrix, line=11, begin_time=t0, indent=1, transition_time=2)

        result = SimpleTexBObject(r"\text{\fbox{\text{$\text{tr}(M)=\pm 2$}}}", color='example')
        display.add_text_in(result, line=11, indent=5)
        t0 = result.write(letter_set={1, 2, 3, 4, 5, 8, 9, 10}, begin_time=t0)
        t0 = 0.5 + result.write(letter_set={0, 6, 7, 11}, begin_time=t0)

        am = ApollonianModel()
        matrix_a = BMatrix(np.array([['1', '0'], ['-2i', '1']]), pre_word='a=', color=am.colors[0], outlined='text')
        matrix_b = BMatrix(np.array([['1-i', '1'], ['1', '1+i']]), pre_word='b=', color=am.colors[1])

        display.write_text_in(matrix_a, line=2, indent=7, begin_time=t0, transition_time=2)
        t0 = 0.5 + display.write_text_in(matrix_b, line=5, indent=7, begin_time=t0, transition_time=2)

        print("finished at ", t0)

    def actors(self):
        cues = self.sub_scenes['actors']
        t0 = 0.5  # cues['start']

        ibpy.set_camera_location(location=[0, -10, 0])

        words = ["Apollonian gasket",
                 "Commutator", "Complex numbers",
                 "Complex plane", "Descart's theorem",
                 "Dihedral group", "Fixed point",
                 "Fractal curve", "Free group",
                 "Limit set", "Linear transformation",
                 "Matrix", r"M\"obius transformation",
                 "Orthogonal group", "Riemann sphere",
                 "Special linear group", "Stereographic projection",
                 "Symmetry"
                 ]
        content = ""
        for i, w in enumerate(words):
            if i > 0:
                if i % 3 == 0:
                    sep = r"\\"
                elif i > 0:
                    sep = "&"
                content += (sep + w)
            else:
                content += w

        expression = r"\fbox{\parbox{1\linewidth}{\underline{\bf List of Actors in alphabetical Order}\\\\\begin{" \
                     r"tabular}{l l l}" + content + r"\end{tabular}}}"
        text = SimpleTexBObject(expression, text_only=True, aligned="center")

        letter = list(range(0, 272))
        box = [0, 166, 167, 271]
        for b in box:
            letter.remove(b)
        text.write(letter_set=letter, begin_time=t0, transition_time=5, sorting='natural')
        print("finished at ", t0)

    def indentations(self):
        cues = self.sub_scenes['indentations']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[0,-20,10])
        ibpy.camera_zoom(lens=4,begin_time=t0,transition_time=0)
        camera_empty=EmptyCube(location=[0,-20,0])
        ibpy.set_camera_view_to(camera_empty)

        fundamental_domain = BObject.from_file(filename='FundamentalDomainFlat', location=[0, 0, 0],emission=1)
        t0=fundamental_domain.appear(begin_time=t0)
        fundamental_domain.add_mesh_modifier(type='WIREFRAME',thickness=0.004,use_even_offset=False)

        am = ApollonianModel()
        b = am.get_generators()[1]

        ibpy.camera_move(shift=[0, 20, 0], begin_time=t0, transition_time=5)
        camera_empty.move(direction=[0,20,0],begin_time=t0,transition_time=5)
        ibpy.camera_zoom(lens=60,begin_time=t0,transition_time=10)

        trafos = [
            lambda x: moebius_vec(b, x),
        ]
        t0=fundamental_domain.transform_mesh(trafos[0], begin_time=t0,transition_time=10)

        ibpy.camera_move(shift=[1, -1, 0], begin_time=t0, transition_time=5)
        camera_empty.move(direction=[1, -1, 0], begin_time=t0, transition_time=5)
        ibpy.camera_zoom(lens=400, begin_time=t0, transition_time=10)

        print("finished ",t0)

    def linear_transformations(self):
        cues = self.sub_scenes['linear_transformations']
        t0 = 0  # cues['start']

        display=Display(location=Vector(),number_of_lines=17,flat=True)
        display.appear(begin_time=t0)

        coords = CoordinateSystem(dim=2, lengths=[4, 4], domains=[[-4, 4], [-4, 4]], radii=[0.03, 0.03],
              all_n_tics=[4, 4],
              labels=[r"x", r"y"],
              all_tic_labels=[np.arange(-2, 2.1,1), np.arange(-2, 2.1,1)],
              colors=['text', 'text'], label_colors=['text', 'text'],
              label_digits=[0, 0], label_units=['', ''],
              axis_label_size='medium',
              tic_label_size='small',
              location_of_origin=[0, -0.2, -1],
              name='CoordinateSystem')
        t0=coords.appear(begin_time=t0, transition_time=2)
        coords.axes[0].labels[2].ref_obj.location = [0.4, 0, -0.20651]

        boost = np.array([[1.25, -0.75,0], [-0.75, 1.25,0],[0,0,1]])

        b_trafo = VectorTransformation()
        # look at the mathematica worksheet "VectorTransformations" to see how the expressions are generated
        b_trafo.set_transformation_function(
            lambda v: Vector(
                [5/4*v.x-3/4*v.z,0,-3/4*v.x+5/4*v.z])
        )
        b_trafo.set_first_derivative_functions(lambda v:  np.array(
            [[5/4,0,-3/4], [0, 0, 0],
             [-3/4,0,5/4]]
        ))
        b_trafo.set_second_derivative_functions(lambda v:  np.array(
            [[[0,0,0], [0, 0, 0],
              [0,0,0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0,0,0], [0, 0, 0],
              [0,0,0]]
             ]))
        coords.draw_transformable_grid(transformations=[b_trafo],begin_time=t0,transition_time=2,colors=['text','text'],thickness=0.3)

        t0+=2.5
        colors=flatten([['drawing']*16,['text'],['example']*4,['text'],['important']*9])
        line = SimpleTexBObject(r"\left(\begin{array}{c c} \tfrac{5}{4} & -\tfrac{3}{4}\\ -\tfrac{3}{4} &\tfrac{5}{4}  \end{array}\right)\cdot\left(\begin{array}{c}1\\0\end{array}\right)=\left(\begin{array}{c}\tfrac{5}{4}\\-\tfrac{3}{4}\end{array}\right)",color=colors)
        display.add_text_in(line,line=0,indent=0.5)
        t0=line.write(letter_set=range(0,22),begin_time=t0,transition_time=4.5)
        t0+=0.5

        sphere  = Sphere(0.1,location=Vector([1,0,0]),color='example')
        coords.add_object(sphere)
        sphere.grow(begin_time=t0-1)

        line.write(letter_set=range(22,31),begin_time=t0+1,transition_time=1)
        coords.grid_next_transform(begin_time=t0,transition_time=4.5)
        coords.grid_change_color(new_color='drawing',begin_time=t0,transition_time=4.5)
        copy = sphere.move_copy_to(target_location=Vector([5/4,0,-3/4]),begin_time=t0,transition_time=4.5,color='important'        )

        print("finshed ",t0)

    def sl2c(self,transparency=True):
        cues = self.sub_scenes['sl2c']
        t0 = 0  # cues['start']

        display = Display(location=Vector(), number_of_lines=7, flat=True,scales=[4,4])
        t0=display.appear(begin_time=t0)

        matrix_a = BMatrix(np.array([['1', '0'], ['-2i', '1']]), pre_word="a=", color='gray_2',outlined='text')
        matrix_b = BMatrix(np.array([['1-i', '1'], ['1', '1+i']]), pre_word="b=",color='joker')

        colors = ['text', 'gray_2', 'text', 'joker', 'text','text','example']
        sl2c = SimpleTexBObject(r'\{a,b\}\in SL_2(\mathbb{C})',color=colors)
        t0=display.write_text_in(matrix_a,indent=0.5,begin_time=t0,line=0,transition_time=2)
        t0+=0.5
        t0=display.write_text_in(matrix_b,indent=0.5,begin_time=t0,line=3,transition_time=2)
        t0+=0.5
        t0=display.write_text_in(sl2c,indent=0.5,begin_time=t0,line=5,transition_time=1)
        t0+=0.5
        print("finished ", t0)

    def inverses(self, transparency=True):
        cues = self.sub_scenes['sl2c']
        t0 = 0  # cues['start']

        display = Display(location=Vector(), number_of_lines=5, flat=True, scales=[4, 4])
        t0 = display.appear(begin_time=t0)

        am = ApollonianModel()
        colors = am.colors
        lines =[
            SimpleTexBObject("a^{-1}=A",color=[colors[0],'text','text','text',colors[2]]),
            SimpleTexBObject("b^{-1}=B",color=[colors[1],'text','text','text',colors[3]]),
            SimpleTexBObject("aA=Aa=1",color=[colors[0],colors[2],'text',colors[2],colors[0],'text']),
            SimpleTexBObject("bB=Bb=1",color=[colors[1],colors[3],'text',colors[3],colors[1],'text'])
        ]

        for l,line in enumerate(lines):
            t0=0.5+display.write_text_in(line,line=l-0.5,indent=1,begin_time=t0)
        print("finished ", t0)

    def further_reading(self):
        cues = self.sub_scenes['further_reading']
        t0 = 0.5  # cues['start']
        print("finished at ", t0)


if __name__ == '__main__':
    try:
        example = ApollonianFractal()
        dict = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dict[i] = scene
        # choice = input("Choose scene:")
        # print("Your choice: ", choice)
        # selected_scene = dict[int(choice)]
        # selected_scene = 'title'
        # selected_scene = 'title2'
        # selected_scene = 'title3'
        # selected_scene = 'title4'
        # selected_scene = 'title5'
        # selected_scene = 'epsilon'
        # selected_scene = 'flyby'
        # selected_scene = 'one_minute_trafo_text'
        # selected_scene = 'one_minute_trafo_one'
        # selected_scene = 'one_minute_trafo_two'
        # selected_scene = 'one_minute_trafo_three'
        # selected_scene = 'one_minute_trafo_four'
        # selected_scene = 'one_minute_tree'
        # selected_scene = 'create_fundamental_domain'
        # selected_scene = 'coding'
        # selected_scene = 'towards_apollonian'
        # selected_scene = 'finale'
        # selected_scene = 'intro'
        # selected_scene = 'intro2'
        # selected_scene = 'curvature_gasket'
        # selected_scene = 'zoom'
        # selected_scene = 'induced_transformation'
        # selected_scene = 'on_the_sphere'
        # selected_scene = 'rotate_sphere'
        # selected_scene = 'algebra'
        # selected_scene = 'summary'
        # selected_scene = 'fixed_points'
        # selected_scene = 'fixed_points2'
        # selected_scene = 'fixed_points3'
        # selected_scene = 'variant'
        # selected_scene = 'doubling'
        # selected_scene = 'pre_full_picture'
        # selected_scene = 'full_picture_start'
        # selected_scene = 'full_picture'
        # selected_scene = 'full_picture_spider'
        # selected_scene = 'groups'
        # selected_scene = 'groups1b'
        # selected_scene = 'groups2'
        # selected_scene = 'groups3'
        # selected_scene = 'discs_and_group'
        # selected_scene = 'tree_extension'
        # selected_scene = 'tree_extension2'
        # selected_scene = 'apollonian'
        # selected_scene = 'apollonian_full'
        # selected_scene = 'commutators'
        # selected_scene = 'commutators_adaptive'
        # selected_scene = 'commutators_adaptive_ii'
        # selected_scene = 'geogebra'
        # selected_scene = 'pattern'
        # selected_scene = 'families'
        # selected_scene = 'ancient'
        # selected_scene = 'curves1'
        # selected_scene = 'curves2'
        # selected_scene = 'curves3'
        # selected_scene = 'curves3b'
        # selected_scene = 'curves3c'
        # selected_scene = 'curves3d'
        # selected_scene = 'curves4'
        # selected_scene = 'curves4b'
        # selected_scene = 'curves5'
        # selected_scene = 'curves6'
        # selected_scene = 'clock'
        # selected_scene = 'scaling'
        # selected_scene = 'theory'
        # selected_scene = 'actors'
        # selected_scene = 'indentations'
        # selected_scene = 'linear_transformations'
        selected_scene = 'inverses'
        # selected_scene = 'sl2c'

        if 'geogebra' in selected_scene or 'pattern' in selected_scene or 'curves6' in selected_scene:
            resolution = [1080, 1080]
        elif 'families' in selected_scene:
            resolution = [500, 1080]
        elif 'ancient' in selected_scene:
            resolution = [720, 720]
        else:
            resolution = [1920, 1080]
        example.create(name=selected_scene, resolution=resolution, start_at_zero=True)
        # example.render(debug=True)
        # doesn't work
        # example.final_render(name=selected_scene,debug=False)
    except:
        print_time_report()
        raise ()
