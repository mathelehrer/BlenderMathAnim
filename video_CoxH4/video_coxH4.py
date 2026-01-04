import itertools
import os
from collections import OrderedDict

import numpy as np
from anytree import Node
from sympy.combinatorics import Permutation

import interface.ibpy
from appearance.textures import get_texture
from compositions.compositions import create_glow_composition
from geometry_nodes.geometry_nodes_modifier import CustomUnfoldModifier, EdgePairingVisualizer
from interface import ibpy
from interface.ibpy import Quaternion, Vector
from interface.interface_constants import BLENDER_EEVEE, CYCLES
from mathematics.geometry.coxA3 import CoxA3, COXA3_SIGNATURES
from mathematics.geometry.coxB3 import CoxB3, COXB3_SIGNATURES
from mathematics.geometry.coxH3 import CoxH3, COXH3_SIGNATURES
from mathematics.geometry.coxH4 import CoxH4
from mathematics.geometry.coxP6 import CoxP6, COXP6_SIGNATURES
from mathematics.geometry.field_extensions import FVector
from addons.solids import SOLID_FACE_SIDES, compute_similarity_transform, apply_similarity_to_vertices, \
    get_solid_data
from geometry_nodes.geometry_nodes_modifier import PolytopeViewerModifier, PolyhedronViewModifier, \
    StereographicProjectionModifier
from objects.arc import Arc2
from objects.bobject import BObject
from objects.circle import Circle2
from objects.codeparser import CodeParser
from objects.curve import Curve
from objects.cylinder import Cylinder
from objects.derived_objects.p_arrow import PArrow
from objects.display import CodeDisplay
from objects.dynkin_diagram import DynkinDiagram
from objects.empties import EmptyCube
from objects.floor import Floor
from objects.geometry.sphere import Sphere
from objects.light.light import SpotLight
from objects.logo import LogoFromInstances
from objects.plane import Plane
from objects.polyhedron import Polyhedron
from objects.table import Table
from objects.tex_bobject import SimpleTexBObject
from objects.text import Text
from perform.scene import Scene
from utils.constants import DATA_DIR, LOC_FILE_DIR
from utils.utils import print_time_report, to_vector
from video_CoxH4.create_csv_file_from_polytope_data import create_csv_file_from_polytope_data

pi = np.pi
tau: float = 2 * pi
r2 = np.sqrt(2)
EPS = 1e-6

def epsilon(rank: int = 4) -> np.ndarray:
    """
    Computes completely antisymmetric tensor
    """
    n = rank ** rank
    comps = np.array([0] * n)
    comps.shape = (rank,) * rank

    permutations = list(itertools.permutations(range(rank)))
    for permutation in permutations:
        p = Permutation(permutation)
        comps[permutation] = p.signature()

    return comps

def normal_to(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    abc = np.tensordot(np.tensordot(a, b, axes=0), c, axes=0)
    n = np.tensordot(epsilon(4), abc, axes=[[1, 2, 3], [0, 1, 2]])
    return interface.ibpy.Vector(n)

def create_full_cell_tree(signature, root_index = -1,root_size=-1):
    group = CoxH4(path="../mathematics/geometry/data")
    cells = group.get_cells(signature)
    faces = group.get_faces(signature)
    vertices = group.point_cloud(signature)

    index_to_cell_map = {i: cell for i, cell in enumerate(cells.keys())}
    index_to_normal_map = {i: cell for i, cell in enumerate(cells.values())}

    if root_index==-1:
        if root_size > 0:
            for idx, cell in index_to_cell_map.items():
                if len(cell) == root_size:
                    root_cell_index = idx
                    break
        else:
            root_cell_index = 0
    else:
        root_cell_index = root_index

    root_cell = index_to_cell_map[root_cell_index]
    root_normal = index_to_normal_map[root_cell_index]
    index_to_cell_map.pop(root_cell_index)

    root = Node((root_cell_index, len(root_cell), set(root_cell), root_normal.real()))

    parents = [root]
    while len(index_to_cell_map) > 0:
        next_level = []
        for parent in parents:
            new_nodes = []
            for index, cell in index_to_cell_map.items():
                cell_set = set(cell)
                # create a parent child connection, when the two cells share at least 3 vertices (i.e. a common face)
                # TODO this has to be made smarter, when one wants to extend this to even higher dimensions
                if len(cell_set & parent.name[2]) > 2:
                    node = Node((index, len(cell_set), cell_set, index_to_normal_map[index].real()), parent=parent)
                    new_nodes.append(node)
                    next_level.append(node)

            # pop children
            for child in new_nodes:
                index_to_cell_map.pop(child.name[0])

        parents = next_level
        print(str(len(parents)) + "new children added ")

    return vertices, faces, cells, root


def read_signatures(filename):
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        data = []
        lines = f.readlines()
        for line in lines:
            # remove new line
            if line[-1] == "\n":
                line = line[0:-1]
            parts = line.split("->")
            signature = eval(parts[0])
            p0 = FVector.parse(parts[1])
            data.append((signature, p0))
    return data


def append_to_good(text, filename):
    with open(os.path.join(DATA_DIR, filename), "a") as f:
        f.write(str(text) + "\n")


def indexing_unfolded_vertices(root, vertices):
    unfolded2vertex_map = {}
    vertex2unfolded_map = {}
    unfolded_vertices = []  # right now, it is just a copy of the original vertices
    new_unfolded_indices = []
    for i in root.name[2]:
        j = len(unfolded2vertex_map)
        unfolded2vertex_map[j] = i
        new_unfolded_indices.append(j)
        unfolded_vertices.append(vertices[i].real())

    local_vertex2unfolded_map = {}
    for j, i in zip(new_unfolded_indices, root.name[2]):
        local_vertex2unfolded_map[i] = j
    vertex2unfolded_map[root.name[0]] = local_vertex2unfolded_map

    parents = [root]
    while len(parents) > 0:
        level_children = []
        for parent in parents:
            level_children += parent.children
            for child in parent.children:
                new_unfolded_indices = []
                for i in child.name[2]:
                    j = len(unfolded2vertex_map)
                    unfolded2vertex_map[j] = i
                    new_unfolded_indices.append(j)
                    unfolded_vertices.append(vertices[i].real())

                local_vertex2unfolded_map = {}
                for j, i in zip(new_unfolded_indices, child.name[2]):
                    local_vertex2unfolded_map[i] = j
                vertex2unfolded_map[child.name[0]] = local_vertex2unfolded_map
        parents = level_children
        print("reindex", len(parents), "children")
    return unfolded2vertex_map, vertex2unfolded_map, unfolded_vertices


def recalculate_vertices_and_normals(root, vertices, unfolded_vertices, unfolded2vertex_map, vertex2unfolded_map):
    """
    setup recursive function
    :param root: root cell
    :param vertices: vertices as vectors over Q(r5)
    :param unfolded_vertices: unfolded vertices as real vectors
    :param unfolded2vertex_map: index mapping between unfolded vertices and original vertices
    :param vertex2unfolded_map: index mapping between original vertices and unfolded vertices for each cell separately
    """
    recalculate_vertices_and_normals_recursive(root, vertices, unfolded_vertices, unfolded_vertices,
                                               vertex2unfolded_map, [])


def apply_rotation(rotation, vertex):
    """
    the application of the rotation is not a simple matrix multiplication,
     since the center of the rotation is not (0,0,0,0)

    from the data of the rotation (three plane points and a rotation matrix) we can compute the projection operator P = u\otimes uT +v\ovtimes vT

    The projection operator P = u\otimes uT +v\ovtimes vT
    """

    p = rotation[0]
    e1 = rotation[1] - p
    u = e1 / np.sqrt(e1.dot(e1))
    e2 = rotation[2] - p
    e2 = e2 - e2.dot(u) * u
    v = np.array(e2 / np.sqrt(e2.dot(e2)))
    u = np.array(u / np.sqrt(u.dot(u)))

    projection = interface.ibpy.Matrix(np.tensordot(u, u, axes=0) + np.tensordot(v, v, axes=0))

    delta = vertex - p
    center = p + projection @ delta

    return rotation[3] @ (vertex - center) + center


def recalculate_vertices_and_normals_recursive(cell, vertices, unfolded_vertices, unfolded2vertex_map,
                                               vertex2unfolded_map, rotations):
    # deal with root
    # each rotation is defined by a rotation matrix and a plane of rotation
    # (similarly as in three dimensions a rotation is defined by a rotation matrix and an axis of rotation)
    # (similarly as in two dimensions a rotation is defined by a rotation matrix and a point of rotation)
    # we store the rotation data as a dictionary, where the key is the cell id, the value is a tuple that contains
    # (three points of the plane and the rotation matrix)

    local_rotations = [rotation for rotation in rotations]  # create local copy

    if cell.parent is not None:
        # deal with non-root cells (generic case)
        parent = cell.parent
        common_indices = cell.name[2] & parent.name[2]
        v2u_local = vertex2unfolded_map[cell.name[0]]

        # transform all vertices with previous rotations (all vertices have to be transformed, since the common
        # indices are stored independently from their parent cell)
        for i in cell.name[2]:
            for rotation in local_rotations:
                j = v2u_local[i]
                unfolded_vertices[j] = apply_rotation(rotation, unfolded_vertices[j])

        # transform normal of child cell with previous rotation
        # the normal is transformed with the transposed rotation
        for rotation in local_rotations:
            cell.name = (cell.name[0], cell.name[1], cell.name[2], rotation[3] @ cell.name[3])

        # align child and parent normals
        child_normal = cell.name[3]
        parent_normal = parent.name[3]

        common_vertices = [unfolded_vertices[v2u_local[i]] for i in common_indices]
        new_rotation = get_rotation(common_vertices, child_normal, parent_normal)

        # transform vertices with the last rotation
        for i in cell.name[2] - common_indices:
            j = v2u_local[i]
            unfolded_vertices[j] = apply_rotation(new_rotation, unfolded_vertices[j])
        # transform normal with the last rotation
        cell.name = (cell.name[0], cell.name[1], cell.name[2], new_rotation[3] @ cell.name[3])
        local_rotations.append(new_rotation)

    else:
        # no rotation is needed, since the root cell need not be unfolded.
        local_rotations = []

    # recursively proceed to the children
    for child in cell.children:
        recalculate_vertices_and_normals_recursive(child, vertices, unfolded_vertices, unfolded2vertex_map,
                                                   vertex2unfolded_map, local_rotations)


def get_rotation(common_vertices, a, b):
    """
    Find rotation that rotates a to b, while leaving e1 and e2 invariant.
    """
    # create ortho-normal base from e1 and e2
    if len(common_vertices)<3:
        raise "Something's wrong with the common face"
    p1 = common_vertices[0]
    p2 = common_vertices[1]
    p3 = common_vertices[2]
    e1 = p2 - p1
    e2 = p3 - p1

    u = e1 / np.sqrt(e1.dot(e1))
    v = (e2 - u.dot(e2) * u)
    v = v / np.sqrt(v.dot(v))
    a = a / np.sqrt(a.dot(a))
    b = b / np.sqrt(b.dot(b))

    # create common face normals for each cell
    na = normal_to(u, v, a)
    na = na / np.sqrt(na.dot(na))
    nb = normal_to(u, v, b)
    nb = nb / np.sqrt(nb.dot(nb))

    # the rotation matrix is quite easy to understand
    # u->u
    # v->v
    # na->nb
    # a->b

    rot = np.tensordot(u, u, axes=0) + np.tensordot(v, v, axes=0) + np.tensordot(nb, na, axes=0) + np.tensordot(b, a,
                                                                                                                axes=0)
    return (p1, p2, p3, interface.ibpy.Matrix(rot))  # return matrix that transforms the vertices


def find_normal_rotation(root):
    """
    find rotation that turns the normal of the root cell into a vector parallel to (0,0,0,1)

    """
    # we need to rotate everything such that the root normal is parallel to (0,0,0,1)
    normal = root.name[3]
    rotations = []
    if abs(normal.x) > EPS:
        a = normal.x
        if abs(normal.y) > EPS:
            b = normal.y
            bprime = np.sqrt(a * a + b * b)
            angle = -np.sign(a) * np.arccos(b / bprime)
            cos = np.cos(angle)
            sin = np.sin(angle)
            rotations.append(interface.ibpy.Matrix([[cos, sin, 0., 0.], [-sin, cos, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]))

            if abs(normal.z) > EPS:
                c = normal.z
                cprime = np.sqrt(bprime * bprime + c * c)
                angle = -np.sign(bprime) * np.arccos(c / cprime)
                cos = np.cos(angle)
                sin = np.sin(angle)
                rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., cos, sin, 0.], [0., -sin, cos, 0.], [0., 0., 0., 1.]]))

                d = normal.w
                dprime = np.sqrt(cprime * cprime + d * d)
                angle = -np.sign(cprime) * np.arccos(d / dprime)
                cos = np.cos(angle)
                sin = np.sin(angle)
                rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., cos, sin], [0., 0., -sin, cos]]))
            else:
                d = normal.w
                dprime = np.sqrt(bprime * bprime + d * d)
                angle = -np.sign(bprime) * np.arccos(d / dprime)
                cos = np.cos(angle)
                sin = np.sin(angle)
                rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., cos, 0., sin], [0., 0., 1., 0.], [0., -sin, 0., cos]]))
        elif abs(normal.z) > EPS:
            c = normal.z
            cprime = np.sqrt(a * a + c * c)
            angle = -np.sign(a) * np.arccos(c / cprime)
            cos = np.cos(angle)
            sin = np.sin(angle)
            rotations.append(interface.ibpy.Matrix([[cos, 0., sin, 0.], [0., 1., 0., 0.], [-sin, 0., cos, 0.], [0., 0., 0., 1.]]))

            d = normal.w
            dprime = np.sqrt(cprime * cprime + d * d)
            angle = -np.sign(cprime) * np.arccos(d / dprime)
            cos = np.cos(angle)
            sin = np.sin(angle)
            rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., cos, sin], [0., 0., -sin, cos]]))
        else:
            d = normal.w
            dprime = np.sqrt(a * a + d * d)
            angle = -np.sign(a) * np.arccos(d / dprime)
            cos = np.cos(angle)
            sin = np.sin(angle)
            rotations.append(interface.ibpy.Matrix([[cos, 0., 0., sin], [0., 1., 0., 0.], [0., 0., 1., 0.], [-sin, 0., 0., cos]]))
    elif abs(normal.y) > EPS:
        b = normal.y
        if abs(normal.z) > EPS:
            c = normal.z
            cprime = np.sqrt(b * b + c * c)
            angle = -np.sign(b) * np.arccos(c / cprime)
            cos = np.cos(angle)
            sin = np.sin(angle)
            rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., cos, sin, 0.], [0., -sin, cos, 0.], [0., 0., 0., 1.]]))

            d = normal.w
            dprime = np.sqrt(cprime * cprime + d * d)
            angle = -np.sign(cprime) * np.arccos(d / dprime)
            cos = np.cos(angle)
            sin = np.sin(angle)
            rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., cos, sin], [0., 0., -sin, cos]]))
        else:
            d = normal.w
            dprime = d / np.sqrt(b * b + d * d)
            angle = -np.sign(b) * np.arccos(dprime)
            cos = np.cos(angle)
            sin = np.sin(angle)
            rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., cos, 0., sin], [0., 0., 1., 0.], [0., -sin, 0., cos]]))
    elif abs(normal.z) > EPS:
        c = normal.z
        d = normal.w
        dprime = d / np.sqrt(c * c + d * d)
        angle = -np.sign(c) * np.arccos(dprime)
        cos = np.cos(angle)
        sin = np.sin(angle)
        rotations.append(interface.ibpy.Matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., cos, sin], [0., 0., -sin, cos]]))

    full_rotation = interface.ibpy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    while len(rotations) > 0:
        full_rotation = full_rotation @ rotations.pop()
    return full_rotation


def rotateIntoWzeroSpace(root, unfolded_vertices, vertex2unfolded_map):
    """
    all unfolded vertices have to be rotated by a global rotation
    """
    rotation = find_normal_rotation(root)

    # rotate vertices around the center of the root cell
    root_indices = root.name[2]
    v2u_local = vertex2unfolded_map[root.name[0]]
    root_vertices = [unfolded_vertices[v2u_local[i]] for i in root_indices]
    center_of_root = sum(root_vertices, interface.ibpy.Vector([0, 0, 0, 0])) / len(root_vertices)

    for i, v in enumerate(unfolded_vertices):
        unfolded_vertices[i] = rotation @ (v - center_of_root) + center_of_root

    # translate center of mass to origin
    center = sum(unfolded_vertices, interface.ibpy.Vector([0, 0, 0, 0])) / len(unfolded_vertices)
    for i, v in enumerate(unfolded_vertices):
        unfolded_vertices[i] = v - center

    pass


def unfold(vertices, root):
    # each vertex can have various unfolded images, depending on the cell that is under consideration
    # the map is bijective when restricted to a single cell, therefore, we store the corresponding map for each cell
    # the inverse map is injective, each unfolded vertex index belongs has a unique vertex index

    unfolded2vertex_map, vertex2unfolded_map, unfolded_vertices = indexing_unfolded_vertices(root, vertices)
    for key, val in vertex2unfolded_map.items():
        if 1000 in val.values():
            print(key, val)

    recalculate_vertices_and_normals(root, vertices, unfolded_vertices, unfolded2vertex_map, vertex2unfolded_map)
    # if abs(root.name[3].w) > EPS:
    rotateIntoWzeroSpace(root, unfolded_vertices, vertex2unfolded_map)
    # shiftTheCenterOfMassToZero()
    return vertex2unfolded_map, unfolded_vertices


def lg(x):
    return np.log(x) / np.log(10)


def edge_length(verts):
    lengths = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            lengths.append((verts[i]-verts[j]).length)
    return min(lengths)


def index_of(v, vertices):
    """ find index of closest vertex in vertices """
    dist = np.inf
    min_idx = -1
    for idx, vertex in enumerate(vertices):
        if (v-vertex).length < dist:
            dist = (v-vertex).length
            min_idx = idx
    return min_idx


def normalize_to_unity(vertices):
    l = to_vector(vertices[0]).length
    return [to_vector(vert)/l for vert in vertices]



class video_CoxH4(Scene):
    def __init__(self):
        self.t0 = None
        self.construction_counter = 0
        self.old = None
        self.sub_scenes = OrderedDict([
            ('normals', {'duration':15}),
            ('reconstruction', {'duration':30}),
            ('algebra', {'duration':30}),
            ('euler', {'duration':30}),
            ('generators', {'duration':7}),
            ('r5', {'duration':1}),
            ('unfolding_with_normals', {'duration':20}),
            ('thumbnail', {'duration':60}),
            ('logo', {'duration':60}),
            ('cd_wizardry', {'duration':20}),
            ('rotating_trunc_icosidodeca', {'duration':60}),
            ('trinity2', {'duration':20}),
            ('trinity', {'duration':20}),
            ('stereographic', {'duration':35}),
            ('edge_highlighting', {'duration':12}),
            ('morphing4', {'duration': 20}),
            ('morphing3', {'duration': 40}),
            ('morphing2', {'duration': 55}),
            ('morphing', {'duration': 55}),
            ('coxeter_h3', {'duration': 10}),
            ('mirrors', {'duration': 40}),
            ('polygon_geometry', {'duration': 20}),
            ('triangles_manual', {'duration': 1}),
            ('triangles', {'duration': 20}),
            ('archimedian', {'duration': 50}),
            ('platonic', {'duration': 30}),
            ('truncated_icosidodecahedron_preparation', {'duration': 1}),
            ('truncated_icosidodecahedron', {'duration': 3200//60}),
            ('applet_tester', {'duration': 60}),
            ('v600_unfolder', {'duration': 62}),
            ('v14400_construction', {'duration': 27}),
            ('v7200d_unfolding', {'duration': 200}),
            ('v7200c_unfolding', {'duration': 270}),
            ('v7200b_unfolding', {'duration': 270}),
            ('v7200a_unfolding', {'duration': 150}),
            ('v14400_unfolding', {'duration': 300}),
            ('v14400_unfolding_rotation', {'duration': 60}),
            ('v14400', {'duration': 135}),
            ('v14400b', {'duration': 260}),
            ('v14400c', {'duration': 210}),
            ('v14400d', {'duration': 195}),
            ('corner', {'duration': 62}),
        ])
        super().__init__(light_energy=2, transparent=False)

    def normals(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False)


        camera_location = [0, -10,2.5]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        camera_empty = EmptyCube(location=Vector([0,0,0]))
        ibpy.set_camera_view_to(camera_empty)

        locations = [Vector([np.cos(tau/5*i),np.sin(tau/5*i),0]) for i in range(5)]
        polygon = BObject(mesh=ibpy.create_mesh(locations,faces=[list(range(5))]),color="trunc_octa")
        t0 =0.5+polygon.grow(begin_time=t0,transition_time=1)

        u = locations[0]
        v=locations[1]
        n = u.cross(v)
        n.normalize()

        directions =[u,v,n]
        colors = ["example","example","trunc_icosidodeca"]
        for i,(color,dir) in enumerate(zip(colors,directions)):
            arrow = PArrow(start=Vector(),end=dir,thickness=0.5,color =color,name="Arrow"+str(i))
            t0 = 0.5+arrow.grow(begin_time=t0,transition_time=1)

        lines = [
            Text(r"\text{cell normal:}\,\, n_i=\varepsilon_{ijkl} u^j v^k w^l "),
            Text(r"\text{face normal:}\,\, \nu_i=\varepsilon_{ijkl} n^j u^k v^l")
        ]

        for i,line in enumerate(lines):
            line.move_to(target_location=[-1,0,-0.5-0.5*i],begin_time=t0,transition_time=0)
            line.rotate(rotation_euler=to_vector(ibpy.camera_alignment_euler(camera_empty,camera_location))+Vector([-pi/2,0,0]),begin_time=t0,transition_time=0)
            t0 = 0.5+line.write(begin_time=t0,transition_time=0.5)

        self.t0 = t0

    def reconstruction(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False)


        camera_location = [0, 0, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, -1], root_size=120)
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        count = 0
        children = []
        selected_cells=[]
        selected_obj =[]
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            if len(verts) == 120:
                color = "trunc_icosidodeca"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism6"

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx),
                              color=color,emission=0.5)
            children.append(cellobj)
            if count<4:
                selected_cells.append(cell)
                selected_obj.append(cellobj)
                count+=1
            else:
                break

        # root_vertex
        root_index = (set(selected_cells[0])&set(selected_cells[1])&set(selected_cells[2])&set(selected_cells[3])).pop()

        spheres = []
        drawn = []
        locations = []

        for i,cell in zip(order,selected_cells):
            for idx in cell:
                if not idx in drawn:
                    drawn.append(idx)
                    location = to_vector(unfolded_vertices[vertex2unfolded_map[i][idx]][0:3])
                    if idx == root_index:
                        spheres.append(Sphere(0.2,color="example",resolution=3,smooth=2,mesh_type="ico",location=location))
                        root_location = location
                    else:
                        spheres.append(Sphere(0.1,color="red",resolution=1,smooth=False,mesh_type="ico",location=location))
                        locations.append(location)
                t0 = spheres[-1].grow(begin_time=t0,transition_time=0.05)

        # nearest neigbors
        edges = []
        count=0
        for location in locations:
            dist=(location-root_location).length
            print(dist)
            if dist<2.9:
                edges.append(Cylinder.from_start_to_end(start=root_location,end=location,thickness=0.5,color="example"))
                edges[-1].grow(begin_time=t0+count*0.1,transition_time=0.5)
                count+=1
        t0 =1 + t0+count*0.1

        # bordering faces
        count = 0
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell) and root_index in face:
                    selected_faces.append([local_map[i] for i in face])

            if len(verts) == 120:
                color = "trunc_icosidodeca"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism6"


            if count < 4:
                cell_center = sum(verts,Vector())/len(verts)
                for face in selected_faces:
                    face_vertices = [verts[i] for i in face]
                    face_center = sum(face_vertices,Vector())/len(face_vertices)
                    vs =[vertex-face_center for vertex in face_vertices]
                    normal = (vs[2]-vs[0]).cross(vs[1]-vs[0])
                    normal.normalize()
                    if normal.dot(cell_center-face_center)<0:
                        normal*=-1
                    root_face = BObject(mesh=interface.ibpy.create_mesh(vertices=[v+0.1*normal for v in vs], faces=[list(range(len(face)))]),
                                        color=color,location=face_center,emission=0.5)
                    t0 = root_face.grow(begin_time=t0, transition_time=0.5,alpha=0.75)
                count += 1
            else:
                break

        # bordering volumes

        for cell in selected_obj:
            t0 = 0.5 + cell.appear(begin_time=t0,transition_time=1,alpha=0.5)

        camera_empty = EmptyCube(location=root_location+Vector([0,0,4]))
        circle = Circle2(center=root_location,radius=55,normal=[0,0,1],thickness=0)
        circle.appear(begin_time=0,transition_time=0,alpha=0)
        circle.rescale(rescale=0.4,begin_time=0.5,transition_time=0)
        circle.rescale(rescale=2.5,begin_time=1,transition_time=t0)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_follow(circle)
        ibpy.camera_follow(circle,initial_value=0,final_value=1,begin_time=0,transition_time=t0)



        self.t0 = t0

    def algebra(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        filename = os.path.join(LOC_FILE_DIR, "show_qr5.py")
        cp = CodeParser(filename, recreate=False)

        display = CodeDisplay(cp, location=Vector([6, 0, 0]), number_of_lines=35, flat=True)
        t0 = display.appear(begin_time=t0)

        t0 = 0.5 + cp.write(display, class_index=0, begin_time=t0, transition_time=25, indent=0.5)

        self.t0 = t0

    def euler(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        col = -10
        row = 1.5
        trico_count = Text(r"120",color="trunc_icosidodeca",aligned="right",location=[col,0,row])
        t0 = 0.5 + trico_count.write(begin_time=t0,transition_time=0.1)

        vertices,faces=get_solid_data("TRUNC_ICOSIDODECA")
        trico =BObject(mesh=ibpy.create_mesh(vertices,faces=faces),color="trunc_icosidodeca",scale=0.137,location=[col+1,0,row])
        t0 = 0.5 + trico.grow(begin_time=t0,transition_time=1)

        trico_vertices = Text(r"120",color="trunc_icosidodeca",aligned="right",location=[col+3,0,row])
        t0 = 0.5 + trico_vertices.write(begin_time=t0,transition_time=0.1)

        row -= 1.5
        trocta_count = Text(r"600", color="trunc_octa", aligned="right", location=[col, 0, row])
        t0 = 0.5 + trocta_count.write(begin_time=t0, transition_time=0.1)

        vertices, faces = get_solid_data("TRUNC_OCTA")
        trocta = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), color="trunc_octa", scale=0.25,
                        location=[col+1, 0, row])
        t0 = 0.5 + trocta.grow(begin_time=t0, transition_time=1)

        trocta_vertices = Text(r"24", color="trunc_octa", aligned="right", location=[col+3, 0, row])
        t0 = 0.5 + trocta_vertices.write(begin_time=t0, transition_time=0.1)

        row -= 1.5
        prism10_count = Text(r"720", color="prism10", aligned="right", location=[col, 0, row])
        t0 = 0.5 + prism10_count.write(begin_time=t0, transition_time=0.1)

        vertices, faces = get_solid_data("PRISM10")
        prism10 = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), color="prism10", scale=0.4, rotation_euler=[pi/3,0,0],
                         location=[col + 1, 0, row])
        t0 = 0.5 + prism10.grow(begin_time=t0, transition_time=1)

        prism10_vertices = Text(r"20", color="prism10", aligned="right", location=[col + 3, 0, row])
        t0 = 0.5 + prism10_vertices.write(begin_time=t0, transition_time=0.1)

        row -= 1.5
        prism6_count = Text(r"1200", color="prism6", aligned="right", location=[col, 0, row])
        t0 = 0.5 + prism6_count.write(begin_time=t0, transition_time=0.1)

        vertices, faces = get_solid_data("PRISM6")
        prism6 = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), color="prism6", scale=0.5,
                          rotation_euler=[pi / 3, 0, 0],
                          location=[col + 1, 0, row])
        t0 = 0.5 + prism6.grow(begin_time=t0, transition_time=1)

        prism6_vertices = Text(r"12", color="prism6", aligned="right", location=[col + 3, 0, row])
        t0 = 0.5 + prism6_vertices.write(begin_time=t0, transition_time=0.1)

        row-=1
        line = Cylinder.from_start_to_end(start=[col-0.75,0,row],end=[col+2,0,row],thickness=0.5,color="text")
        t0=0.5+line.grow(begin_time=t0, transition_time=0.1)

        row-=0.5
        sum = Text(r"2640\,\, \text{cells}",color="text",aligned="left",location=[col-0.81,0,row])
        t0 = 0.5 + sum.write(begin_time=t0,transition_time=0.1)

        euler = Text(r"14400\,\,\text{vertices}-28800\,\,\text{edges}+17040\,\,\text{faces}-2640\,\,\text{cells}=0",color="example",
                     outline_color="joker",aligned="center",text_size="Large",location=[0,0,-5.6],keep_outline=True)
        t0 =0.5 + euler.write(from_letter=0,to_letter=13,begin_time=t0,transition_time=1)

        col = 10.5
        row = 0

        hexa_count = Text(r"4800", color="trunc_octa", aligned="right", location=[col, 0, row])
        t0 = 0.5 + hexa_count.write(begin_time=t0, transition_time=0.1)

        radius = 0.7
        n = 6
        vertices, faces = [Vector([radius*np.cos(tau/n*i),0,radius*np.sin(tau/n*i)]) for i in range(n)],[list(range(n))]
        hexa = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), color="trunc_octa", scale=1,
                        location=[col - 2, 0, row])
        t0 = 0.5 +hexa.grow(begin_time=t0, transition_time=1)

        row -= 1.5
        deca_count = Text(r"1440", color="trunc_icosidodeca", aligned="right", location=[col, 0, row])
        t0 = 0.5 + deca_count.write(begin_time=t0, transition_time=0.1)

        radius = 0.7
        n = 10
        vertices, faces = [Vector([radius * np.cos(tau / n * i), 0, radius * np.sin(tau / n * i)]) for i in range(n)], [
            list(range(n))]
        deca = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), color="trunc_icosidodeca", scale=1,
                        location=[col - 2, 0, row])
        t0 = 0.5 + deca.grow(begin_time=t0, transition_time=1)

        row -= 1.5
        square_count = Text(r"10800", color="gray_5", aligned="right", location=[col, 0, row])
        t0 = 0.5 + square_count.write(begin_time=t0, transition_time=0.1)

        radius = 0.7
        n = 4
        vertices, faces = [Vector([radius * np.cos(tau / n * i), 0, radius * np.sin(tau / n * i)]) for i in range(n)], [
            list(range(n))]
        square = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), color="gray_5", scale=1,
                       location=[col - 2, 0, row])
        t0 = 0.5 + square.grow(begin_time=t0, transition_time=1)

        row -= 1
        line = Cylinder.from_start_to_end(start=[col - 2.7, 0, row], end=[col + 0.05, 0, row], thickness=0.5,
                                          color="text")
        t0 = 0.5 + line.grow(begin_time=t0, transition_time=0.1)

        row -= 0.5
        sum = Text(r"17040\,\, \text{faces}", color="text", aligned="left", location=[col - 2.4, 0, row])
        t0 = 0.5 + sum.write(begin_time=t0, transition_time=0.1)

        t0 = 0.5 + euler.write(from_letter=13,to_letter=24,begin_time=t0,transition_time=1)
        t0 = 0.5 + euler.write(from_letter=24,to_letter=46,begin_time=t0,transition_time=1)
        t0 = 0.5 + euler.write(from_letter=46,to_letter=47,begin_time=t0,transition_time=0.1)

        self.t0 = t0

    def generators(self):

        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)
        group = CoxH4(path="../mathematics/geometry/data")
        prefixes = ["a=","b=","c=","d="]
        generators = [group.generators[i] for i in [0,1,3,2]]
        for i,(prefix, gen) in enumerate(zip(prefixes,generators)):
            matrix = Text(gen.to_latex(prefix=prefix),color="green",location=[-11,0,4-2*i])
            t0 = 0.5 + matrix.write(begin_time=t0,transition_time=1)
        self.t0 = t0

    def r5(self):
        t0 =0
        r5 = Text(r"\sqrt{5}")
        t0 = r5.write(begin_time=t0,transition_time=0.5)
        self.t0 = t0

    def unfolding_with_normals(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -40, 15]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        group = CoxH3()
        signature = COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]
        src_faces = group.get_faces(signature=signature)

        src_vertices = group.get_real_point_cloud(signature=signature)
        src_radius = src_vertices[0].length
        radius = 1
        locations = [radius * Vector([np.cos(tau / 10 * i), np.sin(tau / 10 * i), 0]) for i in range(5)]

        for face in src_faces:
            if len(face) == 10:
                root_face = face
                break

        s, r, t = compute_similarity_transform(*[src_vertices[i] for i in root_face[0:3]], *locations[0:3])
        t.z = 0  # keep center at the origin
        vertices = apply_similarity_to_vertices(src_vertices, s, r, t)

        face_appearance_order = [8, 21, 22, 38, 31, 32, 7, 6, 45, 54, 26,
                                 33, 39, 10, 46, 17, 27, 34, 20, 43, 35,
                                 44, 40, 59, 11, 16, 53,
                                 57, 4, 61, 18, 55, 9, 37, 15, 58,
                                 56, 5, 19, 47, 49, 60, 52, 51, 3, 50, 14, 36, 30, 28, 48,
                                 2, 42, 25, 24, 12, 13, 41, 29, 23, 1, 0]

        trunc_icosidodeca = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            faces=src_faces
        ), location=[0, 0, 0], scale=2
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "trunc_icosidodeca"],
                                        edge_material="example", vertex_material="red", edge_radius=0.025,
                                        face_types=[4, 6, 10],
                                        vertex_radius=0.051, sorting=False, face_appearance_order=face_appearance_order,
                                        show_normals=True,normal_thickness=0.05,
                                        normal_material="important"
                                        )

        trunc_icosidodeca.add_mesh_modifier(type="NODES", node_modifier=modifier)
        trunc_icosidodeca.appear(begin_time=t0, transition_time=0)
        t0 = 0.5 + modifier.grow(begin_time=t0, transition_time=5, max_faces=120)
        camera_empty.move(direction=[-2,0,6],begin_time=t0+1.5,transition_time=3)
        t0 = 0.5 + modifier.unfold(to_value=30,begin_time=t0,transition_time=5)
        self.t0 = t0

    def thumbnail(self):
        t0 = 0
        ibpy.set_volume_scatter_background(density=0.03)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=0.6, type='BLOOM', size=1)


        # projections

        light_location = Vector([8, 0, -6])
        light2_location = Vector([-8, -0, -6])

        screen_pos = Vector([7, 9, 2])
        screen = Plane(u=[-8, 8], v=[-8, 8], normal=screen_pos-light_location, name="TitlePlane")
        screen.move_to(target_location=screen_pos, begin_time=0, transition_time=0)
        screen.grow(begin_time=0, transition_time=0)

        screen2_pos = Vector([-7, 9, 2])
        screen2 = Plane(u=[-8, 8], v=[-8, 8], normal=screen2_pos-light2_location, name="TitlePlane")
        screen2.move_to(target_location=screen2_pos, begin_time=0, transition_time=0)
        screen2.grow(begin_time=0, transition_time=0)

        logo_light = SpotLight(location=light_location, color="movie", src="rotating_net.mp4",
                               target=screen, name="LogoLight", spot_size=pi / 180 * 68,
                               coordinates="UV", duration=60, energy=25000)
        logo_light.change_power(from_value=0, to_value=25000, begin_time=0, transition_time=0)

        logo_light2 = SpotLight(location=light2_location, color="movie", src="rotating_trunc_icosidodeca.mp4",
                                target=screen2, name="LogoLight", spot_size=pi / 180 * 75,
                                coordinates="UV", duration=60, energy=25000)
        logo_light2.change_power(from_value=0, to_value=25000, begin_time=0, transition_time=0)

        # text
        title_text = "GIANTS"
        emission =2
        offsets = [0,-0.06,0,0,0,0]
        for i,(l,offset) in enumerate(zip(title_text,offsets)):
            title = Text(r"\text{"+l+r"}",text_size="Huge",color="plastic_example",outline_color="joker",location=[-0.68+offset,0,3.8-1.3*i],emission=emission,
                         outline_emission=1,aligned="center",keep_outline=True)
            title.write(begin_time=t0,transition_time=0)

        in_text = Text(r"\text{in}",text_size="Large",color="plastic_example",emission=emission,location=[-0.68,0,-4.5])
        in_text.write(begin_time=t0,transition_time=0)

        threeD = Text(r"\text{3D}",location=[-6,0,-5.4],text_size="Huge",color="plastic_example",emission=emission,outline_emission=1,aligned="center",
                      keep_outline=True,outline_color="joker")
        fourD = Text(r"\text{4D}",location=[5,0,-5.4],text_size="Huge",color="plastic_example",emission=emission,outline_emission=1,aligned="center",
                     keep_outline=True,outline_color="joker")

        threeD.write(begin_time=t0,transition_time=0)
        fourD.write(begin_time=t0,transition_time=0)
        self.t0 = t0

    def logo(self):
        t0 = 0
        ibpy.set_volume_scatter_background(density=0.03)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=64,
                               motion_blur=True)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        light_empty = EmptyCube(location = Vector([-9,20,6.3]))
        camera_location = [0, -25, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        create_glow_composition(threshold=0.6, type='BLOOM', size=1)

        # logo
        vertices,faces = get_solid_data("TRUNC_ICOSIDODECA")
        vertices2,faces2 = get_solid_data("TRUNC_OCTA")
        vertices3,faces3 = get_solid_data("RHOMBICUBOCTA")

        vertices=normalize_to_unity(vertices)
        vertices2=normalize_to_unity(vertices2)
        vertices3=normalize_to_unity(vertices3)

        mod1 = CustomUnfoldModifier(face_materials=["trunc_icosidodeca", "trunc_icosidodeca", "trunc_icosidodeca"],
                                    edge_material="example", vertex_material="red", edge_radius=0.01,
                                        face_types=[4, 6, 10],
                                        vertex_radius=0.02, sorting=False,max_faces=120,emission=.5)

        mod2 = CustomUnfoldModifier(face_materials=["trunc_octa", "trunc_octa"],
                                    edge_material="example", vertex_material="red", edge_radius=0.0251,
                                    face_types=[4, 6],
                                    vertex_radius=0.051, sorting=False,max_faces=30,emission=.5)

        mod3 = CustomUnfoldModifier(face_materials=["rhombicubocta","rhombicubocta"],
                                    edge_material="example", vertex_material="red", edge_radius=0.0251,
                                    face_types=[3,4],
                                    vertex_radius=0.051, sorting=False,max_faces=30,emission=.5)

        kwargs_red = {"mesh": ibpy.create_mesh(vertices,faces=faces),"color":"trunc_icosidodeca","geo_node_modifier":mod1 }
        kwargs_green={"mesh":ibpy.create_mesh(vertices3,faces=faces3),"color":"rhombicubocta","geo_node_modifier":mod3 }
        kwargs_blue={"mesh":ibpy.create_mesh(vertices2,faces=faces2),"color":"trunc_octa","geo_node_modifier":mod2 }
        sphere = Sphere(r=0.5, roughness=0, metallic=0,emission=0,color="text")

        logo = LogoFromInstances(instance=BObject, rotation_euler=[pi / 2, 0, 0],
                                 scale=[10] * 3, location=[-9, 20, -9],
                                 details=15,kwargs_blue=kwargs_blue,kwargs_green=kwargs_green,kwargs_red=kwargs_red,special_instances={sphere:("red",0)})



        t0 = 0.5 + logo.grow(begin_time=t0, transition_time=2)
        for instance in logo.get_instances():
            instance.rotate(rotation_euler=[0,tau,0],begin_time=0,transition_time=30)

        # projections

        screen = Plane(u=[-8, 8], v=[-9, 9], normal=[0, 1, 1], name="TitlePlane")
        screen.move_to(target_location=[10, 9, 2], begin_time=0, transition_time=0)
        screen.grow(begin_time=0, transition_time=0)

        logo_light = SpotLight(location=Vector([8, 0, -6]), color="movie", src="rotating_net.mp4",
                               target=screen, name="LogoLight",spot_size=pi/180*70,
                               coordinates="UV", duration=60, energy=25000)
        logo_light.change_power(from_value=0,to_value=25000,begin_time=0,transition_time=2)


        logo_light2 = SpotLight(location=Vector([-3.5, -8, -1]), color="movie", src="rotating_trunc_icosidodeca.mp4",
                                target=light_empty, name="LogoLight",spot_size=pi/180*28,
                                coordinates="UV", duration=60, energy=25000,scale=[1.87,1,1])
        logo_light2.change_power(from_value=0,to_value=25000,begin_time=0,transition_time=2)

        # text
        t0 = 2
        title_text = "GIANTS"
        offsets = [0,-0.06,0,0,0,0]
        for i,(l,offset) in enumerate(zip(title_text,offsets)):
            title = Text(r"\text{"+l+r"}",text_size="Huge",color="plastic_example",outline_color="joker",location=[1.5+offset,0,1.8-1.3*i],emission=0.5,
                         outline_emission=1,aligned="center",keep_outline=True)
            t0  = title.write(begin_time=t0,transition_time=0.1)

        t0 +=0.5

        in_text = Text(r"\text{in}",text_size="Large",color="plastic_example",emission=0.5,location=[1.5,0,-6])
        t0 = 0.5 + in_text.write(begin_time=t0,transition_time=0.5)

        threeD = Text(r"\text{3D}",location=[-4.5,0,-2],text_size="Huge",color="plastic_example",emission=0.5,outline_emission=1,aligned="center",
                      keep_outline=True,outline_color="joker")
        fourD = Text(r"\text{4D}",location=[9,0,-5.4],text_size="Huge",color="plastic_example",emission=0.5,outline_emission=1,aligned="center",
                     keep_outline=True,outline_color="joker")

        t0 = 0.5 + threeD.write(begin_time=t0,transition_time=0.5)
        t0 = 0.5 + fourD.write(begin_time=t0,transition_time=0.5)
        self.t0 = t0

    def cd_wizardry(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30,0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        dia = DynkinDiagram(dim=4,labels=["3","3","5"],location = [0,0,4])
        t0 = dia.appear(begin_time=t0,transition_time=1)
        t0 = 0.5 + dia.appear_customized(rings=[0, 1, 2, 3], begin_time=t0, transition_time=0.5)

        copy = dia.move_copy(direction=[-8.5, 0, -4], begin_time=t0, transition_time=1)
        copy.appear_customized(rings=[0, 1, 2, 3], begin_time=t0, transition_time=0)
        t0 = 0.5 + copy.rescale(rescale=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5  + copy.disappear_customized(nodes=[0],labels=[0],rings=[0],begin_time=t0,transition_time=0.5)

        copy2 = dia.move_copy(direction=[-3,0,-4],begin_time=t0,transition_time=1)
        copy2.appear_customized(rings=[0, 1, 2, 3], begin_time=t0, transition_time=0)
        t0 = 0.5 + copy2.rescale(rescale=0.5, begin_time=t0, transition_time=1)
        t0 =0.5 + copy2.disappear_customized(nodes=[1],labels=[0,1],rings=[1],begin_time=t0,transition_time=0.5)

        copy3 = dia.move_copy(direction=[3, 0, -4], begin_time=t0, transition_time=1)
        copy3.appear_customized(rings=[0, 1, 2, 3], begin_time=t0, transition_time=0)
        t0 = 0.5 + copy3.rescale(rescale=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + copy3.disappear_customized(nodes=[2],labels=[1,2],rings=[2], begin_time=t0, transition_time=1)

        copy4 = dia.move_copy(direction=[8.5, 0, -4], begin_time=t0, transition_time=1)
        copy4.appear_customized(rings=[0, 1, 2, 3], begin_time=t0, transition_time=0)
        t0 = 0.5 + copy4.rescale(rescale=0.5, begin_time=t0, transition_time=1)
        t0 = 0.5 + copy4.disappear_customized(nodes=[3], labels=[2], rings=[3], begin_time=t0, transition_time=1)

        # truncated icosidodecahedron
        vertices,faces = get_solid_data("TRUNC_ICOSIDODECA")
        tr_ico = BObject(mesh=ibpy.create_mesh(vertices,faces=faces),location = [-8,0,-3],rotation_euler=[0,0,0],scale=0.5,
                         color="trunc_icosidodeca")
        modifier = CustomUnfoldModifier(face_materials=["trunc_icosidodeca", "trunc_icosidodeca", "trunc_icosidodeca"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,
                                        face_types=[4, 6, 10],
                                        vertex_radius=0.1, sorting=False
                                        )
        tr_ico.add_mesh_modifier(type="NODES",node_modifier=modifier)
        tr_ico.appear(begin_time=t0,transition_time=0)
        rot0=t0
        tr_ico.rotate(rotation_euler=[0,0,tau],begin_time=rot0,transition_time=10)
        t0 = 0.5 + modifier.grow(begin_time=t0, transition_time=1, max_faces=120)

        # ten prism
        vertices, faces = get_solid_data("PRISM10")
        prism10 = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), location=[-3, 0, -3], rotation_euler=[pi/4, 0, 0],
                         scale=1,
                         color="prism10")
        modifier = CustomUnfoldModifier(face_materials=["prism10", "prism10"],
                                        edge_material="example", vertex_material="red", edge_radius=0.025,
                                        face_types=[4, 10],
                                        vertex_radius=0.051, sorting=False
                                        )
        prism10.add_mesh_modifier(type="NODES", node_modifier=modifier)
        prism10.appear(begin_time=t0, transition_time=0)
        prism10.rotate(rotation_euler=[tau+pi/4,0,0],begin_time=rot0,transition_time=10)
        t0 = 0.5 + modifier.grow(begin_time=t0, transition_time=1, max_faces=20)

        # six prism
        vertices, faces = get_solid_data("PRISM6")
        prism6 = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), location=[3, 0, -3],
                          rotation_euler=[pi / 4, 0, 0],
                          scale=1,
                          color="prism6")
        modifier = CustomUnfoldModifier(face_materials=["prism6", "prism6"],
                                        edge_material="example", vertex_material="red", edge_radius=0.025,
                                        face_types=[4, 6],
                                        vertex_radius=0.051, sorting=False
                                        )
        prism6.add_mesh_modifier(type="NODES", node_modifier=modifier)
        prism6.appear(begin_time=t0, transition_time=0)
        prism6.rotate(rotation_euler=[tau+pi/4,0,0],begin_time=rot0,transition_time=10)
        t0 = 0.5 + modifier.grow(begin_time=t0, transition_time=1, max_faces=12)

        # truncated octahedron
        vertices, faces = get_solid_data("TRUNC_OCTA")
        trunc_octa = BObject(mesh=ibpy.create_mesh(vertices, faces=faces), location=[8, 0, -3],
                         rotation_euler=[pi / 4, 0, 0],
                         scale=1,
                         color="trunc_octa")
        modifier = CustomUnfoldModifier(face_materials=["trunc_octa", "trunc_octa"],
                                        edge_material="example", vertex_material="red", edge_radius=0.025,
                                        face_types=[4, 6],
                                        vertex_radius=0.051, sorting=False
                                        )
        trunc_octa.add_mesh_modifier(type="NODES", node_modifier=modifier)
        trunc_octa.appear(begin_time=t0, transition_time=0)
        trunc_octa.rotate(rotation_euler=[0,0,-tau],begin_time=rot0,transition_time=10)
        t0 = 0.5 + modifier.grow(begin_time=t0, transition_time=1, max_faces=24)


        self.t0=t0

    def rotating_trunc_icosidodeca(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 10]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        group = CoxH3()
        signature = COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]
        src_faces = group.get_faces(signature=signature)

        src_vertices = group.get_real_point_cloud(signature=signature)
        src_radius = src_vertices[0].length
        radius = 1
        locations = [radius * Vector([np.cos(tau / 10 * i), np.sin(tau / 10 * i), 0]) for i in range(5)]

        for face in src_faces:
            if len(face)==10:
                root_face = face
                break

        s, r, t = compute_similarity_transform(*[src_vertices[i] for i in root_face[0:3]], *locations[0:3])
        t.z = 0 # keep center at the origin
        vertices = apply_similarity_to_vertices(src_vertices, s, r, t)

        face_appearance_order = [8,21,22,38,31,32,7,6,45,54,26,
                                 33,39,10,46,17,27,34,20,43,35,
                                 44,40,59,11,16,53,
                                 57,4,61,18,55,9,37,15,58,
                                 56,5,19,47,49,60,52,51,3,50,14,36,30,28,48,
                                 2,42,25,24,12,13,41,29,23,1,0]

        trunc_icosidodeca = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            faces=src_faces
        ),location=[0,0,0],scale=2
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5","trunc_octa","trunc_icosidodeca"],
                                        edge_material="example", vertex_material="red", edge_radius=0.025,face_types=[4,6,10],
                                        vertex_radius=0.051, sorting=False, face_appearance_order=face_appearance_order
                                        )

        trunc_icosidodeca.add_mesh_modifier(type="NODES",node_modifier=modifier)
        trunc_icosidodeca.appear(begin_time=t0,transition_time=0)
        modifier.grow(begin_time=t0,transition_time=10,max_faces=120)
        trunc_icosidodeca.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=60)

        self.t0=60

    def trinity2(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 10]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        group = CoxH3()
        signature = COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]
        src_faces = group.get_faces(signature=signature)
        src_edges = group.get_edges(signature=signature)
        src_vertices = group.get_real_point_cloud(signature=signature)
        src_radius = src_vertices[0].length
        radius = 1
        locations = [radius * Vector([np.cos(tau / 10 * i), np.sin(tau / 10 * i), 0]) for i in range(5)]

        for face in src_faces:
            if len(face)==10:
                root_face = face
                break

        s, r, t = compute_similarity_transform(*[src_vertices[i] for i in root_face[0:3]], *locations[0:3])
        t.z = 0 # keep center at the origin
        vertices = apply_similarity_to_vertices(src_vertices, s, r, t)

        face_appearance_order = [8,21,22,38,31,32,7,6,45,54,26,
                                 33,39,10,46,17,27,34,20,43,35,
                                 44,40,59,11,16,53,
                                 57,4,61,18,55,9,37,15,58,
                                 56,5,19,47,49,60,52,51,3,50,14,36,30,28,48,
                                 2,42,25,24,12,13,41,29,23,1,0]

        trunc_icosidodca = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            faces=src_faces
        ),location=[-8,0,1.5]
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5","trunc_octa","trunc_icosidodeca"],
                                        edge_material="example", vertex_material="red", edge_radius=0.025,face_types=[4,6,10],
                                        vertex_radius=0.051, sorting=False, face_appearance_order=face_appearance_order
                                        )

        trunc_icosidodca.add_mesh_modifier(type="NODES",node_modifier=modifier)
        trunc_icosidodca.appear(begin_time=t0,transition_time=0)

        trunc_icosidodeca2 = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            faces=src_faces
        ), location=[-2.6, 0.6, -0.6]
        )
        modifier2 = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "trunc_icosidodeca"],
                                        edge_material="example", vertex_material="red", edge_radius=0.025,
                                        face_types=[4,6,10],root_index=8,
                                        vertex_radius=0.051, sorting=False, face_appearance_order=face_appearance_order
                                        )

        trunc_icosidodeca2.add_mesh_modifier(type="NODES", node_modifier=modifier2)
        trunc_icosidodeca2.appear(begin_time=t0, transition_time=0)
        modifier2.unfold(begin_time=t0,transition_time=0,to_value=120)

        trunc_icosidodeca3 = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            faces=src_faces
        ), location=[6.5, 0, 2],rotation_euler=ibpy.camera_alignment_euler(camera_empty, camera_location),scale=1
        )

        modifier3 = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "trunc_icosidodeca"],
                                         edge_material="example", vertex_material="red", edge_radius=0.0125,
                                         face_types=[4,6,10], root_index=6,
                                         vertex_radius=0.0251, sorting=False,
                                         face_appearance_order=face_appearance_order[:-1],projection=True
                                         )

        trunc_icosidodeca3.add_mesh_modifier(type="NODES", node_modifier=modifier3)
        trunc_icosidodeca3.appear(begin_time=t0, transition_time=0)

        modifier3.grow(begin_time=t0,transition_time=17,max_faces=119)
        modifier.grow(begin_time=t0,transition_time=17,max_faces=119)
        t0 = 0.5 + modifier2.grow(begin_time=t0,transition_time=17,max_faces=119)

        modifier.grow(begin_time=t0, transition_time=0, max_faces=120)
        t0 = 0.5 + modifier2.grow(begin_time=t0, transition_time=0, max_faces=120)
        self.t0=t0

    def trinity(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True, reflection_color=[0.05, 0, 0, 1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 10]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        group = CoxH3()
        # dodecahedron
        seed = [1, 1, 1]
        # truncated icosidodecahedron
        # seed = [1,1,-1]
        radius = 1.5
        vertices = group.get_real_point_cloud(signature=seed)
        faces = group.get_faces(signature=seed)
        locations = [radius * Vector([np.cos(tau / 5 * i), np.sin(tau / 5 * i), 0]) for i in range(5)]

        s, r, t = compute_similarity_transform(*[vertices[i] for i in faces[0][0:3]], *locations[0:3])
        t.z = 0
        vertices = apply_similarity_to_vertices(vertices, s, r, t)

        dodeca = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            edges=group.get_edges(signature=seed),
            faces=faces
        ),location=[-8,0,0]
        )
        modifier = CustomUnfoldModifier(face_materials=["trunc_octa"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,face_types=[5],
                                        vertex_radius=0.1, sorting=False, face_appearance_order=[6,11,8,10,4,5,7,9,3,1,2,0]
                                        )

        dodeca.add_mesh_modifier(type="NODES",node_modifier=modifier)
        dodeca.appear(begin_time=t0,transition_time=0)

        dodeca2 = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            edges=group.get_edges(signature=seed),
            faces=faces
        ), location=[0, 0, -1.5]
        )
        modifier2 = CustomUnfoldModifier(face_materials=["trunc_octa"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,
                                        face_types=[5],root_index=6,
                                        vertex_radius=0.1, sorting=False,
                                        face_appearance_order=[6, 11, 8, 10, 4, 5, 7, 9, 3, 1, 2, 0]
                                        )

        dodeca2.add_mesh_modifier(type="NODES", node_modifier=modifier2)
        dodeca2.appear(begin_time=t0, transition_time=0)
        modifier2.unfold(begin_time=t0,transition_time=0)

        dodeca3 = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            edges=group.get_edges(signature=seed),
            faces=faces
        ), location=[6.5, 0, 2],rotation_euler=ibpy.camera_alignment_euler(camera_empty, camera_location),scale=1.5
        )

        modifier3 = CustomUnfoldModifier(face_materials=["trunc_octa"],
                                         edge_material="example", vertex_material="red", edge_radius=0.05,
                                         face_types=[5], root_index=6,
                                         vertex_radius=0.1, sorting=False,
                                         face_appearance_order=[6, 11, 8, 10, 4, 5, 7, 9, 3, 1, 2, 0],projection=True
                                         )

        dodeca3.add_mesh_modifier(type="NODES", node_modifier=modifier3)
        dodeca3.appear(begin_time=t0, transition_time=0)

        modifier3.grow(begin_time=t0,transition_time=17,max_faces=12)
        modifier.grow(begin_time=t0,transition_time=17,max_faces=12)
        t0 = 0.5 + modifier2.grow(begin_time=t0,transition_time=17,max_faces=12)


        self.t0=t0

    def stereographic(self):
        t0 = 0
        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * Vector([0, 0, 236]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=Vector([0, 0, 0]))
        camera_location = [0, -30, 10]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=90)

        create_glow_composition(threshold=1, type="BLOOM", size=4)
        group = CoxH3()
        # dodecahedron
        seed = [1, 1, 1]
        # truncated icosidodecahedron
        # seed = [1,1,-1]

        radius = 1.25
        vertices = group.get_real_point_cloud(signature=seed)
        faces = group.get_faces(signature=seed)
        locations = [radius * Vector([np.cos(tau / 5 * i), np.sin(tau / 5 * i), 0]) for i in range(5)]

        s, r, t = compute_similarity_transform(*[vertices[i] for i in faces[0][0:3]], *locations[0:3])
        t.z = 0
        vertices = apply_similarity_to_vertices(vertices, s, r, t)

        trunc_ico = BObject(mesh=ibpy.create_mesh(
            vertices=vertices,
            edges=group.get_edges(signature=seed),
            faces=faces
        )
        )
        modifier = StereographicProjectionModifier(unpublished=True)
        # modifier.rotate(from_value=[0,0,0],rotation_euler=[0, pi / 6, 0], begin_time=0, transition_time=0.1)

        trunc_ico.add_mesh_modifier(type="NODES", node_modifier=modifier)
        ibpy.set_socket_data_for_geometry_node_modifier(trunc_ico,
                                                        {
                                                            "RayCenter": get_texture("important"),
                                                            "PolyhedronVertexMaterial": get_texture(
                                                                "plastic_custom1"),
                                                            "PolyhedronEdgeMaterial": get_texture(
                                                                "plastic_custom2"),
                                                            "PolyhedronFaceMaterial": get_texture("glass_joker",
                                                                                                  ior=1.1,
                                                                                                  scatter_density=0.5,
                                                                                                  absorption_density=0,
                                                                                                  roughness=0.2),
                                                            "ProjectionPlaneMaterial": get_texture("gray_3",
                                                                                                   alpha=0.3),
                                                            "RayMaterial": get_texture("example", emission=1),
                                                            "ProjectionGridMaterial": get_texture("blue"),
                                                            "SphereMaterial": get_texture("green")}
                                                        )

        t0 = 0.5 + trunc_ico.appear(begin_time=t0, transition_time=1)
        trunc_ico.change_alpha(alpha=0.1, slot=5, begin_time=t0, transition_time=0)

        # start with dodecahedron
        # show grid
        t0 = 0.5 + modifier.change_grid_size(from_value=0, to_value=30, begin_time=t0, transition_time=2)

        # show sphere
        t0 = 0.5 + modifier.change_sphere_thickness(from_value=0, to_value=0.005, begin_time=t0, transition_time=1)

        # show ray center
        t0 = 0.5 + modifier.change_ray_center_size(from_value=0, to_value=1, begin_time=t0, transition_time=1)

        # start rays
        modifier.change_sphere_thickness(from_value=0.005, to_value=0, begin_time=t0, transition_time=1)
        t0 = modifier.grow_rays(begin_time=t0, transition_time=2)
        modifier.change_projection_thickness(from_value=0, to_value=0.01, begin_time=t0, transition_time=0)

        modifier.change_ray_center_size(from_value=1, to_value=0, begin_time=t0, transition_time=1)
        modifier.change_grid_size(from_value=30, to_value=0, begin_time=t0, transition_time=1)
        t0 = 0.5 + modifier.rotate(from_value=[0, 0, 0], rotation_euler=[0, 0, tau], begin_time=t0,
                                   transition_time=10)
        t0 = 0.5 + modifier.rotate(from_value=[0, 0, tau], rotation_euler=[0, tau, tau], begin_time=t0,
                                   transition_time=10)

        self.t0 = t0

    def edge_highlighting(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=True,reflection_color=[0.05,0,0,1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, 0, 45]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)
        ibpy.empty_blender_view3d()

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # dodecahedron
        radius = 3
        seed = [radius*Vector([np.cos(tau/5*i),np.sin(tau/5*i),0]) for i in range(5)]

        # spheres = [Sphere(r=0.2,location=seed[i],color="red",mesh_type="ico",smooth=0,resolution=1) for i in range(5)]
        # [sphere.grow(begin_time=t0+0.1,transition_time=0.5) for sphere in spheres]
        # t0 = 1+t0

        vertices,faces = get_solid_data("DODECA")
        s,t,r = compute_similarity_transform(*[Vector(vertices[i]) for i in faces[0]][0:3],*seed[0:3])
        vertices = apply_similarity_to_vertices(vertices,s,t,r)

        dodecahedron = BObject(mesh=interface.ibpy.create_mesh(
            vertices=vertices,
            faces=faces
        ), scale=1, name="Dodecahedron", location=[0, 0, 0],rotation_euler=[pi,0,0],
        )
        modifier = EdgePairingVisualizer(face_material="joker",number_of_edges = 30,
                                        edge_material="example", vertex_material="red", edge_radius=0.1,
                                        vertex_radius=0.2,sorting=False,highlight_root=False,
                                        root_index=0,face_types=[5],face_appearance_order=[0,1,2,3,4,5,7,11,10,9,8,6])

        dodecahedron.add_mesh_modifier(type="NODES", node_modifier=modifier)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
        ibpy.change_default_integer(face_selector_node, from_value=3, to_value=0, begin_time=t0,
                                         transition_time=0)

        dodecahedron.appear(begin_time=t0, transition_time=0)
        modifier.show_faces(begin_time=t0)
        modifier.grow(begin_time=t0,transition_time=0)
        t0 = 0.5 + modifier.unfold(begin_time=t0,transition_time=0)

        dodecahedron.rotate(rotation_euler=[-pi,0,0],begin_time=t0,transition_time=10)
        ibpy.camera_zoom(lens=65, begin_time=t0, transition_time=10)
        t0 = 0.5 + modifier.fold(begin_time=t0,transition_time=10)

        self.t0 = t0

    def morphing4(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=False,reflection_color=[0.05,0,0,1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # prepare merger from truncated truncated cuboctahedron to smaller solids

        # start with the truncated icosidodecahedron
        group = CoxP6()
        signature = COXP6_SIGNATURES["PRISM6"]
        src_faces = group.get_faces(signature=signature)
        src_edges = group.get_edges(signature=signature)
        src_vertices = group.get_real_point_cloud(signature=signature)
        src_vertices =[v*3 for v in src_vertices]
        src_radius = src_vertices[0].length

        prism6 = BObject(mesh=interface.ibpy.create_mesh(
            vertices=src_vertices,
            edges=src_edges,
            faces=src_faces
        ), scale=1, name="TruncatedOctahedron", location=[0, 0, 0]
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,
                                        vertex_radius=0.1,sorting=False,root_material ="example",highlight_root=True,
                                        root_index=0,vertex_root_index=8,root_emission=0.1,face_types=[4,6])

        prism6.add_mesh_modifier(type="NODES", node_modifier=modifier)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")

        ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(src_faces), begin_time=t0,
                                         transition_time=0)
        t0 = 0.5 + prism6.appear(begin_time=t0, transition_time=1)


        # prism3
        derived_polyhedra =["PRISM3"]
        signatures = [COXP6_SIGNATURES[poly] for poly in derived_polyhedra]

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_edges = [group.get_edges(signature=signature) for signature in signatures]
        all_faces =[group.get_faces(signature=signature) for signature in signatures]
        all_scale_factors=[src_radius/vertices[0].length for vertices in all_vertices]
        all_target_vertices=[[v*scale_factor for v in vertices] for scale_factor,vertices in zip(all_scale_factors,all_vertices)]
        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx,src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx,target_v in enumerate(target_vertices):
                    if (src_v-target_v).length<dist:
                        dist = (src_v-target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        # polyhedra
        face_materials={
            "PRISM3": ["gray_5","trunc_octa"],
        }
        face_types={
            "PRISM3": [4,3],
        }

        locations = {
            "PRISM3": [-10.5,-7,-0.5],
        }
        transformations = []
        for j in range(len(derived_polyhedra)):
            transformations.append(lambda i, j=j: all_target_vertices[j][all_maps[j][i]])
            transformations.append(lambda i: src_vertices[i])

        polyhedra = {}
        for vertices,edges,faces,derived_polyhedron in zip(all_target_vertices,all_edges,all_faces,derived_polyhedra):
            polyhedron = BObject(mesh=interface.ibpy.create_mesh(
                vertices=vertices,
                edges=edges,
                faces=faces), scale=0.25, name=derived_polyhedron,location=locations[derived_polyhedron])
            modifier = CustomUnfoldModifier(face_materials=face_materials[derived_polyhedron],face_types=face_types[derived_polyhedron],
                                            edge_material="example", vertex_material="red", edge_radius=0.05,
                                            vertex_radius=0.1,sorting=False)

            polyhedron.add_mesh_modifier(type="NODES", node_modifier=modifier)
            face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
            ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(faces), begin_time=t0,transition_time=0)

            polyhedra[derived_polyhedron]=polyhedron
            first = False


        normals = [group.normals[i].real() for i in range(3)]
        planes = [Plane(u=[-8, 8], v=[-8, 8], color='mirror', name="Mirror"+str(i),
                        shadow=False, normal=normals[i], roughness=0.05,
                        solid=0.1, solidify_mode="SIMPLE",
                        smooth=4,subdivision_type="SIMPLE"
                        ,use_rim=True,use_rim_only=True,offset=0) for i in range(3)]

        center = [5,3.25,0]
        prism6.move_to(target_location=center,begin_time=t0,transition_time=2)
        [plane.move_to(target_location=center,begin_time=t0,transition_time=0) for plane in planes]

        ibpy.camera_move(shift=[20,0,30],begin_time=t0,transition_time=2)
        modifier.change_alpha(2, from_value=1, to_value=0.25, begin_time=t0, transition_time=2)
        t0 = 0.5 + ibpy.camera_zoom(lens=58,begin_time=t0,transition_time=2)

        dia = DynkinDiagram(dim=3, labels=["3"], location=[-7.5, -6, 8])
        rot_x = Quaternion(Vector([1, 0, 0]), -pi / 2)
        quaternion = ibpy.camera_alignment_quaternion(camera_empty, camera_location=[20, -30, 30])
        dia.rotate(rotation_quaternion=quaternion @ rot_x, begin_time=t0, transition_time=0)
        planes[0].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[2], begin_time=t0, transition_time=0.5)
        planes[1].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[0], begin_time=t0, transition_time=0.5)
        planes[2].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[1], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(labels=[0], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=1)

        # prism3
        dia.disappear_customized(rings=[0],begin_time=t0+1,transition_time=1)
        t0 = 0.5 + prism6.index_transform_mesh(transformations=transformations, begin_time=t0, transition_time=3)
        dia2 = DynkinDiagram(dim=3, labels=[ "3"],
                             location=Vector(locations["PRISM3"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_PRISM3")
        dia2.appear(begin_time=t0, transition_time=1)
        dia2.appear_customized(rings=[1, 2], begin_time=t0, transition_time=1)
        polyhedra["PRISM3"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0], begin_time=t0, transition_time=1)
        t0 = 0.5 + prism6.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)


        [plane.disappear(begin_time=t0, transition_time=1) for plane in planes]
        t0 = 1.5 + t0

        prism6.rescale(rescale=0.25, begin_time=t0, transition_time=1)
        t0 = 0.5 + prism6.move_to(target_location=[-3.5, -1, 5.5], begin_time=t0, transition_time=1)

        self.t0 = t0

    def morphing3(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=False,reflection_color=[0.05,0,0,1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # prepare merger from truncated truncated cuboctahedron to smaller solids

        # start with the truncated icosidodecahedron
        group = CoxA3()
        signature = COXA3_SIGNATURES["TRUNC_OCTA"]
        src_faces = group.get_faces(signature=signature)
        src_edges = group.get_edges(signature=signature)
        src_vertices = group.get_real_point_cloud(signature=signature)
        src_vertices =[v*2.5 for v in src_vertices]
        src_radius = src_vertices[0].length

        trunc_octa = BObject(mesh=interface.ibpy.create_mesh(
            vertices=src_vertices,
            edges=src_edges,
            faces=src_faces
        ), scale=1, name="TruncatedOctahedron", location=[0, 0, 0]
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,
                                        vertex_radius=0.1,sorting=False,root_material ="example",highlight_root=True,
                                        root_index=0,vertex_root_index=1,root_emission=0.1,face_types=[4,6])

        trunc_octa.add_mesh_modifier(type="NODES", node_modifier=modifier)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
        ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(src_faces), begin_time=t0,
                                         transition_time=0)
        t0 = 0.5 + trunc_octa.appear(begin_time=t0, transition_time=1)


        # derived polyhedra
        derived_polyhedra =["CUBOCTA","TRUNC_TETRA","OCTA","TETRA"]
        signatures = [COXA3_SIGNATURES[poly] for poly in derived_polyhedra]

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_edges = [group.get_edges(signature=signature) for signature in signatures]
        all_faces =[group.get_faces(signature=signature) for signature in signatures]
        all_scale_factors=[src_radius/vertices[0].length for vertices in all_vertices]
        all_target_vertices=[[v*scale_factor for v in vertices] for scale_factor,vertices in zip(all_scale_factors,all_vertices)]
        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx,src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx,target_v in enumerate(target_vertices):
                    if (src_v-target_v).length<dist:
                        dist = (src_v-target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        # polyhedra
        face_materials={
            "CUBOCTA":["gray_5","trunc_octa"],
            "TRUNC_TETRA": [ "trunc_octa","trunc_octa"],
            "OCTA": ["trunc_octa"],
            "TETRA": ["trunc_octa"],
        }
        face_types={
            "CUBOCTA": [4,3],
            "TRUNC_TETRA":[3,6],
            "OCTA": [3],
            "TETRA": [3],
        }

        locations = {
            "CUBOCTA": [-10.5,-7,-0.5],
            "TRUNC_TETRA": [-7,-4.5,-0.5],
            "OCTA": [-3.5,-2,-0.5],
            "TETRA": [-10.8,-8.5,-7.5],
        }
        transformations = []
        for j in range(len(derived_polyhedra)):
            transformations.append(lambda i, j=j: all_target_vertices[j][all_maps[j][i]])
            transformations.append(lambda i: src_vertices[i])

        polyhedra = {}
        first = True
        for vertices,edges,faces,derived_polyhedron in zip(all_target_vertices,all_edges,all_faces,derived_polyhedra):
            if first:
                # to preserve the different colors for the squares we have to derive it from the full solid
                polyhedron = BObject(mesh=interface.ibpy.create_mesh(
                    vertices=src_vertices,
                    edges=src_edges,
                    faces=src_faces), scale=0.25, name=derived_polyhedron, location=locations[derived_polyhedron])
                polyhedron.index_transform_mesh(transformations = [transformations[0]],begin_time=t0,transition_time=0)
                modifier = CustomUnfoldModifier(face_materials=face_materials[derived_polyhedron],
                                                face_types=[4,6,8],
                                                edge_material="example", vertex_material="red", edge_radius=0.05,
                                                vertex_radius=0.1, sorting=False)
            else:
                polyhedron = BObject(mesh=interface.ibpy.create_mesh(
                    vertices=vertices,
                    edges=edges,
                    faces=faces), scale=0.25, name=derived_polyhedron,location=locations[derived_polyhedron])
                modifier = CustomUnfoldModifier(face_materials=face_materials[derived_polyhedron],face_types=face_types[derived_polyhedron],
                                                edge_material="example", vertex_material="red", edge_radius=0.05,
                                                vertex_radius=0.1,sorting=False)

            polyhedron.add_mesh_modifier(type="NODES", node_modifier=modifier)
            face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
            ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(faces), begin_time=t0,transition_time=0)

            polyhedra[derived_polyhedron]=polyhedron
            first = False


        normals = [group.normals[i].real() for i in range(3)]
        planes = [Plane(u=[-8, 8], v=[-8, 8], color='mirror', name="Mirror"+str(i),
                        shadow=False, normal=normals[i], roughness=0.05,
                        solid=0.1, solidify_mode="SIMPLE",
                        smooth=4,subdivision_type="SIMPLE"
                        ,use_rim=True,use_rim_only=True,offset=0) for i in range(3)]

        center = [5,3.25,0]
        trunc_octa.move_to(target_location=center,begin_time=t0,transition_time=2)
        [plane.move_to(target_location=center,begin_time=t0,transition_time=0) for plane in planes]

        ibpy.camera_move(shift=[20,0,30],begin_time=t0,transition_time=2)
        modifier.change_alpha(2, from_value=1, to_value=0.25, begin_time=t0, transition_time=2)
        t0 = 0.5 + ibpy.camera_zoom(lens=58,begin_time=t0,transition_time=2)

        dia = DynkinDiagram(dim=3, labels=["3", "3"], location=[-7.5, -6, 8])
        rot_x = Quaternion(Vector([1, 0, 0]), -pi / 2)
        quaternion = ibpy.camera_alignment_quaternion(camera_empty, camera_location=[20, -30, 30])
        dia.rotate(rotation_quaternion=quaternion @ rot_x, begin_time=t0, transition_time=0)
        planes[0].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[0], begin_time=t0, transition_time=0.5)
        planes[1].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[2], begin_time=t0, transition_time=0.5)
        planes[2].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[1], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(labels=[0, 1], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=1)

        # cuboctahedron
        t0 = 0.5 + trunc_octa.index_transform_mesh(transformations=transformations, begin_time=t0, transition_time=3)
        dia.appear_customized(rings=[1], begin_time=t0, transition_time=1)
        dia2 = DynkinDiagram(dim=3, labels=["3", "3"],
                             location=Vector(locations["CUBOCTA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_RHOMBICUBOCTA")
        dia2.appear(begin_time=t0, transition_time=1)
        dia2.appear_customized(rings=[0, 2], begin_time=t0, transition_time=1)
        polyhedra["CUBOCTA"].appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # truncated tetrahedron
        dia.disappear_customized(rings=[2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        dia3 = DynkinDiagram(dim=3, labels=["3", "3"],
                             location=Vector(locations["TRUNC_TETRA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_TRUNC_OCTA")
        dia3.appear(begin_time=t0, transition_time=1)
        dia3.appear_customized(rings=[0, 1], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["TRUNC_TETRA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # octahedron
        modifier.change_alpha(1, from_value=1, to_value=0.25, begin_time=t0, transition_time=1)
        dia.disappear_customized(rings=[0,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        dia4 = DynkinDiagram(dim=3, labels=["3", "3"],
                             location=Vector(locations["OCTA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_TRUNC_CUBE")
        dia4.appear(begin_time=t0, transition_time=1)
        dia4.appear_customized(rings=[1], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["OCTA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # tetra
        dia.disappear_customized(rings=[1, 2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        dia4 = DynkinDiagram(dim=3, labels=["3", "3"],
                             location=Vector(locations["TETRA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_CUBOCTA")
        dia4.appear(begin_time=t0, transition_time=1)
        dia4.appear_customized(rings=[0], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["TETRA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[1, 2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        [plane.disappear(begin_time=t0, transition_time=1) for plane in planes]
        t0 = 1.5 + t0

        trunc_octa.rescale(rescale=0.25, begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_octa.move_to(target_location=[-3.5, -1, 5.5], begin_time=t0, transition_time=1)

        self.t0 = t0

    def morphing2(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=False,reflection_color=[0.05,0,0,1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # prepare merger from truncated truncated cuboctahedron to smaller solids

        # start with the truncated icosidodecahedron
        group = CoxB3()
        signature = COXB3_SIGNATURES["TRUNC_CUBOCTA"]
        src_faces = group.get_faces(signature=signature)
        src_edges = group.get_edges(signature=signature)
        src_vertices = group.get_real_point_cloud(signature=signature)
        src_vertices =[v*1.7 for v in src_vertices]
        src_radius = src_vertices[0].length

        trunc_cubocta = BObject(mesh=interface.ibpy.create_mesh(
            vertices=src_vertices,
            edges=src_edges,
            faces=src_faces
        ), scale=1, name="TruncatedCuboctahedron", location=[0, 0, 0]
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "tetra"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,
                                        vertex_radius=0.1,sorting=False,root_material ="example",highlight_root=True,
                                        root_index=0,vertex_root_index=32,root_emission=0.1,face_types=[4,6,8])

        trunc_cubocta.add_mesh_modifier(type="NODES", node_modifier=modifier)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
        ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(src_faces), begin_time=t0,
                                         transition_time=0)
        t0 = 0.5 + trunc_cubocta.appear(begin_time=t0, transition_time=1)


        # derived polyhedra
        derived_polyhedra =["RHOMBICUBOCTA","TRUNC_OCTA","TRUNC_CUBE","CUBOCTA","CUBE","OCTA"]
        signatures = [COXB3_SIGNATURES[poly] for poly in derived_polyhedra]

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_edges = [group.get_edges(signature=signature) for signature in signatures]
        all_faces =[group.get_faces(signature=signature) for signature in signatures]
        all_scale_factors=[src_radius/vertices[0].length for vertices in all_vertices]
        all_target_vertices=[[v*scale_factor for v in vertices] for scale_factor,vertices in zip(all_scale_factors,all_vertices)]
        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx,src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx,target_v in enumerate(target_vertices):
                    if (src_v-target_v).length<dist:
                        dist = (src_v-target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        # polyhedra
        face_materials={
            "RHOMBICUBOCTA": ["gray_5", "trunc_octa", "tetra"],
            "TRUNC_OCTA": [ "tetra","trunc_octa"],
            "TRUNC_CUBE": ["trunc_octa","tetra"],
            "CUBOCTA": ["trunc_octa","tetra"],
            "CUBE":["tetra"],
            "OCTA":["trunc_octa"]
        }
        face_types={
            "RHOMBICUBOCTA": [4,3,4],
            "TRUNC_OCTA": [4,6],
            "TRUNC_CUBE": [3,8],
            "CUBOCTA": [3,4],
            "CUBE": [4],
            "OCTA":[3]
        }

        locations = {
            "RHOMBICUBOCTA": [-10.5,-7,-0.5],
            "TRUNC_OCTA": [-7,-4.5,-0.5],
            "TRUNC_CUBE": [-3.5,-2,-0.5],
            "CUBOCTA": [-10.8,-8.5,-7.5],
            "CUBE": [-7,-5.7,-7.5],
            "OCTA": [-3.2,-3.0,-7.5]
        }
        transformations = []
        for j in range(6):
            transformations.append(lambda i, j=j: all_target_vertices[j][all_maps[j][i]])
            transformations.append(lambda i: src_vertices[i])

        polyhedra = {}
        first = True
        for vertices,edges,faces,derived_polyhedron in zip(all_target_vertices,all_edges,all_faces,derived_polyhedra):
            if first:
                # to preserve the different colors for the squares we have to derive it from the full solid
                polyhedron = BObject(mesh=interface.ibpy.create_mesh(
                    vertices=src_vertices,
                    edges=src_edges,
                    faces=src_faces), scale=0.25, name=derived_polyhedron, location=locations[derived_polyhedron])
                polyhedron.index_transform_mesh(transformations = [transformations[0]],begin_time=t0,transition_time=0)
                modifier = CustomUnfoldModifier(face_materials=face_materials[derived_polyhedron],
                                                face_types=[4,6,8],
                                                edge_material="example", vertex_material="red", edge_radius=0.05,
                                                vertex_radius=0.1, sorting=False)
            else:
                polyhedron = BObject(mesh=interface.ibpy.create_mesh(
                    vertices=vertices,
                    edges=edges,
                    faces=faces), scale=0.25, name=derived_polyhedron,location=locations[derived_polyhedron])
                modifier = CustomUnfoldModifier(face_materials=face_materials[derived_polyhedron],face_types=face_types[derived_polyhedron],
                                                edge_material="example", vertex_material="red", edge_radius=0.05,
                                                vertex_radius=0.1,sorting=False)

            polyhedron.add_mesh_modifier(type="NODES", node_modifier=modifier)
            face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
            ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(faces), begin_time=t0,transition_time=0)

            polyhedra[derived_polyhedron]=polyhedron
            first = False


        normals = [group.normals[i].real() for i in range(3)]
        planes = [Plane(u=[-8, 8], v=[-8, 8], color='mirror', name="Mirror"+str(i),
                        shadow=False, normal=normals[i], roughness=0.05,
                        solid=0.1, solidify_mode="SIMPLE",
                        smooth=4,subdivision_type="SIMPLE"
                        ,use_rim=True,use_rim_only=True,offset=0) for i in range(3)]

        center = [5,3.25,0]
        trunc_cubocta.move_to(target_location=center,begin_time=t0,transition_time=2)
        [plane.move_to(target_location=center,begin_time=t0,transition_time=0) for plane in planes]

        ibpy.camera_move(shift=[20,0,30],begin_time=t0,transition_time=2)
        modifier.change_alpha(2, from_value=1, to_value=0.25, begin_time=t0, transition_time=2)
        t0 = 0.5 + ibpy.camera_zoom(lens=58,begin_time=t0,transition_time=2)

        dia = DynkinDiagram(dim=3, labels=["3", "4"], location=[-7.5, -6, 8])
        rot_x = Quaternion(Vector([1, 0, 0]), -pi / 2)
        quaternion = ibpy.camera_alignment_quaternion(camera_empty, camera_location=[20, -30, 30])
        dia.rotate(rotation_quaternion=quaternion @ rot_x, begin_time=t0, transition_time=0)
        planes[0].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[0], begin_time=t0, transition_time=0.5)
        planes[1].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[2], begin_time=t0, transition_time=0.5)
        planes[2].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[1], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(labels=[0, 1], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=1)

        dia.disappear_customized(rings=[1], begin_time=t0, transition_time=1)

        # rhombicosidodecahedron
        t0 = 0.5 + trunc_cubocta.index_transform_mesh(transformations=transformations,begin_time=t0,transition_time=3)
        dia.appear_customized(rings=[1], begin_time=t0, transition_time=1)
        dia2 = DynkinDiagram(dim=3, labels=["3", "4"], location=Vector(locations["RHOMBICUBOCTA"])+Vector([-0.9,1.3,1.85]),
                             rotation_quaternion=quaternion @ rot_x,scale=0.75,name="DYNK_RHOMBICUBOCTA")
        dia2.appear(begin_time=t0, transition_time=1)
        dia2.appear_customized(rings=[0,2],begin_time=t0,transition_time=1)
        polyhedra["RHOMBICUBOCTA"].appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # truncated octahedron
        dia.disappear_customized(rings=[2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        dia3 = DynkinDiagram(dim=3, labels=["3", "4"],
                             location=Vector(locations["TRUNC_OCTA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75,name="DYNK_TRUNC_OCTA")
        dia3.appear(begin_time=t0, transition_time=1)
        dia3.appear_customized(rings=[0,1], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["TRUNC_OCTA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # truncated cube
        modifier.change_alpha(1, from_value=1, to_value=0.25, begin_time=t0, transition_time=1)
        dia.disappear_customized(rings=[0], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0,transition_time=3)
        dia4 = DynkinDiagram(dim=3, labels=["3", "4"],
                             location=Vector(locations["TRUNC_CUBE"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_TRUNC_CUBE")
        dia4.appear(begin_time=t0, transition_time=1)
        dia4.appear_customized(rings=[ 1,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["TRUNC_CUBE"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # cuboctahedron
        dia.disappear_customized(rings=[0,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        dia4 = DynkinDiagram(dim=3, labels=["3", "4"],
                             location=Vector(locations["CUBOCTA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_CUBOCTA")
        dia4.appear(begin_time=t0, transition_time=1)
        dia4.appear_customized(rings=[1], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["CUBOCTA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0, 2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # cube
        dia.disappear_customized(rings=[0,1], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        t0 = 0.5 + polyhedra["CUBE"].appear(begin_time=t0, transition_time=1)
        dia5 = DynkinDiagram(dim=3, labels=["3", "4"],
                             location=Vector(locations["CUBE"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_CUBE")
        dia5.appear(begin_time=t0, transition_time=1)
        dia5.appear_customized(rings=[2], begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0,1], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # octahedron
        dia.disappear_customized(rings=[1,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        t0 = 0.5 + polyhedra["OCTA"].appear(begin_time=t0, transition_time=1)
        dia6 = DynkinDiagram(dim=3, labels=["3", "4"],
                             location=Vector(locations["OCTA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_OCTA")
        dia6.appear(begin_time=t0, transition_time=1)
        dia6.appear_customized(rings=[0], begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[1, 2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_cubocta.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        [plane.disappear(begin_time=t0, transition_time=1) for plane in planes]
        t0 = 1.5+t0

        trunc_cubocta.rescale(rescale=0.25,begin_time=t0,transition_time=1)
        t0 = 0.5 + trunc_cubocta.move_to(target_location=[-3.5,-1,5.5],begin_time=t0,transition_time=1)

        self.t0 = t0

    def morphing(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(),
                                 reflections=False,reflection_color=[0.05,0,0,1])
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -30, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        create_glow_composition(threshold=1, type="BLOOM", size=4)


        # prepare merger from truncated icosidodecahedron to rhombicosidodecahedron
        # the invariant face is a square

        # start with the truncated icosidodecahedron
        group = CoxH3()
        signature = COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]
        src_faces = group.get_faces(signature=signature)
        src_edges = group.get_edges(signature=signature)
        src_vertices = group.get_real_point_cloud(signature=signature)
        src_radius = src_vertices[0].length

        for face in src_faces:
            if len(face)==10:
                root_face = face
                break

        r = 1.025 / np.tan(pi / 10)
        target_vertices = [interface.ibpy.Vector([r * np.cos(tau / 10 * i), r * np.sin(tau / 10 * i), 0]) for i in
                           range(3)]

        s, r, t = compute_similarity_transform(*[src_vertices[i] for i in root_face[0:3]], *target_vertices)
        verts = apply_similarity_to_vertices(src_vertices, s, r, t)

        trunc_ico_tmp = BObject(mesh=interface.ibpy.create_mesh(
            vertices=verts,
            edges=src_edges,
            faces=src_faces
        ), scale=1, name="TruncatedIcosidodecahedron", location=[0, 0, -6.7087],rotation_euler=[pi,0,0],
        )

        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "trunc_icosidodeca"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,
                                        vertex_radius=0.1,
                                        sorting=False,root_material ="example",highlight_root=False)

        modifier.change_alpha(2, from_value=1, to_value=0.1, begin_time=t0, transition_time=0)
        trunc_ico_tmp.add_mesh_modifier(type="NODES", node_modifier=modifier)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
        ibpy.change_default_integer(face_selector_node, from_value=3, to_value=120, begin_time=t0,
                                         transition_time=0)
        t0 = trunc_ico_tmp.appear(begin_time=t0, transition_time=0)
        ibpy.set_origin(trunc_ico_tmp, "ORIGIN_GEOMETRY")
        modifier.change_alpha(2, from_value=1, to_value=0.1, begin_time=t0, transition_time=0)


        t0 += 0.5

        modifier.change_alpha(2, from_value=0.1, to_value=1, begin_time=t0, transition_time=1)
        trunc_ico_tmp.rotate(rotation_euler=[148.2/180*pi,-pi/2,0],begin_time=t0,transition_time=1)
        t0 = 0.5 + trunc_ico_tmp.rescale(rescale=1.0261,begin_time=t0,transition_time=1)


        # replace the transformed icosidodecahedron with the natural one

        # start with the truncated icosidodecahedron
        group = CoxH3()
        signature = COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]
        src_faces = group.get_faces(signature=signature)
        src_edges = group.get_edges(signature=signature)
        src_vertices = group.get_real_point_cloud(signature=signature)
        src_vertices = [v for v in src_vertices]
        src_radius = src_vertices[0].length

        trunc_ico = BObject(mesh=interface.ibpy.create_mesh(
            vertices=src_vertices,
            edges=src_edges,
            faces=src_faces
        ), scale=1, name="TruncatedIcosidodecahedron", location=[0, 0, 0]
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "trunc_icosidodeca"],
                                        edge_material="example", vertex_material="red", edge_radius=0.05,
                                        vertex_radius=0.1, sorting=False, root_material="example", highlight_root=True,
                                        root_index=0,
                                        vertex_root_index=99, root_emission=0.1, face_types=[4, 6, 10])

        trunc_ico.add_mesh_modifier(type="NODES", node_modifier=modifier)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
        ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(src_faces), begin_time=t0,
                                    transition_time=0)

        # derived polyhedra
        derived_polyhedra =["RHOMBICOSIDODECA","TRUNC_ICOSA","TRUNC_DODECA","ICOSIDODECA","DODECA","ICOSA"]
        signatures = [COXH3_SIGNATURES[poly] for poly in derived_polyhedra]

        all_vertices = [group.get_real_point_cloud(signature=signature) for signature in signatures]
        all_edges = [group.get_edges(signature=signature) for signature in signatures]
        all_faces =[group.get_faces(signature=signature) for signature in signatures]
        all_scale_factors=[src_radius/vertices[0].length for vertices in all_vertices]
        all_target_vertices=[[v*scale_factor for v in vertices] for scale_factor,vertices in zip(all_scale_factors,all_vertices)]
        all_maps = []

        for target_vertices in all_target_vertices:
            index_map = {}
            for src_idx,src_v in enumerate(src_vertices):
                dist = np.inf
                min_idx = -1
                for target_idx,target_v in enumerate(target_vertices):
                    if (src_v-target_v).length<dist:
                        dist = (src_v-target_v).length
                        min_idx = target_idx
                index_map[src_idx] = min_idx
            all_maps.append(index_map)

        # polyhedra
        face_materials={
            "RHOMBICOSIDODECA": ["gray_5", "trunc_octa", "trunc_icosidodeca"],
            "TRUNC_ICOSA": [ "trunc_icosidodeca","trunc_octa"],
            "TRUNC_DODECA": ["trunc_octa","trunc_icosidodeca"],
            "ICOSIDODECA": ["trunc_octa","trunc_icosidodeca"],
            "DODECA":["trunc_icosidodeca"],
            "ICOSA":["trunc_octa"]
        }
        face_types={
            "RHOMBICOSIDODECA": [4,3,5],
            "TRUNC_ICOSA": [5,6],
            "TRUNC_DODECA": [3,10],
            "ICOSIDODECA": [3,5],
            "DODECA": [5],
            "ICOSA":[3]
        }

        locations = {
            "RHOMBICOSIDODECA": [-10.5,-7,-0.5],
            "TRUNC_ICOSA": [-7,-4.5,-0.5],
            "TRUNC_DODECA": [-3.5,-2,-0.5],
            "ICOSIDODECA": [-10.8,-8.5,-7.5],
            "DODECA": [-7,-5.7,-7.5],
            "ICOSA": [-3.2,-3.0,-7.5]
        }

        polyhedra = {}
        for vertices,edges,faces,derived_polyhedron in zip(all_target_vertices,all_edges,all_faces,derived_polyhedra):
            polyhedron = BObject(mesh=interface.ibpy.create_mesh(
                vertices=vertices,
                edges=edges,
                faces=faces), scale=0.25, rotation_euler=[148.2/180*pi,-pi/2,0],name=derived_polyhedron,location=locations[derived_polyhedron])
            modifier = CustomUnfoldModifier(face_materials=face_materials[derived_polyhedron],face_types=face_types[derived_polyhedron],
                                            edge_material="example", vertex_material="red", edge_radius=0.05,
                                            vertex_radius=0.1,sorting=False)

            polyhedron.add_mesh_modifier(type="NODES", node_modifier=modifier)
            face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
            ibpy.change_default_integer(face_selector_node, from_value=3, to_value=len(faces), begin_time=t0,transition_time=0)

            polyhedra[derived_polyhedron]=polyhedron


        normals = [group.normals[i].real() for i in range(3)]
        planes = [Plane(u=[-8, 8], v=[-8, 8], color='mirror', name="Mirror"+str(i),
                        shadow=False, normal=normals[i], roughness=0.05,
                        solid=0.1, solidify_mode="SIMPLE",
                        smooth=4,subdivision_type="SIMPLE"
                        ,use_rim=True,use_rim_only=True,offset=0) for i in range(3)]

        center = [5,3.25,0]
        trunc_ico.move_to(target_location=to_vector(center),begin_time=t0,transition_time=2)
        trunc_ico_tmp.move_to(target_location=to_vector(center),begin_time=t0,transition_time=2)
        [plane.move_to(target_location=center,begin_time=t0,transition_time=0) for plane in planes]

        ibpy.camera_move(shift=[20,0,30],begin_time=t0,transition_time=2)
        modifier.change_alpha(2, from_value=1, to_value=0.25, begin_time=t0, transition_time=2)
        t0 = 0.5 + ibpy.camera_zoom(lens=58,begin_time=t0,transition_time=2)

        dia = DynkinDiagram(dim=3, labels=["3", "5"], location=[-7.5, -6, 8])
        rot_x = Quaternion(Vector([1, 0, 0]), -pi / 2)
        quaternion = ibpy.camera_alignment_quaternion(camera_empty, camera_location=[20, -30, 30])
        dia.rotate(rotation_quaternion=quaternion @ rot_x, begin_time=t0, transition_time=0)
        planes[0].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[0], begin_time=t0, transition_time=0.5)
        planes[1].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[2], begin_time=t0, transition_time=0.5)
        planes[2].grow(begin_time=t0, transition_time=1)
        t0 = 1 + dia.appear_customized(nodes=[1], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(labels=[0, 1], begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(rings=[0, 1, 2], begin_time=t0, transition_time=1)
        trunc_ico.appear(begin_time=t0-0.5, transition_time=0)
        trunc_ico_tmp.disappear(begin_time=t0-0.5, transition_time=0)

        dia.disappear_customized(rings=[1], begin_time=t0, transition_time=1)

        transformations = []
        for j in range(6):
            transformations.append(lambda i,j=j:all_target_vertices[j][all_maps[j][i]])
            transformations.append(lambda i:src_vertices[i])
        # transformations.append(lambda i:all_target_vertices[0][all_maps[0][i]])
        # transformations.append(lambda i:src_vertices[i])
        # transformations.append(lambda i: all_target_vertices[1][all_maps[1][i]])
        # transformations.append(lambda i: src_vertices[i])
        # transformations.append(lambda i: all_target_vertices[2][all_maps[2][i]])
        # transformations.append(lambda i: src_vertices[i])
        # transformations.append(lambda i: all_target_vertices[3][all_maps[3][i]])
        # transformations.append(lambda i: src_vertices[i])
        # transformations.append(lambda i: all_target_vertices[4][all_maps[4][i]])
        # transformations.append(lambda i: src_vertices[i])
        # transformations.append(lambda i: all_target_vertices[5][all_maps[5][i]])
        # transformations.append(lambda i: src_vertices[i])

        # rhombicosidodecahedron
        t0 = 0.5 + trunc_ico.index_transform_mesh(transformations=transformations,begin_time=t0,transition_time=3)
        dia.appear_customized(rings=[1], begin_time=t0, transition_time=1)
        dia2 = DynkinDiagram(dim=3, labels=["3", "5"], location=Vector(locations["RHOMBICOSIDODECA"])+Vector([-0.9,1.3,1.85]),
                             rotation_quaternion=quaternion @ rot_x,scale=0.75,name="DYNK_RHOMBICOSIDODECA")
        dia2.appear(begin_time=t0, transition_time=1)
        dia2.appear_customized(rings=[0,2],begin_time=t0,transition_time=1)
        polyhedra["RHOMBICOSIDODECA"].appear(begin_time=t0, transition_time=1)

        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # truncated icosahedrond
        dia.disappear_customized(rings=[2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        dia3 = DynkinDiagram(dim=3, labels=["3", "5"],
                             location=Vector(locations["TRUNC_ICOSA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75,name="DYNK_TRUNC_ICOSA")
        dia3.appear(begin_time=t0, transition_time=1)
        dia3.appear_customized(rings=[0,1], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["TRUNC_ICOSA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[2], begin_time=t0, transition_time=1)


        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)
        # truncated dodecahedron
        modifier.change_alpha(1, from_value=1, to_value=0.25, begin_time=t0, transition_time=1)
        dia.disappear_customized(rings=[0], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0,transition_time=3)
        dia4 = DynkinDiagram(dim=3, labels=["3", "5"],
                             location=Vector(locations["TRUNC_DODECA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_TRUNC_DODECA")
        dia4.appear(begin_time=t0, transition_time=1)
        dia4.appear_customized(rings=[ 1,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["TRUNC_DODECA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # icosidodecahedron
        dia.disappear_customized(rings=[0,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        dia4 = DynkinDiagram(dim=3, labels=["3", "5"],
                             location=Vector(locations["ICOSIDODECA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_ICOSIDODECA")
        dia4.appear(begin_time=t0, transition_time=1)
        dia4.appear_customized(rings=[1], begin_time=t0, transition_time=1)
        t0 = 0.5 + polyhedra["ICOSIDODECA"].appear(begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0, 2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # dodecahedron
        dia.disappear_customized(rings=[0,1], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        t0 = 0.5 + polyhedra["DODECA"].appear(begin_time=t0, transition_time=1)
        dia5 = DynkinDiagram(dim=3, labels=["3", "5"],
                             location=Vector(locations["DODECA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_DODECA")
        dia5.appear(begin_time=t0, transition_time=1)
        dia5.appear_customized(rings=[2], begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[0,1], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        # icosahedron
        dia.disappear_customized(rings=[1,2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=3)
        t0 = 0.5 + polyhedra["ICOSA"].appear(begin_time=t0, transition_time=1)
        dia6 = DynkinDiagram(dim=3, labels=["3", "5"],
                             location=Vector(locations["ICOSA"]) + Vector([-0.9, 1.3, 1.85]),
                             rotation_quaternion=quaternion @ rot_x, scale=0.75, name="DYNK_ICOSA")
        dia6.appear(begin_time=t0, transition_time=1)
        dia6.appear_customized(rings=[0], begin_time=t0, transition_time=1)
        dia.appear_customized(rings=[1, 2], begin_time=t0, transition_time=1)
        t0 = 0.5 + trunc_ico.transform_mesh_to_next_shape2(begin_time=t0, transition_time=1)

        [plane.disappear(begin_time=t0, transition_time=1) for plane in planes]
        t0 = 1.5+t0

        trunc_ico.rescale(rescale=0.25,begin_time=t0,transition_time=1)
        t0 = 0.5 + trunc_ico.move_to(target_location=[-3.5,-1,5.5],begin_time=t0,transition_time=1)

        self.t0 = t0

    def coxeter_h3(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), reflections=True)
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -20, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        dia = DynkinDiagram(dim=3,labels = ["3","5"])
        t0 = 0.5 + dia.appear_customized(nodes=[0],begin_time=t0,transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(nodes=[2],begin_time=t0,transition_time=0.5)
        t0 = 0.5 + dia.appear_customized(nodes=[1],begin_time=t0,transition_time=0.5)
        t0 = 0.5 +dia.appear_customized(labels=[0,1],begin_time=t0,transition_time=0.5)

        t0 = 0.5 +dia.appear_customized(rings=[0,1,2],begin_time=t0,transition_time=1)

        self.t0 = t0

    def mirrors(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector(), reflections = False)
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 6.7087]))
        camera_location = [0, 0, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # prepare polyhedron data
        group = CoxH3()

        signature = COXH3_SIGNATURES["TRUNC_ICOSIDODECA"]

        faces = group.get_faces(signature=signature)
        for face in faces:
            if len(face) == 10:
                root_face = face
                break
        vertices = group.get_real_point_cloud(signature=signature)
        src_vertices = [vertices[root_face[i]] for i in range(2, -1, -1)]

        r = 1.025 / np.tan(pi / 10)
        target_vertices = [interface.ibpy.Vector([r * np.cos(tau / 10 * i), r * np.sin(tau / 10 * i), 0]) for i in range(3)]

        s, r, t = compute_similarity_transform(*src_vertices, *target_vertices)

        verts = apply_similarity_to_vertices(vertices, s, r, t)


        trunc_ico = BObject(mesh=interface.ibpy.create_mesh(
            vertices=verts,
            edges=group.get_edges(signature=signature),
            faces=faces
        ), scale=1, name="TruncatedIcosidodecahedron", location=[0, 0, 0]
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "trunc_icosidodeca"],
                                        edge_material="example",vertex_material="red",edge_radius=0.05,vertex_radius=0.1,
                                        root_index=0)

        trunc_ico.add_mesh_modifier(type="NODES", node_modifier=modifier)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
        interface.ibpy.change_default_integer(face_selector_node, from_value=3, to_value=120, begin_time=t0,
                                              transition_time=0)

        trunc_ico.appear(begin_time=t0, transition_time=0)
        modifier.change_alpha(0, from_value=1, to_value=0.1, begin_time=t0, transition_time=1)
        modifier.change_alpha(1, from_value=1, to_value=0.1, begin_time=t0, transition_time=1)
        t0 = 0.5 + modifier.change_alpha(2, from_value=1, to_value=0.1, begin_time=t0, transition_time=0)

        floor = Floor(u=[-20, 20], v=[-20, 20], location=[0, 0, -2])
        t0 = 0.5 + floor.grow(begin_time=t0, transition_time=2)

        # rotate back to original configuration
        rot = r.to_3x3()
        rot.transpose()
        center = interface.ibpy.Vector([0, 0, 6.7087])

        trunc_ico.change_material(new_color="gold",begin_time=t0,transition_time=2,slot=4)
        trunc_ico.change_material(new_color="plastic_red",begin_time=t0,transition_time=2,slot=5)
        modifier.change_edge_radius(from_value=0.05,to_value=0.1,begin_time=t0,transition_time=2)
        modifier.change_vertex_radius(from_value=0.1,to_value=0.2,begin_time=t0,transition_time=2)

        t0 = 0.5 + trunc_ico.rotate(rotation_euler=rot.to_euler(),begin_time=t0,transition_time=2,pivot=center)

        # create motion path of camera, it is important that all motion paths are initialized before
        # anyone is keyframed

        radius= 30
        circle_c = Curve(lambda phi: [0, -radius * np.cos(phi), radius * np.sin(phi)], domain=[0.2, np.pi / 3],
                         thickness=0, name='circle_c')
        ibpy.set_camera_follow(target=circle_c)
        circle_a = Curve(lambda phi: [radius * np.sin(phi), -radius * np.cos(phi), 6.7087], domain=[0, np.pi / 3],
                         rotation_euler=[0, 0, 0],
                         thickness=0, name='circle_a')
        ibpy.set_camera_follow(target=circle_a)
        ibpy.camera_change_follow_influence(circle_a,start=1,end=1,begin_time=0,transition_time=0)
        ibpy.camera_change_follow_influence(circle_c,start=0,end=0,begin_time=0,transition_time=0)
        circle_a.rotate(rotation_euler=[-0.2,0,0],begin_time=0,transition_time=1)
        circle_b = Curve(
            lambda phi: [radius * np.sin(phi), -radius * np.cos(phi) * np.cos(0.2), -radius * np.sin(phi) * np.sin(0.2)],
            domain=[0.98 * 2 * np.pi / 3, -0.98 * np.pi / 3],
            rotation_euler=[0, -np.pi / 4, 0], name='circle_b', location=[0, 0, 4],
            thickness=0)
        ibpy.set_camera_follow(target=circle_b)

        # show mirrors and keyframing the camera
        n_a=group.normals[0].real()
        plane_a = Plane(u=[-8, 8], v=[-8, 8], color='mirror', name="Mirror_A",normal=n_a,roughness=0.05,
                        shadow=False)
        plane_a.move_to(target_location=center,begin_time=t0,transition_time=0)
        t0 = 0.5 + plane_a.appear(begin_time=t0, transition_time=0.5)

        t0 = 0.5+ ibpy.camera_follow(circle_a, 0, 1, begin_time=t0 , transition_time=4)
        ibpy.camera_change_follow_influence(circle_a, start=1, end=0, begin_time=t0, transition_time=1)
        ibpy.camera_change_follow_influence(circle_c, start=0, end=1, begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.camera_follow(circle_a, 1, 0, begin_time=t0 , transition_time=4)

        n_c=group.normals[1].real()
        plane_c = Plane(u=[-8, 8], v=[-8, 8], color='mirror', name="Mirror_C", shadow=False,normal=n_c,roughness=0.05)
        plane_c.move_to(target_location=center, begin_time=t0, transition_time=0)

        t0 =  0.5 + plane_a.disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + plane_c.appear(begin_time=t0, transition_time=1)

        t0 = 0.5 + ibpy.camera_follow(circle_c, 0, 1, begin_time=t0, transition_time=4)
        t0 = 0.5 + ibpy.camera_follow(circle_c, 1, 0, begin_time=t0, transition_time=4)

        n_b = group.normals[2].real()
        plane_b = Plane(u=[-8, 8], v=[-8, 8], normal=n_b, color='mirror', name="Mirror_B", shadow=False,roughness=0.05)
        plane_b.move_to(target_location=center, begin_time=t0, transition_time=0)
        t0 = 0.5 + plane_c.disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + plane_b.appear(begin_time=t0, transition_time=1)
        t0= ibpy.camera_follow(circle_b, 0.667, 0, begin_time=t0, transition_time=3)
        t0 = 0.5+ibpy.camera_follow(circle_b, 0, 0.667, begin_time=t0, transition_time=6)

        self.t0 = t0

    def polygon_geometry(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([5, 0, 0]))
        camera_location = [5, -20, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)
        #ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        radius = 5
        verts = [radius * interface.ibpy.Vector([np.cos(tau / 8 * i), 0, np.sin(tau / 8 * i)]) for i in range(8)]
        faces = [[i for i in range(8)]]

        polygon = BObject(mesh=interface.ibpy.create_mesh(verts, faces=faces), scale=1, name="polygon", location=[0, 0, 0],
                          color="drawing")
        mod = PolyhedronViewModifier()
        polygon.add_mesh_modifier(type="NODES", node_modifier=mod)

        t0 = 0.5 + polygon.appear(begin_time=t0, transition_time=1, nice_alpha=True)

        spokes = [Cylinder.from_start_to_end(start=interface.ibpy.Vector(), end=p, color="example", thickness=0.25) for p in verts]
        for i, spoke in enumerate(spokes):
            spoke.grow(begin_time=t0 + 0.1 * i, transition_time=1)

        t0 += 2

        # create labels for alpha and beta
        times = []

        arc_alpha = Arc2(center=interface.ibpy.Vector(),
                         start_point=interface.ibpy.Vector([radius / 2, 0, 0]), rotation_euler=[pi / 2, 0, 0],
                         start_angle=0, end_angle=1.01 * pi / 4, thickness=0.5,
                         color="gray_1")

        text_alpha = SimpleTexBObject(r"\alpha=45^\circ", aligned="center",
                                      location=interface.ibpy.Vector([1.4, 0, 0.56]),
                                      color="gray_1")
        text_alpha.write(begin_time=t0 + 0.5, transition_time=0.5)
        times.append(t0)
        t0 = 0.5 + arc_alpha.grow(begin_time=t0, transition_time=1)

        arc_beta1 = Arc2(center=verts[0],
                         start_point=(verts[0] + 0.45 * (verts[1] - verts[0])), normal=[0, 1, 0],
                         start_angle=0, end_angle=-3 / 8 * pi, thickness=0.5,
                         color="gray_1")
        arc_beta2 = Arc2(center=verts[1],
                         start_point=(verts[1] + 0.45 * (verts[0] - verts[1])), normal=[0, 1, 0],
                         start_angle=0, end_angle=3 / 8 * pi, thickness=0.5,
                         color="gray_1")

        text_beta2 = SimpleTexBObject(r"\beta=67.5^\circ", aligned="center",
                                      location=interface.ibpy.Vector([3.2, 0, 2.4]),
                                      color="gray_1")
        text_beta2.write(begin_time=t0 + 0.5, transition_time=0.5)

        text_beta1 = SimpleTexBObject(r"\beta=67.5^\circ", aligned="center",
                                      location=interface.ibpy.Vector([4.1, 0, 0.37]),
                                      color="gray_1")
        text_beta1.write(begin_time=t0 + 0.5, transition_time=0.5)
        times.append(t0)
        arc_beta2.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + arc_beta1.grow(begin_time=t0, transition_time=1)

        arcs = [Arc2(center=verts[i], start_point=verts[i - 1] + 0.6 * (verts[i] - verts[i - 1]),
                     start_angle=0, end_angle=1.01 * 3 / 4 * pi, thickness=0.5, color="red", normal=[0, 1, 0]) for i in
                range(2, 8)]
        times.append(t0)
        [arc.grow(begin_time=t0, transition_time=1) for arc in arcs]
        text_gamma = SimpleTexBObject(r"\gamma=135^\circ", aligned="center", color="red",
                                      location=interface.ibpy.Vector([-4.2, 0, -0.32]))
        t0 = 0.5 + text_gamma.write(begin_time=t0 + 0.5, transition_time=0.5)

        lines = [
            SimpleTexBObject(r"\alpha = \tfrac{360^\circ}{n}", aligned="left", text_size="Large", color="text",
                             location=interface.ibpy.Vector([5, 0, 3])),
            SimpleTexBObject(r"\beta = \tfrac{1}{2}\left(180^\circ - \alpha\right)", aligned="left", text_size="Large",
                             color="text", location=interface.ibpy.Vector([5, 0, 2])),
            Text(r"\gamma = 180^\circ-\tfrac{360^\circ}{n}", aligned="left", text_size="Large", color="red",
                 outline_color="text", location=interface.ibpy.Vector([5, 0, 1]), keep_outline=True, outline_radius=0.005)
        ]

        for time, line in zip(times, lines):
            line.write(begin_time=time, transition_time=0.5)

        labels = [text_alpha, text_beta1, text_beta2, text_gamma]
        arcs = [arc_alpha, arc_beta1, arc_beta2]
        [[label.grow_letter(l, initial_scale=1, final_scale=0, begin_time=t0, transition_time=0.1) for label in labels]
         for l in range(0, 2)]
        t0 = t0 + 0.5
        shifts = [interface.ibpy.Vector([-0.4084, 0, 0]), interface.ibpy.Vector([-0.6563, 0, -0.04375]), interface.ibpy.Vector([-0.5834, 0, -0.073]),
                  interface.ibpy.Vector([-0.5688, 0, -0.1021])]
        [label.move(direction=shift, begin_time=t0, transition_time=0.5) for shift, label in zip(shifts, labels)]
        [label.rescale(rescale=2, begin_time=t0, transition_time=0.5) for label in labels]
        [label.change_color(new_color="text", begin_time=t0, transition_time=0.5) for label in labels[:-1]]
        [arc.change_color(new_color="text", begin_time=t0, transition_time=0.5) for arc in arcs]
        t0 = t0 + 1

        table_data = [
            [
                SimpleTexBObject(r"n-\text{gon}", text_size="large"),
                SimpleTexBObject(r"\text{Shape}", text_size="large"),
                Text(r"\gamma", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
            [
                SimpleTexBObject(r"3", text_size=2.2),
                SimpleTexBObject(r".", text_size="tiny", color="background"),
                Text(r"60^\circ", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
            [
                SimpleTexBObject(r"4", text_size=2.2),
                SimpleTexBObject(r".", text_size="tiny", color="background"),
                Text(r"90^\circ", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
            [
                SimpleTexBObject(r"5", text_size=2.2),
                SimpleTexBObject(r".", text_size="tiny", color="background"),
                Text(r"108^\circ", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
            [
                SimpleTexBObject(r"6", text_size=2.2),
                SimpleTexBObject(r".", text_size="tiny", color="background"),
                Text(r"120^\circ", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
            [
                SimpleTexBObject(r"8", text_size=2.2),
                SimpleTexBObject(r".", text_size="tiny", color="background"),
                Text(r"135^\circ", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
            [
                SimpleTexBObject(r"9", text_size=2.2),
                SimpleTexBObject(r".", text_size="tiny", color="background"),
                Text(r"140^\circ", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
            [
                SimpleTexBObject(r"10", text_size=2.2),
                SimpleTexBObject(r".", text_size="tiny", color="background"),
                Text(r"144^\circ", text_size=2.2, color="red", outline_color="text", keep_outline=True,
                     outline_radius=0.005)
            ],
        ]

        for line in lines:
            line.move(direction=interface.ibpy.Vector([5, 0, 2]), begin_time=t0, transition_time=1)
        t0 += 1.5

        table = Table(table_data, location=[10, 0, 0], buffer_x=1, scale=1)
        t0 = 0.5 + table.write_row(0, begin_time=t0, transition_time=1)

        polygons = [BObject(
            mesh=interface.ibpy.create_mesh([1 * interface.ibpy.Vector([np.cos(tau / n * i), 0, np.sin(tau / n * i)]) for i in range(n)],
                                            faces=[tuple([i for i in range(n)])]), name="Polygon" + str(n),
            location=[11, -0.1, 3.5 - 0.925 * p],
            color="drawing", scale=7 / 16) for p, n in zip(range(3, 10), [3, 4, 5, 6, 8, 9, 10])]

        [polygon.add_mesh_modifier(type="NODES", node_modifier=PolyhedronViewModifier()) for polygon in polygons]

        for i in range(1, len(table_data)):
            polygons[i - 1].appear(begin_time=t0, transition_time=0.5)
            t0 = 0.1 + table.write_row(i, begin_time=t0, transition_time=0.5)

        self.t0 = t0

    def triangles_manual(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 3.5]))
        camera_location = [0, 0, 40]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)
        ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        locations = [interface.ibpy.Vector([-10, 0, 0]), interface.ibpy.Vector([0, 0, 0]), interface.ibpy.Vector([10, 0, 0])]

        radius = 3
        # create base points
        for k in range(len(locations)):
            base_points = [radius * interface.ibpy.Vector([np.cos(tau / 3 * i), np.sin(tau / 3 * i), 0]) + locations[k] for i in
                           range(3)]
            base_spheres = [Sphere(0.1, location=b, resolution=1, mesh_type="ico", color="red") for b in base_points]
            [sphere.appear(begin_time=t0, transition_time=0) for sphere in base_spheres]

        self.t0 = t0

    def triangles(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 3.5]))
        camera_location = [0, 0, 40]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        create_glow_composition(threshold=1, type="BLOOM", size=1)

        locations = [interface.ibpy.Vector([-5, 0, 0]), interface.ibpy.Vector([0, 0, 0]), interface.ibpy.Vector([5, 0, 0])]
        solid_types = ["TETRA", "OCTA", "ICOSA"]
        initial_state = [3, 4, 5]
        unfolds = []
        face_appearance_orders = [
            [0, 1, 3, 2],
            [0, 1, 2, 3, 7, 6, 4, 5],
            [0, 1, 10, 15, 11, 7, 9, 2, 17, 19, 3, 18, 6, 8, 12, 13, 5, 14, 16, 4]
        ]
        for k, solid_type in enumerate(solid_types):
            verts, faces = get_solid_data(solid_type)
            # scale edge length to one
            m = np.inf
            for i, v in enumerate(verts):
                for j in range(i + 1, len(verts)):
                    diff = interface.ibpy.Vector(v) - interface.ibpy.Vector(verts[j])
                    d = diff.dot(diff)
                    if d < m:
                        m = d

            radius = 3
            # create base points
            base_points = [radius * interface.ibpy.Vector([np.cos(tau / 3 * i), np.sin(tau / 3 * i), 0]) + locations[k] for i in
                           range(3)]

            scale = radius / m ** 0.5
            verts = [scale * interface.ibpy.Vector(vert) for vert in verts]

            verts = apply_similarity_to_vertices(verts,
                                                 *compute_similarity_transform(verts[faces[0][0]], verts[faces[0][1]],
                                                                               verts[faces[0][2]], *base_points))

            bob = BObject(mesh=interface.ibpy.create_mesh(verts, faces=faces), scale=1, name=solid_type,
                          location=locations[k], rotation_euler=[pi, 0, 0])
            unfold = CustomUnfoldModifier(face_types=[3], face_materials=["tetra"], root_index=0, sorting=False,
                                          face_appearance_order=face_appearance_orders[k])
            unfold.unfold(begin_time=t0, transition_time=0)

            bob.add_mesh_modifier(type="NODES", node_modifier=unfold)
            unfold.grow(begin_time=t0, transition_time=0, max_faces=initial_state[k])
            unfolds.append(unfold)
            bob.appear(begin_time=t0, transition_time=0)

        # show angles
        remove = []
        center = to_vector(locations[0]) + radius * (interface.ibpy.Vector([-5 / 3 + np.cos(tau / 3), np.sin(tau / 3), 0]))
        arc180 = Arc2(center=center,
                      start_point=center + 3 * interface.ibpy.Vector([np.cos(pi / 6), np.sin(pi / 6), 0]),
                      start_angle=0, end_angle=1.01 * pi, thickness=0.5,
                      color="trunc_icosidodeca")
        text180 = SimpleTexBObject(r"180^\circ", aligned="center",
                                   location=[-12, 3.5, 0],
                                   rotation_euler=interface.ibpy.Vector(), color="trunc_icosidodeca", text_size="Large")
        text180.write(begin_time=t0 + 0.5, transition_time=0.5)
        t0 = 0.5 + arc180.grow(begin_time=t0, transition_time=1)
        remove.append(text180)
        remove.append(arc180)

        center = to_vector(locations[1]) + radius * (interface.ibpy.Vector([np.cos(tau / 3), np.sin(tau / 3), 0]))
        arc120 = Arc2(center=center,
                      start_point=center + 3 * interface.ibpy.Vector([np.cos(pi / 6), np.sin(pi / 6), 0]),
                      start_angle=0, end_angle=1.01 * 2 * pi / 3, thickness=0.5,
                      color="trunc_icosidodeca")
        text120 = SimpleTexBObject(r"120^\circ", aligned="center",
                                   location=[-1.5, 4, 0],
                                   rotation_euler=interface.ibpy.Vector(), color="trunc_icosidodeca", text_size="Large")
        text120.write(begin_time=t0 + 0.5, transition_time=0.5)
        t0 = 0.5 + arc120.grow(begin_time=t0, transition_time=1)
        remove.append(text120)
        remove.append(arc120)

        center = to_vector(locations[2]) + radius * (interface.ibpy.Vector([5 / 3 + np.cos(tau / 3), np.sin(tau / 3), 0]))
        arc60 = Arc2(center=center,
                     start_point=center + 3 * interface.ibpy.Vector([np.cos(pi / 2), np.sin(pi / 2), 0]),
                     start_angle=0, end_angle=1.01 * pi / 3, thickness=0.5,
                     color="trunc_icosidodeca")
        text60 = SimpleTexBObject(r"60^\circ", aligned="center",
                                  location=[7.7, 4.2, 0],
                                  rotation_euler=interface.ibpy.Vector(), color="trunc_icosidodeca", text_size="Large")
        text60.write(begin_time=t0 + 0.5, transition_time=0.5)
        t0 = 0.5 + arc60.grow(begin_time=t0, transition_time=1)
        remove.append(text60)
        remove.append(arc60)

        # finish tetrahedron
        [rem.disappear(begin_time=t0, transition_time=0.5) for rem in remove]

        tetrahedron, octahedron, icosahedron = unfolds
        ibpy.camera_move(shift=[0, -40 / r2, -40 / r2], begin_time=t0, transition_time=3)
        t0 = 0.5 + tetrahedron.fold(begin_time=t0, transition_time=2)
        t0 = 0.5 + tetrahedron.grow(begin_time=t0, transition_time=1)

        t0 = 0.5 + octahedron.fold(begin_time=t0, transition_time=2)
        t0 = 0.5 + octahedron.grow(begin_time=t0, transition_time=2)

        t0 = 0.5 + icosahedron.fold(begin_time=t0, transition_time=2)
        t0 = 0.5 + icosahedron.grow(begin_time=t0, transition_time=3)

        self.t0 = t0

    def archimedian(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([10, 0, 8.5]))
        camera_location = [10, -35, 8.5]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=35)

        # arrange solids in a V-F-E diagram

        # coord = CoordinateSystem2(domains=[[0,120],[0,200],[0,70]],
        #                           dimension=3,
        #                           lengths=[40,40,40],
        #                           radii=[0.1]*3,
        #                           n_tics=[7,8,6],
        #                           tic_labels=[{"1":1,"2":2,"4":4,"10":10,"20":20,"40":40,"100":100},{"1":1,"2":2,"4":4,"10":10,"20":20,"40":40,"100":100,"200":200},{"1":1,"2":2,"4":4,"10":10,"20":20,"60":60}],
        #                           tic_label_shifts=[Vector([0,0,-0.5]),Vector([-0.8,-0.4,-0.5]),Vector([-1,0,0])],
        #                           colors=["drawing","joker","red"],
        #                           axes_labels={r"\text{Vertices}":Vector([0.65,0,20.5]),
        #                                        r"\text{Edges}":Vector([0,0,20]),r"\text{Faces}":Vector([0,0,10.5])},
        #                           include_zeros=[False]*3,
        #                           name="ArchimedianSolids")
        # # coord.move_to(target_location=Vector([-10,0,-5]),begin_time=t0,transition_time=0)
        # # coord.rotate(rotation_euler=[0,-1/180*pi,-13/180*pi],begin_time=t0,transition_time=0)
        # t0 = 0.5 + coord.grow(begin_time=t0,transition_time=3)
        #
        # coord.log_z(begin_time=t0,transition_time=3)
        # coord.log_y(begin_time=t0,transition_time=3)
        # t0 = 0.5 + coord.log_x(begin_time=t0,transition_time=3 )

        solid_types = ["TETRA", "PRISM3", "OCTA", "CUBE", "PRISM5", "ICOSA", "PRISM6", "TRUNC_TETRA", "CUBOCTAHEDRON",
                       "PRISM8", "DODECA", "PRISM10", "TRUNC_OCTA",
                       "RHOMBICUBOCTA", "TRUNC_HEXA", "ICOSIDODECA", "TRUNC_CUBOCTA", "TRUNC_DODECA", "TRUNC_ICOSA",
                       "RHOMBICOSIDODECA", "TRUNC_ICOSIDODECA"]

        x = -5
        z = 0
        h = 3
        delta = 4
        for solid_type in solid_types:
            verts, faces = get_solid_data(solid_type)
            # scale edge length to one
            m = np.inf
            for i, v in enumerate(verts):
                for j in range(i + 1, len(verts)):
                    diff = interface.ibpy.Vector(v) - interface.ibpy.Vector(verts[j])
                    d = diff.dot(diff)
                    if d < m:
                        m = d

            scale = 1 / m ** 0.5
            verts = [scale * interface.ibpy.Vector(vert) for vert in verts]
            #bob = BObject(mesh=create_mesh(verts,faces=faces),scale=scale,name=solid_type,color=solid_type.lower(),location=Vector([40/lg(120)*lg(len(verts)),40/lg(200)*lg(len(verts)+len(faces)-2),40/lg(70)*lg(len(faces))]))
            bob = BObject(mesh=interface.ibpy.create_mesh(verts, faces=faces), scale=1, name=solid_type, color=solid_type.lower(),
                          location=interface.ibpy.Vector([x, 0, z]))
            #bob = BObject(mesh=create_mesh(verts,faces=faces),scale=scale,name=solid_type,color=solid_type.lower(),location=Vector([40/120*len(verts),40/200*(len(verts)+len(faces)-2),40/70*len(faces)]))
            rot = interface.ibpy.Quaternion(interface.ibpy.Vector(np.random.rand(3)), pi)
            bob.rotate(rotation_quaternion=rot, begin_time=t0, transition_time=40)
            modifier = PolyhedronViewModifier(face_color=solid_type.lower(), edge_color="example", vertex_color="red")
            bob.add_mesh_modifier(type="NODES", node_modifier=modifier)
            t0 = 0.5 + bob.appear(begin_time=t0, transition_time=1)
            x = x + delta
            if x > 25:
                z = z + h
                if z > 10:
                    x = -2
                else:
                    x = -5
                h = h * 1.5
                delta = min(delta * 1.5, 10)

        self.t0 = t0

    def platonic(self):
        t0 = 0
        # ibpy.set_hdri_background("forest", 'exr', simple=True,
        #                          transparent=True,
        #                          rotation_euler=pi / 180 * Vector())
        # t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        # ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
        #                        resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
        #                        motion_blur=False, shadows=False)
        #
        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 3.5]))
        camera_location = [0, -30, 30]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=45)
        # ibpy.empty_blender_view3d()

        solids = []
        rotations = [interface.ibpy.Quaternion(), interface.ibpy.Quaternion(), interface.ibpy.Quaternion(), interface.ibpy.Quaternion(), interface.ibpy.Quaternion()]
        # seed points
        center = interface.ibpy.Vector([-8.5, -3, 0])
        seed = []
        for i in range(3):
            seed.append(interface.ibpy.Vector([3 * np.cos(tau / 3 * i), 3 * np.sin(tau / 3 * i), 0]) + center)
        tetra = Polyhedron.from_points(seed, flipped=True)
        mod = CustomUnfoldModifier(face_materials=["tetra"], edge_material="plastic_text", root_index=1, face_types=[3])
        tetra.add_mesh_modifier(type="NODES", node_modifier=mod)
        t0 = 0.5 + tetra.appear(begin_time=t0, transition_time=0.5)
        ibpy.set_origin(tetra, type='ORIGIN_GEOMETRY')
        solids.append(tetra)

        # seed points
        center = interface.ibpy.Vector([8, -4, 2.5])
        seed = []
        for i in range(3):
            seed.append(interface.ibpy.Vector([3 * np.cos(tau / 3 * i), 3 * np.sin(tau / 3 * i), 0]))
        octa_rot = interface.ibpy.Quaternion([0.888, 0, 0.46, 0])
        rotations[1] = octa_rot
        octa = Polyhedron.from_points(seed, flipped=True, solid_type="OCTA", location=center,
                                      rotation_quaternion=octa_rot)
        mod = CustomUnfoldModifier(face_materials=["octa"], edge_material="plastic_text", root_index=2,
                                   face_types=[3], max_faces=8)
        octa.add_mesh_modifier(type="NODES", node_modifier=mod)
        t0 = 0.5 + octa.appear(begin_time=t0, transition_time=0.5)
        ibpy.set_origin(octa, type='ORIGIN_GEOMETRY')
        solids.append(octa)

        # seed points
        center = interface.ibpy.Vector([0, -4, 0])
        seed = []
        for i in range(3):
            seed.append(interface.ibpy.Vector([3 * np.cos(tau / 4 * i), 3 * np.sin(tau / 4 * i), 0]) + center)
        cube = Polyhedron.from_points(seed, flipped=True, solid_type="CUBE")
        mod = CustomUnfoldModifier(face_materials=["cube"], edge_material="plastic_text", root_index=1,
                                   face_types=[4], max_faces=6)
        cube.add_mesh_modifier(type="NODES", node_modifier=mod)
        t0 = 0.5 + cube.appear(begin_time=t0, transition_time=0.5)
        ibpy.set_origin(cube, type='ORIGIN_GEOMETRY')
        solids.append(cube)

        # seed points
        center = interface.ibpy.Vector([5, 4, 0])
        seed = []
        for i in range(3):
            seed.append(interface.ibpy.Vector([3 * np.cos(tau / 5 * i), 3 * np.sin(tau / 5 * i), 0]) + center)
        dodeca = Polyhedron.from_points(seed, flipped=True, solid_type="DODECA")
        mod = CustomUnfoldModifier(face_materials=["dodeca"], edge_material="plastic_text", root_index=1,
                                   face_types=[5], max_faces=13)
        dodeca.add_mesh_modifier(type="NODES", node_modifier=mod)
        t0 = 0.5 + dodeca.appear(begin_time=t0, transition_time=0.5)
        ibpy.set_origin(dodeca, type='ORIGIN_GEOMETRY')
        solids.append(dodeca)

        # seed points
        center = interface.ibpy.Vector([-5, 4, 0.42 - 3.18 - 0.67])
        seed = []
        for i in range(3):
            seed.append(interface.ibpy.Vector([3 * np.cos(tau / 5 * i), 3 * np.sin(tau / 5 * i), 0]) + center)
        icosa_rot = interface.ibpy.Quaternion([0.947, 0.308, 0.091, -0.03])
        rotations[4] = icosa_rot
        icosa = Polyhedron.from_points(seed, flipped=True, solid_type="ICOSA",
                                       rotation_quaternion=icosa_rot)
        mod = CustomUnfoldModifier(face_materials=["icosa"], edge_material="plastic_text", root_index=1,
                                   face_types=[3], max_faces=20)
        icosa.add_mesh_modifier(type="NODES", node_modifier=mod)
        t0 = 0.5 + icosa.appear(begin_time=t0, transition_time=0.5)
        ibpy.set_origin(icosa, type='ORIGIN_GEOMETRY')
        solids.append(icosa)

        [solid.rotate(rotation_quaternion=interface.ibpy.Quaternion([0, 0, 1], -pi) @ rotations[i], begin_time=0, transition_time=30)
         for i, solid in enumerate(solids)]
        self.t0 = t0

    def truncated_icosidodecahedron_preparation(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, 0, 30]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=45)
        ibpy.empty_blender_view3d()

        create_glow_composition(threshold=1, type="BLOOM", size=1)

        r = 1.025 / np.tan(pi / 10)
        target_vertices = [interface.ibpy.Vector([r * np.cos(tau / 10 * i), r * np.sin(tau / 10 * i), 0]) for i in range(2, 4)]

        spheres = [Sphere(0.05, location=interface.ibpy.Vector(v),
                          mesh_type="ico", resolution=1,
                          color="dark_red", smooth=False) for v in target_vertices]
        for sphere in spheres:
            sphere.grow(begin_time=t0, transition_time=0)

        self.t0 = t0

    def truncated_icosidodecahedron(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=0)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 6.7087]))
        camera_location = [0, 0, 30]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=45)

        create_glow_composition(threshold=1, type="BLOOM", size=4)

        verts,faces =get_solid_data("TRUNC_ICOSIDODECA")

        r = 1.025 / np.tan(pi / 10)
        target_vertices = [interface.ibpy.Vector([r * np.cos(tau / 10 * i), r * np.sin(tau / 10 * i), 0]) for i in range(3)]

        for f in faces:
            if len(f)==10:
                root = f
                break

        verts = [to_vector(v) for v in verts]
        s, r, t = compute_similarity_transform(*[verts[i] for i in root[1:4]], *target_vertices)

        verts = apply_similarity_to_vertices(verts, s, r, t)
        print("edge_length: ",edge_length(verts))


        trunc_ico = BObject(mesh=interface.ibpy.create_mesh(
            vertices=verts,
            faces=faces
        ), scale=1, name="TruncatedIcosidodecahedron", location=[0, 0, 0],rotation_euler=[pi,0,0]
        )
        modifier = CustomUnfoldModifier(face_materials=["gray_5", "trunc_octa", "trunc_icosidodeca"],
                                        edge_material="example",sphere_material="red",
                                        root_index=0, sorting=False,
                                        face_appearance_order=[0,19,53,20,52,18,45,24,33,30,42,
                                                               34,4,43,3,38,10,59,8,48,6,
                                                               25,31,16,17,23,
                                                               47,61,49,37,57,60,44,39,32,54,
                                                               11,14,2,29,5,26,7,22,9,15,
                                                               40,41,46,36,50,21,35,28,58,12,55,27,56,13,51,1
                                                               ])
        trunc_ico.add_mesh_modifier(type="NODES", node_modifier=modifier)
        progress_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="Progress")
        interface.ibpy.change_default_value(progress_node, from_value=0, to_value=20, begin_time=t0,
                                            transition_time=0)  # unfold completely

        t0 = 0.5 + trunc_ico.appear(begin_time=t0, transition_time=0)

        arc120 = Arc2(center=[1, -3, 0],
                      start_point=[1 + np.cos(pi / 3), -3 - np.sin(pi / 3), 0],
                      start_angle=0, end_angle=-2 * pi / 3, thickness=0.5,
                      color="trunc_icosidodeca")
        text120 = SimpleTexBObject(r"120^\circ", aligned="center",
                                   location=[0.7, -3.44, 0.01],
                                   rotation_euler=interface.ibpy.Vector(), color="trunc_icosidodeca", text_size="large")
        text120.write(begin_time=t0 + 0.5, transition_time=0.5)
        t0 = 0.5 + arc120.grow(begin_time=t0, transition_time=1)

        arc144 = Arc2(center=[1, -3, 0], start_point=[0, -3, 0], start_angle=0,
                      end_angle=-144 / 180 * pi,
                      thickness=0.5,
                      color="gray_5")
        text144 = SimpleTexBObject(r"144^\circ", aligned="center",
                                   location=[0.75, -2.6, 0.01],
                                   rotation_euler=interface.ibpy.Vector(), color="gray_5",
                                   text_size="large")
        text144.write(begin_time=t0 + 0.5, transition_time=0.5)
        t0 = 0.5 + arc144.grow(begin_time=t0, transition_time=1)

        arc90 = Arc2(center=[1, -3, 0], start_point=[1 + np.cos(pi / 5), -3 + np.sin(pi / 5), 0],
                     start_angle=0,
                     end_angle=-pi / 2, thickness=0.5, color="trunc_octa")
        text90 = SimpleTexBObject(r"90^\circ", aligned="center",
                                  location=[1.6, -3.0, 0.01],
                                  rotation_euler=interface.ibpy.Vector(), color="trunc_octa", text_size="large")
        text90.write(begin_time=t0 + 0.5, transition_time=0.5)
        t0 = 0.5 + arc90.grow(begin_time=t0, transition_time=1)

        camera_empty.move_to(target_location=[0, 0, 3], begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.camera_move(shift=[0, -30, -20], begin_time=t0, transition_time=3)

        delete = [arc90, text90, arc120, text120, arc144, text144]
        for i, obj in enumerate(delete):
            if isinstance(obj, Arc2):
                obj.shrink(begin_time=t0 + 0.1 * i, transition_time=0.5)
            else:
                obj.disappear(begin_time=t0 + 0.1 * i, transition_time=0.5)

        t0 = 0.5 + ibpy.change_default_value(progress_node, from_value=20, to_value=0, begin_time=t0, transition_time=3)

        ibpy.camera_zoom(lens=35, begin_time=t0, transition_time=1)
        camera_empty.move_to(target_location=[0, 0, 6.7087], begin_time=t0, transition_time=3)
        face_selector_node = interface.ibpy.get_geometry_node_from_modifier(modifier, label="FaceSelector")
        trunc_ico.rotate(rotation_euler=[pi, 0, tau], begin_time=t0, transition_time=40)
        t0 = 0.5 + interface.ibpy.change_default_integer(face_selector_node, from_value=3, to_value=120, begin_time=t0,
                                                         transition_time=15)

        t0 = 0.5 + modifier.change_alpha(2, from_value=1, to_value=0.1, begin_time=t0, transition_time=1)
        t0 = 0.5 + interface.ibpy.camera_move(shift=[0, 0, 6.7087 - 10], begin_time=t0, transition_time=10)
        self.t0 = t0

    def applet_tester(self):
        t0 = 0
        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -416, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # start with all polyhedra that contain triangles
        verts = [interface.ibpy.Vector([np.cos(tau / 3 * i), np.sin(tau / 3 * i), 0]) for i in range(3)]

        for key, val in SOLID_FACE_SIDES.items():

            verts_canon, faces = get_solid_data(key)
            dir = interface.ibpy.Vector()
            if 3 in val:
                dir += interface.ibpy.Vector([1, 0, 0])
            if 6 in val:
                dir += interface.ibpy.Vector([0, -1, 0])
            if 4 in val:
                dir += interface.ibpy.Vector([0, 0, 1])
            if 5 in val:
                dir += interface.ibpy.Vector([-1, 0, 0])
            if 10 in val:
                dir += interface.ibpy.Vector([0, 1, 0])

            location = dir * 4 * np.sqrt(len(faces))

            mesh = ibpy.create_mesh(verts_canon, faces=faces, name=str(key) + "_MESH")
            solid = BObject(mesh=mesh, location=location, name=key)
            print(solid.ref_obj.name)
            t0 = solid.appear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def v600_unfolder(self):
        t0 = 0
        ibpy.set_hdri_background("qwantani_puresky_4k", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector())
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False, shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -416, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        vertices, faces, cells, root = create_full_cell_tree([0, 0, 0, 1])
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [interface.ibpy.Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx))
            t0 = cellobj.appear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def v14400_construction(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -55, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)

        create_glow_composition(threshold=1, type="BLOOM", size=4)
        ibpy.empty_blender_view3d()

        # seed for explicit construction
        seed = [Vector((-2.4805984497070312, 2.8861608505249023, -4.370794296264648)),
                Vector((-0.7485485076904297, 1.8861608505249023, -4.370794296264648)),
                Vector((-0.7485485076904297, -0.1138390302658081, -4.370794296264648)),
                Vector((-2.4805984497070312, -1.1138391494750977, -4.370794296264648)),
                Vector((-4.212650299072266, -0.1138390302658081, -4.370794296264648)),
                Vector( (-4.212650299072266, 1.8861608505249023, -4.370794296264648))]

        # put spheres on construction site
        seed_spheres = [Sphere(0.05, location=interface.ibpy.Vector(v),
                          mesh_type="ico", resolution=1,
                          color="dark_red", smooth=False) for v in seed]

        [sphere.grow(begin_time=t0, transition_time=0) for sphere in seed_spheres]
        t0 = 0.5 + t0


        self.t0 = t0

    def v7200d_unfolding(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -115, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)
        ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([0, 1, -1, 1],root_size=120)
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        count = 0
        children = []
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])
                    # find root plane
                    if idx==2:
                        if len(face)==6:
                            if all([verts[local_map[face[i]]].z==verts[local_map[face[0]]].z for i in range(6)]):
                                print([verts[local_map[face[i]]] for i in range(6)])

            face_types=set([len(face) for face in selected_faces])
            if len(verts) == 120:
                color = "trunc_icosidodeca"
            elif len(verts) == 12 and face_types==set([3,6]):
                color = "trunc_tetra"
            elif len(verts) == 6:
                color = "prism3"
            else:
                color = "joker"

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx),
                              color=color)
            children.append(cellobj)
            t0 = cellobj.appear(begin_time=t0, transition_time=0.10071231) # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("Construction time:",t0)
        container = BObject(children=children)
        container.appear(begin_time=0,transition_time=0)
        container.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=200)
        t0 = ibpy.camera_move(shift=[0,-85,0],begin_time=0,transition_time=10)
        t0 = ibpy.camera_move(shift=[0,-115,0],begin_time=t0,transition_time=110)
        t0 = ibpy.camera_move(shift=[0,-15,0],begin_time=t0,transition_time=20)
        self.t0 = t0

    def v7200c_unfolding(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -55, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)
        ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 0, -1],root_size=60)
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        count = 0
        children = []
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])
                    # find root plane
                    if idx==2:
                        if len(face)==6:
                            if all([verts[local_map[face[i]]].z==verts[local_map[face[0]]].z for i in range(6)]):
                                print([verts[local_map[face[i]]] for i in range(6)])

            face_types=set([len(face) for face in selected_faces])
            if len(verts) == 60:
                color = "rhombicosidodeca"
            elif len(verts) == 12 and face_types==set([3,6]):
                color = "trunc_tetra"
            elif len(verts) == 10:
                color = "prism5"
            else:
                color = "prism6"

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx),
                              color=color)
            children.append(cellobj)
            t0 = cellobj.appear(begin_time=t0, transition_time=0.10071231) # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("Construction time:",t0)
        container = BObject(children=children)
        container.appear(begin_time=0,transition_time=0)
        container.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=260)
        t0 = ibpy.camera_move(shift=[0,-70,0],begin_time=0,transition_time=10)
        t0 = ibpy.camera_move(shift=[0,-115,0],begin_time=t0,transition_time=110)
        t0 = ibpy.camera_move(shift=[0,-50,0],begin_time=t0,transition_time=20)
        self.t0 = t0

    def v7200b_unfolding(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -55, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)
        ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1,1,-1,1])
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        count = 0
        children = []
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])
                    # find root plane
                    if idx==2:
                        if len(face)==6:
                            if all([verts[local_map[face[i]]].z==verts[local_map[face[0]]].z for i in range(6)]):
                                print([verts[local_map[face[i]]] for i in range(6)])

            if len(verts) == 60:
                color = "trunc_dodeca"
            elif len(verts) == 12:
                color = "cubocta"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism3"

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx),
                              color=color)
            children.append(cellobj)
            t0 = cellobj.appear(begin_time=t0, transition_time=0.10071231) # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("Construction time:",t0)
        container = BObject(children=children)
        container.appear(begin_time=0,transition_time=0)
        container.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=260)
        t0 = ibpy.camera_move(shift=[0,-70,0],begin_time=0,transition_time=10)
        t0 = ibpy.camera_move(shift=[0,-115,0],begin_time=t0,transition_time=110)
        t0 = ibpy.camera_move(shift=[0,-50,0],begin_time=t0,transition_time=20)
        self.t0 = t0

    def v7200a_unfolding(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -55, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)
        ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, 0], root_size=60)
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        count = 0
        children = []
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])
                    # find root plane
                    if idx==2:
                        if len(face)==6:
                            if all([verts[local_map[face[i]]].z==verts[local_map[face[0]]].z for i in range(6)]):
                                print([verts[local_map[face[i]]] for i in range(6)])

            if len(verts) == 60:
                color = "trunc_icosa"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 10:
                color = "prism5"
            else:
                color = "joker"

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx),
                              color=color)
            children.append(cellobj)
            t0 = cellobj.appear(begin_time=t0, transition_time=0.10071231) # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("Construction time:",t0)
        container = BObject(children=children)
        container.appear(begin_time=0,transition_time=0)
        container.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=150)
        t0 = ibpy.camera_move(shift=[0,-70,0],begin_time=0,transition_time=10)
        t0 = ibpy.camera_move(shift=[0,-115,0],begin_time=t0,transition_time=110)
        t0 = ibpy.camera_move(shift=[0,-40,0],begin_time=t0,transition_time=20)
        self.t0 = t0

    def v14400_unfolding(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=512,
                               motion_blur=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -55, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=50)
        ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, -1], root_size=120)
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        count = 0
        children = []
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])
                    # find root plane
                    if idx==2:
                        if len(face)==6:
                            if all([verts[local_map[face[i]]].z==verts[local_map[face[0]]].z for i in range(6)]):
                                print([verts[local_map[face[i]]] for i in range(6)])

            if len(verts) == 120:
                color = "trunc_icosidodeca"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism6"

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx),
                              color=color)
            children.append(cellobj)
            if count<4:
                t0 = cellobj.appear(begin_time=0, transition_time=0)
                count+=1
            else:
                t0 = cellobj.appear(begin_time=t0, transition_time=0.10071231) # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        container = BObject(children=children)
        container.appear(begin_time=0,transition_time=0)
        container.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=270)
        t0 = ibpy.camera_move(shift=[0,-100,0],begin_time=0,transition_time=10)
        t0 = ibpy.camera_move(shift=[0,-100,0],begin_time=t0,transition_time=110)
        t0 = ibpy.camera_move(shift=[0,-105,0],begin_time=t0,transition_time=100)
        self.t0 = t0

    def v14400_unfolding_rotation(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0, 0, 60]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -360, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=45)
        ibpy.empty_blender_view3d()
        create_glow_composition(threshold=1, type="BLOOM", size=4)

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, -1], root_size=120)
        vertex2unfolded_map, unfolded_vertices = unfold(vertices, root)

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        count = 0
        children = []
        for idx in order:
            cell = list(cells.keys())[idx]
            v2u_local = vertex2unfolded_map[idx]

            verts = [Vector(unfolded_vertices[v2u_local[i]][0:3]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])
                    # find root plane
                    if idx==2:
                        if len(face)==6:
                            if all([verts[local_map[face[i]]].z==verts[local_map[face[0]]].z for i in range(6)]):
                                print([verts[local_map[face[i]]] for i in range(6)])

            if len(verts) == 120:
                color = "trunc_icosidodeca"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism6"

            cellobj = BObject(mesh=interface.ibpy.create_mesh(vertices=verts, faces=selected_faces), name="Cell" + str(idx),
                              color=color)
            children.append(cellobj)
            cellobj.appear(begin_time=0, transition_time=0)

        container = BObject(children=children)
        container.appear(begin_time=0,transition_time=0)
        container.rotate(rotation_euler=[0,0,tau],begin_time=0,transition_time=60)
        self.t0 = t0

    def v14400(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0,0,50]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
                               motion_blur=False,shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -890, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=400)


        create_glow_composition(threshold=1, type="BLOOM", size=4)
        group = CoxH4(path="../mathematics/geometry/data")

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, -1], root_size=120)
        v0=vertices[0].real()
        l = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2 + v0[3] ** 2) ** 0.5

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        children = {}
        alphas ={}
        zeros = {}
        for idx in order:
            cell = list(cells.keys())[idx]

            def stereo(v):
                vr = v.real()
                return Vector([vr[0]/(1.0001-vr[3]/l),vr[1]/(1.0001-vr[3]/l),vr[2]/(1.0001-vr[3]/l)])

            verts = [stereo(vertices[i]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            if len(verts) == 120:
                color = "trunc_icosidodeca"
                alpha = 0.1
            elif len(verts) == 24:
                alpha=0.2
                color = "trunc_octa"
            elif len(verts) == 20:
                alpha=0.3
                color = "prism10"
            else:
                alpha=0.4
                color = "prism6"
            #if len(verts)==120 or len(verts)==20 or len(verts)==12:
            cellobj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces),
                              name="Cell" + str(idx),
                              color=color)
            center = sum(verts,Vector())/len(verts)
            if center.length>0.1:
                children[cellobj]=center.length
            else:
                zeros[cellobj] = (verts[0]-center).length # radius of the cellobj
            alphas[cellobj]=alpha
        sorted_children = list(k for k,v in sorted(children.items(),key=lambda item: item[1]))
        sorted_zeros = list(k for k, v in sorted(zeros.items(), key=lambda item: item[1])) # there should only be two elements, the center and the outside
        print("centered cells: ",len(sorted_zeros))
        sorted_children.insert(0,sorted_zeros[0])
        sorted_children.append(sorted_zeros[1])
        for child in sorted_children:
            t0 = child.appear(alpha=alphas[child],begin_time=t0,transition_time=0.050071231)  # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("end time:",t0)
        zoom_time=t0
        container = BObject(children=sorted_children,shadows=False)
        container.appear(begin_time=0, transition_time=0)
        container.rotate(rotation_euler=[0, 0, tau], begin_time=0, transition_time=0.9*zoom_time)
        t0 = ibpy.camera_zoom(lens=50, begin_time=0, transition_time=zoom_time)



        self.t0 = t0

    def v14400b(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0,0,50]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False,shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -890, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=400)


        create_glow_composition(threshold=1, type="BLOOM", size=4)
        group = CoxH4(path="../mathematics/geometry/data")

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, -1], root_size=120)
        v0=vertices[0].real()
        l = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2 + v0[3] ** 2) ** 0.5

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        children = {}
        zeros = {}
        for idx in order:
            cell = list(cells.keys())[idx]

            def stereo(v):
                vr = v.real()
                return Vector([vr[0]/(1.0001-vr[3]/l),vr[1]/(1.0001-vr[3]/l),vr[2]/(1.0001-vr[3]/l)])

            verts = [stereo(vertices[i]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            if len(verts) == 120:
                color = "glass_trunc_icosidodeca"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism6"
            if len(verts)==24 or len(verts)==20 or len(verts)==12:
                cellobj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces),
                                  name="Cell" + str(idx),
                                  color=color,scatter_density=0.05,ior=1.1)
                center = sum(verts,Vector())/len(verts)
                if center.length>0.1:
                    children[cellobj]=center.length
                else:
                    zeros[cellobj] = (verts[0]-center).length # radius of the cellobj

        sorted_children = list(k for k,v in sorted(children.items(),key=lambda item: item[1]))
        for child in sorted_children:
            t0 = child.appear(begin_time=t0,transition_time=0.10071231)  # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("end time:",t0)
        zoom_time=t0
        container = BObject(children=sorted_children,shadows=False)
        container.appear(begin_time=0, transition_time=0)
        container.rotate(rotation_euler=[0, 0, tau], begin_time=0, transition_time=0.9*zoom_time)
        t0 = ibpy.camera_zoom(lens=50, begin_time=0, transition_time=zoom_time)



        self.t0 = t0

    def v14400c(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0,0,50]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False,shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -890, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=400)


        create_glow_composition(threshold=1, type="BLOOM", size=4)
        group = CoxH4(path="../mathematics/geometry/data")

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, -1], root_size=120)
        v0=vertices[0].real()
        l = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2 + v0[3] ** 2) ** 0.5

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        children = {}
        zeros = {}
        for idx in order:
            cell = list(cells.keys())[idx]

            def stereo(v):
                vr = v.real()
                return Vector([vr[0]/(1.0001-vr[3]/l),vr[1]/(1.0001-vr[3]/l),vr[2]/(1.0001-vr[3]/l)])

            verts = [stereo(vertices[i]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            if len(verts) == 120:
                color = "trunc_icosidodeca"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism6"
            if len(verts)==120 or len(verts)==20 or len(verts)==12:
                cellobj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces),
                                  name="Cell" + str(idx),
                                  color=color,scatter_density=0.05,ior=1.1)
                center = sum(verts,Vector())/len(verts)
                if center.length>0.1:
                    children[cellobj]=center.length
                else:
                    zeros[cellobj] = (verts[0]-center).length # radius of the cellobj

        sorted_children = list(k for k,v in sorted(children.items(),key=lambda item: item[1]))
        sorted_zeros = list(k for k, v in sorted(zeros.items(), key=lambda item: item[
            1]))  # there should only be two elements, the center and the outside
        sorted_children.insert(0,sorted_zeros[0])
        for child in sorted_children:
            t0 = child.appear(begin_time=t0,transition_time=0.10071231)  # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("end time:",t0)
        zoom_time=t0
        container = BObject(children=sorted_children,shadows=False)
        container.appear(begin_time=0, transition_time=0)
        container.rotate(rotation_euler=[0, 0, tau], begin_time=0, transition_time=0.9*zoom_time)
        t0 = ibpy.camera_zoom(lens=50, begin_time=0, transition_time=zoom_time)



        self.t0 = t0

    def v14400d(self):
        t0 = 0
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True,
                                 rotation_euler=pi / 180 * interface.ibpy.Vector([0,0,50]))
        t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,  # skip initialization frame at 0
                               resolution_percentage=100, engine=BLENDER_EEVEE, taa_render_samples=64,
                               motion_blur=False,shadows=False)

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [0, -890, 0]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=400)


        create_glow_composition(threshold=1, type="BLOOM", size=4)
        group = CoxH4(path="../mathematics/geometry/data")

        # create_glow_composition(threshold=1, type="BLOOM", size=4)
        vertices, faces, cells, root = create_full_cell_tree([1, -1, 1, -1], root_size=120)
        v0=vertices[0].real()
        l = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2 + v0[3] ** 2) ** 0.5

        # get growth order from tree
        order = [root.name[0]]
        next_level = root.children
        while len(next_level) > 0:
            new_level = []
            for child in next_level:
                order.append(child.name[0])
                new_level.extend(child.children)
            next_level = new_level

        children = {}
        zeros = {}
        for idx in order:
            cell = list(cells.keys())[idx]

            def stereo(v):
                vr = v.real()
                return Vector([vr[0]/(1.0001-vr[3]/l),vr[1]/(1.0001-vr[3]/l),vr[2]/(1.0001-vr[3]/l)])

            verts = [stereo(vertices[i]) for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            if len(verts) == 120:
                color = "trunc_icosidodeca"
            elif len(verts) == 24:
                color = "trunc_octa"
            elif len(verts) == 20:
                color = "prism10"
            else:
                color = "prism6"
            if len(verts)==20 or len(verts)==12:
                cellobj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces),
                                  name="Cell" + str(idx),
                                  color=color,scatter_density=0.05,ior=1.1)
                center = sum(verts,Vector())/len(verts)
                children[cellobj]=center.length


        sorted_children = list(k for k,v in sorted(children.items(),key=lambda item: item[1]))
        for child in sorted_children:
            t0 = child.appear(begin_time=t0,transition_time=0.10071231)  # the strange interval avoids black shadows, when cell appearance matches the beginning of teh frame.

        print("end time:",t0)
        zoom_time=t0
        container = BObject(children=sorted_children,shadows=False)
        container.appear(begin_time=0, transition_time=0)
        container.rotate(rotation_euler=[0, 0, tau], begin_time=0, transition_time=0.9*zoom_time)
        t0 = ibpy.camera_zoom(lens=50, begin_time=0, transition_time=zoom_time)



        self.t0 = t0

    def corner(self):
        t0 = 0
        # ibpy.set_hdri_background("quarry_04_puresky_4k", 'exr', simple=True,
        #                          transparent=True,
        #                          rotation_euler=pi / 180 * Vector())
        # t0 = ibpy.set_hdri_strength(1, begin_time=t0, transition_time=1)
        # ibpy.set_render_engine(denoising=False, transparent=False, frame_start=1,  # skip initialization frame at 0
        #                        resolution_percentage=100, engine=CYCLES, taa_render_samples=512,
        #                        motion_blur=False)
        # ibpy.empty_blender_view3d()

        camera_empty = EmptyCube(location=interface.ibpy.Vector([0, 0, 0]))
        camera_location = [15, 15, 5]
        ibpy.set_camera_location(location=camera_location)
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=30)

        for i in range(10):
            sphere = Sphere(r=0.1, color="plastic_example",
                            location=[np.cos(2 * pi / 10 * i), np.sin(2 * pi / 10 * i), 0])
            sphere.grow(begin_time=t0 + i * 0.5, transition_time=0.5)

        self.t0 = t0


if __name__ == '__main__':
    try:
        example = video_CoxH4()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        if len(dictionary) == 1:
            selection = 0
        else:
            selection = input("Choose scene:")
            if len(selection) == 0:
                selection = 0
        print("Your choice: ", selection)
        selected_scene = dictionary[int(selection)]

        if selected_scene == "stereo_projection":
            resolution = [1080, 1920]
        else:
            resolution = [1920, 1080]
        example.create(name=selected_scene, resolution=resolution, start_at_zero=True)

    except:
        print_time_report()
        raise ()
