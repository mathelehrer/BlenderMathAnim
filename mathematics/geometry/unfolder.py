from __future__ import annotations

from collections import defaultdict
from functools import partial
from time import time
from typing import DefaultDict

import numpy as np
from anytree import Node, RenderTree, AsciiStyle

from addons.solids import get_solid_data
from appearance.textures import get_texture, mix_texture
from geometry_nodes.geometry_nodes_modifier import PolyhedronViewModifier
from interface import ibpy
from interface.ibpy import Vector, Matrix, Quaternion, get_node_from_shader
from mathematics.groups.coxA3 import CoxA3
from mathematics.groups.coxA4 import CoxA4, COXA4_SIGNATURES
from mathematics.groups.coxB4 import CoxB4, COXB4_SIGNATURES
from mathematics.groups.coxD4 import CoxD4, COXD4_SIGNATURES
from mathematics.groups.coxF4 import COXF4_SIGNATURES, CoxF4
from mathematics.groups.coxH4 import CoxH4, COXH4_SIGNATURES
from mathematics.geometry.coxeter.diagram_to_matrix import CoxeterDynkinDiagram
from mathematics.geometry.meshface import MeshFace
from mathematics.lin_alg.tensors import epsilon
from mathematics.mathematica.mathematica import unit_tuples
from geometry_nodes.geometry_nodes_modifier import CrystalModifier
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs
from utils.utils import to_vector

namespace = {
    "COXA4_SIGNATURES": COXA4_SIGNATURES,
    "COXB4_SIGNATURES": COXB4_SIGNATURES,
    "COXH4_SIGNATURES": COXH4_SIGNATURES,
    "COXF4_SIGNATURES": COXF4_SIGNATURES,
    "COXD4_SIGNATURES": COXD4_SIGNATURES,
    "COXA4": CoxA4,
    "COXB4": CoxB4,
    "COXH4": CoxH4,
    "COXF4": CoxF4,
    "COXD4": CoxD4,
}

color_dict = {3: ("hexagon", "triangle"), 4: "square", 5: ("decagon", "pentagon"), 6: ("hexagon", "triangle"),
              8: ("octagon", "tetragon"), 10: ("decagon", "pentagon")}

epsilon_tensor = epsilon(4)

# Lookup table for get_color_names: pattern -> (divisor, [(verts, face_types, name), ...])
# Each entry produces {(verts, face_types, dim // divisor): name, ...}
_COLOR_NAMES_TABLE = {
    # A3 family (order 24)
    "x3x3x": (24, [(24, (4, 6), "trunc_octa"), (12, (3, 4), "cubocta"), (12, (3, 6), "trunc_tetra"),
                   (6, (3,), "octa"), (4, (3,), "tetra")]),
    "o3x3x": (24, [(12, (3, 6), "trunc_tetra"), (6, (3,), "octa"), (4, (3,), "tetra")]),
    "x3x3o": (24, [(12, (3, 6), "trunc_tetra"), (6, (3,), "octa"), (4, (3,), "tetra")]),
    "x3o3x": (24, [(12, (3, 4), "cubocta"), (4, (3,), "tetra")]),
    "o3x3o": (24, [(6, (3,), "octa")]),
    "o3o3x": (24, [(4, (3,), "tetra")]),
    "x3o3o": (24, [(4, (3,), "tetra")]),
    # B3 family (order 48)
    "x3x4x": (48, [(48, (4, 6, 8), "trunc_cubocta"), (24, (4, 6), "trunc_octa"), (24, (3, 8), "trunc_cube"),
                   (24, (3, 4), "rhombicubocta"), (12, (3, 4), "cubocta"), (8, (4,), "cube"), (6, (3,), "octa")]),
    "x4x3x": (48, [(48, (4, 6, 8), "trunc_cubocta"), (24, (4, 6), "trunc_octa"), (24, (3, 8), "trunc_cube"),
                   (24, (3, 4), "rhombicubocta"), (12, (3, 4), "cubocta"), (8, (4,), "cube"), (6, (3,), "octa")]),
    "o3x4x": (48, [(24, (3, 8), "trunc_cube"), (12, (3, 4), "cubocta"), (8, (4,), "cube")]),
    "x4x3o": (48, [(24, (3, 8), "trunc_cube"), (12, (3, 4), "cubocta"), (8, (4,), "cube")]),
    "x3o4x": (48, [(24, (3, 4), "rhombicubocta"), (8, (4,), "cube"), (6, (3,), "octa")]),
    "x4o3x": (48, [(24, (3, 4), "rhombicubocta"), (8, (4,), "cube"), (6, (3,), "octa")]),
    "x3x4o": (48, [(24, (4, 6), "trunc_octa"), (12, (3, 4), "cubocta"), (6, (3,), "octa")]),
    "o4x3x": (48, [(24, (4, 6), "trunc_octa"), (12, (3, 4), "cubocta"), (6, (3,), "octa")]),
    "o3o4x": (48, [(8, (4,), "cube")]),
    "x4o3o": (48, [(8, (4,), "cube")]),
    "o3x4o": (48, [(12, (3, 4), "cubocta")]),
    "o4x3o": (48, [(12, (3, 4), "cubocta")]),
    "x3o4o": (48, [(6, (3,), "octa")]),
    "o4o3x": (48, [(6, (3,), "octa")]),
    # H3 family (order 120)
    "x3x5x": (120, [(120, (4, 6, 10), "trunc_icosidodeca"), (60, (5, 6), "trunc_icosidodeca"),
                    (60, (3, 10), "trunc_dodeca"), (60, (3, 4, 5), "rhombicosidodeca"),
                    (30, (3, 5), "icosidodeca"), (20, (5,), "dodeca"), (12, (3,), "icosa")]),
    "x5x3x": (120, [(120, (4, 6, 10), "trunc_icosidodeca"), (60, (5, 6), "trunc_icosidodeca"),
                    (60, (3, 10), "trunc_dodeca"), (60, (3, 4, 5), "rhombicosidodeca"),
                    (30, (3, 5), "icosidodeca"), (20, (5,), "dodeca"), (12, (3,), "icosa")]),
    "o3x5x": (120, [(60, (3, 10), "trunc_dodeca"), (30, (3, 5), "icosidodeca"), (20, (5,), "dodeca")]),
    "x3o5x": (120, [(60, (3, 4, 5), "rhombicosidodeca"), (20, (5,), "dodeca"), (12, (3,), "icosa")]),
    "x3x5o": (120, [(60, (5, 6), "trunc_icosi"), (30, (3, 5), "icosidodeca"), (12, (3,), "icosa")]),
    "x3o5o": (120, [(12, (3,), "icosa")]),
    "o3x5o": (120, [(30, (3, 5), "icosidodeca")]),
    "o3o5x": (120, [(20, (5,), "dodeca")]),
}

# Prism patterns for "." (product) diagrams
_PRISM_PATTERNS = [
    ("x3x", 12, [(12, (4, 6), "prism6"), (6, (3, 4), "prism3")]),
    ("x4x", 16, [(16, (4, 8), "prism8"), (8, (4,), "prism4")]),
    ("x5x", 20, [(20, (4, 10), "prism10"), (10, (4, 5), "prism5")]),
    ("x3o", 12, [(6, (3, 4), "prism3")]),
    ("o3x", 12, [(6, (3, 4), "prism3")]),
    ("x4o", 16, [(8, (4,), "prism4")]),
    ("o4x", 16, [(8, (4,), "prism4")]),
    ("x5o", 20, [(10, (4, 5), "prism5")]),
    ("o5x", 20, [(10, (4, 5), "prism5")]),
]

# D4 special cases
_D4_PATTERNS = {
    "x3x . *b3x": "x3x3x",
    ". x3x *b3x": "x3x3x",
}


def _build_color_dict(entries, dim, divisor):
    return {(v, ft, dim // divisor): name for v, ft, name in entries}


def get_color_names(rep_str, dim):
    # Direct lookup
    for pattern, (divisor, entries) in _COLOR_NAMES_TABLE.items():
        if pattern in rep_str:
            return _build_color_dict(entries, dim, divisor)

    # D4 special cases
    if "*" in rep_str:
        if rep_str in _D4_PATTERNS:
            return get_color_names(_D4_PATTERNS[rep_str], dim)
        if rep_str == "x . x *b3x":
            return {(8, (4,), dim // 8): "prism4"}
        if (rep_str == "o3x . *b3x"
                or rep_str == ". x3o *b3x"):
            return get_color_names("o3x3x", dim)
        if (rep_str == "x3o . *b3x"
                or rep_str == ". o3x *b3x"):
            return get_color_names("x3o3x", dim)
        if (rep_str == "o3x . *b3o"
                or rep_str == ". x3o *b3o"):
            return {(6, (3,), dim // 6): "octa"}
        if (rep_str == "o3o . *b3x"
                or rep_str == ". o3o *b3x"):
            return {(4, (3,), dim // 4): "tetra"}
        raise ValueError("unknown structure in color finder: " + rep_str)

    # Product diagrams
    if "." in rep_str:
        for pattern, divisor, entries in _PRISM_PATTERNS:
            if pattern in rep_str:
                return _build_color_dict(entries, dim, divisor)
        raise ValueError("unknown structure in color finder: " + rep_str)

    raise ValueError("unknown structure in color finder: " + rep_str)


# Lookup table: group_name -> {(vertex_count, frozenset(face_types)): color_name}
_COLOR_NAME_LOOKUP = {
    "coxA4": {
        (24, frozenset({4, 6})): "trunc_octa",
        (12, frozenset({3, 4})): "cubocta",
        (12, frozenset({4, 6})): "prism6",
        (12, frozenset({3, 6})): "trunc_tetra",
        (6, frozenset({3, 4})): "prism3",
        (6, frozenset({3})): "octa",
        (4, frozenset({3})): "tetra",
    },
    "coxB4": {
        (48, frozenset({4, 6, 8})): "trunc_cubocta",
        (24, frozenset({4, 6})): "trunc_octa",
        (24, frozenset({3, 4})): "rhombicubocta",
        (24, frozenset({3, 8})): "trunc_cube",
        (16, frozenset({4, 8})): "prism8",
        (12, frozenset({4, 6})): "prism6",
        (12, frozenset({3, 6})): "trunc_tetra",
        (8, frozenset({4})): "cube",
        (6, frozenset({3, 4})): "prism3",
        (6, frozenset({3})): "octa",
        (4, frozenset({3})): "tetra",
    },
    "coxD4": {
        (24, frozenset({4, 6})): "trunc_octa",
        (12, frozenset({3, 4})): "cubocta",
        (12, frozenset({4, 6})): "prism6",
        (12, frozenset({3, 6})): "trunc_tetra",
        (6, frozenset({3, 4})): "prism3",
        (6, frozenset({3})): "octa",
        (4, frozenset({3})): "tetra",
        (8, frozenset({4})): "cube",
    },
    "coxF4": {
        (48, frozenset({4, 6, 8})): "trunc_cubocta",
        (24, frozenset({3, 4})): "rhombicubocta",
        (24, frozenset({3, 8})): "trunc_cube",
        (24, frozenset({4, 6})): "trunc_octa",
        (12, frozenset({3, 4})): "cubocta",
        (12, frozenset({4, 6})): "prism6",
        (6, frozenset({3, 4})): "prism3",
        (6, frozenset({3})): "octa",
        (4, frozenset({3})): "tetra",
        (8, frozenset({4})): "cube",
    },
    "coxH4": {
        (120, frozenset({4, 6, 10})): "trunc_icosidodeca",
        (24, frozenset({4, 6})): "trunc_octa",
        (20, frozenset({4, 10})): "prism10",
        (12, frozenset({4, 6})): "prism6",
        (30, frozenset({3, 5})): "icosidodeca",
        (60, frozenset({5, 6})): "trunc_icosa",
        (60, frozenset({3, 10})): "trunc_dodeca",
        (60, frozenset({3, 4, 5})): "rhombicosidodeca",
        (10, frozenset({4, 5})): "prism5",
        (6, frozenset({3, 4})): "prism3",
        (6, frozenset({3})): "octa",
        (4, frozenset({3})): "tetra",
        (12, frozenset({3, 6})): "trunc_tetra",
        (12, frozenset({3, 4})): "cubocta",
        (12, frozenset({3})): "icosa",
        (20, frozenset({5})): "dodeca",
    },
}


def get_color_name(group_name, verts, selected_faces, crystal=False):
    v = len(verts)
    face_types = frozenset(len(face) for face in selected_faces)
    group_lookup = _COLOR_NAME_LOOKUP.get(group_name, {})
    color = group_lookup.get((v, face_types), "drawing")
    if crystal:
        color = "crystal_" + color
    return color


def create_materials(cd_string, conjugacy_classes, group_size, crystal, **kwargs):
    materials = []
    cd_dia = CoxeterDynkinDiagram(cd_string)
    sub_diagrams = cd_dia.get_subdiagrams()[1]
    structures = []
    color_names = []
    structure_dict = {}
    for sub_dia in sub_diagrams[3]:
        max_dia = sub_dia.get_maximal_diagram_string()
        n_vert = CoxeterDynkinDiagram(max_dia).get_vertex_count()
        n_class_elements = group_size // n_vert

        key = (sub_dia.get_vertex_count(), n_class_elements)
        if key in structure_dict:
            # for CoxF4 the key is not unique for x3o4x3x
            # dirty workaround, you have to make sure that the colors are attributed correctly
            if isinstance(structure_dict[key], list):
                structure_dict[key] = structure_dict[key] + [sub_dia.diagram_string]
            else:
                structure_dict[key] = [structure_dict[key], sub_dia.diagram_string]
        else:
            structure_dict[key] = sub_dia.diagram_string

    if cd_string == "x3x3x *b3x":
        # swap two entries in one of the value lists
        lst = structure_dict[(24, 8)]
        lst[1], lst[0], lst[2] = lst[0], lst[1], lst[2]
        structure_dict[(24, 8)] = lst
    for i, (rep, members) in enumerate(conjugacy_classes.items()):
        # key: (number of vertices, number of class elements)
        key = (len(rep), len(members))
        while key not in structure_dict:
            key = (key[0], key[1] // 2)
            if key[1] < 1:
                raise "No structure found for conjugacy class"
        rep_str = structure_dict[key]
        if isinstance(rep_str, list):
            color_strings = get_color_names(rep_str[0], group_size)
            structure_dict[key].remove(rep_str[0])
        else:
            color_strings = get_color_names(rep_str, group_size)
        structure = list(color_strings.keys())
        colors = list(color_strings.values())
        if crystal:
            colors = ["crystal_" + col for col in colors]
        colors = ["transparent"] + colors
        structures.append(structure)
        color_names.append(colors)
        # remove other potential names from the kwargs
        if "name" in kwargs:
            kwargs.pop("name")
        materials.append(mix_texture(colors=colors, name=cd_string + str(i), **kwargs))

    return color_names, materials, structures


class NetMorpher:
    def __init__(self, group, sequence, **kwargs):
        self.kwargs = kwargs
        self.unfolders = [Unfolder(group, sig, **kwargs) for sig in sequence]
        self.face_maps = []
        self.vertex_maps = []
        self.morph_maps = []
        self.build_face_tree_from_sequence()
        self.initial_rotation = get_from_kwargs(kwargs, "rotation_quaternion", Quaternion())
        self.initial_location = get_from_kwargs(kwargs, "location", Vector([0, 0, 0]))
        self.initial_scale = get_from_kwargs(kwargs, "scale", [1, 1, 1])
        self.bob = None

    def build_face_tree_from_sequence(self):
        """
        insert a morph sequence from large to small
        """

        # try to build the net without actually creating the bob
        last = self.unfolders[-1]

        unfolder_index = -2
        while unfolder_index >= -len(self.unfolders):
            next_unfolder = self.unfolders[unfolder_index]

            # relate physical vertices of both polyhedra through their closeness
            target2src_map = next_unfolder.create_map(last)
            self.vertex_maps.insert(0, target2src_map)

            # find matching root face
            next_unfolder.root_face_index = next_unfolder.find_root(last.root, target2src_map)
            # pullback existing faces
            tmp_root, indices, src2target_face_map = next_unfolder.pullback(last.root, target2src_map, [], {})
            # add buffer faces to fill gaps
            tmp_root, indices, src2target_face_map = next_unfolder.fill_gaps(tmp_root, indices, src2target_face_map)
            # add left over faces as leaves
            tmp_root, indices, src2target_face_map = next_unfolder.extend_missing_faces(tmp_root, indices,
                                                                                        src2target_face_map)
            next_unfolder.root = tmp_root

            self.face_maps.insert(0, src2target_face_map)
            unfolder_index -= 1
            last = next_unfolder

    def create_net_sequence(self):
        bobs = []
        for unfolder in self.unfolders:
            bobs.append(unfolder.create_net(**self.kwargs))
        return bobs

    def morph_sequence(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, pause=DEFAULT_ANIMATION_TIME / 2,
                       **kwargs):
        kwargs = kwargs | self.kwargs
        self.create_net_sequence()
        current_unfolder = self.unfolders[0]
        largest_net = current_unfolder.create_net(recompute=False, location=self.initial_location,
                                                  rotation_quaternion=self.initial_rotation,
                                                  scale=self.initial_scale, **kwargs)
        self.bob = largest_net
        largest_net.appear(begin_time=begin_time, transition_time=transition_time)

        self._center(largest_net)
        self._rotate(largest_net, normal_to=Vector([0, -1, 0]))
        number_of_vertices = len(current_unfolder.unfolded_vertices)
        transformations = []

        unfolder_index = 1

        while unfolder_index < len(self.unfolders):
            next_unfolder = self.unfolders[unfolder_index]
            src2target_face_map = self.face_maps[unfolder_index - 1]
            src2target_vertex_map = self.vertex_maps[unfolder_index - 1]

            # compute morph map
            def morph_map_to_target(unfolded_source_index, previous_maps=self.morph_maps):
                """
                index is an unfolded original index. A possible pre-image has to be found (it shouldn't matter, which one)
                since we don't have individual faces, we have to trace its origin by search
                The origin can be mapped to the target. The target will be mapped to the target unfolded vertex.
                Here it is important, what kind of face is chosen (the face map should be helpful)
                """

                # apply existing maps
                for morph_map in previous_maps:
                    unfolded_source_index = morph_map[unfolded_source_index]

                # compute current morph transformation
                src_index = current_unfolder.unfolded2vertex_map[unfolded_source_index]
                target_index = src2target_vertex_map[src_index]
                # detect a source face that hosts the src_index
                for face, local_map in current_unfolder.vertex2unfolded_map.items():
                    if local_map.get(src_index, -1) == unfolded_source_index:
                        break

                # find target face
                target_faces = src2target_face_map[face]
                local_vertex2unfolded_map = {}
                for target_face in target_faces:
                    local_vertex2unfolded_map |= next_unfolder.vertex2unfolded_map[target_face]

                unfolded_target_index = local_vertex2unfolded_map.get(target_index, -1)
                if unfolded_target_index != -1:
                    print("morphed ", unfolded_source_index, "->", unfolded_target_index, " best face: ", face)
                    return unfolded_target_index
                else:
                    # find face node that it belongs to
                    host_node = current_unfolder.find_node(current_unfolder.root, unfolded_source_index)
                    parent_node = None
                    if host_node is not None:
                        parent_node = host_node.parent

                    # return closed position of parent face
                    diff = np.inf
                    min_pos = None
                    min_index = -1
                    source_pos = current_unfolder.unfolded_vertices[unfolded_source_index]

                    if parent_node is not None:
                        parent_id = parent_node.name[0]
                        parent_indices = parent_node.name[2]
                        local_src2unfolded_map = current_unfolder.vertex2unfolded_map[parent_id]
                        parent_unfolded_indices = [local_src2unfolded_map[i] for i in parent_indices]
                        parent_positions = [current_unfolder.unfolded_vertices[i] for i in parent_unfolded_indices]

                        for parent_index, pos in zip(parent_unfolded_indices, parent_positions):
                            _diff = (source_pos - pos).length
                            if _diff < diff:
                                diff = _diff
                                min_pos = pos
                                min_index = parent_index
                        if min_pos is not None:
                            print("mapping: ", unfolded_source_index, min_index)
                            return morph_map_to_target(min_index)
                        else:
                            return min_index
                    else:
                        return unfolded_source_index

            morph_map = {}
            for idx in range(number_of_vertices):
                morph_map[idx] = morph_map_to_target(idx)

            self.morph_maps.append(morph_map)

            def morphing(index, unfolder=next_unfolder, mmap=morph_map):
                return unfolder.unfolded_vertices[mmap[index]]

            transformations.append(lambda index, transformation=partial(morphing, unfolder=next_unfolder,
                                                                        mmap=morph_map.copy()): transformation(index))

            unfolder_index += 1
            current_unfolder = next_unfolder

        largest_net.index_transform_mesh(transformations,
                                         begin_time=begin_time + transition_time + pause,
                                         transition_time=transition_time)

        self.morph_colors(largest_net, 1, begin_time=begin_time + transition_time + pause,
                          transition_time=transition_time)

        for i in range(2, unfolder_index):
            largest_net.transform_mesh_to_next_shape2(begin_time=begin_time + i * (transition_time + pause),
                                                      transition_time=transition_time)

            self.morph_colors(largest_net, i, begin_time=begin_time + i * (transition_time + pause),
                              transition_time=transition_time)

        return begin_time + unfolder_index * (transition_time + pause)

    def unmorph_sequence(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.bob.transform_mesh_to_first_shape(begin_time=begin_time, transition_time=transition_time)
        self.unmorph_colors(self.bob, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def morph_colors(self, bob, to_index, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        to_unfolder = self.unfolders[to_index]

        # build face_map upto this transformation for faces that are mapped uniquely
        inverse_face_map = {}
        first = True

        unique_face_map = {}

        for fm in self.face_maps[:to_index][::-1]:
            for key, value in fm.items():
                value = list(value)
                if len(value) == 1:
                    pre_image = inverse_face_map.get(value[0], [])
                    pre_image.append(key)
                inverse_face_map[value[0]] = pre_image

            if first:
                unique_face_map = dict((key, value[0]) for key, value in inverse_face_map.items() if len(value) == 1)
                first = False
            else:
                tmp_unique_face_map = dict(
                    (key, value[0]) for key, value in inverse_face_map.items() if len(value) == 1)
                new_unique_face_map = {}
                for key, value in unique_face_map.items():
                    if value in tmp_unique_face_map.keys():
                        new_unique_face_map[key] = tmp_unique_face_map[value]
                unique_face_map = new_unique_face_map

        # take care of changing colors
        face_class_map = {}
        # face classes defined from the largest polyhedron
        first_unfolder = self.unfolders[0]
        face_classes = first_unfolder.face_classes

        for target, src in unique_face_map.items():
            target_size = len(to_unfolder.faces[target])
            src_size = len(first_unfolder.faces[src])
            for class_index, face_class in enumerate(face_classes.values()):
                src_face = MeshFace(first_unfolder.faces[src])
                if src_face in face_class:
                    face_class_map[class_index] = (src_size, target_size)
                    break

        for slot, (src, target) in face_class_map.items():
            if src != target:
                if target in [6, 8, 10]:
                    ibpy.adjust_mixer(bob, slot, from_value=1, to_value=0, begin_time=begin_time,
                                      transition_time=transition_time)
                else:
                    ibpy.adjust_mixer(bob, slot, from_value=0, to_value=1, begin_time=begin_time,
                                      transition_time=transition_time)

    def unmorph_colors(self, bob, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        first_unfolder = self.unfolders[0]
        face_classes = first_unfolder.face_classes

        for slot in range(len(face_classes)):
            ibpy.adjust_mixer(bob, slot, from_value=1, to_value=0, begin_time=begin_time,
                              transition_time=transition_time)

        return begin_time + transition_time

    def get_state(self, state=0, **kwargs):
        state_bob = self.unfolders[state].create_net(recompute=False,
                                                     location=self.initial_location,
                                                     scale=self.initial_scale,
                                                     rotation_quaternion=self.initial_rotation,
                                                     **kwargs)
        self._center(state_bob)
        self._rotate(state_bob)
        return state_bob

    def get_quadrupole(self, state=0):
        """
        compute the quadrupole moment of the mass distribution
        """
        unfolder = self.unfolders[state]
        vertex_list = [v for v in unfolder.unfolded_vertices]
        for v in vertex_list:
            v[1] = 0
        start = np.array([0] * 9)
        start = start.reshape(3, 3)
        identity = np.identity(3, float)
        quadrupole = sum([3 * np.tensordot(v, v, axes=0) - v.dot(v) * identity for v in vertex_list], start)
        return quadrupole

    def get_main_axis(self, state=0, choice="MAXIMUM"):
        quadrupole = self.get_quadrupole(state)
        ev, vectors = np.linalg.eig(quadrupole)
        abs_evs = np.abs(ev)
        if choice == "MAXIMUM":
            ev0 = max(abs_evs)
        if choice == "MINIMUM":
            ev0 = min(abs_evs)
        idx0 = list(abs_evs).index(ev0)
        return vectors.transpose()[idx0]

    def get_main_axes_and_eigenvalues(self, state=0):
        quadrupole = self.get_quadrupole(state)
        ev, vectors = np.linalg.eig(quadrupole)
        return vectors.transpose(), ev

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.bob.disappear(begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def _center(self, bob):
        vertices = self.unfolders[0].unfolded_vertices
        center = sum(vertices, Vector()) / len(vertices)
        bob.move_to(self.initial_location - center, transition_time=0)

    def _rotate(self, bob, normal_to=Vector([0, -1, 0])):
        normal = self.unfolders[0].root.name[3]
        rotation = align_vectors(normal, normal_to)
        rotation_matrix = Matrix(rotation)
        rotation_quaternion = self.initial_rotation @ rotation_matrix.to_quaternion()

        bob.rotate(rotation_quaternion=rotation_quaternion, transition_time=0)


class NetMorpher4D:
    def __init__(self, group, sequence, **kwargs):
        self.name = get_from_kwargs(kwargs, "name", "NetMorpher4D")
        self.crystal = get_from_kwargs(kwargs, "crystal", None)
        # self.unfolders = [Unfolder4D2(group, sig) for sig in sequence]
        self.unfolders = [Unfolder4D2(group, sig, rotate_to_south_pole=False) for sig in sequence]

        self.cell_maps = []
        self.vertex_maps = []
        self.morph_maps = []
        self.build_face_tree_from_sequence()
        self.initial_rotation = get_from_kwargs(kwargs, "rotation_quaternion", Quaternion())
        self.initial_location = get_from_kwargs(kwargs, "location", Vector([0, 0, 0]))
        self.initial_scale = get_from_kwargs(kwargs, "scale", [1, 1, 1])
        self.bob = None

        # create colors for conjugacy classes
        self.group = group(path="../mathematics/geometry/data")
        self.group_size = self.group.size
        self.conjugacy_classes = self.group.get_cells_in_conjugacy_classes(signature=sequence[0])
        self.materials, self.material_structures = self._create_materials(sequence[0], crystal=self.crystal)
        self.class_representatives = None
        self.class_cells_dictionary = DefaultDict(list)

    def build_face_tree_from_sequence(self):
        """
        insert a morph sequence from large to small
        """

        # try to build the net without actually creating the bob
        last_unfolder = self.unfolders[-1]

        unfolder_index = -2
        while unfolder_index >= -len(self.unfolders):
            next_unfolder = self.unfolders[unfolder_index]
            (next_unfolder.root,
             target2src_map,
             src2target_cell_map) = next_unfolder.pullback_tree(last_unfolder)

            self.vertex_maps.insert(0, target2src_map)
            self.cell_maps.insert(0, src2target_cell_map)
            unfolder_index -= 1
            last_unfolder = next_unfolder

    def create_net_sequence(self):
        bobs = []
        for unfolder in self.unfolders:
            bobs.append(unfolder.create_net(name=self.name))
        return bobs

    def morph_sequence(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, pause=DEFAULT_ANIMATION_TIME / 2,
                       appear_time=None, **kwargs):
        sequentially = get_from_kwargs(kwargs, "sequentially", False)
        self.create_net_sequence()
        largest_unfolder = self.unfolders[0]
        largest_net = largest_unfolder.create_net(recompute=False, location=self.initial_location,
                                                  rotation_quaternion=self.initial_rotation,
                                                  scale=self.initial_scale, **kwargs)
        self.bob = largest_net
        if appear_time is None:
            largest_net.appear(begin_time=begin_time, transition_time=transition_time, sequentially=True)
        else:
            largest_net.appear(begin_time=appear_time, transition_time=pause, sequentially=True)

        self._center(largest_net)
        # self._rotate(largest_net, normal_to=Vector([0, -1, 0, 0]))

        conjugacy_class_members = DefaultDict(list)
        conjugacy_class_face_transformations = {}

        # compute grow children sequentially
        if sequentially:
            # cell2child_map
            cell2child_map = {}
            for child_index, child in enumerate(largest_net.b_children):
                child_info_map = largest_unfolder.child_info_map[child_index]
                (cell_vertex_id2src_map, cell_faces, cell_color, cell_modifier, cell_index,
                 conjugacy_class_index, cell_vertex_id2unfolded_map) = child_info_map
                cell2child_map[cell_index] = child_index

            count = 0
            dt = transition_time / len(largest_net.b_children)
            nodes = largest_unfolder.get_nodes_sorted_by_level(largest_unfolder.root)
            for node in nodes:
                cell_index = node.name[0]
                largest_net.b_children[cell2child_map[cell_index]].grow(begin_time=begin_time + count * dt,
                                                                        transition_time=dt)
                count += 1

        for child_index, child in enumerate(largest_net.b_children):

            child_info_map = largest_unfolder.child_info_map[child_index]
            cell_vertex_id2src_map, cell_faces, cell_color, cell_modifier, cell_index, conjugacy_class_index, cell_vertex_id2unfolded_map = child_info_map
            self.class_cells_dictionary[conjugacy_class_index] += [child_index]
            transformations = []
            unfolder_index = 1
            child_morph_maps = []
            number_of_vertices = len(cell_vertex_id2src_map)

            # override material with morphable material
            # which the regular unfolder net has no idea of

            material = self.materials[conjugacy_class_index]
            ibpy.set_material(child, material, slot=0)

            while unfolder_index < len(self.unfolders):
                next_unfolder = self.unfolders[unfolder_index]

                # since we always map from the largest map, we have to compose the cell and vertex maps
                src2target_vertex_map = self.vertex_maps[0].copy()
                src2target_cell_map = self.cell_maps[0].copy()
                for i in range(1, unfolder_index):
                    for key, val in src2target_vertex_map.items():
                        src2target_vertex_map[key] = self.vertex_maps[i][val]
                    for key, vals in src2target_cell_map.items():
                        targets = []
                        for val in vals:
                            targets += self.cell_maps[i][val]
                        src2target_cell_map[key] = targets

                def morph_map_to_target(source_index, unfolded_index=-1, from_mesh=True, level=0):
                    """
                    index is an unfolded original index. A possible pre-image has to be found (it shouldn't matter which one)
                    since we don't have individual faces, we have to trace its origin by search
                    The origin can be mapped to the target. The target will be mapped to the target unfolded vertex.
                    Here it is important, what kind of face is chosen (the face map should be helpful)
                    """

                    # different to the 3D case, we try to morph each time from the largest net
                    # the mesh doesn't lose vertices during morphing, so we always can think of the vertex indices
                    # as if they belong to the largest net

                    # compute current morph transformation
                    # src_index = current_unfolder.unfolded2vertex_map[unfolded_source_index]
                    if from_mesh:
                        try:
                            src_index = cell_vertex_id2src_map[source_index]
                            unfolded_index = cell_vertex_id2unfolded_map[source_index]
                        except KeyError:
                            print("source index or unfolder index not found: ", source_index)
                            return source_index
                    else:
                        src_index = source_index
                    target_index = src2target_vertex_map[src_index]

                    # it is not obvious which unfolded vertex the given target_index should be mapped to,
                    # therefore, it is necessary to identify the correct target cell.

                    # find possible target cells
                    target_cells = src2target_cell_map[cell_index]

                    local_vertex2unfolded_map = {}
                    for target_cell in target_cells:
                        local_vertex2unfolded_map |= next_unfolder.vertex2unfolded_map[target_cell]
                    unfolded_target_index = local_vertex2unfolded_map.get(target_index, -1)

                    if unfolded_target_index != -1:
                        # print("morphed ", source_index, "->", unfolded_target_index, " best cell: ", cell_index)
                        return unfolded_target_index
                    else:
                        # find the cell node that it belongs to
                        host_node = largest_unfolder.find_node(largest_unfolder.root, unfolded_index)
                        parent_node = None
                        if host_node is not None:
                            parent_node = host_node.parent

                        # find common points between parent and host
                        common = host_node.name[2].intersection(parent_node.name[2])

                        # find the closest common point
                        min_diff = np.inf
                        min_index = -1
                        source_pos = largest_unfolder.unfolded_vertices[unfolded_index]

                        largest_src2unfolded = largest_unfolder.vertex2unfolded_map[cell_index]

                        for common_src_index in common:
                            common_pos = largest_unfolder.unfolded_vertices[largest_src2unfolded[common_src_index]]
                            diff = (source_pos - common_pos).length
                            if diff < min_diff:
                                min_diff = diff
                                min_index = common_src_index

                        # try to find target location for parent cell
                        if level < 5:
                            return morph_map_to_target(min_index, largest_src2unfolded[min_index],
                                                       from_mesh=False, level=level + 1)
                        else:
                            return 0

                morph_map = {}
                for idx in range(number_of_vertices):
                    morph_map[idx] = morph_map_to_target(idx)

                child_morph_maps.append(morph_map)

                def morphing(ids, unfolder=next_unfolder, mmap=morph_map):
                    try:
                        target = unfolder.unfolded_vertices[mmap[ids]]
                    except KeyError:
                        print("key error, mapping error: ", ids, mmap[ids])
                        return Vector()
                    except IndexError:
                        print("index error, mapping error: ", ids, mmap[ids])
                        return Vector()
                    return target

                transformations.append(lambda ii, trafo=partial(morphing, unfolder=next_unfolder,
                                                                mmap=morph_map.copy()): trafo(ii))

                unfolder_index += 1
                #current_unfolder = next_unfolder

            # apply transformations to child
            child.index_transform_mesh(transformations,
                                       begin_time=begin_time + transition_time + pause,
                                       transition_time=transition_time)

            for i in range(2, unfolder_index):
                child.transform_mesh_to_next_shape2(begin_time=begin_time + i * (transition_time + pause),
                                                    transition_time=transition_time)

            # work out material changes for the given transformation if it is the first child in the
            # conjugacy class

            conjugacy_class_index = child_info_map[5]
            members = conjugacy_class_members[conjugacy_class_index]
            if len(members) == 0:
                members.append(child_index)
                cell_faces = child_info_map[1]
                face_transformations = [cell_faces]
                for transformation in transformations:
                    # transform each face index into target points and identify equal points
                    morphed_cell_faces = []
                    morphed_cell_vertices = []
                    for face in cell_faces:
                        morphed_face = set()
                        for idx in face:
                            index = -1
                            morphed_vertex = transformation(idx)
                            for i, v in enumerate(morphed_cell_vertices):
                                if (morphed_vertex - v).length < 0.01:
                                    index = i
                                    break
                            if index == -1:
                                index = len(morphed_cell_vertices)
                                morphed_cell_vertices.append(morphed_vertex)
                            morphed_face.add(index)
                        if len(morphed_face) > 2:
                            morphed_cell_faces.append(morphed_face)
                    face_transformations.append(morphed_cell_faces)
                conjugacy_class_face_transformations[conjugacy_class_index] = face_transformations

        # since the colors only depend on the conjugacy class we can perform the morphing of the material
        # independent of the children

        conjugacy_class_sizes = [len(members) for members in self.conjugacy_classes.values()]
        for (class_index, members) in conjugacy_class_members.items():
            # print("class ",class_index)
            class_face_transformations = conjugacy_class_face_transformations[class_index]

            shader_factors = ibpy.get_all_nodes_from_shader(self.materials[class_index], "FinalMixShader")
            # start with the first color
            old_shader_factors = [0] * len(shader_factors)

            # appear (first shader factor is set to one all others are set to zero
            for shader_factor_index, shader_factor in enumerate(shader_factors):
                # keep it invisible until the time of appearance
                if appear_time is None:
                    ibpy.change_default_value(shader_factor.inputs["Factor"], from_value=0, to_value=0, begin_time=0,
                                              transition_time=begin_time)
                else:
                    ibpy.change_default_value(shader_factor.inputs["Factor"], from_value=0, to_value=0, begin_time=0,
                                              transition_time=appear_time)

            t0 = begin_time
            print(class_face_transformations)
            for t, morphed_selected_faces in enumerate(class_face_transformations):

                all_indices = set(sum([list(index_set) for index_set in morphed_selected_faces], []))
                face_types = set()
                for face in morphed_selected_faces:
                    face_types.add(len(face))
                face_types = tuple(sorted(list(face_types)))

                if sequentially:
                    dt = 0
                else:
                    dt = transition_time
                if appear_time is None:
                    ibpy.change_default_value(shader_factors[0].inputs["Factor"], from_value=old_shader_factors[0],
                                              to_value=1, begin_time=t0, transition_time=dt)
                else:
                    ibpy.change_default_value(shader_factors[0].inputs["Factor"], from_value=old_shader_factors[0],
                                              to_value=1, begin_time=appear_time, transition_time=min(dt, pause))
                old_shader_factors[0] = 1

                structure = (len(all_indices), face_types,
                             conjugacy_class_sizes[class_index])
                if structure[0] > 3 and structure in self.material_structures[class_index]:
                    color_position = self.material_structures[class_index].index(
                        structure) + 1  # account for the transparent color mixer
                    alpha = 1
                else:
                    color_position = -1
                    alpha = 0

                # print("color_position:",color_position)
                # print("old_shader_factors: ",old_shader_factors,t0)

                if alpha == 0:
                    shader_factor = shader_factors[0]
                    ibpy.change_default_value(shader_factor.inputs["Factor"],
                                              from_value=old_shader_factors[0],
                                              to_value=0, begin_time=t0, transition_time=transition_time)
                    old_shader_factors[0] = 0
                    for idx in self.class_cells_dictionary[class_index]:
                        # hide all cells of this class
                        largest_net.b_children[idx].make_invisible(begin_time=t0 + transition_time)

                else:
                    shader_factor = shader_factors[0]
                    ibpy.change_default_value(shader_factor.inputs["Factor"],
                                              from_value=old_shader_factors[0],
                                              to_value=1, begin_time=t0, transition_time=transition_time)
                    old_shader_factors[0] = 1
                    if shader_factor == 0:
                        for idx in self.class_cells_dictionary[class_index]:
                            largest_net.b_children[idx].make_visible(begin_time=t0 + transition_time)

                for shader_factor_index in range(1,
                                                 len(shader_factors)):  #exclude first shader_factor that handles transparency
                    shader_factor = shader_factors[shader_factor_index]
                    if shader_factor_index >= color_position:
                        ibpy.change_default_value(shader_factor.inputs["Factor"],
                                                  from_value=old_shader_factors[shader_factor_index],
                                                  to_value=0, begin_time=t0, transition_time=transition_time)
                        old_shader_factors[shader_factor_index] = 0
                    else:
                        ibpy.change_default_value(shader_factor.inputs["Factor"],
                                                  from_value=old_shader_factors[shader_factor_index],
                                                  to_value=1, begin_time=t0, transition_time=transition_time)
                        old_shader_factors[shader_factor_index] = 1
                t0 = t0 + transition_time + pause

        nt = len(transformations)
        self.bob = largest_net
        return begin_time + unfolder_index * (transition_time + pause)

    def get_state(self, state=0, **kwargs):
        state_bob = self.unfolders[state].create_net(recompute=False,
                                                     location=self.initial_location,
                                                     scale=self.initial_scale,
                                                     rotation_quaternion=self.initial_rotation,
                                                     **kwargs)
        self._center(state_bob)
        self._rotate(state_bob)
        return state_bob

    def get_center(self, state=0):
        unfolder = self.unfolders[state]
        vertex_list = unfolder.unfolded_vertices
        center = sum(vertex_list, Vector()) / len(vertex_list)
        return center

    def get_quadrupole(self, state=0):
        """
        compute the quadrupole moment of the mass distribution
        """
        unfolder = self.unfolders[state]
        vertex_list = [np.array(v[0:3]) for v in unfolder.unfolded_vertices]
        start = np.array([0] * 9)
        start = start.reshape(3, 3)
        identity = np.identity(3, float)
        quadrupole = sum([3 * np.tensordot(v, v, axes=0) - v.dot(v) * identity for v in vertex_list], start)
        return quadrupole

    def get_unfolded_vertices(self, state=0):
        return self.unfolders[state].unfolded_vertices

    def get_main_axis(self, state=0, choice="MAXIMUM"):
        quadrupole = self.get_quadrupole(state)
        ev, vectors = np.linalg.eig(quadrupole)
        abs_evs = np.abs(ev)
        if choice == "MAXIMUM":
            ev0 = max(abs_evs)
        if choice == "MINIMUM":
            ev0 = min(abs_evs)
        idx0 = list(abs_evs).index(ev0)
        return vectors.transpose()[idx0]

    def get_main_axes_and_eigenvalues(self, state=0):
        quadrupole = self.get_quadrupole(state)
        ev, vectors = np.linalg.eig(quadrupole)
        return vectors.transpose(), ev

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.bob.disappear(begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def _center(self, bob):
        vertex_list = self.unfolders[0].unfolded_vertices
        center = sum(vertex_list, Vector()) / len(vertex_list)
        bob.move_to(to_vector(self.initial_location) - center, transition_time=0)

    def _rotate(self, bob, normal_to=Vector([0, -1, 0, 0])):
        normal = self.unfolders[0].root.name[3]
        rotation = align_vectors(normal, normal_to)
        rotation_matrix = Matrix(rotation)
        rotation_quaternion = self.initial_rotation @ rotation_matrix.to_quaternion()

        bob.rotate(rotation_quaternion=rotation_quaternion, transition_time=0)

    def _create_materials(self, cd_string, crystal=False, **kwargs):
        materials = []
        structures = []

        cd_dia = CoxeterDynkinDiagram(cd_string)
        sub_diagrams = cd_dia.get_subdiagrams()[1]

        structure_dict = {}
        for sub_dia in sub_diagrams[3]:
            structure_dict[sub_dia.get_vertex_count()] = sub_dia.diagram_string

        for i, (rep, members) in enumerate(self.conjugacy_classes.items()):
            rep_str = structure_dict[len(rep)]
            color_strings = get_color_names(rep_str, self.group_size)
            # print(rep, color_strings)
            if crystal:
                colors = ["transparent"] + ["crystal_" + col for col in list(color_strings.values())]
            else:
                colors = ["transparent"] + list(color_strings.values())
            structure = list(color_strings.keys())
            materials.append(mix_texture(colors=colors, name=cd_string + str(i), **kwargs))
            structures.append(structure)
        return materials, structures


class Morpher4D:
    def __init__(self, group, sequence, **kwargs):

        self.radius = get_from_kwargs(kwargs, "radius", 5)
        self.crystal = get_from_kwargs(kwargs, "crystal", False)
        self.kwargs = kwargs
        self.mode = get_from_kwargs(kwargs, "mode", "CLOSEST_TO_POLE")
        self.cell_removals = get_from_kwargs(kwargs, "cell_removals", [])

        if callable(group):
            self.group = group(path="../mathematics/geometry/data")
        else:
            self.group = group
        self.group_name = self.group.name
        self.group_size = self.group.size

        self.signatures = [eval(str(group)[-7:-2].upper() + "_SIGNATURES", {}, namespace)[seq.strip()] for seq in
                           sequence]

        point_clouds = [self.group.get_real_point_cloud(sig) for sig in self.signatures]
        cells = [self.group.get_cells(sig) for sig in self.signatures]
        faces = [self.group.get_faces(sig) for sig in self.signatures]
        self.cells = cells[0]
        self.faces = faces[0]

        # scale all point clouds to radius
        radii = [np.sqrt(v[0].dot(v[0])) for v in point_clouds]
        scales = [self.radius / r for r in radii]
        self.point_clouds = [[v * scale for v in cloud] for (scale, cloud) in zip(scales, point_clouds)]
        self.vs = [v for v in self.point_clouds[0]]
        self.centers = [self.center(cell, self.vs) for cell in self.cells]

        self.south_pole = get_from_kwargs(kwargs, "south_pole", 1)
        if self.south_pole == 1:
            south_pole = Vector([0, -1, 0, 0])
        elif self.south_pole == 0:
            south_pole = Vector([-1, 0, 0, 0])
        elif self.south_pole == 2:
            south_pole = Vector([0, 0, -1, 0])
        elif self.south_pole == 3:
            south_pole = Vector([0, 0, 0, -1])

        if self.mode == "CLOSEST_TO_POLE":
            # find center closest to west-pole (the west-pole is not a mirror intersection and therefore, no vertex should ever land on it)
            # no troubles with stereographic projection (hopefully)
            closest = -np.inf
            farthest = np.inf
            for center, cell in zip(self.centers, cells[0]):
                close = (center / center.length).dot(south_pole)
                # print(close, center, cell)
                if close > closest:
                    closest = close
                    closest_cell = cell
                    closest_center = center
                if close < farthest:
                    farthest = close
                    farthest_cell = cell
                    farthest_center = center
            # remove outside cell
            del self.cells[farthest_cell]
            rotation = Matrix(align_vectors(closest_center, south_pole))
        elif self.mode == "CELL_SIZE":
            skip = get_from_kwargs(kwargs, "skip", 0)
            cell_size = get_from_kwargs(kwargs, "cell_size", 20)
            for center, (cell, normal) in zip(self.centers, cells[0].items()):
                if len(cell) == cell_size:
                    if skip == 0:
                        closest_center = center
                        center_cell = cell
                        center_cell_normal = normal
                        break
                    else:
                        skip -= 1
            rotation = find_optimal_rotation(center_cell, center_cell_normal, south_pole, self.faces, self.vs)

        print("external rotation: ", rotation)
        # rotate vertex data
        self.point_clouds = [[rotation @ v for v in cloud] for cloud in self.point_clouds]

        self.vs = [v for v in self.point_clouds[0]]
        self.stereo = self._stereo_projection(self.vs)
        self.bob = None

        # create colors for conjugacy classes
        self.conjugacy_classes = self.group.get_cells_in_conjugacy_classes(signature=sequence[0])
        self.material_names, self.materials, self.material_structures = create_materials(sequence[0],
                                                                                         self.conjugacy_classes,
                                                                                         self.group_size, self.crystal,
                                                                                         **kwargs)

        self.class_representatives = None

    def create_bob(self, scale_elements=1, **kwargs):
        half_way = get_from_kwargs(kwargs, "halfway", True)
        limit_direction = get_from_kwargs(kwargs, "limit_direction", Vector([0, 0, -1]))
        limit = get_from_kwargs(kwargs, "limit", 0)

        cell_removal_list = self.cell_removals

        children = []
        child_index = 0
        child_info_map = {}
        modifier_map = {}

        for cell_index, cell in enumerate(self.cells):
            conjugacy_class_index = -1
            sorted_indices = tuple(sorted(cell))
            for class_index, (rep, members) in enumerate(self.conjugacy_classes.items()):
                if sorted_indices in members:
                    conjugacy_class_index = class_index
                    break

            verts, center = self._stereo_projection_cell(cell, scale=scale_elements)

            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in self.faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            color = get_color_name(self.group_name, verts, selected_faces, self.crystal)
            cellobj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces, orient_faces=True),
                              name="Cell" + str(cell_index), color=self.materials[conjugacy_class_index])
            if self.crystal:
                # if color in modifier_map:
                #     mod = modifier_map[color]
                # else:
                mod = CrystalModifier(material=get_texture(color))
                modifier_map[color] = mod
                cellobj.add_mesh_modifier(type="NODES", node_modifier=mod)
                mod.transfer_material_from(cellobj)
                modifier = mod
            else:
                modifier = None
            if not half_way:
                children.append(cellobj)
                child_info_map[child_index] = ({j: i for j, i in enumerate(cell)}, selected_faces,
                                               color,
                                               modifier, conjugacy_class_index)
                child_index += 1
            else:
                distance = center.dot(limit_direction)
                if cell_index not in cell_removal_list:
                    if -limit <= distance:
                        # only show the first layer in the negative direction
                        if compute_cell_size(verts) != np.abs(distance):
                            children.append(cellobj)
                            child_info_map[child_index] = ({j: i for j, i in enumerate(cell)}, selected_faces, color,
                                                           modifier, conjugacy_class_index)

                            child_index += 1
        return BObject(children=children, **kwargs), child_info_map

    def center(self, cell, point_cloud):
        cntr = sum([point_cloud[i] for i in cell], Vector([0, 0, 0, 0])) / len(cell)
        return cntr

    def morph(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, pause=DEFAULT_ANIMATION_TIME, **kwargs):
        all_kwargs = kwargs | self.kwargs  # merge kwargs
        grow_sequentially = get_from_kwargs(all_kwargs, "grow_sequentially", False)
        scale_elements = get_from_kwargs(all_kwargs, "scale_elements", 1)
        appear_time = begin_time

        bob, child_info_map = self.create_bob(scale_elements=scale_elements, **all_kwargs)

        bob.appear(begin_time=begin_time, transition_time=0, children=False)
        # children appear later, when the final material has been set
        start_time = begin_time + transition_time + pause  # time for the first morph animation

        src2target_maps = [self._create_map(self.vs, target) for target in self.point_clouds[1:]]
        stereo_targets = [self._stereo_projection(target) for target in
                          self.point_clouds]

        conjugacy_class_members = DefaultDict(list)
        conjugacy_class_face_transformations = {}

        positions = {}

        for child_index, child in enumerate(bob.b_children):
            vector_map_results = []
            index_map_results = []

            mesh2vertex = child_info_map[child_index][0]

            for cloud_index in range(1, len(self.point_clouds)):
                # create src target map
                src2target_map = src2target_maps[cloud_index - 1]
                target = stereo_targets[cloud_index]

                def morph_map_to_target(index):
                    return target[src2target_map[mesh2vertex[index]]]

                def morph_map_to_target_index(index):
                    return src2target_map[mesh2vertex[index]]

                vector_map_result = {}
                index_map_result = {}
                for j in range(len(mesh2vertex)):
                    vector_map_result[j] = morph_map_to_target(j)
                    index_map_result[j] = morph_map_to_target_index(j)
                # scale cells appropriately
                if scale_elements != 1:
                    center = sum(vector_map_result.values(), Vector()) / len(vector_map_result)
                    for j in range(len(mesh2vertex)):
                        vector_map_result[j] = center + (vector_map_result[j] - center) * scale_elements

                vector_map_results.append(vector_map_result)
                index_map_results.append(index_map_result)

            vector_transformations = [lambda index, i=i: vector_map_results[i][index] for i in
                                      range(len(vector_map_results))]
            index_transformations = [lambda index, i=i: index_map_results[i][index] for i in
                                     range(len(index_map_results))]

            if len(vector_transformations) > 0:
                t0 = pause + child.index_transform_mesh(vector_transformations, begin_time=start_time,
                                                        transition_time=transition_time, pause_time=pause)

            for t in range(1, len(vector_transformations)):
                t0 = pause + child.transform_mesh_to_next_shape2(begin_time=t0, transition_time=transition_time,
                                                                 pause_time=pause)

            class_index = child_info_map[child_index][4]
            members = conjugacy_class_members[class_index]
            if len(members) == 0:
                members.append(child_index)
                # store the face transformation data for the first member of each class
                selected_faces = child_info_map[child_index][1]

                face_transformations = [selected_faces]
                for t, index_transformation in enumerate(index_transformations):
                    morphed_selected_faces = []
                    for face in selected_faces:
                        morphed_face = set()
                        for idx in face:
                            morphed_face.add(index_transformations[t](idx))
                        if len(morphed_face) > 2:
                            morphed_selected_faces.append(morphed_face)
                    face_transformations.append(morphed_selected_faces)
                conjugacy_class_face_transformations[class_index] = face_transformations

            ibpy.set_origin(child)
            location = ibpy.get_location(child)
            positions[child] = location.dot(location)

            # sort locations
            if grow_sequentially:
                sorted_positions = {k: v for k, v in sorted(positions.items(), key=lambda item: item[1])}
                dt = (transition_time / 2 + pause) / len(sorted_positions)
                for i, c in enumerate(sorted_positions.keys()):
                    c.grow(begin_time=begin_time + i * dt, transition_time=transition_time / 2)
            else:
                # turn all alphas on at appear-time
                child.appear(begin_time=0, transition_time=0)  # appearance is managed by the colormorphing

        # take care of the color for each representative of the conjugacy classes
        conjugacy_class_sizes = [len(members) for members in self.conjugacy_classes.values()]
        for (class_index, members) in conjugacy_class_members.items():
            # print("class ",class_index)
            class_face_transformations = conjugacy_class_face_transformations[class_index]

            shader_factors = ibpy.get_all_nodes_from_shader(self.materials[class_index], "FinalMixShader")

            # start with the first color for the

            old_shader_factors = [0] * len(shader_factors)
            old_shader_factors[0] = 0

            # appear (first shader factor is set to one all others are set to zero
            for shader_factor_index, shader_factor in enumerate(shader_factors):
                # keep it invisible until the time of appearance
                ibpy.change_default_value(shader_factor.inputs["Factor"], from_value=0, to_value=0, begin_time=0,
                                          transition_time=appear_time)

            t0 = begin_time
            for t, morphed_selected_faces in enumerate(class_face_transformations):

                # compute volume
                # take two vertices from 1 face and one from another
                # compute the determinant
                if len(morphed_selected_faces) == 0:
                    vol = 0
                else:
                    try:
                        target = stereo_targets[t]
                        face0 = morphed_selected_faces[0]
                        i1, i2, i3 = list(face0)[0:3]
                        face1 = morphed_selected_faces[1]
                        diff = set(face1).difference(set(face0))
                        if len(diff) > 0:
                            i4 = diff.pop()

                            vs = [target[i] for i in [i1, i2, i3, i4]]
                            e1 = vs[3] - vs[0]
                            e2 = vs[2] - vs[0]
                            e3 = vs[1] - vs[0]

                            vol = e2.cross(e3).dot(e1)
                        else:
                            vol = 0
                    except Exception as e:
                        print(e)
                        vol = 0

                if np.abs(vol) < 0.001:
                    # print(det, morphed_selected_faces)
                    alpha = 0
                else:
                    alpha = 1
                all_indices = set(sum([list(index_set) for index_set in morphed_selected_faces], []))
                face_types = set()
                for face in morphed_selected_faces:
                    face_types.add(len(face))
                face_types = tuple(sorted(list(face_types)))

                # print("->",len(all_indices),face_types)
                if alpha == 0:
                    # print("vanish: ",alpha)
                    for shader_factor_index, shader_factor in enumerate(shader_factors):
                        ibpy.change_default_value(shader_factor.inputs["Factor"],
                                                  from_value=old_shader_factors[shader_factor_index],
                                                  to_value=0, begin_time=t0, transition_time=transition_time)
                        old_shader_factors[shader_factor_index] = 0
                else:

                    alpha_factor = shader_factors[0]
                    if grow_sequentially:
                        ibpy.change_default_value(alpha_factor.inputs["Factor"], from_value=0, to_value=1,
                                                  begin_time=0, transition_time=0)
                    else:
                        ibpy.change_default_value(alpha_factor.inputs["Factor"], from_value=old_shader_factors[0],
                                                  to_value=1, begin_time=t0, transition_time=transition_time)

                    old_shader_factors[0] = 1
                    key = (len(all_indices), face_types,
                           conjugacy_class_sizes[class_index])
                    print(class_index, key)
                    if key == (24, (4, 6), 8):
                        pass
                    color_position = self.material_structures[class_index].index(
                        key) + 1  # account for the transparent color mixer

                    # print("color_position:",color_position)
                    # print("old_shader_factors: ",old_shader_factors,t0)
                    for shader_factor_index in range(1,
                                                     len(shader_factors)):  # exclude first shader_factor that handles transparency
                        shader_factor = shader_factors[shader_factor_index]
                        if shader_factor_index >= color_position:
                            ibpy.change_default_value(shader_factor.inputs["Factor"],
                                                      from_value=old_shader_factors[shader_factor_index],
                                                      to_value=0, begin_time=t0, transition_time=transition_time)
                            old_shader_factors[shader_factor_index] = 0
                        else:
                            ibpy.change_default_value(shader_factor.inputs["Factor"],
                                                      from_value=old_shader_factors[shader_factor_index],
                                                      to_value=1, begin_time=t0, transition_time=transition_time)
                            old_shader_factors[shader_factor_index] = 1
                    # print("old_shader_factors:",old_shader_factors,t0+transition_time)
                t0 = t0 + transition_time + pause

        nt = len(vector_transformations)
        self.bob = bob
        return begin_time + nt * (transition_time + pause)

    def _stereo_projection(self, src):
        dims = set(range(4))
        reduced_dims = dims.difference(set([self.south_pole]))
        return [Vector([v[d] / (1.0001 - v[self.south_pole] / self.radius) for d in reduced_dims]) for v in
                src]

    def _stereo_projection_cell(self, indices=[], scale=1):
        stereo_vs = self._stereo_projection([self.vs[i] for i in indices])
        center = sum(stereo_vs, Vector()) * (1 / len(stereo_vs))
        if scale != 1:
            stereo_vs = [center + (v - center) * scale for v in stereo_vs]
        return stereo_vs, center

    def _create_map(self, source, target):
        src2target_map = {}
        for src_index, src in enumerate(source):
            min_dist = np.inf
            for target_index, tgt in enumerate(target):
                dist = (src - tgt).length
                if dist < min_dist:
                    min_dist = dist
                    src2target_map[src_index] = target_index
        # debugging
        print(src2target_map)
        inverse = DefaultDict(list)
        for src, target in src2target_map.items():
            inverse[target].append(src)

        orders = set()
        for pre_image in inverse.values():
            orders.add(len(pre_image))
        print(inverse)
        print(orders)

        return src2target_map


class Unfolder4D2:
    """
    Unfolder of geometrical shapes in 4D
    The faces are arranged in a tree structure
    """

    def __init__(self, group=CoxA4, coxeter_dynkin_label="x3x3x3x", root_index=-1, mode="LARGEST_CELL", skip=0,
                 radius=5, no_tree=False, **kwargs):

        start = time()
        external_rotation = get_from_kwargs(kwargs, "external_rotation", None)
        rotate_to_south_pole = get_from_kwargs(kwargs, "rotate_to_south_pole", True)

        self.group = group(path="../mathematics/geometry/data")
        self.cd_label = coxeter_dynkin_label
        self.group_size = self.group.size
        signature = eval(self.group.name.upper() + "_SIGNATURES")[coxeter_dynkin_label]
        self.point_cloud = self.group.point_cloud(signature)
        self.group_name = self.group.name
        self.skip = skip
        self.vs = [to_vector(v.real()) for v in self.point_cloud]
        self.radius = (self.vs[0].dot(self.vs[0])) ** 0.5
        scale = radius / self.radius
        self.radius = radius
        self.vs = [scale * v for v in self.vs]
        self.stereo = []  # computed in stereo_projection

        self.cells = self.group.get_cells(signature)
        # make normals real
        for key, val in self.cells.items():
            self.cells[key] = val.real()

        self.faces = self.group.get_faces(signature)
        self.edges = self.group.get_edges(signature)
        self.centers = [self.center(cell, self.vs) for cell in self.cells]

        self.crystal = get_from_kwargs(kwargs, "crystal", False)
        self.south_pole = get_from_kwargs(kwargs, "south_pole", 1)

        if self.south_pole == 1:
            south_pole = Vector([0, -1, 0, 0])
        if self.south_pole == 2:
            south_pole = Vector([0, 0, -1, 0])

        if external_rotation is None:
            # create an external rotation depending on the mode provided by the user
            if mode == "LARGEST_CELL":
                # find largest Cell closest to the pole and
                # find rotation that rotates the center to the pole
                smallest_cell = 0
                for cell in self.cells:
                    if len(cell) > smallest_cell:
                        smallest_cell = len(cell)
                min_dist = np.inf
                min_center = None
                min_index = -1
                for i, (cell, center) in enumerate(zip(self.cells, self.centers)):
                    if len(cell) == smallest_cell:
                        dist = (center - south_pole).length
                        if dist < min_dist:
                            min_dist = dist
                            min_center = center
                            min_index = i
                self.external_rotation = align_vectors(min_center, south_pole)
            elif mode == "SMALLEST_CELL":
                # find largest Cell closest to the pole and
                # find rotation that rotates the center to the pole
                smallest_cell = np.inf
                for cell in self.cells:
                    if len(cell) < smallest_cell:
                        smallest_cell = len(cell)
                min_dist = np.inf
                min_center = None
                min_index = -1
                for i, (cell, center) in enumerate(zip(self.cells, self.centers)):
                    if len(cell) == smallest_cell:
                        dist = (center - south_pole).length
                        if dist < min_dist:
                            min_dist = dist
                            min_center = center
                            min_index = i
                self.external_rotation = align_vectors(min_center, south_pole)
        else:
            self.external_rotation = external_rotation

        # transform vertices and normals according to a given rotation
        self.vs = [self.external_rotation @ v for v in self.vs]
        for key, val in self.cells.items():
            self.cells[key] = self.external_rotation @ val
        print("external rotation", self.external_rotation, "performed on all vertices", str(time() - start) + " ms")

        self.stereo = self.stereo_projection(self.vs, south_pole=self.south_pole)
        print("Stereo projection done", str(time() - start) + " ms")

        self.vertex2unfolded_map = None
        self.unfolded2vertex_map = None
        self.unfolded_vertices = None
        self.child_info_map = None
        self.bob = None
        self.conjugacy_classes = self.group.get_cells_in_conjugacy_classes(signature=signature)
        print("Conjugacy classes computed", str(time() - start) + " ms")
        self.material_names, self.materials, self.structures = create_materials(coxeter_dynkin_label,
                                                                                self.conjugacy_classes, self.group_size,
                                                                                self.crystal, **kwargs)
        for i, material_name in enumerate(self.material_names):
            print(i, material_name)
        print("Materials created ", str(time() - start) + " ms")
        # self.material_names = flatten(self.material_names)
        # self.structures = flatten(self.structures)

        self.root_index = root_index
        self.mode = mode

        if root_index == -1:
            root_index = self._find_root_index(mode, rotate_to_south_pole=rotate_to_south_pole)
        if not no_tree:
            self.root = self._create_full_cell_tree(root_index)
            self.order = self._update_order()

        self.kwargs = kwargs

    def center(self, cell, point_cloud):
        cntr = sum([point_cloud[i] for i in cell], Vector([0, 0, 0, 0])) / len(cell)
        return cntr

    def create_map(self, other):
        """
        create a map that relates the four-dimensional vertices.
        The map identifies the vertices that are close in four-dimensions
        """

        self_points = [v.real() for v in self.point_cloud]
        radius = self_points[0].length
        other_points = [v.real() for v in other.point_cloud]
        other_radius = other_points[0].length
        other_points = [radius / other_radius * v for v in other_points]

        # map from self to other
        vertex_map = DefaultDict(list)
        for idx, src in enumerate(self_points):
            min_dist = np.inf
            for idx2, target in enumerate(other_points):
                dist = src - target
                dist = dist.dot(dist)
                if np.abs(dist - min_dist) < 0.1:  # similar point
                    vertex_map[idx].append(idx2)
                elif dist < min_dist:  # closer point
                    min_dist = dist
                    vertex_map[idx] = [idx2]

        # try to make it unique, if there are single element list, they are fixed and the targets are removed from
        # all other lists
        final_map = {}

        for src, targets in vertex_map.items():
            if len(targets) == 1:
                final_map[src] = targets[0]
            else:
                raise "mapping is not unique"

        # check the inverse map
        inverse = DefaultDict(list)
        for src, tar in vertex_map.items():
            for t in tar:
                inverse[t].append(src)

        pre_image_sizes = set([len(src) for src in inverse.values()])
        print(vertex_map)
        print(inverse)
        if len(pre_image_sizes) != 1:
            raise "The  map is not homogenous"

        return final_map

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, sequentially=True):
        # cell2child_map
        if sequentially and self.bob:
            self.bob.appear(begin_time=begin_time, transition_time=0, children=False)
            cell2child_map = {}
            for child_index, child in enumerate(self.bob.b_children):
                child_info_map = self.child_info_map[child_index]
                (cell_vertex_id2src_map, cell_faces, cell_color, cell_modifier, cell_index,
                 conjugacy_class_index, cell_vertex_id2unfolded_map) = child_info_map
                cell2child_map[cell_index] = child_index

            count = 0
            dt = transition_time / len(self.bob.b_children)
            nodes = self.get_nodes_sorted_by_level(self.root)
            for node in nodes:
                cell_index = node.name[0]
                self.bob.b_children[cell2child_map[cell_index]].appear(begin_time=begin_time + count * dt,
                                                                       transition_time=dt)
                count += 1
        else:
            self.bob.appear(begin_time=begin_time, transition_time=transition_time, children=True)

        return begin_time + transition_time

    def stereo_projection(self, src, south_pole=3):
        dims = set(range(4))
        reduced_dims = dims.difference(set([south_pole]))
        return [Vector([v[d] / (1.0001 - v[south_pole] / self.radius) for d in reduced_dims]) for v in
                src]

    def create_stereo_bob(self, cell_removals=[], **kwargs):
        half_way = get_from_kwargs(kwargs, "half_way", False)
        limit = get_from_kwargs(kwargs, "limit", 0)
        limit_direction = get_from_kwargs(kwargs, "limit_direction", Vector([0, 0, 1]))
        add_cells = get_from_kwargs(kwargs, "add_cells", [])

        children = []
        child_index = 0
        self.child_info_map = {}

        for cell_index, cell in enumerate(self.cells):
            verts = [self.stereo[i] for i in cell]
            center = sum(verts, Vector()) / len(verts)

            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in self.faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            # conjugacy class
            conjugacy_class_index = -1
            sorted_indices = tuple(sorted(cell))
            for class_index, (rep, members) in enumerate(self.conjugacy_classes.items()):
                if sorted_indices in members:
                    conjugacy_class_index = class_index
                    break

            distance = center.dot(limit_direction)
            if not half_way or distance <= limit or cell_index in add_cells:
                # only show the first layer in the negative direction
                # if compute_cell_size(verts) != np.abs(distance):

                face_types = set()
                for face in selected_faces:
                    face_types.add(len(face))

                scale = 1
                color = self.material_names[conjugacy_class_index][1]
                cell_obj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces, orient_faces=True),
                                   name="Cell" + str(cell_index), color=color,
                                   scale=scale)

                if self.crystal:
                    crystal = CrystalModifier(material=get_texture(color), **self.kwargs)
                    if cell_index not in add_cells:
                        cell_obj.add_mesh_modifier(type="NODES", node_modifier=crystal)
                        crystal.transfer_material_from(cell_obj)
                        modifier = crystal
                    else:
                        modifier = None
                else:
                    modifier = None

                self.child_info_map[child_index] = ({j: i for j, i in enumerate(cell)}, selected_faces,
                                                    color, modifier, conjugacy_class_index)
                print("\r added cell ", str(cell_index) + "/" + str(len(self.cells)), end="")
                child_index += 1
                children.append(cell_obj)

        print(" done!")
        children = [child for child in children if int(child.name[4:]) not in cell_removals]
        return BObject(children=children, **kwargs)

    def create_net(self, **kwargs):
        scale_elements = get_from_kwargs(kwargs, "scale_elements", 1)
        pre_color = get_from_kwargs(kwargs, "color", None)
        cells_sorted = get_from_kwargs(kwargs, "cells_sorted", False)

        if self.unfolded_vertices is None:
            self.unfolded2vertex_map, self.vertex2unfolded_map, self.unfolded_vertices = self._unfold()

        children = []
        child_index = 0
        self.child_info_map = {}

        if cells_sorted:
            self.order.sort()
        for idx in self.order:
            cell = list(self.cells.keys())[idx]
            v2u_local = self.vertex2unfolded_map[idx]

            cell_vertices = [self.unfolded_vertices[v2u_local[i]] for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in self.faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            # find conjugacy class
            conjugacy_class_index = -1
            sorted_indices = tuple(sorted(cell))
            for class_index, (rep, members) in enumerate(self.conjugacy_classes.items()):
                if sorted_indices in members:
                    conjugacy_class_index = class_index
                    break

            # resize cell

            center = sum(cell_vertices, Vector()) / len(cell_vertices)
            if scale_elements != 1:
                cell_vertices = [center + (v - center) * scale_elements for v in cell_vertices]

            if pre_color is None:
                color = self.material_names[conjugacy_class_index][1]  #skip the transparent for single usage
                # print(color,"assigned to cell",child_index)
            else:
                color = pre_color
            cell_obj = BObject(mesh=ibpy.create_mesh(vertices=cell_vertices, faces=selected_faces, orient_faces=True),
                               name="Cell" + str(idx),
                               color=color)

            if self.crystal:
                crystal = CrystalModifier()
                cell_obj.add_mesh_modifier(type="NODES", node_modifier=crystal)
                crystal.transfer_material_from(cell_obj)
                modifier = crystal
            else:
                modifier = None

            self.child_info_map[child_index] = ({j: i for j, i in enumerate(cell)}, selected_faces, color, modifier,
                                                idx, conjugacy_class_index,
                                                {j: v2u_local[i] for j, i in enumerate(cell)})
            child_index += 1
            children.append(cell_obj)

        self.bob = BObject(children=children, **kwargs)
        return self.bob

    def _get_quadrupole(self):
        """
        compute the quadrupole moment of the mass distribution
        """
        vertex_list = self.unfolded_vertices
        for v in vertex_list:
            v[1] = 0
        start = np.array([0] * 9)
        start = start.reshape(3, 3)
        identity = np.identity(3, float)
        quadrupole = sum([3 * np.tensordot(v, v, axes=0) - v.dot(v) * identity for v in vertex_list], start)
        return quadrupole

    def get_main_axis(self, choice="MAXIMUM"):
        quadrupole = self._get_quadrupole()
        ev, vectors = np.linalg.eig(quadrupole)
        abs_evs = np.abs(ev)
        if choice == "MAXIMUM":
            ev0 = max(abs_evs)
        if choice == "MINIMUM":
            ev0 = min(abs_evs)
        idx0 = list(abs_evs).index(ev0)
        return vectors.transpose()[idx0]

    def get_main_axes_and_eigenvalues(self):
        quadrupole = self._get_quadrupole()
        ev, vectors = np.linalg.eig(quadrupole)
        return vectors.transpose(), ev

    def show_node_tree(self):
        for pre, _, node in RenderTree(self.root):
            if node.parent is not None:
                parent_idx = set(node.parent.name[2])
                child_idx = set(node.name[2])
                intersection_idx = parent_idx.intersection(child_idx)
                remaining_idx = child_idx - intersection_idx
                print("%s%s%s%s" % (pre, node.name[0], intersection_idx, remaining_idx))
            else:
                print("%s%s%s" % (pre, node.name[0], node.name[2]))

    def show_index_structure(self, vertex_map=None):
        print(
            "\033[1mCell tree for the polytope\033[0m\nblack: original vertex indices\nred: unfolded vertex "
            "indices\n\033[1mbold\033[0m: common indices")
        if self.unfolded2vertex_map is None:
            self.unfolded2vertex_map, self.vertex2unfolded_map, self.unfolde_vertices = self._unfold()

        for pre, _, node in RenderTree(self.root):
            print("%s%s" % (pre, self._index_structure(node, vertex_map=vertex_map)))

    def pullback_tree(self, src_unfolder):
        # relate physical vertices of both polyhedra through their closeness
        target2src_map = self.create_map(src_unfolder)

        # find matching root face
        src_unfolder.root_cell_index = src_unfolder.find_root(src_unfolder.root, target2src_map)
        # pullback existing faces
        tmp_root, indices, src2target_cell_map = self._pullback(src_unfolder.root, target2src_map, [], {})
        # add buffer faces to fill gaps
        old_number_of_cells = 0
        new_number_of_cells = len(indices)
        round = 1
        while new_number_of_cells > old_number_of_cells:
            old_number_of_cells = new_number_of_cells
            print("add buffer cells in round", round)
            tmp_root, indices, src2target_cell_map = self._fill_gaps(tmp_root, indices, src2target_cell_map)
            new_number_of_cells = len(indices)
            round += 1

        # add left over faces as leaves
        tmp_root, indices, src2target_cell_map = self._extend_missing_cells(tmp_root, indices, src2target_cell_map)

        return tmp_root, target2src_map, src2target_cell_map

    def find_node(self, node, vertex_index):
        """
        find the cell that hosts the unfolded index
        """
        unfolded_indices = [self.vertex2unfolded_map[node.name[0]][i] for i in node.name[2]]
        if vertex_index in unfolded_indices:
            return node
        else:
            for child in node.children:
                result = self.find_node(child, vertex_index)
                if result is not None:
                    return result
        return None

    def find_root(self, other_root, src2target_vertex_map) -> int:
        """
        find the face that turns into the root of the other net under the src2target_vertex_map
        """
        other_root_set = set(other_root.name[2])
        for idx, cell in enumerate(self.cells):
            mapped_cell = set([src2target_vertex_map[i] for i in cell])
            if mapped_cell == other_root_set:
                return idx
        return None

    def get_cell(self, index=0):
        return self.bob.b_children[index]

    def _find_root_index(self, mode, rotate_to_south_pole=True):
        """
        find root index, depending on the mode
        LARGEST_CELL: choose the cell with the largest number of vertices as root cell and take the one in position skip
        SMALLEST_CELL: choose the cell with the smallest number of vertices as root cell and take the one in position skip
        CLOSEST_TO_SOUTH_POLE: choose the cell with center closest to (0,0,0,-1)
        """
        skip = self.skip
        if mode == "LARGEST_CELL":
            max_cell_size = 0
            for cell in self.cells.keys():
                if len(cell) > max_cell_size:
                    max_cell_size = len(cell)

            for i, cell in enumerate(self.cells.keys()):
                if len(cell) == max_cell_size:
                    if skip == 0:
                        root_cell = cell
                        root_index = i
                        break
                    else:
                        skip -= 1

            # rotate vertices to make the center of the root_cell sit on the south pole
            # compare Household.nb
            center = sum([self.vs[i] for i in root_cell], Vector([0, 0, 0, 0]))
            p = center / center.length

            if rotate_to_south_pole:
                rot = align_vectors(p, Vector([0, 0, 0, -1]))
                south_pole = rot @ center
                assert (Vector(south_pole[0:3]).length < 1e-4)
                # rotate vertices
                self.vs = [rot @ v for v in self.vs]

                # update normals
                for cell, normal in self.cells.items():
                    old_normal = self.cells[cell]
                    smart_normal = Vector(rot @ old_normal)
                    smart_normal = smart_normal / smart_normal.length
                    self.cells[cell] = smart_normal
        elif mode == "CLOSEST_TO_SOUTH_POLE":
            shortest_dist = np.inf
            closest_cell_index = -1

            for cell_index, cell in enumerate(self.cells.keys()):
                center = sum([self.vs[i] for i in cell], Vector([0, 0, 0, 0]))
                center = center / center.length

                dist = to_vector((Vector([0, 0, 0, -1]) - center)).length
                if dist < shortest_dist:
                    shortest_dist = dist
                    closest_cell_index = cell_index

                # adjust normals
                self.cells[cell] = Vector(self.cells[cell].real())

            root_index = closest_cell_index
        elif mode == "SMALLEST_CELL":
            min_cell_size = np.inf
            for cell in self.cells.keys():
                if len(cell) < min_cell_size:
                    min_cell_size = len(cell)

            for i, cell in enumerate(self.cells.keys()):
                if len(cell) == min_cell_size:
                    if skip == 0:
                        root_cell = cell
                        root_index = i
                        break
                    else:
                        skip -= 1

            # rotate vertices to make the center of the root_cell sit on the south pole
            # compare Household.nb
            center = sum([self.vs[i] for i in root_cell], Vector([0, 0, 0, 0]))
            p = center / center.length

            if rotate_to_south_pole:
                rot = align_vectors(p, Vector([0, 0, 0, -1]))
                south_pole = rot @ center
                assert (Vector(south_pole[0:3]).length < 1e-4)

                # rotate vertices
                self.vs = [rot @ v for v in self.vs]

                # update normals
                for cell, normal in self.cells.items():
                    old_normal = self.cells[cell]
                    smart_normal = Vector(rot @ old_normal)
                    smart_normal = smart_normal / smart_normal.length
                    self.cells[cell] = smart_normal

        return root_index

    def _update_order(self):
        # create order
        self.order = [self.root.name[0]]
        next_level = self.root.children
        while len(next_level) > 0:
            next_level = sorted(next_level, key=lambda x: len(x.name[2]))
            new_level = []
            for child in next_level:
                self.order.append(child.name[0])  # append children from small to large

            # next_level = sorted(next_level,key=lambda x:-len(x.name[2]))
            for child in next_level:
                # add first children of large children
                new_level.extend(child.children)
            next_level = new_level
        return self.order

    def _index_structure(self, node, vertex_map=None):
        """
        highlight common indices in boldface
        """
        out = ""
        cell_indices = node.name[2]
        cell_index = node.name[0]
        parent = node.parent
        common_indices = []
        if parent is not None:
            common_indices = list(set(parent.name[2]) & set(node.name[2]))

        if vertex_map is None:
            for src_vertex_idx in cell_indices:
                local_vertex2unfolded_map = self.vertex2unfolded_map[cell_index]
                if src_vertex_idx in common_indices:
                    out += f"\033[1m{src_vertex_idx}\033[0m(\033[91m{local_vertex2unfolded_map[src_vertex_idx]}\033[0m) "
                else:
                    out += f"{src_vertex_idx}(\033[91m{local_vertex2unfolded_map[src_vertex_idx]}\033[0m) "
        else:
            for src_vertex_idx in cell_indices:
                if src_vertex_idx in common_indices:
                    out += f"\033[91m{vertex_map[src_vertex_idx]}\033[0m "
                else:
                    out += f"\033[91m{vertex_map[src_vertex_idx]}\033[0m "
        return out

    def _create_full_cell_tree(self, root_index=-1, root_size=-1):

        cells = self.cells

        index_to_cell_map = {i: cell for i, cell in enumerate(cells.keys())}
        index_to_normal_map = {i: cell for i, cell in enumerate(cells.values())}

        if root_index == -1:
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

        root = Node((root_cell_index, len(root_cell), set(root_cell), root_normal))

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
                        node = Node((index, len(cell_set), cell_set, index_to_normal_map[index]), parent=parent)
                        new_nodes.append(node)
                        next_level.append(node)

                # pop children
                for child in new_nodes:
                    index_to_cell_map.pop(child.name[0])

            parents = next_level
            print(str(len(parents)) + "new children added ")

        return root

    def _unfold(self):
        """
        each vertex can have various unfolded images, depending on the cell that is under consideration,
        the map is bijective when restricted to a single cell; therefore, we store the corresponding map for each cell
        the inverse map is injective, each unfolded vertex index belongs has a unique vertex index
        """

        unfolded2vertex_map, vertex2unfolded_map, unfolded_vertices = self._indexing_unfolded_vertices()
        self._recalculate_vertices_and_normals(self.root,
                                               unfolded_vertices,
                                               unfolded2vertex_map,
                                               vertex2unfolded_map,
                                               [])
        self._rotate_into_w_zero_space(unfolded_vertices, vertex2unfolded_map)
        return unfolded2vertex_map, vertex2unfolded_map, [to_vector(v[0:3]) for v in unfolded_vertices]

    def _indexing_unfolded_vertices(self):
        """
        The process of unfolding produces additional geometry
        These additional vertices have to be indexed and stored for later use
        Also their relation to the original vertices has to be preserved
        The map is only bijective for a single cell, therefore, an individual map is stored for each cell
        """

        root = self.root
        unfolded2vertex_map = {}
        vertex2unfolded_map = {}
        unfolded_vertices = []  # right now, it is just a copy of the original vertices
        new_unfolded_indices = []
        for i in root.name[2]:
            j = len(unfolded2vertex_map)
            unfolded2vertex_map[j] = i
            new_unfolded_indices.append(j)
            unfolded_vertices.append(self.vs[i])

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
                        unfolded_vertices.append(self.vs[i])

                    local_vertex2unfolded_map = {}
                    for j, i in zip(new_unfolded_indices, child.name[2]):
                        local_vertex2unfolded_map[i] = j
                    vertex2unfolded_map[child.name[0]] = local_vertex2unfolded_map
            parents = level_children
            print("reindex", len(parents), "children")
        return unfolded2vertex_map, vertex2unfolded_map, unfolded_vertices

    def _recalculate_vertices_and_normals(self, cell, unfolded_vertices, unfolded2vertex_map, vertex2unfolded_map,
                                          rotations):
        """
         deal with root
         each rotation is defined by a rotation matrix and a plane of rotation
         (similarly as in three dimensions a rotation is defined by a rotation matrix and an axis of rotation)
         (similarly as in two dimensions a rotation is defined by a rotation matrix and a point of rotation)
         we store the rotation data as a dictionary, where the key is the cell id, the value is a tuple that contains
         (three points of the plane and the rotation matrix)
        """
        local_rotations = [rotation for rotation in rotations]  # create local copy

        if cell.parent is not None:
            # deal with non-root cells (generic case)
            parent = cell.parent
            common_indices = cell.name[2] & parent.name[2]
            v2u_local = vertex2unfolded_map[cell.name[0]]

            # transform all vertices with previous rotations (all vertices have to be transformed, since the common
            # indices are stored independently of their parent cell)
            for i in cell.name[2]:
                for rotation in local_rotations:
                    j = v2u_local[i]
                    unfolded_vertices[j] = self._apply_rotation(rotation, unfolded_vertices[j])

            if cell.name[0] == 7:
                start = unfolded_vertices[v2u_local[list(cell.name[2])[0]]]
                vstart = self.vs[list(cell.name[2])[0]]
                print([cell.name[3].dot(Vector(unfolded_vertices[v2u_local[idx]] - start)) for idx in cell.name[2]])
                print([cell.name[3].dot(Vector(self.vs[idx] - vstart)) for idx in cell.name[2]])
                pass

            # transform normal of a child cell with previous rotation
            for rotation in local_rotations:
                cell.name = (cell.name[0], cell.name[1], cell.name[2], rotation[3] @ cell.name[3])

            # align child and parent normals
            child_normal = cell.name[3]
            parent_normal = parent.name[3]

            common_vertices = [unfolded_vertices[v2u_local[i]] for i in common_indices]

            new_rotation = self._get_rotation(common_vertices, child_normal, parent_normal)

            # transform vertices with the last rotation
            for i in cell.name[2] - common_indices:
                j = v2u_local[i]
                unfolded_vertices[j] = self._apply_rotation(new_rotation, unfolded_vertices[j])
            # transform normal with the last rotation
            cell.name = (cell.name[0], cell.name[1], cell.name[2], new_rotation[3] @ cell.name[3])
            local_rotations.append(new_rotation)
        else:
            # no rotation is needed, since the root cell need not be unfolded.
            local_rotations = []

        # recursively proceed to the children
        for child in cell.children:
            self._recalculate_vertices_and_normals(child, unfolded_vertices, unfolded2vertex_map,
                                                   vertex2unfolded_map, local_rotations)

    def _get_rotation(self, common_vertices, a, b):
        """
         Find rotation that rotates a to b, while leaving e1 and e2 invariant.
        """

        # create orthonormal base from e1 and e2
        if len(common_vertices) < 3:
            raise "Something's wrong with the common face"
        p1 = to_vector(common_vertices[0])
        p2 = to_vector(common_vertices[1])
        p3 = to_vector(common_vertices[2])
        e1 = p2 - p1
        e2 = p3 - p1

        u = e1 / np.sqrt(e1.dot(e1))
        v = (e2 - u.dot(e2) * u)
        v = v / np.sqrt(v.dot(v))
        a = a / np.sqrt(a.dot(a))
        b = b / np.sqrt(b.dot(b))

        # create common face normals for each cell
        na = self._normal_to(u, v, a)
        na = na / np.sqrt(na.dot(na))
        nb = self._normal_to(u, v, b)
        nb = nb / np.sqrt(nb.dot(nb))

        # the rotation matrix is quite easy to understand
        # u->u
        # v->v
        # na->nb
        # a->b

        rot = (np.tensordot(u, u, axes=0) +
               np.tensordot(v, v, axes=0) +
               np.tensordot(nb, na, axes=0) +
               np.tensordot(b, a, axes=0))

        return p1, p2, p3, Matrix(rot)  # return matrix that transforms the vertices

    def _normal_to(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        abc = np.tensordot(np.tensordot(a, b, axes=0), c, axes=0)
        n = np.tensordot(epsilon(4), abc, axes=[[1, 2, 3], [0, 1, 2]])
        return Vector(n)

    def _apply_rotation(self, rotation, vertex):
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
        p = to_vector(p)
        projection = Matrix(np.tensordot(u, u, axes=0) + np.tensordot(v, v, axes=0))
        vertex = to_vector(vertex)
        delta = vertex - p
        center = p + projection @ delta

        return rotation[3] @ (vertex - center) + center

    def _rotate_into_w_zero_space(self, unfolded_vertices, vertex2unfolded_map):
        """
        all unfolded vertices have to be rotated by a global rotation
        """
        normal = self.root.name[3]
        rotation = align_vectors(normal, Vector([0, 0, 0, 1]))

        # rotate vertices around the center of the root cell
        root_indices = self.root.name[2]
        v2u_local = vertex2unfolded_map[self.root.name[0]]
        root_vertices = [to_vector(unfolded_vertices[v2u_local[i]]) for i in root_indices]
        center_of_root = sum(root_vertices, Vector([0, 0, 0, 0])) / len(root_vertices)

        for i, v in enumerate(unfolded_vertices):
            unfolded_vertices[i] = to_vector(rotation @ (to_vector(v) - center_of_root) + center_of_root)

        # translate center of mass to origin
        center = sum(unfolded_vertices, Vector([0, 0, 0, 0])) / len(unfolded_vertices)
        for i, v in enumerate(unfolded_vertices):
            unfolded_vertices[i] = v - center

    def _pullback(self, morphed, vertex_map, indices=[], src2target_cell_map={}):
        """
        Find the original child that turned into the morphed child
        The pullback of the indices should be contained in the original indices
        """
        match = self._find_recursively(self.root, morphed, vertex_map)
        if match is not None:
            indices.append(match.name[0])
            print(morphed.name[0], "->", match.name[0])
            src2target_cell_map[match.name[0]] = [morphed.name[0]]
            local_root = Node((match.name[0], match.name[1], match.name[2], match.name[3]))
            for child in morphed.children:
                pulled_child, indices, src2target_cell_map = self._pullback(child, vertex_map, indices,
                                                                            src2target_cell_map)
                if pulled_child:
                    pulled_child.parent = local_root
                else:
                    print("no source found for ", child.name[0], child.name[2])
            return local_root, indices, src2target_cell_map
        return None

    def _find_recursively(self, original, morphed, vertex_map):
        mapped_original = set([vertex_map[o] for o in original.name[2]])
        morphed_indices = set(morphed.name[2])
        if mapped_original == morphed_indices:
            return original
        else:
            for child in original.children:
                source = self._find_recursively(child, morphed, vertex_map)
                if source:
                    return source
        return None

    def _extend_missing_cells(self, tmp_root, indices, src2target_cell_map):
        """
        try to append missing cells to the tree:
        add them as leaves
        """

        all_indices = set(range(len(self.cells)))
        if len(indices) == all_indices:
            return tmp_root, indices, src2target_cell_map

        # first try to add them as leaves
        round = 1
        old_number_of_indices = 0
        new_number_of_indices = len(indices)
        while new_number_of_indices > old_number_of_indices:
            old_number_of_indices = new_number_of_indices
            for cell_index, (cell_indices, cell_normal) in enumerate(self.cells.items()):
                if cell_index not in indices:
                    success, src2target_cell_map = self._add_as_leaf(tmp_root, cell_index, cell_indices, cell_normal,
                                                                     src2target_cell_map)
                    if success:
                        indices.append(cell_index)
                        print("added cell as leaf", cell_index, "in round", round)
            round += 1
            tmp_root, indices, src2target_cell_map = self._fill_gaps(tmp_root, indices, src2target_cell_map)
            new_number_of_indices = len(indices)

        for cell_index, (cell_indices, cell_normal) in enumerate(self.cells.items()):
            if cell_index not in indices:
                print("create new buffer cells in round", round)
                tmp_root, indices, src2target_cell_map = self._fill_gaps(tmp_root, indices, src2target_cell_map)
                print("add more leaves in round", round)
                tmp_root, indices, src2target_cell_map = self._extend_missing_cells(tmp_root, cell_index, cell_indices,
                                                                                    cell_normal, )

        return tmp_root, indices, src2target_cell_map

    def _add_as_leaf(self, node, cell_index, cell_indices, cell_normal, src2target_cell_map):
        """
        try to add a cell as a leaf
        """

        leaves = self.get_nodes_sorted_by_level(node)
        for leaf in leaves:
            if len(set(cell_indices).intersection(set(leaf.name[2]))) > 2:
                new_node = Node((cell_index, len(cell_indices), set(cell_indices), cell_normal), parent=leaf)
                src2target_cell_map[cell_index] = [src2target_cell_map[leaf.name[0]][0]]
                new_node.parent = leaf
                return True, src2target_cell_map

        return False, src2target_cell_map

        # if len(node.children) == 0:
        #     if len(set(cell_indices).intersection(set(node.name[2]))) > 2:
        #         new_node = Node((cell_index, len(cell_indices), set(cell_indices), cell_normal), parent=node)
        #         src2target_cell_map[cell_index] = [src2target_cell_map[node.name[0]][0]]
        #         new_node.parent = node
        #         return True, src2target_cell_map
        # for child in node.children:
        #     success, src2target_cell_map = self._add_as_leaf(child, cell_index, cell_indices, cell_normal,
        #                                                      src2target_cell_map)
        #     if success:
        #         return True, src2target_cell_map
        # return False, src2target_cell_map

    def _fill_gaps(self, node, indices, src2target_cell_map):
        """
        try to add a cell as a buffer between child and parent
        """
        removables = []
        new_children = []
        for child in node.children:
            parent_idx = set(node.name[2])
            child_idx = set(child.name[2])
            intersection_idx = parent_idx.intersection(child_idx)
            if len(intersection_idx) < 3:
                for idx, (cell_indices, cell_normal) in enumerate(self.cells.items()):
                    if idx not in indices:
                        buffer_idx = set(cell_indices)
                        if len(buffer_idx.intersection(parent_idx)) > 2 and len(buffer_idx.intersection(child_idx)) > 2:
                            print("added cell as buffer", idx)
                            buffer_node = Node((idx, len(cell_indices), set(cell_indices), cell_normal))
                            src2target_cell_map[idx] = [src2target_cell_map[node.name[0]][0],
                                                        src2target_cell_map[child.name[0]][0]]
                            indices.append(idx)
                            removables.append(child)
                            new_children.append(buffer_node)
                            child.parent = buffer_node
        node.children = list(set(node.children).union(set(new_children)) - set(removables))

        for child in node.children:
            self._fill_gaps(child, indices, src2target_cell_map)

        return node, indices, src2target_cell_map

    def get_nodes_sorted_by_level(self, node):
        node_list = []
        new_roots = [node]
        while len(new_roots) > 0:
            tmp_roots = []
            for root in new_roots:
                node_list.append(root)
                for child in root.children:
                    tmp_roots.append(child)
            new_roots = tmp_roots
        return node_list


class Unfolder:
    """
    Unfolder of geometrical shapes in 3D
    The faces are arranged in a tree structure

    """

    def __init__(self, group=CoxA3, signature=[1, 1, -1],
                 root_index=-1, mode="LARGEST_FACE", skip=0, radius=5, **kwargs):
        if callable(group):
            self.group = group(path="../mathematics/geometry/data")
        else:
            self.group = group
        self.signature = signature
        vertices = self.group.point_cloud(signature)
        self.group_name = self.group.name
        self.vs = [to_vector(v.real()) for v in vertices]
        self.radius = self.vs[0].length
        scale = radius / self.radius
        self.radius = radius
        self.vs = [scale * v for v in self.vs]
        self.stereo = []
        self.faces = self.group.get_faces(signature)
        self.face_classes = self.group.get_faces_in_conjugacy_classes(self.signature)
        self.edges = self.group.get_edges(signature)

        if -1 < root_index < len(self.faces):
            self.root_face_index = root_index
        else:
            self.root_face_index = self._find_root_index(mode, skip)
        self.normals = self._compute_normals()

        self.root = self.create_tree(self.root_face_index)
        self.kwargs = kwargs
        self.color = get_from_kwargs(kwargs, "color", "drawing")

        # data structures that capture the unfolding the location of the unfolded vertices is stored in the list:
        # unfolded_vertices
        # there are more unfolded vertices than there are vertices on the initial polyhedron
        # similarly, there are more edges. Only the number of faces is the same.
        #
        # for each face, we need a separate map that stores the connection between original and unfolded vertices
        #
        # unfolded2vertex_map: a single map from unfolded vertices to the original vertices
        # vertex2unfolded_map: a dictionary of maps for each face individually
        # unfolded_vertices: a list of all unfolded vertices

        self.unfolded2vertex_map = {}
        self.vertex2unfolded_map = {}
        self.unfolded_vertices = None
        self.bob = None

    def get_face_classes(self):
        if self.face_classes is None:
            self.face_classes = self.group.get_faces_in_conjugacy_classes(self.signature)
        return self.face_classes

    def create_bob(self, **kwargs):
        verts = [v for v in self.vs]
        faces = [face for face in self.faces]

        color = get_from_kwargs(kwargs, "color", color_dict)
        if color == "color_dict":
            colors = []
            for face_repr in self.face_classes.keys():
                colors.append(color_dict[len(face_repr)])
        else:
            colors = [color] * len(self.get_face_classes())

        def find_face_class(face_indices):
            face = MeshFace(list(face_indices))
            for class_index, face_class in enumerate(self.face_classes.values()):
                if face in face_class:
                    return class_index
            return 0  # default color

        bob = BObject(mesh=ibpy.create_mesh(verts, faces=faces), color=self.color, **(self.kwargs | kwargs))
        modifier = PolyhedronViewModifier(edge_color="example", vertex_color="red")
        bob.add_mesh_modifier(type="NODES", node_modifier=modifier)

        for i, col in enumerate(colors):
            ibpy.set_material(bob, get_texture(col, **kwargs), slot=i)
        ibpy.set_color_to_faces(bob, lambda x: find_face_class(x))

        self.bob = bob
        return bob

    def compute_net(self):
        # reset normals to the original unfolded normals, this is necessary, when the face-tree has changed
        # to accommodate the morphing between to nets
        self._recompute_normals(self.root)
        self.unfolded2vertex_map, self.vertex2unfolded_map, self.unfolded_vertices = self._indexing_unfolded_vertices()
        self._recalculate_vertices_and_normals(self.root)
        self.face_classes = self.group.get_faces_in_conjugacy_classes(self.signature)

    def display_net_data(self):
        print("unfolded_vertices:")
        print([rounded_vector(v, 1) for v in self.unfolded_vertices])
        print("root: ")
        for pre, _, node in RenderTree(self.root, style=AsciiStyle()):
            print("%s %s %s %s %s" % (pre, node.name[0], node.name[1], node.name[2], rounded_vector(node.name[3], 1)))
        print("face_classes: ")
        print(self.face_classes)

    def create_net(self, recompute=True, **kwargs) -> BObject:
        if recompute:
            self.compute_net()

        color = get_from_kwargs(kwargs, "color", "color_dict")

        if color == "color_dict":
            colors = []
            for face_repr in self.face_classes.keys():
                colors.append(color_dict[len(face_repr)])
        else:
            colors = [color] * len(self.face_classes)

        colors = get_from_kwargs(kwargs, "colors", colors)

        # create net as a single mesh instance
        unfolded_faces = []
        for idx, face in enumerate(self.faces):
            v2u_local = self.vertex2unfolded_map[idx]
            unfolded_faces.append([v2u_local[i] for i in face])

        bob = BObject(mesh=ibpy.create_mesh(vertices=self.unfolded_vertices, faces=unfolded_faces),
                      **kwargs)
        for i, col in enumerate(colors):
            ibpy.set_material(bob, get_texture(col, **kwargs), slot=i)

        def find_face_class(face_indices):
            src_face_indices = [self.unfolded2vertex_map[i] for i in face_indices]
            src_face = MeshFace(list(src_face_indices))
            for class_index, face_class in enumerate(self.face_classes.values()):
                if src_face in face_class:
                    return class_index
            return 0

        ibpy.set_color_to_faces(bob, lambda x: find_face_class(x))
        for slot, rep in enumerate(self.face_classes.keys()):
            if len(rep) in [3, 4, 5]:
                ibpy.set_mixer(bob, slot, value=1, begin_time=0)
            else:
                ibpy.set_mixer(bob, slot, value=0, begin_time=0)

        modifier = PolyhedronViewModifier(**kwargs)
        bob.add_mesh_modifier(type="NODES", node_modifier=modifier)
        self.bob = bob
        return bob

    def create_copy_of_net(self, recompute=False, **kwargs) -> BObject:
        """
        The same code as in create_net
        only the result is not linked to self.bob
        and no recomputation is done
        """
        face_classes = self.group.get_faces_in_conjugacy_classes(self.signature)
        color = get_from_kwargs(kwargs, "color", None)
        mats = get_from_kwargs(kwargs, "materials", None)
        if color == "color_dict":
            colors = []
            for face_repr in face_classes.keys():
                colors.append(color_dict[len(face_repr)])
        elif mats:
            colors = mats
        else:
            colors = [color] * len(face_classes)

        # create net as a single mesh instance
        unfolded_faces = []
        for idx, face in enumerate(self.faces):
            v2u_local = self.vertex2unfolded_map[idx]
            unfolded_faces.append([v2u_local[i] for i in face])

        bob = BObject(mesh=ibpy.create_mesh(vertices=self.unfolded_vertices, faces=unfolded_faces), **kwargs)
        for i, col in enumerate(colors):
            if mats:
                for i, col in enumerate(colors):
                    ibpy.set_material(bob, col, slot=i)
            else:
                ibpy.set_material(bob, get_texture(col, **kwargs), slot=i)

        def find_face_class(face_indices):
            src_face_indices = [self.unfolded2vertex_map[i] for i in face_indices]
            src_face = MeshFace(list(src_face_indices))
            for class_index, face_class in enumerate(face_classes.values()):
                if src_face in face_class:
                    return class_index
            return 0

        ibpy.set_color_to_faces(bob, lambda x: find_face_class(x))
        for slot, rep in enumerate(self.face_classes.keys()):
            if len(rep) in [3, 4, 5]:
                ibpy.set_mixer(bob, slot, value=1, begin_time=0)
            else:
                ibpy.set_mixer(bob, slot, value=0, begin_time=0)
        modifier = PolyhedronViewModifier()
        bob.add_mesh_modifier(type="NODES", node_modifier=modifier)
        return bob

    def morph(self, other: Unfolder, begin_time=0,
              transition_time=DEFAULT_ANIMATION_TIME):
        src2target_vertex_map = self.create_map(other)
        target_vertices = [v for v in other.vs]
        self.bob.index_transform_mesh(lambda idx: target_vertices[src2target_vertex_map[idx]], begin_time=begin_time,
                                      transition_time=transition_time)

        # create src2target_face_map (TODO)
        target_faces = [MeshFace(face) for face in other.faces]
        src2target_face_map = defaultdict(set)
        for src_id, face in enumerate(self.faces):
            target_face_indices = [src2target_vertex_map[vertex_id] for vertex_id in face]
            unique_indices = []
            for index in target_face_indices:
                if index not in unique_indices:
                    unique_indices.append(index)
            target_face = MeshFace(unique_indices)
            if target_face in target_faces:
                target_id = target_faces.index(target_face)
                src2target_face_map[src_id].add(target_id)

        self._morph_colors(other, src2target_face_map, begin_time=begin_time,
                           transition_time=transition_time)

        return begin_time + transition_time

    def compute_normals(self):
        normals = []
        for face in self.faces:
            v1 = self.vertices[face[1]] - self.vertices[face[0]]
            v2 = self.vertices[face[2]] - self.vertices[face[0]]
            center = sum(self.vertices, Vector()) / len(self.vertices)
            normal = v1.cross(v2)
            normal.normalize()

            # make sure that normal points outwards (not helpful, rely on face data)
            # if center.dot(normal)<0:
            #     normal*=-1
            # normals.append(normal)

        return normals

    def create_tree(self, root_index):
        if root_index == -1:
            root_index = 0
        if root_index >= len(self.faces):
            root_index = -1

        index_to_face_map = {i: face for i, face in enumerate(self.faces)}
        index_to_normal_map = {i: normal for i, normal in enumerate(self.normals)}

        root_face = self.faces[root_index]
        root = Node((root_index, len(root_face), set(root_face), self.normals[root_index]))

        parents = [root]
        index_to_face_map.pop(root_index)

        while len(index_to_face_map) > 0:
            next_level = []
            for parent in parents:
                new_nodes = []
                for index, face in index_to_face_map.items():
                    face_set = set(face)
                    if len(face_set & parent.name[2]) == 2:  # common edge
                        node = Node((index, len(face), face_set, index_to_normal_map[index]), parent=parent)
                        new_nodes.append(node)
                        next_level.append(node)

                # pop children
                for child in new_nodes:
                    index_to_face_map.pop(child.name[0])
            parents = next_level
            print(str(len(parents)) + " new faces added to tree")

        return root

    def show_node_tree(self):
        for pre, _, node in RenderTree(self.root):
            print("%s%s" % (pre, node.name[0]))

    def show_index_structure(self):
        if self.unfolded2vertex_map == {}:
            self.unfold()

        for pre, _, node in RenderTree(self.root):
            print("%s%s" % (pre, self._index_structure(node)))

    def create_map(self, other: Unfolder):
        """
        create a map that relates the three-dimensional vertices of two different shapes
        The map identifies the vertices that are close in three-dimensions.
        """
        injective = False
        other_points = [v for v in other.vs]

        if len(other_points) > len(self.vs):
            vertex_map = defaultdict(list)
            injective = True
        else:
            vertex_map = {}
        if other.radius != self.radius:
            other_points = [self.radius / other.radius * v for v in other_points]

        # map from self to other
        for idx, src in enumerate(self.vs):
            min_dist = np.inf
            min_index = []
            for idx2, target in enumerate(other.vs):
                dist = (src - target).length
                if dist < 0.99 * min_dist:
                    min_dist = dist
                    min_index = [idx2]
                elif np.abs(dist - min_dist) < 1e-2:
                    min_index.append(idx2)
            if injective:
                vertex_map[idx] = min_index
            else:
                vertex_map[idx] = min_index[0]
        return vertex_map

    def rotate_root_to_south_pole(self, root_face):
        # rotate vertices to make the center of the root_face sit on the south-pole
        # compare Household.nb
        center = sum([self.vs[i] for i in root_face], Vector([0, 0, 0]))
        p = center / center.length

        rot = align_vectors(p, Vector([0, 0, -1]))
        rot = Matrix(rot)
        south_pole = rot @ center
        assert (Vector(south_pole[0:2]).length < 1e-4)

        # rotate vertices
        self.vs = [rot @ v for v in self.vs]

    # private functions
    def _compute_normal(self, face) -> Vector:
        e1 = self.vs[face[0]]
        e2 = self.vs[face[1]]
        e3 = self.vs[face[2]]

        u = e2 - e1
        v = e3 - e1
        n = u.cross(v)
        n.normalize()

        # make sure that the normal is pointing outwards
        face_center = sum([self.vs[i] for i in face], Vector())
        face_center /= len(face)

        if n.dot(face_center) < 0:
            n *= -1
        return n

    def _compute_normals(self):
        normals = []
        for face in self.faces:
            n = self._compute_normal(face)
            normals.append(n)
        return normals

    def _recompute_normals(self, node):
        face_size = len(node.name[2])
        indices = list(node.name[2])
        n = self._compute_normal(indices)
        node.name = (node.name[0], node.name[1], node.name[2], n)

        for child in node.children:
            self._recompute_normals(child)

    def _find_root_index(self, mode, skip):
        """
        find root index, depending on the mode
        LARGEST_FACE: choose the face with the largest number of vertices as root face and take the one in position skip
        SMALLEST_FACE: choose the face with the smallest number of vertices as root face and take the one in position skip
        CLOSEST_TO_SOUTH_POLE: choose the face with center closest to (0,0,0,-1)
        """
        if mode == "LARGEST_FACE":
            max_face_size = 0
            for face in self.faces:
                if len(face) > max_face_size:
                    max_face_size = len(face)
            for i, face in enumerate(self.faces):
                if len(face) == max_face_size:
                    if skip == 0:
                        root_face = face
                        root_index = i
                        break
                    else:
                        skip -= 1

        elif mode == "CLOSEST_TO_SOUTH_POLE":
            shortest_dist = np.inf
            closest_face_index = -1

            for face_index, face in enumerate(self.faces):
                center = sum([self.vs[i] for i in face], Vector([0, 0, 0]))
                center = center / center.length

                dist = to_vector((Vector([0, 0, -1]) - center)).length
                if dist < shortest_dist:
                    shortest_dist = dist
                    closest_face_index = face_index

                # adjust normals
                self.faces[face] = Vector(self.faces[face].real())
            root_index = closest_face_index
        elif mode == "SMALLEST_FACE":
            min_face_size = np.inf
            for face in self.faces:
                if len(face) < min_face_size:
                    min_face_size = len(face)
            for i, face in enumerate(self.faces):
                if len(face) == min_face_size:
                    if skip == 0:
                        root_face = face
                        root_index = i
                        break
                    else:
                        skip -= 1
        return root_index

    def find_node(self, node, unfolded_index):
        """
        find the face that hosts the unfolded index
        """
        unfolded_indices = [self.vertex2unfolded_map[node.name[0]][i] for i in node.name[2]]
        if unfolded_index in unfolded_indices:
            return node
        else:
            for child in node.children:
                result = self.find_node(child, unfolded_index)
                if result is not None:
                    return result
        return None

    def _recalculate_vertices_and_normals(self, face_node, rotations=[]):
        """
        The core function that unfolds the faces.

        each rotation is defined by a rotation matrix and an axis of rotation
        we store the rotation data as a dictionary, where the key is the cell id, the value is a tuple that contains
        (two points of the axis and the rotation matrix)
        """

        local_rotations = [rotation for rotation in rotations]  # create a local copy of the list

        if face_node.parent is not None:
            # deal with non-root face (generic case)
            parent = face_node.parent
            face_id = face_node.name[0]
            face_vertex_indices = face_node.name[2]
            common_indices = set(parent.name[2]) & set(face_vertex_indices)
            v2u_local = self.vertex2unfolded_map[face_id]

            # transform all vertices with the previous rotations (all vertices have to betransfromed, since the common
            # indices are stored independently from their parent face

            for idx in face_vertex_indices:
                for rotation in local_rotations:
                    j = v2u_local[idx]
                    self.unfolded_vertices[j] = self._apply_rotation(rotation, self.unfolded_vertices[j])

            # tranform normal of the child face with the previous rotation

            for rotation in local_rotations:
                face_node.name = (face_id, face_node.name[1], face_vertex_indices, rotation[2] @ face_node.name[3])

            # align child and parent normals
            child_normal = face_node.name[3]
            parent_normal = parent.name[3]

            common_vertices = [self.unfolded_vertices[v2u_local[i]] for i in common_indices]
            aligning_rotation = self._get_rotation(common_vertices, child_normal, parent_normal)

            # transform vertices with the last rotation
            for idx in face_vertex_indices - common_indices:
                j = v2u_local[idx]
                v = self._apply_rotation(aligning_rotation, self.unfolded_vertices[j])
                # print(j,": ",self.unfolded_vertices[j],"->",v)
                self.unfolded_vertices[j] = v

            # transform normal with the last rotation
            face_node.name = (face_id, face_node.name[1], face_vertex_indices, aligning_rotation[2] @ face_node.name[3])
            local_rotations.append(aligning_rotation)
        else:
            # no rotation is needed, since the root face need not be unfolded
            local_rotations = []

        #recursively proceed to the children
        for child in face_node.children:
            self._recalculate_vertices_and_normals(child, local_rotations)

    def _apply_rotation(self, rotation, unfolded_vertex):
        """
        the application of the rotation is not a simple matrix multiplication
        since the center of the rotation is not the origin

        from the data of the rotation (two points and a rotation matrix) we can compute
        the projection operator P = u\otimes u^T , where u is the direction of the rotation axis

        the projection operator determines the center of rotation for every vertex

        """

        p = to_vector(rotation[0])
        u = to_vector(rotation[1]) - to_vector(p)
        u.normalize()

        projection = Matrix(np.tensordot(u, u, axes=0))
        v = to_vector(unfolded_vertex)
        delta = v - p
        center = p + projection @ delta

        return rotation[2] @ (v - center) + center

    def _get_rotation(self, common_vertices, child_normal, parent_normal):
        """
        Find a rotation that rotates a into b, while leaving u invariant
        """
        if len(common_vertices) < 2:
            raise "No axis of rotation is found"

        p = common_vertices[0]
        q = common_vertices[1]

        u = to_vector(p) - to_vector(q)
        u.normalize()
        a = to_vector(child_normal)
        a.normalize()
        b = to_vector(parent_normal)
        b.normalize()

        a_perp = a.cross(u)
        b_perp = b.cross(u)
        a_perp.normalize()
        b_perp.normalize()

        rot = np.tensordot(u, u, axes=0) + np.tensordot(b, a, axes=0) + np.tensordot(b_perp, a_perp, axes=0)
        d = np.linalg.det(rot)
        if np.abs(d - 1) > 0.001:
            print(f"{d} is not the determinant of a rotation matrix")
            raise "Abort"
        # print(b, rot @ a)
        return p, q, Matrix(rot)

    def _indexing_unfolded_vertices(self):
        unfolded2vertex_map = {}
        vertex2unfolded_map = {}
        unfolded_vertices = []

        # deal with the vertices of the root face
        new_unfolded_indices = []
        for i in self.root.name[2]:
            j = len(unfolded_vertices)  # get current index
            unfolded2vertex_map[j] = i
            new_unfolded_indices.append(j)
            unfolded_vertices.append(self.vs[i])

        # a map from original vertices to the unfolded vertex makes only sense for each face, separately
        # otherwise the mapping would be multi-valued, the same source point could be mapped to different targets
        local_vertex2unfolded_map = {}
        for j, i in zip(new_unfolded_indices, self.root.name[2]):
            local_vertex2unfolded_map[i] = j
        vertex2unfolded_map[self.root.name[0]] = local_vertex2unfolded_map

        # now the mapping is repeated for each children and there children
        parents = [self.root]
        while len(parents) > 0:
            level_children = []
            for parent in parents:
                level_children += parent.children  # all children are parents of the next level
                for child in parent.children:
                    new_unfolded_indices = []
                    for i in child.name[2]:
                        j = len(unfolded2vertex_map)
                        unfolded2vertex_map[j] = i
                        new_unfolded_indices.append(j)
                        unfolded_vertices.append(self.vs[i])

                    local_vertex2unfolded_map = {}  # create local map for face
                    for j, i in zip(new_unfolded_indices, child.name[2]):
                        local_vertex2unfolded_map[i] = j
                    vertex2unfolded_map[child.name[0]] = local_vertex2unfolded_map
            parents = level_children

        return unfolded2vertex_map, vertex2unfolded_map, unfolded_vertices

    def _index_structure(self, node):
        """
        highlight common indices in boldface
        """
        out = ""
        face_index = node.name[0]
        parent = node.parent
        common_indices = []
        if parent is not None:
            common_indices = list(set(parent.name[2]) & set(node.name[2]))

        for src_vertex_idx in self.faces[face_index]:
            local_vertex2unfolded_map = self.vertex2unfolded_map[face_index]
            if src_vertex_idx in common_indices:
                out += f"\033[1m{src_vertex_idx}\033[0m(\033[91m{local_vertex2unfolded_map[src_vertex_idx]}\033[0m) "
            else:
                out += f"{src_vertex_idx}(\033[91m{local_vertex2unfolded_map[src_vertex_idx]}\033[0m) "
        return out

    def pullback(self, morphed, vertex_map, indices=[], src2target_face_map={}):
        """
        Find the original child that turned into the morphed child

        """
        match = self._find_recursively(self.root, morphed, vertex_map)
        if match is not None:
            indices.append(match.name[0])
            print(morphed.name[0], "->", match.name[0])
            src2target_face_map[match.name[0]] = [morphed.name[0]]
            local_root = Node((match.name[0], match.name[1], match.name[2], match.name[3]))
            for child in morphed.children:
                pulled_child, indices, src2target_cell_map = self.pullback(child, vertex_map, indices,
                                                                           src2target_face_map)
                if pulled_child:
                    pulled_child.parent = local_root
                else:
                    print("no source found for ", child.name[0], child.name[2])
            return local_root, indices, src2target_face_map

        return None

    def find_root(self, other_root, src2target_vertex_map) -> int:
        """
        find the face that turns into the root of the other net under the src2target_vertex_map
        """
        other_root_set = set(other_root.name[2])
        for idx, face in enumerate(self.faces):
            mapped_face = set([src2target_vertex_map[i] for i in face])
            if mapped_face == other_root_set:
                return idx
        return None

    def _find_recursively(self, original, morphed, vertex_map):
        mapped_original = set([vertex_map[o] for o in original.name[2]])
        morphed_indices = set(morphed.name[2])
        if morphed_indices == mapped_original:
            return original
        else:
            for child in original.children:
                match = self._find_recursively(child, morphed, vertex_map)
                if match is not None:
                    return match
        return None

    def fill_gaps(self, node, indices, src2target_face_map):
        removables = []
        new_children = []
        for child in node.children:
            parent_indices = node.name[2]
            child_indices = child.name[2]
            common_indices = set(parent_indices) & set(child_indices)
            if len(common_indices) < 2:
                # we need to find another face that fits in between these to faces
                for idx, face in enumerate(self.faces):
                    if idx not in indices:  # face is not processed yet
                        buffer_indices = set(face)
                        if len(buffer_indices & parent_indices) > 1 and len(buffer_indices & child_indices) > 1:
                            print("added cell as buffer", idx)
                            buffer_node = Node((idx, len(face), set(face), self._compute_normal(face)))
                            src2target_face_map[idx] = [src2target_face_map[node.name[0]][0],
                                                        src2target_face_map[child.name[0]][0]]
                            indices.append(idx)
                            removables.append(child)
                            new_children.append(buffer_node)
                            child.parent = buffer_node

        node.children = list(set(node.children).union(set(new_children)) - set(removables))
        for child in node.children:
            self.fill_gaps(child, indices, src2target_face_map)

        return node, indices, src2target_face_map

    def extend_missing_faces(self, tmp_root, indices, src2target_face_map):
        for idx, face in enumerate(self.faces):
            if idx not in indices:
                success, src2target_face_map = self._add_as_leaf(tmp_root, idx, face, src2target_face_map)
                if success:
                    indices.append(idx)
                    print("added face as leaf", idx)

        # deal with remaining
        for idx in range(len(self.faces)):
            if idx not in indices:
                print("no face found for index", idx)

        return tmp_root, indices, src2target_face_map

    def _add_as_leaf(self, node, face_index, face, src2target_face_map):
        """
        the face can be added, when no other face occupies the edge that the face would be attached to
        """
        # check for common edge
        edge = set(face) & node.name[2]
        # check, whether the edge is occupied
        if len(edge) > 1:
            occupied = False
            for child in node.children:
                if len(edge & child.name[2]) > 1:
                    occupied = True
                    break
            if not occupied:
                new_node = Node((face_index, len(face), set(face), self._compute_normal(face)))
                src2target_face_map[face_index] = [src2target_face_map[node.name[0]][0]]
                new_node.parent = node
                return True, src2target_face_map
        for child in node.children:
            success, src2target_face_map = self._add_as_leaf(child, face_index, face, src2target_face_map)
            if success:
                return True, src2target_face_map
        return False, src2target_face_map

    def _morph_colors(self, other, src2target_face_map, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):

        # select faces that are mapped uniquely
        inverse_face_map = {}
        for key, value in src2target_face_map.items():
            value = list(value)
            if len(value) == 1:
                pre_image = inverse_face_map.get(value[0], [])
                pre_image.append(key)
            inverse_face_map[value[0]] = pre_image

        unique_face_map = dict((key, value[0]) for key, value in inverse_face_map.items() if len(value) == 1)

        # take care of changing colors
        face_class_map = {}
        for target, src in unique_face_map.items():
            target_size = len(other.faces[target])
            src_size = len(self.faces[src])
            for class_index, face_class in enumerate(self.face_classes.values()):
                src_face = MeshFace(self.faces[src])
                if src_face in face_class:
                    face_class_map[class_index] = (src_size, target_size)
                    break

        for slot, (src, target) in face_class_map.items():
            if src != target:
                if target in [6, 8, 10]:
                    ibpy.adjust_mixer(self.bob, slot, from_value=1, to_value=0, begin_time=begin_time,
                                      transition_time=transition_time)
                else:
                    ibpy.adjust_mixer(self.bob, slot, from_value=0, to_value=1, begin_time=begin_time,
                                      transition_time=transition_time)

    # delegates
    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.bob.disappear(begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def rescale(self, rescale=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.bob.rescale(rescale, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def move_to(self, target_location=Vector(), begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.bob.move_to(target_location=target_location, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time


class Unfolder4D:
    """
    Unfolder of geometrical shapes in 4D
    The faces are arranged in a tree structure
    """

    def __init__(self, group=CoxA4, signature=[1, 1, -1, 1], root_index=0, mode="LARGEST_CELL", skip=0, radius=5,
                 **kwargs):
        g = group(path="../mathematics/geometry/data")
        vertices = g.point_cloud(signature)
        self.group_name = g.name
        self.skip = skip
        self.vs = [to_vector(v.real()) for v in vertices]
        self.radius = (self.vs[0].dot(self.vs[0])) ** 0.5
        scale = radius / self.radius
        self.radius = radius
        self.vs = [scale * v for v in self.vs]
        self.stereo = []  # computed in stereo_projection

        self.cells = g.get_cells(signature)
        self.faces = g.get_faces(signature)
        self.edges = g.get_edges(signature)

        self.crystal = get_from_kwargs(kwargs, "crystal", False)

        root_index = self._find_root_index(mode)
        self.root = self._create_full_cell_tree(root_index)
        self.order = self._update_order()

        self.vertex2unfolded_map = None
        self.unfolded2vertex_map = None
        self.unfolded_vertices = None
        self.child_info_map = None
        self.bob = None

    def create_map(self, other):
        """
        create a map that relates the four-dimensional vertices.
        The map identifies the vertices that are close in four-dimensions
        """
        other_points = [v for v in other.vs]
        if other.radius != self.radius:
            other_points = [self.radius / other.radius * v for v in other_points]

        # map from self to other
        vertex_map = {}
        for idx, src in enumerate(self.vs):
            min_dist = np.inf
            min_index = -1
            for idx2, target in enumerate(other_points):
                dist = to_vector((src - target)).length
                if dist < min_dist:
                    min_dist = dist
                    min_index = idx2
            vertex_map[idx] = min_index
        return vertex_map

    def stereo_projection(self):
        if len(self.stereo) == 0:
            def stereo(v):
                return Vector(
                    [v[0] / (1.0001 - v[3] / self.radius), v[1] / (1.0001 - v[3] / self.radius),
                     v[2] / (1.0001 - v[3] / self.radius)])

            self.stereo = [stereo(v) for v in self.vs]
        return self.stereo

    def create_stereo_bob(self, **kwargs):
        half_way = get_from_kwargs(kwargs, "half_way", False)
        y_limit = get_from_kwargs(kwargs, "y_limit", -0.1)

        children = []
        child_index = 0
        self.child_info_map = {}

        if len(self.stereo) == 0:
            self.stereo = self.stereo_projection()

        for cell_index in self.order:
            cell = list(self.cells.keys())[cell_index]
            verts = [self.stereo[i] for i in cell]
            center = sum(verts, Vector()) / len(verts)

            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in self.faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            if cell_index == self.order[-1]:
                scale = 1.01  # make last cell a bit larger to avoid uneven rendering
            else:
                scale = 1

            color = get_color_name(self.group_name, verts, selected_faces, self.crystal)
            cellobj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces, orient_faces=True),
                              name="Cell" + str(cell_index), color=color, scale=scale)

            if self.crystal:
                crystal = CrystalModifier( material=get_texture(color))
                cellobj.add_mesh_modifier(type="NODES", node_modifier=crystal)
                crystal.transfer_material_from(cellobj)
            if not half_way:
                children.append(cellobj)
            else:
                if self.crystal:
                    modifier = crystal
                else:
                    modifier = None
                if center.y >= y_limit and cell_index != self.order[-1]:
                    self.child_info_map[child_index] = ({j: i for j, i in enumerate(cell)}, selected_faces, color,
                                                        modifier)
                    children.append(cellobj)
                    child_index += 1
        return BObject(children=children, **kwargs)

    def create_net(self, **kwargs):
        if self.unfolded_vertices is None:
            self.unfolded2vertex_map, self.vertex2unfolded_map, self.unfolded_vertices = self._unfold()

        children = []
        child_index = 0
        self.child_info_map = {}

        for idx in self.order:
            cell = list(self.cells.keys())[idx]
            v2u_local = self.vertex2unfolded_map[idx]

            verts = [self.unfolded_vertices[v2u_local[i]] for i in cell]
            # redefine vertex face mapping
            local_map = {i: j for j, i in enumerate(cell)}

            selected_faces = []
            for face in self.faces:
                if set(face) < set(cell):
                    selected_faces.append([local_map[i] for i in face])

            color = get_color_name(self.group_name, verts, selected_faces, self.crystal)
            cellobj = BObject(mesh=ibpy.create_mesh(vertices=verts, faces=selected_faces, orient_faces=True),
                              name="Cell" + str(idx),
                              color=color)

            if self.crystal:
                crystal = CrystalModifier()
                cellobj.add_mesh_modifier(type="NODES", node_modifier=crystal)
                crystal.transfer_material_from(cellobj)
                modifier = crystal
            else:
                modifier = None

            self.child_info_map[child_index] = ({j: i for j, i in enumerate(cell)}, selected_faces, color, modifier,
                                                idx)
            child_index += 1
            children.append(cellobj)

        return BObject(children=children, **kwargs)

    def show_node_tree(self):
        for pre, _, node in RenderTree(self.root):
            if node.parent is not None:
                parent_idx = set(node.parent.name[2])
                child_idx = set(node.name[2])
                intersection_idx = parent_idx.intersection(child_idx)
                remaining_idx = child_idx - intersection_idx
                print("%s%s%s%s" % (pre, node.name[0], intersection_idx, remaining_idx))
            else:
                print("%s%s%s" % (pre, node.name[0], node.name[2]))

    def show_index_structure(self):
        print(
            "\033[1mCell tree for the polytope\033[0m\nblack: original vertex indices\nred: unfolded vertex indices\n\033[1mbold\033[0m: common indices")
        if self.unfolded2vertex_map is None:
            self.unfolded2vertex_map, self.vertex2unfolded_map, self.unfolde_vertices = self._unfold()

        for pre, _, node in RenderTree(self.root):
            print("%s%s" % (pre, self._index_structure(node)))

    def remove_half_the_pieces(self, bob, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        # center children first
        children = bob.b_children
        for child in children:
            ibpy.set_origin(child)

        # remove last child
        dt = transition_time / len(children) * 2
        t0 = children[-1].disappear(begin_time=begin_time, transition_time=dt)

        for child in children[:-1]:
            location = child.get_location()
            if location[1] < -0.1:
                t0 = child.disappear(begin_time=t0, transition_time=dt)

        return begin_time + transition_time

    def morph_net(self, other, begin_time=0, transition_time1=DEFAULT_ANIMATION_TIME,
                  transition_time2=DEFAULT_ANIMATION_TIME):

        src2target_vertex_map = self.create_map(other)  # compute mapping between self and other based on 4d proximity
        if other.unfolded_vertices is None:
            other.create_net()

        other_vertex2unfolded_map = other.vertex2unfolded_map
        other_unfolded_vertices = other.unfolded_vertices

        # pullback morphed tree to src tree
        other_root = other.root
        tmp_root, indices, src2target_cell_map = self._pullback(self.root, other_root, src2target_vertex_map, [], {})
        # squeeze in buffers when possible
        tmp_root, indices, src2target_cell_map = self._fill_gaps(tmp_root, indices, src2target_cell_map)

        # add remaining as leaves
        tmp_root, indices, src2target_cell_map = self._extend_missing_cells(tmp_root, indices, src2target_cell_map)

        # install adjusted root tree
        self.show_node_tree()
        self.root = tmp_root
        self.order = self._update_order()
        self.show_node_tree()

        self.bob = self.create_net(location=[0, 0, 0])
        self.bob.appear(begin_time=begin_time, transition_time=transition_time1, sequentially=True)

        begin_time = begin_time + transition_time1
        transition_time = transition_time2

        children = self.bob.b_children
        for child_index, child in enumerate(children):
            net2vertex = self.child_info_map[child_index][0]
            selected_faces = self.child_info_map[child_index][1]
            color = self.child_info_map[child_index][2]
            modifier = self.child_info_map[child_index][3]
            src_order_idx = self.child_info_map[child_index][4]
            src_vertex2unfolded = self.vertex2unfolded_map[src_order_idx]
            local_vertex2unfolded = {}
            for cell_index in src2target_cell_map[src_order_idx]:
                local_vertex2unfolded = local_vertex2unfolded | other_vertex2unfolded_map[cell_index]

            def morph_map_to_target(cell_size, selected_faces, i):
                try:
                    res = other_unfolded_vertices[local_vertex2unfolded[src2target_vertex_map[net2vertex[i]]]]
                    return res
                except:
                    # this is more complicated. Sometimes a cell doesn't have a parent into the shrinking direction
                    # it has to collapse onto itself. Only a few edges indicate the direction of collapse
                    # the transformation of the few vertices has to be extended onto all vertices of the cell

                    # this deals with prisms
                    # identify the indices of the two faces that collide
                    indices = list(range(cell_size))
                    max_face = 0
                    first = None
                    second = None
                    for face in selected_faces:
                        if len(face) > max_face:
                            max_face = len(face)

                    for face in selected_faces:
                        if len(face) == max_face:
                            if first is None:
                                first = face
                            else:
                                second = face
                    # find the fixed points that define the transformation
                    target_indices = [src2target_vertex_map[net2vertex[i]] for i in indices]

                    mapping = {}
                    for face_idx, trans_idx in enumerate(target_indices):
                        if trans_idx in local_vertex2unfolded.keys():
                            mp = mapping.get(trans_idx, [])
                            mp.append(face_idx)
                            mapping[trans_idx] = mp

                    # use first key value pair to define face shifts
                    for target_index, src_indices in mapping.items():
                        break

                    common_idx = set(first) & set(src_indices)
                    shift_first = other_unfolded_vertices[local_vertex2unfolded[target_index]] - self.unfolded_vertices[
                        src_vertex2unfolded[net2vertex[common_idx.pop()]]]
                    common_idx2 = set(second) & set(src_indices)
                    shift_second = other_unfolded_vertices[local_vertex2unfolded[target_index]] - \
                                   self.unfolded_vertices[src_vertex2unfolded[net2vertex[common_idx2.pop()]]]

                    # compute the final position for each index
                    if i in first:
                        return self.unfolded_vertices[src_vertex2unfolded[net2vertex[i]]] + shift_first
                    else:
                        return self.unfolded_vertices[src_vertex2unfolded[net2vertex[i]]] + shift_second

            def morph_map_to_target_index(i):
                try:
                    res = local_vertex2unfolded[src2target_vertex_map[net2vertex[i]]]
                    return res
                except:
                    return 0  # cell is doomed to vanish

            child.index_transform_mesh(partial(morph_map_to_target, len(net2vertex), selected_faces),
                                       begin_time=begin_time, transition_time=transition_time)

            # identify the structure of morphed cell
            morphed_selected_faces = []
            for face in selected_faces:
                morphed_face = set()
                for idx in face:
                    morphed_face.add(morph_map_to_target_index(idx))
                if len(morphed_face) > 2:
                    morphed_selected_faces.append(morphed_face)

            if len(morphed_selected_faces) < 4:
                child.disappear(begin_time=begin_time, transition_time=transition_time)
                # pass
            else:
                all_indices = set(sum([list(set) for set in morphed_selected_faces], []))
                new_color = get_color_name(self.group_name, all_indices, morphed_selected_faces)
                if self.crystal:
                    new_color = "crystal_" + new_color

                mixed_material = mix_texture(material1=color, material2=new_color)
                ibpy.set_material(child, mixed_material, slot=0)
                factor_node = get_node_from_shader(mixed_material, "FinalMixShader")
                ibpy.change_default_value(factor_node.inputs["Factor"], 0, 1, begin_time=begin_time,
                                          transition_time=transition_time)
                if modifier:
                    modifier.transfer_material_from(child)

        return begin_time + transition_time1 + transition_time2

    def morph_stereo(self, stereo_bob, other, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        map = self.create_map(other)  # compute mapping between self and other
        target = other.stereo  # get target positions

        children = stereo_bob.b_children
        for child_index, child in enumerate(children):
            stereo2vertex = self.child_info_map[child_index][
                0]  # map from child-mesh index back to original 4D vertex index
            selected_faces = self.child_info_map[child_index][1]
            color = self.child_info_map[child_index][2]
            modifier = self.child_info_map[child_index][3]
            morph_map_to_target = lambda idx: target[map[stereo2vertex[idx]]]
            morph_map_to_target_index = lambda idx: map[stereo2vertex[idx]]
            child.index_transform_mesh(morph_map_to_target, begin_time=begin_time, transition_time=transition_time)
            ibpy.set_origin(child)
            child.rescale(rescale=0.975, begin_time=0, transition_time=0)

            # identify morph structure
            morphed_selected_faces = []
            for face in selected_faces:
                morphed_face = set()
                for idx in face:
                    morphed_face.add(morph_map_to_target_index(idx))
                if len(morphed_face) > 2:
                    morphed_selected_faces.append(morphed_face)

            if len(morphed_selected_faces) < 4:
                child.disappear(begin_time=begin_time, transition_time=transition_time)
            else:
                all_indices = set(sum([list(set) for set in morphed_selected_faces], []))
                new_color = get_color_name("A4", all_indices, morphed_selected_faces)
                if self.crystal:
                    new_color = "crystal_" + new_color

                mixed_material = mix_texture(material1=color, material2=new_color)
                ibpy.set_material(child, mixed_material, slot=0)
                factor_node = get_node_from_shader(mixed_material, "FinalMixShader")
                ibpy.change_default_value(factor_node.inputs["Factor"], 0, 1, begin_time=begin_time,
                                          transition_time=transition_time)
                if modifier:
                    modifier.transfer_material_from(child)

        return begin_time + transition_time

    def _find_root_index(self, mode):
        """
        find root index, depending on the mode
        LARGEST_CELL: choose the cell with the largest number of vertices as root cell and take the one in position skip
        SMALLEST_CELL: choose the cell with the smallest number of vertices as root cell and take the one in position skip
        CLOSEST_TO_SOUTH_POLE: choose the cell with center closest to (0,0,0,-1)
        """
        skip = self.skip
        if mode == "LARGEST_CELL":
            max_cell_size = 0
            for cell in self.cells.keys():
                if len(cell) > max_cell_size:
                    max_cell_size = len(cell)

            for i, cell in enumerate(self.cells.keys()):
                if len(cell) == max_cell_size:
                    if skip == 0:
                        root_cell = cell
                        root_index = i
                        break
                    else:
                        skip -= 1

            # rotate vertices to make the center of the root_cell sit on the south pole
            # compare Household.nb
            center = sum([self.vs[i] for i in root_cell], Vector([0, 0, 0, 0]))
            p = center / center.length

            rot = align_vectors(p, Vector([0, 0, 0, -1]))
            south_pole = rot @ center
            assert (Vector(south_pole[0:3]).length < 1e-4)

            # rotate vertices
            self.vs = [rot @ v for v in self.vs]

            # update normals
            for cell, normal in self.cells.items():
                i = 3
                new_normal = Vector([0, 0, 0, 0])
                while new_normal.dot(
                        new_normal) == 0:  # the while loop makes sure that the four selected points are linearly independent
                    e0 = self.vs[cell[0]]
                    e1 = self.vs[cell[1]]
                    e2 = self.vs[cell[2]]
                    e3 = self.vs[cell[i]]
                    u = e1 - e0
                    v = e2 - e0
                    w = e3 - e0

                    new_normal = self._normal_to(u, v, w)
                    i = i + 1
                new_normal = new_normal / np.sqrt(new_normal.dot(new_normal))

                center = sum([to_vector(self.vs[i]) for i in cell], Vector([0, 0, 0, 0])) / len(cell)

                if new_normal.dot(center) < 0:
                    new_normal *= -1
                self.cells[cell] = new_normal
        elif mode == "CLOSEST_TO_SOUTH_POLE":
            shortest_dist = np.inf
            closest_cell_index = -1

            for cell_index, cell in enumerate(self.cells.keys()):
                center = sum([self.vs[i] for i in cell], Vector([0, 0, 0, 0]))
                center = center / center.length

                dist = to_vector((Vector([0, 0, 0, -1]) - center)).length
                if dist < shortest_dist:
                    shortest_dist = dist
                    closest_cell_index = cell_index

                # adjust normals
                self.cells[cell] = Vector(self.cells[cell].real())

            root_index = closest_cell_index
        elif mode == "SMALLEST_CELL":
            min_cell_size = np.inf
            for cell in self.cells.keys():
                if len(cell) < min_cell_size:
                    min_cell_size = len(cell)

            for i, cell in enumerate(self.cells.keys()):
                if len(cell) == min_cell_size:
                    if skip == 0:
                        root_cell = cell
                        root_index = i
                        break
                    else:
                        skip -= 1

            # rotate vertices to make the center of the root_cell sit on the south pole
            # compare Household.nb
            center = sum([self.vs[i] for i in root_cell], Vector([0, 0, 0, 0]))
            p = center / center.length

            rot = align_vectors(p, Vector([0, 0, 0, -1]))
            south_pole = rot @ center
            assert (Vector(south_pole[0:3]).length < 1e-4)

            # rotate vertices
            self.vs = [rot @ v for v in self.vs]

            # update normals
            for cell, normal in self.cells.items():
                e0 = self.vs[cell[0]]
                e1 = self.vs[cell[1]]
                e2 = self.vs[cell[2]]
                e3 = self.vs[cell[3]]
                u = e1 - e0
                v = e2 - e0
                w = e3 - e0

                new_normal = self._normal_to(u, v, w)
                new_normal = new_normal / np.sqrt(new_normal.dot(new_normal))

                center = sum([to_vector(self.vs[i]) for i in cell], Vector([0, 0, 0, 0])) / len(cell)

                if new_normal.dot(center) < 0:
                    new_normal *= -1
                self.cells[cell] = new_normal
        return root_index

    def _update_order(self):
        # create order
        self.order = [self.root.name[0]]
        next_level = self.root.children
        while len(next_level) > 0:
            next_level = sorted(next_level, key=lambda x: len(x.name[2]))
            new_level = []
            for child in next_level:
                self.order.append(child.name[0])  # append children from small to large

            # next_level = sorted(next_level,key=lambda x:-len(x.name[2]))
            for child in next_level:
                # add first children of large children
                new_level.extend(child.children)
            next_level = new_level
        return self.order

    def _index_structure(self, node):
        """
        highlight common indices in boldface
        """
        out = ""
        cell_indices = node.name[2]
        cell_index = node.name[0]
        parent = node.parent
        common_indices = []
        if parent is not None:
            common_indices = list(set(parent.name[2]) & set(node.name[2]))

        for src_vertex_idx in cell_indices:
            local_vertex2unfolded_map = self.vertex2unfolded_map[cell_index]
            if src_vertex_idx in common_indices:
                out += f"\033[1m{src_vertex_idx}\033[0m(\033[91m{local_vertex2unfolded_map[src_vertex_idx]}\033[0m) "
            else:
                out += f"{src_vertex_idx}(\033[91m{local_vertex2unfolded_map[src_vertex_idx]}\033[0m) "
        return out

    def _create_full_cell_tree(self, root_index=-1, root_size=-1):

        cells = self.cells

        index_to_cell_map = {i: cell for i, cell in enumerate(cells.keys())}
        index_to_normal_map = {i: cell for i, cell in enumerate(cells.values())}

        if root_index == -1:
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

        root = Node((root_cell_index, len(root_cell), set(root_cell), root_normal))

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
                        node = Node((index, len(cell_set), cell_set, index_to_normal_map[index]), parent=parent)
                        new_nodes.append(node)
                        next_level.append(node)

                # pop children
                for child in new_nodes:
                    index_to_cell_map.pop(child.name[0])

            parents = next_level
            print(str(len(parents)) + "new children added ")

        return root

    def _unfold(self):
        """
        each vertex can have various unfolded images, depending on the cell that is under consideration
        the map is bijective when restricted to a single cell, therefore, we store the corresponding map for each cell
        the inverse map is injective, each unfolded vertex index belongs has a unique vertex index
        """

        unfolded2vertex_map, vertex2unfolded_map, unfolded_vertices = self._indexing_unfolded_vertices()
        self._recalculate_vertices_and_normals(self.root,
                                               unfolded_vertices,
                                               unfolded2vertex_map,
                                               vertex2unfolded_map,
                                               [])
        self._rotate_into_w_zero_space(unfolded_vertices, vertex2unfolded_map)
        return unfolded2vertex_map, vertex2unfolded_map, [to_vector(v[0:3]) for v in unfolded_vertices]

    def _indexing_unfolded_vertices(self):
        """
        The process of unfolding produces additional geometry
        These additional vertices have to be indexed and stored for later use
        Also their relation to the original vertices has to be preserved
        The map is only bijective for a single cell, therefore, an individual map is stored for each cell
        """

        root = self.root
        unfolded2vertex_map = {}
        vertex2unfolded_map = {}
        unfolded_vertices = []  # right now, it is just a copy of the original vertices
        new_unfolded_indices = []
        for i in root.name[2]:
            j = len(unfolded2vertex_map)
            unfolded2vertex_map[j] = i
            new_unfolded_indices.append(j)
            unfolded_vertices.append(self.vs[i])

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
                        unfolded_vertices.append(self.vs[i])

                    local_vertex2unfolded_map = {}
                    for j, i in zip(new_unfolded_indices, child.name[2]):
                        local_vertex2unfolded_map[i] = j
                    vertex2unfolded_map[child.name[0]] = local_vertex2unfolded_map
            parents = level_children
            print("reindex", len(parents), "children")
        return unfolded2vertex_map, vertex2unfolded_map, unfolded_vertices

    def _recalculate_vertices_and_normals(self, cell, unfolded_vertices, unfolded2vertex_map, vertex2unfolded_map,
                                          rotations):
        """
         deal with root
         each rotation is defined by a rotation matrix and a plane of rotation
         (similarly as in three dimensions a rotation is defined by a rotation matrix and an axis of rotation)
         (similarly as in two dimensions a rotation is defined by a rotation matrix and a point of rotation)
         we store the rotation data as a dictionary, where the key is the cell id, the value is a tuple that contains
         (three points of the plane and the rotation matrix)
        """
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
                    unfolded_vertices[j] = self._apply_rotation(rotation, unfolded_vertices[j])

            # transform normal of child cell with previous rotation
            for rotation in local_rotations:
                cell.name = (cell.name[0], cell.name[1], cell.name[2], rotation[3] @ cell.name[3])

            # align child and parent normals
            child_normal = cell.name[3]
            parent_normal = parent.name[3]

            common_vertices = [unfolded_vertices[v2u_local[i]] for i in common_indices]
            new_rotation = self._get_rotation(common_vertices, child_normal, parent_normal)

            # transform vertices with the last rotation
            for i in cell.name[2] - common_indices:
                j = v2u_local[i]
                unfolded_vertices[j] = self._apply_rotation(new_rotation, unfolded_vertices[j])
            # transform normal with the last rotation
            cell.name = (cell.name[0], cell.name[1], cell.name[2], new_rotation[3] @ cell.name[3])
            local_rotations.append(new_rotation)
        else:
            # no rotation is needed, since the root cell need not be unfolded.
            local_rotations = []

        # recursively proceed to the children
        for child in cell.children:
            self._recalculate_vertices_and_normals(child, unfolded_vertices, unfolded2vertex_map,
                                                   vertex2unfolded_map, local_rotations)

    def _get_rotation(self, common_vertices, a, b):
        """
         Find rotation that rotates a to b, while leaving e1 and e2 invariant.
        """
        # create ortho-normal base from e1 and e2
        if len(common_vertices) < 3:
            raise "Something's wrong with the common face"
        p1 = to_vector(common_vertices[0])
        p2 = to_vector(common_vertices[1])
        p3 = to_vector(common_vertices[2])
        e1 = p2 - p1
        e2 = p3 - p1

        u = e1 / np.sqrt(e1.dot(e1))
        v = (e2 - u.dot(e2) * u)
        v = v / np.sqrt(v.dot(v))
        a = a / np.sqrt(a.dot(a))
        b = b / np.sqrt(b.dot(b))

        # create common face normals for each cell
        na = self._normal_to(u, v, a)
        na = na / np.sqrt(na.dot(na))
        nb = self._normal_to(u, v, b)
        nb = nb / np.sqrt(nb.dot(nb))

        # the rotation matrix is quite easy to understand
        # u->u
        # v->v
        # na->nb
        # a->b

        rot = np.tensordot(u, u, axes=0) + np.tensordot(v, v, axes=0) + np.tensordot(nb, na, axes=0) + np.tensordot(b,
                                                                                                                    a,
                                                                                                                    axes=0)
        return (p1, p2, p3, Matrix(rot))  # return matrix that transforms the vertices

    def _normal_to(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        abc = np.tensordot(np.tensordot(a, b, axes=0), c, axes=0)
        n = np.tensordot(epsilon(4), abc, axes=[[1, 2, 3], [0, 1, 2]])
        return Vector(n)

    def _apply_rotation(self, rotation, vertex):
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
        p = to_vector(p)
        projection = Matrix(np.tensordot(u, u, axes=0) + np.tensordot(v, v, axes=0))
        vertex = to_vector(vertex)
        delta = vertex - p
        center = p + projection @ delta

        return rotation[3] @ (vertex - center) + center

    def _rotate_into_w_zero_space(self, unfolded_vertices, vertex2unfolded_map):
        """
        all unfolded vertices have to be rotated by a global rotation
        """
        normal = self.root.name[3]
        rotation = align_vectors(normal, Vector([0, 0, 0, 1]))

        # rotate vertices around the center of the root cell
        root_indices = self.root.name[2]
        v2u_local = vertex2unfolded_map[self.root.name[0]]
        root_vertices = [to_vector(unfolded_vertices[v2u_local[i]]) for i in root_indices]
        center_of_root = sum(root_vertices, Vector([0, 0, 0, 0])) / len(root_vertices)

        for i, v in enumerate(unfolded_vertices):
            unfolded_vertices[i] = to_vector(rotation @ (to_vector(v) - center_of_root) + center_of_root)

        # translate center of mass to origin
        center = sum(unfolded_vertices, Vector([0, 0, 0, 0])) / len(unfolded_vertices)
        for i, v in enumerate(unfolded_vertices):
            unfolded_vertices[i] = v - center

    def _pullback(self, original_root, morphed, vertex_map, indices=[], src2target_cell_map={}):
        """
        Find the original child that turned into the morphed child
        The pullback of the indices should be contained in the original indices
        """
        match = self._find_recursively(original_root, morphed, vertex_map)
        if match is not None:
            indices.append(match.name[0])
            print(morphed.name[0], "->", match.name[0])
            src2target_cell_map[match.name[0]] = [morphed.name[0]]
            local_root = Node((match.name[0], match.name[1], match.name[2], match.name[3]))
            for child in morphed.children:
                pulled_child, indices, src2target_cell_map = self._pullback(original_root, child, vertex_map, indices,
                                                                            src2target_cell_map)
                if pulled_child:
                    pulled_child.parent = local_root
                else:
                    print("no source found for ", child.name[0], child.name[2])
            return local_root, indices, src2target_cell_map

    def _find_recursively(self, original, morphed, map):
        mapped_original = set([map[o] for o in original.name[2]])
        morphed_indices = set(morphed.name[2])
        if mapped_original == morphed_indices:
            return original
        else:
            for child in original.children:
                source = self._find_recursively(child, morphed, map)
                if source:
                    return source
        return None

    def _extend_missing_cells(self, tmp_root, indices, src2target_cell_map):
        """
        try to append missing cells to the tree:
        add them as leaves
        """

        # first try to add them as leaves
        for cell_index, (cell_indices, cell_normal) in enumerate(self.cells.items()):
            if cell_index not in indices:
                success, src2target_cell_map = self._add_as_leaf(tmp_root, cell_index, cell_indices, cell_normal,
                                                                 src2target_cell_map)
                if success:
                    indices.append(cell_index)
                    print("added cell as leaf", cell_index)
        # deal with remaining cells
        for cell_index, (cell_indices, cell_normal) in enumerate(self.cells.items()):
            if cell_index not in indices:
                raise "Not all cells have been added to tree"

        return tmp_root, indices, src2target_cell_map

    def _add_as_leaf(self, node, cell_index, cell_indices, cell_normal, src2target_cell_map):
        if len(node.children) == 0:
            if len(set(cell_indices).intersection(set(node.name[2]))) > 2:
                new_node = Node((cell_index, len(cell_indices), set(cell_indices), cell_normal), parent=node)
                src2target_cell_map[cell_index] = [src2target_cell_map[node.name[0]][0]]
                new_node.parent = node
                return True, src2target_cell_map
        for child in node.children:
            success, src2target_cell_map = self._add_as_leaf(child, cell_index, cell_indices, cell_normal,
                                                             src2target_cell_map)
            if success:
                return True, src2target_cell_map
        return False, src2target_cell_map

    def _fill_gaps(self, node, indices, src2target_cell_map):
        removables = []
        new_children = []
        for child in node.children:
            parent_idx = set(node.name[2])
            child_idx = set(child.name[2])
            intersection_idx = parent_idx.intersection(child_idx)
            if len(intersection_idx) < 3:
                for idx, (cell_indices, cell_normal) in enumerate(self.cells.items()):
                    if idx not in indices:
                        buffer_idx = set(cell_indices)
                        if len(buffer_idx.intersection(parent_idx)) > 2 and len(buffer_idx.intersection(child_idx)) > 2:
                            print("added cell as buffer", idx)
                            buffer_node = Node((idx, len(cell_indices), set(cell_indices), cell_normal))
                            src2target_cell_map[idx] = [src2target_cell_map[node.name[0]][0],
                                                        src2target_cell_map[child.name[0]][0]]
                            indices.append(idx)
                            removables.append(child)
                            new_children.append(buffer_node)
                            child.parent = buffer_node
        node.children = list(set(node.children).union(set(new_children)) - set(removables))
        for child in node.children:
            self._fill_gaps(child, indices, src2target_cell_map)

        return node, indices, src2target_cell_map


# auxiliary functions

def align_vectors(src, dest):
    """
    create a minimal rotation that rotates the vector src into the vector dest
    """
    dest = to_vector(dest)
    n = -dest / dest.length
    u = src / src.length - n
    if u.dot(u) == 0:
        return np.identity(len(src))
    u = u / u.dot(u) ** 0.5
    u = np.array(u[0:])
    n = np.array(n[0:])

    H = np.identity(len(src)) - 2 * np.tensordot(u, u, axes=0)
    R = np.identity(len(src)) - 2 * np.tensordot(n, n, axes=0)

    return Matrix(R @ H)


def rounded_vector(v, digits):
    return [round(comp, digits) for comp in v]


def find_optimal_rotation(center_cell, center_cell_normal, south_pole, faces, vertices):
    """
    There are many ways to rotate the center_cell_normal to the south_pole direction.
    We also impose that the normal of the largest face of the center_cell is aligned with a coordinate axis
    and that one edge of this face is aligned to another coordinate axis

    From these requirements we construct the rotation that converts two ortho-normal frames
    """

    o1 = center_cell_normal.real()
    on1 = o1 / o1.length

    selected_faces = []
    max_face_size = 0
    max_face = None
    for face in faces:
        if set(face) < set(center_cell):
            if len(face) > max_face_size:
                max_face_size = len(face)
                max_face = face
            selected_faces.append(face)

    # construct face normal
    v1 = vertices[max_face[0]]
    v2 = vertices[max_face[1]]
    v3 = vertices[max_face[2]]

    e1 = v2 - v1
    e2 = v3 - v1

    e12 = np.tensordot(e1, e2, axes=0)
    e123 = np.tensordot(on1, e12, axes=0)

    o2 = np.tensordot(epsilon_tensor, e123, axes=[[1, 2, 3], [0, 1, 2]])
    o2 = Vector(o2)
    on2 = o2 / o2.length

    # construct an edge normal
    dist = np.inf
    neighbor = None

    for i in range(1, len(max_face)):
        v = vertices[max_face[i]]
        diff = v - v1
        if diff.length < dist:
            neighbor = v
            dist = diff.length

    e1 = neighbor - v1

    e123 = np.tensordot(np.tensordot(e1, on2, axes=0), on1, axes=0)
    o3 = Vector(np.tensordot(epsilon_tensor, e123, axes=[[1, 2, 3], [0, 1, 2]]))
    on3 = o3 / o3.length

    # construct the last ortho-normal vector

    o4 = Vector(np.tensordot(epsilon_tensor, np.tensordot(np.tensordot(on1, on2, axes=0), on3, axes=0),
                             axes=[[1, 2, 3], [0, 1, 2]]))
    on4 = o4 / o4.length

    south_pole_direction = [i for i in range(4) if south_pole[i] != 0][0]
    directions = list(range(4))
    directions.remove(south_pole_direction)
    units = [Vector(u) for u in unit_tuples(4)]

    rotation_matrix = (
            np.tensordot(south_pole, on1, axes=0) +
            np.tensordot(units[directions[-1]], on2, axes=0) +
            np.tensordot(units[directions[-2]], on3, axes=0) +
            np.tensordot(units[directions[-3]], on4, axes=0)
    )

    return Matrix(rotation_matrix)


def compute_cell_size(verts):
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf
    for v in verts:
        min_x = min(min_x, v[0])
        min_y = min(min_y, v[1])
        min_z = min(min_z, v[2])
        max_x = max(max_x, v[0])
        max_y = max(max_y, v[1])
        max_z = max(max_z, v[2])
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    return max(size_x, size_y, size_z)


if __name__ == '__main__':
    """
    get data from truncated hexahedron
    and present tree
    """
    vertices, faces = get_solid_data("CUBE")
    uf = Unfolder(vertices, faces)
    uf.show_node_tree()
    uf.show_index_structure()
