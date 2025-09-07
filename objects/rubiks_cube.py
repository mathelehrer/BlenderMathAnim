import os
from math import pi
import random

from mathutils import Vector, Quaternion, Euler

from geometry_nodes.nodes import RotateInstances, layout, TranslateInstances, RealizeInstances, GeometryToInstance, \
    IndexSwitch
from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier, change_default_boolean, change_default_value, \
    change_default_quaternion, change_default_vector, change_default_rotation, create_mesh, get_material, \
    change_default_integer
from mathematics.mathematica.mathematica import tuples
from mathematics.parsing.parser import parse_int_tuple
from new_stuff.geometry_nodes_modifier import RubiksCubeUnfolded, RubiksCubeModifier, RubiksSphereModifier, \
    RubiksCubeGroupStabChain, RubiksCube4x4UnfoldedModifier, MegaMinxModifier, SingmasterDisplayModifier
from objects.bobject import BObject
from objects.cube import Cube
from objects.platonic_solids import Dodecahedron, SubdividedPentagon
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE, GLOBAL_DATA_DIR
from utils.kwargs import get_from_kwargs

def random_word(length=15):
    length = int(0.5+length*(0.5+0.5*random.random()))
    return simplify(''.join(random.choices('FBUDRLfbudrl', k=length)))

def simplify(word):
    oldword = word
    while True:
        newword=oldword
        newword=newword.replace('Ff','')
        newword=newword.replace('fF','')
        newword=newword.replace('Bb','')
        newword=newword.replace('bB','')
        newword=newword.replace('dD','')
        newword=newword.replace('Dd','')
        newword=newword.replace('uU','')
        newword=newword.replace('Uu','')
        newword=newword.replace('lL','')
        newword=newword.replace('Ll','')
        newword=newword.replace('rR','')
        newword=newword.replace('Rr','')
        if len(oldword)==len(newword):
            break
        oldword=newword

    return newword

class GeoRubiksCube(BObject):
    def __init__(self, name="GeoRubiksCube",**kwargs):
        """
        possible customizations
        range: [-1,1]
        orientation: "HORIZONTAL"|"VERTICAL"
        dimension: [1,0.25,0.25]
        location: [0,0,0]
        side_segments: 2
        shape: "cubic" | "cylinder"

        """
        cube = Cube()

        self.rc_geometry = RubiksCubeModifier(name="RubiksCubeModifier", **kwargs)

        self.cube_state = {}
        coords = tuples([0, 1, 2], 3)
        # somewhat awkward list that matches the position of the cubies when the origin of the coordinate system
        # is the (left,front,down) corner and the (2,2,2) corresponds to the (right,back,up) corner
        # the solved state is given by
        cubies = [0, 9, 18, 1, 10, 19, 2, 11, 20, 3, 12, 21, 4, 13, 22, 5, 14, 23, 6, 15, 24, 7, 16, 25, 8, 17, 26]
        for coord, cubie in zip(coords, cubies):
            self.cube_state[coord] = cubie

        self.transformation_maps = {
            "f": {(0, 0, 2): (0, 0, 0), (1, 0, 2): (0, 0, 1), (2, 0, 2): (0, 0, 2), (2, 0, 1): (1, 0, 2),
                  (2, 0, 0): (2, 0, 2), (1, 0, 0): (2, 0, 1), (0, 0, 0): (2, 0, 0), (0, 0, 1): (1, 0, 0),
                  (1, 0, 1): (1, 0, 1)},
            "F": {(0, 0, 2): (2, 0, 2), (0, 0, 1): (1, 0, 2), (0, 0, 0): (0, 0, 2), (1, 0, 0): (0, 0, 1),
                  (2, 0, 0): (0, 0, 0), (2, 0, 1): (1, 0, 0), (2, 0, 2): (2, 0, 0), (1, 0, 2): (2, 0, 1),
                  (1, 0, 1): (1, 0, 1)},
            "l": {(0, 2, 0): (0, 0, 0), (0, 2, 1): (0, 1, 0), (0, 2, 2): (0, 2, 0), (0, 1, 2): (0, 2, 1),
                  (0, 0, 2): (0, 2, 2), (0, 0, 1): (0, 1, 2), (0, 0, 0): (0, 0, 2), (0, 1, 0): (0, 0, 1),
                  (0, 1, 1): (0, 1, 1)},
            "L": {(0, 2, 0): (0, 2, 2), (0, 1, 0): (0, 2, 1), (0, 0, 0): (0, 2, 0), (0, 0, 1): (0, 1, 0),
                  (0, 0, 2): (0, 0, 0), (0, 1, 2): (0, 0, 1), (0, 2, 2): (0, 0, 2), (0, 2, 1): (0, 1, 2),
                  (0, 1, 1): (0, 1, 1)},
            "r": {(2, 0, 0): (2, 2, 0), (2, 0, 1): (2, 1, 0), (2, 0, 2): (2, 0, 0), (2, 1, 2): (2, 0, 1),
                  (2, 2, 2): (2, 0, 2), (2, 2, 1): (2, 1, 2), (2, 2, 0): (2, 2, 2), (2, 1, 0): (2, 2, 1),
                  (2, 1, 1): (2, 1, 1)},
            "R": {(2, 0, 0): (2, 0, 2), (2, 1, 0): (2, 0, 1), (2, 2, 0): (2, 0, 0), (2, 2, 1): (2, 1, 0),
                  (2, 2, 2): (2, 2, 0), (2, 1, 2): (2, 2, 1), (2, 0, 2): (2, 2, 2), (2, 0, 1): (2, 1, 2),
                  (2, 1, 1): (2, 1, 1)},
            "u": {(0, 0, 2): (2, 0, 2), (0, 1, 2): (1, 0, 2), (0, 2, 2): (0, 0, 2), (1, 2, 2): (0, 1, 2),
                  (2, 2, 2): (0, 2, 2), (2, 1, 2): (1, 2, 2), (2, 0, 2): (2, 2, 2), (1, 0, 2): (2, 1, 2),
                  (1, 1, 2): (1, 1, 2)},
            "U": {(0, 0, 2): (0, 2, 2), (1, 0, 2): (0, 1, 2), (2, 0, 2): (0, 0, 2), (2, 1, 2): (1, 0, 2),
                  (2, 2, 2): (2, 0, 2), (1, 2, 2): (2, 1, 2), (0, 2, 2): (2, 2, 2), (0, 1, 2): (1, 2, 2),
                  (1, 1, 2): (1, 1, 2)},
            "d": {(0, 2, 0): (2, 2, 0), (0, 1, 0): (1, 2, 0), (0, 0, 0): (0, 2, 0), (1, 0, 0): (0, 1, 0),
                  (2, 0, 0): (0, 0, 0), (2, 1, 0): (1, 0, 0), (2, 2, 0): (2, 0, 0), (1, 2, 0): (2, 1, 0),
                  (1, 1, 0): (1, 1, 0)},
            "D": {(0, 2, 0): (0, 0, 0), (1, 2, 0): (0, 1, 0), (2, 2, 0): (0, 2, 0), (2, 1, 0): (1, 2, 0),
                  (2, 0, 0): (2, 2, 0), (1, 0, 0): (2, 1, 0), (0, 0, 0): (2, 0, 0), (0, 1, 0): (1, 0, 0),
                  (1, 1, 0): (1, 1, 0)},
            "b": {(2, 2, 0): (0, 2, 0), (2, 2, 1): (1, 2, 0), (2, 2, 2): (2, 2, 0), (1, 2, 2): (2, 2, 1),
                  (0, 2, 2): (2, 2, 2), (0, 2, 1): (1, 2, 2), (0, 2, 0): (0, 2, 2), (1, 2, 0): (0, 2, 1),
                  (1, 2, 1): (1, 2, 1)},
            "B": {(2, 2, 0): (2, 2, 2), (1, 2, 0): (2, 2, 1), (0, 2, 0): (2, 2, 0), (0, 2, 1): (1, 2, 0),
                  (0, 2, 2): (0, 2, 0), (1, 2, 2): (0, 2, 1), (2, 2, 2): (0, 2, 2), (2, 2, 1): (1, 2, 2),
                  (1, 2, 1): (1, 2, 1)}, }

        # The cubies in the center of each face stay in place but will be rotated
        # The cubie in the center of the cube is fixed.
        # The rotation state of each cubie is stored as Euler angles

        self.cubie_rotation_states = {}
        for cubie_idx in range(27):
            self.cubie_rotation_states[cubie_idx] = Quaternion()

        # in the directory self.cube_state, the physical location of the cubies is stored
        # the following transformations act on the physical cube. The directory provides a lookup-table,
        # which cubies are actually affected by the transformation.
        # after the transformation the self.cube_state needs to be updated with the map.

        self.cubie_rotation_angle_map = {
            "f": [Quaternion(Vector([0, 1, 0]), pi / 2),
                  [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 2), (2, 0, 2), (2, 0, 1), (2, 0, 0), (1, 0, 0), (1, 0, 1)]],
            "F": [Quaternion(Vector([0, 1, 0]), -pi / 2),
                  [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 2), (2, 0, 2), (2, 0, 1), (2, 0, 0), (1, 0, 0), (1, 0, 1)]],
            "l": [Quaternion(Vector([1, 0, 0]), pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 1, 2), (0, 0, 2), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)]],
            "L": [Quaternion(Vector([1, 0, 0]), -pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 1, 2), (0, 0, 2), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)]],
            "r": [Quaternion(Vector([1, 0, 0]), -pi / 2),
                  [(2, 0, 0), (2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 1, 2), (2, 0, 2), (2, 0, 1), (2, 1, 1)]],
            "R": [Quaternion(Vector([1, 0, 0]), pi / 2),
                  [(2, 0, 0), (2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 1, 2), (2, 0, 2), (2, 0, 1), (2, 1, 1)]],
            "u": [Quaternion(Vector([0, 0, 1]), -pi / 2),
                  [(0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 1, 2), (2, 0, 2), (1, 0, 2), (1, 1, 2)]],
            "U": [Quaternion(Vector([0, 0, 1]), pi / 2),
                  [(0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 1, 2), (2, 0, 2), (1, 0, 2), (1, 1, 2)]],
            "d": [Quaternion(Vector([0, 0, 1]), pi / 2),
                  [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (2, 1, 0), (2, 0, 0), (1, 0, 0), (1, 1, 0)]],
            "D": [Quaternion(Vector([0, 0, 1]), -pi / 2),
                  [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (2, 1, 0), (2, 0, 0), (1, 0, 0), (1, 1, 0)]],
            "b": [Quaternion(Vector([0, 1, 0]), -pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 0), (1, 2, 0), (1, 2, 1)]],
            "B": [Quaternion(Vector([0, 1, 0]), pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 0), (1, 2, 0), (1, 2, 1)]]
        }

        cube.add_mesh_modifier(type="NODES", node_modifier=self.rc_geometry)

        # get input quaternion nodes for the cubies
        self.cubie_rotation_nodes = [
            ibpy.get_geometry_node_from_modifier(self.rc_geometry, label="CubieRotation_" + str(i)) for i in range(27)]

        super().__init__(obj=cube.ref_obj, name=name, **kwargs)

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        size = get_geometry_node_from_modifier(self.rc_geometry, label="Cube Size")
        change_default_value(size, from_value=0, to_value=0.890, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def transform(self, word,pause=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time / len(word)
        t0 = begin_time
        for letter in word:
            angle, positions = self.cubie_rotation_angle_map[letter]
            transformation = self.transformation_maps[letter]
            # transform relevant cubies
            active_cubies = [self.cube_state[position] for position in positions]
            print(word + ": " + letter + " active cubies for: ", [a + 1 for a in active_cubies])
            for idx in active_cubies:
                from_angle = self.cubie_rotation_states[idx]
                to_angle = angle @ from_angle
                self.cubie_rotation_states[idx] = to_angle
                ibpy.change_default_quaternion(self.cubie_rotation_nodes[idx], from_value=from_angle, to_value=to_angle,
                                               begin_time=t0, transition_time=dt)

            t0 += (dt+pause)
            # update physical state
            new_state = {}
            for src, replacement in transformation.items():
                new_state[src] = self.cube_state[replacement]

            for key, val in new_state.items():
                self.cube_state[key] = val

        return t0

    def edge_parity_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with edge labels 0 and 1
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Edge", "EdgeSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=True, to_value=False, begin_time=begin_time)
        change_default_boolean(switches[6], from_value=False, to_value=True,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=False, to_value=True, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def corner_triality_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with corner labels 0,1 and 2
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Corner", "CornerSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=True, to_value=False, begin_time=begin_time)
        change_default_boolean(switches[6], from_value=False, to_value=True,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=False, to_value=True, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def edge_parity_off(self, begin_time, transition_time):
        """
        undo parity
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Edge", "EdgeSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=False, to_value=True, begin_time=begin_time+transition_time)
        change_default_boolean(switches[6], from_value=True, to_value=False,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=True, to_value=False, begin_time=begin_time )

        return begin_time + transition_time

    def corner_triality_off(self, begin_time, transition_time):
        """
        undo parity
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Corner", "CornerSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=False, to_value=True,
                                   begin_time=begin_time + transition_time)
        change_default_boolean(switches[6], from_value=True, to_value=False,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=True, to_value=False, begin_time=begin_time)

        return begin_time + transition_time

    def cubie_transpositions(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector","Cubie"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        # turn off colors
        for i in range(6):
            change_default_boolean(switches[i], from_value=True, to_value=False,
                                   begin_time=begin_time )
        change_default_boolean(switches[6], from_value=False,to_value=True,begin_time=begin_time+transition_time/2)

        return begin_time+transition_time

    def cubie_transpositions_off(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Cubie"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]


        change_default_boolean(switches[6], from_value=True, to_value=False,
                               begin_time=begin_time )

        for i in range(6):
            change_default_boolean(switches[i], from_value=False, to_value=True,
                                   begin_time=begin_time+ transition_time / 2)

        return begin_time + transition_time

    def face_indices(self,begin_time=0):
        """
        This function makes the face indices visible
        """
        switch_labels = ["Face"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        change_default_boolean(switches[0], from_value=False,to_value=True,begin_time=begin_time)
        return begin_time

class GeoRubiksCubeUnfold(BObject):
    def __init__(self, name="Rubik'sCubeUnfold",**kwargs):
        """
        possible customizations
        range: [-1,1]
        orientation: "HORIZONTAL"|"VERTICAL"
        dimension: [1,0.25,0.25]
        location: [0,0,0]
        side_segments: 2
        shape: "cubic" | "cylinder"

        """
        cube = Cube()
        # take out location from the kwargs, since the argument collides with the node setup
        location = get_from_kwargs(kwargs, "location", [0, 0, 0])

        self.rc_geometry = RubiksCubeUnfolded(name="RubiksCubeUnfoldedModifier", **kwargs)
        cube.add_mesh_modifier(type="NODES", node_modifier=self.rc_geometry)

        self.position_face_map = {i: i for i in range(1, 55)}

        # get instances as the starting point for individual face transformations
        self.last_transform = {i: get_geometry_node_from_modifier(self.rc_geometry, label="InstanceOfFace" + str(i)) for
                               i in range(1, 55)}
        self.final_join = get_geometry_node_from_modifier(self.rc_geometry, label="JoinGeometry")
        super().__init__(obj=cube.ref_obj, name=name, location=location, **kwargs)

    def unfold(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        progressionNode = get_geometry_node_from_modifier(self.rc_geometry, label="UnfoldMeshNode")
        change_default_value(progressionNode.inputs["Progression"], from_value=1.3, to_value=23.9,
                             begin_time=begin_time, transition_time=transition_time)

        show_cubies = get_geometry_node_from_modifier(self.rc_geometry, label="ShowCubies")
        change_default_boolean(show_cubies, from_value=True, to_value=False, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        face_size = get_geometry_node_from_modifier(self.rc_geometry, label="FaceSize")
        change_default_value(face_size, from_value=0, to_value=0.890, begin_time=begin_time,
                             transition_time=transition_time)
        cube_size = get_geometry_node_from_modifier(self.rc_geometry, label="CubeSize")
        change_default_value(cube_size, from_value=0, to_value=0.95, begin_time=begin_time,
                             transition_time=transition_time)

        return begin_time + transition_time

    def rotate_face(self, index, angle, center, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        implement the rotation of individual faces as additional instance rotation nodes
        """
        tree = self.rc_geometry.tree
        links = tree.links

        # create geometry
        last_node = self.last_transform[index]
        final_join = self.final_join

        rotate_instance = RotateInstances(tree, rotation=Vector(), pivot_point=center, local_space=False)
        # cut old link
        for l in last_node.outputs[0].links:
            links.remove(l)
        links.new(last_node.outputs[0], rotate_instance.geometry_in)
        links.new(rotate_instance.geometry_out, final_join.inputs["Geometry"])
        self.last_transform[index] = rotate_instance.node
        # set keyframes
        change_default_rotation(rotate_instance.node.inputs["Rotation"], from_value=Euler(),
                                to_value=Euler([0, 0, angle]), begin_time=begin_time, transition_time=transition_time)

    def rotate_around_pivot(self,rotation_euler=Vector(),pivot=Vector(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        pivot_node = get_geometry_node_from_modifier(self.rc_geometry, label="FinalPivot")
        rotation_node=get_geometry_node_from_modifier(self.rc_geometry,label="FinalRotation")
        change_default_vector(pivot_node,from_value=Vector(),to_value=pivot,begin_time=begin_time,transition_time=transition_time)
        change_default_vector(rotation_node,from_value=Vector(),to_value=rotation_euler,begin_time=begin_time,transition_time=transition_time)

        return begin_time+transition_time

    def only_show_some_faces(self,selected =list(range(55)),begin_time=0):
        face_selector_node = get_geometry_node_from_modifier(self.rc_geometry, label="FaceSelectorSwitch")
        index_map = self.rc_geometry.index_to_face_index

        for i in range(55):
            if index_map[i] not in selected:
                ibpy.change_default_value(face_selector_node.inputs[str(i)],from_value=1,to_value=0,begin_time=begin_time,transition_time=0.1)

        return begin_time

    def only_show_some_labels(self,selected =list(range(55)),begin_time=0):
        face_selector_node = get_geometry_node_from_modifier(self.rc_geometry, label="FaceLabelSwitch")
        index_map = self.rc_geometry.index_to_face_index

        for i in range(55):
            if index_map[i] not in selected:
                ibpy.change_default_value(face_selector_node.inputs[str(i)],from_value=index_map[i],to_value=0,begin_time=begin_time,transition_time=0.1)

        return begin_time

    def hide_some_labels(self,selected=list(range(55)),begin_time=0):
        face_selector_node = get_geometry_node_from_modifier(self.rc_geometry, label="FaceLabelSwitch")
        index_map = self.rc_geometry.index_to_face_index

        for i in range(55):
            if index_map[i] in selected:
                ibpy.change_default_value(face_selector_node.inputs[str(i)],from_value=index_map[i],to_value=0,begin_time=begin_time,transition_time=0.1)

    def translate_face(self, index, translation, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        implement the rotation of individual faces as additional instance translation nodes
        """
        tree = self.rc_geometry.tree
        links = tree.links

        # create geometry
        last_node = self.last_transform[index]
        final_join = self.final_join

        translate_instance = TranslateInstances(tree, translation=translation, local_space=False)
        # cut old link
        for l in last_node.outputs[0].links:
            links.remove(l)
        links.new(last_node.outputs[0], translate_instance.geometry_in)
        links.new(translate_instance.geometry_out, final_join.inputs["Geometry"])
        self.last_transform[index] = translate_instance.node
        # set keyframes
        change_default_value(translate_instance.node.inputs["Translation"], from_value=Vector(),
                             to_value=translation, begin_time=begin_time, transition_time=transition_time)

    def layout_nodes(self):
        layout(self.rc_geometry.tree)

    def transform(self, word, pause=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        The transformations are performed through a simulation node.
        This means that all transformation parameters (angle, translations) have to be entered incrementally per frame
        and they have to be switched on and off in the same frame.
        """
        dt = transition_time / len(word)
        t0 = begin_time
        for letter in word:
            if letter == 'f':
                positions = {19, 20, 21, 22, 23, 24, 25, 26, 27, 10, 13, 16, 28, 31, 34, 37, 40, 43, 46, 49, 52}
                for i in positions:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector(), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[19]
                second = self.position_face_map[22]
                self.position_face_map[19] = self.position_face_map[25]
                self.position_face_map[22] = self.position_face_map[26]
                self.position_face_map[25] = self.position_face_map[27]
                self.position_face_map[26] = self.position_face_map[24]
                self.position_face_map[27] = self.position_face_map[21]
                self.position_face_map[24] = self.position_face_map[20]
                self.position_face_map[21] = first
                self.position_face_map[20] = second
                first = self.position_face_map[16]
                second = self.position_face_map[13]
                third = self.position_face_map[10]
                self.position_face_map[16] = self.position_face_map[52]
                self.position_face_map[13] = self.position_face_map[49]
                self.position_face_map[10] = self.position_face_map[46]
                self.position_face_map[52] = self.position_face_map[34]
                self.position_face_map[49] = self.position_face_map[31]
                self.position_face_map[46] = self.position_face_map[28]
                self.position_face_map[34] = self.position_face_map[43]
                self.position_face_map[31] = self.position_face_map[40]
                self.position_face_map[28] = self.position_face_map[37]
                self.position_face_map[43] = first
                self.position_face_map[40] = second
                self.position_face_map[37] = third

            if letter == 'F':
                faces = {19, 20, 21, 22, 23, 24, 25, 26, 27, 10, 13, 16, 28, 31, 34, 37, 40, 43, 46, 49, 52}
                for i in faces:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector(), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[19]
                second = self.position_face_map[20]
                self.position_face_map[19] = self.position_face_map[21]
                self.position_face_map[20] = self.position_face_map[24]
                self.position_face_map[21] = self.position_face_map[27]
                self.position_face_map[24] = self.position_face_map[26]
                self.position_face_map[27] = self.position_face_map[25]
                self.position_face_map[26] = self.position_face_map[22]
                self.position_face_map[25] = first
                self.position_face_map[22] = second
                first = self.position_face_map[10]
                second = self.position_face_map[13]
                third = self.position_face_map[16]
                self.position_face_map[10] = self.position_face_map[37]
                self.position_face_map[13] = self.position_face_map[40]
                self.position_face_map[16] = self.position_face_map[43]
                self.position_face_map[37] = self.position_face_map[28]
                self.position_face_map[40] = self.position_face_map[31]
                self.position_face_map[43] = self.position_face_map[34]
                self.position_face_map[28] = self.position_face_map[46]
                self.position_face_map[31] = self.position_face_map[49]
                self.position_face_map[34] = self.position_face_map[52]
                self.position_face_map[46] = first
                self.position_face_map[49] = second
                self.position_face_map[52] = third

            if letter == 'b':
                rotations1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
                rotations2 = {12, 15, 18, 30, 33, 36, 48, 51, 54, 39, 42, 45}
                for i in rotations1:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, -6, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in rotations2:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, 0, 0]), begin_time=t0,
                                     transition_time=dt)

                # update the new face positions
                first = self.position_face_map[1]
                second = self.position_face_map[4]
                self.position_face_map[1] = self.position_face_map[7]
                self.position_face_map[4] = self.position_face_map[8]
                self.position_face_map[7] = self.position_face_map[9]
                self.position_face_map[8] = self.position_face_map[6]
                self.position_face_map[9] = self.position_face_map[3]
                self.position_face_map[6] = self.position_face_map[2]
                self.position_face_map[3] = first
                self.position_face_map[2] = second
                first = self.position_face_map[12]
                second = self.position_face_map[15]
                third = self.position_face_map[18]
                self.position_face_map[12] = self.position_face_map[39]
                self.position_face_map[15] = self.position_face_map[42]
                self.position_face_map[18] = self.position_face_map[45]
                self.position_face_map[39] = self.position_face_map[30]
                self.position_face_map[42] = self.position_face_map[33]
                self.position_face_map[45] = self.position_face_map[36]
                self.position_face_map[30] = self.position_face_map[48]
                self.position_face_map[33] = self.position_face_map[51]
                self.position_face_map[36] = self.position_face_map[54]
                self.position_face_map[48] = first
                self.position_face_map[51] = second
                self.position_face_map[54] = third

            if letter == 'B':
                rotations1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
                rotations2 = {12, 15, 18, 30, 33, 36, 48, 51, 54, 39, 42, 45}
                for i in rotations1:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, -6, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in rotations2:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, 0, 0]), begin_time=t0,
                                     transition_time=dt)

                # update the new face positions
                first = self.position_face_map[1]
                second = self.position_face_map[2]
                self.position_face_map[1] = self.position_face_map[3]
                self.position_face_map[2] = self.position_face_map[6]
                self.position_face_map[3] = self.position_face_map[9]
                self.position_face_map[6] = self.position_face_map[8]
                self.position_face_map[9] = self.position_face_map[7]
                self.position_face_map[8] = self.position_face_map[4]
                self.position_face_map[7] = first
                self.position_face_map[4] = second
                first = self.position_face_map[18]
                second = self.position_face_map[15]
                third = self.position_face_map[12]
                self.position_face_map[18] = self.position_face_map[54]
                self.position_face_map[15] = self.position_face_map[51]
                self.position_face_map[12] = self.position_face_map[48]
                self.position_face_map[54] = self.position_face_map[36]
                self.position_face_map[51] = self.position_face_map[33]
                self.position_face_map[48] = self.position_face_map[30]
                self.position_face_map[36] = self.position_face_map[45]
                self.position_face_map[33] = self.position_face_map[42]
                self.position_face_map[30] = self.position_face_map[39]
                self.position_face_map[45] = first
                self.position_face_map[42] = second
                self.position_face_map[39] = third

            if letter == 'd':
                rotations = {10, 11, 12, 13, 14, 15, 16, 17, 18}
                translations = {39, 38, 37, 19, 22, 25}
                mix1 = {52, 53, 54}
                mix2 = {1, 4, 7}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, -3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([3, 0, 0]), begin_time=t0, transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([1.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([-1.5, -4.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[10]
                second = self.position_face_map[13]
                self.position_face_map[10] = self.position_face_map[16]
                self.position_face_map[13] = self.position_face_map[17]
                self.position_face_map[16] = self.position_face_map[18]
                self.position_face_map[17] = self.position_face_map[15]
                self.position_face_map[18] = self.position_face_map[12]
                self.position_face_map[15] = self.position_face_map[11]
                self.position_face_map[12] = first
                self.position_face_map[11] = second
                first = self.position_face_map[54]
                second = self.position_face_map[53]
                third = self.position_face_map[52]
                self.position_face_map[54] = self.position_face_map[25]
                self.position_face_map[53] = self.position_face_map[22]
                self.position_face_map[52] = self.position_face_map[19]
                self.position_face_map[25] = self.position_face_map[37]
                self.position_face_map[22] = self.position_face_map[38]
                self.position_face_map[19] = self.position_face_map[39]
                self.position_face_map[37] = self.position_face_map[7]
                self.position_face_map[38] = self.position_face_map[4]
                self.position_face_map[39] = self.position_face_map[1]
                self.position_face_map[7] = first
                self.position_face_map[4] = second
                self.position_face_map[1] = third

            if letter == 'D':
                rotations = {10, 11, 12, 13, 14, 15, 16, 17, 18}
                translations = {19, 22, 25, 52, 53, 54}
                mix1 = {39, 38, 37}
                mix2 = {1, 4, 7}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, -3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([-3, 0, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], pi, Vector([-1.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], pi, Vector([1.5, -4.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[10]
                second = self.position_face_map[11]
                self.position_face_map[10] = self.position_face_map[12]
                self.position_face_map[11] = self.position_face_map[15]
                self.position_face_map[12] = self.position_face_map[18]
                self.position_face_map[15] = self.position_face_map[17]
                self.position_face_map[18] = self.position_face_map[16]
                self.position_face_map[17] = self.position_face_map[13]
                self.position_face_map[16] = first
                self.position_face_map[13] = second
                first = self.position_face_map[52]
                second = self.position_face_map[53]
                third = self.position_face_map[54]
                self.position_face_map[52] = self.position_face_map[1]
                self.position_face_map[53] = self.position_face_map[4]
                self.position_face_map[54] = self.position_face_map[7]
                self.position_face_map[1] = self.position_face_map[39]
                self.position_face_map[4] = self.position_face_map[38]
                self.position_face_map[7] = self.position_face_map[37]
                self.position_face_map[39] = self.position_face_map[19]
                self.position_face_map[38] = self.position_face_map[22]
                self.position_face_map[37] = self.position_face_map[25]
                self.position_face_map[19] = first
                self.position_face_map[22] = second
                self.position_face_map[25] = third

            if letter == 'u' or letter=='t':
                rotations = {28, 29, 30, 31, 32, 33, 34, 35, 36}
                translations = {27, 24, 21, 46, 47, 48}
                mix1 = {43, 44, 45}
                mix2 = {3, 6, 9}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, 3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([-3, 0, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], pi, Vector([-4.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([6, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], pi, Vector([3, -6.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([-3, 7, 0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[28]
                second = self.position_face_map[31]
                self.position_face_map[28] = self.position_face_map[34]
                self.position_face_map[31] = self.position_face_map[35]
                self.position_face_map[34] = self.position_face_map[36]
                self.position_face_map[35] = self.position_face_map[33]
                self.position_face_map[36] = self.position_face_map[30]
                self.position_face_map[33] = self.position_face_map[29]
                self.position_face_map[30] = first
                self.position_face_map[29] = second
                first = self.position_face_map[21]
                second = self.position_face_map[24]
                third = self.position_face_map[27]
                self.position_face_map[21] = self.position_face_map[46]
                self.position_face_map[24] = self.position_face_map[47]
                self.position_face_map[27] = self.position_face_map[48]
                self.position_face_map[46] = self.position_face_map[3]
                self.position_face_map[47] = self.position_face_map[6]
                self.position_face_map[48] = self.position_face_map[9]
                self.position_face_map[3] = self.position_face_map[45]
                self.position_face_map[6] = self.position_face_map[44]
                self.position_face_map[9] = self.position_face_map[43]
                self.position_face_map[45] = first
                self.position_face_map[44] = second
                self.position_face_map[43] = third

            if letter == 'U' or letter =='T':
                rotations = {28, 29, 30, 31, 32, 33, 34, 35, 36}
                translations = {27, 24, 21, 43, 44, 45}
                mix1 = {46, 47, 48}
                mix2 = {3, 6, 9}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, 3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([3, 0, 0]), begin_time=t0, transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([4.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([-6, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([-3, -6.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([3, 7, 0]), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[28]
                second = self.position_face_map[29]
                self.position_face_map[28] = self.position_face_map[30]
                self.position_face_map[29] = self.position_face_map[33]
                self.position_face_map[30] = self.position_face_map[36]
                self.position_face_map[33] = self.position_face_map[35]
                self.position_face_map[36] = self.position_face_map[34]
                self.position_face_map[35] = self.position_face_map[31]
                self.position_face_map[34] = first
                self.position_face_map[31] = second
                first = self.position_face_map[27]
                second = self.position_face_map[24]
                third = self.position_face_map[21]
                self.position_face_map[27] = self.position_face_map[43]
                self.position_face_map[24] = self.position_face_map[44]
                self.position_face_map[21] = self.position_face_map[45]
                self.position_face_map[43] = self.position_face_map[9]
                self.position_face_map[44] = self.position_face_map[6]
                self.position_face_map[45] = self.position_face_map[3]
                self.position_face_map[9] = self.position_face_map[48]
                self.position_face_map[6] = self.position_face_map[47]
                self.position_face_map[3] = self.position_face_map[46]
                self.position_face_map[48] = first
                self.position_face_map[47] = second
                self.position_face_map[46] = third

            if letter == 'l':
                rotations = {37, 38, 39, 40, 41, 42, 43, 44, 45}
                translations = {16,17,18,19,20,21,28,29,30}
                mix = {7,8,9}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([-3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0, transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], -2*pi,  Vector([-3.5, -6, 0]),begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i],Vector([0,9,0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[37]
                second = self.position_face_map[40]
                self.position_face_map[37] = self.position_face_map[43]
                self.position_face_map[40] = self.position_face_map[44]
                self.position_face_map[43] = self.position_face_map[45]
                self.position_face_map[44] = self.position_face_map[42]
                self.position_face_map[45] = self.position_face_map[39]
                self.position_face_map[42] = self.position_face_map[38]
                self.position_face_map[39] = first
                self.position_face_map[38]= second
                first = self.position_face_map[19]
                second = self.position_face_map[20]
                third = self.position_face_map[21]
                self.position_face_map[19] = self.position_face_map[28]
                self.position_face_map[20] = self.position_face_map[29]
                self.position_face_map[21] = self.position_face_map[30]
                self.position_face_map[28] = self.position_face_map[9]
                self.position_face_map[29] = self.position_face_map[8]
                self.position_face_map[30] = self.position_face_map[7]
                self.position_face_map[9] = self.position_face_map[18]
                self.position_face_map[8] = self.position_face_map[17]
                self.position_face_map[7] = self.position_face_map[16]
                self.position_face_map[18] = first
                self.position_face_map[17] = second
                self.position_face_map[16] = third

            if letter == 'L':
                rotations = {37, 38, 39, 40, 41, 42, 43, 44, 45}
                translations = {16, 17, 18, 19, 20, 21,7, 8, 9 }
                mix = {28, 29, 30}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([-3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], 2 * pi, Vector([-3.5, 3, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, -9, 0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[37]
                second = self.position_face_map[38]
                self.position_face_map[37] = self.position_face_map[39]
                self.position_face_map[38] = self.position_face_map[42]
                self.position_face_map[39] = self.position_face_map[45]
                self.position_face_map[42] = self.position_face_map[44]
                self.position_face_map[45] = self.position_face_map[43]
                self.position_face_map[44] = self.position_face_map[40]
                self.position_face_map[43] = first
                self.position_face_map[40] = second
                first = self.position_face_map[21]
                second = self.position_face_map[20]
                third = self.position_face_map[19]
                self.position_face_map[21] = self.position_face_map[16]
                self.position_face_map[20] = self.position_face_map[17]
                self.position_face_map[19] = self.position_face_map[18]
                self.position_face_map[16] = self.position_face_map[7]
                self.position_face_map[17] = self.position_face_map[8]
                self.position_face_map[18] = self.position_face_map[9]
                self.position_face_map[7] = self.position_face_map[30]
                self.position_face_map[8] = self.position_face_map[29]
                self.position_face_map[9] = self.position_face_map[28]
                self.position_face_map[30] = first
                self.position_face_map[29] = second
                self.position_face_map[28] = third

            if letter == 'r':
                rotations = {46, 47, 48, 49, 50, 51, 52, 53, 54}
                translations = {25,26,27,10,11,12,1,2,3}
                mix = {34,35,36}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0, transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], -2*pi,  Vector([3.5, 3, 0]),begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i],Vector([0,-9,0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[46]
                second = self.position_face_map[49]
                self.position_face_map[46] = self.position_face_map[52]
                self.position_face_map[49] = self.position_face_map[53]
                self.position_face_map[52] = self.position_face_map[54]
                self.position_face_map[53] = self.position_face_map[51]
                self.position_face_map[54] = self.position_face_map[48]
                self.position_face_map[51] = self.position_face_map[47]
                self.position_face_map[48] = first
                self.position_face_map[47]= second
                first = self.position_face_map[1]
                second = self.position_face_map[2]
                third = self.position_face_map[3]
                self.position_face_map[1] = self.position_face_map[36]
                self.position_face_map[2] = self.position_face_map[35]
                self.position_face_map[3] = self.position_face_map[34]
                self.position_face_map[36] = self.position_face_map[27]
                self.position_face_map[35] = self.position_face_map[26]
                self.position_face_map[34] = self.position_face_map[25]
                self.position_face_map[27] = self.position_face_map[10]
                self.position_face_map[26] = self.position_face_map[11]
                self.position_face_map[25] = self.position_face_map[12]
                self.position_face_map[10] = first
                self.position_face_map[11] = second
                self.position_face_map[12] = third

            if letter == 'R':
                rotations = {46, 47, 48, 49, 50, 51, 52, 53, 54}
                translations = {25,26,27,10,11,12,34,35,36}
                mix = {1,2,3}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0, transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], 2*pi,  Vector([3.5, -6, 0]),begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i],Vector([0,9,0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[46]
                second = self.position_face_map[47]
                self.position_face_map[46] = self.position_face_map[48]
                self.position_face_map[47] = self.position_face_map[51]
                self.position_face_map[48] = self.position_face_map[54]
                self.position_face_map[51] = self.position_face_map[53]
                self.position_face_map[54] = self.position_face_map[52]
                self.position_face_map[53] = self.position_face_map[49]
                self.position_face_map[52] = first
                self.position_face_map[49]= second
                first = self.position_face_map[3]
                second = self.position_face_map[2]
                third = self.position_face_map[1]
                self.position_face_map[3] = self.position_face_map[12]
                self.position_face_map[2] = self.position_face_map[11]
                self.position_face_map[1] = self.position_face_map[10]
                self.position_face_map[12] = self.position_face_map[25]
                self.position_face_map[11] = self.position_face_map[26]
                self.position_face_map[10] = self.position_face_map[27]
                self.position_face_map[25] = self.position_face_map[34]
                self.position_face_map[26] = self.position_face_map[35]
                self.position_face_map[27] = self.position_face_map[36]
                self.position_face_map[34] = first
                self.position_face_map[35] = second
                self.position_face_map[36] = third

            t0 = t0 + (dt+pause)

        self.layout_nodes()
        return t0

    def edge_parity_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with edge labels 0 and 1
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Edge", "EdgeSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=True, to_value=False, begin_time=begin_time)
        change_default_boolean(switches[6], from_value=False, to_value=True,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=False, to_value=True, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def corner_triality_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with corner labels 0,1 and 2
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Corner", "CornerSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=True, to_value=False, begin_time=begin_time)
        change_default_boolean(switches[6], from_value=False, to_value=True,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=False, to_value=True, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def edge_parity_off(self, begin_time, transition_time):
        """
        undo parity
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Edge", "EdgeSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=False, to_value=True, begin_time=begin_time+transition_time)
        change_default_boolean(switches[6], from_value=True, to_value=False,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=True, to_value=False, begin_time=begin_time )

        return begin_time + transition_time

    def corner_triality_off(self, begin_time, transition_time):
        """
        undo parity
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Corner", "CornerSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=False, to_value=True,
                                   begin_time=begin_time + transition_time)
        change_default_boolean(switches[6], from_value=True, to_value=False,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=True, to_value=False, begin_time=begin_time)

        return begin_time + transition_time

    def face_indices(self,show=True,begin_time=0):
        switch_labels = ["Face"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        if show:
            change_default_boolean(switches[0], from_value=False,to_value=True,begin_time=begin_time)
        else:
            change_default_boolean(switches[0], from_value=True,to_value=False,begin_time=begin_time)
        return begin_time

class GeoRubiksSphere(BObject):
    def __init__(self, **kwargs):
        """
        possible customizations
        range: [-1,1]
        orientation: "HORIZONTAL"|"VERTICAL"
        dimension: [1,0.25,0.25]
        location: [0,0,0]
        side_segments: 2
        shape: "cubic" | "cylinder"

        """
        cube = Cube()
        name = get_from_kwargs(kwargs,"name","RubiksSphereModifier")
        self.rc_geometry = RubiksSphereModifier(name=name, **kwargs)
        self.scale = get_from_kwargs(kwargs,"scale",1)
        self.cube_state = {}
        coords = tuples([0, 1, 2], 3)
        # somewhat awkward list that matches the position of the cubies when the origin of the coordinate system
        # is the (left,front,down) corner and the (2,2,2) corresponds to the (right,back,up) corner
        # the solved state is given by
        cubies = [0, 9, 18, 1, 10, 19, 2, 11, 20, 3, 12, 21, 4, 13, 22, 5, 14, 23, 6, 15, 24, 7, 16, 25, 8, 17, 26]
        for coord, cubie in zip(coords, cubies):
            self.cube_state[coord] = cubie

        self.transformation_maps = {
            "f": {(0, 0, 2): (0, 0, 0), (1, 0, 2): (0, 0, 1), (2, 0, 2): (0, 0, 2), (2, 0, 1): (1, 0, 2),
                  (2, 0, 0): (2, 0, 2), (1, 0, 0): (2, 0, 1), (0, 0, 0): (2, 0, 0), (0, 0, 1): (1, 0, 0),
                  (1, 0, 1): (1, 0, 1)},
            "F": {(0, 0, 2): (2, 0, 2), (0, 0, 1): (1, 0, 2), (0, 0, 0): (0, 0, 2), (1, 0, 0): (0, 0, 1),
                  (2, 0, 0): (0, 0, 0), (2, 0, 1): (1, 0, 0), (2, 0, 2): (2, 0, 0), (1, 0, 2): (2, 0, 1),
                  (1, 0, 1): (1, 0, 1)},
            "l": {(0, 2, 0): (0, 0, 0), (0, 2, 1): (0, 1, 0), (0, 2, 2): (0, 2, 0), (0, 1, 2): (0, 2, 1),
                  (0, 0, 2): (0, 2, 2), (0, 0, 1): (0, 1, 2), (0, 0, 0): (0, 0, 2), (0, 1, 0): (0, 0, 1),
                  (0, 1, 1): (0, 1, 1)},
            "L": {(0, 2, 0): (0, 2, 2), (0, 1, 0): (0, 2, 1), (0, 0, 0): (0, 2, 0), (0, 0, 1): (0, 1, 0),
                  (0, 0, 2): (0, 0, 0), (0, 1, 2): (0, 0, 1), (0, 2, 2): (0, 0, 2), (0, 2, 1): (0, 1, 2),
                  (0, 1, 1): (0, 1, 1)},
            "r": {(2, 0, 0): (2, 2, 0), (2, 0, 1): (2, 1, 0), (2, 0, 2): (2, 0, 0), (2, 1, 2): (2, 0, 1),
                  (2, 2, 2): (2, 0, 2), (2, 2, 1): (2, 1, 2), (2, 2, 0): (2, 2, 2), (2, 1, 0): (2, 2, 1),
                  (2, 1, 1): (2, 1, 1)},
            "R": {(2, 0, 0): (2, 0, 2), (2, 1, 0): (2, 0, 1), (2, 2, 0): (2, 0, 0), (2, 2, 1): (2, 1, 0),
                  (2, 2, 2): (2, 2, 0), (2, 1, 2): (2, 2, 1), (2, 0, 2): (2, 2, 2), (2, 0, 1): (2, 1, 2),
                  (2, 1, 1): (2, 1, 1)},
            "u": {(0, 0, 2): (2, 0, 2), (0, 1, 2): (1, 0, 2), (0, 2, 2): (0, 0, 2), (1, 2, 2): (0, 1, 2),
                  (2, 2, 2): (0, 2, 2), (2, 1, 2): (1, 2, 2), (2, 0, 2): (2, 2, 2), (1, 0, 2): (2, 1, 2),
                  (1, 1, 2): (1, 1, 2)},
            "U": {(0, 0, 2): (0, 2, 2), (1, 0, 2): (0, 1, 2), (2, 0, 2): (0, 0, 2), (2, 1, 2): (1, 0, 2),
                  (2, 2, 2): (2, 0, 2), (1, 2, 2): (2, 1, 2), (0, 2, 2): (2, 2, 2), (0, 1, 2): (1, 2, 2),
                  (1, 1, 2): (1, 1, 2)},
            "d": {(0, 2, 0): (2, 2, 0), (0, 1, 0): (1, 2, 0), (0, 0, 0): (0, 2, 0), (1, 0, 0): (0, 1, 0),
                  (2, 0, 0): (0, 0, 0), (2, 1, 0): (1, 0, 0), (2, 2, 0): (2, 0, 0), (1, 2, 0): (2, 1, 0),
                  (1, 1, 0): (1, 1, 0)},
            "D": {(0, 2, 0): (0, 0, 0), (1, 2, 0): (0, 1, 0), (2, 2, 0): (0, 2, 0), (2, 1, 0): (1, 2, 0),
                  (2, 0, 0): (2, 2, 0), (1, 0, 0): (2, 1, 0), (0, 0, 0): (2, 0, 0), (0, 1, 0): (1, 0, 0),
                  (1, 1, 0): (1, 1, 0)},
            "b": {(2, 2, 0): (0, 2, 0), (2, 2, 1): (1, 2, 0), (2, 2, 2): (2, 2, 0), (1, 2, 2): (2, 2, 1),
                  (0, 2, 2): (2, 2, 2), (0, 2, 1): (1, 2, 2), (0, 2, 0): (0, 2, 2), (1, 2, 0): (0, 2, 1),
                  (1, 2, 1): (1, 2, 1)},
            "B": {(2, 2, 0): (2, 2, 2), (1, 2, 0): (2, 2, 1), (0, 2, 0): (2, 2, 0), (0, 2, 1): (1, 2, 0),
                  (0, 2, 2): (0, 2, 0), (1, 2, 2): (0, 2, 1), (2, 2, 2): (0, 2, 2), (2, 2, 1): (1, 2, 2),
                  (1, 2, 1): (1, 2, 1)}, }

        # The cubies in the center of each face stay in place but will be rotated
        # The cubie in the center of the cube is fixed.
        # The rotation state of each cubie is stored as Euler angles

        self.cubie_rotation_states = {}
        for cubie_idx in range(27):
            self.cubie_rotation_states[cubie_idx] = Quaternion()

        # in the directory self.cube_state, the physical location of the cubies is stored
        # the following transformations act on the physical cube. The directory provides a lookup-table,
        # which cubies are actually affected by the transformation.
        # after the transformation the self.cube_state needs to be updated with the map.

        self.cubie_rotation_angle_map = {
            "f": [Quaternion(Vector([0, 1, 0]), pi / 2),
                  [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 2), (2, 0, 2), (2, 0, 1), (2, 0, 0), (1, 0, 0), (1, 0, 1)]],
            "F": [Quaternion(Vector([0, 1, 0]), -pi / 2),
                  [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 2), (2, 0, 2), (2, 0, 1), (2, 0, 0), (1, 0, 0), (1, 0, 1)]],
            "l": [Quaternion(Vector([1, 0, 0]), pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 1, 2), (0, 0, 2), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)]],
            "L": [Quaternion(Vector([1, 0, 0]), -pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 1, 2), (0, 0, 2), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)]],
            "r": [Quaternion(Vector([1, 0, 0]), -pi / 2),
                  [(2, 0, 0), (2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 1, 2), (2, 0, 2), (2, 0, 1), (2, 1, 1)]],
            "R": [Quaternion(Vector([1, 0, 0]), pi / 2),
                  [(2, 0, 0), (2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 1, 2), (2, 0, 2), (2, 0, 1), (2, 1, 1)]],
            "u": [Quaternion(Vector([0, 0, 1]), -pi / 2),
                  [(0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 1, 2), (2, 0, 2), (1, 0, 2), (1, 1, 2)]],
            "U": [Quaternion(Vector([0, 0, 1]), pi / 2),
                  [(0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 1, 2), (2, 0, 2), (1, 0, 2), (1, 1, 2)]],
            "d": [Quaternion(Vector([0, 0, 1]), pi / 2),
                  [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (2, 1, 0), (2, 0, 0), (1, 0, 0), (1, 1, 0)]],
            "D": [Quaternion(Vector([0, 0, 1]), -pi / 2),
                  [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (2, 1, 0), (2, 0, 0), (1, 0, 0), (1, 1, 0)]],
            "b": [Quaternion(Vector([0, 1, 0]), -pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 0), (1, 2, 0), (1, 2, 1)]],
            "B": [Quaternion(Vector([0, 1, 0]), pi / 2),
                  [(0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 0), (1, 2, 0), (1, 2, 1)]]
        }

        cube.add_mesh_modifier(type="NODES", node_modifier=self.rc_geometry)

        # get input quaternion nodes for the cubies
        self.cubie_rotation_nodes = [
            ibpy.get_geometry_node_from_modifier(self.rc_geometry, label="CubieRotation_" + str(i)) for i in range(27)]

        super().__init__(obj=cube.ref_obj, name="Rubik'sCube",scale=self.scale, **kwargs)

        initial_state = get_from_kwargs(kwargs,"initial_state","SOLVED")
        self.initial_word = ""
        if initial_state=="RANDOM":
            self.initial_word= random_word(20)
            self.initial_state(word=self.initial_word)

    def solve(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,linear=True):
        self.transform(self.initial_word.swapcase()[::-1],begin_time=begin_time,transition_time=transition_time)
        if linear:
            ibpy.set_linear_action_modifier(self)


    def initial_state(self,word):
        self.transform(word,begin_time=0,transition_time=0)

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        # size = get_geometry_node_from_modifier(self.rc_geometry, label="Radius")
        # change_default_value(size, from_value=1, to_value=1, begin_time=begin_time, transition_time=transition_time)
        ibpy.grow(self,scale=self.scale,begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)
        return begin_time + transition_time

    def transform(self, word,pause=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time / len(word)
        t0 = begin_time
        for letter in word:
            angle, positions = self.cubie_rotation_angle_map[letter]
            transformation = self.transformation_maps[letter]
            # transform relevant cubies
            active_cubies = [self.cube_state[position] for position in positions]
            print(word + ": " + letter + " active cubies for: ", [a + 1 for a in active_cubies])
            for idx in active_cubies:
                from_angle = self.cubie_rotation_states[idx]
                to_angle = angle @ from_angle
                self.cubie_rotation_states[idx] = to_angle
                ibpy.change_default_quaternion(self.cubie_rotation_nodes[idx], from_value=from_angle, to_value=to_angle,
                                               begin_time=t0, transition_time=dt)

            t0 += (dt+pause)
            # update physical state
            new_state = {}
            for src, replacement in transformation.items():
                new_state[src] = self.cube_state[replacement]

            for key, val in new_state.items():
                self.cube_state[key] = val

        return t0

    def change_emission_by_name(self,name_part="color",from_value=0,to_value=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        selected_materials=[]
        for mat in self.rc_geometry.materials:
            if name_part in mat.name:
                selected_materials.append(mat)
        for mat in selected_materials:
            ibpy.change_emission_of_material(mat,from_value=from_value,to_value=to_value,begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)

class RubiksCubeStabChain(BObject):
    def __init__(self, **kwargs):
        """
            Create a visual representation of the stabilizer chain for the
            Rubik's cube group

            The data of the stab chain is represented as geometry.
            The data contains:
            * e.g.: (45,2,"FrFd")
            * the first coordinate is the orbit image and the second coordinate is the level of the stabilizer chain
            * the string that represents the word of the orbit representative is added as an attribute
        """

        # load mesh data
        stabilizer_chain_dictionary={}
        with open(os.path.join(GLOBAL_DATA_DIR,"rg_stab_chain.dat")) as data:
            for line in data:
                parts = line.split(":")
                tup = parse_int_tuple(parts[0])
                # remove newline
                word = parts[1][:-1]
                tup = (tup[0],tup[1],1.5*tup[2])
                stabilizer_chain_dictionary[tup] = word

        vertices = [Vector(v) for v in stabilizer_chain_dictionary.keys()]
        mesh = create_mesh(vertices,name="StabChainMesh")
        raw_object = BObject(mesh=mesh,name="StabChainRawObject")
        self.max = len(vertices)

        name = get_from_kwargs(kwargs, "name", "RubiksCubeGroupStabChain")
        self.sc_geometry = RubiksCubeGroupStabChain(name=name, **kwargs)
        raw_object.add_mesh_modifier(type="NODES",node_modifier=self.sc_geometry)
        self.selected_index = -1

        super().__init__(obj=raw_object.ref_obj, name="Rubik'sCubeStabChain", **kwargs)

    def appear(self,alpha=1,limit = None,begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               clear_data=False, silent=False,linked=False, nice_alpha=False,children=True,**kwargs):

        super().appear(alpha=alpha, begin_time=begin_time,transition_time=transition_time,linked=linked,nice_alpha=nice_alpha,children=children,**kwargs)
        materials = self.sc_geometry.materials
        for mat in materials:
            ibpy.change_alpha_of_material(mat,from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        self.current_alpha=1

        if limit:
            range_max_node = ibpy.get_geometry_node_from_modifier(self.sc_geometry,label="DisplayRangeMax")
            range_min_node = ibpy.get_geometry_node_from_modifier(self.sc_geometry,label="DisplayRangeMin")
            ibpy.change_default_integer(range_max_node,from_value=-1,to_value=limit,begin_time=begin_time,transition_time=transition_time)
            ibpy.change_default_integer(range_min_node,from_value=-1,to_value=-1,begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def change_alpha(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        materials = self.sc_geometry.materials
        for mat in materials:
            ibpy.change_alpha_of_material(mat,from_value=self.current_alpha,to_value=alpha,begin_time=begin_time,transition_time=transition_time)
        self.current_alpha=alpha
        return begin_time + transition_time

    def change_min(self,from_value=-1,to_value=0,begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        range_node = ibpy.get_geometry_node_from_modifier(self.sc_geometry,label="DisplayRangeMin")
        ibpy.change_default_integer(range_node,from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def change_max(self, from_value=-1, to_value=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        range_node = ibpy.get_geometry_node_from_modifier(self.sc_geometry,label="DisplayRangeMax")
        ibpy.change_default_integer(range_node,from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def grow(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        super().appear(alpha=1,begin_time=begin_time,transition_time=transition_time,**kwargs)
        self.current_alpha=1
        self.change_min(from_value=-1,to_value=-1,begin_time=begin_time,transition_time=0,**kwargs)
        return self.change_max(from_value=-1, to_value=self.max, begin_time=begin_time, transition_time=transition_time)

    def open_element(self,index=-1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        """
        You have to find out the index of the element from the blender file
        """
        rot_angle_node = ibpy.get_geometry_node_from_modifier(self.sc_geometry,label="RotationAngle")
        index_node= ibpy.get_geometry_node_from_modifier(self.sc_geometry,label="ElementIndexSelector")
        change_default_integer(index_node,from_value=self.selected_index,to_value=index,begin_time=begin_time,transition_time=0)
        self.selected_index=index
        change_default_value(rot_angle_node,from_value=0,to_value=-pi/2,begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def close_element(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        rot_angle_node = ibpy.get_geometry_node_from_modifier(self.sc_geometry, label="RotationAngle")
        change_default_value(rot_angle_node, from_value=-pi/2, to_value=0, begin_time=begin_time,
                             transition_time=transition_time)
        return begin_time + transition_time



class GeoRubiksCube4x4Unfold(BObject):
    def __init__(self, name="Rubik'sCube4x4Unfold",**kwargs):
        """
        """
        cube = Cube()
        # take out location from the kwargs, since the argument collides with the node setup
        location = get_from_kwargs(kwargs, "location", [0, 0, 0])

        self.rc_geometry = RubiksCube4x4UnfoldedModifier(name="RubiksCube4x4UnfoldedModifier", **kwargs)
        cube.add_mesh_modifier(type="NODES", node_modifier=self.rc_geometry)

        self.position_face_map = {i: i for i in range(1, 55)}

        # get instances as the starting point for individual face transformations
        self.last_transform = {i: get_geometry_node_from_modifier(self.rc_geometry, label="InstanceOfFace" + str(i)) for
                               i in range(1, 55)}
        self.final_join = get_geometry_node_from_modifier(self.rc_geometry, label="JoinGeometry")
        super().__init__(obj=cube.ref_obj, name=name, location=location, **kwargs)

    def unfold(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        progressionNode = get_geometry_node_from_modifier(self.rc_geometry, label="UnfoldMeshNode")
        change_default_value(progressionNode.inputs["Progression"], from_value=1.3, to_value=23.9,
                             begin_time=begin_time, transition_time=transition_time)

        show_cubies = get_geometry_node_from_modifier(self.rc_geometry, label="ShowCubies")
        change_default_boolean(show_cubies, from_value=True, to_value=False, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        face_size = get_geometry_node_from_modifier(self.rc_geometry, label="FaceSize")
        change_default_value(face_size, from_value=0, to_value=0.890, begin_time=begin_time,
                             transition_time=transition_time)
        cube_size = get_geometry_node_from_modifier(self.rc_geometry, label="CubeSize")
        change_default_value(cube_size, from_value=0, to_value=0.95, begin_time=begin_time,
                             transition_time=transition_time)

        return begin_time + transition_time

    def rotate_face(self, index, angle, center, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        implement the rotation of individual faces as additional instance rotation nodes
        """
        tree = self.rc_geometry.tree
        links = tree.links

        # create geometry
        last_node = self.last_transform[index]
        final_join = self.final_join

        rotate_instance = RotateInstances(tree, rotation=Vector(), pivot_point=center, local_space=False)
        # cut old link
        for l in last_node.outputs[0].links:
            links.remove(l)
        links.new(last_node.outputs[0], rotate_instance.geometry_in)
        links.new(rotate_instance.geometry_out, final_join.inputs["Geometry"])
        self.last_transform[index] = rotate_instance.node
        # set keyframes
        change_default_rotation(rotate_instance.node.inputs["Rotation"], from_value=Euler(),
                                to_value=Euler([0, 0, angle]), begin_time=begin_time, transition_time=transition_time)

    def rotate_around_pivot(self,rotation_euler=Vector(),pivot=Vector(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        pivot_node = get_geometry_node_from_modifier(self.rc_geometry, label="FinalPivot")
        rotation_node=get_geometry_node_from_modifier(self.rc_geometry,label="FinalRotation")
        change_default_vector(pivot_node,from_value=Vector(),to_value=pivot,begin_time=begin_time,transition_time=transition_time)
        change_default_vector(rotation_node,from_value=Vector(),to_value=rotation_euler,begin_time=begin_time,transition_time=transition_time)

        return begin_time+transition_time

    def only_show_some_faces(self,selected =list(range(55)),begin_time=0):
        face_selector_node = get_geometry_node_from_modifier(self.rc_geometry, label="FaceSelectorSwitch")
        index_map = self.rc_geometry.index_to_face_index

        for i in range(55):
            if index_map[i] not in selected:
                ibpy.change_default_value(face_selector_node.inputs[str(i)],from_value=1,to_value=0,begin_time=begin_time,transition_time=0.1)

        return begin_time

    def only_show_some_labels(self,selected =list(range(55)),begin_time=0):
        face_selector_node = get_geometry_node_from_modifier(self.rc_geometry, label="FaceLabelSwitch")
        index_map = self.rc_geometry.index_to_face_index

        for i in range(55):
            if index_map[i] not in selected:
                ibpy.change_default_value(face_selector_node.inputs[str(i)],from_value=index_map[i],to_value=0,begin_time=begin_time,transition_time=0.1)

        return begin_time

    def hide_some_labels(self,selected=list(range(55)),begin_time=0):
        face_selector_node = get_geometry_node_from_modifier(self.rc_geometry, label="FaceLabelSwitch")
        index_map = self.rc_geometry.index_to_face_index

        for i in range(55):
            if index_map[i] in selected:
                ibpy.change_default_value(face_selector_node.inputs[str(i)],from_value=index_map[i],to_value=0,begin_time=begin_time,transition_time=0.1)

    def translate_face(self, index, translation, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        implement the rotation of individual faces as additional instance translation nodes
        """
        tree = self.rc_geometry.tree
        links = tree.links

        # create geometry
        last_node = self.last_transform[index]
        final_join = self.final_join

        translate_instance = TranslateInstances(tree, translation=translation, local_space=False)
        # cut old link
        for l in last_node.outputs[0].links:
            links.remove(l)
        links.new(last_node.outputs[0], translate_instance.geometry_in)
        links.new(translate_instance.geometry_out, final_join.inputs["Geometry"])
        self.last_transform[index] = translate_instance.node
        # set keyframes
        change_default_value(translate_instance.node.inputs["Translation"], from_value=Vector(),
                             to_value=translation, begin_time=begin_time, transition_time=transition_time)

    def layout_nodes(self):
        layout(self.rc_geometry.tree)

    def transform(self, word, pause=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        The transformations are performed through a simulation node.
        This means that all transformation parameters (angle, translations) have to be entered incrementally per frame
        and they have to be switched on and off in the same frame.
        """
        dt = transition_time / len(word)
        t0 = begin_time
        for letter in word:
            if letter == 'f':
                positions = {19, 20, 21, 22, 23, 24, 25, 26, 27, 10, 13, 16, 28, 31, 34, 37, 40, 43, 46, 49, 52}
                for i in positions:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector(), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[19]
                second = self.position_face_map[22]
                self.position_face_map[19] = self.position_face_map[25]
                self.position_face_map[22] = self.position_face_map[26]
                self.position_face_map[25] = self.position_face_map[27]
                self.position_face_map[26] = self.position_face_map[24]
                self.position_face_map[27] = self.position_face_map[21]
                self.position_face_map[24] = self.position_face_map[20]
                self.position_face_map[21] = first
                self.position_face_map[20] = second
                first = self.position_face_map[16]
                second = self.position_face_map[13]
                third = self.position_face_map[10]
                self.position_face_map[16] = self.position_face_map[52]
                self.position_face_map[13] = self.position_face_map[49]
                self.position_face_map[10] = self.position_face_map[46]
                self.position_face_map[52] = self.position_face_map[34]
                self.position_face_map[49] = self.position_face_map[31]
                self.position_face_map[46] = self.position_face_map[28]
                self.position_face_map[34] = self.position_face_map[43]
                self.position_face_map[31] = self.position_face_map[40]
                self.position_face_map[28] = self.position_face_map[37]
                self.position_face_map[43] = first
                self.position_face_map[40] = second
                self.position_face_map[37] = third

            if letter == 'F':
                faces = {19, 20, 21, 22, 23, 24, 25, 26, 27, 10, 13, 16, 28, 31, 34, 37, 40, 43, 46, 49, 52}
                for i in faces:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector(), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[19]
                second = self.position_face_map[20]
                self.position_face_map[19] = self.position_face_map[21]
                self.position_face_map[20] = self.position_face_map[24]
                self.position_face_map[21] = self.position_face_map[27]
                self.position_face_map[24] = self.position_face_map[26]
                self.position_face_map[27] = self.position_face_map[25]
                self.position_face_map[26] = self.position_face_map[22]
                self.position_face_map[25] = first
                self.position_face_map[22] = second
                first = self.position_face_map[10]
                second = self.position_face_map[13]
                third = self.position_face_map[16]
                self.position_face_map[10] = self.position_face_map[37]
                self.position_face_map[13] = self.position_face_map[40]
                self.position_face_map[16] = self.position_face_map[43]
                self.position_face_map[37] = self.position_face_map[28]
                self.position_face_map[40] = self.position_face_map[31]
                self.position_face_map[43] = self.position_face_map[34]
                self.position_face_map[28] = self.position_face_map[46]
                self.position_face_map[31] = self.position_face_map[49]
                self.position_face_map[34] = self.position_face_map[52]
                self.position_face_map[46] = first
                self.position_face_map[49] = second
                self.position_face_map[52] = third

            if letter == 'b':
                rotations1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
                rotations2 = {12, 15, 18, 30, 33, 36, 48, 51, 54, 39, 42, 45}
                for i in rotations1:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, -6, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in rotations2:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, 0, 0]), begin_time=t0,
                                     transition_time=dt)

                # update the new face positions
                first = self.position_face_map[1]
                second = self.position_face_map[4]
                self.position_face_map[1] = self.position_face_map[7]
                self.position_face_map[4] = self.position_face_map[8]
                self.position_face_map[7] = self.position_face_map[9]
                self.position_face_map[8] = self.position_face_map[6]
                self.position_face_map[9] = self.position_face_map[3]
                self.position_face_map[6] = self.position_face_map[2]
                self.position_face_map[3] = first
                self.position_face_map[2] = second
                first = self.position_face_map[12]
                second = self.position_face_map[15]
                third = self.position_face_map[18]
                self.position_face_map[12] = self.position_face_map[39]
                self.position_face_map[15] = self.position_face_map[42]
                self.position_face_map[18] = self.position_face_map[45]
                self.position_face_map[39] = self.position_face_map[30]
                self.position_face_map[42] = self.position_face_map[33]
                self.position_face_map[45] = self.position_face_map[36]
                self.position_face_map[30] = self.position_face_map[48]
                self.position_face_map[33] = self.position_face_map[51]
                self.position_face_map[36] = self.position_face_map[54]
                self.position_face_map[48] = first
                self.position_face_map[51] = second
                self.position_face_map[54] = third

            if letter == 'B':
                rotations1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
                rotations2 = {12, 15, 18, 30, 33, 36, 48, 51, 54, 39, 42, 45}
                for i in rotations1:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, -6, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in rotations2:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, 0, 0]), begin_time=t0,
                                     transition_time=dt)

                # update the new face positions
                first = self.position_face_map[1]
                second = self.position_face_map[2]
                self.position_face_map[1] = self.position_face_map[3]
                self.position_face_map[2] = self.position_face_map[6]
                self.position_face_map[3] = self.position_face_map[9]
                self.position_face_map[6] = self.position_face_map[8]
                self.position_face_map[9] = self.position_face_map[7]
                self.position_face_map[8] = self.position_face_map[4]
                self.position_face_map[7] = first
                self.position_face_map[4] = second
                first = self.position_face_map[18]
                second = self.position_face_map[15]
                third = self.position_face_map[12]
                self.position_face_map[18] = self.position_face_map[54]
                self.position_face_map[15] = self.position_face_map[51]
                self.position_face_map[12] = self.position_face_map[48]
                self.position_face_map[54] = self.position_face_map[36]
                self.position_face_map[51] = self.position_face_map[33]
                self.position_face_map[48] = self.position_face_map[30]
                self.position_face_map[36] = self.position_face_map[45]
                self.position_face_map[33] = self.position_face_map[42]
                self.position_face_map[30] = self.position_face_map[39]
                self.position_face_map[45] = first
                self.position_face_map[42] = second
                self.position_face_map[39] = third

            if letter == 'd':
                rotations = {10, 11, 12, 13, 14, 15, 16, 17, 18}
                translations = {39, 38, 37, 19, 22, 25}
                mix1 = {52, 53, 54}
                mix2 = {1, 4, 7}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, -3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([3, 0, 0]), begin_time=t0, transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([1.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([-1.5, -4.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[10]
                second = self.position_face_map[13]
                self.position_face_map[10] = self.position_face_map[16]
                self.position_face_map[13] = self.position_face_map[17]
                self.position_face_map[16] = self.position_face_map[18]
                self.position_face_map[17] = self.position_face_map[15]
                self.position_face_map[18] = self.position_face_map[12]
                self.position_face_map[15] = self.position_face_map[11]
                self.position_face_map[12] = first
                self.position_face_map[11] = second
                first = self.position_face_map[54]
                second = self.position_face_map[53]
                third = self.position_face_map[52]
                self.position_face_map[54] = self.position_face_map[25]
                self.position_face_map[53] = self.position_face_map[22]
                self.position_face_map[52] = self.position_face_map[19]
                self.position_face_map[25] = self.position_face_map[37]
                self.position_face_map[22] = self.position_face_map[38]
                self.position_face_map[19] = self.position_face_map[39]
                self.position_face_map[37] = self.position_face_map[7]
                self.position_face_map[38] = self.position_face_map[4]
                self.position_face_map[39] = self.position_face_map[1]
                self.position_face_map[7] = first
                self.position_face_map[4] = second
                self.position_face_map[1] = third

            if letter == 'D':
                rotations = {10, 11, 12, 13, 14, 15, 16, 17, 18}
                translations = {19, 22, 25, 52, 53, 54}
                mix1 = {39, 38, 37}
                mix2 = {1, 4, 7}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, -3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([-3, 0, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], pi, Vector([-1.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], pi, Vector([1.5, -4.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[10]
                second = self.position_face_map[11]
                self.position_face_map[10] = self.position_face_map[12]
                self.position_face_map[11] = self.position_face_map[15]
                self.position_face_map[12] = self.position_face_map[18]
                self.position_face_map[15] = self.position_face_map[17]
                self.position_face_map[18] = self.position_face_map[16]
                self.position_face_map[17] = self.position_face_map[13]
                self.position_face_map[16] = first
                self.position_face_map[13] = second
                first = self.position_face_map[52]
                second = self.position_face_map[53]
                third = self.position_face_map[54]
                self.position_face_map[52] = self.position_face_map[1]
                self.position_face_map[53] = self.position_face_map[4]
                self.position_face_map[54] = self.position_face_map[7]
                self.position_face_map[1] = self.position_face_map[39]
                self.position_face_map[4] = self.position_face_map[38]
                self.position_face_map[7] = self.position_face_map[37]
                self.position_face_map[39] = self.position_face_map[19]
                self.position_face_map[38] = self.position_face_map[22]
                self.position_face_map[37] = self.position_face_map[25]
                self.position_face_map[19] = first
                self.position_face_map[22] = second
                self.position_face_map[25] = third

            if letter == 'u':
                rotations = {28, 29, 30, 31, 32, 33, 34, 35, 36}
                translations = {27, 24, 21, 46, 47, 48}
                mix1 = {43, 44, 45}
                mix2 = {3, 6, 9}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([0, 3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([-3, 0, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], pi, Vector([-4.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([6, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], pi, Vector([3, -6.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([-3, 7, 0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[28]
                second = self.position_face_map[31]
                self.position_face_map[28] = self.position_face_map[34]
                self.position_face_map[31] = self.position_face_map[35]
                self.position_face_map[34] = self.position_face_map[36]
                self.position_face_map[35] = self.position_face_map[33]
                self.position_face_map[36] = self.position_face_map[30]
                self.position_face_map[33] = self.position_face_map[29]
                self.position_face_map[30] = first
                self.position_face_map[29] = second
                first = self.position_face_map[21]
                second = self.position_face_map[24]
                third = self.position_face_map[27]
                self.position_face_map[21] = self.position_face_map[46]
                self.position_face_map[24] = self.position_face_map[47]
                self.position_face_map[27] = self.position_face_map[48]
                self.position_face_map[46] = self.position_face_map[3]
                self.position_face_map[47] = self.position_face_map[6]
                self.position_face_map[48] = self.position_face_map[9]
                self.position_face_map[3] = self.position_face_map[45]
                self.position_face_map[6] = self.position_face_map[44]
                self.position_face_map[9] = self.position_face_map[43]
                self.position_face_map[45] = first
                self.position_face_map[44] = second
                self.position_face_map[43] = third

            if letter == 'U':
                rotations = {28, 29, 30, 31, 32, 33, 34, 35, 36}
                translations = {27, 24, 21, 43, 44, 45}
                mix1 = {46, 47, 48}
                mix2 = {3, 6, 9}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([0, 3, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([3, 0, 0]), begin_time=t0, transition_time=dt)
                for i in mix1:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([4.5, -1.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([-6, -3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix2:
                    self.rotate_face(self.position_face_map[i], -pi, Vector([-3, -6.5, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([3, 7, 0]), begin_time=t0, transition_time=dt)

                # update the new face positions
                first = self.position_face_map[28]
                second = self.position_face_map[29]
                self.position_face_map[28] = self.position_face_map[30]
                self.position_face_map[29] = self.position_face_map[33]
                self.position_face_map[30] = self.position_face_map[36]
                self.position_face_map[33] = self.position_face_map[35]
                self.position_face_map[36] = self.position_face_map[34]
                self.position_face_map[35] = self.position_face_map[31]
                self.position_face_map[34] = first
                self.position_face_map[31] = second
                first = self.position_face_map[27]
                second = self.position_face_map[24]
                third = self.position_face_map[21]
                self.position_face_map[27] = self.position_face_map[43]
                self.position_face_map[24] = self.position_face_map[44]
                self.position_face_map[21] = self.position_face_map[45]
                self.position_face_map[43] = self.position_face_map[9]
                self.position_face_map[44] = self.position_face_map[6]
                self.position_face_map[45] = self.position_face_map[3]
                self.position_face_map[9] = self.position_face_map[48]
                self.position_face_map[6] = self.position_face_map[47]
                self.position_face_map[3] = self.position_face_map[46]
                self.position_face_map[48] = first
                self.position_face_map[47] = second
                self.position_face_map[46] = third

            if letter == 'l':
                rotations = {37, 38, 39, 40, 41, 42, 43, 44, 45}
                translations = {16,17,18,19,20,21,28,29,30}
                mix = {7,8,9}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([-3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0, transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], -2*pi,  Vector([-3.5, -6, 0]),begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i],Vector([0,9,0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[37]
                second = self.position_face_map[40]
                self.position_face_map[37] = self.position_face_map[43]
                self.position_face_map[40] = self.position_face_map[44]
                self.position_face_map[43] = self.position_face_map[45]
                self.position_face_map[44] = self.position_face_map[42]
                self.position_face_map[45] = self.position_face_map[39]
                self.position_face_map[42] = self.position_face_map[38]
                self.position_face_map[39] = first
                self.position_face_map[38]= second
                first = self.position_face_map[19]
                second = self.position_face_map[20]
                third = self.position_face_map[21]
                self.position_face_map[19] = self.position_face_map[28]
                self.position_face_map[20] = self.position_face_map[29]
                self.position_face_map[21] = self.position_face_map[30]
                self.position_face_map[28] = self.position_face_map[9]
                self.position_face_map[29] = self.position_face_map[8]
                self.position_face_map[30] = self.position_face_map[7]
                self.position_face_map[9] = self.position_face_map[18]
                self.position_face_map[8] = self.position_face_map[17]
                self.position_face_map[7] = self.position_face_map[16]
                self.position_face_map[18] = first
                self.position_face_map[17] = second
                self.position_face_map[16] = third

            if letter == 'L':
                rotations = {37, 38, 39, 40, 41, 42, 43, 44, 45}
                translations = {16, 17, 18, 19, 20, 21,7, 8, 9 }
                mix = {28, 29, 30}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([-3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0,
                                        transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], 2 * pi, Vector([-3.5, 3, 0]), begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i], Vector([0, -9, 0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[37]
                second = self.position_face_map[38]
                self.position_face_map[37] = self.position_face_map[39]
                self.position_face_map[38] = self.position_face_map[42]
                self.position_face_map[39] = self.position_face_map[45]
                self.position_face_map[42] = self.position_face_map[44]
                self.position_face_map[45] = self.position_face_map[43]
                self.position_face_map[44] = self.position_face_map[40]
                self.position_face_map[43] = first
                self.position_face_map[40] = second
                first = self.position_face_map[21]
                second = self.position_face_map[20]
                third = self.position_face_map[19]
                self.position_face_map[21] = self.position_face_map[16]
                self.position_face_map[20] = self.position_face_map[17]
                self.position_face_map[19] = self.position_face_map[18]
                self.position_face_map[16] = self.position_face_map[7]
                self.position_face_map[17] = self.position_face_map[8]
                self.position_face_map[18] = self.position_face_map[9]
                self.position_face_map[7] = self.position_face_map[30]
                self.position_face_map[8] = self.position_face_map[29]
                self.position_face_map[9] = self.position_face_map[28]
                self.position_face_map[30] = first
                self.position_face_map[29] = second
                self.position_face_map[28] = third

            if letter == 'r':
                rotations = {46, 47, 48, 49, 50, 51, 52, 53, 54}
                translations = {25,26,27,10,11,12,1,2,3}
                mix = {34,35,36}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], -pi / 2, Vector([3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, 3, 0]), begin_time=t0, transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], -2*pi,  Vector([3.5, 3, 0]),begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i],Vector([0,-9,0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[46]
                second = self.position_face_map[49]
                self.position_face_map[46] = self.position_face_map[52]
                self.position_face_map[49] = self.position_face_map[53]
                self.position_face_map[52] = self.position_face_map[54]
                self.position_face_map[53] = self.position_face_map[51]
                self.position_face_map[54] = self.position_face_map[48]
                self.position_face_map[51] = self.position_face_map[47]
                self.position_face_map[48] = first
                self.position_face_map[47]= second
                first = self.position_face_map[1]
                second = self.position_face_map[2]
                third = self.position_face_map[3]
                self.position_face_map[1] = self.position_face_map[36]
                self.position_face_map[2] = self.position_face_map[35]
                self.position_face_map[3] = self.position_face_map[34]
                self.position_face_map[36] = self.position_face_map[27]
                self.position_face_map[35] = self.position_face_map[26]
                self.position_face_map[34] = self.position_face_map[25]
                self.position_face_map[27] = self.position_face_map[10]
                self.position_face_map[26] = self.position_face_map[11]
                self.position_face_map[25] = self.position_face_map[12]
                self.position_face_map[10] = first
                self.position_face_map[11] = second
                self.position_face_map[12] = third

            if letter == 'R':
                rotations = {46, 47, 48, 49, 50, 51, 52, 53, 54}
                translations = {25,26,27,10,11,12,34,35,36}
                mix = {1,2,3}
                for i in rotations:
                    self.rotate_face(self.position_face_map[i], pi / 2, Vector([3, 0, 0]), begin_time=t0,
                                     transition_time=dt)
                for i in translations:
                    self.translate_face(self.position_face_map[i], Vector([0, -3, 0]), begin_time=t0, transition_time=dt)
                for i in mix:
                    self.rotate_face(self.position_face_map[i], 2*pi,  Vector([3.5, -6, 0]),begin_time=t0,
                                     transition_time=dt)
                    self.translate_face(self.position_face_map[i],Vector([0,9,0]), begin_time=t0,
                                        transition_time=dt)

                # update the new face positions
                first = self.position_face_map[46]
                second = self.position_face_map[47]
                self.position_face_map[46] = self.position_face_map[48]
                self.position_face_map[47] = self.position_face_map[51]
                self.position_face_map[48] = self.position_face_map[54]
                self.position_face_map[51] = self.position_face_map[53]
                self.position_face_map[54] = self.position_face_map[52]
                self.position_face_map[53] = self.position_face_map[49]
                self.position_face_map[52] = first
                self.position_face_map[49]= second
                first = self.position_face_map[3]
                second = self.position_face_map[2]
                third = self.position_face_map[1]
                self.position_face_map[3] = self.position_face_map[12]
                self.position_face_map[2] = self.position_face_map[11]
                self.position_face_map[1] = self.position_face_map[10]
                self.position_face_map[12] = self.position_face_map[25]
                self.position_face_map[11] = self.position_face_map[26]
                self.position_face_map[10] = self.position_face_map[27]
                self.position_face_map[25] = self.position_face_map[34]
                self.position_face_map[26] = self.position_face_map[35]
                self.position_face_map[27] = self.position_face_map[36]
                self.position_face_map[34] = first
                self.position_face_map[35] = second
                self.position_face_map[36] = third

            t0 = t0 + (dt+pause)

        self.layout_nodes()
        return t0

    def edge_parity_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with edge labels 0 and 1
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Edge", "EdgeSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=True, to_value=False, begin_time=begin_time)
        change_default_boolean(switches[6], from_value=False, to_value=True,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=False, to_value=True, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def corner_triality_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with corner labels 0,1 and 2
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Corner", "CornerSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=True, to_value=False, begin_time=begin_time)
        change_default_boolean(switches[6], from_value=False, to_value=True,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=False, to_value=True, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def edge_parity_off(self, begin_time, transition_time):
        """
        undo parity
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Edge", "EdgeSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=False, to_value=True, begin_time=begin_time+transition_time)
        change_default_boolean(switches[6], from_value=True, to_value=False,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=True, to_value=False, begin_time=begin_time )

        return begin_time + transition_time

    def corner_triality_off(self, begin_time, transition_time):
        """
        undo parity
        """

        switch_labels = ["RedSelector", "GreenSelector", "BlueSelector", "YellowSelector", "WhiteSelector",
                         "OrangeSelector", "Corner", "CornerSelector"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        for i in range(6):
            change_default_boolean(switches[i], from_value=False, to_value=True,
                                   begin_time=begin_time + transition_time)
        change_default_boolean(switches[6], from_value=True, to_value=False,
                               begin_time=begin_time + transition_time / 2)
        change_default_boolean(switches[7], from_value=True, to_value=False, begin_time=begin_time)

        return begin_time + transition_time

    def face_indices(self,begin_time=0):
        switch_labels = ["Face"]
        switches = [get_geometry_node_from_modifier(self.rc_geometry, label=label + "Switch") for label in
                    switch_labels]

        change_default_boolean(switches[0], from_value=False,to_value=True,begin_time=begin_time)
        return begin_time

class GeoMegaMinx(BObject):
    def __init__(self, name="Rubik'sMegaMix",**kwargs):
        """
        """
        dodecahedron = Dodecahedron()
        face = SubdividedPentagon(middle_squeeze=0.70)

        # take out location from the kwargs, since the argument collides with the node setup
        location = get_from_kwargs(kwargs, "location", [0, 0, 0])

        self.megaminx_geometry = MegaMinxModifier(name="RubiksMegaMinxModifier", face=face, **kwargs)
        dodecahedron.add_mesh_modifier(type="NODES", node_modifier=self.megaminx_geometry)

        self.position_face_map = {i: i for i in range(1, 55)}

        super().__init__(obj=dodecahedron.ref_obj, name=name, location=location, **kwargs)

    def unfold(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        progressionNode = get_geometry_node_from_modifier(self.megaminx_geometry, label="UnfoldMeshNode")
        change_default_value(progressionNode.inputs["Progression"], from_value=1.3, to_value=23.9,
                             begin_time=begin_time, transition_time=transition_time)

        return begin_time + transition_time

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        super().appear(begin_time=begin_time, transition_time=transition_time)
        # face_size = get_geometry_node_from_modifier(self.rc_geometry, label="FaceSize")
        # change_default_value(face_size, from_value=0, to_value=0.890, begin_time=begin_time,
        #                      transition_time=transition_time)
        # cube_size = get_geometry_node_from_modifier(self.rc_geometry, label="CubeSize")
        # change_default_value(cube_size, from_value=0, to_value=0.95, begin_time=begin_time,
        #                      transition_time=transition_time)

        return begin_time + transition_time

class SingmasterDisplay(BObject):
    def __init__(self,word="FRDLB",rows =1, columns=5,spacing=5,colors=["text","joker","important"], **kwargs):
        """
            prepare a display of representation
            The user can specify the displayed word
            the number of rows
            and the number of columns
        """

        # load mesh data
        cube = Cube()
        self.sm_modifier=SingmasterDisplayModifier(word=word,rows=rows,columns=columns,spacing=spacing,colors=colors,**kwargs)
        cube.add_mesh_modifier(type="NODES",node_modifier=self.sm_modifier)
        name = get_from_kwargs(kwargs, "name", "Singmaster_"+word)

        row_node =get_geometry_node_from_modifier(self.sm_modifier,label="Rows")
        column_node =get_geometry_node_from_modifier(self.sm_modifier,label="Columns")
        spacing_node=get_geometry_node_from_modifier(self.sm_modifier,label="Spacing")
        letter_node=get_geometry_node_from_modifier(self.sm_modifier,label="NumberOfInstructions")
        word_node=get_geometry_node_from_modifier(self.sm_modifier,label="Word")

        ibpy.change_default_integer(row_node,from_value=0,to_value=rows,begin_time=0,transition_time=0)
        ibpy.change_default_integer(column_node,from_value=0,to_value=columns,begin_time=0,transition_time=0)
        ibpy.change_default_value(spacing_node,from_value=0,to_value=spacing,begin_time=0,transition_time=0)
        ibpy.change_default_integer(letter_node,from_value=0,to_value=len(word),begin_time=0,transition_time=0)

        letter2index={
            'r':3,
            'R':0,
            'l':1,
            'L':2,
            't':5,
            'T':4,
            'u':5,
            'U':4,
            'd':6,
            'D':7,
            'f':9,
            'F':8,
        }

        for i,l in enumerate(word):
            if len(word_node.inputs)>i+1:
                word_node.inputs[i+1].default_value=letter2index[l]
            else:
                word_node.add_item(socket=letter2index[l])

        grid_mat_input=ibpy.get_geometry_node_from_modifier(self.sm_modifier, label="GridMaterial")
        clockwise_mat_input=ibpy.get_geometry_node_from_modifier(self.sm_modifier, label="ClockwiseMaterial")
        reflected_clockwise_mat_input=ibpy.get_geometry_node_from_modifier(self.sm_modifier, label="ReflectedClockwiseMaterial")
        anticlockwise_mat_input = ibpy.get_geometry_node_from_modifier(self.sm_modifier, label="AnticlockwiseMaterial")
        reflected_anticlockwise_mat_input = ibpy.get_geometry_node_from_modifier(self.sm_modifier, label="ReflectedAnticlockwiseMaterial")

        grid_material = get_material(colors[0], **kwargs)
        ibpy.change_default_material(grid_mat_input, to_value=grid_material, begin_time=0)
        clockwise_material = get_material(colors[1], **kwargs)
        ibpy.change_default_material(clockwise_mat_input, to_value=clockwise_material, begin_time=0)
        ibpy.change_default_material(reflected_anticlockwise_mat_input, to_value=clockwise_material, begin_time=0)
        anticlockwise_material = get_material(colors[2], **kwargs)
        self.sm_modifier.materials+=[grid_material,clockwise_material,anticlockwise_material]
        ibpy.change_default_material(anticlockwise_mat_input, to_value=anticlockwise_material, begin_time=0)
        ibpy.change_default_material(reflected_clockwise_mat_input, to_value=anticlockwise_material, begin_time=0)

        super().__init__(obj=cube.ref_obj, name=name, **kwargs)

    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               clear_data=False, silent=False,linked=False, nice_alpha=False,children=True,**kwargs):

        materials = self.sm_modifier.materials
        for mat in materials:
            ibpy.change_alpha_of_material(mat, from_value=0, to_value=1, begin_time=begin_time,
                                          transition_time=transition_time)

        self.current_alpha = 1
        super().appear(alpha=alpha, begin_time=begin_time, transition_time=transition_time,**kwargs)
        return begin_time+transition_time

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               clear_data=False, silent=False,linked=False, nice_alpha=False,**kwargs):

        materials = self.sm_modifier.materials
        for mat in materials:
            ibpy.change_alpha_of_material(mat, from_value=1, to_value=0, begin_time=begin_time,
                                          transition_time=transition_time)

        return super().disappear( begin_time=begin_time, transition_time=transition_time,**kwargs)

    def reflect(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        reflect_node = get_geometry_node_from_modifier(self.sm_modifier, label="ReflectorValue")
        reflect_switch = get_geometry_node_from_modifier(self.sm_modifier, label="BooleanReflected")

        ibpy.change_default_value(reflect_node, from_value=0, to_value=pi, begin_time=begin_time,transition_time=transition_time)
        ibpy.change_default_boolean(reflect_switch,from_value=False, to_value=True,begin_time=begin_time+transition_time/2)
        return begin_time+transition_time

    def reflect_back(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        reflect_node = get_geometry_node_from_modifier(self.sm_modifier, label="ReflectorValue")
        reflect_switch = get_geometry_node_from_modifier(self.sm_modifier, label="BooleanReflected")

        ibpy.change_default_value(reflect_node, from_value=pi, to_value=0, begin_time=begin_time,transition_time=transition_time)
        ibpy.change_default_boolean(reflect_switch,from_value=True, to_value=False,begin_time=begin_time+transition_time/2)
        return begin_time+transition_time
