from math import pi

from mathutils import Vector, Quaternion, Euler

from geometry_nodes.geometry_nodes_modifier import RubiksCubeModifier, RubiksCubeUnfolded
from geometry_nodes.nodes import RotateInstances, layout, TranslateInstances, RealizeInstances, GeometryToInstance
from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier, change_default_boolean, change_default_value, \
    change_default_quaternion, change_default_vector, change_default_rotation
from mathematics.mathematica.mathematica import tuples
from objects.bobject import BObject
from objects.cube import Cube
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.kwargs import get_from_kwargs


class GeoRubiksCube(BObject):
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
        self.rc_geometry = RubiksCubeModifier(name="RubiksCubeModifier", **kwargs)

        self.cube_state = {}
        coords = tuples([0, 1, 2], 3)
        # somewhat awkward list that matches the position of the cubies when the origin of the coordinate system
        # is the (left,front,down) corner and the (2,2,2) corresponds to the (right,back,up) corner
        # the solved state is given by
        cubies = [0, 9, 18, 1, 10, 19, 2, 11, 20, 3, 12, 21, 4, 13, 22, 5, 14, 23, 6, 15, 24, 7, 16, 25, 8, 17, 26]
        for coord, cubie in zip(coords, cubies):
            self.cube_state[coord] = cubie
            print(coord, "->", cubie)

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

        super().__init__(obj=cube.ref_obj, name="Rubik'sCube", **kwargs)

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        size = get_geometry_node_from_modifier(self.rc_geometry, label="Cube Size")
        change_default_value(size, from_value=0, to_value=0.890, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def transform(self, word, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
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

            t0 += dt
            # update physical state
            new_state = {}
            for src, replacement in transformation.items():
                new_state[src] = self.cube_state[replacement]

            for key, val in new_state.items():
                self.cube_state[key] = val

        return t0

    def edge_parity_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with edge labels -1 and 1
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


class GeoRubiksCubeUnfold(BObject):
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
        # take out location from the kwargs, since the argument collides with the node setup
        location = get_from_kwargs(kwargs, "location", [0, 0, 0])

        self.rc_geometry = RubiksCubeUnfolded(name="RubiksCubeUnfoldedModifier", **kwargs)
        cube.add_mesh_modifier(type="NODES", node_modifier=self.rc_geometry)

        self.position_face_map = {i: i for i in range(1, 55)}

        # get instances as the starting point for individual face transformations
        self.last_transform = {i: get_geometry_node_from_modifier(self.rc_geometry, label="InstanceOfFace" + str(i)) for
                               i in range(1, 55)}
        self.final_join = get_geometry_node_from_modifier(self.rc_geometry, label="JoinGeometry")

        super().__init__(obj=cube.ref_obj, name="Rubik'sCubeFoldable", location=location, **kwargs)

    def unfold(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        progressionNode = get_geometry_node_from_modifier(self.rc_geometry, label="UnfoldMeshNode")
        change_default_value(progressionNode.inputs["Progression"], from_value=1.3, to_value=23.9,
                             begin_time=begin_time, transition_time=transition_time)

        show_cubies = get_geometry_node_from_modifier(self.rc_geometry, label="ShowCubies")
        change_default_boolean(show_cubies, from_value=True, to_value=False, begin_time=begin_time + transition_time)

        return begin_time + transition_time

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        face_size = get_geometry_node_from_modifier(self.rc_geometry, label="FaceSize")
        change_default_value(face_size, from_value=0, to_value=0.890, begin_time=begin_time,
                             transition_time=transition_time)
        cube_size = get_geometry_node_from_modifier(self.rc_geometry, label="CubeSize")
        change_default_value(cube_size, from_value=0, to_value=0.95, begin_time=begin_time,
                             transition_time=transition_time)

        return begin_time + transition_time

    def rotate_face(self, index, angle, center, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        tree = self.rc_geometry.tree
        links = tree.links

        # create geometry
        last_node = self.last_transform[index]
        final_join = self.final_join

        rotate_instance = RotateInstances(tree, rotation=Vector(), pivot_point=center, local_space=False)
        realize_instance = RealizeInstances(tree)
        geo_to_instance = GeometryToInstance(tree)
        # cut old link
        for l in last_node.outputs[0].links:
            links.remove(l)
        links.new(last_node.outputs[0], rotate_instance.geometry_in)
        # links.new(rotate_instance.geometry_out, realize_instance.geometry_in)
        # links.new(realize_instance.geometry_out, geo_to_instance.geometry_in)
        # links.new(geo_to_instance.geometry_out, final_join.inputs["Geometry"])
        # self.last_transform[index] = geo_to_instance.node
        links.new(rotate_instance.geometry_out, final_join.inputs["Geometry"])
        self.last_transform[index] = rotate_instance.node
        # set keyframes
        change_default_rotation(rotate_instance.node.inputs["Rotation"], from_value=Euler(),
                                to_value=Euler([0, 0, angle]), begin_time=begin_time, transition_time=transition_time)

    def translate_face(self, index, translation, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        tree = self.rc_geometry.tree
        links = tree.links

        # create geometry
        last_node = self.last_transform[index]
        final_join = self.final_join

        translate_instance = TranslateInstances(tree, translation=translation, local_space=False)
        realize_instance = RealizeInstances(tree)
        geo_to_instance = GeometryToInstance(tree)
        # cut old link
        for l in last_node.outputs[0].links:
            links.remove(l)
        links.new(last_node.outputs[0], translate_instance.geometry_in)
        # links.new(translate_instance.geometry_out,realize_instance.geometry_in)
        # links.new(realize_instance.geometry_out,geo_to_instance.geometry_in)
        # links.new(geo_to_instance.geometry_out, final_join.inputs["Geometry"])
        # self.last_transform[index] = geo_to_instance.node
        links.new(translate_instance.geometry_out, final_join.inputs["Geometry"])
        self.last_transform[index] = translate_instance.node
        # set keyframes
        change_default_value(translate_instance.node.inputs["Translation"], from_value=Vector(),
                             to_value=translation, begin_time=begin_time, transition_time=transition_time)

    def layout_nodes(self):
        layout(self.rc_geometry.tree)

    def transform(self, word, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        The transformations are performed through a simulation node.
        This means that all transformation parameters (angle, translations) have to be entered incrementally per frame
        and they have to be switched on and off in the same frame.
        """
        dt = transition_time / len(word)
        t0 = begin_time
        for letter in word:
            if letter == 'u':
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

            if letter == 'U':
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

            if letter == 'd':
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

            if letter == 'D':
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

            if letter == 'f':
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

            if letter == 'F':
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

            if letter == 'b':
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

            if letter == 'B':
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

            t0 = t0 + dt

        self.layout_nodes()
        return begin_time + transition_time

    def edge_parity_transition(self, begin_time, transition_time):
        """
        here the texture of the edge faces is changed into a two-color mode with edge labels -1 and 1
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
