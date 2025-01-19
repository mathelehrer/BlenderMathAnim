from math import pi

from mathutils import Vector, Quaternion

from geometry_nodes.geometry_nodes_modifier import RubiksCubeModifier
from interface import ibpy
from mathematics.mathematica.mathematica import tuples
from objects.bobject import BObject
from objects.cube import Cube
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class GeoRubiksCube(BObject):
    def __init__(self,**kwargs):
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
        rc_geometry = RubiksCubeModifier(name="RubiksCubeModifier",**kwargs)

        self.cube_state = {}
        coords = tuples([0,1,2],3)
        # somewhat awkward list that matches the position of the cubies when the origin of the coordinate system
        # is the (left,front,down) corner and the (2,2,2) corresponds to the (right,back,up) corner
        # the solved state is given by
        cubies = [0,9,18,1,10,19,2,11,20,3,12,21,4,13,22,5,14,23,6,15,24,7,16,25,8,17,26]
        for coord,cubie in zip(coords,cubies):
            self.cube_state[coord] = cubie
            print(coord,"->",cubie)


        self.transformation_maps={"f":{(0, 0, 2): (0,0,0),(1, 0, 2):(0, 0, 1),(2, 0, 2):(0, 0, 2),(2, 0, 1):(1, 0, 2),(2, 0, 0):(2, 0, 2),(1, 0, 0):(2, 0, 1),(0,0,0):(2, 0, 0),(0, 0, 1):(1, 0, 0),(1, 0, 1):(1, 0, 1)},
                         "F":{(0, 0, 2):(2, 0, 2),(0, 0, 1):(1, 0, 2),(0,0,0):(0, 0, 2),(1, 0, 0):(0, 0, 1),(2, 0, 0): (0,0,0),(2, 0, 1):(1, 0, 0),(2, 0, 2):(2, 0, 0),(1, 0, 2):(2, 0, 1),(1, 0, 1):(1, 0, 1)},
                         "l":{(0, 2, 0): (0,0,0),(0, 2, 1): (0, 1, 0),(0, 2, 2): (0, 2, 0),(0, 1, 2):(0, 2, 1),(0, 0, 2):(0, 2, 2),(0, 0, 1):(0, 1, 2),(0,0,0):(0, 0, 2),(0, 1, 0):(0, 0, 1),(0, 1, 1):(0, 1, 1)},
                         "L":{(0, 2, 0):(0, 2, 2),(0, 1, 0):(0, 2, 1),(0,0,0): (0, 2, 0),(0, 0, 1): (0, 1, 0),(0, 0, 2): (0,0,0),(0, 1, 2):(0, 0, 1),(0, 2, 2):(0, 0, 2),(0, 2, 1):(0, 1, 2),(0, 1, 1):(0, 1, 1)},
                         "r":{(2, 0, 0):(2, 2, 0),(2, 0, 1):(2, 1, 0),(2, 0, 2):(2, 0, 0),(2, 1, 2):(2, 0, 1),(2, 2, 2):(2, 0, 2),(2, 2, 1):(2, 1, 2),(2, 2, 0):(2, 2, 2),(2, 1, 0):(2, 2, 1),(2, 1, 1):(2, 1, 1)},
                         "R":{(2, 0, 0):(2, 0, 2),(2, 1, 0):(2, 0, 1),(2, 2, 0):(2, 0, 0),(2, 2, 1):(2, 1, 0),(2, 2, 2):(2, 2, 0),(2, 1, 2):(2, 2, 1),(2, 0, 2):(2, 2, 2),(2, 0, 1):(2, 1, 2),(2, 1, 1):(2, 1, 1)},
                         "t":{(0, 0, 2):(2, 0, 2),(0, 1, 2):(1, 0, 2),(0, 2, 2):(0, 0, 2),(1, 2, 2):(0, 1, 2),(2, 2, 2):(0, 2, 2),(2, 1, 2):(1, 2, 2),(2, 0, 2):(2, 2, 2),(1, 0, 2):(2, 1, 2),(1, 1, 2):(1, 1, 2)},
                         "T":{(0, 0, 2):(0, 2, 2),(1, 0, 2):(0, 1, 2),(2, 0, 2):(0, 0, 2),(2, 1, 2):(1, 0, 2),(2, 2, 2):(2, 0, 2),(1, 2, 2):(2, 1, 2),(0, 2, 2):(2, 2, 2),(0, 1, 2):(1, 2, 2),(1, 1, 2):(1, 1, 2)},
                         "d":{(0, 2, 0):(2, 2, 0),(0, 1, 0):(1, 2, 0),(0,0,0): (0, 2, 0),(1, 0, 0): (0, 1, 0),(2, 0, 0): (0,0,0),(2, 1, 0):(1, 0, 0),(2, 2, 0):(2, 0, 0),(1, 2, 0):(2, 1, 0),(1,1,0):(1,1,0)},
                         "D":{(0, 2, 0): (0,0,0),(1, 2, 0): (0, 1, 0),(2, 2, 0): (0, 2, 0),(2, 1, 0):(1, 2, 0),(2, 0, 0):(2, 2, 0),(1, 0, 0):(2, 1, 0),(0,0,0):(2, 0, 0),(0, 1, 0):(1, 0, 0),(1,1,0):(1,1,0)},
                                  "b":{(2, 2, 0): (0, 2, 0),(2, 2, 1):(1, 2, 0),(2, 2, 2):(2, 2, 0),(1, 2, 2):(2, 2, 1),(0, 2, 2):(2, 2, 2),(0, 2, 1):(1, 2, 2),(0, 2, 0):(0, 2, 2),(1, 2, 0):(0, 2, 1),(1, 2, 1):(1, 2, 1)},
                                  "B":{(2, 2, 0):(2, 2, 2),(1, 2, 0):(2, 2, 1),(0, 2, 0):(2, 2, 0),(0, 2, 1):(1, 2, 0),(0, 2, 2): (0, 2, 0),(1, 2, 2):(0, 2, 1),(2, 2, 2):(0, 2, 2),(2, 2, 1):(1, 2, 2),(1, 2, 1):(1, 2, 1)},}

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

        self.cubie_rotation_angle_map={
            "f":[Quaternion(Vector([0,1,0]),pi/2),[(0,0,0),(0,0,1),(0,0,2),(1,0,2),(2,0,2),(2,0,1),(2,0,0),(1,0,0),(1,0,1)]],
            "F":[Quaternion(Vector([0,1,0]),-pi/2),[(0,0,0),(0,0,1),(0,0,2),(1,0,2),(2,0,2),(2,0,1),(2,0,0),(1,0,0),(1,0,1)]],
            "l":[Quaternion(Vector([1,0,0]),pi/2),[(0,2,0),(0,2,1),(0,2,2),(0,1,2),(0,0,2),(0,0,1),(0,0,0),(0,1,0),(0,1,1)]],
            "L":[Quaternion(Vector([1,0,0]),-pi/2),[(0,2,0),(0,2,1),(0,2,2),(0,1,2),(0,0,2),(0,0,1),(0,0,0),(0,1,0),(0,1,1)]],
            "r":[Quaternion(Vector([1,0,0]),-pi/2),[(2,0,0),(2,1,0),(2,2,0),(2,2,1),(2,2,2),(2,1,2),(2,0,2),(2,0,1),(2,1,1)]],
            "R":[Quaternion(Vector([1,0,0]),pi/2),[(2,0,0),(2,1,0),(2,2,0),(2,2,1),(2,2,2),(2,1,2),(2,0,2),(2,0,1),(2,1,1)]],
            "t":[Quaternion(Vector([0,0,1]),-pi/2),[(0,0,2),(0,1,2),(0,2,2),(1,2,2),(2,2,2),(2,1,2),(2,0,2),(1,0,2),(1,1,2)]],
            "T":[Quaternion(Vector([0,0,1]),pi/2),[(0,0,2),(0,1,2),(0,2,2),(1,2,2),(2,2,2),(2,1,2),(2,0,2),(1,0,2),(1,1,2)]],
            "d":[Quaternion(Vector([0,0,1]),pi/2),[(0,0,0),(0,1,0),(0,2,0),(1,2,0),(2,2,0),(2,1,0),(2,0,0),(1,0,0),(1,1,0)]],
            "D":[Quaternion(Vector([0,0,1]),-pi/2),[(0,0,0),(0,1,0),(0,2,0),(1,2,0),(2,2,0),(2,1,0),(2,0,0),(1,0,0),(1,1,0)]],
            "b":[Quaternion(Vector([0,1,0]),-pi/2),[(0,2,0),(0,2,1),(0,2,2),(1,2,2),(2,2,2),(2,2,1),(2,2,0),(1,2,0),(1,2,1)]],
            "B":[Quaternion(Vector([0,1,0]),pi/2),[(0,2,0),(0,2,1),(0,2,2),(1,2,2),(2,2,2),(2,2,1),(2,2,0),(1,2,0),(1,2,1)]]
        }

        cube.add_mesh_modifier(type="NODES", node_modifier=rc_geometry)

        # get input quaternion nodes for the cubies
        self.cubie_rotation_nodes = [ibpy.get_geometry_node_from_modifier(rc_geometry,label="CubieRotation_"+str(i)) for i in range(27)]

        super().__init__(obj = cube.ref_obj,name="Rubik'sCube",**kwargs)

    def transform(self,word,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time/len(word)
        t0 = begin_time
        for letter in word:
            angle,positions = self.cubie_rotation_angle_map[letter]
            transformation = self.transformation_maps[letter]
            # transform relevant cubies
            active_cubies = [self.cube_state[position] for position in positions]
            print(word+": "+letter+" active cubies for: ",[a+1 for a in active_cubies])
            for idx in active_cubies:
                from_angle = self.cubie_rotation_states[idx]
                to_angle =angle@ from_angle
                self.cubie_rotation_states[idx] = to_angle
                ibpy.change_default_quaternion(self.cubie_rotation_nodes[idx],from_value=from_angle,to_value=to_angle,begin_time=t0,transition_time=dt)

            t0+=dt
            # update physical state
            new_state = {}
            for src,replacement in transformation.items():
                new_state[src] = self.cube_state[replacement]

            for key,val in new_state.items():
                self.cube_state[key] = val

        return t0
