import numpy as np
from mathutils import Vector, Quaternion, Euler
from sympy import factor

from interface import ibpy
from interface.ibpy import FOLLOW_PATH_DICTIONARY, Vector, Quaternion, Euler
from objects.bobject import BObject
from objects.curve import BezierDataCurve
from objects.empties import EmptyArrow
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.string_utils import remove_digits, remove_punctuation
from utils.utils import flatten, to_vector, pi

# useful constants
r2 = np.sqrt(2)
ex=Vector([1,0,0])
ey=Vector([0,1,0])
ez=Vector([0,0,1])

class BRubiksCube(BObject):
    def __init__(self, **kwargs):
        """
        This version of the cube contains all the transformations
        the cubies are labelled from the back to the front and from top to bottom
        top level
        1 2 3
        4 5 6
        7 8 9
        middle level
        10 11 12
        13 14 15
        16 17 18
        bottom level
        19 20 21
        22 23 24
        25 26 27

        the cube 1 is corner of the top, left and back face
        the cube 27 is corner of the bottom, front and right face
        """
        self.kwargs = kwargs
        location = self.get_from_kwargs('location', [0, 0, 0])
        rotation = self.get_from_kwargs('rotation_euler', [0, 0, 0])
        colors = self.get_from_kwargs('colors', ["background","text","blue","red","green","orange","yellow"])
        name = self.get_from_kwargs('name', "RubiksCube3x3")
        emission = self.get_from_kwargs('emission', 0.1)

        cubies =["Cube0"+str(i) for i in range(1,10)]+["Cube"+str(i) for i in range(10,28)]
        bobs = BObject.from_file("RubiksCube3x3", objects=cubies,
                                 colors=[], name=name,smooth=3)

        # override default colors
        if colors:
            materials = []
            if isinstance(emission, float):
                emission=[emission]*len(colors)
            for color, emission in zip(colors, emission):
                materials.append(ibpy.get_material("plastic_" + color, emission=emission))

            color_dict = {"Black": materials[0], "White": materials[1], "Blue": materials[2],
                          "Red": materials[3], "Green": materials[-3], "Orange": materials[-2],
                          "Yellow": materials[-1]}

            for bob in bobs:
                slots = bob.ref_obj.material_slots
                for slot in slots:
                    mat_name = remove_punctuation(remove_digits(slot.material.name))
                    if mat_name in color_dict:
                        slot.material = color_dict[mat_name].copy() # here you can decide, whether every face gets its own color or whether all red faces have the same color

        self.children = []

        self.facelets = {
            1: {"parent": 3, "orientation": ez, "color": "white"},
            2: {"parent": 6, "orientation": ez, "color": "white"},
            3: {"parent": 9, "orientation": ez, "color": "white"},
            4: {"parent": 2, "orientation": ez, "color": "white"},
            5: {"parent": 1, "orientation": ez, "color": "white"},
            6: {"parent": 8, "orientation": ez, "color": "white"},
            7: {"parent": 7, "orientation": ez, "color": "white"},
            8: {"parent": 4, "orientation": ez, "color": "white"},
            9: {"parent": 3, "orientation": -ex, "color": "orange"},
            10: {"parent": 2, "orientation": -ex, "color": "orange"},
            11: {"parent": 1, "orientation": -ex, "color": "orange"},
            12: {"parent": 12, "orientation": -ex, "color": "orange"},
            13: {"parent": 21, "orientation": -ex, "color": "orange"},
            14: {"parent": 10, "orientation": -ex, "color": "orange"},
            15: {"parent": 19, "orientation": -ex, "color": "orange"},
            16: {"parent": 20, "orientation": -ex, "color": "orange"},
            17: {"parent": 1, "orientation": -ey, "color": "green"},
            18: {"parent": 4, "orientation": -ey, "color": "green"},
            19: {"parent": 7, "orientation": -ey, "color": "green"},
            20: {"parent": 10, "orientation": -ey, "color": "green"},
            21: {"parent": 19, "orientation": -ey, "color": "green"},
            22: {"parent": 16, "orientation": -ey, "color": "green"},
            23: {"parent": 25, "orientation": -ey, "color": "green"},
            24: {"parent": 22, "orientation": -ey, "color": "green"},
            25: {"parent": 7, "orientation": ex, "color": "red"},
            26: {"parent": 8, "orientation": ex, "color": "red"},
            27: {"parent": 9, "orientation": ex, "color": "red"},
            28: {"parent": 16, "orientation": ex, "color": "red"},
            29: {"parent": 25, "orientation": ex, "color": "red"},
            30: {"parent": 18, "orientation": ex, "color": "red"},
            31: {"parent": 27, "orientation": ex, "color": "red"},
            32: {"parent": 26, "orientation": ex, "color": "red"},
            33: {"parent": 9, "orientation": ey, "color": "blue"},
            34: {"parent": 6, "orientation": ey, "color": "blue"},
            35: {"parent": 3, "orientation": ey, "color": "blue"},
            36: {"parent": 18, "orientation": ey, "color": "blue"},
            37: {"parent": 27, "orientation": ey, "color": "blue"},
            38: {"parent": 12, "orientation": ey, "color": "blue"},
            39: {"parent": 21, "orientation": ey, "color": "blue"},
            40: {"parent": 24, "orientation": ey, "color": "blue"},
            41: {"parent": 19, "orientation": -ez, "color": "yellow"},
            42: {"parent": 22, "orientation": -ez, "color": "yellow"},
            43: {"parent": 25, "orientation": -ez, "color": "yellow"},
            44: {"parent": 20, "orientation": -ez, "color": "yellow"},
            45: {"parent": 21, "orientation": -ez, "color": "yellow"},
            46: {"parent": 26, "orientation": -ez, "color": "yellow"},
            47: {"parent": 27, "orientation": -ez, "color": "yellow"},
            48: {"parent": 24, "orientation": -ez, "color": "yellow"},
        }

        # which facelet belongs to which cubie
        self.cubie_face_map = {}
        for idx in range(1, 28):
            self.cubie_face_map[idx] = []

        for face_id, dict in self.facelets.items():
            self.cubie_face_map[dict["parent"]].append(face_id)

        # center of the cube is at (0,0,0)
        self.physical_locs = {
            1: Vector([-2, -2, 2]),
            2: Vector([-2, 0, 2]),
            3: Vector([-2, 2, 2]),
            4: Vector([0, -2, 2]),
            5: Vector([0, 0, 2]),
            6: Vector([0, 2, 2]),
            7: Vector( [2, -2, 2]),
            8: Vector( [2, 0, 2]),
            9: Vector( [2, 2, 2]),
            10: Vector([-2, -2, 0]),
            11: Vector([-2, 0, 0]),
            12: Vector([-2, 2, 0]),
            13: Vector([0, -2, 0]),
            14: Vector([0, 0, 0]),
            15: Vector([0, 2, 0]),
            16: Vector( [2, -2, 0]),
            17: Vector( [2, 0, 0]),
            18: Vector( [2, 2, 0]),
            19: Vector([-2, -2, -2]),
            20: Vector([-2, 0, -2]),
            21: Vector([-2, 2, -2]),
            22: Vector([0, -2, -2]),
            23: Vector([0, 0, -2]),
            24: Vector([0, 2, -2]),
            25: Vector( [2, -2, -2]),
            26: Vector( [2, 0, -2]),
            27: Vector( [2, 2, -2])
        }

        # the cube 1 is corner of the top, left and back face
        # the cube 27 is corner of the bottom, front and right face
        self.corners = [3, 1, 21, 19, 25, 27, 7, 9]
        self.edges = [12, 2, 20, 10, 4, 22, 24, 6, 16, 26, 8, 18]
        self.cubie_states={}

        for i in range(27):
            # initially, each cubie is at its default position in its default rotation
            self.cubie_states[i + 1] = (i + 1, Quaternion())

        for i,child in enumerate(self.children):
            if i<9:
                child.ref_obj.name="cubie0"+str(i+1)
            else:
                child.ref_obj.name="cubie"+str(i+1)

        if len(bobs) > 0:
           self.children = bobs

        self.empty_arrows = self.create_empty_arrows()
        super().__init__(children=self.children, name=name, rotation_euler=rotation, location=location)
        # make parenting
        for arrow, (idx, val) in zip(self.empty_arrows, self.facelets.items()):
            ibpy.set_parent(arrow, self.children[val["parent"] - 1])
            self.children[val["parent"] - 1].b_children.append(arrow)

        for child,name in zip(self.children,cubies):
            child.ref_obj.name=name

    @classmethod
    def from_state(cls, state, **kwargs):
        rubiks_cube = BRubiksCube(**kwargs)
        for idx, (position, rotation) in state.items():
            rubiks_cube.children[idx - 1].rotate(rotation_quaternion=rotation,begin_time=0,transition_time=0)
        rubiks_cube.cubie_states=state
        return rubiks_cube

    # extended construction
    def create_empty_arrows(self):
        """
        create 48 empty arrows to track the rotation states of the cubie facelets
        """
        arrows = []
        for idx, val in self.facelets.items():
            if val["orientation"].dot(ez) > 0:
                rotation = [0, 0, 0]
            elif val["orientation"].dot(ez) < 0:
                rotation = [pi, 0, 0]
            elif val["orientation"].dot(ey) > 0:
                rotation = [-pi / 2, 0, 0]
            elif val["orientation"].dot(ey) < 0:
                rotation = [pi / 2, 0, 0]
            elif val["orientation"].dot(ex) > 0:
                rotation = [0, pi / 2, 0]
            elif val["orientation"].dot(ex) < 0:
                rotation = [0, -pi / 2, 0]
            arrows.append(EmptyArrow(name="face" + str(idx).zfill(2) + val["color"], rotation_euler=rotation,
                                     location=self.physical_locs[val["parent"]]+val["orientation"]))
        return arrows
    # Getter functions

    def get_cubie_at_position(self, src_pos):
        # find cubie
        idx = -1
        for i, val in self.cubie_states.items():
            if val[0] == src_pos:
                idx = i
                break
        return idx

    def get_permutation(self,time=0):
        """
        returns the permutation of the cube.
        Since the cube can be disassembled it might not be an element of the Rubik's cube group.

        This function always returns the state of the cube at the state of the animation,
        when the function is called. Therefore, the time parameter should match the call position in the animation
        creation thread.
        """
        current_face_map={}
        for i in range(1,49):
            current_face_map[i]=i # identity map

        # default orientation
        default_orientations = {}
        for pos in range(1,28):
            default_orientations[pos]={}
            for face_id in self.cubie_face_map[pos]:
                default_orientations[pos][face_id]=self.facelets[face_id]["orientation"]

        # find the new facelet for each default facelet
        frame = time*FRAME_RATE
        old_frame=ibpy.get_frame()
        ibpy.set_frame(frame)
        for pos in range(1,28):
            idx = self.get_cubie_at_position(pos)
            arrows = self.children[idx-1].b_children
            for arrow in arrows:
                # convert arrow in face information
                rot=ibpy.get_world_matrix(arrow).to_3x3()
                arrow_orientation = rot@Vector([0,0,1]) # default construction of the arrow is upwards
                face_id = int(arrow.ref_obj.name[4:6])
                # match with default information
                for default_face_id,default_orientation in default_orientations[pos].items():
                    if arrow_orientation.dot(default_orientation)>0.9:
                        current_face_map[default_face_id]=face_id

        # convert the current_face_map into a permutation
        map_copy = current_face_map.copy()

        permutation = ""
        while len(map_copy)>0:
            permutation += "("
            key =next(iter(map_copy))
            permutation+=str(key)
            next_key = map_copy[key]
            if next_key!=key:
                permutation+=" "
                permutation+=str(next_key)
            map_copy.pop(key)
            while next_key!=key:
                val = map_copy[next_key]
                if val!=key:
                    permutation+=" "
                    permutation+=str(val)
                map_copy.pop(next_key)
                next_key=val
            permutation+=")"

        ibpy.set_frame(old_frame)
        return permutation

    # transformation functions

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):

        for arrow in self.empty_arrows:
            arrow.appear(begin_time=0,transition_time=0)

        return super().appear(begin_time=begin_time, transition_time=transition_time,children=True)

    def invert(self, word, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.transform(word.swapcase()[::-1],begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def transform(self,word, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time/len(word)
        t0 = begin_time
        for letter in word[::-1]: # apply word from right to left!!!
            if letter=='f':
                # rotation of the front face clockwise
                target_state_pos = [1, 10, 19, 22, 25, 16, 7, 4, 13]
                source_state_pos = [19, 22, 25, 16, 7, 4, 1, 10, 13]
                rot = Quaternion(-ey,-pi/2)
            elif letter=='F':
                # rotation of the front face counterclockwise
                source_state_pos = [1, 10, 19, 22, 25, 16, 7, 4, 13]
                target_state_pos = [19, 22, 25, 16, 7, 4, 1, 10, 13]
                rot = Quaternion(-ey,pi/2)
            elif letter == 'r':
                # rotation of the right face clockwise
                source_state_pos = [25, 26, 27, 18, 9, 8, 7, 16, 17]
                target_state_pos = [7, 16, 25, 26, 27, 18, 9, 8, 17]
                rot = Quaternion(ex, -pi / 2)
            elif letter == 'R':
                # rotation of the right face counterclockwise
                target_state_pos = [25, 26, 27, 18, 9, 8, 7, 16, 17]
                source_state_pos = [7, 16, 25, 26, 27, 18, 9, 8, 17]
                rot = Quaternion(ex, pi / 2)
            elif letter == 'l':
                # rotation of the left face clockwise
                target_state_pos = [3, 12, 21, 20, 19, 10, 1, 2, 11]
                source_state_pos = [21, 20, 19, 10, 1, 2, 3, 12, 11]
                rot = Quaternion(-ex, -pi / 2)
            elif letter == 'L':
                # rotation of the left face counterclockwise
                source_state_pos = [3, 12, 21, 20, 19, 10, 1, 2, 11]
                target_state_pos = [21, 20, 19, 10, 1, 2, 3, 12, 11]
                rot = Quaternion(-ex, pi / 2)
            elif letter == 'b':
                # rotation of the back face clockwise
                target_state_pos = [9, 18, 27, 24, 21, 12, 3, 6, 15]
                source_state_pos = [27, 24, 21, 12, 3, 6, 9, 18, 15]
                rot = Quaternion(ey, -pi / 2)
            elif letter == 'B':
                # rotation of the back face counterclockwise
                source_state_pos = [9, 18, 27, 24, 21, 12, 3, 6, 15]
                target_state_pos = [27, 24, 21, 12, 3, 6, 9, 18, 15]
                rot = Quaternion(ey, pi / 2)
            elif letter == 'd':
                # rotation of the down face clockwise
                target_state_pos = [25, 22, 19, 20, 21, 24, 27, 26, 23]
                source_state_pos = [19, 20, 21, 24, 27, 26, 25, 22, 23]
                rot = Quaternion(-ez, -pi / 2)
            elif letter == 'D':
                # rotation of the down face counterclockwise
                source_state_pos = [25, 22, 19, 20, 21, 24, 27, 26, 23]
                target_state_pos = [19, 20, 21, 24, 27, 26, 25, 22, 23]
                rot = Quaternion(-ez, pi / 2)
            elif letter == 't':
                # rotation of the down face clockwise
                target_state_pos = [1, 4, 7, 8, 9, 6, 3, 2, 5]
                source_state_pos = [7, 8, 9, 6, 3, 2, 1, 4, 5]
                rot = Quaternion(ez, -pi / 2)
            elif letter == 'T':
                # rotation of the down face counterclockwise
                source_state_pos = [1, 4, 7, 8, 9, 6, 3, 2, 5]
                target_state_pos = [7, 8, 9, 6, 3, 2, 1, 4, 5]
                rot = Quaternion(ez, pi / 2)

            # we need to grap the indices before the transformation, since the map will change during the transformation
            indices = [self.get_cubie_at_position(src) for src in source_state_pos]
            [self.children[idx-1].change_emission(from_value=0,to_value=10,slots=list(range(1,7)),begin_time=t0,transition_time=0) for idx in indices]
            rotations = [self.cubie_states[idx][1] for idx in indices]
            for idx,old_rotation,target in zip(indices,rotations,target_state_pos):
                new_rotation = rot@old_rotation
                self.children[idx-1].rotate(rotation_quaternion=new_rotation,begin_time=t0,transition_time=dt)
                self.cubie_states[idx]=(target,new_rotation)
            t0  = t0+dt
            [self.children[idx - 1].change_emission(from_value=10, to_value=0,slots=list(range(1,7)), begin_time=t0-dt/2, transition_time=dt/2) for idx
             in indices]

        # make action curves linear to avoid distortions
        for child in self.children:
            ibpy.set_linear_fcurves(child)

        return begin_time+transition_time


class BRubiksCubeLocalCenters(BObject):
    def __init__(self, **kwargs):
        """
        This cube allows the decomposition into cubies
        the cubies are labelled from the back to the front and from top to bottom
        top level
        1 2 3
        4 5 6
        7 8 9
        middle level
        10 11 12
        13 14 15
        16 17 18
        bottom level
        19 20 21
        22 23 24
        25 26 27
        """
        self.kwargs = kwargs
        location = self.get_from_kwargs('location', [0, 0, 0])
        rotation = self.get_from_kwargs('rotation_euler', [0, 0, 0])
        # necessary to load default colors
        colors = self.get_from_kwargs('colors', ["background","text","blue","red","green","orange","yellow"])
        name = self.get_from_kwargs('name', "RubiksCube3x3")
        emission = self.get_from_kwargs('emission', 0.1)
        cubies =["Cube0"+str(i) for i in range(1,10)]+["Cube"+str(i) for i in range(10,28)]
        # load cube with default colors
        bobs = BObject.from_file("RubiksCube3x3LocalCenters", objects=cubies,colors=[],
                                 name=name,smooth=3)

        #override default colors
        if colors:
            materials =[]
            if isinstance(emission,float):
                emission=[emission]*len(colors)
            for color,emission in zip(colors,emission):
                materials.append(ibpy.get_material("plastic_"+color,emission=emission))

            color_dict={"Black":materials[0],"White":materials[1],"Blue":materials[2],
                        "Red":materials[3],"Green":materials[-3],"Orange":materials[-2],
                        "Yellow":materials[-1]}

            for bob in bobs:
                slots = bob.ref_obj.material_slots
                for slot in slots:
                    mat_name=remove_punctuation(remove_digits(slot.material.name))
                    if  mat_name in color_dict:
                        slot.material = color_dict[mat_name].copy()

        self.children = []

        # capture the rotation state of each cubie
        self.cubie_states = {}

        # the cube 1 is corner of the top, left and back face
        # the cube 27 is corner of the bottom, front and right face
        self.corners = [3, 1, 21, 19, 25, 27, 7, 9]
        self.edges = [12, 2, 20, 10, 4, 22, 24, 6, 16, 26, 8, 18]

        for i in range(27):
            # initially, each cubie is at its default position in its default rotation
            self.cubie_states[i + 1]=(i+1, Quaternion())

        if len(bobs) > 0:
           self.children = bobs

        # Mappings between faclets and cubies
        self.facelets = {
            1: {"parent": 3, "orientation": ez, "color": "white"},
            2: {"parent": 6, "orientation": ez, "color": "white"},
            3: {"parent": 9, "orientation": ez, "color": "white"},
            4: {"parent": 2, "orientation": ez, "color": "white"},
            5: {"parent": 1, "orientation": ez, "color": "white"},
            6: {"parent": 8, "orientation": ez, "color": "white"},
            7: {"parent": 7, "orientation": ez, "color": "white"},
            8: {"parent": 4, "orientation": ez, "color": "white"},
            9: {"parent": 3, "orientation": -ex, "color": "orange"},
            10: {"parent": 2, "orientation": -ex, "color": "orange"},
            11: {"parent": 1, "orientation": -ex, "color": "orange"},
            12: {"parent": 12, "orientation": -ex, "color": "orange"},
            13: {"parent": 21, "orientation": -ex, "color": "orange"},
            14: {"parent": 10, "orientation": -ex, "color": "orange"},
            15: {"parent": 19, "orientation": -ex, "color": "orange"},
            16: {"parent": 20, "orientation": -ex, "color": "orange"},
            17: {"parent": 1, "orientation": -ey, "color": "green"},
            18: {"parent": 4, "orientation": -ey, "color": "green"},
            19: {"parent": 7, "orientation": -ey, "color": "green"},
            20: {"parent": 10, "orientation": -ey, "color": "green"},
            21: {"parent": 19, "orientation": -ey, "color": "green"},
            22: {"parent": 16, "orientation": -ey, "color": "green"},
            23: {"parent": 25, "orientation": -ey, "color": "green"},
            24: {"parent": 22, "orientation": -ey, "color": "green"},
            25: {"parent": 7, "orientation": ex, "color": "red"},
            26: {"parent": 8, "orientation": ex, "color": "red"},
            27: {"parent": 9, "orientation": ex, "color": "red"},
            28: {"parent": 16, "orientation": ex, "color": "red"},
            29: {"parent": 25, "orientation": ex, "color": "red"},
            30: {"parent": 18, "orientation": ex, "color": "red"},
            31: {"parent": 27, "orientation": ex, "color": "red"},
            32: {"parent": 26, "orientation": ex, "color": "red"},
            33: {"parent": 9, "orientation": ey, "color": "blue"},
            34: {"parent": 6, "orientation": ey, "color": "blue"},
            35: {"parent": 3, "orientation": ey, "color": "blue"},
            36: {"parent": 18, "orientation": ey, "color": "blue"},
            37: {"parent": 27, "orientation": ey, "color": "blue"},
            38: {"parent": 12, "orientation": ey, "color": "blue"},
            39: {"parent": 21, "orientation": ey, "color": "blue"},
            40: {"parent": 24, "orientation": ey, "color": "blue"},
            41: {"parent": 19, "orientation": -ez, "color": "yellow"},
            42: {"parent": 22, "orientation": -ez, "color": "yellow"},
            43: {"parent": 25, "orientation": -ez, "color": "yellow"},
            44: {"parent": 20, "orientation": -ez, "color": "yellow"},
            45: {"parent": 21, "orientation": -ez, "color": "yellow"},
            46: {"parent": 26, "orientation": -ez, "color": "yellow"},
            47: {"parent": 27, "orientation": -ez, "color": "yellow"},
            48: {"parent": 24, "orientation": -ez, "color": "yellow"},
        }

        # which facelet belongs to which cubie
        self.cubie_face_map = {}
        for idx in range(1, 28):
            self.cubie_face_map[idx] = []

        for face_id,dict  in self.facelets.items():
            self.cubie_face_map[dict["parent"]].append(face_id)

        # center of the cube is at (2,2,2)
        self.physical_locs ={
            1:Vector([0,0,4]),
            2:Vector([0,2,4]),
            3:Vector([0,4,4]),
            4:Vector([2,0,4]),
            5:Vector([2,2,4]),
            6:Vector([2,4,4]),
            7:Vector([4,0,4]),
            8:Vector([4,2,4]),
            9:Vector([4,4,4]),
            10:Vector([0,0,2]),
            11:Vector([0,2,2]),
            12:Vector([0,4,2]),
            13:Vector([2,0,2]),
            14:Vector([2,2,2]),
            15:Vector([2,4,2]),
            16:Vector([4,0,2]),
            17:Vector([4,2,2]),
            18:Vector([4,4,2]),
            19:Vector([0,0,0]),
            20:Vector([0,2,0]),
            21:Vector([0,4,0]),
            22:Vector([2,0,0]),
            23:Vector([2,2,0]),
            24:Vector([2,4,0]),
            25:Vector([4,0,0]),
            26:Vector([4,2,0]),
            27:Vector([4,4,0])
        }

        # map from the Rubik's cube state to the disassembled state
        self.disassemble_rotation ={
            # corners
            3:Euler([pi / 2, 0, pi / 2]).to_quaternion(),
            1:Euler([0, pi / 2, 0]).to_quaternion(),
            21:Euler([-pi / 2, pi, 0]).to_quaternion(),
            19:Euler([-pi / 2, pi / 2, 0]).to_quaternion(),
            25:Euler([-pi / 2, 0, 0]).to_quaternion(),
            27:Euler([-pi / 2, -pi / 2, 0]).to_quaternion(),
            7:Euler([0, 0, 0]).to_quaternion(),
            9:Euler([pi / 2, 0, 0]).to_quaternion(),
            # edges
            12:Euler([pi/2,0,pi/2]).to_quaternion(),
            2: Euler([0,0,pi/2]).to_quaternion(),
            20: Euler([0, pi / 2, pi / 2]).to_quaternion(),
            10: Euler([0, pi / 2, 0]).to_quaternion(),
            4: Euler([0, 0, 0]).to_quaternion(),
            22: Euler([-pi / 2, 0, 0]).to_quaternion(),
            24: Euler([-pi, 0, 0]).to_quaternion(),
            6: Euler([pi / 2, 0, 0]).to_quaternion(),
            16: Euler([0, -pi / 2, 0]).to_quaternion(),
            26: Euler([0, -pi, pi / 2]).to_quaternion(),
            8: Euler([0, 0, -pi / 2]).to_quaternion(),
            18: Euler([pi / 2, 0, -pi / 2]).to_quaternion(),
        }

        for i,(child,loc) in enumerate(zip(self.children,self.physical_locs.values())):
            child.ref_obj.location=loc
            if i<9:
                child.ref_obj.name="cubie0"+str(i+1)
            else:
                child.ref_obj.name="cubie"+str(i+1)

        self.curve_map={}
        self.create_displacement_curves()
        self.empty_arrows=self.create_empty_arrows()
        super().__init__(children=self.children+list(self.curve_map.values())+self.empty_arrows, name=name, rotation_quaternion=Euler(rotation).to_quaternion(), location=location)
        # make parenting
        for arrow,(idx,val) in zip(self.empty_arrows,self.facelets.items()):
            ibpy.set_parent(arrow, self.children[val["parent"] - 1])
            self.children[val["parent"]-1].b_children.append(arrow)

    # extended construction

    def create_empty_arrows(self):
        """
        create 48 empty arrows to track the rotation states of the cubie facelets
        """

        arrows=[]
        for idx,val in self.facelets.items():
            if val["orientation"].dot(ez)>0:
                rotation = [0,0,0]
            elif val["orientation"].dot(ez)<0:
                rotation=[pi,0,0]
            elif val["orientation"].dot(ey)>0:
                rotation=[-pi/2,0,0]
            elif val["orientation"].dot(ey)<0:
                rotation=[pi/2,0,0]
            elif val["orientation"].dot(ex)>0:
                rotation=[0,pi/2,0]
            elif val["orientation"].dot(ex)<0:
                rotation=[0,-pi/2,0]
            arrows.append(EmptyArrow(name="face"+str(idx).zfill(2)+val["color"],rotation_euler=rotation,location=val["orientation"]))
        return arrows

    def create_displacement_curves(self):
        # corners first

        x_vals = [-6 + 16 / 7 * i for i in range(8)]
        line = -10
        for idx, x in zip(self.corners, x_vals):
            vector = Vector(self.get_cubie_location(idx))
            if vector[2] > 0:
                mid = Vector([x, line + 0.75 * (vector[1] - line), 1.5 * (vector[2])])
            else:
                if idx == 19 or idx == 25:
                    sgn = -1
                else:
                    sgn = 1
                mid = Vector([x + sgn * 2 * (x - vector[0]), line + 0.75 * (vector[1] - line), 1.5 * (vector[2])])
            curve = BezierDataCurve(
                data=[
                    [x, line, 0],
                    mid,
                    vector
                ], make_pieces=False)

            self.curve_map[idx] = curve

        line = -6
        x_vals = [-12 + 27 / 11 * i for i in range(12)]
        edge_curves = []
        for idx, x in zip(self.edges,x_vals):
            vector = Vector(self.get_cubie_location(idx))
            if vector[2] > 0:
                mid = Vector([x, line + 0.75 * (vector[1] - line), 2 * (vector[2])])
            else:
                mid = Vector([x, line + 0.75 * (vector[1] - line), 2 * (vector[2])])
            if idx == 24:
                mid = Vector([mid[0], mid[1], -4])
            curve = BezierDataCurve(
                data=[
                    [x, line, 0],
                    mid,
                    vector
                ], make_pieces=False
            )

            self.curve_map[idx]=curve

    # getter

    def get_cubie_location(self,idx):
        return to_vector(self.physical_locs[idx])

    def get_center(self):
        return to_vector(self.locs[13])

    def get_cubie_at_position(self, src_pos):
        # find cubie
        idx = -1
        for i, val in self.cubie_states.items():
            if val[0] == src_pos:
                idx = i
                break
        return idx

    def get_axis_at_position(self, position):
        """
        returns the axis of rotation for a given position
        The axis is radially pointing outward, normalized and can be used to rotate cubies indiviually
        """
        axis = to_vector(self.physical_locs[position]) - to_vector(self.physical_locs[14])
        if (axis.length > 0):
            axis = axis.normalized()
        return axis

    def get_corners(self):
        return self.corners

    def get_edges(self):
        return self.edges

    def get_corner_curves(self):
        return [self.curve_map[idx] for idx in self.corners]

    def get_edge_curves(self):
        return [self.curve_map[idx] for idx in self.edges]

    def get_permutation(self,time=0):
        """
        returns the permutation of the cube.
        Since the cube can be disassembled it might not be an element of the Rubik's cube group.

        This function always returns the state of the cube at the state of the animation,
        when the function is called. Therefore, the time parameter should match the call position in the animation
        creation thread.
        """
        current_face_map={}
        for i in range(1,49):
            current_face_map[i]=i # identity map

        # default orientation
        default_orientations = {}
        for pos in range(1,28):
            default_orientations[pos]={}
            for face_id in self.cubie_face_map[pos]:
                default_orientations[pos][face_id]=self.facelets[face_id]["orientation"]

        # find the new facelet for each default facelet
        frame = time*FRAME_RATE
        old_frame=ibpy.get_frame()
        ibpy.set_frame(frame)
        for pos in range(1,28):
            idx = self.get_cubie_at_position(pos)
            arrows = self.children[idx-1].b_children
            for arrow in arrows:
                # convert arrow in face information
                rot=ibpy.get_world_matrix(arrow).to_3x3()
                arrow_orientation = rot@Vector([0,0,1]) # default construction of the arrow is upwards
                face_id = int(arrow.ref_obj.name[4:6])
                # match with default information
                for default_face_id,default_orientation in default_orientations[pos].items():
                    if arrow_orientation.dot(default_orientation)>0.9:
                        current_face_map[default_face_id]=face_id

        # convert the current_face_map into a permutation
        map_copy = current_face_map.copy()

        permutation = ""
        while len(map_copy)>0:
            permutation += "("
            key =next(iter(map_copy))
            permutation+=str(key)
            next_key = map_copy[key]
            if next_key!=key:
                permutation+=" "
                permutation+=str(next_key)
            map_copy.pop(key)
            while next_key!=key:
                val = map_copy[next_key]
                if val!=key:
                    permutation+=" "
                    permutation+=str(val)
                map_copy.pop(next_key)
                next_key=val
            permutation+=")"



        ibpy.set_frame(old_frame)
        return permutation

    # Appearance and transformations
    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):

        dt = 0.75*transition_time
        shift = 0.25*transition_time/9

        for i,child in enumerate(self.children):
            child.scale(initial_scale=0,final_scale=1,begin_time=begin_time+shift*i,transition_time=dt)

        for arrow in self.empty_arrows:
            arrow.appear(begin_time=0,transition_time=0)
        return super().appear(begin_time=begin_time, transition_time=transition_time,children=False)

    def align_cubie_with_curve_and_rotate(self, idx, curve,rotation,reverse=False, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.children[idx-1].ref_obj.location = Vector()
        if not (self.children[idx-1],curve) in FOLLOW_PATH_DICTIONARY.keys():
            ibpy.set_follow(self.children[idx-1], curve,use_curve_follow=False)
        if reverse:
            # displace cubie onto the representation line and rotate it suitably for camera position
            ibpy.follow(self.children[idx-1], curve,initial_value=1,final_value=0, begin_time=begin_time, transition_time=transition_time)
            if not isinstance(rotation, Quaternion):
                rotation = Euler(rotation).to_quaternion();  # convert to quaternion if necessary
            self.children[idx - 1].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                          transition_time=transition_time)
            self.cubie_states[idx]=(-1,rotation)
        else:
            # take cubie back to its default position and rotation state
            ibpy.follow(self.children[idx-1], curve,initial_value=0,final_value=1, begin_time=begin_time, transition_time=transition_time)
            self.children[idx-1].rotate(rotation_quaternions=Quaternion(),begin_time=begin_time,transition_time=transition_time)
            self.cubie_states[idx]=(idx,Quaternion())
        return begin_time+transition_time

    def swap_position_and_rotate(self, idx, curves, rotations, begin_time, transition_time):
        """The first curve is the original curve that the cubie was assigned to"""
        dt=0.75* transition_time/(len(curves)-1)
        if len(curves)>2:
            pause = 0.25 * transition_time / (len(curves) - 2)
        else:
            pause = 0
        for curve in curves:
            if not (self.children[idx - 1], curve) in FOLLOW_PATH_DICTIONARY.keys():
                ibpy.set_follow(self.children[idx - 1], curve, use_curve_follow=False)
                ibpy.set_follow_influence(self.children[idx - 1], curve, 0,begin_time=0)

        t0 =begin_time
        for i in range(len(curves)-1):
            if i>0: # assume a dissasembled cube, therefore the firs time the cube does not to be moved
                ibpy.follow(self.children[idx - 1], curves[i], initial_value=1, final_value=0, begin_time=t0, transition_time=dt)
            ibpy.follow(self.children[idx - 1], curves[i+1], initial_value=0, final_value=1, begin_time=t0, transition_time=dt)
            ibpy.change_follow_influence(self.children[idx - 1], curves[i], 1,0, begin_time=t0,transition_time=dt)
            ibpy.change_follow_influence(self.children[idx-1],curves[i+1],0,1,begin_time=t0,transition_time=dt)
            t0 = pause + self.children[idx - 1].rotate(rotation_quaternion=Euler(rotations[i]).to_quaternion(),begin_time=t0,transition_time=dt)

        return begin_time+transition_time

    def disassemble_corners(self,sequence,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        t0 = begin_time
        for idx in sequence:
            duration = 5
            dt = duration / len(sequence)
            self.align_cubie_with_curve_and_rotate(idx,self.curve_map[idx],self.disassemble_rotation[idx],reverse=True,begin_time=t0,transition_time=2*dt)
            t0+=dt

        return begin_time+transition_time

    def disassemble_edges(self,sequence,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        t0 = begin_time
        for idx in sequence:
            duration = 5
            dt = duration / len(sequence)
            self.align_cubie_with_curve_and_rotate(idx,self.curve_map[idx],self.disassemble_rotation[idx],reverse=True,begin_time=t0,transition_time=2*dt)
            t0+=dt

        return begin_time+transition_time

    def reassemble_cubie(self,idx,pos,twist=0,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        print("reassemble cubie ", idx, " at position ", pos, " at time ", begin_time, " with duration ",
              transition_time)
        if idx==pos:
            # reassembly of own curve
            rotation = Quaternion()
            ibpy.follow(self.children[idx-1],self.curve_map[idx],initial_value=0,final_value=1,begin_time=begin_time,transition_time=transition_time)
        else:
            curves = [self.curve_map[idx],self.curve_map[pos]]
            old_quaternion = self.disassemble_rotation[idx].copy()
            new_quaternion = self.disassemble_rotation[pos].copy()
            new_quaternion.invert()
            rotation = new_quaternion@old_quaternion
            for curve in curves:
                if not (self.children[idx - 1], curve) in FOLLOW_PATH_DICTIONARY.keys():
                    ibpy.set_follow(self.children[idx - 1], curve, use_curve_follow=False)
                    ibpy.set_follow_influence(self.children[idx - 1], curve, 0, begin_time=0)

            ibpy.follow(self.children[idx - 1], curves[0], initial_value=0, final_value=1, begin_time=begin_time, transition_time=transition_time)
            ibpy.follow(self.children[idx - 1], curves[1], initial_value=0, final_value=1, begin_time=begin_time, transition_time=transition_time)
            ibpy.change_follow_influence(self.children[idx - 1], curves[0], 1, 0, begin_time=begin_time, transition_time=transition_time)
            ibpy.change_follow_influence(self.children[idx - 1], curves[1], 0, 1, begin_time=begin_time, transition_time=transition_time)
        self.children[idx - 1].rotate(rotation_quaternion=rotation,begin_time=begin_time, transition_time=transition_time)
        self.cubie_states[idx]=(pos,rotation)
        return begin_time+transition_time

    def swap_cubies(self, pos1, pos2, rot1=Quaternion(),rot2=Quaternion(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,factor=1):
        # get physical cubies at the swapping positions
        idx1 = self.get_cubie_at_position(pos1)
        idx2 = self.get_cubie_at_position(pos2)

        print("swap cubies ",idx1," and ",idx2)
        # get curves
        curve1 = self.curve_map[pos1]
        curve2 = self.curve_map[pos2]

        for curve in {curve1,curve2}:
            for idx in {idx1,idx2}:
                if not (self.children[idx - 1], curve) in FOLLOW_PATH_DICTIONARY.keys():
                    ibpy.set_follow(self.children[idx - 1], curve, use_curve_follow=False)
                    ibpy.set_follow_influence(self.children[idx - 1], curve, 0, begin_time=0)

        # redirect first cubie
        one = self.children[idx1 - 1]
        ibpy.follow(one, curve1, initial_value=1, final_value=1, begin_time=begin_time, transition_time=transition_time)
        ibpy.follow(one, curve2, initial_value=1, final_value=1, begin_time=begin_time, transition_time=transition_time)
        ibpy.change_follow_influence(one, curve1, 0, factor, begin_time=begin_time, transition_time=transition_time / 2)
        ibpy.change_follow_influence(one, curve2, 0, factor, begin_time=begin_time, transition_time=transition_time / 2)
        ibpy.change_follow_influence(one, curve1, factor, 0, begin_time=begin_time+transition_time/2, transition_time=transition_time / 2)
        ibpy.change_follow_influence(one, curve2, factor, 0, begin_time=begin_time+transition_time/2, transition_time=transition_time / 2)
        self.children[idx1 - 1].move_to(target_location=self.get_cubie_location(pos2),begin_time=begin_time+transition_time/4,transition_time=transition_time/2)

        # redirect second cubie
        two = self.children[idx2 - 1]
        ibpy.follow(two, curve2, initial_value=1, final_value=1, begin_time=begin_time,transition_time=transition_time)
        ibpy.follow(two, curve1, initial_value=1, final_value=1, begin_time=begin_time,transition_time=transition_time)
        ibpy.change_follow_influence(two, curve2, 0, factor, begin_time=begin_time, transition_time=transition_time/2)
        ibpy.change_follow_influence(two, curve1, 0, factor, begin_time=begin_time, transition_time=transition_time/2)
        ibpy.change_follow_influence(two, curve1, factor, 0, begin_time=begin_time + transition_time / 2,transition_time=transition_time / 2)
        ibpy.change_follow_influence(two, curve2, factor, 0, begin_time=begin_time + transition_time / 2,transition_time=transition_time / 2)
        self.children[idx2-1].move_to(target_location=self.get_cubie_location(pos1),begin_time=begin_time+transition_time/4,transition_time=transition_time/2)

        # rotate cubies
        quaternion1 = self.disassemble_rotation[pos1].copy()
        quaternion2 = self.disassemble_rotation[pos2].copy()
        iquaternion1=quaternion1.copy()
        iquaternion2=quaternion2.copy()
        iquaternion1.invert()
        iquaternion2.invert()
        rotation1 = rot1@iquaternion2@quaternion1
        rotation2 = rot2@iquaternion1@quaternion2

        state1 = self.cubie_states[idx1][1]
        state2 = self.cubie_states[idx2][1]
        new_state1 = rotation1 @ state1
        self.children[idx1-1].rotate(rotation_quaternion=new_state1, begin_time=begin_time, transition_time=transition_time)
        new_state2 = rotation2 @ state2
        self.children[idx2-1].rotate(rotation_quaternion=new_state2, begin_time=begin_time, transition_time=transition_time)

        # update state
        self.cubie_states[idx1]=(pos2,new_state1)
        self.cubie_states[idx2]=(pos1,new_state2)

        return begin_time+transition_time

    def move_cubies(self,src_pos,target_positions,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        """
        Move cubie from one position of the rubik's cube to another.
         It is assumed that the cubie is located at the rubik's cube already.
        """
        print("move_cubie from ",src_pos," to ",target_positions," at ",begin_time," with ",transition_time," seconds duration")
        dt = transition_time/len(target_positions)
        t0 = begin_time

        idx = self.get_cubie_at_position(src_pos)

        if idx != -1:
            target_positions.insert(0,src_pos)
            for i in range(1,len(target_positions)):
                pos = target_positions[i]
                previous = target_positions[i-1]
                curves = [self.curve_map[previous],self.curve_map[pos]]
                for curve in curves:
                    if not (self.children[idx - 1], curve) in FOLLOW_PATH_DICTIONARY.keys():
                        ibpy.set_follow(self.children[idx - 1], curve, use_curve_follow=False)
                        ibpy.set_follow_influence(self.children[idx - 1], curve, 0, begin_time=0)

                ibpy.follow(self.children[idx - 1], curves[0], initial_value=1, final_value=0, begin_time=t0,
                            transition_time=dt)
                ibpy.follow(self.children[idx - 1], curves[1], initial_value=0, final_value=1, begin_time=t0,
                            transition_time=dt)
                ibpy.change_follow_influence(self.children[idx - 1], curves[0], 1, 0, begin_time=t0,
                                             transition_time=dt)
                ibpy.change_follow_influence(self.children[idx - 1], curves[1], 0, 1, begin_time=t0,
                                             transition_time=dt)

                old_quaternion = self.disassemble_rotation[idx].copy() # get original rotation into disassembly
                new_quaternion = self.disassemble_rotation[pos].copy()
                new_quaternion.invert()
                rotation =  new_quaternion@old_quaternion
                t0 = self.children[idx - 1].rotate(rotation_quaternion=rotation, begin_time=t0, transition_time=dt)
                self.cubie_states[idx]=(pos,rotation)
        return begin_time + transition_time

    def turn_edge(self, position, begin_time, transition_time):
        print("turning edge at ",position," at time ",begin_time," with transition time ",transition_time)
        idx = self.get_cubie_at_position(position)
        dt = transition_time / 3
        if idx != -1:
            axis = self.get_axis_at_position(position)
            # move cubie partially away
            curve = self.curve_map[position]
            ibpy.follow(self.children[idx - 1], curve, initial_value=1, final_value=0.5, begin_time=begin_time,transition_time=dt)
            rotation = self.cubie_states[idx][1]
            rotation =  Quaternion(axis,pi)@ rotation
            print("rotate cubie at ",position," around ",axis," by ",rotation)
            self.children[idx - 1].rotate(rotation_quaternion=rotation, begin_time=begin_time+dt, transition_time=dt)
            self.cubie_states[idx] = (position, rotation)
            # move cubie back
            ibpy.follow(self.children[idx - 1], curve, initial_value=0.5, final_value=1, begin_time=begin_time+2*dt,transition_time=dt)
            return begin_time+transition_time
        return begin_time

    def turn_corner(self, position,angle=2, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        print("turning edge at ", position, " at time ", begin_time, " with transition time ", transition_time)
        idx = self.get_cubie_at_position(position)
        dt = transition_time / 5
        if idx != -1:
            axis = self.get_axis_at_position(position)
            # move cubie partially away
            curve = self.curve_map[position]
            ibpy.follow(self.children[idx - 1], curve, initial_value=1, final_value=0.5, begin_time=begin_time,
                        transition_time=dt)
            rotation = self.cubie_states[idx][1]
            rotation = Quaternion(axis, 2*pi/3) @ rotation
            print("rotate cubie at ", position, " around ", axis, " by ", rotation)
            self.children[idx - 1].rotate(rotation_quaternion=rotation, begin_time=begin_time + dt, transition_time=dt)
            if angle==2:
                rotation = Quaternion(axis, 2 * pi / 3) @ rotation
                print("rotate cubie at ", position, " around ", axis, " by ", rotation)
                self.children[idx - 1].rotate(rotation_quaternion=rotation, begin_time=begin_time + 3*dt, transition_time=dt)
            self.cubie_states[idx] = (position, rotation)
            # move cubie back
            ibpy.follow(self.children[idx - 1], curve, initial_value=0.5, final_value=1, begin_time=begin_time + 4 * dt,
                        transition_time=dt)
            return begin_time + transition_time
        return begin_time

    def transform(self,word, begin_time=0,steps=1, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time/len(word)
        t0 = begin_time
        for letter in word[::-1]: # apply word from right to left!!!
            if letter=='f':
                # rotation of the front face clockwise
                target_state_pos = [1, 10, 19, 22, 25, 16, 7, 4, 13]
                source_state_pos = [19, 22, 25, 16, 7, 4, 1, 10, 13]
                pivot = self.physical_locs[13]
                rot = Quaternion(-ey,-pi/2/steps)
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter=='F':
                # rotation of the front face counterclockwise
                source_state_pos = [1, 10, 19, 22, 25, 16, 7, 4, 13]
                target_state_pos = [19, 22, 25, 16, 7, 4, 1, 10, 13]
                rot = Quaternion(-ey,pi/2/steps)
                pivot = self.physical_locs[13]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'r':
                # rotation of the right face clockwise
                source_state_pos = [25, 26, 27, 18, 9, 8, 7, 16, 17]
                target_state_pos = [7, 16, 25, 26, 27, 18, 9, 8, 17]
                rot = Quaternion(ex, -pi / 2/steps)
                pivot = self.physical_locs[17]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'R':
                # rotation of the right face counterclockwise
                target_state_pos = [25, 26, 27, 18, 9, 8, 7, 16, 17]
                source_state_pos = [7, 16, 25, 26, 27, 18, 9, 8, 17]
                rot = Quaternion(ex, pi / 2/steps)
                pivot = self.physical_locs[17]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'l':
                # rotation of the left face clockwise
                target_state_pos = [3, 12, 21, 20, 19, 10, 1, 2, 11]
                source_state_pos = [21, 20, 19, 10, 1, 2, 3, 12, 11]
                rot = Quaternion(-ex, -pi / 2/steps)
                pivot = self.physical_locs[11]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'L':
                # rotation of the left face counterclockwise
                source_state_pos = [3, 12, 21, 20, 19, 10, 1, 2, 11]
                target_state_pos = [21, 20, 19, 10, 1, 2, 3, 12, 11]
                rot = Quaternion(-ex, pi / 2/steps)
                pivot = self.physical_locs[11]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'b':
                # rotation of the back face clockwise
                target_state_pos = [9, 18, 27, 24, 21, 12, 3, 6, 15]
                source_state_pos = [27, 24, 21, 12, 3, 6, 9, 18, 15]
                rot = Quaternion(ey, -pi / 2/steps)
                pivot = self.physical_locs[15]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'B':
                # rotation of the back face counterclockwise
                source_state_pos = [9, 18, 27, 24, 21, 12, 3, 6, 15]
                target_state_pos = [27, 24, 21, 12, 3, 6, 9, 18, 15]
                rot = Quaternion(ey, pi / 2/steps)
                pivot = self.physical_locs[15]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'd':
                # rotation of the down face clockwise
                target_state_pos = [25, 22, 19, 20, 21, 24, 27, 26, 23]
                source_state_pos = [19, 20, 21, 24, 27, 26, 25, 22, 23]
                rot = Quaternion(-ez, -pi / 2/steps)
                pivot = self.physical_locs[23]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'D':
                # rotation of the down face counterclockwise
                source_state_pos = [25, 22, 19, 20, 21, 24, 27, 26, 23]
                target_state_pos = [19, 20, 21, 24, 27, 26, 25, 22, 23]
                rot = Quaternion(-ez, pi / 2/steps)
                pivot = self.physical_locs[23]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 't':
                # rotation of the down face clockwise
                target_state_pos = [1, 4, 7, 8, 9, 6, 3, 2, 5]
                source_state_pos = [7, 8, 9, 6, 3, 2, 1, 4, 5]
                rot = Quaternion(ez, -pi / 2/steps)
                pivot = self.physical_locs[5]
                rot_matrix = rot.to_matrix().to_3x3()
            elif letter == 'T':
                # rotation of the down face counterclockwise
                source_state_pos = [1, 4, 7, 8, 9, 6, 3, 2, 5]
                target_state_pos = [7, 8, 9, 6, 3, 2, 1, 4, 5]
                rot = Quaternion(ez, pi / 2/steps)
                pivot = self.physical_locs[5]
                rot_matrix = rot.to_matrix().to_3x3()

            # we need to grap the indices before the transformation, since the map will change during the transformation
            indices = [self.get_cubie_at_position(src) for src in source_state_pos]
            [self.children[idx-1].change_emission(from_value=0,to_value=5,slots=list(range(1,7)),begin_time=t0,transition_time=0) for idx in indices]
            rotations = [self.cubie_states[idx][1] for idx in indices]
            ddt = dt/steps
            for idx,old_rotation,target in zip(indices,rotations,target_state_pos):
                position  = self.physical_locs[self.cubie_states[idx][0]].copy()
                # the transformation has to be broken up into pieces, since the rotation and shift do not transform equally across a quarter turn
                for i in range(steps):
                    new_rotation = rot@old_rotation
                    # make a 10 percent pause between to subsequent animations
                    self.children[idx-1].rotate(rotation_quaternion=new_rotation,begin_time=t0+i*ddt,transition_time=0.9*ddt)
                    shift=position-pivot
                    direction = -(shift - rot_matrix @ shift)
                    if i<steps-1:
                        self.children[idx-1].move(direction =direction, begin_time=t0+i*ddt, transition_time=0.9*ddt)
                    else:
                        # enforce into the correct position
                        self.children[idx-1].move_to(target_location = self.physical_locs[target],begin_time=t0+i*ddt,transition_time=ddt,verbose=False)
                    old_rotation = new_rotation
                    position += direction
                self.cubie_states[idx]=(target,new_rotation)
                # print(idx," arrival: ",position," at target ",target," pos: ",self.physical_locs[target])
            t0  = t0+dt
            [self.children[idx - 1].change_emission(from_value=5, to_value=0,slots=list(range(1,7)), begin_time=t0-dt/2, transition_time=dt/2) for idx
             in indices]

        # make action curves linear to avoid distortions
        for child in self.children:
            ibpy.set_linear_fcurves(child)
        return begin_time+transition_time


    def invert(self, word, steps = 1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.transform(word.swapcase()[::-1],steps=steps,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time


