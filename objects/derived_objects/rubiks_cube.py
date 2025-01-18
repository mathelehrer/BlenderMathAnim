import numpy as np
from mathutils import Vector, Quaternion, Euler

from interface import ibpy
from interface.ibpy import FOLLOW_PATH_DICTIONARY, Vector, Quaternion, Euler
from objects.bobject import BObject
from objects.curve import BezierDataCurve
from objects.empties import EmptyArrow
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.string_utils import remove_digits, remove_punctuation
from utils.utils import flatten, to_vector, pi

r2 = np.sqrt(2)
class BRubiksCube(BObject):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        location = self.get_from_kwargs('location', [0, 0, 0])
        rotation = self.get_from_kwargs('rotation_euler', [0, 0, 0])
        colors = self.get_from_kwargs('colors', []) #["text","red","green","blue","orange","yellow"]
        name = self.get_from_kwargs('name', "RubiksCube3x3")

        cubies =["Cube0"+str(i) for i in range(1,10)]+["Cube"+str(i) for i in range(10,28)]
        bobs = BObject.from_file("RubiksCube3x3", objects=cubies,
                                 colors=colors, name=name,smooth=3)
        self.children = []
        self.keys = []
        self.key_labels = {}

        self.state = {}

        # the cubies are labelled from the back to the front and from top to bottom
        # top level
        # 1 2 3
        # 4 5 6
        # 7 8 9
        # middle level
        # 10 11 12
        # 13 14 15
        # 16 17 18
        # bottom level
        # 19 20 21
        # 22 23 24
        # 25 26 27

        # the cube 1 is corner of the top, left and back face
        # the cube 27 is corner of the bottom, front and right face

        for i in range(27):
            self.state[i+1]=(i,Quaternion([1,0,0,0]))
        if len(bobs) > 0:
           self.children = bobs
        super().__init__(children=self.children, name=name, rotation_euler=rotation, location=location)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        return super().appear(begin_time=begin_time, transition_time=transition_time,children=True)

    def lp(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the left face
        the following permutations are performed
        1->19
        10->22
        19->25
        22->16
        25->7
        16->4
        7->1
        4->10
        """
        source_state_keys = [1,10,19,22,25,16,7,4,13]
        target_state_keys = [19,22,25,16,7,4,1,10,13]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0],self.state[key][1].copy()])
        for source,target_key in zip(sources,target_state_keys):
            rotation = Quaternion([1/r2,0,-1/r2,0])@source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation,begin_time=begin_time,transition_time=transition_time)
            self.state[target_key]=(idx,rotation)
        return begin_time+transition_time

    def lm(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the left face
        the following permutations are performed
        19->11
        22->10
        25->19
        16->22
        7->25
        4->16
        1->7
        10->4
        """
        target_state_keys = [1, 10, 19, 22, 25, 16, 7, 4, 13]
        source_state_keys = [19, 22, 25, 16, 7, 4, 1, 10, 13]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0],self.state[key][1].copy()])
        for source,target_key in zip(sources,target_state_keys):
            rotation = Quaternion([1/r2,0,1/r2,0])@source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation,begin_time=begin_time,transition_time=transition_time)
            self.state[target_key]=(idx,rotation)
        return begin_time+transition_time

    def fp(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the front face
        7->25
        16->26
        25->27
        26->18
        27->9
        18->8
        9->7
        8->16
        17->17

        """
        source_state_keys = [7,16,25,26,27,18,9,8,17]
        target_state_keys = [25,26,27,18,9,8,7,16,17]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2,  1 / r2,0, 0]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def fm(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the front face
        7<-25
        16<-26
        25<-27
        26<-18
        27<-9
        18<-8
        9<-7
        8<-16
        17<-17

        """
        source_state_keys = [25, 26, 27, 18, 9, 8, 7, 16, 17]
        target_state_keys = [7, 16, 25, 26, 27, 18, 9, 8, 17]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, -1 / r2, 0, 0]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def rp(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the right face
        the following permutations are performed
        9->27
        18->24
        27->21
        24->12
        21->3
        12->6
        3->9
        6->18
        15->15
        """
        source_state_keys = [9,18,27,24,21,12,3,6,15]
        target_state_keys = [27,24,21,12,3,6,9,18,15]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, 0, 1 / r2, 0]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def rm(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the right face
        the following permutations are performed
        9<-27
        18<-24
        27<-21
        24<-12
        21<-3
        12<-6
        3<-9
        6<-18
        15<-15
        """
        target_state_keys = [9,18,27,24,21,12,3,6,15]
        source_state_keys = [27,24,21,12,3,6,9,18,15]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, 0, -1 / r2, 0]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time


    def tp(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the top face
        the following permutations are performed
        1->7
        4->8
        7->9
        8->6
        9->3
        6->2
        3->1
        2->4
        5->5
        """
        source_state_keys = [1,4,7,8,9,6,3,2,5]
        target_state_keys = [7,8,9,6,3,2,1,4,5]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, 0, 0,1 / r2]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def tm(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the top face
        the following permutations are performed
        1<-7
        4<-8
        7<-9
        8<-6
        9<-3
        6<-2
        3<-1
        2<-4
        5<-5
        """
        target_state_keys = [1,4,7,8,9,6,3,2,5]
        source_state_keys = [7,8,9,6,3,2,1,4,5]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, 0, 0,-1 / r2]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def dp(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the bottom face
        the following permutations are performed
        25->19
        22->20
        19->21
        20->24
        21->27
        24->26
        27->25
        26->22
        23->23
        """
        source_state_keys = [25,22,19,20,21,24,27,26,23]
        target_state_keys = [19,20,21,24,27,26,25,22,23]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, 0, 0, -1 / r2]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def dm(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the bottom face
        the following permutations are performed
        25<-19
        22<-20
        19<-21
        20<-24
        21<-27
        24<-26
        27<-25
        26<-22
        23<-23
        """
        target_state_keys = [25,22,19,20,21,24,27,26,23]
        source_state_keys = [19,20,21,24,27,26,25,22,23]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, 0, 0, 1 / r2]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def bp(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the back face
        3->21
        12->20
        21->19
        20->10
        19->1
        10->2
        1->3
        2->12
        11->11

        """
        source_state_keys = [3,12,21,20,19,10,1,2,11]
        target_state_keys = [21,20,19,10,1,2,3,12,11]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, -1 / r2, 0, 0]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

    def bm(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        rotation of the back face
        3<-21
        12<-20
        21<-19
        20<-10
        19<-1
        10<-2
        1<-3
        2<-12
        11<-11

        """
        target_state_keys = [3,12,21,20,19,10,1,2,11]
        source_state_keys = [21,20,19,10,1,2,3,12,11]
        sources = []
        for key in source_state_keys:
            sources.append([self.state[key][0], self.state[key][1].copy()])
        for source, target_key in zip(sources, target_state_keys):
            rotation = Quaternion([1 / r2, 1 / r2, 0, 0]) @ source[1]
            idx = source[0]
            self.children[idx].rotate(rotation_quaternion=rotation, begin_time=begin_time,
                                      transition_time=transition_time)
            self.state[target_key] = (idx, rotation)
        return begin_time + transition_time

ex=Vector([1,0,0])
ey=Vector([0,1,0])
ez=Vector([0,0,1])

class BRubiksCubeLocalCenters(BObject):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        location = self.get_from_kwargs('location', [0, 0, 0])
        rotation = self.get_from_kwargs('rotation_euler', [0, 0, 0])
        # necessary to load default colors
        colors = self.get_from_kwargs('colors', []) #["text","red","green","blue","orange","yellow"]
        name = self.get_from_kwargs('name', "RubiksCube3x3")
        emission = self.get_from_kwargs('emission', 0.1)
        cubies =["Cube0"+str(i) for i in range(1,10)]+["Cube"+str(i) for i in range(10,28)]
        # load cube with default colors
        bobs = BObject.from_file("RubiksCube3x3LocalCenters", objects=cubies,colors=[],
                                 name=name,smooth=3)

        #override default colors
        if colors:
            materials =[]
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
                        slot.material = color_dict[mat_name]

        self.children = []
        self.keys = []
        self.key_labels = {}

        # capture the rotation state of each cubie
        self.cubie_states = {}
        # capture the location of each cubie
        self.location = {}
        # the cubies are labelled from the back to the front and from top to bottom
        # top level
        # 1 2 3
        # 4 5 6
        # 7 8 9
        # middle level
        # 10 11 12
        # 13 14 15
        # 16 17 18
        # bottom level
        # 19 20 21
        # 22 23 24
        # 25 26 27

        # the cube 1 is corner of the top, left and back face
        # the cube 27 is corner of the bottom, front and right face
        self.corners = [3, 1, 21, 19, 25, 27, 7, 9]
        self.edges = [12, 2, 20, 10, 4, 22, 24, 6, 16, 26, 8, 18]

        for i in range(27):
            # initially, each cubie is at its default position in its default rotation
            self.cubie_states[i + 1]=(i+1, None)
            self.location[i+1]=i+1

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

        self.cubie_face_map = {}
        for idx in range(1, 28):
            self.cubie_face_map[idx] = []

        for face_id,dict  in self.facelets.items():
            self.cubie_face_map[dict["parent"]].append(face_id)

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
