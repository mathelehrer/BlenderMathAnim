import numpy as np
from mathutils import Vector, Quaternion, Euler

from interface import ibpy
from interface.ibpy import FOLLOW_PATH_DICTIONARY, Vector, Quaternion, Euler
from objects.bobject import BObject
from objects.curve import BezierDataCurve
from utils.constants import OBJECT_APPEARANCE_TIME, DEFAULT_ANIMATION_TIME
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
            self.cubie_states[i + 1]=(i, Quaternion([1, 0, 0, 0]))
            self.location[i+1]=i+1

        if len(bobs) > 0:
           self.children = bobs

        self.locs = [
            [0,0,4],[0,2,4],[0,4,4],[2,0,4],[2,2,4],[2,4,4],[4,0,4],[4,2,4],[4,4,4],
            [0,0,2],[0,2,2],[0,4,2],[2,0,2],[2,2,2],[2,4,2],[4,0,2],[4,2,2],[4,4,2],
            [0,0,0],[0,2,0],[0,4,0],[2,0,0],[2,2,0],[2,4,0],[4,0,0],[4,2,0],[4,4,0]
        ]

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



        for i,(child,loc) in enumerate(zip(self.children,self.locs)):
            child.ref_obj.location=loc
            if i<9:
                child.ref_obj.name="cubie0"+str(i+1)
            else:
                child.ref_obj.name="cubie"+str(i+1)

        self.curve_map={}
        self.create_displacement_curves()

        super().__init__(children=self.children+list(self.curve_map.values()), name=name, rotation_quaternion=Euler(rotation).to_quaternion(), location=location)

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

    def get_corners(self):
        return self.corners

    def get_edges(self):
        return self.edges

    def get_corner_curves(self):
        return [self.curve_map[idx] for idx in self.corners]

    def get_edge_curves(self):
        return [self.curve_map[idx] for idx in self.edges]

    def get_permutation(self):
        """
        returns the permutation of the cube. Since the cube can be disassembled it might not be an element of the Rubik's cube group.
        """


    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):

        dt = 0.75*transition_time
        shift = 0.25*transition_time/9

        for i,child in enumerate(self.children):
            child.scale(initial_scale=0,final_scale=1,begin_time=begin_time+shift*i,transition_time=dt)
        return super().appear(begin_time=begin_time, transition_time=transition_time,children=False)

    def get_cubie_location(self,idx):
        return to_vector(self.locs[idx-1])

    def get_center(self):
        return to_vector(self.locs[13])

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
        curves = [self.curve_map[idx],self.curve_map[pos]]
        old_quaternion = self.disassemble_rotation[idx].copy()
        # old_quaternion.invert()
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
        Move cubie from one position of the rubik's cube to another. It is assumed that the cubie is located at the rubik's cube
        """
        dt = transition_time/len(target_positions)
        t0 = begin_time
        # find cubie
        idx = -1
        for i,val in self.cubie_states.items():
            if val[0]==src_pos:
                idx = i
                break

        if idx != -1:
            for pos in target_positions:
                curves = [self.curve_map[src_pos],self.curve_map[pos]]
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

                new_quaternion = self.disassemble_rotation[pos].copy()
                new_quaternion.invert()
                t0 = self.children[idx - 1].rotate(rotation_quaternion=new_quaternion, begin_time=t0, transition_time=dt)
                self.cubie_states[idx]=(pos,new_quaternion)
        return begin_time + transition_time