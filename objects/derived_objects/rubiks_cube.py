import numpy as np
from mathutils import Vector, Quaternion

from interface import ibpy
from interface.ibpy import FOLLOW_PATH_DICTIONARY
from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME
from utils.utils import flatten

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
        colors = self.get_from_kwargs('colors', []) #["text","red","green","blue","orange","yellow"]
        name = self.get_from_kwargs('name', "RubiksCube3x3")

        cubies =["Cube0"+str(i) for i in range(1,10)]+["Cube"+str(i) for i in range(10,28)]
        bobs = BObject.from_file("RubiksCube3x3LocalCenters", objects=cubies,
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

        self.locs = [
            [0,0,4],[0,2,4],[0,4,4],[2,0,4],[2,2,4],[2,4,4],[4,0,4],[4,2,4],[4,4,4],
            [0,0,2],[0,2,2],[0,4,2],[2,0,2],[2,2,2],[2,4,2],[4,0,2],[4,2,2],[4,4,2],
            [0,0,0],[0,2,0],[0,4,0],[2,0,0],[2,2,0],[2,4,0],[4,0,0],[4,2,0],[4,4,0]
        ]

        for i,(child,loc) in enumerate(zip(self.children,self.locs)):
            child.ref_obj.location=loc
            if i<9:
                child.ref_obj.name="cubie0"+str(i+1)
            else:
                child.ref_obj.name="cubie"+str(i+1)

        super().__init__(children=self.children, name=name, rotation_euler=rotation, location=location)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):

        dt = 0.75*transition_time
        shift = 0.25*transition_time/9

        t0 = begin_time
        for i,child in enumerate(self.children):
            child.scale(initial_scale=0,final_scale=1,begin_time=begin_time+shift*i,transition_time=dt)
        return super().appear(begin_time=begin_time, transition_time=transition_time,children=False)

    def get_cubie_location(self,idx):
        return self.locs[idx-1]

    def align_cubie_with_curve_and_rotate(self, idx, curve,rotation,reverse=False, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.children[idx-1].ref_obj.location = Vector()
        if not (self.children[idx-1],curve) in FOLLOW_PATH_DICTIONARY.keys():
            ibpy.set_follow(self.children[idx-1], curve,use_curve_follow=False)
        if reverse:
            ibpy.follow(self.children[idx-1], curve,initial_value=1,final_value=0, begin_time=begin_time, transition_time=transition_time)
        else:
            ibpy.follow(self.children[idx-1], curve,initial_value=0,final_value=1, begin_time=begin_time, transition_time=transition_time)
        if reverse:
            self.children[idx-1].rotate(rotation_euler=rotation,begin_time=begin_time,transition_time=transition_time)
        else:
            self.children[idx-1].rotate(rotation_euler=[0,0,0],begin_time=begin_time,transition_time=transition_time)
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
            t0 = pause + self.children[idx - 1].rotate(rotation_euler=rotations[i],begin_time=t0,transition_time=dt)

        return begin_time+transition_time

