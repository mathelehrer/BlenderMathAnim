from math import pi
import random

from camb.model import transfer_names
from mathutils import Vector, Quaternion, Euler
from sympy.core.benchmarks.bench_arit import timeit_Add_x05

from geometry_nodes.nodes import RotateInstances, layout, TranslateInstances, RealizeInstances, GeometryToInstance
from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier, change_default_boolean, change_default_value, \
    change_default_quaternion, change_default_vector, change_default_rotation
from mathematics.mathematica.mathematica import tuples
from new_stuff.geometry_nodes_modifier import RubiksCubeUnfolded, RubiksCubeModifier, RubiksSphereModifier, \
    PermutationModifier
from objects.bobject import BObject
from objects.cube import Cube
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.kwargs import get_from_kwargs


class Permutation(BObject):
    def __init__(self,n=8,number_of_cycles=1,**kwargs):
        """
        visual representation of a permutation for at most nine elements
        """
        cube = Cube()

        self.perm_mod= PermutationModifier(name="PermutationModifier",n=n,
                                           number_of_cycles=number_of_cycles, **kwargs)
        cube.add_mesh_modifier(type="NODES",node_modifier=self.perm_mod)
        super().__init__(obj=cube.ref_obj, name="Permutation", **kwargs)
        self.cycle_count = 0

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def apply_cycle(self,cycle_number,up=1,down=1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        '''
        the cycle number 1376 corresponds to a cycle (1 3 7 6)
        the digits from 1 to 9 are encoded with prime numbers
        2, 3, 5, 7, 11, 13, 17, 19, 23
        you can choose, which digits are cycled above and which below the cycle.
        It has to be decided for each cycle separately.
        For this cycle, 3 and 7 can cycle above, where as 1 and 6 should cycle below, to have an
        intersection free animation
        Therefore down=2*13=26 and up = 5*17=85
        if both down and up are equal to 1, the numbers will be alternatingly cycled up and down.

        '''
        prime_dict = {1:2,2:3,3:5,4:7,5:11,6:13,7:17,8:19,9:23}
        digits = []
        self.cycle_count += 1
        self.cycle_number = cycle_number

        cycle_length = len(str(cycle_number))
        for i in range(cycle_length):
            digits.insert(0,cycle_number%10)
            cycle_number//=10

        if up == 1 and down == 1:
            for i,d in enumerate(digits):
                if i%2 == 0:
                    up=up*prime_dict[d]
                else:
                    down=down*prime_dict[d]

        # get the relevant nodes from the modifier
        cycle_node = ibpy.get_geometry_node_from_modifier(self.perm_mod,label="Cycle"+str(self.cycle_count))
        cycle_node.integer=self.cycle_number

        cycle_length_node = ibpy.get_geometry_node_from_modifier(self.perm_mod,label="CycleLength"+str(self.cycle_count))
        cycle_length_node.integer=cycle_length

        up_node = ibpy.get_geometry_node_from_modifier(self.perm_mod,label="UpMover"+str(self.cycle_count))
        if isinstance(up,list):
            final_up = 1
            for u in up:
                final_up*=prime_dict[u]
        else:
            final_up=up
        up_node.integer=final_up

        down_node = ibpy.get_geometry_node_from_modifier(self.perm_mod,label="DownMover"+str(self.cycle_count))
        if isinstance(down, list):
            final_down = 1
            for d in down:
                final_down *= prime_dict[d]
        else:
            final_down = down
        down_node.integer=final_down

        # split the transition_time into pieces
        # 10 percent up
        # 10 percent down
        # 20 percent breaks
        # 60 percent cycle

        t0 = begin_time
        dt = transition_time/10

        # displace cycling digits
        displace_node = ibpy.get_geometry_node_from_modifier(self.perm_mod,label="Displacement"+str(self.cycle_count))

        t0 = dt+ibpy.change_default_value(displace_node,from_value=0,to_value=1,begin_time=t0,transition_time=dt)
        dt2 = dt*6/len(digits)
        first = digits[0]
        digits.append(first)
        digits=digits[1:]
        for i in range(cycle_length):
            mover_node = ibpy.get_geometry_node_from_modifier(self.perm_mod,label="Mover"+str(self.cycle_count)+"_"+str(i+1))
            t0 = ibpy.change_default_value(mover_node,from_value=0,to_value=1,begin_time=t0,transition_time=dt2)
        t0+=dt

        ibpy.change_default_value(displace_node,from_value=1,to_value=0,begin_time=t0,transition_time=dt)
        return begin_time+transition_time
