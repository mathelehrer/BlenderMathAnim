import bmesh
import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from objects.geometry.geo_bobject import GeoBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import get_rotation_quaternion_from_start_and_end, get_save_length, to_vector




class ArrowChain(BObject):
    """
    Arrows are linked by vertex parenting, which attaches them
    tip to tail but lets them rotate freely

    example:
    arrow_chain = ArrowChain(number=3,lengths=[3,2,1],keep_proportions=True)
    t0=0.5+arrow_chain.grow(begin_time=t0,transition_time=1)
    t0 = 0.5 + arrow_chain.rotate(rotation_euler=[0,2 * np.pi, 0], begin_time=t0, transition_time=5)
    """

    def __init__(self,number=1,lengths=[1],periods = [1],**kwargs):
        self.kwargs = kwargs
        self.thickness= self.get_from_kwargs('thickness',1)
        self.name = self.get_from_kwargs('name','ArrowChain')
        self.periods = periods
        self.phases = self.get_from_kwargs('phases',[])
        self.colors = self.get_from_kwargs('colors',['plastic_drawing'])
        self.emission=self.get_from_kwargs('emission',0)
        self.rotation =to_vector(self.get_from_kwargs('rotation_euler',[0,0,0]))
        keep_proportions = self.get_from_kwargs('keep_proportions',False)
        # self.emission = self.get_from_kwargs('emission',0)
        self.arrows = []
        for i in range(number):
            if i == 0:
                name=self.name
            else:
                name='ChainMember'+str(i)
            if len(lengths)>i:
                length=lengths[i]
            else:
                length=lengths[-1]
            if length<0:
                self.phases.append(np.pi)
                length=-length
            else:
                self.phases.append(0)
            if keep_proportions:
                scale = [0.5**(1/(i+1))*length,0.5**(1/(i+1))*length,length]
            else:
                scale = [1,1,length]

            addons = {}
            if len(self.colors)>=number:
                color = self.colors[i]
            elif len(self.colors)==3:
                if self.periods[i]<0:
                    color=self.colors[0]
                elif self.periods[i]>0:
                    color=self.colors[2]
                else:
                    color= self.colors[1]
            else:
                color=  self.colors[-1]
            if color=='hue':
                addons = {'hue_value':self.phases[i]/2/np.pi}
            if i==0:
                color_first = color
            arrow = BObject.from_file("ArrowLinkable", color=color,name=name,
                                      scale=scale,bevel=0.1,emission=self.emission,
                                      rotation_euler=self.rotation,**addons) # it's not possible to transfer **kwargs, because the locations would be applied twice
            # the linking is performed by vertex parenting
            if len(self.arrows)>0:
                ibpy.set_vertex_parent(self.arrows[-1],arrow,65) # 65 is the index of the vertex on the top for this particular mesh-object
            self.arrows.append(arrow)

        super().__init__(obj=self.arrows[0].ref_obj,color=color_first,name=self.name,emission=self.emission,**kwargs)



    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        [arrow.rotate(rotation_euler=self.rotation+Vector([0,phase,0]),begin_time=0,transition_time=0) for arrow,phase in zip(self.arrows,self.phases) if phase!=0]
        [arrow.grow(begin_time=begin_time, transition_time=transition_time) for arrow in self.arrows]
        # make earlier arrows fainter
        [arrow.change_alpha(alpha=0.5+0.5*(i+1)/len(self.arrows),begin_time=1/FRAME_RATE,transition_time=0) for i,arrow in enumerate(self.arrows)]

        return begin_time+transition_time

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        [arrow.disappear(alpha=alpha,begin_time=begin_time,transition_time=transition_time) for arrow in self.arrows]
        return begin_time+transition_time

    def draw(self,angle = 2*np.pi,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for arrow,period,phase in zip(self.arrows,self.periods,self.phases):
            arrow.rotate(rotation_euler=self.rotation+Vector([0,angle*period+phase,0]),begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def get_arrows(self):
        return self.arrows