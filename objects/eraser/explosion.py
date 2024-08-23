import numpy as np

from interface import ibpy
from objects.bobject import BObject
from objects.eraser.fields import Wind, Turbulence
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Explosion(BObject):
    """
    all objects are equipped with a particle system modifer
    and with an explode modifer

    """
    def __init__(self,remove_bobs):
        self.bobs = remove_bobs
        self.psms={} # dictionary for the particle system modifiers
        self.ems={} # dictionary for the explode modifiers
        for bob in remove_bobs:
            obj = bob.ref_obj
            mesh=False
            if 'hand_writing' in obj.name: # do not convert hand_writing into mesh and let it explode since it is not visible anyways
                pass
            else:
                mesh=True
            if mesh:
                modifier=ibpy.add_modifier(bob,'PARTICLE_SYSTEM')
                self.set_attribute(modifier,'render_type','NONE')
                # self.set_attribute(modifier,'effector_weights.gravity',0.1)
                self.psms[bob]=modifier
                modifier=ibpy.add_modifier(bob,'EXPLODE')
                self.set_attribute(modifier,'show_dead',False)
                self.ems[bob]=modifier
        self.set_wind_and_turbulence(wind_location=[6, 0, -5],turbulence_location=[0,0,0], rotation_euler=[0, -np.pi / 4, 0],
                                          turbulence_strength=10)

    def set_attribute(self,modifier,attribute,value):
        if isinstance(modifier, list):
            for mod in modifier:
                setattr(mod,attribute,value)
        else:
            setattr(modifier,attribute,value)

    def set_wind_and_turbulence(self,wind_location=[0,0,0],turbulence_location=[0,0,0],rotation_euler=[0,0,0],wind_strength=5,turbulence_strength=3):
        wind = Wind(location=wind_location,rotation_euler=rotation_euler,strength=wind_strength)
        if turbulence_strength>0:
            turbulence = Turbulence(location=turbulence_location,strength=turbulence_strength)

    def explode(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,modus='x_from_positive'):
        if modus=='x_from_positive':
            self.bobs.sort(key=lambda x:(x.ref_obj.matrix_world@x.ref_obj.location)[0],reverse=True)
        dt = transition_time/len(self.bobs)
        for i,bob in enumerate(self.bobs):
            psm = self.psms[bob]
            begin_frame = int((begin_time + i * dt) * FRAME_RATE)
            self.set_attribute(psm,'frame_start', begin_frame)
            self.set_attribute(psm,'count', 100)
            self.set_attribute(psm,'frame_end',begin_frame+10)
            self.set_attribute(psm,'lifetime',int(transition_time*FRAME_RATE))