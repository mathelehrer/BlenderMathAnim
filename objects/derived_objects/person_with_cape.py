import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME


class PersonWithCape(BObject):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['gray_8','important'])
        name=self.get_from_kwargs('name','PersonWithCape')
        self.simulation_start = self.get_from_kwargs('simulation_start', 0)
        self.simulation_duration = self.get_from_kwargs('simulation_duration', DEFAULT_ANIMATION_TIME)

        bobs = BObject.from_file("PersonWithCape", objects=["Person", "Cape"],colors=colors,name=name)
        self.children=[]
        if len(bobs)>0:
            self.person = bobs[0]
            self.person.ref_obj.location=[0,0,1.5]
            self.children.append(self.person)
            if len(bobs)>1:
                self.cape = bobs[1]
                self.cape.ref_obj.location=[0,-0.52,1.59]
                self.children.append(self.cape)

        verts = self.cape.ref_obj.data.vertices

        # find neck vertices, to identify vertices that get hooked
        # these are basically the once which are closest to the center of the cut out circle (-0.03,-0.52)
        center = Vector([0.03,0.52,0])
        min = np.Infinity
        for v in verts:
            if (v.co-center).length<min:
                min=(v.co-center).length

        print(min)

        self.hooked_verts = []
        self.indices = []
        for i in range(len(verts)):
            if (verts[i].co-center).length<=1.05*min:
                self.hooked_verts.append(verts[i])
                self.indices.append(i)

        vertex_group = ibpy.add_vertex_group(self.cape, self.indices, name='PinnedVertices')
        cm = ibpy.add_modifier(self.cape, 'CLOTH', name=name+'_ClothModifier')
        cm.settings.vertex_group_mass = vertex_group.name
        # demin properties
        cm.settings.quality = 12
        cm.settings.mass = 1
        cm.settings.tension_stiffness = 40
        cm.settings.compression_stiffness = 40
        cm.settings.shear_stiffness = 40
        cm.settings.bending_stiffness = 10
        cm.settings.tension_damping = 25
        cm.settings.compression_damping = 25
        cm.settings.shear_damping = 25
        cm.settings.bending_damping = 0.5
        cm.point_cache.frame_start = int(self.simulation_start * FRAME_RATE)
        cm.point_cache.frame_end = int((self.simulation_start + self.simulation_duration) * FRAME_RATE)
        cm.collision_settings.use_self_collision=True
        ibpy.add_sub_division_surface_modifier(self.cape, level=2)

        ibpy.add_modifier(self.person,'COLLISION',name='CollisionModifier')
        super().__init__(children=self.children,name=name,rotation_euler=rotation,location=location)

    def appear(self,
               begin_time=0,
               transition_time=OBJECT_APPEARANCE_TIME,
               **kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        for child in self.children:
            child.appear(begin_time=begin_time,transition_time=transition_time)


