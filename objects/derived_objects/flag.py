import numpy as np

from interface import ibpy
from objects.bobject import BObject
from tools.images import ImageCreator
from utils.constants import OBJECT_APPEARANCE_TIME, FRAME_RATE, DEFAULT_ANIMATION_TIME


class Flag(BObject):
    def __init__(self,**kwargs):
        self.text_count=0
        self.kwargs = kwargs
        location = self.get_from_kwargs('location',[0,0,0])
        rotation = self.get_from_kwargs('rotation_euler',[0,0,0])
        colors = self.get_from_kwargs('colors',['drawing'])
        name=self.get_from_kwargs('name','Flag')
        mirror=self.get_from_kwargs('mirror',False)
        self.simulation_start = self.get_from_kwargs('simulation_start', 0)
        self.simulation_duration = self.get_from_kwargs('simulation_duration', DEFAULT_ANIMATION_TIME)

        # for the bwm video use Flag instead of Flag2
        bobs = BObject.from_file("Flag2", objects=["Post", "Cloth"],colors=colors,name=name)
        self.children=[]
        if len(bobs)>0:
            self.post = bobs[0]
            self.post.ref_obj.scale = [0.1,0.1,1]
            self.post.ref_obj.location=[0,0,0]
            self.children.append(self.post)
            if len(bobs)>1:
                self.cloth = bobs[1]
                self.cloth.ref_obj.scale = [0.965,0.52,1]
                self.cloth.ref_obj.location=[-1.0498,0,1.45]
                if not mirror:
                    self.cloth.ref_obj.rotation_euler=[np.pi/2,0,0]
                else:
                    self.cloth.ref_obj.rotation_euler=[np.pi/2,0,np.pi]
                self.children.append(self.cloth)

        verts = self.cloth.ref_obj.data.vertices

        # find top row, to identify vertices that get hooked
        y_max = -np.Infinity
        x_min = np.Infinity
        y_min = np.Infinity
        x_max = -np.Infinity
        for v in verts:
            if v.co[1] > y_max:
                y_max = v.co[1]
            if v.co[0] <x_min:
                x_min = v.co[0]
            if v.co[1]<y_min:
                y_min =v.co[1]
            if v.co[0]>x_max:
                x_max=v.co[0]
        print(x_min,x_max,y_min,y_max)

        self.hooked_verts = []
        self.indices = []
        for i in range(len(verts)):
            if verts[i].co[1] == y_max:
                self.hooked_verts.append(verts[i].co.copy())
                self.indices.append(i)
            if verts[i].co[1]==x_max:
                self.hooked_verts.append(verts[i].co.copy())
                self.indices.append(i)

        vertex_group = ibpy.add_vertex_group(self.cloth, self.indices, name='PinnedVertices')
        cm = ibpy.add_modifier(self.cloth, 'CLOTH', name=name+'_ClothModifier')
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
        ibpy.add_sub_division_surface_modifier(self.cloth, level=2)

        ibpy.setup_material_for_alpha_masking(self.cloth)

        super().__init__(children=self.children,name=name,rotation_euler=rotation,location=location,**kwargs)

    def appear(self,begin_time=0,transition_time=OBJECT_APPEARANCE_TIME,**kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time,**kwargs)
        for child in self.children:
            child.appear(begin_time=begin_time,transition_time=transition_time,**kwargs)

    def disappear(self,begin_time=0,transition_time=OBJECT_APPEARANCE_TIME,**kwargs):
        super().disappear(begin_time=begin_time,transition_time=transition_time,**kwargs)
        for child in self.children:
            child.disappear(begin_time=begin_time,transition_time=transition_time,**kwargs)

    def add_image_texture(self,image,frame,frame_duration=1):
        ibpy.add_image_texture(self.cloth,image,frame,frame_duration)

    def set_text(self,text,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        ic = ImageCreator(text, self.text_count, prefix=self.name)
        self.text_count += 1
        image = ic.get_image_path()
        self.add_image_texture(image, begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)