import bmesh
import mathutils
import numpy as np
from mathutils import Vector

from interface import ibpy
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Curtain(BObject):
    def __init__(self,**kwargs):
        self.kwargs=kwargs
        # set default values
        self.name=self.get_from_kwargs('name','Curtain')
        rotation=self.get_from_kwargs('rotation_euler',[np.pi/2,0,0])
        self.color = self.get_from_kwargs('color','important')
        self.scale=self.get_from_kwargs('scale',[1,1,1])
        self.location=self.get_from_kwargs('location',[0,0,0])
        self.simulation_start=self.get_from_kwargs('simulation_start',0)
        self.simulation_duration=self.get_from_kwargs('simulation_duration',DEFAULT_ANIMATION_TIME)

        number_of_hooks=self.get_from_kwargs('number_of_hooks',5)
        plane = ibpy.add_plane()

        # prepare mesh with small subdivision to set up the hooks
        bm = bmesh.new()  # Creates an empty BMesh
        bm.from_mesh(plane.data)  # Fills it in using the plane
        bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=number_of_hooks-1, use_grid_fill=True)
        # bmesh.ops.scale(bm, vec=self.scale, verts=bm.verts)
        # bmesh.ops.translate(bm, vec=self.location, verts=bm.verts)
        bm.to_mesh(plane.data)
        bm.free()

        self.plane=BObject(obj=plane,name=self.name+'_plane',color=self.color,**kwargs)

        #setup the hooks

        verts = plane.data.vertices

        # find top row, to identify vertices that get hooked
        y_max = -np.Infinity
        self.x_min = np.Infinity
        for v in verts:
            if v.co[1]>y_max:
                y_max=v.co[1]
            if v.co[0]<self.x_min:
                self.x_min=v.co[0]

        self.hooked_verts = []
        self.indices=[]
        for i in range(len(verts)):
            if verts[i].co[1] == y_max:
                self.hooked_verts.append(verts[i].co.copy())
                self.indices.append(i)

        self.hook_modifiers=[]
        self.hooks = []

        for i,v in zip(self.indices,self.hooked_verts):
            name = f"Hook_{i}"
            empty = ibpy.add_empty(location=v)
            empty.name = name
            b_empty=BObject(obj=empty)
            self.hooks.append(b_empty)
            hm = ibpy.add_modifier(self.plane,type='HOOK',name=name)
            hm.object = empty
            hm.vertex_indices_set([i])

            self.hook_modifiers.append(hm)

        # add more subdivisions
        bm = bmesh.new()  # Creates an empty BMesh
        bm.from_mesh(plane.data)  # Fills it in using the plane
        bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=5, use_grid_fill=True)
        bm.to_mesh(plane.data)
        bm.free()

        # create vertex group
        vertex_group=ibpy.add_vertex_group(self.plane,self.indices,name='CurtainPins')
        cm=ibpy.add_modifier(self.plane,'CLOTH',name='ClothModifier')
        cm.settings.vertex_group_mass=vertex_group.name
        # demin properties
        cm.settings.quality=12
        cm.settings.mass=1
        cm.settings.tension_stiffness=40
        cm.settings.compression_stiffness=40
        cm.settings.shear_stiffness=40
        cm.settings.bending_stiffness=10
        cm.settings.tension_damping=25
        cm.settings.compression_damping=25
        cm.settings.shear_damping=25
        cm.settings.bending_damping=0.5
        cm.point_cache.frame_start=int(self.simulation_start*FRAME_RATE)
        cm.point_cache.frame_end=int((self.simulation_start+self.simulation_duration)*FRAME_RATE)
        ibpy.add_sub_division_surface_modifier(self.plane,level=3)

        super().__init__(children=[self.plane, *self.hooks],
                         name=self.name,
                         location=self.location,scale=self.scale, rotation_euler=rotation,**kwargs)

    def appear(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        super().appear(begin_time=begin_time,transition_time=transition_time)
        self.plane.appear(begin_time=begin_time,transition_time=transition_time)

    def open(self,fraction=0.2, begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for vert,hook in zip(self.hooked_verts,self.hooks):
            # just move hook towards x=x_min
            loc = vert
            loc_new = loc.copy()
            loc_new[0]=fraction*(loc_new[0]-self.x_min)+self.x_min
            shift = loc_new - loc
            hook.move(direction=shift, begin_time=begin_time, transition_time=transition_time)

    def close(self, begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for vert,hook in zip(self.hooked_verts,self.hooks):
            # just move hook towards x=x_min
            loc = vert
            hook.move_to(target_location=loc, begin_time=begin_time, transition_time=transition_time)
