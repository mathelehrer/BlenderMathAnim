import bmesh
import numpy as np
import bpy
from mathutils import Vector, Quaternion

from interface import ibpy
from objects.bobject import BObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.utils import quaternion_from_normal, to_vector


class Rope(BObject):
    """
    Create a rope that can be fixed at both ends:

    """

    def __init__(self, length=1,thickness=1,start=Vector(),end=Vector(),folded=True, folding_direction =Vector([1,0,0]),resolution=10, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'Rope')
        self.length=length
        if isinstance(start,list):
            self.start=Vector(start)
        else:
            self.start=start

        if isinstance(end,list):
            self.end=Vector(end)
        else:
            self.end=end
        self.middle = 0.5*(self.start+self.end)

        distance = self.end-self.start
        d = distance.length
        if (length**2-d**2)>0:
            h = 1/2*np.sqrt(length**2-d**2)
        else:
            h=0

        folding_direction = to_vector(folding_direction).normalized()
        self.middle = self.middle+float(h)*folding_direction

        self.thickness = thickness*0.01
        self.resolution = resolution

        mesh = self.create_mesh()
        new_object = bpy.data.objects.new(self.name, mesh)

        super().__init__(obj=new_object, name=self.name,**kwargs)
        # ibpy.add_solidify_modifier(self,thickness=0.001,offset=0)
        self.hook_modifiers = []
        self.hooks = []

        names = [self.name+'_Hook_start', self.name+'_Hook_end']

        vertex_groups=[ibpy.create_vertex_group(self, [0], 'start_group'),
                       ibpy.create_vertex_group(self, [self.v_count], 'end_group')
                       ]

        pin_group = ibpy.create_vertex_group(self,[0,self.v_count],'pin_group')

        for i,name in enumerate(names):
            empty = ibpy.add_empty(location = self.hook_locations[i])
            empty.name = name
            b_empty = BObject(obj=empty)
            self.hooks.append(b_empty)
            empty.parent=self.ref_obj
            hm = ibpy.add_modifier(self, type='HOOK', name=name)
            hm.object = empty
            hm.vertex_group=vertex_groups[i].name
            self.hook_modifiers.append(hm)

        #ibpy.add_sub_division_surface_modifier(self, level=2)

        # add cloth modifier
        self.cm = ibpy.add_modifier(self,'CLOTH',name='ClothModifier')
        # leather properties
        self.cm.settings.quality = 15
        self.cm.settings.mass = 0.4
        self.cm.settings.tension_stiffness = 80
        self.cm.settings.compression_stiffness = 80
        self.cm.settings.shear_stiffness = 80
        self.cm.settings.bending_stiffness = 150
        self.cm.settings.tension_damping = 25
        self.cm.settings.compression_damping = 25
        self.cm.settings.shear_damping = 25
        self.cm.settings.bending_damping = 0.5
        # self collisions
        self.cm.collision_settings.use_self_collision=True

        # pinning
        self.cm.settings.vertex_group_mass=pin_group.name

        ibpy.link(self)
        ibpy.set_active(self)
        ibpy.set_edit_mode()
        bpy.ops.mesh.extrude_edges_move(MESH_OT_extrude_edges_indiv={"use_normal_flip": False, "mirror": False},
                                        TRANSFORM_OT_translate={"value": (0, 0, 0.05), "orient_axis_ortho": 'X',
                                                                "orient_type": 'GLOBAL',
                                                                "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                                                "orient_matrix_type": 'GLOBAL',
                                                                "constraint_axis": (False, False, True),
                                                                "mirror": False, "use_proportional_edit": False,
                                                                "proportional_edit_falloff": 'SMOOTH',
                                                                "proportional_size": 1,
                                                                "use_proportional_connected": False,
                                                                "use_proportional_projected": False, "snap": False,
                                                                "snap_target": 'CLOSEST', "snap_point": (0, 0, 0),
                                                                "snap_align": False, "snap_normal": (0, 0, 0),
                                                                "gpencil_strokes": False, "cursor_transform": False,
                                                                "texture_space": False, "remove_on_cancel": False,
                                                                "view2d_edge_pan": False, "release_confirm": False,
                                                                "use_accurate": False,
                                                                "use_automerge_and_split": False})
        ibpy.set_object_mode()

    def set_dynamic(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        self.cm.point_cache.frame_start = int(begin_time * FRAME_RATE)
        self.cm.point_cache.frame_end = int((begin_time+transition_time) * FRAME_RATE)

    def attach_to(self,start_bob,end_bob):
        if start_bob:
            self.hooks[0].ref_obj.parent=start_bob.ref_obj
            self.hooks[0].ref_obj.location=[0]*3 # remove intrinsic location
        if end_bob:
            self.hooks[1].ref_obj.parent=end_bob.ref_obj
            self.hooks[1].ref_obj.location = [0] * 3

    def create_mesh(self):
        # make mesh
        n = self.resolution*self.length
        vertices = []
        edges = []
        faces = []

        one = None
        self.v_count = -1
        e_count = -1
        dir = self.middle-self.start
        self.hook_locations=[]

        for i in range(0,int(n/2)):
            vec1 = self.start + 2 * i / n * dir +Vector([0, 0, -self.thickness / 2])
            vertices.append(vec1)
            self.v_count+=1
            if one:
                edges.append([self.v_count,self.v_count-1])
                e_count+=1
            else:
                self.hook_locations.append(vec1) # hook location for the start of the rope
            one=vec1

        dir = self.end - self.middle
        for i in range(0,int(n/2+1)):
            vec1 = self.middle + 2 * i / n * dir + Vector([0, 0,- self.thickness / 2])
            vertices.append(vec1)
            self.v_count += 1
            edges.append([self.v_count, self.v_count - 1])
            e_count += 1
            one = vec1

        self.hook_locations.append(one) # hook location for the end of the rope
        new_mesh = bpy.data.meshes.new(self.name+'_mesh')
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()

        return new_mesh
