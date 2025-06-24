import bmesh
import bpy

from appearance.textures import apply_material
from interface import ibpy
from interface.ibpy import change_emission, Vector, Quaternion
from utils.constants import *
from utils.kwargs import get_from_kwargs
from utils.utils import to_vector


class BObject(object):
    """
    This is the base class for all objects that can be added to a blender scene and moved
    """
    def __init__(self,no_material=False, **kwargs):
        r"""
            this class can handle the following kwargs
            |
            |
            |
        :param no_material: no material is created, useful for objects created with geometry nodes or empties
        : Keyword Arguments:
            * name: name of the object
            * obj: an existing blender object (constructed in a subclass) can be connected to the base class
            * location: position of the object in the world
            * rotation_euler: rotation of the object
            * rotation_quaternion: rotation of the object in terms of quaternions
            * scale
            * objects: blender children of this object
            * mat
        """
        # register bpy object from sub class or create default object

        self.old_sk = None
        self.appeared = False
        self.transformation_state = 0
        self.kwargs = kwargs
        self.updaters = []
        self.label_sep = 1
        self.name = self.get_from_kwargs('name', 'b_object')
        self.hide = self.get_from_kwargs('hide',False) # per default each object is visible. It can be changed with toggle_hide
        if 'obj' in kwargs:
            ref_obj = kwargs['obj']
            if self.name != 'b_object':
                ref_obj.name = self.name
            kwargs.pop('obj')
        else:
            if 'mesh' in kwargs:
                mesh = kwargs['mesh']
                kwargs.pop('mesh')
                ref_obj = bpy.data.objects.new(self.name, mesh)
            else:
                ref_obj = bpy.data.objects.new(name=self.name, object_data=None)

        self.collection = self.get_from_kwargs('collection',None)
        self.intrinsic_scale = self.get_from_kwargs('scale', 1)
        if isinstance(self.intrinsic_scale, int) or isinstance(self.intrinsic_scale, float):
            self.intrinsic_scale = [self.intrinsic_scale] * 3

        ref_obj.scale = self.intrinsic_scale
        self.total_motion = Vector()
        # override location if it is explicitly given
        if 'location' in kwargs:
            location = self.get_from_kwargs('location', [0, 0, 0])
            ref_obj.location = location
        if 'rotation_euler' in kwargs:
            ref_obj.rotation_mode = 'XYZ'
            ref_obj.rotation_euler = self.get_from_kwargs('rotation_euler', [0, 0, 0])
        elif 'rotation_quaternion' in kwargs:
            ref_obj.rotation_mode = 'QUATERNION'
            ref_obj.rotation_quaternion = self.get_from_kwargs('rotation_quaternion', Quaternion())

        self.ref_obj = ref_obj

        # Blender objects with this b_object as the container
        self.b_children = self.get_from_kwargs('children', [])
        if len(self.b_children)>0:
            parenting=self.get_from_kwargs('parenting',True)
            if isinstance(parenting,list):
                for i,obj in enumerate(self.b_children):
                    if i in parenting:
                        ibpy.set_parent(obj,self)
            elif parenting:
                for obj in self.b_children:
                    ibpy.set_parent(obj,self)


        # set color
        if not no_material:
            self.color = self.get_from_kwargs('color',None)
            self.colors = self.get_from_kwargs('colors', None)


            if self.colors is not None:
                apply_material(self.ref_obj, self.color, colors=self.colors, **kwargs)
            else:
                apply_material(self.ref_obj, self.color,**kwargs)
            # do not automatically apply material to the children
            # [apply_material(child,self.color,**kwargs) for child in self.b_children]
            # for multiple slots


        smooth = self.get_from_kwargs('smooth', 0)
        bevel = self.get_from_kwargs('bevel', 0)
        solid = self.get_from_kwargs('solid', 0)
        if smooth > 1:
            adaptive_subdivision=self.get_from_kwargs('adaptive_subdivision',False)
            dicing_rate = self.get_from_kwargs('dicing_rate',0.5)
            ibpy.add_sub_division_surface_modifier(self, level=smooth,adaptive_subdivision=adaptive_subdivision,dicing_rate=dicing_rate)
        if solid:
            ibpy.add_solidify_modifier(self, thickness=solid, offset=self.get_from_kwargs('offset', -1)),
        if bevel:
            ibpy.add_bevel_modifier(self, width=bevel)

        shadow = self.get_from_kwargs('shadow', True)
        ibpy.set_shadow_of_object(self, shadow=shadow)

        # deal with loop_cuts
        loop_cuts = self.get_from_kwargs('loop_cuts', 0)
        if loop_cuts > 0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(self.ref_obj.data)  # Fills it in using the cylinder
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=int(loop_cuts), use_grid_fill=True)
            bm.to_mesh(self.ref_obj.data)
            bm.free()

        x_loop_cuts = self.get_from_kwargs('x_loop_cuts', 0)
        if x_loop_cuts > 0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(self.ref_obj.data)  # Fills it in using the cylinder
            # only divide edges that are orthogonal to x-axis
            sel_edges = []
            for e in bm.edges:
                v0 = e.verts[0].co.x
                v1 = e.verts[1].co.x
                if np.abs(v0 - v1) > 0.000001:
                    sel_edges.append(e)
            bmesh.ops.subdivide_edges(bm, edges=sel_edges, cuts=int(x_loop_cuts), use_grid_fill=True)
            bm.to_mesh(self.ref_obj.data)
            bm.free()

        y_loop_cuts = self.get_from_kwargs('y_loop_cuts', 0)
        if y_loop_cuts > 0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(self.ref_obj.data)  # Fills it in using the cylinder
            # only divide edges that are orthogonal to x-axis
            sel_edges = []
            for e in bm.edges:
                v0 = e.verts[0].co.y
                v1 = e.verts[-1].co.y
                if np.abs(v0 - v1) > 0.000001:
                    sel_edges.append(e)
            bmesh.ops.subdivide_edges(bm, edges=sel_edges, cuts=int(y_loop_cuts), use_grid_fill=True)
            bm.to_mesh(self.ref_obj.data)
            bm.free()

        z_loop_cuts = self.get_from_kwargs('z_loop_cuts', 0)
        if z_loop_cuts > 0:
            bm = bmesh.new()  # Creates an empty BMesh
            bm.from_mesh(self.ref_obj.data)  # Fills it in using the cylinder
            # only divide edges that are orthogonal to x-axis
            sel_edges = []
            for e in bm.edges:
                v0 = e.verts[0].co.z
                v1 = e.verts[-1].co.z
                if np.abs(v0 - v1) > 0.000001:
                    sel_edges.append(e)
            bmesh.ops.subdivide_edges(bm, edges=sel_edges, cuts=int(z_loop_cuts), use_grid_fill=True)
            bm.to_mesh(self.ref_obj.data)
            bm.free()

        super().__init__()
        self.current_location = self.ref_obj.location  # keep track of the current location for fast moves without determining the location at a given frame

    @classmethod
    def from_name(cls, name=None, **kwargs):
        if name:
            return BObject(obj=ibpy.get_obj_from_name(name), **kwargs)

    @classmethod
    def batch_create_from_single_mesh(cls,mesh,name="Batch",locations=[],colors=['drawing'],**kwargs):
        """
        example

        coords = [[-8 / 16.0, 0 / 16.0, 0], [-4 / 16.0, 7 / 16.0, 0], [4 / 16.0, 7 / 16.0, 0], [8 / 16.0, 0 / 16.0, 0],
                  [4 / 16.0, -7 / 16.0, 0], [-4 / 16.0, -7 / 16.0, 0]]
        faces = [[0, 5, 4, 3], [3, 2, 1, 0]]

        mesh = ibpy.create_mesh(coords,[],faces)

        locations = []
        x_factor = 24 / 16.0
        x_offset = 0.75
        y_factor = 14 / 16.0
        y_offset = 0.875 / 2
        for x_tiles in range(24):
            for y_tiles in range(40):
                locations.append(Vector([x_tiles*x_factor,y_tiles*y_factor,0.0]))
                locations.append(Vector([x_tiles*x_factor+x_offset,y_tiles*y_factor+y_offset,0.0]))

        l = len(locations)
        colors = flatten([['plastic_drawing']*int(l/2),['plastic_important']*int(l/2+1)])
        shuffle(colors)
        bobjects = BObject.batch_create_from_single_mesh(mesh,name = "hexagon",locations=locations,colors=colors,solid=0.1,bevel=0.1)
        [bob.appear(begin_time=t0,transition_time=0.1) for bob in bobjects]


        :param mesh: mesh for the objects
        :param name: base name for the objects
        :param locations: a list of locations
        :param colors: a list of colors
        :param kwargs: further parameters for BObject

        :return: a list of BObject
        """

        obj_names = []
        for i, location in enumerate(locations):
            obj_name = name + "_" + str(i)
            obj_names.append(obj_name)
            ibpy.create(mesh.copy(), obj_name, location)

        bobs = []
        for i, name in enumerate(obj_names):
            if i >= len(colors):
                col= colors[-1]
            else:
                col = colors[i]
            bob = BObject(obj=ibpy.get_obj_from_name(name),color=col,**kwargs)
            bobs.append(bob)

        return bobs

    def batch_create(cls, name="Batch",meshes=[], locations=[], colors=['drawing'], **kwargs):
        obj_names = []
        i=0
        for  mesh,location in zip(meshes,locations):
            obj_name = name + "_" + str(i)
            obj_names.append(obj_name)
            ibpy.create(mesh.copy(), obj_name, location)
            i+=1

        bobs = []
        for i, name in enumerate(obj_names):
            if i >= len(colors):
                col = colors[-1]
            else:
                col = colors[i]
            bob = BObject(obj=ibpy.get_obj_from_name(name), color=col,**kwargs)
            bobs.append(bob)

        return bobs

    @classmethod
    def from_file(cls, filename, objects=None, with_wrapper=True, **kwargs):
        if objects:
            objs = import_objects(filename, objects)
            if 'name' in kwargs:
                name = kwargs['name']
                kwargs.pop('name')
            else:
                name = ''
            if not with_wrapper:
                return objs

            if 'colors' in kwargs:
                cols = kwargs['colors']
                kwargs.pop('colors')
            else:
                cols = ['drawing']

            if 'emission' in kwargs:
                emission = kwargs.pop('emission')
            else:
                emission=0

            bobs = []
            for i, obj in enumerate(objs):
                if len(cols)==0:
                    # no colors presented
                    col="background"
                elif i < len(cols):
                    col = cols[i]
                else:
                    col = cols[-1]
                # change made on 2023-04-13 for the laptop class
                # obj_name = name + '_' + objects[i]
                # converted to
                bpy_name=obj.name
                obj_name=name+'_'+obj.name
                if i==0:
                    # added kwargs on 2024-12-08 to treat all cubies of the rubik's cube equivalently
                    bobs.append(BObject(obj=obj, color=col, name=obj_name,emission=emission,**kwargs))
                else:
                    bobs.append(BObject(obj=obj, color=col, name=obj_name, **kwargs))
                #IMPORTED_OBJECTS.append(obj_name)
                IMPORTED_OBJECTS.append(bpy_name) # 2025-02-18 to allow multiple imports of the same object
            return bobs
        else:
            obj = import_object(filename)
            IMPORTED_OBJECTS.append(
                obj.name)  # save the name of the later object to keep track and do not assign it to two references
            # print(IMPORTED_OBJECTS)
            if not with_wrapper:
                return obj
            return BObject(obj=obj, **kwargs)

    def get_from_kwargs(self, kwarg_str, default):
        if kwarg_str in self.kwargs:
            result = self.kwargs[kwarg_str]
            self.kwargs.pop(
                kwarg_str)  # change introduced on 28.5.2022 automatically remove argument, whenever it was used
            return result
        else:
            return default

    ##########
    # Getter #
    ##########

    def get_world_location(self):
        if self.ref_obj.parent:
            return self.ref_obj.parent.matrix_world @ self.ref_obj.location
        else:
            return self.ref_obj.location

    def get_location(self):
        return self.ref_obj.location

    ############
    # Setter ##
    ###########
    def set_emission_color(self,color):
        ibpy.set_emission_color(self,color)

    def set_rigid_body(self, dynamic=True):
        ibpy.make_rigid_body(self, dynamic=dynamic)

    def adopt(self, bob):
        ibpy.set_parent(bob, self)

    def add_mesh_modifier(self, type='SOLIDIFY', **kwargs):
        """
        add modifier to mesh object
        :param type:
            'DATA_TRANSFER',
            'MESH_CACHE',
            'MESH_SEQUENCE_CACHE',
            'NORMAL_EDIT',
            'WEIGHTED_NORMAL',
            'UV_PROJECT',
            'UV_WARP',
            'VERTEX_WEIGHT_EDIT',
            'VERTEX_WEIGHT_MIX',
            'VERTEX_WEIGHT_PROXIMITY',
            'ARRAY',
            'BEVEL',
            'BOOLEAN',
            'BUILD',
            'DECIMATE',
            'EDGE_SPLIT',
            'NODES',
            'MASK',
            'MIRROR',
            'MESH_TO_VOLUME',
            'MULTIRES',
            'REMESH',
            'SCREW',
            'SKIN',
            'SOLIDIFY',
            'SUBSURF',
            'TRIANGULATE',
            'VOLUME_TO_MESH',
            'WELD',
            'WIREFRAME',
            'ARMATURE',
            'CAST',
            'CURVE',
            'DISPLACE',
            'HOOK',
            'LAPLACIANDEFORM',
            'LATTICE',
            'MESH_DEFORM',
            'SHRINKWRAP',
            'SIMPLE_DEFORM',
            'SMOOTH',
            'CORRECTIVE_SMOOTH',
            'LAPLACIANSMOOTH',
            'SURFACE_DEFORM',
            'WARP',
            'WAVE',
            'VOLUME_DISPLACE',
            'CLOTH',
            'COLLISION',
            'DYNAMIC_PAINT',
            'EXPLODE',
            'FLUID',
            'OCEAN',
            'PARTICLE_INSTANCE',
            'PARTICLE_SYSTEM',
            'SOFT_BODY',
            'SURFACE'
        :param kwargs:
        :return:
        """
        ibpy.add_mesh_modifier(self, type=type, **kwargs)

    def replace_mesh_modifier(self, type='SOLIDIFY', **kwargs):
        """
        replace modifier to mesh object
        :param type:
            'DATA_TRANSFER',
            'MESH_CACHE',
            'MESH_SEQUENCE_CACHE',
            'NORMAL_EDIT',
            'WEIGHTED_NORMAL',
            'UV_PROJECT',
            'UV_WARP',
            'VERTEX_WEIGHT_EDIT',
            'VERTEX_WEIGHT_MIX',
            'VERTEX_WEIGHT_PROXIMITY',
            'ARRAY',
            'BEVEL',
            'BOOLEAN',
            'BUILD',
            'DECIMATE',
            'EDGE_SPLIT',
            'NODES',
            'MASK',
            'MIRROR',
            'MESH_TO_VOLUME',
            'MULTIRES',
            'REMESH',
            'SCREW',
            'SKIN',
            'SOLIDIFY',
            'SUBSURF',
            'TRIANGULATE',
            'VOLUME_TO_MESH',
            'WELD',
            'WIREFRAME',
            'ARMATURE',
            'CAST',
            'CURVE',
            'DISPLACE',
            'HOOK',
            'LAPLACIANDEFORM',
            'LATTICE',
            'MESH_DEFORM',
            'SHRINKWRAP',
            'SIMPLE_DEFORM',
            'SMOOTH',
            'CORRECTIVE_SMOOTH',
            'LAPLACIANSMOOTH',
            'SURFACE_DEFORM',
            'WARP',
            'WAVE',
            'VOLUME_DISPLACE',
            'CLOTH',
            'COLLISION',
            'DYNAMIC_PAINT',
            'EXPLODE',
            'FLUID',
            'OCEAN',
            'PARTICLE_INSTANCE',
            'PARTICLE_SYSTEM',
            'SOFT_BODY',
            'SURFACE'
        :param kwargs:
        :return:
        """
        ibpy.replace_mesh_modifier(self, type=type, **kwargs)


    def add_constraint(self, type='COPY_LOCATION',name=None, **kwargs):
        """
        add modifier to mesh object
        :param kwargs:
        :return:
        """
        ibpy.add_constraint(self, type=type,name=name, **kwargs)

    ################################
    # non-animated transformations #
    ################################

    def rotate_by(self, rotation_euler=[0, 0, 0]):
        ibpy.rotate_by(self, rotation_euler=rotation_euler)

    def copy(self, name=None, **kwargs):
        copy = ibpy.copy(self)

        if not name:
            name = "copy_of_" + self.ref_obj.name
        if 'scale' in kwargs:
            scale=kwargs.pop('scale')
        else:
            scale=self.intrinsic_scale
        bcopy = BObject(obj=copy, name=name,scale=scale,  **kwargs)

        bcopy.old_sk = self.old_sk
        bcopy.appeared = False
        bcopy.transformation_state = self.transformation_state

        if hasattr(self,"color") and self.color:
            # This is important to create an own material for the copy if possible
            # otherwise the material will have the same actions for the copy and the original
            # (alpha changes and so on)
            apply_material(bcopy.ref_obj, self.color, **self.kwargs)
        else:
            for i in range(0, len(self.ref_obj.material_slots)):
                mat = self.ref_obj.material_slots[i].material
                bcopy.ref_obj.material_slots[i].material = mat.copy()

        if 'hidden' in kwargs:
            hidden = kwargs.pop('hidden')
            if hidden:
                ibpy.hide(self, begin_time=0)

        if 'clear_animation_data' in kwargs:
            clear_animation_data = kwargs.pop('clear_animation_data')
            if clear_animation_data:
                ibpy.clear_animation_data(bcopy)
        return bcopy

    ##############
    # Animations #
    ##############
    def anim(self, animation, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """
        returns the end time after animation
        :param animation:
        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        """
        if animation is not None:
            animation(begin_time=begin_time, transition_time=transition_time, **kwargs)
        return begin_time + transition_time

    def highlight(self, color='text', emission=0.5, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.change_color(new_color=color, begin_time=begin_time, transition_time=transition_time)
        self.change_emission(to_value=emission, begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def unhighlight(self, color='drawing', from_emission=1, to_emission=0, begin_time=0,
                    transition_time=DEFAULT_ANIMATION_TIME):
        self.change_color(new_color=color, begin_time=begin_time, transition_time=transition_time)
        self.change_emission(from_value=from_emission, to_value=to_emission, begin_time=begin_time,
                             transition_time=transition_time)
        return begin_time + transition_time

    def transform_mesh(self, transformation, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        if there is only one transformation then the transformation can be performed immediately (backward compatability)
        :param transformation:
        :param begin_time:
        :param transition_time:
        :return:
        """
        if not isinstance(transformation, list):
            transformation = [transformation]

        state = 0
        basis = None
        for t in transformation:
            state += 1
            self.old_sk = ibpy.create_shape_key_from_transformation(self, basis, state, t)

        if len(transformation) == 1:
            self.transformation_state += 1
            ibpy.morph_to_next_shape2(self, self.transformation_state - 1, begin_time * FRAME_RATE,
                                      transition_time * FRAME_RATE)
            return begin_time + transition_time

    def transform_mesh_to_next_shape(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        if 'state' in kwargs:
            self.transformation_state = kwargs.pop('state') - 1
        ibpy.morph_to_next_shape(self, self.transformation_state, begin_time * FRAME_RATE,
                                 transition_time * FRAME_RATE)

        self.transformation_state += 1
        return begin_time + transition_time

    def transform_mesh_to_previous_shape(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        ibpy.morph_to_previous_shape(self, self.transformation_state, begin_time * FRAME_RATE,
                                     transition_time * FRAME_RATE)
        self.transformation_state -= 1
        return begin_time + transition_time

    def change_color(self, new_color, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.change_color(self, new_color, begin_frame=begin_time * FRAME_RATE,
                          final_frame=(begin_time + transition_time) * FRAME_RATE)

        return begin_time + transition_time

    def mix_color(self, old_value, new_value, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.mix_color(self, old_value, new_value, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        return begin_time + transition_time

    def shader_value(self, old_value, new_value, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.shader_value(self, old_value, new_value, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        return begin_time + transition_time

    def appear(self,alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,scale=1,
               clear_data=False, silent=False,linked=False, nice_alpha=False,children=True,
               offset_for_slots=[0],**kwargs):
        """
        makes the object simply fade in with in the transition time
        from alpha = 0 to alpha defined in kwargs (default 1)

        :param nice_alpha: the taa_render_samples are increased for a nice alpha appearance in eevee renders
        :param linked:
        :param silent:
        :param clear_data:
        :param begin_time:
        :param transition_time:
        :param kwargs:
        :return:
        """

        if not self.appeared:
            if not silent:
                print("Appear " + self.ref_obj.name)
            if nice_alpha:
                ibpy.set_taa_render_samples(1024,begin_time*FRAME_RATE)
                ibpy.set_taa_render_samples(64,(begin_time+transition_time)*FRAME_RATE)
            obj = self.ref_obj
            if not linked:
                if obj.name not in bpy.context.scene.objects:
                    ibpy.link(obj,collection=self.collection,**kwargs)

            if clear_data:  # this is useful for copies of objects to remove animation data from inherited from the parent
                ibpy.clear_animation_data(self)
            ibpy.fade_in(self, begin_time * FRAME_RATE, np.maximum(1, transition_time * FRAME_RATE), alpha=alpha,
                         offset_for_slots=offset_for_slots,**kwargs)
            self.appeared = True

        if children:
            for child in self.b_children:
                child.appear(begin_time=begin_time,transition_time=transition_time)
        return begin_time + transition_time

    def change_alpha(self, alpha=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """
        :param transition_time:
        :param begin_time:
        :return:
        """

        if transition_time == 0:
            transition_frames = 1
        else:
            transition_frames = transition_time * FRAME_RATE

        ibpy.change_alpha(self, begin_time*FRAME_RATE, transition_frames, alpha=alpha, **kwargs)
        return begin_time+transition_time

    def toggle_hide(self,begin_time=0):
        self.hide=not self.hide
        ibpy.set_hide(self,self.hide,frame=begin_time*FRAME_RATE)

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, slot=None, **kwargs):
        """
        :param alpha:
        :param transition_time:
        :param begin_time:
        :return:
        """
        if self.appeared:
            disappear_frame = begin_time * FRAME_RATE
            if transition_time == 0:
                transition_frames = 1
            else:
                transition_frames = transition_time * FRAME_RATE

            quick=get_from_kwargs(kwargs,"quick",False)
            if quick:
                ibpy.fade_out_quickly(self, disappear_frame, transition_frames, **kwargs)
            else:
                ibpy.fade_out(self, disappear_frame, transition_frames, slot =slot, alpha=alpha, **kwargs)
        if alpha == 0 and slot is None: # only disappear, when all slots have been zeroed.
            self.appeared = False
        return begin_time + transition_time

    def move_fast(self,direction=Vector(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        """
        move an object without checking its position before
        instead one works with the global variable self.current_location.
        This assumes that the move_fast_to calls are in the correct order

        :param new_location:
        :param begin_time:
        :param transition_time:
        :return:

        """
        new_location=self.current_location+to_vector(direction)
        ibpy.move_fast_from_to(self,start=self.current_location,end=new_location,begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)
        self.current_location = new_location
    def move_fast_to(self,new_location=Vector(),begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        """
        move an object without checking its position before
        instead one works with the global variable self.current_location.
        This assumes that the move_fast_to calls are in the correct order


        :param new_location:
        :param begin_time:
        :param transition_time:
        :return:

        """
        ibpy.move_fast_from_to(self,start=self.current_location,end=new_location,begin_frame=begin_time*FRAME_RATE,frame_duration=transition_time*FRAME_RATE)
        self.current_location = new_location

    def move(self, direction, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        move an object
        :param direction: in the direction
        :param begin_time: beginning of the motion
        :param transition_time: duration of the motion
        :return:
        """
        direction=to_vector(direction)
        ibpy.move(self, direction, begin_time * FRAME_RATE, transition_time * FRAME_RATE)
        return begin_time + transition_time

    def move_to(self, target_location, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, global_system=False,verbose=True):
        """
        move an object. !!! Make sure that the object has appeared before using this function otherwise there will be
        issues with visiblity
        :param global_system: if set True the motion is performed in the world system
        :param target_location: to the target_location
        :param begin_time:
        :param transition_time:
        :return:
        """

        ibpy.move_to(self, target_location, begin_time * FRAME_RATE, transition_time * FRAME_RATE,
                     global_system=global_system,verbose=verbose)
        return begin_time + transition_time

    def move_copy(self, direction=[0, 0, 0], begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        obj_copy = self.copy()
        ibpy.clear_animation_data(obj_copy)
        obj_copy.appeared=False# remove animation data to make it appear independently of the src object
        obj_copy.appear(begin_time=begin_time, transition_time=0)  # make copy appear
        obj_copy.move(direction=direction, begin_time=begin_time, transition_time=transition_time)
        return obj_copy

    def move_copy_to(self, target_location, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        obj_copy = self.copy()
        obj_copy.appear(begin_time=begin_time, transition_time=0)  # make copy appear
        obj_copy.move_to(target_location=target_location, begin_time=begin_time, transition_time=transition_time)
        if 'color' in kwargs:
            color = kwargs.pop('color')
            obj_copy.change_color(new_color=color, begin_time=begin_time, transition_time=transition_time)
        return obj_copy

    def rotate(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, pivot=None, interpolation='BEZIER',
               **kwargs):
        """
        rotate an object
        :param interpolation: CONSTANT for linear interpolation
        :param rotation_euler:
        :param rotation_quaternion:
        :param begin_time:
        :param transition_time:
        :param pivot:
        :return:
        """

        ibpy.rotate_to(self, begin_time * FRAME_RATE, transition_time * FRAME_RATE, pivot, interpolation, **kwargs)
        return begin_time + transition_time

    def scale(self, initial_scale=0, final_scale=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.grow(scale=final_scale, begin_time=begin_time, transition_time=transition_time,
                  initial_scale=initial_scale)
        return begin_time + transition_time

    def rescale(self, rescale=[1, 1, 1], begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        ibpy.rescale(self, rescale, begin_time * FRAME_RATE, np.maximum(1, transition_time * FRAME_RATE), **kwargs)
        return begin_time + transition_time

    def rename(self,name="BObject"):
        ibpy.rename(self,name)

    def grow(self, scale=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, modus='from_center', pivot=None,
             initial_scale=0,alpha=1):
        """
        grow an object from 0 to
        :param scale: the final scale
        :param begin_time: starting time
        :param transition_time: duration
        :param modus: can be 'from_center', 'from_left', 'from_right', 'from_top', 'from_bottom', 'from_front', 'from_back'
        :return:
        """

        self.appear(alpha=alpha,begin_time=begin_time, transition_time=0, silent=True)
        if pivot:
            ibpy.set_pivot(self, pivot)
        if scale is None:
            scale = self.intrinsic_scale
        ibpy.grow(self, scale, begin_time * FRAME_RATE, transition_time * FRAME_RATE, initial_scale, modus)
        return begin_time + transition_time

    def shrink(self,initial_scale=1,scale=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """
        shrink an object to 0
        :param begin_time: starting time
        :param transition_time: duration
        :return:
        """
        ibpy.shrink(self,initial_scale=initial_scale, begin_frame=begin_time*FRAME_RATE, frame_duration=FRAME_RATE*transition_time,scale=scale)
        return begin_time+transition_time

    def next_to(self, parent, direction=RIGHT, buff=SMALL_BUFF, shift=0 * RIGHT):
        """
        aligns self relative to parent
        directions can be {UP, DOWN, LEFT, RIGHT, FRONT, BACK} or any vector in principle
        :param shift:
        :param parent:
        :param direction:
        :param buff:
        :return:
        """
        # ibpy.set_parent(self, parent)
        if direction == UP or direction == DOWN:
            dim = 1
        elif direction == RIGHT or direction == LEFT:
            dim = 0
        else:
            dim = 2

        dist = ibpy.get_dimension(parent, dim) + ibpy.get_dimension(self, dim)
        dist /= 2
        dist *= self.ref_obj.scale[dim]
        location_parent = ibpy.get_location(self)  # ibpy.get_location(parent)
        ibpy.set_location(self, location_parent + (dist + buff) * direction + shift)

    def get_location_at_frame(self, frame):
        ibpy.set_frame(frame)
        location = ibpy.get_location(self)
        return location

    def update_position(self, location_frame_function, begin_time, transition_time=OBJECT_APPEARANCE_TIME,
                        location=None, resolution=1):
        if location is None:
            location = ibpy.get_location(self)
        start_frame = int(begin_time * FRAME_RATE)
        end_frame = int((begin_time + transition_time) * FRAME_RATE)

        ibpy.insert_keyframe(self.ref_obj, "location", start_frame)

        for frame in range(start_frame, end_frame, resolution):
            new_location = location_frame_function(frame + resolution)
            if location != new_location:
                self.ref_obj.location = new_location
                ibpy.insert_keyframe(self.ref_obj, "location", frame + resolution)
                location = new_location.copy()

    def add_child(self, b_object):
        ibpy.set_parent(b_object, self)

    def set_follow(self, curve, influence=1):
        ibpy.set_follow(self, curve)
        ibpy.set_follow_influence(self, curve, influence, begin_time=0)

    def change_follow_influence(self, curve, initial=0, final=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.change_follow_influence(self, curve, initial=initial, final=final, begin_time=begin_time,
                                     transition_time=transition_time)
        return begin_time + transition_time

    def follow(self, curve, initial_value=0, final_value=1, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):
        if 'new_constraint' in kwargs:
            new_constraint = kwargs.pop('new_constraint')
        else:
            new_constraint = True
        if new_constraint:
            ibpy.set_follow(self, curve,**kwargs)
        ibpy.follow(self, curve, initial_value=initial_value, final_value=final_value, begin_time=begin_time,
                    transition_time=transition_time,**kwargs)
        return begin_time+transition_time

    def hide(self, begin_time=0):
        ibpy.hide(self, begin_time=begin_time)

    def un_hide(self, begin_time=0):
        ibpy.unhide(self, begin_time=begin_time)

    def change_emission(self, from_value=0, to_value=1, slot=0,slots=None,begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        ibpy.change_emission(self, from_value=from_value, to_value=to_value, slot=slot,slots=slots,begin_frame=begin_time * FRAME_RATE,
                        frame_duration=transition_time * FRAME_RATE)
        return begin_time + transition_time

    def clear_parent(self):
        ibpy.clear_parent(self)


class AnimBObject(BObject):
    """
        Used for the spider in the Apollonian spider
    """
    def __init__(self, filename, object_name=None, armature=None, **kwargs):
        filepath = os.path.join(PRIMITIVES_DIR, filename + '.blend')
        objects = ibpy.load(filepath)
        self.kwargs = kwargs

        dict = {}
        for obj in objects:
            dict[obj.name] = obj

        if armature and armature in dict:  # if armature is there let it be constructed first and consume all scaling, rotation and translation
            rotation_euler = self.get_from_kwargs('rotation_euler', [0, 0, 0])
            location = self.get_from_kwargs('location', [0, 0, 0])
            scale = self.get_from_kwargs('scale', [1, 1, 1])
            arm = BObject(obj=dict[armature], location=location, rotation_euler=rotation_euler, scale=scale,
                          **self.kwargs)
            ibpy.link(arm)
            self.armature = arm

        if object_name and object_name in dict:
            super().__init__(obj=dict[object_name], **self.kwargs)
        else:
            raise "Object " + object_name + " not found in resource " + str(filepath)


#######################
# abstract function ###
#######################


#####################
# helper function ###
#####################

# Import object from another blend file and return bobject
def import_object(filename):
    DIR = PRIMITIVES_DIR

    filepath = os.path.join(
        DIR,
        filename
    ) + '.blend'

    bpy.ops.wm.append(
        filepath=filepath,
        directory=os.path.join(filepath, 'Mesh'),
        filename=filename
    )

    new_obj = None
    for obj in bpy.data.objects:
        if filename == obj.name:
            new_obj = obj
            break

    if not new_obj:  # an object with precisely the name hasn't been found, maybe due to canonical renaming
        for obj in bpy.data.objects:
            if filename in obj.name and obj.name not in IMPORTED_OBJECTS:
                new_obj = obj
                break

    if new_obj is None:
        raise Warning('Did not find object with same name as file')

    return new_obj


def import_objects(filename, objects):
    DIR = PRIMITIVES_DIR

    filepath = os.path.join(
        DIR,
        filename
    ) + '.blend'

    objs = []
    for obj in objects:
        bpy.ops.wm.append(
            filepath=filepath,
            directory=os.path.join(filepath, 'Mesh'),
            filename=obj
        )

        for o in bpy.data.objects:
            if obj in o.name and o.name not in IMPORTED_OBJECTS:
                objs.append(o)

    return objs
