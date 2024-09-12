import numpy as np
from mathutils import Vector

from interface import ibpy
from interface.ibpy import add_cone, add_cylinder
from objects.cone import Cone
from objects.bobject import BObject
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.empties import EmptyArrow, EmptyAxes
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Book(BObject):
    """
    Create a cylinder with a descent mesh:
    """

    def __init__(self, scale=[1, 1, 0.4], pages=100, cover_thickness=0.01, page_thickness=0.002, **kwargs):
        self.kwargs = kwargs
        self.scale = scale
        self.cover_thickness = 0.01
        self.pages = pages
        self.page_list = []
        name = self.get_from_kwargs('name', 'Book')
        self.cover = Cube(scale=[scale[0], scale[1], cover_thickness],
                          location=[-scale[0] - scale[2] + cover_thickness, 0, 0],
                          origin=[-scale[2] + cover_thickness, 0, 0],
                          name='cover')  # add cover_thickness to avoid gaps while opening the book
        self.back = Cube(scale=[scale[0], scale[1], cover_thickness],
                         location=[scale[0] + scale[2] - cover_thickness, 0, 0],
                         origin=[scale[2] - cover_thickness, 0, 0],
                         name='back')
        self.spine = Cube(scale=[scale[2], scale[1], cover_thickness], name='spine', apply_scale=True)
        ibpy.set_parent(self.cover, self.spine)
        ibpy.set_parent(self.back, self.spine)
        empty_scale = [0.3 * scale[0]] * 3
        self.arrow_empties = [EmptyArrow(location=[-scale[2], 0, 0], scale=empty_scale, apply_scale=True),
                              EmptyArrow(location=[scale[2], 0, 0], scale=empty_scale, apply_scale=True)]
        ibpy.add_constraint(self.spine, type='PIVOT', target=self.arrow_empties[0], rotation_range='NY')
        ibpy.add_constraint(self.spine, type='PIVOT', target=self.arrow_empties[1], rotation_range='Y')
        ibpy.add_constraint(self.cover, type='COPY_ROTATION', target=self.spine, mix_mode='ADD')
        ibpy.add_constraint(self.back, type='COPY_ROTATION', target=self.spine, mix_mode='ADD')
        ibpy.add_constraint(self.cover, type='LIMIT_ROTATION', limit_y=True, min_y=0, max_y=np.pi,use_legacy_behavior=True)
        ibpy.add_constraint(self.back, type='LIMIT_ROTATION', limit_y=True, min_y=-np.pi, max_y=0,use_legacy_behavior=True)

        self.spine.ref_obj.rotation_euler = [0, np.pi / 2, 0]
        self.add_pages(pages, page_thickness)

        super().__init__(children=[self.spine] + self.arrow_empties + self.page_list, name=name, **kwargs)

    def add_pages(self, pages, page_thickness):
        # add pages into the open book
        scale = [0.99 * self.scale[0], 0.99 * self.scale[1], page_thickness]
        for i in range(0, pages + 1):
            location = [
                -self.scale[2] + 2 * self.cover_thickness + 2 * i / pages * (self.scale[2] - 2 * self.cover_thickness),
                0, 0]
            page = Cube(scale=scale, rotation_euler=[np.pi / 2, 0, 0],
                        location=[0.99 * scale[0], 0, 0], origin=Vector(), x_loop_cuts=39, apply_rotation=True,
                        name='page')
            page.ref_obj.location = location
            page.add_mesh_modifier(type='SIMPLE_DEFORM', deform_method='BEND', deform_axis='Z', angle=0, name='deform')
            hook = EmptyAxes(scale=0.15, apply_scale=True, name='hook_for_pg',location=location)
            hook.ref_obj.rotation_euler=[-np.pi/2,-np.pi/2,0]
            vertex_group = ibpy.add_vertex_group(page, name="hook_group",
                                                 weight_function=lambda v: ((2 - 0.95 * v.x * scale[0]) / 2) ** 4)
            page.add_mesh_modifier(type='HOOK', object=hook, vertex_group=vertex_group)
            ibpy.set_parent(hook, self.spine)
            self.page_list.append(page)

            # close the book as start configuration
            page.ref_obj.location =[self.scale[2],0,self.scale[2]-location[0]]
            ibpy.insert_keyframe(page, 'location', 0)
            page.ref_obj.rotation_euler = [-np.pi / 2, 0, 0]
            ibpy.insert_keyframe(page, 'rotation_euler', 0)

    def set_cover_image(self, src,**kwargs):
        mat = ibpy.make_image_material(src,**kwargs)
        ibpy.set_material(self.cover, mat, slot=1)
        ibpy.asign_material_to_faces(self.cover, 1, normal=Vector([0, 0, -1]))

    def set_page_image(self, index,src,**kwargs):
        if index%2==0:
            normal = Vector([0,-1,0])
            slot = 0
        else:
            normal = Vector([0,1,0])
            slot = 1

        mat = ibpy.make_image_material(src,**kwargs)
        page = index // 2 - 1
        ibpy.set_material(self.page_list[page], mat, slot=slot)
        ibpy.asign_material_to_faces(self.page_list[page], slot, normal=normal)

    def closed(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, direction='right'):
        if direction == 'right':
            self.spine.rotate(rotation_euler=[0, -np.pi / 2, 0], begin_time=begin_time, transition_time=transition_time)
        else:
            self.spine.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def open(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.spine.rotate(rotation_euler=[0, 0, 0], begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def turn_page(self, index, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        page = self.page_list[index % (self.pages + 1)]
        half = transition_time / 2
        page.rotate(rotation_euler=[-np.pi / 2, -np.pi, 0], begin_time=begin_time, transition_time=transition_time)
        location = [
            -self.scale[2] + 2 * self.cover_thickness + 2 * index / self.pages * (
                        self.scale[2] - 2 * self.cover_thickness), 0, 0]
        page.move_to(target_location=location, begin_time=begin_time, transition_time=half)
        location2 = [-self.scale[2], 0, self.scale[2] + location[0]]
        page.move_to(target_location=location2, begin_time=begin_time + half, transition_time=half)
        half = int(transition_time * FRAME_RATE / 2)
        ibpy.change_modifier_attribute(page, 'deform', 'angle', 0, np.pi / 4, begin_frame=begin_time * FRAME_RATE,
                                       frame_duration=half)
        ibpy.change_modifier_attribute(page, 'deform', 'angle', np.pi / 4, 0,
                                       begin_frame=begin_time * FRAME_RATE + half + 1, frame_duration=half)
        return begin_time + transition_time


