import math
from fractions import Fraction
from itertools import combinations

import numpy as np
from math import gcd

from camb.model import transfer_names
from mathutils import Vector

from appearance.textures import billiards_cloth_material, billiards_ball_material, get_texture
from interface import ibpy
from interface.ibpy import create_mesh, append_materials, get_geometry_node_from_modifier, change_default_value
from geometry_nodes.geometry_nodes_modifier import ScoreTableModifier, BilliardsTableModifier, BilliardsBallModifier, \
    ReflectableBilliardPaperModifier, BilliardBallRealModifier
from objects.bobject import BObject
from objects.cube import Cube
from objects.cylinder import Cylinder
from objects.plane import Plane
from utils.color_conversion import get_color_from_string
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs

pi = np.pi


def f(w, h):
    ggt = gcd(w, h)
    w /= ggt
    h /= ggt

    if w % 2 == 0:
        return 1
    elif h % 2 == 0:
        return 2
    else:
        return 0


class ReflectableBilliardPaper(BObject):
    def __init__(self, width=10, height=10, location=[0, 0, 0], **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs("name", "ReflectableBilliardPaper")
        self.height = height
        self.width = width

        self.paper = Plane(u=[0, width], v=[0, height], color="gray_1",
                           name="ReflectableBilliardPaper" + str(width) + "x" + str(height), **kwargs)
        self.modifier = ReflectableBilliardPaperModifier(width=width,height=height,**kwargs)
        self.paper.add_mesh_modifier(type="NODES", node_modifier=self.modifier)
        super().__init__(obj=self.paper, **kwargs)
        self.unfold_counter=0

    def unfold(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        rotation_node= ibpy.get_geometry_node_from_modifier(self.modifier,"Rotation"+str(self.unfold_counter))
        ibpy.change_default_value(rotation_node,from_value=0,to_value=pi,begin_time=begin_time,transition_time=transition_time)
        self.unfold_counter+=1
        return begin_time+transition_time

class BilliardPaper(BObject):
    def __init__(self, width=10, height=10, location=[0, 0, 0], **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', "BilliardPaper")
        self.height = height
        self.width = width
        paper_color = get_texture(self.get_from_kwargs("paper_color","gray_1"))

        self.paper = Plane(u=[0, width], v=[0, height],  name="Paper" + str(width) + "x" + str(height),
                           **kwargs)

        # self.paper = Plane(u=[-width/2, width/2], v=[-height/2, height/2], color="gray_1", name="Paper"+str(width)+"x"+str(height),**kwargs)
        self.paper.move_to(target_location=location, begin_time=0, transition_time=0)

        lines = []
        for w in range(width + 1):
            for h in range(height + 1):
                lines.append((0, h, width, h))
                lines.append((w, 0, w, height))
                # lines.append((-width/2,-height/2+h,width/2,-height/2+h))
                # lines.append((-width/2+w,-height/2,-width/2+w,height/2))

        self.cylinders = []
        for line in lines:
            self.cylinders.append(Cylinder.from_start_to_end(
                start=Vector([line[0], line[1], 0]), end=Vector([line[2], line[3], 0]), radius=0.01))
            ibpy.set_parent(self.cylinders[-1], self.paper)

        super().__init__(obj=self.paper, **kwargs)
        # quick and dirty
        self.ref_obj.ref_obj.material_slots[0].material = paper_color

    def grow(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        t2 = transition_time / 2
        t0 = self.paper.grow(begin_time=begin_time, transition_time=t2)

        for l, line in enumerate(self.cylinders):
            line.grow(begin_time=t0 + l * t2 / 2 / len(self.cylinders), transition_time=t2 / 2)

        return begin_time + transition_time

    def show_ball(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        self.ball = BilliardsBall(width=self.width, height=self.height, start_time=begin_time + transition_time,
                                  grid_radius_ratio=1, radius=0.333,
                                  ball_material="billiards_ball_material", trace_material="custom1",
                                  location=[self.width / 2, self.height / 2, 0])
        self.ball.appear(begin_time=begin_time, transition_time=transition_time)
        ibpy.set_parent(self.ball, self.paper)

        return begin_time + transition_time

    def disappear_ball(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        return self.ball.ball_disappear(begin_time=begin_time, transition_time=transition_time, **kwargs)

    def disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        self.ball.line_disappear(begin_time=begin_time, transition_time=transition_time, **kwargs)
        self.paper.disappear(begin_time=begin_time, transition_time=transition_time,children=False, **kwargs)
        [cyl.disappear(begin_time=begin_time, transition_time=transition_time, **kwargs) for cyl in self.cylinders]
        return begin_time + transition_time

class BilliardsTable(BObject):
    def __init__(self, width=10, height=10, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'BilliardsTable')
        self.height = height
        self.width = width

        radius = get_from_kwargs(kwargs, "radius", 1)
        grid_radius_ratio = get_from_kwargs(kwargs, "grid_radius_ratio", 4)
        table_cloth_material = get_from_kwargs(kwargs, "table_cloth_material", "billiards_cloth_material")
        table_rim_material = get_from_kwargs(kwargs, "table_rim_material", "plastic_background")
        grid_material = get_from_kwargs(kwargs, "grid_material", "plastic_example")

        if table_cloth_material == "billiards_cloth_material":
            table_cloth_material = billiards_cloth_material(**kwargs)
        else:
            table_cloth_material = ibpy.get_material(table_cloth_material, **kwargs)
        table_rim_material = ibpy.get_material(table_rim_material, roughness=0.2, **kwargs)
        grid_material = ibpy.get_material(grid_material, emission=1, **kwargs)

        cube = Cube()
        modifier = BilliardsTableModifier(width=width, height=height,
                                          name="BilliardsTable" + str(width) + "x" + str(height), **kwargs)
        cube.add_mesh_modifier(type="NODES",
                               node_modifier=modifier)

        gnmod =ibpy.get_geometry_nodes_modifier(cube)

        if gnmod is not None and gnmod.type == "NODES":
            socket_names = ibpy.get_socket_names_from_modifier(gnmod)

            gnmod[socket_names["Width"]] = self.width
            gnmod[socket_names["Height"]] = self.height
            gnmod[socket_names["Radius"]] = radius
            gnmod[socket_names["GridRadiusRatio"]] = grid_radius_ratio
            gnmod[socket_names["TableClothMaterial"]] = table_cloth_material
            gnmod[socket_names["TableRimMaterial"]] = table_rim_material
            gnmod[socket_names["GridMaterial"]] = grid_material

            modifier.materials.append(table_rim_material)
            modifier.materials.append(table_cloth_material)
            modifier.materials.append(grid_material)

        append_materials(cube, modifier.materials)

        # cube.ref_obj.modifiers.update()
        super().__init__(obj=cube.ref_obj, name=self.name, **kwargs)

class BilliardsBall(BObject):
    def __init__(self, width=10, height=10, subdivisions=5, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'BilliardsBall')
        self.height = height
        self.width = width

        radius = get_from_kwargs(kwargs, "radius", 1)
        speed = get_from_kwargs(kwargs, "speed", 1)
        grid_radius_ratio = get_from_kwargs(kwargs, "grid_radius_ratio", 4)
        initial_position = get_from_kwargs(kwargs, "initial_position", [0, 0, radius])
        ball_material_str = get_from_kwargs(kwargs, "ball_material", "billiards_ball_material")
        if ball_material_str == "billiards_ball_material":
            ball_material = billiards_ball_material(**kwargs)
        else:
            ball_material = ibpy.get_material(ball_material_str, **kwargs)
        trace_material_str = get_from_kwargs(kwargs, "trace_material", "text")
        trace_material = ibpy.get_material(trace_material_str, **kwargs)
        start_time = get_from_kwargs(kwargs, "start_time", 0)

        cube = Cube()
        modifier = BilliardsBallModifier(width=width, height=height,
                                         name="BilliardsBall" + str(width) + "x" + str(height),
                                         subdivisions=subdivisions, **kwargs)

        cube.add_mesh_modifier(type="NODES",
                               node_modifier=modifier)

        gnmod = None
        for gnmod in cube.ref_obj.modifiers:
            if gnmod.type == "NODES":
                break

        if gnmod is not None and gnmod.type == "NODES":
            socket_names = {item.name: f"{item.identifier}" for item in gnmod.node_group.interface.items_tree
                            if item.in_out == "INPUT"}

            gnmod[socket_names["Width"]] = self.width
            gnmod[socket_names["Height"]] = self.height
            gnmod[socket_names["Radius"]] = radius
            gnmod[socket_names["GridRadiusRatio"]] = grid_radius_ratio
            gnmod[socket_names["BallMaterial"]] = ball_material
            gnmod[socket_names["TraceMaterial"]] = trace_material
            gnmod[socket_names["BallPosition"]] = initial_position
            gnmod[socket_names["StartTime"]] = start_time
            gnmod[socket_names["Speed"]] = speed

            modifier.materials.append(ball_material)
            modifier.materials.append(trace_material)

            # only call this function after the materials have been added
            # this way the materials will be added to the slots of the material automatically
            append_materials(cube, modifier.materials)

        # cube.ref_obj.modifiers.update()
        super().__init__(obj=cube.ref_obj, name=self.name, **kwargs)

    def ball_disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        return super().disappear(begin_time=begin_time, transition_time=transition_time, slot=1, **kwargs)

    def line_disappear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        return super().disappear(begin_time=begin_time, transition_time=transition_time, slot=2, **kwargs)

class BilliardBallReal(BObject):
    def __init__(self, width=10, height=10, subdivisions=5, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'BilliardsBall')
        self.height = height
        self.width = width
        self.scale = self.get_from_kwargs("scale",[1]*3)

        color = get_from_kwargs(kwargs,"color","black")
        material = get_texture("billiard_ball_material",color=color, **kwargs)
        number= get_from_kwargs(kwargs,"number","8")
        if callable(number):
            number=str(number(**kwargs))

        solid = get_from_kwargs(kwargs,"solid",False)
        text_material=get_from_kwargs(kwargs,"text_color",get_texture("billiard_ball_material",color="black",**kwargs))

        cube = Cube()
        modifier = BilliardBallRealModifier(
                                         name="BilliardsBallReal", **kwargs)

        cube.add_mesh_modifier(type="NODES",
                               node_modifier=modifier)

        gnmod = None
        for gnmod in cube.ref_obj.modifiers:
            if gnmod.type == "NODES":
                break

        if gnmod is not None and gnmod.type == "NODES":
            socket_names = {item.name: f"{item.identifier}" for item in gnmod.node_group.interface.items_tree
                            if item.in_out == "INPUT"}


            white_material = get_texture("billiard_ball_material",color="white",**kwargs)
            black_material = get_texture("billiard_ball_material",color="black", **kwargs)
            gnmod[socket_names["Number"]] = number
            gnmod[socket_names["White"]] = white_material
            gnmod[socket_names["Color"]] = material
            gnmod[socket_names["TextColor"]] = black_material
            gnmod[socket_names["Solid"]] = solid


            modifier.materials.append(white_material)
            modifier.materials.append(black_material)
            modifier.materials.append(text_material)
            modifier.materials.append(material)

            # only call this function after the materials have been added
            # this way the materials will be added to the slots of the material automatically
            append_materials(cube, modifier.materials)

        # cube.ref_obj.modifiers.update()
        super().__init__(obj=cube.ref_obj, name=self.name, **kwargs)

class ScoreTable(BObject):
    def __init__(self, width=10, height=10, **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'ScoreTable')
        self.height = height
        vertices = []
        for w in range(1, width + 1):
            for h in range(1, height + 1):
                vertices.append([w, f(w, h), h])

        super().__init__(mesh=create_mesh(vertices, [], []), name=self.name, **kwargs)

        modifier = ScoreTableModifier(width=width, height=height, name="ScoreTable", **kwargs)
        ibpy.add_mesh_modifier(self, type="NODES", node_modifier=modifier)
        self.modifier = modifier
        self.index_selector = ibpy.get_geometry_node_from_modifier(modifier, label="IndexSelector")

    def turn_on(self, data_list, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time / len(data_list)
        t0 = begin_time
        for d in data_list:
            t0 = dt + self.single_turn_on(d, begin_time=t0)
        return t0 + transition_time

    def single_turn_on(self, data, begin_time=0):
        ibpy.change_default_value(self.index_selector.inputs[(data[0] - 1) * self.height + data[1]], from_value=False,
                                  to_value=True,
                                  begin_time=begin_time,transition_time=0)
        return begin_time

    def turn_off(self, data_list, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        dt = transition_time / len(data_list)
        t0 = begin_time
        for d in data_list:
            t0 = dt + self.single_turn_off(d, begin_time=t0)
        return t0 + transition_time

    def single_turn_off(self, data, begin_time=0):
        ibpy.change_default_value(self.index_selector.inputs[(data[0] - 1) * self.height + data[1]], from_value=True,
                                  to_value=False,
                                  begin_time=begin_time)
        return begin_time

    def disappear(self, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,  **kwargs):
        """
        make sure that the label colors dissappear first
        """
        dt = transition_time/2
        for mat in self.modifier.materials:
            if "background" in mat.name:
                ibpy.change_alpha_of_material(mat,from_value=1,to_value=alpha,begin_time=begin_time+dt,transition_time=dt)
            else:
                ibpy.change_alpha_of_material(mat,from_value=1,to_value=alpha,begin_time=begin_time,transition_time=dt)


