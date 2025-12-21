import math
from fractions import Fraction
from math import gcd

import numpy as np

from appearance.textures import billiards_cloth_material, get_texture
from interface import ibpy
from interface.ibpy import create_mesh, append_materials, get_geometry_node_from_modifier, change_default_value
from geometry_nodes.geometry_nodes_modifier import BilliardsTableRoundModifier, BilliardBallRoundModifier
from objects.bobject import BObject
from objects.cube import Cube
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

class BilliardsTableRound(BObject):
    def __init__(self,radius=4,  **kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'BilliardsTableRound')

        grid_radius_ratio = get_from_kwargs(kwargs, "grid_radius_ratio", radius+1)
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
        modifier = BilliardsTableRoundModifier(name="BilliardsTableRound", **kwargs)
        cube.add_mesh_modifier(type="NODES",
                               node_modifier=modifier)

        gnmod =ibpy.get_geometry_nodes_modifier(cube)

        if gnmod is not None and gnmod.type == "NODES":
            socket_names = ibpy.get_socket_names_from_modifier(gnmod)

            gnmod[socket_names["Radius"]] = 1
            gnmod[socket_names["GridRadiusRatio"]] = grid_radius_ratio
            gnmod[socket_names["TableClothMaterial"]] = table_cloth_material
            gnmod[socket_names["TableRimMaterial"]] = table_rim_material

            modifier.materials.append(table_rim_material)
            modifier.materials.append(table_cloth_material)
            modifier.materials.append(grid_material)

            append_materials(cube, modifier.materials)

        # cube.ref_obj.modifiers.update()
        super().__init__(obj=cube.ref_obj, name=self.name, **kwargs)

class BilliardBallRound(BObject):
    def __init__(self, ratio = Fraction(1,7), radius = 10, number=8, time_per_path=1,prime=1039,**kwargs):
        self.kwargs = kwargs

        if isinstance(ratio,tuple):
            ratio = Fraction(*ratio)
            n_paths = ratio.denominator  # number of paths
        elif isinstance(ratio,int):
            ratio = Fraction(1,ratio)
            n_paths = ratio.denominator  # number of paths
        elif isinstance(ratio,float):
            n_paths=prime

        self.name = self.get_from_kwargs('name', 'BilliardBallRound')
        self.ratio = ratio
        number_string = str(number)
        if number>8:
            solid = False
            number = number-8
        else:
            solid = True
        color = get_color_from_string("billiard_"+str(number))
        material = get_texture("billiard_ball_material", color=color, **kwargs)

        text_material = get_from_kwargs(kwargs, "text_color",
                                        get_texture("billiard_ball_material", color="black", **kwargs))

        trace_material_str = get_from_kwargs(kwargs, "trace_material", "example")
        trace_material = ibpy.get_material(trace_material_str,emission=0.5, **kwargs)

        reflection_point_material = get_texture("plastic_joker",**kwargs)

        vertices = []
        # compute the number of reflections
        # the reflection points determine the path of the billiard ball. It is always moving from one reflection point to the next
        for i in range(n_paths):
            x = radius * math.cos(2*math.pi*i*ratio)
            y = radius * math.sin(2*math.pi*i*ratio)

            vertices.append([x,y,0])

        reflection_points = BObject(mesh=create_mesh(vertices),name="ReflectionPoints",**kwargs)


        cube = Cube()
        modifier = BilliardBallRoundModifier(name="BilliardBallRound", **kwargs)

        time_per_path_node = get_geometry_node_from_modifier(modifier,label="TimePerPath")
        change_default_value(time_per_path_node,from_value=1,to_value=time_per_path,begin_time=0,transition_time=0)

        cube.add_mesh_modifier(type="NODES",
                               node_modifier=modifier)

        gnmod = None
        for gnmod in cube.ref_obj.modifiers:
            if gnmod.type == "NODES":
                break

        if gnmod is not None and gnmod.type == "NODES":
            socket_names = {item.name: f"{item.identifier}" for item in gnmod.node_group.interface.items_tree
                            if item.in_out == "INPUT"}

            white_material = get_texture("billiard_ball_material", color="white", **kwargs)
            black_material = get_texture("billiard_ball_material", color="black", **kwargs)
            gnmod[socket_names["Number"]] = number_string
            gnmod[socket_names["White"]] = white_material
            gnmod[socket_names["Color"]] = material
            gnmod[socket_names["TextColor"]] = black_material
            gnmod[socket_names["Solid"]] = solid
            gnmod[socket_names["ReflectionPoints"]] = reflection_points.ref_obj
            gnmod[socket_names["TraceColor"]] = trace_material
            gnmod[socket_names["PointColor"]] = reflection_point_material

            modifier.materials.append(white_material)
            modifier.materials.append(black_material)
            modifier.materials.append(text_material)
            modifier.materials.append(material)
            modifier.materials.append(trace_material)
            modifier.materials.append(reflection_point_material)

            # only call this function after the materials have been added
            # this way the materials will be added to the slots of the material automatically
            append_materials(cube, modifier.materials)
            self.modifier = modifier

        self.n_paths = n_paths

        super().__init__(obj=cube.ref_obj, name=self.name, **kwargs)

    def start(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        start_time_node = get_geometry_node_from_modifier(self.modifier,label="StartTime")
        change_default_value(start_time_node,from_value=0,to_value=begin_time)
        time_per_path_node = get_geometry_node_from_modifier(self.modifier,label="TimePerPath")
        change_default_value(time_per_path_node,from_value=1,to_value=transition_time/self.n_paths)

        return begin_time+transition_time

    def change_trace_thickness(self,from_value=1,to_value=0.1,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        trace_thickness_node = get_geometry_node_from_modifier(self.modifier,label="TraceThickness")
        change_default_value(trace_thickness_node,from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def grow_reflection_points(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        reflection_point_slider_node=get_geometry_node_from_modifier(self.modifier,label="ReflectionPointSlider")
        change_default_value(reflection_point_slider_node,from_value=-0.001,to_value=1,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time