import os

import numpy as np
from mathutils import Vector

from appearance.textures import get_texture
from interface import ibpy
from interface.ibpy import append_materials, get_geometry_node_from_modifier, change_default_integer, \
    change_default_value, change_default_vector
from geometry_nodes.geometry_nodes_modifier import ReflectableBilliardPaperModifier, BenfordDiagramModifier, \
    BenfordFibonacciDiagramModifier, BenfordIntervalModifier, BenfordFilesDiagramModifier
from objects.bobject import BObject
from objects.cube import Cube
from objects.plane import Plane
from utils.constants import FRAME_RATE, DEFAULT_ANIMATION_TIME, DATA_DIR
from utils.io_operations import convert_files_to_csv_data
from utils.kwargs import get_from_kwargs

class BenfordFilesDiagram(BObject):
    """A Benford-law bar chart computed from filename lengths in a directory.

    Renders the leading-digit distribution as animated bars on a
    :class:`Plane`, driven by :class:`BenfordFilesDiagramModifier`.
    """

    def __init__(self,path="/usr",max_length=35,max_data=2000, **kwargs):
        """Build a Benford diagram from the contents of ``path``.

        Args:
            path: Root directory to scan. Defaults to ``'/usr'``.
            max_length: Maximum filename length displayed.
            max_data: Maximum number of files to include.
            **kwargs: Forwarded to the modifier and :class:`BObject`.
                Supported keys: ``numer_material``, ``highlighted_number_material``,
                ``bar_material`` (color names), ``transition_frames``
                (animation duration in frames), ``start_time`` (seconds).
        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs("name", "BenfordFilesDiagram")

        self.geo = Plane()
        self.modifier = BenfordFilesDiagramModifier(**kwargs)
        self.geo.add_mesh_modifier(type="NODES", node_modifier=self.modifier)

        number_material = get_texture(get_from_kwargs(kwargs,"numer_material","joker"))
        highlighted_number_material = get_texture(get_from_kwargs(kwargs,"highlighted_number_material","custom1"))
        bar_material = get_texture(get_from_kwargs(kwargs,"bar_material","custom3"))

        convert_files_to_csv_data(path, max_length, max_data, max_number=10000)

        gnmod = ibpy.get_geometry_nodes_modifier(self.geo)
        if gnmod is not None and gnmod.type == "NODES":
            socket_names = ibpy.get_socket_names_from_modifier(gnmod)

            gnmod[socket_names["CSV_Data"]] = os.path.join(DATA_DIR,path.replace("/","")+"_data.csv")
            gnmod[socket_names["TransitionFrames"]] = get_from_kwargs(kwargs, "transition_frames", 30)
            gnmod[socket_names["NumberMaterial"]] = number_material
            gnmod[socket_names["HighlightedNumberMaterial"]] = highlighted_number_material
            gnmod[socket_names["BarMaterial"]] = bar_material
            gnmod[socket_names["StartFrame"]]= get_from_kwargs(kwargs,"start_time",0)*FRAME_RATE

            self.modifier.materials.append(number_material)
            self.modifier.materials.append(highlighted_number_material)
            self.modifier.materials.append(bar_material)

        append_materials(self.geo,self.modifier.materials)

        super().__init__(obj=self.geo, **kwargs)


class BenfordFibonacciDiagram(BObject):
    """A Benford-law bar chart computed from Fibonacci numbers.

    The diagram updates as more Fibonacci values are added via
    :meth:`add_data`."""

    def __init__(self, **kwargs):
        """Build a Benford diagram for an interactively-extended Fibonacci sequence.

        Args:
            **kwargs: Same shading kwargs as :class:`BenfordFilesDiagram`.
        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs("name", "BenfordDiagram")

        self.geo = Plane()
        self.modifier = BenfordFibonacciDiagramModifier(**kwargs)
        self.geo.add_mesh_modifier(type="NODES", node_modifier=self.modifier)


        number_material = get_texture(get_from_kwargs(kwargs,"numer_material","joker"))
        highlighted_number_material = get_texture(get_from_kwargs(kwargs,"highlighted_number_material","custom1"))
        bar_material = get_texture(get_from_kwargs(kwargs,"bar_material","custom3"))


        gnmod = ibpy.get_geometry_nodes_modifier(self.geo)
        if gnmod is not None and gnmod.type == "NODES":
            socket_names = ibpy.get_socket_names_from_modifier(gnmod)

            gnmod[socket_names["TransitionFrames"]] = get_from_kwargs(kwargs, "transition_frames", 30)
            gnmod[socket_names["NumberMaterial"]] = number_material
            gnmod[socket_names["HighlightedNumberMaterial"]] = highlighted_number_material
            gnmod[socket_names["BarMaterial"]] = bar_material
            gnmod[socket_names["StartFrame"]]= get_from_kwargs(kwargs,"start_time",0)*FRAME_RATE

            self.modifier.materials.append(number_material)
            self.modifier.materials.append(highlighted_number_material)
            self.modifier.materials.append(bar_material)

        append_materials(self.geo,self.modifier.materials)

        super().__init__(obj=self.geo, **kwargs)

    def add_data(self,data):
        data_switch=get_geometry_node_from_modifier(self.modifier,label="DataSwitch")
        for i,d in enumerate(data):
            ibpy.add_item_to_switch(data_switch,i,str(d))


class BenfordDiagram(BObject):
    """A generic Benford-law bar chart driven by :class:`BenfordDiagramModifier`."""

    def __init__(self, **kwargs):
        """Build a blank Benford diagram (data is supplied via the modifier).

        Args:
            **kwargs: Same shading kwargs as :class:`BenfordFilesDiagram`.
        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs("name", "BenfordFibonacciDiagram")

        self.geo = Plane()
        self.modifier = BenfordDiagramModifier(**kwargs)
        self.geo.add_mesh_modifier(type="NODES", node_modifier=self.modifier)


        number_material = get_texture(get_from_kwargs(kwargs,"numer_material","joker"))
        highlighted_number_material = get_texture(get_from_kwargs(kwargs,"highlighted_number_material","custom1"))
        bar_material = get_texture(get_from_kwargs(kwargs,"bar_material","custom3"))


        gnmod = ibpy.get_geometry_nodes_modifier(self.geo)
        if gnmod is not None and gnmod.type == "NODES":
            socket_names = ibpy.get_socket_names_from_modifier(gnmod)

            gnmod[socket_names["TransitionFrames"]] = get_from_kwargs(kwargs, "transition_frames", 30)
            gnmod[socket_names["NumberMaterial"]] = number_material
            gnmod[socket_names["HighlightedNumberMaterial"]] = highlighted_number_material
            gnmod[socket_names["BarMaterial"]] = bar_material
            gnmod[socket_names["StartFrame"]]= get_from_kwargs(kwargs,"start_time",0)*FRAME_RATE

            self.modifier.materials.append(number_material)
            self.modifier.materials.append(highlighted_number_material)
            self.modifier.materials.append(bar_material)

        append_materials(self.geo,self.modifier.materials)

        super().__init__(obj=self.geo, **kwargs)


class BenfordInterval(BObject):
    """A number-line-style helper that visualises which leading digit a
    given value would have, driven by :class:`BenfordIntervalModifier`."""

    def __init__(self, **kwargs):
        """Build a Benford interval display.

        Args:
            **kwargs: Forwarded to the modifier and :class:`BObject`.
                Supported keys:
                * ``scale`` (float | list[float]): Scale applied to the plane.
                  Defaults to 1.
                * ``location`` (list[float]): World location.
                  Defaults to ``[0, 0, 0]``.
                * ``axis_material``, ``tic_material``, ``label_material``
                  (str): Color names for the axis, tic marks, and labels.
                  Defaults: ``'text'``, ``'text'``, ``'important'``.
                * Standard BObject kwargs.
        """
        self.kwargs = kwargs
        self.name = self.get_from_kwargs("name", "BenfordInterval")

        self.geo = Plane(name=self.name,**kwargs)
        self.modifier = BenfordIntervalModifier(**kwargs)
        self.geo.add_mesh_modifier(type="NODES", node_modifier=self.modifier)
        scale=get_from_kwargs(kwargs,"scale",1)
        location=get_from_kwargs(kwargs,"location",[0,0,0])
        self.geo.rescale(rescale=scale,begin_time=0,transition_time=0)
        self.geo.move_to(target_location=location,begin_time=0,transition_time=0)
        # set material
        axis_material_node = get_geometry_node_from_modifier(self.modifier,label="AxisMaterial")
        axis_material_node.material = get_texture(get_from_kwargs(kwargs,"axis_material","text"))

        tic_material_node = get_geometry_node_from_modifier(self.modifier,label="TicMaterial")
        tic_material_node.material = get_texture(get_from_kwargs(kwargs,"tic_material","text"))

        label_material_node = get_geometry_node_from_modifier(self.modifier,label="LabelMaterial")
        label_material_node.material = get_texture(get_from_kwargs(kwargs,"label_material","important"))
        self.modifier.materials.append(axis_material_node.material)
        self.modifier.materials.append(tic_material_node.material)
        self.modifier.materials.append(label_material_node.material)
        super().__init__(obj=self.geo,**kwargs)

    def show_tics(self,from_value=0,to_value=9,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        tic_range_node=get_geometry_node_from_modifier(self.modifier,label="TicRange")
        steps = int(to_value-from_value)
        dt = transition_time/steps
        for s in range(steps):
            value = from_value+1
            change_default_integer(tic_range_node,from_value=from_value,to_value=value,begin_time=begin_time+s*dt,transition_time=dt)
            from_value=value
        return begin_time+transition_time

    def show_labels(self,from_value=0,to_value=9,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        label_range_node=get_geometry_node_from_modifier(self.modifier,label="LabelRange")
        steps = int(to_value-from_value)
        dt = transition_time/steps
        for s in range(steps):
            value = from_value+1
            change_default_integer(label_range_node,from_value=from_value,to_value=value,begin_time=begin_time+s*dt,transition_time=dt)
            from_value=value
        return begin_time+transition_time

    def transform_interval(self,from_value=0,to_value=9,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        slider_node=get_geometry_node_from_modifier(self.modifier,label="Slider")
        shift_node=get_geometry_node_from_modifier(self.modifier,label="Shift")
        shift=get_from_kwargs(kwargs,"shift",Vector())
        change_default_vector(shift_node,from_value=Vector(),to_value=shift,begin_time=0,transition_time=0)
        change_default_value(slider_node,from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time