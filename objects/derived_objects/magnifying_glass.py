import numpy as np

from interface import ibpy
from interface.ibpy import Vector
from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.disc import Disc
from objects.torus import Torus
from utils.constants import DEFAULT_ANIMATION_TIME


class MagnifyingGlass(BObject):
    """A magnifying glass whose lens shows a picture of its own.

    The lens does not magnify what lies behind it; it is a window onto an
    image - typically a close-up of the thing it hovers over, such as the
    inside of a node group held over the group node. Which part of the
    image the window shows, and how much of it, can be animated, so the
    picture can be scrolled past under the lens. Until it is switched on
    (:meth:`switch_on`) the lens is clear glass and lets through what lies
    behind it.

    Rim, lens and handle are children of an empty that carries the
    location and rotation, so the glass moves as one with :meth:`move_to`.
    It is built standing in the x-z plane, facing the camera along -y,
    with the handle pointing down to the right.

    The window is a circle of ``window`` image heights across, centred on
    ``(u, v)`` in fractions of the image (``(0, 0)`` its lower left corner,
    ``(1, 1)`` its upper right). Beyond the image's border the texture
    repeats the border pixels, so a window larger than the image shows a
    screenshot's background colour there rather than nothing.

    :param src: image file in ``media/raster``.
    :param radius: radius of the lens.
    :param window: diameter of the visible window in image heights; 1 shows
        the full height of the image across the lens.
    :param center: ``(u, v)`` the window starts centred on.
    :param handle_angle: direction of the handle in the x-z plane, in radians
        from +x.
    :param handle_length: length of the handle, ``1.4 * radius`` by default.
    :param rim_color: color of rim.
    :param handle_color: color of the handle.
    :param emission: emission strength of the lens image.
    :param on: whether the lens shows its image from the start.

    Example::

        glass = MagnifyingGlass("bessel_node1.png", radius=1.6, window=1.2,
                                center=(0.1, 0.5), location=[-3.8, -0.5, 2.4])
        glass.appear(begin_time=0, transition_time=0.3)
        glass.switch_on(begin_time=1)
        glass.look_at(0.9, 0.5, begin_time=1.5, transition_time=10)
    """

    def __init__(self, src, radius=1, window=1, center=(0.5, 0.5),
                 handle_angle=-np.pi / 4, handle_length=None,
                 rim_color="text", handle_color="drawing", emission=1, on=False, **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name', 'MagnifyingGlass')
        if handle_length is None:
            handle_length = 1.4 * radius

        upright = [np.pi / 2, 0, 0]
        self.lens = Disc(radius=radius, resolution=64, rotation_euler=upright,
                         color="image", src=src, emission=emission, name=name + "Lens")
        self.rim = Torus(major_radius=radius, minor_radius=0.06 * radius,
                         major_segments=96, minor_segments=16, rotation_euler=upright,
                         color=rim_color, name=name + "Rim")
        direction = Vector([np.cos(handle_angle), 0, np.sin(handle_angle)])
        self.handle = Cylinder.from_start_to_end(start=1.04 * radius * direction,
                                                 end=(radius + handle_length) * direction,
                                                 radius=0.09 * radius, color=handle_color,
                                                 name=name + "Handle")

        # Generated coordinates put the disc's bounding box on [0,1]^2, the
        # Mapping node turns that into the window: scale = its size as a
        # fraction of the image, location = its lower left corner
        nodes = ibpy.get_material_of(self.lens).node_tree.nodes
        self.mapping = next(node for node in nodes if node.type == 'MAPPING')
        image = next(node for node in nodes if node.type == 'TEX_IMAGE')
        image.extension = 'EXTEND'

        # the switch: the image's alpha, scaled by a factor before it reaches
        # the material's AlphaFactor (which appear/disappear keep for fading)
        tree = ibpy.get_material_of(self.lens).node_tree
        alpha_factor = next(node for node in nodes if node.label == 'AlphaFactor')
        self.switch = tree.nodes.new('ShaderNodeMath')
        self.switch.operation = 'MULTIPLY'
        self.switch.label = 'Switch'
        self.switch.location = (-650, -400)
        self.switch.inputs[1].default_value = 1 if on else 0
        tree.links.new(image.outputs['Alpha'], self.switch.inputs[0])
        tree.links.new(self.switch.outputs[0], alpha_factor.inputs[1])
        self.aspect = image.image.size[0] / image.image.size[1]
        self.window = window
        self.center = Vector(center)
        self.mapping.inputs['Scale'].default_value = self.window_scale(window)
        self.mapping.inputs['Location'].default_value = self.window_corner(self.center, window)

        super().__init__(children=[self.lens, self.rim, self.handle], name=name,
                         no_material=True, **kwargs)

    def switch_on(self, begin_time=0, transition_time=0.5):
        """Fade the image in on the lens, which was clear glass until then.

        :return: the time the lens is fully on.
        """
        return ibpy.change_default_value(self.switch.inputs[1], 0, 1, begin_time=begin_time,
                                         transition_time=transition_time)

    def switch_off(self, begin_time=0, transition_time=0.5):
        """Fade the image out again, back to clear glass.

        :return: the time the lens is fully off.
        """
        return ibpy.change_default_value(self.switch.inputs[1], 1, 0, begin_time=begin_time,
                                         transition_time=transition_time)

    def window_scale(self, window):
        """Scale of the Mapping node for a window ``window`` image heights across."""
        return Vector([window / self.aspect, window, 1])

    def window_corner(self, center, window):
        """Lower left corner, in fractions of the image, of the window centred on ``center``."""
        return Vector([center[0] - window / self.aspect / 2, center[1] - window / 2, 0])

    def look_at(self, u, v, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """Slide the window until it is centred on ``(u, v)``.

        :return: the time the slide ends.
        """
        center = Vector([u, v])
        ibpy.change_default_value(self.mapping.inputs['Location'],
                                  self.window_corner(self.center, self.window),
                                  self.window_corner(center, self.window),
                                  begin_time=begin_time, transition_time=transition_time)
        self.center = center
        return begin_time + transition_time

    def zoom(self, window, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """Change the window to ``window`` image heights across, about its current centre.

        :return: the time the zoom ends.
        """
        ibpy.change_default_value(self.mapping.inputs['Scale'],
                                  self.window_scale(self.window), self.window_scale(window),
                                  begin_time=begin_time, transition_time=transition_time)
        ibpy.change_default_value(self.mapping.inputs['Location'],
                                  self.window_corner(self.center, self.window),
                                  self.window_corner(self.center, window),
                                  begin_time=begin_time, transition_time=transition_time)
        self.window = window
        return begin_time + transition_time
