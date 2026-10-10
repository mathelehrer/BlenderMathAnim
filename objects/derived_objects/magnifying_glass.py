import numpy as np

from interface import ibpy
from interface.ibpy import Vector
from objects.bobject import BObject
from objects.cylinder import Cylinder
from objects.disc import Disc
from objects.torus import Torus
from utils.constants import DEFAULT_ANIMATION_TIME


class MagnifyingGlass(BObject):
    """A magnifying glass that magnifies an image plane, and can show a picture of its own.

    Held over ``background`` - a plane wearing an image material - the lens
    shows that image magnified ``magnification`` times about the lens
    centre, wherever the glass is moved. Off the plane's image it is clear.
    The magnification is done in the lens shader, not by refraction: the
    lens samples the background's image at the point behind it, pulled
    towards the lens centre by ``1/magnification``. That renders the same
    in EEVEE and Cycles and on a transparent film, but it magnifies only
    that one image, not other objects behind the glass. The background has
    to face the same way as the glass and must not be scaled on the object.
    Without a background the lens is plain clear glass.

    Switched on (:meth:`switch_on`), the lens crossfades to a picture of
    its own, ``src`` - typically a close-up of the thing it hovers over,
    such as the inside of a node group held over the group node. Which part
    of the picture the lens shows, and how much of it, can be animated, so
    the picture can be scrolled past under the lens.

    Rim, lens and handle are children of an empty that carries the
    location and rotation, so the glass moves as one with :meth:`move_to`.
    It is built standing in the x-z plane, facing the camera along -y,
    with the handle pointing down to the right.

    The window is a circle of ``window`` image heights across, centred on
    ``(u, v)`` in fractions of the image (``(0, 0)`` its lower left corner,
    ``(1, 1)`` its upper right). Beyond the image's border the texture
    repeats the border pixels, so a window larger than the image shows a
    screenshot's background colour there rather than nothing.

    :param src: image file in ``media/raster`` shown when switched on.
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
    :param on: whether the lens shows ``src`` from the start.
    :param background: the image plane (a BObject) the lens magnifies.
    :param magnification: how many times the lens magnifies the background.

    Example::

        tree = Plane(u=[-3, 3], v=[-1, 1], rotation_euler=[pi / 2, 0, 0],
                     color="image", src="bessel_node0.png", resolution=1)
        glass = MagnifyingGlass("bessel_node1.png", radius=1.6, window=1.2,
                                center=(0.1, 0.5), background=tree, magnification=2,
                                location=[-3.8, -0.5, 2.4])
        glass.appear(begin_time=0, transition_time=0.3)
        glass.switch_on(begin_time=1)
        glass.look_at(0.9, 0.5, begin_time=1.5, transition_time=10)
    """

    def __init__(self, src, radius=1, window=1, center=(0.5, 0.5),
                 handle_angle=-np.pi / 4, handle_length=None,
                 rim_color="text", handle_color="drawing", emission=1, on=False,
                 background=None, magnification=2, **kwargs):
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

        tree = ibpy.get_material_of(self.lens).node_tree
        links = tree.links

        # the magnified background: a point q of the lens, in the lens' own
        # frame, sits in front of the background point p (in the plane's
        # frame); its magnified image is p - q (1 - 1/M). Turned into the
        # plane's Generated coordinates, i.e. its bounding box mapped to [0,1]^2
        behind = tree.nodes.new('ShaderNodeTexImage')
        behind.location = (-900, 400)
        behind.extension = 'CLIP'
        if background is not None:
            plane = ibpy.get_obj(background)
            behind.image = next(node.image for node in plane.active_material.node_tree.nodes
                                if node.type == 'TEX_IMAGE')
            corners = [Vector(c) for c in plane.bound_box]
            low = Vector([min(c.x for c in corners), min(c.y for c in corners), 0])
            size = Vector([max(c.x for c in corners) - low.x, max(c.y for c in corners) - low.y, 1])

            on_plane = tree.nodes.new('ShaderNodeTexCoord')
            on_plane.object = plane
            on_lens = tree.nodes.new('ShaderNodeTexCoord')
            pull = tree.nodes.new('ShaderNodeVectorMath')
            pull.operation = 'SCALE'
            pull.inputs['Scale'].default_value = 1 - 1 / magnification
            seen = tree.nodes.new('ShaderNodeVectorMath')
            seen.operation = 'SUBTRACT'
            shifted = tree.nodes.new('ShaderNodeVectorMath')
            shifted.operation = 'SUBTRACT'
            shifted.inputs[1].default_value = low
            generated = tree.nodes.new('ShaderNodeVectorMath')
            generated.operation = 'DIVIDE'
            generated.inputs[1].default_value = size
            for i, node in enumerate([on_plane, on_lens, pull, seen, shifted, generated]):
                node.location = (-2600 + 280 * i, 600 - 120 * (i % 2))
            links.new(on_lens.outputs['Object'], pull.inputs[0])
            links.new(on_plane.outputs['Object'], seen.inputs[0])
            links.new(pull.outputs['Vector'], seen.inputs[1])
            links.new(seen.outputs['Vector'], shifted.inputs[0])
            links.new(shifted.outputs['Vector'], generated.inputs[0])
            links.new(generated.outputs['Vector'], behind.inputs['Vector'])

        # the switch: 0 shows the magnified background, 1 the lens' own
        # picture. Colour and alpha are crossfaded; the alpha goes on into the
        # material's AlphaFactor, which appear/disappear keep for fading
        self.switch = tree.nodes.new('ShaderNodeValue')
        self.switch.label = 'Switch'
        self.switch.location = (-1100, -600)
        self.switch.outputs[0].default_value = 1 if on else 0
        color = tree.nodes.new('ShaderNodeMix')
        color.data_type = 'RGBA'
        color.location = (-650, 200)
        alpha = tree.nodes.new('ShaderNodeMix')
        alpha.data_type = 'FLOAT'
        alpha.location = (-650, -400)
        for target in [link.to_socket for link in tree.links
                       if link.from_socket == image.outputs['Color']]:
            links.new(color.outputs[2], target)
        alpha_factor = next(node for node in nodes if node.label == 'AlphaFactor')
        links.new(alpha.outputs[0], alpha_factor.inputs[1])
        links.new(self.switch.outputs[0], color.inputs['Factor'])
        links.new(self.switch.outputs[0], alpha.inputs['Factor'])
        links.new(behind.outputs['Color'], color.inputs[6])
        links.new(image.outputs['Color'], color.inputs[7])
        links.new(behind.outputs['Alpha'], alpha.inputs[2])
        links.new(image.outputs['Alpha'], alpha.inputs[3])

        self.aspect = image.image.size[0] / image.image.size[1]
        self.window = window
        self.center = Vector(center)
        self.mapping.inputs['Scale'].default_value = self.window_scale(window)
        self.mapping.inputs['Location'].default_value = self.window_corner(self.center, window)

        super().__init__(children=[self.lens, self.rim, self.handle], name=name,
                         no_material=True, **kwargs)

    def switch_on(self, begin_time=0, transition_time=0.5):
        """Crossfade the lens from the magnified background to its own picture.

        :return: the time the lens is fully on.
        """
        return ibpy.change_default_value(self.switch, 0, 1, begin_time=begin_time,
                                         transition_time=transition_time)

    def switch_off(self, begin_time=0, transition_time=0.5):
        """Crossfade back from the lens' own picture to the magnified background.

        :return: the time the lens is fully off.
        """
        return ibpy.change_default_value(self.switch, 1, 0, begin_time=begin_time,
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
