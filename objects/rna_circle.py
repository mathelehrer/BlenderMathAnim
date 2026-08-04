"""A ring of RNA, as a b_object.

:class:`~geometry_nodes.modifier_video_brainfuck.RNACircleModifier` as
something a scene can call methods on, the way
:class:`~objects.billiards_new_objects.BilliardBallRound` wraps its own
modifier: a cube whose geometry is thrown away and replaced by the node graph,
with the graph's four dials behind ordinary animation methods.
"""

from geometry_nodes.modifier_video_brainfuck import RNACircleModifier
from interface import ibpy
from mathutils import Vector
from objects.bobject import BObject
from objects.cube import Cube
from utils.constants import DEFAULT_ANIMATION_TIME

#: the keyword arguments :func:`ibpy.customize_material` understands. These are
#: copied on to the modifier, where they reach all six materials the graph
#: paints with, so that ``emission=0.6`` works here as it does everywhere else
#: in this video. Everything else stays with the b_object - and has to, since
#: the modifier hands its keywords to node constructors and a ``location``
#: meant for the object would collide with a node's own.
MATERIAL_KEYS = ("emission", "transmission", "roughness", "ior", "metallic",
                 "brightness", "scatter", "specular_tint", "alpha", "shading",
                 "override_material")


class RNACircle(BObject):
    """A single strand of RNA wound once round a circle, drawn base by base.

    The geometry is entirely
    :class:`~geometry_nodes.modifier_video_brainfuck.RNACircleModifier`'s; the
    cube this is built on is a carrier and is never seen. What this class adds
    is the choreography, so that a scene can say ``ring.appear(...)`` and
    ``ring.scale_strand(...)`` instead of reaching into the node tree for
    sockets by name.

    The graph's dials are wired to arguments here:

    ``BasesPerCircle`` <- ``resolution``
        how many bases a whole lap holds, and therefore how far
        :meth:`appear` ramps ``Progress``.
    ``Radius`` <- ``radius``
        the radius of the circle, and with it the size of the molecule riding
        on it - the graph feeds ``Radius`` to the captured radius too, since
        the radius of curvature of a circle is its own radius.
    ``StrandScale`` <- ``strand_scale``, and :meth:`scale_strand` afterwards.
    ``Progress``
        not an argument at all. It starts at zero - an empty ring - and
        :meth:`appear` and :meth:`grow` are what run it up to ``resolution``.

    ``Progress`` is an integer socket, which blender ramps linearly rather than
    with its usual eased bezier (see :func:`ibpy.change_default_integer`). That
    is the right shape here: the bases arrive at a steady rate rather than
    rushing the middle of the lap.

    **Placement is the object's, not the graph's.** ``location``,
    ``rotation_euler`` and ``scale`` go to :class:`BObject` and end up on the
    object's own transform, as they do for every other instance class. They
    could as easily be written into the graph - it carries ``Translation``,
    ``Rotation`` and ``Scale`` nodes, and they are left at identity for anyone
    who wants to animate an offset there - but an in-graph offset is *not* the
    same thing under a parent. Geometry nodes output lands in the object's
    local space, so a translation inside the graph is multiplied by the
    object's own scale where ``ref_obj.location`` is not. Anything that places
    instances and then shrinks them - :class:`~objects.logo.LogoFromInstances`
    hands each child a ``scale`` of ``1/den`` - would pull the ring in toward
    the origin by exactly that factor while the spheres beside it stayed put.

    :param resolution: bases in a whole lap - the graph's ``BasesPerCircle``.
        The last base lands exactly on the first, so a full lap closes.
    :param radius: radius of the circle - the graph's ``Radius``. The default
        of 1 is :class:`~objects.geometry.sphere.Sphere`'s, so a ring dropped
        into a lattice built for spheres comes out the size of the sphere it
        replaces and the caller only has to set ``scale``.
    :param strand_scale: backbone thickness per unit of radius, before any
        later :meth:`scale_strand`.
    :param base_colors: the four base materials, in ``BaseType`` order.
        Defaults to the modifier's.
    :param strand_color: material of the swept backbone.
    :param location: where the ring stands, before ``location_scale``.
    :param rotation_euler: how it is turned, in radians.
    :param location_scale: what ``location`` is multiplied by, for a caller
        writing positions in the units of some picture rather than in blender's.
        Leave it at 1 under a parent that is already scaled - the parent does
        that job, and doing it twice is what ``location_scale`` was reached for
        when the ring was still being moved inside the graph.
    :param molecule_color: material of the spheres sitting on the backbone.
    :param seed: which strand of RNA this is. Changing it deals the bases again.
    :param kwargs: :class:`BObject`'s - ``name``, ``scale`` and the rest. The
        material ones among them (:data:`MATERIAL_KEYS`) are copied on to the
        modifier as well.
    """

    def __init__(self, resolution=18, radius=1., strand_scale=1.,
                 base_colors=None, strand_color="gray_4",location=Vector(),rotation_euler=Vector(),
                 molecule_color="gray_7", seed=4, location_scale=1,**kwargs):
        self.kwargs = kwargs
        self.name = self.get_from_kwargs('name', 'RNACircle')
        self.resolution = resolution
        self.radius = radius
        self.location_scale = location_scale

        # Progress starts at nothing: the ring is drawn by appear() rather than
        # being there and fading in. The graph's own Translation/Rotation/Scale
        # are left at identity - placement is the object's, see the class
        # docstring
        self.modifier = RNACircleModifier(
            progress=0, bases_per_circle=resolution, radius=radius,
            strand_scale=strand_scale, base_colors=base_colors,
            strand_color=strand_color, molecule_color=molecule_color,
            seed=seed, name=self.name,
            **{key: value for key, value in kwargs.items()
               if key in MATERIAL_KEYS})

        cube = Cube()
        # add_mesh_modifier appends the modifier's materials to the object's
        # slots for us, in palette order - Base0..Base3, Strand, Molecule -
        # behind whatever material the cube already carries
        cube.add_mesh_modifier(type="NODES", node_modifier=self.modifier)

        #: the slots :meth:`change_color` acts on. Found by looking for the
        #: base materials rather than assumed to be 0..3, since the carrier
        #: cube brings a material of its own into slot 0
        bases = self.modifier.materials[:len(self.modifier.base_colors)]
        self.base_slots = [index for index, material
                           in enumerate(cube.ref_obj.data.materials)
                           if material in bases]

        super().__init__(obj=cube.ref_obj, name=self.name,
                         location=[location_scale * v for v in location],
                         rotation_euler=list(rotation_euler), **kwargs)

    # ------------------------------------------------------------------
    def _dial(self, name):
        """The control node called ``name``.

        Looked up in the tree by name rather than through
        :func:`ibpy.get_geometry_node_from_modifier`, which matches on the
        *label* as well: asked for ``StrandScale`` it can answer with the Math
        node in the backbone frame that carries that label rather than with the
        dial itself.
        """
        return self.modifier.tree.nodes[name]

    # ------------------------------------------------------------------
    def _draw(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """Run ``Progress`` from nothing to a whole lap.

        The one ramp that draws the ring: the strand gains a base for every
        station the head passes and its tail never moves, so at the end of it
        the two ends meet.
        """
        ibpy.change_default_integer(self._dial("Progress"), from_value=0, to_value=self.resolution,
                                    begin_time=begin_time,
                                    transition_time=transition_time)
        return begin_time + transition_time

    # ------------------------------------------------------------------
    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               **kwargs):
        """Draw the ring, base by base.

        The object itself is made visible at once and the *drawing* is the
        reveal - fading the alpha over the same window as well would leave the
        first bases translucent for as long as the last ones take to arrive.

        :return: the time the ring is closed.
        """
        super().appear(begin_time=begin_time, transition_time=0, **kwargs)
        return self._draw(begin_time=begin_time,
                          transition_time=transition_time)

    # ------------------------------------------------------------------
    def grow(self, scale=None, begin_time=0,
             transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """Scale the object up from nothing *and* draw the ring.

        Both, because both are wanted. :class:`BObject`'s ``grow`` is what
        callers that place a lot of instances use to bring them on - and
        :meth:`~objects.logo.LogoFromInstances.grow` passes each child its own
        ``scale``, so a ring that ignored it would be the one thing in the
        picture that did not grow. Growing a strand of RNA also means gaining
        bases, which is :meth:`_draw`'s ramp, so it does that over the same
        window.

        :param scale: the size to end at. ``None`` means the object's own
            ``intrinsic_scale``, which is what :class:`BObject` does.
        :return: the time the ring is closed.
        """
        # BObject.grow calls self.appear() when the object has not appeared
        # yet, and this class's appear() *draws the ring* as a side effect -
        # at its own begin_time of 0, which would leave a stray pair of
        # Progress keys at the head of the shot. Getting the object on screen
        # here, through BObject's appear rather than this one's, leaves grow()
        # nothing to trigger.
        BObject.appear(self, begin_time=begin_time, transition_time=0)
        return self._draw(begin_time=begin_time,
                          transition_time=transition_time)

    # ------------------------------------------------------------------
    def change_color(self, new_color, slot=None, begin_time=0,
                     transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """Recolour the bases.

        Unlike :meth:`BObject.change_color` this acts on *all* of the base
        slots by default rather than on slot 0, since the four bases are one
        thing on screen - the four colours that say which base is which - and
        turning them one at a time is not usually what is wanted. It is what
        the end of the logo shot does, where the molecule stops being a
        molecule and becomes the drawing.

        The backbone and the beads keep their own materials; pass an explicit
        ``slot`` to reach those.

        :param new_color: the material to fade into.
        :param slot: a single slot, or a list of them, instead of the bases.
        :return: the time the colour has arrived.
        """
        if slot is None:
            slots = self.base_slots
        elif isinstance(slot, int):
            slots = [slot]
        else:
            slots = list(slot)

        for index in slots:
            super().change_color(new_color, slot=index, begin_time=begin_time,
                                 transition_time=transition_time, **kwargs)
        return begin_time + transition_time

    # ------------------------------------------------------------------
    def scale_strand(self, from_value=1.0, to_value=0.25, begin_time=0,
                     transition_time=DEFAULT_ANIMATION_TIME):
        """Thin the backbone down, or fatten it up.

        ``StrandScale`` multiplies the captured radius, and the product drives
        both the swept tube and the beads sitting on it - so this is the whole
        backbone getting thinner while the bases stay the size they were.

        Pass ``from_value=None`` to leave whatever key is already there alone
        and only write the end of the move, which is how consecutive beats are
        chained without flattening the motion to a stop between them.

        :param from_value: the thickness to start from, or ``None``.
        :param to_value: the thickness to end at.
        :return: the time the move is over.
        """
        return ibpy.change_default_value(self._dial("StrandScale"),
                                         from_value=from_value,
                                         to_value=to_value,
                                         begin_time=begin_time,
                                         transition_time=transition_time)
