"""The slide as one object.

:class:`~orbi.slide_modifier.SlideModifier` knows the shape of a slide and,
more to the point, where its rungs are. What it does not know is where the
slide *is* - that is a blender object's job, and every scene so far has done
it by hand: a :class:`~objects.cube.Cube` to stand the modifier on, an
``add_mesh_modifier``, and then the moving and the growing addressed to the
cube while every question about rungs went to the modifier. Two names for one
slide, and the scene had to remember which of them to ask for what.

``Slide`` is that pair as a single :class:`~objects.bobject.BObject`. It
moves, rotates and grows because it is a ``BObject``; it answers ``rung``,
``hold``, ``up`` and ``rung_spacing`` because it forwards them to the modifier
it carries. The forwarding is the point, not a convenience: it is what lets a
``Slide`` be handed straight to
:meth:`~orbi.orbi_modifier.OrbiModifier.fit_slide`, which takes the climb's
whole arithmetic off whatever it is given by reading exactly those names. The
object and the modifier are one slide as far as the climb is concerned, which
is the only way the creature's hands and the geometry's rungs can be kept from
drifting apart.

The host is a cube that is never seen. The modifier's tree has no group input,
so the cube's own geometry never reaches the output - it is somewhere for the
modifier to stand and nothing else. The color therefore has to be the
modifier's (see:data:`DEFAULT_MATERIAL`) rather than the host's, exactly as
it is for the creature in:mod:`orbi.orbi`.

Why the kwargs are split
------------------------
A ``BObject``'s parameters and the modifier's share a name, and it is the one
name that would hurt: ``location``. A ``BObject`` is *placed* by it, and every
geometry node is *drawn* by it - ``create_node`` hands its kwargs on to the
nodes it builds, so a ``location`` meant for the object arrives at a
``Set Material`` that is already being given one, and the slide fails to
build at all. So the modifier's parameters are taken out of the kwargs by
name, and the names are read off the modifier's own signature rather than
copied out here: a parameter added to :class:`SlideModifier` arrives in this
class for free, and cannot be forgotten.
"""
from geometry_nodes.modifier_objects import SlideModifier
from objects.bobject import BObject
from objects.cube import Cube


#: what a slide is made of unless a scene says otherwise. The rails take the
#: same ink as the creature's bones, and the rungs are picked out against them
#: so that the part the creature is actually holding is the part one sees.
#: A slide built without these renders in blender's default grey, since the
#: geometry leaves the node tree without a material and the host's slots are
#: not consulted.

DEFAULT_MATERIAL = {"material":"plastic_example"}
SLIDE_PARAMETERS = {"radius":3.0, "width":2.6, "start":0.55, "depth":0.4}


class Slide(BObject):
    """A cube to stand in for, and a ``SlideModifier`` to be."""

    def __init__(self, name="Slide", **kwargs):
        """
        :param name: the name of the host object, and of the node group.
        :param kwargs: the slide's own shape and paint - ``height``,
            ``width``, ``lean``, ``rung_spacing``, ``rail_radius``,
            ``rung_radius``, ``overshoot``, ``resolution``, ``material``,
            ``rung_material`` - go to :class:`SlideModifier`; everything
            else (``location``, ``rotation_euler``, ``scale``, ...) goes to
            :class:`BObject`.
        """
        shape = {key: kwargs.pop(key)
                 for key in list(kwargs) if key in SLIDE_PARAMETERS}
        self.slide = SlideModifier(name=name, **{**DEFAULT_MATERIAL, **shape})

        # the cube is built first and only its blender object is kept: the
        # modifier has to hang on the object this class *is*, not on a second
        # one parented to it, or a scene moving the Slide would leave the
        # slide behind. ``no_material`` on both halves because the cube is
        # never rendered and the slide is painted inside the tree - a
        # material here would only be a copy of the default nobody ever sees
        host = Cube(name=name + "Host", no_material=True)
        host.add_mesh_modifier(type='NODES', node_modifier=self.slide)
        super().__init__(obj=host.ref_obj, name=name, no_material=True, **kwargs)

    #: what a scene asks a *slide* rather than an *object*: how big it is,
    #: where its rungs are, and which way is up them. They are forwarded
    #: rather than written out because every one of them would be the same
    #: one-line hand-off, and because ``fit_slide`` must not be able to tell
    #: a ``Slide`` from a ``SlideModifier`` - it reads ``climb_stride``,
    #: ``up``, ``lean``, ``rung_spacing`` and ``rung_count`` off whatever it
    #: is handed, and a name missing here is a climb posed against a slide
    #: that is not the one in frame.
    FORWARDED = ("descent","duration","arc_length","omega","fits")

    def __getattr__(self, name):
        """Delegate the slide's own questions to the modifier.

        ``__getattr__`` is only consulted for names that are *not* found the
        ordinary way, so this shadows nothing: ``move``, ``rotate``, ``grow``
        and ``appear`` are still :class:`BObject`'s.

        :raise AttributeError: for anything not in :data:`FORWARDED`, so that
            a typo stays a typo rather than becoming a silent hand-off.
        """
        # the modifier is looked up in the instance dictionary rather than by
        # attribute: reaching it by attribute before ``__init__`` has set it
        # would come straight back here and recur for ever
        slide = self.__dict__.get("slide")
        if slide is not None and name in Slide.FORWARDED:
            return getattr(slide, name)
        raise AttributeError("%r object has no attribute %r"
                             % (type(self).__name__, name))
