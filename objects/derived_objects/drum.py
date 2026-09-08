from appearance.textures import get_texture
from objects.bobject import BObject

import numpy as np

from geometry_nodes.nodes import (BESSEL_OPS, CombineXYZ, CompareNode, Frame, Grid, IndexSwitch, InputInteger,
                                  MathNode, MergeByDistance, NamedAttribute, Reroute, SceneTime,
                                  SetMaterial, SetPosition,
                                  SetShadeSmooth, StoreNamedAttribute, Switch, WireFrame, bessel_jm_rpn,
                                  create_geometry_line)

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import DeleteGeometry, InputValue, Position, make_function, RotateVector
from interface import ibpy
from interface.ibpy import get_geometry_node_from_modifier
from mathutils import Vector
from utils.constants import DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs


pi = np.pi
tau = 2 * pi


class DrumModifier(GeometryNodesModifier):
    r"""Every face above a plane and inside a cylinder, thrown away.

    One ``Delete Geometry``, whose selection is the intersection of a half
    space and a cylinder about the same axis,

    .. math::
        z > h \qquad\text{and}\qquad \sqrt{x^2 + y^2} < R ,

    with :math:`h` the ``Height`` dial and :math:`R` the ``Radius`` one. On a
    drum lying with its axis along z that is exactly the membrane: the skin is
    the part of the shell that is both above the mid-plane and *inside* the
    barrel, while the wall of the barrel sits at the drum's full radius and
    the rim wraps around it, so both survive a cut at any R short of that
    radius. What is left is an open pot with a rim - and a hole to put a
    surface in::

        drum = Drum(rotation_euler=[pi / 2, 0, 0])
        skin = DrumModifier(axis="y")               # see "Which space" below
        drum.add_mesh_modifier(type='NODES', node_modifier=skin)

        t0 = skin.set_radius(0.9, begin_time=t0, transition_time=1)
        membrane = Plane(name="Membrane", u=[-1, 1], v=[-1, 1], resolution=1)
        membrane.add_mesh_modifier(type='NODES',
                                   node_modifier=DrumModeModifier(radius=0.9))

    **Why the radius is a dial and not a number.** Deleting the membrane once
    is a thing that could be done by hand in the mesh, and then it would be
    gone from the first frame of the video. Keeping it on ``Radius`` makes the
    removal an *event*: at :math:`R = 0` the test :math:`r < R` is nowhere
    true, nothing is selected and the drum is whole; ramping R out to the
    drum's radius opens the hole like an iris and the skin peels away from the
    middle outwards. That is why ``radius`` starts at 0 - a modifier that
    changed the object the moment it was added would have to be animated
    backwards to show the drum intact first.

    **Which space the cut is in.** ``Position`` is the object's *own* local
    coordinates, which is the mesh as it was modelled, not as it stands in the
    scene. The bundled ``Drum.blend`` was modelled lying on its side: its two
    skins are at :math:`y = \pm 0.24` and its radius of about 0.95 is in the
    x-z plane, and it is the scene's ``rotation_euler=[pi/2, 0, 0]`` that
    stands it up. So the drum of :class:`~objects.derived_objects.drum.Drum`
    wants ``axis="y"``, and ``axis="z"`` - the default, and the axis the
    formula above is written in - is for a drum that is modelled upright.
    :meth:`bounds` measures a given object and says which it is.

    **Faces, not points.** ``domain="FACE"`` evaluates the test at face
    centres, so a face is removed when its *middle* is inside the cylinder and
    a face straddling the boundary stays. The cut therefore runs about half a
    polygon wide of R and the ring of faces that bridges skin and rim is kept,
    which is what keeps the hole's edge tidy on a dense mesh and, more to the
    point, keeps the drum watertight where it is not cut. ``domain="POINT"``
    is the other reading: it deletes a vertex and with it every face that used
    it, so the hole comes out about half a polygon *bigger* than R and no face
    ever crosses the boundary. Either way ``mode="ALL"``, so the vertices and
    edges that only the deleted faces used go too and nothing is left behind
    as loose geometry.

    The dials, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, label)`` (:meth:`dial`):

    ``Radius``
        R, how far out into the x-y plane the removal reaches. 0 removes
        nothing; anything past the drum's own radius takes the wall with it.
        :meth:`set_radius` keyframes it.
    ``Height``
        h, the plane the removal starts at. 0 is the mid-plane of a drum
        modelled about its centre, which is enough to tell the two skins
        apart; raising it towards the skin is only necessary if something else
        of the model reaches above the middle inside the barrel.

    :param name: name of the node group, and of the modifier in the stack.
    :param radius: what ``Radius`` starts at. 0, so that adding the modifier
        does not change the object until it is animated.
    :param height: what ``Height`` starts at, in the object's local units.
    :param axis: which way is up for this mesh - ``"z"`` (the default),
        ``"x"``, ``"y"`` or one of their negatives. The half space that is
        removed is the one the axis points into, and the radius is measured
        in the plane perpendicular to it, so ``"-z"`` takes the *bottom* skin
        off instead of the top one.
    :param domain: ``"FACE"`` (the default) or ``"POINT"``; see above.
    :param kwargs: passed on to
        :class:`~geometry_nodes.geometry_nodes_modifier.GeometryNodesModifier`.
    """

    #: the axis name -> (sign, axial component, the two perpendicular ones)
    _AXES = {"x": (1, "x", ("y", "z")), "y": (1, "y", ("z", "x")),
             "z": (1, "z", ("x", "y")), "-x": (-1, "x", ("y", "z")),
             "-y": (-1, "y", ("z", "x")), "-z": (-1, "z", ("x", "y"))}

    def __init__(self, name="DrumModifier", radius=0.0, height=0.0, axis="z",
                 domain="FACE", **kwargs):
        self.axis = self.axis_name(axis)
        self.radius = radius
        self.height = height
        self.domain = domain
        # filled in by create_node, so that set_radius has the dial to
        # keyframe without going back through the tree by name
        self.radius_dial = None
        self.height_dial = None
        super().__init__(name=name, automatic_layout=True,
                         group_input=True, group_output=True, **kwargs)

    # ------------------------------------------------------------------
    @classmethod
    def axis_name(cls, axis):
        """The axis as it is stored, from its name. Raises on anything else."""
        key = str(axis).strip().lower()
        if key not in cls._AXES:
            raise ValueError("drum axis %r is not one of %s"
                             % (axis, sorted(cls._AXES)))
        return key

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        geometry = self.group_inputs.outputs["Geometry"]

        position = Position(tree, location=(-4, 0), name="DrumPosition",
                            hide=True)
        rot = get_from_kwargs(kwargs, "rotation_euler", Vector())
        rotation = RotateVector(tree, location=(-3, 0), name="RotatePosition", rotation=rot, hide=True,
                                vector=position.std_out)
        self.radius_dial = InputValue(tree, location=(-4, -1),
                                      value=self.radius, name="Radius")
        self.height_dial = InputValue(tree, location=(-4, -2),
                                      value=self.height, name="Height")

        sign, axial, (first, second) = self._AXES[self.axis]
        # the axial coordinate, turned around for a negative axis so that
        # "above the plane" is always the side the axis points into
        up = "pos_%s" % axial if sign > 0 else "pos_%s,-1,*" % axial
        # the two halves of the selection, and-ed as booleans: ">" and "<" are
        # ShaderNodeMath comparisons and leave a float, which blender reads as
        # a boolean on the way into FunctionNodeBooleanMath (and would read the
        # same way straight into Selection - the house style of Slicer and of
        # the other culls). "cut" is declared a boolean so that the group's
        # own output socket is one too.
        selection = make_function(
            tree, location=(-2, -1), name="MembraneSelection",
            aux_functions={"rad": "pos_%s,pos_%s,*,pos_%s,pos_%s,*,+,sqrt"
                                  % (first, first, second, second)},
            functions={"cut": "%s,height,>,rad,radius,<,and" % up},
            inputs=["pos", "radius", "height"], outputs=["cut"],
            vectors=["pos"], scalars=["radius", "height", "rad"],
            booleans=["cut"])
        tree.links.new(rotation.std_out, selection.inputs["pos"])
        tree.links.new(self.radius_dial.std_out, selection.inputs["radius"])
        tree.links.new(self.height_dial.std_out, selection.inputs["height"])

        hole = DeleteGeometry(tree, location=(0, 0), domain=self.domain,
                              mode="ALL", geometry=geometry,
                              selection=selection.outputs["cut"],
                              name="MembraneCut")
        tree.links.new(hole.geometry_out, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def dial(self, label="Radius"):
        """The node a scene animates, by the name the tree gave it."""
        return get_geometry_node_from_modifier(self, label)

    def set_radius(self, to_value, from_value=None, begin_time=0,
                   transition_time=DEFAULT_ANIMATION_TIME):
        """Open (or close) the hole to ``to_value`` and return when it is there.

        ``from_value=None`` picks the hole up wherever the last call left it
        (``radius`` at first), which is what makes a sequence of calls read as
        one continuous opening. It is *not* passed on as ``None``:
        :func:`ibpy.change_default_value` writes no keyframe at all for a
        start it is not given, and the dial would then hold ``to_value`` from
        the first frame of the scene rather than grow to it.

        :return: ``begin_time + transition_time``, the project's convention
            for chaining a scene's ``t0``.
        """
        if from_value is None:
            from_value = self.radius
        self.radius = to_value
        return ibpy.change_default_value(self.radius_dial.std_out,
                                         from_value=from_value,
                                         to_value=to_value,
                                         begin_time=begin_time,
                                         transition_time=transition_time)

    def set_height(self, to_value, from_value=None, begin_time=0,
                   transition_time=DEFAULT_ANIMATION_TIME):
        """Move the plane the removal starts at, the way :meth:`set_radius`
        moves the cylinder. Ramping it *down* through the drum peels the skin
        off from the rim inwards instead of from the middle outwards."""
        if from_value is None:
            from_value = self.height
        self.height = to_value
        return ibpy.change_default_value(self.height_dial.std_out,
                                         from_value=from_value,
                                         to_value=to_value,
                                         begin_time=begin_time,
                                         transition_time=transition_time)

    # ------------------------------------------------------------------
    def bounds(self, bob):
        """What ``Radius`` and ``Height`` have to be for this object.

        Measured on the object's own vertices in its own local coordinates -
        the space the cut lives in - and along the axis this modifier was
        built with, so the numbers can be read straight into the dials:

        * the **radius** is how far the mesh reaches from the axis. A drum's
          skin stops just short of it (the wall is the widest part), so a
          ``Radius`` a few per cent under this takes the membrane and leaves
          the barrel, and one above it takes everything.
        * **low** and **high** are how far the mesh reaches along the axis.
          ``Height`` between 0 and ``high`` cuts the near skin only; below
          ``low`` it would cut both.

        Running it on a drum also says which way the model lies: the axis of a
        drum is its *short* direction, so if ``high - low`` comes out about as
        large as the radius, this modifier is built on the wrong axis.

        :return: ``(radius, low, high)``.
        """
        obj = ibpy.get_obj(bob)
        vertices = getattr(getattr(obj, "data", None), "vertices", None)
        if vertices:
            points = np.array([list(vertex.co) for vertex in vertices])
        else:
            # a curve, a text, an empty - the bounding box is all there is,
            # and its corners overstate the radius by up to sqrt(2)
            points = np.array([list(corner) for corner in obj.bound_box])

        sign, axial, (first, second) = self._AXES[self.axis]
        column = {"x": 0, "y": 1, "z": 2}
        radius = np.hypot(points[:, column[first]], points[:, column[second]])
        along = sign * points[:, column[axial]]
        return float(radius.max()), float(along.min()), float(along.max())


# ---------------------------------------------------------------------------
#  A DRUM
# ---------------------------------------------------------------------------
#
#: Zeros :math:`\alpha_{mn}` of :math:`J_m` - ``scipy.special.jn_zeros(m, 4)``,
#: to six decimals. A membrane clamped at r = a can only vibrate at radii that
#: put a zero of the Bessel function on the rim, which is what makes this table
#: the spectrum of a drum: the mode (m, n) has k = alpha_mn / a, and the
#: frequencies are those numbers divided by alpha_01 - 1, 1.593, 2.136, 2.296,
#: ... - a series that is not harmonic, which is why a drum has no pitch the
#: way a string does.
BESSEL_ZEROS = {0: (2.404826, 5.520078, 8.653728, 11.791534),
                1: (3.831706, 7.015587, 10.173468, 13.323692),
                2: (5.135622, 8.417244, 11.619841, 14.795952),
                3: (6.380162, 9.761023, 13.015201, 16.223466),
                4: (7.588342, 11.064709, 14.372537, 17.615966)}

#: max |J_m|, which every mode of that order reaches at its first extremum.
#: Dividing by it is what makes ``Amplitude`` mean the height of the crest for
#: every mode rather than only for the fundamental - J_3 peaks at 0.43, so an
#: un-normalised (3,1) would stand less than half as tall as (0,1) on the same
#: dial and the switch would look like a fade.
_BESSEL_PEAKS = {0: 1.0, 1: 0.581865, 2: 0.486499, 3: 0.434394, 4: 0.399652}


class DrumModeModifier(GeometryNodesModifier):
    r"""A clamped disc standing up as one normal mode of a drum.

    :class:`WaveVisualizationModifier` is this modifier without a boundary: a
    source radiates and the field runs off to infinity, so what it shows is
    :math:`H^{(1)}_0`, an outgoing wave. Put a rim on it - clamp the membrane
    at r = a, u(a, t) = 0 - and nothing runs off any more. The wave that comes
    back interferes with the wave going out, only certain frequencies survive
    it, and the solutions of

    .. math::
        \frac{\partial^2 u}{\partial t^2} = c^2\nabla^2 u ,
        \qquad u\big|_{r=a} = 0

    are standing waves, one for each pair of integers:

    .. math::
        u_{mn}(r, \varphi, t) = A\,J_m\!\Big(\alpha_{mn}\frac{r}{a}\Big)
                                \cos m\varphi\,\cos\omega_{mn}t ,
        \qquad \omega_{mn} = \frac{c\,\alpha_{mn}}{a},

    with :math:`\alpha_{mn}` the n-th zero of :math:`J_m` (:data:`BESSEL_ZEROS`).
    The boundary condition *is* that table: the rim can only be held still if a
    zero of the Bessel function lands on it. So **m counts nodal diameters**
    (the lines through the centre that never move, where :math:`\cos m\varphi`
    vanishes) and **n counts nodal circles** (the rings, where :math:`J_m`
    does), and the mode is completely described by saying how many of each.

    Two things about this that the flat scenes cannot show, and this one is
    built to:

    The overtones are **not harmonic**. A string's modes go 1, 2, 3, ...; a
    drum's go :math:`\alpha_{mn}/\alpha_{01}` = 1, 1.593, 2.136, 2.296, 2.653,
    ... The ``Frequency`` dial is the *fundamental*'s, and every mode takes its
    own multiple of it from the table, so switching modes while the surface
    moves also changes the pitch - inaudibly, but visibly, since the (2,1) mode
    beats more than twice as fast as (0,1).

    And a mode with :math:`m>0` has **no motion at the centre**:
    :math:`J_m(0)=0` for every m but zero. The fundamental is a single hill
    rising and falling; (1,1) is a see-saw about a diameter; (2,1) is a
    quadrupole. That is the picture the switch is for.

    **The switch.** Every mode in ``modes`` is built into the tree as a
    function group of its own, and an ``Index Switch`` on the ``Mode`` dial
    picks which one reaches the geometry::

        drum = DrumModeModifier(name="Drum", modes=((0, 1), (1, 1), (2, 1)))
        drum.set_mode((1, 1), begin_time=6)      # or set_mode(1, ...)

    An integer switch is a hard cut, which is what makes it read as *this mode
    now, that mode next* rather than as a morph; the price is that all of the
    modes are evaluated per vertex and only one of them is used, since a field
    has no branches. Six modes on a 100 x 180 mesh is what the default costs,
    and it is the number to bring down first if the viewport goes sticky.

    The tree is four frames, one method each:

    :meth:`_control_frame`
        the dials and the clock - ``Radius``, ``Amplitude``, ``Frequency``,
        ``Mode``, ``StartTime``, and ``Scene Time -> Seconds``, which is why
        the drum moves without a keyframe. What leaves the frame as the time
        is already ``Seconds - StartTime``.
    :meth:`_membrane_frame`
        the disc. A ``Grid`` of ``radial`` x ``angular`` vertices is *not* a
        disc but a square parameter domain, and one ``Set Position`` bends it
        into one: :math:`(s, \varphi) \mapsto (a s\cos\varphi, a s\sin\varphi)`.
        A polar mesh rather than a square grid with its corners deleted,
        because the rim then really is the rim - a clamped edge cut out of a
        square mesh is a staircase, and the silhouette is the one place a drum
        is read. ``Merge by Distance`` welds the seam at
        :math:`\varphi = 0 \equiv 2\pi` and collapses the ``angular``
        coincident vertices at the centre into one.
    :meth:`_mode_frame`
        the arithmetic: one :func:`~geometry_nodes.nodes.make_function` group
        per mode, each computing its own u from ``Position``, the switch, and
        the ``Compare``/``Switch`` pair that holds the whole thing at zero
        before ``StartTime``.
        :math:`J_m` comes from :func:`~geometry_nodes.nodes.bessel_jm_rpn` -
        the ``j0``/``j1`` groups and the upward recurrence - so the tree says
        ``x,j0`` and ``2,x,/,j1,*,j0,-`` rather than carrying a table.
    :meth:`_geometry_frame`
        store, read back, lift, smooth, paint. The elongation is stored
        **before** the lift and read back through a ``Named Attribute`` for the
        same reason as in :class:`WaveVisualizationModifier`: a field is
        evaluated on the geometry the node receives, so an attribute stored
        after the lift would measure r on the *lifted* surface, where it is
        no longer the r the formula means. ``Amplitude`` is stored beside it
        for the material's sake, which is what lets the standard graph shader
        paint this surface too.

    The dials, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, label)``:

    ``Radius``
        a, the radius of the clamped rim. It is in the geometry *and* in the
        wave (through :math:`\alpha_{mn}r/a`), so ramping it grows the drum
        with its mode pattern intact rather than sliding the pattern across it.
    ``Amplitude``
        the height of the crest, for every mode (see ``normalize``).
    ``Frequency``
        the fundamental's frequency in cycles per second of scene time; each
        mode runs at its own multiple of it.
    ``Mode``
        which entry of ``modes`` is showing. An integer; :meth:`set_mode`
        keyframes it.
    ``StartTime``
        the second the drum is struck. Everything downstream reads
        :math:`t - t_0` off the clock rather than t, and the elongation is
        switched to zero while that is negative, so the membrane lies dead
        flat until then and starts its cosine at the crest - which is what a
        drum that is struck once does, and what makes the strike an event in
        the video rather than a mode that was already ringing when the shot
        began.

    :param name: name of the node group, and of the modifier in the stack.
    :param radius: a, in blender units.
    :param radial: vertices along the radius. Ten per radial oscillation is
        smooth; the (1,2) mode needs :math:`\alpha_{12}/\pi \approx 2.2` of
        them across, so 100 is generous and 40 already holds up.
    :param angular: vertices around. A mode with m nodal diameters has 2m
        sectors, so this only has to beat the *silhouette*, which is why it is
        the larger of the two.
    :param modes: the modes to build, as ``(m, n)`` pairs - m nodal diameters,
        n the index of the zero (n = 1 is no interior nodal circle). m is
        limited to 4 by :data:`BESSEL_ZEROS`, and the recurrence behind
        :math:`J_4` is the shakiest thing in the tree (see
        :func:`~geometry_nodes.nodes.bessel_jm_rpn`).
    :param mode: index into ``modes`` the drum starts on.
    :param amplitude: A, in blender units.
    :param frequency: the fundamental's frequency, in cycles per second.
    :param start_time: the scene second the drum is struck. Before it the
        surface is flat, at it every mode is at its crest. The default of 0
        is the drum that has been ringing since the first frame.
    :param normalize: divide each mode by its own :math:`\max|J_m|`, so that
        ``Amplitude`` is the crest height whichever mode is showing. ``False``
        leaves the Bessel functions as they are, which is the honest relative
        amplitude of a membrane that was struck once.
    :param attribute: name of the float attribute the elongation is stored
        under. It displaces the surface and the material reads it back.
    :param material: ``"elongation"`` (the default) builds a divergent colour
        ramp on that attribute through
        :func:`~appearance.textures.gradient_from_attribute`, so crest and
        trough take opposite colours and the nodal lines are the colour in
        between - the whole point of a mode, drawn on the surface that has it.
        ``"function"`` uses the project's standard graph shader,
        :func:`~appearance.textures.function_texture`, instead: it forms
        u = elongation/amplitude out of the two stored attributes and adds the
        emission that goes as u^2, so a drum wears the same material as the
        function tubes and the wave surfaces of the same video (see
        ``InterferenceScene.drum_visualisation2``). A palette name or a
        ``bpy.types.Material`` is set as it stands; ``None`` leaves the disc
        unpainted.
    :param colors: the three palette colours of that ramp, trough to crest.
    :param shade_smooth: smooth-shade the result.
    :param kwargs: passed on to the material (``emission``, ...) and to
        :class:`~geometry_nodes.geometry_nodes_modifier.GeometryNodesModifier`.
    """

    def __init__(self, name="DrumMode", radius=3.0, radial=100, angular=180,
                 modes=((0, 1), (1, 1), (2, 1), (0, 2), (3, 1), (1, 2)),
                 mode=0, amplitude=0.5, frequency=0.4, start_time=0.0,
                 normalize=True, attribute="result", ease_in = 0,
                 material=get_texture("function", alpha_intensity=0.9),
                 shade_smooth=True, wireframe=False, **kwargs):
        """

        :type material: material
        """
        self.name = name
        self.radius = radius
        self.radial = radial
        self.angular = angular
        self.modes = [tuple(entry) for entry in modes]
        if not self.modes:
            raise ValueError("a drum needs at least one mode")
        for m, n in self.modes:
            if m not in BESSEL_ZEROS or not 1 <= n <= len(BESSEL_ZEROS[m]):
                raise ValueError("no zero alpha_%s%s in BESSEL_ZEROS - m is "
                                 "0..4 and n is 1..4" % (m, n))
        self.mode = self.mode_index(mode)
        # what the Mode dial is currently keyframed to; set_mode needs it to
        # write the keyframe that holds the previous mode until the cut
        self.current_mode = self.mode
        self.amplitude = amplitude
        self.frequency = frequency
        self.start_time = start_time
        self.ease_in = ease_in
        self.normalize = normalize
        self.attribute = attribute
        self.paint = material
        self.shade_smooth = shade_smooth
        self.wireframe = wireframe
        self.wireframe_radius = get_from_kwargs(kwargs,"wireframe_radius",0)
        self.kwargs = kwargs
        # filled in by _geometry_frame
        self.material = None
        super().__init__(name=name, automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    def mode_index(self, mode):
        """The index into ``modes`` of ``mode``, given as index or ``(m, n)``."""
        if isinstance(mode, (tuple, list)):
            mode = tuple(mode)
            if mode not in self.modes:
                raise ValueError("mode %s is not one of the modes this drum "
                                 "was built with, %s" % (mode, self.modes))
            return self.modes.index(mode)
        if not 0 <= mode < len(self.modes):
            raise ValueError("mode index %r is outside 0..%d"
                             % (mode, len(self.modes) - 1))
        return int(mode)

    def frequency_ratio(self, mode):
        """omega_mn / omega_01, the mode's frequency in units of the fundamental."""
        m, n = self.modes[self.mode_index(mode)]
        return BESSEL_ZEROS[m][n - 1] / BESSEL_ZEROS[0][0]

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._control_frame(tree)
        membrane = self._membrane_frame(tree, control)
        elongation = self._mode_frame(tree, control)
        geometry = self._geometry_frame(tree, membrane, elongation, control)
        self.group_outputs.location = (25 * 200, 0)
        tree.links.new(geometry.geometry_out, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        """The dials and the clock.

        ``Mode`` is an ``Integer`` node rather than a ``Value`` one because an
        ``Index Switch`` wants an integer and because that is what it is: there
        is no mode between (1,1) and (2,1) to interpolate through.

        The clock the rest of the tree gets is not ``Seconds`` but
        ``Seconds - StartTime``, subtracted once here rather than in each of
        the mode groups: every group would otherwise need the dial as a sixth
        input and would have to spell the subtraction out in its own RPN, and
        the sign of that one difference is also what :meth:`_mode_frame` tests
        to decide whether the drum has been struck yet.

        Nothing here is named so that another node's name contains it -
        ``ibpy.get_geometry_node_from_modifier`` matches by substring, so a
        frame called "Modes" or a function group called "Mode_1_1" would answer
        to ``"Mode"`` before the dial did.

        :return: dict of the sockets the rest of the tree consumes.
        """
        frame = Frame(tree, location=(0, 2), label="Control",
                      name="ControlFrame")
        clock = SceneTime(tree, location=(0, 1), std_out="Seconds",
                          name="Clock", parent=frame)
        radius = InputValue(tree, location=(0, 0), value=self.radius,
                            name="Radius", parent=frame)
        amplitude = InputValue(tree, location=(0, -1), value=self.amplitude,
                               name="Amplitude", parent=frame)
        frequency = InputValue(tree, location=(0, -2), value=self.frequency,
                               name="Frequency", parent=frame)
        selector = InputInteger(tree, location=(0, -3), integer=self.mode,
                                name="Mode", parent=frame)
        start = InputValue(tree, location=(0, 2), value=self.start_time,
                           name="StartTime", parent=frame)
        elapsed = MathNode(tree, location=(1, 2), operation="SUBTRACT",
                           inputs0=clock.std_out, inputs1=start.std_out,
                           name="Elapsed", parent=frame)
        # one point the seven consumers of the shifted clock hang off, so that
        # the frame is not crossed by seven lines from the same socket
        time = Reroute(tree, location=(2, 2), ins=elapsed.std_out,
                       name="ElapsedTime", parent=frame)
        return {"time": time.std_out,
                "radius": radius.std_out,
                "amplitude": amplitude.std_out,
                "frequency": frequency.std_out,
                "mode": selector.std_out}

    # ------------------------------------------------------------------
    def _membrane_frame(self, tree, control):
        r"""The disc, bent out of a square grid.

        The grid runs -0.5..0.5 in both directions, which the map reads as
        :math:`s = x + \tfrac12 \in [0,1]` along the radius and
        :math:`\varphi = (y + \tfrac12)\,2\pi` around. ``Set Position`` in its
        *absolute* mode, not the offset one: the grid's coordinates are a
        parameter domain and nothing about them is a position yet.

        What comes out has two seams, and ``Merge by Distance`` closes both -
        the join at :math:`\varphi = 2\pi`, where the grid's two opposite edges
        land on each other, and the centre, where a whole row of vertices sits
        at r = 0. The tolerance is 1e-4 against a spacing of a/radial, so it
        welds what is coincident and nothing that is merely close.

        :return: the geometry socket carrying the flat disc.
        """
        frame = Frame(tree, location=(0, -2), label="Membrane",
                      name="MembraneFrame")
        grid = Grid(tree, location=(0, 0), size_x=1, size_y=1,
                    vertices_x=self.radial, vertices_y=self.angular,
                    name="ParameterGrid", parent=frame)
        position = Position(tree, location=(0, -2), name="GridPosition",
                            hide=True, parent=frame)
        polar = make_function(tree, location=(1, -2), name="Polar",
                              aux_functions={
                                  "s": "pos_x,0.5,+,radius,*",
                                  "phi": "pos_y,0.5,+,%s,*" % repr(tau)},
                              functions={"disc": ["phi,cos,s,*",
                                                  "phi,sin,s,*",
                                                  "0"]},
                              inputs=["pos", "radius"], outputs=["disc"],
                              vectors=["pos", "disc"],
                              scalars=["radius", "s", "phi"],
                              parent=frame, hide=True)
        tree.links.new(position.std_out, polar.inputs["pos"])
        tree.links.new(control["radius"], polar.inputs["radius"])

        disc = SetPosition(tree, location=(2, 0), geometry=grid.geometry_out,
                           position=polar.outputs["disc"], name="BendIntoDisc",
                           parent=frame)
        welded = MergeByDistance(tree, location=(3, 0),
                                 geometry=disc.geometry_out, distance=1e-4,
                                 name="CloseTheSeam", parent=frame)
        return welded

    # ------------------------------------------------------------------
    def _mode_frame(self, tree, control):
        r"""One function group per mode, and the switch that picks one.

        Each group is the whole of :math:`u_{mn}` for its own mode: the radius
        and azimuth of the vertex it is evaluated on, the Bessel function of
        the order that mode wants, the angular factor, and the clock. Nothing
        is shared between them because nothing can be - a different m is a
        different chain of Bessel groups - which is exactly what makes them
        switchable rather than dialable.

        The ``m = 0`` modes skip the angular factor rather than multiplying by
        :math:`\cos 0 = 1`, so the fundamental costs an ``atan2`` less.

        The strike is the last thing in the frame: the groups are handed
        :math:`t - t_0` and so would run backwards through their cosine before
        the drum is struck, which is a membrane that is already ringing at
        ``StartTime`` and not one that is hit there. A ``Compare`` on the sign
        of that time and a ``Switch`` that answers 0 while it is negative cut
        that off - the surface is flat until the strike, and steps into the
        crest of the mode at it. It sits after the mode switch, not inside the
        groups, so it costs one comparison rather than one per mode.

        :return: the socket carrying the elongation of the selected mode, held
            at zero before ``StartTime``.
        """
        frame = Frame(tree, location=(4, 0), label="Drum modes",
                      name="Solutions")
        position = Position(tree, location=(0, 0), name="DiscPosition",
                            hide=True, parent=frame)

        sockets = []
        for i, (m, n) in enumerate(self.modes):
            alpha = BESSEL_ZEROS[m][n - 1]
            ratio = alpha / BESSEL_ZEROS[0][0]
            peak = _BESSEL_PEAKS[m] if self.normalize else 1.0

            aux = {}
            # the disc is still flat here, so the distance in the plane is the
            # distance in space
            aux["r"] = "pos,length"
            aux["x"] = "r,radius,/,%s,*" % repr(alpha)
            bessel, jm = bessel_jm_rpn("x", m, prefix="b")
            aux.update(bessel)
            aux["wt"] = "time,frequency,*,%s,*" % repr(tau * ratio)
            elongation = "amplitude,%s,/,%s,*,wt,cos,*" % (repr(peak), jm)
            if m:
                aux["ang"] = "pos_y,pos_x,atan2,%s,*,cos" % repr(float(m))
                elongation += ",ang,*"
            aux["u"] = elongation+",-1,*"

            group = make_function(
                tree, location=(1, -i), name="Wave_%d_%d" % (m, n),
                functions={"elongation": "u"}, aux_functions=aux,
                inputs=["pos", "time", "frequency", "radius", "amplitude"],
                outputs=["elongation"], vectors=["pos"],
                scalars=["time", "frequency", "radius", "amplitude",
                         "elongation"] + list(aux),
                custom_ops=BESSEL_OPS, parent=frame, hide=True)
            tree.links.new(position.std_out, group.inputs["pos"])
            for key in ("time", "frequency", "radius", "amplitude"):
                tree.links.new(control[key], group.inputs[key])
            sockets.append(group.outputs["elongation"])

        if len(sockets) == 1:
            elongation = sockets[0]
        else:
            switch = IndexSwitch(tree, location=(3, 0), data_type="FLOAT",
                                 index=control["mode"], name="ModeSwitch",
                                 parent=frame)
            for socket in sockets:
                switch.add_item(socket)
            elongation = switch.std_out

        struck = CompareNode(tree, location=(4, -1), data_type="FLOAT",
                             operation="GREATER_EQUAL", inputs0=control["time"],
                             inputs1=-self.ease_in, name="Struck", parent=frame, hide=True)
        strike = Switch(tree, location=(5, 0), input_type="FLOAT",
                        switch=struck.std_out, false=0, true=elongation,
                        name="Strike", parent=frame)
        return strike.std_out

    # ------------------------------------------------------------------
    def _geometry_frame(self, tree, membrane, elongation, control):
        """Store -> read back -> lift -> smooth -> paint.

        The second attribute, ``amplitude``, is the crest height the dial is
        currently on, written onto every point. Nothing in *this* tree reads
        it - it is there for the material, which needs the elongation and the
        scale it is to be measured against as two attributes of the geometry
        rather than as numbers baked into the shader when it was built. That
        is the convention :func:`~appearance.textures.function_texture` and
        :class:`WaveVisualizationModifier` already share, and storing it here
        is what lets ``material="function"`` work on a drum.

        :return: the geometry socket for the group output.
        """
        geometry_nodes = [membrane]
        frame = Frame(tree, location=(8, 0), label="Geometry",
                      name="GeometryFrame")
        uv_map = StoreNamedAttribute(tree,location=(-1,0),data_type="FLOAT_VECTOR",
                                     domain="POINT",name="UVMap",parent=frame)
        geometry_nodes.append(uv_map)

        stored = StoreNamedAttribute(tree, location=(0, 0), data_type="FLOAT",
                                     domain="POINT", name=self.attribute,
                                     value=elongation, parent=frame)
        geometry_nodes.append(stored)

        amp_store = StoreNamedAttribute(tree, location=(1, 0), data_type="FLOAT",
                                        domain="POINT", name="amplitude",
                                        value=control["amplitude"], parent=frame)
        geometry_nodes.append(amp_store)

        height = NamedAttribute(tree, location=(1, -2), data_type="FLOAT",
                                name=self.attribute, parent=frame, hide=True)
        offset = CombineXYZ(tree, location=(2, -2), z=height.std_out,
                            name="Lift", parent=frame, hide=True)
        lifted = SetPosition(tree, location=(3, 0),
                             offset=offset.std_out, name="Displace",
                             parent=frame)
        geometry_nodes.append(lifted)

        if self.shade_smooth:
            smooth = SetShadeSmooth(tree, location=(4, 0),
                                    name="Smooth", parent=frame)
            geometry_nodes.append(smooth)
        if self.wireframe:
            wireframe = WireFrame(tree, location=(5, 0), radius=self.wireframe_radius,parent=frame)
            geometry_nodes.append(wireframe)

        if self.paint is not None:
            painted = SetMaterial(tree, location=(6, 0),
                                  material=self._texture(), name="Paint",
                                  parent=frame)
            self.material = painted.material
            self.materials.append(painted.material)
            geometry_nodes.append(painted)

        create_geometry_line(tree, geometry_nodes)
        # return the last node for further manipulation
        return geometry_nodes[-1]

    # ------------------------------------------------------------------
    def _texture(self):
        """The material: the elongation as a colour, zero in the middle.

        The ramp is fed ``u/(2A) + 0.5``, so it runs from trough at 0 through
        the nodal value at 0.5 to crest at 1 - which puts the nodal diameters
        and circles on the surface as the one colour that does not move.

        ``material="function"`` asks for the project's standard graph shader,
        :func:`~appearance.textures.function_texture`, instead - the one the
        function tubes and the wave surfaces of this video already wear. It
        does the same thing by a different route: it reads the elongation and
        the ``amplitude`` attribute the geometry frame stores beside it,
        forms u = elongation/amplitude itself, and adds an emission that goes
        as u^2 and an alpha that can be made to follow it. Painting a drum
        with the same shader as the graphs is what makes the surface read as
        the *same quantity* plotted on a disc.
        """
        if not isinstance(self.paint, str):
            return self.paint
        if self.paint == "function":
            from appearance.textures import function_texture
            return function_texture(name=self.name + "Texture",
                                    attribute=self.attribute,
                                    scale_attribute="amplitude",
                                    **self.kwargs)
        if self.paint != "elongation":
            return self.paint
        from appearance.textures import gradient_from_attribute
        from utils.constants import COLOR_NAMES, COLORS_SCALED
        rgba = [COLORS_SCALED[COLOR_NAMES.index(color)] for color in self.colors]
        return gradient_from_attribute(
            name=self.name + "Texture", attr_name=self.attribute,
            function="fac,%s,/,0.5,+" % repr(2 * self.amplitude),
            gradient={0: rgba[0], 0.5: rgba[1], 1: rgba[2]},
            **self.kwargs)

    # ------------------------------------------------------------------
    def set_mode(self, mode, begin_time=0):
        """Cut to another mode, as an index or as an ``(m, n)`` pair.

        Two keyframes one frame apart - the mode it was showing, then the new
        one - so that everything before ``begin_time`` keeps the old mode
        however many times this is called.

        :return: ``begin_time``, so it chains like the other timings.
        """
        from interface import ibpy
        index = self.mode_index(mode)
        dial = ibpy.get_geometry_node_from_modifier(self, "Mode")
        if dial is None:
            raise KeyError("no Mode dial in %s" % self.tree.name)
        ibpy.change_default_integer(dial, from_value=self.current_mode,
                                    to_value=index, begin_time=begin_time,
                                    transition_time=0)
        self.current_mode = index
        return begin_time

    # ------------------------------------------------------------------
    def elongation_numpy(self, points, seconds=0.0, mode=None):
        """The same u, in numpy - the tree's mirror.

        ``scipy.special.jv`` is the exact Bessel function here, so comparing
        the two also measures what the polynomial approximation and the
        recurrence behind :math:`J_m` cost.

        :param points: ``(n, 2)`` or ``(n, 3)`` array of positions on the flat
            disc; only x and y are read.
        :param seconds: the scene time the tree reads off the clock. It is the
            *scene's* second, not the time since the strike, so ``start_time``
            is subtracted here exactly as the control frame subtracts it.
        :param mode: which mode, as an index or an ``(m, n)`` pair. Defaults to
            the one the tree was built showing.
        """
        from scipy.special import jv
        points = np.asarray(points, dtype=float)[:, :2]
        m, n = self.modes[self.mode if mode is None else self.mode_index(mode)]
        alpha = BESSEL_ZEROS[m][n - 1]
        peak = _BESSEL_PEAKS[m] if self.normalize else 1.0
        radius = np.linalg.norm(points, axis=1)
        # the same clamp bessel_jm_rpn applies before it divides by x
        x = np.maximum(alpha * radius / self.radius, 0.01)
        elapsed = seconds - self.start_time
        if elapsed < 0:
            # the Switch in the mode frame: nothing has been struck yet
            return np.zeros(len(points))
        wt = tau * self.frequency * alpha / BESSEL_ZEROS[0][0] * elapsed
        angle = np.cos(m * np.arctan2(points[:, 1], points[:, 0])) if m else 1.0
        return self.amplitude / peak * jv(m, x) * angle * np.cos(wt)


class Drum(BObject):
    """
    A drum model loaded from a bundled .blend asset.
    """

    def __init__(self, **kwargs):
        """Load a drum model.

        The sibling of :class:`~objects.derived_objects.whistle.Whistle`, and
        built the same way: ``Drum.blend`` carries the baked material
        ``drum_material`` and the image it reads is packed into the file, so
        the asset arrives textured on any checkout and nothing has to be
        painted onto it.

        Args:
            **kwargs: Forwarded to :class:`BObject`. Supported keys:
                * ``location`` (list[float]): Left at the imported pose when
                  it is not given.
                * ``rotation_euler`` (list[float]): ditto.
                * ``original_material`` (bool): Keep the ``drum_material``
                  that comes with ``Drum.blend`` instead of painting the
                  model with a palette colour. Defaults to ``True``. Pass
                  ``False`` (together with ``color``/``colors``, if a
                  particular colour is wanted) to recolour the drum the way
                  the other library primitives are recoloured.
        """
        self.kwargs = kwargs
        original_material = self.get_from_kwargs('original_material', True)

        # Only the transform the caller actually asked for is handed on, so
        # that the asset keeps whatever pose it is imported with - passing a
        # default of [0,0,0] would overwrite it instead.
        transform = {key: kwargs.pop(key) for key in
                     ('location', 'rotation_euler', 'rotation_quaternion') if key in kwargs}

        # Appending the object brings its material along, so keeping it is only
        # a matter of not painting over it. The import is therefore always done
        # with no_material, and any color is left to the wrapper below - whose
        # apply_material reaches through to the very same blender object and
        # would otherwise overwrite whatever the import had just painted.
        bobs = BObject.from_file("Drum", objects=["Drum"], no_material=True, **kwargs)

        # obj=bobs[0].ref_obj, not bobs[0]: wrapping the wrapper makes BObject
        # write location and rotation onto a python attribute of the inner
        # BObject, where they have no effect on the blender object at all
        if original_material:
            super().__init__(obj=bobs[0].ref_obj, name="Drum", no_material=True, **transform)
        else:
            if 'colors' not in kwargs:
                kwargs.setdefault('color', 'drawing')  # the library default
            super().__init__(obj=bobs[0].ref_obj, name="Drum", **kwargs, **transform)
