"""
host the modifiers that get instantiated in predefined objects
for example:
Ladder
Slide

"""


import numpy as np

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (
    Grid, Position, SetPosition, make_function,CurveCircle, CurveLine, CurveToMesh, Frame, InstanceOnPoints, JoinGeometry,
    MeshLine, RealizeInstances, SetMaterial, SetShadeSmooth, TransformGeometry,
)
from interface.ibpy import Vector
#: one node-editor grid step, as in ``orbi_modifier``
GRID = 200

#: gravity, in blender units per second squared
GRAVITY = 9.81

#: the factor by which a rolling solid sphere is slower than a sliding point,
#: which is ``1 / (1 + I/ma^2)`` with ``I = 2/5 m a^2``
ROLLING = 5.0 / 7.0


r"""A cycloid slide, and the closed form of what rolls down it.

The curve is not decoration. A cycloid is the *tautochrone*: a body released
from rest on it reaches the bottom in a time that does not depend on where it
started. That is because arc length from the bottom obeys simple harmonic
motion exactly,

.. math::
    \sigma(\theta) = 4R\cos\tfrac{\theta}{2},
    \qquad
    \ddot\sigma = -\frac{k g}{4R}\,\sigma,
    \qquad
    \sigma(t)=\sigma_0\cos\omega t,\quad \omega=\sqrt{\frac{k g}{4R}},

so the descent is a cosine and the drop takes a quarter period whatever the
release point. Nothing else has to be integrated, which is exactly the
property the rest of this workspace is built on: the orbi's whole gait is a
closed-form function of one clock, and a slide whose descent needed numerical
integration would be the only thing in the shot that could not be evaluated at
an arbitrary frame.

``k`` is where a *rolling* ball differs from a sliding one. For a body of
moment of inertia :math:`I` about its own centre, the equation along the curve
carries a factor :math:`1 + I/ma^2`; for a solid sphere that is
:math:`1 + 2/5`, so :math:`k = 5/7` and the ball takes
:math:`\sqrt{7/5}` times as long as a sliding point would. The creature is a
ball, so ``rolling`` defaults to true.

The two approximations, stated plainly
--------------------------------------
The tautochrone result is exact for a *point*. A ball of radius ``a`` rolls
along the offset curve rather than the cycloid itself, and the dynamics there
are not quite the cycloid's, so the descent time is right to order
:math:`a/R`. And the offset curve develops a cusp wherever the cycloid's
radius of curvature :math:`\rho = 4R\sin(\theta/2)` falls below ``a`` - near
the top, where a cycloid is arbitrarily tightly curved. :meth:`SlideModifier.
fits` is the check; keep the ball away from the cusp and neither matters at
the scale of a shot.

Which way it faces
------------------
The curve lies in the ``y``-``z`` plane and the slide is wide in ``x``, so a
ball goes down it travelling ``+y`` and turning about ``x`` - which is the
orbi's own rolling axis (see ``BodyOrient`` in :mod:`orbi.orbi_modifier`). The
top of a cycloid has a *vertical* tangent and the bottom a horizontal one,
which is why it is the right curve for this shot twice over: it leaves the top
of a ladder going straight down and arrives at the water flat.
"""

class SlideModifier(GeometryNodesModifier):
    r"""One arch of a cycloid, swept into a chute.

    :param name: modifier and node-group name.
    :param radius: the cycloid's generating radius ``R``. The slide drops
        ``2R`` and its half-arch is ``4R`` long, so this is the one number
        that sets the size of the thing.
    :param width: how wide the chute is, in ``x``.
    :param start: the parameter the slide starts at, in radians. Zero is the
        cusp, where the tangent is vertical and the curvature infinite; a
        little way past it is where a ball of finite size can actually sit.
        The slide always ends at ``pi``, the bottom.
    :param depth: how far the edges of the chute curve up out of the plane of
        the curve, making a trough rather than a ribbon. Measured along the
        curve's own normal.
    :param resolution: samples along the curve.
    :param across: samples across it. Two is a flat ribbon; more is needed
        only if ``depth`` is non-zero.
    :param rolling: whether what goes down it rolls (the default) or slides.
        This changes the descent time by ``sqrt(7/5)`` and nothing else.
    :param material: what to paint it with.
    """

    def __init__(self, name="Slide", radius=3.0, width=2.4, start=0.55,
                 depth=0.35, resolution=192, across=9, rolling=True,
                 material=None, **kwargs):
        self.radius = radius
        self.width = width
        self.start = start
        self.depth = depth
        self.resolution = resolution
        self.across = across
        self.rolling = rolling
        self.material = material
        super().__init__(name=name, automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    # the curve, in python
    # ------------------------------------------------------------------
    def point(self, theta):
        """A point of the cycloid itself, on the centre line of the chute."""
        return Vector([0.0,
                       self.radius * (theta - np.sin(theta)),
                       self.radius * (np.cos(theta) - 1.0)])

    def normal(self, theta):
        """The unit normal, pointing up out of the chute.

        The tangent is ``(sin(t/2), -cos(t/2))`` in ``(y, z)``, so this is
        that turned a quarter turn - and at the bottom, where ``theta = pi``,
        it is straight up, as it has to be.
        """
        return Vector([0.0, np.cos(0.5 * theta), np.sin(0.5 * theta)])

    def curvature_radius(self, theta):
        """``rho = 4R sin(theta/2)``, which vanishes at the cusp."""
        return 4.0 * self.radius * np.sin(0.5 * theta)

    def centre(self, theta, ball):
        """Where the middle of a ball of radius ``ball`` sits, resting here."""
        return self.point(theta) + ball * self.normal(theta)

    def arc(self, theta):
        """Arc length from the **bottom**, which is the SHM coordinate."""
        return 4.0 * self.radius * np.cos(0.5 * theta)

    def theta_at(self, sigma):
        """The inverse of :meth:`arc`, clamped to the arch."""
        c = min(max(sigma / (4.0 * self.radius), 0.0), 1.0)
        return 2.0 * np.arccos(c)

    # ------------------------------------------------------------------
    # and the descent, likewise
    # ------------------------------------------------------------------
    @property
    def omega(self):
        """The angular frequency of the harmonic motion along the curve."""
        k = ROLLING if self.rolling else 1.0
        return np.sqrt(k * GRAVITY / (4.0 * self.radius))

    @property
    def duration(self):
        """How long the descent takes: a quarter period.

        The tautochrone property in one line - this does not depend on
        :attr:`start`, so moving the release point changes how far the
        creature travels but not how long it is on the slide.
        """
        return 0.5 * np.pi / self.omega

    @property
    def arc_length(self):
        """How far a ball rolls, from the release point to the bottom.

        This is the number the creature needs: rolling without slipping turns
        it through ``arc_length / size`` radians (see
        ``OrbiModifier._create_body_orient``).
        """
        return self.arc(self.start)

    def descent(self, elapsed, ball):
        """Where the ball is, and how far it has turned, at a given moment.

        :param elapsed: seconds since it was released at :attr:`start`.
        :param ball: the radius of the ball.
        :return: ``(centre, turn, theta)`` - the position of its middle, the
            angle it has rolled through in radians (negative, because it
            rolls towards ``+y`` and a right-handed turn about ``+x`` carries
            ``+y`` towards ``+z``), and the curve parameter it is at.
        """
        sigma0 = self.arc_length
        phase = min(max(elapsed, 0.0), self.duration) * self.omega
        sigma = sigma0 * np.cos(phase)
        theta = self.theta_at(sigma)
        return self.centre(theta, ball), -(sigma0 - sigma) / ball, theta

    def fits(self, ball):
        """Whether a ball of this radius can actually sit on the whole slide.

        The centre of a ball rolling on a curve traces the offset curve, and
        that curve cusps wherever the radius of curvature drops below the
        ball's. On a cycloid the curvature is worst at the top, so this is a
        question about :attr:`start` alone.

        :return: ``(ok, rho_min, theta_min)`` - whether it fits, the tightest
            curvature the ball meets, and the parameter it would have to start
            at instead.
        """
        rho = self.curvature_radius(self.start)
        least = 2.0 * np.arcsin(min(ball / (4.0 * self.radius), 1.0))
        return rho > ball, rho, least

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        frame = Frame(tree, location=(0, 0), label="Cycloid",
                      name="CycloidFrame", node_height=GRID)
        # a grid is already a mesh with two parameters on it, so the chute
        # needs no curve nodes at all: one axis of the grid is swept into the
        # curve parameter and the other stays across the slide
        grid = Grid(tree, location=(0, 0), node_height=GRID,
                    size_x=1.0, size_y=self.width,
                    vertices_x=self.resolution, vertices_y=self.across,
                    name="Sheet")
        position = Position(tree, location=(0, -2), node_height=GRID,
                            name="SheetPosition", hide=True)

        shape = make_function(
            tree,
            aux_functions={
                # the grid runs -1/2..1/2 along its first axis; that is the
                # curve parameter, and the second axis stays as it is
                "u": "pos_x,0.5,+",
                "th": "th0,%s,th0,-,u,*,+" % repr(np.pi),
                "half": "th,2,/",
                # the trough, as a fraction of the half width squared, laid
                # on along the curve's own normal so that it is a trough
                # everywhere rather than only at the bottom
                "q": "pos_y,2,*,width,/",
                "bowl": "depth,q,*,q,*",
                "cy": "half,cos",
                "sy": "half,sin",
            },
            functions={"shape": ["pos_y",
                                 "radius,th,th,sin,-,*,bowl,cy,*,+",
                                 "radius,th,cos,1,-,*,bowl,sy,*,+"]},
            inputs=["pos", "radius", "th0", "width", "depth"],
            outputs=["shape"],
            vectors=["pos", "shape"],
            scalars=["radius", "th0", "width", "depth",
                     "u", "th", "half", "q", "bowl", "cy", "sy"],
            name="Cycloid", hide=True)
        shape.location = (2 * GRID, 0)
        frame.add([grid, position, shape])
        tree.links.new(position.std_out, shape.inputs["pos"])
        for socket, value in (("radius", self.radius), ("th0", self.start),
                              ("width", self.width), ("depth", self.depth)):
            shape.inputs[socket].default_value = value

        shaped = SetPosition(tree, location=(4, 0), node_height=GRID,
                             geometry=grid.geometry_out,
                             position=shape.outputs["shape"], name="Shape")
        geometry = shaped.geometry_out
        nodes = [shaped]
        if self.material is not None:
            painted = SetMaterial(tree, location=(5, 0), node_height=GRID,
                                  geometry=geometry, material=self.material,
                                  name="SlideMaterial", **kwargs)
            self.materials.append(painted.material)
            geometry = painted.geometry_out
            nodes.append(painted)
        smooth = SetShadeSmooth(tree, location=(6, 0), node_height=GRID,
                                geometry=geometry, name="SlideShading")
        nodes.append(smooth)
        frame.add(nodes + [self.group_outputs])
        self.group_outputs.location = (7 * GRID, 0)
        tree.links.new(smooth.geometry_out, self.group_outputs.inputs["Geometry"])

"""A ladder for the orbi to climb, and the arithmetic of where its rungs are.

Two rails and a stack of rungs is not much of a node tree, and most of this
file is not the geometry. It is the promise that :meth:`LadderModifier.rung`
returns the position of a rung that is *actually there* - because the climbing
pose reads those positions to decide where to put the creature's hands and
feet, and a hand that lands between two rungs is worse than no ladder at all.

The one idea
------------
The ladder is built standing straight up along ``+z`` and then leaned over as
a whole, by a single ``Transform Geometry`` at the end of the tree. Python
leans the same way, with the same angle, in :meth:`LadderModifier._lean`. So
there is exactly one place the tilt happens on each side, and the two agree by
construction rather than by being typed twice - which is the same discipline
the wave front is built with (see
:func:`~geometry_nodes.nodes.wave_front_gate`).

Leaning about ``x`` is what makes this ladder climbable by *this* creature.
The orbi's legs are apart in ``x`` and it travels along ``+y``, so a ladder
whose rails are apart in ``x`` is one it meets face-on, and a lean about ``x``
tips the top away from it - which is the way a ladder rests against a wall.

What is a dial and what is not
------------------------------
Nothing here is a dial. The orbi makes every number an ``Input Value`` so a
scene can ramp it, and that is right for a creature whose whole animation is
the value of one clock; it is wrong here. The rung count is derived from the
spacing, the tree bakes that count into a ``Mesh Line``, and
:meth:`LadderModifier.rung` computes from the same python numbers. An animated
spacing would move the rungs in the geometry while the count - and the
creature's hands - stayed where they were, and nothing would report the
mismatch. A ladder that needs to change is a second ladder.
"""

class LadderModifier(GeometryNodesModifier):
    """Two rails, a stack of rungs, and a lean.

    :param name: modifier and node-group name.
    :param height: the **vertical** rise, from the foot of the rails to their
        top. Not the length of the rails: a leaning ladder is longer than it
        is tall, and this is the number a scene actually knows ("it has to
        reach the platform"). The rail length follows as
        ``height / cos(lean)``.
    :param width: the distance between the rails, which is also the length of
        a rung.
    :param lean: how far the ladder tips backwards, in radians, measured from
        vertical. Positive leans the top towards ``+y``, which is the
        direction the creature travels, so a leaning ladder tips *away* from
        a creature walking up to it.
    :param rung_spacing: the gap between rungs, measured **along the rail**
        rather than vertically - that is where a rung actually is, and it is
        the number the climb has to take whole multiples of.
    :param rail_radius: radius of a rail.
    :param rung_radius: radius of a rung.
    :param overshoot: how far the rails run on past the top rung. A real
        ladder has this and a scene needs it: it is what the creature holds
        while it steps off the top.
    :param resolution: how many sides the tubes have.
    :param material: what to paint the rails with.
    :param rung_material: what to paint the rungs with. Defaults to
        ``material``.
    """

    def __init__(self, name="Ladder", height=6.0, width=0.9, lean=0.18,
                 rung_spacing=0.45, rail_radius=0.06, rung_radius=0.045,
                 overshoot=0.55, resolution=12,
                 material=None, rung_material=None, **kwargs):
        self.height = height
        self.width = width
        self.lean = lean
        self.rung_spacing = rung_spacing
        self.rail_radius = rail_radius
        self.rung_radius = rung_radius
        self.overshoot = overshoot
        self.resolution = resolution

        self.material = material
        self.rung_material = rung_material if rung_material else material

        super().__init__(name=name, automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    # where things are. This is the half of the file the climb reads
    # ------------------------------------------------------------------
    @property
    def length(self):
        """The length of a rail, which is longer than the ladder is tall."""
        return self.height / np.cos(self.lean)

    @property
    def up(self):
        """The unit vector along the rails, pointing up the ladder.

        The direction the creature travels while climbing, and therefore the
        one :meth:`~orbi.orbi_modifier.OrbiModifier` has to be told about: its
        ``Travel`` goes along ``+y`` on the flat, and up *this* on a ladder.
        """
        return Vector([0, np.sin(self.lean), np.cos(self.lean)])

    @property
    def rung_count(self):
        """How many rungs fit, given the spacing and the overshoot.

        The lowest rung is one spacing up from the foot - a ladder does not
        have a rung lying on the ground - so ``n`` rungs need
        ``n * spacing`` of rail, and what is left over at the top is reserved
        for the :attr:`overshoot`.
        """
        usable = self.length - self.overshoot
        return max(int(np.floor(usable / self.rung_spacing)), 1)

    def _lean(self, point):
        """Tip a point of the upright ladder into the leaning one.

        A rotation about ``x`` by ``-lean``, which is the rotation the tree's
        one ``Transform Geometry`` applies to the whole ladder. Written out
        rather than delegated so that the two sides of the port can be
        compared: ``(0, 0, s)`` has to come out at ``(0, s sin lean,
        s cos lean)``, i.e. up the rails.
        """
        x, y, z = point
        c, s = np.cos(self.lean), np.sin(self.lean)
        return Vector([x, y * c - z * -s, y * -s + z * c])

    def rung(self, index):
        """Where the middle of a rung is, in the modifier's own frame.

        :param index: which rung, counting from ``0`` at the bottom. Negative
            indexes from the top, as a python list does, so ``rung(-1)`` is
            the one the creature steps off.
        :return: the position, leaned.
        """
        count = self.rung_count
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError("rung %d of %d" % (index, count))
        return self._lean(Vector([0, 0, (index + 1) * self.rung_spacing]))

    def hold(self, index, side):
        """Where a hand or a foot goes on a rung.

        Just inside the rail, which is where a rung is strong and where a hand
        does not slide off the end of it.

        :param index: which rung, as :meth:`rung`.
        :param side: ``"Left"`` or ``"Right"``. The creature's left is ``+x``
            for its arms - see the sign convention in
            :mod:`orbi.orbi_modifier`.
        """
        if side not in ("Left", "Right"):
            raise ValueError("side is 'Left' or 'Right', not %r" % side)
        inset = 0.5 * self.width - 2 * self.rung_radius
        offset = inset if side == "Left" else -inset
        return self.rung(index) + Vector([offset, 0, 0])

    @property
    def top(self):
        """The very top of the rails, overshoot included."""
        return self._lean(Vector([0, 0, self.length]))

    def climb_stride(self, rungs=1):
        """A stride that lands on a rung rather than between two.

        The climbing counterpart of the flat gait's stride, and the reason
        this modifier has to be built before the climb is posed: a stride that
        is not a whole number of rungs puts a hand in mid air every step, and
        no amount of easing hides it.

        :param rungs: how many rungs the creature takes per step.
        :return: the distance travelled along :attr:`up` in one stride.
        """
        return rungs * self.rung_spacing

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        rails = self._create_rail_frame(tree)
        rungs = self._create_rung_frame(tree)

        at = lambda x, y: (10 + x, y)
        frame = Frame(tree, location=(10, 1), label="Assemble",
                      name="AssembleFrame", node_height=GRID)
        painted = [self._paint(tree, rails, self.material, at(0, 1),
                               "RailMaterial", frame, **kwargs),
                   self._paint(tree, rungs, self.rung_material, at(0, 0),
                               "RungMaterial", frame, **kwargs)]
        join = JoinGeometry(tree, location=at(1, 0), node_height=GRID,
                            geometry=painted, name="JoinLadder")

        # the one place the ladder tips over. Python leans the same way in
        # _lean, so rung() and the geometry cannot disagree about where a
        # rung is unless one of the two is edited alone
        leaned = TransformGeometry(tree, location=at(2, 0), node_height=GRID,
                                   geometry=join.geometry_out,
                                   rotation=Vector([-self.lean, 0, 0]),
                                   name="Lean")
        smooth = SetShadeSmooth(tree, location=at(3, 0), node_height=GRID,
                                geometry=leaned.geometry_out,
                                name="LadderShading")
        tree.links.new(smooth.geometry_out, self.group_outputs.inputs["Geometry"])
        self.group_outputs.location = ((14 + 1) * GRID, 0)
        frame.add([join, leaned, smooth, self.group_outputs])

    def _paint(self, tree, geometry, material, location, name, frame, **kwargs):
        """One ``Set Material``, or nothing if there is none to set.

        The same reasoning as :meth:`orbi.orbi_modifier.OrbiModifier._paint`:
        geometry leaving a node tree without a material of its own renders in
        blender's default grey, whatever the host object's slots say.
        """
        if material is None:
            return geometry
        node = SetMaterial(tree, location=location, node_height=GRID,
                           geometry=geometry, material=material, name=name,
                           **kwargs)
        frame.add([node])
        self.materials.append(node.material)
        return node.geometry_out

    def _tube(self, tree, start, end, radius, location, name, frame):
        """A straight tube between two points: line, circle, sweep.

        The same three nodes a limb of the creature is made of, which is not a
        coincidence - it is the cheapest way to get a round bar out of a node
        tree, and it is why the ladder needs no primitive of its own.
        """
        line = CurveLine(tree, location=location, node_height=GRID,
                         start=start, end=end, name=name + "Line")
        profile = CurveCircle(tree, location=(location[0], location[1] - 1),
                              node_height=GRID, resolution=self.resolution,
                              radius=radius, name=name + "Profile")
        mesh = CurveToMesh(tree, location=(location[0] + 1, location[1]),
                           node_height=GRID, curve=line.geometry_out,
                           profile_curve=profile.geometry_out,
                           fill_caps=True, name=name)
        frame.add([line, profile, mesh])
        return mesh.geometry_out

    def _create_rail_frame(self, tree):
        """``Rails``: the two long tubes, upright.

        They run the whole :attr:`length`, so the overshoot is not a separate
        piece of geometry - it is simply the part of the rail above the last
        rung.
        """
        frame = Frame(tree, location=(0, 4), label="Rails", name="RailFrame",
                      node_height=GRID)
        half = 0.5 * self.width
        tubes = [self._tube(tree, Vector([sign * half, 0, 0]),
                            Vector([sign * half, 0, self.length]),
                            self.rail_radius, (0, 4 - 2 * row),
                            "Rail" + side, frame)
                 for row, (side, sign) in enumerate((("Left", 1), ("Right", -1)))]
        join = JoinGeometry(tree, location=(3, 3), node_height=GRID,
                            geometry=tubes, name="JoinRails")
        frame.add([join])
        return join.geometry_out

    def _create_rung_frame(self, tree):
        """``Rungs``: one rung, instanced up a line of points.

        A ``Mesh Line`` of :attr:`rung_count` points from the first rung to
        the last carries the spacing, and one rung tube is instanced on every
        one of them. The instances are realised rather than left as instances
        because the ladder is joined with the rails and painted, and a
        ``Set Material`` on unrealised instances paints the *instancer*
        rather than what it instances.
        """
        frame = Frame(tree, location=(0, -1), label="Rungs", name="RungFrame",
                      node_height=GRID)
        count = self.rung_count
        first = self.rung_spacing
        last = count * self.rung_spacing
        points = MeshLine(tree, location=(0, -1), node_height=GRID,
                          mode="END_POINTS", count_mode="TOTAL", count=count,
                          start_location=Vector([0, 0, first]),
                          end_location=Vector([0, 0, last]), name="RungPoints")
        half = 0.5 * self.width
        bar = self._tube(tree, Vector([-half, 0, 0]), Vector([half, 0, 0]),
                         self.rung_radius, (0, -3), "Rung", frame)
        placed = InstanceOnPoints(tree, location=(3, -1), node_height=GRID,
                                  points=points.geometry_out, instance=bar,
                                  name="PlaceRungs")
        real = RealizeInstances(tree, location=(4, -1), node_height=GRID,
                                geometry=placed.geometry_out,
                                name="RealizeRungs")
        frame.add([points, placed, real])
        return real.geometry_out
