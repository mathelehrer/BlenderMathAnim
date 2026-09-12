r"""The organ pipe of the interference video.

One self-contained modifier, :class:`AcousticModifier`, which builds the tree
of ``video_interferences/tmp.xml`` node for node.
"""
import numpy as np

from appearance.textures import get_texture
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (BooleanMath, CombineXYZ, DeleteGeometry,
                                  Frame, IcoSphere, InputValue,
                                  InstanceOnPoints, MathNode, Points, Position,
                                  RandomValue, RealizeInstances, Reroute,
                                  SceneTime, SetMaterial, SetPosition,
                                  StoreNamedAttribute, make_function, InputInteger)

pi = np.pi
tau = 2 * pi

#: A sin(2 pi x / lambda - 2 pi t / T), as an RPN string for ``make_function``.
#: Every symbol in it is a socket of the ``Intensity`` group, so another
#: formula in the same symbols can be passed as ``elongation``.
PLANE_WAVE_ELONGATION = ("amplitude,2,pi,*,wavelength,/,x,*,"
                         "2,pi,*,period,/,time,*,-,sin,*")


class AcousticModifier(GeometryNodesModifier):
    r"""Air in an organ pipe: points drawn from a sound wave, inside a cylinder.

    Candidates are scattered uniformly through the bounding box of the pipe and
    then thrown away twice over: by a rejection test against the intensity s^2
    of a plane wave

    .. math:: s(x, t) = A \sin\!\Big(\frac{2\pi}{\lambda}x - \frac{2\pi}{T}t\Big),

    and by the pipe wall, y^2 + z^2 > R^2. What is left is a train of slabs of
    points, one per compression, and since ``time`` comes off ``Scene Time``
    they travel at lambda / T units per second with no keyframe in the tree.
    The survivors carry three attributes out to the material: ``Intensity``
    (s^2 where the point sits), ``Elongation`` (the signed s there) and
    ``Amplitude`` (what the dial reads), the last because s runs -A..A and the
    first two cannot be read without it.

    :param length: the pipe, along x. Ramping it stretches the box the
        candidates are drawn in; since their number is fixed when the modifier
        is built, a longer pipe is a thinner cloud.
    :param pipe_radius: R, the radius of the cylinder about the x axis.
    :param amplitude: A.
    :param wavelength: lambda, in the same units as the pipe.
    :param period: T, in seconds - the clock is ``Scene Time -> Seconds``, so
        this is real time.
    :param count: how many points to end up with, after the rejection test and
        the wall have both had their say.
    :param radius: radius of the ball instanced on every surviving point.
    :param subdivisions: ico-sphere subdivisions; 1 is the bare icosahedron.
    :param seed: seed of the scatter; the rejection draw uses ``seed + 1``.
    :param elongation: an RPN string in the symbols of
        :data:`PLANE_WAVE_ELONGATION`, or ``None`` for that wave.
    :param color: palette name for the points, turned into a material by
        :func:`~appearance.textures.get_texture`.
    :param material: a ready ``bpy.types.Material`` instead, which takes
        precedence over ``color``. With neither,
        :func:`~appearance.textures.acoustic_texture` is built.

    The dials a scene animates are ``Length``, ``Amplitude``, ``Wavelength``,
    ``Period`` and ``PipeRadius``::

        pipe = AcousticModifier(length=9, pipe_radius=1, amplitude=1,
                                wavelength=tau, period=tau, count=30000,
                                radius=0.012, color="acoustic")
        host = Plane(name="AcousticPipe")
        host.add_mesh_modifier(type='NODES', node_modifier=pipe)
        ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(pipe, "Wavelength"),
                                  from_value=tau, to_value=2,
                                  begin_time=0, transition_time=6)
    """

    def __init__(self, length=9.0, pipe_radius=1.0, amplitude=1.0,
                 wavelength=tau, period=tau, count=30000, radius=0.012,
                 subdivisions=1, seed=0, elongation=None, color=None, mode=None,
                 material=None, name="Acoustic", **kwargs):
        self.length = length
        self.pipe_radius = pipe_radius
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.period = period
        self.count = count
        self.radius = radius
        self.subdivisions = subdivisions
        self.mode = mode
        self.seed = seed
        self.elongation = PLANE_WAVE_ELONGATION if elongation is None else elongation

        if material is None and color is not None:
            material = get_texture(color, **kwargs)
        if material is None:
            # imported here rather than at module level: appearance.textures
            # imports geometry_nodes.nodes, so the other direction has to wait
            # until the module is actually needed
            from appearance.textures import acoustic_texture
            material = acoustic_texture(name="acoustic")
        self.material = material

        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._control_frame(tree)
        candidates, position = self._region_frame(tree, control)
        points = self._sampling_frame(tree, control, candidates, position)
        geometry = self._display_frame(tree, points)
        # every frame sits at the origin and every node carries the coordinate
        # it is actually drawn at - blender shrinks the frames around their
        # children when the tree is opened - so the output is placed by hand too
        self.group_outputs.location = (19 * 200, 3 * 100)
        tree.links.new(geometry, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        if self.mode:
            mode=self.mode
        else:
            mode=0
        frame = Frame(tree, location=(0, 0), label="Control",
                      name="ControlFrame")
        self.length_node = InputValue(tree, location=(1, 5), value=self.length,
                                      name="Length", parent=frame)
        self.amplitude_node = InputValue(tree, location=(1, 3),
                                         value=self.amplitude,
                                         name="Amplitude", parent=frame)
        self.wavelength_node = InputValue(tree, location=(1, 2),
                                          value=self.wavelength,
                                          name="Wavelength", parent=frame)
        self.period_node = InputValue(tree, location=(1, 1), value=self.period,
                                      name="Period", parent=frame)
        self.mode_node = InputInteger(tree,location=(1,0),value=mode,name="Mode",parent=frame)
        # the reason the wave travels with no keyframe in the tree
        self.clock = SceneTime(tree, location=(1, 0), name="Clock",
                               parent=frame)
        return frame

    # ------------------------------------------------------------------
    def _region_frame(self, tree, control):
        frame = Frame(tree, location=(0, 0), label="Region", name="RegionFrame")
        # read in the sampling frame, by the wave and the wall, so it is
        # drawn over there
        position = Position(tree, location=(3, 7), hide=True)

        relay = Reroute(tree, location=(4, -1), ins=self.length_node.std_out,
                        parent=frame)
        # named around "Length": the dials are looked up by a substring of
        # their label, and a node called ...Length would answer to it first

        low = CombineXYZ(tree, location=(6, 0), x=0,
                         y=-self.pipe_radius, z=-self.pipe_radius,
                         name="BoxMin", parent=frame)
        high = CombineXYZ(tree, location=(6, -2), x=relay.std_out,
                          y=self.pipe_radius, z=self.pipe_radius,
                          name="BoxMax", parent=frame)

        # placed by hand rather than scattered through a Mesh to Volume, which
        # ramps its density down below the surface and leaves a box with a soft
        # edge where there should be a face
        cloud = Points(tree, location=(6, 2), count=self.count,
                       name="Candidates", parent=frame)
        draw = RandomValue(tree, location=(7, 0), data_type="FLOAT_VECTOR",
                           min=low.std_out, max=high.std_out, seed=self.seed,
                           name="UniformDraw", parent=frame)
        placed = SetPosition(tree, location=(8, 2), geometry=cloud.geometry_out,
                             position=draw.std_out, name="Uniform",
                             parent=frame)
        return placed.geometry_out, position

    # ------------------------------------------------------------------
    def _sampling_frame(self, tree, control, candidates, position):
        frame = Frame(tree, location=(0, 0), label="Sampling",
                      name="SamplingFrame")
        frame.add(position)

        auxiliaries={"x": "pos_x", "y": "pos_y", "z": "pos_z"}
        if self.mode is not None:
            # wave length is computed from node
            auxiliaries["wavelength"]="len,1,4,/,mode,2,/,+,/"
            inputs = ["pos", "amplitude", "period", "time", "mode", "len"]
        else:
            inputs = ["pos", "amplitude", "wavelength", "period", "time", "mode", "len"]
        auxiliaries["elongation"]= self.elongation # add elongation last, that it can relate to auxiliary wavelength

        # `x`, `y`, `z` so that the formula can be written in the coordinates
        # rather than in pos_x, pos_y, pos_z
        wave = make_function(
            tree, location=(4, 5), parent=frame,
            functions={"intensity": "elongation,2,**",
                       "elongation": "elongation"},
            inputs=inputs,
            aux_functions=auxiliaries,
            outputs=["intensity","elongation"],
            vectors=["pos"],
            scalars=["amplitude", "wavelength", "period", "time",
                     "x", "y", "z", "intensity", "elongation","len"],
            integers=["mode"],
            name="WaveComputer", hide=False)
        tree.links.new(position.std_out, wave.inputs["pos"])

        links = [("amplitude", self.amplitude_node),
                             ("period", self.period_node),
                             ("time", self.clock),
                             ("mode", self.mode_node),
                             ("len",self.length_node)]
        if self.mode is None:
            links.append(("wavelength", self.wavelength_node))
        for socket, dial in links:
            tree.links.new(dial.std_out, wave.inputs[socket])

        # one uniform draw per point, compared against s^2: the point survives
        # where its draw falls under the curve
        draw = RandomValue(tree, location=(5, 9), data_type="FLOAT",
                           min=-1, max=1.0, seed=self.seed + 1,
                           name="RejectionDraw", parent=frame)
        test = make_function(tree, location=(7, 8), parent=frame,
                             functions={"reject": "u,f,amp,+,2,/,>"},
                             inputs=["u", "f","amp"], outputs=["reject"],
                             scalars=["u", "f", "reject","amp"],
                             name="RejectionTest", hide=False)
        tree.links.new(draw.std_out, test.inputs["u"])
        tree.links.new(wave.outputs["elongation"], test.inputs["f"])
        tree.links.new(self.amplitude_node.std_out, test.inputs["amp"])

        # the pipe wall: true for the candidates outside the cylinder
        wall = make_function(
            tree, location=(6, 5), parent=frame,
            functions={"outside": "pos_y,pos_y,*,pos_z,pos_z,*,+,rad,rad,*,>"},
            inputs=["pos", "rad"], outputs=["outside"],
            vectors=["pos"], scalars=["rad", "outside"],
            name="PipeWall", hide=False)
        tree.links.new(position.std_out, wall.inputs["pos"])
        # left out of the control frame on purpose: it is a dial of the
        # *region*, and it sits by the wall it is read by
        self.pipe_radius_node = InputValue(tree, location=(5, 3),
                                           value=self.pipe_radius,
                                           name="PipeRadius")
        tree.links.new(self.pipe_radius_node.std_out, wall.inputs["rad"])

        either = BooleanMath(tree, location=(9, 6), operation="OR",
                             inputs0=test.outputs["reject"],
                             inputs1=wall.outputs["outside"],
                             name="RejectOrOutside", hide=False, parent=frame)
        cull = DeleteGeometry(tree, location=(10, 8), domain="POINT",
                              geometry=candidates, selection=either.std_out,
                              name="Reject", parent=frame)
        points = cull.geometry_out

        loudness = StoreNamedAttribute(tree, location=(11, 10),
                                       name="Amplitude", data_type="FLOAT",
                                       domain="POINT",
                                       value=self.amplitude_node.std_out,
                                       parent=frame)
        tree.links.new(points, loudness.geometry_in)
        intensity = StoreNamedAttribute(tree, location=(12, 10),
                                        name="Intensity", data_type="FLOAT",
                                        domain="POINT",
                                        value=wave.outputs["intensity"],
                                        parent=frame)
        tree.links.new(loudness.geometry_out, intensity.geometry_in)
        result = StoreNamedAttribute(tree, location=(13, 10), name="Elongation",
                                     data_type="FLOAT", domain="POINT",
                                     value=wave.outputs["elongation"],
                                     parent=frame)
        tree.links.new(intensity.geometry_out, result.geometry_in)
        return result.geometry_out

    # ------------------------------------------------------------------
    def _display_frame(self, tree, points):
        frame = Frame(tree, location=(0, 0), label="Display",
                      name="DisplayFrame")
        ball = IcoSphere(tree, location=(14, 2), radius=self.radius,
                         subdivisions=self.subdivisions, name="Ball",
                         parent=frame)
        instances = InstanceOnPoints(tree, location=(15, 3), points=points,
                                     instance=ball.geometry_out,
                                     name="Instances", parent=frame)
        # realised rather than left as instances, so that the three stored
        # attributes reach the shader on the mesh domain it reads them from
        realized = RealizeInstances(tree, location=(16, 3),
                                    geometry=instances.geometry_out,
                                    name="Realize", parent=frame)
        painted = SetMaterial(tree, location=(17, 3),
                              geometry=realized.geometry_out,
                              material=self.material, name="PaintPoints",
                              parent=frame)
        self.materials.append(painted.material)
        return painted.geometry_out

r"""The Airy disc: what a circular aperture makes of a point source.

One self-contained modifier, :class:`AiryDiscModifier`, which paints the
diffracted intensity onto a round disc of unit radius, in one colour.
"""
import numpy as np

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (BESSEL_OPS, Frame, Grid, InputValue,
                                  MergeByDistance, Position, SetMaterial,
                                  SetPosition, SetShadeSmooth,
                                  StoreNamedAttribute, make_function)

pi = np.pi
tau = 2 * pi

#: Zeros of J_1, ``scipy.special.jn_zeros(1, 6)`` - the arguments v at which
#: the dark rings sit. The first is the edge of the disc proper; ``rings``
#: picks one of them and the aperture is solved to land it on the rim.
J1_ZEROS = (3.8317059702075125, 7.015586669815619, 10.173468135062722,
            13.323691936314223, 16.470630050877634, 19.615858510468243)

#: The first of them, on its own, because it is the one with a name.
FIRST_ZERO = J1_ZEROS[0]

#: Zeros of J_2, ``scipy.special.jn_zeros(2, 5)`` - the arguments v at which
#: the *bright* rings peak, since d/dv (J_1(v)/v) = -J_2(v)/v. The central
#: disc is not among them: its peak is v = 0.
J2_ZEROS = (5.135622301840683, 8.417244140399866, 11.61984117214906,
            14.795951782351262, 17.959819494987826)

#: 2 J_1(v)/v, as an RPN string for ``make_function``. ``vc`` is v held off
#: zero, since the quotient is 0/0 on the axis and 1 in the limit.
AIRY_AMPLITUDE = "2,vc,j1,*,vc,/"


class AiryDiscModifier(GeometryNodesModifier):
    r"""The Airy pattern of a circular aperture, on a round disc, in one colour.

    A circular hole of radius a, lit by light of wavelength lambda, does not
    image a point source as a point. What lands on a screen a distance L
    behind it is

    .. math::
        I(\rho) = I_0\left(\frac{2J_1(v)}{v}\right)^{2},
        \qquad v = \frac{2\pi a}{\lambda}\sin\theta,
        \qquad \sin\theta = \frac{\rho}{\sqrt{\rho^2 + L^2}},

    a bright central disc ringed by a set of much fainter ones, the first
    dark ring at :math:`v = 3.832`. The geometry is a *polar* mesh - a
    parameter grid bent into a disc of radius ``Radius`` and welded at its
    seam and at its centre - which is the mesh this pattern wants: the field
    depends on nothing but :math:`\rho`, so a row of the grid is a ring and
    ``angular`` alone decides how round the thing comes out.

    The colour is one hue and the minima are black. The ramp has two stops,
    ``dark`` at zero and ``color`` at ``clip``, so every dark ring is
    genuinely black rather than a darker shade of the colour, and everything
    above ``clip`` is the flat colour - the core saturates and the range
    below it is spent entirely on the rings. The ramp is fed the attribute
    ``Shade``, which is ``Intensity`` raised to ``Gamma``; the raw
    ``Intensity`` is stored beside it for anything that wants the physical
    number.

    That leaves two ways of making the halo brighter, and they do different
    things. ``Gamma`` is an exposure: blender's view transform already takes
    the linear colour to roughly its own 1/2.2 power, so ``Gamma = 2.2``
    shows the intensity itself and 1 is the photographic exposure, while
    anything lower lifts the whole pattern and flattens the fall-off between
    the rings. ``clip`` instead throws away the top of the range, which the
    core does not need - it is saturated either way - and gives it to the
    rings. Reach for ``clip`` first.

    :math:`J_1` is the polynomial approximation of
    :data:`~geometry_nodes.nodes.BESSEL_OPS`, so the formula says ``vc,j1``
    and the tree carries no table. The quotient is held off the axis by
    ``epsilon`` - 2J_1(v)/v is 0/0 at v = 0 and 1 in the limit, and the
    plateau that the clamp leaves behind is far smaller than one vertex.

    :param radius: the disc, in the object's own units. One by default,
        because the thing this is built for is an *instance*: dropped into
        :class:`~objects.logo.LogoFromInstances` in place of a sphere, it is
        scaled by the logo and has to be the size a unit sphere is.
    :param rings: how many rings of the pattern the disc holds, counting the
        central disc as the first. It is not a dial - it *solves* for the
        aperture that lands the right feature of the pattern exactly on the
        rim and then gets out of the way - and together with ``edge`` it is
        the pair to reach for when the picture matters more than the optics.
        ``None`` leaves ``aperture`` as given.
    :param edge: what sits on the rim. ``"peak"`` puts the brightest point
        of the outermost ring there, which is where two instances set side
        by side touch ring to ring; ``"dark"`` puts the dark ring that ends
        it there instead, so the rim is a zero of the field and the object
        has no visible edge at all. ``"peak"`` needs ``rings`` of two or
        more - the first ring is the central disc and its peak is the centre.
    :param aperture: a, the radius of the hole. Ignored when ``rings`` is set.
    :param wavelength: lambda, in the same units as the aperture.
    :param distance: L, from the aperture to the screen.
    :param radial: vertices from the centre to the rim. The rings get
        narrower outwards, so this is what decides how far out the pattern
        stays honest rather than aliased.
    :param angular: vertices around. This is the roundness: the rim is a
        polygon with this many sides.
    :param gamma: the exposure exponent, I^gamma, applied before the view
        transform rather than after it.
    :param epsilon: the floor under v before the division.
    :param color: palette name of the one hue, at the top of the ramp.
    :param dark: palette name at the bottom of it, where the minima land.
    :param clip: the shade at which the ramp reaches full colour; everything
        brighter is that same flat colour. This is the dial that brings the
        faint rings up without touching the exposure, and it works where
        gamma cannot: at a quarter the first ring sits a fourteenth of the
        way up the ramp rather than a sixtieth, and the core saturates
        instead of keeping a range it does not need.
    :param gradient: ``{position: palette name}`` or ``{position: rgba}``
        instead, for anything that wants more than two stops.
    :param shade_smooth: smooth shading. Flat shading makes the facets of the
        outer rings visible as a moire.
    :param material: a ready ``bpy.types.Material`` instead of the ramp.

    As an instance in the logo, which is what ``radius = 1`` is for::

        airy = AiryDiscModifier(rings=3, edge="peak", color="red",
                                radial=160, angular=128)
        disc = {"mesh": ibpy.create_mesh([[0, 0, 0]]), "geo_node_modifier": airy}
        logo = LogoFromInstances(instance=BObject, details=6, scale=[3] * 3,
                                 rotation_euler=[pi / 2, 0, 0],
                                 kwargs_red=disc, kwargs_green=disc,
                                 kwargs_blue=disc)

    The dials a scene animates are ``Radius``, ``Aperture``, ``Wavelength``,
    ``Distance`` and ``Gamma``::

        ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(airy, "Gamma"),
                                  from_value=2.2, to_value=1.0,
                                  begin_time=0, transition_time=6)
    """

    def __init__(self, radius=1.0, rings=3, edge="peak", aperture=0.25,
                 wavelength=0.02, distance=10.0, radial=300, angular=256,
                 gamma=1.0, epsilon=1e-3, color="important",
                 dark="background", clip=0.25, gradient=None,
                 shade_smooth=True, material=None, name="AiryDisc", **kwargs):
        self.radius = radius
        self.wavelength = wavelength
        self.distance = distance
        if rings is not None:
            if edge == "dark":
                rim = J1_ZEROS[rings - 1]
            elif rings < 2:
                raise ValueError("the first ring is the central disc and its "
                                 "peak is the centre; edge='peak' needs "
                                 "rings >= 2")
            else:
                rim = J2_ZEROS[rings - 2]
            # the aperture that lands v = rim on the rim: v is
            # 2 pi a sin(theta) / lambda, and sin(theta) out there is what
            # the screen distance makes of the radius
            sine = radius / np.sqrt(radius ** 2 + distance ** 2)
            aperture = rim * wavelength / (tau * sine)
        self.aperture = aperture
        self.rings = rings
        self.edge = edge
        self.radial = int(radial)
        self.angular = int(angular)
        self.gamma = gamma
        self.epsilon = epsilon
        self.shade_smooth = shade_smooth

        if material is None:
            # imported here rather than at module level: appearance.textures
            # imports geometry_nodes.nodes, so the other direction has to wait
            # until the module is actually needed
            from appearance.textures import gradient_from_attribute
            from utils.constants import COLOR_NAMES, COLORS_SCALED
            if gradient is None:
                gradient = {0.0: dark, clip: color}
            gradient = {position: (COLORS_SCALED[COLOR_NAMES.index(value)]
                                   if isinstance(value, str) else value)
                        for position, value in gradient.items()}
            kwargs.setdefault("emission", 5.0)
            kwargs.setdefault("roughness", 1.0)
            # black specular, or the world lights the dark half of the ramp
            # and the disc reads as a grey coin with a bright spot on it
            kwargs.setdefault("specular_tint", [0, 0, 0, 1])
            material = gradient_from_attribute(name=name + "Texture",
                                               attr_name="Shade",
                                               function="fac",
                                               gradient=gradient, **kwargs)
        self.material = material

        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._control_frame(tree)
        disc = self._disc_frame(tree, control)
        painted = self._intensity_frame(tree, control, disc)
        geometry = self._display_frame(tree, painted)
        # every frame sits at the origin and every node carries the coordinate
        # it is actually drawn at - blender shrinks the frames around their
        # children when the tree is opened - so the output is placed by hand
        self.group_outputs.location = (13 * 200, 0)
        tree.links.new(geometry, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        frame = Frame(tree, location=(0, 0), label="Control",
                      name="ControlFrame")
        radius = InputValue(tree, location=(1, 4), value=self.radius,
                            name="Radius", parent=frame)
        aperture = InputValue(tree, location=(1, 3), value=self.aperture,
                              name="Aperture", parent=frame)
        wavelength = InputValue(tree, location=(1, 2), value=self.wavelength,
                                name="Wavelength", parent=frame)
        distance = InputValue(tree, location=(1, 1), value=self.distance,
                              name="Distance", parent=frame)
        gamma = InputValue(tree, location=(1, 0), value=self.gamma,
                           name="Gamma", parent=frame)
        return {"radius": radius.std_out, "aperture": aperture.std_out,
                "wavelength": wavelength.std_out, "dist": distance.std_out,
                "gamma": gamma.std_out}

    # ------------------------------------------------------------------
    def _disc_frame(self, tree, control):
        frame = Frame(tree, location=(0, 0), label="Disc", name="DiscFrame")
        grid = Grid(tree, location=(3, 1), size_x=1, size_y=1,
                    vertices_x=self.radial, vertices_y=self.angular,
                    name="ParameterGrid", parent=frame)
        # the grid's coordinates are a parameter domain and nothing about them
        # is a position yet, which is why Set Position is used in its absolute
        # mode below rather than as an offset
        cell = Position(tree, location=(3, -1), name="GridPosition",
                        hide=True, parent=frame)
        polar = make_function(tree, location=(4, -1), name="Polar",
                              aux_functions={
                                  "s": "pos_x,0.5,+,radius,*",
                                  "phi": "pos_y,0.5,+,%s,*" % repr(tau)},
                              functions={"disc": ["phi,cos,s,*",
                                                  "phi,sin,s,*", "0"]},
                              inputs=["pos", "radius"], outputs=["disc"],
                              vectors=["pos", "disc"],
                              scalars=["radius", "s", "phi"],
                              parent=frame, hide=True)
        tree.links.new(cell.std_out, polar.inputs["pos"])
        tree.links.new(control["radius"], polar.inputs["radius"])

        bent = SetPosition(tree, location=(5, 1), geometry=grid.geometry_out,
                           position=polar.outputs["disc"], name="BendIntoDisc",
                           parent=frame)
        # two seams: the join at phi = 2 pi, where the grid's opposite edges
        # land on each other, and the centre, where a whole row sits at r = 0
        welded = MergeByDistance(tree, location=(6, 1),
                                 geometry=bent.geometry_out, distance=1e-4,
                                 name="CloseTheSeam", parent=frame)
        return welded.geometry_out

    # ------------------------------------------------------------------
    def _intensity_frame(self, tree, control, disc):
        frame = Frame(tree, location=(0, 0), label="Intensity",
                      name="IntensityFrame")
        # the position *after* the bend, which is the point on the screen
        position = Position(tree, location=(7, 3), name="DiscPosition",
                            hide=True, parent=frame)

        aux = {
            "rho": "pos,length",
            # the exact sine, not rho/L: the disc reaches one unit off the
            # axis with the screen ten away, and the paraxial form would put
            # the outermost ring half a percent too far in
            "sine": "rho,rho,rho,*,dist,dist,*,+,sqrt,/",
            "v": "2,pi,*,aperture,*,wavelength,/,sine,*",
            "vc": "v,%s,max" % repr(self.epsilon),
            "amp": AIRY_AMPLITUDE,
            "raw": "amp,amp,*",
        }
        airy = make_function(
            tree, location=(8, 3), parent=frame,
            functions={"intensity": "raw", "shade": "raw,gamma,**"},
            aux_functions=aux,
            inputs=["pos", "aperture", "wavelength", "dist", "gamma"],
            outputs=["intensity", "shade"],
            vectors=["pos"],
            scalars=["aperture", "wavelength", "dist", "gamma",
                     "intensity", "shade"] + list(aux),
            custom_ops=BESSEL_OPS, name="AiryPattern", hide=False)
        tree.links.new(position.std_out, airy.inputs["pos"])
        for socket in ("aperture", "wavelength", "dist", "gamma"):
            tree.links.new(control[socket], airy.inputs[socket])

        # both are written, and only the second is painted: `Intensity` is the
        # physical number, the one a mirror in numpy can be held against,
        # while `Shade` is what a screen can actually show
        stored = StoreNamedAttribute(tree, location=(9, 1), data_type="FLOAT",
                                     domain="POINT", name="Intensity",
                                     value=airy.outputs["intensity"],
                                     parent=frame)
        tree.links.new(disc, stored.geometry_in)
        shade = StoreNamedAttribute(tree, location=(10, 1), data_type="FLOAT",
                                    domain="POINT", name="Shade",
                                    value=airy.outputs["shade"], parent=frame)
        tree.links.new(stored.geometry_out, shade.geometry_in)
        return shade.geometry_out

    # ------------------------------------------------------------------
    def _display_frame(self, tree, geometry):
        frame = Frame(tree, location=(0, 0), label="Display",
                      name="DisplayFrame")
        if self.shade_smooth:
            smooth = SetShadeSmooth(tree, location=(11, 1), geometry=geometry,
                                    name="Smooth", parent=frame)
            geometry = smooth.geometry_out
        painted = SetMaterial(tree, location=(12, 1), geometry=geometry,
                              material=self.material, name="PaintDisc",
                              parent=frame)
        self.materials.append(painted.material)
        return painted.geometry_out

    # ------------------------------------------------------------------
    def intensity_numpy(self, points, aperture=None, wavelength=None,
                        distance=None):
        """The same I, in numpy - the tree's mirror.

        ``scipy.special.j1`` is the exact Bessel function here, so comparing
        the two also measures what the polynomial approximation costs.

        :param points: ``(n, 2)`` or ``(n, 3)`` array of positions on the
            disc; only x and y are read.
        """
        from scipy.special import j1
        points = np.asarray(points, dtype=float)[:, :2]
        aperture = self.aperture if aperture is None else aperture
        wavelength = self.wavelength if wavelength is None else wavelength
        distance = self.distance if distance is None else distance
        rho = np.linalg.norm(points, axis=1)
        sine = rho / np.sqrt(rho ** 2 + distance ** 2)
        # the same clamp the tree applies before it divides by v
        v = np.maximum(tau * aperture / wavelength * sine, self.epsilon)
        return (2 * j1(v) / v) ** 2

    def first_zero(self, aperture=None, wavelength=None, distance=None):
        """Radius of the first dark ring on the screen, L tan(theta_1).

        The exact form of the 1.22 lambda L / D every optics text quotes,
        which is the same thing with the tangent read as its angle.
        """
        aperture = self.aperture if aperture is None else aperture
        wavelength = self.wavelength if wavelength is None else wavelength
        distance = self.distance if distance is None else distance
        sine = FIRST_ZERO * wavelength / (tau * aperture)
        return distance * sine / np.sqrt(1 - sine ** 2)
