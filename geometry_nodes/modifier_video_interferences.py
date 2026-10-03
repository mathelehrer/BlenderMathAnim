"""Points scattered through a volume with a *prescribed* spatial density.

``video_interferences/tmp.xml`` — the tree authored in the editor — is four
nodes: ``Group Input -> Mesh to Volume -> Distribute Points in Volume -> Group
Output``. It fills a cube with points that are uniformly random, and the
question this module answers is how to bend that uniform cloud into an
arbitrary distribution f(x, y, z).

There are two ways, and both are here, because they fail differently:

``method="grid"``
    Push f into the **volume grid itself**. ``Distribute Points in Volume`` in
    Random mode is openvdb's ``NonUniformPointScatter``: the number of points
    it drops into a voxel is proportional to ``Density`` *times the value
    stored in that voxel*. So a ``Volume Cube`` whose Density input is the
    field f (evaluated at each voxel's position) is sampled with point density
    proportional to f, exactly. One extra node over the xml, and no points are
    ever thrown away.

    The catch is the voxel grid: f is only ever seen at ``resolution**3``
    sample points, so structure finer than a voxel washes out. For an
    interference pattern — whose whole content is fine structure — that is not
    a detail. Measured on two sources 2.4 apart in a box of side 4 with
    lambda = 0.8 (fringes 0.4 apart), as the fraction of the ideal fringe
    contrast that survives:

    ====================  ==========  ==========
    resolution            voxel       contrast
    ====================  ==========  ==========
    12                    0.33        64%
    16                    0.25        80%
    24                    0.17        91%
    32                    0.12        95%
    64                    0.06        99%
    ====================  ==========  ==========

    Six voxels to a fringe is where it stops mattering; one voxel to a fringe
    costs a third of the pattern. (Contrast here is the mean of f over the
    points that were drawn, which is <f^2>/<f> = 0.754 for a perfect draw and
    <f> = 0.508 for a uniform cloud, rescaled to run 0..100%.)

``method="rejection"`` (the default)
    Keep the uniform cloud and **throw points away**: draw a uniform u in
    [0, 1] per point and delete the point where ``u > f``. The survivors are
    distributed as f, with f evaluated at the point's own position —
    analytically, at full precision, with no grid anywhere. Fringes stay sharp
    however thin they get, and the table above reads 100% at any wavelength.

    The catch is that the accepted fraction is the mean of f over the box, so
    the uniform cloud has to be oversampled by 1 / <f> to end up with the
    requested number of points. That factor is worked out in python (Monte
    Carlo over :meth:`SpatialDistributionModifier.density_numpy`), so ``count``
    means what it says either way.

``method="uniform"``
    Do not sample at all. Keep the whole uniform cloud and only *record* f in
    the ``intensity`` attribute, so that a material can show it — as a hue
    through :func:`~appearance.textures.gradient_from_attribute`, or as
    brightness through ``emission_by_density``. The cloud is then a grid of
    probes reading out a field rather than a population drawn from a
    distribution.

    Which is the right way round for a field that is not a probability. An
    instantaneous wave amplitude has a sign, a spherical wave diverges at its
    source, and neither survives being squeezed into the [0, 1] a rejection
    test needs. Emission has no top end, so the same f that cannot be sampled
    can still be looked at. ``count`` is the point count outright here, since
    nothing is thrown away.

    Where the uniform cloud comes from depends on ``shape``: for a box, from
    ``Points`` placed by a random-vector field, which fills it exactly; for an
    incoming mesh (``shape="input"``, the xml's own arrangement), from
    ``Mesh to Volume`` and ``Distribute Points in Volume``, which fills
    anything closed but softens the boundary — see
    :meth:`SpatialDistributionModifier._region_frame`.

The distribution itself is one method, :meth:`SpatialDistributionModifier.density`,
which builds a node group evaluating f at a position field, and its mirror
:meth:`SpatialDistributionModifier.density_numpy`, which evaluates the same f
in numpy. Every subclass writes both, and the pair is what makes the result
checkable: sample the modifier's points in a headless blender and compare them
against ``density_numpy``. Two checks are worth running on a new f — the
histogram along each axis against the marginal of f, and the mean of f over
the *drawn* points against <f^2>/<f>, which the marginals cannot fake because
it is sensitive to where the fringes sit in three dimensions. Every
distribution in this module passes both to within counting noise.

For the two sampling methods f must be normalised to [0, 1] — rejection
sampling against a function that exceeds 1 silently clips, which flattens the
peaks of the distribution instead of erroring. ``method="uniform"`` samples
nothing and so puts no bound on f at all.

Classes:

:class:`SpatialDistributionModifier`
    the machinery, with f = 1, which reproduces ``tmp.xml``.
:class:`InterferenceModifier`
    f = |sum of waves|^2, the point of the exercise: N point sources
    (spherical waves) or N beams (plane waves), with dials for wavelength,
    source positions and phases, so the fringes can be moved in the scene.
:class:`RealInterferenceModifier`
    the same two sources without the time average: f = (sum of
    a_j/r_j sin(k r_j - wt))^2, the instantaneous energy of the real field,
    read out by a uniform cloud and shown as emission. Runs off ``Scene
    Time``, so it moves without a single keyframe.
:class:`GaussianCloudModifier`
    f = a gaussian blob — a distribution with a known answer, which is what
    the verification harness leans on.
:class:`~geometry_nodes.modifier_interferences.AcousticModifier`
    the organ pipe: a *cylinder* of points, drawn from a travelling plane
    wave A sin(2 pi x/lambda - 2 pi t/T). It used to be a subclass of
    :class:`SpatialDistributionModifier` and is now a self-contained modifier
    in :mod:`geometry_nodes.modifier_interferences`, one function per node
    frame, so that the whole pipe can be read in one file.
:class:`PolarGridModifier`
    not a cloud at all, and the only one here that is about *coordinates*
    rather than about a field: a panel ruled with horizontal and vertical
    lines, bent by one dial into the circles and rays of polar coordinates -
    with the colours travelling along with the shapes.
:class:`WaveVisualizationModifier`
    not a cloud either, and the other way round from all of them: a grid whose
    every vertex is *lifted* to u(r, t) = J0(kr) cos wt + Y0(kr) sin wt, so the
    surface is the field rather than a set of probes reading it. The elongation
    is kept in a per-vertex attribute, and the colour comes from
    ``interference_texture`` in its "hankel" model - the same wave, recomputed
    in the shader from the surface's uv.
:class:`DrumModeModifier`
    the same idea as :class:`WaveVisualizationModifier` with a *boundary*: a
    disc clamped at its rim, standing up as one normal mode of the 2+1 wave
    equation, u = J_m(alpha_mn r/a) cos m phi cos omega_mn t. The modes are
    built side by side and an ``Index Switch`` on the ``Mode`` dial picks
    which one is showing, so a scene walks through the overtone series of a
    drum by keyframing one integer.
:class:`FarFieldModifier`
    not a cloud at all: the *directions* a line of equally spaced sources
    radiates into, sin(alpha_n) = n lambda / g, drawn as rays from the centre
    of the array. The prediction to hold an interference pattern against, and
    it moves when the wavelength does.
:class:`BesselVisualizer`
    no field at all, just the two functions the circular wave is made of: two
    lines of vertices along the x axis, one lifted to J0(x) and one to Y0(x),
    drawn to the same scale as a ``CoordinateSystem2`` laid over them.
:class:`Slicer`
    the only one here that *takes* geometry rather than building it: it throws
    away everything on the far side of a plane, so a solid can be opened up
    and its inside shown. One dial, ``SlicerValue``, is where the plane sits,
    and keyframing it cuts the object open on camera.
"""
import numpy as np

from appearance.textures import get_texture
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (BESSEL_OPS, BesselNode, BooleanMath, CombineXYZ, CubeMesh,
                                  CurveCircle, CurveLine, CurveToMesh,
                                  DeleteGeometry, DistributePointsInVolume,
                                  DuplicateElements,
                                  ConeMesh, CylinderMesh, Frame, Grid, IcoSphere, Index, IndexSwitch, InputInteger,
                                  InputValue, InputVector,
                                  InstanceOnPoints, JoinGeometry, MathNode,
                                  MergeByDistance, MeshLine, MeshToCurve, MeshToVolume,
                                  NamedAttribute, Points, Position, RandomValue,
                                  RealizeInstances, Reroute, ResampleCurve, SceneTime,
                                  SeparateXYZ,
                                  SetMaterial, SetPosition,
                                  SetShadeSmooth, StoreNamedAttribute, TransformGeometry, UVSphere,
                                  VolumeCube, WireFrame, bessel_jm_rpn, make_function,
                                  split_rpn, create_geometry_line,
                                  wave_front_gate)
from interface import ibpy
from interface.ibpy import Vector, get_geometry_node_from_modifier
from objects.slide import DEFAULT_MATERIAL
from utils.constants import DEFAULT_SCENE_DURATION, DEFAULT_ANIMATION_TIME
from utils.kwargs import get_from_kwargs

pi = np.pi
tau = 2 * pi


def _vector(value):
    """``Vector`` from a scalar (isotropic), a triple, or a ``Vector``."""
    if isinstance(value, (int, float)):
        return Vector([value, value, value])
    return Vector(value)


# ---------------------------------------------------------------------------
#  the numpy side of an RPN formula
# ---------------------------------------------------------------------------
# ``make_function``'s vocabulary, in numpy, so that a formula handed to a
# modifier as a *string* can be evaluated on both sides: once as nodes, for
# the picture, and once here, for <f> and for the checks the module docstring
# asks for. Only what makes sense on arrays of numbers is here - rotations and
# strings are a node-tree affair.
_RPN_UNARY = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "sgn": np.sign,
    # blender pins the base of LOGARITHM to 10 for "lg"
    "lg": np.log10,
    "round": np.round, "floor": np.floor, "ceil": np.ceil,
    "frac": lambda a: a - np.floor(a),
    "not": lambda a: np.logical_not(a).astype(float),
    # vector ops, which act on the last axis of an (n, 3) array
    "length": lambda a: np.linalg.norm(a, axis=-1),
    "vfloor": np.floor,
    "normalize": lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True),
}

_RPN_BINARY = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide,
    "%": np.mod, "**": np.power, "min": np.minimum, "max": np.maximum,
    "atan2": np.arctan2,
    "<": lambda left, right: (left < right).astype(float),
    ">": lambda left, right: (left > right).astype(float),
    "=": lambda left, right: (left == right).astype(float),
    "and": lambda left, right: np.logical_and(left, right).astype(float),
    "or": lambda left, right: np.logical_or(left, right).astype(float),
    # vector ops
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
    "mod": np.mod,
    # a vector times a *per-point* scalar needs the axis put back
    "scale": lambda left, right: left * (np.expand_dims(right, -1)
                                         if np.ndim(right) else right),
    "dot": lambda left, right: np.sum(left * right, axis=-1),
    "cross": lambda left, right: np.cross(left, right),
}


def rpn_numpy(expression, variables):
    """Evaluate one of :func:`~geometry_nodes.nodes.make_function`'s RPN
    expressions in numpy.

    The mirror of the node group, for a formula that only exists as a string -
    a modifier that takes its distribution that way cannot measure <f> (and so
    the number of candidates the sampler has to draw) without evaluating it.
    Values in ``variables`` may be scalars or arrays, and numpy's broadcasting
    does the rest, so one call evaluates the formula at every point of an
    ``(n, 3)`` cloud at once.

    Tokens are looked up **as operators first**, exactly as ``make_function``
    does it, so the same trap is here: a variable called ``length`` is the
    length of a vector and never the caller's variable. Name them around
    :data:`~interface.ibpy.OPERATORS`.

    :param expression: the RPN string, e.g. ``"a,x,*,sin"``.
    :param variables: ``{name: value}``; components of a vector variable are
        also reachable as ``name_x``, ``name_y``, ``name_z``.
    :raises ValueError: on an unknown token or an expression that does not
        leave exactly one value on the stack - which is what a formula with a
        typo in it does, and is worth hearing about before blender silently
        builds something else.
    """
    channels = dict(variables)
    for key, value in list(variables.items()):
        if np.ndim(value) and np.shape(value)[-1] == 3:
            for i, component in enumerate("xyz"):
                channels["%s_%s" % (key, component)] = np.asarray(value)[..., i]

    stack = []
    for token in split_rpn(expression):
        if token in _RPN_UNARY:
            if not stack:
                raise ValueError("%r has nothing to apply %r to"
                                 % (expression, token))
            stack.append(_RPN_UNARY[token](stack.pop()))
        elif token in _RPN_BINARY:
            if len(stack) < 2:
                raise ValueError("%r has no two operands for %r"
                                 % (expression, token))
            right, left = stack.pop(), stack.pop()
            stack.append(_RPN_BINARY[token](left, right))
        elif token in channels:
            stack.append(channels[token])
        elif token == "pi":
            stack.append(np.pi)
        else:
            try:
                stack.append(float(token))
            except ValueError:
                raise ValueError("%r in %r is neither an operator, one of the "
                                 "variables %s, nor a number"
                                 % (token, expression, sorted(channels)))
    if len(stack) != 1:
        raise ValueError("%r leaves %d values on the stack, not one"
                         % (expression, len(stack)))
    return stack[0]


class SpatialDistributionModifier(GeometryNodesModifier):
    """Points filling a volume, drawn from a spatial distribution f(x, y, z).

    With the default f = 1 this is the uniform cloud of
    ``video_interferences/tmp.xml`` - literally that tree when
    ``shape="input"``. Subclasses override :meth:`density` (and
    :meth:`density_numpy`) to shape it.

    :param size: side lengths of the box, a scalar or a triple.
    :param center: where the box sits.
    :param shape: ``"cube"`` builds the box inside the tree, so the modifier
        can be hung on any object (a ``Plane``, as the other modifiers here
        are). ``"input"`` takes the incoming geometry instead, which is the
        xml's arrangement and lets any closed mesh be filled - a sphere, a
        torus, a letter. ``size`` and ``center`` still say which region the
        density function is normalised over, so keep them around the mesh.
    :param method: ``"rejection"``, ``"grid"`` or ``"uniform"``, as described
        in the module docstring. ``"grid"`` fills the box and only the box, so
        it cannot be combined with ``shape="input"``.
    :param count: how many points to end up with. All three methods honour it
        to within counting noise; ``shape="input"`` lands 13-14% under.
    :param resolution: voxels per side of the ``Volume Cube`` (``"grid"``).
    :param voxel_amount: voxels along the longest side of the bounding box
        when the incoming mesh is converted to a volume (``shape="input"``).
    :param seed: seed of the scatter; the rejection draw uses ``seed + 1``.
    :param radius: radius of the sphere instanced on every point.
    :param subdivisions: ico-sphere subdivisions; 1 is the bare icosahedron,
        12 vertices, and is plenty when the points are small on screen.
    :param color: palette name for the points, or ``None`` to leave the
        geometry unpainted.
    :param color_by_density: paint the points by the value of f at their own
        position instead, through a colour ramp. The value is stored in the
        ``intensity`` attribute either way, so a scene can build its own
        material on it.
    :param gradient: ``{position: rgba}`` stops of that ramp.
    :param emission_by_density: the third way of painting them: one flat
        ``color``, with f driving the *emission strength* instead of the hue
        (see :func:`~appearance.textures.emission_from_attribute`). What
        ``color_by_density`` cannot do is show a field with no upper bound,
        because a ramp has to end somewhere; emission does not, so the peaks
        simply blow out and a bloom in the compositor turns them into light.
        The ``emission`` keyword means the same thing on both paths - how
        bright - but it lands differently: a constant strength for the ramp,
        the factor f is multiplied by here.
    :param material: a ready ``bpy.types.Material`` to paint the points with,
        which takes precedence over the three paths above. The way to hand the
        cloud a material a scene built for itself - see
        :func:`~appearance.textures.acoustic_texture`, which reads the same
        ``intensity`` attribute but does more with it than a ramp can.
    :param box_color: palette name for a wireframe of the box, or ``None``
        for no box.
    :param box_radius: tube radius of that wireframe.
    """

    def __init__(self, size=4.0, center=(0, 0, 0), shape="cube",
                 method="rejection", count=20000, resolution=64,
                 voxel_amount=64.0, seed=0, radius=0.02, subdivisions=1,
                 color="drawing", color_by_density=False, gradient=None,
                 emission_by_density=False, material=None,
                 box_color=None, box_radius=0.01,
                 name="SpatialDistribution", **kwargs):
        if method not in ("rejection", "grid", "uniform"):
            raise ValueError("method is 'rejection', 'grid' or 'uniform', "
                             "not %r" % method)
        if shape not in ("cube", "input"):
            raise ValueError("shape is 'cube' or 'input', not %r" % shape)
        if method == "grid" and shape == "input":
            # the grid method samples a Volume Cube spanning size/center; there
            # is no incoming mesh in that tree to honour, and silently ignoring
            # one is worse than saying so
            raise ValueError("method='grid' fills the box given by size and "
                             "center; it cannot fill an incoming mesh. Use "
                             "method='rejection' with shape='input'.")

        self.size = _vector(size)
        self.center = Vector(center)
        self.shape = shape
        self.method = method
        self.count = count
        self.resolution = resolution
        self.voxel_amount = voxel_amount
        self.seed = seed
        self.radius = radius
        self.subdivisions = subdivisions
        self.color = color
        self.color_by_density = color_by_density
        self.gradient = gradient or {0: [0, 0, 0.35, 1], 0.5: [0.6, 0.1, 0.5, 1],
                                     1: [1, 0.95, 0.6, 1]}
        self.emission_by_density = emission_by_density
        self.material = material
        self.box_color = box_color
        self.box_radius = box_radius
        self.kwargs = kwargs

        self.box_min = self.center - self.size / 2
        self.box_max = self.center + self.size / 2
        self.box_volume = self.size.x * self.size.y * self.size.z

        # <f> is the acceptance rate of the rejection draw, so the uniform
        # cloud has to be oversampled by 1 / <f> to leave `count` survivors.
        # The box path can say that as an exact number of candidate points;
        # the volume path can only ask for points per unit volume.
        if method == "uniform":
            # nothing is thrown away, so `count` is the point count outright -
            # and f, never being compared against a uniform draw, is under no
            # obligation to stay inside [0, 1]. Measuring <f> here would only
            # print a warning about a bound that does not apply.
            self.mean_density = 1.0
        else:
            self.mean_density = self.estimate_mean_density()
        self.candidates = int(round(count / self.mean_density))
        self.point_density = self.candidates / self.box_volume

        super().__init__(name=name, automatic_layout=False,
                         group_input=(shape == "input"))

        # ------------------------------------------------------------------

    def create_node(self, tree, **kwargs):
        candidates, position = self._region_frame(tree)
        points = self._sampling_frame(tree, candidates, position)
        geometry = self._display_frame(tree, points)
        # the coordinates in the three frames are absolute (see
        # :meth:`_region_frame`), so the output node is placed by hand too
        self.group_outputs.location = (19 * 200, 3 * 100)
        tree.links.new(geometry, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    # the distribution: one method in nodes, one in numpy, same function
    # ------------------------------------------------------------------
    def density(self, tree, position, location=(0, 0)):
        """Socket carrying f evaluated at the ``position`` field, or ``None``.

        ``None`` means "uniform", and takes the rejection stage out of the
        tree altogether rather than testing against a constant 1.
        """
        return None

    def density_numpy(self, points):
        """f evaluated at an ``(n, 3)`` array of positions - the same f."""
        return np.ones(len(points))

    def constraint(self, tree, position, location=(0, 0)):
        """Socket that is **true where a point is to be thrown away**, or ``None``.

        The other half of :meth:`density`, and the one that is about the
        *region* rather than the distribution: the candidates are drawn in a
        box, and a subclass that wants them inside something else - a
        cylinder, a sphere, an organ pipe - says so here. The socket is OR-ed
        with the rejection test in :meth:`_sampling_frame`, so the geometry is culled
        by the same ``Delete Geometry`` node as the distribution and costs
        nothing extra.

        ``None``, the default, is "the box, and nothing more".
        """
        return None

    def region_bounds(self, tree, frame=None, location=(0, 0)):
        """The two corners of the box the candidates are drawn in, as *sockets*.

        ``(None, None)``, the default, leaves the ``UniformDraw`` node with the
        constants that ``size`` and ``center`` work out to, which is a box that
        cannot move once the tree is built. A subclass that wants one of its
        sides on a dial - a pipe whose length a scene ramps - builds the nodes
        for it here, inside ``frame``, and hands back the corners.

        Only the box path uses this; the two volume paths scatter into a
        volume rather than drawing coordinates.
        """
        return None, None

    def extra_attributes(self, tree, location=(0, 0)):
        """``[(name, socket)]`` to hang on the surviving points, besides f.

        ``intensity`` is stored by :meth:`_sampling_frame` for every modifier
        here, because a material needs it; anything else a *material* has to
        know is listed here instead. A wave modifier stores its amplitude that
        way, so the shader can tell a quiet wave from a loud one rather than
        having to guess what the peak of ``intensity`` means.
        """
        return []

    def estimate_mean_density(self, samples=200000, seed=1234):
        """<f> over the box by Monte Carlo, which is the acceptance rate.

        Also the only place the [0, 1] contract on f is checked; a peak above
        1 is not an error blender can raise, it just quietly clips.
        """
        rng = np.random.default_rng(seed)
        points = rng.uniform(np.array(self.box_min), np.array(self.box_max),
                             size=(samples, 3))
        values = np.asarray(self.density_numpy(points), dtype=float)
        peak = values.max() if len(values) else 1.0
        if peak > 1 + 1e-6:
            print("Warning: %s density peaks at %.3f > 1; the distribution "
                  "will be clipped there." % (type(self).__name__, peak))
        return float(np.clip(values.mean(), 1e-6, 1.0))

    def expected_count(self):
        """Points the tree should produce - what ``count`` promises.

        ``shape="input"`` comes out 13-14% under this, because the volume the
        incoming mesh is turned into is thinner than the mesh near its
        surface; see :meth:`_region_frame`.
        """
        return self.candidates * self.mean_density


    # ------------------------------------------------------------------
    def _region_frame(self, tree):
        """The cloud the distribution is carved out of, and the ``Position`` field.

        Three shapes of tree, one per (method, shape) combination:

        ``method="grid"``
            ``Volume Cube`` with f on its Density input, scattered. The
            distribution is already in the points that come out, so the
            sampling frame downstream has nothing left to do.

        ``method="rejection"`` or ``"uniform"``, ``shape="cube"``
            ``count / <f>`` points placed by hand, each at a uniform random
            position in the box (and ``<f> = 1``, so exactly ``count`` of them,
            when nothing downstream is going to be thrown away). Not the xml's
            route through a volume, and
            deliberately: ``Mesh to Volume`` ramps the density down over the
            ``Interior Band Width`` below the surface, which thins the outer
            ~5% of a box by 15-20% and costs 13% of the points overall.
            Nobody notices that on a blob, but on a *box* it is a visible
            soft edge where there should be a face.

        ``method="rejection"`` or ``"uniform"``, ``shape="input"``
            the xml's route: the incoming mesh converted to a volume and
            scattered uniformly. Any closed mesh can be filled this way, at
            the price of that same soft boundary - and of a point count that
            comes out 13-14% under ``count``.
        """
        # every frame here sits at the origin and every node carries the
        # coordinate it is actually drawn at, so that the two ways a node ends
        # up in a frame - the ``parent`` keyword and ``Frame.add`` - agree.
        # Blender shrinks a frame around its children when the tree is opened,
        # which is what gives the frames their own place.
        frame = Frame(tree, location=(0, 0), label="Region", name="RegionFrame")
        position = Position(tree, location=(3, 7), hide=True)

        if self.method == "grid":
            # the grid evaluates f in this frame, so the field belongs here;
            # the other two read it downstream, in the sampling frame
            frame.add(position)
            position.node.location = (0, 0)
            density = self.density(tree, position.std_out, location=(1, 0))
            if density is not None:
                density.node.parent = frame.node
            # A Volume Cube puts its `resolution` samples *on* Min and Max, and
            # every sample owns a voxel around itself, so the region it
            # actually fills sticks out half a voxel on all six sides: at
            # resolution 64 that is 4.8% too many points, 4.6% of them outside
            # the box the caller asked for. Pulling Min and Max in by half a
            # voxel makes the filled region exactly the box again.
            inset = self.size / (2 * self.resolution)
            grid = VolumeCube(tree, location=(3, 0),
                              density=1.0 if density is None else density,
                              min=self.box_min + inset, max=self.box_max - inset,
                              resolution_x=self.resolution,
                              resolution_y=self.resolution,
                              resolution_z=self.resolution,
                              name="DensityGrid", parent=frame)
            scatter = DistributePointsInVolume(tree, location=(4, 0),
                                               volume=grid.geometry_out,
                                               mode="Random",
                                               density=self.point_density,
                                               seed=self.seed,
                                               name="Scatter", parent=frame)
            return scatter.geometry_out, position

        if self.shape == "input":
            volume = MeshToVolume(tree, location=(3, 0),
                                  mesh=self.group_inputs.outputs[0], density=1.0,
                                  resolution_mode="Amount",
                                  voxel_amount=self.voxel_amount,
                                  name="Volume", parent=frame)
            scatter = DistributePointsInVolume(tree, location=(4, 0),
                                               volume=volume.geometry_out,
                                               mode="Random",
                                               density=self.point_density,
                                               seed=self.seed,
                                               name="Scatter", parent=frame)
            return scatter.geometry_out, position

        cloud = Points(tree, location=(6, 2), count=self.candidates,
                       name="Candidates", parent=frame)
        low, high = self.region_bounds(tree, frame=frame, location=(4, -1))
        draw = RandomValue(tree, location=(7, 0), data_type="FLOAT_VECTOR",
                           min=self.box_min if low is None else low,
                           max=self.box_max if high is None else high,
                           seed=self.seed, name="UniformDraw", parent=frame)
        placed = SetPosition(tree, location=(8, 2), geometry=cloud.geometry_out,
                             position=draw.std_out, name="Uniform", parent=frame)
        return placed.geometry_out, position

    # ------------------------------------------------------------------
    def _sampling_frame(self, tree, candidates, position):
        """Cull the uniform cloud by f - the rejection step, and nothing else.

        For ``method="grid"`` there is nothing to cull (the grid carries the
        distribution), so this only hangs the ``intensity`` attribute on the
        points when a scene asks to be able to colour by it. For
        ``method="uniform"`` there is nothing to cull either, and hanging that
        attribute on the points is the entire job: it is the only trace f
        leaves in the tree, and what the material downstream reads.

        A subclass that also confines the points to a *region* smaller than
        the box says so in :meth:`constraint`, and the two selections are
        OR-ed together into the one ``Delete Geometry``: a point goes if its
        uniform draw beat f, **or** if it is outside the region. Both
        conditions are fields evaluated on the point itself, so the whole
        cull is still a single node.
        """
        frame = Frame(tree, location=(0, 0), label="Sampling",
                      name="SamplingFrame")
        points = candidates
        density = None
        discard = None

        if self.method != "grid":
            # the field is read here, by f and by the region test, so this is
            # where it is drawn - it was left out of the region frame for it
            frame.add(position)

        if self.method == "rejection":
            density = self.density(tree, position.std_out, location=(4, 5))
            if density is not None:
                density.node.parent = frame.node
                # one uniform draw per point, compared against f: the point
                # survives where its draw falls under the curve
                draw = RandomValue(tree, location=(5, 9), data_type="FLOAT",
                                   min=0.0, max=1.0, seed=self.seed + 1,
                                   name="RejectionDraw", parent=frame)
                test = make_function(tree, location=(7, 8),
                                     functions={"reject": "u,f,>"},
                                     inputs=["u", "f"], outputs=["reject"],
                                     scalars=["u", "f", "reject"],
                                     name="RejectionTest", hide=False)
                test.parent = frame.node
                tree.links.new(draw.std_out, test.inputs["u"])
                tree.links.new(density, test.inputs["f"])
                discard = test.outputs["reject"]
        elif self.method == "uniform" or self.color_by_density \
                or self.emission_by_density:
            density = self.density(tree, position.std_out, location=(4, 5))
            if density is not None:
                density.node.parent = frame.node

        # the region the points are confined to, on top of the box they were
        # drawn in - the pipe wall, for the modifier that has one
        outside = self.constraint(tree, position.std_out, location=(6, 5))
        if outside is not None:
            outside.node.parent = frame.node
            if discard is None:
                discard = outside
            else:
                either = BooleanMath(tree, location=(9, 6), operation="OR",
                                     inputs0=discard, inputs1=outside,
                                     name="RejectOrOutside", hide=False,
                                     parent=frame)
                discard = either.std_out

        if discard is not None:
            cull = DeleteGeometry(tree, location=(11, 8), domain="POINT",
                                  geometry=points, selection=discard,
                                  name="Reject", parent=frame)
            points = cull.geometry_out

        # whatever else the material has to know, one store each, upstream of
        # the intensity store so that the geometry line ends in the same node
        extras = self.extra_attributes(tree, location=(12, 10))
        for i, (attribute, value) in enumerate(extras):
            store = StoreNamedAttribute(tree, location=(13 - len(extras) + i, 10),
                                        name=attribute, data_type="FLOAT",
                                        domain="POINT", value=value)
            store.node.parent = frame.node
            tree.links.new(points, store.geometry_in)
            points = store.geometry_out

        if density is not None:
            # carried on the points so that a material can read it; the value
            # is f at the point's own position, i.e. the local intensity
            store = StoreNamedAttribute(tree, location=(13, 10), name="intensity",
                                        data_type="FLOAT", domain="POINT",
                                        value=density)
            store.node.parent = frame.node
            tree.links.new(points, store.geometry_in)
            points = store.geometry_out

        return points

    # ------------------------------------------------------------------
    def _display_frame(self, tree, points):
        """A small sphere on every point, painted, plus the optional box."""
        frame = Frame(tree, location=(0, 0), label="Display",
                      name="DisplayFrame")
        ball = IcoSphere(tree, location=(14, 2), radius=self.radius,
                         subdivisions=self.subdivisions, name="Ball",
                         parent=frame)
        instances = InstanceOnPoints(tree, location=(15, 3), points=points,
                                     instance=ball.geometry_out,
                                     name="Instances", parent=frame)
        # realised, not left as instances, so that the ``intensity``
        # attribute reaches the shader on the mesh domain it reads
        realized = RealizeInstances(tree, location=(16, 3), name="Realize",
                                    parent=frame)
        tree.links.new(instances.geometry_out, realized.geometry_in)
        geometry = realized.geometry_out

        if self.material is not None:
            # a material the caller built and handed over, ready to go: it
            # reads the `intensity` attribute stored above like the two
            # builders below, only it was not built here
            painted = SetMaterial(tree, location=(17, 3), geometry=geometry,
                                  material=self.material, name="PaintPoints",
                                  parent=frame)
            self.materials.append(painted.material)
            geometry = painted.geometry_out
        elif self.emission_by_density:
            from appearance.textures import emission_from_attribute
            # `emission` says how bright on either painting path, so it is
            # read out of kwargs rather than consumed: the ramp path still
            # needs to forward it to customize_material
            material = emission_from_attribute(name="IntensityEmission",
                                               attr_name="intensity",
                                               attr_type="GEOMETRY",
                                               function="fac",
                                               color=self.color,
                                               strength=self.kwargs.get("emission", 10),
                                               **self.kwargs)
            painted = SetMaterial(tree, location=(17, 3), geometry=geometry,
                                  material=material, name="PaintPoints",
                                  parent=frame)
            self.materials.append(painted.material)
            geometry = painted.geometry_out
        elif self.color_by_density:
            from appearance.textures import gradient_from_attribute
            material = gradient_from_attribute(name="IntensityGradient",
                                               attr_name="intensity",
                                               attr_type="GEOMETRY",
                                               function="fac",
                                               gradient=self.gradient,
                                               **self.kwargs)
            painted = SetMaterial(tree, location=(17, 3), geometry=geometry,
                                  material=material, name="PaintPoints",
                                  parent=frame)
            self.materials.append(painted.material)
            geometry = painted.geometry_out
        elif self.color is not None:
            painted = SetMaterial(tree, location=(17, 3), geometry=geometry,
                                  material=self.color, name="PaintPoints",
                                  parent=frame, **self.kwargs)
            self.materials.append(painted.material)
            geometry = painted.geometry_out

        if self.box_color is not None:
            box = CubeMesh(tree, location=(14, -1), size=self.size,
                           name="BoxOutline", parent=frame)
            shifted = TransformGeometry(tree, location=(15, -1),
                                        geometry=box.geometry_out,
                                        translation=self.center,
                                        name="PlaceBoxOutline", parent=frame)
            wires = WireFrame(tree, location=(16, -1), radius=self.box_radius,
                              geometry=shifted.geometry_out, name="BoxWires",
                              parent=frame)
            box_paint = SetMaterial(tree, location=(17, -1),
                                    geometry=wires.geometry_out,
                                    material=self.box_color, name="PaintBox",
                                    parent=frame, **self.kwargs)
            self.materials.append(box_paint.material)
            joined = JoinGeometry(tree, location=(18, 3), name="JoinDisplay",
                                  parent=frame)
            tree.links.new(box_paint.geometry_out, joined.geometry_in)
            tree.links.new(geometry, joined.geometry_in)
            geometry = joined.geometry_out

        return geometry


class InterferenceModifier(SpatialDistributionModifier):
    r"""Points distributed as the intensity of N interfering waves.

    The distribution is the one physics writes down for the intensity of a
    superposition of N waves of equal amplitude,

    .. math::
        f(\mathbf r) = \frac{1}{N^2}
            \Big| \sum_j e^{\,i(\varphi_j(\mathbf r) + \delta_j)} \Big|^2
            = \frac{1}{N^2}
              \Big[\big(\textstyle\sum_j \cos\big)^2
                 + \big(\textstyle\sum_j \sin\big)^2\Big],

    normalised by N^2 so that the bright fringes - where all N waves arrive in
    phase - sit at f = 1 and nothing gets clipped. The phase is

    ``wave="spherical"``
        :math:`\varphi_j = k\,|\mathbf r - \mathbf s_j|`, N point sources at
        the positions ``sources``. Two of them give the classic hyperboloid
        fringe surfaces; the amplitude is kept flat rather than falling off as
        1/r, so the fringes stay equally visible across the box (a 1/r
        envelope diverges at the sources, and a rejection test against an
        unbounded f has no normalisation).

    ``wave="plane"``
        :math:`\varphi_j = k\,\hat{\mathbf n}_j\cdot\mathbf r`, N beams
        travelling along the directions ``sources`` (normalised here). Three
        or more non-coplanar beams give a 3D standing-wave lattice - an
        optical lattice, the thing atoms get trapped in.

    Everything the fringes depend on is a node with a name, so a scene can
    animate it through
    ``ibpy.get_geometry_node_from_modifier(modifier, label)``:

    ``WaveNumber``
        k = 2*pi/lambda, a ``Value`` node. Animating it sweeps the fringe
        spacing.
    ``Source0`` ... ``SourceN``
        the source positions (or beam directions), ``Vector`` nodes. Pulling
        two sources apart tightens the fringes; ``change_default_vector``
        animates them.
    ``Phase0`` ... ``PhaseN``
        the phase offsets, ``Value`` nodes. Ramping one by 2*pi marches the
        whole fringe pattern through the box once, which is what a moving
        interference pattern looks like.

    Note that these dials move the *density function*, so the points are
    redrawn each frame rather than flowing: the cloud shimmers, it does not
    advect. For flowing points, animate the phase of a distribution the points
    are *displaced* by instead.

    :param sources: N positions (``wave="spherical"``) or N directions
        (``wave="plane"``).
    :param wavelength: lambda, in the same units as the box.
    :param phases: N phase offsets, default all zero.
    :param sharpness: raise the normalised intensity to this power before
        sampling, f -> f**sharpness. Physically it is a lie; on screen it is
        often the difference between a picture and a haze, because a cloud is
        seen *through*, and every dim point between the camera and a bright
        fringe is another veil over it. The 3D lattice of four crossed beams
        in particular reads as noise at sharpness 1 and as a lattice at 3. It
        stays in [0, 1], so nothing downstream changes - except the price:
        the acceptance rate is <f**sharpness>, which for those four beams
        falls from 0.28 to 0.067, so the rejection sampler has to draw four
        times as many candidates for the same ``count``.
    :param background: raises the floor of the dark fringes, f -> (f + b) /
        (1 + b). ``b = 0`` empties them completely, which is correct and can
        look like a hole; a few percent keeps a haze there.
    """

    def __init__(self, sources=((-1, 0, 0), (1, 0, 0)), wavelength=0.5,
                 phases=None, wave="spherical", sharpness=1.0, background=0.0,
                 name="Interference", **kwargs):
        if wave not in ("spherical", "plane"):
            raise ValueError("wave is 'spherical' or 'plane', not %r" % wave)
        self.wave = wave
        self.wavelength = wavelength
        self.k = tau / wavelength
        self.sharpness = sharpness
        self.background = background
        self.sources = [Vector(s) for s in sources]
        if wave == "plane":
            self.sources = [s.normalized() for s in self.sources]
        self.phases = list(phases) if phases is not None else [0.0] * len(self.sources)
        if len(self.phases) != len(self.sources):
            raise ValueError("one phase per source, got %d and %d"
                             % (len(self.phases), len(self.sources)))
        super().__init__(name=name, **kwargs)

    # ------------------------------------------------------------------
    def density(self, tree, position, location=(0, 0)):
        n = len(self.sources)
        x, y = location

        # phase of each wave at the sampled position, then the real and
        # imaginary part of the summed amplitude
        phase_op = "sub,length" if self.wave == "spherical" else "dot"
        aux = {}
        for j in range(n):
            aux["ph%d" % j] = "pos,s%d,%s,k,*,p%d,+" % (j, phase_op, j)
        aux["cs"] = ",".join("ph%d,cos" % j for j in range(n)) + ",+" * (n - 1)
        aux["sn"] = ",".join("ph%d,sin" % j for j in range(n)) + ",+" * (n - 1)

        # |sum|^2 / N^2, then the sharpening power, then the background
        # floor: (f + b) / (1 + b)
        formula = "cs,cs,*,sn,sn,*,+,%s,/" % (n * n)
        if self.sharpness != 1:
            formula += ",%s,**" % self.sharpness
        if self.background > 0:
            formula += ",%s,+,%s,/" % (self.background, 1 + self.background)

        names = ["pos", "k"] + ["s%d" % j for j in range(n)] \
                + ["p%d" % j for j in range(n)]
        function = make_function(tree, location=location,
                                 functions={"density": formula},
                                 aux_functions=aux,
                                 inputs=names, outputs=["density"],
                                 vectors=["pos"] + ["s%d" % j for j in range(n)],
                                 scalars=["k"] + ["p%d" % j for j in range(n)]
                                         + list(aux) + ["density"],
                                 name="Intensity", hide=True)
        tree.links.new(position, function.inputs["pos"])

        # the dials, built once and reused if density() is called again
        if not hasattr(self, "wave_number"):
            self.wave_number = InputValue(tree, location=(x - 2, y + 1),
                                          value=self.k, name="WaveNumber",
                                          hide=True)
            self.source_nodes = [
                InputVector(tree, location=(x - 2, y - j), vector=source,
                            name="Source%d" % j, hide=True)
                for j, source in enumerate(self.sources)]
            self.phase_nodes = [
                InputValue(tree, location=(x - 2, y - n - j), value=phase,
                           name="Phase%d" % j, hide=True)
                for j, phase in enumerate(self.phases)]

        tree.links.new(self.wave_number.std_out, function.inputs["k"])
        for j in range(n):
            tree.links.new(self.source_nodes[j].std_out,
                           function.inputs["s%d" % j])
            tree.links.new(self.phase_nodes[j].std_out,
                           function.inputs["p%d" % j])
        return function.outputs["density"]

    # ------------------------------------------------------------------
    def density_numpy(self, points):
        points = np.asarray(points, dtype=float)
        real = np.zeros(len(points))
        imaginary = np.zeros(len(points))
        for source, phase in zip(self.sources, self.phases):
            source = np.array(source, dtype=float)
            if self.wave == "spherical":
                argument = self.k * np.linalg.norm(points - source, axis=1) + phase
            else:
                argument = self.k * (points @ source) + phase
            real += np.cos(argument)
            imaginary += np.sin(argument)
        density = (real ** 2 + imaginary ** 2) / len(self.sources) ** 2
        if self.sharpness != 1:
            density = density ** self.sharpness
        if self.background > 0:
            density = (density + self.background) / (1 + self.background)
        return density


class RealInterferenceModifier(SpatialDistributionModifier):
    r"""A uniform cloud reading out the *instantaneous* field of N point sources.

    This is the tree of ``video_interferences/tmp.xml``, and the other half of
    the story :class:`InterferenceModifier` tells. That one draws its points
    from the time-*averaged* intensity, which is what a photographic plate
    records: a still pattern of fringes, and a number in [0, 1] that a
    rejection test can be run against. This one keeps the wave:

    .. math::
        f(\mathbf r, t) = \Big[\sum_j \frac{a_j}{r_j}
                          \sin\!\big(k r_j - \omega t\big)\Big]^2 ,
        \qquad r_j = |\mathbf r - \mathbf s_j| ,

    the square of the summed real amplitude at one instant — the energy in the
    field, not its average. Three things follow, and they are why this needs
    ``method="uniform"`` rather than a sampler:

    * the amplitude falls off as 1/r, the honest spherical-wave envelope
      :class:`InterferenceModifier` drops. So f **diverges at the sources**,
      and no rescaling brings it into the [0, 1] a rejection test needs;
    * f is zero on whole surfaces twice a period — the nodal shells breathe in
      and out — and a *sampler* fed that would empty the box on those frames;
    * so the points stay put and uniform, and carry f as the ``intensity``
      attribute for the material to turn into brightness. The cloud is a fixed
      lattice of probes reading out a field, and the field, not the cloud, is
      what moves.

    Which is also why this one moves **without a single keyframe**: ``wt``
    comes from ``Scene Time -> Seconds`` times ``Frequency``, so the shells
    travel outward at the phase velocity :math:`\omega/k` for as long as the
    animation runs. (Seconds, not frames, so it depends on the scene's frame
    rate being the one ``initialize_blender`` sets from ``FRAME_RATE``.)
    Contrast :class:`InterferenceModifier`, whose fringes only move when a
    scene ramps ``Phase0`` by hand — and which shimmers rather than travels,
    because there each frame redraws the cloud.

    The dials, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, label)``:

    ``WaveNumber``
        k, a ``Value`` node. Note that ramping it alone changes the phase
        velocity :math:`\omega/k`, not just the fringe spacing.
    ``Frequency``
        :math:`\omega`. Ramping this one is a *chirp*, not a change of pitch:
        the tree computes :math:`\omega(t)\,t`, so the wave already in flight
        is re-phased along with the wave being emitted.
    ``Source0`` ... ``SourceN``
        the source positions, ``Vector`` nodes.
    ``Amplitude0`` ... ``AmplitudeN``
        the a_j. Taking one to zero fades an interference pattern down to a
        single spherical wave, which is the cleanest way to show what the
        second source contributes.

    :param sources: N source positions.
    :param amplitudes: N amplitudes a_j, default all 1.
    :param wave_number: k, so the wavelength is 2 pi / k.
    :param frequency: omega, in radians per second.
    """

    def __init__(self, sources=((-1.2, 0, 0), (1.2, 0, 0)), amplitudes=None,
                 wave_number=10.0, frequency=10.0, method="uniform",
                 color="yellow", emission_by_density=True, radius=0.001,
                 name="RealInterference", **kwargs):
        self.sources = [Vector(s) for s in sources]
        self.amplitudes = list(amplitudes) if amplitudes is not None \
            else [1.0] * len(self.sources)
        if len(self.amplitudes) != len(self.sources):
            raise ValueError("one amplitude per source, got %d and %d"
                             % (len(self.amplitudes), len(self.sources)))
        self.k = wave_number
        self.omega = frequency
        super().__init__(name=name, method=method, color=color,
                         emission_by_density=emission_by_density,
                         radius=radius, **kwargs)

    # ------------------------------------------------------------------
    def density(self, tree, position, location=(0, 0)):
        n = len(self.sources)
        x, y = location

        # the distance to each source is worth an auxiliary of its own: it is
        # needed twice per wave, once in the phase and once in the envelope
        aux = {}
        for j in range(n):
            aux["r%d" % j] = "pos,s%d,sub,length" % j
        for j in range(n):
            aux["a%d" % j] = "amp%d,r%d,/,k,r%d,*,wt,-,sin,*" % (j, j, j)

        # the amplitudes add, and it is their *sum* that gets squared - the
        # cross term is the interference, and squaring them separately would
        # throw away the only thing this modifier exists to show
        total = ",".join("a%d" % j for j in range(n)) + ",+" * (n - 1)

        names = ["pos", "k"] + ["s%d" % j for j in range(n)] + ["wt"] \
                + ["amp%d" % j for j in range(n)]
        function = make_function(tree, location=location,
                                 functions={"density": "%s,2,**" % total},
                                 aux_functions=aux,
                                 inputs=names, outputs=["density"],
                                 vectors=["pos"] + ["s%d" % j for j in range(n)],
                                 scalars=["k", "wt"]
                                         + ["amp%d" % j for j in range(n)]
                                         + list(aux) + ["density"],
                                 name="Intensity", hide=False)
        tree.links.new(position, function.inputs["pos"])

        # the dials, built once and reused if density() is called again
        if not hasattr(self, "wave_number"):
            self.wave_number = InputValue(tree, location=(x - 2, y + 1),
                                          value=self.k, name="WaveNumber")
            self.amplitude_nodes = [
                InputValue(tree, location=(x - 2, y + 1 + n - j), value=amplitude,
                           name="Amplitude%d" % j)
                for j, amplitude in enumerate(self.amplitudes)]
            self.source_nodes = [
                InputVector(tree, location=(x - 2, y - 1 - j), vector=source,
                            name="Source%d" % j, hide=True)
                for j, source in enumerate(self.sources)]
            # and wt, the reason no scene has to keyframe this modifier at all
            self.clock = SceneTime(tree, location=(x, y + 8))
            self.frequency = InputValue(tree, location=(x, y + 7),
                                        value=self.omega, name="Frequency")
            self.phase = MathNode(tree, location=(x + 1, y + 8),
                                  operation="MULTIPLY",
                                  inputs0=self.clock.std_out,
                                  inputs1=self.frequency.std_out, hide=True)

        tree.links.new(self.wave_number.std_out, function.inputs["k"])
        tree.links.new(self.phase.std_out, function.inputs["wt"])
        for j in range(n):
            tree.links.new(self.source_nodes[j].std_out,
                           function.inputs["s%d" % j])
            tree.links.new(self.amplitude_nodes[j].std_out,
                           function.inputs["amp%d" % j])
        return function.outputs["density"]

    # ------------------------------------------------------------------
    def density_numpy(self, points, seconds=0.0):
        """The same f, in numpy, at one instant.

        ``seconds`` is the scene time the tree reads off the clock; the default
        0 is the frame the modifier is built on. Points sitting exactly on a
        source come out ``inf``, as they do in the node tree - that is the
        1/r, not a bug to be clamped away here.
        """
        points = np.asarray(points, dtype=float)
        amplitude = np.zeros(len(points))
        for source, amp in zip(self.sources, self.amplitudes):
            source = np.array(source, dtype=float)
            radius = np.linalg.norm(points - source, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                amplitude += amp / radius * np.sin(self.k * radius
                                                   - self.omega * seconds)
        return amplitude ** 2


class GaussianCloudModifier(SpatialDistributionModifier):
    """Points distributed as a gaussian blob, f = exp(-r^2 / 2 sigma^2).

    The distribution nothing about the video needs and every check does: its
    profile is known in closed form, so a histogram of the points the tree
    produces can be held against it (see the module docstring).

    :param sigma: width of the blob, about ``center``.
    """

    def __init__(self, sigma=1.0, name="GaussianCloud", **kwargs):
        self.sigma = sigma
        super().__init__(name=name, **kwargs)

    def density(self, tree, position, location=(0, 0)):
        function = make_function(tree, location=location,
                                 aux_functions={"r": "pos,c,sub,length"},
                                 functions={"density":
                                                "0,r,r,*,%s,/,-,exp" % (2 * self.sigma ** 2)},
                                 inputs=["pos", "c"], outputs=["density"],
                                 vectors=["pos", "c"],
                                 scalars=["r", "density"],
                                 name="Gaussian", hide=True)
        centre = InputVector(tree, location=(location[0] - 2, location[1] - 1),
                             vector=self.center, name="Centre", hide=True)
        tree.links.new(position, function.inputs["pos"])
        tree.links.new(centre.std_out, function.inputs["c"])
        return function.outputs["density"]

    def density_numpy(self, points):
        points = np.asarray(points, dtype=float)
        radius = np.linalg.norm(points - np.array(self.center, dtype=float), axis=1)
        return np.exp(-radius ** 2 / (2 * self.sigma ** 2))


class PolarGridModifier(GeometryNodesModifier):
    r"""A cartesian grid of lines bent into a polar one, on a dial.

    Two families of straight lines - ``horizontals`` of them running along x,
    ``verticals`` running along y, filling a panel centred on the object,
    whose width is chosen so that the cells come out square (see
    ``square_cells``) - and one ``Transition`` value that carries every point
    of them from where it is to where polar coordinates would put it:

    .. math::
        \varphi = 2\pi\,\frac{x + w/2}{w}, \qquad
        r = R\,\frac{y + h/2}{h}, \qquad
        \mathbf p(t) = (1-t)\,(x, y) + t\,(r\cos\varphi,\; r\sin\varphi).

    So the panel is read as the (φ, r) rectangle: its **width is one full
    turn** and its **height is the radius**, which is what makes each family
    turn into the thing it should. A horizontal line is y = const, hence
    r = const, hence a **circle**; a vertical line is x = const, hence
    φ = const, hence a **ray**. The bottom edge of the panel is r = 0, so the
    lowest horizontal line closes up into the point where all the rays meet,
    and the left and right edges are φ = 0 and φ = 2π, so the first and last
    vertical line land on the same ray.

    ``t`` is a plain lerp between the two pictures rather than anything
    cleverer, which is the honest thing to animate: at t = 0.5 the shape is
    half way between a straight line and its circle, which is what "the grid
    is being bent" looks like. Nothing about it is a coordinate change of the
    *scene* - only the drawing moves.

    **The colours change with the shape**, because a circle is not a
    horizontal line any more and should not be painted as one. The value of
    the dial is stored on every point as the ``transition`` attribute, and
    each family gets a two-stop
    :func:`~appearance.textures.gradient_from_attribute` ramp on it, so the
    horizontals travel from ``horizontal_color`` to ``circle_color`` and the
    verticals from ``vertical_color`` to ``ray_color`` while they bend. A
    panel left at t = 0 is a cartesian grid in the first pair of colours and
    stays one; that is how the same modifier serves as the "before" panel.

    The dial, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, "Transition")``:

    ``Transition``
        t, 0 = cartesian, 1 = polar. The one thing a scene animates.

    :param size: the panel, a scalar or ``(width, height)``. With
        ``square_cells`` the width is recomputed and only the height is read.
    :param horizontals: how many lines along x - circles, afterwards.
    :param verticals: how many along y - rays, afterwards.
    :param resolution: points per horizontal line. This is the one that has to
        be generous: a circle is only as round as the line it was bent from.
    :param ray_resolution: points per vertical line. A ray is straight and so
        is the lerp that makes it, so this one can be small.
    :param radius: R, how far the outermost circle reaches. Defaults to half
        the smaller side of the panel, which is the largest disc that still
        fits inside it.
    :param thickness: tube radius of the drawn lines.
    :param square_cells: widen (or narrow) the panel so that the cartesian
        grid comes out of equally spaced lines both ways - the default,
        because a grid of oblong cells reads as a stretched picture rather
        than as a coordinate system. It sets ``width = height *
        (verticals - 1)/(horizontals - 1)``, so the two line counts choose
        the panel's aspect ratio: nine circles and twelve rays want a panel
        half again as wide as it is tall. Pass ``False`` to use ``size`` as
        given.
    :param transition: where the dial starts.
    :param horizontal_color: palette name for the lines along x, at t = 0.
    :param vertical_color: palette name for the lines along y, at t = 0.
    :param circle_color: what the horizontals become at t = 1.
    :param ray_color: what the verticals become at t = 1.
    """

    def __init__(self, size=3.0, horizontals=11, verticals=13, resolution=193,
                 ray_resolution=33, radius=None, thickness=0.02,
                 square_cells=True, transition=0.0, horizontal_color="drawing",
                 vertical_color="custom1", circle_color="joker",
                 ray_color="example", name="PolarGrid", half=False,**kwargs):
        if horizontals < 2 or verticals < 2:
            raise ValueError("a grid needs at least two lines each way, "
                             "got %d and %d" % (horizontals, verticals))
        if isinstance(size, (int, float)):
            size = (size, size)
        self.width, self.height = float(size[0]), float(size[1])
        if square_cells:
            # the width follows from the height and the two line counts: the
            # cartesian grid is only *read* as a grid if its cells are square,
            # and the two spacings are h/(horizontals - 1) and
            # w/(verticals - 1), so this is the one width that makes them
            # equal. The height is what is kept because it is also the radius
            # the polar picture has to fit into.
            self.width = self.height * (verticals - 1) / (horizontals - 1)
        self.horizontals = horizontals
        self.verticals = verticals
        self.resolution = resolution
        self.ray_resolution = ray_resolution
        self.radius = min(self.width, self.height) / 2 if radius is None \
            else radius
        self.thickness = thickness
        self.transition = transition
        self.horizontal_color = horizontal_color
        self.vertical_color = vertical_color
        self.circle_color = circle_color
        self.ray_color = ray_color
        self.half = half
        self.kwargs = kwargs
        # kept because the materials are named after the tree: two panels of
        # the same modifier are two node groups, and their four ramps should
        # say which panel they belong to rather than pile up as .001
        self.label = name
        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        # the one dial, outside both frames because both families read it
        self.transition_node = InputValue(tree, location=(0, 2),
                                          value=self.transition,
                                          name="Transition")
        circles = self._family(tree, "horizontal", location=(2, 4))
        rays = self._family(tree, "vertical", location=(2, -4))
        joined = JoinGeometry(tree, location=(11, 0), name="JoinGrid")
        tree.links.new(circles, joined.geometry_in)
        tree.links.new(rays, joined.geometry_in)
        self.group_outputs.location=(17*200,0)

        if self.half:
            pos = Position(tree,location = (14,0),hide=True)
            selector = make_function(tree,name="Selector",
                        functions={
                            "select":"pos_x,-0.01,<"
                        },inputs=["pos"],outputs=["select"],
                        scalars=["select"],vectors=["pos"])
            tree.links.new(pos.std_out,selector.inputs["pos"])
            del_geo = DeleteGeometry(tree,name="DeleteHalf",selection=selector.outputs["select"])
            create_geometry_line(tree,[joined,del_geo],out=self.group_outputs.inputs[0])
        else:
            create_geometry_line(tree,[joined],out=self.group_outputs.inputs[0])


    # ------------------------------------------------------------------
    def _family(self, tree, kind, location=(0, 0)):
        """One family of parallel lines, bent, tubed and painted.

        Both families are the same six nodes: *one* line, resampled, then
        ``Duplicate Elements`` on the spline domain to make the rest of them
        and the ``Duplicate Index`` field to space the copies out. Drawing one
        line and copying it keeps the tree the same size whether the grid has
        nine lines or ninety, and it is the index - not a stack of primitives -
        that says where each one goes.
        """
        x, y = location
        w, h = self.width, self.height
        if kind == "horizontal":
            count, points = self.horizontals, self.resolution
            start = Vector((-w / 2, -h / 2, 0))
            end = Vector((w / 2, -h / 2, 0))
            step = (0.0, h / (count - 1))
            label, colors = "Horizontals", (self.horizontal_color,
                                            self.circle_color)
        else:
            count, points = self.verticals, self.ray_resolution
            start = Vector((-w / 2, -h / 2, 0))
            end = Vector((-w / 2, h / 2, 0))
            step = (w / (count - 1), 0.0)
            label, colors = "Verticals", (self.vertical_color, self.ray_color)

        frame = Frame(tree, location=location, label=label,
                      name=label + "Frame")
        line = CurveLine(tree, location=(0, 0), start=start, end=end,
                         name=label + "Line", parent=frame)
        dense = ResampleCurve(tree, location=(1, 0), curve=line.geometry_out,
                              count=points, name=label + "Resample",
                              parent=frame)
        copies = DuplicateElements(tree, location=(2, 0), domain="SPLINE",
                                   geometry=dense.geometry_out, amount=count,
                                   name=label + "Copies", parent=frame)
        # copy number i sits i steps along, which is the whole of the grid
        spacing = make_function(tree, location=(2, -1),
                                functions={"offset": ["i,%s,*" % repr(step[0]),
                                                      "i,%s,*" % repr(step[1]),
                                                      "0"]},
                                inputs=["i"], outputs=["offset"],
                                scalars=["i"], vectors=["offset"],
                                name=label + "Spacing", hide=True)
        spacing.parent = frame.node
        tree.links.new(copies.duplicate_index, spacing.inputs["i"])
        spread = SetPosition(tree, location=(3, 0),
                             geometry=copies.geometry_out,
                             offset=spacing.outputs["offset"],
                             name=label + "Spread", parent=frame)

        # the map itself, reading the position the lines have just been given
        position = Position(tree, location=(3, -1), hide=True, parent=frame)
        warp = make_function(
            tree, location=(4, -1),
            functions={"position": ["pos_x,1,t,-,*,r,phi,cos,*,t,*,+",
                                    "pos_y,1,t,-,*,r,phi,sin,*,t,*,+",
                                    "pos_z"]},
            aux_functions={"phi": "pos_x,%s,+,%s,/,%s,*" % (repr(w / 2),
                                                            repr(w), repr(tau)),
                           "r": "pos_y,%s,+,%s,/,%s,*" % (repr(h / 2), repr(h),
                                                          repr(self.radius))},
            inputs=["pos", "t"], outputs=["position"],
            vectors=["pos", "position"], scalars=["t", "phi", "r"],
            name=label + "ToPolar", hide=False)
        warp.parent = frame.node
        tree.links.new(position.std_out, warp.inputs["pos"])
        tree.links.new(self.transition_node.std_out, warp.inputs["t"])
        bent = SetPosition(tree, location=(5, 0), geometry=spread.geometry_out,
                           position=warp.outputs["position"],
                           name=label + "ToPolarPosition", parent=frame)

        # the dial, carried on the geometry so that the material can follow
        # the shape: a line that has become a circle is painted as a circle
        store = StoreNamedAttribute(tree, location=(6, 0), name="transition",
                                    data_type="FLOAT", domain="POINT",
                                    value=self.transition_node.std_out)
        store.node.parent = frame.node
        tree.links.new(bent.geometry_out, store.geometry_in)

        profile = CurveCircle(tree, location=(6, -1), radius=self.thickness,
                              resolution=8, name=label + "Profile",
                              parent=frame)
        tube = CurveToMesh(tree, location=(7, 0), curve=store.geometry_out,
                           profile_curve=profile.geometry_out, fill_caps=True,
                           name=label + "Tube", parent=frame)
        smooth = SetShadeSmooth(tree, location=(8, 0),
                                geometry=tube.geometry_out,
                                name=label + "Smooth", parent=frame)
        painted = SetMaterial(tree, location=(9, 0),
                              geometry=smooth.geometry_out,
                              material=self._material(label, *colors),
                              name="Paint" + label, parent=frame)
        self.materials.append(painted.material)
        return painted.geometry_out

    # ------------------------------------------------------------------
    def _material(self, label, before, after):
        """A two-stop ramp on the ``transition`` attribute, ``before`` to ``after``.

        A colour that has to *travel* cannot be a ``Set Material`` on its own -
        that socket takes a material, not a field - so the value of the dial
        goes onto the geometry as an attribute and the material reads it. Two
        stops and a linear ramp is then exactly a mix of the two palette
        colours by t.
        """
        from appearance.textures import gradient_from_attribute
        from utils.color_conversion import get_color_from_string

        gradient = {}
        for stop, color in ((0.0, before), (1.0, after)):
            rgba = get_color_from_string(color) if isinstance(color, str) \
                else color
            if rgba is None:
                raise ValueError("%r is not a palette colour; pass a name from "
                                 "utils.constants.COLOR_NAMES or an rgba"
                                 % color)
            gradient[stop] = list(rgba)
        return gradient_from_attribute(name=self.label + label,
                                       attr_name="transition",
                                       attr_type="GEOMETRY", function="fac",
                                       gradient=gradient, **self.kwargs)


class FarFieldModifier(GeometryNodesModifier):
    r"""Rays along the far-field maxima of a line of equally spaced sources.

    A row of N emitters a distance g apart, all radiating in phase, is a
    diffraction grating, and far enough away its field is a set of beams
    rather than a pattern. Beam n leaves the row at the angle where the path
    difference between neighbouring sources is a whole number of
    wavelengths, g sin(alpha) = n lambda, i.e.

    .. math::  \sin\alpha_n = n\,\lambda/g ,

    measured from the normal to the row. This modifier draws those
    directions as tubes from the centre of the array - one per order,
    outward, of a fixed length - so the beams the interference pattern
    itself produces can be seen against the prediction.

    Which orders exist is part of the statement and is *not* decided in
    python. :math:`|n\lambda/g| \le 1` has to hold for the arcsine to mean
    anything, so an order with no angle is simply not radiated, and the
    tree tests for that per frame::

        s = n lambda / g          direction sine
        e = (|s| < 1)             does this order exist
        c = sqrt(max(1 - s^2, 0)) direction cosine, clamped so the branch
                                  that is about to be switched off does not
                                  hand a NaN to the geometry
        direction = s*axis + c*normal
        length    = reach * e

    The clamp and the multiply are the branchless-select idiom the shader
    side of this video uses for the same reason: both branches are always
    evaluated, so each has to survive arguments meant for the other. An
    order that is not radiated comes out as a zero-length curve at the
    centre of the array, which ``Curve to Mesh`` turns into a degenerate
    ring of ``radius`` there - hidden inside the n = 0 tube, which starts at
    the same point and is exactly as thick.

    That makes the appearance of an order automatic, and it happens in the
    right place: the order emerges at :math:`\lambda = g/|n|` lying flat
    along the row (sin alpha = 1, the endfire direction) and swings inward
    from there as the wavelength drops. A scene sweeping ``Wavelength``
    therefore gets the beams sliding out of the horizon for free, at the
    same instant the pattern behind them flares.

    ``Wavelength`` and ``Spacing`` are ``Value`` nodes reachable by name
    through :func:`~interface.ibpy.get_geometry_node_from_modifier`, so a
    scene animates them with ``ibpy.change_default_value`` exactly as it
    animates the dials of the interference material - which is what keeps
    the two synchronised: one list of wavelengths, keyframed twice.

    ``Ratio`` scales the wavelength the angles are computed from, so a
    modifier can follow one colour of
    :func:`~appearance.textures.rgb_interference_texture`, which derives
    its three wavelengths from one ``Wavelength`` dial the same way. One
    modifier per colour, each with its colour's ratio, then takes the same
    keyframes as the material and draws that colour's beams.

    :param spacing: g, the distance between neighbouring sources, in the
        same units as ``wavelength``.
    :param wavelength: lambda at build time; the dial a scene ramps.
    :param ratio: the wavelength the rays are drawn for is
        ``ratio * Wavelength``. The ``Ratio`` dial; 1 by default.
    :param max_order: highest |n| built. Orders beyond
        ``spacing/(ratio*wavelength)`` never appear, so this only has to
        cover the shortest wavelength a scene reaches:
        ``int(g/(ratio*lambda_min))``.
    :param reach: how long the rays are, in blender units.
    :param radius: tube radius.
    :param resolution: vertices of the circular profile. Eight is plenty
        for a tube a few pixels across.
    :param axis: unit vector along the row, the direction alpha is measured
        towards.
    :param normal: unit vector normal to the row, alpha = 0. Rays only ever
        go into this half-plane, which is the half the sources radiate into
        in a scene that puts them at the edge of the frame.
    :param color: palette name for the tubes, or ``None`` to leave them
        unpainted. ``emission`` and the rest of ``customize_material``'s
        keywords come through ``kwargs``.
    :param zeroth_color: palette name for the n = 0 ray alone; ``None``
        paints it with ``color`` like the rest.
    :param draw_zeroth: whether this modifier draws the n = 0 ray at all.
        sin(alpha_0) = 0 at every wavelength, so several modifiers over one
        array - one per colour - would all draw the same ray on top of each
        other and z-fight. Leave it on for exactly one of them.
    """

    def __init__(self, spacing=1.0, wavelength=0.5, ratio=1.0, max_order=3,
                 reach=2.5, radius=0.008, resolution=8,
                 axis=(1, 0, 0), normal=(0, 0, 1),
                 color="text", zeroth_color=None, draw_zeroth=True,
                 name="FarField", **kwargs):
        self.spacing = spacing
        self.wavelength = wavelength
        self.ratio = ratio
        self.max_order = int(max_order)
        self.reach = reach
        self.radius = radius
        self.resolution = resolution
        self.axis = Vector(axis).normalized()
        self.normal = Vector(normal).normalized()
        self.color = color
        self.zeroth_color = color if zeroth_color is None else zeroth_color
        self.draw_zeroth = draw_zeroth
        self.kwargs = kwargs
        self.orders = list(range(-self.max_order, self.max_order + 1))
        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _tag(order):
        """``m2 m1 0 p1 p2`` - a socket name cannot carry a minus sign."""
        if order == 0:
            return "0"
        return ("p%d" if order > 0 else "m%d") % abs(order)

    def angles(self, wavelength=None):
        """``{n: alpha_n}`` in radians for the orders that exist - the mirror.

        The same statement as the tree makes, in numpy, so that what the
        modifier draws can be held against a number. Orders that are not
        radiated are absent rather than NaN.
        """
        lam = self.wavelength if wavelength is None else wavelength
        lam = lam * self.ratio
        angles = {}
        for order in self.orders:
            sine = order * lam / self.spacing
            if abs(sine) < 1:
                angles[order] = float(np.arcsin(sine))
        return angles

    def directions(self, wavelength=None):
        """The same thing as unit vectors, in the modifier's own frame."""
        return {order: Vector(np.sin(alpha) * np.array(self.axis)
                              + np.cos(alpha) * np.array(self.normal))
                for order, alpha in self.angles(wavelength).items()}

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        links = tree.links

        dials = Frame(tree, location=(0, 0), label="Dials", name="DialsFrame")
        # `Wavelength` and `Spacing` are what a scene reaches for by name;
        # the other three are here rather than baked into the formula so the
        # whole geometry can be moved without rebuilding the tree
        self.wavelength_node = InputValue(tree, location=(0, 1), name="Wavelength",
                                          value=self.wavelength, parent=dials)
        self.spacing_node = InputValue(tree, location=(0, 0), name="Spacing",
                                       value=self.spacing, parent=dials)
        self.reach_node = InputValue(tree, location=(0, -1), name="Reach",
                                     value=self.reach, parent=dials)
        # created after `Wavelength`, and not called "WavelengthRatio", since
        # the dials are found by substring and in creation order
        self.ratio_node = InputValue(tree, location=(0, 2), name="Ratio",
                                     value=self.ratio, parent=dials)
        axis = InputVector(tree, location=(0, -2), vector=self.axis,
                           name="ArrayAxis", parent=dials)
        normal = InputVector(tree, location=(0, -3), vector=self.normal,
                             name="ArrayNormal", parent=dials)

        # one function node for every order at once: the shared inputs are
        # then wired once instead of 2*max_order + 1 times
        aux = {}
        functions = {}
        for order in self.orders:
            tag = self._tag(order)
            aux["s" + tag] = "%s,lam,*,ratio,*,gap,/" % repr(float(order))
            aux["e" + tag] = "s%s,abs,1,<" % tag
            aux["c" + tag] = "1,s{0},s{0},*,-,0,max,sqrt".format(tag)
            functions["dir" + tag] = "up,c{0},scale,axis,s{0},scale,add".format(tag)
            functions["end" + tag] = "reach,e%s,*" % tag

        # `axis`, `up`, `lam`, `gap`, `reach` and none of them spelled like an
        # operator - a variable called `length` would be read as the vector
        # length and the group would build and compute something else
        function = make_function(tree, location=(4, 0),
                                 functions=functions, aux_functions=aux,
                                 inputs=["lam", "ratio", "gap", "reach", "axis", "up"],
                                 outputs=list(functions),
                                 vectors=["axis", "up"]
                                         + ["dir" + self._tag(n) for n in self.orders],
                                 scalars=["lam", "ratio", "gap", "reach"] + list(aux)
                                         + ["end" + self._tag(n) for n in self.orders],
                                 name="MaximaDirections", hide=True)
        links.new(self.wavelength_node.std_out, function.inputs["lam"])
        links.new(self.ratio_node.std_out, function.inputs["ratio"])
        links.new(self.spacing_node.std_out, function.inputs["gap"])
        links.new(self.reach_node.std_out, function.inputs["reach"])
        links.new(axis.std_out, function.inputs["axis"])
        links.new(normal.std_out, function.inputs["up"])

        rays = Frame(tree, location=(8, 0), label="Rays", name="RaysFrame")
        profile = CurveCircle(tree, location=(2, 1), radius=self.radius,
                              resolution=self.resolution, name="RayProfile",
                              parent=rays)

        def tube(curve, location, name):
            return CurveToMesh(tree, location=location, curve=curve,
                               profile_curve=profile.geometry_out,
                               fill_caps=True, name=name,
                               parent=rays).geometry_out

        def paint(geometry, color, location, name):
            if color is None:
                return geometry
            painted = SetMaterial(tree, location=location, geometry=geometry,
                                  material=color, name=name, parent=rays,
                                  **self.kwargs)
            self.materials.append(painted.material)
            return painted.geometry_out

        # the zeroth order is its own branch: it is the same ray for every
        # wavelength, so it gets its own colour, or is left to another host
        painted = []
        if self.draw_zeroth:
            zeroth = CurveLine(tree, location=(0, 0), mode="DIRECTION",
                               start=Vector(), direction=function.outputs["dir0"],
                               length=function.outputs["end0"],
                               name="Order0", parent=rays)
            painted.append(paint(tube(zeroth.geometry_out, (3, 0), "ZerothTube"),
                                 self.zeroth_color, (4, 0), "PaintZeroth"))

        lines = []
        for i, order in enumerate(o for o in self.orders if o != 0):
            tag = self._tag(order)
            line = CurveLine(tree, location=(0, -1 - i), mode="DIRECTION",
                             start=Vector(),
                             direction=function.outputs["dir" + tag],
                             length=function.outputs["end" + tag],
                             name="Order%s" % tag, parent=rays)
            lines.append(line.geometry_out)
        if lines:
            joined = JoinGeometry(tree, location=(2, -1), geometry=lines,
                                  name="JoinRays", parent=rays)
            painted.append(paint(tube(joined.geometry_out, (3, -1), "RayTubes"),
                                 self.color, (4, -1), "PaintRays"))

        geometry = JoinGeometry(tree, location=(5, 0), geometry=painted,
                                name="JoinPainted", parent=rays).geometry_out

        links.new(geometry, self.group_outputs.inputs["Geometry"])


class WaveVisualizationModifier(GeometryNodesModifier):
    r"""A grid standing up as the *exact* outgoing wave of a point source in 2D.

    Every other class in this module reads the field with probes - a cloud of
    points that lights up where the field is strong. This one does the other
    thing a field admits: it **is** the field. A flat, finely tessellated grid
    in the x-y plane, every vertex lifted to

    .. math::
        u(\mathbf r, t) = A' \sum_j \Big[ J_0(k r_j)\cos\omega t
                                        + Y_0(k r_j)\sin\omega t \Big]
                        = A' \sum_j \mathrm{Re}\big[H^{(1)}_0(k r_j)
                                                    e^{-i\omega t}\big],
        \qquad r_j = |\mathbf r - \mathbf c_j|,

    with :math:`k = 2\pi/\lambda` and :math:`\omega = 2\pi f`. That is the
    outgoing solution of the 2+1 dimensional wave equation, not the
    :math:`\sin(kr - \omega t)/\sqrt r` that stands in for it: the elementary
    form is only the asymptotics, and a surface is precisely the display on
    which the difference is visible. Near a source the true wavefronts are
    pulled inwards - the first crest lands at 0.38 lambda rather than 0.50 -
    and the amplitude rises like :math:`\ln r` instead of running away as
    :math:`r^{-1/2}`, which is what makes a source in shot a hill rather than
    a spike. Both cylinder functions come from
    :data:`~geometry_nodes.nodes.BESSEL_OPS`, so the tree says ``x,j0`` and
    ``x,y0`` and the Abramowitz-Stegun approximation behind them is shared by
    every source (see ``geometry_nodes/docs/BesselNode.tex``).

    The tree is three frames, one method each, and they are stages rather than
    decoration - each hands the next exactly one thing:

    :meth:`_control_frame`
        the dials and the clock. ``Wavelength``, ``Frequency``, ``Amplitude``
        and one ``Source<j>`` vector per emitter, plus ``Scene Time ->
        Seconds``, which is why this modifier moves **without a keyframe**.
    :meth:`_wave_frame`
        the arithmetic: one :func:`~geometry_nodes.nodes.make_function` group
        turning the ``Position`` field and those dials into the scalar u.
    :meth:`_geometry_frame`
        the grid, the store, the displacement and the paint.

    Two things in the last frame are ordering, not taste. The elongation is
    stored **before** ``Set Position`` and the displacement then reads it back
    through a ``Named Attribute``: a field is evaluated on the geometry the
    node receives, so an attribute stored *after* the lift would be computed
    from the lifted vertices, whose distance to a source is
    :math:`\sqrt{r^2+u^2}` rather than r - the surface would be right and the
    number wrong. Reading it back also means the seventy-odd math nodes of
    each Bessel group run once per vertex rather than twice.

    And the grid's ``UV Map`` output is stored as a ``FLOAT2`` on the
    **corner** domain under the name ``UVMap``, which is what makes it a uv
    layer rather than an attribute nothing reads. Without it the material's
    ``Texture Coordinate -> UV`` is all zeros and the surface renders in one
    flat colour, which looks like a broken shader rather than a missing
    attribute.

    **The colour comes from the material, and it is the same wave.**
    ``material="interference"`` paints the surface with
    :func:`~appearance.textures.interference_texture` in its ``"hankel"``
    model, which recomputes exactly this sum in the shader from the uv - so
    crest and trough take opposite hues and the nodal rings, where the alpha
    is :math:`u^2`, are cut clean out of the surface. The parameters are
    handed to it from the same python values that build the tree, and
    ``uv_scale`` is set to the grid's own size, so the two agree by
    construction rather than by being typed twice.

    The one thing that does **not** synchronise itself is the clock. A shader
    tree has no ``Scene Time`` node, so the material carries a plain ``Time``
    value while the geometry reads seconds off the scene. A scene that wants
    them locked ramps the material's ``Time`` linearly from 0 to the shot
    length in seconds::

        wave = WaveVisualizationModifier(name="Wave", wavelength=0.8)
        surface = Plane(name="WaveSurface")
        surface.add_mesh_modifier(type='NODES', node_modifier=wave)
        clock = ibpy.get_node_from_shader(wave.material, "Time")
        ibpy.change_default_value(clock, from_value=0, to_value=20,
                                  begin_time=0, transition_time=20)

    Left un-ramped the surface still moves and the colours stand still, which
    is a legitimate look (the pattern of a standing exposure over a moving
    membrane) but not the one this is for.

    The dials, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, label)``:

    ``Wavelength``
        lambda. Feeds k *and* the normalisation, so the far-field amplitude
        stays put as it is ramped.
    ``Frequency``
        f, in cycles per second of scene time. The tree computes
        :math:`2\pi f t`, so ramping it re-phases the wave already in flight -
        a chirp rather than a change of pitch.
    ``Amplitude``
        A, the height one source would reach at r = 1 in the far field. The
        ``pi/sqrt(lambda)`` that turns it into A' is
        :func:`~appearance.textures.interference_texture`'s convention,
        carried over unchanged so that the same number means the same height
        in the tree and the same colour in the shader.
    ``Source<j>``
        the emitters, as vectors in the grid's own plane.

    :param name: name of the node group, and of the modifier in the stack.
    :param size: edge length of the (square) grid, in blender units.
    :param resolution: vertices per side. The default 301 is 90601 vertices,
        which resolves a wavelength of 0.8 on a grid of side 8 with 30 samples
        - well past the point where the crests stop looking faceted. It is the
        one parameter worth turning *down* while composing a shot: the Bessel
        polynomials run per vertex.
    :param sources: emitter positions ``(x, y)`` in grid coordinates, the
        origin being the grid's centre. The sum of solutions is a solution, so
        several of them interfere.
    :param wavelength: lambda, in blender units.
    :param frequency: f, in cycles per second of scene time.
    :param amplitude: A, in blender units (see the ``Amplitude`` dial).
    :param source_radius: how far off the origin r is held. :math:`Y_0` has a
        logarithmic pole at a true point source, so something has to stop it;
        the physical reading is an emitter of this radius. Defaults to
        ``wavelength/20``, which is also
        :func:`~appearance.textures.interference_texture`'s default - pass the
        same number to both or the surface and its colour disagree in the one
        place the eye is drawn to. It is a python constant baked into the
        formula, not a socket, so it does **not** follow an animated
        ``Wavelength``: a scene that halves lambda doubles ``k * source_radius``
        and the pole grows, since the normalisation ``pi/sqrt(lambda)`` rises
        while the clamp stays put. Give it a larger fraction of the wavelength
        (``wavelength/10``) if a sweep is planned and the spike has to stay in
        frame.
    :param attribute: name of the float attribute the elongation is stored
        under, on the point domain. It is what displaces the surface, and it
        is left on the mesh for anything downstream to read.
    :param material: ``"interference"`` builds the matched texture (the
        default); a palette name or a ``bpy.types.Material`` is set as it
        stands; ``None`` leaves the surface unpainted.
    :param shade_smooth: smooth-shade the result. A displaced grid is a
        polyhedron, and at 300 vertices a side the facets are exactly the size
        of the fringes.
    :param front: gate the field with an expanding causal envelope, so that
        the surface is flat until ``impact`` and the disturbance then spreads
        outwards at the phase speed :math:`c=\lambda f` - a stone dropped in a
        pond rather than a source that has been ringing for ever. Off by
        default, which is the steady state every existing scene was composed
        against. See :func:`~geometry_nodes.nodes.wave_front_gate`, and note
        that it is **a look, not a solution**: the wave equation has no
        solution that is exactly the steady state behind a sharp edge.
    :param impact: the scene time, in seconds, at which the source starts.
        The phase is measured from it as well, so the wave leaves the source
        at zero elongation instead of picking up mid-cycle. Reachable as the
        ``Impact`` dial, in the tree and in the material both.
    :param front_width: how far the envelope takes to go from nothing to the
        full steady state, in blender units. Defaults to one wavelength, which
        is about the narrowest that does not read as a crease in the surface.
        The ``FrontWidth`` dial.
    :param time_shift: seconds added to the clock before it is turned into a
        phase, the ``TimeShift`` dial. Only the phase reads it - fronts and
        impacts stay on scene time. It is what lets a sub-scene pick up the
        wave of the one before it: rendered from frame 1, a scene that should
        continue a predecessor of length T passes ``T - 1/FRAME_RATE``. The
        ``"interference"`` material does not know about it; ramp its ``Time``
        from the shifted value if the two have to agree.
    :param source_impacts: one entry per source, a scene time in seconds or
        ``None``. A source with an impact is silent before it and then spreads
        out behind the same causal envelope ``front`` uses (``Impact<j>`` and
        ``FrontWidth`` dials), but - unlike ``front`` - its **phase stays on
        the common clock**. That is a slit opened in a wall a plane wave has
        been hitting all along: it starts in step with the slits that were
        already open. Sources given ``None`` ring for ever. Not combinable
        with ``front``, and geometry-only like ``time_shift``.
    :param intensity: build the ``Intensity`` dial (circular waves only),
        which turns the surface into the time-averaged intensity distribution.
        With :math:`u = C\cos\omega t + S\sin\omega t`, where
        :math:`C = A'\sum_j J_0(kr_j)` and :math:`S = A'\sum_j Y_0(kr_j)`
        (each gated like its wave), the intensity is :math:`C^2 + S^2` - the
        square of the local amplitude, so it does not move. At 1 the surface
        is flat and the ``result`` attribute holds
        ``IntensityGain * (C^2 + S^2)``; in between both fade linearly. The
        colour comes from ``result``, so a ``"function"`` material paints
        the intensity from the background colour (0) up to its crest colour.
        Geometry-only: the ``"interference"`` material knows nothing of it.
    :param intensity_gain: the ``IntensityGain`` dial, the factor that
        brings :math:`C^2 + S^2` into the range of the material's ramp.

    .. note::
        The ``Frequency`` dial used to be built under the name ``Period``
        while holding a frequency, and the wave frame divided by it - which is
        the identity at ``f = 1`` (what every existing sub-scene passes) and
        wrong everywhere else, disagreeing with both the shader and
        :meth:`elongation_numpy`. It now computes ``time,frequency,*,tau,*``,
        the association the docstring above always claimed and the one
        :func:`~appearance.textures.interference_texture` uses. Nothing reads
        the old name, and at ``f = 1`` nothing moves.
    """

    def __init__(self, name="WaveVisualization", size=8.0, resolution=301,
                 sources=None, wavelength=0.8, frequency=1.0,
                 amplitude=0.25, source_radius=None, attribute="result",
                 material="interference", shade_smooth=True,
                 front=False, impact=0.0, front_width=None, time_shift=0.0,
                 source_impacts=None, intensity=False, intensity_gain=1.0,
                 **kwargs):

        # initialize fields
        self.name = name
        self.size = size
        self.resolution = resolution
        if sources is None:
            self.mode = "PLANAR"
        else:
            self.mode = "CIRCULAR"
            self.sources = [Vector((s[0], s[1], 0)) for s in sources]

        self.wavelength = wavelength
        self.frequency = frequency
        self.period = 1 / frequency
        self.amplitude = amplitude
        # the same default as interference_texture, so that the two clamps
        # coincide when neither is given explicitly
        self.source_radius = wavelength / 20 if source_radius is None \
            else source_radius

        # the transient: a front expanding at the phase speed, ahead of which
        # the surface has not heard about the source yet
        self.front = front
        self.impact = impact
        self.front_width = wavelength if front_width is None else front_width
        self.time_shift = time_shift

        # per-source fronts: {j: impact} for the sources that start late
        if source_impacts is None:
            self.source_impacts = {}
        else:
            if front:
                raise ValueError("source_impacts and front are exclusive")
            if self.mode != "CIRCULAR" or len(source_impacts) != len(self.sources):
                raise ValueError("source_impacts needs one entry per source")
            self.source_impacts = {j: impact for j, impact in enumerate(source_impacts)
                                   if impact is not None}

        if intensity and self.mode != "CIRCULAR":
            raise ValueError("intensity needs sources")
        self.intensity = intensity
        self.intensity_gain = intensity_gain

        self.attribute = attribute
        self.paint = material
        self.shade_smooth = shade_smooth
        self.kwargs = kwargs

        # filled in by _geometry_frame; the scene needs it to reach the
        # shader's Time value
        self.material = None

        super().__init__(name=name, automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        # three simple frames that structure the geometry node setup
        control = self._control_frame(tree)
        elongation = self._wave_frame(tree, control)
        geometry = self._geometry_frame(tree, control, elongation)
        self.group_outputs.location = (9 * 200, 0)
        tree.links.new(geometry, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        """The dials and the clock: everything a scene animates, in one place.

        Nothing is computed here - not even ``omega = 2 pi f``. The wave frame
        takes ``time`` and ``frequency`` as separate inputs and forms
        ``time,frequency,*,tau,*`` itself, which is the association
        :func:`~appearance.textures.interference_texture` uses in the shader.
        Matching it matters at the end of a long shot: at t = 60 s the phase
        is some 400 radians, where a float32 ulp is 3e-5, and the two ways of
        bracketing the product differ in the last bits. Same order, same
        wavefronts, and the painted fringes sit on the geometric ones.

        :return: dict of the sockets the wave frame consumes.
        """
        frame = Frame(tree, location=(0, 0), label="Control",
                      name="ControlFrame")
        size = InputValue(tree,location=(0,2),value=self.size,label="Size",hide=True,parent=frame)
        clock = SceneTime(tree, location=(0, 1), std_out="Seconds",
                          name="Clock", parent=frame)
        wavelength = InputValue(tree, location=(0, 0), value=self.wavelength,
                                name="Wavelength", parent=frame)
        # named for what it holds. It used to be called "Period" while
        # carrying a frequency, and the wave frame divided by it - which is
        # the identity at f = 1 and wrong everywhere else. See _wave_frame.
        frequency = InputValue(tree, location=(0, -1), value=self.frequency,
                               name="Frequency", parent=frame)
        amplitude = InputValue(tree, location=(0, -2), value=self.amplitude,
                               name="Amplitude", parent=frame)
        width = InputValue(tree,location=(0,-3),value=self.size,name="Width",parent=frame)
        time_shift = InputValue(tree, location=(-2, 2), value=self.time_shift,
                                name="TimeShift", parent=frame)

        if self.front:
            ease_in = InputValue(tree,location=(-2,1),value=0,name="EaseIn",parent=frame)
            impact = InputValue(tree, location=(-2, 0), value=self.impact,
                                name="Impact", parent=frame)
            edge = InputValue(tree, location=(-2, -1), value=self.front_width,
                              name="FrontWidth", parent=frame)
            transient = {"impact": impact.std_out, "edge": edge.std_out,"ease_in":ease_in.std_out}
        elif self.source_impacts:
            edge = InputValue(tree, location=(-2, -1), value=self.front_width,
                              name="FrontWidth", parent=frame)
            transient = {"edge": edge.std_out}
            for j, value in self.source_impacts.items():
                impact = InputValue(tree, location=(-3, -j), value=value,
                                    name="Impact%d" % j, parent=frame)
                transient["impact%d" % j] = impact.std_out
        else:
            transient = {}

        if self.intensity:
            mix = InputValue(tree, location=(-2, 3), value=0,
                             name="Intensity", parent=frame)
            gain = InputValue(tree, location=(-2, 4), value=self.intensity_gain,
                              name="IntensityGain", parent=frame)
            transient["intensity"] = mix.std_out
            transient["gain"] = gain.std_out

        if self.mode=="CIRCULAR":
            sources = [InputVector(tree, location=(0, -3 - j), vector=source,
                                   name="Source%d" % j, hide=True, parent=frame)
                       for j, source in enumerate(self.sources)]
        else:
            sources = []
        return dict({"size":size.std_out,
                     "time": clock.std_out,
                     "time_shift": time_shift.std_out,
                     "wavelength": wavelength.std_out,
                     "frequency": frequency.std_out,
                     "amplitude": amplitude.std_out,
                     "width":width.std_out,
                     "sources": [s.std_out for s in sources]}, **transient)

    # ------------------------------------------------------------------
    def _wave_frame(self, tree, control):
        r"""
        depending on the ``self.mode`` variable

        a plane wave
        u(r,x) = A sin(k x-w t)
        or a circular wave
        u(r, t)

        is implemented in the geometry node

        The auxiliaries are the arithmetic worth naming, and they are shared
        rather than repeated: ``k`` and ``amp`` are computed once for all
        sources, and each source's ``r`` is used twice - once clamped into the
        Bessel argument, and once (through it) by both cylinder functions.

        ``amp`` is ``A pi / sqrt(lambda)`` restores the asymptotic property:
        :math:`J_0(x) \sim \sqrt{2/\pi x}\cos(x - \pi/4)`, so a far-field
        amplitude of :math:`A/\sqrt r` needs a factor
        :math:`\sqrt{\pi k/2} = \pi/\sqrt\lambda`.

        It is built from the
        ``Wavelength`` socket rather than baked in as a number, so a scene
        that sweeps the wavelength keeps the same wave height as it goes.

        :return: the socket carrying the scalar elongation.
        """
        frame = Frame(tree, location=(1, 0), label="Wave Computation",
                      name="WaveFrame")
        position = Position(tree, location=(0, -1), name="GridPosition",
                            hide=True, parent=frame)

        # the clock the wave is a function of. Without a front it is the scene
        # clock; with one it is measured from the impact, so that the phase at
        # the source is zero when the creature lands rather than wherever the
        # scene happens to have got to
        clock = "time,impact,-" if self.front else "time"
        # the dials the transient adds, by the names the formula reads them
        if self.front:
            transient = ["impact", "edge", "ease_in"]
        elif self.source_impacts:
            transient = ["edge"] + ["impact%d" % j for j in self.source_impacts]
        else:
            transient = []
        if self.intensity:
            transient += ["intensity", "gain"]

        if self.mode == "PLANAR":
            n = 0  # no sources
            aux = {
                "k": "%s,wavelength,/" % tau,
                "wt": "%s,time_shift,+,frequency,*,%s,*" % (clock, tau),
                "u": "amplitude,k,pos_x,*,wt,-,sin,*"
            }
            if self.front:
                # A plane wave has no source point to measure from - it comes
                # from x = -infinity - so a transient one needs an emitter
                # named explicitly, and the grid's own leading edge is the
                # only one the modifier knows about. The wave is then
                # generated at x = -size/2 and marches in from the left, which
                # is what "the wave arrives" looks like on a finite grid.
                # Gating on pos_x itself would instead say the wave had
                # already reached everything to the left of the front, which
                # is the steady state again.
                aux.pop("u")
                aux.update(wave_front_gate("pos_x,%s,+" % repr(0.5 * self.size),
                                           "0", clock))
                aux["u"] = "amplitude,k,pos_x,*,wt,-,sin,*,env0,*"
        else:
            n = len(self.sources)
            aux = {
                "k": "%s,wavelength,/" % tau,
                "wt": "%s,time_shift,+,frequency,*,%s,*" % (clock, tau),
                "amp": "amplitude,pi,*,wavelength,sqrt,/"
            }
            for j in range(n):
                # the grid is still flat where this is evaluated, so the distance
                # in the plane is the distance in space
                aux["r%d" % j] = "pos,c%d,sub,length" % j
            for j in range(n):
                # x = k r, held off the pole of Y0
                aux["x%d" % j] = "r%d,%s,max,k,*" % (j, repr(self.source_radius))
            for j in range(n):
                # the gate goes in before the wave that multiplies by it:
                # aux entries are emitted in insertion order, so an env
                # defined afterwards is not yet a name when w reads it
                gated = self.front or j in self.source_impacts
                if self.front:
                    aux.update(wave_front_gate("r%d" % j, str(j), clock))
                elif gated:
                    # a late source: its own front, but the common phase
                    aux.update(wave_front_gate("r%d" % j, str(j),
                                               "time,impact%d,-" % j))
                if self.intensity:
                    # the two quadratures apart, so that the intensity can
                    # be formed from them without a second Bessel evaluation
                    gate = ",env%d,*" % j if gated else ""
                    aux["cj%d" % j] = "amp,x%d,j0,*%s" % (j, gate)
                    aux["sj%d" % j] = "amp,x%d,y0,*%s" % (j, gate)
                    aux["w%d" % j] = "cj{0},wt,cos,*,sj{0},wt,sin,*,+".format(j)
                else:
                    # Re[H0(kr) exp(-i w t)], the outgoing wave
                    aux["w%d" % j] = ("amp,x{0},j0,wt,cos,*,"
                                      "x{0},y0,wt,sin,*,+,*".format(j)
                                      + (",env%d,*" % j if gated else ""))
            aux["u"] = ",".join("w%d" % j for j in range(n)) + ",+" * (n - 1)
            if self.intensity:
                aux["C"] = ",".join("cj%d" % j for j in range(n)) + ",+" * (n - 1)
                aux["S"] = ",".join("sj%d" % j for j in range(n)) + ",+" * (n - 1)
                # the wave fades out of the surface as the intensity fades in
                aux["uf"] = "u,1,intensity,-,*"

        names = ["pos", "time", "time_shift", "frequency", "wavelength",
                 "amplitude"] + transient + ["c%d" % j for j in range(n)]

        if self.front:
            function = "u,ease_in,*"
        elif self.intensity:
            function = "uf"
        else:
            function = "u"
        functions = {"elongation": function}
        if self.intensity:
            functions["value"] = "uf,C,C,*,S,S,*,+,gain,*,intensity,*,+"
        wave = make_function(tree, location=(1, 0), name="Elongation",
                             functions=functions,
                             aux_functions=aux,
                             inputs=names, outputs=list(functions),
                             vectors=["pos"] + ["c%d" % j for j in range(n)],
                             scalars=["time", "time_shift", "frequency",
                                      "wavelength", "amplitude"]
                                     + list(functions) + transient + list(aux),
                             custom_ops=BESSEL_OPS, parent=frame, hide=False)

        tree.links.new(position.std_out, wave.inputs["pos"])
        for key in ["time", "time_shift", "frequency", "wavelength",
                    "amplitude"] + transient:
            tree.links.new(control[key], wave.inputs[key])
        for j, source in enumerate(control["sources"]):
            tree.links.new(source, wave.inputs["c%d" % j])
        if self.intensity:
            return wave.outputs["elongation"], wave.outputs["value"]
        return wave.outputs["elongation"]

    # ------------------------------------------------------------------
    def _geometry_frame(self, tree, control, elongation):
        """Grid -> uv -> store -> lift -> paint.

        The order is the argument. See the class docstring for why the store
        comes before the lift and why the uv map has to be written out by
        hand.

        :return: the geometry socket for the group output.
        """
        frame = Frame(tree, location=(1, 2), label="Geometry",
                      name="GeometryFrame")
        grid = Grid(tree, location=(0, 0), size_x=control["size"], size_y=control["width"],
                    vertices_x=self.resolution, vertices_y=self.resolution,
                    name="Grid", parent=frame)

        # the grid's uv is an anonymous field; named and put on the corner
        # domain it becomes the uv layer the material samples
        uv = StoreNamedAttribute(tree, location=(1, 0), data_type="FLOAT2",
                                 domain="CORNER", name="UVMap",
                                 value=grid.node.outputs["UV Map"],
                                 parent=frame)

        # with an intensity the colour and the height part ways: the colour
        # reads the attribute, the lift its own copy of the elongation
        if self.intensity:
            elongation, value = elongation
            lift_attribute = self.attribute + "Height"
            painted_value = StoreNamedAttribute(tree, location=(2, 1), data_type="FLOAT",
                                                domain="POINT", name=self.attribute,
                                                value=value, parent=frame)
        else:
            lift_attribute = self.attribute
            painted_value = None
        stored = StoreNamedAttribute(tree, location=(2, 0), data_type="FLOAT",
                                     domain="POINT", name=lift_attribute,
                                     value=elongation, parent=frame)

        amp_store = StoreNamedAttribute(tree, location=(3, 0), data_type="FLOAT",
                                        domain="POINT", name="amplitude",
                                        value=control["amplitude"], parent=frame)

        # read back rather than reusing the socket: this is what makes the
        # elongation the thing that moves the surface, and it costs one node
        # against a second evaluation of every Bessel group
        height = NamedAttribute(tree, location=(2, -2), data_type="FLOAT",
                                name=lift_attribute, parent=frame, hide=True)
        offset = CombineXYZ(tree, location=(3, -2), z=height.std_out,
                            name="Lift", parent=frame, hide=True)
        lifted = SetPosition(tree, location=(4, 0),
                             offset=offset.std_out, name="Displace",
                             parent=frame)
        geometry = lifted.geometry_out

        customs = []
        if self.shade_smooth:
            smooth = SetShadeSmooth(tree, location=(5, 0),
                                    name="Smooth", parent=frame)
            geometry = smooth.geometry_out
            customs.append(smooth)

        if self.paint is not None:
            painted = SetMaterial(tree, location=(5, 0),
                                  material=self._texture(), name="Paint",
                                  parent=frame)
            self.material = painted.material
            self.materials.append(painted.material)
            customs.append(painted)
            geometry = painted.geometry_out

        stores = [stored] if painted_value is None else [painted_value, stored]
        create_geometry_line(tree, [grid, uv] + stores + [amp_store, lifted] + customs)
        return geometry

    # ------------------------------------------------------------------
    def _texture(self):
        """The material, with the tree's own parameters written into it.

        ``interference_texture`` measures in uv, so the sources move into the
        unit square and ``uv_scale`` carries the grid's size - after which its
        ``wavelength`` and ``source_radius`` are lengths in blender units,
        exactly as they are here. Anything else (a palette name, a finished
        material) is handed to ``Set Material`` untouched.
        """
        if not isinstance(self.paint, str) or self.paint != "interference":
            return self.paint
        from appearance.textures import interference_texture
        return interference_texture(
            name=self.name + "Texture",
            model="hankel",
            sources=[(0.5 + source.x / self.size, 0.5 + source.y / self.size)
                     for source in self.sources],
            uv_scale=(self.size, self.size),
            wavelength=self.wavelength, frequency=self.frequency,
            amplitude=self.amplitude, source_radius=self.source_radius,
            # the same three numbers the tree was built with. The shader
            # recomputes the sum independently, so an ungated material would
            # paint fringes on water that is provably flat
            front=self.front, impact=self.impact,
            front_width=self.front_width,
            **self.kwargs)

    # ------------------------------------------------------------------
    def elongation_numpy(self, points, seconds=0.0):
        """The same u, in numpy, at one instant - the tree's mirror.

        The convention of this module: every field that goes into a tree also
        goes into numpy, so the modifier can be checked against something
        other than itself. ``scipy.special.j0``/``y0`` are the exact functions
        here, so a comparison also measures what the polynomial approximation
        costs (5e-8, i.e. nothing).

        :param points: ``(n, 2)`` or ``(n, 3)`` array of positions; only x
            and y are read.
        :param seconds: the scene time the tree reads off the clock.
        """
        from scipy.special import j0, y0
        points = np.asarray(points, dtype=float)[:, :2]
        k = tau / self.wavelength
        elapsed = seconds - self.impact if self.front else seconds
        wt = (elapsed + self.time_shift) * self.frequency * tau
        amp = self.amplitude * pi / np.sqrt(self.wavelength)
        reach = self.wavelength * self.frequency * elapsed
        total = np.zeros(len(points))
        for j, source in enumerate(self.sources):
            radius = np.linalg.norm(points - np.array([source.x, source.y]),
                                    axis=1)
            x = np.maximum(radius, self.source_radius) * k
            wave = amp * (j0(x) * np.cos(wt) + y0(x) * np.sin(wt))
            if j in self.source_impacts:
                reach = self.wavelength * self.frequency \
                        * (seconds - self.source_impacts[j])
            if self.front or j in self.source_impacts:
                gate = np.clip((reach - radius) / self.front_width, 0.0, 1.0)
                wave = wave * gate * gate * (3 - 2 * gate)
            total += wave
        return total

    def ease_in(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        ease_in_node = get_geometry_node_from_modifier(self,"EaseIn")
        ibpy.change_default_value(ease_in_node,from_value=0,to_value=1,begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time


class Slicer(GeometryNodesModifier):
    r"""Everything on the far side of a plane, thrown away.

    The one modifier in this module that *takes* geometry instead of building
    it: the incoming mesh goes through a single ``Delete Geometry`` whose
    selection is the half space

    .. math::
        \vec{x}\cdot\hat{n} > s ,

    with :math:`\hat{n}` the ``Direction`` dial (z by default) and :math:`s`
    the ``SlicerValue`` one. So the default cuts the lid off at a height, and
    ramping ``SlicerValue`` down runs the cut through the object - which is
    what opens a whistle up on camera and shows the pipe inside it::

        slicer = Slicer(value=0.25)
        whistle.add_mesh_modifier(type='NODES', node_modifier=slicer)
        slicer.slice(to_value=0.1, begin_time=1, transition_time=2)

    **This is a cut, not a section.** Deleting points removes every face that
    used one, so what is left is the mesh's own polygons up to the plane and
    an open, slightly ragged rim - the inside of a hollow object becomes
    visible through it, and there is no flat cap over the cut. A cap needs a
    boolean against a half-space solid, which costs a manifold mesh and a
    great deal more time per frame; for a shell (a pipe, a bell, a whistle)
    the open cut is also the *honest* picture, since a capped one would show
    a lid that is not there.

    **Which space the plane is in.** ``Position`` is the object's own local
    coordinates, so the plane is carried along by the object's transform: a
    whistle that is rotated keeps its cut, and ``SlicerValue`` stays the
    number it was. Slicing along a *world* axis of an object that is turned
    means turning ``Direction`` by the same amount.

    The dials, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, label)`` (:meth:`dial`):

    ``SlicerValue``
        where the plane sits, measured along ``Direction``. Above the
        object's extent nothing is removed, below it nothing is left.
    ``Direction``
        the normal of the cutting plane, pointing at the half that goes.
        Need not be normalised - if it is not, ``SlicerValue`` is measured in
        units of its length, which is why :meth:`bounds` reports what the
        object's extent actually is in that measure.

    :param name: name of the node group, and of the modifier in the stack.
    :param direction: ``"x"``, ``"y"``, ``"z"`` (the default), their negatives
        (``"-z"``), or any vector. The half space it points into is the one
        that is removed.
    :param value: what ``SlicerValue`` starts at.
    :param domain: what is selected and deleted. ``"POINT"`` (the default)
        deletes a vertex and every face that used it, so nothing sticks out
        past the plane. ``"FACE"`` evaluates the test at face centres instead,
        which leaves the faces that straddle the plane in place and cuts about
        half a polygon higher, but keeps the rim tidier on a coarse mesh.
    :param invert: keep the half that would be cut and remove the other one.
    :param kwargs: passed on to
        :class:`~geometry_nodes.geometry_nodes_modifier.GeometryNodesModifier`.
    """

    _AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1),
             "-x": (-1, 0, 0), "-y": (0, -1, 0), "-z": (0, 0, -1)}

    def __init__(self, name="Slicer", direction="z", value=0.0,
                 domain="POINT", invert=False, **kwargs):
        self.direction = self.axis(direction)
        self.value = value
        self.domain = domain
        self.invert = invert
        # filled in by create_node, so that slice() has the dial to keyframe
        # without going back through the node tree by name
        self.slicer_value = None
        super().__init__(name=name, automatic_layout=True,
                         group_input=True, group_output=True, **kwargs)

    # ------------------------------------------------------------------
    @classmethod
    def axis(cls, direction):
        """The plane normal as a ``Vector``, from a name or a vector."""
        if isinstance(direction, str):
            key = direction.strip().lower()
            if key not in cls._AXES:
                raise ValueError("slicing direction %r is not one of %s, and "
                                 "not a vector either"
                                 % (direction, sorted(cls._AXES)))
            return Vector(cls._AXES[key])
        return Vector(direction)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        geometry = self.group_inputs.outputs["Geometry"]

        direction = InputVector(tree, location=(-4, -1), vector=self.direction,
                                name="Direction")
        self.slicer_value = InputValue(tree, location=(-4, -2),
                                       value=self.value, name="SlicerValue")
        position = Position(tree, location=(-4, 0), name="SlicePosition",
                            hide=True)

        # the half space, as one dot product against one number. The float the
        # comparison leaves is wired straight into Selection, which blender
        # reads as a boolean - the house style of the other culls in this
        # module (SpatialDistributionModifier.constraint, PolarGridModifier's
        # "DeleteHalf").
        cut = "pos,dir,dot,value,%s" % ("<" if self.invert else ">")
        selection = make_function(tree, location=(-2, -1), name="SliceSelection",
                                  functions={"cut": cut},
                                  inputs=["pos", "dir", "value"],
                                  outputs=["cut"],
                                  vectors=["pos", "dir"],
                                  scalars=["value", "cut"])
        tree.links.new(position.std_out, selection.inputs["pos"])
        tree.links.new(direction.std_out, selection.inputs["dir"])
        tree.links.new(self.slicer_value.std_out, selection.inputs["value"])

        slice_off = DeleteGeometry(tree, location=(0, 0), domain=self.domain,
                                   mode="ALL", geometry=geometry,
                                   selection=selection.outputs["cut"],
                                   name="SliceCut")
        tree.links.new(slice_off.geometry_out,
                       self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def dial(self, label="SlicerValue"):
        """The node a scene animates, by the name the tree gave it."""
        return get_geometry_node_from_modifier(self, label)

    def slice(self, to_value, from_value=None, begin_time=0,
              transition_time=DEFAULT_ANIMATION_TIME):
        """Run the cutting plane to ``to_value`` and return when it arrives.

        ``from_value=None`` picks the plane up wherever the last call left it
        (``value`` at first), which is what makes a sequence of calls read as
        one continuous cut. It is *not* passed on as ``None``:
        :func:`ibpy.change_default_value` writes no keyframe at all for a
        start it is not given, and the dial would then hold ``to_value`` from
        the first frame of the scene rather than move to it.

        :return: ``begin_time + transition_time``, the project's convention
            for chaining a scene's ``t0``.
        """
        if from_value is None:
            from_value = self.value
        self.value = to_value
        return ibpy.change_default_value(self.slicer_value.std_out,
                                         from_value=from_value,
                                         to_value=to_value,
                                         begin_time=begin_time,
                                         transition_time=transition_time)

    # ------------------------------------------------------------------
    def bounds(self, bob):
        """How far ``bob`` reaches along ``Direction``, as ``(min, max)``.

        The two values ``SlicerValue`` has to run between for a cut to go all
        the way through: at the maximum nothing is removed yet, at the minimum
        nothing is left. Measured on the object's own mesh in its own local
        coordinates, which is the space the plane lives in.
        """
        obj = ibpy.get_obj(bob)
        corners = np.array([list(corner) for corner in obj.bound_box])
        reach = corners @ np.array(self.direction)
        return float(reach.min()), float(reach.max())

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

    def __init__(self, radius=1.0, rings=3, edge="peak", aperture=0.289,
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


class BesselVisualizer(GeometryNodesModifier):
    r"""J0 and Y0 as two lines of vertices, lifted off the x axis.

    The circular wave is u(r, t) = J0(kr) cos wt + Y0(kr) sin wt, and this is
    its two ingredients on their own. Each is a ``Mesh Line`` of vertices
    running along the x axis - a grid line - whose vertices are displaced in z
    by ``amount * J0(x)`` or ``amount * Y0(x)``, evaluated by the
    :class:`~geometry_nodes.nodes.BesselNode` groups ``BesselJ0Geo`` and
    ``BesselY0Geo``. With the amount at 0 a line lies flat on the axis, and
    ramping it to 1 bends it into the function.

    The mapping is the one of ``CoordinateSystem2``: the value (0, 0) sits at
    the object's origin, and x and y are scaled by ``width / (x1 - x0)`` and
    ``height / (y1 - y0)``. Hang the host at the coordinate system's location
    with the same domains and lengths, and the curves lie on its axes.

    Vertices lifted outside ``y_domain`` are deleted rather than clamped, which
    cuts the logarithmic dive of Y0 at the bottom edge of the plot instead of
    running it along the edge. Y0's group returns a finite number at x = 0
    (blender's logarithm of 0 is 0), so the Y0 line also drops x <= 0.

    :param x_domain: [x0, x1], the arguments the lines run over.
    :param y_domain: [y0, y1], the values that are drawn; anything outside is
        cut away.
    :param width: world length of x0..x1, the coordinate system's x length.
    :param height: world length of y0..y1, the coordinate system's y length.
    :param count: vertices per line.
    :param thickness: radius of the tube drawn along each line.
    :param j0_amount, y0_amount: start values of the displacement dials.
    :param j0_reveal, y0_reveal: start values of the reveal dials, the
        fraction of x0..x1 that is shown, from the left.
    :param j0_color, y0_color: palette names, turned into materials by
        :func:`~appearance.textures.get_texture`.

    The dials a scene animates are ``J0Reveal``, ``J0Amount``, ``Y0Reveal``,
    ``Y0Amount`` (and ``Thickness``)::

        bessel = BesselVisualizer(x_domain=[0, 20], y_domain=[-1, 1],
                                  width=14, height=6)
        host = Plane(name="BesselLines", location=coords_location,
                     apply_location=False)
        host.add_mesh_modifier(type='NODES', node_modifier=bessel)
        ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(bessel, "J0Amount"),
                                  from_value=0, to_value=1,
                                  begin_time=0, transition_time=3)
    """

    def __init__(self, x_domain=(0, 20), y_domain=(-1, 1), width=14, height=6,
                 count=1000, thickness=0.03, j0_amount=0, y0_amount=0,
                 j0_reveal=0, y0_reveal=0, j0_color="example",
                 y0_color="important", name="BesselVisualizer", **kwargs):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.width = width
        self.height = height
        self.count = count
        self.thickness = thickness
        self.amounts = {"j0": j0_amount, "y0": y0_amount}
        self.reveals = {"j0": j0_reveal, "y0": y0_reveal}
        self.colors = {"j0": j0_color, "y0": y0_color}
        self.kwargs = kwargs
        super().__init__(name=name, automatic_layout=False)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        self._control_frame(tree)
        j0 = self._line_frame(tree, "j0", row=0)
        y0 = self._line_frame(tree, "y0", row=-8)
        join = JoinGeometry(tree, location=(14, -3), geometry=[j0, y0],
                            name="BothLines")
        self.group_outputs.location = (15 * 200, -3 * 100)
        tree.links.new(join.geometry_out, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        frame = Frame(tree, location=(0, 0), label="Control",
                      name="ControlFrame")
        self.x_min_node = InputValue(tree, location=(0, 2),
                                     value=self.x_domain[0], name="XMin",
                                     parent=frame)
        self.x_max_node = InputValue(tree, location=(0, 1),
                                     value=self.x_domain[1], name="XMax",
                                     parent=frame)
        self.y_min_node = InputValue(tree, location=(0, 0),
                                     value=self.y_domain[0], name="YMin",
                                     parent=frame)
        self.y_max_node = InputValue(tree, location=(0, -1),
                                     value=self.y_domain[1], name="YMax",
                                     parent=frame)
        self.width_node = InputValue(tree, location=(0, -2), value=self.width,
                                     name="Width", parent=frame)
        self.height_node = InputValue(tree, location=(0, -3),
                                      value=self.height, name="Height",
                                      parent=frame)
        self.thickness_node = InputValue(tree, location=(0, -4),
                                         value=self.thickness,
                                         name="Thickness", parent=frame)
        # the line is built in argument space, x0..x1 along x, and only
        # scaled to world lengths once it has been lifted
        self.start = CombineXYZ(tree, location=(1, 2),
                                x=self.x_min_node.std_out, name="LineStart",
                                parent=frame)
        self.end = CombineXYZ(tree, location=(1, 0), x=self.x_max_node.std_out,
                              name="LineEnd", parent=frame)
        return frame

    # ------------------------------------------------------------------
    def _line_frame(self, tree, kind, row=0):
        label = kind.upper()
        frame = Frame(tree, location=(0, 0), label=label + " line",
                      name=label + "Frame")
        amount = InputValue(tree, location=(3, row + 4),
                            value=self.amounts[kind], name=label + "Amount",
                            parent=frame)
        reveal = InputValue(tree, location=(3, row + 3),
                            value=self.reveals[kind], name=label + "Reveal",
                            parent=frame)
        line = MeshLine(tree, location=(3, row + 1), count=self.count,
                        start_location=self.start.std_out,
                        end_location=self.end.std_out, name=label + "GridLine",
                        parent=frame)

        position = Position(tree, location=(3, row - 1), parent=frame)
        argument = SeparateXYZ(tree, location=(4, row - 1),
                               vector=position.std_out, parent=frame)
        bessel = BesselNode(tree, kind=kind, location=(5, row - 1),
                            name="Bessel" + label, parent=frame)
        tree.links.new(argument.x, bessel.node.inputs[0])

        cut = ("y,y0,<,y,y1,>,or,"
               "pos_x,x0,-,x1,x0,-,/,reveal,>,or")
        if kind == "y0":
            cut += ",pos_x,0,>,not,or"
        lift = make_function(
            tree, location=(6, row + 1), parent=frame,
            functions={"position": ["pos_x,w,*,x1,x0,-,/", "0",
                                    "y,h,*,y1,y0,-,/"],
                       "cut": cut},
            aux_functions={"y": "f,amp,*"},
            inputs=["pos", "f", "amp", "reveal", "x0", "x1", "y0", "y1", "w",
                    "h"],
            outputs=["position", "cut"],
            vectors=["pos", "position"],
            scalars=["f", "amp", "reveal", "x0", "x1", "y0", "y1", "w", "h",
                     "y", "cut"],
            name=label + "Lift", hide=False)
        tree.links.new(position.std_out, lift.inputs["pos"])
        tree.links.new(bessel.std_out, lift.inputs["f"])
        for socket, dial in (("amp", amount), ("reveal", reveal),
                             ("x0", self.x_min_node), ("x1", self.x_max_node),
                             ("y0", self.y_min_node), ("y1", self.y_max_node),
                             ("w", self.width_node), ("h", self.height_node)):
            tree.links.new(dial.std_out, lift.inputs[socket])

        # deleted before it is moved: both read the position field, and the
        # cut is decided on the unscaled argument
        cull = DeleteGeometry(tree, location=(8, row + 2), domain="POINT",
                              geometry=line.geometry_out,
                              selection=lift.outputs["cut"],
                              name=label + "Cut", parent=frame)
        lifted = SetPosition(tree, location=(9, row + 2),
                             geometry=cull.geometry_out,
                             position=lift.outputs["position"],
                             name=label + "Lifted", parent=frame)

        curve = MeshToCurve(tree, location=(10, row + 2),
                            mesh=lifted.geometry_out, parent=frame)
        profile = CurveCircle(tree, location=(10, row), resolution=8,
                              radius=self.thickness_node.std_out, parent=frame)
        tube = CurveToMesh(tree, location=(11, row + 2),
                           curve=curve.geometry_out,
                           profile_curve=profile.geometry_out, parent=frame)
        smooth = SetShadeSmooth(tree, location=(12, row + 2),
                                geometry=tube.geometry_out, parent=frame)
        painted = SetMaterial(tree, location=(13, row + 2),
                              geometry=smooth.geometry_out,
                              material=get_texture(self.colors[kind],
                                                   **self.kwargs),
                              name=label + "Material", parent=frame)
        self.materials.append(painted.material)
        return painted.geometry_out


#: the depth of the dent a body of the given Depth/Softening makes, as a
#: function of the distance d to it: a softened 1/d, so the bottom is round
WELL = "depth,-1,*,{d},{d},*,softening,softening,*,/,1,+,sqrt,/"


class GravitationalWavesModifier(GeometryNodesModifier):
    r"""Two spheres in orbit on a wire sheet that dents under them and ripples
    away from them.

    A cartoon of a compact binary, not a solution of Einstein's equations.
    The sheet is the familiar rubber-sheet picture of a static field - every
    body presses a softened well :math:`-D/\sqrt{1+d^2/s^2}` into it - and
    on top of that runs what a rotating pair of equal masses radiates to
    lowest order, the mass quadrupole: a wave at **twice** the orbital
    frequency with two arms,

    .. math::
        h(r,\varphi,t) = A\, g(r)\,
            \cos 2\big(\varphi - \Omega\,(t - r/c)\big),
        \qquad \Omega = 2\pi/T,\quad \lambda = \pi c/\Omega,

    which is the two-armed spiral of every LIGO illustration: at r = 0 the
    crests lie along the line through the bodies and they lag behind it by
    the travel time r/c further out. The envelope g is the approximation:
    zero inside the orbit, rising over half a wavelength (where the near zone
    would be) and then falling like :math:`1/\sqrt{1+r/\lambda}`. A 3D wave
    falls as 1/r, but on a sheet seen at an angle that empties the frame
    after two rings; the gentler fall keeps the spiral readable to the edge.

    The sheet is a grid of lines rather than a surface - ``lines`` curves in
    each direction, each resampled to ``resolution`` points and swept into a
    thin tube - so it reads as space rather than as water. The spheres sit
    in the bottom of their own wells.

    Everything is driven by ``Scene Time``, so the binary orbits and the
    spiral runs **without a keyframe**. The dials, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, label)``:

    ``Period``
        T, the orbital period in seconds. The waves have period T/2.
        Ramping it re-phases the orbit (the phase is :math:`2\pi t/T`, not an
        integral), so a chirp has to be short or start from t = 0.
    ``Separation``
        distance between the two centres.
    ``WaveSpeed``
        c. With T it fixes the wavelength :math:`\lambda = cT/2`.
    ``WaveAmplitude``
        A, the height of the ripples.
    ``Depth``, ``Softening``
        D and s of each well: how deep, and how wide the bottom is.
    ``Radius``
        the radius of the spheres.

    :param name: name of the node group and of the modifier.
    :param size: edge length of the square sheet.
    :param lines: number of grid lines in each direction.
    :param resolution: points per grid line; the spiral needs about ten per
        wavelength and the wells about five across ``softening``.
    :param thickness: radius of the wire tubes.
    :param period, separation, wave_speed, wave_amplitude, depth, softening,
        radius: the start values of the dials above.
    :param wire_color, sphere_color: palette names of the two materials.
    :param wire_emission, sphere_emission: emission of the two materials;
        the wire is unlit by default, so it reads as a surface in the scene's
        light rather than as a glowing net.

    For an overlay, the sheet alone on a transparent world::

        waves = GravitationalWavesModifier(size=12, period=2, wave_speed=2)
        sheet = Plane(name="GravitationalWaves", u=[-1, 1], v=[-1, 1])
        sheet.add_mesh_modifier(type='NODES', node_modifier=waves)
        sheet.appear(begin_time=0, transition_time=0)
        amplitude = ibpy.get_geometry_node_from_modifier(waves, "WaveAmplitude")
        ibpy.change_default_value(amplitude, from_value=0, to_value=0.25,
                                  begin_time=1, transition_time=3)
    """

    def __init__(self, name="GravitationalWaves", size=12.0, lines=49,
                 resolution=301, thickness=0.012, period=2.0, separation=1.2,
                 wave_speed=2.0, wave_amplitude=0.25, depth=0.8,
                 softening=0.35, radius=0.25, wire_color="example",
                 sphere_color="important", wire_emission=0.0,
                 sphere_emission=0.5, **kwargs):
        self.size = size
        self.lines = int(lines)
        self.resolution = int(resolution)
        self.thickness = thickness
        self.period = period
        self.separation = separation
        self.wave_speed = wave_speed
        self.wave_amplitude = wave_amplitude
        self.depth = depth
        self.softening = softening
        self.radius = radius
        self.wire_color = wire_color
        self.sphere_color = sphere_color
        self.wire_emission = wire_emission
        self.sphere_emission = sphere_emission
        super().__init__(name=name, automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._control_frame(tree)
        sheet = self._grid_frame(tree)
        wire = self._wave_frame(tree, control, sheet)
        bodies = self._bodies_frame(tree, control)
        geometry = self._display_frame(tree, control, wire, bodies)
        self.group_outputs.location = (14 * 200, 0)
        tree.links.new(geometry, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        frame = Frame(tree, location=(0, 0), label="Control",
                      name="ControlFrame")
        clock = SceneTime(tree, location=(0, 4), std_out="Seconds",
                          name="Clock", parent=frame)
        dials = {}
        for row, (key, label, value) in enumerate([
                ("period", "Period", self.period),
                ("separation", "Separation", self.separation),
                ("wave_speed", "WaveSpeed", self.wave_speed),
                ("wave_amplitude", "WaveAmplitude", self.wave_amplitude),
                ("depth", "Depth", self.depth),
                ("softening", "Softening", self.softening),
                ("radius", "Radius", self.radius)]):
            dial = InputValue(tree, location=(0, 3 - row), value=value,
                              name=label, parent=frame)
            dials[key] = dial.std_out
        dials["time"] = clock.std_out
        return dials

    # ------------------------------------------------------------------
    def _grid_frame(self, tree):
        frame = Frame(tree, location=(0, 0), label="Grid", name="GridFrame")
        half = self.size / 2
        # one line along x, finely resampled, copied onto every point of a
        # coarse line along y: the rows. The columns are the rows turned by
        # a right angle
        row = MeshLine(tree, location=(2, 6), count=self.resolution,
                       start_location=Vector((-half, 0, 0)),
                       end_location=Vector((half, 0, 0)),
                       name="Row", parent=frame)
        row_curve = MeshToCurve(tree, location=(3, 6), mesh=row.geometry_out,
                                name="RowCurve", parent=frame)
        anchors = MeshLine(tree, location=(2, 5), count=self.lines,
                           start_location=Vector((0, -half, 0)),
                           end_location=Vector((0, half, 0)),
                           name="RowAnchors", parent=frame)
        rows = InstanceOnPoints(tree, location=(4, 6),
                                points=anchors.geometry_out,
                                instance=row_curve.geometry_out,
                                name="Rows", parent=frame)
        rows = RealizeInstances(tree, location=(5, 6),
                                geometry=rows.geometry_out,
                                name="RealizeRows", parent=frame)
        columns = TransformGeometry(tree, location=(6, 5),
                                    geometry=rows.geometry_out,
                                    rotation=Vector((0, 0, pi / 2)),
                                    name="Columns", parent=frame)
        sheet = JoinGeometry(tree, location=(7, 6),
                             geometry=[rows.geometry_out, columns.geometry_out],
                             name="Sheet", parent=frame)
        return sheet.geometry_out

    # ------------------------------------------------------------------
    def _wave_frame(self, tree, control, sheet):
        frame = Frame(tree, location=(0, 0), label="Wells and Waves",
                      name="WaveFrame")
        # the sheet is still flat here, so the position is the point in the
        # plane the height is a function of
        position = Position(tree, location=(7, 4), name="SheetPosition",
                            hide=True, parent=frame)
        aux = {
            "om": "%s,period,/" % repr(tau),
            "ph": "om,time,*",
            "bx": "separation,2,/,ph,cos,*",
            "by": "separation,2,/,ph,sin,*",
            "d0": "pos_x,bx,-,pos_x,bx,-,*,pos_y,by,-,pos_y,by,-,*,+,sqrt",
            "d1": "pos_x,bx,+,pos_x,bx,+,*,pos_y,by,+,pos_y,by,+,*,+,sqrt",
            "well": WELL.format(d="d0") + "," + WELL.format(d="d1") + ",+",
            "r": "pos_x,pos_x,*,pos_y,pos_y,*,+,sqrt",
            "phi": "pos_y,pos_x,atan2",
            "lam": "pi,wave_speed,*,om,/",
            # zero inside the orbit, smoothstep up over half a wavelength
            "ramp": "r,separation,2,/,-,lam,2,/,/,0,max,1,min",
            "env": "ramp,ramp,*,3,2,ramp,*,-,*,1,r,lam,/,+,sqrt,/",
            "arg": "phi,ph,-,r,wave_speed,/,om,*,+,2,*",
            "ripple": "wave_amplitude,env,*,arg,cos,*",
        }
        inputs = ["pos", "time", "period", "separation", "wave_speed",
                  "wave_amplitude", "depth", "softening"]
        height = make_function(tree, location=(8, 4), name="Height",
                               functions={"offset": ["0", "0", "well,ripple,+"],
                                          "ripple": "ripple"},
                               aux_functions=aux, inputs=inputs,
                               outputs=["offset", "ripple"],
                               vectors=["pos", "offset"],
                               scalars=inputs[1:] + ["ripple"] + list(aux),
                               parent=frame, hide=False)
        tree.links.new(position.std_out, height.inputs["pos"])
        for key in inputs[1:]:
            tree.links.new(control[key], height.inputs[key])

        # the ripple alone is kept for the material; the well is the shape
        ripple = StoreNamedAttribute(tree, location=(9, 6), data_type="FLOAT",
                                     domain="POINT", name="Ripple",
                                     value=height.outputs["ripple"],
                                     parent=frame)
        tree.links.new(sheet, ripple.geometry_in)
        bent = SetPosition(tree, location=(10, 6),
                           offset=height.outputs["offset"],
                           name="Bend", parent=frame)
        tree.links.new(ripple.geometry_out, bent.geometry_in)
        return bent.geometry_out

    # ------------------------------------------------------------------
    def _bodies_frame(self, tree, control):
        frame = Frame(tree, location=(0, 0), label="Bodies",
                      name="BodiesFrame")
        pair = Points(tree, location=(7, 1), count=2, name="Pair",
                      parent=frame)
        index = Index(tree, location=(7, 0), name="BodyIndex", hide=True,
                      parent=frame)
        aux = {
            "om": "%s,period,/" % repr(tau),
            "ph": "om,time,*",
            "sign": "1,2,index,*,-",
            "own": WELL.format(d="0"),
            "other": WELL.format(d="separation"),
        }
        inputs = ["index", "time", "period", "separation", "depth",
                  "softening", "radius"]
        # each in the bottom of the dent the two wells make together, resting
        # on it rather than sunk to the centre
        orbit = make_function(tree, location=(8, 1), name="Orbit",
                              functions={"center": [
                                  "sign,separation,*,2,/,ph,cos,*",
                                  "sign,separation,*,2,/,ph,sin,*",
                                  "own,other,+,radius,+"]},
                              aux_functions=aux, inputs=inputs,
                              outputs=["center"], vectors=["center"],
                              scalars=inputs + list(aux), parent=frame,
                              hide=False)
        tree.links.new(index.std_out, orbit.inputs["index"])
        for key in inputs[1:]:
            tree.links.new(control[key], orbit.inputs[key])
        placed = SetPosition(tree, location=(9, 1), geometry=pair.geometry_out,
                             position=orbit.outputs["center"],
                             name="PlaceBodies", parent=frame)
        ball = UVSphere(tree, location=(9, 0), radius=1, name="Ball",
                        parent=frame)
        bodies = InstanceOnPoints(tree, location=(10, 1),
                                  points=placed.geometry_out,
                                  instance=ball.geometry_out,
                                  scale=control["radius"], name="Bodies",
                                  parent=frame)
        bodies = RealizeInstances(tree, location=(11, 1),
                                  geometry=bodies.geometry_out,
                                  name="RealizeBodies", parent=frame)
        return bodies.geometry_out

    # ------------------------------------------------------------------
    def _display_frame(self, tree, control, wire, bodies):
        frame = Frame(tree, location=(0, 0), label="Display",
                      name="DisplayFrame")
        profile = CurveCircle(tree, location=(11, 5), resolution=6,
                              radius=self.thickness, name="WireProfile",
                              parent=frame)
        tubes = CurveToMesh(tree, location=(12, 6), curve=wire,
                            profile_curve=profile.geometry_out, fill_caps=False,
                            name="Wire", parent=frame)
        wire_paint = SetMaterial(tree, location=(13, 6),
                                 geometry=tubes.geometry_out,
                                 material=get_texture(self.wire_color,
                                                      emission=self.wire_emission),
                                 name="PaintWire", parent=frame)
        smooth = SetShadeSmooth(tree, location=(12, 1), geometry=bodies,
                                name="SmoothBodies", parent=frame)
        body_paint = SetMaterial(tree, location=(13, 1),
                                 geometry=smooth.geometry_out,
                                 material=get_texture(self.sphere_color,
                                                      emission=self.sphere_emission),
                                 name="PaintBodies", parent=frame)
        self.materials.append(wire_paint.material)
        self.materials.append(body_paint.material)
        joined = JoinGeometry(tree, location=(14, 3),
                              geometry=[wire_paint.geometry_out,
                                        body_paint.geometry_out],
                              name="Binary", parent=frame)
        return joined.geometry_out

    # ------------------------------------------------------------------
    def height_numpy(self, points, seconds=0.0):
        """The same height of the sheet, in numpy - the tree's mirror.

        :param points: ``(n, 2)`` or ``(n, 3)`` positions; only x and y are read.
        :param seconds: the scene time the tree reads off the clock.
        """
        points = np.asarray(points, dtype=float)
        x, y = points[:, 0], points[:, 1]
        om = tau / self.period
        ph = om * seconds
        bx, by = self.separation / 2 * np.cos(ph), self.separation / 2 * np.sin(ph)
        well = sum(-self.depth / np.sqrt(1 + ((x - sx) ** 2 + (y - sy) ** 2)
                                         / self.softening ** 2)
                   for sx, sy in ((bx, by), (-bx, -by)))
        r = np.hypot(x, y)
        lam = pi * self.wave_speed / om
        ramp = np.clip((r - self.separation / 2) / (lam / 2), 0, 1)
        env = ramp * ramp * (3 - 2 * ramp) / np.sqrt(1 + r / lam)
        arg = 2 * (np.arctan2(y, x) - ph + om * r / self.wave_speed)
        return well + self.wave_amplitude * env * np.cos(arg)


class ElectromagneticWaveModifier(GeometryNodesModifier):
    r"""A linearly polarised plane light wave: rows of E and B arrows along
    the direction of propagation, swinging back and forth in step.

    The textbook picture of light. The wave runs along x, the electric field
    points along z and the magnetic field along y, so the three are
    mutually perpendicular, and in vacuum the two fields are in phase:

    .. math::
        \vec E = E_0\,g\,\cos(kx-\omega t)\,\hat z,\qquad
        \vec B = \tfrac{E_0}{c}\,g\,\cos(kx-\omega t)\,\hat y,
        \qquad k = 2\pi/\lambda,\quad \omega = 2\pi f.

    (B is drawn at the height ``b_ratio`` E_0 - in SI units it would be
    invisible.) Each field is drawn twice: as ``arrows`` arrows standing on
    the axis, shaft plus a head that keeps its size and only shrinks as the
    field goes through zero, and as the envelope curve through their tips.

    With ``front`` the wave is *emitted*: g is a smooth step that runs from
    the left end of the axis at the phase speed :math:`c=\lambda f`, starting
    at ``impact``, so the field switches on behind it and is zero ahead of it.
    Without, g = 1 and the wave fills the axis from the first frame.

    Everything is driven by ``Scene Time``. The dials, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, label)``:
    ``Wavelength``, ``Frequency``, ``Amplitude`` (E_0), ``Impact`` and
    ``FrontWidth``.

    :param name: name of the node group and of the modifier.
    :param length: length of the axis, centred on the origin.
    :param arrows: arrows per field.
    :param resolution: points per envelope curve.
    :param wavelength, frequency, amplitude: start values of the dials.
    :param b_ratio: height of B relative to E.
    :param front, impact, front_width: the emission front (see above).
    :param thickness: radius of the arrow shafts; heads, curves and the axis
        are sized from it.
    :param head: length of an arrow head.
    :param e_color, b_color, axis_color: palette names.
    :param emission: emission of the two field materials.

    Example, light switched on two seconds into the shot::

        light = ElectromagneticWaveModifier(length=12, impact=2)
        wave = Plane(name="LightWave", u=[-1, 1], v=[-1, 1])
        wave.add_mesh_modifier(type='NODES', node_modifier=light)
        wave.appear(begin_time=0, transition_time=0)
    """

    def __init__(self, name="ElectromagneticWave", length=12.0, arrows=49,
                 resolution=401, wavelength=3.0, frequency=0.5,
                 amplitude=1.5, b_ratio=0.75, front=True, impact=0.0,
                 front_width=None, thickness=0.018, head=0.15,
                 e_color="custom1", b_color="drawing", axis_color="text",
                 emission=0.3, **kwargs):
        self.length = length
        self.arrows = int(arrows)
        self.resolution = int(resolution)
        self.wavelength = wavelength
        self.frequency = frequency
        self.amplitude = amplitude
        self.b_ratio = b_ratio
        self.front = front
        self.impact = impact
        self.front_width = wavelength / 2 if front_width is None else front_width
        self.thickness = thickness
        self.head = head
        self.e_color = e_color
        self.b_color = b_color
        self.axis_color = axis_color
        self.emission = emission
        super().__init__(name=name, automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        control = self._control_frame(tree)
        axis = self._axis_frame(tree)
        e_field = self._field_frame(tree, control, "E", row=0)
        b_field = self._field_frame(tree, control, "B", row=-6)
        joined = JoinGeometry(tree, location=(15, 0),
                              geometry=[axis, e_field, b_field],
                              name="Light")
        self.group_outputs.location = (16 * 200, 0)
        tree.links.new(joined.geometry_out, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        frame = Frame(tree, location=(0, 0), label="Control",
                      name="ControlFrame")
        clock = SceneTime(tree, location=(0, 4), std_out="Seconds",
                          name="Clock", parent=frame)
        dials = {"time": clock.std_out}
        for row, (key, label, value) in enumerate([
                ("wavelength", "Wavelength", self.wavelength),
                ("frequency", "Frequency", self.frequency),
                ("amplitude", "Amplitude", self.amplitude),
                ("impact", "Impact", self.impact),
                ("edge", "FrontWidth", self.front_width)]):
            dial = InputValue(tree, location=(0, 3 - row), value=value,
                              name=label, parent=frame)
            dials[key] = dial.std_out
        return dials

    # ------------------------------------------------------------------
    def _axis_frame(self, tree):
        frame = Frame(tree, location=(0, 0), label="Axis", name="AxisFrame")
        half = self.length / 2
        line = MeshLine(tree, location=(2, 8), count=2,
                        start_location=Vector((-half, 0, 0)),
                        end_location=Vector((half, 0, 0)),
                        name="AxisLine", parent=frame)
        curve = MeshToCurve(tree, location=(3, 8), mesh=line.geometry_out,
                            name="AxisCurve", parent=frame)
        profile = CurveCircle(tree, location=(3, 7), resolution=8,
                              radius=0.6 * self.thickness,
                              name="AxisProfile", parent=frame)
        tube = CurveToMesh(tree, location=(4, 8), curve=curve.geometry_out,
                           profile_curve=profile.geometry_out,
                           name="AxisTube", parent=frame)
        paint = SetMaterial(tree, location=(5, 8), geometry=tube.geometry_out,
                            material=get_texture(self.axis_color),
                            name="PaintAxis", parent=frame)
        self.materials.append(paint.material)
        return paint.geometry_out

    # ------------------------------------------------------------------
    def _field_frame(self, tree, control, field, row):
        frame = Frame(tree, location=(0, 0), label="%s Field" % field,
                      name="%sFieldFrame" % field)
        half = self.length / 2
        position = Position(tree, location=(2, row + 3), hide=True,
                            name="%sPosition" % field, parent=frame)

        # E stands along z; B along y, which is z turned by -90 degrees
        # about x. A negative value turns the arrow over by another 180.
        if field == "E":
            scale, turn = "amplitude", "0"
            offset = ["0", "0", "s"]
            tip = ["0", "0", "s,sgn,shaft,*"]
            color = self.e_color
        else:
            scale, turn = "amplitude,%s,*" % repr(self.b_ratio), repr(-pi / 2)
            offset = ["0", "s", "0"]
            tip = ["0", "s,sgn,shaft,*", "0"]
            color = self.b_color
        aux = {
            "k": "%s,wavelength,/" % repr(tau),
            "wt": "time,frequency,*,%s,*" % repr(tau),
        }
        if self.front:
            # the front leaves the left end at impact and travels at
            # c = lambda f; the field is switched on over `edge` behind it
            aux["reach"] = ("%s,time,impact,-,wavelength,*,frequency,*,+"
                            % repr(-half))
            aux["g"] = "reach,pos_x,-,edge,/,0,max,1,min"
            aux["gate"] = "g,g,*,3,2,g,*,-,*"
        else:
            aux["gate"] = "1"
        aux["s"] = "%s,gate,*,k,pos_x,*,wt,-,cos,*" % scale
        aux["shaft"] = "s,abs,%s,-,0,max" % repr(self.head)
        inputs = ["pos", "time", "wavelength", "frequency", "amplitude",
                  "impact", "edge"]
        functions = {
            "offset": offset,
            "rotation": ["%s,s,0,<,pi,*,+" % turn, "0", "0"],
            "shaft_scale": ["1", "1", "shaft"],
            "tip": tip,
            "head_scale": "s,abs,%s,/,1,min" % repr(self.head),
            "silent": "gate,0.001,<",
        }
        vectors = ["pos", "offset", "rotation", "shaft_scale", "tip"]
        values = make_function(tree, location=(3, row + 3),
                               name="%sValues" % field,
                               functions=functions, aux_functions=aux,
                               inputs=inputs, outputs=list(functions),
                               vectors=vectors,
                               scalars=inputs[1:] + ["head_scale", "silent"]
                                       + list(aux),
                               parent=frame, hide=False)
        tree.links.new(position.std_out, values.inputs["pos"])
        for key in inputs[1:]:
            tree.links.new(control[key], values.inputs[key])

        # the arrows: a unit shaft standing on its base, stretched to
        # length, and a head of fixed size moved to the top of the shaft
        anchors = MeshLine(tree, location=(4, row + 2), count=self.arrows,
                           start_location=Vector((-half, 0, 0)),
                           end_location=Vector((half, 0, 0)),
                           name="%sAnchors" % field, parent=frame)
        cylinder = CylinderMesh(tree, location=(4, row + 1), vertices=12,
                                radius=self.thickness, depth=1,
                                name="%sShaftMesh" % field, parent=frame)
        shaft_mesh = TransformGeometry(tree, location=(5, row + 1),
                                       geometry=cylinder.geometry_out,
                                       translation=Vector((0, 0, 0.5)),
                                       name="%sShaftBase" % field,
                                       parent=frame)
        cone = ConeMesh(tree, location=(4, row), vertices=16,
                        radius_bottom=3 * self.thickness, depth=self.head,
                        name="%sHeadMesh" % field, parent=frame)
        head_mesh = TransformGeometry(tree, location=(5, row),
                                      geometry=cone.geometry_out,
                                      translation=Vector((0, 0, self.head / 2)),
                                      name="%sHeadBase" % field, parent=frame)
        shafts = InstanceOnPoints(tree, location=(6, row + 2),
                                  points=anchors.geometry_out,
                                  instance=shaft_mesh.geometry_out,
                                  rotation=values.outputs["rotation"],
                                  scale=values.outputs["shaft_scale"],
                                  name="%sShafts" % field, parent=frame)
        # the offset is across the axis, so pos_x - all the field reads -
        # is the same at the tip as at the foot
        tips = SetPosition(tree, location=(6, row + 1),
                           geometry=anchors.geometry_out,
                           offset=values.outputs["tip"],
                           name="%sTips" % field, parent=frame)
        heads = InstanceOnPoints(tree, location=(7, row + 1),
                                 points=tips.geometry_out,
                                 instance=head_mesh.geometry_out,
                                 rotation=values.outputs["rotation"],
                                 scale=values.outputs["head_scale"],
                                 name="%sHeads" % field, parent=frame)

        # the envelope through the tips
        line = MeshLine(tree, location=(4, row + 4), count=self.resolution,
                        start_location=Vector((-half, 0, 0)),
                        end_location=Vector((half, 0, 0)),
                        name="%sLine" % field, parent=frame)
        # ahead of the front the envelope would lie on the axis and paint
        # it in the field's colour, so it is cut back to where the wave is
        ahead = DeleteGeometry(tree, location=(5, row + 5),
                               geometry=line.geometry_out,
                               selection=values.outputs["silent"],
                               name="%sAhead" % field, parent=frame)
        bent = SetPosition(tree, location=(5, row + 4),
                           geometry=ahead.geometry_out,
                           offset=values.outputs["offset"],
                           name="%sBend" % field, parent=frame)
        curve = MeshToCurve(tree, location=(6, row + 4),
                            mesh=bent.geometry_out,
                            name="%sCurve" % field, parent=frame)
        profile = CurveCircle(tree, location=(6, row + 3), resolution=8,
                              radius=1.2 * self.thickness,
                              name="%sProfile" % field, parent=frame)
        envelope = CurveToMesh(tree, location=(7, row + 4),
                               curve=curve.geometry_out,
                               profile_curve=profile.geometry_out,
                               name="%sEnvelope" % field, parent=frame)

        joined = JoinGeometry(tree, location=(8, row + 2),
                              geometry=[shafts.geometry_out, heads.geometry_out,
                                        envelope.geometry_out],
                              name="%sJoin" % field, parent=frame)
        realized = RealizeInstances(tree, location=(9, row + 2),
                                    geometry=joined.geometry_out,
                                    name="%sRealize" % field, parent=frame)
        smooth = SetShadeSmooth(tree, location=(10, row + 2),
                                geometry=realized.geometry_out,
                                name="%sSmooth" % field, parent=frame)
        paint = SetMaterial(tree, location=(11, row + 2),
                            geometry=smooth.geometry_out,
                            material=get_texture(color, emission=self.emission),
                            name="Paint%s" % field, parent=frame)
        self.materials.append(paint.material)
        return paint.geometry_out

    # ------------------------------------------------------------------
    def field_numpy(self, x, seconds=0.0):
        """E_z along the axis, in numpy - the tree's mirror (B_y is
        ``b_ratio`` times it).

        :param x: positions along the axis.
        :param seconds: the scene time the tree reads off the clock.
        """
        x = np.asarray(x, dtype=float)
        if self.front:
            reach = -self.length / 2 + (seconds - self.impact) \
                    * self.wavelength * self.frequency
            g = np.clip((reach - x) / self.front_width, 0, 1)
            gate = g * g * (3 - 2 * g)
        else:
            gate = 1.0
        return self.amplitude * gate * np.cos(tau / self.wavelength * x
                                              - tau * self.frequency * seconds)
