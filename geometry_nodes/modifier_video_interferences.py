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
"""
import numpy as np

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (BESSEL_OPS, CombineXYZ, CubeMesh, CurveCircle,
                                  CurveLine, CurveToMesh,
                                  DeleteGeometry, DistributePointsInVolume,
                                  Frame, Grid, IcoSphere, IndexSwitch, InputInteger,
                                  InputValue, InputVector,
                                  InstanceOnPoints, JoinGeometry, MathNode,
                                  MergeByDistance, MeshToVolume,
                                  NamedAttribute, Points, Position, RandomValue,
                                  RealizeInstances, SceneTime, SetMaterial, SetPosition,
                                  SetShadeSmooth, StoreNamedAttribute, TransformGeometry,
                                  VolumeCube, WireFrame, bessel_jm_rpn, make_function)
from interface.ibpy import Vector

pi = np.pi
tau = 2 * pi


def _vector(value):
    """``Vector`` from a scalar (isotropic), a triple, or a ``Vector``."""
    if isinstance(value, (int, float)):
        return Vector([value, value, value])
    return Vector(value)


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
    :param box_color: palette name for a wireframe of the box, or ``None``
        for no box.
    :param box_radius: tube radius of that wireframe.
    """

    def __init__(self, size=4.0, center=(0, 0, 0), shape="cube",
                 method="rejection", count=20000, resolution=64,
                 voxel_amount=64.0, seed=0, radius=0.02, subdivisions=1,
                 color="drawing", color_by_density=False, gradient=None,
                 emission_by_density=False,
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
    def create_node(self, tree, **kwargs):
        candidates, position = self._region_frame(tree)
        points = self._sampling_frame(tree, candidates, position)
        geometry = self._display_frame(tree, points)
        tree.links.new(geometry, self.group_outputs.inputs["Geometry"])

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
        frame = Frame(tree, location=(0, 0), label="Region", name="RegionFrame")
        position = Position(tree, location=(0, 0), hide=True, parent=frame)

        if self.method == "grid":
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

        cloud = Points(tree, location=(3, 0), count=self.candidates,
                       name="Candidates", parent=frame)
        draw = RandomValue(tree, location=(3, -1), data_type="FLOAT_VECTOR",
                           min=self.box_min, max=self.box_max, seed=self.seed,
                           name="UniformDraw", parent=frame)
        placed = SetPosition(tree, location=(4, 0), geometry=cloud.geometry_out,
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
        """
        frame = Frame(tree, location=(6, 0), label="Sampling",
                      name="SamplingFrame")
        points = candidates
        density = None

        if self.method == "rejection":
            density = self.density(tree, position.std_out, location=(0, -1))
            if density is not None:
                density.node.parent = frame.node
                # one uniform draw per point, compared against f: the point
                # survives where its draw falls under the curve
                draw = RandomValue(tree, location=(0, -2), data_type="FLOAT",
                                   min=0.0, max=1.0, seed=self.seed + 1,
                                   name="RejectionDraw", parent=frame)
                test = make_function(tree, location=(1, -1),
                                     functions={"reject": "u,f,>"},
                                     inputs=["u", "f"], outputs=["reject"],
                                     scalars=["u", "f", "reject"],
                                     name="RejectionTest", hide=True)
                test.parent = frame.node
                tree.links.new(draw.std_out, test.inputs["u"])
                tree.links.new(density, test.inputs["f"])
                cull = DeleteGeometry(tree, location=(2, 0), domain="POINT",
                                      geometry=points,
                                      selection=test.outputs["reject"],
                                      name="Reject", parent=frame)
                points = cull.geometry_out
        elif self.method == "uniform" or self.color_by_density \
                or self.emission_by_density:
            density = self.density(tree, position.std_out, location=(0, -1))
            if density is not None:
                density.node.parent = frame.node

        if density is not None:
            # carried on the points so that a material can read it; the value
            # is f at the point's own position, i.e. the local intensity
            store = StoreNamedAttribute(tree, location=(3, 0), name="intensity",
                                        data_type="FLOAT", domain="POINT",
                                        value=density)
            store.node.parent = frame.node
            tree.links.new(points, store.geometry_in)
            points = store.geometry_out

        return points

    # ------------------------------------------------------------------
    def _display_frame(self, tree, points):
        """A small sphere on every point, painted, plus the optional box."""
        frame = Frame(tree, location=(11, 0), label="Display",
                      name="DisplayFrame")
        ball = IcoSphere(tree, location=(0, -1), radius=self.radius,
                         subdivisions=self.subdivisions, name="Ball",
                         parent=frame)
        instances = InstanceOnPoints(tree, location=(1, 0), points=points,
                                     instance=ball.geometry_out,
                                     name="Instances", parent=frame)
        # realised, not left as instances, so that the ``intensity``
        # attribute reaches the shader on the mesh domain it reads
        realized = RealizeInstances(tree, location=(2, 0), name="Realize",
                                    parent=frame)
        tree.links.new(instances.geometry_out, realized.geometry_in)
        geometry = realized.geometry_out

        if self.emission_by_density:
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
            painted = SetMaterial(tree, location=(3, 0), geometry=geometry,
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
            painted = SetMaterial(tree, location=(3, 0), geometry=geometry,
                                  material=material, name="PaintPoints",
                                  parent=frame)
            self.materials.append(painted.material)
            geometry = painted.geometry_out
        elif self.color is not None:
            painted = SetMaterial(tree, location=(3, 0), geometry=geometry,
                                  material=self.color, name="PaintPoints",
                                  parent=frame, **self.kwargs)
            self.materials.append(painted.material)
            geometry = painted.geometry_out

        if self.box_color is not None:
            box = CubeMesh(tree, location=(0, -3), size=self.size,
                           name="BoxOutline", parent=frame)
            shifted = TransformGeometry(tree, location=(1, -3),
                                        geometry=box.geometry_out,
                                        translation=self.center,
                                        name="PlaceBoxOutline", parent=frame)
            wires = WireFrame(tree, location=(2, -3), radius=self.box_radius,
                              geometry=shifted.geometry_out, name="BoxWires",
                              parent=frame)
            box_paint = SetMaterial(tree, location=(3, -3),
                                    geometry=wires.geometry_out,
                                    material=self.box_color, name="PaintBox",
                                    parent=frame, **self.kwargs)
            self.materials.append(box_paint.material)
            joined = JoinGeometry(tree, location=(4, 0), name="JoinDisplay",
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

    :param spacing: g, the distance between neighbouring sources, in the
        same units as ``wavelength``.
    :param wavelength: lambda at build time; the dial a scene ramps.
    :param max_order: highest |n| built. Orders beyond
        ``spacing/wavelength`` never appear, so this only has to cover the
        shortest wavelength a scene reaches: ``int(g/lambda_min)``.
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
    """

    def __init__(self, spacing=1.0, wavelength=0.5, max_order=3,
                 reach=2.5, radius=0.008, resolution=8,
                 axis=(1, 0, 0), normal=(0, 0, 1),
                 color="text", name="FarField", **kwargs):
        self.spacing = spacing
        self.wavelength = wavelength
        self.max_order = int(max_order)
        self.reach = reach
        self.radius = radius
        self.resolution = resolution
        self.axis = Vector(axis).normalized()
        self.normal = Vector(normal).normalized()
        self.color = color
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
            aux["s" + tag] = "%s,lam,*,gap,/" % repr(float(order))
            aux["e" + tag] = "s%s,abs,1,<" % tag
            aux["c" + tag] = "1,s{0},s{0},*,-,0,max,sqrt".format(tag)
            functions["dir" + tag] = "up,c{0},scale,axis,s{0},scale,add".format(tag)
            functions["end" + tag] = "reach,e%s,*" % tag

        # `axis`, `up`, `lam`, `gap`, `reach` and none of them spelled like an
        # operator - a variable called `length` would be read as the vector
        # length and the group would build and compute something else
        function = make_function(tree, location=(4, 0),
                                 functions=functions, aux_functions=aux,
                                 inputs=["lam", "gap", "reach", "axis", "up"],
                                 outputs=list(functions),
                                 vectors=["axis", "up"]
                                         + ["dir" + self._tag(n) for n in self.orders],
                                 scalars=["lam", "gap", "reach"] + list(aux)
                                         + ["end" + self._tag(n) for n in self.orders],
                                 name="MaximaDirections", hide=True)
        links.new(self.wavelength_node.std_out, function.inputs["lam"])
        links.new(self.spacing_node.std_out, function.inputs["gap"])
        links.new(self.reach_node.std_out, function.inputs["reach"])
        links.new(axis.std_out, function.inputs["axis"])
        links.new(normal.std_out, function.inputs["up"])

        rays = Frame(tree, location=(8, 0), label="Rays", name="RaysFrame")
        lines = []
        for i, order in enumerate(self.orders):
            tag = self._tag(order)
            line = CurveLine(tree, location=(0, -i), mode="DIRECTION",
                             start=Vector(),
                             direction=function.outputs["dir" + tag],
                             length=function.outputs["end" + tag],
                             name="Order%s" % tag, parent=rays)
            lines.append(line.geometry_out)
        joined = JoinGeometry(tree, location=(2, 0), geometry=lines,
                              name="JoinRays", parent=rays)
        profile = CurveCircle(tree, location=(2, -len(lines)), radius=self.radius,
                              resolution=self.resolution, name="RayProfile",
                              parent=rays)
        tubes = CurveToMesh(tree, location=(3, 0), curve=joined.geometry_out,
                            profile_curve=profile.geometry_out, fill_caps=True,
                            name="RayTubes", parent=rays)
        geometry = tubes.geometry_out

        if self.color is not None:
            painted = SetMaterial(tree, location=(4, 0), geometry=geometry,
                                  material=self.color, name="PaintRays",
                                  parent=rays, **self.kwargs)
            self.materials.append(painted.material)
            geometry = painted.geometry_out

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
    """

    def __init__(self, name="WaveVisualization", size=8.0, resolution=301,
                 sources=((0.0, 0.0),), wavelength=0.8, frequency=1.0,
                 amplitude=0.25, source_radius=None, attribute="elongation",
                 material="interference", shade_smooth=True, **kwargs):
        self.name = name
        self.size = size
        self.resolution = resolution
        self.sources = [Vector((s[0], s[1], 0)) for s in sources]
        if not self.sources:
            raise ValueError("a wave needs at least one source")
        self.wavelength = wavelength
        self.frequency = frequency
        self.amplitude = amplitude
        # the same default as interference_texture, so that the two clamps
        # coincide when neither is given explicitly
        self.source_radius = wavelength / 20 if source_radius is None \
            else source_radius
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
        control = self._control_frame(tree)
        elongation = self._wave_frame(tree, control)
        geometry = self._geometry_frame(tree, elongation)
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
        frame = Frame(tree, location=(0, 2), label="Control",
                      name="ControlFrame")
        clock = SceneTime(tree, location=(0, 1), std_out="Seconds",
                          name="Clock", parent=frame)
        wavelength = InputValue(tree, location=(0, 0), value=self.wavelength,
                                name="Wavelength", parent=frame)
        frequency = InputValue(tree, location=(0, -1), value=self.frequency,
                               name="Frequency", parent=frame)
        amplitude = InputValue(tree, location=(0, -2), value=self.amplitude,
                               name="Amplitude", parent=frame)
        sources = [InputVector(tree, location=(0, -3 - j), vector=source,
                               name="Source%d" % j, hide=True, parent=frame)
                   for j, source in enumerate(self.sources)]
        return {"time": clock.std_out,
                "wavelength": wavelength.std_out,
                "frequency": frequency.std_out,
                "amplitude": amplitude.std_out,
                "sources": [s.std_out for s in sources]}

    # ------------------------------------------------------------------
    def _wave_frame(self, tree, control):
        r"""u(r, t), the whole of it, as one function group.

        The auxiliaries are the arithmetic worth naming, and they are shared
        rather than repeated: ``k`` and ``amp`` are computed once for all
        sources, and each source's ``r`` is used twice - once clamped into the
        Bessel argument, and once (through it) by both cylinder functions.

        ``amp`` is ``A pi / sqrt(lambda)``, which looks arbitrary and is not:
        :math:`J_0(x) \sim \sqrt{2/\pi x}\cos(x - \pi/4)`, so a far-field
        amplitude of :math:`A/\sqrt r` needs a factor
        :math:`\sqrt{\pi k/2} = \pi/\sqrt\lambda`. It is built from the
        ``Wavelength`` socket rather than baked in as a number, so a scene
        that sweeps the wavelength keeps the same wave height as it goes.

        :return: the socket carrying the scalar elongation.
        """
        frame = Frame(tree, location=(3, 0), label="Wave Computation",
                      name="WaveFrame")
        position = Position(tree, location=(0, -1), name="GridPosition",
                            hide=True, parent=frame)

        n = len(self.sources)
        aux = {}
        aux["k"] = "%s,wavelength,/" % tau
        aux["wt"] = "time,frequency,*,%s,*" % tau
        aux["amp"] = "amplitude,pi,*,wavelength,sqrt,/"
        for j in range(n):
            # the grid is still flat where this is evaluated, so the distance
            # in the plane is the distance in space
            aux["r%d" % j] = "pos,c%d,sub,length" % j
        for j in range(n):
            # x = k r, held off the pole of Y0
            aux["x%d" % j] = "r%d,%s,max,k,*" % (j, repr(self.source_radius))
        for j in range(n):
            # Re[H0(kr) exp(-i w t)], the outgoing wave
            aux["w%d" % j] = ("amp,x{0},j0,wt,cos,*,"
                              "x{0},y0,wt,sin,*,+,*".format(j))
        aux["u"] = ",".join("w%d" % j for j in range(n)) + ",+" * (n - 1)

        names = ["pos", "time", "frequency", "wavelength", "amplitude"] \
                + ["c%d" % j for j in range(n)]
        wave = make_function(tree, location=(2, 0), name="Elongation",
                             functions={"elongation": "u"},
                             aux_functions=aux,
                             inputs=names, outputs=["elongation"],
                             vectors=["pos"] + ["c%d" % j for j in range(n)],
                             scalars=["time", "frequency", "wavelength",
                                      "amplitude", "elongation"] + list(aux),
                             custom_ops=BESSEL_OPS, parent=frame, hide=False)

        tree.links.new(position.std_out, wave.inputs["pos"])
        for key in ("time", "frequency", "wavelength", "amplitude"):
            tree.links.new(control[key], wave.inputs[key])
        for j, source in enumerate(control["sources"]):
            tree.links.new(source, wave.inputs["c%d" % j])
        return wave.outputs["elongation"]

    # ------------------------------------------------------------------
    def _geometry_frame(self, tree, elongation):
        """Grid -> uv -> store -> lift -> paint.

        The order is the argument. See the class docstring for why the store
        comes before the lift and why the uv map has to be written out by
        hand.

        :return: the geometry socket for the group output.
        """
        frame = Frame(tree, location=(8, 0), label="Geometry",
                      name="GeometryFrame")
        grid = Grid(tree, location=(0, 0), size_x=self.size, size_y=self.size,
                    vertices_x=self.resolution, vertices_y=self.resolution,
                    name="Grid", parent=frame)

        # the grid's uv is an anonymous field; named and put on the corner
        # domain it becomes the uv layer the material samples
        uv = StoreNamedAttribute(tree, location=(1, 0), data_type="FLOAT2",
                                 domain="CORNER", name="UVMap",
                                 value=grid.node.outputs["UV Map"],
                                 parent=frame)
        tree.links.new(grid.geometry_out, uv.geometry_in)

        stored = StoreNamedAttribute(tree, location=(2, 0), data_type="FLOAT",
                                     domain="POINT", name=self.attribute,
                                     value=elongation, parent=frame)
        tree.links.new(uv.geometry_out, stored.geometry_in)

        # read back rather than reusing the socket: this is what makes the
        # elongation the thing that moves the surface, and it costs one node
        # against a second evaluation of every Bessel group
        height = NamedAttribute(tree, location=(3, -2), data_type="FLOAT",
                                name=self.attribute, parent=frame, hide=True)
        offset = CombineXYZ(tree, location=(4, -2), z=height.std_out,
                            name="Lift", parent=frame, hide=True)
        lifted = SetPosition(tree, location=(5, 0),
                             geometry=stored.geometry_out,
                             offset=offset.std_out, name="Displace",
                             parent=frame)
        geometry = lifted.geometry_out

        if self.shade_smooth:
            smooth = SetShadeSmooth(tree, location=(6, 0), geometry=geometry,
                                    name="Smooth", parent=frame)
            geometry = smooth.geometry_out

        if self.paint is not None:
            painted = SetMaterial(tree, location=(7, 0), geometry=geometry,
                                  material=self._texture(), name="Paint",
                                  parent=frame)
            self.material = painted.material
            self.materials.append(painted.material)
            geometry = painted.geometry_out
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
        wt = seconds * self.frequency * tau
        amp = self.amplitude * pi / np.sqrt(self.wavelength)
        total = np.zeros(len(points))
        for source in self.sources:
            radius = np.linalg.norm(points - np.array([source.x, source.y]),
                                    axis=1)
            x = np.maximum(radius, self.source_radius) * k
            total += amp * (j0(x) * np.cos(wt) + y0(x) * np.sin(wt))
        return total


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
        ``Mode``, and ``Scene Time -> Seconds``, which is why the drum moves
        without a keyframe.
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
        per mode, each computing its own u from ``Position``, and the switch.
        :math:`J_m` comes from :func:`~geometry_nodes.nodes.bessel_jm_rpn` -
        the ``j0``/``j1`` groups and the upward recurrence - so the tree says
        ``x,j0`` and ``2,x,/,j1,*,j0,-`` rather than carrying a table.
    :meth:`_geometry_frame`
        store, read back, lift, smooth, paint. The elongation is stored
        **before** the lift and read back through a ``Named Attribute`` for the
        same reason as in :class:`WaveVisualizationModifier`: a field is
        evaluated on the geometry the node receives, so an attribute stored
        after the lift would measure r on the *lifted* surface, where it is
        no longer the r the formula means.

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
        A palette name or a ``bpy.types.Material`` is set as it stands;
        ``None`` leaves the disc unpainted.
    :param colors: the three palette colours of that ramp, trough to crest.
    :param shade_smooth: smooth-shade the result.
    :param kwargs: passed on to the material (``emission``, ...) and to
        :class:`~geometry_nodes.geometry_nodes_modifier.GeometryNodesModifier`.
    """

    def __init__(self, name="DrumMode", radius=3.0, radial=100, angular=180,
                 modes=((0, 1), (1, 1), (2, 1), (0, 2), (3, 1), (1, 2)),
                 mode=0, amplitude=0.5, frequency=0.4, normalize=True,
                 attribute="elongation", material="elongation",
                 colors=("blue", "background", "important"),
                 shade_smooth=True, **kwargs):
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
        self.normalize = normalize
        self.attribute = attribute
        self.paint = material
        self.colors = colors
        self.shade_smooth = shade_smooth
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
        geometry = self._geometry_frame(tree, membrane, elongation)
        tree.links.new(geometry, self.group_outputs.inputs["Geometry"])

    # ------------------------------------------------------------------
    def _control_frame(self, tree):
        """The dials and the clock.

        ``Mode`` is an ``Integer`` node rather than a ``Value`` one because an
        ``Index Switch`` wants an integer and because that is what it is: there
        is no mode between (1,1) and (2,1) to interpolate through.

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
        return {"time": clock.std_out,
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
        return welded.geometry_out

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

        :return: the socket carrying the elongation of the selected mode.
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
            aux["u"] = elongation

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
            return sockets[0]

        switch = IndexSwitch(tree, location=(3, 0), data_type="FLOAT",
                             index=control["mode"], name="ModeSwitch",
                             parent=frame)
        for socket in sockets:
            switch.add_item(socket)
        return switch.std_out

    # ------------------------------------------------------------------
    def _geometry_frame(self, tree, membrane, elongation):
        """Store -> read back -> lift -> smooth -> paint.

        :return: the geometry socket for the group output.
        """
        frame = Frame(tree, location=(8, 0), label="Geometry",
                      name="GeometryFrame")
        stored = StoreNamedAttribute(tree, location=(0, 0), data_type="FLOAT",
                                     domain="POINT", name=self.attribute,
                                     value=elongation, parent=frame)
        tree.links.new(membrane, stored.geometry_in)

        height = NamedAttribute(tree, location=(1, -2), data_type="FLOAT",
                                name=self.attribute, parent=frame, hide=True)
        offset = CombineXYZ(tree, location=(2, -2), z=height.std_out,
                            name="Lift", parent=frame, hide=True)
        lifted = SetPosition(tree, location=(3, 0),
                             geometry=stored.geometry_out,
                             offset=offset.std_out, name="Displace",
                             parent=frame)
        geometry = lifted.geometry_out

        if self.shade_smooth:
            smooth = SetShadeSmooth(tree, location=(4, 0), geometry=geometry,
                                    name="Smooth", parent=frame)
            geometry = smooth.geometry_out

        if self.paint is not None:
            painted = SetMaterial(tree, location=(5, 0), geometry=geometry,
                                  material=self._texture(), name="Paint",
                                  parent=frame)
            self.material = painted.material
            self.materials.append(painted.material)
            geometry = painted.geometry_out
        return geometry

    # ------------------------------------------------------------------
    def _texture(self):
        """The material: the elongation as a colour, zero in the middle.

        The ramp is fed ``u/(2A) + 0.5``, so it runs from trough at 0 through
        the nodal value at 0.5 to crest at 1 - which puts the nodal diameters
        and circles on the surface as the one colour that does not move.
        """
        if not isinstance(self.paint, str) or self.paint != "elongation":
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
        :param seconds: the scene time the tree reads off the clock.
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
        wt = tau * self.frequency * alpha / BESSEL_ZEROS[0][0] * seconds
        angle = np.cos(m * np.arctan2(points[:, 1], points[:, 0])) if m else 1.0
        return self.amplitude / peak * jv(m, x) * angle * np.cos(wt)
