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

f must be normalised to [0, 1] — rejection sampling against a function that
exceeds 1 silently clips, which flattens the peaks of the distribution instead
of erroring.

Classes:

:class:`SpatialDistributionModifier`
    the machinery, with f = 1, which reproduces ``tmp.xml``.
:class:`InterferenceModifier`
    f = |sum of waves|^2, the point of the exercise: N point sources
    (spherical waves) or N beams (plane waves), with dials for wavelength,
    source positions and phases, so the fringes can be moved in the scene.
:class:`GaussianCloudModifier`
    f = a gaussian blob — a distribution with a known answer, which is what
    the verification harness leans on.
"""
import numpy as np

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (CubeMesh, DeleteGeometry, DistributePointsInVolume,
                                  Frame, IcoSphere, InputValue, InputVector,
                                  InstanceOnPoints, JoinGeometry, MeshToVolume, Points,
                                  Position, RandomValue, RealizeInstances, SetMaterial,
                                  SetPosition, StoredNamedAttribute, TransformGeometry,
                                  VolumeCube, WireFrame, make_function)
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
    :param method: ``"rejection"`` or ``"grid"``, as described in the module
        docstring. ``"grid"`` fills the box and only the box, so it cannot be
        combined with ``shape="input"``.
    :param count: how many points to end up with. Both methods honour it to
        within counting noise; ``shape="input"`` lands 13-14% under.
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
    :param box_color: palette name for a wireframe of the box, or ``None``
        for no box.
    :param box_radius: tube radius of that wireframe.
    """

    def __init__(self, size=4.0, center=(0, 0, 0), shape="cube",
                 method="rejection", count=20000, resolution=64,
                 voxel_amount=64.0, seed=0, radius=0.02, subdivisions=1,
                 color="drawing", color_by_density=False, gradient=None,
                 box_color=None, box_radius=0.01,
                 name="SpatialDistribution", **kwargs):
        if method not in ("rejection", "grid"):
            raise ValueError("method is 'rejection' or 'grid', not %r" % method)
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

        ``method="rejection"``, ``shape="cube"``
            ``count / <f>`` points placed by hand, each at a uniform random
            position in the box. Not the xml's route through a volume, and
            deliberately: ``Mesh to Volume`` ramps the density down over the
            ``Interior Band Width`` below the surface, which thins the outer
            ~5% of a box by 15-20% and costs 13% of the points overall.
            Nobody notices that on a blob, but on a *box* it is a visible
            soft edge where there should be a face.

        ``method="rejection"``, ``shape="input"``
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
        points when a scene asks to be able to colour by it.
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
        elif self.color_by_density:
            density = self.density(tree, position.std_out, location=(0, -1))
            if density is not None:
                density.node.parent = frame.node

        if density is not None:
            # carried on the points so that a material can read it; the value
            # is f at the point's own position, i.e. the local intensity
            store = StoredNamedAttribute(tree, location=(3, 0), name="intensity",
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

        if self.color_by_density:
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
