import math

import numpy as np

from appearance.textures import get_texture
from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier, SimpleFunctionModifier
from geometry_nodes.nodes import (CombineXYZ, CurveCircle, CurveLine, CurveToMesh,
                                  CurveToPoints, ExtrudeMesh, Frame, IcoSphere, InputBoolean,
                                  InputInteger, InputValue, InstanceOnPoints, JoinGeometry,
                                  MathNode, MeshToCurve, PointsToVertices, Position,
                                  RealizeInstances, ResampleCurve, SceneTime, SetMaterial,
                                  SetPosition, StoreNamedAttribute, Switch, TransformGeometry,
                                  create_geometry_line, make_function, split_rpn)
from interface import ibpy
from interface.ibpy import create_mesh, get_geometry_nodes_modifier, change_default_value, \
    get_geometry_node_from_modifier
from objects.bobject import BObject
from objects.coordinate_system import CoordinateSystem2
from utils.constants import DEFAULT_ANIMATION_TIME, OBJECT_APPEARANCE_TIME


class SimpleFunction(BObject):
    """Plot ``y = f(x)`` as a polyline mesh driven by a :class:`SimpleFunctionModifier`."""

    def __init__(self,function=lambda x:x,domain=[0,10],num_points=100,name="SimpleFunction",**kwargs):
        """Sample ``function`` over ``domain`` and build an edge-only polyline.

        The polyline is then dressed by a geometry-nodes modifier so the
        curve can grow over time (see :meth:`grow`).

        Args:
            function: Callable ``x -> z`` -- the function to plot.
                Points are placed at ``(x, 0, z)``.
            domain: ``[x_min, x_max]`` sample range. Defaults to ``[0, 10]``.
            num_points: Number of sample points. Defaults to 100.
            name: Object name. Defaults to ``'SimpleFunction'``.
            **kwargs: Forwarded to :class:`SimpleFunctionModifier` and
                :class:`BObject`.
        """
        vertices = []
        self.num_points = num_points

        for i in range(num_points):
            x = domain[0] + i*(domain[1]-domain[0])/num_points
            z = function(x)
            vertices.append((x,0,z))

        edges = []
        for i in range(num_points-1):
            edges.append([i,i+1])

        super().__init__(mesh=create_mesh(vertices=vertices,edges=edges),name=name,**kwargs)

        self.simple_function_modifier=SimpleFunctionModifier(**kwargs)
        self.add_mesh_modifier(type="NODES",node_modifier=self.simple_function_modifier)

    def grow(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        super().appear(begin_time=begin_time,transition_time=0)
        start_time_node = get_geometry_node_from_modifier(self.simple_function_modifier,label="StartTime")
        change_default_value(start_time_node,from_value=0,to_value=begin_time,begin_time=0,transition_time=0)
        transition_time_node = get_geometry_node_from_modifier(self.simple_function_modifier,label="TransitionTime")
        change_default_value(transition_time_node,from_value=0,to_value=transition_time,begin_time=0,transition_time=0)
        num_point_node = get_geometry_node_from_modifier(self.simple_function_modifier,label="NumPoints")
        change_default_value(num_point_node,from_value=100,to_value=self.num_points,begin_time=0,transition_time=0)
        return begin_time+transition_time


# ===========================================================================
#  GeoFunction -- a graph that is built entirely inside a geometry-nodes tree
# ===========================================================================
#
# The node setup below is the one authored in the editor and exported to
# ``video_interferences/tmp.xml``:
#
#     Value(xMin) ---> Combine XYZ --.
#     Value(xMax) ---> Combine XYZ ---> Curve Line -> Resample Curve -+
#     Integer(Resolution) ----------------------------^               |
#                                                                     v
#     Position -> (the math of the function) -> Combine XYZ -> Set Position
#                                                                     |
#     Value(Thickness) -> x 0.01 -+-> Curve Circle -> Curve to Mesh   (the curve)
#                                 |
#                                 +-> Ico Sphere -> Instance on Points
#                                 |                    -> Realize Instances (dots)
#                                 |
#                                 +-> x 0.25 -> Curve Circle ------------.
#                                                                        v
#     Resample Curve -> Curve to Points -> Points to Vertices -> Extrude Mesh
#                     -> Mesh to Curve -> Curve to Mesh              (the lines)
#
#                          Curve to Mesh ----> Switch (True)   ---.
#               Realize Instances + lines ----> Switch (False)     >- Set Material
#                     Boolean(ShowCurve) ----> Switch (Switch) --'         |
#                                                                    Group Output
#
# with two changes.  The chain of Math nodes that computed
# ``A sin(2pi x/lambda - 2pi t/T)`` is now a single
# :func:`~geometry_nodes.nodes.make_function` group built from a formula in
# reverse polish notation, so the tree says what it computes; and the
# ``Set Material`` node at the end is new -- the material is handed in by the
# constructor.


TAU = 2 * np.pi

# --- the same RPN, evaluated in python -------------------------------------
# Used to find out how tall a function is (see :func:`z_range_of`).  The
# operators are the scalar ones of :data:`geometry_nodes.nodes._SCALAR_MATH_OPS`
# and mean exactly what the ``ShaderNodeMath`` node of the same name means, so
# that the number computed here is the number the tree will show.

def _divide(a, b):
    """``ShaderNodeMath`` DIVIDE, zero included: x/0 is 0 there, not infinity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(a, b)
    return np.where(np.isfinite(result), result, 0.0)


_BINARY_OPS = {
    "*": lambda a, b: a * b,
    "/": _divide,
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "%": lambda a, b: np.fmod(a, b),
    "**": lambda a, b: np.power(a, b),
    "<": lambda a, b: (a < b) * 1.0,
    ">": lambda a, b: (a > b) * 1.0,
    "=": lambda a, b: (a == b) * 1.0,
    "min": np.minimum,
    "max": np.maximum,
    "atan2": np.arctan2,
}

_UNARY_OPS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "abs": np.abs, "sgn": np.sign,
    "sqrt": lambda a: np.sqrt(np.maximum(a, 0)),
    "lg": lambda a: np.log10(np.maximum(a, 1e-300)),
    "round": np.round, "floor": np.floor, "ceil": np.ceil,
    "frac": lambda a: a - np.floor(a),
}

_CONSTANTS = {"pi": np.pi, "-pi": -np.pi, "tau": TAU, "e": np.e}


class UnsupportedExpression(ValueError):
    """Raised when a formula contains a token the python evaluator cannot read.

    The tree is built from the formula in any case -- geometry nodes know
    operators (the Bessel groups, custom ops, ...) that this small evaluator
    does not.  It only means that the extent of the graph in z cannot be found
    by sampling and has to be given as ``zMin`` / ``zMax``.
    """


def evaluate_rpn(expression, variables):
    """Evaluate one reverse-polish formula with numpy.

    Args:
        expression: The formula, e.g. ``"tau,lambda,/,pos_x,*,sin,A,*"``.
        variables: Maps every symbol occurring in ``expression`` to a number
            or a numpy array.  Arrays broadcast, so an x of shape ``(n, 1)``
            and a t of shape ``(1, m)`` sample the whole (x, t) rectangle in
            one pass.

    Returns:
        The value of the expression, a scalar or an array.

    Raises:
        UnsupportedExpression: on an unknown token, or when the stack does not
            reduce to exactly one value (a malformed formula).
    """
    stack = []
    for token in split_rpn(expression):
        token = token.strip()
        if token == "":
            continue
        if token in _BINARY_OPS:
            if len(stack) < 2:
                raise UnsupportedExpression("not enough operands for '%s' in %r" % (token, expression))
            right = stack.pop()
            left = stack.pop()
            stack.append(_BINARY_OPS[token](left, right))
        elif token in _UNARY_OPS:
            if len(stack) < 1:
                raise UnsupportedExpression("not enough operands for '%s' in %r" % (token, expression))
            stack.append(_UNARY_OPS[token](stack.pop()))
        elif token in variables:
            stack.append(variables[token])
        elif token in _CONSTANTS:
            stack.append(_CONSTANTS[token])
        else:
            try:
                stack.append(float(token))
            except ValueError:
                raise UnsupportedExpression("cannot evaluate '%s' in %r" % (token, expression))
    if len(stack) != 1:
        raise UnsupportedExpression("%r does not reduce to a single value" % expression)
    return stack[0]


def _round_outwards(value, digits=2):
    """Round ``value`` away from zero to ``digits`` significant digits.

    ``0.99999`` (what a sampled sine actually reaches) becomes ``1.0`` rather
    than ``1.1``: the value is first cleaned to six significant digits, so
    sampling noise cannot push the bound up a whole step.
    """
    if value == 0 or not np.isfinite(value):
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    value = abs(value)
    value = float("%.6g" % value)
    magnitude = 10 ** (digits - 1 - math.floor(math.log10(value)))
    return sign * math.ceil(value * magnitude) / magnitude


def z_range_of(functions, parameters, symbols=None, resolution=500, t_max=10.0,
               t_samples=101, digits=2):
    """Find how far a set of graphs reaches in z, by sampling them.

    Every function is evaluated on ``resolution`` points across
    ``[xMin, xMax]``; if any parameter is the scene clock (value ``"time"``)
    the sampling is repeated at ``t_samples`` instants between 0 and
    ``t_max`` seconds, so that a travelling wave is measured over its whole
    excursion rather than at the one moment t = 0 -- where a sine is flat.

    A parameter may be spelled ``"symbol=name"`` -- the :class:`GeoFunction`
    way of saying "``lambda`` in the formulas, ``wavelength`` on the dial".
    Both halves are bound to the value here, so a formula that uses either one
    can be sampled; that is the whole reason this function knows about symbols
    at all.

    Args:
        functions: List of RPN formulas.
        parameters: The parameter dictionary handed to :class:`GeoFunction`,
            in either spelling of its keys.
        symbols: Extra (or overriding) short-symbol aliases,
            ``{parameter name: symbol}``.
        resolution: Sample points along x.
        t_max: Last instant of scene time that is sampled, in seconds.
        t_samples: Number of instants sampled in ``[0, t_max]``.
        digits: Significant digits the result is rounded outwards to.

    Returns:
        ``(zMin, zMax)``, rounded outwards so the graph fits inside with a
        little air, and never degenerate: a flat function gives ``(-1, 1)``.

    Raises:
        UnsupportedExpression: propagated from :func:`evaluate_rpn`.
    """


    symbols = dict(symbols) if symbols else {}

    # "A=amplitude" is one parameter under two names, and the formula is
    # written in whichever of them its author found readable, so both are
    # bound; the dictionary is keyed by the long name, as everywhere else.
    values_of = {}
    for key, value in parameters.items():
        if "=" in key:
            symbol, key = key.split("=", 1)
            symbols.setdefault(key, symbol)
        values_of[key] = value

    x_min = float(values_of.get("xMin", 0))
    x_max = float(values_of.get("xMax", 10))
    x = np.linspace(x_min, x_max, max(2, int(resolution))).reshape(-1, 1)

    has_time = any(isinstance(v, str) for v in values_of.values())
    t = np.linspace(0, t_max, max(1, int(t_samples))).reshape(1, -1) if has_time else np.zeros((1, 1))

    variables = {"pos_x": x, "pos_y": 0.0, "pos_z": 0.0, "tau": TAU}
    for key, value in values_of.items():
        entry = t if isinstance(value, str) else float(value)
        for name in {key, symbols.get(key, key)}:
            variables[name] = entry

    low, high = np.inf, -np.inf
    for expression in functions:
        values = np.asarray(evaluate_rpn(expression, variables), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        low = min(low, float(values.min()))
        high = max(high, float(values.max()))

    if not np.isfinite(low) or not np.isfinite(high):
        return -1.0, 1.0

    low, high = _round_outwards(low, digits), _round_outwards(high, digits)
    if low == high:
        return (low - 1.0, high + 1.0)
    return low, high

def _at(origin, column, row):
    """A node's place: so many columns right and rows down from a frame's origin."""
    return (origin[0] + column, origin[1] + row)


class FunctionModifier(GeometryNodesModifier):
    r"""The tree of ``video_interferences/tmp.xml``: graphs of f(x) as tubes.

    Four frames, one private method each, laid out left to right the way the
    xml has them:

    :meth:`_create_curve_frame` (``Curve``)
        the domain, and the two ways of drawing it.  A ``Curve Line`` from
        ``(xMin, 0, 0)`` to ``(xMax, 0, 0)``, resampled to ``Resolution``
        points and lifted by a ``Set Position``; from there either swept along
        a circle of radius ``Thickness * 0.01`` by ``Curve to Mesh`` -- the
        graph as a curve -- or handed to an ``Instance on Points`` that puts an
        ``Ico Sphere`` of that same radius on every sample point, which a
        ``Realize Instances`` turns into geometry -- the graph as individual
        data points.  Those dots are joined to the axis by a line each: the
        sample points of the *flat* line become loose vertices
        (``Curve to Points`` in ``EVALUATED`` mode, then
        ``Points to Vertices``), an ``Extrude Mesh`` in ``VERTICES`` mode
        pulls every one of them up by the same ``(0, 0, f)`` offset the graph
        is lifted by -- an edge from the axis to the data point -- and
        ``Mesh to Curve`` + ``Curve to Mesh`` give those edges a quarter of the
        tube's thickness.  A ``Switch`` driven by the ``ShowCurve`` boolean
        dial picks one of the two pictures (``True`` is the curve), so both are
        always built and the choice can be flipped mid-scene.  As data points
        the graph is usually wanted much more coarsely sampled than as a curve:
        the ``Resolution`` dial is what says how many dots there are.
    :meth:`_create_function_frame` (``Function``)
        the arithmetic.  One :func:`~geometry_nodes.nodes.make_function` group
        per formula, reading ``Position`` and the dials it mentions and
        returning **one float** -- the value of f at that point.  A
        ``Combine XYZ`` turns it into ``(0, 0, f)``, which goes into the
        ``Set Position``'s *Offset*: the points are already at ``(x, 0, 0)``,
        so nothing has to say where they are, only how far up.
    :meth:`_create_customization_frame` (``Customization``)
        what a material can read.  Every name in ``store_values`` becomes a
        named attribute on the tube: ``result`` is the value of the function
        at that point, anything else is the parameter of that name --
        ``amplitude``, say, so that a shader can measure ``result`` against
        the height the wave was actually given.
    :meth:`_create_final_transformation_frame` (``FinalTransformation``)
        the one scaling.  ``Transform Geometry`` with a scale of
        ``(xScale, 1, zScale)``, which is where the graph is fitted onto the
        axes of a coordinate system.  It sits *after* the function rather than
        inside it, so the formula is the formula and the plot's aspect ratio
        is a property of the picture, not of the mathematics.

    Every entry of ``parameters`` becomes a node of its own -- a ``Value`` node
    labelled with the parameter's name, or a ``Scene Time`` node when the value
    is the string ``"time"`` -- so each of them is a dial that can be animated
    later::

        dial = ibpy.get_geometry_node_from_modifier(modifier, "Wavelength")
        ibpy.change_default_value(dial, from_value=1, to_value=0.5,
                                  begin_time=1, transition_time=3)

    A formula may address a parameter by its name (``Wavelength``) or by its
    short symbol (``lambda``, see :data:`DEFAULT_SYMBOLS`), and only the
    parameters it actually mentions are wired into it.  ``tau`` is available as
    a constant node (as in the xml), ``pi`` is understood by
    :func:`~geometry_nodes.nodes.make_function` itself, and ``pos_x`` is the
    x coordinate of the point being lifted.

    Node locations are the ones from the xml, rounded to whole node units
    (``200`` px across, ``100`` px down), and the automatic layout is off - the
    graph is meant to be read in the editor in the arrangement it was authored
    in.  A second formula repeats the per-graph nodes two rows further down in
    each frame.

    :param name: Name of the node group and of the modifier in the stack.
    :param functions: List of RPN formulas, one graph each.
    :param parameters: ``{name: initial value}``; the value ``"time"`` asks for
        scene time in seconds, ``"frame"`` for the frame number.
    :param colors: One material per function.  A color name (or anything
        :func:`~appearance.textures.get_texture` accepts), a material, or a
        callable, which is called as ``color(attribute_names=[...], **kwargs)``
        and so can build a material out of the stored attributes.  A shorter
        list is padded with its last entry.
    :param curve: How the graph is drawn to begin with: ``True`` (the default)
        as a curve, ``False`` as one sphere per sample point.  Either way both
        branches are in the tree, and the ``ShowCurve`` dial switches between
        them -- see :meth:`GeoFunction.show_as_points`.
    :param point_subdivisions: Subdivisions of the ico sphere that is instanced
        on the sample points; 1 is a 42-vertex ball, 2 a smooth one.
    :param connect_points: Whether the data points are connected to the axis by
        a line each (a stem plot).  ``False`` leaves the dots on their own and
        the nodes that draw the lines out of the tree.
    :param line_thickness: How thick those lines are drawn, as a fraction of
        the tube's radius.
    :param resolution: Points the line is resampled to.  500 over ten units is
        two samples per 0.04 of a wavelength -- turn it up before turning the
        wavelength down.  It is also the number of data points, which is why a
        graph shown as points usually asks for a much smaller value.
    :param thickness: Tube thickness dial; the radius is ``thickness * 0.01``.
    :param x_scale: The x of the final ``Transform Geometry``.
    :param z_scale: The z of it.  Both are dials of their own.
    :param symbols: Extra short symbols, ``{parameter name: symbol}``.
    :param store_values: What to store on the tube as named attributes, in
        order.  ``"result"`` is the value of the function; every other entry
        names a parameter (by name, symbol or case) and stores its dial.
        Defaults to ``["result"]``.
    :param kwargs: Passed on to :func:`~appearance.textures.get_texture`
        (``emission``, ...) and to :class:`GeometryNodesModifier`.
    """

    #: Parameters that belong to the domain rather than to the function, and
    #: are therefore dialled in the ``Curve`` frame.
    CURVE_PARAMETERS = ("xMin", "xMax")

    #: Where each frame starts, in node units (200 px across, 100 px down),
    #: from the xml rounded to whole units.  A node's location is its frame's
    #: origin plus its own row and column: blender keeps a node where it is
    #: when it is put into a frame, so these are absolute, and two frames that
    #: shared a column would sit on top of each other.  The ``Function`` frame
    #: is the one whose height depends on what it holds, so it is worked out
    #: in the constructor.
    CURVE_ORIGIN = (0, 0)
    CUSTOMIZATION_ORIGIN = (12, 2)
    FINAL_ORIGIN = (16, 1)
    OUTPUT_LOCATION = (20, 1)

    def __init__(self, name="FunctionModifier", functions=None, parameters=None, colors=None,
                 resolution=500, thickness=1, x_scale=1, z_scale=1, symbols=None,
                 store_values=None, curve=True, point_subdivisions=1,
                 connect_points=True, line_thickness=0.25, **kwargs):
        self.functions = functions
        self.parameters = parameters
        self.parameters.setdefault("xMin", 0)
        self.parameters.setdefault("xMax", 5) # only sets value if no "xMax" exists
        if colors is None:
            colors = ["example"]
        elif not isinstance(colors, (list, tuple)):
            colors = [colors]
        self.colors = list(colors)
        while len(self.colors) < len(self.functions):
            self.colors.append(self.colors[-1])

        self.resolution = resolution
        self.thickness = thickness
        self.curve = curve
        self.point_subdivisions = point_subdivisions
        self.connect_points = connect_points
        self.line_thickness = line_thickness
        self.x_scale = x_scale
        self.z_scale = z_scale
        self.store_values = ["result"] if store_values is None else list(store_values)

        #: The tokens of every formula, one set per formula.
        self.tokens = [set(t.strip() for t in split_rpn(f)) for f in self.functions]
        self.parameter_nodes = {}
        self.dials = {}          # every dial, by the name it carries in the editor
        self.slots = {"curve": [], "function": [], "final": []}
        self.function_nodes = []
        self.symbols =symbols

        # the Function frame sits above the Curve frame and shares its columns,
        # so it has to start high enough that its lowest row clears it: one row
        # per dial, two per graph, and three rows of air over the topmost row
        # of the Curve frame, which is the ShowCurve dial's.
        depth = max(len([key for key in self.parameters if key not in self.CURVE_PARAMETERS])
                    + (1 if any("tau" in used for used in self.tokens) else 0),
                    2 * len(self.functions) - 1)
        self.function_origin = (0, 3 + depth)

        super().__init__(name, automatic_layout=False, group_input=False, **kwargs)


    def create_node(self, tree, **kwargs):
        self._create_dials(tree)
        functions = self._create_function_frame(tree)
        meshes = self._create_curve_frame(tree, functions)
        branches = self._create_customization_frame(tree, meshes, functions, **kwargs)
        geometry = self._create_final_transformation_frame(tree, branches)

        out = self.group_outputs
        out.location = (self.OUTPUT_LOCATION[0] * 200, self.OUTPUT_LOCATION[1] * 100)
        tree.links.new(geometry, out.inputs[0])

    # ------------------------------------------------------------------
    def _create_dials(self, tree):
        """Every ``Value``/``Scene Time``/``Integer`` node of the tree, placed.

        They are made in one go because a formula may mention any of them, but
        each is put where the frame that owns it will be: the domain and the
        tube in ``Curve``, the parameters of the function in ``Function``, the
        two scales in ``FinalTransformation``.  :attr:`slots` records which is
        which, and each frame method parents its own.
        """
        # the Curve frame: the domain, how finely it is sampled, how thick it
        # is drawn. Rows as in the xml, counting up from the bottom.
        curve = self.CURVE_ORIGIN
        for row, key in ((-4, "xMin"), (-3, "xMax")):
            self._add_dial(tree, key, self.parameters[key], location=_at(curve, 0, row),
                           slot="curve")
        self.dials["Resolution"] = InputInteger(tree, location=_at(curve, 0, -1),
                                                name="Resolution", integer=int(self.resolution))
        self.dials["Thickness"] = InputValue(tree, location=_at(curve, 0, 0),
                                             name="Thickness", value=self.thickness)
        # curve or data points: the one dial of the tree that is a boolean.  It
        # is not called "Curve" -- node lookup goes by substring, and the frame
        # is full of nodes whose names contain that word.
        self.dials["ShowCurve"] = InputBoolean(tree, location=_at(curve, 0, 1),
                                               name="ShowCurve", value=self.curve)
        self.slots["curve"] += [self.dials["Resolution"], self.dials["Thickness"],
                                self.dials["ShowCurve"]]

        # the Function frame: everything the formulas are written in terms of,
        # one row each under the Position node
        row = -1
        for key, value in self.parameters.items():
            if key in self.CURVE_PARAMETERS:
                continue
            self._add_dial(tree, key, value, location=_at(self.function_origin, 0, row),
                           slot="function")
            row -= 1
        if any("tau" in used for used in self.tokens):
            self.dials["tau"] = InputValue(tree, location=_at(self.function_origin, 0, row),
                                           name="tau", value=TAU)
            self.slots["function"].append(self.dials["tau"])

        # the FinalTransformation frame
        self.dials["xScale"] = InputValue(tree, location=_at(self.FINAL_ORIGIN, 0, -1),
                                          name="xScale", value=self.x_scale)
        self.dials["zScale"] = InputValue(tree, location=_at(self.FINAL_ORIGIN, 0, -2),
                                          name="zScale", value=self.z_scale)
        self.slots["final"] += [self.dials["xScale"], self.dials["zScale"]]

    def _add_dial(self, tree, key, value, location, slot):
        """One parameter: a ``Value`` node, or the scene clock when asked for."""
        if isinstance(value, str):
            if value not in ("time", "frame"):
                raise ValueError("parameter %r: %r is neither a number nor 'time'/'frame'"
                                 % (key, value))
            std_out = "Seconds" if value == "time" else "Frame"
            node = SceneTime(tree, location=location, name=key, std_out=std_out)
        else:
            node = InputValue(tree, location=location, name=key, value=value)
        self.dials[key] = self.parameter_nodes[key] = node
        self.slots[slot].append(node)
        return node

    # ------------------------------------------------------------------
    def _create_function_frame(self, tree):
        """``Function``: the formulas, each returning a single float.

        :return: one dict per formula with ``value`` -- the float socket, which
            the ``Customization`` frame stores as ``result`` -- and ``offset``,
            the same number as ``(0, 0, f)`` for the ``Set Position``.
        """
        links = tree.links
        origin = self.function_origin
        position = Position(tree, location=_at(origin, 0, 0))
        nodes = [position] + self.slots["function"]

        functions = []
        for i, expression in enumerate(self.functions):
            row = -1 - 2 * i
            used = self.tokens[i]

            names = ["pos"]
            scalars = ["result"]
            wiring = []
            for key, node in self.dials.items():
                if key in ("Resolution", "Thickness", "xScale", "zScale"):
                    continue
                # for name in sorted(self.names_of(key) if key in self.parameters else {key}):
                variable = self.symbols.get(key, None)
                if variable is None:
                    variable = key
                if variable in used and key not in names:
                    names.append(variable)
                    scalars.append(variable)
                    wiring.append((variable, node.std_out))

            group = make_function(tree, location=_at(origin, 1, row),
                                  name="Function_" + str(i),
                                  functions={"result": expression},
                                  inputs=names, outputs=["result"],
                                  vectors=["pos"], scalars=scalars)
            self.function_nodes.append(group)
            nodes.append(group)
            links.new(position.std_out, group.inputs["pos"])
            for name, socket in wiring:
                links.new(socket, group.inputs[name])

            # the points sit at (x, 0, 0) already, so the function is an
            # offset rather than a position - which is also why nothing here
            # has to take the curve apart with a Separate XYZ
            offset = CombineXYZ(tree, location=_at(origin, 2, row), z=group.outputs["result"],
                                name="Offset_" + str(i))
            nodes.append(offset)
            functions.append({"result": group.outputs["result"], "offset": offset.std_out})

        frame = Frame(tree, location=origin, label="Function")
        frame.add(nodes)
        return functions

    # ------------------------------------------------------------------
    def _create_curve_frame(self, tree, functions):
        """``Curve``: the sampled domain, lifted and drawn.

        Drawn twice, in fact: as the tube of ``Curve to Mesh``, and as one ico
        sphere per sample point with a line from the axis up to each of them,
        with a ``Switch`` on the ``ShowCurve`` dial choosing which of the two
        leaves the frame.  The spheres are realized rather than left as
        instances, so that the ``Store Named Attribute`` nodes further along
        write on their points the way they do on the tube's.

        The lines are extruded vertices rather than a second lifted curve: the
        sample points of the flat line are turned into loose vertices and each
        is pulled up by ``(0, 0, f)``, the very offset the graph itself is
        lifted by, so the line ends exactly where its data point sits however
        the parameters are dialled.

        :param functions: what :meth:`_create_function_frame` returned.
        :return: one geometry socket per formula -- whichever of the two the
            switch is showing.
        """
        origin = self.CURVE_ORIGIN
        start = CombineXYZ(tree, location=_at(origin, 1, -4), name="Start",
                           x=self.dials["xMin"].std_out)
        end = CombineXYZ(tree, location=_at(origin, 1, -3), name="End",
                         x=self.dials["xMax"].std_out)
        line = CurveLine(tree, location=_at(origin, 2, -3), mode="POINTS",
                         start=start.std_out, end=end.std_out)
        resampled = ResampleCurve(tree, location=_at(origin, 3, -3), curve=line.geometry_out,
                                  count=self.dials["Resolution"].std_out)

        radius = MathNode(tree, location=_at(origin, 3, 0), operation="MULTIPLY",
                          inputs0=self.dials["Thickness"].std_out, inputs1=0.01,
                          name="TubeRadius")
        profile = CurveCircle(tree, location=_at(origin, 4, -1), resolution=8,
                              radius=radius.std_out)
        # the dot of the data-point branch is as thick as the tube is, so that
        # the one dial keeps saying how heavy the graph is drawn
        dot = IcoSphere(tree, location=_at(origin, 4, 1), name="DataPoint",
                        radius=radius.std_out, subdivisions=self.point_subdivisions)

        nodes = self.slots["curve"] + [start, end, line, resampled, radius, profile, dot]

        if self.connect_points:
            # the sample points of the flat line, as a mesh of loose vertices:
            # what the lines are extruded out of.  EVALUATED is the mode that
            # takes the points the Resample Curve made, so that there is one
            # line per data point and no dial of its own to keep in step.
            sample_points = CurveToPoints(tree, location=_at(origin, 5, 1), mode="EVALUATED",
                                          curve=resampled.geometry_out, name="SamplePoints")
            vertices = PointsToVertices(tree, location=_at(origin, 6, 1),
                                        points=sample_points.geometry_out)
            line_radius = MathNode(tree, location=_at(origin, 8, 1), operation="MULTIPLY",
                                   inputs0=radius.std_out, inputs1=self.line_thickness,
                                   name="LineRadius")
            line_profile = CurveCircle(tree, location=_at(origin, 9, 1), resolution=8,
                                       radius=line_radius.std_out, name="LineProfile")
            nodes += [sample_points, vertices, line_radius, line_profile]
        meshes = []
        for i, function in enumerate(functions):
            row = -2 * i
            lifted = SetPosition(tree, location=_at(origin, 4, -2 + row),
                                 offset=function["offset"])
            mesh = CurveToMesh(tree, location=_at(origin, 5, -1 + row),
                               profile_curve=profile.geometry_out, fill_caps=True)
            create_geometry_line(tree, [lifted, mesh], ins=resampled.geometry_out)

            dots = InstanceOnPoints(tree, location=_at(origin, 5, -2 + row), hide=True,
                                    name="DataPoints_" + str(i),
                                    points=lifted.geometry_out, instance=dot.geometry_out)
            realized = RealizeInstances(tree, location=_at(origin, 6, -2 + row),
                                        geometry=dots.geometry_out)

            points = realized.geometry_out
            if self.connect_points:
                # VERTICES mode: every vertex is pulled out on its own and
                # leaves one edge behind it, which is the line we are after
                lines = ExtrudeMesh(tree, location=_at(origin, 7, -2 + row), mode="VERTICES",
                                    mesh=vertices.geometry_out, offset=function["offset"],
                                    name="Lines_" + str(i))
                edges = MeshToCurve(tree, location=_at(origin, 8, -2 + row),
                                    mesh=lines.geometry_out)
                rods = CurveToMesh(tree, location=_at(origin, 9, -2 + row),
                                   curve=edges.geometry_out,
                                   profile_curve=line_profile.geometry_out, fill_caps=False)
                join = JoinGeometry(tree, location=_at(origin, 10, -2 + row),
                                    geometry=[realized.geometry_out, rods.geometry_out])
                points = join.geometry_out
                nodes += [lines, edges, rods, join]

            display = Switch(tree, location=_at(origin, 11, -1 + row), input_type="GEOMETRY",
                             name="Display_" + str(i),
                             switch=self.dials["ShowCurve"].std_out,
                             true=mesh.geometry_out, false=points)

            nodes += [lifted, mesh, dots, realized, display]
            meshes.append(display.geometry_out)

        frame = Frame(tree, location=origin, label="Curve")
        frame.add(nodes)
        return meshes

    # ------------------------------------------------------------------
    def _create_customization_frame(self, tree, meshes, functions, **kwargs):
        """``Customization``: what the material is given, and the material.

        ``result`` -- the value of the function -- is stored *before* the
        material, so that a shader reading it sees the number that belongs to
        the point it is shading.  The parameters after it are constants per
        graph, which is their point: they say what the values in ``result``
        are to be measured against.

        :return: one geometry socket per formula.
        """
        keys = [None if name == "result" else name
                for name in self.store_values]
        for name, key in zip(self.store_values, keys):
            if key is None and name != "result":
                raise ValueError("store_values: no parameter called %r, only %s and 'result'"
                                 % (name, sorted(self.parameters)))
        # the material goes in where the xml has it: after the value of the
        # function has been stored, before the constants that describe it
        cut = self.store_values.index("result") + 1 if "result" in self.store_values else 0

        nodes = []
        branches = []
        for i, (mesh, function, color) in enumerate(zip(meshes, functions, self.colors)):
            row = -2 * i
            stores = []
            for column, (name, key) in enumerate(zip(self.store_values, keys)):
                result = function["result"] if key is None else self.dials[key].std_out
                # the material takes the column at `cut`, the stores behind it
                # move one along
                stores.append(StoreNamedAttribute(
                    tree, name=name, value=result,
                    location=_at(self.CUSTOMIZATION_ORIGIN, column + (column >= cut), row)))

            if callable(color):
                material = color(attribute_names=list(self.store_values), **kwargs)
            else:
                material = get_texture(color, **kwargs)
            painted = SetMaterial(tree, location=_at(self.CUSTOMIZATION_ORIGIN, cut, row),
                                  material=material, name="Material_" + str(i))
            self.materials.append(material)

            chain = stores[:cut] + [painted] + stores[cut:]
            create_geometry_line(tree, chain, ins=mesh)
            nodes += chain
            branches.append(chain[-1].geometry_out)

        frame = Frame(tree, location=self.CUSTOMIZATION_ORIGIN, label="Customization")
        frame.add(nodes)
        return branches

    # ------------------------------------------------------------------
    def _create_final_transformation_frame(self, tree, branches):
        """``FinalTransformation``: the graph fitted onto its axes.

        The scale is ``(xScale, 1, zScale)`` -- y is left alone, or the tube
        would be flattened out of existence.

        :return: the geometry socket for the group output.
        """
        nodes = list(self.slots["final"])

        if len(branches) == 1:
            geometry = branches[0]
        else:
            join = JoinGeometry(tree, location=_at(self.FINAL_ORIGIN, 1, 0), geometry=branches)
            nodes.append(join)
            geometry = join.geometry_out

        scale = CombineXYZ(tree, location=_at(self.FINAL_ORIGIN, 1, -1), name="Scale",
                           x=self.dials["xScale"].std_out, y=1,
                           z=self.dials["zScale"].std_out)
        transform = TransformGeometry(tree, location=_at(self.FINAL_ORIGIN, 2, 0),
                                      geometry=geometry, scale=scale.std_out)
        nodes += [scale, transform]

        frame = Frame(tree, location=self.FINAL_ORIGIN, label="FinalTransformation")
        frame.add(nodes)
        return transform.geometry_out


class GeoFunction(BObject):
    r"""Graphs of one or more functions, drawn by a :class:`FunctionModifier`.

    Unlike :class:`SimpleFunction`, which samples a python callable into a mesh
    once and for all, nothing here is baked: the functions live in the node
    tree as formulas, and their parameters are ``Value`` nodes that can be
    animated afterwards.  A function of the scene clock therefore moves without
    a keyframe, and a swept ``Wavelength`` is one call.

    The default arguments are the sine wave of ``video_interferences/tmp.xml``::

        wave = GeoFunction(parameters={"xMin": 0, "xMax": 10, "Amplitude": 1,
                                       "Wavelength": 1, "Period": 1, "time": "time"},
                           functions=["tau,lambda,/,pos_x,*,tau,T,/,t,*,-,sin,A,*"],
                           colors=["example"], coord=True)
        wave.appear(begin_time=0, transition_time=1)
        wave.change_parameter("Wavelength", from_value=1, to_value=2,
                              begin_time=2, transition_time=4)

    **The extent in z** is found by sampling the formulas over the x domain and
    -- when one of the parameters is the clock -- over ``t_max`` seconds of it
    (:func:`z_range_of`), so the second axis of the coordinate system fits the
    graph without being told.  Pass ``zMin`` / ``zMax`` to fix it by hand
    instead; that is also the way out when a formula uses an operator the
    python evaluator does not know (a Bessel group, a custom op), in which case
    the automatic range falls back to ``[-1, 1]`` with a warning.

    **With** ``coord=True`` a :class:`~objects.coordinate_system.CoordinateSystem2`
    spanning ``[xMin, xMax] x [zMin, zMax]`` is built and parented to the graph,
    so moving the one moves the other.  The graph is fitted onto its axes by
    the modifier's ``xScale`` and ``zScale`` dials, which scale the finished
    tube rather than the function.  By default the x axis is 7 units long and z
    is scaled by the *same* factor, which keeps the picture undistorted; give
    ``lengths=[lx, lz]`` to stretch either axis on purpose.

    :param parameters: ``{name: initial value}``.  The value ``"time"`` asks
        for a ``Scene Time`` node instead of a ``Value`` node.  ``xMin`` and
        ``xMax`` are the ends of the domain and default to 0 and 10.
    :param functions: List of RPN formulas of x (``pos_x``) and the parameters.
    :param colors: One material per function.
    :param coord: Whether to build the coordinate system.
    :param name: Object name.
    :param zMin: Lower end of the z axis; inferred when ``None``.
    :param zMax: Upper end of the z axis; inferred when ``None``.
    :param lengths: ``[lx, lz]``, the lengths of the two axes in world units.
        Defaults to ``[7, 7 * (zMax - zMin) / (xMax - xMin)]``.
    :param resolution: Points the graph is sampled at (both in the tree and
        when the z range is inferred).  As data points this is the number of
        dots, so a graph meant to be shown that way wants a small value.
    :param thickness: Tube thickness dial of the graph; the data points are
        spheres of the same radius.
    :param curve: How the graph is drawn to begin with -- ``True`` as a curve,
        ``False`` as individual data points.  Both are in the tree either way,
        and :meth:`show_as_points` / :meth:`show_as_curve` switch between them
        at any time.
    :param point_subdivisions: Subdivisions of the sphere sitting on each data
        point.
    :param connect_points: Whether each data point is joined to the axis by a
        line, which makes the point picture a stem plot rather than a scatter
        of dots.
    :param line_thickness: Thickness of those lines, as a fraction of the
        thickness of the graph.
    :param t_max: Seconds of scene time sampled when inferring the z range.
    :param symbols: Extra short symbols, ``{parameter name: symbol}``.
    :param store_values: Named attributes the graph carries for its material to
        read -- ``"result"`` is the value of the function, any other entry
        names a parameter.  See :class:`FunctionModifier`.
    :param coord_kwargs: Passed on to
        :class:`~objects.coordinate_system.CoordinateSystem2` (``n_tics``,
        ``axes_labels``, ``colors``, ``radii``, ...).
    :param kwargs: Passed on to :class:`FunctionModifier` (and thereby to
        :func:`~appearance.textures.get_texture`) and to :class:`BObject`.
    """

    def __init__(self, parameters=None, functions=None, colors=None, coord=True,
                 name="GeoFunction", zMin=None, zMax=None, lengths=None,
                 resolution=500, thickness=1, t_max=10.0,
                 store_values=["result"], curve=True, point_subdivisions=1,
                 connect_points=True, line_thickness=0.25,
                 coord_kwargs=None, **kwargs):
        self.parameters = dict(parameters)
        self.parameters.setdefault("xMin", 0)
        self.parameters.setdefault("xMax", 10)
        self.functions = list(functions)
        if colors is None:
            colors = ["example"]
        elif not isinstance(colors, (list, tuple)):
            colors = [colors]

        self.x_min = float(self.parameters["xMin"])
        self.x_max = float(self.parameters["xMax"])
        symbols =dict()

        self.extracted_parameters = dict()
        for key,value in self.parameters.items():
            if "=" in key:
                symbol,variable = key.split("=")
                symbols[variable] = symbol
                self.extracted_parameters[variable]=value
            else:
                self.extracted_parameters[key] = value

        # --- how tall is it? ------------------------------------------------
        if zMin is None or zMax is None:
            try:
                inferred = z_range_of(self.functions, self.extracted_parameters,
                                      symbols=symbols, resolution=resolution, t_max=t_max)
            except UnsupportedExpression as error:
                print("GeoFunction: " + str(error) + " -- falling back to z in [-1,1], "
                      "pass zMin/zMax to set the range explicitly.")
                inferred = (-1.0, 1.0)
            zMin = inferred[0] if zMin is None else zMin
            zMax = inferred[1] if zMax is None else zMax
        self.z_min = float(zMin)
        self.z_max = float(zMax)

        # --- the map from coordinates to the world --------------------------
        if coord:
            if lengths is None:
                x_length = 7
                z_length = x_length * (self.z_max - self.z_min) / (self.x_max - self.x_min)
            else:
                x_length, z_length = lengths[0], lengths[1]
            self.lengths = [x_length, z_length]
            self.x_scale = x_length / (self.x_max - self.x_min)
            self.z_scale = z_length / (self.z_max - self.z_min)
        else:
            self.lengths = [self.x_max - self.x_min, self.z_max - self.z_min]
            self.x_scale = 1
            self.z_scale = 1

        self.coordinate_system = None
        if coord:
            coord_kwargs = dict(coord_kwargs) if coord_kwargs else {}
            coord_kwargs.setdefault("name", name + "CoordinateSystem")
            coord_kwargs.setdefault("n_tics", [int(self.x_max - self.x_min), 2])
            self.coordinate_system = CoordinateSystem2(dimension=2,
                                                       domains=[[self.x_min, self.x_max],
                                                                [self.z_min, self.z_max]],
                                                       lengths=self.lengths,
                                                       **coord_kwargs)

        children = [self.coordinate_system] if self.coordinate_system else []
        super().__init__(mesh=create_mesh(vertices=[(0, 0, 0)]), name=name,
                         no_material=True, children=children, **kwargs)

        self.modifier = FunctionModifier(name=name + "Modifier", functions=self.functions,
                                         parameters=self.extracted_parameters, colors=colors,
                                         resolution=resolution, thickness=thickness,
                                         x_scale=self.x_scale, z_scale=self.z_scale,
                                         store_values=store_values, curve=curve,
                                         point_subdivisions=point_subdivisions,
                                         connect_points=connect_points,
                                         line_thickness=line_thickness,
                                         symbols=symbols, **kwargs)
        self.add_mesh_modifier(type="NODES", node_modifier=self.modifier)

    # ------------------------------------------------------------------
    def get_dial(self, label):
        """The node of a parameter, ready for :func:`ibpy.change_default_value`.

        Works for every entry of ``parameters`` and for ``Resolution``,
        ``Thickness``, ``xScale``, ``zScale`` and ``tau``.
        """
        return ibpy.get_geometry_node_from_modifier(self.modifier, label)

    def show_as_points(self, begin_time=0):
        """Draw the graph as individual data points from ``begin_time`` on.

        Each of them stands on a line down to the axis, unless the graph was
        built with ``connect_points=False``.

        The dots are the sample points of the graph, so how many there are is
        the ``Resolution`` dial -- 500 of them is a solid line of spheres, and
        a graph that is meant to be read as data usually asks for far fewer,
        either from the start (``resolution=...``) or by dialling ``Resolution``
        down before the switch.
        """
        return self._show(False, begin_time)

    def show_as_curve(self, begin_time=0):
        """Draw the graph as a curve from ``begin_time`` on."""
        return self._show(True, begin_time)

    def _show(self, curve, begin_time):
        """Flip the ``ShowCurve`` dial: one keyframe before, one after."""
        node = self.get_dial("ShowCurve")
        if node is None:
            raise KeyError("no dial called 'ShowCurve' in %s" % self.modifier.tree.name)
        return ibpy.change_default_boolean(node, from_value=not curve, to_value=curve,
                                           begin_time=begin_time)

    def change_parameter(self, label, from_value=None, to_value=None, begin_time=0,
                         transition_time=DEFAULT_ANIMATION_TIME):
        """Ramp one of the dials, and hence redraw the graph while it runs."""
        node = self.get_dial(label)
        if node is None:
            raise KeyError("no dial called %r in %s" % (label, self.modifier.tree.name))
        if label in ["Resolution"]: # integers
            ibpy.change_default_integer(node,from_value=from_value,to_value=to_value,begin_time=begin_time,transition_time=transition_time)
        elif label in ["ShowCurve"]: #booleans
            ibpy.change_default_boolean(node,from_value=from_value,to_value=to_value,begin_time=begin_time)
        else: # floats
            ibpy.change_default_value(node, from_value=from_value, to_value=to_value,
                                      begin_time=begin_time, transition_time=transition_time)
        return begin_time + transition_time

    def appear(self, alpha=1, begin_time=0, transition_time=OBJECT_APPEARANCE_TIME, **kwargs):
        """Fade the graph in, and grow the axes of the coordinate system with it."""
        kwargs.pop("children", None)
        if self.coordinate_system is not None:
            self.coordinate_system.appear(begin_time=begin_time,
                                          transition_time=transition_time, alpha=alpha)
        return super().appear(alpha=alpha, begin_time=begin_time,
                              transition_time=transition_time, children=False, **kwargs)
