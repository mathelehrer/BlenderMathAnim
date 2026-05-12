"""
Complex-number arithmetic node groups for Blender geometry nodes and shader nodes.

Encoding convention (VECTOR socket):
    X = Re    (real part)
    Y = Im    (imaginary part)
    Z = |z|   (absolute value = sqrt(Re² + Im²))

The fourth component the user asked for (arg = atan2(Im, Re)) is always
freshly computed inside each node and exposed as a separate FLOAT output
socket.  It is NOT stored in the wire because Blender's ROTATION socket
normalises its quaternion on write, which would corrupt arbitrary (Re, Im,
|z|, arg) data.  A VECTOR socket carries the three independent values
without any normalisation constraint.

Public API
----------
RotationToQuaternion           – utility: separate ROTATION → W, X, Y, Z floats
ComplexInput(re, im)           – pack (Re, Im) → VECTOR(Re, Im, |z|)
ComplexMathNode(method=…)      – geometry-node-tree complex arithmetic group (one per op)
make_complex_shader_group      – shader-node-tree version (for materials)
make_mandelbrot_material       – plane Mandelbrot shader built from the above
make_complex_menu_node_group   – single node with a dropdown menu for all 10 operations

Supported methods
-----------------
Binary : ADD, SUBTRACT, MULTIPLY, DIVIDE
Unary  : CONJUGATE, LENGTH, ANGLE, EXPONENTIATE, SINE, COSINE
"""

import bpy

from geometry_nodes.nodes import NodeGroup, BlueNode
from interface.ibpy import make_new_socket
from utils.kwargs import get_from_kwargs

# Socket type strings for binary/unary dispatch
_BINARY_METHODS = frozenset({"ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"})
_UNARY_METHODS  = frozenset(
    {"CONJUGATE", "LENGTH", "ANGLE", "EXPONENTIATE", "SINE", "COSINE"}
)


# ---------------------------------------------------------------------------
# Utility: unpack a ROTATION socket into its (W, X, Y, Z) quaternion floats
# ---------------------------------------------------------------------------

class RotationToQuaternion(BlueNode):
    """
    Thin wrapper around FunctionNodeRotationToQuaternion.

    Outputs: .w, .x, .y, .z  (NodeSocketFloat)
    """

    def __init__(self, tree, location=(0, 0), rotation=None, **kwargs):
        self.node = tree.nodes.new(type="FunctionNodeRotationToQuaternion")
        super().__init__(tree, location=location, **kwargs)

        self.w = self.node.outputs["W"]
        self.x = self.node.outputs["X"]
        self.y = self.node.outputs["Y"]
        self.z = self.node.outputs["Z"]

        if rotation is not None:
            tree.links.new(rotation, self.node.inputs["Rotation"])


# ---------------------------------------------------------------------------
# Core graph builder (tree-type agnostic)
# ---------------------------------------------------------------------------

def _build_complex_graph(nodes, links, gi, go, method):
    """
    Fill any node tree (GeometryNodeTree or ShaderNodeTree) with the
    complex-math computation for *method*.

    Parameters
    ----------
    nodes  : tree.nodes of the inner (group) tree
    links  : tree.links of the inner (group) tree
    gi     : NodeGroupInput node inside the inner tree
    go     : NodeGroupOutput node inside the inner tree
    method : operation string (ADD, MULTIPLY, SINE, …)
    """

    def math_op(op, *args):
        n = nodes.new(type="ShaderNodeMath")
        n.operation = op
        n.hide = True
        for i, a in enumerate(args):
            if isinstance(a, (int, float)):
                n.inputs[i].default_value = float(a)
            else:
                links.new(a, n.inputs[i])
        return n.outputs["Value"]

    def sep_xyz(vec_socket):
        sep = nodes.new(type="ShaderNodeSeparateXYZ")
        sep.hide = True
        links.new(vec_socket, sep.inputs["Vector"])
        return sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]

    def comb_xyz(x, y, z):
        c = nodes.new(type="ShaderNodeCombineXYZ")
        c.hide = True
        for val, slot in [(x, "X"), (y, "Y"), (z, "Z")]:
            if isinstance(val, (int, float)):
                c.inputs[slot].default_value = float(val)
            else:
                links.new(val, c.inputs[slot])
        return c.outputs["Vector"]

    def abs_of(re, im):
        return math_op(
            "SQRT",
            math_op("ADD",
                    math_op("MULTIPLY", re, re),
                    math_op("MULTIPLY", im, im)),
        )

    def arg_of(re, im):
        return math_op("ARCTAN2", im, re)

    # --- dispatch --------------------------------------------------------
    if method in _BINARY_METHODS:
        re1, im1, ab1 = sep_xyz(gi.outputs["z1"])
        re2, im2, ab2 = sep_xyz(gi.outputs["z2"])

        if method == "ADD":
            re_out = math_op("ADD", re1, re2)
            im_out = math_op("ADD", im1, im2)
            ab_out = abs_of(re_out, im_out)

        elif method == "SUBTRACT":
            re_out = math_op("SUBTRACT", re1, re2)
            im_out = math_op("SUBTRACT", im1, im2)
            ab_out = abs_of(re_out, im_out)

        elif method == "MULTIPLY":
            # z1*z2 = (Re1*Re2 − Im1*Im2) + i(Re1*Im2 + Im1*Re2)
            # |z1*z2| = |z1|*|z2|  (avoids recomputation)
            re_out = math_op("SUBTRACT",
                             math_op("MULTIPLY", re1, re2),
                             math_op("MULTIPLY", im1, im2))
            im_out = math_op("ADD",
                             math_op("MULTIPLY", re1, im2),
                             math_op("MULTIPLY", im1, re2))
            ab_out = math_op("MULTIPLY", ab1, ab2)

        elif method == "DIVIDE":
            # z1/z2 using |z2|² as denominator; |z1/z2| = |z1|/|z2|
            denom = math_op("ADD",
                            math_op("MULTIPLY", re2, re2),
                            math_op("MULTIPLY", im2, im2))
            re_out = math_op("DIVIDE",
                             math_op("ADD",
                                     math_op("MULTIPLY", re1, re2),
                                     math_op("MULTIPLY", im1, im2)),
                             denom)
            im_out = math_op("DIVIDE",
                             math_op("SUBTRACT",
                                     math_op("MULTIPLY", im1, re2),
                                     math_op("MULTIPLY", re1, im2)),
                             denom)
            ab_out = math_op("DIVIDE", ab1, ab2)

        ar_out = arg_of(re_out, im_out)

    else:  # unary
        re1, im1, ab1 = sep_xyz(gi.outputs["z"])

        if method == "CONJUGATE":
            re_out = re1
            im_out = math_op("MULTIPLY", im1, -1.0)
            ab_out = ab1                       # |conj(z)| = |z|

        elif method == "LENGTH":
            ab = abs_of(re1, im1)
            re_out = ab
            ab_out = ab
            im_out = 0.0                       # result is purely real

        elif method == "ANGLE":
            arg = arg_of(re1, im1)
            re_out = arg
            ab_out = math_op("ABSOLUTE", arg)
            im_out = 0.0                       # result is purely real

        elif method == "EXPONENTIATE":
            # e^z = e^Re * (cos Im + i sin Im);  |e^z| = e^Re
            exp_re = math_op("EXPONENT", re1)
            re_out = math_op("MULTIPLY", exp_re, math_op("COSINE", im1))
            im_out = math_op("MULTIPLY", exp_re, math_op("SINE",   im1))
            ab_out = exp_re

        elif method == "SINE":
            # sin z = sin(Re)*cosh(Im) + i*cos(Re)*sinh(Im)
            re_out = math_op("MULTIPLY", math_op("SINE",   re1), math_op("COSH", im1))
            im_out = math_op("MULTIPLY", math_op("COSINE", re1), math_op("SINH", im1))
            ab_out = abs_of(re_out, im_out)

        elif method == "COSINE":
            # cos z = cos(Re)*cosh(Im) − i*sin(Re)*sinh(Im)
            re_out = math_op("MULTIPLY", math_op("COSINE", re1), math_op("COSH", im1))
            im_out = math_op("MULTIPLY", -1.0,
                             math_op("MULTIPLY",
                                     math_op("SINE", re1),
                                     math_op("SINH", im1)))
            ab_out = abs_of(re_out, im_out)

        ar_out = arg_of(re_out, im_out)

    # --- wire group outputs ----------------------------------------------
    result_vec = comb_xyz(re_out, im_out, ab_out)
    links.new(result_vec, go.inputs["result"])
    links.new(ar_out,     go.inputs["arg"])


# ---------------------------------------------------------------------------
# Helper: create a complex-number VECTOR from Re and Im
# ---------------------------------------------------------------------------

class ComplexInput(NodeGroup):
    """
    Packs two real floats (Re, Im) into VECTOR(Re, Im, sqrt(Re²+Im²)).

    Inputs : Re (FLOAT), Im (FLOAT)
    Outputs: z  (VECTOR)
    """

    def __init__(self, tree, re=0.0, im=0.0, **kwargs):
        name = get_from_kwargs(kwargs, "name", "ComplexInput")

        super().__init__(
            tree,
            inputs={"Re": "FLOAT", "Im": "FLOAT"},
            outputs={"z": "VECTOR"},
            auto_layout=True,
            name=name,
            **kwargs,
        )

        self.inputs  = self.node.inputs
        self.outputs = self.node.outputs
        self.std_out = self.node.outputs["z"]

        _connect(tree, re, self.node.inputs["Re"])
        _connect(tree, im, self.node.inputs["Im"])

    def fill_group_with_node(self, tree, **kwargs):
        links = tree.links
        nodes = tree.nodes
        gi = self.group_inputs
        go = self.group_outputs

        def math_op(op, *args):
            n = nodes.new(type="ShaderNodeMath")
            n.operation = op
            n.hide = True
            for i, a in enumerate(args):
                if isinstance(a, (int, float)):
                    n.inputs[i].default_value = float(a)
                else:
                    links.new(a, n.inputs[i])
            return n.outputs["Value"]

        re = gi.outputs["Re"]
        im = gi.outputs["Im"]
        ab = math_op("SQRT",
                     math_op("ADD",
                             math_op("MULTIPLY", re, re),
                             math_op("MULTIPLY", im, im)))

        comb = nodes.new(type="ShaderNodeCombineXYZ")
        comb.hide = True
        links.new(re, comb.inputs["X"])
        links.new(im, comb.inputs["Y"])
        links.new(ab, comb.inputs["Z"])
        links.new(comb.outputs["Vector"], go.inputs["z"])


# ---------------------------------------------------------------------------
# Main class: ComplexMathNode  (geometry-node tree)
# ---------------------------------------------------------------------------

class ComplexMathNode(NodeGroup):
    """
    Complex-number arithmetic node group for geometry node trees.

    Wire format (VECTOR):  X = Re,  Y = Im,  Z = |z|

    Parameters
    ----------
    tree      : geometry node tree
    method    : ADD | SUBTRACT | MULTIPLY | DIVIDE |
                CONJUGATE | LENGTH | ANGLE | EXPONENTIATE | SINE | COSINE
    inputs0   : socket or (Re, Im, |z|) tuple for first operand
    inputs1   : socket or (Re, Im, |z|) tuple for second operand (binary only)

    Outputs
    -------
    result  VECTOR  – complex result as (Re, Im, |z|)
    arg     FLOAT   – atan2(Im_out, Re_out)
    """

    BINARY_METHODS = _BINARY_METHODS
    UNARY_METHODS  = _UNARY_METHODS

    def __init__(self, tree, method="ADD", inputs0=None, inputs1=None, **kwargs):
        if method not in _BINARY_METHODS | _UNARY_METHODS:
            raise ValueError(
                f"Unknown ComplexMathNode method '{method}'. "
                f"Choose from: {sorted(_BINARY_METHODS | _UNARY_METHODS)}"
            )

        self.method = method
        name = get_from_kwargs(kwargs, "name", f"ComplexMath_{method}")

        ins = {"z1": "VECTOR", "z2": "VECTOR"} if method in _BINARY_METHODS else {"z": "VECTOR"}

        super().__init__(
            tree,
            inputs=ins,
            outputs={"result": "VECTOR", "arg": "FLOAT"},
            auto_layout=True,
            name=name,
            **kwargs,
        )

        self.inputs  = self.node.inputs
        self.outputs = self.node.outputs
        self.std_out = self.node.outputs["result"]
        self.arg_out = self.node.outputs["arg"]

        if method in _BINARY_METHODS:
            if inputs0 is not None:
                _connect(tree, inputs0, self.node.inputs["z1"])
            if inputs1 is not None:
                _connect(tree, inputs1, self.node.inputs["z2"])
        else:
            if inputs0 is not None:
                _connect(tree, inputs0, self.node.inputs["z"])

    def fill_group_with_node(self, tree, **kwargs):
        _build_complex_graph(
            tree.nodes, tree.links,
            self.group_inputs, self.group_outputs,
            self.method,
        )


# ---------------------------------------------------------------------------
# Shader-tree version: one complex-op as a ShaderNodeGroup
# ---------------------------------------------------------------------------

def make_complex_shader_group(parent_nodes, method, name=None):
    """
    Create one complex-math operation as a **ShaderNodeGroup** and embed it
    inside *parent_nodes* (e.g. a material's node tree).

    Parameters
    ----------
    parent_nodes : nodes collection of the owning tree (material, etc.)
    method       : same method strings as ComplexMathNode
    name         : optional label; defaults to "ComplexShader_<method>"

    Returns
    -------
    The ShaderNodeGroup node added to parent_nodes.
    """
    if name is None:
        name = f"ComplexShader_{method}"

    inner = bpy.data.node_groups.new(name, "ShaderNodeTree")
    gi = inner.nodes.new("NodeGroupInput")
    go = inner.nodes.new("NodeGroupOutput")

    if method in _BINARY_METHODS:
        make_new_socket(inner, "z1", io="INPUT",  type="NodeSocketVector")
        make_new_socket(inner, "z2", io="INPUT",  type="NodeSocketVector")
    else:
        make_new_socket(inner, "z",  io="INPUT",  type="NodeSocketVector")

    make_new_socket(inner, "result", io="OUTPUT", type="NodeSocketVector")
    make_new_socket(inner, "arg",    io="OUTPUT", type="NodeSocketFloat")

    _build_complex_graph(inner.nodes, inner.links, gi, go, method)

    group = parent_nodes.new("ShaderNodeGroup")
    group.node_tree = name  # assign by name — replaced below
    group.node_tree = inner
    group.name = name
    return group


# ---------------------------------------------------------------------------
# Mandelbrot material using shader-node complex arithmetic
# ---------------------------------------------------------------------------

def make_mandelbrot_material(iterations=20, escape_radius=2.0):
    """
    Build a Blender material that renders the Mandelbrot set on a flat plane.

    The iteration  z_{n+1} = z_n² + c  is unrolled *iterations* times using
    :func:`make_complex_shader_group`.  An escape-count accumulator tracks how
    many steps z stayed inside |z| < *escape_radius*; this is normalised to
    [0, 1] and fed into a colour ramp.

    Parameters
    ----------
    iterations    : number of unrolled Mandelbrot iterations (default 20)
    escape_radius : threshold for |z| (default 2.0)

    Returns
    -------
    bpy.types.Material
    """
    mat = bpy.data.materials.new(name="ComplexMandelbrot")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    out_node = nodes.get("Material Output")

    # Remove default BSDF so we can wire directly to an Emission shader
    for n in list(nodes):
        if n != out_node:
            nodes.remove(n)

    # --- local helper (same as _build_complex_graph's math_op) -----------
    def m(op, *args):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        n.hide = True
        for i, a in enumerate(args):
            if isinstance(a, (int, float)):
                n.inputs[i].default_value = float(a)
            else:
                links.new(a, n.inputs[i])
        return n.outputs["Value"]

    # --- UV / coordinate input -------------------------------------------
    # Use UV coordinates (always [0,1]) mapped to Mandelbrot view:
    #   Re ∈ [-2.25, 1.25],  Im ∈ [-1.25, 1.25]
    tex = nodes.new("ShaderNodeTexCoord")

    sep_uv = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["UV"], sep_uv.inputs["Vector"])
    u = sep_uv.outputs["X"]
    v = sep_uv.outputs["Y"]

    # c_re = 3.5*u − 2.25  →  Re ∈ [−2.25, 1.25] for u ∈ [0,1]
    # c_im = 2.5*v − 1.25  →  Im ∈ [−1.25, 1.25] for v ∈ [0,1]
    c_re = m("MULTIPLY_ADD", u, 3.5, -2.25)
    c_im = m("MULTIPLY_ADD", v, 2.5, -1.25)

    c_abs = m("SQRT", m("ADD", m("MULTIPLY", c_re, c_re), m("MULTIPLY", c_im, c_im)))

    c_vec = nodes.new("ShaderNodeCombineXYZ")
    c_vec.hide = True
    links.new(c_re,  c_vec.inputs["X"])
    links.new(c_im,  c_vec.inputs["Y"])
    links.new(c_abs, c_vec.inputs["Z"])
    c = c_vec.outputs["Vector"]

    # --- Mandelbrot iteration chain --------------------------------------
    # z_0 = (0, 0, 0)
    z_init = nodes.new("ShaderNodeCombineXYZ")
    z_init.hide = True
    z_init.inputs["X"].default_value = 0.0
    z_init.inputs["Y"].default_value = 0.0
    z_init.inputs["Z"].default_value = 0.0
    z = z_init.outputs["Vector"]

    # count: number of iterations where |z| < escape_radius (before squaring)
    count = 0.0  # Python float; first make_math call wires it as a constant

    for i in range(iterations):
        # Extract |z_n| from Z component before squaring
        sep_z = nodes.new("ShaderNodeSeparateXYZ")
        sep_z.hide = True
        links.new(z, sep_z.inputs["Vector"])

        # inside_n = 1.0 if |z_n| < escape_radius, else 0.0
        inside = m("LESS_THAN", sep_z.outputs["Z"], float(escape_radius))
        count  = m("ADD", count, inside)

        # z = z*z + c
        z_sq  = make_complex_shader_group(nodes, "MULTIPLY", name=f"Mandel_Sq_{i}")
        links.new(z, z_sq.inputs["z1"])
        links.new(z, z_sq.inputs["z2"])

        z_add = make_complex_shader_group(nodes, "ADD", name=f"Mandel_Add_{i}")
        links.new(z_sq.outputs["result"], z_add.inputs["z1"])
        links.new(c,                      z_add.inputs["z2"])
        z = z_add.outputs["result"]

    # --- Smooth colouring -----------------------------------------------
    # smooth ∈ [0, 1]: 0 = escaped immediately, 1 = always inside (Mandelbrot set)
    smooth = m("DIVIDE", count, float(iterations))

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "LINEAR"
    # 0.00 → deep blue  (fast escape, far outside)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color    = (0.02, 0.05, 0.35, 1)
    # 1.00 → black      (inside the Mandelbrot set)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color    = (0.0, 0.0, 0.0, 1)
    # midpoint colours for the interesting boundary region
    e1 = ramp.color_ramp.elements.new(0.30)
    e1.color = (0.9, 0.45, 0.05, 1)    # orange-gold
    e2 = ramp.color_ramp.elements.new(0.60)
    e2.color = (0.6, 0.1, 0.55, 1)     # violet
    e3 = ramp.color_ramp.elements.new(0.88)
    e3.color = (0.1, 0.05, 0.3, 1)     # dark indigo
    links.new(smooth, ramp.inputs["Fac"])

    # --- Emission shader for clean flat rendering -----------------------
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    links.new(ramp.outputs["Color"],    emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out_node.inputs["Surface"])

    return mat


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _connect(tree, value, socket):
    """Wire a value (socket, scalar, or tuple) to a node input socket."""
    if isinstance(value, bpy.types.NodeSocket):
        tree.links.new(value, socket)
    elif isinstance(value, (int, float)):
        socket.default_value = float(value)
    elif isinstance(value, (list, tuple)):
        socket.default_value = value


# ---------------------------------------------------------------------------
# Unified node: all 10 operations behind a dropdown menu
# ---------------------------------------------------------------------------

_ALL_METHODS = [
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE",
    "CONJUGATE", "LENGTH", "ANGLE", "EXPONENTIATE", "SINE", "COSINE",
]


def make_complex_menu_node_group(parent_tree, name="ComplexMath"):
    """
    Create a single geometry-node group that exposes **all 10 complex
    operations** through a dropdown menu socket.

    The node computes every operation in parallel and uses two
    ``GeometryNodeMenuSwitch`` nodes to route the selected result to the
    outputs.  Only the selected branch is *evaluated* by Blender at render
    time (lazy evaluation), so unused operations carry no runtime cost.

    Inputs
    ------
    z1        VECTOR  – primary operand (Re, Im, |z|)
    z2        VECTOR  – second operand for binary ops; ignored for unary
    Operation MENU    – dropdown: ADD / SUBTRACT / MULTIPLY / DIVIDE /
                        CONJUGATE / LENGTH / ANGLE / EXPONENTIATE / SINE / COSINE

    Outputs
    -------
    result    VECTOR  – (Re_out, Im_out, |z_out|)
    arg       FLOAT   – atan2(Im_out, Re_out)

    Parameters
    ----------
    parent_tree : node tree that will own the returned group node
    name        : label of the group node (and the inner node-tree)

    Returns
    -------
    bpy.types.GeometryNode  – the ``GeometryNodeGroup`` added to *parent_tree*
    """
    inner = bpy.data.node_groups.new(name, "GeometryNodeTree")

    # ── interface sockets ─────────────────────────────────────────────────
    inner.interface.new_socket("z1",        in_out="INPUT",  socket_type="NodeSocketVector")
    inner.interface.new_socket("z2",        in_out="INPUT",  socket_type="NodeSocketVector")
    inner.interface.new_socket("Operation", in_out="INPUT", socket_type="NodeSocketMenu")
    inner.interface.new_socket("result",    in_out="OUTPUT", socket_type="NodeSocketVector")
    inner.interface.new_socket("arg",       in_out="OUTPUT", socket_type="NodeSocketFloat")

    nodes = inner.nodes
    links = inner.links
    gi    = nodes.new("NodeGroupInput")
    go    = nodes.new("NodeGroupOutput")

    # ── math helpers ──────────────────────────────────────────────────────
    def math(op, *args):
        n = nodes.new("ShaderNodeMath"); n.operation = op; n.hide = True
        for i, a in enumerate(args):
            if isinstance(a, (int, float)): n.inputs[i].default_value = float(a)
            else: links.new(a, n.inputs[i])
        return n.outputs["Value"]

    def sep(vec):
        s = nodes.new("ShaderNodeSeparateXYZ"); s.hide = True
        links.new(vec, s.inputs["Vector"])
        return s.outputs["X"], s.outputs["Y"], s.outputs["Z"]

    def comb(x, y, z):
        c = nodes.new("ShaderNodeCombineXYZ"); c.hide = True
        for val, slot in [(x, "X"), (y, "Y"), (z, "Z")]:
            if isinstance(val, (int, float)): c.inputs[slot].default_value = float(val)
            else: links.new(val, c.inputs[slot])
        return c.outputs["Vector"]

    def abso(re, im):
        return math("SQRT", math("ADD", math("MULTIPLY", re, re), math("MULTIPLY", im, im)))

    def argo(re, im):
        return math("ARCTAN2", im, re)

    # ── compute inputs ────────────────────────────────────────────────────
    re1, im1, ab1 = sep(gi.outputs["z1"])
    re2, im2, ab2 = sep(gi.outputs["z2"])

    res = {}
    arg = {}

    # ADD
    re_a = math("ADD", re1, re2); im_a = math("ADD", im1, im2)
    res["ADD"] = comb(re_a, im_a, abso(re_a, im_a));  arg["ADD"] = argo(re_a, im_a)

    # SUBTRACT
    re_s = math("SUBTRACT", re1, re2); im_s = math("SUBTRACT", im1, im2)
    res["SUBTRACT"] = comb(re_s, im_s, abso(re_s, im_s)); arg["SUBTRACT"] = argo(re_s, im_s)

    # MULTIPLY  z1*z2 = (Re1·Re2−Im1·Im2) + i(Re1·Im2+Im1·Re2), |z1·z2|=|z1|·|z2|
    re_m = math("SUBTRACT", math("MULTIPLY", re1, re2), math("MULTIPLY", im1, im2))
    im_m = math("ADD",      math("MULTIPLY", re1, im2), math("MULTIPLY", im1, re2))
    res["MULTIPLY"] = comb(re_m, im_m, math("MULTIPLY", ab1, ab2))
    arg["MULTIPLY"] = argo(re_m, im_m)

    # DIVIDE  z1/z2, |z1/z2|=|z1|/|z2|
    den = math("ADD", math("MULTIPLY", re2, re2), math("MULTIPLY", im2, im2))
    re_d = math("DIVIDE", math("ADD",      math("MULTIPLY", re1, re2), math("MULTIPLY", im1, im2)), den)
    im_d = math("DIVIDE", math("SUBTRACT", math("MULTIPLY", im1, re2), math("MULTIPLY", re1, im2)), den)
    res["DIVIDE"] = comb(re_d, im_d, math("DIVIDE", ab1, ab2)); arg["DIVIDE"] = argo(re_d, im_d)

    # CONJUGATE  (z2 ignored)
    im_conj = math("MULTIPLY", im1, -1.0)
    res["CONJUGATE"] = comb(re1, im_conj, ab1);  arg["CONJUGATE"] = argo(re1, im_conj)

    # LENGTH → real scalar  (z2 ignored)
    ab_len = abso(re1, im1)
    res["LENGTH"] = comb(ab_len, 0.0, ab_len);  arg["LENGTH"] = math("ARCTAN2", 0.0, ab_len)

    # ANGLE → real scalar  (z2 ignored)
    ar = argo(re1, im1)
    res["ANGLE"] = comb(ar, 0.0, math("ABSOLUTE", ar));  arg["ANGLE"] = math("ARCTAN2", 0.0, ar)

    # EXPONENTIATE  e^z = e^Re·(cos Im + i·sin Im),  |e^z|=e^Re  (z2 ignored)
    er = math("EXPONENT", re1)
    re_e = math("MULTIPLY", er, math("COSINE", im1))
    im_e = math("MULTIPLY", er, math("SINE",   im1))
    res["EXPONENTIATE"] = comb(re_e, im_e, er);  arg["EXPONENTIATE"] = argo(re_e, im_e)

    # SINE  sin z = sin(Re)·cosh(Im) + i·cos(Re)·sinh(Im)  (z2 ignored)
    re_sn = math("MULTIPLY", math("SINE",   re1), math("COSH", im1))
    im_sn = math("MULTIPLY", math("COSINE", re1), math("SINH", im1))
    res["SINE"] = comb(re_sn, im_sn, abso(re_sn, im_sn));  arg["SINE"] = argo(re_sn, im_sn)

    # COSINE  cos z = cos(Re)·cosh(Im) − i·sin(Re)·sinh(Im)  (z2 ignored)
    re_cn = math("MULTIPLY", math("COSINE", re1), math("COSH", im1))
    im_cn = math("MULTIPLY", -1.0, math("MULTIPLY", math("SINE", re1), math("SINH", im1)))
    res["COSINE"] = comb(re_cn, im_cn, abso(re_cn, im_cn));  arg["COSINE"] = argo(re_cn, im_cn)

    # ── menu switch: result (VECTOR) ──────────────────────────────────────
    sw_v = nodes.new("GeometryNodeMenuSwitch")
    sw_v.data_type = "VECTOR"
    sw_v.enum_items[0].name = _ALL_METHODS[0]   # rename default "A" → ADD
    sw_v.enum_items[1].name = _ALL_METHODS[1]   # rename default "B" → SUBTRACT
    for m in _ALL_METHODS[2:]:
        sw_v.enum_items.new(m)
    links.new(gi.outputs["Operation"], sw_v.inputs["Menu"])
    for m in _ALL_METHODS:
        links.new(res[m], sw_v.inputs[m])

    # ── menu switch: arg (FLOAT) ──────────────────────────────────────────
    sw_f = nodes.new("GeometryNodeMenuSwitch")
    sw_f.data_type = "FLOAT"
    sw_f.enum_items[0].name = _ALL_METHODS[0]
    sw_f.enum_items[1].name = _ALL_METHODS[1]
    for m in _ALL_METHODS[2:]:
        sw_f.enum_items.new(m)
    links.new(gi.outputs["Operation"], sw_f.inputs["Menu"])
    for m in _ALL_METHODS:
        links.new(arg[m], sw_f.inputs[m])

    # ── wire group outputs ────────────────────────────────────────────────
    links.new(sw_v.outputs["Output"], go.inputs["result"])
    links.new(sw_f.outputs["Output"], go.inputs["arg"])

    # ── embed in parent tree ──────────────────────────────────────────────
    group = parent_tree.nodes.new("GeometryNodeGroup")
    group.node_tree = inner
    group.name = name
    return group
