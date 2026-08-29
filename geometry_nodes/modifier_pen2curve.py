"""A drawing that draws itself, from an ipad pen recording.

This is the tree of ``video_interferences/media/blend/manual/EnvelopeCalculations.blend``
- authored in the node editor, ported here so a scene can build it - together
with the one thing the hand-built version could not do: a **Progress** dial
that runs 0 to 1 instead of an integer that has to know how many points the
drawing has.

The idea is one node deep. ``pen2curve`` (the app in ``pen2curve/``, imported
through ``pen2curve/blender/import_pen2curve.py`` or, for these scenes,
through :class:`~objects.pen2curve.Pen2CurveObject`) writes the strokes down
**in the order the hand made them**, and the importer samples them into grease
pencil points in that same order. So point index *is* time: keep the points
whose index is below a threshold, throw the rest away, and walk the threshold
from 0 to the point count. What comes out is not an animation of a drawing
appearing - it is the drawing being written, at the speed the threshold moves,
in the order it was written on the ipad.

The nodes::

    Object Info  -> the grease pencil object holding the strokes. Read in
                    ORIGINAL space, so the source object's own transform is
                    ignored and it can be parked anywhere out of the way;
                    what places the drawing in the world is the transform of
                    the object this modifier hangs on.
    Grease Pencil to Curves -> Realize Instances -> Attribute Statistic
                 -> Max of the Index over the points: the last index in the
                    drawing, so the tree knows its own length and nothing has
                    to be told how many points the file held.
    Delete Geometry (POINT, selection = Index < Start or
                     Index > Progress * last index)
                 -> everything the pen has not reached yet, and everything
                    from before the drawing is meant to start.
    Transform Geometry -> ``scale``, kept from the original tree, where the
                    page arrived in metres and had to be blown up by 50.
    Set Curve Radius <- Random Value
                 -> one random thickness per point in [radius]. The recorded
                    pen pressure is *not* what draws the line here: a pen
                    that varies a little at random reads as a felt tip, and
                    the pressure attribute is still on the points for anyone
                    who wants it.
    Object Info x2 -> Join -> Transform -> Set Position <- Sample Index
                 -> the pencil, optional, riding on the last drawn point:
                    Floor of the same pen position is an index, and Sample
                    Index reads that point's position off the curves. The
                    result joins the ink on the way out, so one modifier
                    hands over the writing and the hand that is doing it.

The comparison is done in FLOAT rather than INT (the original multiplied an
integer by three) so that the threshold moves smoothly: with ten thousand
points, an integer dial would step the drawing forward in visible jumps
whenever the rounding changed.

Visibility, which is the one trap in this arrangement: the source object has
to stay in the depsgraph for ``Object Info`` to see it, but it must not appear
in the picture, since the whole drawing would then be on screen from frame
zero. ``hide_render = True`` does exactly that and nothing else - the render
excludes the object itself while the modifier still reads its geometry
(verified by rendering both ways). Hiding it with the *eye* instead drops it
from the viewport depsgraph, and the drawing disappears while you work on it.
:class:`~objects.pen2curve.Pen2CurveObject` sets this up.
"""
from math import pi

from geometry_nodes.geometry_nodes_modifier import GeometryNodesModifier
from geometry_nodes.nodes import (AttributeStatistic, BooleanMath, CompareNode,
                                  DeleteGeometry, GreasePencilToCurves, Index,
                                  InputInteger, InputValue, JoinGeometry, MathNode,
                                  ObjectInfo, Position, RandomValue,
                                  RealizeInstances, SampleIndex, SetCurveRadius,
                                  SetPosition, TransformGeometry,
                                  create_geometry_line, make_function)
from interface.ibpy import Vector


def pencil_parts(pencil):
    """The blender objects a ``pencil`` argument stands for.

    :class:`~objects.derived_objects.pencil.Pencil` is two objects, ``_Wood``
    and ``_Mine``, parented to an empty - and ``Object Info`` reads one object
    at a time, not a hierarchy, so the tree needs them separately. A list, a
    single object and a bare :class:`~objects.bobject.BObject` all work too.
    """
    if pencil is None:
        return []
    if isinstance(pencil, (list, tuple)):
        return list(pencil)
    parts = [getattr(pencil, part, None) for part in ("wood", "mine")]
    parts = [part for part in parts if part is not None]
    return parts or [pencil]


class Pen2CurveModifier(GeometryNodesModifier):
    r"""The strokes of a pen2curve drawing, revealed in the order they were drawn.

    The dial, reachable with
    ``ibpy.get_geometry_node_from_modifier(modifier, "Progress")``:

    ``Progress``
        0 = blank page, 1 = the whole drawing. It is a plain fraction of the
        point count, so it is also linear in *pen distance travelled* rather
        than in strokes - a long stroke takes proportionally longer to draw
        than a short one, which is what makes the result read as a hand.

    :param source: the grease pencil object holding the strokes, or its name.
        It must stay in the depsgraph; see the module docstring on how to keep
        it out of the picture without hiding it from geometry nodes.
    :param pencil: what to park on the point being written, or ``None`` for
        ink alone. A :class:`~objects.derived_objects.pencil.Pencil` (whose
        ``wood`` and ``mine`` are read as two objects), a list of objects, or
        one object - see :func:`pencil_parts`. Whatever it is, it is read in
        ORIGINAL space, so its own transform is ignored and only
        ``pencil_rotation`` and ``pencil_scale`` decide how it is held.
    :param pencil_rotation: how it is held, in radians. The default is the
        pose measured in the node editor: leaning back and to the left, the
        angle a right hand makes.
    :param pencil_scale: how big it is, as a number or three.
    :param progress: where the dial starts.
    :param start_index: how many points at the head of the recording to drop.
        The pen usually taps once before it starts writing, and index 0 is
        that tap rather than the first letter. It is not part of the dial:
        ``Progress`` still runs 0 to 1 over the whole drawing, this only
        decides what the beginning of it is. Reachable afterwards as the
        ``Start`` node.
    :param scale: uniform (or per-axis) scale for the page.
    :param radius: the ink thickness, as ``(min, max)`` of a random draw per
        point, or a single number for a constant one. ``None`` keeps the
        radius the importer wrote from the recorded pen pressure.
    :param seed: the seed of that random draw.
    """

    def __init__(self, source=None, pencil=None,
                 pencil_rotation=(135/180*pi, -1.2374383211135864, 0.0),
                 pencil_scale=0.75, progress=0.0, start_index=0, scale=1.0,
                 radius=(0.04, 0.05), seed=0, name="Pen2CurveModifier", **kwargs):
        self.start_node = None
        self.progress_node = None
        if source is None:
            raise ValueError("Pen2CurveModifier needs the grease pencil object "
                             "that holds the strokes")
        self.source = source
        self.pencil = pencil_parts(pencil)
        self.pencil_rotation = Vector(pencil_rotation)
        if isinstance(pencil_scale, (int, float)):
            pencil_scale = [pencil_scale] * 3
        self.pencil_scale = Vector(pencil_scale)
        self.progress = progress
        self.start_index = int(start_index)
        if isinstance(scale, (int, float)):
            scale = Vector([scale] * 3)
        self.scale = Vector(scale)
        if isinstance(radius, (int, float)):
            radius = (radius, radius)
        self.radius = radius
        self.seed = seed
        super().__init__(name=name, automatic_layout=False, **kwargs)

    # ------------------------------------------------------------------
    def create_node(self, tree, **kwargs):
        drawing = ObjectInfo(tree, location=(-7.4, 0.5), object=self.source,
                             transform_space="ORIGINAL", name="Drawing")

        # --- how many points there are, asked of the drawing itself -------
        # A grease pencil component is layers of drawings rather than points,
        # so nothing measures it until it has been through Grease Pencil to
        # Curves - and that node hands its curves over one instance per layer
        # unless they are realized, which is a geometry the POINT domain of
        # an Attribute Statistic still finds empty. Both nodes are here for
        # that one reason: Max of the Index over the realized points is the
        # last index in the drawing.
        curves = GreasePencilToCurves(tree, location=(-6.4, -0.3),
                                      grease_pencil=drawing.geometry_out,
                                      name="StrokesAsCurves")
        realized = RealizeInstances(tree, location=(-5.4, -0.9),
                                    geometry=curves.geometry_out,
                                    name="OneDrawing")
        last = AttributeStatistic(tree, location=(-4.5, -0.4), data_type="FLOAT",
                                  domain="POINT", std_out="Max",
                                  geometry=realized.geometry_out,
                                  attribute=Index(tree, location=(-5.8, -1.8),
                                                  name="CountIndex").std_out,
                                  name="LastPoint")

        # --- the dial, and the index it is compared against ---------------
        self.progress_node = InputValue(tree, location=(-5.8, -2.4),
                                        value=self.progress, name="Progress")
        pen = MathNode(tree, location=(-3.1, -0.8), operation="MULTIPLY",
                       inputs0=last.std_out, inputs1=self.progress_node.std_out,
                       name="PenPosition")
        index = Index(tree, location=(-3.1, -2.1), name="PointIndex")
        # everything the pen has not reached yet
        ahead = CompareNode(tree, location=(-1.6, -1.8), data_type="FLOAT",
                            operation="GREATER_THAN", inputs0=index.std_out,
                            inputs1=pen.std_out, name="NotYetDrawn")
        # ...and everything from before the drawing is meant to start. A
        # recording usually opens with a stray tap or two - the pen touching
        # down to see that it writes - and those are the first thing the
        # threshold would reveal. This is a fixed offset rather than a dial:
        # the pen never travels over that part, it was never the drawing.
        self.start_node = InputInteger(tree, location=(-5.7, -2.2),
                                       integer=self.start_index, name="Start")
        before = CompareNode(tree, location=(-1.6, -0.9), data_type="FLOAT",
                             operation="LESS_THAN", inputs0=index.std_out,
                             inputs1=self.start_node.std_out, name="BeforeStart")
        gone = BooleanMath(tree, location=(-0.5, -1.1), operation="OR",
                           inputs0=before.std_out, inputs1=ahead.std_out,
                           name="NotOnThePage")

        undrawn = DeleteGeometry(tree, location=(0.4, -0.6), domain="POINT",
                                 mode="ALL", geometry=drawing.geometry_out,
                                 selection=gone.std_out, name="Undrawn")
        page = TransformGeometry(tree, location=(2, 0), scale=self.scale,
                                 name="PageScale")

        line = [undrawn, page]
        if self.radius is not None:
            thickness = RandomValue(tree, location=(4, -2), data_type="FLOAT",
                                    min=self.radius[0], max=self.radius[1],
                                    seed=self.seed, name="InkThickness")
            line.append(SetCurveRadius(tree, location=(5, 0),
                                       radius=thickness.std_out, name="Ink"))

        self.group_outputs.location = (1400, 0)
        if not self.pencil:
            create_geometry_line(tree, line, out=self.group_outputs.inputs[0])
            return

        # --- the pencil that is doing the writing -------------------------
        # It rides on the *last drawn point*: Floor of the same pen position
        # the ink is cut at is an index, and Sample Index reads that point's
        # position out of the realized curves. So the pencil cannot drift
        # away from the ink - both are the one number, one as a threshold and
        # one as an index - and it jumps between strokes exactly where the
        # hand did, because that is what the recording says happened.
        #
        # The position is sampled *before* ``PageScale``, so a scaled page
        # would leave the pencil behind; with the default scale of 1 there is
        # nothing between them.
        parts = JoinGeometry(tree, location=(-5.7, 2.1), name="PencilParts")
        for i, part in enumerate(self.pencil):
            piece = ObjectInfo(tree, location=(-6.9, 1.7 + 1.2 * i), object=part,
                               transform_space="ORIGINAL",
                               name="PencilPart%d" % i)
            tree.links.new(piece.geometry_out, parts.geometry_in)

        pen_selector = make_function(tree, name="PenSelector",
                                     functions={
                                         "delete": "progress,0,>,progress,1,<,and,not"
                                     }, inputs=["progress"], outputs=["delete"],
                                     scalars=["progress", "delete"], vectors=[], location=(-5, 2))
        tree.links.new(self.progress_node.std_out,pen_selector.inputs["progress"])
        delete_geometry = DeleteGeometry(tree, location=(-4, 2.1), selection=pen_selector.outputs["delete"])
        # how the pencil is held: leaning back over its own tip, which is
        # where its origin is, so that moving the origin to a point of the
        # drawing puts the tip on the ink rather than the middle of the shaft
        posed = TransformGeometry(tree, location=(1.7, 1.7),
                                  rotation=self.pencil_rotation,
                                  scale=self.pencil_scale, name="PencilPose")

        create_geometry_line(tree, [parts, delete_geometry, posed])
        tip = MathNode(tree, location=(-0.8, 0.3), operation="FLOOR",
                       inputs0=pen.std_out, name="PenPoint")
        where = SampleIndex(tree, location=(0.6, 1.3), data_type="FLOAT_VECTOR",
                            domain="POINT", geometry=realized.geometry_out,
                            value=Position(tree, location=(-1, 1),
                                           name="StrokePosition").std_out,
                            index=tip.std_out, name="PenTip")
        held = SetPosition(tree, location=(4.6, 1.4), geometry=posed.geometry_out,
                           offset=where.std_out, name="PencilAtTip")

        both = JoinGeometry(tree, location=(6, 0.5), name="InkAndPencil")
        create_geometry_line(tree, line, out=both.geometry_in)
        tree.links.new(held.geometry_out, both.geometry_in)
        tree.links.new(both.geometry_out, self.group_outputs.inputs[0])
