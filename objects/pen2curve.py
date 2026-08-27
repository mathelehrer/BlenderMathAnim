"""Handwriting recorded on an ipad, written onto a blender scene.

``pen2curve`` (the app in ``pen2curve/`` of the workspace) records pen strokes
and exports them as fitted bezier curves - position, handles, radius and pen
force per knot - in a json file whose ``format`` is ``"pen2curve-curves"``.
:class:`Pen2CurveObject` is what a scene asks for: it reads such a file from
the data folder, samples the curves back into grease pencil strokes, and hangs
a :class:`~geometry_nodes.modifier_pen2curve.Pen2CurveModifier` on a cube so
that the drawing can be *written* rather than switched on.

The sampling below is the same arithmetic as
``pen2curve/blender/import_pen2curve.py`` (the add-on that does this
interactively through File > Import). It is repeated here rather than imported
because that file lives outside this package, next to the app it belongs to,
and a scene should not have to know where the workspace keeps its tools.

Two objects come out of one :class:`Pen2CurveObject`:

* the **source**, a grease pencil object carrying every point of the drawing.
  It is never rendered: ``hide_render = True`` keeps it out of the picture
  while the *render* depsgraph still evaluates it, which is what
  ``Object Info`` inside the modifier needs. ``hide_viewport`` (on by
  default) takes it out of the viewport as well, at the price named on
  :class:`Pen2CurveObject`: the viewport depsgraph then drops it too, so the
  modifier previews as nothing while the render is unaffected - measured as
  pixel-identical frames with the flag either way.
* the **object itself**, a cube whose geometry the modifier replaces with as
  much of that drawing as ``Progress`` has reached. This is the one with a
  transform, so location, rotation and scale place the writing in the world.

The page arrives lying in its own plane: ``orientation="FRONT"`` stands it up
in x-z facing -y, the same way :class:`~objects.tex_bobject.SimpleTexBObject`
stands up glyphs, so a camera looking along +y reads it head on;
``orientation="TOP"`` lays it flat in x-y like paper on a table.
"""

import json
import os

import bpy

from appearance.textures import get_color_from_name
from geometry_nodes.modifier_pen2curve import Pen2CurveModifier, pencil_parts
from interface import ibpy
from interface.ibpy import Vector, get_obj
from objects.cube import Cube
from utils.constants import COLOR_NAMES, DATA_DIR, DEFAULT_ANIMATION_TIME, FRAME_RATE


# ===========================================================================
#  The file, and the strokes in it
# ===========================================================================
def load_pen2curve_document(filename):
    """The json of a pen2curve "Export for Blender", by name or by path.

    A bare name is looked up in ``DATA_DIR``, which is ``data`` next to the
    scene being run - ``video_interferences/data`` for the interference video.
    """
    path = filename if os.path.isabs(filename) else os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("format") != "pen2curve-curves":
        raise ValueError("%s is not a pen2curve blender export (format is %r); "
                         "use the app's 'Export for Blender' button rather than "
                         "'Export raw JSON'" % (path, document.get("format")))
    return document


def _srgb_to_linear(channel):
    """Blender works in linear light; the file stores sRGB hex."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _hex_to_rgba(value):
    text = (value or "#000000").lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(_srgb_to_linear(int(text[i:i + 2], 16) / 255.0)
                 for i in (0, 2, 4)) + (1.0,)


def _color_to_rgba(color):
    """A palette name, or a hex string from the file, as linear rgba."""
    if color in COLOR_NAMES:
        return tuple(get_color_from_name(color))
    return _hex_to_rgba(color)


def _bezier_at(p0, p1, p2, p3, t):
    s = 1.0 - t
    a, b, c, d = s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def _chord_length(p0, p1, p2, p3):
    """Rough arc length, used only to choose a sample count."""
    total, previous = 0.0, p0
    for i in range(1, 9):
        point = _bezier_at(p0, p1, p2, p3, i / 8.0)
        total += ((point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2) ** 0.5
        previous = point
    return total


def _mean_radius(strokes):
    radii = [node.get("r", 1.0) for stroke in strokes
             for node in stroke.get("nodes", [])]
    mean = (sum(radii) / len(radii)) if radii else 1.0
    return mean if mean > 0 else 1.0


def plan_strokes(document, page_width=1.0, orientation="FRONT",
                 use_pressure=True, samples_per_px=0.35, max_per_segment=64):
    """The fitted curves, walked back into polylines, in drawing order.

    Each segment is sampled at a density set by its own arc length, so a long
    sweep gets more points than a tick and the point index stays a fair
    measure of how far the pen has travelled - which is what
    :class:`~geometry_nodes.modifier_pen2curve.Pen2CurveModifier` turns into
    the passage of time.

    Document y points down and blender's y points up, so y is negated; the
    drawing then reads the right way round rather than mirrored.

    :return: ``(strokes, size)`` - a list of ``{"color", "points"}`` with each
        point a ``{"co", "radius", "pressure"}``, and the ``(width, height)``
        of the ink itself, both in the units set by ``page_width``.
    """
    page = document.get("page") or {}
    width_px = page.get("width") or 0.0
    scale = (page_width / width_px) if width_px > 0 else page_width
    strokes = document.get("strokes") or []
    mean_radius = _mean_radius(strokes)

    def to3(x, y):
        u, v = x * scale, -y * scale
        return (u, v, 0.0) if orientation == "TOP" else (u, 0.0, v)

    planned = []
    for stroke in strokes:
        nodes = stroke.get("nodes") or []
        if len(nodes) < 2:
            continue
        points = []
        for i in range(len(nodes) - 1):
            here, there = nodes[i], nodes[i + 1]
            p0, p1 = here["co"], here["hr"]
            p2, p3 = there["hl"], there["co"]
            steps = max(2, min(max_per_segment,
                               int(_chord_length(p0, p1, p2, p3) * samples_per_px) + 1))
            r0, r1 = here.get("r", mean_radius), there.get("r", mean_radius)
            f0, f1 = here.get("p", 0.5), there.get("p", 0.5)
            # skip t = 0 after the first segment: it repeats the previous end
            for k in range(0 if i == 0 else 1, steps + 1):
                t = k / steps
                x, y = _bezier_at(p0, p1, p2, p3, t)
                radius = (r0 + (r1 - r0) * t) if use_pressure else mean_radius
                points.append({"co": to3(x, y), "radius": radius * scale,
                               "pressure": f0 + (f1 - f0) * t})
        if len(points) >= 2:
            planned.append({"color": stroke.get("color", "#000000"),
                            "points": points})
    return planned, _extent(planned)


def _extent(strokes):
    """``(low, high)`` corners of the ink, as vectors."""
    coordinates = [point["co"] for stroke in strokes for point in stroke["points"]]
    if not coordinates:
        return Vector(), Vector()
    low = Vector([min(c[axis] for c in coordinates) for axis in range(3)])
    high = Vector([max(c[axis] for c in coordinates) for axis in range(3)])
    return low, high


def _place(strokes, factor=1.0, shift=Vector()):
    """Scale every point about the origin and move it, in place."""
    for stroke in strokes:
        for point in stroke["points"]:
            point["co"] = tuple(point["co"][axis] * factor + shift[axis]
                                for axis in range(3))
            point["radius"] *= factor


# ===========================================================================
#  Sources the modifier reads but nobody sees
# ===========================================================================
def _fcurve_holders(action):
    """Where an action keeps its f-curves, on either blender's model."""
    if getattr(action, "fcurves", None) is not None:
        yield action
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                yield channelbag


def hide_from_camera(obj, hide_viewport=True):
    """Take an object out of the picture without taking it out of geometry nodes.

    ``hide_render`` alone does that: the render depsgraph still evaluates the
    object, so ``Object Info`` reads it while the camera never sees it, and
    ``hide_viewport`` does the same for the viewport (at the price of the
    modifier previewing as nothing there).

    The keyframes are the catch, and the reason this is a function rather
    than two assignments. :meth:`~objects.bobject.BObject.appear` goes
    through ``ibpy.unhide_frm``, which *keyframes* ``hide_render`` - and an
    animated property beats whatever the flag was set to, silently, at render
    time only. So anything that has ever been made to appear has to have
    those two channels cleared before it will stay hidden. Anything that
    appears *afterwards* keyframes them again, which is why
    :class:`Pen2CurveObject` reapplies this on every ``write``.
    """
    obj = get_obj(obj)
    animation = getattr(obj, "animation_data", None)
    action = None if animation is None else animation.action
    if action is not None:
        for holder in _fcurve_holders(action):
            for fcurve in list(holder.fcurves):
                if fcurve.data_path in ("hide_render", "hide_viewport"):
                    holder.fcurves.remove(fcurve)
    obj.hide_render = True
    obj.hide_viewport = hide_viewport
    return obj


# ===========================================================================
#  The grease pencil object the modifier reads
# ===========================================================================
def _grease_pencil_material(color):
    """One ink colour, as a grease pencil material.

    Grease pencil strokes are drawn with their material's *stroke* colour and
    are not lit, which is what makes ink read as ink: the same flat line
    wherever it lies in the scene.
    """
    name = "pen2curve GP " + str(color)
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    creator = getattr(bpy.data.materials, "create_gpencil_data", None)
    if creator is not None:  # adds the .grease_pencil settings block
        creator(material)
    rgba = _color_to_rgba(color)
    settings = getattr(material, "grease_pencil", None)
    if settings is not None:
        settings.color = rgba
        settings.show_stroke = True
        settings.show_fill = False
    else:  # pragma: no cover - a blender without grease pencil materials
        material.use_nodes = True
        material.diffuse_color = rgba
    return material


def build_grease_pencil(name, strokes, ink=None, frame=1, hide_viewport=True):
    """A grease pencil object holding every point of the drawing.

    :param ink: how to colour the strokes. ``None`` keeps the ink colours the
        file recorded; a single palette name (or hex string) paints all of
        them; a dict maps the file's own hex colours onto palette names, and
        anything the dict does not mention keeps its own colour.
    :param hide_viewport: also take the object out of the viewport, not only
        out of the render. See :class:`Pen2CurveObject` for what that costs.
    :return: ``(object, number of points)``.
    """
    def recolour(color):
        if ink is None:
            return color
        if isinstance(ink, dict):
            return ink.get(color, color)
        return ink

    data = bpy.data.grease_pencils.new(name)
    obj = bpy.data.objects.new(name, data)
    ibpy.link(obj)
    # out of the render, so it is never the whole drawing sitting on top of
    # the one being written - the render depsgraph evaluates it either way,
    # which is what Object Info needs. hide_viewport does the same for the
    # viewport, where the depsgraph does drop it: the modifier then shows
    # nothing until the frame is rendered
    hide_from_camera(obj, hide_viewport)

    slots = {}
    for stroke in strokes:
        color = recolour(stroke["color"])
        if color not in slots:
            data.materials.append(_grease_pencil_material(color))
            slots[color] = len(data.materials) - 1
        stroke["slot"] = slots[color]

    layer = data.layers.new("pen2curve")
    drawing = layer.frames.new(frame).drawing
    drawing.add_strokes([len(stroke["points"]) for stroke in strokes])

    pressures = []
    for gp_stroke, stroke in zip(drawing.strokes, strokes):
        gp_stroke.material_index = stroke["slot"]
        for gp_point, point in zip(gp_stroke.points, stroke["points"]):
            gp_point.position = point["co"]
            gp_point.radius = point["radius"]
            gp_point.opacity = 1.0
            pressures.append(point["pressure"])

    # the raw pen force, for anything that wants to react to how hard the pen
    # was pressed rather than to how thick the line is
    attribute = drawing.attributes.new("pressure", "FLOAT", "POINT")
    for i, value in enumerate(pressures):
        attribute.data[i].value = value

    return obj, len(pressures)


# ===========================================================================
class Pen2CurveObject(Cube):
    r"""A pen2curve drawing that writes itself onto the scene.

    Example - the whistle calculation, filling six units of height on the
    back of an envelope, written over eighteen seconds::

        note = Pen2CurveObject("envelope_calculation.json", ink_height=6,
                               ink={"#111318": "text", "#dc2626": "important"},
                               location=[0, -0.02, 0.3], name="Note")
        note.write(begin_time=3, transition_time=18)

    :param filename: the json, by name in ``DATA_DIR`` or by full path.
    :param page_width: how wide the *page* is, in blender units. The ink
        covers rather less than the page, so this is usually not the number
        you want to think in - see ``ink_width`` and ``ink_height``.
    :param ink_width: how wide the *ink* should be. Overrides ``page_width``.
    :param ink_height: how tall the ink should be. Overrides both, and is the
        handy one for writing onto something of a known size.
    :param orientation: ``"FRONT"`` stands the page up in x-z facing -y (the
        default: a camera looking along +y reads it head on), ``"TOP"`` lays
        it flat in x-y.
    :param center: ``"ink"`` puts the middle of the writing at the object's
        origin - the useful one when the drawing has to be placed on
        something; ``"page"`` puts the middle of the page there, as the
        importer does; ``None`` keeps the document's own corner origin.
    :param ink: palette colours for the strokes. A name, or a dict keyed by
        the hex colours in the file - ``{"#111318": "text"}`` turns the black
        ink light without touching the red. Black ink is invisible on these
        scenes' black background, so this is nearly always worth setting.
    :param radius: ink thickness, ``(min, max)`` of a random draw per point.
        ``None`` uses the recorded pen pressure instead.
    :param hide_viewport: whether the grease pencil source is hidden in the
        viewport as well as in the render. On by default, because otherwise
        the whole drawing sits in the viewport from the first frame, on top
        of the one being written, and every object in the scene has to be
        picked out from behind it.

        What it costs is the preview: the viewport depsgraph drops a hidden
        object, so ``Object Info`` inside the modifier finds nothing there
        and the writing shows up only when a frame is rendered. The render
        itself is untouched - frames rendered with the flag on and off come
        out pixel for pixel identical. Turn it off while placing the drawing
        on something by eye, and leave it on otherwise.
    :param pencil: a :class:`~objects.derived_objects.pencil.Pencil` (or any
        object, or a list of them) to ride on the point being written, so the
        note is not only written but *seen* being written. ``None`` leaves
        the ink to appear by itself. ``pencil_rotation`` and ``pencil_scale``
        pass through to :class:`~geometry_nodes.modifier_pen2curve.Pen2CurveModifier`.
    :param hide_pencil: hide the pencil that was handed in, the same way the
        drawing's own source is hidden - the modifier reads it either way, so
        without this there is a second pencil in the shot, lying at wherever
        it was built. Note that ``Object Info`` reads it in ORIGINAL space:
        the pencil's *transform* is ignored, so animating the real one by
        ``grow`` or ``move`` changes nothing on screen. What does still come
        through is its material, an alpha fade included.
    :param progress: where the dial starts. 0 is a blank page.
    :param start_index: points at the head of the recording never to draw -
        the tap the pen makes before it starts writing, usually. Note that it
        does not move the drawing: the skipped points still count towards the
        ink extent that ``center`` and ``ink_height`` are measured from, so a
        stray mark far from the writing is worth cutting out of the json
        rather than skipping here.
    :param use_pressure: whether the sampled radius follows the pen force. Has
        no effect unless ``radius`` is ``None``, since the modifier overwrites
        the radius otherwise.
    :param samples_per_px: how densely the fitted curves are walked. Higher is
        smoother, heavier, and slower to write through.
    """

    def __init__(self, filename="envelope_calculation.json", page_width=1.0,
                 ink_width=None, ink_height=None, orientation="FRONT",
                 center="ink", ink=None, radius=(0.04, 0.05),
                 hide_viewport=True, pencil=None, hide_pencil=True,
                 progress=0.0, start_index=0, use_pressure=True,
                 samples_per_px=0.35, seed=0, name="Pen2Curve", **kwargs):
        # the pencil's pose belongs to the modifier, which owns the defaults
        pose = {key: kwargs.pop(key) for key in ("pencil_rotation", "pencil_scale")
                if key in kwargs}
        document = load_pen2curve_document(filename)
        strokes, (low, high) = plan_strokes(
            document, page_width=page_width, orientation=orientation,
            use_pressure=use_pressure, samples_per_px=samples_per_px)
        if not strokes:
            raise ValueError("%s holds no strokes" % filename)

        size = high - low
        # the page is the horizontal axis and the plane's other axis, whichever
        # way round the page was stood up
        across, along = (0, 1) if orientation == "TOP" else (0, 2)
        factor = 1.0
        if ink_height is not None and size[along] > 0:
            factor = ink_height / size[along]
        elif ink_width is not None and size[across] > 0:
            factor = ink_width / size[across]

        shift = Vector()
        if center == "ink":
            shift = -0.5 * (low + high) * factor
        elif center == "page":
            page = document.get("page") or {}
            width = page.get("width") or 0.0
            scale = (page_width / width * factor) if width > 0 else factor
            middle = (width / 2.0 * scale,
                      (page.get("height") or 0.0) / 2.0 * scale)
            shift = Vector((-middle[0], middle[1], 0.0)) if orientation == "TOP" \
                else Vector((-middle[0], 0.0, middle[1]))
        _place(strokes, factor, shift)

        low, high = _extent(strokes)
        self.extent = (low, high)
        self.ink_size = high - low
        self.ink_width = self.ink_size[across]
        self.ink_height = self.ink_size[along]

        self.source, self.points = build_grease_pencil(
            name + "Ink", strokes, ink=ink, hide_viewport=hide_viewport)
        # the modifier counts the points itself, out of the geometry it is
        # handed; self.points is kept because a scene likes to report it
        self.pencil = pencil_parts(pencil)
        # the same arrangement as the drawing's own source: read by the
        # modifier, never rendered on its own account, or there would be a
        # second pencil lying wherever the real one was built
        self.hide_pencil = hide_pencil
        self.hide_viewport = hide_viewport
        self._hide_the_pencil()
        self.modifier = Pen2CurveModifier(source=self.source, pencil=self.pencil,
                                          progress=progress, radius=radius,
                                          start_index=start_index, seed=seed,
                                          name=name + "Modifier", **pose)

        kwargs["name"] = name
        kwargs.setdefault("no_material", True)
        super().__init__(**kwargs)
        self.add_mesh_modifier(type='NODES', node_modifier=self.modifier)
        self.progress = ibpy.get_geometry_node_from_modifier(
            self.modifier, "Progress").outputs[0]
        # where the last write left the pen, so the next one can pick it up
        # there instead of drifting towards its own end value from wherever
        # the interpolation happens to be
        self.written = progress

    # ------------------------------------------------------------------
    def _hide_the_pencil(self):
        """Keep the pencil the modifier reads out of the picture.

        Reapplied on every :meth:`write`, because a ``Pencil`` that is made
        to appear - or grown, which appears first - keyframes its own
        visibility back on, and a scene naturally does that *after* building
        the object that reads it. See :func:`hide_from_camera`.
        """
        if not self.hide_pencil:
            return
        for part in self.pencil:
            hide_from_camera(part, self.hide_viewport)

    # ------------------------------------------------------------------
    def write(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
              from_value=None, to_value=None, linear=True, **kwargs):
        """Run the pen across the page.

        Between ``begin_time`` and ``begin_time + transition_time`` the
        ``Progress`` dial goes from 0 to 1 - the blank page to the finished
        drawing - or from ``from_value`` to ``to_value`` when those are given,
        which is how a drawing gets written in instalments (write the first
        half, talk about it, write the rest) or taken back off the page again
        by handing them over the other way round.

        A write that is not told where to start takes over from where the
        last one stopped, and a keyframe goes in at ``begin_time`` to say so.
        Without that keyframe the dial would leave its old value the moment
        the previous write ended and creep towards the new one across the
        pause in between - the pen would carry on writing while nothing is
        supposed to be happening.

        :param linear: keyframe interpolation. A hand that accelerates into
            the page and coasts to a halt is the eased default, and it is
            wrong: writing happens at the speed of writing. Pass ``False``
            for the eased curve.
        :return: when the pen is finished, so it can start the next thing.
        """
        self.appear(begin_time=begin_time, transition_time=0, silent=True)
        self._hide_the_pencil()
        if from_value is None:
            from_value = self.written
        if to_value is None:
            to_value = 1
        self.written = to_value
        ibpy.change_default_value(self.progress, from_value=from_value,
                                  to_value=to_value, begin_time=begin_time,
                                  transition_time=transition_time)
        if linear:
            self.linearize(begin_time * FRAME_RATE)
        return begin_time + transition_time

    def linearize(self, frame=0):
        """Flatten this modifier's keyframes from ``frame`` on.

        Blender 5 layered actions have no ``action.fcurves``;
        ``ibpy.iter_action_fcurves`` is the accessor that still works.
        """
        animation = getattr(self.modifier.get_node_tree(), "animation_data", None)
        if animation is None or animation.action is None:
            return
        for fcurve in ibpy.iter_action_fcurves(animation.action):
            for keyframe in fcurve.keyframe_points:
                if keyframe.co[0] >= frame:
                    keyframe.interpolation = 'LINEAR'
