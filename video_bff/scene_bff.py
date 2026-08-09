"""Blender scenes for the BFF machine — the self-modifying Brainfuck of
Agüera y Arcas et al., *Computational Life* (arXiv:2406.19108).

Companion to ``brainfuck/bff/`` in this repository, which holds the
interpreter, the soup experiment and the notebook. The plan these scenes
implement is ``brainfuck/bff/Youtube.md``.

The centrepiece is :meth:`BffScene.replication`, which animates the paper's
replicator copying itself, and :meth:`BffScene.mechanism`, which explains
*how* — the open item in ``Youtube.md`` §8, now resolved:

    The program reads forwards and writes **backwards** (``head1`` starts at 0
    and ``{`` wraps it to 127), so tape B is filled in reversed. That works
    only because the program is a **palindrome** — which is exactly the
    "tail is the head reversed" shape the paper reports without explaining.

Every number these scenes put on screen comes from :mod:`bff_trace`, which
derives it from the real interpreter and re-asserts the mechanism claims on
every run.

Run with::

    cd video_bff && ../.venv/bin/python scene_bff.py
"""
import os
import sys
from collections import OrderedDict
from math import pi, atan2, ceil, tau

import bpy
import numpy as np

from compositions.compositions import set_alpha_composition, create_glow_composition, create_alpha_over_composition
from geometry_nodes.nodes import Quadrilateral
from mathematics.geometry.coxeter.diagram_to_matrix import letters
from objects.bobject import BObject
from objects.curve import Curve, BezierDataCurve
from objects.derived_objects.person_with_cape import PersonWithCape
from objects.eraser.fields import Force
from objects.logo import logo_curve, Logo, LogoFromInstances
from objects.quadrilateral import BQuadrilateral
from objects.rna_circle import RNACircle
from objects.table import Table

# Allow running as a plain script from inside this folder as well as being
# imported as ``video_bff.scene_bff`` (the workspace convention).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_nodes.modifier_video_brainfuck import (BFFNode, BrainFuckSimpleModifier,
                                                     BrainFuckExtendedModifier, SoupWatcherModifierSingle,
                                                     SoupWatcherModifierSingleStarWars,
                                                     DNAModifier, RNAGridModifier, RNALogoModifier, MorphModifier,
                                                     OutlineMorphModifier, TubeMorphModifier, MovingTapeModifier,
                                                     LifeOnEarthModifier, BrainFuckTransitionModifier,
                                                     EpochCounterModifier, BrainFuckHelloModifier)
from interface import ibpy
from interface.ibpy import Vector
from objects.book import Book
from objects.coordinate_system import CoordinateSystem2
from objects.data import Data
from objects.empties import EmptyCube
from objects.plane import Plane
from objects.tex_bobject import SimpleTexBObject
from perform.scene import Scene
from utils.constants import COLOR_NAMES, COLORS_SCALED, DATA_DIR, FRAME_RATE
from utils.utils import print_time_report, z2vec


# ===========================================================================
# Shared helpers (same conventions as video_hat_tile/scene_hat_tile.py)
# ===========================================================================
def _set_world_background(color="background"):
    """Make camera rays see ``color`` instead of Blender's default grey.

    ``initialize_blender`` builds a world of ``Mix(Factor = Is Camera Ray,
    A = <project background>)`` and leaves **B unconnected**. Since camera rays
    give ``Factor = 1``, the visible background is B's default value -- 0.5
    grey -- while A (black) only ever lights reflections. Every scene that
    calls ``set_hdri_background`` hides this by replacing the whole world, so
    it only shows up in scenes like these that keep the default one.

    ``ShaderNodeMix`` carries one same-named socket per ``data_type`` (VALUE,
    VECTOR, RGBA, ROTATION) simultaneously, and a plain ``inputs['B']`` name
    lookup returns the FLOAT one, which the RGBA path ignores. So the socket
    has to be picked by type -- the same trap documented in
    ``video_hat_tile/scene_hat_tile.py``'s ``_rgba_socket``.
    """
    rgba = COLORS_SCALED[COLOR_NAMES.index(color)]
    for world in bpy.data.worlds:
        if world.node_tree is None:
            continue
        for node in world.node_tree.nodes:
            if node.type == 'MIX' and getattr(node, "data_type", None) == 'RGBA':
                for socket in node.inputs:
                    if socket.name in ("A", "B") and socket.type == 'RGBA':
                        socket.default_value = rgba


def _read_tapes(names):
    """The two csv files of a machine, end to end, as the machine starts.

    The same numbers the ``Import CSV`` nodes read, so that a python run of the
    program can be compared with what the graph puts on screen. The first line
    of each file is its header - blender spends it on the name of the column.

    :param names: the file names, relative to ``DATA_DIR``.
    :return: one list of cell values, the first tape and then the second.
    """
    memory = []
    for name in names:
        with open(os.path.join(DATA_DIR, name + ".csv")) as file:
            rows = [line.strip() for line in file if line.strip()]
        memory += [int(row.split(",")[0]) for row in rows[1:]]
    return memory


def _read_compression(name="compression.csv"):
    """How many bytes the compressed soup takes, epoch by epoch.

    The second, much smaller file ``brainfuck/bff/soup_watcher.py`` writes as
    it runs: a ``;``-delimited csv of ``epoch;bytes`` with one row per epoch,
    ``bytes`` being what the whole soup shrinks to under brotli - the same
    compressor ``soup.py`` measures its complexity with.

    :param name: the file name, relative to ``DATA_DIR``.
    :return: ``(epochs, sizes)``, two lists of ints of the same length.
    """
    with open(os.path.join(DATA_DIR, name)) as file:
        rows = [line.strip() for line in file if line.strip()]
    values = [row.split(";") for row in rows[1:]]
    return [int(v[0]) for v in values], [int(v[1]) for v in values]


def _raw_size(sizes):
    """The *uncompressed* size of the soup in bytes, read out of its own curve.

    The soup is a power-of-two number of 64 byte tapes, so its raw size is a
    power of two; and for as long as it is still noise it does not compress at
    all, so the largest size in the file is that power of two give or take the
    handful of bytes brotli spends on its container. Rounding the maximum to
    the nearest power of two therefore recovers the size of the soup exactly,
    and the plot does not have to be told how many tapes were run.
    """
    return 1 << int(round(np.log2(max(sizes))))


def _transition_epoch(epochs, sizes, window=50):
    """The epoch at which the compressed size falls off its cliff.

    The takeover is not gradual - the steepest ``window`` epochs of the whole
    run drop the size by more than half, two orders of magnitude more than any
    stretch of the noise phase - so the steepest window simply *is* the
    transition and nothing has to be thresholded. Reading it off the data
    rather than writing 8600 into the scene means a re-run of the soup
    re-times the annotation instead of leaving it pointing at empty space.
    """
    drops = [(sizes[i] - sizes[i + window], epochs[i])
             for i in range(len(sizes) - window)]
    return max(drops)[1]


def _nice_step(span, n_tics=5):
    """A round number roughly ``span / n_tics`` big, for tic labels.

    ``CoordinateSystem2``'s automatic tics divide the domain into equal parts
    whatever that does to the numbers (a 288 unit axis in five steps is
    labelled 57.6, 115.2, ...), so axes whose domain is set by the data rather
    than chosen by hand need their tic values rounded to something readable
    first.
    """
    raw = span / n_tics
    power = 10 ** np.floor(np.log10(raw))
    step = power
    for multiple in (2, 2.5, 5, 10):
        if multiple * power <= raw:
            step = multiple * power
    return step


def _tic_labels(step, upper, lower=0):
    """``{latex: value}`` for every multiple of ``step`` up to ``upper``.

    The form ``CoordinateSystem2`` wants when the tics are not left to
    ``'AUTO'``: the key is typeset as the label, the value says where on the
    axis it goes. Zero is left out - the two axes cross there and their labels
    would print on top of each other.
    """
    fmt = "%d" if step == int(step) else "%g"
    return {fmt % v: float(v) for v in np.arange(lower + step, upper, step)}


def _setup_render(hdri=None, engine="BLENDER_EEVEE", transparent=False,
                  exposure=1, background="background", rotation_euler=None, frame_start=1):
    """Render settings.

    ``hdri=None`` (the default here, unlike the hat-tile scenes) keeps the
    default world and paints it flat ``background`` black. These scenes are
    flat, code-like diagrams lit mainly by their own emission; an HDRI sky
    behind them washes out the palette, and with ``simple=True`` that sky is
    exactly what camera rays see.

    ``rotation_euler`` turns the environment. Only the DNA shot cares which way
    round it is - it is the one scene whose subject is lit by the HDRI rather
    than by its own emission - so the default is what the rest of the scenes
    have always used.
    """
    if hdri is not None:
        if rotation_euler is None:
            rotation_euler = [0, 0, 0]
        ibpy.set_hdri_background(hdri, 'exr', rotation_euler=rotation_euler,
                                 simple=True, transparent=transparent)
    elif background is not None:
        _set_world_background(background)
    ibpy.set_render_engine(denoising=False, transparent=transparent,
                           resolution_percentage=100, engine=engine,
                           taa_render_samples=128, frame_start=frame_start,
                           exposure=exposure)


def _light_hero(target=(0, 0, 0), strength=1.0, ambient=0.5):
    """A four-lamp rig for one bright object against a black background.

    The DNA shot used to be lit by an interior HDRI and nothing else, which is
    why it read as a diagram: an environment map is light arriving from
    *everywhere*, so every tube of the helix came back the same grey and the
    near and far sides of a loop were indistinguishable. What tells the eye a
    thing is round and where it is in space is light arriving from *somewhere* -
    and, on black, edges: a subject with no lit outline has no silhouette, it
    simply stops.

    So: key, rim, kick, fill, the standard arrangement, with the two that do
    the most work here being the ones behind the subject.

    ``key``
        Warm, high and to the left, in front. It does the modelling - the
        gradient around each tube from lit to unlit - and it is the only lamp
        casting a shadow the eye is meant to notice, the loop's shadow on the
        run of helix behind it.
    ``rim``
        Cool, high and to the right, *behind*. This is the one that matters.
        The camera sits on -y for the whole shot, so anything at +y is a
        backlight for every frame of it, and a backlit tube gets a bright line
        down its edge. That line is the silhouette, and it is what stops the
        molecule dissolving into the background wherever it is not facing the
        key.
    ``kick``
        Amber, low and to the left, behind. A second edge in the opposite hue,
        under the subject rather than over it, so that the bottom of the helix
        - the part the split happens on - is not left to the fill alone.
    ``fill``
        Broad, dim, cool, from beside the camera. It exists to keep the four
        base colours legible on the shadow side; on black, unlit is invisible,
        and invisible bases would cost the shot the one thing the palette is
        there to say.

    All four are suns rather than lamps with a position, because the molecule
    is thirty-two units wide and a lamp close enough to be soft on the near end
    is half as bright at the far one. A sun has no falloff at all, so the helix
    is lit the same the whole way across the frame, and its ``angle`` - the
    apparent size of the disc, several degrees rather than the half a degree of
    the real sun - is what makes the shadows and the speculars soft anyway.

    The world is not black either. It is a dim cool-above/warm-below gradient,
    invisible to the camera (the film is transparent) and worth its cost purely
    in what the spheres reflect: with a truly black environment every sphere on
    the backbone is a matte dot with one specular pin-prick, and with this it
    has a bright top and a dark underside, which is most of what makes them
    read as spheres at all.

    :param target: what the lamps point at.
    :param strength: scales all four lamps together, for matching this shot to
        the ones on either side of it without disturbing the ratios between
        them - which are the rig.
    :param ambient: scales the environment alone. Lower it towards zero for a
        harder look - at zero the shadow side of the molecule is genuinely
        black - and raise it towards 2 or 3 and the rig dissolves back into the
        flat museum lighting it replaced.
    :return: the four lamps, by name, for a scene that wants to keyframe one.
    """
    for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)

    target = Vector(target)

    def beam(name, location, color, energy, angle):
        """One sun, at ``location`` in the sense of "shining from over there"."""
        bpy.ops.object.light_add(type='SUN', location=location)
        lamp = bpy.context.object
        lamp.name = name
        lamp.data.name = name
        lamp.data.color = color
        lamp.data.energy = energy * strength
        lamp.data.angle = angle
        lamp.data.use_shadow = True
        # a sun ignores where it is and shines along its own -z, so the
        # position above is only a way of writing down a direction
        lamp.rotation_euler = Vector((0.0, 0.0, -1.0)).rotation_difference(
            target - Vector(location)).to_euler()
        return lamp

    # The ratios are the rig and they are not what a first guess produces. A
    # rim six times the key is what a first guess produces, and it renders
    # almost exactly like the HDRI did: four lamps of comparable strength from
    # four directions average back out to light from everywhere. The rim has to
    # be several times the key before it reads as an edge rather than as more
    # brightness, and the fill has to be almost nothing - a tenth of the key -
    # before the shadow side is dark enough for an edge to have anything to be
    # bright against.
    lamps = {
        "KeyLight": beam("KeyLight", (-19.0, -9.0, 12.0),
                         (1.00, 0.87, 0.70), 2.2, 0.14),
        "RimLight": beam("RimLight", (9.0, 22.0, 11.0),
                         (0.40, 0.66, 1.00), 14.0, 0.07),
        # "KickLight": beam("KickLight", (-15.0, 15.0, -8.0),
        #                   (1.00, 0.38, 0.14), 9.0, 0.07),
        "FillLight": beam("FillLight", (7.0, -22.0, -7.0),
                          (0.55, 0.70, 1.00), 0.30, 0.50),
    }

    return lamps


def _setup_standard_camera(distance=20, shift_x=0, shift_z=0):
    """Head-on camera looking along +y — the natural view for a flat tape."""
    ibpy.set_camera_location(location=[shift_x, -distance, shift_z])
    empty = EmptyCube(location=Vector((shift_x, 0, shift_z)))
    ibpy.set_camera_view_to(empty)
    return empty


def _frame_distance(width, height, fill=0.85):
    """How far back a head-on camera has to sit to hold a ``width x height`` box.

    Read off the camera rather than assumed: ``initialize_blender`` fits a
    30 mm lens, not the 50 mm a "standard" lens usually means, and a shot
    placed on the wrong one is out by more than half again. With the sensor
    fitted to the long side (which it is, for a landscape render), the frame
    is ``sensor_width / lens`` across per unit of distance and ``res_y /
    res_x`` of that down, so both constraints are one division apart.

    :param fill: how much of the frame the box should take, in its tighter
        direction - the rest is margin.
    :return: the distance along -y at which the box just fits.
    """
    camera = ibpy.get_camera().data
    scene = ibpy.get_scene().render
    across = camera.sensor_width / camera.lens  # frame width per unit of distance
    down = across * scene.resolution_y / scene.resolution_x
    return max(width / fill / across, height / fill / down)


def _setup_tilted_camera(location=(0, -15, 6.5), target=(0, 0, 0.3)):
    """Camera that looks *down* on the tape, for geometry that stands up.

    A head-on camera can show either a tape lying flat or letters standing on
    it, never both: the two are orthogonal by construction. Tilting the camera
    down splits the difference, at the price that flat text in the x-z plane is
    no longer face-on to it.

    Returns ``(pitch, up, target)``: ``pitch`` is the rotation about x that
    turns such a text back towards the camera, ``up`` is the unit vector along
    which "higher on screen" runs in world space, and ``target`` is what the
    camera looks at. Together they let captions be placed by screen height
    rather than by world z -- see ``BffScene.tape_node``.
    """
    location, target = Vector(location), Vector(target)
    view = (target - location).normalized()
    up = Vector((0, -view.z, view.y))
    pitch = -np.arctan2(location.z - target.z, target.y - location.y)
    ibpy.set_camera_location(location=location)
    empty = EmptyCube(location=target)
    ibpy.set_camera_view_to(empty)
    return pitch, up, target


def _setup_row_camera(width, center_x=0.0, tilt=(0, 5.8, -10), lens=30,
                      bottom=-0.9, aspect=16 / 9):
    """Look down on a flat sheet so that its *first* row fills the frame.

    The sheet lies in the x-y plane with row 0 nearest the camera (see
    ``SoupWatcherModifierSingle``), and what this solves for is the one
    camera position at which that row is exactly as wide as the frame and
    sits just above its bottom edge. Rows behind it recede and eventually
    leave the top of the screen, which is the point of the arrangement: the
    near tape is legible, the rest is context.

    ``tilt`` is the view *direction*, not a location - the shot's angle is
    the thing being preserved here, while the distance and height that go
    with it are derived. It must have no x component, which is what makes
    the camera's right vector world x and so keeps the rows horizontal on
    screen (and lets the row be fitted by width alone: every point of a row
    is then at the same depth).

    :param width: how wide the row is in world units.
    :param center_x: the row's centre, which the camera is placed above.
    :param bottom: where the row's centre lands vertically in normalized
        device coordinates, -1 being the bottom edge of the frame.
    :return: the :class:`EmptyCube` the camera tracks, sitting where the view
        ray meets the sheet.
    """
    view = Vector(tilt).normalized()
    up = -view.cross(Vector((1, 0, 0)))  # right is world x, so this is up
    tan_x = 0.5 * 36 / lens  # 36 mm sensor, fit to the width
    tan_y = tan_x / aspect

    # the depth at which half the frame is half a row wide, and the height at
    # that depth that puts the row `bottom` of the way down the frame
    depth = 0.5 * width / tan_x
    height = bottom * depth * tan_y

    # depth and height are the view- and up-components of (row centre - camera);
    # with the camera directly above the row's centre in x, that is two
    # equations for its y and z
    y, z = np.linalg.solve([[-view.y, -view.z], [-up.y, -up.z]], [depth, height])
    location = Vector((center_x, y, z))

    ibpy.set_camera_lens(lens=lens)
    ibpy.set_camera_location(location=location)
    empty = EmptyCube(location=location - view * (z / view.z))
    ibpy.set_camera_view_to(empty)
    return empty


def _ndc_projector(location, view_direction, lens=30, aspect=16 / 9):
    """Where a world point lands on screen, as a function.

    Returns normalized device coordinates: ``(0, 0)`` is the middle of the
    frame and each edge is at 1, so a shot can be choreographed against the
    frame itself ("stop when this reaches the middle") instead of against
    world coordinates that have to be re-derived by hand whenever the camera
    moves.

    Only valid for a camera aimed along ``view_direction`` with no roll and
    no x component in that direction - the case :func:`_setup_row_camera`
    builds - since it assumes the camera's right vector is world x.

    A point *behind* the camera is reported far below the frame rather than
    projected: the perspective divide flips sign there, which would otherwise
    make a search like :func:`_crawl_distance_at` believe a point that has
    not entered the shot yet is already at the top of it.
    """
    location, view = Vector(location), Vector(view_direction).normalized()
    right = Vector((1, 0, 0))
    up = -view.cross(right)
    tan_x = 0.5 * 36 / lens  # 36 mm sensor, fit to the width
    tan_y = tan_x / aspect

    def project(point):
        offset = Vector(point) - location
        depth = offset.dot(view)
        if depth <= 1e-6:
            return Vector((0, -1e6))
        return Vector((offset.dot(right) / (depth * tan_x),
                       offset.dot(up) / (depth * tan_y)))

    return project


def _crawl_distance_at(project, place, screen_y, lo=-20.0, hi=200.0, steps=80):
    """How far up a crawl something has to be to reach a given screen height.

    ``place`` turns a crawl distance into a world point (see
    ``SoupWatcherModifierSingleStarWars.crawl_position``). A crawl runs from
    the bottom of the frame towards its vanishing point and its screen
    height increases the whole way, so the distance can simply be bisected -
    which is how the shot times its beats off the frame rather than off a
    stopwatch.
    """
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if project(place(mid)).y < screen_y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# The opening shot: framing a sentence that turns into a planet
# ===========================================================================
# How much room `LifeOnEarthModifier`'s flat sentence takes, at its defaults
# (`Life on` and `Earth` set in Bfont at size 2, the second word 5.8 further
# along the baseline). Measured off the evaluated tree rather than derived:
# how wide a word is, is a property of the font, and nothing in the graph
# knows it. Only `how_on_earth` uses them, and only to place the camera - a
# different font or text size wants them measured again.
SENTENCE_WIDTH = 10.05
SENTENCE_HEIGHT = 1.45


def _linearize_from(obj, frame):
    """Make every keyframe from ``frame`` on interpolate linearly.

    ``BObject.rotate(interpolation=...)`` cannot do this on Blender 5: it
    reaches for ``action.fcurves``, which layered actions no longer have.
    :func:`ibpy.iter_action_fcurves` is the accessor that works, and going
    through it by hand also keeps the earlier keyframes of the same object
    (the write-on, the morph) on their eased default.
    """
    for holder in (obj, getattr(obj, "data", None)):
        anim = getattr(holder, "animation_data", None)
        if anim is None or anim.action is None:
            continue
        for fcurve in ibpy.iter_action_fcurves(anim.action):
            for keyframe in fcurve.keyframe_points:
                if keyframe.co[0] >= frame:
                    keyframe.interpolation = 'LINEAR'


# ===========================================================================
#  A temple, drawn rather than built
# ===========================================================================
TEMPLE = dict(bay=5.5, n_bays=5, aisle=5.0, height=12.0, radius=0.9,
              step=0.45, tread=0.9)


def _frame(axis):
    """A right-handed frame with ``axis`` as its third vector.

    Rings and tubes both need somewhere to put a circle, and the only thing
    that matters about the two perpendicular directions is that they are
    perpendicular -- which is why they can be picked off a fixed reference
    and never have to be passed in.
    """
    axis = Vector(axis).normalized()
    reference = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))
    u = axis.cross(reference).normalized()
    return axis, u, axis.cross(u).normalized()


def _pen(a, b, rng, bow=0.012, overshoot=0.02, jitter=0.008, samples=7, via=None):
    """One straight-ish stroke of a pen from ``a`` to ``b``.

    A ruled line reads as CAD. Three things make a line read as *drawn*
    instead: it bows a little between its ends, it wobbles on the way, and
    the hand does not stop exactly on the corner. The first two are
    relative to the length of the stroke, so a thirty-unit beam and a
    one-unit tick come out of the same hand.

    The bow is a half sine and therefore vanishes at both ends: corners
    still meet, even though nothing in between is straight. Which
    direction it bows in is picked per stroke out of the two directions
    perpendicular to it -- in three dimensions a stroke can sag in any of
    them, and picking one per stroke is what keeps a colonnade from
    looking like one column copied twelve times.

    ``via`` bends the stroke: given a control point it is a quadratic bezier
    from ``a`` to ``b`` instead of a line, which is how anything genuinely
    curved gets drawn -- the arm of a microscope, say. A bent stroke keeps
    the wobble but loses the overshoot: pushing the ends out along the chord
    would drag the curve's tangents out with them.
    """
    a, b = Vector(a), Vector(b)
    span = b - a
    length = span.length
    direction = span / length
    if via is None:
        a = a - direction * length * overshoot * rng.uniform(0.2, 1.0)
        b = b + direction * length * overshoot * rng.uniform(0.2, 1.0)
    else:
        via = Vector(via)

    _, n1, n2 = _frame(direction)
    sag1, sag2 = rng.normal(0, bow * length, 2)

    points = []
    for i in range(samples):
        t = i / (samples - 1)
        arc = np.sin(np.pi * t)
        if via is None:
            point = a.lerp(b, t)
        else:
            point = (1 - t) ** 2 * a + 2 * (1 - t) * t * via + t ** 2 * b
        point = point + n1 * (sag1 * arc + rng.normal(0, jitter))
        point = point + n2 * (sag2 * arc + rng.normal(0, jitter))
        points.append(point)
    return points


def _ring(center, radius, rng, gap=0.09, samples=17, axis=(0, 0, 1)):
    """A drawn circle: not quite round, not quite closed, not necessarily flat.

    The gap is the point of it. A closed ellipse is a machine part; a ring
    whose ends miss each other by a few degrees is a wrist.

    ``axis`` is what the circle is perpendicular to, and the default keeps
    it lying in the ground plane. Everything that is a barrel rather than a
    tower needs the other cases: the eyepiece of a microscope is a ring
    across a tube that leans, and the focus wheel is a ring on its side.
    """
    center = Vector(center)
    axis, u, v = _frame(axis)
    start = rng.uniform(0, tau)
    points = []
    for i in range(samples):
        t = i / (samples - 1)
        angle = start + tau * (1 - gap) * t
        r = radius * (1 + rng.normal(0, 0.02))
        points.append(center + u * (r * np.cos(angle)) + v * (r * np.sin(angle))
                      + axis * rng.normal(0, 0.01))
    return points


def _ink(points, name, **kwargs):
    """One stroke, as an object that can be grown.

    :class:`BezierDataCurve` takes ``name`` for the curve *data* and pops it
    before :class:`BObject` ever sees it, so the object itself ends up called
    ``b_object`` -- and ``ibpy.get_curve_for_b_object``, which every ``grow``
    goes through, looks the data up under the *object's* name. Handing both
    the same name is the whole job here, and without it a stroke cannot be
    drawn on at all.
    """
    stroke = BezierDataCurve(data=points, name=name, make_pieces=False, **kwargs)
    stroke.ref_obj.name = name
    stroke.ref_obj.data.name = name
    return stroke


def _temple_drawing(rng, geo=TEMPLE):
    """The whole temple as a list of pen strokes, in no particular order.

    Every stroke carries the two things the choreography needs:

    * ``part`` -- what it belongs to (``ground``, ``column``, ``beam``,
      ``roof``, ``altar``), which decides how it is drawn and in what
      colour;
    * ``x`` -- where along the aisle it stands, which decides *when*: the
      pen works a fixed distance ahead of the walker, so the temple is
      always being drawn just beyond them and is never waiting for them.

    ``sweep`` marks the few strokes that run the length of the building.
    Those are not drawn bay by bay but in one long pull that outruns the
    figure -- the lines that race away to the vanishing point are what
    sell a colonnade, and they only do it if the eye can follow them
    going.

    Nothing here is solid. The columns are five tapering flutes and two
    rings, which is a column from every angle without ever being a
    cylinder, and the roof is rafters and a ridge with nothing on them.
    A drawing does not have to enclose anything to be a building.
    """
    bay, n_bays = geo['bay'], geo['n_bays']
    aisle, height, radius = geo['aisle'], geo['height'], geo['radius']
    step, tread = geo['step'], geo['tread']

    length = n_bays * bay
    columns_x = [k * bay for k in range(n_bays + 1)]
    x0, x1 = -0.6 * bay, length + 0.6 * bay
    strokes = []

    def add(points, part, x, order=0, sweep=False):
        strokes.append(dict(points=points, part=part, x=x, order=order,
                            sweep=sweep))

    # --- the stylobate ---------------------------------------------------
    # three steps, drawn as the six long lines they read as from inside: the
    # eye takes parallel lines that recede and drop as a stair without ever
    # being shown a riser.
    for j in range(3):
        y, z = aisle + 1.6 + j * tread, -j * step
        xa, xb = x0 - j * tread, x1 + j * tread
        for sign in (1, -1):
            add(_pen((xa, sign * y, z), (xb, sign * y, z), rng, samples=14),
                'ground', xa, order=j, sweep=True)
        add(_pen((xa, -y, z), (xa, y, z), rng), 'ground', xa, order=j)
        add(_pen((xb, -y, z), (xb, y, z), rng), 'ground', xb, order=j)

    # --- the colonnade ---------------------------------------------------
    for x in columns_x:
        for sign in (1, -1):
            foot = Vector((x, sign * aisle, 0))
            phase = rng.uniform(0, tau)
            for i in range(5):
                angle = phase + tau * i / 5
                bottom = foot + Vector((radius * np.cos(angle),
                                        radius * np.sin(angle), 0))
                # the taper is what stops a column being a pipe
                top = foot + Vector((0.82 * radius * np.cos(angle),
                                     0.82 * radius * np.sin(angle), height))
                add(_pen(bottom, top, rng, samples=9), 'column', x, order=i)
            add(_ring(foot + Vector((0, 0, 0.12)), 1.25 * radius, rng),
                'column', x, order=5)
            add(_ring(foot + Vector((0, 0, height + 0.05)), 1.35 * radius, rng),
                'column', x, order=6)

    # --- the entablature -------------------------------------------------
    # four long edges a side, plus a tick over every column: the beam
    # inherits the rhythm of what holds it up.
    for sign in (1, -1):
        for dy in (-0.5, 0.5):
            for z in (height + 0.25, height + 1.9):
                add(_pen((x0, sign * (aisle + dy), z),
                         (x1, sign * (aisle + dy), z), rng, samples=14),
                    'beam', x0, sweep=True)
        for x in columns_x:
            add(_pen((x, sign * (aisle + 0.5), height + 0.4),
                     (x, sign * (aisle + 0.5), height + 1.75), rng),
                'beam', x, order=2)

    # --- the roof --------------------------------------------------------
    ridge = height + 4.6
    for x in columns_x:
        for sign in (1, -1):
            add(_pen((x, sign * (aisle + 0.5), height + 1.9), (x, 0, ridge),
                     rng, samples=7), 'roof', x, order=1)
    add(_pen((x0, 0, ridge), (x1, 0, ridge), rng, samples=14),
        'roof', x0, sweep=True)
    # the far gable: the two rafters of the last bay and this line close a
    # triangle, and a triangle at the end of an aisle is a destination.
    add(_pen((length, -(aisle + 0.5), height + 1.9),
             (length, aisle + 0.5, height + 1.9), rng, samples=9),
        'roof', length, order=2)

    # --- what the walk is for --------------------------------------------
    # an altar, if a drum of six strokes on the axis is an altar, with a ring
    # hanging over it that never touches anything.
    ax = length + 3.4
    for i in range(6):
        angle = tau * i / 6 + 0.2
        add(_pen((ax + 1.3 * np.cos(angle), 1.3 * np.sin(angle), 0),
                 (ax + 1.15 * np.cos(angle), 1.15 * np.sin(angle), 1.8), rng),
            'altar', ax, order=i)
    add(_ring((ax, 0, 0.05), 1.45, rng), 'altar', ax, order=6)
    add(_ring((ax, 0, 1.85), 1.35, rng), 'altar', ax, order=7)
    add(_ring((ax, 0, 3.4), 1.0, rng, gap=0.03), 'altar', ax, order=9)

    return strokes


# ===========================================================================
#  A microscope, drawn the same way, and big enough to walk around
# ===========================================================================
MICROSCOPE = dict(arena=7.4, stage=4.25, height=10.0, axis=-0.2)


def _microscope_drawing(rng):
    """The instrument as a list of pen strokes, in the same hand as the temple.

    It is ten units tall, which is about three times a person: at the size
    of a real microscope it would be a speck between the two figures, and
    the point of the shot is that they are small and it is not. Blown up
    that far it stops being a tool and becomes the thing in the middle of
    the circle -- which is what the walk around it is about.

    Every stroke carries its height in ``z``, and that is the whole
    choreography: the drawing is scheduled bottom to top, so the foot is
    down before the arm curves up out of it and the eyepiece is the last
    thing in the air. An instrument assembles from the bench upwards; a
    drawing of one may as well admit it.

    The exception is the specimen on the stage, which is drawn last of all
    however low it sits. It is the only warm thing in the picture and the
    only thing the instrument is for.
    """
    stage_z = MICROSCOPE['stage']
    ax_y = MICROSCOPE['axis']  # the optical axis sits forward of centre
    strokes = []

    def add(points, part='body', z=None, sweep=False):
        if z is None:
            z = sum(p.z for p in points) / len(points)
        strokes.append(dict(points=points, part=part, z=z, sweep=sweep))

    # --- the arena -------------------------------------------------------
    # two rings on the floor: the circle the figures will walk, drawn before
    # there is anything in the middle of it to walk around
    for r in (MICROSCOPE['arena'] - 0.3, MICROSCOPE['arena'] + 0.35):
        add(_ring((0, 0, 0.01), r, rng, gap=0.015, samples=44), 'arena', z=0,
            sweep=True)

    # --- the foot --------------------------------------------------------
    add(_ring((0, 0.15, 0.02), 3.0, rng, samples=30), z=0.0)
    add(_ring((0, 0.15, 0.62), 2.45, rng, samples=30), z=0.62)
    for i in range(7):
        angle = tau * i / 7 + 0.3
        add(_pen((3.00 * np.cos(angle), 0.15 + 3.00 * np.sin(angle), 0.02),
                 (2.45 * np.cos(angle), 0.15 + 2.45 * np.sin(angle), 0.62), rng))

    # --- the arm ---------------------------------------------------------
    # the one curved thing in the instrument, and the reason _pen learned to
    # bend: it leaves the foot going up, bulges backwards to clear the stage
    # and comes back in to take the tube
    for dx in (-0.75, 0.0, 0.75):
        add(_pen((dx, 1.5, 0.55), (0.7 * dx, 1.9, 6.4), rng,
                 via=(dx, 3.6, 3.4), samples=15, bow=0.004))
    for dx in (-0.55, 0.55):
        add(_pen((dx, 1.9, 6.3), (dx, 0.45, 6.05), rng))
    # the focus wheels, on their sides on the outside of the bulge
    for dx in (-1.05, 1.05):
        add(_ring((dx, 2.75, 3.4), 0.95, rng, samples=22, axis=(1, 0, 0)))
        add(_ring((dx, 2.75, 3.4), 0.5, rng, samples=16, axis=(1, 0, 0)))

    # --- the stage -------------------------------------------------------
    x0, x1 = -2.3, 2.3
    y0, y1 = ax_y - 1.7, ax_y + 1.7
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        add(_pen((p[0], p[1], stage_z), (q[0], q[1], stage_z), rng, samples=9),
            z=stage_z)
    add(_ring((0, ax_y, stage_z + 0.02), 0.62, rng, samples=20), z=stage_z)
    for dx in (-1.0, 1.0):  # the clips that hold a slide down
        add(_pen((dx, y0 + 0.4, stage_z + 0.06),
                 (0.55 * dx, ax_y - 0.5, stage_z + 0.06), rng), z=stage_z)
        add(_pen((dx, y1, stage_z), (dx, 2.6, stage_z - 0.25), rng), z=stage_z)

    # --- the mirror, tilted up under the stage ---------------------------
    add(_ring((0, ax_y + 0.15, 2.05), 1.05, rng, samples=24,
              axis=(0, -0.55, 0.84)), z=2.05)
    add(_pen((0, ax_y + 0.15, 2.05), (0, 1.5, 1.15), rng, samples=5), z=1.7)

    # --- the nosepiece and the objective ---------------------------------
    nose_z = 5.05
    add(_ring((0, ax_y, nose_z), 0.95, rng, samples=22), z=nose_z)
    for i in range(4):
        angle = tau * i / 4 + 0.4
        add(_pen((0.50 * np.cos(angle), ax_y + 0.50 * np.sin(angle), nose_z - 0.05),
                 (0.20 * np.cos(angle), ax_y + 0.20 * np.sin(angle), stage_z + 0.35),
                 rng, samples=5))
    add(_ring((0, ax_y, stage_z + 0.32), 0.22, rng, samples=14))

    # --- the tube --------------------------------------------------------
    # five lines and two rings, the same recipe as a column of the temple,
    # only leaning back: an eyepiece has to be somewhere a head can get to
    bottom = Vector((0, ax_y, nose_z + 0.05))
    top = Vector((0, ax_y + 2.3, 9.4))
    axis, u, v = _frame(top - bottom)
    for i in range(5):
        angle = tau * i / 5 + 0.7
        radial = u * np.cos(angle) + v * np.sin(angle)
        add(_pen(bottom + radial * 0.52, top + radial * 0.46, rng, samples=9))
    add(_ring(bottom, 0.55, rng, samples=18, axis=axis))
    add(_ring(top, 0.5, rng, samples=18, axis=axis))
    # the eyepiece: a wider drum stuck on the end of it
    eye = top + axis * 0.6
    for i in range(5):
        angle = tau * i / 5 + 0.7
        radial = u * np.cos(angle) + v * np.sin(angle)
        add(_pen(top + radial * 0.48, eye + radial * 0.72, rng, samples=5))
    add(_ring(eye, 0.74, rng, samples=20, axis=axis))

    # --- what it is all pointed at ---------------------------------------
    spot = Vector((0, ax_y, stage_z + 0.07))
    for dx in (-1, 1):
        add(_pen((dx * 0.2, ax_y + dx * 0.06, stage_z + 0.33),
                 (dx * 0.34, ax_y + dx * 0.1, stage_z + 0.07), rng, samples=5),
            'specimen', z=stage_z)
    add(_ring(spot, 0.36, rng, gap=0.03, samples=18), 'specimen', z=stage_z)

    # A microscope is a profile. Everything above is built facing the camera,
    # because that is the frame the numbers are easiest to think in -- and
    # head-on is the one view in which it is not a microscope: the arm hides
    # behind the tube, the wheels hide behind the arm, and the lean that makes
    # an eyepiece an eyepiece points straight away from the lens. So the whole
    # instrument turns a quarter before it is handed over. The arena does not
    # turn: a circle is a circle, and rotating it would only re-roll its gap.
    for stroke in strokes:
        if stroke['part'] != 'arena':
            stroke['points'] = [Vector((p.y, -p.x, p.z)) for p in stroke['points']]

    return strokes


# ===========================================================================
class BffScene(Scene):
    def __init__(self):
        self.t0 = 0
        self.sub_scenes = OrderedDict([
            ('bf_overview', {'duration': 10}),
            ('dna_flyby', {'duration': 45}),
            ('rna_grid', {'duration': 16}),
            ('rna_logo', {'duration': 20}),
            ('paper', {'duration': 28}),
            ('soup_watcher', {'duration': 167}),
            ('close_up', {'duration': ceil(2000 / FRAME_RATE)}),
            ('close_up_star_wars', {'duration': 21}),
            ('simple_brain_fuck', {'duration': 22}),
            ('simple_brain_fuck2', {'duration': 30}),
            ('simple_brain_fuck3', {'duration': 30}),
            ('simple_brain_fuck4', {'duration': 30}),
            ('brain_fuck_extended', {'duration': 200}),
            ('plot', {'duration': 40}),
            ('soup_watcher_star_wars', {'duration': 60}),
            ('morphing', {'duration': 12}),
            ('moving_tape', {'duration': 13}),
            ('how_on_earth', {'duration': 30}),
            ('temple_person', {'duration': 15}),
            ('microscope_person', {'duration': 18}),
            ('frame', {'duration': 1}),
            ('bf_to_bff', {'duration': 20}),
            ('epoch_counter', {'duration': 167}),
            ('epoch_counter2', {'duration': ceil(2000/60)}),
            ('hello_extended', {'duration': 24}),
        ])
        super().__init__(light_energy=1, transparent=False)

    def bf_overview(self):
        t0 = 0
        _setup_render(hdri="cayley_interior_4k",transparent=True)

        title = SimpleTexBObject(r"\text{B**** F***}",text_size="Huge",color="example",
                                 location = [0,0,6],aligned="center")

        t0 = 0.5+title.write(begin_time=t0,transition_time=0.5)

        table_strings = [
            "+", "-", r"\text{change value}",
                      "<", ">", r"\text{move position}",
                                ".", ",", r"\text{read write}",
                                          "[", "]", r"\text{repeat}",
            r"\{",r"\}",r"\text{move position 2}",
            "0","1",r"\text{stop}"
        ]

        colors = ["important","orange","text",
                  "joker","joker","text",
                  "some_logo_blue","some_logo_blue","text",
                  "x14_color","x14_color","text",
                  "custom1","custom1","text",
                  "text","text","text"]
        table_data = [SimpleTexBObject(string,color=color) for string,color in zip(table_strings,colors)]
        table_data = np.resize(table_data, (6, 3))
        btable = Table(table_data,scale=4,alignment=["c","c","l"],
                       bufferx=0.8,buffery=0.375,location=[0.63,0,1])

        for row in range(2):
            for col in range(2):
                t0 = btable.write_entry(row,col,begin_time=t0,transition_time=0.05)
        t0 = 0.5 + btable.write_entry(2,0,begin_time=t0,transition_time=0.05)

        t0 = 0.5 + btable.write_entry(0,2,begin_time=t0,transition_time=0.3)
        t0 = 0.5 + btable.write_entry(1,2,begin_time=t0,transition_time=0.3)
        t0 = 0.5 + table_data[2,2].write(letter_set=[0,1,2,3],begin_time=t0,transition_time=0.15)
        t0 = btable.write_entry(3,0,begin_time=t0,transition_time=0.05)
        t0 = 0.5 + btable.write_entry(3,1,begin_time=t0,transition_time=0.05)
        t0 = 0.5 + btable.write_entry(3,2,begin_time=t0,transition_time=0.3)
        t0 = 0.5 + btable.write_entry(2,1,begin_time=t0,transition_time=0.05)
        t0 = 0.5 + table_data[2,2].write(letter_set=[4,5,6,7,8],begin_time=t0,transition_time=0.15)
        t0 = 0.5 + btable.write_row(4,begin_time=t0,transition_time=0.3)
        t0 = btable.write_entry(5,0,begin_time=t0,transition_time=0.05)
        t0 = 0.5 + btable.write_entry(5,2,begin_time=t0,transition_time=0.15)
        self.t0 = t0

    def frame(self):
        t0 = 0
        _setup_render(hdri="cayley_interior_4k", transparent=True)
        create_glow_composition(threshold=1, strength=1, saturation=1, size=5, tint=Vector([0.9, 1, 0.6, 1]))

        frame = BQuadrilateral(vertices=[
            Vector([-12, 0, 6.7]),
            Vector([12, 0, 6.7]),
            Vector([12, 0, -6.7]),
            Vector([-12, 0, -6.7])
        ], color="example", emission=1, resolution=1000, thickness=8, scale=0.5)

        frame.grow(begin_time=t0, transition_time=0)
        self.t0 = 0

    # -------------------------------------------------------------------
    def how_on_earth(self):
        """The very first shot: the sentence writes itself, then becomes the planet.

        Covers the opening paragraph of ``script.md`` -- "How did life emerge
        on Earth? I mean, how on Earth did life emerge?" -- and it needs a
        picture that is in on the joke rather than a screenshot of an
        encyclopaedia article.

        The whole shot is one geometry-node tree,
        :class:`LifeOnEarthModifier`, ported from ``tmp.xml``. Three beats:

        1. ``Life on Earth`` is written out, one letter growing in at a time.
        2. a beat to read it.
        3. every glyph outline inflates into a circle on a sphere - the
           letters of ``Life on`` into the meridians, the letters of ``Earth``
           into the parallels - and an ocean-coloured ball arrives underneath
           them just as they settle. The sentence *is* the graticule; nothing
           is swapped for anything.

        Which is the handoff the video needs: what the shot ends on is a
        planet, and the next one is what is floating in its water.

        **The quarter turn.** The tree builds the sentence in the x-y plane
        (which is where ``String to Curves`` puts text) and puts the poles of
        its sphere on **y**. Turning the carrier by ninety degrees about x
        stands the sentence up in the x-z plane, facing the camera on -y, and
        stands the globe's axis upright at the same time - one rotation
        settles both, which is why the modifier does not try to settle either.

        **The camera does the framing the geometry does not.** The sentence
        runs from the origin out to about ten units along x while the globe is
        centred on the origin with a radius of ten, so no single camera
        position holds both: a shot wide enough for the planet leaves the
        lettering a quarter of the frame wide and unreadable. So the camera
        starts on the sentence and pulls back onto the globe while the letters
        are in flight - the reveal and the transformation are one move.

        **Why the globe turns at the end.** Partly because a planet that does
        not is a picture of one, and partly for ``render_with_skips``: the
        whole animation lives in ``Scene Time`` inside the node tree and
        keyframes nothing, so without one genuinely animated object every
        frame would be copied from its predecessor. The camera move covers the
        first half of the shot; the spin covers the rest. Same reason the
        machine scenes creep - see :meth:`_run_simple_bff`.
        """
        duration = self.sub_scenes["how_on_earth"]["duration"]
        _setup_render(hdri="cayley_interior_4k", transparent=True)

        create_glow_composition(threshold=1, strength=1, saturation=1, size=5, tint=Vector([0.9, 1, 0.6, 1]))

        # the lettering and the ball are emissive (see LifeOnEarthModifier's
        # `letter_emission`), so the rig is only here to model the tubes of the
        # graticule against the ball behind them
        _light_hero(target=(0, 0, 0), strength=0.55, ambient=0.35)

        # the timings are the modifier's own defaults, which are the editor's
        # control panel: this shot is the tree's shot, and a duplicate set of
        # numbers here would only be a second place to have to change them.
        # What the scene still has to know is *when* things happen, which is
        # what `timeline()` hands back
        modifier = LifeOnEarthModifier(letter_emission=1, emission=1, globe_color="image", src="earth.jpg",
                                       coordinates="Object", projection="SPHERE", rotation=Vector([pi / 2, 0, 0]))

        written, transform_begin, globe_in, transform_end = modifier.timeline()

        # the quarter turn about x that stands the sentence up and the globe's
        # axis with it - see the docstring
        machine = Plane(name="LifeOnEarth", rotation_euler=[pi / 2, 0, 0])
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)

        tube_radius_node = ibpy.get_geometry_node_from_modifier(modifier, label="TubeRadius")
        ibpy.change_default_value(tube_radius_node, from_value=0.03, to_value=0.01, begin_time=transform_begin,
                                  transition_time=transform_end - transform_begin)
        # the sentence is wide and flat, so what frames it is the width of the
        # shot; the globe is as tall as it is wide, so what frames that is the
        # height. Both distances come off the camera itself - see
        # :func:`_frame_distance`
        near = _frame_distance(SENTENCE_WIDTH, SENTENCE_HEIGHT)
        far = _frame_distance(4 * modifier.radius, 4 * modifier.radius)
        empty = _setup_standard_camera(distance=near, shift_x=0.5 * SENTENCE_WIDTH,
                                       shift_z=0.5 * SENTENCE_HEIGHT)

        # the pull-back starts with the letters and lands a beat after they do,
        # so the ball is already there when the move settles
        pull_time = transform_end - transform_begin + 0.5
        ibpy.camera_move([-0.5 * SENTENCE_WIDTH, -(far - near),
                          -0.5 * SENTENCE_HEIGHT],
                         begin_time=transform_begin, transition_time=pull_time)
        # the camera tracks an empty, so the empty has to walk to the middle of
        # the globe as well or the shot ends aimed at where the sentence was
        empty.move([-0.5 * SENTENCE_WIDTH, 0, -0.5 * SENTENCE_HEIGHT],
                   begin_time=transform_begin, transition_time=pull_time)

        # `rotation_euler` is applied x, then y, then z, so the z term turns
        # the already-upright globe about its own axis
        spin_begin = transform_end
        spin_time = duration - spin_begin - 0.5
        spin_angle = tau
        machine.rotate(rotation_euler=[pi / 2, 0, spin_angle],
                       begin_time=spin_begin, transition_time=spin_time)
        _linearize_from(machine.ref_obj, int(spin_begin * FRAME_RATE))

        machine.change_alpha(from_value=0.0, to_value=1.0, begin_time=9, transition_time=1, slot=2)

        self.t0 = spin_begin + spin_time
        print("how_on_earth: sentence written by %.1fs, wraps %.1fs..%.1fs "
              "(ball from %.1fs), camera pulls %.1f -> %.1f, globe turns "
              "%.0f degrees to %.1fs"
              % (written, transform_begin, transform_end, globe_in, near, far,
                 spin_angle * 180 / pi, self.t0))

    # -------------------------------------------------------------------
    def dna_flyby(self):
        """The opening shot: the molecule flies, the camera does not.

        Covers the paragraph of ``script.md`` that begins "Everyone is familiar
        with this kind of molecule" — the beat that spends no words explaining
        what DNA is, and so needs the picture to do all of it.

        The camera is nailed down. What moves is the molecule, which slides
        along a fixed track (:func:`dna_flight_path`) by having its
        ``HeadOffset`` animated: point *i* of the helix sits at arc length
        ``HeadOffset - i * Spacing`` along the track, so raising that one number
        pulls the whole thing forwards and the shape it takes on the way is the
        shape of the track. That is the time-dependent modulation the sine used
        to be — and unlike a sine it can loop, because a track is parametric
        and comes back over itself where a function of x cannot.

        It flies in from the right, turns a full roller-coaster loop in the
        upper half, runs on down to the left, and — before it reaches the
        border — swings through a 180 degree turn that drops it into the lower
        half heading right again. The loop steps back in y as it goes, so where
        it crosses its own entry on screen the two are really ``TRACK_DEPTH``
        apart in depth and nothing flies through itself.

        Then it unzips — and not all at once. ``StrandSeparation``,
        ``TiltLength`` and ``BaseSize`` are keyframed here as before, but what
        they set is only what the molecule looks like *where it is open*; the
        modifier's ``Unzipping Gate`` decides where that is, from each base
        pair's own y. The track climbs past the gate only once it is out of the
        turn, so the fork stands still at the bottom left of the frame and the
        molecule unzips itself by flying through it, wound on one side and open
        on the other. ``PeelHeight`` is gated with everything else, so the
        second strand climbs out of the top of the frame *from the fork* rather
        than lifting off along its whole length at once — which is the
        replication fork the rest of the video is going towards, and is also
        the only way a 26 unit lift can be applied to a molecule that is still
        half double helix without tearing it in two.

        The beats are timed against the track's own landmarks rather than
        against numbers written here, so retuning the path cannot silently put
        the split in the middle of the turn.
        """
        duration = self.sub_scenes["dna_flyby"]["duration"]
        _setup_render(hdri="cayley_interior_4k", transparent=True, frame_start=200)
        # aimed a little way into the screen rather than at the origin, because
        # that is where the molecule spends the shot: the track runs from y = 0
        # to y = 9, and the half of it the split happens on is the far half
        lamps = _light_hero(target=(0, 2, 0))
        set_alpha_composition()
        modifier = DNAModifier()
        molecule = Plane(name="DNA")
        molecule.add_mesh_modifier(type='NODES', node_modifier=modifier)
        molecule.appear(begin_time=0, transition_time=0)

        # A 40 mm lens 36 units back frames x in [-16.2, 16.2] and z in
        # [-9.1, 9.1], which is what the track was drawn to fit.
        ibpy.set_camera_location(location=Vector((13.5, -17, 14.5)))
        camera_empty = EmptyCube(location=Vector((0, 0, 2.25)))
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=150, clip_end=2000)

        ibpy.camera_zoom(lens=38, begin_time=150 / FRAME_RATE, transition_time=2)
        ibpy.camera_move(shift=[-13.5, -15, -14.5], begin_time=13, transition_time=2)
        camera_empty.move(direction=[0, 0, -2.25], begin_time=13, transition_time=2)

        def dial(label):
            """The output socket of one of the control frame's Value nodes."""
            return ibpy.get_geometry_node_from_modifier(
                modifier, label).outputs[0]

        head = dial("HeadOffset")
        marks = modifier.track_marks
        distances = modifier.distances
        molecule_length = modifier.molecule_length
        total_distance = distances["end_point"]

        # Where the head has to be for each beat to have happened. Adding the
        # molecule's own length to a landmark is the difference between "the
        # nose has got there" and "all of it has".
        beats = [
            (0.0, 0),  # off screen, right
            (duration * distances["loop_start"] / total_distance, marks["loop_start"] + 11.0),  # nose round the loop
            (duration * distances["loop_end"] / total_distance, marks["loop_end"] + 5.0),  # tail out of the loop
            (duration * distances["turn_start"] / total_distance, marks["turn_start"] + 8.0),  # deep in the turn
            (duration * distances["turn_end"] / total_distance, marks["turn_end"] + 5.0),  # out of it, heading right
            (duration * distances["turn_end"] / total_distance + 1, marks["turn_end"] + 8.0),  # fully straight
            # just far enough that the tail crosses the right border as the
            # shot ends. Overshooting here empties the frame early, because
            # everything past the end of the track piles up at the end of it.
            (duration, modifier.track_length + molecule_length + 2.0),  # gone
        ]
        # Each beat is a transition from the one before. Only the first call
        # writes a starting key; after that ``from_value=None`` leaves the
        # previous beat's end key alone instead of stamping a duplicate on top
        # of it, which would flatten the motion to a stop at every beat.
        previous_t, previous_v = beats[0]
        for when, value in beats[1:]:
            ibpy.change_default_value(head, previous_v, value,
                                      begin_time=previous_t,
                                      transition_time=when - previous_t)
            previous_t, previous_v = when, None

        # --- the split -------------------------------------------------
        # These no longer *are* the split - the gate is - they arm it. Until
        # they have moved, the open state and the wound state are the same
        # molecule and the gate has nothing to crossfade, so the first
        # seventeen seconds are wound wherever the track has taken them. From
        # here on the gate has two different things to choose between, and the
        # unzipping is where the molecule is rather than when. The window is
        # still where it was: the head reaches the turn's exit at 17, which is
        # the first moment there is anything past the gate to open.
        split_start, split_end = 17.0, 20.5
        # 3.2 puts the two strands at +-1.6 about a track that runs at
        # ``TRACK_Z_OUT``, so the pair spans z in [-2.6, 0.6] - a band across
        # the middle of the frame with the whole of the top half free for the
        # one that leaves
        ibpy.change_default_value(dial("StrandSeparation"),
                                  modifier.strand_separation, 3.2,
                                  begin_time=split_start,
                                  transition_time=split_end - split_start)
        # 600 rather than a mere 60. At 60 the "unwound" molecule still makes
        # half a turn over its two hundred pairs, which puts the strand that is
        # about to be peeled *underneath* its partner over the back half of it,
        # and peeling lifts in world z - so that half would be dragged up
        # through the strand it is supposed to be leaving behind. At 600 the
        # residual lean is twenty degrees over the whole molecule, which still
        # reads as a molecule rather than a ladder and never crosses.
        ibpy.change_default_value(dial("TiltLength"), modifier.tilt_length,
                                  600.0, begin_time=split_start,
                                  transition_time=split_end - split_start)
        # # The bases shrink with the twist, but not to nothing: left as stubs
        # # they read as unpaired bases on a single strand, which is both what
        # # actually happens at a replication fork and the only thing keeping the
        # # four colours on screen once the pairing is gone.
        # ibpy.change_default_value(dial("BaseSize"), modifier.base_size, 0.18,
        #                           begin_time=split_start,
        #                           transition_time=split_end - split_start - 0.5)

        # --- one strand leaves ----------------------------------------
        # Eight, and the ceiling is what picks it. The peeled strand ends up at
        # ``TRACK_Z_OUT + 1.6 + PeelHeight``; the frame at the depth the fork
        # sits at - y = TRACK_Y_OPEN, so 41 units from a 40 mm lens - reaches
        # z = 10.5. Ten puts the strand at 10.7, which is over the edge by two
        # tenths: not gone, just cut, with its bases left hanging off the top
        # of the frame for the last third of the shot. Eight leaves it a clear
        # two units below the edge, which is what the shot wants now anyway -
        # the point of moving the fork up to the middle was to be able to *see*
        # the two strands come apart, and a strand that has left the frame is
        # not a strand you can see the separation of.
        #
        # It is applied through the gate, so it reaches only as far back along
        # the molecule as the fork has, and the strand climbs away from the
        # fork instead of the whole of it rising at once.
        ibpy.change_default_value(dial("PeelHeight"), 0.0, 11.0,
                                  begin_time=split_end, transition_time=3.5)

        # --- and the light hardens with it ----------------------------
        # The rig is static for the first two thirds of the shot, which is
        # right while the molecule is only flying. Once it starts coming apart
        # the light comes up behind it and the fill goes away, so the second
        # half of the shot is colder and more contrasted than the first and the
        # strand that leaves does so out of a darker molecule. It is a slow
        # third of a stop, timed to the split rather than to the clock; delete
        # these two calls and the rig is simply static.
        ibpy.change_power(lamps["RimLight"], from_value=14.0, to_value=19.0,
                          begin_frame=int(split_start * FRAME_RATE),
                          frame_duration=int(7.0 * FRAME_RATE))
        ibpy.change_power(lamps["FillLight"], from_value=0.30, to_value=0.12,
                          begin_frame=int(split_start * FRAME_RATE),
                          frame_duration=int(7.0 * FRAME_RATE))

        self.t0 = duration

    # -------------------------------------------------------------------
    def rna_grid(self):
        """Every byte there is, written out as a strand: 256 of them at once.

        The beat between the molecule and the machine. ``dna_flyby`` ends with
        the helix unzipped and one strand peeled away, its bases left unpaired;
        this shot picks that strand up and makes the point the rest of the video
        needs: once a strand is single it is no longer a molecule holding a copy
        of something, it is a row of four symbols, and a row of four symbols is
        a number.

        :class:`RNAGridModifier` puts all 256 of them on screen at once - a
        16 x 16 grid, one cell per byte, each with a four-base strand and the
        number that strand spells beside it, in decimal and in base 4, the
        digits painted in the colour of the base that stands for them. Three of
        the four base colours are the ones the double helix was drawn in. The
        fourth is not, because the fourth base of a strand that is not DNA is
        uracil - which is the other half of the point, and the reason the grid
        can stand for a tape rather than for a chromosome.

        Column-major, so the grid is also a multiplication table: a column is
        sixteen consecutive numbers, its top two base-4 digits are constant all
        the way down, and only the bottom two bases change as the eye runs down
        it.

        It fills column by column and gets faster as it goes - the first columns
        far enough apart to read one at a time, the last ones almost together.
        The delay is not a number of frames written here but one solved for out
        of the duration this sub-scene declares, so the geometric series always
        lands its last column exactly ``hold`` seconds before the camera stops.
        Retiming the shot cannot leave half the grid unbuilt when it ends.

        The camera creeps in over the whole shot. Partly for life, but mostly
        for the reason ``_run_simple_bff`` creeps: nothing here keyframes
        anything - the grid is driven by ``Scene Time`` inside the node tree -
        and ``render_with_skips`` decides which frames are worth rendering by
        looking for f-curves that change. Without one animated object, every
        frame after the first would be classified as still and copied from its
        predecessor, and the grid would never appear to grow.
        """
        duration = self.sub_scenes["rna_grid"]["duration"]
        _setup_render(hdri="cayley_interior_4k", transparent=False)

        lead_in = 0.5  # seconds before the first column arrives
        hold = 3.0  # seconds to look at the finished grid
        growth = 0.55  # seconds one column takes to grow to full size
        speed_up = 0.88

        # RNAGridModifier starts column c at
        # ``start_frame + column_delay * (1 - q^c)/(1 - q)``. Everything in that
        # is chosen above except the delay, so it is what the shot's length is
        # solved for: the sum of the fifteen gaps has to fill the time between
        # the lead-in and the last column's growth.
        columns = RNAGridModifier.COLUMNS
        gaps = (1.0 - speed_up ** (columns - 1)) / (1.0 - speed_up)
        growth_frames = growth * FRAME_RATE
        fill = (duration - lead_in - hold) * FRAME_RATE - growth_frames

        # like every other sub-scene here, this one counts from zero - it is
        # built with ``start_at_zero`` and cut in afterwards - so the modifier's
        # frame numbers are the shot's own
        modifier = RNAGridModifier(start_frame=lead_in * FRAME_RATE,
                                   column_delay=fill / gaps,
                                   speed_up=speed_up,
                                   growth_frames=growth_frames,
                                   emission=0.7)
        grid = Plane(name="RNAGrid")
        grid.add_mesh_modifier(type='NODES', node_modifier=modifier)
        grid.appear(begin_time=0, transition_time=0)

        # --- the camera, keyed to the reveal ----------------------------
        # A 40 mm lens on a 36 mm sensor frames a half width of 18 * d / lens,
        # and at 16:9 a half height of 10.125 * d / lens. The finished grid is
        # about 49 by 28, which is 16:9 to within a percent, so one distance -
        # around 59 units - fits it in both directions at once.
        #
        # It only needs to be that far back at the end, though. Until then most
        # of that frame is empty, so the camera starts in close on the left,
        # near enough that a cell can be read, at the price of the top and
        # bottom rows being out of frame - and pulls out in step with the
        # columns. One keyframe per column, at the frame that column starts to
        # grow, framing everything that has been built by then: the camera
        # therefore accelerates exactly as the grid does, because it is reading
        # its times from the same series (:meth:`column_start_frame`).
        lens = 40.0
        margin = 1.08  # a little air around the content
        closest = 20.0  # any nearer and one cell is taller than the frame

        # how far a cell reaches on either side of its strand's axis: the
        # longest base to the left, the last base-4 digit to the right
        cell_left = -((max(RNAGridModifier.BASE_ATOMS) - 1) * modifier.bond_length
                      + RNAGridModifier.BASE_ATOM_RADIUS)
        cell_right = (modifier.text_x + 0.5 * (RNAGridModifier.BASES - 1)
                      * modifier.digit_spacing + 0.5 * modifier.glyph_size)
        left = -0.5 * (columns - 1) * modifier.column_spacing + cell_left

        ibpy.set_camera_lens(lens=lens, clip_end=2000)

        ibpy.set_camera_location(
            location=Vector((0, -56, 0)),
            frame=0)

        print("rna_grid: %d cells, columns %.1f frames apart at first and "
              "%.1f at last, grid complete at %.1f s of %d"
              % (RNAGridModifier.COLUMNS * RNAGridModifier.ROWS,
                 modifier.column_delay,
                 modifier.column_delay * speed_up ** (columns - 2),
                 modifier.reveal_end_frame() / FRAME_RATE, duration))

        self.t0 = duration

    def rna_logo(self):
        """One strand of RNA writes the logo out along its outline.

        The molecule is :class:`RNALogoModifier`, which is
        :class:`DNAModifier`'s flight along a track with the track replaced by
        the logo: ``logo_outline``, the same closed chain of circles
        :meth:`branding` draws, resampled into base stations.

        It used to be the Apollonian limit set out of
        ``apollonian_0.0001.dat`` - the fractal the logo is a picture of - and
        that is what ``n`` replaced. A fractal has no tangent, so the spokes the
        bases stand on flipped about from one base to the next, and no honest
        curvature, so their size came out of whatever window it was measured
        over; the outline has both defined everywhere and reads as a molecule
        instead of as noise. ``n`` is how many circles a side the chain has, 4
        here as in :meth:`branding`; raising it adds circles at the bottom of
        the picture, and the bases shrink with them.

        One dial does all of it. ``Progress`` is both how far round the head
        has got and how many bases are behind it, so a single ramp *draws* the
        logo: the strand gains a base for every station the head passes and its
        tail never moves. :attr:`track_bases` is a whole lap - the chain's
        circles times ``BasesPerCircle`` - so ramping to it draws all of it.

        The stations are not evenly spaced. Each circle of the chain carries
        ``BasesPerCircle`` bases whatever its radius, so the molecule is the
        same shape on the big circles and the small ones instead of thinning to
        one base per circle at the end of the chain.

        Both dials count in *stations*, not in blender units, because the
        stations are not evenly spaced - they crowd together where the logo
        turns tightly, so that the bases, which shrink there too, stay as dense
        on the page as they are out on the big circles.

        The camera looks straight down. The logo is built in the x-y plane,
        which is where ``z2vec`` puts a complex number and where the rest of
        this file draws, and at the default ``scale`` it spans x in [-6, 6] and
        y in [0, 12]. Height is the tight dimension of a 16:9 frame: a 40 mm
        lens 27 units up reaches z = +-6.8 across the short side, which is the
        twelve units of logo and a little under a unit of air.
        """
        cues = self.sub_scenes['rna_logo']
        t0 = 0.5  # cues['start']
        duration = cues['duration'] - 4
        _setup_render(hdri="cayley_interior_4k", transparent=True, frame_start=1)

        modifier = RNALogoModifier(emission=0, progess=730, bases_per_circle=18, n=6)
        molecule = Plane(name='RNALogo')
        molecule.add_mesh_modifier(type='NODES', node_modifier=modifier)
        molecule.appear(begin_time=0, transition_time=0)

        ibpy.set_camera_location(location=Vector([0, 6, 27]))
        camera_empty = EmptyCube(location=Vector([0, 6, 0]))
        ibpy.set_camera_view_to(camera_empty)
        ibpy.set_camera_lens(lens=40)

        # The head goes exactly once round and the strand grows by exactly as
        # much, over the same window and therefore with the same easing. Both
        # are floats, which is the whole reason the tail holds still: the same
        # ramp applied to the head and to the length cancels in
        # head - length whatever shape blender gives it, and the strand ends up
        # laid along the whole logo with its two ends meeting.
        head = ibpy.get_geometry_node_from_modifier(modifier, 'Progress')
        ibpy.change_default_value(head, from_value=0,to_value=modifier.track_bases+0.1,
                                  begin_time=t0, transition_time=duration)

        t0 = duration + 1

        strand_scale_node = ibpy.get_geometry_node_from_modifier(modifier, 'StrandScale')
        ibpy.change_default_value(strand_scale_node, from_value=1, to_value=0.25, begin_time=t0, transition_time=0.5)

        # create remaining parts of the logo.
        # no location_scale: the rings are placed by their own object transform
        # now, exactly as the red spheres are, and the parent's scale of 6 is
        # the only multiplication there should be
        logo = LogoFromInstances(instance_red=RNACircle, instance_green=RNACircle, instance_blue=RNACircle,
                                 rotation_euler=[-pi / 2, 0, 0], mode="XZ", details=20, scale=6,
                                 kwargs_blue={}, kwargs_green={},kwargs_red={"skip":[0,-1,1,2,-2,3]})
        t0 = logo.grow(begin_time=t0, transition_time=1)

        for obj in logo.get_red_instances():
            obj.change_color(new_color="red", begin_time=t0, transition_time=1)
            obj.scale_strand(from_value=1, to_value=0.3, begin_time=t0, transition_time=1)
        for obj in logo.get_blue_instances():
            obj.change_color(new_color="blue", begin_time=t0, transition_time=1)
            obj.scale_strand(from_value=1, to_value=0.3, begin_time=t0, transition_time=1)
        for obj in logo.get_green_instances():
            obj.scale_strand(from_value=1, to_value=0.3, begin_time=t0, transition_time=1)
            obj.change_color(new_color="green", begin_time=t0, transition_time=1)
        # change color of bases to logo color
        for slot in [0, 1, 2, 3, 4]:
            molecule.change_color(new_color="red", slot=slot, begin_time=t0, transition_time=1)
        t0 += 1.5

        self.t0 = t0

    # -------------------------------------------------------------------
    def paper(self):
        """The paper itself, as a book that pages through its own text.

        Agüera y Arcas et al., *Computational Life* (arXiv:2406.19108,
        ``docs/AgueraYArcas.pdf``) rendered page by page into
        ``media/raster/aya_*.png`` and wrapped onto a :class:`Book`, the same
        turning-page rig ``video_cmb/cmb.py``'s ``multipole_moments`` uses for
        its paper. ``aya_0.png`` is the title page and sits on the cover;
        ``aya_1.png`` through ``aya_19.png`` are the other nineteen pages,
        unwrapped alternately across the recto/verso of each leaf by
        :meth:`Book.set_page_image` (its ``index // 2 - 1`` pairs up two
        consecutive image indices per leaf).

        The book holds on the title cover for ten seconds - long enough to
        read it - then opens and turns every leaf slowly enough to watch,
        rather than the fast riffle ``cmb.py`` uses when the pages are just
        set dressing for other content.
        """
        _setup_render(hdri="forest", transparent=True)
        t0 = ibpy.set_hdri_strength(1, begin_time=0, transition_time=0)
        # the book's cover faces up (+Z), the way it is built - a head-on
        # camera looking along +Y sees it edge-on as a thin sliver, so this
        # needs the tilted, looking-down camera instead (as in video_ising's
        # equivalent book() scene)
        ibpy.set_camera_location(location=[0, 0, 4.3])
        camera_empty = EmptyCube(location=(0, 0, 0.15))
        ibpy.set_camera_view_to(camera_empty)

        # _setup_tilted_camera(location=(1, -4.5, 3.8), target=(1, 0, 0.15))

        paper = Book(pages=10, scale=[1, 1.4, 0.05], cover_thickness=0.01,
                     page_thickness=0.001, name="Paper")
        # Generated coordinates on this face come out mirrored left-right;
        # video_ising/scene_ising.py's book() hits the same thing on its
        # cover and fixes it the same way, flipping X and re-centering.
        paper.set_cover_image("aya_0.png", scale=Vector([-1, 1, 1]),
                              location=Vector([1, 0, 0]), emission=0, brightness=-0.5)
        for i in range(1, 2 * paper.pages):
            paper.set_page_image(i, "aya_%d.png" % i, extension="REPEAT",
                                 coordinates='UV', emission=0, scale=[-4] * 3, brightness=-0.5,
                                 location=[1.53, 0, 0])

        # Book.appear() (unlike most BObject methods) returns None, since it
        # fans out over cover/back/spine/pages rather than returning
        # begin_time + transition_time - so t0 is tracked by hand here. (Also
        # deliberately not using grow(): it always keyframes scale from 0 at
        # its own begin_time, and Blender extrapolates that 0 backward across
        # the whole hold before it - the book would stay invisible for the
        # ten seconds this scene needs it on screen.)
        paper.appear(begin_time=0, transition_time=1)
        t0 = 1 + 10  # fade in, then ten seconds to read the title cover
        ibpy.camera_move(shift=Vector([0, -4.5, -0.45]), begin_time=t0 - 1, transition_time=1)
        t0 = paper.open(begin_time=t0, transition_time=2) - 1
        for i in range(paper.pages):
            t0 = paper.turn_page(i, begin_time=t0, transition_time=1.2)

        self.t0 = t0 + 1.5

    # -------------------------------------------------------------------
    def soup_watcher(self):
        """Watch the soup evolve: 100 tapes at a time, in one column.

        Reads ``data/soup_evolution_bytes.csv`` - the byte form of the file
        ``brainfuck/bff/soup_watcher.py`` appends to as it runs (64 integer
        columns, one per cell) - through :class:`SoupWatcherModifierSingle`.
        Every ten frames the block of 100 tapes on screen is replaced by the
        next one recorded in the file, so watching this scene run is watching
        the same soup ``soup_watcher.py`` watched, compressed into however
        many snapshots the file holds.

        The snapshot is stacked as a single column rather than two of fifty,
        and the camera is placed by :func:`_setup_row_camera` so that the
        nearest tape runs the full width of the frame: at this size its 64
        cells are individually readable, which is the whole point of watching
        the soup rather than plotting it. The rest of the column recedes
        behind it and the far end of it runs off the top of the screen -
        about two thirds of the hundred tapes are outside the frame at any
        time, and that is the trade being made. The tilt is unchanged (it
        lives in the camera, not in the geometry: the sheet is built flat and
        looked down on), so only the distance and the centring differ from
        the two-column version.

        The whole file is piped into the node graph through one ``Import CSV``
        node (``max_snapshots=None``): every snapshot the file holds is
        available, and the shot walks through them one block of 100 tapes per
        ``frames_per_snapshot`` frames. This sub-scene's declared duration is
        what ends the shot, not the length of the data - a file recorded past
        that point simply has its tail unused, and a file shorter than the
        shot is shown in full and then loops (the modifier picks
        the snapshot with ``... mod NumSnapshots``, so it wraps back to the
        start rather than running off the end).
        """
        _setup_render(hdri="cayley_interior_4k", transparent=True, engine="CYCLES")
        camera_empty = EmptyCube(location=[2.8, 2.2, 0])
        ibpy.set_camera_lens(lens=21.5)
        ibpy.set_camera_location(location=[2.8, -1.7, 3.4])
        set_alpha_composition()
        ibpy.camera_set_track(camera_empty, influence=1)
        t0 = 0

        frames_per_snapshot = 1
        duration = self.sub_scenes["soup_watcher"]["duration"]
        modifier = SoupWatcherModifierSingle(
            frames_per_snapshot=frames_per_snapshot, glyph_size=1.5, stick_out=0.025, emission=0.7)
        # the tilt is the direction the two-column shot looked from, kept as
        # the shot's angle while the distance that fits one tape across the
        # frame is re-derived from the tape's own width

        machine = Plane(name="SoupWatcher")
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=t0, transition_time=0)
        machine.change_alpha(from_value=1, to_value=0.1, slot=12, begin_time=0, transition_time=0)

        shown = min(modifier.num_snapshots * modifier.frames_per_snapshot,
                    duration * FRAME_RATE) / FRAME_RATE
        self.t0 = t0 + shown

    # -------------------------------------------------------------------
    def soup_watcher_star_wars(self):
        """:meth:`soup_watcher`, ending on a title crawl and the project page.

        The same shot as ``soup_watcher`` - same camera, same render, same
        tapes cycling through ``data/soup_evolution_bytes.csv`` - until its
        last twenty-odd seconds, when :class:`SoupWatcherModifierSingleStarWars`
        earns its name: the sheet of tapes is pushed away along the line of
        sight until it is small and far, and an end card climbs out of the
        bottom of the frame in its place.

        The card is not a flat overlay. It lies in a plane tilted away from
        the camera (see the modifier), so that "crawling up the screen" and
        "receding into the distance" are the same motion and the lettering
        shrinks towards a vanishing point just past the top of the frame -
        which is the whole trick of a receding title crawl, and the reason
        the tapes have to be got out of the way first: that plane sinks below
        theirs as it goes.

        Riding six units behind the text, in the same plane, is
        ``media/raster/clr.png``. It is built exactly as wide as the frame is
        at ``fill_distance``, so when it has climbed far enough to hold the
        bottom half of the screen the shot can simply stand it up - one
        rotation about x, from lying in the crawl to facing the camera
        square - and let it fill the frame as the end card of the video.

        Every one of those beats is measured against the frame rather than
        guessed: :func:`_ndc_projector` says where a point on the crawl lands
        on screen, and :func:`_crawl_distance_at` inverts it, so "start below
        the bottom edge" and "stop when the screenshot reaches the middle"
        are solved for, not tuned by eye.
        """
        duration = self.sub_scenes["soup_watcher_star_wars"]["duration"]

        # --- the soup_watcher shot, unchanged --------------------------
        _setup_render(hdri="cayley_interior_4k", transparent=True, engine="CYCLES")
        lens = 21.5
        camera_location = Vector((2.8, -1.7, 3.4))
        camera_target = Vector((2.8, 2.2, 0))
        view = camera_target - camera_location
        camera_empty = EmptyCube(location=camera_target)
        ibpy.set_camera_lens(lens=lens)
        ibpy.set_camera_location(location=camera_location)
        set_alpha_composition()
        ibpy.camera_set_track(camera_empty, influence=1)

        frames_per_snapshot = 10
        modifier = SoupWatcherModifierSingleStarWars(
            frames_per_snapshot=frames_per_snapshot, glyph_size=1.5,
            stick_out=0.025, view_direction=view, emission=0.7)
        machine = Plane(name="SoupWatcherStarWars")
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)
        machine.change_alpha(from_value=1, to_value=0.1, slot=12,
                             begin_time=0, transition_time=0)

        def dial(label):
            """The Value node of the modifier the scene turns by that name."""
            return ibpy.get_geometry_node_from_modifier(modifier, label)

        # --- where things land on screen -------------------------------
        project = _ndc_projector(camera_location, view, lens=lens)
        place = modifier.crawl_position

        # the screenshot is 16:9, like the frame, and sized so that the frame
        # is exactly as wide as it is at `fill_distance` - standing it up
        # there covers the shot with no scaling and no black edges
        half_width = 3.0
        half_height = half_width * 9 / 16
        fill_distance = half_width / (0.5 * 36 / lens)

        # half a block of three lines, near enough to start the card just
        # under the bottom edge rather than half-way through it
        card_reach = 1.5 * modifier.line_spacing * modifier.crawl_size
        trail = 6.0  # how far the screenshot rides behind the text

        crawl_start = _crawl_distance_at(project, place, -1.0) - card_reach
        # the crawl ends when the screenshot's far edge reaches the middle of
        # the frame, which is the moment it holds half the screen
        image_stop = _crawl_distance_at(project, place, 0.0) - half_height
        crawl_stop = image_stop + trail

        # apply_location=False, or Plane bakes the starting point into the mesh
        # and leaves the object at the origin - which then moves the geometry
        # twice as far as it should when `move_to` takes over
        image = Plane(u=[-half_width, half_width], v=[-half_height, half_height],
                      color='image', src='clr.png', emission=1, name="Screenshot",
                      location=place(crawl_start - trail), apply_location=False,
                      rotation_euler=[-modifier.crawl_tilt, 0, 0])
        image.appear(begin_time=0, transition_time=0)

        # --- the outro -------------------------------------------------
        outro, stand_up, hold = 24, 2.5, 2.0
        t0 = duration - outro

        # the tapes leave first: the crawl's plane sinks below theirs, so it
        # would cut into the sheet if the sheet were still there
        ibpy.change_default_value(dial("Recede"), 0, 6,
                                  begin_time=t0, transition_time=3)

        crawl_time = outro - 1.5 - stand_up - hold
        crawl_begin = t0 + 1.5
        crawl_end = crawl_begin + crawl_time
        ibpy.change_default_value(dial("CrawlDistance"), crawl_start, crawl_stop,
                                  begin_time=crawl_begin, transition_time=crawl_time)
        # the screenshot is keyframed rather than driven by the modifier, but
        # over the same interval and along the same straight line, so the two
        # ride the crawl together whatever easing blender puts on them
        image.move_to(target_location=place(image_stop),
                      begin_time=crawl_begin, transition_time=crawl_time)

        # standing up: the plane's normal ends up pointing back down the line
        # of sight, which for a camera with no roll is one rotation about x
        upright = np.arctan2(view.y, -view.z)
        image.rotate(rotation_euler=[upright, 0, 0],
                     begin_time=crawl_end, transition_time=stand_up)
        image.move_to(target_location=camera_location + fill_distance * view.normalized(),
                      begin_time=crawl_end, transition_time=stand_up)

        print("soup_watcher_star_wars: soup for %.1fs, crawl %.2f -> %.2f over "
              "%.1fs, screenshot stands up at %.1fs" %
              (t0, crawl_start, crawl_stop, crawl_time, crawl_end))

        self.t0 = crawl_end + stand_up + hold

    # -------------------------------------------------------------------
    def morphing(self):
        """A picture frame turning into an arrow, with the two made compatible.

        ``tmp.xml``'s tree is :class:`MorphModifier`, and it is the reason
        ``geometry_nodes/docs/theory_morphing.tex`` exists: a tube bent into
        a rectangle and a solid cone on a cylinder share neither a point
        count nor a topology, so every correspondence between them is a
        choice of how to be wrong. Pairing by index tangles the surface;
        pairing by nearest point shrink-wraps it flat. Neither is a fault of
        ``MorphNode`` - there is no right answer to find.

        So this scene uses :class:`OutlineMorphModifier`, which takes the
        theory's first answer instead: stop looking for a correspondence and
        make the shapes compatible. Both are built as closed curves, both
        resampled to the same 128 points, both swept along the same profile
        circle - so the two meshes are the same mesh twice over, index
        pairing is exact, and the morph is a plain interpolation that cannot
        fold. The arrow is its own silhouette, which is what makes it the
        same kind of object as the frame.

        The one line back to the faithful tree is ``MorphModifier(...)`` in
        place of the modifier below; the two are worth watching back to back.
        """
        _setup_render()
        # the colour has to be added *inside* the tree - a material in the
        # object's slot never reaches geometry that nodes create
        modifier = TubeMorphModifier(color='example', emission=0.3,
                                     samples=128, profile_resolution=24)
        # both shapes at once: the frame reaches out to x = -6.4, the arrow
        # sits on the origin, so the camera is centred between them
        _setup_standard_camera(distance=8, shift_x=-2.9)

        machine = Plane(name="Morph")
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)

        print("morphing: %d samples along each curve, barb at %.2f of the axis"
              % (modifier.samples, modifier.barb_fraction))

        t0 = 2  # a moment to read the frame before it goes anywhere
        t0 = ibpy.change_default_value(
            ibpy.get_geometry_node_from_modifier(modifier, "MorphParameter"),
            0, 1, begin_time=t0, transition_time=6)

        self.t0 = t0 + 3

    # -------------------------------------------------------------------
    def moving_tape(self):
        """A hundred bytes of tape travelling through the shot.

        ``tmp.xml``'s tree as it stands now, :class:`MovingTapeModifier`: a
        strip of cells with a random byte on each, sliding in from the left,
        past the camera and out to the right, and cut off at both edges of the
        frame so that the ninety-odd cells that are not on screen cost nothing.

        Shot head-on, because that is what the geometry is built for: the
        digits stand upright out of the tape and only the cells are tilted, by
        the 18 degrees the editor gives them. A camera looking along +y
        therefore has the numbers facing it squarely, and the tilt is exactly
        what keeps the tape itself from being a line - it opens the cells into
        a thin band for the numbers to stand on. Tilting the camera instead
        would trade the one for the other.

        The other is the speed, and the editor's ``TransitionTime`` of 20
        seconds is it: 11 cells a second, a cell crossing the frame in a second
        and a half, its number readable on the way past. At the 5 seconds the
        tree started out with, the whole tape went by in 2.8 (see
        :meth:`MovingTapeModifier.crossing_times`) - 45 cells a second, a blur.

        The camera creep is not decoration: the whole animation lives in
        ``Scene Time`` inside the graph and keyframes nothing, and
        ``render_with_skips`` decides which frames to render by looking for
        f-curves that change - so without one animated object every frame after
        the first would be copied from its predecessor and the tape would never
        move. The same reason the machine scenes creep; see
        :meth:`_run_simple_bff`.
        """
        _setup_render(hdri="cayley_interior_4k", transparent=True)
        modifier = MovingTapeModifier(number_offset=Vector([0, 0, 0.075]), emission=0.3)
        tape = Plane(name="MovingTape")
        tape.add_mesh_modifier(type='NODES', node_modifier=modifier)
        tape.appear(begin_time=0, transition_time=0)

        # 11 units back on a 30 mm lens frames 13 units of tape, a dozen cells,
        # and is aimed at the height of the digits rather than at the tape
        _setup_standard_camera(distance=11, shift_z=0.4)

        enter, leave = modifier.crossing_times()
        print("moving_tape: %d cells of %d bytes, %.1f units apart, on screen "
              "from %.1fs to %.1fs at %.1f cells/s"
              % (modifier.tape_length, modifier.max_value + 1,
                 modifier.tape_span / (modifier.tape_length - 1), enter, leave,
                 modifier.travel_distance / modifier.transition_time
                 / (modifier.tape_span / (modifier.tape_length - 1))))

        # ``leave`` is measured at the cutoff, half a second of travel outside
        # the frame, so a short tail is already a beat of empty tape
        tail = 0.3
        ibpy.camera_move([0, 1.5, -0.5], begin_time=0,
                         transition_time=leave + tail)
        self.t0 = leave + tail

    def close_up(self):
        """
        play the soup-watcher in the range from 8000 to 10000 in slow-mo
        add compression diagram as overlay
        """

        duration = self.sub_scenes["close_up"]["duration"]
        t0 = 0

        # --- the soup_watcher shot, unchanged --------------------------
        _setup_render(hdri="cayley_interior_4k", transparent=True, engine="CYCLES")
        lens = 21.5
        camera_location = Vector([2.8, 0, 0]) + 2 * Vector((0, -1.7, 3.4))
        camera_target = Vector((2.8, 2.2, 0))

        camera_empty = EmptyCube(location=camera_target)
        ibpy.set_camera_lens(lens=lens)
        ibpy.set_camera_location(location=camera_location)
        set_alpha_composition()
        ibpy.camera_set_track(camera_empty, influence=1)

        frames_per_snapshot = 1
        modifier = SoupWatcherModifierSingle(data_file="soup_evolution_bytes_8000_10000.csv",
                                             frames_per_snapshot=frames_per_snapshot, glyph_size=1.5,
                                             stick_out=0.025, emission=0.7)
        machine = Plane(name="SoupWatcherCloseUp")
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)
        machine.rescale(rescale=2, begin_time=t0, transition_time=0)
        machine.move_to(target_location=Vector([-2.5, 0, -1.0]), begin_time=t0, transition_time=0)
        machine.change_alpha(from_value=1, to_value=0.1, slot=12,
                             begin_time=0, transition_time=0)

        # run the diagram in parallel

        epochs, sizes = _read_compression(name="compression_8000_10000.csv")
        raw = _raw_size(sizes)
        transition = _transition_epoch(epochs, sizes)

        # kilobytes, not kibibytes: the y axis is read by an audience, and a
        # 256 that is really 262144 invites the wrong arithmetic against the
        # "4096 times 64 bytes" the script says out loud
        kilo = 1000
        x_min = epochs[0]
        x_max = epochs[-1]
        y_max = 1.1 * raw / kilo
        x_step, y_step = _nice_step(x_max - x_min), _nice_step(y_max)

        # 11 x 6.2 units in a frame that a 45 mm lens 20 units back fills out
        # to x in [-8, 8], z in [-4.5, 4.5]: wide enough that 10000 epochs are
        # not squeezed, and clear of both edges - the tic labels of the x axis
        # hang 0.6 below it, so an axis on the floor of the frame would have
        # its labels rendered off-screen
        origin, width, height = Vector([-2, -0.7, 0]), 9, 4.5

        def world(epoch, kilobytes):
            """Where a data point of the plot lands in world coordinates.

            The axes and the data rows are children of the coordinate system
            and are placed by its modifiers; the captions are not, so they
            need the same mapping applied by hand.
            """
            return origin

        coords = CoordinateSystem2(
            location=origin, lengths=[width, height], colors=['text', 'text'],
            domains=[[x_min, x_max], [0, y_max]], tic_label_digits=[0, 0],
            tic_labels=[_tic_labels(x_step, x_max + x_step / 2, x_min),
                        _tic_labels(y_step, y_max)],
            axes_labels={r"\text{epoch}": [-0.5, 0, width + 0.4],
                         r"\text{compressed size [kB]}": [-1.15, 0, 5.15]},
            # tic labels are anchored 12 axis radii to the side of their tic
            # and, with the default 'left', run from there into the plot -
            # which puts three digit labels on top of the y axis. Centering
            # them instead is what lets the x labels sit under their tics, and
            # the shift then moves the y column clear of the axis
            aligned='center', tic_label_shifts=[Vector(), [-1.3, 0, 0]])
        coords.rotate(rotation_euler=Vector([atan2(camera_location.z, camera_location.y - camera_target.y) + pi, 0, 0]),
                      begin_time=0, transition_time=0)
        coords.appear(begin_time=t0, transition_time=3)

        # what the soup costs when it is still noise, as a line to fall away
        # from - the plot only says something once there is something to
        # compare the curve against
        raw_line = Data(data=[[0, 0, raw / kilo], [x_max, 0, raw / kilo]],
                        coordinate_system=coords, name="RawSize",
                        material="drawing", emission=0.3, linesize=0.5,
                        width=width, height=height)

        raw_line.appear(begin_time=t0, transition_time=1.5)

        # the curve draws itself in for as long as the shot has left, minus a
        # tail to hold on the finished plot

        sweep = 2000 / FRAME_RATE  # 2000 epochs
        curve = Data(data=[[e, 0, s / kilo] for e, s in zip(epochs, sizes)],
                     coordinate_system=coords, name="Compression",
                     material="example", emission=0.5, linesize=1,
                     width=width, height=height)
        curve.appear(begin_time=0, transition_time=sweep)

        self.t0 = duration

    def close_up_star_wars(self):
        """
        play the soup-watcher in the range from 8000 to 10000 in slow-mo
        add compression diagram as overlay
        """

        t0 = 0

        # --- the soup_watcher shot, unchanged --------------------------
        _setup_render(hdri="cayley_interior_4k", transparent=True, engine="CYCLES")
        lens = 21.5
        camera_location = Vector([2.8, 0, 0]) + 2 * Vector((0, -1.7, 3.4))
        camera_target = Vector((2.8, 2.2, 0))

        camera_empty = EmptyCube(location=camera_target)
        ibpy.set_camera_lens(lens=lens)
        ibpy.set_camera_location(location=camera_location)
        set_alpha_composition()
        ibpy.camera_set_track(camera_empty, influence=1)

        frames_per_snapshot = 10000
        modifier = SoupWatcherModifierSingle(data_file="soup_evolution_bytes_9900.csv",
                                             frames_per_snapshot=frames_per_snapshot, glyph_size=1.5,
                                             stick_out=0.025, emission=0.7)
        machine = Plane(name="SoupWatcherCloseUp")
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)
        machine.rescale(rescale=2, begin_time=t0, transition_time=0)
        machine.move_to(target_location=Vector([-2.5, 0, -1.0]), begin_time=t0, transition_time=0)
        machine.change_alpha(from_value=1, to_value=0.1, slot=12,
                             begin_time=0, transition_time=0)

        machine.move(direction=Vector([0, 20, 0]), begin_time=t0, transition_time=20)

        # moving text

        lines = [r"\text{Computational Life Reactor}",
                 r"\text{by Alex Borger}",
                 r"\text{https:\/\/alexborger.com/clr-computational-life-reactor}"]

        for i, line in enumerate(lines):
            size = "Large"
            sep = 1.5
            if i == 2:
                size = "normal"
                sep = 1.3
            btext = SimpleTexBObject(line, location=Vector([2.8, -3 - sep * i, 0]), text_size=size, aligned="center",
                                     color='example', rotation_euler=[0, 0, 0], emission=1)
            btext.write(begin_time=0, transition_time=0)

            btext.move(direction=Vector([0, 20, 0]), begin_time=0, transition_time=20)

        # moving plane

        trans_plane = Plane(u=[0, 2 * 1.920], v=[0, 2 * 1.080], color="image", src="clr.png", emission=1)
        trans_plane.move_to(target_location=Vector([2.8 - 1.920 * 2.768, -13, 0]), begin_time=t0, transition_time=0)
        trans_plane.rescale(rescale=2.768, begin_time=t0, transition_time=0)
        trans_plane.appear(begin_time=t0, transition_time=0)
        trans_plane.move(direction=Vector([0, -1.6726 + 13, 0]), begin_time=t0 + 2, transition_time=-1.6726 + 13)
        trans_plane.rotate(
            rotation_euler=[atan2(camera_location.z, camera_location.y - camera_target.y) - pi / 2, 0, 0],
            begin_time=-1.6726 + 15 + t0, transition_time=2)

        t0 += 20 + 0.5

        self.t0 = t0

    # -------------------------------------------------------------------
    def _run_simple_bff(self, program, step_duration, name, tape_size=5,
                        cell_size=1.5, start_time=3.0, tail=2.0):
        """Build one brainfuck machine and let it run to the end.

        Everything the four ``simple_brain_fuck*`` scenes have in common. They
        differ only in the program and in how fast it is stepped through - the
        looping variants execute far more instructions than they contain, so
        they need a shorter step to finish in about the same time.

        The program is written out once across the input display, one column
        per instruction, and does not move; the small box that used to hold the
        current instruction runs along it instead, standing around the one
        about to be executed. Every instruction carries its own colour, the one
        it has in every other scene of the video (``INSTRUCTION_COLORS``), until
        the machine has been past it: instructions that have run and will not
        run again go dark, the ones inside a loop that is still open turn
        yellow because they are coming back, and the one under the box is in
        the colour of the head marker.

        The camera creeps in over the whole scene. That is partly to give the
        shot some life, but mostly because ``render_with_skips`` decides which
        frames to render by looking for objects whose *f-curves* change: a
        geometry-nodes simulation keyframes nothing, so without one animated
        object every frame after the first would be classified as still and
        copied from its predecessor, and the machine would never appear to run.

        :param step_duration: seconds one instruction is on screen
        :param tail: seconds to hold the finished machine for
        """
        _setup_render()

        # the tape sits between x=0 and x=TapeSize*CellSize and everything else
        # is centred on the middle of it, so that is what the camera looks at
        middle = 0.5 * tape_size * cell_size

        machine = Plane(name=name)
        # These scenes are lit almost entirely by emission on a black
        # background, so the palette has to glow a little to be seen at all -
        # the plain materials of define_materials() render nearly black.
        modifier = BrainFuckSimpleModifier(program=program, tape_size=tape_size,
                                           cell_size=cell_size,
                                           step_duration=step_duration,
                                           start_time=start_time,
                                           name=name + "Machine", emission=0.6)
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)

        # The machine is now a single column - code table, tape, program,
        # output - about ten units tall and as wide as the code table, so the
        # shot is aimed at the middle of that column rather than at the tape,
        # and pulled back far enough to hold all of it.
        _setup_standard_camera(distance=17, shift_x=middle, shift_z=-1.75)
        # how long the machine runs is the number of instructions *executed*,
        # which for a program with a loop in it is not its length - so ask the
        # python model of the machine rather than counting characters
        steps, output, _ = BrainFuckSimpleModifier.simulate(program, tape_size)
        print("%s: %d instructions, %d steps, prints %r"
              % (name, len(program), steps, output))
        run_time = start_time + steps * step_duration
        #ibpy.camera_move([0, 2, 0], begin_time=0, transition_time=run_time + tail)

        self.t0 = run_time + tail

    # -------------------------------------------------------------------
    def simple_brain_fuck(self):
        """A whole brainfuck machine, running inside geometry nodes.

        Nothing in this scene is animated from python. It builds
        ``SimpleBrainFuckModifier`` and lets it run: the modifier holds a
        simulation zone that keeps the tape, the head and the two strings from
        one frame to the next, and executes one instruction per
        ``step_duration``.

        The point of the exercise is the encoding. Cells hold 1 for ``A`` up to
        26 for ``Z`` instead of ascii, and the table that says so is drawn
        above the tape. ``HELLO`` is then 8, 5, 12, 12, 15, and the whole
        program is 27 instructions::

            ++++++++.>+++++.<++++..+++.

        Cell 0 is raised to 8 and printed as ``H``, the head steps right and
        cell 1 is raised to 5 for ``E``, then the head steps back to cell 0 and
        tops it up to 12 for the two ``L``s and to 15 for the ``O``. In ascii
        the same output needs more than seventy ``+``.

        The eight ``+`` at the front are the part that does not scale, and the
        three scenes after this one replace them with a loop - see
        :meth:`simple_brain_fuck2`.
        """
        self._run_simple_bff(BrainFuckSimpleModifier.HELLO, step_duration=0.5,
                             name="SimpleBff")

    # -------------------------------------------------------------------
    def simple_brain_fuck2(self):
        """The same HELLO, with the 8 counted out by a loop.

        ``++++++++`` says eight in the only way a machine without loops can:
        eight times. ``[`` and ``]`` turn that into multiplication. The program
        is 31 instructions::

            ++++[>++<-]>.>+++++.<++++..+++.

        ``++++`` puts 4 into cell 0. ``[`` looks at the cell under the head and
        skips past the matching ``]`` when it holds zero, so the body
        ``>++<-`` runs while cell 0 lasts: step right, add 2 to cell 1, step
        back, take 1 off cell 0. Four turns, 2 each, and cell 1 holds 8 - which
        the ``>`` after the loop steps onto and ``.`` prints as ``H``.

        The tail is the same as before, one cell further right: cell 2 is
        raised to 5 for the ``E``, then cell 1 is topped up to 12 and 15.

        The interesting part is what this costs. The loop is four instructions
        longer than the eight ``+`` it replaces, and it *executes* 49
        instructions rather than 27 - the counting down and stepping back and
        forth is work the straight version never does. Loops pay off when the
        number is big, and 8 is not big. What they buy here is that the same 11
        instructions would build 100 as easily as 8.
        """
        self._run_simple_bff(BrainFuckSimpleModifier.HELLO_LOOP,
                             step_duration=0.45, name="SimpleBffLoop")

    def hello_extended(self):
        """The two-headed machine writing HELLO, on one tape you can read.

        The argument the whole video has been building to, in one shot: the
        machine of the paper has no output at all - no print, no read-out -
        and it writes ``HELLO`` anyway, because a BFF program says what it has
        to say by writing it onto the tape it is itself written on.

        The program is the one-headed ``HELLO`` with its five prints turned
        into copies::

            {{{{{{++++[>++<-]>.}>+++++.}<++++.}.}+++.}

        Six ``{`` walk the second head left off cell 0 and round the ring onto
        the last cells of memory; the arithmetic between the ``}`` is the same
        8, 5, 12, 12, 15 the one-headed machine printed, and each ``.`` copies
        one of them across. So the two ends of the tape are the two halves of
        the idea - the program at one end, what it has to say at the other -
        and nothing in between ever leaves the tape.

        The tape is :attr:`BrainFuckHelloModifier.tape_size` cells and not
        one more: three for the first head to work in, forty-one for the
        program, one zero to end it on, and five for the answer - one per
        letter, with nothing left over, because the five ``}`` bring the
        second head round the ring to cell 0 again.

        The two cells the first head adds up in start at 64, which is where
        the capitals begin in ascii, and that is the whole trick: counting
        eight into a cell holding 64 leaves 72, and 72 is ``H``. So the
        machine writes ascii rather than something that has to be translated,
        and the table above the tape is the right table to read it in. The
        answer cells show the numbers as they arrive and are read as letters
        the moment the machine halts.

        Both heads start on cell 0, one drawn above the tape and one below.
        The counter does not: it starts on the first cell that is not zero,
        which is the first instruction, and stops on the zero that was put
        after the last one - so the cursor is only ever on the program, and
        the machine halts by itself after
        :attr:`BrainFuckHelloModifier.steps` instructions.

        The camera creeps for the reason every machine scene creeps: the whole
        animation lives in a simulation zone and keyframes nothing, and
        ``render_with_skips`` decides what to render by looking for f-curves
        that change - see :meth:`_run_simple_bff`.
        """
        _setup_render(hdri="cayley_interior_4k", transparent=True)

        step_duration, start_time, tail = 0.25, 1.0, 2.0
        machine = Plane(name="HelloExtended")
        # emissive, like every machine scene: these are lit by their own
        # palette against a dark background
        modifier = BrainFuckHelloModifier(step_duration=step_duration,
                                          start_time=start_time, cell_size=0.6,
                                          emission=0.6, name="HelloExtended")
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)

        # the tape is the subject and the ascii table stands above it, so the
        # shot is wide enough for the one and tall enough to keep the other
        # the tape sits on z=0 and the ascii table reaches up to about ten, so
        # the shot is aimed between them rather than at either. Fifty cells is
        # short enough to be read across the frame, so it is given nearly all
        # of the width rather than the 85% _frame_distance leaves by default.
        width = modifier.tape_size * modifier.cell_size
        _setup_standard_camera(
            distance=_frame_distance(1.08*width, 12, fill=0.97), shift_z=4.8)

        steps, tape = modifier.simulate()
        written = "".join(modifier.LETTERS[value - 1]
                          for value in tape[-modifier.spare:] if 0 < value <= 26)
        run_time = start_time + steps * step_duration
        print("hello_extended: %d cells, %d steps, writes %r, ends at %.1fs"
              % (modifier.tape_size, steps, written, run_time))

        self.t0 = run_time + tail

    # -------------------------------------------------------------------
    def epoch_counter(self):
        """``Epoch: n``, counting up ten a frame from nothing to ten thousand.

        An overlay rather than a shot: the number the soup is on, to be cut
        over the run of it. Nothing here is keyframed and nothing is
        simulated - :class:`EpochCounterModifier` reads ``Scene Time`` and
        writes the number that follows from it, so the count is a property of
        the frame rather than of anything that has happened, and a render that
        skips frames still puts the right number on the ones it keeps.

        It counts for :attr:`EpochCounterModifier.frames` frames - a thousand
        of them, sixteen and two thirds seconds at sixty a second - and then
        holds on ten thousand, so the cut can leave whenever it likes without
        the number running away underneath it.

        The background is transparent and the lettering is emissive, which is
        what makes it an overlay: what the compositor gets is the word and
        the number and nothing else - and a bloom over the two of them, so
        that the read-out carries the same glow as everything else the soup
        is lit by.
        """
        _setup_render(hdri="cayley_interior_4k", transparent=True)
        # the lettering is the only thing in frame and it is emissive, so the
        # threshold sits under its brightness (yellow at emission 0.6) rather
        # than at the 1 a lit scene would want
        create_glow_composition(threshold=0.4, size=5)

        counter = Plane(name="EpochCounter")
        # emissive, because this is composited over the soup rather than lit
        # with it - see the same reasoning in _run_simple_bff
        modifier = EpochCounterModifier(step=10, last_epoch=10000,
                                        label="Epoch: ", text_size=1,
                                        color="example", emission=0.6,
                                        name="EpochCounter")
        counter.add_mesh_modifier(type='NODES', node_modifier=modifier)
        counter.appear(begin_time=0, transition_time=0)

        # "Epoch: 10000" is about nine units of lettering; the camera is
        # placed to hold that rather than at a distance picked by eye
        _setup_standard_camera(distance=_frame_distance(9.5, 2.5))

        print("epoch_counter: 0 to %d in steps of %d, %d frames, %.2f s"
              % (modifier.last_epoch, modifier.step, modifier.frames,
                 modifier.duration))
        self.t0 = modifier.duration + 1

    def epoch_counter2(self):
        """``Epoch: n``, counting up ten a frame from nothing to ten thousand.

        An overlay rather than a shot: the number the soup is on, to be cut
        over the run of it. Nothing here is keyframed and nothing is
        simulated - :class:`EpochCounterModifier` reads ``Scene Time`` and
        writes the number that follows from it, so the count is a property of
        the frame rather than of anything that has happened, and a render that
        skips frames still puts the right number on the ones it keeps.

        It counts for :attr:`EpochCounterModifier.frames` frames - a thousand
        of them, sixteen and two thirds seconds at sixty a second - and then
        holds on ten thousand, so the cut can leave whenever it likes without
        the number running away underneath it.

        The background is transparent and the lettering is emissive, which is
        what makes it an overlay: what the compositor gets is the word and
        the number and nothing else - and a bloom over the two of them, so
        that the read-out carries the same glow as everything else the soup
        is lit by.
        """
        _setup_render(hdri="cayley_interior_4k", transparent=True)
        # the lettering is the only thing in frame and it is emissive, so the
        # threshold sits under its brightness (yellow at emission 0.6) rather
        # than at the 1 a lit scene would want
        create_glow_composition(threshold=0.4, size=5)

        counter = Plane(name="EpochCounter")
        # emissive, because this is composited over the soup rather than lit
        # with it - see the same reasoning in _run_simple_bff
        modifier = EpochCounterModifier(step=1, last_epoch=10000,first_epoch=8000,
                                        label="Epoch: ", text_size=1,frame_skip=1,
                                        color="example", emission=0.6,
                                        name="EpochCounter")
        counter.add_mesh_modifier(type='NODES', node_modifier=modifier)
        counter.appear(begin_time=0, transition_time=0)

        # "Epoch: 10000" is about nine units of lettering; the camera is
        # placed to hold that rather than at a distance picked by eye
        _setup_standard_camera(distance=_frame_distance(9.5, 2.5))

        print("epoch_counter: 0 to %d in steps of %d, %d frames, %.2f s"
              % (modifier.last_epoch, modifier.step, modifier.frames,
                 modifier.duration))
        self.t0 = modifier.duration + 1

    # -------------------------------------------------------------------
    def bf_to_bff(self):
        """The brainfuck machine turning into the tape BFF runs on.

        One :class:`BrainFuckTransitionModifier` and no cuts. The machine
        finishes its HELLO almost at once (``step_duration`` is a thousandth
        of a second), and the rest of the shot takes it apart and puts the
        pieces back as a BFF tape - where the tape *is* the program and there
        is no separate machine to look at.

        The order is the argument, and each step has its own moment:

        ``2 - 3``
            the tape opens: five fat cells into sixty-four thin ones, moved
            left so that they fit. That only works because this machine
            rebuilds its tape inside the simulation zone every frame - see
            :meth:`BrainFuckTransitionModifier._tape_in_zone`.
        ``copy_time``
            a copy of the program appears and is moved and squeezed until it
            stands over the tape at the pitch of the cells.
        ``letter_time``
            it is written onto the cells, one instruction per
            ``letter_duration``, left to right, each keeping the colour it has
            had since the first machine scene. The tape is now the program.
        ``wipe_time``
            the original program and the box it was written in are wiped away
            left to right - the copy on the tape is the only one left.
        ``output_time``
            ``HELLO`` is carried out of the read-out and onto the cells at
            ``output_offset``, shrinking as it goes; then the empty box turns
            ``custom1`` and unrolls into an arrow standing over the ``O``.
        ``table_time``
            the last transform: the alphabet of the simple machine leaves to
            the right and the tape and its two arrows step down, making room
            for the ascii table the extended machine reads. The shot ends on
            the tape, a pointer, and the alphabet of the machine that comes
            next.

        The camera creeps for the reason every machine scene creeps: the whole
        animation is in the node tree and ``render_with_skips`` looks for
        f-curves - see :meth:`_run_simple_bff`.
        """
        t0 = 0
        _setup_render(hdri="cayley_interior_4k", transparent=True)

        program = BrainFuckSimpleModifier.HELLO_LOOP

        # the timeline, in one place. The letters take one letter_duration
        # each, so the wipe cannot start before the last of them has landed.
        copy_time = 5           # the copy of the program appears
        letter_time = 7         # its first letter reaches the tape
        letter_duration = 0.1   # ... and one more every tenth of a second
        letters_done = letter_time + len(program) * letter_duration
        wipe_time = letters_done + 0.4      # the original program is wiped
        wipe_duration = 1
        output_time = wipe_time + 1.5       # HELLO sets off for the tape
        move_duration = 1.5                 # ... and how long it takes
        output_done = output_time + move_duration
        recolor_time = output_done + 0.5    # the empty box turns custom1
        morph_time = recolor_time + 1       # ... and becomes an arrow
        morph_duration = 2
        table_time = morph_time + morph_duration + 0.5   # the tables change over
        table_duration = 1

        machine = Plane(name="BFTransition")
        # These scenes are lit almost entirely by emission on a black
        # background, so the palette has to glow a little to be seen at all -
        # the plain materials of define_materials() render nearly black.
        modifier = BrainFuckTransitionModifier(
            program=program, tape_size=5, cell_size=1.5, step_duration=0.001,
            start_time=t0, name="TransitionMachine",
            glyph_size=0.8,
            copy_program_time=copy_time,
            switch_letter_time=letter_time, letter_duration=letter_duration,
            program_disappear_time=wipe_time,
            program_disappear_duration=wipe_duration,
            output_offset=32, output_disappear_time=output_time,
            output_move_duration=move_duration,
            output_recolor_time=recolor_time,
            output_morph_time=morph_time, output_morph_duration=morph_duration,
            replace_code_table=table_time,
            replace_code_table_duration=table_duration,
            emission=0.6)

        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)
        _setup_standard_camera(distance=17, shift_x=0.5*5*1.5, shift_z=-1.75)

        steps, output, _ = BrainFuckSimpleModifier.simulate(program, 5)
        print("bf_to_bff: %d instructions, %d steps, prints %r onto cells %d-%d, "
              "arrow over cell %d, ends at %.1fs"
              % (len(program), steps, output, modifier.output_offset,
                 modifier.output_offset + len(output) - 1, modifier.arrow_cell,
                 morph_time + morph_duration))

        # --- the tape opens ---------------------------------------------
        t0 = 2
        tape_size_node = ibpy.get_geometry_node_from_modifier(modifier,label="TapeSize")
        ibpy.change_default_integer(tape_size_node, from_value=5, to_value=40,begin_time=t0,transition_time=1)

        cell_size_node = ibpy.get_geometry_node_from_modifier(modifier,label="CellSize")
        ibpy.change_default_value(cell_size_node, from_value=1.5, to_value=0.48,begin_time=t0,transition_time=1)

        tape_position_node = ibpy.get_geometry_node_from_modifier(modifier,label="TapePosition")
        ibpy.change_default_vector(tape_position_node, from_value=Vector([0, 0, 0]), to_value=Vector([-6, 0, 0]),
                                   begin_time=t0,transition_time=1)

        # --- move the copy of the program and resize it ------------------
        # it has to be in place before the first letter leaves it, and it
        # only exists from copy_time on, so it moves the moment it appears
        t0 = copy_time
        program_shift_node = ibpy.get_geometry_node_from_modifier(modifier,label="ProgramShift")
        shrink_font_node = ibpy.get_geometry_node_from_modifier(modifier,label="ShrinkFontSize")
        shrink_spacing_node = ibpy.get_geometry_node_from_modifier(modifier,label="ShrinkSpacing")

        ibpy.change_default_vector(program_shift_node, from_value=Vector([0, 0, 0]), to_value=Vector([-2, 0, 3.7]),
                                   begin_time=t0,transition_time=1)
        ibpy.change_default_value(shrink_font_node, from_value=1.0, to_value=0.75,begin_time=t0,transition_time=1)
        ibpy.change_default_value(shrink_spacing_node, from_value=1.0, to_value=0.985,begin_time=t0,transition_time=1)

        # everything from here on is driven by Scene Time inside the tree,
        # from the constructor arguments above - nothing left to keyframe
        self.t0 = table_time + table_duration + 1

    # -------------------------------------------------------------------
    def simple_brain_fuck3(self):
        """HELLO with a loop inside a loop: 8 as 2 x (2 x 2).

        ::

            ++[>++[>++<-]<-]>>.>+++++.<++++..+++.

        The inner loop ``[>++<-]`` is the one from :meth:`simple_brain_fuck2`,
        moved one cell along: it empties cell 1 into cell 2 at 2 per turn. The
        outer loop runs *that* twice, refilling cell 1 with ``>++`` each time
        round, so cell 2 grows by 4 per turn of the outer loop and ends at 8.

        Watch the head: it now sweeps across three cells rather than two, and
        the two counters run down one after the other in a pattern that
        repeats. 62 steps, against 49 for a single loop and 27 for none.
        """
        self._run_simple_bff(BrainFuckSimpleModifier.HELLO_LOOP2,
                             step_duration=0.35, name="SimpleBffLoop2")

    # -------------------------------------------------------------------
    def simple_brain_fuck4(self):
        """HELLO with three nested loops: 8 as 2 x (2 x (2 x 1)).

        ::

            ++[>++[>++[>+<-]<-]<-]>>>.>+++++.<++++..+++.

        The innermost loop adds a single ``+`` at a time, and each level above
        it multiplies by two. Every level needs a counter cell of its own, so
        the ``H`` is built in cell 3 and the tape is used to its full width -
        with five cells there is exactly room for the ``E`` in cell 4.

        This is where nesting stops paying: 107 steps to print five letters,
        four times the straight version, for a number that eight ``+`` write
        down directly. It is on screen because the *shape* is the point - it is
        the same construction as the two scenes before it, one level deeper,
        and the counters running down in step are worth watching.
        """
        self._run_simple_bff(BrainFuckSimpleModifier.HELLO_LOOP3,
                             step_duration=0.2, name="SimpleBffLoop3")

    def brain_fuck_extended(self):
        """The two-headed machine of the paper, on a 128 cell tape.

        Everything the one-headed scenes have, plus a second arrow. The tape is
        128 cells folded onto two lines of 64 - the two 64-byte programs of
        ``brainfuck/bff/`` - and both arrows roam the whole of it: the lower one
        is moved by ``<`` and ``>`` and is the one ``+``, ``-`` and the loops
        work on, the upper one is moved by ``{`` and ``}`` and does nothing but
        sit there and be copied to and from.

        That copying is the whole point. ``.`` writes the cell under the lower
        arrow into the cell under the upper one and ``,`` writes it back, which
        is how a program on this tape can write another program - which is what
        the paper is about, and what none of the one-headed scenes can do.

        Above the tape is the printable ascii table instead of a 26 letter
        alphabet, with the ten characters the machine reads as instructions
        drawn bold and in the colour of their family. It is the legend for the
        numbers in the cells: the program here builds 66 and copies it, then 70
        twice, which the table reads as ``B``, ``F``, ``F``.

        Nothing prints and nothing is read in, so there are no read-out boxes
        at all - the tape is the whole of the machine's state.
        """
        _setup_render()
        t0 = 0

        # 0.1 s an instruction, not 0.15: the copy is done after 256 of them
        # and the sub-scene has 27 seconds of running in it, so a tenth of a
        # second each is what fits the whole replication into the shot
        step_duration, start_time = 0.1, 3.0
        tape_size, cell_size = 64, 0.55

        machine = Plane(name="BffExtended")
        tape_files = ("replicator2", "food")
        modifier = BrainFuckExtendedModifier(tape_size=tape_size, cell_size=cell_size,
                                             step_duration=step_duration,
                                             start_time=start_time,
                                             tape_files=tape_files,
                                             name="BffExtendedMachine", emission=0.6)
        machine.add_mesh_modifier(type='NODES', node_modifier=modifier)
        machine.appear(begin_time=0, transition_time=0)

        # 64 cells across is a wide, shallow strip, so the camera is set by the
        # width and the tape ends up in a band across the middle of the frame

        _setup_standard_camera(distance=40, shift_x=0, shift_z=0)
        ibpy.camera_zoom(lens=38, begin_time=t0, transition_time=0)

        # How far the machine gets in the time this sub-scene has. A replicator
        # does not halt - it is still copying when the clock runs out - so the
        # length of the animation is a decision rather than something to read
        # off the program, and what the python run is for here is to say what
        # the tape should look like when the camera stops.
        duration = self.sub_scenes["brain_fuck_extended"]["duration"]
        steps = int((duration - start_time - 2) / step_duration)
        memory = _read_tapes(tape_files)
        done, ends, head0, head1, counter = BFFNode.simulate(memory, steps=steps)
        copied = ends[tape_size:] == memory[:tape_size]
        print("brain_fuck_my_extended: %d of %d cells hold an instruction, "
              "%d steps run, heads at %d and %d, counter at %d, tape 2 %s tape 1"
              % (sum(1 for v in memory[:tape_size]
                     if chr(v) in BFFNode.COMMANDS),
                 tape_size, done, head0, head1, counter,
                 "is a copy of" if copied else "is not yet a copy of"))
        run_time = start_time + steps * step_duration

        self.t0 = run_time + 2

    # -------------------------------------------------------------------
    def plot(self):
        """The order parameter: how much space the soup needs, epoch by epoch.

        The plot ``script.md``'s "The order parameter" section asks for -
        "the file size of our soup as a function of time ... the sharp
        transition from chaos to order around epoch 8600" - built the way
        ``video_cmb/cmb.py``'s ``power_spectrum`` builds the Planck spectrum:
        a :class:`CoordinateSystem2` with the measurement in it as a
        :class:`Data` polyline, and no perspective anywhere. That is
        deliberate, and ``Youtube.md`` §4 is emphatic about it: this plot is
        the *evidence* for the whole video, so it has to read as evidence and
        not as an effect.

        The numbers come from ``data/compression.csv``, which
        ``brainfuck/bff/soup_watcher.py`` writes one row per epoch while the
        soup runs (see :func:`_read_compression`). Nothing about the run is
        written into this method: the number of tapes comes from
        :func:`_raw_size`, the length of the run from the last row, and the
        epoch of the transition from :func:`_transition_epoch`, so re-running
        the soup and dropping in a new csv re-scales the axes and re-times the
        annotations by itself.

        ``Data`` reveals itself left to right - its modifier hides every point
        whose x lies further along the domain than the elapsed fraction of
        ``[T0, T1]`` - so drawing the curve in *is* the animation, and the
        marker at the transition, whose two vertices share one x, springs into
        being at the exact instant the curve reaches the cliff rather than
        being keyframed to a time computed here. Only the caption needs a
        clock, and it takes it from the same mapping.
        """
        duration = self.sub_scenes["plot"]["duration"]
        _setup_render(hdri="cayley_interior_4k", transparent=True)
        _setup_standard_camera(distance=20)
        ibpy.set_camera_lens(lens=45)

        epochs, sizes = _read_compression()
        raw = _raw_size(sizes)
        transition = _transition_epoch(epochs, sizes)

        # kilobytes, not kibibytes: the y axis is read by an audience, and a
        # 256 that is really 262144 invites the wrong arithmetic against the
        # "4096 times 64 bytes" the script says out loud
        kilo = 1000
        x_max = epochs[-1]
        y_max = 1.1 * raw / kilo
        x_step, y_step = _nice_step(x_max), _nice_step(y_max)

        # 11 x 6.2 units in a frame that a 45 mm lens 20 units back fills out
        # to x in [-8, 8], z in [-4.5, 4.5]: wide enough that 10000 epochs are
        # not squeezed, and clear of both edges - the tic labels of the x axis
        # hang 0.6 below it, so an axis on the floor of the frame would have
        # its labels rendered off-screen
        origin, width, height = Vector((-5.5, 0, -3.35)), 11, 6.1

        def world(epoch, kilobytes):
            """Where a data point of the plot lands in world coordinates.

            The axes and the data rows are children of the coordinate system
            and are placed by its modifiers; the captions are not, so they
            need the same mapping applied by hand.
            """
            return origin + Vector((width * epoch / x_max, 0,
                                    height * kilobytes / y_max))

        title = SimpleTexBObject(r"\text{How much space does the soup need?}",
                                 aligned='center', text_size='large',
                                 color='important', location=[0, 0, 4],
                                 emission=0.1)
        t0 = 0.5 + title.write(begin_time=0, transition_time=1.5)

        coords = CoordinateSystem2(
            location=origin, lengths=[width, height], colors=['text', 'text'],
            domains=[[0, x_max], [0, y_max]], tic_label_digits=[0, 0],
            tic_labels=[_tic_labels(x_step, x_max + x_step / 2),
                        _tic_labels(y_step, y_max)],
            axes_labels={r"\text{epoch}": [-0.5, 0, width + 0.4],
                         r"\text{compressed size [kB]}": [0.25, 0, height + 0.35]},
            # tic labels are anchored 12 axis radii to the side of their tic
            # and, with the default 'left', run from there into the plot -
            # which puts three digit labels on top of the y axis. Centering
            # them instead is what lets the x labels sit under their tics, and
            # the shift then moves the y column clear of the axis
            aligned='center', tic_label_shifts=[Vector(), [-1.3, 0, 0]])
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=3)

        # what the soup costs when it is still noise, as a line to fall away
        # from - the plot only says something once there is something to
        # compare the curve against
        raw_line = Data(data=[[0, 0, raw / kilo], [x_max, 0, raw / kilo]],
                        coordinate_system=coords, name="RawSize",
                        material="drawing", emission=0.3, linesize=0.5,
                        width=width, height=height)
        raw_label = SimpleTexBObject(r"%d\times 64\text{ bytes of noise}" % (raw // 64),
                                     color='drawing', text_size='normal',
                                     location=world(0.32 * x_max, raw / kilo + 0.04 * y_max),
                                     emission=0.1)
        raw_line.appear(begin_time=t0, transition_time=1.5)
        t0 = 0.5 + raw_label.write(begin_time=t0, transition_time=1.5)

        # the curve draws itself in for as long as the shot has left, minus a
        # tail to hold on the finished plot
        tail = 4
        sweep = duration - t0 - tail
        curve = Data(data=[[e, 0, s / kilo] for e, s in zip(epochs, sizes)],
                     coordinate_system=coords, name="Compression",
                     material="example", emission=0.5, linesize=1,
                     width=width, height=height)
        curve.appear(begin_time=t0, transition_time=sweep)

        marker = Data(data=[[transition, 0, 0], [transition, 0, y_max]],
                      coordinate_system=coords, name="Transition",
                      material="important", emission=0.5, linesize=0.5,
                      width=width, height=height)
        marker.appear(begin_time=t0, transition_time=sweep)

        # the sweep is linear in the epoch, so the moment the curve falls is
        # where the transition sits in the run
        t_cliff = t0 + sweep * transition / x_max
        caption = SimpleTexBObject(r"\text{phase transition, epoch }%d"
                                   % round(transition, -2),
                                   color='important', text_size='normal',
                                   aligned='right', emission=0.1,
                                   location=world(0.97 * transition, 0.62 * y_max))
        caption.write(begin_time=t_cliff + 0.2, transition_time=1)

        print("plot: %d bytes of soup (%d tapes), %d epochs recorded, "
              "compressed to %d bytes at its smallest, transition at epoch %d "
              "(%.1fs into the shot)"
              % (raw, raw // 64, x_max, min(sizes), transition, t_cliff))

        self.t0 = t0 + sweep + tail

    # -------------------------------------------------------------------
    def temple_person(self):
        """A walk down a temple that is being drawn as it is walked through.

        There is no temple asset and this scene does not try to fake one.
        Instead the building is a drawing that happens to be
        three-dimensional: a hundred and forty pen strokes hung in space,
        every one of them a bezier tube a few centimetres thick that is
        *grown* rather than switched on, so the shot is not a temple
        appearing but a temple being drawn. See :func:`_temple_drawing` for
        what the strokes are and :func:`_pen` for why none of them is
        straight.

        The idea the shot is built on is that the pen works **ahead of the
        walker**. Every stroke knows where along the aisle it stands, and
        that is what schedules it: a column is drawn when the figure is
        ``lead`` units short of it, so there is always construction
        happening at the far edge of the frame and never any waiting for
        the walk to catch up. Behind the figure nothing is taken away --
        what they have walked through stays drawn, and the last beat lifts
        the camera off the floor to show that it added up to a building.

        The long lines are the exception, and they are the shot's one real
        trick: the six lines of the stylobate, the eight of the
        entablature and the ridge are not drawn bay by bay but in a single
        pull that starts early and moves *faster* than the walk. They
        arrive at the vanishing point ahead of the figure, which is what
        makes the perspective read; grown at walking pace they would only
        ever be a line ending beside them.

        **Constant speed, on purpose.** The dolly, the walker and every
        stroke's cue are locked to one another, and the lock only holds if
        the schedule below and Blender's keyframes agree about where the
        figure is at a given second. Blender's default two-keyframe ease
        would put it somewhere else for all but three instants of the
        move, so the walk keyframes are linearised (:func:`_linearize_from`)
        and the schedule can then be plain arithmetic. It also happens to
        be what a walk looks like: people do not ease in.

        **The figure goes in later.** The whole shot is framed around an
        empty named ``TempleWalker``, which travels the aisle on the
        timings printed at the end of this method, so a
        :class:`PersonWithCape` can be dropped in without a single number
        here changing -- see the commented block below. Nothing else in the
        scene refers to the figure: an empty aisle is a real shot of a
        temple, and the same shot with someone in it is a real shot of a
        walk.
        """
        duration = self.sub_scenes["temple_person"]["duration"]
        _setup_render()
        # ink on black: the strokes are the only light in the scene, and the
        # bloom is what turns a thin tube into something with a nib's weight
        create_glow_composition(threshold=0.4, strength=0.6, size=6)
        ibpy.set_camera_lens(lens=28)  # a wide lens is half of an interior

        geometry = TEMPLE
        bay, length = geometry['bay'], geometry['n_bays'] * geometry['bay']
        altar_x = length + 3.4

        # --- the walk --------------------------------------------------
        walk_start, walk_end = -5.0, length + 0.9  # stops short of the altar
        t_walk, walk_time = 0.6, 10.0
        speed = (walk_end - walk_start) / walk_time

        def walker_at(x):
            """The second at which the figure passes ``x``, clamped to the walk."""
            return t_walk + (min(max(x, walk_start), walk_end) - walk_start) / speed

        # the figure keeps a little to one side of the axis and the camera to
        # the other: dead centre would put them in front of the vanishing
        # point, which is the one thing in frame they must not stand on
        walker = EmptyCube(location=Vector((walk_start, 1.4, 1.7)),
                           name='TempleWalker')
        walker.appear(begin_time=0, transition_time=0)

        # far enough back that the figure is a figure and not a foreground,
        # aimed almost straight down the aisle and a little above the horizon:
        # near-one-point perspective, with the columns leaning in overhead
        ibpy.set_camera_location(location=Vector((walk_start - 15.0, -2.6, 3.6)))
        target = EmptyCube(location=Vector((walk_start + 14.0, -0.2, 7.4)),
                           name='TempleView')
        ibpy.set_camera_view_to(target)

        # camera, aim point and figure all take the same shift over the same
        # window: the relationship between them never changes, so the only
        # motion in frame is the building going past
        shift = Vector((walk_end - walk_start, 0, 0))
        ibpy.camera_move(shift=shift, begin_time=t_walk, transition_time=walk_time)
        target.move(direction=shift, begin_time=t_walk, transition_time=walk_time)
        walker.move(direction=shift, begin_time=t_walk, transition_time=walk_time)
        for follower in (ibpy.get_camera(), target.ref_obj, walker.ref_obj):
            _linearize_from(follower, t_walk * FRAME_RATE)

        # --- the figure ------------------------------------------------
        # PersonWithCape rides the same three lines as the camera and the aim
        # point, so it needs nothing from the schedule below. The cape wants a
        # bake window that covers the walk, and the rotation turns the figure
        # to face the way it is going (check which way the asset faces at
        # rest -- this assumes +y, so a quarter turn puts it on +x).
        #
        from objects.derived_objects.person_with_cape import PersonWithCape
        person = PersonWithCape(location=Vector((walk_start, 1.4, 0)),
                                rotation_euler=[0, 0, -pi / 2],
                                colors=['gray_8', 'important'],
                                simulation_start=0,
                                simulation_duration=t_walk + walk_time + 2)
        person.appear(begin_time=0, transition_time=0)
        person.move(direction=shift, begin_time=t_walk, transition_time=walk_time)
        _linearize_from(person.ref_obj, t_walk * FRAME_RATE)

        # --- the drawing -----------------------------------------------
        # one seed, so the same hand draws the same temple every render
        rng = np.random.default_rng(20260806)
        strokes = _temple_drawing(rng, geometry)

        # cool blue underfoot, white for the stone that stands up, and the
        # altar the one warm thing in the building
        style = {
            'ground': dict(color='drawing', emission=0.5, thickness=0.5),
            'column': dict(color='text', emission=0.9, thickness=0.6),
            'beam': dict(color='text', emission=0.9, thickness=0.6),
            'roof': dict(color='drawing', emission=0.7, thickness=0.5),
            'altar': dict(color='important', emission=1.6, thickness=0.7),
        }
        lead = 15.0  # how far ahead of the figure the pen works
        t_sweep = 8.6  # one pull for the lines that run the whole building
        last = 0

        for index, stroke in enumerate(strokes):
            part = stroke['part']
            curve = _ink(stroke['points'], 'Temple_%s_%d' % (part, index),
                         extrude=0, **style[part])
            if stroke['sweep']:
                # starts before the walk and outruns it
                begin, transition = 0.3, t_sweep
            elif part == 'ground':
                begin, transition = 0.15 * stroke['order'], 1.4
            elif part == 'altar':
                begin = walker_at(altar_x - 14) + 0.12 * stroke['order']
                transition = 1.2
            else:
                # the pen is a fixed distance up the aisle; the first bays are
                # already around the camera when the shot opens, so they get a
                # stagger of their own rather than all arriving at t_walk
                begin = max(walker_at(stroke['x'] - lead),
                            0.5 + 0.3 * stroke['x'] / bay)
                begin += 0.07 * stroke['order']
                transition = 0.9 if part == 'column' else 0.7
            curve.grow(begin_time=begin, transition_time=transition)
            last = max(last, begin + transition)

        # --- the last beat ---------------------------------------------
        # the figure stops in front of the altar and the camera does not: it
        # leaves sideways through the colonnade and pulls back until the whole
        # thing is in frame at once. What has been a corridor for ten seconds
        # is a building for the last four.
        #
        # The exit is worth a number. Leaving at 37 units of -y over the same
        # 12 of height, the camera crosses the line of the columns about 8% of
        # the way through the move, and 8% of the +2 in x lands it at x = 13.6:
        # halfway between the columns at 11 and 16.5, which is the only reason
        # it goes through the gap instead of through a column and its bloom.
        t_end = t_walk + walk_time
        t_lift = duration - t_end - 0.6
        ibpy.camera_move(shift=[2.0, -37.4, 12.4], begin_time=t_end,
                         transition_time=t_lift)
        # ... and the aim point falls back to the middle of the building, so
        # the pull-out ends on the elevation: colonnade broadside, roof line
        # over it, the altar the one warm thing at the end of it
        target.move(direction=Vector((16.0 - (walk_end + 14.0), 0.2, -1.4)),
                    begin_time=t_end, transition_time=t_lift)

        print("temple_person: %d strokes, %d columns, aisle %.1f long, "
              "walk %.1f -> %.1f at %.2f u/s from t=%.1f to t=%.1f, "
              "last stroke finished at t=%.1f"
              % (len(strokes), 2 * (geometry['n_bays'] + 1), length,
                 walk_start, walk_end, speed, t_walk, t_end, last))

        self.t0 = duration

    # -------------------------------------------------------------------
    def microscope_person(self):
        """Two figures walking a circle round a microscope three times their size.

        Same hand as :meth:`temple_person` -- :func:`_pen`, :func:`_ring`,
        :func:`_ink`, nothing solid anywhere -- and the same idea that a
        drawing is something that gets drawn: the instrument assembles
        bottom upwards out of its own strokes (:func:`_microscope_drawing`
        schedules by height), the floor circle first of all, the specimen
        last of all and glowing.

        **Why the cape points outwards.** The requirement is about the cape,
        but the cape is not what gets animated: it hangs off the figure's
        back, at ``-y`` in the figure's own coordinates, so pointing it away
        from the middle is a statement about which way the figure is turned.
        Standing at angle ``t`` on the circle, the outward direction is
        ``(cos t, sin t)``, and the figure's local ``-y`` lands there when it
        is turned by ``t + pi/2`` -- which is also the turn that has it
        facing the microscope. So the two are the same instruction: keep
        looking at the thing in the middle and the cape takes care of
        itself. A figure walking a circle the ordinary way, facing where it
        is going, would trail its cape sideways instead.

        Turning is not quite enough, though, because the cape is cloth and
        cloth answers to physics, not to intent: dragged round a circle it
        trails behind the shoulders, and a trailing cape hangs on the chord,
        which is *inside* the ring. Hence the force field at the centre. A
        Blender ``FORCE`` effector pushes radially away from its own origin
        with no falloff by default, so one in the middle of the circle leans
        every cape outwards, equally, wherever its owner has got to -- the
        shape of the effector is the shape of the requirement. It is also
        why the walk is slow: at two units a second there is not much lag
        for the field to have to cancel.

        ``strength`` is fifteen thousand because that is what it measured
        at, and the number is worth writing down because it is not the
        number anyone would guess. Cape-scale forces here are three orders
        of magnitude above gravity's nine-point-eight: simulated through the
        walk and measured as the offset of the cape's bulk from the
        shoulders it hangs on, 400 still leaves it hanging (0.06 units
        *inwards*, the lag of the walk), 4000 only cancels that (+0.02), and
        15000 leans it out and up -- +0.12 out, 0.26 down, twenty-five
        degrees off vertical, the same to within a hundredth on both
        figures. Turn it up for more; there is no cliff on the way there,
        only a cape that gets flatter.

        Underneath the cloth the geometry is exact rather than approximate:
        the cape's outward axis matches the radial direction to 1.000 at
        every point of the walk, so whatever the simulation does with it, it
        does it on the far side of the figure from the microscope.

        **Half a turn, not a whole one.** A full circuit of a nine-unit ring
        in the time available would have them jogging. Half of it is a walk,
        and it ends with each figure standing where the other started, which
        is a better place to end than back where you began.
        """
        duration = self.sub_scenes["microscope_person"]["duration"]
        _setup_render()
        create_glow_composition(threshold=0.4, strength=0.6, size=6)
        ibpy.set_camera_lens(lens=25)
        # the strokes light themselves, but the two figures are the only solid
        # things in the shot and an unlit person is a grey lump: this rig is
        # here for them and for the shading that makes a cape read as cloth
        _light_hero(target=(0, 0, 2.5), strength=0.5, ambient=0.35)

        arena = MICROSCOPE['arena']

        # low and wide to start, so the instrument stands over the horizon;
        # the move is a straight translation past it rather than a true orbit,
        # which at this distance reads as one anyway and costs two keyframes
        # high enough to look down on whoever is on the near side of the ring:
        # at eye level a figure passing between the camera and the instrument
        # is nine units away and fills the frame, and this asset is a smooth
        # abstract shape that does not repay being that close
        ibpy.set_camera_location(location=Vector((-13.5, -18.0, 4.2)))
        target = EmptyCube(location=Vector((0, 0, 4.6)), name='MicroscopeView')
        ibpy.set_camera_view_to(target)
        ibpy.camera_move(shift=[21.0, -3.0, 4.4], begin_time=0,
                         transition_time=duration - 0.4)

        # --- the drawing -------------------------------------------------
        rng = np.random.default_rng(20260807)
        strokes = _microscope_drawing(rng)
        style = {
            'arena': dict(color='drawing', emission=0.5, thickness=0.5),
            'body': dict(color='text', emission=0.9, thickness=0.6),
            'specimen': dict(color='important', emission=1.8, thickness=0.7),
        }
        t_draw, span = 0.6, 4.8
        for index, stroke in enumerate(strokes):
            part = stroke['part']
            curve = _ink(stroke['points'], 'Scope_%s_%d' % (part, index),
                         extrude=0, **style[part])
            if stroke['sweep']:
                begin, transition = 0.1, 2.4
            elif part == 'specimen':
                begin, transition = t_draw + span + 0.35, 0.8
            else:
                height = min(max(stroke['z'] / MICROSCOPE['height'], 0.0), 1.0)
                begin = t_draw + span * height
                transition = 0.7
            curve.grow(begin_time=begin, transition_time=transition)

        # --- the two who came to look ------------------------------------
        t_appear, t_walk = 5.4, 6.3
        walk_time = duration - t_walk - 0.7
        turn = pi  # half the ring each, so they swap sides
        steps = 48  # a 3.75-degree polygon: rounder than the ink

        # the wind that holds the capes out. It is a point force at the middle
        # of the circle, which is exactly the shape of the requirement: away
        # from the microscope, wherever you are standing.
        force = Force(name='CapeWind', location=[0, 0, 2.1], strength=15000)
        force.appear(begin_time=0, transition_time=0)

        for k, cape_colour in enumerate(('important', 'joker')):
            phase = k * pi
            person = PersonWithCape(
                location=Vector((arena * np.cos(phase), arena * np.sin(phase), 0)),
                rotation_euler=[0, 0, phase + pi / 2],
                colors=['gray_8', cape_colour],
                name='Visitor%d' % k,
                # the cloth cache has to start when the figure does, not at
                # frame zero: cloth only steps from the frame it last
                # simulated, so a cache that opens while the cape is still
                # hidden never advances at all and the cape stays in its rest
                # pose for the whole shot -- silently, and looking exactly
                # like a force field that is not working
                simulation_start=t_appear + 0.1,
                simulation_duration=duration - t_appear)
            person.appear(begin_time=t_appear, transition_time=0.9)
            # person.rescale(rescale=3,begin_time=0,transition_time=0)

            # the orbit, keyframed by hand: a constraint would have to be
            # talked out of aligning the figure with its own direction of
            # travel, which is the one thing this walk must not do
            walker = person.ref_obj
            for i in range(steps + 1):
                angle = phase + turn * i / steps
                walker.location = Vector((arena * np.cos(angle),
                                          arena * np.sin(angle), 0))
                walker.rotation_euler = [0, 0, angle + pi / 2]
                frame = int((t_walk + walk_time * i / steps) * FRAME_RATE)
                walker.keyframe_insert(data_path='location', frame=frame)
                walker.keyframe_insert(data_path='rotation_euler', frame=frame)
            # forty-eight eased hops would be a stagger; one linear sweep is a walk
            _linearize_from(walker, t_walk * FRAME_RATE)

        print("microscope_person: %d strokes, %.1f units tall, two figures on "
              "r=%.1f walking %.0f degrees in %.1fs (%.2f u/s), cape force %d"
              % (len(strokes), MICROSCOPE['height'], arena, np.degrees(turn),
                 walk_time, turn * arena / walk_time, force.strength))

        self.t0 = duration


if __name__ == '__main__':
    try:
        example = BffScene()
        dictionary = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dictionary[i] = scene
        if len(dictionary) == 1:
            selection = 0
        else:
            selection = input("Choose scene:")
            if len(selection) == 0:
                selection = 0
        print("Your choice: ", selection)
        selected_scene = dictionary[int(selection)]

        example.create(name=selected_scene, resolution=[1920, 1080],
                       start_at_zero=True)
    except Exception:
        print_time_report()
        raise
