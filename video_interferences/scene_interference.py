import os
import sys
from collections import OrderedDict
from math import pi, tau

import bpy
import numpy as np

from appearance.textures import get_texture
from interface.interface_constants import BLENDER_EEVEE
from objects.bderivation import BDerivation
from objects.choreography import highlight_letters
from objects.derived_objects.drum import Drum, DrumModeModifier
from objects.derived_objects.pencil import Pencil
from objects.derived_objects.whistle import Whistle
from objects.derived_objects.magnifying_glass import MagnifyingGlass
from objects.derived_objects.ticker import Ticker
from objects.geometry.sphere import Sphere
from objects.slider import BSlider

# Allow running as a plain script from inside this folder as well as being
# imported as ``video_interferences.scene_interference`` (the workspace
# convention, see video_bff/scene_bff.py).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compositions.compositions import create_glow_composition, set_alpha_composition
from geometry_nodes.modifier_video_interferences import (SpatialDistributionModifier,
                                                         InterferenceModifier,
                                                         RealInterferenceModifier,
                                                         GaussianCloudModifier,
                                                         PolarGridModifier,
                                                         FarFieldModifier,
                                                         WaveVisualizationModifier,
                                                         Slicer, AiryDiscModifier, AcousticModifier,
                                                         BesselVisualizer,
                                                         GravitationalWavesModifier,
                                                         ElectromagneticWaveModifier)
from geometry_nodes.nodes import _BESSEL_F0, _BESSEL_TH, _J0_SMALL
from interface import ibpy
from interface.ibpy import Vector
from objects.cube import Cube
from objects.curve import BezierDataCurve
from objects.empties import EmptyCube
from objects.functions import GeoFunction
from objects.bobject import BObject
from objects.codeparser import CodeParser
from objects.display import CodeDisplay
from objects.coordinate_system import CoordinateSystem2
from objects.import_bobject import ImportBObject
from objects.logo import LogoFromInstances
from objects.pen2curve import Pen2CurveObject
from objects.plane import Plane
from objects.tex_bobject import SimpleTexBObject
from perform.scene import Scene
from utils.constants import COLOR_NAMES, COLORS_SCALED, FRAME_RATE
from utils.utils import print_time_report


# ===========================================================================
#  Shared helpers (same conventions as video_bff/scene_bff.py)
# ===========================================================================
def _set_world_background(color="background"):
    """Make camera rays see ``color`` instead of blender's default grey.

    ``initialize_blender`` builds a world of ``Mix(Factor = Is Camera Ray,
    A = <project background>)`` and leaves **B unconnected**, so camera rays
    (Factor = 1) see B's default 0.5 grey. Scenes that call
    ``set_hdri_background`` replace the whole world and never notice; these
    keep the default one. Written up at length in ``video_bff/scene_bff.py``.
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


def _setup_render(transparent=True, background="background", exposure=1.0):
    """Eevee, black world, no denoising - these scenes are their own light."""
    ibpy.set_hdri_background("forest", "exr", simple=True, transparent=True)
    ibpy.set_render_engine(denoising=False, transparent=transparent,
                           resolution_percentage=100, engine="BLENDER_EEVEE",
                           taa_render_samples=128, frame_start=1,
                           exposure=exposure)


def _lights(target=(0, 0, 0), strength=1.0):
    """Two suns and nothing else.

    A cloud of ten thousand small emissive spheres lights itself; what it
    cannot do is tell the eye which of them are nearer. That is what these are
    for - a key from over the camera's left shoulder to put a gradient on each
    sphere, and a rim from behind to draw the far side of the cloud as
    outlines rather than dots. Suns rather than lamps because the cloud is
    four units deep and a lamp close enough to model the near face falls off
    across the far one.
    """
    for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)

    target = Vector(target)
    lamps = {}
    for name, location, color, energy in (
            ("key", (-6, -9, 7), (1.0, 0.96, 0.9), 3.0),
            ("rim", (5, 8, 4), (0.75, 0.85, 1.0), 2.0)):
        bpy.ops.object.light_add(type='SUN', location=location)
        lamp = bpy.context.object
        lamp.name = lamp.data.name = name
        lamp.data.color = color
        lamp.data.energy = energy * strength
        lamp.data.angle = np.radians(8)
        # a sun shines along its own -z and ignores where it is, so the
        # location above is only a way of writing down a direction
        lamp.rotation_euler = Vector((0.0, 0.0, -1.0)).rotation_difference(
            target - Vector(location)).to_euler()
        lamps[name] = lamp
    return lamps


# where the camera of the current scene stands, so that _title can turn to
# face it without every scene having to say so twice
_CAMERA_LOCATION = Vector((0, -13, 0))


def _camera(location=(7.5, -9.5, 5.5), target=(0, 0, 0), lens=38):
    """Three-quarter view, which is the only one that shows a cube is a cube."""
    global _CAMERA_LOCATION
    _CAMERA_LOCATION = Vector(location)
    ibpy.set_camera_location(location=Vector(location))
    empty = EmptyCube(location=Vector(target))
    ibpy.set_camera_view_to(empty)
    ibpy.set_camera_lens(lens=lens)
    return empty


def _linearize(host, frame=0):
    """Make the node-tree keyframes from ``frame`` on interpolate linearly.

    A phase that eases in and out does not march, it breathes; a fringe
    pattern travelling across the box has to move at a constant rate or it
    reads as an animation of the *dials* rather than of the physics.
    ``iter_action_fcurves`` is the blender-5 accessor - layered actions have
    no ``action.fcurves`` any more.

    ``host`` is a geometry-nodes modifier or, for the shader scenes, a
    material's ``node_tree`` directly - the keyframes sit on the tree either
    way, only the route to it differs.
    """
    tree = host.get_node_tree() if hasattr(host, "get_node_tree") else host
    anim = getattr(tree, "animation_data", None)
    if anim is None or anim.action is None:
        return
    for fcurve in ibpy.iter_action_fcurves(anim.action):
        for keyframe in fcurve.keyframe_points:
            if keyframe.co[0] >= frame:
                keyframe.interpolation = 'LINEAR'


def _ramp_through(slot, values, begin_time=0, transition_time=1):
    """Keyframe a dial through a whole curve rather than between two ends.

    ``ibpy.change_default_value`` gives a dial a start and a finish, which is
    all a linear sweep needs. A dial that has to follow a *shape* - a
    wavelength that sweeps linearly in k rather than in lambda, an exposure
    that has to dip where the array resonates - needs the curve sampled, and
    this lays those samples down as evenly spaced keyframes. Only the first
    segment writes its start value; the rest inherit it from the segment
    before, so the result is ``len(values)`` keyframes and not twice that.

    Run :func:`_linearize` afterwards, or the eased default turns the
    piecewise-linear curve into a string of little accelerations.
    """
    steps = len(values) - 1
    span = transition_time / steps
    for i in range(steps):
        ibpy.change_default_value(slot,
                                  from_value=values[0] if i == 0 else None,
                                  to_value=values[i + 1],
                                  begin_time=begin_time + i * span,
                                  transition_time=span)
    return begin_time + transition_time


def _turntable(host, begin_time=0, transition_time=9, turns=1, base=(0, 0, 0)):
    """One steady revolution of an object about the world z axis.

    ``BObject.rotate(interpolation='LINEAR')`` cannot do the steady part on
    blender 5: it reaches for ``action.fcurves``, which layered actions no
    longer have, and raises. So the rotation goes in on the eased default and
    the keyframes are flattened afterwards through
    :func:`ibpy.iter_action_fcurves`, the accessor that does work. Eased is
    wrong here for the same reason it is wrong for the phase - a cloud that
    slows to a halt at the end of a turn reads as a camera move that ran out
    of road.

    ``base`` is the pose the object is to keep while it turns, and it is added
    to rather than replaced because ``rotate`` takes an *absolute* euler: a
    model posed at ``[tilt, 0, 0]`` and sent to ``[0, 0, tau]`` would snap
    upright on the first frame. Only the third angle is swept, which is what
    keeps this a turn about the world vertical - blender's XYZ euler applies
    ``R = Rz . Ry . Rx``, so the z rotation is the outermost one and the pose
    set by the other two rides round inside it. That is also why a tilted
    model traces a cone here rather than spinning about its own long axis.
    """
    base = Vector(base)
    host.rotate(rotation_euler=[base.x, base.y, base.z + turns * tau],
                begin_time=begin_time, transition_time=transition_time)
    for holder in (host.ref_obj, getattr(host.ref_obj, "data", None)):
        anim = getattr(holder, "animation_data", None)
        if anim is None or anim.action is None:
            continue
        for fcurve in ibpy.iter_action_fcurves(anim.action):
            for keyframe in fcurve.keyframe_points:
                if keyframe.co[0] >= begin_time * FRAME_RATE:
                    keyframe.interpolation = 'LINEAR'
    return begin_time + transition_time


def _cloud(modifier, name="Cloud", location=(0, 0, 0)):
    """Hang a distribution modifier on an object and report what it built."""
    host = Plane(name=name, location=Vector(location))
    host.add_mesh_modifier(type='NODES', node_modifier=modifier)
    host.appear(begin_time=0, transition_time=0)
    print("%-22s %s: %d candidates, acceptance <f> = %.3f, "
          "expected %d points"
          % (name, type(modifier).__name__, modifier.candidates,
             modifier.mean_density, round(modifier.expected_count())))
    return host


def _panel(modifier, name="Panel", **kwargs):
    """Hang a node modifier on a plane standing at ``location``.

    The sibling of :func:`_cloud` for the modifiers that draw a *picture*
    rather than a cloud: the host's own geometry is thrown away by the tree
    (none of them takes a group input), so the plane is only a place to stand
    and a set of material slots. Appearing is left to the caller, which is
    what lets two panels of the same modifier come up together.

    Two things the cloud helper does not have to care about and this one
    does, both because the modifier draws in its *own* local x-y plane:

    ``apply_location=False``, because a ``Plane`` bakes its location into the
    mesh by default and leaves the object at the origin - which is exactly
    what a tree that ignores the incoming mesh cannot see. The panel has to
    be moved by the object transform or it is not moved at all.

    And the quarter turn about x, which stands the drawing up in the x-z
    plane facing the camera - the same thing ``SimpleTexBObject`` does to
    glyphs, and for the same reason: flat on the floor, an oblique camera
    sees a picture edge-on as a line.
    """
    host = Plane(name=name, apply_location=False, **kwargs)

    host.add_mesh_modifier(type='NODES', node_modifier=modifier)
    return host


def _title(text, location=(0, 0, 4), color="text", size="normal"):
    """A line of type above the box, turned to face the camera.

    Two things it has to get right, neither of them about the words:

    z = 2.6 rather than anything higher because the shot is framed on the box.
    A 38 mm lens 14 units back holds roughly z in [-3.5, 3.5] at the origin, so
    a title parked well clear of the cube is also parked outside the frame.

    And the rotation: glyphs arrive from the svg lying in the x-y plane, and
    ``SimpleTexBObject`` stands them up with a default ``rotation_euler`` of
    ``[pi/2, 0, 0]`` - so text faces -y, face-on only to a camera that looks
    straight down +y. These scenes look at the cube from a corner, where flat
    text is a slanted, foreshortened line that reads as a mistake.

    The yaw below turns it back to the camera, and it is written as the
    *third* euler angle on top of that same pi/2, not as a replacement for it:
    blender's XYZ euler applies R = Rz . Ry . Rx, so ``[pi/2, 0, yaw]`` stands
    the text up first and then swings it round the vertical. Passing a
    rotation that does not carry the pi/2 lays the title flat on the floor,
    where an oblique camera sees it edge-on as a dotted line.
    """
    location = Vector(location)
    direction = _CAMERA_LOCATION - location
    # after the pi/2 the text's normal is (0, -1, 0); Rz(yaw) sends that to
    # (sin yaw, -cos yaw, 0), which is the horizontal direction to the camera
    # exactly when yaw = atan2(dx, -dy)
    yaw = np.arctan2(direction.x, -direction.y)
    return SimpleTexBObject(r"\text{%s}" % text, text_size=size, color=color,
                            location=location, aligned="center",
                            rotation_euler=[pi / 2, 0, yaw])


# ===========================================================================
#  A hand, borrowed from video_bff/scene_bff.py
# ===========================================================================
# The temple and the microscope over there are drawn rather than built, out of
# three functions: a stroke that bows and wobbles, a curve object that can be
# grown, and a schedule that decides what is drawn when. The envelope below is
# the same hand and the same three functions, kept here rather than imported
# because scene_bff pulls in a whole video's worth of machinery to get at
# thirty lines of geometry.


def _frame_of(axis):
    """A right-handed frame with ``axis`` as its third vector."""
    axis = Vector(axis).normalized()
    reference = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))
    u = axis.cross(reference).normalized()
    return axis, u, axis.cross(u).normalized()


def _pen(a, b, rng, bow=0.012, overshoot=0.02, jitter=0.006, samples=7):
    """One straight-ish stroke of a pen from ``a`` to ``b``.

    A ruled line reads as CAD. Three things make a line read as *drawn*: it
    bows a little between its ends, it wobbles on the way, and the hand does
    not stop exactly on the corner. Bow and overshoot are fractions of the
    stroke's own length, so a long edge and a short seam come out of the same
    hand; the bow is a half sine and therefore vanishes at both ends, which is
    what lets corners still meet.
    """
    a, b = Vector(a), Vector(b)
    span = b - a
    length = span.length
    direction = span / length
    a = a - direction * length * overshoot * rng.uniform(0.2, 1.0)
    b = b + direction * length * overshoot * rng.uniform(0.2, 1.0)

    _, n1, n2 = _frame_of(direction)
    sag1, sag2 = rng.normal(0, bow * length, 2)

    points = []
    for i in range(samples):
        t = i / (samples - 1)
        arc = np.sin(pi * t)
        point = a.lerp(b, t)
        point = point + n1 * (sag1 * arc + rng.normal(0, jitter))
        point = point + n2 * (sag2 * arc + rng.normal(0, jitter))
        points.append(point)
    return points


def _ring(center, radius, rng, gap=0.09, samples=17, axis=(0, 0, 1), wobble=0.02):
    """A drawn circle: not quite round, not quite closed, not necessarily flat.

    The gap is the point of it. A closed ellipse is a machine part; a ring
    whose ends miss each other by a few degrees is a wrist. ``axis`` is what
    the circle is perpendicular to - ``(0, 1, 0)`` for anything drawn on
    something standing up in x-z, facing the camera - and ``wobble`` is how
    much the radius wanders, which is the difference between a circle and a
    blob of wax.

    That wandering is three harmonics of the angle rather than noise at each
    sample, because a blob is lumpy at *low* frequency: a radius drawn
    independently at every point comes out a sawtooth star, which is a fine
    way to draw a splash and no way at all to draw wax. The harmonics are
    functions of the absolute angle, so the shape still closes up where the
    ends of the ring meet.
    """
    center = Vector(center)
    axis, u, v = _frame_of(axis)
    start = rng.uniform(0, tau)
    lumps = [(k, wobble * rng.uniform(0.5, 1.0) / k, rng.uniform(0, tau))
             for k in (1, 2, 3)]
    points = []
    for i in range(samples):
        t = i / (samples - 1)
        angle = start + tau * (1 - gap) * t
        r = radius * (1 + sum(amplitude * np.sin(k * angle + phase)
                              for k, amplitude, phase in lumps)
                      + rng.normal(0, 0.012))
        points.append(center + u * (r * np.cos(angle)) + v * (r * np.sin(angle))
                      + axis * rng.normal(0, 0.01))
    return points


def _ink(points, name, **kwargs):
    """One stroke, as an object that can be grown.

    :class:`BezierDataCurve` takes ``name`` for the curve *data* and pops it
    before :class:`BObject` sees it, so the object itself would end up called
    ``b_object`` - and ``ibpy.get_curve_for_b_object``, which every ``grow``
    goes through, looks the data up under the *object's* name. Handing both
    the same name is the whole job here.
    """
    stroke = BezierDataCurve(data=points, name=name, make_pieces=False, **kwargs)
    stroke.ref_obj.name = name
    stroke.ref_obj.data.name = name
    return stroke


# The back of a C6 envelope, standing on its short edge: portrait, because
# the note that gets written on it is portrait.
#
# The numbers are the four folds of a traditional envelope, which is one sheet
# of paper cut as a cross and folded inwards in a fixed order - sides first,
# then the bottom flap over them, then the pointed top flap down over that and
# sealed. Everything below is where those folds show as creases on the back.
ENVELOPE = dict(width=8.4, height=11.4,
                flap=3.5,  # z where the top flap's point comes to rest
                tip=0.55,  # half width of that point, blunted as flaps are
                fold=-4.8,  # z of the bottom flap's fold
                inset=0.6,  # how far its corners are cut back from the sides
                side=3.1,  # |x| of the two side flaps' inner edges
                seal=0.46)  # radius of the wax


def _envelope_drawing(rng, geo=ENVELOPE):
    """The back of a folded, sealed envelope, as pen strokes in drawing order.

    Nothing here is solid: an envelope is recognised from its creases, and
    the creases are where the four flaps of the cross-shaped sheet lie over
    each other.

    * the **side flaps**, folded in first, show as two long inner edges
      running the height of the back;
    * the **bottom flap**, folded up over them, as a wide fold whose corners
      are cut back - the slant at each end is the giveaway that a flap has
      been folded rather than a line drawn;
    * the **top flap**, folded down last, as two slopes from the top corners
      meeting in a blunt point. Blunt rather than sharp because that is what
      a real flap is: a point that sharp would not survive being posted;
    * the **seal** over that point, a blob of wax with a smaller ring
      impressed in it. It is the only closed shape in the drawing and the
      only warm colour in the envelope, so it is where the eye starts - and
      it is the one stroke that says *sealed* rather than merely *folded*.

    The middle is left alone on purpose. The side seams stand wider than the
    writing, the bottom fold sits below it and the wax above it, so the note
    that gets written afterwards lands in the panel the folds leave empty -
    which is exactly the part of a real envelope anyone writes on.

    Each stroke carries ``order``, which is both the order the hand draws
    them in and, since they are scheduled evenly, when: the outline, then the
    folds in the order the paper was actually folded, then the wax.
    """
    w, h = geo['width'] / 2, geo['height'] / 2
    flap, tip, fold = geo['flap'], geo['tip'], geo['fold']
    inset, side, seal = geo['inset'], geo['side'], geo['seal']

    def at(x, z):
        return Vector((x, 0.0, z))

    strokes = []

    def add(points, part, order):
        strokes.append(dict(points=points, part=part, order=order))

    # --- the sheet itself ------------------------------------------------
    corners = [at(-w, -h), at(w, -h), at(w, h), at(-w, h)]
    for i, (start, end) in enumerate(zip(corners, corners[1:] + corners[:1])):
        add(_pen(start, end, rng, samples=13), 'edge', i)

    # --- the side flaps, folded in first ---------------------------------
    # they run from under the top flap to under the bottom one; where they
    # meet the top flap's slope is where their visible edge starts
    slope = (h - flap) / (w - tip)
    for sign in (-1, 1):
        top = h - slope * (w - side)
        add(_pen(at(sign * side, top), at(sign * side, fold), rng, samples=13),
            'seam', 4 if sign < 0 else 5)

    # --- the bottom flap, folded up over them ----------------------------
    add(_pen(at(-w, -h), at(-w + inset, fold), rng), 'seam', 6)
    add(_pen(at(-w + inset, fold), at(w - inset, fold), rng, samples=13),
        'seam', 7)
    add(_pen(at(w, -h), at(w - inset, fold), rng), 'seam', 8)

    # --- the top flap, folded down last and pointing at the wax ----------
    add(_pen(at(-w, h), at(-tip, flap), rng, samples=11), 'seam', 9)
    add(_pen(at(-tip, flap), at(tip, flap), rng), 'seam', 10)
    add(_pen(at(w, h), at(tip, flap), rng, samples=11), 'seam', 11)

    # --- the wax ---------------------------------------------------------
    # closed and lumpy, unlike everything else here, because that is what
    # distinguishes a seal from a circle
    add(_ring(at(0, flap), seal, rng, gap=0.0, samples=29, axis=(0, 1, 0),
              wobble=0.13), 'seal', 12)
    add(_ring(at(0, flap), 0.5 * seal, rng, gap=0.12, samples=17,
              axis=(0, 1, 0), wobble=0.05), 'seal', 13)

    return strokes


# ===========================================================================
class InterferenceScene(Scene):
    def __init__(self):
        self.t0 = 0
        self.sub_scenes = OrderedDict([
            ('wave_equation_intro', {'duration': 30}),
            ('sine_wave_intro', {'duration': 20}),
            ('wave_equation', {'duration': 30}),
            ('example_whistle', {'duration': 30}),
            ('frequency_list', {'duration': 10}),
            ('table_of_contents', {'duration': 30}),
            ('standing_wave', {'duration': 95}),
            ('bessel_function_dissection', {'duration': 20}),
            ('bessel_node', {'duration': 40}),
            ('towards_two_dimensions', {'duration': 60}),
            ('towards_slit', {'duration': 95}),
            ('towards_double_slit', {'duration': 45}),
            ('towards_grating', {'duration': 25}),
            ('towards_rainbow', {'duration': 35}),
            ('separation_overview', {'duration': 62}),
            ('separation_of_variables', {'duration': 60}),
            ('outlook', {'duration': 50}),
            ('gravi_wave_overlay', {'duration': 20}),
            ('light_wave_overlay', {'duration': 20}),
            ('two_sources', {'duration': 30}),
            ('two_sources_real', {'duration': 24}),
            ('interference_2d', {'duration': 22}),
            ('wave_visualization', {'duration': 24}),
            ('line_array', {'duration': 24}),
            ('line_array_rgb', {'duration': 24}),
            ('airy_disc', {'duration': 32}),
            ('airy_logo', {'duration': 16}),
            ('plane_waves', {'duration': 20}),
            ('gaussian', {'duration': 14}),
            ('samplers', {'duration': 16}),
            ('envelope_computation', {'duration': 35}),
            ('promo_intro', {'duration': 15}),
            ('promo_offer', {'duration': 12}),
            ('watermark', {'duration': 1}),
            ('thumbnail', {'duration': 1}),
            ('lambda_rgb_test', {'duration': 22}),
        ])
        super().__init__(light_energy=1, transparent=False)

    def wave_equation_intro(self):
        r"""
        The title card: the whistle turning up in the corner, and the wave
        equation written across the middle once it has had its ten seconds.

        The whistle is the video's running example, and here it is to be
        looked at rather than read, so it keeps the corner for the whole
        sub-scene and goes on turning underneath the equation. The equation
        is centred on the origin because that is where the camera is aimed;
        the whistle is the thing that had to be placed by hand, and
        :data:`_WHISTLE_LOCATION` below says how.
        """
        t0 = 0
        duration = self.sub_scenes['wave_equation_intro']['duration']

        # background and render settings, as in :meth:`example_whistle` - the
        # model is metal, and with a transparent film the hdri is the only
        # thing it has to reflect. Dropping ``reflections`` here costs the
        # whistle every highlight that tells the eye it is round.
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * Vector([0, 0, 110]),
                                 reflections=True)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=128, motion_blur=False)

        # straight on and aimed at the origin, so that "centred" is the same
        # thing for the equation as for the camera. At 38 mm from 13 units
        # back the frame holds x in [-6.2, 6.2] and z in [-3.5, 3.5] about the
        # origin, which is the rectangle _WHISTLE_LOCATION is a corner of.
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)

        # The model, up in the right-hand corner. The tilt is what makes this
        # read as a turning object rather than a flickering one: the whistle
        # is a long thin thing, and lying flat it would shrink to a blob twice
        # per revolution, as its axis swings through the line of sight. Tilted
        # out of the horizontal it sweeps a cone instead and keeps its length
        # on screen the whole way round.

        _WHISTLE_LOCATION = Vector((3.8, 0, 2.05))
        _WHISTLE_SCALE = 1.4
        _WHISTLE_TILT = 10.5 * pi / 9

        whistle = Whistle(location=_WHISTLE_LOCATION,
                          rotation_euler=[_WHISTLE_TILT, 0, 0])
        t0 = whistle.grow(scale=_WHISTLE_SCALE, begin_time=t0,
                          transition_time=1)

        # and then it turns for the rest of the sub-scene, at a steady two
        # revolutions - slow enough to be read as a model on a turntable
        # rather than as a thing being shown off.
        spin_end = _turntable(whistle, base=[_WHISTLE_TILT, 0, 0],
                              begin_time=t0, transition_time=duration - t0,
                              turns=3)

        # ten seconds of whistle before a word is said, and then the equation
        # the whole video is about
        we = SimpleTexBObject(
            r"\frac{\partial^2 u(x,t)}{\partial t^2}"
            r"=v^2\,\frac{\partial^2 u(x,t)}{\partial x^2}",
            text_size="Large", aligned="center", color="example",
            location=(0, 0, 0))
        we.write(begin_time=10, transition_time=2)

        self.t0 = spin_end

    def sine_wave_intro(self):
        r"""
        show the phase dependence of the sine function
        """
        t0 = 0

        _setup_render()
        _lights(target=(0, 0, 0), strength=0.6)
        # straight on: a graph seen from a corner is a graph with a perspective
        # error in it, and every quantity here is read off an axis
        _camera(location=(0, -12, 0.2), target=(0, 0, 0.2), lens=38)

        create_glow_composition(threshold=0.5, type="BLOOM", size=4)
        wave = GeoFunction(name="SineWave",
                           parameters={"xMin": 0, "xMax": 10, "phi": 0, "amplitude": 1},
                           functions=["pos_x,phi,-,sin,amplitude,*"],
                           colors=["function"],
                           coord=True,
                           # the graph is a physics plot, not a map: x is
                           # stretched to fill the frame and z is given room of
                           # its own, rather than the isotropic default
                           lengths=[9, 3],
                           location=[-4.5, 0, 0],
                           coord_kwargs={"colors": ["text"] * 2,
                                         "radii": [0.025, 0.025],
                                         "tic_label_digits": [0, 0],
                                         "tic_label_shifts": [Vector([0, 0, -0.1]), Vector([-1.2, 0, 0])],
                                         "include_zeros": [True, False],
                                         "axes_labels": {"x": [0.2, 0, 9.3], "u": [0, 0, 3.3]},
                                         "tic_labels": [{"0": 0, r"\pi": pi, r"2\pi": tau, r"3\pi": 3 * pi},
                                                        {"-1": -1, "+1": 1}]},
                           store_values=["result", "amplitude"],
                           emission=0.3)
        t0 = 0.5 + wave.appear(begin_time=t0, transition_time=1.5)

        title = _title(r"$u(x)=\sin(x-\varphi)$",
                       location=(0, 0, 3), size="large")

        phi_slider = BSlider(label=r"\varphi", range=[-pi, pi], side_segments=10, color="text",
                             location=Vector([0, 0, -2]),
                             numberline=True, tic_labels={"0": 0, r"\pi": pi, r"-\pi": -pi},
                             domain=[-pi, pi], tic_label_shift=Vector([0, 0, -0.4]))
        phi_slider.grow(begin_time=t0, transition_time=1)
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        phi_slider.change_value(from_value=0, to_value=pi, begin_time=t0, transition_time=4)
        t0 = 1 + wave.change_parameter("phi", from_value=0, to_value=pi,
                                       begin_time=t0, transition_time=4)

        phi_slider.change_value(from_value=pi, to_value=-pi, begin_time=t0, transition_time=4)
        t0 = 1 + wave.change_parameter("phi", from_value=pi, to_value=-pi,
                                       begin_time=t0, transition_time=4)

        phi_slider.change_value(from_value=-pi, to_value=0, begin_time=t0, transition_time=4)
        t0 = 1 + wave.change_parameter("phi", from_value=-pi, to_value=0,
                                       begin_time=t0, transition_time=4)

        # _linearize(wave.modifier)
        t0 = 0.5 + phi_slider.disappear(begin_time=t0, transition_time=0.5)
        self.t0 = t0

    def wave_equation(self):
        r"""
        Now the transition to the 1+1 dimensional plane wave solution is explained
        """
        t0 = 0

        _setup_render()
        _lights(target=(0, 0, 0), strength=0.6)
        # straight on: a graph seen from a corner is a graph with a perspective
        # error in it, and every quantity here is read off an axis
        location = Vector([0, -12, 0.2])
        _camera(location=location, target=(0, 0, 0.2), lens=38)

        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        # prepare the morph of the title

        title = BDerivation(r"u(x,t)=A\sin\left(\frac{2\pi}{\lambda}x-\frac{2\pi}{T} t\right)",
                            location=(0, 0, 2.5), text_size="large", aligned="center")
        # sine_wave reached this title through '='-aligned steps, so its '='
        # still sits where the centred first line u(x)=sin(x-phi) put it.
        # Hang the title from an unwritten copy of that line so the cut
        # between the two scenes does not jump.
        sine_wave_start = SimpleTexBObject(r"u(x)=\sin(x-\varphi)", name="TitleReference",
                                           location=(0, 0, 2.5), text_size="large", aligned="center")
        title.current.align(sine_wave_start, char_index=title.current.find_letters("=")[0],
                            other_char_index=sine_wave_start.find_letters("=")[0])

        title.write(begin_time=0, transition_time=0)

        # The derivation is now four lines, not two: each second derivative is
        # reached through its own first derivative. The first derivatives are
        # written small - they are the step, not the result - and it is the
        # two full-size second derivatives that the brace collects.
        #
        # All four hang from their equal signs rather than from a common left
        # margin, because the four left-hand sides are of four different
        # widths - du/dx against d^2u/dx^2, and the small size against the
        # full one - so a shared margin would scatter the signs across more
        # than a unit and read as four unrelated lines.
        #
        # z of each line. The two inside a branch sit close together and the
        # branches sit far apart, so the eye groups "first derivative, then
        # second" before it groups x against t. Pushing the branches apart is
        # also what puts the two second derivatives further from each other
        # than the 2.0 they had when they were the only two lines.
        _UX_Z = 1.50
        _UXX_Z = 0.75
        _UT_Z = -0.95
        _UTT_Z = -1.70

        argument = r"\frac{2\pi}{\lambda}x-\frac{2\pi}{T} t"
        lines = [
            (r"\frac{\partial u}{\partial x}="
             r"\frac{2\pi}{\lambda} A \cos\left(%s\right)" % argument,
             "small", _UX_Z),
            (r"\frac{\partial^2 u}{\partial x^2}="
             r"-\frac{4\pi^2}{\lambda^2} A \sin\left(%s\right)" % argument,
             "normal", _UXX_Z),
            (r"\frac{\partial u}{\partial t}="
             r"-\frac{2\pi}{T} A \cos\left(%s\right)" % argument,
             "small", _UT_Z),
            (r"\frac{\partial^2 u}{\partial t^2}="
             r"-\frac{4\pi^2}{T^2} A \sin\left(%s\right)" % argument,
             "normal", _UTT_Z),
        ]

        # Build all four first: aligning is a comparison, so the line that
        # sets the column has to exist before the others can be hung from it.
        rows = [SimpleTexBObject(expression, text_size=size, location=(-5, 0, z))
                for expression, size, z in lines]
        equal_signs = [row.find_letters("=")[0] for row in rows]

        # The x second derivative sets the column - it keeps the left margin
        # of -5 that the two lines used to have, so the block still starts
        # where it did. `align` moves each other line to that same x and then
        # slides its letters until the two addressed glyphs coincide, which is
        # the whole of "line the equal signs up".
        reference, reference_equals = rows[1], equal_signs[1]
        for row, equals in zip(rows, equal_signs):
            if row is not reference:
                row.align(reference, char_index=equals,
                          other_char_index=reference_equals)

        for row, equals in zip(rows, equal_signs):
            # left-hand side in one stroke, then the result - split at the '='
            # rather than at a hand-counted index, so the small lines (whose
            # '=' is the sixth glyph, not the eighth) split in the same place
            t0 = 0.5 + row.write(letter_set=list(range(equals + 1)),
                                 begin_time=t0, transition_time=0.3)
            t0 = 0.5 + row.write(
                letter_set=list(range(equals + 1, len(row.letters))),
                begin_time=t0, transition_time=1)

        # The brace grew with the block it collects: the two second
        # derivatives are further apart than they were, so ``10ex`` no longer
        # reaches from one to the other. It is centred on the pair rather than
        # on the origin, because the four lines are not symmetric about z = 0.
        _BRACE_HEIGHT_EX = 12
        _BRACE_Z = 0.5 * (_UXX_Z + _UTT_Z)
        brace = SimpleTexBObject(
            r"\left.\rule{0em}{%dex}\right\}" % _BRACE_HEIGHT_EX,
            location=(0, 0, _BRACE_Z))
        t0 = 0.5 + brace.write(begin_time=t0, transition_time=0.3)

        # Both of these keep the offsets they had from the brace (+0.5 and
        # -1.0) rather than their old absolute z: the brace is what points at
        # them, and it has moved down with the block it collects.
        wave_eqn = SimpleTexBObject(r"\frac{\partial^2 u}{\partial t^2}=v^2\frac{\partial^2 u}{\partial x^2}",
                                    text_size="large", location=(2.79, 0, _BRACE_Z + 0.5), aligned="center")
        t0 = 0.5 + wave_eqn.write(begin_time=t0, transition_time=1)

        speed = SimpleTexBObject(r"v=\frac{\lambda}{T}", text_size="large", location=(2, 0, _BRACE_Z - 1.0))
        t0 = 0.5 + speed.write(begin_time=t0, transition_time=0.5)

        wave_eqn.rotate(rotation_euler=Vector([pi / 2, 0, tau]), begin_time=t0, transition_time=1)
        t0 = 0.5 + wave_eqn.change_color(new_color="example", begin_time=t0, transition_time=1)

        self.t0 = t0

    def example_whistle(self):
        t0 = 0
        duration = self.sub_scenes['example_whistle']['duration']
        # background and render settings
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * Vector([0, 0, 110]),
                                 reflections=True)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=128, motion_blur=False)

        # camera
        ibpy.set_camera_lens(lens=50)
        ibpy.set_camera_location(location=[0, -5, 5])
        camera_empty = EmptyCube()
        ibpy.set_camera_view_to(camera_empty)

        set_alpha_composition()
        whistle = Whistle(rotation_euler=[pi, 0, -pi / 2])
        t0 = 0.5 + whistle.appear(begin_time=t0, transition_time=0.5)

        # and open it up. The plane starts clear of the model (whose local z
        # reaches 0.245) and travels down to 0.1, taking the wall off the pipe
        # while the camera holds - the inside of a whistle is what the scene
        # is about, and this is the only way to see it.
        #
        # direction="-z" and not the default: the half turn about x above puts
        # the model's local +z pointing *down* in the world, so cutting along
        # local z opens the whistle on the side away from the camera. The
        # value is what it would be either way - the model is symmetric about
        # its own z - it is only the half that goes that swaps over.
        slicer = Slicer(direction="-z", value=0.25)
        whistle.add_mesh_modifier(type='NODES', node_modifier=slicer)
        slicer.slice(to_value=0.1, begin_time=t0, transition_time=2)
        t0 = 0.5 + ibpy.camera_zoom(lens=130, begin_time=t0, transition_time=2)

        l = 14  #(to match with physical pipe)
        n = 0  # number of nodes
        L = l / (1 / 4 + n / 2)  # wavelength (initially)
        T = tau / 4  # period (initially)

        elongation_in = "amplitude,2,pi,*,wavelength,/,x,*,2,pi,*,period,/,time,*,-,sin,*"
        elongation_out = "amplitude,2,pi,*,wavelength,/,x,*,2,pi,*,period,/,time,*,+,pi,+,sin,*"
        elongation = elongation_in + "," + elongation_out + ",+"  # superposition of incoming and reflected wave

        pipe = AcousticModifier(name="AcousticPipe", length=l, pipe_radius=1,
                                amplitude=1, mode=n, period=T,
                                color="acoustic", count=2 ** 15, radius=0.012,
                                zero_color="text",
                                # use standing wave
                                elongation=elongation
                                )

        host = Plane(name="AcousticPipe")
        host.add_mesh_modifier(type='NODES', node_modifier=pipe)
        host.move_to(target_location=[-0.55, 0, 0.1], begin_time=t0, transition_time=0)
        host.rescale(rescale=0.1, begin_time=t0, transition_time=0)

        t0 = 2.5 + host.appear(begin_time=t0, transition_time=1)

        mode_node = ibpy.get_geometry_node_from_modifier(pipe, label="Mode")
        t0 = 0.5 + ibpy.change_default_integer(mode_node, from_value=1, to_value=10, begin_time=t0, transition_time=20)

        self.t0 = t0

    def frequency_list(self):
        r"""
        The four resonant peaks of the whistle, written one after another.

        A transparent-background overlay: nothing is lit and nothing is
        modelled, so this renders over whatever it is cut against - the
        spectrum of the recording, in the script.

        The four are the odd harmonics of the fundamental, 2.1 kHz times 1,
        3, 5 and 7, which is what a tube closed at one end can hold and the
        reason the whistle sounds the way it does. They are written out
        rather than computed from the fundamental so that what goes on screen
        is exactly what the script reads out.
        """
        t0 = 0

        _setup_render()
        # no _lights: the glyph material carries its own emission, and a lamp
        # here would only put a gradient across a flat overlay
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)

        frequencies = ["2.1", "6.3", "10.5", "14.7"]

        # x of the column's right-hand edge, z of the first row, and the drop
        # from one row to the next. The rows hang from the top rather than
        # being centred on the origin, so the list grows downwards as it is
        # written, the way a list being read out should.
        right = 1.2
        top = 1.5
        gap = 1.0
        separation = 2  # seconds between one frequency appearing and the next

        # Right-aligned, so the column hangs from its right-hand edge at
        # ``right`` rather than being built out from a left margin: the
        # numbers have two, three and four digits between them, and it is the
        # units that have to line up, not the first figure.
        for i, frequency in enumerate(frequencies):
            row = SimpleTexBObject(r"%s\,kH\!z" % frequency,
                                   text_size="Large", color="text",
                                   aligned="right",
                                   location=(right, 0, top - i * gap))
            row.write(begin_time=t0, transition_time=1)
            t0 += separation

        # t0 has already been stepped past the last line, so what is left of
        # it after the final write is the hold on the finished list
        self.t0 = t0

    def table_of_contents(self):
        r"""
        The six topics of the presentation, written one after another.

        These are the ``(* ... *)`` cues of the script's own table of
        contents, in its order: each one lands after the sentence of
        narration that introduces it, so they arrive spread out rather than
        all at once. ``separation`` below is the only number here that the
        script does not fix - it is the pacing of the voice-over - so retiming
        the list is a one-line change.

        Left-aligned, which for a contents list is the point: the topics are
        of very different lengths, and it is the start of each line that the
        eye comes back to.
        """
        t0 = 0

        _setup_render()
        # no _lights: the glyph material carries its own emission, and a lamp
        # here would only put a gradient across a flat overlay
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)

        topics = ["Plane Waves", "Standing Waves", "Circular Waves",
                  "Diffraction", "Separation of Variables",
                  "Bessel Functions on the GPU"]

        # x of the column's left-hand edge, z of the first row, and the drop
        # from one row to the next. The left margin is set from the longest
        # topic rather than by eye - "Bessel Functions on the GPU" measures
        # 6.28 units at this size, so hanging the column at -3.2 ends it at
        # +3.08 and leaves the block centred in a frame that reaches +-6.16.
        left = -3.2
        top = 2.25
        gap = 0.9
        separation = 5  # seconds between one topic appearing and the next

        create_glow_composition(threshold=0.75, type="BLOOM", size=4)

        for i, topic in enumerate(topics):
            row = SimpleTexBObject(r"\text{%s}" % topic,
                                   text_size="large", color="text",
                                   aligned="left",
                                   location=(left, 0, top - i * gap))
            row.write(begin_time=t0, transition_time=1)
            t0 += separation

        # t0 has already been stepped past the last line, so what is left of
        # it after the final write is the hold on the finished list
        self.t0 = t0

    def standing_wave(self):
        """
               Now the transition to the 1+1 dimensional plane wave solution is explained
               """
        t0 = 0

        _setup_render()
        _lights(target=(0, 0, 0), strength=0.6)
        # straight on: a graph seen from a corner is a graph with a perspective
        # error in it, and every quantity here is read off an axis
        location = Vector([0, -15, 0.2])
        _camera(location=location, target=(0, 0, 0.2), lens=38)

        create_glow_composition(threshold=0.5, type="BLOOM", size=4)
        # the whistle: a 4.1 cm tube, closed at the wall and open at x = 0.
        # With n nodes in front of the wall, (n/2+1/4) wavelengths fit in
        wall_pos = 4.1
        length = 5
        speed_of_sound = 340  # m/s

        def wavelength(n):
            return wall_pos / (n / 2 + 1 / 4)

        def period(n):
            return 5 * wavelength(n) / 16.1

        def world_x(x):
            # graph coordinate -> world coordinate (x in [0, length] spans 9 units)
            return (x - length / 2) * 9 / length

        def node_x(n, i):
            # the i-th of n nodes, counted from the open end; i = n is the wall
            return world_x(wall_pos - wavelength(n) / 2 * (n - i))

        def cm(value):
            return r"%.3g\,\text{cm}" % value

        n_max = 4
        lambda_start = wavelength(n_max)
        in_wave = GeoFunction(name="SineWave",
                              parameters={"xMin": 0, "xMax": length, "A=amplitude": 1, "cut": 0,
                                          "lambda=wavelength": lambda_start, "T=period": 1, "t=time": "time",
                                          "start": 0},
                              functions=[
                                  f"tau,lambda,/,pos_x,*,tau,T,/,t,*,start,*,-,sin,A,*,1,pos_x,{wall_pos},>,cut,*,-,*"],
                              colors=["function"],
                              coord=True,
                              # the graph is a physics plot, not a map: x is
                              # stretched to fill the frame and z is given room of
                              # its own, rather than the isotropic default
                              lengths=[9, 3],
                              location=[-4.5, 0, 2],
                              coord_kwargs={"colors": ["text"] * 2,
                                            "radii": [0.025, 0.025],
                                            "tic_label_digits": [0, 0],
                                            "tic_label_shifts": [Vector([0, 0, -0.1]), Vector([-1.2, 0, 0])],
                                            "include_zeros": [True, False],
                                            "axes_labels": {"x": [0.2, 0, 9.3], "u": [0, 0, 3.3]},
                                            "tic_labels": [{"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
                                                           {"-1": -1, "+1": 1}]},
                              store_values=["result", "amplitude"],  # needed for the intensity shader
                              emission=0.3)

        t0 = 0.5 + in_wave.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + in_wave.change_parameter("start", from_value=0, to_value=1, begin_time=t0, transition_time=1)

        wall = Plane(location=Vector(), normal=Vector([-1, 0, 0]), solid=0.2, side_segments=10, color="example")
        wall.move_to(target_location=Vector([world_x(wall_pos), 0, 2]), begin_time=t0, transition_time=0)
        wall.rescale(rescale=[1.75, 1, 1], begin_time=t0, transition_time=0)

        in_wave.change_parameter("cut", from_value=0, to_value=1, begin_time=t0, transition_time=1)
        t0 = 0.5 + wall.appear(begin_time=t0, transition_time=1)

        out_wave = GeoFunction(name="SineWave",
                               parameters={"xMin": 0, "xMax": length, "A=amplitude": 1, "cut": 1,
                                           "lambda=wavelength": lambda_start, "T=period": 1, "t=time": "time"},
                               functions=[
                                   f"tau,lambda,/,pos_x,*,tau,T,/,t,*,+,pi,+,sin,A,*,1,pos_x,{wall_pos},>,cut,*,-,*"],
                               colors=["function"],
                               coord=True,
                               # the graph is a physics plot, not a map: x is
                               # stretched to fill the frame and z is given room of
                               # its own, rather than the isotropic default
                               lengths=[9, 3],
                               location=[-4.5, 0, -2],
                               coord_kwargs={"colors": ["text"] * 2,
                                             "radii": [0.025, 0.025],
                                             "tic_label_digits": [0, 0],
                                             "tic_label_shifts": [Vector([0, 0, -0.1]), Vector([-1.2, 0, 0])],
                                             "include_zeros": [True, False],
                                             "axes_labels": {"x": [0.2, 0, 9.3], "u": [0, 0, 3.3]},
                                             "tic_labels": [{"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
                                                            {"-1": -1, "+1": 1}]},
                               store_values=["result", "amplitude"],  # needed for the intensity shader
                               emission=0.3)

        wall2 = Plane(location=Vector(), normal=Vector([-1, 0, 0]), solid=0.2, side_segments=10, color="example")
        wall2.move_to(target_location=Vector([world_x(wall_pos), 0, -2]), begin_time=t0, transition_time=0)
        wall2.rescale(rescale=[1.75, 1, 1], begin_time=t0, transition_time=0)
        wall2.appear(begin_time=t0, transition_time=1)
        t0 = 5 + out_wave.appear(begin_time=t0, transition_time=1)

        # move the two systems together

        in_wave.move(direction=Vector([0, 0, -2]), begin_time=t0, transition_time=1)
        wall.move(direction=Vector([0, 0, -0.25]), begin_time=t0, transition_time=1)
        wall2.move(direction=Vector([0, 0, 0.25]), begin_time=t0, transition_time=1)
        t0 = out_wave.move(direction=Vector([0, 0, 2]), begin_time=t0, transition_time=1)
        in_wave.disappear(begin_time=t0, transition_time=0)
        out_wave.disappear(begin_time=t0, transition_time=0)

        standing_wave = GeoFunction(name="SineWave",
                                    parameters={"xMin": 0, "xMax": length, "A=amplitude": 1, "w=wall": wall_pos,
                                                "lambda=wavelength": lambda_start, "T=period": 1, "t=time": "time",
                                                "cut": 1},
                                    functions=[
                                        f"tau,lambda,/,pos_x,*,tau,T,/,t,*,+,pi,+,sin,A,*,tau,lambda,/,pos_x,*,tau,T,/,t,*,-,sin,A,*,+,1,pos_x,{wall_pos},>,cut,*,-,*"],
                                    colors=["function"],
                                    coord=True,
                                    # the graph is a physics plot, not a map: x is
                                    # stretched to fill the frame and z is given room of
                                    # its own, rather than the isotropic default
                                    lengths=[9, 6],
                                    location=[-4.5, 0, 0],
                                    coord_kwargs={"colors": ["text"] * 2,
                                                  "radii": [0.025, 0.025],
                                                  "tic_label_digits": [0, 0],
                                                  "tic_label_shifts": [Vector([0, 0, -0.1]), Vector([-1.2, 0, 0])],
                                                  "include_zeros": [True, False],
                                                  "axes_labels": {"x": [0.2, 0, 9.3], "u": [0, 0, 6.3]},
                                                  "tic_labels": [{"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
                                                                 {"-2": -2, "-1": -1, "+1": 1, "+2": 2}]},
                                    store_values=["result", "amplitude"],  # needed for the intensity shader
                                    emission=0.3)

        t0 = standing_wave.appear(begin_time=t0, transition_time=0)

        dirichlet = SimpleTexBObject(r"u(4.1,t)=0", location=[world_x(wall_pos) + 0.65, 0, 0.5])
        dirichlet.write(begin_time=t0, transition_time=0.5)

        neumann = SimpleTexBObject(r"\left.{\partial u(x,t)\over \partial x}\right|_{x=0}=0", location=[-7, 0, 0.5])
        t0 = 5 + neumann.write(begin_time=t0, transition_time=0.5)

        nodes = [Sphere(r=0.1, color="custom1", location=[node_x(n_max, i), 0, 0]) for i in range(n_max)]
        colors = ["text", "custom1", "text"]
        n_counter = BDerivation("n=%d" % n_max, aligned="center", location=[0.32, 0, 3], color="custom1")
        spectrum = SimpleTexBObject(r"\left(\frac{n}{2}+\frac{1}{4}\right)\lambda = 4.1\,\text{cm}", aligned="center",
                                    location=[0, 0, 3.5], color=colors)
        for i, node in enumerate(nodes):
            node.grow(begin_time=t0 + i * 0.1, transition_time=0.2)
        n_counter.write(begin_time=t0 + 5, transition_time=0.2)
        t0 = 1 + spectrum.write(begin_time=t0, transition_time=0.5)

        def wavelength_list(n_min):
            return r"\lambda\in\left\{" + ",".join(cm(wavelength(k)) for k in range(n_max, n_min - 1, -1)) + r"\right\}"

        waves = BDerivation(wavelength_list(n_max), aligned="left", location=[-4.15, 0, -3.5])
        t0 = 5 + waves.write(begin_time=t0, transition_time=0.5)

        # the mode runs down: each lost node takes half a wavelength with it.
        # The node next to the wall slides into it and vanishes
        for n in range(n_max - 1, -1, -1):
            for i in range(n + 1):
                nodes[i].move_to(target_location=[node_x(n, i), 0, 0], begin_time=t0, transition_time=0.5)
            nodes[-1].shrink(begin_time=t0, transition_time=0.5)
            nodes.remove(nodes[-1])
            n_counter.step(r"n=%d" % n, mode="replace", map={r"n=": r"n="}, begin_time=t0, transition_time=0.5)
            standing_wave.change_parameter("wavelength", from_value=wavelength(n + 1), to_value=wavelength(n),
                                           begin_time=t0, transition_time=0.5)
            standing_wave.change_parameter("period", from_value=period(n + 1), to_value=period(n), begin_time=t0,
                                           transition_time=0.5)
            kept = {r"\lambda\in": r"\lambda\in"}
            kept.update({cm(wavelength(k)): cm(wavelength(k)) for k in range(n_max, n, -1)})
            t0 = 5 + waves.step(wavelength_list(n), mode="replace", map=kept, begin_time=t0, transition_time=0.5)

        # --- our whistle once more: the computations ------------------------
        # the wavelength list and the boundary condition at the wall make room
        # for the computation (bottom) and the collected spectrum (right)
        waves.disappear(begin_time=t0, transition_time=0.5)
        neumann.disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + dirichlet.disappear(begin_time=t0, transition_time=0.5)

        def khz(n):
            # frequency of the mode with n nodes, (2n+1) times the fundamental
            return r"%.1f\,\text{kHz}" % (speed_of_sound / (wavelength(n) / 100) / 1000)  # cm -> m

        # the fundamental: a quarter of a wavelength fits into the tube
        calc = BDerivation(r"\tfrac{1}{4}\lambda_0\approx4.1\,\text{cm}", aligned="center", location=[0, 0, -3.4])
        t0 = 2 + calc.write(begin_time=t0, transition_time=0.5)
        # centred rather than '='-aligned, or the lines drift into the wall
        t0 = 3 + calc.step(r"\lambda_0\approx4\cdot 4.1\,\text{cm}\approx" + cm(wavelength(0)), mode="replace",
                           align=False,
                           map={r"\lambda_0": r"\lambda_0", r"4.1\,\text{cm}": r"4.1\,\text{cm}"},
                           begin_time=t0, transition_time=0.5)
        t0 = 1 + calc.step(r"f_0=\tfrac{c}{\lambda_0}\approx\tfrac{%d\,\text{m/s}}{%.3f\,\text{m}}\approx%s"
                           % (speed_of_sound, wavelength(0) / 100, khz(0)), mode="replace", align=False,
                           begin_time=t0, transition_time=0.5)
        spectrum_lines = BDerivation(r"f_0\approx" + khz(0), location=[world_x(wall_pos) + 0.75, 0, 2.6],
                                     line_spacing=Vector((0, 0, -0.7)))
        t0 = 3 + spectrum_lines.write(begin_time=t0, transition_time=0.5)

        # the mode builds up again: a new node emerges from the wall with every
        # extra half wavelength, and the frequency grows by twice the fundamental
        for n in range(1, n_max):
            odd = 2 * n + 1
            node = Sphere(r=0.1, color="custom1", location=[node_x(n - 1, n - 1), 0, 0])
            node.grow(begin_time=t0, transition_time=0.3)
            nodes.append(node)
            for i in range(n):
                nodes[i].move_to(target_location=[node_x(n, i), 0, 0], begin_time=t0, transition_time=0.5)
            n_counter.step(r"n=%d" % n, mode="replace", map={r"n=": r"n="}, begin_time=t0, transition_time=0.5)
            standing_wave.change_parameter("wavelength", from_value=wavelength(n - 1), to_value=wavelength(n),
                                           begin_time=t0, transition_time=0.5)
            standing_wave.change_parameter("period", from_value=period(n - 1), to_value=period(n), begin_time=t0,
                                           transition_time=0.5)
            pause = 3 if n == 1 else 1.5  # the first one is explained, the others continue the pattern
            t0 = pause + calc.step(r"\tfrac{%d}{4}\lambda_%d\approx4.1\,\text{cm}" % (odd, n),
                                   mode="replace", align=False, begin_time=t0, transition_time=0.5)
            t0 = pause + calc.step(r"\lambda_%d=\tfrac{\lambda_0}{%d}\approx%s" % (n, odd, cm(wavelength(n))),
                                   mode="replace", align=False, begin_time=t0, transition_time=0.5)
            t0 = 0.5 + calc.step(r"f_%d=%d\,f_0\approx%s" % (n, odd, khz(n)), mode="replace", align=False,
                                 begin_time=t0, transition_time=0.5)
            t0 = pause + spectrum_lines.step(r"f_%d=%d\,f_0\approx%s" % (n, odd, khz(n)), begin_time=t0,
                                             transition_time=0.5)

        self.t0 = t0

    def bessel_function_dissection(self):
        r"""The circular wave taken apart into its two Bessel functions.

        A coordinate system is drawn, and two grid lines run out along its x
        axis. Then they are bent, one after the other, into :math:`J_0(x)` and
        :math:`Y_0(x)` - the two halves of the circular wave. Both lines are one
        :class:`~geometry_nodes.modifier_video_interferences.BesselVisualizer`,
        hung at the coordinate system's location with the same domains and
        lengths, so the curves lie on its axes.
        """
        t0 = 0
        duration = self.sub_scenes['bessel_function_dissection']['duration']

        _setup_render()
        create_glow_composition(threshold=0.5, type="BLOOM", size=4, transparent=True)
        # straight on: at 45 mm from 20 units back the frame holds
        # x in [-8, 8] and z in [-4.5, 4.5]
        _camera(location=(10, -20, 0), target=(10, 0, 0), lens=45)

        # value (0, 0) of the coordinate system is its location
        x_domain, y_domain = [0, 20], [-1, 1]
        width, height = 20,6
        origin = Vector([0, 0, 0])

        coords = CoordinateSystem2(
            location=origin, lengths=[width, height], colors=['text', 'text'],
            domains=[x_domain, y_domain], tic_label_digits=[0, 0],
            tic_labels=[{str(x): x for x in range(0, 21, 5)},
                        {"-1": -1, "0": 0, "1": 1}],
            axes_labels={r"x": [-0.5, 0, width + 0.4],
                         # measured from the bottom of the axis, not from y = 0
                         r"y": [-0.5, 0, height + 0.4]},
            aligned='center', tic_label_shifts=[Vector(), [-0.4, 0, 0]])
        t0 = 0.5 + coords.appear(begin_time=t0, transition_time=2)

        bessel = BesselVisualizer(x_domain=x_domain, y_domain=y_domain,
                                  width=width, height=height, count=1000,
                                  thickness=0.04, j0_color="orange",
                                  y0_color="cyan", emission=0.5)
        # a Plane bakes its location into the mesh by default, and the
        # modifier throws that mesh away - so the location goes on the object
        lines = Plane(name="BesselLines", location=origin, apply_location=False)
        lines.add_mesh_modifier(type='NODES', node_modifier=bessel)
        lines.appear(begin_time=0, transition_time=0)

        def dial(name, begin_time, transition_time):
            ibpy.change_default_value(
                ibpy.get_geometry_node_from_modifier(bessel, name),
                from_value=0, to_value=1,
                begin_time=begin_time, transition_time=transition_time)
            return begin_time + transition_time

        # J0: a grid line along the axis, then bent into the function
        t0 = 0.5 + dial("J0Reveal", t0, 1.5)
        t0 = dial("J0Amount", t0, 3)
        j0_label = SimpleTexBObject(r"J_0(x)", color="orange", text_size="large",
                                    location=origin + Vector([1.3, 0, 2.6]),
                                    aligned="left")
        t0 = 1 + j0_label.write(begin_time=t0, transition_time=0.5)

        # Y0 the same way; its dive towards x = 0 is cut at the bottom of the plot
        t0 = 0.5 + dial("Y0Reveal", t0, 1.5)
        t0 = dial("Y0Amount", t0, 3)
        y0_label = SimpleTexBObject(r"Y_0(x)", color="cyan", text_size="large",
                                    location=origin + Vector([2.6, 0, 2.0]),
                                    aligned="left")
        t0 = 1 + y0_label.write(begin_time=t0, transition_time=0.5)

        print("bessel_function_dissection: animation ends at %.2f of %d" % (t0, duration))
        self.t0 = duration

    def bessel_node(self):
        r"""What is inside the ``BesselJ0`` node.

        The screenshot of the node tree that uses it (``bessel_node0.png``)
        fills the upper left; a
        :class:`~objects.derived_objects.magnifying_glass.MagnifyingGlass`
        flies in, magnifying the tree as it crosses it, and stops over the
        ``BesselJ0`` group node, where its lens is switched over to what that
        group holds - the screenshot of the group's inside
        (``bessel_node1.png``), its full height across the lens, scrolled past
        once from the group input to the group output.

        Meanwhile the approximation it implements, Abramowitz & Stegun 9.4.1
        and 9.4.3, is written below the tree - the two branches behind one
        brace - and its coefficients run along the bottom of the frame on a
        :class:`~objects.derived_objects.ticker.Ticker`, while the RPN that builds the
        group is typed out on a :class:`~objects.display.CodeDisplay` on the
        right. The code comes from ``code_snippets/bessel_j0_rpn.py``, a
        comment-free copy of :func:`~geometry_nodes.nodes.bessel_j0_y0_rpn`
        reduced to J0 (:class:`~objects.codeparser.CodeParser` cannot read
        comments) - it yields the same aux formulas, letter for letter.
        """
        t0 = 0
        duration = self.sub_scenes['bessel_node']['duration']
        _setup_render()
        _lights(target=(0, 0, 0), strength=0.4)
        # 32 mm from 13 back holds x in [-7.3, 7.3], z in [-4.1, 4.1]
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=32)

        # ---- the node tree that uses the group --------------------------------
        # Plane turns its location along with the mesh, so locations are
        # given in its own frame: (x, z, -y) for a world point (x, y, z)
        tree_width = 7.2
        tree_height = tree_width * 542 / 1518
        tree_center = Vector([-3.4, 0, 2.4])
        tree = Plane(u=[-tree_width / 2, tree_width / 2],
                     v=[-tree_height / 2, tree_height / 2],
                     rotation_euler=[pi / 2, 0, 0],
                     location=[tree_center.x, tree_center.z, -tree_center.y],
                     color="image", src="bessel_node0.png", emission=0.6,
                     resolution=1, name="BesselNodeTree")
        t0 = 0.5 + tree.appear(begin_time=t0, transition_time=1)

        # the BesselJ0 node sits at pixel (672, 306) of the 1518 x 542 screenshot
        node = tree_center + Vector([(672 / 1518 - 0.5) * tree_width, -0.5,
                                     (0.5 - 306 / 542) * tree_height])

        # ---- the magnifying glass ---------------------------------------------
        # its lens is a window onto the inside of the group, a little more
        # than the full height of the screenshot across, run once from the
        # group input on the left to the group output on the right
        window = 1.15
        edge = window * 1159 / 5067 / 2  # half the window's width, in image widths
        glass = MagnifyingGlass("bessel_node1.png", radius=1.6, window=window,
                                center=(edge, 0.5), handle_angle=-pi / 6,
                                handle_length=1.9, background=tree, magnification=2,
                                location=node + Vector([9, 0, 0]))
        glass.appear(begin_time=t0, transition_time=0.3)
        t0 = glass.move_to(target_location=node, begin_time=t0, transition_time=2.5)
        # magnifying the tree on the way in, switched to the group's inside over the node
        t0 = 0.2 + glass.switch_on(begin_time=t0, transition_time=0.5)
        glass.look_at(1 - edge, 0.5, begin_time=t0, transition_time=14)

        # ---- the approximation ------------------------------------------------
        formulas = [(r"\text{Abramowitz \& Stegun 9.4.1, 9.4.3}", "example", -7, 0.2),
                    (r"J_0(x)\approx\begin{cases}\displaystyle\sum_{k=0}^{6}a_k\left(\frac{x}{3}\right)^{2k},"
                     r"& 0\le x\le 3\\[12pt]\dfrac{f_0}{\sqrt{x}}\cos\theta_0, & x\ge 3\end{cases}",
                     "text", -7, -1.0),
                    (r"f_0=\textstyle\sum_{k=0}^{6}b_k\left(\tfrac{3}{x}\right)^{k}", "text", -7, -2.45),
                    (r"\theta_0=x-\tfrac{\pi}{4}+\textstyle\sum_{k=1}^{6}c_k\left(\tfrac{3}{x}\right)^{k}",
                     "text", -7, -3.05),
                    (r"|\varepsilon|<5\cdot 10^{-8}", "example", -2.2, -2.45)]
        for tex, color, x, z in formulas:
            line = SimpleTexBObject(tex, text_size="normal", color=color, aligned="left",
                                    location=(x, 0, z))
            t0 = 0.4 + line.write(begin_time=t0, transition_time=1)

        # ---- the coefficients, as a news ticker along the bottom --------------
        # the same numbers the code holds, taken from where the node takes them
        families = [("a", _J0_SMALL, 0), ("b", _BESSEL_F0, 0), ("c", _BESSEL_TH, 1)]
        entries = [r"%s_{%d}=%s" % (name, first + k, np.format_float_positional(c, trim='0'))
                   for name, values, first in families for k, c in enumerate(values)]
        ticker = Ticker(entries, width=15, height=0.4)
        ticker.appear(begin_time=t0, transition_time=0.5)
        ticker.run(begin_time=t0, transition_time=duration - t0)
        ticker.move_to(target_location=Vector([0, -0.05, -3.8]), begin_time=0, transition_time=0)

        # ---- the RPN that builds the group ------------------------------------
        code = CodeParser(os.path.join("code_snippets", "bessel_j0_rpn.py"))
        display = CodeDisplay(code, class_index=0, flat=True, location=Vector([4.1, 0.3, 0.45]),
                              scales=[3.1, 3.6])
        t0 = display.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + code.write(display, class_index=0, begin_time=t0, transition_time=12,
                              indent=0.25)

        print("bessel_node: animation ends at %.2f of %d" % (t0, duration))
        self.t0 = duration

    def towards_two_dimensions(self):
        t0 = 0
        _setup_render()
        # straight on: a graph seen from a corner is a graph with a perspective
        # error in it, and every quantity here is read off an axis
        location = Vector([0, -12, 0.2])
        _camera(location=location, target=(0, 0, 0.2), lens=38)
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        # 1+1 dimensional wave equation
        wave_eqn = BDerivation(r"\frac{\partial^2 u}{\partial t^2}=v^2\frac{\partial^2 u}{\partial x^2}",
                               text_size="large", location=(0, 0, 2.5), aligned="center")
        t0 = 0.5 + wave_eqn.write(begin_time=t0, transition_time=1)

        t0 = 0.5 + wave_eqn.step(r"\frac{\partial^2 u}{\partial t^2}=v^2\left(\frac{\partial^2 u}{\partial "
                                 r"x^2}+\frac{\partial^2 u}{\partial y^2}\right)",
                                 mode="replace",
                                 map={r"\frac{\partial^2 u}{\partial t^2}=v^2": r"\frac{\partial^2 u}{\partial "
                                                                                r"t^2}=v^2",
                                      r"\frac{\partial^2 u}{\partial x^2}": r"\frac{\partial^2 u}{\partial x^2}"},
                                 align=False,
                                 begin_time=t0, transition_time=1)

        # ---- the change of coordinates, twice: before and during ----------
        # The same modifier hangs on both panels. The left one is never
        # touched, so it stays the cartesian grid the eye can hold the right
        # one against; the right one's `Transition` dial is what the whole
        # shot is about. Two panels rather than one shape that changes,
        # because the point of polar coordinates is that they describe the
        # *same* plane - and a before that has gone away cannot say that.
        # the height; `square_cells` picks the width from the two line
        # counts, so ten circles and twelve rays give a panel 3.0 by 3.6 with
        # square cells - a grid of oblong ones reads as a stretched picture
        # rather than as a coordinate system
        grid_kwargs = dict(size=3.0, horizontals=11, verticals=13,
                           thickness=0.018, emission=0.3)
        cartesian = PolarGridModifier(name="CartesianGrid", **grid_kwargs)
        polar = PolarGridModifier(name="PolarGrid", **grid_kwargs)
        left = _panel(cartesian, name="CartesianPanel",
                      location=(-2.8, 0, -1.15), rotation_euler=[pi / 2, 0, 0])
        right = _panel(polar, name="PolarPanel", location=(2.8, 0, -1.15), rotation_euler=[pi / 2, 0, 0])
        left.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + right.appear(begin_time=t0, transition_time=1)

        # The formulas over the panel that performs them, one glyph one
        # colour: `x` and `y` in the colours of the lines they label - a
        # vertical line is x = const, a horizontal one is y = const - and
        # `r` and `varphi` in the colours those lines are about to become.
        # The list is one entry per rendered glyph, left to right, so \cos is
        # three of them and the parentheses are one each.
        #                x    =     r      c     o     s     (     phi     )
        x_colors = ["custom1", "text", "joker", "text", "text", "text",
                    "text", "example", "text"]
        #                y    =     r      s     i     n     (     phi     )
        y_colors = ["drawing", "text", "joker", "text", "text", "text",
                    "text", "example", "text"]
        x_map = SimpleTexBObject(r"x=r\cos(\varphi)", color=x_colors,
                                 location=[2.8, 0, 1.55], aligned="center")
        y_map = SimpleTexBObject(r"y=r\sin(\varphi)", color=y_colors,
                                 location=[2.8, 0, 0.95], aligned="center")
        t0 = 0.5 + x_map.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + y_map.write(begin_time=t0, transition_time=1)

        # and the bend itself. The panel is read as the (varphi, r) rectangle
        # - one full turn wide, R tall - so the horizontals close into circles
        # and the verticals fan out into rays, changing colour as they go.
        transition = ibpy.get_geometry_node_from_modifier(polar, "Transition")
        t0 = 1 + ibpy.change_default_value(transition, from_value=0, to_value=1,
                                           begin_time=t0, transition_time=6)

        left.disappear(begin_time=t0, transition_time=1)
        t0 = 0.5 + right.disappear(begin_time=t0, transition_time=1)

        # 2+1-dimensional wave equation
        wave_eqn2 = BDerivation(
            r"\frac{\partial^2 u}{\partial t^2}=v^2\left(\frac{\partial^2 u}{\partial r^2}+\frac{1}{r}\frac{\partial "
            r"u}{\partial r}+\frac{1}{r^2}\frac{\partial^2 u}{\partial \varphi^2}\right)",
            text_size="large", location=(0, 0, -0.5), aligned="center")

        t0 = 0.5 + wave_eqn2.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + wave_eqn2.step(
            r"\frac{\partial^2 u}{\partial t^2}=v^2\left(\frac{\partial^2 u}{\partial r^2}+\frac{1}{r}\frac{\partial "
            r"u}{\partial r}\right)",
            mode="replace", align=False, begin_time=t0, transition_time=1,
            map={r"\frac{\partial^2 u}{\partial t^2}=": r"\frac{\partial^2 u}{\partial t^2}=",
                 r"\frac{\partial^2 u}{\partial r^2}+": r"\frac{\partial^2 u}{\partial r^2}+",
                 r"\frac{1}{r}\frac{\partial u}{\partial r}": r"\frac{1}{r}\frac{\partial u}{\partial r}", })

        solution = SimpleTexBObject(
            r"u(r,t) = A\left( {\rm J}_0(\frac{2\pi}{\lambda}r)\cos(\frac{2\pi}{T} t)+{\rm Y}_0(\frac{2\pi}{\lambda} "
            r"r)\sin(\frac{2\pi}{T} t)\right)",
            aligned="center", text_size="large", location=(0, 0, -2))

        t0 = 0.5 + solution.write(begin_time=t0, transition_time=2)

        wave_eqn.disappear(begin_time=t0, transition_time=0.5)
        x_map.disappear(begin_time=t0, transition_time=0.5)
        y_map.disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + wave_eqn2.disappear(begin_time=t0, transition_time=0.5)

        wave = WaveVisualizationModifier(name="WaveVisualization", size=16.0,
                                         resolution=301, sources=((0, 0),),
                                         wavelength=1.5, frequency=1.0,
                                         amplitude=0.6, material=get_texture("function", alpha_intensity=0.9),
                                         source_radius=2 / 10,
                                         emission_strength=0.8)
        surface = Plane(name="WaveSurface", u=[-1, 1], v=[-1, 1], resolution=1)
        surface.add_mesh_modifier(type='NODES', node_modifier=wave)
        surface.move(direction=[0, 0, 1.65], begin_time=t0, transition_time=0)
        surface.rotate(rotation_euler=[pi / 6, 0, 0], begin_time=t0, transition_time=0)
        t0 = 10 + surface.appear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def towards_slit(self):
        """
        a plane wave on the left-hand side of the wall
        a slit
        a circular wave on the right-hand side of the wall

        Continued by :meth:`towards_double_slit`, which rebuilds the last frame
        of this one and runs the waves on from this scene's clock (see
        :meth:`_slit_time_shift`). Anything changed about the final state here
        has to be changed there as well.
        """

        t0 = 0
        _setup_render()
        # straight on: a graph seen from a corner is a graph with a perspective
        # error in it, and every quantity here is read off an axis
        location = Vector([0, -12, 0.2])
        _camera(location=location, target=(0, 0, 0.2), lens=38)
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        wave = WaveVisualizationModifier(name="WaveVisualization", size=16,
                                         resolution=301,
                                         wavelength=1.5, frequency=1.0,
                                         amplitude=0.6, material=get_texture("function", alpha_intensity=0.9),
                                         source_radius=2 / 10,
                                         emission_strength=0.8)
        surface = Plane(name="WaveSurface", u=[-1, 1], v=[-1, 1], resolution=1)
        surface.add_mesh_modifier(type='NODES', node_modifier=wave)
        surface.move(direction=[-8, 0, 0], begin_time=t0, transition_time=0)
        width_node = ibpy.get_geometry_node_from_modifier(wave, label="Width")
        ibpy.change_default_value(width_node, from_value=0, to_value=0.1, begin_time=t0, transition_time=0)

        # make wave strech full screen initially

        ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(wave, label="Size"), from_value=16, to_value=43,
                                  begin_time=t0, transition_time=0)
        t0 = 11.5 + surface.appear(begin_time=t0, transition_time=1)
        t0 = 0.5 + ibpy.camera_move(shift=[-9, 0, 7], begin_time=t0, transition_time=10)

        camera_location = ibpy.get_camera().location
        plane_wave_txt = SimpleTexBObject(r"\text{Plane Wave}", aligned="center", text_size="large",
                                          location=[-3, 0, 3.5])
        camera_angle = ibpy.camera_alignment_euler(plane_wave_txt, camera_location).x
        plane_wave_txt.rotate(rotation_euler=[76 / 180 * np.pi, 0, -pi / 6], begin_time=t0, transition_time=0)
        plane_wave_txt.write(begin_time=t0, transition_time=0.5)

        t0 = 0.5 + ibpy.change_default_value(width_node, from_value=0.1, to_value=6, begin_time=t0, transition_time=4.5)
        t0 = surface.move(direction=[0, 0, 1], begin_time=t0, transition_time=2)
        t0 = 1 + surface.move(direction=[0, 0, -2], begin_time=t0, transition_time=2)

        wall_right = Plane(name="WallRight", u=[-2, 2], v=[0, 3], resolution=1, solid=0.025, color="important")
        wall_left = Plane(name="WallLeft", u=[-2, 2], v=[-3, 0], resolution=1, solid=0.025, color="important")
        walls = [wall_left, wall_right]
        [wall.appear(begin_time=t0, transition_time=1) for wall in walls]
        t0 = 0.5 + ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(wave, label="Size"), from_value=43,
                                             to_value=16,
                                             begin_time=t0, transition_time=1)

        [wall.rotate(rotation_euler=[0, pi / 2, 0], begin_time=0, transition_time=0) for wall in walls]
        [wall.rescale(rescale=[0.5, 1, 1], begin_time=t0, transition_time=1) for wall in walls]
        t0 = 8 + surface.move(direction=[0, 0, 1], begin_time=t0, transition_time=2)

        # open slit and propagate circular wave

        wave2 = WaveVisualizationModifier(name="WaveVisualization", size=8.0,
                                          resolution=301, sources=((-4, 0),),
                                          wavelength=1.5, frequency=1.0,
                                          amplitude=0.31, material=get_texture("function", alpha_intensity=0.9),
                                          source_radius=2 / 10,
                                          emission_strength=0.8)
        surface2 = Plane(name="WaveSurface", u=[-1, 1], v=[-1, 1], resolution=1)
        surface2.add_mesh_modifier(type='NODES', node_modifier=wave2)
        surface2.move(direction=[4, 0, 0], begin_time=00, transition_time=0)

        t0 = np.ceil(t0)
        surface2.appear(begin_time=t0, transition_time=1)
        wall_left.move(direction=[0, -0.1, 0], begin_time=t0, transition_time=1)
        t0 = 0.5 + wall_right.move(direction=[0, 0.1, 0], begin_time=t0, transition_time=1)
        plane_wave_txt.rotate(rotation_euler=[camera_angle, 0, 0], begin_time=t0, transition_time=0)
        t0 = 0.5 + ibpy.camera_move(shift=[9, 0, 0], begin_time=t0, transition_time=2)

        t0 += 15.5
        camera_location = ibpy.get_camera().location
        circular_wave_txt = SimpleTexBObject(r"\text{Circular Wave}", aligned="center", text_size="large",
                                             location=[3, 0, 3.5])
        circular_wave_txt.rotate(rotation_euler=[camera_angle, 0, 0], begin_time=t0, transition_time=0)
        t0 = 0.5 + circular_wave_txt.write(begin_time=t0, transition_time=0.5)

        t0 += 4
        wave_eqn = SimpleTexBObject(r"\frac{\partial^2 u}{\partial t^2}=c^2\frac{\partial^2 u}{\partial x^2}",
                                    location=(-3, 0, 2.5), aligned="center")
        wave_eqn.rotate(rotation_euler=[camera_angle, 0, 0], begin_time=t0, transition_time=0)
        t0 = 0.5 + wave_eqn.write(begin_time=t0, transition_time=0.5)

        t0 += 4
        wave_eqn2 = SimpleTexBObject(r"\frac{\partial^2 u}{\partial t^2}=c^2\left(\frac{\partial^2 u}{\partial "
                                     r"r^2}+\frac{1}{r}\frac{\partial u}{\partial r}\right)",
                                     aligned="center", location=[3, 0, 2.5])
        wave_eqn2.rotate(rotation_euler=[camera_angle, 0, 0], begin_time=t0, transition_time=0)
        t0 = 0.5 + wave_eqn2.write(begin_time=t0, transition_time=0.5)

        grid_kwargs = dict(size=3.0, horizontals=11, verticals=13,
                           thickness=0.018, emission=0.3)
        cartesian = PolarGridModifier(name="CartesianGrid", **grid_kwargs)
        polar = PolarGridModifier(name="PolarGrid", **grid_kwargs, half=True)
        left = _panel(cartesian, name="CartesianPanel",
                      location=Vector([-3.63, 0, 0]))
        left.rescale(rescale=2, begin_time=t0, transition_time=0)
        right = _panel(polar, name="PolarPanel", location=Vector([0, 0, 0]))
        right.rescale(rescale=2, begin_time=t0, transition_time=0)
        left.appear(begin_time=t0, transition_time=1)
        transition = ibpy.get_geometry_node_from_modifier(polar, "Transition")
        ibpy.change_default_value(transition, from_value=0, to_value=1,
                                  begin_time=t0, transition_time=0)
        t0 = 0.5 + right.appear(begin_time=t0, transition_time=1)

        self.t0 = t0

    def _slit_time_shift(self):
        """The clock offset that makes :meth:`towards_double_slit` continue
        :meth:`towards_slit` without a jump of phase.

        Both waves read ``Scene Time``, and both scenes are rendered from frame
        1 (see :func:`_setup_render`) up to ``duration * FRAME_RATE - 1``. So
        the last frame of ``towards_slit`` shows ``T - 1/FRAME_RATE`` and the
        next one would show ``T``; the first frame of the continuation shows
        ``1/FRAME_RATE`` and has to be moved on by ``T - 1/FRAME_RATE``.
        Starting the new scene at the "same phase" at t = 0 would not be
        enough, even for a whole number of periods T: the frame at ``T`` would
        be skipped and the wave would jump by one frame's worth of phase.
        """
        return self.sub_scenes['towards_slit']['duration'] - 1 / FRAME_RATE

    def towards_double_slit(self):
        """
        continues :meth:`towards_slit` frame for frame: the equations and grids
        make way, the camera swings round to where the wall is seen from the
        side, the slit slides off the axis, a second slit opens at its
        mirror image, and its circular wave spreads out over the first one,
        turning it into the two-slit interference pattern - symmetric about
        the axis, where the central maximum lies.

        The first frame is the last frame of ``towards_slit``, rebuilt with
        transition times of zero - camera, walls, both waves, labels, equations
        and grid panels. The waves continue that scene's clock through the
        ``TimeShift`` dial, so they carry on in phase across the cut.

        The second slit is lit by the same plane wave as the first, whose
        crests reach the wall everywhere at once, so it starts in phase with
        the first slit. That is what ``source_impacts`` models: the new source
        is silent before the slit opens and then spreads out at the phase
        speed, but its phase is taken from the common clock and not from the
        moment of opening.
        """
        t0 = 0
        _setup_render()
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)
        time_shift = self._slit_time_shift()

        # the camera where towards_slit left it: two moves, (-9, 0, 7) and
        # (9, 0, 0), away from (0, -12, 0.2), always tracking the same target
        camera_target = _camera(location=Vector([0, -12, 7.2]), target=(0, 0, 0.2), lens=38)

        # the plane wave, on the left of the wall
        wave = WaveVisualizationModifier(name="WaveVisualization", size=16,
                                         resolution=301,
                                         wavelength=1.5, frequency=1.0,
                                         amplitude=0.6, material=get_texture("function", alpha_intensity=0.9),
                                         source_radius=2 / 10,
                                         emission_strength=0.8,
                                         time_shift=time_shift)
        surface = Plane(name="WaveSurface", u=[-1, 1], v=[-1, 1], resolution=1)
        surface.add_mesh_modifier(type='NODES', node_modifier=wave)
        surface.move(direction=[-8, 0, 0], begin_time=t0, transition_time=0)
        width_node = ibpy.get_geometry_node_from_modifier(wave, label="Width")
        ibpy.change_default_value(width_node, from_value=6, to_value=6, begin_time=t0, transition_time=0)
        surface.appear(begin_time=t0, transition_time=0)

        # the wall. towards_slit has two halves, [-3.1, -0.1] and [0.1, 3.1],
        # with the slit on the axis. The slits end up at +-a, so the pieces are
        # cut for that: the far half is anchored at its outer end and shrinks
        # to [a + h, 3.1] as the slit slides out to +a; the near half is cut
        # at -a + h into a middle piece, anchored there, that stretches to
        # follow the slit, and an outer piece that slides out by a slit's
        # width to open the second one at -a (its far end then sits at
        # -3.1 - 2h instead of -3.1)
        slit_distance = 2.5
        a = slit_distance / 2
        h = 0.1  # half the width of a slit
        wall_right = Plane(name="WallRight", u=[-2, 2], v=[-3, 0],
                           resolution=1, solid=0.025, color="important")
        wall_right.move(direction=[0, 3 + h, 0], begin_time=t0, transition_time=0)
        wall_middle = Plane(name="WallMiddle", u=[-2, 2], v=[0, a - 2 * h],
                            resolution=1, solid=0.025, color="important")
        wall_middle.move(direction=[0, -a + h, 0], begin_time=t0, transition_time=0)
        wall_left = Plane(name="WallLeft", u=[-2, 2], v=[-3 - h, -a + h],
                          resolution=1, solid=0.025, color="important")
        walls = [wall_left, wall_middle, wall_right]
        for wall in walls:
            wall.appear(begin_time=t0, transition_time=0)
            wall.rotate(rotation_euler=[0, pi / 2, 0], begin_time=t0, transition_time=0)
            wall.rescale(rescale=[0.5, 1, 1], begin_time=t0, transition_time=0)

        # the labels, equations and grids as towards_slit leaves them. The
        # labels were turned to face the camera from its intermediate position
        plane_wave_txt = SimpleTexBObject(r"\text{Plane Wave}", aligned="center", text_size="large",
                                          location=[-3, 0, 3.5])
        camera_angle = ibpy.camera_alignment_euler(plane_wave_txt, Vector([-9, -12, 7.2])).x
        circular_wave_txt = SimpleTexBObject(r"\text{Circular Wave}", aligned="center", text_size="large",
                                             location=[3, 0, 3.5])
        wave_eqn = SimpleTexBObject(r"\frac{\partial^2 u}{\partial t^2}=c^2\frac{\partial^2 u}{\partial x^2}",
                                    location=(-3, 0, 2.5), aligned="center")
        wave_eqn2 = SimpleTexBObject(r"\frac{\partial^2 u}{\partial t^2}=c^2\left(\frac{\partial^2 u}{\partial "
                                     r"r^2}+\frac{1}{r}\frac{\partial u}{\partial r}\right)",
                                     aligned="center", location=[3, 0, 2.5])
        texts = [plane_wave_txt, circular_wave_txt, wave_eqn, wave_eqn2]
        for text in texts:
            text.rotate(rotation_euler=[camera_angle, 0, 0], begin_time=t0, transition_time=0)
            text.write(begin_time=t0, transition_time=0)

        grid_kwargs = dict(size=3.0, horizontals=11, verticals=13,
                           thickness=0.018, emission=0.3)
        cartesian = PolarGridModifier(name="CartesianGrid", **grid_kwargs)
        polar = PolarGridModifier(name="PolarGrid", **grid_kwargs, half=True)
        left = _panel(cartesian, name="CartesianPanel",
                      location=Vector([-3.63, 0, 0]))
        right = _panel(polar, name="PolarPanel", location=Vector([0, 0, 0]))
        transition = ibpy.get_geometry_node_from_modifier(polar, "Transition")
        ibpy.change_default_value(transition, from_value=1, to_value=1,
                                  begin_time=t0, transition_time=0)
        panels = [left, right]
        for panel in panels:
            panel.rescale(rescale=2, begin_time=t0, transition_time=0)
            panel.appear(begin_time=t0, transition_time=0)

        # the waves on the right: the old slit rings on, the new one is silent
        # until it opens. Its impact is put in the middle of the opening
        t_camera = 2
        t_move = t_camera + 3
        t_open = t_move + 3
        wave2 = WaveVisualizationModifier(name="DoubleSlitWave", size=8.0,
                                          resolution=301,
                                          sources=((-4, 0), (-4, -a)),
                                          source_impacts=(None, t_open + 0.5),
                                          wavelength=1.5, frequency=1.0,
                                          amplitude=0.31, material=get_texture("function", alpha_intensity=0.9),
                                          source_radius=2 / 10,
                                          emission_strength=0.8,
                                          time_shift=time_shift,
                                          intensity=True, intensity_gain=2.0)
        surface2 = Plane(name="WaveSurface", u=[-1, 1], v=[-1, 1], resolution=1)
        surface2.add_mesh_modifier(type='NODES', node_modifier=wave2)
        surface2.move(direction=[4, 0, 0], begin_time=t0, transition_time=0)
        surface2.appear(begin_time=t0, transition_time=0)

        # clear the stage
        t0 = 0.5
        for obj in [wave_eqn, wave_eqn2] + panels:
            obj.disappear(begin_time=t0, transition_time=1)
        t0 = 0.5 + circular_wave_txt.disappear(begin_time=t0, transition_time=1)

        # swing the camera round to the plane-wave side and up, where the wall
        # is seen face-on enough for its slits to show, and turn the title to
        # face it on the way. Straight along the wall, as in towards_slit, the
        # wall is an edge and a slit is a break in a line
        t0 = t_camera
        camera_location = Vector([-6, -12, 9])
        ibpy.camera_move(shift=camera_location - Vector([0, -12, 7.2]),
                         begin_time=t0, transition_time=2.5)
        camera_target.move(direction=Vector([1.5, 0, 0]) - Vector([0, 0, 0.5]),
                           begin_time=t0, transition_time=2.5)

        def facing(location):
            """the rotation that turns a title at ``location`` to the camera"""
            return (camera_location - Vector(location)).to_track_quat('Z', 'Y').to_euler()

        plane_wave_location = Vector([-2.1, 0, 4.2])
        plane_wave_txt.move(direction=plane_wave_location - Vector([-3, 0, 4.1]),
                            begin_time=t0, transition_time=2.5)
        plane_wave_txt.rotate(rotation_euler=[59 / 180 * pi, 0, -33 / 180 * pi],
                              begin_time=t0, transition_time=2.5)

        # slide the slit out to +a. The source goes with it, on the same
        # (default, eased) keyframe curve as the wall edges, so the wave stays
        # centred on the gap all the way
        t0 = t_move
        wall_right.rescale(rescale=[1, (3 - a) / 3, 1], begin_time=t0, transition_time=2)
        wall_middle.rescale(rescale=[1, (2 * a - 2 * h) / (a - 2 * h), 1], begin_time=t0, transition_time=2)
        ibpy.change_default_vector(ibpy.get_geometry_node_from_modifier(wave2, "Source0"),
                                   from_value=Vector([-4, 0, 0]), to_value=Vector([-4, a, 0]),
                                   begin_time=t0, transition_time=2)

        # open the second slit
        t0 = t_open
        t0 = wall_left.move(direction=[0, -2 * h, 0], begin_time=t0, transition_time=1)

        # the new wave reaches the far corner of the grid, some ten units
        # away, after about seven seconds at c = 1.5
        t0 = ibpy.camera_zoom(lens=53, begin_time=t0, transition_time=7)

        interference_location = [3.53, -2.75, 3.37]
        interference_txt = SimpleTexBObject(r"\text{Interference}", aligned="center", text_size="large",
                                            location=interference_location)
        interference_txt.rescale(rescale=1.116, begin_time=t0, transition_time=0)
        interference_txt.rotate(rotation_euler=[55 / 180 * pi, 0, -31 / 180 * pi], begin_time=t0, transition_time=0)
        t0 = 0.5 + interference_txt.write(begin_time=t0, transition_time=0.5)

        # fifteen seconds of the pattern building up, then the view from
        # above: the titles go, the camera climbs over the right-hand side
        # until the wall is the bottom border of the picture, and the surface
        # settles flat into the time-averaged intensity C^2 + S^2 - the
        # fringes stand still, the dark hyperbolae of the classic figure
        t0 += 15
        for text in [plane_wave_txt, interference_txt]:
            text.disappear(begin_time=t0, transition_time=1)
        t0 += 0.5

        # straight down onto x in [-0.1, 7.9], which now runs up the short
        # side of the 16:9 frame: at lens 53 that takes a height of 21. A
        # hair off the vertical, towards -x: a track-to with UP_Y has no up
        # direction when the camera looks exactly along -z, and the side of
        # the hair sets it - +x points up, the slits sit at the bottom
        top_target = Vector([3.9 - 1.75, 0, 0])
        top_location = Vector([3.89 - 1.75, 0, 21.0])
        ibpy.camera_move(shift=top_location - camera_location,
                         begin_time=t0, transition_time=3)
        camera_target.move(direction=top_target - Vector([1.5, 0, -0.3]),
                           begin_time=t0, transition_time=3)
        t0 = ibpy.camera_zoom(lens=93, begin_time=t0, transition_time=3)
        t0 += 2

        intensity_switch_node = ibpy.get_shader_node_from_material(wave2.materials[0], "IntensitySwitch")
        alpha_intensity_node = ibpy.get_shader_node_from_material(wave2.materials[0], "AlphaIntensity")

        ibpy.change_default_value(intensity_switch_node, from_value=0, to_value=1, begin_time=t0, transition_time=2)
        t0 = 1 + ibpy.change_default_value(alpha_intensity_node, from_value=0.9, to_value=1, begin_time=t0,
                                           transition_time=2)

        self.t0 = t0

    # ------------------------------------------------------------------
    def towards_grating(self):
        r"""
        continues the last picture of :meth:`towards_double_slit` in light: the
        two slits become a grating, and its wavelength is tuned.

        All lengths are now nanometres, at 100 nm to the blender unit, and the
        surface is painted by :func:`~appearance.textures.grating_texture`,
        whose colour is the ``Wavelength`` dial through
        :class:`~shader_nodes.shader_nodes.WaveLengthToRGB`. The picture of
        ``towards_double_slit`` is lambda = 1.5 with the slits 2.5 apart; the
        wavelength whose colour is the pure yellow (1, 1, 0) of that picture is
        570 nm, so the whole scene is scaled by s = 5.7/1.5 = 3.8: the slits
        sit at +-475 nm, 950 nm apart, and camera, wall and surface are the old
        ones times s. The field normalisation pi/sqrt(lambda) is not scale free,
        so the amplitude is sqrt(s), and the clock continues the old one
        through ``TimeShift``. The first frame is therefore the last frame of
        ``towards_double_slit``.

        Then the camera backs off, and only once it has, two more slits open
        at +-1425 nm, one period out on either side; their waves spread out
        on the common phase. Finally the wavelength is tuned from 570 nm to red
        at 650 nm, where the orders sin(alpha_n) = n lambda/d open up (the first
        one from 36.9 to 43.2 degrees), back through yellow and on to blue at
        450 nm, where they close in to 28.3 degrees and the second order comes
        in at 71.3 degrees.
        """
        t0 = 0
        _setup_render()
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        s = 3.8
        nm_per_unit = 100
        a = 1.25 * s
        h = 0.1 * s

        # towards_double_slit ends at (2.14, 0, 21), aimed at (2.15, 0, 0)
        camera_location = Vector([2.14, 0, 21]) * s
        camera_target = _camera(location=camera_location, target=Vector([2.15, 0, 0]) * s, lens=93)

        # the wall as towards_double_slit leaves it, from y = -3.3 s to 3.1 s,
        # every piece built with its origin at the edge of a slit, so that it
        # can grow outwards
        wall_right = Plane(name="WallRight", u=[-2 * s, 2 * s], v=[0, 3.1 * s - a - h],
                           resolution=1, solid=0.025 * s, color="important")
        wall_right.move(direction=[0, a + h, 0], begin_time=t0, transition_time=0)
        wall_middle = Plane(name="WallMiddle", u=[-2 * s, 2 * s], v=[-a + h, a - h],
                            resolution=1, solid=0.025 * s, color="important")
        wall_left = Plane(name="WallLeft", u=[-2 * s, 2 * s], v=[a + h - 3.3 * s, 0],
                          resolution=1, solid=0.025 * s, color="important")
        wall_left.move(direction=[0, -a - h, 0], begin_time=t0, transition_time=0)
        # the outer pieces of the grating, waiting outside the picture
        far = 130
        wall_far_right = Plane(name="WallFarRight", u=[-2 * s, 2 * s], v=[0, far],
                               resolution=1, solid=0.025 * s, color="important")
        wall_far_right.move(direction=[0, 3 * a + h + far, 0], begin_time=t0, transition_time=0)
        wall_far_left = Plane(name="WallFarLeft", u=[-2 * s, 2 * s], v=[-far, 0],
                              resolution=1, solid=0.025 * s, color="important")
        wall_far_left.move(direction=[0, -3 * a - h - far, 0], begin_time=t0, transition_time=0)
        walls = [wall_left, wall_middle, wall_right, wall_far_left, wall_far_right]
        for wall in walls:
            wall.appear(begin_time=t0, transition_time=0)
            wall.rotate(rotation_euler=[0, pi / 2, 0], begin_time=t0, transition_time=0)
            wall.rescale(rescale=[0.5, 1, 1], begin_time=t0, transition_time=0)

        # the surface: x in [0, 8 s] as before, and wide enough for the zoom
        # out, cut to the old |y| < 4 s by HalfWidth until then
        t_zoom = 1
        zoom_time = 12
        t_open = t_zoom + zoom_time + 0.5
        t_impact = t_open + 2
        time_shift = self._slit_time_shift() + self.sub_scenes['towards_double_slit']['duration'] \
                     - 1 / FRAME_RATE
        duration = self.sub_scenes['towards_grating']['duration']

        # the clock of the waves runs at scene speed until the zoom, speeds up
        # during it and runs four times as fast from then on: at the distance
        # of the far field the old pace would leave the new waves twenty
        # seconds to reach the top of the picture
        speed = 4
        t_fast = t_zoom + zoom_time
        clock = [0, t_zoom, t_zoom + zoom_time * (1 + speed) / 2]
        clock.append(clock[-1] + speed * (duration - t_fast))
        impact = clock[2] + speed * (t_impact - t_fast)

        screen = Plane(u=[0, 135], v=[-125, 125], resolution=1, name="GratingSurface",
                       color="grating", sources=[(0, -3 * a), (0, -a), (0, a), (0, 3 * a)],
                       source_impacts=[impact, None, None, impact],
                       wavelength=570, nm_per_unit=nm_per_unit, frequency=1.0,
                       amplitude=np.sqrt(s), time_shift=time_shift,
                       source_radius=0.2 * s, half_width=4 * s)
        screen.appear(begin_time=t0, transition_time=0)
        material = ibpy.get_material_of(screen)
        time_node = ibpy.get_node_from_shader(material, "Time")
        for (begin, end), (start, stop) in zip([(0, t_zoom), (t_zoom, t_fast), (t_fast, duration)],
                                               zip(clock, clock[1:])):
            ibpy.change_default_value(time_node, from_value=start, to_value=stop,
                                      begin_time=begin, transition_time=end - begin)
        _linearize(material.node_tree)

        # the directions of the orders, sin(alpha_n) = n lambda/g, drawn from
        # the middle of the grating. They shoot out shortly after the start,
        # and the host is scaled with the zoom, so that length and thickness
        # keep their size in the picture
        reach = 14
        rays = FarFieldModifier(name="FarField", spacing=2 * a, wavelength=570 / nm_per_unit,
                                max_order=int(2 * a / (450 / nm_per_unit)), reach=reach,
                                radius=0.05, axis=(0, 1, 0), normal=(1, 0, 0),
                                color="text", emission=1.0)
        host = Cube(name="FarFieldHost", location=Vector((0, 0, 1)))
        host.add_mesh_modifier(type='NODES', node_modifier=rays, name="far field approximation")
        host.appear(begin_time=0.5, transition_time=0)
        reach_node = ibpy.get_geometry_node_from_modifier(rays, "Reach")
        ibpy.change_default_value(reach_node, from_value=0, to_value=reach,
                                  begin_time=0.5, transition_time=1)

        # back off first, far enough for the far field of the four slits,
        # which sets in at D^2/lambda = 2850^2/570 nm, some 14 um: at lens 50
        # the frame is 231 wide and 130 high, and the wall stays at the bottom
        t0 = t_zoom
        new_location = Vector([64.4, 0, 321])
        ibpy.camera_move(shift=new_location - camera_location, begin_time=t0, transition_time=zoom_time)
        camera_target.move(direction=Vector([64.4 + 0.04, 0, 0]) - Vector([2.15, 0, 0]) * s,
                           begin_time=t0, transition_time=zoom_time)
        ibpy.camera_zoom(lens=50, begin_time=t0, transition_time=zoom_time)
        host.rescale(rescale=130 / (2 * 8.69), begin_time=t0, transition_time=zoom_time)
        ibpy.change_default_value(ibpy.get_node_from_shader(material, "HalfWidth"),
                                  from_value=4 * s, to_value=125, begin_time=t0, transition_time=zoom_time)

        # then two more slits, one period out on either side: the wall grows
        # out to the new slits and the outer pieces slide in behind them
        t0 = t_open
        wall_right.rescale(rescale=[1, (2 * a - 2 * h) / (3.1 * s - a - h), 1], begin_time=t0, transition_time=2)
        wall_left.rescale(rescale=[1, (2 * a - 2 * h) / (3.3 * s - a - h), 1], begin_time=t0, transition_time=2)
        wall_far_right.move(direction=[0, -far, 0], begin_time=t0, transition_time=0.5)
        t0 = wall_far_left.move(direction=[0, far, 0], begin_time=t0, transition_time=0.5)

        # the new waves reach the top of the picture, some 130 away, after
        # about six seconds at 4 c = 23 per second
        t0 += 3

        # towards red: the orders open up; back to yellow and on to blue: the
        # orders close in, and the second one comes in from the wall. The
        # rays take the same keyframes as the material, in blender units
        wavelength = ibpy.get_node_from_shader(material, "Wavelength")
        ray_wavelength = ibpy.get_geometry_node_from_modifier(rays, "Wavelength")
        for start, stop, transition_time, pause in [(570, 650, 2, 0.5), (650, 570, 3, 0), (570, 450, 2, 0.5)]:
            ibpy.change_default_value(ray_wavelength, from_value=start / nm_per_unit,
                                      to_value=stop / nm_per_unit, begin_time=t0,
                                      transition_time=transition_time)
            t0 = pause + ibpy.change_default_value(wavelength, from_value=start, to_value=stop,
                                                   begin_time=t0, transition_time=transition_time)

        self.t0 = t0

    def towards_rainbow(self):
        r"""
        continues :meth:`towards_grating` from its last frame, in blue at
        450 nm, and blends in green at 530 nm and then red at 650 nm, each
        with its own fan of rays: three colours from one grating, each sent
        into its own directions.

        The first frame is the last one of ``towards_grating``, rebuilt with
        transition times of zero: the camera backed off to lens 50, the four
        slits 950 nm apart, the white rays scaled with the zoom, and the clock
        running four times as fast, carried on through ``Time``. The white
        rays of blue then turn blue; the zeroth order, which every colour
        shares, turns cyan with green and white again with red.

        The surface is :func:`~appearance.textures.grating_texture` with three
        channels, and a channel's ``Weight`` is what blends it in; the rays of
        a new colour shoot out while it does.
        """
        t0 = 0
        _setup_render()
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        s = 3.8
        nm_per_unit = 100
        a = 1.25 * s
        h = 0.1 * s
        duration = self.sub_scenes['towards_rainbow']['duration']

        location = Vector([64.4, 0, 321])
        _camera(location=location, target=location + Vector([0.04, 0, -321]), lens=50)

        # the wall as towards_grating leaves it: slits at +-a and +-3a
        far = 130
        for name, v in [("WallFarLeft", [-3 * a - h - far, -3 * a - h]), ("WallLeft", [-3 * a + h, -a - h]),
                        ("WallMiddle", [-a + h, a - h]), ("WallRight", [a + h, 3 * a - h]),
                        ("WallFarRight", [3 * a + h, 3 * a + h + far])]:
            wall = Plane(name=name, u=[-2 * s, 2 * s], v=v, resolution=1, solid=0.025 * s, color="important")
            wall.appear(begin_time=t0, transition_time=0)
            wall.rotate(rotation_euler=[0, pi / 2, 0], begin_time=t0, transition_time=0)
            wall.rescale(rescale=[0.5, 1, 1], begin_time=t0, transition_time=0)

        # towards_grating's clock reads 111 at its end, t_zoom + zoom_time
        # (1 + speed)/2 + speed (duration - t_zoom - zoom_time) with 1, 4, 4
        # and 30, and runs at four times scene speed; its last frame shows
        # 111 - 4/FRAME_RATE and the first frame here the next one
        speed = 4
        clock = 111 - speed / FRAME_RATE
        time_shift = self._slit_time_shift() + self.sub_scenes['towards_double_slit']['duration'] \
                     - 1 / FRAME_RATE
        wavelengths = [450, 530, 650]
        screen = Plane(u=[0, 135], v=[-125, 125], resolution=1, name="GratingSurface",
                       color="grating", sources=[(0, -3 * a), (0, -a), (0, a), (0, 3 * a)],
                       wavelengths=wavelengths, weights=[1, 0, 0], nm_per_unit=nm_per_unit,
                       frequency=1.0, amplitude=np.sqrt(s), time_shift=time_shift,
                       source_radius=0.2 * s, half_width=125)
        screen.appear(begin_time=t0, transition_time=0)
        material = ibpy.get_material_of(screen)
        ibpy.change_default_value(ibpy.get_node_from_shader(material, "Time"),
                                  from_value=clock, to_value=clock + speed * duration,
                                  begin_time=0, transition_time=duration)
        _linearize(material.node_tree)

        # one fan of rays per colour: blue's is towards_grating's white one,
        # the new colours draw only their own orders
        reach = 14
        zoom = 130 / (2 * 8.69)
        reaches = []
        fans = []
        for c, (lam, color) in enumerate(zip(wavelengths, ["text", "green", "red"])):
            rays = FarFieldModifier(name="FarField%d" % c, spacing=2 * a, wavelength=lam / nm_per_unit,
                                    max_order=int(2 * a / (lam / nm_per_unit)), reach=reach if c == 0 else 0,
                                    radius=0.05, axis=(0, 1, 0), normal=(1, 0, 0),
                                    color=color, draw_zeroth=c == 0, emission=1.0 if c == 0 else 3.0)
            host = Cube(name="FarFieldHost%d" % c, location=Vector((0, 0, 1)))
            host.add_mesh_modifier(type='NODES', node_modifier=rays, name="far field approximation %d" % c)
            host.rescale(rescale=zoom, begin_time=t0, transition_time=0)
            host.appear(begin_time=t0, transition_time=0)
            reaches.append(ibpy.get_geometry_node_from_modifier(rays, "Reach"))
            fans.append(rays)

        # blue's rays turn from white to blue before the other colours come
        for ray_material in fans[0].materials:
            ibpy.change_default_value(ibpy.create_color_mixing_find_previous_color(ray_material, "blue"),
                                      from_value=0, to_value=1, begin_time=0.5, transition_time=0.25)
            ibpy.change_default_value(ray_material.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"],
                                      from_value=1, to_value=3, begin_time=0.5, transition_time=0.25)

        alpha_factor_node = ibpy.get_node_from_shader(ibpy.get_material_at_slot(screen, 0), "AlphaFactor")

        # green, then red
        t0 = 3
        alphas = [1, 0.5, 0.2]
        for c in [1, 2]:
            ibpy.change_default_value(ibpy.get_node_from_shader(material, "Weight%d" % c),
                                      from_value=0, to_value=1, begin_time=t0, transition_time=3)
            ibpy.change_default_value(reaches[c], from_value=0, to_value=reach,
                                      begin_time=t0 + 0.5, transition_time=2)
            # the zeroth order is every colour's: cyan for blue and green,
            # white once red is there as well
            ibpy.change_default_value(ibpy.create_color_mixing_find_previous_color(fans[0].materials[0],
                                                                                   ["cyan", "text"][c - 1]),
                                      from_value=0, to_value=1, begin_time=t0, transition_time=3)
            ibpy.change_default_value(alpha_factor_node.inputs["Factor"],
                                      from_value=alphas[c - 1], to_value=alphas[c], begin_time=t0, transition_time=2)
            t0 += 3

        self.t0 = t0

    def separation_overview(self):
        r"""
        The first half of the script section *Separation of variables*, up
        to "Let's see how this works": what the method does, where it led,
        and the book everything ended up in.

        1. **The workflow**, sketched by hand like a diagram on a whiteboard:
           the wave equation, the product ansatz, and the two branches it
           falls apart into - the ODE in t (red), solved at once by sine and
           cosine, and the PDE in r and phi (blue), which with circular
           symmetry becomes Bessel's equation. Boxes and arrows are pen
           strokes (:func:`_pen`), drawn in reading order. The spatial
           branch is then marked as the hard one.
        2. **The zoo**: the equations in the order the script names them -
           waves, heat, quantum mechanics, then the ODEs of Bessel, Legendre,
           Hermite and Airy, each with its name.
        3. **The bible of special functions**: the cover of Abramowitz and
           Stegun and, laid over it, the NIST Digital Library that continues
           it. Both pictures come from ``media/raster``.
        """
        t0 = 0
        duration = self.sub_scenes['separation_overview']['duration']
        _setup_render()
        _lights(target=(0, 0, 0), strength=0.4)
        # 32 mm from 13 back holds x in [-7.3, 7.3], z in [-4.1, 4.1]
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=32)
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        rng = np.random.default_rng(20261003)  # one seed, one hand
        strokes = []

        def extent(text):
            # the glyphs' box in world x and z; the text stands in x-z, so
            # its local y is world z
            boxes = np.array([ibpy.get_bounding_box_for_letter(letter)
                              for letter in text.letters])
            scale, origin = text.ref_obj.scale, text.ref_obj.location
            return (origin.x + scale[0] * boxes[:, 0].min(),
                    origin.x + scale[0] * boxes[:, 3].max(),
                    origin.z + scale[1] * boxes[:, 1].min(),
                    origin.z + scale[1] * boxes[:, 4].max())

        def sketch(points, name, color, begin_time, thickness=0.45,
                   emission=0.5, transition_time=0.35):
            curve = _ink(points, name, extrude=0, color=color,
                         emission=emission, thickness=thickness)
            curve.grow(begin_time=begin_time, transition_time=transition_time)
            strokes.append(curve)
            return begin_time + transition_time

        def box(frame, name, color, begin_time, **style):
            x0, x1, z0, z1 = frame
            corners = [Vector((x0, 0, z1)), Vector((x1, 0, z1)),
                       Vector((x1, 0, z0)), Vector((x0, 0, z0))]
            t = begin_time
            for i in range(4):
                t = sketch(_pen(corners[i], corners[(i + 1) % 4], rng,
                                samples=9), "%s_%d" % (name, i), color, t,
                           transition_time=0.25, **style)
            return t

        def arrow(start, end, name, color, begin_time, head=0.22):
            start, end = Vector(start), Vector(end)
            t = sketch(_pen(start, end, rng, samples=9), name + "_shaft",
                       color, begin_time, transition_time=0.4)
            back = (start - end).normalized() * head
            for sign in (-1, 1):
                turned = Vector((back.x * np.cos(0.45) - sign * back.z * np.sin(0.45),
                                 0,
                                 sign * back.x * np.sin(0.45) + back.z * np.cos(0.45)))
                sketch(_pen(end, end + turned, rng, samples=4, bow=0, overshoot=0),
                       "%s_head%d" % (name, sign + 1), color, t,
                       transition_time=0.15)
            return t + 0.15

        def caption(text, x, z, color="joker"):
            return SimpleTexBObject(r"\text{%s}" % text, text_size="small",
                                    color=color, aligned="center",
                                    location=(x, 0, z))

        # ---- 1. the workflow -------------------------------------------------
        title = SimpleTexBObject(r"\text{Separation of Variables}",
                                 text_size="large", color="example",
                                 aligned="center", location=(0, 0, 3.6))
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.2)

        left_x, right_x = -3.7, 3.4
        rows = [2.25, 0.55, -1.15, -3.05]
        nodes = [
            # (formula, x, z, caption, color)
            (r"\frac{\partial^2 u}{\partial t^2}=c^2\,\Delta u", 0, rows[0],
             "wave equation (PDE)", "example"),
            (r"u(\vec r,t)=H(\vec r)\,T(t)", 0, rows[1],
             "ansatz: a product", "example"),
            (r"\ddot T(t)=-\omega^2\,T(t)", left_x, rows[2],
             "ODE in $t$", "custom1"),
            (r"\frac{\partial^2 H}{\partial r^2}+\frac{1}{r}\frac{\partial H}"
             r"{\partial r}+\frac{1}{r^2}\frac{\partial^2 H}{\partial\varphi^2}"
             r"+k^2H=0", right_x, rows[2], r"PDE in $r,\varphi$", "drawing"),
            (r"T(t)\in\left\{\cos\omega t,\;\sin\omega t\right\}", left_x,
             rows[3], "sine and cosine", "custom1"),
            (r"H''+\frac{1}{r}H'+k^2H=0", right_x, rows[3],
             "Bessel's equation", "drawing"),
        ]
        # every formula is built (unwritten) and measured first, so that the
        # arrows can be aimed at boxes that are not drawn yet
        texts, frames, captions = [], [], []
        for i, (formula, x, z, label, color) in enumerate(nodes):
            text = SimpleTexBObject(formula, text_size="small", color="text",
                                    aligned="center", location=(x, 0, z),
                                    name="SeparationNode%d" % i)
            x0, x1, z0, z1 = extent(text)
            frames.append((x0 - 0.28, x1 + 0.28, z0 - 0.2, z1 + 0.2))
            texts.append(text)
            captions.append(caption(label, x, frames[-1][3] + 0.3))

        def draw_node(i, begin_time):
            t = texts[i].write(begin_time=begin_time, transition_time=1.0)
            t = box(frames[i], "SeparationBox%d" % i, nodes[i][4], t - 0.4)
            captions[i].write(begin_time=t - 0.6, transition_time=0.6)
            return t

        def below(i):
            return frames[i][2] - 0.06

        def above(i):
            # over the caption, not into it
            return frames[i][3] + 0.62

        t0 = 0.5 + draw_node(0, t0)
        t0 = arrow((0, 0, below(0)), (0, 0, above(1)), "SeparationArrow01",
                   "text", t0)
        t0 = 1.0 + draw_node(1, t0)
        # the product falls apart: one arrow to each branch
        f1 = frames[1]
        arrow((f1[0] + 0.3, 0, below(1)), (left_x + 0.6, 0, above(2)),
              "SeparationArrow12", "custom1", t0)
        t0 = arrow((f1[1] - 0.3, 0, below(1)), (right_x - 0.6, 0, above(3)),
                   "SeparationArrow13", "drawing", t0)
        draw_node(2, t0)
        t0 = 1.0 + draw_node(3, t0 + 0.6)
        # the time part is solved quickly
        t0 = arrow((left_x, 0, below(2)), (left_x, 0, above(4)),
                   "SeparationArrow24", "custom1", t0)
        t0 = 1.5 + draw_node(4, t0)
        # the spatial part: with circular symmetry, nothing depends on phi
        t0 = arrow((right_x, 0, below(3)), (right_x, 0, above(5)),
                   "SeparationArrow35", "drawing", t0)
        symmetry = SimpleTexBObject(r"\partial_\varphi=0", text_size="small",
                                    color="joker", aligned="left",
                                    location=(right_x + 0.3, 0,
                                              0.5 * (below(3) + above(5))))
        symmetry.write(begin_time=t0 - 0.4, transition_time=0.6)
        t0 = 1.0 + draw_node(5, t0)

        # the spatial part is the hard one: both of its boxes are drawn over
        # once more, wider and glowing, and the bottom row says why
        for i in (3, 5):
            x0, x1, z0, z1 = frames[i]
            t_hard = box((x0 - 0.12, x1 + 0.12, z0 - 0.1, z1 + 0.1),
                         "SeparationHard%d" % i, "important", t0,
                         thickness=0.7, emission=1.4)
        hard_label = SimpleTexBObject(r"\text{the hard part}", text_size="normal",
                                      color="important", aligned="center",
                                      location=(0.15, 0, rows[3]))
        t0 = hard_label.write(begin_time=t_hard, transition_time=1)
        # held while the narration says why the spatial part is the hard one
        t0 = max(t0 + 2.5, 32)

        # ---- clear the board -------------------------------------------------
        for obj in texts + captions + strokes + [symmetry, hard_label, title]:
            obj.disappear(begin_time=t0, transition_time=1)
        t0 += 1.5

        # ---- 2. the zoo, in the order the script names it ---------------------
        zoo = [
            (r"\frac{\partial^2 u}{\partial t^2}=c^2\,\Delta u", "waves"),
            (r"\frac{\partial u}{\partial t}=D\,\Delta u", "heat"),
            (r"i\hbar\frac{\partial\psi}{\partial t}=-\frac{\hbar^2}{2m}"
             r"\Delta\psi+V\psi", "quantum mechanics"),
            (r"x^2y''+xy'+\left(x^2-n^2\right)y=0", "Bessel"),
            (r"\left(1-x^2\right)y''-2xy'+\ell(\ell+1)\,y=0", "Legendre"),
            (r"y''-2xy'+2ny=0", "Hermite"),
            (r"y''-xy=0", "Airy"),
        ]
        list_x, label_x = -6.9, -1.9
        zoo_z = [2.9, 2.0, 1.1, -0.35, -1.25, -2.15, -3.05]
        heading_pde = caption("linear PDEs of physics", -3.95, 3.6,
                              color="example")
        heading_ode = caption("and the special functions they demand", -3.95,
                              0.4, color="example")
        t0 = 0.3 + heading_pde.write(begin_time=t0, transition_time=0.8)
        zoo_texts = []
        for i, ((formula, name), z) in enumerate(zip(zoo, zoo_z)):
            if i == 3:
                t0 = 0.3 + heading_ode.write(begin_time=t0, transition_time=0.8)
            equation = SimpleTexBObject(formula, text_size="small",
                                        color="drawing" if i < 3 else "custom1",
                                        aligned="left", location=(list_x, 0, z),
                                        name="Zoo%d" % i)
            label = SimpleTexBObject(r"\text{%s}" % name, text_size="small",
                                     color="joker", aligned="right",
                                     location=(label_x, 0, z))
            equation.write(begin_time=t0, transition_time=0.8)
            t0 = 0.9 + label.write(begin_time=t0 + 0.4, transition_time=0.5)
            zoo_texts += [equation, label]

        # ---- 3. the handbook and its online continuation ----------------------
        # Plane turns its location along with the mesh, so locations are
        # given in its own frame: (x, z, -y) for a world point (x, y, z)
        book_height = 4.6
        book = Plane(u=[-book_height * 602 / 768 / 2, book_height * 602 / 768 / 2],
                     v=[-book_height / 2, book_height / 2],
                     rotation_euler=[pi / 2, 0, 0], location=[3.1, 0.75, 0],
                     color="image", src="Abramowitz-Stegun1965.png",
                     emission=0.6, resolution=1, name="AbramowitzStegun")
        t0 = 0.3 + book.appear(begin_time=t0, transition_time=1)
        book_title = SimpleTexBObject(
            r"\text{Abramowitz \& Stegun, 1964}", text_size="normal",
            color="example", aligned="center", location=(3.1, 0, 3.5))
        t0 = 2.0 + book_title.write(begin_time=t0, transition_time=1)

        nist_width = 4.4
        nist = Plane(u=[-nist_width / 2, nist_width / 2],
                     v=[-nist_width * 961 / 1418 / 2, nist_width * 961 / 1418 / 2],
                     rotation_euler=[pi / 2, 0, 0], location=[4.6, -1.75, 0.3],
                     color="image", src="NIST.png", emission=0.25, resolution=1,
                     name="NISTDLMF")
        t0 = 0.3 + nist.appear(begin_time=t0, transition_time=1)
        # a web page is mostly white paper, and lit by the suns it clears
        # the glow threshold and blooms into a light box; the image is
        # multiplied down before it reaches the shader
        material = ibpy.get_obj(nist).active_material
        tree = material.node_tree
        image = next(node for node in tree.nodes if node.type == 'TEX_IMAGE')
        dim = tree.nodes.new('ShaderNodeMix')
        dim.data_type = 'RGBA'
        dim.blend_type = 'MULTIPLY'
        dim.inputs[0].default_value = 1
        dim.inputs[7].default_value = (0.55, 0.55, 0.55, 1)
        for link in [link for link in tree.links
                     if link.from_socket == image.outputs['Color']]:
            target = link.to_socket
            tree.links.remove(link)
            tree.links.new(dim.outputs[2], target)
        tree.links.new(image.outputs['Color'], dim.inputs[6])
        nist_title = SimpleTexBObject(
            r"\text{NIST Digital Library of Mathematical Functions}",
            text_size="small", color="example", aligned="center",
            location=(4.0, 0, -3.65))
        t0 = 1.0 + nist_title.write(begin_time=t0, transition_time=1)

        self.t0 = max(t0, duration)

    def separation_of_variables(self):
        r"""
        The circular wave out of the radial wave equation, by separation of
        variables - the scene for the script section of the same name.

        The ansatz ``u = H(r) T(t)`` turns the radial wave equation into one
        line whose left-hand side only knows t and whose right-hand side only
        knows r; both are set equal to ``-k^2``. The line then falls apart:
        its left half becomes the oscillator ``T'' = -omega^2 T`` (solved by
        cosine and sine), its right half Bessel's equation of order zero
        (solved by J_0 and Y_0). The outgoing wave is built from the two
        solution lists - not a single product but a linear combination
        ``A J_0 cos + B Y_0 sin`` - and, with the far-field forms of J_0 and
        Y_0 as a side note, ``B = A`` selects the outgoing wave, which
        collapses into one travelling cosine ``cos(kr - omega t - pi/4)``.

        Layout: derivation hanging from the left margin, premises and side
        notes in a right-hand column at the height of the line they act on.
        """
        t0 = 0
        _setup_render()
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        T = r"T(t)"
        ddT = r"\ddot{T}(t)"
        H = r"H(r)"
        J0 = r"{\rm J}_0(kr)"
        Y0 = r"{\rm Y}_0(kr)"
        cos_wt = r"\cos\omega t"
        sin_wt = r"\sin\omega t"
        note_x = 3.9

        title = SimpleTexBObject(r"\text{Separation of Variables}",
                                 text_size="normal", color="example",
                                 aligned="center", location=(0, 0, 3.0))
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1)

        # ---- the radial wave equation and the ansatz --------------------------
        derivation = BDerivation(
            r"\frac{\partial^2 u}{\partial t^2}=c^2\left(\frac{\partial^2 u}"
            r"{\partial r^2}+\frac{1}{r}\frac{\partial u}{\partial r}\right)",
            location=(-4.5, 0, 2.0), line_spacing=(0, 0, -1.15),
            text_size="normal", color="text")
        t0 = 0.5 + derivation.write(begin_time=t0, transition_time=1.5)

        ansatz = SimpleTexBObject(r"u(r,t)=%s\,%s" % (H, T), text_size="normal",
                                  color="important", aligned="center",
                                  location=(note_x, 0, 2.0))
        t0 = 0.5 + ansatz.write(begin_time=t0, transition_time=1)
        highlight_letters(ansatz, list(range(len(ansatz.letters))),
                          color="important", begin_time=t0, transition_time=1)

        # every u becomes the product; the derivatives act on one factor only
        d_tt = r"\frac{\partial^2 u}{\partial t^2}"
        d_rr = r"\frac{\partial^2 u}{\partial r^2}"
        d_r = r"\frac{\partial u}{\partial r}"
        t0 = 1 + derivation.step(
            r"%s\,%s=c^2\left(H''(r)\,%s+\frac{1}{r}H'(r)\,%s\right)"
            % (H, ddT, T, T),
            map={d_tt: r"%s\,%s" % (H, ddT),
                 d_rr: r"H''(r)\,%s" % T,
                 d_r: r"H'(r)\,%s" % T},
            highlight=[d_tt, d_rr, d_r], highlight_color="important",
            begin_time=t0, transition_time=2)

        # ---- divide by u ------------------------------------------------------
        divide = SimpleTexBObject(r"\div\; %s\,%s" % (H, T), text_size="normal",
                                  color="joker", aligned="center",
                                  location=(note_x, 0, 0.3))
        t0 = 0.5 + divide.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + derivation.step(
            r"\frac{%s}{%s}=c^2\left(\frac{H''(r)}{%s}+\frac{1}{r}"
            r"\frac{H'(r)}{%s}\right)" % (ddT, T, H, H),
            map={ddT: ddT, "H''(r)": "H''(r)", "H'(r)": "H'(r)"},
            begin_time=t0, transition_time=2)
        # c^2 crosses to the left; the bracket is no longer needed
        t0 = 1 + derivation.step(
            r"c^{-2}\frac{%s}{%s}=\frac{H''(r)}{%s}+\frac{1}{r}"
            r"\frac{H'(r)}{%s}" % (ddT, T, H, H),
            mode="replace", map={"c^2": "c^{-2}"},
            begin_time=t0, transition_time=1.5)

        # ---- one side only knows t, the other only r --------------------------
        left = r"c^{-2}\frac{%s}{%s}" % (ddT, T)
        right = r"\frac{H''(r)}{%s}+\frac{1}{r}\frac{H'(r)}{%s}" % (H, H)
        derivation.highlight(left, color="custom1", begin_time=t0,
                             transition_time=2)
        t0 = 0.5 + derivation.highlight(right, color="drawing",
                                        begin_time=t0 + 1, transition_time=2)
        t0 = 0.5 + derivation.step(
            r"%s=%s=-k^2" % (left, right),
            mode="replace", map={None: "=-k^2"},
            begin_time=t0, transition_time=1)
        separated = derivation.current
        wave_number = SimpleTexBObject(r"k=\frac{2\pi}{\lambda}",
                                       text_size="normal", color="joker",
                                       aligned="center",
                                       location=(note_x, 0, -0.3))
        t0 = 1 + wave_number.write(begin_time=t0, transition_time=1)

        # ---- the time part: an oscillator -------------------------------------
        t0 = 0.5 + derivation.step(
            r"%s=-k^2c^2\,%s" % (ddT, T),
            map={"c^{-2}": "c^2", ddT: ddT, T: T,
                 "-k^2": "-k^2", "=@0": "="},
            auto=False, highlight=left, highlight_color="custom1",
            begin_time=t0, transition_time=2)
        omega = SimpleTexBObject(r"\omega=kc=\frac{2\pi}{T}", text_size="normal",
                                 color="joker", aligned="center",
                                 location=(note_x, 0, -1.45))
        t0 = 0.5 + omega.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + derivation.step(r"%s=-\omega^2\,%s" % (ddT, T),
                                   mode="replace", map={"k^2c^2": r"\omega^2"},
                                   begin_time=t0, transition_time=1.5)
        t0 = 1 + derivation.step(r"%s\in\left\{%s,\;%s\right\}" % (T, cos_wt, sin_wt),
                                 mode="replace", new_color="custom1",
                                 begin_time=t0, transition_time=1.5)
        time_line = derivation.current

        # ---- the radial part: Bessel's equation -------------------------------
        # the right half of the separated line flies down and is multiplied
        # by H(r); what is left of that line goes. The fade is keyed after
        # the step: the flying copies duplicate their donor's material
        # action, so a fade keyed before would be copied into them as well
        step_begin = t0
        t0 = 0.5 + derivation.step(
            r"H''(r)+\frac{1}{r}H'(r)+k^2%s=0" % H,
            auto=False, align=False,
            sources=[(separated, "H''(r)", "H''(r)"),
                     (separated, r"\frac{1}{r}", r"\frac{1}{r}"),
                     (separated, "H'(r)", "H'(r)"),
                     (separated, "k^2", "k^2")],
            begin_time=t0, transition_time=2)
        separated.disappear(begin_time=step_begin + 1, transition_time=1)
        bessel = SimpleTexBObject(r"\text{Bessel's equation}",
                                  text_size="normal", color="joker",
                                  aligned="center", location=(note_x, 0, -2.6))
        t0 = 1 + bessel.write(begin_time=t0, transition_time=1)
        t0 = 1 + derivation.step(r"%s\in\left\{%s,\;%s\right\}" % (H, J0, Y0), mode="replace",
                                 align=False, begin_time=t0,
                                 transition_time=1.5)
        radial_line = derivation.current

        # ---- clear the board down to the two solution lists -------------------
        for text in [title, ansatz, divide, wave_number, omega, bessel] \
                + derivation.lines[:2]:
            text.disappear(begin_time=t0, transition_time=1)
        t0 += 1.5
        t0 = 0.5 + derivation.move(
            Vector((0, 0, 2.0 - time_line.ref_obj.location.z)),
            begin_time=t0, transition_time=1)

        # ---- the outgoing wave: a sum of two products -------------------------
        # (fades keyed after the step, see the radial part)
        step_begin = t0
        t0 = 1 + derivation.step(
            r"u(r,t)=A\,%s%s+B\,%s%s" % (J0, cos_wt, Y0, sin_wt),
            map={J0: J0, Y0: Y0}, auto=False, align=False,
            sources=[(time_line, cos_wt, cos_wt),
                     (time_line, sin_wt, sin_wt)],
            begin_time=t0, transition_time=2)
        time_line.disappear(begin_time=step_begin + 2, transition_time=1)
        radial_line.disappear(begin_time=step_begin + 2, transition_time=1)
        derivation.highlight(J0 + cos_wt, color="custom1",
                             begin_time=t0, transition_time=1.5)
        t0 = 1 + derivation.highlight(Y0 + sin_wt, color="drawing",
                                      begin_time=t0 + 1, transition_time=1.5)

        # ---- far from the source ----------------------------------------------
        far_field = SimpleTexBObject(
            r"{\rm J}_0(x)\approx\sqrt{\tfrac{2}{\pi x}}\cos\left(x-\tfrac{\pi}"
            r"{4}\right),\quad{\rm Y}_0(x)\approx\sqrt{\tfrac{2}{\pi x}}"
            r"\sin\left(x-\tfrac{\pi}{4}\right)",
            text_size="normal", color="joker", aligned="center",
            location=(0, 0, -2.9))
        t0 = 1 + far_field.write(begin_time=t0, transition_time=2)
        cos_kr = r"\cos\left(kr-\tfrac{\pi}{4}\right)"
        sin_kr = r"\sin\left(kr-\tfrac{\pi}{4}\right)"
        t0 = 1 + derivation.step(
            r"u(r,t)\approx\sqrt{\tfrac{2}{\pi kr}}\left(A\,%s%s+B\,%s%s\right)"
            % (cos_kr, cos_wt, sin_kr, sin_wt),
            map={J0: cos_kr, Y0: sin_kr, cos_wt: cos_wt, sin_wt: sin_wt},
            highlight=[J0, Y0], highlight_color="joker",
            begin_time=t0, transition_time=2)
        # only B = A makes the wave outgoing. Squeezed into the existing
        # timing: the note is written in the pause after the far-field step
        # and the former 2s collapse is split into two 1s steps
        outgoing = SimpleTexBObject(r"B=A", text_size="normal", color="joker",
                                    aligned="center",
                                    location=(note_x, 0, -0.3))
        outgoing.write(begin_time=t0 - 1, transition_time=1)
        # the general solution above turns into the outgoing one alongside
        derivation.replace_line(
            -2, r"u(r,t)=A\left(%s%s+%s%s\right)" % (J0, cos_wt, Y0, sin_wt),
            map={"B": None}, begin_time=t0, transition_time=1)
        t0 = derivation.step(
            r"u(r,t)\approx A\sqrt{\tfrac{2}{\pi kr}}\left(%s%s+%s%s\right)"
            % (cos_kr, cos_wt, sin_kr, sin_wt),
            mode="replace", map={"A": "A", "B": None},
            begin_time=t0, transition_time=1)
        # cos a cos b + sin a sin b = cos(a - b)
        t0 = 0.5 + derivation.step(
            r"u(r,t)\approx A\sqrt{\tfrac{2}{\pi kr}}\cos\left(kr-\omega t"
            r"-\tfrac{\pi}{4}\right)",
            mode="replace", begin_time=t0, transition_time=1)
        t0 = 1 + derivation.current.change_color(new_color="important",
                                                 begin_time=t0,
                                                 transition_time=1)

        self.t0 = t0

    def outlook(self):
        r"""
        The outlook of the script: the wave equation turns up wherever one
        looks, first in electromagnetism, then in gravity.

        1. Maxwell. The four source-free equations as a block on the left;
           on the right, the wave equation they combine into for E, and with
           it the speed ``c = 1 / sqrt(mu_0 eps_0)`` worked out from the two
           bench constants: 3 * 10^8 m/s, the speed of light.
        2. The d'Alembertian. The block clears, the E equation is brought to
           one side, the second derivatives are factored out and collected
           into a box: ``box E = 0``.
        3. Gravity. ``box E = 0`` steps down to the left and the general
           ``box u = 0`` takes the top. On the right, flat space plus a small
           perturbation in the vacuum equations, linearised in the harmonic
           gauge, gives ``box h_mn = 0`` next to it: light and gravitational
           waves, the same equation, the same speed.

        Signature (-,+,+,+), so ``box = -1/c^2 d_t^2 + Delta``.
        """
        t0 = 0
        _setup_render()
        # straight on, as for every sub-scene that is read rather than looked
        # at: 38 mm from 13 units back holds x in [-6.2, 6.2], z in [-3.5, 3.5]
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        E = r"\vec{E}"
        B = r"\vec{B}"
        me = r"\mu_0\varepsilon_0"
        h_mn = r"h_{\mu\nu}"
        d_tt = r"\frac{\partial^2}{\partial t^2}"

        # ---- Maxwell ---------------------------------------------------------
        # one text object per equation, hung from a common '=' column
        title = SimpleTexBObject(r"\text{Maxwell equations in vacuum}",
                                 text_size="normal", color="text",
                                 aligned="center", location=(-3.9, 0, 3.0))
        _MAXWELL_X = -5.9
        maxwell = [SimpleTexBObject(expression, text_size="normal",
                                    color="example",
                                    location=(_MAXWELL_X, 0, z))
                   for expression, z in zip(
                [r"\nabla\cdot %s=0" % E,
                 r"\nabla\cdot %s=0" % B,
                 r"\nabla\times %s=-\frac{\partial %s}{\partial t}" % (E, B),
                 r"\nabla\times %s=%s\frac{\partial %s}{\partial t}"
                 % (B, me, E)],
                [2.15, 1.35, 0.45, -0.55])]
        ampere = maxwell[-1]
        for row in maxwell[:-1]:
            row.align(ampere, char_index=row.find_letters("=")[0],
                      other_char_index=ampere.find_letters("=")[0])

        t0 = 0.3 + title.write(begin_time=t0, transition_time=1)
        for row in maxwell:
            t0 = 0.2 + row.write(begin_time=t0, transition_time=0.8)
        t0 += 0.5

        # what they combine into, on the right
        wave_E = BDerivation(
            r"\frac{1}{c^2}\frac{\partial^2 %s}{\partial t^2}=\Delta\,%s"
            % (E, E),
            location=(0.6, 0, 1.6), text_size="normal", color="text")
        t0 = 0.5 + wave_E.write(begin_time=t0, transition_time=1.5)

        # the Laplacian, spelled out in cartesian coordinates as a side note
        # under the equation that introduces it
        laplace = SimpleTexBObject(
            r"\Delta=\frac{\partial^2}{\partial x^2}"
            r"+\frac{\partial^2}{\partial y^2}+\frac{\partial^2}{\partial z^2}",
            text_size="small", color="joker", aligned="center",
            location=(2.3, 0, 0.55))
        wave_E.highlight(r"\Delta", color="joker", begin_time=t0,
                         transition_time=1.5)
        t0 = 0.5 + laplace.write(begin_time=t0, transition_time=1.5)

        speed = BDerivation(r"c=\frac{1}{\sqrt{%s}}" % me,
                            location=(1.6, 0, -0.5), text_size="normal",
                            color="text")
        t0 = 0.5 + speed.write(begin_time=t0, transition_time=1)

        # the two bench constants, each with its unit
        constants = SimpleTexBObject(
            r"\varepsilon_0=8.854\cdot 10^{-12}\,\tfrac{\text{As}}{\text{Vm}}"
            r"\qquad\mu_0=1.257\cdot 10^{-6}\,\tfrac{\text{Vs}}{\text{Am}}",
            text_size="small", color="example", aligned="center",
            location=(3.0, 0, -1.7))
        t0 = 0.5 + constants.write(begin_time=t0, transition_time=1.5)
        t0 = 0.5 + speed.step(
            r"c=2.998\cdot 10^{8}\,\tfrac{\text{m}}{\text{s}}",
            mode="replace", new_color="important",
            begin_time=t0, transition_time=2)
        t0 = 1 + speed.current.change_color(new_color="important",
                                            begin_time=t0, transition_time=1)

        # ---- the d'Alembertian -----------------------------------------------
        for text in maxwell + [title, constants, speed.current]:
            text.disappear(begin_time=t0, transition_time=1)
        t0 += 1
        laplace.move(Vector((-2.3, 0, -2.6)), begin_time=t0, transition_time=1)
        line = wave_E.current.ref_obj.location
        t0 = 0.5 + wave_E.move(Vector((-1.8 - line.x, 0, 1.2 - line.z)),
                               begin_time=t0, transition_time=1)

        # everything on one side ...
        t0 = 0.5 + wave_E.step(
            r"-\frac{1}{c^2}\frac{\partial^2 %s}{\partial t^2}+\Delta\,%s=0"
            % (E, E),
            mode="replace", begin_time=t0, transition_time=1.5)
        # ... the derivatives factored out ...
        t0 = 0.5 + wave_E.step(
            r"\left(-\frac{1}{c^2}%s+\Delta\right)%s=0" % (d_tt, E),
            mode="replace", begin_time=t0, transition_time=1.5)
        # ... and boxed up
        box = SimpleTexBObject(
            r"\Box=-\frac{1}{c^2}%s+\Delta" % d_tt,
            text_size="normal", color="joker", aligned="center",
            location=(0, 0, -1.0))
        t0 = 0.5 + box.write(begin_time=t0, transition_time=1.5)
        bracket = r"\left(-\frac{1}{c^2}%s+\Delta\right)" % d_tt
        wave_E.highlight(bracket, color="joker", begin_time=t0,
                         transition_time=1)
        t0 = 1 + wave_E.step(
            r"\Box\,%s=0" % E, mode="replace", map={bracket: r"\Box"},
            new_color="important", begin_time=t0 + 0.5, transition_time=1.5)
        # the pristine letters swapped in at the end carry the old colour
        wave_E.current.change_color(new_color="important", begin_time=t0,
                                    transition_time=0.5)
        laplace.disappear(begin_time=t0, transition_time=0.5)
        t0 = 0.5 + box.disappear(begin_time=t0, transition_time=0.5)

        # ---- gravity ---------------------------------------------------------
        # box E = 0 steps down to the left, the general form takes the top
        line = wave_E.current.ref_obj.location
        t0 = 0.3 + wave_E.move(Vector((-4.6 - line.x, 0, -0.9 - line.z)),
                               begin_time=t0, transition_time=1)
        wave_u = SimpleTexBObject(r"\Box\,u=0", text_size="large",
                                  color="important", aligned="center",
                                  location=(0, 0, 2.6))
        t0 = 0.5 + wave_u.write(begin_time=t0, transition_time=1)
        light = SimpleTexBObject(r"\text{light}", text_size="normal",
                                 color="example", aligned="center",
                                 location=(-3.6, 0, -1.9))
        t0 = 0.5 + light.write(begin_time=t0, transition_time=0.8)

        # on the right: flat space plus a ripple, in the vacuum equations
        metric = SimpleTexBObject(
            r"g_{\mu\nu}=\eta_{\mu\nu}+%s,\quad|%s|\ll 1" % (h_mn, h_mn),
            text_size="normal", color="example", aligned="center",
            location=(3.4, 0, 1.3))
        vacuum = SimpleTexBObject(r"R_{\mu\nu}=0", text_size="normal",
                                  color="example", aligned="center",
                                  location=(3.4, 0, 0.5))
        t0 = 0.3 + metric.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + vacuum.write(begin_time=t0, transition_time=0.8)

        # linearised, in harmonic gauge, the Ricci tensor is a box of h
        gauge = SimpleTexBObject(r"\text{linearised, harmonic gauge}",
                                 text_size="small", color="joker",
                                 aligned="center", location=(3.4, 0, -0.1))
        gravity = BDerivation(
            r"R_{\mu\nu}=-\tfrac{1}{2}\Box\,%s" % h_mn,
            location=(1.9, 0, -0.75), line_spacing=(0, 0, -0.85),
            text_size="normal", color="text")
        gauge.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + gravity.write(begin_time=t0, transition_time=1.5)

        # and the vacuum sets it to zero; only box h travels down - left
        # alone, the matcher also flies the R into the box
        highlight_letters(vacuum, list(range(len(vacuum.letters))),
                          begin_time=t0, transition_time=1)
        t0 = 0.5 + gravity.step(
            r"\Box\,%s=0" % h_mn,
            map={r"\Box\,%s" % h_mn: r"\Box\,%s" % h_mn}, auto=False,
            new_color="important", begin_time=t0 + 0.5, transition_time=1.5)
        gravity.current.change_color(new_color="important", begin_time=t0,
                                     transition_time=0.5)

        # clear the premises; box h = 0 settles level with box E = 0
        for text in [metric, vacuum, gauge] + gravity.lines[:-1]:
            text.disappear(begin_time=t0, transition_time=1)
        # level by the '=' signs: each glyph set is centred on its own
        # bounding box, so the '=' sits at a different height in each line
        def equals_z(tex):
            equals = tex.letters[tex.find_letters("=")[0]]
            return tex.ref_obj.location.z + equals.ref_obj.location.y

        h_x = gravity.current.ref_obj.location.x
        t0 = 0.5 + gravity.move(
            Vector((2.2 - h_x, 0,
                    equals_z(wave_E.current) - equals_z(gravity.current))),
            begin_time=t0, transition_time=1)

        waves = SimpleTexBObject(r"\text{gravitational waves}",
                                 text_size="normal", color="example",
                                 aligned="center", location=(3.6, 0, -1.9))
        t0 = 0.5 + waves.write(begin_time=t0, transition_time=1)
        detected = SimpleTexBObject(r"\text{measured since 2015}",
                                    text_size="small", color="joker",
                                    aligned="center", location=(3.6, 0, -2.45))
        t0 = 1 + detected.write(begin_time=t0, transition_time=1)

        self.t0 = t0

    def gravi_wave_overlay(self):
        r"""
        Gravitational waves, as an overlay for :meth:`outlook`: two spheres
        in orbit on a wire sheet that dents under them, with the two-armed
        spiral of the quadrupole wave running out from them. No text, and a
        transparent world, so it can be laid under or over the equations.

        All motion comes from ``Scene Time`` inside
        :class:`~geometry_nodes.modifier_video_interferences.GravitationalWavesModifier`;
        the only keyframes here fade the ripples in, so the shot opens on the
        static wells and the waves then start to run.
        """
        _setup_render()
        _lights()
        # low and oblique: the wells and the spiral are relief, which a view
        # from overhead flattens away
        _camera(location=(0, -11.5, 5.5), target=(0, 0, -0.4), lens=38)
        # transparent=True leaves out the Set Alpha that would make the frame
        # opaque. The halo then still lands where the world is transparent
        # and would be cut away, so it is given an alpha of its own: the
        # larger of the render's alpha and the brightness of the glare
        glare = create_glow_composition(threshold=1, type="BLOOM", size=6,
                                        transparent=True)
        # blender 5 measures the bloom size from 0 to 1; 1 is the widest
        glare.inputs["Size"].default_value = 1.0
        glare.inputs["Strength"].default_value = 2.5
        nodes = bpy.context.scene.compositing_node_group.nodes
        links = bpy.context.scene.compositing_node_group.links
        halo = nodes.new(type="CompositorNodeRGBToBW")
        alpha = nodes.new(type="ShaderNodeMath")
        alpha.operation = "MAXIMUM"
        alpha.use_clamp = True
        with_halo = nodes.new(type="CompositorNodeSetAlpha")
        with_halo.inputs["Type"].default_value = "Replace Alpha"
        links.new(glare.outputs["Glare"], halo.inputs["Image"])
        links.new(nodes["Render Layers"].outputs["Alpha"], alpha.inputs[0])
        links.new(halo.outputs["Val"], alpha.inputs[1])
        links.new(glare.outputs["Image"], with_halo.inputs["Image"])
        links.new(alpha.outputs["Value"], with_halo.inputs["Alpha"])
        for target in (nodes["Viewer"], nodes["Group Output"]):
            links.new(with_halo.outputs["Image"], target.inputs["Image"])

        waves = GravitationalWavesModifier(name="GravitationalWaves",
                                           size=7.6, lines=219, resolution=561,
                                           thickness=0.0025,
                                           period=2.0, separation=1.2,
                                           wave_speed=2.0, wave_amplitude=0.075,
                                           wire_color="drawing",
                                           sphere_color="important",
                                           sphere_emission=6)
        sheet = Plane(name="GravitationalWaves", u=[-1, 1], v=[-1, 1],
                      resolution=1)
        sheet.add_mesh_modifier(type='NODES', node_modifier=waves)
        sheet.appear(begin_time=0, transition_time=0)

        amplitude = ibpy.get_geometry_node_from_modifier(waves, "WaveAmplitude")
        ibpy.change_default_value(amplitude, from_value=0, to_value=0.075,
                                  begin_time=0.5, transition_time=3)
        self.t0 = 20

    def light_wave_overlay(self):
        r"""
        Light, as an overlay for :meth:`outlook`: a linearly polarised plane
        wave, emitted from the left. E (red) swings along z, B (sky
        blue) along y, both across the direction of travel x and in phase.
        No text and a transparent world, like :meth:`gravi_wave_overlay`.

        All motion comes from ``Scene Time`` inside
        :class:`~geometry_nodes.modifier_video_interferences.ElectromagneticWaveModifier`:
        the front leaves the left end at t = 0.5 s and crosses the axis at
        c = lambda f = 1.5 units per second, after which the wave runs on.
        """
        _setup_render()
        _lights()
        # from the front left and well above, so that the vertical E and the
        # horizontal B both open up, and the wave runs away to the right
        _camera(location=(-4, -12, 6), target=(0, 0, 0), lens=27)

        # transparent=True leaves out the Set Alpha that would make the frame
        # opaque. The halo then still lands where the world is transparent
        # and would be cut away, so it is given an alpha of its own: the
        # larger of the render's alpha and the brightness of the glare
        glare = create_glow_composition(threshold=1, type="BLOOM", size=6,
                                        transparent=True)
        # blender 5 measures the bloom size from 0 to 1; 1 is the widest
        glare.inputs["Size"].default_value = 1.0
        glare.inputs["Strength"].default_value = 1.5
        nodes = bpy.context.scene.compositing_node_group.nodes
        links = bpy.context.scene.compositing_node_group.links
        halo = nodes.new(type="CompositorNodeRGBToBW")
        alpha = nodes.new(type="ShaderNodeMath")
        alpha.operation = "MAXIMUM"
        alpha.use_clamp = True
        with_halo = nodes.new(type="CompositorNodeSetAlpha")
        with_halo.inputs["Type"].default_value = "Replace Alpha"
        links.new(glare.outputs["Glare"], halo.inputs["Image"])
        links.new(nodes["Render Layers"].outputs["Alpha"], alpha.inputs[0])
        links.new(halo.outputs["Val"], alpha.inputs[1])
        links.new(glare.outputs["Image"], with_halo.inputs["Image"])
        links.new(alpha.outputs["Value"], with_halo.inputs["Alpha"])
        for target in (nodes["Viewer"], nodes["Group Output"]):
            links.new(with_halo.outputs["Image"], target.inputs["Image"])

        light = ElectromagneticWaveModifier(name="ElectromagneticWave",
                                            length=12, arrows=49,
                                            wavelength=3.0, frequency=0.5,
                                            amplitude=1.5, b_ratio=0.75,
                                            front=True, impact=0.5)
        wave = Plane(name="LightWave", u=[-1, 1], v=[-1, 1], resolution=1)
        wave.add_mesh_modifier(type='NODES', node_modifier=light)
        wave.appear(begin_time=0, transition_time=0)
        self.t0 = 20

    def two_sources(self):
        """The answer to the question: points that follow an interference pattern.

        Two point sources on the x axis, f = cos^2(k dr / 2) with dr the
        difference of the distances to them - so the bright fringes are the
        surfaces of constant path difference, which for two points are
        hyperboloids of revolution about the line joining them. The cloud is a
        stack of nested shells, and turning it shows they really are surfaces
        rather than a pattern painted on one plane.

        Then the two dials that make it move:

        - ``Phase1`` ramped by 2*pi marches every fringe one full spacing
          along the axis, which is what a moving interference pattern is;
        - ``Source0``/``Source1`` pulled apart tightens the fringes, because
          the path difference then changes faster with position.

        Both redraw the cloud each frame from the moved density function: the
        points shimmer rather than flow. Points that *flow* would have to be
        displaced by the field rather than drawn from it.
        """
        t0 = 0
        _setup_render()
        _lights()
        _camera(location=(7.5, -10.5, 5.0))

        separation = 1.2
        fringes = InterferenceModifier(name="TwoSources", size=4, count=26000,
                                       sources=((-separation, 0, 0),
                                                (separation, 0, 0)),
                                       wavelength=0.8, radius=0.02,
                                       color_by_density=True, emission=0.6,
                                       box_color="text", box_radius=0.012)
        host = _cloud(fringes, name="TwoSources")

        title = _title("two sources: $f = |e^{ikr_1} + e^{ikr_2}|^2/4$")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        # a turn first: the fringes are surfaces, and one still frame of a
        # point cloud is the one view that cannot say so
        t0 = 0.5 + _turntable(host, begin_time=t0, transition_time=10)

        # march the pattern: one full 2*pi of phase, at a constant rate
        phase = ibpy.get_geometry_node_from_modifier(fringes, "Phase1")
        t0 = 0.5 + ibpy.change_default_value(phase, from_value=0, to_value=tau,
                                             begin_time=t0, transition_time=8)

        # then pull the sources apart, which tightens the fringes
        left = ibpy.get_geometry_node_from_modifier(fringes, "Source0")
        right = ibpy.get_geometry_node_from_modifier(fringes, "Source1")
        ibpy.change_default_vector(left, from_value=Vector((-separation, 0, 0)),
                                   to_value=Vector((-2 * separation, 0, 0)),
                                   begin_time=t0, transition_time=8)
        t0 = ibpy.change_default_vector(right, from_value=Vector((separation, 0, 0)),
                                        to_value=Vector((2 * separation, 0, 0)),
                                        begin_time=t0, transition_time=8)
        _linearize(fringes)
        self.t0 = t0

    # ------------------------------------------------------------------
    def two_sources_real(self):
        r"""The same two sources, but the field instead of its average.

        :meth:`two_sources` draws its points from the time-averaged intensity,
        so its fringes are the pattern a long exposure records: standing,
        still, and moving only when a scene ramps the phase by hand. This one
        keeps the wave, f = (sum_j a_j/r_j sin(k r_j - wt))^2 - the energy in
        the field at one instant - and the shells travel outward on their own,
        because ``wt`` comes from ``Scene Time``. Not one keyframe in the tree.

        Three things about the picture, all of them consequences of that:

        The points do not move and are not culled. f here is unbounded (1/r
        at each source) and vanishes everywhere twice a period, so it cannot
        be a sampling density - a sampler fed it would empty the box on the
        zero crossings. The cloud is 205344 fixed probes and f is *brightness*
        instead, which has no top end to overflow.

        Hence the bloom, and hence the tiny points. A radius of 0.001 is well
        under a pixel at this distance, so the cloud does not render as
        geometry at all - each point is a sub-pixel emitter, and what reaches
        the frame is the glare node's bloom around it. Threshold 0.01 is low
        enough that the dim shells out at the box wall still glow rather than
        clipping to black, which is where the interference is most legible.

        And the beat in the middle: taking ``Amplitude1`` to zero and back
        leaves one source radiating alone. The concentric shells that appear
        are the thing being interfered - it is the cleanest statement of what
        the second source contributes, and it costs nothing but one dial.
        """
        t0 = 0
        _setup_render()
        _lights()
        _camera(location=(7.5, -10.5, 5.0))
        # unclamped emission is what the glare node's threshold is for; a low
        # threshold keeps the faint outer shells alight instead of crushing
        # them, since nothing else in this shot is bright enough to bloom
        create_glow_composition(threshold=0.01, type="BLOOM", size=1)

        separation = 1.2
        # the count is the one from tmp.xml; with method="uniform" it is the
        # point count outright, not a candidate count to be culled down
        field = RealInterferenceModifier(name="TwoSourcesReal", size=4,
                                         count=205344,
                                         sources=((-separation, 0, 0),
                                                  (separation, 0, 0)),
                                         wave_number=10, frequency=10,
                                         radius=0.001, color="yellow",
                                         emission=10, box_color="text",
                                         box_radius=0.012)
        host = _cloud(field, name="TwoSourcesReal")

        title = _title(r"the field itself: "
                       r"$f=\left(\sum_j \frac{a_j}{r_j}\sin(kr_j-\omega t)\right)^2$")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        # a turn, during which the wave travels of its own accord: the shells
        # move outward at omega/k = 1 unit per second while the camera goes
        # round, and the two motions together say the shells are surfaces
        t0 = 0.5 + _turntable(host, begin_time=t0, transition_time=10)

        # then silence one source and bring it back
        second = ibpy.get_geometry_node_from_modifier(field, "Amplitude1")
        t0 = 0.5 + ibpy.change_default_value(second, from_value=1, to_value=0,
                                             begin_time=t0, transition_time=4)
        t0 = ibpy.change_default_value(second, from_value=0, to_value=1,
                                       begin_time=t0, transition_time=4)
        self.t0 = t0 + 2

    # ------------------------------------------------------------------
    def interference_2d(self):
        r"""The same physics one dimension down, and done entirely in a shader.

        Five emitters in a line, each radiating a circular wave
        A/sqrt(r) sin(2 pi r/lambda - 2 pi f t), summed and painted on a
        single plane by :func:`~appearance.textures.interference_texture` -
        the port of ``Material.002`` in ``trails/2D.blend``. No points, no
        geometry nodes, no modifier: one quad and a texture.

        Which is the whole reason this scene sits next to the others. The 3D
        scenes have to *build* the field out of a quarter of a million probes
        because a volume has no surface to paint; in 2D the field lives
        exactly where the pixels are, so the same picture costs one plane and
        renders in milliseconds. The difference between the two files is not
        the physics, it is that a shader can only paint what a surface
        already covers.

        Two details carried over from the blend rather than invented here.
        The colour ramp is centred on the *undisturbed* surface - the factor
        is (sum + 1)/2, so black is zero, and crest and trough take opposite
        hues - which is what makes the standing pattern read as a wave rather
        than as a brightness map. And the alpha is the energy, sum**2, so the
        nodal lines are cut clean out of the plane; against the dark
        background they read as the black hyperbolae of the classic
        double-slit figure, only there are five slits here rather than two.

        The model is ``"hankel"`` rather than the blend's ``A/sqrt(r)
        sin(kr - wt)``, i.e. each source radiates the actual solution of the
        2D wave equation, ``J0(kr) cos wt + Y0(kr) sin wt``, evaluated in the
        shader by a polynomial approximation. Far from a source the two are
        the same wave up to a constant phase - the elementary form *is* the
        asymptotics of the Bessel one - so what it buys is the first
        wavelength or so around each emitter, where the true wavefronts are
        pulled inwards from where a constant-wavelength model puts them: the
        first ring lands at 0.38 lambda rather than 0.50.

        Which is why the emitters no longer sit on the u = 0 edge as they do
        in the blend. There they were half out of frame, and deliberately so
        - the far-field envelope diverges as r^-1/2 at each of them, and an
        emitter in the middle of the plane was a white-hot dot that the eye
        went to instead of the fringes. The Bessel model only diverges
        logarithmically, and it is held off the pole by ``source_radius``
        besides, so a source can be looked at directly. Moved inwards they
        become the subject: five distinct radiators whose circular
        wavefronts are visibly curved and unevenly spaced before they merge
        into the familiar straight-fringed pattern downstream.
        ``model="farfield"`` puts the blend's formula back.
        """
        t0 = 0
        _setup_render()
        _lights()
        # 11 units back rather than the 9 that frames a 4-unit plane tightly:
        # at 38 mm this holds z in [-2.9, 2.9], which is the room the title
        # needs above the plane's own [-2, 2]
        _camera(location=(0, -11, 0), lens=38)

        # standing in the xz plane so the camera meets it face on: a texture
        # seen at an angle is a texture whose fringe spacing is a lie
        # the emitters brought in off the u = 0 edge to a column at u = 0.3,
        # short of centre so that most of the plane is still downstream of
        # them. At lambda = 0.16 they sit 1.19 wavelengths apart and the
        # near field - within one wavelength of some source - covers 30% of
        # the frame, which is what makes it the subject rather than a
        # detail. The amplitude is what puts the 98th percentile of the
        # field at 1.0, the top of the colour ramp: any more and the
        # fringes flatten into solid red and green.
        sources = [(0.3, v) for v in (0.12, 0.31, 0.5, 0.69, 0.88)]
        screen = Plane(u=[-2, 2], v=[-2, 2], resolution=1,
                       name="Interference2D", color="interference",
                       model="hankel", sources=sources, wavelength=0.16,
                       amplitude=0.125, rotation_euler=[pi / 2, 0, 0])
        screen.appear(begin_time=0, transition_time=0)
        material = ibpy.get_material_of(screen)

        title = _title(r"five sources: $f=\left(\sum_j J_0(kr_j)\cos\omega t"
                       r"+Y_0(kr_j)\sin\omega t\right)^2$",
                       location=(0, 0, 2.3), size="small")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        # the shader editor has no Scene Time node, so the clock is a plain
        # Value that the scene ramps by hand - linearly, or the wavefronts
        # would ease to a halt at the end of the shot
        clock = ibpy.get_node_from_shader(material, "Time")
        t0 = ibpy.change_default_value(clock, from_value=0, to_value=2,
                                       begin_time=t0, transition_time=16)
        _linearize(material.node_tree)
        self.t0 = t0 + 2

    # ------------------------------------------------------------------
    def wave_visualization(self):
        r"""The field as a surface: a grid standing up as :math:`H^{(1)}_0`.

        :meth:`interference_2d` paints the wave on a flat plane, which is a
        picture of the field. This one *is* the field:
        :class:`~geometry_nodes.modifier_video_interferences.WaveVisualizationModifier`
        lifts every vertex of a 301 x 301 grid to

        .. math::
            u(r, t) = A'\big[J_0(kr)\cos\omega t + Y_0(kr)\sin\omega t\big],

        the exact outgoing solution of the 2+1 wave equation, evaluated with
        the ``j0``/``y0`` node groups. So the crests are hills, the nodal
        circles are the flat rings between them, and the logarithmic pole at
        the source is a chimney rather than a bright dot - the one feature of
        the true solution that the far-field :math:`\sin(kr-\omega t)/\sqrt r`
        gets qualitatively wrong, and it is standing right in the middle of
        the shot.

        The colour is the ``"interference"`` texture in its ``"hankel"``
        model: the same sum, recomputed in the shader from the surface's own
        uv, so red and green are crest and trough and the alpha (which is
        :math:`u^2`) cuts the nodal rings out of the mesh. The modifier hands
        the texture its own wavelength, frequency, amplitude, source radius
        and ``uv_scale``, so the painted rings sit on the geometric ones
        without either being typed twice.

        Two things this scene has to do by hand, both of them consequences of
        the field living in two trees at once:

        The clock. Geometry nodes read ``Scene Time -> Seconds``; a shader
        tree has no such node, so the material's ``Time`` is ramped from 0 to
        the length of the shot, in seconds, starting at t = 0. That makes the
        two clocks the same clock. Start the ramp anywhere else and the colour
        runs a fixed phase behind the relief - which looks like a rendering
        artefact and is in fact two different times.

        And the wavelength sweep, which has to move *both* ``Wavelength``
        dials in lockstep for the same reason. Watching it, the rings tighten
        and the surface simultaneously rises, because the wave is normalised
        to a fixed far-field amplitude :math:`A/\sqrt r`: shorter waves need a
        taller :math:`A' = A\pi/\sqrt\lambda` to get there. That is physics
        the flat scenes cannot show, since a colour ramp has no height.
        """
        t0 = 0
        duration = 24
        _setup_render()
        _lights()
        # low and close: the whole point is the relief, and a surface seen
        # from overhead is a flat picture with extra steps
        _camera(location=(0, -11.5, 6.2), target=(0, 0, 0.2), lens=38)

        wavelength = 0.8
        wave = WaveVisualizationModifier(name="WaveVisualization", size=8.0,
                                         resolution=301, sources=((0, 0),),
                                         wavelength=wavelength, frequency=1.0,
                                         amplitude=0.6,
                                         source_radius=wavelength / 10,
                                         emission_strength=0.8)
        surface = Plane(name="WaveSurface", u=[-1, 1], v=[-1, 1], resolution=1)
        surface.add_mesh_modifier(type='NODES', node_modifier=wave)
        surface.appear(begin_time=0, transition_time=0)

        title = _title(r"$u(r,t)=J_0(kr)\cos\omega t+Y_0(kr)\sin\omega t$",
                       location=(0, 0, 2.6), size="small")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        # the shader's clock, ramped to *be* scene time: from 0 at t = 0 to
        # `duration` seconds at the end of the shot
        clock = ibpy.get_node_from_shader(wave.material, "Time")
        ibpy.change_default_value(clock, from_value=0, to_value=duration,
                                  begin_time=0, transition_time=duration)

        # then the wavelength, in both trees at once. Halving it doubles the
        # number of rings on the grid and lifts the surface by sqrt(2)
        tree_dial = ibpy.get_geometry_node_from_modifier(wave, "Wavelength")
        shader_dial = ibpy.get_node_from_shader(wave.material, "Wavelength")
        for dial in (tree_dial, shader_dial):
            ibpy.change_default_value(dial, from_value=wavelength,
                                      to_value=wavelength * 0.625,
                                      begin_time=t0 + 6, transition_time=8)

        _linearize(wave)
        _linearize(wave.material.node_tree)
        self.t0 = duration


    # ------------------------------------------------------------------
    def line_array(self):
        r"""Twenty emitters along the bottom edge, and the wavelength swept.

        The plane is the whole frame here - 3.84 x 2.16 blender units at
        1920 x 1080 - so there is no title and no background, only the field.
        A surface that is not square needs :func:`interference_texture`'s
        ``uv_scale``, because uv runs 0..1 in both directions whatever shape
        the surface is and a circular wave would otherwise come out
        elliptical, stretched 16:9. Passing the world extent puts the metric
        back and makes ``wavelength`` a length in blender units.

        What the sweep is for. Twenty sources in a row a distance d apart
        are a diffraction grating, and the one number that decides what a
        grating does is d/lambda. Sweeping the wavelength from 0.45 down to
        0.10 takes it from 0.40 to 1.82 and walks through the whole
        behaviour:

        d < lambda
            no grating lobes exist at all. The row radiates as a single line
            source and the field is a stack of near-flat wavefronts marching
            up the screen - twenty emitters that read as one.
        d = lambda
            the first order arrives, and it arrives *along the row*: every
            source is in phase with its neighbour in that direction, so the
            strip containing them flares into a picket fence while the rest
            of the frame goes dark. This is the loud moment of the sweep and
            the only place the colour ramp clips - 11% of the frame at its
            worst against 2% elsewhere.
        d > lambda
            the orders lift off the edge and fan out. Two beams become four
            become six, crossing into the lattice of spots that fills the
            last third of the sweep.

        The sweep is linear in k rather than in lambda, because d/lambda is
        what the picture responds to and a linear ramp of the wavelength
        would spend most of its time in the featureless part. Both that and
        the exposure go in through :func:`_ramp_through`; the numbers below
        were computed offline rather than in here, since finding them means
        evaluating the field over the frame at sixteen phases for each of
        twenty-five wavelengths.

        **The prediction, drawn over it.** A cube at the centre of the row
        carries a modifier called "far field approximation"
        (:class:`~geometry_nodes.modifier_video_interferences.FarFieldModifier`)
        which draws the grating maxima, sin(alpha_n) = n lambda / d, as rays.
        Its ``Wavelength`` dial is keyframed from the *same* list as the
        material's, over the same twenty seconds, so the angles and the
        pattern cannot come apart. With d = 0.182 and the sweep stopping at
        lambda = 0.1, only n = 0 and n = +-1 are ever radiated: the first
        order needs lambda <= d, and the second would need d/lambda >= 2
        against the 1.82 this reaches. So the shot opens with a single ray
        up the middle, and at lambda = d the other two appear lying flat
        along the row and swing up to +-33.4 degrees by the end.

        What the rays mark is worth being exact about, because it is not
        what "maximum" suggests. The array is 3.46 units long and the frame
        is two units tall, so everything on screen is inside the array's own
        near field - its far field starts about 180 units away - and there
        are no clean beams here to sit on top of. alpha_n is nonetheless a
        real feature of the picture and not a line drawn over it: it is the
        edge of the region where the n-th order overlaps the zeroth, and the
        fringe texture changes across it. Measured on the rendered frames,
        binning the field into 2.5-degree sectors about the array centre and
        looking for the sharpest step in fringe rate along the arc, the step
        lands at +-33.8 degrees against a predicted +-33.4 (lambda = 0.1),
        at +-43.8 against +-43.0 (lambda = 0.124), and at +-58.8 against
        +-57.8 (lambda = 0.154) - inside one bin every time, with the other
        step of the pair at 0 degrees, on the zeroth order.
        """
        t0 = 0
        _setup_render()
        _lights()
        # 38 mm sees 2 x 18/38 x D across; at D = 3.97 that is 3.76 units
        # against the plane's 3.84, so the plane overfills the frame by 2%
        # and no edge can drift into shot
        _camera(location=(0, -3.97, 0), lens=38)

        # a row along the bottom, just far enough in (v = 0.06, a seventh of
        # a unit) that the emitters themselves are on screen rather than
        # bisected by the frame edge
        extent = (3.84, 2.16)
        row_v = 0.06
        us = np.linspace(0.5 - 0.5 / 6, 0.5 + 0.5 / 6, 6)
        sources = [(u, row_v) for u in us]
        screen = Plane(u=[-1.92, 1.92], v=[-1.08, 1.08], resolution=1,
                       name="LineArray", color="interference", model="hankel",
                       sources=sources, uv_scale=extent,
                       wavelength=0.45, amplitude=0.2331, source_radius=0.012,
                       frequency=1.0, rotation_euler=[pi / 2, 0, 0])
        screen.appear(begin_time=0, transition_time=0)
        material = ibpy.get_material_of(screen)

        # lambda = 1/k on a linear k ramp from 1/0.45 to 1/0.10
        wavelengths = [0.4500, 0.3927, 0.3484, 0.3130, 0.2842, 0.2602,
                       0.2400, 0.2227, 0.2077, 0.1946, 0.1831, 0.1728,
                       0.1636, 0.1554, 0.1479, 0.1412, 0.1350, 0.1293,
                       0.1241, 0.1193, 0.1149, 0.1108, 0.1069, 0.1033,
                       0.1000]
        wavelengths = [l / 2 for l in wavelengths]
        # and the exposure that holds 2% of the frame at the top of the
        # colour ramp at each of them - the amplitude that reads well at
        # lambda = 0.45 is five times too much at the resonance. Floored at
        # 0.11 and smoothed over three taps, because the raw curve dives by
        # 5.5x across two steps and that reads as the picture failing rather
        # than as the array resonating; the floor is what lets the flare
        # through.
        amplitudes = [1 / len(sources)] * 25

        # the clock runs the whole shot at a constant rate - 30 periods in
        # 24 s, a little over one a second, which is slow enough that a
        # fringe moves about a pixel a frame at the short-wavelength end
        ibpy.change_default_value(
            ibpy.get_node_from_shader(material, "Time"),
            from_value=0, to_value=30, begin_time=0, transition_time=24)

        # two seconds at each end to read the extremes before and after
        _ramp_through(ibpy.get_node_from_shader(material, "Wavelength"),
                      wavelengths, begin_time=2, transition_time=20)
        _ramp_through(ibpy.get_node_from_shader(material, "Amplitude"),
                      amplitudes, begin_time=2, transition_time=20)
        _linearize(material.node_tree)

        # ---- the prediction, over the top of the thing it predicts -------
        # the plane is stood up by rotation_euler=[pi/2, 0, 0], which sends
        # local (x, y, 0) to world (x, 0, y), so uv (u, v) is at
        # x = (u - 1/2) sx, z = (v - 1/2) sy. Both the array centre and the
        # source spacing therefore come out of the same numbers the sources
        # were built from and cannot drift away from them.
        center = Vector(((us.mean() - 0.5) * extent[0], -0.02,
                         (row_v - 0.5) * extent[1]))
        spacing = (us[1] - us[0]) * extent[0]
        # an order needs |n| lambda <= g to exist at all, so the sweep's
        # shortest wavelength says how many there will ever be: g/0.1 = 1.82,
        # and the second order would want g/lambda >= 2
        rays = FarFieldModifier(name="FarFieldApproximation", spacing=spacing,
                                wavelength=wavelengths[0],
                                max_order=int(spacing / min(wavelengths)),
                                reach=2.6, radius=0.008, color="text",
                                emission=1.0)
        # y = -0.02 puts the tubes in front of the plane rather than through
        # it; the camera is 3.97 back, so they are drawn 0.5% too large and
        # nothing else changes
        host = Cube(name="FarFieldHost", location=center)
        host.add_mesh_modifier(type='NODES', node_modifier=rays,
                               name="far field approximation")
        host.appear(begin_time=0, transition_time=0)

        # the same list of wavelengths, keyframed a second time onto the
        # tree's own dial. Not a copy of the animation - the *same* numbers
        # over the same twenty seconds, so alpha_n and the pattern cannot
        # come apart no matter what either one is edited to later
        _ramp_through(ibpy.get_geometry_node_from_modifier(rays, "Wavelength"), wavelengths,
                      begin_time=2, transition_time=20)
        _linearize(rays)
        print("far field: g = %.4f, orders %s, alpha at lambda = %.3f: %s"
              % (spacing, list(rays.angles(wavelengths[-1])),
                 wavelengths[-1],
                 ["%.1f deg" % np.degrees(a)
                  for a in rays.angles(wavelengths[-1]).values()]))
        self.t0 = t0 + 24

    # ------------------------------------------------------------------
    def line_array_rgb(self):
        r""":meth:`line_array` in white light: red, green and blue at once.

        The same six emitters, the same frame and the same sweep, but the
        plane wears :func:`~appearance.textures.rgb_interference_texture`
        (``video_interferences/shader.xml``): three interference patterns,
        one per colour, whose wavelengths are the ``Wavelength`` dial times
        1, 0.9 and 0.8. A grating's orders sit at sin(alpha_n) = n lambda/g,
        so the shorter the wavelength the steeper the beam - blue leaves
        the row closest to the normal, red furthest from it, and each order
        is a little spectrum. The zeroth order is the same for every colour,
        which is why it stays white - and why only the red host draws it,
        in white.

        The prediction is drawn once per colour: three
        :class:`~geometry_nodes.modifier_video_interferences.FarFieldModifier`
        hosts, each with its colour's ``ratio``, each keyframed with the
        material's list of wavelengths. The ratio lives in the tree as a
        dial rather than in the keyframes, so all four animations are
        literally the same numbers, as in :meth:`line_array`.

        Sweeping the red wavelength down to 0.05 takes blue to 0.04; with
        g = 0.128 that is g/lambda = 3.2 for blue against 2.56 for red, so
        blue reaches its third order and red stops at the second.
        """
        t0 = 0
        _setup_render()
        _lights()
        _camera(location=(0, -3.97, 0), lens=38)

        extent = (3.84, 2.16)
        row_v = 0.06
        us = np.linspace(0.5 - 0.5 / 6, 0.5 + 0.5 / 6, 6)
        sources = [(u, row_v) for u in us]
        # red, green, blue - the blend's multipliers on the one Wavelength
        ratios = (1.0, 0.9, 0.8)
        screen = Plane(u=[-1.92, 1.92], v=[-1.08, 1.08], resolution=1,
                       name="LineArrayRGB", color="rgb_interference",
                       model="hankel", sources=sources, uv_scale=extent,
                       ratios=ratios, wavelength=0.45,
                       amplitude=1 / len(sources), source_radius=0.012,
                       frequency=1.0, rotation_euler=[pi / 2, 0, 0])
        screen.appear(begin_time=0, transition_time=0)
        material = ibpy.get_material_of(screen)

        # the sweep of line_array, lambda = 1/k on a linear k ramp, halved
        wavelengths = [0.4500, 0.3927, 0.3484, 0.3130, 0.2842, 0.2602,
                       0.2400, 0.2227, 0.2077, 0.1946, 0.1831, 0.1728,
                       0.1636, 0.1554, 0.1479, 0.1412, 0.1350, 0.1293,
                       0.1241, 0.1193, 0.1149, 0.1108, 0.1069, 0.1033,
                       0.1000]
        wavelengths = [l / 2 for l in wavelengths]

        ibpy.change_default_value(
            ibpy.get_node_from_shader(material, "Time"),
            from_value=0, to_value=30, begin_time=0, transition_time=24)
        _ramp_through(ibpy.get_node_from_shader(material, "Wavelength"),
                      wavelengths, begin_time=2, transition_time=20)
        _linearize(material.node_tree)

        # ---- one prediction per colour ----------------------------------
        center = Vector(((us.mean() - 0.5) * extent[0], 0,
                         (row_v - 0.5) * extent[1]))
        spacing = (us[1] - us[0]) * extent[0]
        for i, (color, ratio) in enumerate(zip(("red", "green", "blue"),
                                               ratios)):
            rays = FarFieldModifier(name="FarField" + color.capitalize(),
                                    spacing=spacing, wavelength=wavelengths[0],
                                    ratio=ratio,
                                    max_order=int(spacing / (ratio * min(wavelengths))),
                                    reach=2.6, radius=0.008, color=color,
                                    zeroth_color="text", draw_zeroth=i == 0,
                                    emission=1.0)
            # the n = 0 ray is the same for every colour - white, where all
            # three agree - so only the first host draws it; three copies
            # in one place would z-fight
            host = Cube(name="FarFieldHost" + color.capitalize(),
                        location=center + Vector((0, -0.02, 0)))
            host.add_mesh_modifier(type='NODES', node_modifier=rays,
                                   name="far field approximation " + color)
            host.appear(begin_time=0, transition_time=0)
            _ramp_through(ibpy.get_geometry_node_from_modifier(rays, "Wavelength"), wavelengths,
                          begin_time=2, transition_time=20)
            _linearize(rays)
            print("far field %s: g = %.4f, lambda = %.3f: %s"
                  % (color, spacing, ratio * wavelengths[-1],
                     ["%.1f deg" % np.degrees(a)
                      for a in rays.angles(wavelengths[-1]).values()]))
        self.t0 = t0 + 24

    # ------------------------------------------------------------------
    def airy_disc(self):
        r"""A round hole cannot make a point: the Airy disc.

        Everything before this scene interferes waves that came from
        *separate* sources. This one interferes a single wavefront with
        itself: a circular aperture of radius a, lit by light of wavelength
        lambda, sends onto a screen a distance L behind it

        .. math::
            I(\rho) = I_0\left(\frac{2J_1(v)}{v}\right)^{2},
            \qquad v = \frac{2\pi a}{\lambda}\sin\theta ,

        which is :class:`~geometry_nodes.modifier_airy.AiryDiscModifier` on a
        polar mesh: one colour, the minima black, and the same :math:`J_1`
        the drum uses, reached through the ``j1`` operator, so the picture
        carries no table of values.

        Three things the shot is built to show.

        **The rings are not decoration, they are the resolution.** The first
        dark ring sits at :math:`\rho = 1.22\lambda L/D`, and two point
        sources closer together than that on the sky arrive as one blob. That
        is the diffraction limit of every telescope and every lens, and it is
        this radius.

        **The halo is faint because it is faint.** The first three rings
        carry 1.75%, 0.42% and 0.16% of the central brightness, which on
        screen is a sixth, a twelfth and a twentieth of the core. The shot
        opens at ``Gamma = 2.2``, where the disc shows the intensity itself -
        blender's view transform brings its own 1/2.2 - and there is nothing
        around it at all. Ramping to 1 opens the shutter far enough for the
        halo and no further. Nothing about the field changes while that
        happens, only the map from intensity to brightness.

        **The pattern scales as lambda/a and nothing else.** Doubling the
        aperture halves every radius, and lengthening the wavelength grows
        them in the same proportion - the two ramps at the end, which are the
        statement that a bigger telescope and a bluer colour buy exactly the
        same thing.
        """
        t0 = 0
        _setup_render()
        # straight on, and no lights: the ramp material emits, so the disc is
        # its own illumination. A key light would put a diffuse grey into the
        # dark rings, and the dark rings are what makes the halo read
        _camera(location=(0, -11, 0), lens=38)
        create_glow_composition(threshold=0.75, type="BLOOM", size=4)

        # radius 2 fills the frame, and the fifth dark ring put on the rim
        # leaves four rings of halo inside it and a core a quarter as wide.
        # `edge="dark"` rather than the default, because this one is looked
        # at on its own and a rim that is a zero of the field has no edge.
        # 400 rings of mesh against 512 spokes is what samples the outermost
        # of them without aliasing
        airy = AiryDiscModifier(name="AiryDisc", radius=2, rings=5,
                                edge="dark", radial=400, angular=512,
                                gamma=2.2, color="important")
        disc = _panel(airy, name="AiryPanel", rotation_euler=[pi / 2, 0, 0])
        disc.appear(begin_time=0, transition_time=0)
        print("airy disc: aperture %.4f, first dark ring at rho = %.4f"
              % (airy.aperture, airy.first_zero()))

        title = _title(r"a circular aperture: "
                       r"$I=I_0\left(\frac{2J_1(v)}{v}\right)^2$, "
                       r"$v=\frac{2\pi a}{\lambda}\sin\theta$",
                       location=(0, 0, 2.5), size="small")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        # a hold on the intensity itself, which is a bare disc: this is what
        # the eye is given before the halo is made visible
        t0 += 1.5

        # the exposure, and the only beat of the shot in which the field
        # does not change at all
        t0 = 0.5 + ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(airy, "Gamma"),
                                             from_value=2.2, to_value=1.0,
                                             begin_time=t0, transition_time=6)

        # the aperture: twice as wide a hole, half as wide a core, twice as
        # many rings in the same disc
        opening = airy.aperture
        aperture = ibpy.get_geometry_node_from_modifier(airy, "Aperture")
        t0 = ibpy.change_default_value(aperture, from_value=opening,
                                       to_value=2 * opening, begin_time=t0,
                                       transition_time=5)
        t0 = 0.5 + ibpy.change_default_value(aperture, from_value=2 * opening,
                                             to_value=opening, begin_time=t0,
                                             transition_time=5)

        # and the wavelength, the other half of lambda/a: red light through
        # the same hole spreads twice as far as blue
        wavelength = ibpy.get_geometry_node_from_modifier(airy, "Wavelength")
        t0 = ibpy.change_default_value(wavelength, from_value=0.02,
                                       to_value=0.04, begin_time=t0,
                                       transition_time=5)
        t0 = ibpy.change_default_value(wavelength, from_value=0.04,
                                       to_value=0.02, begin_time=t0,
                                       transition_time=4)
        self.t0 = t0 + 1

    # ------------------------------------------------------------------
    def airy_logo(self):
        r"""The logo, with an Airy disc where every sphere used to be.

        This is the reason the modifier draws a disc of radius one rather
        than a square plate of whatever size the shot wants.
        :class:`~objects.logo.LogoFromInstances` lays its instances out on
        the parabolas of the logo and hands each one a scale of 1/den, so
        whatever it instances has to be the size a unit sphere is and has to
        be round. An Airy disc is both, and unlike a sphere it does not need
        a light: the ramp emits, so each instance carries its own glow and
        the row of them reads as a row of point sources seen through the same
        aperture - which is what a telescope's field of stars actually looks
        like.

        Three modifiers, one per colour of the logo, and every instance of a
        colour shares its tree. Nothing is animated per instance, so there is
        nothing to be gained by giving each its own copy, and a shared node
        group is a shared evaluation - thirty-odd discs cost three.

        The instances are plain :class:`~objects.bobject.BObject` carrying a
        single vertex. The tree throws away the geometry it is hung on, so
        the mesh is only somewhere for the modifier to live and something for
        the logo's transform to move.

        Each disc carries its own colour - red, green and blue, the three
        the logo is named for - and is cut off at the *peak* of its third
        ring, so an instance is a core and two rings, the outer of them
        brightest exactly at the boundary. The logo sets its circles tangent,
        so that is what makes neighbours touch ring to ring rather than
        leaving a dark gap between them. The outer ring carries four parts in
        a thousand of the core and is faint by any measure; what makes it
        visible at all is ``clip``, which spends the bottom of the ramp on
        exactly this.

        The shot grows the logo and then opens the exposure on all three
        colours at once, which turns a field of bare dots into a field of
        haloed ones.
        """
        t0 = 0
        _setup_render()
        _camera(location=(0, -11, 0), lens=38)
        create_glow_composition(threshold=0.75, type="BLOOM", size=4)

        # three rings with the peak of the third on the rim: the core out to
        # 0.46 of the radius, a first ring peaking at 0.61, and the second -
        # four times fainter again - brightest exactly where the disc ends,
        # so neighbouring instances, which the logo sets tangent, touch ring
        # to ring. 160 rings of mesh is plenty at the size one is seen at
        red = AiryDiscModifier(name="AiryRed", rings=3, color="red",
                               radial=160, angular=128, gamma=2.2)
        green = AiryDiscModifier(name="AiryGreen", rings=3, color="green",
                                 radial=160, angular=128, gamma=2.2)
        blue = AiryDiscModifier(name="AiryBlue", rings=3, color="blue",
                                radial=160, angular=128, gamma=2.2)

        # rotated a quarter turn about x, like every other flat thing in this
        # file: the logo is laid out in x-y and the camera looks down +y. It
        # grows upwards from its origin and reaches two units, so the scale
        # and a drop of the same amount are what centre it in the frame
        logo = LogoFromInstances(
            instance=BObject, details=6, name="AiryLogo",
            rotation_euler=[pi / 2, 0, 0], scale=[2.4] * 3,
            location=[0, 0, -2.4],
            kwargs_red={"mesh": ibpy.create_mesh([[0, 0, 0]]),
                        "geo_node_modifier": red},
            kwargs_green={"mesh": ibpy.create_mesh([[0, 0, 0]]),
                          "geo_node_modifier": green},
            kwargs_blue={"mesh": ibpy.create_mesh([[0, 0, 0]]),
                         "geo_node_modifier": blue})
        t0 = 1 + logo.grow(begin_time=t0, transition_time=3)

        # one ramp per colour, all three over the same window: the exposure
        # is a property of the picture, not of any one disc
        for modifier in (red, green, blue):
            ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(modifier, "Aperture"), from_value=0.5,
                                      to_value=0.286, begin_time=0,
                                      transition_time=6)
            t0 = ibpy.change_default_value(ibpy.get_geometry_node_from_modifier(modifier, "Gamma"),
                                           from_value=2.2, to_value=0.9,
                                           begin_time=0, transition_time=6)

        t0 = 0.5 + logo.move(direction=[-2.5, 0, 0], begin_time=t0, transition_time=0.5)
        toc = ["Plane Waves", "Standing Waves", "Circular Waves", "Diffraction", "Fuchsian Theory"]

        for i, line in enumerate(toc):
            line_obj = SimpleTexBObject(r"\text{" + line + "}", location=[1, 0, 2.25 - 1.1 * i], text_size="large")
            t0 = 0.5 + line_obj.write(begin_time=t0, transition_time=0.5)
        self.t0 = t0 + 2

    # ------------------------------------------------------------------
    def plane_waves(self):
        """Four beams crossing: an optical lattice, in points.

        With plane waves the phase is k n.r rather than k|r - s|, so the
        fringes are flat and periodic instead of curved. Four beams along the
        diagonals of a cube - the tetrahedral arrangement, each pair at the
        tetrahedral angle to the next - interfere into a genuinely 3D lattice
        of bright spots, the thing cold atoms get trapped in. Beams in a less
        symmetric arrangement are worth trying and mostly give *columns*: two
        counter-propagating pairs sharing a plane leave the third direction
        unmodulated, and the cloud comes out as strands rather than points.

        ``sharpness=3`` is doing visible work here. A cloud is seen through,
        so at sharpness 1 the dim points between the camera and each bright
        spot veil it, and 22000 points read as purple fog; cubing the
        intensity empties the space between the spots and the lattice appears.
        It is a lie about the physics, told for the same reason a long
        exposure is.

        Sweeping the wavelength scales the whole lattice about the origin,
        which is the clearest way to see that the spacing is set by lambda and
        nothing else.
        """
        t0 = 0
        _setup_render()
        _lights()
        _camera(location=(8, -9, 6))

        wavelength = 1.1
        lattice = InterferenceModifier(name="PlaneWaves", size=4, count=22000,
                                       wave="plane",
                                       sources=((1, 1, 1), (1, -1, -1),
                                                (-1, 1, -1), (-1, -1, 1)),
                                       wavelength=wavelength, sharpness=3,
                                       radius=0.022, color_by_density=True,
                                       emission=0.6, box_color="text",
                                       box_radius=0.012)
        host = _cloud(lattice, name="PlaneWaves")

        title = _title("four beams: a standing-wave lattice")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        t0 = 0.5 + _turntable(host, begin_time=t0, transition_time=9)

        # k = 2 pi / lambda, so halving k doubles the lattice spacing
        number = ibpy.get_geometry_node_from_modifier(lattice, "WaveNumber")
        t0 = ibpy.change_default_value(number, from_value=tau / wavelength,
                                       to_value=tau / (2 * wavelength),
                                       begin_time=t0, transition_time=6)
        _linearize(lattice)
        self.t0 = t0

    # ------------------------------------------------------------------
    def gaussian(self):
        """Any function at all: a blob, to show the machinery is not special.

        f = exp(-r^2 / 2 sigma^2). Nothing about the interference scenes is
        built into the sampler - a distribution is a node group returning a
        number in [0, 1] at a position, and this one is four nodes.
        """
        t0 = 0
        _setup_render()
        _lights()
        _camera()

        blob = GaussianCloudModifier(name="Blob", sigma=0.8, size=4,
                                     count=16000, radius=0.02,
                                     color_by_density=True, emission=0.5,
                                     box_color="text", box_radius=0.012)
        host = _cloud(blob, name="Gaussian")

        title = _title(r"any $f$: $e^{-r^2/2\sigma^2}$")
        t0 = 0.5 + title.write(begin_time=t0, transition_time=1.5)

        self.t0 = _turntable(host, begin_time=t0, transition_time=10)

    # ------------------------------------------------------------------
    def samplers(self):
        """The two ways, side by side, on the same distribution.

        Left: the density pushed into a ``Volume Cube`` at resolution 12, so
        each fringe is worth about one voxel. Right: rejection sampling, which
        evaluates f at each point's own position and has no grid to be limited
        by. Same f, same point count, same seed - the left cloud is visibly
        the blurred one, and by the measure in the module docstring it keeps
        64% of the fringe contrast against the right one's 100%.

        Both clouds are slabs a unit thick rather than cubes, and face-on.
        A point cloud is transparent: looking through 3.4 units of a cube
        stacks every fringe on every other one and the difference this scene
        exists to show is the first thing that goes.

        Raising the left resolution to 64 makes the two indistinguishable;
        the cost is that the grid is resolution^3 voxels, so the memory goes
        up by a factor of 150 to buy back a third of a pattern that rejection
        sampling gets for nothing.
        """
        t0 = 0
        _setup_render()
        _lights()
        _camera(location=(0, -13, 4.5), lens=38)

        shared = dict(size=(3.4, 1.0, 3.4), count=14000, wavelength=0.8,
                      radius=0.018, color_by_density=True, emission=0.6,
                      box_color="text", box_radius=0.01)

        def two_beams(center):
            """The same pair of sources, carried along with the slab.

            The sources are world positions, so a slab moved sideways without
            them would show a different *part* of the pattern - and the two
            halves of a comparison would then differ by where they are as
            well as by how they were sampled.
            """
            return (center[0] - 1.0, 0, 0), (center[0] + 1.0, 0, 0)

        left_center, right_center = (-2.1, 0, 0), (2.1, 0, 0)
        coarse = InterferenceModifier(name="GridSampler", method="grid",
                                      resolution=12, center=left_center,
                                      sources=two_beams(left_center), **shared)
        exact = InterferenceModifier(name="RejectionSampler", method="rejection",
                                     center=right_center,
                                     sources=two_beams(right_center), **shared)
        _cloud(coarse, name="GridCloud")
        _cloud(exact, name="RejectionCloud")

        left = _title("grid, resolution 12", location=(-2.1, 0, 2.2))
        right = _title("rejection", location=(2.1, 0, 2.2))
        t0 = 0.5 + left.write(begin_time=t0, transition_time=1)
        t0 = 0.5 + right.write(begin_time=t0, transition_time=1)
        self.t0 = t0 + 2

    # ------------------------------------------------------------------
    def envelope_computation(self):
        r"""The whistle's pipe length, worked out on the back of an envelope.

        Two hands, one after the other. The first is
        :func:`_envelope_drawing`, the pen of ``video_bff`` - nine bowed,
        overshooting strokes that are an envelope because of where they are
        rather than because anything is solid. The second is a real hand: the
        calculation was written on an ipad, exported by ``pen2curve`` as
        fitted bezier curves and arrives here through
        :class:`~objects.pen2curve.Pen2CurveObject`.

        What that object buys is that the writing is *written*. The strokes
        are stored in the order the pen made them, so a threshold walked
        along the point index reproduces the hand: the pen crosses the page
        at the speed of the ``Progress`` dial, in the order the ink went
        down, with no stroke ever appearing before the one that preceded it.
        Hence ``write(begin_time, transition_time)`` and nothing else - the
        choreography is already in the file.

        The ink is recoloured on the way in. The pen wrote on white paper in
        black, blue and red; on this scene's black background black ink is
        invisible, so it is mapped to the palette's ``text`` and the red
        result box to ``important``, which is where the eye ends up.

        Both hands are the same size and the writing is the reason for the
        envelope's proportions: the note is portrait, half again as tall as
        it is wide, so the envelope stands on its short edge. It is also the
        reason the folds are where they are - see :func:`_envelope_drawing`,
        which leaves the middle panel clear so that the calculation lands on
        blank paper rather than across a crease.
        """

        _setup_render()
        create_glow_composition(threshold=0.6, strength=0.4, size=5)
        # the strokes light themselves; the suns are only here to keep the
        # bevelled tubes of the envelope from reading as flat tape
        _lights(target=(0, 0, 0), strength=0.4)
        _camera(location=(0, -20, 0), target=(0, 0, 0), lens=28)

        # --- the envelope --------------------------------------------------
        rng = np.random.default_rng(20260826)  # one seed, one hand, every render
        style = {
            'edge': dict(color='drawing', emission=0.6, thickness=0.5),
            # the creases are paper, not outline: dimmer and thinner, so the
            # folds are read rather than looked at, and the writing that
            # crosses none of them stays the brightest thing on the envelope
            'seam': dict(color='drawing', emission=0.3, thickness=0.32),
            'seal': dict(color='custom1', emission=0.75, thickness=0.55),
        }
        strokes = _envelope_drawing(rng)
        t_draw, span = 0.3, 3.2
        orders = max(stroke['order'] for stroke in strokes)
        for index, stroke in enumerate(strokes):
            curve = _ink(stroke['points'], 'Envelope_%s_%d' % (stroke['part'], index),
                         extrude=0, **style[stroke['part']])
            curve.grow(begin_time=t_draw + span * stroke['order'] / orders,
                       transition_time=0.55)
        t0 = t_draw + span + 0.55

        pencil = Pencil(colors=['wood', 'text'])
        t0 = 0.5 + pencil.grow(begin_time=t0, transition_time=1)

        # --- what is written on it -----------------------------------------
        # sized and placed to land in the panel the folds leave empty: below
        # the wax, above the bottom fold, inside the two side seams. y is a
        # hair in front of the envelope's own strokes, so the two drawings
        # never fight over the same pixels
        note = Pen2CurveObject("envelope_calculation.json",
                               ink_height=7.2, orientation="FRONT",
                               ink={'#111318': 'text', '#2563eb': 'drawing',
                                    '#dc2626': 'important'},
                               location=[0, -0.08, -0.65],
                               radius=(0.009, 0.01), pencil=pencil,
                               # the recording opens with a tap of the pen
                               # well above the writing; the drawing starts
                               # after it
                               start_index=5, name="EnvelopeNote")
        # everything left over, minus five seconds to read the result by
        parts = [-0.01, 0.239, 0.375, 0.555, 0.893, 1]

        for i in range(1, len(parts)):
            if i == 5:
                pencil.change_pencil_color(new_color='important', begin_time=t0 - 0.5, transition_time=0.5)
            t0 = 0.5 + note.write(begin_time=t0, transition_time=(parts[i] - parts[i - 1]) * 25,
                                  from_value=parts[i - 1], to_value=parts[i])

        self.t0 = t0

    def promo_intro(self):
        r"""
        The promo's opening shot: the whistle and the drum side by side, each
        filling its half of the frame and turning about the vertical through
        its own centre.

        They are the video's two examples, a pipe and a membrane, and here
        they are only to be looked at, so nothing else is in the frame. Both
        grow in over the first second and make one revolution over the whole
        sub-scene.
        """
        t0 = 0
        duration = self.sub_scenes['promo_intro']['duration']

        # background and render settings, as in :meth:`wave_equation_intro` -
        # both models are shiny, and with a transparent film the hdri is the
        # only thing they have to reflect
        ibpy.set_hdri_background("forest", 'exr', simple=True,
                                 transparent=True, no_transmission_ray=False,
                                 rotation_euler=pi / 180 * Vector([0, 0, 110]),
                                 reflections=True)

        ibpy.set_render_engine(denoising=False, transparent=True, frame_start=1,
                               resolution_percentage=100, engine=BLENDER_EEVEE,
                               taa_render_samples=128, motion_blur=False)

        # straight on and aimed at the origin: 38 mm from 13 units back holds
        # x in [-6.2, 6.2] and z in [-3.5, 3.5], so each model gets a half of
        # 6.2 x 6.9 centred on x = -+3.1
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)

        # What has to fit in a half is not the model but the solid it sweeps
        # out in a turn, a cylinder about the vertical through its centre. Both
        # meshes are modelled about their own origin, and measured on them:
        # the drum, lying on its side with a skin to the camera, sweeps a
        # radius of 0.96 and stands 1.9 tall; the whistle, a tube 1.9 long
        # tilted 45 degrees out of the horizontal, sweeps 0.83 and stands 1.68.
        # Both cylinders are square in profile - that is what 45 degrees is
        # for; lying flat the whistle would shrink to a blob twice a turn, as
        # its axis swings through the line of sight - so one factor between
        # them (0.96 / 0.83) makes the two the same size, and the drum's
        # scale makes that size 84% of a half's width. The rest is margin for
        # perspective: the half of the cylinder nearer the camera stands up to
        # 2.6 units closer to it than the centre, and a rim swinging through
        # there comes out wider than the cylinder measured on paper - at 88%
        # the drum's rim grazed the right edge of the frame.
        _HALF = 3.08
        _DRUM_SCALE = 2.7
        _WHISTLE_SCALE = _DRUM_SCALE * 0.96 / 0.83

        # blender's XYZ euler applies R = Rz . Ry . Rx: the x angle tilts the
        # tube (modelled along y) out of the horizontal, and the quarter turn
        # about z then swings it into the picture plane, so the first frame
        # shows its full length on the diagonal with the ring at the bottom.
        # The drum's half turn puts the white skin to the camera.
        whistle_pose = Vector([pi / 4, pi, pi / 2])
        drum_pose = Vector([0, 0, pi])

        whistle = Whistle(location=[-_HALF, 0, 0], rotation_euler=whistle_pose)
        drum = Drum(location=[_HALF, 0, 0], rotation_euler=drum_pose)
        whistle.grow(scale=_WHISTLE_SCALE, begin_time=t0, transition_time=1)
        drum.grow(scale=_DRUM_SCALE, begin_time=t0, transition_time=1)

        # and they turn from the first frame on, growing in while they do -
        # one revolution over the sub-scene. rotate takes an absolute euler,
        # so the turn is added to the pose: only the z angle is swept, and
        # since Rz is the outermost factor that is a turn about the world
        # vertical with the pose riding round inside it.
        whistle.rotate(rotation_euler=whistle_pose + Vector([0, 0, 1.5 * tau]),
                       begin_time=t0, transition_time=duration)
        t0 = drum.rotate(rotation_euler=drum_pose + Vector([0, 0, 1.5 * tau]),
                         begin_time=t0, transition_time=duration)

        self.t0 = t0

    def promo_offer(self):
        r"""
        The promo's call to action: "Try Meshy.ai now!" over "50% off with
        the link in the description", on a grey band across the full width of
        an otherwise transparent frame - an overlay for the edit.

        The band opens from a glowing line across the middle of the frame,
        orange rules sweep in along its edges, and the two lines are written.
        From then on the band keeps itself alive: a glint runs along the first
        line every few seconds, "now!" blinks, and the "50%" swells and
        settles again with a flash. At the end the band closes back into the
        line it came from.
        """
        duration = self.sub_scenes['promo_offer']['duration']
        _setup_render()
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        def keys(obj, data_path, timeline):
            """Keyframe ``data_path`` of ``obj`` to each (time, value)."""
            obj = ibpy.get_obj(obj)
            for t, value in timeline:
                setattr(obj, data_path, value)
                ibpy.insert_keyframe(obj, data_path, int(t * FRAME_RATE))

        # 38 mm from 13 units back holds x in [-6.16, 6.16] and z in
        # [-3.46, 3.46]: the band, 2.5 tall, is a good third of the frame, and
        # it is cut a little wider than the frame so its ends are never seen
        _HALF_HEIGHT = 1.25
        _HALF_WIDTH = 6.6
        # measured: at text_size 1 the title is 2.51 wide and 0.30 tall,
        # the second line's two parts are 0.79 and 4.63 wide and 0.15 tall
        # (both centred on their location). These sizes make the title 10.8
        # wide and 1.3 tall and the second line, with a 0.35 gap, 10.4 wide -
        # stacked on the band with room for the 50% to swell. The 50% is set
        # larger than the rest of its line, it is the point of the offer
        _TITLE_SIZE = 4.3
        _OFFER_SIZE = 1.85
        _FIFTY_SIZE = 2.5
        _FIFTY_WIDTH = 0.794 * _FIFTY_SIZE
        _LINE2_LEFT = -0.5 * (_FIFTY_WIDTH + 0.35 + 4.629 * _OFFER_SIZE)
        _FIFTY_X = _LINE2_LEFT + 0.5 * _FIFTY_WIDTH
        _OFFER_X = _LINE2_LEFT + _FIFTY_WIDTH + 0.35

        band = Plane(u=[-_HALF_WIDTH, _HALF_WIDTH], v=[-_HALF_HEIGHT, _HALF_HEIGHT],
                     resolution=1, name="PromoBand", color="gray_1",
                     location=(0, 0.1, 0), rotation_euler=[pi / 2, 0, 0],
                     apply_location=False)
        band.appear(begin_time=0, transition_time=0, alpha=0.85)
        # blended, not eevee's dithered alpha, which speckles a flat surface
        # (see :meth:`watermark`)
        # gray_1 still comes out a light grey under the hdri, too pale for
        # white letters - so a darker base
        band_material = ibpy.get_material_of(band)
        band_material.surface_render_method = 'BLENDED'
        band_material.node_tree.nodes['Principled BSDF'].inputs['Base Color'] \
            .default_value = (0.045, 0.045, 0.05, 1)
        # first a hairline drawn out from the centre, then the band opens
        # vertically from it - and the reverse at the end. The plane is turned
        # a quarter about x, so its local y is the frame's vertical
        close = duration - 1
        keys(band, "scale", [(0, (0, 0.02, 1)), (0.6, (1, 0.02, 1)),
                             (1.1, (1, 1, 1)),
                             (close, (1, 1, 1)), (close + 0.5, (1, 0.02, 1)),
                             (duration - 0.1, (0, 0.02, 1))])

        # orange rules along both edges, sweeping in from opposite sides
        rules = []
        for sign in (1, -1):
            rule = Plane(u=[-_HALF_WIDTH, _HALF_WIDTH], v=[-0.03, 0.03],
                         resolution=1, name="PromoRule", color="important",
                         location=(-sign * _HALF_WIDTH, 0.05, sign * _HALF_HEIGHT),
                         rotation_euler=[pi / 2, 0, 0], apply_location=False)
            # the glow in the rule's own orange, not the default white
            bsdf = ibpy.get_material_of(rule).node_tree.nodes['Principled BSDF']
            bsdf.inputs['Emission Color'].default_value = \
                bsdf.inputs['Base Color'].default_value
            rule.appear(begin_time=0.9, transition_time=0)
            ibpy.change_emission_to(rule, 4, begin_time=0.9, transition_time=0.1)
            # pivot at the end the rule comes from: the plane is shifted by
            # half a width and stretched from zero
            keys(rule, "location", [(0.9, (-sign * _HALF_WIDTH, 0.05, sign * _HALF_HEIGHT)),
                                    (1.7, (0, 0.05, sign * _HALF_HEIGHT)),
                                    (close - 0.3, (0, 0.05, sign * _HALF_HEIGHT)),
                                    (close + 0.2, (sign * _HALF_WIDTH, 0.05, sign * _HALF_HEIGHT))])
            keys(rule, "scale", [(0.9, (0, 1, 1)), (1.7, (1, 1, 1)),
                                 (close - 0.3, (1, 1, 1)), (close + 0.2, (0, 1, 1))])
            rules.append(rule)

        # ---- the two lines --------------------------------------------------
        title = SimpleTexBObject(r"\text{Try Meshy.ai now!}", text_size=_TITLE_SIZE,
                                 color="text", aligned="center",
                                 location=(0, 0, 0.45))
        meshy = title.find_letters(r"Meshy.ai")
        now = title.find_letters(r"now!")
        t0 = title.write(begin_time=1.4, transition_time=1.2)
        title.change_color_of_letters(meshy, "joker", begin_time=t0, transition_time=0.4)

        # the second line is two objects, so that the "50%" can swell about
        # its own centre (it is centred, the rest left-aligned after it)
        fifty = SimpleTexBObject(r"\text{50\%}", text_size=_FIFTY_SIZE,
                                 color="important", aligned="center",
                                 location=(_FIFTY_X, 0, -0.75))
        offer = SimpleTexBObject(r"\text{off with the link in the description}",
                                 text_size=_OFFER_SIZE, color="text",
                                 aligned="left", location=(_OFFER_X, 0, -0.75))
        fifty.write(begin_time=t0, transition_time=0.5)
        t0 = offer.write(begin_time=t0 + 0.4, transition_time=1.2)

        # ---- keeping it alive -----------------------------------------------
        beat = 3.0
        t = t0 + 0.3
        while t + beat < close:
            # the 50% swells and settles, flashing as it peaks
            keys(fifty, "scale", [(t, (1, 1, 1)), (t + 0.35, (1.25, 1.25, 1.25)),
                                  (t + 0.9, (1, 1, 1))])
            for letter in fifty.letters:
                ibpy.change_emission_to(letter, 5, begin_time=t, transition_time=0.35)
                ibpy.change_emission_to(letter, 0, begin_time=t + 0.35, transition_time=0.55)
            # a glint runs along the first line, letter by letter; each letter
            # is put back by hand, since highlight_letters would restore the
            # colour it was built with and turn "Meshy.ai" white again
            for index in range(len(title.letters)):
                glint = t + 1.2 + 0.04 * index
                highlight_letters(title, [index], color="important", emission=4,
                                  begin_time=glint, transition_time=0.45,
                                  restore=False)
                title.change_color_of_letters(
                    [index], "joker" if index in meshy else "text",
                    begin_time=glint + 0.3, transition_time=0.15)
            # and "now!" blinks twice
            for blink in (0, 0.35):
                for index in now:
                    ibpy.change_emission_to(title.letters[index], 6,
                                            begin_time=t + 2.1 + blink, transition_time=0.1)
                    ibpy.change_emission_to(title.letters[index], 0,
                                            begin_time=t + 2.25 + blink, transition_time=0.1)
            t += beat

        # ---- and away -------------------------------------------------------
        for text in (title, fifty, offer):
            text.disappear(begin_time=close - 0.4, transition_time=0.5)

        self.t0 = duration

    def watermark(self):
        r"""
        "Preview", a quarter opaque, in the upper left corner of an otherwise
        transparent frame - an overlay for the cuts that go out before the
        final render.

        Nothing in it moves, so one still of it held over the whole cut does
        the job; the sub-scene is only there to render that still.
        """
        _setup_render()
        # no _lights, as in :meth:`frequency_list`: the glyph material carries
        # its own emission
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)

        # 38 mm from 13 units back holds x in [-6.16, 6.16] and z in
        # [-3.46, 3.46], 156 px to the unit at 1920 x 1080. Left-aligned, the
        # text is built out from its left edge, and this puts the top of the P
        # 60 px from both edges of the frame
        preview = SimpleTexBObject(r"\text{Preview}", text_size="Large",
                                   color="text", aligned="left",
                                   location=(-5.76, 0, 2.83))
        # write rather than appear, because appear does not pass alpha on; and
        # writing=False, so the glyphs are simply there, without the pen
        # stroke that writing draws round them first
        preview.write(begin_time=0, transition_time=0, alpha=0.25,
                      writing=False)
        # eevee's default render method is dithered, which renders an alpha as
        # the fraction of samples that see the surface: at 128 samples that is
        # 0.25 give or take 0.04 from pixel to pixel, a speckle on a flat
        # overlay. Blended is the real thing, 0.25 everywhere.
        for letter in preview.letters:
            for slot in ibpy.get_obj(letter).material_slots:
                if slot.material is not None:
                    slot.material.surface_render_method = 'BLENDED'

        self.t0 = 0

    def thumbnail(self):
        r"""
        The thumbnail: the wave equation in its shortest form, Box u = 0,
        huge and alone in the middle of the frame.

        Like :meth:`watermark` nothing moves - the sub-scene only renders the
        still.
        """
        _setup_render()
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)
        create_glow_composition(threshold=0.5, type="BLOOM", size=4)

        equation = SimpleTexBObject(r"\Box\,u=0", text_size="Huge",
                                    color="example", aligned="center",
                                    location=(0, 0, 0))
        equation.write(begin_time=0, transition_time=0, writing=False)

        self.t0 = 0

    def lambda_rgb_test(self):
        r"""A plane in the colour of monochromatic light, swept from 400 to 800 nm.

        The test of :class:`~shader_nodes.shader_nodes.WaveLengthToRGB`: the
        plane wears :func:`~appearance.textures.lambda_to_rgb_texture`, and its
        ``Lambda`` dial runs across the range in twenty seconds.
        """
        _setup_render()
        _camera(location=(0, -13, 0), target=(0, 0, 0), lens=38)

        plane = Plane(u=[-4, 4], v=[-2.25, 2.25], resolution=1, name="LambdaPlane",
                      color="LambdaToRGB", rotation_euler=[pi / 2, 0, 0])
        plane.appear(begin_time=0, transition_time=0)
        ibpy.change_default_value(ibpy.get_node_from_shader(ibpy.get_material_of(plane), "Lambda"),
                                  from_value=400, to_value=800, begin_time=1, transition_time=20)

        self.t0 = 22


if __name__ == '__main__':
    try:
        example = InterferenceScene()
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
