"""
Letter-flight choreography for equation animations.

Turns "these letter copies travel from A to B" into polished keyframes:
arcing paths (instead of straight lines through the equation), staggered
per-letter timing, and highlight flashes.  Used by
:class:`objects.bderivation.BDerivation`; the pure-math part (schedules,
apex geometry) is bpy-free and unit-tested in
``tests/unit/objects/test_choreography.py``.
"""

import numpy as np

from interface import ibpy
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE

__all__ = [
    "stagger_schedule",
    "arc_apex",
    "fly_letter",
    "fly_letters",
    "cancel_letters",
    "highlight_letters",
]


# ---------------------------------------------------------------------------
# pure math
# ---------------------------------------------------------------------------

def stagger_schedule(n, begin_time, transition_time, stagger=0.5, order='left_to_right'):
    """Per-letter start times and durations for a staggered flight.

    A ``stagger`` fraction of the transition is spent fanning out the
    starts; every letter flies for the same (shortened) duration and the
    last letter still lands exactly at ``begin_time + transition_time``.

    :param n: number of letters
    :param stagger: 0 = all at once, 0.5 = last letter starts at half time
    :param order: 'left_to_right' (list order), 'right_to_left'
    :return: list of (start_time, duration), one per letter in list order
    """
    if n <= 0:
        return []
    stagger = min(max(stagger, 0.0), 0.95)
    total_delay = stagger * transition_time if n > 1 else 0.0
    duration = transition_time - total_delay
    schedule = []
    for i in range(n):
        rank = i if order == 'left_to_right' else n - 1 - i
        delay = total_delay * rank / max(n - 1, 1)
        schedule.append((begin_time + delay, duration))
    return schedule


def arc_apex(start, end, arc=0.3, normal=(0, 0, 1)):
    """Apex point of an arced flight from ``start`` to ``end``.

    The apex sits at the midpoint, offset perpendicular to the travel
    direction within the plane defined by ``normal`` (the text plane), by
    ``arc`` times the travel distance.  The sign is chosen so the arc bends
    towards positive y ("over" the equation); pass a negative ``arc`` to go
    under.

    :return: numpy array (3,)
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    distance = float(np.linalg.norm(direction))
    if distance < 1e-9:
        return (start + end) / 2
    perpendicular = np.cross(np.asarray(normal, dtype=float), direction / distance)
    norm = np.linalg.norm(perpendicular)
    if norm < 1e-9:  # travel along the normal -- no meaningful arc plane
        return (start + end) / 2
    perpendicular /= norm
    if perpendicular[1] < 0:  # bend upwards by default
        perpendicular = -perpendicular
    return (start + end) / 2 + arc * distance * perpendicular


# ---------------------------------------------------------------------------
# bpy layer
# ---------------------------------------------------------------------------

def fly_letter(bobject, start, end, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
               arc=0.3, normal=(0, 0, 1)):
    """Fly one letter (copy) from ``start`` to ``end`` along an arc.

    Three location keyframes (start, apex, end) on the object's own location
    channel; Blender's default bezier easing turns them into a smooth arc.
    Locations are in the object's parent frame.

    :return: begin_time + transition_time
    """
    obj = bobject.ref_obj if hasattr(bobject, 'ref_obj') else bobject
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    f0 = begin_time * FRAME_RATE
    f1 = (begin_time + transition_time) * FRAME_RATE

    obj.location = start
    ibpy.insert_keyframe(obj, "location", frame=f0)
    if arc:
        obj.location = arc_apex(start, end, arc=arc, normal=normal)
        ibpy.insert_keyframe(obj, "location", frame=(f0 + f1) / 2)
    obj.location = end
    ibpy.insert_keyframe(obj, "location", frame=f1)
    return begin_time + transition_time


def fly_letters(flights, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
                arc=0.3, stagger=0.5, order='left_to_right', normal=(0, 0, 1)):
    """Fly letter (copies) along arcs with staggered timing.

    :param flights: list of (bobject, start_location, end_location); the
        locations are in the object's parent frame.  The list should be
        ordered left-to-right for natural staggering.
    :param arc: apex offset as fraction of travel distance (0 = straight)
    :param stagger: fraction of the transition spent fanning out starts
    :return: begin_time + transition_time
    """
    schedule = stagger_schedule(len(flights), begin_time, transition_time,
                                stagger=stagger, order=order)
    for (bobject, start, end), (t0, duration) in zip(flights, schedule):
        fly_letter(bobject, start, end, begin_time=t0, transition_time=duration,
                   arc=arc, normal=normal)
    return begin_time + transition_time


def cancel_letters(groups, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
                   arc=0.2, shrink_fraction=0.4, normal=(0, 0, 1)):
    """Annihilate letter groups: they fly to their joint midpoint and shrink.

    Typical use: the ``+5`` and ``-5`` of an equation cancel -- copies of
    both terms arc towards each other (alternating over/under) and scale to
    nothing as they meet.

    :param groups: list of groups; each group is a list of
        (bobject, start_location) in a common parent frame
    :param arc: flight arc; consecutive groups alternate the arc sign so the
        groups approach on a pincer path
    :param shrink_fraction: last fraction of the transition during which the
        letters scale to zero
    :return: begin_time + transition_time
    """
    all_starts = [np.asarray(start, dtype=float)
                  for group in groups for _, start in group]
    if not all_starts:
        return begin_time + transition_time
    midpoint = np.mean(all_starts, axis=0)

    shrink_time = shrink_fraction * transition_time
    shrink_frame = (begin_time + transition_time - shrink_time) * FRAME_RATE
    end_frame = (begin_time + transition_time) * FRAME_RATE

    for g, group in enumerate(groups):
        group_arc = arc if g % 2 == 0 else -arc
        for bobject, start in group:
            obj = bobject.ref_obj if hasattr(bobject, 'ref_obj') else bobject
            fly_letter(bobject, start, midpoint, begin_time=begin_time,
                       transition_time=transition_time, arc=group_arc, normal=normal)
            ibpy.insert_keyframe(obj, "scale", frame=shrink_frame)
            obj.scale = (0, 0, 0)
            ibpy.insert_keyframe(obj, "scale", frame=end_frame)
    return begin_time + transition_time


def highlight_letters(text, indices, color='important', emission=3,
                      begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
                      restore=True):
    """Flash a group of letters: tint + emission pulse, then settle back.

    :param text: SimpleTexBObject
    :param indices: letter indices to highlight
    :param color: flash color ('important', 'joker', ...)
    :param emission: peak emission strength of the pulse (0 to skip)
    :param restore: return the letters to their original colors during the
        last third of the transition
    :return: begin_time + transition_time
    """
    flash_time = transition_time / 3
    text.change_color_of_letters(indices, color, begin_time=begin_time,
                                 transition_time=flash_time)
    if emission:
        for index in indices:
            letter = text.letters[index]
            ibpy.change_emission_to(letter, emission, begin_time=begin_time,
                                    transition_time=flash_time)
            ibpy.change_emission_to(letter, 0, begin_time=begin_time + transition_time - flash_time,
                                    transition_time=flash_time)
    if restore:
        for index in indices:
            original = text.color_map.get(text.letters[index], 'text')
            text.change_color_of_letters([index], original,
                                         begin_time=begin_time + transition_time - flash_time,
                                         transition_time=flash_time)
    return begin_time + transition_time
