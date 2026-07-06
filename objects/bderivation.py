"""
BDerivation -- animated equation transformations.

Automates the derivation idiom used throughout the videos (see
``video_bwm/scene_bwm.py::addition_formal``): each step of a derivation is a
LaTeX line; matched letters fly from the previous line into the new one,
glue glyphs are written in, terms can be highlighted -- but letters are
addressed by **substring** instead of hand-counted glyph indices, and the
letter correspondence between steps is computed automatically
(:mod:`objects.morph_planning`).

Example
-------
>>> d = BDerivation(r"(x+y)^2 = 9\\cdot(1+x\\cdot y)", display=display, line=1)
>>> t = d.write(begin_time=0, transition_time=1)
>>> t = d.step(r"x^2+2xy+y^2 = 9+9\\cdot x\\cdot y",
...            map={"(x+y)^2": "x^2+2xy+y^2"},
...            highlight="(x+y)^2",
...            begin_time=t + 0.5, transition_time=2)
>>> t = d.highlight("2xy", begin_time=t + 0.5, transition_time=1)

Modes
-----
* ``mode='new_line'``: the new step appears on the next line (of the
  optional :class:`~objects.display.Display`), matched letters fly down as
  copies along arcs with staggered timing, the rest is written in.
* ``mode='replace'``: the line transforms in place (shape-key morph of
  letter copies, then a hot swap to the pristine new line).
"""

import numpy as np
from interface import ibpy
from interface.ibpy import Vector
from objects.bmorph_text import BMorphText, PairMorph
from objects.choreography import fly_letter, highlight_letters, stagger_schedule
from objects.morph_planning import glyph_signature, plan_transition
from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE

__all__ = ["BDerivation"]


class BDerivation:
    """A growing chain of equation lines with animated transitions.

    Not itself a BObject -- it coordinates one :class:`SimpleTexBObject` per
    derivation line (``self.lines``), placed on a :class:`Display` or at
    plain vertical offsets.
    """

    def __init__(self, first_expression, display=None, line=1, line_step=1,
                 indent=0, column=1, scale=0.7, align_char='=',
                 location=None, line_spacing=Vector((0, 0, -1)),
                 name="BDerivation", **tex_kwargs):
        """
        :param first_expression: LaTeX of the first line
        :param display: optional Display; lines go on rows ``line``,
            ``line + line_step``, ...
        :param line, line_step, indent, column, scale: Display placement
        :param align_char: character whose occurrences are vertically
            aligned between consecutive lines ('=' by default; None or a
            missing character skips alignment)
        :param location: first-line location when no display is used
        :param line_spacing: per-line offset vector when no display is used
        :param tex_kwargs: forwarded to every SimpleTexBObject line
            (color, text_size, thickness, ...)
        """
        self.display = display
        self.align_char = align_char
        self.name = name
        self.tex_kwargs = dict(tex_kwargs)
        self.line_spacing = Vector(line_spacing)
        self.location = location
        self.scale = scale
        self.indent = indent
        self.column = column
        self.line_step = line_step
        self._next_line = line
        self._row0 = line

        first = self._make_line(first_expression, index=0)
        self.lines = [first]
        self.plans = []  # one MorphPlan per step, for introspection
        self._place_line(first)

    # ------------------------------------------------------------------
    # line management
    # ------------------------------------------------------------------

    @property
    def current(self):
        """The most recent derivation line."""
        return self.lines[-1]

    def _make_line(self, expression, index, **overrides):
        kwargs = dict(self.tex_kwargs)
        kwargs.update(overrides)
        kwargs.setdefault('name', "%s_line_%d" % (self.name, index))
        return SimpleTexBObject(expression, **kwargs)

    def _place_line(self, tex, same_slot=False):
        """Put a line on its display row (or plain offset position)."""
        row = self.lines[-1].derivation_row if same_slot else self._next_line
        if self.display is not None:
            self.display.add_text_in(tex, scale=self.scale, line=row,
                                     indent=self.indent, column=self.column)
        else:
            base = Vector(self.location) if self.location is not None else Vector()
            tex.ref_obj.location = base + self.line_spacing * (row - self._row0)
        tex.derivation_row = row
        if not same_slot:
            self._next_line += self.line_step

    def _align(self, tex):
        """Vertically align ``align_char`` of ``tex`` with the current line."""
        if self.align_char is None:
            return
        try:
            src = self.current.find_letters(self.align_char, occurrence=0)
            tgt = tex.find_letters(self.align_char, occurrence=0)
        except Exception:
            return  # no alignment character in one of the lines
        tex.align(self.current, char_index=tgt[0], other_char_index=src[0])

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    @staticmethod
    def _signatures(tex):
        sigs = []
        for i, letter in enumerate(tex.letters):
            splines, _ = BMorphText._extract_glyph(letter)
            sigs.append(glyph_signature(i, splines, location=list(letter.ref_obj.location)))
        return sigs

    def _resolve_map(self, tex, user_map, src_sigs, tgt_sigs):
        """Translate a substring map into pinned per-letter pairs.

        Accepts a dict ``{src_substr: tgt_substr | None, None: tgt_substr}``
        or an iterable of such pairs.  Substrings may carry ``@n``/``@all``
        occurrence suffixes.  Group-to-group entries are refined into
        per-letter pins by a nested plan_transition on just those letters.
        """
        if user_map is None:
            return []
        items = user_map.items() if isinstance(user_map, dict) else list(user_map)

        pins = []
        for src_spec, tgt_spec in items:
            if src_spec is None and tgt_spec is None:
                raise ValueError("map entry (None, None) is meaningless")
            if src_spec is None:  # forced fresh appearance
                for j in tex.find_letters(tgt_spec):
                    pins.append((None, j))
                continue
            src_indices = self.current.find_letters(src_spec)
            if tgt_spec is None:  # forced vanish
                for i in src_indices:
                    pins.append((i, None))
                continue
            tgt_indices = tex.find_letters(tgt_spec)
            # refine group->group into per-letter pins
            local_plan = plan_transition([src_sigs[i] for i in src_indices],
                                         [tgt_sigs[j] for j in tgt_indices])
            pins.extend((i, j) for i, j in local_plan.pairs)
            pins.extend((i, None) for i in local_plan.vanish)
            pins.extend((None, j) for j in local_plan.appear)
        return pins

    # ------------------------------------------------------------------
    # animation
    # ------------------------------------------------------------------

    def write(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """Write the current (first) line."""
        return self.current.write(begin_time=begin_time, transition_time=transition_time, **kwargs)

    def highlight(self, substring, occurrence=None, color='important', emission=3,
                  begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, restore=True):
        """Flash a substring of the current line (tint + emission pulse)."""
        indices = self.current.find_letters(substring, occurrence=occurrence)
        return highlight_letters(self.current, indices, color=color, emission=emission,
                                 begin_time=begin_time, transition_time=transition_time,
                                 restore=restore)

    def step(self, expression, mode='new_line', map=None, auto=True, auto_threshold=None,
             align=True, fade_old=None, arc=0.3, stagger=0.5,
             highlight=None, highlight_color='important', new_color=None,
             cancel=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
             **tex_overrides):
        """Advance the derivation to the next expression.

        :param expression: LaTeX of the next line
        :param mode: 'new_line' (letters fly to the next line) or 'replace'
            (the line morphs in place)
        :param map: manual matching, e.g.
            ``{"(x+y)^2": "x^2+2xy+y^2", "9@0": None, None: "42"}``
            (``None`` value = vanish, ``None`` key = fresh appearance);
            unmentioned letters are matched automatically unless ``auto=False``
        :param align: vertically align the ``align_char`` with the previous line
        :param fade_old: what happens to the previous line in 'new_line'
            mode: None (keep), 'dim' or 'hide'
        :param arc: flight arc as fraction of travel distance (0 = straight)
        :param stagger: fraction of the transition spent fanning out letters
        :param highlight: substring(s) of the current line to flash before
            they move
        :param new_color: color the flown copies change into mid-flight
        :param cancel: pairs of substrings that annihilate instead of
            carrying over, e.g. ``[("+5", "-5")]`` -- copies of both terms
            fly to their joint midpoint and shrink to nothing
        :return: begin_time + transition_time
        """
        source = self.current
        tex = self._make_line(expression, index=len(self.lines), **tex_overrides)
        self._place_line(tex, same_slot=(mode == 'replace'))
        if align:
            self._align(tex)

        # cancelling letters are taken out of the matching entirely
        cancel_groups = []
        cancelled = set()
        if cancel:
            pairs = [cancel] if isinstance(cancel, tuple) else list(cancel)
            for spec_a, spec_b in pairs:
                group_a = source.find_letters(spec_a)
                group_b = source.find_letters(spec_b)
                cancel_groups.append((group_a, group_b))
                cancelled.update(group_a)
                cancelled.update(group_b)

        src_sigs = self._signatures(source)
        tgt_sigs = self._signatures(tex)
        pins = self._resolve_map(tex, map, src_sigs, tgt_sigs)
        pins.extend((i, None) for i in cancelled)
        plan = plan_transition(src_sigs, tgt_sigs, mapping=pins or None,
                               auto=auto, auto_threshold=auto_threshold)
        # the annihilation animation replaces the default vanish treatment
        plan.vanish = [i for i in plan.vanish if i not in cancelled]
        self.plans.append(plan)

        # optional pre-flight highlight of the moving material
        flight_begin, flight_time = begin_time, transition_time
        if highlight is not None:
            specs = highlight if isinstance(highlight, (list, tuple)) else [highlight]
            indices = []
            for spec in specs:
                indices.extend(source.find_letters(spec))
            highlight_letters(source, indices, color=highlight_color,
                              begin_time=begin_time, transition_time=0.5 * transition_time)
            flight_begin = begin_time + 0.25 * transition_time
            flight_time = 0.75 * transition_time

        if mode == 'new_line':
            self._step_new_line(source, tex, plan, flight_begin, flight_time,
                                arc=arc, stagger=stagger, new_color=new_color,
                                fade_old=fade_old,
                                begin_time=begin_time, transition_time=transition_time)
        elif mode == 'replace':
            self._step_replace(source, tex, plan, flight_begin, flight_time)
        else:
            raise ValueError("unknown mode %r (use 'new_line' or 'replace')" % mode)

        for group_a, group_b in cancel_groups:
            self._animate_cancel(source, group_a, group_b, arc,
                                 hide_originals=(mode == 'replace'),
                                 begin_time=flight_begin, transition_time=flight_time)

        self.lines.append(tex)
        return begin_time + transition_time

    def _animate_cancel(self, source, group_a, group_b, arc, hide_originals,
                        begin_time, transition_time):
        """Copies of two letter groups fly together and annihilate."""
        from objects.choreography import cancel_letters

        groups = []
        for indices in (group_a, group_b):
            group = []
            for i in indices:
                letter = source.letters[i]
                copy = letter.copy(clear_animation_data=True)
                source.copies_of_letters.append(copy)
                copy.appear(begin_time=begin_time, transition_time=0, clear_data=True)
                group.append((copy, Vector(letter.ref_obj.location)))
                if hide_originals:
                    obj = letter.ref_obj
                    ibpy.insert_keyframe(obj, "scale", frame=begin_time * FRAME_RATE - 1)
                    obj.scale = (0, 0, 0)
                    ibpy.insert_keyframe(obj, "scale", frame=begin_time * FRAME_RATE)
            groups.append(group)
        cancel_letters(groups, begin_time=begin_time, transition_time=transition_time,
                       arc=max(abs(arc), 0.1))

    def _step_new_line(self, source, tex, plan, flight_begin, flight_time,
                       arc, stagger, new_color, fade_old, begin_time, transition_time):
        # inter-line shift in the source container frame (both lines share
        # the same parent and orientation; cf. move_letters_to)
        scale = source.ref_obj.scale
        shift = ibpy.get_location(tex) - ibpy.get_location(source)
        for i in range(3):
            shift[i] /= scale[i]
        depth_offset = Vector((0, 0, -0.001))  # behind the written target letter

        pairs = sorted(plan.pairs, key=lambda p: source.letters[p[0]].ref_obj.location[0])
        schedule = stagger_schedule(len(pairs), flight_begin, flight_time, stagger=stagger)
        written = set()
        for (i, j), (t0, duration) in zip(pairs, schedule):
            letter = source.letters[i]
            copy = letter.copy(clear_animation_data=True)
            source.copies_of_letters.append(copy)
            copy.appear(begin_time=t0, transition_time=0, clear_data=True)
            start = Vector(letter.ref_obj.location)
            end = shift + Vector(tex.letters[j].ref_obj.location) + depth_offset
            fly_letter(copy, start, end, begin_time=t0, transition_time=duration, arc=arc)
            if new_color:
                copy.change_color(new_color=new_color, begin_time=t0 + duration / 2,
                                  transition_time=duration / 2)
            arrival = t0 + duration
            if j not in written:  # merges write the target letter only once
                tex.write(letter_set=[j], begin_time=arrival, transition_time=0)
                written.add(j)
            copy.disappear(begin_time=arrival, transition_time=0.1)

        if plan.appear:
            tex.write(letter_set=list(plan.appear),
                      begin_time=begin_time + 0.5 * transition_time,
                      transition_time=0.5 * transition_time)

        if fade_old == 'hide':
            source.disappear(begin_time=begin_time + transition_time,
                             transition_time=0.5 * transition_time)
        elif fade_old == 'dim':
            source.disappear(alpha=0.25, begin_time=begin_time + transition_time,
                             transition_time=0.5 * transition_time)

    def _step_replace(self, source, tex, plan, flight_begin, flight_time):
        morph = PairMorph(source, tex, plan)
        morph.animate(begin_time=flight_begin, transition_time=flight_time)
