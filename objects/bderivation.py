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
* ``mode='new_line_copy'``: 'new_line' for a step that carries its letters
  over unchanged -- the flown copies dance directly into their final slots
  and rest there; the pristine target letters take over in one invisible
  hot swap on the end frame (no per-letter fade/rewrite on arrival).
* ``mode='replace'``: the line transforms in place (shape-key morph of
  letter copies, then a hot swap to the pristine new line).
* ``mode='add_subtract'``: an in-place step in which the right-hand terms
  of the ``map`` fly across the ``=`` sign, flip their sign and merge into
  the matching left-hand terms (e.g. ``... = 9 + 495x`` collected onto the
  left).  See :meth:`BDerivation._step_add_subtract`.
* ``mode='swap'``: an in-place step in which every ``map`` term that sits
  on opposite sides of the ``=`` in source and target flies across while
  its leading sign **morphs** into the opposite sign; the term does not
  merge with anything -- it lands intact in its own slot on the other
  side.  See :meth:`BDerivation._step_swap`.
"""

import numpy as np
from interface import ibpy
from interface.ibpy import Vector
from objects.bmorph_text import (BMorphText, PairMorph, extract_glyph,
                                 make_curve_copy, rebuild_curve)
from objects.choreography import fly_letter, highlight_letters, stagger_schedule
from objects.morph_planning import compile_chain, glyph_signature, plan_transition
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
        # deferred stand-in -> pristine-letter handoffs (see _fly_foreign):
        # (tex, letter indices, stand-in copies), resolved when the next
        # step begins
        self._handoffs = []
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
             align=True, fade_old=None, arc=0.3, stagger=0.5, lift=1.0,
             highlight=None, highlight_color='important', new_color=None,
             cancel=None, sources=None,
             begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
             **tex_overrides):
        """Advance the derivation to the next expression.

        :param expression: LaTeX of the next line
        :param mode: 'new_line' (letters fly to the next line),
            'new_line_copy' (like 'new_line' for an unchanged line: the flown
            copies stay in their slots and hot-swap invisibly at the end
            instead of fading into a rewritten letter on arrival), 'replace'
            (the line morphs in place), 'add_subtract' (right-hand terms of
            the ``map`` fly across the ``=`` sign, flip sign and merge into
            their left-hand partners; see :meth:`_step_add_subtract`) or
            'swap' (``map`` terms that change sides fly across the ``=``
            with their sign morphing into its opposite, without merging;
            see :meth:`_step_swap`)
        :param lift: for ``mode='add_subtract'`` and ``mode='swap'``, how far
            (in line units) a travelling term rises above the baseline as it
            crosses over; a single value applies to every term, or pass a
            list with one value per moving term (ordered left-to-right on
            the source line) to lift them to different heights -- useful
            when two terms cross in opposite directions
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
        :param sources: ('new_line' modes only) list of
            ``(text, src_spec, tgt_spec)``: the new line's ``tgt_spec``
            letters are brought in by flying the ``src_spec`` letters of the
            foreign ``text`` (e.g. a brace label) instead of being written or
            matched from the current line; the foreign originals hide at
            take-off and the arrived copies BUILD the new line -- they stay
            put, and the pristine target letters only take over (invisibly,
            on identical geometry) at the moment the next step begins.
            Combine with ``map`` pins so these targets stay unmatched (a
            clash raises).
        :return: begin_time + transition_time
        """
        if self._handoffs:
            self._resolve_handoffs(begin_time)
        source = self.current
        tex = self._make_line(expression, index=len(self.lines), **tex_overrides)
        self._place_line(tex, same_slot=(mode in ('replace', 'add_subtract', 'swap')))
        if align:
            self._align(tex)

        if mode == 'add_subtract':
            self._step_add_subtract(source, tex, map, begin_time, transition_time,
                                    lift=lift, highlight_color=highlight_color)
            self.lines.append(tex)
            return begin_time + transition_time

        if mode == 'swap':
            self._step_swap(source, tex, map, begin_time, transition_time, lift=lift)
            self.lines.append(tex)
            return begin_time + transition_time

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

        # letters flown in from foreign text objects (e.g. brace labels)
        foreign = []
        if sources:
            if mode not in ('new_line', 'new_line_copy'):
                raise ValueError("sources are only supported in the new_line modes")
            for src_text, src_spec, tgt_spec in sources:
                src_idx = src_text.find_letters(src_spec)
                tgt_idx = tex.find_letters(tgt_spec)
                fsigs = self._signatures(src_text)
                local = plan_transition([fsigs[i] for i in src_idx],
                                        [tgt_sigs[j] for j in tgt_idx])
                if not local.pairs:
                    raise ValueError("source %r does not match target %r"
                                     % (src_spec, tgt_spec))
                foreign.append((src_text, local.pairs))
            foreign_targets = {j for _, fpairs in foreign for _, j in fpairs}
            clash = foreign_targets & {j for _, j in plan.pairs}
            if clash:
                raise ValueError("targets %r are both matched from the current "
                                 "line and supplied by sources; pin them to "
                                 "None in the map" % sorted(clash))
            # they are neither written in nor matched -- they fly in
            plan.appear = [j for j in plan.appear if j not in foreign_targets]

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

        if mode in ('new_line', 'new_line_copy'):
            self._step_new_line(source, tex, plan, flight_begin, flight_time,
                                arc=arc, stagger=stagger, new_color=new_color,
                                fade_old=fade_old,
                                begin_time=begin_time, transition_time=transition_time,
                                hot_swap_at_end=(mode == 'new_line_copy'))
            for src_text, fpairs in foreign:
                self._fly_foreign(src_text, tex, fpairs, flight_begin, flight_time,
                                  arc=arc, stagger=stagger,
                                  begin_time=begin_time,
                                  transition_time=transition_time)
        elif mode == 'replace':
            self._step_replace(source, tex, plan, flight_begin, flight_time)
        else:
            raise ValueError("unknown mode %r (use 'new_line', 'new_line_copy', "
                             "'replace', 'add_subtract' or 'swap')" % mode)

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
                       arc, stagger, new_color, fade_old, begin_time, transition_time,
                       hot_swap_at_end=False):
        # inter-line shift in the source LOCAL frame (where the letters and
        # their copies live): the line objects are rotated into the text
        # plane, so the parent-frame delta between the lines must be rotated
        # back before it can be added to letter locations
        scale = source.ref_obj.scale
        delta = ibpy.get_location(tex) - ibpy.get_location(source)
        shift = source.ref_obj.rotation_euler.to_matrix().inverted() @ delta
        for i in range(3):
            shift[i] /= scale[i]
        depth_offset = Vector((0, 0, -0.001))  # behind the written target letter

        swap_frame = (begin_time + transition_time) * FRAME_RATE
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
            if hot_swap_at_end:
                # the copy IS the letter until the very end: it rests in its
                # slot and pops off on the swap frame, where the pristine
                # target letter (0.001 in front) takes over seamlessly
                obj = copy.ref_obj
                base = list(obj.scale)
                ibpy.insert_keyframe(obj, "scale", frame=swap_frame)
                obj.scale = (0, 0, 0)
                ibpy.insert_keyframe(obj, "scale", frame=swap_frame + 0.5)
                obj.scale = base
                continue
            arrival = t0 + duration
            if j not in written:  # merges write the target letter only once
                tex.write(letter_set=[j], begin_time=arrival, transition_time=0)
                written.add(j)
            copy.disappear(begin_time=arrival, transition_time=0.1)

        if hot_swap_at_end and pairs:
            tex.write(letter_set=sorted({j for _, j in pairs}),
                      begin_time=begin_time + transition_time, transition_time=0)

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

    def _fly_foreign(self, src_text, tex, pairs, flight_begin, flight_time,
                     arc, stagger, begin_time, transition_time):
        """Fly letters of a foreign text object into their new-line slots.

        Used for ``step(sources=...)``: e.g. the ``Q_n`` under an underbrace
        travels up and becomes the new line's ``Q_n``.  Each copy is a
        shape-key morph from the foreign glyph to the pristine target glyph
        (an underbrace label is script-size, so it grows to text size in
        flight and lands as exact pristine geometry).  The copies are
        parented into the new line's frame (so the foreign object may
        disappear while they fly), the originals hide at take-off, and the
        arrived copies stay put as the visible line.  The pristine target
        letters take over only when the next step begins (registered in
        ``self._handoffs``), so nothing is visibly replaced.  Assumes both
        objects share their orientation (as SimpleTexBObjects do).
        """
        FR = FRAME_RATE
        # foreign start positions expressed in the new line's local frame
        scale = tex.ref_obj.scale
        fscale = src_text.ref_obj.scale
        delta = ibpy.get_location(src_text) - ibpy.get_location(tex)
        shift = tex.ref_obj.rotation_euler.to_matrix().inverted() @ delta
        ratio = [fscale[k] / scale[k] for k in range(3)]
        for k in range(3):
            shift[k] /= scale[k]
        depth_offset = Vector((0, 0, -0.001))

        pairs = sorted(pairs, key=lambda p: src_text.letters[p[0]].ref_obj.location[0])
        schedule = stagger_schedule(len(pairs), flight_begin, flight_time,
                                    stagger=stagger)
        arrived = []
        for (i, j), (t0, duration) in zip(pairs, schedule):
            letter = src_text.letters[i]
            tgt_letter = tex.letters[j]
            # the original label letter hides as its copy takes off
            obj = letter.ref_obj
            src_scale = list(obj.scale)
            ibpy.insert_keyframe(obj, "scale", frame=t0 * FR - 1)
            obj.scale = (0, 0, 0)
            ibpy.insert_keyframe(obj, "scale", frame=t0 * FR)

            # morph copy: foreign glyph -> pristine target glyph
            splines_src, cyc_src = extract_glyph(letter)
            splines_tgt, cyc_tgt = extract_glyph(tgt_letter)
            snapshots, cyclic = compile_chain([splines_src, splines_tgt],
                                              cyclic_flags=[cyc_src, cyc_tgt])
            copy = make_curve_copy(letter.ref_obj)
            rebuild_curve(copy, snapshots, cyclic)
            copy.parent = tex.ref_obj
            copy.matrix_parent_inverse = \
                tgt_letter.ref_obj.matrix_parent_inverse.copy()
            copy.rotation_euler = tgt_letter.ref_obj.rotation_euler
            ibpy.link(copy)

            # pop on at take-off, morphing from the label's letter scale to
            # the pristine letter's over the flight
            copy.scale = (0, 0, 0)
            ibpy.insert_keyframe(copy, "scale", frame=t0 * FR - 1)
            copy.scale = src_scale
            ibpy.insert_keyframe(copy, "scale", frame=t0 * FR)
            copy.scale = list(tgt_letter.ref_obj.scale)
            ibpy.insert_keyframe(copy, "scale", frame=(t0 + duration) * FR)

            p = Vector(letter.ref_obj.location)
            start = shift + Vector([ratio[k] * p[k] for k in range(3)])
            end = Vector(tgt_letter.ref_obj.location) + depth_offset
            fly_letter(copy, start, end, begin_time=t0, transition_time=duration,
                       arc=arc)
            shape_keys = copy.data.shape_keys
            blocks = shape_keys.key_blocks
            shape_keys.eval_time = blocks[0].frame
            shape_keys.keyframe_insert(data_path='eval_time', frame=t0 * FR)
            shape_keys.eval_time = blocks[-1].frame
            shape_keys.keyframe_insert(data_path='eval_time',
                                       frame=(t0 + duration) * FR)
            shape_keys.eval_time = 0
            arrived.append(copy)
        # the arrived copies ARE the new letters; the pristine ones take
        # over invisibly when the next step begins
        self._handoffs.append((tex, {j for _, j in pairs}, arrived))

    def _resolve_handoffs(self, begin_time):
        """Swap resting stand-in copies for the pristine letters they built.

        Runs on the first frame of the following step: the pristine letters
        are written and the stand-ins pop off on that same frame, on
        identical geometry, so the crossover is invisible -- and the new
        step's animation immediately takes over the pristine letters.
        """
        frame = begin_time * FRAME_RATE
        for tex, letter_set, copies in self._handoffs:
            tex.write(letter_set=sorted(letter_set), begin_time=begin_time,
                      transition_time=0)
            for copy in copies:
                obj = copy.ref_obj if hasattr(copy, 'ref_obj') else copy
                ibpy.insert_keyframe(obj, "scale", frame=frame - 1)
                obj.scale = (0, 0, 0)
                ibpy.insert_keyframe(obj, "scale", frame=frame)
        self._handoffs = []

    def _step_replace(self, source, tex, plan, flight_begin, flight_time):
        morph = PairMorph(source, tex, plan)
        morph.animate(begin_time=flight_begin, transition_time=flight_time)

    # ------------------------------------------------------------------
    # add / subtract: move a term across the '=' sign
    # ------------------------------------------------------------------

    def _step_add_subtract(self, source, tex, user_map, begin_time, transition_time,
                           lift=1.0, highlight_color='important'):
        """Move terms across the ``=`` sign (an add/subtract manipulation).

        Each merge in ``user_map`` (two source terms sharing one target term)
        is read as a move.  The term on the **right** of the ``=`` (the mover)

        1. lifts straight up above the equation (each term by its own ``lift``
           if a per-term list is given),
        2. travels left and, as it crosses the ``=``, turns into a leading
           ``-`` (``+9`` becomes ``-9``): a term that already carries a leading
           operator flies with it and cross-fades it into the ``-``, otherwise
           a fresh ``-`` grows in,
        3. arrives above the term it joins on the left -- both flash
           ``highlight_color`` --,
        4. sinks into that term and shrinks away while the term morphs in
           place into the combined result (``3025`` becomes ``3016``), which
           then hot-swaps back to its normal (white) colour.

        Terms on the left keep their slot; everything not mentioned (the
        emptied right-hand side, a fresh ``0``, unchanged glyphs) is carried
        by an in-place :class:`PairMorph` timed to the sink phase.  The line
        stays in the same slot, so the derivation does not grow a new row.
        """
        T, b, FR = float(transition_time), float(begin_time), FRAME_RATE
        # phase boundaries as fractions of the transition
        f_lift, f_land, f_hold, f_sink = 0.22, 0.55, 0.66, 0.9

        eq = source.find_letters(self.align_char or '=')
        eq_x = source.letters[eq[0]].ref_obj.location[0]

        # operator glyphs (+/-) sitting on the right of '=' -- candidates to be
        # attached to the mover immediately on their right (its leading sign).
        # ``minus_ops`` remembers which of them are '-' so a mover that starts
        # negative can flip to a '+' as it crosses (rather than always to '-').
        right_ops, minus_ops = [], set()
        for spec in ('+', '-'):
            try:
                found = source.find_letters(spec + '@all')
            except Exception:
                found = []
            right_ops.extend(found)
            if spec == '-':
                minus_ops.update(found)
        right_ops = [o for o in right_ops
                     if source.letters[o].ref_obj.location[0] > eq_x]

        # --- read the map into moves (mover right of '=', partner left) -----
        items = user_map.items() if isinstance(user_map, dict) else list(user_map)
        by_target = {}
        for src_spec, tgt_spec in items:
            if src_spec is None or tgt_spec is None:
                raise ValueError("add_subtract map entries need a source and a target")
            src_idx = source.find_letters(src_spec)
            centre = np.mean([source.letters[i].ref_obj.location[0] for i in src_idx])
            side = 'mover' if centre > eq_x else 'stationary'
            by_target.setdefault(tgt_spec, {'mover': [], 'stationary': []})[side].append(src_idx)

        # --- refine the stationary (joined) terms into per-glyph morph pins --
        src_sigs, tgt_sigs = self._signatures(source), self._signatures(tex)
        stationary_pins, mover_glyphs, moves = [], set(), []
        for tgt_spec, sides in by_target.items():
            tgt_idx = tex.find_letters(tgt_spec)
            stationary = [i for group in sides['stationary'] for i in group]
            for group in sides['stationary']:
                # signatures carry their global letter index, so plan.pairs are
                # already (source_index, target_index) -- as in _resolve_map
                local = plan_transition([src_sigs[i] for i in group],
                                        [tgt_sigs[j] for j in tgt_idx])
                stationary_pins.extend(local.pairs)
            if stationary:
                land = (float(np.mean([source.letters[i].ref_obj.location[0] for i in stationary])),
                        float(np.mean([source.letters[i].ref_obj.location[1] for i in stationary])))
            else:
                land = (float(np.mean([tex.letters[j].ref_obj.location[0] for j in tgt_idx])),
                        float(np.mean([tex.letters[j].ref_obj.location[1] for j in tgt_idx])))
            for group in sides['mover']:
                mover_glyphs.update(group)
                # the leading operator is the +/- adjacent on the term's left
                min_x = min(source.letters[i].ref_obj.location[0] for i in group)
                cand = [o for o in right_ops
                        if 0 < min_x - source.letters[o].ref_obj.location[0] < 1.0]
                op = max(cand, key=lambda o: source.letters[o].ref_obj.location[0]) \
                    if cand else None
                moves.append({'glyphs': group, 'stationary': stationary,
                              'land': land, 'op': op})

        attached_ops = {mv['op'] for mv in moves if mv['op'] is not None}

        # --- in-place morph plan -------------------------------------------
        # The '=' partitions the line: everything on the right that is not a
        # mover is emptied out (forced vanish) and the target's right-hand side
        # (typically a fresh '0') is written in (forced appear).  This keeps the
        # greedy auto-matcher from dragging a leftover glyph across the '=' --
        # only the left-hand side (plus the unchanged '=' and 'x^2') is matched
        # automatically.  The movers themselves are pinned out and fly instead.
        tex_eq_x = tex.letters[tex.find_letters(self.align_char or '=')[0]].ref_obj.location[0]
        pinned_src = {i for i, _ in stationary_pins} | mover_glyphs
        pinned_tgt = {j for _, j in stationary_pins}
        right_src = [i for i, letter in enumerate(source.letters)
                     if i not in pinned_src and letter.ref_obj.location[0] > eq_x]
        right_tgt = [j for j, letter in enumerate(tex.letters)
                     if j not in pinned_tgt and letter.ref_obj.location[0] > tex_eq_x]
        # right-of-'=' terms that survive unchanged (e.g. a '7a_n' that never
        # moves) are matched source->target and morphed in place rather than
        # emptied and rewritten; only the genuine remainder vanishes/appears.
        # auto_threshold=0 keeps just the identical glyphs (LCS matches), so a
        # right side that collapses to a fresh '0' still pairs nothing and the
        # classic vanish-all / appear-'0' behaviour is preserved.
        right_keep = (plan_transition([src_sigs[i] for i in right_src],
                                      [tgt_sigs[j] for j in right_tgt],
                                      auto_threshold=0.0).pairs
                      if right_src and right_tgt else [])
        stationary_pins.extend(right_keep)
        kept_src = {i for i, _ in right_keep}
        kept_tgt = {j for _, j in right_keep}
        right_src = [i for i in right_src if i not in kept_src]
        right_tgt = [j for j in right_tgt if j not in kept_tgt]
        mapping = (stationary_pins
                   + [(i, None) for i in sorted(mover_glyphs | set(right_src))]
                   + [(None, j) for j in right_tgt])
        plan = plan_transition(src_sigs, tgt_sigs, mapping=mapping, auto=True)
        # movers and attached operators fly as copies instead of vanishing
        plan.vanish = [i for i in plan.vanish
                       if i not in mover_glyphs and i not in attached_ops]
        self.plans.append(plan)

        # right-of-'=' leftovers (e.g. the spare '+' between the two movers)
        # clear out early, as the terms they joined lift off
        early = [i for i in plan.vanish
                 if source.letters[i].ref_obj.location[0] > eq_x]
        plan.vanish = [i for i in plan.vanish if i not in early]
        for i in early:
            obj = source.letters[i].ref_obj
            ibpy.insert_keyframe(obj, "scale", frame=b * FR)
            obj.scale = (0, 0, 0)
            ibpy.insert_keyframe(obj, "scale", frame=(b + f_lift * T) * FR)

        # --- morph the joined terms in place, timed to the sink phase -------
        morph = PairMorph(source, tex, plan)
        morph.animate(begin_time=b + f_hold * T, transition_time=(f_sink - f_hold) * T)

        # keep the joined terms coloured while they morph; they hot-swap back
        # to white at the end of the morph
        joined = {i for move in moves for i in move['stationary']}
        for copy, src_letter, _ in morph.morphers:
            if source.letters.index(src_letter) in joined:
                ibpy.change_color(copy, highlight_color,
                                  begin_frame=(b + f_hold * T) * FR - 1,
                                  final_frame=(b + f_hold * T) * FR)

        # --- fly each mover across the '=' ----------------------------------
        # order the terms left-to-right so a per-term 'lift' list lines up with
        # the way they read on the source line
        moves.sort(key=lambda mv: np.mean(
            [source.letters[i].ref_obj.location[0] for i in mv['glyphs']]))
        if isinstance(lift, (list, tuple)):
            if len(lift) != len(moves):
                raise ValueError("lift has %d entries but there are %d moving terms"
                                 % (len(lift), len(moves)))
            lifts = list(lift)
        else:
            lifts = [lift] * len(moves)

        minus = self._minus_letter(tex, source)
        plus = self._plus_letter(tex, source)
        for move, mv_lift in zip(moves, lifts):
            # a term that started with a leading '-' flips to '+' on crossing;
            # everything else (a '+' or bare term) flips to '-'
            sign = plus if move['op'] in minus_ops else minus
            self._fly_mover(source, move, sign, eq_x, b, T,
                            f_lift, f_land, f_hold, f_sink, mv_lift, highlight_color)
            if move['stationary']:
                source.change_color_of_letters(
                    move['stationary'], highlight_color,
                    begin_time=b + f_land * T, transition_time=0.06 * T)

        return b + T

    def _minus_letter(self, tex, source):
        """A ``-`` glyph (a live letter) to clone as the sign movers acquire.

        Borrowed from the target line (or the source), so it matches the font;
        only if neither line contains one is a throwaway rendered.
        """
        for text in (tex, source):
            try:
                found = text.find_letters('-@all')
            except Exception:
                found = []
            if found:
                return text.letters[found[0]]
        stub = SimpleTexBObject(r"-", **self.tex_kwargs)
        return stub.letters[0]

    def _plus_letter(self, tex, source):
        """A ``+`` glyph to clone as the sign a negative mover flips into.

        Mirror of :meth:`_minus_letter`; a mover that started with a leading
        ``-`` grows this ``+`` as it crosses the ``=``.
        """
        for text in (tex, source):
            try:
                found = text.find_letters('+@all')
            except Exception:
                found = []
            if found:
                return text.letters[found[0]]
        stub = SimpleTexBObject(r"+", **self.tex_kwargs)
        return stub.letters[0]

    def _fly_mover(self, source, move, sign_letter, eq_x, b, T,
                   f_lift, f_land, f_hold, f_sink, lift, highlight_color):
        """Choreograph one term's flight across the ``=`` (see :meth:`_step_add_subtract`)."""
        FR = FRAME_RATE
        glyphs = sorted(move['glyphs'],
                        key=lambda i: source.letters[i].ref_obj.location[0])
        starts = [Vector(source.letters[i].ref_obj.location) for i in glyphs]
        group_x = float(np.mean([s.x for s in starts]))
        land_x, land_y = move['land']
        dx = land_x - group_x

        def F(frac):
            return (b + frac * T) * FR

        # fraction of the traverse at which the term centre crosses the '='
        s_cross = 0.5 if abs(dx) < 1e-6 else (eq_x - group_x) / dx
        s_cross = min(max(s_cross, 0.0), 1.0)
        f_cross = f_lift + s_cross * (f_land - f_lift)

        def fly_location(obj, start):
            for frame, loc in ((F(0), start),
                               (F(f_lift), Vector((start.x, start.y + lift, start.z))),
                               (F(f_land), Vector((start.x + dx, start.y + lift, start.z))),
                               (F(f_hold), Vector((start.x + dx, start.y + lift, start.z))),
                               (F(f_sink), Vector((start.x + dx, land_y, start.z)))):
                obj.location = loc
                ibpy.insert_keyframe(obj, "location", frame=frame)

        def fly_scale(obj, base, keys):
            for frame, factor in keys:
                obj.scale = [factor * s for s in base]
                ibpy.insert_keyframe(obj, "scale", frame=frame)

        # the number/term itself: hide the original, fly a copy, shrink at the end
        copies = []
        for i, start in zip(glyphs, starts):
            orig = source.letters[i].ref_obj
            ibpy.insert_keyframe(orig, "scale", frame=F(0) - 1)
            orig.scale = (0, 0, 0)
            ibpy.insert_keyframe(orig, "scale", frame=F(0))

            copy = source.letters[i].copy(clear_animation_data=True)
            source.copies_of_letters.append(copy)
            copy.appear(begin_time=b, transition_time=0, clear_data=True)
            base = list(copy.ref_obj.scale)
            fly_location(copy.ref_obj, start)
            fly_scale(copy.ref_obj, base, ((F(0), 1), (F(f_hold), 1), (F(f_sink), 0)))
            copies.append(copy)

        # an attached leading operator (e.g. the '+' of a second term) flies
        # along with the group and shrinks out at the crossing, cross-fading
        # into the '-' that grows in there
        op = move.get('op')
        if op is not None:
            op_start = Vector(source.letters[op].ref_obj.location)
            orig = source.letters[op].ref_obj
            ibpy.insert_keyframe(orig, "scale", frame=F(0) - 1)
            orig.scale = (0, 0, 0)
            ibpy.insert_keyframe(orig, "scale", frame=F(0))
            op_copy = source.letters[op].copy(clear_animation_data=True)
            source.copies_of_letters.append(op_copy)
            op_copy.appear(begin_time=b, transition_time=0, clear_data=True)
            op_base = list(op_copy.ref_obj.scale)
            fly_location(op_copy.ref_obj, op_start)
            fly_scale(op_copy.ref_obj, op_base,
                      ((F(0), 1), (F(f_cross) - 2, 1), (F(f_cross) + 2, 0)))

        # the leading sign ('-' or, for a mover that started negative, '+')
        # that grows in as the term crosses the '='
        xs = sorted(s.x for s in starts)
        gap = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.35
        sign_start = Vector((xs[0] - gap, starts[0].y, starts[0].z))
        sign = sign_letter.copy(clear_animation_data=True)
        source.copies_of_letters.append(sign)
        sign.appear(begin_time=b, transition_time=0, clear_data=True)
        # move it into the flying term's frame (the borrowed glyph may live in
        # the target line, whose origin differs after '=' alignment)
        sign.ref_obj.parent = source.ref_obj
        sign.ref_obj.matrix_parent_inverse = \
            source.letters[glyphs[0]].ref_obj.matrix_parent_inverse.copy()
        sign_base = list(sign.ref_obj.scale)
        fly_location(sign.ref_obj, sign_start)
        fly_scale(sign.ref_obj, sign_base,
                  ((F(0), 0), (F(f_cross) - 1, 0), (F(f_cross) + 3, 1),
                   (F(f_hold), 1), (F(f_sink), 0)))

        # both terms flash on arrival; the movers keep the colour as they sink
        for copy in copies + [sign]:
            copy.change_color(highlight_color, begin_time=b + f_land * T,
                              transition_time=0.06 * T)

    # ------------------------------------------------------------------
    # swap: carry a term to the other side of the '=' sign
    # ------------------------------------------------------------------

    def _step_swap(self, source, tex, user_map, begin_time, transition_time,
                   lift=1.0):
        """Carry terms across the ``=`` sign without merging them.

        Every ``map`` entry whose source term and target term sit on opposite
        sides of the ``=`` is a *mover*: the term

        1. lifts above the equation,
        2. travels across the ``=`` while its leading sign **morphs** into the
           opposite sign (the ``-`` of ``-a_{n-1}`` reshapes into the ``+`` of
           ``+a_{n-1}``) -- the sign is a matched shape-key pair, not a
           cross-fade,
        3. sinks into its own slot on the other side, where it hot-swaps to
           the pristine target letters.

        Unlike ``add_subtract`` nothing merges or shrinks away: the term
        survives the crossing intact.  Map entries that stay on their side
        (and every unmentioned glyph) morph in place over the full transition,
        so the line reflows around the travelling term.  The line keeps its
        slot; the derivation does not grow a new row.

        A term whose sign has no partner glyph on the other side keeps it
        attached: the sign flies along with the term and reshapes into its
        opposite during the traverse, then is absorbed as the term sinks
        (``-a_{n-1}^2`` leaves as an implicit ``+a_{n-1}^2``).  A term that
        acquires a sign it never had grows the fresh sign as it crosses the
        ``=`` (as in ``add_subtract``).  ``lift`` may be a list with one
        height per moving term (ordered left-to-right on the source line),
        so terms crossing in opposite directions pass at different heights.
        """
        T, b, FR = float(transition_time), float(begin_time), FRAME_RATE
        f_lift, f_land = 0.25, 0.8  # rise until f_lift, traverse, then sink

        eq = self.align_char or '='
        eq_x = source.letters[source.find_letters(eq)[0]].ref_obj.location[0]
        tex_eq_x = tex.letters[tex.find_letters(eq)[0]].ref_obj.location[0]

        src_sigs, tgt_sigs = self._signatures(source), self._signatures(tex)

        # --- per-glyph pins for every mapped term ------------------------
        items = user_map.items() if isinstance(user_map, dict) else list(user_map or [])
        pins, entries = [], []
        term_src, term_tgt = set(), set()
        for src_spec, tgt_spec in items:
            if src_spec is None or tgt_spec is None:
                raise ValueError("swap map entries need a source and a target")
            src_idx = source.find_letters(src_spec)
            tgt_idx = tex.find_letters(tgt_spec)
            local = plan_transition([src_sigs[i] for i in src_idx],
                                    [tgt_sigs[j] for j in tgt_idx])
            pins.extend(local.pairs)
            term_src.update(src_idx)
            term_tgt.update(tgt_idx)
            entries.append((src_idx, tgt_idx, local.pairs))

        claimed_src, claimed_tgt = set(), set()

        def leading_op(text, term_idx, term_glyphs, eq_pos, claimed):
            """The +/- operator glyph adjacent on the term's left, if any.

            Glyphs inside a mapped term are never operators (the subscript
            '+' of ``a_{n+1}`` must not be picked up), an operator across
            the ``=`` is never a term's sign (on a narrow line the glyph
            just left of a term may belong to the other side), and an
            operator already claimed by another mover is taken.
            """
            ops = []
            for spec in ('+', '-'):
                try:
                    ops.extend(text.find_letters(spec + '@all'))
                except Exception:
                    pass
            min_x = min(text.letters[i].ref_obj.location[0] for i in term_idx)
            right_side = min_x > eq_pos
            cand = [o for o in ops if o not in term_glyphs and o not in claimed
                    and (text.letters[o].ref_obj.location[0] > eq_pos) == right_side
                    and 0 < min_x - text.letters[o].ref_obj.location[0] < 1.0]
            return (max(cand, key=lambda o: text.letters[o].ref_obj.location[0])
                    if cand else None)

        # --- movers: map entries that change sides; pair their signs -----
        movers = []  # per mover: the glyph pairs that fly + its odd signs
        for src_idx, tgt_idx, term_pairs in entries:
            src_centre = float(np.mean(
                [source.letters[i].ref_obj.location[0] for i in src_idx]))
            tgt_centre = float(np.mean(
                [tex.letters[j].ref_obj.location[0] for j in tgt_idx]))
            if (src_centre > eq_x) == (tgt_centre > tex_eq_x):
                continue  # stays on its side -- plain in-place morph
            move = {'pairs': list(term_pairs), 'shed': None, 'grown': None}
            src_op = leading_op(source, src_idx, term_src, eq_x, claimed_src)
            tgt_op = leading_op(tex, tgt_idx, term_tgt, tex_eq_x, claimed_tgt)
            if src_op is not None:
                claimed_src.add(src_op)
            if tgt_op is not None:
                claimed_tgt.add(tgt_op)
            if src_op is not None and tgt_op is not None:
                pins.append((src_op, tgt_op))  # '-' reshapes into '+'
                move['pairs'].append((src_op, tgt_op))
            elif src_op is not None:
                # the sign sticks to the term and flips while crossing, then
                # is absorbed (the term leads its new side, written bare)
                pins.append((src_op, None))
                move['shed'] = src_op
            elif tgt_op is not None:
                # the term acquires a sign: a fresh one grows at the crossing
                pins.append((None, tgt_op))
                move['grown'] = tgt_op
            movers.append(move)

        plan = plan_transition(src_sigs, tgt_sigs, mapping=pins or None, auto=True)
        # shed signs fly as morph copies and grown signs grow in mid-flight
        # (see _fly_swap_signs) -- take them out of the default treatment
        shed = {mv['shed'] for mv in movers if mv['shed'] is not None}
        grown = {mv['grown'] for mv in movers if mv['grown'] is not None}
        plan.vanish = [i for i in plan.vanish if i not in shed]
        plan.appear = [j for j in plan.appear if j not in grown]
        self.plans.append(plan)

        # in-place morph of the whole line: unmoved glyphs reflow, the
        # movers get their straight travel bent into a flight below
        morph = PairMorph(source, tex, plan)
        morph.animate(begin_time=b, transition_time=T)

        # --- bend the movers' travel into lift / traverse / sink ---------
        # order the terms left-to-right so a per-term 'lift' list lines up
        # with the way they read on the source line (as in add_subtract)
        movers.sort(key=lambda mv: np.mean(
            [source.letters[i].ref_obj.location[0] for i, _ in mv['pairs']]))
        if isinstance(lift, (list, tuple)):
            if len(lift) != len(movers):
                raise ValueError("lift has %d entries but there are %d moving terms"
                                 % (len(lift), len(movers)))
            lifts = list(lift)
        else:
            lifts = [lift] * len(movers)

        by_source = {source.letters.index(src_letter): copy
                     for copy, src_letter, _ in morph.morphers}
        depth_offset = Vector((0, 0, -0.001))  # as in PairMorph.animate
        for move, mv_lift in zip(movers, lifts):
            starts, ends = {}, {}
            for i, j in move['pairs']:
                starts[i] = Vector(source.letters[i].ref_obj.location)
                ends[i] = (morph.shift + Vector(tex.letters[j].ref_obj.location)
                           + depth_offset)
                copy = by_source.get(i)
                if copy is None:
                    continue
                for frac, loc in ((f_lift, Vector((starts[i].x, starts[i].y + mv_lift,
                                                   starts[i].z))),
                                  (f_land, Vector((ends[i].x, ends[i].y + mv_lift,
                                                   ends[i].z)))):
                    copy.location = loc
                    ibpy.insert_keyframe(copy, "location", frame=(b + frac * T) * FR)
            if move['shed'] is not None or move['grown'] is not None:
                self._fly_swap_signs(source, tex, move, starts, ends, morph.shift,
                                     eq_x, b, T, f_lift, f_land, mv_lift)

        return b + T

    def _fly_swap_signs(self, source, tex, move, starts, ends, shift,
                        eq_x, b, T, f_lift, f_land, lift):
        """Fly the signs of a swapping term that have no partner glyph.

        * ``move['shed']``: the term's leading sign does not exist on the
          other side (the term lands leading its new side, written bare).
          The sign sticks to the term: a shape-key morph copy flies along,
          reshapes into the opposite sign during the traverse and shrinks
          away as the term sinks into its slot.
        * ``move['grown']``: the term acquires a sign it never had.  A clone
          of the pristine target sign grows in as the term crosses the ``=``,
          flies the rest of the way and hot-swaps into the real letter on
          the end frame.

        ``starts``/``ends`` are the term glyphs' flight endpoints (in the
        source frame) computed by the mover loop.
        """
        FR = FRAME_RATE

        def F(frac):
            return (b + frac * T) * FR

        # the leftmost term glyph defines the sign slot and travel
        lead = min(starts, key=lambda i: starts[i].x)
        delta = ends[lead] - starts[lead]
        xs = sorted(s.x for s in starts.values())
        gap = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.35
        # fraction of the traverse at which the term centre crosses the '='
        group_x = float(np.mean([s.x for s in starts.values()]))
        dx = float(np.mean([e.x for e in ends.values()])) - group_x
        s_cross = 0.5 if abs(dx) < 1e-6 else (eq_x - group_x) / dx
        s_cross = min(max(s_cross, 0.0), 1.0)
        f_cross = f_lift + s_cross * (f_land - f_lift)

        def fly(obj, start, end):
            for frame, loc in ((F(0), start),
                               (F(f_lift), Vector((start.x, start.y + lift, start.z))),
                               (F(f_land), Vector((end.x, end.y + lift, end.z))),
                               (F(1.0), end)):
                obj.location = loc
                ibpy.insert_keyframe(obj, "location", frame=frame)

        if move['shed'] is not None:
            src_letter = source.letters[move['shed']]
            # hide the original; a morph copy flies in its place
            orig = src_letter.ref_obj
            base = list(orig.scale)
            ibpy.insert_keyframe(orig, "scale", frame=F(0) - 1)
            orig.scale = (0, 0, 0)
            ibpy.insert_keyframe(orig, "scale", frame=F(0))

            try:
                is_minus = move['shed'] in source.find_letters('-@all')
            except Exception:
                is_minus = True
            donor = (self._plus_letter(tex, source) if is_minus
                     else self._minus_letter(tex, source))
            splines_src, cyc_src = extract_glyph(src_letter)
            splines_tgt, cyc_tgt = extract_glyph(donor)
            snapshots, cyclic = compile_chain([splines_src, splines_tgt],
                                              cyclic_flags=[cyc_src, cyc_tgt])
            sign = make_curve_copy(src_letter.ref_obj)
            rebuild_curve(sign, snapshots, cyclic)
            sign.parent = src_letter.ref_obj.parent
            sign.rotation_euler = src_letter.ref_obj.rotation_euler
            ibpy.link(sign)

            start = Vector(src_letter.ref_obj.location)
            fly(sign, start, start + delta)
            for frame, factor in ((F(0) - 1, 0), (F(0), 1),
                                  (F(f_land), 1), (F(1.0), 0)):
                sign.scale = [factor * s for s in base]
                ibpy.insert_keyframe(sign, "scale", frame=frame)
            # reshape into the opposite sign while crossing the '='
            shape_keys = sign.data.shape_keys
            blocks = shape_keys.key_blocks
            shape_keys.eval_time = blocks[0].frame
            shape_keys.keyframe_insert(data_path='eval_time', frame=F(f_lift))
            shape_keys.eval_time = blocks[-1].frame
            shape_keys.keyframe_insert(data_path='eval_time', frame=F(f_land))
            shape_keys.eval_time = 0

        if move['grown'] is not None:
            tgt_letter = tex.letters[move['grown']]
            sign = tgt_letter.copy(clear_animation_data=True)
            source.copies_of_letters.append(sign)
            sign.appear(begin_time=b, transition_time=0, clear_data=True)
            # move it into the flying term's frame (the pristine glyph lives
            # in the target line, whose origin differs after '=' alignment)
            sign.ref_obj.parent = source.ref_obj
            sign.ref_obj.matrix_parent_inverse = \
                source.letters[lead].ref_obj.matrix_parent_inverse.copy()
            start = Vector((xs[0] - gap, starts[lead].y, starts[lead].z))
            end = (shift + Vector(tgt_letter.ref_obj.location)
                   + Vector((0, 0, -0.001)))
            fly(sign.ref_obj, start, end)
            base = list(sign.ref_obj.scale)
            for frame, factor in ((F(0), 0), (F(f_cross) - 1, 0),
                                  (F(f_cross) + 3, 1), (F(1.0), 1),
                                  (F(1.0) + 0.5, 0)):
                sign.ref_obj.scale = [factor * s for s in base]
                ibpy.insert_keyframe(sign.ref_obj, "scale", frame=frame)
            # the pristine target sign takes over on the end frame
            tex.write(letter_set=[move['grown']], begin_time=b + T,
                      transition_time=0)
