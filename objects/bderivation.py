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
* ``mode='add_subtract'``: an in-place step in which the right-hand terms
  of the ``map`` fly across the ``=`` sign, flip their sign and merge into
  the matching left-hand terms (e.g. ``... = 9 + 495x`` collected onto the
  left).  See :meth:`BDerivation._step_add_subtract`.
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
             align=True, fade_old=None, arc=0.3, stagger=0.5, lift=1.0,
             highlight=None, highlight_color='important', new_color=None,
             cancel=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
             **tex_overrides):
        """Advance the derivation to the next expression.

        :param expression: LaTeX of the next line
        :param mode: 'new_line' (letters fly to the next line), 'replace'
            (the line morphs in place) or 'add_subtract' (right-hand terms of
            the ``map`` fly across the ``=`` sign, flip sign and merge into
            their left-hand partners; see :meth:`_step_add_subtract`)
        :param lift: for ``mode='add_subtract'``, how far (in line units) a
            travelling term rises above the baseline as it crosses over; a
            single value applies to every term, or pass a list with one value
            per moving term (ordered left-to-right on the source line) to lift
            them to different heights
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
        self._place_line(tex, same_slot=(mode in ('replace', 'add_subtract')))
        if align:
            self._align(tex)

        if mode == 'add_subtract':
            self._step_add_subtract(source, tex, map, begin_time, transition_time,
                                    lift=lift, highlight_color=highlight_color)
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
            raise ValueError("unknown mode %r (use 'new_line', 'replace' or "
                             "'add_subtract')" % mode)

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
