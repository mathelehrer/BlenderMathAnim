"""
BMorphText -- a text object whose expression can morph through a chain of
LaTeX expressions.

Replaces both older morphing frameworks:

* the geometry-nodes classes ``MorphText`` / ``MorphText2`` / ``MorphText3``
  (``objects/text.py``), whose new splines always grew out of the text origin
  and whose manual mapping (``src_fix``/``target_fix`` switch lists) was
  hard to use, and
* the shape-key morphing of :class:`objects.tex_bobject.SimpleTexBObject`
  (``add_to_morph_chain`` / ``perform_morphing``), whose automatic letter
  matching was fragile and required an explicit ``perform_morphing`` call.

Key properties
--------------
* **Fully automatic matching**: identical glyphs are matched by a
  longest-common-subsequence on shape fingerprints, remaining letters are
  paired by shape similarity, everything else vanishes/appears *in place*.
* **Precise manual control**: a mapping per transition can pin any letter to
  any letter(s), including one-to-many (splits) and many-to-one (merges);
  the rest is still auto-matched (or not, with ``auto=False``).
* **No origin growth**: appearing letters grow at their own location and
  appearing splines emerge from their own centroid (see
  :func:`objects.morph_planning.compile_chain`).
* **No second phase**: everything is precomputed at construction; ``write``
  and ``morph`` just insert keyframes.  No ``perform_morphing`` needed.

Example
-------
>>> eqn = BMorphText(r"(x+y)^2 = 9\\cdot (1+x\\cdot y)",
...                  r"(x+55)^2 = 9\\cdot (1+x\\cdot 55)",
...                  color="text", aligned="center")
>>> t0 = 0.5 + eqn.write(begin_time=0, transition_time=0.5)
>>> t0 = 0.5 + eqn.morph(begin_time=t0, transition_time=0.5)

with manual control (letter 3 splits into the two 5s):

>>> eqn = BMorphText(expr1, expr2, mapping={3: (3, 4)})
"""

import numpy as np

from interface import ibpy
from objects.morph_planning import compile_chain, glyph_signature, plan_transition
from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


# ---------------------------------------------------------------------------
# reusable single-letter morph machinery (also used by objects/bderivation.py)
# ---------------------------------------------------------------------------

def extract_glyph(letter):
    """Read the bezier splines of a letter into numpy arrays.

    :return: (splines, cyclic_flags) with splines in the letter's local
        frame, each an array (n,3,3) of (co, handle_left, handle_right)
    """
    splines, cyclic = [], []
    for spline in letter.ref_obj.data.splines:
        if len(spline.bezier_points) == 0:
            continue
        arr = np.array([[list(p.co), list(p.handle_left), list(p.handle_right)]
                        for p in spline.bezier_points], dtype=float)
        splines.append(arr)
        cyclic.append(bool(spline.use_cyclic_u))
    return splines, cyclic


def make_curve_copy(donor):
    """Independent copy of a letter curve (own data, own materials)."""
    curve = donor.copy()
    curve.data = donor.data.copy()
    curve.animation_data_clear()
    curve.name = donor.name + "_morph_copy"
    for slot in curve.material_slots:
        if slot.material is not None:
            slot.material = slot.material.copy()
    return curve


def rebuild_curve(curve, snapshots, cyclic):
    """Rewrite the curve's splines to the compiled layout and add one
    absolute shape key block per snapshot."""
    data = curve.data
    resolution = data.splines[0].resolution_u if len(data.splines) else 12
    data.splines.clear()
    for arr, cyc in zip(snapshots[0], cyclic):
        spline = data.splines.new('BEZIER')
        spline.bezier_points.add(len(arr) - 1)
        spline.use_cyclic_u = cyc
        spline.resolution_u = resolution
        for point, (co, hl, hr) in zip(spline.bezier_points, arr):
            point.handle_left_type = 'FREE'
            point.handle_right_type = 'FREE'
            point.co = co
            point.handle_left = hl
            point.handle_right = hr

    # absolute shape keys: block k = snapshot k, animated via eval_time
    basis = curve.shape_key_add(name="step_0", from_mix=False)
    basis.interpolation = 'KEY_LINEAR'
    curve.data.shape_keys.use_relative = False
    for k in range(1, len(snapshots)):
        block = curve.shape_key_add(name="step_%d" % k, from_mix=False)
        block.interpolation = 'KEY_LINEAR'
        index = 0
        for arr in snapshots[k]:
            for co, hl, hr in arr:
                block.data[index].co = co
                block.data[index].handle_left = hl
                block.data[index].handle_right = hr
                index += 1
    curve.data.shape_keys.eval_time = 0


class BMorphText(SimpleTexBObject):
    """A :class:`SimpleTexBObject` that morphs through a chain of expressions.

    The object *is* the first expression (so all the usual animation methods
    -- ``write``, ``move_to``, ``rotate``, ... -- keep working); every call to
    :meth:`morph` advances it to the next expression of the chain.
    """

    def __init__(self, *expressions, **kwargs):
        """
        :param expressions: two or more LaTeX strings (a single list/tuple
            argument is unpacked as well)
        :param kwargs: forwarded to :class:`SimpleTexBObject` for every
            expression in the chain.  Additional keys:

            * ``mapping``: manual mapping for the (single) transition of a
              two-expression chain.  Dict ``{src: tgt}`` where ``tgt`` may be
              an int, a tuple of ints (split) or ``None`` (vanish in place);
              or an iterable of ``(src, tgt)`` pairs where ``src=None``
              forces a fresh appearance of ``tgt``.
            * ``mappings``: list with one such mapping (or ``None``) per
              transition for longer chains.
            * ``auto``: if ``False``, only manually mapped letters morph;
              everything else vanishes/appears.  Defaults to ``True``.
            * ``auto_threshold``: optional cost cutoff for the similarity
              pass of the automatic matcher; ``None`` (default) morphs every
              leftover letter it can pair up.
            * ``colors``: optional list (one entry per expression) of color
              specs (string or list of strings per letter); defaults to the
              common ``color`` kwarg.  Color changes of matched letters are
              animated during the morph.
        """
        if len(expressions) == 1 and isinstance(expressions[0], (list, tuple)):
            expressions = tuple(expressions[0])
        if len(expressions) < 2:
            raise ValueError("BMorphText needs at least two expressions")
        self.expressions = [str(e) for e in expressions]

        mappings = kwargs.pop('mappings', None)
        mapping = kwargs.pop('mapping', None)
        if mapping is not None:
            if mappings is not None:
                raise ValueError("pass either 'mapping' or 'mappings', not both")
            mappings = [mapping]
        if mappings is None:
            mappings = [None] * (len(self.expressions) - 1)
        if len(mappings) != len(self.expressions) - 1:
            raise ValueError("need one mapping per transition (%d), got %d"
                             % (len(self.expressions) - 1, len(mappings)))
        self.mappings = mappings
        self.auto = kwargs.pop('auto', True)
        self.auto_threshold = kwargs.pop('auto_threshold', None)

        colors = kwargs.pop('colors', None)
        base_name = kwargs.get('name', 'BMorphText')

        # the object itself is the first expression
        first_kwargs = dict(kwargs)
        if colors is not None:
            first_kwargs['color'] = colors[0]
        super().__init__(self.expressions[0], **first_kwargs)

        # hidden donor texts for the remaining expressions (never linked)
        self.targets = []
        for k, expression in enumerate(self.expressions[1:], start=1):
            target_kwargs = dict(kwargs)
            target_kwargs['name'] = base_name + "_target_" + str(k)
            if colors is not None:
                target_kwargs['color'] = colors[k]
            self.targets.append(SimpleTexBObject(expression, **target_kwargs))

        self.current_transition = 0
        self.plans = []
        self.chains = []
        self._plan()
        self._build_chains()
        self._materialize_chains()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @property
    def texts(self):
        """All expressions of the chain as SimpleTexBObjects (self first)."""
        return [self] + self.targets

    def number_of_transitions(self):
        return len(self.expressions) - 1

    # promoted to module level (see extract_glyph above); kept as aliases
    _extract_glyph = staticmethod(extract_glyph)
    _rebuild_curve = staticmethod(rebuild_curve)

    def _plan(self):
        """Compute glyph signatures and one MorphPlan per transition."""
        self._glyphs = []      # per expression: list of (splines, cyclic)
        self._signatures = []  # per expression: list of GlyphSig
        for text in self.texts:
            glyphs, sigs = [], []
            for i, letter in enumerate(text.letters):
                splines, cyclic = self._extract_glyph(letter)
                glyphs.append((splines, cyclic))
                sigs.append(glyph_signature(i, splines, location=list(letter.ref_obj.location)))
            self._glyphs.append(glyphs)
            self._signatures.append(sigs)

        for t in range(self.number_of_transitions()):
            plan = plan_transition(self._signatures[t], self._signatures[t + 1],
                                   mapping=self.mappings[t], auto=self.auto,
                                   auto_threshold=self.auto_threshold)
            self.plans.append(plan)

    def _build_chains(self):
        """Trace every physical curve through the expression chain."""
        texts = self.texts
        # active: target glyph index -> list of chains currently showing it;
        # the first entry is the primary (used as donor for splits)
        active = {}
        for i, letter in enumerate(self.letters):
            chain = _Chain(first_step=0, letters=[letter])
            self.chains.append(chain)
            active[i] = [chain]

        for t, plan in enumerate(self.plans):
            next_letters = texts[t + 1].letters
            by_src = {}
            for i, j in plan.pairs:
                by_src.setdefault(i, []).append(j)

            new_active = {}
            for i, tgts in by_src.items():
                chains_i = active.get(i, [])
                if not chains_i:
                    continue
                history = list(chains_i[0].letters)  # before extending
                for chain in chains_i:
                    chain.letters.append(next_letters[tgts[0]])
                    new_active.setdefault(tgts[0], []).append(chain)
                for j in tgts[1:]:
                    clone = _Chain(first_step=chains_i[0].first_step,
                                   letters=history + [next_letters[j]],
                                   split_step=t)
                    self.chains.append(clone)
                    new_active.setdefault(j, []).append(clone)

            for i in plan.vanish:
                for chain in active.get(i, []):
                    chain.vanish_step = t

            for j in plan.appear:
                chain = _Chain(first_step=t + 1, letters=[next_letters[j]])
                self.chains.append(chain)
                new_active.setdefault(j, []).append(chain)

            active = new_active

    def _materialize_chains(self):
        """Create/rebuild the blender curves and bake all shape keys."""
        texts = self.texts
        for chain in self.chains:
            # collect the glyph geometry the chain passes through
            geoms, cyclics, colors = [], [], []
            for step_offset, letter in enumerate(chain.letters):
                k = chain.first_step + step_offset
                index = texts[k].letters.index(letter)
                splines, cyclic = self._glyphs[k][index]
                geoms.append(splines)
                cyclics.append(cyclic)
                colors.append(texts[k].color_map.get(letter))
            chain.colors = colors

            snapshots, cyclic = compile_chain(geoms, cyclic_flags=cyclics)

            is_primary = chain.first_step == 0 and chain.split_step is None
            if is_primary:
                chain.curve = chain.letters[0].ref_obj
            else:
                chain.curve = self._make_curve_copy(chain.letters[0].ref_obj)

            self._rebuild_curve(chain.curve, snapshots, cyclic)

            if not is_primary:
                # copies live in the container of this (the first) expression;
                # since all texts share the same container transform, local
                # letter coordinates carry over directly
                donor = chain.letters[0].ref_obj
                chain.curve.parent = self.ref_obj
                chain.curve.location = donor.location
                chain.curve.rotation_euler = donor.rotation_euler
                ibpy.link(chain.curve)
                # invisible until its split/appear transition
                chain.curve.scale = (0, 0, 0)
                ibpy.insert_keyframe(chain.curve, "scale", frame=0)

            # color transitions
            for step_offset in range(len(chain.letters) - 1):
                c1, c2 = colors[step_offset], colors[step_offset + 1]
                if c1 != c2 and c2 is not None and chain.curve.material_slots:
                    material = chain.curve.material_slots[0].material
                    dialer = ibpy.create_color_mixing(material, c1, c2)
                    dialer.default_value = 0
                    chain.dialers[chain.first_step + step_offset] = dialer

    def _make_curve_copy(self, donor):
        return make_curve_copy(donor)

    # ------------------------------------------------------------------
    # animation
    # ------------------------------------------------------------------

    def morph(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, step=None):
        """Morph to the next expression of the chain.

        :param step: optional explicit transition index (0-based); by default
            an internal cursor advances one transition per call
        :return: begin_time + transition_time
        """
        if step is None:
            step = self.current_transition
            self.current_transition += 1
        if not 0 <= step < self.number_of_transitions():
            raise IndexError("transition %d out of range (chain has %d transitions)"
                             % (step, self.number_of_transitions()))
        self._animate_transition(step, begin_time, transition_time, reverse=False)
        return begin_time + transition_time

    def unmorph(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, step=None):
        """Play a transition backwards (default: the one played last)."""
        if step is None:
            self.current_transition -= 1
            step = self.current_transition
        if not 0 <= step < self.number_of_transitions():
            raise IndexError("transition %d out of range (chain has %d transitions)"
                             % (step, self.number_of_transitions()))
        self._animate_transition(step, begin_time, transition_time, reverse=True)
        return begin_time + transition_time

    def _animate_transition(self, t, begin_time, transition_time, reverse=False):
        begin_frame = begin_time * FRAME_RATE
        end_frame = (begin_time + transition_time) * FRAME_RATE

        for chain in self.chains:
            offset = t - chain.first_step  # snapshot index before the transition

            if chain.vanish_step == t:
                # shrink away in place (or regrow when reversed)
                start, end = (1, 0) if not reverse else (0, 1)
                self._keyframe_scale(chain.curve, chain.letters[-1].ref_obj.scale,
                                     start, end, begin_frame, end_frame)
                continue

            participates = 0 <= offset < len(chain.letters) - 1
            if not participates:
                if chain.first_step == t + 1:
                    # letter appears: grow at its own location
                    start, end = (0, 1) if not reverse else (1, 0)
                    self._keyframe_scale(chain.curve, chain.letters[0].ref_obj.scale,
                                         start, end, begin_frame, end_frame)
                continue

            if chain.split_step == t:
                # split copies pop up on top of their primary right at the
                # start of the transition (they are identical at that moment)
                scale = chain.letters[offset].ref_obj.scale
                if not reverse:
                    self._keyframe_scale(chain.curve, scale, 0, 1, begin_frame - 1, begin_frame)
                else:
                    self._keyframe_scale(chain.curve, scale, 1, 0, end_frame, end_frame + 1)

            source = chain.letters[offset].ref_obj
            target = chain.letters[offset + 1].ref_obj
            if reverse:
                source, target = target, source

            # transform keyframes (location, rotation, scale of the letter)
            curve = chain.curve
            ibpy.insert_keyframe(curve, "location", frame=begin_frame)
            curve.location = target.location
            ibpy.insert_keyframe(curve, "location", frame=end_frame)

            ibpy.insert_keyframe(curve, "rotation_euler", frame=begin_frame)
            curve.rotation_euler = target.rotation_euler
            ibpy.insert_keyframe(curve, "rotation_euler", frame=end_frame)

            # shape morph via eval_time between the absolute key blocks
            key_blocks = curve.data.shape_keys.key_blocks
            from_block, to_block = offset, offset + 1
            if reverse:
                from_block, to_block = to_block, from_block
            shape_keys = curve.data.shape_keys
            shape_keys.eval_time = key_blocks[from_block].frame
            shape_keys.keyframe_insert(data_path='eval_time', frame=begin_frame)
            shape_keys.eval_time = key_blocks[to_block].frame
            shape_keys.keyframe_insert(data_path='eval_time', frame=end_frame)
            shape_keys.eval_time = 0

            # color transition
            dialer = chain.dialers.get(t)
            if dialer is not None:
                dialer.default_value = 1 if reverse else 0
                ibpy.insert_keyframe(dialer, 'default_value', frame=begin_frame)
                dialer.default_value = 0 if reverse else 1
                ibpy.insert_keyframe(dialer, 'default_value', frame=end_frame)

    @staticmethod
    def _keyframe_scale(curve, base_scale, start_factor, end_factor, begin_frame, end_frame):
        base = list(base_scale)
        curve.scale = [start_factor * s for s in base]
        ibpy.insert_keyframe(curve, "scale", frame=begin_frame)
        curve.scale = [end_factor * s for s in base]
        ibpy.insert_keyframe(curve, "scale", frame=end_frame)


class _Chain:
    """One physical curve and the letters it represents along the chain.

    ``letters[k]`` is the letter of expression ``first_step + k`` the curve
    shows after ``k`` of its transitions.  A chain with ``first_step > 0``
    (and no ``split_step``) appears fresh; a chain with a ``split_step`` is a
    copy that branches off its primary during that transition; a chain with a
    ``vanish_step`` shrinks away during that transition and its ``letters``
    end there.
    """

    def __init__(self, first_step, letters, split_step=None):
        self.first_step = first_step
        self.letters = letters
        self.split_step = split_step
        self.vanish_step = None
        self.curve = None
        self.colors = []
        self.dialers = {}


class PairMorph:
    """One-shot in-place morph between two already-rendered tex objects.

    Unlike :class:`BMorphText` (which needs its whole expression chain at
    construction because shape-key point counts are fixed per curve), a
    PairMorph handles exactly one transition on **copies** of the source
    letters, so it can be created incrementally -- this is what powers
    ``BDerivation.step(mode='replace')``:

    * matched letters: a copy of the source letter gets two shape-key blocks
      (source glyph, target glyph) and morphs while its location travels to
      the target letter; the original letter hides the moment the copy
      appears,
    * vanishing letters shrink away in place (the original letter itself),
    * appearing letters are written in during the second half,
    * at the end frame the morph copies vanish and the pristine target
      letters are written at full visibility on the same frame (hot swap) --
      the geometry is identical at that moment, so the crossover is
      invisible and the target text starts its life with clean,
      un-shape-keyed letters.
    """

    def __init__(self, source, target, plan):
        """
        :param source: SimpleTexBObject currently on screen
        :param target: hidden SimpleTexBObject at the same slot (same parent)
        :param plan: :class:`objects.morph_planning.MorphPlan` between them
        """
        self.source = source
        self.target = target
        self.plan = plan

        scale = source.ref_obj.scale
        self.shift = ibpy.get_location(target) - ibpy.get_location(source)
        for i in range(3):
            self.shift[i] /= scale[i]

        self.morphers = []  # (copy curve, source letter, target letter)
        for i, j in plan.pairs:
            src_letter = source.letters[i]
            tgt_letter = target.letters[j]
            splines_src, cyclic_src = extract_glyph(src_letter)
            splines_tgt, cyclic_tgt = extract_glyph(tgt_letter)
            snapshots, cyclic = compile_chain([splines_src, splines_tgt],
                                              cyclic_flags=[cyclic_src, cyclic_tgt])
            copy = make_curve_copy(src_letter.ref_obj)
            rebuild_curve(copy, snapshots, cyclic)
            copy.parent = src_letter.ref_obj.parent
            copy.location = src_letter.ref_obj.location
            copy.rotation_euler = src_letter.ref_obj.rotation_euler
            ibpy.link(copy)
            copy.scale = (0, 0, 0)
            ibpy.insert_keyframe(copy, "scale", frame=0)
            self.morphers.append((copy, src_letter, tgt_letter))

    def animate(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
                depth_offset=(0, 0, -0.001)):
        """Insert all keyframes for the transition.

        :return: begin_time + transition_time
        """
        begin_frame = begin_time * FRAME_RATE
        end_frame = (begin_time + transition_time) * FRAME_RATE
        end_time = begin_time + transition_time
        depth_offset = ibpy.Vector(depth_offset)

        matched_sources = {i for i, _ in self.plan.pairs}
        matched_targets = sorted({j for _, j in self.plan.pairs})

        # matched source letters hide the moment their morph copy appears
        for i in matched_sources:
            obj = self.source.letters[i].ref_obj
            base = list(obj.scale)
            ibpy.insert_keyframe(obj, "scale", frame=begin_frame - 1)
            obj.scale = (0, 0, 0)
            ibpy.insert_keyframe(obj, "scale", frame=begin_frame)
            obj.scale = base

        # vanishing letters shrink away in place
        for i in self.plan.vanish:
            obj = self.source.letters[i].ref_obj
            base = list(obj.scale)
            ibpy.insert_keyframe(obj, "scale", frame=begin_frame)
            obj.scale = (0, 0, 0)
            ibpy.insert_keyframe(obj, "scale", frame=end_frame)

        for copy, src_letter, tgt_letter in self.morphers:
            base = list(src_letter.ref_obj.scale)
            # pop on at the begin frame ...
            copy.scale = (0, 0, 0)
            ibpy.insert_keyframe(copy, "scale", frame=begin_frame - 1)
            copy.scale = base
            ibpy.insert_keyframe(copy, "scale", frame=begin_frame)
            # ... hold until the swap, then pop off
            ibpy.insert_keyframe(copy, "scale", frame=end_frame)
            copy.scale = (0, 0, 0)
            ibpy.insert_keyframe(copy, "scale", frame=end_frame + 0.5)

            # shape morph
            shape_keys = copy.data.shape_keys
            blocks = shape_keys.key_blocks
            shape_keys.eval_time = blocks[0].frame
            shape_keys.keyframe_insert(data_path='eval_time', frame=begin_frame)
            shape_keys.eval_time = blocks[-1].frame
            shape_keys.keyframe_insert(data_path='eval_time', frame=end_frame)
            shape_keys.eval_time = 0

            # travel to the target letter (slightly behind it)
            copy.location = src_letter.ref_obj.location
            ibpy.insert_keyframe(copy, "location", frame=begin_frame)
            copy.location = self.shift + tgt_letter.ref_obj.location + depth_offset
            ibpy.insert_keyframe(copy, "location", frame=end_frame)

        # appearing letters are written in during the second half
        if self.plan.appear:
            self.target.write(letter_set=list(self.plan.appear),
                              begin_time=begin_time + 0.5 * transition_time,
                              transition_time=0.5 * transition_time)

        # hot swap: pristine target letters take over on the end frame
        if matched_targets:
            self.target.write(letter_set=matched_targets, begin_time=end_time,
                              transition_time=0)
        return end_time
