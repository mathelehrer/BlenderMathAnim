# BDerivation — animated equation transformations

`objects/bderivation.py` provides **BDerivation**, a tool for animating algebraic
derivations: every step of the derivation is a LaTeX string, letters are addressed
by **substring** instead of hand-counted glyph indices, and the letter
correspondence between consecutive steps is computed automatically. It automates
the derivation idiom used throughout the videos (stacked lines on a display,
`move_copy_to` with index lists, glue glyphs written in — see
`video_bwm/scene_bwm.py::addition_formal` for the manual version).

Related tools:

| Tool | Use for |
|---|---|
| `BDerivation` | multi-step derivations; new lines and/or in-place steps, choreography |
| `BMorphText` (`objects/bmorph_text.py`) | a single text that morphs through a fixed chain of expressions in place |
| `SimpleTexBObject.find_letters` | substring → letter indices anywhere (also outside BDerivation) |
| `objects/choreography.py` | raw flight/highlight/cancel primitives for custom animations |

---

## Quick start

```python
from objects.bderivation import BDerivation

d = BDerivation(r"(x+55)^2 = 9\cdot (1+x\cdot 55)", name="AlgebraDerivation")

# expand both sides onto a new line (letters fly down on arcs)
t0 = 0.5 + d.step(r"x^2+2\cdot 55\cdot x+55^2 = 9+9\cdot 55\cdot x",
                  map={"(x+55)^2": r"x^2+2\cdot 55\cdot x+55^2",
                       r"9\cdot (1+x\cdot 55)": r"9+9\cdot 55\cdot x"},
                  begin_time=t0, transition_time=1)

# evaluate the products in place on the same line
t0 = 0.5 + d.step(r"x^2+110\cdot x+3025 = 9+495\cdot x", mode='replace',
                  map={r"2\cdot 55": "110", "55^2": "3025", r"9\cdot 55": "495"},
                  begin_time=t0, transition_time=1)

# collect on a new line; two terms MERGE into one target
t0 = 0.5 + d.step(r"x^2-385\cdot x+3016 = 0",
                  map={r"110\cdot x": r"-385\cdot x",
                       r"495\cdot x": r"-385\cdot x",   # same target = merge
                       "3025": "3016",
                       "9@0": "3016"},                   # '9' is ambiguous -> pick occurrence 0
                  begin_time=t0, transition_time=1)
```

(This is the working `intro_algebra_overlay` continuation in
`video_hat_tile/scene_hat_tile.py`.)

The first line of a BDerivation can be shown with `d.write(...)`, **or** left
unwritten so it acts as an invisible anchor under an existing text object (as
above, where a `BMorphText` already shows the same expression at the same
location — the flights then appear to start from the visible text).

---

## Addressing letters by substring

Everywhere a substring is accepted, it refers to a fragment of the **LaTeX
source** of the respective line. Matching is token-based and
whitespace-insensitive (`"x\cdot y"` matches `x \cdot y`).

| Syntax | Meaning |
|---|---|
| `"2xy"` | the unique occurrence; **raises** `AmbiguousTokenError` if it occurs more than once |
| `"y@1"` | occurrence 1 (0-based, ordered by position in the source) |
| `"y@all"` | all occurrences |
| `("y", 1)` | tuple form of the same |

Rules and behavior:

* A match must be brace-balanced (`"}+{"` never matches) and may not end on a
  bare `^`/`_`.
* Script arguments are handled: mapping `"2"` in `x^2` works — the argument is
  braced on the fly (`x^{...2...}`), which never changes spacing.
* Substrings that produce **no glyphs** (pure spacing like `\,`) raise an error.
* Digits inside larger numbers count as occurrences: in `"= 9+495"`, plain
  `"9"` is ambiguous (`9@0` is the standalone one, `9@1` the digit in 495).

**How it works** (`objects/token_mapping.py`): the expression is recompiled with
the target substring wrapped in raw dvi color specials
(`\special{color push rgb k/255 0 0} ... \special{color pop}`). Specials are
metric-free, so glyph positions stay bit-identical; dvisvgm turns them into
`fill` colors in the SVG, which is parsed directly (pure python) and matched to
the live Blender letters geometrically. The colored and uncolored renders are
compared — if a special ever broke a ligature or kern (only possible inside
`\text{...}`), a `TokenMappingError` names the substring instead of silently
mis-mapping. Results are cached both on disk (the ordinary tex/svg cache) and
per object.

### `SimpleTexBObject.find_letters(substring, occurrence=None) -> list[int]`

The standalone entry point, usable in any scene:

```python
line1.move_copy_to(target=line2,
                   src_letter_indices=line1.find_letters(r"b\,i@0"),
                   target_letter_indices=line2.find_letters("b"))
```

---

## Class reference

### `BDerivation(first_expression, **kwargs)`

Coordinates one `SimpleTexBObject` per derivation line. **Not** itself a
BObject; access the current line via `d.current` (a full SimpleTexBObject) for
moves, color changes etc.

| Parameter | Default | Description |
|---|---|---|
| `first_expression` | — | LaTeX of the first line |
| `display` | `None` | optional `Display`; lines are placed with `add_text_in` on rows `line`, `line+line_step`, ... |
| `line`, `line_step` | `1`, `1` | first display row and row increment |
| `indent`, `column`, `scale` | `0`, `1`, `0.7` | display placement |
| `align_char` | `'='` | character aligned vertically between consecutive lines (first occurrence in each); `None` or a missing character skips alignment |
| `location` | `None` | first-line position when no display is used |
| `line_spacing` | `Vector((0,0,-1))` | per-line offset vector when no display is used |
| `name` | `"BDerivation"` | line objects are named `name_line_0`, `name_line_1`, ... |
| `**tex_kwargs` | — | forwarded to every `SimpleTexBObject` line (`color`, `text_size`, `thickness`, ...) |

Attributes: `lines` (all SimpleTexBObjects, in order), `current` (the latest),
`plans` (one `MorphPlan` per step — inspect `pairs`/`vanish`/`appear` when
debugging a matching).

### `write(begin_time=0, transition_time=..., **kwargs) -> float`

Writes the first line (standard write-on animation). Returns
`begin_time + transition_time`, like every animation call.

### `step(expression, **kwargs) -> float`

Advances the derivation to `expression`.

| Parameter | Default | Description |
|---|---|---|
| `mode` | `'new_line'` | `'new_line'`: the step appears on the next line, matched letters fly down as copies. `'replace'`: the current line transforms in place. `'add_subtract'`: right-hand terms of `map` fly across the `=`, flip sign and merge into their left-hand partners (in place) |
| `map` | `None` | manual matching, see below |
| `auto` | `True` | auto-match the letters not covered by `map`; with `False` they vanish/appear |
| `auto_threshold` | `None` | similarity cutoff for the automatic matcher; `None` pairs every leftover it can |
| `align` | `True` | align `align_char` with the previous line |
| `fade_old` | `None` | previous line in `'new_line'` mode: `None` keep, `'dim'` (alpha 0.25), `'hide'` |
| `arc` | `0.3` | flight arc as fraction of travel distance (0 = straight, negative = bend down) |
| `stagger` | `0.5` | fraction of the transition spent fanning out the letter starts (0 = all at once) |
| `lift` | `1.0` | `'add_subtract'` only: how far (in line units) a travelling term rises above the baseline as it crosses the `=`; pass a list (one value per moving term, ordered left-to-right on the source line) to lift them to different heights |
| `highlight` | `None` | substring(s) of the current line flashed before they move |
| `highlight_color` | `'important'` | flash color |
| `new_color` | `None` | color the flown copies fade to mid-flight |
| `cancel` | `None` | pairs of substrings that annihilate, e.g. `[("+5", "-5")]` — copies of both fly to their joint midpoint and shrink |
| `begin_time`, `transition_time` | `0`, default | timing; returns `begin_time + transition_time` |
| `**tex_overrides` | — | per-line overrides of the tex kwargs |

#### The `map` argument

A dict (or iterable of pairs) from source substrings to target substrings:

```python
map={
    "(x+y)^2": "x^2+2xy+y^2",   # group -> group (refined per letter automatically)
    "9@0": None,                 # force vanish (shrinks in place)
    None: "42",                  # force fresh appearance (written in)
}
```

* **Group → group** entries are refined into per-letter pairs by a nested run
  of the fingerprint matcher on just those letters — identical glyphs inside
  the groups (parens, digits, `+`) still map onto their counterparts.
* **Merges**: two entries with the same target substring make both source
  groups converge on it.
* **Splits**: a source letter matched to several targets flies as several
  copies (in `new_line` mode).
* Everything not mentioned is matched automatically (LCS on glyph
  fingerprints, then shape similarity), unless `auto=False`.

#### `mode='new_line'`

Builds the next line (hidden), aligns it, then per matched pair: a **copy** of
the source letter appears, flies along an arc (staggered left-to-right) to the
target letter's slot (slightly behind it, z −0.001), the target letter is
written at full visibility on arrival and the copy disappears. Unmatched target
letters are written in during the second half of the transition. The old line
persists (subject to `fade_old`).

#### `mode='replace'`

Runs a `PairMorph` (see below): copies of the source letters shape-morph into
the target glyphs while the originals hide; on the end frame the pristine
target line takes over (hot swap — the geometry is identical at that moment).
Vanishing letters shrink in place, appearing ones are written in. After the
step, `d.current` is the *new*, clean SimpleTexBObject — steps can be chained
indefinitely and mixed freely with `new_line`.

#### `mode='add_subtract'`

An in-place step that animates *moving a term to the other side of the `=`*.
Each **merge** in `map` (two source terms sharing one target term) is read as a
move: the term on the **right** of the `=` (the *mover*) and the term it joins
on the **left** (the *partner*, which morphs into the combined result) are told
apart automatically by their position relative to the `=`. Every mover then

1. lifts straight up above the equation (by `lift` line units — or its own
   entry when `lift` is a per-term list),
2. travels left and, as it crosses the `=`, turns into a leading `-` (`+9`
   becomes `-9`); a term that already carries a leading operator flies with it
   and cross-fades it into the `-`, otherwise a fresh `-` grows in,
3. arrives above its partner — both flash `highlight_color` —,
4. sinks into the partner and shrinks away while the partner morphs in place
   into the result (`3025` → `3016`), which hot-swaps back to white.

The `=` partitions the line: the emptied right-hand side (leftover operators)
clears out and the target's right-hand side (a fresh `0`, say) is written in;
the left-hand side and the unchanged `x^2`/`=` are matched automatically, so
nothing is ever dragged across the `=` by the auto-matcher. The line keeps its
slot (no new row).

```python
# ... = 9 + 495x  ->  collect both right-hand terms onto the left
t0 = d.step(r"x^2-385\cdot x+3016 = 0", mode='add_subtract',
            map={r"110\cdot x": r"-385\cdot x",   # partner (left)  \  merge
                 r"495\cdot x": r"-385\cdot x",   # mover  (right)  /  -> -385x
                 "3025": "3016",                   # partner (left)  \  merge
                 "9@0": "3016"},                   # mover  (right)  /  -> 3016
            begin_time=t0, transition_time=3)
```

This is the working last step of `intro_algebra_overlay` in
`video_hat_tile/scene_hat_tile.py`. A move whose target has **no** left-hand
partner is supported too: the mover simply flies to where the target term is
written and shrinks in as it appears.

### `highlight(substring, occurrence=None, color='important', emission=3, begin_time=0, transition_time=..., restore=True) -> float`

Flash a substring of the current line: tint + emission pulse over the first
third, hold, restore over the last third (`restore=False` keeps the color).

---

## Supporting modules

### `objects/choreography.py`

Reusable primitives (all keyframe-based, no constraints):

* `stagger_schedule(n, begin_time, transition_time, stagger=0.5, order='left_to_right')`
  → per-letter `(start, duration)`; the last letter always lands on time.
* `arc_apex(start, end, arc=0.3, normal=(0,0,1))` → apex point of an arced
  flight (bends toward +y by default; negative `arc` bends down).
* `fly_letter(bobject, start, end, begin_time, transition_time, arc, normal)` —
  one flight: three location keyframes (start / apex / end), Blender's bezier
  easing makes the arc smooth.
* `fly_letters(flights, ...)` — batch with stagger; `flights` is a list of
  `(bobject, start_location, end_location)` in a common parent frame.
* `highlight_letters(text, indices, color, emission, begin_time, transition_time, restore)`.
* `cancel_letters(groups, begin_time, transition_time, arc=0.2, shrink_fraction=0.4)`
  — groups fly to their joint midpoint (alternating over/under) and scale to
  zero during the last `shrink_fraction`.

### `objects/bmorph_text.py::PairMorph(source, target, plan)`

The single-transition, in-place morph engine behind `mode='replace'`. Build it
with two rendered `SimpleTexBObject`s (same parent/slot) and a
`MorphPlan`, then call `animate(begin_time, transition_time)`. Unlike
`BMorphText` — which needs its whole expression chain at construction because
shape-key point counts are fixed per curve — a PairMorph works on copies, so it
can be created step by step.

### `objects/morph_planning.py::plan_transition(src_sigs, tgt_sigs, mapping=None, auto=True, auto_threshold=None)`

The pure-python matcher shared with `BMorphText`: exact glyph matches via a
longest-common-subsequence on translation-invariant fingerprints, leftovers by
shape similarity, manual pins take precedence. Returns
`MorphPlan(pairs, vanish, appear)`.

---

## Practical notes

* **Timing convention**: every animation method takes
  `begin_time`/`transition_time` in seconds and returns
  `begin_time + transition_time`.
* **Layout**: with `align_char='='`, longer lines grow to the *left* of the
  `=` column; the derivation grows downward by `line_spacing` per step. Check
  the camera framing, or compress with e.g.
  `line_spacing=Vector((0,0,-0.8))`.
* **Anchoring under existing text**: constructing a BDerivation with the same
  expression/kwargs as an already-visible text (e.g. after a `BMorphText`
  morph) and *not* writing it places an invisible, geometrically identical
  first line — flights then visually start from the existing text.
* **Cost**: each substring lookup triggers at most one extra (cached) latex
  compile; repeated scene builds hit the disk cache.

## Troubleshooting

| Error | Cause / fix |
|---|---|
| `AmbiguousTokenError: substring 'y' occurs 2 times ...` | add an occurrence selector: `"y@0"`, `"y@all"` |
| `TokenMappingError: substring ... not found` | the fragment is not *contiguous* in the LaTeX source (e.g. other terms in between) — split it into several map entries |
| `TokenMappingError: tagging ... changed the glyph count / shifted positions` | a color special broke a ligature or kern inside `\text{...}` — choose a substring that does not cut through a ligature (`ff`, `fi`, ...) |
| `TokenMappingError: glyph count mismatch` | the rendered object and its SVG disagree (e.g. letters were deleted); create the mapping before modifying the letters |
| `TokenMappingError: geometric matching ambiguous` | letters were moved individually after rendering; call `find_letters` before rearranging |

## Tests

* `tests/unit/objects/test_token_mapping.py`, `test_choreography.py`
  (pure python; SVG fixtures in `tests/fixtures/svg/`, regenerate with
  `generate_fixtures.py` there)
* `tests/integration/test_token_mapping_bpy.py`, `test_bderivation.py`
  (need bpy + latex; `pytest.mark.bpy`)
