"""
Pure-python planning and bezier geometry core for :class:`objects.bmorph_text.BMorphText`.

This module deliberately imports **no bpy** so the whole morph-planning pipeline
(glyph matching, spline pairing, resampling, chain compilation) can be unit
tested without Blender (see ``tests/unit/objects/test_morph_planning.py``).

Geometry representation
-----------------------
A *spline* is a numpy array of shape ``(n, 3, 3)``:
``spline[i] = (co, handle_left, handle_right)``, each a 3-vector, matching a
Blender bezier spline.  A *glyph* is a list of such splines.

Pipeline
--------
1. :func:`glyph_signature` reduces a glyph to a :class:`GlyphSig` (fingerprint
   hash + coarse shape features) used for matching.
2. :func:`plan_transition` decides which source letter morphs into which
   target letter (fully automatic, fully manual, or mixed).
3. :func:`compile_chain` turns the sequence of glyphs a single letter passes
   through into shape-key snapshots with consistent spline/point counts.
   Splines that appear/disappear along the way grow from / collapse to their
   own location (never the text origin).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GlyphSig",
    "MorphPlan",
    "glyph_signature",
    "plan_transition",
    "compile_chain",
    "resample_spline",
    "split_segment",
    "spline_centroid",
    "spline_length",
    "pair_splines",
    "align_to_reference",
]


# ---------------------------------------------------------------------------
# bezier helpers
# ---------------------------------------------------------------------------

def split_segment(p0, p1, t=0.5):
    """De-Casteljau split of one bezier segment, exact in shape.

    :param p0: start point, array (3,3) = (co, handle_left, handle_right)
    :param p1: end point, array (3,3)
    :param t: split parameter
    :return: (new_p0, mid, new_p1) with adjusted handles
    """
    a, b = p0[0], p0[2]  # co, handle_right of start
    c, d = p1[1], p1[0]  # handle_left, co of end

    ab = (1 - t) * a + t * b
    bc = (1 - t) * b + t * c
    cd = (1 - t) * c + t * d
    abbc = (1 - t) * ab + t * bc
    bccd = (1 - t) * bc + t * cd
    mid_co = (1 - t) * abbc + t * bccd

    new_p0 = np.array([p0[0], p0[1], ab])
    mid = np.array([mid_co, abbc, bccd])
    new_p1 = np.array([p1[0], cd, p1[2]])
    return new_p0, mid, new_p1


def _segment_indices(n, cyclic):
    """Index pairs of the bezier segments of a spline with n points."""
    if cyclic:
        return [(i, (i + 1) % n) for i in range(n)]
    return [(i, i + 1) for i in range(n - 1)]


def resample_spline(points, n, cyclic=True):
    """Return a copy of ``points`` with exactly ``n`` control points.

    Growth is exact (repeated de-Casteljau splits of the longest chord
    segment); the shape of the curve is preserved.  ``n`` smaller than the
    current count is not supported (the caller always grows).

    :param points: array (m,3,3)
    :param n: target number of control points, n >= m
    :param cyclic: closed spline
    """
    pts = [np.array(p) for p in points]
    if n < len(pts):
        raise ValueError("resample_spline only grows splines (%d -> %d requested)" % (len(pts), n))
    while len(pts) < n:
        segs = _segment_indices(len(pts), cyclic)
        lengths = [np.linalg.norm(pts[j][0] - pts[i][0]) for i, j in segs]
        i, j = segs[int(np.argmax(lengths))]
        new_i, mid, new_j = split_segment(pts[i], pts[j])
        pts[i] = new_i
        pts[j] = new_j
        pts.insert(i + 1, mid)
    return np.array(pts)


def spline_centroid(points):
    """Average of the control point locations, array (3,)."""
    return np.mean(np.asarray(points)[:, 0, :], axis=0)


def spline_length(points, cyclic=True):
    """Total chord length of the control polygon."""
    pts = np.asarray(points)[:, 0, :]
    n = len(pts)
    total = 0.0
    for i, j in _segment_indices(n, cyclic):
        total += float(np.linalg.norm(pts[j] - pts[i]))
    return total


def degenerate_spline(anchor, n):
    """A spline of ``n`` coinciding control points sitting at ``anchor``.

    Used as the pre-birth / post-death shape of splines so that geometry
    grows out of (collapses into) its own location instead of the origin.
    """
    anchor = np.asarray(anchor, dtype=float)
    return np.broadcast_to(anchor, (n, 3, 3)).copy()


def align_to_reference(reference, points, cyclic=True):
    """Reindex a cyclic spline so it matches ``reference`` point-for-point
    with the least total deviation.

    Tries every cyclic offset and both orientations and returns the rolled
    (possibly reversed) copy of ``points`` that minimizes the summed squared
    distance of the point locations to ``reference``.  This prevents the
    morph from 'twisting' around the spline.
    """
    ref = np.asarray(reference)[:, 0, :]
    pts = np.asarray(points)
    if len(pts) != len(ref) or not cyclic or len(pts) < 2:
        return np.array(pts)

    def candidates():
        yield pts
        # reversed orientation: reverse point order and swap the handles
        rev = pts[::-1].copy()
        rev = rev[:, [0, 2, 1], :]
        yield rev

    best, best_cost = None, None
    for cand in candidates():
        cos = cand[:, 0, :]
        for offset in range(len(cand)):
            cost = float(np.sum((np.roll(cos, -offset, axis=0) - ref) ** 2))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = np.roll(cand, -offset, axis=0)
    return best


# ---------------------------------------------------------------------------
# glyph signatures and matching
# ---------------------------------------------------------------------------

@dataclass
class GlyphSig:
    """Coarse, comparison-friendly description of one glyph."""
    index: int
    fingerprint: str
    center: tuple
    width: float
    height: float
    spline_count: int
    length: float


def glyph_signature(index, splines, location=(0, 0, 0), digits=3):
    """Build a :class:`GlyphSig` from the splines of one letter.

    The fingerprint is a hash of the control data after translating the
    glyph centroid to the origin, so two occurrences of the same character
    produce the same fingerprint regardless of their position in the text.

    :param index: index of the letter within its expression
    :param splines: list of arrays (n,3,3), local letter coordinates
    :param location: location of the letter object in the text frame
    :param digits: rounding used for the fingerprint (same tolerance the
        old framework used in ``are_chars_same``)
    """
    location = np.asarray(location, dtype=float)
    all_cos = np.concatenate([np.asarray(s)[:, 0, :] for s in splines])
    centroid = np.mean(all_cos, axis=0)

    hasher = hashlib.sha256()
    hasher.update(str(len(splines)).encode())
    for s in splines:
        arr = np.round(np.asarray(s) - centroid, digits) + 0.0  # +0.0 kills -0.0
        hasher.update(str(len(s)).encode())
        hasher.update(arr.tobytes())

    mins = np.min(all_cos, axis=0)
    maxs = np.max(all_cos, axis=0)
    total_length = sum(spline_length(s) for s in splines)
    return GlyphSig(
        index=index,
        fingerprint=hasher.hexdigest(),
        center=tuple(centroid + location),
        width=float(maxs[0] - mins[0]),
        height=float(maxs[1] - mins[1]),
        spline_count=len(splines),
        length=total_length,
    )


@dataclass
class MorphPlan:
    """Result of planning one transition.

    ``pairs`` maps source letter indices to target letter indices.  A source
    index occurring twice means the letter splits (a copy is made); a target
    index occurring twice means two letters merge onto the same target.
    ``vanish`` letters shrink away in place, ``appear`` letters grow at their
    own position.
    """
    pairs: list = field(default_factory=list)
    vanish: list = field(default_factory=list)
    appear: list = field(default_factory=list)


def _lcs_pairs(src_sigs, tgt_sigs):
    """Longest common subsequence on glyph fingerprints.

    Maximizes the number of identical glyphs matched in order; among optimal
    solutions prefers the one with the least total horizontal displacement.
    Returns a list of (src_index, tgt_index) pairs.
    """
    n, m = len(src_sigs), len(tgt_sigs)
    NEG = (-1, 0.0)
    # dp[i][j] = (matches, -displacement) achievable for suffixes i:, j:
    dp = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = max(dp[i + 1][j], dp[i][j + 1])
            if src_sigs[i].fingerprint == tgt_sigs[j].fingerprint:
                disp = abs(src_sigs[i].center[0] - tgt_sigs[j].center[0])
                take = (dp[i + 1][j + 1][0] + 1, dp[i + 1][j + 1][1] - disp)
                best = max(best, take)
            dp[i][j] = best

    pairs = []
    i = j = 0
    while i < n and j < m:
        if src_sigs[i].fingerprint == tgt_sigs[j].fingerprint:
            disp = abs(src_sigs[i].center[0] - tgt_sigs[j].center[0])
            take = (dp[i + 1][j + 1][0] + 1, dp[i + 1][j + 1][1] - disp)
            if take == dp[i][j]:
                pairs.append((src_sigs[i].index, tgt_sigs[j].index))
                i += 1
                j += 1
                continue
        if dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _shape_cost(s, t, scale):
    """Dissimilarity of two glyphs plus a mild positional penalty.

    ``scale`` is a typical glyph extent used to normalize the position term.
    """
    eps = 1e-6

    def log_ratio(a, b):
        return abs(np.log((a + eps) / (b + eps)))

    cost = log_ratio(s.width, t.width) + log_ratio(s.height, t.height)
    cost += 0.4 * abs(s.spline_count - t.spline_count)
    cost += log_ratio(s.length, t.length)
    cost += 0.1 * abs(s.center[0] - t.center[0]) / (scale + eps)
    return cost


def _similarity_pairs(src_left, tgt_left, scale, threshold):
    """Greedy one-to-one matching of the leftover glyphs by shape cost."""
    candidates = []
    for s in src_left:
        for t in tgt_left:
            cost = _shape_cost(s, t, scale)
            if threshold is None or cost <= threshold:
                candidates.append((cost, s.index, t.index))
    candidates.sort(key=lambda c: c[0])

    used_src, used_tgt, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_src or j in used_tgt:
            continue
        used_src.add(i)
        used_tgt.add(j)
        pairs.append((i, j))
    return pairs


def _normalize_mapping(mapping):
    """Bring a user mapping into canonical form.

    Accepted forms::

        {3: 3, 15: (16, 17), 5: None}     # dict, value None = force vanish
        [(3, 3), (15, 16), (15, 17), (None, 8), (5, None)]

    ``(None, j)`` forces target ``j`` to appear fresh, ``(i, None)`` forces
    source ``i`` to vanish.

    :return: (pinned pairs, forced vanish set, forced appear set)
    """
    if mapping is None:
        return [], set(), set()

    items = []
    if isinstance(mapping, dict):
        for i, v in mapping.items():
            if v is None:
                items.append((i, None))
            elif isinstance(v, (list, tuple)):
                items.extend((i, j) for j in v)
            else:
                items.append((i, v))
    else:
        items = list(mapping)

    pairs, vanish, appear = [], set(), set()
    for i, j in items:
        if i is None and j is None:
            raise ValueError("mapping entry (None, None) is meaningless")
        if j is None:
            vanish.add(i)
        elif i is None:
            appear.add(j)
        else:
            pairs.append((int(i), int(j)))
    return pairs, vanish, appear


def plan_transition(src_sigs, tgt_sigs, mapping=None, auto=True, auto_threshold=None):
    """Decide which source letter morphs into which target letter.

    :param src_sigs: list of :class:`GlyphSig` of the source expression
    :param tgt_sigs: list of :class:`GlyphSig` of the target expression
    :param mapping: optional manual mapping (see :func:`_normalize_mapping`);
        pinned letters are taken out of the automatic matching
    :param auto: if False, only the manual mapping is used and everything
        else vanishes/appears
    :param auto_threshold: optional cost limit for the similarity pass;
        ``None`` pairs up every leftover glyph it can (maximal morphing)
    :return: :class:`MorphPlan`
    """
    pinned, forced_vanish, forced_appear = _normalize_mapping(mapping)

    for i, j in pinned:
        if not 0 <= i < len(src_sigs):
            raise IndexError("mapping source index %d out of range (0..%d)" % (i, len(src_sigs) - 1))
        if not 0 <= j < len(tgt_sigs):
            raise IndexError("mapping target index %d out of range (0..%d)" % (j, len(tgt_sigs) - 1))

    pinned_src = {i for i, _ in pinned} | forced_vanish
    pinned_tgt = {j for _, j in pinned} | forced_appear

    pairs = list(pinned)
    if auto:
        free_src = [s for s in src_sigs if s.index not in pinned_src]
        free_tgt = [t for t in tgt_sigs if t.index not in pinned_tgt]

        lcs = _lcs_pairs(free_src, free_tgt)
        pairs.extend(lcs)
        matched_src = {i for i, _ in lcs}
        matched_tgt = {j for _, j in lcs}

        src_left = [s for s in free_src if s.index not in matched_src]
        tgt_left = [t for t in free_tgt if t.index not in matched_tgt]
        scale = max([max(s.width, s.height) for s in src_sigs + tgt_sigs] + [1e-6])
        pairs.extend(_similarity_pairs(src_left, tgt_left, scale, auto_threshold))

    covered_src = {i for i, _ in pairs}
    covered_tgt = {j for _, j in pairs}
    vanish = [s.index for s in src_sigs if s.index not in covered_src]
    appear = [t.index for t in tgt_sigs if t.index not in covered_tgt]

    pairs.sort()
    return MorphPlan(pairs=pairs, vanish=sorted(vanish), appear=sorted(appear))


# ---------------------------------------------------------------------------
# chain compilation: sequence of glyphs -> shape key snapshots
# ---------------------------------------------------------------------------

@dataclass
class _Slot:
    """One physical spline of the final curve and its life story."""
    born: int                       # snapshot index of first real shape
    shapes: dict                    # snapshot index -> array (m,3,3) (real shapes only)
    cyclic: bool = True
    died: int = None                # first snapshot index without a real shape


def compile_chain(glyphs, cyclic_flags=None):
    """Compile the glyph sequence of one letter into shape-key snapshots.

    All snapshots share the same spline layout (count, point counts, cyclic
    flags).  Splines that only exist in later glyphs are collapsed to a
    degenerate point *at their own birth location* in earlier snapshots, and
    splines that die collapse to their last location -- this is what makes
    new geometry grow out of the letter instead of the text origin.

    :param glyphs: list of glyphs; glyph = list of splines, array (n,3,3),
        each glyph given in the local frame of its letter
    :param cyclic_flags: optional list (per glyph) of lists of bools
    :return: (snapshots, cyclic) where snapshots is a list (one per glyph)
        of lists of arrays (n_slot, 3, 3) and cyclic the per-slot flags
    """
    if len(glyphs) == 0:
        raise ValueError("compile_chain needs at least one glyph")
    if cyclic_flags is None:
        cyclic_flags = [[True] * len(g) for g in glyphs]

    # --- build slot life stories -----------------------------------------
    slots = []
    alive = []  # slot indices alive after the current step, parallel to glyph splines
    for si, spline in enumerate(glyphs[0]):
        slots.append(_Slot(born=0, shapes={0: np.asarray(spline)}, cyclic=cyclic_flags[0][si]))
        alive.append(si)

    for k in range(1, len(glyphs)):
        target = [np.asarray(s) for s in glyphs[k]]
        current_shapes = [slots[s].shapes[k - 1] for s in alive]
        pairs, extra_cur, extra_tgt = pair_splines(current_shapes, target)

        new_alive = [None] * len(target)
        for ci, ti in pairs:
            slot = slots[alive[ci]]
            slot.shapes[k] = target[ti]
            new_alive[ti] = alive[ci]
        for ci in extra_cur:
            slots[alive[ci]].died = k
        for ti in extra_tgt:
            slots.append(_Slot(born=k, shapes={k: target[ti]}, cyclic=cyclic_flags[k][ti]))
            new_alive[ti] = len(slots) - 1
        alive = new_alive

    # --- equalize point counts per slot -----------------------------------
    for slot in slots:
        n = max(len(shape) for shape in slot.shapes.values())
        for k in sorted(slot.shapes):
            resampled = resample_spline(slot.shapes[k], n, cyclic=slot.cyclic)
            if k > slot.born and k - 1 in slot.shapes:
                resampled = align_to_reference(slot.shapes[k - 1], resampled, cyclic=slot.cyclic)
            slot.shapes[k] = resampled

    # --- assemble snapshots ------------------------------------------------
    snapshots = []
    for k in range(len(glyphs)):
        snapshot = []
        for slot in slots:
            n = len(next(iter(slot.shapes.values())))
            if k in slot.shapes:
                snapshot.append(slot.shapes[k])
            elif k < slot.born:
                anchor = spline_centroid(slot.shapes[slot.born])
                snapshot.append(degenerate_spline(anchor, n))
            else:  # k >= slot.died or after last shape
                last = max(kk for kk in slot.shapes if kk < k)
                anchor = spline_centroid(slot.shapes[last])
                snapshot.append(degenerate_spline(anchor, n))
        snapshots.append(snapshot)

    cyclic = [slot.cyclic for slot in slots]
    return snapshots, cyclic


def pair_splines(current, target):
    """Greedy pairing of the splines of two glyphs.

    Cost combines centroid distance and relative length difference -- this
    fixes the old framework's length-rank-only pairing (its own TODO).

    :param current: list of arrays (n,3,3)
    :param target: list of arrays (m,3,3)
    :return: (pairs, extra_current, extra_target) index lists
    """
    eps = 1e-6
    candidates = []
    cur_info = [(spline_centroid(s), spline_length(s)) for s in current]
    tgt_info = [(spline_centroid(s), spline_length(s)) for s in target]
    scale = max([length for _, length in cur_info + tgt_info] + [eps])

    for i, (c_cen, c_len) in enumerate(cur_info):
        for j, (t_cen, t_len) in enumerate(tgt_info):
            cost = float(np.linalg.norm(c_cen - t_cen)) / scale
            cost += 0.5 * abs(np.log((c_len + eps) / (t_len + eps)))
            candidates.append((cost, i, j))
    candidates.sort(key=lambda c: c[0])

    used_c, used_t, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_c or j in used_t:
            continue
        used_c.add(i)
        used_t.add(j)
        pairs.append((i, j))

    extra_c = [i for i in range(len(current)) if i not in used_c]
    extra_t = [j for j in range(len(target)) if j not in used_t]
    return pairs, extra_c, extra_t
