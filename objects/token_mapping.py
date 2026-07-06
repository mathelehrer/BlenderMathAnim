"""
Substring -> glyph-index mapping for rendered LaTeX expressions.

Lets animation code address the letters of a :class:`SimpleTexBObject` by
LaTeX substring instead of hand-counted glyph indices::

    indices = tex.find_letters("2xy")          # unique occurrence
    indices = tex.find_letters("y", occurrence=1)
    indices = tex.find_letters("y@1")          # same, shorthand

How it works (the "color special" trick, empirically verified with the
project's template + dvisvgm 3.2.1):

1. The target substring(s) are located token-wise in the LaTeX source and
   wrapped in raw dvi color specials ``\\special{color push rgb k/255 0 0}``
   ... ``\\special{color pop}``.  Specials are metric-free whatsits, so the
   tagged expression renders with **bit-identical** glyph positions.
2. The tagged expression is compiled through the ordinary cached
   latex->dvi->svg pipeline (:mod:`utils.tex_compile`).
3. The SVG is parsed directly (pure python): every ``<use>`` glyph instance
   carries an inherited ``fill`` -- instances filled ``#0k0000`` belong to
   target ``k-1``.
4. Glyph instances are matched to the live letters of the tex object by
   geometry (normalized bbox centers, nearest-neighbour bijection), so the
   Blender import order never has to be re-derived.

Everything in this module is bpy-free; the thin glue lives in
``SimpleTexBObject.find_letters`` (objects/tex_bobject.py).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Token",
    "TokenSpan",
    "GlyphInstance",
    "TokenMappingError",
    "AmbiguousTokenError",
    "tokenize_latex",
    "find_occurrences",
    "parse_target",
    "tag_expression",
    "parse_svg_instances",
    "match_instances_to_letters",
    "letters_for_substrings",
]


class TokenMappingError(Exception):
    """Raised when a substring cannot be mapped to glyphs reliably."""


class AmbiguousTokenError(TokenMappingError):
    """Raised when a substring occurs more than once and no occurrence was
    selected."""


# ---------------------------------------------------------------------------
# LaTeX tokenizing and occurrence search
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str   # 'command' | 'char' | 'open' | 'close' | 'script'
    text: str
    start: int  # char span in the original string
    end: int


def tokenize_latex(expression):
    """Tokenize a LaTeX string, skipping whitespace.

    Whitespace produces no tokens (TeX math ignores it and it never yields a
    glyph), which makes substring matching whitespace-insensitive while every
    token keeps its exact character span in the original string.
    """
    tokens = []
    i, n = 0, len(expression)
    while i < n:
        c = expression[i]
        if c.isspace():
            i += 1
            continue
        if c == '\\':
            j = i + 1
            while j < n and expression[j].isalpha():
                j += 1
            if j == i + 1:  # control symbol like \, or \{
                j = min(i + 2, n)
            tokens.append(Token('command', expression[i:j], i, j))
            i = j
        elif c == '{':
            tokens.append(Token('open', c, i, i + 1))
            i += 1
        elif c == '}':
            tokens.append(Token('close', c, i, i + 1))
            i += 1
        elif c in '^_':
            tokens.append(Token('script', c, i, i + 1))
            i += 1
        else:
            tokens.append(Token('char', c, i, i + 1))
            i += 1
    return tokens


@dataclass
class TokenSpan:
    first: int       # token index (inclusive)
    last: int        # token index (inclusive)
    char_start: int  # char span in the original string
    char_end: int


def _span_is_valid(tokens, first, last):
    """A span must be brace-balanced and may not end with a script token
    (its argument would be outside the span)."""
    depth = 0
    for t in tokens[first:last + 1]:
        if t.kind == 'open':
            depth += 1
        elif t.kind == 'close':
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False
    if tokens[last].kind == 'script':
        return False
    return True


def find_occurrences(expression, substring):
    """All token-level occurrences of ``substring`` in ``expression``.

    Matching is done on the token sequences (kind + text), so it is
    whitespace-insensitive.  Brace-unbalanced matches are skipped -- they can
    never correspond to a visual unit.

    :return: list of :class:`TokenSpan`
    """
    tokens = tokenize_latex(expression)
    needle = tokenize_latex(substring)
    if not needle:
        raise TokenMappingError("empty substring")

    spans = []
    for i in range(len(tokens) - len(needle) + 1):
        if all(tokens[i + k].kind == needle[k].kind and tokens[i + k].text == needle[k].text
               for k in range(len(needle))):
            j = i + len(needle) - 1
            if _span_is_valid(tokens, i, j):
                spans.append(TokenSpan(i, j, tokens[i].start, tokens[j].end))
    return spans


def parse_target(target):
    """Split the ``"substring@occurrence"`` shorthand.

    :param target: ``"2xy"``, ``("2xy", 1)``, ``"2xy@1"`` or ``"y@all"``
    :return: (substring, occurrence) with occurrence int, 'all' or None
    """
    if isinstance(target, (tuple, list)):
        substring, occurrence = target
        return str(substring), occurrence
    target = str(target)
    match = re.search(r'@(\d+|all)$', target)
    if match:
        occ = match.group(1)
        return target[:match.start()], ('all' if occ == 'all' else int(occ))
    return target, None


def _select_spans(expression, substring, occurrence):
    spans = find_occurrences(expression, substring)
    if not spans:
        raise TokenMappingError("substring %r not found in %r" % (substring, expression))
    if occurrence is None:
        if len(spans) > 1:
            positions = [expression[s.char_start:s.char_end] for s in spans]
            raise AmbiguousTokenError(
                "substring %r occurs %d times in %r (char offsets %s); "
                "select one with '%s@n' (n=0..%d) or '%s@all'"
                % (substring, len(spans), expression,
                   [s.char_start for s in spans], substring, len(spans) - 1, substring))
        return spans
    if occurrence == 'all':
        return spans
    if not 0 <= occurrence < len(spans):
        raise TokenMappingError("occurrence %d out of range, %r occurs %d times"
                                % (occurrence, substring, len(spans)))
    return [spans[occurrence]]


# ---------------------------------------------------------------------------
# color tagging
# ---------------------------------------------------------------------------

def _color_specials(index):
    """dvi specials and resulting hex fill for target ``index`` (0-based)."""
    k = index + 1
    if k > 255:
        raise TokenMappingError("too many targets (max 255)")
    push = "\\special{color push rgb %.8f 0 0}" % (k / 255.0)
    pop = "\\special{color pop}"
    return push, pop, "#%02x0000" % k


def tag_expression(expression, targets):
    """Wrap each target substring in color specials.

    :param targets: list of ``(substring, occurrence)`` with occurrence an
        int, ``'all'`` or ``None`` (None raises on ambiguity)
    :return: (tagged expression, list of hex colors -- one per target)
    """
    tokens = tokenize_latex(expression)
    insertions = []  # (char_start, char_end, prefix, suffix)
    colors = []
    used = []  # (char_start, char_end) spans to detect overlap

    for index, (substring, occurrence) in enumerate(targets):
        push, pop, color = _color_specials(index)
        colors.append(color)
        for span in _select_spans(expression, substring, occurrence):
            for (a, b) in used:
                if span.char_start < b and a < span.char_end:
                    raise TokenMappingError(
                        "target %r overlaps another target in %r" % (substring, expression))
            used.append((span.char_start, span.char_end))

            char_start, char_end = span.char_start, span.char_end
            prefix, suffix = push, pop
            # bracing rule: if the span is the (unbraced) argument of ^ or _,
            # wrap it in braces -- script arguments are groups already, so
            # this never changes spacing:  x^2  ->  x^{<push>2<pop>}
            if span.first > 0 and tokens[span.first - 1].kind == 'script' \
                    and tokens[span.first].kind != 'open':
                prefix = '{' + push
                suffix = pop + '}'
            insertions.append((char_start, char_end, prefix, suffix))

    # apply right-to-left so earlier offsets stay valid
    tagged = expression
    for char_start, char_end, prefix, suffix in sorted(insertions, key=lambda x: -x[0]):
        tagged = tagged[:char_start] + prefix + tagged[char_start:char_end] + suffix + tagged[char_end:]
    return tagged, colors


# ---------------------------------------------------------------------------
# SVG parsing (dvisvgm output)
# ---------------------------------------------------------------------------

@dataclass
class GlyphInstance:
    order: int      # document order
    href: str       # glyph definition id
    x: float
    y: float
    bbox: tuple     # (xmin, ymin, xmax, ymax) in SVG coordinates
    fill: str       # inherited fill, '#000000' if unset


_NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def _parse_path_numbers(chunk):
    return [float(m) for m in _NUMBER.findall(chunk)]


def _cubic_axis_extrema(p0, p1, p2, p3):
    """Interior extremum values of one cubic bezier component."""
    # derivative: 3[(p1-p0) + 2t(p2-2p1+p0) + t^2(p3-3p2+3p1-p0)]
    a = p3 - 3 * p2 + 3 * p1 - p0
    b = 2 * (p2 - 2 * p1 + p0)
    c = p1 - p0
    values = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            t = -c / b
            if 0 < t < 1:
                values.append(t)
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            root = disc ** 0.5
            for t in ((-b + root) / (2 * a), (-b - root) / (2 * a)):
                if 0 < t < 1:
                    values.append(t)
    out = []
    for t in values:
        s = 1 - t
        out.append(s ** 3 * p0 + 3 * s ** 2 * t * p1 + 3 * s * t ** 2 * p2 + t ** 3 * p3)
    return out


def _path_bbox(d):
    """Exact bounding box of an SVG path (M/L/H/V/C/S/Q/T/Z, absolute or
    relative), including bezier extrema.  dvisvgm emits M/L/H/V/C/S/Z."""
    xs, ys = [], []
    cur = np.zeros(2)
    start = np.zeros(2)
    prev_cubic_ctrl = None
    prev_quad_ctrl = None

    def add_point(p):
        xs.append(p[0])
        ys.append(p[1])

    def add_cubic(p0, c1, c2, p1):
        add_point(p0)
        add_point(p1)
        for axis in (0, 1):
            for v in _cubic_axis_extrema(p0[axis], c1[axis], c2[axis], p1[axis]):
                (xs if axis == 0 else ys).append(v)

    for match in re.finditer(r'([MLHVCSQTZmlhvcsqtz])([^MLHVCSQTZmlhvcsqtz]*)', d):
        cmd, chunk = match.group(1), match.group(2)
        nums = _parse_path_numbers(chunk)
        rel = cmd.islower()
        op = cmd.upper()
        i = 0

        def take(k):
            nonlocal i
            vals = nums[i:i + k]
            i += k
            return vals

        while True:
            if op == 'Z':
                cur = start.copy()
                prev_cubic_ctrl = prev_quad_ctrl = None
                break
            if i >= len(nums):
                break
            if op == 'M':
                p = np.array(take(2))
                cur = cur + p if rel else p
                start = cur.copy()
                add_point(cur)
                op = 'L'  # implicit lineto for further pairs
                prev_cubic_ctrl = prev_quad_ctrl = None
            elif op == 'L':
                p = np.array(take(2))
                cur = cur + p if rel else p
                add_point(cur)
                prev_cubic_ctrl = prev_quad_ctrl = None
            elif op == 'H':
                (v,) = take(1)
                cur = np.array([cur[0] + v if rel else v, cur[1]])
                add_point(cur)
                prev_cubic_ctrl = prev_quad_ctrl = None
            elif op == 'V':
                (v,) = take(1)
                cur = np.array([cur[0], cur[1] + v if rel else v])
                add_point(cur)
                prev_cubic_ctrl = prev_quad_ctrl = None
            elif op in ('C', 'S'):
                if op == 'C':
                    c1 = np.array(take(2))
                    if rel:
                        c1 = cur + c1
                else:  # S: first control = reflection of previous
                    c1 = 2 * cur - prev_cubic_ctrl if prev_cubic_ctrl is not None else cur.copy()
                c2 = np.array(take(2))
                p = np.array(take(2))
                if rel:
                    c2 = cur + c2
                    p = cur + p
                add_cubic(cur, c1, c2, p)
                prev_cubic_ctrl = c2
                prev_quad_ctrl = None
                cur = p
            elif op in ('Q', 'T'):
                if op == 'Q':
                    q = np.array(take(2))
                    if rel:
                        q = cur + q
                else:
                    q = 2 * cur - prev_quad_ctrl if prev_quad_ctrl is not None else cur.copy()
                p = np.array(take(2))
                if rel:
                    p = cur + p
                # elevate the quadratic to a cubic for the extrema math
                c1 = cur + 2.0 / 3.0 * (q - cur)
                c2 = p + 2.0 / 3.0 * (q - p)
                add_cubic(cur, c1, c2, p)
                prev_quad_ctrl = q
                prev_cubic_ctrl = None
                cur = p
            else:
                raise TokenMappingError("unsupported svg path command %r" % cmd)

    if not xs:
        raise TokenMappingError("empty svg path")
    return min(xs), min(ys), max(xs), max(ys)


def _strip_ns(tag):
    return tag.split('}')[-1]


def parse_svg_instances(svg_path):
    """Parse a dvisvgm SVG into glyph instances in document order.

    :return: list of :class:`GlyphInstance` with exact bboxes and the
        inherited fill color of each instance
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    glyph_bboxes = {}
    instances = []

    def walk(element, fill):
        tag = _strip_ns(element.tag)
        if 'transform' in element.attrib:
            raise TokenMappingError(
                "unexpected transform in %s -- unsupported dvisvgm output layout" % svg_path)
        fill = element.get('fill', fill)
        if tag == 'defs':
            for child in element:
                if _strip_ns(child.tag) == 'path' and 'id' in child.attrib:
                    glyph_bboxes[child.get('id')] = _path_bbox(child.get('d'))
            return
        if tag == 'use':
            href = (element.get('{http://www.w3.org/1999/xlink}href')
                    or element.get('href') or '').lstrip('#')
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            if href not in glyph_bboxes:
                raise TokenMappingError("use references unknown glyph %r" % href)
            b = glyph_bboxes[href]
            instances.append(GlyphInstance(
                order=len(instances), href=href, x=x, y=y,
                bbox=(b[0] + x, b[1] + y, b[2] + x, b[3] + y),
                fill=fill))
            return
        # rects appear for \rule / fraction bars; treat them as glyphs too
        if tag == 'rect':
            x, y = float(element.get('x', 0)), float(element.get('y', 0))
            w, h = float(element.get('width', 0)), float(element.get('height', 0))
            instances.append(GlyphInstance(
                order=len(instances), href='rect', x=x, y=y,
                bbox=(x, y, x + w, y + h), fill=fill))
            return
        for child in element:
            walk(child, fill)

    walk(root, '#000000')
    return instances


def bezier_glyph_center(splines, cyclic_flags=None):
    """Exact 2d bbox center of a glyph given as bezier control arrays.

    :param splines: list of arrays (n,3,3) of (co, handle_left, handle_right)
        -- the representation used by objects/morph_planning.py
    :param cyclic_flags: per-spline closed flag (defaults to closed)
    :return: (cx, cy) of the exact curve bounding box
    """
    if cyclic_flags is None:
        cyclic_flags = [True] * len(splines)
    xs, ys = [], []
    for spline, cyclic in zip(splines, cyclic_flags):
        pts = np.asarray(spline)
        n = len(pts)
        segments = [(i, (i + 1) % n) for i in range(n)] if cyclic else \
            [(i, i + 1) for i in range(n - 1)]
        for i, j in segments:
            p0, c1 = pts[i][0], pts[i][2]
            c2, p1 = pts[j][1], pts[j][0]
            for axis, acc in ((0, xs), (1, ys)):
                acc.append(p0[axis])
                acc.append(p1[axis])
                acc.extend(_cubic_axis_extrema(p0[axis], c1[axis], c2[axis], p1[axis]))
    if not xs:
        raise TokenMappingError("glyph without bezier points")
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def instance_centers(instances):
    return np.array([[(g.bbox[0] + g.bbox[2]) / 2, (g.bbox[1] + g.bbox[3]) / 2]
                     for g in instances])


def drop_reference_h(instances):
    """Remove the alignment 'H' that generate_tex_file prepends.

    It must be both the first instance in document order and the leftmost
    one; if the two criteria disagree something is off and we refuse to
    guess.
    """
    if not instances:
        raise TokenMappingError("svg contains no glyph instances")
    leftmost = min(range(len(instances)), key=lambda i: instances[i].bbox[0])
    if leftmost != 0:
        raise TokenMappingError(
            "reference 'H' is not the leftmost glyph -- cannot strip it safely")
    return instances[1:]


# ---------------------------------------------------------------------------
# geometric instance -> letter matching
# ---------------------------------------------------------------------------

def _normalize(points, flip_y):
    pts = np.array(points, dtype=float)
    if flip_y:
        pts[:, 1] = -pts[:, 1]
    pts -= np.mean(pts, axis=0)
    extent = np.max(pts[:, 0]) - np.min(pts[:, 0])
    if extent < 1e-9:  # single letter / vertical stack: use overall extent
        extent = max(np.max(np.abs(pts)), 1e-9)
    return pts / extent


def _greedy_bijection(a, b):
    """Nearest-neighbour bijection between two equal-length point sets.

    :return: (mapping list a_index->b_index, cost of the worst match)
    """
    pairs = []
    for i in range(len(a)):
        for j in range(len(b)):
            pairs.append((float(np.linalg.norm(a[i] - b[j])), i, j))
    pairs.sort(key=lambda p: p[0])
    mapping = [None] * len(a)
    used_b = set()
    worst = 0.0
    matched = 0
    for dist, i, j in pairs:
        if mapping[i] is not None or j in used_b:
            continue
        mapping[i] = j
        used_b.add(j)
        worst = max(worst, dist)
        matched += 1
        if matched == len(a):
            break
    return mapping, worst


def match_instances_to_letters(inst_centers, letter_centers):
    """Match SVG glyph instances to Blender letters by normalized geometry.

    Both point sets are centered and scaled by their x-extent; the SVG y-flip
    is absorbed by trying both orientations.

    :param inst_centers: (n,2) bbox centers of the SVG instances (H removed)
    :param letter_centers: (n,2) bbox centers of the letters (container frame)
    :return: list mapping instance index -> letter index
    :raises TokenMappingError: on count mismatch or ambiguous geometry
    """
    inst_centers = np.asarray(inst_centers, dtype=float)
    letter_centers = np.asarray(letter_centers, dtype=float)
    if len(inst_centers) != len(letter_centers):
        raise TokenMappingError(
            "glyph count mismatch: %d svg instances vs %d letters"
            % (len(inst_centers), len(letter_centers)))
    if len(inst_centers) == 1:
        return [0]

    normalized_letters = _normalize(letter_centers, flip_y=False)
    best_mapping, best_worst = None, None
    for flip in (True, False):
        mapping, worst = _greedy_bijection(_normalize(inst_centers, flip_y=flip),
                                           normalized_letters)
        if best_worst is None or worst < best_worst:
            best_mapping, best_worst = mapping, worst

    # tolerance: half the median horizontal gap between letters
    xs = np.sort(normalized_letters[:, 0])
    gaps = np.diff(xs)
    gaps = gaps[gaps > 1e-9]
    pitch = float(np.median(gaps)) if len(gaps) else 0.1
    if best_worst > 0.5 * pitch:
        raise TokenMappingError(
            "geometric matching ambiguous (worst distance %.4f vs pitch %.4f)"
            % (best_worst, pitch))
    return best_mapping


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def letters_for_substrings(expression, targets, svg_path, letter_centers,
                           typeface='default', text_only=False, recreate=False):
    """Map substrings of ``expression`` to letter indices of its rendered object.

    :param expression: the raw LaTeX source of the tex object
    :param targets: list of targets, each ``"substr"``, ``"substr@n"``,
        ``"substr@all"`` or ``(substr, occurrence)``
    :param svg_path: path of the object's own (uncolored) cached SVG
    :param letter_centers: (n,2) bbox centers of the live letters, in the
        text container frame (x right, y up)
    :return: list (one entry per target) of sorted letter-index lists
    """
    # imported lazily so the parsing/matching core stays importable without
    # the constants of a configured runtime environment
    from utils.tex_compile import tex_to_svg_file
    from utils.constants import TEMPLATE_TEX_FILE, TEMPLATE_TEXT_FILE

    parsed = [parse_target(t) for t in targets]
    tagged, colors = tag_expression(expression, parsed)

    template = TEMPLATE_TEXT_FILE if text_only else TEMPLATE_TEX_FILE
    if typeface != 'default':
        template = template[:-10] + '_' + typeface + '.tex'
    tagged_svg = tex_to_svg_file(tagged, template, typeface, text_only, recreate)

    plain = parse_svg_instances(svg_path)
    colored = parse_svg_instances(tagged_svg)

    # validation: colors must not have moved anything (broken ligature/kern)
    if len(plain) != len(colored):
        raise TokenMappingError(
            "tagging %r changed the glyph count (%d -> %d) -- a color special "
            "probably broke a ligature or kern; rephrase the substring"
            % ([t[0] for t in parsed], len(plain), len(colored)))
    plain_pos = np.array([[g.x, g.y] for g in plain])
    colored_pos = np.array([[g.x, g.y] for g in colored])
    if not np.allclose(plain_pos, colored_pos, atol=1e-3):
        worst = np.max(np.abs(plain_pos - colored_pos))
        raise TokenMappingError(
            "tagging %r shifted glyph positions (max %.4f) -- a color special "
            "probably broke a ligature or kern; rephrase the substring"
            % ([t[0] for t in parsed], worst))

    plain = drop_reference_h(plain)
    colored = drop_reference_h(colored)

    mapping = match_instances_to_letters(instance_centers(plain), letter_centers)

    result = []
    for color in colors:
        letters = sorted(mapping[i] for i, g in enumerate(colored) if g.fill == color)
        if not letters:
            raise TokenMappingError(
                "target with color %s produced no glyphs (spacing-only substring?)" % color)
        result.append(letters)
    return result
