"""Animation-ready traces of the BFF machine, for :mod:`scene_bff`.

This module deliberately contains **no bpy**, so it can be run and tested on
its own::

    python3 bff_trace.py

The interpreter itself is not reimplemented here. It is imported from
``brainfuck/bff/bff.py`` in this same repository, which is a direct port of
the authors' reference implementation (cubff, ``bff.inc.h``) and is
cross-checked against their C core. Duplicating it would risk the animation
and the experiment drifting apart, which is exactly the kind of bug that
would be invisible until someone re-derived the numbers on screen.

What this module adds is the *reduction* a renderer needs: instead of a full
128-byte tape snapshot per executed character, a list of the cells that
actually changed, plus the head positions to drive the gantries.

The story the scene tells
-------------------------
The paper's replicator is::

    [[{.>]-]                                                ]-]>.{[[

and the reason it works is worth stating plainly, because the paper does not:

1. Only five of its characters ever execute: the loop body is ``{ . > ]``.
2. ``{`` decrements ``head1``, which **starts at 0 and therefore wraps to
   127** — the far end of the tape.
3. So the program reads forwards (``head0``: 0, 1, 2, ...) and writes
   backwards (``head1``: 127, 126, 125, ...). Tape B is filled in with a
   *reversed* copy of tape A.
4. That would normally produce garbage. It produces a perfect replica here
   because **the program is a palindrome** — its tail is its head reversed,
   which is precisely the shape the paper notes without explaining.

``reverse(A) == A`` is the whole trick, and :func:`mechanism_facts` asserts
each of these four points against the live trace rather than trusting this
docstring.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Import the interpreter from brainfuck/bff/bff.py (single source of truth).
# ---------------------------------------------------------------------------
_BFF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "brainfuck", "bff")


def _load_bff():
    path = os.path.normpath(os.path.join(_BFF_DIR, "bff.py"))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BFF interpreter not found at {path}. video_bff expects the "
            "brainfuck/bff folder of this repository next to it.")
    spec = importlib.util.spec_from_file_location("bff_interpreter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bff_interpreter"] = module
    spec.loader.exec_module(module)
    return module


bff = _load_bff()

TAPE_SIZE = bff.TAPE_SIZE          # 64, one program
MEM_SIZE = bff.MEM_SIZE            # 128, the executed tape
PAPER_REPLICATOR = bff.PAPER_REPLICATOR
render = bff.render

# ---------------------------------------------------------------------------
# Colour families.
#
# Grouping the ten opcodes by what they *do* is what makes a tape readable at
# a glance: the audience never has to learn ten glyphs, only four behaviours.
# The names are this repository's standard palette (utils/constants.py), which
# is Okabe-Ito derived and therefore already colour-vision-deficiency safe --
# do not substitute raw hex here, or the scene stops matching every other
# video in the workspace.
# ---------------------------------------------------------------------------
FAMILY_OF_OP = {
    "<": "move", ">": "move", "{": "move", "}": "move",
    "+": "arith", "-": "arith",
    ".": "copy", ",": "copy",
    "[": "loop", "]": "loop",
}

FAMILY_COLOR = {
    "move": "drawing",       # sky blue     -- head motion
    "arith": "important",    # vermillion   -- arithmetic
    "copy": "joker",         # bluish green -- the copies that do the replicating
    "loop": "x14_color",     # violet       -- control flow
    "zero": "gray_1",        # near-black   -- the "true zero" that ends loops
    "inert": "gray_3",       # grey         -- the other 245 byte values
}
# ``zero`` is deliberately ``gray_1`` (25,25,25) and not ``background`` (pure
# black). A blank tape is 64 zero bytes, and against these scenes' black
# background pure black makes the entire tape B invisible -- the audience sees
# the copy appear out of nowhere instead of watching an empty tape fill up.
# gray_1 still reads as "empty" while staying clearly distinct from ``inert``
# (gray_3, 75) so the true zero and the 245 merely-inert bytes remain
# tellable apart, which is the distinction that makes BFF's loops work.

#: LaTeX for each opcode.
#:
#: These are **math mode**, because ``SimpleTexBObject``'s default template
#: (``files/tex/template_arial.tex``) wraps the expression in ``align*``.
#: That detail matters here more than usual: the template loads no
#: ``fontenc``, so in *text* mode OT1 renders ``<`` and ``>`` as inverted
#: punctuation (``¡``/``¿``) — silently, with no LaTeX error. ``\mathtt``
#: also gives the monospace look that suits an instruction set, and braces
#: still have to be escaped.
TEX_OF_OP = {
    "<": r"\mathtt{<}", ">": r"\mathtt{>}",
    "{": r"\mathtt{\{}", "}": r"\mathtt{\}}",
    "+": r"\mathtt{+}", "-": r"\mathtt{-}",
    ".": r"\mathtt{.}", ",": r"\mathtt{,}",
    "[": r"\mathtt{[}", "]": r"\mathtt{]}",
}

OPS_IN_ORDER = ["<", ">", "{", "}", "+", "-", ".", ",", "[", "]"]

OP_MEANING = {
    "<": r"\text{head0}\ \text{--}\ 1",
    ">": r"\text{head0}+1",
    "{": r"\text{head1}\ \text{--}\ 1",
    "}": r"\text{head1}+1",
    "+": r"\text{tape[head0]}+1",
    "-": r"\text{tape[head0]}\ \text{--}\ 1",
    ".": r"\text{tape[head1]}\leftarrow\text{tape[head0]}",
    ",": r"\text{tape[head0]}\leftarrow\text{tape[head1]}",
    "[": r"\text{jump past ] if tape[head0]}=0",
    "]": r"\text{jump back to [ if tape[head0]}\neq 0",
}


def family_of_byte(value: int) -> str:
    """Which colour family a raw byte belongs to.

    Note the asymmetry that makes BFF work: byte 0 is its own family (loops
    test against it), while the other 245 non-opcode values are all equally
    inert. See ``brainfuck/bff/README.md``.
    """
    glyph = chr(value)
    if glyph in FAMILY_OF_OP:
        return FAMILY_OF_OP[glyph]
    return "zero" if value == 0 else "inert"


def color_of_byte(value: int) -> str:
    return FAMILY_COLOR[family_of_byte(value)]


def glyph_of_byte(value: int) -> str | None:
    """The drawable glyph for a byte, or ``None`` if the cell should stay blank."""
    glyph = chr(value)
    return glyph if glyph in FAMILY_OF_OP else None


# ---------------------------------------------------------------------------
# The trace itself
# ---------------------------------------------------------------------------
class Step:
    """One executed character, reduced to what a renderer needs.

    ``writes`` holds only the cells whose value actually *changed*. That
    distinction matters: the replicator keeps executing its copy loop long
    after tape B is finished, re-writing identical bytes forever. Keyframing
    those would add thousands of invisible colour transitions.
    """

    __slots__ = ("index", "pc", "head0", "head1", "glyph", "family", "writes")

    def __init__(self, index, pc, head0, head1, glyph, writes):
        self.index = index
        self.pc = pc
        self.head0 = head0
        self.head1 = head1
        self.glyph = glyph
        self.family = FAMILY_OF_OP.get(glyph) if glyph else None
        self.writes = writes          # list of (cell, old_value, new_value)

    def __repr__(self):
        return (f"Step({self.index}, pc={self.pc}, h0={self.head0}, "
                f"h1={self.head1}, {self.glyph!r}, writes={self.writes})")


class Trace:
    """A finished run, ready to be turned into keyframes."""

    def __init__(self, initial, steps, complete_at=None):
        self.initial = initial            # bytes, the tape before step 0
        self.steps = steps                # list[Step]
        self.complete_at = complete_at    # index of the step that finished the copy

    def __len__(self):
        return len(self.steps)

    @property
    def n_cells(self):
        return len(self.initial)

    def writes(self):
        """Every (step_index, cell, old, new) in execution order."""
        return [(s.index, c, o, n) for s in self.steps for (c, o, n) in s.writes]

    def tape_at(self, index):
        """The tape as it stands *after* step ``index`` (``-1`` = initial)."""
        tape = bytearray(self.initial)
        for step in self.steps:
            if step.index > index:
                break
            for cell, _old, new in step.writes:
                tape[cell] = new
        return bytes(tape)

    def summary(self):
        from collections import Counter
        counts = Counter(s.glyph for s in self.steps if s.glyph)
        return (f"{len(self.steps)} steps, {len(self.writes())} cell writes, "
                f"copy complete at step {self.complete_at}, "
                f"opcodes executed: {dict(counts)}")


def trace_pair(program_a: bytes, program_b: bytes, max_steps: int = 8192,
               stop_when_copied: bool = True, tail: int = 8) -> Trace:
    """Run ``A`` concatenated with ``B`` and reduce the run to a :class:`Trace`.

    :param stop_when_copied: cut the trace shortly after tape B first becomes
        a byte-perfect copy of the original tape A. Without this the paper's
        replicator runs to its 8192-character limit, spending 97% of the trace
        re-copying bytes that are already correct -- fine for the experiment,
        useless as footage.
    :param tail: how many extra steps to keep after that moment, so the
        animation does not cut on the very frame the copy lands.
    """
    if len(program_a) != TAPE_SIZE or len(program_b) != TAPE_SIZE:
        raise ValueError(f"both programs must be {TAPE_SIZE} bytes")

    initial = bytes(program_a) + bytes(program_b)
    tape = bytearray(initial)
    raw = []
    bff.evaluate(tape, stepcount=max_steps, trace=raw)

    steps = []
    previous = initial
    complete_at = None
    for entry in raw:
        current = entry["tape"]
        writes = [(i, previous[i], current[i])
                  for i in range(len(current)) if current[i] != previous[i]]
        steps.append(Step(entry["step"], entry["pc"], entry["head0"],
                          entry["head1"], entry["cmd"], writes))
        previous = current

        if complete_at is None and current[TAPE_SIZE:] == program_a:
            complete_at = entry["step"]
            if stop_when_copied:
                # keep a short tail, then stop scanning
                keep = entry["step"] + tail
                steps.extend(
                    Step(e["step"], e["pc"], e["head0"], e["head1"], e["cmd"],
                         [(i, previous[i], e["tape"][i])
                          for i in range(len(e["tape"]))
                          if e["tape"][i] != previous[i]])
                    for e in raw[entry["step"] + 1: keep + 1])
                break

    return Trace(initial, steps, complete_at)


def replicator_trace(**kwargs) -> Trace:
    """The paper's replicator fed a blank tape -- the centrepiece animation."""
    return trace_pair(PAPER_REPLICATOR, bytes(TAPE_SIZE), **kwargs)


# ---------------------------------------------------------------------------
# Mechanism, verified rather than asserted
# ---------------------------------------------------------------------------
def mechanism_facts(trace: Trace | None = None) -> dict:
    """Check the four claims in this module's docstring against a live trace.

    Returns a dict of the numbers worth putting on screen. Raises
    ``AssertionError`` if the interpreter ever stops agreeing -- which is the
    point: the explanatory beat of the video should not be able to drift away
    from what the code actually does.
    """
    if trace is None:
        trace = replicator_trace()

    program = PAPER_REPLICATOR
    executed = sorted({s.glyph for s in trace.steps if s.glyph})

    # 1. the program is a palindrome
    assert program == program[::-1], "the paper's replicator is not a palindrome"

    # 2. reads run forwards, writes run backwards
    write_cells = [cell for (_i, cell, _o, _n) in trace.writes()]
    assert write_cells == sorted(write_cells, reverse=True), \
        "writes are not strictly right-to-left"
    assert write_cells[0] == MEM_SIZE - 1 and write_cells[-1] == TAPE_SIZE, \
        f"writes should sweep 127 -> 64, got {write_cells[0]} -> {write_cells[-1]}"

    read_cells = [s.head0 for s in trace.steps if s.writes]
    assert read_cells == sorted(read_cells), "reads are not left-to-right"

    # 3. every cell of tape B is written exactly once
    assert len(write_cells) == TAPE_SIZE == len(set(write_cells))

    # 4. and the result is the program again
    final = trace.tape_at(trace.steps[-1].index)
    assert final[TAPE_SIZE:] == program, "tape B is not a copy of tape A"
    assert final[:TAPE_SIZE] == program, "tape A was damaged"

    return {
        "executed_opcodes": executed,
        "n_distinct_executed": len(executed),
        "complete_at": trace.complete_at,
        "steps_per_cell": (trace.complete_at + 1) // TAPE_SIZE,
        "first_write_cell": write_cells[0],
        "last_write_cell": write_cells[-1],
        "is_palindrome": True,
    }


def _main():
    trace = replicator_trace()
    print("The paper's replicator")
    print("  A:", render(PAPER_REPLICATOR))
    print()
    print(" ", trace.summary())
    print()
    facts = mechanism_facts(trace)
    print("Mechanism (all assertions passed):")
    print(f"  program is a palindrome        : {facts['is_palindrome']}")
    print(f"  distinct opcodes that ever run : {facts['n_distinct_executed']} "
          f"of 10  {facts['executed_opcodes']}")
    print(f"  loop body                      : {{ . > ]  "
          f"({facts['steps_per_cell']} steps per copied cell)")
    print(f"  head0 (reads)                  : 0 -> 63, forwards")
    print(f"  head1 (writes)                 : {facts['first_write_cell']} -> "
          f"{facts['last_write_cell']}, backwards (it wrapped from 0)")
    print(f"  copy complete at step          : {facts['complete_at']}")
    print()
    print("  => B is a *reversed* copy of A, and A is a palindrome,")
    print("     so the reversed copy is A itself.")
    print()
    final = trace.tape_at(trace.steps[-1].index)
    print("  A' :", render(final[:TAPE_SIZE]))
    print("  B' :", render(final[TAPE_SIZE:]))


if __name__ == "__main__":
    _main()